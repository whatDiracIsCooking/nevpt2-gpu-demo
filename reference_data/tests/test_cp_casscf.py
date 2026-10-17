"""Hand cases for ``nevpt2_grad.py --reference casscf``'s coupled CP-CASSCF
solve (docs/gradient-theory.md, sections 1.4 and 2.7): the operator it builds
densely is the true Jacobian of ``gen_g_hop``'s own gradient, the basis the
system is solved on is what it claims, and the flags the branch is not
validated for are refused before anything is computed.

On the committed N2/cc-pVDZ CAS(4,4) CASSCF state
(``golden/n2_ccpvdz_cas44_casscf.mogold``), built through
``generate_golden._build_state``'s own ``--mo-from`` path -- a few seconds of
CPU, no card, no NEVPT2 content in the operator checks. The whole-run
numbers (the NEVPT2 gradient, the smoke test against ``pyscf.grad.casscf``)
are in docs/reference-data.md, "The analytic gradient"; these pin the pieces
a run reads them through.

The Hessian-vector tolerance is written out here, NOT read from the module:
a test that compared against ``HVP_TOL`` would move with it.
"""

from __future__ import annotations

import sys

import nevpt2_grad as ng
import numpy as np
import pytest

HVP_TOL = 1e-6  # nevpt2_grad.HVP_TOL, written out on purpose


@pytest.fixture(scope="module")
def grads():
    st = ng.build_state("n2_ccpvdz_cas44", dict(ng.CASES["n2_ccpvdz_cas44"]),
                        reference="casscf",
                        mo_from=ng.REPO / "golden" / "n2_ccpvdz_cas44_casscf.mogold")
    src = ng.SourceTerms(kappa=np.zeros((st.nmo, st.nmo)), ci=np.zeros(st.ci.shape),
                         origin="test")
    g = ng.CasscfGradients(st, src, with_corr=False)
    # What check_hessian_vector and basis() read, without prepare()'s sweep.
    _g_all, g._h_op, g._h_diag = g.fn.casscf_g_hop(st.mo_coeff, st.ci)
    return g


def test_the_state_is_casscf_shaped(grads):
    st = grads.st
    assert st.casscf and st.n_act_occ == st.nact
    # (S1) is exactly gen_g_hop's variable list: core-active, core-virtual,
    # active-virtual, no active-active and no core-core/virtual-virtual.
    nc, na, nv = st.ncore, st.nact, st.nvirt
    assert grads.nvar == nc * na + nc * nv + na * nv
    kinds = {kind for kind, _p, _q in grads.cons}
    assert kinds == {"S", "f"}


def test_hessian_vector_is_the_jacobian(grads):
    checks = ng.check_hessian_vector(grads)
    assert len(checks) == len(ng.HVP_SEEDS) >= 3
    for chk in checks:
        assert chk.ok, chk.summary
        assert chk.numbers["err"] <= HVP_TOL
        assert chk.numbers["run"] >= 3


@pytest.mark.parametrize("what", ["scaled", "orbital-ci dropped"])
def test_hessian_vector_check_fails_on_a_wrong_operator(grads, what):
    """The check must be able to fail: a 1e-3 relative error in the whole
    operator, or the orbital-CI coupling block dropped (the CASCI branch's
    decoupling, wrongly carried over), is caught by every seed."""
    good, nvar = grads._h_op, grads.nvar

    def bad(x):
        x = np.asarray(x, dtype=float)
        if what == "scaled":
            return 1.001 * good(x)
        orb = np.concatenate([x[:nvar], np.zeros(x.size - nvar)])
        ci = np.concatenate([np.zeros(nvar), x[nvar:]])
        a, b = good(orb), good(ci)
        return np.concatenate([a[:nvar], b[nvar:]])

    grads._h_op = bad
    try:
        checks = ng.check_hessian_vector(grads)
    finally:
        grads._h_op = good
    assert not any(chk.ok for chk in checks)


def test_basis_is_orthonormal_in_sector_and_orthogonal_to_c(grads):
    st = grads.st
    b = grads.basis()
    ndet = st.ci.size
    assert b.shape == (grads.nvar + ndet, grads.nvar + int(grads.allowed.sum()) - 1)
    assert np.abs(b.T @ b - np.eye(b.shape[1])).max() < 1e-12
    ci_block = b[grads.nvar:, grads.nvar:]
    assert np.abs(st.ci.ravel() @ ci_block).max() < 1e-12
    assert np.abs(ci_block[~grads.allowed]).max() == 0.0


def test_dense_operator_is_symmetric_and_positive(grads):
    """At a converged CASSCF minimum the orbital+CI Hessian is symmetric and
    positive definite on the coupled space -- what makes the dense solve and
    the base class's CG both well posed."""
    b = grads.basis()
    j = np.column_stack([b.T @ grads._h_op(b[:, k]) for k in range(b.shape[1])])
    assert np.abs(j - j.T).max() < 1e-10 * np.abs(j).max()
    assert np.linalg.eigvalsh(0.5 * (j + j.T)).min() > 0.0


@pytest.mark.parametrize("extra", [
    ["--energy-ordered"],
    ["--pc", "--source", "dump", "--source-path", "x.gradsrc"],
    ["--pc", "--source", "both", "--source-path", "x.gradsrc"],
])
def test_casscf_refuses_what_it_is_not_validated_for(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["nevpt2_grad.py", "--case", "n2_ccpvdz_cas44",
                                      "--reference", "casscf", *extra])
    with pytest.raises(SystemExit, match="--reference casscf"):
        ng.main()


def test_casscf_takes_an_sc_dump_in_its_own_basis(grads, tmp_path):
    """#125: an SC ``.gradsrc`` on a CASSCF state is read as written. The dump
    is in the golden's basis, and on this branch the state is held in that
    basis too (``act_rot`` is the identity: no section-2.6 rotation), so
    ``A = Y - Y^T`` and ``y`` come through bit for bit -- and the fingerprint
    still refuses another state's CI vector."""
    st = grads.st
    assert np.array_equal(st.act_rot, np.eye(st.nact))
    rng = np.random.default_rng(125)
    y_mat = rng.standard_normal((st.nmo, st.nmo))
    y_ci = rng.standard_normal(st.ci.shape)
    # Through a file object: np.savez appends ".npz" to a bare path.
    path = tmp_path / "sc.gradsrc"
    with path.open("wb") as fh:
        np.savez(fh, kappa_source=y_mat, ci_source=y_ci, ci=st.ci, e_corr=st.e_corr,
                 ncore=st.ncore, nact=st.nact, nvirt=st.nvirt)
    src = ng.load_source_terms(path, st)
    assert src.from_dump
    assert np.array_equal(src.kappa, y_mat - y_mat.T)
    assert np.array_equal(src.ci, y_ci)
    other = st.ci.ravel().copy()
    other[int(np.argmax(np.abs(other)))] *= 0.5
    other = (other / np.linalg.norm(other)).reshape(st.ci.shape)
    bad = tmp_path / "other.gradsrc"
    with bad.open("wb") as fh:
        np.savez(fh, kappa_source=y_mat, ci_source=y_ci, ci=other, e_corr=st.e_corr)
    with pytest.raises(SystemExit, match="different state"):
        ng.load_source_terms(bad, st)


def test_casscf_pc_state_is_the_goldens():
    """``--reference casscf --pc`` (#126): the state carries block2's PC energy
    on the committed CASSCF orbitals, through the same conditioning guard, and
    it is the golden's own ``e_pc_total``; the gradient object differentiates
    PC (its Functional, its steps) while the coupled operator stays the
    reference's."""
    st = ng.build_state("n2_ccpvdz_cas44", dict(ng.CASES["n2_ccpvdz_cas44"]), pc=True,
                        reference="casscf",
                        mo_from=ng.REPO / "golden" / "n2_ccpvdz_cas44_casscf.mogold")
    gold = ng.ngrad.read_golden_pc_total(
        ng.REPO / "golden" / "n2_ccpvdz_cas44_casscf.nevpt2gold")
    assert abs(st.e_corr_pc - gold) < 1e-10
    assert st.pc_gap >= 1e3
    src = ng.SourceTerms(kappa=np.zeros((st.nmo, st.nmo)), ci=np.zeros(st.ci.shape),
                         origin="test")
    g = ng.CasscfGradients(st, src, pc=True)
    assert g.pc and g.fn.pc
    assert (g.step, g.kappa_step) == ng.scheme_steps(True)
    sc = ng.CasscfGradients(st, src)
    assert not sc.pc and (sc.step, sc.kappa_step) == ng.scheme_steps(False)
    assert g.cons == sc.cons and g.nvar == sc.nvar


def test_casci_pc_still_needs_grad_num(monkeypatch):
    """Only the CASSCF branch lost the ``--pc needs --grad-num`` refusal."""
    monkeypatch.setattr(sys, "argv", ["nevpt2_grad.py", "--case", "n2_ccpvdz_cas44", "--pc"])
    with pytest.raises(SystemExit, match="--pc needs --grad-num"):
        ng.main()


@pytest.mark.parametrize(("written", "asked", "ok"), [
    (None, "casscf", False),        # an old CASCI file carries no reference field
    ("casci", "casscf", False),
    ("casscf", "casci", False),
    ("casscf", "casscf", True),
    (None, "casci", True),
])
def test_grad_num_reference_must_match(tmp_path, written, asked, ok):
    """``--grad-num`` reads a ``nevpt2_numgrad.py --npz`` file; its reference
    must be the one this run differentiates, or the two gradients are of
    different functions."""
    path = tmp_path / "g.npz"
    extra = {} if written is None else {"reference": np.array(written)}
    np.savez(path, case=np.array("n2_ccpvdz_cas44"), grad_num=np.zeros((2, 3)),
             grad_num_probed=np.ones((2, 3)), **extra)
    if ok:
        assert ng.load_grad_num(path, "n2_ccpvdz_cas44", pc=False,
                                reference=asked)["grad_num"].shape == (2, 3)
    else:
        with pytest.raises(SystemExit, match="different functions"):
            ng.load_grad_num(path, "n2_ccpvdz_cas44", pc=False, reference=asked)


@pytest.mark.parametrize("extra", [["--mo-from", "x.mogold"], ["--cp-solve", "cg"]])
def test_casscf_flags_refused_on_the_casci_branch(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["nevpt2_grad.py", "--case", "n2_ccpvdz_cas44", *extra])
    with pytest.raises(SystemExit, match="belong to --reference casscf"):
        ng.main()
