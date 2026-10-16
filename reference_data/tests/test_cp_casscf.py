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
    ["--pc"],
    ["--source", "dump", "--source-path", "x.gradsrc"],
    ["--energy-ordered"],
    ["--grad-num", "x.npz"],
])
def test_casscf_refuses_what_it_is_not_validated_for(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["nevpt2_grad.py", "--case", "n2_ccpvdz_cas44",
                                      "--reference", "casscf", *extra])
    with pytest.raises(SystemExit, match="--reference casscf"):
        ng.main()


@pytest.mark.parametrize("extra", [["--mo-from", "x.mogold"], ["--cp-solve", "cg"]])
def test_casscf_flags_refused_on_the_casci_branch(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["nevpt2_grad.py", "--case", "n2_ccpvdz_cas44", *extra])
    with pytest.raises(SystemExit, match="belong to --reference casscf"):
        ng.main()
