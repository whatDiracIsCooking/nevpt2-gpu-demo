"""Hand cases for ``nevpt2_grad.py --df`` (#166), the density-fitted analytic
gradient: that the state it differentiates is ``golden/n2_ccpvdz_cas44_df``'s
own, that every :class:`Functional` -- at the reference and at a displaced
geometry -- runs on fitted integrals (so the PySCF-sourced side is PySCF's
DF-NEVPT2, not a conventional calculation), that a dump of the other flavour
is refused by its own fingerprint, and that the flag combinations the fitted
route is not validated for are refused before any SCF.

On the committed N2/cc-pVDZ CAS(4,4) state: a few seconds of CPU, no card.
The whole-run verdict (both providers against #167's ``grad_num``) is
docs/testing.md, "The density-fitted gradient against the numerical one";
these pin the pieces a run reads it through.
"""

from __future__ import annotations

import sys

import golden_format as gf
import nevpt2_grad as ng
import numpy as np
import pytest

GOLDEN = ng.REPO / "golden"
#: the golden-state tolerance, written out on purpose (nevpt2_grad.REF_STATE_TOL)
REF_STATE_TOL = 1e-10


@pytest.fixture(scope="module")
def df_state():
    return ng.build_state("n2_ccpvdz_cas44", dict(ng.CASES["n2_ccpvdz_cas44"]),
                          auxbasis="cc-pvdz-jkfit")


def test_df_state_is_the_df_goldens(df_state):
    """The fitted state is golden/n2_ccpvdz_cas44_df's, and visibly not the
    conventional golden's: the two flavours' E_corr differ by far more than
    the tolerance either is rebuilt to, so the golden check cannot pass the
    wrong one."""
    st = df_state
    assert st.df and st.auxbasis == "cc-pvdz-jkfit"
    df = gf.read_golden_header(GOLDEN / "n2_ccpvdz_cas44_df.nevpt2gold")
    conv = gf.read_golden_header(GOLDEN / "n2_ccpvdz_cas44.nevpt2gold")
    assert abs(st.e_casci - df["e_casci"]) <= REF_STATE_TOL
    assert abs(st.e_corr - df["e_nevpt2_total"]) <= REF_STATE_TOL
    assert abs(st.e_corr - conv["e_nevpt2_total"]) > 1e-6
    assert abs(st.e_casci - conv["e_casci"]) > 1e-6


def test_functional_is_fitted_at_the_reference(df_state):
    """The Functional on a fitted state reproduces the state's E_CASCI,
    E_corr, f and F from its own fitted integrals -- dfnevpt2._ERIS and the
    fitted RHF's JK -- so a quotient of it is a quotient of the DF energy."""
    st = df_state
    fn = ng.Functional(st)
    assert fn.with_df is not None and fn.mf.with_df is fn.with_df
    assert fn.mc.with_df is fn.with_df
    b = fn.evaluate(st.mo_coeff, st.ci)
    assert abs(b["e_casci"] - st.e_casci) < 1e-11
    assert abs(b["e_corr"] - st.e_corr) < 1e-11
    assert np.abs(b["f"] - st.f).max() < 1e-11
    assert np.abs(b["fock_rhf"] - st.fock_rhf).max() < 1e-11


def test_functional_fits_on_the_displaced_geometry(df_state):
    """A displaced Functional builds its own fit: the auxiliary functions sit
    on the displaced nuclei, which is what carries the fit's own nuclear
    derivative into the frozen path's quotient."""
    st = df_state
    geom = ng.ngrad.displace(st.geom, 1, 2, 0.01)
    fn = ng.Functional(st, geom)
    auxmol = fn.with_df.auxmol
    assert np.allclose(auxmol.atom_coords(), fn.mol.atom_coords())
    ref = ng.Functional(st)
    assert not np.allclose(auxmol.atom_coords(), ref.with_df.auxmol.atom_coords())
    assert fn.with_df.get_naoaux() == 140


@pytest.mark.parametrize("which", ["conventional", "df"])
def test_dump_of_the_other_flavour_is_refused(df_state, tmp_path, which):
    """A .gradsrc carries no flavour marker; the CI vector it was taken at
    is the fingerprint, and the two flavours' states differ there (8.5e-5 on
    N2 CAS(4,4)) far beyond DUMP_CI_TOL. A dump with the fitted state's CI
    vector but the conventional E_corr is refused by the energy check behind
    it."""
    st = df_state
    conv = ng.build_state("n2_ccpvdz_cas44", dict(ng.CASES["n2_ccpvdz_cas44"]))
    rng = np.random.default_rng(166)
    y_mat = rng.standard_normal((st.nmo, st.nmo))
    y_ci = rng.standard_normal(st.ci.shape)
    path = tmp_path / "x.gradsrc"
    if which == "conventional":
        with path.open("wb") as fh:
            np.savez(fh, kappa_source=y_mat, ci_source=y_ci, ci=conv.ci_golden,
                     e_corr=conv.e_corr, ncore=st.ncore, nact=st.nact, nvirt=st.nvirt)
        with pytest.raises(SystemExit, match="different state"):
            ng.load_source_terms(path, st)
        return
    with path.open("wb") as fh:
        np.savez(fh, kappa_source=y_mat, ci_source=y_ci, ci=st.ci_golden,
                 e_corr=conv.e_corr, ncore=st.ncore, nact=st.nact, nvirt=st.nvirt)
    with pytest.raises(SystemExit, match="E_corr is"):
        ng.load_source_terms(path, st)
    with path.open("wb") as fh:
        np.savez(fh, kappa_source=y_mat, ci_source=y_ci, ci=st.ci_golden,
                 e_corr=st.e_corr, ncore=st.ncore, nact=st.nact, nvirt=st.nvirt)
    assert ng.load_source_terms(path, st).from_dump


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["nevpt2_grad.py", *argv])
    ng.main()


@pytest.mark.parametrize(("argv", "msg"), [
    (["--df", "--pc", "--grad-num", "x.npz"], "--pc"),
    (["--df", "--reference", "casscf"], "--reference casscf"),
    (["--df", "--casci-only"], "--casci-only"),
    (["--df", "--grad-num", "x.npz"], "--grad-num"),
    (["--auxbasis", "cc-pvdz-jkfit"], "--auxbasis needs --df"),
])
def test_df_refused_flag_combinations(argv, msg, monkeypatch, capsys):
    monkeypatch.setattr(ng, "build_state", _no_scf)
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "--case", "n2_ccpvdz_cas44", *argv)
    assert exc.value.code == 2
    assert msg in capsys.readouterr().err


def _no_scf(*_a, **_k):
    raise AssertionError("the state was built before the refusal")


@pytest.mark.parametrize(("argv", "msg"), [
    (["--case", "lif_def2svp_cas46"], "no golden/lif_def2svp_cas46_df.nevpt2gold"),
    (["--case", "n2_ccpvdz_cas44", "--energy-ordered"], "--energy-ordered make this"),
    (["--case", "n2_ccpvdz_cas44", "--ncas", "4"], "--ncas make this"),
    (["--case", "n2_ccpvdz_cas44", "--auxbasis", "cc-pvdz-ri"],
     "fitted in 140 auxiliary functions"),
])
def test_df_refused_before_any_scf(argv, msg, monkeypatch):
    """A state with no _df golden, an overridden selection, or an auxiliary
    basis the golden was not fitted in: refused before the state is built."""
    monkeypatch.setattr(ng, "build_state", _no_scf)
    with pytest.raises(SystemExit, match=msg):
        _main(monkeypatch, "--df", *argv)


def test_df_state_refuses_pc_and_casscf():
    spec = dict(ng.CASES["n2_ccpvdz_cas44"])
    for kw in (dict(pc=True), dict(reference="casscf")):
        with pytest.raises(SystemExit, match="SC on the CASCI branch only"):
            ng.build_state("n2_ccpvdz_cas44", spec, auxbasis="cc-pvdz-jkfit", **kw)
