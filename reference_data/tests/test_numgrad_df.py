"""Hand cases for ``nevpt2_numgrad.py --df`` (#167), the density-fitted
numerical gradient: that a symmetry-changing displacement is refused on the
fitted route exactly as on the conventional one (the in-plane-``Cs`` rule,
docs/reference-data.md, "The numerical gradient"), and refused before any
SCF; the flavour refusals around it (a golden or sidecar of the other
flavour, an auxiliary basis that is not the golden's); the flag combinations
``main`` refuses; and that the committed density-fitted sidecar carries its
``grad_num`` through ``golden_format`` and no second parser.

Symmetry detection and file headers only -- no SCF. A stand-in for
``nevpt2_numgrad.energy`` that fails the test is installed wherever a refusal
must land before the first solve, so "refused" cannot quietly become "refused
after the sweep".
"""

from __future__ import annotations

import sys
from pathlib import Path

import golden_format as gf
import nevpt2_numgrad as ng
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "golden"
ACROLEIN = REPO / "reference_data" / "geometries" / "acrolein.xyz"


def _no_scf(*_a, **_k):
    raise AssertionError("a displaced or reference SCF ran before the refusal")


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["nevpt2_numgrad.py", *argv])
    ng.main()


# --- the in-plane-Cs rule, on the fitted route -------------------------------


def test_df_symmetry_changing_displacement_is_refused(monkeypatch, capsys):
    """Acrolein (planar Cs) on the --df route, with an out-of-plane component
    named: refused with the point group it breaks to, before any SCF. No
    acrolein _df golden exists, so the two header reads are stood in for; what
    is under test is that the fitted route reaches the same refusal."""
    monkeypatch.setattr(ng, "energy", _no_scf)
    monkeypatch.setattr(ng.gf, "read_golden_header",
                        lambda _p: {"reference_kind": "casci"})
    monkeypatch.setattr(ng, "_check_golden_flavour", lambda _p, _g, case: (
        ng.fitted_naux(_g, case["basis"], case["auxbasis"])))
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "--df", "--auxbasis", "weigend", "--case", "acrolein_df_handcase",
              "--geometry", str(ACROLEIN), "--basis", "sto-3g", "--ncas", "5",
              "--nelecas", "6",
              # exists, so the --df existence check passes; its header is stood in for
              "--golden", str(GOLDEN / "n2_ccpvdz_cas44_df.nevpt2gold"),
              "--components", "1x,0z", "--no-write")
    msg = str(exc.value.code)
    assert "atom 0 (O) z" in msg
    assert "the detected point group changes from Cs (used: Cs)" in msg
    assert "to C1 (used: C1)" in msg
    assert "Refused rather than differenced" in msg
    assert "in-plane displacements" in msg


def test_df_auto_keeps_out_of_plane_components_as_symmetry_zeros():
    """--components auto drops the eight out-of-plane components as the exact
    zeros they are and keeps the sixteen in-plane ones; the selection does not
    depend on the fit, so the fitted route inherits it unchanged."""
    geom = ng.read_xyz(ACROLEIN)
    ref = ng.point_group(geom, "sto-3g")
    assert ref == ("Cs", "Cs")
    probed, skipped = ng.select_components(geom, "sto-3g", ref, ng.STEP_BOHR, None)
    assert sorted(skipped) == [(i, 2) for i in range(8)]
    assert sorted(probed) == [(i, c) for i in range(8) for c in (0, 1)]


def test_point_group_refusal_text():
    with pytest.raises(SystemExit, match=r"changes from Cs \(used: Cs\) at the "
                       r"reference geometry to C1 \(used: C1\)"):
        ng._check_point_group(("Cs", "Cs"), ("C1", "C1"), "atom 0 (O) z, step +0.0025 bohr")


# --- the flavour refusals -----------------------------------------------------


def _n2_case(auxbasis):
    return dict(ng.CASES["n2_ccpvdz_cas44"], auxbasis=auxbasis)


N2 = ng.read_xyz(REPO / "reference_data" / "geometries" / "n2.xyz")


def test_df_golden_flavour_accepted():
    naux = ng._check_golden_flavour(GOLDEN / "n2_ccpvdz_cas44_df.nevpt2gold", N2,
                                    _n2_case("cc-pvdz-jkfit"))
    assert naux == 140


def test_conventional_golden_refused_on_the_df_route():
    with pytest.raises(SystemExit, match="is a conventional golden"):
        ng._check_golden_flavour(GOLDEN / "n2_ccpvdz_cas44.nevpt2gold", N2,
                                 _n2_case("cc-pvdz-jkfit"))


def test_df_golden_refused_on_the_conventional_route():
    with pytest.raises(SystemExit, match="is a density-fitted golden"):
        ng._check_golden_flavour(GOLDEN / "n2_ccpvdz_cas44_df.nevpt2gold", N2,
                                 _n2_case(None))


def test_another_auxiliary_basis_refused():
    with pytest.raises(SystemExit, match="fitted in 140 auxiliary functions"):
        ng._check_golden_flavour(GOLDEN / "n2_ccpvdz_cas44_df.nevpt2gold", N2,
                                 _n2_case("cc-pvdz-ri"))


@pytest.mark.parametrize(("header", "df", "naux", "refused"), [
    ({"nao": 1}, False, None, False),
    ({"nao": 1, "naux": 140}, True, None, False),
    ({"nao": 1, "naux": 140}, True, 140, False),
    ({"nao": 1, "naux": 140}, True, 112, True),
    ({"nao": 1}, True, None, True),
    ({"nao": 1, "naux": 140}, False, None, True),
])
def test_sidecar_flavour(header, df, naux, refused):
    if refused:
        with pytest.raises(SystemExit, match="Refused rather than written"):
            ng._check_sidecar_flavour(Path("x.gradgold"), header, df, naux)
    else:
        ng._check_sidecar_flavour(Path("x.gradgold"), header, df, naux)


def test_df_write_into_a_committed_sidecar_without_grad_num_refused_before_any_scf(
        monkeypatch):
    """CAS(8,8)'s fitted sidecar is committed without grad_num: the in-place
    write is refused before the sweep, not after it."""
    monkeypatch.setattr(ng, "run", _no_scf)
    with pytest.raises(SystemExit, match="n2_ccpvdz_cas88_df.gradgold is committed"):
        _main(monkeypatch, "--df", "--case", "n2_ccpvdz_cas88")


def test_df_write_into_a_conventional_sidecar_refused_before_any_scf(monkeypatch):
    monkeypatch.setattr(ng, "run", _no_scf)
    with pytest.raises(SystemExit, match="a conventional \\(NEVPT2D1\\) sidecar"):
        _main(monkeypatch, "--df", "--case", "n2_ccpvdz_cas44",
              "--sidecar", str(GOLDEN / "n2_ccpvdz_cas44.gradgold"),
              "--out", "unused.gradgold")


@pytest.mark.parametrize(("argv", "msg"), [
    (["--df", "--pc"], "--df is SC only"),
    (["--df", "--reference", "casscf", "--no-write"], "--reference casci only"),
    (["--auxbasis", "cc-pvdz-jkfit"], "--auxbasis needs --df"),
])
def test_df_refused_flag_combinations(argv, msg, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, *argv)
    assert exc.value.code == 2
    assert msg in capsys.readouterr().err


# --- the committed density-fitted grad_num --------------------------------------


def test_committed_df_sidecar_carries_grad_num():
    """golden/n2_ccpvdz_cas44_df.gradgold: the NEVPT2F1 sidecar, its twelve
    --df --gradgold arrays first and the four grad_num* after them, read by
    golden_format alone. N2 is linear under every displacement, so all six
    components are probed; the two axial ones are equal and opposite."""
    header, arrays = gf.read_gradgold(GOLDEN / "n2_ccpvdz_cas44_df.gradgold")
    assert header["naux"] == 140
    names = list(arrays)
    assert names[-4:] == ["grad_num", "grad_num_residual", "grad_num_probed",
                          "grad_num_step"]
    assert len(names) == 16
    assert arrays["grad_num"].shape == (2, 3)
    assert arrays["grad_num_step"].tolist() == [ng.STEP_BOHR]
    assert np.all(arrays["grad_num_probed"] == 1.0)
    g = arrays["grad_num"]
    assert g[0, 2] > 0 and abs(g[0, 2] + g[1, 2]) < 1e-8
    assert np.abs(g[:, :2]).max() < 1e-9
