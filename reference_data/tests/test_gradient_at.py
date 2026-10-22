"""Hand cases for ``gradient_at.py``'s refusals (#212): a geometry that
changes the point group, a work dir inside ``golden/``, and the guards that
run before any SCF -- plus the continuity verdict's energy-signal rule. No
SCF beyond symmetry detection, no CASSCF, no card: every refusal here fires
before the first one would.

As in ``test_numgrad_casscf.py``, the reach of the energy signal is written
out as a number, NOT read from the module: moving it means editing this file
too, in the same change, saying why.
"""

from __future__ import annotations

import gradient_at as gat
import nevpt2_numgrad as ng
import numpy as np
import pc_on_states as pos
import pytest

#: nevpt2_numgrad.STEP_BOHR, the stencil the energy signal is calibrated on.
ENERGY_GUARD_REACH = 0.0025

ACROLEIN = "acrolein_def2svp_cas65"
N2 = "n2_ccpvdz_cas44"


def _ref(case: str) -> list:
    return gat.reference_geometry(gat.case_spec(case))


def test_reach_value():
    assert gat.ENERGY_GUARD_REACH == ENERGY_GUARD_REACH
    assert gat.TILES == 3


def test_out_of_plane_displacement_refused(tmp_path):
    # Acrolein is planar Cs in z = 0: moving the O atom out of the plane
    # lowers the group to C1, and the call refuses before writing anything.
    geom = ng.displace(_ref(ACROLEIN), 0, 2, 0.01)
    with pytest.raises(SystemExit, match="point group changes from Cs"):
        gat.gradient_at(geom, ACROLEIN, work_dir=tmp_path / "w")
    assert not (tmp_path / "w").exists()


def test_in_plane_displacement_passes_the_group_check(tmp_path):
    # The same atom moved in the plane keeps Cs, so the call gets past the
    # point-group check and is stopped by the next free refusal (the work dir).
    geom = ng.displace(_ref(ACROLEIN), 0, 0, 0.01)
    with pytest.raises(SystemExit, match="inside"):
        gat.gradient_at(geom, ACROLEIN, work_dir=gat.REPO / "golden" / "transient")


def test_work_dir_inside_golden_refused(tmp_path):
    with pytest.raises(SystemExit, match="inside"):
        gat.gradient_at(_ref(N2), N2, work_dir=gat.REPO / "golden")
    with pytest.raises(SystemExit, match="inside"):
        gat.gradient_at(_ref(N2), N2, work_dir=gat.REPO / "golden" / "x" / "y")
    assert not (gat.REPO / "golden" / "x").exists()
    gat.check_work_dir(tmp_path)


def test_pc_on_states_shares_the_guard():
    # Factored out, not copied: pc_on_states's names are gradient_at's.
    assert pos.check_work_dir is gat.check_work_dir
    assert pos.with_placeholder_pc is gat.with_placeholder_pc
    assert pos.gpu_lock is gat.gpu_lock


def test_casci_reference_refused():
    with pytest.raises(SystemExit, match="reference='casci'"):
        gat.gradient_at(_ref(N2), N2, reference="casci")


def test_unknown_case_and_case_without_casscf_reference_refused():
    with pytest.raises(SystemExit, match="is not one of"):
        gat.gradient_at(_ref(N2), "no_such_case")
    # N2 CAS(8,8) is a committed case with no golden/<case>_casscf.mogold.
    with pytest.raises(SystemExit, match="no committed"):
        gat.gradient_at(_ref(N2), "n2_ccpvdz_cas88")


def test_atoms_out_of_order_refused():
    geom = list(reversed(_ref(ACROLEIN)))
    with pytest.raises(SystemExit, match="not the reference's"):
        gat.gradient_at(geom, ACROLEIN)


def test_geometry_forms_agree():
    ref = _ref(N2)
    symbols = [s for s, _ in ref]
    xyz = np.array([v for _, v in ref])
    assert gat.as_geometry((symbols, xyz)) == gat.as_geometry(ref)


def _sig(**kw):
    sig = dict(e_jump=1e-7, converged=True, g_opt=4e-9, act_defect=1e-7, ci_defect=1e-8)
    sig.update(kw)
    return sig


@pytest.mark.parametrize(("d", "applies"), [
    (0.0, True),
    (ENERGY_GUARD_REACH, True),
    (ENERGY_GUARD_REACH / 4, True),
    (-ENERGY_GUARD_REACH * 1.01, False),
    (0.1, False),
])
def test_energy_guard_reach(d, applies):
    dvec = np.zeros((3, 3))
    dvec[1, 0] = d
    assert gat.energy_guard_applies(dvec) is applies


def test_continuity_verdict():
    # Within reach the energy signal counts, NaN included (a signal that could
    # not be evaluated trips); beyond it only the other two do.
    assert gat.continuity_verdict(_sig(e_jump=1.0), energy_enforced=True) == ["energy"]
    assert gat.continuity_verdict(_sig(e_jump=float("nan")), energy_enforced=False) == []
    assert gat.continuity_verdict(_sig(e_jump=float("nan"), act_defect=1.0),
                                  energy_enforced=False) == ["overlap"]
    assert gat.continuity_verdict(_sig(converged=False), energy_enforced=False) == ["gradient"]


def test_placeholder_needs_a_no_pc_golden(tmp_path):
    with pytest.raises(SystemExit, match="placeholder"):
        gat.write_transient("N 0 0 0; N 0 0 1.1", ng.CASES[N2], tmp_path / "m.mogold",
                            tmp_path / "g.nevpt2gold", name="x", pc=True, placeholder_pc=True)


def test_find_binary_refuses_a_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="no such"):
        gat.find_binary(tmp_path / "nevpt2_sc_pseudodensity")
