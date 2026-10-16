"""Hand cases for ``pc_on_states.py``'s plumbing, and for the reader it gets
its states through (``golden_format.read_states_manifest``): the manifest
reader's refusals and its round trip with ``nevpt2_numgrad``'s writer, the
parse of ``nevpt2_demo --pc``'s table and full-precision line, the PC
conditioning guard applied to that table, the NaN placeholders a transient
golden carries, and the ``golden/`` refusal. Synthetic inputs only: no SCF,
no card.

The guard's threshold is written out here as a number, as in
``test_refusal_guards.py``: moving it means editing this file too.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import golden_format as gf
import nevpt2_numgrad as ng
import numpy as np
import pc_on_states as pos
import pytest

PC_MIN_GAP = 1e3  # generate_golden.PC_MIN_GAP

# --- golden_format.read_states_manifest --------------------------------------

HEAD = dict(case="acrolein_def2svp_cas65", reference="casscf", basis="def2-svp",
            ncas=5, nelecas=6, cas_irreps={"A'": 1, 'A"': 4}, step=0.0025,
            finish=dict(method="mcscf.newton"))


def _state(label, **kw):
    st = dict(label=label, atom="H 0 0 0; H 0 0 0.74", s=0.0, e_casci=-1.0,
              e_corr=-0.1, mogold=f"{label}.mogold", tripped=[])
    st.update(kw)
    return st


def _manifest(tmp_path, states, files=None, **head):
    for name in (files if files is not None else [s["mogold"] for s in states
                                                    if "mogold" in s]):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "states.json").write_text(json.dumps(dict(HEAD, **head, states=states)))
    return tmp_path


def test_manifest_kept_and_not_kept(tmp_path):
    states = [_state("ref"), _state("0x+0", s=0.0025),
              _state("0x-0", tripped=["energy"]),
              _state("1y+2", refused="the --mo-from gate refuses")]
    m = gf.read_states_manifest(_manifest(tmp_path, states))
    assert [s["label"] for s in m["states"]] == ["ref", "0x+0", "0x-0", "1y+2"]
    assert [s["kept"] for s in m["states"]] == [True, True, False, False]
    assert m["states"][0]["mogold_path"] == tmp_path / "ref.mogold"
    assert m["states"][2]["mogold_path"] is None
    assert m["case"] == HEAD["case"] and m["step"] == HEAD["step"]


def test_manifest_tripped_state_needs_no_mogold(tmp_path):
    st = _state("0x-0", tripped=["overlap"])
    del st["mogold"], st["e_casci"], st["e_corr"]
    m = gf.read_states_manifest(_manifest(tmp_path, [_state("ref"), st]))
    assert not m["states"][1]["kept"]


@pytest.mark.parametrize(("states", "files", "head", "match"), [
    ([_state("ref"), _state("ref")], None, {}, "two states are labelled 'ref'"),
    ([_state("ref")], [], {}, "ref.mogold is not there"),
    ([{k: v for k, v in _state("ref").items() if k != "e_corr"}], None, {}, "lacks e_corr"),
    ([{k: v for k, v in _state("ref").items() if k != "atom"}], None, {}, "lacks atom"),
    ([_state("ref")], None, dict(reference="casci"), "only the casscf route"),
])
def test_manifest_refusals(tmp_path, states, files, head, match):
    with pytest.raises(gf.FormatError, match=match):
        gf.read_states_manifest(_manifest(tmp_path, states, files, **head))


def test_manifest_missing_header_field(tmp_path):
    (tmp_path / "states.json").write_text(json.dumps({"case": "x", "states": []}))
    with pytest.raises(gf.FormatError, match="missing reference"):
        gf.read_states_manifest(tmp_path)


def test_manifest_not_json(tmp_path):
    (tmp_path / "states.json").write_text("{")
    with pytest.raises(gf.FormatError, match="not JSON"):
        gf.read_states_manifest(tmp_path)


def test_manifest_round_trips_nevpt2_numgrad_writer(tmp_path):
    """What ``nevpt2_numgrad._write_manifest`` writes is what the reader takes."""
    natm = 2
    args = SimpleNamespace(case=HEAD["case"], step=0.0025)
    rec = dict(natm=natm, basis="def2-svp", ncas=5, nelecas=6, cas_irreps=HEAD["cas_irreps"])
    r0 = dict(_state("ref"), e_casscf=-1.1)
    unit = np.zeros((natm, 3))
    unit[0, 0] = 1.0
    st = dict(_state("0x+0", s=0.0025), e_casscf=-1.1, key="0x", rung=0, sign=1,
              task=dict(unit=unit))
    bad = dict(label="0x-0", atom="H 0 0 0; H 0 0 0.74", s=-0.0025, tripped=["energy"],
               key="0x", rung=0, sign=-1, task=dict(unit=unit))
    for name in ("ref.mogold", "0x+0.mogold"):
        (tmp_path / name).write_bytes(b"")
    ng._write_manifest(tmp_path, args, rec, r0, [st, bad])
    m = gf.read_states_manifest(tmp_path)
    assert [s["label"] for s in m["states"]] == ["ref", "0x+0", "0x-0"]
    assert [s["kept"] for s in m["states"]] == [True, True, False]
    assert m["states"][1]["displacement_bohr"][0] == [0.0025, 0.0, 0.0]
    assert m["finish"]["conv_tol_grad"] == ng.FINISH_CONV_TOL_GRAD


# --- parse_pc and pc_guard ---------------------------------------------------

#: nevpt2_demo --pc's table on acrolein's reference state (#138's run), with
#: the block2 columns NaN as a transient golden makes them. Column padding is
#: collapsed to two spaces: parse_pc splits on whitespace.
TABLE = """\
PC-NEVPT2 (all eight classes)
S cut at tau=1e-13 * s_max; a class is refused below a gap of 1e+03 (...)
class  d dropped  max dropped  min kept  gap  min den  PC energy (Ha)  block2 ref (Ha)  |delta|
Sr  125  50  7.1e-17  9.8e-10  1.4e+07  0.606  -0.0143856616  nan  nan
Si  125  85  1.7e-16  1.1e-07  6.5e+08  0.613  -0.0001401760  nan  nan
Sijrs  -  -  -  -  -  -  -0.2568239251  nan  nan  (PC = SC)
Sijr  5  0  -  1.2e-03  1.2e+10  1.260  -0.0118534238  nan  nan
Srsi  5  0  -  3.1e-02  3.1e+11  1.252  -0.1281583297  nan  nan
Srs  25  0  -  2.5e-07  2.5e+06  1.540  -0.0241837547  nan  nan
Sij  25  0  -  2.9e-10  {sij_gap}  1.554  -0.0014680155  nan  nan
Sir  50  2  3.3e-17  7.4e-08  2.2e+09  0.637  -0.0858873194  nan  nan

E_corr (PC-NEVPT2, ours)   = -0.5229006058 Ha
E_corr (PC-NEVPT2, block2) = nan Ha  |delta|=nan
E_corr full precision (PC-NEVPT2) = -0.52290060578354236 Ha
PC check: 8 of 8 classes computed, 0 refused, max class |delta| vs block2 0.00e+00

FAIL: PC-NEVPT2 per-class energies and the total match the golden block2 reference to 1e-07
"""


def test_parse_pc_table():
    pc = pos.parse_pc(TABLE.format(sij_gap="2.9e+03"))
    assert pc["total"] == -0.52290060578354236
    sr, sijrs, sir = pc["classes"][0], pc["classes"][2], pc["classes"][7]
    assert (sr["d"], sr["dropped"], sr["max_dropped"], sr["min_kept"], sr["gap"]) == (
        125, 50, 7.1e-17, 9.8e-10, 1.4e7)
    assert sr["energy"] == -0.0143856616 and sr["status"] == "done"
    assert sijrs["gap"] is None and sijrs["energy"] == -0.2568239251
    assert pc["classes"][3]["max_dropped"] == 0.0  # "-": nothing dropped
    assert sir["dropped"] == 2


def test_parse_pc_refused_class_has_no_energy_and_no_total():
    text = TABLE.format(sij_gap="2.9e+03").replace(
        "  -0.0014680155  nan  nan",
        "  REFUSED  nan  -  (S's gap 9.0e+02 < PC_MIN_GAP 1e+03)")
    text = text.replace("E_corr full precision (PC-NEVPT2) = -0.52290060578354236 Ha\n", "")
    pc = pos.parse_pc(text)
    sij = pc["classes"][6]
    assert sij["status"] == "refused" and sij["energy"] is None
    assert "PC_MIN_GAP" in sij["why"]
    assert pc["total"] is None
    ok, _, why = pos.pc_guard(pc)
    assert not ok and "Sij: refused" in why


def test_parse_pc_missing_class_refused():
    text = "\n".join(line for line in TABLE.format(sij_gap="2.9e+03").splitlines()
                     if not line.startswith("Sir "))
    with pytest.raises(SystemExit, match="no row for Sir"):
        pos.parse_pc(text)


@pytest.mark.parametrize(("gap", "ok"), [
    (1.01 * PC_MIN_GAP, True), (0.99 * PC_MIN_GAP, False), (2.9e3, True), (1.2, False),
])
def test_pc_guard_threshold(gap, ok):
    """The guard is generate_golden._check_pc_conditioning on the device's
    gap column: the worst class decides, at PC_MIN_GAP."""
    pc = pos.parse_pc(TABLE.format(sij_gap=f"{gap:.6e}"))
    got_ok, min_gap, why = pos.pc_guard(pc)
    assert got_ok is ok
    assert min_gap == pytest.approx(gap)
    assert (why == "") is ok
    if not ok:
        assert "PC block Sij" in why


# --- the transient golden ----------------------------------------------------


def test_placeholder_pc_is_nan_and_round_trips(tmp_path):
    arrays = {"ci": np.ones(4), "class_energies": np.zeros(8)}
    got = pos.with_placeholder_pc(arrays)
    assert np.isnan(got["pc_class_energies"]).all() and got["pc_class_energies"].shape == (8,)
    assert got["e_pc_total"].shape == (1,) and np.isnan(got["e_pc_total"][0])
    assert "pc_class_energies" not in arrays  # the input is not modified
    path = tmp_path / "t.nevpt2gold"
    pos.gg.write_golden(path, ncas=2, nelec_a=1, nelec_b=1, ncore=0, ndet=4, e_casci=-1.0,
                        e_nevpt2_total=-0.1, reference_kind="casscf", arrays=got)
    assert np.isnan(ng.read_golden_pc_total(path))


def test_placeholder_refuses_a_golden_with_pc_fields():
    with pytest.raises(SystemExit, match="already has PC fields"):
        pos.with_placeholder_pc({"e_pc_total": np.array([-0.5])})


def test_work_dir_inside_golden_refused(tmp_path):
    with pytest.raises(SystemExit, match="never written there"):
        pos.check_work_dir(pos.REPO / "golden")
    with pytest.raises(SystemExit, match="never written there"):
        pos.check_work_dir(pos.REPO / "golden" / "transient")
    pos.check_work_dir(tmp_path)


def test_tolerances():
    assert pos.REF_PC_TOL == 1e-7
    assert pos.WIRING_TOL == 1e-10
    assert pos.TILES == 3
