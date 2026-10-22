"""Hand cases for ``nevpt2_seminum_hessian.py`` and the ``.hessgold`` format
(#217): the sidecar's round trip and its not-computed contract, the
coordinate set and its refusals, the ladder rule, the sum rules and the
frequencies on an exactly known Hessian, and the cache. No CASSCF and no
card: every refusal here fires before the first SCF would, and the Hessians
are a pair potential's, whose derivatives are written out.

As in ``test_gradient_at.py``, the ladder and the gates are written out as
numbers, NOT read from the module: moving one means editing this file too,
in the same change, saying why.
"""

from __future__ import annotations

import json

import golden_format as gf
import gradient_at as gat
import nevpt2_grad as ngr
import nevpt2_numgrad as ng
import nevpt2_seminum_hessian as snh
import numpy as np
import pytest

#: The shipped ladder and gates (docs/testing.md, "The seminumerical nuclear
#: Hessian").
LADDER = (7, 11)
SYMMETRY_TOL = 1e-5
SUM_RULE_TOL = 1e-5

ACROLEIN = "acrolein_def2svp_cas65"
HSOH = "hsoh_def2svp_cas66"
N2 = "n2_ccpvdz_cas44"


def test_pinned_numbers():
    assert snh.LADDER == LADDER
    assert snh.SYMMETRY_TOL == SYMMETRY_TOL
    assert snh.SUM_RULE_TOL == SUM_RULE_TOL
    assert ngr.LADDER_RUN == 3


# --- a pair potential: an exactly known gradient and Hessian -----------------


def _pair(x: np.ndarray, k: float = 0.7, r0: float = 1.9):
    """``E = sum_{i<j} k (r_ij - r0)^2`` at ``x`` ``(natm, 3)``: its gradient
    ``(3 natm,)`` and Hessian, translationally and rotationally invariant at
    every geometry."""
    n = len(x)
    g = np.zeros((n, 3))
    h = np.zeros((n, 3, n, 3))
    for i in range(n):
        for j in range(i + 1, n):
            d = x[i] - x[j]
            r = np.linalg.norm(d)
            u = d / r
            g[i] += 2 * k * (r - r0) * u
            g[j] -= 2 * k * (r - r0) * u
            b = 2 * k * (np.outer(u, u) + (r - r0) / r * (np.eye(3) - np.outer(u, u)))
            h[i, :, i] += b
            h[j, :, j] += b
            h[i, :, j] -= b
            h[j, :, i] -= b
    return g.ravel(), h.reshape(3 * n, 3 * n)


#: A planar four-atom geometry in z = 0, bohr, off its pair-potential minimum.
PLANAR = np.array([[0.0, 0.0, 0.0], [2.1, 0.3, 0.0], [-0.4, 1.8, 0.0], [1.7, 2.4, 0.0]])
PLANAR_SYMBOLS = ["O", "C", "H", "H"]
IN_PLANE = [k for k in range(12) if k % 3 != 2]


def test_rules_vanish_on_an_exact_hessian():
    g, h = _pair(PLANAR)
    an = snh.analyse(h, list(range(12)), g, PLANAR)
    assert an["max_asym"] < 1e-14
    assert an["trans_columns"] < 1e-13 and an["trans_rows"] < 1e-13
    assert an["translations_in_set"] == ["x", "y", "z"]
    assert an["rot_rule"] < 1e-12
    assert np.isnan(an["zero_by_symmetry"])


def test_in_plane_subset_marks_the_rest_not_computed():
    g, h = _pair(PLANAR)
    raw = np.full_like(h, np.nan)
    raw[:, IN_PLANE] = h[:, IN_PLANE]
    an = snh.analyse(raw, IN_PLANE, g, PLANAR)
    m = np.zeros(12, dtype=bool)
    m[IN_PLANE] = True
    assert np.isfinite(an["hessian"][np.ix_(m, m)]).all()
    assert np.isnan(an["hessian"][~np.outer(m, m)]).all()
    assert np.isnan(an["asymmetry"][~np.outer(m, m)]).all()
    assert an["translations_in_set"] == ["x", "y"]
    assert an["rotations_in_set"] == ["z"]
    assert an["trans_columns"] < 1e-13 and an["trans_rows"] < 1e-13
    assert an["rot_rule"] < 1e-12
    # the out-of-plane rows of in-plane columns: zero by symmetry, measured
    assert an["zero_by_symmetry"] < 1e-15


def test_a_broken_rule_is_seen():
    g, h = _pair(PLANAR)
    h = h.copy()
    h[4, 7] += 1e-3
    an = snh.analyse(h, list(range(12)), g, PLANAR)
    assert an["max_asym"] == pytest.approx(1e-3)
    assert an["trans_columns"] == pytest.approx(1e-3)


def test_frequencies_whole_set_is_harmonic_analysis():
    from pyscf.hessian import thermo

    g, h = _pair(PLANAR)
    fr = snh.frequencies(PLANAR_SYMBOLS, PLANAR, h, list(range(12)))
    mol = snh._mol(PLANAR_SYMBOLS, PLANAR)
    ref = thermo.harmonic_analysis(mol, h.reshape(4, 3, 4, 3).transpose(0, 2, 1, 3),
                                   imaginary_freq=False)
    np.testing.assert_array_equal(fr["freq_cm"], ref["freq_wavenumber"])
    assert fr["via"] == "harmonic_analysis"


def test_frequencies_in_plane_are_the_whole_sets_in_plane_modes():
    # In-plane and out-of-plane decouple at a planar geometry, so the whole
    # set's harmonic analysis is the union of the two blocks': the in-plane
    # subset's frequencies are exactly its in-plane ones (2 translations and
    # 1 rotation projected: 8 - 3 = 5 of them).
    g, h = _pair(PLANAR)
    assert np.abs(h[np.ix_(IN_PLANE, [2, 5, 8, 11])]).max() < 1e-15
    whole = snh.frequencies(PLANAR_SYMBOLS, PLANAR, h, list(range(12)))["freq_cm"]
    sub = snh.frequencies(PLANAR_SYMBOLS, PLANAR, h, IN_PLANE)
    assert sub["projected"] == 3
    assert len(sub["freq_cm"]) == 5
    for f in sub["freq_cm"]:
        assert np.min(np.abs(whole - f)) < 1e-6 * max(1.0, abs(f))


def test_frequencies_refuse_a_set_that_is_not_a_symmetry_block():
    g, h = _pair(PLANAR)
    with pytest.raises(SystemExit, match="straddle"):
        snh.frequencies(PLANAR_SYMBOLS, PLANAR, h, [0, 1, 3, 4, 6, 7, 9])


# --- the ladder rule ----------------------------------------------------------


def _quadratic_ladder(exps, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    steps = ngr.ladder_steps(exps)
    truth = rng.standard_normal(6)
    c = rng.standard_normal(6)
    fd = np.array([truth + c * h**2 + 0.7 * c * h**4 + noise * rng.standard_normal(6) / h
                   for h in steps])
    return truth, fd


def test_judge_column_converges_on_h_squared():
    truth, fd = _quadratic_ladder((3, 9))
    j = snh.judge_column(fd)
    assert j["ok"] and j["run"] >= 3
    assert np.abs(j["value"] - truth).max() < 1e-8
    assert np.isfinite(j["residual"]).all()


def test_judge_column_refuses_noise():
    truth, fd = _quadratic_ladder((10, 14), noise=1e-6)
    j = snh.judge_column(fd)
    assert not j["ok"]
    assert np.isfinite(j["value"]).all()


def test_ladder_too_short_refused():
    with pytest.raises(SystemExit, match="rungs"):
        snh.ladder_steps((7, 10))
    assert snh.ladder_steps((7, 11)) == [2.0**-k for k in range(7, 12)]


# --- the coordinate set -------------------------------------------------------


def _steps():
    return snh.ladder_steps(LADDER)


def test_acrolein_differences_the_in_plane_coordinates():
    kept, skipped = snh.coordinate_set(gat.case_spec(ACROLEIN), _steps())
    assert kept == [k for k in range(24) if k % 3 != 2]
    assert skipped == [k for k in range(24) if k % 3 == 2]


def test_c1_case_differences_all_3n():
    kept, skipped = snh.coordinate_set(gat.case_spec(HSOH), _steps())
    assert kept == list(range(12)) and skipped == []


def test_named_out_of_plane_coordinate_refused():
    with pytest.raises(SystemExit, match="point group changes from Cs"):
        snh.coordinate_set(gat.case_spec(ACROLEIN), _steps(), snh.parse_coords("1x,0z", 8))


def test_diatomic_keeps_every_coordinate_by_redetection():
    # The test is gradient_at's -- re-detection -- so N2 keeps all six, as
    # nevpt2_numgrad.py probes all six: a displaced homonuclear diatomic is
    # D_inf_h again, in a moved frame.
    kept, skipped = snh.coordinate_set(gat.case_spec(N2), _steps())
    assert kept == list(range(6)) and skipped == []


def test_no_keeping_coordinate_refused(monkeypatch):
    spec = gat.case_spec(N2)
    ref = gat.reference_geometry(spec)
    real = ng.point_group
    monkeypatch.setattr(ng, "point_group",
                        lambda geom, basis: real(geom, basis) if geom == ref else ("C1", "C1"))
    with pytest.raises(SystemExit, match="nothing to difference"):
        snh.coordinate_set(spec, _steps())


def test_parse_coords():
    assert snh.parse_coords("auto", 8) is None
    assert snh.parse_coords("all", 2) == list(range(6))
    assert snh.parse_coords("3y,0x,3y", 8) == [0, 10]


def test_points_and_labels():
    spec = gat.case_spec(HSOH)
    pts = snh.points(spec, [4], (7, 11))
    assert [p["label"] for p in pts] == ["ref"] + [f"1y{s}{e}" for e in range(7, 12)
                                                   for s in "+-"]
    ref = gat.coords_bohr(pts[0]["geom"])
    d = gat.coords_bohr(pts[1]["geom"]) - ref
    assert abs(d[1, 1] - 2.0**-7) < 1e-12 and np.abs(np.delete(d.ravel(), 4)).max() == 0.0


# --- the cache, and a refusal through a pool worker ----------------------------


def test_cache_round_trip_and_mismatch(tmp_path):
    spec = gat.case_spec(HSOH)
    task = snh.points(spec, [0], (7, 11))[1]
    path = tmp_path / "p.npz"
    rec = dict(case=HSOH, scheme="sc", rev="abc", label=task["label"],
               geom=gat.coords_bohr(task["geom"]), card="cuda", grad=np.arange(12.0),
               energy=-1.5, ledger=dict(wall=np.float64(3.0), x=(1, 2), f=float("nan")))
    snh.save_point(path, rec)
    got = snh.load_point(path, task, HSOH, "sc", "abc")
    np.testing.assert_array_equal(got["grad"], np.arange(12.0))
    assert got["ledger"]["wall"] == 3.0 and got["ledger"]["x"] == [1, 2]
    assert snh.load_point(tmp_path / "none.npz", task, HSOH, "sc", "abc") is None
    with pytest.raises(SystemExit, match="rev"):
        snh.load_point(path, task, HSOH, "sc", "other")
    with pytest.raises(SystemExit, match="scheme"):
        snh.load_point(path, task, HSOH, "pc", "abc")
    other = snh.points(spec, [0], (7, 11))[2]
    with pytest.raises(SystemExit, match="label"):
        snh.load_point(path, other, HSOH, "sc", "abc")


def test_a_refused_point_is_cached_as_a_refusal(tmp_path):
    # An out-of-plane acrolein point: gradient_at refuses it before any SCF,
    # and the worker caches the refusal instead of a gradient.
    geom = ng.displace(gat.reference_geometry(gat.case_spec(ACROLEIN)), 0, 2, 2.0**-7)
    task = dict(case=ACROLEIN, schemes=["sc", "pc"], rev="r", label="0z+7", geom=geom,
                caches=[str(tmp_path / s / "0z+7.npz") for s in ("sc", "pc")],
                work=str(tmp_path), binary="none", lock=None, card="cuda", workers=1)
    out = snh.run_point(task)
    assert "point group changes" in out["refused"]
    for s in ("sc", "pc"):
        got = snh.load_point(tmp_path / s / "0z+7.npz", task, ACROLEIN, s, "r")
        assert "point group changes" in str(got["refused"])
        assert "grad" not in got


def test_gradients_at_refuses_bad_schemes():
    ref = gat.reference_geometry(gat.case_spec(N2))
    for bad in ((), ("sc", "sc"), ("xc",)):
        with pytest.raises(SystemExit, match="distinct"):
            gat.gradients_at(ref, N2, schemes=bad)


def test_work_dir_inside_golden_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", "--case", HSOH, "--work-dir",
                                     str(gat.REPO / "golden" / "w"), "--dry-run"])
    with pytest.raises(SystemExit, match="inside"):
        snh.main()


# --- the sidecar ---------------------------------------------------------------


def _sidecar(kept=IN_PLANE):
    g, h = _pair(PLANAR)
    steps = np.array(snh.ladder_steps(LADDER))
    n = 12
    m = np.zeros(n, dtype=bool)
    m[kept] = True
    raw = np.full((n, n), np.nan)
    raw[:, m] = h[:, m]
    an = snh.analyse(raw, list(kept), g, PLANAR)
    fd = np.full((len(steps), n, n), np.nan)
    fd[:, :, m] = raw[:, m]
    col = np.where(m, 3.0, np.nan)
    header = dict(natm=4, nsteps=len(steps), ladder_run=3, energy=-12.5)
    fields = dict(case="planar", scheme="sc", reference="casscf",
                  symbols=" ".join(PLANAR_SYMBOLS), rev="abc",
                  ledger=json.dumps(dict(points={}, refused=[])))
    arrays = dict(geom=PLANAR, gradient=g, coords=m.astype(float), steps=steps,
                  hessian=an["hessian"], hessian_raw=raw, asymmetry=an["asymmetry"],
                  richardson_residual=np.where(m[None, :], 1e-9, np.nan) * np.ones((n, 1)),
                  fd=fd, column_run=col, column_end=col)
    return header, fields, arrays


@pytest.mark.parametrize("kept", [IN_PLANE, list(range(12))])
def test_hessgold_round_trip(tmp_path, kept):
    header, fields, arrays = _sidecar(kept)
    p = tmp_path / "x_casscf.hessgold"
    gf.write_hessgold(p, header=header, fields=fields, arrays=arrays)
    h, f, a = gf.read_hessgold(p)
    assert h == header and f == fields
    assert list(a) == [n for n, _ in gf.HESSGOLD_ARRAYS]
    for k, v in arrays.items():
        np.testing.assert_array_equal(a[k], v, err_msg=k)
    q = tmp_path / "y.hessgold"
    gf.write_hessgold(q, header=h, fields=f, arrays=a)
    assert q.read_bytes() == p.read_bytes()
    assert p.read_bytes()[:8] == b"NEVPT2H1"


def test_hessgold_refuses_a_zero_where_nothing_was_computed(tmp_path):
    header, fields, arrays = _sidecar()
    arrays["hessian_raw"][:, 2] = 0.0
    with pytest.raises(gf.FormatError, match="never filled"):
        gf.write_hessgold(tmp_path / "x.hessgold", header=header, fields=fields, arrays=arrays)


def test_hessgold_refuses_a_nan_where_something_was(tmp_path):
    header, fields, arrays = _sidecar()
    arrays["hessian"][0, 1] = np.nan
    with pytest.raises(gf.FormatError, match="not finite"):
        gf.write_hessgold(tmp_path / "x.hessgold", header=header, fields=fields, arrays=arrays)


@pytest.mark.parametrize(("what", "edit", "msg"), [
    ("scheme", lambda h, f, a: f.update(scheme="xx"), "scheme"),
    ("ledger", lambda h, f, a: f.update(ledger="[1]"), "JSON object"),
    ("symbols", lambda h, f, a: f.update(symbols="O C"), "symbols"),
    ("steps", lambda h, f, a: a.update(steps=a["steps"][::-1].copy()), "decreasing"),
    ("coords", lambda h, f, a: a.update(coords=a["coords"] * 2), "mask"),
    ("shape", lambda h, f, a: a.update(gradient=a["gradient"][:6]), "expected"),
])
def test_hessgold_refusals(tmp_path, what, edit, msg):
    header, fields, arrays = _sidecar()
    edit(header, fields, arrays)
    with pytest.raises(gf.FormatError, match=msg):
        gf.write_hessgold(tmp_path / f"{what}.hessgold", header=header, fields=fields,
                          arrays=arrays)


def test_hessgold_names():
    assert gf.hessgold_name(ACROLEIN, "sc") == f"{ACROLEIN}_casscf.hessgold"
    assert gf.hessgold_name(ACROLEIN, "pc") == f"{ACROLEIN}_casscf_pc.hessgold"
    assert gf.hessgold_case(f"golden/{ACROLEIN}_casscf_pc.hessgold") == (ACROLEIN, "pc")
    assert gf.hessgold_case(f"{HSOH}_casscf.hessgold") == (HSOH, "sc")
    with pytest.raises(gf.FormatError, match="named"):
        gf.hessgold_case("x.hessgold")
