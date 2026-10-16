"""Hand cases for ``nevpt2_numgrad.py --reference casscf``'s own logic: the
continuity guard's verdict on its three signals, the overlap signals' gauge
invariance and CI convention, the reference-kind refusal both ways, the
flag combinations ``main`` refuses, and the displacement helpers. Synthetic
inputs only -- no SCF, no CASSCF.

As in ``test_refusal_guards.py``, the thresholds are written out here as
numbers, NOT read from the module's constants: a loosened guard is exactly
the regression these exist to catch, so moving a threshold means editing this
file too, in the same change, saying why (docs/reference-data.md, "The
continuity guard's thresholds", has the measurements behind them).
"""

from __future__ import annotations

import sys

import nevpt2_numgrad as ng
import numpy as np
import pytest

CONTINUITY_E_TOL = 3e-5
CONTINUITY_ACT_TOL = 1e-4
CONTINUITY_CI_TOL = 1e-5
CONV_TOL_GRAD = 1e-7  # generate_golden.CASSCF_CONV_TOL_GRAD


def test_threshold_values():
    assert ng.CONTINUITY_E_TOL == CONTINUITY_E_TOL
    assert ng.CONTINUITY_ACT_TOL == CONTINUITY_ACT_TOL
    assert ng.CONTINUITY_CI_TOL == CONTINUITY_CI_TOL
    assert ng.gg.CASSCF_CONV_TOL_GRAD == CONV_TOL_GRAD


def _sig(**kw):
    sig = dict(e_jump=1e-7, converged=True, g_opt=4e-9, act_defect=1e-7, ci_defect=1e-8)
    sig.update(kw)
    return sig


def test_clean_state_trips_nothing():
    assert ng.continuity_tripped(_sig()) == []


@pytest.mark.parametrize(("kw", "tripped"), [
    (dict(e_jump=CONTINUITY_E_TOL * 0.99), []),
    (dict(e_jump=CONTINUITY_E_TOL * 1.01), ["energy"]),
    (dict(g_opt=CONV_TOL_GRAD * 0.99), []),
    (dict(g_opt=CONV_TOL_GRAD * 1.01), ["gradient"]),
    (dict(converged=False), ["gradient"]),
    (dict(act_defect=CONTINUITY_ACT_TOL * 0.99), []),
    (dict(act_defect=CONTINUITY_ACT_TOL * 1.01), ["overlap"]),
    (dict(ci_defect=CONTINUITY_CI_TOL * 0.99), []),
    (dict(ci_defect=CONTINUITY_CI_TOL * 1.01), ["overlap"]),
    (dict(e_jump=1.0, converged=False, act_defect=1.0), ["energy", "gradient", "overlap"]),
])
def test_each_signal_at_its_threshold(kw, tripped):
    assert ng.continuity_tripped(_sig(**kw)) == tripped


@pytest.mark.parametrize("key", ["e_jump", "g_opt", "act_defect", "ci_defect"])
def test_unevaluable_signal_trips(key):
    assert ng.continuity_tripped(_sig(**{key: float("nan")}))


def test_check_continuity_names_every_refused_state():
    states = [
        dict(label="0x+0", tripped=[], signals=_sig()),
        dict(label="0z+0", tripped=["energy"], signals=_sig(e_jump=1.0)),
        dict(label="1z-2", refused="the --mo-from gate"),
    ]
    with pytest.raises(SystemExit, match=r"refuses 2 displaced state\(s\) of 3") as exc:
        ng._check_continuity(states)
    assert "0z+0: tripped energy" in str(exc.value)
    assert "1z-2: refused while building it" in str(exc.value)
    assert "0x+0" not in str(exc.value)
    ng._check_continuity(states[:1])  # a clean sweep passes


# --- overlap_signals --------------------------------------------------------
#
# An orthonormal AO basis (S = 1), six orbitals, ncore = 1, ncas = 3 with
# (1, 1) electrons: small enough for fci.addons.transform_ci by hand.
NMO, NCORE, NCAS, NELEC = 6, 1, 3, (1, 1)


def _ci(seed):
    from pyscf import fci

    na = fci.cistring.num_strings(NCAS, NELEC[0])
    v = np.random.default_rng(seed).standard_normal((na, na))
    return v / np.linalg.norm(v)


def _rotation(seed, n):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))
    return q


def test_overlap_identical_state_is_zero():
    mo, ci = np.eye(NMO), _ci(1)
    act, cid = ng.overlap_signals(mo, ci, np.eye(NMO), mo, ci, NCORE, NCAS, NELEC)
    assert act == pytest.approx(0.0, abs=1e-14)
    assert cid == pytest.approx(0.0, abs=1e-14)


def test_overlap_is_blind_to_the_active_gauge():
    """A rotation among the active orbitals, with the CI vector carried along
    (new orbitals = old @ u), is the same state: both signals stay at
    round-off. This is also what pins transform_ci's convention -- the wrong
    one moves the CI signal by O(1) (checked below)."""
    from pyscf.fci import addons

    mo, ci = np.eye(NMO), _ci(2)
    u = _rotation(3, NCAS)
    mo_s = mo.copy()
    mo_s[:, NCORE:NCORE + NCAS] = mo[:, NCORE:NCORE + NCAS] @ u
    ci_s = addons.transform_ci(ci, NELEC, u)  # the same state, in the rotated basis
    act, cid = ng.overlap_signals(mo_s, ci_s, np.eye(NMO), mo, ci, NCORE, NCAS, NELEC)
    assert act < 1e-14
    assert cid < 1e-12
    wrong = addons.transform_ci(ci, NELEC, u.T)
    _, cid_wrong = ng.overlap_signals(mo_s, wrong, np.eye(NMO), mo, ci, NCORE, NCAS, NELEC)
    assert cid_wrong > 1e-2


def test_overlap_sees_a_swapped_active_orbital():
    mo, ci = np.eye(NMO), _ci(4)
    swap = list(range(NMO))
    swap[NCORE + NCAS - 1], swap[NCORE + NCAS] = swap[NCORE + NCAS], swap[NCORE + NCAS - 1]
    act, _ = ng.overlap_signals(mo[:, swap], ci, np.eye(NMO), mo, ci, NCORE, NCAS, NELEC)
    assert act == pytest.approx(1.0, abs=1e-12)
    assert act > CONTINUITY_ACT_TOL


def test_overlap_sees_a_root_flip():
    mo, ci = np.eye(NMO), _ci(5)
    other = _ci(6)
    other -= np.vdot(ci, other) * ci
    other /= np.linalg.norm(other)
    _, cid = ng.overlap_signals(mo, other, np.eye(NMO), mo, ci, NCORE, NCAS, NELEC)
    assert cid == pytest.approx(1.0, abs=1e-12)


def test_carry_orbitals_is_orthonormal_in_the_new_metric():
    rng = np.random.default_rng(7)
    a = rng.standard_normal((NMO, NMO))
    s = np.eye(NMO) + 0.01 * (a + a.T)
    c = ng.carry_orbitals(np.eye(NMO), s)
    assert np.abs(c.T @ s @ c - np.eye(NMO)).max() < 1e-13


# --- the reference kind, both ways -------------------------------------------

def test_reference_kind_casci_refuses_a_casscf_golden():
    with pytest.raises(SystemExit, match="Pass --reference casscf"):
        ng._check_reference_kind("casscf", "g.nevpt2gold", "casci")


def test_reference_kind_casscf_refuses_a_casci_golden():
    with pytest.raises(SystemExit, match="Pass --reference casci"):
        ng._check_reference_kind("casci", "g.nevpt2gold", "casscf")


@pytest.mark.parametrize("kind", ["casci", "casscf"])
def test_reference_kind_match_accepted(kind):
    ng._check_reference_kind(kind, "g.nevpt2gold", kind)


# --- the flags main() refuses, before any solve -------------------------------

@pytest.mark.parametrize(("argv", "msg"), [
    (["--reference", "casscf", "--pc", "--npz", "x.npz"], "SC only"),
    (["--reference", "casscf"], "writes no sidecar"),
    (["--reference", "casscf", "--directional", "4", "--no-write"], "needs --analytic"),
    (["--directional", "4"], "belong to --reference casscf"),
    (["--workers", "4", "--warm-probe", "2"], "belong to --reference casscf"),
])
def test_refused_flag_combinations(argv, msg, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["nevpt2_numgrad.py", *argv])
    with pytest.raises(SystemExit) as exc:
        ng.main()
    assert exc.value.code == 2
    assert msg in capsys.readouterr().err


# --- displacement helpers ----------------------------------------------------

GEOM = [("O", (0.1, -0.2, 0.0)), ("C", (1.3, 0.4, 0.0)), ("H", (-0.5, 0.9, 0.0))]


@pytest.mark.parametrize(("iatom", "comp", "s"), [(0, 0, 0.0025), (2, 1, -0.000625)])
def test_displace_vec_is_displace_to_the_bit(iatom, comp, s):
    d = np.zeros((3, 3))
    d[iatom, comp] = 1.0
    assert ng.displace_vec(GEOM, s * d) == ng.displace(GEOM, iatom, comp, s)


def test_geom_spec_round_trips_exactly():
    moved = ng.displace(GEOM, 1, 0, 0.0025)
    parsed = [(t.split()[0], tuple(float(x) for x in t.split()[1:]))
              for t in ng.geom_spec(moved).split("; ")]
    assert parsed == moved


def test_directions_are_unit_and_in_plane():
    probed = [(0, 0), (0, 1), (1, 0), (2, 1)]
    dirs = ng.directions(probed, 3, 4, 124)
    assert np.allclose(np.linalg.norm(dirs.reshape(4, -1), axis=1), 1.0)
    mask = np.zeros((3, 3), bool)
    for i, c in probed:
        mask[i, c] = True
    assert not dirs[:, ~mask].any()
    assert np.array_equal(dirs, ng.directions(probed, 3, 4, 124))  # seeded
