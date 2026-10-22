"""Hand cases for ``casscf_third_derivative.py``: level 2's CASSCF-branch
term ``T(u, v) = Z . d^2 g_all / dtheta dtheta [u, v]`` (docs/hessian-theory.md,
section 7), on a bent C1 water CASSCF(4,4) in 6-31G (seconds), with a seeded
``(Z, z)`` -- ``T`` is linear in it, so the gradient's own multipliers add
nothing a seeded one does not exercise (the committed cases with the
SC-NEVPT2 multipliers are docs/reference-data.md, "Level 2's CASSCF
third-derivative term").

Each piece is held to ``gen_g_hop``, which it does not call to compute
``T``: the series' lower coefficients to ``g_all`` and ``h_op`` far from
stationarity, ``T`` to a ladder of ``g_all`` differenced along both
directions. Tolerances are written out here, not read from the module.
"""

from __future__ import annotations

import casscf_hessian as ch
import casscf_third_derivative as ctd
import nevpt2_grad as ngr
import numpy as np
import pytest
from pyscf import mcscf, scf

GEOM = [("O", (0.0, 0.05, 0.11)), ("H", (0.02, 0.76, -0.47)), ("H", (-0.03, -0.77, -0.45))]
BASIS = "6-31g"
NCORE, NCAS, NELECAS = 3, 4, (2, 2)

LADDER_TOL = 1e-6  # ctd.LADDER_TOL, written out on purpose
ROUND_OFF = 1e-10  # ctd.SYMMETRY_TOL / IDENTITY_TOL, written out on purpose


@pytest.fixture(scope="module")
def water():
    """The converged CASSCF(4,4), and the same orbitals and CI vector moved
    off it along a seeded step (a point with ``|g_all|`` of order one)."""
    mol = ch.mol_at(GEOM, BASIS)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, NCAS, NELECAS)
    mc.conv_tol = 1e-12
    mc.fcisolver.conv_tol = 1e-14
    mc.run()
    th = ctd.Third(mol, mc.mo_coeff, mc.ci, NCORE, NCAS, NELECAS)
    step = ctd.direction(th, "full", np.random.default_rng(5))
    off = ctd.Third(mol, *th.at(0.1 * step), NCORE, NCAS, NELECAS)
    return dict(th=th, off=off, e=float(mc.e_tot), Z=ctd.seeded_multipliers(th))


def test_lower_coefficients_are_gen_g_hops(water):
    """At the stationary point the constant term is the CASSCF energy; at a
    point far from it the ``a`` and ``a b`` coefficients are still
    ``g_all . u`` and ``u . h_op(v)`` (``h_op`` is the local Hessian
    anywhere), and the ``a s`` coefficient is the **Jacobian** of ``g_all``
    -- which off stationarity is not ``h_op``."""
    th, off = water["th"], water["off"]
    assert abs(th.series(*ctd.pair(th, "full", 0), water["Z"])[0] - water["e"]) < 1e-9
    g, h_op = off.g_hop()
    assert np.linalg.norm(g) > 0.1
    Z = ctd.seeded_multipliers(off)
    miss = []
    for kind in ctd.KINDS:
        u, v = ctd.pair(off, kind, 0)
        s = off.series(u, v, Z)
        assert abs(s[1] - g @ u) < ROUND_OFF
        assert abs(s[3] - u @ h_op(v)) < ROUND_OFF
        e = 1e-4
        jac = (off.g_all(e * u) - off.g_all(-e * u)) / (2 * e)
        assert abs(s[5] - Z @ jac) < 1e-7
        miss.append(abs(s[5] - Z @ h_op(u)))
    # orbital, ci, mixed, full: h_op is the Jacobian on the CI block only
    assert miss[1] < ROUND_OFF
    assert min(miss[0], miss[2], miss[3]) > 1e-4


@pytest.mark.parametrize("kind", ctd.KINDS)
def test_t_matches_the_g_all_ladder(water, kind):
    th, Z = water["th"], water["Z"]
    for seed in ctd.DIRECTION_SEEDS:
        u, v = ctd.pair(th, kind, seed)
        t = th.contract(u, v, Z)
        chk = ctd.ladder(th, u, v, Z, t, kind)
        assert chk.ok, chk.summary
        assert chk.numbers["err"] <= LADDER_TOL
        assert chk.numbers["run"] >= 3


def test_t_is_symmetric_and_bilinear(water):
    th, Z = water["th"], water["Z"]
    u, v = ctd.pair(th, "full", 1)
    _u2, w = ctd.pair(th, "mixed", 1)
    t = th.contract(u, v, Z)
    assert abs(t - th.contract(v, u, Z)) < ROUND_OFF
    assert abs(th.contract(u, v + 2.0 * w, Z) - t - 2.0 * th.contract(u, w, Z)) < ROUND_OFF
    assert abs(th.contract(u, v, 3.0 * Z) - 3.0 * t) < ROUND_OFF


def test_action_is_t(water):
    th, Z = water["th"], water["Z"]
    u, v = ctd.pair(th, "full", 2)
    w = th.action(u, Z)
    assert w.shape == (th.nvar + th.ndet,)
    assert abs(w @ v - th.contract(u, v, Z)) < ROUND_OFF


@pytest.mark.parametrize("kind", ["orbital", "full"])
def test_ladder_rejects_the_global_third_derivative(water, kind):
    """The check can fail: the symmetric ``d^3 E[u, v, Z]`` in one global
    chart (``Z`` not stepped from the displaced point) differs from ``T`` by
    the connection, and the ladder says so."""
    th, Z = water["th"], water["Z"]
    u, v = ctd.pair(th, kind, 0)
    wrong = th.global_third(u, v, Z)
    assert abs(wrong - th.contract(u, v, Z)) > 1e-4
    chk = ctd.ladder(th, u, v, Z, wrong, kind)
    assert not chk.ok


def test_h_op_differenced_is_not_the_oracle(water):
    """The module docstring's claim: ``Z . h_op(theta + h v)[u]`` differenced
    along ``v`` converges to ``T`` on a CI pair and misses it on an orbital
    one."""
    th, Z = water["th"], water["Z"]
    steps = ngr.ladder_steps((8, 10))
    for kind, hit in (("ci", True), ("orbital", False)):
        u, v = ctd.pair(th, kind, 0)
        q = ctd.hop_ladder(th, u, v, Z, steps)
        miss = abs((4.0 * q[-1] - q[-2]) / 3.0 - th.contract(u, v, Z))
        assert (miss < 1e-7) == hit, (kind, miss)


def test_committed_n2_with_seeded_multipliers():
    """The script end to end on the committed N2 CAS(4,4) CASSCF state,
    without the NEVPT2 multipliers."""
    assert ctd.main(["--case", "n2_ccpvdz_cas44", "--seeded-multipliers"]) == 0
