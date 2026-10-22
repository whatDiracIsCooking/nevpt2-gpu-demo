"""Hand cases for ``casscf_constraint_mixed.py``: the CASSCF constraint's
R-mixed terms ``Z . d^2 g / dtheta dR`` and ``Z . d^2 g / dR^2``
(docs/hessian-theory.md, sections 7 and 8), on a bent C1 water CASSCF(4,4)
in 6-31G with a seeded ``(Z, z)`` (both terms are linear in it; the
committed cases with the SC-NEVPT2 multipliers are docs/reference-data.md,
"The CASSCF constraint's R-mixed terms").

The state is the converged CASSCF moved off stationarity along a seeded
step, so ``|g_all|`` is of order one and nothing can lean on stationarity.
Each piece is held to something it does not compute with: the densities to
``gen_g_hop`` and #216's series, ``Z . dg/dR`` to #219's formula, the mixed
term to #216's series at #215's perturbed Hamiltonian, and both terms to
ladders of ``gen_g_hop``'s ``g_all`` along the frozen path. Tolerances are
written out here, not read from the module.
"""

from __future__ import annotations

import casscf_constraint_mixed as ccm
import casscf_hessian as ch
import casscf_third_derivative as ctd
import numpy as np
import pytest
from pyscf import mcscf, scf

GEOM = [("O", (0.0, 0.05, 0.11)), ("H", (0.02, 0.76, -0.47)), ("H", (-0.03, -0.77, -0.45))]
BASIS = "6-31g"
NCORE, NCAS, NELECAS = 3, 4, (2, 2)

LADDER_TOL = 1e-6  # ccm.LADDER_TOL, written out on purpose
ROUND_OFF = 1e-10  # the exact identities (ccm.IDENTITY_TOL is 1e-9)


@pytest.fixture(scope="module")
def water():
    """The constraint terms at a point a seeded step off the converged
    CASSCF(4,4), with a seeded unit ``(Z, z)``."""
    mol = ch.mol_at(GEOM, BASIS)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, NCAS, NELECAS)
    mc.conv_tol = 1e-12
    mc.fcisolver.conv_tol = 1e-14
    mc.run()
    th = ctd.Third(mol, mc.mo_coeff, mc.ci, NCORE, NCAS, NELECAS)
    step = ctd.direction(th, "full", np.random.default_rng(5))
    C, ci = th.at(0.1 * step)
    con = ccm.Constraint(GEOM, BASIS, C, ci, NCORE, NCAS, NELECAS)
    Z = ctd.seeded_multipliers(con.th)
    M, terms = con.second(Z)
    us = ccm.theta_directions(con)
    return dict(con=con, Z=Z, M=M, terms=terms, us=us,
                mixed={k: con.mixed(u, Z) for k, u in us.items()},
                zgrad=con.z_gradient(Z))


def test_state_is_not_stationary(water):
    assert np.linalg.norm(water["con"].th.g_hop()[0]) > 0.1


def test_densities_rebuild_e_g_all_and_the_jacobian(water):
    """Mask 0 rebuilds the energy, mask 4 ``Z . g_all`` (``gen_g_hop``) and
    mask 5 ``Z . J[u]`` (#216's ``a s`` coefficient), at the reference
    integrals, far from stationarity."""
    con, Z = water["con"], water["Z"]
    th = con.th
    u0 = water["us"]["full/0"]
    e = float(th.series(u0, u0, Z)[0])
    dens = con.densities(Z)
    assert abs(con.contract(*dens[0]) + th.e_nuc - e) < ROUND_OFF
    assert abs(con.contract(*dens[4]) - float(Z @ th.g_hop()[0])) < ROUND_OFF
    for u in water["us"].values():
        s5 = float(th.series(u, np.zeros_like(u), Z)[5])
        assert abs(con.contract(*con.densities(Z, u)[5]) - s5) < ROUND_OFF


def test_densities_have_the_integrals_symmetry(water):
    con, Z = water["con"], water["Z"]
    P, G = con.densities(Z, water["us"]["full/1"])[5]
    assert np.abs(P - P.T).max() < 1e-14
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        assert np.abs(G - G.transpose(perm)).max() < 1e-14


def test_z_gradient_is_219s_formula(water):
    con, Z = water["con"], water["Z"]
    assert np.abs(water["zgrad"] - con.z_gradient_219(Z)).max() < ROUND_OFF


def test_mixed_is_the_series_at_the_perturbed_hamiltonian(water):
    """The mixed term two ways: #215's gradient driver at the ``a s``
    densities, and #216's series with #215's perturbed Hamiltonian swapped
    in for the reference integrals."""
    con, Z = water["con"], water["Z"]
    for name in ("orbital/0", "ci/0", "full/1"):
        u = water["us"][name]
        for k in (0, 4, 8):
            assert abs(con.mixed_by_series(u, Z, k) - water["mixed"][name][k]) < ROUND_OFF


def test_linear_in_z_and_u(water):
    con, Z = water["con"], water["Z"]
    u, w = water["us"]["orbital/1"], water["us"]["ci/1"]
    assert np.abs(con.mixed(u + 2.0 * w, Z) - water["mixed"]["orbital/1"]
                  - 2.0 * water["mixed"]["ci/1"]).max() < ROUND_OFF
    M3, _ = con.second(3.0 * Z)
    assert np.abs(M3 - 3.0 * water["M"]).max() < ROUND_OFF


def test_symmetry_and_translational_sum_rules(water):
    con, M = water["con"], water["M"]
    assert np.abs(M - M.T).max() < ROUND_OFF
    assert np.abs(M.reshape(con.natm, 3, con.natm, 3).sum(axis=2)).max() < 1e-9
    assert np.abs(water["zgrad"].reshape(con.natm, 3).sum(axis=0)).max() < ROUND_OFF
    for w in water["mixed"].values():
        assert np.abs(w.reshape(con.natm, 3).sum(axis=0)).max() < ROUND_OFF
    for name, v in water["terms"].items():
        assert np.abs(v - v.T).max() < ROUND_OFF, name


@pytest.fixture(scope="module")
def ladders(water):
    """Every ladder on two coordinates and one whole-molecule direction."""
    con = water["con"]
    dirs = {"e1": ch.unit(con.natm, 1), "e5": ch.unit(con.natm, 5),
            "d": ch.seeded_direction(con.natm, 232)}
    us = {k: water["us"][k] for k in ("orbital/0", "ci/0", "full/0")}
    return ccm.run_ladders(con, water["Z"], us, dirs, water["M"], water["mixed"], workers=1,
                           log=lambda *_a: None)


def test_every_ladder_passes(ladders):
    kinds = {"mixed": 0, "second": 0, "column": 0}
    for c in ladders:
        assert c.ok, (c.name, c.summary)
        assert c.numbers["err"] <= LADDER_TOL
        assert c.numbers["run"] >= 3
        kinds["mixed" if c.name.startswith("d2g/dth") else
              "column" if "column" in c.name else "second"] += 1
    assert kinds == {"mixed": 9, "second": 6, "column": 3}


def test_ladder_rejects_the_term_without_the_connection(water):
    """The check can fail: ``Z . d^2 g / dR^2`` with the frozen path's
    second-order connection term dropped misses the ladder."""
    con, Z = water["con"], water["Z"]
    wrong = water["M"] - water["terms"]["conn2"]
    d = ch.unit(con.natm, 1)
    assert abs(d.ravel() @ water["terms"]["conn2"] @ d.ravel()) > 1e-4
    checks = ccm.run_ladders(con, Z, {}, {"e1": d}, wrong, {}, workers=1, columns=False,
                             log=lambda *_a: None)
    assert len(checks) == 1 and not checks[0].ok


def test_ladder_rejects_the_global_chart_mixed_term(water):
    """And the mixed term without the multipliers' local step (``Z`` added
    in the global chart, #216's wrong operator) cannot stand in for it: the
    two differ, and the ladder takes the right one."""
    con, Z = water["con"], water["Z"]
    u = water["us"]["orbital/0"]
    k = 1
    # global chart: the a s coefficient of E at exp(K(a u + s Z)), at h^(k)
    th = ctd.Third.__new__(ctd.Third)
    th.__dict__.update(con.th.__dict__)
    th.h_mo = con.sk.perturbed_h1(k)
    th.eri_mo = con.sk.perturbed_eri(k, (slice(None),) * 4)
    th.e_nuc = 0.0
    wrong = float(th.energy(
        ctd.hd_expm({1: th.kappa(u), 4: th.kappa(Z)}, th.nmo),
        ctd.hd_normalise({0: th.ci.ravel(), 1: u[th.nvar:], 4: Z[th.nvar:]})).get(5, 0.0))
    right = water["mixed"]["orbital/0"][k]
    assert abs(wrong - right) > 1e-4
    w = np.zeros(con.ncoord)
    w[k] = wrong
    checks = ccm.run_ladders(con, Z, {"u": u}, {"e1": ch.unit(con.natm, k)}, water["M"],
                             {"u": w}, workers=1, columns=False, log=lambda *_a: None)
    mixed = [c for c in checks if c.name.startswith("d2g/dth")]
    assert len(mixed) == 1 and not mixed[0].ok


def test_committed_n2_with_seeded_multipliers():
    """The script end to end on the committed N2 CAS(4,4) CASSCF state,
    without the NEVPT2 multipliers or the column ladders."""
    assert ccm.main(["--case", "n2_ccpvdz_cas44", "--seeded-multipliers",
                     "--no-columns"]) == 0
