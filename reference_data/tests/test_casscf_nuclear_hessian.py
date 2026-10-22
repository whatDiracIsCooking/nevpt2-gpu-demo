"""Hand cases for ``casscf_nuclear_hessian.py``: the analytic CASSCF nuclear
Hessian and its ``3N`` parameter responses, piece by piece, on a bent C1
water CASSCF(4,4) in 6-31G (thirteen AOs, seconds), plus the point-group test
on the committed geometries and the committed response sidecars' layout.

Each piece is held to something it does not compute itself: the gradient
formula to ``gen_g_hop``'s ``g_all`` at a non-stationary point, the
right-hand sides to central differences of ``g_all`` along the frozen path,
the responses to re-optimised states, the Hessian to ``pyscf.grad.casscf``.
The committed-golden runs are docs/reference-data.md, "The analytic CASSCF
nuclear Hessian".

Tolerances are written out here, not read from the module.
"""

from __future__ import annotations

import casscf_hessian as ch
import casscf_nuclear_hessian as cnh
import generate_golden as gg
import golden_format as gf
import nevpt2_grad as ngr
import nevpt2_numgrad as ngrad
import numpy as np
import pytest
import scipy.linalg
from pyscf import fci, mcscf, scf

GEOM = [("O", (0.0, 0.05, 0.11)), ("H", (0.02, 0.76, -0.47)), ("H", (-0.03, -0.77, -0.45))]
BASIS = "6-31g"
NCORE = 3  # ten electrons, four of them active


def _state(C, ci, mol, e) -> cnh.State:
    D, G = ch.full_rdms(NCORE, 4, (2, 2), ci)
    ref = ch.Reference(GEOM, BASIS, C.copy(), NCORE, 4, D, G, e)
    return cnh.State(case="water", ref=ref, ci=np.asarray(ci).copy(), nelecas=(2, 2),
                     e_casscf=e, mol_sym=mol)


@pytest.fixture(scope="module")
def water():
    """The CASSCF(4,4), polished to ``|g_all|`` <= 1e-11 by the module's own
    chord, and its responses."""
    mol = ch.mol_at(GEOM, BASIS)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.conv_tol = 1e-12
    mc.fcisolver.conv_tol = 1e-14
    mc.run()
    st0 = _state(np.asarray(mc.mo_coeff), np.asarray(mc.ci), mol, float(mc.e_tot))
    sk0 = ch.Skeleton(st0.ref)
    resp0 = cnh.solve_responses(sk0, st0, log=lambda *_: None)
    st, info = cnh.polished(st0, sk0, resp0.chord)
    assert info["converged"]
    sk = ch.Skeleton(st.ref)
    resp = cnh.solve_responses(sk, st, log=lambda *_: None)
    L, _ = sk.hessian()
    return dict(st=st, sk=sk, resp=resp, H=cnh.assemble(L, resp), de=sk.gradient(), mol=mol)


def test_energy_gradient_is_gen_g_hops_off_stationarity(water):
    """``dE/dTheta`` from the module's formula equals ``gen_g_hop``'s
    ``g_all`` at a point that is NOT stationary (orbitals rotated, CI vector
    moved), where both are far from zero."""
    st, sk = water["st"], water["sk"]
    rng = np.random.default_rng(3)
    mc = cnh.casscf_object(sk.mol, 4, (2, 2))
    var = cnh.variables(mc, sk.nmo)
    C = st.ref.C0 @ scipy.linalg.expm(cnh.unpack_kappa(var, 0.05 * rng.standard_normal(len(var)),
                                                       sk.nmo))
    ci = st.ci + 0.05 * rng.standard_normal(st.ci.shape)
    ci /= np.linalg.norm(ci)
    D, G = ch.full_rdms(NCORE, 4, (2, 2), ci)
    sk1 = ch.Skeleton(ch.Reference(GEOM, BASIS, C, NCORE, 4, D, G))
    g_all, _ = cnh.g_hop(mc, C, ci)
    mine = cnh.energy_gradient(sk1, var, ci, (2, 2), sk1.h_mo,
                               sk1.mo_eri(slice(None), sk1.occ, sk1.occ, sk1.occ))
    assert np.abs(g_all).max() > 1e-2
    assert np.abs(mine - g_all).max() < 1e-12


def test_right_hand_sides_are_g_alls_frozen_path_derivative(water):
    """``L_{Theta R_k}`` against the central difference of ``g_all`` at
    ``(C_f(R0 +- h e_k), c0)``, every coordinate."""
    st, sk, resp = water["st"], water["sk"], water["resp"]
    h = 1e-4

    def g_at(k, s):
        geom = ngrad.displace_vec(GEOM, s * h * ch.unit(sk.natm, k))
        mol = ch.mol_at(geom, BASIS)
        Cf = ngr.frozen_coeff(st.ref.C0, sk.S0_half, mol.intor("int1e_ovlp"))
        return cnh.g_hop(cnh.casscf_object(mol, 4, (2, 2)), Cf, st.ci)[0]

    for k in range(sk.ncoord):
        fd = (g_at(k, 1) - g_at(k, -1)) / (2 * h)
        assert np.abs(fd - resp.rhs[k]).max() < 1e-7, k


def test_exact_identities(water):
    H, resp, sk = water["H"], water["resp"], water["sk"]
    ids = cnh.identities(H, sk.natm)
    assert ids["symmetry"] < 1e-12
    assert ids["translation"] < 1e-10
    assert resp.info["residual"] < 1e-12
    assert resp.info["residual_hop"] < 1e-12
    # the responses carry no component along c, and K is antisymmetric with
    # zero redundant blocks
    assert np.abs(resp.ci @ water["st"].ci.ravel()).max() < 1e-14
    assert np.abs(resp.kappa + resp.kappa.transpose(0, 2, 1)).max() == 0.0
    for b in (slice(0, NCORE), slice(NCORE, NCORE + 4), slice(NCORE + 4, sk.nmo)):
        assert np.abs(resp.kappa[:, b, b]).max() == 0.0
    # at a stationary state the rotational rule holds to round-off
    assert cnh.rotation_rule(H, water["de"], sk.mol.atom_coords()) < 1e-9


def test_rotation_rule_catches_a_wrong_column(water):
    """The rotational rule must be able to fail: one off-diagonal pair of
    ``H`` moved by 1e-6 trips it."""
    H = water["H"].copy()
    H[0, 4] += 1e-6
    H[4, 0] += 1e-6
    assert cnh.rotation_rule(H, water["de"], water["sk"].mol.atom_coords()) > 1e-7


@pytest.mark.parametrize("k", [1, 5])
def test_responses_and_hessian_against_reoptimised_states(water, k):
    """A short ladder along one coordinate: the gauge-fixed parameters of
    re-optimised states, and ``pyscf.grad.casscf`` there, Richardson-
    extrapolated from ``h`` and ``h/2``."""
    st, sk, resp, H = water["st"], water["sk"], water["resp"], water["H"]
    d = ch.unit(sk.natm, k)

    def at(h):
        return cnh.reoptimise(st.ref, st.ci, (2, 2), ngrad.displace_vec(GEOM, h * d),
                              sk.S0_half, resp.chord)

    def quotient(h, key):
        p, m = at(h), at(-h)
        return (p[key] - m[key]) / (2 * h)

    # h^4 remainder ~2e-9 here; the states' |g_all| <= 1e-11 over 2h ~1e-8
    h = 2.0 ** -8
    for key, ref in (("kappa", resp.kappa[k]), ("ci", resp.ci[k]), ("grad", H[:, k])):
        a, b = quotient(h, key), quotient(h / 2, key)
        rich = (4 * b - a) / 3
        assert np.abs(rich - ref).max() < 1e-7, key
        # and the plain quotient is O(h^2) off: the comparison is not vacuous
        assert np.abs(a - ref).max() > np.abs(rich - ref).max()


def test_gauge_fixed_parameters_recover_a_known_rotation():
    """``U = exp(K) W`` with ``K`` off the diagonal blocks and ``W``
    block-diagonal: the decomposition returns ``K``, and the CI vector it
    carries through ``W``'s active block is the one the rotated active
    orbitals need (the active Hamiltonian's expectation is unchanged)."""
    rng = np.random.default_rng(5)
    nmo, nc, na = 9, 2, 4
    blocks = [slice(0, nc), slice(nc, nc + na), slice(nc + na, nmo)]
    K = 0.05 * rng.standard_normal((nmo, nmo))
    K = K - K.T
    A = 0.3 * rng.standard_normal((nmo, nmo))
    A = A - A.T
    Wgen = np.zeros_like(A)
    for b in blocks:
        K[b, b] = 0.0
        Wgen[b, b] = A[b, b]
    W = scipy.linalg.expm(Wgen)
    U = scipy.linalg.expm(K) @ W
    c0 = rng.standard_normal((6, 6))
    c0 /= np.linalg.norm(c0)
    # c' is c0 written in the orbitals rotated by W's active block
    a = blocks[1]
    cp = fci.addons.transform_ci(c0, (2, 2), W[a, a])
    Kg, c, _it = cnh.gauge_fixed_parameters(U, nc, na, cp, (2, 2), c0)
    assert np.abs(Kg - K).max() < 1e-12
    assert np.abs(c - c0.ravel()).max() < 1e-12


def test_point_group_keeping_directions():
    """N2: only the stretch (a single-atom z breaks inversion, though the
    re-detected group would still be D_inf_h); LiF: z and the stretch;
    acrolein: the in-plane coordinates; HSOH (C1): all twelve."""
    def dirs(case, geom=None):
        spec = ngrad.CASES[case]
        geom = geom or ngrad.read_xyz(ngrad.REPO / spec["geometry"])
        mol = gg._build_mol(geom, spec["basis"])
        st = cnh.State(case=case, ref=ch.Reference(geom, spec["basis"], None, 0, 0, None, None),
                       ci=np.zeros(1), nelecas=(0, 0), e_casscf=0.0, mol_sym=mol)
        return cnh.kept_directions(st), mol

    got, _ = dirs("n2_ccpvdz_cas44")
    assert list(got) == ["stretch"]
    got, _ = dirs("lif_def2svp_cas46")
    assert list(got) == ["e2", "e5", "stretch"]
    got, mol = dirs("acrolein_def2svp_cas65")
    assert len(got) == 16
    assert all(not cnh.keeps_point_group(mol, ch.unit(8, 3 * i + 2)) for i in range(8))
    got, _ = dirs("hsoh_def2svp_cas66")
    assert list(got) == [f"e{k}" for k in range(12)]


@pytest.mark.parametrize("case", ["n2_ccpvdz_cas44", "hsoh_def2svp_cas66"])
def test_committed_response_sidecar(case, tmp_path):
    """The committed ``.respgold`` reads back, re-writes byte for byte, is in
    its golden's frame, and its Hessian and responses are what the module's
    solve gives (the same floating-point path; 1e-10 is headroom)."""
    path = ngrad.REPO / "golden" / f"{case}_casscf.respgold"
    header, arrays = gf.read_respgold(path)
    out = tmp_path / path.name
    gf.write_respgold(out, header=header, arrays=arrays)
    assert out.read_bytes() == path.read_bytes()
    st = cnh.state_from_case(case)
    assert cnh.check_golden_frame(st) == []
    assert np.array_equal(arrays["ci"], st.ci) and np.array_equal(arrays["mo_coeff"], st.ref.C0)
    sk = ch.Skeleton(st.ref)
    resp = cnh.solve_responses(sk, st, log=lambda *_: None)
    L, _ = sk.hessian()
    H = cnh.assemble(L, resp)
    assert np.abs(arrays["hessian"] - H).max() < 1e-10
    assert np.abs(arrays["kappa"] - resp.kappa).max() < 1e-10
    assert np.abs(arrays["ci_response"] - resp.ci).max() < 1e-10
    assert np.array_equal(arrays["kept"], cnh.kept_mask(st))


def test_respgold_refuses_a_misshapen_array(tmp_path):
    header, arrays = gf.read_respgold(ngrad.REPO / "golden" / "n2_ccpvdz_cas44_casscf.respgold")
    bad = dict(arrays, kappa=arrays["kappa"][:-1])
    with pytest.raises(gf.FormatError, match="kappa"):
        gf.write_respgold(tmp_path / "x.respgold", header=header, arrays=bad)
    missing = {k: v for k, v in arrays.items() if k != "kept"}
    with pytest.raises(gf.FormatError, match="arrays"):
        gf.write_respgold(tmp_path / "y.respgold", header=header, arrays=missing)
