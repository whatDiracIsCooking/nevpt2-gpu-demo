"""Hand cases for ``casscf_hessian.py``: the CASSCF reference's skeleton
nuclear derivatives along ``nevpt2_grad.frozen_coeff``'s path, piece by
piece, on a bent C1 water in STO-3G (seven AOs, a fraction of a second).

The state is a CASCI(4,4) on RHF orbitals -- deliberately **not** a CASSCF
stationary point, so the connection's first-order term (the generalized
Fock against ``t``) does not vanish and every term of ``L_RR`` is exercised.
Each check is against central differences of the path itself, re-evaluated
from scratch at displaced geometries; the committed-golden runs are
docs/reference-data.md, "The CASSCF skeleton nuclear derivatives".

Tolerances are written out here, not read from the module.
"""

from __future__ import annotations

import casscf_hessian as ch
import nevpt2_grad as ngr
import nevpt2_numgrad as ngrad
import numpy as np
import pytest
from pyscf import mcscf, scf

GEOM = [("O", (0.0, 0.05, 0.11)), ("H", (0.02, 0.76, -0.47)), ("H", (-0.03, -0.77, -0.45))]
BASIS = "sto-3g"
H1 = 1e-4  # first differences: O(h^2) ~ 1e-8 truncation
H2 = 2e-3  # second differences: round-off ~ 1e-14 / h^2


@pytest.fixture(scope="module")
def sk():
    mol = ch.mol_at(GEOM, BASIS)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, 4, 4).run()
    D, G = ch.full_rdms(mc.ncore, mc.ncas, mc.nelecas, mc.ci)
    ref = ch.Reference(GEOM, BASIS, mc.mo_coeff.copy(), mc.ncore, mc.ncas, D, G,
                       float(mc.e_tot))
    return ch.Skeleton(ref)


def _at(sk, k, s):
    return ngrad.displace_vec(GEOM, s * ch.unit(sk.natm, k))


def _energy(sk, geom):
    return ch.path_energy(sk.ref, geom, sk.S0_half)


def test_densities_rebuild_the_energy(sk):
    assert ch.check_reference_energy(sk) < 1e-10


def test_rdm2_has_the_integrals_eightfold_symmetry(sk):
    G = sk.ref.G
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        assert np.abs(G - G.transpose(perm)).max() < 1e-14


def test_state_is_not_stationary(sk):
    """The point of the hand case: ``F`` is not symmetric, so the first-order
    connection term carries weight."""
    o = sk.occ
    asym = sk.F[o, :] - sk.F[o, :].T
    assert np.abs(asym).max() > 1e-3


def test_overlap_derivatives(sk):
    S1, S2 = sk.ovlp1(), sk.ovlp2()
    for k in range(sk.ncoord):
        Sp = ch.mol_at(_at(sk, k, H1), BASIS).intor("int1e_ovlp")
        Sm = ch.mol_at(_at(sk, k, -H1), BASIS).intor("int1e_ovlp")
        assert np.abs((Sp - Sm) / (2 * H1) - S1[k]).max() < 1e-7
        assert np.abs(S2[k] - S2[:, k].transpose(0, 2, 1)).max() < 1e-14
    # a mixed second derivative, two atoms
    k, m = 1, 5

    def S(a, b):
        d = a * ch.unit(sk.natm, k) + b * ch.unit(sk.natm, m)
        return ch.mol_at(ngrad.displace_vec(GEOM, d), BASIS).intor("int1e_ovlp")
    fd = (S(H2, H2) - S(H2, -H2) - S(-H2, H2) + S(-H2, -H2)) / (4 * H2**2)
    assert np.abs(fd - S2[k, m]).max() < 1e-5


def test_connection_is_the_frozen_paths(sk):
    """``t^k``, ``t^kk`` against differences of ``C0^T S0 C(R) - 1`` with
    ``C(R)`` from ``nevpt2_grad.frozen_coeff`` itself."""
    t1, t2 = sk.connection()
    for k in range(sk.ncoord):
        def t(s, k=k):
            S = ch.mol_at(_at(sk, k, s), BASIS).intor("int1e_ovlp")
            C = ngr.frozen_coeff(sk.ref.C0, sk.S0_half, S)
            return sk.ref.C0.T @ sk.S0 @ C - np.eye(sk.nmo)
        tp, tm = t(H1), t(-H1)
        assert np.abs((tp - tm) / (2 * H1) - t1[k]).max() < 1e-7
        assert np.abs((tp + tm) / H1**2 - t2[k, k]).max() < 1e-5
    # the first-order connection is S-antisymmetric's partner: t + t^T = -dS_MO
    s1 = np.einsum("mp,kmn,nq->kpq", sk.ref.C0, sk.ovlp1(), sk.ref.C0)
    assert np.abs(t1 + t1.transpose(0, 2, 1) + s1).max() < 1e-12


def test_perturbed_hamiltonian_is_the_mo_integrals_derivative(sk):
    """``h^(k)`` and ``(pq|rs)^(k)`` over mixed ranges against the MO
    integrals rebuilt along the path."""
    from pyscf import ao2mo

    k = 4
    occ, allmo = sk.occ, slice(None)
    ranges = (allmo, occ, slice(2, 6), allmo)

    def mo(s):
        mol = ch.mol_at(_at(sk, k, s), BASIS)
        C = ngr.frozen_coeff(sk.ref.C0, sk.S0_half, mol.intor("int1e_ovlp"))
        h = C.T @ scf.hf.get_hcore(mol) @ C
        Cs = [C[:, r] for r in ranges]
        g = ao2mo.general(mol, Cs, compact=False).reshape([c.shape[1] for c in Cs])
        return h, g

    (hp, gp), (hm, gm) = mo(H1), mo(-H1)
    assert np.abs((hp - hm) / (2 * H1) - sk.perturbed_h1(k)).max() < 1e-7
    assert np.abs((gp - gm) / (2 * H1) - sk.perturbed_eri(k, ranges)).max() < 1e-7


def test_gradient_against_the_path(sk):
    de = sk.gradient()
    for k in range(sk.ncoord):
        fd = (_energy(sk, _at(sk, k, H1)) - _energy(sk, _at(sk, k, -H1))) / (2 * H1)
        assert abs(fd - de.flat[k]) < 1e-7


@pytest.fixture(scope="module")
def hess(sk):
    return sk.hessian()


def test_hessian_against_the_path(sk, hess):
    """Every diagonal element by a second difference, and a mixed pair on
    two different atoms by the four-point stencil."""
    L, _terms = hess
    e0 = _energy(sk, GEOM)

    def richardson(fd):
        return (4 * fd(H2) - fd(2 * H2)) / 3

    for k in range(sk.ncoord):
        def diag(h, k=k):
            return (_energy(sk, _at(sk, k, h)) - 2 * e0 + _energy(sk, _at(sk, k, -h))) / h**2
        assert abs(richardson(diag) - L[k, k]) < 1e-7
    for k, m in ((0, 4), (2, 7), (3, 8)):
        def e(a, b, k=k, m=m):
            d = a * ch.unit(sk.natm, k) + b * ch.unit(sk.natm, m)
            return _energy(sk, ngrad.displace_vec(GEOM, d))

        def mixed(h, e=e):
            return (e(h, h) - e(h, -h) - e(-h, h) + e(-h, -h)) / (4 * h**2)
        assert abs(richardson(mixed) - L[k, m]) < 1e-7


def test_hessian_is_the_gradients_derivative(sk, hess):
    """A column of ``L`` against a central difference of the analytic
    gradient: the path is consistent (re-anchoring ``frozen_coeff`` at a point
    on it reproduces it), so the gradient at ``C(R)`` is the path's slope."""
    L, _terms = hess
    k = 5

    def grad(s):
        geom = _at(sk, k, s)
        S = ch.mol_at(geom, BASIS).intor("int1e_ovlp")
        C = ngr.frozen_coeff(sk.ref.C0, sk.S0_half, S)
        r = sk.ref
        return ch.Skeleton(ch.Reference(geom, BASIS, C, r.ncore, r.ncas, r.D, r.G)).gradient()

    fd = ((grad(H1) - grad(-H1)) / (2 * H1)).ravel()
    assert np.abs(fd - L[:, k]).max() < 1e-6


def test_sum_rules_and_symmetry(sk, hess):
    L, terms = hess
    rules = ch.sum_rules(sk.gradient(), L, sk.natm)
    assert rules["grad_translation"] < 1e-10
    assert rules["hess_translation"] < 1e-9
    assert rules["hess_symmetry"] < 1e-12
    # every term is symmetric on its own
    for name, v in terms.items():
        assert np.abs(v - v.T).max() < 1e-10, name


def test_judge_reads_a_symmetry_zero_and_refuses_a_wrong_one():
    steps = ngr.ladder_steps((1, 9))
    flat = [1e-12 * (-1) ** i for i in range(len(steps))]
    assert ch.judge("zero", flat, steps, 1e-15).ok
    # a nonzero analytic value against a flat ladder is not a symmetry zero
    assert not ch.judge("wrong", flat, steps, 1e-6).ok
    # an ordinary ladder: c h^2 truncation to a known limit
    good = [2.5 + 0.3 * h**2 for h in steps]
    assert ch.judge("good", good, steps, 2.5).ok
    assert not ch.judge("off", good, steps, 2.5 + 5e-6).ok


def test_incomplete_mo_basis_is_refused(sk):
    r = sk.ref
    bad = ch.Reference(GEOM, BASIS, r.C0[:, :-1], r.ncore, r.ncas, r.D, r.G)
    with pytest.raises(SystemExit, match="complete MO basis"):
        ch.Skeleton(bad)
