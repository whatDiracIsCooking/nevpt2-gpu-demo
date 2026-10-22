"""Hand cases for ``integral_perturbation.py`` and the ``.pertgold`` format it
writes (#220): the format round-trips and refuses a sidecar missing any one
array; the inventory is linear in ``(h, (pq|rs))`` and the symmetrization
idempotent; and the analytic perturbed integrals -- conventional, and the
FITTED ones, whose skeleton derivative (three-centre integrals on the moving
atom, orbital and auxiliary, and the metric's) is this module's own -- are the
derivative of the integrals rebuilt from scratch along the frozen path, on a
bent C1 water in 6-31G (thirteen AOs, seconds).

The committed sidecars are the layout contract's (``golden_layout_check.py``)
and the generator's own checks' (its ladders, before it writes).

Tolerances are written out here, not read from the module.
"""

from __future__ import annotations

import casscf_hessian as ch
import golden_format as gf
import integral_perturbation as ip
import nevpt2_grad as ngr
import nevpt2_numgrad as ngrad
import numpy as np
import pytest
from pyscf import scf
from pyscf.grad import rhf as rhf_grad

GEOM = [("O", (0.0, 0.05, 0.11)), ("H", (0.02, 0.76, -0.47)), ("H", (-0.03, -0.77, -0.45))]
BASIS = "6-31g"
AUXBASIS = "weigend"
NCORE, NACT = 3, 2


def _header(natm=1, nao=6, ncore=1, nact=2, nvirt=3, naux=0) -> dict:
    return dict(natm=natm, nao=nao, nmo=ncore + nact + nvirt, ncore=ncore, nact=nact,
                nvirt=nvirt, naux=naux)


def _arrays(h: dict) -> dict[str, np.ndarray]:
    """Every array of the format at the shapes ``h`` gives it, random."""
    rng = np.random.default_rng(220)
    dims = dict(h, ncoord=3 * h["natm"])
    return {n: rng.standard_normal(tuple(dims[d] for d in s)) for n, s in gf.PERTGOLD_ARRAYS}


def test_pertgold_round_trip_and_rewrite(tmp_path):
    h = _header(naux=7)
    a = _arrays(h)
    p = tmp_path / "x.pertgold"
    gf.write_pertgold(p, header=h, arrays=a)
    hb, ab = gf.read_pertgold(p)
    assert hb == h
    assert list(ab) == [n for n, _s in gf.PERTGOLD_ARRAYS]
    for n, v in a.items():
        np.testing.assert_array_equal(ab[n], v, err_msg=n)
    q = tmp_path / "y.pertgold"
    gf.write_pertgold(q, header=hb, arrays=ab)
    assert q.read_bytes() == p.read_bytes()
    assert p.read_bytes()[:8] == gf.PERTGOLD_MAGIC


@pytest.mark.parametrize("missing", [n for n, _s in gf.PERTGOLD_ARRAYS])
def test_pertgold_missing_array_is_refused_whole(tmp_path, missing):
    h = _header()
    a = _arrays(h)
    del a[missing]
    with pytest.raises(gf.FormatError):
        gf.write_pertgold(tmp_path / "w.pertgold", header=h, arrays=a)
    # And a file written past the writer (the array section alone, short one
    # array) is refused by the reader, naming it.
    p = tmp_path / "r.pertgold"
    with open(p, "wb") as f:
        f.write(gf.pack_header(gf.PERTGOLD_HEADERS, gf.PERTGOLD_MAGIC, h))
        gf.write_arrays(f, a)
    with pytest.raises(gf.FormatError, match=f"missing {missing}"):
        gf.read_pertgold(p)


def test_pertgold_refuses_a_misshapen_array_and_a_bad_partition(tmp_path):
    h = _header()
    a = _arrays(h)
    a["g_xv_cv"] = a["g_xv_cv"][:, :, :, :, :-1]
    with pytest.raises(gf.FormatError, match="g_xv_cv"):
        gf.write_pertgold(tmp_path / "s.pertgold", header=h, arrays=a)
    with pytest.raises(gf.FormatError, match="partition"):
        gf.write_pertgold(tmp_path / "p.pertgold", header=dict(h, nmo=h["nmo"] + 1),
                          arrays=_arrays(h))


def test_inventory_is_linear_and_symmetrized_idempotent():
    rng = np.random.default_rng(7)
    n, gamma = NCORE + NACT + 4, rng.standard_normal((NACT, NACT))
    g1 = ip.symmetrized(rng.standard_normal((n,) * 4))
    g2 = ip.symmetrized(rng.standard_normal((n,) * 4))
    h1 = rng.standard_normal((n, n))
    h2 = rng.standard_normal((n, n))
    np.testing.assert_allclose(ip.symmetrized(g1), g1, atol=1e-15)
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)):
        np.testing.assert_allclose(g1.transpose(perm), g1, atol=1e-15)
    a, b = 0.7, -1.3
    mix = ip.inventory(a * h1 + b * h2, a * g1 + b * g2, gamma, NCORE, NACT)
    one = ip.inventory(h1, g1, gamma, NCORE, NACT)
    two = ip.inventory(h2, g2, gamma, NCORE, NACT)
    assert list(mix) == list(ip.NAMES)
    for k in mix:
        np.testing.assert_allclose(mix[k], a * one[k] + b * two[k], atol=1e-12, err_msg=k)


@pytest.fixture(scope="module")
def water():
    mol = ch.mol_at(GEOM, BASIS)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    ci = np.random.default_rng(3).standard_normal((2, 2))
    ci /= np.linalg.norm(ci)
    D, G = ch.full_rdms(NCORE, NACT, (1, 1), ci)
    return ch.Reference(GEOM, BASIS, np.asarray(mf.mo_coeff).copy(), NCORE, NACT, D, G)


@pytest.mark.parametrize("auxbasis", [None, AUXBASIS])
def test_perturbed_integrals_are_the_frozen_path_derivative(water, auxbasis):
    """Analytic ``(pq|rs)^(k)`` and ``h^(k)`` against a Richardson-extrapolated
    central difference of the integrals rebuilt at displaced geometries,
    orbitals on the frozen path -- every coordinate, conventional and fitted."""
    sk = ch.Skeleton(water) if auxbasis is None else ip.DfSkeleton(water, auxbasis)
    allmo = slice(None)
    t1, _ = sk.connection(second=False)
    h1 = sk.hcore1()

    def integrals(dvec):
        mol = ch.mol_at(ngrad.displace_vec(GEOM, dvec), BASIS)
        C = ngr.frozen_coeff(water.C0, sk.S0_half, mol.intor("int1e_ovlp"))
        return C.T @ scf.hf.get_hcore(mol) @ C, ip.full_eri(mol, C, auxbasis)

    step = 1e-3
    for ia in range(sk.natm):
        skel = sk.eri1_skel(ia, allmo, allmo, allmo, allmo)
        for x in range(3):
            k = 3 * ia + x
            hk = sk.perturbed_h1(k, t1, h1)
            gk = sk.perturbed_eri(k, (allmo,) * 4, t1, skel[x])
            quot = []
            for s in (step, step / 2):
                hp, gp = integrals(s * ch.unit(sk.natm, k))
                hm, gm = integrals(-s * ch.unit(sk.natm, k))
                quot.append(((hp - hm) / (2 * s), (gp - gm) / (2 * s)))
            hr = (4 * quot[1][0] - quot[0][0]) / 3
            gr = (4 * quot[1][1] - quot[0][1]) / 3
            assert np.abs(hr - hk).max() < 1e-8, (k, np.abs(hr - hk).max())
            assert np.abs(gr - gk).max() < 1e-8, (k, np.abs(gr - gk).max())


def test_inventory_adjoint_is_the_transpose():
    """#247: ``inventory_adjoint`` is :func:`inventory`'s transpose -- a seeded
    ``ibar`` paired with the inventory of a seeded ``(h, g)`` equals ``(Hbar,
    Gbar)`` paired with ``(h, g)`` -- and the symmetrized densities pair the
    same against a symmetric ``(h, g)``."""
    rng = np.random.default_rng(247)
    n = NCORE + NACT + 4
    gamma = rng.standard_normal((NACT, NACT))
    gamma = gamma + gamma.T
    shapes = {k: v.shape for k, v in ip.inventory(np.zeros((n, n)), np.zeros((n,) * 4),
                                                   gamma, NCORE, NACT).items()}
    ibar = {k: rng.standard_normal(s) for k, s in shapes.items()}
    Hbar, Gbar = ip.inventory_adjoint(ibar, gamma, NCORE, NACT)
    for sym in (False, True):
        h = rng.standard_normal((n, n))
        g = rng.standard_normal((n,) * 4)
        if sym:
            h, g = h + h.T, ip.symmetrized(g)
        inv = ip.inventory(h, g, gamma, NCORE, NACT)
        lhs = sum(float(np.sum(ibar[k] * inv[k])) for k in ip.NAMES)
        rhs = float(np.sum(Hbar * h) + np.sum(Gbar * g))
        assert abs(lhs - rhs) < 1e-10 * max(1.0, abs(lhs)), (sym, lhs, rhs)
        if sym:
            P, Gamma = ip.adjoint_densities(Hbar, Gbar)
            dens = float(np.sum(P * h) + 0.5 * np.sum(Gamma * g))
            assert abs(lhs - dens) < 1e-10 * max(1.0, abs(lhs))


@pytest.mark.parametrize("auxbasis", [None, AUXBASIS])
def test_path_gradient_is_the_skeleton_gradient(water, auxbasis):
    """#247: ``path_gradient``'s adjoint-side derivative equals #215's
    forward one (``Skeleton.gradient`` at the same densities over the whole
    MO range, ``E_nuc`` taken off) for seeded full-range densities --
    conventional and fitted -- and obeys the translational sum rule."""
    rng = np.random.default_rng(2047)
    n = water.C0.shape[1]
    P = rng.standard_normal((n, n))
    P = P + P.T
    Gamma = ip.symmetrized(rng.standard_normal((n,) * 4))
    full = ch.Reference(GEOM, BASIS, water.C0, 0, n, P, Gamma)
    fwd_sk = ch.Skeleton(full) if auxbasis is None else ip.DfSkeleton(full, auxbasis)
    fwd = fwd_sk.gradient().ravel() - np.asarray(rhf_grad.grad_nuc(fwd_sk.mol)).ravel()
    sk = ch.Skeleton(water) if auxbasis is None else ip.DfSkeleton(water, auxbasis)
    adj = ip.path_gradient(sk, P, Gamma)
    scale = max(1.0, float(np.abs(fwd).max()))
    assert np.abs(adj - fwd).max() < 1e-10 * scale, np.abs(adj - fwd).max()
    # a rigid translation moves no overlap, so the frozen path's integrals are
    # invariant along it: the sum rule, exact
    assert np.abs(adj.reshape(-1, 3).sum(axis=0)).max() < 1e-10 * scale
