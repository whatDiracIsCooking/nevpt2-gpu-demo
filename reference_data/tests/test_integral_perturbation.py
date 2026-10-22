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
