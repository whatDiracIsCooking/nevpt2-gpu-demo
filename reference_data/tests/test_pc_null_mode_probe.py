"""Hand cases for ``pc_null_mode_probe.py``'s algebra: the S-cut restricted
inverse's first and second derivatives, and the second-order lift, on a
synthetic Gram family built to have exactly the structure the PC classes have
(docs/pc-nevpt2.md, "What the cut drops, and how it moves"):

* ``S(c) = Phi(c)^T Phi(c)`` with ``Phi`` linear in ``c`` and fewer rows than
  columns, so the rank is the same at every ``c`` while the null space
  rotates -- CAS(4,4)'s Sr in miniature;
* ``K(c)`` annihilates S's null space at the base point ``c0`` only (an
  eigenstate), through a term that vanishes there;
* ``b = S x + S_1 h1`` with ``S_1 = Phi^T chi``, so ``b`` stays in S's range.

Synthetic inputs only: no golden, no PySCF solve, no card.
"""

from __future__ import annotations

import numpy as np
import pc_null_mode_probe as probe
import pytest

NROW, NCOL, NC = 4, 7, 3  # Phi is (NROW, NCOL): rank 4, three null modes
DELTA = 1.5


class Family:
    def __init__(self, seed: int, deficient: bool = False):
        rng = np.random.default_rng(seed)
        self.O = rng.standard_normal((NC, NROW, NCOL))
        if deficient:  # Phi(c0) = O_0 of rank NROW - 1: one more null mode at c0 only
            self.O[0] = rng.standard_normal((NROW, NROW - 1)) @ rng.standard_normal(
                (NROW - 1, NCOL))
        self.R = 0.3 * rng.standard_normal((NC, NROW, NCOL))
        h = rng.standard_normal((NROW, NROW))
        self.H = h @ h.T + 3 * np.eye(NROW)
        self.G = 0.2 * rng.standard_normal((NROW, NROW))
        self.chi = rng.standard_normal((NC, NROW))
        self.x = rng.standard_normal(NCOL)
        self.h1 = rng.standard_normal(1)
        self.c0 = np.eye(NC)[0]

    def mats(self, c):
        phi = np.einsum("k,knd->nd", c, self.O)
        psi = np.einsum("k,knd->nd", c - self.c0, self.R)  # zero at c0
        S = phi.T @ phi
        K = phi.T @ self.H @ phi + phi.T @ self.G @ psi + psi.T @ self.G.T @ phi
        S1 = (phi.T @ np.einsum("k,kn->n", c, self.chi))[:, None]
        return S, K, S1

    def coeffs(self, u, v, at=None):
        """The quadratic polynomial's coefficients along ``at + s u + t v``
        (``at`` defaults to c0)."""
        base = self.c0 if at is None else at

        def f(s, t):
            return self.mats(base + s * u + t * v)
        f0, fp0, fm0 = f(0, 0), f(1, 0), f(-1, 0)
        f0p, f0m = f(0, 1), f(0, -1)
        fpp, fpm, fmp, fmm = f(1, 1), f(1, -1), f(-1, 1), f(-1, -1)
        out = {}
        for i, name in enumerate(("S", "K", "S1")):
            out[name] = f0[i]
            out[name + "u"] = (fp0[i] - fm0[i]) / 2
            out[name + "v"] = (f0p[i] - f0m[i]) / 2
            out[name + "uu"] = (fp0[i] + fm0[i]) / 2 - f0[i]
            out[name + "uv"] = (fpp[i] - fpm[i] - fmp[i] + fmm[i]) / 4
        return out

    def energy(self, c, nd):
        S, K, S1 = self.mats(c)
        _, _, Vk, _ = probe.split(S, nd)
        M = probe.restricted_inverse(probe.sym(K) + DELTA * probe.sym(S), Vk)
        b = S @ self.x + S1 @ self.h1
        return b @ M @ b

    def edot(self, c, v, nd, exact):
        S, K, S1 = (probe.sym(m) if i < 2 else m for i, m in enumerate(self.mats(c)))
        cf = self.coeffs(np.zeros(NC), v, at=c)
        s, Vn, Vk, _ = probe.split(S, nd)
        A, Av = K + DELTA * S, cf["Kv"] + DELTA * cf["Sv"]
        M = probe.restricted_inverse(A, Vk)
        b = S @ self.x + S1 @ self.h1
        bv = cf["Sv"] @ self.x + cf["S1v"] @ self.h1
        kw = dict(Sp=probe.kept_pinv(s, Vk), Sv=cf["Sv"], Vn=Vn) if exact else {}
        return probe.edot(b, bv, A, Av, M, **kw)


def _dirs():
    return np.array([0.0, 0.8, -0.6]), np.array([0.0, 0.3, 0.95])


def _fd(f, pows=(8, 9, 10)):
    fd = [(f(2.0**-p) - f(-(2.0**-p))) / (2 * 2.0**-p) for p in pows]
    r1 = [(4 * fd[i + 1] - fd[i]) / 3 for i in range(len(fd) - 1)]
    return (16 * r1[1] - r1[0]) / 15


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rank_constant_and_null_space_rotates(seed):
    fam = Family(seed)
    u, v = _dirs()
    cf = fam.coeffs(u, v)
    s, Vn, Vk, nd = probe.split(cf["S"])
    assert nd == NCOL - NROW
    for h in (0.3, 0.05):
        assert probe.split(fam.mats(fam.c0 + h * u)[0])[3] == nd
    Sp = probe.kept_pinv(s, Vk)
    # constant rank: the second-order lift vanishes; rotation: S_u couples
    lift = probe.lift_matrix(cf["Su"], cf["Sv"], probe.sym(cf["Suv"]), Sp, Vn)
    assert np.linalg.norm(lift) < 1e-10 * np.linalg.norm(cf["Suv"])
    assert np.linalg.norm(Vn.T @ cf["Su"] @ Vk) > 1e-2 * np.linalg.norm(cf["Su"])


def test_second_order_lift_sees_a_rank_rise():
    fam = Family(4, deficient=True)
    u, v = _dirs()
    cf = fam.coeffs(u, v)
    s, Vn, Vk, nd = probe.split(cf["S"])
    assert nd == NCOL - NROW + 1  # one more than generic, at c0 only
    Sp = probe.kept_pinv(s, Vk)
    luu = probe.lift_matrix(cf["Su"], cf["Su"], 2 * cf["Suu"], Sp, Vn) / 2
    mu = np.linalg.eigvalsh(probe.sym(luu)).max()
    h = 1e-3
    e = np.linalg.eigvalsh(fam.mats(fam.c0 + h * u)[0])
    assert mu > 1e-6 * s[-1]
    assert e[nd - 1] == pytest.approx(mu * h * h, rel=1e-2)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_first_derivative_needs_the_third_term_off_the_eigenstate(seed):
    fam = Family(seed)
    u, v = _dirs()
    nd = NCOL - NROW
    for c, needs in ((fam.c0, False), (fam.c0 + 0.2 * u, True)):
        ref = _fd(lambda h, c=c: fam.energy(c + h * v, nd))
        exact, device = fam.edot(c, v, nd, True), fam.edot(c, v, nd, False)
        assert exact == pytest.approx(ref, rel=1e-8)
        if needs:
            assert abs(device - ref) > 1e-4 * abs(ref)
        else:
            assert device == pytest.approx(ref, rel=1e-8)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_second_derivative_needs_the_rotation_terms(seed):
    fam = Family(seed)
    u, v = _dirs()
    nd = NCOL - NROW
    cf = fam.coeffs(u, v)
    S, K = probe.sym(cf["S"]), probe.sym(cf["K"])
    s, Vn, Vk, _ = probe.split(S, nd)
    A = K + DELTA * S
    M = probe.restricted_inverse(A, Vk)
    x, h1 = fam.x, fam.h1
    b = S @ x + cf["S1"] @ h1
    bu, bv = cf["Su"] @ x + cf["S1u"] @ h1, cf["Sv"] @ x + cf["S1v"] @ h1
    buv = cf["Suv"] @ x + cf["S1uv"] @ h1
    Au, Av = cf["Ku"] + DELTA * cf["Su"], cf["Kv"] + DELTA * cf["Sv"]
    Auv = probe.sym(cf["Kuv"]) + DELTA * probe.sym(cf["Suv"])
    base, rot = probe.eddot(b, bu, bv, buv, A, probe.sym(Au), probe.sym(Av), Auv, M,
                            probe.kept_pinv(s, Vk), probe.sym(cf["Su"]),
                            probe.sym(cf["Sv"]), Vn)
    ref = _fd(lambda h: fam.edot(fam.c0 + h * u, v, nd, True))
    assert base + rot == pytest.approx(ref, rel=1e-7)
    assert abs(base - ref) > 1e-3 * abs(ref)
    # ...and the device's Edot, differenced, converges to neither
    dev = _fd(lambda h: fam.edot(fam.c0 + h * u, v, nd, False))
    assert abs(dev - ref) > 1e-4 * abs(ref)
