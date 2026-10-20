#!/usr/bin/env python3
r"""What PC-NEVPT2's singular-metric cut drops, how it moves with the CI
vector, and what that does to the PC energy's second derivative -- measured
offline on a committed golden, with PySCF on the CPU (no card, no build).

It rebuilds every PC class's metric ``S``, Dyall block ``K`` and one-body
column ``S_1`` from the golden's CI vector and integrals exactly the way the
device does (``src/energy/energy_pc_classes.cppm``; the table in
docs/pc-nevpt2.md, "Per class"), checks that the class energies reproduce the
golden's ``pc_class_energies``, and then exploits the one structural fact the
whole analysis rests on: **the RDM tuple is exactly quadratic in the CI
vector** (docs/hessian-theory.md, section 3), so along ``c + s u + t v`` every
class matrix is a polynomial in ``(s, t)`` whose coefficients are the RDMs'
symmetric bilinear forms, ``B(a, b) = (Q(a+b) - Q(a-b)) / 4``. No step size
enters any of the algebra below; only the reference derivative is a finite
difference.

The device's solve is the **S-cut restricted inverse**: with ``Q`` S's kept
eigenvectors (``s_k > PC_TAU s_max``), ``M = Q (Q^T A Q)^-1 Q^T``,
``A = K + Delta S``, ``E_t = -b^T M b``. That is not the Moore-Penrose
inverse of ``A`` once ``K`` stops annihilating S's dropped space, which it
does only at an eigenstate. Its derivatives therefore carry ``S^+`` (the
metric's pseudo-inverse), not ``A^+ A^+``:

  dM = -M A' M + P S' S^+ M + M S^+ S' P - M S^+ S' P A M - M A P S' S^+ M

with ``P`` the dropped-space projector. At the eigenstate (``P A = 0``) the
first derivative is the device's ``-(2 x.b' - x.A'.x)``; off it there is a
third term ``-2 rho.S'.P.A.x`` (``rho = S^+ x``), and the second derivative at
the eigenstate is

  F_uv = 2 b_uv.x - x.A_uv.x + 2 g_u.M.g_v                       (sec. 6.1's)
       + 2 rho.S_u.P.g_v - 2 rho.S_v.P.A_u.x + 2 rho.S_v.P.S_u.sigma (the rotation)

with ``g = b' - A' x`` and ``sigma = S^+ b`` (:func:`eddot`). The probe checks
both lines against a Richardson-extrapolated central difference of the exact
first derivative (:func:`edot` with ``exact=True``).

Four parts, each a table:

``energies``  the class energies against the golden's block2 PC fields, and
              S's spectrum (dropped, largest dropped, smallest kept, gap).
``structure`` which dropped modes are null for EVERY CI vector of a kind --
              the full fixed-Ms determinant space, the singlets, the
              singlets in the state's abelian irrep (``sector``), and the
              states sharing the Hamiltonian's full symmetry (``family``:
              ground states of ``h1 + b h1^2, a h2``) -- and each kind's
              per-vector nullity (constant rank with a rotating null space
              shows as per-vector nullity = the state's, common nullity
              smaller).
``order``     per direction kind (``Epq``: the device's pcTangentDirection,
              ``family``, ``sector``): the kept-dropped coupling, the
              second-order lift ``L_uv = P (S_uv - S_u S^+ S_v - S_v S^+ S_u) P``
              (the h^2 coefficient of the dropped eigenvalues; zero is
              constant rank to second order) and the order at which the
              dropped eigenvalues actually leave zero along ``c + h u``.
``hessian``   per direction kind: Eddot without and with the rotation terms
              against the reference difference, the difference of the
              DEVICE's Edot (what nevpt2_pc_tangent's ladder differences),
              and the amplification: smallest kept ``s``, the gap, the
              rotation terms' size, and how far a CI-convergence-sized
              perturbation of the state moves the result.

The measured numbers are in docs/pc-nevpt2.md, "What the cut drops, and how
it moves"; the theory they settle is docs/hessian-theory.md, section 6.2.

Usage::

    uv run python reference_data/pc_null_mode_probe.py golden/n2_ccpvdz_cas44.nevpt2gold
    uv run python reference_data/pc_null_mode_probe.py golden/n2_ccpvdz_cas88.nevpt2gold \
        --parts structure,order --kinds Epq,family

Threads default to 6 (``OMP_NUM_THREADS`` etc. are only set when absent).
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "6")

import golden_format as gf  # noqa: E402
import numpy as np  # noqa: E402
from pyscf import fci  # noqa: E402
from pyscf.fci import spin_op  # noqa: E402
from pyscf.mrpt import nevpt2 as pyscf_nevpt2  # noqa: E402

#: S's cut and the gap it must clear: ``src/energy/energy_finish.cppm``'s
#: PC_TAU and PC_MIN_GAP (docs/pc-nevpt2.md, "The singular-metric convention").
PC_TAU = 1e-13
PC_MIN_GAP = 1e3

#: The golden's ``pc_class_energies`` order.
GOLDEN_ORDER = ("Sr", "Si", "Sijrs", "Sijr", "Srsi", "Srs", "Sij", "Sir")
PC_CLASSES = ("Sr", "Si", "Sijr", "Srsi", "Srs", "Sij", "Sir")
#: The classes whose cut drops anything on a committed golden.
DROPPING = ("Sr", "Si", "Sir")
#: What the second-derivative check runs: the dropping classes, plus Srs as
#: the no-drop control (the rotation terms must then be exactly zero).
#: Sij/Sijr are affine in the RDMs, so a straight line is not their path
#: (docs/pc-nevpt2.md, "What the check can and cannot displace"), and Sijr /
#: Srsi drop nothing.
HESSIAN_CLASSES = ("Sr", "Si", "Sir", "Srs")
#: A null eigenvalue for the structural tests, relative to the largest.
NULL_REL = 1e-11
DIRECTION_KINDS = ("Epq", "family", "sector")


def sym(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)


# --- the state and its RDM tuple, a quadratic form in the CI vector -----------


class State:
    """A golden's CI vector and integrals, and the RDM tuple ``Q(v)`` (dm1,
    dm2, dm3 and the two 4-RDM digests f3ca/f3ac) of any vector ``v`` --
    quadratic, unnormalised: what PySCF's NEVPT2 builds at ``v = c``."""

    def __init__(self, path):
        self.hdr, self.a = gf.read_golden(path)
        self.n = self.hdr["ncas"]
        self.nelec = (self.hdr["nelec_a"], self.hdr["nelec_b"])
        self.c = self.a["ci"] / np.linalg.norm(self.a["ci"])
        # PySCF's (pq|rs) back from the golden's h2e (ppaa.transpose(0,2,1,3)).
        self.eri = np.ascontiguousarray(self.a["h2e"].transpose(0, 2, 1, 3))
        n, ne = self.n, self.nelec
        self.link = (fci.cistring.gen_linkstr_index(range(n), ne[0]),
                     fci.cistring.gen_linkstr_index(range(n), ne[1]))

    def Q(self, v: np.ndarray, f3: bool = True) -> dict:
        d1, d2, d3 = fci.rdm.make_dm123("FCI3pdm_kern_sf", v, v, self.n, self.nelec)
        D = {"1": d1, "2": d2, "3": d3}
        if f3:
            for key, kern in (("f3ca", "NEVPTkern_cedf_aedf"), ("f3ac", "NEVPTkern_aedf_ecdf")):
                D[key] = pyscf_nevpt2._contract4pdm(kern, self.eri, np.ascontiguousarray(v),
                                                    self.n, self.nelec, self.link)
        else:
            D["f3ca"] = D["f3ac"] = np.zeros((self.n,) * 6)
        return D

    def B(self, a: np.ndarray, b: np.ndarray, f3: bool = True) -> dict:
        p, m = self.Q(a + b, f3), self.Q(a - b, f3)
        return {k: (p[k] - m[k]) / 4 for k in p}


def comb(terms) -> dict:
    """``sum_i w_i D_i`` over RDM tuples."""
    out: dict = {}
    for w, D in terms:
        for k, v in D.items():
            out[k] = out.get(k, 0) + w * v
    return out


def along(Dc, Bcu, Bcv, Quu, Buv, Qvv, s: float, t: float) -> dict:
    """The RDM tuple at ``c + s u + t v``, exactly."""
    return comb([(1, Dc), (2 * s, Bcu), (2 * t, Bcv), (s * s, Quu), (2 * s * t, Buv),
                 (t * t, Qvv)])


# --- the class matrices, the device's transcription ---------------------------


def build(cls: str, D: dict, st: State):
    """``(S, K, S_1)`` of one class from an RDM tuple; affine in ``D``."""
    a, n, nel = st.a, st.n, st.nelec
    h1e, h2e = a["h1e"], a["h2e"]
    d1, d2, d3 = D["1"], D["2"], D["3"]
    eye = np.eye(n)
    if cls == "Sr":
        S = np.einsum("rpqbac->pqrabc", d3).reshape(n**3, n**3)
        K = pyscf_nevpt2.make_a16(h1e, h2e, D, None, n, nel).reshape(n**3, n**3)
        return S, K, np.einsum("rpqa->pqra", d2).reshape(n**3, n)
    if cls == "Si":
        dm3_h = 2 * np.einsum("abef,cd->abcdef", d2, eye) - d3.transpose(0, 1, 3, 2, 4, 5)
        dm2_h = 2 * np.einsum("ab,cd->abcd", d1, eye) - d2.transpose(0, 1, 3, 2)
        S = np.einsum("rpqbac->pqrabc", dm3_h).reshape(n**3, n**3)
        K = pyscf_nevpt2.make_a22(h1e, h2e, D, None, n, nel).reshape(n**3, n**3)
        return S, K, np.einsum("rpqa->pqra", dm2_h).reshape(n**3, n)
    if cls == "Sijr":
        hdm1 = pyscf_nevpt2.make_hdm1(d1)
        return hdm1, pyscf_nevpt2.make_a3(h1e, h2e, d1, d2, hdm1), None
    if cls == "Srsi":
        return d1, pyscf_nevpt2.make_k27(h1e, h2e, d1, d2), None
    if cls == "Srs":
        rm2, a7 = pyscf_nevpt2.make_a7(h1e, h2e, d1, d2, d3)
        return rm2.transpose(0, 1, 3, 2).reshape(n * n, n * n), a7.reshape(n * n, n * n), None
    if cls == "Sij":
        hdm1 = pyscf_nevpt2.make_hdm1(d1)
        hdm2 = pyscf_nevpt2.make_hdm2(d1, d2)
        hdm3 = pyscf_nevpt2.make_hdm3(d1, d2, d3, hdm1, hdm2)
        a9 = pyscf_nevpt2.make_a9(h1e, h2e, hdm1, hdm2, hdm3)
        return hdm2.reshape(n * n, n * n), a9.reshape(n * n, n * n), None
    if cls == "Sir":
        off = np.array([[2.0, -1.0], [-1.0, 0.0]])
        b22 = np.array([[0.0, 0.0], [0.0, 1.0]])
        S22 = (2 * np.einsum("pa,qb->pqab", eye, d1) - np.einsum("qbap->pqab", d2)
               + np.einsum("ab,qp->pqab", eye, d1))
        S = np.einsum("xy,qpab->xpqyab", off, d2) + np.einsum("xy,pqab->xpqyab", b22, S22)
        a12 = pyscf_nevpt2.make_a12(h1e, h2e, d1, d2, d3)
        a13 = pyscf_nevpt2.make_a13(h1e, h2e, d1, d2, d3)
        K = np.einsum("xy,pqab->xpqyab", off, a12) + np.einsum("xy,pqab->xpqyab", b22, a13)
        S1 = np.einsum("x,qp->xpq", np.array([2.0, -1.0]), d1)
        m = 2 * n * n
        return S.reshape(m, m), K.reshape(m, m), S1.reshape(m, 1)
    raise ValueError(cls)


def build_S(cls: str, D: dict, st: State) -> np.ndarray:
    return sym(build(cls, D, st)[0])


def tuples(cls: str, st: State) -> list:
    """``(weight, x1, x2, h1, Delta)`` per external tuple, so the class energy
    is ``-sum weight * b(x1)^T M b(x2)`` with ``b(x) = S x + S_1 h1`` (``x2``
    None: the same as ``x1``; only Sijr/Srsi's second coupling sets it)."""
    a = st.a
    ec, ev = a["e_core"], a["e_virt"]
    out = []
    if cls == "Sr":
        X, H = a["h2e_v_Sr"], a["h1e_v_Sr"]
        out = [(1.0, X[i].reshape(-1), None, H[i], ev[i]) for i in range(X.shape[0])]
    elif cls == "Si":
        X, H = a["h2e_v_Si"], a["h1e_v_Si"]
        out = [(1.0, np.einsum("qpr->pqr", X[:, :, i, :]).reshape(-1), None, H[:, i], -ec[i])
               for i in range(X.shape[2])]
    elif cls == "Sijr":  # (v, a, c, c): rpji, the second coupling swaps i and j
        X = a["h2e_v_Sijr"]
        for r, j, i in itertools.product(range(X.shape[0]), range(X.shape[2]), range(X.shape[3])):
            x, d = X[r, :, j, i], ev[r] - ec[j] - ec[i]
            out += [(2.0, x, None, None, d), (-1.0, x, X[r, :, i, j], None, d)]
    elif cls == "Srsi":  # (v, v, c, a): rsip, the second coupling swaps r and s
        X = a["h2e_v_Srsi"]
        for r, s, i in itertools.product(range(X.shape[0]), range(X.shape[1]), range(X.shape[2])):
            x, d = X[r, s, i], ev[r] + ev[s] - ec[i]
            out += [(2.0, x, None, None, d), (-1.0, x, X[s, r, i], None, d)]
    elif cls == "Srs":
        X = a["h2e_v_Srs"]
        out = [(0.5, X[r, s].T.reshape(-1), None, None, ev[r] + ev[s])
               for r in range(X.shape[0]) for s in range(X.shape[1])]
    elif cls == "Sij":
        X = a["h2e_v_Sij"]
        out = [(0.5, X[:, :, i, j].T.reshape(-1), None, None, -(ec[i] + ec[j]))
               for i in range(X.shape[2]) for j in range(X.shape[3])]
    elif cls == "Sir":
        X1, X2, H = a["h2e_v1_Sir"], a["h2e_v2_Sir"], a["h1e_v_Sir"]
        out = [(1.0, np.concatenate([X1[r, :, i, :].reshape(-1), X2[r, :, :, i].reshape(-1)]),
                None, np.array([H[r, i]]), ev[r] - ec[i])
               for r in range(X1.shape[0]) for i in range(X1.shape[2])]
    return out


# --- the S-cut solve and its derivatives: pure algebra on matrices ------------


def split(S: np.ndarray, nd: int | None = None):
    """S's eigenvalues (ascending), dropped and kept eigenvectors, and the
    dropped count (by PC_TAU unless ``nd`` fixes it)."""
    s, V = np.linalg.eigh(sym(S))
    if nd is None:
        nd = int(np.sum(s <= PC_TAU * s[-1]))
    return s, V[:, :nd], V[:, nd:], nd


def restricted_inverse(A: np.ndarray, Vk: np.ndarray) -> np.ndarray:
    """``M = Q (Q^T A Q)^-1 Q^T``: the device's solve, Q = S's kept space."""
    return Vk @ np.linalg.solve(Vk.T @ A @ Vk, Vk.T)


def kept_pinv(s: np.ndarray, Vk: np.ndarray) -> np.ndarray:
    """``S^+`` on the kept space."""
    nd = len(s) - Vk.shape[1]
    return (Vk / s[nd:]) @ Vk.T


def edot(b, bv, A, Av, M, *, Sp=None, Sv=None, Vn=None) -> float:
    """``d/dv (b^T M b)``. The device's two terms; with ``Sp``/``Sv``/``Vn``
    also the third, ``-2 rho.S_v.P.A.x``, which vanishes only where ``P A = 0``
    (an eigenstate)."""
    x = M @ b
    out = 2 * bv @ x - x @ Av @ x
    if Sp is not None and Vn.shape[1]:
        out -= 2 * (Sp @ x) @ Sv @ Vn @ (Vn.T @ (A @ x))
    return out


def eddot(b, bu, bv, buv, A, Au, Av, Auv, M, Sp, Su, Sv, Vn) -> tuple[float, float]:
    """``d^2/du dv (b^T M b)`` at a state where ``P A = 0``, as (section 6.1's
    terms, the rotation terms). The second is zero when nothing is dropped."""
    x = M @ b
    gu, gv = bu - Au @ x, bv - Av @ x
    base = 2 * buv @ x - x @ Auv @ x + 2 * gu @ M @ gv
    if not Vn.shape[1]:
        return base, 0.0
    rho, sigma = Sp @ x, Sp @ b

    def P(y):
        return Vn @ (Vn.T @ y)

    rot = (2 * rho @ Su @ P(gv) - 2 * rho @ Sv @ P(Au @ x)
           + 2 * rho @ Sv @ P(Su @ sigma))
    return base, rot


def lift_matrix(Su, Sv, Suv, Sp, Vn) -> np.ndarray:
    """The second-order effective metric on the dropped space along (u, v):
    ``P (S_uv - S_u S^+ S_v - S_v S^+ S_u) P``. Its (u, u) eigenvalues times
    h^2 are the dropped eigenvalues of ``S(c + h u)``; zero means constant rank
    to second order."""
    return Vn.T @ (Suv - Su @ Sp @ Sv - Sv @ Sp @ Su) @ Vn


# --- per class -----------------------------------------------------------------


def class_energy(cls: str, D: dict, st: State, nd: int | None = None):
    S, K, S1 = build(cls, D, st)
    S, K = sym(S), sym(K)
    s, Vn, Vk, nd = split(S, nd)
    E, minden = 0.0, np.inf
    X = Vk / np.sqrt(s[nd:])
    lam, W = np.linalg.eigh(X.T @ K @ X)
    U = X @ W
    for w, x1, x2, h1, d in tuples(cls, st):
        y1 = U.T @ (S @ x1 + (S1 @ h1 if h1 is not None else 0))
        y2 = y1 if x2 is None else U.T @ (S @ x2)
        den = lam + d
        minden = min(minden, den.min())
        E -= w * np.sum(y1 * y2 / den)
    return E, dict(s=s, nd=nd, minden=minden)


def spectrum(s: np.ndarray, nd: int) -> dict:
    smax = s[-1]
    big_drop = np.abs(s[:nd]).max() / smax if nd else 0.0
    kept = s[nd] / smax
    return dict(nd=nd, d=len(s), dropped=big_drop, kept=kept,
                gap=kept / (big_drop if nd else PC_TAU))


class Derivs:
    """One class's matrices at c and their polynomial coefficients along
    ``c + s u + t v``."""

    def __init__(self, cls, st, Dc, Bcu, Bcv, Buv, Quu):
        Z = {k: np.zeros_like(v) for k, v in Dc.items()}
        S0, K0, S10 = build(cls, Z, st)

        def lin(D):
            S, K, S1 = build(cls, D, st)
            return sym(S - S0), sym(K - K0), (None if S1 is None else S1 - S10)

        S, K, self.S1 = build(cls, Dc, st)
        self.S, self.K = sym(S), sym(K)
        self.Su, self.Ku, self.S1u = lin(comb([(2, Bcu)]))
        self.Sv, self.Kv, self.S1v = lin(comb([(2, Bcv)]))
        self.Suv, self.Kuv, self.S1uv = lin(comb([(2, Buv)]))
        self.Suu = lin(Quu)[0]
        self.s, self.Vn, self.Vk, self.nd = split(self.S)
        self.Sp = kept_pinv(self.s, self.Vk)

    def hessian(self, cls, st):
        """Summed over tuples: (Eddot without, rotation terms, Edot device,
        Edot exact) -- all with the class's -weight."""
        tot = np.zeros(4)
        for w, x1, x2, h1, d in tuples(cls, st):
            assert x2 is None, cls

            def b_(S_, S1_, x=x1, h1=h1):
                return S_ @ x + (S1_ @ h1 if h1 is not None else 0)

            b, bu, bv = b_(self.S, self.S1), b_(self.Su, self.S1u), b_(self.Sv, self.S1v)
            buv = b_(self.Suv, self.S1uv)
            A = self.K + d * self.S
            Au, Av = self.Ku + d * self.Su, self.Kv + d * self.Sv
            Auv = self.Kuv + d * self.Suv
            M = restricted_inverse(A, self.Vk)
            base, rot = eddot(b, bu, bv, buv, A, Au, Av, Auv, M, self.Sp, self.Su, self.Sv,
                              self.Vn)
            e_dev = edot(b, bv, A, Av, M)
            e_ex = edot(b, bv, A, Av, M, Sp=self.Sp, Sv=self.Sv, Vn=self.Vn)
            tot -= w * np.array([base, rot, e_dev, e_ex])
        return tot


def edot_at(cls, st, D, Dv, nd, exact):
    """Edot along ``v`` at the state whose RDM tuple is ``D`` (``Dv`` the
    tuple's linear tangent), cut at the fixed count ``nd``: the device's
    formula, or with ``exact`` the true derivative of the fixed-count
    energy."""
    Z = {k: np.zeros_like(v) for k, v in D.items()}
    S0, K0, S10 = build(cls, Z, st)
    S, K, S1 = build(cls, D, st)
    Sv, Kv, S1v = build(cls, Dv, st)
    S, K, Sv, Kv = sym(S), sym(K), sym(Sv - S0), sym(Kv - K0)
    S1v = None if S1 is None else S1v - S10
    s, Vn, Vk, nd = split(S, nd)
    Sp = kept_pinv(s, Vk)
    out = 0.0
    for w, x1, _x2, h1, d in tuples(cls, st):
        b = S @ x1 + (S1 @ h1 if h1 is not None else 0)
        bv = Sv @ x1 + (S1v @ h1 if h1 is not None else 0)
        A, Av = K + d * S, Kv + d * Sv
        M = restricted_inverse(A, Vk)
        kw = dict(Sp=Sp, Sv=Sv, Vn=Vn) if exact else {}
        out -= w * edot(b, bv, A, Av, M, **kw)
    return out


def richardson_fd(f, pows) -> tuple[float, float]:
    """Central differences of ``f`` at h = 2^-p, extrapolated twice; returns
    (value, spread of the last two once-extrapolated values, relative)."""
    fd = [(f(2.0**-p) - f(-(2.0**-p))) / (2 * 2.0**-p) for p in pows]
    r1 = [(4 * fd[i + 1] - fd[i]) / 3 for i in range(len(fd) - 1)]
    r2 = (16 * r1[-1] - r1[-2]) / 15
    return r2, abs(r1[-1] - r1[-2]) / max(abs(r2), 1e-300)


# --- vectors of each kind --------------------------------------------------------


def sign_group(st: State, tol: float = 1e-10) -> np.ndarray:
    """Every orbital sign pattern leaving h1e and h2e invariant: the abelian
    point group the integrals carry, as diagonal +-1 operations."""
    h1, h2 = st.a["h1e"], st.a["h2e"]
    G = []
    for bits in itertools.product((1.0, -1.0), repeat=st.n):
        g = np.array(bits)
        if (np.abs(h1 * np.outer(g, g) - h1).max() <= tol * np.abs(h1).max()
                and np.abs(h2 * np.einsum("p,q,r,s->pqrs", g, g, g, g) - h2).max()
                <= tol * np.abs(h2).max()):
            G.append(g)
    return np.array(G)


def sector_mask(st: State) -> tuple[np.ndarray, int]:
    """The determinants with the state's character under :func:`sign_group`."""
    G = sign_group(st)

    def chars(nel):
        strs = fci.cistring.make_strings(range(st.n), nel)
        occ = np.array([[(int(x) >> p) & 1 for p in range(st.n)] for x in strs])
        return np.prod(np.where(occ[:, None, :] == 1, G[None], 1.0), axis=2)

    ch = chars(st.nelec[0])[:, None, :] * chars(st.nelec[1])[None, :, :]
    top = np.unravel_index(np.argmax(np.abs(st.c)), st.c.shape)
    return np.all(ch == ch[top][None, None, :], axis=2), len(G)


def singlet(st: State, v: np.ndarray) -> np.ndarray:
    """The S = 0 component of ``v``, by the S^2 polynomial."""
    na, nb = st.nelec
    out = v
    for S in np.arange(abs(na - nb) / 2 + 1, (na + nb) / 2 + 0.5):
        out = (spin_op.contract_ss(out, st.n, st.nelec) - S * (S + 1) * out) / (-S * (S + 1))
    return out


def family(st: State, rng, count: int) -> list[np.ndarray]:
    """States sharing the Hamiltonian's full symmetry, not only its abelian
    part: the root nearest the state of ``(h1 + b h1^2, a h2)`` at random
    ``(a, b)`` near ``(1, 0)``, spin fixed to singlet."""
    out = []
    for _ in range(count):
        al, be = 0.7 + 0.6 * rng.random(), 0.1 * rng.random() - 0.05
        solver = fci.addons.fix_spin_(fci.direct_spin1.FCI(), ss=0)
        solver.conv_tol, solver.nroots = 1e-12, 4
        _, vs = solver.kernel(st.a["h1e"] + be * st.a["h1e"] @ st.a["h1e"], al * st.eri,
                              st.n, st.nelec)
        v = vs[int(np.argmax([abs(np.vdot(x, st.c)) for x in vs]))]
        out.append(v / np.linalg.norm(v))
    return out


def vector(st: State, kind: str, rng, mask) -> np.ndarray:
    if kind == "Epq":  # pcTangentDirection: sum_pq w_pq E_pq |c>, dm1[p,q] != 0
        dm1 = st.a["dm1"]
        f = rng.uniform(-1, 1, (st.n, st.n))
        f[np.abs(dm1) <= 1e-12 * np.abs(dm1).max()] = 0
        v = fci.direct_spin1.contract_1e(f, st.c, st.n, st.nelec)
    elif kind == "family":
        v = family(st, rng, 1)[0]
    elif kind == "sector":
        v = singlet(st, rng.standard_normal(st.c.shape) * mask)
    elif kind == "singlet":
        v = singlet(st, rng.standard_normal(st.c.shape))
    elif kind == "full":
        v = rng.standard_normal(st.c.shape)
    else:
        raise ValueError(kind)
    return v / np.linalg.norm(v)


def given_direction(st: State, weights: list[float]) -> np.ndarray:
    """pcTangentDirection with its weights handed in: ``w`` over the pairs
    with ``dm1[p,q] != 0``, in row-major ``(p, q)`` order -- the values the
    device's ``std::mt19937_64(seed)`` + ``uniform_real_distribution(-1, 1)``
    draws, so a run can be repeated on the CPU direction for direction."""
    dm1 = st.a["dm1"]
    pairs = [(p, q) for p in range(st.n) for q in range(st.n)
             if abs(dm1[p, q]) > 1e-12 * np.abs(dm1).max()]
    if len(pairs) != len(weights):
        raise SystemExit(f"--given-u/--given-v need {len(pairs)} weights (one per "
                         f"dm1-nonzero pair), got {len(weights)}")
    f = np.zeros((st.n, st.n))
    for (p, q), w in zip(pairs, weights, strict=True):
        f[p, q] = w
    v = fci.direct_spin1.contract_1e(f, st.c, st.n, st.nelec)
    v = v - np.vdot(st.c, v) * st.c
    return v / np.linalg.norm(v)


def direction(st, kind, rng, mask) -> np.ndarray:
    v = vector(st, kind, rng, mask)
    v = v - np.vdot(st.c, v) * st.c
    return v / np.linalg.norm(v)


# --- the four parts -------------------------------------------------------------


def part_energies(st: State, Dc: dict) -> dict:
    print("energies: the CPU rebuild against the golden's block2 PC fields, and S's cut")
    print(f"  {'class':5s} {'E (Ha)':>16s} {'- golden':>9s} {'d':>5s} {'dropped':>7s} "
          f"{'max dropped':>11s} {'min kept':>9s} {'gap':>8s} {'min den':>8s}")
    nds = {}
    for cls in PC_CLASSES:
        E, info = class_energy(cls, Dc, st)
        sp = spectrum(info["s"], info["nd"])
        nds[cls] = info["nd"]
        ref = st.a["pc_class_energies"][GOLDEN_ORDER.index(cls)]
        print(f"  {cls:5s} {E:16.12f} {E - ref:9.1e} {sp['d']:5d} {sp['nd']:7d} "
              f"{sp['dropped']:11.1e} {sp['kept']:9.1e} {sp['gap']:8.1e} {info['minden']:8.3f}")
    return nds


def part_structure(st: State, Dc: dict, rng, mask, nrand: int, nfam: int) -> None:
    print(f"structure: dropped modes null for EVERY vector of a kind (common), and each "
          f"vector's own nullity ({nrand} random vectors per kind, {nfam} for family)")
    fam = family(st, rng, nfam)
    vecs = {k: [vector(st, k, rng, mask) for _ in range(nrand)]
            for k in ("full", "singlet", "sector")}
    basis = np.array([st.c.ravel()] + [v.ravel() for v in fam])
    vecs["family"] = []
    for _ in range(nrand):
        v = (rng.standard_normal(len(basis)) @ basis).reshape(st.c.shape)
        vecs["family"].append(v / np.linalg.norm(v))
    Ds = {k: [st.Q(v, f3=False) for v in vs] for k, vs in vecs.items()}
    for cls in DROPPING:
        S, K, _ = build(cls, Dc, st)
        s, Vn, Vk, nd = split(S)
        pk = np.linalg.norm(Vn.T @ sym(K)) / np.linalg.norm(K) if nd else 0.0
        cells = []
        for kind, Dl in Ds.items():
            acc, each = 0.0, []
            for D in Dl:
                Sk = build_S(cls, D, st)
                e = np.linalg.eigvalsh(Sk)
                each.append(int(np.sum(e <= NULL_REL * e[-1])))
                acc = acc + Sk
            e = np.linalg.eigvalsh(acc)
            common = int(np.sum(e <= NULL_REL * e[-1]))
            lo, hi = min(each), max(each)
            cells.append(f"{kind} {common}/{lo if lo == hi else f'{lo}-{hi}'}")
        print(f"  {cls:4s} d={len(s):5d} dropped at c={nd:4d}  |P K|/|K|={pk:.1e}  "
              f"common/per-vector: " + "  ".join(cells))


def lift_order(cls, st, Dc, Bcu, Quu, nd) -> str:
    """The order in h at which the dropped eigenvalues of S(c + h u) leave
    zero: the slope of the largest between h = 2^-6 and 2^-8."""
    mus = []
    for p in (6, 8):
        h = 2.0**-p
        e = np.linalg.eigvalsh(build_S(cls, comb([(1, Dc), (2 * h, Bcu), (h * h, Quu)]), st))
        mus.append(np.abs(e[:nd]).max() / e[-1])
    if mus[1] < 1e-15:
        return f"none (<1e-15 at 2^-8; {mus[0]:.0e} at 2^-6)"
    return f"h^{np.log2(mus[0] / mus[1]) / 2:.1f} ({mus[1]:.0e} at 2^-8)"


def a_lift(cls, st, Dc, Bcu, Quu, nd) -> str:
    """``A = K + Delta S``'s own eigenvalues on the ``nd`` smallest slots
    along ``c + h u`` (Delta of the class's first tuple), at h = 2^-6 and
    2^-8, over its largest: S keeps its rank there, A need not."""
    d = tuples(cls, st)[0][4]
    out = []
    for p in (6, 8):
        h = 2.0**-p
        S, K, _ = build(cls, comb([(1, Dc), (2 * h, Bcu), (h * h, Quu)]), st)
        e = np.sort(np.abs(np.linalg.eigvalsh(sym(K) + d * sym(S))))
        out.append(e[nd - 1] / e[-1])
    return f"{out[0]:.0e} / {out[1]:.0e}"


def part_order_and_hessian(st, Dc, rng, mask, kinds, parts, fd_pows, eps, given) -> None:
    for kind in kinds:
        if kind == "given":
            u, v = (given_direction(st, w) for w in given)
        else:
            u, v = direction(st, kind, rng, mask), direction(st, kind, rng, mask)
        Bcu, Bcv, Buv = st.B(st.c, u), st.B(st.c, v), st.B(u, v)
        Quu, Qvv = st.Q(u), st.Q(v)
        print(f"directions {kind}: u, v orthogonal to c, unit")
        for cls in HESSIAN_CLASSES:
            dv = Derivs(cls, st, Dc, Bcu, Bcv, Buv, Quu)
            nd, nrm = dv.nd, np.linalg.norm
            if "order" in parts and nd:
                Vn, Vk = dv.Vn, dv.Vk
                L = lift_matrix(dv.Su, dv.Sv, dv.Suv, dv.Sp, Vn)
                Luu = lift_matrix(dv.Su, dv.Su, 2 * dv.Suu, dv.Sp, Vn) / 2
                luu = np.abs(np.linalg.eigvalsh(sym(Luu))).max() / dv.s[-1]
                print(f"  order   {cls:4s}"
                      f" |P S_u P|/|S_u|={nrm(Vn.T @ dv.Su @ Vn) / nrm(dv.Su):.1e}"
                      f" |P S_u Q|/|S_u|={nrm(Vn.T @ dv.Su @ Vk) / nrm(dv.Su):.2f}"
                      f" |P S_uv P|/|S_uv|={nrm(Vn.T @ dv.Suv @ Vn) / nrm(dv.Suv):.1e}"
                      f" |L_uv|/|S_uv|={nrm(L) / nrm(dv.Suv):.1e}"
                      f" max eig L_uu/s_max={luu:.1e}"
                      f" |P K_u Q|/|K_u|={nrm(Vn.T @ dv.Ku @ Vk) / nrm(dv.Ku):.1e}"
                      f" |P K_u P|/|K_u|={nrm(Vn.T @ dv.Ku @ Vn) / nrm(dv.Ku):.1e}"
                      f" lift along c+hu: {lift_order(cls, st, Dc, Bcu, Quu, nd)}"
                      f"; A's at 2^-6 / 2^-8: {a_lift(cls, st, Dc, Bcu, Quu, nd)}")
            if "hessian" not in parts:
                continue
            base, rot, e_dev, e_ex = dv.hessian(cls, st)
            full = base + rot

            def ex(h, exact=True, cls=cls, nd=nd, pc=(Bcu, Bcv, Quu, Buv, Qvv)):
                bcu, bcv, quu, buv, qvv = pc
                D = along(Dc, bcu, bcv, quu, buv, qvv, h, 0.0)
                Dv = comb([(2, bcv), (2 * h, buv)])  # the tuple's tangent along v there
                return edot_at(cls, st, D, Dv, nd, exact)

            ref, spread = richardson_fd(ex, fd_pows)
            dev, _ = richardson_fd(lambda h, ex=ex: ex(h, False), fd_pows)
            sc = max(abs(ref), 1e-300)
            moved = ""
            if eps > 0:  # a CI-convergence-sized perturbation of the state itself
                c2 = st.c + eps * direction(st, "sector", rng, mask)
                c2 = c2 / np.linalg.norm(c2)
                dv2 = Derivs(cls, st, st.Q(c2), st.B(c2, u), st.B(c2, v), Buv, Quu)
                b2, r2, _, _ = dv2.hessian(cls, st)
                moved = f"  repaired moves {abs(b2 + r2 - full) / sc:.0e} at |dc|={eps:.0e}"
            print(f"  hessian {cls:4s} ref={ref: .10e} (spread {spread:.0e})"
                  f" sec.6.1={-0.0 + base: .10e} FD(device Edot)={dev: .10e}"
                  f"  |sec.6.1-ref|/|ref|={abs(base - ref) / sc:.1e}"
                  f"  |repaired-ref|/|ref|={abs(full - ref) / sc:.1e}"
                  f"  rotation/ref={rot / sc: .1e}"
                  f"  |FD(device Edot)-ref|/|ref|={abs(dev - ref) / sc:.1e}"
                  f"  device-vs-exact Edot at c={abs(e_dev - e_ex) / max(abs(e_ex), 1e-300):.0e}"
                  f"  s_min={dv.s[dv.nd] / dv.s[-1]:.1e}" + moved)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("golden", help="a committed .nevpt2gold with PC fields")
    ap.add_argument("--parts", default="energies,structure,order,hessian")
    ap.add_argument("--kinds", default=",".join(DIRECTION_KINDS),
                    help="direction kinds for order/hessian: Epq, family, sector, and "
                         "singlet (any irrep: what a non-totally-symmetric response is)")
    ap.add_argument("--given-u", default="", help="kind 'given': u's pcTangentDirection "
                    "weights, comma-separated (see given_direction)")
    ap.add_argument("--given-v", default="", help="kind 'given': v's weights")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--nrand", type=int, default=8, help="random vectors per kind (structure)")
    ap.add_argument("--nfamily", type=int, default=4, help="family states (structure)")
    ap.add_argument("--fd-pows", default="12,13,14",
                    help="the reference difference's steps, h = 2^-p (three or more)")
    ap.add_argument("--eps", type=float, default=1e-9,
                    help="size of the state perturbation the sensitivity column uses")
    args = ap.parse_args(argv)
    parts = set(args.parts.split(","))
    t0 = time.time()
    st = State(args.golden)
    rng = np.random.default_rng(args.seed)
    mask, gorder = sector_mask(st)
    h2 = fci.direct_spin1.absorb_h1e(st.a["h1e"], st.eri, st.n, st.nelec, 0.5)
    hc = fci.direct_spin1.contract_2e(h2, st.c, st.n, st.nelec)
    resid = np.linalg.norm(hc - np.vdot(st.c, hc) * st.c)
    print(f"{args.golden}: CAS({sum(st.nelec)},{st.n}), {st.c.size} determinants, abelian "
          f"group order {gorder}, {int(mask.sum())} determinants in the state's irrep, "
          f"CI residual |Hc - Ec| = {resid:.1e}; threads {os.environ['OMP_NUM_THREADS']}")
    Dc = st.Q(st.c)
    if "energies" in parts:
        part_energies(st, Dc)
    if "structure" in parts:
        part_structure(st, Dc, rng, mask, args.nrand, args.nfamily)
    if parts & {"order", "hessian"}:
        part_order_and_hessian(st, Dc, rng, mask, args.kinds.split(","), parts,
                               [int(p) for p in args.fd_pows.split(",")], args.eps,
                               [[float(x) for x in w.split(",")] if w else []
                                for w in (args.given_u, args.given_v)])
    print(f"wall {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
