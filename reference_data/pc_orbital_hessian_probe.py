#!/usr/bin/env python3
r"""PC-NEVPT2's orbital Hessian blocks, offline: whether the S-cut's dropped
space moves along an orbital direction, what the mixed block H_kc carries
because it moves along a CI one, and whether SC's empty-perturber pathology
reaches PC -- measured on a committed golden with PySCF on the CPU (no card,
no build). The theory is docs/hessian-theory.md, sections 6.4-6.8; the numbers
are in docs/pc-nevpt2.md, "The orbital blocks, offline".

The energy the device differentiates is ``E(kappa, c)``: the integrals
rotated by ``C exp(kappa)`` at a FIXED CI vector. PC's metric ``S`` and
one-body column ``S_1`` are built from the RDMs alone, so along any orbital
direction ``w`` -- active-active included -- they do not move at all, the
kept span ``Q`` is fixed, and with ``M = Q (Q^T A Q)^-1 Q^T``

  dE/dw = -(2 x.b_w - x.A_w.x),   x = M b,   exactly, at ANY CI vector

(``A_w = K_w`` here: the orbital energies are held fixed, the functional
``pcHessianDevice`` differentiates). Along a CI direction ``u`` the kept span
turns (docs/hessian-theory.md, section 6.2), so the mixed derivative is

  d2E/dw du = -( 2 b_wu.x - x.A_wu.x + 2 g_w.M.g_u          (sec. 6.1's terms)
                 + 2 g_w.(M_u + M A_u M).b )                (the rotation)

with ``g = b' - A' x``, ``M_u`` the S-cut inverse's full derivative and, at an
eigenstate, the rotation equal to ``-2 rho.S_u.P.A_w.x`` (``rho = S^+ x``).
It vanishes only if ``P K_w x = 0`` -- and ``K_w``, the Dyall block at
dotted integrals, annihilates ``S``'s dropped space from the bra side only.

Three parts:

``kk``     per class, along two random ACTIVE-ACTIVE directions ``v``, ``w``:
           the orbital block in closed form with NO projector or rotation
           term (``S`` is fixed along both),

             d2E/dv dw = -( 2 b_vw.x - x.A_vw.x + 2 g_v.M.g_w ),

           ``b_vw``, ``A_vw`` at the inputs' second derivative along
           ``exp(s v + t w)`` (every slot once by ``W = (vw + wv)/2``, and
           pairs of slots once by each), against a Richardson-extrapolated
           four-point ladder of the rotated energy; and the identity that
           ties it to the mixed block with no step size, from
           ``exp(t v) exp(s w)`` (Baker-Campbell-Hausdorff) and the gauge
           below:

             Eddot_kk(v, w) + Edot([v, w]) / 2 = Eddot_kc(v, w^ c).

``mixed``  per class (Sr, Si, Sir and the no-drop control Srs) and CI
           direction kind, along a random ACTIVE-ACTIVE ``w`` (the only
           orbital direction the golden alone can rotate exactly): the
           first-order check (the formula above against a Richardson
           difference of the rotated energy), the size of ``P K_w x``, the
           mixed derivative without and with the rotation term against a
           Richardson difference of the exact ``dE/dw`` along ``c + t u``,
           and the gauge identity that holds the whole block against the
           CI block with no step size: ``E(exp(s w), c) = E(1, exp(s w^) c)``
           for an active ``w`` (``w^`` its determinant-space generator), so

             Eddot_kc(w, u) = Eddot_cc(w^ c, u) + Edot(w^ u).

``empty``  per class, along core-core and virtual-virtual pairs (the
           intra-space rotations ``sc_empty_perturber_probe.py`` runs: the
           metric and the Dyall block stay fixed, the stored blocks and the
           orbital energies move), the PC class energy's mixed second
           derivative in closed form -- the empty tuples' share
           ``-2 b_u.M.b_v`` on its own -- against a Richardson-extrapolated
           four-point ladder of the PC class energy, beside SC's
           proportionality defect ``rho`` on the same (class, pair). Where
           SC's ``rho`` is large SC has no Hessian along the pair; PC's
           ladder must still land on its bilinear form.

Usage::

    uv run python reference_data/pc_orbital_hessian_probe.py golden/n2_ccpvdz_cas44.nevpt2gold
    uv run python reference_data/pc_orbital_hessian_probe.py \
        golden/acrolein_def2svp_cas65.nevpt2gold --parts kk,mixed \
        --fd-pows 18,19,20 --first-pows 14,15,16 --kk-pows 12,13,14

Threads default to 6 (``OMP_NUM_THREADS`` etc. are only set when absent).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "6")

import numpy as np  # noqa: E402
import pc_null_mode_probe as pnm  # noqa: E402
import sc_empty_perturber_probe as sep  # noqa: E402
from pyscf.fci import direct_nosym  # noqa: E402
from scipy.linalg import expm  # noqa: E402

#: The ACTIVE axes of every array the PC classes read, in the golden's
#: layouts (pc_null_mode_probe's ``tuples`` and ``build``). An active-active
#: rotation turns these and nothing else: ``h1e_v_Sr``'s subtracted trace and
#: the core dressing in ``h1eff`` are invariant under it, and ``h1e_v_Sir``
#: has no active index (listed with no axes, so that its dot is zero).
ACT_AXES = {
    "h1e": (0, 1),
    "h2e": (0, 1, 2, 3),
    "h2e_v_Sr": (1, 2, 3),
    "h1e_v_Sr": (1,),
    "h2e_v_Si": (0, 1, 3),
    "h1e_v_Si": (0,),
    "h2e_v_Sijr": (1,),
    "h2e_v_Srsi": (3,),
    "h2e_v_Srs": (2, 3),
    "h2e_v_Sij": (0, 1),
    "h2e_v1_Sir": (1, 3),
    "h2e_v2_Sir": (1, 2),
    "h1e_v_Sir": (),
}
#: What the mixed part runs: the dropping classes, and Srs as the control
#: whose rotation term must be exactly zero (nothing dropped).
MIXED_CLASSES = pnm.HESSIAN_CLASSES
#: A tuple whose largest feeding integral is at or below this is EMPTY.
X_ZERO = sep.X_ZERO


def sym(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)


# --- the algebra: pure functions on matrices ------------------------------------


def orbital_first(b, bw, Aw, M) -> float:
    """``d/dw (b^T M b)`` along a direction that leaves ``S`` (so ``Q``) fixed:
    ``2 b_w.x - x.A_w.x``, exact at any CI vector -- no projector term."""
    x = M @ b
    return 2 * bw @ x - x @ Aw @ x


def cut_inverse_derivative(M, A, Au, Su, Sp, Vn) -> np.ndarray:
    """The S-cut restricted inverse's derivative along a direction that moves
    ``S`` (docs/hessian-theory.md, section 6.2), at any CI vector:
    ``-M A' M + P S' S^+ M + M S^+ S' P - M S^+ S' P A M - M A P S' S^+ M``."""
    P = Vn @ Vn.T
    return (-M @ Au @ M + P @ Su @ Sp @ M + M @ Sp @ Su @ P - M @ Sp @ Su @ P @ A @ M
            - M @ A @ P @ Su @ Sp @ M)


def orbital_mixed(b, bw, bu, bwu, A, Aw, Au, Awu, M, Sp, Su, Vn):
    """``d2/dw du (b^T M b)`` with ``w`` orbital (``S_w = 0``) and ``u`` a CI
    direction: (section 6.1's terms, the rotation term exactly, the rotation
    term in its eigenstate form ``-2 rho.S_u.P.A_w.x``)."""
    x = M @ b
    gw, gu = bw - Aw @ x, bu - Au @ x
    base = 2 * bwu @ x - x @ Awu @ x + 2 * gw @ M @ gu
    if not Vn.shape[1]:
        return base, 0.0, 0.0
    Mu = cut_inverse_derivative(M, A, Au, Su, Sp, Vn)
    rot = 2 * gw @ (Mu + M @ Au @ M) @ b
    rho = Sp @ x
    rot_eig = -2 * rho @ Su @ Vn @ (Vn.T @ (Aw @ x))
    return base, rot, rot_eig


def pair_second(b1, b2, b1u, b2u, b1v, b2v, b1uv, b2uv, Auv, M) -> float:
    """``d2/du dv (b1^T M b2)`` when only ``Delta`` moves the solve, and only
    at second order (an intra-space pair: ``A_u = A_v = 0``,
    ``A_uv = Delta_uv S``). Bilinear in the two first-order sets whatever
    ``b1``, ``b2`` are -- an empty tuple (``b = 0``) included."""
    x1, x2 = M @ b1, M @ b2
    return (b1uv @ x2 + x1 @ b2uv - x1 @ Auv @ x2 + b1u @ M @ b2v + b1v @ M @ b2u)


def second_dotted(blocks: dict, axes: dict, ku: dict, kv: dict) -> dict:
    """The mixed second derivative of every block along ``exp(s ku + t kv)``
    (``ku``, ``kv`` per space): each rotated axis once by ``W = (ku kv + kv
    ku)/2``, and every ordered pair of different axes once by ``ku`` and once
    by ``kv``. ``axes[name]`` lists ``(axis, space)``."""
    out = {}
    for name, ax in axes.items():
        X = blocks[name]
        acc = np.zeros_like(X)
        for a, sa in ax:
            acc += sep.contract(X, a, 0.5 * (ku[sa] @ kv[sa] + kv[sa] @ ku[sa]))
            for b, sb in ax:
                if b != a:
                    acc += sep.contract(sep.contract(X, a, ku[sa]), b, kv[sb])
        out[name] = acc
    return out


def eps_second(e: np.ndarray, ku: np.ndarray, kv: np.ndarray) -> np.ndarray:
    """``d2/ds dt`` of :func:`sc_empty_perturber_probe.rotate_eps` at
    ``exp(s ku + t kv)``: ``2 sum_q ku_qp kv_qp e_q + 2 W_pp e_p``. The first
    derivative is zero (an antisymmetric kappa has no diagonal)."""
    W = 0.5 * (ku @ kv + kv @ ku)
    return 2 * np.einsum("qp,qp,q->p", ku, kv, e) + 2 * np.diag(W) * e


def richardson(vals) -> float:
    r1 = [(4 * b - a) / 3 for a, b in zip(vals, vals[1:], strict=False)]
    return (16 * r1[-1] - r1[-2]) / 15


# --- a rotated view of a golden --------------------------------------------------


def view(st: pnm.State, over: dict) -> pnm.State:
    """``st`` with some arrays replaced (``over``), its ``eri`` rebuilt from
    the replaced ``h2e``: what pc_null_mode_probe's ``build``, ``tuples`` and
    ``State.Q`` read."""
    out = copy.copy(st)
    out.a = {**st.a, **over}
    if "h2e" in over:
        out.eri = np.ascontiguousarray(over["h2e"].transpose(0, 2, 1, 3))
    return out


def act_rotated(st: pnm.State, U: np.ndarray) -> pnm.State:
    over = {}
    for name, axes in ACT_AXES.items():
        X = st.a[name]
        for ax in axes:
            X = sep.contract(X, ax, U)
        over[name] = X
    return view(st, over)


def act_dotted(st: pnm.State, w: np.ndarray) -> pnm.State:
    over = {}
    for name, axes in ACT_AXES.items():
        X = st.a[name]
        acc = np.zeros_like(X)
        for ax in axes:
            acc += sep.contract(X, ax, w)
        over[name] = acc
    return view(st, over)


def act_second(st: pnm.State, v: np.ndarray, w: np.ndarray) -> pnm.State:
    """Every active-axis array's mixed second derivative along
    ``exp(s v + t w)`` (:func:`second_dotted` on the active axes)."""
    axes = {name: tuple((ax, "a") for ax in ACT_AXES[name]) for name in ACT_AXES}
    return view(st, second_dotted(st.a, axes, {"a": v}, {"a": w}))


def rotated_energy(st: pnm.State, cls: str, U: np.ndarray, nd: int) -> float:
    """One class's PC energy with the integrals rotated by ``C -> C U`` on the
    active space, at the golden's CI vector and orbital energies; the cut
    count held at ``nd``."""
    sth = act_rotated(st, U)
    S, K, S1 = pnm.build(cls, sth.Q(st.c), sth)
    S, K = sym(S), sym(K)
    _, _, Vk, _ = pnm.split(S, nd)
    out = 0.0
    for wt, x, _x2, h1, d in pnm.tuples(cls, sth):
        b = b_of(S, S1, x, h1)
        out -= wt * b @ pnm.restricted_inverse(K + d * S, Vk) @ b
    return out


def lin_part(cls, D, st):
    """The class matrices' part linear in the RDM tuple ``D`` (the RDM-free
    terms subtracted), at ``st``'s integrals."""
    Z = {k: np.zeros_like(v) for k, v in D.items()}
    S, K, S1 = pnm.build(cls, D, st)
    S0, K0, S10 = pnm.build(cls, Z, st)
    return sym(S - S0), sym(K - K0), (None if S1 is None else S1 - S10)


def b_of(S, S1, x, h1):
    return S @ x + (S1 @ h1 if h1 is not None else 0)


# --- the mixed part ----------------------------------------------------------------


class Mixed:
    """One class at the state, along one active-active ``w``."""

    def __init__(self, cls, st, stw, Dc, Dcw):
        self.cls, self.st, self.stw = cls, st, stw
        S, K, self.S1 = pnm.build(cls, Dc, st)
        self.S, self.K = sym(S), sym(K)
        self.Kw = sym(pnm.build(cls, Dcw, stw)[1])  # K is linear in (h1e, h2e, f3)
        self.s, self.Vn, self.Vk, self.nd = pnm.split(self.S)
        self.Sp = pnm.kept_pinv(self.s, self.Vk)
        self.tup = pnm.tuples(cls, st)
        self.tupw = pnm.tuples(cls, stw)

    def first(self) -> float:
        out = 0.0
        for (wt, x, _x2, h1, d), (_, xw, _, h1w, _) in zip(self.tup, self.tupw, strict=True):
            A = self.K + d * self.S
            M = pnm.restricted_inverse(A, self.Vk)
            out -= wt * orbital_first(b_of(self.S, self.S1, x, h1),
                                      b_of(self.S, self.S1, xw, h1w), self.Kw, M)
        return out

    def first_at(self, D, Dw) -> float:
        """``dE/dw`` at the (unnormalised) CI vector whose RDM tuple is ``D``,
        ``Dw`` the same with the digests at the dotted ``eri``; the cut count
        held at the state's."""
        S, K, S1 = pnm.build(self.cls, D, self.st)
        S, K = sym(S), sym(K)
        Kw = sym(pnm.build(self.cls, Dw, self.stw)[1])
        _, _, Vk, _ = pnm.split(S, self.nd)
        out = 0.0
        for (wt, x, _x2, h1, d), (_, xw, _, h1w, _) in zip(self.tup, self.tupw, strict=True):
            M = pnm.restricted_inverse(K + d * S, Vk)
            out -= wt * orbital_first(b_of(S, S1, x, h1), b_of(S, S1, xw, h1w), Kw, M)
        return out

    def lift(self, Bcu, Quu) -> float:
        """The CI leg's second-order lift ``L_uu``'s largest eigenvalue over
        ``s_max`` (zero: the rank is constant along u through second order;
        the device refuses the (class, u) above PC_LIFT_TOL). Along ``w`` the
        lift is identically zero: ``S`` does not move."""
        if not self.nd:
            return 0.0
        Su = lin_part(self.cls, pnm.comb([(2, Bcu)]), self.st)[0]
        Suu = lin_part(self.cls, Quu, self.st)[0]
        L = pnm.lift_matrix(Su, Su, 2 * Suu, self.Sp, self.Vn) / 2
        return np.abs(np.linalg.eigvalsh(sym(L))).max() / self.s[-1]

    def mixed(self, Bcu, Bcuw):
        """(sec. 6.1's terms, rotation exact, rotation at the eigenstate, the
        size of P K_w x over K_w x), each summed over the tuples with the
        class's -weight."""
        Su, Ku, S1u = lin_part(self.cls, pnm.comb([(2, Bcu)]), self.st)
        Kwu = lin_part(self.cls, pnm.comb([(2, Bcuw)]), self.stw)[1]
        tot = np.zeros(3)
        pk, ak = 0.0, 0.0
        for (wt, x, _x2, h1, d), (_, xw, _, h1w, _) in zip(self.tup, self.tupw, strict=True):
            A = self.K + d * self.S
            M = pnm.restricted_inverse(A, self.Vk)
            b, bw = b_of(self.S, self.S1, x, h1), b_of(self.S, self.S1, xw, h1w)
            bu, bwu = b_of(Su, S1u, x, h1), b_of(Su, S1u, xw, h1w)
            terms = orbital_mixed(b, bw, bu, bwu, A, self.Kw, Ku + d * Su, Kwu, M, self.Sp,
                                  Su, self.Vn)
            tot -= wt * np.array(terms)
            kx = self.Kw @ (M @ b)
            ak += kx @ kx
            if self.nd:
                pk += np.linalg.norm(self.Vn.T @ kx) ** 2
        return tot, (np.sqrt(pk / ak) if ak else 0.0)


def ci_generator(st: pnm.State, w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``sum_pq w_pq a+_p a_q v``, spin-summed: the determinant-space generator
    of the active rotation ``C -> C exp(w)`` at fixed CI coefficients, so that
    ``E(exp(s w), c) = E(1, exp(s w^) c)``. (direct_nosym: direct_spin1's
    ``contract_1e`` reads only the lower triangle, i.e. assumes symmetry.)"""
    return direct_nosym.contract_1e(w, v, st.n, st.nelec)


def part_mixed(st, rng, mask, kinds, pows, first_pows) -> None:
    n = st.n
    w = sep.antisym(rng, n)
    stw = act_dotted(st, w)
    Dc, Dcw = st.Q(st.c), stw.Q(st.c)
    print(f"mixed: one random active-active w (|w| = 1), orbital energies fixed; "
          f"Richardson steps 2^-{pows}. Absolute values (Eh per unit direction squared)")
    mx = {cls: Mixed(cls, st, stw, Dc, Dcw) for cls in MIXED_CLASSES}
    first = {cls: m.first() for cls, m in mx.items()}
    gc = ci_generator(st, w, st.c)
    for cls, m in mx.items():

        def energy(h, m=m, cls=cls):
            return rotated_energy(st, cls, expm(h * w), m.nd)

        ref = richardson([(energy(2.0**-p) - energy(-(2.0**-p))) / (2 * 2.0**-p)
                          for p in first_pows])
        gauge = pnm.edot_at(cls, st, Dc, pnm.comb([(2, st.B(st.c, gc))]), m.nd, True)
        print(f"  first   {cls:4s} dE/dw={first[cls]: .10e}  |formula - ladder|="
              f"{abs(first[cls] - ref):.1e}  gauge |dE/dw - Edot(w^ c)|="
              f"{abs(first[cls] - gauge):.1e}  dropped={m.nd}")
    Bcg, Qg = st.B(st.c, gc), st.Q(gc)
    for kind in kinds:
        u = pnm.direction(st, kind, rng, mask)
        Bcu, Bcuw, Quu, Quuw = st.B(st.c, u), stw.B(st.c, u), st.Q(u), stw.Q(u)
        Bcgu = st.B(st.c, ci_generator(st, w, u))
        print(f"  direction u: {kind}")
        for cls, m in mx.items():
            (base, rot, rot_eig), pkx = m.mixed(Bcu, Bcuw)

            def ew(t, m=m, b=(Bcu, Quu, Bcuw, Quuw)):
                bcu, quu, bcuw, quuw = b
                D = pnm.comb([(1, Dc), (2 * t, bcu), (t * t, quu)])
                Dw = pnm.comb([(1, Dcw), (2 * t, bcuw), (t * t, quuw)])
                return m.first_at(D, Dw)

            ref = richardson([(ew(2.0**-p) - ew(-(2.0**-p))) / (2 * 2.0**-p) for p in pows])
            # The gauge: Eddot_cc(w^ c, u) + Edot(w^ u) by the CI block's own formulas
            # (pc_null_mode_probe's), with and without ITS rotation terms -- along
            # w^ c the dropped space does turn (S rotates orthogonally).
            dv = pnm.Derivs(cls, st, Dc, Bcg, Bcu, st.B(gc, u), Qg)
            hb, hr, _, _ = dv.hessian(cls, st)
            yu = pnm.edot_at(cls, st, Dc, pnm.comb([(2, Bcgu)]), m.nd, True)
            print(f"  mixed   {cls:4s} ref={ref: .10e}"
                  f"  |sec.6.1 - ref|={abs(base - ref):.1e}"
                  f"  |with rotation - ref|={abs(base + rot - ref):.1e}"
                  f"  rotation={rot: .1e}"
                  f"  |rot - eigenstate form|={abs(rot - rot_eig):.0e}"
                  f"  |P K_w x|/|K_w x|={pkx:.1e}"
                  f"  L_uu/s_max={m.lift(Bcu, Quu):.0e}"
                  f"  gauge: |Eddot_kc - Eddot_cc(w^c,u) - Edot(w^u)|="
                  f"{abs(base + rot - (hb + hr + yu)):.1e}"
                  f" (the CI side without its rotation terms: {abs(base + rot - (hb + yu)):.1e})")


# --- the orbital block --------------------------------------------------------------


def orbital_block(cls, st, Dc, legs) -> float:
    """``d2E/dv dw`` of one class along two active-active directions, S fixed:
    ``legs`` the (view, RDM tuple) pairs at v's, w's and the mixed second
    derivative's dotted inputs (each RDM tuple's digests at that view's eri)."""
    (stv, Dv), (stw, Dw), (stvw, Dvw) = legs
    S, K, S1 = pnm.build(cls, Dc, st)
    S, K = sym(S), sym(K)
    Kv, Kw = sym(pnm.build(cls, Dv, stv)[1]), sym(pnm.build(cls, Dw, stw)[1])
    Kvw = sym(pnm.build(cls, Dvw, stvw)[1])
    _, _, Vk, _ = pnm.split(S)
    out = 0.0
    rows = zip(pnm.tuples(cls, st), pnm.tuples(cls, stv), pnm.tuples(cls, stw),
               pnm.tuples(cls, stvw), strict=True)
    for (wt, x, _, h1, d), (_, xv, _, h1v, _), (_, xw, _, h1w, _), (_, xvw, _, h1vw, _) in rows:
        A = K + d * S
        M = pnm.restricted_inverse(A, Vk)
        b = b_of(S, S1, x, h1)
        xx = M @ b
        gv = b_of(S, S1, xv, h1v) - Kv @ xx
        gw = b_of(S, S1, xw, h1w) - Kw @ xx
        out -= wt * (2 * b_of(S, S1, xvw, h1vw) @ xx - xx @ Kvw @ xx + 2 * gv @ M @ gw)
    return out


def part_kk(st, rng, pows) -> None:
    n = st.n
    v, w = sep.antisym(rng, n), sep.antisym(rng, n)
    stv, stw, stvw = act_dotted(st, v), act_dotted(st, w), act_second(st, v, w)
    Dc = st.Q(st.c)
    legs = ((stv, stv.Q(st.c)), (stw, stw.Q(st.c)), (stvw, stvw.Q(st.c)))
    comm = v @ w - w @ v
    stc = act_dotted(st, comm)
    gc = ci_generator(st, w, st.c)
    print(f"kk: two random active-active v, w (|v| = |w| = 1), orbital energies fixed; "
          f"four-point ladder along exp(s v + t w) at 2^-{pows}")
    for cls in MIXED_CLASSES:
        an = orbital_block(cls, st, Dc, legs)
        nd = pnm.split(sym(pnm.build(cls, Dc, st)[0]))[3]

        def energy(s, t, cls=cls, nd=nd):
            return rotated_energy(st, cls, expm(s * v + t * w), nd)

        lad = richardson([(energy(h, h) - energy(h, -h) - energy(-h, h) + energy(-h, -h))
                          / (4 * h * h) for h in (2.0**-p for p in pows)])
        # Eddot_kc(v, w^ c) - Edot([v, w]) / 2, by the mixed part's own formulas
        mv = Mixed(cls, st, stv, Dc, legs[0][1])
        base, rot, _ = mv.mixed(st.B(st.c, gc), stv.B(st.c, gc))[0]
        half = 0.5 * Mixed(cls, st, stc, Dc, stc.Q(st.c)).first()
        print(f"  kk      {cls:4s} Eddot_kk={an: .10e}  |ladder - formula|={abs(lad - an):.1e}"
              f"  dropped={nd}  |Eddot_kk + Edot([v,w])/2 - Eddot_kc(v, w^ c)|="
              f"{abs(an + half - (base + rot)):.1e} (Eddot_kc's rotation term {rot: .1e})")


# --- the empty part ----------------------------------------------------------------


#: The stored blocks' core / virtual axes, from sc_empty_perturber_probe.
INTRA_AXES = sep.AXES


def part_empty(path, st, rng, pows) -> None:
    case = sep.Case(path)
    nc, nv = case.nc, case.nv
    zc, zv = np.zeros((nc, nc)), np.zeros((nv, nv))
    dirs = {"core": (sep.antisym(rng, nc), zv), "core2": (sep.antisym(rng, nc), zv),
            "virt": (zc, sep.antisym(rng, nv)), "virt2": (zc, sep.antisym(rng, nv))}
    pairs = [("core", "core2"), ("virt", "virt2"), ("core", "virt")]
    Dc = st.Q(st.c)
    ec, ev = st.a["e_core"], st.a["e_virt"]
    print(f"empty: the PC class energies' mixed second derivative along intra-space pairs "
          f"(orbital energies rotated), against a four-point ladder at 2^-{pows}; SC's worst "
          f"rho on the same (class, pair) beside it")
    print(f"  {'pair':14} {'class':5} {'empty':>5} {'both':>4} {'analytic':>15} "
          f"{'empty share':>12} {'|ladder-an|/|an|':>16} {'SC both':>7} {'SC max rho':>10}")
    worst = 0.0
    for pu, pv in pairs:
        ku = {"c": dirs[pu][0], "v": dirs[pu][1]}
        kv = {"c": dirs[pv][0], "v": dirs[pv][1]}
        xu = sep.dotted(case.blocks, *dirs[pu])
        xv = sep.dotted(case.blocks, *dirs[pv])
        xuv = second_dotted(case.blocks, INTRA_AXES, ku, kv)
        stu, stv, stuv = (view(st, x) for x in (xu, xv, xuv))
        sted = view(st, {"e_core": eps_second(ec, ku["c"], kv["c"]),
                         "e_virt": eps_second(ev, ku["v"], kv["v"])})
        for cls in pnm.PC_CLASSES:
            S, K, S1 = pnm.build(cls, Dc, st)
            S, K = sym(S), sym(K)
            _, _, Vk, _ = pnm.split(S)
            tot, emp, nemp, nboth = 0.0, 0.0, 0, 0
            rows = zip(pnm.tuples(cls, st), pnm.tuples(cls, stu), pnm.tuples(cls, stv),
                       pnm.tuples(cls, stuv), pnm.tuples(cls, sted), strict=True)
            for t0, tu, tv, tuv, tdd in rows:
                wt, x1, x2, h1, d = t0
                M = pnm.restricted_inverse(K + d * S, Vk)

                def bb(t, S=S, S1=S1):
                    _, y1, y2, hh, _ = t
                    b1 = b_of(S, S1, y1, hh)
                    return b1, (b1 if y2 is None else S @ y2)

                b1, b2 = bb(t0)
                b1u, b2u = bb(tu)
                b1v, b2v = bb(tv)
                b1uv, b2uv = bb(tuv)
                val = -wt * pair_second(b1, b2, b1u, b2u, b1v, b2v, b1uv, b2uv, tdd[4] * S, M)
                tot += val
                xmax = max(np.abs(x1).max(), 0.0 if x2 is None else np.abs(x2).max(),
                           0.0 if h1 is None else np.abs(h1).max())
                if xmax <= X_ZERO:
                    nemp += 1
                    emp += val
                    if b1u @ b1u > sep.POPULATED and b1v @ b1v > sep.POPULATED:
                        nboth += 1

            def energy(s, t, cls=cls, S=S, K=K, S1=S1, Vk=Vk, ku=ku, kv=kv):
                Uc = expm(s * ku["c"] + t * kv["c"])
                Uv = expm(s * ku["v"] + t * kv["v"])
                stx = view(st, {**sep.rotate(case.blocks, Uc, Uv),
                                "e_core": sep.rotate_eps(ec, Uc),
                                "e_virt": sep.rotate_eps(ev, Uv)})
                out = 0.0
                for wt, y1, y2, hh, d in pnm.tuples(cls, stx):
                    b1 = b_of(S, S1, y1, hh)
                    b2 = b1 if y2 is None else S @ y2
                    out -= wt * b1 @ pnm.restricted_inverse(K + d * S, Vk) @ b2
                return out

            lad = richardson([(energy(h, h) - energy(h, -h) - energy(-h, h) + energy(-h, -h))
                              / (4 * h * h) for h in (2.0**-p for p in pows)])
            rel = abs(lad - tot) / max(abs(tot), 1e-300)
            worst = max(worst, rel)
            # SC on the same (class, pair): sc_empty_perturber_probe's own tables.
            g = case.empty[cls]
            T = {k: v[g] for k, v in sep.cross_tables(cls, xu, xv, case.A).items()}
            both = sep.populated(T["nuu"]) & sep.populated(T["nvv"])
            rho = sep.proportionality(T, case.d0[cls][g]).max(initial=0.0)
            print(f"  {pu + ' x ' + pv:14} {cls:5} {nemp:5d} {nboth:4d} {tot: .8e} "
                  f"{emp: .5e} {rel:16.1e} {int(both.sum()):7d} {rho:10.1e}")
    print(f"  worst |ladder - analytic| / |analytic|: {worst:.1e}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("golden", help="a committed .nevpt2gold with PC fields")
    ap.add_argument("--parts", default="kk,mixed,empty")
    ap.add_argument("--kinds", default="Epq,sector,singlet",
                    help="CI direction kinds for the mixed part (pc_null_mode_probe's)")
    ap.add_argument("--seed", type=int, default=199)
    ap.add_argument("--fd-pows", default="12,13,14",
                    help="the mixed part's reference difference along c + t u, t = 2^-p")
    ap.add_argument("--first-pows", default="6,7,8",
                    help="the mixed part's first-order check along exp(h w), h = 2^-p")
    ap.add_argument("--kk-pows", default="6,7,8",
                    help="the kk part's four-point ladder, h = 2^-p")
    ap.add_argument("--ladder-pows", default="5,6,7",
                    help="the empty part's four-point ladder, h = 2^-p")
    args = ap.parse_args(argv)
    parts = set(args.parts.split(","))
    t0 = time.time()
    st = pnm.State(args.golden)
    rng = np.random.default_rng(args.seed)
    mask, _ = pnm.sector_mask(st)
    print(f"{args.golden}: CAS({sum(st.nelec)},{st.n}), {st.c.size} determinants; "
          f"threads {os.environ['OMP_NUM_THREADS']}")
    if "kk" in parts:
        part_kk(st, rng, [int(p) for p in args.kk_pows.split(",")])
    if "mixed" in parts:
        part_mixed(st, rng, mask, args.kinds.split(","),
                   [int(p) for p in args.fd_pows.split(",")],
                   [int(p) for p in args.first_pows.split(",")])
    if "empty" in parts:
        part_empty(args.golden, st, rng, [int(p) for p in args.ladder_pows.split(",")])
    print(f"wall {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
