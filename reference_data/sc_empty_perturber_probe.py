#!/usr/bin/env python3
r"""What SC-NEVPT2's guarded (empty) perturbers do to the energy's second
derivative in the orbitals -- measured offline on a committed golden, with
PySCF's own class formulas on the CPU (no card, no build).

A perturber tuple ``t`` whose norm is under the energy stage's
``NUMERICAL_ZERO`` guard is skipped by every class energy and, in the tree, by
every derivative of it. When all its external integrals vanish (``x_t = 0``,
an EMPTY tuple: on the committed goldens, by point-group symmetry) its
perturber is exactly zero. Along an orbital rotation that breaks that
symmetry, its integrals move at first order,
``x_t(s u + t v) = s x'_t[u] + t x'_t[v] + O(2)``, so its norm and Dyall
element are quadratic forms in ``(s, t)`` and

  E_t(s, t) = F_t(s u + t v) + O(3),   F_t(w) = -q_N(w)^2 / (q_N(w) Delta_t + q_H(w))

with ``q_N(w) = n_t[w, w]``, ``q_H(w) = h_t[w, w]`` the class's own norm and
H formulas evaluated at the first-order dotted integrals (the bilinear forms
``n_t[u, v]`` by polarisation, no step size). ``F_t`` is homogeneous of degree
two. It is a quadratic form -- so the energy is twice differentiable there --
exactly when the Rayleigh quotient ``q_H / q_N`` is constant on the plane,
i.e. when the 2x2 tables ``[h] = eta [n]``. docs/hessian-theory.md, section
5.6, has the derivation; this probe measures it.

For every class (``GOLDEN_ORDER``) and direction pair it prints:

``census``   tuples, guarded tuples, and how many of the guarded ones are
             EMPTY (every integral zero, so no CI direction can populate
             them either) or merely SMALL (a real perturber under the
             guard's threshold, which the guard zeroes in a neighbourhood
             of the reference, so it contributes nothing to any derivative
             of the guarded energy).
``pairs``    per (class, pair): the empty tuples each direction populates
             and both do, the worst proportionality defect ``rho`` (zero
             exactly when ``F_t`` is a quadratic form on the plane), the
             bilinear sixth term ``-N^{uv} / (Delta + eta)`` where it exists,
             the polarisation ``(F(u+v) - F(u-v)) / 2`` the four-point ladder
             converges to, the same quantity from a Richardson-extrapolated
             ladder of the empty tuples' energies along ``exp(s u + t v)``,
             and the parallelogram defect
             ``D2(u+v) + D2(u-v) - 2 D2(u) - 2 D2(v)`` of the directional
             second derivatives ``D2(w) = 2 F(w)`` (zero for a quadratic form).

The rotations are intra-space (core-core, virtual-virtual), so the active
blocks, the RDMs and every class's metric and Dyall block stay fixed and only
the stored external blocks and the orbital energies move -- the same
transformation the device's ``--sc-hessian-kk`` ladder applies. The
direction kinds are a random core-core rotation (``core``, ``core2``), a
random virtual-virtual one (``virt``, ``virt2``), and two controls built to be
proportional: ``virt+sym``, ``virt`` plus a random rotation inside the
generators that populate no empty tuple (a totally symmetric rotation,
found from the integrals, no irreps needed), and ``2virt``.

The measured numbers are in docs/testing.md, "The empty-perturber term,
offline".

Usage::

    uv run python reference_data/sc_empty_perturber_probe.py golden/n2_ccpvdz_cas44.nevpt2gold

Threads default to 4 (``OMP_NUM_THREADS`` etc. are only set when absent).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np  # noqa: E402
import pc_null_mode_probe as pnm  # noqa: E402
from pyscf.mrpt import nevpt2 as pyscf_nevpt2  # noqa: E402
from scipy.linalg import expm  # noqa: E402

#: The energy stage's vanishing-norm guard (src/energy/energy_finish.cppm,
#: NUMERICAL_ZERO; pyscf.mrpt.nevpt2's own).
NUMERICAL_ZERO = 1e-14
#: An empty tuple's first-order population ``n_t[w, w]`` (a unit direction's
#: h^2 coefficient) counts as populated above this. It is the same guard
#: applied to the coefficient; the probe prints the largest value seen below
#: it and the smallest above, so the choice is visible.
POPULATED = NUMERICAL_ZERO
#: A guarded tuple whose largest integral is at or below this is EMPTY (its
#: perturber is exactly zero, whatever the state); above it the tuple is
#: merely SMALL -- a real perturber under the guard's threshold, whose energy
#: the guard zeroes in a neighbourhood of the reference. The probe prints the
#: largest |x_t| on the empty tuples and the smallest on the small ones.
X_ZERO = 1e-12
#: The proportionality defect at or below which a (class, pair) is a
#: quadratic form on the plane and its bilinear sixth term is printed: the
#: host's suggestion for the device's refusal threshold, between the
#: proportional controls' round-off and the random pairs' defects (the
#: summary line prints both).
RHO_TOL = 1e-7
#: The golden's class order (generate_golden.py's CLASSES).
GOLDEN_ORDER = pnm.GOLDEN_ORDER
#: The classes with a Dyall element; Sijrs's is identically zero.
H_CLASSES = tuple(c for c in GOLDEN_ORDER if c != "Sijrs")

#: Which axes of each stored external block are core (``c``) or virtual
#: (``v``) indices -- the golden's layouts, generate_golden.py's transposes
#: of PySCF's eris. Active axes are absent: an intra-space rotation leaves
#: them alone.
AXES = {
    "h2e_v_Sr": ((0, "v"),),
    "h1e_v_Sr": ((0, "v"),),
    "h2e_v_Si": ((2, "c"),),
    "h1e_v_Si": ((1, "c"),),
    "cvcv": ((0, "c"), (1, "v"), (2, "c"), (3, "v")),
    "h2e_v_Sijr": ((0, "v"), (2, "c"), (3, "c")),
    "h2e_v_Srsi": ((0, "v"), (1, "v"), (2, "c")),
    "h2e_v_Srs": ((0, "v"), (1, "v")),
    "h2e_v_Sij": ((2, "c"), (3, "c")),
    "h2e_v1_Sir": ((0, "v"), (2, "c")),
    "h2e_v2_Sir": ((0, "v"), (3, "c")),
    "h1e_v_Sir": ((0, "v"), (1, "c")),
}


# --- the external blocks under an intra-space rotation ------------------------


def contract(X: np.ndarray, axis: int, M: np.ndarray) -> np.ndarray:
    """``X'[.., p, ..] = sum_q M[q, p] X[.., q, ..]``: one index of ``X``
    carried by the orbital transformation ``C -> C M``."""
    return np.moveaxis(np.tensordot(M, X, axes=([0], [axis])), 0, axis)


def rotate(blocks: dict, Uc: np.ndarray, Uv: np.ndarray) -> dict:
    """Every external block with its core and virtual indices rotated."""
    out = {}
    for name, axes in AXES.items():
        X = blocks[name]
        for ax, sp in axes:
            X = contract(X, ax, Uc if sp == "c" else Uv)
        out[name] = X
    return out


def dotted(blocks: dict, kc: np.ndarray, kv: np.ndarray) -> dict:
    """The first-order change of :func:`rotate` along ``exp(kappa)``: each
    rotated index in turn carried by ``kappa``, the rest fixed."""
    out = {}
    for name, axes in AXES.items():
        X = blocks[name]
        acc = np.zeros_like(X)
        for ax, sp in axes:
            acc += contract(X, ax, kc if sp == "c" else kv)
        out[name] = acc
    return out


def rotate_eps(e: np.ndarray, U: np.ndarray) -> np.ndarray:
    """The diagonal of ``U^T diag(e) U``: the orbital energies the rotated
    Fock carries, the reference block being diagonal (pseudocanonical)."""
    return (U * U).T @ e


def comb(a: dict, b: dict, wa: float, wb: float) -> dict:
    return {k: wa * a[k] + wb * b[k] for k in a}


# --- PySCF's SC class formulas, per tuple -------------------------------------


class Actives:
    """Every class's active-space intermediate (metrics and Dyall blocks) at
    the golden's CI vector and active integrals: fixed under an intra-space
    rotation."""

    def __init__(self, st: pnm.State):
        a, n, nel = st.a, st.n, st.nelec
        D = st.Q(st.c)
        h1e, h2e = a["h1e"], a["h2e"]
        d1, d2, d3 = D["1"], D["2"], D["3"]
        self.d1, self.d2, self.d3 = d1, d2, d3
        eye = np.eye(n)
        self.a16 = pyscf_nevpt2.make_a16(h1e, h2e, D, None, n, nel)
        self.a17 = pyscf_nevpt2.make_a17(h1e, h2e, d2, d3)
        self.a19 = pyscf_nevpt2.make_a19(h1e, h2e, d1, d2)
        self.a22 = pyscf_nevpt2.make_a22(h1e, h2e, D, None, n, nel)
        self.a23 = pyscf_nevpt2.make_a23(h1e, h2e, d1, d2, d3)
        self.a25 = pyscf_nevpt2.make_a25(h1e, h2e, d1, d2)
        self.dm3_h = 2 * np.einsum("abef,cd->abcdef", d2, eye) - d3.transpose(0, 1, 3, 2, 4, 5)
        self.dm2_h = 2 * np.einsum("ab,cd->abcd", d1, eye) - d2.transpose(0, 1, 3, 2)
        self.dm1_h = 2 * eye - d1.T
        self.hdm1 = pyscf_nevpt2.make_hdm1(d1)
        self.a3 = pyscf_nevpt2.make_a3(h1e, h2e, d1, d2, self.hdm1)
        self.k27 = pyscf_nevpt2.make_k27(h1e, h2e, d1, d2)
        self.rm2, self.a7 = pyscf_nevpt2.make_a7(h1e, h2e, d1, d2, d3)
        self.hdm2 = pyscf_nevpt2.make_hdm2(d1, d2)
        hdm3 = pyscf_nevpt2.make_hdm3(d1, d2, d3, self.hdm1, self.hdm2)
        self.a9 = pyscf_nevpt2.make_a9(h1e, h2e, self.hdm1, self.hdm2, hdm3)
        self.a12 = pyscf_nevpt2.make_a12(h1e, h2e, d1, d2, d3)
        self.a13 = pyscf_nevpt2.make_a13(h1e, h2e, d1, d2, d3)


def _pair_sym(m: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    """PySCF's symmetrisation of Sijr / Srsi over a pair of equal-space
    indices: ``m + m^T``, the diagonal halved."""
    out = m + np.swapaxes(m, *axes)
    n = m.shape[axes[0]]
    idx = [slice(None)] * m.ndim
    for k in range(n):
        idx[axes[0]] = idx[axes[1]] = k
        out[tuple(idx)] *= 0.5
    return out


def norm_h(cls: str, B: dict, A: Actives) -> tuple[np.ndarray, np.ndarray]:
    """``(N_t, H_t)`` over the class's tuples, flattened in PySCF's order
    (``pyscf.mrpt.nevpt2.Sr`` ... ``Sir`` transcribed onto the golden's
    blocks). Each is a homogeneous quadratic form in the blocks ``B``."""
    e = np.einsum
    if cls == "Sr":
        X, H1 = B["h2e_v_Sr"], B["h1e_v_Sr"]
        h = (e("ipqr,pqrabc,iabc->i", X, A.a16, X) + 2 * e("ipqr,pqra,ia->i", X, A.a17, H1)
             + e("ip,pa,ia->i", H1, A.a19, H1))
        n = (e("ipqr,rpqbac,iabc->i", X, A.d3, X) + 2 * e("ipqr,rpqa,ia->i", X, A.d2, H1)
             + e("ip,pa,ia->i", H1, A.d1, H1))
        return n, h
    if cls == "Si":
        X, H1 = B["h2e_v_Si"], B["h1e_v_Si"]
        h = (e("qpir,pqrabc,baic->i", X, A.a22, X) + 2 * e("qpir,pqra,ai->i", X, A.a23, H1)
             + e("pi,pa,ai->i", H1, A.a25, H1))
        n = (e("qpir,rpqbac,baic->i", X, A.dm3_h, X) + 2 * e("qpir,rpqa,ai->i", X, A.dm2_h, H1)
             + e("pi,pa,ai->i", H1, A.dm1_h, H1))
        return n, h
    if cls == "Sijrs":
        g = B["cvcv"]  # (i, a, j, b)
        n = g * (2 * g - g.transpose(0, 3, 2, 1))
        return n.reshape(-1), np.zeros(n.size)
    if cls == "Sijr":
        X = B["h2e_v_Sijr"]  # (r, p, j, i)
        tri = np.triu_indices(X.shape[2])

        def f(m):
            return (2 * e("rpji,raji,pa->rji", X, X, m) - e("rpji,raij,pa->rji", X, X, m))
        n, h = _pair_sym(f(A.hdm1), (1, 2)), _pair_sym(f(A.a3), (1, 2))
        return n[:, tri[0], tri[1]].reshape(-1), h[:, tri[0], tri[1]].reshape(-1)
    if cls == "Srsi":
        X = B["h2e_v_Srsi"]  # (r, s, i, p)
        tri = np.triu_indices(X.shape[0])

        def f(m):
            return (2 * e("rsip,rsia,pa->rsi", X, X, m) - e("rsip,sria,pa->rsi", X, X, m))
        n, h = _pair_sym(f(A.d1), (0, 1)), _pair_sym(f(A.k27), (0, 1))
        return n[tri].reshape(-1), h[tri].reshape(-1)
    if cls == "Srs":
        X = B["h2e_v_Srs"]
        n = 0.5 * e("rsqp,rsba,pqba->rs", X, X, A.rm2)
        h = 0.5 * e("rsqp,rsba,pqab->rs", X, X, A.a7)
        return n.reshape(-1), h.reshape(-1)
    if cls == "Sij":
        X = B["h2e_v_Sij"]
        n = 0.5 * e("qpij,baij,pqab->ij", X, X, A.hdm2)
        h = 0.5 * e("qpij,baij,pqab->ij", X, X, A.a9)
        return n.reshape(-1), h.reshape(-1)
    if cls == "Sir":
        X1, X2, H1 = B["h2e_v1_Sir"], B["h2e_v2_Sir"], B["h1e_v_Sir"]
        d1, d2 = A.d1, A.d2
        n = (2 * e("rpiq,raib,qpab->ir", X1, X1, d2) - e("rpiq,rabi,qpab->ir", X1, X2, d2)
             - e("rpqi,raib,qpab->ir", X2, X1, d2) + 2 * e("raqi,rabi,qb->ir", X2, X2, d1)
             - e("rpqi,rabi,qbap->ir", X2, X2, d2) + e("rpqi,raai,qp->ir", X2, X2, d1)
             + 4 * e("rpiq,ri,qp->ir", X1, H1, d1) - 2 * e("rpqi,ri,qp->ir", X2, H1, d1)
             + 2 * e("ri,ri->ir", H1, H1))
        h = (2 * e("rpiq,raib,pqab->ir", X1, X1, A.a12) - e("rpiq,rabi,pqab->ir", X1, X2, A.a12)
             - e("rpqi,raib,pqab->ir", X2, X1, A.a12) + e("rpqi,rabi,pqab->ir", X2, X2, A.a13))
        return n.reshape(-1), h.reshape(-1)
    raise ValueError(cls)


def deltas(cls: str, ec: np.ndarray, ev: np.ndarray) -> np.ndarray:
    """``Delta_t`` in :func:`norm_h`'s order, the sign PySCF hands
    ``_norm_to_energy`` (so ``E_t = -N / (Delta + H/N)``)."""
    if cls == "Sr":
        return ev.copy()
    if cls == "Si":
        return -ec
    if cls == "Sijrs":
        return (ev[None, :, None, None] + ev[None, None, None, :] - ec[:, None, None, None]
                - ec[None, None, :, None]).reshape(-1)
    if cls == "Sijr":
        d = ev[:, None, None] - ec[None, :, None] - ec[None, None, :]
        tri = np.triu_indices(ec.size)
        return d[:, tri[0], tri[1]].reshape(-1)
    if cls == "Srsi":
        d = ev[:, None, None] + ev[None, :, None] - ec[None, None, :]
        return d[np.triu_indices(ev.size)].reshape(-1)
    if cls == "Srs":
        return (ev[:, None] + ev[None, :]).reshape(-1)
    if cls == "Sij":
        return -(ec[:, None] + ec[None, :]).reshape(-1)
    if cls == "Sir":
        return -(ec[:, None] - ev[None, :]).reshape(-1)
    raise ValueError(cls)


def tuple_energies(cls: str, n: np.ndarray, h: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Per-tuple ``E_t``: ``_norm_to_energy``'s guard for the H classes, and
    Sijrs's ``-N/Delta`` with no guard (PySCF's Sijrs has none: it is linear
    in N)."""
    if cls == "Sijrs":
        return -n / d
    out = np.zeros_like(n)
    keep = np.abs(n) > NUMERICAL_ZERO
    out[keep] = -n[keep] / (d[keep] + h[keep] / n[keep])
    return out


def integral_max(cls: str, B: dict) -> np.ndarray:
    """The largest |integral| feeding each tuple, in :func:`norm_h`'s order:
    zero means the tuple is empty by its integrals, whatever the state."""
    a = np.abs
    if cls == "Sr":
        return np.maximum(a(B["h2e_v_Sr"]).max(axis=(1, 2, 3)), a(B["h1e_v_Sr"]).max(axis=1))
    if cls == "Si":
        return np.maximum(a(B["h2e_v_Si"]).max(axis=(0, 1, 3)), a(B["h1e_v_Si"]).max(axis=0))
    if cls == "Sijrs":
        g = a(B["cvcv"])
        return np.maximum(g, g.transpose(0, 3, 2, 1)).reshape(-1)
    if cls == "Sijr":
        m = a(B["h2e_v_Sijr"]).max(axis=1)  # (r, j, i)
        m = np.maximum(m, m.transpose(0, 2, 1))
        tri = np.triu_indices(m.shape[1])
        return m[:, tri[0], tri[1]].reshape(-1)
    if cls == "Srsi":
        m = a(B["h2e_v_Srsi"]).max(axis=3)  # (r, s, i)
        m = np.maximum(m, m.transpose(1, 0, 2))
        return m[np.triu_indices(m.shape[0])].reshape(-1)
    if cls == "Srs":
        return a(B["h2e_v_Srs"]).max(axis=(2, 3)).reshape(-1)
    if cls == "Sij":
        return a(B["h2e_v_Sij"]).max(axis=(0, 1)).reshape(-1)
    if cls == "Sir":
        m1 = a(B["h2e_v1_Sir"]).max(axis=(1, 3))  # (r, i)
        m2 = a(B["h2e_v2_Sir"]).max(axis=(1, 2))
        return np.maximum(np.maximum(m1, m2), a(B["h1e_v_Sir"])).T.reshape(-1)
    raise ValueError(cls)


# --- the empty tuples' second-order algebra ------------------------------------


def cross_tables(cls: str, xu: dict, xv: dict, A: Actives) -> dict:
    """``n[u,u], n[u,v], n[v,v]`` and the same for ``h``, per tuple, by
    polarisation: each is a quadratic form in the blocks, so
    ``B(a, b) = (Q(a + b) - Q(a - b)) / 4`` exactly."""
    nu, hu = norm_h(cls, xu, A)
    nv, hv = norm_h(cls, xv, A)
    np_, hp = norm_h(cls, comb(xu, xv, 1, 1), A)
    nm, hm = norm_h(cls, comb(xu, xv, 1, -1), A)
    return dict(nuu=nu, nvv=nv, nuv=(np_ - nm) / 4, huu=hu, hvv=hv, huv=(hp - hm) / 4)


def populated(n: np.ndarray) -> np.ndarray:
    """A first-order population above the guard. Sijrs's per-tuple ``N`` is
    ``g (2g - g')`` and may be negative, so the test is on ``|n|``; every
    other class's metric is positive semi-definite."""
    return np.abs(n) > POPULATED


def F(qn: np.ndarray, qh: np.ndarray, d: np.ndarray) -> np.ndarray:
    """``-q_N^2 / (q_N Delta + q_H)``: an empty tuple's energy at unit step
    along a direction whose first-order tables are ``(q_N, q_H)``; zero where
    the direction does not populate it."""
    out = np.zeros_like(qn)
    keep = populated(qn)
    out[keep] = -qn[keep] ** 2 / (qn[keep] * d[keep] + qh[keep])
    return out


def proportionality(T: dict, d: np.ndarray) -> np.ndarray:
    """Per tuple, the defect ``rho`` of ``[h] = eta [n]`` on the plane, in a
    basis where both directions have unit population (so it is invariant
    under rescaling either one): with ``c = n_uv / sqrt(n_uu n_vv)``,
    ``eta_w = h_ww / n_ww`` and ``k = h_uv / sqrt(n_uu n_vv)``,

      rho = max(|eta_u - eta_v|, 2 |k - c (eta_u + eta_v)/2|) / |Delta + eta|

    -- zero exactly when the Rayleigh quotient is constant on the plane, and
    the relative size of the empty tuple's departure from a quadratic form.
    Only tuples both directions populate are scored (elsewhere the mixed
    term vanishes and the energy is a quadratic form along the pair)."""
    both = populated(T["nuu"]) & populated(T["nvv"])
    rho = np.zeros_like(d)
    if not both.any():
        return rho
    nuu, nvv, nuv = T["nuu"][both], T["nvv"][both], T["nuv"][both]
    eu, ev = T["huu"][both] / nuu, T["hvv"][both] / nvv
    s = np.sqrt(np.abs(nuu * nvv))
    c, k = nuv / s, T["huv"][both] / s
    eta = 0.5 * (eu + ev)
    rho[both] = np.maximum(np.abs(eu - ev), 2 * np.abs(k - c * eta)) / np.abs(d[both] + eta)
    return rho


def sixth_term(T: dict, d: np.ndarray) -> dict:
    """The empty tuples' share of the mixed second derivative, three ways.

    ``bilinear``  ``-N^{uv} / (Delta + eta)`` with ``N^{uv} = 2 n_uv`` (the
                  doc's K4 convention, both slots symmetrised) and ``eta``
                  the Rayleigh quotient -- K4 at the ray-limit multipliers.
                  Meaningful only where ``rho`` is zero.
    ``polar``     ``(F(u+v) - F(u-v)) / 2``: what the four-point ladder
                  converges to, whether or not F is a quadratic form.
    ``parallelogram``  ``D2(u+v) + D2(u-v) - 2 D2(u) - 2 D2(v)`` with
                  ``D2(w) = 2 F(w)``, the directional second derivatives:
                  zero for a quadratic form, and the plain test that the
                  second derivative is not one."""
    qn = {k: T["nuu"] + sgn * 2 * T["nuv"] + T["nvv"] for k, sgn in (("p", 1), ("m", -1))}
    qh = {k: T["huu"] + sgn * 2 * T["huv"] + T["hvv"] for k, sgn in (("p", 1), ("m", -1))}
    fp, fm = F(qn["p"], qh["p"], d), F(qn["m"], qh["m"], d)
    fu, fv = F(T["nuu"], T["huu"], d), F(T["nvv"], T["hvv"], d)
    pop = populated(T["nuu"])
    eta = np.where(pop, T["huu"] / np.where(pop, T["nuu"], 1.0), 0.0)
    popv = populated(T["nvv"])
    eta = np.where(pop, eta, np.where(popv, T["hvv"] / np.where(popv, T["nvv"], 1.0), 0.0))
    bil = np.where(pop & popv, -2 * T["nuv"] / (d + eta), 0.0)
    return dict(bilinear=bil.sum(), polar=0.5 * (fp - fm).sum(),
                parallelogram=(2 * fp + 2 * fm - 4 * fu - 4 * fv).sum())


# --- the probe ----------------------------------------------------------------


class Case:
    def __init__(self, path):
        self.st = pnm.State(path)
        a = self.st.a
        self.nc, self.nv = a["e_core"].size, a["e_virt"].size
        self.blocks = {k: a[k] for k in AXES}
        self.ec, self.ev = a["e_core"], a["e_virt"]
        self.A = Actives(self.st)
        self.n0, self.h0, self.d0, self.guard, self.xmax, self.empty = {}, {}, {}, {}, {}, {}
        for cls in GOLDEN_ORDER:
            n, h = norm_h(cls, self.blocks, self.A)
            self.n0[cls], self.h0[cls] = n, h
            self.d0[cls] = deltas(cls, self.ec, self.ev)
            self.guard[cls] = np.abs(n) <= NUMERICAL_ZERO
            self.xmax[cls] = integral_max(cls, self.blocks)
            self.empty[cls] = self.guard[cls] & (self.xmax[cls] <= X_ZERO)

    def class_energies(self) -> np.ndarray:
        return np.array([tuple_energies(c, self.n0[c], self.h0[c], self.d0[c]).sum()
                         for c in GOLDEN_ORDER])

    def empty_energy(self, cls: str, kc: np.ndarray, kv: np.ndarray) -> float:
        """The reference's empty tuples' energy at ``exp(kappa)``, with the
        rotated orbital energies."""
        Uc, Uv = expm(kc), expm(kv)
        n, h = norm_h(cls, rotate(self.blocks, Uc, Uv), self.A)
        d = deltas(cls, rotate_eps(self.ec, Uc), rotate_eps(self.ev, Uv))
        g = self.empty[cls]
        return tuple_energies(cls, n[g], h[g], d[g]).sum()


def antisym(rng, n: int, mask: np.ndarray | None = None) -> np.ndarray:
    a = rng.standard_normal((n, n))
    if mask is not None:
        a = a * mask
    a = a - a.T
    return a / np.linalg.norm(a)


def symmetric_mask(case: Case, space: str) -> np.ndarray:
    """The generators ``(p, q)`` of one space whose rotation populates no
    empty tuple of any class: the totally symmetric ones (same irrep),
    read off the integrals."""
    n = case.nc if space == "c" else case.nv
    mask = np.zeros((n, n))
    zc, zv = np.zeros((case.nc, case.nc)), np.zeros((case.nv, case.nv))
    for p in range(n):
        for q in range(p + 1, n):
            g = np.zeros((n, n))
            g[p, q], g[q, p] = 1.0, -1.0
            x = dotted(case.blocks, g, zv) if space == "c" else dotted(case.blocks, zc, g)
            worst = max(np.abs(norm_h(c, x, case.A)[0][case.empty[c]]).max(initial=0.0)
                        for c in GOLDEN_ORDER)
            if worst <= POPULATED:
                mask[p, q] = mask[q, p] = 1.0
    return mask


def ladder(case: Case, cls: str, u: tuple, v: tuple, pows) -> float:
    """Richardson-extrapolated four-point mixed difference of the empty
    tuples' energy along ``exp(s u + t v)`` (``u``, ``v`` as (kc, kv))."""
    vals = []
    for p in pows:
        hh = 2.0**-p
        acc = 0.0
        for s, t, w in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            kc = s * hh * u[0] + t * hh * v[0]
            kv = s * hh * u[1] + t * hh * v[1]
            acc += w * case.empty_energy(cls, kc, kv)
        vals.append(acc / (4 * hh * hh))
    r1 = [(4 * b - a) / 3 for a, b in zip(vals, vals[1:], strict=False)]
    return (16 * r1[-1] - r1[-2]) / 15


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("golden")
    ap.add_argument("--seed", type=int, default=190)
    ap.add_argument("--pows", default="5,6,7", help="ladder steps h = 2^-p")
    args = ap.parse_args(argv)
    t0 = time.time()
    case = Case(args.golden)
    want = case.st.a["class_energies"]
    got = case.class_energies()
    print(f"{args.golden}: ncore {case.nc}, nvirt {case.nv}; class energies against the "
          f"golden's: worst |delta| {np.abs(got - want).max():.1e}; "
          f"threads {os.environ['OMP_NUM_THREADS']}")

    print("\n[census] guarded tuples (|N_t| <= NUMERICAL_ZERO): EMPTY ones have every integral "
          "zero (x_t = 0), SMALL ones do not")
    print(f"  {'class':6} {'tuples':>7} {'guarded':>8} {'empty':>8} {'small':>6} "
          f"{'max |x_t| empty':>16} {'min |x_t| small':>16}")
    for cls in GOLDEN_ORDER:
        g, e = case.guard[cls], case.empty[cls]
        small = g & ~e
        xe = case.xmax[cls][e].max(initial=0.0)
        xs = f"{case.xmax[cls][small].min():16.1e}" if small.any() else f"{'-':>16}"
        print(f"  {cls:6} {g.size:7d} {int(g.sum()):8d} {int(e.sum()):8d} {int(small.sum()):6d} "
              f"{xe:16.1e} {xs}")

    rng = np.random.default_rng(args.seed)
    zc, zv = np.zeros((case.nc, case.nc)), np.zeros((case.nv, case.nv))
    symv = symmetric_mask(case, "v")
    dirs = {
        "core": (antisym(rng, case.nc), zv),
        "core2": (antisym(rng, case.nc), zv),
        "virt": (zc, antisym(rng, case.nv)),
        "virt2": (zc, antisym(rng, case.nv)),
    }
    wsym = antisym(rng, case.nv, symv) if symv.any() else zv
    dirs["virt+sym"] = (zc, dirs["virt"][1] + wsym)
    dirs["2virt"] = (zc, 2 * dirs["virt"][1])
    print(f"\n  totally symmetric virtual generators (populate no empty tuple): "
          f"{int(symv.sum() / 2)} of {case.nv * (case.nv - 1) // 2}")
    pairs = [("core", "core2"), ("core", "virt"), ("virt", "virt2"),
             ("virt", "virt+sym"), ("virt", "2virt")]
    pows = [int(p) for p in args.pows.split(",")]
    xd = {k: dotted(case.blocks, *w) for k, w in dirs.items()}

    print("\n[pairs] the empty tuples' share of Eddot_kk(u, v)")
    hdr = (f"  {'pair':16} {'class':6} {'u':>4} {'v':>4} {'both':>4} {'max rho':>9} "
           f"{'bilinear':>13} {'polar':>13} {'ladder':>13} {'parallelogram':>13}")
    print(hdr)
    lo_pop, hi_noise = np.inf, 0.0
    rho_control, rho_real, worst_ladder = 0.0, np.inf, 0.0
    for pu, pv in pairs:
        for cls in GOLDEN_ORDER:
            g = case.empty[cls]
            T = {k: v[g] for k, v in cross_tables(cls, xd[pu], xd[pv], case.A).items()}
            d = case.d0[cls][g]
            if cls in H_CLASSES:  # Sijrs's N_t = g (2g - g') can sit anywhere near zero
                for key in ("nuu", "nvv"):
                    mag = np.abs(T[key])
                    if populated(mag).any():
                        lo_pop = min(lo_pop, mag[populated(mag)].min())
                    if (~populated(mag)).any():
                        hi_noise = max(hi_noise, mag[~populated(mag)].max())
            nu = int(populated(T["nuu"]).sum())
            nv = int(populated(T["nvv"]).sum())
            both = int((populated(T["nuu"]) & populated(T["nvv"])).sum())
            if not (nu or nv):
                continue
            rho = proportionality(T, d).max(initial=0.0)
            six = sixth_term(T, d)
            lad = ladder(case, cls, dirs[pu], dirs[pv], pows)
            worst_ladder = max(worst_ladder, abs(lad - six["polar"]))
            if both and cls in H_CLASSES:
                if pv in ("virt+sym", "2virt"):
                    rho_control = max(rho_control, rho)
                else:
                    rho_real = min(rho_real, rho)
            bil = f"{six['bilinear']:13.6e}" if rho <= RHO_TOL else f"{'-':>13}"
            print(f"  {pu + ' x ' + pv:16} {cls:6} {nu:4d} {nv:4d} {both:4d} "
                  f"{rho:9.1e} {bil} {six['polar']:13.6e} {lad:13.6e} "
                  f"{six['parallelogram']:13.6e}")
    print(f"\n  first-order population (H classes): smallest above POPULATED {lo_pop:.1e}, "
          f"largest at or below it {hi_noise:.1e}")
    print(f"  rho (H classes, tuples both populate): proportional controls at most "
          f"{rho_control:.1e}, random pairs at least {rho_real:.1e}  (RHO_TOL {RHO_TOL:.0e})")
    print(f"  |ladder - polar|, worst: {worst_ladder:.1e}")

    print("\n[first order] what the guard drops from Edot along each direction: Sijrs's "
          "-Ndot/Delta and the H classes' largest |Ndot| on their guarded tuples (zero on "
          "empty ones, whose x_t vanishes)")
    for name in ("core", "virt"):
        x0, xw = case.blocks, xd[name]
        sij, worst = 0.0, 0.0
        for cls in GOLDEN_ORDER:
            g = case.guard[cls]
            ndot = (norm_h(cls, comb(x0, xw, 1, 1), case.A)[0]
                    - norm_h(cls, comb(x0, xw, 1, -1), case.A)[0]) / 2
            if cls == "Sijrs":
                sij = abs((-ndot[g] / case.d0[cls][g]).sum())
            else:
                worst = max(worst, np.abs(ndot[g]).max(initial=0.0))
        print(f"  {name:5} Sijrs |sum -Ndot/Delta| {sij:.1e}   H classes max |Ndot| {worst:.1e}")
    print(f"wall {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
