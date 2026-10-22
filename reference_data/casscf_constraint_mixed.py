#!/usr/bin/env python3
r"""The CASSCF constraint's **R-mixed** terms, offline: the gradient's
multipliers contracted with the constraint's parameter-coordinate and
coordinate-coordinate second derivatives, ``Z . d^2 g / dtheta dR`` and
``Z . d^2 g / dR^2`` (docs/hessian-theory.md, sections 7 and 8).

THE OBJECT
----------

The constraint is ``pyscf.mcscf.newton_casscf.gen_g_hop``'s ``g_all`` (the
CASSCF orbital and CI gradient, ``nevpt2_grad.py``'s (S1) and (C1)), as a
function of ``gen_g_hop``'s step variables ``theta = (x, y)`` **and** the
geometry ``R``, along #215's frozen-connection path (``casscf_hessian.py``)::

    C(theta, R) = C_f(R) exp(K(x)),   C_f(R) = S(R)^{-1/2} S(R0)^{1/2} C0
    c(theta)    = (c0 + y) / |c0 + y|                    (fixed CI: no R in it)

and ``Z . g_all`` there is the energy's slope along the multipliers stepped
from that point (``casscf_third_derivative.py``, "THE OBJECT")::

    Z . g(theta, R) = d/ds E_R( C(theta, R) exp(s K(Z)), norm(c(theta) + s z) )

The two terms this module delivers, with ``(Z, z)`` undifferentiated (the
2n+2 rule, section 7):

* ``Z . d^2 g / dtheta dR_A [u]``: an **action** on a full-range ``theta``
  direction ``u = (kappa, c)``, one number per Cartesian coordinate ``A``
  (:meth:`Constraint.mixed`);
* ``Z . d^2 g / dR_A dR_B``: a symmetric ``(3N, 3N)`` matrix
  (:meth:`Constraint.second`).

HOW: Z-WEIGHTED DENSITIES THROUGH #215's DRIVERS
------------------------------------------------

Write the orbitals as ``C_f(R) U`` with ``U`` orthogonal, ``U_o`` its
occupied columns. Because ``frozen_coeff`` is linear in the orbitals it
carries (``C_f(R; C0 U) = C_f(R; C0) U``), the energy at ``(U, c)`` and any
``R`` is::

    E_R(U, c) = E_nuc(R) + sum_pq h~_pq(R) P_pq + 1/2 sum_pqrs g~_pqrs(R) Gamma_pqrs
    P = U_o D(c) U_o^T,   Gamma = U_o^(x4) G(c)

with ``h~(R)``, ``g~(R)`` the MO integrals of ``C_f(R)`` (#215's path) and
``P``, ``Gamma`` the CASSCF densities carried into the **full** MO range of
``C0``. ``E_R`` is linear in ``(h~, g~)`` at fixed ``(U, c)``, and ``R``
enters only there. So every derivative this module needs is #215's
derivative of the path energy -- its perturbed Hamiltonians
(:meth:`casscf_hessian.Skeleton.gradient`) or its second partial ``L_RR``
(:meth:`casscf_hessian.Skeleton.hessian`) -- at **other densities**, with
``E_nuc``'s term removed (it carries no ``s``):

* ``Z . dg/dR`` and ``Z . d^2 g / dR^2``: the densities' ``s`` coefficient,
  ``(P^Z, Gamma^Z)``, with ``U = exp(s K(Z))``, ``c = norm(c0 + s z)``;
* ``Z . d^2 g / dtheta dR [u]``: their ``a s`` coefficient, ``(P^{uZ},
  Gamma^{uZ})``, with ``U = exp(a K(u)) exp(s K(Z))``, ``c = norm(norm(c0 +
  a u_c) + s z)``.

The densities are carried as ``casscf_third_derivative.py``'s exact
truncated Taylor series in nilpotent units (:func:`ctd.hd_mul`; ``a`` mask
1, ``s`` mask 4), so nothing is differenced. ``D(c)``, ``G(c)`` are
``casscf_hessian.full_rdms`` -- quadratic in ``c``, with a core part that is
``<c|c>`` times a constant; on the series ``<c|c> = 1`` exactly, so every
coefficient past the constant is the **bilinear** part alone, read off by
polarisation (:meth:`Constraint.rdm_bilinear`). The densities are symmetric
and ``Gamma`` has the integrals' eightfold symmetry, which is all #215's
formulas assume; they are nonzero on rows outside the occupied range (a
rotation mixes virtuals in), so they are handed to ``casscf_hessian`` as a
``Reference`` whose "occupied" range is the whole MO space
(:meth:`Constraint.skeleton`). Nothing here assumes stationarity.

THE ORACLE: LADDERS OF ``Z . g`` ITSELF
---------------------------------------

As #216 found, ``gen_g_hop``'s ``h_op`` is the Jacobian of ``g_all`` only at
a stationary point, so the oracle differences ``g_all`` itself: at every
point ``(theta, R)``, :func:`zg` builds the molecule at ``R``, ``C_f(R)``
through ``nevpt2_grad.frozen_coeff``, applies ``exp(K(theta))`` and the CI
step, and calls ``gen_g_hop``. Then:

* ``Z . d^2 g / dtheta dR [u, d]`` against the four-point quotient
  ``Z . [g(h u, R + h d) - g(-h u, R + h d) - g(h u, R - h d)
  + g(-h u, R - h d)] / 4 h^2``;
* ``d^T (Z . d^2 g / dR^2) d`` against the second difference of ``Z . g(0,
  R + s h d)``, and ``d^T (...) d'`` against the four-point quotient along
  two directions;
* (``--columns``, on by default) each kept direction's column ``(Z . d^2 g /
  dR^2) d`` against the central difference of the **analytic** ``Z .
  dg/dR`` re-anchored on the path at ``R +- h d``
  (:func:`casscf_hessian.path_gradient`'s consistency argument): every
  element of the column at once, :func:`casscf_hessian.judge_column`.

Each is gated by ``nevpt2_grad.ladder_check``'s run rule and Richardson
extrapolant at ``nevpt2_grad.LADDER_TOL``. Directions ``d``: on a molecule
with point-group symmetry, the point-group-keeping ones
(``casscf_nuclear_hessian.keeps_point_group``; N2's stretch, LiF's two axial
coordinates and its stretch), on the C1 case every Cartesian coordinate.

The exact identities, read to round-off (:data:`IDENTITY_TOL`): the
densities rebuild ``E``, ``Z . g_all`` (``gen_g_hop``) and ``Z . J[u]``
(``casscf_third_derivative.Third``'s ``a s`` coefficient) at the reference
integrals; ``Z . dg/dR`` equals ``Z`` against
``casscf_nuclear_hessian.mixed_derivatives`` (#219's independent formula);
``Z . d^2 g / dR^2`` is symmetric; and both R-terms obey the translational
sum rule (a rigid translation moves no overlap, so the path's integrals are
invariant along it).

Usage::

    uv run python reference_data/casscf_constraint_mixed.py --case n2_ccpvdz_cas44 --workers 6
    uv run python reference_data/casscf_constraint_mixed.py --case lif_def2svp_cas46 --workers 6
    uv run python reference_data/casscf_constraint_mixed.py --case hsoh_def2svp_cas66 --workers 6
    uv run python reference_data/casscf_constraint_mixed.py --case n2_ccpvdz_cas44 \
        --seeded-multipliers          # no NEVPT2: a seeded (Z, z)

The multipliers are by default the SC-NEVPT2 gradient's own, solved by
``casscf_third_derivative.multipliers`` (``nevpt2_grad.CasscfGradients``).
The measured numbers are in docs/reference-data.md, "The CASSCF constraint's
R-mixed terms". Nothing is written; no committed file is read except through
``nevpt2_grad.build_state``.
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads
# (docs/reference-data.md, "Reproducibility").
import generate_golden as gg  # noqa: F401

# isort: split
import casscf_hessian as ch
import casscf_nuclear_hessian as cnh
import casscf_third_derivative as ctd
import nevpt2_grad as ngr
import nevpt2_numgrad as ngrad

# isort: split
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from pyscf.grad import rhf as rhf_grad
from scipy.linalg import expm

#: The ladders of ``Z . g`` (four-point and second differences), ``h = 2^-k``
#: along unit directions: #216's range, whose fine end keeps the quotients'
#: ``eps / h^2`` round-off two decades under the gate.
LADDER = (3, 11)

#: The column ladders: central differences of the analytic ``Z . dg/dR``,
#: whose round-off grows only as ``1/h`` (``casscf_hessian.COLUMN_LADDER``).
COLUMN_LADDER = ch.COLUMN_LADDER

#: The gate on ``|Richardson - analytic|``: ``nevpt2_grad.LADDER_TOL``.
LADDER_TOL = ngr.LADDER_TOL

#: The exact identities (density rebuilds, #219's formula, symmetry, the
#: translational sum rules), read to round-off.
IDENTITY_TOL = 1e-9

#: The ``theta`` directions of the mixed term: one seeded unit direction per
#: kind and seed (``casscf_third_derivative.direction``'s blocks).
THETA_KINDS = ("orbital", "ci", "full")
THETA_SEEDS = (0, 1)

#: The cases the acceptance names (``_casscf`` goldens with a ``.mogold``).
CASES = ("n2_ccpvdz_cas44", "lif_def2svp_cas46", "hsoh_def2svp_cas66")

#: The seed of ``--seeded-multipliers``' ``(Z, z)`` (#216's).
MULTIPLIER_SEED = ctd.MULTIPLIER_SEED


# --------------------------------------------------------------------------
# The Z-weighted densities and the two terms
# --------------------------------------------------------------------------


class Constraint:
    """The R-mixed constraint terms at one CASSCF (or any) state: orbitals
    ``C0`` at ``geom`` (square, S-orthonormal), CI vector ``ci``, and
    ``gen_g_hop``'s variables (:class:`casscf_third_derivative.Third`'s)."""

    def __init__(self, geom, basis: str, C0: np.ndarray, ci: np.ndarray, ncore: int,
                 ncas: int, nelecas):
        self.geom, self.basis = geom, basis
        self.th = ctd.Third(ch.mol_at(geom, basis), C0, ci, ncore, ncas, nelecas)
        th = self.th
        self.ncore, self.ncas, self.nelecas = ncore, ncas, th.nelecas
        self.nocc, self.nmo, self.nvar = th.nocc, th.nmo, th.nvar
        D, G = ch.full_rdms(ncore, ncas, th.nelecas, th.ci)
        self.ref = ch.Reference(geom=geom, basis=basis, C0=th.C0, ncore=ncore, ncas=ncas,
                                D=D, G=G)
        self.sk = ch.Skeleton(self.ref)
        self.natm, self.ncoord = self.sk.natm, self.sk.ncoord
        self.grad_nuc = np.asarray(rhf_grad.grad_nuc(self.sk.mol)).ravel()

    # -- the densities as series -------------------------------------------

    def rdm_bilinear(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The symmetric bilinear part of ``casscf_hessian.full_rdms`` at
        ``(x, y)``, by polarisation: ``(Q(x + y) - Q(x - y)) / 4`` -- exact,
        ``Q`` being a constant plus a quadratic form."""
        shape = self.th.ci.shape
        Dp, Gp = ch.full_rdms(self.ncore, self.ncas, self.nelecas, (x + y).reshape(shape))
        Dm, Gm = ch.full_rdms(self.ncore, self.ncas, self.nelecas, (x - y).reshape(shape))
        return 0.25 * (Dp - Dm), 0.25 * (Gp - Gm)

    def densities(self, Z: np.ndarray, u: np.ndarray | None = None) -> dict:
        """``(P, Gamma)`` over the full MO range of ``C0``, as a series in the
        units ``a`` (mask 1, along ``u``; absent when ``u`` is ``None``) and
        ``s`` (mask 4, along ``Z`` from the displaced point): mask 0 is the
        state's own, mask 4 ``(P^Z, Gamma^Z)``, mask 5 ``(P^{uZ},
        Gamma^{uZ})``."""
        th, n, nocc = self.th, self.nvar, self.nocc
        eye = np.eye(self.nmo)
        c0 = th.ci.ravel()
        if u is None:
            U = {0: eye, 4: th.kappa(Z)}
            c = ctd.hd_normalise({0: c0, 4: np.asarray(Z[n:], dtype=float)})
        else:
            U = ctd.hd_mul(lambda p, q: p @ q, ctd.hd_expm({1: th.kappa(u)}, self.nmo),
                           {0: eye, 4: th.kappa(Z)})
            c = ctd.hd_normalise({0: c0, 1: np.asarray(u[n:], dtype=float)})
            c = ctd.hd_normalise(ctd.hd_add(c, {4: np.asarray(Z[n:], dtype=float)}))
        # the CI densities: the state's at mask 0, the bilinear part elsewhere
        dens: dict = {0: (self.ref.D, self.ref.G)}
        masks = sorted(c)
        for i, m1 in enumerate(masks):
            for m2 in masks[i + 1:]:
                if m1 & m2:
                    continue
                Db, Gb = self.rdm_bilinear(c[m1], c[m2])
                m = m1 | m2
                prev = dens.get(m)
                dens[m] = ((2.0 * Db, 2.0 * Gb) if prev is None
                           else (prev[0] + 2.0 * Db, prev[1] + 2.0 * Gb))
        Uo = {m: x[:, :nocc] for m, x in U.items()}
        P = ctd.hd_mul(lambda p, q: p @ q, Uo, {m: d[0] for m, d in dens.items()})
        P = ctd.hd_mul(lambda p, q: p @ q.T, P, Uo)
        g: dict = {m: d[1] for m, d in dens.items()}
        for ax in range(4):
            g = ctd.hd_mul(lambda Gm, Um, ax=ax: np.moveaxis(
                np.tensordot(Um, Gm, axes=([1], [ax])), 0, ax), g, Uo)
        out = {}
        for m in P:
            Pm = 0.5 * (P[m] + P[m].T)
            Gm = g[m]
            Gm = 0.5 * (Gm + Gm.transpose(1, 0, 3, 2))
            Gm = 0.5 * (Gm + Gm.transpose(2, 3, 0, 1))
            Gm = 0.5 * (Gm + Gm.transpose(1, 0, 2, 3))
            Gm = 0.5 * (Gm + Gm.transpose(0, 1, 3, 2))
            out[m] = (Pm, Gm)
        return out

    def contract(self, P: np.ndarray, Gamma: np.ndarray) -> float:
        """``sum h~ P + 1/2 sum g~ Gamma`` at the reference integrals."""
        return float(np.einsum("pq,pq", self.th.h_mo, P)
                     + 0.5 * np.einsum("pqrs,pqrs", self.th.eri_mo, Gamma))

    # -- #215's drivers at the Z-weighted densities -------------------------

    def skeleton(self, P: np.ndarray, Gamma: np.ndarray) -> ch.Skeleton:
        """#215's :class:`casscf_hessian.Skeleton` at densities over the whole
        MO range: its "occupied" range is all of it."""
        return ch.Skeleton(ch.Reference(geom=self.geom, basis=self.basis, C0=self.th.C0,
                                        ncore=0, ncas=self.nmo, D=P, G=Gamma))

    def z_gradient(self, Z: np.ndarray) -> np.ndarray:
        """``Z . dg/dR``, ``(3N,)``."""
        P, Gamma = self.densities(Z)[4]
        return self.skeleton(P, Gamma).gradient().ravel() - self.grad_nuc

    def mixed(self, u: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """``Z . d^2 g / dtheta dR_A [u]`` for every coordinate ``A``,
        ``(3N,)``."""
        P, Gamma = self.densities(Z, u)[5]
        return self.skeleton(P, Gamma).gradient().ravel() - self.grad_nuc

    def second(self, Z: np.ndarray) -> tuple[np.ndarray, dict]:
        """``Z . d^2 g / dR_A dR_B``, ``(3N, 3N)``, and its terms by
        ``casscf_hessian.Skeleton.hessian``'s names (``nuc`` dropped)."""
        P, Gamma = self.densities(Z)[4]
        _L, terms = self.skeleton(P, Gamma).hessian()
        terms.pop("nuc")
        return sum(terms.values()), terms

    # -- the independent formulas the identities hold these to -------------

    def z_gradient_219(self, Z: np.ndarray) -> np.ndarray:
        """``Z`` against #219's ``dg_all/dR`` (``casscf_nuclear_hessian.
        mixed_derivatives``): the same quantity from ``gen_g_hop``'s gradient
        formula at the perturbed Hamiltonian."""
        return cnh.mixed_derivatives(self.sk, self.th.var, self.th.ci, self.nelecas) @ Z

    def mixed_by_series(self, u: np.ndarray, Z: np.ndarray, k: int) -> float:
        """``Z . d^2 g / dtheta dR_k [u]`` the other way round: #216's
        energy series (:class:`casscf_third_derivative.Third`) with the
        reference integrals replaced by #215's perturbed Hamiltonian
        ``h^(k)``, ``(pq|rs)^(k)`` over the full range -- its ``a s``
        coefficient. One coordinate; for the hand cases."""
        allmo = slice(None)
        th = ctd.Third.__new__(ctd.Third)
        th.__dict__.update(self.th.__dict__)
        th.h_mo = self.sk.perturbed_h1(k)
        th.eri_mo = self.sk.perturbed_eri(k, (allmo,) * 4)
        th.e_nuc = 0.0
        return float(th.series(u, np.zeros_like(u), Z).get(5, 0.0))


# --------------------------------------------------------------------------
# The oracle: Z . g along the frozen path
# --------------------------------------------------------------------------


@dataclass
class Path:
    """What :func:`zg` needs to rebuild ``g_all`` at any ``(theta, R)``."""

    geom: list
    basis: str
    C0: np.ndarray
    S0_half: np.ndarray
    ci: np.ndarray
    ncas: int
    nelecas: tuple
    var: np.ndarray

    @classmethod
    def of(cls, con: Constraint) -> Path:
        return cls(geom=con.geom, basis=con.basis, C0=con.th.C0, S0_half=con.sk.S0_half,
                   ci=con.th.ci, ncas=con.ncas, nelecas=con.nelecas, var=con.th.var)


def g_all_at(path: Path, theta: np.ndarray, dvec: np.ndarray) -> np.ndarray:
    """``gen_g_hop``'s ``g_all`` at ``C_f(R0 + dvec) exp(K(x))`` and
    ``(c0 + y) / |c0 + y|``, on the molecule at ``R0 + dvec``."""
    geom = ngrad.displace_vec(path.geom, dvec)
    mol = ch.mol_at(geom, path.basis)
    nmo, nvar = path.C0.shape[1], len(path.var)
    C = ngr.frozen_coeff(path.C0, path.S0_half, mol.intor("int1e_ovlp"))
    C = C @ expm(cnh.unpack_kappa(path.var, theta[:nvar], nmo))
    c = path.ci.ravel() + theta[nvar:]
    mc = cnh.casscf_object(mol, path.ncas, path.nelecas)
    return cnh.g_hop(mc, C, (c / np.linalg.norm(c)).reshape(path.ci.shape))[0]


_PATH: Path | None = None


def _init_worker(path: Path) -> None:
    global _PATH
    _PATH = path


def _g_task(task: tuple) -> tuple:
    key, theta, dvec = task
    assert _PATH is not None
    return key, g_all_at(_PATH, theta, dvec)


def g_all_points(path: Path, points: dict, workers: int) -> dict:
    """``g_all`` at every ``key -> (theta, dvec)`` of ``points``."""
    tasks = [(k, th, d) for k, (th, d) in points.items()]
    if workers <= 1:
        _init_worker(path)
        return dict(_g_task(t) for t in tasks)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(path,)) as pool:
        return dict(pool.map(_g_task, tasks, chunksize=4))


def _column_task(task: tuple) -> tuple:
    key, ref, S0_half, dvec = task
    geom = ngrad.displace_vec(ref.geom, dvec)
    nuc = np.asarray(rhf_grad.grad_nuc(ch.mol_at(geom, ref.basis))).ravel()
    return key, ch.path_gradient(ref, geom, S0_half) - nuc


def z_gradients(con: Constraint, Z: np.ndarray, points: dict, workers: int) -> dict:
    """The analytic ``Z . dg/dR`` at every ``key -> dvec`` of ``points``, on
    the frozen path (re-anchored at ``C_f(R)``, the same densities):
    ``casscf_hessian.path_gradient`` at the Z-weighted densities, less the
    nuclear term."""
    P, Gamma = con.densities(Z)[4]
    ref = ch.Reference(geom=con.geom, basis=con.basis, C0=con.th.C0, ncore=0, ncas=con.nmo,
                       D=P, G=Gamma)
    tasks = [(k, ref, con.sk.S0_half, d) for k, d in points.items()]
    if workers <= 1:
        return dict(_column_task(t) for t in tasks)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_column_task, tasks))


# --------------------------------------------------------------------------
# Directions, the checks
# --------------------------------------------------------------------------


def kept_directions(geom, mol_sym) -> dict:
    """The coordinate directions the checks run along, by name: each
    Cartesian ``e<k>`` that keeps ``mol_sym``'s point group
    (``casscf_nuclear_hessian.keeps_point_group``; every one on a C1
    molecule), and on a diatomic the stretch."""
    natm = len(geom)
    cands = {f"e{k}": ch.unit(natm, k) for k in range(3 * natm)}
    if natm == 2:
        xyz = mol_sym.atom_coords()
        a = (xyz[1] - xyz[0]) / np.linalg.norm(xyz[1] - xyz[0])
        cands["stretch"] = np.array([-a, a]) / np.sqrt(2.0)
    return {name: d for name, d in cands.items() if cnh.keeps_point_group(mol_sym, d)}


def theta_directions(con: Constraint) -> dict:
    """The seeded unit ``theta`` directions, by name (``kind/seed``)."""
    out = {}
    for i, kind in enumerate(THETA_KINDS):
        for seed in THETA_SEEDS:
            rng = np.random.default_rng(2320 + 100 * i + seed)
            out[f"{kind}/{seed}"] = ctd.direction(con.th, kind, rng)
    return out


def identities(con: Constraint, Z: np.ndarray, us: dict, M: np.ndarray, zgrad: np.ndarray,
               mixed: dict, e_state: float | None) -> dict:
    """Every exact identity, as a maximum."""
    th = con.th
    out: dict = {}
    dens = con.densities(Z)
    if e_state is not None:
        out["E rebuilt"] = abs(con.contract(*dens[0]) + th.e_nuc - e_state)
    out["Z.g_all rebuilt"] = abs(con.contract(*dens[4]) - float(Z @ th.g_hop()[0]))
    worst = 0.0
    for u in us.values():
        s5 = con.contract(*con.densities(Z, u)[5])
        worst = max(worst, abs(s5 - float(th.series(u, np.zeros_like(u), Z).get(5, 0.0))))
    out["Z.J[u] rebuilt"] = worst
    out["Z.dg/dR vs #219"] = float(np.abs(zgrad - con.z_gradient_219(Z)).max())
    out["d2g/dR2 symmetry"] = float(np.abs(M - M.T).max())
    natm = con.natm
    out["d2g/dR2 translation"] = float(np.abs(M.reshape(natm, 3, natm, 3).sum(axis=2)).max())
    out["dg/dR translation"] = float(np.abs(zgrad.reshape(natm, 3).sum(axis=0)).max())
    out["d2g/dthdR translation"] = max(float(np.abs(w.reshape(natm, 3).sum(axis=0)).max())
                                       for w in mixed.values())
    return out


def run_ladders(con: Constraint, Z: np.ndarray, us: dict, dirs: dict, M: np.ndarray,
                mixed: dict, workers: int, columns: bool = True, log=print) -> list:
    """Every ladder: the mixed term along each ``(u, d)`` pair, ``d^T M d``
    along each direction, ``d^T M d'`` along each pair of directions, and
    (``columns``) ``M d`` against the analytic ``Z . dg/dR`` differenced."""
    steps = ngr.ladder_steps(LADDER)
    nt = con.nvar + con.th.ndet
    zero_t = np.zeros(nt)
    zero_r = np.zeros((con.natm, 3))
    names = list(dirs)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    points: dict = {("0",): (zero_t, zero_r)}
    for un, u in us.items():
        for dn, d in dirs.items():
            for h in steps:
                for sa in (1, -1):
                    for sb in (1, -1):
                        points[("mix", un, dn, sa, sb, h)] = (sa * h * u, sb * h * d)
    for dn, d in dirs.items():
        for h in steps:
            for s in (1, -1):
                points[("diag", dn, s, h)] = (zero_t, s * h * d)
    for a, b in pairs:
        for h in steps:
            for sa in (1, -1):
                for sb in (1, -1):
                    points[("pair", a, b, sa, sb, h)] = (zero_t, h * (sa * dirs[a] + sb * dirs[b]))
    t0 = time.perf_counter()
    g = g_all_points(Path.of(con), points, workers)
    zg = {k: float(Z @ v) for k, v in g.items()}
    log(f"  {len(points)} g_all evaluations along the frozen path in "
        f"{time.perf_counter() - t0:.1f} s ({workers} worker(s))")
    checks = []
    for un in us:
        for dn, d in dirs.items():
            fd = [(zg[("mix", un, dn, 1, 1, h)] - zg[("mix", un, dn, -1, 1, h)]
                   - zg[("mix", un, dn, 1, -1, h)] + zg[("mix", un, dn, -1, -1, h)])
                  / (4.0 * h * h) for h in steps]
            checks.append(ch.judge(f"d2g/dth dR [{un}, {dn}]", fd, steps,
                                   float(d.ravel() @ mixed[un])))
    z0 = zg[("0",)]
    for dn, d in dirs.items():
        fd = [(zg[("diag", dn, 1, h)] - 2.0 * z0 + zg[("diag", dn, -1, h)]) / (h * h)
              for h in steps]
        dv = d.ravel()
        checks.append(ch.judge(f"d2g/dR2 [{dn}, {dn}]", fd, steps, float(dv @ M @ dv)))
    for a, b in pairs:
        fd = [(zg[("pair", a, b, 1, 1, h)] - zg[("pair", a, b, 1, -1, h)]
               - zg[("pair", a, b, -1, 1, h)] + zg[("pair", a, b, -1, -1, h)]) / (4.0 * h * h)
              for h in steps]
        checks.append(ch.judge(f"d2g/dR2 [{a}, {b}]", fd, steps,
                               float(dirs[a].ravel() @ M @ dirs[b].ravel())))
    if not columns:
        return checks
    csteps = ngr.ladder_steps(COLUMN_LADDER)
    cpoints = {(dn, s, h): s * h * d for dn, d in dirs.items() for h in csteps
               for s in (1, -1)}
    t0 = time.perf_counter()
    cg = z_gradients(con, Z, cpoints, workers)
    log(f"  {len(cpoints)} analytic Z . dg/dR on the path in "
        f"{time.perf_counter() - t0:.1f} s ({workers} worker(s))")
    for dn, d in dirs.items():
        fd = np.array([(cg[(dn, 1, h)] - cg[(dn, -1, h)]) / (2.0 * h) for h in csteps])
        checks.append(ch.judge_column(f"d2g/dR2 [:, {dn}] (Z.dg/dR column)", fd, csteps,
                                      M @ d.ravel()))
    return checks


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=CASES, required=True)
    ap.add_argument("--workers", type=int, default=1,
                    help="processes for the multipliers' kappa sweep and the ladders")
    ap.add_argument("--seeded-multipliers", action="store_true",
                    help="a seeded unit (Z, z) instead of the SC-NEVPT2 gradient's")
    ap.add_argument("--no-columns", action="store_true",
                    help="skip the Z . dg/dR column ladders")
    ap.add_argument("--show-ladders", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    print(f"[{args.case}] the CASSCF constraint's R-mixed terms, Z . d2g/dth dR and "
          "Z . d2g/dR2")
    mo_from = ngr.REPO / "golden" / f"{args.case}_casscf.mogold"
    if args.seeded_multipliers:
        st = ngr.build_state(args.case, dict(ngr.CASES[args.case]), reference="casscf",
                             mo_from=mo_from)
        Z, info = None, None
    else:
        st, Z, info = ctd.multipliers(args.case, args.workers)
    con = Constraint(st.geom, st.basis, st.mo_coeff, st.ci, st.ncore, st.nact, st.nelecas)
    if Z is None:
        Z = ctd.seeded_multipliers(con.th)
        print(f"  seeded unit (Z, z), seed {MULTIPLIER_SEED}")
    elif not np.array_equal(info["var"], con.th.var):
        raise SystemExit("the multipliers' orbital variables are not gen_g_hop's here")
    print(f"  n_mo = {con.nmo} = {con.ncore} core + {con.ncas} active + "
          f"{con.nmo - con.nocc} virtual; {con.nvar} orbital variables, {con.th.ndet} "
          f"determinants; {con.ncoord} coordinates; |g_all| at the state = "
          f"{np.linalg.norm(con.th.g_hop()[0]):.1e}")
    t1 = time.perf_counter()
    zgrad = con.z_gradient(Z)
    M, terms = con.second(Z)
    t_second = time.perf_counter() - t1
    us = theta_directions(con)
    t1 = time.perf_counter()
    mixed = {name: con.mixed(u, Z) for name, u in us.items()}
    t_mixed = (time.perf_counter() - t1) / len(us)
    print(f"  Z . dg/dR and Z . d2g/dR2 in {t_second:.1f} s; Z . d2g/dth dR [u] "
          f"{t_mixed:.1f} s per direction (all {con.ncoord} coordinates)")
    print("    max |Z . d2g/dR2| = " + f"{np.abs(M).max():.4e}; by term: "
          + ", ".join(f"{k} {np.abs(v).max():.2e}" for k, v in terms.items()))
    ids = identities(con, Z, us, M, zgrad, mixed, float(st.mc.e_tot))
    ok = True
    for name, v in ids.items():
        good = v <= IDENTITY_TOL
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name:<24} {v:.2e}  (allowed {IDENTITY_TOL:g})")
    dirs = kept_directions(st.geom, st.pt.mol)
    print(f"  directions ({len(dirs)}): {', '.join(dirs)}; theta directions: "
          f"{', '.join(us)}")
    print(f"  ladders: Z . g_all four-point / second differences h = 2^-{LADDER[0]} .. "
          f"2^-{LADDER[1]}, columns 2^-{COLUMN_LADDER[0]} .. 2^-{COLUMN_LADDER[1]}; gate "
          f"{LADDER_TOL:g}")
    checks = run_ladders(con, Z, us, dirs, M, mixed, args.workers,
                         columns=not args.no_columns)
    worst: dict = {}
    count: dict = {}
    nzero = 0
    for c in checks:
        kind = ("mixed" if c.name.startswith("d2g/dth") else
                "column" if "column" in c.name else "second")
        count[kind] = count.get(kind, 0) + 1
        if c.numbers.get("zero"):
            nzero += 1
        else:
            worst[kind] = max(worst.get(kind, 0.0), c.numbers["err"])
        ok &= c.ok
        if args.show_ladders or not c.ok:
            print(f"  {'ok  ' if c.ok else 'FAIL'} {c.name}: {c.summary}")
            print("\n".join(c.table))
    print(f"  {sum(c.ok for c in checks)}/{len(checks)} ladders pass ({nzero} zero by "
          f"symmetry); worst |Richardson - analytic| over the nonzero ones:")
    for kind in ("mixed", "second", "column"):
        if kind in count:
            print(f"    {kind:<7} ({count[kind]:4d}) {worst.get(kind, 0.0):.2e}")
    mags = [abs(float(d.ravel() @ w)) for w in mixed.values() for d in dirs.values()]
    print(f"  |d . Z.d2g/dth dR [u]| over the checked pairs: {min(mags):.1e} .. "
          f"{max(mags):.1e}")
    print(f"{'PASS' if ok else 'FAIL'} ({time.perf_counter() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
