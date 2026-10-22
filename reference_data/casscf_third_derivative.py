#!/usr/bin/env python3
r"""Level 2's parameter-parameter term on the CASSCF branch, offline:
``Z . d^2 g / dtheta dtheta``, the gradient's multipliers contracted with
the second derivative of the CASSCF stationarity constraint -- a **third**
derivative of the CASSCF energy (docs/hessian-theory.md, section 7).

THE OBJECT
----------

On the CASSCF branch the constraint is ``pyscf.mcscf.newton_casscf.gen_g_hop``'s
``g_all = [g_orb ; g_ci]`` (``nevpt2_grad.py``'s (S1) and (C1)), evaluated at
the parameters ``theta = (x, y)`` exactly as ``nevpt2_grad.casscf_step``
evaluates it::

    C(theta) = C0 exp(K(x)),   K_ij = x_(i,j), K_ji = -x_(i,j)   (gen_g_hop's packed variables)
    c(theta) = (c0 + y) / |c0 + y|

and ``g_all`` there is the energy's slope in the **local** step variables at
that point: ``Z . g_all(theta) = d/ds E(C(theta) exp(s K(Z_orb)), (c(theta) +
s z) / |c(theta) + s z|)`` at ``s = 0`` (``g_orb = 2 (F - F^T)``,
``g_ci = 2 (H - E) c``, the multipliers ``(Z, z)`` in that normalisation).
So the level-2 term is the action

::

    T(u, v) = d^2/da db [ Z . g_all(a u + b v) ]                    at 0
            = d^3/da db ds  E( C0 exp(K(a u + b v)) exp(s K(Z)),
                               norm(norm(c0 + a u_c + b v_c) + s z) )

over full-range directions ``u``, ``v`` in ``gen_g_hop``'s variables
(orbital, CI or both: the (kappa,kappa), (kappa,c) and (c,c) blocks at once).
It is symmetric in ``(u, v)`` by construction; it is **not** the symmetric
third derivative ``d^3 E[u, v, Z]`` of ``E`` in one global chart, because
``Z`` steps from the displaced point (the local frame ``g_all`` is written
in), and that difference is first order in the connection
(:func:`Third.global_third` keeps the global one, as the wrong operator the
hand cases show the ladder catching). The multipliers are used
undifferentiated -- the 2n+2 rule (section 7).

HOW IT IS COMPUTED: EXACT TRUNCATED TAYLOR SERIES
-------------------------------------------------

Every quantity is carried as a polynomial in three nilpotent units ``a``,
``b``, ``s`` (``a^2 = b^2 = s^2 = 0``; :func:`hd_mul` keeps the multilinear
coefficients, indexed by a bitmask), and ``T`` is the ``a b s`` coefficient
of the energy. Nothing is differenced, so ``T`` has no step error: ``exp``
is its series to third order (exact, ``K^4 = 0`` in this algebra), the
normalisation is ``x^{-1/2}``'s Taylor series to third order (exact), the
MO integrals over the occupied range are the reference's full ``(pq|rs)``
transformed by ``U = exp(K(a u + b v)) exp(s K(Z))``'s occupied columns, and
the energy is ``E_core(U) + <c|H_act(U)|c>`` with ``H_act`` through
``direct_spin1``. The lower coefficients are the familiar objects --
``E``, ``g_all . u``, ``u . Hess . v``, ``Z . Hess . u`` -- and the hand
cases hold them to ``gen_g_hop``'s.

THE ORACLE, AND WHY IT IS NOT ``h_op`` DIFFERENCED
--------------------------------------------------

The natural ladder is ``Z . h_op(theta + h v)[u]``, ``gen_g_hop``'s
Hessian-vector product at the displaced point, differenced along ``v``. It
does **not** converge to ``T``, for two reasons, both measured
(docs/reference-data.md, "Level 2's CASSCF third-derivative term"):

* ``h_op`` is the **Hessian** of the energy in the local chart, at any
  point (the series' ``(a, b)`` coefficient is ``u . h_op(v)`` to
  round-off far from stationarity; a hand case pins it), and that is the
  Jacobian of ``g_all`` only **at a stationary point**: off it, the
  orbital block of the Jacobian carries terms proportional to the orbital
  gradient that ``h_op`` does not, so ``h_op(theta + h v)`` misses the
  Jacobian at first order in ``h`` and so does its derivative (on CI-only
  pairs the two agree, and so does the ladder);
* ``h_op(theta + h v)[u]`` steps along ``u`` from the displaced point's
  frame, so its ``v``-derivative carries the connection term
  ``-1/2 Z . Hess [K(v), K(u)]`` -- antisymmetric in ``(u, v)``, including
  core-core, active-active and virtual-virtual blocks ``h_op`` cannot take.

So the oracle (:func:`ladder`) differences ``gen_g_hop``'s ``g_all`` itself
-- the constraint, which is exact everywhere -- twice: the Jacobian along
``u`` at ``theta + h v`` as a central difference, differenced along ``v``,
i.e. the four-point quotient ``Z . [g(h(u + v)) - g(h(u - v)) - g(h(-u + v))
+ g(-h(u + v))] / 4 h^2`` over a step ladder, gated by
``nevpt2_grad.ladder_check``'s run rule and Richardson extrapolant at
``nevpt2_grad.LADDER_TOL``. The ``h_op`` ladder is printed beside it,
informational.

THE CHECKS (``main``)
---------------------

On the case's committed CASSCF state (``nevpt2_grad.build_state(...,
reference="casscf")``, its frame and every gate), with **the gradient's own
multipliers** -- the SC-NEVPT2 ``(Z, z)`` of ``nevpt2_grad.py --reference
casscf``, solved here by its own classes (:func:`multipliers`; minutes of
CPU: the source terms and one kappa sweep) -- or, with ``--seeded-multipliers``,
a seeded unit ``(Z, z)`` (``T`` is linear in it):

* for :data:`DIRECTION_SEEDS` seeded unit pairs of each kind -- orbital
  ``(kappa, kappa)``, CI ``(c, c)`` with ``u_c, v_c`` orthogonal to ``c``,
  mixed ``(kappa, c)``, and full ``(theta, theta)`` -- ``T(u, v)`` against
  the ladder;
* ``|T(u, v) - T(v, u)|`` under :data:`SYMMETRY_TOL` (bookkeeping: the
  series is symmetric by construction);
* the series' first-order coefficient against ``g_all . u`` and its
  ``(a, b)`` coefficient against ``u . h_op(v)``, under
  :data:`IDENTITY_TOL`: identities at any point, so the parametrisation is
  ``gen_g_hop``'s own.

The measured numbers are in docs/reference-data.md, "Level 2's CASSCF
third-derivative term".

Usage::

    uv run python reference_data/casscf_third_derivative.py --case n2_ccpvdz_cas44 --workers 6
    uv run python reference_data/casscf_third_derivative.py --case lif_def2svp_cas46 --workers 6
    uv run python reference_data/casscf_third_derivative.py --case n2_ccpvdz_cas44 \
        --seeded-multipliers          # no NEVPT2: a seeded (Z, z), seconds

Nothing is written; no committed file is read except through
``nevpt2_grad.build_state``.
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads
# (docs/reference-data.md, "Reproducibility").
import generate_golden as gg  # noqa: F401

# isort: split
import casscf_hessian as ch
import casscf_nuclear_hessian as cnh
import nevpt2_grad as ngr

# isort: split
import argparse
import sys
import time
from collections.abc import Callable

import numpy as np
from pyscf import ao2mo, scf
from pyscf.fci import direct_spin1
from scipy.linalg import expm

#: The ladder, ``h = 2^-k`` along unit directions: the four-point quotient's
#: round-off grows as ``eps / h^2``, so the fine end stops where it is still
#: two decades under the gate.
LADDER = (3, 11)

#: The gate on ``|Richardson - T|``: ``nevpt2_grad.LADDER_TOL``, the
#: precision every ladder of the gradient and Hessian path is held to.
LADDER_TOL = ngr.LADDER_TOL

#: ``|T(u, v) - T(v, u)|``: the series is symmetric by construction, so
#: this is round-off.
SYMMETRY_TOL = 1e-10

#: The series' ``a`` and ``a b`` coefficients against ``gen_g_hop``'s
#: ``g_all . u`` and ``u . h_op(v)``: identities at any point, read to
#: round-off.
IDENTITY_TOL = 1e-10

#: Seeds of the direction pairs, per kind.
DIRECTION_SEEDS = (0, 1, 2)

#: The pair kinds: which blocks of ``(u, v)`` are filled.
KINDS = ("orbital", "ci", "mixed", "full")

#: The cases the acceptance names (``_casscf`` goldens with a ``.mogold``).
CASES = ("n2_ccpvdz_cas44", "lif_def2svp_cas46")

#: The seed of ``--seeded-multipliers``' ``(Z, z)``.
MULTIPLIER_SEED = 216

Series = dict  # bitmask -> coefficient (a: 1, b: 2, s: 4)


# --------------------------------------------------------------------------
# Truncated multilinear Taylor series in three nilpotent units
# --------------------------------------------------------------------------


def hd_mul(f: Callable, A: Series, B: Series) -> Series:
    """The product ``f`` (any bilinear map) of two series: every pair of
    coefficients whose units are disjoint, ``a^2 = b^2 = s^2 = 0``."""
    out: Series = {}
    for m1, x in A.items():
        for m2, y in B.items():
            if m1 & m2:
                continue
            r = f(x, y)
            m = m1 | m2
            out[m] = out[m] + r if m in out else r
    return out


def hd_add(A: Series, B: Series) -> Series:
    out = dict(A)
    for m, y in B.items():
        out[m] = out[m] + y if m in out else y
    return out


def hd_scale(A: Series, s: float) -> Series:
    return {m: s * x for m, x in A.items()}


def hd_expm(X: Series, n: int) -> Series:
    """``exp(X)`` for a series with no constant term: ``sum_k X^k / k!`` to
    ``k = 3``, exact (``X^4 = 0`` with three units)."""
    eye = np.eye(n)
    out: Series = {0: eye}
    term: Series = {0: eye}
    for k in range(1, 4):
        term = hd_scale(hd_mul(lambda p, q: p @ q, term, X), 1.0 / k)
        out = hd_add(out, term)
    return out


def hd_rsqrt(A: Series) -> Series:
    """``A^{-1/2}`` for a scalar series with a positive constant term: its
    Taylor series in the nilpotent part, to third order (exact)."""
    x0 = float(A[0])
    nil = {m: x for m, x in A.items() if m}
    coef = (-0.5 * x0 ** -1.5, 0.375 * x0 ** -2.5, -0.3125 * x0 ** -3.5)
    out: Series = {0: x0 ** -0.5}
    power: Series = {0: 1.0}
    for c in coef:
        power = hd_mul(lambda p, q: p * q, power, nil)
        out = hd_add(out, hd_scale(power, c))
    return out


def hd_normalise(c: Series) -> Series:
    """``c / |c|`` for a vector series."""
    return hd_mul(lambda p, q: p * q, hd_rsqrt(hd_mul(lambda p, q: p @ q, c, c)), c)


# --------------------------------------------------------------------------
# The CASSCF energy as a series, and T
# --------------------------------------------------------------------------


class Third:
    """The level-2 term at one CASSCF state: orbitals ``C0`` (on a
    symmetry-off ``mol`` in the state's AO frame), CI vector ``ci``, and
    ``gen_g_hop``'s variables."""

    def __init__(self, mol, C0: np.ndarray, ci: np.ndarray, ncore: int, ncas: int,
                 nelecas):
        self.mol = mol
        self.C0 = np.asarray(C0, dtype=float)
        self.ci = np.asarray(ci, dtype=float) / np.linalg.norm(ci)
        self.ncore, self.ncas, self.nelecas = ncore, ncas, tuple(nelecas)
        self.nocc = ncore + ncas
        self.nmo = self.C0.shape[1]
        self.mc = cnh.casscf_object(mol, ncas, self.nelecas)
        self.var = cnh.variables(self.mc, self.nmo)
        self.nvar = len(self.var)
        self.ndet = self.ci.size
        self.h_mo = self.C0.T @ scf.hf.get_hcore(mol) @ self.C0
        self.eri_mo = ao2mo.restore(1, ao2mo.full(mol, self.C0), self.nmo)
        self.e_nuc = float(mol.energy_nuc())

    # -- the parametrisation ---------------------------------------------

    def kappa(self, x: np.ndarray) -> np.ndarray:
        return cnh.unpack_kappa(self.var, x[:self.nvar], self.nmo)

    def at(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(C(theta), c(theta))``: ``nevpt2_grad.casscf_step``'s point."""
        C = self.C0 @ expm(self.kappa(theta))
        c = self.ci.ravel() + theta[self.nvar:]
        return C, (c / np.linalg.norm(c)).reshape(self.ci.shape)

    def g_all(self, theta: np.ndarray) -> np.ndarray:
        """``gen_g_hop``'s ``g_all`` at :meth:`at`'s point."""
        return cnh.g_hop(self.mc, *self.at(theta))[0]

    def g_hop(self, theta: np.ndarray | None = None):
        """``gen_g_hop``'s ``(g_all, h_op)`` at :meth:`at`'s point."""
        theta = np.zeros(self.nvar + self.ndet) if theta is None else theta
        return cnh.g_hop(self.mc, *self.at(theta))

    # -- the energy as a series ------------------------------------------

    def energy(self, U: Series, c: Series) -> Series:
        """``E`` at orbitals ``C0 U`` and the (normalised) CI series ``c``:
        ``E_core(U) + <c|H_act(U)|c>``, every MO integral over the occupied
        range transformed from the reference's."""
        nc, a = self.ncore, slice(self.ncore, self.nocc)
        Uo = {m: u[:, :self.nocc] for m, u in U.items()}
        h = hd_mul(lambda p, q: q.T @ p, hd_mul(lambda p, q: p @ q, {0: self.h_mo}, Uo), Uo)
        g: Series = {0: self.eri_mo}
        for ax in range(4):
            g = hd_mul(lambda p, q, ax=ax: np.moveaxis(np.tensordot(p, q, axes=([ax], [0])),
                                                       -1, ax), g, Uo)
        Hc: Series = {}
        for m1, hh in h.items():
            gg_ = g[m1]
            core = gg_[:nc, :nc, :nc, :nc]
            e0 = (2.0 * np.trace(hh[:nc, :nc]) + 2.0 * np.einsum("iijj->", core)
                  - np.einsum("ijji->", core) + (self.e_nuc if m1 == 0 else 0.0))
            heff = (hh[a, a] + 2.0 * np.einsum("tuii->tu", gg_[a, a, :nc, :nc])
                    - np.einsum("tiiu->tu", gg_[a, :nc, :nc, a]))
            h2 = direct_spin1.absorb_h1e(heff, gg_[a, a, a, a], self.ncas, self.nelecas, 0.5)
            for m2, cc in c.items():
                if m1 & m2:
                    continue
                r = (direct_spin1.contract_2e(h2, cc.reshape(self.ci.shape), self.ncas,
                                              self.nelecas).ravel() + e0 * cc)
                m = m1 | m2
                Hc[m] = Hc[m] + r if m in Hc else r
        return hd_mul(lambda p, q: float(p @ q), c, Hc)

    def series(self, u: np.ndarray, v: np.ndarray, Z: np.ndarray) -> Series:
        """``E(C0 exp(K(a u + b v)) exp(s K(Z)), norm(norm(c0 + a u_c + b
        v_c) + s z))`` as a series: the ``a b s`` coefficient (mask 7) is
        ``T(u, v)``."""
        n = self.nvar
        U = hd_mul(lambda p, q: p @ q,
                   hd_expm({1: self.kappa(u), 2: self.kappa(v)}, self.nmo),
                   {0: np.eye(self.nmo), 4: self.kappa(Z)})
        c = hd_normalise({0: self.ci.ravel(), 1: u[n:], 2: v[n:]})
        c = hd_normalise(hd_add(c, {4: np.asarray(Z[n:], dtype=float)}))
        return self.energy(U, c)

    def contract(self, u: np.ndarray, v: np.ndarray, Z: np.ndarray) -> float:
        """``T(u, v) = Z . d^2 g_all / dtheta dtheta [u, v]``."""
        return float(self.series(u, v, Z).get(7, 0.0))

    def action(self, u: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """``T(u, .)`` over ``gen_g_hop``'s full variable list (orbital
        variables, then every determinant): one series per variable."""
        n = self.nvar + self.ndet
        out = np.empty(n)
        for k in range(n):
            e = np.zeros(n)
            e[k] = 1.0
            out[k] = self.contract(u, e, Z)
        return out

    def global_third(self, u: np.ndarray, v: np.ndarray, Z: np.ndarray) -> float:
        """The symmetric ``d^3 E[u, v, Z]`` in the one global chart (``Z``
        added to ``a u + b v`` before the exponential and the
        normalisation) -- NOT the level-2 term, which steps ``Z`` from the
        displaced point. Kept as the wrong operator the hand cases show the
        ladder rejecting."""
        n = self.nvar
        U = hd_expm({1: self.kappa(u), 2: self.kappa(v), 4: self.kappa(Z)}, self.nmo)
        c = hd_normalise({0: self.ci.ravel(), 1: u[n:], 2: v[n:],
                          4: np.asarray(Z[n:], dtype=float)})
        return float(self.energy(U, c).get(7, 0.0))


# --------------------------------------------------------------------------
# Directions, the ladder
# --------------------------------------------------------------------------


def direction(th: Third, block: str, rng: np.random.Generator) -> np.ndarray:
    """A seeded unit direction in ``gen_g_hop``'s variables: ``orbital``
    (CI part zero), ``ci`` (orbital part zero; orthogonal to ``c``, every
    determinant, in or out of the state's sector) or ``full``."""
    x = rng.standard_normal(th.nvar + th.ndet)
    c = th.ci.ravel()
    x[th.nvar:] -= float(c @ x[th.nvar:]) * c
    if block == "orbital":
        x[th.nvar:] = 0.0
    elif block == "ci":
        x[:th.nvar] = 0.0
    return x / np.linalg.norm(x)


def pair(th: Third, kind: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``(u, v)`` pair of :data:`KINDS` ``kind`` for ``seed``."""
    rng = np.random.default_rng(1000 * KINDS.index(kind) + seed)
    blocks = {"orbital": ("orbital", "orbital"), "ci": ("ci", "ci"),
              "mixed": ("orbital", "ci"), "full": ("full", "full")}[kind]
    return direction(th, blocks[0], rng), direction(th, blocks[1], rng)


def ladder_quotients(th: Third, u: np.ndarray, v: np.ndarray, Z: np.ndarray,
                     steps: list[float]) -> list[float]:
    """The four-point quotient of ``Z . g_all`` at every step: ``gen_g_hop``'s
    Jacobian along ``u`` at ``theta + h v`` (a central difference of its
    ``g_all``), differenced along ``v``."""
    return [float(Z @ (th.g_all(h * (u + v)) - th.g_all(h * (u - v))
                       - th.g_all(h * (v - u)) + th.g_all(-h * (u + v)))) / (4.0 * h * h)
            for h in steps]


def ladder(th: Third, u: np.ndarray, v: np.ndarray, Z: np.ndarray, t: float,
           name: str) -> ngr.Check:
    """``T(u, v)`` against :func:`ladder_quotients` over :data:`LADDER`:
    ``nevpt2_grad.ladder_check``'s run rule and Richardson extrapolant,
    gated at :data:`LADDER_TOL`."""
    steps = ngr.ladder_steps(LADDER)
    return ngr.ladder_check(name, ladder_quotients(th, u, v, Z, steps), steps, t, LADDER_TOL)


def hop_ladder(th: Third, u: np.ndarray, v: np.ndarray, Z: np.ndarray,
               steps: list[float]) -> list[float]:
    """The literal ``Z . h_op(theta + h v)[u]`` differenced along ``v``:
    informational (the module docstring says why it is not the oracle)."""
    out = []
    for h in steps:
        _gp, hp = th.g_hop(h * v)
        _gm, hm = th.g_hop(-h * v)
        out.append(float(Z @ (hp(u) - hm(u))) / (2.0 * h))
    return out


# --------------------------------------------------------------------------
# The multipliers
# --------------------------------------------------------------------------


def multipliers(case: str, workers: int, log=print) -> tuple[ngr.State, np.ndarray, dict]:
    """The SC-NEVPT2 gradient's ``(Z, z)`` on the committed CASSCF state, as
    ``nevpt2_grad.py --reference casscf`` solves them (PySCF's CI source term,
    one kappa sweep, ``z^c`` then the dense coupled system), laid out over
    ``gen_g_hop``'s full variable list: ``[Z over its packed orbital
    variables ; z over every determinant]``."""
    st = ngr.build_state(case, dict(ngr.CASES[case]), reference="casscf",
                         mo_from=ngr.REPO / "golden" / f"{case}_casscf.mogold")
    fn = ngr.Functional(st)
    t0 = time.perf_counter()
    y = ngr.pyscf_ci_source(fn, st)
    src = ngr.SourceTerms(kappa=np.zeros((st.nmo, st.nmo)), ci=y, origin="pyscf")
    g = ngr.CasscfGradients(st, src, workers=workers)
    g.prepare(log=lambda *_a: None)
    g.solve_multipliers(log=lambda *_a: None)
    Zorb = np.zeros(g.nvar)
    Zorb[g.s_var] = g.lam[g.s_rows]
    info = dict(seconds=time.perf_counter() - t0, var=g.var,
                cp_operator=float(g.residuals["cp_operator"]),
                cp_rhs_norm=float(g.residuals["cp_rhs_norm"]))
    log(f"  the multipliers (nevpt2_grad.CasscfGradients, SC): {info['seconds']:.0f} s, "
        f"max |Z| = {np.abs(Zorb).max():.3e}, |z| = {np.linalg.norm(g.z):.3e}, "
        f"|J L + b| = {info['cp_operator']:.1e} against |b| = {info['cp_rhs_norm']:.1e}")
    return st, np.concatenate([Zorb, g.z.ravel()]), info


def seeded_multipliers(th: Third) -> np.ndarray:
    """A seeded unit ``(Z, z)``, ``z`` orthogonal to ``c``."""
    return direction(th, "full", np.random.default_rng(MULTIPLIER_SEED))


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def run_checks(th: Third, Z: np.ndarray, show_ladders: bool = False, log=print) -> list:
    """Every pair of every kind: ``T`` against the ladder, ``T``'s symmetry,
    and the series' lower coefficients against ``gen_g_hop``."""
    g_all, h_op = th.g_hop()
    steps = ngr.ladder_steps(LADDER)
    checks = []
    for kind in KINDS:
        for seed in DIRECTION_SEEDS:
            u, v = pair(th, kind, seed)
            t0 = time.perf_counter()
            s_uv = th.series(u, v, Z)
            t = float(s_uv[7])
            t_vu = th.contract(v, u, Z)
            dt = time.perf_counter() - t0
            name = f"T[{kind}, seed {seed}]"
            chk = ladder(th, u, v, Z, t, name)
            sym = abs(t - t_vu)
            hop = hop_ladder(th, u, v, Z, steps[-2:])
            hop_rich = (4.0 * hop[-1] - hop[-2]) / 3.0
            chk.numbers.update(t=t, symmetry=sym, seconds=dt,
                               grad_u=abs(float(s_uv[1]) - float(g_all @ u)),
                               hess_uv=abs(float(s_uv[3]) - float(u @ h_op(v))),
                               hop_miss=abs(hop_rich - t))
            chk.ok = (chk.ok and sym <= SYMMETRY_TOL
                      and max(chk.numbers["grad_u"], chk.numbers["hess_uv"]) <= IDENTITY_TOL)
            log(f"  {'ok  ' if chk.ok else 'FAIL'} {name}: T = {t:+.10e} ({dt:.2f} s for "
                f"two series); {chk.summary}; |T(u,v) - T(v,u)| = {sym:.1e} (allowed "
                f"{SYMMETRY_TOL:g})")
            log(f"       series vs gen_g_hop: |E_a - g_all.u| = {chk.numbers['grad_u']:.1e}, "
                f"|E_ab - u.h_op(v)| = {chk.numbers['hess_uv']:.1e} (allowed "
                f"{IDENTITY_TOL:g}); the h_op ladder "
                f"(informational) misses T by {chk.numbers['hop_miss']:.2e}")
            if show_ladders or not chk.ok:
                for line in chk.table:
                    log(line)
            checks.append(chk)
    return checks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=CASES, required=True)
    ap.add_argument("--workers", type=int, default=1,
                    help="processes for the multipliers' kappa sweep")
    ap.add_argument("--seeded-multipliers", action="store_true",
                    help="a seeded unit (Z, z) instead of the SC-NEVPT2 gradient's")
    ap.add_argument("--show-ladders", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    print(f"[{args.case}] level 2's CASSCF third-derivative term, Z . d^2 g / dtheta^2")
    if args.seeded_multipliers:
        st = ngr.build_state(args.case, dict(ngr.CASES[args.case]), reference="casscf",
                             mo_from=ngr.REPO / "golden" / f"{args.case}_casscf.mogold")
        Z = None
    else:
        st, Z, info = multipliers(args.case, args.workers)
    th = Third(ch.mol_at(st.geom, st.basis), st.mo_coeff, st.ci, st.ncore, st.nact,
               st.nelecas)
    if Z is None:
        Z = seeded_multipliers(th)
        print(f"  seeded unit (Z, z), seed {MULTIPLIER_SEED}")
    elif not np.array_equal(info["var"], th.var):
        raise SystemExit("the multipliers' orbital variables are not gen_g_hop's here")
    print(f"  n_mo = {th.nmo} = {th.ncore} core + {th.ncas} active + "
          f"{th.nmo - th.nocc} virtual; {th.nvar} orbital variables, {th.ndet} determinants; "
          f"|g_all| at the state = {np.linalg.norm(th.g_hop()[0]):.1e}")
    print(f"  ladder h = 2^-{LADDER[0]} .. 2^-{LADDER[1]}, the four-point quotient of "
          "Z . g_all; gate |Richardson - T| <= " + f"{LADDER_TOL:g}")
    checks = run_checks(th, Z, show_ladders=args.show_ladders)
    bad = [c for c in checks if not c.ok]
    worst = max(c.numbers["err"] for c in checks)
    print(f"\n{len(checks) - len(bad)}/{len(checks)} pass; worst |Richardson - T| = "
          f"{worst:.2e}, worst |T(u,v) - T(v,u)| = "
          f"{max(c.numbers['symmetry'] for c in checks):.1e}; "
          f"{time.perf_counter() - t0:.0f} s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
