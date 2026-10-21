#!/usr/bin/env python3
r"""The CASSCF reference's **skeleton** nuclear derivatives, offline: the
perturbed Hamiltonians ``dh/dR_A``, ``d(pq|rs)/dR_A`` in the MO basis and the
second partial ``L_RR`` (docs/hessian-theory.md, section 8, item 1).

"Skeleton" means the wavefunction parameters are held fixed **in the
connected sense**: the CI vector does not move, and the orbitals ride the
frozen orthonormal path of ``nevpt2_grad.py``'s :func:`frozen_coeff`::

    C(R) = S(R)^{-1/2} S(R0)^{1/2} C(R0)  =  C0 (1 + t(R)),
    t(R) = C0^T S0 (S(R)^{-1/2} S0^{1/2} - 1) C0

so ``C(R)^T S(R) C(R) = 1`` at every geometry. Along that path the energy
is one scalar::

    E(R) = E_nuc(R) + sum_pq h_pq(R) D_pq + 1/2 sum_pqrs (pq|rs)(R) G_pqrs

with ``D``, ``G`` the CASSCF state's full-space spin-summed 1- and 2-RDM
(core included; nonzero on the occupied = core + active range only). Its
first derivative is the **perturbed Hamiltonian** contracted with the
densities, and its second is ``L_RR`` -- the term of section 8's assembly
that carries no parameter response. Neither assumes stationarity: every
formula below holds for any orthonormal ``C0`` and any CI vector, and the
hand cases (``tests/test_casscf_hessian.py``) exercise them on a CASCI that
is not a CASSCF stationary point, where the connection's first-order term
does not vanish.

THE PERTURBED HAMILTONIAN (one Cartesian coordinate ``A``)
-----------------------------------------------------------

With ``h~ = C0^T h C0`` and ``g~`` the MO integrals at the reference::

    h^(A)      = C0^T h^A C0 + t^A^T h~ + h~ t^A
    (pq|rs)^(A) = (pq|rs)^A_skel + sum_t [t^A_tp (tq|rs) + t^A_tq (pt|rs)
                                          + t^A_tr (pq|ts) + t^A_ts (pq|rt)]

``h^A`` is ``pyscf.grad.rhf``'s ``hcore_generator`` (kinetic, nuclear
attraction and its operator derivative); the skeleton two-electron term is
``-int2e_ip1`` on each of the four index positions; ``t^A`` is the
connection's first derivative, from ``dS/dR_A`` (:meth:`Skeleton.connection`).

L_RR
----

Expanding ``E`` to second order in two coordinates ``A``, ``B``::

    L_AB = E_nuc^AB                                      hess_nuc
         + sum h^AB P + 1/2 sum (mu nu|la si)^AB G_AO    the skeleton second derivative
         + 2 sum_{p, a} t^AB_pa F_pa                     the connection, second order
         + 2 sum t^A_pa F^B_pa + 2 sum t^B_pa F^A_pa     connection x skeleton
         + sum t^A_pa M_{pa,qb} t^B_qb                   connection x connection

with ``F_pa = sum_b h~_pb D_ab + sum_bcd (pb|cd) G_abcd`` the generalized
Fock (``a`` occupied), ``F^B`` the same at the skeleton derivative integrals,
and::

    M_{pa,qb} = 2 h~_pq D_ab + 2 sum_cd (pq|cd) G_abcd + 4 sum_cd (pc|qd) G_acbd

(``G`` symmetrised over the eight permutations the integrals have, which
changes no contraction). The skeleton second derivative uses
``pyscf.hessian.rhf``'s drivers: its ``hcore_generator`` for ``h^AB``,
``hess_nuc``, and the integrals ``int2e_ipip1`` (both derivatives on one
function, ``A = B``), ``int2e_ipvip1`` (on the two functions of one charge
distribution) and ``int2e_ip1ip2`` (one on each), contracted with the AO
2-RDM ``G_AO``, with weights 2, 2 and 4 from the four positions' symmetry.

THE CHECKS (``main``)
---------------------

* ``E(R0)`` rebuilt from the densities against the state's own energy;
* the gradient (``<perturbed Hamiltonian, densities>``) against a
  Richardson ladder of ``E`` along every Cartesian coordinate, the frozen
  path re-evaluated at each displaced geometry
  (:func:`nevpt2_grad.ladder_check`'s run rule);
* ``L_RR``'s diagonal against the second-difference ladder of the same
  energies, and ``d1^T L d2`` along seeded whole-molecule direction pairs
  (and, up to :data:`PAIR_LADDER_MAX_COORDS` coordinates, every coordinate
  pair) against four-point mixed ladders of ``E``;
* **every element** of ``L_RR`` against a ladder of the analytic gradient
  along its column's coordinate, re-evaluated on the path at each rung
  (:func:`path_gradient`; ``--no-columns`` skips it). A first difference of
  a gradient keeps more rungs above round-off than a four-point difference
  of an energy, which runs out of fourfold halvings on the small
  off-diagonal elements;
* the translational sum rules ``sum_A dE/dR_A = 0`` and
  ``sum_B L_{R_A R_B} = 0`` per Cartesian direction pair, and ``L``'s
  symmetry -- exact identities, read to round-off;
* at a converged CASSCF state, the gradient against ``pyscf.grad.casscf``'s
  (which assumes stationarity; the two differ by the orbital gradient's
  size), informational.

The measured numbers are in docs/reference-data.md, "The CASSCF skeleton
nuclear derivatives".

Usage::

    uv run python reference_data/casscf_hessian.py --case n2_ccpvdz_cas44 --workers 12
    uv run python reference_data/casscf_hessian.py --case lif_def2svp_cas46 --workers 12
    uv run python reference_data/casscf_hessian.py --case acrolein_def2svp_cas65 --workers 12

The state is the case's committed CASSCF orbitals,
``golden/<case>_casscf.mogold``, through ``generate_golden._build_state``'s
own ``--mo-from`` path (its fingerprint, symmetry and stationarity gates
included). ``--out`` writes the gradient, ``L_RR`` and its terms as an
``.npz``; nothing is written by default and no committed file is touched.
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads
# (docs/reference-data.md, "Reproducibility").
import generate_golden as gg
import nevpt2_grad as ngr
import nevpt2_numgrad as ngrad

# isort: split
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyscf import ao2mo, fci, gto, scf
from pyscf.grad import rhf as rhf_grad
from pyscf.hessian import rhf as rhf_hess

#: The first-order ladder, ``h = 2^-k`` bohr: ``nevpt2_grad.NUC_LADDER``'s
#: coarse end, cut where the energy's round-off (``~1e-14 Eh / h``) is still
#: far under the gate.
GRAD_LADDER = (3, 13)

#: The second-order ladders. A second difference's round-off grows as
#: ``1/h^2``, so the ladder starts coarser and stops earlier.
HESS_LADDER = (1, 9)

#: The column ladders: central differences of the analytic gradient, whose
#: round-off grows only as ``1/h``. Every element of ``L_RR`` is read off
#: them; the energy's own four-point ladders are too short for the small
#: off-diagonal elements (round-off meets truncation within three halvings).
COLUMN_LADDER = (3, 13)

#: Both gates, Eh/bohr and Eh/bohr^2: ``nevpt2_grad.LADDER_TOL``, the
#: precision the gradient itself is held to.
LADDER_TOL = ngr.LADDER_TOL

#: Translational sum rules and ``L``'s symmetry are exact identities of the
#: frozen path (a rigid translation moves no overlap, so ``t = 0`` along
#: it); this is round-off headroom, not a tolerance on a physical quantity.
SUM_RULE_TOL = 1e-9

#: Up to this many Cartesian coordinates (the diatomics) every coordinate
#: pair also gets a four-point ``E`` ladder. Above it the four-point ladders
#: are the seeded pairs alone: on acrolein's 276 pairs a few small elements
#: run out of fourfold halvings before round-off (docs/reference-data.md,
#: "The CASSCF skeleton nuclear derivatives"), and the gradient columns read
#: every element instead.
PAIR_LADDER_MAX_COORDS = 9

#: Seeds of the four-point energy ladders' direction pairs.
DIRECTION_SEEDS = (0, 1, 2)


def mol_at(geom, basis: str):
    """``generate_golden._build_mol``'s molecule **without** symmetry: the
    same atoms, basis, charge and spin, in the input frame. The committed
    geometries are already in their symmetry frame (``_build_mol`` moves
    nothing on them), so the AO basis is the same one the state's orbitals
    are written in -- :func:`check_reference_energy` confirms it -- and a
    symmetry-lowering displacement cannot re-orient it."""
    return gto.M(atom=geom, basis=basis, spin=0, verbose=0, symmetry=False)


def full_rdms(ncore: int, ncas: int, nelecas, ci: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The full-space spin-summed 1- and 2-RDM on the occupied (core +
    active) range, ``E = sum h D + 1/2 sum (pq|rs) G``, with ``G`` symmetrised
    over the eight permutations real integrals have."""
    dm1a, dm2a = fci.direct_spin1.make_rdm12(ci, ncas, nelecas)
    nocc = ncore + ncas
    c, a = slice(0, ncore), slice(ncore, nocc)
    D = np.zeros((nocc, nocc))
    D[c, c] = 2.0 * np.eye(ncore)
    D[a, a] = dm1a
    G = np.zeros((nocc,) * 4)
    i = np.arange(ncore)
    for k in range(ncore):
        G[i, i, k, k] += 4.0
        G[i, k, k, i] -= 2.0
        G[k, k, a, a] += 2.0 * dm1a
        G[a, a, k, k] += 2.0 * dm1a
        G[k, a, a, k] -= dm1a
        G[a, k, k, a] -= dm1a
    G[a, a, a, a] = dm2a
    G = (G + G.transpose(1, 0, 3, 2)) / 2
    G = (G + G.transpose(2, 3, 0, 1)) / 2
    G = (G + G.transpose(1, 0, 2, 3)) / 2
    G = (G + G.transpose(0, 1, 3, 2)) / 2
    return D, G


@dataclass
class Reference:
    """A fixed wavefunction: orbitals ``C0`` at geometry ``geom``, and the
    densities its CI vector gives. Everything this module differentiates is
    a function of the geometry alone once this is fixed."""

    geom: list
    basis: str
    C0: np.ndarray
    ncore: int
    ncas: int
    D: np.ndarray
    G: np.ndarray
    #: the energy the state's own solver reported, for the rebuild check
    e_ref: float | None = None

    @property
    def nocc(self) -> int:
        return self.ncore + self.ncas


def reference_from_case(case: str) -> tuple[Reference, object]:
    """The committed CASSCF state of ``case`` (``golden/<case>_casscf.mogold``),
    through ``generate_golden._build_state``'s ``--mo-from`` path and every
    refusal it runs. Returns the reference and the CASCI object (for the
    ``pyscf.grad.casscf`` comparison)."""
    spec = ngrad.CASES[case]
    mo_from = ngrad.REPO / "golden" / f"{case}_casscf.mogold"
    geom = ngrad.read_xyz(ngrad.REPO / spec["geometry"])
    mc, _pt = gg._build_state(geom, spec["basis"], spec["ncas"], spec["nelecas"], None,
                              cas_irreps=spec.get("cas_irreps"),
                              core_irreps=spec.get("core_irreps"), mo_from=mo_from)
    D, G = full_rdms(mc.ncore, mc.ncas, mc.nelecas, np.asarray(mc.ci))
    ref = Reference(geom=geom, basis=spec["basis"], C0=np.asarray(mc.mo_coeff).copy(),
                    ncore=mc.ncore, ncas=mc.ncas, D=D, G=G, e_ref=float(mc.e_tot))
    return ref, mc


def path_energy(ref: Reference, geom, S0_half: np.ndarray | None = None) -> float:
    """``E`` at ``geom`` along the frozen orthonormal path anchored at
    ``ref``: ``nevpt2_grad.frozen_coeff``'s coefficients, the fixed
    densities, everything else rebuilt from scratch at ``geom``."""
    if S0_half is None:
        S0_half = ngr.sym_power(mol_at(ref.geom, ref.basis).intor("int1e_ovlp"), 0.5)
    mol = mol_at(geom, ref.basis)
    C = ngr.frozen_coeff(ref.C0, S0_half, mol.intor("int1e_ovlp"))[:, : ref.nocc]
    h = C.T @ scf.hf.get_hcore(mol) @ C
    g = ao2mo.restore(1, ao2mo.kernel(mol, C), ref.nocc)
    return float(mol.energy_nuc() + np.einsum("pq,pq", h, ref.D)
                 + 0.5 * np.einsum("pqrs,pqrs", g, ref.G))


def _tx(X: np.ndarray, Ca, Cb, Cc, Cd) -> np.ndarray:
    """``einsum('xmnls,ma,nb,lc,sd->xabcd')`` as four contractions."""
    T = np.tensordot(X, Cd, axes=([4], [0]))
    T = np.tensordot(T, Cc, axes=([3], [0]))  # x m n d c
    T = np.tensordot(T, Cb, axes=([2], [0]))  # x m d c b
    T = np.tensordot(T, Ca, axes=([1], [0]))  # x d c b a
    return T.transpose(0, 4, 3, 2, 1)


class Skeleton:
    """The perturbed Hamiltonians and ``L_RR`` of one :class:`Reference`, at
    its own geometry. Coordinates are indexed ``k = 3 * atom + axis``."""

    def __init__(self, ref: Reference):
        self.ref = ref
        mol = mol_at(ref.geom, ref.basis)
        self.mol = mol
        nao = mol.nao_nr()
        C0 = ref.C0
        if C0.shape != (nao, nao):
            raise SystemExit(f"Skeleton: C0 is {C0.shape}, not square in the {nao} AOs: the "
                             "connection t = C0^-1 dC needs a complete MO basis")
        self.nao = self.nmo = nao
        self.natm = mol.natm
        self.ncoord = 3 * mol.natm
        self.aoslices = mol.aoslice_by_atom()
        S0 = mol.intor("int1e_ovlp")
        defect = np.abs(C0.T @ S0 @ C0 - np.eye(nao)).max()
        if defect > 1e-8:
            raise SystemExit(f"Skeleton: C0 is not S-orthonormal (defect {defect:.1e})")
        self.S0 = S0
        self.S0_half = ngr.sym_power(S0, 0.5)
        self.sigma, self.V = np.linalg.eigh(S0)
        #: ``O = sigma^{1/2} V^T C0``, orthogonal: ``t = C0^T S0 (V z V^T) S0^{1/2}
        #: C0 = O^T sigma^{1/2} z O`` for ``z`` a derivative of ``S^{-1/2}`` in
        #: ``S0``'s eigenbasis.
        self.O = (np.sqrt(self.sigma)[:, None] * self.V.T) @ C0
        self.hcore = scf.hf.get_hcore(mol)
        self.h_mo = C0.T @ self.hcore @ C0
        self._mo_cache: dict = {}
        mf = scf.RHF(mol)
        self._hcore1 = rhf_grad.Gradients(mf).hcore_generator(mol)
        self._hcore2 = rhf_hess.Hessian(mf).hcore_generator(mol)
        self.occ = slice(0, ref.nocc)
        self.F = self.gen_fock(self.h_mo, self.mo_eri(slice(None), self.occ, self.occ,
                                                      self.occ))

    # -- MO integrals at the reference ------------------------------------

    def mo_eri(self, *ranges: slice) -> np.ndarray:
        """``(pq|rs)`` at the reference, each index over a slice of MOs."""
        key = tuple((r.start, r.stop) for r in ranges)
        got = self._mo_cache.get(key)
        if got is None:
            Cs = [self.ref.C0[:, r] for r in ranges]
            shape = [c.shape[1] for c in Cs]
            got = ao2mo.general(self.mol, Cs, compact=False).reshape(shape)
            self._mo_cache[key] = got
        return got

    def gen_fock(self, h: np.ndarray, g_xooo: np.ndarray) -> np.ndarray:
        """``F_pa = sum_b h_pb D_ab + sum_bcd (pb|cd) G_abcd``: ``(nmo, nocc)``."""
        D, G = self.ref.D, self.ref.G
        return h[:, self.occ] @ D + np.einsum("pbcd,abcd->pa", g_xooo, G, optimize=True)

    # -- the overlap and the connection -----------------------------------

    def ovlp1(self) -> np.ndarray:
        """``dS/dR_k``, ``(3N, nao, nao)``."""
        ip = self.mol.intor("int1e_ipovlp", comp=3)
        out = np.zeros((self.ncoord, self.nao, self.nao))
        for ia, (_s0, _s1, p0, p1) in enumerate(self.aoslices):
            for x in range(3):
                d = np.zeros((self.nao, self.nao))
                d[p0:p1] = ip[x, p0:p1]
                out[3 * ia + x] = -(d + d.T)
        return out

    def ovlp2(self) -> np.ndarray:
        """``d^2 S / dR_k dR_l``, ``(3N, 3N, nao, nao)``."""
        n = self.nao
        ipip = self.mol.intor("int1e_ipipovlp", comp=9).reshape(3, 3, n, n)
        ipvip = self.mol.intor("int1e_ipovlpip", comp=9).reshape(3, 3, n, n)
        out = np.zeros((self.ncoord, self.ncoord, n, n))
        sl = self.aoslices
        for ia in range(self.natm):
            a0, a1 = sl[ia][2], sl[ia][3]
            for ib in range(self.natm):
                b0, b1 = sl[ib][2], sl[ib][3]
                for x in range(3):
                    for y in range(3):
                        d = np.zeros((n, n))
                        if ia == ib:
                            d[a0:a1] += ipip[x, y, a0:a1]
                        # <d_x mu | d_y nu>, mu on A and nu on B
                        d[a0:a1, b0:b1] += ipvip[x, y, a0:a1, b0:b1]
                        out[3 * ia + x, 3 * ib + y] = d
        # The placements with the derivatives on nu instead of mu are the AO
        # transposes of the above (<d_y mu|d_x nu> = <d_x nu|d_y mu>).
        return out + out.transpose(0, 1, 3, 2)

    def _to_t(self, z: np.ndarray) -> np.ndarray:
        return self.O.T @ (np.sqrt(self.sigma)[:, None] * z) @ self.O

    def connection(self, second: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        """``t^k`` and ``t^kl``: the frozen path's coefficient derivatives in
        the reference MO basis, ``dC/dR = C0 t``. With ``Z = S^{-1/2}`` and
        ``Q = Z^2 = S^{-1}``, in ``S0``'s eigenbasis (where ``Z0`` is
        diagonal) ``Z' Z + Z Z' = Q'`` is solved elementwise, once for the
        first derivative and once for the second with ``Z^k Z^l + Z^l Z^k``
        moved to the right-hand side. ``second=False`` skips ``t^kl``
        (returned as ``None``)."""
        sig, V = self.sigma, self.V
        qd = 1.0 / sig
        den = 1.0 / np.sqrt(sig)
        den = den[:, None] + den[None, :]
        s1 = np.einsum("mi,kmn,nj->kij", V, self.ovlp1(), V, optimize=True)
        q1 = -qd[None, :, None] * s1 * qd[None, None, :]
        z1 = q1 / den
        t1 = np.array([self._to_t(z) for z in z1])
        if not second:
            return t1, None
        s2 = np.einsum("mi,klmn,nj->klij", V, self.ovlp2(), V, optimize=True)
        # Q^kl = Q S^k Q S^l Q + Q S^l Q S^k Q - Q S^kl Q, Q diagonal here.
        qs = qd[None, :, None] * s1  # (Q S^k)
        a = np.einsum("kij,ljm->klim", qs, qs) * qd[None, None, None, :]
        q2 = a + a.transpose(1, 0, 2, 3) - qd[None, None, :, None] * s2 * qd[None, None, None, :]
        zz = np.einsum("kij,ljm->klim", z1, z1)
        z2 = (q2 - zz - zz.transpose(1, 0, 2, 3)) / den
        t2 = np.array([[self._to_t(z2[k, m]) for m in range(self.ncoord)]
                       for k in range(self.ncoord)])
        return t1, t2

    # -- the perturbed Hamiltonian -----------------------------------------

    def hcore1(self) -> np.ndarray:
        """``dh/dR_k`` in the AO basis, ``(3N, nao, nao)``."""
        return np.concatenate([self._hcore1(ia) for ia in range(self.natm)])

    def eri1_skel(self, ia: int, *ranges: slice) -> np.ndarray:
        """``d(pq|rs)/dR_{ia,x}`` with the MO coefficients held fixed (the
        skeleton term alone), ``(3, n1, n2, n3, n4)``: ``-int2e_ip1`` with
        the derivative on each of the four positions in turn."""
        _s0, _s1, p0, p1 = self.aoslices[ia]
        sh0, sh1 = self.aoslices[ia][0], self.aoslices[ia][1]
        nbas = self.mol.nbas
        X = self.mol.intor("int2e_ip1", comp=3, aosym="s1",
                           shls_slice=(sh0, sh1, 0, nbas, 0, nbas, 0, nbas))
        C = [self.ref.C0[:, r] for r in ranges]
        CA = [c[p0:p1] for c in C]
        t = _tx(X, CA[0], C[1], C[2], C[3])
        t = t + _tx(X, CA[1], C[0], C[2], C[3]).transpose(0, 2, 1, 3, 4)
        t = t + _tx(X, CA[2], C[3], C[0], C[1]).transpose(0, 3, 4, 1, 2)
        t = t + _tx(X, CA[3], C[2], C[0], C[1]).transpose(0, 3, 4, 2, 1)
        return -t

    def perturbed_h1(self, k: int, t1: np.ndarray | None = None,
                     h1_ao: np.ndarray | None = None) -> np.ndarray:
        """``h^(k)``: the MO one-electron Hamiltonian's derivative along the
        frozen path, ``(nmo, nmo)``, skeleton plus connection."""
        if t1 is None:
            t1 = self.connection(second=False)[0]
        if h1_ao is None:
            h1_ao = self.hcore1()
        t = t1[k]
        return self.ref.C0.T @ h1_ao[k] @ self.ref.C0 + t.T @ self.h_mo + self.h_mo @ t

    def perturbed_eri(self, k: int, ranges: tuple[slice, slice, slice, slice],
                      t1: np.ndarray | None = None,
                      skel: np.ndarray | None = None) -> np.ndarray:
        """``(pq|rs)^(k)``: the MO two-electron integrals' derivative along
        the frozen path over ``ranges`` (four slices of MOs), skeleton plus
        connection. ``skel``, if given, is :meth:`eri1_skel`'s component
        ``k % 3`` over the same ranges."""
        if t1 is None:
            t1 = self.connection(second=False)[0]
        if skel is None:
            skel = self.eri1_skel(k // 3, *ranges)[k % 3]
        t = t1[k]
        allmo = slice(None)
        r1, r2, r3, r4 = ranges
        out = skel.copy()
        out += np.einsum("tp,tqrs->pqrs", t[:, r1], self.mo_eri(allmo, r2, r3, r4))
        out += np.einsum("tq,ptrs->pqrs", t[:, r2], self.mo_eri(r1, allmo, r3, r4))
        out += np.einsum("tr,pqts->pqrs", t[:, r3], self.mo_eri(r1, r2, allmo, r4))
        out += np.einsum("ts,pqrt->pqrs", t[:, r4], self.mo_eri(r1, r2, r3, allmo))
        return out

    # -- the first derivative ----------------------------------------------

    def gradient(self) -> np.ndarray:
        """``dE/dR``, ``(natm, 3)``: ``E_nuc``'s derivative plus the perturbed
        Hamiltonian (:meth:`perturbed_h1`, :meth:`perturbed_eri` over the
        occupied range) contracted with the densities."""
        o = self.occ
        t1, _ = self.connection(second=False)
        h1 = self.hcore1()
        D, G = self.ref.D, self.ref.G
        de = np.array(rhf_grad.grad_nuc(self.mol)).ravel()
        for ia in range(self.natm):
            skel = self.eri1_skel(ia, o, o, o, o)
            for x in range(3):
                k = 3 * ia + x
                hk = self.perturbed_h1(k, t1, h1)[o, o]
                gk = self.perturbed_eri(k, (o, o, o, o), t1, skel[x])
                de[k] += np.einsum("pq,pq", hk, D) + 0.5 * np.einsum("pqrs,pqrs", gk, G)
        return de.reshape(self.natm, 3)

    # -- the second derivative ---------------------------------------------

    def ao_rdm2(self) -> np.ndarray:
        """``G`` in the AO basis, ``(nao,) * 4``."""
        Co = self.ref.C0[:, self.occ]
        return _tx(self.ref.G[None], Co.T, Co.T, Co.T, Co.T)[0]

    def eri2_skel(self) -> np.ndarray:
        """``1/2 sum (mu nu|la si)^kl G_AO``: the two-electron skeleton
        second derivative, ``(3N, 3N)``. Weights from the four index
        positions: 2 for ``int2e_ipip1`` (both derivatives on one function,
        so ``A = B``), 2 for ``int2e_ipvip1`` (on ``mu`` and ``nu``), 4 for
        ``int2e_ip1ip2`` (on ``mu`` and ``la``); two nuclear derivatives
        carry two minus signs."""
        G = self.ao_rdm2()
        nbas = self.mol.nbas
        n = self.nao
        H = np.zeros((self.natm, 3, self.natm, 3))
        sl = self.aoslices
        for ia in range(self.natm):
            sh0, sh1, p0, p1 = sl[ia]
            shls = (sh0, sh1, 0, nbas, 0, nbas, 0, nbas)
            Ga = G[p0:p1]
            X = self.mol.intor("int2e_ipip1", comp=9, aosym="s1", shls_slice=shls)
            H[ia, :, ia, :] += 2.0 * np.einsum("xmnls,mnls->x", X, Ga,
                                               optimize=True).reshape(3, 3)
            X = self.mol.intor("int2e_ipvip1", comp=9, aosym="s1", shls_slice=shls)
            v = 2.0 * np.einsum("xmnls,mnls->xn", X, Ga, optimize=True).reshape(3, 3, n)
            X = self.mol.intor("int2e_ip1ip2", comp=9, aosym="s1", shls_slice=shls)
            v += 4.0 * np.einsum("xmnls,mnls->xl", X, Ga, optimize=True).reshape(3, 3, n)
            del X
            for ib in range(self.natm):
                q0, q1 = sl[ib][2], sl[ib][3]
                H[ia, :, ib, :] += v[:, :, q0:q1].sum(axis=2)
        return H.reshape(self.ncoord, self.ncoord)

    def hessian(self) -> tuple[np.ndarray, dict]:
        """``L_RR``, ``(3N, 3N)``, and its terms by name (each ``(3N, 3N)``)."""
        o = self.occ
        allmo = slice(None)
        nc, nmo, nocc = self.ncoord, self.nmo, self.ref.nocc
        D, G = self.ref.D, self.ref.G
        t1, t2 = self.connection()
        terms: dict = {}
        terms["nuc"] = rhf_hess.hess_nuc(self.mol).transpose(0, 2, 1, 3).reshape(nc, nc)
        P = self.ref.C0[:, o] @ D @ self.ref.C0[:, o].T
        h2 = np.zeros((self.natm, 3, self.natm, 3))
        for ia in range(self.natm):
            for ib in range(self.natm):
                h2[ia, :, ib, :] = np.einsum("xymn,mn->xy", self._hcore2(ia, ib), P)
        terms["h_skel"] = h2.reshape(nc, nc)
        terms["eri_skel"] = self.eri2_skel()
        # the connection at second order, against the reference's generalized Fock
        terms["conn2"] = 2.0 * np.einsum("klpa,pa->kl", t2[:, :, :, o], self.F, optimize=True)
        # connection x skeleton: F^k at the skeleton derivative integrals
        h1 = self.hcore1()
        Fk = np.zeros((nc, nmo, nocc))
        for ia in range(self.natm):
            skel = self.eri1_skel(ia, allmo, o, o, o)
            for x in range(3):
                k = 3 * ia + x
                Fk[k] = self.gen_fock(self.ref.C0.T @ h1[k] @ self.ref.C0, skel[x])
        T = t1[:, :, o].reshape(nc, -1)
        cross = 2.0 * T @ Fk.reshape(nc, -1).T
        terms["conn_x_skel"] = cross + cross.T
        # connection x connection
        g_vvoo = self.mo_eri(allmo, allmo, o, o)
        g_vovo = self.mo_eri(allmo, o, allmo, o)
        M = (2.0 * np.einsum("pq,ab->paqb", self.h_mo, D)
             + 2.0 * np.einsum("pqcd,abcd->paqb", g_vvoo, G, optimize=True)
             + 4.0 * np.einsum("pcqd,acbd->paqb", g_vovo, G, optimize=True))
        M = M.reshape(nmo * nocc, nmo * nocc)
        terms["conn_x_conn"] = T @ M @ T.T
        L = sum(terms.values())
        return L, terms


# --------------------------------------------------------------------------
# The ladders
# --------------------------------------------------------------------------


def _energy_task(task: tuple) -> tuple:
    key, ref, S0_half, dvec = task
    return key, path_energy(ref, ngrad.displace_vec(ref.geom, dvec), S0_half)


def energies(ref: Reference, S0_half: np.ndarray, points: dict, workers: int) -> dict:
    """``E`` at every displacement in ``points`` (``key -> (natm, 3)`` bohr),
    ``workers`` processes at a time."""
    tasks = [(k, ref, S0_half, d) for k, d in points.items()]
    if workers <= 1:
        return dict(_energy_task(t) for t in tasks)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_energy_task, tasks, chunksize=4))


def unit(natm: int, k: int) -> np.ndarray:
    d = np.zeros((natm, 3))
    d.flat[k] = 1.0
    return d


def seeded_direction(natm: int, seed: int) -> np.ndarray:
    """A unit whole-molecule displacement, every coordinate moved."""
    d = np.random.default_rng(seed).standard_normal((natm, 3))
    return d / np.linalg.norm(d)


def judge(name: str, fd: list[float], steps: list[float], ref: float) -> ngr.Check:
    """:func:`nevpt2_grad.ladder_check`, plus the one case its run rule
    cannot read: a component **zero by symmetry**. The energy is then even
    (or the mixed quotient odd) along the displacement, every quotient is
    zero to round-off at every rung, and there is no truncation error to
    walk down. Such a ladder passes as ``zero`` iff the analytic value is
    zero to :data:`SUM_RULE_TOL` and every rung is under :data:`LADDER_TOL`;
    anything else goes to the run rule."""
    c = ngr.ladder_check(name, fd, steps, ref, LADDER_TOL)
    if not c.ok and abs(ref) <= SUM_RULE_TOL and max(abs(v) for v in fd) <= LADDER_TOL:
        worst = max(abs(v) for v in fd)
        c = ngr.Check(name, True, f"zero by symmetry: every rung under {worst:.1e}, "
                      f"ref {ref:+.1e}", c.table,
                      dict(c.numbers, err=max(worst, abs(ref)), zero=True))
    return c


def _gradient_task(task: tuple) -> tuple:
    key, ref, S0_half, dvec = task
    return key, path_gradient(ref, ngrad.displace_vec(ref.geom, dvec), S0_half)


def path_gradient(ref: Reference, geom, S0_half: np.ndarray) -> np.ndarray:
    """The analytic :meth:`Skeleton.gradient` at ``geom`` **on** ``ref``'s
    frozen path: re-anchored at ``C(R) = frozen_coeff(C0)`` with the same
    densities. The path is consistent -- ``S(R')^{-1/2} S(R)^{1/2} C(R) =
    S(R')^{-1/2} S0^{1/2} C0`` -- so this is the slope of ``ref``'s own
    path energy at ``geom``, and its difference quotients ladder ``L_RR``."""
    mol = mol_at(geom, ref.basis)
    C = ngr.frozen_coeff(ref.C0, S0_half, mol.intor("int1e_ovlp"))
    moved = Reference(geom=geom, basis=ref.basis, C0=C, ncore=ref.ncore, ncas=ref.ncas,
                      D=ref.D, G=ref.G)
    return Skeleton(moved).gradient().ravel()


def gradients(ref: Reference, S0_half: np.ndarray, points: dict, workers: int) -> dict:
    """:func:`path_gradient` at every displacement in ``points``."""
    tasks = [(k, ref, S0_half, d) for k, d in points.items()]
    if workers <= 1:
        return dict(_gradient_task(t) for t in tasks)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_gradient_task, tasks))


def run_checks(sk: Skeleton, de: np.ndarray, L: np.ndarray, workers: int,
               columns: bool = True) -> list:
    """Every ladder: the gradient and ``L``'s diagonal against ``E`` along
    each coordinate, ``d^T L d'`` against four-point ``E`` ladders along the
    seeded direction pairs, and (``columns``) every element of ``L`` against
    a ladder of the analytic gradient along its column's coordinate."""
    ref = sk.ref
    natm, nc = sk.natm, sk.ncoord
    gsteps = ngr.ladder_steps(GRAD_LADDER)
    hsteps = ngr.ladder_steps(HESS_LADDER)
    csteps = ngr.ladder_steps(COLUMN_LADDER)
    dirs: dict = {f"e{k}": unit(natm, k) for k in range(nc)}
    pairs: list = []
    if nc <= PAIR_LADDER_MAX_COORDS:
        pairs = [(f"e{k}", f"e{m}") for k in range(nc) for m in range(k + 1, nc)]
    for s in DIRECTION_SEEDS:
        dirs[f"d{s}"] = seeded_direction(natm, 100 + s)
        dirs[f"d{s}'"] = seeded_direction(natm, 200 + s)
        pairs.append((f"d{s}", f"d{s}'"))
    points = {("0",): np.zeros((natm, 3))}
    for name, d in dirs.items():
        for h in (gsteps if name.startswith("e") else []) + hsteps:
            for s in (1, -1):
                points[(name, s, h)] = s * h * d
    for a, b in pairs:
        for h in hsteps:
            for sa in (1, -1):
                for sb in (1, -1):
                    points[(a, b, sa, sb, h)] = h * (sa * dirs[a] + sb * dirs[b])
    t0 = time.time()
    E = energies(ref, sk.S0_half, points, workers)
    print(f"  {len(points)} path energies in {time.time() - t0:.1f} s")
    e0 = E[("0",)]
    checks = []
    for k in range(nc):
        fd = [(E[(f"e{k}", 1, h)] - E[(f"e{k}", -1, h)]) / (2 * h) for h in gsteps]
        checks.append(judge(f"dE/dR[{k // 3},{'xyz'[k % 3]}]", fd, gsteps, float(de.flat[k])))
    for name, d in dirs.items():
        fd = [(E[(name, 1, h)] - 2 * e0 + E[(name, -1, h)]) / h**2 for h in hsteps]
        dv = d.ravel()
        checks.append(judge(f"L[{name},{name}]", fd, hsteps, float(dv @ L @ dv)))
    for a, b in pairs:
        fd = [(E[(a, b, 1, 1, h)] - E[(a, b, 1, -1, h)] - E[(a, b, -1, 1, h)]
               + E[(a, b, -1, -1, h)]) / (4 * h**2) for h in hsteps]
        checks.append(judge(f"L[{a},{b}]", fd, hsteps,
                            float(dirs[a].ravel() @ L @ dirs[b].ravel())))
    if not columns:
        return checks
    gpoints = {(k, s, h): s * h * unit(natm, k)
               for k in range(nc) for h in csteps for s in (1, -1)}
    t0 = time.time()
    g = gradients(ref, sk.S0_half, gpoints, workers)
    print(f"  {len(gpoints)} path gradients in {time.time() - t0:.1f} s")
    for k in range(nc):
        fd = np.array([(g[(k, 1, h)] - g[(k, -1, h)]) / (2 * h) for h in csteps])
        checks.append(judge_column(f"L[:,{k}] (gradient column)", fd, csteps, L[:, k]))
    return checks


def judge_column(name: str, fd: np.ndarray, steps: list[float], ref: np.ndarray) -> ngr.Check:
    """The run rule on a whole column at once: ``fd[i]`` is the column's
    quotient vector at ``steps[i]``, the rung-to-rung change is its max-norm,
    and the gate is on the **worst element** of the Richardson vector against
    ``ref``. One column is one difference quotient of one gradient, so its
    convergence is read off its largest elements, whose truncation error
    dominates; a small element alone can show fewer than three fourfold
    halvings before round-off (its truncation is tiny) without being any
    less converged."""
    dif = [float(np.abs(fd[i] - fd[i + 1]).max()) for i in range(len(fd) - 1)]
    runs = ngr.fourfold_runs(dif)
    good = [r for r in runs if r[0] >= ngr.LADDER_RUN]
    run, end = good[-1] if good else max(runs, default=(0, 0))
    if run:
        value = (4.0 * fd[end + 1] - fd[end]) / 3.0
        err = float(np.abs(value - ref).max())
    else:
        err = float("inf")
    ok = run >= ngr.LADDER_RUN and err <= LADDER_TOL
    table = [f"      2^-{round(-np.log2(h)):<3d} max|change| "
             + (f"{dif[i - 1]:10.3e}" if i else f"{'-':>10}") for i, h in enumerate(steps)]
    summary = (f"{run} consecutive fourfold halvings (need {ngr.LADDER_RUN}); "
               f"max |Richardson - ref| = {err:.2e} (allowed {LADDER_TOL:g})")
    return ngr.Check(name, ok, summary, table, dict(err=err, run=run))


def sum_rules(de: np.ndarray, L: np.ndarray, natm: int) -> dict:
    """The translational sum rules and ``L``'s symmetry, as maxima."""
    L4 = L.reshape(natm, 3, natm, 3)
    return dict(
        grad_translation=float(np.abs(de.sum(axis=0)).max()),
        hess_translation=float(np.abs(L4.sum(axis=2)).max()),
        hess_symmetry=float(np.abs(L - L.T).max()),
    )


def check_reference_energy(sk: Skeleton) -> float:
    """``|E(R0) - E_state|``: the densities and the AO frame are the state's."""
    e0 = path_energy(sk.ref, sk.ref.geom, sk.S0_half)
    return abs(e0 - sk.ref.e_ref) if sk.ref.e_ref is not None else float("nan")


def casscf_gradient_difference(mc, de: np.ndarray) -> float:
    """``max |dE/dR - pyscf.grad.casscf|`` at a converged CASSCF state: the
    frozen path's gradient equals the stationary one up to the orbital
    gradient's size. Informational."""
    from pyscf import mcscf

    mcs = mcscf.CASSCF(mc._scf, mc.ncas, mc.nelecas)
    mcs.mo_coeff = mc.mo_coeff
    mcs.ci = mc.ci
    g = ngrad.casscf_gradient(_as_converged(mcs, mc))
    return float(np.abs(g - de).max())


def _as_converged(mcs, mc):
    mcs.e_tot = mc.e_tot
    mcs.converged = True
    return mcs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, choices=sorted(ngrad.CASES))
    ap.add_argument("--no-columns", action="store_true",
                    help="skip the gradient-column ladders (every element of L_RR); the "
                    "energy ladders still run")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes for the ladders' displaced energies and gradients")
    ap.add_argument("--out", type=Path, help="write the gradient, L_RR and its terms (.npz)")
    args = ap.parse_args(argv)

    t0 = time.time()
    ref, mc = reference_from_case(args.case)
    sk = Skeleton(ref)
    print(f"== {args.case} _casscf: {sk.natm} atoms, nao {sk.nao}, ncore {ref.ncore}, "
          f"ncas {ref.ncas}  (state built in {time.time() - t0:.1f} s)")
    d_e0 = check_reference_energy(sk)
    print(f"  |E(R0) - E_CASSCF| = {d_e0:.2e}")
    t0 = time.time()
    de = sk.gradient()
    t_grad = time.time() - t0
    t0 = time.time()
    L, terms = sk.hessian()
    t_hess = time.time() - t0
    print(f"  gradient {t_grad:.1f} s, L_RR {t_hess:.1f} s")
    for name, v in terms.items():
        print(f"    {name:<12} max |.| {np.abs(v).max():.6e}")
    rules = sum_rules(de, L, sk.natm)
    for name, v in rules.items():
        print(f"  {name:<18} {v:.2e}  (allowed {SUM_RULE_TOL:g})")
    d_pyscf = casscf_gradient_difference(mc, de)
    print(f"  max |dE/dR - pyscf.grad.casscf| = {d_pyscf:.2e}  (informational)")
    print(f"  ladders (bohr): E first differences 2^-{GRAD_LADDER[0]}..2^-{GRAD_LADDER[1]}, "
          f"E second / four-point differences 2^-{HESS_LADDER[0]}..2^-{HESS_LADDER[1]}, "
          f"gradient columns 2^-{COLUMN_LADDER[0]}..2^-{COLUMN_LADDER[1]}")
    checks = run_checks(sk, de, L, args.workers, columns=not args.no_columns)
    ok = all(rules[k] <= SUM_RULE_TOL for k in rules) and d_e0 <= 1e-9
    worst: dict = {"grad": 0.0, "hess": 0.0, "col": 0.0}
    count: dict = {"grad": 0, "hess": 0, "col": 0}
    nzero = 0
    for c in checks:
        kind = ("grad" if c.name.startswith("dE") else
                "col" if c.name.endswith("(gradient column)") else "hess")
        count[kind] += 1
        if c.numbers.get("zero"):
            nzero += 1
        else:
            worst[kind] = max(worst[kind], c.numbers["err"])
        ok &= c.ok
        if not c.ok:
            print(f"  FAIL {c.name}: {c.summary}")
            print("\n".join(c.table))
    print(f"  {sum(c.ok for c in checks)}/{len(checks)} ladders pass ({nzero} of them zero by "
          f"symmetry); worst |Richardson - analytic| over the nonzero ones (allowed "
          f"{LADDER_TOL:g}):")
    print(f"    dE/dR vs E ladders          ({count['grad']:4d}) {worst['grad']:.2e} Eh/bohr")
    print(f"    L_RR vs E ladders           ({count['hess']:4d}) {worst['hess']:.2e} Eh/bohr^2")
    print(f"    L_RR vs gradient columns    ({count['col']:4d}) {worst['col']:.2e} Eh/bohr^2")
    if args.out is not None:
        np.savez(args.out, case=args.case, gradient=de, hessian=L,
                 **{f"term_{k}": v for k, v in terms.items()})
        print(f"  wrote {args.out}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
