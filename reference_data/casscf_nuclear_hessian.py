#!/usr/bin/env python3
r"""The analytic **CASSCF nuclear Hessian**, offline, and the ``3N``
first-order parameter responses it is assembled from (docs/hessian-theory.md,
section 8, for the bare reference energy, where the multipliers vanish).

The skeleton half -- the perturbed Hamiltonians and ``L_RR`` along
``nevpt2_grad.py``'s frozen orthonormal path -- is ``casscf_hessian.py``'s
(#215) and is imported, not re-derived. This module adds the responses and
the assembly.

THE PARAMETERS AND THE RESPONSES
--------------------------------

At a geometry ``R`` the wavefunction is parametrised around the frozen path
``C_f(R) = S(R)^{-1/2} S(R0)^{1/2} C0`` exactly as
``pyscf.mcscf.newton_casscf.gen_g_hop`` parametrises a step::

    C(R, x) = C_f(R) exp(K(x)),   K_ij = x_(i,j), K_ji = -x_(i,j)
    c(R, v) = (c0 + v) / |c0 + v|,  v orthogonal to c0

with ``x`` over ``gen_g_hop``'s packed variables (``i > j``: core-active,
core-virtual, active-virtual). Core-core and virtual-virtual rotations are
redundant for the CASSCF energy and the active-active ones are the CI
vector's gauge (docs/gradient-theory.md, sections 1.3, 1.4), so they are
not parameters and their blocks of every response are **exactly zero** --
the gauge the gradient's coupled solve uses (section 2.7). ``gen_g_hop``'s
``g_all = [2 (F - F^T)_packed ; 2 (H - E) c]`` is then exactly ``dE/dTheta``
(``F_pa = sum_b h_pb D_ab + sum_bcd (pb|cd) G_abcd``, ``Skeleton.gen_fock``;
the hand cases check it against ``gen_g_hop`` at a non-stationary point),
and its ``h_op`` is ``d^2 E / dTheta^2`` at the stationary point.

The CASSCF state is ``g_all(Theta(R), R) = 0`` for all ``R``; differentiate::

    J Theta^(1)_B = - dg_all/dR_B,     J = h_op on the coupled space

The right-hand side ``L_{Theta R_B}`` is ``g_all`` evaluated at the
**perturbed Hamiltonian** -- ``g_all`` is linear in the MO integrals at fixed
densities, and along the frozen path the integrals' derivative is
``Skeleton.perturbed_h1`` / ``perturbed_eri``, over ``(all, occ, occ, occ)``
-- which is all of ``F`` and all of the active Hamiltonian
``h_eff_tu = h_tu + sum_i [2 (tu|ii) - (ti|iu)]``, ``(tu|vw)``.

``J`` is built densely, one ``h_op`` product per column, on the coupled
space: every orbital variable, and an orthonormal basis of the CI space
orthogonal to ``c0`` -- the **whole** determinant space, not the state's
symmetry sector, so the responses along symmetry-breaking coordinates exist
too (a symmetry-preserving coordinate's response stays in the sector, and
the run reports the leak). It is symmetrised (its asymmetry is reported) and
LU-factorised **once** for all ``3N`` right-hand sides.

THE ASSEMBLY
------------

::

    d^2 E / dR_A dR_B = L_RR[A, B] + L_{R_A Theta} . Theta^(1)_B

(symmetric because ``J`` is: the second term is ``-Theta_A^T J Theta_B``).

THE CHECKS (``main``)
---------------------

On the case's committed CASSCF state (``golden/<case>_casscf.mogold``
through ``generate_golden._build_state``'s ``--mo-from`` path) in **the
golden's own frame** -- the canonicalised orbitals and CI vector the
``_casscf`` golden and gradient sidecar hold, checked bit for bit -- and on
the same state **polished** in memory to ``|g_all|`` under
:data:`REOPT_G_TOL` (:func:`polished`):

* the exact identities, on both: ``H``'s symmetry, the translational sum
  rule ``sum_B H_AB = 0`` per Cartesian direction pair, ``|J Theta + b|``
  (dense and through ``h_op``), and the rotational sum rule
  (:func:`rotation_rule`; exact only at a stationary state);
* along every **point-group-keeping** direction ``d``
  (:func:`keeps_point_group`: each Cartesian coordinate totally symmetric
  under the reference's operations, and on a diatomic the stretch -- N2's
  only one), the CASSCF is **re-optimised** at ``R0 + s h d`` around the
  polished state, from its frozen-path guess (:func:`reoptimise`), and:

  - ``H d`` -- the polished state's and the committed one's -- against a
    Richardson ladder of ``pyscf.grad.casscf``'s analytic gradient at those
    states;
  - the polished state's response ``sum_B d_B Theta^(1)_B`` -- ``kappa``
    and the CI part separately -- against the ladder of the re-optimised
    parameters, read back in the parametrisation above with the redundancy
    fixed (:func:`gauge_fixed_parameters`: the overlap ``C_f^T S C'``
    decomposed as ``exp(K) W``, ``K`` zero on the three diagonal blocks and
    ``W`` block-diagonal; the CI vector carried through ``W``'s active
    block), and its CI part held inside the state's irrep;

* the committed state's responses against the polished state's, reported:
  the ladders pass through the stationary point, which the committed state
  misses by its residual orbital gradient, and the responses there differ
  by as much (``O(|g_orb| / lambda_min)``).

Every ladder is judged by ``casscf_hessian.judge_column``: the run rule on
the whole quotient vector's max-norm change, and the worst element's
``|Richardson - analytic|`` against :data:`LADDER_TOL`. The measured
numbers are in docs/reference-data.md, "The analytic CASSCF nuclear
Hessian", and the gate's rationale in docs/testing.md, "The analytic CASSCF
nuclear Hessian".

THE RESPONSE SIDECAR (``--write``)
----------------------------------

``golden/<case>_casscf.respgold`` (``NEVPT2R1``,
:func:`golden_format.write_respgold`): the ``3N`` responses as full-range
``(kappa, c)`` directions in the golden's frame -- the direction set the
GPU's bilinear form reads (#222) -- with the Hessian, ``L_RR``, the
gradient, the reference orbitals and CI vector the directions are written
against, and which coordinates keep the point group. Its layout is in
:mod:`golden_format`.

Usage::

    uv run python reference_data/casscf_nuclear_hessian.py --case n2_ccpvdz_cas44 --workers 12
    uv run python reference_data/casscf_nuclear_hessian.py --case hsoh_def2svp_cas66 \
        --workers 12 --write golden/hsoh_def2svp_cas66_casscf.respgold
    uv run python reference_data/casscf_nuclear_hessian.py --case hsoh_def2svp_cas66 \
        --no-ladders --write /tmp/x.respgold   # the sidecar alone, no re-optimisations
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads
# (docs/reference-data.md, "Reproducibility").
import generate_golden as gg

# isort: split
import casscf_hessian as ch
import nevpt2_grad as ngr
import nevpt2_numgrad as ngrad

# isort: split
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import golden_format as gf
import numpy as np
import scipy.linalg
from pyscf import fci, scf
from pyscf.fci import direct_spin1
from pyscf.mcscf import mc1step, newton_casscf

#: The ladder, ``h = 2^-k`` bohr along a unit direction: central differences
#: of re-optimised states (``|g_all|`` <= :data:`REOPT_G_TOL`, a parameter
#: error ~1e-11 / lambda_min that a quotient divides by ``2h``). The run rule
#: reads the finest run of fourfold halvings, so rungs past the noise cost
#: only their re-optimisations.
LADDER = (3, 13)

#: Both gates: Eh/bohr^2 for the Hessian column, 1/bohr for the responses
#: (unit ``kappa``, unit CI vector per bohr). ``nevpt2_grad.LADDER_TOL``, the
#: precision the gradient and ``L_RR`` are held to.
LADDER_TOL = ngr.LADDER_TOL

#: Symmetry and the translational sum rule of the assembled Hessian, and the
#: solve's residual: exact identities, read to round-off.
IDENTITY_TOL = 1e-9

#: The re-optimisations' chord (:func:`converge`) never steps along the
#: reference operator's modes at or under this eigenvalue: LiF's
#: symmetry-breaking saddle pair (-7.3e-3) and near-null pair (2.8e-6),
#: along which every point-group-keeping right-hand side vanishes. The
#: next mode up on every case is above 2e-3.
CHORD_EIG_CUT = 1e-4

#: The rotational sum rule (:func:`rotation_rule`): not exact at a committed
#: state, whose residual orbital gradient (<= 1e-6 by the ``--mo-from``
#: gate) enters it at first order. The skeleton gradient ``de`` is itself
#: off ``pyscf.grad.casscf``'s by that much (docs/reference-data.md, "The
#: CASSCF skeleton nuclear derivatives").
ROTATION_TOL = 1e-6

#: The re-optimisation (:func:`reoptimise`): chord-Newton steps from the
#: frozen-path guess until ``|g_all|`` is under this, at most this many.
REOPT_G_TOL = 1e-11
REOPT_MAX_ITER = 200

#: The FCI solver's tolerance where one runs (``casscf_object``; only
#: ``gen_g_hop`` and ``pyscf.grad.casscf`` read the object here).
REOPT_FCI_CONV_TOL = 1e-14

#: A re-optimised state further than this from the reference's CI vector
#: (``1 - |<c0|c(h)>|`` in the gauge-fixed frame) has left its branch and is
#: refused, as ``nevpt2_numgrad.CONTINUITY_CI_TOL`` refuses one (looser:
#: the coarsest rung is 0.125 bohr, not 0.0025).
CONTINUITY_CI_TOL = 1e-2

#: The gauge decomposition :func:`gauge_fixed_parameters` iterates to this
#: (max off-diagonal-block element of ``exp(-K) U``, and max change of
#: ``W``): two decades above ``logm``/``expm``'s round-off floor (~1e-14),
#: which a ladder quotient divides by ``2h`` >= 2^-9 to stay under 1e-9.
GAUGE_TOL = 1e-12
GAUGE_MAX_ITER = 200

#: A CI direction is "outside the sector" when the state's own irrep mask
#: (``nevpt2_grad.allowed_determinants``'s construction) excludes it; the
#: response along a point-group-keeping direction must not leak past this.
SECTOR_LEAK_TOL = 1e-9

#: The cases the script runs (``nevpt2_numgrad.CASES``' with a committed
#: ``_casscf.mogold``).
CASES = ("n2_ccpvdz_cas44", "lif_def2svp_cas46", "acrolein_def2svp_cas65",
         "hsoh_def2svp_cas66")


# --------------------------------------------------------------------------
# The reference, in the golden's frame
# --------------------------------------------------------------------------


@dataclass
class State:
    """The committed CASSCF state in the golden's frame: ``ch.Reference`` plus
    the CI vector, its shape and what the golden calls it."""

    case: str
    ref: ch.Reference
    ci: np.ndarray
    nelecas: tuple
    e_casscf: float
    mol_sym: object  # the symmetry-on molecule, for the sector mask


def state_from_case(case: str) -> State:
    """``generate_golden._build_state``'s ``--mo-from`` state, every gate
    included, in the frame the ``_casscf`` golden is written in: ``pt``'s
    canonicalised orbitals and natural-orbital CI vector (not ``mc``'s
    sidecar orbitals, which ``casscf_hessian.reference_from_case`` uses: the
    Hessian is the same in both, the responses are not)."""
    spec = ngrad.CASES[case]
    mo_from = ngrad.REPO / "golden" / f"{case}_casscf.mogold"
    geom = ngrad.read_xyz(ngrad.REPO / spec["geometry"])
    mc, pt = gg._build_state(geom, spec["basis"], spec["ncas"], spec["nelecas"], None,
                             cas_irreps=spec.get("cas_irreps"),
                             core_irreps=spec.get("core_irreps"), mo_from=mo_from)
    ci = np.asarray(pt.ci).copy()
    D, G = ch.full_rdms(pt.ncore, pt.ncas, pt.nelecas, ci)
    ref = ch.Reference(geom=geom, basis=spec["basis"], C0=np.asarray(pt.mo_coeff).copy(),
                       ncore=pt.ncore, ncas=pt.ncas, D=D, G=G, e_ref=float(mc.e_tot))
    return State(case=case, ref=ref, ci=ci, nelecas=tuple(pt.nelecas),
                 e_casscf=float(mc.e_tot), mol_sym=pt.mol)


def check_golden_frame(st: State) -> list[str]:
    """The state is the one the committed ``_casscf`` golden and gradient
    sidecar hold: their ``ci`` and ``mo_coeff``, bit for bit. Returns the
    disagreements (empty: none)."""
    out = []
    golden = ngrad.REPO / "golden" / f"{st.case}_casscf.nevpt2gold"
    _h, ga = gf.read_golden(golden, want={"ci"})
    if not np.array_equal(ga["ci"].reshape(st.ci.shape), st.ci):
        out.append(f"{golden.name}'s ci is not the state's")
    side = golden.with_suffix(".gradgold")
    if side.exists():
        _s, sa = gf.read_gradgold(side)
        if not np.array_equal(sa["mo_coeff"], st.ref.C0):
            out.append(f"{side.name}'s mo_coeff is not the state's")
    return out


def sector_mask(st: State) -> np.ndarray:
    """The determinants in the state's own irrep (flat boolean), built as
    ``nevpt2_grad.allowed_determinants`` builds it; all of them on a C1
    molecule or when the active orbitals are not symmetry-pure."""
    mol = st.mol_sym
    if not getattr(mol, "symmetry", False) or mol.groupname == "C1":
        return np.ones(st.ci.size, dtype=bool)
    from pyscf import symm
    from pyscf.fci import cistring, direct_spin1_symm

    r = st.ref
    act = slice(r.ncore, r.nocc)
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, r.C0[:, act], check=False)
    irr_a = direct_spin1_symm._gen_strs_irrep(
        cistring.make_strings(range(r.ncas), st.nelecas[0]), orbsym)
    irr_b = direct_spin1_symm._gen_strs_irrep(
        cistring.make_strings(range(r.ncas), st.nelecas[1]), orbsym)
    det = (irr_a[:, None] ^ irr_b[None, :]).ravel()
    c = st.ci.ravel()
    mask = det == int(det[int(np.argmax(np.abs(c)))])
    if (~mask).any() and float(np.abs(c[~mask]).max()) > 1e-9:
        return np.ones(st.ci.size, dtype=bool)
    return mask


# --------------------------------------------------------------------------
# gen_g_hop on a symmetry-off molecule
# --------------------------------------------------------------------------


def casscf_object(mol, ncas: int, nelecas):
    """A plain ``mc1step.CASSCF`` with a plain ``direct_spin1`` FCI solver on
    an RHF object over ``mol`` -- nothing symmetry-adapted, so no step
    variable or determinant is masked by irrep (the coupled space is the
    whole one, by construction)."""
    mf = scf.RHF(mol)
    mf._eri = mol.intor("int2e", aosym="s8")
    mc = mc1step.CASSCF(mf, ncas, nelecas)
    mc.fcisolver = fci.direct_spin1.FCI(mol)
    mc.fcisolver.conv_tol = REOPT_FCI_CONV_TOL
    return mc


def variables(mc, nmo: int) -> np.ndarray:
    """``gen_g_hop``'s packed orbital variables as ``(i, j)``, ``i > j``, in
    packing order."""
    return np.argwhere(mc.uniq_var_indices(nmo, mc.ncore, mc.ncas, mc.frozen))


def g_hop(mc, C: np.ndarray, ci: np.ndarray):
    """``newton_casscf.gen_g_hop`` at ``C`` and the normalised ``ci``:
    ``(g_all, h_op)``."""
    c = np.asarray(ci, dtype=float).ravel()
    c = c / np.linalg.norm(c)
    mc.mo_coeff = C
    g_all, _gu, h_op, _hd = newton_casscf.gen_g_hop(mc, C, c, mc.ao2mo(C))
    return np.asarray(g_all), h_op


def unpack_kappa(var: np.ndarray, x: np.ndarray, nmo: int) -> np.ndarray:
    """``K(x)``: ``K_ij = x``, ``K_ji = -x`` (``unpack_uniq_var``)."""
    k = np.zeros((nmo, nmo))
    k[var[:, 0], var[:, 1]] = x
    return k - k.T


# --------------------------------------------------------------------------
# dE/dTheta at given integrals: what the right-hand sides are made of
# --------------------------------------------------------------------------


def energy_gradient(sk: ch.Skeleton, var: np.ndarray, ci: np.ndarray, nelecas,
                    h: np.ndarray, g_xooo: np.ndarray) -> np.ndarray:
    """``dE/dTheta`` in ``gen_g_hop``'s variables at MO integrals ``h``
    (``(nmo, nmo)``) and ``g_xooo`` (``(nmo, nocc, nocc, nocc)``), the
    reference's densities and CI vector: ``[2 (F - F^T)_(i,j) ; 2 (H - E) c]``.

    Linear in ``(h, g)`` (the CI part's ``E`` is ``<c|H|c>`` at the same
    integrals), so at the reference integrals it is ``gen_g_hop``'s
    ``g_all`` and at the perturbed Hamiltonian ``h^(B)``, ``g^(B)`` it is
    ``dg_all/dR_B`` along the frozen path."""
    r = sk.ref
    nmo, nc, nocc = sk.nmo, r.ncore, r.nocc
    F = np.zeros((nmo, nmo))
    F[:, :nocc] = sk.gen_fock(h, g_xooo)
    g_orb = 2.0 * (F - F.T)[var[:, 0], var[:, 1]]
    a = slice(nc, nocc)
    i = np.arange(nc)
    heff = (h[a, a] + 2.0 * g_xooo[a, a][:, :, i, i].sum(axis=2)
            - g_xooo[a][:, i, i, a.start:a.stop].sum(axis=1))
    eri = g_xooo[a, a, a, a]
    c = np.asarray(ci, dtype=float).reshape(ci.shape)
    h2 = direct_spin1.absorb_h1e(heff, eri, r.ncas, nelecas, 0.5)
    hc = direct_spin1.contract_2e(h2, c, r.ncas, nelecas).ravel()
    cf = c.ravel()
    g_ci = 2.0 * (hc - (cf @ hc) * cf)
    return np.concatenate([g_orb, g_ci])


def mixed_derivatives(sk: ch.Skeleton, var: np.ndarray, ci: np.ndarray,
                      nelecas) -> np.ndarray:
    """``L_{Theta R_B}`` for every Cartesian coordinate ``B``,
    ``(3N, nvar + ndet)``: :func:`energy_gradient` at the perturbed
    Hamiltonian along the frozen path."""
    o, allmo = sk.occ, slice(None)
    t1, _ = sk.connection(second=False)
    h1ao = sk.hcore1()
    out = []
    for ia in range(sk.natm):
        skel = sk.eri1_skel(ia, allmo, o, o, o)
        for x in range(3):
            k = 3 * ia + x
            hk = sk.perturbed_h1(k, t1, h1ao)
            gk = sk.perturbed_eri(k, (allmo, o, o, o), t1, skel[x])
            out.append(energy_gradient(sk, var, ci, nelecas, hk, gk))
    return np.array(out)


# --------------------------------------------------------------------------
# The coupled operator and the responses
# --------------------------------------------------------------------------


def coupled_basis(nvar: int, ci: np.ndarray) -> np.ndarray:
    """``B``, ``(nvar + ndet, nvar + ndet - 1)`` with orthonormal columns: the
    identity on the orbital variables, and on the CI block an orthonormal
    basis of the whole determinant space with ``c`` projected out."""
    c = ci.ravel() / np.linalg.norm(ci)
    n = c.size
    u, s, _vt = np.linalg.svd(np.eye(n) - np.outer(c, c))
    keep = s > 0.5
    if int(keep.sum()) != n - 1:
        raise SystemExit(f"coupled_basis: the CI complement has rank {int(keep.sum())}, "
                         f"expected {n - 1}")
    b = np.zeros((nvar + n, nvar + n - 1))
    b[:nvar, :nvar] = np.eye(nvar)
    b[nvar:, nvar:] = u[:, keep]
    return b


@dataclass
class Responses:
    """The solved system: ``theta`` ``(3N, nvar + ndet)`` in ``gen_g_hop``'s
    full variables, the right-hand sides ``rhs`` (``dg_all/dR``), ``kappa``
    ``(3N, nmo, nmo)``, ``ci`` ``(3N, ndet)``, and the operator's numbers."""

    var: np.ndarray
    rhs: np.ndarray
    theta: np.ndarray
    kappa: np.ndarray
    ci: np.ndarray
    info: dict
    chord: Chord


def solve_responses(sk: ch.Skeleton, st: State, log=print) -> Responses:
    """``J Theta^(1)_B = -dg_all/dR_B`` for all ``3N`` ``B`` at once: ``J``
    built densely on :func:`coupled_basis` through ``gen_g_hop``'s ``h_op``,
    symmetrised, LU-factorised once."""
    r = st.ref
    mc = casscf_object(sk.mol, r.ncas, st.nelecas)
    var = variables(mc, sk.nmo)
    nvar = len(var)
    g_all, h_op = g_hop(mc, r.C0, st.ci)
    mine = energy_gradient(sk, var, st.ci, st.nelecas, sk.h_mo,
                           sk.mo_eri(slice(None), sk.occ, sk.occ, sk.occ))
    info: dict = dict(nvar=nvar, ndet=st.ci.size,
                      g_orb=float(np.linalg.norm(g_all[:nvar])),
                      g_ci=float(np.linalg.norm(g_all[nvar:])),
                      g_all_vs_formula=float(np.abs(g_all - mine).max()))
    t0 = time.perf_counter()
    rhs = mixed_derivatives(sk, var, st.ci, st.nelecas)
    info["rhs_seconds"] = time.perf_counter() - t0
    bas = coupled_basis(nvar, st.ci)
    n = bas.shape[1]
    t0 = time.perf_counter()
    J = np.empty((n, n))
    for j in range(n):
        J[:, j] = bas.T @ h_op(bas[:, j])
    info["build_seconds"] = time.perf_counter() - t0
    info["dimension"] = n
    info["asymmetry"] = float(np.abs(J - J.T).max() / np.abs(J).max())
    Js = 0.5 * (J + J.T)
    eig, vec = np.linalg.eigh(Js)
    info.update(eig_lowest=eig[:5].copy(), eig_min_abs=float(np.abs(eig).min()),
                negative=int(np.count_nonzero(eig < 0.0)),
                cond=float(np.abs(eig).max() / np.abs(eig).min()))
    # The re-optimisations' chord: J's inverse with its negative and
    # near-null modes frozen (CHORD_EIG_CUT).
    up = eig > CHORD_EIG_CUT
    info["chord_frozen"] = int(np.count_nonzero(~up))
    chord_inv = (vec[:, up] / eig[up]) @ vec[:, up].T
    t0 = time.perf_counter()
    lu = scipy.linalg.lu_factor(Js)
    b_red = rhs @ bas  # (3N, n): B^T b for every B
    sol = scipy.linalg.lu_solve(lu, -b_red.T).T
    info["solve_seconds"] = time.perf_counter() - t0
    info["residual"] = float(np.abs(sol @ Js + b_red).max())
    # the same solution through h_op, which never reads the dense matrix
    info["residual_hop"] = float(max(np.abs(bas.T @ h_op(bas @ s) + b).max()
                                     for s, b in zip(sol, b_red, strict=True)))
    theta = sol @ bas.T
    kappa = np.array([unpack_kappa(var, t[:nvar], sk.nmo) for t in theta])
    ci = theta[:, nvar:].copy()
    info["ci_overlap"] = float(np.abs(ci @ st.ci.ravel()).max())
    log(f"  coupled space: {nvar} orbital variables + {n - nvar} CI directions (the "
        f"{st.ci.size} determinants less c) = {n}; built in {info['build_seconds']:.1f} s, "
        f"max |J - J^T| / max |J| = {info['asymmetry']:.1e}")
    log("    lowest eigenvalues: " + " ".join(f"{e:+.6e}" for e in info["eig_lowest"])
        + f"; {info['negative']} negative, cond {info['cond']:.2e}")
    log(f"    |g_all| at the reference: orbital {info['g_orb']:.2e}, CI {info['g_ci']:.2e}; "
        f"max |g_all - formula| = {info['g_all_vs_formula']:.1e}")
    log(f"    3N = {sk.ncoord} right-hand sides in {info['rhs_seconds']:.1f} s, one LU, "
        f"solved in {info['solve_seconds']:.2f} s; max |J Theta + b| = "
        f"{info['residual']:.1e} (dense), {info['residual_hop']:.1e} (through h_op)")
    if info["chord_frozen"]:
        log(f"    {info['chord_frozen']} mode(s) at or under {CHORD_EIG_CUT:g} frozen in the "
            "re-optimisations' chord; max |b . v| / |b| over them = "
            f"{float(np.abs(b_red @ vec[:, ~up]).max() / np.abs(b_red).max()):.1e}")
    return Responses(var=var, rhs=rhs, theta=theta, kappa=kappa, ci=ci, info=info,
                     chord=Chord(inv=chord_inv, basis=bas, var=var))


def assemble(L: np.ndarray, resp: Responses) -> np.ndarray:
    """``d^2E/dR_A dR_B = L_RR + L_{R_A Theta} . Theta^(1)_B``."""
    return L + resp.rhs @ resp.theta.T


def identities(H: np.ndarray, natm: int) -> dict:
    """``H``'s symmetry and the translational sum rule ``sum_B H_AB = 0``
    per Cartesian direction pair, as maxima."""
    H4 = H.reshape(natm, 3, natm, 3)
    return dict(symmetry=float(np.abs(H - H.T).max()),
                translation=float(np.abs(H4.sum(axis=2)).max()))


def rotation_rule(H: np.ndarray, de: np.ndarray, xyz: np.ndarray) -> float:
    """The rotational sum rule, ``H (Omega R) = Omega g`` for each of the
    three generators ``Omega`` (``E(Q R) = E(R)`` differentiated once in
    ``R`` and once in ``Q`` at ``Q = 1``; ``R`` the positions in bohr, ``g``
    the gradient), as the maximum residual. Unlike the translational rule it
    reads the columns along **every** coordinate the rotation moves, so it
    is the one identity that holds the symmetry-breaking responses to
    something; it holds only at a stationary state, to the size of the
    state's residual orbital gradient."""
    g = de.reshape(-1, 3)
    out = 0.0
    for a in range(3):
        om = np.zeros((3, 3))
        om[(a + 1) % 3, (a + 2) % 3], om[(a + 2) % 3, (a + 1) % 3] = -1.0, 1.0
        out = max(out, float(np.abs(H @ (xyz @ om.T).ravel() - (g @ om.T).ravel()).max()))
    return out


# --------------------------------------------------------------------------
# The re-optimised ladders
# --------------------------------------------------------------------------


def _block_slices(ncore: int, nocc: int, nmo: int) -> list[slice]:
    return [slice(0, ncore), slice(ncore, nocc), slice(nocc, nmo)]


def gauge_fixed_parameters(U: np.ndarray, ncore: int, ncas: int, ci: np.ndarray,
                           nelecas, c0: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """A re-optimised state ``(C', c')`` read back in the response's
    parametrisation: ``U = C_f^T S C'`` decomposed as ``exp(K) W`` with ``K``
    zero on the core-core, active-active and virtual-virtual blocks and
    ``W`` block-diagonal orthogonal, by fixed-point iteration from ``W = 1``
    (``K`` = the off-diagonal blocks of ``log(U W^T)``, ``W`` = the polar
    factor of ``exp(-K) U``'s diagonal blocks). The CI vector is carried into
    the orbitals ``C_f exp(K) = C' W^T``: its active block is
    ``fci.addons.transform_ci``'s ``u = W_aa^T`` (new orbitals = old @ u),
    and its sign is fixed by ``<c0|c> > 0``. Returns ``(K, c, iterations)``."""
    nmo = U.shape[0]
    blocks = _block_slices(ncore, ncore + ncas, nmo)
    W = np.eye(nmo)
    n_it = 0
    for _ in range(GAUGE_MAX_ITER):
        n_it += 1
        K = np.real(scipy.linalg.logm(U @ W.T))
        K = 0.5 * (K - K.T)
        for b in blocks:
            K[b, b] = 0.0
        M = scipy.linalg.expm(-K) @ U
        off = M.copy()
        for b in blocks:
            off[b, b] = 0.0
        Wn = np.zeros_like(W)
        for b in blocks:
            if b.stop > b.start:
                u, _s, vt = np.linalg.svd(M[b, b])
                Wn[b, b] = u @ vt
        done = float(np.abs(off).max()) <= GAUGE_TOL and float(np.abs(Wn - W).max()) <= GAUGE_TOL
        W = Wn
        if done:
            break
    else:
        raise SystemExit(f"gauge_fixed_parameters: no convergence in {GAUGE_MAX_ITER} "
                         f"iterations (off-block {float(np.abs(off).max()):.1e})")
    a = slice(ncore, ncore + ncas)
    c = fci.addons.transform_ci(np.asarray(ci), nelecas, W[a, a].T)
    c = np.asarray(c).ravel()
    c = c / np.linalg.norm(c)
    if float(c @ c0.ravel()) < 0.0:
        c = -c
    return K, c, n_it


@dataclass
class Chord:
    """What every re-optimisation steps with (:func:`converge`): the
    reference's coupled operator, inverted on its modes above
    :data:`CHORD_EIG_CUT` (``inv``, on :func:`coupled_basis`'s ``basis``),
    and ``gen_g_hop``'s variables."""

    inv: np.ndarray
    basis: np.ndarray
    var: np.ndarray


def converge(mc, C: np.ndarray, c: np.ndarray, chord: Chord) -> tuple:
    """Chord-Newton steps from ``(C, c)`` on ``mc``'s molecule: ``delta =
    -J0^+ g_all`` with ``J0^+`` the chord's inverse and ``g_all``
    ``gen_g_hop``'s gradient at the current point (``C <- C exp(K(delta))``,
    ``c <- (c + delta_c) / |.|``), until ``|g_all|`` is under
    :data:`REOPT_G_TOL` or :data:`REOPT_MAX_ITER` steps. Returns ``(C, c,
    g_all, |g_all| per step)``.

    Not ``mcscf.newton`` / ``mc1step``: from the frozen-path start on N2
    they stall at ``|g_orb|`` ~6e-7 (``ah_lindep``;
    ``nevpt2_numgrad.finish_casscf`` has the same finding), which a ladder
    quotient divides by ``2h``. The chord converges linearly, at a rate set
    by ``|J(R) - J0|``. Its frozen modes (LiF's symmetry-breaking saddle,
    docs/reference-data.md, "LiF and acrolein on CASSCF orbitals") are never
    stepped along, so a point-group-keeping displacement stays on the
    reference's symmetric branch instead of rolling off the saddle."""
    nmo, nvar = C.shape[1], len(chord.var)
    shape = c.shape
    c = c.ravel() / np.linalg.norm(c)
    norms = []
    for _it in range(REOPT_MAX_ITER):
        g, _h = g_hop(mc, C, c)
        norms.append(float(np.linalg.norm(g)))
        if norms[-1] <= REOPT_G_TOL:
            break
        step = chord.basis @ (chord.inv @ -(chord.basis.T @ g))
        C = C @ scipy.linalg.expm(unpack_kappa(chord.var, step[:nvar], nmo))
        c = c + step[nvar:]
        c = c / np.linalg.norm(c)
    return C, c.reshape(shape), g, norms


def reoptimise(ref: ch.Reference, ci0: np.ndarray, nelecas, geom, S0_half: np.ndarray,
               chord: Chord, *, gradient: bool = True) -> dict:
    """The CASSCF re-optimised at ``geom`` from the frozen-path guess
    ``(C_f, c0)`` (:func:`converge`), read back by
    :func:`gauge_fixed_parameters`, with ``pyscf.grad.casscf``'s analytic
    gradient at the converged state (``gradient``)."""
    mol = ch.mol_at(geom, ref.basis)
    S = mol.intor("int1e_ovlp")
    Cf = ngr.frozen_coeff(ref.C0, S0_half, S)
    mc = casscf_object(mol, ref.ncas, nelecas)
    nvar = len(chord.var)
    C, cis, g, norms = converge(mc, Cf, ci0, chord)
    U = Cf.T @ S @ C
    K, cg, iters = gauge_fixed_parameters(U, ref.ncore, ref.ncas, cis, nelecas, ci0)
    out = dict(converged=norms[-1] <= REOPT_G_TOL, n_iter=len(norms), g_norm=norms[-1],
               kappa=K, ci=cg, gauge_iter=iters, g_orb=float(np.linalg.norm(g[:nvar])),
               g_ci=float(np.linalg.norm(g[nvar:])),
               ci_defect=float(1.0 - abs(cg @ ci0.ravel())))
    if gradient:
        mc.mo_coeff, mc.ci, mc.converged = C, cis, True
        out["grad"] = ngrad.casscf_gradient(mc).ravel()
    return out


def polished(st: State, sk: ch.Skeleton, chord: Chord) -> tuple[State, dict]:
    """The committed state converged to ``|g_all|`` under
    :data:`REOPT_G_TOL` at its own geometry (:func:`converge`, from itself):
    the exactly stationary point the re-optimised ladders pass through at
    ``h = 0``. In memory only. The committed one sits ``O(|g_orb|)`` from
    it, and the responses there differ from these by as much -- which is
    what the ladders measure when they are read against the committed
    state's (docs/reference-data.md, "The analytic CASSCF nuclear
    Hessian")."""
    r = st.ref
    C, c, _g, norms = converge(casscf_object(sk.mol, r.ncas, st.nelecas), r.C0, st.ci, chord)
    if c.ravel() @ st.ci.ravel() < 0.0:
        c = -c
    D, G = ch.full_rdms(r.ncore, r.ncas, st.nelecas, c)
    ref = ch.Reference(geom=r.geom, basis=r.basis, C0=C, ncore=r.ncore, ncas=r.ncas,
                       D=D, G=G, e_ref=None)
    e = ch.path_energy(ref, r.geom, sk.S0_half)
    ref.e_ref = e
    info = dict(n_iter=len(norms), g_start=norms[0], g_end=norms[-1],
                converged=norms[-1] <= REOPT_G_TOL, d_e=e - st.e_casscf,
                rotation=float(np.abs(scipy.linalg.logm(r.C0.T @ sk.S0 @ C)).max()),
                d_ci=float(np.abs(c - st.ci).max()))
    return State(case=st.case, ref=ref, ci=c, nelecas=st.nelecas, e_casscf=e,
                 mol_sym=st.mol_sym), info


#: The chord each worker steps with, set once per process by
#: :func:`_init_worker` rather than pickled with every task.
_WORKER: dict = {}


def _init_worker(chord: Chord) -> None:
    _WORKER["chord"] = chord


def _reopt_task(task: tuple) -> tuple:
    key, ref, ci0, nelecas, S0_half, dvec = task
    return key, reoptimise(ref, ci0, nelecas, ngrad.displace_vec(ref.geom, dvec), S0_half,
                           _WORKER["chord"])


def reoptimised(st: State, S0_half: np.ndarray, chord: Chord, points: dict,
                workers: int) -> dict:
    tasks = [(k, st.ref, st.ci, st.nelecas, S0_half, d) for k, d in points.items()]
    if workers <= 1:
        _init_worker(chord)
        return dict(_reopt_task(t) for t in tasks)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(chord,)) as pool:
        return dict(pool.map(_reopt_task, tasks))


def keeps_point_group(mol, d: np.ndarray, tol: float = 1e-10) -> bool:
    """Whether the displacement ``d`` (``(natm, 3)``, the input frame) is
    **totally symmetric** under every operation of ``mol``'s point group (the
    group ``generate_golden._build_mol`` builds in, in its own frame
    ``mol._symm_orig`` / ``mol._symm_axes``): each operation maps the
    displaced molecule onto itself, so the displaced geometry keeps the
    reference's group *with the same operations* -- what keeps the state in
    its sector. Re-detecting the group at the displaced geometry is not the
    same test: a diatomic stays ``D_inf_h`` under any displacement, in a
    rotated frame."""
    from pyscf import symm

    if not getattr(mol, "symmetry", False):
        return True
    axes = np.asarray(mol._symm_axes)
    Q = (mol.atom_coords() - np.asarray(mol._symm_orig)) @ axes.T
    D = np.asarray(d) @ axes.T
    ops = symm.geom.symm_ops(mol.groupname)
    names = [mol.atom_symbol(i) for i in range(mol.natm)]
    for op_name in symm.param.OPERATOR_TABLE[mol.groupname]:
        rot = np.asarray(ops[op_name], dtype=float)
        if rot.ndim == 0:
            rot = float(rot) * np.eye(3)
        Qo, Do = Q @ rot.T, D @ rot.T
        for i in range(mol.natm):
            j = [j for j in range(mol.natm)
                 if names[j] == names[i] and np.abs(Q[j] - Qo[i]).max() < 1e-6]
            if len(j) != 1:
                raise SystemExit(f"keeps_point_group: operation {op_name} maps atom {i} "
                                 "onto no atom: the frame is not the group's")
            if np.abs(D[j[0]] - Do[i]).max() > tol:
                return False
    return True


def kept_directions(st: State) -> dict:
    """The point-group-keeping unit directions, by name
    (:func:`keeps_point_group`): each Cartesian coordinate ``e<k>`` that is
    one, and on a diatomic the stretch (N2's only one)."""
    r = st.ref
    natm = len(r.geom)
    cands = {f"e{k}": ch.unit(natm, k) for k in range(3 * natm)}
    if natm == 2:
        xyz = st.mol_sym.atom_coords()
        a = (xyz[1] - xyz[0]) / np.linalg.norm(xyz[1] - xyz[0])
        cands["stretch"] = np.array([-a, a]) / np.sqrt(2.0)
    return {name: d for name, d in cands.items() if keeps_point_group(st.mol_sym, d)}


def run_ladders(st: State, sk: ch.Skeleton, H: np.ndarray, resp: Responses,
                H_committed: np.ndarray, workers: int, log=print) -> tuple[list, dict]:
    """Along every :func:`kept_directions` direction, re-optimising around
    the **polished** state ``st`` (:func:`polished`; its frozen path, its
    gauge): ``H d`` -- the polished state's and the committed one's -- against
    the ladder of ``pyscf.grad.casscf`` at the re-optimised states, and the
    polished state's response along ``d`` -- ``kappa`` and CI -- against the
    ladder of the re-optimised parameters. Returns the checks and the
    re-optimisations' numbers."""
    steps = ngr.ladder_steps(LADDER)
    dirs = kept_directions(st)
    log(f"  point-group-keeping directions ({len(dirs)}): {', '.join(dirs)}")
    points = {(name, s, h): s * h * d for name, d in dirs.items() for h in steps
              for s in (1, -1)}
    t0 = time.perf_counter()
    got = reoptimised(st, sk.S0_half, resp.chord, points, workers)
    wall = time.perf_counter() - t0
    stats = dict(n=len(got), wall=wall,
                 g_orb=max(v["g_orb"] for v in got.values()),
                 g_ci=max(v["g_ci"] for v in got.values()),
                 ci_defect=max(v["ci_defect"] for v in got.values()),
                 unconverged=sum(not v["converged"] for v in got.values()),
                 n_iter=max(v["n_iter"] for v in got.values()),
                 gauge_iter=max(v["gauge_iter"] for v in got.values()))
    log(f"  {len(got)} re-optimised states in {wall:.1f} s: chord iterations <= "
        f"{stats['n_iter']}, max |g_orb| {stats['g_orb']:.1e}, max |g_ci| "
        f"{stats['g_ci']:.1e}, max 1-|<c0|c>| {stats['ci_defect']:.1e}, gauge iterations <= "
        f"{stats['gauge_iter']}")
    checks = [ngr.Check("re-optimisation", stats["unconverged"] == 0,
                        f"{stats['unconverged']} of {len(got)} states not under |g_all| "
                        f"{REOPT_G_TOL:g} in {REOPT_MAX_ITER} chord steps")]
    bad = [k for k, v in got.items() if v["ci_defect"] > CONTINUITY_CI_TOL]
    if bad:
        checks.append(ngr.Check("continuity", False, f"{len(bad)} re-optimised state(s) "
                                f"left the branch (1-|<c0|c>| > {CONTINUITY_CI_TOL:g}): "
                                f"{bad[:4]}"))
    mask = sector_mask(st)
    for name, d in dirs.items():
        dv = d.ravel()
        fd_g = np.array([(got[(name, 1, h)]["grad"] - got[(name, -1, h)]["grad"]) / (2 * h)
                         for h in steps])
        checks.append(ch.judge_column(f"H d [{name}] vs pyscf.grad.casscf", fd_g, steps,
                                      H @ dv))
        checks.append(ch.judge_column(f"H_committed d [{name}] vs pyscf.grad.casscf", fd_g,
                                      steps, H_committed @ dv))
        fd_k = np.array([((got[(name, 1, h)]["kappa"] - got[(name, -1, h)]["kappa"])
                          / (2 * h)).ravel() for h in steps])
        checks.append(ch.judge_column(f"kappa [{name}] vs re-optimised", fd_k, steps,
                                      np.tensordot(dv, resp.kappa, axes=1).ravel()))
        fd_c = np.array([(got[(name, 1, h)]["ci"] - got[(name, -1, h)]["ci"]) / (2 * h)
                         for h in steps])
        cref = dv @ resp.ci
        checks.append(ch.judge_column(f"ci [{name}] vs re-optimised", fd_c, steps, cref))
        leak = float(np.abs(cref[~mask]).max()) if (~mask).any() else 0.0
        checks.append(ngr.Check(f"ci [{name}] sector leak", leak <= SECTOR_LEAK_TOL,
                                f"max |response outside the state's irrep| = {leak:.1e} "
                                f"(allowed {SECTOR_LEAK_TOL:g})", numbers=dict(err=leak)))
    return checks, stats


# --------------------------------------------------------------------------
# The sidecar
# --------------------------------------------------------------------------


def sidecar(st: State, sk: ch.Skeleton, de: np.ndarray, L: np.ndarray, H: np.ndarray,
            resp: Responses, kept: np.ndarray) -> tuple[dict, dict]:
    """The ``.respgold`` header and arrays (:func:`golden_format.write_respgold`)."""
    r = st.ref
    header = dict(natm=sk.natm, nao=sk.nao, nmo=sk.nmo, ncore=r.ncore, nact=r.ncas,
                  nvirt=sk.nmo - r.nocc, nelec_a=int(st.nelecas[0]),
                  nelec_b=int(st.nelecas[1]), ndet=int(st.ci.size), e_casscf=st.e_casscf)
    arrays = {
        "mo_coeff": r.C0, "ci": st.ci,
        "kappa": resp.kappa, "ci_response": resp.ci,
        "hessian": H, "l_rr": L, "gradient": de.ravel(),
        "kept": kept.astype(np.float64),
    }
    return header, arrays


def kept_mask(st: State) -> np.ndarray:
    """``1`` on each Cartesian coordinate that keeps the point group
    (:func:`keeps_point_group`), ``0`` elsewhere."""
    dirs = kept_directions(st)
    return np.array([1.0 if f"e{k}" in dirs else 0.0 for k in range(3 * len(st.ref.geom))])


def analyse(st: State, sk: ch.Skeleton, log=print) -> tuple:
    """Gradient, ``L_RR``, the responses and the assembled Hessian of one
    state, with every exact identity checked and printed. Returns ``(de, L,
    resp, H, ok)``."""
    t0 = time.time()
    de = sk.gradient()
    L, _terms = sk.hessian()
    log(f"  gradient and L_RR in {time.time() - t0:.1f} s")
    resp = solve_responses(sk, st, log=log)
    H = assemble(L, resp)
    ids = identities(H, sk.natm)
    ids.update((k, resp.info[k]) for k in ("residual", "residual_hop", "g_all_vs_formula",
                                           "ci_overlap"))
    ok = True
    for name, v in ids.items():
        log(f"  {name:<18} {v:.2e}  (allowed {IDENTITY_TOL:g})")
        ok &= v <= IDENTITY_TOL
    rot = rotation_rule(H, de, sk.mol.atom_coords())
    log(f"  {'rotation':<18} {rot:.2e}  (allowed {ROTATION_TOL:g})")
    ok &= rot <= ROTATION_TOL
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    log("  Hessian eigenvalues (Eh/bohr^2): " + " ".join(f"{e:+.4e}" for e in ev))
    return de, L, resp, H, ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, choices=CASES)
    ap.add_argument("--workers", type=int, default=1,
                    help="processes for the re-optimised ladders")
    ap.add_argument("--no-ladders", action="store_true",
                    help="skip the re-optimised ladders (the identities still run)")
    ap.add_argument("--verbose", action="store_true",
                    help="print every ladder's table, not only the failing ones'")
    ap.add_argument("--write", type=Path,
                    help="write the response sidecar (.respgold) here")
    args = ap.parse_args(argv)

    t0 = time.time()
    st = state_from_case(args.case)
    sk = ch.Skeleton(st.ref)
    print(f"== {args.case} _casscf: {sk.natm} atoms, nao {sk.nao}, ncore {st.ref.ncore}, "
          f"ncas {st.ref.ncas}, ndet {st.ci.size}  (state built in {time.time() - t0:.1f} s)")
    frame = check_golden_frame(st)
    for f in frame:
        print(f"  FAIL frame: {f}")
    ok = not frame
    print(f"  the golden's frame: {'its ci and mo_coeff, bit for bit' if not frame else 'NO'}")
    d_e0 = ch.check_reference_energy(sk)
    print(f"  |E(R0) - E_CASSCF| = {d_e0:.2e}")
    ok &= d_e0 <= 1e-9
    print("-- the committed state (what --write writes)")
    de, L, resp, H, good = analyse(st, sk)
    ok &= good
    if args.write is not None and not ok:
        print(f"  NOT writing {args.write}: the state is not the golden's, or an identity "
              "failed (above)")
    elif args.write is not None:
        header, arrays = sidecar(st, sk, de, L, H, resp, kept_mask(st))
        gf.write_respgold(args.write, header=header, arrays=arrays)
        print(f"  wrote {args.write}")
    if args.no_ladders:
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1

    print(f"-- the polished state: converged in place to |g_all| <= {REOPT_G_TOL:g}")
    stp, pinfo = polished(st, sk, resp.chord)
    print(f"  {pinfo['n_iter']} chord steps, |g_all| {pinfo['g_start']:.2e} -> "
          f"{pinfo['g_end']:.2e}; E - E_committed = {pinfo['d_e']:+.2e}, max |log(C0^T S "
          f"C)| = {pinfo['rotation']:.2e}, max |dc| = {pinfo['d_ci']:.2e}")
    ok &= pinfo["converged"]
    skp = ch.Skeleton(stp.ref)
    _dep, _Lp, respp, Hp, good = analyse(stp, skp)
    ok &= good
    kept = kept_mask(st) > 0
    dk, dc = np.abs(resp.kappa - respp.kappa), np.abs(resp.ci - respp.ci)
    print(f"  committed - polished (the committed state's non-stationarity, "
          f"informational): max |dH| {np.abs(H - Hp).max():.2e} Eh/bohr^2; max |dkappa| "
          f"{dk.max():.2e}, max |dc| {dc.max():.2e} per bohr over all 3N coordinates, "
          f"{dk[kept].max() if kept.any() else 0.0:.2e} and "
          f"{dc[kept].max() if kept.any() else 0.0:.2e} over the {int(kept.sum())} "
          "point-group-keeping ones")
    if not args.no_ladders:
        print(f"  ladder (bohr): 2^-{LADDER[0]}..2^-{LADDER[1]}, gate {LADDER_TOL:g}")
        checks, _stats = run_ladders(stp, skp, Hp, respp, H, args.workers)
        worst: dict = {}
        for c in checks:
            kind = c.name.split(" ")[0] + (" leak" if "leak" in c.name else "")
            worst[kind] = max(worst.get(kind, 0.0), c.numbers.get("err", 0.0))
            ok &= c.ok
            if args.verbose and c.ok:
                print(f"  ok {c.name}: {c.summary}")
                print("\n".join(c.table))
            if not c.ok:
                print(f"  FAIL {c.name}: {c.summary}")
                print("\n".join(c.table))
        print(f"  {sum(c.ok for c in checks)}/{len(checks)} ladder checks pass; worst "
              "|Richardson - analytic|:")
        for kind, v in worst.items():
            print(f"    {kind:<14} {v:.2e}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
