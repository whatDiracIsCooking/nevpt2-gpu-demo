#!/usr/bin/env python3
r"""Build a golden SC-NEVPT2 CASSCF-input reference file, via PySCF.

This is the **only** file in this demo that defines the golden state
(``block2_pc_probe.py`` imports :func:`_build_state`). It runs a plain
``mcscf.CASCI`` diagonalisation on converged RHF orbitals -- bit-identical
run to run given the thread and Davidson settings below, and deterministic
by construction, with no orbital optimiser in the chain (see
:func:`_build_state`, and docs/reference-data.md, "What 'CASSCF input' means
here, and why the golden state is CASCI", for what a fresh CASSCF measurably
does instead; ``--casscf`` / ``--mo-from``, below, is how to get a
CASSCF-orbital golden without putting an optimiser in this chain either) --
canonicalises it exactly as
``pyscf.mrpt.NEVPT.kernel``
does, and writes everything a CASSCF/CASCI package would hand a downstream
NEVPT2 driver into one flat binary file (see :func:`write_golden`'s
docstring for the exact layout -- ``src/golden/golden.cppm`` is its C++ reader):

* the CI vector,
* the 1- and 2-particle density matrices (cheap; PySCF's ``fci.rdm.make_dm123``
  convention),
* every MO-integral block the eight SC-NEVPT2 classes read
  (:func:`_integral_blocks`), and
* PySCF's own per-class SC-NEVPT2 norms/energies, as the golden answer
  ``apps/integral_direct/main.cppm`` checks its GPU-built result against,
* and (unless ``--no-pc``) block2's per-class **partially-contracted**
  NEVPT2 energies, ``pc_class_energies`` (eight, :data:`CLASSES` order) and
  ``e_pc_total`` (one element), from ``pyblock2.icmr.icnevpt2_full`` on the
  same state (:func:`_block2_pc`). PySCF has no PC-NEVPT2, so these are
  block2's answer and **nothing else checks them**; what is checked is that
  block2 sees this state (its SC equals PySCF's per class, PC Sijrs equals
  SC Sijrs) and that the PC metric is gapped enough for the number to be
  well-defined (:func:`_check_pc_conditioning`, which refuses N2
  CAS(12,12) -- hence ``--no-pc`` for those two files, and CAS(10,12) as the
  12-orbital PC case). The header's ``e_nevpt2_total`` stays the SC total.

``--df`` writes the **density-fitted** variant instead (``*_df.nevpt2gold``,
the input ``apps/density_fit/main.cppm`` checks itself against). The whole
calculation is then density-fitted -- RHF, CASCI, canonicalisation, ``h1eff``
and every NEVPT2 integral block -- with one auxiliary basis (``--auxbasis``,
default ``<basis>-jkfit``, which is what PySCF's own ``density_fit()`` picks
for the cc-pVXZ family). The golden answer is then PySCF's DF-NEVPT2
(``pyscf.mrpt.dfnevpt2``), NOT the conventional one: the two differ by the
fitting error (~1e-5 Eh), so comparing a DF result against a conventional
golden at 1e-7 could not tell a working DF path from a broken one. On top of
the arrays above (the same names, rebuilt from the fitted integrals, so the
unmodified ``nevpt2_demo`` runs on a DF file too) a DF file carries the four
MO three-index blocks SC-NEVPT2 needs -- ``B_aa`` ``(L|tu)``, ``B_ca``
``(L|it)``, ``B_va`` ``(L|at)``, ``B_cv`` ``(L|ia)`` -- in the
symmetric-metric convention, ``(pq|rs) = sum_L B[L,p,q] B[L,r,s]``. No
virtual-virtual block is needed by any SC-NEVPT2 class.

What it deliberately does **not** write: ``dm3``/``f3ac``/``f3ca``. Building
those -- the ``n_act^6`` 3-RDM and the two integral-contracted 4-RDM digests --
is the expensive, exponential-in-active-space step the demo puts on the
GPU; producing that from the CI vector here in PySCF would defeat the point
of the demo.

Usage::

    uv run python reference_data/generate_golden.py

regenerates the committed ``golden/n2_ccpvdz_cas1010.nevpt2gold`` (the default
case, N2/cc-pVDZ CAS(10,10)). Pass ``--atom``/``--basis``/``--ncas``/``--nelecas``/``--out`` for a
different active space -- e.g. a fast sanity case::

    uv run python reference_data/generate_golden.py \\
        --atom "N 0 0 0; N 0 0 1.1" --basis sto-3g --ncas 4 --nelecas 4 \\
        --name n2_sto3g_cas44 --out /tmp/n2_sto3g_cas44.nevpt2gold

the committed 12-orbital files (``--nelecas`` before ``--ncas`` in the
name: CAS(electrons, orbitals))::

    uv run python reference_data/generate_golden.py --ncas 12 --nelecas 10 \\
        --name n2_ccpvdz_cas1012                    # SC + PC, ~1 h (4-RDM)
    uv run python reference_data/generate_golden.py --ncas 12 --nelecas 12 --no-pc \\
        --name n2_ccpvdz_cas1212

and the committed density-fitted files::

    uv run python reference_data/generate_golden.py --df          # n2_ccpvdz_cas1010_df
    uv run python reference_data/generate_golden.py --df --ncas 12 --nelecas 12 --no-pc \\
        --name n2_ccpvdz_cas1212_df
    uv run python reference_data/generate_golden.py --df --basis cc-pvtz \\
        --name n2_ccpvtz_cas1010_df

``--gradgold`` additionally writes the **gradient sidecar**
``<out>.gradgold`` beside the golden -- a separate file, leaving the golden's
bytes untouched, carrying the MO-integral inventory an analytic NEVPT2
gradient reads plus the full-MO-range Fock matrices and ``mo_coeff``
(:func:`write_gradgold`, :func:`_gradient_arrays`;
docs/gradient-theory.md, section 3). The committed sidecars are the two small
cases::

    uv run python reference_data/generate_golden.py --ncas 4 --nelecas 4 \\
        --name n2_ccpvdz_cas44 --gradgold
    uv run python reference_data/generate_golden.py --ncas 8 --nelecas 8 \\
        --name n2_ccpvdz_cas88 --gradgold

With ``--df`` the sidecar is the **density-fitted** flavour (magic
``NEVPT2F1``, :func:`_df_gradient_arrays`; docs/gradient-theory.md, section
3.8): the two three-index blocks the ``_df`` golden lacks (``B_cc``,
``B_vv``), the fitted calculation's own full-range matrices, and the four tied
orbital-energy arrays, committed for the same two small cases::

    uv run python reference_data/generate_golden.py --df --ncas 4 --nelecas 4 \\
        --name n2_ccpvdz_cas44_df --gradgold
    uv run python reference_data/generate_golden.py --df --ncas 8 --nelecas 8 \\
        --name n2_ccpvdz_cas88_df --gradgold

and the salicylaldimine CAS(8,8)/6-31G** pair (gitignored, 145/166 MB; the
pi space picked by irrep, the DF one fitted with 6-31G**-RIFIT)::

    uv run python reference_data/generate_golden.py \\
        --atom reference_data/geometries/salicylaldimine.xyz --basis '6-31g**' \\
        --ncas 8 --nelecas 8 --cas-irreps 'A"=8' --core-irreps "A'=27,A\"=1" \\
        --name salicylaldimine_631gss_cas88       # add --df --auxbasis
        # reference_data/basis/6-31gss-rifit.nw and a _df name for the twin

``--casscf --mo-out FILE`` / ``--mo-from FILE`` build a golden on **converged
CASSCF orbitals** in two steps, so that no golden's state depends on an
orbital optimiser. The first step optimises once and commits the orbitals
(:func:`optimise_mo`, :func:`write_mo_sidecar`); the second runs the same
deterministic CASCI as every other golden, on those orbitals
(:func:`_build_state`), and records ``reference_kind="casscf"``. Pass the
*same* case flags to both -- they are the sidecar's fingerprint, and a
mismatch is refused::

    uv run python reference_data/generate_golden.py --ncas 4 --nelecas 4 \\
        --name n2_ccpvdz_cas44_casscf --casscf \\
        --mo-out golden/n2_ccpvdz_cas44_casscf.mogold
    uv run python reference_data/generate_golden.py --ncas 4 --nelecas 4 \\
        --name n2_ccpvdz_cas44_casscf \\
        --mo-from golden/n2_ccpvdz_cas44_casscf.mogold
        # and likewise for the other two committed pairs, --ncas 10
        # --nelecas 10 --name n2_ccpvdz_cas1010_casscf, and --df --ncas 4
        # --nelecas 4 --name n2_ccpvdz_cas44_casscf_df (docs/reference-data.md,
        # "Regenerating the golden file" has both commands of each in full)

Three such pairs are committed -- N2/cc-pVDZ CAS(4,4), CAS(10,10) and a DF
CAS(4,4) -- and each golden regenerates byte-identically from its committed
sidecar. The second step is bit-reproducible like every other golden, and
:func:`_check_casscf_stationary` refuses orbitals that are not a converged
stationary point of that state (RHF orbitals handed in as a sidecar, a loosely
converged optimisation) just as :func:`_check_degeneracy` refuses a state whose
answer would depend on noise.

``--gradgold`` is permitted with ``--mo-from`` (and refused with ``--casscf``,
which writes no golden to sit beside): it writes the same 24 arrays -- the
one-general-index integrals, ``h``, ``heff``, ``f``, ``f_h``, ``fock_rhf`` and
``mo_coeff`` -- evaluated on the committed CASSCF orbitals, with no optimiser
in this step, so a second run reproduces it byte for byte::

    uv run python reference_data/generate_golden.py --ncas 4 --nelecas 4 \\
        --name n2_ccpvdz_cas44_casscf \\
        --mo-from golden/n2_ccpvdz_cas44_casscf.mogold --gradgold

Its provenance contract is weaker than a CASCI sidecar's: an orbital optimiser
sits upstream of it in the regeneration chain (the ``.mogold``), so it is
reproducible from scratch only to a tolerance, not byte for byte, and
``devtools/golden-repro.sh`` makes no claim over it (docs/reference-data.md,
"The CASSCF-reference gradient sidecar"). Of the three invariants
:func:`_check_gradient_invariants` asserts, only (O2) is asserted on it; (O1)
and (O3) state that the active space is a set of RHF canonical orbitals, which
CASSCF orbitals are not. Permitting the integral arrays is not permission to
difference committed orbitals at a displaced geometry: ``nevpt2_numgrad.py
--reference casscf`` re-optimises every displaced state instead.
"""

from __future__ import annotations

import os

#: Single-threaded, forced (not setdefault: an inherited thread count would
#: silently bring the nondeterminism back). Threaded, PySCF's OpenMP/BLAS
#: reductions differ at ~1e-14 run to run, and that was enough to move a
#: default-tolerance CASCI by 7e-10 and E_corr by 7e-8 on BOTH paths. Must
#: precede the numpy/PySCF imports: the thread pools read it once.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import golden_format as gf  # noqa: E402
import numpy as np  # noqa: E402

#: The CASCI Davidson tolerance. PySCF's default (1e-8) stops at a residual of
#: ~1e-4 (the residual threshold is sqrt(conv_tol)), and E_corr feeds on the CI
#: vector linearly: at CAS(10,10) the default-tolerance E_corr sat ~1e-7 from
#: the converged one. Tighter buys nothing measurable: the FCI solver's
#: ``lindep`` (1e-12) drops any correction with |r|^2 below it, so the residual
#: floors at ~1e-6 whatever conv_tol says (that is why 1e-14 "runs out of
#: cycles"). With point-group symmetry on (:func:`_build_mol`), every committed
#: case at 1e-12 agrees
#: with a residual-1e-10 run (``lindep`` lowered) to <=1.4e-9 in E_corr
#: (docs/reference-data.md, "Reproducibility").
FCI_CONV_TOL = 1e-12

#: Abelian subgroup used for each linear point group. Symmetry is NOT
#: optional: N2's canonical virtuals come in exactly degenerate
#: pairs (pi, delta), and the Srs class (``-N/(diff + h/N)`` summed per
#: virtual pair) is not invariant to a rotation inside such a pair -- rotating
#: them on one fixed CI vector moved Srs by 3e-6 while Sr stayed put to 1e-15.
#: Without symmetry that rotation is picked by numerical noise, and CAS(12,12)
#: E_corr scattered over ~2e-5 even between fully converged CI vectors.
#: :func:`_build_mol` auto-detects the group; PySCF already drops a
#: non-abelian group to an abelian subgroup (Td -> D2) but keeps the linear
#: ones, whose cylindrical-symmetry FCI refuses an active space holding half
#: of a pi pair (N2 CAS(10,10)) -- hence this map.
LINEAR_SUBGROUP = {"Dooh": "D2h", "Coov": "C2v"}

#: Two canonical core or virtual orbitals in the same irrep closer than this
#: (Eh) make the pair classes ill-defined again: an abelian subgroup does not
#: split every degenerate set (Td's E becomes A + A in D2), and a gap only
#: pins the rotation between them to ~(Fock noise)/gap radians.
#: :func:`_check_degeneracy` refuses such a state rather than write a golden
#: whose E_corr depends on noise. The numbers behind 1e-4, all measured on N2:
#: Fock noise ~1.4e-9 (how far symmetry-degenerate partners split), Srs
#: sensitivity ~2.3e-5 Eh/rad (a measured rotation test), so a 1e-4 gap costs
#: ~3e-10 Eh; the smallest real same-irrep gap in the committed goldens is
#: 2.5e-2 (cc-pVTZ). A 1.5e-6 gap (CH4 CAS(4,4), whose active space cuts a
#: t2 set) is refused: ~2e-8 Eh, too close to the 1e-7 check.
DEGENERACY_TOL = 1e-4

#: The eight class labels, fixed order -- the order
#: ``pyscf.mrpt.nevpt2.NEVPT.kernel`` evaluates them in. The
#: C++ reader (``src/energy/energy.cpp``) hardcodes this same order rather than
#: reading class names out of the binary file, so the two must stay in sync.
CLASSES = ("Sr", "Si", "Sijrs", "Sijr", "Srsi", "Srs", "Sij", "Sir")

#: block2's class keys (``ic.sub_eners``, both kernels) -> :data:`CLASSES`.
#: The PC ``sub_spaces`` (``rabcpqg*``, ``ijrs+``, ...) fold onto the same
#: eight keys inside block2.
B2_TO_CLASS = {
    "r": "Sr", "i": "Si", "ijrs": "Sijrs", "ijr": "Sijr",
    "rsi": "Srsi", "rs": "Srs", "ij": "Sij", "ir": "Sir",
}

#: block2 SC must reproduce PySCF SC to this, class by class, before any PC
#: number from it is written: anything else means block2 saw a different
#: state, integral set or RDM, and its PC answer would be for that instead.
#: Measured agreement is <=4.4e-10 on every committed state.
#: The same tolerance gates PC Sijrs == SC Sijrs (no active index, so the
#: two contractions are one function).
BLOCK2_SC_TOL = 1e-7

#: The PC conditioning guard (:func:`_check_pc_conditioning`). block2 solves
#: each external tuple's ``H_D c = b`` by ``lstsq(rcond=None)`` -- a
#: pseudo-inverse dropping singular values at or below ``eps * d * s_max`` --
#: and the PC energy is a well-defined number only if that cut sits in a gap
#: of the spectrum: then every cut inside the gap (block2's, or the one a GPU
#: solve picks) drops the same modes and gives the same energy. The guard is
#: the gap ratio, smallest kept / largest dropped singular value (or / the
#: cut, if nothing is dropped), minimised over every tuple of every PC block,
#: and must be at least this. A fixed raised-cut test does not work: on
#: cc-pVTZ CAS(10,10) real modes start at 2.5e-11, so any fixed probe cut
#: near there drops physics, not noise. Measured: N2 cc-pVDZ
#: CAS(10,10) ~1e6 (null modes <=1e-16, real >=1.8e-10), cc-pVTZ CAS(10,10)
#: ~1e4 (2.2e-15 / 2.5e-11), cc-pVDZ CAS(12,12) ~1.2 (3.8e-13 / 4.5e-13: a
#: continuum, one 1s orbital in the active space, PC Sr moving ~2e-7 between
#: defensible cuts -- more than the 1e-7 the GPU is checked to). 1e3 leaves a
#: solve three decades to place its own cut. A state below it is refused, as
#: :func:`_check_degeneracy` refuses a noise-dependent SC one.
PC_MIN_GAP = 1e3

#: The order ``icnevpt2_full.kernel`` calls ``_linear_solve`` in: every PC
#: ``sub_spaces`` key except the ``...2`` halves, which are solved jointly
#: with their ``...1`` partner. ``rabcpqg*`` is Sr, ``iabcpqg*`` Si,
#: ``irabpq1`` Sir; the ``+``/``-`` pairs are the other classes' two spin
#: couplings.
PC_SOLVE_ORDER = (
    "ijrs+", "ijrs-", "rsiap+", "rsiap-", "ijrap+", "ijrap-", "rsabpq+",
    "rsabpq-", "ijabpq+", "ijabpq-", "irabpq1", "rabcpqg*", "iabcpqg*",
)

#: The format constants, re-exported from :mod:`golden_format` (the one
#: module that reads and packs the three formats here, and the only place a
#: header layout is written down): ``G2`` is the current golden layout, ending
#: its scalar header with the ``reference_kind`` field (:data:`REFERENCE_KINDS`);
#: ``G1`` is the layout before it, which every committed **CASCI** golden is
#: written in and which means ``"casci"`` (docs/reference-data.md, "The file
#: format"). Accepting it is what keeps the committed files from having to be
#: rewritten for a one-field header change. ``src/golden/golden.cppm``'s
#: ``ReferenceKind`` is the same two codes, and every derivative in this tree
#: is derived for the CASCI-on-RHF Lagrangian (docs/gradient-theory.md, §1.1
#: and §1.2), so the gradient paths refuse a ``"casscf"`` golden.
GOLDEN_MAGIC = gf.GOLDEN_MAGIC
GOLDEN_MAGIC_V1 = gf.GOLDEN_MAGIC_V1
GOLDEN_MAGICS = gf.GOLDEN_MAGICS
REFERENCE_KINDS = gf.REFERENCE_KINDS
REFERENCE_KIND_CODES = gf.REFERENCE_KIND_CODES

#: Magic bytes identifying a gradient **sidecar** (``*.gradgold``) -- a
#: SEPARATE file from the golden it sits beside, carrying only what an analytic
#: NEVPT2 gradient needs on top of it (:func:`write_gradgold`'s docstring has
#: the layout; ``src/golden/golden.cppm`` reads it). It is a sidecar and not
#: extra arrays in the golden on purpose: the golden format is extensible
#: (magic + array count), but adding arrays would change every committed
#: golden's bytes, and a regenerated golden differing from the committed one is
#: this project's signal that something changed (docs/reference-data.md,
#: "Regenerating the golden file").
GRADGOLD_MAGIC = gf.GRADGOLD_MAGIC

#: How far from zero the RHF Fock's core-active, core-virtual and
#: active-virtual blocks may be in the canonicalised MO basis before
#: :func:`_check_gradient_invariants` refuses to write a sidecar. This is
#: invariant (O1) of docs/gradient-theory.md, §3.6 -- the single statement
#: that the active space is a set of RHF canonical orbitals -- and it holds
#: only up to RHF convergence -- ``mf.get_fock()`` is built from the converged
#: density, while ``mf.mo_coeff`` diagonalises the previous iteration's Fock --
#: so the bound is the SCF's, not machine epsilon, and it is NOT tightened by
#: raising ``mf.conv_tol``, which would change the golden state. Two decades
#: above the measured residual, and still four decades below what a wrongly
#: selected active space would show. The measured residuals are in
#: docs/reference-data.md, "The gradient sidecar".
O1_FOCK_TOL = 1e-5

#: Likewise for invariant (O2): the generalized Fock's core-core and
#: virtual-virtual off-diagonals, which ``canonicalize`` zeroes by
#: diagonalising f inside each space, and its diagonals, which must reproduce
#: the golden's ``e_core`` / ``e_virt``. A diagonalisation is exact to
#: ~machine precision on these sizes, so this one is tight.
O2_FOCK_TOL = 1e-10

#: Magic bytes identifying an **MO sidecar** (``*.mogold``) -- the converged
#: CASSCF orbitals ``--casscf --mo-out`` writes and ``--mo-from`` reads back.
#: A third file in the ``.gradgold`` idiom (own magic, own scalar header, the
#: same named-array section), and a sidecar for the same reason: it leaves
#: every golden's bytes alone (:func:`write_mo_sidecar` has the layout).
#: Committing the orbitals, rather than re-optimising them, is about
#: **provenance and independence from the optimiser** -- the golden then
#: depends on a file plus the same deterministic ``mcscf.CASCI`` the RHF path
#: runs, and no CASSCF run at all. It is NOT a workaround for an irreproducible
#: optimiser: a fresh CASSCF under :data:`CASSCF_CONV_TOL` /
#: :data:`CASSCF_CONV_TOL_GRAD` reproduces bit for bit, measured
#: (docs/reference-data.md, "A fresh CASSCF reproduces, measured").
MO_SIDECAR_MAGIC = gf.MO_SIDECAR_MAGIC

#: ``--casscf``'s optimiser settings, all three far past PySCF's defaults
#: (1e-7 energy, ``sqrt(conv_tol)`` ~ 3.2e-4 gradient, 50 macro iterations).
#: Measured (docs/reference-data.md, "The ``--casscf`` / ``--mo-from`` gate
#: thresholds"): PySCF's own defaults stop at ||g_orb|| 2.8e-5, whose E_corr
#: is 3.6e-6 from the converged one -- far outside the 1e-7 the demos check
#: to, and refused by :data:`CASSCF_GRAD_TOL`. ``conv_tol_grad`` 1e-7
#: converges at CAS(10,10) in 16 macro iterations and reaches ||g_orb||
#: 2.5e-8, 40x inside the gate, with a bit-identical E_corr. 1e-8 does **not**
#: converge there -- it runs to a 200-macro ceiling while producing that same
#: answer -- so it is not what ``mc.converged`` is asserted against.
CASSCF_CONV_TOL = 1e-12
CASSCF_CONV_TOL_GRAD = 1e-7
CASSCF_MAX_CYCLE_MACRO = 200

#: The polish :func:`optimise_mo` runs **only when the optimisation above has
#: not converged**: a second ``mc1step`` run, warm-started from the first run's
#: orbitals and CI vector, with the augmented-Hessian solver's ``ah_conv_tol``
#: lowered from PySCF's 1e-12 to this. Nothing else changes, and no guard is
#: loosened. Why it is needed (measured, docs/reference-data.md, "The CASSCF
#: polish"): ``davidson_cc`` accepts an AH step once its residual is under
#: ``sqrt(ah_conv_tol)`` = 1e-6, which is no longer a small residual once
#: ``||g_orb||`` itself is ~1e-7. LiF CAS(4,6) and acrolein CAS(6,5) stall there
#: for all 200 macro iterations (``||g_orb||`` 1.2e-7 / 2.2e-7, just over
#: ``conv_tol_grad``); with this the polish converges both in 2 macro iterations
#: to the same stationary point. It is a fallback rather than the setting
#: because applying it from the start moves N2 CAS(10,10)'s committed
#: ``.mogold`` (a different, equally converged point inside the 1e-7
#: tolerance), and a regenerated file that differs is this tree's signal that
#: something changed. Every committed N2 sidecar converges without it, so none
#: of them runs the polish. 1e-15, 1e-16 and 1e-18 polish both cases to the
#: same E_CASSCF (12 decimals) and ``||g_orb||`` (4 digits); 1e-16 is inside
#: that plateau.
CASSCF_POLISH_AH_CONV_TOL = 1e-16

#: How far from stationary a ``--mo-from`` sidecar's orbitals may be: the
#: 2-norm of ``mc.get_grad(mo, (casdm1, casdm2), eris)`` -- i.e.
#: ``mc1step.gen_g_hop(mo, 1, casdm1, casdm2, eris)[0]``, the quantity
#: ``conv_tol_grad`` gates -- evaluated at the sidecar's orbitals with the
#: RDMs of the fresh CASCI that just ran on them
#: (:func:`_check_casscf_stationary`). Measured, all in
#: docs/reference-data.md, "The ``--casscf`` / ``--mo-from`` gate thresholds":
#: a converged CASSCF lands at 2.5e-8 (CAS(10,10)) / 5.9e-9 (CAS(4,4)), RHF
#: orbitals at ~7e-2 (five decades over), a PySCF-default-converged CASSCF at
#: 2.8e-5. It is a **stationarity** bound and not an E_corr guarantee:
#: dE_corr/d||g_orb|| is 0.11-0.13 Eh per unit norm at CAS(10,10), so
#: orbitals accepted right AT 1e-6 could sit ~1.2e-7 from the converged
#: E_corr. That is why ``--casscf`` optimises to
#: :data:`CASSCF_CONV_TOL_GRAD` (1e-7, reaching 2.5e-8) instead of merely to
#: the gate; tightening the gate itself to 1e-7 would leave only 4x of
#: headroom over what ``mc1step`` reaches and could refuse a bigger case's
#: legitimate sidecar.
CASSCF_GRAD_TOL = 1e-6

#: How far the deterministic CASCI on a sidecar's orbitals may land from the
#: ``e_casscf`` the sidecar recorded (:func:`_check_casscf_stationary`). This
#: is the fingerprint in energy form -- a wrong basis, geometry or active
#: space shows up here by decades (RHF orbitals: 2.8e-2 / 8.1e-2), as does a
#: basis-data or PySCF change that actually moves this state, which is why
#: :func:`_mo_fingerprint` does not compare version strings. Measured
#: agreement at the real thing: 2.0e-13 (CAS(4,4)), 8.5e-14 (CAS(10,10)) --
#: four decades of headroom.
CASSCF_E_TOL = 1e-9

#: How impure a sidecar's orbitals may be, as the fraction of an orbital's
#: norm living outside its own irrep (:func:`_irrep_purity`). ``--mo-from``
#: re-derives ``orbsym`` from the committed coefficients, so the coefficients
#: have to be pure enough for that label to mean something:
#: :func:`_check_mo_symmetry` has what PySCF does with an impure set instead,
#: and why neither branch of it is acceptable here. 1e-9 is PySCF's own
#: "not strictly symmetrized" warning threshold -- five decades above what a
#: converged CASSCF measures (1.7e-14, docs/reference-data.md, "A fresh CASSCF
#: reproduces, measured") and two decades below the 1e-7 at which
#: ``label_orb_symm`` raises instead of warning.
MO_PURITY_TOL = 1e-9

#: How far a ``--mo-from`` run's geometry may be from the sidecar's, per
#: coordinate in Bohr, in the symmetry frame :func:`_build_mol` puts the
#: molecule in. Both sides come from the same parse of the same input, so the
#: agreement is normally exact; this leaves room only for a different spelling
#: of the same geometry, and is decades below any real displacement.
MO_GEOM_TOL = 1e-10

#: Sidecar fingerprint fields recorded for provenance and NOT compared by
#: :func:`_check_mo_fingerprint`. ``atom_spec`` is the ``--atom`` string as
#: typed, which may be a path to an .xyz file whose contents can change under
#: it (the geometry is compared as coordinates instead); the two version
#: strings would turn every harmless dependency bump into a refusal, while a
#: change that actually moves this state is caught by :data:`CASSCF_E_TOL`.
MO_PROVENANCE_FIELDS = frozenset({"atom_spec", "pyscf", "numpy"})


def write_golden(
    path: Path,
    *,
    ncas: int,
    nelec_a: int,
    nelec_b: int,
    ncore: int,
    ndet: int,
    e_casci: float,
    e_nevpt2_total: float,
    reference_kind: str,
    arrays: dict[str, np.ndarray],
) -> None:
    """Write the custom little-endian binary golden-reference format.

    Deliberately not ``.npz`` (a zip container): the runtime demo is a plain
    C++ GPU program (``src/``), and parsing a zip + ``.npy`` headers there
    would need either an extra dependency or a hand-rolled zip reader for no
    real benefit. This format is a flat, trivially-parsed sequence instead.

    Layout (all little-endian, doubles are f64, no padding)::

        8s      magic "NEVPT2G2"
        i i i i q  ncas, nelec_a, nelec_b, ncore, ndet
        d d     e_casci, e_nevpt2_total
        i       reference_kind (0 = casci, 1 = casscf)
        i       n_arrays
        repeated n_arrays times:
            i           name_len
            <name_len>s name (not NUL-terminated)
            i           ndim
            q * ndim    shape
            d * prod(shape)  data, C (row-major) order

    ``reference_kind`` is which reference produced the orbitals the state sits
    in (:data:`REFERENCE_KINDS`), recorded so a consumer can refuse what it
    does not support: the gradient paths are derived for a CASCI on converged
    RHF orbitals and refuse a ``"casscf"`` golden (docs/gradient-theory.md,
    §1.1). It is the one field the previous layout, :data:`GOLDEN_MAGIC_V1`,
    does not have -- a ``G1`` file is a CASCI by definition and every reader
    here, ``loadGolden`` included, still accepts one, so the committed CASCI
    goldens did not have to be rewritten for it. Regenerating one now does
    rewrite it as ``G2``, which is why the check in docs/reference-data.md,
    "Regenerating the golden file" skips the header for those files; the
    committed ``*_casscf*`` goldens were written as ``G2`` and compare whole.

    The scalar header is packed from ``golden_format.GOLDEN_HEADERS``, the
    one table every Python reader also reads it by; ``src/golden/golden.cpp``'s
    ``loadGolden`` is the C++ reader. A header change is made in both, under a
    new magic, and ``golden_layout_check.py`` must pass afterwards
    (docs/reference-data.md, "The file format").
    """
    if reference_kind not in REFERENCE_KIND_CODES:
        raise SystemExit(
            f"write_golden: reference_kind {reference_kind!r} is not one of "
            f"{sorted(REFERENCE_KIND_CODES)}"
        )
    with open(path, "wb") as f:
        f.write(gf.pack_header(gf.GOLDEN_HEADERS, GOLDEN_MAGIC, dict(
            ncas=ncas, nelec_a=nelec_a, nelec_b=nelec_b, ncore=ncore, ndet=ndet,
            e_casci=e_casci, e_nevpt2_total=e_nevpt2_total,
            reference_kind=REFERENCE_KIND_CODES[reference_kind],
        )))
        gf.write_arrays(f, arrays)


def write_gradgold(
    path: Path,
    *,
    nao: int,
    nmo: int,
    ncore: int,
    nact: int,
    nvirt: int,
    nelec_rhf: int,
    arrays: dict[str, np.ndarray],
    naux: int | None = None,
) -> None:
    """Write the gradient sidecar (``*.gradgold``) for the golden beside it.

    ``naux`` given writes the **density-fitted** flavour instead (#164):
    magic ``NEVPT2F1``, the same six header fields followed by ``i naux``,
    and :func:`_df_gradient_arrays`'s arrays -- see that function for what
    they are and why. ``golden_format.read_gradgold`` hands ``naux`` back in
    the header of such a file and only there, so a read-back header written
    with ``**header`` picks the magic it was read with.

    Same named-array section as :func:`write_golden` (``gf.write_arrays``),
    a different magic and a different scalar header -- the sidecar's header is
    the orbital-space partition the array shapes are written in terms of, not
    a CI state, because the state is the golden's job.

    Layout (all little-endian, doubles are f64, no padding)::

        8s      magic "NEVPT2D1"
        i i i i i i  nao, nmo, ncore, nact, nvirt, nelec_rhf
        i       n_arrays
        repeated n_arrays times:
            i           name_len
            <name_len>s name (not NUL-terminated)
            i           ndim
            q * ndim    shape
            d * prod(shape)  data, C (row-major) order

    What it carries, and why only this much: the seventeen one-general-index
    MO-integral blocks Park's Eq. 28 reads
    (:func:`_gradient_arrays`; docs/gradient-theory.md, §3.3), the
    full-MO-range one-electron and Fock matrices (§3.5), and ``mo_coeff``.
    NOT an ``n_mo^4`` dump: the eighteenth block ``(xv|vv)`` is the largest of
    the eighteen and is never needed, because no integral the energy reads
    carries three virtual labels and promotion cannot create one (§3.4). That
    omission is a constant factor ``O(n_occ/n_virt)``, not a change of order,
    so the inventory makes the small cases cheap and a large case is still
    large -- which is why the gradient validation set is small molecules.

    ``src/golden/golden.cppm``'s ``loadGradGold`` reads exactly this layout,
    and ``golden_format.read_gradgold`` is the Python reader (the header is
    ``golden_format.GRADGOLD_HEADERS``).
    """
    magic = GRADGOLD_MAGIC if naux is None else gf.GRADGOLD_MAGIC_DF
    with open(path, "wb") as f:
        f.write(gf.pack_header(gf.GRADGOLD_HEADERS, magic, dict(
            nao=nao, nmo=nmo, ncore=ncore, nact=nact, nvirt=nvirt, nelec_rhf=nelec_rhf,
            naux=naux,
        )))
        gf.write_arrays(f, arrays)


def write_mo_sidecar(
    path: Path,
    *,
    nao: int,
    nmo: int,
    ncore: int,
    ncas: int,
    nelec_a: int,
    nelec_b: int,
    e_casscf: float,
    fields: dict[str, str],
    arrays: dict[str, np.ndarray],
) -> None:
    """Write the MO sidecar (``*.mogold``): the converged CASSCF orbitals plus
    the fingerprint of the calculation that produced them.

    Written by ``--casscf --mo-out FILE`` (:func:`optimise_mo`) and read back
    by ``--mo-from FILE`` (:func:`read_mo_sidecar`, :func:`_build_state`).
    Nothing in ``src/`` reads it: it is an *input* to the generator, not a
    reference for the demos, which is why it has no C++ reader.

    Layout (all little-endian, doubles are f64, no padding)::

        8s      magic "NEVPT2M1"
        i i i i i i  nao, nmo, ncore, ncas, nelec_a, nelec_b
        d       e_casscf
        i       n_fields
        repeated n_fields times:
            i           key_len
            <key_len>s  key   (UTF-8)
            i           val_len
            <val_len>s  value (UTF-8)
        i       n_arrays
        repeated n_arrays times:    (identical to the golden's section)
            i           name_len
            <name_len>s name (not NUL-terminated)
            i           ndim
            q * ndim    shape
            d * prod(shape)  data, C (row-major) order

    The two arrays are ``mo_coeff`` ``(nao, nmo)`` -- already in
    core/active/virtual column order, so ``--mo-from`` hands them to
    ``mcscf.CASCI`` unsorted -- and ``geom``, the molecule's coordinates in
    Bohr in the symmetry frame :func:`_build_mol` puts it in. The scalar
    header's ``e_casscf`` is the optimiser's own energy, which
    :func:`_check_casscf_stationary` holds the loaded state to; the rest of
    the fingerprint is :func:`_mo_fingerprint`.
    """
    with open(path, "wb") as f:
        f.write(gf.pack_header(gf.MO_SIDECAR_HEADERS, MO_SIDECAR_MAGIC, dict(
            nao=nao, nmo=nmo, ncore=ncore, ncas=ncas, nelec_a=nelec_a, nelec_b=nelec_b,
            e_casscf=e_casscf,
        )))
        gf.write_fields(f, fields)
        gf.write_arrays(f, arrays)


def read_mo_sidecar(path: Path) -> dict:
    """An MO sidecar back as a dict: the scalar header's fields, ``fields``
    (the text section) and every array by name (layout:
    :func:`write_mo_sidecar`).

    A sidecar is *input*, so what this can diagnose is a refusal and not an
    assertion: the wrong magic (the usual mistake -- a golden handed to
    ``--mo-from``), a missing or mis-shaped array, trailing bytes."""
    try:
        header, fields, arrays = gf.read_mo_sidecar(path)
    except gf.FormatError as e:
        # The usual mistake is a golden handed to --mo-from.
        raise SystemExit(
            f"{e} (--mo-from reads what --casscf --mo-out writes, magic "
            f"{MO_SIDECAR_MAGIC!r})"
        ) from e
    out: dict = dict(header, fields=fields)
    out.update(arrays)
    for name in ("mo_coeff", "geom"):
        if name not in out:
            raise SystemExit(f"{path}: no {name} array")
    want = (out["nao"], out["nmo"])
    if out["mo_coeff"].shape != want:
        raise SystemExit(
            f"{path}: mo_coeff is {out['mo_coeff'].shape}, but its header says "
            f"{want}"
        )
    return out


def _irrep_purity(mol, mo) -> np.ndarray:
    """``1 - max_i w_i`` per orbital, where ``w_i`` is the fraction of the
    orbital's norm living in irrep ``i``.

    The irrep subspaces are S-orthogonal and span everything, so the weights
    sum to 1 and this is 0 for a symmetry-pure orbital.
    :func:`_check_mo_symmetry` is what refuses on it; ``casscf_repro_probe.py``
    is where it was measured on converged CASSCF orbitals."""
    s = mol.intor("int1e_ovlp")
    nrm = np.einsum("pi,pq,qi->i", mo, s, mo)
    weights = []
    for blk in mol.symm_orb:
        if blk.shape[1] == 0:
            continue
        bsb = blk.T @ s @ blk
        proj = blk.T @ s @ mo
        weights.append(np.einsum("km,km->m", np.linalg.solve(bsb, proj), proj) / nrm)
    return 1.0 - np.max(np.stack(weights), axis=0)


def _orbsym_names(mol, mo) -> list[str]:
    """PySCF irrep names of ``mo``, in order."""
    from pyscf import symm

    ids = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo)
    return [symm.irrep_id2name(mol.groupname, i) for i in ids]


def _check_mo_symmetry(mol, mo) -> tuple[np.ndarray, list[str], float]:
    """``orbsym`` re-derived from a set of MO coefficients, refusing orbitals
    that are not symmetry-pure. Returns ``(mo tagged with orbsym, irrep names,
    worst purity defect)``.

    The tag is not decoration: ``mc1step_symm`` masks its unique orbital
    variables with ``mo_coeff.orbsym``, so ``get_grad``
    (:func:`_check_casscf_stationary`) needs it, and PySCF's
    ``hf_symm.get_orbsym`` takes a present tag as the answer.

    Why the purity is a refusal and not a warning: handed an impure
    ``mo_coeff``, ``casci_symm.label_symmetry_`` either **replaces** the
    orbitals with ``symm.symmetrize_space``'s projection of them (above
    ``label_orb_symm``'s own 1e-7, a ValueError it catches) or labels them
    anyway with a ``logger.warn`` that ``verbose=0`` never prints (between
    1e-9 and 1e-7). The first silently builds a golden for different orbitals
    than the sidecar's; the second has the symmetry-adapted FCI solver impose
    a symmetry the orbitals do not have. Both are wrong answers rather than
    noisy ones, so :data:`MO_PURITY_TOL` sits at the lower edge of that
    window."""
    from pyscf import lib, symm

    defect = _irrep_purity(mol, mo)
    worst = int(np.argmax(defect))
    if defect[worst] > MO_PURITY_TOL:
        raise SystemExit(
            f"orbital {worst} is not symmetry-pure in {mol.groupname}: "
            f"{defect[worst]:.1e} of its norm lies outside its own irrep "
            f"(> {MO_PURITY_TOL:g}). PySCF would answer that by projecting "
            "these orbitals onto the irreps (a different state than the "
            "sidecar's) or by labelling them anyway and imposing a symmetry "
            "they do not have. Re-run --casscf in this point group, or "
            "symmetrize the orbitals (pyscf.symm.symmetrize_space)."
        )
    ids = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo)
    names = [symm.irrep_id2name(mol.groupname, i) for i in ids]
    return lib.tag_array(mo, orbsym=ids), names, float(defect[worst])


def _check_mo_irreps(names: list[str], ncore: int, ncas: int,
                     cas_irreps: dict[str, int] | None,
                     core_irreps: dict[str, int] | None) -> None:
    """With ``--cas-irreps``/``--core-irreps``, the irrep composition of a
    sidecar's core and active windows must be the one that was asked for.
    ``--mo-from`` does NOT re-sort the loaded orbitals -- ``--casscf`` already
    sorted them, and re-sorting converged orbitals by irrep could only shuffle
    them -- so this is what checks that the active window really holds the
    requested orbitals."""
    from collections import Counter

    spans = (("active", names[ncore : ncore + ncas], cas_irreps),
             ("core", names[:ncore], core_irreps))
    for label, got, want in spans:
        if want is None:
            continue
        if Counter(got) != Counter(want):
            raise SystemExit(
                f"the {label} window of these orbitals holds "
                f"{dict(sorted(Counter(got).items()))}, but the requested "
                f"{label} irreps are {dict(sorted(want.items()))}: the "
                "sidecar's orbitals are not ordered the way this run's flags "
                "describe."
            )


def _irrep_text(counts: dict[str, int] | None) -> str:
    """``{'A"': 8, "A'": 1}`` -> ``"A'=1,A\\"=8"``: the canonical, order-free
    spelling of an irrep selection, so a fingerprint compare does not depend
    on the order the flag was typed in. ``None`` is the empty string."""
    if counts is None:
        return ""
    return ",".join(f"{k}={v}" for k, v in sorted(counts.items()))


def _mo_fingerprint(
    mol,
    mc,
    nmo: int,
    atom: str,
    basis: str,
    auxbasis: str | None,
    cas_irreps: dict[str, int] | None,
    core_irreps: dict[str, int] | None,
) -> tuple[dict, dict[str, str], np.ndarray]:
    """The sidecar's own description of the calculation its orbitals came from:
    ``(scalars, fields, geom)``, exactly what :func:`write_mo_sidecar` takes.

    One function so that the writer and :func:`_check_mo_fingerprint` cannot
    drift: ``--casscf`` writes this, and ``--mo-from`` rebuilds it from the
    molecule and flags in front of it and refuses on any difference.

    Compared on load: the AO and MO counts, the orbital-space partition
    (``ncore``/``ncas``/``nelecas``), the detected point group, the basis and
    auxiliary-basis names, the irrep selection, and the **geometry** -- as
    coordinates in the symmetry frame, not as the ``--atom`` string, because
    that string can be a path to an .xyz file whose contents can change under
    it. Recorded and not compared: :data:`MO_PROVENANCE_FIELDS`."""
    import pyscf

    na, nb = mc.nelecas
    scalars = dict(
        nao=int(mol.nao),
        nmo=int(nmo),
        ncore=int(mc.ncore),
        ncas=int(mc.ncas),
        nelec_a=int(na),
        nelec_b=int(nb),
    )
    fields = {
        "symbols": " ".join(mol.atom_symbol(i) for i in range(mol.natm)),
        "basis": basis,
        "auxbasis": auxbasis or "",
        "groupname": mol.groupname,
        "cas_irreps": _irrep_text(cas_irreps),
        "core_irreps": _irrep_text(core_irreps),
        "atom_spec": atom,
        "pyscf": pyscf.__version__,
        "numpy": np.__version__,
    }
    return scalars, fields, np.ascontiguousarray(mol.atom_coords())


def _check_mo_fingerprint(side: dict, path: Path, *args) -> None:
    """Refuse a ``--mo-from`` sidecar that does not describe the calculation
    this run is doing. ``*args`` is :func:`_mo_fingerprint`'s argument list
    for *this* run.

    Every difference is collected and reported at once, because the usual
    cause is one wrong flag and the full list says which."""
    scalars, fields, geom = _mo_fingerprint(*args)
    bad = [
        f"{k}: sidecar {side[k]}, this run {v}"
        for k, v in scalars.items()
        if side[k] != v
    ]
    bad += [
        f"{k}: sidecar {side['fields'].get(k)!r}, this run {v!r}"
        for k, v in fields.items()
        if k not in MO_PROVENANCE_FIELDS and side["fields"].get(k) != v
    ]
    if side["geom"].shape != geom.shape:
        bad.append(
            f"geometry: sidecar has {side['geom'].shape[0]} atom(s), this run "
            f"{geom.shape[0]}"
        )
    else:
        d = float(np.max(np.abs(side["geom"] - geom))) if geom.size else 0.0
        if d > MO_GEOM_TOL:
            bad.append(
                f"geometry: {d:.2e} Bohr from the sidecar's, per coordinate "
                f"(> {MO_GEOM_TOL:g})"
            )
    if bad:
        raise SystemExit(
            f"{path} was written for a different calculation:\n  "
            + "\n  ".join(bad)
            + "\nThe orbitals in an MO sidecar only mean anything for the state "
            "they were optimised for, so --mo-from refuses rather than "
            "silently building a golden for something else. Re-run "
            "--casscf --mo-out with this run's flags, or fix the flags."
        )


def _casscf(mf, ncas: int, nelecas, auxbasis: str | None):
    """``mcscf.CASSCF``, or ``DFCASSCF`` sharing the RHF's own ``with_df`` so
    one fitted integral set feeds the optimiser too. Nothing is optimised
    here; :func:`_check_casscf_stationary` builds one of these purely to
    evaluate a gradient."""
    from pyscf import mcscf

    if auxbasis is not None:
        mc = mcscf.DFCASSCF(mf, ncas, nelecas)
        assert mc.with_df is mf.with_df, "DFCASSCF built its own DF object"
    else:
        mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver.conv_tol = FCI_CONV_TOL
    return mc


def _check_casscf_stationary(
    mf, mc, mo: np.ndarray, e_casscf: float, auxbasis: str | None
) -> tuple[float, float]:
    """Refuse a ``--mo-from`` sidecar whose orbitals are not a converged CASSCF
    stationary point of the state ``mc`` just solved -- the way
    :func:`_check_degeneracy` refuses a state whose answer would depend on
    noise. Returns ``(||g_orb||, |E_CASCI - E_CASSCF|)``.

    Two independent refusals, both measured in docs/reference-data.md, "The
    ``--casscf`` / ``--mo-from`` gate thresholds":

    * :data:`CASSCF_GRAD_TOL` on the CASSCF orbital-gradient norm at these
      orbitals, with the RDMs of the fresh CASCI that just ran on them. This
      is the gate that catches RHF orbitals handed in as a sidecar (five
      decades over) and a loosely converged optimisation (PySCF's own
      defaults: one decade over, and 3.6e-6 off in E_corr). It bounds
      **stationarity**, not E_corr -- see :data:`CASSCF_GRAD_TOL` for the
      measured slope.
    * :data:`CASSCF_E_TOL` on ``|E_CASCI - e_casscf|``, the fingerprint in
      energy form: a wrong basis, geometry or active space moves it by
      decades.

    ``get_grad`` lives on ``mc1step``, not on ``CASCI``, so an unoptimised
    ``mcscf.CASSCF``/``DFCASSCF`` on the same ``_scf`` is built here to
    evaluate it. It runs one ``ao2mo`` and one ``gen_g_hop``; no optimiser is
    entered."""
    mcs = _casscf(mf, mc.ncas, mc.nelecas, auxbasis)
    mcs.mo_coeff = mo
    eris = mcs.ao2mo(mo)
    casdm1, casdm2 = mcs.fcisolver.make_rdm12(mc.ci, mcs.ncas, mcs.nelecas)
    g_norm = float(np.linalg.norm(mcs.get_grad(mo, (casdm1, casdm2), eris)))
    d_e = abs(float(mc.e_tot) - e_casscf)
    if g_norm > CASSCF_GRAD_TOL:
        raise SystemExit(
            f"these orbitals are not a CASSCF stationary point: the orbital "
            f"gradient norm is {g_norm:.3e} (> {CASSCF_GRAD_TOL:g}). RHF "
            "orbitals measure ~7e-2 and a PySCF-default-converged CASSCF "
            "2.8e-5, whose E_corr is already 3.6e-6 from the converged one -- "
            "far outside the 1e-7 the demos check to. Re-run --casscf, which "
            f"optimises to conv_tol_grad {CASSCF_CONV_TOL_GRAD:g} and reaches "
            "~2.5e-8 (docs/reference-data.md, \"The --casscf / --mo-from gate "
            "thresholds\")."
        )
    if d_e > CASSCF_E_TOL:
        raise SystemExit(
            f"the CASCI on these orbitals gives {mc.e_tot:.12f}, "
            f"{d_e:.3e} from the sidecar's E_CASSCF {e_casscf:.12f} "
            f"(> {CASSCF_E_TOL:g}). The same orbitals in the same state "
            "reproduce it to ~1e-13, so this is the fingerprint failing in "
            "energy form: a different basis, geometry, active space or "
            "basis-set/PySCF definition of them."
        )
    return g_norm, d_e


def optimise_mo(
    atom: str,
    basis: str,
    ncas: int,
    nelecas: int,
    name: str,
    auxbasis: str | None = None,
    cas_irreps: dict[str, int] | None = None,
    core_irreps: dict[str, int] | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """``--casscf``: one symmetry-adapted ``mcscf.CASSCF`` from the usual RHF +
    ``--cas-irreps``/``--core-irreps`` start, returning what
    :func:`write_mo_sidecar` needs to commit its converged orbitals.

    This is the only entry point in this file that runs an orbital optimiser
    (:func:`casscf_optimise` is its body), and it writes no golden: the
    orbitals are committed, and a *second* invocation, ``--mo-from``, builds
    the golden by running the same deterministic ``mcscf.CASCI`` on them that
    the RHF path runs on RHF orbitals. So a golden's state depends on a file,
    not on an optimiser -- which is the point, rather than any
    irreproducibility of the optimiser itself (it reproduces;
    :data:`MO_SIDECAR_MAGIC`).

    Settings: :data:`CASSCF_CONV_TOL` / :data:`CASSCF_CONV_TOL_GRAD` /
    :data:`CASSCF_MAX_CYCLE_MACRO`, and :data:`FCI_CONV_TOL` on the Davidson,
    in the point group :func:`_build_mol` detects. If that run does not
    converge, one warm-started polish with :data:`CASSCF_POLISH_AH_CONV_TOL`
    follows (it is what converges LiF CAS(4,6) and acrolein CAS(6,5)); a case
    that converges first time never runs it. Three refusals before a
    sidecar is written -- an unconverged optimisation, orbitals that are not
    symmetry-pure (:func:`_check_mo_symmetry`), and an orbital gradient above
    :data:`CASSCF_GRAD_TOL` -- so ``--mo-out`` never writes a file
    ``--mo-from`` would turn around and refuse."""
    out = casscf_optimise(atom, basis, ncas, nelecas, name, auxbasis,
                          cas_irreps=cas_irreps, core_irreps=core_irreps)
    return out["meta"], out["arrays"]


def casscf_optimise(
    atom,
    basis: str,
    ncas: int,
    nelecas: int,
    name: str,
    auxbasis: str | None = None,
    cas_irreps: dict[str, int] | None = None,
    core_irreps: dict[str, int] | None = None,
    *,
    start: tuple[np.ndarray, np.ndarray | None] | None = None,
    refuse: bool = True,
    verbose: bool = True,
) -> dict:
    """:func:`optimise_mo`'s body, shared with ``nevpt2_numgrad.py --reference
    casscf``, which re-optimises at every displaced geometry and must do so
    through **this** code path, so that a displaced state and the committed
    reference are defined by one optimiser (it then finishes each state
    itself: ``nevpt2_numgrad.finish_casscf``).

    Returns a dict: ``meta`` and ``arrays`` (what :func:`write_mo_sidecar`
    takes), ``mol``, ``mf``, ``mc``, and the optimiser's own diagnostics --
    ``converged``, ``first_converged`` (whether the first run converged),
    ``n_macro``, ``polish_macro`` (``None`` if no polish ran), ``g_norm`` (the
    2-norm of ``mc.get_grad`` with the optimiser's own CI vector: the quantity
    ``conv_tol_grad`` gates), ``purity`` and ``names`` (the irrep labels).

    The keyword-only arguments are the numerical gradient's, and every one
    defaults to what ``--casscf`` does:

    * ``start`` -- ``(mo, ci)`` to start from instead of the RHF orbitals
      sorted by ``--cas-irreps``: orbitals already in core/active/virtual
      order, as ``--mo-from`` hands them to ``mcscf.CASCI``. The numerical
      gradient's *warm-start* probe; its oracle is the cold start (``None``).
    * ``refuse`` -- ``False`` returns an unconverged optimisation, or one whose
      orbitals fail the irrep-selection or :data:`CASSCF_GRAD_TOL` checks,
      instead of exiting, so the caller's guard can name the failure. Impure
      orbitals are refused either way (:func:`_check_mo_symmetry`): no irrep
      label means anything on them."""
    from pyscf import scf

    t0 = time.perf_counter()
    mol = _build_mol(atom, basis)
    mf = scf.RHF(mol)
    if auxbasis is not None:
        mf = mf.density_fit(auxbasis=auxbasis)
    mf.kernel()
    assert mf.converged, "RHF did not converge"

    mc = _casscf(mf, ncas, nelecas, auxbasis)
    mc.conv_tol = CASSCF_CONV_TOL
    mc.conv_tol_grad = CASSCF_CONV_TOL_GRAD
    mc.max_cycle_macro = CASSCF_MAX_CYCLE_MACRO
    macro: list = []
    mc.callback = lambda envs: macro.append(envs.get("imacro"))
    ci_start = None
    if start is not None:
        mo_start, ci_start = start
    else:
        mo_start = mf.mo_coeff
        if cas_irreps is not None:
            assert sum(cas_irreps.values()) == ncas, f"{cas_irreps} is not {ncas} orbitals"
            mo_start = mc.sort_mo_by_irrep(cas_irreps, core_irreps, mo_coeff=mo_start)
    mc.kernel(mo_start, ci0=ci_start)
    n_macro = (max(macro) + 1) if macro else 0
    first_converged = bool(mc.converged)
    polish_macro = None
    if not mc.converged:
        # The fallback (CASSCF_POLISH_AH_CONV_TOL): same settings, warm-started
        # from where the first run stalled, tighter AH solve. Never runs for a
        # case that converged above, so it moves no file that did.
        if verbose:
            print(
                f"[{name}] CASSCF not converged in {CASSCF_MAX_CYCLE_MACRO} macro "
                f"iterations; polishing with ah_conv_tol {CASSCF_POLISH_AH_CONV_TOL:g}"
            )
        mcp = _casscf(mf, ncas, nelecas, auxbasis)
        mcp.conv_tol = CASSCF_CONV_TOL
        mcp.conv_tol_grad = CASSCF_CONV_TOL_GRAD
        mcp.max_cycle_macro = CASSCF_MAX_CYCLE_MACRO
        mcp.ah_conv_tol = CASSCF_POLISH_AH_CONV_TOL
        p_macro: list = []
        mcp.callback = lambda envs: p_macro.append(envs.get("imacro"))
        mcp.kernel(mc.mo_coeff, ci0=mc.ci)
        polish_macro = max(p_macro) if p_macro else 0
        mc = mcp
    polish_text = "" if polish_macro is None else f" + polish macro={polish_macro}"
    if not mc.converged and refuse:
        raise SystemExit(
            f"CASSCF did not converge to conv_tol {CASSCF_CONV_TOL:g} / "
            f"conv_tol_grad {CASSCF_CONV_TOL_GRAD:g} in "
            f"{CASSCF_MAX_CYCLE_MACRO} macro iterations, nor after the "
            f"ah_conv_tol {CASSCF_POLISH_AH_CONV_TOL:g} polish. Non-stationary "
            "orbitals are what --mo-from's gate exists to refuse, so none are "
            "written."
        )

    mo = mc.mo_coeff
    _, names, purity = _check_mo_symmetry(mol, mo)
    if refuse:
        _check_mo_irreps(names, mc.ncore, ncas, cas_irreps, core_irreps)
    eris = mc.ao2mo(mo)
    casdm1, casdm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, mc.nelecas)
    g_norm = float(np.linalg.norm(mc.get_grad(mo, (casdm1, casdm2), eris)))
    if g_norm > CASSCF_GRAD_TOL and refuse:
        raise SystemExit(
            f"CASSCF reported convergence but its orbital gradient norm is "
            f"{g_norm:.3e} (> {CASSCF_GRAD_TOL:g}), which --mo-from would "
            "refuse: not writing a sidecar."
        )

    na, nb = mc.nelecas
    nmo = mo.shape[1]
    if verbose:
        print(
            f"[{name}] CASSCF CAS({ncas},{na + nb}) ncore={mc.ncore} "
            f"E_CASSCF={mc.e_tot:.10f} |g_orb|={g_norm:.3e} "
            f"purity defect {purity:.1e} macro={n_macro}{polish_text} "
            f"({time.perf_counter() - t0:.1f}s)"
        )
        print(f"[{name}] active irreps: {names[mc.ncore : mc.ncore + ncas]}")
    scalars, fields, geom = _mo_fingerprint(
        mol, mc, nmo, atom, basis, auxbasis, cas_irreps, core_irreps
    )
    meta = dict(**scalars, e_casscf=float(mc.e_tot), fields=fields)
    return dict(
        meta=meta, arrays=dict(mo_coeff=mo, geom=geom), mol=mol, mf=mf, mc=mc,
        converged=bool(mc.converged), first_converged=first_converged,
        n_macro=n_macro, polish_macro=polish_macro, g_norm=g_norm,
        purity=purity, names=names, wall=time.perf_counter() - t0,
    )


def _build_state(
    atom: str,
    basis: str,
    ncas: int,
    nelecas: int,
    auxbasis: str | None = None,
    cas_irreps: dict[str, int] | None = None,
    core_irreps: dict[str, int] | None = None,
    mo_from: str | Path | None = None,
):
    """Deterministic CASCI + NEVPT canonicalisation: a plain CASCI
    diagonalisation on converged RHF orbitals, with no orbital optimiser in
    the chain. A fresh CASSCF **is** reproducible under these settings
    (measured: docs/reference-data.md, "What 'CASSCF input' means here, and
    why the golden state is CASCI", and
    ``reference_data/casscf_repro_probe.py``); what it is not is the same
    reference -- optimising the orbitals moves E_corr by tens of
    millihartree.

    ``mo_from`` is that other reference, without an optimiser in this chain
    either: the same deterministic CASCI, on the converged CASSCF orbitals
    committed in an MO sidecar (``--casscf --mo-out``, :func:`optimise_mo`)
    instead of on RHF's. The loaded orbitals are already in
    core/active/virtual order, so they are handed to ``mcscf.CASCI``
    unsorted, and three refusals run around the solve:
    :func:`_check_mo_fingerprint` (the sidecar describes this calculation),
    :func:`_check_mo_symmetry` + :func:`_check_mo_irreps` (``orbsym``
    re-derived from the committed coefficients, refusing impure orbitals) and
    :func:`_check_casscf_stationary` (they are a converged stationary point).
    Everything downstream -- the canonicalisation, :func:`_check_degeneracy`,
    the RDMs, PySCF's per-class energies, block2's PC -- is unchanged, and the
    golden records ``reference_kind="casscf"``.

    With ``auxbasis`` the RHF is density-fitted and the CASCI is a
    ``DFCASCI`` sharing the RHF's ``with_df`` object, so one fitted integral
    set feeds everything downstream.

    Both paths are bit-identical run to run only because of two settings,
    neither of which PySCF defaults to (docs/reference-data.md,
    "Reproducibility"): single-threading, forced at the top of this module,
    and :data:`FCI_CONV_TOL` on the Davidson. Without them, both paths
    measured E_corr up to ~9e-8 apart back to back at CAS(10,10). Reproducible
    is not the same as well-defined: point-group symmetry (:func:`_build_mol`)
    is what makes the CAS(12,12) E_corr a converged number rather than one of
    many, and :func:`_check_degeneracy` refuses a state where it
    cannot.

    ``cas_irreps`` picks the active space by irrep (``{'A"': 8}``: eight
    out-of-plane orbitals in Cs) instead of the ``ncas`` RHF orbitals
    around the Fermi level, through PySCF's ``sort_mo_by_irrep``;
    ``core_irreps`` fixes the core per irrep, since PySCF's own guess (the
    ``ncore`` lowest orbitals) can leave an occupied orbital of another irrep
    out of both core and active space."""
    from pyscf import mcscf, mrpt, scf

    mol = _build_mol(atom, basis)
    mf = scf.RHF(mol)
    if auxbasis is not None:
        mf = mf.density_fit(auxbasis=auxbasis)
    mf.kernel()
    if auxbasis is not None:
        mc = mcscf.DFCASCI(mf, ncas, nelecas)
        assert mc.with_df is mf.with_df, "DFCASCI built its own DF object"
    else:
        mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver.conv_tol = FCI_CONV_TOL
    side = None
    if mo_from is not None:
        side = read_mo_sidecar(Path(mo_from))
        _check_mo_fingerprint(
            side, Path(mo_from), mol, mc, mf.mo_coeff.shape[1], atom, basis,
            auxbasis, cas_irreps, core_irreps,
        )
        mo, names, purity = _check_mo_symmetry(mol, side["mo_coeff"])
        _check_mo_irreps(names, mc.ncore, ncas, cas_irreps, core_irreps)
        print(
            f"  --mo-from {mo_from}: E_CASSCF={side['e_casscf']:.10f} "
            f"purity defect {purity:.1e} active irreps "
            f"{names[mc.ncore : mc.ncore + ncas]}"
        )
        # Already in core/active/virtual order: --casscf sorted them.
        mc.kernel(mo)
    elif cas_irreps is not None:
        assert sum(cas_irreps.values()) == ncas, f"{cas_irreps} is not {ncas} orbitals"
        mc.kernel(mc.sort_mo_by_irrep(cas_irreps, core_irreps))
    else:
        mc.kernel()
    assert mc.converged, f"CASCI did not converge to {FCI_CONV_TOL:g}"
    if side is not None:
        g_norm, d_e = _check_casscf_stationary(
            mf, mc, mo, side["e_casscf"], auxbasis
        )
        # Kept for nevpt2_numgrad.py --reference casscf, which records the
        # gate's two numbers for every displaced state it builds here.
        mc.mo_gate = (g_norm, d_e)
        print(
            f"  --mo-from gate: |g_orb|={g_norm:.3e} (<= {CASSCF_GRAD_TOL:g}) "
            f"|E_CASCI-E_CASSCF|={d_e:.3e} (<= {CASSCF_E_TOL:g})"
        )
    pt = mrpt.NEVPT(mc)
    pt.mo_coeff, single_ci, pt.mo_energy = pt.canonicalize(
        pt.mo_coeff, ci=pt.load_ci(), cas_natorb=True
    )
    pt.ci = single_ci
    _check_degeneracy(pt)
    return mc, pt


def _build_mol(atom: str, basis: str):
    """The molecule in its detected point group, linear groups mapped to
    :data:`LINEAR_SUBGROUP`. For N2 this is exactly ``symmetry="D2h"`` --
    same frame, same symmetry orbitals -- so the committed goldens are
    unchanged; unlike a hard-coded group, it also accepts a molecule that
    lacks D2h symmetry (H2O, CO, ...)."""
    from pyscf import gto

    kw = dict(atom=atom, basis=basis, spin=0, verbose=0, symmetry=True)
    mol = gto.M(**kw)
    sub = LINEAR_SUBGROUP.get(mol.topgroup)
    if sub is not None:
        mol = gto.M(**kw, symmetry_subgroup=sub)
    return mol


def _check_degeneracy(pt) -> None:
    """Refuse a state whose pair classes are not well-defined: two canonical
    core (Sij) or virtual (Srs) orbitals of the same irrep within
    :data:`DEGENERACY_TOL`. Within one irrep, symmetry cannot pin the
    rotation between them, so numerical noise would."""
    from pyscf import symm

    mol = pt.mol
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, pt.mo_coeff)
    nocc = pt.ncore + pt.ncas
    for name, span in (("core", range(pt.ncore)), ("virtual", range(nocc, len(orbsym)))):
        idx = list(span)
        for a, p in enumerate(idx):
            for q in idx[a + 1 :]:
                gap = abs(pt.mo_energy[p] - pt.mo_energy[q])
                if orbsym[p] == orbsym[q] and gap < DEGENERACY_TOL:
                    raise SystemExit(
                        f"{name} orbitals {p} and {q} are degenerate "
                        f"(gap {gap:.1e} Eh) within irrep "
                        f"{symm.irrep_id2name(mol.groupname, orbsym[p])} of "
                        f"{mol.groupname}: the SC-NEVPT2 pair classes would "
                        "depend on numerical noise. Pick an "
                        "active space that does not cut a degenerate set, "
                        "or another molecule."
                    )


def _pyscf_dms(pt, with_df=None):
    """The RDMs + ``f3`` objects PySCF's own ``NEVPT.kernel`` builds.

    With ``with_df`` the integrals come from ``pyscf.mrpt.dfnevpt2._ERIS``,
    called DIRECTLY: ``NEVPT.kernel`` only takes the DF path when a memory
    heuristic agrees and otherwise falls back to conventional integrals
    without saying so, which would quietly make a "DF" golden file
    conventional."""
    from pyscf import fci
    from pyscf.mrpt import dfnevpt2
    from pyscf.mrpt import nevpt2 as pn

    ncas = pt.ncas
    ci = pt.load_ci()
    dm1, dm2, dm3 = fci.rdm.make_dm123(
        "FCI3pdm_kern_sf", ci, ci, ncas, pt.nelecas
    )
    if with_df is None:
        eris = pn._ERIS(pt, pt.mo_coeff)
    else:
        eris = dfnevpt2._ERIS(pt, pt.mo_coeff, with_df)
        # The in-core branch hands back arrays; the out-of-core one parks
        # cvcv in an HDF5 temp file, which _integral_blocks cannot slice.
        assert isinstance(eris["cvcv"], np.ndarray), "DF eris went out of core"
    ncore, nocc = pt.ncore, pt.ncore + ncas
    aaaa = eris["ppaa"][ncore:nocc, ncore:nocc].copy()
    la = fci.cistring.gen_linkstr_index(range(ncas), pt.nelecas[0])
    lb = fci.cistring.gen_linkstr_index(range(ncas), pt.nelecas[1])
    dms = {
        "1": dm1,
        "2": dm2,
        "3": dm3,
        "4": None,
        "f3ca": pn._contract4pdm(
            "NEVPTkern_cedf_aedf", aaaa, ci, ncas, pt.nelecas, (la, lb)
        ),
        "f3ac": pn._contract4pdm(
            "NEVPTkern_aedf_ecdf", aaaa, ci, ncas, pt.nelecas, (la, lb)
        ),
    }
    return dm1, dm2, dms, eris


def _pyscf_per_class(pt, dms, eris) -> dict:
    """PySCF's eight ``Sxxx`` energies on the shared state -- the golden answer."""
    from pyscf.mrpt import nevpt2 as pn

    ci = pt.load_ci()
    return {
        "Sr": pn.Sr(pt, ci, dms, eris),
        "Si": pn.Si(pt, ci, dms, eris),
        "Sijrs": pn.Sijrs(pt, eris),
        "Sijr": pn.Sijr(pt, dms, eris),
        "Srsi": pn.Srsi(pt, dms, eris),
        "Srs": pn.Srs(pt, dms, eris),
        "Sij": pn.Sij(pt, dms, eris),
        "Sir": pn.Sir(pt, dms, eris),
    }


def _block2_df_eris(pt, with_df):
    """block2's ``_ChemistsERIs`` filled from PySCF's DF-NEVPT2 integrals --
    the same ``ppaa/papa/pacv/cvcv/h1eff`` a ``_df`` golden is built from.

    Needed because block2 0.5.4's own ``eri_helper.init_eris`` always
    transforms the conventional 4-index AO ERIs while taking ``h1eff`` from
    the (DF) SCF's ``get_jk``, so handed a ``DFCASCI`` it silently computes a
    mixed answer; and ``kernel(eris=...)`` raises (its ``teris`` timer is
    only bound on the build path). docs/pc-nevpt2.md, "The block2 reference"."""
    from pyblock2.icmr import eri_helper
    from pyscf.mrpt import dfnevpt2

    e = dfnevpt2._ERIS(pt, pt.mo_coeff, with_df)
    ncore, ncas = pt.ncore, pt.ncas
    nmo = pt.mo_coeff.shape[1]
    nvir = nmo - ncore - ncas
    eris = eri_helper._ChemistsERIs()
    eris.ncore, eris.ncas, eris.nocc = ncore, ncas, ncore + ncas
    eris.h1eff = np.asarray(e["h1eff"])
    eris.known = ["ppaa", "papa", "pacv", "cvcv"]
    eris.ppaa = np.asarray(e["ppaa"]).reshape(nmo, nmo, ncas, ncas)
    eris.papa = np.asarray(e["papa"]).reshape(nmo, ncas, nmo, ncas)
    eris.pacv = np.asarray(e["pacv"]).reshape(nmo, ncas, ncore, nvir)
    eris.cvcv = np.asarray(e["cvcv"]).reshape(ncore, nvir, ncore, nvir)
    return eris


def _pc_cut_gap(a: np.ndarray) -> tuple[float, float, float]:
    """Where ``lstsq(rcond=None)`` cuts one PC block, as relative singular
    values: ``(gap, largest dropped, smallest kept)``, the gap being kept /
    dropped (/ the cut where a tuple drops nothing), each minimised (maxed)
    over the block's external tuples. All-zero tuples have nothing to solve
    and are skipped."""
    if len(a) == 0:  # e.g. ijrs- with a single core orbital: no tuples
        return np.inf, 0.0, np.inf
    sv = np.linalg.svd(a, compute_uv=False)
    smax = sv[:, :1]
    live = smax[:, 0] > 0
    rel = sv[live] / smax[live]
    cut = np.finfo(float).eps * a.shape[-1]
    kept = rel > cut
    min_kept = np.where(kept, rel, np.inf).min(axis=1)
    max_dropped = np.where(kept, 0.0, rel).max(axis=1)
    gap = min_kept / np.maximum(np.where((~kept).any(axis=1), max_dropped, cut),
                                np.finfo(float).tiny)
    if not len(gap):
        return np.inf, 0.0, np.inf
    return float(gap.min()), float(max_dropped.max()), float(min_kept.min())


def _block2_icmr(mc, kernel: str, *, dm4_cache: dict, gaps: list | None = None,
                 df_eris=None, rdm4_threads: int = 1, log=None) -> dict:
    """One ``pyblock2.icmr`` kernel on ``mc``'s **current** ``mo_coeff``,
    ``ci`` and ``mo_energy``, per class: ``kernel="sc"`` is
    ``scnevpt2.WickSCNEVPT2``, ``kernel="pc"`` is
    ``icnevpt2_full.WickICNEVPT2``. ``canonicalized = True``, so block2 takes
    the orbital energies it is handed rather than re-canonicalising -- which
    is what makes this a pure function of what ``mc`` holds.

    Both kernels build the dense ``n^8`` 4-RDM themselves; ``dm4_cache`` is
    where it lands, so a caller that holds the CI vector fixed across many
    calls (an orbital sweep) builds it once (``kernel(pdms=...)`` raises in
    0.5.4, hence the ``fci.rdm.make_dm1234`` patch). **The cache is keyed on
    nothing**: a caller that moves the CI vector must hand in a fresh dict,
    or block2 would contract new integrals against the old state's 4-RDM.
    ``rdm4_threads`` applies to that build alone -- the state is already
    single-threaded.

    ``gaps``, when a list is given, receives one entry per per-external-tuple
    linear solve (:data:`PC_SOLVE_ORDER` order, PC only) for
    :func:`_check_pc_conditioning`; it costs an SVD per block, so a sweep
    leaves it ``None``. ``df_eris`` replaces block2's own integral transform
    (:func:`_block2_df_eris`)."""
    from pyblock2.icmr import eri_helper, icnevpt2_full, scnevpt2
    from pyscf import lib
    from pyscf.fci import rdm as fci_rdm

    cls = {"sc": scnevpt2.WickSCNEVPT2, "pc": icnevpt2_full.WickICNEVPT2}[kernel]
    make_dm1234, init_eris = fci_rdm.make_dm1234, eri_helper.init_eris
    linear_solve = icnevpt2_full._linear_solve

    def cached_dm1234(*a, **k):
        if "dms" not in dm4_cache:
            t = time.perf_counter()
            with lib.with_omp_threads(rdm4_threads):
                dm4_cache["dms"] = make_dm1234(*a, **k)
            if log is not None:
                log(f"  block2 4-RDM: {time.perf_counter() - t:.1f}s "
                    f"({rdm4_threads} thread(s))")
        return dm4_cache["dms"]

    def recording_solve(a, b):
        gaps.append((PC_SOLVE_ORDER[len(gaps)], a.shape[-1], *_pc_cut_gap(a)))
        return linear_solve(a, b)

    fci_rdm.make_dm1234 = cached_dm1234
    if gaps is not None:
        icnevpt2_full._linear_solve = recording_solve
    if df_eris is not None:
        eri_helper.init_eris = lambda *a, **k: df_eris
    try:
        ic = cls(mc)
        ic.verbose = 0
        ic.canonicalized = True
        ic.kernel()
        return {B2_TO_CLASS[k]: float(v) for k, v in ic.sub_eners.items()}
    finally:
        fci_rdm.make_dm1234, eri_helper.init_eris = make_dm1234, init_eris
        icnevpt2_full._linear_solve = linear_solve


def _block2_pc(mc, pt, with_df, sc_ref: dict, rdm4_threads: int) -> dict:
    """block2's partially-contracted NEVPT2 (``WickICNEVPT2``) on the golden's
    own state -- its canonical orbitals and CI vector, ``canonicalized =
    True`` -- returned per class. PySCF has no PC-NEVPT2; this is the only
    reference, and nothing else checks it (docs/reference-data.md, "PC fields").

    Gated before it is returned: block2's SC (``WickSCNEVPT2``, same state,
    same 4-RDM) must equal ``sc_ref`` (PySCF's) to :data:`BLOCK2_SC_TOL` per
    class, PC Sijrs must equal SC Sijrs, and every PC block must pass
    :func:`_check_pc_conditioning`.

    Both kernels run through :func:`_block2_icmr`, sharing one 4-RDM (the CI
    vector does not move between them)."""
    mc.mo_coeff, mc.ci, mc.mo_energy = pt.mo_coeff, pt.ci, pt.mo_energy
    df_eris = _block2_df_eris(pt, with_df) if with_df is not None else None
    cache: dict = {}
    gaps: list = []
    common = dict(dm4_cache=cache, df_eris=df_eris, rdm4_threads=rdm4_threads,
                  log=print)
    sc = _block2_icmr(mc, "sc", **common)
    pc = _block2_icmr(mc, "pc", gaps=gaps, **common)

    for c in CLASSES:
        d = sc[c] - sc_ref[c]
        assert abs(d) <= BLOCK2_SC_TOL, (
            f"block2 SC {c} is {d:.1e} from PySCF's: block2 is not seeing the "
            "golden's state, so its PC answer would not be for it"
        )
    d = pc["Sijrs"] - sc["Sijrs"]
    assert abs(d) <= BLOCK2_SC_TOL, f"PC Sijrs - SC Sijrs = {d:.1e}"
    print("  block2 SC - PySCF SC, max over classes: "
          f"{max(abs(sc[c] - sc_ref[c]) for c in CLASSES):.1e}; PC lstsq cut "
          "per block (relative singular values):")
    for key, d, gap, dropped, kept in gaps:
        print(f"    {key:9} d={d:5} largest dropped {dropped:.1e}, "
              f"smallest kept {kept:.1e}, gap {gap:.1e}")
    _check_pc_conditioning(gaps)
    return pc


def _check_pc_conditioning(gaps: list) -> None:
    """Refuse a state where lstsq's cut does not sit in a gap of at least
    :data:`PC_MIN_GAP` in every PC block: there the PC energy depends on
    where the pseudo-inverse is cut, and the golden would record one
    implementation's choice, not a number a GPU solve could be checked
    against to 1e-7."""
    key, _, gap, dropped, kept = min(gaps, key=lambda g: g[2])
    if gap < PC_MIN_GAP:
        raise SystemExit(
            f"PC block {key}: lstsq's cut has only a {gap:.1e} gap (largest "
            f"dropped singular value {dropped:.1e}, smallest kept {kept:.1e}, "
            f"relative; need {PC_MIN_GAP:g}): the spectrum is a continuum "
            "through the cut, so PC-NEVPT2 is not well-defined on this state "
            "(docs/reference-data.md, \"PC fields\"). Pass --no-pc, or pick an active "
            "space that keeps near-doubly-occupied orbitals in the core."
        )


def _integral_blocks(pt, eris) -> dict:
    """Every MO-integral block the eight SC-NEVPT2 classes read (the fields
    of the C++ ``NevptIntegralsDevice``, ``src/energy/energy.cppm``),
    sliced out of PySCF's ``eris``."""
    ncas = pt.ncas
    ncore, nocc = pt.ncore, pt.ncore + ncas

    h2e_v_Sr = eris["ppaa"][nocc:, ncore:nocc].transpose(0, 2, 1, 3)
    h1e_v_Sr = eris["h1eff"][nocc:, ncore:nocc] - np.einsum(
        "mbbn->mn", h2e_v_Sr
    )
    nvirt = pt.mo_coeff.shape[1] - nocc
    cvcv = np.asarray(eris["cvcv"]).reshape(ncore, nvirt, ncore, nvirt)

    return dict(
        h1e=eris["h1eff"][ncore:nocc, ncore:nocc],
        h2e=eris["ppaa"][ncore:nocc, ncore:nocc].transpose(0, 2, 1, 3),
        e_core=pt.mo_energy[:ncore],
        e_virt=pt.mo_energy[nocc:],
        h2e_v_Sr=h2e_v_Sr,
        h1e_v_Sr=h1e_v_Sr,
        h2e_v_Si=eris["ppaa"][ncore:nocc, :ncore].transpose(0, 2, 1, 3),
        h1e_v_Si=eris["h1eff"][ncore:nocc, :ncore],
        cvcv=cvcv,
        h2e_v_Sijr=eris["pacv"][:ncore].transpose(3, 1, 2, 0),
        h2e_v_Srsi=eris["pacv"][nocc:].transpose(3, 0, 2, 1),
        h2e_v_Srs=eris["papa"][nocc:, :, nocc:].transpose(0, 2, 1, 3),
        h2e_v_Sij=eris["papa"][:ncore, :, :ncore].transpose(1, 3, 0, 2),
        h2e_v1_Sir=eris["ppaa"][nocc:, :ncore].transpose(0, 2, 1, 3),
        h2e_v2_Sir=eris["papa"][nocc:, :, :ncore].transpose(0, 3, 1, 2),
        h1e_v_Sir=eris["h1eff"][nocc:, :ncore],
    )


def _df_lpq(pt, with_df) -> np.ndarray:
    """The whole MO three-index tensor ``B[L,p,q]``, ``(naux, nmo, nmo)``,
    with ``(pq|rs) = sum_L B[L,p,q] B[L,r,s]`` -- the same fitted integrals
    ``dfnevpt2._ERIS`` built its four-index blocks from (PySCF stores
    ``_cderi`` already multiplied by the inverse square root of the metric, so
    no metric appears here). :func:`_df_blocks` slices the golden's four
    blocks out of it and :func:`_df_gradient_arrays` the sidecar's two."""
    from pyscf import lib

    mo = pt.mo_coeff
    naux = with_df.get_naoaux()
    nmo = mo.shape[1]
    lpq = np.empty((naux, nmo, nmo))
    p0 = 0
    for blk in with_df.loop():
        n = blk.shape[0]
        lao = lib.unpack_tril(blk)
        lpq[p0 : p0 + n] = lib.einsum("Lmn,mp,nq->Lpq", lao, mo, mo)
        p0 += n
    assert p0 == naux
    return lpq


def _df_blocks(pt, lpq: np.ndarray, eris) -> dict:
    """The four MO three-index blocks a ``--df`` golden carries, sliced from
    :func:`_df_lpq`'s tensor.

    Checked before returning: every four-index block ``eris`` holds is rebuilt
    from these and must agree to 1e-10, so a file whose ``B`` and four-index
    arrays disagree is never written."""
    from pyscf import lib

    ncore, ncas = pt.ncore, pt.ncas
    nocc = ncore + ncas
    nmo = lpq.shape[1]

    c, a, v = slice(0, ncore), slice(ncore, nocc), slice(nocc, None)

    def rebuilt(s1, s2, s3, s4):
        return lib.einsum("Lpq,Lrs->pqrs", lpq[:, s1, s2], lpq[:, s3, s4])

    full = slice(None)
    checks = {
        "ppaa": (eris["ppaa"], rebuilt(full, full, a, a)),
        "papa": (eris["papa"], rebuilt(full, a, full, a)),
        "pacv": (eris["pacv"], rebuilt(full, a, c, v)),
        "cvcv": (
            np.asarray(eris["cvcv"]).reshape(ncore, nmo - nocc, ncore, nmo - nocc),
            rebuilt(c, v, c, v),
        ),
    }
    for key, (ref, ours) in checks.items():
        err = float(np.max(np.abs(ref - ours))) if ref.size else 0.0
        assert err < 1e-10, f"B rebuild of {key} off by {err:.2e}"

    return dict(
        B_aa=lpq[:, a, a],
        B_ca=lpq[:, c, a],
        B_va=lpq[:, v, a],
        B_cv=lpq[:, c, v],
    )


def _gradient_arrays(mc, pt, eris, dm1: np.ndarray) -> dict:
    r"""The gradient sidecar's arrays: the one-general-index MO-integral
    inventory of docs/gradient-theory.md, §3.3, plus §3.5's full-MO-range
    matrices and ``mo_coeff``.

    Naming: ``g_xP_QR`` is the array in which the general index ``x`` shares a
    charge distribution with a label from space ``P``, the other distribution
    holding ``Q`` and ``R`` (``c``/``a``/``v`` for core/active/virtual), stored
    in chemists' order as ``arr[x, p, q, r] = (x p | q r)``. The three
    pairings of the same four space labels are *different* arrays
    (``g_xv_ca``, ``g_xa_cv``, ``g_xc_va``), as §3.3 says.

    Fourteen rows are stored in full. Three are stored only on the restricted
    slices the gradient actually reaches, which is where §3.1's two structural
    facts pay off:

    * ``(xc|cc)`` -- the core density is diagonal, so every route to it runs
      through ``(pq|jj)`` or ``(pj|jq)`` with two core labels already tied.
      Two slices, not an ``n_mo n_core^3`` array: ``g_xc_cc_j[x,i,j]`` is
      ``(x i | j j)`` (the Coulomb route) and ``g_xc_cc_k[x,j,i]`` is
      ``(x j | j i)`` (the exchange route). They are the only two independent
      ones -- ``(x i | j k)`` with ``i = k`` is ``(x k | k j)`` by the
      permutational symmetry of a real MO integral.
    * ``(xa|vv)`` and ``(xc|vv)`` -- these come only from
      ``d eps_r / d kappa`` in the form ``(rr|x.)``, so the two virtual
      labels coincide: ``g_xa_vv[x,t,r] = (x t | r r)``,
      ``g_xc_vv[x,i,r] = (x i | r r)``, both ``n_mo n_virt``-sized rather
      than ``n_mo n_virt^2``.

    Each restricted row is built as a Coulomb (or exchange) matrix of the
    rank-1 density of one orbital, back-transformed to MOs, so neither
    unrestricted parent is ever formed:
    ``(x q | j j) = [C^T J(C_j C_j^T) C]_xq`` and
    ``(x j | j q) = [C^T K(C_j C_j^T) C]_xq``.

    The full rows come from five ``ao2mo.general`` calls, one per second
    charge distribution ``(Q,R)``, each sliced into its three ``P`` rows --
    the union of a group is ``(x q | Q R)`` over all ``q``, so the five
    transients are together exactly the size of the fourteen stored rows. The
    ``(Q,R) = (a,a)`` group is PySCF's own ``ppaa``, so the sidecar's
    ``g_x*_aa`` and the golden's ``h2e`` come from one transform.

    The eighteenth block, ``(xv|vv)``, is deliberately absent: §3.4.
    """
    from pyscf import ao2mo

    mf = mc._scf
    mol = pt.mol
    mo = pt.mo_coeff
    nmo = mo.shape[1]
    ncore, nact = pt.ncore, pt.ncas
    nocc = ncore + nact
    c, a, v = slice(0, ncore), slice(ncore, nocc), slice(nocc, None)
    mo_c, mo_a, mo_v = mo[:, c], mo[:, a], mo[:, v]

    def general(q3, q4):
        """``(x q | Q R)`` for every general ``x`` and ``q``: the whole
        ``(Q,R)`` group of §3.3's table in one transform."""
        eri = ao2mo.general(mol, (mo, mo, q3, q4), compact=False)
        return eri.reshape(nmo, nmo, q3.shape[1], q4.shape[1])

    g_aa = np.asarray(eris["ppaa"]).reshape(nmo, nmo, nact, nact)
    g_va = general(mo_v, mo_a)
    g_ca = general(mo_c, mo_a)
    g_cv = general(mo_c, mo_v)
    g_cc = general(mo_c, mo_c)

    # The rank-1 densities of the core and virtual orbitals, in one get_jk
    # pass: J gives (x q | p p), K gives (x p | p q).
    dms_c = np.einsum("mj,nj->jmn", mo_c, mo_c)
    dms_v = np.einsum("mr,nr->rmn", mo_v, mo_v)
    vj, vk = mf.get_jk(mol, list(dms_c) + list(dms_v))
    vj, vk = np.asarray(vj), np.asarray(vk)
    gj = np.einsum("mp,Pmn,nq->Ppq", mo, vj, mo)  # [P, x, q] = (x q | P P)
    gk = np.einsum("mp,Pmn,nq->Ppq", mo, vk, mo)  # [P, x, q] = (x P | P q)

    return dict(
        **_gradient_matrices(mc, pt, eris, dm1),
        g_xa_aa=g_aa[:, a],
        g_xv_aa=g_aa[:, v],
        g_xc_aa=g_aa[:, c],
        g_xa_va=g_va[:, a],
        g_xv_va=g_va[:, v],
        g_xc_va=g_va[:, c],
        g_xa_ca=g_ca[:, a],
        g_xv_ca=g_ca[:, v],
        g_xc_ca=g_ca[:, c],
        g_xa_cv=g_cv[:, a],
        g_xv_cv=g_cv[:, v],
        g_xc_cv=g_cv[:, c],
        g_xa_cc=g_cc[:, a],
        g_xv_cc=g_cc[:, v],
        # (x i | j j) and (x j | j i): gj/gk's first axis is the tied core j.
        g_xc_cc_j=gj[:ncore, :, c].transpose(1, 2, 0),
        g_xc_cc_k=gk[:ncore, :, c].transpose(1, 0, 2),
        # (x t | r r) and (x i | r r): the first axis is the tied virtual r.
        g_xa_vv=gj[ncore:, :, a].transpose(1, 2, 0),
        g_xc_vv=gj[ncore:, :, c].transpose(1, 2, 0),
    )


def _gradient_matrices(mc, pt, eris, dm1: np.ndarray) -> dict:
    """§3.5's full-MO-range matrices and ``mo_coeff``, the first six arrays of
    a gradient sidecar of either flavour: ``h``, ``h^eff``, the generalized
    Fock ``f`` of the golden's ``dm1`` and ``f^h`` of a full active space, and
    the RHF Fock. Every one of them comes out of ``eris`` and ``mc._scf``, so
    on a ``--df`` state they are the fitted calculation's own, as
    docs/gradient-theory.md, §3.8 requires: ``eris`` is ``dfnevpt2._ERIS``
    and the RHF is density-fitted on the same ``with_df``."""
    mf = mc._scf
    mo = pt.mo_coeff
    nmo = mo.shape[1]
    nact = pt.ncas
    g_aa = np.asarray(eris["ppaa"]).reshape(nmo, nmo, nact, nact)
    heff = np.asarray(eris["h1eff"])  # h^eff: h + the core dressing
    h = heff - np.asarray(eris["vhf_c"])
    papa = np.asarray(eris["papa"]).reshape(nmo, nact, nmo, nact)

    def fock(gamma: np.ndarray) -> np.ndarray:
        """``h^eff + sum_tu gamma_tu [ (pq|tu) - (1/2)(pt|uq) ]`` over the
        full MO range: the generalized Fock of the active density ``gamma``."""
        return (
            heff
            + np.einsum("tu,pqtu->pq", gamma, g_aa)
            - 0.5 * np.einsum("tu,ptqu->pq", gamma, papa)
        )

    return dict(
        mo_coeff=mo,
        h=h,
        heff=heff,
        f=fock(dm1),
        # f^h: the same Fock with the active space fully occupied (gamma = 2I).
        f_h=fock(2.0 * np.eye(nact)),
        fock_rhf=mo.T @ mf.get_fock() @ mo,
    )


#: How closely the density-fitted sidecar's four tied orbital-energy arrays,
#: with its ``h^eff``, must rebuild the golden's ``e_core`` / ``e_virt``
#: (``e_p = h^eff_pp + sum_ut gamma_ut [(pp|ut) - (1/2)(pu|tp)]``), relative,
#: before :func:`_df_gradient_arrays` lets a file be written. The C++ consumer
#: (``loadOrbitalEnergyResponse``, ``--gradient-source``) makes the same check
#: at its own tolerance; this one is (O2)'s, since both sides are one
#: canonicalisation's arithmetic.
DF_EPS_REBUILD_TOL = 1e-10


def _df_gradient_arrays(mc, pt, eris, dm1: np.ndarray, lpq: np.ndarray) -> dict:
    r"""The **density-fitted** gradient sidecar's arrays (#164;
    docs/gradient-theory.md, §3.8), beside a ``--df`` golden that already
    carries ``B_aa``, ``B_ca``, ``B_va`` and ``B_cv``:

    * :func:`_gradient_matrices`' six -- ``mo_coeff``, ``h``, ``heff``,
      ``f``, ``f_h``, ``fock_rhf`` -- of the fitted calculation;
    * ``B_cc`` ``(naux, ncore, ncore)`` and ``B_vv`` ``(naux, nvirt, nvirt)``:
      the two blocks of ``(L|pq)`` the golden lacks, §3.8's "missing blocks".
      With the golden's four they are the whole tensor, which is what ``Y``
      over its full range reads. Stored square, like the golden's four (the
      packed and diagonal-only sizes are in §3.8's table): every consumer
      indexes them the same way as the four, and at the committed sizes the
      packing would save well under a megabyte;
    * the four tied arrays the orbital energies' response to the active
      density reads (``gradient_plan.cppm``'s ``OrbitalEnergyResponse``),
      under the **same names and shapes** as a conventional sidecar's --
      ``g_xa_cc`` ``(x t | i j)``, ``g_xc_ca`` ``(x i | j t)``, ``g_xa_vv``
      ``(x t | r r)`` (tied), ``g_xv_va`` ``(x r | s t)`` -- so the one C++
      reader of that response reads a fitted sidecar unchanged. Each is
      ``sum_L B B``: redundant with the six blocks, carried so the
      ``eps``-response mode needs nothing but this file and its golden.

    No conventional ``(xP|QR)`` array beyond those four: the gradient's
    density-fitted form never builds a four-index array (§3.8), and these
    four exist only for the orbital energies. Before returning, the four with
    ``h^eff`` must rebuild the golden's ``e_core`` / ``e_virt`` to
    :data:`DF_EPS_REBUILD_TOL`, so a sidecar whose response does not belong
    to its golden is never written."""
    ncore, nact = pt.ncore, pt.ncas
    nocc = ncore + nact
    c, a, v = slice(0, ncore), slice(ncore, nocc), slice(nocc, None)
    b_cc = lpq[:, c, c]
    b_vv = lpq[:, v, v]
    b_vv_diag = np.einsum("Lrr->Lr", b_vv)

    tied = dict(
        g_xa_cc=np.einsum("Lxt,Lij->xtij", lpq[:, :, a], b_cc),
        g_xc_ca=np.einsum("Lxi,Ljt->xijt", lpq[:, :, c], lpq[:, c, a]),
        g_xa_vv=np.einsum("Lxt,Lr->xtr", lpq[:, :, a], b_vv_diag),
        g_xv_va=np.einsum("Lxr,Lst->xrst", lpq[:, :, v], lpq[:, v, a]),
    )

    mats = _gradient_matrices(mc, pt, eris, dm1)
    heff = mats["heff"]
    e_core = np.array([
        heff[i, i] + np.einsum("ut,ut->", dm1, tied["g_xa_cc"][a, :, i, i]
                               - 0.5 * tied["g_xc_ca"][a, i, i, :])
        for i in range(ncore)
    ])
    e_virt = np.array([
        heff[nocc + r, nocc + r]
        + np.einsum("ut,ut->", dm1, tied["g_xa_vv"][a, :, r]
                    - 0.5 * tied["g_xv_va"][a, r, r, :])
        for r in range(lpq.shape[1] - nocc)
    ])
    mine = np.concatenate([e_core, e_virt])
    gold = np.concatenate([pt.mo_energy[:ncore], pt.mo_energy[nocc:]])
    gap = float(np.linalg.norm(mine - gold) / max(np.linalg.norm(gold), 1e-300))
    if gap > DF_EPS_REBUILD_TOL:
        raise SystemExit(
            f"the density-fitted sidecar's h^eff and orbital-energy integrals rebuild "
            f"the golden's e_core / e_virt only to {gap:.1e} relative "
            f"(> {DF_EPS_REBUILD_TOL:g}): the fitted integrals are not the ones the "
            "golden's canonicalisation used."
        )
    print(f"  df sidecar: e_core / e_virt rebuilt from h^eff + <dm1, W_p>: "
          f"{gap:.2e} relative")
    return dict(**mats, B_cc=b_cc, B_vv=b_vv, **tied)


def _check_gradient_invariants(
    pt, grad: dict, nelec_rhf: int, mo_energy_rhf, reference_kind: str = "casci"
) -> dict:
    """The three invariants of docs/gradient-theory.md, §3.6, asserted before
    a sidecar is written. They are exact by construction, so a violation means
    the state is not the one the derivation describes and every multiplier
    downstream would be meaningless -- the same stance
    :func:`_check_degeneracy` takes on a noise-dependent SC energy.

    With ``reference_kind="casscf"`` (a ``--mo-from`` golden) only (O2) is
    asserted. (O1) and (O3) are the statement that core, active and virtual
    are each spanned by RHF canonical orbitals (§1.2), and converged CASSCF
    orbitals are by construction not that: the optimiser rotated them between
    the spaces. Their stationarity is the ``--mo-from`` gate's job
    (:func:`_check_casscf_stationary`), which has already run on this state.
    (O1)'s residual is still measured and returned, so the printout shows how
    far from RHF-canonical the orbitals are; (O3) is skipped (``act_occ`` is
    ``None``).

    Returns the measured residuals, for the generator's printout."""
    if reference_kind not in REFERENCE_KIND_CODES:
        raise SystemExit(f"unknown reference_kind {reference_kind!r}")
    rhf_canonical = reference_kind == "casci"
    ncore, nact = pt.ncore, pt.ncas
    nocc = ncore + nact
    c, a, v = slice(0, ncore), slice(ncore, nocc), slice(nocc, None)
    fock_rhf, f = grad["fock_rhf"], grad["f"]

    # (O1) F is block-diagonal over core/active/virtual: the single statement
    # that the active space is a set of RHF canonical orbitals.
    blocks = {"core-act": fock_rhf[c, a], "core-virt": fock_rhf[c, v],
              "act-virt": fock_rhf[a, v]}
    o1_each = {k: (float(np.max(np.abs(b))) if b.size else 0.0) for k, b in blocks.items()}
    o1 = max(o1_each.values())
    if rhf_canonical and o1 > O1_FOCK_TOL:
        raise SystemExit(
            f"(O1) violated: the RHF Fock's core/active/virtual off-blocks reach "
            f"{o1:.1e} (> {O1_FOCK_TOL:g}). The active space is not a set of RHF "
            "canonical orbitals, so docs/gradient-theory.md's derivation does not "
            "describe this state."
        )

    # (O2) f is diagonal inside core and inside virtual, and those diagonals
    # are the golden's e_core / e_virt.
    o2 = 0.0
    for span in (c, v):
        block = f[span, span]
        if block.size:
            o2 = max(o2, float(np.max(np.abs(block - np.diag(np.diag(block))))))
    e_cv = np.concatenate([np.diag(f[c, c]), np.diag(f[v, v])])
    e_ref = np.concatenate([pt.mo_energy[:ncore], pt.mo_energy[nocc:]])
    o2 = max(o2, float(np.max(np.abs(e_cv - e_ref))) if e_cv.size else 0.0)
    if o2 > O2_FOCK_TOL:
        raise SystemExit(
            f"(O2) violated: the generalized Fock is off pseudocanonical form, or "
            f"off the golden's e_core/e_virt, by {o2:.1e} (> {O2_FOCK_TOL:g})."
        )

    # (O3) Diagonalising F's active block splits the active space into exactly
    # nelec_rhf/2 - ncore eigenvalues below the RHF gap and the rest above it,
    # with none inside: the recovery of act_O / act_V (§2.6) working.
    nocc_rhf = nelec_rhf // 2
    homo, lumo = mo_energy_rhf[nocc_rhf - 1], mo_energy_rhf[nocc_rhf]
    if not rhf_canonical:
        return {"o1": o1, "o1_each": o1_each, "o2": o2, "act_occ": None,
                "homo": homo, "lumo": lumo}
    w = np.linalg.eigvalsh(fock_rhf[a, a])
    n_below = int(np.count_nonzero(w <= homo + O1_FOCK_TOL))
    inside = w[(w > homo + O1_FOCK_TOL) & (w < lumo - O1_FOCK_TOL)]
    if n_below != nocc_rhf - ncore or inside.size:
        raise SystemExit(
            f"(O3) violated: F's active block has {n_below} eigenvalue(s) at or "
            f"below the RHF HOMO ({homo:.6f}) and {inside.size} inside the gap "
            f"({homo:.6f} .. {lumo:.6f}), but act_O must hold "
            f"{nocc_rhf - ncore} and the gap none."
        )
    return {"o1": o1, "o1_each": o1_each, "o2": o2, "act_occ": n_below,
            "homo": homo, "lumo": lumo}


def generate(
    atom: str,
    basis: str,
    ncas: int,
    nelecas: int,
    name: str,
    auxbasis: str | None = None,
    pc: bool = True,
    rdm4_threads: int = 1,
    cas_irreps: dict[str, int] | None = None,
    core_irreps: dict[str, int] | None = None,
    gradgold: bool = False,
    mo_from: str | Path | None = None,
) -> tuple[dict, dict, tuple[dict, dict] | None]:
    t0 = time.perf_counter()
    mc, pt = _build_state(
        atom, basis, ncas, nelecas, auxbasis, cas_irreps, core_irreps, mo_from
    )
    with_df = mc.with_df if auxbasis is not None else None
    dm1, dm2, dms, eris = _pyscf_dms(pt, with_df)
    ref = _pyscf_per_class(pt, dms, eris)
    blocks = _integral_blocks(pt, eris)
    lpq = None
    if with_df is not None:
        lpq = _df_lpq(pt, with_df)
        blocks.update(_df_blocks(pt, lpq, eris))
    pc_ref = _block2_pc(mc, pt, with_df, {c: ref[c][1] for c in CLASSES},
                        rdm4_threads) if pc else None
    wall = time.perf_counter() - t0

    class_norms = np.array([ref[c][0] for c in CLASSES])
    class_energies = np.array([ref[c][1] for c in CLASSES])
    e_corr = float(class_energies.sum())

    na, nb = pt.nelecas
    print(
        f"[{name}] CAS({ncas},{na + nb}) ncore={pt.ncore} n_det={pt.ci.size} "
        f"E_CASCI={mc.e_tot:.10f} E_corr(NEVPT2)={e_corr:.10f} "
        f"E_tot={mc.e_tot + e_corr:.10f}  ({wall:.1f}s to build)"
    )
    if cas_irreps is not None:
        print(f"[{name}] active space by irrep: {cas_irreps}, core: {core_irreps}")
    if with_df is not None:
        print(
            f"[{name}] density-fitted: auxbasis={auxbasis} "
            f"naux={with_df.get_naoaux()} nmo={pt.mo_coeff.shape[1]}"
        )
    if pc_ref is not None:
        print(f"[{name}] PC (block2): E_corr={sum(pc_ref.values()):.10f}  "
              + " ".join(f"{c}={pc_ref[c]:.12f}" for c in CLASSES))

    meta = dict(
        ncas=ncas,
        nelec_a=na,
        nelec_b=nb,
        ncore=pt.ncore,
        ndet=int(pt.ci.size),
        e_casci=float(mc.e_tot),
        e_nevpt2_total=e_corr,
        # Which reference produced the orbitals this CASCI diagonalised:
        # RHF's, or the converged CASSCF orbitals of a --mo-from sidecar. The
        # state is a CASCI either way; the field records the orbital source,
        # which is what a consumer has to be able to refuse (the gradient
        # paths do -- docs/gradient-theory.md, section 1.1).
        reference_kind="casscf" if mo_from is not None else "casci",
    )
    arrays = dict(
        ci=np.ascontiguousarray(np.asarray(pt.load_ci(), dtype=np.float64)),
        dm1=dm1,
        dm2=dm2,
        class_norms=class_norms,
        class_energies=class_energies,
        **blocks,
    )
    if pc_ref is not None:
        pc_class_energies = np.array([pc_ref[c] for c in CLASSES])
        arrays["pc_class_energies"] = pc_class_energies
        arrays["e_pc_total"] = np.array([pc_class_energies.sum()])

    grad = None
    if gradgold:
        nmo = pt.mo_coeff.shape[1]
        # Density-fitted: the fitted calculation's own matrices, the two
        # missing B blocks and the four tied eps arrays (#164, §3.8), and the
        # same invariants asserted against the fitted Fock -- which is the one
        # this state's derivation names (§3.8, "The rest carries over").
        grad_arrays = (_gradient_arrays(mc, pt, eris, dm1) if lpq is None
                       else _df_gradient_arrays(mc, pt, eris, dm1, lpq))
        res = _check_gradient_invariants(
            pt, grad_arrays, mc.mol.nelectron, mc._scf.mo_energy,
            reference_kind=meta["reference_kind"],
        )
        grad = (
            dict(
                nao=pt.mo_coeff.shape[0],
                nmo=nmo,
                ncore=pt.ncore,
                nact=ncas,
                nvirt=nmo - pt.ncore - ncas,
                nelec_rhf=mc.mol.nelectron,
                **({} if lpq is None else dict(naux=lpq.shape[0])),
            ),
            grad_arrays,
        )
        asserted = res["act_occ"] is not None
        o3 = (
            f"(O3) {res['act_occ']} of {ncas} active orbitals below the RHF HOMO "
            f"({res['homo']:.6f}; LUMO {res['lumo']:.6f})"
            if asserted
            else "(O3) not asserted on casscf orbitals"
        )
        print(
            f"[{name}] gradient sidecar: {len(grad_arrays)} arrays, "
            f"{sum(a.size for a in grad_arrays.values()) * 8 / 1e6:.2f} MB; "
            + ("(O1) " if asserted else "(O1, measured, not asserted) ")
            + " ".join(f"{k} {x:.2e}" for k, x in res["o1_each"].items())
            + f" (O2) {res['o2']:.2e} {o3}"
        )
    return meta, arrays, grad


def _irrep_counts(text: str) -> dict[str, int]:
    """``"A'=27,A\"=1"`` -> ``{"A'": 27, 'A"': 1}`` (PySCF irrep names)."""
    counts = {}
    for item in text.split(","):
        name, _, n = item.partition("=")
        counts[name.strip()] = int(n)
    return counts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, with every flag-combination refusal applied (each an
    ``ap.error``, so ``SystemExit(2)``). Split from :func:`main` so the
    refusals are testable without a solve (``reference_data/tests/``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atom", default="N 0 0 0; N 0 0 1.1")
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--ncas", type=int, default=10)
    ap.add_argument("--nelecas", type=int, default=10)
    ap.add_argument(
        "--name",
        default=None,
        help="case name (default: n2_ccpvdz_cas1010, with a _df suffix under --df)",
    )
    ap.add_argument(
        "--df",
        action="store_true",
        help="density-fit the whole calculation and write the B_* blocks",
    )
    ap.add_argument(
        "--auxbasis",
        default=None,
        help="auxiliary basis for --df (default: <basis>-jkfit)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output path (default: golden/<name>.nevpt2gold at the repo root)",
    )
    ap.add_argument(
        "--no-pc",
        action="store_true",
        help="skip block2's PC-NEVPT2: no pc_class_energies/e_pc_total (the "
        "CAS(12,12) files, whose PC metric is refused as ungapped)",
    )
    ap.add_argument(
        "--rdm4-threads",
        type=int,
        default=1,
        help="OpenMP threads for block2's dense 4-RDM only (default 1; the "
        "committed files were built with 1)",
    )
    ap.add_argument(
        "--cas-irreps",
        type=_irrep_counts,
        default=None,
        help="active orbitals per irrep, e.g. 'A\"=8' (default: the --ncas RHF "
        "orbitals around the Fermi level)",
    )
    ap.add_argument(
        "--gradgold",
        action="store_true",
        help="also write the gradient sidecar <out>.gradgold beside the golden "
        "(the MO-integral inventory an analytic NEVPT2 gradient reads, plus the "
        "full-range Fock matrices and mo_coeff; docs/gradient-theory.md, section 3). "
        "With --df, the density-fitted flavour: B_cc, B_vv, the fitted matrices "
        "and the four tied orbital-energy arrays (section 3.8)",
    )
    ap.add_argument(
        "--core-irreps",
        type=_irrep_counts,
        default=None,
        help="core orbitals per irrep, same syntax (default: PySCF's guess, the "
        "ncore lowest orbitals; needs --cas-irreps)",
    )
    ap.add_argument(
        "--casscf",
        action="store_true",
        help="optimise the orbitals ONCE: run a symmetry-adapted mcscf.CASSCF "
        "from the usual RHF + --cas-irreps start and write its converged "
        "mo_coeff to --mo-out. Writes no golden; feed the sidecar back with "
        "--mo-from",
    )
    ap.add_argument(
        "--mo-out",
        default=None,
        help="where --casscf writes its MO sidecar (the .mogold idiom)",
    )
    ap.add_argument(
        "--mo-from",
        default=None,
        help="build the golden by running the usual deterministic CASCI on this "
        "MO sidecar's committed CASSCF orbitals instead of on RHF's; the "
        "golden's reference_kind is then casscf",
    )
    args = ap.parse_args(argv)
    if args.auxbasis is not None and not args.df:
        ap.error("--auxbasis needs --df")
    if args.core_irreps is not None and args.cas_irreps is None:
        ap.error("--core-irreps needs --cas-irreps")
    if args.casscf and args.mo_from:
        ap.error("--casscf writes a sidecar, --mo-from reads one: not both")
    if args.casscf and args.mo_out is None:
        ap.error("--casscf needs --mo-out FILE (it writes no golden)")
    if args.mo_out is not None and not args.casscf:
        ap.error("--mo-out needs --casscf")
    # --gradgold beside a --mo-from golden is permitted: the sidecar's arrays
    # are integrals on the committed CASSCF orbitals, built with no optimiser
    # in this step. What it costs is the provenance contract -- the .mogold
    # upstream came out of one, so the sidecar reproduces from scratch only to
    # a tolerance (docs/reference-data.md, "The CASSCF-reference gradient
    # sidecar"). --casscf writes no golden, so there is nothing to sit beside.
    # A numerical gradient over committed orbitals stays refused, by
    # nevpt2_numgrad.py's reference_kind check.
    if args.gradgold and args.casscf:
        ap.error("--gradgold writes beside a golden and --casscf writes none: "
                 "pass --gradgold to the --mo-from step instead")
    # --df --gradgold is permitted (#164): docs/gradient-theory.md, section 3.8
    # carries the derivation over to three-index integrals unchanged (the
    # fitting metric is inert under an orbital rotation), and names the
    # matrices a fitted gradient reads as the fitted calculation's own, so
    # (O1)-(O3) are asserted against the fitted Fock. The sidecar is the
    # NEVPT2F1 flavour (_df_gradient_arrays).
    return args


def main() -> None:
    args = _parse_args()
    auxbasis = (args.auxbasis or f"{args.basis}-jkfit") if args.df else None
    if args.name is None:
        args.name = "n2_ccpvdz_cas1010" + ("_df" if args.df else "")

    if args.casscf:
        mo_path = Path(args.mo_out)
        mo_path.parent.mkdir(parents=True, exist_ok=True)
        meta, arrays = optimise_mo(
            args.atom, args.basis, args.ncas, args.nelecas, args.name, auxbasis,
            cas_irreps=args.cas_irreps, core_irreps=args.core_irreps,
        )
        write_mo_sidecar(mo_path, **meta, arrays=arrays)
        print(f"wrote {mo_path}")
        return

    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "golden" / f"{args.name}.nevpt2gold"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta, arrays, grad = generate(
        args.atom, args.basis, args.ncas, args.nelecas, args.name, auxbasis,
        pc=not args.no_pc, rdm4_threads=args.rdm4_threads,
        cas_irreps=args.cas_irreps, core_irreps=args.core_irreps,
        gradgold=args.gradgold, mo_from=args.mo_from,
    )
    write_golden(out_path, **meta, arrays=arrays)
    print(f"wrote {out_path}")
    if grad is not None:
        grad_path = out_path.with_suffix(".gradgold")
        write_gradgold(grad_path, **grad[0], arrays=grad[1])
        print(f"wrote {grad_path}")


if __name__ == "__main__":
    main()
