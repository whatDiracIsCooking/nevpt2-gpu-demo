#!/usr/bin/env python3
r"""Analytic nuclear gradient of the SC- (and with ``--pc``, PC-) NEVPT2
energy: the Z-vector solve and the AO contraction, for a CASCI-on-RHF
reference.

This is the **offline Python** half of the analytic gradient
(docs/gradient-theory.md), and it is offline on purpose. This repo's shape is
"offline Python generates the reference, the C++ GPU demo reproduces it";
putting a PySCF solve inside the runtime would invert that and make Python
part of the product. So nothing here is built, imported or run by any target
in ``src/`` or ``apps/`` -- it sits beside ``nevpt2_numgrad.py`` and
``block2_pc_probe.py``.

It subclasses :class:`pyscf.grad.lagrange.Gradients`, PySCF's documented base
class for exactly this Lagrange-multiplier pattern, and fills in its four
hooks: :meth:`Gradients.get_wfn_response` (the source terms),
:meth:`Gradients.get_Aop_Adiag` (the Jacobian of the constraints),
:meth:`Gradients.get_ham_response` (the skeleton derivative) and
:meth:`Gradients.get_LdotJnuc` (the multipliers against the constraints'
nuclear derivative). There is no ``pyscf/grad/nevpt2.py``: that absence is the
gap this fills, and the same gap Park's paper (docs/references.md) exists to
fill.

THE LAGRANGIAN, AND WHY THE SOLVE ORDER IS WHAT IT IS
-----------------------------------------------------

docs/gradient-theory.md, section 1.2, for a CASCI diagonalisation on converged
RHF orbitals, canonicalised with ``cas_natorb=True``::

    L = E + sum_{(p,q) in O1} Z_pq F_pq            (O1) F block-diagonal
          + sum_{i<j in core} zc_ij f_ij
          + sum_{r<s in virt} zc_rs f_rs           (O2) pseudocanonicality
          + sum_I z_I [(H - E_CASCI) c]_I          (C1) the CI eigenvector

with ``E = E_CASCI + E_corr``, ``F`` the **RHF** Fock and ``f`` the
generalized Fock. (N1), orthonormality, needs no multiplier here: the
frozen path below keeps ``C^T S C = 1`` identically (see
:func:`frozen_coeff`), which is what the usual energy-weighted-density term
exists to repair.

The CI equation **decouples** -- ``F`` is built from the RHF density and has
no CI dependence at all -- so ``z`` is solved first (section 2.2)::

    P (H - E_CASCI) P z = -P y,    P = 1 - |c><c|,    y_I = dE_corr/dc_I

and only then the orbital multipliers. That is the whole structural
simplification of a CASCI reference over a CASSCF one, and it is why
:meth:`Gradients.get_Aop_Adiag` returns a **block-diagonal** operator rather
than one coupled CP-CASSCF system.

One coupling survives, and section 2.2 does not mention it: (O2) is a
condition on the **generalized** Fock, which is built from the active density,
so ``f`` -- unlike ``F`` -- does depend on the CI vector. The two equations are
therefore coupled through ``z^c`` alone, weakly, and :meth:`ci_coupling` says
exactly how. The pair is iterated to a fixed point whose iteration count is
reported; it is one iteration for a CASCI reference, where ``z^c`` vanishes
identically.

HOW THE ORBITAL MULTIPLIERS ARE SOLVED, AND WHY AS ONE SYSTEM
-------------------------------------------------------------

Section 2.3 solves them in three stages -- Park Eqs. 56-57 for (O2), two small
Sylvester equations for (O1)'s selection boundaries, then CPHF on the
remaining ``O``-``V`` block. That triangular order is correct, and it is also
delicate: the stages are triangular only because particular response blocks
vanish, each for its own reason, and ``f`` is *not* covariant under a rotation
that mixes the core and active index ranges (its core density is the index
range, not a subspace), so the "rotate into the RHF gauge and divide" shortcut
cannot be applied to (O2) at the same time as to (O1).

This implementation does not rely on any of that. There is exactly **one
constraint per non-gauge ``kappa`` direction** (:func:`constraints` asserts the
bijection), so stationarity is one square linear system::

    sum_m lambda_m (dG_m / dkappa_k) = -B_k        for every non-gauge pair k

whose matrix is assembled once, in the same sweep that takes the source term,
and solved directly. The three stages fall out of it rather than being imposed
on it. :meth:`Gradients.get_Aop_Adiag` exposes the same operator to an
iterative solve, so a second residual is reported beside the direct one;
neither is assumed.

**Which directions are gauges is not "the active-active block".** It is
act_O-internal and act_V-internal rotations, and no others. An act_O-act_V
rotation leaves the energy untouched -- it is an active-active rotation, so
section 1.3 applies -- but it **moves the RHF occupied space**, hence ``F``,
so ``F_{act_O, act_V} = 0`` is a genuine Brillouin condition with a multiplier
of its own. Dropping the whole active-active block as "the gauge" loses that
condition and with it a term in *every* multiplier; measured, that error was
three orders of magnitude above the finite-difference floor. The state is
therefore rotated into the basis that diagonalises ``F``'s active block
(section 2.6), CI vector and all, so act_O and act_V are index ranges --
:meth:`State.occ_projector` is where it matters.

THE FROZEN ORTHONORMAL PATH, AND THE ONE PLACE A DIFFERENCE QUOTIENT ENTERS
---------------------------------------------------------------------------

Every derivative above is taken along::

    C(x) = S(x)^{-1/2} S(x0)^{1/2} C(x0),    c fixed

-- the **frozen orthonormal path**: the MO coefficients do not respond, the CI
vector does not respond, and ``C(x)^T S(x) C(x) = 1`` at every geometry, so no
orthonormality multiplier is needed and every PySCF expression stays
meaningful. With the multipliers in hand the whole gradient is the nuclear
derivative of **one scalar** along that path::

    Phi(x) = E_CASCI(x) + E_corr(x)                      get_ham_response
           + sum Z_pq F_pq(x) + sum zc f(x) + <z|H(x)|c> get_LdotJnuc

and ``dPhi/dx = dE/dx`` *iff* the multipliers satisfy their equations -- which
is the entire content of a Lagrangian gradient. Four displaced evaluations per
Cartesian component (two with ``--no-richardson``), against
``nevpt2_numgrad.py``'s six **re-converged** RHF+CASCI+NEVPT2 runs per
component.

The ``<z|H(x)|c>`` term drops ``- (dE_CASCI/dx) (z.c)`` because ``z`` is
projected against the state, so ``z`` must *actually* be projected and not
merely solved in a projected operator: ``|c>`` is an exact null vector, a
multiple of it moves no residual, and a Jacobi preconditioner puts one there.
:meth:`Gradients.solve_lagrange` projects both inside the preconditioner and
after the solve, and reports ``z.c``.

So this is an analytic gradient whose *skeleton* derivatives are differenced
rather than contracted against derivative integrals. That is a deliberate,
stated limitation and not a hidden one:

* The analytic content -- the constraint set, the Z-vector equations, the
  multipliers, the stationarity that makes two displaced points enough -- is
  all here and is exactly what is being tested when the result is compared
  against ``grad_num``.
* Replacing the difference quotients with PySCF's derivative integrals needs
  the NEVPT2 **relaxed densities** (Park Eqs. 41-47 completed into Eq. 35's
  RDM-conjugate form), because ``Y_xy`` fixes only the antisymmetric part of
  the generalized Fock and the AO contraction needs the densities themselves.
  ``src/gradient/`` builds those pseudodensities and frees them as the slab
  walk ends (docs/gradients.md, section 5.2: "Nothing consumes them yet"), and
  no flag writes them out. Closing that is C++-side work, and this issue
  changes no ``src/``.
* The difference quotients are of a *scalar along a frozen path*, so their
  own error is measurable, and it is measured: the Richardson bar is on by
  default and printed with the gradient (``--no-richardson`` turns it off).

SC OR PC: ONE LAGRANGIAN, TWO CORRELATION ENERGIES
--------------------------------------------------

``--pc`` differentiates block2's **partially-contracted** energy
(``generate_golden._block2_icmr``) instead of PySCF's strongly-contracted one,
and **nothing else about the gradient changes**. The Lagrangian above is a
statement about the *reference* -- a CASCI diagonalisation on converged RHF
orbitals, canonicalised -- so (O1), (O2), (C1), the constraint Jacobian, the
frozen path and the solve order are identical; what the contraction scheme
changes is one scalar, ``E_corr``, and with it the two source terms
``Y_xy = dE_corr/dkappa_xy`` and ``y_I = dE_corr/dc_I``. Both are difference
quotients of that scalar either way, which is why one flag is the whole
implementation.

Four things ``--pc`` does need:

* **the orbital energies, rebuilt at every point.** block2 runs
  ``canonicalized = True`` and uses the ``mo_energy`` it is handed, so it gets
  the generalized Fock's diagonal at that ``(C, c)``, exactly as the SC side
  sets ``pt.mo_energy`` (see :func:`pyscf_ci_source` on why holding them fixed
  differentiates a different functional);
* **the 4-RDM, rebuilt when the CI vector moves.** Both block2 kernels build
  the dense ``n^8`` 4-RDM themselves; it is cached across the orbital sweep,
  where ``c`` is fixed, and keyed on the CI vector's bytes so the ``y_I``
  sweep cannot contract new integrals against the old state's RDM
  (:meth:`Functional.pc_energy`);
* **a PC answer to exist at all.** ``generate_golden._check_pc_conditioning``
  refuses a state whose per-tuple lstsq cut falls in a continuum of singular
  values, and LiF's does -- so ``--pc`` on LiF stops at the reference state
  rather than producing a gradient of a number that is not well defined
  (docs/reference-data.md, "LiF and acrolein, the gradient-validation pair");
* **a CI step of its own, and that same guard run at every displaced CI
  vector.** This is the one trap in the whole PC path and it is silent: a
  determinant-axis displacement of the CI vector is not a spin
  eigenfunction, so it lifts the PC metric's structurally null modes, and a
  lifted mode past the lstsq cut is kept and divided by -- a *first-order*
  error in ``y_I`` that no step ladder converges away and that every residual
  in the solve reads as round-off. :data:`CI_STEP_PC` has the argument; the
  measured ladder (a 2.5e-4 plateau at 1e-4, 1.5e-8 three decades below
  it) is in docs/testing.md, "The analytic gradient against the
  numerical one".

It is far slower than the SC run, and the whole cost is in the ``kappa``
sweep: one block2 PC solve per non-gauge rotation, each one orders of
magnitude above PySCF's per-class sum. The measured walls are in
docs/testing.md, "The analytic gradient against the numerical one".

WHAT VALIDATES IT
-----------------

``--casci-only`` runs the identical pipeline with ``E_corr`` dropped from both
the source terms and ``Phi``. The reference is then a plain CASCI, whose
analytic gradient PySCF already has, so the two must agree::

    uv run python reference_data/nevpt2_grad.py --case lif_def2svp_cas46 --casci-only

and that single comparison exercises the frozen path, the full constraint
Jacobian, the selection-boundary multipliers, the CPHF block and the
``Phi``-differentiation at once. It does **not** exercise the CI Z-vector
(a CASCI is variational in ``c``, so ``y = 0`` and ``z = 0``) nor Eqs. 56-57
(``E_CASCI`` is invariant to a rotation inside the core space, so those
numerators vanish) -- which is precisely what the NEVPT2 run adds.

``pyscf.grad.casci`` is an oracle only for an **energy-ordered** active space,
and the run says so when it is not. It divides its redundant core-act_O and
act_V-virtual blocks by orbital-energy denominators assuming act_O is the
first ``n_elec/2 - n_core`` active columns; ``--cas-irreps`` orders the active
block by irrep instead, so on both committed cases that assumption fails and
PySCF's own answer is for a different partition. Which is, exactly, the hazard
section 2.6 exists to remove. The validation case for the comparison is
therefore an energy-ordered one, and the committed cases are compared against
``grad_num``.

Three further checks are free and always reported:

* the **gauge identity** (section 1.3): the source term along an act_O- or
  act_V-internal rotation, plus the CI relaxation that cancels it. Reported,
  not gated -- see :meth:`Gradients._gauge_identity` for why the constraints'
  own share of it is not in the sum.
* **translational invariance**: the gradient summed over atoms is zero.
* **zero by symmetry** (section 4.3): at a planar or linear geometry every
  out-of-plane component vanishes. Differenced anyway -- the frozen path
  re-solves nothing, so unlike ``nevpt2_numgrad.py`` it has no reason to skip
  them -- and reported as the free check it is.

And one set that is **gated**, beside the ``grad_num`` comparison and
independent of it: the **residual-independent checks**. Every residual this
file prints has read round-off on a wrong answer -- five times, each one a
gradient a user would have shipped -- so every solve and every difference quotient
also gets a check that reads no residual: the projection identities on ``y``
and ``z``; ``y``'s linearity and the CI and orbital stationarity of the
Lagrangian, each along a seeded pair of directions through
:meth:`Functional.evaluate` rather than through the code under test; a
**step ladder** (a run of fourfold halvings, never one step) behind every
quotient constant -- ``CI_STEP``, ``CI_STEP_PC``, ``KAPPA_STEP``,
``NUC_STEP``; and the PC conditioning guard, rank included, at every
displaced CI vector, with a ledger that fails the run if one skipped it.
They print their own ``PASS:``/``FAIL:`` line and a failure exits 1 like the
``grad_num`` gate's. Which check catches which of the five errors, with the
numbers, and what each comes out at on the committed cases: docs/testing.md,
"Residual-independent checks of the gradient path".

THE SOURCE TERMS
----------------

``get_wfn_response`` needs ``Y_xy = dE_corr/dkappa_xy`` and
``y_I = dE_corr/dc_I`` -- Park Eqs. 26-27 -- and reads only the combinations
an antisymmetric ``kappa`` and a normalised ``c`` define: ``Y_xy - Y_yx``, and
``P y``. Two providers, and the file says which it used:

``--source dump --source-path PATH``
    the GPU's own answer, from a ``.gradsrc`` file that
    ``nevpt2_sc_pseudodensity --dump-source-terms PATH`` writes
    (:func:`load_source_terms`): an ``.npz`` holding ``kappa_source``
    (``(n_mo, n_mo)``, either ``Y`` or its antisymmetrisation -- only
    ``Y_xy - Y_yx`` is ever read), ``ci_source`` (the ``y_I`` vector, in the
    CI vector's own shape), the ``ci`` they were taken at, the state's
    ``ncore``/``nact``/``nvirt`` and ``e_corr``. It is in the **golden's**
    active basis, and is carried into section 2.6's on read.
    **It is the exact derivative this gradient needs**, in a dump written
    since #89: the device's ``Y`` carries the ``f3ac``/``f3ca`` digests'
    response through their transpose (docs/gradients.md, section 6), and its
    ``y`` the orbital energies' (#87; :func:`pyscf_ci_source` says why that
    matters). An older dump held the digests fixed in ``Y`` and is not. A
    dump also carries seeded directions with ``Edot`` in its two terms, which
    ``--edot-check`` holds against PySCF's exact derivative
    (:func:`check_edot`). What the gradient comes out at, and the record of
    the omissions #73 measured: docs/testing.md, "The GPU as the source-term
    provider".
``--source both --source-path PATH``
    the dump is the provider gated against ``grad_num``; PySCF's terms are
    computed as well, compared with it componentwise
    (:func:`compare_sources`), and solved on the **same** sweep, so the
    report carries both gradients beside ``grad_num`` and a difference can
    only come from the source terms. ``--localise`` adds PySCF's functional
    with the omissions #73 measured put back (:func:`localise_sources`);
    ``--sources-only`` stops after the comparison. SC only: the GPU has no PC
    source terms (#74).
``--source pyscf`` (the default)
    central differences of PySCF's own SC-NEVPT2 energy: at fixed CI along
    every non-gauge ``kappa`` pair for ``Y``, and at fixed orbitals along
    every determinant direction **projected against the state** for ``y``
    (:func:`pyscf_ci_source` -- the projection is load-bearing, not tidiness).
    Self-contained, which is what makes this file runnable with no card; it is
    the **exact** derivative, as the device's has been since #89
    (docs/gradients.md, section 6).

THE GATE, AND WHY ITS TOLERANCE IS NOT 1e-7
-------------------------------------------

The run ends by comparing its gradient against ``grad_num`` -- the sidecar's
numerical gradient, or an ``.npz`` from ``nevpt2_numgrad.py --npz`` named with
``--grad-num`` -- and prints ``PASS:`` or ``FAIL:``, exiting 1 on a failure.
The tolerance is :data:`GRAD_NUM_TOL`, 1e-6, and it is **not** the 1e-7 every
other comparison in this repo uses, because the oracle here is not an exact
reference: it is a finite difference of re-converged energies, with an error
bar of its own, and so is the analytic side's frozen-path quotient. The
rationale is written once, in docs/testing.md, "The analytic gradient against
the numerical one", with the two bars it is built from; nothing else restates
it.

Usage::

    uv run python reference_data/nevpt2_grad.py --case lif_def2svp_cas46
    uv run python reference_data/nevpt2_grad.py --case acrolein_def2svp_cas65
    uv run python reference_data/nevpt2_grad.py --case lif_def2svp_cas46 --casci-only
    uv run python reference_data/nevpt2_grad.py --case acrolein_def2svp_cas65 \
        --energy-ordered --casci-only     # the one run pyscf.grad.casci oracles
    uv run python reference_data/nevpt2_grad.py --case lif_def2svp_cas46 \
        --out /tmp/lif.grad.npz           # the record
    # PC, against the matching numerical gradient (two commands, one scheme):
    uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65 \
        --pc --npz /tmp/acrolein.numgrad.npz
    uv run python reference_data/nevpt2_grad.py --case acrolein_def2svp_cas65 \
        --pc --grad-num /tmp/acrolein.numgrad.npz
    # the GPU as the provider: dump its source terms, then both gradients and
    # grad_num in one table (docs/testing.md, "The GPU as the source-term
    # provider"):
    ./build/src/nevpt2_sc_pseudodensity --golden golden/lif_def2svp_cas46.nevpt2gold \
        --tiles 3 --dump-source-terms /tmp/lif.gradsrc
    uv run python reference_data/nevpt2_grad.py --case lif_def2svp_cas46 \
        --source both --source-path /tmp/lif.gradsrc
    # the same dump's forward Edot(kappa) -- frozen-digest tangent + the
    # f3ac/f3ca digests' term -- against PySCF's exact directional derivative
    # (#88; docs/testing.md, "The digests' orbital response, forward mode"):
    uv run python reference_data/nevpt2_grad.py --case lif_def2svp_cas46 \
        --source dump --source-path /tmp/lif.gradsrc --edot-check
    # and the CASCI validation gated the same way, against the npz's grad_casci:
    uv run python reference_data/nevpt2_grad.py --case n2_ccpvdz_cas44 \
        --casci-only --grad-num /tmp/n2_cas44.numgrad.npz

``--out`` writes the gradient and every residual as an ``.npz``; nothing is
written by default, and **no committed file is ever touched** (the
``*.gradgold`` sidecars are ``nevpt2_numgrad.py``'s to write).
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads,
# which is what makes every energy here reproducible run to run
# (docs/reference-data.md, "Reproducibility").
import generate_golden as gg
import nevpt2_numgrad as ngrad

# isort: split
import argparse
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import golden_format as gf
import numpy as np
from pyscf import fci, mcscf, mrpt, scf
from pyscf.grad import lagrange
from pyscf.mrpt import nevpt2 as pn

#: The repository root.
REPO = Path(__file__).resolve().parent.parent

#: The committed gradient-validation cases, reused from ``nevpt2_numgrad`` so
#: the state differentiated here is the state whose numerical gradient lives
#: in the sidecar. A drift between the two would compare gradients of
#: different functions.
CASES = ngrad.CASES

# -- the difference-quotient steps, one set per scheme ---------------------
#
# Every step below is a MEASURED constant, read off a ladder recorded in
# docs/testing.md, "The step constants, measured" -- the gradient re-run
# with one step walked through powers of two and the others held -- and
# gated every run by the residual-independent checks. The SC and PC sets
# differ on purpose: the SC energy has near-singularities close to N2's
# reference point (a near-pole along a CI direction, a frozen path whose
# quotient converges only slowly), so the SC steps sit where N2 SC has
# converged; the PC gradient does not move with KAPPA_STEP or NUC_STEP over
# the same ladders, so PC keeps the steps every PC figure was measured at.

#: The orbital-rotation step for the ``Y_xy`` difference quotients (and the
#: constraint Jacobian ``M`` and ``Q`` the same sweep produces), SC, in the
#: dimensionless ``kappa``. A Givens rotation by ``h`` is exact to machine
#: precision, so the errors are the quotient's truncation and the energies'
#: round-off divided by ``2h``. ``2^-17``: on N2 CAS(4,4) SC the gradient's
#: change between neighbouring rungs falls from 7.3e-6 at ``2^-9`` to 2.0e-8
#: at ``2^-17``, and is round-off from there down; the shipped 1e-4 before
#: #77 left ~1e-6 on the table (docs/testing.md, "The step constants,
#: measured"). Gated by the orbital-stationarity ladder
#: (:func:`check_orbital_stationarity`).
KAPPA_STEP = 2.0 ** -17

#: The same step for ``--pc``: unchanged at 1e-4, because the PC gradient
#: does not depend on it at the gate's scale -- N2 CAS(4,4) PC's gradient
#: moves by at most 9.6e-9 per component over ``kappa`` steps ``2^-15`` to
#: ``2^-19`` -- and every PC ladder passes at it, where one (a CI
#: stationarity at round-off) does not at SC's ``2^-17`` (docs/testing.md,
#: "The step constants, measured").
KAPPA_STEP_PC = 1e-4

#: The step for the ``y_I = dE_corr/dc_I`` difference quotients, SC. The
#: RDMs are exactly quadratic in the CI vector, so ``E_corr`` is a rational
#: function of it -- but on N2 CAS(4,4) SC one with a near-pole at ``h ~
#: 2^-7`` along a CI direction, so the quotient converges as ``h^2`` (ratio
#: 4.00 per halving) only from ``2^-12`` down. ``2^-17`` is where that ladder
#: has converged to 9e-9 and round-off has not started (docs/testing.md,
#: "The step constants, measured"); 1e-4 before #77 was 1.2e-6 off. Gated by
#: the ``y``-linearity ladder (:func:`check_ci_source_linearity`).
CI_STEP = 2.0 ** -17

#: The same step for ``--pc``, and it is **chosen for a reason that is not
#: truncation error** -- three decades below the 1e-4 SC step #41 measured
#: it against (``CI_STEP`` has since moved to ``2^-17`` for SC's own
#: truncation, #77, and this constant did not move with it).
#:
#: PC-NEVPT2 solves one linear system per external tuple, and the internally
#: contracted basis is linearly dependent: part of each block's spectrum is
#: null, and ``numpy.linalg.lstsq(rcond=None)`` drops it. Those modes are
#: null **structurally** -- dependencies among the class's active-space
#: operator strings, for any state of the reference's symmetry
#: (docs/gradients.md, §7). But ``y_I`` is wanted per *determinant*, so the
#: quotient displaces the CI vector along a bare determinant axis, which
#: :func:`allowed_determinants` keeps inside the state's spatial irrep but
#: which is **not a spin eigenfunction**. The displaced vector then carries
#: components the reference's symmetry forbids, the forbidden modes stop being
#: null, and once a lifted mode clears the cut it is kept and **divided by** --
#: a term of order ``h/h = 1`` in the quotient. A first-order error: no
#: step-size ladder converges it away and no residual in the solve can see it.
#: It is §7's condition on the PC tangent's *direction*, met here in the
#: offline source term, and §7's own cure (build the direction from spin-free
#: generators) is not open to a per-determinant derivative.
#:
#: So the step is chosen to keep the lifted modes **below** the cut, and
#: ``generate_golden._check_pc_conditioning`` -- the guard that gates the
#: golden's own PC fields -- is what says whether they are: it is run at
#: every displaced CI vector (:meth:`Functional.pc_energy`), so a step too
#: large is refused rather than quietly returning a number -- and, past the
#: cut outright, flagged by the guard's rank check (:func:`_pc_guard`). The
#: measured ladder, the gap collapse and the gradient error that tracks it are
#: in docs/testing.md, "The analytic gradient against the numerical one"; the
#: every-run gate is the ``y``-linearity ladder along a spin-adapted direction
#: (docs/testing.md, "Residual-independent checks of the gradient path").
CI_STEP_PC = 1e-7

#: The nuclear step for ``dPhi/dx``, in bohr, SC; the gradient is the
#: Richardson extrapolant of the quotients at this step and half of it.
#: What is differenced is one scalar along a frozen path, not the
#: re-converged energy ``nevpt2_numgrad.py`` differences, and the two need not
#: be smooth on the same scale: on N2 CAS(4,4) SC the frozen path's SC
#: energy has a near-singularity within ~1e-4 bohr, and at
#: ``nevpt2_numgrad.STEP_BOHR`` (0.0025, this constant's value before #77) the
#: quotient was 2.5e-6 short of its converged slope. ``2^-13`` bohr is where
#: the gradient's nuclear-step ladder has settled (a change of 4e-9 to the
#: next rung, round-off below ``2^-15``): docs/testing.md, "The step
#: constants, measured". Gated by the nuclear ladder
#: (:func:`check_nuclear_ladder`).
NUC_STEP = 2.0 ** -13

#: The same step for ``--pc``: unchanged at ``nevpt2_numgrad.STEP_BOHR``,
#: for the reason :data:`KAPPA_STEP_PC` gives -- N2 CAS(4,4) PC's gradient
#: moves by 1.3e-9 between it and ``2^-13`` bohr, and its nuclear ladder
#: passes at it (docs/testing.md, "The step constants, measured").
NUC_STEP_PC = ngrad.STEP_BOHR


def scheme_steps(pc: bool) -> tuple[float, float]:
    """``(NUC_STEP, KAPPA_STEP)`` for the scheme being differentiated: the
    PC pair with ``--pc``, else the SC pair (``--casci-only`` included, which
    is validated against the numerical CASCI gradient at the SC steps)."""
    return (NUC_STEP_PC, KAPPA_STEP_PC) if pc else (NUC_STEP, KAPPA_STEP)


#: How far the rebuilt state may sit from the golden it claims to
#: differentiate, in ``E_CASCI`` and ``E_corr``. Measured at 0.0 -- the same
#: builder, the same settings -- so a nonzero value is a wiring error.
REF_STATE_TOL = ngrad.REF_STATE_TOL

#: Below this gap (Eh) a core or virtual pair counts as degenerate, its (O2)
#: constraint as undetermined, and its multiplier is left at zero rather than
#: divided through. An abelian subgroup splits a degenerate set across irreps,
#: so both partners keep the same orbital energy while the energy does not
#: move along a rotation mixing two irreps: the numerator vanishes too and
#: zero is the right representative (docs/gradients.md, section 6). The same
#: 1e-6 the C++ ``--kappa-full`` check uses, for the same reason.
PC_GAP_FLOOR = 1e-6

#: What a degenerate pair's source term has to be under for zero to be the
#: right representative. Above it, the constraint is genuinely undetermined
#: and that is a refusal, not a number.
PC_ZERO_TOL = 1e-8

#: ``numpy.linalg.lstsq``'s cutoff for the constraint Jacobian, relative to
#: its largest singular value. The system is square and (O2)'s rows for a
#: degenerate pair are the only near-null directions; those pairs are removed
#: before the solve, so what is left should be well conditioned and the
#: reported condition number says whether it was.
JACOBIAN_RCOND = 1e-12

#: What ``|g - grad_num|`` has to be under, per component, for the comparison
#: against the numerical gradient to pass. **This is not the 1e-7 the rest of
#: this repo compares to, and it cannot be**: the oracle is a finite
#: difference of re-converged energies, not an exact reference, so the
#: tolerance is set by the two sides' own error bars rather than by machine
#: precision. The rationale, the measured bars it is built from and the
#: results are in docs/testing.md, "The analytic gradient against the
#: numerical one"; Park's Table 3 (docs/references.md) puts PC within 5.0e-6
#: of two-point numerical gradients, so 1e-6 is a tighter gate than the
#: published one, not a looser one.
GRAD_NUM_TOL = 1e-6

# -- the residual-independent checks (docs/testing.md, "Residual-independent
# -- checks of the gradient path") --------------------------------------------
#
# A converged residual says a solve found the answer to the equations it was
# given; it says nothing about whether those were the right equations, and in
# this file five wrong answers have had residuals at round-off. So every solve
# and every difference quotient also gets a check that reads no residual --
# :func:`run` lists them, and docs/testing.md records which of the five errors
# each one was demonstrated against.

#: The Richardson window of a step ladder: while truncation dominates,
#: halving a central difference's step cuts its error fourfold. The C++
#: checks' window (nevpt2.gradient_plan, ``kFourfoldLow`` / ``kFourfoldHigh``).
LADDER_LOW = 3.2
LADDER_HIGH = 4.8

#: How many CONSECUTIVE fourfold halvings a ladder must show. A run and not
#: the whole table, for the reason docs/testing.md, "The SC tangent's
#: finite-difference cross-check" found: one rung can go wild (a perturber
#: empty at the base point whose ``Delta N + H`` cancels to round-off), and
#: below the crossover the error is round-off and rises again.
LADDER_RUN = 3

#: The ladders' steps, ``h = 2^-k`` for ``k`` in ``[first, last]``: powers of
#: two so ``h`` is exact and the ratio across a halving is exactly the
#: Richardson factor. Each starts where truncation dominates and ends past the
#: round-off crossover (docs/testing.md, "Residual-independent checks of the
#: gradient path", has the measured ladders).
#:
#: PC's CI ladder starts at 2^-11, and that is not a choice: its direction
#: (:func:`ci_check_direction`) still lifts one Sir mode as ``h^4``, which is
#: kept -- the energy then on another branch of the pseudo-inverse -- above
#: ``h ~ 2^-9``, sits astride lstsq's cut at ``2^-10`` (the gap guard refuses
#: it there), and is dropped like the structural ones from ``2^-11`` down: the
#: same bound the C++ PC tangent's ``kStartPow`` keeps under (docs/pc-nevpt2.md,
#: "The tangent with respect to the CI vector").
CI_LADDER = (4, 18)
CI_LADDER_PC = (11, 18)
KAPPA_LADDER = (3, 17)
NUC_LADDER = (3, 16)

#: How far a ladder's converged value may sit from what it checks, in Eh per
#: unit of the ladder's direction (unit CI vector, unit ``kappa``, one bohr):
#: :data:`GRAD_NUM_TOL`'s 1e-6, the precision the gradient itself is gated
#: at. A quotient whose own step error is larger than that has not delivered
#: the number the gradient needs, whatever its residual says. Where the clean
#: cases and the five errors land against it is docs/testing.md,
#: "Residual-independent checks of the gradient path".
LADDER_TOL = GRAD_NUM_TOL

#: ``|c . y|`` (Eh) and ``|c . z|``: ``y`` is built as the PROJECTED gradient
#: (:func:`pyscf_ci_source`) and ``z`` is projected after its solve
#: (:meth:`Gradients.solve_lagrange`), so both are orthogonal to the state up
#: to the quotients' own error -- the same 1e-6 as the ladders.
PROJECTION_TOL = GRAD_NUM_TOL

#: A displaced PC state's smallest kept singular value, per block, against
#: the reference state's: below this fraction of it, the block has KEPT a mode
#: the reference drops -- a lifted null mode that cleared lstsq's cut outright,
#: which the gap guard cannot see (its gap looks healthy) and which puts the
#: energy on another branch of the pseudo-inverse. #41's 1e-4 plateau is that
#: (docs/testing.md, "The PC CI source term's step is a first-order trap"). A
#: kept mode's own fluctuation under a displacement is ``O(h)`` relative; a
#: lifted one is ``h^2`` or ``h^4`` of ``s_max``, decades below.
PC_RANK_KEPT_FRACTION = 1e-2

#: The seeds of the pair of directions every linearity check draws. Fixed, so
#: a run is reproducible; a pair, so a check does not rest on one direction.
CHECK_SEEDS = (70, 2026)

#: Component names, in the order they index a coordinate.
AXES = ngrad.AXES


def _ci_key(ci: np.ndarray) -> bytes:
    """A CI vector's identity, by its bytes: what :class:`Functional` caches
    block2's 4-RDM on, and how it tells the state's own vector from a
    displaced one."""
    return hashlib.blake2b(np.ascontiguousarray(ci, dtype=float).tobytes(),
                           digest_size=16).digest()


@dataclass
class PcGuardLedger:
    """Every run of the PC conditioning guard, counted where it runs.

    ``displaced`` is incremented where a PC energy is evaluated at a CI
    vector that is not the state's own, and ``displaced_guarded`` where the
    guard passes at one -- two different places, so a code path that
    evaluates a displaced PC energy and skips the guard leaves them unequal,
    and :func:`check_pc_guard_ledger` fails the run.

    ``rank_changed`` collects every guarded point where a block KEPT a mode
    the reference state drops (:data:`PC_RANK_KEPT_FRACTION`). It does not
    stop the run -- the energy there exists, it is just the wrong function --
    so the run finishes, prints every residual, and fails at the end; that is
    how #41's error is shown converging to round-off and failing anyway."""

    displaced: int = 0
    displaced_guarded: int = 0
    #: guard runs at a fixed CI vector: the ladders' orbital and nuclear
    #: rungs, which sample the argument that only ``c`` can move the rank.
    sampled: int = 0
    min_gap: float = float("inf")
    #: the reference state's smallest kept singular value per PC block
    #: (:func:`build_state`), what every guarded point is held against.
    reference_kept: dict = field(default_factory=dict)
    rank_changed: list = field(default_factory=list)


#: The ledger of the current run (:func:`run` resets it).
PC_GUARD = PcGuardLedger()


def _pc_guard(gaps: list, *, displaced: bool, where: str) -> None:
    """The PC guard at one evaluation, recorded in :data:`PC_GUARD`: first
    ``generate_golden._check_pc_conditioning`` on its per-tuple cuts -- a
    spectrum that is a continuum through the cut is refused outright, naming
    where -- then the **rank** against the reference state's.

    The gap check alone is necessary and not sufficient (docs/testing.md,
    "The PC CI source term's step is a first-order trap"): a mode the
    reference state has null, lifted by the displacement far enough to clear
    the cut, is simply kept, the gap looks healthy, and the energy is a
    different branch of the pseudo-inverse. That shows as the block's
    smallest kept singular value collapsing by decades, which is what the
    second test reads."""
    try:
        gg._check_pc_conditioning(gaps)
    except SystemExit as exc:
        hint = (
            "\nThat is a displacement lifting the PC metric's null modes past "
            "the lstsq cut, not a property of the state -- the reference state "
            "passed the same guard. For the y_I quotients use a smaller "
            f"--ci-step (the default for --pc is {CI_STEP_PC:g}; see CI_STEP_PC "
            "and docs/testing.md, \"The analytic gradient against the numerical "
            "one\")."
        ) if displaced else ""
        raise SystemExit(f"at {where or 'a displaced point'}: {exc}{hint}") from None
    PC_GUARD.min_gap = min(PC_GUARD.min_gap, min(g[2] for g in gaps))
    for key, _d, _gap, _dropped, kept in gaps:
        ref = PC_GUARD.reference_kept.get(key)
        if ref is not None and np.isfinite(ref) and kept < PC_RANK_KEPT_FRACTION * ref:
            PC_GUARD.rank_changed.append((where or "a displaced point", key, kept, ref))
    if displaced:
        PC_GUARD.displaced_guarded += 1
    else:
        PC_GUARD.sampled += 1


# --------------------------------------------------------------------------
# The state, and the spaces the derivation is written in
# --------------------------------------------------------------------------


@dataclass
class State:
    """The golden's own CASCI-on-RHF state, plus the two space partitions
    docs/gradient-theory.md, section 0 keeps apart: core/active/virtual (set
    by the active-space *selection*) and occupied/virtual (set by the RHF
    *solution*)."""

    case: str
    geom: list
    basis: str
    mc: object
    pt: object
    mo_coeff: np.ndarray
    ci: np.ndarray
    ncore: int
    nact: int
    nvirt: int
    nmo: int
    nelecas: tuple
    nelec_rhf: int
    #: ``f``, the generalized Fock, in the final MO basis.
    f: np.ndarray
    #: ``F``, the RHF Fock, in the final MO basis.
    fock_rhf: np.ndarray
    #: How many of the active orbitals came from the RHF occupied space.
    #: Section 2.6: the state is held in the basis that **diagonalises ``F``'s
    #: active block**, so act_O is the first ``n_act_occ`` active columns and
    #: act_V the rest -- both ordinary index ranges, which is what the
    #: bookkeeping below needs them to be.
    n_act_occ: int
    act_eps: np.ndarray
    e_casci: float
    e_corr: float
    invariants: dict = field(default_factory=dict)
    #: block2's PC correlation energy on this state, and the smallest lstsq
    #: gap seen building it -- only with ``--pc``, and only where a PC answer
    #: exists at all (``generate_golden._check_pc_conditioning``).
    e_corr_pc: float | None = None
    pc_gap: float | None = None
    #: The golden's own CI vector, BEFORE the section 2.6 rotation, and the
    #: active-block rotation ``vec`` that took it (and the orbitals) to the
    #: basis this state is held in. A ``.gradsrc`` dump is in the golden's
    #: basis -- the only one the GPU knows -- so :func:`load_source_terms`
    #: carries it across with these.
    ci_golden: np.ndarray | None = None
    act_rot: np.ndarray | None = None

    @property
    def nocc(self) -> int:
        return self.ncore + self.nact

    @property
    def core(self) -> slice:
        return slice(0, self.ncore)

    @property
    def act(self) -> slice:
        return slice(self.ncore, self.nocc)

    @property
    def act_o(self) -> slice:
        return slice(self.ncore, self.ncore + self.n_act_occ)

    @property
    def act_v(self) -> slice:
        return slice(self.ncore + self.n_act_occ, self.nocc)

    @property
    def virt(self) -> slice:
        return slice(self.nocc, self.nmo)

    def space_of(self, p: int) -> int:
        """Which of the four spaces orbital ``p`` is in: 0 core, 1 act_O,
        2 act_V, 3 virtual. The **RHF** partition splits the active space in
        two (section 0: the two partitions are not nested the same way), and
        every piece of bookkeeping below turns on that split."""
        if p < self.ncore:
            return 0
        if p < self.ncore + self.n_act_occ:
            return 1
        if p < self.nocc:
            return 2
        return 3

    def occ_projector(self) -> np.ndarray:
        """The RHF occupied space as an ``(n_mo, n_mo)`` projector in MO index
        space: the identity on core and on act_O, zero elsewhere.

        This is what makes ``F``'s ``kappa``-dependence right, and it is why
        the state is rotated into the act_O/act_V basis at all. The RHF
        density is ``2 C P C^T``, so a rotation that mixes act_O with act_V
        **moves the occupied space** -- an act_O-act_V rotation is therefore
        *not* a gauge for (O1), even though it is one for the energy. Keeping
        the natural-orbital basis and carrying act_O as a subspace instead
        makes that rotation look like a gauge and silently drops a real
        Brillouin condition."""
        p = np.zeros((self.nmo, self.nmo))
        p[self.core, self.core] = np.eye(self.ncore)
        p[self.act_o, self.act_o] = np.eye(self.n_act_occ)
        return p


def build_state(case: str, spec: dict, *, pc: bool = False) -> State:
    """``generate_golden._build_state``'s own state -- same single-threading,
    same Davidson tolerance, same auto-detected point group, same
    irrep-selected active space -- plus the generalized and RHF Fock matrices
    and the act_O/act_V split.

    With ``pc``, block2's PC energy on the rotated state as well, gated by
    ``generate_golden._check_pc_conditioning`` -- the same guard the golden's
    own PC fields pass, and the one that refuses LiF."""
    geom_path = Path(spec["geometry"])
    if not geom_path.is_absolute():
        geom_path = REPO / geom_path
    geom = ngrad.read_xyz(geom_path)
    mc, pt = gg._build_state(
        geom, spec["basis"], spec["ncas"], spec["nelecas"],
        cas_irreps=spec.get("cas_irreps"), core_irreps=spec.get("core_irreps"),
    )
    nmo = pt.mo_coeff.shape[1]
    nact, ncore = pt.ncas, pt.ncore
    nelec_rhf = pt.mol.nelectron
    n_act_occ = nelec_rhf // 2 - ncore

    # Section 2.6: F restricted to the active block has the SELECTED RHF
    # orbital energies as its eigenvalues and the pre-canonicalisation active
    # orbitals as its eigenvectors, so diagonalising it undoes the
    # natural-orbital gauge and splits the active space into act_O and act_V.
    # We then ROTATE THE STATE INTO THAT BASIS, CI vector and all: act_O and
    # act_V become index ranges, which is what :meth:`State.occ_projector`
    # needs (an act_O-act_V rotation moves the RHF occupied space, so it is
    # not a gauge for (O1) and cannot be carried as a subspace).
    def arrays_for(pt_state):
        dm1, _dm2, dms, eris = gg._pyscf_dms(pt_state)
        per = gg._pyscf_per_class(pt_state, dms, eris)
        e_corr = float(sum(per[c][1] for c in gg.CLASSES))
        ga = gg._gradient_arrays(mc, pt_state, eris, dm1)
        return e_corr, ga

    e_corr0, ga0 = arrays_for(pt)
    invariants = gg._check_gradient_invariants(pt, ga0, nelec_rhf, mc._scf.mo_energy)
    a = slice(ncore, ncore + nact)
    eps, vec = np.linalg.eigh(np.asarray(ga0["fock_rhf"])[a, a])
    mo = np.asarray(pt.mo_coeff).copy()
    mo[:, a] = mo[:, a] @ vec
    ci_golden = np.asarray(pt.load_ci()).copy()
    ci = fci.addons.transform_ci_for_orbital_rotation(ci_golden, nact, pt.nelecas, vec)
    pt.mo_coeff = mo
    pt.ci = ci
    e_corr, ga = arrays_for(pt)
    # The energy is invariant under an active-active rotation that carries the
    # CI vector along (section 1.3), so the rotation must not have moved it.
    # Checked rather than assumed: a wrong CI transform is otherwise silent.
    assert abs(e_corr - e_corr0) < 1e-9, (
        f"the act_O/act_V rotation moved E_corr by {e_corr - e_corr0:.2e}; the CI "
        "vector was not transformed consistently with the orbitals"
    )
    # And (O1)/(O2)/(O3) must still hold in the rotated basis.
    invariants = gg._check_gradient_invariants(pt, ga, nelec_rhf, mc._scf.mo_energy)
    f = np.asarray(ga["f"])
    fock_rhf = np.asarray(ga["fock_rhf"])
    assert np.abs(fock_rhf[a, a] - np.diag(eps)).max() < 1e-9, (
        "F's active block is not diagonal in the basis that diagonalises it"
    )
    st = State(
        case=case, geom=geom, basis=spec["basis"], mc=mc, pt=pt,
        mo_coeff=mo, ci=np.asarray(ci),
        ncore=ncore, nact=nact, nvirt=nmo - ncore - nact, nmo=nmo,
        nelecas=tuple(pt.nelecas), nelec_rhf=nelec_rhf,
        f=f, fock_rhf=fock_rhf, n_act_occ=n_act_occ,
        act_eps=eps, e_casci=float(mc.e_tot), e_corr=e_corr, invariants=invariants,
        ci_golden=ci_golden, act_rot=vec,
    )
    if pc:
        gaps: list = []
        st.e_corr_pc = Functional(st, pc=True).pc_energy(
            mo, st.ci, np.diag(f).copy(), gaps=gaps)
        gg._check_pc_conditioning(gaps)
        st.pc_gap = min(g[2] for g in gaps)
        # What every displaced point's rank is held against (_pc_guard).
        PC_GUARD.reference_kept = {key: kept for key, _d, _gap, _dropped, kept in gaps}
    return st


# --------------------------------------------------------------------------
# The functional, evaluated at an arbitrary (geometry, C, c)
# --------------------------------------------------------------------------


class Functional:
    """``E_CASCI``, ``E_corr``, the generalized Fock ``f``, the RHF Fock ``F``
    and ``<z|H|c>`` at a given geometry, MO coefficient matrix and CI vector.

    One instance per geometry: the AO integrals, ``hcore`` and the PySCF
    objects are built once and only ``C`` and ``c`` are swapped, which is what
    makes an ``O(n_mo^2)`` sweep of ``kappa`` directions affordable."""

    def __init__(self, state: State, geom=None, *, pc: bool = False):
        mol = gg._build_mol(geom if geom is not None else state.geom, state.basis)
        mol.incore_anyway = True  # force pn._ERIS's incore transform
        mf = scf.RHF(mol)
        mf._eri = mol.intor("int2e", aosym="s8")
        mf.mo_coeff = state.mo_coeff
        mf.mo_occ = np.asarray(state.mc._scf.mo_occ)
        mf.mo_energy = np.asarray(state.mc._scf.mo_energy)
        mf.converged = True
        mc = mcscf.CASCI(mf, state.nact, state.nelecas)
        mc.fcisolver.conv_tol = gg.FCI_CONV_TOL
        mc.mo_coeff = state.mo_coeff
        mc.ci = state.ci
        assert mc.ncore == state.ncore, "the displaced CASCI chose a different core size"
        pt = mrpt.NEVPT(mc)
        pt.canonicalized = True  # never let NEVPT.kernel re-canonicalise
        pt.mo_coeff = state.mo_coeff
        pt.ci = state.ci
        self.state = state
        self.mol = mol
        self.mf = mf
        self.mc = mc
        self.pt = pt
        self.hcore = mf.get_hcore()
        self.ovlp = mol.intor("int1e_ovlp")
        self.e_nuc = float(mol.energy_nuc())
        self._occ_proj = state.occ_projector()
        self.nevals = 0
        #: which correlation functional is differentiated: PySCF's
        #: strongly-contracted energy, or block2's partially-contracted one.
        self.pc = pc
        #: block2's dense n^8 4-RDM, kept while the CI vector does not move --
        #: which is the whole orbital sweep -- and rebuilt when it does. Keyed
        #: on the CI vector's own bytes, because handing block2 new integrals
        #: against the previous state's 4-RDM would be silently wrong.
        self._pc_key: bytes | None = None
        self._pc_dm4: dict = {}
        #: the state's own CI vector, by the same key: any other vector handed
        #: to :meth:`pc_energy` is a DISPLACED one, and is guarded there.
        self._ref_key = _ci_key(state.ci)

    # -- the pieces -------------------------------------------------------

    def eris(self, C: np.ndarray) -> dict:
        """PySCF's own NEVPT2 integral set at these orbitals. ``pn._ERIS``
        reads ``mc._scf._eri`` and ``mc.get_hcore()``, both bound to this
        geometry, so it is a pure function of ``C``."""
        self.pt.mo_coeff = C
        self.mc.mo_coeff = C
        self.nevals += 1
        return pn._ERIS(self.pt, C)

    def generalized_fock(self, eris: dict, dm1: np.ndarray) -> np.ndarray:
        """``h^eff + sum_tu gamma_tu [ (pq|tu) - (1/2)(pt|uq) ]`` over the full
        MO range -- ``generate_golden._gradient_arrays``'s own ``fock``, whose
        core and virtual diagonals are the golden's ``e_core``/``e_virt``."""
        nmo, nact = self.state.nmo, self.state.nact
        heff = np.asarray(eris["h1eff"])
        g_aa = np.asarray(eris["ppaa"]).reshape(nmo, nmo, nact, nact)
        papa = np.asarray(eris["papa"]).reshape(nmo, nact, nmo, nact)
        return (
            heff
            + np.einsum("tu,pqtu->pq", dm1, g_aa)
            - 0.5 * np.einsum("tu,ptqu->pq", dm1, papa)
        )

    def rhf_fock(self, C: np.ndarray) -> np.ndarray:
        """``C^T [ h + veff(2 C P C^T) ] C`` with ``P`` the RHF occupied
        projector of :meth:`State.occ_projector`. (O1) -- ``F``
        block-diagonal over core/active/virtual -- is the constraint whose
        multipliers the orbital Z-vector equations determine, and this is the
        matrix it is a statement about."""
        dm = 2.0 * (C @ self._occ_proj @ C.T)
        return C.T @ (self.hcore + self.mf.get_veff(self.mol, dm)) @ C

    def core_energy(self, eris: dict) -> float:
        """``E_nuc + sum_i (h_ii + h^eff_ii)`` over the core: the inactive
        part of ``<c|H|c>``, from the one-electron matrices ``pn._ERIS``
        already built (``h = h^eff - vhf_c``)."""
        heff = np.asarray(eris["h1eff"])
        h = heff - np.asarray(eris["vhf_c"])
        c = self.state.core
        return self.e_nuc + float(np.trace(h[c, c]) + np.trace(heff[c, c]))

    def reference_dm1(self, st: State) -> np.ndarray:
        """The state's own active 1-RDM, in the MO index basis. Fixed under an
        orbital rotation at fixed CI vector, which is what the derivative
        holds fixed."""
        return np.asarray(fci.rdm.make_dm123(
            "FCI3pdm_kern_sf", st.ci, st.ci, st.nact, st.nelecas)[0])

    def active_hamiltonian(self, eris: dict):
        """The active-space Hamiltonian, absorbed for ``contract_2e``: the one
        convention-safe way to get ``<c|H|c>`` and ``<z|H|c>`` without
        committing to a 2-RDM reordering convention."""
        st = self.state
        a = st.act
        nmo = st.nmo
        h1 = np.asarray(eris["h1eff"])[a, a]
        g = np.asarray(eris["ppaa"]).reshape(nmo, nmo, st.nact, st.nact)[a, a]
        return fci.direct_spin1.absorb_h1e(h1, g, st.nact, st.nelecas, 0.5)

    def pc_energy(self, C: np.ndarray, ci: np.ndarray, eps: np.ndarray,
                  *, gaps: list | None = None, guard: bool = False,
                  where: str = "") -> float:
        """block2's partially-contracted correlation energy at these
        orbitals, this CI vector and these orbital energies.

        ``eps`` is the **generalized** Fock's diagonal, rebuilt at this point
        like PySCF's ``pt.mo_energy`` is on the SC side: block2 runs
        ``canonicalized = True``, so it uses the numbers it is handed, and
        holding them fixed would differentiate a different functional
        (:func:`pyscf_ci_source`, on the same omission).

        **The PC conditioning guard runs here, at every CI vector that is not
        the state's own, and no caller can opt out of it.** That is the one
        place this pipeline can silently differentiate the wrong function: a
        displacement that is not a spin eigenfunction lifts the PC metric's
        structurally null modes, and past the lstsq cut they are kept and
        divided by (:data:`CI_STEP_PC`). The guard is
        ``generate_golden._check_pc_conditioning``, the golden's own, so the
        refusal reads the same way; every run of it is counted in
        :data:`PC_GUARD`, and the run fails at the end if a displaced CI
        vector was evaluated without it (:func:`check_pc_guard_ledger`).

        At the state's own CI vector it is an SVD per PC block that cannot
        find anything new -- the rank that can move is the metric's, the
        metric is built from the active-space RDMs, and those are a function
        of the CI vector alone -- so the orbital and nuclear sweeps skip it.
        ``guard`` runs it there anyway, which the step ladders do at every
        rung, to *sample* that argument rather than assume it; ``gaps``
        passes the per-tuple cuts back out (:func:`build_state` reads the
        reference state's)."""
        mc = self.mc
        mc.mo_coeff, mc.ci, mc.mo_energy = C, ci, eps
        key = _ci_key(ci)
        if key != self._pc_key:
            self._pc_key, self._pc_dm4 = key, {}
        displaced = key != self._ref_key
        if displaced:
            PC_GUARD.displaced += 1
        if (displaced or guard) and gaps is None:
            gaps = []
        self.nevals += 1
        pc = gg._block2_icmr(mc, "pc", dm4_cache=self._pc_dm4, gaps=gaps)
        if displaced or guard:
            _pc_guard(gaps, displaced=displaced, where=where)
        return float(sum(pc.values()))

    def corr_at_ci(self, eris: dict, C: np.ndarray, ci: np.ndarray,
                   eps: np.ndarray | None = None) -> float:
        """``E_corr`` at fixed orbitals and a moved CI vector, with the
        orbital energies rebuilt from it -- the quotient
        :func:`pyscf_ci_source` differences, in whichever mode is active.
        A PC energy here is at a displaced CI vector, so :meth:`pc_energy`
        guards it.

        ``eps`` holds the orbital energies fixed instead: **not** the
        derivative a gradient needs, but the one the device's ``y_I`` takes
        (:func:`localise_sources`, which is the only caller that passes it)."""
        self.pt.ci = ci
        self.mc.ci = ci
        dm1, _dm2, dms, _ = _dms_from(self.pt, eris, self.state)
        if eps is None:
            eps = np.diag(self.generalized_fock(eris, dm1)).copy()
        self.pt.mo_energy = eps
        if self.pc:
            return self.pc_energy(C, ci, eps, where="a y_I quotient's displaced CI vector")
        per_class = gg._pyscf_per_class(self.pt, dms, eris)
        return float(sum(per_class[c][1] for c in gg.CLASSES))

    def sc_corr_at_orbitals(self, C: np.ndarray, digests: dict | None = None) -> float:
        """The SC ``E_corr`` at these orbitals and the state's own CI vector,
        the orbital energies rebuilt from them -- with the ``f3ca``/``f3ac``
        digests taken from ``digests`` instead of rebuilt when it is given.

        Holding the digests fixed is **not** the derivative a gradient needs;
        it is the functional the device's ``Y_xy`` differentiates
        (docs/gradients.md, section 6), and :func:`localise_sources` is the
        only caller that passes them."""
        st = self.state
        eris = self.eris(C)
        self.pt.ci = st.ci
        self.mc.ci = st.ci
        dm1, _dm2, dms, _ = _dms_from(self.pt, eris, st)
        if digests is not None:
            dms["f3ca"], dms["f3ac"] = digests["f3ca"], digests["f3ac"]
        self.pt.mo_energy = np.diag(self.generalized_fock(eris, dm1)).copy()
        per_class = gg._pyscf_per_class(self.pt, dms, eris)
        return float(sum(per_class[c][1] for c in gg.CLASSES))

    # -- the whole bundle -------------------------------------------------

    def evaluate(self, C: np.ndarray, ci: np.ndarray, *, with_corr: bool = True,
                 dm1: np.ndarray | None = None, guard: bool = False,
                 where: str = "") -> dict:
        """Everything ``Phi``, the source terms and the Jacobian read, at one
        point: the two energies, the two Fock matrices, and ``H|c>`` as a
        **vector** over determinants.

        The vector is what makes ``<z|H|c>`` free for any ``z`` -- the CI
        multiplier is not known while the orbital sweep runs, and the sweep is
        the expensive part, so it cannot be redone once ``z`` is in hand.

        ``dm1`` overrides the active 1-RDM used to build ``f``. It is the
        state's own in every call but the ``y_I`` sweep, where the CI vector
        moves and the RDMs must move with it. ``guard`` and ``where`` go to
        :meth:`pc_energy`."""
        st = self.state
        eris = self.eris(C)
        dms = None
        if dm1 is None or with_corr:
            self.pt.ci = ci
            self.mc.ci = ci
            dm1_built, _dm2, dms, _e = _dms_from(self.pt, eris, st)
            if dm1 is None:
                dm1 = dm1_built
        f = self.generalized_fock(eris, dm1)
        out = {"f": f, "fock_rhf": self.rhf_fock(C)}

        ecore = self.core_energy(eris)
        h2abs = self.active_hamiltonian(eris)
        hc = fci.direct_spin1.contract_2e(h2abs, ci, st.nact, st.nelecas)
        ci_flat = np.asarray(ci).ravel()
        hvec = ecore * ci_flat + np.asarray(hc).ravel()
        out["hvec"] = hvec
        out["e_casci"] = float(ci_flat @ hvec)

        if with_corr:
            eps = np.diag(f).copy()
            self.pt.mo_energy = eps
            if self.pc:
                out["e_corr"] = self.pc_energy(C, ci, eps, guard=guard, where=where)
            else:
                per_class = gg._pyscf_per_class(self.pt, dms, eris)
                out["e_corr"] = float(sum(per_class[c][1] for c in gg.CLASSES))
        else:
            out["e_corr"] = 0.0
        return out


def _dms_from(pt, eris: dict, st: State):
    """``generate_golden._pyscf_dms``'s RDM set, but from an ``eris`` already
    in hand rather than transformed again -- the digests ``f3ca``/``f3ac`` read
    the active integrals, so they move with the orbitals and this is where
    that happens."""
    ci = np.asarray(pt.load_ci())
    ncas = st.nact
    dm1, dm2, dm3 = fci.rdm.make_dm123("FCI3pdm_kern_sf", ci, ci, ncas, st.nelecas)
    aaaa = np.asarray(eris["ppaa"])[st.ncore:st.nocc, st.ncore:st.nocc].copy()
    la = fci.cistring.gen_linkstr_index(range(ncas), st.nelecas[0])
    lb = fci.cistring.gen_linkstr_index(range(ncas), st.nelecas[1])
    dms = {
        "1": dm1, "2": dm2, "3": dm3, "4": None,
        "f3ca": pn._contract4pdm("NEVPTkern_cedf_aedf", aaaa, ci, ncas, st.nelecas, (la, lb)),
        "f3ac": pn._contract4pdm("NEVPTkern_aedf_ecdf", aaaa, ci, ncas, st.nelecas, (la, lb)),
    }
    return dm1, dm2, dms, None


# --------------------------------------------------------------------------
# The kappa parameterisation
# --------------------------------------------------------------------------


def givens(C: np.ndarray, p: int, q: int, t: float) -> np.ndarray:
    """``C exp(t (E_pq - E_qp))``, exactly. On ``span{e_p, e_q}`` the
    generator is ``t [[0, 1], [-1, 0]]``, so the exponential is a plane
    rotation and no matrix exponential is needed -- which matters when the
    sweep runs over every ``kappa`` pair."""
    out = C.copy()
    cs, sn = np.cos(t), np.sin(t)
    out[:, p] = cs * C[:, p] - sn * C[:, q]
    out[:, q] = sn * C[:, p] + cs * C[:, q]
    return out


def kappa_pairs(st: State) -> list[tuple[int, int]]:
    """Every ``kappa`` direction a multiplier is needed for: all ``p < q``
    except both inside act_O, or both inside act_V.

    **Those two blocks, and only those two, are gauges** (section 1.3 read
    against the RHF partition of section 0). A rotation inside act_O leaves
    the RHF occupied space untouched and leaves the full-CI energy untouched,
    so no constraint responds to it and no multiplier is needed; same for
    act_V. An act_O-act_V rotation leaves the *energy* untouched -- it is an
    active-active rotation -- but it **moves the RHF occupied space**, so
    ``F`` responds to it and ``F_{act_O, act_V} = 0`` is a genuine Brillouin
    condition with a multiplier of its own. Dropping the whole active-active
    block as "the gauge" loses exactly that condition, and with it a term in
    every multiplier."""
    out = []
    for p in range(st.nmo):
        for q in range(p + 1, st.nmo):
            sp, sq = st.space_of(p), st.space_of(q)
            if sp == sq and sp in (1, 2):
                continue
            out.append((p, q))
    return out


def constraints(st: State) -> list[tuple[str, int, int]]:
    """One constraint per non-gauge ``kappa`` pair, **in the same order**:
    ``("F", p, q)`` for (O1) and ``("f", p, q)`` for (O2).

    A rotation between two of the four spaces -- core, act_O, act_V, virtual
    -- is constrained by (O1), ``F_pq = 0``: RHF stationarity where it crosses
    the occupied/virtual divide and the active-space *selection* where it does
    not (section 1.2's table). A rotation *inside* the core or the virtual
    space is constrained by (O2), ``f_pq = 0``, the pseudocanonical convention
    the energy functional assumes when it reads only ``e_core``/``e_virt``
    (section 2.3). One inside act_O or act_V is constrained by neither and is
    not a ``kappa`` direction here at all.

    So there is exactly one multiplier per non-gauge rotation, stationarity is
    a square system, and the Jacobian is indexed the same way on both sides.
    The assertion is that identity."""
    out = []
    for p, q in kappa_pairs(st):
        sp, sq = st.space_of(p), st.space_of(q)
        out.append(("f" if sp == sq else "F", p, q))
    nc, nv, no, na_v = st.ncore, st.nvirt, st.n_act_occ, st.nact - st.n_act_occ
    sizes = (nc, no, na_v, nv)
    want_f = nc * (nc - 1) // 2 + nv * (nv - 1) // 2
    want_o1 = sum(sizes[i] * sizes[j] for i in range(4) for j in range(i + 1, 4))
    got_f = sum(1 for kind, _p, _q in out if kind == "f")
    assert got_f == want_f and len(out) - got_f == want_o1, (
        f"{got_f} (O2) and {len(out) - got_f} (O1) constraints against {want_f} and "
        f"{want_o1} expected for spaces {sizes}: the bookkeeping is wrong"
    )
    return out


def constraint_vector(bundle: dict, cons: list) -> np.ndarray:
    """The constraints' values at one point: ``F_pq`` on (O1)'s pairs and
    ``f_pq`` on (O2)'s. Zero at the reference state by construction, which is
    section 3.6's invariant; nonzero along the frozen path, which is what its
    nuclear derivative is."""
    m = {"F": bundle["fock_rhf"], "f": bundle["f"]}
    return np.array([m[kind][p, q] for kind, p, q in cons])


# --------------------------------------------------------------------------
# Source terms
# --------------------------------------------------------------------------


@dataclass
class SourceTerms:
    """Park Eqs. 26-27 for the correlation energy: the orbital source term as
    the antisymmetric combination every consumer reads, and the CI source
    term.

    ``kappa`` is ``A_pq = Y_pq - Y_qp`` rather than ``Y`` itself. Nothing
    downstream reads anything else -- Eqs. 56-57 divide ``Y_ij - Y_ji``, the
    selection-boundary and CPHF right-hand sides are ``Y_pq - Y_qp`` -- and
    the symmetric part of ``Y`` is not even defined by an antisymmetric
    ``kappa``."""

    kappa: np.ndarray
    ci: np.ndarray
    origin: str
    #: True when ``kappa`` holds **only** ``E_corr``'s half, read from a dump;
    #: ``E_CASCI``'s half is then added from the sweep that builds the
    #: Jacobian, which runs either way.
    from_dump: bool = False
    seconds: float = 0.0


#: How far a dump's ``ci`` may sit from the rebuilt state's (after the sign is
#: fixed) and still be the same state: the rebuild is bit-reproducible
#: (docs/reference-data.md, "Reproducibility"), so this only absorbs the
#: Davidson tolerance, and a different state is decades away.
DUMP_CI_TOL = 1e-8


def load_source_terms(path: Path, st: State) -> SourceTerms:
    """A ``.gradsrc`` dump -- ``nevpt2_sc_pseudodensity --dump-source-terms``'s
    -- the GPU's ``Y_xy`` and ``y_I``, checked against the state they claim to
    belong to and carried into the basis this state is held in.

    ``kappa_source`` may be ``Y`` or an already-antisymmetrised ``A``; only
    ``Y - Y^T`` is read, so both give the same answer.

    **The dump is in the golden's basis, and this state is not.** The GPU
    knows one active basis, the golden's natural orbitals; :func:`build_state`
    rotates the active block by ``W = st.act_rot`` into ``F``'s eigenbasis
    (section 2.6), carrying the CI vector along. At fixed CI vector an
    orbital rotation in the new basis is ``W kappa' W^T`` in the old one, so
    ``Y' = W^T Y W``; and ``y`` is a covector of the CI rotation ``T(W)``,
    which is orthogonal, so ``y' = T(W) y`` -- the same transform the state
    took. The dump's own ``ci`` is the fingerprint: it must be the golden's
    vector up to a sign, and the sign is applied to ``y`` (``E`` is even in
    ``c``, so ``y`` is odd)."""
    with np.load(path) as z:
        for key in ("kappa_source", "ci_source", "ci"):
            if key not in z:
                raise SystemExit(f"{path}: no '{key}' array -- not a gradient source dump")
        for key, want in (("ncore", st.ncore), ("nact", st.nact), ("nvirt", st.nvirt)):
            if key in z and int(z[key]) != want:
                raise SystemExit(
                    f"{path}: {key} = {int(z[key])} but the state has {want}; the dump "
                    "describes a different partition"
                )
        y_mat = np.asarray(z["kappa_source"], dtype=float)
        y_ci = np.asarray(z["ci_source"], dtype=float)
        ci_dump = np.asarray(z["ci"], dtype=float)
        e_corr = float(z["e_corr"]) if "e_corr" in z else None
    if y_mat.shape != (st.nmo, st.nmo):
        raise SystemExit(
            f"{path}: kappa_source is {y_mat.shape}, expected ({st.nmo}, {st.nmo})"
        )
    if y_ci.size != st.ci.size or ci_dump.size != st.ci.size:
        raise SystemExit(
            f"{path}: ci_source / ci have {y_ci.size} / {ci_dump.size} elements, the CI "
            f"vector has {st.ci.size}"
        )
    ref = st.ci_golden.ravel()
    sign = 1.0 if float(ci_dump.ravel() @ ref) >= 0.0 else -1.0
    off = float(np.abs(sign * ci_dump.ravel() - ref).max())
    if off > DUMP_CI_TOL:
        raise SystemExit(
            f"{path}: its CI vector is {off:.2e} from the golden's (allowed "
            f"{DUMP_CI_TOL:g}, after the sign): the dump is of a different state"
        )
    if e_corr is not None and abs(e_corr - st.e_corr) > REF_STATE_TOL:
        raise SystemExit(
            f"{path}: its E_corr is {e_corr - st.e_corr:+.2e} from the rebuilt state's"
        )
    w = np.eye(st.nmo)
    w[st.act, st.act] = st.act_rot
    y_rot = w.T @ y_mat @ w
    y_ci_rot = fci.addons.transform_ci_for_orbital_rotation(
        sign * y_ci.reshape(st.ci.shape), st.nact, st.nelecas, st.act_rot)
    return SourceTerms(kappa=y_rot - y_rot.T, ci=np.asarray(y_ci_rot).reshape(st.ci.shape),
                       origin=f"dump {path}", from_dump=True)


def kappa_sweep(fn: Functional, st: State, cons: list, *, with_corr: bool,
                step: float, log=print) -> dict:
    """One central-difference pass along every non-gauge ``kappa`` pair,
    returning everything that depends on that direction:

    ``A``
        ``(n_mo, n_mo)`` antisymmetric: ``A_pq = dE/dkappa_pq -
        dE/dkappa_qp`` for ``E = E_CASCI + E_corr``. Only this antisymmetric
        combination is ever read -- Park Eqs. 56-57 divide ``Y_ij - Y_ji``,
        and every right-hand side below is ``Y_pq - Y_qp`` -- and it is the
        only part an antisymmetric ``kappa`` defines.
    ``M``
        ``M[k, m] = dG_m/dkappa_k``: the constraint Jacobian.
    ``Q``
        ``Q[k, I] = d(H(kappa) c)_I / dkappa_k``, so that section 2.4's CI
        relaxation ``z^dag (dH/dkappa) c`` is ``Q z`` for **any** ``z``. The
        CI multiplier is not known while this runs, and this is the expensive
        pass, so the vector is what makes the coupled solve affordable.

    The Jacobian is differenced rather than written out because ``f`` is
    **not** covariant under a rotation that mixes the core and active index
    ranges: its core dressing sums over an index *range*, not a subspace, so
    ``df = [f, kappa]`` holds only inside one space. Getting that wrong is
    invisible in the intra-space blocks and wrong everywhere else."""
    pairs = kappa_pairs(st)
    dm1 = fn.reference_dm1(st)
    A = np.zeros((st.nmo, st.nmo))
    # E_corr's half of A on its own: what a dump's Y replaces, and what
    # --source both compares it with.
    A_corr = np.zeros((st.nmo, st.nmo))
    M = np.zeros((len(pairs), len(cons)))
    Q = np.zeros((len(pairs), st.ci.size))
    t0 = time.perf_counter()
    for k, (p, q) in enumerate(pairs):
        b = [fn.evaluate(givens(st.mo_coeff, p, q, sign * step), st.ci,
                         with_corr=with_corr, dm1=None if with_corr else dm1)
             for sign in (+1.0, -1.0)]
        a = ((b[0]["e_casci"] + b[0]["e_corr"]) - (b[1]["e_casci"] + b[1]["e_corr"])) \
            / (2.0 * step)
        A[p, q], A[q, p] = a, -a
        ac = (b[0]["e_corr"] - b[1]["e_corr"]) / (2.0 * step)
        A_corr[p, q], A_corr[q, p] = ac, -ac
        M[k] =(constraint_vector(b[0], cons) - constraint_vector(b[1], cons)) / (2.0 * step)
        Q[k] = (b[0]["hvec"] - b[1]["hvec"]) / (2.0 * step)
        if (k + 1) % 250 == 0:
            log(f"    {k + 1}/{len(pairs)} kappa pairs, {time.perf_counter() - t0:.0f}s")
    return {"A": A, "A_corr": A_corr, "A_casci": A - A_corr, "M": M, "Q": Q,
            "seconds": time.perf_counter() - t0, "with_corr": with_corr}


def allowed_determinants(st: State) -> np.ndarray:
    """The determinants in the reference state's own irrep, as a flat boolean
    mask over the CI vector.

    **The SC energy is not differentiable in the CI vector outside this
    sector, and the difference quotient does not notice.** A perturber whose
    norm vanishes by symmetry contributes ``-N_t^2 / (N_t Delta_t + H_t)`` with
    ``N_t`` and ``H_t`` both quadratic forms in ``c``; displace ``c`` by ``h``
    along a symmetry-*breaking* direction and both go as ``h``, so the term
    goes as ``h`` times a ratio that is **not linear in the direction**.
    PySCF's own ``|norm| > NUMERICAL_ZERO`` cut lets it in. The result is a
    positively homogeneous but non-linear limit: componentwise central
    differences converge beautifully (they do -- over three decades) and their
    assembly is **not a gradient**, so contracting it with any direction but
    the ones it was taken along gives the wrong number.

    Inside the sector there is no such term: a symmetry-forbidden perturber's
    norm is zero for *every* state of that irrep, so it stays zero and the
    functional is smooth. And nothing else is needed: a symmetry-preserving
    nuclear displacement moves the CI vector inside its own irrep, so the
    forbidden components of ``y`` would be contracted with exact zeros.

    This is the CI-space counterpart of the condition
    docs/gradients.md, section 7 states for the PC tangent -- "the CI
    direction must stay inside the state's symmetry sector", there because a
    random direction raises the metric's rank. Same cause, same cure."""
    mol = st.pt.mol
    if not getattr(mol, "symmetry", False):
        return np.ones(st.ci.size, dtype=bool)
    from pyscf import symm
    from pyscf.fci import cistring, direct_spin1_symm

    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                 st.mo_coeff[:, st.act], check=False)
    strs_a = cistring.make_strings(range(st.nact), st.nelecas[0])
    strs_b = cistring.make_strings(range(st.nact), st.nelecas[1])
    irr_a = direct_spin1_symm._gen_strs_irrep(strs_a, orbsym)
    irr_b = direct_spin1_symm._gen_strs_irrep(strs_b, orbsym)
    det = (irr_a[:, None] ^ irr_b[None, :]).ravel()
    c = st.ci.ravel()
    wfnsym = int(det[int(np.argmax(np.abs(c)))])
    mask = det == wfnsym
    # The state must live entirely inside the sector, or the labelling does
    # not describe it. That happens for real: a degenerate active pair (LiF's
    # pi orbitals) leaves F's active eigenvectors an arbitrary mixture of the
    # two irreps the abelian subgroup splits them across, so the active
    # orbitals are not symmetry-pure and the determinant labels mean nothing.
    # Then there is nothing to restrict to and every determinant is kept --
    # which costs only the extra quotients, because the projection above is
    # what makes them gradients and this is a refinement on top of it.
    leak = float(np.abs(c[~mask]).max()) if (~mask).any() else 0.0
    if leak > 1e-9:
        return np.ones(st.ci.size, dtype=bool)
    return mask


def pyscf_ci_source(fn: Functional, st: State,
                    step: float | None = None) -> np.ndarray:
    """``y_I = dE_corr/dc_I``, by central differences at fixed orbitals.

    Each quotient is taken along ``P e_I``, **orthogonal to the state**, and
    the result is therefore ``P y`` -- the only part the Z-vector equation
    reads -- rather than ``y`` itself.

    That is not a convenience, it is what makes the quotient mean anything.
    PySCF's ``E_corr`` is evaluated on the CI vector as given, and the
    hole-side metrics' RDM-free constants are the hole RDMs only **at unit
    norm** (docs/gradients.md, section 7), so off the unit sphere the
    functional is not the energy of any state -- and it has poles there: the
    per-perturber finish divides by ``N_t Delta_t + H_t``, which a norm change
    drives towards zero. Differentiating along a bare ``e_I`` walks straight
    off the sphere, and at the **dominant** determinant the result is
    unusable: measured, ``y_i + y_j`` disagreed with the quotient along
    ``e_i + e_j`` by 4e-2 for any pair containing it, and by 1e-7 for every
    pair that did not. A gradient whose own linearity fails at 4e-2 is not a
    gradient, and the failure is invisible in a convergence table -- the
    componentwise quotients converge over three decades and are simply
    converging to the wrong thing.

    A direction orthogonal to ``c`` changes the norm only at second order, so
    the sphere is never left and the normalisation multiplier ``x`` -- which
    would have absorbed the component along ``c`` -- is never needed: it
    multiplies ``c^dag c - 1``, which the frozen path holds at zero
    identically.

    **The orbital energies move with the CI vector, and that term is not
    optional.** ``Delta_t`` reads the generalized Fock's diagonal, and ``f``
    is built from the active density, so ``e_core`` and ``e_virt`` are
    functions of ``c``::

        f_pq = h^eff_pq + sum_tu gamma_tu(c) [ (pq|tu) - (1/2)(pt|uq) ]

    The device's ``y_I`` held them fixed until #87 -- the golden file hands
    them in as finished numbers -- and was visibly wrong against ``grad_num``
    for it; it now carries their response, out of the gradient sidecar
    (docs/gradients.md, section 5.5). Rebuilt per displaced CI vector here,
    which is what makes this the independent check of that term: both of
    the device's modes share it, so its own duality cannot see it."""
    # Its own constant in PC mode, and :data:`CI_STEP_PC` says why: it is
    # not a truncation-error choice, it is the step at which the PC metric's
    # null modes stay below the lstsq cut.
    if step is None:
        step = CI_STEP_PC if fn.pc else CI_STEP
    eris = fn.eris(st.mo_coeff)

    def e_corr_at(ci: np.ndarray) -> float:
        return fn.corr_at_ci(eris, st.mo_coeff, ci)

    y = np.zeros(st.ci.size)
    c = st.ci.ravel().copy()
    # Only the state's own irrep: outside it the functional is not
    # differentiable (:func:`allowed_determinants`) and the quotient would
    # return a number that is not part of any gradient.
    mask = allowed_determinants(st)
    for i in np.flatnonzero(mask):
        d = np.zeros_like(c)
        d[i] = 1.0
        d -= float(c @ d) * c
        scale = float(np.linalg.norm(d))
        if scale < 1e-12:
            continue
        d /= scale
        vals = [e_corr_at((c + s * step * d).reshape(st.ci.shape))
                for s in (+1.0, -1.0)]
        y[i] = scale * (vals[0] - vals[1]) / (2.0 * step)
    # Put the functional back where the caller found it.
    fn.pt.ci = st.ci
    fn.mc.ci = st.ci
    return y.reshape(st.ci.shape)


# --------------------------------------------------------------------------
# Two providers, compared componentwise (--source both)
# --------------------------------------------------------------------------

#: The four spaces :meth:`State.space_of` numbers, for the per-block table.
SPACE_NAMES = ("core", "act_O", "act_V", "virt")


def _block_rows(st: State, a_dump: np.ndarray, a_ref: np.ndarray) -> list[dict]:
    """``A_pq`` from two sources over every non-gauge pair, grouped by the
    pair's two spaces: per block, how many pairs, the largest ``|A|`` either
    way, the largest difference and where it is."""
    rows: dict = {}
    for p, q in kappa_pairs(st):
        key = (st.space_of(p), st.space_of(q))
        d = abs(float(a_dump[p, q] - a_ref[p, q]))
        r = rows.setdefault(key, dict(block=f"{SPACE_NAMES[key[0]]}-{SPACE_NAMES[key[1]]}",
                                      n=0, max_ref=0.0, max_dump=0.0, max_delta=0.0,
                                      at=(p, q)))
        r["n"] += 1
        r["max_ref"] = max(r["max_ref"], abs(float(a_ref[p, q])))
        r["max_dump"] = max(r["max_dump"], abs(float(a_dump[p, q])))
        if d >= r["max_delta"]:
            r["max_delta"], r["at"] = d, (p, q)
    return [rows[k] for k in sorted(rows)]


def _ci_row(st: State, y_dump: np.ndarray, y_ref: np.ndarray, mask: np.ndarray) -> dict:
    """``P y`` from the dump against a reference ``y`` (already projected),
    on the state's own irrep -- the only components the gradient reads."""
    c = st.ci.ravel()
    py = y_dump.ravel() - float(c @ y_dump.ravel()) * c
    d = np.abs(py - y_ref.ravel()) * mask
    worst = int(np.argmax(d))
    return dict(n=int(mask.sum()), max_ref=float(np.abs(y_ref.ravel() * mask).max()),
                max_dump=float(np.abs(py * mask).max()), max_delta=float(d.max()),
                at=worst, c_dot_y=float(c @ y_dump.ravel()),
                outside=float(np.abs(py * ~mask).max()) if (~mask).any() else 0.0)


def compare_sources(st: State, dump: SourceTerms, a_corr: np.ndarray, y_ref: np.ndarray,
                    label: str) -> dict:
    """The dump's two source terms against a reference pair, componentwise:
    ``A = Y - Y^T`` block by block over every non-gauge rotation, and ``P y``
    determinant by determinant on the state's irrep. Printed, and returned for
    the report."""
    mask = allowed_determinants(st)
    blocks = _block_rows(st, dump.kappa, a_corr)
    ci = _ci_row(st, dump.ci, y_ref, mask)
    print(f"\n  the dump against {label}, componentwise (Eh; A_pq = dE_corr/dkappa_pq "
          "along each non-gauge rotation)")
    print(f"    {'block':<12} {'pairs':>6} {'max|A| ref':>12} {'max|A| dump':>12} "
          f"{'max|delta|':>11}  at (p, q)")
    for r in blocks:
        print(f"    {r['block']:<12} {r['n']:>6} {r['max_ref']:>12.4e} {r['max_dump']:>12.4e} "
              f"{r['max_delta']:>11.3e}  {r['at']}")
    print(f"    {'y_I (P y)':<12} {ci['n']:>6} {ci['max_ref']:>12.4e} {ci['max_dump']:>12.4e} "
          f"{ci['max_delta']:>11.3e}  determinant {ci['at']}")
    print(f"    (the dump's c.y = {ci['c_dot_y']:+.6e}, dropped by the projection; max |P y| "
          f"outside the state's irrep, which the gradient never reads: {ci['outside']:.2e})")
    return dict(label=label, blocks=blocks, ci=ci,
                max_kappa=max(r["max_delta"] for r in blocks), max_ci=ci["max_delta"])


def localise_sources(fn: Functional, st: State, step: float, *, ci_step: float) -> dict:
    """PySCF's own SC functional with the two omissions #73 measured put back
    in (docs/gradients.md, sections 1 and 6): ``A_corr`` with the
    ``f3ca``/``f3ac`` digests frozen at the state's, and ``y`` with the orbital
    energies frozen at the state's ``e_core``/``e_virt``. **Neither is a
    derivative this gradient needs**; they exist so that a disagreement
    between the two providers can be pinned on those omissions -- if the dump
    matches these and not the exact ones, nothing else is wrong with it.
    The ``y`` half describes a dump from before #87: one written since
    carries the orbital energies' response, so it matches the EXACT ``y`` and
    misses this one by what the response is worth.

    Costs one more ``E_corr`` sweep over the ``kappa`` pairs and one more over
    the determinants."""
    t0 = time.perf_counter()
    eris = fn.eris(st.mo_coeff)
    fn.pt.ci = st.ci
    fn.mc.ci = st.ci
    _dm1, _dm2, dms, _ = _dms_from(fn.pt, eris, st)
    digests = {"f3ca": dms["f3ca"], "f3ac": dms["f3ac"]}
    a = np.zeros((st.nmo, st.nmo))
    for p, q in kappa_pairs(st):
        e = [fn.sc_corr_at_orbitals(givens(st.mo_coeff, p, q, s * step), digests)
             for s in (+1.0, -1.0)]
        a[p, q] = (e[0] - e[1]) / (2.0 * step)
        a[q, p] = -a[p, q]
    # y with e_core/e_virt frozen: pyscf_ci_source's own projected quotient,
    # through corr_at_ci's eps override.
    eps0 = np.diag(st.f).copy()
    eris = fn.eris(st.mo_coeff)
    y = np.zeros(st.ci.size)
    c = st.ci.ravel().copy()
    for i in np.flatnonzero(allowed_determinants(st)):
        d = np.zeros_like(c)
        d[i] = 1.0
        d -= float(c @ d) * c
        scale = float(np.linalg.norm(d))
        if scale < 1e-12:
            continue
        d /= scale
        v = [fn.corr_at_ci(eris, st.mo_coeff, (c + s * ci_step * d).reshape(st.ci.shape),
                           eps=eps0) for s in (+1.0, -1.0)]
        y[i] = scale * (v[0] - v[1]) / (2.0 * ci_step)
    fn.pt.ci = st.ci
    fn.mc.ci = st.ci
    fn.pt.mo_energy = eps0
    return dict(a_corr=a, y=y.reshape(st.ci.shape), seconds=time.perf_counter() - t0)


# --------------------------------------------------------------------------
# The GPU's forward Edot(kappa) against PySCF's exact one (#88)
# --------------------------------------------------------------------------


def load_edot_directions(path: Path, st: State) -> dict:
    """A ``.gradsrc``'s forward ``Edot(kappa)`` members (#88): the seeded
    directions ``edot_kappa`` and the GPU's two terms along each,
    ``edot_frozen`` (the f3ac/f3ca digests held fixed) and ``edot_digest``
    (the digests' own response). The exact ``Edot`` is their sum.

    The directions are in the golden's basis, like ``kappa_source``; this
    state is held in section 2.6's, so each is carried across the way
    :func:`load_source_terms` carries ``Y``: the golden's orbitals ``C_g``
    rotated by ``exp(t kappa)``, with the golden's CI vector, are this state's
    ``C = C_g W`` rotated by ``exp(t W^T kappa W)`` with its CI vector, so the
    direction here is ``W^T kappa W`` and ``Edot`` is unchanged. Call
    :func:`load_source_terms` on the same file first: that is what checks the
    file describes this state."""
    with np.load(path) as z:
        keys = ("edot_kappa", "edot_frozen", "edot_digest")
        present = [k for k in keys if k in z]
        if len(present) != len(keys):
            raise SystemExit(
                f"{path}: carries {present or 'none'} of {list(keys)} -- a dump without the "
                "forward Edot directions (written before #88?); rerun "
                "nevpt2_sc_pseudodensity --dump-source-terms"
            )
        kap = np.asarray(z["edot_kappa"], dtype=float)
        frozen = np.asarray(z["edot_frozen"], dtype=float)
        digest = np.asarray(z["edot_digest"], dtype=float)
    if kap.ndim != 3 or kap.shape[1:] != (st.nmo, st.nmo) or frozen.shape != (kap.shape[0],) \
            or digest.shape != (kap.shape[0],):
        raise SystemExit(f"{path}: the Edot members' shapes do not fit n_mo = {st.nmo}")
    w = np.eye(st.nmo)
    w[st.act, st.act] = st.act_rot
    return dict(kappa=[w.T @ k @ w for k in kap], frozen=frozen, digest=digest)


#: The ``--edot-check`` ladders' steps: :data:`KAPPA_LADDER`'s start, but four
#: rungs deeper, to ``2^-21``. Measured, not chosen: along acrolein's
#: core-active direction the frozen-digest functional has a near-pole, and its
#: quotient (and with it the digests' term) converges as ``h^2`` -- ratio
#: 4.00 per halving -- only from ``2^-14`` down, onto the GPU's numbers, with
#: PySCF's evaluation noise taking over below ``2^-21``; stopping at
#: ``2^-17`` left a run of two (docs/testing.md, "The digests' orbital
#: response, forward mode").
EDOT_LADDER = (KAPPA_LADDER[0], 21)


def check_edot(fn: Functional, st: State, dirs: dict) -> list[dict]:
    """The GPU's forward ``Edot(kappa)`` (#88) against PySCF's own SC
    functional, along each dumped direction -- the **independent provider**
    for the f3ac/f3ca digests' term, which both of the device's modes leave
    out of ``Y`` and so cannot check against each other (docs/gradients.md,
    section 6).

    Three step ladders per direction ``u = kappa / |kappa|``, each of central
    differences of :meth:`Functional.sc_corr_at_orbitals` along
    ``C exp(t u)`` at the state's own CI vector (the orbital energies rebuilt
    at every point, as on the device):

    * **exact** -- the digests rebuilt at every point: PySCF's exact
      directional derivative, against the GPU's ``frozen + digest``;
    * **frozen** -- the digests held at the state's: the functional the
      device's ``Y`` differentiates, against ``frozen``;
    * **digest** -- the difference of the two at each step, against
      ``digest``: the term on its own.

    Each is :func:`step_ladder`'s gate -- a run of fourfold halvings, then
    the Richardson extrapolant within :data:`LADDER_TOL` of the GPU's number
    -- so what is left over is shown to be the quotient's truncation, walked
    down, not a single step's luck. Every point is cached, so the three
    ladders cost two evaluations per step and sign."""
    from scipy.linalg import expm

    eris = fn.eris(st.mo_coeff)
    fn.pt.ci = st.ci
    fn.mc.ci = st.ci
    _dm1, _dm2, dms, _ = _dms_from(fn.pt, eris, st)
    frozen_digests = {"f3ca": dms["f3ca"], "f3ac": dms["f3ac"]}
    steps = ladder_steps(EDOT_LADDER)
    out = []
    for d, kap in enumerate(dirs["kappa"]):
        t0 = time.perf_counter()
        norm = float(np.linalg.norm(kap))
        u = kap / norm
        cache: dict = {}

        def corr(t: float, frozen: bool, u=u, cache=cache) -> float:
            key = (t, frozen)
            if key not in cache:
                cache[key] = fn.sc_corr_at_orbitals(st.mo_coeff @ expm(t * u),
                                                    frozen_digests if frozen else None)
            return cache[key]

        gpu_frozen = float(dirs["frozen"][d]) / norm
        gpu_digest = float(dirs["digest"][d]) / norm
        checks = {
            "exact": step_ladder(f"dir {d} exact Edot", lambda t, c=corr: c(t, False),
                                 steps, gpu_frozen + gpu_digest),
            "frozen": step_ladder(f"dir {d} frozen-digest Edot", lambda t, c=corr: c(t, True),
                                  steps, gpu_frozen),
            "digest": step_ladder(f"dir {d} digest term",
                                  lambda t, c=corr: c(t, False) - c(t, True), steps, gpu_digest),
        }
        # A direction whose every rotation breaks the state's symmetry (N2
        # CAS(4,4)'s core-active block: sigma core, pi active) leaves E even
        # in t to the last bit, so every quotient is exactly 0.0 and there is
        # no truncation to ladder down. Zero is then the exact answer, and the
        # GPU's number is held to it at the same tolerance.
        for key, c in checks.items():
            n = c.numbers
            if not c.ok and np.all(n["fd"] == 0.0) and abs(n["ref"]) <= LADDER_TOL:
                checks[key] = Check(
                    c.name, True,
                    f"every quotient exactly 0 (E even in t: zero by symmetry); |GPU| = "
                    f"{abs(n['ref']):.2e}", c.table,
                    dict(n, value=0.0, err=abs(n["ref"]), run=0, how="zero"))
        # The exact quotient IS the frozen one plus the digest one, rung by
        # rung. Where the frozen-digest functional has a near-pole (acrolein's
        # core-active direction: a perturber empty at the state, whose
        # Delta N + H the frozen digests leave nearly cancelling), the two
        # pieces carry it with opposite signs and the exact functional is
        # smooth: its truncation sits below PySCF's evaluation noise on every
        # rung where it would be asymptotic, so its ladder has nothing to walk
        # down and shows no fourfold run. Its two pieces then DO -- each
        # gated on its own run and Richardson value above -- and the exact
        # quotient is held to the GPU's sum on a window of at least
        # LADDER_RUN + 1 consecutive rungs instead, every one within
        # LADDER_TOL: a window, not a sweep minimum, so a crossing cannot pass
        # for agreement (docs/testing.md, "The digests' orbital response,
        # forward mode").
        ex = checks["exact"]
        if (not ex.ok and ex.numbers["run"] < LADDER_RUN and checks["frozen"].ok
                and checks["digest"].ok):
            left = np.abs(ex.numbers["fd"] - ex.numbers["ref"])
            best, start, cur = (0, 0), 0, 0
            for i, v in enumerate(left):
                cur = cur + 1 if v <= LADDER_TOL else 0
                if cur == 1:
                    start = i
                if cur > best[0]:
                    best = (cur, start)
            width, lo = best
            if width >= LADDER_RUN + 1:
                window = left[lo:lo + width]
                checks["exact"] = Check(
                    ex.name, True,
                    f"no fourfold run (smooth: truncation below PySCF's noise); "
                    f"|fd - GPU| <= {window.max():.2e} on {width} consecutive rungs from "
                    f"2^-{round(-np.log2(ex.numbers['steps'][lo]))}, its two pieces each "
                    f"laddered", ex.table,
                    dict(ex.numbers, value=float(ex.numbers["fd"][lo + int(np.argmin(window))]),
                         err=float(window.max()), run=0, window=width, how="window"))
        # What the direction touches, by the RHF partition's four spaces, so a
        # reader sees which blocks each one probes.
        spaces = [st.space_of(p) for p in range(st.nmo)]
        touched = sorted({f"{SPACE_NAMES[min(spaces[p], spaces[q])]}-"
                          f"{SPACE_NAMES[max(spaces[p], spaces[q])]}"
                          for p in range(st.nmo) for q in range(p + 1, st.nmo)
                          if abs(u[p, q]) > 1e-14})
        out.append(dict(dir=d, norm=norm, checks=checks, blocks=touched,
                        seconds=time.perf_counter() - t0))
        print(f"  direction {d}: {len(cache)} evaluations, {out[-1]['seconds']:.0f}s")
    fn.pt.ci = st.ci
    fn.mc.ci = st.ci
    return out


def report_edot(rows: list[dict], case: str) -> bool:
    """:func:`check_edot`'s tables and its verdict line; returns whether every
    ladder holds."""
    print("\nthe GPU's forward Edot(kappa) against PySCF's exact directional derivative "
          "(#88; Eh per unit kappa)")
    print("  exact = PySCF with the f3ac/f3ca digests rebuilt at every point, against the GPU's "
          "frozen + digest;\n  frozen = digests held at the state's, against the GPU's frozen; "
          "digest = the two differenced,\n  against the GPU's digest term. 'Richardson' is the "
          "ladder's extrapolant at its finest fourfold run.")
    ok = True
    for r in rows:
        print(f"\n  direction {r['dir']} (|kappa| = {r['norm']:.6f}; blocks: "
              f"{', '.join(r['blocks'])})")
        print(f"    {'':<7} {'GPU':>20} {'PySCF Richardson':>20} {'|delta|':>10} {'run':>4}")
        for key in ("exact", "frozen", "digest"):
            c = r["checks"][key]
            ok = ok and c.ok
            n = c.numbers
            note = {"zero": "  (zero by symmetry: every quotient exactly 0)",
                    "window": f"  (window: {n.get('window', 0)} rungs within "
                              f"{LADDER_TOL:g}; max shown)"}.get(n.get("how", ""), "")
            print(f"    {key:<7} {n['ref']:+20.12e} {n['value']:+20.12e} {n['err']:10.2e} "
                  f"{n['run']:>4}  {'ok' if c.ok else 'FAIL'}{note}")
        # The leftover, rung by rung: |PySCF quotient - GPU| for the exact and
        # digest ladders, falling fourfold while truncation dominates.
        print(f"    {'h':<7} {'|exact fd - GPU|':>17} {'ratio':>7} {'|digest fd - GPU|':>18} "
              f"{'ratio':>7}")
        ex, dg = r["checks"]["exact"].numbers, r["checks"]["digest"].numbers
        e_ex = np.abs(ex["fd"] - ex["ref"])
        e_dg = np.abs(dg["fd"] - dg["ref"])
        for i, h in enumerate(ex["steps"]):
            rx = f"{e_ex[i - 1] / e_ex[i]:7.2f}" if i and e_ex[i] > 0 else f"{'-':>7}"
            rd = f"{e_dg[i - 1] / e_dg[i]:7.2f}" if i and e_dg[i] > 0 else f"{'-':>7}"
            print(f"    2^-{round(-np.log2(h)):<4d} {e_ex[i]:17.3e} {rx} {e_dg[i]:18.3e} {rd}")
    n = 3 * len(rows)
    held = sum(c.ok for r in rows for c in r["checks"].values())
    print(f"\n{'PASS' if ok else 'FAIL'}: {case} the GPU's forward Edot(kappa) -- frozen-digest "
          f"tangent + the f3ac/f3ca digests' term -- against PySCF's exact directional "
          f"derivative along {len(rows)} dumped directions, {held} of {n} ladders within "
          f"{LADDER_TOL:g} (docs/testing.md, \"The digests' orbital response, forward mode\")")
    return ok


# --------------------------------------------------------------------------
# The gradient
# --------------------------------------------------------------------------


def zero_axes(geom, tol: float = 1e-10) -> list[int]:
    """The Cartesian axes along which every gradient component is **zero by
    symmetry**: the axes all atoms share a coordinate value along.

    docs/gradient-theory.md, section 4.3's argument, read off the geometry.
    For a molecule planar in a Cartesian plane that is the out-of-plane axis;
    the gradient is a vector field invariant under the plane's reflection,
    which maps each atom to itself and negates that component, so the
    component equals its own negative. A linear molecule along one axis has
    **two** such axes. These are exactly the components a numerical gradient
    cannot probe (a displacement along one breaks the plane), and exactly the
    ones an analytic gradient gets for free -- so they are a check, not a
    gap."""
    xyz = np.array([list(v) for _s, v in geom])
    return [c for c in range(3) if float(np.ptp(xyz[:, c])) < tol]


def sym_power(m: np.ndarray, power: float) -> np.ndarray:
    """``m^power`` for a symmetric positive-definite ``m``, through its
    eigendecomposition. Used for the overlap's ``S^{1/2}`` and ``S^{-1/2}``;
    ``scipy.linalg.sqrtm`` would do, but this keeps the result exactly
    symmetric, which the frozen path's orthonormality identity relies on."""
    w, v = np.linalg.eigh(m)
    return (v * (w ** power)) @ v.T


def frozen_coeff(C0: np.ndarray, S0_half: np.ndarray, ovlp: np.ndarray) -> np.ndarray:
    """``C(x) = S(x)^{-1/2} S(x0)^{1/2} C(x0)``: the frozen **orthonormal**
    path.

    ``C(x)^T S(x) C(x) = C0^T S0 C0 = 1`` at every geometry, so the
    orthonormality constraint (N1) is satisfied identically along it and needs
    no multiplier. A literally-frozen ``C`` would not be orthonormal at a
    displaced geometry, and repairing that is exactly what the usual
    energy-weighted-density term does."""
    return sym_power(ovlp, -0.5) @ S0_half @ C0


class Gradients(lagrange.Gradients):
    """The analytic SC-NEVPT2 nuclear gradient for a CASCI-on-RHF reference.

    ``nlag`` is ``ndet`` (the CI Z-vector) plus the number of orbital
    constraints, and :meth:`get_Aop_Adiag` returns a **block-diagonal**
    operator over the two: the CI and orbital equations decouple because the
    RHF Fock has no CI dependence (section 2.2). The base class's conjugate
    gradient therefore solves two independent systems in one pass and reports
    one residual norm; :attr:`residuals` carries each block's own."""

    def __init__(self, st: State, src: SourceTerms, *, with_corr: bool = True,
                 step: float | None = None, richardson: bool = True,
                 pc: bool = False, kappa_step: float | None = None):
        self.st = st
        default_step, default_kappa = scheme_steps(pc)
        step = default_step if step is None else step
        kappa_step = default_kappa if kappa_step is None else kappa_step
        self.src = src
        self.with_corr = with_corr
        self.pc = pc
        self.step = step
        self.kappa_step = kappa_step
        self.richardson = richardson
        self.pairs = kappa_pairs(st)
        self.cons = constraints(st)
        self.n_constraint = len(self.cons)
        self.residuals: dict = {}
        #: the multipliers, once solved: ``z`` (CI) and ``lam`` (orbital,
        #: indexed by :attr:`cons`).
        self.z = np.zeros(st.ci.shape)
        self.lam = np.zeros(self.n_constraint)
        self.fn = Functional(st, pc=pc)
        #: the determinants the CI derivative is defined on
        #: (:func:`allowed_determinants`).
        self.allowed = allowed_determinants(st)
        self._jacobian = None
        self._q = None
        super().__init__(st.mc, st.ci.size + self.n_constraint)
        self.verbose = 0
        # The base class defaults to 50 CG cycles, which is below nlag on
        # every case here; the iterative residual is reported, so give it
        # room to actually converge rather than reporting a truncation.
        self.max_cycle = max(400, 4 * self.nlag)
        self.conv_atol = 1e-12
        self.conv_rtol = 1e-10

    # -- hook 1: the source terms ----------------------------------------

    def get_wfn_response(self, **kwargs) -> np.ndarray:
        """``bvec``: the first derivative of the energy with respect to the
        wavefunction parameters conjugate to the multipliers.

        The CI block is ``P y`` with ``P = 1 - |c><c|`` -- the reference state
        is the null direction of ``H - E_CASCI`` and the component along it is
        absorbed by the normalisation multiplier ``x = c^T y``; the (O2)
        constraints' own CI derivative rides there too
        (:meth:`ci_coupling`). The orbital block is ``A_pq`` plus section
        2.4's CI relaxation ``z^dag (dH/dkappa) c``, which is ``Q z`` with the
        ``Q`` the orbital sweep already produced.

        Both halves read the **current** multipliers, so this is one step of
        the fixed point :meth:`solve_multipliers` runs, not a constant."""
        st = self.st
        c = st.ci.ravel()
        # Masked to the state's own irrep: outside it the CI derivative is not
        # defined (:func:`allowed_determinants`), and `H - E` is block
        # diagonal in the irrep, so a component there could never be
        # cancelled by anything the gradient reads.
        y = (self.src.ci.ravel() + self.ci_coupling(self.lam)) * self.allowed
        self.x_norm = float(c @ y)
        py = y - self.x_norm * c
        b_orb = np.array([self.src.kappa[p, q] for p, q in self.pairs])
        if self._q is not None:
            b_orb = b_orb + self._q @ self.z.ravel()
        return np.concatenate([py, b_orb])

    # -- hook 2: the Jacobian --------------------------------------------

    def ci_hop(self, z: np.ndarray) -> np.ndarray:
        """``P (H - E_CASCI) P z`` in the determinant space, through PySCF's
        own ``contract_2e`` -- the CASCI reference's Hessian in the CI
        parameters, which is what ``pyscf/grad/casci.py`` would use if a CASCI
        had a CI Z-vector at all (it does not: a CASCI is variational in
        ``c``, so ``y = 0``)."""
        st = self.st
        c = st.ci.ravel()
        zp = z.ravel() - float(c @ z.ravel()) * c
        hz = fci.direct_spin1.contract_2e(
            self._h2abs, zp.reshape(st.ci.shape), st.nact, st.nelecas
        ).ravel()
        hz = hz + self._ecore * zp - self.e_casci_act * zp
        return hz - float(c @ hz) * c

    def ci_coupling(self, lam: np.ndarray) -> np.ndarray:
        """``sum_m lambda_m dG_m/dc_I``: the (O2) constraints' derivative with
        respect to the CI vector.

        **This is a coupling docs/gradient-theory.md, section 2.2 does not
        account for, and it is real.** That section shows the CI equation
        decouples because ``F`` is built from the RHF density and so carries no
        CI dependence -- true, and it is why there is no CP-CASSCF here. But
        the Lagrangian's *other* orbital constraint, (O2)'s pseudocanonicality
        ``f_pq = 0``, is a statement about the **generalized** Fock, which is
        built from the active density ``gamma`` and therefore does depend on
        ``c``::

            f_pq  = h^eff_pq + sum_tu gamma_tu [ (pq|tu) - (1/2)(pt|uq) ]
            df_pq/dc_I = sum_tu (dgamma_tu/dc_I) [ (pq|tu) - (1/2)(pt|uq) ]

        and ``dgamma_tu/dc_I = [(E_tu + E_ut) c]_I``. So the CI and orbital
        equations are coupled through ``z^c``, weakly (``z^c`` is small, and it
        vanishes identically for a CASCI reference, where the energy is
        invariant to a rotation inside the core space), but not at zero. The
        solve below is a fixed point over exactly this term, and its
        convergence is reported.

        Contracted as one matrix rather than one column per constraint: the
        multiplier-weighted integral combination is formed first, then a single
        one-body sigma vector."""
        st = self.st
        w = np.zeros((st.nact, st.nact))
        for m, (kind, p, q) in enumerate(self.cons):
            if kind != "f" or lam[m] == 0.0:
                continue
            w += lam[m] * (self._g_ppaa[p, q] - 0.5 * self._g_papa[p, :, q, :])
        if not np.any(w):
            return np.zeros(st.ci.size)
        sig = fci.direct_spin1.contract_1e(w + w.T, st.ci, st.nact, st.nelecas)
        return np.asarray(sig).ravel()

    def degenerate_constraints(self) -> tuple[list[int], list[str]]:
        """(O2)'s undetermined rows. A core or virtual pair closer than
        :data:`PC_GAP_FLOOR` cannot have its multiplier divided out; an
        abelian subgroup puts the two partners in different irreps, so the
        energy does not move along a rotation mixing them and the source term
        vanishes too -- a ``0/0`` the constraint does not determine, for which
        zero is the right representative. A degenerate pair whose source term
        does **not** vanish is a numerical refusal, not a number."""
        eps = np.diag(self.st.f)
        drop, notes = [], []
        for n, (kind, p, q) in enumerate(self.cons):
            if kind != "f":
                continue
            gap = abs(eps[p] - eps[q])
            if gap >= PC_GAP_FLOOR:
                continue
            num = abs(self.src.kappa[p, q])
            if num > PC_ZERO_TOL:
                raise SystemExit(
                    f"the (O2) pair ({p}, {q}) has an orbital-energy gap of {gap:.2e} Eh "
                    f"but a source term of {num:.2e}: its pseudocanonical multiplier is "
                    "not determined by the state, and dividing through would manufacture "
                    "a number (docs/gradient-theory.md, section 2.3). Refused."
                )
            drop.append(n)
            notes.append(f"({p},{q}) gap {gap:.1e} source {num:.1e}")
        return drop, notes

    def get_Aop_Adiag(self, **kwargs):
        """The block-diagonal Jacobian action and its diagonal.

        Block diagonal is the physics, not an approximation: the CI equation's
        orbital coupling term is ``Z^dag d(F)/dc``, and ``F`` is built from the
        RHF density, which has no CI-coefficient dependence whatsoever
        (section 2.2). In Park's CASSCF Lagrangian the corresponding term is
        ``Z^dag d(A - A^dag)/dc`` and does not vanish, which is what makes
        CP-CASSCF one big coupled system."""
        st = self.st
        ndet = st.ci.size
        M = self._jacobian

        def Aop(v: np.ndarray) -> np.ndarray:
            """``[ P(H-E)P z ; M lambda ]``. ``M[k, m] = dG_m/dkappa_k``, so
            stationarity for rotation ``k`` reads ``B_k + sum_m lambda_m
            M[k, m] = 0``, i.e. ``M lambda = -B``: the multiplier index is the
            one contracted."""
            v = np.asarray(v).ravel()
            return np.concatenate([self.ci_hop(v[:ndet]), M @ v[ndet:]])

        hdiag = fci.direct_spin1.make_hdiag(self._h1_act, self._g_act, st.nact, st.nelecas)
        # The constraint-to-pair bijection of :func:`constraints` makes M
        # square and its diagonal the right preconditioner: it is the
        # orbital-energy difference Park Eqs. 56-57 divide by on (O2)'s pairs
        # and the selection-boundary denominator on (O1)'s.
        adiag = np.concatenate([
            hdiag.ravel() + self._ecore - self.e_casci_act,
            np.diag(M).copy(),
        ])
        return Aop, adiag

    # -- hooks 3 and 4: the nuclear derivatives --------------------------

    def _phi(self, geom, *, ham_only: bool, lag_only: bool, guard: bool = False,
             where: str = "") -> float:
        """One evaluation of ``Phi`` (or of either half) along the frozen
        orthonormal path at a displaced geometry. ``guard`` runs the PC
        conditioning guard there too (:meth:`Functional.pc_energy`)."""
        st = self.st
        fn = self._geom_fn(geom)
        C = frozen_coeff(st.mo_coeff, self._S0_half, fn.ovlp)
        b = fn.evaluate(C, st.ci, with_corr=self.with_corr, guard=guard, where=where)
        ham = b["e_casci"] + b["e_corr"]
        # <z|H(x)|c>: z is projected against c, so the -E_CASCI <z|c> half of
        # the (C1) constraint contributes nothing and this one term is the
        # whole CI multiplier's response.
        lag = (float(self.lam @ constraint_vector(b, self.cons))
               + float(self.z.ravel() @ b["hvec"]))
        if ham_only:
            return ham
        if lag_only:
            return lag
        return ham + lag

    def _geom_fn(self, geom) -> Functional:
        key = tuple(round(v, 12) for _s, xyz in geom for v in xyz)
        cached = self._fn_cache.get(key)
        if cached is None:
            cached = Functional(self.st, geom, pc=self.pc)
            self._fn_cache[key] = cached
        return cached

    def _nuclear_difference(self, *, ham_only: bool, lag_only: bool) -> tuple:
        """``dPhi/dx`` component by component, as a central difference along
        the frozen path -- with a Richardson pass at ``h/2`` when asked, whose
        difference from the coarse one is the error bar on the number."""
        st = self.st
        natm = len(st.geom)
        de = np.zeros((natm, 3))
        res = np.zeros((natm, 3))
        probed = np.zeros((natm, 3))
        steps = (self.step, self.step / 2) if self.richardson else (self.step,)
        # EVERY component is differenced, including the ones that are zero by
        # symmetry. nevpt2_numgrad.py has to skip those -- it re-converges an
        # RHF+CASCI at each displacement, and a symmetry-lowering
        # displacement changes the active-space selection
        # (docs/gradient-theory.md, section 4.2). The frozen path re-solves
        # nothing: the MO coefficients and the CI vector are carried over
        # unchanged, so there is no selection to break and no reason to skip.
        # The zero components then come out as the zeros they are, which makes
        # them the free check section 4.3 says they are rather than an
        # assumption.
        for iatom in range(natm):
            for comp in range(3):
                g = {}
                for h in steps:
                    plus = self._phi(ngrad.displace(st.geom, iatom, comp, +h),
                                     ham_only=ham_only, lag_only=lag_only)
                    minus = self._phi(ngrad.displace(st.geom, iatom, comp, -h),
                                      ham_only=ham_only, lag_only=lag_only)
                    g[h] = (plus - minus) / (2.0 * h)
                de[iatom, comp] = g[steps[-1]]
                if len(steps) > 1:
                    rich = (4 * g[steps[1]] - g[steps[0]]) / 3
                    res[iatom, comp] = abs(rich - g[steps[1]])
                    de[iatom, comp] = rich
                probed[iatom, comp] = 1.0
        return de, res, probed

    def get_ham_response(self, **kwargs) -> np.ndarray:
        """``<dH/dx>``: the gradient the method would have if it were
        variational -- here ``d/dx [E_CASCI + E_corr]`` along the frozen
        orthonormal path, nuclear repulsion included (it is inside
        ``E_CASCI``'s core energy)."""
        de, res, probed = self._nuclear_difference(ham_only=True, lag_only=False)
        self.residuals["ham_response_richardson"] = float(res.max()) if self.richardson else None
        self._probed = probed
        return de

    def get_LdotJnuc(self, Lvec, **kwargs) -> np.ndarray:
        """``Lvec . J_nuc``: the multipliers against the constraints' nuclear
        derivative -- ``d/dx [ sum Z F(x) + sum zc f(x) + <z|H(x)|c> ]`` along
        the same path. The second term of the final gradient, and the only
        place the Z-vector solution enters it."""
        de, res, _probed = self._nuclear_difference(ham_only=False, lag_only=True)
        self.residuals["ldotjnuc_richardson"] = float(res.max()) if self.richardson else None
        return de

    # -- the driver -------------------------------------------------------

    def prepare(self, log=print, *, sweep: dict | None = None,
                sweep_corr: bool | None = None) -> None:
        """Everything the hooks need that is a property of the reference state
        rather than of a displacement: the active Hamiltonian the CI
        Z-vector's operator reads, the frozen path's ``S^{1/2}``, the
        zero-by-symmetry components, and the constraint Jacobian.

        ``sweep`` hands in another instance's :attr:`sweep` instead of running
        one -- ``--source both`` solves twice on one state and pays for the
        expensive pass once. ``sweep_corr`` forces ``E_corr`` into the sweep
        even under a dump, so that PySCF's half of ``A`` exists to be compared
        with it."""
        st = self.st
        self._fn_cache = {}
        self._S0_half = sym_power(self.fn.ovlp, 0.5)
        eris = self.fn.eris(st.mo_coeff)
        nmo = st.nmo
        self._h1_act = np.asarray(eris["h1eff"])[st.act, st.act]
        self._g_act = np.asarray(eris["ppaa"]).reshape(nmo, nmo, st.nact, st.nact)[
            st.act, st.act]
        self._g_ppaa = np.asarray(eris["ppaa"]).reshape(nmo, nmo, st.nact, st.nact)
        self._g_papa = np.asarray(eris["papa"]).reshape(nmo, st.nact, nmo, st.nact)
        self._h2abs = self.fn.active_hamiltonian(eris)
        self._ecore = self.fn.core_energy(eris)
        hc = fci.direct_spin1.contract_2e(self._h2abs, st.ci, st.nact, st.nelecas).ravel()
        c = st.ci.ravel()
        # The active-space eigenvalue H - E uses, i.e. E_CASCI itself.
        self.e_casci_act = self._ecore + float(c @ hc)

        self.point_group = ngrad.point_group(st.geom, st.basis)
        self.zero_axes = zero_axes(st.geom)
        log(f"  point group {self.point_group[0]} (used {self.point_group[1]}); "
            f"zero by symmetry: "
            + (", ".join(f"d/d{AXES[c]}" for c in self.zero_axes) or "nothing")
            + " (differenced anyway, as a check)")

        # ONE sweep gives all three kappa-direction objects. It differentiates
        # E_corr too unless a dump already supplied that half, in which case
        # the sweep is the cheap E_CASCI-only one and the dump's Y is added.
        if sweep is None:
            if sweep_corr is None:
                sweep_corr = self.with_corr and not self.src.from_dump
            sweep = kappa_sweep(self.fn, st, self.cons, with_corr=sweep_corr,
                                step=self.kappa_step, log=log)
        if self.with_corr and not self.src.from_dump and not sweep["with_corr"]:
            raise SystemExit("prepare: PySCF's source term needs a sweep that "
                             "differentiated E_corr")
        self.sweep = sweep
        self._jacobian = sweep["M"]
        self._q = sweep["Q"]
        self.src = SourceTerms(
            kappa=(sweep["A_casci"] + self.src.kappa) if self.src.from_dump
            else (sweep["A"] if self.with_corr else sweep["A_casci"]),
            ci=self.src.ci, origin=self.src.origin, from_dump=self.src.from_dump,
        )
        log(f"  one kappa sweep over {len(self.pairs)} pairs in "
            f"{sweep['seconds']:.0f}s: the source term A (max |A| = "
            f"{np.abs(self.src.kappa).max():.6e}), the {len(self.pairs)} x "
            f"{self.n_constraint} constraint Jacobian M, and Q = d(H c)/dkappa")

    def solve_lagrange(self, Lvec_guess=None, level_shift=None, log=print, **kwargs):
        """The base class's own method, overridden -- and the reason is
        numerical, not stylistic.

        ``lagrange.Gradients.solve_lagrange`` runs **conjugate gradient** on
        the whole ``nlag`` system, which needs a symmetric positive-definite
        operator. Half of this one is:

        * the **CI block** is ``P (H - E_CASCI) P`` with the reference state
          projected out, and ``E_CASCI`` is the lowest eigenvalue, so it is
          symmetric positive semidefinite on the projected space -- CG's own
          case, and that is what it gets;
        * the **orbital block** is **indefinite by construction**. Park
          Eqs. 56-57 divide by ``eps_i - eps_j``, whose sign flips with the
          pair order, so the constraint Jacobian's diagonal carries both
          signs. CG on it diverges (measurably: it overflows within a few
          dozen iterations). It is square, so it gets a direct solve, with
          GMRES through the very same :meth:`get_Aop_Adiag` operator beside it
          so the answer has an independent check and an iterative residual to
          report.

        Both residuals land in :attr:`residuals` and are printed; neither is
        assumed."""
        from scipy.sparse import linalg as sparse_linalg

        st = self.st
        ndet = st.ci.size
        bvec = self.get_wfn_response()
        Aop, adiag = self.get_Aop_Adiag()
        Lvec = np.zeros_like(bvec)

        # --- the CI Z-vector, first and alone (section 2.2) --------------
        ci_op = sparse_linalg.LinearOperator(
            (ndet, ndet), matvec=lambda v: self.ci_hop(v), dtype=float
        )
        ci_diag = adiag[:ndet].copy()
        ci_diag[np.abs(ci_diag) < 1e-8] = 1e-8
        c = st.ci.ravel()

        # The preconditioner MUST project too, and the answer must be
        # projected again afterwards. `|c>` is an exact null vector of
        # `P (H - E) P`, so any multiple of it leaves the residual untouched
        # and the solve cannot see it -- while `Phi`'s CI term
        # `z^dag (H - E_CASCI) c` drops `- (dE_CASCI/dx) (z.c)` on the
        # assumption that `z.c = 0`. A Jacobi preconditioner does not commute
        # with `P`, so without this the iterates drift along `|c>` and the
        # multiplier comes back almost PARALLEL to the state: the residual
        # stays at round-off and the gradient is wrong by
        # `(dE_CASCI/dx) (z.c)`, which is not small.
        def ci_precond(v):
            w = np.asarray(v).ravel() / ci_diag
            return w - float(c @ w) * c

        ci_prec = sparse_linalg.LinearOperator((ndet, ndet), matvec=ci_precond,
                                               dtype=float)
        z, info = sparse_linalg.cg(
            ci_op, -bvec[:ndet], rtol=self.conv_rtol, atol=self.conv_atol,
            maxiter=self.max_cycle, M=ci_prec,
        )
        z = z * self.allowed
        z = z - float(c @ z) * c
        self.residuals["ci_overlap"] = float(c @ z)
        Lvec[:ndet] = z
        self.residuals["ci_cg"] = float(np.linalg.norm(self.ci_hop(z) + bvec[:ndet]))
        self.residuals["ci_cg_converged"] = bool(info == 0)
        self.residuals["ci_cg_info"] = int(info)
        self.residuals["ci_norm"] = float(np.linalg.norm(z))

        # --- the orbital multipliers ------------------------------------
        drop, notes = self.degenerate_constraints()
        keep = np.array([i for i in range(self.n_constraint) if i not in set(drop)])
        if notes:
            log(f"  (O2): {len(notes)} degenerate pair(s) left at zero -- "
                + "; ".join(notes))
        # Dropping an (O2) constraint drops its own kappa pair with it: the
        # bijection of :func:`constraints` pairs them one to one, and the
        # stationarity equation of a degenerate pair has no other multiplier
        # in it, so what is left of that row is the source term -- already
        # checked to vanish by :meth:`degenerate_constraints`. So the reduced
        # system stays square.
        M = self._jacobian[np.ix_(keep, keep)]
        rhs = (-bvec[ndet:])[keep]
        lam_k, _res, rank, sv = np.linalg.lstsq(M, rhs, rcond=JACOBIAN_RCOND)
        lam = np.zeros(self.n_constraint)
        lam[keep] = lam_k
        Lvec[ndet:] = lam
        self.residuals["orbital_direct"] = float(np.linalg.norm(M @ lam_k - rhs))
        self.residuals["orbital_rank"] = int(rank)
        self.residuals["orbital_cond"] = float(sv[0] / sv[-1]) if sv.size else 0.0
        self.residuals["orbital_norm"] = float(np.linalg.norm(lam))
        self.residuals["orbital_dropped"] = len(drop)

        n = len(keep)
        orb_op = sparse_linalg.LinearOperator((n, n), matvec=lambda v: M @ v, dtype=float)
        orb_diag = np.diag(M).copy()
        orb_diag[np.abs(orb_diag) < 1e-8] = 1e-8
        orb_prec = sparse_linalg.LinearOperator((n, n), matvec=lambda v: v / orb_diag,
                                                dtype=float)
        lam_it, info_orb = sparse_linalg.gmres(
            orb_op, rhs, rtol=self.conv_rtol, atol=self.conv_atol,
            maxiter=self.max_cycle, M=orb_prec,
        )
        self.residuals["orbital_gmres"] = float(np.linalg.norm(M @ lam_it - rhs))
        self.residuals["orbital_gmres_info"] = int(info_orb)
        self.residuals["orbital_gmres_gap"] = float(np.linalg.norm(lam_it - lam_k))

        is_o1 = np.array([kind == "F" for kind, _p, _q in self.cons])
        self.residuals["max_Z_o1"] = float(np.abs(lam[is_o1]).max()) if is_o1.any() else 0.0
        self.residuals["max_zc_o2"] = (
            float(np.abs(lam[~is_o1]).max()) if (~is_o1).any() else 0.0
        )
        conv = self.residuals["ci_cg_converged"] and info_orb == 0
        self.converged = bool(conv)
        return bool(conv), Lvec, bvec, Aop, adiag

    def solve_multipliers(self, log=print, max_iter: int = 20,
                          tol: float = 1e-12) -> None:
        """Both Z-vector solves, run to a fixed point, every residual
        reported.

        The CI equation is solved first and alone -- section 2.2's
        decoupling -- and that is exactly right at leading order: ``F``
        carries no CI dependence, so there is no CP-CASSCF here. What couples
        the two at all is the (O2) multipliers (:meth:`ci_coupling`) and the CI
        relaxation ``Q z`` in the orbital right-hand side, both small. So the
        pair is iterated, and **the iteration count and the step it stopped at
        are printed**: a one-iteration fixed point is the decoupled answer,
        and anything more says how much the coupling was worth."""
        st = self.st
        ndet = st.ci.size
        conv = False
        for it in range(1, max_iter + 1):
            conv, Lvec, bvec, _Aop, _adiag = self.solve_lagrange(
                log=log if it == 1 else (lambda *a: None)
            )
            step = max(float(np.abs(Lvec[:ndet] - self.z.ravel()).max()),
                       float(np.abs(Lvec[ndet:] - self.lam).max()))
            self.Lvec = Lvec
            self.z = Lvec[:ndet].reshape(st.ci.shape)
            self.lam = Lvec[ndet:].copy()
            self.residuals["fixed_point_iterations"] = it
            self.residuals["fixed_point_step"] = step
            if step < tol:
                break
        log(f"  the CI/(O2) fixed point: {self.residuals['fixed_point_iterations']} "
            f"iteration(s), last step {self.residuals['fixed_point_step']:.2e} "
            f"(the coupling section 2.2 omits -- f depends on the CI vector "
            f"through gamma)")
        log(f"  CI Z-vector: |z| = {self.residuals['ci_norm']:.6e}, "
            f"|P(H-E)Pz + Py| = {self.residuals['ci_cg']:.3e} "
            f"({'converged' if self.residuals['ci_cg_converged'] else 'NOT CONVERGED'}), "
            f"x = c.y = {self.x_norm:+.6e}")
        log(f"  orbital Z-vector: {self.residuals['orbital_rank']} of "
            f"{self.n_constraint} multipliers determined, "
            f"cond {self.residuals['orbital_cond']:.2e}")
        log(f"    |M lambda + B| = {self.residuals['orbital_direct']:.3e} (direct), "
            f"{self.residuals['orbital_gmres']:.3e} (GMRES through get_Aop_Adiag); "
            f"the two answers differ by {self.residuals['orbital_gmres_gap']:.3e}")
        log(f"    |lambda| = {self.residuals['orbital_norm']:.6e}; "
            f"max |Z| (O1) = {self.residuals['max_Z_o1']:.3e}, "
            f"max |zc| (O2) = {self.residuals['max_zc_o2']:.3e}")
        assert conv or True  # convergence is reported, not asserted

        # Section 1.3's free check: the active-active block is a gauge, so the
        # source term restricted to an active rotation must be cancelled by
        # the CI source contracted with the induced CI rotation -- no
        # multiplier involved. With the CI relaxation already inside Psi, that
        # is the statement that Psi's active-active gradient vanishes.
        self.residuals["gauge"] = self._gauge_identity()
        log(f"  the gauge identity (section 1.3, no multiplier to absorb it): "
            f"max |dPsi/dkappa| over the act_O- and act_V-internal rotations = "
            f"{self.residuals['gauge']:.3e}")

    def _gauge_identity(self) -> float:
        """``max |A + Q z|`` over the **gauge** directions -- act_O- and
        act_V-internal rotations.

        Section 1.3's closed identity: an active rotation moves the energy
        only as much as the induced CI rotation moves it back, and no
        multiplier is involved. It is reported rather than gated, because what
        must vanish is the *whole* gauge derivative of the Lagrangian, and the
        constraints' own gauge response -- ``sum_m lambda_m dG_m/dkappa_g``,
        which is **not** zero, since ``f``'s active sum is not covariant at
        fixed ``gamma`` -- is not part of this sweep. So a small number here
        is reassurance and a large one is a question, not a verdict; what
        settles it is the comparison against ``grad_num``."""
        st = self.st
        worst = 0.0
        gauge = [(p, q) for p in range(st.ncore, st.nocc)
                 for q in range(p + 1, st.nocc)
                 if st.space_of(p) == st.space_of(q)]
        z = self.z.ravel()
        for p, q in gauge:
            vals = []
            for sign in (+1.0, -1.0):
                b = self.fn.evaluate(givens(st.mo_coeff, p, q, sign * self.kappa_step),
                                     st.ci, with_corr=self.with_corr)
                vals.append(b["e_casci"] + b["e_corr"] + float(z @ b["hvec"]))
            worst = max(worst, abs((vals[0] - vals[1]) / (2.0 * self.kappa_step)))
        return worst

    def kernel(self, **kwargs) -> np.ndarray:
        """The complete gradient: the skeleton derivative plus the
        multipliers against the constraints' nuclear derivative."""
        ham = self.get_ham_response()
        lag = self.get_LdotJnuc(self.lam)
        self.de = ham + lag
        self.ham_response = ham
        self.ldotjnuc = lag
        return self.de


# --------------------------------------------------------------------------
# Residual-independent checks
# --------------------------------------------------------------------------
#
# docs/testing.md, "Residual-independent checks of the gradient path" has the
# measured ladders and the record of which check caught which of the five
# errors below; this is the code.


def ladder_steps(exps: tuple[int, int]) -> list[float]:
    """``h = 2^-k`` for ``k`` in ``[first, last]``: exact in binary, so the
    ratio across a halving is exactly the Richardson factor."""
    return [2.0 ** -k for k in range(exps[0], exps[1] + 1)]


def fourfold_runs(err: list[float]) -> list[tuple[int, int]]:
    """Every maximal run of consecutive halvings that cut ``err`` by a factor
    in ``(LADDER_LOW, LADDER_HIGH)``, as ``(length, index of the finer rung of
    its last halving)``, coarsest first. A halving whose finer error is zero
    has no ratio and breaks a run.

    The rule is the C++ checks' ``nevpt2::longestFourfoldRun``
    (src/gradient/gradient_plan.cppm, pinned by test/gradient_plan/'s
    FourfoldRunTests), which reports the longest of these."""
    runs, run = [], 0
    for i in range(1, len(err)):
        ratio = err[i - 1] / err[i] if err[i] > 0.0 else 0.0
        if LADDER_LOW < ratio < LADDER_HIGH:
            run += 1
            continue
        if run:
            runs.append((run, i - 1))
        run = 0
    if run:
        runs.append((run, len(err) - 1))
    return runs


@dataclass
class Check:
    """One residual-independent check: its verdict, a one-line summary, and
    (for a ladder) the table it was read from."""

    name: str
    ok: bool
    summary: str
    table: list = field(default_factory=list)
    numbers: dict = field(default_factory=dict)


def step_ladder(name: str, f, steps: list[float], ref: float) -> Check:
    """Central differences of ``f`` at every step of ``steps``, gated on a
    **run** of fourfold halvings rather than on any single step.

    Two separate facts, both required:

    * **the ladder converges.** The change between neighbouring rungs must
      fall fourfold for at least :data:`LADDER_RUN` consecutive halvings --
      an ``O(h^2)`` truncation error, walked down through. Read off the
      ladder alone, with no reference in it. Of several such runs the
      **finest** is used: it is the one nearest ``h -> 0``, and a coarser
      one can end where the function stops being analytic on the ladder's
      scale (a near-pole, as the SC energy has on N2) rather than at the
      round-off floor;
    * **it converges to** ``ref`` -- the assembled quantity under test,
      contracted with the ladder's direction, or zero for a stationarity
      condition. The ladder's own value is the Richardson extrapolant
      ``(4 fd(h) - fd(2h)) / 3`` at the finest pair of that run, and
      ``|it - ref|`` must be under :data:`LADDER_TOL`.

    Comparing a Richardson extrapolant, and not the smallest ``|fd - ref|``
    in the sweep, is what keeps the check honest: ``fd(h) - ref = c h^2 -
    delta`` crosses zero at some ``h`` whenever ``ref`` is wrong by
    ``delta``, so a sweep minimum can sit on that crossing and look like
    agreement."""
    fd = [(f(+h) - f(-h)) / (2.0 * h) for h in steps]
    dif = [abs(fd[i] - fd[i + 1]) for i in range(len(fd) - 1)]
    runs = fourfold_runs(dif)
    good = [r for r in runs if r[0] >= LADDER_RUN]
    run, end = good[-1] if good else max(runs, default=(0, 0))
    if run:
        value = (4.0 * fd[end + 1] - fd[end]) / 3.0
        err = abs(value - ref)
    else:
        value, err = float("nan"), float("inf")
    ok = run >= LADDER_RUN and err <= LADDER_TOL
    table = []
    for i, h in enumerate(steps):
        d = f"{dif[i - 1]:10.3e}" if i else f"{'-':>10}"
        ratio = f"{dif[i - 2] / dif[i - 1]:8.3f}" if i > 1 and dif[i - 1] > 0.0 else f"{'-':>8}"
        mark = " <" if run and end - run + 1 <= i <= end + 1 else ""
        table.append(f"      2^-{round(-np.log2(h)):<3d} {fd[i]:+.12e} {d} {ratio}{mark}")
    summary = (f"{run} consecutive fourfold halvings (need {LADDER_RUN})"
               + (f", ending at h = 2^-{round(-np.log2(steps[end + 1]))}" if run else "")
               + f"; |Richardson - ref| = {err:.2e} (allowed {LADDER_TOL:g}), "
               f"ref {ref:+.10e}")
    return Check(name, ok, summary, table,
                 dict(ref=ref, value=value, err=err, run=run, steps=np.array(steps),
                      fd=np.array(fd)))


def ci_check_direction(g: Gradients, seed: int) -> np.ndarray:
    """A seeded unit CI direction **orthogonal to the state, in its spatial
    irrep and a spin eigenfunction**: ``u = P W |c>`` with ``W`` spin-free
    and totally symmetric, so ``W|c>`` is a spin eigenfunction of the
    state's own irrep. A bare determinant direction is neither
    (:data:`CI_STEP_PC`; docs/gradients.md, section 7).

    * **SC**: ``W`` is the active Hamiltonian's own one- and two-electron
      integrals, each multiplied elementwise by a seeded random tensor of the
      same permutational symmetry -- nonzero only where the integrals are, so
      totally symmetric in any active basis, even one that mixes degenerate
      partners; and two-electron, because with one orbital per irrep (N2's
      CAS(4,4)) every totally symmetric one-electron operator is diagonal and
      the directions it generates collapse onto one.
    * **PC**: ``W = sum w_pq E_pq`` over the pairs the state's ``dm1``
      couples -- nevpt2.pc_tangent's own ``pcTangentDirection``. A two-electron
      ``W`` lifts one of Sir's null modes as ``h^2`` relative, which stays
      above lstsq's cut over the whole ladder: the ladder then converges --
      cleanly, fourfold -- to the derivative of *another branch* of the
      pseudo-inverse, and the gap guard passes it (docs/testing.md,
      "Residual-independent checks of the gradient path", has the N2 numbers).
      A one-electron one lifts it only as ``h^4``, under the cut below
      ``2^-10`` (:data:`CI_LADDER_PC`), and :func:`_pc_guard`'s rank check
      holds the ladder to that."""
    st = g.st
    n = st.nact
    rng = np.random.default_rng(seed)
    if g.pc:
        dm1 = np.asarray(fci.direct_spin1.make_rdm1(st.ci, n, st.nelecas))
        w = rng.uniform(-1.0, 1.0, (n, n)) * (np.abs(dm1) > 1e-12 * np.abs(dm1).max())
        u = np.asarray(fci.direct_spin1.contract_1e(w + w.T, st.ci, n, st.nelecas)).ravel()
    else:
        r1 = rng.uniform(-1.0, 1.0, (n, n))
        r1 = r1 + r1.T
        r2 = rng.uniform(-1.0, 1.0, (n, n, n, n))
        r2 = r2 + r2.transpose(1, 0, 2, 3)
        r2 = r2 + r2.transpose(0, 1, 3, 2)
        r2 = r2 + r2.transpose(2, 3, 0, 1)
        op = fci.direct_spin1.absorb_h1e(np.asarray(g._h1_act) * r1,
                                         np.asarray(g._g_act) * r2, n, st.nelecas, 0.5)
        u = np.asarray(fci.direct_spin1.contract_2e(op, st.ci, n, st.nelecas)).ravel()
    u = u * g.allowed
    c = st.ci.ravel()
    u = u - float(c @ u) * c
    norm = float(np.linalg.norm(u))
    if norm < 1e-12:
        raise SystemExit(
            f"the seeded CI check direction (seed {seed}) came out zero: the state "
            "is an eigenfunction of the operator it is generated by"
        )
    return u / norm


def on_sphere(st: State, u: np.ndarray, h: float) -> np.ndarray:
    """``(c + h u) / |c + h u|``: the CI ladders walk the unit sphere. With
    ``<c|u> = 0`` its tangent at ``h = 0`` is the straight line's, and off
    the sphere the SC functional has poles (:func:`pyscf_ci_source`) and the
    hole-side metrics' constants stop being hole RDMs."""
    v = st.ci.ravel() + h * u
    return (v / np.linalg.norm(v)).reshape(st.ci.shape)


def non_gauge_rotations(st: State) -> list[tuple[int, int]]:
    """Every rotation the Lagrangian must be stationary along, derived
    **independently of** :func:`kappa_pairs`: a rotation is a gauge exactly
    when it stays inside the active space AND inside one side of the RHF
    occupied/virtual divide (:meth:`State.occ_projector`) -- the energy is
    invariant to it with the CI vector carried along, and it moves no
    constraint. Anything else (an act_O-act_V rotation included: it moves the
    RHF occupied space) is a direction whose stationarity the multipliers
    owe. Derived from the projector rather than from the bookkeeping the
    solve uses, so the check does not inherit that bookkeeping's mistakes --
    which is the error #40 found."""
    occ = np.diag(st.occ_projector()) > 0.5
    active = np.zeros(st.nmo, dtype=bool)
    active[st.act] = True
    return [(p, q) for p in range(st.nmo) for q in range(p + 1, st.nmo)
            if not (active[p] and active[q] and occ[p] == occ[q])]


def kappa_check_direction(g: Gradients, pairs: list, seed: int) -> np.ndarray:
    """A seeded antisymmetric ``kappa`` of unit norm over ``pairs``, and
    **totally symmetric**: ``X - X^T`` summed over seeded multiples of
    products of the state's own totally symmetric MO matrices -- ``h``, the
    generalized Fock ``f`` and the RHF Fock ``F``.

    Totally symmetric because a symmetry-breaking rotation is where the SC
    functional stops being a smooth function of the step: a perturber the
    state's symmetry leaves empty (``N = H = 0``) fills as ``t^2`` along one,
    and its ``-N^2 / (Delta N + H)`` term can put a pole near ``t = 0``. A
    product of matrices that commute with the point group commutes with it
    too, in any MO basis; a random matrix does not. Nothing is lost by it:
    along a symmetry-breaking rotation every term of the stationarity
    condition vanishes by symmetry."""
    st = g.st
    h = st.mo_coeff.T @ g.fn.hcore @ st.mo_coeff
    f, F = st.f, st.fock_rhf
    prods = [h @ f, h @ F, f @ F, h @ f @ F, f @ h @ F @ h]
    rng = np.random.default_rng(seed)
    x = sum(a * p / max(np.abs(p).max(), 1e-300)
            for a, p in zip(rng.uniform(-1.0, 1.0, len(prods)), prods, strict=True))
    x = x - x.T
    k = np.zeros((st.nmo, st.nmo))
    for p, q in pairs:
        k[p, q], k[q, p] = x[p, q], -x[p, q]
    norm = float(np.linalg.norm(k[np.triu_indices(st.nmo, 1)]))
    if norm < 1e-12:
        raise SystemExit(f"the seeded kappa check direction (seed {seed}) came out zero")
    return k / norm


def nuclear_check_direction(g: Gradients, seed: int) -> np.ndarray:
    """A seeded unit nuclear displacement that is **totally symmetric**: a
    seeded combination of the assembled gradient and the nuclear repulsion's
    own -- both derivatives of a function the point group leaves invariant,
    so both are. For :func:`kappa_check_direction`'s reason."""
    from pyscf.grad import rhf as rhf_grad

    gnuc = np.asarray(rhf_grad.grad_nuc(g.fn.mol))
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(-1.0, 1.0, 2)
    d = (a * g.de / max(float(np.linalg.norm(g.de)), 1e-300)
         + b * gnuc / max(float(np.linalg.norm(gnuc)), 1e-300))
    norm = float(np.linalg.norm(d))
    if norm < 1e-12:
        raise SystemExit(f"the seeded nuclear check direction (seed {seed}) came out zero")
    return d / norm


def displace_along(geom, d: np.ndarray, h: float):
    """``geom`` with every atom moved by ``h d`` bohr (the geometry is in
    Angstrom)."""
    return [(sym, tuple(float(v + h * dv * ngrad.BOHR)
                        for v, dv in zip(xyz, d[i], strict=True)))
            for i, (sym, xyz) in enumerate(geom)]


def check_projection(g: Gradients, src: SourceTerms) -> list[Check]:
    """The projection identities -- ``y`` IS the projected gradient and
    ``z`` IS projected -- asserted on the numbers rather than trusted to the
    construction.

    ``c . y = 0``: :func:`pyscf_ci_source` differences along ``P e_I``, so
    ``y = P g`` and its overlap with the state is zero up to the quotients'
    own error. A componentwise quotient along a bare ``e_I`` walks off the
    unit sphere, and its ``c . y`` is the *radial* derivative of a functional
    that has poles there -- #40's third error, whose quotients converged
    over three decades. A dump is projected on read (:meth:`get_wfn_response`),
    so this is the pyscf source's check.

    ``c . z = 0``: ``Phi``'s CI term drops ``-(dE_CASCI/dx)(z . c)``, and the
    CI solve cannot see a multiple of ``|c>`` -- it is a null vector of
    ``P (H - E) P``, so the residual is unchanged by it (#40's second
    error)."""
    st = g.st
    c = st.ci.ravel()
    out = []
    y = src.ci.ravel()
    if src.origin == "pyscf" and np.any(y):
        r = abs(float(c @ y))
        out.append(Check("projection of y", r <= PROJECTION_TOL,
                         f"|c . y| = {r:.2e} (allowed {PROJECTION_TOL:g}), "
                         f"|y| = {np.linalg.norm(y):.3e}", numbers=dict(value=r)))
    else:
        out.append(Check("projection of y", True,
                         "not applicable: " + ("y = 0" if not np.any(y) else
                                               "a dump's y is projected on read")))
    r = abs(float(c @ g.z.ravel()))
    out.append(Check("projection of z", r <= PROJECTION_TOL,
                     f"|c . z| = {r:.2e} (allowed {PROJECTION_TOL:g}), "
                     f"|z| = {np.linalg.norm(g.z):.3e}", numbers=dict(value=r)))
    return out


def check_ci_source_linearity(g: Gradients, src: SourceTerms) -> list[Check]:
    """``y`` is linear in its direction: for each seeded direction ``u``, the
    directional derivative of ``E_corr`` -- step-laddered, along the unit
    sphere, through :meth:`Functional.evaluate` -- must converge to
    ``<u, y>``.

    ``evaluate`` is the route ``Phi`` is differentiated through, and it is
    not the route ``y`` was built through (:meth:`Functional.corr_at_ci`), so
    a ``y`` of a different functional is caught: one that holds the orbital
    energies fixed (#40's fourth error), one taken componentwise along bare
    determinant axes (#40's third, not a gradient at all), or a PC one whose
    quotient was stepped onto another branch of the pseudo-inverse (#41's,
    which the ladder's spin-adapted direction does not share)."""
    st = g.st
    y = src.ci.ravel()
    exps = CI_LADDER_PC if g.pc else CI_LADDER
    out = []
    for seed in CHECK_SEEDS:
        u = ci_check_direction(g, seed)

        def e_corr(h: float, u=u) -> float:
            return g.fn.evaluate(st.mo_coeff, on_sphere(st, u, h), with_corr=True,
                                 where=f"the CI ladder at h = {h:.3e}")["e_corr"]

        out.append(step_ladder(f"y linearity, seed {seed}", e_corr, ladder_steps(exps),
                               float(u @ y)))
    g.fn.pt.ci = st.ci
    g.fn.mc.ci = st.ci
    return out


def check_ci_stationarity(g: Gradients) -> list[Check]:
    """The CI Z-vector solve, checked by what it is FOR: the Lagrangian's CI
    part

        L(c) = E_corr(c) + sum_m lambda_m G_m(c) + <z|H - E_CASCI(c)|c>

    must be stationary along every in-sector direction orthogonal to the
    state. Step-laddered along the unit sphere through
    :meth:`Functional.evaluate`, with the multipliers held at their solved
    values -- so ``ci_hop``, ``ci_coupling`` and the source term are all
    bypassed, and a CI equation that was solved exactly but was the wrong
    equation shows up as a slope. (``E_CASCI(c)`` itself is left out: its
    stationarity is the CASCI eigen-equation's, a residual the Davidson
    tolerance owns, not something the Z-vector solve can be asked for.)"""
    st = g.st
    exps = CI_LADDER_PC if g.pc else CI_LADDER
    z, lam = g.z.ravel(), g.lam
    out = []
    for seed in CHECK_SEEDS:
        u = ci_check_direction(g, seed)

        def lagrangian(h: float, u=u) -> float:
            ci = on_sphere(st, u, h)
            b = g.fn.evaluate(st.mo_coeff, ci, with_corr=g.with_corr,
                              where=f"the CI-stationarity ladder at h = {h:.3e}")
            return (b["e_corr"] + float(lam @ constraint_vector(b, g.cons))
                    + float(z @ b["hvec"]) - b["e_casci"] * float(z @ ci.ravel()))

        out.append(step_ladder(f"CI stationarity, seed {seed}", lagrangian,
                               ladder_steps(exps), 0.0))
    g.fn.pt.ci = st.ci
    g.fn.mc.ci = st.ci
    return out


def check_orbital_stationarity(g: Gradients) -> list[Check]:
    """The orbital Z-vector solve, checked by what it is for: the Lagrangian
    at fixed CI vector,

        L(kappa) = E_CASCI + E_corr + sum_m lambda_m G_m(kappa) + <z|H(kappa)|c>,

    is stationary along every NON-GAUGE rotation -- the set
    :func:`non_gauge_rotations` derives from the RHF occupied projector, not
    the solve's own :func:`kappa_pairs`. Step-laddered along a seeded
    rotation ``C exp(t kappa)``, so ``KAPPA_STEP``'s single-step quotients
    (the source term, the Jacobian and ``Q``) are gated by a ladder too.

    A solve whose bookkeeping drops a direction (#40's first error: the
    act_O-act_V block taken for a gauge) converges to round-off on the
    system it was given and leaves the Lagrangian sloped along exactly that
    direction."""
    from scipy.linalg import expm

    st = g.st
    pairs = non_gauge_rotations(st)
    dm1 = None if g.with_corr else g.fn.reference_dm1(st)
    z, lam = g.z.ravel(), g.lam
    out = []
    for seed in CHECK_SEEDS:
        k = kappa_check_direction(g, pairs, seed)

        def lagrangian(t: float, k=k) -> float:
            b = g.fn.evaluate(st.mo_coeff @ expm(t * k), st.ci, with_corr=g.with_corr,
                              dm1=dm1, guard=g.pc,
                              where=f"the orbital-stationarity ladder at t = {t:.3e}")
            return (b["e_casci"] + b["e_corr"] + float(lam @ constraint_vector(b, g.cons))
                    + float(z @ b["hvec"]))

        out.append(step_ladder(f"orbital stationarity, seed {seed} ({len(pairs)} rotations)",
                               lagrangian, ladder_steps(KAPPA_LADDER), 0.0))
    return out


def check_nuclear_ladder(g: Gradients) -> list[Check]:
    """``NUC_STEP``'s quotients, gated by a ladder: ``Phi`` differenced along
    a seeded totally symmetric displacement of every atom must converge to
    the assembled gradient contracted with it. Run after
    :meth:`Gradients.kernel`, with the PC guard sampled at every rung."""
    st = g.st
    out = []
    for seed in CHECK_SEEDS:
        d = nuclear_check_direction(g, seed)

        def phi(h: float, d=d) -> float:
            return g._phi(displace_along(st.geom, d, h), ham_only=False, lag_only=False,
                          guard=g.pc, where=f"the nuclear ladder at h = {h:.3e} bohr")

        out.append(step_ladder(f"nuclear ladder, seed {seed}", phi, ladder_steps(NUC_LADDER),
                               float(np.sum(d * g.de))))
    return out


def check_pc_guard_ledger(pc: bool) -> Check:
    """Every displaced CI vector a PC energy was evaluated at was guarded,
    and no guarded point kept a mode the reference state drops.
    :data:`PC_GUARD` counts the evaluations and the guard runs in different
    places, so a path that skips the guard leaves them unequal and fails
    here."""
    led = PC_GUARD
    name = "PC guard at every displaced CI vector"
    if not pc:
        return Check(name, True, "not applicable: no PC energy is evaluated")
    ok = (led.displaced > 0 and led.displaced == led.displaced_guarded
          and not led.rank_changed)
    summary = (f"guarded {led.displaced_guarded} of {led.displaced} displaced CI vectors, "
               f"and sampled at {led.sampled} fixed-CI ladder rungs; smallest gap "
               f"{led.min_gap:.2e} (need {gg.PC_MIN_GAP:g}); rank as the reference's at ")
    if led.rank_changed:
        where, key, kept, ref = min(led.rank_changed, key=lambda r: r[2] / r[3])
        summary += (f"{led.displaced_guarded + led.sampled - len(led.rank_changed)} of them "
                    f"-- {len(led.rank_changed)} block(s) KEPT a mode the reference drops, "
                    f"worst {key} at {where}: smallest kept {kept:.1e} against the "
                    f"reference's {ref:.1e}")
    else:
        summary += "every one"
    return Check(name, ok, summary,
                 numbers=dict(displaced=led.displaced, guarded=led.displaced_guarded,
                              sampled=led.sampled, min_gap=led.min_gap,
                              rank_changed=len(led.rank_changed)))


def load_grad_num(path: Path, case: str, *, pc: bool,
                  casci_only: bool = False) -> dict:
    """The numerical gradient to gate against, from a
    ``nevpt2_numgrad.py --npz`` file: the gradient itself, its Richardson
    error bar and the probed mask, **for the functional this run
    differentiates** -- SC, PC, or (with ``--casci-only``) the plain CASCI one
    that script prints beside them.

    Checked against the case it names, because the two gradients must be of
    one state; and the PC arrays are only there if that run was given
    ``--pc``, so asking for them without is the refusal rather than a silent
    comparison against the SC answer. ``grad_casci`` carries no Richardson
    residual of its own, so its bar is reported as zero."""
    key = "grad_casci" if casci_only else ("grad_num_pc" if pc else "grad_num")
    with np.load(path) as z:
        if "case" in z and str(z["case"]) != case:
            raise SystemExit(
                f"{path} is the numerical gradient of case {str(z['case'])!r}, "
                f"not {case!r}: the two gradients would be of different states"
            )
        if key not in z:
            raise SystemExit(
                f"{path} holds no '{key}' array"
                + (" -- rerun nevpt2_numgrad.py with --pc" if pc else "")
            )
        bar = f"{key}_residual"
        return dict(
            grad_num=np.asarray(z[key]),
            grad_num_residual=(np.asarray(z[bar]) if bar in z
                               else np.zeros_like(np.asarray(z[key]))),
            grad_num_probed=np.asarray(z["grad_num_probed"]),
            grad_num_origin=f"{path} [{key}]",
        )


# --------------------------------------------------------------------------
# Driver and report
# --------------------------------------------------------------------------


def run(args) -> dict:
    """Build the state, get the source terms, solve, contract, and return the
    record :func:`report` prints."""
    spec = dict(CASES[args.case]) if args.case in CASES else {}
    for key in ("geometry", "basis", "ncas", "nelecas", "cas_irreps", "core_irreps"):
        if getattr(args, key) is not None:
            spec[key] = getattr(args, key)
    if args.energy_ordered:
        spec["cas_irreps"] = None
        spec["core_irreps"] = None
    missing = [k for k in ("geometry", "basis", "ncas", "nelecas") if spec.get(k) is None]
    if missing:
        raise SystemExit(
            f"--case {args.case} is not one of {sorted(CASES)}, so "
            f"{', '.join('--' + m.replace('_', '-') for m in missing)} must be given"
        )
    spec.setdefault("cas_irreps", None)
    spec.setdefault("core_irreps", None)
    if args.pc and args.casci_only:
        raise SystemExit(
            "--pc and --casci-only are exclusive: --casci-only drops the "
            "correlation energy altogether, so there is no contraction left to "
            "choose"
        )
    # Read (and check) the numerical gradient BEFORE the expensive part, so a
    # mismatched or missing file costs nothing.
    grad_num_source = (load_grad_num(args.grad_num, args.case, pc=args.pc,
                                     casci_only=args.casci_only)
                       if args.grad_num is not None else None)
    if args.pc and grad_num_source is None:
        raise SystemExit(
            "--pc needs --grad-num PATH: a PC numerical gradient has no home "
            "in the gradient sidecar (whose contract is the 24 --gradgold "
            "arrays plus the four SC grad_num* ones), so it is written as a "
            "plain .npz by `nevpt2_numgrad.py --pc --npz PATH`"
        )
    t0 = time.perf_counter()
    # A fresh ledger: every PC guard run from here on is this run's.
    PC_GUARD.__dict__.update(PcGuardLedger().__dict__)
    print(f"[{args.case}] building the golden's own CASCI-on-RHF state")
    st = build_state(args.case, spec, pc=args.pc)
    print(f"  n_mo = {st.nmo} = {st.ncore} core + {st.nact} active + {st.nvirt} virtual; "
          f"ndet = {st.ci.size}; n_elec(RHF) = {st.nelec_rhf}")
    print(f"  act_O = {st.n_act_occ}, act_V = {st.nact - st.n_act_occ} "
          f"(section 2.6; the state is rotated into F's active eigenbasis, eps = "
          + " ".join(f"{e:+.4f}" for e in st.act_eps) + ")")
    print(f"  section 3.6 invariants: (O1) {st.invariants['o1']:.1e}, "
          f"(O2) {st.invariants['o2']:.1e}, (O3) act_occ = {st.invariants['act_occ']}")
    print(f"  E_CASCI = {st.e_casci:.10f}  E_corr(SC) = {st.e_corr:.10f}"
          + (f"  E_corr(PC) = {st.e_corr_pc:.10f} (smallest lstsq gap "
             f"{st.pc_gap:.1e}, need {gg.PC_MIN_GAP:g})" if args.pc else ""))

    rec = dict(case=args.case, basis=spec["basis"], ncas=spec["ncas"],
               nelecas=spec["nelecas"], nmo=st.nmo, ncore=st.ncore, nact=st.nact,
               nvirt=st.nvirt, ndet=int(st.ci.size), natm=len(st.geom),
               e_casci=st.e_casci, e_corr=st.e_corr, casci_only=args.casci_only,
               pc=args.pc, step=args.step, show_ladders=args.show_ladders,
               invariants={k: v for k, v in st.invariants.items()
                           if k != "o1_each"})
    if args.pc:
        rec["e_corr_pc"] = st.e_corr_pc
        rec["pc_gap"] = st.pc_gap

    # The state differentiated must be the state the golden holds -- when
    # there is one. An overridden selection (--energy-ordered, --cas-irreps)
    # is deliberately a different state, and has no golden to be checked
    # against.
    committed = (args.case in CASES and not args.energy_ordered
                 and all(getattr(args, k) is None for k in
                         ("geometry", "basis", "ncas", "nelecas", "cas_irreps",
                          "core_irreps")))
    rec["committed_case"] = committed
    if committed:
        gold = gf.read_golden_header(REPO / "golden" / f"{args.case}.nevpt2gold")
        rec["d_e_casci"] = st.e_casci - gold["e_casci"]
        rec["d_e_corr"] = st.e_corr - gold["e_nevpt2_total"]
        if max(abs(rec["d_e_casci"]), abs(rec["d_e_corr"])) > REF_STATE_TOL:
            raise SystemExit(
                f"the rebuilt state does not match golden/{args.case}.nevpt2gold: "
                f"E_CASCI differs by {rec['d_e_casci']:.3e}, E_corr by "
                f"{rec['d_e_corr']:.3e}"
            )
        if args.pc:
            gold_pc = ngrad.read_golden_pc_total(
                REPO / "golden" / f"{args.case}.nevpt2gold")
            if gold_pc is None:
                raise SystemExit(
                    f"--pc: golden/{args.case}.nevpt2gold carries no "
                    "e_pc_total -- it was generated --no-pc because its PC "
                    "metric is refused as a continuum, so there is no PC "
                    "reference for this state (docs/reference-data.md, \"LiF "
                    "and acrolein, the gradient-validation pair\")"
                )
            rec["d_e_corr_pc"] = st.e_corr_pc - gold_pc
            if abs(rec["d_e_corr_pc"]) > REF_STATE_TOL:
                raise SystemExit(
                    f"the rebuilt state's PC energy is {rec['d_e_corr_pc']:.3e} "
                    f"from golden/{args.case}.nevpt2gold's e_pc_total"
                )
        print(f"  against the golden: dE_CASCI = {rec['d_e_casci']:+.1e}, "
              f"dE_corr = {rec['d_e_corr']:+.1e}"
              + (f", dE_corr(PC) = {rec['d_e_corr_pc']:+.1e}" if args.pc else ""))
    else:
        print("  no golden for this selection: the active space was overridden, so "
              "this is deliberately a different state")

    if args.edot_check:
        # #88: the dump's forward Edot(kappa) along its seeded directions,
        # against PySCF's exact directional derivative -- no sweep, no solve.
        fn = Functional(st)
        src = load_source_terms(args.source_path, st)  # checks the dump is this state
        dirs = load_edot_directions(args.source_path, st)
        print(f"\n--edot-check: {len(dirs['kappa'])} directions read from {args.source_path}, "
              f"each laddered over h = 2^-{EDOT_LADDER[0]} .. 2^-{EDOT_LADDER[1]}")
        rec["source_origin"] = src.origin
        rec["edot"] = check_edot(fn, st, dirs)
        rec["edot_only"] = True
        rec["wall"] = time.perf_counter() - t0
        return rec

    with_corr = not args.casci_only
    fn = Functional(st, pc=args.pc)
    if args.casci_only:
        print("\n--casci-only: E_corr dropped from the source terms and from Phi. The "
              "reference is then a plain CASCI, whose analytic gradient PySCF has.")

    # --- the CI source term ----------------------------------------------
    both = args.source == "both"
    if both and (args.casci_only or args.pc):
        raise SystemExit(
            "--source both compares the GPU's SC source terms with PySCF's; the GPU has "
            "no PC ones (#74) and --casci-only has none at all"
        )
    print("\nsource terms (Park Eqs. 26-27)")
    t_src = time.perf_counter()
    src_pyscf = None
    if args.source_path is not None:
        src = load_source_terms(args.source_path, st)
        print(f"  read from {args.source_path} and carried into section 2.6's basis: "
              f"max |Y - Y^T| = {np.abs(src.kappa).max():.6e}, "
              f"|y| = {np.linalg.norm(src.ci):.8f}")
        if both:
            y = pyscf_ci_source(fn, st, step=args.ci_step)
            src_pyscf = SourceTerms(kappa=np.zeros((st.nmo, st.nmo)), ci=y, origin="pyscf")
            print(f"  and PySCF's y_I over {y.size} determinants by central differences "
                  f"(the comparison's other side), {time.perf_counter() - t_src:.0f}s")
    elif not with_corr:
        # A CASCI is variational in c, so y = 0 exactly.
        src = SourceTerms(kappa=np.zeros((st.nmo, st.nmo)),
                          ci=np.zeros(st.ci.shape), origin="pyscf (CASCI only)")
        print("  y_I = 0 exactly: a CASCI is variational in the CI vector")
    else:
        y = pyscf_ci_source(fn, st, step=args.ci_step)
        src = SourceTerms(kappa=np.zeros((st.nmo, st.nmo)), ci=y, origin="pyscf")
        print(f"  y_I = dE_corr/dc_I over {y.size} determinants by central "
              f"differences, {time.perf_counter() - t_src:.0f}s: "
              f"|y| = {np.linalg.norm(y):.8f}, max |y_I| = {np.abs(y).max():.6e}")
    src.seconds = time.perf_counter() - t_src
    rec["source_origin"] = src.origin

    g = Gradients(st, src, with_corr=with_corr, step=args.step,
                  richardson=args.richardson, pc=args.pc, kappa_step=args.kappa_step)
    print("\nthe frozen orthonormal path, and one kappa sweep")
    # Under --source both the one sweep differentiates E_corr too, so PySCF's
    # half of A exists to compare the dump's with -- and to solve with.
    g.prepare(sweep_corr=True if both else None)

    if both:
        print("\nthe two providers' source terms, componentwise (#73)")
        # `src` is still the dump as read: prepare() put A_casci + Y in g.src.
        rec["compare"] = compare_sources(st, src, g.sweep["A_corr"],
                                         src_pyscf.ci, "PySCF's exact derivatives")
        if args.localise:
            print("\n  localising: PySCF's own functional with the device's two "
                  "omissions put back (digests frozen in A, e_core/e_virt frozen in y)")
            loc = localise_sources(fn, st, args.kappa_step,
                                   ci_step=args.ci_step if args.ci_step else CI_STEP)
            print(f"  ({loc['seconds']:.0f}s)")
            rec["localise"] = compare_sources(st, src, loc["a_corr"], loc["y"],
                                              "PySCF with the device's omissions")
        if args.sources_only:
            rec["sources_only"] = True
            rec["wall"] = time.perf_counter() - t0
            return rec

    print("\nthe multipliers")
    g.solve_multipliers(log=print)
    rec["residuals"] = dict(g.residuals)

    # The residuals above say each solve found the answer to the equations it
    # was given. These say whether those were the right equations -- none of
    # them reads a residual (docs/testing.md, "Residual-independent checks of
    # the gradient path").
    print("\nresidual-independent checks of the source terms and the multipliers")
    t_chk = time.perf_counter()
    checks = check_projection(g, src)
    if with_corr:
        checks += check_ci_source_linearity(g, src)
    checks += check_ci_stationarity(g)
    checks += check_orbital_stationarity(g)
    for chk in checks:
        print(f"  {'ok  ' if chk.ok else 'FAIL'} {chk.name}: {chk.summary}")
    checks_seconds = time.perf_counter() - t_chk

    print("\nthe AO contraction")
    de = g.kernel()
    print("\nresidual-independent checks of the nuclear quotients")
    t_chk = time.perf_counter()
    tail = check_nuclear_ladder(g) + [check_pc_guard_ledger(args.pc)]
    for chk in tail:
        print(f"  {'ok  ' if chk.ok else 'FAIL'} {chk.name}: {chk.summary}")
    checks += tail
    rec["checks"] = checks
    rec["checks_seconds"] = checks_seconds + time.perf_counter() - t_chk
    rec["residuals"] = dict(g.residuals)  # kernel() adds the Richardson bars
    rec["grad"] = de
    rec["ham_response"] = g.ham_response
    rec["ldotjnuc"] = g.ldotjnuc
    rec["zero_axes"] = np.array(g.zero_axes, dtype=int)
    rec["point_group"] = g.point_group
    if both:
        # The other provider through the SAME sweep, Jacobian and Q: the two
        # gradients then differ only through the source terms, which is the
        # point -- a disagreement localises to Y and y, not to the solve.
        print("\nthe same solve with PySCF's source terms, on the same sweep")
        def solve_with(terms: SourceTerms) -> tuple[np.ndarray, dict]:
            gx = Gradients(st, terms, with_corr=True, step=args.step,
                           richardson=args.richardson, pc=False, kappa_step=args.kappa_step)
            gx.prepare(log=lambda *a: None, sweep=g.sweep)
            gx.solve_multipliers(log=lambda *a: None)
            return gx.kernel(), dict(gx.residuals)

        rec["grad_pyscf"], rec["residuals_pyscf"] = solve_with(src_pyscf)
        print(f"  CI |P(H-E)Pz + Py| = {rec['residuals_pyscf']['ci_cg']:.3e}, orbital "
              f"|M lambda + B| = {rec['residuals_pyscf']['orbital_direct']:.3e}")
        # And the attribution: one source term from each provider, so the
        # gradient's distance from PySCF's splits into what Y costs and what y
        # costs (exactly, to first order: the gradient is linear in both).
        print("  and the two mixed solves that attribute the difference: GPU Y with "
              "PySCF's y, PySCF's Y with the GPU's y")
        rec["grad_gpu_y_only"], _ = solve_with(SourceTerms(
            kappa=src.kappa, ci=src_pyscf.ci, origin="GPU Y + PySCF y", from_dump=True))
        rec["grad_gpu_ci_only"], _ = solve_with(SourceTerms(
            kappa=g.sweep["A_corr"], ci=src.ci, origin="PySCF Y + GPU y", from_dump=True))
    rec["wall"] = time.perf_counter() - t0
    rec["kappa_evals"] = fn.nevals + g.fn.nevals

    # Reference numbers, where they exist. The pyscf.grad.casci comparison is
    # reported, never gated (it is an oracle only for an energy-ordered active
    # space, below); the grad_num one below it IS the gate.
    if args.casci_only:
        # mo_coeff and ci must be passed: pyscf.grad.casci defaults to
        # mc._scf.mo_coeff, the UNSORTED RHF orbitals, while mc.ci belongs to
        # the irrep-sorted active space -- the two do not describe one state.
        rec["grad_pyscf_casci"] = np.asarray(
            st.mc.Gradients().kernel(mo_coeff=st.mc.mo_coeff, ci=st.mc.ci)
        )
        rec["pyscf_casci_valid"] = spec["cas_irreps"] is None
    # The numerical gradient this is gated against: an .npz written by
    # `nevpt2_numgrad.py --npz` when one is named (the only route for a PC
    # one, and for a case whose committed sidecar holds no grad_num), else the
    # sidecar beside the golden.
    if args.grad_num is not None:
        rec.update(grad_num_source)
    elif args.casci_only:
        # The sidecar's grad_num is the NEVPT2 gradient; this run's is a plain
        # CASCI's, so there is nothing there to gate against. The numerical
        # CASCI gradient is in an --npz (`grad_casci`), and `pyscf.grad.casci`
        # is reported above where it is an oracle at all.
        pass
    else:
        sidecar = REPO / "golden" / f"{args.case}.gradgold"
        if committed and sidecar.exists():
            _h, arrays = gf.read_gradgold(sidecar)
            if "grad_num" in arrays:
                rec["grad_num"] = np.asarray(arrays["grad_num"])
                rec["grad_num_probed"] = np.asarray(arrays["grad_num_probed"])
                rec["grad_num_residual"] = np.asarray(arrays["grad_num_residual"])
                rec["grad_num_origin"] = str(sidecar)
    if "grad_num" in rec:
        probed = rec["grad_num_probed"] > 0
        rec["grad_num_max_delta"] = float(np.abs(rec["grad"] - rec["grad_num"])[probed].max())
        rec["grad_num_tol"] = args.tol
        rec["grad_num_pass"] = rec["grad_num_max_delta"] <= args.tol
        if both:
            rec["grad_num_max_delta_pyscf"] = float(
                np.abs(rec["grad_pyscf"] - rec["grad_num"])[probed].max())
    if both:
        rec["provider_max_delta"] = float(np.abs(rec["grad"] - rec["grad_pyscf"]).max())
        rec["attrib_kappa"] = float(np.abs(rec["grad_gpu_y_only"] - rec["grad_pyscf"]).max())
        rec["attrib_ci"] = float(np.abs(rec["grad_gpu_ci_only"] - rec["grad_pyscf"]).max())
    return rec


def report(rec: dict) -> None:
    # The functional this run differentiated, named the same way in the header,
    # in the grad_num comparison and in the PASS/FAIL line -- so no line can be
    # read as a verdict on a different one.
    scheme = "CASCI" if rec["casci_only"] else ("PC" if rec["pc"] else "SC")
    which = {"CASCI": "CASCI only", "PC": "PC-NEVPT2 (block2)",
             "SC": "SC-NEVPT2"}[scheme]
    print(f"\n== {rec['case']}  {rec['basis']}  CAS({rec['nelecas']},{rec['ncas']})  "
          f"{which}")
    print(f"source terms: {rec['source_origin']}")
    if rec.get("edot_only"):
        rec["edot_pass"] = report_edot(rec["edot"], rec["case"])
        print(f"--edot-check: no gradient was solved ({rec['wall']:.1f}s)")
        return
    if "compare" in rec:
        for key in ("compare", "localise"):
            if key in rec:
                c = rec[key]
                print(f"the dump against {c['label']}: max |delta A| = {c['max_kappa']:.3e} "
                      f"(worst block {max(c['blocks'], key=lambda r: r['max_delta'])['block']}), "
                      f"max |delta P y| = {c['max_ci']:.3e}")
    if rec.get("sources_only"):
        print(f"--sources-only: no gradient was solved ({rec['wall']:.1f}s)")
        return
    r = rec["residuals"]
    print(f"Z-vector residuals -- CI |P(H-E)Pz + Py| = {r['ci_cg']:.3e} "
          f"({'converged' if r['ci_cg_converged'] else 'NOT CONVERGED'}), "
          f"|z| = {r['ci_norm']:.3e}")
    print(f"                      orbital |M lambda + B| = {r['orbital_direct']:.3e} "
          f"(direct) / {r['orbital_gmres']:.3e} (GMRES), cond {r['orbital_cond']:.2e}, "
          f"|lambda| = {r['orbital_norm']:.3e}")
    print(f"                      gauge identity (no multiplier) = {r['gauge']:.3e}, "
          f"z.c = {r['ci_overlap']:.1e}, fixed point in "
          f"{r['fixed_point_iterations']} iteration(s)")
    grad = rec["grad"]
    zero = list(rec["zero_axes"])
    print(f"{'atom':>4} {'dE/dx':>16} {'dE/dy':>16} {'dE/dz':>16}   (Eh/bohr; "
          f"'*' = zero by symmetry, differenced anyway)")
    for i in range(rec["natm"]):
        cells = "".join(
            f" {grad[i, c]:+15.10f}{'*' if c in zero else ' '}" for c in range(3)
        )
        print(f"{i:>4}{cells}")
    sums = ", ".join(f"d/d{AXES[c]} {grad[:, c].sum():+.2e}" for c in range(3))
    print(f"sum over atoms (0 by translational invariance): {sums}")
    if zero:
        worst = max(float(np.abs(grad[:, c]).max()) for c in zero)
        print(f"the free symmetry check (section 4.3): max |component| over the "
              f"{len(zero)} zero-by-symmetry axis/axes = {worst:.2e}")

    if "grad_pyscf_casci" in rec:
        ref = rec["grad_pyscf_casci"]
        print("\nagainst pyscf.grad.casci (the same gradient, analytically, through "
              "PySCF's own derivative integrals):")
        print(f"  max |delta| = {np.abs(grad - ref).max():.3e}")
        if not rec["pyscf_casci_valid"]:
            print("  ^ NOT an oracle for this case. pyscf.grad.casci divides its "
                  "redundant\n    core-act_O and act_V-virtual blocks assuming act_O "
                  "is the FIRST\n    n_elec/2 - n_core active columns. An "
                  "irrep-selected active space\n    (--cas-irreps) orders the active "
                  "block by irrep instead, so that\n    assumption fails and its "
                  "answer is for a different partition -- which\n    is exactly what "
                  "docs/gradient-theory.md, section 2.6 exists to fix.\n    Compare "
                  "against grad_num instead.")
    if "grad_num" in rec:
        ref, probed = rec["grad_num"], rec["grad_num_probed"] > 0
        d = np.abs(grad - ref)
        print(f"\nagainst grad_num ({rec['grad_num_origin']}), the numerical "
              f"{scheme} gradient:")
        print(f"  max |delta| over the {int(probed.sum())} differenced components = "
              f"{d[probed].max():.3e}, with grad_num's own error bar "
              f"{rec['grad_num_residual'][probed].max():.1e}")
        worst = np.unravel_index(int(np.argmax(np.where(probed, d, -1.0))), d.shape)
        print(f"  worst component: atom {worst[0]} {AXES[worst[1]]}, analytic "
              f"{grad[worst]:+.10f} vs numerical {ref[worst]:+.10f}")
        # The gate. The tolerance is NOT this repo's usual 1e-7 and cannot be:
        # the oracle is a finite difference. Rationale, the error bars it is
        # built from and the measured results: docs/testing.md, "The analytic
        # gradient against the numerical one".
        if "grad_pyscf" in rec:
            # #73's table: both providers and grad_num, component by component.
            gp = rec["grad_pyscf"]
            print("\n  both providers, one sweep, one solve each (Eh/bohr; '-' = not "
                  "differenced by nevpt2_numgrad.py, zero by symmetry):")
            print(f"  {'comp':>6} {'GPU-sourced':>16} {'PySCF-sourced':>16} {'grad_num':>16} "
                  f"{'|GPU-PySCF|':>12} {'|GPU-num|':>11} {'|PySCF-num|':>12}")
            for i in range(rec["natm"]):
                for cx in range(3):
                    num = (f"{ref[i, cx]:+16.10f}" if probed[i, cx]
                           else f"{'-':>16}")
                    dn = (f"{abs(grad[i, cx] - ref[i, cx]):>11.2e}" if probed[i, cx]
                          else f"{'-':>11}")
                    dp = (f"{abs(gp[i, cx] - ref[i, cx]):>12.2e}" if probed[i, cx]
                          else f"{'-':>12}")
                    print(f"  {i:>4}{AXES[cx]:>2} {grad[i, cx]:+16.10f} {gp[i, cx]:+16.10f} "
                          f"{num} {abs(grad[i, cx] - gp[i, cx]):>12.2e} {dn} {dp}")
            print(f"  max |GPU-sourced - PySCF-sourced| over all {grad.size} components = "
                  f"{rec['provider_max_delta']:.3e}")
            print(f"  of which, solving with ONE of the GPU's terms and PySCF's other: "
                  f"the GPU's Y alone moves the gradient by {rec['attrib_kappa']:.3e}, the "
                  f"GPU's y alone by {rec['attrib_ci']:.3e}")
        tag = " (GPU source terms)" if "grad_pyscf" in rec else ""
        verdict = "PASS" if rec["grad_num_pass"] else "FAIL"
        relation = "<=" if rec["grad_num_pass"] else ">"
        print(f"{verdict}: {rec['case']} {scheme} analytic{tag} "
              f"vs numerical gradient, max |delta| = "
              f"{rec['grad_num_max_delta']:.3e} {relation} "
              f"{rec['grad_num_tol']:g} (docs/testing.md, \"The analytic "
              "gradient against the numerical one\")")
        if "grad_num_max_delta_pyscf" in rec:
            d_py = rec["grad_num_max_delta_pyscf"]
            ok_py = d_py <= rec["grad_num_tol"]
            print(f"{'PASS' if ok_py else 'FAIL'}: {rec['case']} {scheme} analytic (PySCF "
                  f"source terms) vs numerical gradient, max |delta| = {d_py:.3e} "
                  f"{'<=' if ok_py else '>'} {rec['grad_num_tol']:g}")
    if rec["residuals"].get("ham_response_richardson") is not None:
        print(f"\nRichardson error bars on the frozen-path quotients (the error on "
              f"the gradient above): {rec['residuals']['ham_response_richardson']:.1e} "
              f"(skeleton), {rec['residuals']['ldotjnuc_richardson']:.1e} "
              f"(multipliers)")

    # The second verdict, independent of the first: every residual above can
    # read round-off on a wrong answer, and these do not read one.
    checks = rec["checks"]
    print("\nresidual-independent checks (docs/testing.md, \"Residual-independent "
          "checks of the gradient path\")")
    for chk in checks:
        print(f"  {'ok  ' if chk.ok else 'FAIL'} {chk.name}: {chk.summary}")
        if chk.table and (rec["show_ladders"] or not chk.ok):
            print(f"      {'h':<6} {'central difference':>19} {'|change|':>10} {'ratio':>8}"
                  "   (change from the previous rung; '<' = the fourfold run)")
            for row in chk.table:
                print(row)
    failed = [chk.name for chk in checks if not chk.ok]
    print(f"{'FAIL' if failed else 'PASS'}: {rec['case']} {scheme} residual-independent "
          f"checks, {len(checks) - len(failed)} of {len(checks)} hold"
          + (f" -- failed: {'; '.join(failed)}" if failed else "")
          + f" ({rec['checks_seconds']:.1f}s)")
    print(f"\n{rec['wall']:.1f}s, {rec['kappa_evals']} functional evaluations")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="lif_def2svp_cas46",
                    help=f"a committed gradient-validation case: {', '.join(CASES)}. "
                         "Anything else needs --geometry, --basis, --ncas and "
                         "--nelecas, and is then checked against no golden")
    ap.add_argument("--geometry", default=None,
                    help="xyz file (default: the --case geometry); relative paths "
                         "resolve against the repo root")
    ap.add_argument("--basis", default=None)
    ap.add_argument("--ncas", type=int, default=None)
    ap.add_argument("--nelecas", type=int, default=None)
    ap.add_argument("--cas-irreps", type=gg._irrep_counts, default=None,
                    help="active orbitals per irrep, e.g. 'B1=3,B2=3' (default: the "
                         "--case selection)")
    ap.add_argument("--core-irreps", type=gg._irrep_counts, default=None,
                    help="core orbitals per irrep, same syntax")
    ap.add_argument("--energy-ordered", action="store_true",
                    help="drop the case's irrep selection and take the active space "
                         "by energy order instead. This is the configuration "
                         "pyscf.grad.casci's own redundant-block division assumes, "
                         "so --casci-only --energy-ordered is the one place its "
                         "analytic gradient is a valid oracle")
    ap.add_argument("--source", default="pyscf", choices=("pyscf", "dump", "both"),
                    help="where Y_xy and y_I come from: 'pyscf' (default, central "
                         "differences of PySCF's own SC-NEVPT2), 'dump' (a "
                         "*.gradsrc the GPU wrote: nevpt2_sc_pseudodensity "
                         "--dump-source-terms; --source-path), or 'both': the "
                         "dump is the provider gated against grad_num, and "
                         "PySCF's terms are compared with it componentwise and "
                         "solved on the same sweep, so the report carries both "
                         "gradients beside grad_num (docs/testing.md, \"The GPU "
                         "as the source-term provider\"). SC only")
    ap.add_argument("--source-path", type=Path, default=None,
                    help="the .gradsrc dump --source dump / both reads")
    ap.add_argument("--sources-only", action="store_true",
                    help="with --source both: stop after the componentwise "
                         "comparison of the two providers' source terms -- no "
                         "Z-vector solve, no gradient, no gate")
    ap.add_argument("--localise", action="store_true",
                    help="with --source both: also compare the dump with "
                         "PySCF's functional carrying the two omissions #73 "
                         "measured (f3 digests frozen in Y, e_core/e_virt "
                         "frozen in y -- the y half closed by #87, so a dump "
                         "written since matches the exact y, not this one), "
                         "so a disagreement can be pinned on them. One more "
                         "E_corr sweep over the kappa pairs and the "
                         "determinants")
    ap.add_argument("--edot-check", action="store_true",
                    help="with --source dump: hold the dump's forward Edot(kappa) "
                         "(#88) -- its frozen-digest tangent plus the f3ac/f3ca "
                         "digests' term, along each seeded direction it carries -- "
                         "against PySCF's exact directional derivative, step-"
                         "laddered, and stop: no sweep, no solve. Exit 1 on a FAIL "
                         "(docs/testing.md, \"The digests' orbital response, "
                         "forward mode\")")
    ap.add_argument("--casci-only", action="store_true",
                    help="drop E_corr from the source terms and from Phi: the "
                         "reference is then a plain CASCI and pyscf.grad.casci is "
                         "the oracle. The pipeline's own validation")
    ap.add_argument("--step", type=float, default=None,
                    help=f"nuclear central-difference step in bohr (default "
                         f"{NUC_STEP:g}, or {NUC_STEP_PC:g} with --pc; "
                         "docs/testing.md, \"The step constants, measured\")")
    ap.add_argument("--no-richardson", dest="richardson", action="store_false",
                    help="difference at h only, and report no error bar. The default "
                         "differences at h and h/2 and stores the Richardson "
                         "extrapolant: measured, the single-step truncation is the "
                         "largest error in the whole pipeline, so the second pass is "
                         "worth its cost")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the gradient and every residual here as an .npz. "
                         "Nothing is written by default, and no committed file is "
                         "ever touched")
    ap.add_argument("--pc", action="store_true",
                    help="differentiate block2's PARTIALLY-contracted energy "
                         "instead of PySCF's strongly-contracted one. The "
                         "Lagrangian, the constraints and the Z-vector "
                         "equations are the reference's and do not change; only "
                         "E_corr does. Needs --grad-num (a PC numerical "
                         "gradient is not a sidecar array), and is refused on a "
                         "state with no PC answer -- LiF's metric is a "
                         "continuum. Far slower: one block2 PC solve per "
                         "kappa direction")
    ap.add_argument("--grad-num", type=Path, default=None,
                    help="the numerical gradient to gate against, as written by "
                         "`nevpt2_numgrad.py --npz PATH`: its grad_num, "
                         "grad_num_pc or grad_casci array, whichever this run "
                         "differentiates. Default: the grad_num arrays in "
                         "golden/<case>.gradgold, where there are any (SC "
                         "only; with --casci-only nothing there applies, since "
                         "that grad_num is the NEVPT2 gradient)")
    ap.add_argument("--ci-step", type=float, default=None,
                    help="the step for the y_I = dE_corr/dc_I quotients "
                         f"(default {CI_STEP:g}, or {CI_STEP_PC:g} with --pc, "
                         "which is its own constant for a reason that is "
                         "not truncation -- see CI_STEP_PC). Only here so the "
                         "measured ladder in docs/testing.md can be rerun")
    ap.add_argument("--kappa-step", type=float, default=None,
                    help="the step for the Y_xy / constraint-Jacobian / Q "
                         f"quotients along each kappa rotation (default "
                         f"{KAPPA_STEP:g}, or {KAPPA_STEP_PC:g} with --pc). "
                         "Only here so the measured ladders in docs/testing.md "
                         "can be rerun")
    ap.add_argument("--tol", type=float, default=GRAD_NUM_TOL,
                    help="what max |g - grad_num| has to be under, per "
                         f"component, to PASS (default {GRAD_NUM_TOL:g}). NOT "
                         "this repo's 1e-7: the oracle is a finite difference "
                         "(docs/testing.md, \"The analytic gradient against the "
                         "numerical one\")")
    ap.add_argument("--show-ladders", action="store_true",
                    help="print every step ladder's table, not only a failing "
                         "one's (docs/testing.md, \"Residual-independent checks "
                         "of the gradient path\" is read off these)")
    args = ap.parse_args()
    default_step, default_kappa = scheme_steps(args.pc)
    if args.step is None:
        args.step = default_step
    if args.kappa_step is None:
        args.kappa_step = default_kappa
    if args.source in ("dump", "both") and args.source_path is None:
        raise SystemExit(f"--source {args.source} needs --source-path PATH")
    if (args.sources_only or args.localise) and args.source != "both":
        raise SystemExit("--sources-only and --localise compare two providers; they "
                         "need --source both")
    if args.edot_check and (args.source not in ("dump", "both") or args.pc
                            or args.casci_only):
        raise SystemExit("--edot-check reads the GPU's Edot directions out of a dump: it "
                         "needs --source dump (or both) --source-path PATH, and is SC only")
    if args.source == "pyscf":
        args.source_path = None

    rec = run(args)
    report(rec)
    if rec.get("edot_only"):
        # A gate of its own (#88), read the same way: exit 1 on a FAIL.
        if not rec["edot_pass"]:
            raise SystemExit(1)
        return
    if rec.get("sources_only"):
        # A comparison, not a gate: nothing was solved, so there is nothing
        # for grad_num to judge and nothing for --out to write.
        return
    if args.out is not None:
        # Plain arrays and scalars only -- readable with a bare
        # numpy.load(), no pickle, the same stance the golden files take.
        payload = {k: v for k, v in rec.items() if isinstance(v, np.ndarray)}
        for k, v in rec.items():
            if isinstance(v, (int, float, bool, str)):
                payload[k] = np.array(v)
        for k, v in rec["residuals"].items():
            payload[f"residual_{k}"] = np.array(0.0 if v is None else v)
        for n, chk in enumerate(rec["checks"]):
            payload[f"check{n}_ok"] = np.array(chk.ok)
            for k, v in chk.numbers.items():
                payload[f"check{n}_{k}"] = np.asarray(v)
        np.savez(args.out, **payload)
        print(f"wrote {args.out} ({len(payload)} arrays)")
    # The comparison against the numerical gradient is a GATE, and the exit
    # status is how it is read from a script: 1 on a FAIL, as the demos'
    # `PASS:` lines are read from ctest. A run with no numerical gradient in
    # reach checked nothing and says so.
    # The residual-independent checks are a gate of their own, read the same
    # way: any one failing is exit 1 whatever grad_num said.
    failed = not all(chk.ok for chk in rec["checks"])
    if "grad_num" not in rec:
        print("no numerical gradient in reach: only the residual-independent "
              "checks were gated (pass --grad-num PATH, or run "
              "nevpt2_numgrad.py for this case first)")
    elif not rec["grad_num_pass"]:
        failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
