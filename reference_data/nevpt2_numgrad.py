#!/usr/bin/env python3
r"""Numerical nuclear gradient of the SC- (and with ``--pc``, PC-) NEVPT2
energy, by displaced CASCI runs.

Writes ``grad_num`` into the **gradient sidecar** (``<case>.gradgold``) beside
a golden file, with the step size it was taken at and a Richardson residual
that bounds its own error: the reference an analytic NEVPT2 gradient
(docs/gradient-theory.md) is checked against. It is OFFLINE tooling, like
``block2_pc_probe.py`` -- no build and no test runs it -- and it touches no
``*.nevpt2gold`` byte, only the sidecar's.

The displaced states come from ``generate_golden._build_state`` itself
(imported, not copied), so every one of them carries the same single-threading,
the same 1e-12 Davidson tolerance, the same auto-detected point group and the
same irrep-selected active space the committed golden does. The energy
differentiated is ``E_CASCI + E_corr(SC-NEVPT2)``, PySCF's own per-class sum
(``generate_golden._pyscf_per_class``) -- the golden's SC answer, at a
displaced geometry.

**``--pc`` differences the partially-contracted energy too**, in the same
sweep: block2's ``WickICNEVPT2`` on each displaced state
(``generate_golden._block2_icmr``), gated per state by
``generate_golden._check_pc_conditioning`` exactly as the golden's own PC
fields are -- a displaced state whose PC metric is ungapped is refused, not
differenced. PySCF has no PC-NEVPT2, so block2 is the only reference
(docs/pc-nevpt2.md, "The block2 reference"), and **LiF has no PC answer at
all**: its metric is refused as a continuum, so ``--pc`` on it stops at the
reference state (docs/reference-data.md, "LiF and acrolein, the
gradient-validation pair"). One PC evaluation costs far more than the SC
sweep around it, so ``--pc`` is opt-in and it reports both gradients from the
one set of displaced states rather than being a separate run.

``--pc`` writes **no sidecar**: the sidecar's contract is the 24 arrays
``--gradgold`` writes plus the four SC ``grad_num*`` ones, and the committed
files hold exactly that. ``--npz PATH`` is how a PC (or SC) numerical
gradient leaves this script for ``nevpt2_grad.py --grad-num PATH`` to compare
against, without a committed byte moving.

**Why a numerical gradient is possible for this reference at all: it is a
CASCI, not a CASSCF.** RHF followed by a CASCI diagonalisation is
deterministic at every displaced geometry, so two displaced energies are two
values of one function and their difference quotient means something. A CASSCF
reference would converge to a differently-rotated solution of the same energy
at each displacement, and the numerical gradient would be garbage
(docs/gradient-theory.md, §1.1, and §4.4 for why that is a *precondition* for
this script rather than the hazard below). That is a statement about
*displaced* geometries: at one fixed geometry a fresh CASSCF reproduces bit
for bit, measured in docs/reference-data.md, "What 'CASSCF input' means here,
and why the golden state is CASCI". The golden's header says which reference
built its orbitals, so this is enforced and not merely assumed: a golden whose
``reference_kind`` is not the ``--reference`` asked for is **refused**
(:func:`_check_reference_kind`), before any energy is computed.

**``--reference casscf`` is the other route** (:func:`run_casscf`). Every
displaced state is re-OPTIMISED from a cold start through
``generate_golden.casscf_optimise`` -- the body of ``--casscf``, polish
included -- then finished (:func:`finish_casscf`), then given the golden's
own ``--mo-from`` CASCI + SC-NEVPT2. Cold start defines the oracle; a warm
start on the first coarse-step geometries is a free probe for a second
solution (:func:`warm_probe`). Every displaced state passes a **continuity
guard** before it is differenced -- an energy jump, an unconverged
optimiser, or an active space / CI vector that moved too far
(:func:`continuity_tripped`) -- and the reference must tie to the committed
``<case>_casscf.mogold`` and golden bit for bit. ``--directional N
--analytic PATH`` is the cheap **screen** (``<g, d>`` against a central
difference along N seeded in-plane directions), run before the
componentwise gate. ``--workers N`` runs the independent states as N
single-threaded processes, bit-identically to one. Each state's MO sidecar
and a ``states.json`` manifest stay in ``--states-dir``, so later consumers
(``pc_on_states.py``'s GPU PC energies, #126's block2 spot-checks) reuse
this sweep's states rather than optimising their
own. SC only, and it writes ``--npz`` only: the committed
``<case>_casscf.gradgold`` stays the 24 ``--gradgold`` arrays. Its PC twin
runs no sweep at all: ``--pc-states NPZ`` differences the GPU's PC energies
on a finished sweep's states with the same stencil
(:func:`difference_pc_states`, #126).
docs/reference-data.md, "The numerical gradient of a CASSCF reference", has
the measurements.

**In-plane displacements only, and the refusal.** The active space is selected
by irrep, and the irrep labels exist only as long as the point group does, so
a displacement that lowers the symmetry silently replaces the active space
with a different set of orbitals (docs/gradient-theory.md, §4.2 and §4.3).
This script therefore **refuses** any displacement whose detected point group
differs from the reference geometry's (:func:`_check_point_group`), and
refuses one whose core/active irrep labels differ even at an unchanged group
(:func:`_check_selection`), rather than produce a number from a different
active space. Nothing is lost: the components an out-of-plane displacement
would probe are exactly the ones that are zero by symmetry at a planar
geometry, so ``--components auto`` (the default) records them as the exact
zeros they are and says so, and they stay a free check on the analytic side
(§4.3). Asking for one explicitly -- ``--components all``, or a named
out-of-plane component -- is the refusal.

**The stencil.** Per component, six displaced energies: a central difference
at :data:`STEP_BOHR` and at its halves,
``g(h) = [E(+h) - E(-h)] / 2h``, then two Richardson extrapolations
``R(h) = [4 g(h/2) - g(h)] / 3``. ``grad_num`` is the finer extrapolant
``R(h/2)`` and ``grad_num_residual`` is ``|R(h/2) - R(h)|``, which is an
error bar on the stored number rather than the size of a correction to it.
The measured residuals are in docs/reference-data.md, "The numerical
gradient".

Usage::

    uv run python reference_data/nevpt2_numgrad.py --case lif_def2svp_cas46
    uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65

each rewriting ``golden/<case>.gradgold`` in place with the four ``grad_num*``
arrays appended (idempotently: a second run reproduces the file byte for
byte). ``--no-write`` computes and prints without touching the file,
``--out PATH`` writes elsewhere, and ``--components`` selects what to
differentiate::

    uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65 \
        --components 1x,1y --no-write        # two components, nothing written
    uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65 \
        --components all                     # refused: 0z breaks Cs
    uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65 \
        --pc --npz /tmp/acrolein.numgrad.npz # SC and PC, nothing written in golden/
    uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65 \
        --reference casscf --directional 4 --analytic /tmp/acrolein.grad.npz \
        --workers 12 --states-dir /tmp/acrolein_states --npz /tmp/screen.npz
    uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65 \
        --reference casscf --workers 12 --states-dir /tmp/acrolein_states \
        --npz /tmp/acrolein_casscf.numgrad.npz   # the gate, once the screen passes
    uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65 \
        --reference casscf --pc-states /tmp/acrolein_pc_states.npz \
        --npz /tmp/acrolein_casscf_pc.numgrad.npz  # PC, off pc_on_states.py's file
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads.
import generate_golden as gg

# isort: split
import argparse
import contextlib
import io
import json
import re
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import golden_format as gf
import numpy as np

# BOHR is PySCF's own Angstrom per bohr, imported rather than written out: the
# geometries are in Angstrom (reference_data/geometries/*.xyz) and the gradient
# is in Eh/bohr, the convention every analytic-gradient implementation this is
# compared against uses, so the displacements differ by exactly the constant
# PySCF built the molecule with.
from pyscf.data.nist import BOHR

#: The repository root: this file's parent's parent.
REPO = Path(__file__).resolve().parent.parent

#: The coarse central-difference step, in bohr; the stencil also uses h/2 and
#: h/4. Chosen from a measured ladder, 0.04 down to 0.000625 bohr on
#: acrolein's worst components (docs/reference-data.md, "The numerical
#: gradient", which tabulates it): the Richardson residual falls off with h
#: until the step is small enough for the energies' own convergence floor,
#: amplified by 1/2h, to show through, and then turns back up. This is one
#: halving above that minimum -- the extrapolant has converged and the
#: residual is not yet reading the floor.
#: Bit-reproducibility (forced by ``generate_golden``'s single-threading) is
#: not smoothness: the energies are exact functions of the geometry, but each
#: carries its converged-to tolerance, and a difference quotient divides that
#: by the step.
STEP_BOHR = 0.0025

#: How far the rebuilt reference state may sit from the golden it claims to
#: differentiate, in E_CASCI and in E_corr. The rebuilt state is the golden's
#: own (same :func:`generate_golden._build_state`, same settings), so this is
#: a wiring check, not a tolerance to be tuned: it is measured at 0.0, exactly
#: bit-identical, on both committed cases. A nonzero value means the geometry,
#: basis or active-space selection handed in here is not the one the golden
#: was built from, and the gradient would be of a different function.
REF_STATE_TOL = 1e-10

# -- the CASSCF reference (--reference casscf) --------------------------------
#
# Every displaced state is re-OPTIMISED, cold, through generate_golden's own
# optimiser (casscf_optimise, the body of --casscf), then gets the same
# deterministic CASCI on those orbitals the committed golden got (--mo-from).
# docs/reference-data.md, "The numerical gradient of a CASSCF reference", has
# every number quoted below.

#: The references this script differentiates; the golden's ``reference_kind``
#: must be the one asked for (:func:`_check_reference_kind`).
REFERENCES = ("casci", "casscf")

#: The finish (:func:`finish_casscf`), two stages from the generator's
#: converged state: ``mc1step`` with the polish's settings and ``ah_lindep``
#: at FINISH_AH_LINDEP, then ``mcscf.newton`` with its FCI solver's
#: ``lindep`` and ``conv_tol`` lowered, at this gradient threshold and macro
#: ceiling. ``mc1step`` converges the
#: orbitals against a CI vector whose residual floors near 1e-6, which leaves
#: the orbitals stationary for that vector and not for the accurate one: the
#: ``--mo-from`` gate, evaluated with the fresh 1e-12-Davidson CASCI, scatters
#: from 1.1e-7 to 1.2e-6 over acrolein's displaced states (and refuses two of
#: 24) while their optimiser's own ``|g_orb|`` is <= 1.1e-8, and E_corr --
#: first order in the orbitals -- carries ~1e-8 Eh of noise: a 6e-5 Eh/bohr
#: Richardson residual. Finished, the gate reads ~1e-9 and that direction's
#: residual is 1.1e-7 (docs/reference-data.md, "Why the displaced states are
#: finished"). The threshold is often not met to the letter -- newton stalls
#: between 4e-10 and 1e-9 -- which is why the ceiling is part of the
#: definition and the guard reads the norm reached, not the flag.
FINISH_CONV_TOL = 1e-13
FINISH_CONV_TOL_GRAD = 1e-9
FINISH_MAX_CYCLE_MACRO = 100
FINISH_AH_LINDEP = 1e-20
FINISH_FCI_LINDEP = 1e-16
FINISH_FCI_CONV_TOL = 1e-14

#: The continuity guard's three signals, refused past these. Absolute, and
#: calibrated on the shipped stencil (:data:`STEP_BOHR` and its halves):
#: each sits a decade or more above the largest value any step of the LiF and
#: acrolein sweeps measured and far below what a displacement that leaves the
#: reference's branch -- or is simply too large for the stencil to resolve --
#: produces (``casscf_repro_probe.py --displace`` measures both sides;
#: docs/reference-data.md, "The continuity guard's thresholds").
#:
#: ``energy``: ``|E_CASSCF(s) - E_CASSCF(0) - s <g_CASSCF, d>|``, the
#: displaced optimiser energy against its first-order prediction from
#: ``pyscf.grad.casscf`` at the reference. The cold-start run-to-run band at a
#: fixed geometry is exactly zero (docs/reference-data.md, "A fresh CASSCF
#: reproduces, measured"), so what is left is the second-order remainder.
CONTINUITY_E_TOL = 3e-5
#: ``overlap``: ``1 - sigma_min`` of the active-subspace overlap with the
#: reference geometry's active orbitals (carried into the displaced AO basis,
#: :func:`carry_orbitals`): an active orbital swapped with a core or virtual.
CONTINUITY_ACT_TOL = 1e-4
#: ``overlap`` too: ``1 - |<ci_0|ci(s)>|`` with the displaced CI vector
#: rotated into the reference's active basis (``fci.addons.transform_ci``):
#: a root flip, or a CI vector that moved by more than a smooth state does.
CONTINUITY_CI_TOL = 1e-5
#: ``gradient``: the generator's optimiser's own ``converged`` flag (its
#: ``conv_tol_grad``, generate_golden.CASSCF_CONV_TOL_GRAD), and the
#: finished state's own ``|g_orb|`` against that same 1e-7 -- measured at
#: <= 1e-9, so a state over it did not finish at all. The golden's
#: ``--mo-from`` gate (``CASSCF_GRAD_TOL`` on the fresh CASCI) also runs,
#: unchanged and refusing, on every displaced state; its number is recorded
#: in the log and the ``--npz``.

#: The directional screen's directions: how many, and the seed they are drawn
#: with (fixed, so the screen is one reproducible calculation).
SCREEN_DIRECTIONS = 4
SCREEN_SEED = 124

#: Declared wall-clock budgets, in seconds, for ``--workers 12`` (one per
#: physical core; docs/reference-data.md has the measured walls): the screen is
#: held to #105's 30-minute shape on every case; the componentwise gate, run
#: once the screen is green, gets its own. A bound, not a gate: a run over it
#: says so and does not fail (as ``nevpt2_grad.CASSCF_WALL_BUDGET``).
SCREEN_WALL_BUDGET = 1800.0
GATE_WALL_BUDGET = {
    "lif_def2svp_cas46": 600.0,
    # #125's degeneracy probe, declared before it ran from LiF's measured
    # wall (36 states, 20.4 s at --workers 12): the same 600 s.
    "n2_ccpvdz_cas44": 600.0,
    "acrolein_def2svp_cas65": 3600.0,
}

#: How many displaced geometries the cold-vs-warm probe re-optimises from the
#: reference's converged orbitals (:func:`warm_probe`).
WARM_PROBE_DEFAULT = 2

#: The cases with a committed golden + sidecar, keyed by their golden's name.
#: These MUST stay the ``--gradgold`` invocations in docs/reference-data.md,
#: "Regenerating the golden file": the gradient is of the state the sidecar
#: beside it describes, so a drifted geometry, basis or active space here
#: would produce a number for a state no file holds. ``n2.xyz`` carries
#: exactly that file's ``--atom "N 0 0 0; N 0 0 1.1"``, and the check against
#: the golden's two energies (:data:`REF_STATE_TOL`, measured at 0.0) is what
#: says so.
#:
#: **N2 is the deliberately symmetric case** (docs/testing.md, "The analytic
#: gradient against the numerical one"): D2h, six exactly degenerate inactive
#: pairs, and the molecule stays linear under every displacement, so every
#: component survives the point-group check. Its sidecar is committed and
#: holds **no** ``grad_num`` -- it stays the 24 arrays ``--gradgold`` writes --
#: so ``sidecar_grad_num`` is False for it and the in-place write is refused
#: rather than moving a committed file's bytes; ``--npz`` is the route.
CASES = {
    "lif_def2svp_cas46": dict(
        geometry="reference_data/geometries/lif.xyz",
        basis="def2-svp",
        ncas=6,
        nelecas=4,
        cas_irreps={"B1": 3, "B2": 3},
        core_irreps=None,
        sidecar_grad_num=True,
    ),
    "acrolein_def2svp_cas65": dict(
        geometry="reference_data/geometries/acrolein.xyz",
        basis="def2-svp",
        ncas=5,
        nelecas=6,
        cas_irreps={"A'": 1, 'A"': 4},
        core_irreps=None,
        sidecar_grad_num=True,
    ),
    "n2_ccpvdz_cas44": dict(
        geometry="reference_data/geometries/n2.xyz",
        basis="cc-pvdz",
        ncas=4,
        nelecas=4,
        cas_irreps=None,
        core_irreps=None,
        sidecar_grad_num=False,
    ),
    # Here for nevpt2_grad.py --source both --sources-only (#73): the second
    # size its GPU source terms are compared with PySCF's at. Its committed
    # sidecar carries no grad_num either, and no gradient gate is run on it.
    "n2_ccpvdz_cas88": dict(
        geometry="reference_data/geometries/n2.xyz",
        basis="cc-pvdz",
        ncas=8,
        nelecas=8,
        cas_irreps=None,
        core_irreps=None,
        sidecar_grad_num=False,
    ),
}

#: Component names, in the order they index a coordinate.
AXES = ("x", "y", "z")


def read_xyz(path: Path) -> list[tuple[str, tuple[float, float, float]]]:
    """An ``.xyz`` file as the ``[(symbol, (x, y, z))]`` list PySCF's ``gto.M``
    takes, Angstrom, exact: handing ``gto.M`` the parsed list rather than the
    path gives bit-identical bohr coordinates (checked below against the
    golden's own energies), and it is the form a displacement can be added
    to."""
    lines = path.read_text().splitlines()
    natm = int(lines[0].split()[0])
    geom = []
    for line in lines[2 : 2 + natm]:
        sym, x, y, z = line.split()[:4]
        geom.append((sym, (float(x), float(y), float(z))))
    assert len(geom) == natm, f"{path}: header says {natm} atoms, found {len(geom)}"
    return geom


def _check_reference_kind(kind: str, path: Path, reference: str = "casci") -> None:
    """Refuse a golden whose reference is not the one this run differentiates.

    ``--reference casci`` (the default) differences a CASCI on converged RHF
    orbitals: RHF followed by a CASCI diagonalisation is one deterministic
    function of the geometry, so two displaced energies are two values of it.
    A ``casscf`` golden is a different function -- its orbitals are a CASSCF
    stationary point, and committed orbitals carried to a displaced geometry
    are not the CASSCF solution there (docs/gradient-theory.md, §1.1, §4.4) --
    so differencing it on the CASCI route would differentiate the wrong
    functional. ``--reference casscf`` is that route: it re-optimises at every
    displaced geometry, and refuses a ``casci`` golden the same way. Refused,
    not warned: the number either mismatch would produce looks like a
    gradient. (``apps/integral_direct``'s ``--sc-tangent`` and
    ``--gradient-source`` refuse a casscf golden too, as an
    ``err_unsupported``.)"""
    if kind != reference:
        raise SystemExit(
            f"{path}: this golden's reference_kind is {kind!r}, and this run "
            f"differentiates --reference {reference}. "
            + ("A casci run would difference a CASCI on RHF orbitals, which "
               "is not the state a casscf golden holds: its orbitals are a "
               "CASSCF stationary point, and at a displaced geometry they "
               "have to be re-optimised, not carried over "
               "(docs/gradient-theory.md, §1.1 and §4.4). Pass --reference "
               "casscf."
               if kind == "casscf" else
               "A casscf run re-optimises the orbitals at every displaced "
               "geometry, which is not the function a casci golden holds. "
               "Pass --reference casci.")
            + " Refused rather than differenced."
        )


def read_golden_pc_total(path: Path) -> float | None:
    """The golden's ``e_pc_total`` -- block2's PC correlation energy on that
    state -- or ``None`` when the file carries no PC fields (``--no-pc``, e.g.
    LiF, whose PC metric is refused as a continuum). Scanned out of the array
    table by :func:`golden_format.read_golden`, so it reads either golden
    layout without knowing either."""
    _h, arrays = gf.read_golden(path, {"e_pc_total"})
    a = arrays.get("e_pc_total")
    return None if a is None else float(a.ravel()[0])


def pc_energy(mc, pt, *, check: bool = True) -> float:
    """block2's PC correlation energy on the state ``pt`` holds, as one
    number.

    ``check`` runs ``generate_golden._check_pc_conditioning`` on this state's
    own lstsq cuts -- the guard the golden's PC fields pass before they are
    written -- so a displaced state whose PC metric has gone ungapped is
    refused rather than differenced. It costs an SVD per PC block, which is
    nothing beside the solve itself, so it is on for every displaced state."""
    mc.mo_coeff, mc.ci, mc.mo_energy = pt.mo_coeff, pt.ci, pt.mo_energy
    gaps: list | None = [] if check else None
    pc = gg._block2_icmr(mc, "pc", dm4_cache={}, gaps=gaps)
    if gaps is not None:
        gg._check_pc_conditioning(gaps)
    return float(sum(pc.values()))


def displace(
    geom: list[tuple[str, tuple[float, float, float]]],
    iatom: int,
    comp: int,
    step_bohr: float,
) -> list[tuple[str, tuple[float, float, float]]]:
    """``geom`` with one Cartesian component of one atom moved by
    ``step_bohr`` bohr (the geometry is in Angstrom; the gradient is in
    Eh/bohr)."""
    moved = []
    for k, (sym, xyz) in enumerate(geom):
        v = list(xyz)
        if k == iatom:
            v[comp] += step_bohr * BOHR
        moved.append((sym, (v[0], v[1], v[2])))
    return moved


def point_group(geom, basis: str) -> tuple[str, str]:
    """The ``(detected, used)`` point group of a geometry: what PySCF detects
    and what ``generate_golden._build_mol`` ends up building it in (a linear
    group mapped through ``LINEAR_SUBGROUP``). Cheap -- symmetry detection
    only, no integrals -- so every displacement's group is checked before any
    displaced energy is computed."""
    mol = gg._build_mol(geom, basis)
    return mol.topgroup, mol.groupname


def _check_point_group(ref: tuple[str, str], got: tuple[str, str], what: str) -> None:
    """Refuse a displacement that changed the point group. The active space is
    selected by irrep and the group is auto-detected per geometry, so at a
    lowered symmetry the labels the selection is written in do not exist and
    the "same" active space is a different set of orbitals -- the energy is
    then not a value of the function being differentiated
    (docs/gradient-theory.md, §4.2, §4.3). Refused, not warned: the number it
    would produce looks like a gradient."""
    if got != ref:
        raise SystemExit(
            f"{what}: the detected point group changes from {ref[0]} (used: "
            f"{ref[1]}) at the reference geometry to {got[0]} (used: {got[1]}) "
            "under this displacement, so the irrep labels the active space is "
            "selected by do not survive it and the displaced energy is not a "
            "value of the same method. Refused rather than differenced "
            "(docs/gradient-theory.md, §4.3: numerical validation uses "
            "in-plane displacements of a planar Cs molecule only, and the "
            "out-of-plane components it would probe are zero by symmetry). "
            "Drop this component, or use --components auto."
        )


def selection_labels(pt) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The irrep labels of the core and of the active orbitals, sorted: the
    selection's signature, as far as irreps can see it. Built the way
    ``generate_golden._check_degeneracy`` builds its own labels, on the
    canonicalised orbitals."""
    from pyscf import symm

    mol = pt.mol
    orbsym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, pt.mo_coeff)
    nocc = pt.ncore + pt.ncas
    labels = [str(s) for s in orbsym]
    return tuple(sorted(labels[: pt.ncore])), tuple(sorted(labels[pt.ncore : nocc]))


def _check_selection(ref, got, what: str) -> None:
    """Refuse a displaced state whose core or active space is a different set
    of irreps from the reference's, at an unchanged point group: the second of
    the §4.2 hazards, a selection that reordered across a space boundary
    rather than a group that broke. A reorder *within* one irrep is invisible
    here and shows up instead as a Richardson residual that will not settle,
    which is why the residual is stored beside every component."""
    if got != ref:
        raise SystemExit(
            f"{what}: the displaced state's selection is core {got[0]}, active "
            f"{got[1]}, but the reference geometry's is core {ref[0]}, active "
            f"{ref[1]}. The active space moved across a space boundary under "
            "this displacement, so the two energies belong to different "
            "methods (docs/gradient-theory.md, §4.2). Refused rather than "
            "differenced."
        )


def _parse_components(spec: str, natm: int) -> list[tuple[int, int]] | None:
    """``"auto"`` -> ``None`` (every symmetry-preserving component, chosen by
    :func:`select_components`); ``"all"`` -> every component, each of which
    must then survive :func:`_check_point_group`; ``"0z,3x"`` -> just those."""
    spec = spec.strip()
    if spec == "auto":
        return None
    if spec == "all":
        return [(i, c) for i in range(natm) for c in range(3)]
    out = []
    for item in spec.split(","):
        m = re.fullmatch(r"\s*(\d+)\s*[:]?\s*([xyzXYZ])\s*", item)
        if m is None:
            raise SystemExit(
                f"--components: cannot read {item!r}; expected 'auto', 'all', "
                "or a list like '0z,3x' (atom index, then x/y/z)"
            )
        iatom, axis = int(m.group(1)), m.group(2).lower()
        if iatom >= natm:
            raise SystemExit(f"--components: atom index {iatom} but only {natm} atoms")
        out.append((iatom, AXES.index(axis)))
    return out


def select_components(
    geom, basis: str, ref_group: tuple[str, str], step_bohr: float,
    want: list[tuple[int, int]] | None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Partition the components into ``(probed, zero_by_symmetry)``, checking
    every displacement of every probed one first.

    With ``want`` given (``--components all`` or a named list), nothing is
    skipped: a component whose displacement changes the point group is
    **refused**. With ``want`` ``None`` (``--components auto``) a
    symmetry-breaking component is instead recorded as the exact zero it is at
    a planar geometry (docs/gradient-theory.md, §4.3) and only the rest are
    differentiated. Either way the check runs over every step of the stencil,
    before the first displaced SCF."""
    natm = len(geom)
    steps = stencil(step_bohr)
    probed, skipped = [], []
    for iatom, comp in (want if want is not None else
                        [(i, c) for i in range(natm) for c in range(3)]):
        label = f"atom {iatom} ({geom[iatom][0]}) {AXES[comp]}"
        groups = {
            s: point_group(displace(geom, iatom, comp, s), basis)
            for s in (*steps, *(-s for s in steps))
        }
        bad = {s: g for s, g in groups.items() if g != ref_group}
        if not bad:
            probed.append((iatom, comp))
        elif want is None:
            skipped.append((iatom, comp))
        else:
            s = next(iter(bad))
            _check_point_group(ref_group, bad[s], f"{label}, step {s:+g} bohr")
    return probed, skipped


def stencil(step_bohr: float) -> tuple[float, float, float]:
    """The three central-difference steps, coarse first: ``h``, ``h/2``,
    ``h/4``. Two Richardson extrapolations fit on them, so the second is an
    error bar on the first."""
    return step_bohr, step_bohr / 2, step_bohr / 4


def richardson(diffs: dict[float, float], steps) -> tuple[float, float]:
    """``(value, residual)`` from the three central differences: the finer
    Richardson extrapolant ``[4 g(h/2) - g(h)] / 3`` taken at ``(h/2, h/4)``,
    and how far it moved from the one taken at ``(h, h/2)``. The leading
    ``h^2`` error cancels in each, so what is left in the residual is the next
    order plus whatever the energies' own convergence floor contributes at
    this step: an error bar on the number stored, unlike the size of the
    Richardson correction itself, which only says how wrong the raw central
    difference was."""
    coarse = (4 * diffs[steps[1]] - diffs[steps[0]]) / 3
    fine = (4 * diffs[steps[2]] - diffs[steps[1]]) / 3
    return fine, abs(fine - coarse)


def energy(geom, case: dict, *, pc: bool = False) -> tuple[float, float, float | None, tuple]:
    """One displaced point: ``(E_CASCI, E_corr(SC), E_corr(PC) or None,
    selection signature)``, from ``generate_golden``'s own state builder and
    PySCF's own per-class energies -- the golden's answer, at a displaced
    geometry. With ``pc`` the same state is handed to block2 as well
    (:func:`pc_energy`), so both gradients come out of one sweep."""
    mc, pt = gg._build_state(
        geom,
        case["basis"],
        case["ncas"],
        case["nelecas"],
        cas_irreps=case["cas_irreps"],
        core_irreps=case["core_irreps"],
    )
    _dm1, _dm2, dms, eris = gg._pyscf_dms(pt)
    per_class = gg._pyscf_per_class(pt, dms, eris)
    e_corr = float(sum(per_class[c][1] for c in gg.CLASSES))
    e_pc = pc_energy(mc, pt) if pc else None
    return float(mc.e_tot), e_corr, e_pc, selection_labels(pt)


def numgrad(geom, case: dict, probed, step_bohr: float, ref_labels, *,
            pc: bool = False) -> dict:
    """Every probed component's gradient, by the six-point stencil. Returns
    the two ``(natm, 3)`` arrays the sidecar carries plus the CASCI-only
    gradient, which is not stored but is printed: it localises a later
    disagreement to the reference or to the perturbation. With ``pc``, the PC
    gradient and its own Richardson bar come out of the same displaced
    energies."""
    natm = len(geom)
    steps = stencil(step_bohr)
    grad = np.zeros((natm, 3))
    grad_casci = np.zeros((natm, 3))
    grad_pc = np.zeros((natm, 3))
    residual = np.zeros((natm, 3))
    residual_pc = np.zeros((natm, 3))
    probed_mask = np.zeros((natm, 3))
    for iatom, comp in probed:
        tot: dict[float, float] = {}
        cas: dict[float, float] = {}
        tot_pc: dict[float, float] = {}
        for h in steps:
            e = {}
            for sign in (+1, -1):
                moved = displace(geom, iatom, comp, sign * h)
                e_casci, e_corr, e_pc, labels = energy(moved, case, pc=pc)
                _check_selection(
                    ref_labels, labels,
                    f"atom {iatom} ({geom[iatom][0]}) {AXES[comp]}, step "
                    f"{sign * h:+g} bohr",
                )
                e[sign] = (e_casci, e_corr, e_pc)
            tot[h] = ((e[1][0] + e[1][1]) - (e[-1][0] + e[-1][1])) / (2 * h)
            cas[h] = (e[1][0] - e[-1][0]) / (2 * h)
            if pc:
                tot_pc[h] = ((e[1][0] + e[1][2]) - (e[-1][0] + e[-1][2])) / (2 * h)
        grad[iatom, comp], residual[iatom, comp] = richardson(tot, steps)
        grad_casci[iatom, comp], _ = richardson(cas, steps)
        if pc:
            grad_pc[iatom, comp], residual_pc[iatom, comp] = richardson(tot_pc, steps)
        probed_mask[iatom, comp] = 1.0
        print(
            f"  {iatom} {geom[iatom][0]:<2} {AXES[comp]}  "
            f"grad {grad[iatom, comp]:+.10f}  residual "
            f"{residual[iatom, comp]:.1e}  (CASCI only "
            f"{grad_casci[iatom, comp]:+.10f}; raw "
            + " ".join(f"{tot[h]:+.8f}" for h in steps)
            + ")"
            + (f"\n       PC   grad {grad_pc[iatom, comp]:+.10f}  residual "
               f"{residual_pc[iatom, comp]:.1e}" if pc else "")
        )
    out = dict(
        grad_num=grad, grad_num_residual=residual, grad_num_probed=probed_mask,
        grad_num_step=np.array([step_bohr]), grad_casci=grad_casci,
    )
    if pc:
        out["grad_num_pc"] = grad_pc
        out["grad_num_pc_residual"] = residual_pc
    return out


# --------------------------------------------------------------------------
# The CASSCF reference (--reference casscf)
# --------------------------------------------------------------------------


def geom_spec(geom) -> str:
    """A geometry as the ``--atom`` string PySCF parses, Angstrom, every
    coordinate written with ``repr`` so it round-trips exactly: what each
    displaced state's MO sidecar records as its ``atom_spec``, and what a later
    consumer of that sidecar (``generate_golden.py --atom SPEC --mo-from``)
    hands back to rebuild the identical molecule."""
    return "; ".join(f"{sym} {x!r} {y!r} {z!r}" for sym, (x, y, z) in geom)


def displace_vec(geom, dvec_bohr: np.ndarray):
    """``geom`` moved by a whole ``(natm, 3)`` displacement in bohr. For a
    single-component ``dvec`` it is :func:`displace` to the bit (every other
    coordinate gets ``+ 0.0``)."""
    moved = []
    for k, (sym, xyz) in enumerate(geom):
        moved.append((sym, tuple(float(xyz[c] + dvec_bohr[k, c] * BOHR) for c in range(3))))
    return moved


def carry_orbitals(mo: np.ndarray, s_ao: np.ndarray) -> np.ndarray:
    """Orbitals from one geometry carried to another: the same AO coefficients
    (PySCF keeps the input frame, so an AO index means the same function on
    the same atom at both), Löwdin-orthonormalised in the displaced overlap
    ``s_ao``. What the overlap signal compares a displaced state's active
    space against, and the warm start's orbitals."""
    w, v = np.linalg.eigh(mo.T @ s_ao @ mo)
    return mo @ (v * w**-0.5) @ v.T


def overlap_signals(mo_s, ci_s, s_ao, mo_0, ci_0, ncore: int, ncas: int,
                    nelecas) -> tuple[float, float]:
    """``(1 - sigma_min, 1 - |<ci_0|ci_s>|)``: how far a state's active space
    and CI vector are from a reference's, both carried into one basis. The
    active overlap is ``C_s^T S C_0`` over the active columns
    (:func:`carry_orbitals` for ``C_0``); the CI vector is rotated from the
    state's active orbitals to the reference's by that overlap's closest
    orthogonal matrix (``fci.addons.transform_ci``: new orbitals = old @ u),
    so a rotation *within* the active space -- the gauge -- costs nothing,
    while a different root or a swapped orbital does."""
    from pyscf.fci import addons

    c0 = carry_orbitals(mo_0, s_ao)
    act = slice(ncore, ncore + ncas)
    m = mo_s[:, act].T @ s_ao @ c0[:, act]
    uu, sv, vt = np.linalg.svd(m)
    ci_in_0 = addons.transform_ci(np.asarray(ci_s), nelecas, uu @ vt)
    ci_ovlp = abs(float(np.vdot(np.asarray(ci_0).ravel(), np.asarray(ci_in_0).ravel())))
    return float(1.0 - sv.min()), float(1.0 - ci_ovlp)


def continuity_tripped(sig: dict) -> list[str]:
    """Which of the guard's three signals a displaced state trips, by name:
    ``energy`` (:data:`CONTINUITY_E_TOL`), ``gradient`` (not converged, or the
    optimiser's own ``|g_orb|`` over ``CASSCF_CONV_TOL_GRAD``) and ``overlap``
    (:data:`CONTINUITY_ACT_TOL` or :data:`CONTINUITY_CI_TOL`). A NaN -- a
    signal that could not be evaluated -- trips it."""
    out = []
    if not sig["e_jump"] <= CONTINUITY_E_TOL:
        out.append("energy")
    if not (sig["converged"] and sig["g_opt"] <= gg.CASSCF_CONV_TOL_GRAD):
        out.append("gradient")
    if not (sig["act_defect"] <= CONTINUITY_ACT_TOL and sig["ci_defect"] <= CONTINUITY_CI_TOL):
        out.append("overlap")
    return out


def signal_text(sig: dict) -> str:
    """One state's three signals, each beside its threshold."""
    return (f"energy |E-E0-s<g,d>| {sig['e_jump']:.2e} (<= {CONTINUITY_E_TOL:g}); "
            f"gradient converged={sig['converged']} |g_orb| {sig['g_opt']:.2e} "
            f"(<= {gg.CASSCF_CONV_TOL_GRAD:g}); overlap 1-sv_act {sig['act_defect']:.2e} "
            f"(<= {CONTINUITY_ACT_TOL:g}), 1-|<ci0|ci>| {sig['ci_defect']:.2e} "
            f"(<= {CONTINUITY_CI_TOL:g})")


def _check_continuity(states: list[dict]) -> None:
    """Refuse the sweep if any displaced state landed on a different
    stationary point, or could not be shown not to: every refused state is
    named with the signals it tripped, after all of them have been evaluated
    (the whole picture is the diagnosis). Refused, not differenced -- the
    number a branch change would produce looks like a gradient."""
    bad = [st for st in states if st.get("tripped") or st.get("refused")]
    if not bad:
        return
    lines = []
    for st in bad:
        if st.get("refused"):
            lines.append(f"{st['label']}: refused while building it: {st['refused']}")
        else:
            lines.append(f"{st['label']}: tripped {', '.join(st['tripped'])} -- "
                         + signal_text(st["signals"]))
    raise SystemExit(
        f"the continuity guard refuses {len(bad)} displaced state(s) of "
        f"{len(states)}:\n  " + "\n  ".join(lines) + "\nA displaced CASSCF that "
        "is not provably on the reference geometry's branch is refused rather "
        "than differenced (docs/reference-data.md, \"The numerical gradient of "
        "a CASSCF reference\")."
    )


def casscf_point(task: dict) -> dict:
    """One displaced CASSCF state, end to end: the optimiser (cold, unless the
    task carries a warm start), the three continuity signals against the
    reference, then -- only if none trips -- the state's MO sidecar and the
    golden's own ``--mo-from`` CASCI + SC-NEVPT2 on it.

    A top-level function of one picklable dict so that ``--workers`` can farm it
    out; ``generate_golden``'s single-threading holds in every worker (the
    module is imported before numpy in each). Never raises a refusal: a
    ``SystemExit`` from a guard inside the CASCI is returned as ``refused`` so
    the parent can report every state before it refuses the sweep."""
    from pyscf import symm

    t0 = time.perf_counter()
    case, geom, label = task["case"], task["geom"], task["label"]
    spec = geom_spec(geom)
    ref = task.get("ref")
    start = None
    if task.get("warm"):
        # The warm start: the reference geometry's converged orbitals, carried
        # into this geometry's AO basis, each projected onto its own irrep and
        # re-orthonormalised (the irrep blocks are exactly orthogonal, so the
        # Löwdin step keeps them pure and in order), with its CI vector.
        mol = gg._build_mol(spec, case["basis"])
        s_ao = mol.intor_symmetric("int1e_ovlp")
        mo = symm.symmetrize_orb(mol, carry_orbitals(ref["mo"], s_ao), s=s_ao)
        start = (carry_orbitals(mo, s_ao), ref["ci_opt"])
    out = dict(label=label, atom=spec, warm=bool(task.get("warm")), s=task["s"],
               slope=task["slope"])
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            o = gg.casscf_optimise(
                spec, case["basis"], case["ncas"], case["nelecas"], label,
                cas_irreps=case["cas_irreps"], core_irreps=case["core_irreps"],
                start=start, refuse=False, verbose=False,
            )
            mc, fin = finish_casscf(o)
    except SystemExit as exc:  # impure orbitals: no label means anything
        out["refused"] = str(exc)
        out["wall"] = time.perf_counter() - t0
        return out
    s_ao = o["mol"].intor_symmetric("int1e_ovlp")
    ncore, ncas = mc.ncore, mc.ncas
    out.update(
        e_casscf=float(mc.e_tot), converged=o["converged"],
        first_converged=o["first_converged"], n_macro=o["n_macro"],
        polish_macro=o["polish_macro"], g_gen=o["g_norm"], g_opt=fin["g_norm"],
        finish_macro=fin["n_macro"], finish_converged=fin["converged"],
        e_casscf_gen=float(o["meta"]["e_casscf"]), mo_gen=np.asarray(o["mc"].mo_coeff),
        wall_opt=o["wall"] + fin["wall"],
        mo=np.asarray(mc.mo_coeff), ci_opt=np.asarray(mc.ci), s_ao=s_ao,
        act_irreps=o["names"][ncore : ncore + ncas], ncore=ncore, ncas=ncas,
        nelecas=tuple(mc.nelecas),
    )
    if task.get("keep_mc"):
        out["mc"] = mc
    if ref is None:
        act_defect = ci_defect = 0.0
        e_jump = 0.0
    else:
        act_defect, ci_defect = overlap_signals(
            out["mo"], out["ci_opt"], s_ao, ref["mo"], ref["ci_opt"], ncore, ncas,
            mc.nelecas)
        e_jump = abs(out["e_casscf"] - ref["e_casscf"] - task["s"] * task["slope"])
    out["signals"] = dict(e_jump=float(e_jump), converged=o["converged"],
                          g_opt=fin["g_norm"], act_defect=act_defect, ci_defect=ci_defect)
    out["tripped"] = continuity_tripped(out["signals"])
    if out["tripped"] or task.get("warm"):
        # A warm-started state is a probe of the cold one, never differenced,
        # so it needs no CASCI and writes no sidecar.
        out["wall"] = time.perf_counter() - t0
        return out
    path = Path(task["states_dir"]) / f"{label}.mogold"
    gg.write_mo_sidecar(path, **dict(o["meta"], e_casscf=float(mc.e_tot)),
                        arrays=dict(o["arrays"], mo_coeff=np.asarray(mc.mo_coeff)))
    out["mogold"] = path.name
    try:
        with contextlib.redirect_stdout(buf):
            cmc, pt = gg._build_state(
                spec, case["basis"], case["ncas"], case["nelecas"],
                cas_irreps=case["cas_irreps"], core_irreps=case["core_irreps"],
                mo_from=path,
            )
            _dm1, _dm2, dms, eris = gg._pyscf_dms(pt)
            per_class = gg._pyscf_per_class(pt, dms, eris)
    except SystemExit as exc:  # the --mo-from gate, _check_degeneracy, ...
        out["refused"] = str(exc)
        out["wall"] = time.perf_counter() - t0
        return out
    out.update(
        e_casci=float(cmc.e_tot),
        e_corr=float(sum(per_class[c][1] for c in gg.CLASSES)),
        labels=selection_labels(pt), gate=tuple(float(x) for x in cmc.mo_gate),
        ci_casci=np.asarray(cmc.ci),
    )
    out["wall"] = time.perf_counter() - t0
    return out


def congruent_spread(states: list[dict]) -> tuple[int, float, float]:
    """``(pairs, max spread of E_CASSCF, max spread of E_CASCI + E_corr)``
    over displaced states whose geometries are congruent -- the same sorted
    interatomic distances, to 1e-10 Å: a mirror image or a translation of one
    another, so in exact arithmetic the same energy. What they differ by is
    the sweep's own energy noise, measured directly rather than inferred from
    a Richardson residual (on LiF, ``+x``/``-x`` and Li ``+z``/F ``-z`` are
    such pairs; acrolein's in-plane stencil has none)."""
    groups: dict[tuple, list[dict]] = {}
    for st in states:
        xyz = np.array([c for _, c in st["task"]["geom"]])
        d = np.sort(np.linalg.norm(xyz[:, None] - xyz[None], axis=-1).ravel())
        groups.setdefault(tuple(np.round(d, 10)), []).append(st)
    pairs, e_scf, e_tot = 0, 0.0, 0.0
    for g in groups.values():
        if len(g) < 2:
            continue
        pairs += len(g) - 1
        e = [st["e_casscf"] for st in g]
        t = [st["e_casci"] + st["e_corr"] for st in g]
        e_scf = max(e_scf, max(e) - min(e))
        e_tot = max(e_tot, max(t) - min(t))
    return pairs, e_scf, e_tot


def finish_casscf(o: dict):
    """The numerical gradient's finish, in two stages from
    ``generate_golden.casscf_optimise``'s converged state: the generator's
    own ``mc1step`` polish with ``ah_lindep`` lowered to
    :data:`FINISH_AH_LINDEP` (so every state, converged-first-time or not,
    reaches the same ~4e-9), then ``mcscf.newton`` -- the second-order
    optimiser, orbitals and CI coupled -- with the FCI
    solver's ``lindep`` and ``conv_tol`` lowered (:data:`FINISH_FCI_LINDEP`,
    :data:`FINISH_FCI_CONV_TOL`) and the gradient threshold at
    :data:`FINISH_CONV_TOL_GRAD`. Returns ``(mc, info)`` with ``info`` the
    finish's own ``|g_orb|`` (its own CI vector), macro count, ``converged``
    and wall.

    Why it exists (measured: docs/reference-data.md, "Why the displaced
    states are finished"): ``mc1step``'s optimiser converges the orbitals
    against a CI vector whose residual floors near 1e-6 (the Davidson's
    ``lindep``), so "converged" means stationary for *that* CI vector. With
    an accurate one the same orbitals measure ``|g_orb|`` up to ~1e-6 -- the
    ``--mo-from`` gate's scatter -- and E_corr, first order in the orbitals,
    carries 1e-8-sized noise that a central difference divides by 2h. On
    acrolein that was a 6e-5 Richardson residual. Finished, the gate measures
    ~1e-9 and the same direction's residual is 1.1e-7. The finish runs after
    the generator's path, never instead of it, so the committed orbitals are
    still what that path reproduces bit for bit (``run_casscf`` checks it)."""
    from pyscf import mcscf

    t0 = time.perf_counter()
    # Stage 1: generate_golden's own polish settings, with ah_lindep under
    # |g_orb|^2. mc1step's augmented Hessian starts from (1, g), whose overlap
    # has smallest eigenvalue |g|^2; below ah_lindep (PySCF's 1e-14, i.e.
    # |g| < 1e-7) its step is exactly zero. So a first run that converged just
    # under conv_tol_grad -- acrolein's 0x+2 at 7.5e-8, LiF's Li z+h at 9.4e-8
    # -- is frozen there, while its stalled-and-polished neighbours sit at
    # ~4e-9; newton from those two different starts lands 1.1e-9 Eh apart in
    # E_corr (1.2e-6 Eh/bohr in acrolein's 0x). This stage brings every state
    # to the same ~4e-9 before stage 2.
    mcp = gg._casscf(o["mf"], o["mc"].ncas, o["mc"].nelecas, None)
    mcp.conv_tol = gg.CASSCF_CONV_TOL
    mcp.conv_tol_grad = gg.CASSCF_CONV_TOL_GRAD
    mcp.max_cycle_macro = gg.CASSCF_MAX_CYCLE_MACRO
    mcp.ah_conv_tol = gg.CASSCF_POLISH_AH_CONV_TOL
    mcp.ah_lindep = FINISH_AH_LINDEP
    mcp.kernel(o["mc"].mo_coeff, ci0=o["mc"].ci)
    # Stage 2: newton, orbitals and CI coupled, with an accurate FCI solver.
    mcn = mcscf.newton(gg._casscf(o["mf"], o["mc"].ncas, o["mc"].nelecas, None))
    mcn.conv_tol = FINISH_CONV_TOL
    mcn.conv_tol_grad = FINISH_CONV_TOL_GRAD
    mcn.max_cycle_macro = FINISH_MAX_CYCLE_MACRO
    mcn.ah_conv_tol = gg.CASSCF_POLISH_AH_CONV_TOL
    mcn.ah_lindep = FINISH_AH_LINDEP
    mcn.fcisolver.lindep = FINISH_FCI_LINDEP
    mcn.fcisolver.conv_tol = FINISH_FCI_CONV_TOL
    macro: list = []
    mcn.callback = lambda envs: macro.append(envs.get("imacro"))
    mcn.kernel(mcp.mo_coeff, ci0=mcp.ci)
    gg._check_mo_symmetry(o["mol"], mcn.mo_coeff)
    eris = mcn.ao2mo(mcn.mo_coeff)
    casdm1, casdm2 = mcn.fcisolver.make_rdm12(mcn.ci, mcn.ncas, mcn.nelecas)
    g = float(np.linalg.norm(mcn.get_grad(mcn.mo_coeff, (casdm1, casdm2), eris)))
    return mcn, dict(g_norm=g, n_macro=(max(macro) + 1) if macro else 0,
                     converged=bool(mcn.converged), wall=time.perf_counter() - t0)


def casscf_gradient(mc) -> np.ndarray:
    """``pyscf.grad.casscf`` at an optimised state, ``(natm, 3)`` Eh/bohr: the
    CASSCF energy's own analytic gradient, which the energy signal's
    prediction and the CASSCF-only free check use."""
    from pyscf.grad import casscf as casscf_grad

    return np.asarray(casscf_grad.Gradients(mc).kernel(mo_coeff=mc.mo_coeff, ci=mc.ci))


def run_points(tasks: list[dict], workers: int) -> list[dict]:
    """:func:`casscf_point` over every task, ``workers`` PROCESSES at a time
    (never threads: each worker is one single-threaded process, so
    ``generate_golden``'s forced single-threading -- what makes a state
    bit-reproducible -- holds in every one). The ``spawn`` context starts each
    worker from a fresh interpreter that re-imports ``generate_golden``
    before numpy, rather than forking a parent whose thread pools already
    exist. Every task is an independent cold start, and the results are
    collected by the task's label, never by completion order, so the output
    is bit-identical for any ``--workers`` (measured: docs/reference-data.md,
    "The numerical gradient of a CASSCF reference")."""
    import multiprocessing

    if workers <= 1 or len(tasks) <= 1:
        return [casscf_point(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {t["label"]: pool.submit(casscf_point, t) for t in tasks}
        assert len(futures) == len(tasks), "two tasks share a label"
        return [futures[t["label"]].result() for t in tasks]


def directions(probed, natm: int, n: int, seed: int) -> np.ndarray:
    """``n`` seeded random unit directions in the probed components' span,
    ``(n, natm, 3)``, bohr: every symmetry-zero component exactly zero, so a
    displacement along one keeps the point group whenever its components do."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, natm, 3))
    for k in range(n):
        v = rng.standard_normal(len(probed))
        v /= np.linalg.norm(v)
        for (iatom, comp), x in zip(probed, v, strict=True):
            out[k, iatom, comp] = x
    return out


def load_analytic(path: Path, case: str, *, pc: bool = False) -> dict:
    """The analytic gradient the directional screen holds the numerical
    derivative to: an ``nevpt2_grad.py --reference casscf --out PATH`` file,
    checked to be the SC (not ``--casci-only``, not ``--pc``) gradient of a
    ``casscf`` reference of this case -- or, with ``pc`` (``--pc-states``),
    the ``--pc`` one."""
    scheme = "PC" if pc else "SC"
    with np.load(path) as z:
        got = {k: (str(z[k]) if k in z else None)
               for k in ("case", "reference", "casci_only", "pc")}
        if got["case"] != case or got["reference"] != "casscf" or \
                got["casci_only"] != "False" or got["pc"] != str(pc):
            raise SystemExit(
                f"{path} is not the {scheme} gradient of --case {case} --reference casscf "
                f"(it says case={got['case']!r}, reference={got['reference']!r}, "
                f"casci_only={got['casci_only']}, pc={got['pc']}): the screen would "
                "compare two different functions"
            )
        return dict(grad=np.asarray(z["grad"]), mo_from=str(z["mo_from"])
                    if "mo_from" in z else "")


def _stencil_tasks(key, dvec_unit, steps, slope, geom, case, ref, states_dir):
    """The six tasks of one coordinate (a component or a direction): signs
    ``+/-`` at each of the three steps, labelled ``<key><sign><rung>``."""
    tasks = []
    for k, h in enumerate(steps):
        for sign in (+1, -1):
            s = sign * h
            tasks.append(dict(
                case=case, geom=displace_vec(geom, s * dvec_unit),
                label=f"{key}{'+' if sign > 0 else '-'}{k}", s=s, slope=slope,
                ref=ref, states_dir=str(states_dir), key=key, rung=k, sign=sign,
                unit=dvec_unit,
            ))
    return tasks


def _difference(states: list[dict], steps, field) -> tuple[float, float]:
    """Richardson over one coordinate's six states, of ``field(state)``."""
    by = {(st["rung"], st["sign"]): field(st) for st in states}
    diffs = {h: (by[(k, 1)] - by[(k, -1)]) / (2 * h) for k, h in enumerate(steps)}
    return richardson(diffs, steps)


#: A ``--pc-states`` key: a component (``<atom><axis>``) or a screen
#: direction (``d<n>``), as :func:`_stencil_tasks` labels them.
_COMPONENT_KEY = re.compile(r"^(\d+)([xyz])$")
_DIRECTION_KEY = re.compile(r"^d(\d+)$")


def difference_pc_states(path: Path, case: str) -> dict:
    """``--pc-states``: the PC numerical gradient of a CASSCF reference, from
    the GPU's PC energies on a ``--reference casscf`` sweep's kept states
    (``pc_on_states.py``'s keyed ``.npz``, #138), differenced with **this
    sweep's own stencil** (:func:`stencil`, :func:`_difference`) -- #126.

    Nothing is optimised and nothing runs: the states, their ``E_CASCI`` and
    their keys are the sweep's, so the PC gradient is taken on exactly the
    states the SC one was. Every state a coordinate needs must have been
    built, passed the per-state PC conditioning guard and carry a finite
    ``e_pc``, or the run is **refused**, naming the states: a refused state
    is a hole in the stencil, never a number to difference around.

    Beside ``E_CASCI + E_PC`` it differences ``E_CASCI + E_corr(SC)`` from the
    same file -- PySCF's on the transient goldens, and the device's -- so the
    SC gradient of the sweep that made the states is reproduced from this
    file alone, the wiring check of the whole hand-off."""
    with np.load(path) as z:
        f = {k: z[k] for k in z.files}
    if str(f["case"]) != case:
        raise SystemExit(f"{path} holds the PC energies of case {str(f['case'])!r}, "
                         f"not {case!r}")
    step = float(f["step"])
    steps = stencil(step)
    labels = [str(x) for x in f["labels"]]
    usable = f["built"] & f["guard_ok"] & np.isfinite(f["e_pc"])
    bad = [f"{lab} ({f['guard_why'][i] or f['build_why'][i] or 'no e_pc'})"
           for i, lab in enumerate(labels) if not usable[i]]
    if bad:
        raise SystemExit(
            f"{path}: {len(bad)} state(s) have no usable PC energy -- {'; '.join(bad)}. "
            "A refused state is a hole in the stencil: the coordinate it belongs to is "
            "not differenced around it")
    natm = f["displacement_bohr"].shape[1]
    by_key: dict[str, list[dict]] = {}
    for i, key in enumerate(str(k) for k in f["key"]):
        if key == "ref":
            continue
        by_key.setdefault(key, []).append(dict(
            rung=int(f["rung"][i]), sign=int(f["sign"][i]), s=float(f["s"][i]),
            unit=f["displacement_bohr"][i] / float(f["s"][i]),
            pc=float(f["e_casci"][i] + f["e_pc"][i]),
            sc=float(f["e_casci"][i] + f["e_corr_sc"][i]),
            sc_gpu=float(f["e_casci"][i] + f["e_sc_gpu"][i])))
    want = {(k, sign) for k in range(len(steps)) for sign in (1, -1)}
    for key, sts in by_key.items():
        got = [(st["rung"], st["sign"]) for st in sts]
        if sorted(got) != sorted(want):
            raise SystemExit(f"{path}: coordinate {key} has states {sorted(got)}, not the "
                             f"six of the h, h/2, h/4 stencil")
    rec = dict(case=case, step=step, natm=natm, pc_states=str(path),
               ref_pc_offset=float(f["ref_pc_offset"]),
               guard_min_gap=float(f["guard_min_gap"].min()), n_states=len(labels))
    grad = {n: np.zeros((natm, 3)) for n in ("pc", "sc", "sc_gpu")}
    res = {n: np.zeros((natm, 3)) for n in ("pc", "sc", "sc_gpu")}
    probed = np.zeros((natm, 3))
    dirs: dict[int, tuple] = {}
    for key, sts in by_key.items():
        m = _COMPONENT_KEY.match(key)
        if m:
            iatom, comp = int(m.group(1)), AXES.index(m.group(2))
            for n in grad:
                grad[n][iatom, comp], res[n][iatom, comp] = _difference(
                    sts, steps, lambda st, n=n: st[n])
            probed[iatom, comp] = 1.0
            continue
        m = _DIRECTION_KEY.match(key)
        if not m:
            raise SystemExit(f"{path}: key {key!r} is neither a component nor a direction")
        unit = next(st["unit"] for st in sts if st["rung"] == 0 and st["sign"] == 1)
        dirs[int(m.group(1))] = (unit, *(_difference(sts, steps, lambda st, n=n: st[n])
                                         for n in ("pc", "sc")))
    rec.update(grad_num_pc=grad["pc"], grad_num_pc_residual=res["pc"],
               grad_num=grad["sc"], grad_num_residual=res["sc"],
               grad_num_sc_gpu=grad["sc_gpu"], grad_num_probed=probed,
               grad_num_step=np.array([step]))
    if dirs:
        order = sorted(dirs)
        rec.update(dir_vectors=np.array([dirs[n][0] for n in order]),
                   dir_num_pc=np.array([dirs[n][1][0] for n in order]),
                   dir_residual_pc=np.array([dirs[n][1][1] for n in order]),
                   dir_num=np.array([dirs[n][2][0] for n in order]),
                   dir_residual=np.array([dirs[n][2][1] for n in order]))
    return rec


def report_pc_states(rec: dict, analytic: dict | None) -> bool:
    """``--pc-states``' report: the PC numerical gradient beside the SC one
    from the same file, and -- with ``analytic``, an ``nevpt2_grad.py
    --reference casscf --pc --out`` file -- each component and each screen
    direction against it. The directions are gated here at
    ``nevpt2_grad.GRAD_NUM_TOL``, as the SC screen is
    (:func:`report_casscf`); the componentwise gate is ``nevpt2_grad.py
    --grad-num``'s. Returns whether the directions passed (True with none)."""
    import nevpt2_grad  # lazily: it imports this module

    tol = nevpt2_grad.GRAD_NUM_TOL
    probed = rec["grad_num_probed"] > 0
    print(f"== {rec['case']}  --reference casscf --pc-states {rec['pc_states']}")
    print(f"{rec['n_states']} states, every one built, guarded (smallest gap "
          f"{rec['guard_min_gap']:.2g}) and with a PC energy; reference E_PC - committed "
          f"e_pc_total {rec['ref_pc_offset']:+.2e}; step {rec['step']:g} bohr, stencil "
          + "/".join(f"{h:g}" for h in stencil(rec["step"])))
    g, r = rec["grad_num_pc"], rec["grad_num_pc_residual"]
    a = analytic["grad"] if analytic else None
    print(f"{'comp':>5} {'PC numerical':>16} {'residual':>9} {'SC numerical':>16}"
          + (f" {'PC analytic':>16} {'|delta|':>9}" if a is not None else ""))
    for i in range(rec["natm"]):
        for c in range(3):
            tail = (f" {a[i, c]:+16.10f} "
                    + (f"{abs(a[i, c] - g[i, c]):9.2e}" if probed[i, c] else f"{'0 (sym)':>9}")
                    if a is not None else "")
            if probed[i, c]:
                print(f"{i:>4}{AXES[c]} {g[i, c]:+16.10f} {r[i, c]:9.1e} "
                      f"{rec['grad_num'][i, c]:+16.10f}{tail}")
            else:
                print(f"{i:>4}{AXES[c]} {'0 (sym)':>16} {'.':>9} {'0 (sym)':>16}{tail}")
    print(f"max PC residual {r[probed].max():.1e}; |SC PySCF - SC GPU| over the probed "
          f"components {np.abs(rec['grad_num'] - rec['grad_num_sc_gpu'])[probed].max():.1e}")
    full = probed.all(axis=0)
    if full.any():
        print("sum over atoms (0 by translational invariance), PC: "
              + ", ".join(f"d/d{AXES[c]} {g[:, c].sum():+.2e}" for c in range(3) if full[c]))
    if a is not None:
        d = np.abs(a - g)[probed]
        print(f"against --analytic: max |g - grad_num_pc| {d.max():.2e}, RMS "
              f"{np.sqrt(np.mean(d ** 2)):.2e} over {int(probed.sum())} components; "
              f"max |g| on the {int((~probed).sum())} zero-by-symmetry ones "
              f"{np.abs(a[~probed]).max() if (~probed).any() else 0.0:.2e} (the gate, "
              f"{tol:g}, is nevpt2_grad.py --grad-num's)")
    ok = True
    if "dir_num_pc" in rec:
        print(f"{'dir':>4} {'PC numerical':>16} {'residual':>9}"
              + (f" {'<g, d>':>16} {'|delta|':>9}" if a is not None else ""))
        for n, (num, res) in enumerate(zip(rec["dir_num_pc"], rec["dir_residual_pc"],
                                           strict=True)):
            if a is None:
                print(f"  d{n} {num:+16.10f} {res:9.1e}")
                continue
            an = float(np.sum(a * rec["dir_vectors"][n]))
            ok = ok and abs(num - an) <= tol
            print(f"  d{n} {num:+16.10f} {res:9.1e} {an:+16.10f} {abs(num - an):9.1e}")
        if a is not None:
            print(f"{'PASS' if ok else 'FAIL'}: the PC directional screen, "
                  f"{len(rec['dir_num_pc'])} directions, allowed |delta| {tol:g}")
    return ok


def warm_probe(cold: list[dict], n: int) -> list[dict]:
    """Cold start is the oracle's definition; warm start -- from the
    reference geometry's converged orbitals and CI vector -- is a free probe
    for a second solution. Re-optimise the first ``n`` coarse-step geometries
    warm and compare each with its cold state at the same geometry: the
    energy against ``generate_golden.CASSCF_E_TOL`` (the fingerprint in
    energy form), the active space and CI vector against the continuity
    guard's overlap thresholds. A mismatch means the geometry has two
    solutions within reach, and it is refused like a tripped signal."""
    picks = [st for st in cold if st["rung"] == 0][:n]
    tasks = [dict(st["task"], warm=True, label=st["label"] + "w") for st in picks]
    # Sequential by design: two or three states, and a probe of the
    # cold ones rather than part of the sweep.
    warm = [casscf_point(t) for t in tasks]
    out = []
    for c, w in zip(picks, warm, strict=True):
        rec = dict(label=c["label"], refused=w.get("refused"))
        if not w.get("refused"):
            act, ci = overlap_signals(w["mo"], w["ci_opt"], c["s_ao"], c["mo"],
                                      c["ci_opt"], c["ncore"], c["ncas"], c["nelecas"])
            rec.update(d_e=w["e_casscf"] - c["e_casscf"], act_defect=act, ci_defect=ci,
                       g_cold=c["g_opt"], g_warm=w["g_opt"], macro_cold=c["n_macro"],
                       macro_warm=w["n_macro"], polish_warm=w["polish_macro"],
                       wall_cold=c["wall_opt"], wall_warm=w["wall_opt"])
            rec["same"] = (abs(rec["d_e"]) <= gg.CASSCF_E_TOL and act <= CONTINUITY_ACT_TOL
                           and ci <= CONTINUITY_CI_TOL)
        else:
            rec["same"] = False
        out.append(rec)
    return out


def run_casscf(args, case: dict, geom, rec: dict) -> dict:
    """``--reference casscf``: the reference state re-optimised and checked
    against the committed golden, then every displaced state re-optimised
    cold, guarded, built and differenced -- componentwise, or with
    ``--directional N`` the screen along ``N`` seeded directions."""
    t0 = time.perf_counter()
    states_dir = args.states_dir
    if states_dir is None:
        states_dir = Path(tempfile.mkdtemp(prefix=f"numgrad-{args.case}-casscf-"))
    states_dir.mkdir(parents=True, exist_ok=True)
    rec["states_dir"] = str(states_dir)

    # The reference geometry, through the very path every displaced state
    # takes. Two checks tie it to the committed files, neither loosened:
    # generate_golden's own optimiser (before the finish) must reproduce the
    # committed .mogold's orbitals BIT FOR BIT, and those orbitals must
    # reproduce the golden's E_CASCI and E_corr within REF_STATE_TOL (both
    # measured at exactly 0). The finished reference the differences are taken
    # around is then reported against them: same stationary point (E_CASSCF
    # within generate_golden.CASSCF_E_TOL), with the E_corr the finish moved.
    ref_task = dict(case=case, geom=geom, label="ref", s=0.0, slope=0.0, ref=None,
                    states_dir=str(states_dir), key="ref", rung=0, sign=0,
                    unit=np.zeros((len(geom), 3)), keep_mc=True)
    r0 = casscf_point(ref_task)
    if r0.get("refused") or r0["tripped"]:
        raise SystemExit(f"the reference geometry's own CASSCF is refused: "
                         f"{r0.get('refused') or signal_text(r0['signals'])}")
    gold = gf.read_golden_header(args.golden)
    mo_side = gg.read_mo_sidecar(args.mo_from)
    rec["mo_bitwise"] = bool(np.array_equal(r0["mo_gen"], mo_side["mo_coeff"]))
    rec["d_e_casscf_gen"] = r0["e_casscf_gen"] - mo_side["e_casscf"]
    with contextlib.redirect_stdout(io.StringIO()):
        cmc, cpt = gg._build_state(
            geom_spec(geom), case["basis"], case["ncas"], case["nelecas"],
            cas_irreps=case["cas_irreps"], core_irreps=case["core_irreps"],
            mo_from=args.mo_from)
        _d1, _d2, dms, eris = gg._pyscf_dms(cpt)
        per_class = gg._pyscf_per_class(cpt, dms, eris)
    e_corr_c = float(sum(per_class[c][1] for c in gg.CLASSES))
    rec["d_e_casci"] = float(cmc.e_tot) - gold["e_casci"]
    rec["d_e_corr"] = e_corr_c - gold["e_nevpt2_total"]
    if not rec["mo_bitwise"] or \
            max(abs(rec["d_e_casci"]), abs(rec["d_e_corr"])) > REF_STATE_TOL:
        raise SystemExit(
            f"the reference does not tie to {args.golden} / {args.mo_from}: "
            f"re-optimised orbitals bit-identical to the committed ones: "
            f"{rec['mo_bitwise']} (E_CASSCF {rec['d_e_casscf_gen']:+.3e}); the "
            f"committed orbitals' E_CASCI {rec['d_e_casci']:+.3e}, E_corr "
            f"{rec['d_e_corr']:+.3e} from the golden (tolerance {REF_STATE_TOL:g}). "
            "The optimiser path here is not the one those files were written by, so "
            "its displaced states would be of a different function."
        )
    rec.update(e_casci=r0["e_casci"], e_corr=r0["e_corr"], e_pc=None,
               labels=r0["labels"], e_casscf=r0["e_casscf"], ref_state=r0)
    rec["finish_shift"] = dict(e_casscf=r0["e_casscf"] - mo_side["e_casscf"],
                               e_casci=r0["e_casci"] - gold["e_casci"],
                               e_corr=r0["e_corr"] - gold["e_nevpt2_total"])
    if abs(rec["finish_shift"]["e_casscf"]) > gg.CASSCF_E_TOL:
        raise SystemExit(
            f"the finish moved the reference's E_CASSCF by "
            f"{rec['finish_shift']['e_casscf']:+.3e} (> {gg.CASSCF_E_TOL:g}): it found "
            "a different stationary point from the committed one, not the same one "
            "converged further."
        )
    # The energy signal's first-order prediction: PySCF's own analytic CASSCF
    # gradient at the reference (the CASSCF energy is variational, so this is
    # exact to first order whatever the NEVPT2 machinery does).
    mc0 = r0.pop("mc")
    with contextlib.redirect_stdout(io.StringIO()):
        g_casscf = casscf_gradient(mc0)
    rec["g_casscf"] = g_casscf
    ref = dict(mo=r0["mo"], ci_opt=r0["ci_opt"], e_casscf=r0["e_casscf"])

    want = _parse_components(args.components, len(geom))
    probed, skipped = select_components(geom, case["basis"], rec["group"], args.step, want)
    rec["probed"], rec["zero_by_symmetry"] = probed, skipped
    steps = stencil(args.step)
    natm = len(geom)
    coords = []  # (key, unit displacement, slope)
    if args.directional:
        dirs = directions(probed, natm, args.directional, SCREEN_SEED)
        for n, d in enumerate(dirs):
            for s in (*steps, *(-s for s in steps)):
                _check_point_group(rec["group"], point_group(displace_vec(geom, s * d),
                                                             case["basis"]),
                                   f"direction d{n}, step {s:+g} bohr")
            coords.append((f"d{n}", d, float(np.sum(g_casscf * d))))
        rec["dir_vectors"] = dirs
    else:
        for iatom, comp in probed:
            d = np.zeros((natm, 3))
            d[iatom, comp] = 1.0
            coords.append((f"{iatom}{AXES[comp]}", d, float(g_casscf[iatom, comp])))
    tasks = []
    for key, d, slope in coords:
        tasks += _stencil_tasks(key, d, steps, slope, geom, case, ref, states_dir)
    what = (f"{args.directional} directions (the screen)" if args.directional else
            f"{len(probed)} components, {len(skipped)} zero by symmetry")
    print(f"[{args.case}] --reference casscf: {what}, {len(tasks)} displaced "
          f"CASSCF+CASCI+NEVPT2 states, --workers {args.workers}, states in {states_dir}",
          flush=True)
    t_sweep = time.perf_counter()
    states = run_points(tasks, args.workers)
    rec["wall_sweep"] = time.perf_counter() - t_sweep
    for t, st in zip(tasks, states, strict=True):
        st.update(task=t, key=t["key"], rung=t["rung"], sign=t["sign"])
    _write_manifest(states_dir, args, rec, r0, states)
    for st in states:
        if st.get("refused"):
            print(f"  {st['label']:<8} REFUSED: {st['refused']}")
            continue
        sig = st["signals"]
        print(f"  {st['label']:<8} E_CASSCF {st['e_casscf']:+.12f}  macro {st['n_macro']:3d}"
              f"+{st['polish_macro']}+{st['finish_macro']}n  |g_orb| {st['g_gen']:.1e}"
              f"->{sig['g_opt']:.1e}  e_jump "
              f"{sig['e_jump']:.1e}  1-sv {sig['act_defect']:.1e}  1-ci {sig['ci_defect']:.1e}"
              + (f"  gate {st['gate'][0]:.1e}" if "gate" in st else "")
              + (f"  TRIPPED {','.join(st['tripped'])}" if st["tripped"] else "")
              + f"  ({st['wall']:.1f}s)")
    rec["signals_max"] = {
        k: max(st["signals"][k] for st in states if "signals" in st)
        for k in ("e_jump", "g_opt", "act_defect", "ci_defect")
    }
    rec["gate_max"] = max((st["gate"][0] for st in states if "gate" in st), default=float("nan"))
    # Every displaced state's numbers, in task order, for the --npz: the
    # --mo-from gate's (recorded, not refusing) beside the optimiser's own.
    rec["state_labels"] = np.array([st["label"] for st in states])
    rec["state_gate"] = np.array([st["gate"][0] if "gate" in st else np.nan for st in states])
    rec["state_g_opt"] = np.array([st.get("g_opt", np.nan) for st in states])
    rec["state_e_jump"] = np.array([st["signals"]["e_jump"] if "signals" in st else np.nan
                                    for st in states])
    rec["state_act_defect"] = np.array([st["signals"]["act_defect"] if "signals" in st
                                        else np.nan for st in states])
    rec["state_ci_defect"] = np.array([st["signals"]["ci_defect"] if "signals" in st
                                       else np.nan for st in states])
    rec["first_converged"] = sorted(st["label"] for st in states if st.get("first_converged"))
    rec["congruent"] = congruent_spread([s for s in states if "e_corr" in s])
    _check_continuity(states)
    for st in states:
        _check_selection(r0["labels"], st["labels"], f"state {st['label']}")

    # The cold-vs-warm probe, on the first coarse-step geometries.
    if args.warm_probe:
        rec["warm"] = warm_probe(states, args.warm_probe)
        for w in rec["warm"]:
            if w.get("refused"):
                print(f"  warm {w['label']}: REFUSED while building it: {w['refused']}")
                continue
            print(f"  warm {w['label']}: dE_CASSCF {w['d_e']:+.1e}  1-sv {w['act_defect']:.1e}"
                  f"  1-ci {w['ci_defect']:.1e}  |g| cold {w['g_cold']:.1e} warm "
                  f"{w['g_warm']:.1e}  macro cold {w['macro_cold']} warm "
                  f"{w['macro_warm']}+{w['polish_warm']}  wall cold {w['wall_cold']:.1f}s "
                  f"warm {w['wall_warm']:.1f}s  -> "
                  + ("same stationary point" if w["same"] else "DIFFERENT"))
        bad = [w["label"] for w in rec["warm"] if not w["same"]]
        if bad:
            raise SystemExit(
                f"cold and warm start land on different stationary points at {bad}: "
                "those geometries are ambiguous, and the continuity guard refuses "
                "them rather than letting the sweep order pick the solution."
            )

    by_key: dict[str, list[dict]] = {}
    for st in states:
        by_key.setdefault(st["key"], []).append(st)
    tot = lambda st: st["e_casci"] + st["e_corr"]  # noqa: E731
    cas = lambda st: st["e_casci"]  # noqa: E731
    if args.directional:
        num = np.array([_difference(by_key[f"d{n}"], steps, tot) for n in range(len(coords))])
        num_cas = np.array([_difference(by_key[f"d{n}"], steps, cas)[0]
                            for n in range(len(coords))])
        rec.update(dir_num=num[:, 0], dir_residual=num[:, 1], dir_casscf_num=num_cas,
                   dir_casscf_pyscf=np.array([c[2] for c in coords]))
    else:
        grad = np.zeros((natm, 3))
        residual = np.zeros((natm, 3))
        grad_cas = np.zeros((natm, 3))
        mask = np.zeros((natm, 3))
        for iatom, comp in probed:
            key = f"{iatom}{AXES[comp]}"
            grad[iatom, comp], residual[iatom, comp] = _difference(by_key[key], steps, tot)
            grad_cas[iatom, comp], _ = _difference(by_key[key], steps, cas)
            mask[iatom, comp] = 1.0
        rec.update(grad_num=grad, grad_num_residual=residual, grad_num_probed=mask,
                   grad_num_step=np.array([args.step]), grad_casci=grad_cas)
    rec["wall"] = time.perf_counter() - t0
    return rec


def _write_manifest(states_dir: Path, args, rec: dict, r0: dict, states: list[dict]) -> None:
    """``states.json`` beside the displaced states' MO sidecars: what each one
    is (label, displacement, ``--atom`` spec), what it measured, and where its
    ``.mogold`` is; ``golden_format.read_states_manifest`` reads it back.
    This is how a later consumer -- ``pc_on_states.py``'s per-state
    ``--no-pc`` goldens for the GPU, #126's block2 spot-checks -- reuses
    **this** sweep's states instead of re-optimising them: each sidecar is a
    ``--mo-from`` input, rebuilt with ``generate_golden.py --atom SPEC`` and
    this case's flags (docs/reference-data.md, "The numerical gradient of a
    CASSCF reference").

    One directory holds one sweep's states for one case: the screen's
    (``d<n>...``) and the componentwise gate's (``<atom><axis>...``) labels do
    not collide, so a second run into the same directory MERGES into the
    manifest by label rather than replacing it, and refuses a directory
    written for a different case or step."""
    keep = ("label", "atom", "s", "e_casscf", "e_casci", "e_corr", "converged",
            "first_converged", "n_macro", "polish_macro", "g_opt", "gate", "mogold",
            "tripped", "refused", "signals", "wall", "key", "rung", "sign")
    entries = []
    for st in [dict(r0, key="ref", rung=0, sign=0), *states]:
        e = {k: st[k] for k in keep if k in st}
        unit = st["task"]["unit"] if "task" in st else np.zeros((rec["natm"], 3))
        e["displacement_bohr"] = (st["s"] * unit).tolist()
        entries.append(e)
    path = states_dir / "states.json"
    head = dict(
        case=args.case, reference="casscf", basis=rec["basis"], ncas=rec["ncas"],
        nelecas=rec["nelecas"], cas_irreps=rec["cas_irreps"], step=args.step,
        finish=dict(method="mcscf.newton", conv_tol=FINISH_CONV_TOL,
                    conv_tol_grad=FINISH_CONV_TOL_GRAD,
                    max_cycle_macro=FINISH_MAX_CYCLE_MACRO, ah_lindep=FINISH_AH_LINDEP,
                    fci_lindep=FINISH_FCI_LINDEP, fci_conv_tol=FINISH_FCI_CONV_TOL),
    )
    merged: dict[str, dict] = {}
    if path.exists():
        old = json.loads(path.read_text())
        clash = [k for k in ("case", "step", "basis") if old.get(k) != head[k]]
        if clash:
            raise SystemExit(f"{path} holds another sweep's states ({', '.join(clash)} "
                             "differ): pass a fresh --states-dir")
        merged = {e["label"]: e for e in old["states"]}
    merged.update({e["label"]: e for e in entries})
    head["states"] = list(merged.values())
    path.write_text(json.dumps(head, indent=1, default=str))


def run(args) -> dict:
    """Check the reference state against its golden, pick the components,
    difference them, and return the record :func:`report` prints."""
    case = dict(CASES[args.case]) if args.case in CASES else {}
    for key in ("geometry", "basis", "ncas", "nelecas", "cas_irreps", "core_irreps"):
        if getattr(args, key) is not None:
            case[key] = getattr(args, key)
    missing = [k for k in ("geometry", "basis", "ncas", "nelecas") if case.get(k) is None]
    if missing:
        raise SystemExit(
            f"--case {args.case} is not one of {sorted(CASES)}, so "
            f"{', '.join('--' + m.replace('_', '-') for m in missing)} must be given"
        )
    case.setdefault("cas_irreps", None)
    case.setdefault("core_irreps", None)

    geom_path = Path(case["geometry"])
    if not geom_path.is_absolute():
        geom_path = REPO / geom_path
    geom = read_xyz(geom_path)
    rec = dict(case=args.case, geometry=str(geom_path), basis=case["basis"],
               ncas=case["ncas"], nelecas=case["nelecas"],
               cas_irreps=case["cas_irreps"], step=args.step, natm=len(geom))

    # The golden's header first: reading it costs nothing, and its reference
    # kind decides whether differencing this state means anything at all, so
    # the refusal lands before the reference energy is computed.
    gold = gf.read_golden_header(args.golden)
    rec["reference_kind"] = gold["reference_kind"]
    _check_reference_kind(gold["reference_kind"], args.golden, args.reference)

    if args.reference == "casscf":
        rec["group"] = point_group(geom, case["basis"])
        rec["pc"] = False
        return run_casscf(args, case, geom, rec)

    t0 = time.perf_counter()
    ref_group = point_group(geom, case["basis"])
    e_casci, e_corr, e_pc, ref_labels = energy(geom, case, pc=args.pc)
    rec.update(group=ref_group, e_casci=e_casci, e_corr=e_corr, e_pc=e_pc,
               labels=ref_labels, pc=args.pc)

    rec["d_e_casci"] = e_casci - gold["e_casci"]
    rec["d_e_corr"] = e_corr - gold["e_nevpt2_total"]
    if args.pc:
        # The golden's own PC answer, from the same block2 kernel on the same
        # state: a wiring check, like the SC pair below, and the one place a
        # PC reference exists at all.
        gold_pc = read_golden_pc_total(args.golden)
        if gold_pc is None:
            raise SystemExit(
                f"--pc: {args.golden} carries no e_pc_total. That golden was "
                "generated --no-pc because its PC metric is refused as a "
                "continuum (generate_golden._check_pc_conditioning), so there "
                "is no PC reference for this state and no PC gradient to check "
                "against one (docs/reference-data.md, \"LiF and acrolein, the "
                "gradient-validation pair\")."
            )
        rec["d_e_pc"] = e_pc - gold_pc
        if abs(rec["d_e_pc"]) > REF_STATE_TOL:
            raise SystemExit(
                f"the rebuilt reference state's PC energy is {rec['d_e_pc']:.3e} "
                f"from {args.golden}'s e_pc_total (tolerance {REF_STATE_TOL:g})"
            )
    # The state differentiated must be the state the golden holds, or the
    # gradient is of a different function. Measured at exactly 0.0 on both
    # committed cases -- the same builder, the same settings, the same
    # geometry -- so a nonzero value is a wiring error, not a tolerance.
    if max(abs(rec["d_e_casci"]), abs(rec["d_e_corr"])) > REF_STATE_TOL:
        raise SystemExit(
            f"the rebuilt reference state does not match {args.golden}: "
            f"E_CASCI differs by {rec['d_e_casci']:.3e}, E_corr by "
            f"{rec['d_e_corr']:.3e} (tolerance {REF_STATE_TOL:g}). The "
            "geometry, basis or active-space selection given here is not the "
            "one that golden was built from, so its gradient would be of a "
            "different function."
        )

    want = _parse_components(args.components, len(geom))
    probed, skipped = select_components(geom, case["basis"], ref_group, args.step, want)
    rec["probed"] = probed
    rec["zero_by_symmetry"] = skipped
    print(f"[{args.case}] {len(probed)} components to difference, "
          f"{len(skipped)} zero by symmetry, "
          f"{6 * len(probed)} displaced CASCI+NEVPT2"
          f"{'+block2 PC' if args.pc else ''} runs:")
    rec.update(numgrad(geom, case, probed, args.step, ref_labels, pc=args.pc))
    rec["wall"] = time.perf_counter() - t0
    return rec


def report(rec: dict) -> None:
    print(f"== {rec['case']}  {rec['geometry']}  {rec['basis']} "
          f"CAS({rec['nelecas']},{rec['ncas']})  irreps {rec['cas_irreps']}")
    print(f"reference: group {rec['group'][0]} (used {rec['group'][1]}), core "
          f"{rec['labels'][0]} active {rec['labels'][1]}, "
          f"E_CASCI={rec['e_casci']:.10f} ({rec['d_e_casci']:+.1e} vs golden) "
          f"E_corr={rec['e_corr']:.10f} ({rec['d_e_corr']:+.1e})"
          + (f" E_corr(PC)={rec['e_pc']:.10f} ({rec['d_e_pc']:+.1e})"
             if rec.get("pc") else ""))
    print(f"step {rec['step']:g} bohr, stencil "
          + "/".join(f"{h:g}" for h in stencil(rec['step']))
          + f", {rec['wall']:.1f}s")
    grad, res, probed = rec["grad_num"], rec["grad_num_residual"], rec["grad_num_probed"]
    print(f"{'atom':>4} {'dE/dx':>16} {'dE/dy':>16} {'dE/dz':>16}   "
          "residuals (Eh/bohr; '.' = zero by symmetry)")
    for i in range(rec["natm"]):
        cells = "".join(
            f" {grad[i, c]:+16.10f}" if probed[i, c] else f" {'0 (sym)':>16}"
            for c in range(3)
        )
        bars = "  ".join(f"{res[i, c]:.1e}" if probed[i, c] else "      ." for c in range(3))
        print(f"{i:>4}{cells}   {bars}")
    print(f"max |residual| over the probed components: {res[probed > 0].max():.2e}"
          if probed.any() else "nothing probed")
    # Translational invariance: the gradient of an energy that does not depend
    # on where the molecule sits must sum to zero over the atoms. Only a
    # column every atom was probed in can say so.
    full = probed.all(axis=0)
    if full.any():
        sums = ", ".join(f"d/d{AXES[c]} {grad[:, c].sum():+.2e}"
                         for c in range(3) if full[c])
        print(f"sum over atoms (0 by translational invariance): {sums}")
    if rec.get("pc"):
        gpc, rpc = rec["grad_num_pc"], rec["grad_num_pc_residual"]
        print("PC (block2) on the same displaced states:")
        for i in range(rec["natm"]):
            cells = "".join(
                f" {gpc[i, c]:+16.10f}" if probed[i, c] else f" {'0 (sym)':>16}"
                for c in range(3)
            )
            bars = "  ".join(f"{rpc[i, c]:.1e}" if probed[i, c] else "      ."
                             for c in range(3))
            print(f"{i:>4}{cells}   {bars}")
        print(f"max |residual| over the probed components: "
              f"{rpc[probed > 0].max():.2e}; max |PC - SC| component "
              f"{np.abs(gpc - grad)[probed > 0].max():.2e}")
        if full.any():
            sums = ", ".join(f"d/d{AXES[c]} {gpc[:, c].sum():+.2e}"
                             for c in range(3) if full[c])
            print(f"sum over atoms (0 by translational invariance): {sums}")


def report_casscf(rec: dict, analytic: dict | None) -> None:
    """``--reference casscf``'s report: the reference against its committed
    files, the guard's measured maxima, the CASSCF-only free check, and then
    either the componentwise table (:func:`report`) or the screen. With
    ``analytic`` the screen is GATED here -- ``|D_d - <g, d>|`` against
    ``nevpt2_grad.GRAD_NUM_TOL`` per direction -- and ``rec["screen_pass"]``
    says how it went; componentwise, the comparison is printed and the gate
    is ``nevpt2_grad.py --grad-num``'s."""
    import nevpt2_grad  # lazily: it imports this module

    r0 = rec["ref_state"]
    print(f"== {rec['case']}  --reference casscf  {rec['basis']} "
          f"CAS({rec['nelecas']},{rec['ncas']})  irreps {rec['cas_irreps']}")
    fs = rec["finish_shift"]
    print(f"reference re-optimised by generate_golden's path: orbitals bit-identical "
          f"to the committed .mogold: {rec['mo_bitwise']} (E_CASSCF "
          f"{rec['d_e_casscf_gen']:+.1e}, macro {r0['n_macro']}+{r0['polish_macro']}, "
          f"|g_orb| {r0['g_gen']:.2e}); committed orbitals vs golden: E_CASCI "
          f"{rec['d_e_casci']:+.1e}, E_corr {rec['d_e_corr']:+.1e}")
    print(f"reference finished (the point differenced around): E_CASSCF "
          f"{rec['e_casscf']:.12f} ({fs['e_casscf']:+.1e}), E_CASCI {fs['e_casci']:+.1e}, "
          f"E_corr {rec['e_corr']:.12f} ({fs['e_corr']:+.1e}) vs the committed files; "
          f"|g_orb| {r0['g_opt']:.2e} in {r0['finish_macro']} newton macro, gate "
          f"{r0['gate'][0]:.2e}")
    m = rec["signals_max"]
    print(f"continuity guard, max over the {rec['wall_sweep']:.0f}s sweep's states: "
          f"e_jump {m['e_jump']:.2e} (<= {CONTINUITY_E_TOL:g}), |g_orb| {m['g_opt']:.2e} "
          f"(<= {gg.CASSCF_CONV_TOL_GRAD:g}), 1-sv_act {m['act_defect']:.2e} "
          f"(<= {CONTINUITY_ACT_TOL:g}), 1-|<ci0|ci>| {m['ci_defect']:.2e} "
          f"(<= {CONTINUITY_CI_TOL:g}); --mo-from gate max {rec['gate_max']:.2e} "
          f"(<= {gg.CASSCF_GRAD_TOL:g}); first run converged (no stall) at "
          f"{len(rec['first_converged'])} state(s): {', '.join(rec['first_converged']) or '-'}")
    pairs, c_scf, c_tot = rec["congruent"]
    if pairs:
        print(f"energy noise, measured on {pairs} congruent pair(s) of displaced states "
              f"(mirror images / translations): E_CASSCF {c_scf:.1e}, E_CASCI + E_corr "
              f"{c_tot:.1e} -- over the finest 2h, {c_tot / (2 * rec['step'] / 4):.1e} "
              "Eh/bohr in a central difference")
    tol = nevpt2_grad.GRAD_NUM_TOL
    if "dir_num" in rec:
        free = np.abs(rec["dir_casscf_num"] - rec["dir_casscf_pyscf"]).max()
        print(f"free check: CASSCF-only directional derivative vs pyscf.grad.casscf, "
              f"max |delta| {free:.2e}")
        print(f"{'dir':>4} {'numerical':>16} {'residual':>9} {'<g, d>':>16} {'|delta|':>9}")
        ok = True
        for n, (num, res) in enumerate(zip(rec["dir_num"], rec["dir_residual"], strict=True)):
            an = float(np.sum(analytic["grad"] * rec["dir_vectors"][n])) if analytic else None
            delta = abs(num - an) if an is not None else float("nan")
            ok = ok and delta <= tol
            print(f"  d{n} {num:+16.10f} {res:9.1e} "
                  + (f"{an:+16.10f} {delta:9.1e}" if an is not None else "(no --analytic)"))
        rec["screen_pass"] = ok if analytic else None
        budget = SCREEN_WALL_BUDGET
        verdict = ("PASS" if ok else "FAIL") if analytic else "not gated"
        print(f"{verdict}: the directional screen, {len(rec['dir_num'])} directions, "
              f"max residual {rec['dir_residual'].max():.1e}, allowed |delta| {tol:g}")
    else:
        probed = rec["grad_num_probed"] > 0
        free = np.abs(rec["grad_casci"] - rec["g_casscf"])[probed].max()
        print(f"free check: CASSCF-only numerical gradient vs pyscf.grad.casscf, "
              f"max |delta| {free:.2e}")
        report(rec)
        if analytic is not None:
            d = np.abs(analytic["grad"] - rec["grad_num"])[probed]
            print(f"against --analytic: max |g - grad_num| {d.max():.2e} (the gate, "
                  f"{tol:g}, is nevpt2_grad.py --grad-num's)")
        budget = GATE_WALL_BUDGET.get(rec["case"], float("nan"))
    over = rec["wall"] > budget
    print(f"wall {rec['wall']:.1f}s, declared budget {budget:g}s"
          + (" -- OVER BUDGET (a bound, not a gate)" if over else ""))
    for w in rec.get("warm", []):
        if not w.get("refused"):
            print(f"cold vs warm at {w['label']}: dE_CASSCF {w['d_e']:+.1e}, 1-sv "
                  f"{w['act_defect']:.1e}, 1-ci {w['ci_defect']:.1e}: "
                  + ("the same stationary point" if w["same"] else "DIFFERENT"))
    print(f"states and their MO sidecars: {rec['states_dir']}/states.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="lif_def2svp_cas46",
                    help=f"a committed case: {', '.join(CASES)} (default: "
                         "lif_def2svp_cas46). Anything else needs --geometry, "
                         "--basis, --ncas and --nelecas")
    ap.add_argument("--geometry", default=None,
                    help="xyz file (default: the --case geometry); relative "
                         "paths resolve against the repo root")
    ap.add_argument("--basis", default=None)
    ap.add_argument("--ncas", type=int, default=None)
    ap.add_argument("--nelecas", type=int, default=None)
    ap.add_argument("--cas-irreps", type=gg._irrep_counts, default=None,
                    help="active orbitals per irrep, e.g. 'B1=3,B2=3' "
                         "(default: the --case selection)")
    ap.add_argument("--core-irreps", type=gg._irrep_counts, default=None,
                    help="core orbitals per irrep, same syntax")
    ap.add_argument("--step", type=float, default=STEP_BOHR,
                    help=f"coarse central-difference step in bohr (default "
                         f"{STEP_BOHR:g}); the stencil also uses h/2 and h/4")
    ap.add_argument("--components", default="auto",
                    help="'auto' (default: every component whose displacement "
                         "keeps the detected point group; the rest are the "
                         "zeros symmetry makes them), 'all', or a list like "
                         "'0z,3x'. A named component whose displacement "
                         "changes the point group is REFUSED")
    ap.add_argument("--golden", type=Path, default=None,
                    help="default: golden/<case>.nevpt2gold")
    ap.add_argument("--sidecar", type=Path, default=None,
                    help="the gradient sidecar to add grad_num to (default: "
                         "golden/<case>.gradgold, written by "
                         "generate_golden.py --gradgold)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the updated sidecar here instead of in place")
    ap.add_argument("--no-write", action="store_true",
                    help="compute and print only; touch no file")
    ap.add_argument("--pc", action="store_true",
                    help="difference block2's PARTIALLY-contracted energy too, "
                         "on the same displaced states, and check each one's PC "
                         "conditioning. Writes no sidecar (use --npz); far "
                         "slower than the SC sweep it rides along with, and "
                         "refused on a golden with no PC fields (LiF)")
    ap.add_argument("--npz", type=Path, default=None,
                    help="write the gradient arrays here as a plain .npz "
                         "(grad_num, its residual, the probed mask, the step, "
                         "grad_casci, and with --pc the two grad_num_pc "
                         "arrays). This is what nevpt2_grad.py --grad-num "
                         "reads, and the only way a PC numerical gradient "
                         "leaves this script: no committed byte moves")
    ap.add_argument("--reference", default="casci", choices=REFERENCES,
                    help="'casci' (default): displaced RHF -> CASCI states, "
                         "differencing golden/<case>.nevpt2gold. 'casscf': every "
                         "displaced state re-OPTIMISED, cold, through "
                         "generate_golden's own optimiser, then the --mo-from "
                         "CASCI on it, differencing golden/<case>_casscf."
                         "nevpt2gold; guarded by the continuity check. SC only; "
                         "writes no sidecar (use --npz)")
    ap.add_argument("--mo-from", type=Path, default=None,
                    help="--reference casscf: the committed MO sidecar the "
                         "re-optimised reference must reproduce (default: "
                         "golden/<case>_casscf.mogold)")
    ap.add_argument("--directional", type=int, default=0, metavar="N",
                    help="--reference casscf: the directional SCREEN instead of "
                         "the componentwise sweep -- the derivative along N "
                         f"seeded random in-plane directions (the screen uses "
                         f"{SCREEN_DIRECTIONS}), held to <g, d> from --analytic")
    ap.add_argument("--analytic", type=Path, default=None,
                    help="--directional: the analytic gradient to hold the "
                         "screen to, an `nevpt2_grad.py --reference casscf --out "
                         "PATH` file")
    ap.add_argument("--workers", type=int, default=1,
                    help="--reference casscf: displaced states optimised in "
                         "parallel, one single-threaded process each (the "
                         "result does not depend on it)")
    ap.add_argument("--states-dir", type=Path, default=None,
                    help="--reference casscf: where each displaced state's MO "
                         "sidecar and states.json go, for reuse by later "
                         "consumers (default: a fresh temporary directory)")
    ap.add_argument("--warm-probe", type=int, default=None, metavar="N",
                    help="--reference casscf: re-optimise the first N coarse-step "
                         "geometries warm-started from the reference's orbitals "
                         "and refuse any that lands elsewhere than its cold "
                         f"state (default {WARM_PROBE_DEFAULT}; 0 skips it)")
    ap.add_argument("--pc-states", type=Path, default=None, metavar="NPZ",
                    help="--reference casscf: run NO sweep; difference the GPU's PC "
                         "energies on a finished sweep's kept states (the keyed "
                         ".npz pc_on_states.py writes) with this stencil, into "
                         "grad_num_pc for nevpt2_grad.py --reference casscf --pc "
                         "--grad-num (#126). Refused if any state lacks a guarded "
                         "PC energy. --analytic is then a --pc gradient, and the "
                         "screen directions in the file are gated against it")
    args = ap.parse_args()
    casscf = args.reference == "casscf"
    if not casscf:
        named = [f for f, on in (("--mo-from", args.mo_from), ("--directional",
                 args.directional), ("--analytic", args.analytic), ("--states-dir",
                 args.states_dir), ("--warm-probe", args.warm_probe),
                 ("--pc-states", args.pc_states)) if on]
        if named:
            ap.error(f"{', '.join(named)} belong to --reference casscf")
    else:
        if args.pc:
            ap.error("--reference casscf is SC only: the PC gradient of a CASSCF "
                     "reference differences the GPU's PC energies on this sweep's "
                     "kept states (pc_on_states.py, then --pc-states NPZ) rather "
                     "than running block2 here")
        if args.pc_states is not None and (args.directional or args.states_dir):
            ap.error("--pc-states differences a finished sweep's PC energies; it runs "
                     "no sweep, so --directional and --states-dir do not apply")
        if args.npz is None and not args.no_write:
            ap.error("--reference casscf writes no sidecar (the committed "
                     "<case>_casscf.gradgold is the 24 --gradgold arrays and stays "
                     "byte-identical): pass --npz PATH or --no-write")
        if args.directional and args.analytic is None:
            ap.error("--directional needs --analytic PATH: the screen holds the "
                     "numerical directional derivative to <g, d>")
        if args.warm_probe is None:
            args.warm_probe = WARM_PROBE_DEFAULT
    stem = f"{args.case}_casscf" if casscf else args.case
    if args.golden is None:
        args.golden = REPO / "golden" / f"{stem}.nevpt2gold"
    if args.sidecar is None:
        args.sidecar = REPO / "golden" / f"{stem}.gradgold"
    if casscf and args.mo_from is None:
        args.mo_from = REPO / "golden" / f"{stem}.mogold"
    if args.pc_states is not None:
        # #126: no sweep -- the PC energies of one that already ran, differenced.
        analytic = load_analytic(args.analytic, args.case, pc=True) if args.analytic else None
        rec = difference_pc_states(args.pc_states, args.case)
        ok = report_pc_states(rec, analytic)
        if args.npz is not None:
            names = [n for n in ("grad_num_pc", "grad_num_pc_residual", "grad_num",
                                 "grad_num_residual", "grad_num_sc_gpu", "grad_num_probed",
                                 "grad_num_step", "dir_vectors", "dir_num_pc",
                                 "dir_residual_pc", "dir_num", "dir_residual") if n in rec]
            np.savez(args.npz, case=np.array(rec["case"]), pc=np.array(True),
                     reference=np.array("casscf"), **{n: rec[n] for n in names})
            print(f"wrote {args.npz} ({len(names) + 3} arrays)")
        if not ok:
            raise SystemExit(1)
        return
    analytic = load_analytic(args.analytic, args.case) if args.analytic else None

    rec = run(args)
    if casscf:
        report_casscf(rec, analytic)
        if args.npz is not None:
            names = [n for n in ("grad_num", "grad_num_residual", "grad_num_probed",
                                 "grad_num_step", "grad_casci", "g_casscf",
                                 "dir_vectors", "dir_num", "dir_residual",
                                 "dir_casscf_num", "dir_casscf_pyscf", "state_labels",
                                 "state_gate", "state_g_opt", "state_e_jump",
                                 "state_act_defect", "state_ci_defect") if n in rec]
            np.savez(args.npz, case=np.array(rec["case"]), pc=np.array(False),
                     reference=np.array("casscf"), **{n: rec[n] for n in names})
            print(f"wrote {args.npz} ({len(names) + 3} arrays)")
        if rec.get("screen_pass") is False:
            raise SystemExit(1)
        return
    report(rec)
    if args.npz is not None:
        names = ["grad_num", "grad_num_residual", "grad_num_probed",
                 "grad_num_step", "grad_casci"]
        if args.pc:
            names += ["grad_num_pc", "grad_num_pc_residual"]
        np.savez(args.npz, case=np.array(rec["case"]), pc=np.array(args.pc),
                 **{n: rec[n] for n in names})
        print(f"wrote {args.npz} ({len(names) + 2} arrays)")
    if args.pc:
        # The sidecar holds the 24 arrays --gradgold writes plus the four SC
        # grad_num* ones, and the committed files hold exactly that; a PC
        # gradient has no home there (see the module docstring).
        print("--pc: the sidecar is unchanged")
        return
    if args.no_write:
        print("--no-write: the sidecar is unchanged")
        return
    if not CASES.get(args.case, {}).get("sidecar_grad_num", True):
        # Some sidecars are committed holding only the 24 arrays --gradgold
        # writes (N2 CAS(4,4)). Adding grad_num* to one of those would move a
        # committed golden's bytes, which is never what a run of this script
        # should do silently.
        raise SystemExit(
            f"golden/{args.case}.gradgold is committed as the 24 arrays "
            "generate_golden.py --gradgold writes, with no grad_num in it, so "
            "writing one there would move a committed file's bytes. Pass "
            "--npz PATH (which is what nevpt2_grad.py --grad-num reads), "
            "--out PATH, or --no-write."
        )

    header, arrays = gf.read_gradgold(args.sidecar)
    # Replace rather than append on a re-run, so the file is a function of the
    # inputs: same order, same bytes.
    arrays = {k: v for k, v in arrays.items() if not k.startswith("grad_num")}
    for name in ("grad_num", "grad_num_residual", "grad_num_probed", "grad_num_step"):
        arrays[name] = rec[name]
    out = args.out or args.sidecar
    gg.write_gradgold(out, **header, arrays=arrays)
    print(f"wrote {out} ({len(arrays)} arrays, "
          f"{out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
