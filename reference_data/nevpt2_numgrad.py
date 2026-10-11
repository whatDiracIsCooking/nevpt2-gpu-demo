#!/usr/bin/env python3
r"""Numerical nuclear gradient of the SC-NEVPT2 energy, by displaced CASCI runs.

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
displaced geometry. Nothing here is partially contracted: PC is block2's
answer and LiF has none (docs/reference-data.md, "LiF and acrolein, the
gradient-validation pair").

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
``reference_kind`` is not ``"casci"`` is **refused**
(:func:`_check_reference_kind`), before any energy is computed.

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
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads.
import generate_golden as gg

# isort: split
import argparse
import re
import struct
import time
from pathlib import Path

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

#: The cases with a committed golden + sidecar, keyed by their golden's name.
#: These MUST stay the ``--gradgold`` invocations in docs/reference-data.md,
#: "Regenerating the golden file": the gradient is of the state the sidecar
#: beside it describes, so a drifted geometry, basis or active space here
#: would produce a number for a state no file holds.
CASES = {
    "lif_def2svp_cas46": dict(
        geometry="reference_data/geometries/lif.xyz",
        basis="def2-svp",
        ncas=6,
        nelecas=4,
        cas_irreps={"B1": 3, "B2": 3},
        core_irreps=None,
    ),
    "acrolein_def2svp_cas65": dict(
        geometry="reference_data/geometries/acrolein.xyz",
        basis="def2-svp",
        ncas=5,
        nelecas=6,
        cas_irreps={"A'": 1, 'A"': 4},
        core_irreps=None,
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


def read_golden_header(path: Path) -> dict:
    """The scalar header of a ``.nevpt2gold`` file -- the two energies this
    script checks its rebuilt reference state against, and the reference kind
    :func:`_check_reference_kind` gates on (layout:
    ``generate_golden.write_golden``).

    Both magics are accepted: a pre-kind ``NEVPT2G1`` file carries no
    ``reference_kind`` field and is a CASCI by definition, which is what every
    committed golden is."""
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic in gg.GOLDEN_MAGICS, f"{path} is not a golden file"
        ncas, nelec_a, nelec_b, ncore, ndet = struct.unpack("<iiiiq", f.read(24))
        e_casci, e_corr = struct.unpack("<dd", f.read(16))
        kind = "casci"
        if magic == gg.GOLDEN_MAGIC:
            (code,) = struct.unpack("<i", f.read(4))
            assert code in gg.REFERENCE_KINDS, f"{path}: unknown reference_kind {code}"
            kind = gg.REFERENCE_KINDS[code]
    return dict(
        ncas=ncas, nelec_a=nelec_a, nelec_b=nelec_b, ncore=ncore, ndet=ndet,
        e_casci=e_casci, e_corr=e_corr, reference_kind=kind,
    )


def _check_reference_kind(kind: str, path: Path) -> None:
    """Refuse a golden whose orbitals did not come from a CASCI on converged
    RHF orbitals. Everything here differentiates *that* reference: RHF
    followed by a CASCI diagonalisation is one deterministic function of the
    geometry, so two displaced energies are two values of it. A CASSCF
    reference is a different Lagrangian -- CP-CASSCF rather than RHF
    Brillouin/CPHF plus an active-space selection (docs/gradient-theory.md,
    §1.1, §1.2 and §5) -- and differencing one needs a different strategy
    anyway, since each displaced geometry lands on a differently-rotated
    solution of the same energy (§4.4). Refused, not warned: the number it
    would produce looks like a gradient. (``apps/integral_direct``'s
    ``--sc-tangent`` and ``--gradient-source`` refuse the same input, as an
    ``err_unsupported``.)"""
    if kind != "casci":
        raise SystemExit(
            f"{path}: this golden's reference_kind is {kind!r}, and this "
            "script differentiates a CASCI on converged RHF orbitals. A "
            "CASSCF reference needs CP-CASSCF and a different "
            "numerical-gradient strategy -- each displaced geometry would "
            "land on a differently-rotated solution of the same energy, so a "
            "difference quotient of it is not a gradient "
            "(docs/gradient-theory.md, §1.1 and §4.4). Refused rather than "
            "differenced."
        )


def read_gradgold(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """A ``.gradgold`` sidecar back into the ``(header, arrays)`` pair
    :func:`generate_golden.write_gradgold` writes, so this script can append
    its arrays and write the file out again unchanged otherwise."""
    with open(path, "rb") as f:
        assert f.read(8) == gg.GRADGOLD_MAGIC, f"{path} is not a gradient sidecar"
        names = ("nao", "nmo", "ncore", "nact", "nvirt", "nelec_rhf")
        header = dict(zip(names, struct.unpack("<6i", f.read(24)), strict=True))
        (n_arrays,) = struct.unpack("<i", f.read(4))
        arrays: dict[str, np.ndarray] = {}
        for _ in range(n_arrays):
            (name_len,) = struct.unpack("<i", f.read(4))
            name = f.read(name_len).decode()
            (ndim,) = struct.unpack("<i", f.read(4))
            shape = struct.unpack(f"<{ndim}q", f.read(8 * ndim))
            count = int(np.prod(shape, dtype=np.int64))
            arrays[name] = np.frombuffer(f.read(8 * count), "<f8").reshape(shape)
        assert not f.read(1), f"{path}: trailing bytes after the last array"
    return header, arrays


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


def energy(geom, case: dict) -> tuple[float, float, tuple]:
    """One displaced point: ``(E_CASCI, E_corr(SC-NEVPT2), selection
    signature)``, from ``generate_golden``'s own state builder and PySCF's own
    per-class energies -- the golden's answer, at a displaced geometry."""
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
    return float(mc.e_tot), e_corr, selection_labels(pt)


def numgrad(geom, case: dict, probed, step_bohr: float, ref_labels) -> dict:
    """Every probed component's gradient, by the six-point stencil. Returns
    the two ``(natm, 3)`` arrays the sidecar carries plus the CASCI-only
    gradient, which is not stored but is printed: it localises a later
    disagreement to the reference or to the perturbation."""
    natm = len(geom)
    steps = stencil(step_bohr)
    grad = np.zeros((natm, 3))
    grad_casci = np.zeros((natm, 3))
    residual = np.zeros((natm, 3))
    probed_mask = np.zeros((natm, 3))
    for iatom, comp in probed:
        tot: dict[float, float] = {}
        cas: dict[float, float] = {}
        for h in steps:
            e = {}
            for sign in (+1, -1):
                moved = displace(geom, iatom, comp, sign * h)
                e_casci, e_corr, labels = energy(moved, case)
                _check_selection(
                    ref_labels, labels,
                    f"atom {iatom} ({geom[iatom][0]}) {AXES[comp]}, step "
                    f"{sign * h:+g} bohr",
                )
                e[sign] = (e_casci, e_corr)
            tot[h] = ((e[1][0] + e[1][1]) - (e[-1][0] + e[-1][1])) / (2 * h)
            cas[h] = (e[1][0] - e[-1][0]) / (2 * h)
        grad[iatom, comp], residual[iatom, comp] = richardson(tot, steps)
        grad_casci[iatom, comp], _ = richardson(cas, steps)
        probed_mask[iatom, comp] = 1.0
        print(
            f"  {iatom} {geom[iatom][0]:<2} {AXES[comp]}  "
            f"grad {grad[iatom, comp]:+.10f}  residual "
            f"{residual[iatom, comp]:.1e}  (CASCI only "
            f"{grad_casci[iatom, comp]:+.10f}; raw "
            + " ".join(f"{tot[h]:+.8f}" for h in steps)
            + ")"
        )
    return dict(
        grad_num=grad, grad_num_residual=residual, grad_num_probed=probed_mask,
        grad_num_step=np.array([step_bohr]), grad_casci=grad_casci,
    )


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
    gold = read_golden_header(args.golden)
    rec["reference_kind"] = gold["reference_kind"]
    _check_reference_kind(gold["reference_kind"], args.golden)

    t0 = time.perf_counter()
    ref_group = point_group(geom, case["basis"])
    e_casci, e_corr, ref_labels = energy(geom, case)
    rec.update(group=ref_group, e_casci=e_casci, e_corr=e_corr, labels=ref_labels)

    rec["d_e_casci"] = e_casci - gold["e_casci"]
    rec["d_e_corr"] = e_corr - gold["e_corr"]
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
          f"{6 * len(probed)} displaced CASCI+NEVPT2 runs:")
    rec.update(numgrad(geom, case, probed, args.step, ref_labels))
    rec["wall"] = time.perf_counter() - t0
    return rec


def report(rec: dict) -> None:
    print(f"== {rec['case']}  {rec['geometry']}  {rec['basis']} "
          f"CAS({rec['nelecas']},{rec['ncas']})  irreps {rec['cas_irreps']}")
    print(f"reference: group {rec['group'][0]} (used {rec['group'][1]}), core "
          f"{rec['labels'][0]} active {rec['labels'][1]}, "
          f"E_CASCI={rec['e_casci']:.10f} ({rec['d_e_casci']:+.1e} vs golden) "
          f"E_corr={rec['e_corr']:.10f} ({rec['d_e_corr']:+.1e})")
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
    args = ap.parse_args()
    if args.golden is None:
        args.golden = REPO / "golden" / f"{args.case}.nevpt2gold"
    if args.sidecar is None:
        args.sidecar = REPO / "golden" / f"{args.case}.gradgold"

    rec = run(args)
    report(rec)
    if args.no_write:
        print("--no-write: the sidecar is unchanged")
        return

    header, arrays = read_gradgold(args.sidecar)
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
