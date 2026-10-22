#!/usr/bin/env python3
r"""The NEVPT2 nuclear gradient at an arbitrary geometry, in one call:
:func:`gradient_at`, GPU-sourced, SC or (``pc=True``) PC, on a CASSCF
reference.

It is OFFLINE tooling like the rest of ``reference_data/``: no build and no
ctest runs it. It needs a built ``nevpt2_sc_pseudodensity`` and a card. Its
callers are the seminumerical nuclear Hessian (#217), which differences it
over the Cartesian coordinates that keep the point group, and the geometry
optimiser (#218), which follows it downhill; each passes the previous call's
orbitals as the next call's guess.

It chains four steps that already exist, and copies none of them:

1. **The CASSCF state at this geometry**, re-optimised from the guess
   orbitals: ``nevpt2_numgrad.py --reference casscf``'s own path --
   :func:`nevpt2_numgrad.warm_start` (the guess carried into this geometry's
   AO basis, irrep-projected), ``generate_golden.casscf_optimise`` and
   :func:`nevpt2_numgrad.finish_casscf` -- then that sweep's **continuity
   guard** against the guess (:func:`nevpt2_numgrad.continuity_signals`; the
   energy signal under the rule in :func:`continuity_verdict`), and its
   selection refusal (:func:`nevpt2_numgrad._check_selection`) against the
   reference's. The finished orbitals go to a ``.mogold`` in the work dir.
2. **A transient golden plus its ``.gradgold`` sidecar**, in a work dir that
   is never under ``golden/`` (:func:`check_work_dir`):
   ``generate_golden.generate`` with ``--mo-from`` that ``.mogold`` and
   ``--gradgold`` (:func:`write_transient`, shared with ``pc_on_states.py``).
   With ``pc`` it carries block2's PC fields, which the PC dump is held to.
3. **The GPU's source terms**: ``nevpt2_sc_pseudodensity --dump-source-terms``
   (``--pc`` for PC) on that golden, under the card's lock (:func:`gpu_lock`).
4. **The Z-vector solve and the AO contraction**: ``nevpt2_grad.py
   --reference casscf --source dump``'s own -- :func:`nevpt2_grad.build_state`
   at this geometry on that ``.mogold``, :func:`nevpt2_grad.load_source_terms`
   and :func:`nevpt2_grad.solve_gradient`, the coupled CP-CASSCF solve that
   ``nevpt2_grad.py``'s ``--source both`` runs for its second provider.

**What it refuses**, each a ``SystemExit`` naming why, as every script here
does:

* a geometry whose detected point group is not the reference geometry's
  (:func:`nevpt2_numgrad._check_point_group`), before anything is computed:
  the active space is selected by irrep, and the labels do not survive a
  lowered symmetry (docs/gradient-theory.md, section 4.3);
* a state whose core or active irreps are not the reference's
  (:func:`nevpt2_numgrad._check_selection`), after the CASSCF;
* a work dir inside ``golden/``;
* a CASSCF the continuity guard trips: unconverged, an active space or CI
  vector that moved from the guess's past the sweep's thresholds, or (within
  the stencil) an energy off its first-order prediction;
* a transient golden that does not reproduce the state the solve rebuilds
  (``nevpt2_numgrad.REF_STATE_TOL``), a dump of another state, a PC metric
  the conditioning guard refuses at this geometry, and a PC guard ledger
  that skipped a displaced CI vector or saw a rank change.

**What it returns**: :class:`GradientAt` -- the gradient ``(natm, 3)``
Eh/bohr, the total energy it is the gradient of (``E_CASCI + E_corr``, SC or
PC), the converged orbitals as a :class:`MoGuess` for the next geometry, and
the guard ledger: every number each refusal above read, so a caller can say
why a geometry was taken.

**The reference.** ``case`` is a committed ``nevpt2_numgrad.CASES`` entry with
a committed ``golden/<case>_casscf.mogold``: its geometry fixes the point
group and those orbitals fix the core/active irreps every call is held to,
and they are the default guess. At the reference geometry with that guess the
optimiser starts converged, and the finish then moves the orbitals the way the
numerical sweep's own reference state is moved (docs/reference-data.md, "Why
the displaced states are finished"); the gradient is of the finished state.

The measured agreement with the committed gradients, and with
``nevpt2_grad.py`` run by hand at a displaced geometry: docs/reference-data.md,
"The gradient at an arbitrary geometry".

Usage (the CLI is one call; ``--grad-num`` gates it against a numerical
gradient at the reference geometry, ``--compare`` against a ``nevpt2_grad.py
--out`` file)::

    uv run python reference_data/gradient_at.py --case acrolein_def2svp_cas65 \
        --binary build/src/nevpt2_sc_pseudodensity --workers 6 \
        --gpu-lock /tmp/nevpt2-gpu-cuda.lock --grad-num /tmp/acrolein_casscf.numgrad.npz
    uv run python reference_data/gradient_at.py --case acrolein_def2svp_cas65 --pc \
        --geometry /tmp/acrolein_displaced.xyz --binary build/src/nevpt2_sc_pseudodensity \
        --work-dir /tmp/gat --keep --out /tmp/acrolein_displaced_pc.gat.npz
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads.
import generate_golden as gg

# isort: split
import argparse
import contextlib
import fcntl
import io
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import nevpt2_grad as ngr
import nevpt2_numgrad as ngrad
import numpy as np
from pyscf.data.nist import BOHR

REPO = Path(__file__).resolve().parent.parent

#: The references :func:`gradient_at` differentiates. A CASCI-on-RHF state
#: has no orbital guess to carry and no continuity to guard; its gradient is
#: ``nevpt2_grad.py --reference casci``'s, and it is not wired here.
REFERENCES = ("casscf",)

#: The tile count every dump runs at (CLAUDE.md, "Tile counts are measured,
#: not guessed": a small CAS takes the shipped 3).
TILES = 3

#: Where :func:`find_binary` looks for ``nevpt2_sc_pseudodensity`` when none
#: is named, in this order: the ``default`` (CUDA) and ``hip`` presets'.
BINARIES = ("build/src/nevpt2_sc_pseudodensity", "build-hip/src/nevpt2_sc_pseudodensity")

#: How far from the guess's geometry the continuity guard's ENERGY signal is
#: enforced, per Cartesian component, bohr: the stencil its threshold was
#: calibrated on (``nevpt2_numgrad.STEP_BOHR``, the coarse step;
#: docs/reference-data.md, "The continuity guard's thresholds"). Beyond it the
#: second-order remainder alone can exceed ``CONTINUITY_E_TOL``, so the signal
#: is not evaluated there and the ledger says so (:func:`continuity_verdict`).
ENERGY_GUARD_REACH = ngrad.STEP_BOHR


# --------------------------------------------------------------------------
# The work dir, the transient golden and the card (shared with pc_on_states)
# --------------------------------------------------------------------------


def check_work_dir(work_dir: Path) -> None:
    """Refuse a work dir inside ``golden/``: the transient goldens are never
    committed and never sit beside files that are."""
    golden = (REPO / "golden").resolve()
    w = Path(work_dir).resolve()
    if w == golden or golden in w.parents:
        raise SystemExit(f"--work-dir {work_dir} is inside {golden}: the transient "
                         "goldens are never written there")


def with_placeholder_pc(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A ``--no-pc`` golden's arrays plus NaN for block2's two PC fields.
    ``nevpt2_demo --pc`` refuses a golden without them, and a NaN matches no
    energy, so the file can never pass for a block2 reference."""
    if "pc_class_energies" in arrays or "e_pc_total" in arrays:
        raise SystemExit("with_placeholder_pc: this golden already has PC fields")
    return dict(arrays, pc_class_energies=np.full(len(gg.CLASSES), np.nan),
                e_pc_total=np.array([np.nan]))


def write_transient(atom: str, case: dict, mogold: Path, out: Path, *, name: str,
                    pc: bool, gradgold: bool = False,
                    placeholder_pc: bool = False) -> dict:
    """One transient golden at ``out``: the in-process ``generate_golden.py
    --atom ATOM --mo-from MOGOLD [--gradgold] [--no-pc]``, every refusal of
    ``_build_state`` in force, its output swallowed. ``pc`` runs block2 for the
    PC fields; ``placeholder_pc`` writes NaN ones instead
    (:func:`with_placeholder_pc`, ``pc_on_states.py``'s); ``gradgold`` writes
    the ``.gradgold`` sidecar beside it. Returns the golden's ``meta``."""
    if placeholder_pc and pc:
        raise SystemExit("write_transient: placeholder PC fields are for a --no-pc golden")
    with contextlib.redirect_stdout(io.StringIO()):
        meta, arrays, grad, _full = gg.generate(
            atom, case["basis"], case["ncas"], case["nelecas"], name, pc=pc,
            cas_irreps=case["cas_irreps"], core_irreps=case["core_irreps"],
            gradgold=gradgold, mo_from=mogold,
        )
        gg.write_golden(Path(out), **meta,
                        arrays=with_placeholder_pc(arrays) if placeholder_pc else arrays)
        if grad is not None:
            gg.write_gradgold(Path(out).with_suffix(".gradgold"), **grad[0], arrays=grad[1])
    return meta


@contextlib.contextmanager
def gpu_lock(path: str | None):
    """Hold ``flock(path)`` for the GPU stage, the box's per-card lock, so
    another agent's runs on the same card wait rather than interleave."""
    if path is None:
        yield
        return
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def find_binary(binary: Path | None) -> Path:
    """The ``nevpt2_sc_pseudodensity`` to dump with: ``binary`` if given,
    else the first of :data:`BINARIES` that exists."""
    if binary is not None:
        if not Path(binary).is_file():
            raise SystemExit(f"{binary}: no such nevpt2_sc_pseudodensity")
        return Path(binary)
    for rel in BINARIES:
        if (REPO / rel).is_file():
            return REPO / rel
    raise SystemExit("no built nevpt2_sc_pseudodensity in " + " or ".join(BINARIES)
                     + ": build it, or name one")


# --------------------------------------------------------------------------
# The reference, the guess and the guard
# --------------------------------------------------------------------------


def case_spec(case: str) -> dict:
    """A committed case's spec (``nevpt2_numgrad.CASES``), refusing one with
    no committed ``golden/<case>_casscf.mogold`` -- the reference every call
    is held to."""
    if case not in ngrad.CASES:
        raise SystemExit(f"case {case!r} is not one of {sorted(ngrad.CASES)}")
    mogold = REPO / "golden" / f"{case}_casscf.mogold"
    if not mogold.exists():
        raise SystemExit(f"case {case!r} has no committed {mogold.relative_to(REPO)}: "
                         "there is no CASSCF reference to hold the point group and the "
                         "selection to")
    return dict(ngrad.CASES[case], case=case, mogold=mogold)


def reference_geometry(spec: dict) -> list:
    """The case's own geometry, Angstrom (``nevpt2_numgrad.read_xyz``)."""
    path = Path(spec["geometry"])
    return ngrad.read_xyz(path if path.is_absolute() else REPO / path)


def as_geometry(geom) -> list:
    """``geom`` as ``[(symbol, (x, y, z))]`` floats, Angstrom: the form every
    builder here takes. Accepts that form or a ``(natm, 3)`` array beside the
    symbols (``(symbols, coords)``)."""
    if isinstance(geom, tuple) and len(geom) == 2 and not isinstance(geom[0], str):
        symbols, xyz = geom
        return [(str(s), tuple(float(v) for v in row))
                for s, row in zip(symbols, np.asarray(xyz, dtype=float), strict=True)]
    return [(str(s), tuple(float(v) for v in xyz)) for s, xyz in geom]


def coords_bohr(geom) -> np.ndarray:
    """A geometry's coordinates, ``(natm, 3)`` bohr."""
    return np.array([list(xyz) for _s, xyz in geom], dtype=float) / BOHR


def _check_atoms(ref: list, geom: list, what: str) -> None:
    """The same atoms in the same order, or the two geometries are not two
    points of one surface."""
    a, b = [s for s, _ in ref], [s for s, _ in geom]
    if a != b:
        raise SystemExit(f"{what}: atoms {b} are not the reference's {a}, in order")


_REF_LABELS: dict[str, tuple] = {}


def reference_labels(spec: dict) -> tuple:
    """The core/active irrep labels of the committed CASSCF orbitals at the
    case's geometry (:func:`nevpt2_numgrad.selection_labels`), the selection
    every call's state is held to. Cached per case: it reads the committed
    ``.mogold`` once."""
    name = spec["case"]
    if name not in _REF_LABELS:
        side = gg.read_mo_sidecar(spec["mogold"])
        _REF_LABELS[name] = state_labels(reference_geometry(spec), spec["basis"],
                                         side["mo_coeff"], side["ncore"], side["ncas"])
    return _REF_LABELS[name]


def state_labels(geom, basis: str, mo: np.ndarray, ncore: int, ncas: int) -> tuple:
    """:func:`nevpt2_numgrad.selection_labels` on a set of orbitals. The core
    and active irreps are read off the CASSCF orbitals directly: the
    canonicalisation ``nevpt2_numgrad.py`` reads them after rotates within
    each space and within each irrep, so the sorted labels are the same."""
    mol = gg._build_mol(geom, basis)
    return ngrad.selection_labels(SimpleNamespace(mol=mol, mo_coeff=mo, ncore=ncore,
                                                  ncas=ncas))


@dataclass
class MoGuess:
    """Converged CASSCF orbitals at one geometry: the guess the next
    geometry's optimiser starts from, and what its continuity guard is held
    against. ``geom`` is where they were converged (Angstrom), ``ci`` their
    CI vector, ``e_casscf`` their energy. ``g_casscf`` is
    ``pyscf.grad.casscf`` there, computed only when the energy signal needs it
    (:meth:`casscf_gradient`)."""

    geom: list
    mo_coeff: np.ndarray
    ci: np.ndarray
    e_casscf: float
    origin: str
    g_casscf: np.ndarray | None = None

    @classmethod
    def from_mogold(cls, path: Path, spec: dict) -> MoGuess:
        """A ``.mogold``'s orbitals at the geometry they were converged at
        (its ``geom`` array), with the CI vector of the golden's own
        ``--mo-from`` CASCI on them -- which also runs that path's gate, so a
        sidecar that is not a converged state of this case is refused here."""
        side = gg.read_mo_sidecar(Path(path))
        symbols = side["fields"]["symbols"].split()
        geom = as_geometry((symbols, np.asarray(side["geom"]) * BOHR))
        with contextlib.redirect_stdout(io.StringIO()):
            mc, _pt = gg._build_state(
                geom, spec["basis"], spec["ncas"], spec["nelecas"],
                cas_irreps=spec["cas_irreps"], core_irreps=spec["core_irreps"],
                mo_from=Path(path))
        return cls(geom=geom, mo_coeff=np.asarray(side["mo_coeff"]),
                   ci=np.asarray(mc.ci), e_casscf=float(side["e_casscf"]),
                   origin=str(path))

    def casscf_gradient(self, spec: dict) -> np.ndarray:
        """``pyscf.grad.casscf`` at these orbitals and CI vector
        (:func:`nevpt2_numgrad.casscf_gradient`), cached: the slope the energy
        signal's first-order prediction is taken along."""
        if self.g_casscf is None:
            from pyscf import scf

            mol = gg._build_mol(self.geom, spec["basis"])
            mf = scf.RHF(mol)
            mf.kernel()
            mcs = gg._casscf(mf, spec["ncas"], spec["nelecas"], None)
            mcs.mo_coeff = self.mo_coeff
            mcs.ci = self.ci
            with contextlib.redirect_stdout(io.StringIO()):
                self.g_casscf = ngrad.casscf_gradient(mcs)
        return self.g_casscf


def continuity_verdict(sig: dict, *, energy_enforced: bool) -> list[str]:
    """Which continuity signals a state trips (:func:`nevpt2_numgrad.
    continuity_tripped`), with the energy signal counted only when
    ``energy_enforced``: within :data:`ENERGY_GUARD_REACH` of the guess's
    geometry, the stencil its threshold was calibrated on. Outside it the
    signal is not evaluated (``e_jump`` NaN), and the convergence and overlap
    signals still are."""
    tripped = ngrad.continuity_tripped(sig)
    return tripped if energy_enforced else [t for t in tripped if t != "energy"]


def energy_guard_applies(d_bohr: np.ndarray) -> bool:
    """Whether a displacement from the guess's geometry is within the
    energy signal's reach (:data:`ENERGY_GUARD_REACH`, per component)."""
    return float(np.abs(d_bohr).max(initial=0.0)) <= ENERGY_GUARD_REACH * (1 + 1e-12)


@dataclass
class GradientAt:
    """One :func:`gradient_at` result: the gradient ``(natm, 3)`` Eh/bohr of
    ``energy`` (``E_CASCI + E_corr``, SC or PC) at ``geom``, the converged
    orbitals as the next call's guess, and the guard ledger."""

    geom: list
    grad: np.ndarray
    energy: float
    mo: MoGuess
    ledger: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# The one call
# --------------------------------------------------------------------------


def gradient_at(geom, case: str, *, pc: bool = False, reference: str = "casscf",
                mo_guess: MoGuess | Path | str | None = None,
                binary: Path | None = None, work_dir: Path | None = None,
                gpu_lock_path: str | None = None, workers: int = 1, label: str = "state",
                keep: bool = False, log=print) -> GradientAt:
    """The GPU-sourced NEVPT2 (SC, or PC with ``pc``) nuclear gradient of
    ``case``'s CASSCF-reference state at ``geom`` (Angstrom, ``[(symbol,
    (x, y, z))]`` in the reference's atom order). The module docstring has
    the four steps and every refusal.

    ``mo_guess`` is the previous call's :attr:`GradientAt.mo`, a ``.mogold``
    path, or ``None`` for the committed ``golden/<case>_casscf.mogold``.
    ``binary`` is the ``nevpt2_sc_pseudodensity`` (default: :func:`find_binary`),
    ``gpu_lock_path`` the card's lock file, ``workers`` the processes the
    ``kappa`` sweep runs in. Files go to ``work_dir`` (default: a fresh
    temporary directory) named by ``label``, and are deleted on return unless
    ``keep``."""
    t0 = time.perf_counter()
    if reference not in REFERENCES:
        raise SystemExit(f"gradient_at: reference={reference!r} is not one of {REFERENCES}: "
                         "the CASCI-on-RHF gradient is nevpt2_grad.py --reference casci's")
    spec = case_spec(case)
    geom = as_geometry(geom)
    ref_geom = reference_geometry(spec)
    what = f"gradient_at({case})"
    _check_atoms(ref_geom, geom, what)
    # The refusals that cost nothing, before anything is computed or written.
    ref_group = ngrad.point_group(ref_geom, spec["basis"])
    group = ngrad.point_group(geom, spec["basis"])
    ngrad._check_point_group(ref_group, group, what)
    own_dir = work_dir is None
    work = Path(tempfile.mkdtemp(prefix=f"gradient-at-{case}-")) if own_dir else Path(work_dir)
    check_work_dir(work)
    work.mkdir(parents=True, exist_ok=True)
    exe = find_binary(binary)
    written: list[Path] = []
    try:
        out = _gradient_at(geom, spec, pc=pc, mo_guess=mo_guess, exe=exe, work=work,
                           gpu_lock_path=gpu_lock_path, workers=workers, label=label,
                           written=written, group=group, log=log)
    finally:
        if not keep:
            for p in written:
                p.unlink(missing_ok=True)
            if own_dir:
                shutil.rmtree(work, ignore_errors=True)
    out.ledger["wall"] = time.perf_counter() - t0
    out.ledger["work_dir"] = str(work)
    out.ledger["kept"] = bool(keep)
    return out


def _gradient_at(geom: list, spec: dict, *, pc: bool, mo_guess, exe: Path, work: Path,
                 gpu_lock_path: str | None, workers: int, label: str, written: list,
                 group: tuple, log) -> GradientAt:
    """:func:`gradient_at`'s four steps, once the free refusals have run."""
    case = spec["case"]
    what = f"gradient_at({case})"
    walls: dict[str, float] = {}
    ledger: dict = dict(case=case, pc=pc, reference="casscf", label=label,
                        point_group=group, binary=str(exe), tiles=TILES, workers=workers)

    # -- 1. the CASSCF state, re-optimised from the guess -------------------
    t = time.perf_counter()
    if mo_guess is None:
        mo_guess = spec["mogold"]
    guess = mo_guess if isinstance(mo_guess, MoGuess) else MoGuess.from_mogold(mo_guess, spec)
    _check_atoms(guess.geom, geom, f"{what}: the guess")
    ledger["guess"] = guess.origin
    atom = ngrad.geom_spec(geom)
    start = ngrad.warm_start(atom, spec["basis"], guess.mo_coeff, guess.ci)
    o, mc, fin = ngrad.optimise_finished(spec, atom, label, start)
    rec = ngrad.optimised_record(o, mc, fin)
    d = coords_bohr(geom) - coords_bohr(guess.geom)
    enforced = energy_guard_applies(d)
    # A .mogold's geometry comes back through bohr, so "the same geometry" is
    # equal to round-off, not bit for bit: no slope is needed below 1e-10.
    moved = float(np.abs(d).max(initial=0.0)) > 1e-10
    e_pred = float(np.sum(guess.casscf_gradient(spec) * d)) if enforced and moved else 0.0
    sig = ngrad.continuity_signals(
        rec, dict(mo=guess.mo_coeff, ci_opt=guess.ci, e_casscf=guess.e_casscf), e_pred)
    if not enforced:
        sig["e_jump"] = float("nan")
    tripped = continuity_verdict(sig, energy_enforced=enforced)
    ledger["continuity"] = dict(
        signals=sig, tripped=tripped, energy_enforced=enforced,
        displacement_max_bohr=float(np.abs(d).max(initial=0.0)),
        text=ngrad.signal_text(sig))
    ledger["casscf"] = {k: rec[k] for k in (
        "e_casscf", "e_casscf_gen", "converged", "first_converged", "n_macro", "polish_macro",
        "finish_macro", "finish_converged", "g_gen", "g_opt", "act_irreps")}
    if tripped:
        raise SystemExit(f"{what}: the continuity guard refuses this geometry's CASSCF "
                         f"(tripped {', '.join(tripped)}) -- {ngrad.signal_text(sig)}"
                         + ("" if enforced else "; the energy signal is not evaluated "
                            f"beyond {ENERGY_GUARD_REACH:g} bohr of the guess"))
    labels = state_labels(geom, spec["basis"], rec["mo"], rec["ncore"], rec["ncas"])
    ref_labels = reference_labels(spec)
    ledger["labels"], ledger["ref_labels"] = labels, ref_labels
    ngrad._check_selection(ref_labels, labels, what)
    mogold = work / f"{label}.mogold"
    written.append(mogold)
    ngrad.write_state_mogold(mogold, o, mc)
    walls["casscf"] = time.perf_counter() - t
    log(f"[{case}] {label}: CASSCF {rec['e_casscf']:+.12f} from {guess.origin} "
        f"({walls['casscf']:.1f}s); {ngrad.signal_text(sig)}")

    # -- 2. the transient golden and its gradient sidecar -------------------
    t = time.perf_counter()
    golden = work / f"{label}.nevpt2gold"
    written += [golden, golden.with_suffix(".gradgold")]
    meta = write_transient(atom, spec, mogold, golden, name=label, pc=pc, gradgold=True)
    walls["golden"] = time.perf_counter() - t
    log(f"  transient golden {golden} (+ .gradgold"
        + (", block2's PC fields" if pc else "") + f", {walls['golden']:.1f}s)")

    # -- 3. the GPU's source terms ------------------------------------------
    t = time.perf_counter()
    gradsrc = work / f"{label}.gradsrc"
    gpu_log = work / f"{label}.dump.log"
    written += [gradsrc, gpu_log]
    cmd = [str(exe), "--golden", str(golden), "--tiles", str(TILES)] + (
        ["--pc"] if pc else []) + ["--dump-source-terms", str(gradsrc)]
    with gpu_lock(gpu_lock_path):
        # The wait for the card's lock is not the dump's cost: timed apart.
        walls["gpu_wait"] = time.perf_counter() - t
        t = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    gpu_log.write_text(f"$ {' '.join(cmd)}\n{p.stdout}{p.stderr}\nexit {p.returncode}\n")
    walls["gpu"] = time.perf_counter() - t
    ledger["gpu"] = dict(cmd=" ".join(cmd), exit=p.returncode, log=str(gpu_log))
    if p.returncode != 0 or not gradsrc.exists():
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-8:])
        raise SystemExit(f"{what}: {exe.name} --dump-source-terms exited {p.returncode}"
                         f" (log {gpu_log}):\n{tail}")
    log(f"  GPU source terms {gradsrc} ({walls['gpu']:.1f}s)")

    # -- 4. the state, the solve and the contraction -------------------------
    t = time.perf_counter()
    ngr.PC_GUARD.__dict__.update(ngr.PcGuardLedger().__dict__)
    with contextlib.redirect_stdout(io.StringIO()):
        st = ngr.build_state(case, spec, pc=pc, reference="casscf", mo_from=mogold, geom=geom)
    wiring = dict(d_e_casci=st.e_casci - meta["e_casci"],
                  d_e_corr=st.e_corr - meta["e_nevpt2_total"])
    if pc:
        wiring["d_e_corr_pc"] = st.e_corr_pc - ngrad.read_golden_pc_total(golden)
    ledger["wiring"] = wiring
    ledger["mo_gate"] = tuple(float(x) for x in st.mc.mo_gate)
    if max(abs(v) for v in wiring.values()) > ngrad.REF_STATE_TOL:
        raise SystemExit(f"{what}: the rebuilt state is not the transient golden's: "
                         + ", ".join(f"{k} {v:+.2e}" for k, v in wiring.items())
                         + f" (> {ngrad.REF_STATE_TOL:g})")
    src = ngr.load_source_terms(gradsrc, st, pc=pc)
    g = ngr.solve_gradient(st, src, pc=pc, workers=workers, log=lambda *a: None)
    walls["solve"] = time.perf_counter() - t
    led = ngr.PC_GUARD
    ledger["pc_guard"] = dict(
        applies=pc, gap=st.pc_gap, displaced=led.displaced, guarded=led.displaced_guarded,
        sampled=led.sampled, min_gap=led.min_gap, rank_changed=list(led.rank_changed))
    if pc and (led.displaced != led.displaced_guarded or led.rank_changed):
        raise SystemExit(f"{what}: the PC guard ledger fails: "
                         f"{led.displaced_guarded} of {led.displaced} displaced CI vectors "
                         f"guarded, {len(led.rank_changed)} rank change(s)")
    ledger["residuals"] = {k: v for k, v in g.residuals.items()
                           if isinstance(v, (int, float, bool, type(None)))}
    ledger["energies"] = dict(e_casscf=rec["e_casscf"], e_casci=st.e_casci,
                              e_corr_sc=st.e_corr, e_corr_pc=st.e_corr_pc)
    ledger["walls"] = walls
    energy = st.e_casci + (st.e_corr_pc if pc else st.e_corr)
    log(f"  gradient ({'PC' if pc else 'SC'}) solved in {walls['solve']:.1f}s: E = "
        f"{energy:+.12f}, max |g| = {np.abs(g.de).max():.6e}, sum over atoms "
        f"{np.abs(g.de.sum(axis=0)).max():.1e}")
    nxt = MoGuess(geom=geom, mo_coeff=rec["mo"], ci=rec["ci_opt"], e_casscf=rec["e_casscf"],
                  origin=f"gradient_at({case}) {label}")
    return GradientAt(geom=geom, grad=np.asarray(g.de), energy=float(energy), mo=nxt,
                      ledger=ledger)


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help=f"one of {', '.join(ngrad.CASES)}")
    ap.add_argument("--geometry", type=Path, default=None,
                    help="an .xyz in the case's atom order (default: the case's own)")
    ap.add_argument("--pc", action="store_true", help="the PC-NEVPT2 gradient")
    ap.add_argument("--mo-guess", type=Path, default=None,
                    help="a .mogold to start from (default: golden/<case>_casscf.mogold)")
    ap.add_argument("--binary", type=Path, default=None,
                    help="nevpt2_sc_pseudodensity (default: " + " or ".join(BINARIES) + ")")
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="where the transient files go (default: a temporary directory; "
                         "never golden/)")
    ap.add_argument("--keep", action="store_true", help="leave the transient files")
    ap.add_argument("--gpu-lock", default=None, help="flock this file around the dump")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes the kappa sweep runs in")
    ap.add_argument("--label", default="state", help="the transient files' stem")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the gradient, the energy and the ledger's numbers (.npz)")
    ap.add_argument("--grad-num", type=Path, default=None,
                    help="gate against this nevpt2_numgrad.py --reference casscf --npz "
                         "(the reference geometry only)")
    ap.add_argument("--compare", type=Path, default=None,
                    help="report the distance from a nevpt2_grad.py --out file's gradient(s)")
    ap.add_argument("--tol", type=float, default=ngr.GRAD_NUM_TOL,
                    help=f"the --grad-num gate (default {ngr.GRAD_NUM_TOL:g})")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    spec = case_spec(args.case)
    geom = ngrad.read_xyz(args.geometry) if args.geometry else reference_geometry(spec)
    at_ref = np.array_equal(coords_bohr(as_geometry(geom)),
                            coords_bohr(reference_geometry(spec)))
    if args.grad_num is not None and not at_ref:
        ap.error("--grad-num is a numerical gradient at the case's own geometry; "
                 "this run is at another")
    grad_num = (ngr.load_grad_num(args.grad_num, args.case, pc=args.pc, reference="casscf")
                if args.grad_num is not None else None)
    r = gradient_at(geom, args.case, pc=args.pc, mo_guess=args.mo_guess, binary=args.binary,
                    work_dir=args.work_dir, gpu_lock_path=args.gpu_lock,
                    workers=args.workers, label=args.label, keep=args.keep)
    led = r.ledger
    print(f"\n{args.case} {'PC' if args.pc else 'SC'}-NEVPT2 on a CASSCF reference, GPU-sourced"
          f" (point group {led['point_group'][0]}): E = {r.energy:.12f} Eh, wall "
          f"{led['wall']:.1f}s ({', '.join(f'{k} {v:.1f}s' for k, v in led['walls'].items())})")
    print(f"  continuity: {led['continuity']['text']} (energy signal "
          + ("enforced" if led["continuity"]["energy_enforced"] else "not evaluated") + ")")
    print("  wiring: " + ", ".join(f"{k} {v:+.1e}" for k, v in led["wiring"].items())
          + f"; --mo-from gate |g_orb| {led['mo_gate'][0]:.1e}")
    if args.pc:
        pg = led["pc_guard"]
        # A dump-sourced solve evaluates no PC energy at a displaced CI vector
        # (y is the dump's), so the ledger is usually empty past the state.
        print(f"  PC guard: state gap {pg['gap']:.1e}, {pg['guarded']} of {pg['displaced']} "
              f"displaced CI vectors guarded, {pg['sampled']} sampled"
              + (f", min gap {pg['min_gap']:.1e}" if np.isfinite(pg["min_gap"]) else "")
              + f", {len(pg['rank_changed'])} rank change(s)")
    for i, (sym, _xyz) in enumerate(r.geom):
        print(f"  {i} {sym:<2} " + " ".join(f"{v:+.10f}" for v in r.grad[i]))
    print(f"  sum over atoms, max per column {np.abs(r.grad.sum(axis=0)).max():.1e}")
    ok = True
    if args.compare is not None:
        with np.load(args.compare) as z:
            for key in ("grad", "grad_pyscf"):
                if key in z:
                    print(f"  max |g - {args.compare.name}[{key}]| = "
                          f"{np.abs(r.grad - np.asarray(z[key])).max():.3e}")
    if grad_num is not None:
        probed = grad_num["grad_num_probed"] > 0
        delta = float(np.abs(r.grad - grad_num["grad_num"])[probed].max())
        ok = delta <= args.tol
        print(f"{'PASS' if ok else 'FAIL'}: max |g - grad_num| over {int(probed.sum())} "
              f"probed components = {delta:.3e} (gate {args.tol:g}; "
              f"{grad_num['grad_num_origin']})")
    if args.out is not None:
        np.savez(args.out, grad=r.grad, energy=np.array(r.energy), case=np.array(args.case),
                 pc=np.array(args.pc), geom=coords_bohr(r.geom),
                 symbols=np.array([s for s, _ in r.geom]),
                 e_casscf=np.array(led["energies"]["e_casscf"]),
                 e_casci=np.array(led["energies"]["e_casci"]),
                 e_corr_sc=np.array(led["energies"]["e_corr_sc"]),
                 wall=np.array(led["wall"]))
        print(f"wrote {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
