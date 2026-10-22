#!/usr/bin/env python3
r"""The **seminumerical NEVPT2 nuclear Hessian** of a CASSCF reference:
central differences of the GPU-sourced analytic gradient
(:func:`gradient_at.gradient_at`, #212) over the Cartesian coordinates that
keep the point group, SC or (``--pc``) PC, through a Richardson step ladder.
It writes a ``.hessgold`` sidecar (``NEVPT2H1``, #217;
:data:`golden_format.HESSGOLD_ARRAYS`), the oracle the analytic Hessian is
gated against (docs/hessian-theory.md, section 8).

It is OFFLINE tooling like the rest of ``reference_data/``: no build and no
ctest runs it. It needs a built ``nevpt2_sc_pseudodensity`` and a card.

**What is differenced.** Column ``j`` of ``H[i, j] = d g_i / d R_j`` is the
ladder of central differences ``[g(R + h e_j) - g(R - h e_j)] / 2h`` at
``h = 2^-k`` bohr, ``k`` over :data:`LADDER`. The coordinate set is every
Cartesian coordinate each of whose displacements keeps the reference
geometry's detected point group (:func:`coordinate_set`): all ``3N`` on a C1
case, the in-plane ones on planar Cs acrolein. A coordinate outside it is
**not computed**: every element it would reach is NaN in the sidecar, never
zero. Asking for one by name (``--coords 0z``) is refused, as
``nevpt2_numgrad.py`` refuses it.

**The ladder** is judged per column by the rule every ladder here uses
(``nevpt2_grad.fourfold_runs``, ``casscf_hessian.judge_column``): the
max-norm change of the column between neighbouring rungs must fall fourfold
(within ``(LADDER_LOW, LADDER_HIGH)``) for ``LADDER_RUN`` consecutive
halvings, and the column is the Richardson extrapolant ``(4 fd(h) -
fd(2h)) / 3`` at the finest pair of the finest such run.

**Every point is one ``gradient_at`` call** from the committed
``golden/<case>_casscf.mogold`` orbitals, each re-optimised at its own
geometry, so the points are independent: they run in a process pool
(``--jobs``), each call's ``kappa`` sweep in ``--workers`` processes of its
own, the GPU dumps under the card's lock. **Whatever** ``gradient_at``
**refuses** -- a point group or selection change, the continuity guard, a
wiring mismatch, the PC guard ledger -- refuses the Hessian: every refused
point is named after the pool drains, and no sidecar is written.

**The cache.** Every finished point (and every refusal) is written to
``WORK/cache/<case>/<scheme>/<rev>/<label>.npz`` the moment it lands, keyed
by case, scheme, coordinate, step (the label: ``3y-9`` is atom 3, ``y``,
minus, ``h = 2^-9``) and the generator revision (:func:`generator_rev`: the
scripts that compute a gradient, the lock file and the device source). A
re-run, or a crash's re-run, computes only what is missing. Each record holds
the gradient, the energy, the geometry and the converged orbitals and CI
vector (the next caller's guess: the optimiser #218 and #221 reuse them) and
the point's guard ledger.

**The analysis** (:func:`analyse`): the unsymmetrised residual ``|H - H^T|``
over the differenced block, then the symmetrised Hessian; the translational
sum rules (each differenced column summed over atoms per axis, and each row
of the symmetrised block over the translations the set contains), both exact
at any geometry; the rotational rule ``H (Omega R) = Omega g`` over the
rotations the set contains, exact only at an exactly stationary state, so
reported, not gated; and the harmonic frequencies through
``pyscf.hessian.thermo.harmonic_analysis`` (:func:`frequencies`), projecting
the translations and rotations the coordinate set contains.

**The gates** (:data:`SYMMETRY_TOL`, :data:`SUM_RULE_TOL`) and the measured
numbers are in docs/testing.md, "The seminumerical nuclear Hessian", and
docs/reference-data.md, "The seminumerical nuclear Hessian".

Usage::

    uv run python reference_data/nevpt2_seminum_hessian.py --case hsoh_def2svp_cas66 \
        --work-dir /tmp/seminum --jobs 12 --out /tmp/hsoh.hessgold
    uv run python reference_data/nevpt2_seminum_hessian.py --case acrolein_def2svp_cas65 \
        --both --work-dir /tmp/seminum --jobs 12 --write   # golden/<case>_casscf[_pc].hessgold
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads.
import generate_golden  # noqa: F401

# isort: split
import argparse
import contextlib
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import golden_format as gf
import gradient_at as gat
import nevpt2_grad as ngr
import nevpt2_numgrad as ngrad
import numpy as np

REPO = Path(__file__).resolve().parent.parent

#: The step ladder, ``h = 2^-k`` bohr for ``k`` over ``[first, last]``. Its
#: coarse end is where ``gradient_at``'s continuity guard still passes a
#: point started from the reference's orbitals; its fine end is where the
#: gradients' own convergence floor, amplified by ``1 / 2h``, reaches the
#: truncation (docs/testing.md, "The seminumerical nuclear Hessian", has the
#: measured ladders).
LADDER = (7, 11)

#: The gate on the unsymmetrised residual, ``max |H - H^T|`` over the
#: differenced block, Eh/bohr^2 (docs/testing.md, "The seminumerical nuclear
#: Hessian": why this number).
SYMMETRY_TOL = 1e-5

#: The gate on the translational sum rules, Eh/bohr^2, the same reason.
SUM_RULE_TOL = 1e-5

#: The cards a dump may run on: the build each preset puts the binary in, and
#: the box's per-card lock (CLAUDE.md, "Tile counts are measured": a small
#: CAS takes ``gradient_at.TILES``).
CARDS = {
    "cuda": ("build/src/nevpt2_sc_pseudodensity", "/tmp/nevpt2-gpu-cuda.lock"),
    "hip": ("build-hip/src/nevpt2_sc_pseudodensity", "/tmp/nevpt2-gpu-hip.lock"),
}

#: What a differenced gradient is a function of, besides the geometry: the
#: scripts that compute it and the lock file that pins their libraries. With
#: the device source (:func:`generator_rev`) they key the cache.
#: ``golden_format.py`` is not among them: it only moves bytes, and a change
#: that moved them differently fails ``golden_layout_check.py`` on every
#: committed file before any gradient could read it.
REV_FILES = (
    "reference_data/generate_golden.py",
    "reference_data/nevpt2_numgrad.py",
    "reference_data/nevpt2_grad.py",
    "reference_data/gradient_at.py",
    "uv.lock",
)

#: The schemes, by the name the cache, the sidecar and the ``--pc`` flag use.
SCHEMES = ("sc", "pc")


# --------------------------------------------------------------------------
# The coordinate set, the ladder and the points
# --------------------------------------------------------------------------


def ladder_steps(exps: tuple[int, int] = LADDER) -> list[float]:
    """The ladder's steps, coarse first (``nevpt2_grad.ladder_steps``)."""
    if exps[1] - exps[0] < ngr.LADDER_RUN + 1:
        raise SystemExit(f"ladder 2^-{exps[0]} .. 2^-{exps[1]} has {exps[1] - exps[0] + 1} "
                         f"rungs: a run of {ngr.LADDER_RUN} fourfold halvings needs "
                         f"{ngr.LADDER_RUN + 2}")
    return ngr.ladder_steps(exps)


def coord_name(k: int, geom) -> str:
    """``3y (H)``: coordinate ``k = 3 atom + axis`` by atom index, axis and
    element."""
    return f"{k // 3}{ngrad.AXES[k % 3]} ({geom[k // 3][0]})"


def point_label(k: int, sign: int, exp: int) -> str:
    """A displaced point's label and cache key: ``3y-9`` is atom 3, ``y``,
    displaced by ``-2^-9`` bohr."""
    return f"{k // 3}{ngrad.AXES[k % 3]}{'+' if sign > 0 else '-'}{exp}"


#: The reference geometry's own point: the gradient the sidecar stores.
REF_LABEL = "ref"


def coordinate_set(spec: dict, steps: list[float],
                   want: list[int] | None = None) -> tuple[list[int], list[int]]:
    """``(differenced, not computed)`` coordinate indices. A coordinate is
    differenced when every displacement of the ladder, both signs, keeps the
    reference geometry's detected point group (``nevpt2_numgrad.point_group``,
    the test ``gradient_at`` refuses on). With ``want`` (named coordinates)
    nothing is skipped: one that changes the group is refused here, before
    any point is computed."""
    geom = gat.reference_geometry(spec)
    ref_group = ngrad.point_group(geom, spec["basis"])
    kept, skipped = [], []
    for k in (want if want is not None else range(3 * len(geom))):
        for h in steps:
            bad = None
            for s in (h, -h):
                g = ngrad.point_group(ngrad.displace(geom, k // 3, k % 3, s), spec["basis"])
                if g != ref_group:
                    bad = (s, g)
                    break
            if bad is not None:
                break
        if bad is None:
            kept.append(k)
        elif want is None:
            skipped.append(k)
        else:
            ngrad._check_point_group(ref_group, bad[1],
                                     f"{coord_name(k, geom)}, step {bad[0]:+g} bohr")
    if not kept:
        raise SystemExit(f"{spec['case']}: no Cartesian coordinate keeps the point group "
                         f"{ref_group[0]} over the whole ladder, so there is nothing to "
                         "difference")
    return kept, skipped


def parse_coords(text: str, natm: int) -> list[int] | None:
    """``--coords``: ``auto`` (None: every point-group-keeping coordinate),
    ``all``, or ``0x,3y`` (``nevpt2_numgrad._parse_components``'s syntax)."""
    comps = ngrad._parse_components(text, natm)
    return None if comps is None else sorted({3 * a + c for a, c in comps})


def points(spec: dict, kept: list[int], exps: tuple[int, int]) -> list[dict]:
    """Every point the Hessian needs: the reference geometry, then each
    differenced coordinate's ladder, both signs. Each a dict with ``label``,
    ``geom`` (Angstrom, ``gradient_at``'s form), ``k``, ``sign``, ``exp``."""
    geom = gat.reference_geometry(spec)
    out = [dict(label=REF_LABEL, geom=geom, k=-1, sign=0, exp=0)]
    for k in kept:
        for exp in range(exps[0], exps[1] + 1):
            for sign in (+1, -1):
                out.append(dict(label=point_label(k, sign, exp), k=k, sign=sign, exp=exp,
                                geom=ngrad.displace(geom, k // 3, k % 3, sign * 2.0 ** -exp)))
    return out


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------


def generator_rev() -> str:
    """The revision a gradient was computed at: a digest of the
    :data:`REV_FILES` as they are on disk (so an uncommitted edit is a new
    revision) and of the device source the dump binary is built from
    (``git rev-parse HEAD:src`` plus any uncommitted diff under ``src/``).
    Sixteen hex digits."""
    h = hashlib.sha256()
    for rel in REV_FILES:
        h.update(rel.encode())
        h.update((REPO / rel).read_bytes())
    for cmd in (["git", "rev-parse", "HEAD:src"], ["git", "diff", "HEAD", "--", "src"]):
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, check=False)
        if p.returncode != 0:
            raise SystemExit(f"generator_rev: {' '.join(cmd)} failed: {p.stderr.decode()}")
        h.update(p.stdout)
    return h.hexdigest()[:16]


def cache_dir(work: Path, case: str, scheme: str, rev: str) -> Path:
    """Where a case's points of one scheme at one revision are cached."""
    return Path(work) / "cache" / case / scheme / rev


def _jsonable(x):
    """The ledger's values as JSON: numpy scalars and arrays, tuples, Paths."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else repr(v)
    if x is None or isinstance(x, str):
        return x
    return str(x)


def save_point(path: Path, rec: dict) -> None:
    """One point's record, written whole or not at all (a temporary file
    renamed over the name), so a crash never leaves a half record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    arrays = {k: np.asarray(v) for k, v in rec.items() if k != "ledger"}
    arrays["ledger"] = np.array(json.dumps(_jsonable(rec.get("ledger", {}))))
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_point(path: Path, task: dict, case: str, scheme: str, rev: str) -> dict | None:
    """A cached record, or ``None`` if there is none. A record whose case,
    scheme, revision, label or geometry is not the task's is refused: it
    would be a gradient of another point."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        rec = {k: z[k] for k in z.files}
    rec["ledger"] = json.loads(str(rec["ledger"]))
    for key, want in (("case", case), ("scheme", scheme), ("rev", rev),
                      ("label", task["label"])):
        if str(rec[key]) != want:
            raise SystemExit(f"{path}: cached {key} {str(rec[key])!r}, expected {want!r}")
    if not np.array_equal(rec["geom"], gat.coords_bohr(task["geom"])):
        raise SystemExit(f"{path}: the cached geometry is not the task's")
    return rec


def point_ledger(ledger: dict) -> dict:
    """What the sidecar keeps of one call's guard ledger: every number a
    refusal read, and the walls."""
    keep = ("continuity", "casscf", "labels", "ref_labels", "wiring", "mo_gate", "pc_guard",
            "walls", "wall", "point_group", "binary", "workers", "guess")
    out = {k: ledger[k] for k in keep if k in ledger}
    if "continuity" in out:
        out["continuity"] = {k: v for k, v in out["continuity"].items() if k != "text"}
    return out


def run_point(task: dict) -> dict:
    """One point, in a pool worker: ``gradient_at.gradients_at`` at the
    task's geometry for every scheme it still lacks (``task["schemes"]``,
    each with its cache path in ``task["caches"]``), sharing the CASSCF, the
    transient golden, the state and the ``kappa`` sweep between them. Each
    record is saved to the cache before returning. A refusal is a record too
    (``refused``, for every scheme asked), so a re-run does not repeat it."""
    schemes = tuple(task["schemes"])
    paths = {s: Path(p) for s, p in zip(schemes, task["caches"], strict=True)}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    log_path = paths[schemes[0]].with_suffix(".log")
    work = (Path(task["work"]) / "transient"
            / f"{'-'.join(schemes)}-{task['label']}-{os.getpid()}")
    t0 = time.perf_counter()
    base = dict(case=task["case"], rev=task["rev"], label=task["label"],
                geom=gat.coords_bohr(task["geom"]), card=task["card"])
    with open(log_path, "w") as lf:
        def log(*a):
            print(*a, file=lf, flush=True)
        try:
            out = gat.gradients_at(task["geom"], task["case"], schemes=schemes,
                                   binary=Path(task["binary"]), work_dir=work,
                                   gpu_lock_path=task["lock"], workers=task["workers"],
                                   label=task["label"], log=log)
        except SystemExit as e:
            log(f"REFUSED: {e}")
            for s in schemes:
                save_point(paths[s], dict(base, scheme=s, refused=str(e),
                                          ledger=dict(wall=time.perf_counter() - t0)))
            return dict(label=task["label"], schemes=schemes, refused=str(e))
        finally:
            shutil.rmtree(work, ignore_errors=True)
    for s, r in out.items():
        save_point(paths[s], dict(base, scheme=s, grad=r.grad, energy=r.energy,
                                  mo_coeff=r.mo.mo_coeff, ci=r.mo.ci, e_casscf=r.mo.e_casscf,
                                  ledger=point_ledger(r.ledger)))
    return dict(label=task["label"], schemes=schemes, wall=time.perf_counter() - t0)


def run_points(tasks: list[dict], jobs: int, log=print) -> None:
    """Every task not yet cached, ``jobs`` at a time in fresh (spawned)
    processes, each record saved as it lands. A worker that dies is reported
    and the rest carry on; what it left uncached is computed on a re-run."""
    if not tasks:
        return
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(run_point, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            done += 1
            try:
                r = fut.result()
            except Exception as e:  # a crashed worker is a finding, not an abort
                log(f"  [{done}/{len(tasks)}] {'+'.join(t['schemes'])} {t['label']}: worker "
                    f"failed: {type(e).__name__}: {e}")
                continue
            what = (f"REFUSED: {r['refused'].splitlines()[0][:160]}" if "refused" in r
                    else f"{r['wall']:.0f}s")
            log(f"  [{done}/{len(tasks)}] {'+'.join(t['schemes'])} {t['label']} on "
                f"{t['card']}: {what} (elapsed {time.perf_counter() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------
# The ladder, the Hessian and its analysis
# --------------------------------------------------------------------------


def judge_column(fd: np.ndarray) -> dict:
    """One column's ladder: ``fd[i]`` is the quotient vector at rung ``i``.
    The run rule on the max-norm change between rungs (``nevpt2_grad.
    fourfold_runs``, ``casscf_hessian.judge_column``'s), the Richardson
    extrapolant at the finest pair of the finest qualifying run, and per
    element its move from the coarser pair's extrapolant. With no qualifying
    run the longest run is used (finest pair if none), and ``ok`` is False."""
    dif = [float(np.abs(fd[i] - fd[i + 1]).max()) for i in range(len(fd) - 1)]
    runs = ngr.fourfold_runs(dif)
    good = [r for r in runs if r[0] >= ngr.LADDER_RUN]
    run, end = good[-1] if good else max(runs, default=(0, len(fd) - 2))
    value = (4.0 * fd[end + 1] - fd[end]) / 3.0
    prev = (4.0 * fd[end] - fd[end - 1]) / 3.0 if end >= 1 else np.full_like(value, np.nan)
    return dict(ok=run >= ngr.LADDER_RUN, run=run, end=end, value=value,
                residual=np.abs(value - prev), dif=dif)


def assemble(records: dict, kept: list[int], natm: int, exps: tuple[int, int]) -> dict:
    """The columns from the cached points: ``fd`` ``(nsteps, ncoord,
    ncoord)``, each column judged (:func:`judge_column`). Columns not
    differenced are NaN throughout."""
    steps = ladder_steps(exps)
    n = 3 * natm
    fd = np.full((len(steps), n, n), np.nan)
    raw = np.full((n, n), np.nan)
    resid = np.full((n, n), np.nan)
    run = np.full(n, np.nan)
    end = np.full(n, np.nan)
    judged = {}
    for k in kept:
        for i, (exp, h) in enumerate(zip(range(exps[0], exps[1] + 1), steps, strict=True)):
            gp = records[point_label(k, +1, exp)]["grad"].ravel()
            gm = records[point_label(k, -1, exp)]["grad"].ravel()
            fd[i, :, k] = (gp - gm) / (2.0 * h)
        j = judge_column(fd[:, :, k])
        judged[k] = j
        raw[:, k], resid[:, k] = j["value"], j["residual"]
        run[k], end[k] = j["run"], j["end"] + 1
    return dict(fd=fd, raw=raw, residual=resid, run=run, end=end, judged=judged, steps=steps)


def translation_vectors(natm: int) -> np.ndarray:
    """The three unit-mass Cartesian translations, ``(3, 3 natm)``."""
    return np.stack([np.tile(np.eye(3)[a], natm) for a in range(3)])


def rotation_vectors(geom_bohr: np.ndarray) -> np.ndarray:
    """The three infinitesimal rotations about the centroid, ``(3, 3 natm)``:
    ``(omega_a x (R_A - c))``."""
    r = geom_bohr - geom_bohr.mean(axis=0)
    return np.stack([np.cross(np.eye(3)[a], r).ravel() for a in range(3)])


def contained(vecs: np.ndarray, kept_mask: np.ndarray, tol: float = 1e-8) -> list[int]:
    """The rows of ``vecs`` lying within the coordinate set (no weight outside
    it beyond ``tol`` of their norm)."""
    out = []
    for i, v in enumerate(vecs):
        nv = np.linalg.norm(v)
        if nv > 0 and np.linalg.norm(v[~kept_mask]) <= tol * nv:
            out.append(i)
    return out


def analyse(raw: np.ndarray, kept: list[int], grad: np.ndarray, geom_bohr: np.ndarray) -> dict:
    """The symmetrisation and the sum rules, on the unsymmetrised columns
    ``raw`` (NaN outside the differenced columns). Returns the symmetrised
    Hessian and asymmetry (NaN outside the differenced block) and the rules'
    maxima."""
    n = raw.shape[0]
    natm = n // 3
    m = np.zeros(n, dtype=bool)
    m[kept] = True
    block = np.outer(m, m)
    asym = np.full((n, n), np.nan)
    hs = np.full((n, n), np.nan)
    sub = raw[np.ix_(m, m)]
    asym[block] = np.abs(sub - sub.T).ravel()
    hs[block] = (0.5 * (sub + sub.T)).ravel()
    cols = raw[:, m]
    trans = translation_vectors(natm)
    # The column rule: sum over atoms of each differenced column, per axis --
    # the gradient's own translational invariance, differenced.
    col_rule = float(np.abs(trans @ cols).max())
    # The row rule: each row of the symmetrised block against the
    # translations the set contains.
    t_in = contained(trans, m)
    row_rule = (float(np.abs(hs[np.ix_(m, m)] @ trans[t_in][:, m].T).max())
                if t_in else float("nan"))
    rots = rotation_vectors(geom_bohr)
    r_in = contained(rots, m)
    g = np.asarray(grad, dtype=float).reshape(natm, 3)
    rot_rule = float("nan")
    if r_in:
        res = []
        for a in r_in:
            omega_g = np.cross(np.eye(3)[a], g).ravel()
            res.append(np.abs(cols @ rots[a][m] - omega_g).max())
        rot_rule = float(max(res))
    # What symmetry makes exactly zero: the differenced columns' rows outside
    # the set (an out-of-plane gradient component, at any planar geometry).
    off = float(np.abs(raw[np.ix_(~m, m)]).max()) if (~m).any() else float("nan")
    return dict(hessian=hs, asymmetry=asym, max_asym=float(np.nanmax(asym)),
                trans_columns=col_rule, trans_rows=row_rule,
                translations_in_set=[ngrad.AXES[a] for a in t_in],
                rotations_in_set=[ngrad.AXES[a] for a in r_in], rot_rule=rot_rule,
                zero_by_symmetry=off)


def _mol(symbols, geom_bohr: np.ndarray):
    """A bare molecule for its masses and coordinates (no basis is used)."""
    from pyscf import gto

    return gto.M(atom=[(s, tuple(x)) for s, x in zip(symbols, geom_bohr, strict=True)],
                 basis="sto-3g", unit="Bohr", verbose=0)


def frequencies(symbols, geom_bohr: np.ndarray, hs: np.ndarray, kept: list[int]) -> dict:
    """Harmonic frequencies (cm^-1, imaginary as negative) of the symmetrised
    Hessian over the coordinate set.

    On the whole ``3N`` set this is ``pyscf.hessian.thermo.harmonic_analysis``
    itself, translations and rotations projected. On a subset it is the same
    analysis on the subset: the same masses, the same translation and
    rotation vectors (``thermo._get_TR``), of which those lying within the set
    are projected (a vector straddling it is refused: the set is then not a
    symmetry block), the same ``LINDEP_THRESHOLD`` and the same unit
    conversion. On a planar molecule's in-plane set that is the A' block:
    two translations and one rotation projected."""
    from pyscf.data import nist
    from pyscf.hessian import thermo

    mol = _mol(symbols, geom_bohr)
    natm = len(symbols)
    n = 3 * natm
    m = np.zeros(n, dtype=bool)
    m[kept] = True
    if m.all():
        res = thermo.harmonic_analysis(mol, hs.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3),
                                       imaginary_freq=False)
        return dict(freq_cm=np.asarray(res["freq_wavenumber"], dtype=float), projected=6,
                    via="harmonic_analysis")
    mass = mol.atom_mass_list(isotope_avg=True)
    coords = mol.atom_coords()
    coords = coords - np.einsum("z,zx->x", mass, coords) / mass.sum()
    tr = np.array(thermo._get_TR(mass, coords))
    rotor = thermo._get_rotor_type(thermo.rotation_const(mass, coords))
    tr = {"ATOM": tr[:3], "LINEAR": tr[:5]}.get(rotor, tr)
    inside, straddle = [], []
    for v in tr:
        out = np.linalg.norm(v[~m]) / np.linalg.norm(v)
        if out <= 1e-8:
            inside.append(v[m])
        elif out < 1.0 - 1e-8:
            straddle.append(v)
    if straddle:
        raise SystemExit(f"frequencies: {len(straddle)} translation/rotation vector(s) straddle "
                         "the coordinate set, so it is not a symmetry block and its "
                         "frequencies are not the molecule's")
    w = np.repeat(mass, 3)[m] ** -0.5
    h = hs[np.ix_(m, m)] * np.outer(w, w)
    if inside:
        q, _r = np.linalg.qr(np.array(inside).T)
        p = np.eye(m.sum()) - q @ q.T
        ev, vec = np.linalg.eigh(p)
        bvec = vec[:, ev > thermo.LINDEP_THRESHOLD]
        h = bvec.T @ h @ bvec
    force = np.linalg.eigvalsh(h)
    freq_au = np.sign(force) * np.sqrt(np.abs(force))
    au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2)) ** 0.5 / (2 * np.pi)
    return dict(freq_cm=freq_au * au2hz / nist.LIGHT_SPEED_SI * 1e-2, projected=len(inside),
                via="harmonic_analysis's projection on the coordinate set")


# --------------------------------------------------------------------------
# One scheme, end to end
# --------------------------------------------------------------------------


def ledger_summary(records: dict, refused: list[str]) -> dict:
    """The guard ledger the sidecar carries: per point, every number a
    refusal read (:func:`point_ledger`), and their maxima."""
    pts = {lab: dict(r["ledger"], card=str(r["card"])) for lab, r in sorted(records.items())}
    sig = [p["continuity"]["signals"] for p in pts.values() if "continuity" in p]

    def worst(key):
        vals = [float(s[key]) for s in sig if isinstance(s.get(key), (int, float))
                and np.isfinite(s[key])]
        return max(vals, default=float("nan"))

    wiring = [abs(float(v)) for p in pts.values() for v in p.get("wiring", {}).values()]
    return dict(
        points=pts, refused=refused,
        max_act_defect=worst("act_defect"), max_ci_defect=worst("ci_defect"),
        max_g_opt=worst("g_opt"), max_e_jump_enforced=worst("e_jump"),
        energy_enforced=sum(bool(p["continuity"]["energy_enforced"]) for p in pts.values()
                            if "continuity" in p),
        max_wiring=max(wiring, default=float("nan")),
    )


def build(case: str, scheme: str, records: dict, kept: list[int], exps: tuple[int, int],
          rev: str) -> tuple[dict, dict, dict, dict]:
    """``(header, fields, arrays, report)`` of one scheme's sidecar, from its
    cached points."""
    spec = gat.case_spec(case)
    geom = gat.reference_geometry(spec)
    natm = len(geom)
    ref = records[REF_LABEL]
    a = assemble(records, kept, natm, exps)
    an = analyse(a["raw"], kept, ref["grad"], gat.coords_bohr(geom))
    coords = np.zeros(3 * natm)
    coords[kept] = 1.0
    symbols = [s for s, _ in geom]
    ledger = ledger_summary(records, [])
    ledger["ladder"] = {coord_name(k, geom): dict(run=j["run"], ok=j["ok"], dif=j["dif"])
                        for k, j in a["judged"].items()}
    header = dict(natm=natm, nsteps=len(a["steps"]), ladder_run=ngr.LADDER_RUN,
                  energy=float(ref["energy"]))
    fields = dict(case=case, scheme=scheme, reference="casscf", symbols=" ".join(symbols),
                  rev=rev, ledger=json.dumps(_jsonable(ledger)))
    arrays = dict(geom=gat.coords_bohr(geom), gradient=np.asarray(ref["grad"]).ravel(),
                  coords=coords, steps=np.array(a["steps"]), hessian=an["hessian"],
                  hessian_raw=a["raw"], asymmetry=an["asymmetry"],
                  richardson_residual=a["residual"], fd=a["fd"], column_run=a["run"],
                  column_end=a["end"])
    return header, fields, arrays, dict(assembled=a, analysis=an, ledger=ledger,
                                        symbols=symbols, geom=geom)


def report(case: str, scheme: str, kept: list[int], skipped: list[int], rep: dict) -> bool:
    """Print one scheme's ladders, rules and frequencies; return PASS."""
    geom, a, an, led = rep["geom"], rep["assembled"], rep["analysis"], rep["ledger"]
    print(f"\n{case} {scheme.upper()}-NEVPT2 seminumerical Hessian: {len(kept)} coordinates "
          f"differenced, {len(skipped)} not computed"
          + (f" ({', '.join(coord_name(k, geom) for k in skipped)})" if skipped else ""))
    print(f"  ladder h = 2^-{round(-np.log2(a['steps'][0]))} .. 2^-"
          f"{round(-np.log2(a['steps'][-1]))} bohr; each column needs {ngr.LADDER_RUN} "
          f"consecutive fourfold halvings of its max-norm change")
    ok = True
    for k in kept:
        j = a["judged"][k]
        ratios = " ".join(f"{j['dif'][i - 1] / j['dif'][i]:6.2f}" if j["dif"][i] > 0 else "   -  "
                          for i in range(1, len(j["dif"])))
        print(f"    {coord_name(k, geom):<9} run {j['run']} "
              f"{'ok ' if j['ok'] else 'NO '} changes "
              + " ".join(f"{d:8.2e}" for d in j["dif"]) + f"  ratios {ratios}"
              f"  max Richardson residual {np.nanmax(j['residual']):.1e}")
        ok &= j["ok"]
    sym_ok = an["max_asym"] <= SYMMETRY_TOL
    rule = max(an["trans_columns"], an["trans_rows"] if np.isfinite(an["trans_rows"]) else 0.0)
    rule_ok = rule <= SUM_RULE_TOL
    print(f"  max |H - H^T| over the differenced block = {an['max_asym']:.2e} "
          f"(gate {SYMMETRY_TOL:g}): {'ok' if sym_ok else 'OVER'}")
    print(f"  translational sum rule: columns {an['trans_columns']:.2e}, rows over "
          f"{','.join(an['translations_in_set']) or 'none'} {an['trans_rows']:.2e} "
          f"(gate {SUM_RULE_TOL:g}): {'ok' if rule_ok else 'OVER'}")
    print(f"  rotational rule H (Omega R) = Omega g over "
          f"{','.join(an['rotations_in_set']) or 'none'}: {an['rot_rule']:.2e} (reported: exact "
          "only at an exactly stationary state)")
    if np.isfinite(an["zero_by_symmetry"]):
        print(f"  rows outside the set (zero by symmetry): max {an['zero_by_symmetry']:.1e}")
    print(f"  guard ledger: max 1-sv_act {led['max_act_defect']:.1e}, 1-|<ci0|ci>| "
          f"{led['max_ci_defect']:.1e}, |g_orb| {led['max_g_opt']:.1e}, energy signal "
          f"enforced at {led['energy_enforced']} point(s), max wiring {led['max_wiring']:.1e}")
    with contextlib.suppress(SystemExit):
        fr = frequencies(rep["symbols"], gat.coords_bohr(geom), an["hessian"], kept)
        print(f"  harmonic frequencies, cm^-1 ({fr['via']}; {fr['projected']} "
              "translations/rotations projected):")
        print("    " + " ".join(f"{f:.1f}" for f in fr["freq_cm"]))
    passed = ok and sym_ok and rule_ok
    print(f"{'PASS' if passed else 'FAIL'}: {case} {scheme.upper()} -- every column's ladder "
          f"{'converges' if ok else 'does NOT converge'}, |H - H^T| {an['max_asym']:.1e}, "
          f"sum rule {rule:.1e}")
    return passed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help=f"one of {', '.join(ngrad.CASES)}")
    sch = ap.add_mutually_exclusive_group()
    sch.add_argument("--pc", action="store_true", help="the PC-NEVPT2 Hessian")
    sch.add_argument("--both", action="store_true", help="SC and PC, through one pool")
    ap.add_argument("--work-dir", type=Path, required=True,
                    help="the cache and the transient files (never golden/)")
    ap.add_argument("--coords", default="auto",
                    help="auto (every point-group-keeping coordinate), all, or e.g. 0x,3y")
    ap.add_argument("--ladder", default=f"{LADDER[0]}:{LADDER[1]}",
                    help="first:last exponents, h = 2^-k bohr (default %(default)s)")
    ap.add_argument("--jobs", type=int, default=12, help="gradient_at calls at once")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes each call's kappa sweep runs in")
    ap.add_argument("--cards", default="cuda",
                    help=f"comma-separated, of {', '.join(CARDS)}: dumps go round-robin")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the sidecar here (one scheme; with --both, a directory)")
    ap.add_argument("--write", action="store_true",
                    help="write golden/<case>_casscf[_pc].hessgold, only on PASS")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the points and what the cache holds, compute nothing")
    args = ap.parse_args()
    if args.jobs < 1 or args.workers < 1:
        ap.error("--jobs and --workers must be >= 1")
    t0 = time.perf_counter()
    spec = gat.case_spec(args.case)
    gat.check_work_dir(args.work_dir)
    schemes = list(SCHEMES) if args.both else ["pc" if args.pc else "sc"]
    exps = tuple(int(x) for x in args.ladder.split(":"))
    steps = ladder_steps(exps)
    geom = gat.reference_geometry(spec)
    kept, skipped = coordinate_set(spec, steps, parse_coords(args.coords, len(geom)))
    cards = []
    for name in args.cards.split(","):
        if name not in CARDS:
            ap.error(f"--cards: {name!r} is not one of {list(CARDS)}")
        binary = gat.find_binary(REPO / CARDS[name][0])
        cards.append((name, str(binary), CARDS[name][1]))
    rev = generator_rev()
    pts = points(spec, kept, exps)
    print(f"{args.case}: {len(kept)} coordinates differenced, {len(skipped)} not computed; "
          f"{len(pts)} points per scheme x {len(schemes)}; rev {rev}; --jobs {args.jobs} "
          f"--workers {args.workers}, cards {', '.join(c[0] for c in cards)}", flush=True)
    tasks, cached = [], 0
    for p in pts:
        lacking, caches = [], []
        for scheme in schemes:
            path = cache_dir(args.work_dir, args.case, scheme, rev) / f"{p['label']}.npz"
            if load_point(path, p, args.case, scheme, rev) is not None:
                cached += 1
                continue
            lacking.append(scheme)
            caches.append(str(path))
        if lacking:
            name, binary, lock = cards[len(tasks) % len(cards)]
            tasks.append(dict(case=args.case, schemes=lacking, caches=caches, rev=rev,
                              label=p["label"], geom=p["geom"], work=str(args.work_dir),
                              binary=binary, lock=lock, card=name, workers=args.workers))
    print(f"  cached: {cached} scheme-points, to compute: {len(tasks)} calls", flush=True)
    if args.dry_run:
        return
    run_points(tasks, args.jobs)
    passed = True
    for scheme in schemes:
        cdir = cache_dir(args.work_dir, args.case, scheme, rev)
        recs, refused, missing = {}, [], []
        for p in pts:
            rec = load_point(cdir / f"{p['label']}.npz", p, args.case, scheme, rev)
            if rec is None:
                missing.append(p["label"])
            elif "refused" in rec:
                refused.append(f"{p['label']}: {rec['refused']}")
            else:
                recs[p["label"]] = rec
        if missing:
            raise SystemExit(f"{args.case} {scheme}: {len(missing)} point(s) not computed "
                             f"(a worker failed; re-run to compute them): {', '.join(missing)}")
        if refused:
            raise SystemExit(f"{args.case} {scheme}: gradient_at refused {len(refused)} "
                             "point(s), so the Hessian is refused (no sidecar written):\n  "
                             + "\n  ".join(refused))
        header, fields, arrays, rep = build(args.case, scheme, recs, kept, exps, rev)
        ok = report(args.case, scheme, kept, skipped, rep)
        passed &= ok
        walls = [float(r["ledger"]["wall"]) for r in recs.values()]
        print(f"  {len(walls)} points: per-call wall median {np.median(walls):.0f}s, "
              f"max {max(walls):.0f}s, sum {sum(walls):.0f}s")
        out = None
        if args.out is not None:
            out = args.out / gf.hessgold_name(args.case, scheme) if args.both else args.out
        elif args.write:
            out = REPO / "golden" / gf.hessgold_name(args.case, scheme) if ok else None
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            gf.write_hessgold(out, header=header, fields=fields, arrays=arrays)
            print(f"  wrote {out}")
    print(f"\nwhole run: {time.perf_counter() - t0:.0f}s wall")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
