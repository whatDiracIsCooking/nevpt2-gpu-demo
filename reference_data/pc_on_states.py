#!/usr/bin/env python3
r"""The GPU's PC-NEVPT2 energy at every displaced CASSCF state a
``nevpt2_numgrad.py --reference casscf --states-dir DIR`` sweep kept, written
to one keyed ``.npz`` for #126 to difference.

It is OFFLINE tooling like the rest of ``reference_data/``: no build and no
ctest runs it. It differences nothing and decides nothing; #126 does both on
the ``.npz`` it writes. It needs a built ``nevpt2_demo`` and a card.

**The states are the sweep's, not new ones.** The manifest and each state's
``.mogold`` are read through :func:`golden_format.read_states_manifest`, the
one reader of what the sweep hands on (docs/reference-data.md, "The numerical
gradient of a CASSCF reference", "One sweep, three consumers"). The finished
orbitals in each ``.mogold`` already carry #124's two-stage finish, so nothing
is optimised here.

Per kept state, three steps:

1. **A transient golden** (CPU, ``--workers`` processes). This is
   ``generate_golden.generate`` with ``pc=False``, the in-process
   ``generate_golden.py --atom SPEC ... --mo-from <label>.mogold --no-pc``:
   the golden's own ``--mo-from`` CASCI + SC-NEVPT2 on the sweep's orbitals,
   with every refusal of ``_build_state`` in force. block2 never runs.
   ``nevpt2_demo --pc`` refuses a golden without block2's two PC fields, so
   the transient file carries them as **NaN placeholders**
   (:func:`with_placeholder_pc`). A NaN can match no energy, so the demo's PC
   comparison fails by construction; its exit status is not read, and the
   parse below is what counts. The files go to ``--work-dir``, never to
   ``golden/`` (refused), and are deleted afterwards unless
   ``--keep-goldens``.
2. **The wiring check.** The transient golden's ``E_CASCI`` and ``E_corr``
   (PySCF's SC) must equal the sweep's own for the same state, from the
   manifest, within ``nevpt2_numgrad.REF_STATE_TOL``. It is the same code
   path on the same orbitals, so the difference is round-off.
3. **The GPU** (serial, under ``--gpu-lock`` if given). ``nevpt2_demo
   --golden <transient> --tiles 3`` must print ``PASS``: the device's SC
   against PySCF's on this state, a second wiring check. ``nevpt2_demo
   --golden <transient> --pc --tiles 3`` gives the PC energy, parsed from #134's
   ``E_corr full precision (PC-NEVPT2) = %.17g Ha`` line, and the per-class
   table's ``d``, ``dropped``, ``max dropped``, ``min kept`` and ``gap``
   columns.

**The PC conditioning guard, per state.** :func:`pc_guard` hands the demo's
per-class spectrum to ``generate_golden._check_pc_conditioning``, the guard
the golden's own PC fields pass and ``nevpt2_numgrad.py --pc`` applies, at
the same ``PC_MIN_GAP``. block2 does not run here, so the gaps it judges are
the device's: the eigenvalues of each class's metric ``S``, cut at
``PC_TAU * s_max``, rather than block2's per-tuple lstsq singular values. The
device refuses such a class on its own as well. A refused state is named and
recorded, and its ``e_pc`` is NaN. It is not dropped, so a metric mode moving
in or out between geometries shows up in the ``dropped`` and ``gap`` arrays.

**The reference geometry.** The state labelled ``ref`` is the sweep's
finished reference. Its GPU PC energy is held to the committed
``golden/<case>_casscf.nevpt2gold``'s ``e_pc_total`` within the demos' 1e-7
(:data:`REF_PC_TOL`), and the offset is printed. #124's finish moved the
sweep's reference ``E_corr`` by ~1e-8 from the committed one.

**The ``.npz``**, one row per manifest state (refused ones included), in
manifest order, keyed by ``labels``. ``e_pc`` is the GPU's PC correlation
energy. ``e_casci`` and ``e_corr_sc`` are the transient golden's (PySCF);
``e_sc_gpu`` is the device's SC. ``wiring_d_casci`` and ``wiring_d_corr`` are
the transient golden minus the manifest. ``guard_ok``, ``guard_min_gap`` and
``guard_why`` are the guard's verdict; ``built`` and ``build_why`` say whether
a transient golden was built. The per-class ``(n, 8)`` arrays ``pc_class``,
``d``, ``dropped``, ``max_dropped``, ``min_kept`` and ``gap`` are in
``classes`` order. ``s``, ``key``, ``rung``, ``sign`` and
``displacement_bohr`` come from the manifest, so the file differences on its
own. The scalars are ``case``, ``step``, ``tiles``, ``workers``,
``wall_build``, ``wall_gpu``, ``wall``, ``ref_e_pc_committed`` and
``ref_pc_offset``.

Usage (docs/reference-data.md, "The GPU's PC energies on the kept states")::

    uv run python reference_data/pc_on_states.py \
        --states-dir /tmp/acrolein_states --demo build/src/nevpt2_demo \
        --workers 8 --work-dir /tmp/pc_states --npz /tmp/acrolein_pc_states.npz
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads.
import generate_golden as gg

# isort: split
import argparse
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import golden_format as gf
import nevpt2_numgrad as ngrad
import numpy as np

# The work-dir guard, the transient golden and the card's lock are shared with
# gradient_at.py (#212), which defines them; with_placeholder_pc is re-exported.
from gradient_at import (  # noqa: F401
    check_work_dir,
    gpu_lock,
    with_placeholder_pc,
    write_transient,
)

REPO = Path(__file__).resolve().parent.parent

#: How far the reference geometry's GPU PC energy may sit from the committed
#: golden's ``e_pc_total``: the demos' own 1e-7, not a tolerance to tune.
REF_PC_TOL = 1e-7

#: How far a transient golden's ``E_CASCI`` / ``E_corr`` may sit from the
#: sweep's own for the same state. It is the same deterministic path on the same
#: orbitals, so this is nevpt2_numgrad's wiring tolerance.
WIRING_TOL = ngrad.REF_STATE_TOL

#: The tile count every run uses (CLAUDE.md, "Tile counts are measured, not
#: guessed": a small CAS takes the shipped 3).
TILES = 3

#: The full-precision lines #134 added (``%.17g`` round-trips a double).
PC_FULL = re.compile(r"^E_corr full precision \(PC-NEVPT2\) = (\S+) Ha$", re.M)
SC_FULL = re.compile(r"^E_corr full precision \(SC-NEVPT2\) = (\S+) Ha$", re.M)
#: The SC run's verdict line (``apps/integral_direct/main.cppm``).
SC_PASS = re.compile(r"^PASS: per-class norms/energies and the total match", re.M)


def build_transient(task: dict) -> dict:
    """One state's transient golden (step 1), returning the energies the
    wiring check reads. A top-level function of one picklable dict, so
    ``--workers`` can farm it out, with ``generate_golden``'s
    single-threading in every worker. It never raises a refusal: a
    ``SystemExit`` from a guard inside ``_build_state`` comes back as
    ``refused``."""
    t0 = time.perf_counter()
    out = dict(label=task["label"])
    try:
        meta = write_transient(task["atom"], task["case"], task["mogold"], Path(task["out"]),
                               name=task["label"], pc=False, placeholder_pc=True)
    except SystemExit as exc:
        out["refused"] = str(exc)
    else:
        out.update(e_casci=meta["e_casci"], e_corr=meta["e_nevpt2_total"],
                   golden=task["out"])
    out["wall"] = time.perf_counter() - t0
    return out


def build_all(tasks: list[dict], workers: int) -> list[dict]:
    """:func:`build_transient` over every task, ``workers`` spawned processes
    at a time, collected by label and not by completion order
    (``nevpt2_numgrad.run_points``'s idiom)."""
    import multiprocessing

    if workers <= 1 or len(tasks) <= 1:
        return [build_transient(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {t["label"]: pool.submit(build_transient, t) for t in tasks}
        assert len(futures) == len(tasks), "two tasks share a label"
        return [futures[t["label"]].result() for t in tasks]


def _num(tok: str) -> float:
    """A table cell: ``-`` (nothing dropped, or no denominator) is 0."""
    return 0.0 if tok == "-" else float(tok)


def parse_pc(text: str) -> dict:
    """``nevpt2_demo --pc``'s output as numbers: ``total`` (the full-precision
    line, or ``None`` when the demo did not sum all eight classes), and per
    class, in :data:`generate_golden.CLASSES` order, ``energy`` (``None`` when
    refused), ``status`` (``done`` / ``refused`` / ``not yet``), ``why`` and
    the spectrum columns ``d``, ``dropped``, ``max_dropped``, ``min_kept``,
    ``gap`` (``None`` when the device printed no spectrum). A table with a
    class missing or doubled is refused: it is not the output this was written
    against."""
    rows: dict[str, dict] = {}
    for line in text.splitlines():
        tok = line.split()
        if not tok or tok[0] not in gg.CLASSES or len(tok) < 8:
            continue
        if tok[0] in rows:
            raise SystemExit(f"nevpt2_demo --pc printed class {tok[0]} twice")
        row = dict(d=None, dropped=None, max_dropped=None, min_kept=None, gap=None,
                   energy=None, why="")
        if tok[1] != "-":
            row.update(d=int(tok[1]), dropped=int(tok[2]), max_dropped=_num(tok[3]),
                       min_kept=float(tok[4]), gap=float(tok[5]))
        if tok[7] == "REFUSED":
            row["status"] = "refused"
            m = re.search(r"\((.*)\)\s*$", line)
            row["why"] = m.group(1) if m else ""
        elif tok[7] == "not":
            row["status"] = "not yet"
        else:
            row["status"] = "done"
            row["energy"] = float(tok[7])
        rows[tok[0]] = row
    missing = [c for c in gg.CLASSES if c not in rows]
    if missing:
        raise SystemExit(f"nevpt2_demo --pc printed no row for {', '.join(missing)}")
    m = PC_FULL.search(text)
    return dict(total=float(m.group(1)) if m else None,
                classes=[rows[c] for c in gg.CLASSES])


def pc_guard(pc: dict) -> tuple[bool, float, str]:
    """The PC conditioning guard on one state: ``(ok, min gap, why)``. A
    class the device refused or did not compute fails it with the device's
    reason; otherwise every class's ``(name, d, gap, max dropped, min kept)``
    goes to ``generate_golden._check_pc_conditioning`` as is, and its refusal
    text is the reason."""
    gaps = [(c, r["d"], r["gap"], r["max_dropped"], r["min_kept"])
            for c, r in zip(gg.CLASSES, pc["classes"], strict=True)
            if r["gap"] is not None]
    min_gap = min((g[2] for g in gaps), default=float("nan"))
    bad = [f"{c}: {r['status']}" + (f" ({r['why']})" if r["why"] else "")
           for c, r in zip(gg.CLASSES, pc["classes"], strict=True)
           if r["status"] != "done"]
    if bad:
        return False, min_gap, "the device did not compute " + "; ".join(bad)
    try:
        gg._check_pc_conditioning(gaps)
    except SystemExit as exc:
        return False, min_gap, str(exc)
    return True, min_gap, ""


def run_demo(demo: Path, golden: Path, log: Path, *, pc: bool) -> str:
    """One ``nevpt2_demo`` run, its output kept in ``log``."""
    cmd = [str(demo), "--golden", str(golden), "--tiles", str(TILES)] + (["--pc"] if pc else [])
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    text = p.stdout + p.stderr
    log.write_text(f"$ {' '.join(cmd)}\n{text}\nexit {p.returncode}\n")
    return text


def run(args) -> int:
    t0 = time.perf_counter()
    man = gf.read_states_manifest(args.states_dir)
    name = man["case"]
    if name not in ngrad.CASES:
        raise SystemExit(f"{args.states_dir}: case {name!r} is not one nevpt2_numgrad knows")
    case = dict(ngrad.CASES[name], case=name)
    for k in ("basis", "ncas", "nelecas", "cas_irreps"):
        if man[k] != case[k]:
            raise SystemExit(f"{args.states_dir}: the manifest's {k} {man[k]!r} is not "
                             f"{name}'s {case[k]!r}")
    states = man["states"]
    if args.labels:
        want = set(args.labels.split(","))
        unknown = want - {st["label"] for st in states}
        if unknown:
            raise SystemExit(f"no state labelled {', '.join(sorted(unknown))}")
        states = [st for st in states if st["label"] in want]

    work = args.work_dir or Path(tempfile.mkdtemp(prefix=f"pc-states-{name}-"))
    check_work_dir(work)
    work.mkdir(parents=True, exist_ok=True)
    tasks = [dict(label=st["label"], atom=st["atom"], mogold=str(st["mogold_path"]),
                  out=str(work / f"{st['label']}.nevpt2gold"), case=case)
             for st in states if st["kept"]]
    print(f"{name}: {len(states)} state(s) in {args.states_dir}, {len(tasks)} kept; "
          f"transient goldens in {work}, --workers {args.workers}", flush=True)
    built = {b["label"]: b for b in build_all(tasks, args.workers)}
    wall_build = time.perf_counter() - t0
    print(f"transient goldens built in {wall_build:.1f}s", flush=True)

    n = len(states)
    nc = len(gg.CLASSES)
    rec: dict[str, np.ndarray] = dict(
        e_pc=np.full(n, np.nan), e_sc_gpu=np.full(n, np.nan),
        e_casci=np.full(n, np.nan), e_corr_sc=np.full(n, np.nan),
        wiring_d_casci=np.full(n, np.nan), wiring_d_corr=np.full(n, np.nan),
        guard_ok=np.zeros(n, bool), guard_min_gap=np.full(n, np.nan),
        built=np.zeros(n, bool),
        pc_class=np.full((n, nc), np.nan), d=np.full((n, nc), -1),
        dropped=np.full((n, nc), -1), max_dropped=np.full((n, nc), np.nan),
        min_kept=np.full((n, nc), np.nan), gap=np.full((n, nc), np.nan),
    )
    guard_why = [""] * n
    build_why = [""] * n
    problems: list[str] = []

    t1 = time.perf_counter()
    with gpu_lock(args.gpu_lock):
        for i, st in enumerate(states):
            label = st["label"]
            if not st["kept"]:
                build_why[i] = guard_why[i] = "not kept by the sweep: " + (
                    st.get("refused") or "tripped " + ", ".join(st.get("tripped", [])))
                continue
            b = built[label]
            if "refused" in b:
                build_why[i] = guard_why[i] = b["refused"]
                problems.append(f"{label}: the transient golden was refused: {b['refused']}")
                continue
            rec["built"][i] = True
            rec["e_casci"][i], rec["e_corr_sc"][i] = b["e_casci"], b["e_corr"]
            rec["wiring_d_casci"][i] = b["e_casci"] - st["e_casci"]
            rec["wiring_d_corr"][i] = b["e_corr"] - st["e_corr"]
            golden = Path(b["golden"])
            sc_text = run_demo(args.demo, golden, work / f"{label}.sc.log", pc=False)
            m = SC_FULL.search(sc_text)
            if m:
                rec["e_sc_gpu"][i] = float(m.group(1))
            if not (m and SC_PASS.search(sc_text)):
                problems.append(f"{label}: nevpt2_demo (SC) did not PASS against the "
                                f"transient golden; see {work / (label + '.sc.log')}")
            pc_text = run_demo(args.demo, golden, work / f"{label}.pc.log", pc=True)
            try:
                pc = parse_pc(pc_text)
            except SystemExit as exc:
                guard_why[i] = str(exc)
                problems.append(f"{label}: {exc}; see {work / (label + '.pc.log')}")
                continue
            for j, r in enumerate(pc["classes"]):
                if r["energy"] is not None:
                    rec["pc_class"][i, j] = r["energy"]
                if r["gap"] is not None:
                    rec["d"][i, j], rec["dropped"][i, j] = r["d"], r["dropped"]
                    rec["max_dropped"][i, j] = r["max_dropped"]
                    rec["min_kept"][i, j], rec["gap"][i, j] = r["min_kept"], r["gap"]
            ok, rec["guard_min_gap"][i], guard_why[i] = pc_guard(pc)
            rec["guard_ok"][i] = ok
            if ok:
                if pc["total"] is None:
                    problems.append(f"{label}: no full-precision PC line")
                else:
                    rec["e_pc"][i] = pc["total"]
    wall_gpu = time.perf_counter() - t1

    # The wiring check, over every state built.
    w = rec["built"]
    d_cas = np.abs(rec["wiring_d_casci"][w])
    d_cor = np.abs(rec["wiring_d_corr"][w])
    wmax = (float(d_cas.max()) if d_cas.size else float("nan"),
            float(d_cor.max()) if d_cor.size else float("nan"))
    if d_cas.size and max(wmax) > WIRING_TOL:
        problems.append(f"wiring: a transient golden is {wmax[0]:.1e} (E_CASCI) / "
                        f"{wmax[1]:.1e} (E_corr) from the sweep's own state, over "
                        f"{WIRING_TOL:g}")
    d_gpu = np.abs(rec["e_sc_gpu"][w] - rec["e_corr_sc"][w])
    print(f"wiring check over {int(w.sum())} state(s): max |transient - sweep| "
          f"E_CASCI {wmax[0]:.1e}, E_corr {wmax[1]:.1e} (<= {WIRING_TOL:g}); max "
          f"|GPU SC - PySCF SC| {np.nanmax(d_gpu) if d_gpu.size else float('nan'):.1e}")

    # The reference geometry against the committed golden.
    ref_pc = ngrad.read_golden_pc_total(REPO / "golden" / f"{name}_casscf.nevpt2gold")
    labels = [st["label"] for st in states]
    offset = float("nan")
    if "ref" in labels:
        i = labels.index("ref")
        offset = rec["e_pc"][i] - ref_pc if ref_pc is not None else float("nan")
        print(f"reference geometry: GPU E_PC = {rec['e_pc'][i]:.17g}, committed "
              f"e_pc_total = {ref_pc!r}, offset {offset:+.2e} (<= {REF_PC_TOL:g})")
        if not abs(offset) <= REF_PC_TOL:
            problems.append(f"reference geometry: the GPU PC energy is {offset:+.2e} "
                            f"from the committed e_pc_total, over {REF_PC_TOL:g}")

    refused = [(lb, why) for lb, ok, b, why in zip(labels, rec["guard_ok"], rec["built"],
                                                   guard_why, strict=True)
               if b and not ok]
    passing = rec["guard_min_gap"][rec["guard_ok"]]
    print(f"PC conditioning guard: {int(rec['guard_ok'].sum())} of {int(w.sum())} "
          f"state(s) pass; min gap over passing states "
          f"{passing.min() if passing.size else float('nan'):.1e}")
    for lb, why in refused:
        print(f"  refused {lb}: {why}")
    for i in np.flatnonzero(~rec["built"]):
        print(f"  not built {labels[i]}: {build_why[i]}")

    # A metric mode moving between geometries: say whether `dropped` is the
    # same in every state the guard passed.
    ok_rows = rec["dropped"][rec["guard_ok"]]
    if len(ok_rows):
        flips = [c for j, c in enumerate(gg.CLASSES) if len(set(ok_rows[:, j])) > 1]
        print("dropped modes per class: "
              + " ".join(f"{c}={sorted(set(ok_rows[:, j].tolist()))}"
                         for j, c in enumerate(gg.CLASSES))
              + (f"; varies in {', '.join(flips)}" if flips else "; the same in every state"))

    wall = time.perf_counter() - t0
    print(f"wall {wall:.1f}s (transient goldens {wall_build:.1f}s with --workers "
          f"{args.workers}, GPU {wall_gpu:.1f}s)")
    if args.npz is not None:
        keys = ("s", "key", "rung", "sign")
        np.savez(
            args.npz, labels=np.array(labels), classes=np.array(gg.CLASSES),
            guard_why=np.array(guard_why), build_why=np.array(build_why),
            displacement_bohr=np.array([st["displacement_bohr"] for st in states]),
            **{k: np.array([st[k] for st in states]) for k in keys},
            **rec, case=np.array(name), step=np.array(man["step"]),
            tiles=np.array(TILES), workers=np.array(args.workers),
            wall_build=np.array(wall_build), wall_gpu=np.array(wall_gpu),
            wall=np.array(wall),
            ref_e_pc_committed=np.array(np.nan if ref_pc is None else ref_pc),
            ref_pc_offset=np.array(offset),
        )
        print(f"wrote {args.npz}")
    if not args.keep_goldens:
        for t in tasks:
            Path(t["out"]).unlink(missing_ok=True)
        if args.work_dir is None and not any(work.glob("*.log")):
            shutil.rmtree(work, ignore_errors=True)
    if problems:
        print(f"FAIL: {len(problems)} problem(s)")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"PASS: {int(w.sum())} transient golden(s) wired to the sweep, the GPU's SC "
          f"matches PySCF's on each, {int(rec['guard_ok'].sum())} PC energies, "
          f"{len(refused)} refused by the PC conditioning guard")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states-dir", type=Path, required=True,
                    help="a nevpt2_numgrad.py --reference casscf --states-dir: "
                         "states.json and the kept states' .mogold files")
    ap.add_argument("--demo", type=Path, required=True,
                    help="the built nevpt2_demo (e.g. build/src/nevpt2_demo)")
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="where the transient goldens and the demo logs go "
                         "(default: a fresh temporary directory; never golden/)")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes building the transient goldens (CPU); the "
                         "GPU runs are serial")
    ap.add_argument("--npz", type=Path, default=None, help="the keyed output")
    ap.add_argument("--labels", default=None,
                    help="comma-separated labels: only these states")
    ap.add_argument("--gpu-lock", default=None,
                    help="flock this file around the GPU stage (the box's per-card lock)")
    ap.add_argument("--keep-goldens", action="store_true",
                    help="leave the transient goldens in --work-dir")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    if not args.demo.is_file():
        ap.error(f"--demo {args.demo}: no such file")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
