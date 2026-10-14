#!/usr/bin/env python3
r"""PySCF's **exact** SC-NEVPT2 source terms for one golden's state (with
``--pc``, block2's PC-NEVPT2 ones: "THE PC FLAVOUR" below), written as a
``golden/<case>.srcgold`` -- the committed oracle
``nevpt2_sc_pseudodensity --source-oracle`` holds the GPU's own source terms
to, block by block, on a card (#92; docs/testing.md, "The SC source terms
against PySCF's, on a card").

Why it exists. Every on-card check of the device's source terms compares the
device with itself -- forward against reverse, at round-off -- so a term both
modes leave out cancels and no such check can see it. ``nevpt2_grad.py
--source both`` is the independent oracle that did (#73), but it is an
offline gate of minutes to hours. This file is that oracle's answer,
computed once and committed, so the comparison is a ``gpu`` ctest entry.

WHAT IS WRITTEN -- nothing new is computed here. Every number comes out of
``nevpt2_grad.py``'s own machinery, at its converged SC steps (#77):

``a_corr`` ``(n_mo, n_mo)``
    ``A_pq = dE_corr/dkappa_pq - dE_corr/dkappa_qp`` along every non-gauge
    rotation (``nevpt2_grad.kappa_pairs``: every ``p < q`` but those inside
    act_O or inside act_V; zero there), at fixed CI vector -- the
    ``A_corr`` of ``nevpt2_grad.kappa_sweep``, which is exactly what
    ``--source both`` compares a dump's ``Y - Y^T`` with. It is in
    ``nevpt2_grad``'s basis (docs/gradient-theory.md, section 2.6: the
    active block rotated into ``F``'s eigenbasis, so act_O and act_V are
    index ranges), and the gauge entries are not known, so it cannot be
    carried back to the golden's basis; the GPU's ``Y`` is carried to it
    instead, by
``act_rot`` ``(n_act, n_act)``
    ``W``, the active-block rotation that basis is: ``Y' = W^T Y W`` on the
    active block (``nevpt2_grad.load_source_terms``' own transform).
``py`` ``(n_det,)``
    ``P y = (1 - |c><c|) dE_corr/dc``, PySCF's projected quotient
    (``nevpt2_grad.pyscf_ci_source``) on the state's irrep, carried **back**
    into the golden's determinant basis -- the CI vector the GPU reads -- by
    the orthogonal ``T(W^T)``. Exact, because the irrep sector maps onto a
    sector of the golden's basis (checked here, and refused if it does not),
    so ``py`` is ``P y`` for the golden's own ``ci``, sign included.
``ci_mask`` ``(n_det,)``
    1.0 on that sector of the golden's basis, 0.0 elsewhere: the components
    the gradient reads, and the only ones compared.
``ci`` ``(n_det,)``
    the golden's CI vector, as a fingerprint: the reader refuses an oracle
    whose ``ci`` is not its golden's.

Deterministic: the state is ``generate_golden``'s (single-threaded, the
1e-12 Davidson tolerance, the detected point group), and nothing else here is
iterative, so a regenerated file is the committed one byte for byte
(docs/reference-data.md, "The SC source-term oracle")::

    uv run python reference_data/sc_source_oracle.py --case n2_ccpvdz_cas44
    uv run python reference_data/sc_source_oracle.py --case lif_def2svp_cas46
    # --out PATH writes elsewhere (the reproducibility check: cmp it)

THE PC FLAVOUR (#95). ``--pc`` writes the same file for block2's
**partially-contracted** energy instead -- ``golden/<case>_pc.srcgold``, magic
``NEVPT2P1``, the same header and arrays -- out of ``nevpt2_grad.py --pc``'s
own machinery: its rebuilt state with block2's PC energy (checked against the
golden's ``e_pc_total``), :class:`nevpt2_grad.Functional` in PC mode, its
``kappa_sweep`` and its ``pyscf_ci_source``, at two steps:

``KAPPA_STEP`` (``2^-17``, #77's converged step), **not** ``KAPPA_STEP_PC``
    ``KAPPA_STEP_PC`` (1e-4) is right for the *gradient*, which does not move
    with it at the gate's scale, but this file is read per element, and the
    PC step ladder puts 1e-4 1.5e-8 from converged in core-virt where
    ``2^-17`` sits at the ladder's ~1e-9 floor (docs/testing.md, "The PC
    source terms' oracle");
``CI_STEP_PC`` (1e-7)
    not a truncation choice: a per-determinant displacement is not a spin
    eigenfunction, and above ``CI_STEP_PC`` it lifts the PC metric's null
    modes past the lstsq cut (docs/testing.md, "The PC CI source term's step
    is a first-order trap"). Below it the quotient is at round-off already.

Every displaced CI vector is guarded -- the conditioning check and the rank
against the reference state's -- and a run that trips either is refused
rather than written. Only N2 CAS(4,4): LiF has no PC answer
(docs/testing.md, "LiF has no PC gradient, and that is the answer"). It is
what #74's GPU PC source terms will be held to; until then the comparator's
host self-test reads it (docs/testing.md, "The PC source terms' oracle")::

    uv run python reference_data/sc_source_oracle.py --case n2_ccpvdz_cas44 --pc

Offline Python, never run by a build or a test.
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads,
# which is what makes every energy here reproducible run to run
# (docs/reference-data.md, "Reproducibility").
import generate_golden  # noqa: F401  (imported for that side effect alone)

# isort: split
import argparse
import time
from pathlib import Path

import golden_format as gf
import nevpt2_grad as ng
import numpy as np
from pyscf import fci

REPO = Path(__file__).resolve().parent.parent

#: The cases a ``.srcgold`` is committed for (#92): the two the oracle's
#: comparison is cheap at. N2 CAS(8,8)'s ``P y`` alone is 6481 s of quotients
#: (docs/testing.md, "The GPU as the source-term provider").
ORACLE_CASES = ("n2_ccpvdz_cas44", "lif_def2svp_cas46")

#: The cases a PC ``.srcgold`` is committed for (#95): N2 CAS(4,4) alone.
#: LiF has no PC answer to differentiate (its golden carries no PC fields),
#: and acrolein is #74's manual gate (6.3 h).
PC_ORACLE_CASES = ("n2_ccpvdz_cas44",)

#: How far an off-sector element of ``T(W^T)`` may be from zero for the irrep
#: sector to count as mapped onto a sector of the golden's basis. ``T`` is a
#: polynomial in ``W``'s elements, so a symmetry-preserving ``W`` gives exact
#: zeros up to round-off; a symmetry-mixing one gives O(1).
SECTOR_TOL = 1e-10


def ci_rotation(st: ng.State, u: np.ndarray) -> np.ndarray:
    """``T(u)`` as an explicit ``(n_det, n_det)`` matrix: column ``I`` is
    determinant ``I`` carried to the basis ``|new> = |old> u``. Linear in the
    CI vector, so the columns are the matrix."""
    ndet = st.ci.size
    out = np.empty((ndet, ndet))
    for i in range(ndet):
        e = np.zeros(ndet)
        e[i] = 1.0
        out[:, i] = np.asarray(fci.addons.transform_ci_for_orbital_rotation(
            e.reshape(st.ci.shape), st.nact, st.nelecas, u)).ravel()
    return out


def build(case: str, *, kappa_step: float, ci_step: float,
          pc: bool = False) -> tuple[dict, dict]:
    spec = ng.CASES[case]
    t0 = time.perf_counter()
    print(f"[{case}] building the golden's own CASCI-on-RHF state"
          + (", with block2's PC energy" if pc else ""), flush=True)
    st = ng.build_state(case, spec, pc=pc)
    golden_path = REPO / "golden" / f"{case}.nevpt2gold"
    gold, garr = gf.read_golden(golden_path, want={"ci", "e_pc_total"} if pc else {"ci"})
    d_casci, d_corr = st.e_casci - gold["e_casci"], st.e_corr - gold["e_nevpt2_total"]
    if max(abs(d_casci), abs(d_corr)) > ng.REF_STATE_TOL:
        raise SystemExit(f"the rebuilt state does not match {golden_path.name}: E_CASCI "
                         f"{d_casci:+.2e}, E_corr {d_corr:+.2e}")
    if pc:
        # The PC energy differentiated must be the golden's own PC answer.
        if "e_pc_total" not in garr:
            raise SystemExit(f"{golden_path.name} carries no e_pc_total: no PC reference")
        d_pc = st.e_corr_pc - float(garr["e_pc_total"].ravel()[0])
        if abs(d_pc) > ng.REF_STATE_TOL:
            raise SystemExit(f"the rebuilt state's PC energy is {d_pc:+.2e} from "
                             f"{golden_path.name}'s e_pc_total")
        print(f"  E_corr(PC) = {st.e_corr_pc:.10f}, against the golden {d_pc:+.1e}; "
              f"smallest lstsq gap {st.pc_gap:.1e}", flush=True)
    print(f"  n_mo = {st.nmo} = {st.ncore} core + {st.nact} active ({st.n_act_occ} act_O) "
          f"+ {st.nvirt} virtual; ndet = {st.ci.size}; against the golden dE_CASCI = "
          f"{d_casci:+.1e}, dE_corr = {d_corr:+.1e}", flush=True)

    # The golden's CI vector, and the sign the rebuild came out with relative
    # to it (y is odd in c, so P y carries the sign).
    ci_file = garr["ci"].ravel()
    ci_rebuilt = st.ci_golden.ravel()
    sign = 1.0 if float(ci_file @ ci_rebuilt) >= 0.0 else -1.0
    off = float(np.abs(sign * ci_rebuilt - ci_file).max())
    if off > ng.DUMP_CI_TOL:
        raise SystemExit(f"the rebuilt CI vector is {off:.2e} from {golden_path.name}'s")

    fn = ng.Functional(st, pc=pc)
    # The order --source both runs them in: y first, then the sweep.
    t = time.perf_counter()
    y_state = ng.pyscf_ci_source(fn, st, step=ci_step).ravel()
    mask_state = ng.allowed_determinants(st)
    print(f"  P y over {int(mask_state.sum())} of {st.ci.size} determinants, step "
          f"{ci_step:g}, {time.perf_counter() - t:.0f}s", flush=True)
    sweep = ng.kappa_sweep(fn, st, ng.constraints(st), with_corr=True, step=kappa_step,
                           log=print)
    a_corr = np.asarray(sweep["A_corr"])
    print(f"  A over {len(ng.kappa_pairs(st))} non-gauge pairs, step {kappa_step:g}, "
          f"{sweep['seconds']:.0f}s; max |A| = {np.abs(a_corr).max():.6e}", flush=True)
    if pc:
        # Every displaced CI vector went through the guard (Functional.pc_energy
        # runs it there), and none of them changed a PC block's rank: a step
        # that lifts a null mode past the cut differentiates a different
        # branch, and its numbers are not to be written.
        led = ng.PC_GUARD
        if led.displaced != led.displaced_guarded:
            raise SystemExit(f"{led.displaced} displaced CI vectors, only "
                             f"{led.displaced_guarded} guarded")
        if led.rank_changed:
            where, key, kept, ref = led.rank_changed[0]
            raise SystemExit(
                f"{len(led.rank_changed)} displaced CI vectors changed a PC block's rank "
                f"(first: {where}, block {key}, smallest kept {kept:.1e} against the "
                f"reference's {ref:.1e}) -- --ci-step {ci_step:g} is past CI_STEP_PC's "
                "regime (docs/testing.md, \"The PC CI source term's step is a first-order "
                "trap\")")
        print(f"  PC guard: {led.displaced_guarded} displaced CI vectors guarded, none "
              f"changed a block's rank; smallest gap {led.min_gap:.1e}", flush=True)

    # P y, carried back into the golden's determinant basis.
    w = np.asarray(st.act_rot)
    back = ci_rotation(st, w.T)
    fwd = ci_rotation(st, w)
    unitary = float(np.abs(back @ fwd - np.eye(st.ci.size)).max())
    if unitary > SECTOR_TOL:
        raise SystemExit(f"T(W^T) T(W) is {unitary:.2e} from the identity")
    if float(np.abs(back @ st.ci.ravel() - ci_rebuilt).max()) > SECTOR_TOL:
        raise SystemExit("T(W^T) does not carry the state's CI vector back to the golden's")
    rows = np.abs(back[:, mask_state]).max(axis=1) > SECTOR_TOL
    leak = float(np.abs(back[np.ix_(rows, ~mask_state)]).max()) if (~mask_state).any() else 0.0
    if leak > SECTOR_TOL or int(rows.sum()) != int(mask_state.sum()):
        raise SystemExit(
            f"the state's irrep sector ({int(mask_state.sum())} determinants) does not map "
            f"onto a sector of the golden's basis ({int(rows.sum())} rows, leak {leak:.1e}): "
            "P y cannot be carried back exactly")
    py = sign * (back[:, mask_state] @ y_state[mask_state])
    py[~rows] = 0.0
    c_dot = float(ci_file @ py)
    print(f"  P y carried to the golden's basis: {int(rows.sum())} determinants, max |P y| = "
          f"{np.abs(py).max():.6e}, c.(P y) = {c_dot:+.1e}", flush=True)

    header = dict(nmo=st.nmo, ncore=st.ncore, nact=st.nact, nvirt=st.nvirt,
                  n_act_occ=st.n_act_occ, ndet=int(st.ci.size), kappa_step=kappa_step,
                  ci_step=ci_step, scheme="pc" if pc else "sc")
    arrays = {"a_corr": a_corr, "act_rot": w, "py": py,
              "ci_mask": rows.astype(np.float64), "ci": ci_file.copy()}
    print(f"  {time.perf_counter() - t0:.0f}s in all", flush=True)
    return header, arrays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, choices=ORACLE_CASES)
    ap.add_argument("--pc", action="store_true",
                    help="block2's PC-NEVPT2 source terms instead of PySCF's SC ones "
                         f"(#95; cases: {', '.join(PC_ORACLE_CASES)})")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write (default: golden/<case>.srcgold, or "
                         "golden/<case>_pc.srcgold with --pc)")
    ap.add_argument("--kappa-step", type=float, default=ng.KAPPA_STEP,
                    help=f"default nevpt2_grad.KAPPA_STEP = {ng.KAPPA_STEP:g}, with --pc too "
                         "(see the docstring); only here so the step ladders in "
                         "docs/testing.md can be rerun")
    ap.add_argument("--ci-step", type=float, default=None,
                    help=f"default nevpt2_grad.CI_STEP = {ng.CI_STEP:g} (CI_STEP_PC = "
                         f"{ng.CI_STEP_PC:g} with --pc); likewise")
    args = ap.parse_args()
    if args.pc and args.case not in PC_ORACLE_CASES:
        ap.error(f"--pc: no PC oracle for {args.case} (cases: {', '.join(PC_ORACLE_CASES)})")
    kappa_step = args.kappa_step
    ci_step = args.ci_step if args.ci_step is not None else (
        ng.CI_STEP_PC if args.pc else ng.CI_STEP)
    header, arrays = build(args.case, kappa_step=kappa_step, ci_step=ci_step, pc=args.pc)
    name = f"{args.case}_pc.srcgold" if args.pc else f"{args.case}.srcgold"
    out = args.out if args.out is not None else REPO / "golden" / name
    gf.write_srcgold(out, header=header, arrays=arrays)
    back_header, back_arrays = gf.read_srcgold(out)
    if back_header != header or any(
            not np.array_equal(back_arrays[k], v) for k, v in arrays.items()):
        raise SystemExit(f"{out} does not read back as written")
    print(f"wrote {out} ({out.stat().st_size} bytes), read back unchanged")


if __name__ == "__main__":
    main()
