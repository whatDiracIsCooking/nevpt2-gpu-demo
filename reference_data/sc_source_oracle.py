#!/usr/bin/env python3
r"""PySCF's **exact** SC-NEVPT2 source terms for one golden's state (with
``--pc``, block2's PC-NEVPT2 ones: "THE PC FLAVOUR" below), written as a
``golden/<case>.srcgold`` -- the committed oracle
``nevpt2_sc_pseudodensity --source-oracle`` holds the GPU's own source terms
to, block by block, on a card (#92; docs/testing.md, "The SC source terms
against PySCF's, on a card"). With ``--directional``, the same check along
seeded directions per block instead of element by element: "THE DIRECTIONAL
FLAVOUR" below (#105).

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
what the GPU's PC source terms (#74) are held to, on a card
(``nevpt2_sc_pseudodensity --pc --source-oracle``), and the comparator's host
self-test reads it too (docs/testing.md, "The PC source terms' oracle")::

    uv run python reference_data/sc_source_oracle.py --case n2_ccpvdz_cas44 --pc

THE DIRECTIONAL FLAVOUR (#105). The full flavour differences EVERY element
of ``A`` -- one rotation pair at a time, two energies each -- and every
in-sector determinant of ``P y``. That is seconds on N2 CAS(4,4) and LiF, and
hours on acrolein CAS(6,5): 2,846 pairs, two block2 PC solves each. But
``A`` and ``P y`` enter the gradient linearly, so a wrong or missing term in
a block shows up along a seeded random direction in that block with
probability 1, and the block resolution is what the known-gap list keys on
anyway. ``--directional`` writes ``golden/<case>[_pc]_dir.srcgold`` (magic
``NEVPT2SD`` / ``NEVPT2PD``, the same scalar header) holding, per non-empty
block of the eight (core-core, core-act_O, ..., virt-virt, in
``nevpt2_grad.SPACE_NAMES`` order) :data:`KAPPA_DIRECTIONS` seeded unit
directions and the oracle's derivative along each, and the same for the CI
half:

``kappa_dirs`` ``(n_k, n_mo, n_mo)``
    each an antisymmetric ``kappa`` of unit norm over the pairs ``p < q`` of
    ONE block, uniform random weights, in the oracle's (section 2.6) basis,
    like ``a_corr``; stored explicitly, so no RNG has to match across
    languages;
``kappa_block`` ``(n_k,)``
    which of the eight blocks each one lies in (0..7, as a float);
``kappa_fd`` ``(n_k, n_rungs)``
    the central differences of ``E_corr(C exp(h kappa))`` at fixed CI vector
    (``Functional.evaluate``, the route ``kappa_sweep`` takes), at ``h =
    kappa_step * 2^-i``: the ladder each value is read off, kept as its
    evidence;
``kappa_val`` ``(n_k,)``
    the value: ``<kappa, A> = sum_{p<q} kappa_pq A_pq``, as the Richardson
    extrapolant of the first two rungs, ``(4 fd(h/2) - fd(h)) / 3``;
``ci_dirs`` ``(n_c, n_det)``
    :data:`CI_DIRECTIONS` seeded unit CI directions in the GOLDEN's
    determinant basis -- ``nevpt2_grad.ci_direction``'s: orthogonal to the
    state, in its irrep, and a spin eigenfunction (a bare determinant is
    not one, and that is what forced ``CI_STEP_PC``'s 1e-7) -- carried
    there by the orthogonal ``T(W^T)``, the golden's CI sign applied;
``ci_fd`` / ``ci_val``
    the same ladder and value for ``E_corr`` along the unit sphere
    (``nevpt2_grad.on_sphere``): ``<u, P y>``;
``act_rot``, ``ci``
    as in the full flavour.

The steps are not the full flavour's. A direction is not a bare
determinant, so the PC CI quotient is not trapped at ``CI_STEP_PC``: it runs
on #104's PC CI ladder (``CI_LADDER_PC``), from ``2^-12``, one rung inside
its first, where each evaluation's round-off -- 100-400 ulp of
``E_corr`` on acrolein PC, docs/testing.md "PC's CI-direction ladders,
measured" -- is divided by a step ten thousand times larger than 1e-7, and
the Richardson pair removes the ``h^2`` truncation the coarser step leaves.
The committed steps and rung counts (:data:`DIRECTIONAL_STEPS`) and the
tolerance were chosen from long ladders of this same generator
(``--kappa-rungs`` / ``--ci-rungs``), measured and recorded in docs/testing.md, "The directional
source-term oracle"; every run is held to the 30-minute single-threaded
budget, and prints its own wall time::

    uv run python reference_data/sc_source_oracle.py --case acrolein_def2svp_cas65 --directional
    uv run python reference_data/sc_source_oracle.py --case acrolein_def2svp_cas65 \
        --directional --pc

``--against-dump PATH`` (a ``.gradsrc`` from ``nevpt2_sc_pseudodensity
--dump-source-terms``) prints the GPU's own projection beside every rung --
the ladder evidence the tolerance is read from. It writes nothing extra.

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
from scipy.linalg import expm

REPO = Path(__file__).resolve().parent.parent

#: The cases a ``.srcgold`` is committed for (#92): the two the oracle's
#: comparison is cheap at. N2 CAS(8,8)'s ``P y`` alone is 6481 s of quotients
#: (docs/testing.md, "The GPU as the source-term provider").
ORACLE_CASES = ("n2_ccpvdz_cas44", "lif_def2svp_cas46")

#: The cases a PC ``.srcgold`` is committed for (#95): N2 CAS(4,4) alone.
#: LiF has no PC answer to differentiate (its golden carries no PC fields),
#: and acrolein's full PC sweep is 21,136 s -- it has a directional oracle.
PC_ORACLE_CASES = ("n2_ccpvdz_cas44",)

#: The cases a DIRECTIONAL ``.srcgold`` is committed for (#105), per scheme:
#: acrolein CAS(6,5), the low-symmetry case no full oracle fits, and N2
#: CAS(4,4) and LiF beside their full oracles, as the cross-check of the two
#: flavours against each other.
DIRECTIONAL_CASES = {
    "sc": ("n2_ccpvdz_cas44", "lif_def2svp_cas46", "acrolein_def2svp_cas65"),
    "pc": ("n2_ccpvdz_cas44", "acrolein_def2svp_cas65"),
}

#: Seeded ``kappa`` directions per non-empty block, and seeded CI directions
#: (#105: "at least 2 directions per block", chosen by measurement --
#: docs/testing.md, "The directional source-term oracle").
KAPPA_DIRECTIONS = 2
CI_DIRECTIONS = 3

#: Each scheme's first ladder rung and rung count, ``(kappa_step,
#: kappa_rungs, ci_step, ci_rungs)``: the stored value is the Richardson
#: extrapolant of the first two rungs, and the rest are its evidence. Read off
#: long ladders of this generator (``--kappa-rungs`` / ``--ci-rungs``, against
#: GPU dumps), docs/testing.md, "The directional source-term oracle": SC's
#: ``2^-15`` is where N2's core-virt and CI near-poles have let go of the
#: ladder; PC's kappa ``2^-13`` is where acrolein's act_O-act_V has; PC's CI
#: ``2^-12`` is one rung inside #104's ``CI_LADDER_PC``, because at its first
#: rung, ``2^-11``, the PC guard refuses one of N2's seeded directions.
DIRECTIONAL_STEPS = {
    "sc": (2.0 ** -15, 4, 2.0 ** -15, 4),
    "pc": (2.0 ** -13, 4, 2.0 ** -12, 4),
}

#: The seeds: block ``b``'s ``j``-th direction is seeded ``SEED + 100 b + j``,
#: the ``j``-th CI one ``SEED + 1000 + j``. Only reproducibility rides on
#: them -- the directions themselves are stored.
SEED = 105

#: The eight non-gauge blocks, as pairs of ``nevpt2_grad.SPACE_NAMES``
#: indices, in the order the comparator's rows are (``source_compare.cpp``):
#: every ``s <= t`` but act_O-act_O and act_V-act_V, the gauges.
BLOCKS = tuple((s, t) for s in range(4) for t in range(s, 4) if not (s == t and s in (1, 2)))

#: How far an off-sector element of ``T(W^T)`` may be from zero for the irrep
#: sector to count as mapped onto a sector of the golden's basis. ``T`` is a
#: polynomial in ``W``'s elements, so a symmetry-preserving ``W`` gives exact
#: zeros up to round-off; a symmetry-mixing one gives O(1).
SECTOR_TOL = 1e-10


def block_name(b: int) -> str:
    s, t = BLOCKS[b]
    return f"{ng.SPACE_NAMES[s]}-{ng.SPACE_NAMES[t]}"


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


def rebuilt_state(case: str, *, pc: bool) -> tuple[ng.State, np.ndarray, float]:
    """``nevpt2_grad``'s rebuild of ``case``'s golden state -- held to the
    golden's ``E_CASCI`` / ``E_corr`` (and, with ``pc``, ``e_pc_total``) --
    with the golden's own CI vector and the sign the rebuild came out with
    relative to it (``y`` is odd in ``c``, so a CI source term carries it)."""
    spec = ng.CASES[case]
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
    ci_file = garr["ci"].ravel()
    ci_rebuilt = st.ci_golden.ravel()
    sign = 1.0 if float(ci_file @ ci_rebuilt) >= 0.0 else -1.0
    off = float(np.abs(sign * ci_rebuilt - ci_file).max())
    if off > ng.DUMP_CI_TOL:
        raise SystemExit(f"the rebuilt CI vector is {off:.2e} from {golden_path.name}'s")
    return st, ci_file.copy(), sign


def back_rotation(st: ng.State) -> np.ndarray:
    """``T(W^T)``, which carries a CI covector in this state's basis back to
    the golden's -- checked orthogonal, and checked to carry the state's CI
    vector back to the golden's."""
    w = np.asarray(st.act_rot)
    back = ci_rotation(st, w.T)
    fwd = ci_rotation(st, w)
    unitary = float(np.abs(back @ fwd - np.eye(st.ci.size)).max())
    if unitary > SECTOR_TOL:
        raise SystemExit(f"T(W^T) T(W) is {unitary:.2e} from the identity")
    if float(np.abs(back @ st.ci.ravel() - st.ci_golden.ravel()).max()) > SECTOR_TOL:
        raise SystemExit("T(W^T) does not carry the state's CI vector back to the golden's")
    return back


def check_pc_guard(ci_desc: str) -> None:
    """Every displaced CI vector went through the guard (Functional.pc_energy
    runs it there), and none of them changed a PC block's rank: a step that
    lifts a null mode past the cut differentiates a different branch, and its
    numbers are not to be written."""
    led = ng.PC_GUARD
    if led.displaced != led.displaced_guarded:
        raise SystemExit(f"{led.displaced} displaced CI vectors, only "
                         f"{led.displaced_guarded} guarded")
    if led.rank_changed:
        where, key, kept, ref = led.rank_changed[0]
        raise SystemExit(
            f"{len(led.rank_changed)} displaced CI vectors changed a PC block's rank "
            f"(first: {where}, block {key}, smallest kept {kept:.1e} against the "
            f"reference's {ref:.1e}) -- {ci_desc} is past the PC CI regime "
            "(docs/testing.md, \"The PC CI source term's step is a first-order trap\")")
    print(f"  PC guard: {led.displaced_guarded} displaced CI vectors guarded, none "
          f"changed a block's rank; smallest gap {led.min_gap:.1e}", flush=True)


def build(case: str, *, kappa_step: float, ci_step: float,
          pc: bool = False) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    st, ci_file, sign = rebuilt_state(case, pc=pc)

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
        check_pc_guard(f"--ci-step {ci_step:g}")

    # P y, carried back into the golden's determinant basis.
    w = np.asarray(st.act_rot)
    back = back_rotation(st)
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
                  ci_step=ci_step, scheme="pc" if pc else "sc", flavour="full")
    arrays = {"a_corr": a_corr, "act_rot": w, "py": py,
              "ci_mask": rows.astype(np.float64), "ci": ci_file}
    print(f"  {time.perf_counter() - t0:.0f}s in all", flush=True)
    return header, arrays


# --------------------------------------------------------------------------
# The directional flavour (#105)
# --------------------------------------------------------------------------


def block_pairs(st: ng.State, b: int) -> list[tuple[int, int]]:
    """The non-gauge pairs ``p < q`` of block ``b`` (``p``'s space is never
    after ``q``'s: the spaces are index ranges, in order)."""
    return [(p, q) for p, q in ng.kappa_pairs(st)
            if (st.space_of(p), st.space_of(q)) == BLOCKS[b]]


def kappa_direction(st: ng.State, b: int, seed: int) -> np.ndarray:
    """A seeded unit antisymmetric ``kappa`` over block ``b``'s pairs: uniform
    weights in ``[-1, 1)``, normalised over ``p < q``. Every pair of the
    block is in it, symmetry-breaking ones included -- the point is that a
    wrong element anywhere in the block moves the projection."""
    rng = np.random.default_rng(seed)
    pairs = block_pairs(st, b)
    w = rng.uniform(-1.0, 1.0, len(pairs))
    w /= float(np.linalg.norm(w))
    k = np.zeros((st.nmo, st.nmo))
    for (p, q), x in zip(pairs, w, strict=True):
        k[p, q], k[q, p] = x, -x
    return k


def ladder(f, h0: float, rungs: int) -> np.ndarray:
    """Central differences of ``f`` at ``h = h0 2^-i``, ``i < rungs``: powers
    of two below ``h0``, so the ratio across a halving is the Richardson
    factor exactly."""
    hs = [h0 * 2.0 ** -i for i in range(rungs)]
    return np.array([(f(+h) - f(-h)) / (2.0 * h) for h in hs])


def richardson(fd: np.ndarray) -> np.ndarray:
    """The Richardson extrapolant of each neighbouring pair of rungs, ``(4
    fd(h/2) - fd(h)) / 3``; the first is the value written."""
    return (4.0 * fd[1:] - fd[:-1]) / 3.0


def print_ladder(label: str, h0: float, fd: np.ndarray, ref: float | None) -> None:
    """One direction's ladder: each rung's quotient, the change from the rung
    above and its ratio (4 while ``h^2`` truncation dominates), the
    Richardson value of the pair ending there, its distance from the written
    value (the first) and, given ``ref`` (a GPU dump's projection), from
    that."""
    rich = richardson(fd)
    print(f"    {label}")
    print(f"      {'h':<7} {'fd':>21} {'|change|':>10} {'ratio':>7} {'Richardson':>21} "
          f"{'|R - R0|':>9}" + (f" {'|R - GPU|':>10} {'|fd - GPU|':>10}" if ref is not None
                                else ""))
    for i in range(len(fd)):
        h = h0 * 2.0 ** -i
        ch = f"{abs(fd[i] - fd[i - 1]):10.3e}" if i else f"{'-':>10}"
        ratio = (f"{abs(fd[i - 2] - fd[i - 1]) / abs(fd[i] - fd[i - 1]):7.2f}"
                 if i > 1 and fd[i] != fd[i - 1] else f"{'-':>7}")
        r = f"{rich[i - 1]:+21.13e} {abs(rich[i - 1] - rich[0]):9.2e}" if i else \
            f"{'-':>21} {'-':>9}"
        g = ""
        if ref is not None:
            g = (f" {abs(rich[i - 1] - ref):10.2e}" if i else f" {'-':>10}") + \
                f" {abs(fd[i] - ref):10.2e}"
        print(f"      2^-{round(-np.log2(h)):<4d} {fd[i]:+21.13e} {ch} {ratio} {r}{g}")


def build_directional(case: str, *, pc: bool, kappa_step: float, kappa_rungs: int,
                      ci_step: float, ci_rungs: int, n_kappa: int, n_ci: int,
                      against: Path | None = None) -> tuple[dict, dict]:
    """The directional flavour's header and arrays (the docstring's "THE
    DIRECTIONAL FLAVOUR"): ``n_kappa`` directions per non-empty block, each
    laddered over ``kappa_rungs`` rungs from ``kappa_step``, and ``n_ci`` CI
    directions over ``ci_rungs`` from ``ci_step``."""
    if min(kappa_rungs, ci_rungs) < 2:
        raise SystemExit("a Richardson value needs at least 2 rungs")
    t0 = time.perf_counter()
    ng.PC_GUARD.__dict__.update(ng.PcGuardLedger().__dict__)
    st, ci_file, sign = rebuilt_state(case, pc=pc)
    fn = ng.Functional(st, pc=pc)
    gpu = ng.load_source_terms(against, st, pc=pc) if against is not None else None
    if gpu is not None:
        print(f"  --against-dump {against}: the GPU's A and y, carried into this basis",
              flush=True)

    def e_corr(C: np.ndarray, ci: np.ndarray, where: str) -> float:
        return fn.evaluate(C, ci, with_corr=True, where=where)["e_corr"]

    # --- kappa: n_kappa directions per non-empty block ------------------------
    dirs, blocks, fds = [], [], []
    for b in range(len(BLOCKS)):
        npairs = len(block_pairs(st, b))
        if npairs == 0:
            print(f"  {block_name(b)}: no pair in this partition, no direction", flush=True)
            continue
        for j in range(n_kappa):
            t = time.perf_counter()
            k = kappa_direction(st, b, SEED + 100 * b + j)
            fd = ladder(lambda h, k=k: e_corr(st.mo_coeff @ expm(h * k), st.ci,
                                              "a kappa direction"), kappa_step, kappa_rungs)
            dirs.append(k)
            blocks.append(float(b))
            fds.append(fd)
            ref = (float(np.sum(np.triu(k, 1) * gpu.kappa)) if gpu is not None else None)
            print_ladder(f"{block_name(b)} direction {j} ({npairs} pairs), "
                         f"{time.perf_counter() - t:.0f}s", kappa_step, fd, ref)
            print(f"      elapsed {time.perf_counter() - t0:.0f}s", flush=True)

    # --- CI: n_ci spin-adapted directions in the state's irrep ----------------
    eris = fn.eris(st.mo_coeff)
    nmo = st.nmo
    h1_act = np.asarray(eris["h1eff"])[st.act, st.act]
    g_act = np.asarray(eris["ppaa"]).reshape(nmo, nmo, st.nact, st.nact)[st.act, st.act]
    allowed = ng.allowed_determinants(st)
    back = back_rotation(st)
    udirs, ufds = [], []
    for j in range(n_ci):
        t = time.perf_counter()
        u = ng.ci_direction(st, SEED + 1000 + j, pc=pc, h1_act=h1_act, g_act=g_act,
                            allowed=allowed)
        fd = ladder(lambda h, u=u: e_corr(st.mo_coeff, ng.on_sphere(st, u, h),
                                          f"a CI direction at h = {h:.3e}"), ci_step, ci_rungs)
        u_golden = sign * (back @ u)
        if abs(float(ci_file @ u_golden)) > SECTOR_TOL:
            raise SystemExit("a CI direction carried to the golden's basis is not orthogonal "
                             "to its CI vector")
        udirs.append(u_golden)
        ufds.append(fd)
        ref = float(u @ gpu.ci.ravel()) if gpu is not None else None
        print_ladder(f"P y direction {j}, {time.perf_counter() - t:.0f}s", ci_step, fd, ref)
        print(f"      elapsed {time.perf_counter() - t0:.0f}s", flush=True)
    fn.pt.ci = st.ci
    fn.mc.ci = st.ci
    if pc:
        check_pc_guard(f"a CI ladder from {ci_step:g}")

    kfd, cfd = np.array(fds), np.array(ufds)
    header = dict(nmo=st.nmo, ncore=st.ncore, nact=st.nact, nvirt=st.nvirt,
                  n_act_occ=st.n_act_occ, ndet=int(st.ci.size), kappa_step=kappa_step,
                  ci_step=ci_step, scheme="pc" if pc else "sc", flavour="directional")
    arrays = {
        "kappa_dirs": np.array(dirs), "kappa_block": np.array(blocks),
        "kappa_fd": kfd, "kappa_val": richardson(kfd.T)[0],
        "ci_dirs": np.array(udirs), "ci_fd": cfd, "ci_val": richardson(cfd.T)[0],
        "act_rot": np.asarray(st.act_rot), "ci": ci_file,
    }
    print(f"  {len(dirs)} kappa directions x {kappa_rungs} rungs, {n_ci} CI directions x "
          f"{ci_rungs} rungs; {fn.nevals} evaluations, {time.perf_counter() - t0:.0f}s in all",
          flush=True)
    return header, arrays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True,
                    choices=sorted(set(ORACLE_CASES) | set(DIRECTIONAL_CASES["sc"])))
    ap.add_argument("--pc", action="store_true",
                    help="block2's PC-NEVPT2 source terms instead of PySCF's SC ones "
                         f"(#95; cases: {', '.join(PC_ORACLE_CASES)}, directional: "
                         f"{', '.join(DIRECTIONAL_CASES['pc'])})")
    ap.add_argument("--directional", action="store_true",
                    help="the directional flavour (#105): seeded directions per block, "
                         "golden/<case>[_pc]_dir.srcgold")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write (default: golden/<case>[_pc][_dir].srcgold)")
    ap.add_argument("--kappa-step", type=float, default=None,
                    help=f"default nevpt2_grad.KAPPA_STEP = {ng.KAPPA_STEP:g}, with --pc too "
                         "(see the docstring); with --directional the first rung, default "
                         "DIRECTIONAL_STEPS'. Only here so the step ladders in "
                         "docs/testing.md can be rerun")
    ap.add_argument("--ci-step", type=float, default=None,
                    help=f"default nevpt2_grad.CI_STEP = {ng.CI_STEP:g} (CI_STEP_PC = "
                         f"{ng.CI_STEP_PC:g} with --pc); with --directional the first rung, "
                         "default DIRECTIONAL_STEPS'; likewise")
    ap.add_argument("--kappa-rungs", type=int, default=None,
                    help="--directional: rungs per kappa ladder (default DIRECTIONAL_STEPS'); "
                         "a long ladder is how the steps were measured")
    ap.add_argument("--ci-rungs", type=int, default=None,
                    help="--directional: rungs per CI ladder (default DIRECTIONAL_STEPS')")
    ap.add_argument("--kappa-dirs", type=int, default=KAPPA_DIRECTIONS,
                    help="--directional: directions per block (default %(default)s)")
    ap.add_argument("--ci-dirs", type=int, default=CI_DIRECTIONS,
                    help="--directional: CI directions (default %(default)s)")
    ap.add_argument("--against-dump", type=Path, default=None,
                    help="--directional: a .gradsrc whose GPU projections are printed beside "
                         "every rung (the tolerance's ladder evidence); changes nothing written")
    args = ap.parse_args()
    scheme = "pc" if args.pc else "sc"
    if args.directional:
        if args.case not in DIRECTIONAL_CASES[scheme]:
            ap.error(f"--directional{' --pc' if args.pc else ''}: no oracle for {args.case} "
                     f"(cases: {', '.join(DIRECTIONAL_CASES[scheme])})")
        ks, kr, cs, cr = DIRECTIONAL_STEPS[scheme]
        header, arrays = build_directional(
            args.case, pc=args.pc,
            kappa_step=args.kappa_step if args.kappa_step is not None else ks,
            kappa_rungs=args.kappa_rungs if args.kappa_rungs is not None else kr,
            ci_step=args.ci_step if args.ci_step is not None else cs,
            ci_rungs=args.ci_rungs if args.ci_rungs is not None else cr,
            n_kappa=args.kappa_dirs, n_ci=args.ci_dirs, against=args.against_dump)
    else:
        if args.case not in ORACLE_CASES:
            ap.error(f"no full oracle for {args.case} (cases: {', '.join(ORACLE_CASES)}); "
                     "pass --directional")
        if args.pc and args.case not in PC_ORACLE_CASES:
            ap.error(f"--pc: no PC oracle for {args.case} (cases: {', '.join(PC_ORACLE_CASES)})")
        if (args.kappa_rungs, args.ci_rungs, args.against_dump) != (None, None, None):
            ap.error("--kappa-rungs, --ci-rungs and --against-dump are --directional's")
        kappa_step = args.kappa_step if args.kappa_step is not None else ng.KAPPA_STEP
        ci_step = args.ci_step if args.ci_step is not None else (
            ng.CI_STEP_PC if args.pc else ng.CI_STEP)
        header, arrays = build(args.case, kappa_step=kappa_step, ci_step=ci_step, pc=args.pc)
    name = gf.srcgold_name(args.case, scheme, header["flavour"])
    out = args.out if args.out is not None else REPO / "golden" / name
    gf.write_srcgold(out, header=header, arrays=arrays)
    back_header, back_arrays = gf.read_srcgold(out)
    if back_header != header or any(
            not np.array_equal(back_arrays[k], v) for k, v in arrays.items()):
        raise SystemExit(f"{out} does not read back as written")
    print(f"wrote {out} ({out.stat().st_size} bytes), read back unchanged")


if __name__ == "__main__":
    main()
