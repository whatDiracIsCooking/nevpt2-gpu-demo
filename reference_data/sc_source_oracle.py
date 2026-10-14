#!/usr/bin/env python3
r"""PySCF's **exact** SC-NEVPT2 source terms for one golden's state, written
as a ``golden/<case>.srcgold`` -- the committed oracle
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


def build(case: str, *, kappa_step: float, ci_step: float) -> tuple[dict, dict]:
    spec = ng.CASES[case]
    t0 = time.perf_counter()
    print(f"[{case}] building the golden's own CASCI-on-RHF state", flush=True)
    st = ng.build_state(case, spec)
    golden_path = REPO / "golden" / f"{case}.nevpt2gold"
    gold, garr = gf.read_golden(golden_path, want={"ci"})
    d_casci, d_corr = st.e_casci - gold["e_casci"], st.e_corr - gold["e_nevpt2_total"]
    if max(abs(d_casci), abs(d_corr)) > ng.REF_STATE_TOL:
        raise SystemExit(f"the rebuilt state does not match {golden_path.name}: E_CASCI "
                         f"{d_casci:+.2e}, E_corr {d_corr:+.2e}")
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

    fn = ng.Functional(st)
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
                  ci_step=ci_step)
    arrays = {"a_corr": a_corr, "act_rot": w, "py": py,
              "ci_mask": rows.astype(np.float64), "ci": ci_file.copy()}
    print(f"  {time.perf_counter() - t0:.0f}s in all", flush=True)
    return header, arrays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, choices=ORACLE_CASES)
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write (default: golden/<case>.srcgold)")
    ap.add_argument("--kappa-step", type=float, default=ng.KAPPA_STEP,
                    help=f"default nevpt2_grad.KAPPA_STEP = {ng.KAPPA_STEP:g}; only here so "
                         "the step ladder in docs/testing.md can be rerun")
    ap.add_argument("--ci-step", type=float, default=ng.CI_STEP,
                    help=f"default nevpt2_grad.CI_STEP = {ng.CI_STEP:g}; likewise")
    args = ap.parse_args()
    header, arrays = build(args.case, kappa_step=args.kappa_step, ci_step=args.ci_step)
    out = args.out if args.out is not None else REPO / "golden" / f"{args.case}.srcgold"
    gf.write_srcgold(out, header=header, arrays=arrays)
    back_header, back_arrays = gf.read_srcgold(out)
    if back_header != header or any(
            not np.array_equal(back_arrays[k], v) for k, v in arrays.items()):
        raise SystemExit(f"{out} does not read back as written")
    print(f"wrote {out} ({out.stat().st_size} bytes), read back unchanged")


if __name__ == "__main__":
    main()
