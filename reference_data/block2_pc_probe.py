#!/usr/bin/env python3
r"""Run block2's SC- and PC-NEVPT2 on exactly the state a golden file holds.

Kept because docs/pc-nevpt2.md, "The block2 reference", quotes its
output. It is OFFLINE, like ``generate_golden.py``: no
build or test runs it, and it writes no golden file.

The state is built by ``generate_golden._build_state`` itself (imported, not
copied), so it carries the same single-threading, 1e-12 Davidson tolerance
and point-group symmetry; the probe first checks that the rebuilt CI vector
and E_CASCI are the committed golden's. Then it runs, on that state:

* ``pyblock2.icmr.scnevpt2.WickSCNEVPT2`` -- must reproduce the golden's
  per-class SC energies (``class_energies``) to 1e-7, or nothing block2 says
  about PC on this state means anything;
* ``pyblock2.icmr.icnevpt2_full.WickICNEVPT2`` -- the partially-contracted
  energies, with every per-external-tuple linear solve intercepted to record
  the singular values ``np.linalg.lstsq(rcond=None)`` sees.

``--orbitals golden`` (default) hands block2 the golden's own canonical
orbitals (``cas_natorb=True``) and tells it they are canonical;
``--orbitals block2`` starts block2 from the raw CASCI orbitals and lets it
canonicalize them its own way (``cas_natorb=False``).

``--df`` builds the density-fitted state (``generate_golden.py --df``) and
feeds block2 PySCF's ``dfnevpt2._ERIS`` integrals, because block2's own
``eri_helper.init_eris`` always transforms the conventional 4-index AO ERIs.

Usage::

    uv run python reference_data/block2_pc_probe.py                        # CAS(10,10)
    uv run python reference_data/block2_pc_probe.py --ncas 12 --nelecas 12 --json out.json
    uv run python reference_data/block2_pc_probe.py --df
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads.
import generate_golden as gg

# isort: split
import argparse
import json
import resource
import struct
import time
from pathlib import Path

import numpy as np

#: block2's class keys (``ic.sub_eners``) -> generate_golden.CLASSES.
B2_TO_PYSCF = gg.B2_TO_CLASS

#: The order ``icnevpt2_full.kernel`` calls ``_linear_solve`` in: every
#: ``sub_spaces`` key except the ``...2`` halves, which are solved jointly
#: with their ``...1`` partner.
PC_SOLVE_ORDER = gg.PC_SOLVE_ORDER

#: Relative singular-value cutoffs the PC energy is re-evaluated at, to show
#: how much of each class rides on near-null directions.
RCONDS = (1e-12, 1e-10, 1e-8, 1e-6)


def read_golden(path: Path) -> dict:
    """The scalars and the arrays this probe needs out of a .nevpt2gold file
    (layout: ``generate_golden.write_golden``)."""
    want = {"ci", "class_energies", "class_norms"}
    out = {}
    with open(path, "rb") as f:
        assert f.read(8) == gg.GOLDEN_MAGIC, f"{path} is not a golden file"
        out["ncas"], _, _, out["ncore"], _ = struct.unpack("<iiiiq", f.read(24))
        out["e_casci"], out["e_corr"] = struct.unpack("<dd", f.read(16))
        (n,) = struct.unpack("<i", f.read(4))
        for _ in range(n):
            (ln,) = struct.unpack("<i", f.read(4))
            name = f.read(ln).decode()
            (nd,) = struct.unpack("<i", f.read(4))
            shape = struct.unpack(f"<{nd}q", f.read(8 * nd))
            nbytes = 8 * int(np.prod(shape))
            if name in want:
                out[name] = np.frombuffer(f.read(nbytes), "<f8").reshape(shape)
            else:
                f.seek(nbytes, 1)
    return out


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


#: Moved to generate_golden.py, which now writes block2's PC answer itself.
df_chemists_eris = gg._block2_df_eris


def solve_stats(key: str, a: np.ndarray, b: np.ndarray) -> dict:
    """What ``lstsq(rcond=None)`` sees for one class: per external tuple a
    ``d x d`` matrix; its cutoff is ``eps * d * s_max``."""
    s = np.linalg.svd(a, compute_uv=False) if len(a) else np.zeros((0, 0))
    d = a.shape[-1]
    smax = s[:, :1] if len(s) else s
    nonzero = smax[:, 0] > 0 if len(s) else np.zeros(0, bool)
    rel = np.where(smax > 0, s / np.where(smax > 0, smax, 1), 0.0)
    cut = np.finfo(float).eps * d
    kept = rel > cut
    r = rel[nonzero]
    k = kept[nonzero]
    # Energy at other cutoffs, via the SVD: c = V S^-1 U^T b on kept modes.
    u, sv, vt = np.linalg.svd(a) if len(a) else (None, None, None)
    eners = {}
    for rc in (None, *RCONDS):
        thr = (cut if rc is None else rc) * (sv[:, :1] if len(a) else 0)
        e = 0.0
        if len(a):
            inv = np.where(sv > thr, 1.0 / np.where(sv > thr, sv, 1), 0.0)
            ub = np.einsum("nji,nj->ni", u, b)
            c = np.einsum("nij,ni->nj", vt, inv * ub)
            e = float(-(c * b).sum())
        eners["lstsq" if rc is None else f"{rc:g}"] = e
    asym = float(np.abs(a - a.transpose(0, 2, 1)).max() / np.abs(a).max()) if len(a) else 0.0
    # block2's own answer for this class, to check the SVD re-solve against.
    e_block2 = -sum(float((np.linalg.lstsq(a[n], b[n], rcond=None)[0] * b[n]).sum())
                    for n in range(len(a)))
    return {
        "key": key,
        "energy_block2_lstsq": e_block2,
        "max_rel_asymmetry": asym,
        "d": int(d),
        "tuples": int(len(a)),
        "zero_tuples": int((~nonzero).sum()),
        "dropped_modes": int((~k).sum()),
        "rank_deficient_tuples": int((~k).any(axis=1).sum()) if r.size else 0,
        "min_kept_rel_sv": float(r[k].min()) if k.any() else None,
        "max_dropped_rel_sv": float(r[~k].max()) if (~k).any() else None,
        "energy_by_rcond": eners,
    }


def pc_metrics(mc, ncore: int, nvirt: int) -> dict:
    """The PC metric ``S = <bra|ket>`` per class, derived with block2's own
    Wick machinery (``icnevpt2_full`` builds only ``<bra|[H_D, ket]>`` and
    never forms S) and reshaped exactly as ``kernel`` reshapes its matrices."""
    from pyblock2.icmr import eri_helper
    from pyblock2.icmr import icnevpt2_full as m

    E1, E2, E3, E4 = eri_helper.init_pdms(mc=mc, pdm_eqs=m.pdm_eqs)
    ncas = mc.ncas
    md = {"E1": E1, "E2": E2, "E3": E3, "E4": E4, "deltaII": np.eye(ncore),
          "deltaEE": np.eye(nvirt), "deltaAA": np.eye(ncas),
          "ident1": np.ones((1,)), "ident2": np.ones((1, 1)),
          "ident3": np.ones((1, 1, 1))}
    out = {}
    for key in PC_SOLVE_ORDER:
        n = len(key)
        hkey = key[:-1]
        shape = [[ncore, ncas, nvirt][ix] for ix in m._key_idx(hkey)]
        bra_map = dict(zip(key[9 - n:4 - n], key[4 - n:-1], strict=True))

        def norm(bk, kk, hkey=hkey, shape=shape, bra_map=bra_map):
            ket = m.P(m.sub_spaces[kk])
            bra = m.P(m.sub_spaces[bk]).index_map(m.MapStrStr(bra_map))
            s = np.zeros(shape)
            eq = m.SP(bra.conjugate() * ket).to_einsum(m.PT(f"s[{hkey}]"))
            exec(eq, {"np": np}, {"s": s, **md})
            return s

        if key[-1] == "1":
            k2 = key[:-1] + "2"
            s = np.concatenate([x[..., None] for x in (
                norm(key, key), norm(key, k2), norm(k2, key), norm(k2, k2))], axis=-1)
            dcas = ncas ** (n - 5)
            s = s.reshape(-1, dcas, dcas, 2, 2).transpose(0, 1, 3, 2, 4)
            s = s.reshape(-1, dcas * 2, dcas * 2)
        else:
            s = norm(key, key)
            restrict = key[-1] in "+-"
            if n - 5 == 2 and restrict:
                dcas = ncas * (ncas + (1 if key[-1] == "+" else -1)) // 2
            else:
                dcas = ncas ** (n - 5)
            if 9 - n >= 2:
                grid = np.indices(s.shape, dtype=np.int16)
                idx = m._grid_restrict(hkey, grid, restrict, key[-1] == "-")
                s = s[idx].reshape(-1, dcas, dcas)
            else:
                s = s.reshape(-1, dcas, dcas)
        out[key] = s
    return out


#: Relative metric-eigenvalue thresholds for the orthogonalized solve.
METRIC_THRESHOLDS = (1e-14, 1e-12, 1e-10, 1e-8, 1e-6)


def metric_stats(s: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """Spectrum of the metric, and the class energy from the conventional
    ic route -- orthogonalize on S's eigenvectors above a threshold, solve the
    projected ``H_D`` there -- next to what lstsq on ``H_D`` alone gives."""
    if len(s) == 0:  # e.g. ijrs- with a single core orbital: no tuples
        return {}
    w, v = np.linalg.eigh(0.5 * (s + s.transpose(0, 2, 1)))
    wmax = np.abs(w).max(axis=1, keepdims=True)
    rel = w / np.where(wmax > 0, wmax, 1)
    eners = {}
    neg_h = 0
    for thr in METRIC_THRESHOLDS:
        e = 0.0
        for n in range(len(s)):
            keep = rel[n] > thr
            x = v[n][:, keep] / np.sqrt(w[n][keep])
            ah = x.T @ a[n] @ x
            bh = x.T @ b[n]
            if thr == METRIC_THRESHOLDS[0]:
                neg_h += int((np.linalg.eigvalsh(0.5 * (ah + ah.T)) <= 0).sum())
            e -= float(bh @ np.linalg.solve(ah, bh))
        eners[f"{thr:g}"] = e
    # Where lstsq's A_D has negative eigenvalues, and what they carry.
    wa, va = np.linalg.eigh(0.5 * (a + a.transpose(0, 2, 1)))
    amax = np.abs(wa).max(axis=1, keepdims=True)
    cut = np.finfo(float).eps * a.shape[-1] * amax
    proj = np.einsum("nji,nj->ni", va, b)
    kept = np.abs(wa) > cut
    contrib = np.where(kept, -proj**2 / np.where(kept, wa, 1), 0.0)
    negk = kept & (wa < 0)
    return {
        "metric_min_rel_eig": float(rel.min()),
        "metric_neg_eigs": int((rel < 0).sum()),
        "metric_min_kept_rel_eig": float(rel[rel > METRIC_THRESHOLDS[0]].min()),
        "metric_max_null_rel_eig": float(rel[rel <= METRIC_THRESHOLDS[0]].max())
        if (rel <= METRIC_THRESHOLDS[0]).any() else None,
        "metric_eigs_below": {f"{t:g}": int((rel <= t).sum()) for t in METRIC_THRESHOLDS},
        "orth_projected_HD_nonpositive": neg_h,
        "energy_by_metric_threshold": eners,
        "lstsq_HD_neg_eigs_kept": int(negk.sum()),
        "lstsq_HD_neg_eig_energy": float(contrib[negk].sum()),
        "lstsq_HD_max_rel_neg_eig": float((-wa / amax)[negk].max()) if negk.any() else 0.0,
    }


def run(args) -> dict:
    from pyblock2.icmr import eri_helper, icnevpt2_full, scnevpt2

    auxbasis = f"{args.basis}-jkfit" if args.df else None
    t0 = time.perf_counter()
    mc, pt = gg._build_state(args.atom, args.basis, args.ncas, args.nelecas, auxbasis)
    t_state = time.perf_counter() - t0
    rec: dict = {"case": args.golden.stem, "orbitals": args.orbitals,
                 "rdm_threads": args.rdm_threads,
                 "t_state": t_state, "rss_state_gib": rss_gib()}

    gold = read_golden(args.golden)
    rec["ci_max_abs_diff"] = float(np.abs(np.asarray(pt.ci) - gold["ci"]).max())
    rec["ci_bit_identical"] = bool(np.array_equal(np.asarray(pt.ci), gold["ci"]))
    rec["e_casci_diff"] = float(mc.e_tot - gold["e_casci"])

    if args.orbitals == "golden":
        mc.mo_coeff, mc.ci, mc.mo_energy = pt.mo_coeff, pt.ci, pt.mo_energy
    else:
        # What block2 does on its own: canonicalize the raw CASCI orbitals
        # with cas_natorb=False. Compare to the golden's orbitals.
        mo, _, moe = mc.canonicalize(mc.mo_coeff, ci=mc.ci, cas_natorb=False)
        ncore, nocc = mc.ncore, mc.ncore + mc.ncas
        s = mo.T @ mc._scf.get_ovlp() @ pt.mo_coeff
        rec["canon_core_virt_overlap_dev"] = float(max(
            np.abs(np.abs(np.diag(s)[:ncore]) - 1).max(),
            np.abs(np.abs(np.diag(s)[nocc:]) - 1).max()))
        rec["canon_mo_energy_core_virt_diff"] = float(max(
            np.abs(moe[:ncore] - pt.mo_energy[:ncore]).max(),
            np.abs(moe[nocc:] - pt.mo_energy[nocc:]).max()))
        rec["canon_active_unchanged"] = bool(
            np.array_equal(mo[:, ncore:nocc], mc.mo_coeff[:, ncore:nocc]))

    # Both kernels build the dense 4-RDM through fci.rdm.make_dm1234 (and
    # kernel(pdms=...) leaves its `tpdms` timer unbound and raises); build it
    # once, time it, and hand the same arrays to both.
    from pyscf.fci import rdm as fci_rdm

    make_dm1234, cache = fci_rdm.make_dm1234, {}

    def cached_dm1234(*a, **k):
        if "dms" not in cache:
            t = time.perf_counter()
            from pyscf import lib

            # Threads here only: the state is already built single-threaded.
            with lib.with_omp_threads(args.rdm_threads):
                cache["dms"] = make_dm1234(*a, **k)
            rec["t_dm1234"] = time.perf_counter() - t
            rec["rss_dm1234_gib"] = rss_gib()
        return cache["dms"]

    fci_rdm.make_dm1234 = cached_dm1234

    if args.df:
        df_eris = df_chemists_eris(pt, mc.with_df)
        # kernel(eris=...) leaves its `teris` timer unbound and raises, so the
        # DF integrals go in through the helper it calls instead.
        eri_helper.init_eris = lambda *a, **k: df_eris

    def one(mod, cls, label):
        ic = cls(mc)
        ic.verbose = 0
        ic.canonicalized = args.orbitals == "golden"
        t = time.perf_counter()
        ic.kernel()
        rec[f"t_{label}"] = time.perf_counter() - t
        rec[f"rss_{label}_gib"] = rss_gib()
        rec[f"{label}_times"] = {k: float(v) for k, v in ic.sub_times.items()}
        return {B2_TO_PYSCF[k]: float(v) for k, v in ic.sub_eners.items()}

    golden = dict(zip(gg.CLASSES, map(float, gold["class_energies"]), strict=True))
    rec["golden"] = golden
    if "sc" in args.which:
        rec["sc"] = one(scnevpt2, scnevpt2.WickSCNEVPT2, "sc")
    if "pc" in args.which:
        stats: list = []
        solve = icnevpt2_full._linear_solve

        mats: dict = {}

        def recording(a, b):
            key = PC_SOLVE_ORDER[len(stats)]
            stats.append(solve_stats(key, a, b))
            if args.metric:
                mats[key] = (a, b)
            return solve(a, b)

        icnevpt2_full._linear_solve = recording
        rec["pc"] = one(icnevpt2_full, icnevpt2_full.WickICNEVPT2, "pc")
        rec["pc_solves"] = stats
        if args.metric:
            t = time.perf_counter()
            nvirt = mc.mo_coeff.shape[1] - mc.ncore - mc.ncas
            for st, (key, s) in zip(stats, pc_metrics(mc, mc.ncore, nvirt).items(),
                                    strict=True):
                st.update(metric_stats(s, *mats[key]))
            rec["t_metric"] = time.perf_counter() - t
    return rec


def report(rec: dict) -> None:
    print(f"== {rec['case']}  orbitals={rec['orbitals']}")
    print(f"state: ci max|diff| vs golden {rec['ci_max_abs_diff']:.1e} "
          f"(bit-identical: {rec['ci_bit_identical']}), "
          f"E_CASCI diff {rec['e_casci_diff']:.1e}, {rec['t_state']:.1f}s")
    for k in ("canon_core_virt_overlap_dev", "canon_mo_energy_core_virt_diff",
              "canon_active_unchanged"):
        if k in rec:
            print(f"{k}: {rec[k]}")
    g = rec["golden"]
    print(f"{'class':6} {'golden SC':>18} {'block2 SC':>18} {'SC-gold':>9}"
          f" {'block2 PC':>18} {'PC-SC':>9}")
    for c in gg.CLASSES:
        sc, pc = rec.get("sc", {}).get(c), rec.get("pc", {}).get(c)
        print(f"{c:6} {g[c]:18.12f} "
              + (f"{sc:18.12f} {sc - g[c]:9.1e} " if sc is not None else " " * 29)
              + (f"{pc:18.12f} {pc - (sc if sc is not None else g[c]):9.1e}"
                 if pc is not None else ""))
    tot = {k: sum(rec[k].values()) for k in ("golden", "sc", "pc") if k in rec}
    print("total  " + "  ".join(f"{k}={v:.12f}" for k, v in tot.items()))
    if "t_dm1234" in rec:
        print(f"make_dm1234 (shared by SC and PC): {rec['t_dm1234']:.1f}s, "
              f"peak RSS after {rec['rss_dm1234_gib']:.2f} GiB")
    for lab in ("sc", "pc"):
        if f"t_{lab}" in rec:
            ts = " ".join(f"{k}={v:.1f}" for k, v in rec[f"{lab}_times"].items())
            print(f"{lab}: {rec[f't_{lab}']:.1f}s wall, peak RSS "
                  f"{rec[f'rss_{lab}_gib']:.2f} GiB  [{ts}]")
    for s in rec.get("pc_solves", []):
        e = s["energy_by_rcond"]
        print(f"  {s['key']:9} d={s['d']:5} tuples={s['tuples']:4} "
              f"zero={s['zero_tuples']:4} dropped={s['dropped_modes']:6} "
              f"min_kept={s['min_kept_rel_sv'] or 0:.1e} "
              f"max_dropped={s['max_dropped_rel_sv'] or 0:.1e} "
              f"asym={s['max_rel_asymmetry']:.0e} E={s['energy_block2_lstsq']:.12f} "
              f"svd-lstsq={e['lstsq'] - s['energy_block2_lstsq']:.0e}  "
              + " ".join(f"dE[{k}]={v - e['lstsq']:.1e}"
                         for k, v in e.items() if k != "lstsq"))
        if "metric_min_rel_eig" in s:
            em = s["energy_by_metric_threshold"]
            print(f"  {'':9} metric: min_rel_eig={s['metric_min_rel_eig']:.1e} "
                  f"min_kept={s['metric_min_kept_rel_eig']:.1e} "
                  f"max_null={s['metric_max_null_rel_eig'] or 0:.1e} "
                  f"neg={s['metric_neg_eigs']} below={s['metric_eigs_below']} "
                  f"projHD<=0: {s['orth_projected_HD_nonpositive']}; lstsq H_D: "
                  f"{s['lstsq_HD_neg_eigs_kept']} neg eigs kept (max rel "
                  f"{s['lstsq_HD_max_rel_neg_eig']:.1e}) carrying "
                  f"{s['lstsq_HD_neg_eig_energy']:.2e} Eh")
            print(f"  {'':9} orthogonalized E - lstsq E: "
                  + " ".join(f"[{k}]={v - e['lstsq']:.1e}" for k, v in em.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atom", default="N 0 0 0; N 0 0 1.1")
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--ncas", type=int, default=10)
    ap.add_argument("--nelecas", type=int, default=10)
    ap.add_argument("--df", action="store_true")
    ap.add_argument("--golden", type=Path, default=None,
                    help="default: golden/n2_<basis>_cas<nelecas><ncas>[_df]")
    ap.add_argument("--orbitals", choices=("golden", "block2"), default="golden")
    ap.add_argument("--which", choices=("sc", "pc", "scpc"), default="scpc")
    ap.add_argument("--metric", action="store_true",
                    help="also build the PC metric S and re-solve each class "
                         "the orthogonalized way (pc_metrics)")
    ap.add_argument("--rdm-threads", type=int, default=1,
                    help="OpenMP threads for the dense 4-RDM only (default 1, "
                         "the golden's setting; anything else changes its bits)")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the record, floats as exact hex strings")
    args = ap.parse_args()
    if args.golden is None:
        name = (f"n2_{args.basis.replace('-', '')}_cas{args.nelecas}{args.ncas}"
                + ("_df" if args.df else ""))
        args.golden = Path(__file__).resolve().parent.parent / "golden" / f"{name}.nevpt2gold"
    rec = run(args)
    report(rec)
    if args.json:
        def hexify(x):
            if isinstance(x, float):
                return x.hex()
            if isinstance(x, dict):
                return {k: hexify(v) for k, v in x.items()}
            if isinstance(x, list):
                return [hexify(v) for v in x]
            return x
        args.json.write_text(json.dumps(hexify(rec), indent=1))


if __name__ == "__main__":
    main()
