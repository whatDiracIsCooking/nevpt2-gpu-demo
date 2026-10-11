#!/usr/bin/env python3
r"""Measure how reproducible a fresh CASSCF is under ``generate_golden.py``'s
settings, and what a ``--mo-from`` gate would have to allow.

Offline, like ``generate_golden.py`` and ``block2_pc_probe.py``: no build or
test runs it, and it writes no golden file. It exists because
docs/reference-data.md, "What 'CASSCF input' means here, and why the golden
state is CASCI", used to assert a run-to-run spread with no measurement
behind it; what it prints is what that section now quotes.

For each of ``--runs`` back-to-back runs it does, from scratch:

* RHF, then a **symmetry-adapted** ``mcscf.CASSCF`` with tight
  :data:`CASSCF_CONV_TOL` / :data:`CASSCF_CONV_TOL_GRAD` and
  ``generate_golden.FCI_CONV_TOL`` on the Davidson, in the point group
  ``generate_golden._build_mol`` detects (single-threading is forced by
  importing that module);
* the CASSCF **orbital-gradient norm** at the converged orbitals
  (``mc.get_grad`` = ``mc1step.gen_g_hop(...)[0]``, the quantity
  ``conv_tol_grad`` gates) -- the first threshold a ``--mo-from`` sidecar
  gate needs;
* a **fresh deterministic** ``mcscf.CASCI`` on those orbitals -- exactly what
  ``generate_golden._build_state`` would run on a committed sidecar -- giving
  |E_CASCI - E_CASSCF|, the second threshold;
* the generator's own downstream: ``NEVPT.canonicalize(cas_natorb=True)``,
  ``_check_degeneracy``, ``_pyscf_dms`` and ``_pyscf_per_class``, so
  **E_corr(SC)** is the number a golden built from these orbitals would
  carry;
* a fingerprint of the **stationary point**: the active-space natural
  occupations, the active and core irrep labels, how symmetry-pure the
  converged orbitals are, the core/active subspace overlaps with run 0 and
  the CI-vector overlap with run 0. A spread in E_CASSCF is numerical
  jitter; a changed fingerprint is a different solution, and the two are
  reported separately.

``--perturb EPS`` additionally rotates the RHF start by a random
**same-irrep** (so symmetry-preserving) rotation of size EPS rad, with a
different seed per run, to probe for a *different* stationary point rather
than for jitter. Without it, every run is the same deterministic calculation
and the spread it measures is the spread the generator would see.

Usage::

    uv run python reference_data/casscf_repro_probe.py --ncas 4 --nelecas 4
    uv run python reference_data/casscf_repro_probe.py              # CAS(10,10)
    uv run python reference_data/casscf_repro_probe.py --ncas 4 --nelecas 4 \
        --perturb 0.1 --json /tmp/cas44_perturbed.json
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads.
import generate_golden as gg

# isort: split
import argparse
import json
import time
from pathlib import Path

import numpy as np

#: CASSCF convergence thresholds, both far tighter than PySCF's defaults
#: (1e-7 energy, sqrt(conv_tol) ~ 3e-4 gradient). The point of the probe is
#: what a *converged* CASSCF reproduces, so the optimiser is not allowed to
#: stop on the energy while the orbital gradient is still 1e-4.
CASSCF_CONV_TOL = 1e-12
CASSCF_CONV_TOL_GRAD = 1e-8

#: Macro-iteration ceiling. Tight thresholds need more cycles than PySCF's
#: default 50; a run that hits this is reported as unconverged rather than
#: quoted.
CASSCF_MAX_CYCLE_MACRO = 200


def _irrep_purity(mol, mo) -> np.ndarray:
    """``1 - max_i w_i`` per orbital, where ``w_i`` is the fraction of the
    orbital's norm living in irrep ``i``.

    The irrep subspaces are S-orthogonal and span everything, so the weights
    sum to 1 and this is 0 for a symmetry-pure orbital. A ``--mo-from`` gate
    has to refuse orbitals that are not pure (it re-derives ``orbsym`` from
    them), so this says how pure a converged CASSCF leaves them."""
    s = mol.intor("int1e_ovlp")
    nrm = np.einsum("pi,pq,qi->i", mo, s, mo)
    weights = []
    for blk in mol.symm_orb:
        if blk.shape[1] == 0:
            continue
        bsb = blk.T @ s @ blk
        proj = blk.T @ s @ mo
        weights.append(np.einsum("km,km->m", np.linalg.solve(bsb, proj), proj) / nrm)
    return 1.0 - np.max(np.stack(weights), axis=0)


def _orbsym_names(mol, mo) -> list[str]:
    """PySCF irrep names of ``mo``, in order."""
    from pyscf import symm

    ids = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo)
    return [symm.irrep_id2name(mol.groupname, i) for i in ids]


def _perturb_within_irreps(mol, mo, eps: float, seed: int) -> np.ndarray:
    """``mo @ expm(K)`` with ``K`` antisymmetric and nonzero only between
    orbitals of the same irrep: a random start that is still symmetry-pure,
    so the optimiser starts somewhere else without the point group changing
    under it."""
    from pyscf import symm
    from scipy.linalg import expm

    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo)
    rng = np.random.default_rng(seed)
    nmo = mo.shape[1]
    k = np.zeros((nmo, nmo))
    for p in range(nmo):
        for q in range(p):
            if orbsym[p] == orbsym[q]:
                a = eps * rng.standard_normal()
                k[p, q], k[q, p] = a, -a
    return mo @ expm(k)


def _casscf_once(
    atom: str,
    basis: str,
    ncas: int,
    nelecas: int,
    cas_irreps: dict[str, int] | None,
    core_irreps: dict[str, int] | None,
    perturb: float,
    seed: int,
    conv_tol: float = CASSCF_CONV_TOL,
    conv_tol_grad: float = CASSCF_CONV_TOL_GRAD,
) -> dict:
    """One full RHF -> CASSCF -> gradient -> CASCI -> SC-NEVPT2 pass."""
    from pyscf import mcscf, mrpt, scf

    t0 = time.perf_counter()
    mol = gg._build_mol(atom, basis)
    mf = scf.RHF(mol)
    mf.kernel()
    assert mf.converged, "RHF did not converge"

    mo_start = mf.mo_coeff
    if perturb:
        mo_start = _perturb_within_irreps(mol, mo_start, perturb, seed)

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.conv_tol = conv_tol
    mc.conv_tol_grad = conv_tol_grad
    mc.max_cycle_macro = CASSCF_MAX_CYCLE_MACRO
    mc.fcisolver.conv_tol = gg.FCI_CONV_TOL
    macro = []
    mc.callback = lambda envs: macro.append(envs.get("imacro"))
    if cas_irreps is not None:
        assert sum(cas_irreps.values()) == ncas, f"{cas_irreps} is not {ncas} orbitals"
        mo_start = mc.sort_mo_by_irrep(cas_irreps, core_irreps, mo_coeff=mo_start)
    mc.kernel(mo_start)
    t_casscf = time.perf_counter() - t0

    # The gradient norm conv_tol_grad gates, at the converged orbitals: the
    # same expression a --mo-from gate would evaluate on a sidecar.
    eris = mc.ao2mo(mc.mo_coeff)
    _, _, civec = mc.casci(mc.mo_coeff, mc.ci, eris)
    casdm1, casdm2 = mc.fcisolver.make_rdm12(civec, mc.ncas, mc.nelecas)
    g_orb = mc.get_grad(mc.mo_coeff, (casdm1, casdm2), eris)

    # The same expression at the RHF orbitals, for scale: that is the input a
    # --mo-from gate has to refuse as non-stationary.
    rhf_eris = mc.ao2mo(mf.mo_coeff)
    e_casci_rhf, _, rhf_ci = mc.casci(mf.mo_coeff, None, rhf_eris)
    rhf_dm1, rhf_dm2 = mc.fcisolver.make_rdm12(rhf_ci, mc.ncas, mc.nelecas)
    g_rhf = mc.get_grad(mf.mo_coeff, (rhf_dm1, rhf_dm2), rhf_eris)

    # A fresh, deterministic CASCI on the converged orbitals -- what
    # generate_golden would run on a committed sidecar -- then the
    # generator's own downstream, unchanged.
    ci_mc = mcscf.CASCI(mf, ncas, nelecas)
    ci_mc.fcisolver.conv_tol = gg.FCI_CONV_TOL
    ci_mc.kernel(mc.mo_coeff)
    assert ci_mc.converged, "CASCI on the CASSCF orbitals did not converge"

    pt = mrpt.NEVPT(ci_mc)
    pt.mo_coeff, single_ci, pt.mo_energy = pt.canonicalize(
        pt.mo_coeff, ci=pt.load_ci(), cas_natorb=True
    )
    pt.ci = single_ci
    degeneracy = "ok"
    try:
        gg._check_degeneracy(pt)
    except SystemExit as exc:  # a probe reports the refusal instead of dying
        degeneracy = str(exc)
    _, _, dms, nev_eris = gg._pyscf_dms(pt)
    per_class = gg._pyscf_per_class(pt, dms, nev_eris)
    e_corr = float(sum(per_class[c][1] for c in gg.CLASSES))

    ncore, nocc = mc.ncore, mc.ncore + ncas
    occ = np.linalg.eigvalsh(casdm1)[::-1]
    names = _orbsym_names(mol, mc.mo_coeff)
    purity = _irrep_purity(mol, mc.mo_coeff)
    return dict(
        seed=seed,
        converged=bool(mc.converged),
        n_macro=(max(macro) + 1) if macro else 0,
        e_rhf=float(mf.e_tot),
        e_casscf=float(mc.e_tot),
        e_casci=float(ci_mc.e_tot),
        grad_norm=float(np.linalg.norm(g_orb)),
        grad_max=float(np.abs(g_orb).max()),
        grad_norm_rhf=float(np.linalg.norm(g_rhf)),
        e_casci_rhf=float(e_casci_rhf),
        e_corr=e_corr,
        class_energies={c: float(per_class[c][1]) for c in gg.CLASSES},
        nat_occ=occ.tolist(),
        core_irreps=names[:ncore],
        act_irreps=names[ncore:nocc],
        purity_max=float(purity.max()),
        degeneracy=degeneracy,
        mo_coeff=mc.mo_coeff,
        ci=np.asarray(pt.load_ci()),
        wall_casscf=t_casscf,
        wall=time.perf_counter() - t0,
    )


def _compare(run: dict, ref: dict, ovlp: np.ndarray, ncore: int, ncas: int) -> dict:
    """How far this run's stationary point is from run 0's: subspace
    overlaps (1 = the same space), the per-orbital active overlap matrix's
    permutation, and the CI-vector overlap."""
    out = {}
    for name, span in (("core", slice(0, ncore)), ("act", slice(ncore, ncore + ncas))):
        m = run["mo_coeff"][:, span].T @ ovlp @ ref["mo_coeff"][:, span]
        sv = np.linalg.svd(m, compute_uv=False)
        out[f"{name}_subspace_min_sv"] = float(sv.min())
        if name == "act":
            order = np.abs(m).argmax(axis=1).tolist()
            out["act_permuted"] = order != list(range(ncas))
            out["act_order"] = order
            out["act_diag_min"] = float(np.abs(np.diag(m)).min())
    ci_ovlp = float(abs(np.dot(run["ci"].ravel(), ref["ci"].ravel())))
    out["ci_overlap_defect"] = 1.0 - ci_ovlp
    out["occ_max_delta"] = float(
        np.abs(np.array(run["nat_occ"]) - np.array(ref["nat_occ"])).max()
    )
    out["same_irreps"] = (
        run["act_irreps"] == ref["act_irreps"]
        and run["core_irreps"] == ref["core_irreps"]
    )
    return out


def _spread(label: str, values: list[float]) -> str:
    lo, hi = min(values), max(values)
    return f"{label:<14} min {lo:.12f}  max {hi:.12f}  spread {hi - lo:.3e}"


def main() -> None:
    import pyscf

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atom", default="N 0 0 0; N 0 0 1.1")
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--ncas", type=int, default=10)
    ap.add_argument("--nelecas", type=int, default=10)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument(
        "--perturb",
        type=float,
        default=0.0,
        help="rotate the RHF start by this many rad within each irrep, with a "
        "per-run seed (default 0: identical back-to-back runs)",
    )
    ap.add_argument(
        "--conv-tol", type=float, default=CASSCF_CONV_TOL,
        help=f"CASSCF energy threshold (default {CASSCF_CONV_TOL:g}; PySCF's own "
        "is 1e-7)",
    )
    ap.add_argument(
        "--conv-tol-grad", type=float, default=CASSCF_CONV_TOL_GRAD,
        help=f"CASSCF orbital-gradient threshold (default {CASSCF_CONV_TOL_GRAD:g}; "
        "PySCF's own is sqrt(conv_tol))",
    )
    ap.add_argument("--cas-irreps", type=gg._irrep_counts, default=None)
    ap.add_argument("--core-irreps", type=gg._irrep_counts, default=None)
    ap.add_argument("--json", default=None, help="write the per-run record here")
    args = ap.parse_args()
    if args.core_irreps is not None and args.cas_irreps is None:
        ap.error("--core-irreps needs --cas-irreps")

    tag = f"CAS({args.ncas},{args.nelecas}) {args.basis}"
    print(
        f"{tag}  runs={args.runs}  perturb={args.perturb} rad  "
        f"PySCF {pyscf.__version__}  numpy {np.__version__}\n"
        f"CASSCF conv_tol={args.conv_tol:g} conv_tol_grad={args.conv_tol_grad:g} "
        f"FCI conv_tol={gg.FCI_CONV_TOL:g}",
        flush=True,
    )

    runs = []
    for i in range(args.runs):
        r = _casscf_once(
            args.atom, args.basis, args.ncas, args.nelecas,
            args.cas_irreps, args.core_irreps, args.perturb, seed=i + 1,
            conv_tol=args.conv_tol, conv_tol_grad=args.conv_tol_grad,
        )
        runs.append(r)
        print(
            f"run {i}: conv={r['converged']} macro={r['n_macro']:3d} "
            f"E_CASSCF={r['e_casscf']:.12f} |g_orb|={r['grad_norm']:.3e} "
            f"E_CASCI={r['e_casci']:.12f} "
            f"|E_CASCI-E_CASSCF|={abs(r['e_casci'] - r['e_casscf']):.3e} "
            f"E_corr={r['e_corr']:.12f} purity={r['purity_max']:.1e} "
            f"({r['wall']:.1f}s)",
            flush=True,
        )
        if r["degeneracy"] != "ok":
            print(f"run {i}: _check_degeneracy refused: {r['degeneracy']}", flush=True)

    mol = gg._build_mol(args.atom, args.basis)
    ovlp = mol.intor("int1e_ovlp")
    ncore = (mol.nelectron - args.nelecas) // 2
    cmp = [_compare(r, runs[0], ovlp, ncore, args.ncas) for r in runs]

    print("\n-- spreads over the runs --")
    print(_spread("E_CASSCF", [r["e_casscf"] for r in runs]))
    print(_spread("E_CASCI", [r["e_casci"] for r in runs]))
    print(_spread("E_corr(SC)", [r["e_corr"] for r in runs]))
    print(_spread("|g_orb|", [r["grad_norm"] for r in runs]))
    print(
        f"{'|dE_CI-SCF|':<14} max "
        f"{max(abs(r['e_casci'] - r['e_casscf']) for r in runs):.3e}"
    )
    print(f"{'purity':<14} max {max(r['purity_max'] for r in runs):.3e}")
    per_class_spread = max(
        max(r["class_energies"][c] for r in runs)
        - min(r["class_energies"][c] for r in runs)
        for c in gg.CLASSES
    )
    print(f"{'per class':<14} max spread {per_class_spread:.3e}")
    r0 = runs[0]
    print(
        f"\nRHF orbitals, for scale: |g_orb|={r0['grad_norm_rhf']:.3e}  "
        f"E_CASCI(RHF)={r0['e_casci_rhf']:.12f}  "
        f"E_CASCI(RHF)-E_CASSCF={r0['e_casci_rhf'] - r0['e_casscf']:.3e}"
    )

    print("\n-- stationary point vs run 0 --")
    for i, c in enumerate(cmp):
        print(
            f"run {i}: act_min_sv={c['act_subspace_min_sv']:.12f} "
            f"core_min_sv={c['core_subspace_min_sv']:.12f} "
            f"act_permuted={c['act_permuted']} same_irreps={c['same_irreps']} "
            f"d_occ={c['occ_max_delta']:.3e} "
            f"1-|<ci|ci_0>|={c['ci_overlap_defect']:.3e}"
        )
    changed = [
        i for i, c in enumerate(cmp)
        if c["act_permuted"] or not c["same_irreps"]
        or c["act_subspace_min_sv"] < 0.99 or c["ci_overlap_defect"] > 1e-6
    ]
    print(
        "different stationary point: "
        + (f"runs {changed}" if changed else "none -- every run is the same solution")
    )
    print(f"active irreps: {runs[0]['act_irreps']}")
    print("natural occupations: " + " ".join(f"{x:.6f}" for x in runs[0]["nat_occ"]))

    if args.json:
        rec = dict(
            pyscf=pyscf.__version__,
            numpy=np.__version__,
            atom=args.atom,
            basis=args.basis,
            ncas=args.ncas,
            nelecas=args.nelecas,
            perturb=args.perturb,
            casscf_conv_tol=CASSCF_CONV_TOL,
            casscf_conv_tol_grad=CASSCF_CONV_TOL_GRAD,
            fci_conv_tol=gg.FCI_CONV_TOL,
            groupname=mol.groupname,
            runs=[{k: v for k, v in r.items() if k not in ("mo_coeff", "ci")} for r in runs],
            compare=cmp,
        )
        Path(args.json).write_text(json.dumps(rec, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
