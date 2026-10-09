#!/usr/bin/env python3
r"""Build a golden SC-NEVPT2 CASSCF-input reference file, via PySCF.

This is the **only** file in this demo that defines the golden state
(``block2_pc_probe.py`` imports :func:`_build_state`). It runs a plain
``mcscf.CASCI`` diagonalisation on converged RHF orbitals -- bit-identical
run to run given the thread and Davidson settings below, unlike
a fresh CASSCF orbital optimisation (see :func:`_build_state` for why CASCI
is used as the reproducible golden state) -- canonicalises it exactly as
``pyscf.mrpt.NEVPT.kernel``
does, and writes everything a CASSCF/CASCI package would hand a downstream
NEVPT2 driver into one flat binary file (see :func:`write_golden`'s
docstring for the exact layout -- ``src/golden/golden.cppm`` is its C++ reader):

* the CI vector,
* the 1- and 2-particle density matrices (cheap; PySCF's ``fci.rdm.make_dm123``
  convention),
* every MO-integral block the eight SC-NEVPT2 classes read
  (:func:`_integral_blocks`), and
* PySCF's own per-class SC-NEVPT2 norms/energies, as the golden answer
  ``apps/integral_direct/main.cppm`` checks its GPU-built result against,
* and (unless ``--no-pc``) block2's per-class **partially-contracted**
  NEVPT2 energies, ``pc_class_energies`` (eight, :data:`CLASSES` order) and
  ``e_pc_total`` (one element), from ``pyblock2.icmr.icnevpt2_full`` on the
  same state (:func:`_block2_pc`). PySCF has no PC-NEVPT2, so these are
  block2's answer and **nothing else checks them**; what is checked is that
  block2 sees this state (its SC equals PySCF's per class, PC Sijrs equals
  SC Sijrs) and that the PC metric is gapped enough for the number to be
  well-defined (:func:`_check_pc_conditioning`, which refuses N2
  CAS(12,12) -- hence ``--no-pc`` for those two files, and CAS(10,12) as the
  12-orbital PC case). The header's ``e_nevpt2_total`` stays the SC total.

``--df`` writes the **density-fitted** variant instead (``*_df.nevpt2gold``,
the input ``apps/density_fit/main.cppm`` checks itself against). The whole
calculation is then density-fitted -- RHF, CASCI, canonicalisation, ``h1eff``
and every NEVPT2 integral block -- with one auxiliary basis (``--auxbasis``,
default ``<basis>-jkfit``, which is what PySCF's own ``density_fit()`` picks
for the cc-pVXZ family). The golden answer is then PySCF's DF-NEVPT2
(``pyscf.mrpt.dfnevpt2``), NOT the conventional one: the two differ by the
fitting error (~1e-5 Eh), so comparing a DF result against a conventional
golden at 1e-7 could not tell a working DF path from a broken one. On top of
the arrays above (the same names, rebuilt from the fitted integrals, so the
unmodified ``nevpt2_demo`` runs on a DF file too) a DF file carries the four
MO three-index blocks SC-NEVPT2 needs -- ``B_aa`` ``(L|tu)``, ``B_ca``
``(L|it)``, ``B_va`` ``(L|at)``, ``B_cv`` ``(L|ia)`` -- in the
symmetric-metric convention, ``(pq|rs) = sum_L B[L,p,q] B[L,r,s]``. No
virtual-virtual block is needed by any SC-NEVPT2 class.

What it deliberately does **not** write: ``dm3``/``f3ac``/``f3ca``. Building
those -- the ``n_act^6`` 3-RDM and the two integral-contracted 4-RDM digests --
is the expensive, exponential-in-active-space step the demo puts on the
GPU; producing that from the CI vector here in PySCF would defeat the point
of the demo.

Usage::

    uv run python reference_data/generate_golden.py

regenerates the committed ``golden/n2_ccpvdz_cas1010.nevpt2gold`` (the default
case, N2/cc-pVDZ CAS(10,10)). Pass ``--atom``/``--basis``/``--ncas``/``--nelecas``/``--out`` for a
different active space -- e.g. a fast sanity case::

    uv run python reference_data/generate_golden.py \\
        --atom "N 0 0 0; N 0 0 1.1" --basis sto-3g --ncas 4 --nelecas 4 \\
        --name n2_sto3g_cas44 --out /tmp/n2_sto3g_cas44.nevpt2gold

the committed 12-orbital files (``--nelecas`` before ``--ncas`` in the
name: CAS(electrons, orbitals))::

    uv run python reference_data/generate_golden.py --ncas 12 --nelecas 10 \\
        --name n2_ccpvdz_cas1012                    # SC + PC, ~1 h (4-RDM)
    uv run python reference_data/generate_golden.py --ncas 12 --nelecas 12 --no-pc \\
        --name n2_ccpvdz_cas1212

and the committed density-fitted files::

    uv run python reference_data/generate_golden.py --df          # n2_ccpvdz_cas1010_df
    uv run python reference_data/generate_golden.py --df --ncas 12 --nelecas 12 --no-pc \\
        --name n2_ccpvdz_cas1212_df
    uv run python reference_data/generate_golden.py --df --basis cc-pvtz \\
        --name n2_ccpvtz_cas1010_df

and the salicylaldimine CAS(8,8)/6-31G** pair (gitignored, 145/166 MB; the
pi space picked by irrep, the DF one fitted with 6-31G**-RIFIT)::

    uv run python reference_data/generate_golden.py \\
        --atom reference_data/geometries/salicylaldimine.xyz --basis '6-31g**' \\
        --ncas 8 --nelecas 8 --cas-irreps 'A"=8' --core-irreps "A'=27,A\"=1" \\
        --name salicylaldimine_631gss_cas88       # add --df --auxbasis
        # reference_data/basis/6-31gss-rifit.nw and a _df name for the twin
"""

from __future__ import annotations

import os

#: Single-threaded, forced (not setdefault: an inherited thread count would
#: silently bring the nondeterminism back). Threaded, PySCF's OpenMP/BLAS
#: reductions differ at ~1e-14 run to run, and that was enough to move a
#: default-tolerance CASCI by 7e-10 and E_corr by 7e-8 on BOTH paths. Must
#: precede the numpy/PySCF imports: the thread pools read it once.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

#: The CASCI Davidson tolerance. PySCF's default (1e-8) stops at a residual of
#: ~1e-4 (the residual threshold is sqrt(conv_tol)), and E_corr feeds on the CI
#: vector linearly: at CAS(10,10) the default-tolerance E_corr sat ~1e-7 from
#: the converged one. Tighter buys nothing measurable: the FCI solver's
#: ``lindep`` (1e-12) drops any correction with |r|^2 below it, so the residual
#: floors at ~1e-6 whatever conv_tol says (that is why 1e-14 "runs out of
#: cycles"). With point-group symmetry on (:func:`_build_mol`), every committed
#: case at 1e-12 agrees
#: with a residual-1e-10 run (``lindep`` lowered) to <=1.4e-9 in E_corr
#: (docs/reference-data.md, "Reproducibility").
FCI_CONV_TOL = 1e-12

#: Abelian subgroup used for each linear point group. Symmetry is NOT
#: optional: N2's canonical virtuals come in exactly degenerate
#: pairs (pi, delta), and the Srs class (``-N/(diff + h/N)`` summed per
#: virtual pair) is not invariant to a rotation inside such a pair -- rotating
#: them on one fixed CI vector moved Srs by 3e-6 while Sr stayed put to 1e-15.
#: Without symmetry that rotation is picked by numerical noise, and CAS(12,12)
#: E_corr scattered over ~2e-5 even between fully converged CI vectors.
#: :func:`_build_mol` auto-detects the group; PySCF already drops a
#: non-abelian group to an abelian subgroup (Td -> D2) but keeps the linear
#: ones, whose cylindrical-symmetry FCI refuses an active space holding half
#: of a pi pair (N2 CAS(10,10)) -- hence this map.
LINEAR_SUBGROUP = {"Dooh": "D2h", "Coov": "C2v"}

#: Two canonical core or virtual orbitals in the same irrep closer than this
#: (Eh) make the pair classes ill-defined again: an abelian subgroup does not
#: split every degenerate set (Td's E becomes A + A in D2), and a gap only
#: pins the rotation between them to ~(Fock noise)/gap radians.
#: :func:`_check_degeneracy` refuses such a state rather than write a golden
#: whose E_corr depends on noise. The numbers behind 1e-4, all measured on N2:
#: Fock noise ~1.4e-9 (how far symmetry-degenerate partners split), Srs
#: sensitivity ~2.3e-5 Eh/rad (a measured rotation test), so a 1e-4 gap costs
#: ~3e-10 Eh; the smallest real same-irrep gap in the committed goldens is
#: 2.5e-2 (cc-pVTZ). A 1.5e-6 gap (CH4 CAS(4,4), whose active space cuts a
#: t2 set) is refused: ~2e-8 Eh, too close to the 1e-7 check.
DEGENERACY_TOL = 1e-4

#: The eight class labels, fixed order -- the order
#: ``pyscf.mrpt.nevpt2.NEVPT.kernel`` evaluates them in. The
#: C++ reader (``src/energy/energy.cpp``) hardcodes this same order rather than
#: reading class names out of the binary file, so the two must stay in sync.
CLASSES = ("Sr", "Si", "Sijrs", "Sijr", "Srsi", "Srs", "Sij", "Sir")

#: block2's class keys (``ic.sub_eners``, both kernels) -> :data:`CLASSES`.
#: The PC ``sub_spaces`` (``rabcpqg*``, ``ijrs+``, ...) fold onto the same
#: eight keys inside block2.
B2_TO_CLASS = {
    "r": "Sr", "i": "Si", "ijrs": "Sijrs", "ijr": "Sijr",
    "rsi": "Srsi", "rs": "Srs", "ij": "Sij", "ir": "Sir",
}

#: block2 SC must reproduce PySCF SC to this, class by class, before any PC
#: number from it is written: anything else means block2 saw a different
#: state, integral set or RDM, and its PC answer would be for that instead.
#: Measured agreement is <=4.4e-10 on every committed state.
#: The same tolerance gates PC Sijrs == SC Sijrs (no active index, so the
#: two contractions are one function).
BLOCK2_SC_TOL = 1e-7

#: The PC conditioning guard (:func:`_check_pc_conditioning`). block2 solves
#: each external tuple's ``H_D c = b`` by ``lstsq(rcond=None)`` -- a
#: pseudo-inverse dropping singular values at or below ``eps * d * s_max`` --
#: and the PC energy is a well-defined number only if that cut sits in a gap
#: of the spectrum: then every cut inside the gap (block2's, or the one a GPU
#: solve picks) drops the same modes and gives the same energy. The guard is
#: the gap ratio, smallest kept / largest dropped singular value (or / the
#: cut, if nothing is dropped), minimised over every tuple of every PC block,
#: and must be at least this. A fixed raised-cut test does not work: on
#: cc-pVTZ CAS(10,10) real modes start at 2.5e-11, so any fixed probe cut
#: near there drops physics, not noise. Measured: N2 cc-pVDZ
#: CAS(10,10) ~1e6 (null modes <=1e-16, real >=1.8e-10), cc-pVTZ CAS(10,10)
#: ~1e4 (2.2e-15 / 2.5e-11), cc-pVDZ CAS(12,12) ~1.2 (3.8e-13 / 4.5e-13: a
#: continuum, one 1s orbital in the active space, PC Sr moving ~2e-7 between
#: defensible cuts -- more than the 1e-7 the GPU is checked to). 1e3 leaves a
#: solve three decades to place its own cut. A state below it is refused, as
#: :func:`_check_degeneracy` refuses a noise-dependent SC one.
PC_MIN_GAP = 1e3

#: The order ``icnevpt2_full.kernel`` calls ``_linear_solve`` in: every PC
#: ``sub_spaces`` key except the ``...2`` halves, which are solved jointly
#: with their ``...1`` partner. ``rabcpqg*`` is Sr, ``iabcpqg*`` Si,
#: ``irabpq1`` Sir; the ``+``/``-`` pairs are the other classes' two spin
#: couplings.
PC_SOLVE_ORDER = (
    "ijrs+", "ijrs-", "rsiap+", "rsiap-", "ijrap+", "ijrap-", "rsabpq+",
    "rsabpq-", "ijabpq+", "ijabpq-", "irabpq1", "rabcpqg*", "iabcpqg*",
)

#: Magic bytes identifying a golden-reference file -- see :func:`write_golden`'s
#: docstring for the full binary layout (``src/golden/golden.cppm`` reads it).
GOLDEN_MAGIC = b"NEVPT2G1"


def write_golden(
    path: Path,
    *,
    ncas: int,
    nelec_a: int,
    nelec_b: int,
    ncore: int,
    ndet: int,
    e_casci: float,
    e_nevpt2_total: float,
    arrays: dict[str, np.ndarray],
) -> None:
    """Write the custom little-endian binary golden-reference format.

    Deliberately not ``.npz`` (a zip container): the runtime demo is a plain
    C++ GPU program (``src/``), and parsing a zip + ``.npy`` headers there
    would need either an extra dependency or a hand-rolled zip reader for no
    real benefit. This format is a flat, trivially-parsed sequence instead.

    Layout (all little-endian, doubles are f64, no padding)::

        8s      magic "NEVPT2G1"
        i i i i q  ncas, nelec_a, nelec_b, ncore, ndet
        d d     e_casci, e_nevpt2_total
        i       n_arrays
        repeated n_arrays times:
            i           name_len
            <name_len>s name (not NUL-terminated)
            i           ndim
            q * ndim    shape
            d * prod(shape)  data, C (row-major) order

    ``src/golden/golden.cppm`` is the reader for exactly this layout.
    """
    with open(path, "wb") as f:
        f.write(GOLDEN_MAGIC)
        f.write(struct.pack("<iiiiq", ncas, nelec_a, nelec_b, ncore, ndet))
        f.write(struct.pack("<dd", e_casci, e_nevpt2_total))
        f.write(struct.pack("<i", len(arrays)))
        for name, arr in arrays.items():
            arr = np.ascontiguousarray(arr, dtype=np.float64)
            name_b = name.encode("ascii")
            f.write(struct.pack("<i", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<i", arr.ndim))
            f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))
            f.write(arr.tobytes(order="C"))


def _build_state(
    atom: str,
    basis: str,
    ncas: int,
    nelecas: int,
    auxbasis: str | None = None,
    cas_irreps: dict[str, int] | None = None,
    core_irreps: dict[str, int] | None = None,
):
    """Deterministic CASCI + NEVPT canonicalisation: a plain CASCI
    diagonalisation on converged RHF orbitals, where a fresh CASSCF orbital
    optimisation would converge to differently-rotated degenerate solutions
    run to run.

    With ``auxbasis`` the RHF is density-fitted and the CASCI is a
    ``DFCASCI`` sharing the RHF's ``with_df`` object, so one fitted integral
    set feeds everything downstream.

    Both paths are bit-identical run to run only because of two settings,
    neither of which PySCF defaults to (docs/reference-data.md,
    "Reproducibility"): single-threading, forced at the top of this module,
    and :data:`FCI_CONV_TOL` on the Davidson. Without them, both paths
    measured E_corr up to ~9e-8 apart back to back at CAS(10,10). Reproducible
    is not the same as well-defined: point-group symmetry (:func:`_build_mol`)
    is what makes the CAS(12,12) E_corr a converged number rather than one of
    many, and :func:`_check_degeneracy` refuses a state where it
    cannot.

    ``cas_irreps`` picks the active space by irrep (``{'A"': 8}``: eight
    out-of-plane orbitals in Cs) instead of the ``ncas`` RHF orbitals
    around the Fermi level, through PySCF's ``sort_mo_by_irrep``;
    ``core_irreps`` fixes the core per irrep, since PySCF's own guess (the
    ``ncore`` lowest orbitals) can leave an occupied orbital of another irrep
    out of both core and active space."""
    from pyscf import mcscf, mrpt, scf

    mol = _build_mol(atom, basis)
    mf = scf.RHF(mol)
    if auxbasis is not None:
        mf = mf.density_fit(auxbasis=auxbasis)
    mf.kernel()
    if auxbasis is not None:
        mc = mcscf.DFCASCI(mf, ncas, nelecas)
        assert mc.with_df is mf.with_df, "DFCASCI built its own DF object"
    else:
        mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver.conv_tol = FCI_CONV_TOL
    if cas_irreps is not None:
        assert sum(cas_irreps.values()) == ncas, f"{cas_irreps} is not {ncas} orbitals"
        mc.kernel(mc.sort_mo_by_irrep(cas_irreps, core_irreps))
    else:
        mc.kernel()
    assert mc.converged, f"CASCI did not converge to {FCI_CONV_TOL:g}"
    pt = mrpt.NEVPT(mc)
    pt.mo_coeff, single_ci, pt.mo_energy = pt.canonicalize(
        pt.mo_coeff, ci=pt.load_ci(), cas_natorb=True
    )
    pt.ci = single_ci
    _check_degeneracy(pt)
    return mc, pt


def _build_mol(atom: str, basis: str):
    """The molecule in its detected point group, linear groups mapped to
    :data:`LINEAR_SUBGROUP`. For N2 this is exactly ``symmetry="D2h"`` --
    same frame, same symmetry orbitals -- so the committed goldens are
    unchanged; unlike a hard-coded group, it also accepts a molecule that
    lacks D2h symmetry (H2O, CO, ...)."""
    from pyscf import gto

    kw = dict(atom=atom, basis=basis, spin=0, verbose=0, symmetry=True)
    mol = gto.M(**kw)
    sub = LINEAR_SUBGROUP.get(mol.topgroup)
    if sub is not None:
        mol = gto.M(**kw, symmetry_subgroup=sub)
    return mol


def _check_degeneracy(pt) -> None:
    """Refuse a state whose pair classes are not well-defined: two canonical
    core (Sij) or virtual (Srs) orbitals of the same irrep within
    :data:`DEGENERACY_TOL`. Within one irrep, symmetry cannot pin the
    rotation between them, so numerical noise would."""
    from pyscf import symm

    mol = pt.mol
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, pt.mo_coeff)
    nocc = pt.ncore + pt.ncas
    for name, span in (("core", range(pt.ncore)), ("virtual", range(nocc, len(orbsym)))):
        idx = list(span)
        for a, p in enumerate(idx):
            for q in idx[a + 1 :]:
                gap = abs(pt.mo_energy[p] - pt.mo_energy[q])
                if orbsym[p] == orbsym[q] and gap < DEGENERACY_TOL:
                    raise SystemExit(
                        f"{name} orbitals {p} and {q} are degenerate "
                        f"(gap {gap:.1e} Eh) within irrep "
                        f"{symm.irrep_id2name(mol.groupname, orbsym[p])} of "
                        f"{mol.groupname}: the SC-NEVPT2 pair classes would "
                        "depend on numerical noise. Pick an "
                        "active space that does not cut a degenerate set, "
                        "or another molecule."
                    )


def _pyscf_dms(pt, with_df=None):
    """The RDMs + ``f3`` objects PySCF's own ``NEVPT.kernel`` builds.

    With ``with_df`` the integrals come from ``pyscf.mrpt.dfnevpt2._ERIS``,
    called DIRECTLY: ``NEVPT.kernel`` only takes the DF path when a memory
    heuristic agrees and otherwise falls back to conventional integrals
    without saying so, which would quietly make a "DF" golden file
    conventional."""
    from pyscf import fci
    from pyscf.mrpt import dfnevpt2
    from pyscf.mrpt import nevpt2 as pn

    ncas = pt.ncas
    ci = pt.load_ci()
    dm1, dm2, dm3 = fci.rdm.make_dm123(
        "FCI3pdm_kern_sf", ci, ci, ncas, pt.nelecas
    )
    if with_df is None:
        eris = pn._ERIS(pt, pt.mo_coeff)
    else:
        eris = dfnevpt2._ERIS(pt, pt.mo_coeff, with_df)
        # The in-core branch hands back arrays; the out-of-core one parks
        # cvcv in an HDF5 temp file, which _integral_blocks cannot slice.
        assert isinstance(eris["cvcv"], np.ndarray), "DF eris went out of core"
    ncore, nocc = pt.ncore, pt.ncore + ncas
    aaaa = eris["ppaa"][ncore:nocc, ncore:nocc].copy()
    la = fci.cistring.gen_linkstr_index(range(ncas), pt.nelecas[0])
    lb = fci.cistring.gen_linkstr_index(range(ncas), pt.nelecas[1])
    dms = {
        "1": dm1,
        "2": dm2,
        "3": dm3,
        "4": None,
        "f3ca": pn._contract4pdm(
            "NEVPTkern_cedf_aedf", aaaa, ci, ncas, pt.nelecas, (la, lb)
        ),
        "f3ac": pn._contract4pdm(
            "NEVPTkern_aedf_ecdf", aaaa, ci, ncas, pt.nelecas, (la, lb)
        ),
    }
    return dm1, dm2, dms, eris


def _pyscf_per_class(pt, dms, eris) -> dict:
    """PySCF's eight ``Sxxx`` energies on the shared state -- the golden answer."""
    from pyscf.mrpt import nevpt2 as pn

    ci = pt.load_ci()
    return {
        "Sr": pn.Sr(pt, ci, dms, eris),
        "Si": pn.Si(pt, ci, dms, eris),
        "Sijrs": pn.Sijrs(pt, eris),
        "Sijr": pn.Sijr(pt, dms, eris),
        "Srsi": pn.Srsi(pt, dms, eris),
        "Srs": pn.Srs(pt, dms, eris),
        "Sij": pn.Sij(pt, dms, eris),
        "Sir": pn.Sir(pt, dms, eris),
    }


def _block2_df_eris(pt, with_df):
    """block2's ``_ChemistsERIs`` filled from PySCF's DF-NEVPT2 integrals --
    the same ``ppaa/papa/pacv/cvcv/h1eff`` a ``_df`` golden is built from.

    Needed because block2 0.5.4's own ``eri_helper.init_eris`` always
    transforms the conventional 4-index AO ERIs while taking ``h1eff`` from
    the (DF) SCF's ``get_jk``, so handed a ``DFCASCI`` it silently computes a
    mixed answer; and ``kernel(eris=...)`` raises (its ``teris`` timer is
    only bound on the build path). docs/pc-nevpt2.md, "The block2 reference"."""
    from pyblock2.icmr import eri_helper
    from pyscf.mrpt import dfnevpt2

    e = dfnevpt2._ERIS(pt, pt.mo_coeff, with_df)
    ncore, ncas = pt.ncore, pt.ncas
    nmo = pt.mo_coeff.shape[1]
    nvir = nmo - ncore - ncas
    eris = eri_helper._ChemistsERIs()
    eris.ncore, eris.ncas, eris.nocc = ncore, ncas, ncore + ncas
    eris.h1eff = np.asarray(e["h1eff"])
    eris.known = ["ppaa", "papa", "pacv", "cvcv"]
    eris.ppaa = np.asarray(e["ppaa"]).reshape(nmo, nmo, ncas, ncas)
    eris.papa = np.asarray(e["papa"]).reshape(nmo, ncas, nmo, ncas)
    eris.pacv = np.asarray(e["pacv"]).reshape(nmo, ncas, ncore, nvir)
    eris.cvcv = np.asarray(e["cvcv"]).reshape(ncore, nvir, ncore, nvir)
    return eris


def _pc_cut_gap(a: np.ndarray) -> tuple[float, float, float]:
    """Where ``lstsq(rcond=None)`` cuts one PC block, as relative singular
    values: ``(gap, largest dropped, smallest kept)``, the gap being kept /
    dropped (/ the cut where a tuple drops nothing), each minimised (maxed)
    over the block's external tuples. All-zero tuples have nothing to solve
    and are skipped."""
    if len(a) == 0:  # e.g. ijrs- with a single core orbital: no tuples
        return np.inf, 0.0, np.inf
    sv = np.linalg.svd(a, compute_uv=False)
    smax = sv[:, :1]
    live = smax[:, 0] > 0
    rel = sv[live] / smax[live]
    cut = np.finfo(float).eps * a.shape[-1]
    kept = rel > cut
    min_kept = np.where(kept, rel, np.inf).min(axis=1)
    max_dropped = np.where(kept, 0.0, rel).max(axis=1)
    gap = min_kept / np.maximum(np.where((~kept).any(axis=1), max_dropped, cut),
                                np.finfo(float).tiny)
    if not len(gap):
        return np.inf, 0.0, np.inf
    return float(gap.min()), float(max_dropped.max()), float(min_kept.min())


def _block2_pc(mc, pt, with_df, sc_ref: dict, rdm4_threads: int) -> dict:
    """block2's partially-contracted NEVPT2 (``WickICNEVPT2``) on the golden's
    own state -- its canonical orbitals and CI vector, ``canonicalized =
    True`` -- returned per class. PySCF has no PC-NEVPT2; this is the only
    reference, and nothing else checks it (docs/reference-data.md, "PC fields").

    Gated before it is returned: block2's SC (``WickSCNEVPT2``, same state,
    same 4-RDM) must equal ``sc_ref`` (PySCF's) to :data:`BLOCK2_SC_TOL` per
    class, PC Sijrs must equal SC Sijrs, and every PC block must pass
    :func:`_check_pc_conditioning`.

    Both block2 kernels build the dense ``n^8`` 4-RDM themselves; it is built
    once here and handed to both (``kernel(pdms=...)`` raises in 0.5.4, so
    by patching ``fci.rdm.make_dm1234`` for the duration). ``rdm4_threads``
    applies to that build only -- the state is already single-threaded."""
    from pyblock2.icmr import eri_helper, icnevpt2_full, scnevpt2
    from pyscf import lib
    from pyscf.fci import rdm as fci_rdm

    mc.mo_coeff, mc.ci, mc.mo_energy = pt.mo_coeff, pt.ci, pt.mo_energy

    make_dm1234, init_eris = fci_rdm.make_dm1234, eri_helper.init_eris
    linear_solve = icnevpt2_full._linear_solve
    cache: dict = {}
    gaps: list = []

    def cached_dm1234(*a, **k):
        if "dms" not in cache:
            t = time.perf_counter()
            with lib.with_omp_threads(rdm4_threads):
                cache["dms"] = make_dm1234(*a, **k)
            print(f"  block2 4-RDM: {time.perf_counter() - t:.1f}s "
                  f"({rdm4_threads} thread(s))")
        return cache["dms"]

    def recording_solve(a, b):
        gaps.append((PC_SOLVE_ORDER[len(gaps)], a.shape[-1], *_pc_cut_gap(a)))
        return linear_solve(a, b)

    def run(cls) -> dict:
        ic = cls(mc)
        ic.verbose = 0
        ic.canonicalized = True
        ic.kernel()
        return {B2_TO_CLASS[k]: float(v) for k, v in ic.sub_eners.items()}

    fci_rdm.make_dm1234 = cached_dm1234
    icnevpt2_full._linear_solve = recording_solve
    if with_df is not None:
        df_eris = _block2_df_eris(pt, with_df)
        eri_helper.init_eris = lambda *a, **k: df_eris
    try:
        sc = run(scnevpt2.WickSCNEVPT2)
        pc = run(icnevpt2_full.WickICNEVPT2)
    finally:
        fci_rdm.make_dm1234, eri_helper.init_eris = make_dm1234, init_eris
        icnevpt2_full._linear_solve = linear_solve

    for c in CLASSES:
        d = sc[c] - sc_ref[c]
        assert abs(d) <= BLOCK2_SC_TOL, (
            f"block2 SC {c} is {d:.1e} from PySCF's: block2 is not seeing the "
            "golden's state, so its PC answer would not be for it"
        )
    d = pc["Sijrs"] - sc["Sijrs"]
    assert abs(d) <= BLOCK2_SC_TOL, f"PC Sijrs - SC Sijrs = {d:.1e}"
    print("  block2 SC - PySCF SC, max over classes: "
          f"{max(abs(sc[c] - sc_ref[c]) for c in CLASSES):.1e}; PC lstsq cut "
          "per block (relative singular values):")
    for key, d, gap, dropped, kept in gaps:
        print(f"    {key:9} d={d:5} largest dropped {dropped:.1e}, "
              f"smallest kept {kept:.1e}, gap {gap:.1e}")
    _check_pc_conditioning(gaps)
    return pc


def _check_pc_conditioning(gaps: list) -> None:
    """Refuse a state where lstsq's cut does not sit in a gap of at least
    :data:`PC_MIN_GAP` in every PC block: there the PC energy depends on
    where the pseudo-inverse is cut, and the golden would record one
    implementation's choice, not a number a GPU solve could be checked
    against to 1e-7."""
    key, _, gap, dropped, kept = min(gaps, key=lambda g: g[2])
    if gap < PC_MIN_GAP:
        raise SystemExit(
            f"PC block {key}: lstsq's cut has only a {gap:.1e} gap (largest "
            f"dropped singular value {dropped:.1e}, smallest kept {kept:.1e}, "
            f"relative; need {PC_MIN_GAP:g}): the spectrum is a continuum "
            "through the cut, so PC-NEVPT2 is not well-defined on this state "
            "(docs/reference-data.md, \"PC fields\"). Pass --no-pc, or pick an active "
            "space that keeps near-doubly-occupied orbitals in the core."
        )


def _integral_blocks(pt, eris) -> dict:
    """Every MO-integral block the eight SC-NEVPT2 classes read (the fields
    of the C++ ``NevptIntegralsDevice``, ``src/energy/energy.cppm``),
    sliced out of PySCF's ``eris``."""
    ncas = pt.ncas
    ncore, nocc = pt.ncore, pt.ncore + ncas

    h2e_v_Sr = eris["ppaa"][nocc:, ncore:nocc].transpose(0, 2, 1, 3)
    h1e_v_Sr = eris["h1eff"][nocc:, ncore:nocc] - np.einsum(
        "mbbn->mn", h2e_v_Sr
    )
    nvirt = pt.mo_coeff.shape[1] - nocc
    cvcv = np.asarray(eris["cvcv"]).reshape(ncore, nvirt, ncore, nvirt)

    return dict(
        h1e=eris["h1eff"][ncore:nocc, ncore:nocc],
        h2e=eris["ppaa"][ncore:nocc, ncore:nocc].transpose(0, 2, 1, 3),
        e_core=pt.mo_energy[:ncore],
        e_virt=pt.mo_energy[nocc:],
        h2e_v_Sr=h2e_v_Sr,
        h1e_v_Sr=h1e_v_Sr,
        h2e_v_Si=eris["ppaa"][ncore:nocc, :ncore].transpose(0, 2, 1, 3),
        h1e_v_Si=eris["h1eff"][ncore:nocc, :ncore],
        cvcv=cvcv,
        h2e_v_Sijr=eris["pacv"][:ncore].transpose(3, 1, 2, 0),
        h2e_v_Srsi=eris["pacv"][nocc:].transpose(3, 0, 2, 1),
        h2e_v_Srs=eris["papa"][nocc:, :, nocc:].transpose(0, 2, 1, 3),
        h2e_v_Sij=eris["papa"][:ncore, :, :ncore].transpose(1, 3, 0, 2),
        h2e_v1_Sir=eris["ppaa"][nocc:, :ncore].transpose(0, 2, 1, 3),
        h2e_v2_Sir=eris["papa"][nocc:, :, :ncore].transpose(0, 3, 1, 2),
        h1e_v_Sir=eris["h1eff"][nocc:, :ncore],
    )


def _df_blocks(pt, with_df, eris) -> dict:
    """The MO three-index blocks ``B[L,p,q]`` with ``(pq|rs) = sum_L
    B[L,p,q] B[L,r,s]`` -- the same fitted integrals ``dfnevpt2._ERIS`` built
    its four-index blocks from (PySCF stores ``_cderi`` already multiplied by
    the inverse square root of the metric, so no metric appears here).

    Checked before returning: every four-index block ``eris`` holds is rebuilt
    from these and must agree to 1e-10, so a file whose ``B`` and four-index
    arrays disagree is never written."""
    from pyscf import lib

    mo = pt.mo_coeff
    ncore, ncas = pt.ncore, pt.ncas
    nocc = ncore + ncas
    naux = with_df.get_naoaux()
    nmo = mo.shape[1]
    lpq = np.empty((naux, nmo, nmo))
    p0 = 0
    for blk in with_df.loop():
        n = blk.shape[0]
        lao = lib.unpack_tril(blk)
        lpq[p0 : p0 + n] = lib.einsum("Lmn,mp,nq->Lpq", lao, mo, mo)
        p0 += n
    assert p0 == naux

    c, a, v = slice(0, ncore), slice(ncore, nocc), slice(nocc, None)

    def rebuilt(s1, s2, s3, s4):
        return lib.einsum("Lpq,Lrs->pqrs", lpq[:, s1, s2], lpq[:, s3, s4])

    full = slice(None)
    checks = {
        "ppaa": (eris["ppaa"], rebuilt(full, full, a, a)),
        "papa": (eris["papa"], rebuilt(full, a, full, a)),
        "pacv": (eris["pacv"], rebuilt(full, a, c, v)),
        "cvcv": (
            np.asarray(eris["cvcv"]).reshape(ncore, nmo - nocc, ncore, nmo - nocc),
            rebuilt(c, v, c, v),
        ),
    }
    for key, (ref, ours) in checks.items():
        err = float(np.max(np.abs(ref - ours))) if ref.size else 0.0
        assert err < 1e-10, f"B rebuild of {key} off by {err:.2e}"

    return dict(
        B_aa=lpq[:, a, a],
        B_ca=lpq[:, c, a],
        B_va=lpq[:, v, a],
        B_cv=lpq[:, c, v],
    )


def generate(
    atom: str,
    basis: str,
    ncas: int,
    nelecas: int,
    name: str,
    auxbasis: str | None = None,
    pc: bool = True,
    rdm4_threads: int = 1,
    cas_irreps: dict[str, int] | None = None,
    core_irreps: dict[str, int] | None = None,
) -> dict:
    t0 = time.perf_counter()
    mc, pt = _build_state(atom, basis, ncas, nelecas, auxbasis, cas_irreps, core_irreps)
    with_df = mc.with_df if auxbasis is not None else None
    dm1, dm2, dms, eris = _pyscf_dms(pt, with_df)
    ref = _pyscf_per_class(pt, dms, eris)
    blocks = _integral_blocks(pt, eris)
    if with_df is not None:
        blocks.update(_df_blocks(pt, with_df, eris))
    pc_ref = _block2_pc(mc, pt, with_df, {c: ref[c][1] for c in CLASSES},
                        rdm4_threads) if pc else None
    wall = time.perf_counter() - t0

    class_norms = np.array([ref[c][0] for c in CLASSES])
    class_energies = np.array([ref[c][1] for c in CLASSES])
    e_corr = float(class_energies.sum())

    na, nb = pt.nelecas
    print(
        f"[{name}] CAS({ncas},{na + nb}) ncore={pt.ncore} n_det={pt.ci.size} "
        f"E_CASCI={mc.e_tot:.10f} E_corr(NEVPT2)={e_corr:.10f} "
        f"E_tot={mc.e_tot + e_corr:.10f}  ({wall:.1f}s to build)"
    )
    if cas_irreps is not None:
        print(f"[{name}] active space by irrep: {cas_irreps}, core: {core_irreps}")
    if with_df is not None:
        print(
            f"[{name}] density-fitted: auxbasis={auxbasis} "
            f"naux={with_df.get_naoaux()} nmo={pt.mo_coeff.shape[1]}"
        )
    if pc_ref is not None:
        print(f"[{name}] PC (block2): E_corr={sum(pc_ref.values()):.10f}  "
              + " ".join(f"{c}={pc_ref[c]:.12f}" for c in CLASSES))

    meta = dict(
        ncas=ncas,
        nelec_a=na,
        nelec_b=nb,
        ncore=pt.ncore,
        ndet=int(pt.ci.size),
        e_casci=float(mc.e_tot),
        e_nevpt2_total=e_corr,
    )
    arrays = dict(
        ci=np.ascontiguousarray(np.asarray(pt.load_ci(), dtype=np.float64)),
        dm1=dm1,
        dm2=dm2,
        class_norms=class_norms,
        class_energies=class_energies,
        **blocks,
    )
    if pc_ref is not None:
        pc_class_energies = np.array([pc_ref[c] for c in CLASSES])
        arrays["pc_class_energies"] = pc_class_energies
        arrays["e_pc_total"] = np.array([pc_class_energies.sum()])
    return meta, arrays


def _irrep_counts(text: str) -> dict[str, int]:
    """``"A'=27,A\"=1"`` -> ``{"A'": 27, 'A"': 1}`` (PySCF irrep names)."""
    counts = {}
    for item in text.split(","):
        name, _, n = item.partition("=")
        counts[name.strip()] = int(n)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atom", default="N 0 0 0; N 0 0 1.1")
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--ncas", type=int, default=10)
    ap.add_argument("--nelecas", type=int, default=10)
    ap.add_argument(
        "--name",
        default=None,
        help="case name (default: n2_ccpvdz_cas1010, with a _df suffix under --df)",
    )
    ap.add_argument(
        "--df",
        action="store_true",
        help="density-fit the whole calculation and write the B_* blocks",
    )
    ap.add_argument(
        "--auxbasis",
        default=None,
        help="auxiliary basis for --df (default: <basis>-jkfit)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output path (default: golden/<name>.nevpt2gold at the repo root)",
    )
    ap.add_argument(
        "--no-pc",
        action="store_true",
        help="skip block2's PC-NEVPT2: no pc_class_energies/e_pc_total (the "
        "CAS(12,12) files, whose PC metric is refused as ungapped)",
    )
    ap.add_argument(
        "--rdm4-threads",
        type=int,
        default=1,
        help="OpenMP threads for block2's dense 4-RDM only (default 1; the "
        "committed files were built with 1)",
    )
    ap.add_argument(
        "--cas-irreps",
        type=_irrep_counts,
        default=None,
        help="active orbitals per irrep, e.g. 'A\"=8' (default: the --ncas RHF "
        "orbitals around the Fermi level)",
    )
    ap.add_argument(
        "--core-irreps",
        type=_irrep_counts,
        default=None,
        help="core orbitals per irrep, same syntax (default: PySCF's guess, the "
        "ncore lowest orbitals; needs --cas-irreps)",
    )
    args = ap.parse_args()
    if args.auxbasis is not None and not args.df:
        ap.error("--auxbasis needs --df")
    if args.core_irreps is not None and args.cas_irreps is None:
        ap.error("--core-irreps needs --cas-irreps")
    auxbasis = (args.auxbasis or f"{args.basis}-jkfit") if args.df else None
    if args.name is None:
        args.name = "n2_ccpvdz_cas1010" + ("_df" if args.df else "")

    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "golden" / f"{args.name}.nevpt2gold"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta, arrays = generate(
        args.atom, args.basis, args.ncas, args.nelecas, args.name, auxbasis,
        pc=not args.no_pc, rdm4_threads=args.rdm4_threads,
        cas_irreps=args.cas_irreps, core_irreps=args.core_irreps,
    )
    write_golden(out_path, **meta, arrays=arrays)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
