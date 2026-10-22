#!/usr/bin/env python3
r"""The analytic **CASSCF nuclear Hessian** against **OpenMolcas's**, offline
(#223): an independent analytic answer for ``casscf_nuclear_hessian.py``
(#219), so the reference side -- ``casscf_hessian.py``'s skeleton terms
(#215), the CP-CASSCF responses and the assembly -- is checked against another
code's analytic Hessian rather than only against finite differences.

What this cannot check: the ``Z``-weighted third-derivative term (#216) and
every NEVPT2 piece. CASSCF's own multipliers vanish.

WHAT RUNS
---------

OpenMolcas (``docker/install-openmolcas.sh``'s pinned build, through its
``pymolcas`` driver) on the same molecule, basis and active space as the
case's committed ``_casscf`` state:

* ``GATEWAY``: the state's own bohr coordinates (``casscf_hessian.Skeleton``'s
  symmetry-off ``mol.atom_coords()``, which is the frame the ``.respgold``
  Hessian is written in), the basis from OpenMolcas's library by the same
  name, **no symmetry** (no ``Symmetry`` keyword: C1), so nothing is
  reoriented or symmetry-adapted. Atoms are grouped per element, as
  ``GATEWAY``'s basis blocks require, and mapped back.
* ``SEWARD`` (conventional integrals, no Cholesky: McKinley needs them),
  then ``RASSCF`` single-state CASSCF with ``Inactive = ncore``, ``Ras2 =
  ncas``, the same active electron count and spin, started (``LUMORB``) from
  the committed orbitals and converged to :data:`RASSCF_THRS`.
* ``ALASKA`` (the gradient, reported, and a check of the geometry and basis),
  ``MCKINLEY`` (the second-derivative integrals, ``CUTOFF`` :data:`CUTOFF`,
  ``NOMCLR``) and ``MCLR`` (the CP-CASSCF response and the assembled
  Hessian, ``RESIDUAL`` :data:`MCLR_RESIDUAL`, the absolute PCG residual).
  The Hessian is read from MCLR's ``UNSYM`` file (``ES20.12``, Cartesian,
  not projected), its coordinates checked against the input.

THE STATE, AND THE ORBITAL GUESS
--------------------------------

RASSCF starts from the committed state's orbitals (the ``.mogold`` state in
the golden's frame, ``casscf_nuclear_hessian.state_from_case``). It has to:
from its own SCF guess OpenMolcas converges HSOH to a different CASSCF
solution, 0.042 Eh higher. The orbitals are carried across exactly
(:func:`molcas_orbitals`): OpenMolcas's own library entry for each element is
built as a PySCF basis, the orbitals are projected onto it through the
overlap (the two libraries segment the same space differently: PySCF's
cc-pVDZ drops the diffuse primitive from its contracted ``s`` functions),
and reordered to OpenMolcas's AO order. That the import is exact is checked
three ways: the projection keeps the orbitals' overlap to
:data:`PROJECTION_TOL`; RASSCF's first iteration -- the CASCI on the guess --
is printed against the committed energy; and the run **refuses** unless
OpenMolcas's converged CASSCF energy matches the committed state's to
:data:`ENERGY_TOL`, which a different root, active space or CASSCF solution
would not do.

Neither state is exactly stationary. The committed state's residual orbital
gradient is ``|g_orb|`` 1.8e-7 (N2) and 9.3e-7 (HSOH) (docs/reference-data.md,
"The analytic CASSCF nuclear Hessian"), and its Hessian is off the
stationary point's by ``O(|g_orb|)`` (``casscf_nuclear_hessian.py``'s
committed - polished row). So OpenMolcas is compared against **both**: the
committed ``.respgold`` Hessian, and ``casscf_nuclear_hessian.polished``'s
-- the same state converged in memory to ``|g_all|`` <= 1e-11 -- which is
the one gated. The committed comparison is reported.

THE TOLERANCE (derived, printed, never tuned)
---------------------------------------------

OpenMolcas runs **twice**: from the committed orbitals (the ``.mogold``
guess) and from the polished ones. Both converge to the same state; each
Hessian is gated against the polished PySCF one. OpenMolcas's answer differs
from that through its two convergence thresholds:

* **its state**: RASSCF stops with an electronic-gradient norm ``g_M``
  (printed, under :data:`RASSCF_THRS`' gradient threshold), and where it
  stops depends on where it started. The term is the larger of OpenMolcas's
  own **spread** -- ``max |H_M(committed start) - H_M(polished start)|``, two
  of its converged answers for one state -- and ``kappa g_M``, with the
  Hessian's sensitivity to a residual gradient measured on this state,
  ``kappa = max|H_committed - H_polished| / |g_orb,committed|``. The spread
  is there because ``kappa g_M`` alone under-bounds it: ``kappa`` is the
  sensitivity along the committed state's residual direction, and RASSCF's
  residual points elsewhere. On N2 the committed start stops at ``g_M``
  2.4e-8 with its Hessian 5.4e-8 off the polished one, where ``kappa g_M``
  is 3.8e-9 (docs/reference-data.md, "The CASSCF Hessian against
  OpenMolcas", records the investigation).
* **its response**: MCLR stops each perturbation's PCG at an absolute
  residual ``|r_B|`` <= ``eps`` (:data:`MCLR_RESIDUAL`). The assembled
  element ``H_AB = L_RR + b_A . Theta_B`` then moves by ``b_A . J^-1 r_B =
  -Theta_A . r_B`` (``J`` symmetric, ``J Theta_A = -b_A``), so by at most
  ``|Theta_A| eps``. Bound: ``max_A |Theta_A| eps``, with the polished
  state's responses.

The gate is :data:`SAFETY` times their sum: the factor covers the two
programs' different scalings of the orbital gradient and the response
residual (factors of 2 between orbital-rotation conventions), which
the bound does not model. The integral screening (``CUTOFF`` 1e-14) and
the energy threshold are orders below either term.

Usage::

    uv run python reference_data/molcas_casscf_hessian.py --case n2_ccpvdz_cas44
    uv run python reference_data/molcas_casscf_hessian.py --case hsoh_def2svp_cas66 \
        --pymolcas /path/to/pymolcas --keep /tmp/hsoh-molcas

``--pymolcas`` defaults to ``$PYMOLCAS``, then ``pymolcas`` on ``PATH``
(``docker/install-openmolcas.sh``'s wrapper, ``/usr/local/bin/pymolcas`` in
the images). The measured agreement is in
docs/reference-data.md, "The CASSCF Hessian against OpenMolcas".
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads
# (docs/reference-data.md, "Reproducibility").
import generate_golden as gg  # noqa: F401  (imported for its side effect)

# isort: split
import casscf_hessian as ch
import casscf_nuclear_hessian as cnh
import nevpt2_numgrad as ngrad

# isort: split
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import golden_format as gf
import numpy as np

#: The cases with a committed ``.respgold`` (#219) -- the two #222 reads.
CASES = ("n2_ccpvdz_cas44", "hsoh_def2svp_cas66")

#: PySCF's basis name -> OpenMolcas's library label (``basis_library/``).
MOLCAS_BASIS = {"cc-pvdz": "cc-pVDZ", "def2-svp": "def2-SVP"}

#: RASSCF's ``THRS``: energy change, largest orbital rotation, largest BLB
#: (orbital gradient) element; OpenMolcas's defaults are (1e-8, 1e-4, 1e-4).
#: The BLB threshold is as tight as RASSCF reaches here: started from N2's
#: committed orbitals it sits at max BLB 1.1e-8 taking zero-length steps,
#: and at a 1e-8 threshold runs out its 200 iterations
#: (``_RC_NOT_CONVERGED_``); from the polished orbitals it reaches ~1e-12.
#: What it reaches, not the threshold, enters the tolerance (``g_M``, module
#: docstring).
RASSCF_THRS = (1e-12, 1e-8, 5e-8)

#: MCLR's ``RESIDUAL``, per case: the absolute residual norm each
#: perturbation's PCG stops at (``wfctl_hess.F90``, ``iBreak = 2``; MCLR's
#: own default is a relative 1e-4). As tight as MCLR gets on each: N2
#: reaches 1e-9 in at most ~20 steps; HSOH reaches 1e-7 in under 200, and at
#: 1e-8 its second perturbation stagnates (no convergence in 3000 steps).
#: Whichever is used enters the tolerance (``eps``, module docstring).
MCLR_RESIDUAL = {"n2_ccpvdz_cas44": 1e-9, "hsoh_def2svp_cas66": 1e-7}
#: MCLR's ``ITERATIONS`` (default 200).
MCLR_ITERATIONS = 1000

#: ALASKA's and McKinley's integral-derivative prescreening (both default
#: 1e-7, which is the size of the effects this compares).
CUTOFF = 1e-14

#: The same-state gate. The energy is read from the title of RASSCF's
#: natural-orbital file (``<project>.RasOrb.1``, 12 decimals; the log prints
#: 8). Both states are converged to a residual gradient whose effect on the
#: energy is second order (~1e-14 here); a different basis, geometry, root or
#: CASSCF solution moves it at first order. The ``--mo-from`` gate's own
#: ``|E_CASCI - E_CASSCF|`` bound.
ENERGY_TOL = 1e-9

#: The factor on the derived bound (module docstring, "THE TOLERANCE").
SAFETY = 10.0

#: OpenMolcas prints its input geometry back with 10 decimals (``FCOOR``).
COORD_TOL = 1e-9

#: The committed orbitals projected onto OpenMolcas's basis must keep their
#: overlap matrix to round-off: the two libraries' contractions span the same
#: space, only segmented differently.
PROJECTION_TOL = 1e-10

#: The orbital guess file RASSCF reads (``LUMORB``), in the run directory.
GUESS = "guess.InpOrb"


# --------------------------------------------------------------------------
# OpenMolcas: input, run, output
# --------------------------------------------------------------------------


def molcas_input(mol, basis: str, ncore: int, ncas: int, nelecas: tuple,
                 residual: float) -> tuple[str, list]:
    """The OpenMolcas input for the state, and the order its atoms come out in
    (``order[k]`` = the PySCF atom index of OpenMolcas's ``k``-th)."""
    label = MOLCAS_BASIS[basis.lower()]
    xyz = mol.atom_coords()  # bohr, the frame the .respgold Hessian is in
    symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    elements = list(dict.fromkeys(symbols))
    order: list[int] = []
    lines = ["&GATEWAY"]
    for el in elements:
        lines += ["  Basis set", f"  {el}.{label}"]
        for i, s in enumerate(symbols):
            if s == el:
                order.append(i)
                x, y, z = (repr(float(v)) for v in xyz[i])
                lines.append(f"  {el}{i + 1} {x} {y} {z}")
        lines.append("  End of basis")
    na, nb = nelecas
    lines += [
        "&SEWARD",
        "&RASSCF",
        "  LumOrb",
        f"  FileOrb = {GUESS}",
        f"  Spin = {na - nb + 1}",
        f"  Nactel = {na + nb} 0 0",
        f"  Inactive = {ncore}",
        f"  Ras2 = {ncas}",
        "  CIRoot = 1 1 1",
        "  Thrs = " + " ".join(f"{t:.1e}" for t in RASSCF_THRS),
        "&ALASKA",
        f"  CutOff = {CUTOFF:.1e}",
        "&MCKINLEY",
        f"  CutOff = {CUTOFF:.1e}",
        "  NoMCLR",
        "&MCLR",
        f"  Residual = {residual:.1e}",
        f"  Iterations = {MCLR_ITERATIONS}",
    ]
    return "\n".join(lines) + "\n", order


def molcas_root(arg: str | None, pymolcas: str) -> Path:
    """OpenMolcas's installed tree: ``--molcas-root``, ``$MOLCAS``, the
    ``MOLCAS=`` line ``docker/install-openmolcas.sh``'s wrapper sets, or
    ``/opt/openmolcas``."""
    if arg:
        return Path(arg)
    if os.environ.get("MOLCAS"):
        return Path(os.environ["MOLCAS"])
    try:
        for line in Path(pymolcas).read_text().splitlines():
            if line.startswith("MOLCAS="):
                return Path(line.split("=", 1)[1].strip())
    except (OSError, UnicodeDecodeError):
        pass
    return Path("/opt/openmolcas")


def molcas_basis(root: Path, label: str, element: str) -> list:
    """One element's entry of OpenMolcas's basis library, as a PySCF basis
    (one generally contracted shell per ``l``): exactly the exponents and
    contraction coefficients OpenMolcas uses, which need not be PySCF's
    segmentation of the same space (cc-pVDZ's are not)."""
    lib = root / "basis_library" / label.upper()
    lines = lib.read_text().splitlines()
    head = f"/{element}.{label}.".lower()
    start = next((k for k, s in enumerate(lines) if s.lower().startswith(head)), None)
    if start is None:
        sys.exit(f"molcas_casscf_hessian: no {element}.{label} in {lib}")
    # two reference lines follow the label; then "charge lmax", then per l a
    # "* x-type" comment, "nprim ncontr", the exponents, the coefficient rows
    nums: list[float] = []
    for s in lines[start + 3:]:
        if s.startswith("/"):
            break
        if s.lstrip().startswith("*") or not s.strip():
            continue
        nums += [float(t.replace("D", "E").replace("d", "e")) for t in s.split()]
    _charge, lmax = nums[0], int(nums[1])
    k, shells = 2, []
    for ang in range(lmax + 1):
        nprim, nctr = int(nums[k]), int(nums[k + 1])
        k += 2
        exps = nums[k:k + nprim]
        k += nprim
        coef = np.array(nums[k:k + nprim * nctr]).reshape(nprim, nctr)
        k += nprim * nctr
        shells.append([ang] + [[e, *row] for e, row in zip(exps, coef, strict=True)])
    return shells


def molcas_mol(mol, order: list[int], root: Path, label: str) -> tuple:
    """``mol``'s atoms in ``order`` with OpenMolcas's basis (a PySCF
    molecule), and ``perm``: OpenMolcas's AO ``k`` is its AO ``perm[k]``."""
    from pyscf import gto

    els = list(dict.fromkeys(mol.atom_pure_symbol(i) for i in order))
    basis = {el: molcas_basis(root, label, el) for el in els}
    atoms = [(mol.atom_pure_symbol(i), tuple(mol.atom_coord(i))) for i in order]
    mm = gto.M(atom=atoms, unit="Bohr", basis=basis, charge=mol.charge, spin=mol.spin,
               cart=False, verbose=0)
    perm, off = [], 0
    for ish in range(mm.nbas):
        ang, nctr = mm.bas_angular(ish), mm.bas_nctr(ish)
        nm = 2 * ang + 1
        perm += [off + c * nm + m for m in range(nm) for c in range(nctr)]
        off += nm * nctr
    return mm, perm


def molcas_orbitals(mol, C: np.ndarray, order: list[int], root: Path, label: str) -> tuple:
    """``C`` (PySCF's AOs) in OpenMolcas's AOs and AO order: centres in
    ``order``, then ``l``, then the component (``px, py, pz``; ``d2-`` ..
    ``d2+``, PySCF's own ``m`` order), then the contracted function -- the
    order OpenMolcas's orbital printout labels. Projected through the overlap,
    ``C_M = S_MM^-1 S_MP C``, exact when the two bases span the same space;
    returns ``C_M`` and the defect of that (``|C_M^T S_MM C_M - C^T S C|``)."""
    from pyscf import gto

    mm, perm = molcas_mol(mol, order, root, label)
    S_mm = mm.intor("int1e_ovlp")
    S_mp = gto.intor_cross("int1e_ovlp", mm, mol)
    C_m = np.linalg.solve(S_mm, S_mp @ C)
    defect = float(np.abs(C_m.T @ S_mm @ C_m - C.T @ mol.intor("int1e_ovlp") @ C).max())
    return C_m[perm], defect


def write_inporb(path: Path, C: np.ndarray, occ: np.ndarray) -> None:
    """An OpenMolcas ``INPORB 2.2`` orbital file (one irrep), the layout
    RASSCF's own ``RasOrb`` has, for ``LUMORB`` to read."""
    nbas, norb = C.shape

    def block(v) -> list[str]:
        return ["".join(f"{x:22.14E}" for x in v[i:i + 5]) for i in range(0, len(v), 5)]

    out = ["#INPORB 2.2", "#INFO", "* a _casscf state's orbitals (molcas_casscf_hessian.py)",
           "       0       1       0", f"{nbas:8d}", f"{norb:8d}", "#ORB"]
    for j in range(norb):
        out.append(f"* ORBITAL{1:5d}{j + 1:5d}")
        out += block(C[:, j])
    out += ["#OCC", "* OCCUPATION NUMBERS"] + block(occ)
    path.write_text("\n".join(out) + "\n")


def find_pymolcas(arg: str | None) -> str:
    for cand in (arg, os.environ.get("PYMOLCAS"), shutil.which("pymolcas")):
        if cand:
            return cand
    sys.exit("molcas_casscf_hessian: no pymolcas (--pymolcas, $PYMOLCAS, or PATH); "
             "docker/install-openmolcas.sh installs it")


def run_molcas(pymolcas: str, text: str, work: Path, project: str = "hess") -> tuple[str, Path]:
    """Run ``text`` in ``work``; returns the log and the project's WorkDir."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "scratch").mkdir(exist_ok=True)
    inp = work / f"{project}.input"
    inp.write_text(text)
    env = dict(os.environ, MOLCAS_WORKDIR=str(work / "scratch"), MOLCAS_NPROCS="1",
               OMP_NUM_THREADS="1", MOLCAS_PRINT="NORMAL")
    proc = subprocess.run([pymolcas, "-np", "1", inp.name], cwd=work, env=env,
                          capture_output=True, text=True, check=False)
    log = proc.stdout + proc.stderr
    (work / f"{project}.log").write_text(log)
    if proc.returncode != 0 or "Happy landing" not in log:
        tail = "\n".join(log.splitlines()[-30:])
        sys.exit(f"molcas_casscf_hessian: OpenMolcas failed (rc {proc.returncode}); "
                 f"log in {work / (project + '.log')}\n{tail}")
    return log, work / "scratch" / project


def _floats(s: str) -> list[float]:
    return [float(t) for t in s.replace("D", "E").split()]


def parse_log(log: str, natm: int) -> dict:
    """The numbers this compares and gates on, from OpenMolcas's output."""
    out: dict = {}
    # RASSCF's iteration table: the first row's energy is the CASCI on the
    # guess, the committed orbitals -- the import's own check
    rows = re.findall(r"^\s+(\d+)\s+\d+\s+\d+\s+\d+\s+(-\d+\.\d+)\s", log, flags=re.M)
    out["e_first"] = float(rows[0][1]) if rows else float("nan")
    m = re.findall(r"Convergence after\s+(\d+) iterations", log)
    out["iterations"] = int(m[-1]) if m else -1
    out["blb"] = float(re.findall(r"Maximum BLB matrix element\s+(\S+)", log)[-1])
    out["g_norm"] = float(re.findall(r"Norm of electronic gradient\s+(\S+)", log)[-1])
    conv = re.findall(r"Perturbation no:\s+(\d+) converged in\s+(\d+) steps", log)
    out["mclr_converged"] = len(conv)
    out["mclr_max_steps"] = max(int(n) for _p, n in conv) if conv else 0
    out["mclr_failed"] = "No convergence for perturbation" in log
    grad = []
    lines = log.splitlines()
    for k, line in enumerate(lines):
        if "Molecular gradients" in line and "no TR" not in line:
            j = k + 1
            while j < len(lines) and not re.match(r"\s*-{20,}", lines[j]):
                j += 1
            j += 1  # the X Y Z header
            while j < len(lines) and not re.match(r"\s*-{20,}", lines[j]):
                j += 1
            j += 1
            while len(grad) < natm and j < len(lines):
                parts = lines[j].split()
                if len(parts) >= 4:
                    grad.append(_floats(" ".join(parts[-3:])))
                j += 1
            break
    out["gradient"] = np.array(grad).ravel() if len(grad) == natm else None
    return out


def rasorb_energy(path: Path) -> float:
    """The CASSCF energy in the title line of RASSCF's natural-orbital file
    (``* RASSCF natural orbitals for root number  1 E=  <energy>``)."""
    m = re.search(r"root number\s+1\s+E=\s*(\S+)", path.read_text())
    if m is None:
        sys.exit(f"molcas_casscf_hessian: no root-1 energy in {path}")
    return float(m.group(1))


def read_unsym(path: Path, natm: int) -> tuple[np.ndarray, np.ndarray]:
    """MCLR's ``UNSYM`` file: the coordinates (bohr) and the Cartesian
    Hessian, in OpenMolcas's atom order."""
    text = path.read_text().splitlines()
    i = text.index("*BEGIN COORDINATES") + 2
    xyz = np.array([_floats(" ".join(text[i + a].split()[1:4])) for a in range(natm)])
    i = text.index("*BEGIN HESSIAN") + 2
    n = 3 * natm
    cols: list[list[float]] = []
    while text[i] != "*END HESSIAN":
        if text[i].startswith("* Column"):
            cols.append([])
        else:
            cols[-1] += _floats(text[i])
        i += 1
    H = np.array(cols).T
    assert H.shape == (n, n), f"{path}: Hessian is {H.shape}, want {(n, n)}"
    return xyz, H


def permute(H: np.ndarray, order: list[int]) -> np.ndarray:
    """OpenMolcas's atom order back to PySCF's."""
    idx = np.array([3 * a + x for a in order for x in range(3)])
    out = np.empty_like(H)
    out[np.ix_(idx, idx)] = H
    return out


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, choices=CASES)
    ap.add_argument("--pymolcas", help="the OpenMolcas driver (default $PYMOLCAS, then PATH)")
    ap.add_argument("--molcas-root", help="OpenMolcas's installed tree, for its basis "
                    "library (default $MOLCAS, then the wrapper's MOLCAS=, then /opt/openmolcas)")
    ap.add_argument("--mclr-residual", type=float,
                    help="MCLR's absolute PCG residual (default: MCLR_RESIDUAL's, per "
                    "case; it enters the tolerance)")
    ap.add_argument("--keep", type=Path,
                    help="run OpenMolcas in this directory and keep it (default: a temp dir)")
    args = ap.parse_args(argv)
    pymolcas = find_pymolcas(args.pymolcas)

    t0 = time.time()
    st = cnh.state_from_case(args.case)
    sk = ch.Skeleton(st.ref)
    natm, r = sk.natm, st.ref
    print(f"== {args.case} _casscf: {natm} atoms, nao {sk.nao}, ncore {r.ncore}, ncas "
          f"{r.ncas}, nelecas {st.nelecas}, E_CASSCF {st.e_casscf:.10f}")
    if sk.mol.cart:
        sys.exit("molcas_casscf_hessian: a Cartesian-basis state; the input assumes spherical")
    resp_path = ngrad.REPO / "golden" / f"{args.case}_casscf.respgold"
    _h, ra = gf.read_respgold(resp_path)
    H_comm, g_comm = ra["hessian"], ra["gradient"]

    # --- the PySCF side: the committed state's Hessian and the polished one's
    quiet: list[str] = []
    _de, _L, resp, H_c, ok_c = cnh.analyse(st, sk, log=quiet.append)
    d_file = float(np.abs(H_c - H_comm).max())
    print(f"-- PySCF (casscf_nuclear_hessian.py), {time.time() - t0:.1f} s")
    print(f"  committed state: identities {'hold' if ok_c else 'FAIL'}; max |H - "
          f"{resp_path.name}'s| = {d_file:.1e}; |g_orb| {resp.info['g_orb']:.2e}")
    stp, pinfo = cnh.polished(st, sk, resp.chord)
    skp = ch.Skeleton(stp.ref)
    dep, _Lp, respp, H_p, ok_p = cnh.analyse(stp, skp, log=quiet.append)
    lam = float(respp.info["eig_min_abs"])
    print(f"  polished state: |g_all| {pinfo['g_end']:.1e} after {pinfo['n_iter']} steps; "
          f"identities {'hold' if ok_p else 'FAIL'}; lambda_min(J) {lam:.3e}")
    ok = ok_c and ok_p and pinfo["converged"] and d_file <= cnh.IDENTITY_TOL

    # --- OpenMolcas, twice: from the committed orbitals and from the polished
    eps = args.mclr_residual or MCLR_RESIDUAL[args.case]
    text, order = molcas_input(sk.mol, r.basis, r.ncore, r.ncas, st.nelecas, eps)
    work = args.keep if args.keep else Path(tempfile.mkdtemp(prefix="molcas_hess_"))
    root = molcas_root(args.molcas_root, pymolcas)
    label = MOLCAS_BASIS[r.basis.lower()]
    print(f"-- OpenMolcas ({pymolcas}; basis library {root / 'basis_library'}), "
          f"MCLR residual {eps:g}, in {work}")
    runs: dict[str, np.ndarray] = {}
    g_ms: list[float] = []
    for name, C in (("committed", r.C0), ("polished", stp.ref.C0)):
        t1 = time.time()
        C_m, defect = molcas_orbitals(sk.mol, C, order, root, label)
        occ = np.zeros(C_m.shape[1])
        occ[:r.ncore] = 2.0
        occ[r.ncore:r.nocc] = sum(st.nelecas) / r.ncas
        sub = work / name
        sub.mkdir(parents=True, exist_ok=True)
        write_inporb(sub / GUESS, C_m, occ)
        log, wdir = run_molcas(pymolcas, text, sub)
        mo = parse_log(log, natm)
        mo["energy"] = rasorb_energy(sub / "hess.RasOrb.1")
        xyz_m, H_raw = read_unsym(wdir / "hess.UnSym", natm)
        runs[name] = permute(H_raw, order)
        g_ms.append(mo["g_norm"])
        d_xyz = float(np.abs(xyz_m - sk.mol.atom_coords()[order]).max())
        d_e = mo["energy"] - st.e_casscf
        print(f"  from the {name} orbitals ({time.time() - t1:.1f} s): overlap kept to "
              f"{defect:.1e} (allowed {PROJECTION_TOL:g}); geometry read back to {d_xyz:.1e} "
              f"bohr (allowed {COORD_TOL:g})")
        print(f"    RASSCF: first iteration (CASCI on the guess) {mo['e_first']:.8f}; "
              f"{mo['iterations']} iterations to max BLB {mo['blb']:.2e}, |electronic "
              f"gradient| g_M {mo['g_norm']:.2e}; E {mo['energy']:.12f}, - committed "
              f"{d_e:+.1e} (allowed {ENERGY_TOL:.0e})")
        print(f"    MCLR: {mo['mclr_converged']} perturbations converged, at most "
              f"{mo['mclr_max_steps']} PCG steps"
              + ("; A PERTURBATION DID NOT CONVERGE" if mo["mclr_failed"] else ""))
        ok &= defect <= PROJECTION_TOL and d_xyz <= COORD_TOL and abs(d_e) <= ENERGY_TOL
        ok &= not mo["mclr_failed"] and mo["mclr_converged"] >= 3 * natm
        if mo["gradient"] is not None:
            gm = mo["gradient"].reshape(natm, 3)
            gm_p = np.empty_like(gm)
            gm_p[order] = gm
            print(f"    gradient (ALASKA) vs PySCF, informational: max |dg| "
                  f"{np.abs(gm_p.ravel() - np.ravel(dep)).max():.1e} (polished), "
                  f"{np.abs(gm_p.ravel() - g_comm).max():.1e} (committed) Eh/bohr")
        H_m = runs[name]
        print(f"    its own identities: max |H - H^T| {np.abs(H_m - H_m.T).max():.1e}, "
              "translational rule "
              f"{np.abs(H_m.reshape(natm, 3, natm, 3).sum(axis=2)).max():.1e}")

    # --- the tolerance, from OpenMolcas's convergence
    kappa = float(np.abs(H_c - H_p).max()) / resp.info["g_orb"]
    spread = float(np.abs(runs["committed"] - runs["polished"]).max())
    theta_max = float(np.linalg.norm(respp.theta, axis=1).max())
    g_m = max(g_ms)
    t_state = max(spread, kappa * g_m)
    t_mclr = theta_max * eps
    tol = SAFETY * (t_state + t_mclr)
    print(f"-- tolerance: {SAFETY:g} x (state max(spread {spread:.1e}, {kappa:.2e} x "
          f"{g_m:.2e} = {kappa * g_m:.1e})  +  response {theta_max:.2e} x {eps:g} = "
          f"{t_mclr:.1e}) = {tol:.1e} Eh/bohr^2")

    # --- the comparison
    scale = float(np.abs(H_p).max())
    ev_p = np.linalg.eigvalsh(0.5 * (H_p + H_p.T))
    for name, H_m in runs.items():
        d_p = float(np.abs(H_m - H_p).max())
        d_c = float(np.abs(H_m - H_comm).max())
        print(f"  OpenMolcas from the {name} orbitals: max |H_M - H_polished| = {d_p:.2e} "
              f"(relative {d_p / scale:.1e}; allowed {tol:.1e}); max |H_M - H_committed| "
              f"= {d_c:.2e} (informational)")
        ok &= d_p <= tol
        ev_m = np.linalg.eigvalsh(0.5 * (H_m + H_m.T))
        print("    eigenvalues: " + " ".join(f"{e:+.6e}" for e in ev_m))
    print("  eigenvalues, polished PySCF: " + " ".join(f"{e:+.6e}" for e in ev_p))
    print("PASS" if ok else "FAIL")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
