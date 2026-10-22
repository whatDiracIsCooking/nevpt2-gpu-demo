#!/usr/bin/env python3
r"""The **integral perturbation sidecar** (``*.pertgold``, #220), offline: the
first derivative of a ``_casscf`` state's MO integrals along every nuclear
coordinate, on the frozen orthonormal path, written in the gradient sidecar's
general-range inventory -- what ``nevpt2_sc_pseudodensity
--sc-source-tangent`` differentiates the SC source terms ``Y`` and ``y``
along (docs/hessian-theory.md, section 8: the NEVPT2 half of ``L_{R Theta}``).

WHAT IS WRITTEN. For each Cartesian coordinate ``k = 3 atom + axis`` the
perturbed Hamiltonian of ``casscf_hessian.py`` (#215) --
:meth:`casscf_hessian.Skeleton.perturbed_h1` and ``perturbed_eri``, skeleton
plus connection, ``dC/dR = C0 t^k`` -- pushed through the definitions the
golden and its gradient sidecar are written with, at **fixed orbitals and
fixed active density**: ``h``; ``h^eff = h + sum_j [2 (pq|jj) - (pj|jq)]``;
the generalized Fock ``f = h^eff + sum_tu gamma_tu [(pq|tu) - (1/2)(pt|qu)]``
at the golden's ``dm1``; and the eighteen one-general-index rows ``g_xP_QR``
(``generate_golden._gradient_arrays``). Each is LINEAR in ``(h, (pq|rs))``,
so its derivative is the same definition evaluated at the perturbed
Hamiltonian, which is how it is computed (:func:`inventory`). The orbital
energies ``e_core`` / ``e_virt`` are ``f``'s diagonal, and every block the
golden stores is a slice of these, so neither is written again: the C++
reader derives them (``integralPerturbationDevice``).

A CONVENTIONAL CASE NEEDS THE FULL TENSOR, the same data question as #180:
the connection term promotes one index of every integral, ``(pq|rs)^(k)``
reading ``(tq|rs)`` with ``t`` general, so a one-general-index row of the
derivative is built from two-general-index integrals. :func:`full_eri` forms
the whole ``n_mo^4`` tensor, which the committed cases afford; the sidecar
itself is the one-general-index inventory and nothing more.

DENSITY-FITTED (``--df``): the perturbation of the FITTED integrals
``(pq|rs) = sum_PQ (pq|P) [J^-1]_PQ (Q|rs)``, the metric's derivative and the
auxiliary functions' included (:class:`DfSkeleton`), in the same four-index
inventory. Four-index because that form is gauge-free: the golden's ``B`` is
a Cholesky factor of the metric, whose own derivative is a choice, while
``sum_L B B`` is not. ``h`` is the conventional one-electron derivative.

THE CHECKS, before anything is written:

* the frame: the state is the committed golden's (its ``ci``) and gradient
  sidecar's (its ``mo_coeff``), bit for bit;
* the definitions: :func:`inventory` at the reference integrals rebuilds the
  committed gradient sidecar's arrays (the fitted sidecar carries only the
  three matrices and four tied rows of them);
* the derivative: per coordinate, the inventory at the displaced geometries
  ``R0 +- h e_k`` (orbitals on the frozen path, integrals rebuilt from
  scratch -- fitted ones with the auxiliary basis moved) differenced and
  Richardson-verified against the analytic arrays, every array at once
  (``casscf_hessian.judge_column``'s run rule), unless ``--no-ladders``.

Usage::

    uv run python reference_data/integral_perturbation.py --case n2_ccpvdz_cas44 \
        --write golden/n2_ccpvdz_cas44_casscf.pertgold
    uv run python reference_data/integral_perturbation.py --case n2_ccpvdz_cas44 --df \
        --write golden/n2_ccpvdz_cas44_casscf_df.pertgold

The state is ``generate_golden._build_state``'s ``--mo-from`` one, every gate
included, in the golden's canonicalised frame (``pt``'s orbitals, as
``casscf_nuclear_hessian.state_from_case``). The measured numbers are
docs/reference-data.md, "The integral perturbation sidecar".
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads
# (docs/reference-data.md, "Reproducibility").
import generate_golden as gg

# isort: split
import casscf_hessian as ch
import nevpt2_grad as ngr
import nevpt2_numgrad as ngrad

# isort: split
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import golden_format as gf
import numpy as np
import scipy.linalg
from pyscf import ao2mo, df, scf

#: The derivative ladders: central differences of the integrals along each
#: coordinate, ``h = 2^-k`` bohr. Their round-off grows as ``1/h`` only, so
#: they run as far down as the gradient columns of ``casscf_hessian.py``.
LADDER = (3, 13)

#: The definitions' rebuild of the committed gradient sidecar: two transforms
#: of the same integrals, reassociation only.
REBUILD_TOL = 1e-10


@dataclass
class State:
    """The committed ``_casscf`` state in its golden's frame."""

    case: str
    df: bool
    ref: ch.Reference
    auxbasis: str | None
    gamma: np.ndarray  # the golden's dm1, the active density f is built at
    golden: Path
    gradgold: Path


def golden_name(case: str, fitted: bool) -> str:
    return f"{case}_casscf" + ("_df" if fitted else "")


def state_from_case(case: str, fitted: bool) -> State:
    """``_build_state``'s ``--mo-from`` state (fitted on the case's default
    ``<basis>-jkfit`` with ``--df``, as ``generate_golden.py --df`` builds
    it), in the frame its golden is written in."""
    spec = ngrad.CASES[case]
    name = golden_name(case, fitted)
    mo_from = ngrad.REPO / "golden" / f"{name}.mogold"
    geom = ngrad.read_xyz(ngrad.REPO / spec["geometry"])
    auxbasis = f"{spec['basis']}-jkfit" if fitted else None
    _mc, pt = gg._build_state(geom, spec["basis"], spec["ncas"], spec["nelecas"], auxbasis,
                              cas_irreps=spec.get("cas_irreps"),
                              core_irreps=spec.get("core_irreps"), mo_from=mo_from)
    golden = ngrad.REPO / "golden" / f"{name}.nevpt2gold"
    _h, ga = gf.read_golden(golden, want={"ci", "dm1"})
    ci = np.asarray(pt.ci)
    if not np.array_equal(ga["ci"].reshape(ci.shape), ci):
        raise SystemExit(f"{golden.name}'s ci is not the state's: the frame is not the golden's")
    D, G = ch.full_rdms(pt.ncore, pt.ncas, pt.nelecas, ci)
    ref = ch.Reference(geom=geom, basis=spec["basis"], C0=np.asarray(pt.mo_coeff).copy(),
                       ncore=pt.ncore, ncas=pt.ncas, D=D, G=G)
    return State(case=case, df=fitted, ref=ref, auxbasis=auxbasis, gamma=ga["dm1"],
                 golden=golden, gradgold=golden.with_suffix(".gradgold"))


# --------------------------------------------------------------------------
# The definitions: the gradient sidecar's arrays from (h, (pq|rs))
# --------------------------------------------------------------------------


def inventory(h: np.ndarray, g: np.ndarray, gamma: np.ndarray, ncore: int,
              nact: int) -> dict[str, np.ndarray]:
    """The gradient sidecar's general-range inventory (its three matrices and
    eighteen rows, :data:`golden_format.PERTGOLD_ARRAYS`' names) of the
    one-electron matrix ``h`` and the full chemists'-order tensor ``g``, at
    the active density ``gamma``. Linear in ``(h, g)``: handed a perturbed
    Hamiltonian it returns that Hamiltonian's dotted inventory."""
    nocc = ncore + nact
    c, a, v = slice(0, ncore), slice(ncore, nocc), slice(nocc, None)
    heff = (h + 2.0 * np.einsum("pqjj->pq", g[:, :, c, c])
            - np.einsum("pjjq->pq", g[:, c, c, :]))
    f = (heff + np.einsum("tu,pqtu->pq", gamma, g[:, :, a, a])
         - 0.5 * np.einsum("tu,ptqu->pq", gamma, g[:, a, :, a]))
    return dict(
        h=h, heff=heff, f=f,
        g_xa_aa=g[:, a, a, a], g_xv_aa=g[:, v, a, a], g_xc_aa=g[:, c, a, a],
        g_xa_va=g[:, a, v, a], g_xv_va=g[:, v, v, a], g_xc_va=g[:, c, v, a],
        g_xa_ca=g[:, a, c, a], g_xv_ca=g[:, v, c, a], g_xc_ca=g[:, c, c, a],
        g_xa_cv=g[:, a, c, v], g_xv_cv=g[:, v, c, v], g_xc_cv=g[:, c, c, v],
        g_xa_cc=g[:, a, c, c], g_xv_cc=g[:, v, c, c],
        # (x i | j j) and (x j | j i)
        g_xc_cc_j=np.einsum("xijj->xij", g[:, c, c, c]),
        g_xc_cc_k=np.einsum("xjji->xji", g[:, c, c, c]),
        # (x t | r r) and (x i | r r)
        g_xa_vv=np.einsum("xtrr->xtr", g[:, a, v, v]),
        g_xc_vv=np.einsum("xirr->xir", g[:, c, v, v]),
    )


#: The inventory's names, in the sidecar's order (mo_coeff aside).
NAMES = tuple(n for n, _s in gf.PERTGOLD_ARRAYS if n != "mo_coeff")


# --------------------------------------------------------------------------
# The integrals and their derivatives
# --------------------------------------------------------------------------


def _metric_solve(j2: np.ndarray):
    """``x -> J^-1 x`` over the auxiliary axis (the first), through the
    metric's Cholesky factor: PySCF's own route to its ``_cderi``, and the one
    whose round-off does not grow with the metric's condition number the way
    an explicit inverse's does."""
    factor = scipy.linalg.cho_factor(j2)

    def solve(x: np.ndarray) -> np.ndarray:
        flat = x.reshape(x.shape[0], -1)
        return scipy.linalg.cho_solve(factor, flat).reshape(x.shape)

    return solve


def fitted_tensor(mol, auxbasis: str, C: np.ndarray) -> np.ndarray:
    """The fitted ``(pq|rs) = sum_PQ (pq|P) [J^-1]_PQ (Q|rs)`` over the whole
    MO range of ``C``."""
    aux = df.addons.make_auxmol(mol, auxbasis)
    j3 = df.incore.aux_e2(mol, aux, "int3c2e", aosym="s1")  # (mu nu | P)
    t = np.einsum("mnP,mp,nq->Ppq", j3, C, C, optimize=True)
    u = _metric_solve(aux.intor("int2c2e"))(t)
    return np.einsum("Ppq,Prs->pqrs", t, u, optimize=True)


def full_eri(mol, C: np.ndarray, auxbasis: str | None) -> np.ndarray:
    """The whole MO tensor ``(pq|rs)`` of ``C`` (#180's data question:
    module docstring), conventional or fitted."""
    if auxbasis is not None:
        return fitted_tensor(mol, auxbasis, C)
    n = C.shape[1]
    return ao2mo.restore(1, ao2mo.kernel(mol, C), n).reshape(n, n, n, n)


class DfSkeleton(ch.Skeleton):
    """:class:`casscf_hessian.Skeleton` for the FITTED integrals: the same
    frozen path, connection and one-electron derivative, with every
    two-electron integral the fitted one and its skeleton derivative the
    fitted one's -- the three-centre integrals' (orbital and auxiliary
    functions on the moving atom) and the metric's."""

    def __init__(self, ref: ch.Reference, auxbasis: str):
        self.auxbasis = auxbasis
        self._fit: dict | None = None
        super().__init__(ref)

    def _fitting(self) -> dict:
        if self._fit is None:
            mol = self.mol
            aux = df.addons.make_auxmol(mol, self.auxbasis)
            C = self.ref.C0
            j3 = df.incore.aux_e2(mol, aux, "int3c2e", aosym="s1")
            solve = _metric_solve(aux.intor("int2c2e"))
            t = np.einsum("mnP,mp,nq->Ppq", j3, C, C, optimize=True)
            self._fit = dict(
                aux=aux, solve=solve, t=t, u=solve(t),
                ip1=df.incore.aux_e2(mol, aux, "int3c2e_ip1", aosym="s1", comp=3),
                ip2=df.incore.aux_e2(mol, aux, "int3c2e_ip2", aosym="s1", comp=3),
                j2ip=aux.intor("int2c2e_ip1", comp=3),
            )
        return self._fit

    def mo_eri(self, *ranges: slice) -> np.ndarray:
        key = tuple((r.start, r.stop) for r in ranges)
        got = self._mo_cache.get(key)
        if got is None:
            f = self._fitting()
            t1 = f["t"][:, ranges[0], ranges[1]]
            u2 = f["u"][:, ranges[2], ranges[3]]
            got = np.einsum("Ppq,Prs->pqrs", t1, u2, optimize=True)
            self._mo_cache[key] = got
        return got

    def eri1_skel(self, ia: int, *ranges: slice) -> np.ndarray:
        """``d(pq|rs)_fit / dR_{ia,x}`` at fixed coefficients, ``(3, n1..n4)``:
        each factor's nuclear derivative, ``d chi / dR_A = - nabla chi`` for a
        function on atom ``A`` (orbital or auxiliary)."""
        f = self._fitting()
        C = self.ref.C0
        nao = self.nao
        _s0, _s1, p0, p1 = self.aoslices[ia]
        aux = f["aux"]
        q0, q1 = aux.aoslice_by_atom()[ia][2:]
        out = []
        for x in range(3):
            d3 = np.zeros((nao, nao, aux.nao_nr()))
            d3[p0:p1] -= f["ip1"][x, p0:p1]
            d3[:, p0:p1] -= f["ip1"][x, p0:p1].transpose(1, 0, 2)
            d3[:, :, q0:q1] -= f["ip2"][x, :, :, q0:q1]
            naux = aux.nao_nr()
            dj = np.zeros((naux, naux))
            dj[q0:q1] -= f["j2ip"][x, q0:q1]
            dj[:, q0:q1] -= f["j2ip"][x, q0:q1].T
            # (pq|P)' J^-1 (Q|rs) + (pq|P) J^-1 (Q|rs)' - (pq|P) J^-1 J' J^-1 (Q|rs)
            dt = np.einsum("mnP,mp,nq->Ppq", d3, C, C, optimize=True)
            du = f["solve"](dt)
            t, u = f["t"], f["u"]
            r1, r2, r3, r4 = ranges
            val = (np.einsum("Ppq,Prs->pqrs", dt[:, r1, r2], u[:, r3, r4], optimize=True)
                   + np.einsum("Ppq,Prs->pqrs", t[:, r1, r2], du[:, r3, r4], optimize=True)
                   - np.einsum("Ppq,PQ,Qrs->pqrs", u[:, r1, r2], dj, u[:, r3, r4],
                               optimize=True))
            out.append(val)
        return np.array(out)


def symmetrized(g: np.ndarray) -> np.ndarray:
    """``g`` averaged over the eight permutations a real ``(pq|rs)`` is
    invariant under. The derivative has them exactly, but its terms reach
    them only up to round-off -- the fitted one's through a solve with the
    metric, whose condition number amplifies it -- and every row of the
    inventory reads a different one of the eight, so an unsymmetrized
    derivative would put that round-off between rows the consumer's
    identities hold equal."""
    g = 0.5 * (g + g.transpose(1, 0, 2, 3))
    g = 0.5 * (g + g.transpose(0, 1, 3, 2))
    return 0.5 * (g + g.transpose(2, 3, 0, 1))


def skeleton_of(st: State) -> ch.Skeleton:
    return DfSkeleton(st.ref, st.auxbasis) if st.df else ch.Skeleton(st.ref)


def perturbations(st: State, sk: ch.Skeleton) -> dict[str, np.ndarray]:
    """Every coordinate's dotted inventory, stacked on a leading ``3 natm``
    axis in :data:`NAMES`' order."""
    allmo = slice(None)
    t1, _ = sk.connection(second=False)
    h1 = sk.hcore1()
    out: dict[str, list] = {n: [] for n in NAMES}
    for ia in range(sk.natm):
        skel = sk.eri1_skel(ia, allmo, allmo, allmo, allmo)
        for x in range(3):
            k = 3 * ia + x
            hk = sk.perturbed_h1(k, t1, h1)
            gk = sk.perturbed_eri(k, (allmo,) * 4, t1, skel[x])
            inv = inventory(0.5 * (hk + hk.T), symmetrized(gk), st.gamma, st.ref.ncore,
                            st.ref.ncas)
            for n in NAMES:
                out[n].append(inv[n])
    return {n: np.array(v) for n, v in out.items()}


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------


def check_definitions(st: State, sk: ch.Skeleton) -> float:
    """:func:`inventory` at the reference integrals against the committed
    gradient sidecar, every array it carries of the inventory (a fitted one
    carries the matrices and the four tied rows): the worst max-abs gap."""
    allmo = slice(None)
    base = inventory(sk.h_mo, sk.mo_eri(allmo, allmo, allmo, allmo), st.gamma,
                     st.ref.ncore, st.ref.ncas)
    _s, sa = gf.read_gradgold(st.gradgold)
    if not np.array_equal(sa["mo_coeff"], st.ref.C0):
        raise SystemExit(f"{st.gradgold.name}'s mo_coeff is not the state's: the frame is "
                         "not the golden's")
    worst = 0.0
    for n in NAMES:
        if n in sa:
            worst = max(worst, float(np.abs(base[n] - sa[n]).max()))
    return worst


def displaced_inventory(st: State, S0_half: np.ndarray, dvec: np.ndarray) -> np.ndarray:
    """The inventory at ``R0 + dvec`` on the frozen path, every array
    flattened into one vector in :data:`NAMES`' order."""
    geom = ngrad.displace_vec(st.ref.geom, dvec)
    mol = ch.mol_at(geom, st.ref.basis)
    C = ngr.frozen_coeff(st.ref.C0, S0_half, mol.intor("int1e_ovlp"))
    h = C.T @ scf.hf.get_hcore(mol) @ C
    inv = inventory(h, full_eri(mol, C, st.auxbasis), st.gamma, st.ref.ncore, st.ref.ncas)
    return np.concatenate([inv[n].ravel() for n in NAMES])


def check_ladders(st: State, sk: ch.Skeleton, dot: dict[str, np.ndarray]) -> list:
    """Per coordinate, the central differences of the displaced inventory
    against the analytic one, every array at once."""
    steps = ngr.ladder_steps(LADDER)
    checks = []
    for k in range(sk.ncoord):
        ref = np.concatenate([dot[n][k].ravel() for n in NAMES])
        fd = []
        for h in steps:
            plus = displaced_inventory(st, sk.S0_half, h * ch.unit(sk.natm, k))
            minus = displaced_inventory(st, sk.S0_half, -h * ch.unit(sk.natm, k))
            fd.append((plus - minus) / (2.0 * h))
        checks.append(ch.judge_column(f"d(inventory)/dR[{k // 3},{'xyz'[k % 3]}]",
                                      np.array(fd), steps, ref))
    return checks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, choices=sorted(ngrad.CASES))
    ap.add_argument("--df", action="store_true",
                    help="the density-fitted twin's state and its fitted integrals")
    ap.add_argument("--no-ladders", action="store_true",
                    help="skip the derivative ladders (the definitions and the frame are "
                    "still checked)")
    ap.add_argument("--write", type=Path, help="write the .pertgold here")
    args = ap.parse_args(argv)

    t0 = time.time()
    st = state_from_case(args.case, args.df)
    sk = skeleton_of(st)
    print(f"== {golden_name(args.case, args.df)}: {sk.natm} atoms, nmo {sk.nmo}, ncore "
          f"{st.ref.ncore}, nact {st.ref.ncas}"
          + (f", fitted on {st.auxbasis}" if st.df else "")
          + f"  (state built in {time.time() - t0:.1f} s)")
    gap = check_definitions(st, sk)
    ok = gap <= REBUILD_TOL
    print("  the frame: the golden's ci and the gradient sidecar's mo_coeff, bit for bit")
    print(f"  the definitions at the reference integrals vs {st.gradgold.name}: max |delta| = "
          f"{gap:.2e} (<= {REBUILD_TOL:g}: {'ok' if ok else 'FAILED'})")
    t0 = time.time()
    dot = perturbations(st, sk)
    print(f"  {sk.ncoord} perturbations in {time.time() - t0:.1f} s; max |dot| per coordinate: "
          + " ".join(f"{max(float(np.abs(dot[n][k]).max()) for n in NAMES):.3e}"
                     for k in range(sk.ncoord)))
    if not args.no_ladders:
        t0 = time.time()
        checks = check_ladders(st, sk, dot)
        worst = max(c.numbers["err"] for c in checks)
        for c in checks:
            print(f"  {'ok  ' if c.ok else 'FAIL'} {c.name}: {c.summary}")
            if not c.ok:
                print("\n".join(c.table))
            ok &= c.ok
        print(f"  {sum(c.ok for c in checks)}/{len(checks)} ladders pass in "
              f"{time.time() - t0:.1f} s; worst |Richardson - analytic| = {worst:.2e}")
    if args.write is not None:
        if not ok:
            print("FAIL: nothing written")
            return 1
        _s, sa = gf.read_gradgold(st.gradgold)
        header = dict(natm=sk.natm, nao=sk.nao, nmo=sk.nmo, ncore=st.ref.ncore,
                      nact=st.ref.ncas, nvirt=sk.nmo - st.ref.ncore - st.ref.ncas,
                      naux=_s.get("naux", 0))
        gf.write_pertgold(args.write, header=header, arrays=dict(mo_coeff=sa["mo_coeff"], **dot))
        print(f"  wrote {args.write} ({args.write.stat().st_size} B)")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
