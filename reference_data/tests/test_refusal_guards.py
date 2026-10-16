"""Hand cases for the guards that keep a noise-dependent state out of
``golden/``: ``_check_degeneracy``, ``_check_pc_conditioning`` (with the
``_pc_cut_gap`` it reads) and ``_check_gradient_invariants`` (both reference
kinds), plus the ``--gradgold`` flag combinations ``_parse_args`` refuses. Each
accepts a clean case and refuses one just across each threshold, on synthetic
inputs -- no SCF, no CASCI, no block2.

The thresholds are written out here as numbers, NOT read from the module's
constants: a test that compared against ``DEGENERACY_TOL`` would move with it,
and a loosened guard is exactly the regression these exist to catch. Moving a
threshold means editing this file too, in the same change, saying why.

What these cannot reach without a solve: whether the inputs the generator
hands the guards are the right ones -- ``_gradient_arrays`` building
``fock_rhf`` / ``f``, ``_block2_icmr`` recording the gaps -- and
``_check_casscf_stationary`` (it evaluates a CASSCF orbital gradient on the
state) and ``_block2_pc``'s block2-SC-equals-PySCF-SC gate (it runs both).
Those stay covered by regeneration (``devtools/golden-repro.sh``) and by the
refusals the generator is documented to make on real states
(docs/reference-data.md, "The degeneracy refusal" and "PC fields").
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import generate_golden as gg
import numpy as np
import pytest

# --- _check_degeneracy ------------------------------------------------------
#
# A real molecule (gto.M builds the basis and the symmetry orbitals; it solves
# nothing) and an orthonormal, symmetry-pure orbital set built straight from
# mol.symm_orb, so PySCF's own label_orb_symm labels each column -- the guard
# runs unmodified. N2/STO-3G in D2h has ten orbitals:
#   columns 0-2 Ag, 3 B2g, 4 B3g, 5-7 B1u, 8 B2u, 9 B3u.
AG, B2G, B3G, B1U, B2U, B3U = (0, 1, 2), 3, 4, (5, 6, 7), 8, 9

DEGENERACY_TOL = 1e-4  # generate_golden.DEGENERACY_TOL, written out on purpose


@pytest.fixture(scope="module")
def n2():
    from pyscf import gto  # a plain import: missing PySCF fails, never skips

    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", symmetry="D2h", verbose=0)
    s = mol.intor_symmetric("int1e_ovlp")
    blocks = []
    for c in mol.symm_orb:
        w, v = np.linalg.eigh(c.T @ s @ c)
        blocks.append(c @ v @ np.diag(w**-0.5) @ v.T)
    return mol, np.hstack(blocks)


def _pt(n2, order, energies, ncore, ncas):
    mol, mo = n2
    return SimpleNamespace(mol=mol, mo_coeff=mo[:, list(order)], ncore=ncore, ncas=ncas,
                           mo_energy=np.asarray(energies, dtype=float))


#: ncore=2, ncas=2, six virtuals. Orbitals 0,1 (core) are both Ag, and
#: orbitals 6 and 8 (virtual, NOT adjacent: the guard compares every pair, not
#: neighbours) are both B1u; 4,5 are B2g/B3g, a pi-like pair in two irreps.
DEGENERACY_ORDER = [AG[0], AG[1], AG[2], B1U[1], B2G, B3G, B1U[2], B2U, B1U[0], B3U]
DEGENERACY_ENERGIES = [-15.0, -14.0, -0.6, -0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def _degeneracy_case(*, core_gap=None, virt_gap=None, cross_irrep_virt=False,
                     active_degenerate=False):
    """Every gap at least 0.1 Eh except the one asked for."""
    e = list(DEGENERACY_ENERGIES)
    if core_gap is not None:
        e[1] = e[0] + core_gap
    if virt_gap is not None:
        e[8] = e[6] + virt_gap
    if cross_irrep_virt:
        e[5] = e[4]
    if active_degenerate:  # two Ag ACTIVE orbitals, exactly degenerate
        e[3] = e[2]
        return [AG[0], B1U[0], AG[1], AG[2], B2G, B3G, B1U[1], B2U, B1U[2], B3U], e
    return DEGENERACY_ORDER, e


def test_degeneracy_clean_case_accepted(n2):
    order, e = _degeneracy_case()
    gg._check_degeneracy(_pt(n2, order, e, 2, 2))


def test_degeneracy_threshold_value():
    """The constant itself, so a loosened one fails here even if every case
    below were rewritten around it."""
    assert gg.DEGENERACY_TOL == DEGENERACY_TOL


@pytest.mark.parametrize("gap, refused", [
    (0.0, True), (0.99 * DEGENERACY_TOL, True), (1.01 * DEGENERACY_TOL, False),
])
def test_degeneracy_core(n2, gap, refused):
    order, e = _degeneracy_case(core_gap=gap)
    pt = _pt(n2, order, e, 2, 2)
    if refused:
        with pytest.raises(SystemExit, match=r"core orbitals 0 and 1 are degenerate"):
            gg._check_degeneracy(pt)
    else:
        gg._check_degeneracy(pt)


@pytest.mark.parametrize("gap, refused", [
    (0.99 * DEGENERACY_TOL, True), (-0.99 * DEGENERACY_TOL, True),
    (1.01 * DEGENERACY_TOL, False), (-1.01 * DEGENERACY_TOL, False),
])
def test_degeneracy_virtual(n2, gap, refused):
    order, e = _degeneracy_case(virt_gap=gap)
    pt = _pt(n2, order, e, 2, 2)
    if refused:
        with pytest.raises(SystemExit, match=r"virtual orbitals 6 and 8 .* irrep B1u of D2h"):
            gg._check_degeneracy(pt)
    else:
        gg._check_degeneracy(pt)


def test_degeneracy_across_irreps_accepted(n2):
    """An exact degeneracy between two irreps (N2's pi pairs under D2h) is
    what symmetry pins: accepted."""
    order, e = _degeneracy_case(cross_irrep_virt=True)
    gg._check_degeneracy(_pt(n2, order, e, 2, 2))


def test_degeneracy_in_the_active_space_is_not_checked(n2):
    """Only the core and virtual pair classes are rotation-sensitive; an active
    degeneracy is the CI's business."""
    order, e = _degeneracy_case(active_degenerate=True)
    gg._check_degeneracy(_pt(n2, order, e, 2, 2))


# --- _check_pc_conditioning and _pc_cut_gap ---------------------------------

PC_MIN_GAP = 1e3  # generate_golden.PC_MIN_GAP, written out on purpose


def test_pc_threshold_value():
    assert gg.PC_MIN_GAP == PC_MIN_GAP


def _gaps(*values):
    """``_block2_icmr``'s gap records: (key, dim, gap, largest dropped,
    smallest kept)."""
    return [(gg.PC_SOLVE_ORDER[i], 4, g, 1e-16, 1e-16 * g) for i, g in enumerate(values)]


@pytest.mark.parametrize("gap, refused", [
    (1e6, False), (1.01 * PC_MIN_GAP, False), (0.99 * PC_MIN_GAP, True), (1.2, True),
])
def test_pc_conditioning_threshold(gap, refused):
    gaps = _gaps(1e8, gap, 1e9)
    if refused:
        with pytest.raises(SystemExit, match=rf"PC block {re.escape(gg.PC_SOLVE_ORDER[1])}: "):
            gg._check_pc_conditioning(gaps)
    else:
        gg._check_pc_conditioning(gaps)


def test_pc_conditioning_names_the_worst_block():
    key = re.escape(gg.PC_SOLVE_ORDER[2])
    with pytest.raises(SystemExit, match=rf"PC block {key}: .* 5\.0e\+02 gap"):
        gg._check_pc_conditioning(_gaps(900.0, 1e4, 500.0, 800.0))


def test_pc_conditioning_empty_block_accepted():
    """A block with no tuples (``ijrs-`` with one core orbital) reports an
    infinite gap and never refuses."""
    gg._check_pc_conditioning([("ijrs-", 4, *gg._pc_cut_gap(np.zeros((0, 4, 4))))])


def _diag_stack(*spectra):
    """Diagonal matrices: their singular values are exactly ``|spectrum|``."""
    return np.stack([np.diag(s) for s in spectra])


def test_pc_cut_gap_dropped_modes():
    """Two tuples. lstsq's cut is eps * d relative to s_max; the gap is the
    smallest kept over the largest dropped, minimised over the tuples."""
    d = 3
    a = _diag_stack([2.0, 2e-3, 2e-20], [1.0, 1e-5, 1e-22])
    gap, dropped, kept = gg._pc_cut_gap(a)
    assert np.finfo(float).eps * d > 1e-20  # both small modes are below the cut
    assert kept == pytest.approx(1e-5)
    assert dropped == pytest.approx(1e-20)
    assert gap == pytest.approx(min(1e-3 / 1e-20, 1e-5 / 1e-22))


def test_pc_cut_gap_nothing_dropped_uses_the_cut():
    a = _diag_stack([1.0, 1e-6, 1e-9])
    gap, dropped, kept = gg._pc_cut_gap(a)
    assert dropped == 0.0 and kept == pytest.approx(1e-9)
    assert gap == pytest.approx(1e-9 / (np.finfo(float).eps * 3))


def test_pc_cut_gap_continuum_is_refused():
    """A spectrum running straight through the cut -- CAS(12,12)'s case --
    has a gap near 1 and the guard refuses it."""
    cut = np.finfo(float).eps * 3
    a = _diag_stack([1.0, 1.5 * cut, 0.8 * cut])
    gap, _, _ = gg._pc_cut_gap(a)
    assert gap == pytest.approx(1.5 / 0.8)
    with pytest.raises(SystemExit):
        gg._check_pc_conditioning([("rabcpqg*", 3, *gg._pc_cut_gap(a))])


def test_pc_cut_gap_skips_all_zero_tuples():
    a = _diag_stack([0.0, 0.0, 0.0], [1.0, 1e-3, 1e-20])
    assert gg._pc_cut_gap(a) == gg._pc_cut_gap(a[1:])
    assert gg._pc_cut_gap(_diag_stack([0.0, 0.0]))[0] == np.inf


# --- _check_gradient_invariants ---------------------------------------------
#
# ncore=1, nact=3, nvirt=2; 4 RHF electrons, so act_O holds 1 orbital.
O1_FOCK_TOL = 1e-5   # generate_golden.O1_FOCK_TOL, written out on purpose
O2_FOCK_TOL = 1e-10  # generate_golden.O2_FOCK_TOL

NCORE, NACT = 1, 3
C, A, V = slice(0, 1), slice(1, 4), slice(4, 6)
MO_ENERGY_RHF = np.array([-2.0, -0.5, 0.3, 0.6, 1.0, 1.5])  # HOMO -0.5, LUMO 0.3
NELEC_RHF = 4


def test_gradient_threshold_values():
    assert (gg.O1_FOCK_TOL, gg.O2_FOCK_TOL) == (O1_FOCK_TOL, O2_FOCK_TOL)


def _rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _grad_case(act_eigs=(-0.5, 0.3, 0.6)):
    """A canonical RHF Fock, its active block rotated (allowed: only the
    off-blocks must vanish), and a generalized Fock already pseudocanonical in
    core and virtual, its diagonals the golden's e_core / e_virt."""
    fock = np.diag(MO_ENERGY_RHF).copy()
    r = _rotation(0.3)
    fock[A, A] = r @ np.diag(act_eigs) @ r.T
    f = np.diag([-1.9, 0.0, 0.0, 0.0, 1.1, 1.6])
    f[A, A] = [[0.1, 0.02, 0.0], [0.02, 0.2, 0.01], [0.0, 0.01, 0.3]]  # free
    pt = SimpleNamespace(ncore=NCORE, ncas=NACT,
                         mo_energy=np.array([-1.9, 9.0, 9.0, 9.0, 1.1, 1.6]))
    return pt, {"fock_rhf": fock, "f": f}


def _check(pt, grad):
    return gg._check_gradient_invariants(pt, grad, NELEC_RHF, MO_ENERGY_RHF)


def test_gradient_clean_case_accepted():
    res = _check(*_grad_case())
    assert res["o1"] == 0.0 and res["act_occ"] == 1
    assert res["o2"] <= 1e-15
    assert (res["homo"], res["lumo"]) == (-0.5, 0.3)


@pytest.mark.parametrize("block", ["core-act", "core-virt", "act-virt"])
@pytest.mark.parametrize("size, refused", [(0.99 * O1_FOCK_TOL, False),
                                           (1.01 * O1_FOCK_TOL, True)])
def test_gradient_o1(block, size, refused):
    pt, grad = _grad_case()
    rows, cols = {"core-act": (0, 2), "core-virt": (0, 5), "act-virt": (3, 4)}[block]
    grad["fock_rhf"][rows, cols] = grad["fock_rhf"][cols, rows] = -size
    if refused:
        with pytest.raises(SystemExit, match=r"\(O1\) violated"):
            _check(pt, grad)
    else:
        assert _check(pt, grad)["o1_each"][block] == pytest.approx(size)


@pytest.mark.parametrize("where", ["virt-offdiag", "virt-diag", "core-diag"])
@pytest.mark.parametrize("size, refused", [(0.9 * O2_FOCK_TOL, False),
                                           (1.1 * O2_FOCK_TOL, True)])
def test_gradient_o2(where, size, refused):
    pt, grad = _grad_case()
    f = grad["f"]
    if where == "virt-offdiag":
        f[4, 5] += size
    elif where == "virt-diag":
        f[5, 5] += size
    else:
        f[0, 0] -= size
    if refused:
        with pytest.raises(SystemExit, match=r"\(O2\) violated"):
            _check(pt, grad)
    else:
        _check(pt, grad)


@pytest.mark.parametrize("act_eigs, refused", [
    ((-0.5 + 0.99 * O1_FOCK_TOL, 0.3, 0.6), False),  # at the HOMO, within tolerance
    ((-0.5 + 1.01 * O1_FOCK_TOL, 0.3, 0.6), True),   # just inside the gap
    ((-0.5, 0.0, 0.6), True),                        # one squarely in the gap
    ((-0.7, -0.6, 0.6), True),                       # act_O would hold two
    ((0.3, 0.4, 0.6), True),                         # act_O would hold none
])
def test_gradient_o3(act_eigs, refused):
    pt, grad = _grad_case(act_eigs)
    if refused:
        with pytest.raises(SystemExit, match=r"\(O3\) violated"):
            _check(pt, grad)
    else:
        assert _check(pt, grad)["act_occ"] == 1


# --- _check_gradient_invariants on a CASSCF-reference sidecar (#120) ---------
#
# --mo-from --gradgold: (O2) is still asserted, (O1) and (O3) are not -- they
# say the spaces are spanned by RHF canonical orbitals, which converged CASSCF
# orbitals are not. (O1) is still measured and returned.


def _check_casscf(pt, grad):
    return gg._check_gradient_invariants(pt, grad, NELEC_RHF, MO_ENERGY_RHF,
                                         reference_kind="casscf")


def test_gradient_casscf_skips_o1_and_o3():
    """An RHF Fock far off block-diagonal (an act-virt element of 0.28, the
    committed CAS(4,4) CASSCF sidecar's own size) and an active block with an
    eigenvalue squarely in the RHF gap: refused as casci, accepted as casscf,
    with (O1) reported."""
    pt, grad = _grad_case((-0.5, 0.0, 0.6))
    grad["fock_rhf"][3, 4] = grad["fock_rhf"][4, 3] = 0.28
    with pytest.raises(SystemExit, match=r"\(O1\) violated"):
        _check(pt, grad)
    res = _check_casscf(pt, grad)
    assert res["o1_each"]["act-virt"] == pytest.approx(0.28)
    assert res["act_occ"] is None


@pytest.mark.parametrize("size, refused", [(0.9 * O2_FOCK_TOL, False),
                                           (1.1 * O2_FOCK_TOL, True)])
def test_gradient_casscf_still_asserts_o2(size, refused):
    pt, grad = _grad_case()
    grad["f"][4, 5] += size
    if refused:
        with pytest.raises(SystemExit, match=r"\(O2\) violated"):
            _check_casscf(pt, grad)
    else:
        _check_casscf(pt, grad)


def test_gradient_unknown_reference_kind_refused():
    with pytest.raises(SystemExit, match=r"unknown reference_kind"):
        gg._check_gradient_invariants(*_grad_case(), NELEC_RHF, MO_ENERGY_RHF,
                                      reference_kind="rhf")


# --- the --gradgold flag combinations (#120) ---------------------------------
#
# Parsed only: _parse_args applies every refusal and solves nothing.


def test_gradgold_with_mo_from_permitted():
    args = gg._parse_args(["--ncas", "4", "--nelecas", "4", "--mo-from", "x.mogold",
                           "--gradgold"])
    assert args.gradgold and args.mo_from == "x.mogold"


@pytest.mark.parametrize("argv", [
    ["--casscf", "--mo-out", "x.mogold", "--gradgold"],  # writes no golden to sit beside
    ["--df", "--gradgold"],                              # conventional integrals only
    ["--df", "--mo-from", "x.mogold", "--gradgold"],
])
def test_gradgold_refused_combinations(argv, capsys):
    with pytest.raises(SystemExit) as e:
        gg._parse_args(argv)
    assert e.value.code == 2
    assert "--gradgold" in capsys.readouterr().err
