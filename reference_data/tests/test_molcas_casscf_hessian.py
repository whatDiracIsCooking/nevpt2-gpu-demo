"""Hand cases for ``molcas_casscf_hessian.py``'s OpenMolcas plumbing, none of
which needs OpenMolcas: reading a basis-library entry, carrying orbitals into
OpenMolcas's basis and AO order and back, the orbital file it writes, reading
MCLR's ``UNSYM`` Hessian, and the atom-order map. The cross-check itself
needs the program (docs/reference-data.md, "The CASSCF Hessian against
OpenMolcas").

Tolerances are written out here, not read from the module.
"""

from __future__ import annotations

import molcas_casscf_hessian as mch
import numpy as np
import pytest
from pyscf import gto, scf

# A library entry in OpenMolcas's layout: label, two reference lines,
# "charge lmax", then per l a comment, "nprim ncontr", the exponents (here
# one in the library's own e-4 spelling) and the coefficient rows.
H_ENTRY = """\
/H.toy-basis.Nobody.3s1p.2s1p.
a reference line
another reference line
     1.0   1
* s-type functions
    3    2
       13.0100000000
        1.9620000000
     1220.0000000e-4
  0.0196850000000000  0.0000000000000000
  0.1379770000000000  0.0000000000000000
  0.4781480000000000  1.0000000000000000
* p-type functions
    1    1
        0.7270000000
  1.0000000000000000

/He.toy-basis.Nobody.1s.1s.
x
y
     2.0   0
* s-type functions
    1    1
        1.0
  1.0
"""

# O in the same layout: s (3 contracted from 4), p (2 from 3), d (1).
O_ENTRY = """\
/O.toy-basis.Nobody.4s3p1d.3s2p1d.
r
r
     8.0   2
* s-type functions
    4    3
     5484.6717
      825.23495
       40.775539
        0.2700058
  0.0018311   -0.0001000  0.0
  0.0139501   -0.0010000  0.0
  0.4456860    0.3000000  0.0
  0.0000000    0.6000000  1.0
* p-type functions
    3    2
       15.539616
        1.2087700
        0.2700058
  0.0708743  0.0
  0.4568800  0.0
  0.4000000  1.0
* d-type functions
    1    1
        0.8000000
  1.0
"""


@pytest.fixture
def root(tmp_path):
    lib = tmp_path / "basis_library"
    lib.mkdir()
    (lib / "TOY-BASIS").write_text(H_ENTRY + "\n" + O_ENTRY)
    return tmp_path


def test_basis_entry_parses(root):
    shells = mch.molcas_basis(root, "toy-basis", "H")
    assert [s[0] for s in shells] == [0, 1]
    s = np.array(shells[0][1:])
    assert s.shape == (3, 3)  # three primitives: exponent + two contractions
    np.testing.assert_allclose(s[:, 0], [13.01, 1.962, 0.122])
    np.testing.assert_allclose(s[2, 1:], [0.478148, 1.0])
    assert shells[1][1:] == [[0.727, 1.0]]
    # the next entry's label ends this one
    assert len(mch.molcas_basis(root, "toy-basis", "He")) == 1


def test_missing_entry_refused(root):
    with pytest.raises(SystemExit):
        mch.molcas_basis(root, "toy-basis", "Li")


def _water(root):
    """Water with PySCF's own copy of the toy basis, deliberately segmented
    differently from the library's general contraction (the cc-pVDZ
    situation): same span, different functions."""
    basis = {"O": mch.molcas_basis(root, "toy-basis", "O"),
             "H": mch.molcas_basis(root, "toy-basis", "H")}
    # re-segment: split each generally contracted shell into its columns
    seg = {el: [[sh[0]] + [[p[0], p[1 + c]] for p in sh[1:] if p[1 + c] != 0.0]
                for sh in shells for c in range(len(sh[1]) - 1)]
           for el, shells in basis.items()}
    # H first, so the element grouping reorders the atoms
    return gto.M(atom=[("H", (0.0, 0.76, -0.47)), ("O", (0.0, 0.05, 0.11)),
                       ("H", (0.0, -0.77, -0.45))], basis=seg, verbose=0)


def test_ao_order_is_molcas(root):
    mol = _water(root)
    order = [1, 0, 2]
    mm, perm = mch.molcas_mol(mol, order, root, "toy-basis")
    labels = [mm.ao_labels()[p].split(None, 2)[2].strip() for p in perm[:14]]
    # centre, then l, then the component, then the contracted function:
    # OpenMolcas's own printout order (S1 2px, 3px, *px, 2py, ...)
    assert labels == ["1s", "2s", "3s", "2px", "3px", "2py", "3py", "2pz", "3pz",
                      "3dxy", "3dyz", "3dz^2", "3dxz", "3dx2-y2"]
    assert sorted(perm) == list(range(mm.nao))


def test_orbitals_round_trip(root):
    mol = _water(root)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    order = [1, 0, 2]
    C_m, defect = mch.molcas_orbitals(mol, mf.mo_coeff, order, root, "toy-basis")
    assert defect < 1e-10
    mm, perm = mch.molcas_mol(mol, order, root, "toy-basis")
    back = np.empty_like(C_m)
    back[perm] = C_m
    S = mol.intor("int1e_ovlp")
    C_p = np.linalg.solve(S, gto.intor_cross("int1e_ovlp", mol, mm) @ back)
    np.testing.assert_allclose(C_p, mf.mo_coeff, atol=1e-10)


def test_inporb_layout(tmp_path):
    C = np.arange(12.0).reshape(4, 3) - 5.5
    mch.write_inporb(tmp_path / "x.InpOrb", C, np.array([2.0, 1.0, 0.0]))
    lines = (tmp_path / "x.InpOrb").read_text().splitlines()
    assert lines[:2] == ["#INPORB 2.2", "#INFO"]
    assert lines[3].split() == ["0", "1", "0"] and lines[4:6] == ["       4", "       3"]
    k = lines.index("* ORBITAL    1    2")
    assert [float(t) for t in lines[k + 1].split()] == list(C[:, 1])
    assert len(lines[k + 1]) == 4 * 22
    assert lines[lines.index("#OCC") + 2].split()[1] == "1.00000000000000E+00"


def test_unsym_and_atom_order(tmp_path):
    natm = 2
    H = np.arange(36.0).reshape(6, 6)
    H = H + H.T
    text = ["*BEGIN COORDINATES", "*LABEL COORDINATES CHARGE",
            " N10            0.0000000000        0.0000000000        0.0000000000   7",
            " O20            0.0000000000        0.0000000000        2.0786987370   8",
            "*END COORDINATES", "Unsymmetrized Hessian", "*BEGIN HESSIAN",
            "*Number of pert.     6"]
    for j in range(6):
        text.append(f"* Column{j + 1:9d}")
        col = [f"{v:20.12E}" for v in H[:, j]]
        text += ["".join(col[:4]), "".join(col[4:])]
    text.append("*END HESSIAN")
    (tmp_path / "UnSym").write_text("\n".join(text) + "\n")
    xyz, Hr = mch.read_unsym(tmp_path / "UnSym", natm)
    np.testing.assert_array_equal(Hr, H)
    assert xyz[1, 2] == 2.078698737
    # OpenMolcas's atom k is PySCF's order[k]: swapping two atoms swaps blocks
    Hp = mch.permute(H, [1, 0])
    np.testing.assert_array_equal(Hp[:3, :3], H[3:, 3:])
    np.testing.assert_array_equal(Hp[:3, 3:], H[3:, :3])
