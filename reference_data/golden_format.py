#!/usr/bin/env python3
r"""The one Python reader (and the byte-level writer) of the three binary
formats in ``golden/``: the ``*.nevpt2gold`` golden, the ``*.gradgold``
gradient sidecar (conventional or density-fitted) and the ``*.mogold`` MO
sidecar -- and of the
``*.srcgold`` source-term oracle, the ``*.respgold`` CASSCF response
sidecar, the ``*.pertgold`` integral perturbation sidecar and the
``*.erigold`` full MO-integral tensor sidecar, which end in the same array
section. It
also reads ``states.json``, the manifest a CASSCF-reference sweep keeps its
displaced states' ``.mogold`` files under (:func:`read_states_manifest`).

Every reader in ``reference_data/`` goes through this module; none parses a
header, or seeks to an offset, of its own. The scalar header of each format is
written down **once**, as a field table keyed by magic (:data:`GOLDEN_HEADERS`,
:data:`GRADGOLD_HEADERS`, :data:`MO_SIDECAR_HEADERS`,
:data:`SRCGOLD_HEADERS`, :data:`RESPGOLD_HEADERS`, :data:`PERTGOLD_HEADERS`,
:data:`ERIGOLD_HEADERS`), and
both directions are
driven by that table: :func:`pack_header` writes it, :func:`_read_header`
reads it, and where the array table begins is wherever the header read stopped
(``f.tell()``) -- derived, never known. A header change is therefore an edit to
one table here (plus a new magic, so the committed files stay readable), and
``src/golden/golden.cpp``'s ``loadGolden`` / ``gradgold.cpp``'s
``loadGradGold`` on the C++ side. The full inventory of readers, and the check
that holds them to both layouts, is docs/reference-data.md, "The file format".

Deliberately light: ``struct`` and ``numpy`` only, and no thread-pool
settings. ``generate_golden.py`` forces single-threading at import, and
``pyscf_contract_time.py`` must not inherit that, so the format lives here and
not there. ``generate_golden.py`` re-exports the magics under their old names.

Readers raise :class:`FormatError` (a ``ValueError``) on a file they cannot
interpret: a wrong magic, an unknown ``reference_kind``, a truncated header or
array, a malformed array header, or bytes after the last array -- the same
refusals ``loadGolden`` makes.

Run as a script it prints where a golden's array table begins, which is what
the regeneration check in docs/reference-data.md hands to ``cmp -i`` instead
of a header length written out by hand::

    uv run python reference_data/golden_format.py arrays-at golden/n2_ccpvdz_cas44.nevpt2gold
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

#: Magic bytes identifying a golden-reference file. ``G2`` is the current
#: layout: it ends its scalar header with the ``reference_kind`` field
#: (:data:`REFERENCE_KINDS`). ``generate_golden.write_golden``'s docstring has
#: the full layout.
GOLDEN_MAGIC = b"NEVPT2G2"

#: The layout before ``reference_kind``, which every committed **CASCI**
#: golden is written in. It stays readable and means ``"casci"`` -- the only
#: thing it could mean (docs/reference-data.md, "The file format").
GOLDEN_MAGIC_V1 = b"NEVPT2G1"

#: Every magic a golden file may carry, newest first.
GOLDEN_MAGICS = (GOLDEN_MAGIC, GOLDEN_MAGIC_V1)

#: The ``reference_kind`` codes a ``G2`` scalar header may hold
#: (``src/golden/golden.cppm``'s ``ReferenceKind`` is the same two codes).
REFERENCE_KINDS = {0: "casci", 1: "casscf"}

#: ``REFERENCE_KINDS`` inverted: the name a writer takes.
REFERENCE_KIND_CODES = {name: code for code, name in REFERENCE_KINDS.items()}

#: Magic bytes identifying a gradient sidecar (``*.gradgold``).
GRADGOLD_MAGIC = b"NEVPT2D1"

#: The DENSITY-FITTED gradient sidecar (#164), ``generate_golden.py --df
#: --gradgold``: beside a ``*_df`` golden, carrying the two three-index blocks
#: that golden lacks (``B_cc``, ``B_vv``; docs/gradient-theory.md, section
#: 3.8), the same full-range matrices as a ``D1`` file but of the fitted
#: calculation, and the four tied orbital-energy arrays. Its own magic because
#: its integrals are a different calculation's: the magic says which, so no
#: consumer guesses it from the array names. Its header is ``D1``'s plus
#: ``naux``.
GRADGOLD_MAGIC_DF = b"NEVPT2F1"

#: Every magic a gradient sidecar may carry.
GRADGOLD_MAGICS = (GRADGOLD_MAGIC, GRADGOLD_MAGIC_DF)

#: Magic bytes identifying an MO sidecar (``*.mogold``).
MO_SIDECAR_MAGIC = b"NEVPT2M1"

#: Magic bytes identifying an SC source-term oracle (``*.srcgold``): PySCF's
#: exact SC-NEVPT2 source terms for one golden's state, which
#: ``nevpt2_sc_pseudodensity --source-oracle`` holds the GPU's to (#92;
#: ``sc_source_oracle.py`` writes it).
SRCGOLD_MAGIC = b"NEVPT2S1"

#: The same format's PC flavour (#95): block2's partially-contracted source
#: terms, ``sc_source_oracle.py --pc``. One layout, one reader; the magic is
#: the only thing that differs, and it is what says which energy the arrays
#: differentiate, so an SC comparison can never read a PC oracle as its own.
SRCGOLD_MAGIC_PC = b"NEVPT2P1"

#: The DIRECTIONAL flavour of each (#105): the same scalar header, but in
#: place of every element of ``A`` and ``P y`` the file holds seeded
#: directions, each inside one block, and the oracle's converged derivative
#: along each (``sc_source_oracle.py --directional``; its docstring has the
#: arrays). The magic says which of the four kinds a file is, so no reader
#: guesses it from the array names.
SRCGOLD_MAGIC_DIR = b"NEVPT2SD"
SRCGOLD_MAGIC_PC_DIR = b"NEVPT2PD"

#: Magic bytes identifying a CASSCF **response sidecar** (``*.respgold``,
#: #219): the ``3N`` first-order parameter responses of a ``_casscf`` state
#: to the nuclear coordinates, as full-range ``(kappa, c)`` directions in the
#: golden's own frame, with the analytic CASSCF nuclear Hessian they assemble
#: (``casscf_nuclear_hessian.py --write``; :data:`RESPGOLD_ARRAYS` has the
#: arrays). The direction set the GPU's bilinear form reads (#222).
RESPGOLD_MAGIC = b"NEVPT2R1"

#: Magic bytes identifying an **integral perturbation sidecar**
#: (``*.pertgold``, #220): the first derivative of a ``_casscf`` state's MO
#: integrals along each nuclear coordinate on the frozen orthonormal path --
#: ``casscf_hessian.Skeleton``'s perturbed Hamiltonians, written in the
#: gradient sidecar's general-range inventory (``integral_perturbation.py
#: --write``; :data:`PERTGOLD_ARRAYS` has the arrays). What
#: ``nevpt2_sc_pseudodensity --sc-source-tangent`` differentiates the source
#: terms along. One magic for both calculations: ``naux`` in the header is 0
#: for the conventional integrals and the auxiliary basis size for the
#: density-fitted ones, as a ``NEVPT2F1`` sidecar's is.
PERTGOLD_MAGIC = b"NEVPT2I1"

#: Magic bytes identifying a **full MO-integral tensor sidecar**
#: (``*.erigold``, #180): every ``(pq|rs)`` of a CONVENTIONAL golden's state
#: over the whole MO range, in the gradient sidecar's orbitals
#: (``generate_golden.py --erigold``; :data:`ERIGOLD_ARRAYS` has the arrays).
#: What the orbital Hessian's cross-slot curvature K2 reads for a direction
#: pair with no active-active member (docs/hessian-theory.md, Sec. 5.4): two
#: general indices reach every MO integral, so nothing smaller serves it. An
#: opt-in sidecar of its own, so no committed ``.gradgold`` grows or changes a
#: byte. A density-fitted case needs none: its ``B`` is already the whole
#: tensor in three-index form.
ERIGOLD_MAGIC = b"NEVPT2E1"

#: A source-term oracle's ``(scheme, flavour)``, by magic: what
#: :func:`read_srcgold` reports as ``header["scheme"]`` /
#: ``header["flavour"]`` and :func:`write_srcgold` picks the magic from
#: (``src/golden/golden.cppm``'s ``SourceScheme`` / ``SourceFlavour`` are the
#: same pairs).
SRCGOLD_KINDS = {
    SRCGOLD_MAGIC: ("sc", "full"),
    SRCGOLD_MAGIC_PC: ("pc", "full"),
    SRCGOLD_MAGIC_DIR: ("sc", "directional"),
    SRCGOLD_MAGIC_PC_DIR: ("pc", "directional"),
}

#: ``SRCGOLD_KINDS`` inverted: the magic a writer takes for ``(scheme, flavour)``.
SRCGOLD_MAGIC_FOR = {kind: magic for magic, kind in SRCGOLD_KINDS.items()}

#: A golden's state fields, common to both layouts: ``struct`` codes, all
#: little-endian with no padding.
_GOLDEN_STATE = (
    ("ncas", "i"), ("nelec_a", "i"), ("nelec_b", "i"), ("ncore", "i"), ("ndet", "q"),
    ("e_casci", "d"), ("e_nevpt2_total", "d"),
)

#: THE golden scalar header, per magic, in file order. Adding a field means a
#: new magic and a new entry here -- never an edit to an existing entry, which
#: would misread every committed file written in that layout (and which
#: ``golden_layout_check.py`` exists to catch).
GOLDEN_HEADERS: dict[bytes, tuple[tuple[str, str], ...]] = {
    GOLDEN_MAGIC: _GOLDEN_STATE + (("reference_kind", "i"),),
    GOLDEN_MAGIC_V1: _GOLDEN_STATE,
}

#: THE gradient sidecar's scalar header: the orbital-space partition, and in
#: the density-fitted flavour also the auxiliary-basis size, every ``B_*``
#: block's leading extent.
_GRADGOLD_PARTITION = tuple(
    (n, "i") for n in ("nao", "nmo", "ncore", "nact", "nvirt", "nelec_rhf")
)
GRADGOLD_HEADERS: dict[bytes, tuple[tuple[str, str], ...]] = {
    GRADGOLD_MAGIC: _GRADGOLD_PARTITION,
    GRADGOLD_MAGIC_DF: _GRADGOLD_PARTITION + (("naux", "i"),),
}

#: THE MO sidecar's scalar header. Its text section and array table follow.
MO_SIDECAR_HEADERS: dict[bytes, tuple[tuple[str, str], ...]] = {
    MO_SIDECAR_MAGIC: tuple(
        (n, "i") for n in ("nao", "nmo", "ncore", "ncas", "nelec_a", "nelec_b")
    ) + (("e_casscf", "d"),),
}

#: THE source-term oracle's scalar header: the orbital-space partition, with
#: the RHF split of the active space (docs/gradient-theory.md, section 2.6:
#: the first ``n_act_occ`` active orbitals of the basis ``a_corr`` is written
#: in are act_O), the determinant count, and the two difference-quotient
#: steps the arrays were taken at. All four kinds share it; in the directional
#: flavour the two steps are the COARSE rung of the Richardson pair each
#: stored value is extrapolated from.
_SRCGOLD_LAYOUT = tuple(
    (n, "i") for n in ("nmo", "ncore", "nact", "nvirt", "n_act_occ")
) + (("ndet", "q"), ("kappa_step", "d"), ("ci_step", "d"))
SRCGOLD_HEADERS: dict[bytes, tuple[tuple[str, str], ...]] = {
    magic: _SRCGOLD_LAYOUT for magic in SRCGOLD_KINDS
}

#: THE response sidecar's scalar header: the atom count (``3 natm`` is every
#: per-coordinate array's leading extent), the orbital-space partition, the
#: active electron counts, the determinant count and the state's CASSCF
#: energy.
RESPGOLD_HEADERS: dict[bytes, tuple[tuple[str, str], ...]] = {
    RESPGOLD_MAGIC: tuple(
        (n, "i") for n in ("natm", "nao", "nmo", "ncore", "nact", "nvirt", "nelec_a",
                           "nelec_b")
    ) + (("ndet", "q"), ("e_casscf", "d")),
}

#: The arrays a response sidecar holds, in file order, each with its shape in
#: header fields (``ncoord`` = ``3 natm``; ``ci`` keeps the golden's own
#: ``(n_alpha_strings, n_beta_strings)`` shape, whose product is ``ndet``).
#: Coordinates are ``k = 3 atom + axis``, bohr.
RESPGOLD_ARRAYS = (
    # the frame the directions are written in: the _casscf golden's own
    # orbitals and CI vector (its gradient sidecar's mo_coeff, its ci)
    ("mo_coeff", ("nao", "nmo")),
    ("ci", None),
    # kappa^(1)_k, antisymmetric: C(R0 + h e_k) = C_f exp(h kappa_k) + O(h^2)
    # on the frozen path C_f = S^-1/2 S0^1/2 C0; zero on the core-core,
    # active-active and virtual-virtual blocks (the redundancy's gauge)
    ("kappa", ("ncoord", "nmo", "nmo")),
    # c^(1)_k, orthogonal to ci, in the same orbitals
    ("ci_response", ("ncoord", "ndet")),
    # d^2 E_CASSCF / dR dR (Eh/bohr^2), its skeleton half L_RR, and dE/dR
    ("hessian", ("ncoord", "ncoord")),
    ("l_rr", ("ncoord", "ncoord")),
    ("gradient", ("ncoord",)),
    # 1.0 where a displacement along the coordinate keeps the point group
    ("kept", ("ncoord",)),
)

#: THE integral perturbation sidecar's scalar header: the atom count (``3
#: natm`` is every dotted array's leading extent), the orbital-space partition
#: and the auxiliary basis size (0: the conventional integrals).
PERTGOLD_HEADERS: dict[bytes, tuple[tuple[str, str], ...]] = {
    PERTGOLD_MAGIC: tuple(
        (n, "i") for n in ("natm", "nao", "nmo", "ncore", "nact", "nvirt", "naux")
    ),
}

#: The arrays an integral perturbation sidecar holds, in file order, each with
#: its shape in header fields (``ncoord`` = ``3 natm``; ``nmo``, ``ncore``,
#: ``nact``, ``nvirt``). Coordinates are ``k = 3 atom + axis``, bohr. Every
#: array after ``mo_coeff`` is the gradient sidecar's array of the same name
#: (``generate_golden._gradient_arrays``; docs/gradient-theory.md, sections
#: 3.3 and 3.5) DIFFERENTIATED along coordinate ``k`` at fixed orbitals and
#: fixed active density, stacked on a leading ``ncoord`` axis: ``h`` the bare
#: one-electron matrix, ``heff`` its core dressing, ``f`` the generalized
#: Fock at the golden's ``dm1``, and the eighteen one-general-index
#: two-electron rows. The golden's own blocks are slices of these and are not
#: stored again (docs/reference-data.md, "The integral perturbation sidecar").
PERTGOLD_ARRAYS = (
    # the frame the perturbations are written in: the golden's own orbitals
    # (its gradient sidecar's mo_coeff, bit for bit)
    ("mo_coeff", ("nao", "nmo")),
    ("h", ("ncoord", "nmo", "nmo")),
    ("heff", ("ncoord", "nmo", "nmo")),
    ("f", ("ncoord", "nmo", "nmo")),
    ("g_xa_aa", ("ncoord", "nmo", "nact", "nact", "nact")),
    ("g_xv_aa", ("ncoord", "nmo", "nvirt", "nact", "nact")),
    ("g_xc_aa", ("ncoord", "nmo", "ncore", "nact", "nact")),
    ("g_xa_va", ("ncoord", "nmo", "nact", "nvirt", "nact")),
    ("g_xv_va", ("ncoord", "nmo", "nvirt", "nvirt", "nact")),
    ("g_xc_va", ("ncoord", "nmo", "ncore", "nvirt", "nact")),
    ("g_xa_ca", ("ncoord", "nmo", "nact", "ncore", "nact")),
    ("g_xv_ca", ("ncoord", "nmo", "nvirt", "ncore", "nact")),
    ("g_xc_ca", ("ncoord", "nmo", "ncore", "ncore", "nact")),
    ("g_xa_cv", ("ncoord", "nmo", "nact", "ncore", "nvirt")),
    ("g_xv_cv", ("ncoord", "nmo", "nvirt", "ncore", "nvirt")),
    ("g_xc_cv", ("ncoord", "nmo", "ncore", "ncore", "nvirt")),
    ("g_xa_cc", ("ncoord", "nmo", "nact", "ncore", "ncore")),
    ("g_xv_cc", ("ncoord", "nmo", "nvirt", "ncore", "ncore")),
    ("g_xc_cc_j", ("ncoord", "nmo", "ncore", "ncore")),
    ("g_xc_cc_k", ("ncoord", "nmo", "ncore", "ncore")),
    ("g_xa_vv", ("ncoord", "nmo", "nact", "nvirt")),
    ("g_xc_vv", ("ncoord", "nmo", "ncore", "nvirt")),
)

#: THE full-tensor sidecar's scalar header: the orbital-space partition of the
#: gradient sidecar beside it (less ``nelec_rhf``).
ERIGOLD_HEADERS: dict[bytes, tuple[tuple[str, str], ...]] = {
    ERIGOLD_MAGIC: tuple((n, "i") for n in ("nao", "nmo", "ncore", "nact", "nvirt")),
}

#: The arrays a full-tensor sidecar holds, in file order, each with its shape
#: in header fields. ``eri[p, q, r, s] = (pq|rs)`` in chemists' order over the
#: whole MO range: ``n_mo^4`` doubles, every permutational copy stored, so the
#: C++ reader uploads it as it is.
ERIGOLD_ARRAYS = (
    # the frame: the golden's orbitals (its gradient sidecar's mo_coeff, bit
    # for bit)
    ("mo_coeff", ("nao", "nmo")),
    ("eri", ("nmo", "nmo", "nmo", "nmo")),
)

#: No array the generator writes has more than a handful of axes; a rank past
#: this is a misread header (``loadGolden``'s ``kMaxRank``).
MAX_RANK = 16


class FormatError(ValueError):
    """A file this module cannot interpret as the format it was asked for."""


def _struct_format(fields) -> str:
    return "<" + "".join(code for _name, code in fields)


def pack_header(layouts: dict, magic: bytes, values: dict) -> bytes:
    """``magic`` followed by the scalar header ``layouts[magic]`` describes,
    each field taken from ``values`` by name -- the one place a header is
    packed, so a writer and :func:`_read_header` cannot disagree on order or
    width."""
    fields = layouts[magic]
    return magic + struct.pack(
        _struct_format(fields), *(values[name] for name, _code in fields)
    )


def write_arrays(f, arrays: dict[str, np.ndarray]) -> None:
    """The named-array section every format here ends with: an int32 count,
    then per array a name, a rank, int64 extents and C-order f64 data."""
    f.write(struct.pack("<i", len(arrays)))
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        name_b = name.encode("ascii")
        f.write(struct.pack("<i", len(name_b)))
        f.write(name_b)
        f.write(struct.pack("<i", arr.ndim))
        f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))
        f.write(arr.tobytes(order="C"))


def write_fields(f, fields: dict[str, str]) -> None:
    """The MO sidecar's text section: an int32 count, then per field a UTF-8
    key and a UTF-8 value, each length-prefixed."""
    f.write(struct.pack("<i", len(fields)))
    for key, val in fields.items():
        for text in (key, val):
            b = text.encode("utf-8")
            f.write(struct.pack("<i", len(b)))
            f.write(b)


def _read_exact(f, n: int, path, what: str) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise FormatError(f"{path}: unexpected EOF reading {what}")
    return b


def _unpack(f, fmt: str, path, what: str) -> tuple:
    return struct.unpack(fmt, _read_exact(f, struct.calcsize(fmt), path, what))


def _read_header(f, path, layouts: dict, label: str) -> dict:
    """The magic and the scalar header ``layouts`` gives for it, as a dict
    holding ``magic`` plus every field by name. Leaves ``f`` at the first byte
    after the header."""
    magic = _read_exact(f, 8, path, "the magic")
    fields = layouts.get(magic)
    if fields is None:
        raise FormatError(
            f"{path} is not a {label}: magic {magic!r}, expected one of "
            f"{sorted(layouts)}"
        )
    values = _unpack(f, _struct_format(fields), path, "the scalar header")
    header = dict(zip((name for name, _code in fields), values, strict=True))
    header["magic"] = magic
    return header


def _read_text_fields(f, path) -> dict[str, str]:
    (n,) = _unpack(f, "<i", path, "the field count")
    if n < 0:
        raise FormatError(f"{path}: negative field count {n}")
    fields: dict[str, str] = {}
    for _ in range(n):
        pair = []
        for _ in range(2):
            (ln,) = _unpack(f, "<i", path, "a field length")
            if ln < 0:
                raise FormatError(f"{path}: negative field length {ln}")
            pair.append(_read_exact(f, ln, path, "a field").decode("utf-8"))
        fields[pair[0]] = pair[1]
    return fields


def _read_arrays(f, path, want: set[str] | None) -> dict[str, np.ndarray]:
    """The array section from ``f``'s current position to the end of the
    file, keeping the arrays named in ``want`` (every array when ``None``).
    Every array header is validated before it sizes anything, and the section
    must end exactly at the end of the file: a header misread by even one
    field fails here rather than handing back shifted data."""
    size = os.fstat(f.fileno()).st_size
    (n_arrays,) = _unpack(f, "<i", path, "the array count")
    if n_arrays < 0:
        raise FormatError(f"{path}: negative array count {n_arrays}")
    arrays: dict[str, np.ndarray] = {}
    for _ in range(n_arrays):
        (name_len,) = _unpack(f, "<i", path, "an array name length")
        if name_len < 0 or name_len > size - f.tell():
            raise FormatError(f"{path}: bad array name length {name_len}")
        raw = _read_exact(f, name_len, path, "an array name")
        try:
            name = raw.decode("ascii")
        except UnicodeDecodeError as e:
            raise FormatError(f"{path}: array name {raw[:32]!r} is not ASCII") from e
        (ndim,) = _unpack(f, "<i", path, f"array '{name}''s rank")
        if not 0 <= ndim <= MAX_RANK:
            raise FormatError(f"{path}: array '{name}' has rank {ndim} (expected 0..{MAX_RANK})")
        shape = _unpack(f, f"<{ndim}q", path, f"array '{name}''s shape")
        if any(s < 0 for s in shape):
            raise FormatError(f"{path}: array '{name}' has a negative extent {shape}")
        nbytes = 8 * int(np.prod(shape, dtype=np.int64))
        if nbytes > size - f.tell():
            raise FormatError(f"{path}: array '{name}' runs past the end of the file")
        if want is None or name in want:
            data = _read_exact(f, nbytes, path, f"array '{name}'")
            arrays[name] = np.frombuffer(data, "<f8").reshape(shape).copy()
        else:
            f.seek(nbytes, 1)
    if f.tell() != size:
        raise FormatError(
            f"{path}: {size - f.tell()} bytes after the last of its {n_arrays} arrays"
        )
    return arrays


def _golden_header(f, path) -> dict:
    header = _read_header(f, path, GOLDEN_HEADERS, "golden file")
    # A G1 file has no field and is a CASCI on RHF orbitals by definition.
    code = header.get("reference_kind", REFERENCE_KIND_CODES["casci"])
    if code not in REFERENCE_KINDS:
        raise FormatError(f"{path}: unknown reference_kind {code}")
    header["reference_kind"] = REFERENCE_KINDS[code]
    # Where the array table begins, i.e. how long this file's header turned out
    # to be: read off the stream, never computed from a layout.
    header["arrays_at"] = f.tell()
    return header


def read_golden_header(path) -> dict:
    """A golden's scalar header: ``ncas``, ``nelec_a``, ``nelec_b``,
    ``ncore``, ``ndet``, ``e_casci``, ``e_nevpt2_total``, ``reference_kind``
    (a name: ``"casci"`` for a ``G1`` file), plus ``magic`` and ``arrays_at``
    (the byte offset of the array table, as read). Reads only the header."""
    with open(path, "rb") as f:
        return _golden_header(f, path)


def read_golden(path, want: set[str] | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    """``(header, arrays)`` from a golden: :func:`read_golden_header`'s dict,
    and the arrays named in ``want`` (all of them when ``None``; the files run
    to tens of MB, so a caller needing one scalar names it). The whole array
    table is walked either way, so a misread header fails even when one array
    is asked for, and a ``ci`` that is read must hold ``ndet`` elements."""
    with open(path, "rb") as f:
        header = _golden_header(f, path)
        arrays = _read_arrays(f, path, want)
    if "ci" in arrays and arrays["ci"].size != header["ndet"]:
        raise FormatError(
            f"{path}: ci has {arrays['ci'].size} elements but the header says "
            f"ndet = {header['ndet']}"
        )
    return header, arrays


def read_gradgold(path) -> tuple[dict, dict[str, np.ndarray]]:
    """``(header, arrays)`` from a gradient sidecar of either flavour.
    ``header`` holds exactly the six partition fields of a conventional
    (``NEVPT2D1``) sidecar, plus ``naux`` for a density-fitted (``NEVPT2F1``)
    one -- which is how a caller tells the two apart -- so
    ``generate_golden.write_gradgold(path, **header, arrays=arrays)`` writes
    the same file back."""
    with open(path, "rb") as f:
        header = _read_header(f, path, GRADGOLD_HEADERS, "gradient sidecar")
        del header["magic"]
        arrays = _read_arrays(f, path, None)
    return header, arrays


def read_mo_sidecar(path) -> tuple[dict, dict[str, str], dict[str, np.ndarray]]:
    """``(header, fields, arrays)`` from an MO sidecar: the seven scalar
    fields, the text section, and every array. The checks on what the arrays
    must be are ``generate_golden.read_mo_sidecar``'s."""
    with open(path, "rb") as f:
        header = _read_header(f, path, MO_SIDECAR_HEADERS, "MO sidecar")
        del header["magic"]
        fields = _read_text_fields(f, path)
        arrays = _read_arrays(f, path, None)
    return header, fields, arrays


def write_srcgold(path, *, header: dict, arrays: dict[str, np.ndarray]) -> None:
    """Write a source-term oracle (``*.srcgold``): the magic of
    ``(header["scheme"], header["flavour"])`` (:data:`SRCGOLD_KINDS`:
    ``NEVPT2S1`` / ``NEVPT2P1`` full, ``NEVPT2SD`` / ``NEVPT2PD``
    directional), the :data:`SRCGOLD_HEADERS` scalar header from ``header``,
    then the named-array section every format here ends with.
    :func:`read_srcgold` reads it back into the same ``(header, arrays)``."""
    magic = SRCGOLD_MAGIC_FOR[(header["scheme"], header["flavour"])]
    with open(path, "wb") as f:
        f.write(pack_header(SRCGOLD_HEADERS, magic, header))
        write_arrays(f, arrays)


def read_srcgold(path) -> tuple[dict, dict[str, np.ndarray]]:
    """``(header, arrays)`` from a source-term oracle. ``header`` holds the
    :data:`SRCGOLD_HEADERS` fields plus ``scheme`` (``"sc"`` or ``"pc"``) and
    ``flavour`` (``"full"`` or ``"directional"``), both off the magic, so
    :func:`write_srcgold` writes the same file back."""
    with open(path, "rb") as f:
        header = _read_header(f, path, SRCGOLD_HEADERS, "source-term oracle")
        header["scheme"], header["flavour"] = SRCGOLD_KINDS[header.pop("magic")]
        arrays = _read_arrays(f, path, None)
    return header, arrays


def _check_respgold(path, header: dict, arrays: dict[str, np.ndarray]) -> None:
    """Exactly the :data:`RESPGOLD_ARRAYS`, in order, each at the shape the
    header gives it."""
    dims = dict(header, ncoord=3 * header["natm"])
    names = [n for n, _s in RESPGOLD_ARRAYS]
    if list(arrays) != names:
        raise FormatError(f"{path}: arrays {list(arrays)}, expected {names}")
    for name, shape in RESPGOLD_ARRAYS:
        got = arrays[name].shape
        if shape is None:
            if int(np.prod(got)) != dims["ndet"]:
                raise FormatError(f"{path}: {name} has {int(np.prod(got))} elements, "
                                  f"ndet = {dims['ndet']}")
        elif got != tuple(dims[d] for d in shape):
            raise FormatError(f"{path}: {name} is {got}, expected {shape} = "
                              f"{tuple(dims[d] for d in shape)}")


def write_respgold(path, *, header: dict, arrays: dict[str, np.ndarray]) -> None:
    """Write a CASSCF response sidecar (``*.respgold``, ``NEVPT2R1``): the
    :data:`RESPGOLD_HEADERS` scalar header from ``header``, then the
    :data:`RESPGOLD_ARRAYS` in their order, held to their shapes before a
    byte is written. :func:`read_respgold` reads it back into the same
    ``(header, arrays)``."""
    names = [n for n, _s in RESPGOLD_ARRAYS]
    if set(arrays) != set(names):
        raise FormatError(f"{path}: arrays {sorted(arrays)}, expected {sorted(names)}")
    arrays = {n: np.asarray(arrays[n], dtype=np.float64) for n in names}
    _check_respgold(path, header, arrays)
    with open(path, "wb") as f:
        f.write(pack_header(RESPGOLD_HEADERS, RESPGOLD_MAGIC, header))
        write_arrays(f, arrays)


def read_respgold(path) -> tuple[dict, dict[str, np.ndarray]]:
    """``(header, arrays)`` from a CASSCF response sidecar, every array held
    to :data:`RESPGOLD_ARRAYS`' order and shapes, so :func:`write_respgold`
    writes the same file back."""
    with open(path, "rb") as f:
        header = _read_header(f, path, RESPGOLD_HEADERS, "response sidecar")
        del header["magic"]
        arrays = _read_arrays(f, path, None)
    _check_respgold(path, header, arrays)
    return header, arrays


def _check_pertgold(path, header: dict, arrays: dict[str, np.ndarray]) -> None:
    """Exactly the :data:`PERTGOLD_ARRAYS`, in order, each at the shape the
    header gives it; a partition that does not add up, or a negative
    ``naux``, is refused before any shape is read against it. A sidecar
    missing any array is refused whole, never read in part."""
    if header["ncore"] + header["nact"] + header["nvirt"] != header["nmo"] or header["naux"] < 0:
        raise FormatError(f"{path}: header {header} is not a usable partition")
    dims = dict(header, ncoord=3 * header["natm"])
    names = [n for n, _s in PERTGOLD_ARRAYS]
    if list(arrays) != names:
        missing = [n for n in names if n not in arrays]
        raise FormatError(f"{path}: arrays {list(arrays)}, expected {names}"
                          + (f" (missing {', '.join(missing)})" if missing else ""))
    for name, shape in PERTGOLD_ARRAYS:
        want = tuple(dims[d] for d in shape)
        if arrays[name].shape != want:
            raise FormatError(f"{path}: {name} is {arrays[name].shape}, expected {shape} = {want}")


def write_pertgold(path, *, header: dict, arrays: dict[str, np.ndarray]) -> None:
    """Write an integral perturbation sidecar (``*.pertgold``, ``NEVPT2I1``):
    the :data:`PERTGOLD_HEADERS` scalar header from ``header``, then the
    :data:`PERTGOLD_ARRAYS` in their order, held to their shapes before a
    byte is written. :func:`read_pertgold` reads it back into the same
    ``(header, arrays)``."""
    names = [n for n, _s in PERTGOLD_ARRAYS]
    if set(arrays) != set(names):
        raise FormatError(f"{path}: arrays {sorted(arrays)}, expected {sorted(names)}")
    arrays = {n: np.asarray(arrays[n], dtype=np.float64) for n in names}
    _check_pertgold(path, header, arrays)
    with open(path, "wb") as f:
        f.write(pack_header(PERTGOLD_HEADERS, PERTGOLD_MAGIC, header))
        write_arrays(f, arrays)


def read_pertgold(path) -> tuple[dict, dict[str, np.ndarray]]:
    """``(header, arrays)`` from an integral perturbation sidecar, every array
    held to :data:`PERTGOLD_ARRAYS`' order and shapes, so
    :func:`write_pertgold` writes the same file back."""
    with open(path, "rb") as f:
        header = _read_header(f, path, PERTGOLD_HEADERS, "integral perturbation sidecar")
        del header["magic"]
        arrays = _read_arrays(f, path, None)
    _check_pertgold(path, header, arrays)
    return header, arrays


def _check_erigold(path, header: dict, arrays: dict[str, np.ndarray]) -> None:
    """Exactly the :data:`ERIGOLD_ARRAYS`, in order, each at the shape the
    header gives it; a partition that does not add up is refused before any
    shape is read against it. A sidecar missing an array is refused whole."""
    if (header["ncore"] + header["nact"] + header["nvirt"] != header["nmo"]
            or header["nmo"] < 1 or header["nao"] < 1 or header["nact"] < 1
            or header["ncore"] < 0 or header["nvirt"] < 0):
        raise FormatError(f"{path}: header {header} is not a usable partition")
    names = [n for n, _s in ERIGOLD_ARRAYS]
    if list(arrays) != names:
        missing = [n for n in names if n not in arrays]
        raise FormatError(f"{path}: arrays {list(arrays)}, expected {names}"
                          + (f" (missing {', '.join(missing)})" if missing else ""))
    for name, shape in ERIGOLD_ARRAYS:
        want = tuple(header[d] for d in shape)
        if arrays[name].shape != want:
            raise FormatError(f"{path}: {name} is {arrays[name].shape}, expected {shape} = {want}")


def write_erigold(path, *, header: dict, arrays: dict[str, np.ndarray]) -> None:
    """Write a full MO-integral tensor sidecar (``*.erigold``, ``NEVPT2E1``):
    the :data:`ERIGOLD_HEADERS` scalar header from ``header``, then the
    :data:`ERIGOLD_ARRAYS` in their order, held to their shapes before a byte
    is written. :func:`read_erigold` reads it back into the same
    ``(header, arrays)``."""
    names = [n for n, _s in ERIGOLD_ARRAYS]
    if set(arrays) != set(names):
        raise FormatError(f"{path}: arrays {sorted(arrays)}, expected {sorted(names)}")
    arrays = {n: np.asarray(arrays[n], dtype=np.float64) for n in names}
    _check_erigold(path, header, arrays)
    with open(path, "wb") as f:
        f.write(pack_header(ERIGOLD_HEADERS, ERIGOLD_MAGIC, header))
        write_arrays(f, arrays)


def read_erigold(path) -> tuple[dict, dict[str, np.ndarray]]:
    """``(header, arrays)`` from a full MO-integral tensor sidecar, every
    array held to :data:`ERIGOLD_ARRAYS`' order and shapes, so
    :func:`write_erigold` writes the same file back."""
    with open(path, "rb") as f:
        header = _read_header(f, path, ERIGOLD_HEADERS, "full MO-integral tensor sidecar")
        del header["magic"]
        arrays = _read_arrays(f, path, None)
    _check_erigold(path, header, arrays)
    return header, arrays


# --------------------------------------------------------------------------
# The .gradsrc dump (#73; the integral adjoint, #246)
# --------------------------------------------------------------------------

#: The GPU's source-term dump (``nevpt2_sc_pseudodensity --dump-source-terms``,
#: written by ``apps/sc_pseudodensity/gradsrc.cpp``): a NumPy ``.npz`` of
#: STORED NPY v1.0 members, which ``numpy.load`` reads as it is. Its members
#: in file order, each with its shape as one letter per axis (``n`` = n_mo,
#: ``c`` / ``a`` / ``v`` = n_core / n_act / n_virt, ``d`` = n_dir, ``*`` the
#: CI vector's shape, ``0`` a 0-d scalar), and its dtype. ``pc`` is present
#: (= 1) only in a PC dump; the three ``edot_*`` members are a set, and so are
#: the 21 ``ibar_*`` ones -- the integral adjoint, dE/d(each array of the
#: gradient sidecar's inventory), in :data:`PERTGOLD_ARRAYS`' order and the
#: golden's frame, what a ``.pertgold``'s per-coordinate dotted inventory is
#: contracted with (docs/gradients.md, "The integral adjoint").
GRADSRC_CORE = (
    ("kappa_source", "nn", "<f8"),
    ("ci_source", "*", "<f8"),
    ("ci", "*", "<f8"),
    ("ncore", "0", "<i8"),
    ("nact", "0", "<i8"),
    ("nvirt", "0", "<i8"),
    ("e_corr", "0", "<f8"),
)
GRADSRC_PC = (("pc", "0", "<i8"),)
GRADSRC_EDOT = (
    ("edot_kappa", "dnn", "<f8"),
    ("edot_frozen", "d", "<f8"),
    ("edot_digest", "d", "<f8"),
)
#: The integral adjoint's arrays, ``ibar_<name>``: each the inventory array's
#: shape (:data:`PERTGOLD_ARRAYS` less its leading ``ncoord`` axis).
GRADSRC_IBAR = tuple(
    (f"ibar_{name}",
     "".join({"nmo": "n", "ncore": "c", "nact": "a", "nvirt": "v"}[s] for s in shape[1:]),
     "<f8")
    for name, shape in PERTGOLD_ARRAYS if name != "mo_coeff")

#: Every member a dump may carry, in the order the C++ writer emits them.
GRADSRC_MEMBERS = GRADSRC_CORE + GRADSRC_PC + GRADSRC_EDOT + GRADSRC_IBAR

#: A fixed DOS date (1980-01-01) and time (00:00): the archive is a function of
#: its contents, as the C++ writer's is.
_ZIP_DOS_DATE = (0 << 9) | (1 << 5) | 1
_ZIP_VERSION = 20


def _gradsrc_shape(path, axes: str, sizes: dict, ci_shape) -> tuple:
    if axes == "0":
        return ()
    if axes == "*":
        return tuple(ci_shape)
    return tuple(sizes[c] for c in axes)


def _check_gradsrc(path, members: dict[str, np.ndarray]) -> None:
    """The dump's own rules, both directions: the core members, the two sets
    whole or absent, every shape and dtype, and no member it does not write."""
    known = {n for n, _a, _t in GRADSRC_MEMBERS}
    stray = sorted(set(members) - known)
    if stray:
        raise FormatError(f"{path}: members a .gradsrc does not carry: {stray}")
    for n, _a, _t in GRADSRC_CORE:
        if n not in members:
            raise FormatError(f"{path}: no '{n}' member -- not a gradient source dump")
    for label, group in (("edot", GRADSRC_EDOT), ("integral adjoint", GRADSRC_IBAR)):
        have = [n for n, _a, _t in group if n in members]
        if have and len(have) != len(group):
            raise FormatError(f"{path}: the {label} members are a set of {len(group)}, and "
                              f"{len(have)} of them are here")
    if "pc" in members and int(members["pc"]) != 1:
        raise FormatError(f"{path}: 'pc' is {int(members['pc'])}, and a dump writes it only as 1")
    nc, na, nv = (int(members[k]) for k in ("ncore", "nact", "nvirt"))
    sizes = dict(n=nc + na + nv, c=nc, a=na, v=nv)
    if "edot_kappa" in members:
        sizes["d"] = int(np.asarray(members["edot_kappa"]).shape[0])
    ci_shape = np.asarray(members["ci"]).shape
    for n, axes, dtype in GRADSRC_MEMBERS:
        if n not in members:
            continue
        a = np.asarray(members[n])
        want = _gradsrc_shape(path, axes, sizes, ci_shape)
        if a.shape != want or a.dtype != np.dtype(dtype):
            raise FormatError(f"{path}: '{n}' is {a.dtype}{a.shape}, expected {dtype}{want}")


def read_gradsrc(path) -> dict[str, np.ndarray]:
    """A ``.gradsrc`` dump's members, by name (0-d members as 0-d arrays),
    held to :func:`_check_gradsrc`'s rules. ``numpy.load`` reads the archive:
    numpy is the format's consumer, so its reading is the one that counts."""
    with np.load(path) as z:
        members = {n: np.array(z[n]) for n in z.files}
    _check_gradsrc(path, members)
    return members


def _npy_bytes(a: np.ndarray) -> bytes:
    """One member as the C++ writer emits it: NPY v1.0, the header dict in
    numpy's spelling, padded with spaces to a 64-byte data offset."""
    shape = a.shape
    spelt = "()" if not shape else (f"({shape[0]},)" if len(shape) == 1
                                     else "(" + ", ".join(str(s) for s in shape) + ")")
    header = f"{{'descr': '{a.dtype.str}', 'fortran_order': False, 'shape': {spelt}, }}"
    fixed = 6 + 2 + 2
    total = (fixed + len(header) + 1 + 63) // 64 * 64
    header = header + " " * (total - fixed - len(header) - 1) + "\n"
    return (b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(header))
            + header.encode("latin1") + np.ascontiguousarray(a).tobytes())


def write_gradsrc(path, members: dict[str, np.ndarray]) -> None:
    """Writes a ``.gradsrc`` byte for byte as ``gradsrc.cpp``'s
    ``writeGradSrc`` does -- the members in :data:`GRADSRC_MEMBERS`' order,
    a stored zip with a fixed timestamp -- after :func:`_check_gradsrc`."""
    import zlib

    _check_gradsrc(path, members)
    out = bytearray()
    central = bytearray()
    n = 0
    for name, _axes, dtype in GRADSRC_MEMBERS:
        if name not in members:
            continue
        data = _npy_bytes(np.asarray(members[name], dtype=np.dtype(dtype)))
        fname = f"{name}.npy".encode()
        crc = zlib.crc32(data) & 0xFFFFFFFF
        offset = len(out)
        # flags, method (stored), time, date, crc, csize, usize, name length
        common = struct.pack("<HHHHIIIH", 0, 0, 0, _ZIP_DOS_DATE, crc, len(data), len(data),
                             len(fname))
        out += struct.pack("<IH", 0x04034B50, _ZIP_VERSION) + common + struct.pack("<H", 0)
        out += fname + data
        central += (struct.pack("<IHH", 0x02014B50, _ZIP_VERSION, _ZIP_VERSION) + common
                    + struct.pack("<HHHHII", 0, 0, 0, 0, 0, offset) + fname)
        n += 1
    at = len(out)
    out += central
    out += struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, n, n, len(central), at, 0)
    Path(path).write_bytes(bytes(out))


def srcgold_name(case: str, scheme: str, flavour: str) -> str:
    """The file name an oracle of ``case`` and this kind is committed under,
    ``<case>[_pc][_dir].srcgold``; :func:`srcgold_case` inverts it."""
    return (case + ("_pc" if scheme == "pc" else "")
            + ("_dir" if flavour == "directional" else "") + ".srcgold")


def srcgold_case(path, scheme: str, flavour: str = "full") -> str:
    """The case an oracle belongs to: its file name's stem, less the ``_dir``
    a directional oracle carries and then the ``_pc`` a PC one does
    (``golden/<case>_pc_dir.srcgold`` sits beside the same
    ``<case>.nevpt2gold`` / ``.gradgold`` as ``<case>.srcgold``). The C++
    side is ``srcGoldCase`` in ``src/golden/golden.cppm``."""
    stem = Path(path).stem
    if flavour == "directional":
        if not stem.endswith("_dir"):
            raise FormatError(f"{path}: a directional oracle's name must end in _dir.srcgold")
        stem = stem[: -len("_dir")]
    if scheme == "pc":
        if not stem.endswith("_pc"):
            want = "_pc_dir" if flavour == "directional" else "_pc"
            raise FormatError(f"{path}: a PC oracle's name must end in {want}.srcgold")
        stem = stem[: -len("_pc")]
    return stem


#: The name of the manifest ``nevpt2_numgrad.py --reference casscf
#: --states-dir DIR`` writes beside the displaced states' ``.mogold`` files.
STATES_MANIFEST = "states.json"

#: What every ``states.json`` carries at the top, and what every state in it
#: carries (``nevpt2_numgrad._write_manifest``).
_MANIFEST_HEAD = ("case", "reference", "basis", "ncas", "nelecas", "cas_irreps",
                  "step", "finish", "states")
_MANIFEST_STATE = ("label", "atom", "s")


def read_states_manifest(states_dir) -> dict:
    """``DIR/states.json``, the hand-off of a CASSCF-reference sweep's
    displaced states (docs/reference-data.md, "The numerical gradient of a
    CASSCF reference", "One sweep, three consumers"): the manifest as written,
    with two fields added to every state -- ``kept`` (the continuity guard
    tripped nothing and nothing refused it, so it was differenced) and
    ``mogold_path`` (its ``.mogold``, resolved and checked to exist, or
    ``None`` for a state that is not kept).

    JSON, not a binary layout, but read here so there is still one reader of
    what the sweep hands on. Refused with :class:`FormatError`: a missing
    header field, a reference other than ``casscf`` (only that route keeps
    states), a state without a label, ``--atom`` spec or ``s``, two states
    with one label, and a kept state whose ``.mogold`` or energies are
    missing."""
    path = Path(states_dir) / STATES_MANIFEST
    try:
        head = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise FormatError(f"{path}: not JSON ({e})") from None
    if not isinstance(head, dict):
        raise FormatError(f"{path}: not a states manifest (top level is not an object)")
    missing = [k for k in _MANIFEST_HEAD if k not in head]
    if missing:
        raise FormatError(f"{path}: not a states manifest, missing {', '.join(missing)}")
    if head["reference"] != "casscf":
        raise FormatError(f"{path}: reference {head['reference']!r}; only the casscf "
                          "route keeps its displaced states")
    seen: set[str] = set()
    for st in head["states"]:
        absent = [k for k in _MANIFEST_STATE if k not in st]
        if absent:
            raise FormatError(f"{path}: a state lacks {', '.join(absent)}: {st}")
        label = st["label"]
        if label in seen:
            raise FormatError(f"{path}: two states are labelled {label!r}")
        seen.add(label)
        st["kept"] = not st.get("tripped") and "refused" not in st
        st["mogold_path"] = None
        if not st["kept"]:
            continue
        absent = [k for k in ("mogold", "e_casci", "e_corr") if k not in st]
        if absent:
            raise FormatError(f"{path}: kept state {label!r} lacks {', '.join(absent)}")
        mo = Path(states_dir) / st["mogold"]
        if not mo.is_file():
            raise FormatError(f"{path}: kept state {label!r}'s {mo.name} is not there")
        st["mogold_path"] = mo
    return head


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "arrays-at":
        print("usage: golden_format.py arrays-at FILE.nevpt2gold", file=sys.stderr)
        return 2
    print(read_golden_header(Path(argv[1]))["arrays_at"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
