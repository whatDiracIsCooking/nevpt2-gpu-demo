#!/usr/bin/env python3
r"""The one Python reader (and the byte-level writer) of the three binary
formats in ``golden/``: the ``*.nevpt2gold`` golden, the ``*.gradgold``
gradient sidecar (conventional or density-fitted) and the ``*.mogold`` MO
sidecar -- and of the
``*.srcgold`` source-term oracle, which ends in the same array section. It
also reads ``states.json``, the manifest a CASSCF-reference sweep keeps its
displaced states' ``.mogold`` files under (:func:`read_states_manifest`).

Every reader in ``reference_data/`` goes through this module; none parses a
header, or seeks to an offset, of its own. The scalar header of each format is
written down **once**, as a field table keyed by magic (:data:`GOLDEN_HEADERS`,
:data:`GRADGOLD_HEADERS`, :data:`MO_SIDECAR_HEADERS`,
:data:`SRCGOLD_HEADERS`), and both directions are
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
