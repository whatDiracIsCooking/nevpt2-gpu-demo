#!/usr/bin/env python3
r"""The golden-file **layout contract**: every committed golden, of both
header layouts, round-trips through every Python reader in this directory, so a
header change that one reader does not follow fails here instead of silently
misreading a golden.

It exists because the ``NEVPT2G1`` -> ``NEVPT2G2`` header change (one 4-byte
``reference_kind`` field) broke three separate readers, found at three
different times, the last by luck (issue #71). Since then there is one parser,
:mod:`golden_format`, and this script holds the tree to it:

1. **No stray parser.** No ``reference_data/*.py`` other than
   ``golden_format.py`` (and this file) may unpack bytes, seek, or spell a
   format magic; no file in ``src/`` or ``apps/`` outside ``src/golden/`` may
   open, read or seek a file, or spell a magic. A fourth reader written later
   is caught by its text, before it is ever run. A C++ file that reads a
   non-golden format is named, with why, in ``CPP_NON_GOLDEN_READERS``; it may
   open a file but still fails on a seek or a magic.
2. **Every committed file, every reader.** Each ``golden/*.nevpt2gold`` --
   ``G1`` (the CASCI goldens) and ``G2`` (the ``*_casscf*`` ones); both layouts
   must be present -- is read by :func:`golden_format.read_golden`, re-packed
   from the header table of its own magic, and must reproduce its own bytes
   exactly; ``G2`` ones also through ``generate_golden.write_golden``. Then
   each reader entry point (``golden_format.read_golden_header``,
   ``nevpt2_numgrad.read_golden_pc_total``, ``block2_pc_probe.read_golden``,
   ``pyscf_contract_time.read_golden``) must return the same header and the
   same arrays. Each ``golden/*.gradgold`` and ``golden/*.mogold`` likewise,
   through its reader and back through its writer, byte for byte (both
   gradient-sidecar flavours must be present: ``NEVPT2D1`` and the
   density-fitted ``NEVPT2F1``, #164, which must sit beside a density-fitted
   golden of its own ``naux``) -- and each
   ``golden/*.srcgold`` source-term oracle (#92) through
   ``golden_format.read_srcgold`` / ``write_srcgold``, held to the golden
   beside it (partition, determinant count and CI vector, bit for bit); all
   four kinds, ``NEVPT2S1`` (SC) and ``NEVPT2P1`` (PC, #95, named
   ``<case>_pc.srcgold``) and their directional flavours ``NEVPT2SD`` /
   ``NEVPT2PD`` (#105, ``<case>[_pc]_dir.srcgold``), must be present. Each
   ``golden/*.respgold`` CASSCF response sidecar (#219, ``NEVPT2R1``)
   through ``golden_format.read_respgold`` / ``write_respgold``, held to the
   ``_casscf`` golden and gradient sidecar beside it (partition, ``ci`` and
   ``mo_coeff`` bit for bit: the frame its directions are written in). Each
   ``golden/*.pertgold`` integral perturbation sidecar (#220, ``NEVPT2I1``)
   through ``golden_format.read_pertgold`` / ``write_pertgold``, held to the
   golden and gradient sidecar beside it (partition, calculation -- ``naux``
   -- and ``mo_coeff`` bit for bit); both flavours, conventional and
   density-fitted, must be present. Each ``golden/*.erigold`` full
   MO-integral tensor sidecar (#180, ``NEVPT2E1``) through
   ``golden_format.read_erigold`` / ``write_erigold``, held to the
   conventional golden and gradient sidecar beside it (partition and
   ``mo_coeff`` bit for bit), and its tensor to the gradient sidecar's
   ``g_xa_aa`` row it overlaps; at least one must be present.
3. **The writer as it is now.** A golden written by the current
   ``generate_golden.write_golden`` (both reference kinds) is read back by
   every reader, so a field added to the writer and not to the readers fails
   even before any file is regenerated.
4. **The CASSCF sweep's hand-off** (#138). A ``states.json`` from
   ``nevpt2_numgrad._write_manifest`` must read back through
   ``golden_format.read_states_manifest``. A transient golden as
   ``pc_on_states.py`` writes it (NaN PC placeholders) must read back through
   every golden reader -- and a golden under ``golden/`` holding any NaN or
   infinity is refused, so such a file can never be committed.
5. **The ``.gradsrc`` dump** (#73, with the integral adjoint of #246). None
   is committed, so every flavour (SC / PC, with and without the ``edot_*``
   and ``ibar_*`` sets) is synthesised, written by
   ``golden_format.write_gradsrc``, read back by ``read_gradsrc`` and by a
   bare ``numpy.load``, and rewritten byte for byte; a partial ``ibar_*``
   set, a wrong shape and a stray member are refused; and the C++ writer's
   ``kIntegralAdjointNames`` table (``apps/sc_pseudodensity/gradsrc.cppm``)
   must be ``GRADSRC_IBAR``. ``--gradsrc DUMP...`` adds dumps the C++ writer
   wrote: each must be ``write_gradsrc``'s bytes exactly.
6. **The ``.parhess`` sidecar** (#222), the GPU's parameter Hessian over a
   response sidecar's directions. An output, so none is committed: a
   synthetic one is written by ``golden_format.write_parhess``, read back by
   ``read_parhess`` and rewritten byte for byte, and a missing array, a
   mis-shaped one and a partition that does not add up are refused.
   ``--parhess FILE...`` adds files the C++ writer (``writeParHess``) wrote:
   each must be ``write_parhess``'s bytes exactly.

The C++ half is ``test/golden``'s ``GoldenLayoutContractTests``, which loads
every committed golden and sidecar with ``loadGolden`` / ``loadGradGold`` /
``loadSrcGold`` in the unit tier. This half is offline Python (it imports PySCF and block2 through
the scripts it checks), so like the gradient comparison it is a manual gate
and registers no ctest entry (docs/testing.md, "The golden layout contract")::

    uv run python reference_data/golden_layout_check.py

It reads ``golden/`` and writes only to a temporary directory. Prints a line
per file and ends ``PASS:`` (exit 0) or ``FAIL:`` with every failure (exit 1).
"""

from __future__ import annotations

# Imported first: it forces OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads.
import generate_golden as gg

# isort: split
import block2_pc_probe as b2p
import nevpt2_numgrad as ngrad
import pc_on_states as pos
import pyscf_contract_time as pct

# isort: split
import io
import re
import sys
import tempfile
from pathlib import Path

import golden_format as gf
import numpy as np

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "golden"

#: What only golden_format.py may do in reference_data/: unpack bytes, seek, or
#: spell a magic.
PY_PARSER = re.compile(
    r"struct\.(unpack|calcsize)|unpack_from|\.seek\(|np\.fromfile|np\.frombuffer"
    r"|b[\"']NEVPT2"
)
#: What only src/golden/ may do in the C++ tree: open, read or seek a file, or
#: spell a magic.
CPP_PARSER = re.compile(r"\bfopen\b|\bfread\b|\bfseek\b|\bseekg\b|\bifstream\b|NEVPT2[GDFMI]\d")
#: C++ files outside src/golden/ that read a file which is NOT one of the three
#: golden layouts, each with why. They may open a file; they are still held to
#: CPP_GOLDEN_SIGNS, so one that grows a seek or a golden magic fails as before.
CPP_NON_GOLDEN_READERS = {
    # The .gradsrc dump is a NumPy .npz (#73), parsed by its own zip/npy reader
    # for the round-trip test; it shares no header with .nevpt2gold/.gradgold/
    # .mogold, so no golden header change can make it misread.
    "apps/sc_pseudodensity/gradsrc.cpp": ".gradsrc (.npz) reader",
}
#: What even a CPP_NON_GOLDEN_READERS file may not do: seek, or spell a magic.
CPP_GOLDEN_SIGNS = re.compile(r"\bfseek\b|\bseekg\b|NEVPT2[GDFMI]\d")


class Failures:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, what: str) -> None:
        self.items.append(what)
        print(f"  FAILED {what}", flush=True)

    def check(self, cond: bool, what: str) -> None:
        if not cond:
            self.add(what)

    def run(self, what: str, fn, *args):
        """``fn(*args)``, or ``None`` with a failure recorded when it raises:
        a reader that cannot parse a file is a failure of the contract, not of
        this script."""
        try:
            return fn(*args)
        except (Exception, SystemExit) as e:  # every refusal is a finding
            self.add(f"{what}: {type(e).__name__}: {e}")
            return None


def stray_parsers(fail: Failures) -> None:
    here = Path(__file__).resolve()
    allowed = {here, (REPO / "reference_data" / "golden_format.py").resolve()}
    for py in sorted((REPO / "reference_data").glob("*.py")):
        if py.resolve() in allowed:
            continue
        for n, line in enumerate(py.read_text().splitlines(), 1):
            if PY_PARSER.search(line):
                fail.add(f"stray parser: {py.relative_to(REPO)}:{n}: {line.strip()}")
    for root in ("src", "apps"):
        for f in sorted((REPO / root).rglob("*")):
            if f.suffix not in {".cpp", ".cppm", ".h", ".cu", ".cuh"}:
                continue
            if (REPO / "src" / "golden") in f.parents:
                continue
            rel = f.relative_to(REPO).as_posix()
            pattern = CPP_GOLDEN_SIGNS if rel in CPP_NON_GOLDEN_READERS else CPP_PARSER
            for n, line in enumerate(f.read_text().splitlines(), 1):
                code = line.split("//", 1)[0]
                if pattern.search(code):
                    fail.add(f"stray C++ reader: {f.relative_to(REPO)}:{n}: {line.strip()}")


def packed_golden(header: dict, arrays: dict[str, np.ndarray]) -> bytes:
    """The bytes a golden with this header and these arrays has, in the
    layout of its own magic."""
    values = dict(header, reference_kind=gf.REFERENCE_KIND_CODES[header["reference_kind"]])
    buf = io.BytesIO()
    buf.write(gf.pack_header(gf.GOLDEN_HEADERS, header["magic"], values))
    gf.write_arrays(buf, arrays)
    return buf.getvalue()


def same_arrays(a: dict, b: dict, keys) -> bool:
    # NaN-equal: a transient golden's PC placeholders are NaN (pc_on_states).
    return all(k in a and k in b and np.array_equal(a[k], b[k], equal_nan=True)
               for k in keys)


def every_golden_reader(fail: Failures, path: Path, header: dict, arrays: dict) -> None:
    """``path`` through every reader entry point, each held to the canonical
    ``(header, arrays)``."""
    name = path.name
    h = fail.run(f"{name}: golden_format.read_golden_header", gf.read_golden_header, path)
    if h is not None:
        fail.check(h == header, f"{name}: read_golden_header disagrees: {h} != {header}")

    pc = fail.run(f"{name}: nevpt2_numgrad.read_golden_pc_total",
                  ngrad.read_golden_pc_total, path)
    want_pc = None if "e_pc_total" not in arrays else float(arrays["e_pc_total"].ravel()[0])
    fail.check(pc == want_pc or (pc is not None and want_pc is not None
                                 and np.isnan(pc) and np.isnan(want_pc)),
               f"{name}: read_golden_pc_total = {pc}, expected {want_pc}")

    b = fail.run(f"{name}: block2_pc_probe.read_golden", b2p.read_golden, path)
    if b is not None:
        fail.check(
            (b["ncas"], b["ncore"], b["e_casci"], b["e_corr"], b["reference_kind"])
            == (header["ncas"], header["ncore"], header["e_casci"],
                header["e_nevpt2_total"], header["reference_kind"]),
            f"{name}: block2_pc_probe.read_golden's header disagrees",
        )
        keys = [k for k in ("ci", "class_energies", "class_norms") if k in arrays]
        fail.check(same_arrays(b, arrays, keys),
                   f"{name}: block2_pc_probe.read_golden's arrays disagree")

    r = fail.run(f"{name}: pyscf_contract_time.read_golden", pct.read_golden, path)
    if r is not None:
        meta, arr = r
        fail.check(meta == header, f"{name}: pyscf_contract_time's header disagrees")
        fail.check(list(arr) == list(arrays) and same_arrays(arr, arrays, arrays),
                   f"{name}: pyscf_contract_time's arrays disagree")


def non_finite(arrays: dict[str, np.ndarray]) -> list[str]:
    """The arrays holding a NaN or an infinity. No committed golden may hold
    one: a NaN is what ``pc_on_states.py``'s transient goldens carry in place
    of block2's PC fields, so a NaN under ``golden/`` is a transient golden
    that was committed, and every ``--pc`` check against it would fail or
    mean nothing."""
    return [k for k, a in arrays.items() if not np.isfinite(a).all()]


def committed_goldens(fail: Failures, tmp: Path) -> set[bytes]:
    layouts: set[bytes] = set()
    for path in sorted(GOLDEN.glob("*.nevpt2gold")):
        r = fail.run(f"{path.name}: golden_format.read_golden", gf.read_golden, path)
        if r is None:
            continue
        header, arrays = r
        layouts.add(header["magic"])
        bad = non_finite(arrays)
        fail.check(not bad, f"{path.name}: non-finite values in {', '.join(bad)} -- a "
                   "transient golden's NaN placeholders (pc_on_states.py) are never "
                   "committed")
        raw = path.read_bytes()
        packed = packed_golden(header, arrays)
        fail.check(packed == raw, f"{path.name}: re-packed from its "
                   f"{header['magic'].decode()} header table it is not the same bytes")
        fail.check(len(gf.pack_header(gf.GOLDEN_HEADERS, header["magic"], dict(
            header, reference_kind=0))) == header["arrays_at"],
            f"{path.name}: arrays_at {header['arrays_at']} is not where its header ends")
        if header["magic"] == gf.GOLDEN_MAGIC:
            out = tmp / path.name
            gg.write_golden(out, **{k: header[k] for k in (
                "ncas", "nelec_a", "nelec_b", "ncore", "ndet", "e_casci", "e_nevpt2_total",
                "reference_kind")}, arrays=arrays)
            fail.check(out.read_bytes() == raw,
                       f"{path.name}: generate_golden.write_golden does not reproduce it")
            out.unlink()
        every_golden_reader(fail, path, header, arrays)
        print(f"  {path.name}: {header['magic'].decode()}, arrays at byte "
              f"{header['arrays_at']}, {len(arrays)} arrays, {header['reference_kind']}",
              flush=True)
    return layouts


def committed_sidecars(fail: Failures, tmp: Path) -> None:
    flavours: set[bytes] = set()
    for path in sorted(GOLDEN.glob("*.gradgold")):
        r = fail.run(f"{path.name}: golden_format.read_gradgold", gf.read_gradgold, path)
        if r is None:
            continue
        header, arrays = r
        # The magic is the first eight bytes; read_gradgold reports the
        # flavour as the presence of naux, and the two must agree.
        magic = path.read_bytes()[:8]
        flavours.add(magic)
        fail.check(("naux" in header) == (magic == gf.GRADGOLD_MAGIC_DF),
                   f"{path.name}: {magic!r} but naux {'in' if 'naux' in header else 'not in'} "
                   "the header read back")
        out = tmp / path.name
        gg.write_gradgold(out, **header, arrays=arrays)
        fail.check(out.read_bytes() == path.read_bytes(),
                   f"{path.name}: generate_golden.write_gradgold does not reproduce it")
        out.unlink()
        twin = path.with_suffix(".nevpt2gold")
        if twin.exists():
            g, ga = gf.read_golden(twin, want={"B_aa"})
            fail.check((header["nact"], header["ncore"]) == (g["ncas"], g["ncore"]),
                       f"{path.name}: its partition is not its golden's")
            # A density-fitted sidecar sits beside a density-fitted golden (one
            # carrying B_aa) of the same auxiliary basis, and a conventional
            # one beside a conventional golden.
            fail.check(("naux" in header) == ("B_aa" in ga),
                       f"{path.name}: its flavour is not its golden's (density-fitted "
                       "sidecar beside a conventional golden, or the reverse)")
            if "naux" in header and "B_aa" in ga:
                fail.check(ga["B_aa"].shape[0] == header["naux"],
                           f"{path.name}: naux {header['naux']} but its golden's B_aa "
                           f"is {ga['B_aa'].shape}")
        extra = f", naux {header['naux']}" if "naux" in header else ""
        print(f"  {path.name}: {magic.decode()}, {len(arrays)} arrays{extra}", flush=True)
    for magic in gf.GRADGOLD_MAGICS:
        fail.check(magic in flavours, f"no committed {magic.decode()} *.gradgold was read, "
                   "so that flavour is untested")

    for path in sorted(GOLDEN.glob("*.mogold")):
        r = fail.run(f"{path.name}: golden_format.read_mo_sidecar", gf.read_mo_sidecar, path)
        if r is None:
            continue
        header, fields, arrays = r
        side = fail.run(f"{path.name}: generate_golden.read_mo_sidecar",
                        gg.read_mo_sidecar, path)
        if side is not None:
            fail.check(
                all(side[k] == v for k, v in header.items()) and side["fields"] == fields
                and same_arrays(side, arrays, arrays),
                f"{path.name}: generate_golden.read_mo_sidecar disagrees",
            )
        out = tmp / path.name
        gg.write_mo_sidecar(out, **header, fields=fields, arrays=arrays)
        fail.check(out.read_bytes() == path.read_bytes(),
                   f"{path.name}: generate_golden.write_mo_sidecar does not reproduce it")
        out.unlink()
        print(f"  {path.name}: {len(fields)} fields, {len(arrays)} arrays", flush=True)

    oracles = sorted(GOLDEN.glob("*.srcgold"))
    fail.check(bool(oracles), "no committed *.srcgold was read, so that layout is untested")
    kinds: set[tuple[str, str]] = set()
    for path in oracles:
        r = fail.run(f"{path.name}: golden_format.read_srcgold", gf.read_srcgold, path)
        if r is None:
            continue
        header, arrays = r
        kinds.add((header["scheme"], header["flavour"]))
        out = tmp / path.name
        gf.write_srcgold(out, header=header, arrays=arrays)
        fail.check(out.read_bytes() == path.read_bytes(),
                   f"{path.name}: golden_format.write_srcgold does not reproduce it")
        out.unlink()
        # A PC oracle, <case>_pc.srcgold, and a directional one,
        # <case>[_pc]_dir.srcgold (#105), sit beside the same <case>.* as the
        # SC one.
        case = fail.run(f"{path.name}: golden_format.srcgold_case", gf.srcgold_case, path,
                        header["scheme"], header["flavour"])
        if case is None:
            continue
        twin = path.with_name(f"{case}.nevpt2gold")
        fail.check(twin.exists(), f"{path.name}: no golden beside it")
        if twin.exists():
            g, ga = gf.read_golden(twin, want={"ci"})
            fail.check((header["nact"], header["ncore"], header["ndet"])
                       == (g["ncas"], g["ncore"], g["ndet"]),
                       f"{path.name}: its partition is not its golden's")
            fail.check("ci" in arrays and np.array_equal(arrays["ci"], ga["ci"].ravel()),
                       f"{path.name}: its ci is not its golden's, bit for bit")
        side = path.with_name(f"{case}.gradgold")
        if side.exists():
            s, _ = gf.read_gradgold(side)
            fail.check((header["nmo"], header["nvirt"]) == (s["nmo"], s["nvirt"]),
                       f"{path.name}: its n_mo is not its gradient sidecar's")
        print(f"  {path.name}: {header['scheme'].upper()} {header['flavour']}, "
              f"{len(arrays)} arrays", flush=True)
    for scheme, flavour in gf.SRCGOLD_MAGIC_FOR:
        fail.check((scheme, flavour) in kinds,
                   f"no committed {scheme.upper()} {flavour} *.srcgold was read, so that "
                   "kind is untested")

    responses = sorted(GOLDEN.glob("*.respgold"))
    fail.check(bool(responses), "no committed *.respgold was read, so that layout is untested")
    for path in responses:
        r = fail.run(f"{path.name}: golden_format.read_respgold", gf.read_respgold, path)
        if r is None:
            continue
        header, arrays = r
        bad = non_finite(arrays)
        fail.check(not bad, f"{path.name}: non-finite values in {', '.join(bad)}")
        out = tmp / path.name
        fail.run(f"{path.name}: golden_format.write_respgold",
                 lambda o=out, h=header, a=arrays: gf.write_respgold(o, header=h, arrays=a))
        fail.check(out.exists() and out.read_bytes() == path.read_bytes(),
                   f"{path.name}: golden_format.write_respgold does not reproduce it")
        out.unlink(missing_ok=True)
        # <case>_casscf.respgold sits beside <case>_casscf.nevpt2gold / .gradgold,
        # and its directions are written in their frame, bit for bit.
        twin = path.with_suffix(".nevpt2gold")
        fail.check(twin.exists(), f"{path.name}: no golden beside it")
        if twin.exists():
            g, ga = gf.read_golden(twin, want={"ci"})
            fail.check(g["reference_kind"] == "casscf",
                       f"{path.name}: its golden is not a casscf one")
            fail.check((header["nact"], header["ncore"], header["ndet"], header["nelec_a"],
                        header["nelec_b"])
                       == (g["ncas"], g["ncore"], g["ndet"], g["nelec_a"], g["nelec_b"]),
                       f"{path.name}: its partition is not its golden's")
            fail.check(np.array_equal(arrays["ci"], ga["ci"]),
                       f"{path.name}: its ci is not its golden's, bit for bit")
        side = path.with_suffix(".gradgold")
        fail.check(side.exists(), f"{path.name}: no gradient sidecar beside it")
        if side.exists():
            s, sa = gf.read_gradgold(side)
            fail.check((header["nao"], header["nmo"], header["nvirt"])
                       == (s["nao"], s["nmo"], s["nvirt"]),
                       f"{path.name}: its orbital space is not its gradient sidecar's")
            fail.check(np.array_equal(arrays["mo_coeff"], sa["mo_coeff"]),
                       f"{path.name}: its mo_coeff is not its gradient sidecar's, bit for bit")
        print(f"  {path.name}: {gf.RESPGOLD_MAGIC.decode()}, {3 * header['natm']} "
              f"coordinates, {len(arrays)} arrays", flush=True)

    perts = sorted(GOLDEN.glob("*.pertgold"))
    pert_kinds: set[bool] = set()
    fail.check(bool(perts), "no committed *.pertgold was read, so that layout is untested")
    for path in perts:
        r = fail.run(f"{path.name}: golden_format.read_pertgold", gf.read_pertgold, path)
        if r is None:
            continue
        header, arrays = r
        pert_kinds.add(header["naux"] > 0)
        bad = non_finite(arrays)
        fail.check(not bad, f"{path.name}: non-finite values in {', '.join(bad)}")
        out = tmp / path.name
        fail.run(f"{path.name}: golden_format.write_pertgold",
                 lambda o=out, h=header, a=arrays: gf.write_pertgold(o, header=h, arrays=a))
        fail.check(out.exists() and out.read_bytes() == path.read_bytes(),
                   f"{path.name}: golden_format.write_pertgold does not reproduce it")
        out.unlink(missing_ok=True)
        # <stem>.pertgold sits beside <stem>.nevpt2gold / .gradgold, perturbs
        # the same calculation's integrals and is written in their frame.
        twin = path.with_suffix(".nevpt2gold")
        fail.check(twin.exists(), f"{path.name}: no golden beside it")
        if twin.exists():
            g, ga = gf.read_golden(twin, want={"B_aa"})
            fail.check((header["nact"], header["ncore"]) == (g["ncas"], g["ncore"]),
                       f"{path.name}: its partition is not its golden's")
            fail.check((header["naux"] > 0) == ("B_aa" in ga)
                       and (header["naux"] == 0 or ga["B_aa"].shape[0] == header["naux"]),
                       f"{path.name}: naux {header['naux']} is not its golden's calculation")
        side = path.with_suffix(".gradgold")
        fail.check(side.exists(), f"{path.name}: no gradient sidecar beside it")
        if side.exists():
            s, sa = gf.read_gradgold(side)
            fail.check((header["nao"], header["nmo"], header["nvirt"], header["naux"])
                       == (s["nao"], s["nmo"], s["nvirt"], s.get("naux", 0)),
                       f"{path.name}: its orbital space is not its gradient sidecar's")
            fail.check(np.array_equal(arrays["mo_coeff"], sa["mo_coeff"]),
                       f"{path.name}: its mo_coeff is not its gradient sidecar's, bit for bit")
        print(f"  {path.name}: {gf.PERTGOLD_MAGIC.decode()}, {3 * header['natm']} "
              f"coordinates, {len(arrays)} arrays, naux {header['naux']}", flush=True)
    for fitted in (False, True):
        fail.check(fitted in pert_kinds,
                   f"no committed {'density-fitted' if fitted else 'conventional'} *.pertgold "
                   "was read, so that flavour is untested")

    eris = sorted(GOLDEN.glob("*.erigold"))
    fail.check(bool(eris), "no committed *.erigold was read, so that layout is untested")
    for path in eris:
        r = fail.run(f"{path.name}: golden_format.read_erigold", gf.read_erigold, path)
        if r is None:
            continue
        header, arrays = r
        bad = non_finite(arrays)
        fail.check(not bad, f"{path.name}: non-finite values in {', '.join(bad)}")
        out = tmp / path.name
        fail.run(f"{path.name}: golden_format.write_erigold",
                 lambda o=out, h=header, a=arrays: gf.write_erigold(o, header=h, arrays=a))
        fail.check(out.exists() and out.read_bytes() == path.read_bytes(),
                   f"{path.name}: golden_format.write_erigold does not reproduce it")
        out.unlink(missing_ok=True)
        # <stem>.erigold sits beside a CONVENTIONAL <stem>.nevpt2gold /
        # .gradgold and is written in their frame.
        twin = path.with_suffix(".nevpt2gold")
        fail.check(twin.exists(), f"{path.name}: no golden beside it")
        if twin.exists():
            g, ga = gf.read_golden(twin, want={"B_aa"})
            fail.check((header["nact"], header["ncore"]) == (g["ncas"], g["ncore"]),
                       f"{path.name}: its partition is not its golden's")
            fail.check("B_aa" not in ga, f"{path.name}: its golden is density-fitted")
        side = path.with_suffix(".gradgold")
        fail.check(side.exists(), f"{path.name}: no gradient sidecar beside it")
        if side.exists():
            s, sa = gf.read_gradgold(side)
            fail.check("naux" not in s, f"{path.name}: its gradient sidecar is density-fitted")
            fail.check((header["nao"], header["nmo"], header["ncore"], header["nact"],
                        header["nvirt"])
                       == (s["nao"], s["nmo"], s["ncore"], s["nact"], s["nvirt"]),
                       f"{path.name}: its orbital space is not its gradient sidecar's")
            fail.check(np.array_equal(arrays["mo_coeff"], sa["mo_coeff"]),
                       f"{path.name}: its mo_coeff is not its gradient sidecar's, bit for bit")
            a = slice(header["ncore"], header["ncore"] + header["nact"])
            if "g_xa_aa" in sa:
                gap = float(np.abs(arrays["eri"][:, a, a, a] - sa["g_xa_aa"]).max())
                fail.check(gap < 1e-10, f"{path.name}: its (x a | a a) is {gap:.1e} off the "
                           "gradient sidecar's g_xa_aa -- not the same integrals")
        print(f"  {path.name}: {gf.ERIGOLD_MAGIC.decode()}, nmo {header['nmo']}, "
              f"{arrays['eri'].nbytes / 1e6:.2f} MB", flush=True)


def fresh_writer(fail: Failures, tmp: Path) -> None:
    """A small golden from the writer as it is now, both kinds, through every
    reader. Not a fixture standing in for a committed file -- those are read
    above -- but the one case no committed file can cover: a writer change
    that has not been regenerated into ``golden/`` yet."""
    rng = np.random.default_rng(71)
    ndet = 36
    arrays = {
        "ci": rng.standard_normal(ndet),
        "class_energies": rng.standard_normal(8),
        "class_norms": rng.standard_normal(8),
        "h2e": rng.standard_normal((4, 4, 4, 4)),
        "e_pc_total": np.array([-0.25]),
    }
    for kind in gf.REFERENCE_KIND_CODES:
        path = tmp / f"fresh_{kind}.nevpt2gold"
        values = dict(ncas=4, nelec_a=2, nelec_b=2, ncore=5, ndet=ndet,
                      e_casci=-109.0, e_nevpt2_total=-0.2, reference_kind=kind)
        def write(p=path, v=values) -> bool:
            gg.write_golden(p, **v, arrays=arrays)
            return True

        if fail.run(f"{path.name}: generate_golden.write_golden", write) is None:
            continue
        r = fail.run(f"{path.name}: golden_format.read_golden", gf.read_golden, path)
        if r is None:
            continue
        header, got = r
        fail.check(all(header[k] == v for k, v in values.items()),
                   f"{path.name}: the header read back is not the one written")
        fail.check(list(got) == list(arrays) and same_arrays(got, arrays, arrays),
                   f"{path.name}: the arrays read back are not the ones written")
        every_golden_reader(fail, path, header, got)
        print(f"  {path.name}: written by generate_golden.write_golden, read back", flush=True)


def states_hand_off(fail: Failures, tmp: Path) -> None:
    """The CASSCF sweep's hand-off to ``pc_on_states.py`` (#138). A
    ``states.json`` written by ``nevpt2_numgrad._write_manifest`` is read back
    by ``golden_format.read_states_manifest``. A transient golden (``--no-pc``
    arrays plus ``pc_on_states.with_placeholder_pc``'s NaN PC fields, through
    ``generate_golden.write_golden``) goes through every golden reader.
    Neither kind of file is committed, so this is the only place either is
    held to the readers."""
    from types import SimpleNamespace

    d = tmp / "states"
    d.mkdir()
    (d / "ref.mogold").write_bytes(b"")
    r0 = dict(label="ref", atom="H 0 0 0; H 0 0 0.74", s=0.0, e_casci=-1.0, e_corr=-0.1,
              mogold="ref.mogold", tripped=[])
    bad = dict(label="0z+0", atom="H 0 0 0; H 0 0 0.75", s=0.0025, tripped=["energy"],
               key="0z", rung=0, sign=1, task=dict(unit=np.eye(2, 3)))

    def write_manifest() -> bool:
        ngrad._write_manifest(
            d, SimpleNamespace(case="h2", step=0.0025),
            dict(natm=2, basis="sto-3g", ncas=2, nelecas=2, cas_irreps=None), r0, [bad])
        return True

    if fail.run("nevpt2_numgrad._write_manifest", write_manifest) is not None:
        m = fail.run("golden_format.read_states_manifest", gf.read_states_manifest, d)
        if m is not None:
            fail.check([(s["label"], s["kept"]) for s in m["states"]]
                       == [("ref", True), ("0z+0", False)],
                       "read_states_manifest disagrees with _write_manifest")
            print("  states.json: written by nevpt2_numgrad._write_manifest, read back",
                  flush=True)

    path = tmp / "transient.nevpt2gold"
    values = dict(ncas=4, nelec_a=2, nelec_b=2, ncore=5, ndet=36, e_casci=-109.0,
                  e_nevpt2_total=-0.2, reference_kind="casscf")
    arrays = pos.with_placeholder_pc({"ci": np.ones(36), "class_energies": np.zeros(8),
                                      "class_norms": np.zeros(8)})

    def write() -> bool:
        gg.write_golden(path, **values, arrays=arrays)
        return True

    if fail.run(f"{path.name}: generate_golden.write_golden", write) is None:
        return
    r = fail.run(f"{path.name}: golden_format.read_golden", gf.read_golden, path)
    if r is not None:
        header, got = r
        fail.check(list(got) == list(arrays) and same_arrays(got, arrays, arrays),
                   f"{path.name}: the arrays read back are not the ones written")
        every_golden_reader(fail, path, header, got)
        fail.check(non_finite(got) == ["pc_class_energies", "e_pc_total"],
                   f"{path.name}: the committed-golden NaN refusal would not catch it")
        print(f"  {path.name}: a transient golden with NaN PC placeholders, read back "
              "(and refused if committed)",
              flush=True)


#: The C++ writer's integral-adjoint table (#246), read off its source: the
#: names and axis letters must be golden_format.GRADSRC_IBAR's, in order.
GRADSRC_CPP = REPO / "apps" / "sc_pseudodensity" / "gradsrc.cppm"
GRADSRC_CPP_ENTRY = re.compile(r'\{"(\w+)", "([ncav]+)"\}')


def synthetic_gradsrc(seed: int, *, pc: bool, edot: bool, ibar: bool) -> dict:
    """A small dump with every member drawn from ``seed``: n_core 2, n_act 3,
    n_virt 4, a (3, 3) CI vector, two Edot directions."""
    rng = np.random.default_rng(seed)
    sizes = dict(n=9, c=2, a=3, v=4, d=2)
    m: dict[str, np.ndarray] = {}
    groups = [gf.GRADSRC_CORE] + ([gf.GRADSRC_PC] if pc else []) \
        + ([gf.GRADSRC_EDOT] if edot else []) + ([gf.GRADSRC_IBAR] if ibar else [])
    for group in groups:
        for name, axes, dtype in group:
            if axes == "0":
                # the integers are pc's 1 here; the partition is set below
                m[name] = np.array(1 if dtype == "<i8" else rng.standard_normal(), dtype=dtype)
            elif axes == "*":
                m[name] = rng.standard_normal((3, 3))
            else:
                m[name] = rng.standard_normal(tuple(sizes[c] for c in axes))
    m["ncore"], m["nact"], m["nvirt"] = (np.array(v, dtype="<i8") for v in (2, 3, 4))
    return m


def gradsrc_dumps(fail: Failures, tmp: Path, paths: list[Path]) -> None:
    """The ``.gradsrc`` dump (#73, #246): no file of it is committed, so the
    contract is held on synthetic dumps of every flavour -- written by
    ``golden_format.write_gradsrc``, read back by ``read_gradsrc`` and by a
    bare ``numpy.load`` (the consumer), and written again byte for byte -- plus
    any dump the C++ writer produced that is named on the command line (read,
    re-written, compared byte for byte: the two writers are one layout). The
    refusals: a partial integral-adjoint set, a wrong shape, a stray member.
    And the C++ table of the integral adjoint's arrays against this one."""
    cpp = GRADSRC_CPP_ENTRY.findall(GRADSRC_CPP.read_text())
    py = [(n.removeprefix("ibar_"), a) for n, a, _t in gf.GRADSRC_IBAR]
    fail.check(cpp == py, f"{GRADSRC_CPP.relative_to(REPO)}'s kIntegralAdjointNames is not "
               f"golden_format.GRADSRC_IBAR: {cpp} != {py}")
    flavours = [dict(pc=pc, edot=edot, ibar=ibar)
                for pc in (False, True) for edot in (False, True) for ibar in (False, True)]
    for k, fl in enumerate(flavours):
        name = "synthetic " + "/".join(f for f, on in fl.items() if on) if any(fl.values()) \
            else "synthetic core"
        m = synthetic_gradsrc(246 + k, **fl)
        p = tmp / f"synthetic_{k}.gradsrc"
        if fail.run(f"{name}: write_gradsrc", gf.write_gradsrc, p, m) is None and not p.exists():
            continue
        back = fail.run(f"{name}: read_gradsrc", gf.read_gradsrc, p)
        if back is None:
            continue
        fail.check(set(back) == set(m) and all(np.array_equal(back[n], m[n]) for n in m),
                   f"{name}: read_gradsrc does not return what was written")
        with np.load(p) as z:
            fail.check(set(z.files) == set(m) and all(np.array_equal(z[n], m[n]) for n in m),
                       f"{name}: numpy.load does not read what was written")
        q = tmp / f"synthetic_{k}_again.gradsrc"
        gf.write_gradsrc(q, back)
        fail.check(q.read_bytes() == p.read_bytes(), f"{name}: not byte-identical rewritten")
        print(f"  {name}: {len(m)} members round-trip, byte for byte", flush=True)
    full = synthetic_gradsrc(1, pc=False, edot=True, ibar=True)
    for what, bad in (
            ("a partial integral-adjoint set", {n: v for n, v in full.items() if n != "ibar_f"}),
            ("a mis-shaped ibar_heff", {**full, "ibar_heff": np.zeros((3, 3))}),
            ("a stray member", {**full, "ibar_mo_coeff": np.zeros((9, 9))})):
        try:
            gf.write_gradsrc(tmp / "refused.gradsrc", bad)
            fail.add(f"write_gradsrc accepted {what}")
        except gf.FormatError:
            print(f"  refused: {what}", flush=True)
    for path in paths:
        got = fail.run(f"{path.name}: read_gradsrc", gf.read_gradsrc, path)
        if got is None:
            continue
        q = tmp / "cpp_again.gradsrc"
        gf.write_gradsrc(q, got)
        fail.check(q.read_bytes() == path.read_bytes(),
                   f"{path.name}: the C++ writer's bytes are not golden_format's")
        print(f"  {path}: {len(got)} members, C++ writer == golden_format.write_gradsrc, "
              "byte for byte", flush=True)


def parhess_files(fail: Failures, tmp: Path, paths: list[Path]) -> None:
    """The ``.parhess`` parameter Hessian sidecar (#222): an OUTPUT, so none is
    committed. The contract is held on a synthetic one -- written by
    ``golden_format.write_parhess``, read back by ``read_parhess`` and written
    again byte for byte -- plus any file the C++ writer (``writeParHess``)
    produced that is named on the command line, which must be
    ``write_parhess``'s bytes exactly. The refusals: a missing array, a
    mis-shaped one, a partition that does not add up."""
    rng = np.random.default_rng(222)
    header = dict(natm=2, nmo=9, ncore=2, nact=3, nvirt=4, ndet=9)
    arrays = {"coords": np.array([0.0, 2.0, 3.0, 5.0])}
    arrays |= {n: rng.standard_normal((4, 4)) for n, _s in gf.PARHESS_ARRAYS if n != "coords"}
    p = tmp / "synthetic.parhess"
    def write(path, h, a):
        gf.write_parhess(path, header=h, arrays=a)
        return True

    if fail.run("synthetic: write_parhess", write, p, header, arrays) is not None:
        back = fail.run("synthetic: read_parhess", gf.read_parhess, p)
        if back is not None:
            h, a = back
            fail.check(h == header and list(a) == [n for n, _s in gf.PARHESS_ARRAYS]
                       and all(np.array_equal(a[n], arrays[n]) for n in arrays),
                       "synthetic: read_parhess does not return what was written")
            q = tmp / "synthetic_again.parhess"
            gf.write_parhess(q, header=h, arrays=a)
            fail.check(q.read_bytes() == p.read_bytes(), "synthetic: not byte-identical rewritten")
            print(f"  synthetic: {len(a)} arrays round-trip, byte for byte", flush=True)
    for what, hdr, bad in (
            ("a missing h_cc", header, {n: v for n, v in arrays.items() if n != "h_cc"}),
            ("a mis-shaped h_kk", header, {**arrays, "h_kk": np.zeros((5, 4))}),
            ("coordinates out of order", header, {**arrays, "coords": np.array([0., 3, 2, 5])}),
            ("a coordinate past 3 natm", header, {**arrays, "coords": np.array([0., 2, 3, 6])}),
            ("a partition that does not add up", dict(header, nvirt=5), arrays)):
        try:
            gf.write_parhess(tmp / "refused.parhess", header=hdr, arrays=bad)
            fail.add(f"write_parhess accepted {what}")
        except gf.FormatError:
            print(f"  refused: {what}", flush=True)
    for path in paths:
        got = fail.run(f"{path.name}: read_parhess", gf.read_parhess, path)
        if got is None:
            continue
        q = tmp / "cpp_again.parhess"
        gf.write_parhess(q, header=got[0], arrays=got[1])
        fail.check(q.read_bytes() == path.read_bytes(),
                   f"{path.name}: the C++ writer's bytes are not golden_format's")
        print(f"  {path}: natm {got[0]['natm']}, C++ writer == golden_format.write_parhess, "
              "byte for byte", flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # [--gradsrc DUMP...] [--parhess FILE...], in either order.
    lists: dict[str, list[Path]] = {"--gradsrc": [], "--parhess": []}
    current = None
    for a in argv:
        if a in lists:
            current = a
        elif current is None:
            print("usage: golden_layout_check.py [--gradsrc DUMP...] [--parhess FILE...]")
            return 2
        else:
            lists[current].append(Path(a))
    if any(a in lists and not lists[a] for a in argv):
        print("usage: golden_layout_check.py [--gradsrc DUMP...] [--parhess FILE...]")
        return 2
    gradsrc_paths = lists["--gradsrc"]
    parhess_paths = lists["--parhess"]
    fail = Failures()
    print("stray parsers (reference_data/, src/, apps/):", flush=True)
    stray_parsers(fail)
    with tempfile.TemporaryDirectory(prefix="golden_layout_check_") as d:
        tmp = Path(d)
        print("committed goldens:", flush=True)
        layouts = committed_goldens(fail, tmp)
        for magic in gf.GOLDEN_MAGICS:
            fail.check(magic in layouts,
                       f"no committed golden in the {magic.decode()} layout was read, so "
                       "that layout is untested")
        print("committed sidecars:", flush=True)
        committed_sidecars(fail, tmp)
        print("the writer as it is now:", flush=True)
        fresh_writer(fail, tmp)
        print("the CASSCF sweep's hand-off:", flush=True)
        states_hand_off(fail, tmp)
        print("the .gradsrc dump:", flush=True)
        gradsrc_dumps(fail, tmp, gradsrc_paths)
        print("the .parhess sidecar:", flush=True)
        parhess_files(fail, tmp, parhess_paths)
    if fail.items:
        print(f"FAIL: {len(fail.items)} layout-contract failure(s)")
        for item in fail.items:
            print(f"  {item}")
        return 1
    print("PASS: every golden, gradient sidecar, MO sidecar, source-term oracle, "
          "response sidecar, integral perturbation sidecar and full MO-integral tensor "
          "sidecar, the .gradsrc dump's every flavour and the .parhess sidecar, round-trips "
          f"through every reader, layouts {sorted(m.decode() for m in layouts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
