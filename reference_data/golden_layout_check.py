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
   through its reader and back through its writer, byte for byte -- and each
   ``golden/*.srcgold`` source-term oracle (#92) through
   ``golden_format.read_srcgold`` / ``write_srcgold``, held to the golden
   beside it (partition, determinant count and CI vector, bit for bit); all
   four kinds, ``NEVPT2S1`` (SC) and ``NEVPT2P1`` (PC, #95, named
   ``<case>_pc.srcgold``) and their directional flavours ``NEVPT2SD`` /
   ``NEVPT2PD`` (#105, ``<case>[_pc]_dir.srcgold``), must be present.
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
CPP_PARSER = re.compile(r"\bfopen\b|\bfread\b|\bfseek\b|\bseekg\b|\bifstream\b|NEVPT2[GDM]\d")
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
CPP_GOLDEN_SIGNS = re.compile(r"\bfseek\b|\bseekg\b|NEVPT2[GDM]\d")


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
    for path in sorted(GOLDEN.glob("*.gradgold")):
        r = fail.run(f"{path.name}: golden_format.read_gradgold", gf.read_gradgold, path)
        if r is None:
            continue
        header, arrays = r
        out = tmp / path.name
        gg.write_gradgold(out, **header, arrays=arrays)
        fail.check(out.read_bytes() == path.read_bytes(),
                   f"{path.name}: generate_golden.write_gradgold does not reproduce it")
        out.unlink()
        twin = path.with_suffix(".nevpt2gold")
        if twin.exists():
            g = gf.read_golden_header(twin)
            fail.check((header["nact"], header["ncore"]) == (g["ncas"], g["ncore"]),
                       f"{path.name}: its partition is not its golden's")
        print(f"  {path.name}: {len(arrays)} arrays", flush=True)

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


def main() -> int:
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
    if fail.items:
        print(f"FAIL: {len(fail.items)} layout-contract failure(s)")
        for item in fail.items:
            print(f"  {item}")
        return 1
    print("PASS: every golden, gradient sidecar, MO sidecar and source-term oracle "
          "round-trips through "
          f"every reader, layouts {sorted(m.decode() for m in layouts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
