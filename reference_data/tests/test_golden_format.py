"""Hand cases for ``golden_format.py``, the one reader (and byte-level writer)
of every binary format in ``golden/``: each magic round-trips through its
writer and reader, and every refusal the module documents fires.

Synthetic files only, in ``tmp_path``. ``golden_layout_check.py`` is the
other half: it pushes the COMMITTED files through every reader; this pins the
readers' behaviour on files built to sit exactly on each edge.
"""

from __future__ import annotations

import struct

import generate_golden as gg
import golden_format as gf
import numpy as np
import pytest

RNG = np.random.default_rng(96)

GOLDEN_STATE = dict(
    ncas=4, nelec_a=2, nelec_b=2, ncore=3, ndet=6,
    e_casci=-108.912345678901, e_nevpt2_total=-0.123456789012,
)
GRAD_HEADER = dict(nao=10, nmo=10, ncore=3, nact=4, nvirt=3, nelec_rhf=14)
MO_HEADER = dict(nao=10, nmo=10, ncore=3, ncas=4, nelec_a=2, nelec_b=2,
                 e_casscf=-108.95)
SRC_HEADER = dict(nmo=10, ncore=3, nact=4, nvirt=3, n_act_occ=2, ndet=6,
                  kappa_step=2.0**-17, ci_step=1e-7)


def _arrays() -> dict[str, np.ndarray]:
    """The shapes the array section has to carry: a one-element vector (how
    ``e_pc_total`` is written), an empty array, a vector, and a non-contiguous
    rank 3 (written in C order regardless). The same values on every call, so a
    read-back compares with a fresh call. Rank 0: see
    :func:`test_rank_zero`."""
    rng = np.random.default_rng(96)
    return {
        "ci": rng.standard_normal(6),
        "one": np.array([3.25]),
        "empty": np.zeros((0, 4)),
        "dm2": rng.standard_normal((2, 3, 4)).transpose(2, 0, 1),
    }


def _assert_arrays_equal(got: dict, want: dict) -> None:
    assert list(got) == list(want)
    for name, arr in want.items():
        assert got[name].shape == arr.shape, name
        assert got[name].dtype == np.float64, name
        np.testing.assert_array_equal(got[name], arr, err_msg=name)


def _write_golden(path, reference_kind="casci", arrays=None, **over):
    gg.write_golden(path, **dict(GOLDEN_STATE, **over), reference_kind=reference_kind,
                    arrays=_arrays() if arrays is None else arrays)


def _write_golden_v1(path, arrays=None, **over):
    """A ``NEVPT2G1`` golden -- the layout every committed CASCI golden is in,
    which no writer produces any more -- packed from the same field table."""
    with open(path, "wb") as f:
        f.write(gf.pack_header(gf.GOLDEN_HEADERS, gf.GOLDEN_MAGIC_V1,
                               dict(GOLDEN_STATE, **over)))
        gf.write_arrays(f, _arrays() if arrays is None else arrays)


def _write_gradgold(path):
    gg.write_gradgold(path, **GRAD_HEADER, arrays=_arrays())


GRAD_HEADER_DF = dict(GRAD_HEADER, naux=17)


def _write_gradgold_df(path):
    gg.write_gradgold(path, **GRAD_HEADER_DF, arrays=_arrays())


def _write_mo_sidecar(path, fields=None, arrays=None):
    gg.write_mo_sidecar(
        path, **MO_HEADER,
        fields={"atom_spec": "N 0 0 0; N 0 0 1.1", "pyscf": "2.14.0",
                "note": "Å, ü -- UTF-8"} if fields is None else fields,
        arrays={"mo_coeff": RNG.standard_normal((10, 10)),
                "geom": RNG.standard_normal((2, 3))} if arrays is None else arrays,
    )


def _write_srcgold(path, scheme="sc", flavour="full"):
    gf.write_srcgold(path, header=dict(SRC_HEADER, scheme=scheme, flavour=flavour),
                     arrays=_arrays())


def _header_len(layouts, magic) -> int:
    return 8 + struct.calcsize(gf._struct_format(layouts[magic]))


# --- round trips, one per magic ---------------------------------------------


@pytest.mark.parametrize("kind", ["casci", "casscf"])
def test_golden_g2_round_trip(tmp_path, kind):
    p = tmp_path / "g.nevpt2gold"
    arrays = _arrays()
    _write_golden(p, reference_kind=kind, arrays=arrays)
    header, got = gf.read_golden(p)
    assert header["magic"] == gf.GOLDEN_MAGIC == b"NEVPT2G2"
    assert header["reference_kind"] == kind
    assert header["arrays_at"] == _header_len(gf.GOLDEN_HEADERS, gf.GOLDEN_MAGIC) == 52
    for k, v in GOLDEN_STATE.items():
        assert header[k] == v, k
    _assert_arrays_equal(got, arrays)
    assert gf.read_golden_header(p) == header


def test_golden_v1_round_trip_reads_as_casci(tmp_path):
    p = tmp_path / "g1.nevpt2gold"
    arrays = _arrays()
    _write_golden_v1(p, arrays=arrays)
    header, got = gf.read_golden(p)
    assert header["magic"] == gf.GOLDEN_MAGIC_V1 == b"NEVPT2G1"
    assert header["reference_kind"] == "casci"
    # The array table starts four bytes earlier than G2's: no reference_kind.
    assert header["arrays_at"] == 48
    for k, v in GOLDEN_STATE.items():
        assert header[k] == v, k
    _assert_arrays_equal(got, arrays)


def test_golden_g1_and_g2_share_the_array_bytes(tmp_path):
    """What golden-repro.sh's G1 -> G2 comparison relies on: the same state
    and arrays written in both layouts differ only by the magic's last byte and
    the four reference_kind bytes."""
    arrays = _arrays()
    _write_golden_v1(tmp_path / "1", arrays=arrays)
    _write_golden(tmp_path / "2", arrays=arrays)
    b1, b2 = (tmp_path / "1").read_bytes(), (tmp_path / "2").read_bytes()
    assert b1[:7] == b2[:7] and b1[8:48] == b2[8:48]
    assert b2[48:52] == struct.pack("<i", 0)
    assert b1[48:] == b2[52:]


def test_golden_rewrite_is_byte_identical(tmp_path):
    p, q = tmp_path / "a", tmp_path / "b"
    _write_golden(p, reference_kind="casscf")
    header, arrays = gf.read_golden(p)
    state = {k: header[k] for k in GOLDEN_STATE}
    gg.write_golden(q, **state, reference_kind=header["reference_kind"], arrays=arrays)
    assert p.read_bytes() == q.read_bytes()


def test_gradgold_round_trip_and_rewrite(tmp_path):
    p, q = tmp_path / "a.gradgold", tmp_path / "b.gradgold"
    _write_gradgold(p)
    header, arrays = gf.read_gradgold(p)
    assert header == GRAD_HEADER  # exactly the six partition fields
    _assert_arrays_equal(arrays, _arrays())
    assert p.read_bytes()[:8] == gf.GRADGOLD_MAGIC == b"NEVPT2D1"
    gg.write_gradgold(q, **header, arrays=arrays)
    assert p.read_bytes() == q.read_bytes()


def test_gradgold_df_header(tmp_path):
    """The density-fitted sidecar (#164): magic ``NEVPT2F1``, the six
    partition fields in ``D1``'s order and then ``naux`` -- seven int32s, so
    the array table starts at byte 36 -- read back with ``naux`` in the header
    and written back byte for byte from it."""
    p, q = tmp_path / "a.gradgold", tmp_path / "b.gradgold"
    _write_gradgold_df(p)
    raw = p.read_bytes()
    assert raw[:8] == gf.GRADGOLD_MAGIC_DF == b"NEVPT2F1"
    assert _header_len(gf.GRADGOLD_HEADERS, gf.GRADGOLD_MAGIC_DF) == 36
    assert struct.unpack_from("<7i", raw, 8) == (10, 10, 3, 4, 3, 14, 17)
    header, arrays = gf.read_gradgold(p)
    assert header == GRAD_HEADER_DF
    assert list(header) == [*GRAD_HEADER, "naux"]
    _assert_arrays_equal(arrays, _arrays())
    gg.write_gradgold(q, **header, arrays=arrays)
    assert raw == q.read_bytes()


def test_gradgold_flavours_differ_only_in_magic_and_naux(tmp_path):
    """The same partition and arrays in both flavours: the D1 file is the F1
    one with its magic's letter D for F and less the four naux bytes."""
    _write_gradgold(tmp_path / "d")
    _write_gradgold_df(tmp_path / "f")
    d, f = (tmp_path / "d").read_bytes(), (tmp_path / "f").read_bytes()
    assert (d[6:7], f[6:7]) == (b"D", b"F")
    assert d[:6] + d[7:32] == f[:6] + f[7:32]
    assert f[32:36] == struct.pack("<i", 17)
    assert d[32:] == f[36:]


def test_mo_sidecar_round_trip(tmp_path):
    p = tmp_path / "m.mogold"
    fields = {"atom_spec": "N 0 0 0; N 0 0 1.1", "empty": "", "note": "Å, ü -- UTF-8"}
    arrays = {"mo_coeff": RNG.standard_normal((10, 10)), "geom": RNG.standard_normal((2, 3))}
    _write_mo_sidecar(p, fields=fields, arrays=arrays)
    header, got_fields, got = gf.read_mo_sidecar(p)
    assert header == MO_HEADER
    assert got_fields == fields and list(got_fields) == list(fields)
    _assert_arrays_equal(got, arrays)
    # generate_golden's reader: the same, flattened, with its own checks passed.
    flat = gg.read_mo_sidecar(p)
    assert flat["fields"] == fields and flat["e_casscf"] == MO_HEADER["e_casscf"]
    np.testing.assert_array_equal(flat["mo_coeff"], arrays["mo_coeff"])


SRC_KINDS = [("sc", "full", b"NEVPT2S1"), ("pc", "full", b"NEVPT2P1"),
             ("sc", "directional", b"NEVPT2SD"), ("pc", "directional", b"NEVPT2PD")]


@pytest.mark.parametrize("scheme, flavour, magic", SRC_KINDS)
def test_srcgold_round_trip_and_rewrite(tmp_path, scheme, flavour, magic):
    p, q = tmp_path / "a.srcgold", tmp_path / "b.srcgold"
    _write_srcgold(p, scheme, flavour)
    assert p.read_bytes()[:8] == magic == gf.SRCGOLD_MAGIC_FOR[(scheme, flavour)]
    header, arrays = gf.read_srcgold(p)
    assert header == dict(SRC_HEADER, scheme=scheme, flavour=flavour)
    _assert_arrays_equal(arrays, _arrays())
    gf.write_srcgold(q, header=header, arrays=arrays)
    assert p.read_bytes() == q.read_bytes()


def test_srcgold_kinds_differ_only_in_the_magic(tmp_path):
    files = []
    for scheme, flavour, _magic in SRC_KINDS:
        _write_srcgold(tmp_path / f"{scheme}_{flavour}", scheme, flavour)
        files.append((tmp_path / f"{scheme}_{flavour}").read_bytes())
    assert len({f[:8] for f in files}) == len(SRC_KINDS)
    assert len({f[8:] for f in files}) == 1


def test_rank_zero(tmp_path):
    """The reader takes a rank-0 array (ndim 0, no extents, one f64). The
    writer cannot produce one: ``np.ascontiguousarray`` promotes a 0-d array to
    shape ``(1,)``, which is why a scalar is written as a one-element vector
    (``e_pc_total``). Pinned, because changing either side changes bytes."""
    p = tmp_path / "g"
    with open(p, "wb") as f:
        f.write(gf.pack_header(gf.GOLDEN_HEADERS, gf.GOLDEN_MAGIC,
                               dict(GOLDEN_STATE, reference_kind=0)))
        f.write(struct.pack("<ii", 1, 1) + b"s" + struct.pack("<id", 0, 3.25))
    got = gf.read_golden(p)[1]["s"]
    assert got.shape == () and got == 3.25
    q = tmp_path / "q"
    _write_golden(q, arrays={"s": np.array(3.25)})
    assert gf.read_golden(q)[1]["s"].shape == (1,)


def test_srcgold_case():
    assert gf.srcgold_case("golden/n2_ccpvdz_cas44.srcgold", "sc") == "n2_ccpvdz_cas44"
    assert gf.srcgold_case("golden/n2_ccpvdz_cas44_pc.srcgold", "pc") == "n2_ccpvdz_cas44"
    with pytest.raises(gf.FormatError, match="must end in _pc"):
        gf.srcgold_case("golden/n2_ccpvdz_cas44.srcgold", "pc")


def test_srcgold_case_directional():
    for scheme, flavour, _magic in SRC_KINDS:
        name = gf.srcgold_name("acrolein_def2svp_cas65", scheme, flavour)
        assert gf.srcgold_case(f"golden/{name}", scheme, flavour) == "acrolein_def2svp_cas65"
    assert gf.srcgold_name("x", "pc", "directional") == "x_pc_dir.srcgold"
    with pytest.raises(gf.FormatError, match="must end in _dir"):
        gf.srcgold_case("golden/x_pc.srcgold", "pc", "directional")
    with pytest.raises(gf.FormatError, match="must end in _pc_dir"):
        gf.srcgold_case("golden/x_dir.srcgold", "pc", "directional")


def test_arrays_at_cli(tmp_path, capsys):
    p = tmp_path / "g.nevpt2gold"
    _write_golden_v1(p)
    assert gf.main(["arrays-at", str(p)]) == 0
    assert capsys.readouterr().out.strip() == "48"
    assert gf.main(["nonsense"]) == 2


# --- refusals ---------------------------------------------------------------

WRITERS = {
    "golden": (_write_golden, gf.read_golden),
    "golden_v1": (_write_golden_v1, gf.read_golden),
    "gradgold": (_write_gradgold, gf.read_gradgold),
    "gradgold_df": (_write_gradgold_df, gf.read_gradgold),
    "mogold": (_write_mo_sidecar, gf.read_mo_sidecar),
    "srcgold_sc": (lambda p: _write_srcgold(p, "sc"), gf.read_srcgold),
    "srcgold_pc": (lambda p: _write_srcgold(p, "pc"), gf.read_srcgold),
    "srcgold_sc_dir": (lambda p: _write_srcgold(p, "sc", "directional"), gf.read_srcgold),
    "srcgold_pc_dir": (lambda p: _write_srcgold(p, "pc", "directional"), gf.read_srcgold),
}


@pytest.mark.parametrize("fmt", WRITERS)
def test_every_truncation_is_refused(tmp_path, fmt):
    """Cut the file at EVERY byte: no prefix may read as a shorter file."""
    write, read = WRITERS[fmt]
    p = tmp_path / "f"
    write(p)
    whole = p.read_bytes()
    read(p)  # the whole file reads
    cut = tmp_path / "cut"
    for n in range(len(whole)):
        cut.write_bytes(whole[:n])
        with pytest.raises(gf.FormatError):
            read(cut)


@pytest.mark.parametrize("fmt", WRITERS)
def test_trailing_bytes_are_refused(tmp_path, fmt):
    write, read = WRITERS[fmt]
    p = tmp_path / "f"
    write(p)
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(gf.FormatError, match="bytes after the last"):
        read(p)


@pytest.mark.parametrize("src", WRITERS)
@pytest.mark.parametrize("dst", WRITERS)
def test_wrong_magic_is_refused(tmp_path, src, dst):
    """Every reader refuses every other format's file by its magic."""
    write, _ = WRITERS[src]
    _, read = WRITERS[dst]
    if read is WRITERS[src][1]:
        return  # same reader: not a wrong magic
    p = tmp_path / "f"
    write(p)
    with pytest.raises(gf.FormatError, match="is not a"):
        read(p)


def test_garbage_magic_is_refused(tmp_path):
    p = tmp_path / "g"
    _write_golden(p)
    p.write_bytes(b"NEVPT2G9" + p.read_bytes()[8:])
    with pytest.raises(gf.FormatError, match="is not a golden file"):
        gf.read_golden_header(p)


def _first_array_offsets(path) -> dict[str, int]:
    """Byte offsets of the first array's fields in a golden."""
    at = gf.read_golden_header(path)["arrays_at"]
    raw = path.read_bytes()
    (name_len,) = struct.unpack_from("<i", raw, at + 4)
    ndim_at = at + 8 + name_len
    return dict(count=at, name_len=at + 4, name=at + 8, ndim=ndim_at, shape=ndim_at + 4)


def _patch(path, offset: int, fmt: str, value) -> None:
    raw = bytearray(path.read_bytes())
    struct.pack_into(fmt, raw, offset, value)
    path.write_bytes(bytes(raw))


@pytest.mark.parametrize("rank, ok", [(gf.MAX_RANK, True), (gf.MAX_RANK + 1, False),
                                      (-1, False)])
def test_rank_bound(tmp_path, rank, ok):
    """A rank of MAX_RANK reads; one above it (or a negative one) is a misread
    header, refused before it sizes anything."""
    assert gf.MAX_RANK == 16
    p = tmp_path / "g"
    if ok:
        arrays = {"ci": RNG.standard_normal(6), "t": np.ones((1,) * rank)}
        _write_golden(p, arrays=arrays)
        _assert_arrays_equal(gf.read_golden(p)[1], arrays)
        return
    _write_golden(p, arrays={"ci": RNG.standard_normal(6)})
    _patch(p, _first_array_offsets(p)["ndim"], "<i", rank)
    with pytest.raises(gf.FormatError, match=f"rank {rank} "):
        gf.read_golden(p)


def test_negative_extent_is_refused(tmp_path):
    p = tmp_path / "g"
    _write_golden(p, arrays={"ci": RNG.standard_normal(6)})
    _patch(p, _first_array_offsets(p)["shape"], "<q", -6)
    with pytest.raises(gf.FormatError, match="negative extent"):
        gf.read_golden(p)


def test_extent_past_the_end_is_refused(tmp_path):
    p = tmp_path / "g"
    _write_golden(p, arrays={"ci": RNG.standard_normal(6)})
    _patch(p, _first_array_offsets(p)["shape"], "<q", 7)
    with pytest.raises(gf.FormatError, match="runs past the end"):
        gf.read_golden(p)


@pytest.mark.parametrize("field, value, msg", [
    ("count", -1, "negative array count"),
    ("name_len", -1, "bad array name length"),
    ("name_len", 1 << 30, "bad array name length"),
])
def test_bad_counts_are_refused(tmp_path, field, value, msg):
    p = tmp_path / "g"
    _write_golden(p)
    _patch(p, _first_array_offsets(p)[field], "<i", value)
    with pytest.raises(gf.FormatError, match=msg):
        gf.read_golden(p)


def test_non_ascii_array_name_is_refused(tmp_path):
    p = tmp_path / "g"
    _write_golden(p, arrays={"ci": RNG.standard_normal(6)})
    raw = bytearray(p.read_bytes())
    raw[_first_array_offsets(p)["name"]] = 0xFF
    p.write_bytes(bytes(raw))
    with pytest.raises(gf.FormatError, match="is not ASCII"):
        gf.read_golden(p)


def test_unknown_reference_kind_is_refused(tmp_path):
    p = tmp_path / "g"
    _write_golden(p)
    _patch(p, 48, "<i", 2)
    with pytest.raises(gf.FormatError, match="unknown reference_kind 2"):
        gf.read_golden_header(p)


def test_ci_must_hold_ndet_elements(tmp_path):
    p = tmp_path / "g"
    _write_golden(p, ndet=7)
    with pytest.raises(gf.FormatError, match="ndet = 7"):
        gf.read_golden(p)
    # ... but only when ci is read: the header alone does not know.
    assert gf.read_golden(p, want={"dm2"})[0]["ndet"] == 7


def test_want_skips_unrequested_arrays(tmp_path):
    p = tmp_path / "g"
    arrays = _arrays()
    _write_golden(p, arrays=arrays)
    _, got = gf.read_golden(p, want={"dm2", "not_there"})
    assert list(got) == ["dm2"]
    np.testing.assert_array_equal(got["dm2"], arrays["dm2"])
    assert gf.read_golden(p, want=set())[1] == {}


def test_want_still_validates_skipped_arrays(tmp_path):
    """A skipped array is walked, not trusted: a broken header behind the one
    array asked for still fails the read."""
    p = tmp_path / "g"
    _write_golden(p, arrays={"ci": RNG.standard_normal(6), "dm1": np.eye(2)})
    raw = p.read_bytes()
    dm1_ndim = raw.index(b"dm1") + 3
    _patch(p, dm1_ndim, "<i", gf.MAX_RANK + 1)
    with pytest.raises(gf.FormatError, match="rank 17"):
        gf.read_golden(p, want={"ci"})


def test_mo_sidecar_negative_field_count_is_refused(tmp_path):
    p = tmp_path / "m"
    _write_mo_sidecar(p)
    _patch(p, _header_len(gf.MO_SIDECAR_HEADERS, gf.MO_SIDECAR_MAGIC), "<i", -1)
    with pytest.raises(gf.FormatError, match="negative field count"):
        gf.read_mo_sidecar(p)


def test_generate_golden_mo_sidecar_refusals(tmp_path):
    """``--mo-from``'s reader turns a wrong file into a SystemExit naming the
    format it wanted, and refuses a sidecar missing or mis-shaping its
    arrays."""
    g = tmp_path / "g"
    _write_golden(g)
    with pytest.raises(SystemExit, match="--mo-from reads what"):
        gg.read_mo_sidecar(g)
    m = tmp_path / "m"
    _write_mo_sidecar(m, arrays={"mo_coeff": np.eye(10)})
    with pytest.raises(SystemExit, match="no geom array"):
        gg.read_mo_sidecar(m)
    _write_mo_sidecar(m, arrays={"mo_coeff": np.eye(9), "geom": np.zeros((2, 3))})
    with pytest.raises(SystemExit, match=r"mo_coeff is \(9, 9\)"):
        gg.read_mo_sidecar(m)


def test_write_golden_refuses_an_unknown_reference_kind(tmp_path):
    with pytest.raises(SystemExit, match="reference_kind 'rhf'"):
        _write_golden(tmp_path / "g", reference_kind="rhf")
