"""Hand cases for the ``.parhess`` parameter Hessian sidecar (#222), the
GPU's bilinear form over a response sidecar's directions: the format
round-trips byte for byte, and refuses a sidecar missing any array, a
misshapen matrix, coordinates that are not strictly increasing integers
within ``[0, 3 natm)``, and a partition that does not add up -- on the write
side and on the read side alike.

None is committed (it is an output); the layout contract
(``golden_layout_check.py --parhess``) holds a C++-written one to these bytes.
"""

from __future__ import annotations

import golden_format as gf
import numpy as np
import pytest


def _header(natm=2, ncore=1, nact=2, nvirt=3) -> dict:
    return dict(natm=natm, nmo=ncore + nact + nvirt, ncore=ncore, nact=nact, nvirt=nvirt,
                ndet=4)


def _arrays(coords=(0, 2, 5)) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(222)
    n = len(coords)
    a = {"coords": np.asarray(coords, dtype=np.float64)}
    a |= {name: rng.standard_normal((n, n)) for name, _s in gf.PARHESS_ARRAYS if name != "coords"}
    return a


def _raw(path, header, arrays) -> None:
    """Bytes the checks never saw: what a reader must refuse on its own."""
    with open(path, "wb") as f:
        f.write(gf.pack_header(gf.PARHESS_HEADERS, gf.PARHESS_MAGIC, header))
        gf.write_arrays(f, arrays)


def test_parhess_round_trip_and_rewrite(tmp_path):
    h, a = _header(), _arrays()
    p = tmp_path / "x.parhess"
    gf.write_parhess(p, header=h, arrays=a)
    hb, ab = gf.read_parhess(p)
    assert hb == h
    assert list(ab) == [n for n, _s in gf.PARHESS_ARRAYS]
    for n, v in a.items():
        np.testing.assert_array_equal(ab[n], v, err_msg=n)
    q = tmp_path / "y.parhess"
    gf.write_parhess(q, header=hb, arrays=ab)
    assert q.read_bytes() == p.read_bytes()
    assert p.read_bytes()[:8] == gf.PARHESS_MAGIC


@pytest.mark.parametrize("missing", [n for n, _s in gf.PARHESS_ARRAYS])
def test_parhess_missing_array_is_refused_whole(tmp_path, missing):
    h, a = _header(), _arrays()
    del a[missing]
    with pytest.raises(gf.FormatError):
        gf.write_parhess(tmp_path / "w.parhess", header=h, arrays=a)
    p = tmp_path / "r.parhess"
    _raw(p, h, a)
    with pytest.raises(gf.FormatError, match="expected"):
        gf.read_parhess(p)


@pytest.mark.parametrize("coords", [(2, 0, 5), (0, 2, 2), (0, 2, 6), (-1, 2, 5), (0, 2.5, 5)])
def test_parhess_bad_coordinates_are_refused(tmp_path, coords):
    h, a = _header(), _arrays()
    a["coords"] = np.asarray(coords, dtype=np.float64)
    with pytest.raises(gf.FormatError, match="coords"):
        gf.write_parhess(tmp_path / "w.parhess", header=h, arrays=a)
    p = tmp_path / "r.parhess"
    _raw(p, h, a)
    with pytest.raises(gf.FormatError, match="coords"):
        gf.read_parhess(p)


def test_parhess_misshapen_matrix_and_bad_partition_are_refused(tmp_path):
    h, a = _header(), _arrays()
    with pytest.raises(gf.FormatError, match="h_kc"):
        gf.write_parhess(tmp_path / "s.parhess", header=h, arrays={**a, "h_kc": np.zeros((3, 2))})
    with pytest.raises(gf.FormatError, match="partition"):
        gf.write_parhess(tmp_path / "p.parhess", header=dict(h, nvirt=4), arrays=a)
