"""Hand cases for the ``.erigold`` full MO-integral tensor sidecar (#180) and
``generate_golden.py --erigold``, which writes it: the format round-trips
byte for byte, refuses a sidecar missing either array, a misshapen array and
a partition that does not add up; ``_parse_args`` refuses ``--erigold``
without ``--gradgold`` and with ``--df``; and ``_full_eri`` -- the generator's
own check -- accepts a tensor that reproduces the arrays it overlaps and
refuses one written in another index order.

The committed sidecar is the layout contract's (``golden_layout_check.py``)
and the regeneration gate's (``devtools/golden-repro.sh``).
"""

from __future__ import annotations

from types import SimpleNamespace

import generate_golden as gg
import golden_format as gf
import numpy as np
import pytest


def _header(nao=7, ncore=1, nact=2, nvirt=3) -> dict:
    return dict(nao=nao, nmo=ncore + nact + nvirt, ncore=ncore, nact=nact, nvirt=nvirt)


def _arrays(h: dict) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(180)
    return {n: rng.standard_normal(tuple(h[d] for d in s)) for n, s in gf.ERIGOLD_ARRAYS}


def test_erigold_round_trip_and_rewrite(tmp_path):
    h = _header()
    a = _arrays(h)
    p = tmp_path / "x.erigold"
    gf.write_erigold(p, header=h, arrays=a)
    hb, ab = gf.read_erigold(p)
    assert hb == h
    assert list(ab) == [n for n, _s in gf.ERIGOLD_ARRAYS]
    for n, v in a.items():
        np.testing.assert_array_equal(ab[n], v, err_msg=n)
    q = tmp_path / "y.erigold"
    gf.write_erigold(q, header=hb, arrays=ab)
    assert q.read_bytes() == p.read_bytes()
    assert p.read_bytes()[:8] == gf.ERIGOLD_MAGIC


@pytest.mark.parametrize("missing", [n for n, _s in gf.ERIGOLD_ARRAYS])
def test_erigold_missing_array_is_refused_whole(tmp_path, missing):
    h = _header()
    a = _arrays(h)
    del a[missing]
    with pytest.raises(gf.FormatError):
        gf.write_erigold(tmp_path / "w.erigold", header=h, arrays=a)
    p = tmp_path / "r.erigold"
    with open(p, "wb") as f:
        f.write(gf.pack_header(gf.ERIGOLD_HEADERS, gf.ERIGOLD_MAGIC, h))
        gf.write_arrays(f, a)
    with pytest.raises(gf.FormatError, match=f"missing {missing}"):
        gf.read_erigold(p)


def test_erigold_refuses_a_misshapen_array_and_a_bad_partition(tmp_path):
    h = _header()
    a = _arrays(h)
    a["eri"] = a["eri"][:, :, :, :-1]
    with pytest.raises(gf.FormatError, match="eri"):
        gf.write_erigold(tmp_path / "s.erigold", header=h, arrays=a)
    with pytest.raises(gf.FormatError, match="partition"):
        gf.write_erigold(tmp_path / "p.erigold", header=dict(h, nmo=h["nmo"] + 1),
                         arrays=_arrays(h))


def test_erigold_is_not_a_gradient_sidecar(tmp_path):
    h = _header()
    p = tmp_path / "x.erigold"
    gf.write_erigold(p, header=h, arrays=_arrays(h))
    with pytest.raises(gf.FormatError, match="not a gradient sidecar"):
        gf.read_gradgold(p)


# --- the flag (#180): parsed only, nothing solved --------------------------------

def test_erigold_with_gradgold_permitted():
    args = gg._parse_args(["--ncas", "4", "--nelecas", "4", "--gradgold", "--erigold"])
    assert args.erigold and args.gradgold


@pytest.mark.parametrize("argv, why", [
    (["--erigold"], "needs --gradgold"),
    (["--df", "--gradgold", "--erigold"], "conventional"),
])
def test_erigold_refused_combinations(argv, why, capsys):
    with pytest.raises(SystemExit):
        gg._parse_args(["--ncas", "4", "--nelecas", "4", *argv])
    assert why in capsys.readouterr().err


# --- the generator's own check: _full_eri on a hand-made state -------------------

def _toy_state(nmo=6, ncore=1, nact=2, seed=5):
    """A symmetric random (pq|rs) behind a stand-in PySCF object, with the
    sidecar rows and golden blocks _gradient_arrays / _integral_blocks would
    slice out of it -- so _full_eri has something to hold its own transform to.
    ao2mo is replaced by the tensor itself."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((nmo,) * 4)
    g = g + g.transpose(1, 0, 2, 3)
    g = g + g.transpose(0, 1, 3, 2)
    g = g + g.transpose(2, 3, 0, 1)
    nocc = ncore + nact
    c, a, v = slice(0, ncore), slice(ncore, nocc), slice(nocc, nmo)
    sp = dict(c=c, a=a, v=v)
    rows = {}
    for p in "avc":
        for q, r in (("a", "a"), ("v", "a"), ("c", "a"), ("c", "v"), ("c", "c")):
            if (q, r) == ("c", "c") and p == "c":
                continue
            rows[f"g_x{p}_{q}{r}"] = g[:, sp[p], sp[q], sp[r]]
    rows["g_xc_cc_j"] = np.einsum("xijj->xij", g[:, c, c, c])
    rows["g_xc_cc_k"] = np.einsum("xjji->xji", g[:, c, c, c])
    rows["g_xa_vv"] = np.einsum("xtrr->xtr", g[:, a, v, v])
    rows["g_xc_vv"] = np.einsum("xirr->xir", g[:, c, v, v])
    blocks = dict(
        h2e=g[a, a, a, a].transpose(0, 2, 1, 3),
        h2e_v_Sr=g[v, a, a, a].transpose(0, 2, 1, 3),
        h2e_v_Si=g[a, c, a, a].transpose(0, 2, 1, 3),
        cvcv=g[c, v, c, v],
        h2e_v_Sijr=g[c, a, c, v].transpose(3, 1, 2, 0),
        h2e_v_Srsi=g[v, a, c, v].transpose(3, 0, 2, 1),
        h2e_v_Srs=g[v, a, v, a].transpose(0, 2, 1, 3),
        h2e_v_Sij=g[c, a, c, a].transpose(1, 3, 0, 2),
        h2e_v1_Sir=g[v, c, a, a].transpose(0, 2, 1, 3),
        h2e_v2_Sir=g[v, a, c, a].transpose(0, 3, 1, 2),
    )
    pt = SimpleNamespace(mo_coeff=np.eye(nmo), ncore=ncore, ncas=nact, mol=None)
    return pt, g, rows, blocks


def test_full_eri_accepts_its_own_tensor_and_refuses_a_transposed_one(monkeypatch):
    from pyscf import ao2mo

    pt, g, rows, blocks = _toy_state()
    monkeypatch.setattr(ao2mo, "full", lambda mol, mo, compact=False: g.reshape(36, 36))
    header, arrays = gg._full_eri(pt, rows, blocks)
    assert header == dict(nao=6, nmo=6, ncore=1, nact=2, nvirt=3)
    np.testing.assert_array_equal(arrays["eri"], g)
    # Physicists' order in place of chemists': the same numbers, the wrong slots.
    monkeypatch.setattr(ao2mo, "full",
                        lambda mol, mo, compact=False: g.transpose(0, 2, 1, 3).reshape(36, 36))
    with pytest.raises(SystemExit, match="refusing to write it"):
        gg._full_eri(pt, rows, blocks)
