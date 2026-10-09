r"""Time PySCF's NEVPT2 RDM contraction on a committed golden CI vector.

The CPU baseline for the GPU demo's head-to-head: reads the *frozen* golden
`.nevpt2gold` file (the same CI vector and active integrals the CUDA demo
consumes -- no CASCI re-solve, so it is deterministic and directly comparable),
and times PySCF's `make_dm123` (dm1/2/3) + `_contract4pdm` x2 (f3ca/f3ac) -- the
`NEVPTcontract` step, PySCF's analog of the demo's GPU RDM build.

Only PySCF (a `dev` dependency) is imported; the golden reader is a ~15-line
struct parser for the format `generate_golden.write_golden` documents. Not a
pytest test (wall-clock is environment-sensitive); run directly:

    uv run python reference_data/pyscf_contract_time.py \
        [--golden golden/n2_ccpvdz_cas1212.nevpt2gold]

The energy-class contraction (PySCF's other ~5s at CAS(12,12)) is deliberately
not timed here: it needs PySCF's `eris` dict, which the golden's integral
layout does not carry, and it is the minor term -- the RDM build is ~96% of the
NEVPT2 contract at CAS(12,12).
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np


def read_golden(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Parse the little-endian `.nevpt2gold` format (see generate_golden.py)."""
    with open(path, "rb") as f:
        buf = f.read()
    off = 0

    def take(fmt: str):
        nonlocal off
        sz = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, buf, off)
        off += sz
        return vals

    magic = buf[:8]
    off = 8
    if magic != b"NEVPT2G1":
        raise ValueError(f"bad magic {magic!r}")
    ncas, nelec_a, nelec_b, ncore, ndet = take("<iiiiq")
    e_casci, e_nevpt2_total = take("<dd")
    (n_arrays,) = take("<i")
    meta = dict(ncas=ncas, nelec_a=nelec_a, nelec_b=nelec_b, ncore=ncore,
                ndet=ndet, e_casci=e_casci, e_nevpt2_total=e_nevpt2_total)
    arrays: dict[str, np.ndarray] = {}
    for _ in range(n_arrays):
        (name_len,) = take("<i")
        name = buf[off:off + name_len].decode("ascii")
        off += name_len
        (ndim,) = take("<i")
        shape = take(f"<{ndim}q")
        n = int(np.prod(shape)) if shape else 1
        data = np.frombuffer(buf, dtype="<f8", count=n, offset=off).reshape(shape)
        off += n * 8
        arrays[name] = data.copy()
    return meta, arrays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", default=str(
        Path(__file__).resolve().parent.parent / "golden" / "n2_ccpvdz_cas1212.nevpt2gold"))
    args = ap.parse_args()

    from pyscf import fci
    from pyscf.mrpt import nevpt2 as pn

    meta, arrays = read_golden(Path(args.golden))
    ncas = meta["ncas"]
    nelecas = (meta["nelec_a"], meta["nelec_b"])
    na = fci.cistring.num_strings(ncas, nelecas[0])
    nb = fci.cistring.num_strings(ncas, nelecas[1])
    ci = np.ascontiguousarray(arrays["ci"].reshape(na, nb))
    # golden stores h2e = eris["ppaa"][act,act].transpose(0,2,1,3) (physicist);
    # _contract4pdm wants the chemist (aa|aa) = that transpose undone (self-inverse).
    aaaa = np.ascontiguousarray(arrays["h2e"].transpose(0, 2, 1, 3))
    la = fci.cistring.gen_linkstr_index(range(ncas), nelecas[0])
    lb = fci.cistring.gen_linkstr_index(range(ncas), nelecas[1])

    print(f"golden: {Path(args.golden).name}  CAS({ncas},{sum(nelecas)})  "
          f"n_det={meta['ndet']}", flush=True)

    t = time.perf_counter()
    fci.rdm.make_dm123("FCI3pdm_kern_sf", ci, ci, ncas, nelecas)
    t_dm123 = time.perf_counter() - t

    t = time.perf_counter()
    pn._contract4pdm("NEVPTkern_cedf_aedf", aaaa, ci, ncas, nelecas, (la, lb))
    pn._contract4pdm("NEVPTkern_aedf_ecdf", aaaa, ci, ncas, nelecas, (la, lb))
    t_f3 = time.perf_counter() - t

    print(f"PySCF make_dm123 (dm1/2/3):         {t_dm123:8.2f}s", flush=True)
    print(f"PySCF _contract4pdm x2 (f3ca/f3ac): {t_f3:8.2f}s", flush=True)
    print(f"PySCF NEVPTcontract (RDM build):    {t_dm123 + t_f3:8.2f}s", flush=True)


if __name__ == "__main__":
    main()
