r"""Time PySCF's NEVPT2 RDM contraction on a committed golden CI vector.

The CPU baseline for the GPU demo's head-to-head: reads the *frozen* golden
`.nevpt2gold` file (the same CI vector and active integrals the CUDA demo
consumes -- no CASCI re-solve, so it is deterministic and directly comparable),
and times PySCF's `make_dm123` (dm1/2/3) + `_contract4pdm` x2 (f3ca/f3ac) -- the
`NEVPTcontract` step, PySCF's analog of the demo's GPU RDM build.

Only PySCF (a `dev` dependency) is imported; the golden reader is
`golden_format.read_golden`, the one parser of the format
`generate_golden.write_golden` documents. Not a
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
import time
from pathlib import Path

import golden_format as gf
import numpy as np


def read_golden(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """A golden's ``(meta, arrays)``, parsed by
    :func:`golden_format.read_golden` -- either layout. That module, not
    ``generate_golden``, because importing the generator would force its
    single-threading on this timing script; ``golden_format`` sets nothing."""
    return gf.read_golden(path)


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
