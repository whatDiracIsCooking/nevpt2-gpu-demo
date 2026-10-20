# PC-NEVPT2

The partially-contracted variant (`--pc`, `src/energy/energy_pc.cpp`): where
its reference comes from, how the device computes it, how accurate it is and
what it costs. How the stage fits into a run is
[`implementation.md`](implementation.md), §4; this file holds the derivation
details and every measured figure. Each figure names the card, the case and
the flags.

## The block2 reference

PySCF implements only strongly-contracted NEVPT2, so the PC reference comes
from [block2](https://github.com/block-hczhai/block2-preview)'s
`pyblock2.icmr`: `scnevpt2.WickSCNEVPT2` and `icnevpt2_full.WickICNEVPT2`
(PC; the "full" variant builds each class's `H_D` matrix and solves it
densely). `generate_golden.py` writes block2's PC answer into the goldens as
`pc_class_energies` / `e_pc_total` (see
[`reference-data.md`](reference-data.md), "PC fields").
`reference_data/block2_pc_probe.py` runs both block2 kernels on **exactly**
a committed state: it imports `generate_golden._build_state`, so the
single-threading, the 1e-12 Davidson tolerance and point-group symmetry come
along, and it checks that the rebuilt CI vector is bit-identical to the
committed one before comparing anything. CPU only, block2 0.5.4, PySCF
2.14.0.

**The gate: block2 SC reproduces the committed PySCF SC, class by class.**
If it did not, block2 would be seeing a different state, integrals or RDMs,
and a PC number from it would mean nothing. It does, and PC `Sijrs` (no
active index, so nothing to contract differently) equals SC `Sijrs`:

| class | golden SC (PySCF) | block2 SC − golden | block2 PC | PC − SC |
|---|---|---|---|---|
| CAS(10,10) | | | | |
| Sr | −0.105257007542 | 2.8e-10 | −0.109589876721 | −4.3e-3 |
| Si | −0.000187007146 | 9.7e-14 | −0.000187234275 | −2.3e-7 |
| Sijrs | −0.000556837842 | 3.9e-16 | −0.000556837842 | −3.9e-16 |
| Sijr | −0.000302064314 | 0 | −0.000302065524 | −1.2e-9 |
| Srsi | −0.001902551638 | 0 | −0.001902925014 | −3.7e-7 |
| Srs | −0.085184828945 | −6.9e-17 | −0.086215530604 | −1.0e-3 |
| Sij | −0.000059231660 | 9.5e-20 | −0.000059242156 | −1.0e-8 |
| Sir | −0.001202443128 | −9.9e-14 | −0.001202885694 | −4.4e-7 |
| **E_corr** | −0.194651972215 | 2.8e-10 | −0.200016597830 | −5.4e-3 |
| CAS(12,12) | | | | |
| Sr | −0.089638582696 | −4.4e-10 | −0.107847896548 | **−1.8e-2** |
| Si | −0.000183962631 | −6.0e-14 | −0.000200695349 | −1.7e-5 |
| Sijrs | −0.000188793035 | −1.9e-19 | −0.000188793035 | 1.9e-19 |
| Sijr | −0.000064500254 | 7.6e-19 | −0.000064500555 | −3.0e-10 |
| Srsi | −0.000750440432 | −1.1e-19 | −0.000761192726 | −1.1e-5 |
| Srs | −0.072516390621 | 1.4e-17 | −0.079888946981 | −7.4e-3 |
| Sij | −0.000024232093 | 6.3e-17 | −0.000024238809 | −6.7e-9 |
| Sir | −0.000887686793 | 9.6e-14 | −0.000939885777 | −5.2e-5 |
| **E_corr** | −0.164254588555 | −4.4e-10 | −0.189916149779 | −2.6e-2 |

The density-fitted states give the same picture: block2 SC − golden is
≤2.8e-10 per class on `n2_ccpvdz_cas1010_df` and ≤4.4e-10 on
`n2_ccpvdz_cas1212_df`, and PC `Sijrs` − SC `Sijrs` ≤1.1e-19 on both. The
largest SC difference is always Sr's: block2 builds Sr from the dense 4-RDM
where PySCF uses its contracted `f3` digests. It is ≥200× inside the 1e-7
tolerance. PC is below SC in every class but `Sijrs`, as it must be: the PC
space contains the SC function.

Class names map one to one: block2's `sub_eners` keys `r, i, ijrs, ijr, rsi,
rs, ij, ir` are PySCF's `Sr, Si, Sijrs, Sijr, Srsi, Srs, Sij, Sir` (the PC
`sub_spaces` keys `rabcpqg*, iabcpqg*, ijrs±, ijrap±, rsiap±, rsabpq±,
ijabpq±, irabpq1/2` fold onto the same eight).

### What else to know about block2

- **Its linux-x86_64 wheel needs MKL.** It links `libmkl_intel_lp64.so.2`
  and declares `mkl<=2024.2.2` and `intel-openmp` in its own metadata,
  which the sdist and the macOS wheel do not, so a plain lock resolves it
  without them (`ImportError: libmkl_intel_lp64.so.2`). `pyproject.toml`
  names both, marker-gated to linux-x86_64.
- **It re-canonicalizes, and that changes nothing here.** Unless told
  `canonicalized = True`, both kernels call `mc.canonicalize(...,
  cas_natorb=False)`; the golden canonicalizes with `cas_natorb=True`.
  `--orbitals block2` lets block2 do it from the raw CASCI orbitals:
  core/virtual orbitals match the golden's to 3.4e-14 (overlap), and every
  SC and PC class energy matches the golden-orbital run to ≤3.4e-15 at
  CAS(10,10) and ≤7.7e-13 at CAS(12,12) (larger there for the conditioning
  reason below). The default, `--orbitals golden`, hands block2 the golden's
  own orbitals and CI vector with `canonicalized = True`, which is what a PC
  golden does.
- **It is reproducible to the bit.** Two separate processes per state agree
  on every energy and every conditioning statistic the probe records (211
  fields at each size): the golden's single-threading covers block2's
  NumPy/`lstsq` work too. A 24-thread 4-RDM (`--rdm-threads 24`) gives the
  same bits as the single-threaded one.
- **Its cost is the dense 4-RDM.** Single-threaded:

  | | CAS(10,10) | CAS(12,12) |
  |---|---|---|
  | `fci.rdm.make_dm1234` (dense `dm4`, `n^8`) | 66–71 s | 55–56 min (12.6 min with `--rdm-threads 24`) |
  | rest of SC (all eight classes) | ~2 s | ~17 s (+ ~30 s turning `dm4` into `E4`) |
  | rest of PC (all eight classes; Sr dominates) | ~11 s | ~1.5–2 min (Sr alone ~1 min) |
  | peak RSS, SC + PC in one process | 2.5 GiB | 10.3 GiB |

  Both kernels build the 4-RDM themselves, so a plain SC-then-PC run pays for
  it twice; the probe builds it once and hands it to both. The CAS(12,12)
  times are upper bounds (five probes shared 24 cores).
- **It takes DF integrals only by injection.** Its own
  `eri_helper.init_eris` always transforms conventional 4-index AO ERIs,
  while `h1eff` comes from `mc._scf.get_jk`, which is DF-JK on a DF SCF. So
  handed a `DFCASCI` it silently computes a *mixed* answer. And in 0.5.4
  `kernel(eris=...)` / `kernel(pdms=...)` raise `UnboundLocalError` (the
  `teris`/`tpdms` timers are bound only on the build path). The probe and
  the generator therefore replace `eri_helper.init_eris` with one returning a
  `_ChemistsERIs` filled from PySCF's `dfnevpt2._ERIS`, the same
  `ppaa/papa/pacv/cvcv/h1eff` the `_df` golden is built from. With that,
  block2 reproduces the `_df` goldens (above).

### Conditioning of the committed states

block2's `_linear_solve` solves each external tuple's `H_D c = b` with
`np.linalg.lstsq(rcond=None)`, a pseudo-inverse dropping singular values
below `eps · d · s_max`, on `H_D` *alone*; it never forms the metric `S`. The
probe's `--metric` builds `S` with block2's Wick engine and re-solves every
class the textbook way (orthogonalize on `S`'s eigenvectors above a
threshold, solve the projected `H_D`):

| PC block (`d` per tuple) | tuples | modes lstsq drops | metric eigenvalues ≤1e-14·max | smallest kept, `H_D` / `S` (rel.) | ≤1e-8 / ≤1e-6 |
|---|---|---|---|---|---|
| CAS(10,10) | | | | | |
| `rabcpqg*` → Sr (1000) | 16 | 160 | 160 | 3.2e-10 / 2.0e-10 | 1056 / 4688 |
| `iabcpqg*` → Si (1000) | 2 | 20 | 20 | 1.8e-10 / 1.5e-10 | 124 / 600 |
| `irabpq1+2` → Sir (200) | 32 | 32 | 32 | 1.4e-7 / 1.2e-7 | 32 / 416 |
| `rsabpq±` → Srs (55/45) | 136/120 | 0 | 0 | 1.7e-6 / 1.5e-6 | 0 / 0 |
| `ijabpq±` → Sij (55/45) | 3/1 | 0 | 0 | 2.1e-6 / 2.1e-6 | 0 / 0 |
| `rsiap±`, `ijrap±` → Srsi, Sijr (10) | — | 0 | 0 | ≥1.9e-3 | 0 / 0 |
| CAS(12,12) | | | | | |
| `rabcpqg*` → Sr (1728) | 15 | 390 | 360 | **4.5e-13 / 6.4e-13** | 3285 / 9135 |
| `iabcpqg*` → Si (1728) | 1 | 30 | 24 | 9.2e-13 / 1.0e-14 | 281 / 656 |
| `irabpq1+2` → Sir (288) | 15 | 30 | 30 | 4.5e-10 / 3.4e-10 | 180 / 630 |
| `ijabpq+` → Sij (78) | 1 | 0 | 0 | 1.7e-10 / 1.5e-10 | 3 / 6 |
| `rsabpq±` → Srs (78/66) | 120/105 | 0 | 0 | 4.3e-7 / 2.3e-6 | 0 / 0 |
| `rsiap±`, `ijrap+` → Srsi, Sijr (12) | — | 0 | 0 | ≥4.0e-6 | 0 / 0 |

(`ijrs-`, `ijrap-` and `ijabpq-` have no tuples at CAS(12,12): one core
orbital, so no `i<j` pair.) In every class at both sizes the SVD re-solve
reproduces block2's `lstsq` energy to ≤4e-16, and no kept mode has a
non-positive projected `H_D`.

**CAS(10,10) is clean.** The null space is exact and structural: `ncas`
modes per external tuple for Sr and Si, one for Sir, at ~1e-16 relative
(rounding noise around zero), and lstsq drops exactly the modes the metric
calls null. Six orders of magnitude separate them from the smallest real
eigenvalue (1e-16 against 2e-10), and the orthogonalized energy at any
threshold from 1e-14 to 1e-10 equals lstsq's to ≤3.5e-15 in every class. So
on a gapped state, lstsq on `H_D` and orthogonalization are the same number.
The threshold still matters past the gap: cutting the metric at 1e-8 moves Sr
by +1.3e-6, and at 1e-6 by +2.4e-5 (Si: 7.6e-11 / 4.5e-9; Sir: 0 / 2.0e-9),
so a 1e-8 cut would fail the 1e-7 check on Sr.

**N2 CAS(12,12) has no gap, and PC Sr is defined only to ~2e-7 there.** Sr's
metric has twice the structural null count below 1e-14 (24 per tuple, not
12) and then a continuum (495 eigenvalues below 1e-12, 1005 below 1e-10), and
lstsq's own cutoff (`eps · 1728` ≈ 3.8e-13) lands inside it: largest dropped
`H_D` singular value 3.8e-13, smallest kept 4.5e-13. Orthogonalized Sr minus
lstsq Sr is −1.2e-8 / +7.2e-8 / +2.0e-7 at thresholds 1e-14 / 1e-12 / 1e-10
(+2.5e-6 at 1e-8). So the −1.8e-2 PC−SC difference is real, but two
defensible implementations can disagree on Sr by ~2e-7, more than the 1e-7
this project checks to. Si moves ≤5e-10 over the same range; every other
class is gapped.

The likely cause is the active space, not block2. N2 has 14 electrons, so
CAS(12,12) leaves one core orbital and pulls a 1s combination (ε = −15.69 Eh)
into the active space, where it is all but doubly occupied (2 − n = 8.0e-6;
CAS(10,10)'s most occupied active orbital has 2 − n = 3.7e-3). Excitations
that need a hole there have metric eigenvalues of that size and its products
with the other small occupations, and the 12 extra near-null modes per tuple
fit that (one per active index), though the attribution is not proven. **So
the N2 CAS(12,12) goldens carry no PC fields**, and the 12-orbital PC case is
`n2_ccpvdz_cas1012`, CAS(10,12) with both 1s orbitals in the core (627k
determinants against 854k). The SC CAS(12,12) golden is unaffected.

### Reproducing the probe's numbers

From the repo root after `uv sync`. Each run prints its tables and writes
the full record, floats as exact hex, to `--json`. A `bit-identical: False`
on the `state:` line means the environment no longer builds the golden's
state, and nothing after it should be compared. CPU only; the runs are
independent and safe to run concurrently.

```bash
# CAS(10,10), ~2 min each single-threaded
uv run python reference_data/block2_pc_probe.py --json c1010_a.json        # energy table, cost
uv run python reference_data/block2_pc_probe.py --json c1010_b.json        # repeat: compare with _a
uv run python reference_data/block2_pc_probe.py --orbitals block2 --json c1010_b2orb.json
uv run python reference_data/block2_pc_probe.py --df --json c1010_df.json  # vs n2_ccpvdz_cas1010_df
uv run python reference_data/block2_pc_probe.py --which pc --metric --json c1010_metric.json

# CAS(12,12), ~55 min each single-threaded (the 4-RDM)
uv run python reference_data/block2_pc_probe.py --ncas 12 --nelecas 12 --json c1212_a.json
uv run python reference_data/block2_pc_probe.py --ncas 12 --nelecas 12 --json c1212_b.json
uv run python reference_data/block2_pc_probe.py --ncas 12 --nelecas 12 --orbitals block2 --json c1212_b2orb.json
uv run python reference_data/block2_pc_probe.py --ncas 12 --nelecas 12 --df --json c1212_df.json
uv run python reference_data/block2_pc_probe.py --ncas 12 --nelecas 12 --metric --rdm-threads 24 \
    --json c1212_metric_t24.json                            # ~15 min; same bits
```

| claim | run(s) |
|---|---|
| energy tables, PC `Sijrs` = SC `Sijrs` | `c1010_a`, `c1212_a` |
| DF reproduces the `_df` goldens | `c1010_df`, `c1212_df` |
| block2's own canonicalization changes nothing | `_b2orb` vs `_a` (`sc`/`pc` fields) |
| bit-reproducible | `_a` vs `_b`: every JSON field except `t_*`, `rss_*`, `*_times` (hex strings, so equal strings are equal bits) |
| a 24-thread 4-RDM gives the same bits | `c1212_metric_t24` vs `c1212_a` (`sc`/`pc` fields) |
| cost table | the `make_dm1234` / `sc:` / `pc:` lines |
| conditioning tables and cut sensitivities | `c1010_metric`, `c1212_metric_t24` (the indented `metric:` and `orthogonalized E - lstsq E` lines) |

Two figures above come from short scripts rather than the probe:

```bash
# Active natural occupations and orbital energies (the 1s explanation):
PYTHONPATH=reference_data uv run python -c '
import generate_golden as gg, numpy as np
for n in (10, 12):
    mc, pt = gg._build_state("N 0 0 0; N 0 0 1.1", "cc-pvdz", n, n)
    occ = np.sort(np.linalg.eigvalsh(mc.fcisolver.make_rdm1(pt.ci, n, pt.nelecas)))[::-1]
    print(n, "ncore", pt.ncore, "2-n_max %.2e" % (2 - occ[0]),
          "eps_act", np.round(pt.mo_energy[pt.ncore:pt.ncore + n], 3))'

# kernel(eris=...) / kernel(pdms=...) raise UnboundLocalError in block2 0.5.4:
uv run python -c '
from pyscf import gto, scf, mcscf
from pyblock2.icmr import eri_helper, scnevpt2, icnevpt2_full as ic
mc = mcscf.CASCI(scf.RHF(gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", verbose=0)).run(), 4, 4).run()
for m, C in ((scnevpt2, scnevpt2.WickSCNEVPT2), (ic, ic.WickICNEVPT2)):
    x = C(mc); x.verbose = 0; x.kernel()
    for kw, v in (("eris", x.eris), ("pdms", eri_helper.init_pdms(mc, m.pdm_eqs))):
        y = C(mc); y.verbose = 0
        try: y.kernel(**{kw: v}); print(C.__name__, kw, "ok")
        except Exception as e: print(C.__name__, kw, type(e).__name__, e)'
```

## PC-NEVPT2 on the device: the design

Every formula here was checked offline, before the device code existed,
against every golden with PC fields: a numpy script built `dm3`/`f3ac`/`f3ca`
from the CI vector the way PySCF does, used only the intermediates
`energy.cpp` builds, and reproduced block2's `pc_class_energies` in every
class (the same figures the device now gives, below). Rebuilt as quadratic
forms, the same `S`/`K` matrices also reproduce the golden SC
`class_energies` to ≤2e-15, which confirms they are the ones SC contracts.

### One idea: PC is SC without the final contraction

Every SC class computes, for each external tuple `t` (a virtual `r`, a core
`i`, a pair, ...), two quadratic forms of the same integral vector `x_t`, a
slab row:

```
N_t = x_t^T S x_t        ("norm" einsums: dm1/dm2/dm3, hdm*, rm2, dm*_h)
H_t = x_t^T K x_t        ("ener"/"h" einsums: a16, a22, a3, k27, a7, a9, a12/a13)
E_t^SC = -N_t / (Delta_t + H_t / N_t)
```

`S` is the metric of the class's internal functions `|Phi_mu>` (operator
strings over active indices only), `K` is `<Phi_mu|(H_D - E_0)|Phi_nu>`, and
neither depends on the tuple. `Delta_t` is the tuple's orbital-energy sum. SC
uses the single function `sum_mu x_mu |Phi_mu>`; PC uses the whole span:

```
E_t^PC = -b_t^T (K + Delta_t S)^+ b_t,      b_t = S x_t + S_1 h1_t
```

In one dimension this is the SC expression (`b = S x`,
`-b^2/(K + Delta S) = -N/(Delta + H/N)`). `K + Delta_t S` is block2's `H_D`:
`icnevpt2_full` puts the core/virtual orbital energies inside `H_D`, so its
per-tuple `H_D` is exactly `K + Delta_t S`.

So PC needs no new RDM intermediates (see
[`implementation.md`](implementation.md), §4.1, for which `make_a*` it
shares). The PC space is the span of the multi-index functions (block2's
basis). The one-body part of `V|Psi_0>` enters only through the right-hand
side, as its projection onto that span, `S_1 h1_t`, where `S_1` is the metric
column SC's `2 x S_1 h1` norm term already uses. That is why SC's one-body
`K` entries (`a17`, `a19`, `a23`, `a25`) are not needed. SC's Sir `h` has no
one-body term at all, and the PC check still matches there.

### The solve: once per class, not once per tuple

`S` and `K` are the same for every tuple, so each class is diagonalised
once (`solveClass`; the five steps are in
[`implementation.md`](implementation.md), §4.2). Per slab that leaves one
GEMM, `Y (tuples x m) = X_slab (tuples x d) . T (d x m)`, plus a rank-1
update for the one-body column and an elementwise reduction. The slab is
GEMM'd straight into the eigenbasis, so the `IntegralSource` walk, `--batch`
and the DF source carry over unchanged. `Y` is at most a few thousand rows
by `m <= n^3` (Srs at cc-pVTZ: 48^2 × 100), so the finishing reduction stays
on the host, like `normToEnergy`.

**The `d x d` products are BLAS, not the generic einsum.** The four products
in `solveClass` (`(K X)^T`, `K'`, `U^T`, `T = S U`) go through
`wwrblasDgemm` on `DeviceResources`' BLAS handle. Through `deviceEinsum`,
whose kernel does one `atomicAdd` per (output, contracted index), a `d^3`
product at `d = n^3` is ~5e9 atomics: the whole PC stage took 2.98s
(CAS(10,10)) and 17.17s (CAS(10,12)) that way on the RTX 3080, against 0.56s
and 4.06s with BLAS, the class energies agreeing to the printed digits. The
small `t_1` product and the slab einsums stay on the einsum.

The two eigensolves use `wwrsolverDnDsyevd` on `DeviceResources`' dense-solver
handle, bound to the one stream. Its workspace and `devInfo` come from the
pool, and reading `devInfo` back is a `downloadTensor`, which synchronizes.

**Why not the alternatives:**

- **`wwrsolverDnDsygvd`** (the generalised eigensolver) Cholesky-factors `S`,
  and `S` is singular by construction: Sr and Si drop 10 exact null modes at
  CAS(10,10), 16 at CAS(8,8) and 24 at CAS(10,12), and Sir drops 1–2. Two
  `Dsyevd` calls do the same job and let the cut be chosen.
- **block2's per-tuple `lstsq(rcond=None)` on `H_D`** means one SVD of a
  `d x d` matrix per external tuple: `nvirt` SVDs of 1728² for Sr at 12
  orbitals, `nvirt²` of 100² for Srs. The two give the same number exactly
  when `H_D`'s dropped modes are `S`'s null space, which is what a gapped
  metric means; on the committed goldens they agree to ≤6.8e-10 (see
  "Accuracy against block2").

### Symmetrisation

`S` is symmetric as written (≤6e-15 relative, measured). `K` is symmetric
only to the CI vector's convergence, because `<Phi|[H_D,Phi]>` is Hermitian
only for an exact eigenfunction: measured ≤9e-8 relative across the committed
goldens. A quadratic form defines only `K`'s symmetric part, so the device
solves with `K_sym = (K + K^T)/2`.

### The singular-metric convention: cut `S` at `tau = 1e-13 * s_max`

The cut is on `S`'s eigenvalues relative to its largest, once per class, so
it depends neither on the tuple nor on `Delta` (`PC_TAU` in
`energy.cppm`). On every golden with PC fields the structural null modes sit
at ≤1.2e-15 relative and the real modes from ~2e-11 up (cc-pVTZ is the tight
one). On `S` itself, at `tau = 1e-13`, with the dropped count the same at
1e-14 and 1e-12 in every file (so the energy at any cut in that range is the
same number):

| golden | class (d) | dropped | largest dropped | smallest kept | kept / dropped |
|---|---|---|---|---|---|
| `cas44`, `cas44_df` | Sr, Si (64) / Sir (32) | 44 / 2 | ≤3.2e-16 / ≤3.9e-17 | 9.1e-4 / 1.6e-5 | >1e11 |
| `cas88`, `cas88_df` | Sr (512) | 16 | 7.5e-16 | 1.0e-10 | 1.3e5 |
| | Si (512) / Sir (128) | 16 / 2 | ≤2.6e-16 | 3.1e-10 / 1.6e-7 | >1e6 |
| `cas1010`, `cas1010_df` | Sr (1000) | 10 | 1.4e-16 | 2.0e-10 | 1.4e6 |
| | Si (1000) / Sir (200) | 10 / 1 | ≤1.8e-16 | 1.5e-10 / 1.2e-7 | >8e5 |
| `ccpvtz_cas1010_df` | Sr (1000) | 10 | 1.2e-15 | 4.2e-11 | 3.5e4 |
| | Si (1000) | 10 | 9.4e-16 | 2.1e-11 | **2.2e4** |
| | Sir (200) | 1 | 1.0e-15 | 3.7e-8 | 3.7e7 |
| `cas1012` | Sr (1728) / Si (1728) | 24 / 24 | 9.5e-16 / 3.5e-16 | 4.6e-10 / 6.0e-10 | >4.8e5 |
| | Sir (288) | 2 | 2.0e-16 | 2.9e-7 | >1e9 |

Sijr and Srsi (`d = n`) and Srs and Sij (`d = n^2`) drop nothing on any of
these (smallest relative eigenvalue ≥1.6e-7). The cut sits about two decades
above the noise and two below the first real mode on cc-pVTZ. A cut that is
too high drops physics: at `tau = 1e-10`, cc-pVTZ drops 6 more Sr modes, Sr
moves +3.0e-7, and the 1e-7 check fails. (The same reason rules out a
fixed raised cut as the generator's conditioning test: see
[`reference-data.md`](reference-data.md), "PC fields".)

**The device checks the gap rather than assuming it.** Per class it prints
the dropped count, the largest dropped `|s|` and the smallest kept, and
**refuses** the class when smallest kept / largest dropped (or / `tau` when
nothing is dropped) is below `PC_MIN_GAP` = 1e3, the generator's
`_check_pc_conditioning` rule applied to `S`. On N2 CAS(12,12) it would
refuse: an offline run shows the continuum the probe found (largest dropped
9.6e-13, smallest kept 1.1e-12 at `tau = 1e-12`). `--pc` in fact refuses
those files earlier, before the RDM build, because they have no PC fields.

**Every denominator `lambda_k + Delta_t` must be positive** (`H_D` is
positive on the kept range). The device refuses a class rather than divide
through a sign change. The smallest measured is 0.828 Eh (Si, CAS(4,4)); for
Sr it is 1.13 Eh (cc-pVTZ).

### What the cut drops, and how it moves

The cut above is a **threshold** on `S`'s eigenvalues (`pcRankCut`); nothing
in the code decides which modes are structural. What it drops, and how that
space behaves when the CI vector moves, was measured offline with
`reference_data/pc_null_mode_probe.py` (2026-10-20; **no card** — CPU, PySCF
2.14.0, numpy's `eigh`, 6 OpenMP threads on the 12-physical-core Ryzen
5900X). The probe rebuilds every class's `S`, `K` and `S_1` from the golden's
CI vector the way the device does, reproduces the golden's block2
`pc_class_energies` (worst 2.4e-10, CAS(8,8) Sr, as in "Accuracy against
block2"), and uses the RDMs' exact quadratic dependence on the CI vector, so
every derivative of `S` and `K` below is exact; only the reference second
derivative is a difference (Richardson, steps `2^-p` as listed). The theory
these numbers settle is [`hessian-theory.md`](hessian-theory.md), §6.2.

```
uv run python reference_data/pc_null_mode_probe.py golden/n2_ccpvdz_cas44.nevpt2gold
uv run python reference_data/pc_null_mode_probe.py golden/n2_ccpvdz_cas44.nevpt2gold \
    --parts hessian --kinds given \
    --given-u=0.17786578376125828,-0.80668865406914825,-0.16107826557652827,0.90981717673859919 \
    --given-v=-0.063060125646273169,0.68941125217425281,0.59723938945350552,-0.18775891053169191
uv run python reference_data/pc_null_mode_probe.py golden/acrolein_def2svp_cas65.nevpt2gold \
    --kinds Epq,family,sector,singlet --fd-pows 18,19,20
uv run python reference_data/pc_null_mode_probe.py golden/n2_ccpvdz_cas88.nevpt2gold --fd-pows 16,17,18
uv run python reference_data/pc_null_mode_probe.py golden/n2_ccpvdz_cas1010.nevpt2gold --fd-pows 16,17,18
uv run python reference_data/pc_null_mode_probe.py golden/n2_ccpvdz_cas1012.nevpt2gold \
    --parts energies,structure --nrand 4 --nfamily 2         # 9.6 min
uv run python reference_data/pc_null_mode_probe.py golden/n2_ccpvdz_cas1012.nevpt2gold \
    --parts order --kinds Epq                                  # 13.5 min
```

**No dropped mode is null for every CI vector.** Over random vectors of the
whole fixed-`M_s` determinant space no mode is common to all of them (each
such vector still has a nullity of its own where a rank bound applies: 16 in
CAS(4,4) Sr and Si, 25 in acrolein Si, 0 elsewhere). Restricted to
singlets, to singlets of the state's abelian irrep, and to the *family* of
states sharing the Hamiltonian's full symmetry (ground states of
`h1 + b h1^2`, `a h2` near `(a, b) = (1, 0)`), the dropped modes split three
ways. "Null for every" is the nullity of a sum of the vectors' metrics (8
random vectors per kind, 4 at CAS(10,12); the family is sampled from 4
states, 2 at CAS(10,12), so its column is an upper bound); "each" is every
sampled singlet's own nullity:

| golden | class (`d`) | dropped | null for every singlet | each singlet's own | null across the family |
|---|---|---:|---:|---:|---:|
| `cas44` | Sr, Si (64) | 44 | 4 | 44 | 10 |
| | Sir (32) | 2 | 1 | 1 | 2 |
| `acrolein_def2svp_cas65` | Sr (125) | 50 | 5 | 50 | 10 |
| | Si (125) | 85 | 5 | 85 | 14 |
| | Sir (50) | 2 | 1 | 1 (2 in the A' irrep) | 2 |
| `cas88` | Sr, Si (512) | 16 | 8 | 8 | 16 |
| | Sir (128) | 2 | 1 | 1 | 2 |
| `cas1010` | Sr, Si (1000) | 10 | 10 | 10 | 10 |
| | Sir (200) | 1 | 1 | 1 | 1 |
| `cas1012` | Sr, Si (1728) | 24 | 12 | 12 | 24 |
| | Sir (288) | 2 | 1 | 1 | 2 |

- **Spin** (`ncas` modes in Sr and in Si, one in Sir): combinations that
  two orderings of a repeated active index produce identically on an `S = 0`
  state — in Sr's basis `delta_pr e_q - delta_pq e_r`, one per orbital `q`.
  Null for every singlet, not for every `M_s = 0` vector. At CAS(10,10) this
  is everything the cut drops.
- **A rank bound** (CAS(4,4) and acrolein): Sr's functions lie in the
  `(N-1)`-electron doublet space of the active orbitals and Si's in the
  `(N+1)`-electron one, of dimension 20 and 20 at CAS(4,4), 75 and 40 at
  CAS(6,5). Every singlet has nullity `d` minus that (44, 44; 50, 85), but
  only the spin modes are common: the rank is constant, the null space turns
  with the state. CAS(8,8)'s 2352-dimensional space is past `d = 512`, so
  there is no bound there.
- **The state's spatial symmetry**: the rest of CAS(8,8)'s and CAS(10,12)'s
  (8 and 12 in Sr and Si) and Sir's second mode on N2 are null across the
  family but not for a generic singlet of the D2h irrep the CI vector is
  masked to — the linear molecule's full symmetry, which the abelian mask does
  not carry. On acrolein (Cs) the abelian irrep is the whole group: 5 more Sr
  and Si modes, and Sir's second, are null for every A' singlet (Si's family
  column, 14, is a sampled upper bound).

`K` annihilates the dropped space at the state to its convergence:
`|P K|/|K|` ≤ 1.2e-12 except CAS(8,8) (3.6e-10 / 5.0e-10, Sr / Si) and
CAS(10,12) (3.0e-10 / 3.2e-10), whose CI residuals `|Hc - Ec|` are 4.3e-7 and
3.7e-7 against ≤3.5e-15 for the others.

**How the dropped space moves, per kind of direction.** `Epq` is
`nevpt2_pc_tangent`'s direction (`sum w_pq E_pq |c>` over `dm1`'s nonzero
pairs — on these states the number operators of the state's own orbitals);
`family` a family state; `sector` a random singlet of the abelian irrep;
`singlet` one of any irrep (acrolein only: what a non-totally-symmetric
response is). "Coupling" is `|P S_u Q|/|S_u|` (zero: the null space does not
turn); "lift" is the order at which the dropped eigenvalues of
`S(c + h u)` leave zero (none: below 1e-15 relative at `h = 2^-8`), and
`L_uu` the second-order lift's largest eigenvalue over `s_max`
([`hessian-theory.md`](hessian-theory.md), §6.2). `|P S_u P|/|S_u|` is zero
to round-off in every row — ≤1.5e-11, 2.6e-9 along CAS(8,8)'s lifting
`sector` direction — because a Gram metric's dropped eigenvalues never move
at first order; `|P K_u P|/|K_u|` likewise (≤4.0e-11, and ≤3.0e-9 on the two
states with the 4e-7 CI residuals):

| golden, class | `Epq`: coupling, lift | `family` | `sector` | `singlet` |
|---|---|---|---|---|
| `cas44` Sr, Si | 0.56, none | 0.52, none | 0.66, none | |
| `cas44` Sir | 0.04, `h^4` | 0.00, none | 0.03, **`h^2`** (`L_uu` 0.27) | |
| `acrolein` Sr | 0.48, none | 0.43, none | 0.54, none | 0.52, none |
| `acrolein` Si | 0.56, none | 0.58, none | 0.62, none | 0.61, none |
| `acrolein` Sir | 0.00, none | 0.00, none | 0.00, none | 0.04, **`h^2`** (`L_uu` 1.1e-2) |
| `cas88` Sr, Si | 0.02, `h^4` | 0.00, none | 0.04, **`h^2`** (`L_uu` 1.9e-2) | |
| `cas88` Sir | 0.01, `h^4` | 0.00, none | 0.01, **`h^2`** (`L_uu` 3.4e-2) | |
| `cas1010` Sr, Si, Sir | 0.00, none | 0.00, none | 0.00, none | |
| `cas1012` Sr, Si | 0.03, `h^4` | | | |
| `cas1012` Sir | 0.02, `h^4` | | | |

Where nothing lifts at second order (none or `h^4`), `L_uu` is round-off:
≤2.8e-10 over `s_max`, 2.2e-9 at CAS(10,12), and acrolein Sr's 1.6e-8
(`sector`) and 3.4e-8 (`singlet`) — `S^+` (smallest kept 1.8e-10 to 4.6e-10)
amplifying rounding. Every measured `h^2` lift is ≥1.1e-2. The `h^4` modes
sit at 2e-12 to 3e-11 relative at `h = 2^-8` (`Epq`); Sir's is the one "What
the check can and cannot displace" steps around.

`A = K + Delta S`'s own eigenvalues on `S`'s dropped slots are not null off
the eigenstate even where `S`'s are: along `Epq` at `h = 2^-6` / `2^-8`,
1e-6 / 2e-7 of its largest for CAS(4,4) Sr, 2e-5 / 1e-6 for acrolein Sr,
against ≤3e-17 for CAS(4,4) Sir (`S`'s dropped eigenvalues stay ≤1e-15 in
all three). That is the Moore–Penrose inverse of `A` — and block2's
per-tuple `lstsq` on `H_D` — changing rank where the device's `S`-cut solve
does not.

**The second derivative.** `Eddot` without the rotation terms (§6.1 as
first written), with them, and the difference quotient of the device's
two-term `Edot`, each against the Richardson difference of the exact
three-term `Edot` (relative; "ref spread" is the difference between its last
two once-extrapolated values, the reference's own uncertainty):

| golden, class, direction | ref spread | without | repaired | FD of device `Edot` |
|---|---:|---:|---:|---:|
| `cas44` Sr, #152's own `u, v` | 3e-10 | 2.8e-1 | 1.3e-10 | 1.4e-1 |
| `cas44` Sir, #152's own `u, v` | 1e-10 | 1.5e-6 | 5.4e-12 | 9.3e-7 |
| `cas44` Sr, `Epq` / `family` / `sector` | ≤2e-10 | 2.8e-1 / 3.7e-1 / 1.0 | ≤1.8e-10 | 1.4e-1 / 2.0e-1 / 3.4e-2 |
| `cas44` Sir, `Epq` / `sector` | ≤1e-8 | 5.4e-6 / 5.6e-4 | ≤3.9e-10 | 4.5e-6 / 5.1e-4 |
| `acrolein` Sr, `Epq` / `sector` / `singlet` | ≤3e-6 | 1.7e-5 / 2.0 / 7.2e-1 | ≤1.7e-5 | 1.6e-5 / 7.0e-1 / 1.5 |
| `acrolein` Si, `Epq` / `sector` / `singlet` | ≤4e-7 | 5.7e-3 / 1.4e-1 / 1.0e-2 | ≤2.6e-7 | 2.6e-3 / 1.3e-1 / 1.7e-3 |
| `acrolein` Sir, `singlet` | 2e-10 | 2.6e-4 | 1.5e-10 | 2.2e-5 |
| `cas88` Sr, `Epq` / `sector` | ≤5e-7 | 3.1e-7 / 1.3e-5 | ≤3.5e-9 | 7.7e-8 / 9.2e-6 |
| `cas88` Sir, `Epq` / `sector` | ≤7e-9 | 1.8e-5 / 5.8e-6 | ≤3.4e-9 | 1.8e-6 / 3.2e-6 |
| `cas88` every class `family`; `cas1010` every class and kind | ≤5e-6 | = repaired (no rotation) | ≤3.8e-8 | = ref |

The "#152's own" rows take `nevpt2_pc_tangent`'s two seeds' weights
(`--given-u`/`--given-v`, what libc++'s `mt19937_64` draws for `0x5eed0033`
and `0x5eed0152`) and reproduce #152's RTX 3080 run to its printed digits:
without the rotation terms Sr is 0.1265035562 and Sir 4.0340320600, and the
device-`Edot` quotient 0.1508600517 and 4.0340343134 (its Sir row,
4.0340343378, is a coarser ladder). The second derivative of the energy the
device computes is **0.1752129067** (Sr) and **4.0340380951** (Sir): neither
of #152's two numbers. The quotient of the device's `Edot` plateaus because
that `Edot` lacks a term off the eigenstate; the analytic one because it
lacks the rotation; and they miss by different amounts.

**What the rotation terms cost, and what `PC_MIN_GAP` does not see.** They
carry `S^+`, so the smallest kept eigenvalue — not the gap — sets their
size. They reach O(1) of the class's second derivative on the rank-bound
states (CAS(4,4) Sr: up to 1.0; acrolein Sr: up to 2.0) and are exactly zero
where nothing turns. Separately, where the smallest kept `s` is ~1e-10
relative (CAS(8,8)–CAS(10,12) Sr and Si, acrolein Sr) the second derivative
itself is large (|Sr| from 40 to 1.3e4 Eh per unit CI direction squared,
against `E_Sr` of 0.03–0.11 Eh) and short-ranged: the reference difference
needed steps of `2^-16` (N2) and `2^-18` (acrolein) to converge, against
`2^-12` at CAS(4,4) (smallest kept 9.1e-4), and moving the CI vector by 1e-9
moves the repaired value by up to 4e-4 relative (acrolein Sr, `sector`; ≤6e-7
on the N2 states). The gap check (CAS(8,8) Sr 4.2e5, acrolein Sr 7.6e4)
passes all of these.

Along CAS(8,8)'s `sector` direction the fixed-rank continuation is smooth —
its reference converges and the repaired formula matches it to 2.6e-10 —
but it is not the device's energy: with `L_uu` 1.9e-2 the lifted
eigenvalue passes `PC_TAU` at `h ≈ 2^-19`, after which the threshold keeps a
mode the continuation drops.

### Per class

`n` = `ncas`. Layouts are `energy.cppm`'s `ExtBlock` table, and einsum
subscripts are in `energy.cpp`'s convention (what `deviceEinsum` takes; each
one is a GEMM). "Factor" is `c` in `E_t = -c sum_k y_tk^2/(lambda_k +
Delta_t)`. `S`/`K` are in PySCF index order, flattened row-major to `d x d`.

| class | basis, `d` | `S` (and `S_1`) | `K` | slab → `Y` | `Delta_t` | factor |
|---|---|---|---|---|---|---|
| **Sr** | `(p,q,r)`, `n^3` | `dm3` `rpqbac→pqrabc`; `S_1 = dm2` `rpqa→pqra` | `a16` | `Sr (v,a,a,a)` axis 0: `ipqr,pqrk->ik` + `h1e_v_Sr` `ia,ak->ik` | `e_virt[r]` | 1 |
| **Si** | `(p,q,r)`, `n^3` | `dm3_h` `rpqbac→pqrabc`; `S_1 = dm2_h` `rpqa→pqra` | `a22` | `Si (a,a,c,a)` axis 2: `qpir,pqrk->ik` + `h1e_v_Si` `ai,ak->ik` | `-e_core[i]` | 1 |
| **Sijrs** | — | — | — | unchanged: `energy_Sijrs` | | |
| **Sijr** | `p`, `n` | `hdm1` | `a3` | `Sijr (v,a,c,c)` axis 0: `rpji,pk->rjik` | `e_v[r]-e_c[j]-e_c[i]` | see below |
| **Srsi** | `p`, `n` | `dm1` | `k27` | `Srsi (v,v,c,a)` axis 0: `rsip,pk->rsik`; `SrsiT` axis 1: `srip,pk->rsik` | `e_v[r]+e_v[s]-e_c[i]` | see below |
| **Srs** | `(p,q)`, `n^2` | `rm2` `pqba→pqab` | `a7` | `Srs (v,v,a,a)` axis 0: `rsqp,pqk->rsk` | `e_v[r]+e_v[s]` | ½, all ordered `(r,s)` |
| **Sij** | `(p,q)`, `n^2` | `hdm2` | `a9` | `Sij (a,a,c,c)` axis 2: `qpij,pqk->ijk` | `-(e_c[i]+e_c[j])` | ½, all ordered `(i,j)` |
| **Sir** | `(p,q)` ⊕ `(p,q)`, `2n^2` | blocks below; `S_1` | blocks below | `Sir1 (v,a,c,a)` `rpiq,pqk->rik` + `Sir2 (v,a,a,c)` `rpqi,pqk->rik` + `h1e_v_Sir[r,i] t_1[k]` | `e_v[r]-e_c[i]` | 1 |

In each slab einsum the second operand is `T = S U`, reshaped to the basis
indices plus `k`. For Sr, Si and Sir the one-body term uses `t_1 = S_1^T U`
instead.

#### Sr, Si

`d = n^3`: 1000 at CAS(10,10), 1728 at 12 orbitals. `a16`/`a22` are
`make_a16`/`make_a22`, and `dm3_h`/`dm2_h` the ones `energy_Si` builds,
shared with SC rather than copied. `S_1` is `dm2` / `dm2_h` read as
`(pqr, a)`. Each `n^3 x n^3` matrix is 24 MB at `n = 12`; the class holds
about ten of them (`S`, `K`, `S_sym`, `K_sym`, `V`, `X`, `(K X)^T`, `W`, `U`,
`T`, plus `dm3_h` in Si), all freed when the class returns (see "Cost" for
what that does to the pool).

#### Sijr, Srsi: two spin couplings, no explicit +- split

SC's `2·direct − exchange` pattern is the coupling matrix
`[[2,-1],[-1,2]] ⊗ S`. Its eigenvectors are block2's `±` combinations
(weights 1 and 3). Summed over **ordered** pairs the PC energy is

```
Sijr: E = -sum_{r,j,i} sum_k (2 Y[r,j,i,k]^2 - Y[r,j,i,k] Y[r,i,j,k]) / (lambda_k + Delta)
Srsi: E = -sum_{r,s,i} sum_k (2 Y[r,s,i,k]^2 - Y[r,s,i,k] YT[r,s,i,k]) / (lambda_k + Delta)
```

(the `i = j` term comes out as `Y^2`, which is correct), with `YT` from the
`SrsiT` slab. One `n x n` solve covers both couplings. Unlike SC, Srsi needs
no full `(v,v,c)` output buffer: each `r`-batch has both `Y` and `YT` in
hand.

#### Srs, Sij: no +- split either

Each ordered tuple `(r,s)` gets factor ½, on the slab exactly as SC walks
it. For `r ≠ s` the tuples `(r,s)` and `(s,r)` are the same function. For
`r = s`, `x` is symmetric in `(p,q)` and the solve keeps it in that sector,
which is block2's `+` block. So one `n^2 x n^2` solve replaces block2's
`n(n±1)/2` pair of blocks.

#### Sir

`d = 2n^2` (200 / 288). With `(p,q)` row and `(a,b)` column in each block:
`S11 = 2 dm2[q,p,a,b]`, `S12 = S21 = -dm2[q,p,a,b]`, `S22 = 2 δ_pa dm1[q,b] -
dm2[q,b,a,p] + δ_ab dm1[q,p]` (symmetric as a whole), and
`S_1 = [2 dm1[q,p] ; -dm1[q,p]]` against `h1e_v_Sir`. `K11 = 2 a12`,
`K12 = K21 = -a12`, `K22 = a13` (all `[p,q,a,b]`), then symmetrised. These
blocks are SC's nine norm terms and four `h` terms read as matrices. The
matrices are held as `(x, p, q, y, a, b)` with `x, y` the block, and each
block term is one outer-product einsum against a 2×2 pattern
(`[[2,-1],[-1,0]]` for the `dm2`/`a12` terms, `[[0,0],[0,1]]` for
`S22`/`a13`). `reshapeView` lets the `n^2 x n^2` solve and the `(p, q, k)`
slab einsums read the same memory without a copy.

#### Sijrs

No active index, so PC = SC: `pcEnergiesDevice` calls `energy_Sijrs`
itself. block2 agrees to 1e-19.

## Accuracy against block2

**The gate.** `--pc` ends with its own line,

```
PASS: PC-NEVPT2 per-class energies and the total match the golden block2 reference to 1e-07
```

which reads PASS (exit 0) only when all eight classes were computed, none
was refused, every class is within 1e-7 of block2's `pc_class_energies`,
**and** the sum of the eight is within 1e-7 of `e_pc_total`. The total is
checked on its own because eight deltas just under 1e-7 could add up past
it. Otherwise the line reads `FAIL:` and the exit status is 1. It is worded
differently from SC's `PASS:` line, so a `*_pc` ctest entry cannot pass on
SC output, nor an SC entry on PC output. `--pc` on a golden without both PC
fields (the two CAS(12,12) files) is refused before the RDM build.

`|E_corr(PC, ours) − e_pc_total|` and the largest per-class `|delta|`, RTX
3080 / RX 9060 XT, `--tiles 3` (`cas1012`: `--tiles 40`), DF at the default
`--batch 8` except `ccpvtz_cas1010_df` (`--batch 5`):

| golden | total `|delta|` | max class `|delta|` |
|---|---|---|
| `n2_ccpvdz_cas44` | 8.3e-17 / 1.1e-16 | 6.9e-17 / 9.7e-17 |
| `n2_ccpvdz_cas44_df` | 2.8e-17 / 8.3e-17 | 4.9e-17 / 3.1e-17 |
| `n2_ccpvdz_cas88` | 2.69e-10 / 2.69e-10 | 2.44e-10 / 2.44e-10 |
| `n2_ccpvdz_cas88_df` | 2.68e-10 / 2.68e-10 | 2.44e-10 / 2.44e-10 |
| `n2_ccpvdz_cas1010` | 4.31e-11 / 4.31e-11 | 4.31e-11 / 4.31e-11 |
| `n2_ccpvdz_cas1010_df` | 4.34e-11 / 4.34e-11 | 4.34e-11 / 4.34e-11 |
| `n2_ccpvtz_cas1010_df` | 6.80e-10 / 6.80e-10 | 6.80e-10 / 6.80e-10 |
| `n2_ccpvdz_cas1012` | 8.56e-12 / 8.56e-12 | 8.55e-12 / 8.55e-12 |

- **The largest class difference is always Sr's**, the same value on both
  cards and with the same sign (PC below block2). It is the method's, not the
  arithmetic's: block2 builds Sr from the dense 4-RDM and this path from the
  `f3` digests, the same reason block2's own SC Sr sits up to 4.4e-10 off
  PySCF's. The worst, cc-pVTZ, is 147× inside 1e-7. No class was refused
  anywhere.
- **The other classes are far smaller.** Si is ≤2.4e-11 (CAS(8,8)); Srs and
  Sir ≤3.1e-13 (Srs at cc-pVTZ); Sij ≤2.5e-16; Sijrs, Sijr and Srsi ≤1e-16.
  Si at CAS(4,4) is exactly zero, as in SC: the class is empty by symmetry
  ([`testing.md`](testing.md), "The small cases").
- **The device's spectrum matches the cut table above**, the same counts on
  both cards and in the DF twins. The tightest gap is Si at cc-pVTZ, 2.3e4,
  23× above `PC_MIN_GAP`; Sir's gaps run 3.6e7 to 9.4e11, Srs and Sij 1.6e6
  to 6.7e8, Sijr and Srsi 8.3e9 to 2.7e11. The largest dropped eigenvalue is
  noise (≤1.0e-15) and differs between the cards; the counts and the kept
  figures do not.
- **`--batch` does not matter.** `cas1010_df` at `--batch 0` gives the same
  figures to within 3e-16.

**The ctest entries** (all `gpu`, on both backends; the same binaries with
`--pc` and the PC PASS regex):

| entry | golden | flags | label |
|---|---|---|---|
| `nevpt2_cas1010_pc` | `n2_ccpvdz_cas1010` | `--tiles 3` | |
| `nevpt2_cas1012_pc` | `n2_ccpvdz_cas1012` | `--tiles 40` | `slow` |
| `nevpt2_df_cas1010_pc` | `n2_ccpvdz_cas1010_df` | `--tiles 3` (`--batch 8`) | |
| `nevpt2_df_ccpvtz_cas1010_pc` | `n2_ccpvtz_cas1010_df` | `--tiles 3 --batch 5` | |
| `nevpt2_cas44_pc`, `nevpt2_cas88_pc` | `n2_ccpvdz_cas44`, `_cas88` | `--tiles 3` | `small` |
| `nevpt2_df_cas44_pc`, `nevpt2_df_cas88_pc` | their `_df` twins | `--tiles 3` | `small` |
| `nevpt2_cas1010_casscf_pc` | `n2_ccpvdz_cas1010_casscf` | `--tiles 3` | |
| `nevpt2_cas44_casscf_pc`, `nevpt2_df_cas44_casscf_pc` | `n2_ccpvdz_cas44_casscf` and its `_df` twin | `--tiles 3` | `small` |

`nevpt2_cas1012_pc` is `slow` because it takes ~29s on CUDA and ~36s on HIP,
against ≤3s for every other PC entry. The six `small` entries put the PC
stage (the eigensolves, the BLAS GEMMs, the slab einsums) under the
sanitizer tier.

The last three entries read the **CASSCF-orbital** goldens
([`reference-data.md`](reference-data.md), "The MO sidecar: committed CASSCF
orbitals"): nothing in the PC stage sees the reference kind, so they are the
same check on a different state, with block2's PC answer on those orbitals as
the reference and metric gaps decades inside `PC_MIN_GAP`
([`reference-data.md`](reference-data.md), "PC fields").

### The PC total's run-to-run spread

The 10-decimal `E_corr (PC-NEVPT2, ours)` line caps anything that
differences the PC total (a finite-difference gradient oracle) at ~1e-10 Eh.
So beside it `--pc` also prints

```
E_corr full precision (PC-NEVPT2) = <%.17g> Ha
```

(`printPcReport`, so `nevpt2_demo` and `nevpt2_df_demo` alike), and the SC
paths print the same line as `(SC-NEVPT2)` / `(DF-NEVPT2)`. A caller reads
that line, never the human-readable one. Nothing else in the output, and no
gate, changed.

The `atomicAdd` reductions make that total vary run to run. Measured with the
same protocol as the recorded envelope ([`testing.md`](testing.md), "The
kernels' same-run self-variation"): RTX 3080, CUDA, Release (preset
`default`), `nevpt2_demo --golden <case> --pc --tiles <t>`, **30 runs per
case**, back to back in separate processes with the card held exclusively.
Every run was a PASS.

| golden | `--tiles` | runs | min / max of the full-precision total (Eh) | spread (max − min) | distinct values | `|E_corr − e_pc_total|`, min / max |
|---|---|---|---|---|---|---|
| `acrolein_def2svp_cas65` | 3 | 30 | -0.5473408112453471 / -0.5473408112453451 | **2.0e-15** | 16 | 4.53e-14 / 4.73e-14 |
| `acrolein_def2svp_cas65_casscf` | 3 | 30 | -0.5229006180667681 / -0.5229006180667674 | **7.8e-16** | 8 | 8.9e-16 / 1.7e-15 |
| `n2_ccpvdz_cas1012` | 40 | 30 | -0.17488203955444606 / -0.17488203955444437 | **1.7e-15** | 25 | 8.56e-12 / 8.56e-12 |

- **The spread is tens of ulps of the total** (7 to 18 at acrolein's
  0.52-0.55 Eh, ulp 1.1e-16; 61 at N2's 0.17 Eh, ulp 2.8e-17), and in Eh it
  does not grow with the active space: CAS(10,12) at `--tiles 40` varies no
  more than CAS(6,5). It confirms the scouting figures #126 was scoped on
  (acrolein 1.5e-15 over 10 runs; N2 CAS(10,12) below the 10-decimal print
  over 3).
- **What it means for a finite difference.** A central difference at
  `h = 0.0025` bohr scales energy noise by `1/(2h) = 200`, so a 2e-15 Eh
  spread is ~4e-13 Eh/bohr, and ~1.6e-12 at `h/4`: some five and a half
  decades under a ~1e-6 gradient gate.
- **The spread is not the error.** Against block2 the CASCI acrolein total
  sits ~4.6e-14 off and N2 CAS(10,12) 8.56e-12 off (Sr's digest-vs-4-RDM
  difference, "Accuracy against block2" above), both far larger than the
  spread and the same in every run, so they cancel in a difference taken on
  one card.

## Cost

**PC vs SC wall time.** Same binary and flags, `--pc` against none, three
runs each with the card held exclusively, medians of the stage times the
demo prints (`GPU RDM build`, then `GPU energy contraction` for SC or `GPU PC
energy stage` for PC; the DF ones include the slab builds). The three runs
agreed to ≤0.07s, except HIP CAS(10,10)'s SC energy stage (1.01–1.09s).

| case | card | RDM build | SC energy | PC energy | SC run | PC run |
|---|---|---:|---:|---:|---:|---:|
| CAS(4,4) | RTX 3080 | 0.03 | 0.00 | 0.03 | 0.03 | 0.06 |
| CAS(8,8) | RTX 3080 | 0.06 | 0.07 | 0.16 | 0.13 | 0.21 |
| CAS(10,10) | RTX 3080 | 1.05 | 0.36 | 0.54 | 1.41 | 1.59 |
| CAS(10,12), `--tiles 40` | RTX 3080 | 26.6 | 1.47 | 2.03 | 28.04 | 28.90 |
| DF CAS(10,10) | RTX 3080 | 1.01 | 0.37 | 0.55 | 1.38 | 1.56 |
| DF cc-pVTZ CAS(10,10), `--batch 5` | RTX 3080 | 1.01 | 0.54 | 0.57 | 1.55 | 1.58 |
| CAS(4,4) | RX 9060 XT | 0.18 | 0.01 | 0.05 | 0.19 | 0.23 |
| CAS(8,8) | RX 9060 XT | 0.22 | 0.24 | 0.30 | 0.46 | 0.51 |
| CAS(10,10) | RX 9060 XT | 1.39 | 1.04 | 1.21 | 2.42 | 2.61 |
| CAS(10,12), `--tiles 40` | RX 9060 XT | 30.6 | 3.77 | 5.25 | 34.37 | 35.90 |
| DF CAS(10,10) | RX 9060 XT | 1.19 | 1.03 | 1.19 | 2.21 | 2.39 |
| DF cc-pVTZ CAS(10,10), `--batch 5` | RX 9060 XT | 1.18 | 2.28 | 1.50 | 3.47 | 2.70 |

(Each run's RDM build is its own; the column gives the PC run's where the
two differ, by ≤0.05s.)

- **PC's energy stage is slower than SC's in every case but one:** 1.5× /
  1.4× on the RTX 3080 at CAS(10,10) / CAS(10,12), 1.16× / 1.39× on the RX
  9060 XT. Most of it is the two `n^3` classes' `d x d` eigensolves and
  GEMMs, which SC does not have.
- **The exception is cc-pVTZ DF on HIP, where PC is faster:** 1.50s against
  SC's 2.28s. There SC's energy stage is the outlier (2.28s against 1.03s at
  cc-pVDZ, while CUDA goes 0.37s → 0.54s), and the PC stage does not share
  it. Why has not been profiled.
- **End to end, PC costs at most 13% more than SC from CAS(10,10) up**
  (CUDA CAS(10,10), +0.18s), and 3% (CUDA) / 4% (HIP) at CAS(10,12), where
  the RDM build is 92% / 85% of the run. The sub-second CAS(4,4)/(8,8) runs
  cost up to +100% (+0.03–0.08s on a tiny base). The RDM build is the same
  code for both, so the gap closes as the active space grows.

**Memory: `--pc` does not raise the pool's used high-water wherever the RDM
build dominates.** The PC intermediates are freed when each class returns,
after the RDM build's tile blocks are gone, so the whole-run peak is still
the RDM build's. `n2_ccpvdz_cas1012`, pool used / reserved, MB:

| card | `--tiles` | `--pc` | SC |
|---|---|---|---|
| RTX 3080 | 14 (the floor) | 7284.9 / 7296.0 | 7284.9 / 7296.0 |
| RTX 3080 | 40 | 2613.6 / 2624.0 | 2613.6 / 2624.0 |
| RTX 3080 | 150 | 769.0 / 800.0 | 769.0 / 800.0 |
| RX 9060 XT | 7 (the floor) | 14471.6 / 14560.0 | 14471.6 / 14528.0 |
| RX 9060 XT | 14 | 7284.9 / 7456.0 | 7284.9 / 7328.0 |
| RX 9060 XT | 40 | 2613.6 / 2848.0 | 2613.6 / 2656.0 |

- **The tile floor is the same for `--pc` and SC** on both cards. Below it
  (13 on the RTX 3080, with 1.3–1.8 GB of the card held externally; 6 on the
  RX 9060 XT) both fail in the RDM build, before any PC code runs.
- **On HIP the pool reserves 32–192 MB more for `--pc`** at the same used
  peak: fragmentation, not demand.
- **The one exception is CAS(4,4),** where the RDM build's peak is smaller
  than the PC stage's, so `--pc` does raise the used high-water: 1.4 MB
  against SC's 0.4 MB on the RTX 3080, 0.6 MB against 0.4 MB on the RX
  9060 XT.

## The tangent with respect to the CI vector

`pcTangentsDevice` (`src/energy/energy_pc.cpp`, over the `:pc_tangent`
partition in `energy_pc_tangent.cppm`) differentiates every
PC class energy with respect to the CI vector, with the integrals held fixed
— the CI half of the analytic gradient, which is where PC is the branch worth
carrying (Park's Table 3: PC within 5.0e-6 of numerical, SC unstable).
`nevpt2_pc_tangent` (`apps/pc_tangent/`, over `nevpt2.pc_tangent`) is what
checks it.

**It differentiates the solve, not the eigendecomposition.** Only `S`, `K`
and `S_1` carry the CI vector, so with `x = A_t^+ b_t` — the solve the energy
already ran —

```
Edot_t = -c (2 x^T bdot_t - x^T Adot_t x)
```

and no `dU/dh` or `dlambda/dh` appears. `A^+` is a pseudo-inverse, so this is
legitimate only where the rank is locally constant, which is exactly what
`PC_MIN_GAP` enforces (see "The singular-metric convention"): the dropped
modes are exact null modes of the state's metric, three decades below the
smallest kept one, so the cut cannot move under a small change of the state.
(They are not null for every CI vector, and some turn with the state: that
leaves this first derivative exact at the state, and changes the second —
"What the cut drops, and how it moves".) **A class the gap or
denominator check refuses is refused in the tangent too** — never given a
zero derivative. The full argument is in `src/energy/energy.cppm`'s header
and at `tangentClass` in `energy_pc_solve.cppm`.

### What the check can and cannot displace

A derivative of a *solve* has no algebraic identity to check it against (that
is what the SC pseudodensities have — `docs/testing.md`, "The SC
pseudodensity identities"), so the check displaces the state and re-solves.
Three things had to be measured before that worked, all of them on
`n2_ccpvdz_cas44` with the RTX 3080:

- **A random direction raises the metric's rank, at every step size.** `Sr`
  and `Si` drop 44 of 64 modes at the state and only 20 at `c + h u`; the 24
  new modes come in with eigenvalues `~h^2` relative and Dyall eigenvalues
  that make `lambda_k + Delta_t` negative — `minDen` = −7.1 at `h = 2^-3` and
  −6.9 at `h = 2^-8`, so the PC stage refuses the displaced state at *any*
  `h`. That is the pseudo-inverse's rank condition failing, as the derivation
  says it must, and it is a property of the direction. The structural nullity
  is the state's own symmetry: `|ci>` is a spin eigenstate in one spatial
  irrep. The direction is therefore built as `u = sum_pq w_pq E_pq|ci>` over
  the pairs where `dm1[p,q]` is nonzero — the spin-free generators commute
  with `S^2`, and those pairs are the ones whose orbitals share an irrep — and
  then every displaced state comes back with the state's own dropped counts.
  It is also the only kind of direction phase 2 needs: a CI response stays
  inside the sector it was solved in.
- **The path must be the unit sphere, not the straight line.** The hole-side
  metrics carry RDM-free Kronecker-delta terms (`make_hdm1`'s `2 delta`,
  `make_hdm2`'s two `delta delta`), which are the hole RDMs only at unit
  norm. At `norm^2 = 1 + h^2` they are on a different scale from the RDM
  terms, and the difference appears as metric eigenvalues of order `h^2` *of
  the wrong sign*: at `h = 2^-4` on the straight line, `Sij`'s metric grew 7
  eigenvalues at −4.1e-3 relative (= −`h^2`) and the class was refused for its
  gap (0.79). On the normalized path they are gone. With `<c|u> = 0` and
  `||c|| = 1` the two paths have the same tangent at `h = 0`, so nothing is
  lost.
- **`Sir` still grows one mode, at `~h^4`, so the step must be small.** With
  the direction and path above, `Sir` drops 1 of 32 instead of 2 at
  `h = 2^-8`, the new mode sitting at 5.4e-12 relative with `minDen` = −33.7.
  It scales as `h^4`, so below `h ~ 2^-10` it falls under `PC_TAU = 1e-13 *
  s_max` and is dropped like the structural ones. The check runs from
  `h = 2^-12`, where every class comes back with the state's own dropped
  counts and the `h^2` truncation error is ~3e-6 Eh — five decades above the
  difference's own noise floor (~1e-11 Eh: the RDM build's reassociation
  divided by `2h`).

### What it checks, and the measured agreement

Two checks, both in `printPcTangentReport`:

- **Euler, exact and step-free, on five classes.** Where a class's `S`, `K`
  and `S_1` are *linear* in the RDMs, its energy is homogeneous of degree 2
  in the CI vector, so `Edot(c, c) = 2 E` exactly. That is `Sr`, `Si`,
  `Srsi`, `Srs` and `Sir`. It does **not** hold for `Sijr` and `Sij` (the
  hole-side delta terms above make them affine, not linear) or for `Sijrs`
  (no RDM at all, so degree 0 and tangent exactly 0) — those are printed and
  not gated (`kPcHomogeneousInCi`).

  **Five here, three for the SC tangent** (`Sr`, `Srsi`, `Srs` —
  `docs/testing.md`, "The SC tangent's finite-difference cross-check"), and
  the two are consistent: the extra pair is `Si` and `Sir`, and what makes
  them affine on the SC side is machinery PC does not use. SC's `Si` reads
  `a25`, whose `2 h1e^T` carries no RDM, and SC's `Sir` norm has a term that
  is purely one-electron; PC needs neither, because its one-body part enters
  only as `S_1 h1_t` inside `b_t`, and `S_1` is linear in the RDMs. Both
  tangents agree on the other six classes.
- **A central finite difference of the PC total**, `(E(c(h)) - E(c(-h)))/2h`,
  whose error must fall 4× per halving of `h` and whose Richardson
  extrapolation must be the analytic tangent. The PC energy is not a
  polynomial in the CI vector (there is a solve), so unlike the RDMs' own
  tangents this is a genuine second-order difference — the RDMs are quadratic
  in the CI vector, so a straight-line difference of *theirs* has no
  truncation error at all and their own Richardson table needs a curved path
  (`test/rdm_host_dm/`).

RTX 3080, CUDA, the default (emitted) digest, steps halving from `h = 2^-12`:

| case | `--tiles` | steps | ratios | \|Richardson − Edot\| | Euler, worst of the five | \|E − `pcEnergiesDevice`\| | wall |
|---|---:|---:|---|---:|---:|---:|---:|
| CAS(4,4) | 3 | 4 | 4.000, 4.000, 4.000 | 5.6e-12 | 1.5e-16 | 9.0e-17 | 0.2s |
| CAS(8,8) | 3 | 4 | 4.003, 4.001, 4.000 | 1.6e-10 | 8.3e-15 | 3.3e-16 | 2.0s |
| CAS(10,12) | 40 | 3 | 4.002, 4.000 | 7.4e-10 | 5.0e-16 | 1.4e-15 | 318s |

The two right-hand columns, and the Richardson residual, move by up to about
a factor of two run to run — the RDM build's `atomicAdd`s reassociate, so the
energies themselves carry ~1e-15 relative and the difference quotient divides
that by `2h` (the same effect the SC tangent's sweep records in
[`testing.md`](testing.md), "The SC tangent's finite-difference
cross-check"). The ratios do not move at this `h`.

**Where round-off takes over: `h ≈ 2^-20`,** five halvings below `2^-15`, the
finest step any ctest entry takes. Measured 2026-10-11 on the RTX 3080 (CUDA,
Release, the default emitted digest, `--tiles 3`) by sweeping past the
entries' own steps with `--fd-start-pow`, which is what that flag is for.
These are **deliberate diagnostic runs**, and the two that reach past the
crossover FAIL the gate — that is the point of them:

| run | `h` range | what happens |
|---|---|---|
| CAS(4,4), `--fd-start-pow 12 --fd-steps 8` | 2^-12 … 2^-19 | still truncation throughout: ratios 4.000, 4.000, 4.000, 3.999, 3.978, 4.097, 3.649, and \|fd − Edot\| falls 2.7e-6 → 1.8e-10. PASSes |
| CAS(4,4), `--fd-start-pow 17 --fd-steps 8` | 2^-17 … 2^-24 | the last fourfold halving is 2^-19. From **2^-20** the ratios collapse — 1.03, 0.79, 1.09, 2.70, 0.11 — and \|fd − Edot\| floors at 6e-11 to 5.7e-10. Richardson lands at 7.8e-10, still inside its own 1e-8 gate; what fails is the ratio band |
| CAS(8,8), `--fd-start-pow 14 --fd-steps 8` | 2^-14 … 2^-21 | ratios 4.000 to 3.905 down to 2^-19 (\|fd − Edot\| 6.4e-9), then **2^-20** overshoots to 5.8e-11 on a cancellation (ratio 112) and 2^-21 rises again to 2.6e-9. Fails on the ratios too; Richardson 3.4e-9 |

So the default start, `h = 2^-12`, sits eight halvings above the crossover,
and the entries' four steps stay entirely inside the truncation regime — which
is why their ratios read 4.000 and the shape criterion is the right gate. Past
`h ≈ 2^-20` the error is the energies' own reassociation divided by `2h` and
the difference quotient measures nothing; CAS(10,12) was not swept (each step
is two RDM builds at 12 orbitals, see the cost below).

The gate is 1e-8 on the Richardson difference, 4 ± 0.5 on every ratio — and
those ratios a run of at least two, with the gap check's record present and
passing at every displaced state ([`testing.md`](testing.md), "The PC
tangent against a finite difference") — and 1e-9 on the two exact checks, so
each has two decades or more of margin. The
`|E - pcEnergiesDevice|` column is why the tangent partition may build each
class's `S` and `K` itself: the two transcriptions agree to rounding, and a
drift between them fails the entry rather than passing quietly.

**Cost: the CAS(10,12) entry is the slowest in the tree.** Every step is two
displaced states, each a host `dm1`/`dm2` rebuild
(`nevpt2.rdm_host_dm` — the golden's arrays are the state's own) plus a full
device RDM build and PC stage; the three sizes above run 10, 10 and 8 of
them. At CAS(10,12) that is the 318s, nearly all of it RDM builds, which is
why that entry is `slow` and takes `--fd-steps 3`.

### The second derivative, Eddot(u, v)

`pcHessianDevice` (`src/energy/energy_pc.cpp`, over the `:pc_hessian`
partition, #152) is the PC energy's second directional derivative with
respect to the CI vector at fixed orbital energies:
[`hessian-theory.md`](hessian-theory.md), §6.2's repaired formula — §6.1's
terms plus the three rotation terms in `S^+ = U U^T` and the dropped-space
projector `P` — evaluated in the solve's eigenbasis with the solve's own
dropped eigenvectors. Beside it, the same walk returns the **three-term**
`Edot` (the device tangent plus `-2 rho^T S_u P A x`), exact at any CI vector.
`nevpt2_pc_tangent --pc-hessian` checks it. The check runs Euler at second
order on the five homogeneous classes, then a negative control along a random
amplitude vector `w`, then a Richardson-verified central difference of the
three-term `Edot(v)` along `u` on the unit sphere. Its `u` is the tangent
check's own direction (seed `0x5eed0033`) and its `v` is seed `0x5eed0152`,
the pair "What the cut drops, and how it moves" reproduces offline. A
(class, direction pair) whose dropped modes lift is refused. The test is the
first-order block `P S_a P`, or §6.2's `L_ab`, above `PC_LIFT_TOL = 1e-5`
relative to `s_max`, a threshold that sits between that section's measured
noise and its smallest real lift.

RTX 3080, CUDA, `default` preset, emitted digest, `--tiles 3`, steps halving
from `2^-12` (2026-10-20):

| `n2_ccpvdz_cas44` | value |
|---|---|
| `Eddot(u, v)`, Sr / Sir / total | 0.1752129067 / 4.0340380951 / 5.3056290740 Eh |
| Sr / Sir against the offline probe's second derivative (above) | equal to all ten printed digits |
| ratios of the four-step ladder | 4.001, 4.000, 3.999 |
| \|Richardson − `Eddot`\| | 2.6e-11 – 6.2e-11 (two runs; the RDM build's `atomicAdd`s reassociate) |
| Euler, worst of the five | 1.5e-15 |
| \|E − `pcEnergiesDevice`\| / \|three-term − two-term `Edot`\| | ≤2.8e-17 / ≤7.1e-16 |
| lift along `(u, v)`, Sr / Si / Sir | 1.2e-14 / 1.5e-14 / 8.6e-15 |
| lift along the control `(u, w)`, Sr / Si / Sir (refused) | 0.42 / 0.40 / 0.078 |
| smallest kept `s / s_max` (Sir) | 1.6e-5 |
| wall | 0.3 s |

Without the rotation terms the same run printed 0.1265035562 (Sr) and
4.0340320600 (Sir). A difference of the device's two-term `Edot` plateaued at
0.1508600520 (Sr), which is #152's first ladder.

**Why there is no CAS(8,8) entry.** The same run on `n2_ccpvdz_cas88`
(diagnostic, not a ctest entry) has smallest kept `s / s_max` 1.0e-10 (Sr)
and `Eddot` 30.19 Eh. Its ladder is fourfold from `2^-12` (4.009, 4.002,
4.003), but the Richardson value stands 5.5e-6 Eh (1.8e-7 relative) off the
analytic one, both from `2^-12` and from `2^-16`. The two Richardson values
agree with each other to 1.1e-6 Eh, so the difference has converged. What is
left is the analytic formula's eigenstate assumption: `P A = 0` holds only to
the golden's CI residual (4.3e-7, above), and `S^+` amplifies that by up to
`1e10`. The three-term `Edot` differs from the two-term one by 1.3e-11 there,
against 7e-16 at CAS(4,4). Meeting a 1e-8 gate at that conditioning would need
a tighter CI vector, not a different step.

### The Hessian-vector product, H_cc u

`pcCiHessianVectorDevice` (`src/gradient/gradient.cpp`, over the
`:pc_classes` partition, #155) is PC's reverse mode at second order. It takes
the PC adjoint's walk and differentiates it along `u`, then makes two sigma
calls, `H_cc u = sigma(gdot[u], c) + sigma(g, u)`
([`hessian-theory.md`](hessian-theory.md), §4.2). `gdot[u]` comes from the
same per-class blocks `pcHessianDevice` builds (`pcClassDirection` wraps that
path's `tangentClass` and `rotationTangent`). Those blocks are taken in
reverse in `v` (`pcAdjointSecond*`, the reverse of `pcQuadSecond`), with §6.2's
rotation terms, and then through the class's own RDM sweep. The `y` it makes
is the **three-term** one: the device adjoint plus `-2 rho^T S_v P A x`. That
`y` is the gradient at any CI vector, and `pcCiSourceThreeTermDevice` returns
it alone. `pcCiSourceDevice`'s two-term `y` is unchanged.

`nevpt2_pc_tangent --pc-hessian-vector` checks it ([`testing.md`](testing.md),
"The PC Hessian-vector product's checks", has the gates). `u` and the first
`v` are `--pc-hessian`'s seeds; the other two `v` are `0x5eed1155` and
`0x5eed2155`, and the control `w` is `0x5eedc155`. RTX 3080, CUDA, `default`
preset, emitted digest, `--tiles 3`, steps halving from `2^-12`, five runs
(2026-10-20; the RDM build's `atomicAdd`s reassociate, so the residuals move
run to run):

| `n2_ccpvdz_cas44` | value |
|---|---|
| `‖H_cc u‖`, Sr / Sir (Sijrs and Si: exactly 0) | 0.3266513567 / 5.0565814852 |
| `<v, H_cc u>` against `Eddot(u, v)`, worst over classes, four `v` | 3.0e-13 relative (≤5.3e-14 in four of the five runs) |
| `<v, H_cc u>` against `<u, H_cc v>` | ≤8.9e-14 (on 5.3056290740 Eh) |
| the walk's own `Edot(u)` / `<u, y>` against `pcHessianDevice`'s | ≤6.4e-16 / ≤7.4e-16 |
| Euler `‖H_cc c - y‖ / ‖y‖`, worst of the five homogeneous classes | 4.9e-14 (Sir) |
| ratios of the four-step ladder, every class and the total | 3.988 – 4.018 |
| Richardson against `H_cc u`, relative: Sr / Sir / worst class (Sij) / total | ≤1.3e-11 / ≤2.0e-11 / ≤1.8e-10 / ≤1.2e-10 |
| the same, projected on the first `v` (`--pc-hessian`'s difference of `Edot(v)`, read through `y`) | 2.6e-11 – 3.4e-11 Eh |
| lift along `(u, u)`, Sr / Si / Sir | ≤3.4e-15 / 1.4e-14 / ≤7.2e-16 |
| lift along the control `w`, Sr / Si / Sir (refused) | 0.43 / 0.40 / 0.098 |
| wall | 0.9 s |

**With either piece removed, the difference fails.** These are two diagnostic
runs, each with one term switched off by a temporary edit that is not in the
tree. Both used the same card, flags and seeds.

- **Difference the two-term `y` instead.** Sr's error stays at 0.94 of
  `‖H_cc u‖` at every step (ratio 1.000). Sir's ratios fall from 3.92 to
  3.16 and then 1.51, and its Richardson value misses by 1.3e-6 relative.
  The duality still passes, because neither of its two sides moved.
- **Drop the rotation terms from `gdot[u]`.** The duality fails against
  `pcHessianDevice`: Sr by 4.9e-2 – 5.4e-2 relative, Sir by 1.5e-6. The
  difference plateaus: Sr at 1.9 relative, and Sir's Richardson value misses
  by 3.5e-6.

**Why there is no CAS(8,8) entry.** `n2_ccpvdz_cas88` fails the same way, and
for the same reason, as "The second derivative, Eddot(u, v)" above (three
diagnostic runs, not a ctest entry, 51 s each). Its smallest kept
`s / s_max` is 1.0e-10 (Sr), and `‖H_cc u‖` is 1.0e4 for Sr. The ladder is
fourfold from `2^-12` for every class but Srs, whose last ratio is 2.48 at
its kept `s` of 5.6e-7. Sr's Richardson value is 4.2e-6 relative off `H_cc u`.
From `2^-16` the difference stays 4.3e-6 – 5.0e-6 off: it has converged.
Projected on the first `v`, the offset is 5.7e-6 Eh. That is
`--pc-hessian`'s own 5.5e-6 Eh above, the same quantity read a second way.
The duality's worst class is ≤3.4e-9 relative, the symmetry ≤1.7e-7 Eh on
30.19 Eh, and Euler 1.1e-7 (Sr). The formula is §6.2's at the eigenstate,
which holds only to the golden's CI residual (4.3e-7), and `S^+` amplifies
that by up to `1e10`. The fix would be a tighter CI vector, not a looser gate.

## The source terms by reverse mode

#74 gives PC what SC has had since #73–#89: `y_I = dE_pc/dc_I` and
`Y_xy = dE_pc/dkappa_xy` by the **adjoint of the PC solve**, with both of the
dependences the #73 amendment names — the `f3ac`/`f3ca` digests in `Y`
(their transpose, #89) and the orbital energies in `y` (#87) — and the forward
modes they are held against. The derivation is
[`gradients.md`](gradients.md), §5.6 and §6.1, and the module header of
`src/gradient/gradient_pc_classes.cppm`; it rests on the singular-metric
convention above exactly as the tangent does, so the adjoint exists where the
energy accepts a class and is refused where it does not.

Unless a row says otherwise: Release, `--tiles 3`, the default digest of each
backend (emitted on the RTX 3080, preset `default`; native-fp64 BLAS on the
RX 9060 XT, preset `hip`).

### The dualities

Both modes of each derivative are the device's own, so their agreement is
exact arithmetic and the gate is the tree's 1e-7:

| | N2 CAS(4,4) | N2 CAS(8,8) | acrolein CAS(6,5) |
|---|---:|---:|---:|
| CI: `<u, y>` vs `Edot_pc(u)`, 4 seeds + the state + (CAS(4,4)) all 36 Jacobian columns, worst, RTX 3080 | 1.0e-15 | 1.9e-11 | — |
| the same, RX 9060 XT | 9.2e-16 | 3.6e-11 | — |
| rotation, integral halves (dump check 5), 6 seeded full directions, worst, RTX 3080 | 6.9e-18 | 1.2e-14 | 2.6e-16 |
| rotation, full duality with both digests' terms, worst, RTX 3080 | 6.1e-18 | 4.8e-10 | 3.7e-13 |
| dump check 4 (exact `Edot` vs `sum kappa.Y` off the written matrix), RTX 3080 | 4.0e-17 | 6.9e-10 | 7.9e-13 |
| the same, RX 9060 XT | 6.8e-17 | 4.4e-10 | — |

The CI forward side is `pcTangentsDevice` with its `epsDot` (#74 added it),
whose own license is the finite difference above ("The tangent with respect to
the CI vector"); the rotation's forward side is new code
(`pcRotationTangent*`), which is why the dump gates the integral halves too.
The acrolein column is a dump with its sidecar regenerated out of tree (as
#89's was), so not a ctest entry. CAS(8,8)'s larger figures are its
conditioning, not a drift: its Sr and Si metrics keep modes down to ~1e-10
relative ("The singular-metric convention"), so `U` carries `s^-1/2` of
~1e5 into both modes' rounding — still three decades inside the gate.

### block2's exact source terms, on a card

`nevpt2_sc_pseudodensity --pc --source-oracle golden/n2_ccpvdz_cas44_pc.srcgold`
(the `oracle_pc` entry), against the PC oracle tolerance (testing.md, "The PC
source terms' oracle", 1e-7, the known-gap list empty). Max \|GPU − block2\|
per block:

| block | both cards |
|---|---:|
| core–core | 5.4e-11 |
| core–act_O / core–act_V | 0 / 0 |
| core–virt | 1.8e-10 |
| act_O–act_V | 1.5e-11 |
| act_O–virt | 2.2e-11 |
| act_V–virt | 1.0e-11 |
| virt–virt | 4.7e-11 |
| `P y` (12 determinants) | 8.6e-10 |

The two cards agree to every digit shown. Every block passes with two to
three decades to spare: what is left is the
oracle's own ladder spread ("The PC source terms' oracle"). The `P y` row is
the orbital energies' response in `y`, and the act–virt rows the digests'
transpose in `Y` — the two terms the amendment is about; the negative
controls below show each failing here when dropped.

**Acrolein CAS(6,5), directionally (#105).** Acrolein's full PC oracle would
cost hours to generate, so its PC terms are held to a **directional** one,
`golden/acrolein_def2svp_cas65_pc_dir.srcgold`: two seeded directions per
block and three CI directions, block2's derivative along each, generated in
under ten minutes (testing.md, "The directional source-term oracle", which has
the format's reasons, the ladders, the tolerance — 2e-8 — and the budget).
`nevpt2_sc_pseudodensity_oracle_dir_pc_acrolein_def2svp_cas65` (`gpu;slow`;
registered where acrolein's gitignored sidecar has been regenerated), max
\|GPU − block2\| per block over its directions:

| block | RTX 3080 | RX 9060 XT |
|---|---:|---:|
| core–core | 8.7e-11 | 8.7e-11 |
| core–act_O | 3.4e-10 | 2.3e-10 |
| core–act_V | 4.4e-11 | 4.8e-10 |
| core–virt | 4.8e-10 | 4.9e-10 |
| act_O–act_V | 7.2e-10 | 3.8e-9 |
| act_O–virt | 1.0e-9 | 7.7e-10 |
| act_V–virt | 1.1e-9 | 1.0e-9 |
| virt–virt | 1.3e-10 | 1.3e-10 |
| `P y` (3 directions) | 4.9e-10 | 4.2e-10 |

Unlike N2's, the two cards differ here in the third decade below the
tolerance: act_O–act_V by 3e-9 and core–act_V by 4e-10. Each card is
self-consistent run to run (three reruns on the RX 9060 XT agree to the
digits shown), and the cards
run different default digests (emitted on CUDA, BLAS on HIP); which part of
`Y`'s 2,850 basis directions carries the difference was not separated. Both
are at least five times inside the tolerance.

### On a CASSCF reference

#160 holds the same PC terms on `n2_ccpvdz_cas44_casscf` (the CAS(4,4)
state on its CASSCF orbitals, #117) to that golden's full PC oracle,
`golden/n2_ccpvdz_cas44_casscf_pc.srcgold` — block2's exact PC source terms,
written by `sc_source_oracle.py --pc --mo-from`. Nothing on the device
changes: `Y` and `y` are derivatives of the energy functional at fixed
orbitals and CI vector, so `requireSourceTermReference` accepts both
reference kinds. What changes is the oracle's **partition**. With no RHF to
split the active space, it is one range (`n_act_occ = n_act`, every
active–active rotation a gauge), and the state stays in its canonicalised
basis, so `act_rot` is the identity ([`gradient-theory.md`](gradient-theory.md),
§1.4). `act_V` is therefore empty, and core–act_V, act_O–act_V and
act_V–virt have no pairs; the comparator requires exactly those three rows
empty on a `_casscf` oracle (`test/source_compare/`).

`nevpt2_sc_pseudodensity --golden golden/n2_ccpvdz_cas44_casscf.nevpt2gold
--tiles 3 --pc --source-oracle golden/n2_ccpvdz_cas44_casscf_pc.srcgold`
(`nevpt2_sc_pseudodensity_oracle_pc_n2_ccpvdz_cas44_casscf`), RTX 3080,
preset `default`, against the PC oracle tolerance (1e-7), `kKnownGaps`
empty. Max \|GPU − block2\| per block:

| block | pairs | RTX 3080 |
|---|---:|---:|
| core–core | 10 | 8.7e-11 |
| core–act_O | 20 | 0 |
| core–act_V / act_O–act_V / act_V–virt | 0 / 0 / 0 | empty |
| core–virt | 95 | 1.9e-10 |
| act_O–virt | 76 | 2.8e-11 |
| virt–virt | 171 | 2.7e-11 |
| `P y` | 12 determinants | 2.5e-10 |

**PASS, 6 blocks agree, 0 known gaps.** The worst block is two and a half
decades inside the tolerance, as on the CASCI twin above. core–act_O is 0.0
on both sides: on N2 it vanishes by symmetry, as on the CASCI twin, so it
checks nothing here, and five blocks carry content.

`--pc --dump-source-terms` on the same golden
(`nevpt2_sc_pseudodensity_dump_pc_n2_ccpvdz_cas44_casscf`), same run, its
five checks:

| check | value |
|---|---|
| 1. the pairing: Σ κ·Y off the matrix vs `<kappa, Y>` direct, one seeded full direction | 3.1e-17 (gate 1e-9) |
| 2. `y`'s walk ran on this state: its `E_corr` vs `pcEnergiesDevice` | 1.1e-16 (gate 1e-7) |
| 3. the `.gradsrc` read back | bit for bit |
| 4. each direction's exact `Edot` (frozen + digest) vs Σ κ·Y off the matrix, 6 directions | 5.7e-17 (gate 1e-7) |
| 5. the PC integral halves' duality, adjoint vs forward tangent, 6 directions | ≤ 8.0e-18 (gate 1e-7) |

The orbital energies' response is in `y` here too: `e_core` / `e_virt`
rebuilt from the CASSCF sidecar agree with the golden's to 1.6e-14
(relative). ctest walls: dump 1.58 s, oracle 1.44 s. The SC and directional
twins on this golden have their numbers in [`testing.md`](testing.md),
"The source terms on CASSCF orbitals, measured".

### The nuclear gradient from the GPU's terms

`nevpt2_grad.py --pc --source both --source-path <dump> --grad-num <npz>`,
PC `grad_num` from `nevpt2_numgrad.py --pc --no-write --npz`, single-threaded
Python on the RTX 3080 box, PySCF 2.14.0, block2 0.5.4; the dumps from the
RTX 3080. (#74 also fixed `--source both`'s PySCF-sourced solve, which had
`pc=False` written into it: harmless while `--pc` refused `both`, and wrong —
2.2e-3 from `grad_num` at N2 — the first time it ran.)

| | N2 CAS(4,4) | acrolein CAS(6,5) |
|---|---:|---:|
| components compared | 6 | 16 (in-plane) |
| **GPU-sourced**: max \|g − grad_num\| | **1.9e-8 — PASS** | **6.0e-8 — PASS** |
| PySCF-sourced, same sweep | 1.5e-8 — PASS | 8.3e-8 — PASS |
| max \|GPU-sourced − PySCF-sourced\| | 3.5e-9 | 9.0e-8 |
| … the GPU's `Y` alone / `y` alone | 3.6e-9 / 7.1e-11 | 2.1e-9 / 9.2e-8 |
| the dump vs PySCF's `A` at `KAPPA_STEP_PC`, worst block | 1.5e-8 (core–virt) | 3.4e-8 (core–virt) |
| the dump's `P y` vs PySCF's | 8.6e-10 | 6.5e-7 |
| residual-independent checks | 11 of 11 since #104 (below; 10 of 11 at #74) | 7 of 11 (below) |
| `--source both` wall | 182 s | 24616 s |

The worst block is core–virt at `KAPPA_STEP_PC` = 1e-4, which "The PC source
terms' oracle" measures 1.5e-8 from converged — PySCF's truncation, not the
device's: against the oracle's converged step the same block is 1.8e-10.

**N2's residual-independent checks: 11 of 11 from both providers since
#104.** At #74 one check failed on run length alone: CI stationarity, seed
2026 (two fourfold halvings where the rule wants three, 8.3e-13 from zero),
while the PySCF-sourced run passed it with three. #104 measured the cause
([`testing.md`](testing.md), "PC's CI-direction ladders, measured"). At a
solution the Lagrangian's pieces cancel, so its truncation term is ~40
times smaller than theirs. Its round-off is not, and the check's `z` term
carried `ecore`'s for nothing. Between the guard-fixed start at `2^-11` and
that floor, the ladder sits on the rule's edge, and which provider falls off
it changes from run to run. The fix is in the check, not in the rule: the
`z` term is formed without `ecore`, and when the whole shows no run the
pieces are laddered one by one. The rest held throughout, the PC guard's
ledger included (88 of 88 displaced CI vectors guarded, rank unchanged at
every one).

**Acrolein: the `P y` finding is resolved, by #105; the run-length
failures stay on record.** At #74 four of acrolein's eleven
residual-independent checks failed, all on ladder run length: `y` linearity,
seed 70 (1 fourfold halving, Richardson − ref 1.2e-10) and seed 2026 (2,
6.3e-10); CI stationarity, seed 70 (1, 1.2e-10) and seed 2026 (2, 6.1e-10).
The other seven held, the PC guard at 168 of 168 displaced CI vectors with no
rank change. The checks run once per invocation, on the GPU-sourced solve; no
PySCF-sourced acrolein PC run of them was made, and #105 did not re-run them.
Separately, the dump's `P y` differed from PySCF's by 6.5e-7 at `CI_STEP_PC`
(1e-7), and by 1.9e-6, 3.3e-6 and 1.3e-5 at 5e-8, 2.5e-8 and 1e-8; that
attribution row put 9.2e-8 of the gradient's GPU-vs-PySCF difference on `y`.

**That difference was the oracle's, not the device's.** #105's directional
oracle differences block2's `E_corr` along spin-adapted CI directions, which
the PC guard accepts at a step some 2,400 times `CI_STEP_PC` (`2^-12`; [`testing.md`](testing.md), "The directional source-term oracle"),
and there the GPU's `P y` agrees with block2's to **4.2e-10 on the RX 9060 XT
and 4.9e-10 on the RTX 3080**, every one of the three directions. The
per-determinant quotient's 6.5e-7 was block2's round-off — ~100–400 ulp of
`E_corr(PC)` per evaluation on this state, against N2's one ([`testing.md`](testing.md),
"PC's CI-direction ladders, measured") — divided by a 1e-7 step, which is
also why it grew as `1/h`. The two failing `y`-linearity ladders and the two
CI-stationarity ones difference that same noisy functional, which is #103's
hypothesis for them; they are unchanged, an offline diagnostic and not a gate.
**The acrolein PC regression gate is now the directional ctest entry**
(`nevpt2_sc_pseudodensity_oracle_dir_pc_acrolein_def2svp_cas65`), every
block within the PC directional tolerance on both cards, the known-gap list
empty. The offline `--source both` run above (24616 s) and acrolein's full-`A`
oracle sweep (#74's 21,136 s scratch run) are **optional diagnostics**, kept
as the record, never gates.

**`CI_STEP_PC` and the guard.** No finite difference enters the device side.
The ones beside it are `nevpt2_pc_tangent`'s (symmetry-sector directions, the
guard recorded at every displaced state) and PySCF's in `nevpt2_grad.py --pc`
and `sc_source_oracle.py --pc`, at `CI_STEP_PC` with the guard and the rank
check run at every displaced CI vector — the ledger line above.

### Refusals

`test/pc_adjoint/` (`gpu`): on N2 CAS(4,4)'s CI vector displaced off its
symmetry sector, the adjoint reports exactly the PC energy's refusals, class
by class with the reason, forms no `y`, and the digests' adjoint and the
matrix of `Y` are Numerical Errors naming the class. Both kinds of refusal:
displaced by 0.25 the denominators fail (Sr −19.3, Si −25.6, Sir −0.086 Eh);
displaced by 1e-6 the gap does (Sr and Si 1.3, Sir 1.5, against
`PC_MIN_GAP` 1e3). The forward tangent refuses the same classes, and the
golden's own state is accepted with the walk at block2's `e_pc_total`.

### Negative controls

Scratch patches, never committed; RTX 3080, N2 CAS(4,4):

| patch | dualities | block2's oracle | gradient vs `grad_num` |
|---|---|---|---|
| 1. the adjoint perturbed: `dE/dK`'s weight ×(1 + 1e-3), reverse only | **FAIL**: CI 1.5e-4; rotation halves 2.1e-6, check 4 2.1e-6 | **FAIL**: core–virt, act_O–virt, act_V–virt, `P y` | — |
| 2. a shared omission in `Y`: the digests' term dropped from both modes | pass: halves 6.9e-18, check 4 1.1e-16 | **FAIL**: act_O–virt 1.5e-2, act_V–virt 3.8e-3 | **FAIL**: 7.5e-3 |
| 3. a shared omission in `y`: the orbital energies' response dropped from both modes | pass: CI 2.0e-16 | **FAIL**: `P y` 3.2e-4 | **FAIL**: 1.6e-4 |

Patches 2 and 3 are the amendment's point: an omission both modes share is
invisible to every duality, and the independent provider — block2's committed
terms on the card, `grad_num` offline — catches it. (`--source dump` for the
two gradient rows; their residual-independent checks fail as well: 8 and 7 of
11.)

### What the entries cost, and the sanitizers

ctest walls, RTX 3080 / RX 9060 XT: `nevpt2_gradient_source_pc_cas44` 0.88 /
1.18 s, `_cas88` 2.59 / 24.3 s, `nevpt2_sc_pseudodensity_dump_pc_n2_ccpvdz_cas44`
1.66 / 2.61 s, `_cas88` 29.9 / 394 s, `…_oracle_pc_n2_ccpvdz_cas44`
1.53 / 2.44 s, `PcAdjointRefusalTests` 0.50 / 0.57 s. The two
`gradient_source_pc` entries are `small`, so the sanitizer tier runs the PC
adjoint's device work; `PcAdjointRefusalTests` rides in every preset's unit
tier. The CAS(8,8)
dump is labelled `slow`: its `Y` probe rebuilds every class's Dyall block per
basis direction (236 s of its 394 s on the RX 9060 XT, 25 s on the RTX 3080).

Each preset's own tier passes with the PC entries in it: `asan` 206/206,
`ubsan` 206/206, `compute-sanitizer` memcheck 197/197 and initcheck 194/194
(RTX 3080), `hip-asan` 204/204 and `hip-ubsan` 204/204 (RX 9060 XT). The two
non-`small` CAS(4,4) entries (`dump_pc`, `oracle_pc`) were also run by name
in each of the six builds, with the presets' sanitizer options: all pass, no
report — seconds, dump / oracle: `asan` 5.6 / 5.0, `ubsan` 4.1 / 3.8,
memcheck 21.1 / 19.5, initcheck 14.1 / 13.0, `hip-asan` 15.8 / 14.5,
`hip-ubsan` 13.0 / 12.2.

## The nuclear gradient on a CASSCF reference

The PC nuclear gradient of acrolein CAS(6,5) on a CASSCF reference (#126) is
`nevpt2_grad.py --reference casscf --pc`, block2's PC source terms fed into
the CP-CASSCF coupled solve. It is gated against a numerical gradient
differenced from **this device's** PC energies on #124's re-optimised
states (`pc_on_states.py`, then `nevpt2_numgrad.py --pc-states`), with block2
spot-checked at displaced geometries. Differencing our own energy is sound
for the reason "The PC total's run-to-run spread" above gives. No
GPU-derived `grad_num_pc` is committed: the sweep is a manual validation run.
The verdict, per component, with the spot-checks and the walls, has one
home: [`testing.md`](testing.md), "PC on a CASSCF reference: acrolein".

## What is not supported

- **N2 CAS(12,12)**, conventional or DF: its metric has no gap (see
  "Conditioning of the committed states"), so the goldens carry no PC fields
  and `--pc` refuses them before the RDM build. The 12-orbital PC case is
  CAS(10,12).
- **Any golden without `pc_class_energies` and `e_pc_total`**, for the same
  reason: there is nothing to check against.
- **A state whose metric is not gapped** by at least `PC_MIN_GAP`: the class
  is refused, with no energy, rather than reported at an arbitrary cut. The
  tangent is refused with it ("The tangent with respect to the CI vector").
- **A CI direction that leaves the state's spin and spatial symmetry**: the
  metric gains rank there, so the pseudo-inverse is not differentiable along
  it and the displaced state has no PC energy at all. Measured above. (The
  *derivative* at the state along such a direction is defined, and the
  source terms read it on every determinant axis: "The source terms by
  reverse mode".)
