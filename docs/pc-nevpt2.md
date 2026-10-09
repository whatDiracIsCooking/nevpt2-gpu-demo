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

`nevpt2_cas1012_pc` is `slow` because it takes ~29s on CUDA and ~36s on HIP,
against ≤3s for every other PC entry. The four `small` entries put the PC
stage (the eigensolves, the BLAS GEMMs, the slab einsums) under the
sanitizer tier.

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

## What is not supported

- **N2 CAS(12,12)**, conventional or DF: its metric has no gap (see
  "Conditioning of the committed states"), so the goldens carry no PC fields
  and `--pc` refuses them before the RDM build. The 12-orbital PC case is
  CAS(10,12).
- **Any golden without `pc_class_energies` and `e_pc_total`**, for the same
  reason: there is nothing to check against.
- **A state whose metric is not gapped** by at least `PC_MIN_GAP`: the class
  is refused, with no energy, rather than reported at an arbitrary cut.
