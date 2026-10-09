# Performance

What this code costs to run, where the time goes, which knobs matter, and
which ideas were measured and did not help. Every figure names the card, the
case and the flags it was measured with. Anything not listed here was not
measured.

**The cards.** NVIDIA RTX 3080 (10 GB, CUDA 13.0) and AMD RX 9060 XT
(gfx1200, 16 GB, ROCm 7.2.4). The RTX 3080 also drives a display and was
often shared with other processes, which typically held about 1 GB of its
memory. That makes absolute times noisy, so comparisons are interleaved
runs taken in the same session. **A ratio is more reliable than an absolute
time**, and two figures from different tables can differ by more than the
change being compared.

**What every run checks.** Each timed run below also passed its golden
check (every class energy and the total to 1e-7), unless the row says
otherwise.

## Where the time goes

A run has two GPU stages: the **RDM build** (dm3 and the two f3 digests,
tiled over determinants) and the **energy contraction** (the eight class
energies). The demo prints both, and the wall time also includes reading the
golden file and setting up the device.

Default digest on each card (emitted on CUDA, BLAS on HIP), medians of 5
runs (3 for DF CAS(12,12)):

| case | card | wall | RDM build | energy |
|---|---|---:|---:|---:|
| CAS(10,10) `--tiles 3` | RTX 3080 | 1.91s | 1.08s | 0.40s |
| DF CAS(10,10) `--tiles 3` | RTX 3080 | 1.90s | 1.02s | 0.40s |
| CAS(12,12) `--tiles 40` | RTX 3080 | 41.37s | 39.39s | 1.59s |
| DF CAS(12,12) `--tiles 40` | RTX 3080 | 41.42s | 39.45s | 1.57s |
| CAS(10,10) `--tiles 3` | RX 9060 XT | 2.75s | 1.51s | 1.02s |
| DF CAS(10,10) `--tiles 3` | RX 9060 XT | 2.77s | 1.19s | 1.02s |
| CAS(12,12) `--tiles 40` | RX 9060 XT | 45.70s | 41.75s | 3.82s |
| DF CAS(12,12) `--tiles 40` | RX 9060 XT | 45.85s | 41.65s | 3.87s |

**Both stages are kernel-bound.** With `--profile`, the summed GPU time of
each stage is essentially its whole wall time. At CAS(12,12) `--tiles 40`
the RDM build's kernels account for 38.32–38.35s of a 38.34–38.37s build on
the RTX 3080, and 41.74–41.79s of 41.78–41.85s on the RX 9060 XT. The energy
stage is 1.53–1.54s of 1.54s and 3.82–3.84s of 3.83–3.84s. So nothing
outside the kernels is left to optimise, and the launch architecture around
them does not matter (see "Device-resident RDM build" below).

**Inside the RDM build, the three digest GEMMs dominate.** On the default
paths:

- **RTX 3080, emitted digest, CAS(12,12) `--tiles 40`:** each of the three
  digests takes ~11.5s, and the consume step 2.0s of a 38.9s build.
  `produce` takes about 1.0s.
- **RTX 3080, `--cublas --tiles 80`:** the three emulated digests take about
  7.4s together, consume ~2.1s (19%) and `produce` 1.0s, in a 10.85s build.
- **RX 9060 XT, BLAS digest, CAS(12,12):** the digests are 87% of kernel
  time, and `produce` (6.4%) is next.

**Inside the energy stage, one shape family dominates.** `--profile`'s
energy table, RTX 3080, CAS(10,10), grouping the 144 distinct einsum call
shapes by output size and contracted size:

| outSize × contracted | calls | total ms | % of kernel time | ns/thread |
|---:|---:|---:|---:|---:|
| 1,000,000 × 100 | 12 | 238.6 | 62.6% | 0.199 |
| 10,000 × 1,000 | 15 | 55.2 | 14.5% | 0.368 |
| 16 × 1,000,000 | 2 | 54.8 | 14.4% | **1.714** |
| everything else (115 calls) | — | 32.3 | 8.5% | — |

The first row is the `make_a16`/`make_a22` terms shaped like
`"kbia,rpqcki->pqrabc"`, a true two-operand contraction that could be
written as a GEMM. The third is `energy_Sr`/`energy_Si`'s small output over
a huge contraction. The kernel assigns one thread per (output, contracted)
pair and accumulates with `atomicAdd` (see `device_einsum.cu`'s header
comment), and that row's 8.6× higher cost per thread is the atomic
contention this design accepts. Neither shape is tuned: the energy stage is
small and shrinks further at larger active spaces (next section).

**`--profile` costs nothing measurable.** At CAS(12,12) `--tiles 40`, two
profiled and two unprofiled runs interleaved on each card differed by less
than two unprofiled runs differ from each other (0.76s on the RTX 3080,
0.46s on the RX 9060 XT).

## Scaling with active-space size

The largest energy-stage shape grows as `norb^8`, which suggests the energy
contraction would matter more at a larger active space. It matters less. The
RDM build tracks the determinant count, which grows combinatorially
(`C(norb, norb/2)^2`: 63,504 at CAS(10,10), 853,776 at CAS(12,12), ~13.4×),
while the energy contraction grows only polynomially. RTX 3080, emitted
digest, from the table above:

| | CAS(10,10) `--tiles 3` | CAS(12,12) `--tiles 40` |
|---|---:|---:|
| RDM build | 1.08s (57% of wall) | 39.39s (**95%**) |
| energy contraction | 0.40s (21%) | 1.59s (**3.8%**) |
| wall | 1.91s | 41.37s |

Within the energy stage, the `make_a16`/`make_a22` family's share does grow
as predicted (62.6% at CAS(10,10), 70.6% at CAS(12,12)), and the
atomic-contention row shrinks (14.4% to 9.8%). But it is a growing slice of
a shrinking stage. **Speed at larger active spaces is the RDM build's
digests**, which is where the digest options below act.

## Head-to-head vs PySCF

PySCF's `NEVPTcontract` (`make_dm123` + `_contract4pdm` ×2) is the CPU
equivalent of the RDM build. `reference_data/pyscf_contract_time.py` times it
on the **same committed CI vector** the demo reads, with no CASCI re-solve,
so the two sides are directly comparable. PySCF on 24 CPU threads against
the GPU `--cublas` RDM build on the RTX 3080:

| case | PySCF `NEVPTcontract` (24 threads) | GPU `--cublas` RDM build | speedup |
|---|---:|---:|---:|
| CAS(10,10) | ~4.0s | ~0.8s | ~5× |
| CAS(12,12) | **~110–160s** | **~15s** | **~7–11× (order 10×)** |

**The advantage grows with the active space**, because the digest is a
larger share of the build there and the digest is what `--cublas`
accelerates.

How far to trust these figures:

- **Both sides were timed on a shared machine.** PySCF's CAS(12,12) build
  measured 108s on one occasion and 160s on another, and the GPU side ran
  under light display load. So the ratio is an order of magnitude, not a
  single figure. The claim that holds up is roughly 10× at CAS(12,12),
  several-fold at CAS(10,10), and rising.
- **The GPU side is older than the consume DGEMM.** The `--cublas` builds
  in this table predate running consume as a DGEMM (below). The current
  `--cublas` build is 0.69s at CAS(10,10) and 10.85s at CAS(12,12) `--tiles
  80`, but it has not been timed against PySCF again.
- **It is one GPU against 24 CPU threads,** and the CASCI that produces the
  CI vector is excluded from both sides. Correctness is not in question:
  every GPU run here passed its golden check.

## The digests

Four ways to run the three digest GEMMs, all accurate to fp64 and all
checked against the golden at 1e-7. RDM build at CAS(12,12):

| digest | flag | RTX 3080 | RX 9060 XT |
|---|---|---:|---:|
| emitted 16×16 tiles | default on CUDA; `--digest-emitted` | 38.9s (`--tiles 40`) | 138.2s (`--tiles 40`) |
| native-fp64 BLAS | default on HIP; `--blas-digest` | 51.3s (`--tiles 40`), 43.2s (`--tiles 80`) | 42.8s (`--tiles 40`), 41.9s (`--tiles 80`) |
| cuBLAS fp64 emulation | `--cublas` (CUDA only) | **10.85s** (`--tiles 80`) | — |
| hand-written int8 Ozaki | `--ozaki` | 63.6s (`--tiles 80`) | 82.9s (`--tiles 80`) |

The rows come from different sessions, so compare within a section below
rather than across rows.

### Native-fp64 BLAS digest

`--blas-digest` runs the three digest GEMMs as plain native-fp64
`wwrblasDgemm` (cuBLAS or rocBLAS through `wwr.blas`) on both backends: dm3
with `beta=1` straight into its accumulator, and each f3 order into an
`n^6` temp that `f3_scatter.cu` folds into place. **It is the default on HIP
and off on CUDA**, because that is how the two cards came out. Emitted and
`--blas-digest` back to back, same session:

| | emitted build | `--blas-digest` build | per digest GEMM, emitted → BLAS |
|---|---:|---:|---:|
| HIP CAS(10,10) `--tiles 3` | 4.25s | **2.45s (1.7×)** | 366ms → 108–161ms |
| HIP CAS(12,12) `--tiles 40` | 138.18s | **42.80s (3.2×)** | 1088ms → 300–305ms (3.6×) |
| CUDA CAS(10,10) `--tiles 3` | 1.20s | 1.40s (slower) | 94–106ms → 132–137ms |
| CUDA CAS(12,12) `--tiles 40` | 39.18s | 51.27s (slower) | 297ms → 398ms |
| CUDA CAS(12,12) `--tiles 80` | 38.85s | 43.22s (slower) | 148ms → 165ms |

Why the cards disagree, in GFLOP/s. One CAS(12,12) digest GEMM at `--tiles
40` is `2·n^4·n^2·K` = 1.28e11 flop.

- **RTX 3080:** the emitted tiles already run at ~430 GFLOP/s, about the
  card's fp64 peak, so a library GEMM has nothing to gain. cuBLAS's native
  DGEMM reaches ~320 at tile width 21,345 and ~385 at 10,673.
- **RX 9060 XT:** the same emitted tiles reach only ~117 GFLOP/s, and
  rocBLAS ~425. The emitted kernels were tuned on the NVIDIA card and leave
  most of RDNA4's fp64 throughput unused.

So with its default digest, the HIP CAS(12,12) build is level with the CUDA
one.

### The cuBLAS fixed-point-emulation digest (`--cublas`)

The RTX 3080 has no fp64 tensor cores, so CUDA 13.0's fp64 emulation is the
lever. `--cublas` runs the three digest GEMMs through `cublasGemmEx` with
`CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` (EAGER strategy, FIXED mantissa
control). This is the int8 Ozaki scheme, done inside cuBLAS: each fp64
operand is split into int8 slices, multiplied exactly on the integer tensor
cores, and summed in fp64 (see [`references.md`](references.md),
"Floating-point emulation"). `--mantissa-bits` defaults to **53**, which
matches native fp64. It is CUDA-only, because ROCm has no equivalent.

Same session, CUDA, consume as a DGEMM in both columns:

| | emitted build | `--cublas` build | speedup |
|---|---:|---:|---:|
| CAS(10,10) `--tiles 3` | 1.20s | 0.69s | ~1.7× |
| CAS(12,12) emitted `--tiles 40`, `--cublas --tiles 80` | 38.92s | **10.85s** | ~3.6× |

The speedup grows with the active space, because the digests are a larger
share of the build there. It is smaller than the speedup of an isolated
digest GEMM, because `produce` and consume are untouched.

**Read the `engaged bits=` line.** Above a per-tile `K` (the tile width),
cuBLAS declines to emulate and quietly runs native fp64 instead. The demo
reports this as `engaged bits=-1 (DECLINED -> ran native fp64!)`, and a
timing from a declined run means nothing. At CAS(12,12):

| `--tiles` (tile width) | emulation |
|---|---|
| 17 (the memory floor) | declines (~50–54s, native fp64) |
| 40 (21,345) | declined once (77.75s); engaged (bits=53) in three later sessions |
| 80 (10,673) and 150 (5,692) | engages (bits=53), every time measured |

So **`--cublas` needs more tiles than the emitted path**, and `--tiles 80`
is the recommendation at CAS(12,12). With `--cublas`, tiling both fits the
build in memory and keeps each digest operand under cuBLAS's engagement
ceiling. The ceiling lies somewhere between widths 10,673 and 21,345.

The ceiling depends on `M` as well as on `K`. Probing one GEMM in isolation
(`probe_emul_bits` at `max_mantissa_bits=53`, RTX 3080, 2026-10-09):

| `M` × `N` (row-major digest shape) | engaged | declined (`bits=-1`) |
|---|---|---|
| 20,736 × 144 (CAS(12,12)'s `n⁴` × `n²`) | `K` = 10,673, 21,345, 25,000 | `K` = 30,000 |
| 1 × 1 | `K` = 1,048,576 and 1,048,577 | `K` = 1,500,000 and above |
| 10,000 × 100 (CAS(10,10)'s; `--tiles 3` is `K` = 21,168) | `K` = 2,000 and 21,168 | not probed |

In isolation, width 21,345 engaged. Under compute-sanitizer memcheck, the
declined 1 × 1 probe shows the mechanism: inside `digest_gemm`, cuBLAS's own
workspace `cudaMallocAsync` fails with `out of memory`, and cuBLAS then falls
back to native fp64. So a decline is a failed workspace allocation, and the
ceiling may depend on free device memory as well as on the shape. That
would fit the demo declining only once at width 21,345, but nobody has
measured it. `test/cublas/` relies on the 1 × 1 row for its declined case
(`K` = 4,000,000) and on the 10,000 × 100 row for its engaged case.

### The int8 Ozaki digest (`--ozaki`)

`--ozaki` (`src/ozaki/`) is a hand-written version of the same int8 scheme,
on **both** backends through WarpWraps' `<wmma.h>` (`nvcuda::wmma` on CUDA,
rocWMMA on HIP; on gfx1200 it runs
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`). It is the only int8
route open on HIP. How it works:

- **The split is exact.** Each row of both operands is scaled by `2^(e+1)`
  (with `max|row| < 2^e`), and each digit is `rint(r·128)` followed by
  `r ← r·128 − d`. Every step is exact in fp64, so `|d| ≤ 64`. Error comes
  only from dropped digit pairs, from truncation (an element `2^-j` below its
  row's scale keeps 56 − `j` bits), and from the final fp64 sum.
- **Pairs are grouped by level `g = p+q`.** Every pair at one level carries
  the same weight `2^-7(g+2)`, so each level has one int32 accumulator, and
  each digit tile is loaded once per K step. An output stays exact while
  `8·64²·K < 2³¹`. A K wider than that (`kOzakiMaxK` = 65535) is cut into
  near-equal chunks, each scaled and split separately and added with
  `beta = 1`. One warp computes one 16×16 output tile.
- **The per-level fold synchronizes one warp,** through a warp-sized
  `cooperative_groups::tiled_partition`. A block-wide sync cost about 2× on
  the RTX 3080. On the RX 9060 XT the block-wide version was slightly faster
  (2.17–2.20s against 2.45–2.46s at cc-pVTZ), but the warp version is kept
  for the larger CUDA difference.

`--ozaki-pairs P` keeps the pairs with `p+q ≤ P` (14, the default, keeps all
64). `--ozaki-check` also runs every digest GEMM natively and reports the
largest elementwise difference relative to that call's `max|C|`.

**Accuracy: with every pair kept, it matches native fp64; truncation fails
by refusal, not by a wrong number.** The strictest test is
`n2_ccpvtz_cas1010_df --pc --tiles 3 --batch 5`, which has the tightest PC
metric gap among the committed goldens (the gap is Si's: smallest kept over
largest dropped singular value):

| | digest vs native (CUDA / HIP) | Si gap | result |
|---|---:|---:|---|
| default digest | — | 2.3e4 | PASS (Sr 6.80e-10 vs block2) |
| `--ozaki` (P=14, 64 pairs) | 3.1e-15 / 3.3e-15 | 2.3e4 | PASS, every class at the baseline |
| `--ozaki-pairs 9` (55 pairs) | 3.1e-15 / — | 2.4e4 | PASS |
| `--ozaki-pairs 7` (36 pairs) | 4.7e-15 / 5.0e-15 | 1.8e4–1.9e4 | PASS |
| `--ozaki-pairs 6` (28 pairs) | 2.3e-13 / 2.4e-13 | 3.2e3 | PASS, gap close to `PC_MIN_GAP` |
| `--ozaki-pairs 5` (21 pairs) | 4.8e-11 / 4.8e-11 | 18 | **FAIL: Sr, Si refused** |
| `--ozaki-pairs 4` (15 pairs) | 1.8e-9 / — | 8e-3 | FAIL: Sr, Si refused |
| `--ozaki-pairs 3` (10 pairs) | 9.1e-7 / — | 3.9e2 | FAIL: Sr, Si refused; Srs off by 9.5e-8 |

Sr's 6.80e-10 is the same on both cards and on every digest. It is a
difference of convention from block2, not digest noise. Truncation lifts
the structural zero modes of `S` from ~1e-15 to ~2e-12 (at P=5), next to
the first real modes at 2e-11, and the gap guard then refuses the class.

SC is far more forgiving, because it inverts no metric. `n2_ccpvdz_cas1212
--tiles 80` has the widest RDM dynamic range among the committed cases (one
N 1s orbital in the active space, with 2 − n = 8.0e-6). There, E_corr
|delta| is 1.4e-14 to 1.7e-14 with all 64 pairs on both cards, and still
1.56e-10 at P=5, ~640× inside 1e-7.

**To test a lossy digest, run `n2_ccpvtz_cas1010_df --pc` first and read the
gap column, not just PASS.** It fails before anything else does, and it
fails by refusal. Then run CAS(12,12) SC.

**Speed: slower than the default digest on both cards.** RDM build,
unprofiled:

| | default digest | emitted | `--ozaki` | per digest GEMM (profiled), default → `--ozaki` |
|---|---:|---:|---:|---:|
| CUDA cc-pVTZ DF `--pc --tiles 3` (two runs) | 1.02s (emitted) | 1.02s | 1.93s | — → ~210 ms |
| HIP cc-pVTZ DF `--pc --tiles 3` (two runs) | 1.19–1.20s (BLAS) | 2.06–2.07s | 2.45–2.46s | 108 → 248 ms |
| CUDA CAS(12,12) `--tiles 80` | 37.00s (emitted) | — | 63.58s | — |
| HIP CAS(12,12) `--tiles 80` | 41.93s (BLAS) | — | 82.87s | — |

On the RTX 3080, ~210 ms for a 4.2e10-flop call is ~200 GFLOP/s
fp64-equivalent, against ~430 for the emitted tiles, and `--cublas` builds
the same case in 0.45s. On the RX 9060 XT it is behind rocBLAS's native fp64
(2.3× per call) and slightly behind even the emitted tiles. So, *for this
kernel*, RDNA4's int8 WMMA does not beat its fp64. Keeping 36 pairs instead
of 64 changes nothing, so the tensor-core multiplies are not the bottleneck.
The likely cost is the 16 digit-fragment loads per K step with no reuse
across warps. A faster kernel would need register blocking (several output
tiles per warp, sharing the A digit fragments). That has not been tried.

**Memory:** the digit planes are `8·n^4·K` bytes, the size of `L2` itself,
for each K chunk. With one chunk, that doubles the per-tile working set:
1633 MB of scratch on top of cc-pVTZ CAS(10,10)'s 1647 MB, and ~3.3 GiB at
CAS(12,12) `--tiles 40`, which is why the CAS(12,12) runs use `--tiles 80`.
Smaller chunks shrink it (316 MB with a 4096-wide chunk at cc-pVTZ).

### Consume as a DGEMM

The f3 consume step, per tile and per order,

```
ca: W[a,f,K] = sum_{pqx} eri[a,x,q,p] * L2[p,q,x,f,K]
ac: W[a,f,K] = sum_{pqx} eri[a,x,q,p] * L2[p,q,f,x,K]
```

runs as `wwrblasDgemmStridedBatched` (cuBLAS or hipBLAS) on both backends:
for each `x`, one GEMM batched over `f`, `k = n^2`, accumulating with
`beta=1`. A single `n^4` permuted integral copy, `E[x,a,p,q] =
eri[a,x,q,p]`, serves both orders. `--consume-emitted` runs `kernels.cu`'s
loop kernels instead, for comparison.

**The product is skinny, and the two BLAS libraries handle it
differently.** There are only `n` output columns (`a`) per call. Sweeping
one strided-batched DGEMM (`m = 21,345`, batch 12) over the column count and
`k`:

- **cuBLAS 13.0, RTX 3080:** with ≤10 columns and `k ≤ 128` it runs a skinny
  kernel at the card's fp64 peak (400–470 GFLOP/s). At 11 or more columns,
  or `k = 144`, it falls back to a 32-column tile and wastes the padding:
  ~150 GFLOP/s, flat up to 32 columns. Unchunked, CAS(12,12) (12 columns,
  `k = 144`) was no faster than the loop kernels (5.37s against 5.21s).
- **hipBLAS / rocBLAS, RX 9060 XT:** one 16-column tile, time flat up to 16
  columns and indifferent to `k`. The cuBLAS-shaped split (12 columns as 2×6)
  doubled HIP's consume time (5.77s against 2.80s at CAS(12,12)).

So `rdm_build.cpp` chunks each per-`x` GEMM by backend: ≤10 columns ×
`k ≤ 128` on CUDA, ≤16 columns × the whole `k` on HIP. **This is tuning to
each library's heuristics, not a property of the math**, and another card
or library version may want different limits. Writing ca as one plain GEMM
(`L2` as `[pqr, (f,K)]`, `k = n^3`) was 3.5× slower on cuBLAS at CAS(10,10).

Same session for each pair:

| | consume, emitted → BLAS | RDM build, emitted → BLAS |
|---|---:|---:|
| CUDA CAS(10,10) `--tiles 3` | 151ms → 55ms (2.7×) | 1.26s → 1.20s |
| CUDA CAS(10,10) `--tiles 3 --cublas` | 144ms → 60ms (2.4×) | 0.81s → 0.69s |
| HIP CAS(10,10) `--tiles 3` (emitted digest) | 328ms → 103ms (3.2×) | 3.81s → 3.57s |
| CUDA CAS(12,12) `--tiles 40` | 5.21s → 2.00s (2.6×) | 42.37s → 38.92s |
| CUDA CAS(12,12) `--cublas --tiles 80` | 5.22s → 2.11s (2.5×) | 13.91s → **10.85s** |
| HIP CAS(12,12) `--tiles 40` (emitted digest) | 10.93s → 2.80s (3.9×) | 145.40s → 137.92s |

At CAS(12,12) the CUDA BLAS consume runs at ~425 GFLOP/s per order, close
to the card's fp64 peak. It costs one extra `n^4` integral copy plus the
BLAS handle's workspace, and it does not change the tile floor.

## Memory and `--tiles`

### Tiling is a memory lever, not a speed lever

`--tiles` splits the RDM build over determinant tiles so that one tile's
transition blocks (`R`, `L2`, `W`) fit on the card. Within the range that
fits, the tile count barely changes the build time. At CAS(12,12) on the
RTX 3080, emitted digest, 18, 20 and 40 tiles all build in ~38s, and 40 and
150 tiles measure the same on a quiet card. **So use the fewest tiles that
are safely above the floor, and never fewer.** At CAS(10,10), too few tiles
corrupts device memory (`an illegal memory access`) instead of failing
cleanly. See "Why CAS(10,10) needs tiling" and "What too few tiles looks
like" below.

### Why CAS(10,10) needs tiling

At `norb = 10` each of the produce phase's transition blocks is
`norb^4 * n_det * 8` bytes, about 5 GB for `L2` at `n_det = 63,504`, with
`R` and `W` on top. That is why the build is tiled. `--tiles` (default 3,
`RdmBuildOptions::nTiles`) splits the determinant axis into that many tiles
up front, and the tile width is computed from the count, not sized from
however much free memory the driver reports.

That is deliberate. Sizing a tile from the reported free memory under-counts
the real peak at this active space: on the RTX 3080 it picked an unequal
2-tile split that **corrupted device memory** (`an illegal memory access`)
instead of failing cleanly. 3 tiles, an exact split of 63,504, is the
shipped count. Equal splits of 2 tiles (3 runs) and 1 tile (1 run) also
pass on the RTX 3080, but with little margin, and the unequal auto-sized
split that corrupted memory was measured on a build that held one more
`n^4 * width` block per tile than today's and has not been re-run. So
`--tiles 3` stays.

Each tile runs produce, the dm3 digest, then consume and the f3 digest for
each of the two orders, and adds its partial `dm3`/`f3ac`/`f3ca` into
persistent device accumulators (see "Device-resident RDM build"). The fdm2
correction and the wedge reconstruction need the fully summed partials, so
they run once, after the tile loop.

### What too few tiles looks like

Every GPU and BLAS status goes through one check, `nevpt2::gpuCheck`
(`nevpt2.wwr`). A failure prints
`GPU error at <file>:<line> in <function>: <NAME> (<description>)` to stderr
and calls `std::abort()`, so the process dies on `SIGABRT` (exit 134). There
is deliberately **no** out-of-memory policy: no retry and no automatic
re-tiling. The failures come in two kinds, and they mean different things:

- **An out-of-memory abort during the RDM build** means one tile's
  transition blocks do not fit the card at this width: **raise `--tiles`**.
  This is how CAS(12,12) fails below its floor. The message names the
  `allocDeviceBuffer` call in `buildRdmsDevice` that asked for the buffer,
  for example at `--tiles 16` on the RTX 3080:
  `GPU error at …/rdm_build.cpp:<line> in RdmBuildResult nevpt2::buildRdmsDevice(…): cudaErrorMemoryAllocation (out of memory)`.
- **`an illegal memory access` is not an out-of-memory signal.** The
  under-provisioned CAS(10,10) split above did not fail an allocation; it
  corrupted device memory, which surfaces from whichever check next touches
  the device. There is nothing to tune against: go back to the measured
  `--tiles 3`.

### Tile floors

The fewest tiles that run, with about 1 GB of the card held by other
processes (an idle card was not measured). Below the floor the run fails
with a clean out-of-memory **during the RDM build**, at the
`allocDeviceBuffer` call in `buildRdmsDevice`:

| case | card | floor | recommended |
|---|---|---:|---:|
| CAS(10,10), every variant | both | not measured; equal splits of 2 and 1 tiles pass on the RTX 3080 | `--tiles 3` (see "Why CAS(10,10) needs tiling") |
| CAS(12,12), emitted or BLAS | RTX 3080 | **17** (16 and 15: out of memory) | `--tiles 40` |
| CAS(12,12), `--cublas` | RTX 3080 | 17, but it declines there | `--tiles 80` (see `--cublas` above) |
| CAS(10,12), default digest, SC or `--pc` | RTX 3080 / RX 9060 XT | 14 / 7 | `--tiles 40` |

HIP's CAS(12,12) floor has not been measured. The per-tile working set
(`R`, `L2`, `W`; the build prints it as `per-tile transition blocks`) is
3423.7 MB at CAS(12,12) `--tiles 40` and 1647 MB at CAS(10,10) `--tiles 3`.
`--ozaki` adds its digit planes on top (see above).

### The memory pool

Every device allocation is a `DeviceBuffer` drawn from one stream-ordered
memory pool (`wwrMallocFromPoolAsync`, freed with `wwrFreeAsync` on the
same stream). Both demos print the pool's high-water marks after the RDM
build and for the whole run. RTX 3080 (the RX 9060 XT gives the same figures
at CAS(10,10) `--tiles 3` and CAS(12,12) `--tiles 40`, the two it was
measured at):

| run | pool used / reserved high-water |
|---|---:|
| CAS(10,10) `--tiles 3` | 1678.9 / 1696.0 MB |
| CAS(12,12) `--tiles 17` | 8156.1 / 8160.0 MB |
| CAS(12,12) `--tiles 40` | 3524.1 / 3552.0 MB |
| CAS(12,12) `--tiles 80` | 1812.3 / 1824.0 MB |
| CAS(12,12) `--tiles 150` | 1013.3 / 1024.0 MB |
| CAS(12,12) `--tiles 400` | 442.8 / 448.0 MB |

- **The peak is the RDM build's tile sweep,** even at 400 tiles. The
  per-tile buffers are freed when the sweep's scope closes, before the
  fdm2/wedge step and the energy stage allocate. Holding them longer made a
  tile count above the floor run out of memory in the energy stage, after a
  build that had succeeded. Energy intermediates are freed when their class
  returns.
- **The pool does not fragment its way past the floor.** The reserved peak
  is at most 28 MB above the used peak, on both cards.
- **The release threshold is `max`** (`--pool-threshold`, default `max`).
  `0`, `max` and 1 GiB gave identical high-water marks on both cards. The
  only timing difference was CUDA CAS(10,10)'s RDM build, ~0.05s slower at
  `0`, because the pool hands memory back and maps it again. The cost of
  `max` is that memory the pool holds is not available to cuBLAS's own
  workspace allocations. At 40, 80 and 150 tiles this did not matter.
- **The pool's fixed cost is small.** On the RTX 3080, `cudaMemPoolDestroy`
  takes 39 ms at exit, and the first growth of the pool makes its 83
  allocations cost 15 ms in total, against 3 ms for plain `cudaMalloc`.
  Together that is ~0.07s of wall time at CAS(10,10), and lost in the noise
  at CAS(12,12).
- **Every buffer is zero-filled,** because the buffer type has no
  uninitialised allocation. That costs at most 12 ms of GPU time per run
  (83 memsets, at CAS(12,12) `--tiles 17`, the widest tiles), 0.03% of the
  build. In exchange, no tile buffer can be read before it is written.

## One stream

All GPU work runs on one stream created `wwrStreamNonBlocking` by the demo's
`DeviceResources`. Every launch, copy, memset, BLAS call, event, allocation
and free is enqueued on it. Frees and downloads are ordered on the stream, so
almost no host-side synchronisation is needed. The few
`wwrStreamSynchronize` calls left each say what they order.

**A single call off that stream gives silently wrong answers.** A
non-blocking stream has no implicit ordering with the legacy default stream,
in either direction. So a stream-less `wwrMemcpy`, a synchronous
`wwrMalloc`/`wwrFree`, a launch with no stream, a library handle left on
stream 0, or a `nullptr` stream is not ordered against the real work. When
this happened here, the symptom was **intermittent wrong energies on an
identical binary and input**: some runs matched the golden to 1e-14, others
were off by 2e-4 or by 0.2 Ha, and the exit code changed from run to run,
because the race depends on GPU scheduling. A related, deterministic
version: a class energy downloaded with a stream-less copy, right after its
kernels were queued on another stream, read back exactly 0.

**No sanitizer catches this.** A kernel on the legacy stream reading a
buffer that `cudaFreeAsync` had freed on a non-blocking stream ran clean
under compute-sanitizer memcheck 3 times out of 3 (see
[`testing.md`](testing.md), "Sanitizers"). Repeated
runs are not evidence either. So the guard is textual:
`devtools/stream-lint.sh --strict` fails `cpp-tier.sh` and
`cross-backend-check.sh` on any stream-less or literal-stream call. It reads
text, not types, so a null stream that arrives through a variable or a
default member initializer is invisible to it. Don't write one.

## What did not help

Measured, and kept here so nobody repeats them without a new reason:

- **Several streams for the class energies.** One stream per class was
  meant to let the GPU overlap them. It measured the same as one stream
  (energy stage at CAS(12,12) `--tiles 40`: median 1.57s with seven streams,
  1.61s with none, 1.57s with one). The classes ran back to back anyway, and
  the energy stage's summed kernel time equals its wall time, so there was
  no overlap to gain. The energy stage is also 4% of the run at CAS(12,12).
- **Fewer or more tiles, for speed.** See "Tiling is a memory lever" above.
- **Dropping `R2`** (a permuted copy of `L2` the f3 digest used to read):
  it halved the CAS(12,12) floor (33 to 17 tiles) and the per-tile working
  set (6800.6 MB to 3423.7 MB at `--tiles 40`), and changed the speed by
  nothing measurable. The f3 digest takes the same time reading `L2`
  directly (one GEMM 290 ms before, 291 ms after).
- **The native-fp64 BLAS digest on CUDA,** and **`--ozaki` on either
  card.** See "The digests".
- **A lower pool release threshold.** See "The memory pool".
- **Consume as one plain GEMM, or unchunked.** See "Consume as a DGEMM".
- **WarpWraps' `wwr.tensor` (cuTENSOR / hipTensor) in place of the
  home-brewed einsum** (`src/einsum/`). Probed standalone in f64 at WarpWraps
  `b5b0cce`. On CUDA (cuTENSOR 2.8.1, RTX 3080), every primitive the port
  needs matched a brute-force host einsum to <=4e-15: contraction (outer
  products, batched/Hadamard modes, a mode summed in one operand only),
  repeated labels (`jbij`, `mjjn`) via a summed-stride view, reduction,
  permutation, and in-place elementwise-binary accumulate (the strided
  diagonal write included). Elementwise-binary *broadcast* gave wrong answers
  (not needed here), and there is no trinary contraction, so 3-operand
  einsums would split in two. On HIP (hipTensor, ROCm 7.2.4, RX 9060 XT
  gfx1200) it is unusable twice over. hipTensor has no f64 on this card (its
  README lists gfx90a, gfx942, gfx950 and gfx1250 only), and the 1e-7 check
  needs f64. Merely *linking* libhiptensor also aborts the process at load: a
  global constructor exits with "Cannot proceed: unsupported host device
  detected." before `main()`. Adopting it would have broken `ctest --preset
  hip` on this card or split the tree into a CUDA-only path. Revisit if
  hipTensor gains f64 on gfx12.

### Device-resident RDM build

The tile loop accumulates each tile's `dm3`/`f3ac`/`f3ca` digest output
straight into a persistent device buffer (`rdm_accumulate.cu`'s
`accumulateInplace`), instead of downloading each tile's partials, adding
them on the host, and discarding the device copy. The host then never blocks
during the sweep. **It is correct, and no faster.** An interleaved A/B at
CAS(12,12) `--tiles 40` gave 42.5s/42.7s, 42.5s/41.7s and 43.3s/42.1s
(host-accumulated / device-resident), within run-to-run noise.

The host round trip it removes was small. At CAS(12,12) each tile downloads
three `n^6` partials (24 MB in all) and adds them on the host, about 1 ms
per copy and 1.5 ms for the add, 40 times, against a ~42s build: ~0.7%. The
large transition blocks (`L2`, ~1.7 GB at a tile width of 21,168) were never
downloaded in either version. The build's time is kernel compute (see
"Where the time goes"), so **a faster RDM build has to come from faster
kernels**, not from the code that launches them. The device-resident version
stays, because a host round trip that isn't needed is better not had.
