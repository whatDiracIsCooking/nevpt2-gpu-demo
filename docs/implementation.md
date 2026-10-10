# Implementation

How a number gets computed here, end to end: from a golden CASCI wavefunction
to the `PASS:` line. This is the *pipeline*; how the tree is layered, built and
split across vendors is [`architecture.md`](architecture.md).

Like that document, this one quotes **no measured numbers** — timings, tile
sweeps, scaling and negative results live in [`performance.md`](performance.md),
[`pc-nevpt2.md`](pc-nevpt2.md), [`reference-data.md`](reference-data.md) and
[`testing.md`](testing.md), cited below by section name.

```
  golden .nevpt2gold
        |
        |  CI vector, dm1, dm2, MO-integral blocks, PySCF's own answer
        v
  [1] link tables  (host, Knowles-Handy)
        |
  [2] produce      R[tu,K], L2[pqrs,K]      per determinant tile
        |
  [3] digests      dm3 = L2 . R             consume -> W,  f3 = L2 . W
        |
  [4] fdm2 + wedge once, on the fully-summed tensors
        |
        v
  dm3, f3ac, f3ca   (device-resident, never downloaded)
        |
        +-----------------------------+
        v                             v
  [5] SC energies                [5'] PC energies   (--pc)
      x^T S x, x^T K x                solve in span(S, K)
      146 einsums over slabs          one eigensolve pair per class
        |                             |
        v                             v
      normToEnergy (host)           E_t = -c sum_k y^2/(lambda+Delta)  (host)
        |                             |
        v                             v
      PASS: ... golden PySCF        PASS: PC-NEVPT2 ... block2
```

## 1. The input contract

The oracle is a committed binary written offline by `generate_golden.py` — the
**only** file that defines the CASCI state. It carries the CI vector, the
cheap host-computed `dm1`/`dm2`, every MO-integral block, the orbital
energies, and PySCF's own per-class NEVPT2 answer. No Python and no PySCF run
at build or test time.

Most goldens also carry `pc_class_energies`/`e_pc_total` — **block2's**
partially-contracted answer, because PySCF implements no PC. `--pc` refuses a
golden lacking those two fields before the RDM build even starts.

Two generator settings are load-bearing and must not be dropped to make
generation faster: forced single-threading with a 1e-12 Davidson tolerance
(without them both paths wandered in `E_corr`), and point-group symmetry
(without it N2's exactly degenerate virtual pairs get a noise-chosen rotation
that the `Srs` class is not invariant to). See [`reference-data.md`](reference-data.md), "Regenerating the
golden file".

## 2. Stage 1 — the RDM build (`src/rdm/`)

`nevpt2.rdm_build` turns the CI vector into three device-resident
`(n_act,)^6` tensors: the active-space 3-RDM `dm3`, and the two
integral-contracted 4-RDM digests `f3ca` and `f3ac`. **The full 4-RDM is never
formed** — that is the memory wall the digests exist to avoid.

It is shared by both demos, and deliberately so: the only integrals it touches
are the **active** `h2e` (`n_act^4`), which the density-fitted demo rebuilds
exactly from `B_aa`. The build does not know or care which demo called it.

### 2.1 Link tables (host)

`nevpt2.link_tables` builds Knowles–Handy forward and reverse link tables in
PySCF's `fci.cistring` convention — pure host C++, the main host-resident part
of the build (the other is the BLAS consume's one permuted `eri` copy, §2.4). Each row is `(creation, annihilation, target-string, sign)`.

### 2.2 The tile plan

The determinant axis is split into `--tiles` tiles, and the tile width is
**computed explicitly** rather than asked of the driver. `--tiles` is a
**memory-capacity lever, not a speed lever**: [`performance.md`](performance.md),
"Memory and `--tiles`", records the sweeps that established this at both
active spaces.

This is the one place where getting it wrong is actively dangerous. An
under-provisioned tile count at CAS(10,10) **corrupts device memory** (`an
illegal memory access`) rather than failing cleanly, and at CAS(12,12) it
fails *during* the RDM build. **Never lower a tile count to make something
faster.** The measured floors per case are in docs/performance.md, "Tile
floors", and why CAS(10,10) needs tiling at all in docs/performance.md, "Why
CAS(10,10) needs tiling".

### 2.3 Produce: `R` and `L2` (`kernels.cu`, K1)

One thread per determinant `K` in the tile, where `K` decomposes into an alpha
string `A = K / nb` and a beta string `B = K % nb`:

| | what | how |
|---|---|---|
| `R[tu, kl]` | `(E_tu \|0>)[K]` | a **reverse**-link gather: sources `A'` with an excitation `A' -> A` (beta spectator), then the beta half |
| `L2[pqrs, kl]` | `(E_pq E_rs \|0>)[K]` | **two nested forward walks** over the link tables — alpha-then-alpha, alpha-then-beta, beta-then-alpha, beta-then-beta |

Both are laid out `[index, K]` at row stride `width`, which is what makes the
next step a GEMM.

`kernels.cu` holds eight such CAS-independent kernels and is compiled **once**,
serving any `(norb, nelec)` at run time — `norb`, `na`/`nb`, `nla`/`nlb` are
kernel arguments, not codegen. (It is `#include`d by `rdm_launch.cu`; see
[`architecture.md`](architecture.md) §3.)

### 2.4 The digests

**`dm3`** is one large GEMM, `dm3[pqrs, tu] = sum_K L2[pqrs, K] R[tu, K]`,
summed across tiles into a zeroed final accumulator — how depends on the
digest path (§2.5; the per-path table below).

**`f3ca`/`f3ac`** take two steps. First *consume* contracts the active `h2e`
into `W`:

```
  ca:  W[a,f,K] = sum_{pqx} eri[a,x,q,p] * L2[p,q,x,f,K]
  ac:  W[a,f,K] = sum_{pqx} eri[a,x,q,p] * L2[p,q,f,x,K]
```

The two orders differ only in which `L2` slot the third contracted index `x`
and the free index `f` occupy, so both run the same loop. That loop runs two
ways on both backends: by default as BLAS — for each `x`, one strided-batched
`wwrblasDgemm` over `f` against `E[x,a,p,q] = eri[a,x,q,p]`, a permuted copy of
the integrals built once on the host — or, with `--consume-emitted`, as the
`consume_{ca,ac}_generic` kernels in `kernels.cu`
(docs/performance.md, "Consume as a DGEMM"). Then the f3 digest:

```
  f3[r*n^2 + fr*n + a]  (ca)  =  sum_K W[a*n + fr, K] * L2[r, K]
  f3[r*n^2 + a*n + fr]  (ac)  =  sum_K W[a*n + fr, K] * L2[r, K]
```

That form is why there is no `R2`. The digest's target row **is** `L2`'s own
row `r = wvut` — so letting the contraction's `N` index run over `L2`'s rows
directly removes the need for `R2`, a full permuted copy of `L2` (the same
`n^4 * width`) that previously existed only to put the index in `tuvw` order.
No path allocates `R2` any more, and no `r2_transpose` kernel exists. What
that bought — a lower tile floor at CAS(12,12), with the speed unchanged — is
in docs/performance.md, "What did not help".

**Tiles accumulate on the device**, replacing a download / host-add /
re-upload round trip per tile. Where the add happens depends on the digest
path:

| | GEMM digests (`--blas-digest`, `--cublas`) | emitted digest |
|---|---|---|
| `dm3` | the GEMM itself, `beta = 1` into the final accumulator | `digest_dm3_generic` overwrites a per-tile buffer; `rdm_accumulate.cu`'s `accumulateInplace` adds it in |
| `f3ca` | GEMM into a temporary (`beta = 0`), then `f3_scatter.cu`'s permute-scatter `+=` | `f3_digest.cu` adds `+=` into the accumulator directly |
| `f3ac` | GEMM into a temporary, then `accumulateInplace` (its target index is the source index) | as `f3ca` |

With `--fused-digest` the GEMM column collapses to one call: produce writes
`R` and both consumes write `W_ca`/`W_ac` into one stacked `[3·n^2, width]`
operand, a single GEMM (`beta = 0`) writes the `(n^4, 3·n^2)` product into
the temporary, and `f3_scatter.cu`'s `fusedDigestSplit` adds its three
column blocks into `dm3`, `f3ca` (with the ca transpose) and `f3ac`.

### 2.5 Four digest paths, and which is default

The same three digest GEMMs can run four ways. They differ in speed, not in
what they compute:

| path | flag | available |
|---|---|---|
| emitted | `--digest-emitted` | both backends — hand-written tiled kernels |
| native-fp64 BLAS | `--blas-digest` | both backends — `wwrblasDgemm` (cuBLAS / hipBLAS) |
| fixed-point emulation | `--cublas` | **CUDA only** — `cublasGemmEx` with `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` |
| int8 Ozaki | `--ozaki` | both backends — `src/ozaki/`, hand-written on int8 tensor cores through WarpWraps' `<wmma.h>` |

**The default differs by backend**: BLAS on HIP, emitted on CUDA, each because
that is what measured faster on the card in hand (docs/performance.md, "Native-fp64 BLAS
digest"). A HIP timing recorded before that switch is an emitted one.

`--ozaki` is the scheme `--cublas` gets from cuBLAS, written out: each fp64
operand row is scaled by a power of two and split into eight 7-bit int8
digits (exactly), every digit pair is multiplied on the int8 tensor cores with
an exact int32 accumulator (pairs with the same `p+q` share one), and the
weighted levels are summed in fp64. With all 64 pairs (the default) the
result differs from native DGEMM only at fp64 rounding. `--ozaki-pairs P`
drops the pairs with `p+q > P`; `--ozaki-check` runs native DGEMM beside it
and prints the largest difference. It is not the default anywhere: it is
slower than the default digest on both cards (docs/performance.md,
"The int8 Ozaki digest"). Its digit planes are as large as `L2`, so it
doubles the per-tile working set.

`--cublas` **prints whether emulation actually engaged**, and the line must be
read before quoting any `--cublas` number: `engaged bits=53` is the win, while
`engaged bits=-1 (DECLINED -> ran native fp64!)` means cuBLAS quietly took the
slow path and the measurement is meaningless. At CAS(12,12) it also wants a
higher tile count than the emitted path before it will engage at all — see
docs/performance.md, "Tile floors".

### 2.6 fdm2 and wedge

Run **once**, after the tile loop, on the fully-summed `dm3`/`f3`: the `fdm2`
correction and the wedge reconstruction, each in both `ca` and `ac` orders.
They need the complete tensors, not a per-tile slice.

The three accumulators then move into the result as owning `DeviceTensor`s.
Nothing is downloaded — the energy stage reads them in place.

## 3. Stage 2 — the eight SC class energies (`src/energy/`)

For each class and each external tuple `t` (a virtual `r`, a core `i`, a pair,
...), with `x_t` the tuple's integral vector:

```
  N_t = x_t^T S x_t                  "norm" einsums
  H_t = x_t^T K x_t                  "ener" einsums
  E_class = -sum_t N_t / (Delta_t + H_t/N_t)     over |N_t| > NUMERICAL_ZERO
```

`S` is the metric of the class's internal functions (active indices only),
`K = <Phi_mu|(H_D - E_0)|Phi_nu>`, and **neither depends on the external
tuple**. `Delta_t` is the tuple's orbital-energy sum. The final reduction is
`normToEnergy`, a verbatim transcription of `pyscf.mrpt.nevpt2._norm_to_energy`
including its vanishing-norm guard.

The eight classes, in the fixed order PySCF evaluates them and
`generate_golden.py` writes them: `Sr`, `Si`, `Sijrs`, `Sijr`, `Srsi`, `Srs`,
`Sij`, `Sir`.

### 3.1 One generic einsum, 146 call sites

The reference is a NumPy implementation of ~140 `np.einsum` calls. Rather than
hand-deriving 140 index loops, there is **one generic device contraction
kernel**, and the call sites keep the *same subscript strings*:

```cpp
DeviceTensor ener = deviceEinsumNew("ipqr,pqrabc,iabc->i", {&h2e_v, &a16, &h2e_v}, 1.0, dr);
deviceEinsumAccum("ipqr,pqra,ia->i", {&h2e_v, &a17, &h1e_v}, ener, 2.0, s);
deviceEinsumAccum("ip,pa,ia->i",     {&h1e_v, &a19, &h1e_v}, ener, 1.0, s);
```

`src/energy/energy.cpp` holds **146** `deviceEinsum*` calls plus **10**
`deviceTranspose*` ones. The port is therefore a mechanical, checkable
transcription rather than 140 opportunities for a new index bug. (How it
was validated is "Porting the class energies", below.)

The host planner (`src/einsum/einsum.cpp`) parses a subscript string into the
POD `EinsumPlan`, whose sizes — 3 operands, rank 6, 6 contracted labels, 8
distinct labels — are the *exact maxima measured across every call*, not
round-number guesses; `buildPlan` aborts through `check()` if a call ever exceeds them.

Two kernel decisions worth knowing:

- **One thread per `(output element, contracted-index tuple)`**, not per
  output element. Several calls here have a *small* output contracted over a
  *large* product of summed labels — `"ipqr,pqrabc,iabc->i"` has output size
  `nvirt` but sums over six active-space labels. One-thread-per-output would
  leave almost the whole card idle. The cost is an `atomicAdd` per term, which
  **reassociates the sum** — which is exactly why every comparison is against
  a 1e-7 tolerance and never bit-identical.
- **cuTENSOR / hipTensor was tried and rejected** (2026-10-06), and the
  negative result is recorded in `src/einsum/einsum.cppm`: hipTensor has no
  fp64 on gfx1200, and merely *linking* it aborts the process in a global
  constructor before `main()`. Adopting it would have broken the green HIP run
  or split the one tree. Revisit only if that changes.

Three hand-written "diagonal slice" kernels cover the accumulations the NumPy
reference does with plain indexing rather than an einsum
(`a16[:, i, ..., i] += fdm2` and the two `a22` variants).

#### Porting the class energies

The danger in porting ~140 `np.einsum` calls is not a crash but a plausible
wrong number: one mis-derived output index and the demo reports a believable,
incorrect correlation energy. So the port separated two questions, each
checked on its own:

1. **Was each einsum transcribed correctly?** An all-host C++ reference came
   first, on a generic host `einsum` that took the same subscript strings as
   NumPy (any number of operands, any rank, repeated labels within an operand
   as NumPy's diagonal convention). Every call site was a line-for-line
   transcription; the few manual slice-assignments were hand-translated with
   the index correspondence in a comment. It was checked against golden
   per-class PySCF energies at CAS(4,4) and CAS(6,6), from `dm3`/`f3ac`/`f3ca`
   computed independently in Python, before any device code existed.
2. **Is the device kernel right?** The generic kernel was tested standalone
   first (a matmul, a diagonal and a trace, a full reduction to a scalar, a
   transpose, a genuine three-operand contraction, the small-output /
   large-contraction shape, and the accumulate/scale semantics). Then the full
   device energy computation was checked against the same CAS(4,4)/CAS(6,6)
   data, independently of the real RDM build, before it was wired into the
   demo.

Once the device port matched the golden per-class energies directly, the
host reference was removed: a serial copy of the same calls added nothing the
golden check does not already cover. `nevpt2::transpose` is what survives of
that host layer, in `src/tensor/`. Today the golden per-class check is the
backstop for a transcription slip.

### 3.2 The `make_a*` intermediates

Each class builds its RDM-only intermediates **once**, before the slab loop,
and frees them when the class returns:

| class | metric `S` | Hamiltonian `K` |
|---|---|---|
| `Sr` | `dm3` | `a16`, with `a17`/`a19` for the one-body terms |
| `Si` | `dm3_h`, `dm2_h`, `dm1_h` (hole-side) | `a22`, with `a23`/`a25` |
| `Sijr` | `hdm1` | `a3` |
| `Srsi` | `dm1` | `k27` |
| `Srs` | `rm2` | `a7` |
| `Sij` | `hdm2` (with `hdm1`, `hdm3`) | `a9` |
| `Sir` | a 2x2-block form over `dm1`/`dm2` | `a12`, `a13` |

`make_a16` and `make_a22` read the `f3ac`/`f3ca` digests — they are the only
consumers of stage 1's 4-RDM contractions. Several of these builders are
declared in the internal partition `nevpt2.energy:shared`
(`energy_shared.cppm`) — nothing in it is exported, so they have module
linkage, visible to every unit that does `import :shared;` and to no importer —
because PC reuses them verbatim (§4): `energy.cpp` and `energy_pc.cpp` both
import it.

### 3.3 Slabs: how the external integrals arrive

Every class's output carries at least one external index, and each output
element needs only the integral entries carrying **its** external index. So
each class walks that index in batches and asks the `IntegralSource` for just
that slab:

| block | layout | batched axis |
|---|---|---|
| `Sr` | `(v,a,a,a)` | 0 |
| `Si` | `(a,a,c,a)` | 2 |
| `Sijrs` | `(c,v,c,v)` | 0 |
| `Sijr` | `(v,a,c,c)` | 0 |
| `Srsi` | `(v,v,c,a)` | 0 |
| `SrsiT` | `(v,v,c,a)` | 1 |
| `Srs` | `(v,v,a,a)` | 0 |
| `Sij` | `(a,a,c,c)` | 2 |
| `Sir1` | `(v,a,c,a)` | 0 |
| `Sir2` | `(v,a,a,c)` | 0 |

Each axis is chosen so that every output element depends on exactly one index
along it, which is what makes the per-slab energies simply add. `SrsiT` is the
`Srsi` block again batched along its *second* axis, because `Srsi` contracts
`h[r,s,..]` against `h[s,r,..]` — a batch of `r` needs both.

The per-slab `norm`/`H` are read back with `downloadTensor`, which
synchronizes the stream itself, so **no class synchronizes first**; each
download is already enqueued behind the kernels that wrote its source.

### 3.4 `Sijrs`, the exception

`Sijrs` touches no RDM at all — it is the MP2-like class — so it gets a
dedicated reduction kernel instead of einsums. Per element it forms
`theta = 2*g[i,J,a,b] - g[i,J,b,a]`, then accumulates `norm += g*theta` and
`ener += g/Delta * theta` in a grid-stride loop, folding each through
`block_reduce` (warp shuffles) into one `atomicAdd` per block per quantity.
Its slab is
`(ni, nvirt, ncore, nvirt)`, so the full `(ncore*nvirt)^2` block never has to
exist anywhere.

### 3.5 What stays on the host

Only the finishing arithmetic: assembling `Delta_t` (e.g. `e_virt(r) +
e_virt(s)` for `Srs`, `-(e_core(i) + e_core(j))` for `Sij`) and
`normToEnergy`. These operate on tensors of size `nvirt`, `ncore`, or
`nvirt*ncore` — a handful to a few hundred doubles. There is nothing there for
a GPU to accelerate.

## 4. Stage 2' — PC-NEVPT2 (`src/energy/energy_pc.cpp`, `--pc`)

### 4.1 PC is SC without the final contraction

SC collapses `S` and `K` to two scalars per tuple. PC keeps the whole span of
the class's internal functions:

```
  E_t = -c * b_t^T (K + Delta_t S)^+ b_t,     b_t = S x_t + S_1 h1_t
```

In one dimension this reduces exactly to the SC expression. `K + Delta_t S` is
block2's `H_D`.

**So PC needs no new RDM intermediates.** It calls the *same* `make_a16`,
`make_a7`, `make_hdm1`, `make_a3`, `make_k27`, `make_a9`, `make_a12`/`make_a13`
that SC does — that is why they live in the `nevpt2.energy:shared`
partition, which both `energy.cpp` and `energy_pc.cpp` import. SC's one-body `K`
entries (`a17`, `a19`, `a23`, `a25`) are **not needed**: the one-body part of
`V|Psi_0>` enters only as its projection onto the span, `S_1 h1_t`.

`Sijrs` has no active index, so PC = SC there — `pcEnergiesDevice` calls
`energy_Sijrs` itself, not a copy.

### 4.2 One solve per class

Because `S` and `K` are the same for every tuple and only `Delta_t` and the
right-hand side vary, each class is diagonalised **once** (`solveClass`):

1. `S = V s V^T` — `wwrsolverDnDsyevd` on the symmetrised `S`
2. drop modes with `s_k <= PC_TAU * s_max` (`PC_TAU` = 1e-13);
   `X = V_kept s_kept^-1/2`
3. `K' = X^T K_sym X` (two GEMMs); `K' = W lambda W^T` — a second `Dsyevd`
4. `U = X W`; on the kept range `(K + Delta S)^+ = U diag(1/(lambda+Delta)) U^T`
5. precompute `T = S U` and `t_1 = S_1^T U`

Then `y_t = T^T x_t + t_1^T h1_t`, so **each slab is one einsum straight into
the eigenbasis** — `pc_Sr`'s is `"ipqr,pqrk->ik"` against SC's three-operand
`"ipqr,pqrabc,iabc->i"` — and
`E_t = -c sum_k y_tk^2 / (lambda_k + Delta_t)` finishes on the host, exactly as
`normToEnergy` does. `src/energy/energy_pc.cpp` needs **29**
`deviceEinsum*` calls plus **1** `deviceTransposeAccum` for all eight
classes.

`K` is symmetrised to `(K + K^T)/2`, because `<Phi|[H_D,Phi]>` is Hermitian
only for an exact eigenfunction and a quadratic form defines only the
symmetric part anyway.

Two alternatives were rejected, with reasons in [`pc-nevpt2.md`](pc-nevpt2.md), "The solve:
once per class, not once per tuple":

- **`Dsygvd`** (generalised eigensolver) Cholesky-factors `S`, and `S` is
  **singular by construction** — `Sr`/`Si` drop exact null modes at every
  active space.
- **block2's per-tuple `lstsq`** means one `d x d` SVD *per external tuple*.
  The once-per-class solve does one eigendecomposition per class, and the two
  agree closely on every committed PC golden.

### 4.3 The gap is checked, not assumed

Each class result carries `S`'s spectrum as the cut saw it — basis dimension,
dropped count, largest dropped, smallest kept, the ratio — and the class is
**refused outright, with no energy**, when

```
  smallest_kept / largest_dropped  <  PC_MIN_GAP   (1e3)
```

(or `/ PC_TAU` when nothing was dropped). A non-positive denominator
`lambda_k + Delta_t` refuses the class too, rather than dividing through a
sign change. This is `generate_golden.py`'s `_check_pc_conditioning` rule
applied to `S`.

It has a visible consequence in the golden set: **N2 CAS(12,12) has an
ungapped metric** (a 1s orbital in the active space), so those files carry no
PC fields at all, and `n2_ccpvdz_cas1012` — CAS(10,12), both 1s in the core —
is the 12-orbital PC case instead.

## 5. The check

Each demo ends with a gate line, and the two are worded differently on purpose
so a `*_pc` ctest entry cannot pass on SC output or vice versa.

**SC** (`apps/integral_direct/main.cppm`): all eight per-class norms *and*
energies *and* the total within `kAtol` = 1e-7 of the golden PySCF answer,
plus `norms[c] >= -kAtol`, then

```
PASS: per-class norms/energies and the total match the golden PySCF reference to 1e-07
```

**PC** (`printPcReport`): all eight classes computed — none refused, none
`NotYet` — each within 1e-7 of block2's `pc_class_energies`, **and** the sum
within 1e-7 of `e_pc_total`. The total is checked on its own rather than taken
as implied, because eight deltas just under tolerance could add up past it.

Either way the exit status is the verdict, which is what the ctest entries
match on. The demo is the test of every number: the GoogleTest unit tier
(`test/`) checks host logic only, never an energy.

## 6. The density-fitted variant

`nevpt2_df_demo` changes **one** thing: `DfIntegralSource` builds each external
slab on the device from the three-index `B_*` tensors (one `wwrblasDgemm` plus
a permutation) instead of viewing an uploaded full block, and rebuilds the
active `h2e` from `B_aa`. Every einsum subscript, the whole RDM build, and
both energy paths (SC and PC) are untouched.

It is checked against a **different** reference — PySCF's DF-NEVPT2, a
different CASCI state, the `*_df` goldens — not against the conventional
numbers. `--batch` bounds slab length and is a memory lever with no measured
benefit at the committed sizes ([`reference-data.md`](reference-data.md), "Density fitting").

## 7. Where to look next

| for | read |
|---|---|
| vendor layering, modules, bridges, streams, build | [`architecture.md`](architecture.md) |
| every measurement, sweep and negative result | [`performance.md`](performance.md), [`pc-nevpt2.md`](pc-nevpt2.md), [`reference-data.md`](reference-data.md), [`testing.md`](testing.md) |
| tile floors, flag defaults, what to run before claiming anything | [`performance.md`](performance.md), [`testing.md`](testing.md) |
| the CMake macro layer | `cmake/README.md` |
