# Testing

How this tree is checked: the two test tiers, why there is no CI, the local
gate, the sanitizer tier — what each tool catches, what it cannot, and
how the tier proves its own tools are on — and host code coverage. Every
figure names its card.

## Two tiers

- **The golden tier** is the numerical one. Each demo checks itself against
  a committed golden file (every class energy and the total to 1e-7) and
  prints `PASS:`; the ctest entry matches that line. The one non-demo binary
  in it, `nevpt2_sc_pseudodensity`, checks the gradient pseudodensities'
  exact identities the same way ("The SC pseudodensity identities", below).
  Every golden entry is labelled `gpu`, because there is no CPU path.
  `ctest --preset fast` runs every entry not labelled `slow`.
- **The unit tier** (`test/`, GoogleTest) checks host logic one component at
  a time — `test/<x>/` tests `nevpt2.<x>` — and is labelled `unit`
  (`ctest --preset unit`). A host-only suite has no `gpu` label and runs with
  no card visible; a suite that needs a card is `REQUIRES_GPU`. It proves
  logic (error kinds, propagation, index helpers), **never an energy**. Each
  binary also registers a `<target>.SuiteListIsComplete` drift guard
  (`cmake/README.md`, "Tests"). It is added from the root `CMakeLists.txt`
  only, so the standalone `cmake -S src` build has no gtest.

## No CI, and the local gate

There is no `.github/workflows/`, on purpose. A hosted runner has no GPU, so
it could prove only that the tree compiles and that the host helpers behave,
while every claim this project makes is numerical. A green badge that cannot
fail on a wrong answer invites the belief that something was checked.

**If you are sending a change, please don't add a workflow.** Run the gate
below on whatever card you have, and say in the pull request what you ran,
on which GPU and backend, and what it printed. A change that touches a
kernel or an energy needs a `PASS:` line from a real card. A change that
only compiles for the other backend says so.

So the gate is local, and **saying what you ran is the whole protocol**:

```bash
devtools/cpp-tier.sh                # CUDA, on a box with a card (runs both lints first)
devtools/cross-backend-check.sh     # does the OTHER backend still compile (lints too)
devtools/stream-lint.sh --strict    # every GPU call on the one stream; both scripts above run it
devtools/throw-lint.sh              # no `throw` in src/, apps/ or test/; likewise
```

`cross-backend-check.sh` proves the other backend *compiles*, not that it is
right: a numerical claim needs that backend's card. The `compile-cuda` /
`compile-hip` presets are compile-and-link-only configurations (testing
off).

## The SC pseudodensity identities

The golden tier's third kind of entry, beside the SC and PC energy checks:
`nevpt2_sc_pseudodensity_cas44`, `_cas88` and `_cas1010`, all `gpu`, run
`nevpt2_sc_pseudodensity` (`apps/sc_pseudodensity/`) and match
`PASS: SC pseudodensity identities hold`. What they assert is **exact**, not a
finite difference.

`src/gradient/` assembles the SC-NEVPT2 gradient's pseudodensities (Park's
Eqs. 41-47 — [`references.md`](references.md), "Analytical gradients"). A
class's norm and Dyall-Hamiltonian expectation value are quadratic forms in
that class's external integrals and linear in the active-space blocks they are
contracted against, so contracting the assembled pseudodensities back against
their own operands must reproduce the class energy — Park Eq. 40 — as an
algebraic identity. Each entry therefore checks, for all eight classes:

| check | against | tolerance |
|---|---|---|
| `(1/2) sum_blocks <x, D>` (Eq. 41's `D`) | the class energy from the amplitudes, `sum_t T_t N_t` | 1e-10 |
| `sum_G <G, M_G>` (Eqs. 42-43's `M` / `E`) | the same, less the one Sir term with no active block | 1e-10 |
| Srs' active integrals and hole RDMs contracted back (Eqs. 44-47) | `<K, E>` | 1e-10 |
| the amplitude energy, per class | `nevpt2.energy`'s own answer for the same state | 1e-10 |
| the amplitude energy, per class | the golden PySCF per-class energy | 1e-7 |

On an RTX 3080 (CUDA, Release, `--tiles 3`) **every** residual at CAS(4,4),
CAS(8,8) and CAS(10,10) is at or below 1.1e-14 — the largest are Sr's, the
class with the n^6 contractions — so the 1e-10 tolerance is four decades of
headroom, and a single mis-transcribed index does not fit inside it. The
assembly itself takes 0.00s / 0.10s / 0.45s at the three sizes, on top of the
RDM build and the class energies the entry also runs.

CAS(4,4) and CAS(8,8) are labelled `small` as well, so the sanitizer tier
covers the new device work; CAS(10,10) is not, for the reason the small cases
exist (below). The host half — the amplitude and multiplier formulas, and the
subscript rewriting that differentiates a class term — is `test/gradient/`,
card-free.

## Sanitizers

Some bugs show up only some of the time — a call left on the legacy stream
racing the one stream, or a stream-ordered free running before the last
kernel that reads the buffer. Repeated runs are weak evidence against those:
a race that fails 10% of the time survives 20 runs ~12% of the time. So the
tree has a sanitizer tier: five presets, small golden cases to run them on,
and canaries that prove each tool is actually on.

### Running them

```bash
# CUDA (RTX 3080)
cmake --preset asan  && cmake --build --preset asan  -j && ctest --preset asan
cmake --preset ubsan && cmake --build --preset ubsan -j && ctest --preset ubsan
cmake --preset compute-sanitizer && cmake --build --preset compute-sanitizer -j
ctest --preset compute-sanitizer                                  # memcheck (the default tool)
cmake --preset compute-sanitizer -DNEVPT2_COMPUTE_SANITIZER_TOOL=initcheck && ctest --preset compute-sanitizer
# (racecheck / synccheck the same way; switching tool needs no rebuild)

# HIP (RX 9060 XT): host only
cmake --preset hip-asan  && cmake --build --preset hip-asan  -j && ctest --preset hip-asan
cmake --preset hip-ubsan && cmake --build --preset hip-ubsan -j && ctest --preset hip-ubsan

# or through the tier script, which picks the test preset of the same name:
devtools/cpp-tier.sh --preset asan
```

Every sanitizer option defaults OFF, so `default`, `hip` and the other
presets are unaffected. Each sanitizer preset is a Debug build in its own
`build-<preset>/` directory, and its test preset runs only the entries
labelled `small`, the whole unit tier (`unit`) and the canaries that build
registered. The `compute-sanitizer` test preset drops the `death` suites:
`EXPECT_DEATH` forks, and compute-sanitizer does not follow the child. Those
suites still run under the four host presets.

**What a change touching streams, allocation or frees must be clean under:**
`asan`, `ubsan`, `hip-asan`, and `compute-sanitizer` with memcheck and with
initcheck, every canary green.

**memcheck's leak check is on by default**
(`NEVPT2_COMPUTE_SANITIZER_LEAK_CHECK`, ON; inert unless compute-sanitizer
runs memcheck). Both demos leak 0 device allocations, so a buffer that
outlives its context is a failing test: fix the leak, don't turn the check
off.

**LSan suppressions** go in `cmake/nevpt2_lsan.supp` and must name a vendor
library, never one of our frames. Today it holds one family: the ROCm
runtime leaks ~105 host allocations (~6 KB) of its own at exit under
`hip-asan`, with stacks entirely inside `libhsa-runtime64.so` /
`libamdhip64.so`. The CUDA runtime reports none.

### What each tool does, and what it does NOT catch

| tool | preset(s) | catches | does NOT catch |
|---|---|---|---|
| **AddressSanitizer** (+ LeakSanitizer) | `asan`, `hip-asan` | host heap/stack/global out-of-bounds, host use-after-free, host leaks at exit | anything in device memory or device code. Device code is uninstrumented on both backends: nvcc has no device ASan, and on HIP device ASan needs an `xnack+` target ID, which gfx1200 does not have (below) |
| **UBSan** (`-fno-sanitize-recover`) | `ubsan`, `hip-ubsan` | host signed overflow, misaligned/null access, bad shifts, invalid enum/bool loads, ...; the first finding aborts | device code (clang does not support `-fsanitize=undefined` for `amdgcn`, and nvcc has none) |
| **compute-sanitizer memcheck** | `compute-sanitizer` | device out-of-bounds and misaligned accesses, accesses to freed device memory (including a **stream-ordered** `wwrFreeAsync` that precedes the read in stream order), CUDA API errors, and with the leak check, device allocations never freed | uninitialized reads (initcheck's job); host memory; a **cross-stream race** (below) |
| **initcheck** | `compute-sanitizer` + `-DNEVPT2_COMPUTE_SANITIZER_TOOL=initcheck` | device global-memory reads of bytes nothing wrote | races, out-of-bounds |
| **racecheck** | ... `=racecheck` | shared-memory hazards *between threads of one block* (RAW/WAR/WAW on `__shared__`) in every kernel of ours (cuBLAS's are excluded, below) | anything in global memory, and anything *between kernels or streams* |
| **synccheck** | ... `=synccheck` | invalid barrier use *inside a kernel*: warps of one block at different `__syncthreads()`, bad `__syncwarp` masks | stream or event ordering between kernels |
| *(none, on HIP)* | | | ROCm has no compute-sanitizer counterpart; HIP gets host ASan/UBSan only |

**Nothing here detects a cross-stream race directly.** racecheck and
synccheck look at shared memory and barriers *inside one kernel*; no tool
checks the ordering *between* streams. memcheck catches a race's
*consequence* only when it is an access to memory already freed at the
moment of the access. Measured on the RTX 3080 with a scratch program: a long
kernel on the legacy stream reading a buffer while `cudaFreeAsync` frees it
on a non-blocking stream, with no ordering between the two, ran clean under
memcheck **3 out of 3 times**. The deterministic form of that bug — a free
that comes *before* the read in stream order — is reported every time (the
`free_before_read` canary), and still reported when the buffer is a
pool-backed `DeviceBuffer` whose freed block the demos' pool keeps mapped
(the `pool_free_before_read` canary). So the tier is strong evidence against "freed
before its last reader on the same stream", and **no evidence** against "a
call left on the legacy stream". That is `devtools/stream-lint.sh --strict`'s
job (see [`performance.md`](performance.md), "One stream").

**Device ASan on gfx1200 is not possible.** `rocminfo` reports `XNACK
enabled: NO`. ROCm 7.2.4's clang rejects the target ID outright ("invalid
target ID 'gfx1200:xnack+'"), and on plain `gfx1200` it ignores
`-fsanitize=address` for the device pass with a warning. So `hip-asan`
instruments host code only, and `add_device_library` adds
`-fno-gpu-sanitize` under the sanitizer options to make that explicit.

**racecheck excludes cuBLAS** (`--kernel-name-exclude kns=cublas`). Unscoped,
`nevpt2_cas88_cublas` failed with 32 hazard sites (~10^7 individual hazards),
every one between a read and a write inside cuBLASLt's
`cublasLt_fused_imma_dgemm_kernel_sm80`, the `--cublas` emulated DGEMM.
Every other entry, including `--blas-digest`'s native cuBLAS DGEMM and every
kernel of ours, was clean. A shared-memory hazard is internal to its kernel,
so no argument of ours can cause one; whether it is a cuBLAS bug or a pattern
racecheck cannot model, it is NVIDIA's, and the `--cublas` results still
match the golden to 1e-7. memcheck and initcheck stay unscoped, because a bad
argument of ours *does* fault inside a vendor kernel. synccheck runs
unscoped too, cuBLAS included, and is clean.

**The compute-sanitizer build leaves out `--cublas`'s declined case.**
`CublasEmulDeclineTests` (`test/cublas/`) asserts that `probe_emul_bits`
returns -1 on a shape cuBLAS declines. cuBLAS declines because its own
workspace `cudaMallocAsync` fails with `out of memory` (see
[`performance.md`](performance.md), "The cuBLAS fixed-point-emulation digest
(`--cublas`)"). memcheck reports each such failed call as an API error, 4 for
that case's two GEMMs, so the entry cannot be clean under it. Nothing we do
causes that failure, and the check is not loosened. The suite is compiled out
of the `compute-sanitizer` build and runs in every other build on the CUDA
card. The engaged cases (`CublasEmulDigestTests`, `CublasEmulProbeTests`) run
under memcheck and initcheck, and both report them clean.

**What the tier has caught.** initcheck found `--cublas`'s engaged-bits probe
reading uninitialized device memory (6,528 errors at CAS(4,4), inside
cuBLAS's `max_scale_pack_ker`): it ran on freshly allocated operands. The
probe now runs on the first tile's operands right after that tile's produce
wrote them (the first tile is always full width, so the probe's `K` is
exactly what was written); results and timings did not change. memcheck's
leak check found 63 (`nevpt2_demo`) and 54 (`nevpt2_df_demo`) device
allocations never freed; every allocation is now a self-freeing
`DeviceBuffer`, and both demos leak 0 at CAS(4,4) and CAS(8,8).

### The small cases

compute-sanitizer multiplies run time (racecheck worst of all), so the tier
runs on small goldens: CAS(4,4) and CAS(8,8) on N2/cc-pVDZ, conventional and
DF, from `generate_golden.py` with its usual single-threaded, 1e-12,
point-group-symmetric settings. All four pass its degeneracy guard,
regenerate byte-identically, and carry PC fields.

```bash
uv run python reference_data/generate_golden.py --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44
uv run python reference_data/generate_golden.py --ncas 4 --nelecas 4 --df --name n2_ccpvdz_cas44_df
uv run python reference_data/generate_golden.py --ncas 8 --nelecas 8 --name n2_ccpvdz_cas88
uv run python reference_data/generate_golden.py --ncas 8 --nelecas 8 --df --name n2_ccpvdz_cas88_df
```

| case | ncore | n_det | E_corr (conventional / DF) |
|---|---:|---:|---|
| CAS(4,4) | 5 | 36 | −0.2349210164 / −0.2349326445 |
| CAS(8,8) | 3 | 4900 | −0.2170319290 / −0.2170716671 |

**Nothing in the code assumes a size.** The RDM kernels take `norb`, the
string counts and the link counts at run time, and the tile plan is a
ceil-division of `n_det`: CAS(4,4)'s 36 determinants at `--tiles 3` make
three 12-wide tiles, narrower than the digests' 16-wide shared-memory tile.
`--cublas` engages (`bits=53`) at both sizes.

**CAS(4,4) does not exercise every class; CAS(8,8) does.** At CAS(4,4) the
active space is the four π orbitals and the five core orbitals are all σ, so
`Si` is zero by symmetry (norm 0, energy 0, in PySCF's reference too). Its
kernels still run, but a check against an exact zero says little. CAS(8,8)
gives all eight classes non-zero norms (`Si` 0.0159). So CAS(4,4) is the
fastest smoke case, and every path variant runs on CAS(8,8):

| ctest entry (label `small`) | what it adds |
|---|---|
| `nevpt2_cas44`, `nevpt2_cas88` | the default digest and BLAS consume, `--tiles 3` (the tile loop and the on-device accumulate run 3×) |
| `nevpt2_cas88_consume_emitted` | `kernels.cu`'s consume kernels |
| `nevpt2_cas88_profile`, `nevpt2_df_cas88_profile` | the event-bracketed launches (`--profile`), DF slab DGEMMs included |
| `nevpt2_cas88_blas_digest` / `nevpt2_cas88_digest_emitted` | the other backend's default digest (CUDA / HIP) |
| `nevpt2_cas88_cublas` | the emulated digest, its probe and the f3 scatter (CUDA only) |
| `nevpt2_cas88_ozaki`, `nevpt2_df_cas88_pc_ozaki` | the int8 Ozaki digest, SC and PC |
| `nevpt2_df_cas44`, `nevpt2_df_cas88` | density fitting with `--check-blocks`; the default `--batch 8` leaves a ragged last slab on both (19 virtuals = 8 + 8 + 3, 17 = 8 + 8 + 1) |
| `nevpt2_cas44_pc`, `nevpt2_cas88_pc`, `nevpt2_df_cas44_pc`, `nevpt2_df_cas88_pc` | `--pc`: the solver handle's eigensolves, the `d x d` BLAS GEMMs and the PC slab einsums, on both demos |
| `nevpt2_cas88_rdm_tangent` | `--rdm-tangent`: the tangent RDM build beside the plain one, checked against twice it (the derivative itself is checked against a finite difference in the unit tier, `test/rdm/rdm_tangent_tests.cpp`) |
| `nevpt2_sc_pseudodensity_cas44`, `nevpt2_sc_pseudodensity_cas88` | the SC gradient's pseudodensity assembly and its identities ("The SC pseudodensity identities", above) |

They are ordinary golden checks too, so `ctest --preset fast` and
`ctest --preset hip` run them on every backend.

### Canaries

A canary is a deliberately buggy run (`nevpt2_sanitizer_canary <mode>`, its
own small binary in `apps/sanitizer_canary/`, so neither demo carries a
planted bug). It passes only when the run fails **and** prints the tool's
report (`cmake/nevpt2_check_sanitizer_canary.cmake`; not `WILL_FAIL`, which
would credit a crash for the wrong reason). Each is registered only in the
build whose sanitizer must catch it, so **a green canary proves that
sanitizer is on in this build**. A canary that goes red means its sanitizer
is off or not failing runs: fix that, never delete the canary.

| canary | sanitizer | the bug | expected report |
|---|---|---|---|
| `nevpt2_canary_asan_heap_overflow` | ASan (both backends) | reads one past a `new double[64]` | `AddressSanitizer: heap-buffer-overflow` |
| `nevpt2_canary_ubsan_signed_overflow` | UBSan (both backends) | `INT_MAX + 1` | `runtime error: signed integer overflow` |
| `nevpt2_canary_memcheck_device_oob` | memcheck | a kernel reads one past its buffer | `Invalid __global__ read of size 8` |
| `nevpt2_canary_memcheck_free_before_read` | memcheck | `wwrFreeAsync(buf, S)`, *then* a kernel on S reads `buf` | `Invalid __global__ read of size 8` |
| `nevpt2_canary_memcheck_pool_free_before_read` | memcheck | the same, through a pool-backed `DeviceBuffer` at the demos' release threshold (the freed block stays in the pool) | `Invalid __global__ read of size 8` |
| `nevpt2_canary_memcheck_leak` (only with the leak check, the default) | memcheck `--leak-check full` | a 512-byte `wwrMallocAsync` never freed | `Leaked 512 bytes at` |
| `nevpt2_canary_initcheck_uninit_read` | initcheck | a kernel reads a fresh `wwrMallocAsync` buffer | `Uninitialized __global__ memory read of size 8` |
| `nevpt2_canary_racecheck_shared_race` | racecheck | thread 0 writes a `__shared__` slot that all 64 threads read, no barrier | `Race reported between` |
| `nevpt2_canary_synccheck_divergent_barrier` | synccheck | the block's two warps wait at two different `__syncthreads()` | `Barrier error detected` |

**Every canary goes red with its sanitizer off.** Run by hand through the
same verdict script with no launcher — against a plain Release build for the
host canaries, against `build-compute-sanitizer/`'s binary for the device
ones — each mode ran to completion, printed `nothing caught it`, exited 0,
and was failed ("Canary exited 0: the sanitizer did not fail the run"). The
ASan canary is red under `ubsan` and the UBSan canary under `asan`; on a
plain `hip` build both host canaries are red. The leak canary is red without
the leak flag (plain memcheck: `0 errors`). Two forms that look like the bug
and are **not** reported, both on the RTX 3080: a `__syncthreads()` that half
a block skips (synccheck is silent), and every thread writing a `__shared__`
slot and then reading its own value back (racecheck is silent, because each
read is forwarded from the thread's own write).

**The device canaries also prove the demo entries are wrapped.**
`CMAKE_TEST_LAUNCHER` is read when an executable target is *created*, not at
`add_test`. Set beside the root `CMakeLists.txt`'s `add_test` calls, it left
every demo entry unwrapped — memcheck "passed" in 0.4s per case — while the
canaries, handed the launcher explicitly, still passed. So the launcher is
set in `src/` before the targets exist (`cmake/nevpt2_sanitizers.cmake`), and
the canaries read it back off their own target's `TEST_LAUNCHER` property,
the same property ctest wraps the demos with. A launcher that fails to attach
turns them red.

**A finding can't hide behind `PASS:`.** `PASS_REGULAR_EXPRESSION` makes
ctest ignore the exit code. compute-sanitizer reports an error and lets the
demo carry on to print `PASS:`, and LSan reports at exit, after `PASS:`, so
either would pass on the regex alone. Every demo entry also carries a
`FAIL_REGULAR_EXPRESSION` on the tools' own report lines
(`ERROR SUMMARY: [1-9]`, `RACECHECK SUMMARY: [1-9]`, `ERROR: ...Sanitizer`,
`runtime error:`). Every unit-tier suite entry carries the same expression
(`cmake/add_gtest_suite_tests.cmake`). gtest's exit code already fails a
suite whose assertion fails, but a sanitizer report that leaves the exit code
alone would not.

### The unit tier under the sanitizers

Every unit-tier suite runs in each sanitizer preset. A `REQUIRES_GPU` suite
runs under compute-sanitizer and a host-only one runs bare
(`test/CMakeLists.txt`). Last run 2026-10-09/10: CUDA on the RTX 3080, HIP on
the RX 9060 XT, each preset's own Debug build, memcheck with the leak check.
**No suite needed a fix and no LSan suppression was added.**

| preset | entries run | unit entries | result |
|---|---:|---:|---|
| `asan` | 106 | 90 | all pass, canary green |
| `ubsan` | 106 | 90 | all pass, canary green |
| `compute-sanitizer`, memcheck + leak check | 103 | 84 (the 5 `death` suites dropped) | all pass, 4 `sanitizer_canary` entries green; all 36 GPU unit entries report `LEAK SUMMARY: 0 bytes leaked in 0 allocations`, `ERROR SUMMARY: 0 errors` |
| `compute-sanitizer`, initcheck | 100 | 84 | all pass, canary green |
| `hip-asan` | 104 | 89 | all pass, canary green |
| `hip-ubsan` | 104 | 89 | all pass, canary green |

The two `compute-sanitizer` rows were re-run last, after `cublas_emul_tests`
joined the tier. Their build has 89 unit entries: 70 suites (5 of them
`death`) plus 19 `<target>.SuiteListIsComplete` guards. The 84 that ran are
those less the 5 `death` suites. `CublasEmulDeclineTests` is compiled out of
that build (above). In both rows the unit tier added 25.6 s of `sec*proc`
(ctest's label summary). The two `hip-*` rows were re-run on the same tree,
with every unit binary in. Their 89 unit entries are 70 suites (6 of them
`death`, which run there, `CublasEmulStubDeathTest` among them) plus 19
guards: the HIP build has the `cublas_emul` stub suites where CUDA has the
digest ones. There the unit tier took 134–154 s of `sec*proc`. The `asan` and
`ubsan` rows were re-run on that tree too. Their 90 unit entries are 71 suites
(5 of them `death`) plus 19 guards: the compute-sanitizer build's 70 plus
`CublasEmulDeclineTests`, which is compiled in everywhere but there. In those
two rows the unit tier took 20–25 s of `sec*proc`.

### Wall times

Seconds per ctest entry: CUDA on the RTX 3080, HIP on the RX 9060 XT, each
sanitizer preset's own Debug build, racecheck with cuBLAS excluded,
memcheck with the leak check. "plain" is the same entry unsanitized
(`ctest --preset fast` / `hip`, Release).

| entry | asan | ubsan | memcheck | initcheck | racecheck | synccheck | hip-asan | hip-ubsan | plain CUDA | plain HIP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nevpt2_cas44` | 0.54 | 0.34 | 0.65 | 0.46 | 1.71 | 0.55 | 1.60 | 0.35 | 0.43 | 0.70 |
| `nevpt2_cas88` | 0.53 | 0.43 | 3.41 | 1.51 | 98.81 | 1.45 | 3.43 | 2.54 | 0.42 | 1.64 |
| `nevpt2_cas88_consume_emitted` | 0.46 | 0.38 | 3.55 | 1.65 | 98.89 | 1.30 | 3.51 | 2.72 | 0.36 | 1.65 |
| `nevpt2_cas88_profile` | 0.51 | 0.40 | 3.31 | 1.40 | 102.16 | 1.46 | 3.42 | 2.54 | 0.40 | 1.57 |
| `nevpt2_cas88_blas_digest` | 0.50 | 0.40 | 2.26 | 1.16 | 23.50 | 0.70 | — | — | 0.39 | — |
| `nevpt2_cas88_cublas` | 0.54 | 0.42 | 2.61 | 1.41 | 14.05 | 0.85 | — | — | 0.40 | — |
| `nevpt2_cas88_digest_emitted` | — | — | — | — | — | — | 5.82 | 5.02 | — | 1.64 |
| `nevpt2_df_cas44` | 0.42 | 0.30 | 0.46 | 0.46 | 2.92 | 0.45 | 2.28 | 1.37 | 0.30 | 1.41 |
| `nevpt2_df_cas88` | 0.52 | 0.42 | 3.31 | 1.46 | 104.19 | 1.44 | 3.42 | 2.50 | 0.41 | 1.64 |
| `nevpt2_cas44_pc` | 0.51 | 0.34 | 0.72 | 0.67 | — | — | 1.37 | 0.38 | — | — |
| `nevpt2_cas88_pc` | 0.67 | 0.50 | 4.22 | 2.17 | — | — | 2.91 | 1.60 | — | — |
| `nevpt2_df_cas44_pc` | 0.51 | 0.35 | 0.72 | 0.67 | — | — | 1.42 | 0.41 | — | — |
| `nevpt2_df_cas88_pc` | 0.67 | 0.51 | 4.17 | 2.17 | — | — | 2.94 | 1.64 | — | — |
| each canary | 0.36 | 0.35 | 0.75–0.86 | 0.49 | 0.53 | 0.77 | 0.37 | 0.36 | — | — |

The `nevpt2_cas88_profile` row was measured with the RDM-build-only profiler
that `--profile` has since absorbed. `nevpt2_df_cas88_profile` and the two
`--ozaki` small entries have not been timed under the tier; the `--ozaki`
pair has been run clean under memcheck (with the leak check), racecheck,
initcheck, synccheck, `asan` and `ubsan`, but not yet `hip-asan` /
`hip-ubsan`. racecheck and synccheck have not been run on the `--pc` entries.

Everything except racecheck finishes in well under a minute, so CAS(8,8) is
affordable under every tool. racecheck costs ~100s per CAS(8,8) entry on the
emitted digest's 16×16 shared-memory tiles (the BLAS paths are cheaper
because cuBLAS is excluded), ~7 minutes for the whole preset. memcheck takes
~8× the plain run at CAS(8,8), and initcheck ~3.5×. On HIP the sanitizer
builds are Debug, so their device code is `-O0` too, which is why
`nevpt2_cas88_digest_emitted` is ~3.5× the plain run there while host ASan on
CUDA costs almost nothing.

### What the tier leaves out, and why

- **A kernel-name *include* filter for racecheck/synccheck**
  (`--kernel-name kns=<namespace>`, plus a build check that every kernel
  mangles that namespace). racecheck *excludes* cuBLAS instead, and synccheck
  runs unscoped. An include filter would need every kernel in one namespace,
  and ours are not (`kernels.cu`'s sit in the global namespace). The exclude
  keeps every kernel of ours in scope with no naming rule to enforce.
- **A `no_sanitizer` label and a CI sanitizer preset.** No entry needs
  exempting, and there is no CI.
- **The large cases.** CAS(10,10) and up are too slow under
  compute-sanitizer. The salicylaldimine goldens stay out too, so the tier
  stays on N2.

## Coverage

clang source-based coverage of **host** code: which lines of `src/` and the
two demos under `apps/` the `fast` test set executes. It is a map of what the
tests never reach, not a measure of correctness — a line counted as covered
was run, not checked. The golden tier stays the only numerical claim.

```bash
devtools/coverage.sh                          # CUDA: configure, build, ctest, report
devtools/coverage.sh --preset hip-coverage    # the same on the AMD card
devtools/coverage.sh --html build-coverage/html --lcov build-coverage/coverage.lcov
```

The `coverage` / `hip-coverage` presets are **Release** builds with
`NEVPT2_ENABLE_COVERAGE` (cmake/README.md, "`nevpt2_coverage.cmake`"):
the configuration `fast` tests. Source-based counters go in before the
optimizer, so the counts are exact at `-O3`. A Debug build was tried first
and dropped: on HIP it compiles device code `-O0` too, the CAS(10,10)
`--digest-emitted` entries took ~104 s each on the RX 9060 XT, and
`nevpt2_cas1010_ozaki` had not finished after more than ten minutes (5.9 s under Release).
Each preset's test preset runs what `fast` runs, each process writing
`build-<preset>/profraw/<pid>.profraw`. The script merges those with
`llvm-profdata`, finds every binary carrying a coverage map, and reports
through `llvm-cov`, dropping `COVERAGE_IGNORE_REGEX` (devtools/config.sh:
`deps/`, `_deps/`, `test/`, `apps/sanitizer_canary/`, system headers).

ctest runs **serially**. Several golden entries at once on one card abort
(six of them did under `-j4` on the RTX 3080, every one passing when run
alone), and an aborted entry's host paths would then read as never executed.
A failing test still produces a report, but the script exits 1.

### What it cannot see

- **No `.cu`, on either backend** — kernels and their launchers alike. There
  is no llvm-cov for device code, and instrumenting only the host half of a
  `.cu` would differ by backend (nvcc's host compiler under CUDA, a `-x hip`
  CXX unit whose device pass would take the flags under HIP). A kernel's
  correctness is the golden tier's to prove.
- **Anything on the way to an abort.** A process that ends in `std::abort()`
  writes no profile: the `check`/`gpuCheck`/`narrowTo` tier, and every
  `EXPECT_DEATH` child. Those paths read as unexecuted even though the
  `death` suites exercise them. This is most of what `src/common/common.cppm`
  misses.
- **The `slow` entries.** CAS(12,12) and the salicylaldimine cases are not in
  `fast`, so a path only they reach (large tile counts, for one) is missed.

### Measured

`devtools/coverage.sh`, Release, clang 20.1.8, totals over `src/` + the two
demos (covered / total):

| card, preset | entries | regions | functions | lines | branches |
|---|---:|---:|---:|---:|---:|
| RTX 3080, `coverage` | 76/76 passed | 1338/1500 (89.20%) | 235/241 (97.51%) | 2982/3197 (93.27%) | 770/985 (78.17%) |
| RX 9060 XT, `hip-coverage` | 71/71 passed | 1316/1495 (88.03%) | 231/241 (95.85%) | 2942/3182 (92.46%) | 751/983 (76.40%) |

Each run merged one profile per entry from 13 instrumented binaries; the
ctest phase took 44.9 s on the 3080 and 143.5 s on the 9060 XT. A Debug build
of the CUDA preset gave the identical CUDA totals, as it should. The HIP
totals are lower mostly because `src/cublas/` there is the refusing stub
(`--cublas` is rejected at flag parsing on HIP), which nothing calls.

The lowest files are the two demos' `main.cppm` (the error-reporting branches
of argument and golden-file handling, which no ctest entry provokes) and
`src/common/common.cppm` (`narrowTo`'s failure path, which aborts).

