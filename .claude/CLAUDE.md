# CLAUDE.md

## What this is

`nevpt2-gpu-demo` — an end-to-end **SC-NEVPT2** (and, with `--pc`,
**PC-NEVPT2**) calculation on the GPU. Given a CASSCF/CASCI wavefunction (a CI
vector plus MO integrals), it builds the active-space 3-RDM (`dm3`) and the two
integral-contracted 4-RDM digests (`f3ac`/`f3ca`) on the device, assembles the
eight perturber-class energies **also on the device**, and checks every class
and the total correlation energy against a committed reference — PySCF for SC,
block2 for PC.

It is a **numerical-correctness-first** project: every claim it makes is
"matches the golden reference to 1e-7", and the golden files are the oracle.
Measured results, **including negative ones**, live in four current-state
files: `docs/performance.md` (what it costs, which knobs matter, and "What did
not help"), `docs/pc-nevpt2.md`, `docs/reference-data.md` and
`docs/testing.md`. **Do not quietly "fix" a recorded null result into a claim
of improvement.** `README.md` is the summary and carries no measured numbers.

## Two backends, one tree via WarpWraps

| | here |
|---|---|
| language | host C++23 named modules, clang + libc++ (gcc refused), `import std;`; device code is **C++20** |
| one `src/` tree | `import wwr.runtime_api;` (re-exported by `nevpt2.wwr`) |
| WarpWraps | submodule in `deps/`, `add_subdirectory`'d, `EXCLUDE_FROM_ALL` + `WWR_COMPILE_TIME_ONLY` (`cmake/nevpt2_warpwraps.cmake`) |
| kernels | `.cu` → STATIC device library, linked in, runtime-API launch, via `add_device_library` (over WarpWraps' `wwr_add_gpu_device_library`; + `parallel_for` for `rdm_accumulate`/`f3_scatter`), called through a `*_bridge.h`; `kernels.cu` through `rdm_launch.cu`, which `#include`s it. Nothing is loaded at run time (docs/architecture.md) |
| CMake macros in `cmake/` | `add_cxx_module_library`, `add_device_library`, `add_gtest_executable`, `add_gtest_suite_tests`; see `cmake/README.md` |
| tests | **two tiers** — the demos check every number against the golden and print `PASS:`; a GoogleTest unit tier (`test/<component>/`, googletest `v1.17.0` by `FetchContent`, only when `NEVPT2_BUILD_TESTING` is ON, every entry labelled `unit`) checks host and kernel logic, never an energy |
| oracle | **committed golden files**, not a CPU LAPACK reference |
| install tier / exported package | **none** — nothing is installed |

**How the one tree splits by backend** (`NEVPT2_GPU_BACKEND`, CUDA default):

- **Host code** names the GPU runtime only as `wwr*` (`wwrMalloc`,
  `wwrStream_t`, `wwrMemcpyHostToDevice`, …), written **bare** inside
  `namespace nevpt2`, never `wwr::wwrX`. `src/wwr/` (`wwr.h` / the
  `nevpt2.wwr` module) `using`s them in; a name not yet listed gets its
  `using` added there; `wwr::extension::*` and code outside the namespace stay
  qualified. Host code reaches all of it through `import nevpt2.wwr;`, which
  re-exports `wwr.runtime_api` and the vendor-free `nevpt2.error_handling`,
  and defines `gpuCheck`. `NEVPT2_GPU_BACKEND` is forwarded as
  `WWR_GPU_BACKEND` and the names bind to `::cuda*` or `::hip*`. **Prefer a
  `wwr*` name to a vendor one, always.**
- **Kernel files** (`*.cu`) are C++20 and `#include <runtime.h>` (WarpWraps),
  which picks the vendor header off `__CUDACC__`/`__HIP__`. On HIP they are
  CXX sources that clang compiles with `-x hip` (not hipcc). They cannot
  `import`; a POD shared with the host (`einsum/einsum_plan.h`) stays a
  header, reached root-relative.
- **The one host file allowed to name a vendor API** is
  `src/cublas/cublas_emul.cpp` — CUDA-only by design (`--cublas`, below) and
  compiled to refusing stubs on HIP. Everything else reaches the GPU only
  through `wwr*` names and `runtime.h`. A raw `cu*`/`cuda*`/`hip*` call
  anywhere else breaks the other backend; `devtools/cross-backend-check.sh`
  is what catches it.
- **`--cublas` is CUDA-only, on purpose:** CUDA 13.0's
  `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` has no ROCm equivalent.
  `nevpt2.cublas_emul` is real under `NEVPT2_HAVE_CUBLAS_EMUL` (CUDA builds)
  and refusing stubs otherwise; a HIP binary rejects `--cublas` at flag
  parsing. `f3_scatter.cu`, the scatter `--cublas` uses, is built on **both**
  backends; on HIP its caller is the BLAS digest (the same digest GEMMs as
  native-fp64 `wwrblasDgemm`). **The default digest differs by backend:** BLAS
  on HIP (`--digest-emitted` for the A/B), emitted on CUDA (`--blas-digest` to
  run BLAS) — docs/performance.md, "Native-fp64 BLAS digest".
- **`--ozaki` is the int8 tensor-core route on BOTH backends** (`src/ozaki/`):
  the Ozaki scheme `--cublas` gets from cuBLAS, hand-written through
  WarpWraps' `<wmma.h>` (`wwr::wwrwmma` = `nvcuda::wmma` / rocWMMA), so it
  names no vendor. All 64 digit pairs by default; `--ozaki-pairs P`
  truncates, and P <= 5 gets cc-pVTZ's PC Sr/Si refused. Never the default:
  slower than the default digest on both cards (docs/performance.md, "The
  int8 Ozaki digest").

## The tree

| | |
|---|---|
| `src/` | one directory per component, each a module library and/or a device library. Every component is used by both demos except `df_integrals` (the density-fitted demo's alone): `common` (`idivup`/`align_up`; `int64_t` via `int64.h`, the one std using-declaration, so write it bare inside `nevpt2` and `std::int64_t` outside; `narrowTo<To>`, the one checked host narrowing; plus device-only `block_params.h`, `device_index.h` (`idx2`/`idx4`/`idx6`, `gridFor`), `warp_reduce.cuh`, `block_reduce.cuh` via the `nevpt2::common::device` target), `wwr`, `error_handling`, `device_resources` (`DeviceResources`, `DeviceBuffer<T>`/`DeviceBufferView<T>`), `profile`, `tensor`, `golden`, `einsum`, `cublas`, `ozaki` (device library only), `rdm`, `energy` (SC `energy.cpp` and PC `energy_pc.cpp`), `cli` (the flags both demos share), `df_integrals`. docs/architecture.md has the component graph. |
| `apps/` | `apps/<app>/{main.cppm,CMakeLists.txt}`: `integral_direct/` → `nevpt2_demo`, `density_fit/` → `nevpt2_df_demo` (checked against PySCF's **DF**-NEVPT2, the `*_df` goldens), `sanitizer_canary/` → `nevpt2_sanitizer_canary`. Each `main.cppm` is a module unit nothing imports (`export module nevpt2.app.<app>;`, plain `int main`). |
| `test/` | the GoogleTest unit tier, added from the root `CMakeLists.txt` only. Host-only suites carry no `gpu` label and run with no card; a suite that needs a card is `REQUIRES_GPU` → `gpu`; `EXPECT_DEATH` suites are also `death`. Each binary has a `<target>.SuiteListIsComplete` drift guard. |
| `cmake/` | the build's helper layer: `nevpt2_toolchain.cmake` (everything before `project()`), the module/device/gtest macros, `nevpt2_warpwraps.cmake`, the sanitizer tier (`nevpt2_sanitizers.cmake`, `add_sanitizer_canary`, `nevpt2_lsan.supp`) and host coverage (`nevpt2_coverage.cmake`). See `cmake/README.md`. |
| `docs/` | current-state, not chronological; `docs/README.md` is the index. No measured numbers in `architecture.md`, `implementation.md` or `references.md`. **Every measured number lives in exactly one of** `performance.md`, `pc-nevpt2.md`, `reference-data.md`, `testing.md`, at its latest measurement with its card and flags. Code comments and other docs cite them by file and section name (`docs/testing.md, "Sanitizers"`), so a figure added anywhere else is drift. |
| `deps/WarpWraps` | submodule ([WarpWraps](https://github.com/whatDiracIsCooking/WarpWraps)): `wwr.runtime_api` (host) and `runtime.h` (kernels). `EXCLUDE_FROM_ALL`: only what our targets link is built, and none of its tests register with our ctest. **Required**: `git submodule update --init --recursive`; `src/CMakeLists.txt` refuses to configure without it. |
| `golden/` | the committed references (docs/reference-data.md has the inventory). Generation is **bit-reproducible only because** `generate_golden.py` forces single-threading and a 1e-12 Davidson tolerance, and builds the molecule **with point-group symmetry** (auto-detected, linear groups mapped by `LINEAR_SUBGROUP`; never hard-code a group). Do not drop any of these to make generation faster: a regenerated file that differs from the committed one means something changed. `_check_degeneracy` and `_check_pc_conditioning` refuse states whose answer would depend on numerical noise. The PC fields are block2's answer (PySCF has no PC); the CAS(12,12) goldens carry none, and CAS(10,12) is the 12-orbital PC case. CAS(14,14) and the salicylaldimine pair are gitignored (too big) — regenerate them. |
| `reference_data/` | offline Python, never run by a build or test: `generate_golden.py` (the only file that defines the CASCI state), `block2_pc_probe.py` (block2's SC/PC on a golden's exact state; docs/pc-nevpt2.md, "The block2 reference"), `pyscf_contract_time.py` (the CPU side of the head-to-head). |
| `CMakeLists.txt` (root) | adds `src/`, registers the golden ctest entries, and under `NEVPT2_BUILD_TESTING` fetches GoogleTest and adds `test/`. `src/CMakeLists.txt` owns the backend choice and stays a self-contained project (`cmake -S src -B src/build`). |
| `devtools/`, `docker/`, `.devcontainer/` | configured for this project via `devtools/config.sh`. |

The golden tier is 100% `gpu`-labelled: every numerical check needs a card.
When you run something, say which of these it touched.

## Setup

```bash
git submodule update --init --recursive   # deps/WarpWraps -- REQUIRED by the build
uv sync                             # creates .venv from uv.lock (PySCF, offline tools only)
pre-commit install                  # commit-time ruff + whitespace
cmake --preset default && cmake --build --preset default -j   # CUDA
ctest --preset fast                 # every entry not labelled `slow`; needs a card
ctest --preset unit                 # the GoogleTest unit tier alone; host-only suites need no card
```

`uv` is the only assumed host tool, and `uv.lock` is the only place Python
versions live — after editing `pyproject.toml`, run `uv lock` and commit the
result; never `uv pip install`. **The C++ build fetches one thing, and only
with testing on**: GoogleTest by `FetchContent`, built from source because
everything links libc++. `compile-cuda`/`compile-hip` (testing OFF) and the
standalone `cmake -S src` fetch nothing. The build needs CMake ≥ 4.2 (the
`import std` gate UUID is pinned per release in
`cmake/nevpt2_toolchain.cmake`), clang + libc++ with `clang-scan-deps` and
`libc++.modules.json`, and a CUDA toolkit or ROCm.

Run `devtools/doctor.sh` first when anything behaves oddly.

## There is no CI, on purpose

**Do not add a `.github/workflows/`.** Every golden ctest entry needs a real
card; a GitHub-hosted runner has none, so CI could prove only that the tree
compiles and the host-only unit suites pass, while the claims this project
makes are numerical. A green badge that cannot fail on a wrong answer invites
the belief that something was checked (docs/testing.md, "No CI, and the local
gate").

**The gate is local and manual, and saying what you ran is the protocol:**

```bash
devtools/cpp-tier.sh                    # CUDA, on a box with a card (runs both lints first)
devtools/cross-backend-check.sh         # does the OTHER backend still compile (lints too)
devtools/stream-lint.sh --strict        # every GPU call on the one stream
devtools/throw-lint.sh                  # no `throw` in src/ or apps/
```

`src/` builds for both backends, so a CUDA-only spelling outside
`src/cublas/cublas_emul.cpp` (or an nvcc-only construct in a kernel) breaks
only the HIP build; `cross-backend-check.sh` catches that, but only proves it
*compiles* — a numerical claim needs the card. The `combined` devcontainer
reaches both (`devtools/devcontainer.sh --combined`, then `CMAKE_PRESET=hip
CTEST_PRESET=hip devtools/cpp-tier.sh` for the ROCm half).

`compile-cuda` / `compile-hip` are the compile-and-link-only configurations
(`NEVPT2_BUILD_TESTING=OFF`); `cross-backend-check.sh` drives `compile-hip`
through `CROSS_CHECK_PRESET`.

## Build and test

Two equivalent paths, which must not share a build directory (a CMake cache
records absolute paths):

```bash
# Top level: picks a backend, registers ctest entries.
cmake --preset default  && cmake --build --preset default -j && ctest --preset fast
cmake --preset hip      && cmake --build --preset hip -j      && ctest --preset hip

# Standalone -- no ctest, run the binary yourself.
cmake -S src -B src/build && cmake --build src/build -j      # -DNEVPT2_GPU_BACKEND=HIP for ROCm
./src/build/nevpt2_demo --golden golden/n2_ccpvdz_cas1010.nevpt2gold --tiles 3
```

**Tile counts are measured, not guessed.** `--tiles` is a memory-capacity
lever, not a speed lever (docs/performance.md, "Memory and `--tiles`"):

| case | emitted path | `--cublas` |
|---|---|---|
| CAS(10,10) | `--tiles 3`, the shipped count (docs/performance.md, "Why CAS(10,10) needs tiling") | `--tiles 3` engages |
| CAS(10,12) (`--pc`, the PC case) | `--tiles 40`, above the measured floor on both cards; below it the RDM build fails before any PC code runs | not measured |
| CAS(12,12) | `--tiles 40`, above the measured floor; below it the RDM build fails | needs **more** tiles: 40 has been seen both engaging and declining, so read the line; `80` or `150` engage and stay the recommendation |

The floors are in docs/performance.md, "Tile floors". An under-provisioned tile
count at CAS(10,10) **corrupts device memory** (`an illegal memory access`)
rather than failing cleanly. Do not lower a tile count to make something
faster.

**`--cublas` prints whether emulation actually engaged.** `engaged bits=53` is
the win; `engaged bits=-1 (DECLINED -> ran native fp64!)` means cuBLAS quietly
ran the slow path and the measurement is meaningless. Always read that line
before quoting a `--cublas` number.

**The sanitizer tier.** Five presets, each a Debug build in its own
`build-<preset>/`, whose test preset runs only the `small` CAS(4,4)/CAS(8,8)
entries, the `unit` tier (less its `death` suites under compute-sanitizer:
EXPECT_DEATH forks) and that sanitizer's canaries. memcheck's `--leak-check full`
(`NEVPT2_COMPUTE_SANITIZER_LEAK_CHECK`) is **ON** by default (both demos leak
0).

```bash
# CUDA card
cmake --preset asan  && cmake --build --preset asan  -j && ctest --preset asan    # host ASan+LSan
cmake --preset ubsan && cmake --build --preset ubsan -j && ctest --preset ubsan   # host UBSan
cmake --preset compute-sanitizer && cmake --build --preset compute-sanitizer -j
ctest --preset compute-sanitizer                                                   # memcheck
cmake --preset compute-sanitizer -DNEVPT2_COMPUTE_SANITIZER_TOOL=initcheck && ctest --preset compute-sanitizer
#   (racecheck / synccheck likewise; switching tool needs no rebuild)
# AMD card: host only -- no compute-sanitizer on ROCm, and gfx1200 cannot do device ASan
cmake --preset hip-asan  && cmake --build --preset hip-asan  -j && ctest --preset hip-asan
cmake --preset hip-ubsan && cmake --build --preset hip-ubsan -j && ctest --preset hip-ubsan
# devtools/cpp-tier.sh --preset <any of the five> does configure + build + ctest
```

A change to memory or stream handling must be clean under `asan`, `ubsan`,
`hip-asan`, and `compute-sanitizer` with memcheck and with initcheck, every
canary green. Know the limits (docs/testing.md, "Sanitizers"): **no tool sees
a cross-stream race directly** — that class of bug is
`devtools/stream-lint.sh`'s job. A canary that goes red means its sanitizer
is off or not failing runs. Fix that; never delete the canary. A leaked device
buffer is a failing test, so fix the leak rather than turning the check off.
New LSan suppressions go in `cmake/nevpt2_lsan.supp` and must name a vendor
library, never one of our frames.

**Host coverage.** `devtools/coverage.sh` (CUDA, preset `coverage`) or
`devtools/coverage.sh --preset hip-coverage` configures a Release build with
`NEVPT2_ENABLE_COVERAGE`, runs what `fast` runs **serially** (golden entries
in parallel on one card abort), and prints llvm-cov's table for `src/` +
`apps/` (`--html DIR`, `--lcov FILE`, `--report-only`). It instruments host
CXX units only: every `.cu` stays out on both backends
(`NEVPT2_DEVICE_LIBRARY`), and a process that aborts writes no profile, so
the abort tier reads as unexecuted. A covered line was run, not checked —
never quote coverage as a correctness claim. The measured numbers are in
docs/testing.md, "Coverage".

## Conventions

- **Host C++23 named modules; device C++20.** One directory per component
  under `src/`, each a STATIC module library declared with
  `add_cxx_module_library` (dotted name `nevpt2.<x>`, linked by its
  `nevpt2::<x>` alias) and/or a device library from `add_device_library`
  (pinned to C++20 on HIP too, and joined to the `nevpt2_device_libraries`
  umbrella that `cross-backend-check.sh --device-only` builds). Every module
  unit does `import std;` — qualify `std::size_t`/`std::uint64_t` (the one
  exception: `int64_t`, bare inside `namespace nevpt2` via `common/int64.h`),
  and `#include <cstdio>` in the global module fragment where `stderr` is
  needed. A macro cannot cross a module boundary, so error checks (`check`,
  `gpuCheck`, `narrowTo`) are functions taking a `std::source_location`; the
  one macro, `NEVPT2_TRY`, is a plain header
  (`"error_handling/error_macros.h"`) included in the global module fragment.
- **Two error tiers, and no exceptions** (docs/architecture.md, "Error
  handling"). **Our bug** — a broken invariant — is `check(cond, what)`
  (`nevpt2.error_handling`); a failed GPU runtime / BLAS / solver call is
  `gpuCheck(status)` (`nevpt2.wwr`); both print the caller's file:line and
  `std::abort()`. **Not our bug** — bad input, bad configuration, an
  unsupported request, a numerical refusal — is a returned `Result<T>` /
  `Status` holding an `Error` made by `err_io` / `err_config` /
  `err_unsupported` / `err_numerical`, propagated unchanged with `NEVPT2_TRY`
  and `report`ed **once**, at `main`, which returns 1. Pick by whose fault it
  is, not by how deep the call is. **Never `throw`** (and no `fprintf` +
  `std::exit` below `main`): `devtools/throw-lint.sh` fails on the keyword
  anywhere in `src/` or `apps/`. It is a lint, not `-fno-exceptions`, because
  the `import std` / WarpWraps BMIs must share our flags and std can still
  throw (`std::bad_alloc`), which, uncaught, terminates — the abort tier
  already.
- **Namespaces: everything is `nevpt2` (or a `nevpt2::<x>` sub-namespace
  with a real API — `profile`, `link_tables`, `device`, `canary`); there is
  no `detail`.** A module hides a helper by *not exporting it*: a plain
  `namespace nevpt2 { … }` outside the `export namespace` block, only when the
  interface's own inline/template code needs it (`narrowFailed` in
  `common.cppm`). Anything else an implementation unit hides goes in an
  **internal partition named for what it provides** (`nevpt2.golden:reader`,
  `nevpt2.einsum:planner`, `nevpt2.energy:sc_classes`; never `:internal` or
  `:detail`), one `<component>_<part>.cppm` each, listed under `PARTITIONS`
  and `import :<part>;`ed where used. **No anonymous namespace (and no
  namespace-scope `static`) outside a `.cu` file.** A demo's `main.cppm` puts
  its helpers in `export module nevpt2.app.<app>:<part>;` partitions that
  `export` nothing, pulled in with `export import :<part>;` (`apps/density_fit/block_check.cppm`). A GoogleTest
  `.cpp` puts its helpers in `namespace nevpt2::test::<suite>`. A `.cu` keeps
  its file-local helpers in **one** contiguous anonymous namespace. Headers a
  `.cu` includes use plain namespaces: reductions and functors in
  `nevpt2::device`, index and launch-shape helpers in `nevpt2`.
- **`#include "..."` for this repo's headers, `#include <...>` for
  everything else** — the standard library, the vendor SDKs, and WarpWraps
  (`<runtime.h>`, `<device_guard.h>`, …). In-repo headers are reached
  root-relative (`"common/int64.h"`) or, within one component, by bare name.
- **`src/rdm/kernels.cu` is never compiled on its own:** `rdm_launch.cu`
  `#include`s it and holds its launchers (a `__global__` can only be launched
  from its own TU), so a kernel there whose signature changes needs its
  launcher updated. The other kernels (`device_einsum`, `rdm_accumulate`,
  `f3_digest`, `f3_scatter`) each have their own file and device library.
- **Every comparison is against a tolerance (1e-7), never bit-identical.** The
  device kernels reassociate sums (`atomicAdd`), so exact equality with NumPy
  is not a property this code has or wants.
- **One non-blocking stream; nothing runs on the legacy stream, and the lint
  enforces it** (docs/performance.md, "One stream"). Each demo's `main()`
  creates exactly one `nevpt2::DeviceResources` (`DeviceResources::create`) —
  the stream, a memory pool, a BLAS handle and a dense-solver handle on device
  0 — and passes it (or its `stream()`) down. Every launch, copy, memset, BLAS
  call, event and sync is on that stream, and every device allocation is a
  `nevpt2::DeviceBuffer<T>` (built as `DeviceBuffer<T>(n,
  res.shared_from_this())`), which draws from the pool with
  `wwrMallocFromPoolAsync`, zero-fills, and frees itself with `wwrFreeAsync`
  when it goes out of scope. **There is no manual free**: hold the buffer as
  long as its last enqueued reader, and scope it when the free point matters
  for the peak. Never hand out a view that outlives its owner. **No stream
  parameter has a default**, and nothing spells a literal `nullptr`/`0`
  stream: the stream is `wwrStreamNonBlocking`, so a single stream-less call
  is a **silent race**, not a slowdown, and no sanitizer sees it.
  `devtools/stream-lint.sh --strict` is the gate; it reads text, not types, so
  it cannot see a null stream arriving through a variable or a default member
  initializer — don't write one. Because frees are stream-ordered, a buffer
  may be freed while kernels reading it are still queued, and a download
  (`downloadTensor`, which synchronizes itself) needs no sync before it. Every
  remaining `wwrStreamSynchronize` says why; do not add one "to be safe" or
  remove one without reasoning it out in a comment. This holds identically on
  HIP.
- Python ≥3.12, `from __future__ import annotations` everywhere.
- Lint is narrow (`E,F,I,UP,B`) with **no formatter hook**. `src/` is
  excluded (C++).
- **`.claude/hooks/protect-main.py` guards the primary checkout** and is on by
  default. Set `CLAUDE_ALLOW_MAIN_EDITS=1` for a session deliberately editing
  it. It cannot expand shell variables, so a `$VAR/path` in a Bash command
  resolves against the repo root and gets denied even when it points at
  `/tmp` — use literal absolute paths in scratch commands.

## Skills

| Skill | Reach for it to… |
|---|---|
| [devbox](skills/devbox/SKILL.md) | drive the containers and container-backed sibling worktrees — `devcontainer.sh`, `worktree.sh`. |
| [worktree](skills/worktree/SKILL.md) | create/clean the lightweight `.claude/worktrees/<name>` checkouts. |
| [pr](skills/pr/SKILL.md) | commit → push → PR → merge. |
| [doctor](skills/doctor/SKILL.md) | diagnose a degraded environment. |
| [audit](skills/audit/SKILL.md) | check that docs still match the tree. |
| [milestone](skills/milestone/SKILL.md) | turn a plan into a GitHub milestone + issue DAG. |
| [milestone-run](skills/milestone-run/SKILL.md) | execute a milestone's issue DAG with parallel subagents, re-testing and merging their PRs one at a time. |

## Containers

Four files chaining by **tag**, not by stage, so build them only with
`docker/build.sh <base|cuda|hip|combined>`:

```
                Dockerfile.base          clang-20, CMake 4.2, Ninja, sccache, uv
                 /           \
  Dockerfile.cuda             Dockerfile.hip
            |                       :
  Dockerfile.combined ..............:  (re-runs the ROCm install scripts, ~40GB)
```

`Dockerfile.base` does `COPY pyproject.toml uv.lock` — **both must exist at the
repo root or the base build fails**, and every other image is downstream of it.

Claude Code is pinned in the three GPU files (not a devcontainer feature), so
the version is declared three times; bump all three with
`devtools/claude-version.sh --apply`.
