# Architecture

How this tree is put together: the layering that lets **one `src/` build for
either GPU vendor**, where the host/device language boundary falls, and the
resource and integer disciplines every component obeys. It says nothing about
*what* the code computes — that is [`implementation.md`](implementation.md).

It also quotes **no measured numbers**. Every timing, sweep and negative
result lives in [`performance.md`](performance.md), [`pc-nevpt2.md`](pc-nevpt2.md),
[`reference-data.md`](reference-data.md) or [`testing.md`](testing.md), cited below by section name so
the citations read the same way the code's own comments do.

## 1. One tree, two backends

A build targets **exactly one** backend, chosen by `NEVPT2_GPU_BACKEND`
(`CUDA`, the default, or `HIP`). The same files serve both. Three layers make
that work:

```
  host module unit        import nevpt2.wwr;            <- what our code writes
         |                (wwr* bare in nevpt2; re-exports wwr.runtime_api
         |                 + nevpt2.error_handling)
         v
  WarpWraps               wwrMalloc, wwrStream_t, wwrMemcpyHostToDevice, ...
         |                bound by WWR_GPU_BACKEND
         v
  vendor runtime          ::cudaMalloc / ::hipMalloc
```

`src/CMakeLists.txt` forwards `NEVPT2_GPU_BACKEND` as WarpWraps'
`WWR_GPU_BACKEND`, and the `wwr*` names bind to `::cuda*` or `::hip*`. So
**host code is written once, against `wwr*` names only.** Prefer a `wwr*`
spelling to a vendor one, always.

Every one of those names already carries the `wwr` prefix, so inside
`namespace nevpt2` they are written **bare** — `wwrStream_t`, not
`wwr::wwrStream_t`. `src/wwr/wwr.h` `using`s them into the namespace for
kernel files (`wwrStream_t` always; `wwrError_t`, `wwrSuccess`,
`wwrGetLastError` in a device pass, the same gate as `runtime.h`'s), and the
`nevpt2.wwr` module (`src/wwr/wwr.cppm`)
republishes that header and adds the rest of the runtime, BLAS and solver
names host code uses. Only names the tree uses are listed: a new call that does
not compile bare gets a `using` there, not a `wwr::` in front of it. Code
outside the namespace (the `main()`s) still writes `wwr::wwrX`, and
`wwr::extension::*` — a namespace, not a prefixed name — stays qualified.

`nevpt2.wwr` is also the single door: `import nevpt2.wwr;`
alone names the runtime surface *and* the error checks, because it re-exports
`wwr.runtime_api` and `nevpt2.error_handling`. (A separate `nevpt2.gpu` did that
re-exporting until it was folded into `nevpt2.wwr`; `nevpt2.common` stays
GPU-free.) `gpuCheck` itself is defined in `nevpt2.wwr`, the module that knows
the `wwr*` status types; `nevpt2.error_handling` is vendor-free.
It also carries `kGpuBackendName`, switched on `WWR_SELECTED_CUDA` — the
same macro the runtime names bind by, so the banner can never disagree with
where the calls actually go.

### The two sanctioned vendor-naming sites

| site | why |
|---|---|
| `src/cublas/cublas_emul.cpp` | **CUDA-only by design.** CUDA 13.0's `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` (the `--cublas` digest) has no ROCm equivalent. Real under `NEVPT2_HAVE_CUBLAS_EMUL`, refusing stubs otherwise; a HIP binary rejects `--cublas` at flag parsing, through the stub `requireCublasEmul()`'s `Unsupported` error. |
| kernel files (`*.cu`) | They `#include <runtime.h>` (WarpWraps), which picks the vendor header off the compiler's own device macro (`__CUDACC__` / `__HIP__`). They cannot `import` — see §2. |

A raw `cu*`/`cuda*`/`hip*` call anywhere else compiles on one backend and
breaks the other, and **nothing automated catches it** — there is no CI here,
on purpose (§9). `devtools/cross-backend-check.sh` is the gate. It proves the
other backend *compiles*, not that it is right: a numerical claim needs a card.

## 2. Host C++23 modules, device C++20

| | host | device (`*.cu`) |
|---|---|---|
| standard | C++23 named modules, clang + libc++ | C++20 |
| std library | `import std;` | ordinary `#include` |
| GPU names | `import nevpt2.wwr;` (`wwr*`) | `#include <runtime.h>` |
| compiled by | clang | nvcc / clang (`-x hip`; on HIP the `.cu` files are CXX sources) |

Consequences that shape the code:

- **A macro cannot cross a module boundary.** So the checks are *functions*
  taking a `std::source_location` — `gpuCheck` (which replaced the old
  `GPU_CHECK` macro), `check` and `narrowTo` — and a failure names the call site, not
  the check's own file.
- **`import std;` does not carry macros either.** A unit needing `stderr` puts
  `#include <cstdio>` in its global module fragment. Qualify `std::size_t`,
  `std::uint64_t` and friends — except `int64_t`, bare inside
  `namespace nevpt2` (§5).
- **Kernel files cannot `import`.** Anything shared with the host stays a
  plain header, reached **root-relative** from `src/`, which is on the include
  path as every component's parent:

| header | shared between |
|---|---|
| `einsum/einsum_plan.h` | the host einsum planner and `device_einsum.cu` — a POD contraction descriptor |
| `common/align_up.h`, `common/int64.h` | `idivup`/`align_up` and `nevpt2::int64_t` (§5), so a kernel file can name them too; `nevpt2.common` re-exports both |
| `wwr/wwr.h` | the `wwr*` names `using`'d into `namespace nevpt2`, so a bridge or kernel file writes them bare; `nevpt2.wwr` re-exports it with the host-only rest |
| `common/block_params.h`, `device_index.h`, `device_functor.h`, `warp_reduce.cuh`, `block_reduce.cuh` | device-only, no module face; reached through the `nevpt2::common::device` INTERFACE target |

`EinsumPlan` crosses by value, so a `static_assert` holds it to
`is_trivially_copyable_v && is_standard_layout_v`: nvcc compiles the `.cu`'s
host half with its own host compiler, which need not share libc++'s ABI.

## 3. Kernels are linked in, never loaded

Every kernel file is a **STATIC device library** linked into the binary
(`add_device_library`, over WarpWraps' `wwr_add_gpu_device_library`)
and launched through the runtime API. **Nothing is loaded at run time**, and
the vendor Module API is not used anywhere in the tree — the CUDA binary
does not even link `libcuda`. A launch is an ordinary, type-checked function
call rather than a look-up-by-name with `void**` args.

The seam is one `*_bridge.h` per kernel file. Two details in it are
load-bearing (`src/rdm/rdm_accumulate_bridge.h` is the pattern the others
follow):

- **`extern "C++" { ... }`** around the include in the host unit. A name
  declared in a module's purview is otherwise attached to that module and gets
  module linkage, so it could never bind to the definition in a plain TU —
  which is exactly what a `.cu` is. A linkage-specification attaches it to the
  *global* module instead, giving the same mangled name.
- **Included after the imports**, not in the global module fragment. The
  launchers return `wwrError_t`, which a *host* include of `wwr/wwr.h`
  does not declare (it, like `runtime.h`, gives a host compile `wwrStream_t`
  only); after `import nevpt2.wwr;` it is. The
  includer puts `runtime.h` itself in its GMF, so the vendor headers stay out
  of the purview.

Only plain types cross: pointers, integers, the vendor stream handle, the
error enum by its `wwr*` name. Every launcher returns the **launch** status
straight into `gpuCheck`, with no intermediate check; a
kernel's own execution errors surface at the next synchronization, as with any
launch.

**`src/rdm/kernels.cu` is the one exception to one-file-one-library:** it is
never compiled on its own. `rdm_launch.cu` `#include`s it and holds its
launchers, because a `__global__` can only be launched from its own
translation unit. Change a kernel signature there and its launcher in
`rdm_launch.cu` needs the same change.

## 4. Component graph

One directory per component under `src/`, each a STATIC module library
(`add_cxx_module_library`, dotted name `nevpt2.<x>`, linked by its
`nevpt2::<x>` alias) and/or a device library. Every directory is something
**both** methods run, except `df_integrals`, which only the density-fitted
demo links, and `gradient`, whose modules no *energy* run needs: the
integral-direct demo links `nevpt2::gradient` for `--gradient-source` alone
(and `nevpt2::rdm_host_dm` for that and `--sc-tangent`), the density-fitted
demo links neither, `nevpt2::pc_tangent` is `nevpt2_pc_tangent`'s alone, and
`nevpt2::kappa_rotation` is `nevpt2_sc_pseudodensity`'s (for `--kappa-act`).
In dependency order:

```
  error_handling ... nevpt2.error_handling  Error / Result / Status, check,
                                            error_macros.h (NEVPT2_TRY)  (std only)
  common ........... nevpt2.common          idivup / align_up, int64_t,
                                            checked narrowTo<To>         (no GPU)
                     nevpt2.common.device   INTERFACE: device-only headers
  wwr .............. nevpt2.wwr             wwr* names, bare in nevpt2, gpuCheck,
                                            kGpuBackendName; re-exports
                                            wwr.runtime_api, error_handling,
                                            wwr.extension.memory_buffer;
                                            WarpWraps' RAII wrappers with Abort
  device_resources . nevpt2.device_resources  DeviceResources, DeviceBuffer<T>
  profile .......... nevpt2.profile         --profile event pairs
  tensor ........... nevpt2.tensor          host N-d Tensor               (no GPU)
                     nevpt2.device_tensor   DeviceTensor + upload/download
  golden ........... nevpt2.golden          .nevpt2gold + .gradgold + .srcgold
                                            readers
  einsum ........... nevpt2.einsum          planner/launcher + device_einsum.cu
                     nevpt2.einsum.plan     INTERFACE: einsum_plan.h
  cublas ........... nevpt2.cublas_emul     --cublas digest (CUDA real, HIP stubs)
  ozaki ............ nevpt2.ozaki.device    --ozaki digest: int8 tensor cores via
                                            WarpWraps' <wmma.h>  (device lib only)
  rdm .............. nevpt2.link_tables     Knowles-Handy tables          (no GPU)
                     nevpt2.rdm_host_dm     host dm1/dm2 from a CI vector,
                                            and their tangents           (no GPU)
                     nevpt2.rdm_plan        tile plan, consume chunking   (no GPU)
                     nevpt2.rdm_build       the tiled dm3/f3ac/f3ca build
                     nevpt2.sigma_vector    3-body / 4-body-contracted sigma
                                            vectors + sigma_vector.cu
                     + device libs: rdm_launch.cu (#includes kernels.cu),
                                    rdm_accumulate.cu, f3_digest.cu, f3_scatter.cu,
                                    f3_transpose.cu, sigma_vector.cu;
                     links nevpt2.ozaki.device
  energy ........... nevpt2.energy_finish   slab walk, SC/PC finish arithmetic
                                            and the SC finish's derivative,
                                            PC result types               (no GPU)
                     nevpt2.energy          the eight class energies, SC (energy.cpp)
                                            and PC (energy_pc.cpp), plus the SC
                                            tangent in the RDMs (:sc_tangent) and
                                            its second derivative (:sc_hessian);
                                            re-exports nevpt2.energy_finish
  df_integrals ..... nevpt2.df_integrals    DF IntegralSource: slabs from B
                                            by wwrblasDgemm      (energy, wwr.blas)
  gradient ......... nevpt2.sc_amplitudes   the SC amplitude T, the gradient
                                            multipliers P/Q/R, the
                                            pseudocanonical multipliers z^c,
                                            and the einsum subscript rewriting
                                            that differentiates a class term
                            (energy_finish, error_handling; no GPU)
                     nevpt2.kappa_rotation  the orbital-rotation
                                            parameterisation: an antisymmetric
                                            kappa, exp(kappa), and an array's
                                            active axes transformed
                                                              (tensor; no GPU)
                     nevpt2.gradient_plan   the device assembly's pure host
                                            arithmetic: the promotion table,
                                            the classes' Delta vectors and
                                            slab multipliers, termHasActive,
                                            the CI source term's host fold
                     (tensor, sc_amplitudes, error_handling; no GPU)
                     nevpt2.gradient        the SC pseudodensities, the
                                            orbital source term Y_xy over the
                                            WHOLE kappa space, and the CI
                                            source term y_I = dE/dc_I (Park
                                            Eq. 48), assembled over the class
                                            energies' own slab walk, and the
                                            f3ac/f3ca digests' orbital
                                            response in forward mode (the RDM
                                            build rerun at the dotted h2e),
                                            and H_cc u, the walk behind y_I
                                            differentiated along a CI
                                            direction, SC's (#156) and PC's
                                            (#155);
                                            re-exports the amplitudes and the
                                            plan. A LEAF: nothing in src/
                                            imports it
                                   (energy, gradient_plan, sigma_vector, rdm_build)
                     nevpt2.pc_tangent      the PC CI-tangent check: the
                                            analytic tangent against a finite
                                            difference of the displaced
                                            states, and the checks of PC's
                                            Eddot and H_cc u. A LEAF; with
                                            nevpt2.gradient, one of the two
                                            modules that link the RDM build
                                            and the energy stage both
                     (energy, rdm_build, rdm_host_dm; gradient privately)
  cli .............. nevpt2.cli             every golden-reading binary's flags,
                                            and the binary/flag table
                                            (:binary_flags)              (no GPU)
```

Then the binaries, in `apps/` — a sibling of `src/`, but added from
`src/CMakeLists.txt` so the standalone `cmake -S src` path still builds them.
Each is `apps/<app>/{main.cppm,CMakeLists.txt}`; `main.cppm` is a module unit
nothing imports (`export module nevpt2.app.<app>;`), and its plain `int main`
is attached to the global module by clang:

| target | from |
|---|---|
| `nevpt2_demo` | `apps/integral_direct` — full MO-integral blocks, read from the golden file |
| `nevpt2_df_demo` | `apps/density_fit` + `src/df_integrals` (`nevpt2.df_integrals`) — external slabs built from three-index `B_*` tensors |
| `nevpt2_sc_pseudodensity` | `apps/sc_pseudodensity` + `src/gradient` (`nevpt2.gradient`) — the SC gradient's pseudodensities, and the exact identities that check them; with `--kappa-act` (`kappa_check.cppm`), the active-active block of the orbital source term instead, adjoint against tangent against a finite difference; with `--kappa-full` (`kappa_full_check.cppm`), the whole source term over the full `kappa` space, reading the gradient sidecar beside the golden; with `--dump-source-terms PATH` (`source_dump.cppm`), that whole `Y_xy` and the CI source term `y_I` written to a `.gradsrc` for `reference_data/nevpt2_grad.py`, through `nevpt2.gradsrc`, one of the two module libraries under `apps/`; with `--source-oracle PATH` (`source_oracle.cppm`), the same two terms compared block by block with PySCF's exact ones from a committed `golden/*.srcgold`, every block held to agree (the known-gap list is empty since #89), through the other, `nevpt2.source_compare`, which also holds the PC oracle's classification (#95); with `--pc`, the dump and the oracle are PC-NEVPT2's (#74), against `nevpt2.gradient`'s PC reverse mode and block2's `golden/n2_ccpvdz_cas44_pc.srcgold` |
| `nevpt2_pc_tangent` | `apps/pc_tangent` + `src/gradient` (`nevpt2.pc_tangent`) — PC's derivative with respect to the CI vector, against the exact Euler identity and a Richardson-verified finite difference. The only binary that links `nevpt2::pc_tangent` |
| `nevpt2_sanitizer_canary` | `apps/sanitizer_canary` — deliberately buggy runs, the sanitizer tier's canaries |

`rdm/` **is** shared between the two demos, and that is not a coincidence: the
`f3ac`/`f3ca` digests contract the 4-RDM with the **active** `h2e` only, which
the density-fitted demo rebuilds exactly from `B_aa` and hands to the same
build. What density fitting replaces is the *external* blocks the class
energies read — hence the `IntegralSource` seam (§7).

The per-file inventory below was README.md's "Files" table until README.md
was rewritten for public readers (2026-10-09); the golden files and the offline
Python tools are in [`reference-data.md`](reference-data.md), "The files in
`golden/` and `reference_data/`".

| Path | Role |
|---|---|
| `cmake/` | the build's helper layer: `nevpt2_toolchain.cmake` (everything before `project()` — clang + libc++, the `import std` gate, the backend and device arch), `add_cxx_module_library`, `add_device_library` (a linked-in kernel library, C++20 on both backends, joined to the `nevpt2_device_libraries` umbrella), `nevpt2_warpwraps.cmake` (embeds `deps/WarpWraps`), and the sanitizer tier: `nevpt2_sanitizers.cmake` (ASan/UBSan/compute-sanitizer options, all OFF by default), `add_sanitizer_canary` + its verdict script, and `nevpt2_lsan.supp`; plus the gtest helpers `add_gtest_executable` / `add_gtest_suite_tests` and the drift guard's `nevpt2_check_gtest_suites.cmake`. See `cmake/README.md` and "Sanitizers" in [`testing.md`](testing.md) |
| `test/` | the GoogleTest unit tier, added from the root `CMakeLists.txt` only. `test/error_handling/` (`error_handling_tests`): `ErrorKind`/`kindName`, `err_*` → `Result`/`Status`, `NEVPT2_TRY` propagating an `Error` unchanged, `report`'s text, `nevpt2.common`'s `idivup`/`align_up`/`narrowTo`, and the abort tier as `EXPECT_DEATH` (`death` label). Host-only: no `gpu` label, runs with no card. One ctest entry per suite plus `error_handling_tests.SuiteListIsComplete`. `test/device_tensor/` (`device_tensor_tests`): `DeviceTensor`/`DeviceBuffer` upload-download round trips and the zero fill, `REQUIRES_GPU` (`gpu`), on the one `DeviceResources` per process that `test/utils/shared_resources/` (`nevpt2.test.shared_resources`) creates lazily and releases in a gtest environment's `TearDown()` |
| `apps/sanitizer_canary/` | `nevpt2_sanitizer_canary`: one deliberately buggy run per mode (host heap overflow, signed overflow, device out-of-bounds, free-before-read (plain and through the demos' pool), device leak, uninitialized read, shared-memory race, divergent barrier), the sanitizer tier's canaries. Built in every configuration, registered with ctest only under the sanitizer that must catch it |
| `src/CMakeLists.txt` | the build, for either backend: one subdirectory per component, each a STATIC C++23 module library and/or a linked-in device library (`add_device_library`), then the host binaries. Every directory under `src/` is a component both methods run (common, error_handling, wwr, device_resources, profile, tensor, golden, einsum, cublas, ozaki, rdm, energy, cli), except `df_integrals`, which only the density-fitted demo uses, and `gradient`, which no energy run needs (§4); each host binary is `apps/<app>/{main.cppm,CMakeLists.txt}` — `apps/integral_direct/` (SC-NEVPT2 from the full MO-integral blocks, `nevpt2_demo`), `apps/density_fit/` (SC-NEVPT2 from three-index tensors, `nevpt2_df_demo`), the two gradient checks `apps/sc_pseudodensity/` and `apps/pc_tangent/`, and `apps/sanitizer_canary/`. Every binary links the runtime, BLAS and dense-solver libraries: on CUDA `libcudart` (the `wwr*` names), `libcublas` and `libcusolver` (no `libcuda`); on HIP `libamdhip64`, `libhipblas` and `libhipsolver` (over rocBLAS / rocSOLVER). BLAS is for the consume, digest and DF GEMMs; the solver is for `--pc`'s eigensolves |
| `src/common/` | `nevpt2.common`: `idivup`/`align_up` (`align_up.h`, host and device); `int64_t` (`int64.h`, host and device: `namespace nevpt2 { using std::int64_t; }`, the one host size/extent type and the tree's one using-declaration of a std name); and `narrowTo<To>(v[, hint])`, the one checked host narrowing — it aborts with the caller's file:line when `v` does not fit (e.g. `narrowTo<int>(nK, "use more --tiles")` at the `--cublas` digest). The consume and `--blas-digest` GEMMs call WarpWraps' `wwrblasDgemm_64` / `wwrblasDgemmStridedBatched_64` and narrow nothing. Plus device-only headers with no module face, reached through the `nevpt2::common::device` INTERFACE target: `block_params.h` (`kWarpSize`, `num_warps`), `device_index.h` (`idx2`/`idx4`/`idx6` flattening in `int64_t` and the one `gridFor` every hand-written launch sizes its grid with), `warp_reduce.cuh` and `block_reduce.cuh` (the reductions `energy_sijrs` folds with). |
| `src/error_handling/` | `nevpt2.error_handling`: vendor-free (imports only `std`, links nothing GPU-side). The value tier — `Error` (an `ErrorKind`, a message and the `std::source_location` where it was created), `Result<T>` / `Status` over `std::expected`, the `err_*` factories, `report` — with `error_macros.h`'s `NEVPT2_TRY` (unwrap, or propagate the `Error` unchanged) — for input, configuration and numerical failures (`loadGolden`, `DeviceResources::create`, `requireCublasEmul`, `DfIntegralSource::create`, `pcEnergiesDevice`, ...), reported once at `main`. The abort tier, `check(cond, what)`, for broken invariants; `narrowTo` fails through it. `gpuCheck` lives in `nevpt2.wwr`. Nothing throws (`devtools/throw-lint.sh`); docs/architecture.md, "Error handling" |
| `src/wwr/` | `nevpt2.wwr`: the one door host code reaches the GPU through — WarpWraps' `wwr*` runtime/BLAS/solver names `using`'d into `nevpt2` so they are written bare (`wwr.h` is the kernel-file half), `wwr.runtime_api` and `nevpt2.error_handling` re-exported, plus `gpuCheck` (one template over every status type WarpWraps' error layer knows, aborting through `kit::AbortPolicy`; every `*_bridge.h` launcher's `wwrError_t` goes through it; here rather than in the vendor-free `nevpt2.error_handling`) and `kGpuBackendName`, and WarpWraps' RAII wrappers with `nevpt2::Abort` in every slot (the four `DeviceResources` holds, and the `DeviceBufferSuite<Handles>` template). The one host file that names a vendor API is `src/cublas/cublas_emul.cpp`, CUDA-only by design |
| `src/device_resources/` | `nevpt2.device_resources`: `DeviceResources`, the one non-blocking stream, memory pool, BLAS handle and dense-solver handle each demo creates once in `main()` and passes down — WarpWraps' `StreamWrapper` / `MemPoolWrapper` / `BlasHandleWrapper` / `SolverDnHandleWrapper` with `nevpt2::Abort` in every policy slot, as `nevpt2.wwr` instantiates them. Also `DeviceBuffer<T>`: WarpWraps' buffer suite (`nevpt2.wwr`'s `DeviceBufferSuite<const DeviceResources>`, i.e. `kit::device_buffer_suite<single_policy_map<Abort>, const DeviceResources>`), the one spelling of a device allocation — `wwrMallocFromPoolAsync` + zero-fill on the stream at construction, `wwrFreeAsync` on it at destruction, co-owning the `DeviceResources`; `highWater()` reads the pool's `UsedMemHigh`/`ReservedMemHigh`. See "The memory pool" and "One stream" in docs/performance.md |
| `src/rdm/kernels.cu` | the 8 CAS-independent RDM-build kernels, on WarpWraps' `runtime.h`. Not compiled on its own: `rdm_launch.cu` `#include`s it. |
| `src/rdm/rdm_launch.cu` | the hand-written launchers for `kernels.cu`, in the same translation unit (it `#include`s `kernels.cu`); the `nevpt2.rdm.kernels.device` library, called from `rdm_build.cpp` through `rdm_kernels_bridge.h` |
| `src/rdm/f3_digest.cu` | the emitted path's f3 digest GEMM: `kernels.cu`'s 16×16 tile, reading `L2` directly instead of `R2` and adding into the f3 accumulator (`+=`) instead of a per-tile output. A linked-in device library on both backends (`f3_digest_bridge.h`). See "What did not help" in docs/performance.md |
| `src/rdm/rdm_accumulate.cu` | the one kernel the device-resident RDM-build launcher needs — an on-device tile-partial accumulate, replacing a download+host-add+discard round trip (see "Device-resident RDM build" in docs/performance.md) |
| `src/rdm/link_tables.*` | `nevpt2.link_tables`: Knowles–Handy link tables, pure host C++, in PySCF's `fci.cistring` convention |
| `src/rdm/rdm_host_dm.*` | `nevpt2.rdm_host_dm`: the active-space `dm1`/`dm2` rebuilt **on the host** from a CI vector over `nevpt2.link_tables` — `dm1[p,q]` the expectation of `E_pq` and `dm2[p,q,r,s]` that of `E_pq E_rs`, the golden file's (PySCF `make_dm123`) convention — plus `dm12Dot`, the same bilinear forms with one copy of `c` replaced by a direction. No GPU and no kernel; `nevpt2.ci_checks` links it for both demos' `--sc-tangent` and `--gradient-source`. It exists because `dm1`/`dm2` are golden *inputs*, so nothing else recomputes them from the `ci` beside them; `test/rdm_host_dm/` rebuilds each committed pair card-free |
| `src/rdm/rdm_plan.cppm` | `nevpt2.rdm_plan`: the RDM build's pure host index arithmetic — the determinant-axis tile plan (`tileWidthForCount`, `planTiles`), the consume GEMM's chunking (`evenChunks`) and its permuted integral copy (`permuteEriConsume`). No GPU; `nevpt2.rdm_build` imports it, and `test/rdm_plan/` tests it card-free |
| `src/rdm/rdm_build.*` | `nevpt2.rdm_build`: the tiled, device-resident dm3/f3ac/f3ca build (link tables → produce → digests → fdm2/wedge, `--tiles`, `--cublas`), lifted out of `nevpt2_demo`'s `main` unchanged so `nevpt2_df_demo` runs the same build. Shared because it contracts only the **active** `h2e`, which density fitting rebuilds exactly. `src/rdm/rdm_build_tangent.cppm` is its `:tangent` partition: `buildRdmTangentsDevice`, the same pipeline run with one copy of the CI vector replaced by a direction — every output is bilinear in `ci`, so the tangent is the symmetrised cross term and needs no kernel of its own, only a second `produce` per tile and each digest twice — plus `checkRdmTangentsDevice`, the `--rdm-tangent` self-check both demos run (the tangent along the state itself is twice the build), and `buildRdmHessianTangentsDevice` (#153), the same sweep over THREE vectors `(c, u, v)` giving the cross terms at `(c, u)`, `(c, v)` and `(u, v)` — `produce` once per vector, not once per pair — which the energy's second derivative reads. Both run one cross-term sweep, `buildRdmCrossTermsDevice`. Their tile floors are their own, since they hold two or three copies of `produce`'s output per tile; `test/rdm/rdm_tangent_tests.cpp` checks the tangent against a finite difference of the plain build, and each of the three-argument sweep's sets against the tangent at its pair. `src/rdm/rdm_build_adjoint.cppm` is the `:adjoint` partition (#89): `buildDigestAdjointDevice`, the f3 path's TRANSPOSE with respect to `h2e` — given `dE/d(f3ac)` and `dE/d(f3ca)`, `dE/d(h2e)` through the digests — the forward steps run backwards over `f3_transpose.cu`'s kernels and one native-fp64 DGEMM per order, re-producing `L2` per tile with the plain build's own `produce`, so the 4-RDM is never formed; the same suite checks it against the plain build run at a direction, and element by element at CAS(4,4) |
| `src/rdm/sigma_vector.*` | `nevpt2.sigma_vector` and its device library `src/rdm/sigma_vector.cu` (`sigma_vector_bridge.h`), built on **both** backends: a coefficient tensor over active indices contracted with the excited wavefunctions it multiplies, one amplitude per determinant — `sum_{pqrstu} d3 (E_pq E_rs E_tu |ci>)[K]`, and the 4-body form with the `f3` ERI contraction folded in. The point is that the `n^6`/`n^8` RDM *derivative* never exists: contracted on the fly it is a vector of length `ndet`, the same trick `f3ac`/`f3ca` play for the energy. One thread owns one output determinant and walks the **reverse** link tables backwards, so there is no atomic and no shuffle; the 4-body one factors through an `(n^2, ndet)` intermediate rather than paying a fourth nested walk, and `sigmaVector4DaggerDevice` is the adjoint half of the same object (the two kernel passes with their coefficient blocks exchanged, since the dagger puts the ERI on the outermost operators) — forward plus dagger is what a CI derivative of the `f3` contraction is. `nevpt2_demo` *calls* it through `nevpt2.gradient`'s `--gradient-source`, and `nevpt2_sc_pseudodensity` links it transitively through the same component without using it; `test/sigma_vector/` checks every sigma against an explicit host construction, against the build's own `dm3`/`f3ca`/`f3ac`, and against an exact central difference of the host contraction |
| `src/rdm/f3_transpose.cu` | the f3 digests' transpose with respect to the integrals (#89): the adjoints of the wedge (onto the pre-wedge digest, in the digest GEMM's operand layout, and onto `fdm2`), of `fdm2` and of `consume` (each onto the physicist-ordered `h2e` adjoint). One thread per output element, no atomic and no shuffle, launched through `parallel_for` — a linked-in device library (`f3_transpose_bridge.h`) on **both** backends, driven by `nevpt2.rdm_build`'s `:adjoint` partition |
| `src/rdm/f3_scatter.cu` | the f3 permute-scatter the GEMM digests (`--cublas`, `--blas-digest` — HIP's default — and `--ozaki`) fold the winning `M=n^4,N=n^2` f3 GEMM into — a linked-in device library (`f3_scatter_bridge.h`) built on **both** backends; its ac order is `rdm_accumulate`'s `accumulateInplace`. Also `fusedDigestSplit`, the one-kernel fold of `--fused-digest`'s `(n^4, 3·n^2)` product into all three accumulators. It lived in `src/cublas/` until 2026-10-08, from when `--cublas` was its only caller |
| `src/cublas/` | **CUDA only.** `nevpt2.cublas_emul`, the `--cublas` digest: a `cublasGemmEx` (**not** `cublasDgemm`) wrapper requesting `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` (EAGER/FIXED/`max-mantissa-bits`), plus the engaged-bits probe (see "The cuBLAS fixed-point-emulation digest" in docs/performance.md); compiled to refusing stubs on HIP, where only its `requireCublasEmul()` is called (an `Unsupported` error, to reject `--cublas`) |
| `src/ozaki/` | **Both backends.** `nevpt2.ozaki.device`, the `--ozaki` digest: the three digest GEMMs on int8 tensor cores by the Ozaki scheme, through WarpWraps' `<wmma.h>` (`wwr::wwrwmma` — `nvcuda::wmma` / rocWMMA), so it names no vendor. A device library only, called from `rdm_build.cpp` through `ozaki_digest_bridge.h`. Accurate to fp64 rounding with all 64 digit pairs, and slower than the default digest on both cards (see "The int8 Ozaki digest" in docs/performance.md) |
| `src/golden/` | `nevpt2.golden`: reads the golden binary file (`loadGolden`), and the analytic-gradient **sidecar** beside it (`loadGradGold`, `gradgold.cpp`) — a separate `*.gradgold` file so the committed goldens' bytes never change, carrying the general-range MO integrals a gradient reads (docs/gradient-theory.md, §3; docs/reference-data.md, "The gradient sidecar") — or, in its density-fitted flavour beside a `*_df` golden, the three-index blocks that golden lacks (§3.8), and the committed source-term oracles (`loadSrcGold`, `srcgold.cpp`: the `*.srcgold` files, all four kinds, full and directional, SC and PC; docs/reference-data.md, "The SC source-term oracle"). All three formats end with the same named-array section, so all three parse it with the one `nevpt2.golden:reader` partition |
| `src/tensor/tensor.*` | `nevpt2.tensor`: a small dense host N-d tensor — the container the golden file loads into and that `DeviceTensor` wraps — plus a `transpose` (still used by `rdm_build.cpp`, for the chemists'-order `h2e` both demos' RDM build reads). The generic host `einsum` it was originally built around went with `energy.cpp`; the device kernel is the demo's only einsum now |
| `src/tensor/device_tensor.*` | `nevpt2.device_tensor`: a device-resident tensor (shape/stride bookkeeping + an owning `DeviceBuffer<double>` or a non-owning `DeviceBufferView<double>`) — the GPU twin of `Tensor`. Move-only; `view()` and `sliceAxis` make views. Allocating calls take the `DeviceResources`, copies the stream |
| `src/einsum/einsum_plan.h` | the POD contraction descriptor shared between the host planner and the device kernel — a header, not a module, because the device side is not a module-aware compile |
| `src/einsum/device_einsum.cu` | the generic device contraction kernel + the three hand-written "diagonal slice" kernels `make_a16`/`make_a22` need + `energy_sijrs`, the Sijrs (MP2-like) slab reduction that replaced the host loop over `cvcv` — this demo's own kernels, not copied from anywhere. A linked-in device library on both backends, launched through `device_einsum_bridge.h` |
| `src/einsum/einsum.*` | `nevpt2.einsum`: parses an einsum subscript string into an `EinsumPlan` and launches `device_einsum.cu`'s kernel — the GPU twin of the former host `einsum` |
| `src/profile/profile.*` | `nevpt2.profile`: the one `--profile` recorder — labelled event pairs on the caller's stream, `profile::time(label, stream, fn)`, grouped by `profile::Section` into one table per stage (RDM build, DF integrals, energy). Replaced einsum's registry and the RDM build's `PhaseTimer` (`--profile-rdm`) |
| `src/energy/` | `nevpt2.energy`: the eight class energies, GPU version — both demos' only energy path, a direct transcription of a NumPy reference (see "Porting the class energies" in docs/implementation.md); also the home of `CLASSES`/`EnergyResult` since `energy.hpp` was removed (`NUMERICAL_ZERO` is `nevpt2.energy_finish`'s, re-exported). `src/energy/energy_finish.cppm` is `nevpt2.energy_finish`, the pure host arithmetic the classes share — the slab walk `forBatches`, SC's `normToEnergy` and its derivative (`normToEnergyDerivative`/`normToEnergyTangent`, Park Eqs. 36-37) and second derivative (`normToEnergyHessian`, the 3×3 in the `d` form, and the multiplier tangents `applyFinishHessian`/`normToEnergyMultiplierTangent`; docs/hessian-theory.md, §2), PC's rank cut and gap decision `pcRankCut` (with `PC_TAU`/`PC_MIN_GAP`), `finishSingle`/`checkDenominator` and the `PcClassResult` types they fill — no GPU, tested card-free by `test/energy_finish/`. `src/energy/energy_sc_tangent.cppm` is the `:sc_tangent` internal partition: `energiesTangentDevice`, the SC class energies' forward derivative with respect to the RDMs. The class contractions are linear in the RDMs, so it reruns `:sc_classes`' own per-slab chains with the tangent RDMs substituted (plus three tangent-only builders, for the metrics that carry an RDM-free term) and differentiates nothing but the scalar finish; `nevpt2_demo --sc-tangent` is the check (docs/testing.md, "The SC tangent's finite-difference cross-check"). Its per-class block builders (`srBlocks`, `siBlocks`, …, at an RDM set or at a tangent one) are shared with `src/energy/energy_sc_hessian.cppm`, the `:sc_hessian` partition (#153): `energiesHessianDevice`, the second directional derivative `Eddot(u, v)` at fixed orbital energies, the same walks carrying four RDM sets (plain, `u`'s and `v`'s tangents, the second-order one) and finishing with `normToEnergyHessianSandwich` (the 3×3 between the two first-order tangents, in `nevpt2.energy_finish`); `nevpt2_demo --sc-hessian` is its check (docs/testing.md, "The SC Hessian's finite-difference cross-check"). Each class takes its external two-electron block from an `IntegralSource`, one slab of its external index at a time: `FullBlockSource` (views into uploaded full blocks, one slab — the integral-direct path) or `DfIntegralSource` |
| `src/energy/energy_pc*` | the PC-NEVPT2 half of `nevpt2.energy` (`pcEnergiesDevice`, `--pc` in both demos; its own module, `nevpt2.energy_pc`, until it was folded in): `energy_pc.cpp` (`pcEnergiesDevice` and the report) over two internal partitions, `:pc_solve` (`energy_pc_solve.cppm`, the per-class GEMM and eigensolve) and `:pc_classes` (`energy_pc_classes.cppm`, seven of the classes; SC's seven are `:sc_classes` the same way). Same inputs as the SC classes and built from their intermediates — one `wwrsolverDnDsyevd` pair per class on `DeviceResources`' solver handle, every slab GEMM'd into the eigenbasis, S's gap printed and checked against `PC_MIN_GAP`. All eight classes are computed (Sijrs = SC's), and the `PASS: PC-NEVPT2 ...` line needs all eight and `e_pc_total` to 1e-7 with none refused; the `*_pc` ctest entries check it. See "PC-NEVPT2 on the device: the design" in [`pc-nevpt2.md`](pc-nevpt2.md). A third internal partition, `:pc_tangent` (`energy_pc_tangent.cppm`), is `pcTangentsDevice`: the same eight classes differentiated with respect to the CI vector — the *solve* differentiated, never the eigendecomposition, which is legitimate exactly where `PC_MIN_GAP` holds and refused where it does not (`energy.cppm`'s module header carries the argument). A fourth, `:pc_hessian` (`energy_pc_hessian.cppm`, #152), is `pcHessianDevice`: the solve differentiated twice, with [`hessian-theory.md`](hessian-theory.md), §6.2's rotation terms from the solve's own dropped eigenvectors, refusing a class whose dropped modes lift along the direction pair |
| `src/gradient/` | the SC-NEVPT2 gradient's first half, a **leaf**: nothing in `src/` imports it (both demos link it through `nevpt2::ci_checks` for `--sc-tangent` / `--gradient-source`, `nevpt2_sc_pseudodensity` for the identities, `--kappa-act`, `--kappa-full` and `--dump-source-terms`). `sc_amplitudes.cppm` is `nevpt2.sc_amplitudes` — the amplitude `T` already implicit in `normToEnergy`'s division, the multipliers `P`/`Q`/`R` that are its derivatives with respect to `N`, `H` and `Delta`, and `derivativeSubscripts`, which rewrites one of `nevpt2.energy`'s class einsums into the derivative of that term with respect to one operand. No GPU, so `test/gradient/` tests it card-free. `kappa_rotation.cppm` is `nevpt2.kappa_rotation` — a seeded antisymmetric `kappa`, `U = exp(kappa)` by scaling and squaring, and `rotateAxes`, the transformation `U` induces on the active axes of one stored array. Also no GPU (the device twin of the first-order action is `nevpt2.gradient`'s own), so `test/gradient/` checks card-free the two properties the construction rests on: `U` orthogonal to the tolerance that suite asserts, and a rotation by `kappa = 0` leaving a block bit-identical (the figure is in [`testing.md`](testing.md), "The active-rotation source term"). `gradient_plan.cppm` is `nevpt2.gradient_plan` — the device assembly's pure host arithmetic, pulled out of the internal partitions below (which no test can import) the way `nevpt2.rdm_plan` was pulled out of the RDM build: the promotion table (`rotBlockInfo`, `activeAxesOf`, `rotBlockName`, `spanOf`, `generalPickPlan` — which one-general-index array a promotion reads, and how its second distribution is normalised — and `promotionPlan` / `promotionSubscripts`, the rule's per-slot einsum), the eight classes' `Delta` vectors and the two symmetrized classes' slab multipliers (`sijrMultipliers`, `srsiMultipliers`, `plainMultipliers`), `termHasActive`, and the CI source term's host fold (`ciSourceFolded`, `ciSourceSymmetrized`, `ciSourceAddInto`) — plus `longestFourfoldRun`, the step-ladder rule the SC tangent, the two orbital source-term checks and the PC tangent all gate their finite differences on (so `nevpt2.pc_tangent` links it too). No GPU; `nevpt2.gradient` re-exports it and `test/gradient_plan/` pins it card-free. `gradient.cppm` + `gradient.cpp` are `nevpt2.gradient`, the device assembly over the class energies' own `IntegralSource` slab walk (so the density-fitted path needs no change), with eight internal partitions. Three carry the ORBITAL source term: `:general` (`gradient_general.cppm` — the per-block promotion table of docs/gradient-theory.md, §3.1 (from `nevpt2.gradient_plan`) read against the sidecar's device arrays, the general-range action it generates out of the gradient sidecar's one-general-index arrays, the orbital energies' response, and the restriction identities that check all of it), `:rotation` (`gradient_rotation.cppm` — the kappa directions, the device action on a tensor's active axes, and a Dyall block paired with its own tangent), `:terms` (`gradient_terms.cppm` — `ScTerm`, evaluating a term list, differentiating it per operand, evaluating its forward tangent under the product rule, contracting a pseudodensity back) and `:sc_classes` (`gradient_sc_classes.cppm` — the eight classes, each written as the term list its `energy_<class>` evaluates). Two more carry the CI source term `y_I = dE/dc_I` (Park Eq. 48): `:rdm_adjoint` (`gradient_rdm_adjoint.cppm` — every `make_*` chain written out again as a term list and swept BACKWARDS, reverse-topologically, so the per-block accumulators `dE/dG` become the five RDM coefficient tensors `dE/d(dm1 … f3ca)`) and `:ci_source` (`gradient_ci_source.cppm` — those tensors contracted with the RDMs' own CI derivatives through `nevpt2.sigma_vector`, with `dm1` and `dm2` folded onto `dm3`'s diagonals so one 3-body sigma carries all three; its one definition, `ciSourceVector`, is declared in the exported interface so `test/gradient_plan/` can check its shape, its accumulation, and the fold against an unfolded host reference), which is what `scCiSourceDevice` returns. The seventh, `:pc_classes` (`gradient_pc_classes.cppm`, #74), is PC-NEVPT2's reverse mode: each PC class's solve (`pcClassMatrices` / `pcSolveClass`, exported from `nevpt2.energy` for it) differentiated into `dE/dT`, `dE/dK`, `dE/dS`, `dE/dS_1` and the slab pseudodensity, swept down through `:rdm_adjoint`'s per-block adjoints and paired along rotations with `:terms`' `slabDots` — `pcCiSourceDevice`, `pcFullRotationDevice` / `pcFullRotationMatrix`, `pcDigestAdjointDevice` and `pcDigestRotationTangentDevice`, refusing every class the PC solve refuses. `scActiveRotationDevice` is the same walk with rotation directions attached: it returns both modes of the active-active orbital source term `Y_act` per direction, and `scActiveRotationMatrix` probes it into a matrix; what each holds fixed, and the two pieces of the exact derivative that are therefore absent, are written out in `gradient.cppm`, "The active-rotation source term". `scFullRotationDevice` is the same again over the WHOLE `kappa` space, taking a `GeneralIntegralsDevice` (the sidecar, uploaded) beside the integrals: it adds every block's core, virtual and mixed promotions and the orbital energies' response — which closes one of those two pieces — and `scFullRotationMatrix` probes any requested set of index pairs into `Y_xy`. `scDigestRotationTangentDevice` is the one piece both of those hold fixed, in forward mode: the `f3ac`/`f3ca` digests' response along each direction, `buildRdmsDevice` rerun at the direction's dotted `h2e` and fed through `energiesTangentDevice` — no new kernel, and the reason `nevpt2.gradient` links `nevpt2::rdm_build` (#88). `scDigestAdjointDevice` is the same piece in reverse mode (#89): the walk's RDM sweep for `dE/d(f3ac)` / `dE/d(f3ca)`, then `nevpt2.rdm_build`'s transpose, giving `dE/d(h2e)` through the digests; handed to `scFullRotationDevice` / `scFullRotationMatrix` as `digestAdjoint`, it makes `Y` the exact source term. `scPseudocanonicalMultipliers` (in `nevpt2.sc_amplitudes`, host-only) turns that matrix's two inactive blocks into Park Eqs. 56-57's multipliers, refusing a degeneracy its division cannot resolve; `gradient.cppm`, "The full source term Y_xy", is the contract for that half. `scMixedHessianDevice` / `scMixedHessianVectorDevice` (#184) are the mixed Hessian block `H_kc` in its two orientations, the scalar forward over forward and the vector reverse in the CI vector (docs/hessian-theory.md, §5.5): the same walk carrying a mixed direction, with the digests' share merged per tuple (`ScRotationDir::f3acDot`); `gradient.cppm`, "The mixed orbital-CI block, H_kc", is their contract. `scOrbitalCurvatureDevice` (#154) is the first term of the orbital block `H_kk`, the integrals' own curvature `<Ehat', I''[u, v]> = K1 + K2` (docs/hessian-theory.md, §5.1 and §5.4): the same walk over directions that carry whole DOTTED INPUT SETS (`:rotation`'s `ScDotted`, which `:terms`' `slabDots` answers from before any promotion), built by the eighth partition, `:curvature` (`gradient_curvature.cppm` — the first-order rule at any matrix with the traced pairs an antisymmetric kappa cancels restored, the cross-slot term composed from the sidecar when one direction is active–active, and on a fitted case the three-index product jets of `B`, which `dfGeneralIntegralsDevice` now keeps as `GeneralIntegralsDevice::b`); `scMatrixRotationDevice`, `scDottedPairingDevice` and `scRestoredTracePairingDevice` expose its pieces, `orbitalSecondInputsDevice` / `orbitalMatrixInputsDevice` hand out its inputs whole, and `gradient.cppm`, "The orbital block's integral curvature", is the contract. `pcOrbitalCurvatureDevice` (#200) is the same term for PC-NEVPT2: the same generators paired against `:pc_classes`' walk (`pcMatrixRotationDevice`, `pcDottedPairingDevice` and `pcRestoredTracePairingDevice` are its pieces). `pcOrbitalSandwichDevice` / `pcOrbitalHessianDevice` (#202) are PC's other term of `H_kk` and the assembled `Eddot_kk`: `:pc_classes`' `pcClassSandwich` walks both directions' dotted slabs and `K'` blocks per class and finishes each tuple with `nevpt2.energy_finish`'s two-leg cross term, reverse through `pcAdjointSecond*` with the second direction as its direction; Sijrs is SC's `gradientSijrs` on the pair. It builds its metric and Dyall-Hamiltonian blocks with `nevpt2.energy`'s own exported `make_*`, which is why those are exported rather than hidden in `:shared`. The equation numbers throughout are Park's (see [`references.md`](references.md), "Analytical gradients"); what the assembly is checked against is in "The SC pseudodensity identities", "The active-rotation source term", "The full orbital source term" and "The CI source term's forward-reverse duality" in [`testing.md`](testing.md) |
| `apps/sc_pseudodensity/main.cppm` | `nevpt2_sc_pseudodensity`: the same golden file, RDM build and integrals as `nevpt2_demo`, then `nevpt2.gradient`'s assembly and the identity report its three ctest entries match. No finite difference anywhere. `kappa_check.cppm` is its `:kappa_check` partition — `--kappa-act`, the active-active orbital source term both ways, plus the one finite difference that pins the rotation convention (the energy stage rerun at rotated integrals). `kappa_full_check.cppm` is `:kappa_full_check` — `--kappa-full` (and `--gradgold PATH`), the FULL source term out of the sidecar: the duality over the whole `kappa` space, the sidecar's promotion table against the arrays it reads, the active block against `--kappa-act`'s own entry point, a core-internal and a virtual-internal finite difference, and Park Eqs. 56-57 — plus the digests' term in both modes beside the duality, the full duality with it, and its own finite difference. `hessian_kc_check.cppm` is `:hessian_kc_check` — `--sc-hessian-kc` (#184): the mixed block `H_kc`'s two orientations against each other by duality, Euler in the CI vector, the intra-space zeros, the full-range `f` identity and three finite differences of `y` along `kappa`, displaced with `:kappa_full_check`'s and `:kappa_check`'s own host rotations; with no sidecar, the active–active `w` alone and the refusals. `hessian_kc_pc_check.cppm` is `:hessian_kc_pc_check` — `--pc --sc-hessian-kc` (#201): PC's mixed block, its two orientations by duality, Euler in the CI vector against the gated `Y_pc`, the step-free active-rotation gauge against `pcHessianDevice`, the intra-space zeros of the one rotation term, per-class finite differences of the three-term `y` along `kappa` displaced as `:hessian_kc_check`'s are, and the refusals; on a fitted case too. `hessian_kk_check.cppm` is `:hessian_kk_check` — `--sc-hessian-kk-int` (#154): the orbital block's integral curvature, each term's two modes against each other, the traced pairs' and the intra-space zeros, the refusals, and ladders of the fixed-pseudodensity pairing along `exp(s u + t v)`, every block's axes rotated on the host by a block-diagonal `U`; on a fitted case the sidecar's and the three-index routes against each other. `hessian_kk_sandwich_check.cppm` is `:hessian_kk_sandwich_check` — `--sc-hessian-kk` (#158): the orbital block's sandwich K3–K5 forward against reverse, its zeros and its digests' curvature, the guarded-perturber refusals, and Richardson-extrapolated ladders of the SC total along `exp(s u + t v)` against the assembled `Eddot_kk`, displacing as `:hessian_kk_check` does. `hessian_kk_pc_check.cppm` is `:hessian_kk_pc_check` — `--pc --sc-hessian-kk` (#202): PC's sandwich forward against reverse in both orbital-energy modes, its merge, zeros and digests' curvature against `pcHessianDevice`, the same ladders on the PC total (`:hessian_kk_sandwich_check`'s, with PC's corners and pairing), the step-free gauge against `:hessian_kc_pc_check`'s `H_kc`, and the K2 refusal. `source_dump.cppm` is `:source_dump` — `--dump-source-terms PATH`: `scFullRotationMatrix` over every MO pair, `scCiSourceDevice`, and `Edot(kappa)` in its two terms along seeded directions, written to `PATH` and read back; its `assembleSourceTerms` is shared with `source_oracle.cppm`, `:source_oracle` — `--source-oracle PATH`: those two terms against PySCF's exact ones in a committed `.srcgold` (`loadSrcGold`), block by block, with `kKnownGaps`, the one list of blocks known wrong, each of which must still miss the tolerance while every other block must meet it — empty since #89, so every block must meet it. The comparison, that list and both schemes' tolerances are `nevpt2.source_compare` (`source_compare.cppm`, `source_compare.cpp`), a module library beside the binary for `nevpt2.gradsrc`'s reason, so `test/source_compare/` can hold it to every committed oracle — SC and PC (#95) — card-free; the binary compares a PC oracle with `--pc` alone (#74), and an SC one without it. The writer is `nevpt2.gradsrc` (`gradsrc.cppm`, `gradsrc.cpp` and its `:archive` partition): a NumPy `.npz` — stored NPY v1.0 members in a zip archive — written by hand, so neither a compression library nor Python enters the build. It is a module library declared beside the binary rather than a component under `src/` on purpose: `nevpt2.gradient` computes, and deciding to write what it computed is the binary's business; it is a library at all only so `test/gradsrc/` can round-trip it card-free |
| `src/gradient/ci_checks.*` | `nevpt2.ci_checks`, a **leaf**: the checks of the energy's CI-vector derivative both demos run in place of the golden comparison. `ci_checks_tangent.cpp` is `--sc-tangent` (`checkScTangent`): `energiesTangentDevice` against the class norms' exact affine increment, Euler's identity on the three homogeneous classes, and a one-time central finite difference (docs/testing.md, "The SC tangent's finite-difference cross-check"); and `--sc-hessian` (`checkScHessian`, #153, integral-direct only): `energiesHessianDevice`'s `Eddot(u, v)` against Euler at second order, the norms' second-order increment and one finite difference of `Edot` (docs/testing.md, "The SC Hessian's finite-difference cross-check"). `ci_checks_source.cpp` is `--gradient-source` (`checkCiSource`): `scCiSourceDevice`'s `y_I` against that tangent by the duality `<u, y> = Edot(u)`, plus the full Jacobian where `ndet` allows (docs/testing.md, "The CI source term's forward-reverse duality"); and `checkPcCiSource`, the same for PC's `y` (#74), integral-direct only. The SC pair takes a `CiCheckIntegrals` — an `ActiveIntegralsDevice`, any `IntegralSource`, its `--batch` and the active `h2e` the RDM build read — so one body serves `FullBlockSource` and `DfIntegralSource` alike (#162): `y_I` differentiates the CI vector at fixed integrals, so nothing in it differentiates a B tensor |
| `src/gradient/pc_tangent.*` | `nevpt2.pc_tangent`, a **leaf** too: the finite-difference check of `pcTangentsDevice`. It drives the analytic tangent at one state and `pcEnergiesDevice` at the displaced states `(c ± h u)/‖c ± h u‖`, forms the central differences and their Richardson table, and reports. It links the RDM build *and* the energy stage (as `nevpt2.gradient` now does too, for the digests' orbital response, #88), because a displaced state needs its own RDMs: `nevpt2.rdm_host_dm` for `dm1`/`dm2` on the host, `nevpt2.rdm_build` for `dm3`/`f3ac`/`f3ca` on the device. What it checks, and the three findings that constrain the direction and the step size, are in "The tangent with respect to the CI vector" in [`pc-nevpt2.md`](pc-nevpt2.md) |
| `apps/pc_tangent/main.cppm` | `nevpt2_pc_tangent`: the same golden file and flags as `nevpt2_demo --pc` (plus `--fd-steps`, `--fd-start-pow` and `--pc-hessian`, which runs `nevpt2.pc_tangent`'s second-derivative check instead, #152), then `nevpt2.pc_tangent`'s check and the report its four ctest entries match |
| `src/cli/` | `nevpt2.cli`: the whole command line of the four golden-reading binaries. `parseArgs(Binary, args)` walks argv against one table of every flag and which binary implements it (the `:binary_flags` partition; README.md, "Which binary takes which flag"): a flag no binary takes is a usage error, and a flag another binary implements is refused as `err_unsupported` rather than parsed and ignored. The common flags (`--golden`, `--tiles`, the digest family, `--pc`, `--profile`, `--rdm-tangent`, `--pool-threshold`) go through `parseCommonFlag`, each binary's own (`--sc-tangent`, `--gradient-source`, `--sc-hessian`; `--batch`, `--check-blocks`; `--kappa-act`, `--kappa-full`, `--gradgold`; `--fd-steps`, `--fd-start-pow`) into `Options`; `finalize` applies the checks that span the common flags (`--tiles >= 1`, `--ozaki-pairs` in 0..14, `--ozaki-pairs` / `--ozaki-check` only with `--ozaki`, `--mantissa-bits` in 1..53 and only with `--cublas`, one digest at most, `--cublas` only where it is built, `--rdm-tangent` only with a digest the tangent build has), after `checkHessianDigests` has refused any Hessian flag beside `--cublas`, `--ozaki` or `--fused-digest` as `err_unsupported` (#157; docs/hessian-theory.md, §9.1), and `parseArgs` then each binary's own. Every bad argument is a returned `Error`. Host-only suites in `test/cli/` |
| `apps/integral_direct/main.cppm` | `nevpt2_demo`: reads every integral block from the golden file, runs `nevpt2.rdm_build` and the device energy contraction, prints the report. `--sc-tangent`, `--gradient-source` and `--sc-hessian` are `nevpt2.ci_checks` (`src/gradient/ci_checks.*`), run over a `FullBlockSource` |
| `src/df_integrals/` | `nevpt2.df_integrals`: `DfIntegralSource`, the external integral slabs built on the device from the golden file's `B_aa`/`B_ca`/`B_va`/`B_cv` — one `wwrblasDgemm` (cuBLAS or hipBLAS through WarpWraps' `wwr.blas`) plus one permutation per slab — and the active `h2e` from `B_aa` |
| `apps/density_fit/main.cppm` | `nevpt2_df_demo`: reads a `*_df` golden file, uses **no** four-index two-electron array from it (except under the `--check-blocks` diagnostic), runs the same RDM build and class energies with `--batch`-sized slabs, and checks against PySCF's DF-NEVPT2. See "Density fitting" in [`reference-data.md`](reference-data.md). `--sc-tangent` and `--gradient-source` (#162) run `nevpt2.ci_checks` over its `DfIntegralSource`, with the orbital energies held fixed |

## 5. One host integer type

**`std::int64_t` is the one host size, extent, stride and loop-index type**,
spelled bare as `int64_t` inside `namespace nevpt2` through the tree's one
using-declaration of a std name (`src/common/int64.h`:
`namespace nevpt2 { using std::int64_t; }`, re-exported by `nevpt2.common`
and by `nevpt2.tensor`); code outside the namespace writes `std::int64_t`. `Tensor`/`DeviceTensor` dims, strides, `rank()`, `dim()`,
`size()`, and `sliceAxis`/`reshapeView`/`DeviceTensor::zeros`/`deviceEye` are
all `int64_t`; so are the link tables' counts and addresses, the RDM build's
`norb` and tile arithmetic, einsum plan building, the energy module's extents
and slab bounds, the golden reader's header sizes, and both demos.

**Every remaining 32-bit boundary narrows through `narrowTo` exactly once**. `narrowTo<To>(v[, hint])` aborts with the caller's `file:line`
when `v` does not fit, and the optional hint turns a limit into an
instruction — `narrowTo<int>(nK, "use more --tiles")` at the `--cublas`
digest. The boundaries are enumerable:

- the RDM, diag-slice and Sijrs kernel launchers (`norb`, `na`/`nb`, `nla`/`nlb`)
- the link tables' `int` storage
- `EinsumPlan`'s `int` counts
- the 32-bit `wwrblasDgemm` in `energy_pc.cpp`'s `gemm` and in the DF slab build
- the solver's `d`
- allocation sizes narrowing to `std::size_t` — so a negative size aborts
  instead of wrapping

**On the device, offsets widen to `int64_t` in one place**:
`common/device_index.h`'s `idx2`/`idx4`/`idx6` flatten a row-major index in
`int64_t`, so a kernel never writes a `(size_t)`/`(long long)` cast at the head
of an offset chain, and `gridFor(n, block)` (plus a 2-D
`gridFor(nx, ny, dim3)`, both over `idivup`) sizes every hand-written
`<<<>>>` grid. Kernel-local `int` indices (thread index, link-table entries,
`n2`/`n4`) stay `int` on purpose: 32-bit is cheaper on the GPU.

`narrowTo`'s failure goes through `nevpt2.error_handling`'s abort tier,
`check` (`error at file:line in function: ...`, then `std::abort()`). That
module imports only `std`, so `nevpt2.common` importing it pulls no GPU
runtime in.

The 64-bit BLAS entry points (`wwrblasDgemm_64`,
`wwrblasDgemmStridedBatched_64`) exist precisely so the consume and
`--blas-digest` GEMMs narrow nothing — before them, a checked `int` narrowing
of a leading dimension demanded more `--tiles` than the memory did.

## 6. Device resources: one stream, one pool, RAII everywhere

Each demo's `main()` creates exactly **one** `nevpt2::DeviceResources`
(`src/device_resources/`, `DeviceResources::create`) — the stream, a memory pool, a BLAS
handle and a dense-solver handle on device 0 — and passes it, or its
`stream()`, down. The stream comes from that one place and nowhere else.

**Every** launch, copy, memset, BLAS call, solver call, event and sync is on
that stream, and every allocation and free is stream-ordered on it. Every
device allocation is a `nevpt2::DeviceBuffer<T>`: it draws from the pool with
`wwrMallocFromPoolAsync`, zero-fills, and frees itself with `wwrFreeAsync`
when it goes out of scope. A `DeviceTensor` owns one or views one. **There is
no manual free** — scope the buffer when the free point matters for the peak,
and never hand out a view that outlives its owner.

`DeviceResources` satisfies WarpWraps' `device_handle_pool` concept, and
`create()` returns a `shared_ptr` (inside a `Result`, §8) with a private
constructor, so every buffer
can co-own it through `shared_from_this()` — the pool and stream always
outlive the last free.

Why the discipline is strict: the stream is `wwrStreamNonBlocking`, so it does
**not** synchronize with the legacy default stream in *either* direction. One
stream-less call — a `wwrMemcpy`, a synchronous `wwrMalloc`, a launch without
a stream, a library handle left on stream 0 — is a **silent race**, not a
slowdown. And **no sanitizer sees it**: compute-sanitizer's memcheck ran a
cross-stream premature free clean on every attempt ([`testing.md`](testing.md), "Sanitizers").

So `devtools/stream-lint.sh --strict` is a **gate**, not advice —
`cpp-tier.sh` and `cross-backend-check.sh` both fail on any finding. It flags
synchronous alloc/free, stream-less copies and memsets, device-wide syncs, a
literal `nullptr`/`0` where a stream goes, a *defaulted* stream parameter, and
a `<<<>>>` launch without a stream. It **reads text, not types**, so it cannot
see a null stream arriving through a variable or a default member initializer
— so don't write one. No stream parameter has a default.

Because frees are stream-ordered, a buffer may be freed while kernels reading
it are still queued, and `downloadTensor` (which synchronizes the stream
itself) needs no sync before it. The `wwrStreamSynchronize` calls that remain
each say why in a comment. Do not add a sync "to be safe" without saying what
it orders, and do not remove one without reasoning it out. A library that
allocates for itself (cuBLAS's workspace) does not draw from the pool. All of
this holds identically on HIP.

See [`performance.md`](performance.md), "One stream" and "The memory pool", for the
measurements.

## 7. Two binaries, one seam

The two demos differ in **exactly one** thing: where the external
two-electron integral blocks come from. `nevpt2.energy`'s SC and PC
entry points take an abstract `IntegralSource`, which hands back one
*slab* of a block at a time — the block with one axis restricted:

- `FullBlockSource` (integral-direct): slabs are views into full blocks
  uploaded from the golden file.
- `DfIntegralSource` (`src/df_integrals`): each slab is built on the device
  from three-index `B_*` tensors by a `wwrblasDgemm` plus a permutation, so no
  full external four-index block ever exists on the device.

The einsum subscripts are identical either way; only the operand is a slab.
That is what makes density fitting an *addition* rather than a second energy
path, and what lets both demos share `nevpt2.rdm_build` unchanged. `--batch`
bounds slab length. See [`implementation.md`](implementation.md) §3.3 (and §6
for the density-fitted variant) for the slab geometry, and [`reference-data.md`](reference-data.md), "Density fitting", for what it measured.

## 8. Error handling

A failure goes one of **two** ways, chosen by whose fault it is — and never
a third:

| tier | for | spelled | ends |
|---|---|---|---|
| **abort** | our bug: a broken invariant | `check(cond, what)` (`nevpt2.error_handling`) | `error at file:line in function: what`, then `std::abort()` |
| **abort** | a failed GPU runtime / BLAS / solver call | `gpuCheck(status)` (`nevpt2.wwr`) | WarpWraps' `kit::AbortPolicy`: `GPU error at file:line ...`, then `std::abort()` |
| **value** | not our bug: bad input, bad configuration, an unsupported request, a numerical refusal | return `Result<T>` / `Status` holding an `Error`; propagate with `NEVPT2_TRY` | `report(error)` once, at `main`, which returns 1 |

**The abort tier.** `check` takes the caller's `std::source_location`, so the
message names the call site, not `check`'s own file; a routine that checks on
its caller's behalf takes a defaulted `std::source_location` and passes it down
(`deviceEinsum*` → `buildPlan`, `Tensor::fromFlat`, `reshapeView`, PC's
`gemm`). `narrowTo` fails through it; so do einsum's plan checks, the tensor
shape checks, the energy contiguity checks, `GoldenFile::get` of an array
that was never `require`d, and the HIP `--cublas` stubs that
`requireCublasEmul` makes unreachable. `gpuCheck` lives in **`nevpt2.wwr`**,
not in `nevpt2.error_handling`: it is one template over every status type
WarpWraps' error layer knows (`wwrError_t`, `wwrblasStatus_t`,
`wwrsolverStatus_t`), so it belongs with the module that knows those types,
and every `*_bridge.h` launcher returns its `wwrError_t` straight into it.
The RAII wrappers `nevpt2.wwr` instantiates for `DeviceResources` (and the
device-buffer suite) hand the same `Abort<E>` policy to WarpWraps.

**The value tier.** `Error` carries an `ErrorKind` (`IO`, `InvalidConfig`,
`Unsupported`, `Numerical` — only `report`'s wording depends on it), a message,
and the `std::source_location` where it was **created**: the `err_io` /
`err_config` / `err_unsupported` / `err_numerical` factories default it to
their call site, and `NEVPT2_TRY` (a macro, so it lives in the plain header
`error_handling/error_macros.h`, pulled into a global module fragment) passes
the `Error` up unchanged, so `report` names the site that detected the
failure, not the `main` that printed it. `Result<T>` is
`std::expected<T, Error>` and `Status` is `std::expected<void, Error>`. What
returns one:

- `loadGolden` → `Result<GoldenFile>` (cannot open, short read, bad magic, a
  `reference_kind` it does not know, a header whose rank, extents or size the
  file cannot hold: `IO`), and `GoldenFile::require({...})` → `Status`, which
  each demo calls on the arrays it reads before any GPU work (the DF demo a
  second time for `--check-blocks`'s; its `B_*` tensors and both demos'
  `--pc` fields are tested with `arrays.contains` instead).
  `GoldenFile::requireSourceTermReference(what)` and
  `requireLagrangianReference(what)` → `Status` are the other side of that
  split: a golden built on CASSCF orbitals parses fine, the source-term gate
  (`--sc-tangent`, `--gradient-source`, `--dump-source-terms`,
  `--source-oracle`) accepts it because the source terms are derivatives at
  fixed orbitals and CI vector, and the Lagrangian gate refuses it as
  `Unsupported`, not `IO`, because the Z-vector → CPHF chain is derived for
  CASCI on RHF orbitals (docs/reference-data.md, "The file format").
- `loadGradGold` → `Result<GradGoldFile>`, the same contract on the gradient
  sidecar, plus the header checks its array shapes are written in terms of: a
  space partition that does not add up to `nmo`, an RHF electron count that
  cannot fill the core, an `mo_coeff` that is not `[nao, nmo]` (`IO`). A
  sidecar is found beside a golden by name, so `GradGoldFile::checkAgainst(golden)`
  → `Status` is what says the two describe one state — the same active space
  and the same core (`IO`).
- `loadSrcGold` → `Result<SrcGoldFile>`, the same contract on the SC
  source-term oracle (`*.srcgold`, #92): a partition that does not add up, or
  any of its five arrays missing or misshapen (`IO`).
  `SrcGoldFile::checkAgainst(golden, sidecar)` → `Status` refuses an oracle
  written for another state — its CI vector must be the golden's bit for bit
  (`IO`).
- `parsePoolThreshold` and each demo's `parseMantissaBits` → `Result`
  (`InvalidConfig`).
- `DeviceResources::create` → `Result` (no selectable device, or one without
  memory pools: `InvalidConfig`; creating the stream, pool and handles on a
  device that passed those checks still aborts through `gpuCheck`).
- `requireCublasEmul()` → `Status` (`Unsupported` on HIP).
- `DfIntegralSource::create` → `Result` (inconsistent `B_*` shapes: `IO`).
- `pcEnergiesDevice` → `Result` (a `Dsyevd` `devInfo != 0`: `Numerical`).
  A *refused* PC class is not an `Error` — it is a value, `PcStatus`, in the
  result.

`main` returns `int`, so it cannot `NEVPT2_TRY`: it tests each `Result` /
`Status` it receives, calls `report` on a failure and returns 1. A check
`main` makes itself on its own flags (a missing `--golden`, a negative
`--batch`) prints and returns 1 directly. There is no exit-code table; ctest
only needs nonzero.

**No exceptions.** Nothing in `src/` or `apps/` throws, and
`devtools/throw-lint.sh` keeps it that way: it fails on the keyword `throw`
(comments and string literals blanked first) and runs beside the stream lint
in `cpp-tier.sh` and `cross-backend-check.sh`. It is a lint rather than
`-fno-exceptions` because the `import std` and WarpWraps BMIs must be built
with the same flags as their importers, and because the standard library can
still throw (`std::bad_alloc`) — uncaught, that terminates, which is the abort
tier already.

`nevpt2.error_handling` is vendor-free (it imports only `std`), so
`nevpt2.common` can import it without pulling in a GPU runtime.
`nevpt2.wwr` re-exports it, so `import nevpt2.wwr;` names both tiers and
`gpuCheck`.

## 9. Testing, and why there is no CI

**Do not add a `.github/workflows/`.** Every golden ctest entry is labelled
`gpu` and needs a real card; `ctest -LE gpu` selects only the host-only
GoogleTest suites, which check logic and never an energy (and, in an
`asan`/`ubsan` build, that build's host canary, which proves only that the
sanitizer is on). A hosted runner could prove only that the tree compiles and
those helpers behave, while every claim this project makes is numerical. A
green badge that cannot fail on a wrong answer is worse than no
badge: it invites the belief that something was checked.

The gate is local and manual, and **saying what you ran is the whole
protocol**:

```bash
devtools/cpp-tier.sh                # CUDA, on a box with a card (runs the five gates below first)
devtools/cross-backend-check.sh     # does the OTHER backend still compile (the two lints too)
devtools/stream-lint.sh --strict    # both scripts above run this
devtools/throw-lint.sh              # no `throw` in src/, apps/ or test/ (section 8); likewise
uv run python reference_data/golden_layout_check.py   # the golden layout contract; cpp-tier.sh runs it
uv run ruff check                   # the Python lint; likewise
uv run pytest                       # reference_data/tests/, the oracle's own tests; likewise
devtools/golden-repro.sh            # NOT a phase: the small goldens regenerate byte for byte
```

[`testing.md`](testing.md), "No CI, and the local gate", has what each of
them does and why `golden-repro.sh` is not a phase.

**Two tiers.** The *golden tier* is the numerical one: each demo checks itself
against the committed golden and prints `PASS:`, and every such entry is `gpu`.
The *unit tier* (`test/`, GoogleTest) checks host logic one
component at a time — `test/<x>/` tests `nevpt2.<x>`, mirroring `src/` — and
is labelled `unit` (`ctest --preset unit`). A host-only suite has no `gpu`
label and runs with no card visible; one that needs a card is `REQUIRES_GPU`.
Each binary is registered one ctest entry per suite, plus a
`<target>.SuiteListIsComplete` drift guard that fails when the hand-written
suite list and the binary disagree (`cmake/README.md`, "Tests"). A test TU is
a plain TU: `#include <gtest/gtest.h>` first, then `import std;` and the
modules. It is added from the root `CMakeLists.txt` only, so the standalone
`cmake -S src` build stays gtest-free. The `compile-cuda` / `compile-hip` presets are
compile-and-link only; `cross-backend-check.sh` drives `compile-hip` through `CROSS_CHECK_PRESET`.

For the two tiers in full, the five sanitizer presets, what a change must be
clean under, and the limits of each tool, see [`testing.md`](testing.md).

## 10. The build layer

| piece | role |
|---|---|
| `cmake/nevpt2_toolchain.cmake` | everything before `project()` — clang + libc++, the `import std` gate UUID (pinned per CMake release), the backend and device arch |
| `cmake/add_cxx_module_library.cmake` | a host C++23 module library |
| `cmake/add_device_library.cmake` | a linked-in kernel library, pinned to C++20 on both backends, joined to the `nevpt2_device_libraries` umbrella that `cross-backend-check.sh --device-only` builds |
| `cmake/nevpt2_warpwraps.cmake` | embeds `deps/WarpWraps` with `add_subdirectory`, `EXCLUDE_FROM_ALL` + `WWR_COMPILE_TIME_ONLY` |
| `cmake/nevpt2_sanitizers.cmake` + canary helpers | the sanitizer tier; all three options default OFF |
| `cmake/add_gtest_executable.cmake`, `cmake/add_gtest_suite_tests.cmake` + `nevpt2_check_gtest_suites.cmake` | the GoogleTest unit tier: one binary, one ctest entry per suite (`unit`, plus `gpu`/`death` on request), and the drift guard |

See `cmake/README.md` for the macro signatures.

**The C++ build fetches one thing, and only with testing on**: GoogleTest
`v1.17.0`, by `FetchContent` in the root `CMakeLists.txt` under
`NEVPT2_BUILD_TESTING`, built from source because everything links libc++
(`BUILD_GMOCK` and `INSTALL_GTEST` OFF). So a first `default`/`hip` configure
needs the network; `compile-cuda`/`compile-hip` (testing OFF) and the standalone
`cmake -S src` fetch nothing. The one
submodule, `deps/WarpWraps`, is built in-tree, and `src/CMakeLists.txt`
refuses to configure without it (`git submodule update --init --recursive`).
Requirements: CMake ≥ 4.2, clang + libc++ with `clang-scan-deps` and
`libc++.modules.json`, and a CUDA toolkit (≥ 13.0, checked explicitly because
`find_package(CUDAToolkit)` does not necessarily pick the newest on a
multi-toolkit box) or ROCm.

Two equivalent build paths, which **must not share a build directory** (a
CMake cache records absolute paths): the top-level presets, which register the
ctest entries, and standalone `cmake -S src -B src/build`, which registers
none.
