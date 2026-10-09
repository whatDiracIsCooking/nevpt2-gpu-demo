# cmake/ — the build's helper layer

Kernels are linked in as a STATIC device library, launched through the
runtime API, via `add_device_library`; nothing is loaded at run time.

```
cmake/
├── README.md                              # this file
├── nevpt2_toolchain.cmake                 # everything BEFORE project(): clang + libc++,
│                                          #   the `import std` gate, NEVPT2_GPU_BACKEND, device arch
├── nevpt2_internal_helpers.cmake          # _nevpt2_require_args, _nevpt2_create_alias
├── add_cxx_module_library.cmake    # macro: a STATIC C++23 module library
├── add_device_library.cmake        # function: .cu -> a linked-in STATIC device library
├── nevpt2_warpwraps.cmake                 # embeds deps/WarpWraps with add_subdirectory
├── nevpt2_sanitizers.cmake                # ASan / UBSan / compute-sanitizer options (all OFF)
├── nevpt2_coverage.cmake                  # NEVPT2_ENABLE_COVERAGE: host source-based coverage (OFF)
├── add_sanitizer_canary.cmake      # function: a canary ctest entry
├── nevpt2_check_sanitizer_canary.cmake    # script mode: a canary's verdict
├── add_gtest_executable.cmake             # function: one GoogleTest binary under test/
├── add_gtest_suite_tests.cmake            # function: one ctest entry per suite + the drift guard
├── nevpt2_check_gtest_suites.cmake        # script mode: the drift guard's verdict
└── nevpt2_lsan.supp                       # LeakSanitizer suppressions (vendor runtimes only)
```

Both entry points, the root `CMakeLists.txt` and the standalone
`cmake -S src`, put this directory on `CMAKE_MODULE_PATH` and
`include(nevpt2_toolchain)` before their `project()` call. Every included
file has an `include_guard(GLOBAL)`, so the second include is a no-op when
`src/` is added from the root. (`nevpt2_check_sanitizer_canary.cmake` and
`nevpt2_check_gtest_suites.cmake` are not included; ctest runs them with
`cmake -P`.)

## `nevpt2_toolchain.cmake`

Sets everything that has to be decided before `project()`, because it picks
the compilers and languages:

- **clang++ with libc++**, plus `clang-scan-deps` and `libc++.modules.json`.
  It searches `/usr/lib/llvm-*` newest-first; override with
  `-DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS=…` / `-DCMAKE_CXX_STDLIB_MODULES_JSON=…`.
- **The `import std` gate UUID, pinned per CMake release.** A stale UUID is
  ignored silently and every `import std` then fails, so any release without
  a verified value is a hard error. Keep it in step with
  `deps/WarpWraps/CMakeLists.txt`.
- **`NEVPT2_GPU_BACKEND`** (CUDA | HIP). It sets `NEVPT2_LANGUAGES`: the CUDA
  language is enabled only for CUDA builds, and only because WarpWraps needs
  it.
- **The device arch, set once.** `CUDA_ARCH` (`sm_86`) becomes
  `CMAKE_CUDA_ARCHITECTURES`, and `HIP_ARCH` (`gfx1200`) becomes
  `GPU_TARGETS`. Left unset, hip-config's `GPU_TARGETS` silently falls back
  to gfx906 on a box with no AMD card.

## `add_cxx_module_library`

```cmake
add_cxx_module_library(
  NAME              nevpt2.einsum            # dotted; alias nevpt2::einsum
  PRIMARY_INTERFACE einsum.cppm              # export module nevpt2.einsum;
  [PARTITIONS       energy_shared.cppm ...]  # module nevpt2.energy:shared;
  [IMPLEMENTATION   einsum.cpp ...]          # module nevpt2.einsum;
  [LINK_PUBLIC      nevpt2::wwr ...]         # what the exported surface names
  [LINK_PRIVATE     nevpt2::einsum::plan ...]
  [DEFINES_PRIVATE  NEVPT2_HAVE_CUBLAS_EMUL ...])
```

The macro always:

- creates the target as `STATIC`, with the interface and any `PARTITIONS` in
  a `CXX_MODULES` file set;
- sets `cxx_std_23` PUBLIC and `CXX_MODULE_STD ON`;
- adds `-Wall`;
- creates the `::` alias.

**Use one spelling.** Declare a target by its dotted name and link it by its
`::` alias. A mistyped `::` name is a configure error, while a mistyped plain
name silently turns into `-l` and fails at link time, if at all.

**PUBLIC vs PRIVATE follows what the interface names.** If a module
`export import`s another, or its exported signatures name another module's
types, link that module PUBLIC (e.g. `nevpt2.einsum` → `nevpt2::wwr`,
`nevpt2::device_tensor`). Anything only the bodies use is PRIVATE.

**A define used by the interface can still be PRIVATE.** An importer reads
the compiled interface, not the macro. `nevpt2.cublas_emul` relies on this:
`kHaveCublasEmul` is fixed when the BMI is built, and `NEVPT2_HAVE_CUBLAS_EMUL`
never reaches a demo's `main.cppm`.

**`PARTITIONS` are how a module hides host code: there is no anonymous
namespace outside a `.cu` file.** An internal partition (`module
nevpt2.<x>:<part>;`, no `export`) holds what the implementation units need
and no importer may see: its declarations have module linkage, so they are
reachable from every unit that does `import :<part>;`. Two kinds:

- *Shared* declarations several units need, defined in one of them:
  `nevpt2.energy:shared` (`src/energy/energy_shared.cppm`), the
  intermediates SC's `energy.cpp` and PC's `energy_pc.cpp` both use.
- *Helpers*, moved whole (definitions included) out of the unit that uses
  them: `nevpt2.golden:reader`, `nevpt2.einsum:planner` and `:launch`,
  `nevpt2.rdm_build:staging`/`:blas`, and so on — one
  `<component>_<part>.cppm` each, as many as the helpers split into.

A partition is named for what it provides, never `:internal` or `:detail`.
Every helper in one module shares that module's linkage, so two same-named
helpers in different units of one module are a compile or link error rather
than silently separate (as they were under anonymous namespaces): rename one.
A partition that needs a macro (`NEVPT2_TRY`, `stdout`) or a kernel bridge
includes it itself, in its own global module fragment or `extern "C++"`
block. A partition may `import` its own primary interface for the types that
interface exports (clang 20 accepts it; [module.import] forbids that only to
non-partition implementation units), as long as the primary interface does
not import the partition back. A backend-specific partition is listed only
in that backend's `add_cxx_module_library` call and imported under the same
`#if` (`nevpt2.cublas_emul:raw_handle` / `:unavailable`).

The demos are not module libraries, but the same rule holds: each
`main.cppm` is its module's *primary interface*, which may import only
interface partitions and must re-export them ([module.unit]). So a demo's
helpers go in `export module nevpt2.app.<app>:<part>;` partitions that
`export` nothing, listed in the executable's `CXX_MODULES` file set and
pulled in with `export import :<part>;`. Nothing in them is exported, so
nothing escapes, and nothing imports a demo anyway. GoogleTest `.cpp` files
are not module units at all; they use `namespace nevpt2::test::<suite>`
instead.

The macro has no `IMPORT_STD` switch (it is always on), and no
`INCLUDE_CUDA_TOOLKIT`, `NO_CUDA_DEVICE_LINKING` or install-interface include
dirs. Add a keyword when a real call site needs it.

## `add_device_library`

```cmake
add_device_library(
  NAME         nevpt2.rdm.accumulate.device    # dotted, ends in .device
  SOURCES      rdm_accumulate.cu
  [LINK_PRIVATE wwr::extension::parallel_for]) # or wwr::device for runtime.h alone
target_include_directories(nevpt2.rdm.accumulate.device PRIVATE ${NEVPT2_SRC_ROOT})
```

A thin layer over WarpWraps' `wwr_add_gpu_device_library` (a STATIC library of
device-compiled `.cu`: the CUDA language under CUDA, CXX plus `hip::device`'s
`-x hip` under HIP). On top of it, the function:

- **pins device code to C++20 on both backends.** WarpWraps sets
  `CUDA_STANDARD 20` under CUDA, but under HIP the `.cu` is a CXX source and
  would inherit the host's `CMAKE_CXX_STANDARD 23`. A `CXX_STANDARD 20` target
  property, so CMake emits one `-std=gnu++20` (checked in
  `compile_commands.json`) and nothing about the host module scan changes;
- **joins the `nevpt2_device_libraries` umbrella**, which is what
  `devtools/cross-backend-check.sh --device-only` builds. A new kernel library
  is covered by being declared, rather than by someone remembering to edit a
  list (the hand-kept list it replaced had already missed `f3_digest`);
- **keeps the device pass uninstrumented under the host sanitizers on HIP**
  (`-fno-gpu-sanitize` when `NEVPT2_ENABLE_ASAN`/`_UBSAN` is on): gfx1200 cannot
  take device ASan, and clang would otherwise warn on every unit;
- **marks the target `NEVPT2_DEVICE_LIBRARY`**, a custom target property that
  `nevpt2_coverage.cmake`'s compile options test, so no `.cu` is ever
  coverage-instrumented on either backend.

The host side calls the kernel through a `*_bridge.h` included in the global
module fragment of the module unit that uses it; a purview declaration would
get module linkage and never bind to the `.cu`'s definition.

## `nevpt2_warpwraps.cmake`

`add_subdirectory(deps/WarpWraps … EXCLUDE_FROM_ALL)` with:

- `WWR_GPU_BACKEND` forwarded from `NEVPT2_GPU_BACKEND`;
- `WWR_COMPILE_TIME_ONLY=ON`, which skips WarpWraps' own GoogleTest fetch and
  test suites (ours is a separate, root-level fetch -- "Tests" below);
- `WWR_INSTALL=OFF`.

Only the WarpWraps targets ours link get built. Its tests and examples are
configured but not built, and `ctest -N` confirms none of them register with
our ctest.

**Imported targets are directory-scoped.** WarpWraps finds `hip` inside its
own subdirectory, where `src/` cannot see the result. So `src/CMakeLists.txt`
calls `find_package(hip)` itself; WarpWraps' second call is then a cached
no-op. On CUDA, `src/`'s own `find_package(CUDAToolkit)` already does the
same job. Without the HIP call, configure fails with "`hip::host` … target was
not found".

## `nevpt2_sanitizers.cmake`

The sanitizer tier. `src/CMakeLists.txt` includes it after the GPU toolkit is
found and **before** `nevpt2_warpwraps.cmake`: `add_compile_options` is a
directory property, so this is what makes the flags reach WarpWraps' targets
as well as ours. Every option defaults OFF except
`NEVPT2_COMPUTE_SANITIZER_LEAK_CHECK`, which does nothing unless
compute-sanitizer runs memcheck.

| option | effect |
|---|---|
| `NEVPT2_ENABLE_ASAN` | `-fsanitize=address -fno-omit-frame-pointer` on CXX units and, through `-Xcompiler`, the host side of every CUDA `.cu`. Link `-fsanitize=address` |
| `NEVPT2_ENABLE_UBSAN` | the same with `-fsanitize=undefined -fno-sanitize-recover=undefined` |
| `NEVPT2_COMPUTE_SANITIZER` | CUDA only (a HIP configure refuses it): `-lineinfo` on device code, and `CMAKE_TEST_LAUNCHER` = `compute-sanitizer --tool <NEVPT2_COMPUTE_SANITIZER_TOOL> --error-exitcode 1` (plus `--kernel-name-exclude kns=cublas` for racecheck, below) |
| `NEVPT2_COMPUTE_SANITIZER_TOOL` | `memcheck` (default), `initcheck`, `racecheck` or `synccheck`; a cache variable, so a reconfigure switches it without a rebuild |
| `NEVPT2_COMPUTE_SANITIZER_LEAK_CHECK` | memcheck's `--leak-check full`; **ON**: every device allocation is RAII, so both demos leak 0. Adds the `nevpt2_canary_memcheck_leak` canary. An existing cache keeps the value it was configured with |
| `NEVPT2_TEST_TIMEOUT_MULTIPLIER` | scales every ctest `TIMEOUT` the root `CMakeLists.txt` sets |

Three things to get right:

- **nvcc's host compiler.** Under ASan/UBSan on CUDA, `nevpt2_toolchain.cmake`
  sets `CMAKE_CUDA_HOST_COMPILER` to the same clang as CXX (before
  `project()`), so both halves of a `.cu` use one sanitizer runtime. The normal
  presets keep nvcc's default host compiler.
- **Do not set `WWR_ENABLE_ASAN`** (nor a bare `ENABLE_ASAN`, which WarpWraps
  still honours as its own): the directory-level flags above already reach
  WarpWraps, and its option would add `-fsanitize=address` a second time.
- **`CMAKE_TEST_LAUNCHER` is read at `add_executable`, not at `add_test`.** It
  initializes each executable target's `TEST_LAUNCHER`, which ctest applies to
  `add_test(COMMAND <target>)`. So it is set here, in `src/`'s scope before
  the demos exist. Set beside the root's `add_test` calls it would wrap
  nothing, silently. `test/` is `src/`'s sibling and does not inherit it:
  `test/CMakeLists.txt` copies it as `NEVPT2_GPU_TEST_LAUNCHER`, which
  `add_gtest_suite_tests` prepends to `REQUIRES_GPU` entries only ("Tests",
  below).

Running an ASan binary against a GPU needs `ASAN_OPTIONS=protect_shadow_gap=0`
(the runtime's address reservations collide with ASan's shadow gap). The
`asan` and `hip-asan` test presets set it, plus
`LSAN_OPTIONS=suppressions=cmake/nevpt2_lsan.supp`. A hand-run binary needs
the same.

racecheck is scoped by exclusion, not inclusion: it *excludes* cuBLAS
(`--kernel-name-exclude kns=cublas`), after measuring that its only reports
were inside cuBLASLt's emulated DGEMM. An include filter
(`--kernel-name kns=<namespace>`) would need every kernel of ours in one
namespace, and they are not (`kernels.cu`'s sit in the global namespace); the
exclude keeps every kernel of ours in scope with no naming rule to enforce.
synccheck runs unscoped (clean). There is no `--require-cuda-init no` (every
wrapped entry touches CUDA; the host-only unit suites are not wrapped).

## `nevpt2_coverage.cmake`

`NEVPT2_ENABLE_COVERAGE` (default OFF; the `coverage` and `hip-coverage`
presets turn it on) adds `-fprofile-instr-generate -fcoverage-mapping` as a
directory compile option, in the same place as the sanitizer flags and for the
same reason: it has to reach WarpWraps' targets and, through
`test/CMakeLists.txt`'s copy of src/'s directory options, the gtest binaries.
The option is a generator expression that is true only for a CXX unit whose
target is **not** `NEVPT2_DEVICE_LIBRARY`. That is how the `.cu` files stay
uninstrumented under HIP too, where they are CXX units: on both backends the
report is host module code and nothing else. `-fprofile-instr-generate` is
also a link option, because every binary links an instrumented library.

It refuses to combine with the sanitizer options. `devtools/coverage.sh`
does configure, build, ctest, merge and report; docs/testing.md, "Coverage"
says what the number does and does not mean.

## `add_sanitizer_canary`

```cmake
add_sanitizer_canary(
  NAME   nevpt2_canary_memcheck_free_before_read
  TARGET nevpt2_sanitizer_canary
  ARGS   device-free-before-read                   # the bug to run
  EXPECT "Invalid __global__ read of size 8"       # the tool's report line
  [REQUIRES_GPU])                                  # adds the `gpu` label
```

Registers a ctest entry that runs `nevpt2_check_sanitizer_canary.cmake`, which
passes only when the run exits non-zero **and** its output matches `EXPECT`.
It is not `WILL_FAIL`, which would credit any crash and miss a report printed
with exit 0. Labelled `sanitizer_canary`. The root `CMakeLists.txt` registers
each canary only when its sanitizer is on. The launcher is read off the
target's own `TEST_LAUNCHER` property, not passed as an argument, so
a device canary also proves the demo entries are wrapped. To show a canary
going red with its sanitizer off, run the script by hand with an empty
`LAUNCHER` against an unsanitized binary:

```bash
cmake -DLAUNCHER= -DEXE=build/src/nevpt2_sanitizer_canary -DARGS=host-heap-overflow \
      -DEXPECT="AddressSanitizer: heap-buffer-overflow" -P cmake/nevpt2_check_sanitizer_canary.cmake
# -> "Canary exited 0: the sanitizer did not fail the run ..." and a non-zero exit
```

## Tests

The GoogleTest unit tier. The golden tier -- the demo entries the
root `CMakeLists.txt` registers, all `gpu` -- owns every numerical claim; the
unit tier checks host logic one component at a time.

**GoogleTest** is fetched by the root `CMakeLists.txt` with `FetchContent`
(tag `v1.17.0`, `NEVPT2_GOOGLETEST_TAG`), only when `NEVPT2_BUILD_TESTING` is
ON, so `compile-cuda`/`compile-hip` configure with no network. It is built from source
because everything links libc++ (a distro libstdc++ GoogleTest would not
link); `BUILD_GMOCK` and `INSTALL_GTEST` are OFF, and `gtest`/`gtest_main` get
`CXX_SCAN_FOR_MODULES OFF` (plain C++, nothing to scan). `test/` is added from
the root only, after the golden entries, so the standalone `cmake -S src`
build has no gtest.

**`test/CMakeLists.txt`** copies `src/`'s directory `COMPILE_OPTIONS` and
`LINK_OPTIONS` (the libc++ link flag, the host ASan/UBSan flags): `test/` is
`src/`'s sibling, not its child, and a test binary has to be compiled and
linked like the module libraries it imports. One directory per component,
`test/<x>/` testing `nevpt2.<x>`.

### `add_gtest_executable`

```cmake
add_gtest_executable(
  NAME    error_handling_tests
  SOURCES error_handling_tests.cpp          # no main(): GTest::gtest_main
  LINK    nevpt2::error_handling nevpt2::common)
```

Links `GTest::gtest_main`, sets `cxx_std_23`, `CXX_MODULE_STD ON`,
`CXX_SCAN_FOR_MODULES ON` and `-Wall`. A test TU is a plain TU, not a module
unit: `#include <gtest/gtest.h>` (and any repo header, e.g.
`"error_handling/error_macros.h"` for `NEVPT2_TRY`) **first**, then
`import std;` and `import nevpt2.<x>;`. Ported from WarpWraps'
`wwr_add_gtest_executable` without its CUDA device-linking switch, which
turned out not to be needed: a suite's kernels are an `add_device_library`
built with separable compilation OFF, like every device library here, so the
CXX test binary that links it has no device-link step (`test/common/`).

### `add_gtest_suite_tests`

```cmake
add_gtest_suite_tests(
  TARGET  error_handling_tests
  SUITES  ErrorKindTests ErrorValueTests TryMacroTests ReportTests CommonIntegerTests
  [TIMEOUT 120]                             # seconds, x NEVPT2_TEST_TIMEOUT_MULTIPLIER
  [REQUIRES_GPU]                            # + label `gpu`
  [DEATH])                                  # + label `death` (EXPECT_DEATH suites)
```

One ctest entry per suite, named after it, running
`<target> --gtest_filter=<Suite>.*`, labelled `unit` (+ `gpu` / `death`).
Per suite rather than per binary so a failure names the suite; per suite
rather than per case so the list stays short enough to keep by hand.

`REQUIRES_GPU` marks a binary that allocates, launches or makes a library
handle. Nothing calls `GTEST_SKIP`, so on a card-less box such a suite fails
rather than skips; the label is what lets `-LE gpu` leave it out by name. A
host-only suite leaves it off and runs with no card visible. Under the
`compute-sanitizer` preset a `REQUIRES_GPU` entry runs wrapped in
`NEVPT2_GPU_TEST_LAUNCHER` (`src/`'s `CMAKE_TEST_LAUNCHER`, copied by
`test/CMakeLists.txt`); a host-only entry is never wrapped, because
compute-sanitizer fails a process that makes no GPU runtime call ("Target
application terminated before first instrumented API call").

A GPU suite gets its device from **`nevpt2.test.shared_resources`**
(`test/utils/shared_resources/`): `nevpt2::test::sharedResources()`
returns the process's one `DeviceResources` (device 0), created on the first
call -- so a host-only suite in the same binary never brings a device up --
and released by a registered `::testing::Environment`'s `TearDown()`, inside
`RUN_ALL_TESTS`, while the runtime is still up. It returns
`DeviceResources::create`'s `Result`, so a card-less run fails each case with
that `Error`'s message instead of aborting. `test/utils/` holds such
test-support libraries and registers no ctest entry; it is added before the
suites. `test/device_tensor/` (`device_tensor_tests`, `REQUIRES_GPU`) is the
first GPU suite: `DeviceTensor` / `DeviceBuffer` upload-download round trips
and the zero fill. `test/common/` is the first with kernels of
its own: `warp_reduce.cuh` / `block_reduce.cuh` exercised through one
`add_device_library` per suite and a plain-type `*_bridge.h` the test TU
includes after its imports (`<runtime.h>` before them); under HIP that
directory calls `find_package(hip)` itself, as `src/` does, because the
`hip::*` imported targets are not visible in `test/`. `DEATH` marks
`EXPECT_DEATH` suites (the abort tier); name them `...DeathTest`.

**The drift guard.** Because the suite list is written by hand, a suite left
off it would build, never run, and leave ctest green. So each target also
gets `<target>.SuiteListIsComplete`, which runs `nevpt2_check_gtest_suites.cmake`
on `<target> --gtest_list_tests` and fails if the binary's suites and the
union of every `SUITES` given for that target differ in either direction
("registered but NOT listed" / "listed but not registered"). It is registered
once per target, deferred to the end of the directory, so a target split over
several calls (plain suites, then the `DEATH` ones) is checked against the
union. Labelled `unit` only, never `gpu`: listing tests constructs no fixture.

Ported from WarpWraps' `wwr_add_gtest_suite_tests`, which is **not called**:
it finds its script via `${PROJECT_SOURCE_DIR}/cmake/`, which is this
project's `cmake/` when WarpWraps is embedded. This one resolves the script
next to itself (`CMAKE_CURRENT_FUNCTION_LIST_DIR`). `TYPED_SUITES`/`TYPES` were
dropped until a typed suite exists.

```bash
ctest --preset unit                         # the whole unit tier, guards included
ctest --preset unit -LE gpu                 # just what runs with no card
ctest --preset unit -L death                # just the death tests
```
