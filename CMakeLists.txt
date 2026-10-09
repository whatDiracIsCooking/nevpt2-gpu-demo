# The SC-NEVPT2 GPU demo: one source tree (src/), two backends, C++23 modules.
#
# A BUILD TARGETS EXACTLY ONE BACKEND. NEVPT2_GPU_BACKEND is CUDA or HIP and is
# read BEFORE project() (cmake/nevpt2_toolchain.cmake), because it decides
# which toolchain has to exist: the CUDA build needs nvcc and never looks for
# ROCm, the HIP build needs hipcc and never looks for CUDA. Both build the SAME
# src/ files -- WarpWraps (deps/WarpWraps) supplies the backend-neutral wwr*
# names that make that possible; see src/CMakeLists.txt.
#
# This file only wires up testing; the real build lives in src/CMakeLists.txt,
# which is still a self-contained project you can configure directly:
#
#   cmake -S src -B src/build && cmake --build src/build -j
#
# That standalone path is what README.md documents and is deliberately left
# working: both entry points include the same cmake/nevpt2_toolchain.cmake
# before their project() call. What the top level adds is the ctest
# registration: the golden-file entries and the GoogleTest unit tier (test/).
cmake_minimum_required(VERSION 4.2)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(nevpt2_toolchain)

project(nevpt2 LANGUAGES ${NEVPT2_LANGUAGES})

message(STATUS "nevpt2: GPU backend = ${NEVPT2_GPU_BACKEND}")
add_subdirectory(src)

# --- tests ------------------------------------------------------------------
#
# The demo IS the test: it checks its own per-class norms/energies and the
# total against the golden PySCF answer baked into the .nevpt2gold file and
# prints `PASS:` only when every one matches to 1e-7. So a ctest entry is the
# binary plus a PASS_REGULAR_EXPRESSION -- there is no separate assertion tier
# to write, and a run that merely exits 0 without printing PASS (an output
# format change, say) still fails.
#
# EVERY GOLDEN ENTRY NEEDS A REAL CARD and is labelled `gpu` for it. The only
# entries that do not are the GoogleTest unit tier at the bottom (test/,
# labelled `unit`) and a host-sanitizer build's one host canary.
# So `ctest -LE gpu` runs the host-only unit suites -- logic (error kinds,
# propagation, index helpers), never an energy. A green `-LE gpu` run says
# nothing about a number this project computes.
#
# NEVPT2_BUILD_TESTING ON (the default) registers both tiers and FETCHES
# GoogleTest at configure time (FetchContent, below), so a first configure
# needs the network. OFF -- the compile-cuda / compile-hip compile-only presets --
# registers nothing and fetches nothing.
option(NEVPT2_BUILD_TESTING
       "Register the golden-file ctest entries and the GoogleTest unit tier (fetches googletest)"
       ON)

if(NEVPT2_BUILD_TESTING)
  enable_testing()

  # --- GoogleTest ---------------------------------------------------------------
  #
  # Fetched and built from source rather than found: everything here compiles
  # and links against libc++ (cmake/nevpt2_toolchain.cmake), so a distro
  # GoogleTest built against libstdc++ would not link. Built here, it inherits
  # CMAKE_CXX_FLAGS_INIT's -stdlib=libc++. Only under NEVPT2_BUILD_TESTING, so
  # a compile-only configure needs no network. No gmock (nothing mocks).
  # INSTALL_GTEST OFF: this project installs nothing.
  include(FetchContent)
  set(NEVPT2_GOOGLETEST_TAG "v1.17.0" CACHE STRING "googletest git tag to build against")
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG ${NEVPT2_GOOGLETEST_TAG}
    GIT_SHALLOW TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  # GoogleTest's own sources are plain C++ -- no named modules, no `import
  # std` -- so skip clang-scan-deps on every one of its TUs.
  foreach(_gt_target IN ITEMS gtest gtest_main)
    if(TARGET ${_gt_target})
      set_target_properties(${_gt_target} PROPERTIES CXX_SCAN_FOR_MODULES OFF)
    endif()
  endforeach()
  unset(_gt_target)

  # The sanitizer tier (cmake/nevpt2_sanitizers.cmake, all OFF by default).
  # Under the compute-sanitizer preset every demo entry below runs wrapped in
  # compute-sanitizer: the demo targets carry a TEST_LAUNCHER, set in src/
  # when they were created, which ctest applies to add_test(COMMAND <target>).
  # Nothing here sets it. The multiplier scales every TIMEOUT below.
  math(EXPR _nevpt2_timeout "1800 * ${NEVPT2_TEST_TIMEOUT_MULTIPLIER}")

  set(_gold_1010 "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas1010.nevpt2gold")
  set(_gold_1212 "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas1212.nevpt2gold")

  # A missing golden is a WARNING, not an error, and the test is simply not
  # registered -- the same convention the CAS(14,14) case relies on, since that
  # one is gitignored and absent in a fresh clone. Registering a test whose
  # input does not exist would turn "the data is missing" into a failure that
  # reads like "the demo is broken".
  #
  # Both binaries print the same PASS line prefix; add_golden_test runs
  # the integral-direct nevpt2_demo, add_df_golden_test the
  # density-fitted nevpt2_df_demo.
  function(_add_golden_test_with binary name gold)
    if(NOT EXISTS "${gold}")
      message(WARNING
        "nevpt2: golden file missing, test '${name}' not registered: ${gold}")
      return()
    endif()
    add_test(NAME ${name}
             COMMAND ${binary} --golden "${gold}" ${ARGN})
    # PASS_REGULAR_EXPRESSION makes ctest IGNORE the exit code, so a
    # sanitizer finding that does not stop the run -- compute-sanitizer
    # reports an error and lets the process carry on to print PASS; LSan
    # reports at exit, after it -- would pass on the regex alone, whatever
    # --error-exitcode says. FAIL_REGULAR_EXPRESSION wins over it: these are
    # the summary/report lines each tool prints only when it found something
    # (compute-sanitizer's ERROR/RACECHECK SUMMARY with a non-zero count,
    # ASan/LSan's "ERROR: ...Sanitizer", UBSan's "runtime error:"). None of
    # them can appear in an unsanitized run, so the normal presets are
    # unaffected.
    set_tests_properties(${name} PROPERTIES
      FAIL_REGULAR_EXPRESSION
        "ERROR SUMMARY: [1-9];RACECHECK SUMMARY: [1-9];ERROR: [A-Za-z]+Sanitizer;runtime error:"
      PASS_REGULAR_EXPRESSION "PASS: per-class norms/energies and the total match"
      LABELS "gpu"
      TIMEOUT ${_nevpt2_timeout})
  endfunction()
  function(add_golden_test name gold)
    _add_golden_test_with(nevpt2_demo ${name} "${gold}" ${ARGN})
  endfunction()
  function(add_df_golden_test name gold)
    _add_golden_test_with(nevpt2_df_demo ${name} "${gold}" ${ARGN})
  endfunction()

  # --pc: the same binaries checked against block2's
  # partially-contracted answer (`pc_class_energies`/`e_pc_total`, which only
  # some goldens carry -- docs/reference-data.md, "PC fields"). The demo prints
  # `PASS: PC-NEVPT2 ...` only when all eight classes AND the total match to
  # 1e-7 and no class was refused, and that PASS line is worded differently
  # from the SC one, so neither kind of entry can pass on the other's output.
  set(_nevpt2_pc_pass "PASS: PC-NEVPT2 per-class energies and the total match")
  function(add_pc_golden_test name gold)
    _add_golden_test_with(nevpt2_demo ${name} "${gold}" --pc ${ARGN})
    if(TEST ${name})
      set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "${_nevpt2_pc_pass}")
    endif()
  endfunction()
  function(add_df_pc_golden_test name gold)
    _add_golden_test_with(nevpt2_df_demo ${name} "${gold}" --pc ${ARGN})
    if(TEST ${name})
      set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "${_nevpt2_pc_pass}")
    endif()
  endfunction()

  # --tiles 3 is the fewest MEASURED to run correctly at CAS(10,10) on a 10 GB
  # card -- not a number picked for looks. docs/performance.md ("Why CAS(10,10)
  # needs tiling") has why an auto-sized budget picks a split that corrupts device
  # memory instead of failing cleanly.
  add_golden_test(nevpt2_cas1010 "${_gold_1010}" --tiles 3)

  # CAS(12,12) is ~47s on an RTX 3080 against CAS(10,10)'s ~1.8s, so it is
  # labelled `slow` as well and `ctest -LE slow` drops it. --tiles 40 is
  # comfortably above the measured floor: 17 now that the path allocates no
  # R2 (33 with it), with ~1 GB of the card held externally.
  add_golden_test(nevpt2_cas1212 "${_gold_1212}" --tiles 40)
  if(TEST nevpt2_cas1212)
    set_tests_properties(nevpt2_cas1212 PROPERTIES LABELS "gpu;slow")
  endif()

  # The cuBLAS fixed-point-emulation digest is a CUDA-only path (the
  # tensor-core trick has no ROCm equivalent; a HIP build rejects --cublas),
  # and it needs a SMALLER tile than the emitted path: cuBLAS silently
  # declines to emulate above a per-tile-K threshold and runs native fp64
  # instead. --tiles 3 at CAS(10,10) engages; see docs/performance.md, "The cuBLAS
  # fixed-point-emulation digest".
  if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
    add_golden_test(nevpt2_cas1010_cublas "${_gold_1010}" --tiles 3 --cublas)
  endif()

  # --- *_pc: PC-NEVPT2 ----------------------------------------------
  #
  # The same RDM build as the SC entry at the same --tiles; the PC stage runs
  # after the tile blocks are freed and does not raise the pool's used
  # high-water (docs/pc-nevpt2.md, "Cost"). The 12-orbital PC case is
  # CAS(10,12), not CAS(12,12): the cas1212 goldens carry no PC fields (the
  # generator refuses their ungapped metric). Its measured --tiles floor is 14
  # on the RTX 3080 and 7 on the RX 9060 XT (for --pc and SC alike); 40 is
  # the count PC was measured with, the same as the CAS(12,12) entries'.
  set(_gold_1012 "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas1012.nevpt2gold")
  add_pc_golden_test(nevpt2_cas1010_pc "${_gold_1010}" --tiles 3)
  add_pc_golden_test(nevpt2_cas1012_pc "${_gold_1012}" --tiles 40)
  if(TEST nevpt2_cas1012_pc)
    set_tests_properties(nevpt2_cas1012_pc PROPERTIES LABELS "gpu;slow")
  endif()

  # --- density-fitted ---------------------------------------------------------
  #
  # The *_df golden files are `generate_golden.py --df`: the whole calculation
  # density-fitted, the answer PySCF's DF-NEVPT2. They are a DIFFERENT
  # reference from the conventional files above (different integrals, so a
  # different CASCI state and E_corr, ~5e-5 Eh apart at CAS(10,10)) -- each
  # binary is only ever checked against a file of its own kind.
  set(_gold_1010_df "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas1010_df.nevpt2gold")
  set(_gold_1212_df "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas1212_df.nevpt2gold")
  set(_gold_tz1010_df "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvtz_cas1010_df.nevpt2gold")

  # A DF file also carries the four-index blocks rebuilt from its B tensors,
  # so the UNMODIFIED integral-direct binary must pass on it too: this checks
  # the DF golden file is self-consistent, independently of any DF code.
  add_golden_test(nevpt2_cas1010_df_blocks "${_gold_1010_df}" --tiles 3)

  # The DF binary, from B alone. --check-blocks also compares every DF-built
  # block against the file's four-index arrays (1e-10). The default --batch 8
  # splits every virtual-indexed block into two slabs here (nvirt=16).
  add_df_golden_test(nevpt2_df_cas1010 "${_gold_1010_df}" --tiles 3 --check-blocks)

  # cc-pVTZ: nvirt=48, so --batch 5 leaves a ragged last slab (48 = 9*5 + 3)
  # -- the slab boundaries are what this case is for, not memory pressure
  # (every external block here is < 2 MB).
  add_df_golden_test(nevpt2_df_ccpvtz_cas1010 "${_gold_tz1010_df}"
                     --tiles 3 --batch 5 --check-blocks)

  # The same RDM build as nevpt2_cas1212, so the same measured tile floor (17)
  # and the same --tiles 40.
  add_df_golden_test(nevpt2_df_cas1212 "${_gold_1212_df}" --tiles 40)
  if(TEST nevpt2_df_cas1212)
    set_tests_properties(nevpt2_df_cas1212 PROPERTIES LABELS "gpu;slow")
  endif()

  if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
    add_df_golden_test(nevpt2_df_cas1010_cublas "${_gold_1010_df}" --tiles 3 --cublas)
  endif()

  # --- *_pc, density-fitted -------------------------------------------
  #
  # The DF goldens that carry PC fields at these sizes; n2_ccpvdz_cas1212_df
  # has none. Same --tiles/--batch as the SC entries above (cc-pVTZ's --batch 5
  # for the ragged last slab), so the PC slab walk crosses the same boundaries.
  add_df_pc_golden_test(nevpt2_df_cas1010_pc "${_gold_1010_df}" --tiles 3)
  add_df_pc_golden_test(nevpt2_df_ccpvtz_cas1010_pc "${_gold_tz1010_df}"
                        --tiles 3 --batch 5)

  # --- --consume-emitted -------------------------------------------------------
  #
  # The f3 consume step runs through wwr.blas by default; the
  # generated loop kernels stay reachable for the A/B, and these entries keep
  # them PASSing on both backends (and next to the --cublas digest on CUDA).
  add_golden_test(nevpt2_cas1010_consume_emitted "${_gold_1010}"
                  --tiles 3 --consume-emitted)
  if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
    add_golden_test(nevpt2_cas1010_cublas_consume_emitted "${_gold_1010}"
                    --tiles 3 --cublas --consume-emitted)
  endif()

  # --- --blas-digest / --digest-emitted ----------------------------------------
  #
  # The --cublas digest's GEMM shape as plain native-fp64 wwrblasDgemm (cuBLAS
  # / rocBLAS), so -- unlike --cublas -- on both backends. The
  # default digest differs by backend because the A/B did (docs/performance.md,
  # "Native-fp64 BLAS digest"): BLAS on HIP, the emitted kernels on CUDA. So
  # each backend's entries run the digest that is NOT its default -- every
  # other entry above already runs the default one.
  if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
    add_golden_test(nevpt2_cas1010_blas_digest "${_gold_1010}" --tiles 3 --blas-digest)
    add_df_golden_test(nevpt2_df_cas1010_blas_digest "${_gold_1010_df}"
                       --tiles 3 --blas-digest)
  else()
    add_golden_test(nevpt2_cas1010_digest_emitted "${_gold_1010}"
                    --tiles 3 --digest-emitted)
    add_df_golden_test(nevpt2_df_cas1010_digest_emitted "${_gold_1010_df}"
                       --tiles 3 --digest-emitted)
  endif()

  # --- --ozaki --------------------------------------------------------------------
  #
  # The three digest GEMMs on int8 tensor cores by the Ozaki scheme
  # (src/ozaki/), through WarpWraps' <wmma.h>, so on both backends. All 64
  # digit pairs (the default --ozaki-pairs 14). The cc-pVTZ --pc entry is the
  # gate for any lossy digest: the tightest PC metric gap of the committed
  # goldens, where a truncated digest (--ozaki-pairs 5) gets Sr/Si refused
  # rather than drifting (docs/performance.md, "The int8 Ozaki
  # digest"). The small entries put the kernel under the sanitizer tier.
  add_golden_test(nevpt2_cas1010_ozaki "${_gold_1010}" --tiles 3 --ozaki)
  add_df_pc_golden_test(nevpt2_df_ccpvtz_cas1010_pc_ozaki "${_gold_tz1010_df}"
                        --tiles 3 --batch 5 --ozaki)

  # --- --profile -----------------------------------------------------------------
  #
  # --profile brackets every RDM-build launch and GEMM, every energy-stage
  # einsum and (df demo) every slab DGEMM with an event pair (nevpt2.profile).
  # These entries keep the
  # instrumented paths exercised and still PASS, i.e. the wrapping did not
  # reorder or drop a launch. The tables themselves are diagnostic output, not
  # asserted on.
  add_golden_test(nevpt2_cas1010_profile "${_gold_1010}" --tiles 3 --profile)
  add_df_golden_test(nevpt2_df_cas1010_profile "${_gold_1010_df}" --tiles 3 --profile)
  if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
    add_golden_test(nevpt2_cas1010_cublas_profile "${_gold_1010}"
                    --tiles 3 --cublas --profile)
  endif()

  # --- small active spaces: CAS(4,4) and CAS(8,8) --------------------------------
  #
  # The cases the sanitizer tier runs: compute-sanitizer multiplies
  # a run's time, so its test preset selects ONLY the `small` label (plus the
  # canaries). They are ordinary golden checks too -- `ctest --preset fast`
  # runs them on every backend. Same molecule and basis as the production
  # files (N2/cc-pVDZ), from the same generate_golden.py settings;
  # docs/testing.md ("The small cases") has why these two sizes.
  #
  # Every RDM-build and energy path is covered at least once: the default
  # digest and consume, the other digest (--blas-digest on CUDA,
  # --digest-emitted on HIP), --consume-emitted, --cublas (CUDA; engaged
  # bits=53 at CAS(8,8)), the --profile events, and density fitting with
  # --check-blocks. --tiles 3 is NOT a measured floor here (one tile fits
  # easily): it is there so the tile loop and the device-resident accumulate
  # run more than once. Density fitting's default --batch 8 leaves a ragged
  # last slab on both: 19 virtuals = 8 + 8 + 3 at CAS(4,4), 17 = 8 + 8 + 1 at
  # CAS(8,8).
  set(_gold_44 "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44.nevpt2gold")
  set(_gold_88 "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas88.nevpt2gold")
  set(_gold_44_df "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44_df.nevpt2gold")
  set(_gold_88_df "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas88_df.nevpt2gold")

  add_golden_test(nevpt2_cas44 "${_gold_44}" --tiles 3)
  add_golden_test(nevpt2_cas88 "${_gold_88}" --tiles 3)
  add_golden_test(nevpt2_cas88_consume_emitted "${_gold_88}" --tiles 3 --consume-emitted)
  add_golden_test(nevpt2_cas88_profile "${_gold_88}" --tiles 3 --profile)
  if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
    add_golden_test(nevpt2_cas88_blas_digest "${_gold_88}" --tiles 3 --blas-digest)
    add_golden_test(nevpt2_cas88_cublas "${_gold_88}" --tiles 3 --cublas)
  else()
    add_golden_test(nevpt2_cas88_digest_emitted "${_gold_88}" --tiles 3 --digest-emitted)
  endif()
  add_df_golden_test(nevpt2_df_cas44 "${_gold_44_df}" --tiles 3 --check-blocks)
  add_df_golden_test(nevpt2_df_cas88 "${_gold_88_df}" --tiles 3 --check-blocks)
  add_df_golden_test(nevpt2_df_cas88_profile "${_gold_88_df}" --tiles 3 --profile)
  add_golden_test(nevpt2_cas88_ozaki "${_gold_88}" --tiles 3 --ozaki)
  add_df_pc_golden_test(nevpt2_df_cas88_pc_ozaki "${_gold_88_df}" --tiles 3 --ozaki)
  foreach(_t nevpt2_cas44 nevpt2_cas88 nevpt2_cas88_consume_emitted nevpt2_cas88_profile
             nevpt2_cas88_blas_digest nevpt2_cas88_cublas nevpt2_cas88_digest_emitted
             nevpt2_df_cas44 nevpt2_df_cas88 nevpt2_df_cas88_profile
             nevpt2_cas88_ozaki nevpt2_df_cas88_pc_ozaki)
    if(TEST ${_t})
      set_tests_properties(${_t} PROPERTIES LABELS "gpu;small")
    endif()
  endforeach()

  # --- *_pc, small ----------------------------------------------------
  #
  # --pc at CAS(4,4)/CAS(8,8), both demos, labelled `small` so the sanitizer
  # tier runs the PC stage (its eigensolves on the solver handle, the BLAS
  # GEMMs, the slab einsums) too. All four goldens carry PC fields.
  add_pc_golden_test(nevpt2_cas44_pc "${_gold_44}" --tiles 3)
  add_pc_golden_test(nevpt2_cas88_pc "${_gold_88}" --tiles 3)
  add_df_pc_golden_test(nevpt2_df_cas44_pc "${_gold_44_df}" --tiles 3)
  add_df_pc_golden_test(nevpt2_df_cas88_pc "${_gold_88_df}" --tiles 3)
  foreach(_t nevpt2_cas44_pc nevpt2_cas88_pc nevpt2_df_cas44_pc nevpt2_df_cas88_pc)
    if(TEST ${_t})
      set_tests_properties(${_t} PROPERTIES LABELS "gpu;small")
    endif()
  endforeach()

  # --- salicylaldimine CAS(8,8)/6-31G** -------------------------------------------
  #
  # The first molecule that is not N2: salicylaldimine (C7H7NO, PubChem CID
  # 260739, flattened to Cs), its 4 pi + 4 pi* orbitals as the active space
  # (generate_golden.py --cas-irreps; docs/reference-data.md, "Salicylaldimine
  # CAS(8,8)"). 161 MOs, ncore=28, nvirt=125 -- the external spaces are far
  # larger than N2's, the active space the same size as n2_ccpvdz_cas88. The
  # DF file is fitted with 6-31G**-RIFIT (reference_data/basis/), not a JK
  # basis. --tiles 3 is not a measured floor (one tile fits, as for the N2
  # CAS(8,8)); the default --batch 8 leaves a ragged last slab (125 = 15*8 + 5).
  # Not `small`: the sanitizer tier stays on N2. The two SC entries are `slow`
  # because of HIP: ~35s on the RX 9060 XT, nearly all of it a few
  # small-output einsums in the SC energy stage, against ~1.2s on the RTX 3080
  # (the PC entries take ~2s / ~0.8s). The goldens are gitignored (145/166 MB) -- generate
  # them first, or these four are not registered.
  set(_gold_sal88 "${CMAKE_CURRENT_SOURCE_DIR}/golden/salicylaldimine_631gss_cas88.nevpt2gold")
  set(_gold_sal88_df "${CMAKE_CURRENT_SOURCE_DIR}/golden/salicylaldimine_631gss_cas88_df.nevpt2gold")
  add_golden_test(nevpt2_salicylaldimine_cas88 "${_gold_sal88}" --tiles 3)
  add_pc_golden_test(nevpt2_salicylaldimine_cas88_pc "${_gold_sal88}" --tiles 3)
  add_df_golden_test(nevpt2_df_salicylaldimine_cas88 "${_gold_sal88_df}"
                     --tiles 3 --check-blocks)
  add_df_pc_golden_test(nevpt2_df_salicylaldimine_cas88_pc "${_gold_sal88_df}" --tiles 3)
  foreach(_t nevpt2_salicylaldimine_cas88 nevpt2_df_salicylaldimine_cas88)
    if(TEST ${_t})
      set_tests_properties(${_t} PROPERTIES LABELS "gpu;slow")
    endif()
  endforeach()

  # --- sanitizer canaries ---------------------------------------------------------
  #
  # A deliberately buggy run (apps/sanitizer_canary/) that passes only when its
  # sanitizer reports the bug AND fails the run (add_sanitizer_canary).
  # Registered only under the sanitizer that must catch it, so a green canary
  # proves that sanitizer is on in this build. docs/testing.md ("Canaries") records
  # each one going red with its sanitizer off.
  include(add_sanitizer_canary)
  if(NEVPT2_ENABLE_ASAN)
    add_sanitizer_canary(
      NAME nevpt2_canary_asan_heap_overflow
      TARGET nevpt2_sanitizer_canary
      ARGS host-heap-overflow
      EXPECT "AddressSanitizer: heap-buffer-overflow")
  endif()
  if(NEVPT2_ENABLE_UBSAN)
    add_sanitizer_canary(
      NAME nevpt2_canary_ubsan_signed_overflow
      TARGET nevpt2_sanitizer_canary
      ARGS host-signed-overflow
      EXPECT "runtime error: signed integer overflow")
  endif()
  if(NEVPT2_COMPUTE_SANITIZER)
    set(_tool "${NEVPT2_COMPUTE_SANITIZER_TOOL}")
    if(_tool STREQUAL "memcheck")
      add_sanitizer_canary(
        NAME nevpt2_canary_memcheck_device_oob
        TARGET nevpt2_sanitizer_canary
        ARGS device-oob
        EXPECT "Invalid __global__ read of size 8"
        REQUIRES_GPU)
      # The bug stream-ordered allocation can introduce: a free that runs
      # before the kernel still reading the buffer.
      add_sanitizer_canary(
        NAME nevpt2_canary_memcheck_free_before_read
        TARGET nevpt2_sanitizer_canary
        ARGS device-free-before-read
        EXPECT "Invalid __global__ read of size 8"
        REQUIRES_GPU)
      # The same bug through the demos' own allocator: a pool-backed
      # DeviceBuffer, whose freed block the pool keeps mapped.
      add_sanitizer_canary(
        NAME nevpt2_canary_memcheck_pool_free_before_read
        TARGET nevpt2_sanitizer_canary
        ARGS device-pool-free-before-read
        EXPECT "Invalid __global__ read of size 8"
        REQUIRES_GPU)
      # A device allocation never freed: proves --leak-check full
      # is really on, so the demo entries' 0 leaks is a measurement.
      if(NEVPT2_COMPUTE_SANITIZER_LEAK_CHECK)
        add_sanitizer_canary(
          NAME nevpt2_canary_memcheck_leak
          TARGET nevpt2_sanitizer_canary
          ARGS device-leak
          EXPECT "Leaked 512 bytes at"
          REQUIRES_GPU)
      endif()
    elseif(_tool STREQUAL "initcheck")
      add_sanitizer_canary(
        NAME nevpt2_canary_initcheck_uninit_read
        TARGET nevpt2_sanitizer_canary
        ARGS device-uninit-read
        EXPECT "Uninitialized __global__ memory read of size 8"
        REQUIRES_GPU)
    elseif(_tool STREQUAL "racecheck")
      add_sanitizer_canary(
        NAME nevpt2_canary_racecheck_shared_race
        TARGET nevpt2_sanitizer_canary
        ARGS device-shared-race
        EXPECT "Race reported between"
        REQUIRES_GPU)
    elseif(_tool STREQUAL "synccheck")
      add_sanitizer_canary(
        NAME nevpt2_canary_synccheck_divergent_barrier
        TARGET nevpt2_sanitizer_canary
        ARGS device-divergent-barrier
        EXPECT "Barrier error detected"
        REQUIRES_GPU)
    endif()
    unset(_tool)
  endif()

  # --- the GoogleTest unit tier ------------------------------------
  #
  # One directory per component under test/, mirroring src/; every entry is
  # labelled `unit` (`ctest --preset unit`), and `gpu` too only if it needs a
  # card. Added from here only: the standalone `cmake -S src` build stays
  # gtest-free. cmake/README.md, "Tests".
  add_subdirectory(test)
endif()
