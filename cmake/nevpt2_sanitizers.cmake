# nevpt2_sanitizers.cmake -- the sanitizer tier: host ASan/UBSan on
# both backends, NVIDIA compute-sanitizer on CUDA. All three default OFF, so
# the normal presets build and run exactly as before; the `asan`, `ubsan`,
# `compute-sanitizer`, `hip-asan` and `hip-ubsan` presets turn one on, each in
# its own build-* directory.
#
# Included once by src/CMakeLists.txt, AFTER the GPU toolkit is found (the
# compute-sanitizer search uses CUDAToolkit_BIN_DIR) and BEFORE
# nevpt2_warpwraps.cmake: add_compile_options is a directory property, so
# setting it here is what makes it reach WarpWraps' targets as well as ours.
# See docs/testing.md ("Sanitizers") for what each tool does and does NOT catch.
#
# nvcc's host compiler: under ASan/UBSan on CUDA, nevpt2_toolchain.cmake makes
# it the same clang as CXX (before project()), so both halves of a `.cu` link
# against ONE sanitizer runtime. The normal presets keep nvcc's default.
include_guard(GLOBAL)

# ========================================================================
# AddressSanitizer (host code only)
# ========================================================================
# NEVPT2_ENABLE_ASAN, not a bare ENABLE_ASAN: WarpWraps still honours a bare
# ENABLE_ASAN (with a deprecation warning) as its own WWR_ENABLE_ASAN, so the
# shared name would turn its ASan on too and apply the flags twice. For the
# same reason WWR_ENABLE_ASAN is deliberately NOT set: the options below
# already reach WarpWraps' targets (see the header).
#
# Host code is CXX units AND the host side of every `.cu`: under CUDA through
# -Xcompiler to nvcc's (clang) host compiler; under HIP a `.cu` is already a
# CXX unit (-x hip). Device code stays uninstrumented: nvcc has no device
# ASan at all, and on HIP -fno-gpu-sanitize keeps clang from trying (device
# ASan needs an xnack+ target, which gfx1200 is not -- docs/testing.md,
# "Sanitizers").
#
# Running a binary against a real GPU needs ASAN_OPTIONS=protect_shadow_gap=0:
# the CUDA/HIP runtime's address-space reservations collide with ASan's
# guarded shadow gap, and the first runtime call then fails with an
# out-of-memory on a card with plenty free. The asan/hip-asan test presets
# set it; a hand-run binary must too.
option(NEVPT2_ENABLE_ASAN "AddressSanitizer on host code (CXX + the host side of .cu)" OFF)

# ========================================================================
# UndefinedBehaviorSanitizer (host code only)
# ========================================================================
# Same reach as ASan. -fno-sanitize-recover makes the first finding abort the
# process, so a report can never sit in a passing test's log.
option(NEVPT2_ENABLE_UBSAN "UndefinedBehaviorSanitizer on host code (CXX + the host side of .cu)"
       OFF)

set(_nevpt2_host_sanitizers "")
if(NEVPT2_ENABLE_ASAN)
  list(APPEND _nevpt2_host_sanitizers address)
endif()
if(NEVPT2_ENABLE_UBSAN)
  list(APPEND _nevpt2_host_sanitizers undefined)
endif()
if(_nevpt2_host_sanitizers)
  list(JOIN _nevpt2_host_sanitizers "," _nevpt2_san)
  message(STATUS "nevpt2: -fsanitize=${_nevpt2_san} on host code (device code uninstrumented)")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${_nevpt2_san}>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=${_nevpt2_san}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-omit-frame-pointer>")
  if(NEVPT2_ENABLE_UBSAN)
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CXX>:-fno-sanitize-recover=undefined>
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-sanitize-recover=undefined>")
  endif()
  # On HIP, add_device_library adds -fno-gpu-sanitize to every -x hip
  # unit: host-only, explicitly. Without it clang tries the device pass too and,
  # for gfx1200, warns that it is ignoring -fsanitize there (device ASan needs
  # an xnack+ target ID, and "gfx1200:xnack+" is not a valid one). Not added
  # globally: on a plain CXX unit it is an unused-argument warning.
  add_link_options(-fsanitize=${_nevpt2_san})
  unset(_nevpt2_san)
endif()
unset(_nevpt2_host_sanitizers)

# ========================================================================
# NVIDIA compute-sanitizer (CUDA only)
# ========================================================================
# Not a compile-time sanitizer: it instruments at RUN time. So the build only
# adds -lineinfo (so a report names a source line), and every demo ctest
# entry runs wrapped in compute-sanitizer through CMAKE_TEST_LAUNCHER (set
# below). Wrapping per
# test, not around ctest, keeps labels, -R and per-entry verdicts. AMD has no
# counterpart, so a HIP configure refuses the option.
option(NEVPT2_COMPUTE_SANITIZER "Run each demo ctest entry under compute-sanitizer (CUDA only)"
       OFF)
set(NEVPT2_COMPUTE_SANITIZER_TOOL "memcheck"
    CACHE STRING "compute-sanitizer --tool: memcheck, initcheck, racecheck or synccheck")
set_property(CACHE NEVPT2_COMPUTE_SANITIZER_TOOL
             PROPERTY STRINGS memcheck initcheck racecheck synccheck)
# memcheck's --leak-check full: device allocations still live at context
# teardown. ON by default: every device allocation is an RAII
# DeviceBuffer, and both demos end with 0 leaked allocations. A leak counts
# as a memcheck
# error, so it fails the run; nevpt2_canary_memcheck_leak proves it does. Note
# that an existing build-compute-sanitizer cache keeps the value it was
# configured with.
option(NEVPT2_COMPUTE_SANITIZER_LEAK_CHECK "memcheck: also --leak-check full" ON)

if(NEVPT2_COMPUTE_SANITIZER)
  if(NOT NEVPT2_GPU_BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR
      "NEVPT2_COMPUTE_SANITIZER needs the CUDA backend: compute-sanitizer is "
      "NVIDIA's, and ROCm has no counterpart (docs/testing.md, \"Sanitizers\").")
  endif()
  if(NOT NEVPT2_COMPUTE_SANITIZER_TOOL MATCHES "^(memcheck|initcheck|racecheck|synccheck)$")
    message(FATAL_ERROR
      "NEVPT2_COMPUTE_SANITIZER_TOOL must be memcheck, initcheck, racecheck or "
      "synccheck, not '${NEVPT2_COMPUTE_SANITIZER_TOOL}'")
  endif()
  find_program(NEVPT2_COMPUTE_SANITIZER_EXECUTABLE NAMES compute-sanitizer
               HINTS "${CUDAToolkit_BIN_DIR}" REQUIRED)
  message(STATUS "nevpt2: compute-sanitizer (${NEVPT2_COMPUTE_SANITIZER_TOOL}) wraps the "
                 "demo ctest entries: ${NEVPT2_COMPUTE_SANITIZER_EXECUTABLE}")
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

  # CMAKE_TEST_LAUNCHER initializes the TEST_LAUNCHER property of every
  # executable target CREATED after this point -- it is read at
  # add_executable, NOT at add_test, so it has to be set here, in src/'s scope
  # before the demos' add_subdirectory, and not next to the root
  # CMakeLists.txt's add_test calls (where it silently wraps nothing: the
  # first draft of this tier did exactly that). ctest then runs every
  # add_test(COMMAND <that target>) under it. The canaries read the same
  # property off their own target (add_sanitizer_canary), so a launcher
  # that failed to attach turns them red instead of leaving every demo entry
  # silently unwrapped.
  #
  set(CMAKE_TEST_LAUNCHER "${NEVPT2_COMPUTE_SANITIZER_EXECUTABLE}" --tool
                          "${NEVPT2_COMPUTE_SANITIZER_TOOL}" --error-exitcode 1)
  if(NEVPT2_COMPUTE_SANITIZER_TOOL STREQUAL "memcheck" AND NEVPT2_COMPUTE_SANITIZER_LEAK_CHECK)
    list(APPEND CMAKE_TEST_LAUNCHER --leak-check full)
  endif()
  # racecheck: cuBLAS's own kernels are EXCLUDED, ours all stay in. Measured
  # (docs/testing.md, "Sanitizers"): unscoped, racecheck reports 32 shared-memory
  # hazard sites (~10^7 hazards) inside cuBLASLt's
  # cublasLt_fused_imma_dgemm_kernel_sm80, the --cublas emulated DGEMM, and
  # nothing in any kernel of ours. A shared-memory hazard is internal to the
  # kernel, so no argument we pass can cause one: it is cuBLAS's to answer
  # for. memcheck/initcheck stay unscoped, because a bad argument we pass DOES
  # fault inside the vendor's kernel. An include filter
  # (--kernel-name kns=<namespace>) would be the other way round, but it needs
  # every kernel in one namespace, and ours are not (kernels.cu's sit in the
  # global namespace). The exclude keeps every kernel of ours checked with no
  # naming rule to enforce.
  if(NEVPT2_COMPUTE_SANITIZER_TOOL STREQUAL "racecheck")
    list(APPEND CMAKE_TEST_LAUNCHER --kernel-name-exclude kns=cublas)
  endif()
endif()

# Scales every TIMEOUT the root CMakeLists.txt sets; a sanitized run is slower
# than the timeouts were sized for.
set(NEVPT2_TEST_TIMEOUT_MULTIPLIER "1"
    CACHE STRING "Integer factor applied to the nevpt2 ctest TIMEOUTs")
