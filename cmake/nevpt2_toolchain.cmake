# nevpt2_toolchain.cmake -- everything that has to be decided BEFORE project().
#
# Included by whichever CMakeLists.txt calls project() first: the top level, or
# src/CMakeLists.txt when it is configured standalone (cmake -S src -B
# src/build). The include guard makes the second inclusion -- src/ added from
# the top level -- a no-op.
#
# Three things live here, and all three are read before project() because they
# decide which compilers and languages get enabled:
#   1. the C++23-modules toolchain: clang++ with libc++, clang-scan-deps, and
#      the release-specific `import std` gate UUID;
#   2. NEVPT2_GPU_BACKEND (CUDA or HIP), which decides whether the CUDA
#      language is enabled at all;
#   3. the device architecture, translated into the variable each backend's
#      CMake machinery reads (CMAKE_CUDA_ARCHITECTURES / GPU_TARGETS).
#
# The same preamble is what WarpWraps' own top-level CMakeLists.txt runs; it has
# to be repeated here because WarpWraps is embedded with add_subdirectory
# (cmake/nevpt2_warpwraps.cmake), i.e. AFTER this project's project() call, so
# its preamble arrives too late to choose the compiler for our targets.
include_guard(GLOBAL)

# --- `import std` gate ------------------------------------------------------
# Still experimental as of CMake 4.4, and CMake accepts a different gate UUID
# per release ON PURPOSE. A stale UUID is IGNORED silently and every `import
# std` then fails to resolve -- so pin it per verified release and hard-error on
# anything else. Keep in step with deps/WarpWraps/CMakeLists.txt.
#   4.2.x  ->  d0edc3af-4c50-42ea-a356-e2862fe7a444   (confirmed against 4.2.3)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.2 AND CMAKE_VERSION VERSION_LESS 4.3)
  set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
else()
  message(
    FATAL_ERROR
      "No verified CMAKE_EXPERIMENTAL_CXX_IMPORT_STD gate UUID for CMake "
      "${CMAKE_VERSION}. The UUID is release-specific and a stale one is "
      "ignored silently, breaking every `import std`. Look up this release's "
      "value (Help/dev/experimental.rst, or `strings $(command -v cmake)`), "
      "verify it, and add a branch in cmake/nevpt2_toolchain.cmake."
  )
endif()

# --- clang + libc++ -----------------------------------------------------------
# libc++ because `import std` resolves against libc++.modules.json; clang
# because it is the compiler that pairs with it. Both BEFORE project(), so
# compiler detection already sees them.
set(CMAKE_CXX_STANDARD_LIBRARY "libc++")
set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")
find_program(_nevpt2_cxx_compiler NAMES clang++ REQUIRED)
set(CMAKE_CXX_COMPILER "${_nevpt2_cxx_compiler}")

# Debian/Ubuntu put clang-scan-deps and libc++.modules.json under a versioned
# /usr/lib/llvm-<N>/ that is not on PATH; search there, newest first. Override
# with -DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS=... / -DCMAKE_CXX_STDLIB_MODULES_JSON=...
file(GLOB _llvm_roots LIST_DIRECTORIES true /usr/lib/llvm-* /usr/local/llvm-*)
list(SORT _llvm_roots COMPARE NATURAL ORDER DESCENDING)
set(_llvm_bin_hints "")
set(_llvm_lib_hints "")
foreach(_root IN LISTS _llvm_roots)
  list(APPEND _llvm_bin_hints "${_root}/bin")
  list(APPEND _llvm_lib_hints "${_root}/lib")
endforeach()

if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
  find_program(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS NAMES clang-scan-deps
               HINTS ${_llvm_bin_hints})
  if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
    message(FATAL_ERROR
      "clang-scan-deps not found (required for C++23 module scanning). Install "
      "the matching LLVM tools package, or pass "
      "-DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS=<path>.")
  endif()
endif()
if(NOT CMAKE_CXX_STDLIB_MODULES_JSON)
  find_file(CMAKE_CXX_STDLIB_MODULES_JSON NAMES libc++.modules.json
            HINTS ${_llvm_lib_hints} PATHS /usr/lib /usr/local/lib)
  if(NOT CMAKE_CXX_STDLIB_MODULES_JSON)
    message(FATAL_ERROR
      "libc++.modules.json not found (required for `import std`). Install "
      "libc++-dev built with module support, or pass "
      "-DCMAKE_CXX_STDLIB_MODULES_JSON=<path>.")
  endif()
endif()
unset(_llvm_roots)
unset(_llvm_bin_hints)
unset(_llvm_lib_hints)

# --- GPU backend --------------------------------------------------------------
# A build targets exactly one backend. The CUDA build needs nvcc and never looks
# for ROCm; the HIP build needs ROCm and never looks for CUDA.
set(NEVPT2_GPU_BACKEND "CUDA" CACHE STRING "GPU backend to build: CUDA or HIP")
set_property(CACHE NEVPT2_GPU_BACKEND PROPERTY STRINGS CUDA HIP)
if(NOT NEVPT2_GPU_BACKEND MATCHES "^(CUDA|HIP)$")
  message(FATAL_ERROR
    "NEVPT2_GPU_BACKEND must be CUDA or HIP, not '${NEVPT2_GPU_BACKEND}'")
endif()

# The CUDA language is enabled for the CUDA backend only because WarpWraps
# compiles every device library (cmake/add_device_library.cmake) as a
# CMake CUDA target; under HIP those sources are CXX with `-x hip`.
if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
  set(NEVPT2_LANGUAGES CXX CUDA)
  # Under the host sanitizers (cmake/nevpt2_sanitizers.cmake), nvcc's host
  # compiler is the same clang as CXX rather than whatever g++ is on PATH, so
  # both halves of a `.cu` are instrumented by, and link against, ONE sanitizer
  # runtime. Only then: the normal presets keep nvcc's default host compiler,
  # so they build exactly as before. -DCMAKE_CUDA_HOST_COMPILER overrides it.
  if((NEVPT2_ENABLE_ASAN OR NEVPT2_ENABLE_UBSAN) AND NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  endif()
else()
  set(NEVPT2_LANGUAGES CXX)
endif()

# --- device architecture ------------------------------------------------------
# One knob per backend, in the spelling the device compiler takes (CUDA_ARCH
# sm_86 as for nvcc -arch, HIP_ARCH gfx1200 as for hipcc --offload-arch),
# translated here into what CMake's own machinery reads, so the two cannot
# disagree. It is THE knob because every preset and doc speaks this spelling,
# and setting CMAKE_CUDA_ARCHITECTURES / GPU_TARGETS directly still wins,
# since each is only defaulted here when unset.
if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
  set(CUDA_ARCH "sm_86" CACHE STRING "device arch (RTX 3080); becomes CMAKE_CUDA_ARCHITECTURES")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "^sm_" "" _nevpt2_cuda_arch_num "${CUDA_ARCH}")
    set(CMAKE_CUDA_ARCHITECTURES "${_nevpt2_cuda_arch_num}")
    unset(_nevpt2_cuda_arch_num)
  endif()
else()
  set(HIP_ARCH "gfx1200" CACHE STRING "device arch (RX 9060 XT); becomes GPU_TARGETS")
  # hip-config-amd.cmake reads GPU_TARGETS; left unset it runs amdgpu-arch and,
  # on a box with no AMD GPU, silently falls back to gfx906 (wave64).
  if(NOT DEFINED GPU_TARGETS)
    set(GPU_TARGETS "${HIP_ARCH}" CACHE STRING "AMD GPU targets (from HIP_ARCH)")
  endif()
endif()
