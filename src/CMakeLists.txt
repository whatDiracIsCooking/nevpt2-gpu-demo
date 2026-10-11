# SC- and PC-NEVPT2 demos -- one source tree, CUDA or HIP, C++23 named modules.
#
# A BUILD TARGETS EXACTLY ONE BACKEND, picked by NEVPT2_GPU_BACKEND (CUDA, the
# default, or HIP). The same files serve both, through WarpWraps
# (../deps/WarpWraps, a git submodule, embedded with add_subdirectory --
# cmake/nevpt2_warpwraps.cmake):
#   - host code is C++23 named modules (clang + libc++, `import std;`), written
#     once against WarpWraps' wwr* runtime names via `import wwr.runtime_api;`
#     (re-exported by nevpt2.wwr);
#   - kernel files are C++20 and #include WarpWraps' `runtime.h`, which picks
#     the vendor runtime header off the compiler's own device macro;
#   - every kernel file is a STATIC device library linked into the binary
#     (cmake/add_device_library.cmake, over WarpWraps'
#     wwr_add_gpu_device_library) and launched through the runtime API.
#     Nothing is loaded at run time, and no host unit but the CUDA-only
#     cublas/cublas_emul.cpp names a vendor API.
#
# One directory per component, each its own STATIC module library (or, for
# kernels, device library), linked by its `::` alias. Every directory here is
# something both methods run -- the RDM build, the class energies, and the
# layers under them -- except df_integrals, the density-fitted demo's alone,
# and gradient, which only apps/sc_pseudodensity/ drives. The host binaries are in ../apps/, a sibling of src/ but
# built from here so the standalone `cmake -S src` path keeps them:
#
#   ../apps/integral_direct/  SC- and PC-NEVPT2 from the full MO-integral blocks (nevpt2_demo)
#   ../apps/density_fit/      SC- and PC-NEVPT2 from three-index (L|pq) tensors (nevpt2_df_demo)
#   ../apps/sc_pseudodensity/ the SC gradient's pseudodensity identities (nevpt2_sc_pseudodensity)
#   ../apps/sanitizer_canary/ deliberately buggy runs for the sanitizer tier
#
# Dependency order:
#
#   error_handling nevpt2.error_handling  Error / Result / Status, check,
#                                error_macros.h (NEVPT2_TRY) (std only)
#   common   nevpt2.common       idivup / align_up, int64_t + checked
#                                narrowTo<To> (error_handling; no GPU)
#            nevpt2.common.device  INTERFACE: device-only headers (block_params,
#                                device_index, device_functor, warp_reduce,
#                                block_reduce) for kernel files
#   wwr      nevpt2.wwr          wwr* names, bare in nevpt2, gpuCheck +
#                                kGpuBackendName; re-exports wwr.runtime_api
#                                and error_handling; WarpWraps' RAII wrappers
#                                with Abort                (wwr.extension.*)
#   device_resources nevpt2.device_resources  DeviceResources: the one
#                                non-blocking stream, memory pool, BLAS and
#                                solver handles; DeviceBuffer<T>        (wwr)
#   profile  nevpt2.profile      the --profile recorder: labelled GPU event
#                                pairs                        (wwr, common)
#   tensor   nevpt2.tensor       host N-d Tensor (no GPU)
#            nevpt2.device_tensor  DeviceTensor + upload/download     (device)
#   golden   nevpt2.golden       the .nevpt2gold reader               (tensor)
#   cublas   nevpt2.cublas_emul  --cublas digest; real on CUDA, refusing stubs on HIP (device)
#   rdm      nevpt2.link_tables  Knowles-Handy link tables (no GPU)
#            nevpt2.rdm_plan     tile plan, consume chunking / permuted eri (no GPU)
#            nevpt2.rdm_build    the tiled dm3/f3ac/f3ca build        (device, cublas)
#            + device library: rdm_launch.cu (#includes kernels.cu)
#            + device library: rdm_accumulate.cu
#            + device library: f3_digest.cu
#            + device library: f3_scatter.cu (both GEMM digests)
#   einsum   nevpt2.einsum       device einsum planner/launcher       (device_tensor)
#            nevpt2.einsum.plan  einsum_plan.h, shared host/device (INTERFACE)
#            + device library: device_einsum.cu
#   energy   nevpt2.energy_finish  slab walk, SC/PC finish arithmetic, PC
#                                result types                (tensor; no GPU)
#            nevpt2.energy       the eight class energies, over slabs of an
#                                IntegralSource: SC, and PC (one eigensolve
#                                per class on DeviceResources' solver
#                                handle, the same slabs)              (einsum)
#   df_integrals nevpt2.df_integrals  density-fitted IntegralSource: slabs
#                                built from B by DGEMM          (energy, wwr.blas)
#   gradient nevpt2.sc_amplitudes  the SC amplitude T and the gradient's
#                                multipliers P/Q/R, plus the einsum-subscript
#                                rewriting that differentiates a class term
#                                                     (energy_finish; no GPU)
#            nevpt2.gradient     the SC pseudodensities, assembled over the
#                                class energies' own slab walk and contracted
#                                back against their integrals. A LEAF:
#                                nothing in src/ imports it and neither demo
#                                links it                 (energy, einsum)
#   ../apps/integral_direct      nevpt2_demo     main.cppm
#   ../apps/density_fit          nevpt2_df_demo  main.cppm
#   ../apps/sc_pseudodensity     nevpt2_sc_pseudodensity  the SC pseudodensity
#                                identity checks (nevpt2.gradient's only caller)
#   ../apps/sanitizer_canary     nevpt2_sanitizer_canary  deliberately buggy runs
#                                for the sanitizer tier's canaries (+ device library)
#
# rdm/ IS shared: the f3ac/f3ca digests contract the 4-RDM with the ACTIVE
# h2e only (n_act^4), which the density-fitted demo rebuilds exactly from B_aa
# and hands to the same build. What density fitting replaces is the external
# blocks the class energies read -- hence energy/'s IntegralSource.
#
# The kernels are compiled ONCE, regardless of active space. Every kernel file
# is a device library (add_device_library) linked into the
# binary and called through a *_bridge.h; nothing is loaded at run time.
# kernels.cu is reached through rdm_launch.cu, which #includes it and holds
# its launchers.
#
# Standalone (no ctest) works too: cmake -S src -B src/build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 4.2)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
  include(nevpt2_toolchain)
  project(nevpt2_demo LANGUAGES ${NEVPT2_LANGUAGES})
endif()

# Host C++23. EXTENSIONS must match what libc++'s std module was compiled with,
# which is CMake's default (ON) -- leave it there.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(FATAL_ERROR "nevpt2: C++23 named modules need clang (got ${CMAKE_CXX_COMPILER_ID})")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_link_options($<$<LINK_LANGUAGE:CXX>:-stdlib=libc++>)

include(add_cxx_module_library)

# --- the GPU toolkit -----------------------------------------------------------
if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
  find_package(CUDAToolkit REQUIRED)
  # The --cublas digest needs CUDA 13.0 symbols (CUBLAS_COMPUTE_64F_EMULATED_
  # FIXEDPOINT, cublasSetFixedPointEmulationMantissaBitCountPointer), and
  # WarpWraps' floor is 13.0 too. On a box with several toolkits
  # find_package(CUDAToolkit) does NOT necessarily pick the newest -- the
  # reference box has 12.1, 12.9 and 13.0 and it chose 12.9 unprompted, so every
  # preset pins CUDAToolkit_ROOT. Checked here, so the failure is one line
  # naming the fix.
  if(CUDAToolkit_VERSION VERSION_LESS 13.0)
    message(FATAL_ERROR
      "nevpt2: found CUDA ${CUDAToolkit_VERSION} at ${CUDAToolkit_LIBRARY_ROOT}, "
      "but the CUDA backend needs >= 13.0 (the --cublas fixed-point-emulation "
      "digest, and WarpWraps' floor). Point CMake at a newer toolkit, e.g. "
      "-DCUDAToolkit_ROOT=/usr/local/cuda-13.0")
  endif()
  message(STATUS "nevpt2: CUDA ${CUDAToolkit_VERSION} (${CUDAToolkit_LIBRARY_ROOT}), ${CUDA_ARCH}")
else()
  # Found HERE as well as inside WarpWraps, and not redundantly: an IMPORTED
  # target is visible only in the directory that found it and below, and
  # WarpWraps is a sibling subdirectory -- so without this, hip::host (linked
  # by wwr::runtime_api, and hip::device by every device library) does not exist in
  # src/. WarpWraps' own find_package(hip) is then a cached no-op. (The CUDA
  # half needs no counterpart: find_package(CUDAToolkit) above already puts
  # CUDA::* in scope.)
  find_package(hip CONFIG REQUIRED)
  message(STATUS "nevpt2: HIP ${hip_VERSION}, ${HIP_ARCH}")
endif()
# The include root kernels and einsum_plan.h's users reach root-relative
# ("einsum/einsum_plan.h", "common/block_reduce.cuh",
# "error_handling/error_macros.h"): src/ itself, every component's parent. The
# demo binary still lands in NEVPT2_SRC_BINARY_DIR (src/build/nevpt2_demo).
set(NEVPT2_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(NEVPT2_SRC_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
include(add_device_library)

# --- sanitizers (all default OFF) ------------------------------------------------
# Before WarpWraps, so the host ASan/UBSan flags reach its targets too (a
# directory property); after the toolkit, which the compute-sanitizer search
# needs. See cmake/nevpt2_sanitizers.cmake.
include(nevpt2_sanitizers)

# --- coverage (default OFF) -------------------------------------------------------
# Same placement, same reason: a directory property that must reach WarpWraps'
# targets and, through test/'s copy, the gtest binaries. See
# cmake/nevpt2_coverage.cmake.
include(nevpt2_coverage)

# --- WarpWraps ----------------------------------------------------------------
include(nevpt2_warpwraps)

# --- components, in dependency order -------------------------------------------
add_subdirectory(error_handling)
add_subdirectory(common)
add_subdirectory(wwr)
add_subdirectory(device_resources)
add_subdirectory(profile)
add_subdirectory(tensor)
add_subdirectory(golden)
add_subdirectory(einsum)
add_subdirectory(gradient)
add_subdirectory(cublas)
add_subdirectory(ozaki)
add_subdirectory(rdm)
add_subdirectory(energy)
add_subdirectory(df_integrals)
add_subdirectory(cli)
# ../apps/ is outside this directory, so each needs an explicit binary dir; the
# binaries themselves still land in NEVPT2_SRC_BINARY_DIR (RUNTIME_OUTPUT_DIRECTORY).
set(_nevpt2_apps "${CMAKE_CURRENT_SOURCE_DIR}/../apps")
add_subdirectory(${_nevpt2_apps}/integral_direct  apps/integral_direct)
add_subdirectory(${_nevpt2_apps}/density_fit      apps/density_fit)
add_subdirectory(${_nevpt2_apps}/sc_pseudodensity apps/sc_pseudodensity)
add_subdirectory(${_nevpt2_apps}/sanitizer_canary apps/sanitizer_canary)
