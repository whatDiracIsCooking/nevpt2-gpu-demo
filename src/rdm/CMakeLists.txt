# The RDM build: host-side link tables, and the linked-in device libraries the
# tiled dm3/f3ac/f3ca build launches -- the RDM-build kernels (kernels.cu, via
# rdm_launch.cu), the accumulate kernel, and the f3 digest and scatter.

# nevpt2.link_tables -- Knowles-Handy link tables. Pure host integer
# combinatorics, no GPU.
add_cxx_module_library(
  NAME nevpt2.link_tables
  PRIMARY_INTERFACE link_tables.cppm
  PARTITIONS link_tables_binomial.cppm
  IMPLEMENTATION link_tables.cpp
  LINK_PUBLIC nevpt2::common
)

# nevpt2.rdm_plan -- the RDM build's pure host index arithmetic: the tile plan
# (tileWidthForCount, planTiles), the consume GEMM's chunking (evenChunks) and
# its permuted integral copy (permuteEriConsume). No GPU, so the unit tier
# tests it card-free (test/rdm_plan/); nevpt2.rdm_build imports it.
add_cxx_module_library(
  NAME nevpt2.rdm_plan
  PRIMARY_INTERFACE rdm_plan.cppm
  LINK_PUBLIC nevpt2::common
)

# nevpt2.rdm_build -- the tiled, device-resident dm3/f3ac/f3ca build both
# demos run (it touches only the ACTIVE h2e, so density fitting does not
# change it). The --cublas digest lives in ../cublas and is linked privately:
# real on CUDA, refusing stubs on HIP. wwr::blas (cuBLAS / hipBLAS) is the
# backend-neutral consume GEMM (the default; --consume-emitted for the A/B).
add_cxx_module_library(
  NAME nevpt2.rdm_build
  PRIMARY_INTERFACE rdm_build.cppm
  PARTITIONS rdm_build_staging.cppm rdm_build_blas.cppm
  IMPLEMENTATION rdm_build.cpp
  # device_tensor (re-exports nevpt2.tensor and nevpt2.device_resources): the result
  # is DeviceTensors.
  LINK_PUBLIC nevpt2::wwr nevpt2::device_tensor
  # wwr::extension::blas: the wwrblasStatus_t specializations that let
  # nevpt2::gpuCheck check a BLAS status.
  LINK_PRIVATE nevpt2::common nevpt2::link_tables nevpt2::rdm_plan nevpt2::profile
    nevpt2::cublas_emul wwr::blas wwr::extension::blas
)

# kernels.cu -- the 8 CAS-independent generic RDM-build kernels. LINKED IN,
# like rdm_accumulate.cu below, but kernels.cu is not itself a source:
# rdm_launch.cu #includes it and holds the launchers, because a __global__
# cannot be launched from another TU (the device library builds with separable
# compilation off). rdm_build.cpp calls them through rdm_kernels_bridge.h.
# No parallel_for (the launchers keep their own <<<>>>; see rdm_launch.cu), so
# it links wwr::device for runtime.h directly.
add_device_library(
  NAME nevpt2.rdm.kernels.device
  SOURCES rdm_launch.cu
  LINK_PRIVATE wwr::device nevpt2::common::device
)
target_include_directories(nevpt2.rdm.kernels.device PRIVATE ${NEVPT2_SRC_ROOT})

# rdm_accumulate.cu -- the device-resident launcher's on-device tile-partial
# accumulate. Its own file, like every kernel added after kernels.cu.
#
# LINKED in, like every kernel: add_device_library, over WarpWraps' wwr_add_gpu_device_library (a STATIC library of device-compiled
# .cu, CUDA language under CUDA, CXX + hip::device's -x hip under HIP), called
# through rdm_accumulate_bridge.h. wwr::extension::parallel_for brings the
# launcher and WarpWraps' src/ include root; nevpt2::common::device the src/
# root, for "rdm/rdm_accumulate_bridge.h" and the "wwr/wwr.h" it includes.
add_device_library(
  NAME nevpt2.rdm.accumulate.device
  SOURCES rdm_accumulate.cu
  LINK_PRIVATE wwr::extension::parallel_for nevpt2::common::device
)

# f3_digest.cu -- the emitted path's f3 digest GEMM, reading L2 directly
# instead of the R2 copy kernels.cu's digests once read. Its own
# <<<>>> (the 16x16 tile is baked into the kernel), so wwr::device rather than
# parallel_for.
add_device_library(
  NAME nevpt2.rdm.f3_digest.device
  SOURCES f3_digest.cu
  LINK_PRIVATE wwr::device nevpt2::common::device
)
target_include_directories(nevpt2.rdm.f3_digest.device PRIVATE ${NEVPT2_SRC_ROOT})

# f3_scatter.cu -- the f3 permute-scatter both GEMM digests (--cublas and the
# native-fp64 --blas-digest; HIP's default) fold their (n^4, n^2)
# f3 result back with. Backend-neutral and built on both backends; it lived in
# ../cublas until the BLAS digest made it more than a --cublas helper.
add_device_library(
  NAME nevpt2.rdm.f3_scatter.device
  SOURCES f3_scatter.cu
  LINK_PRIVATE wwr::extension::parallel_for nevpt2::common::device
)
target_include_directories(nevpt2.rdm.f3_scatter.device PRIVATE ${NEVPT2_SRC_ROOT})

# rdm_build.cpp includes the bridges (after its imports, in extern "C++";
# runtime.h in its global module fragment), so it needs
# the same include root, plus wwr::backend for the WWR_GPU_BACKEND_* define
# selected_backend.h reads when there is no device-compile pass to go by.
target_include_directories(nevpt2.rdm_build PRIVATE ${NEVPT2_SRC_ROOT})
target_link_libraries(nevpt2.rdm_build PRIVATE nevpt2.rdm.kernels.device
  nevpt2.rdm.accumulate.device nevpt2.rdm.f3_digest.device
  nevpt2.rdm.f3_scatter.device nevpt2.ozaki.device wwr::backend)
