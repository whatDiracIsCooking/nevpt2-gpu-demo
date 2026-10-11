# The RDM build: host-side link tables, and the linked-in device libraries the
# tiled dm3/f3ac/f3ca build launches -- the RDM-build kernels (kernels.cu, via
# rdm_launch.cu), the accumulate kernel, and the f3 digest and scatter. Plus
# the things built on the same link tables that the build itself does not use:
# the host dm1/dm2 (nevpt2.rdm_host_dm) and the sigma-vector kernels
# (nevpt2.sigma_vector, sigma_vector.cu).

# nevpt2.link_tables -- Knowles-Handy link tables. Pure host integer
# combinatorics, no GPU.
add_cxx_module_library(
  NAME nevpt2.link_tables
  PRIMARY_INTERFACE link_tables.cppm
  PARTITIONS link_tables_binomial.cppm
  IMPLEMENTATION link_tables.cpp
  LINK_PUBLIC nevpt2::common
)

# nevpt2.rdm_host_dm -- the active-space dm1/dm2 rebuilt on the HOST from a CI
# vector over the link tables above, plus their tangents along a CI direction.
# No GPU and no kernel: it is the reference the golden file's dm1/dm2 (PySCF
# inputs, never computed here) are checked against, and the differentiable
# bilinear forms the gradient work needs. Neither demo links it; the unit tier
# tests it card-free (test/rdm_host_dm/).
add_cxx_module_library(
  NAME nevpt2.rdm_host_dm
  PRIMARY_INTERFACE rdm_host_dm.cppm
  IMPLEMENTATION rdm_host_dm.cpp
  # tensor (re-exports nevpt2.common): dm1/dm2 are Tensors. error_handling:
  # create() hands back a Result, and the shape checks abort through check().
  LINK_PUBLIC nevpt2::tensor nevpt2::error_handling
  LINK_PRIVATE nevpt2::link_tables
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
# rdm_build_tangent.cppm is the :tangent partition: the same pipeline run with
# one copy of the CI vector replaced by a direction (buildRdmTangentsDevice,
# and the --rdm-tangent self-check), which needs no kernel of its own.
add_cxx_module_library(
  NAME nevpt2.rdm_build
  PRIMARY_INTERFACE rdm_build.cppm
  PARTITIONS rdm_build_staging.cppm rdm_build_blas.cppm rdm_build_tangent.cppm
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

# sigma_vector.cu -- the 3-body and 4-body-contracted sigma-vector kernels: a
# coefficient tensor over active indices contracted with the excited
# wavefunctions it multiplies, one amplitude per determinant, so that the n^6 /
# n^8 RDM derivative never exists. Its own file and its own device library,
# like every kernel added after kernels.cu. Backend-neutral (one thread per
# output determinant, no atomic and no shuffle), so wwr::device for runtime.h
# rather than parallel_for -- the launchers keep their own <<<>>>.
add_device_library(
  NAME nevpt2.rdm.sigma_vector.device
  SOURCES sigma_vector.cu
  LINK_PRIVATE wwr::device nevpt2::common::device
)
target_include_directories(nevpt2.rdm.sigma_vector.device PRIVATE ${NEVPT2_SRC_ROOT})

# nevpt2.sigma_vector -- the host side of those kernels: the reverse link
# tables, the coefficient repacking each kernel reads, the tile plan and the
# launches. Neither demo links it yet (the analytic-gradient work is what
# contracts a pseudodensity through it); the unit tier drives it against an
# explicit host construction and against the device build's own dm3/f3
# (test/sigma_vector/).
add_cxx_module_library(
  NAME nevpt2.sigma_vector
  PRIMARY_INTERFACE sigma_vector.cppm
  IMPLEMENTATION sigma_vector.cpp
  # device_tensor (re-exports nevpt2.tensor and nevpt2.device_resources): the
  # inputs and the result are Tensors, allocated from the caller's pool.
  LINK_PUBLIC nevpt2::wwr nevpt2::device_tensor
  LINK_PRIVATE nevpt2::common nevpt2::error_handling nevpt2::link_tables nevpt2::rdm_plan
)
# It includes sigma_vector_bridge.h after its imports, in extern "C++"
# (runtime.h in its global module fragment), so it needs the src/ include root
# and wwr::backend for the WWR_GPU_BACKEND_* define selected_backend.h reads
# when there is no device-compile pass to go by -- as nevpt2.rdm_build does.
target_include_directories(nevpt2.sigma_vector PRIVATE ${NEVPT2_SRC_ROOT})
target_link_libraries(nevpt2.sigma_vector PRIVATE nevpt2.rdm.sigma_vector.device wwr::backend)

# rdm_build.cpp includes the bridges (after its imports, in extern "C++";
# runtime.h in its global module fragment), so it needs
# the same include root, plus wwr::backend for the WWR_GPU_BACKEND_* define
# selected_backend.h reads when there is no device-compile pass to go by.
target_include_directories(nevpt2.rdm_build PRIVATE ${NEVPT2_SRC_ROOT})
target_link_libraries(nevpt2.rdm_build PRIVATE nevpt2.rdm.kernels.device
  nevpt2.rdm.accumulate.device nevpt2.rdm.f3_digest.device
  nevpt2.rdm.f3_scatter.device nevpt2.ozaki.device wwr::backend)
