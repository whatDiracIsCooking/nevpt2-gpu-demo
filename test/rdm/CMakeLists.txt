# GPU suites for the RDM build's linked-in device libraries:
# accumulateInplace (rdm_accumulate_bridge.h), f3ScatterCa (f3_scatter_bridge.h)
# and f3DigestAccumulate (f3_digest_bridge.h), each on seeded random small
# inputs (norb 4..6) against a host loop written from the bridge's contract.
# The kernels, not PySCF agreement: energies stay with the golden demos.
#
# No .cu here: the binary links src/rdm's own device libraries, which are
# built with separable compilation OFF, so this CXX executable needs no
# device-link step. The test TU includes the bridges, so it needs the src/
# root (for "rdm/*_bridge.h" and the "wwr/wwr.h" they include) and
# wwr::backend (runtime.h's include path and the WWR_GPU_BACKEND_* define
# selected_backend.h reads in a host-only pass), as nevpt2.rdm_build does.
#
# REQUIRES_GPU: every case launches a kernel; the drift guard is `unit` only
# and runs card-free.
add_gtest_executable(
  NAME rdm_kernels_tests
  SOURCES rdm_kernels_tests.cpp
  LINK nevpt2::wwr nevpt2::test::shared_resources
    nevpt2.rdm.accumulate.device nevpt2.rdm.f3_scatter.device nevpt2.rdm.f3_digest.device
    wwr::backend)
target_include_directories(rdm_kernels_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")

add_gtest_suite_tests(
  TARGET rdm_kernels_tests
  SUITES AccumulateInplaceTests F3ScatterCaTests F3DigestAccumulateTests
  REQUIRES_GPU)
