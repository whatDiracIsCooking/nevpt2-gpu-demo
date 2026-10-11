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

# The dm3 build in kernels.cu (produce_generic + digest_dm3_generic, through
# rdm_launch.cu), driven end to end by nevpt2.rdm_build's buildRdmsDevice on a
# seeded random CI vector at CAS(4,4) and CAS(6,6): identities that hold for
# any N-electron state (hermiticity, pair-swap commutators and the
# normal-ordered pair symmetry, partial traces against a host dm1/dm2 built
# from nevpt2.link_tables) and tiling invariance. Logic, not energies.
#
# Its own binary: it imports nevpt2.rdm_build, which links every RDM device
# library itself, rather than calling a bridge directly. REQUIRES_GPU: every
# case but the host-reference sanity check builds dm3 on the card.
add_gtest_executable(
  NAME rdm_dm3_tests
  SOURCES rdm_dm3_tests.cpp
  LINK nevpt2::rdm_build nevpt2::link_tables nevpt2::test::shared_resources)

add_gtest_suite_tests(
  TARGET rdm_dm3_tests
  SUITES Dm3HermiticityTests Dm3PairSymmetryTests Dm3PartialTraceTests
    Dm3TilingInvarianceTests
  REQUIRES_GPU)

# The TANGENT build (nevpt2.rdm_build's :tangent partition,
# src/rdm/rdm_build_tangent.cppm): buildRdmTangentsDevice against a central
# finite difference of buildRdmsDevice itself, on a seeded random state and a
# seeded random direction at CAS(4,4) and CAS(8,8) -- along the straight line
# (exact, the forms are quadratic) and Richardson-verified along the
# normalized path. Logic, not energies: no golden file is read and the h2e is
# seeded random.
#
# Its own binary, like rdm_dm3_tests: it imports nevpt2.rdm_build, which links
# every RDM device library itself. REQUIRES_GPU -- every case builds on the
# card.
add_gtest_executable(
  NAME rdm_tangent_tests
  SOURCES rdm_tangent_tests.cpp
  LINK nevpt2::rdm_build nevpt2::link_tables nevpt2::test::shared_resources)

# The CAS(8,8) entries build 9 times each (the tangent plus 4 central
# differences), so the Richardson suite is the heaviest here: ~1 s in the
# Release build and well inside 120 s x the sanitizer multiplier in a Debug
# one under compute-sanitizer.
add_gtest_suite_tests(
  TARGET rdm_tangent_tests
  SUITES RdmTangentDifferenceTests RdmTangentRichardsonTests RdmTangentEulerTests
    RdmTangentPathTests
  REQUIRES_GPU)
