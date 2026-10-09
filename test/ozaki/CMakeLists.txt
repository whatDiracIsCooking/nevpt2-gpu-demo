# The --ozaki digest GEMM (src/ozaki/ozaki_digest.cu), called through its
# bridge (ozaki_digest_bridge.h) on seeded random operands against a
# long-double host reference of the bridge's contract, within an error bound
# derived from the scheme (the test file's comment). Padding, pair
# truncation, K chunking past kOzakiMaxK, beta, per-row scaling and the
# scratch size. Built and run on both backends: the kernel goes through
# WarpWraps' <wmma.h>.
#
# No .cu here: the binary links src/ozaki's device library, built with
# separable compilation OFF, so this CXX executable needs no device-link step.
# The test TU includes the bridge, so it needs the src/ root (for
# "ozaki/ozaki_digest_bridge.h" and the "wwr/wwr.h" it includes) and
# wwr::backend (runtime.h's include path and the WWR_GPU_BACKEND_* define),
# as test/rdm's rdm_kernels_tests does.
#
# REQUIRES_GPU: every suite but OzakiScratchBytesTests' first case launches;
# the drift guard is `unit` only and runs card-free.
add_gtest_executable(
  NAME ozaki_digest_tests
  SOURCES ozaki_digest_tests.cpp
  LINK nevpt2::wwr nevpt2::test::shared_resources nevpt2.ozaki.device wwr::backend)
target_include_directories(ozaki_digest_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")

add_gtest_suite_tests(
  TARGET ozaki_digest_tests
  SUITES OzakiAllPairsTests OzakiPairTruncationTests OzakiKChunkingTests OzakiBetaTests
    OzakiRowScalingTests OzakiScratchBytesTests
  REQUIRES_GPU)
