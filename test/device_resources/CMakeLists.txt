# Suites for nevpt2.device_resources (src/device_resources/).
#
# pool_threshold_tests: host-only suites for the free helpers
# parsePoolThreshold and toMB. The binary links the GPU runtime through
# nevpt2::device_resources but never calls it (no DeviceResources is made):
# NOT REQUIRES_GPU, so these run with no card visible.
add_gtest_executable(
  NAME pool_threshold_tests
  SOURCES pool_threshold_tests.cpp
  LINK nevpt2::device_resources)

add_gtest_suite_tests(
  TARGET pool_threshold_tests
  SUITES PoolThresholdTests ToMBTests)

# device_resources_tests: DeviceResources itself -- the pool's high-water and
# release threshold, the stream's flags, the BLAS/solver handles' stream, and
# printPoolHighWater's two formats -- on the process's one DeviceResources
# (nevpt2.test.shared_resources). A sibling binary, not more suites in the one
# above, so that one stays card-free. REQUIRES_GPU: labelled `gpu`; with no
# card visible they FAIL with DeviceResources::create's message (they do not
# skip). The binary's drift guard is `unit` only and runs card-free.
add_gtest_executable(
  NAME device_resources_tests
  SOURCES device_resources_tests.cpp
  LINK nevpt2::test::shared_resources)

add_gtest_suite_tests(
  TARGET device_resources_tests
  SUITES PoolHighWaterTests ReleaseThresholdTests StreamBindingTests PrintPoolHighWaterTests
  REQUIRES_GPU)
