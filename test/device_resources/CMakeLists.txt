# Suites for nevpt2.device_resources (src/device_resources/).
#
# pool_threshold_tests: host-only suites for the free helpers
# parsePoolThreshold and toMB. The binary links the GPU runtime through
# nevpt2::device_resources but never calls it (no DeviceResources is made):
# NOT REQUIRES_GPU, so these run with no card visible. A GPU suite for
# DeviceResources itself belongs in a sibling binary here, REQUIRES_GPU and
# linking nevpt2::test::shared_resources, so the host-only one stays card-free.
add_gtest_executable(
  NAME pool_threshold_tests
  SOURCES pool_threshold_tests.cpp
  LINK nevpt2::device_resources)

add_gtest_suite_tests(
  TARGET pool_threshold_tests
  SUITES PoolThresholdTests ToMBTests)
