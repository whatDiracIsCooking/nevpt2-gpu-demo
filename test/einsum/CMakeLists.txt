# GPU suites for nevpt2.einsum: deviceEinsumNew /
# deviceEinsumAccum / deviceTransposeNew / deviceTransposeAccum, each checked
# against a host loop over Tensor to a stated tolerance (einsum_tests.cpp's
# header), through the host module API only -- no .cu of its own. They upload
# operands and launch on the one DeviceResources nevpt2.test.shared_resources
# holds, so REQUIRES_GPU: labelled `gpu`, and with no card visible they FAIL
# (they do not skip). The binary's drift guard is `unit` only and runs
# card-free.
add_gtest_executable(
  NAME einsum_tests
  SOURCES einsum_tests.cpp
  LINK nevpt2::einsum nevpt2::test::shared_resources)

add_gtest_suite_tests(
  TARGET einsum_tests
  SUITES EinsumNewTests EinsumAccumTests TransposeAccumTests
  REQUIRES_GPU)

# The abort tier: a malformed or ill-fitting subscript string aborts through
# check(). EXPECT_DEATH forks per assertion, hence `death`; the child brings
# the device up to get a DeviceResources to pass, hence REQUIRES_GPU too.
add_gtest_suite_tests(
  TARGET einsum_tests
  SUITES EinsumDeathTest
  REQUIRES_GPU
  DEATH)
