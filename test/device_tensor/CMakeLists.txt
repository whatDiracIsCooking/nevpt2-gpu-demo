# The first GPU suites: nevpt2.device_tensor's upload/download
# round trip and DeviceBuffer's zero fill, on the one DeviceResources
# nevpt2.test.shared_resources holds for the process. They allocate device
# memory and copy on the stream, so REQUIRES_GPU: labelled `gpu`, and on a box
# with no card visible they FAIL with DeviceResources::create's message (they
# do not skip). The binary's drift guard is `unit` only and runs card-free.
add_gtest_executable(
  NAME device_tensor_tests
  SOURCES device_tensor_tests.cpp
  LINK nevpt2::device_tensor nevpt2::test::shared_resources)

add_gtest_suite_tests(
  TARGET device_tensor_tests
  SUITES DeviceTensorRoundTripTests DeviceBufferZeroFillTests
  REQUIRES_GPU)
