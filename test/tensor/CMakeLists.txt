# Host-only suites for nevpt2.tensor's host Tensor (src/tensor/tensor.cppm):
# fromFlat's row-major shape and strides, operator() / flat agreeing on one
# offset, rank / dim / size -- and fromFlat with a size that does not match its
# shape, which is our bug and aborts (the abort tier). No device, no kernel, no
# golden: NOT REQUIRES_GPU, so these run on a box with no card visible.
add_gtest_executable(
  NAME tensor_tests
  SOURCES tensor_tests.cpp
  LINK nevpt2::tensor)

add_gtest_suite_tests(
  TARGET tensor_tests
  SUITES TensorShapeTests TensorIndexTests)

# The abort tier: EXPECT_DEATH forks per assertion, hence its own `death` label.
add_gtest_suite_tests(
  TARGET tensor_tests
  SUITES TensorDeathTest
  DEATH)
