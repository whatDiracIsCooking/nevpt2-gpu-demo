# Host-only suites for nevpt2.error_handling -- the value tier (Error, kindName,
# err_* -> Result / Status, NEVPT2_TRY, report) and the abort tier as narrowTo
# reaches it -- plus nevpt2.common's idivup / align_up, the integer helpers
# narrowTo sits beside. No device, no kernel, no golden: NOT REQUIRES_GPU, so
# these run on a box with no card visible.
add_gtest_executable(
  NAME error_handling_tests
  SOURCES error_handling_tests.cpp
  LINK nevpt2::error_handling nevpt2::common)

add_gtest_suite_tests(
  TARGET error_handling_tests
  SUITES ErrorKindTests ErrorValueTests TryMacroTests ReportTests CommonIntegerTests)

# The abort tier: EXPECT_DEATH forks per assertion, hence its own `death` label.
add_gtest_suite_tests(
  TARGET error_handling_tests
  SUITES AbortTierDeathTest
  DEATH)
