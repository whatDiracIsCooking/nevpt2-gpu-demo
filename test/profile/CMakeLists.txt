# Suites for nevpt2.profile (src/profile/): time() off and on, Section
# nesting, report()'s grouping and setEnabled(false). A recorded span is an
# event pair on the one stream nevpt2.test.shared_resources holds, so
# REQUIRES_GPU: labelled `gpu`, and with no card visible they FAIL (they do
# not skip). The binary's drift guard is `unit` only and runs card-free.
add_gtest_executable(
  NAME profile_tests
  SOURCES profile_tests.cpp
  LINK nevpt2::profile nevpt2::test::shared_resources)

add_gtest_suite_tests(
  TARGET profile_tests
  SUITES ProfileDisabledTests ProfileSectionTests ProfileReportTests ProfileSetEnabledTests
  REQUIRES_GPU)

# The abort tier: a span with no Section open is check()'s. EXPECT_DEATH forks
# per assertion, hence `death`; the child brings the device up to get a
# stream, hence REQUIRES_GPU too.
add_gtest_suite_tests(
  TARGET profile_tests
  SUITES ProfileDeathTest
  REQUIRES_GPU
  DEATH)
