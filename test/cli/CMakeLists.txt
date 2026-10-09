# Host-only suites for nevpt2.cli (src/cli/): the common flags each demo
# takes, their integer parsing, and the checks that span flags. Pure argument
# handling -- it links the device libraries through nevpt2.rdm_build but never
# calls the GPU runtime: NOT REQUIRES_GPU, so these run with no card visible.
add_gtest_executable(
  NAME cli_tests
  SOURCES cli_tests.cpp
  LINK nevpt2::cli)

add_gtest_suite_tests(
  TARGET cli_tests
  SUITES CliIntegerTests CliFlagTests CliFinalizeTests CliUsageTests)
