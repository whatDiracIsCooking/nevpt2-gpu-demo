# Host-only suites for nevpt2.gradsrc (apps/sc_pseudodensity/gradsrc.cppm):
# the `.gradsrc` writer nevpt2_sc_pseudodensity --dump-source-terms uses, round
# tripped through its own reader and read byte by byte by an independent parser
# in the test (a stored zip of NPY v1.0 members, the layout numpy.load reads).
# Files go to a per-test temp directory. No device: NOT REQUIRES_GPU, so these
# run with no card visible. Nothing here checks a derivative.
add_gtest_executable(
  NAME gradsrc_tests
  SOURCES gradsrc_tests.cpp
  LINK nevpt2::gradsrc)

add_gtest_suite_tests(
  TARGET gradsrc_tests
  SUITES GradSrcRoundTripTests GradSrcFormatTests GradSrcErrorTests)
