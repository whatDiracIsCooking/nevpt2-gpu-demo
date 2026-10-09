# Host-only suites for nevpt2.golden: loadGolden's refusals on a
# missing, truncated or wrongly-tagged file (temp files the test writes) come
# back as an IO Result, never an abort; GoldenFile::require names a missing
# array; and the committed CAS(4,4) golden parses with the expected header and
# array shapes. Pure host file parsing, no device: NOT REQUIRES_GPU, so these
# run with no card visible. Nothing here checks an energy.
add_gtest_executable(
  NAME golden_tests
  SOURCES golden_tests.cpp
  LINK nevpt2::golden)

# The committed golden the happy path reads, passed in rather than hard-coded
# in the source. Read only, never written.
set(_golden_cas44 "${PROJECT_SOURCE_DIR}/golden/n2_ccpvdz_cas44.nevpt2gold")
if(NOT EXISTS "${_golden_cas44}")
  message(FATAL_ERROR "test/golden: committed golden missing: ${_golden_cas44}")
endif()
target_compile_definitions(golden_tests PRIVATE NEVPT2_TEST_GOLDEN_CAS44="${_golden_cas44}")
unset(_golden_cas44)

add_gtest_suite_tests(
  TARGET golden_tests
  SUITES GoldenLoadErrorTests GoldenRequireTests GoldenCas44Tests)
