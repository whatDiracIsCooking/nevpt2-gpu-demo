# Host-only suites for nevpt2.source_compare (apps/sc_pseudodensity/
# source_compare.cppm): --source-oracle's block-by-block comparison, held to
# every committed source-term oracle (golden/*.srcgold, SC #92 and PC #95)
# with no card. The oracle compared with itself passes, a copy with one block
# perturbed fails naming that block, the known-gap list's pinning works both
# ways, and a Y carried through oracleBasisA lands back on the oracle's A.
# Until #74 there are no GPU PC source terms, so this is the PC oracle's whole
# gate (docs/testing.md, "The PC source terms' oracle"). No device: NOT
# REQUIRES_GPU. Nothing here checks a derivative -- the oracle IS the
# derivative, and it is only compared with itself.
add_gtest_executable(
  NAME source_compare_tests
  SOURCES source_compare_tests.cpp
  LINK nevpt2::source_compare)

# Every oracle in golden/ at configure time, '|'-joined like test/golden's.
file(GLOB _source_compare_oracles CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/golden/*.srcgold")
list(SORT _source_compare_oracles)
list(JOIN _source_compare_oracles "|" _source_compare_oracles)
target_compile_definitions(
  source_compare_tests PRIVATE NEVPT2_TEST_COMMITTED_SRCGOLDS="${_source_compare_oracles}")
unset(_source_compare_oracles)

add_gtest_suite_tests(
  TARGET source_compare_tests
  SUITES SourceCompareSelfTests SourceCompareKnownGapTests SourceCompareBasisTests)
