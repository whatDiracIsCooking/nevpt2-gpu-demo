# Host-only suites for nevpt2.link_tables (src/rdm/link_tables.cppm) -- the
# determinant-string combinatorics the RDM build's link tables come from:
# num_strings / nlink, str2addr <-> addr2str, excit_sign, gen_linkstr_index and
# reverse_link, each checked exactly against a brute-force bit-twiddling
# reference. Pure host integer logic: NOT REQUIRES_GPU, so these run on a box
# with no card visible.
add_gtest_executable(
  NAME link_tables_tests
  SOURCES link_tables_tests.cpp
  LINK nevpt2::link_tables)

add_gtest_suite_tests(
  TARGET link_tables_tests
  SUITES NumStringsTests AddressingTests ExcitSignTests LinkTableTests ReverseLinkTests)
