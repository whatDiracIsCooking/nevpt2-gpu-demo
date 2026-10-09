# Host-only suites for nevpt2.rdm_plan (src/rdm/rdm_plan.cppm) -- the RDM
# build's pure host index arithmetic: the determinant-axis tile plan
# (tileWidthForCount, planTiles), the consume GEMM's chunking (evenChunks) and
# its permuted integral copy (permuteEriConsume). No device, no kernel, no
# golden: NOT REQUIRES_GPU, so these run on a box with no card visible.
add_gtest_executable(
  NAME rdm_plan_tests
  SOURCES rdm_plan_tests.cpp
  LINK nevpt2::rdm_plan)

add_gtest_suite_tests(
  TARGET rdm_plan_tests
  SUITES TilePlanTests EvenChunksTests PermuteEriConsumeTests)

# The abort tier: a tile past int is narrowTo's check(); EXPECT_DEATH forks per
# assertion, hence its own `death` label.
add_gtest_suite_tests(
  TARGET rdm_plan_tests
  SUITES TilePlanDeathTest
  DEATH)
