# Host-only suites for nevpt2.gradient_plan (src/gradient/gradient_plan.cppm)
# -- the gradient path's pure host arithmetic, extracted from nevpt2.gradient's
# internal partitions the way nevpt2.rdm_plan was from the RDM build: the
# promotion table (which sidecar array each slot of each block promotes into,
# over what span), the eight classes' Delta vectors and the two symmetrized
# classes' slab multipliers, termHasActive, and the CI source term's host
# folding -- and longestFourfoldRun, the step-ladder rule every finite-difference
# check of the gradient path gates on. No device, no kernel, no golden: NOT
# REQUIRES_GPU, so these run on
# a box with no card visible. The source term those pieces assemble is a
# numerical claim and stays with the golden tier's `gpu` duality entries.
add_gtest_executable(
  NAME gradient_plan_tests
  SOURCES gradient_plan_tests.cpp
  LINK nevpt2::gradient_plan)

add_gtest_suite_tests(
  TARGET gradient_plan_tests
  SUITES RotBlockTableTests GeneralPickTests PromotionPlanTests TermClassificationTests
         DeltaTests SlabMultiplierTests CiSourceHostTests FourfoldRunTests)

# The abort tier: a promotion of a slot that is not a two-electron one, and a
# shape mismatch in ciSourceAddInto. EXPECT_DEATH forks per assertion, hence
# its own `death` label.
add_gtest_suite_tests(
  TARGET gradient_plan_tests
  SUITES GradientPlanDeathTest
  DEATH)
