# Suites for nevpt2.gradient_plan (src/gradient/gradient_plan.cppm) -- the
# gradient path's pure host arithmetic, extracted from nevpt2.gradient's
# internal partitions the way nevpt2.rdm_plan was from the RDM build: the
# promotion table (which sidecar array each slot of each block promotes into,
# over what span), the eight classes' Delta vectors and the two symmetrized
# classes' slab multipliers, termHasActive, and the CI source term's host
# folding -- and longestFourfoldRun, the step-ladder rule every finite-difference
# check of the gradient path gates on.
#
# The CI source fold (#63) is checked here against an UNFOLDED host reference:
# dm1/dm2/dm3 rebuilt on the host from two committed goldens' CI vectors over
# nevpt2.rdm_host_dm's applyAllE, so these suites read CAS(4,4) and CAS(8,8)
# goldens (read only) -- still no device, no kernel: NOT REQUIRES_GPU, so they
# run on a box with no card visible. Their negative controls assert that the
# same comparison FAILS for a perturbed fold.
#
# The one exception is CiSourceVectorTests, REQUIRES_GPU: nevpt2.gradient's
# ciSourceVector launches the sigma kernels, so its shape, its accumulation and
# its y_I against the unfolded host derivative need a card (nevpt2::gradient
# and the shared DeviceResources are linked for it alone). The source term
# those pieces assemble is a numerical claim and stays with the golden tier's
# `gpu` duality entries.
add_gtest_executable(
  NAME gradient_plan_tests
  SOURCES gradient_plan_tests.cpp
  LINK nevpt2::gradient_plan nevpt2::gradient nevpt2::rdm_host_dm nevpt2::golden
    nevpt2::test::shared_resources)

# The committed goldens the fold suites read, passed in rather than hard-coded
# in the source.
foreach(_case cas44 cas88)
  set(_path "${PROJECT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.nevpt2gold")
  if(NOT EXISTS "${_path}")
    message(FATAL_ERROR "test/gradient_plan: committed golden missing: ${_path}")
  endif()
  string(TOUPPER "${_case}" _upper)
  target_compile_definitions(gradient_plan_tests
                             PRIVATE NEVPT2_TEST_GOLDEN_${_upper}="${_path}")
  unset(_path)
  unset(_upper)
endforeach()

add_gtest_suite_tests(
  TARGET gradient_plan_tests
  SUITES RotBlockTableTests GeneralPickTests PromotionPlanTests TermClassificationTests
         DeltaTests SlabMultiplierTests CiSourceHostTests CiSourceFoldTests
         CiSourceFoldNegativeControlTests FourfoldRunTests)

# ciSourceVector launches the sigma kernels on the process's one
# DeviceResources (nevpt2.test.shared_resources).
add_gtest_suite_tests(
  TARGET gradient_plan_tests
  SUITES CiSourceVectorTests
  REQUIRES_GPU)

# The abort tier: a promotion of a slot that is not a two-electron one, and a
# shape mismatch in ciSourceAddInto. EXPECT_DEATH forks per assertion, hence
# its own `death` label.
add_gtest_suite_tests(
  TARGET gradient_plan_tests
  SUITES GradientPlanDeathTest
  DEATH)
