# Suites for nevpt2.gradient_plan (src/gradient/gradient_plan.cppm) -- the
# gradient path's pure host arithmetic, extracted from nevpt2.gradient's
# internal partitions the way nevpt2.rdm_plan was from the RDM build: the
# promotion table (which sidecar array each slot of each block promotes into,
# over what span), the eight classes' Delta vectors and the two symmetrized
# classes' slab multipliers, termHasActive, and the CI source term's host
# folding, h^eff's core-dressing terms (heffDressingPlan: the exchange pair
# that must not collapse, #64, checked over the CAS(4,4) gradient sidecar),
# the orbital energies' CI response (#87: W out of the sidecar, its forward
# and reverse uses, and each class's Delta transposed onto dE/de) --
# and longestFourfoldRun, the step-ladder rule every finite-difference
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

# The committed gradient SIDECARS: CAS(4,4)'s, which HeffDressingTests (#64)
# evaluates the core dressing's terms over on the host, and all three, which
# OrbitalEnergyResponseTests (#87) rebuild their goldens' e_core / e_virt from
# -- read only, still no device. LiF's golden comes along for the last.
foreach(_gg n2_ccpvdz_cas44:CAS44 n2_ccpvdz_cas88:CAS88 lif_def2svp_cas46:LIF)
  string(REPLACE ":" ";" _pair "${_gg}")
  list(GET _pair 0 _name)
  list(GET _pair 1 _upper)
  set(_gg_path "${PROJECT_SOURCE_DIR}/golden/${_name}.gradgold")
  if(NOT EXISTS "${_gg_path}")
    message(FATAL_ERROR "test/gradient_plan: committed gradient sidecar missing: ${_gg_path}")
  endif()
  target_compile_definitions(gradient_plan_tests
                             PRIVATE NEVPT2_TEST_GRADGOLD_${_upper}="${_gg_path}")
  unset(_gg_path)
  unset(_pair)
  unset(_name)
  unset(_upper)
endforeach()
set(_lif_path "${PROJECT_SOURCE_DIR}/golden/lif_def2svp_cas46.nevpt2gold")
if(NOT EXISTS "${_lif_path}")
  message(FATAL_ERROR "test/gradient_plan: committed golden missing: ${_lif_path}")
endif()
target_compile_definitions(gradient_plan_tests PRIVATE NEVPT2_TEST_GOLDEN_LIF="${_lif_path}")
unset(_lif_path)

add_gtest_suite_tests(
  TARGET gradient_plan_tests
  SUITES RotBlockTableTests GeneralPickTests PromotionPlanTests HeffDressingTests
         TermClassificationTests DeltaTests DeltaAdjointTests SlabMultiplierTests
         MixedTermsTests PairTermsTests GuardedCensusTests SijrsFinishTests
         CiSourceHostTests CiSourceFoldTests CiSourceFoldNegativeControlTests
         OrbitalEnergyResponseTests FourfoldRunTests)

# ciSourceVector launches the sigma kernels on the process's one
# DeviceResources (nevpt2.test.shared_resources).
add_gtest_suite_tests(
  TARGET gradient_plan_tests
  SUITES CiSourceVectorTests
  REQUIRES_GPU)

# The abort tier: a promotion of a slot that is not a two-electron one, a
# shape mismatch in ciSourceAddInto, and the orbital-energy response's shape
# checks (#87). EXPECT_DEATH forks per assertion, hence
# its own `death` label.
add_gtest_suite_tests(
  TARGET gradient_plan_tests
  SUITES GradientPlanDeathTest
  DEATH)
