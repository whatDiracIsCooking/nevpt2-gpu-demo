# Host-only suites for nevpt2.energy_finish (src/energy/energy_finish.cppm) --
# the class energies' pure host arithmetic: the slab walk (forBatches), the SC
# finish (normToEnergy, against PySCF's _norm_to_energy on a hand case) and
# ITS DERIVATIVE (normToEnergyDerivative / normToEnergyTangent, the forward
# tangent's only calculus -- Park Eqs. 36-37, their Eq. 40 identity, and
# Richardson tables against a central difference of the finish itself -- and
# normToEnergyDeltaDerivative / normToEnergyDeltaTangent, its derivative in
# Delta, Park Eq. 38, for the orbital energies' response, #87) and ITS SECOND
# DERIVATIVE (normToEnergyHessian in the d form and the multiplier tangents,
# #150: docs/hessian-theory.md Sec. 2's exact zeros and structural relations,
# N = 0 giving zero, Richardson tables against differences of P, Q, R), the
# PC rank cut and gap decision (pcRankCut, #94), the PC finish and denominator
# check (finishSingle, checkDenominator) and the PC
# TANGENT's per-tuple arithmetic (pcCoefficients, pcQuadForm, tangentSingle),
# and the PC ADJOINT's (#74: pcAdjointSingle / pcAdjointPair, the forward
# rotation tangent pcRotationTangent*, pcDeltaTangentSingle), each against an
# independent host energy with the full matrix inverted.
# Logic on hand cases, never an energy of a real state. No device, no kernel,
# no golden: NOT REQUIRES_GPU, so these run on a box with no card visible.
add_gtest_executable(
  NAME energy_finish_tests
  SOURCES energy_finish_tests.cpp
  LINK nevpt2::energy_finish)

add_gtest_suite_tests(
  TARGET energy_finish_tests
  SUITES ForBatchesTests NormToEnergyTests CheckDenominatorTests PcRankCutTests FinishSingleTests
         NormToEnergyDerivativeTests NormToEnergyTangentTests NormToEnergyDeltaTests
         PcCoefficientsTests PcQuadFormTests TangentSingleTests PcAdjointTests
         PcRotationTangentTests PcDeltaTangentTests NormToEnergyHessianTests
         NormToEnergyMultiplierTangentTests)

# The abort tier: normToEnergyTangent's five vectors (and the Delta and
# multiplier tangents' four and six) must be one class's, so differing lengths
# are check(). EXPECT_DEATH forks per assertion, hence its own `death` label.
add_gtest_suite_tests(
  TARGET energy_finish_tests
  SUITES NormToEnergyTangentDeathTest NormToEnergyDeltaDeathTest
         NormToEnergyMultiplierTangentDeathTest
  DEATH)
