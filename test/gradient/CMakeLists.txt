# Host-only suites for nevpt2.sc_amplitudes (src/gradient/sc_amplitudes.cppm)
# -- the strongly-contracted amplitude T, the gradient multipliers P / Q / R
# (Park Eqs. 12, 36-38) and the einsum-subscript rewriting that turns one of
# nevpt2.energy's class terms into its own derivative. No device, no kernel, no
# golden: NOT REQUIRES_GPU, so these run on a box with no card visible.
#
# What they prove is logic and arithmetic on hand cases -- that the amplitude
# is the reciprocal of normToEnergy's denominator, that the multipliers are its
# derivatives, that P_k N_k + Q_k H_k == T_k N_k, and that a rewritten
# derivative string is the one the eight classes need. The pseudodensities
# assembled from them are a numerical claim and stay with the golden tier: the
# `nevpt2_sc_pseudodensity_*` entries in the root CMakeLists.txt, which check
# Park Eq. 40 on a card at CAS(4,4), CAS(8,8) and CAS(10,10).
#
# nevpt2::gradient itself is NOT linked here: it is the device half, and
# nothing it does can be checked without a card.
add_gtest_executable(
  NAME gradient_tests
  SOURCES gradient_tests.cpp
  LINK nevpt2::sc_amplitudes)

add_gtest_suite_tests(
  TARGET gradient_tests
  SUITES ScMultipliersTests DotHostTests DerivativeSubscriptsTests)

# The abort tier: a mismatched length, a slot that is not an operand, a
# malformed subscript string, and the one operand whose derivative an einsum
# output cannot spell. EXPECT_DEATH forks per assertion, hence its own `death`
# label.
add_gtest_suite_tests(
  TARGET gradient_tests
  SUITES ScAmplitudesDeathTest
  DEATH)
