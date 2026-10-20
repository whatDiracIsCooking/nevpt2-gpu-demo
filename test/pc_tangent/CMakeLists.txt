# Host-only suite for nevpt2.pc_tangent's report (src/gradient/pc_tangent.cpp)
# -- the gate printPcTangentReport applies to a PcTangentCheck. No device, no
# kernel, no golden: NOT REQUIRES_GPU, so it runs on a box with no card
# visible (the module links the RDM build, but nothing this suite calls
# touches it).
#
# What it proves is which combinations the gate accepts, on made-up numbers --
# above all that a class REFUSED by the gap or denominator check fails the gate
# instead of being read as a zero derivative, which is the one thing about the
# tangent that must never be quiet -- and that the finite difference is gated
# on a RUN of fourfold halvings with the conditioning guard recorded at every
# displaced state (PcTangentLadderTests). The numerical claim -- the analytic
# tangent against a Richardson-verified finite difference of a real state --
# is the golden tier's: the `nevpt2_pc_tangent_*` entries in the root
# CMakeLists.txt, which need a card.
add_gtest_executable(
  NAME pc_tangent_tests
  SOURCES pc_tangent_tests.cpp
  LINK nevpt2::pc_tangent)

add_gtest_suite_tests(
  TARGET pc_tangent_tests
  SUITES PcTangentReportTests PcTangentLadderTests PcHessianReportTests)
