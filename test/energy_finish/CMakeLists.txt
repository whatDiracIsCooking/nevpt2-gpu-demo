# Host-only suites for nevpt2.energy_finish (src/energy/energy_finish.cppm) --
# the class energies' pure host arithmetic: the slab walk (forBatches), the SC
# finish (normToEnergy, against PySCF's _norm_to_energy on a hand case) and
# the PC finish and denominator check (finishSingle, checkDenominator). Logic
# on hand cases, never an energy of a real state. No device, no kernel, no
# golden: NOT REQUIRES_GPU, so these run on a box with no card visible.
add_gtest_executable(
  NAME energy_finish_tests
  SOURCES energy_finish_tests.cpp
  LINK nevpt2::energy_finish)

add_gtest_suite_tests(
  TARGET energy_finish_tests
  SUITES ForBatchesTests NormToEnergyTests CheckDenominatorTests FinishSingleTests)
