# GPU suites for nevpt2.df_integrals' DfIntegralSource: every external block's
# slab() over several [b0, b1) ranges, and activeH2e(), on seeded synthetic B
# blocks (naux 6, ncore 3, ncas 4, nvirt 5) against a host four-index
# reference sliced the way generate_golden.py's _integral_blocks slices PySCF's
# eris; the size accounting (extent, fullBlockDoubles, the peak); and
# create()'s IO refusal of inconsistent blocks, with nothing uploaded. Logic,
# not energies: those stay with the DF golden demos.
#
# REQUIRES_GPU: every case uploads B blocks and runs the slab DGEMMs on the
# card; the drift guard is `unit` only and runs card-free.
add_gtest_executable(
  NAME df_integrals_tests
  SOURCES df_integrals_tests.cpp
  LINK nevpt2::df_integrals nevpt2::test::shared_resources)

add_gtest_suite_tests(
  TARGET df_integrals_tests
  SUITES DfSlabTests DfActiveH2eTests DfSizeAccountingTests DfCreateValidationTests
  REQUIRES_GPU)
