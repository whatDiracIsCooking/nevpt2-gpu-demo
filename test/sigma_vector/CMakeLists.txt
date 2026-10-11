# GPU suites for nevpt2.sigma_vector -- the 3-body and 4-body-contracted
# sigma vectors (src/rdm/sigma_vector.cu through src/rdm/sigma_vector.cpp) on
# seeded random coefficient tensors at CAS(4,4) and CAS(6,6): the full vector
# against an explicit host construction by repeated FORWARD application of
# E_pq (the device walks the reverse tables), the CI derivative of the host
# dm3 contraction by an exact central difference, the two contraction
# identities against the device build's own dm3/f3ca/f3ac, and tiling
# invariance. Logic, not an energy: no golden file is read.
#
# No .cu here: the binary imports nevpt2.sigma_vector, which links the device
# library itself, as rdm_dm3_tests does with nevpt2.rdm_build. nevpt2::rdm_build
# is linked for the two contraction identities (the device dm3/f3), and
# nevpt2::rdm_plan for the tile width a case checks its own plan against.
#
# REQUIRES_GPU: every case launches the sigma kernels; the drift guard is
# `unit` only and runs card-free.
add_gtest_executable(
  NAME sigma_vector_tests
  SOURCES sigma_vector_tests.cpp
  LINK nevpt2::sigma_vector nevpt2::rdm_build nevpt2::rdm_plan nevpt2::link_tables
    nevpt2::test::shared_resources)

add_gtest_suite_tests(
  TARGET sigma_vector_tests
  SUITES SigmaVector3Tests SigmaVector3GammaTests SigmaVector4Tests
  REQUIRES_GPU)
