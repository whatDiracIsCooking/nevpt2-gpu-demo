# GPU suite for PC-NEVPT2's reverse mode (#74; nevpt2.gradient's :pc_classes
# partition): that a class the PC solve refuses -- PC_MIN_GAP's gap or a
# non-positive denominator -- is refused by the adjoint too, class by class
# and named, in the CI source term (no y is formed), the digests' adjoint and
# the orbital source term's matrix (Numerical Errors), and the forward
# tangent beside them; and that the golden's own state is accepted. The same
# for PC's H_cc u and its three-term y (#155), with the lift refusal along a
# symmetry-breaking direction matched to pcHessianDevice's. Logic,
# not an energy: the PC source terms' numbers are the golden tier's
# (docs/testing.md, "The PC source terms on the device").
#
# REQUIRES_GPU: every case runs the RDM build and the PC solves.
add_gtest_executable(
  NAME pc_adjoint_tests
  SOURCES pc_adjoint_tests.cpp
  LINK nevpt2::gradient nevpt2::rdm_host_dm nevpt2::golden nevpt2::test::shared_resources)

foreach(_suffix nevpt2gold:GOLDEN gradgold:GRADGOLD)
  string(REPLACE ":" ";" _pair "${_suffix}")
  list(GET _pair 0 _ext)
  list(GET _pair 1 _what)
  set(_path "${PROJECT_SOURCE_DIR}/golden/n2_ccpvdz_cas44.${_ext}")
  if(NOT EXISTS "${_path}")
    message(FATAL_ERROR "test/pc_adjoint: committed input missing: ${_path}")
  endif()
  target_compile_definitions(pc_adjoint_tests PRIVATE NEVPT2_TEST_${_what}_CAS44="${_path}")
  unset(_path)
  unset(_pair)
  unset(_ext)
  unset(_what)
endforeach()

add_gtest_suite_tests(
  TARGET pc_adjoint_tests
  SUITES PcAdjointRefusalTests
  REQUIRES_GPU)
