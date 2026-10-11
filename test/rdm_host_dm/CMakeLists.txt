# Host-only suites for nevpt2.rdm_host_dm (src/rdm/rdm_host_dm.cppm) -- the
# host rebuild of the active-space dm1/dm2 from a CI vector over
# nevpt2.link_tables, and their tangents. No device, no kernel, no BLAS: NOT
# REQUIRES_GPU, so these run on a box with no card visible.
#
# What they prove is logic, never an energy: the rebuild reproduces each
# committed golden's own dm1/dm2 (PySCF's make_dm123 arrays, which nothing
# else in the tree recomputes) to 1e-12, and the tangent is the derivative of
# that same builder -- exactly along a straight line (the forms are
# quadratic), and with a 4x-per-halving Richardson table along the normalized
# path.
add_gtest_executable(
  NAME rdm_host_dm_tests
  SOURCES rdm_host_dm_tests.cpp
  LINK nevpt2::rdm_host_dm nevpt2::golden)

# The three committed goldens the suites read, passed in rather than
# hard-coded in the source. Read only, never written.
foreach(_case cas44 cas88 cas1010)
  set(_path "${PROJECT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.nevpt2gold")
  if(NOT EXISTS "${_path}")
    message(FATAL_ERROR "test/rdm_host_dm: committed golden missing: ${_path}")
  endif()
  string(TOUPPER "${_case}" _upper)
  target_compile_definitions(rdm_host_dm_tests
                             PRIVATE NEVPT2_TEST_GOLDEN_${_upper}="${_path}")
  unset(_path)
  unset(_upper)
endforeach()

# The default timeout is enough: the heaviest entry is the CAS(10,10) rebuild,
# n^2 (n^2 + 1) / 2 = 5050 length-63504 dot products (the dm2 Gram triangle),
# a fraction of a second in the Release build and well inside 120 s x the
# sanitizer multiplier in a Debug one.
add_gtest_suite_tests(
  TARGET rdm_host_dm_tests
  SUITES HostDm12GoldenTests HostDm12TangentTests HostDm12RichardsonTests
    HostDm12CreateTests)

# The abort tier: a ci or dir of the wrong shape is check(). EXPECT_DEATH
# forks per assertion, hence its own `death` label.
add_gtest_suite_tests(
  TARGET rdm_host_dm_tests
  SUITES HostDm12DeathTest
  DEATH)
