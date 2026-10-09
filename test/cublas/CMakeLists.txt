# Suites for nevpt2.cublas_emul (src/cublas/): the --cublas digest. One
# binary, whose sources follow the backend exactly as src/cublas/ does --
# real bodies under NEVPT2_HAVE_CUBLAS_EMUL (CUDA), refusing stubs otherwise:
#
#   cublas_emul_tests.cpp         both backends: requireCublasEmul() agrees
#                                 with kHaveCublasEmul and the backend; host
#                                 only
#   cublas_emul_digest_tests.cpp  CUDA only: digest_gemm against native fp64
#                                 and the probe's engaged/declined bits (the
#                                 declined case not under compute-sanitizer)
#   cublas_emul_stub_tests.cpp    HIP only: requireCublasEmul()'s Unsupported
#                                 Error, and the stubs' aborts
#
# So the CUDA-only cases are compiled out of a HIP build (and the stub cases
# out of a CUDA one) rather than skipped, and each file static_asserts
# kHaveCublasEmul to match. The drift guard checks the union of the SUITES
# below, which is per backend too.
#
# The test TUs name no vendor API (only src/cublas/cublas_emul.cpp may): they
# go through nevpt2.cublas_emul and wwr* names (wwrblasDgemm_64 for the
# native-fp64 comparison, through the shared DeviceResources' BLAS handle).
if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
  set(_cublas_emul_backend_source cublas_emul_digest_tests.cpp)
else()
  set(_cublas_emul_backend_source cublas_emul_stub_tests.cpp)
endif()

add_gtest_executable(
  NAME cublas_emul_tests
  SOURCES cublas_emul_tests.cpp ${_cublas_emul_backend_source}
  LINK nevpt2::cublas_emul nevpt2::test::shared_resources)
unset(_cublas_emul_backend_source)

# Host-only on both backends: no handle, no device. Runs card-free.
add_gtest_suite_tests(
  TARGET cublas_emul_tests
  SUITES CublasEmulAvailabilityTests)

if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
  # Each case creates a cuBLAS handle and launches on the shared stream.
  add_gtest_suite_tests(
    TARGET cublas_emul_tests
    SUITES CublasEmulDigestTests CublasEmulProbeTests
    REQUIRES_GPU)
  # The declined case, compiled in (NEVPT2_TEST_CUBLAS_DECLINE) everywhere but
  # the compute-sanitizer build. cuBLAS declines to emulate when its own
  # workspace cudaMallocAsync fails "out of memory", and memcheck reports
  # that failed API call as an error, so this entry can never be clean under
  # it -- not a bug of ours to fix, and not a check to loosen. Compiled out
  # rather than skipped, so the drift guard stays exact (docs/testing.md,
  # "Sanitizers").
  if(NOT NEVPT2_COMPUTE_SANITIZER)
    target_compile_definitions(cublas_emul_tests PRIVATE NEVPT2_TEST_CUBLAS_DECLINE)
    add_gtest_suite_tests(
      TARGET cublas_emul_tests
      SUITES CublasEmulDeclineTests
      REQUIRES_GPU)
  endif()
else()
  # requireCublasEmul()'s refusal: host-only.
  add_gtest_suite_tests(
    TARGET cublas_emul_tests
    SUITES CublasEmulStubTests)
  # The abort tier: EXPECT_DEATH forks per assertion, hence `death`; the child
  # brings the device up for a stream / DeviceResources&, hence REQUIRES_GPU.
  add_gtest_suite_tests(
    TARGET cublas_emul_tests
    SUITES CublasEmulStubDeathTest
    REQUIRES_GPU
    DEATH)
endif()
