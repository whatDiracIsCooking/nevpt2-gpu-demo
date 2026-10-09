// Suites for nevpt2.cublas_emul (src/cublas/) that hold on EVERY backend:
//
//   CublasEmulAvailabilityTests  requireCublasEmul() and kHaveCublasEmul tell
//                                  the same story, and it is the backend's:
//                                  the emulated digest exactly on CUDA;
//                                  destroy_emul_handle(nullptr) is a no-op on
//                                  both (rdm_build.cpp destroys its handle
//                                  unconditionally, and makes one only under
//                                  --cublas)
//
// Host-only: nothing here creates a handle or touches a device, so these run
// with no card visible. The backend-specific halves are their own files, each
// built into this binary on one backend only (test/cublas/CMakeLists.txt):
// cublas_emul_digest_tests.cpp on CUDA (the emulated GEMM and its probe), and
// cublas_emul_stub_tests.cpp elsewhere (the stubs refuse and abort).
#include <gtest/gtest.h>

import std;
import nevpt2.wwr;  // kGpuBackendName
import nevpt2.cublas_emul;

// Not a module unit (gtest is a textual header), so no partition: the suites
// live in a named namespace instead of an anonymous one.
namespace nevpt2::test::cublas_emul {

TEST(CublasEmulAvailabilityTests, RequireAgreesWithTheCompileTimeFlag) {
  EXPECT_EQ(requireCublasEmul().has_value(), kHaveCublasEmul);
  // The emulated compute type is CUDA's alone (cublas/CMakeLists.txt).
  EXPECT_EQ(kHaveCublasEmul, std::string_view(kGpuBackendName) == "CUDA");
}

TEST(CublasEmulAvailabilityTests, DestroyingANullHandleIsANoOp) {
  destroy_emul_handle(nullptr);
  SUCCEED();
}

}  // namespace nevpt2::test::cublas_emul
