// Suites for nevpt2.cublas_emul's STUBS (src/cublas/cublas_emul.cpp without
// NEVPT2_HAVE_CUBLAS_EMUL) -- every backend but CUDA, i.e. HIP:
//
//   CublasEmulStubTests      requireCublasEmul() refuses: an Unsupported Error
//                              (the Result tier, which main() reports as a
//                              bad flag), naming --cublas as CUDA-only and the
//                              backend this build has
//   CublasEmulStubDeathTest  the handle, GEMM and probe stubs never run: each
//                              aborts through check() (our bug: main() refused
//                              --cublas before any of them could be reached)
//
// This file is compiled into cublas_emul_tests on non-CUDA backends ONLY
// (test/cublas/CMakeLists.txt); a CUDA build gets cublas_emul_digest_tests.cpp
// instead.
#include <gtest/gtest.h>

#include <cstdio>

import std;
import nevpt2.wwr;  // kGpuBackendName
import nevpt2.cublas_emul;
import nevpt2.test.shared_resources;

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::cublas_emul {

static_assert(!kHaveCublasEmul,
              "cublas_emul_stub_tests.cpp is built without the emulated digest only "
              "(test/cublas/CMakeLists.txt)");

// --- CublasEmulStubTests -----------------------------------------------------------

TEST(CublasEmulStubTests, RequireRefusesAsUnsupported) {
  const Status st = requireCublasEmul();
  ASSERT_FALSE(st.has_value());
  EXPECT_EQ(st.error().kind, ErrorKind::Unsupported);
  EXPECT_TRUE(st.error().message.starts_with("the --cublas digest is CUDA-only"))
      << st.error().message;
  EXPECT_NE(st.error().message.find(kGpuBackendName), std::string::npos) << st.error().message;
}

// --- CublasEmulStubDeathTest -------------------------------------------------------

// The abort check() prints for every stub (cublas_emul_unavailable.cppm).
constexpr const char* kUnavailable =
    "error at .*cublas_emul\\.cpp:[0-9]+ .*the --cublas digest was reached in a build "
    "without it";

// The process's DeviceResources in the death-test child: the stubs take a
// stream and a DeviceResources&. If the child cannot bring the device up it
// exits with a message the regex does not match, so the death test fails
// rather than passing on the wrong abort.
const DeviceResources& resourcesInChild() {
  const auto& res = test::sharedResources();
  if (!res.has_value()) {
    std::fputs("no device in the death-test child\n", stderr);
    std::_Exit(3);
  }
  return **res;
}

void makeHandleInChild() { make_emul_handle(53, resourcesInChild().stream()); }

// The pointers are never read: the stub aborts on entry.
void digestGemmInChild() { digest_gemm(nullptr, nullptr, nullptr, nullptr, 1, 1, 1, 0.0); }

void probeInChild() {
  probe_emul_bits(nullptr, resourcesInChild(), nullptr, nullptr, nullptr, 1, 1, 1);
}

TEST(CublasEmulStubDeathTest, MakeHandleAborts) {
  EXPECT_DEATH(makeHandleInChild(), kUnavailable);
}

TEST(CublasEmulStubDeathTest, DigestGemmAborts) {
  EXPECT_DEATH(digestGemmInChild(), kUnavailable);
}

TEST(CublasEmulStubDeathTest, ProbeAborts) { EXPECT_DEATH(probeInChild(), kUnavailable); }

}  // namespace nevpt2::test::cublas_emul
