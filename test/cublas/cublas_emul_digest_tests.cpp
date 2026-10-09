// Suites for nevpt2.cublas_emul's REAL bodies (src/cublas/cublas_emul.cpp under
// NEVPT2_HAVE_CUBLAS_EMUL) -- the --cublas digest, CUDA only:
//
//   CublasEmulDigestTests  digest_gemm at max_mantissa_bits = 53 against native
//                            fp64 (wwrblasDgemm_64 on the same operands, the
//                            --blas-digest mapping) and against a long-double
//                            host reference, on seeded random operands over
//                            shapes that are and are not multiples of a tile;
//                            nonzero beta accumulates into C
//   CublasEmulProbeTests   probe_emul_bits returns the mantissa bits cuBLAS
//                            used: the configured count where it engages
//                            (53, 40, 20; and 53 on the CAS(10,10) --tiles 3
//                            golden path's digest shape), and fewer bits are
//                            really used
//   CublasEmulDeclineTests -1 on a shape it declines, where the GEMM must
//                            still be right, run natively. Built only outside
//                            the compute-sanitizer preset
//                            (NEVPT2_TEST_CUBLAS_DECLINE, set by
//                            test/cublas/CMakeLists.txt): cuBLAS declines
//                            because its own workspace cudaMallocAsync fails
//                            with "out of memory", and memcheck reports that
//                            failed API call as an error (docs/testing.md,
//                            "Sanitizers")
//
// This file is compiled into cublas_emul_tests on the CUDA backend ONLY
// (test/cublas/CMakeLists.txt); a HIP build never sees it, and gets
// cublas_emul_stub_tests.cpp instead. It names no vendor API: everything goes
// through nevpt2.cublas_emul and wwr* names (only src/cublas/cublas_emul.cpp
// may name cuBLAS).
//
// THE BOUND. Write u = 2^-53, S[i,j] = sum_k |A[i,k] B[j,k]|, and
// mA[i] = max_k |A[i,k]|, mB[j] = max_k |B[j,k]|. Native fp64 in ANY summation
// order is within gamma_K S <= gamma_K K mA mB (Higham, Thm 3.4), gamma_K =
// K u / (1 - K u). The fixed-point emulation splits each row against its own
// scale, so its error is naturally measured against K mA mB, not S: keeping
// 53 mantissa bits per row loses at most ~2u of each row's max per element,
// i.e. ~2 * 2u K mA mB on the products, and the slice products are summed
// exactly in integers and combined in fp64 (a few more roundings of the
// result). Both sides are therefore held to
//
//   |C - ref| <= (gamma_K + 8u) K mA mB + 2u |beta C0| + 4 * 2^-64 K mA mB
//
// (the last term is the long-double reference's own error, as in
// test/ozaki). On O(1) random operands and the K of CublasEmulDigestTests
// (at most 4096) that is at most ~2e-9 absolute: far below what a transposed
// operand, a wrong leading dimension or a dropped K term would cost, and below
// what 20 mantissa bits give (CublasEmulProbeTests checks that 20 bits does NOT
// meet it).
//
// REQUIRES_GPU: every case creates a cuBLAS handle and launches on the
// process's one DeviceResources stream (nevpt2.test.shared_resources); the
// handle is bound to that stream, as rdm_build.cpp binds its own.
#include <gtest/gtest.h>

import std;
import nevpt2.wwr;
import nevpt2.cublas_emul;
import nevpt2.test.shared_resources;

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::cublas_emul {

static_assert(kHaveCublasEmul,
              "cublas_emul_digest_tests.cpp is built on CUDA only (test/cublas/CMakeLists.txt)");
static_assert(std::numeric_limits<long double>::digits >= 64,
              "the host reference needs an extended-precision long double");

constexpr double kU = 0x1p-53;

// The process's DeviceResources, or nullptr after recording the creation
// Error as this test's failure (callers ASSERT_NE on it).
const DeviceResources* resourcesOrFail() {
  const auto& res = test::sharedResources();
  if (!res.has_value()) {
    ADD_FAILURE() << "DeviceResources::create failed: " << kindName(res.error().kind) << ": "
                  << res.error().message;
    return nullptr;
  }
  return res->get();
}

// An emulation handle bound to the shared stream, destroyed with the scope.
class Handle {
 public:
  Handle(const int maxMantissaBits, const DeviceResources& res)
      : h_(make_emul_handle(maxMantissaBits, res.stream())) {}
  ~Handle() { destroy_emul_handle(h_); }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  EmulHandle get() const { return h_; }

 private:
  EmulHandle h_;
};

DeviceBuffer<double> upload(const std::vector<double>& host, const DeviceResources& res) {
  DeviceBuffer<double> buf(host.size(), res.shared_from_this());
  gpuCheck(wwrMemcpyAsync(buf.data(), host.data(), buf.size_bytes(), wwrMemcpyHostToDevice,
                          res.stream()));
  return buf;
}

std::vector<double> download(const DeviceBuffer<double>& buf, const DeviceResources& res) {
  std::vector<double> host(buf.num_elements());
  gpuCheck(wwrMemcpyAsync(host.data(), buf.data(), buf.size_bytes(), wwrMemcpyDeviceToHost,
                          res.stream()));
  // The host reads `host` next; an async copy into pageable memory may return
  // before it has landed (and the GEMMs queued before it must have run).
  gpuCheck(wwrStreamSynchronize(res.stream()));
  return host;
}

// `n` uniform values in [-1, 1) from a fixed seed, so a failure reproduces.
std::vector<double> randomVector(const std::size_t n, const std::uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> v(n);
  for (double& x : v) x = dist(gen);
  return v;
}

struct Gemm {
  int M = 0, N = 0, K = 0;
  std::vector<double> A;   // [M, K]
  std::vector<double> B;   // [N, K]
  std::vector<double> C0;  // [M, N], C on entry
  double beta = 0.0;
};

Gemm randomGemm(const int M, const int N, const int K, const double beta,
                const std::uint64_t seed) {
  const auto m = static_cast<std::size_t>(M), n = static_cast<std::size_t>(N),
             k = static_cast<std::size_t>(K);
  return Gemm{M,
              N,
              K,
              randomVector(m * k, seed + 1),
              randomVector(n * k, seed + 2),
              beta == 0.0 ? std::vector<double>(m * n) : randomVector(m * n, seed + 3),
              beta};
}

// max_k |X[r, k]| per row.
std::vector<double> rowMax(const std::vector<double>& X, const int rows, const int K) {
  std::vector<double> m(static_cast<std::size_t>(rows), 0.0);
  for (std::size_t r = 0; r < m.size(); ++r)
    for (std::size_t k = 0; k < static_cast<std::size_t>(K); ++k)
      m[r] = std::max(m[r], std::abs(X[r * static_cast<std::size_t>(K) + k]));
  return m;
}

// digest_gemm's contract (cublas_emul.cppm), in long double with
// Neumaier-compensated summation: sum_K A[i,k] B[j,k] + beta * C0[i,j].
std::vector<long double> reference(const Gemm& g) {
  const auto K = static_cast<std::size_t>(g.K);
  std::vector<long double> ref(static_cast<std::size_t>(g.M) * g.N);
  for (std::size_t i = 0; i < static_cast<std::size_t>(g.M); ++i) {
    for (std::size_t j = 0; j < static_cast<std::size_t>(g.N); ++j) {
      long double sum = 0.0L, comp = 0.0L;
      for (std::size_t k = 0; k < K; ++k) {
        const long double t =
            static_cast<long double>(g.A[i * K + k]) * static_cast<long double>(g.B[j * K + k]);
        const long double s = sum + t;
        comp += std::abs(sum) >= std::abs(t) ? (sum - s) + t : (t - s) + sum;
        sum = s;
      }
      const std::size_t ij = i * static_cast<std::size_t>(g.N) + j;
      ref[ij] = sum + comp + static_cast<long double>(g.beta) * g.C0[ij];
    }
  }
  return ref;
}

// Checks C against the file comment's bound; returns the largest
// |C - ref| / (K mA mB).
double checkAgainstBound(const Gemm& g, const std::vector<double>& c, const std::string_view what,
                         const bool expectWithin = true) {
  const std::vector<long double> ref = reference(g);
  const std::vector<double> mA = rowMax(g.A, g.M, g.K);
  const std::vector<double> mB = rowMax(g.B, g.N, g.K);
  const double K = static_cast<double>(g.K);
  const double perK = K * kU / (1.0 - K * kU) + 8.0 * kU + 4.0 * 0x1p-64;
  double worst = 0.0;
  std::size_t bad = 0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(g.M); ++i) {
    for (std::size_t j = 0; j < static_cast<std::size_t>(g.N); ++j) {
      const std::size_t ij = i * static_cast<std::size_t>(g.N) + j;
      const double kmm = K * mA[i] * mB[j];
      const double allowed = kmm * perK + 2.0 * kU * std::abs(g.beta * g.C0[ij]);
      const auto err =
          static_cast<double>(std::abs(static_cast<long double>(c[ij]) - ref[ij]));
      if (kmm > 0.0) worst = std::max(worst, err / kmm);
      if (err <= allowed) continue;
      if (expectWithin && ++bad <= 5) {
        ADD_FAILURE() << what << ": C[" << i << "," << j << "] got " << c[ij] << ", want "
                      << static_cast<double>(ref[ij]) << " (|diff| " << err << ", allowed "
                      << allowed << ")";
      }
    }
  }
  if (expectWithin) EXPECT_EQ(bad, 0u) << what << ": elements outside the bound";
  return worst;
}

// One digest_gemm on g through `h`; returns C.
std::vector<double> runEmulated(const DeviceResources& res, const Handle& h, const Gemm& g) {
  const DeviceBuffer<double> dA = upload(g.A, res);
  const DeviceBuffer<double> dB = upload(g.B, res);
  DeviceBuffer<double> dC = upload(g.C0, res);
  digest_gemm(h.get(), dA.data(), dB.data(), dC.data(), g.M, g.N, g.K, g.beta);
  return download(dC, res);
}

// The same GEMM as native fp64, through DeviceResources' BLAS handle with
// --blas-digest's operand mapping (rdm_build_blas.cppm, digestGemmNative):
// row-major C[M,N] = A[M,K] . B[N,K]^T is column-major C^T = op_T(B) . A.
std::vector<double> runNative(const DeviceResources& res, const Gemm& g) {
  const DeviceBuffer<double> dA = upload(g.A, res);
  const DeviceBuffer<double> dB = upload(g.B, res);
  DeviceBuffer<double> dC = upload(g.C0, res);
  const double one = 1.0;
  gpuCheck(wwrblasDgemm_64(res.blas(), WWRBLAS_OP_T, WWRBLAS_OP_N, g.N, g.M, g.K, &one,
                           dB.data(), g.K, dA.data(), g.K, &g.beta, dC.data(), g.N));
  return download(dC, res);
}

// The probe on g (beta = 0 by contract): the bits it reports, and the C it
// wrote.
struct Probe {
  int bits = 0;
  std::vector<double> c;
};

Probe runProbe(const DeviceResources& res, const Handle& h, const Gemm& g) {
  const DeviceBuffer<double> dA = upload(g.A, res);
  const DeviceBuffer<double> dB = upload(g.B, res);
  DeviceBuffer<double> dC(static_cast<std::size_t>(g.M) * static_cast<std::size_t>(g.N),
                          res.shared_from_this());
  const int bits = probe_emul_bits(h.get(), res, dA.data(), dB.data(), dC.data(), g.M, g.N, g.K);
  return {bits, download(dC, res)};
}

// --- CublasEmulDigestTests --------------------------------------------------------

constexpr std::array<std::array<int, 3>, 6> kShapes{{{1, 1, 1},
                                                     {16, 16, 16},
                                                     {17, 15, 33},
                                                     {64, 48, 129},
                                                     {70, 33, 1000},
                                                     {5, 40, 4096}}};

// Each shape is probed first and must ENGAGE at 53 bits: a declined shape
// would run native fp64 and make the comparison vacuous. The probe's own C
// (beta = 0) must be the GEMM too, and digest_gemm after it must still work --
// the probe un-registers its bit-count pointer before freeing it.
TEST(CublasEmulDigestTests, MatchesNativeFp64OnRandomOperands) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const Handle h(53, *res);
  std::uint64_t seed = 100;
  for (const auto& [M, N, K] : kShapes) {
    const std::string what = std::format("M={} N={} K={}", M, N, K);
    SCOPED_TRACE(what);
    const Gemm g = randomGemm(M, N, K, 0.0, seed += 10);

    const Probe p = runProbe(*res, h, g);
    ASSERT_EQ(p.bits, 53) << "emulation did not engage on this shape: the comparison below "
                             "would be native fp64 against itself";
    checkAgainstBound(g, p.c, what + " (probe)");

    const std::vector<double> emulated = runEmulated(*res, h, g);
    const std::vector<double> native = runNative(*res, g);
    checkAgainstBound(g, emulated, what + " (emulated)");
    checkAgainstBound(g, native, what + " (native)");
    // Emulated against native directly: both are within the bound of the
    // reference, so of each other within twice it.
    const std::vector<double> mA = rowMax(g.A, M, K);
    const std::vector<double> mB = rowMax(g.B, N, K);
    const double perK =
        static_cast<double>(K) * kU / (1.0 - static_cast<double>(K) * kU) + 8.0 * kU;
    for (std::size_t i = 0; i < static_cast<std::size_t>(M); ++i)
      for (std::size_t j = 0; j < static_cast<std::size_t>(N); ++j) {
        const std::size_t ij = i * static_cast<std::size_t>(N) + j;
        EXPECT_LE(std::abs(emulated[ij] - native[ij]), 2.0 * perK * K * mA[i] * mB[j])
            << "C[" << i << "," << j << "]";
      }
  }
}

// C = A B^T + beta C0 for nonzero beta (rdm_build.cpp accumulates dm3 with
// beta = 1), and twice with beta = 1 into the same C.
TEST(CublasEmulDigestTests, NonzeroBetaAccumulatesIntoC) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const Handle h(53, *res);
  for (const double beta : {1.0, -0.5, 3.25}) {
    const std::string what = std::format("beta={}", beta);
    SCOPED_TRACE(what);
    const Gemm g = randomGemm(19, 21, 37, beta, 4000);
    checkAgainstBound(g, runEmulated(*res, h, g), what);
  }
  Gemm g = randomGemm(9, 23, 50, 1.0, 4100);
  g.C0 = runEmulated(*res, h, g);
  checkAgainstBound(g, runEmulated(*res, h, g), "second beta=1 call");
}

// --- CublasEmulProbeTests ---------------------------------------------------------

// Where cuBLAS engages, the probe reports exactly the configured cap: what
// rdm_build.cpp prints as "engaged bits=<n>".
TEST(CublasEmulProbeTests, ReportsTheConfiguredBitsWhenEngaged) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const Gemm g = randomGemm(64, 48, 129, 0.0, 7000);
  for (const int bits : {53, 40, 20}) {
    SCOPED_TRACE(std::format("max_mantissa_bits={}", bits));
    const Handle h(bits, *res);
    EXPECT_EQ(runProbe(*res, h, g).bits, bits);
  }
}

// The golden path's own digest shape: nevpt2_cas1010_cublas (CAS(10,10),
// --tiles 3) probes M = n^4 = 10000, N = n^2 = 100, K = its tile width, 21168
// (the demo's "tile width=" line), and the golden entry needs it engaged
// (docs/performance.md, "The cuBLAS fixed-point-emulation digest (`--cublas`)",
// the probe table). The operands are the DeviceBuffers' zero fill, which spares
// uploading an n^4 x K operand; that table's row was measured on them.
TEST(CublasEmulProbeTests, EngagesOnTheCas1010GoldenDigestShape) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  constexpr int M = 10000, N = 100, K = 21168;
  const Handle h(53, *res);
  const DeviceBuffer<double> dA(std::size_t{M} * K, res->shared_from_this());
  const DeviceBuffer<double> dB(std::size_t{N} * K, res->shared_from_this());
  DeviceBuffer<double> dC(std::size_t{M} * N, res->shared_from_this());
  EXPECT_EQ(probe_emul_bits(h.get(), *res, dA.data(), dB.data(), dC.data(), M, N, K), 53);
}

// The cap is honoured, not just echoed: at 20 bits the digest's error is
// orders of magnitude above its error at 53.
TEST(CublasEmulProbeTests, FewerBitsAreReallyUsed) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const Gemm g = randomGemm(64, 48, 129, 0.0, 9000);
  const Handle h53(53, *res);
  const Handle h20(20, *res);
  const double err53 = checkAgainstBound(g, runEmulated(*res, h53, g), "53 bits");
  const double err20 =
      checkAgainstBound(g, runEmulated(*res, h20, g), "20 bits", /*expectWithin=*/false);
  EXPECT_GT(err20, 1e6 * err53) << "53 bits: " << err53 << ", 20 bits: " << err20
                                << " (worst |C - ref| / (K mA mB))";
}

#if defined(NEVPT2_TEST_CUBLAS_DECLINE)

// --- CublasEmulDeclineTests -------------------------------------------------------

// A shape cuBLAS declines (docs/performance.md, "The cuBLAS
// fixed-point-emulation digest (`--cublas`)", the probe table: a K this long
// declines even at M = N = 1): the probe reports -1 -- what rdm_build.cpp
// prints as "(DECLINED -> ran native fp64!)" -- and the GEMM it ran natively
// is still right. The bound is native fp64's, loose at this K (~2e-3
// absolute) but far below what a wrong result would be off by.
TEST(CublasEmulDeclineTests, ReportsMinusOneWhenDeclined) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const Handle h(53, *res);
  const Gemm g = randomGemm(1, 1, 4'000'000, 0.0, 8000);
  const Probe p = runProbe(*res, h, g);
  EXPECT_EQ(p.bits, -1) << "expected cuBLAS to decline this shape";
  checkAgainstBound(g, p.c, "declined probe");
  checkAgainstBound(g, runEmulated(*res, h, g), "declined digest_gemm");
}

#endif  // NEVPT2_TEST_CUBLAS_DECLINE

}  // namespace nevpt2::test::cublas_emul
