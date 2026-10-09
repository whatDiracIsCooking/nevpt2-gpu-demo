// Suites for the --ozaki digest GEMM (src/ozaki/ozaki_digest.cu), called
// through its bridge, ozaki_digest_bridge.h:
//
//     C[M,N] = sum_K A[M,K] * B[N,K]  (+ beta * C),   both K-contiguous,
//
// emulated on int8 tensor cores by the Ozaki scheme. Each case runs on seeded
// random operands against a host reference written from that contract, not
// from the kernel.
//
//   OzakiAllPairsTests       maxPairSum = 14 (all 64 pairs) matches fp64 within
//                              the bound below, over M/N/K that are and are
//                              not multiples of the 16-wide WMMA tile
//   OzakiPairTruncationTests maxPairSum 0..14: each within its own bound, and
//                              the error does not grow as pairs are added
//   OzakiKChunkingTests      K = kOzakiMaxK, kOzakiMaxK + 1 and 2*kOzakiMaxK + 1
//                              (1, 2 and 3 chunks), beta zero and not
//   OzakiBetaTests           nonzero beta accumulates into C
//   OzakiRowScalingTests     rows of very different magnitudes, a row whose
//                              elements span 2^-44..1, an all-zero row of A
//                              and of B
//   OzakiScratchBytesTests   ozakiScratchBytes at the 16-padding edges, and a
//                              launch with exactly that much scratch
//
// Every launch gets EXACTLY ozakiScratchBytes(M, N, K) bytes of scratch,
// followed by a sentinel guard that must come back untouched, so an
// under-sized scratch formula fails here even where the pool hides the
// overrun from compute-sanitizer.
//
// The error bound, derived from the scheme (the file comment of
// ozaki_digest.cu and docs/performance.md, "The int8 Ozaki digest"), not
// fitted to measured errors. Write u = 2^-53, mA = max_k |A[i,k]| and
// mB = max_k |B[j,k]| (over the whole K). Each row is scaled by a power of two
// s = 2^(e+1) with max|row| = f * 2^e, f in [1/2, 1), so s <= 4 * max|row| and
// sA * sB <= 16 mA mB; |x / s| <= 1/2. Each of the 8 digits has |d| <= 64 and
// is worth 2^-7(p+1), so x / s = sum_p d_p 2^-7(p+1) + rho, |rho| <= 2^-57.
// The int32 level sums are exact (8 * 64^2 * K < 2^31 for every chunk), and
// the level weights and the scale-back are powers of two, so exact. That
// leaves four sources, each per output element:
//
//  1. Digit truncation: |a b - a~ b~| <= sA sB 2^-57 (1/2 + 1/2 + 2^-57)
//     per term, so <= K mA mB 16 * 2^-57 = K mA mB 2^-53 in all.
//  2. The final fp64 sum of at most 15 exact level terms: <= gamma_14 times
//     the sum of their magnitudes, which is at most
//     sA sB K (sum_p 64 * 2^-7(p+1))^2 < 16 K mA mB (64/127)^2.
//  3. Dropped levels (maxPairSum = P < 14): level g holds n_g = min(g+1, 15-g)
//     pairs, each term <= 64^2 2^-7(g+2) = 2^-2-7g after scaling down, so the
//     dropped part is <= K mA mB 4 sum_{g > P} n_g 2^-7g.
//  4. The update of C: each of the nChunks launches computes v + beta' * C
//     (beta' = beta for the first chunk, 1 after), at most two roundings of a
//     value no larger than K mA mB + |beta C0| (to first order in u).
//
// The host reference sums the K products in long double (64-bit mantissa,
// static_asserted) with Neumaier compensation: its error is
// <= (3 u_L + O(K u_L^2)) K mA mB, u_L = 2^-64, covered by a 4 * 2^-64 term.
//
// So per element:
//   |C - ref| <= K mA mB (2^-53 + 16 (64/127)^2 gamma_14 + drop(P) + 4 * 2^-64)
//                + nChunks * 2u (K mA mB + |beta C0|).
// At P = 14 that is ~6.4e-15 K mA mB: loose next to the ~3e-15 of max|C| the
// demos measure, and still ~10^14 below what a misplaced index or a dropped
// K term would cost on O(1) inputs.
//
// REQUIRES_GPU: every case launches on the process's one DeviceResources
// stream (nevpt2.test.shared_resources).
//
// TU shape: gtest's header and runtime.h FIRST (runtime.h before the imports,
// so the bridge's own #include of it is a #pragma once no-op), then the
// imports, then the bridge, which needs nevpt2.wwr's wwrError_t. A plain TU is
// already in the global module, so the bridge needs no extern "C++" here.
#include <gtest/gtest.h>

#include <runtime.h>

import std;
import nevpt2.wwr;
import nevpt2.test.shared_resources;

#include "ozaki/ozaki_digest_bridge.h"

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::ozaki {

using device::kOzakiMaxK;
using device::ozakiDigestGemm;
using device::ozakiScratchBytes;

static_assert(std::numeric_limits<long double>::digits >= 64,
              "the host reference needs an extended-precision long double");

constexpr int kAllPairs = 14;
constexpr double kU = 0x1p-53;
constexpr double kGamma14 = 14.0 * kU / (1.0 - 14.0 * kU);

// Per unit of K mA mB, sources 1-3 of the file comment plus the reference's.
double perKBound(const int maxPairSum) {
  double drop = 0.0;
  for (int g = maxPairSum + 1; g <= 14; ++g) {
    const int pairs = std::min(g + 1, 15 - g);
    drop += 4.0 * pairs * std::ldexp(1.0, -7 * g);
  }
  const double sumMag = 16.0 * (64.0 / 127.0) * (64.0 / 127.0);
  return 0x1p-53 + sumMag * kGamma14 + drop + 4.0 * 0x1p-64;
}

int chunksFor(const int K) { return (K + kOzakiMaxK - 1) / kOzakiMaxK; }

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
  // before it has landed (and the kernels queued before it must have run).
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

// The contract, in long double with Neumaier-compensated summation:
// sum_K A[i,k] B[j,k] + beta * C0[i,j].
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

// Guard doubles after the scratch, and the value they hold.
constexpr std::size_t kGuard = 512;
constexpr double kSentinel = -1234.5;

// One ozakiDigestGemm on g with exactly ozakiScratchBytes of scratch plus a
// guard that must survive; returns C.
std::vector<double> runOzaki(const DeviceResources& res, const Gemm& g, const int maxPairSum) {
  const std::size_t bytes = ozakiScratchBytes(g.M, g.N, g.K);
  EXPECT_EQ(bytes % sizeof(double), 0u) << "scratch bytes " << bytes;
  const std::size_t words = bytes / sizeof(double);
  std::vector<double> hScratch(words + kGuard, 0.0);
  std::fill(hScratch.begin() + static_cast<std::ptrdiff_t>(words), hScratch.end(), kSentinel);

  const DeviceBuffer<double> dA = upload(g.A, res);
  const DeviceBuffer<double> dB = upload(g.B, res);
  DeviceBuffer<double> dC = upload(g.C0, res);
  DeviceBuffer<double> dScratch = upload(hScratch, res);
  gpuCheck(ozakiDigestGemm(res.stream(), dA.data(), dB.data(), dC.data(), g.M, g.N, g.K, g.beta,
                           maxPairSum, dScratch.data()));
  std::vector<double> c = download(dC, res);

  const std::vector<double> after = download(dScratch, res);
  std::size_t clobbered = 0;
  for (std::size_t w = words; w < after.size(); ++w) clobbered += after[w] != kSentinel ? 1 : 0;
  EXPECT_EQ(clobbered, 0u) << "guard words past the " << bytes
                           << "-byte scratch were overwritten";
  EXPECT_EQ(download(dA, res), g.A) << "A must be unchanged";
  EXPECT_EQ(download(dB, res), g.B) << "B must be unchanged";
  return c;
}

// Checks C against the file comment's bound and returns the largest
// |C - ref| / (K mA mB) over the elements where that is nonzero.
double checkAgainstBound(const Gemm& g, const std::vector<double>& c, const int maxPairSum,
                         const std::string_view what) {
  const std::vector<long double> ref = reference(g);
  const std::vector<double> mA = rowMax(g.A, g.M, g.K);
  const std::vector<double> mB = rowMax(g.B, g.N, g.K);
  const double perK = perKBound(maxPairSum);
  const double update = chunksFor(g.K) * 2.0 * kU;
  double worst = 0.0;
  std::size_t bad = 0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(g.M); ++i) {
    for (std::size_t j = 0; j < static_cast<std::size_t>(g.N); ++j) {
      const std::size_t ij = i * static_cast<std::size_t>(g.N) + j;
      const double kmm = static_cast<double>(g.K) * mA[i] * mB[j];
      const double betaC = std::abs(g.beta * g.C0[ij]);
      const double allowed = kmm * perK + update * (kmm + betaC);
      const auto err =
          static_cast<double>(std::abs(static_cast<long double>(c[ij]) - ref[ij]));
      if (kmm > 0.0) worst = std::max(worst, err / kmm);
      if (err <= allowed) continue;
      if (++bad <= 5) {
        ADD_FAILURE() << what << ": C[" << i << "," << j << "] got " << c[ij] << ", want "
                      << static_cast<double>(ref[ij]) << " (|diff| " << err << ", allowed "
                      << allowed << ")";
      }
    }
  }
  EXPECT_EQ(bad, 0u) << what << ": elements outside the scheme's bound";
  return worst;
}

// --- OzakiAllPairsTests ----------------------------------------------------------

// Every dimension below, at and above the 16-wide tile; M = 70 spans two blocks
// of four M tiles, N = 33 three N tiles.
constexpr std::array<std::array<int, 3>, 8> kShapes{{{1, 1, 1},
                                                     {16, 16, 16},
                                                     {15, 17, 31},
                                                     {17, 15, 33},
                                                     {32, 48, 64},
                                                     {70, 33, 129},
                                                     {5, 40, 1000},
                                                     {64, 1, 17}}};

TEST(OzakiAllPairsTests, MatchesFp64AcrossPaddedShapes) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  std::uint64_t seed = 100;
  for (const auto& [M, N, K] : kShapes) {
    const std::string what = std::format("M={} N={} K={}", M, N, K);
    SCOPED_TRACE(what);
    const Gemm g = randomGemm(M, N, K, 0.0, seed += 10);
    checkAgainstBound(g, runOzaki(*res, g, kAllPairs), kAllPairs, what);
  }
}

// --- OzakiPairTruncationTests ------------------------------------------------------

// Each maxPairSum meets its own bound, and the normalized error does not grow
// as pairs are added. Dropped level g alone is worth up to ~2^-7g, so between
// neighbouring P the error falls by orders of magnitude until it reaches the
// rounding floor (sources 1, 2 and 4 at P = 14); past there, adding a level
// below the floor may move the rounding either way, so err(P + 1) is allowed
// err(P) plus that floor and no more.
TEST(OzakiPairTruncationTests, ErrorIsNonIncreasingInMaxPairSum) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const Gemm g = randomGemm(40, 24, 300, 0.0, 2000);
  const double roundingFloor = perKBound(kAllPairs) + chunksFor(g.K) * 2.0 * kU;
  std::array<double, kAllPairs + 1> err{};
  for (int p = 0; p <= kAllPairs; ++p) {
    const std::string what = std::format("maxPairSum={}", p);
    SCOPED_TRACE(what);
    err[static_cast<std::size_t>(p)] = checkAgainstBound(g, runOzaki(*res, g, p), p, what);
  }
  for (std::size_t p = 0; p < kAllPairs; ++p) {
    EXPECT_LE(err[p + 1], err[p] + roundingFloor)
        << "error grew from maxPairSum " << p << " (" << err[p] << ") to " << p + 1 << " ("
        << err[p + 1] << ")";
  }
  // maxPairSum is honoured at all: P = 0 drops level 1, ~2^-7 of level 0,
  // which is far above the P = 14 rounding floor.
  EXPECT_GT(err[0], 1e6 * roundingFloor) <<"maxPairSum = 0 lost no accuracy";
}

// --- OzakiKChunkingTests ----------------------------------------------------------

// K past kOzakiMaxK is cut into near-equal chunks, each scaled on its own and
// added with beta = 1. M and N stay tiny: the operands are 3 x K doubles. In
// A's row 0 the second half of K is 2^-30 smaller than the first, so the
// chunks get different row scales.
TEST(OzakiKChunkingTests, WideKAccumulatesAcrossChunks) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  for (const int K : {kOzakiMaxK, kOzakiMaxK + 1, 2 * kOzakiMaxK + 1}) {
    for (const double beta : {0.0, -0.75}) {
      const std::string what =
          std::format("K={} ({} chunks) beta={}", K, chunksFor(K), beta);
      SCOPED_TRACE(what);
      Gemm g = randomGemm(3, 2, K, beta, 3000 + static_cast<std::uint64_t>(K));
      for (std::size_t k = static_cast<std::size_t>(K) / 2; k < static_cast<std::size_t>(K); ++k)
        g.A[k] = std::ldexp(g.A[k], -30);
      checkAgainstBound(g, runOzaki(*res, g, kAllPairs), kAllPairs, what);
    }
  }
}

// --- OzakiBetaTests ------------------------------------------------------------------

// C = GEMM + beta * C0 for a nonzero beta, on a padded shape; and beta = 1
// twice into the same C gives C0 + 2 * GEMM.
TEST(OzakiBetaTests, NonzeroBetaAccumulatesIntoC) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  for (const double beta : {1.0, -0.5, 3.25}) {
    const std::string what = std::format("beta={}", beta);
    SCOPED_TRACE(what);
    const Gemm g = randomGemm(19, 21, 37, beta, 4000);
    checkAgainstBound(g, runOzaki(*res, g, kAllPairs), kAllPairs, what);
  }

  // Twice with beta = 1: the first result is C0' for the second call, so the
  // per-call bound applies to each step.
  Gemm g = randomGemm(9, 23, 50, 1.0, 4100);
  g.C0 = runOzaki(*res, g, kAllPairs);
  checkAgainstBound(g, runOzaki(*res, g, kAllPairs), kAllPairs, "second beta=1 call");
}

// --- OzakiRowScalingTests ------------------------------------------------------------

// Each row is scaled on its own, so rows 2^80 apart in magnitude keep their
// own relative accuracy; a row spanning 2^-44..1 internally loses only what
// the bound's truncation term allows; an all-zero row (scale 1) gives exactly
// zero, or exactly beta * C0.
TEST(OzakiRowScalingTests, RowsOfDifferentMagnitudesAndZeroRows) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const int M = 7, N = 5, K = 45;
  const std::array<double, M> aScale{0x1p-40, 1e-12, 1.0, 3.7e5, 0x1p40, 0.0, 1.0};
  const std::array<double, N> bScale{1e-9, 0.0, 1e6, 0x1p-20, 1.0};
  for (const double beta : {0.0, 2.0}) {
    const std::string what = std::format("beta={}", beta);
    SCOPED_TRACE(what);
    Gemm g = randomGemm(M, N, K, beta, 5000);
    for (std::size_t i = 0; i < M; ++i)
      for (std::size_t k = 0; k < K; ++k) g.A[i * K + k] *= aScale[i];
    for (std::size_t j = 0; j < N; ++j)
      for (std::size_t k = 0; k < K; ++k) g.B[j * K + k] *= bScale[j];
    // Row 6 of A: element k is 2^-k below the row's largest, down to 2^-44.
    for (std::size_t k = 0; k < K; ++k) g.A[6 * K + k] = std::ldexp(0.75, -static_cast<int>(k));

    const std::vector<double> c = runOzaki(*res, g, kAllPairs);
    checkAgainstBound(g, c, kAllPairs, what);
    // A zero row of A (row 5) or of B (row 1): v is exactly 0, so C is
    // exactly beta * C0, a single rounding on either side.
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        if (aScale[i] != 0.0 && bScale[j] != 0.0) continue;
        const std::size_t ij = i * N + j;
        EXPECT_EQ(c[ij], beta * g.C0[ij]) << "zero row: C[" << i << "," << j << "]";
      }
    }
  }
}

// --- OzakiScratchBytesTests ----------------------------------------------------------

// The bridge's contract: the int8 digit planes (8 per operand) of both
// operands padded to 16 for one K chunk (at most kOzakiMaxK wide), plus the
// two padded per-row scale vectors. So it covers that much, it is the same for
// every size that pads to the same multiple of 16, it grows across a padding
// edge, and it stops growing at kOzakiMaxK.
TEST(OzakiScratchBytesTests, FollowsTheSixteenPaddingAndTheKCap) {
  const auto pad16 = [](const std::size_t x) { return (x + 15) / 16 * 16; };
  for (const int M : {1, 15, 16, 17, 32, 33}) {
    for (const int N : {1, 15, 16, 17, 32, 33}) {
      for (const int K : {1, 15, 16, 17, 1000, kOzakiMaxK, kOzakiMaxK + 1, 3 * kOzakiMaxK}) {
        SCOPED_TRACE(::testing::Message() << "M=" << M << " N=" << N << " K=" << K);
        const std::size_t bytes = ozakiScratchBytes(M, N, K);
        const std::size_t mp = pad16(static_cast<std::size_t>(M));
        const std::size_t np = pad16(static_cast<std::size_t>(N));
        const std::size_t kp = pad16(static_cast<std::size_t>(std::min(K, kOzakiMaxK)));
        EXPECT_GE(bytes, 8 * (mp + np) * kp + (mp + np) * sizeof(double));
        // Same padded size, same scratch: x and pad16(x) round alike.
        EXPECT_EQ(bytes, ozakiScratchBytes(static_cast<int>(mp), static_cast<int>(np),
                                           static_cast<int>(std::min<std::size_t>(
                                               kp, static_cast<std::size_t>(kOzakiMaxK)))));
      }
    }
  }
  // Across each padding edge, 16 -> 17, it grows.
  EXPECT_LT(ozakiScratchBytes(16, 16, 16), ozakiScratchBytes(17, 16, 16));
  EXPECT_LT(ozakiScratchBytes(16, 16, 16), ozakiScratchBytes(16, 17, 16));
  EXPECT_LT(ozakiScratchBytes(16, 16, 16), ozakiScratchBytes(16, 16, 17));
  // K is capped at one chunk.
  EXPECT_EQ(ozakiScratchBytes(3, 2, kOzakiMaxK), ozakiScratchBytes(3, 2, kOzakiMaxK + 1));
  EXPECT_EQ(ozakiScratchBytes(3, 2, kOzakiMaxK), ozakiScratchBytes(3, 2, 10 * kOzakiMaxK));
}

// Launches on exactly ozakiScratchBytes at each edge of the padding (runOzaki
// checks the guard past it), at a K that leaves the last chunk's padding
// ragged too.
TEST(OzakiScratchBytesTests, ExactScratchSufficesAtPaddingEdges) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  std::uint64_t seed = 6000;
  for (const int M : {15, 16, 17}) {
    for (const int N : {15, 16, 17}) {
      for (const int K : {15, 16, 17}) {
        const std::string what = std::format("M={} N={} K={}", M, N, K);
        SCOPED_TRACE(what);
        const Gemm g = randomGemm(M, N, K, 0.0, seed += 10);
        checkAgainstBound(g, runOzaki(*res, g, kAllPairs), kAllPairs, what);
      }
    }
  }
}

}  // namespace nevpt2::test::ozaki
