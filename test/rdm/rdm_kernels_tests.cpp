// Suites for the RDM build's linked-in device libraries: the three
// launchers rdm_build.cpp calls through a *_bridge.h, each on seeded random
// small inputs (norb 4..6) against a host loop written from the contract the
// bridge header states -- not from the kernel.
//
//   AccumulateInplaceTests   accumulateInplace (rdm_accumulate_bridge.h):
//                              acc[i] += part[i], i in [0, n)
//   F3ScatterCaTests         f3ScatterCa (f3_scatter_bridge.h):
//                              f3[r*n^2 + fr*norb + a] += C[r*n^2 + a*norb + fr]
//   F3DigestAccumulateTests  f3DigestAccumulate (f3_digest_bridge.h):
//                              ca: f3[r, fr, a] += sum_K W[a*n + fr, K] * L2[r, K]
//                              ac: f3[r, a, fr] += sum_K W[a*n + fr, K] * L2[r, K]
//
// This checks the kernels, not PySCF agreement: full-RDM and energy numbers
// stay with the golden demos.
//
// REQUIRES_GPU: every case launches on the process's one DeviceResources
// stream (nevpt2.test.shared_resources).
//
// Tolerances, per launcher:
//  - accumulateInplace and f3ScatterCa compute ONE double addition per output
//    element (a += b, with a and b read exactly), which IEEE 754 rounds the
//    same way on any conforming device. So they are compared EXACTLY: a
//    tolerance would hide an element added twice (if it were small) or the
//    fr/a transpose landing on a near-equal value.
//  - f3DigestAccumulate is a width-long dot product per element, which the
//    device may contract into FMAs and the host may not (or vice versa), so
//    the two need not agree bit for bit. The bound is the standard one for a
//    recursively summed dot product, |err| <= gamma_K * sum_K |W||L2|, plus
//    the one rounding of the final +=: with K <= 40 terms and eps = 2^-53,
//    gamma_K < 5e-15. The check uses 1e-13 * (|f3_0| + sum_K |W*L2|) per
//    element, ~20x that bound, and still ~10^9 below the size of any wrong
//    index (inputs are O(1), so a misplaced or dropped term is O(1e-2) or
//    more).
//
// TU shape: gtest's header and runtime.h FIRST (runtime.h before the imports,
// as a module's global module fragment would have it, so each bridge's own
// #include of it is a #pragma once no-op), then the imports, then the bridges,
// which need nevpt2.wwr's wwrError_t. A plain TU is already in the global
// module, so the bridges need no extern "C++" here.
#include <gtest/gtest.h>

#include <runtime.h>

import std;
import nevpt2.wwr;
import nevpt2.test.shared_resources;

#include "rdm/f3_digest_bridge.h"
#include "rdm/f3_scatter_bridge.h"
#include "rdm/rdm_accumulate_bridge.h"

namespace nevpt2 {
namespace {

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

// A device copy of `host`, drawn from res's pool and filled on its stream.
DeviceBuffer<double> upload(const std::vector<double>& host, const DeviceResources& res) {
  DeviceBuffer<double> buf(host.size(), res.shared_from_this());
  gpuCheck(wwrMemcpyAsync(buf.data(), host.data(), buf.size_bytes(), wwrMemcpyHostToDevice,
                          res.stream()));
  return buf;
}

// A whole DeviceBuffer read back on res's stream.
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

// Exact per-element equality, reported as the first few mismatches plus a
// count, so a wholly wrong n^6 result does not print 46656 lines.
void expectExact(const std::vector<double>& got, const std::vector<double>& want,
                 const std::string_view what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  std::size_t bad = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] == want[i]) continue;
    if (++bad <= 5) {
      ADD_FAILURE() << what << ": element " << i << " got " << got[i] << ", want " << want[i];
    }
  }
  EXPECT_EQ(bad, 0u) << what << ": mismatched elements";
}

// |got - want| <= rtol * scale[i] per element, reported like expectExact.
void expectWithin(const std::vector<double>& got, const std::vector<double>& want,
                  const std::vector<double>& scale, const double rtol,
                  const std::string_view what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  ASSERT_EQ(scale.size(), want.size()) << what;
  std::size_t bad = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::abs(got[i] - want[i]) <= rtol * scale[i]) continue;
    if (++bad <= 5) {
      ADD_FAILURE() << what << ": element " << i << " got " << got[i] << ", want " << want[i]
                    << " (|diff| " << std::abs(got[i] - want[i]) << ", allowed "
                    << rtol * scale[i] << ")";
    }
  }
  EXPECT_EQ(bad, 0u) << what << ": elements outside tolerance";
}

constexpr std::array<int, 3> kNorbs{4, 5, 6};

std::size_t pow2(const int norb) { return static_cast<std::size_t>(norb) * norb; }

// --- AccumulateInplaceTests ---------------------------------------------------

// acc += part over an n^6 accumulator, the size the RDM build uses it at, and
// part is left alone.
TEST(AccumulateInplaceTests, AddsPartIntoAccN6) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  for (const int norb : kNorbs) {
    SCOPED_TRACE(::testing::Message() << "norb " << norb);
    const std::size_t n = pow2(norb) * pow2(norb) * pow2(norb);
    const std::vector<double> acc0 = randomVector(n, 1000 + norb);
    const std::vector<double> part = randomVector(n, 2000 + norb);

    std::vector<double> want(n);
    for (std::size_t i = 0; i < n; ++i) want[i] = acc0[i] + part[i];

    DeviceBuffer<double> dAcc = upload(acc0, *res);
    const DeviceBuffer<double> dPart = upload(part, *res);
    gpuCheck(device::accumulateInplace(res->stream(), dAcc.data(), dPart.data(),
                                       static_cast<long long>(n)));
    expectExact(download(dAcc, *res), want, "acc");
    expectExact(download(dPart, *res), part, "part (must be unchanged)");
  }
}

// It accumulates: a second launch adds part again, as the tile sweep relies
// on (one launch per tile into the same accumulator).
TEST(AccumulateInplaceTests, RepeatedLaunchesAccumulate) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const std::size_t n = 4096;
  const std::vector<double> acc0 = randomVector(n, 3001);
  const std::vector<double> part = randomVector(n, 3002);

  std::vector<double> want(n);
  for (std::size_t i = 0; i < n; ++i) want[i] = (acc0[i] + part[i]) + part[i];

  DeviceBuffer<double> dAcc = upload(acc0, *res);
  const DeviceBuffer<double> dPart = upload(part, *res);
  for (int rep = 0; rep < 2; ++rep) {
    gpuCheck(device::accumulateInplace(res->stream(), dAcc.data(), dPart.data(),
                                       static_cast<long long>(n)));
  }
  expectExact(download(dAcc, *res), want, "acc after two launches");
}

// Exactly [0, n): elements past n are not touched, for counts that are not a
// multiple of any block size (and 1).
TEST(AccumulateInplaceTests, TouchesOnlyTheFirstN) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const std::size_t cap = 1200;
  for (const std::size_t n : {std::size_t{1}, std::size_t{127}, std::size_t{129},
                              std::size_t{1025}}) {
    SCOPED_TRACE(::testing::Message() << "n " << n);
    const std::vector<double> acc0 = randomVector(cap, 4000 + n);
    const std::vector<double> part = randomVector(cap, 5000 + n);

    std::vector<double> want = acc0;
    for (std::size_t i = 0; i < n; ++i) want[i] = acc0[i] + part[i];

    DeviceBuffer<double> dAcc = upload(acc0, *res);
    const DeviceBuffer<double> dPart = upload(part, *res);
    gpuCheck(device::accumulateInplace(res->stream(), dAcc.data(), dPart.data(),
                                       static_cast<long long>(n)));
    expectExact(download(dAcc, *res), want, "acc");
  }
}

// --- F3ScatterCaTests ---------------------------------------------------------

// The host reference: for every row r in [0, n^4) and a, fr in [0, norb),
// f3[r*n^2 + fr*norb + a] += C[r*n^2 + a*norb + fr].
std::vector<double> scatterCaReference(const std::vector<double>& c, std::vector<double> f3,
                                       const int norb) {
  const std::size_t n2 = pow2(norb);
  const std::size_t n4 = n2 * n2;
  for (std::size_t r = 0; r < n4; ++r) {
    for (int a = 0; a < norb; ++a) {
      for (int fr = 0; fr < norb; ++fr) {
        f3[r * n2 + static_cast<std::size_t>(fr * norb + a)] +=
            c[r * n2 + static_cast<std::size_t>(a * norb + fr)];
      }
    }
  }
  return f3;
}

TEST(F3ScatterCaTests, TransposesTheLastTwoAxesAndAccumulates) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  for (const int norb : kNorbs) {
    SCOPED_TRACE(::testing::Message() << "norb " << norb);
    const std::size_t n6 = pow2(norb) * pow2(norb) * pow2(norb);
    const std::vector<double> c = randomVector(n6, 6000 + norb);
    const std::vector<double> f30 = randomVector(n6, 7000 + norb);

    const DeviceBuffer<double> dC = upload(c, *res);
    DeviceBuffer<double> dF3 = upload(f30, *res);
    gpuCheck(device::f3ScatterCa(res->stream(), dC.data(), dF3.data(), norb));
    expectExact(download(dF3, *res), scatterCaReference(c, f30, norb), "f3");
    expectExact(download(dC, *res), c, "C (must be unchanged)");
  }
}

// From a zero accumulator the result is C with each row's (a, fr) block
// transposed -- so, unlike the random case, a transpose that is accidentally
// the identity cannot hide behind a symmetric input: C here is not symmetric
// in (a, fr) anywhere (every element distinct).
TEST(F3ScatterCaTests, FromZeroIsAPureTranspose) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const int norb = 5;
  const std::size_t n2 = pow2(norb);
  const std::size_t n6 = n2 * n2 * n2;
  std::vector<double> c(n6);
  for (std::size_t i = 0; i < n6; ++i) c[i] = static_cast<double>(i) + 1.0;  // exact

  std::vector<double> want(n6);
  for (std::size_t r = 0; r < n2 * n2; ++r) {
    for (int a = 0; a < norb; ++a) {
      for (int fr = 0; fr < norb; ++fr) {
        want[r * n2 + static_cast<std::size_t>(fr * norb + a)] =
            c[r * n2 + static_cast<std::size_t>(a * norb + fr)];
      }
    }
  }

  const DeviceBuffer<double> dC = upload(c, *res);
  DeviceBuffer<double> dF3(n6, res->shared_from_this());  // zero-filled
  gpuCheck(device::f3ScatterCa(res->stream(), dC.data(), dF3.data(), norb));
  expectExact(download(dF3, *res), want, "f3");
}

// --- F3DigestAccumulateTests --------------------------------------------------

// The host reference and the per-element error scale for one launch:
//   want[r, last2]  = f3_0[r, last2] + sum_K W[a*n + fr, K] * L2[r, K]
//   scale[r, last2] = |f3_0[r, last2]| + sum_K |W[a*n + fr, K] * L2[r, K]|
// with last2 = fr*n + a for ca (order 0) and a*n + fr for ac (order 1).
struct DigestReference {
  std::vector<double> want;
  std::vector<double> scale;
};

DigestReference digestReference(const int order, const std::vector<double>& w,
                                const std::vector<double>& l2, const std::vector<double>& f30,
                                const int norb, const int width) {
  const std::size_t n2 = pow2(norb);
  const std::size_t n4 = n2 * n2;
  const std::size_t k = static_cast<std::size_t>(width);
  DigestReference ref{f30, std::vector<double>(f30.size())};
  for (std::size_t i = 0; i < f30.size(); ++i) ref.scale[i] = std::abs(f30[i]);
  for (std::size_t r = 0; r < n4; ++r) {
    for (int a = 0; a < norb; ++a) {
      for (int fr = 0; fr < norb; ++fr) {
        const std::size_t wrow = static_cast<std::size_t>(a * norb + fr);
        double sum = 0.0;
        double mag = 0.0;
        for (std::size_t kk = 0; kk < k; ++kk) {
          const double term = w[wrow * k + kk] * l2[r * k + kk];
          sum += term;
          mag += std::abs(term);
        }
        const std::size_t last2 =
            static_cast<std::size_t>(order == 0 ? fr * norb + a : a * norb + fr);
        ref.want[r * n2 + last2] += sum;
        ref.scale[r * n2 + last2] += mag;
      }
    }
  }
  return ref;
}

// See the file comment: ~20x the recursive-summation bound at width 40.
constexpr double kDigestRtol = 1e-13;

// Widths straddling the kernel's 16-wide K tile: under one tile, exactly one,
// one past, and several with a ragged last tile.
constexpr std::array<int, 5> kWidths{1, 7, 16, 17, 40};

void runDigest(const DeviceResources& res, const int order, const char* name) {
  for (const int norb : kNorbs) {
    for (const int width : kWidths) {
      SCOPED_TRACE(::testing::Message() << name << " norb " << norb << " width " << width);
      const std::size_t n2 = pow2(norb);
      const std::size_t n4 = n2 * n2;
      const std::size_t n6 = n4 * n2;
      const std::size_t k = static_cast<std::size_t>(width);
      const std::uint64_t seed = 10000u * static_cast<std::uint64_t>(order + 1) +
                                 100u * static_cast<std::uint64_t>(norb) +
                                 static_cast<std::uint64_t>(width);
      const std::vector<double> w = randomVector(n2 * k, seed + 1);
      const std::vector<double> l2 = randomVector(n4 * k, seed + 2);
      const std::vector<double> f30 = randomVector(n6, seed + 3);
      const DigestReference ref = digestReference(order, w, l2, f30, norb, width);

      const DeviceBuffer<double> dW = upload(w, res);
      const DeviceBuffer<double> dL2 = upload(l2, res);
      DeviceBuffer<double> dF3 = upload(f30, res);
      gpuCheck(device::f3DigestAccumulate(res.stream(), order, dW.data(), dL2.data(),
                                          dF3.data(), norb, width));
      expectWithin(download(dF3, res), ref.want, ref.scale, kDigestRtol, "f3");
      expectExact(download(dW, res), w, "W (must be unchanged)");
      expectExact(download(dL2, res), l2, "L2 (must be unchanged)");
    }
  }
}

TEST(F3DigestAccumulateTests, CaOrderMatchesHostReference) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  runDigest(*res, 0, "ca");
}

TEST(F3DigestAccumulateTests, AcOrderMatchesHostReference) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  runDigest(*res, 1, "ac");
}

// The two orders are the same GEMM landing on transposed (a, fr) slots: from
// a zero accumulator, ca[r, fr, a] == ac[r, a, fr] bit for bit (each element
// is the same device sum, written to a different place).
TEST(F3DigestAccumulateTests, CaAndAcAreTransposesOfEachOther) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const int norb = 6;
  const int width = 23;
  const std::size_t n2 = pow2(norb);
  const std::size_t n4 = n2 * n2;
  const std::size_t n6 = n4 * n2;
  const std::size_t k = static_cast<std::size_t>(width);
  const std::vector<double> w = randomVector(n2 * k, 20001);
  const std::vector<double> l2 = randomVector(n4 * k, 20002);

  const DeviceBuffer<double> dW = upload(w, *res);
  const DeviceBuffer<double> dL2 = upload(l2, *res);
  DeviceBuffer<double> dCa(n6, res->shared_from_this());  // zero-filled
  DeviceBuffer<double> dAc(n6, res->shared_from_this());
  gpuCheck(device::f3DigestAccumulate(res->stream(), 0, dW.data(), dL2.data(), dCa.data(), norb,
                                      width));
  gpuCheck(device::f3DigestAccumulate(res->stream(), 1, dW.data(), dL2.data(), dAc.data(), norb,
                                      width));
  const std::vector<double> ca = download(dCa, *res);
  const std::vector<double> ac = download(dAc, *res);

  std::vector<double> acTransposed(n6);
  for (std::size_t r = 0; r < n4; ++r) {
    for (int a = 0; a < norb; ++a) {
      for (int fr = 0; fr < norb; ++fr) {
        acTransposed[r * n2 + static_cast<std::size_t>(fr * norb + a)] =
            ac[r * n2 + static_cast<std::size_t>(a * norb + fr)];
      }
    }
  }
  expectExact(ca, acTransposed, "ca vs transposed ac");
}

// It accumulates (+=), not overwrites: two launches into the same accumulator
// give f3_0 + 2 * GEMM, within the digest tolerance doubled.
TEST(F3DigestAccumulateTests, RepeatedLaunchesAccumulate) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const int norb = 4;
  const int width = 33;
  const std::size_t n2 = pow2(norb);
  const std::size_t n4 = n2 * n2;
  const std::size_t n6 = n4 * n2;
  const std::size_t k = static_cast<std::size_t>(width);
  const std::vector<double> w = randomVector(n2 * k, 30001);
  const std::vector<double> l2 = randomVector(n4 * k, 30002);
  const std::vector<double> f30 = randomVector(n6, 30003);
  const DigestReference once = digestReference(0, w, l2, f30, norb, width);
  const DigestReference twice = digestReference(0, w, l2, once.want, norb, width);

  const DeviceBuffer<double> dW = upload(w, *res);
  const DeviceBuffer<double> dL2 = upload(l2, *res);
  DeviceBuffer<double> dF3 = upload(f30, *res);
  for (int rep = 0; rep < 2; ++rep) {
    gpuCheck(device::f3DigestAccumulate(res->stream(), 0, dW.data(), dL2.data(), dF3.data(),
                                        norb, width));
  }
  expectWithin(download(dF3, *res), twice.want, twice.scale, 2 * kDigestRtol,
               "f3 after two launches");
}

}  // namespace
}  // namespace nevpt2
