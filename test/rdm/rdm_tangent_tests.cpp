// Suites for nevpt2.rdm_build's TANGENT build (src/rdm/rdm_build_tangent.cppm,
// the :tangent partition): buildRdmTangentsDevice, the directional derivative
// of buildRdmsDevice with respect to the CI vector, at CAS(4,4) and CAS(8,8)
// on seeded random states. What they check is that it IS that derivative --
// logic, not an energy:
//
//   RdmTangentDifferenceTests  dm3_dot / f3ca_dot / f3ac_dot against a central
//                              difference of buildRdmsDevice along a STRAIGHT
//                              line through a seeded random direction (exact,
//                              see below), and the tangent is not ~0
//   RdmTangentRichardsonTests  the same against a central difference along the
//                              NORMALIZED path, whose O(h^2) error falls 4x
//                              each time h halves
//   RdmTangentEulerTests       the direction equal to the state gives exactly
//                              twice the plain build (what the demos'
//                              --rdm-tangent checks at the sizes they run)
//   RdmTangentPathTests        every digest / consume / tile count gives one
//                              tangent
//
// WHY TWO DIFFERENCE TESTS, and why the curved path. Every output of this
// build is BILINEAR in the CI vector -- produce is linear, consume is linear,
// the digests multiply two produce outputs, and fdm2/wedge are linear -- so
// each is homogeneous of degree 2, and a central difference along the straight
// line c + h u has NO truncation error at all: the h^2 term cancels and only
// roundoff (amplified by 1/2h) is left. That is the stronger check and it is
// the first suite. It is also why a Richardson table needs a curved path:
// along c(h) = (c + h u)/||c + h u|| with ||c|| = ||u|| = 1 and <c|u> = 0,
// homogeneity gives
//
//   X(c(h)) = [X(c) + h X_dot(c, u) + h^2 X(u)] / (1 + h^2)
//
// so the central difference is exactly X_dot/(1 + h^2) and its error is
// -h^2/(1 + h^2) X_dot -- a genuine second-order error whose ratio across a
// halving is 4 (1 + h^2/4)/(1 + h^2): just under 4, and closer to it with
// every halving. (This is test/rdm_host_dm/'s result for the host dm1/dm2,
// which are quadratic in the same way; it is not re-derived here.) Both
// suites difference buildRdmsDevice itself; neither differentiates anything by
// hand.
//
// The integrals are a seeded random h2e, not a physical ERI: f3ac/f3ca
// contract it linearly, so any h2e drives the same code path, and what is
// checked is a derivative of this build, not an energy. The approximate
// digests (--cublas, --ozaki) and the fused GEMM have no tangent path and are
// refused at flag parsing, so only the emitted and native-fp64 BLAS digests
// appear here.
//
// Tolerances. Every comparison is scaled by the tangent's own largest element:
// the device reassociates each K-sum, so nothing here is compared bit for bit.
// The straight-line difference and the Richardson table are differences of
// independent builds, so they carry the 1/2h roundoff amplification; the
// path-agreement and Euler suites compare two builds of the same form and use
// 1e-12, as test/rdm/rdm_dm3_tests.cpp does for tiling invariance.
//
// REQUIRES_GPU: every case builds on the process's one DeviceResources
// (nevpt2.test.shared_resources).
//
// TU shape: gtest's header FIRST, then `import std;` and the modules.
#include <gtest/gtest.h>

import std;
import nevpt2.link_tables;
import nevpt2.rdm_build;
import nevpt2.test.shared_resources;

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one. One binary
// per suite file, so a name clash here is a link error.
namespace nevpt2::test::rdm_tangent {

// Two builds of the same bilinear form, differing only in summation order.
constexpr double kSameFormTol = 1e-12;
// A central difference of two independent builds: no truncation error along
// the straight line, but the 1/2h amplification of their roundoff. At the
// smallest h used here (1/64) that is ~1e-13 relative; this is three orders
// above it and many orders below any misplaced index.
constexpr double kDifferenceTol = 1e-9;

// The three outputs, downloaded flat (n^6, row-major), in the order the
// messages name them.
constexpr std::array<const char*, 3> kNames{"dm3_dot", "f3ca_dot", "f3ac_dot"};

struct Built {
  std::array<std::vector<double>, 3> t;
};

struct CasCase {
  int64_t norb;
  int64_t nelecA;
  int64_t nelecB;
  std::uint64_t seed;
  // The tile count every suite but RdmTangentPathTests builds at: 3, so the
  // tile loop and the device-resident accumulate run more than once (as the
  // small golden entries do).
  int64_t nTiles;
};

// CAS(4,4): 36 determinants. CAS(8,8): 4900 -- the two sizes the small golden
// entries and the sanitizer tier run.
const std::vector<CasCase>& cases() {
  static const std::vector<CasCase> kCases = {
      {4, 2, 2, 0x5eed'0044u, 3},
      {8, 4, 4, 0x5eed'0088u, 3},
  };
  return kCases;
}

std::string caseName(const CasCase& c) {
  return std::format("CAS({},{})", c.nelecA + c.nelecB, c.norb);
}

int64_t ndetOf(const CasCase& c) {
  return link_tables::num_strings(c.norb, c.nelecA) * link_tables::num_strings(c.norb, c.nelecB);
}

// The process's DeviceResources, or nullptr after recording the creation Error
// as this test's failure (callers ASSERT_NE on it).
const DeviceResources* resourcesOrFail() {
  const auto& res = test::sharedResources();
  if (!res.has_value()) {
    ADD_FAILURE() << "DeviceResources::create failed: " << kindName(res.error().kind) << ": "
                  << res.error().message;
    return nullptr;
  }
  return res->get();
}

// x scaled so <x|x> = 1.
Tensor normalized(Tensor x) {
  double norm2 = 0.0;
  for (int64_t i = 0; i < x.size(); ++i) norm2 += x.flat(i) * x.flat(i);
  const double inv = 1.0 / std::sqrt(norm2);
  for (int64_t i = 0; i < x.size(); ++i) x.flatRef(i) *= inv;
  return x;
}

// A seeded random CI vector of shape (na, nb), normalized to 1.
Tensor randomCi(const CasCase& c) {
  const int64_t na = link_tables::num_strings(c.norb, c.nelecA);
  const int64_t nb = link_tables::num_strings(c.norb, c.nelecB);
  Tensor ci({na, nb});
  std::mt19937_64 gen(c.seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int64_t i = 0; i < ci.size(); ++i) ci.flatRef(i) = dist(gen);
  return normalized(std::move(ci));
}

// A seeded random active h2e (n^4): see the header for why it need not be a
// physical ERI.
Tensor randomH2e(const CasCase& c) {
  Tensor h2e({c.norb, c.norb, c.norb, c.norb});
  std::mt19937_64 gen(c.seed ^ 0xe21'0001u);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int64_t i = 0; i < h2e.size(); ++i) h2e.flatRef(i) = dist(gen);
  return h2e;
}

// A seeded random direction in ci's space, projected against ci and
// normalized: the <c|u> = 0, ||u|| = 1 the Richardson identity assumes.
Tensor direction(const Tensor& ci, std::uint64_t seed) {
  Tensor u(ci.dims());
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int64_t i = 0; i < u.size(); ++i) u.flatRef(i) = dist(gen);
  double overlap = 0.0;
  for (int64_t i = 0; i < u.size(); ++i) overlap += ci.flat(i) * u.flat(i);
  for (int64_t i = 0; i < u.size(); ++i) u.flatRef(i) -= overlap * ci.flat(i);
  return normalized(std::move(u));
}

// ci + h u, same shape.
Tensor step(const Tensor& ci, double h, const Tensor& u) {
  Tensor out = ci;
  for (int64_t i = 0; i < out.size(); ++i) out.flatRef(i) += h * u.flat(i);
  return out;
}

// The emitted digest unless `blasDigest`, and the BLAS consume unless
// `consumeEmitted`: the two exact digests, on both backends (HIP would
// otherwise default to the BLAS one).
RdmBuildOptions options(int64_t nTiles, bool blasDigest, bool consumeEmitted) {
  RdmBuildOptions opt;
  opt.nTiles = nTiles;
  opt.cublas = false;
  opt.ozaki = false;
  opt.fusedDigest = false;
  opt.blasDigest = blasDigest;
  opt.consumeGemm = !consumeEmitted;
  return opt;
}

// The three results read back flat. downloadTensor synchronizes the stream
// itself, so the build's last kernel is complete before these are read.
Built download(const RdmBuildResult& r, const DeviceResources& res) {
  Built b;
  b.t[0] = downloadTensor(r.dm3, res.stream()).data();
  b.t[1] = downloadTensor(r.f3ca, res.stream()).data();
  b.t[2] = downloadTensor(r.f3ac, res.stream()).data();
  return b;
}

Built plainBuild(const CasCase& c, const Tensor& ci, const Tensor& h2e,
                 const RdmBuildOptions& opt, const DeviceResources& res) {
  return download(buildRdmsDevice(opt, ci, c.norb, c.nelecA, c.nelecB, h2e, res), res);
}

Built tangentBuild(const CasCase& c, const Tensor& ci, const Tensor& u, const Tensor& h2e,
                   const RdmBuildOptions& opt, const DeviceResources& res) {
  return download(buildRdmTangentsDevice(opt, ci, u, c.norb, c.nelecA, c.nelecB, h2e, res), res);
}

double maxAbs(const Built& b) {
  double m = 0.0;
  for (const std::vector<double>& v : b.t)
    for (double x : v) m = std::max(m, std::abs(x));
  return m;
}

// max |a - b| over one of the three tensors.
double maxDiff(const std::vector<double>& a, const std::vector<double>& b) {
  EXPECT_EQ(a.size(), b.size());
  double m = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

// (f(ci + h u) - f(ci - h u)) / 2h, with f the PLAIN build. `normalize` walks
// the unit sphere instead of the straight line (see the header).
Built centralDifference(const CasCase& c, const Tensor& ci, const Tensor& u, const Tensor& h2e,
                        double h, bool normalize, const RdmBuildOptions& opt,
                        const DeviceResources& res) {
  Tensor plus = step(ci, h, u);
  Tensor minus = step(ci, -h, u);
  if (normalize) {
    plus = normalized(std::move(plus));
    minus = normalized(std::move(minus));
  }
  const Built hi = plainBuild(c, plus, h2e, opt, res);
  const Built lo = plainBuild(c, minus, h2e, opt, res);
  Built out;
  const double scale = 0.5 / h;
  for (std::size_t k = 0; k < out.t.size(); ++k) {
    out.t[k].resize(hi.t[k].size());
    for (std::size_t i = 0; i < out.t[k].size(); ++i)
      out.t[k][i] = (hi.t[k][i] - lo.t[k][i]) * scale;
  }
  return out;
}

// --- RdmTangentDifferenceTests -------------------------------------------------

TEST(RdmTangentDifferenceTests, MatchesAStraightLineCentralDifferenceToRoundoff) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  for (const CasCase& c : cases()) {
    SCOPED_TRACE(caseName(c));
    const RdmBuildOptions opt = options(c.nTiles, /*blasDigest=*/false, /*consumeEmitted=*/false);
    const Tensor ci = randomCi(c);
    const Tensor h2e = randomH2e(c);
    const Tensor u = direction(ci, c.seed ^ 0xd1'5ec7u);
    const Built dot = tangentBuild(c, ci, u, h2e, opt, *res);
    // Not itself ~0, which would make the comparison below vacuous.
    const double scale = std::max(1.0, maxAbs(dot));
    EXPECT_GT(maxAbs(dot), 1e-3);
    // No truncation error to shrink (the forms are quadratic), so the gap is
    // roundoff amplified by 1/2h -- it does not fall with h, and the tolerance
    // is sized for the smaller h.
    for (const double h : {0.125, 0.015625}) {
      SCOPED_TRACE(std::format("h={}", h));
      const Built cd =
          centralDifference(c, ci, u, h2e, h, /*normalize=*/false, opt, *res);
      for (std::size_t k = 0; k < kNames.size(); ++k)
        EXPECT_LT(maxDiff(cd.t[k], dot.t[k]), kDifferenceTol * scale) << kNames[k];
    }
  }
}

// --- RdmTangentRichardsonTests -------------------------------------------------

TEST(RdmTangentRichardsonTests, NormalizedPathErrorFallsFourfoldPerHalving) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  // h from 1/8 down to 1/64, so the relative error runs over h^2 = 1.6e-2 down
  // to 2.4e-4 -- every value far above the roundoff floor of the difference
  // itself (eps |X| / 2h).
  constexpr std::array<double, 4> kSteps{0.125, 0.0625, 0.03125, 0.015625};
  for (const CasCase& c : cases()) {
    SCOPED_TRACE(caseName(c));
    const RdmBuildOptions opt = options(c.nTiles, /*blasDigest=*/false, /*consumeEmitted=*/false);
    const Tensor ci = randomCi(c);
    const Tensor h2e = randomH2e(c);
    const Tensor u = direction(ci, c.seed ^ 0xd1'5ec7u);
    const Built dot = tangentBuild(c, ci, u, h2e, opt, *res);

    std::array<std::array<double, kSteps.size()>, 3> err{};
    for (std::size_t i = 0; i < kSteps.size(); ++i) {
      const Built cd =
          centralDifference(c, ci, u, h2e, kSteps[i], /*normalize=*/true, opt, *res);
      for (std::size_t k = 0; k < kNames.size(); ++k) err[k][i] = maxDiff(cd.t[k], dot.t[k]);
    }
    // The error is -h^2/(1+h^2) X_dot elementwise, so each ratio is
    // 4 (1 + h^2/4)/(1 + h^2) -- just under 4 at the first halving, within a
    // thousandth of it at the last.
    for (std::size_t k = 0; k < kNames.size(); ++k) {
      SCOPED_TRACE(kNames[k]);
      for (std::size_t i = 0; i + 1 < kSteps.size(); ++i) {
        SCOPED_TRACE(std::format("h {} -> {}", kSteps[i], kSteps[i + 1]));
        ASSERT_GT(err[k][i + 1], 0.0);
        EXPECT_NEAR(err[k][i] / err[k][i + 1], 4.0, 0.2) << "Richardson ratio";
      }
      // And the h^2 coefficient is the tangent itself: err(h) ~ h^2 |X_dot|.
      double worst = 0.0;
      for (double x : dot.t[k]) worst = std::max(worst, std::abs(x));
      const double h = kSteps.back();
      EXPECT_NEAR(err[k].back(), h * h * worst, 0.05 * h * h * worst);
    }
  }
}

// --- RdmTangentEulerTests ------------------------------------------------------

TEST(RdmTangentEulerTests, DirectionEqualToTheStateGivesTwiceThePlainBuild) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  // Every output is homogeneous of degree 2 in the CI vector, so Euler's
  // identity holds exactly: X_dot(c, c) = 2 X(c). This is the identity the
  // demos' --rdm-tangent checks at CAS(10,10) and up.
  for (const CasCase& c : cases()) {
    SCOPED_TRACE(caseName(c));
    const RdmBuildOptions opt = options(c.nTiles, /*blasDigest=*/false, /*consumeEmitted=*/false);
    const Tensor ci = randomCi(c);
    const Tensor h2e = randomH2e(c);
    const Built plain = plainBuild(c, ci, h2e, opt, *res);
    const Built dot = tangentBuild(c, ci, ci, h2e, opt, *res);
    const double tol = kSameFormTol * std::max(1.0, maxAbs(dot));
    for (std::size_t k = 0; k < kNames.size(); ++k) {
      SCOPED_TRACE(kNames[k]);
      ASSERT_EQ(dot.t[k].size(), plain.t[k].size());
      double worst = 0.0;
      for (std::size_t i = 0; i < dot.t[k].size(); ++i)
        worst = std::max(worst, std::abs(dot.t[k][i] - 2.0 * plain.t[k][i]));
      EXPECT_LT(worst, tol) << "worst |dot - 2 x build|";
    }
  }
}

// --- RdmTangentPathTests -------------------------------------------------------

TEST(RdmTangentPathTests, EveryDigestConsumeAndTileCountGivesTheSameTangent) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  // CAS(4,4) only: this varies the build's own knobs, which are
  // CAS-independent, and 36 determinants make 5 tiles leave a short last one
  // (checked, not assumed).
  const CasCase c = cases().front();
  const Tensor ci = randomCi(c);
  const Tensor h2e = randomH2e(c);
  const Tensor u = direction(ci, c.seed ^ 0xd1'5ec7u);
  const Built ref = tangentBuild(c, ci, u, h2e,
                                 options(1, /*blasDigest=*/false, /*consumeEmitted=*/false), *res);
  const double tol = kSameFormTol * std::max(1.0, maxAbs(ref));

  const int64_t ndet = ndetOf(c);
  bool shortLastTile = false;
  struct Variant {
    int64_t nTiles;
    bool blasDigest;
    bool consumeEmitted;
  };
  const std::array<Variant, 6> kVariants{
      Variant{2, false, false}, Variant{3, false, false}, Variant{5, false, false},
      Variant{3, true, false},  Variant{3, false, true},  Variant{5, true, true}};
  for (const Variant& v : kVariants) {
    SCOPED_TRACE(std::format("tiles={} blasDigest={} consumeEmitted={}", v.nTiles, v.blasDigest,
                             v.consumeEmitted));
    shortLastTile |= (ndet % ((ndet + v.nTiles - 1) / v.nTiles)) != 0;
    const Built got =
        tangentBuild(c, ci, u, h2e, options(v.nTiles, v.blasDigest, v.consumeEmitted), *res);
    for (std::size_t k = 0; k < kNames.size(); ++k)
      EXPECT_LT(maxDiff(got.t[k], ref.t[k]), tol) << kNames[k];
  }
  EXPECT_TRUE(shortLastTile) << "no tile count leaves a short last tile";
}

}  // namespace nevpt2::test::rdm_tangent
