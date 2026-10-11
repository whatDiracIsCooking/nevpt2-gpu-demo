// Suites for nevpt2.rdm_host_dm (src/rdm/rdm_host_dm.cppm) -- the host rebuild
// of the active-space dm1/dm2 from a CI vector, and their tangents.
//
// Host-only ON PURPOSE: the module has no device code, so this binary is not
// REQUIRES_GPU and runs with no card visible. Nothing here checks an energy;
// the numerical claims about energies stay with the golden demo tier.
//
//   HostDm12GoldenTests      the rebuild of each committed golden's dm1/dm2
//                            from the `ci` beside them, to 1e-12, at
//                            CAS(4,4), CAS(8,8) and CAS(10,10) -- plus
//                            Tr dm1 = N on each
//   HostDm12TangentTests     dm12Dot against a central difference of the
//                            builder along a STRAIGHT line (exact, see
//                            below), the degree-2 Euler identity
//                            dm12Dot(c, c) = 2 dm12(c), and linearity in the
//                            direction
//   HostDm12RichardsonTests  dm12Dot against a central difference along the
//                            NORMALIZED path, whose O(h^2) error falls 4x
//                            each time h halves
//   HostDm12CreateTests      an out-of-range active space is an
//                            InvalidConfig Error, never an abort; the
//                            accessors of a good builder
//   HostDm12DeathTest        a ci (or dir) of the wrong shape aborts
//
// WHY TWO DIFFERENCE TESTS. dm1 and dm2 are exactly quadratic in the CI
// vector, so a central difference along the straight line c + h u has NO
// truncation error at all: the h^2 term cancels and only roundoff (which
// grows as h shrinks, through the 1/2h) is left. That is the stronger check,
// and it is the first suite. It is also why a Richardson table needs a curved
// path: along c(h) = (c + h u)/||c + h u|| with ||c|| = ||u|| = 1 and
// <c|u> = 0, homogeneity gives
//
//   dm(c(h)) = [dm(c) + h dm_dot(c, u) + h^2 dm(u)] / (1 + h^2)
//
// so the central difference is exactly dm_dot/(1 + h^2) and its error is
// -h^2/(1 + h^2) dm_dot -- a genuine second-order error, whose ratio across a
// halving is 4 (1 + h^2/4)/(1 + h^2): just under 4, and closer to it with
// every halving. Both suites difference the builder itself; neither
// differentiates anything by hand.
//
// The committed goldens are read only, and their paths come from CMake, not
// from this source.
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.golden;
import nevpt2.rdm_host_dm;

#ifndef NEVPT2_TEST_GOLDEN_CAS44
#error "NEVPT2_TEST_GOLDEN_CAS44 must be defined by test/rdm_host_dm/CMakeLists.txt"
#endif
#ifndef NEVPT2_TEST_GOLDEN_CAS88
#error "NEVPT2_TEST_GOLDEN_CAS88 must be defined by test/rdm_host_dm/CMakeLists.txt"
#endif
#ifndef NEVPT2_TEST_GOLDEN_CAS1010
#error "NEVPT2_TEST_GOLDEN_CAS1010 must be defined by test/rdm_host_dm/CMakeLists.txt"
#endif

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one. One
// binary per suite file, so a name clash here is a link error.
namespace nevpt2::test::rdm_host_dm {

// The tolerance for the rebuild against the golden's own arrays. The two
// differ only by summation order (this walks determinants, PySCF's
// make_dm123 kernel walks them its own way), so the gap is a few ulps of the
// largest element -- orders of magnitude inside this bound at every case
// here.
constexpr double kGoldenTol = 1e-12;

constexpr std::string_view kGoldenCas44 = NEVPT2_TEST_GOLDEN_CAS44;
constexpr std::string_view kGoldenCas88 = NEVPT2_TEST_GOLDEN_CAS88;
constexpr std::string_view kGoldenCas1010 = NEVPT2_TEST_GOLDEN_CAS1010;

// Counts |got - want| > tol, reporting the first few, so a wholly wrong n^4
// tensor does not print 10000 lines.
class Mismatches {
 public:
  Mismatches(std::string what, double tol) : what_(std::move(what)), tol_(tol) {}
  void check(double got, double want, std::string_view where) {
    const double d = std::abs(got - want);
    worst_ = std::max(worst_, d);
    if (d <= tol_) return;
    if (++bad_ <= 5) {
      ADD_FAILURE() << what_ << " at " << where << ": got " << got << ", want " << want
                    << " (|diff| " << d << " > " << tol_ << ")";
    }
  }
  void expectNone() const {
    EXPECT_EQ(bad_, 0u) << what_ << ": mismatched elements (worst |diff| " << worst_ << ")";
  }

 private:
  std::string what_;
  double tol_;
  double worst_ = 0.0;
  std::size_t bad_ = 0;
};

// A golden file and the builder for its active space.
struct Loaded {
  GoldenFile golden;
  HostDm12Builder builder;
};

// `path` loaded and its active space built, or nullopt after recording the
// failure (callers ASSERT_TRUE on it). The golden file is the test's input:
// every way it can be missing or malformed is an Error, never an abort.
std::optional<Loaded> load(std::string_view path) {
  Result<GoldenFile> golden = loadGolden(std::string(path));
  if (!golden.has_value()) {
    ADD_FAILURE() << "loadGolden(" << path << ") failed: " << kindName(golden.error().kind) << ": "
                  << golden.error().message;
    return std::nullopt;
  }
  if (const Status s = golden->require({"ci", "dm1", "dm2"}); !s.has_value()) {
    ADD_FAILURE() << path << ": " << s.error().message;
    return std::nullopt;
  }
  Result<HostDm12Builder> builder =
      HostDm12Builder::create(golden->ncas, golden->nelecA, golden->nelecB);
  if (!builder.has_value()) {
    ADD_FAILURE() << path << ": HostDm12Builder::create failed: "
                  << kindName(builder.error().kind) << ": " << builder.error().message;
    return std::nullopt;
  }
  return Loaded{std::move(*golden), std::move(*builder)};
}

double maxAbs(const Tensor& x) {
  double m = 0.0;
  for (int64_t i = 0; i < x.size(); ++i) m = std::max(m, std::abs(x.flat(i)));
  return m;
}

// max |a - b| over two same-shaped tensors.
double maxDiff(const Tensor& a, const Tensor& b) {
  EXPECT_EQ(a.size(), b.size());
  double m = 0.0;
  for (int64_t i = 0; i < std::min(a.size(), b.size()); ++i)
    m = std::max(m, std::abs(a.flat(i) - b.flat(i)));
  return m;
}

// x scaled so <x|x> = 1.
Tensor normalized(Tensor x) {
  double norm2 = 0.0;
  for (int64_t i = 0; i < x.size(); ++i) norm2 += x.flat(i) * x.flat(i);
  const double inv = 1.0 / std::sqrt(norm2);
  for (int64_t i = 0; i < x.size(); ++i) x.flatRef(i) *= inv;
  return x;
}

// ci + h u, same shape.
Tensor step(const Tensor& ci, double h, const Tensor& u) {
  Tensor out = ci;
  for (int64_t i = 0; i < out.size(); ++i) out.flatRef(i) += h * u.flat(i);
  return out;
}

// A seeded random direction in ci's space, projected against ci and
// normalized: the <c|u> = 0, ||u|| = 1 the Richardson identity above assumes.
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

// (f(ci + h u) - f(ci - h u)) / 2h, with f the builder itself. `normalize`
// walks the unit sphere instead of the straight line (see the header).
HostDm12 centralDifference(const HostDm12Builder& b, const Tensor& ci, const Tensor& u, double h,
                           bool normalize) {
  Tensor plus = step(ci, h, u);
  Tensor minus = step(ci, -h, u);
  if (normalize) {
    plus = normalized(std::move(plus));
    minus = normalized(std::move(minus));
  }
  const HostDm12 hi = b.dm12(plus);
  const HostDm12 lo = b.dm12(minus);
  HostDm12 out{Tensor(hi.dm1.dims()), Tensor(hi.dm2.dims())};
  const double scale = 0.5 / h;
  for (int64_t i = 0; i < out.dm1.size(); ++i)
    out.dm1.flatRef(i) = (hi.dm1.flat(i) - lo.dm1.flat(i)) * scale;
  for (int64_t i = 0; i < out.dm2.size(); ++i)
    out.dm2.flatRef(i) = (hi.dm2.flat(i) - lo.dm2.flat(i)) * scale;
  return out;
}

// The rebuild against one golden's own dm1/dm2, plus Tr dm1 = N.
void expectRebuildMatches(std::string_view path) {
  const std::optional<Loaded> l = load(path);
  ASSERT_TRUE(l.has_value());
  const int64_t n = l->golden.ncas;
  const Tensor& ci = l->golden.get("ci");
  const Tensor& want1 = l->golden.get("dm1");
  const Tensor& want2 = l->golden.get("dm2");
  ASSERT_EQ(ci.size(), l->builder.ndet());
  ASSERT_EQ(want1.size(), n * n);
  ASSERT_EQ(want2.size(), n * n * n * n);

  const HostDm12 got = l->builder.dm12(ci);
  ASSERT_EQ(got.dm1.rank(), 2);
  ASSERT_EQ(got.dm2.rank(), 4);
  EXPECT_EQ(got.dm1.dim(0), n);
  EXPECT_EQ(got.dm2.dim(3), n);

  Mismatches m1("dm1 vs the golden", kGoldenTol);
  for (int64_t p = 0; p < n; ++p)
    for (int64_t q = 0; q < n; ++q)
      m1.check(got.dm1(p, q), want1(p, q), std::format("[{},{}]", p, q));
  m1.expectNone();

  Mismatches m2("dm2 vs the golden", kGoldenTol);
  for (int64_t p = 0; p < n; ++p)
    for (int64_t q = 0; q < n; ++q)
      for (int64_t r = 0; r < n; ++r)
        for (int64_t s = 0; s < n; ++s)
          m2.check(got.dm2(p, q, r, s), want2(p, q, r, s),
                   std::format("[{},{},{},{}]", p, q, r, s));
  m2.expectNone();

  // A normalized CI vector, so the trace is the electron count exactly.
  double trace = 0.0;
  for (int64_t p = 0; p < n; ++p) trace += got.dm1(p, p);
  const double nElec = static_cast<double>(l->golden.nelecA + l->golden.nelecB);
  EXPECT_NEAR(trace, nElec, 1e-12 * nElec) << "Tr dm1";
}

// --- HostDm12GoldenTests -------------------------------------------------------

TEST(HostDm12GoldenTests, RebuildsTheCas44Golden) { expectRebuildMatches(kGoldenCas44); }

TEST(HostDm12GoldenTests, RebuildsTheCas88Golden) { expectRebuildMatches(kGoldenCas88); }

TEST(HostDm12GoldenTests, RebuildsTheCas1010Golden) { expectRebuildMatches(kGoldenCas1010); }

// --- HostDm12TangentTests ------------------------------------------------------

// The two cases the tangent suites run on: CAS(4,4) (36 determinants) and
// CAS(8,8) (4900), each with its own seed.
struct TangentCase {
  std::string_view path;
  std::uint64_t seed;
};

const std::array<TangentCase, 2>& tangentCases() {
  static const std::array<TangentCase, 2> kCases{TangentCase{kGoldenCas44, 0x5eed'0044u},
                                                 TangentCase{kGoldenCas88, 0x5eed'0088u}};
  return kCases;
}

TEST(HostDm12TangentTests, MatchesAStraightLineCentralDifferenceToRoundoff) {
  for (const TangentCase& c : tangentCases()) {
    SCOPED_TRACE(c.path);
    const std::optional<Loaded> l = load(c.path);
    ASSERT_TRUE(l.has_value());
    const Tensor& ci = l->golden.get("ci");
    const Tensor u = direction(ci, c.seed);
    const HostDm12 dot = l->builder.dm12Dot(ci, u);
    // No truncation error to shrink (the forms are quadratic), so the gap is
    // roundoff amplified by 1/2h -- it does not fall with h, and the
    // tolerance is sized for the smaller h.
    for (const double h : {0.125, 0.015625}) {
      SCOPED_TRACE(std::format("h={}", h));
      const HostDm12 cd = centralDifference(l->builder, ci, u, h, /*normalize=*/false);
      EXPECT_LT(maxDiff(cd.dm1, dot.dm1), 1e-9);
      EXPECT_LT(maxDiff(cd.dm2, dot.dm2), 1e-9);
    }
    // ... and the tangent is not itself ~0, which would make the above
    // vacuous.
    EXPECT_GT(maxAbs(dot.dm1), 1e-3);
    EXPECT_GT(maxAbs(dot.dm2), 1e-3);
  }
}

TEST(HostDm12TangentTests, DirectionEqualToTheStateGivesTwiceTheRdms) {
  // dm1 and dm2 are homogeneous of degree 2, so Euler's identity holds
  // exactly: dm_dot(c, c) = 2 dm(c).
  for (const TangentCase& c : tangentCases()) {
    SCOPED_TRACE(c.path);
    const std::optional<Loaded> l = load(c.path);
    ASSERT_TRUE(l.has_value());
    const Tensor& ci = l->golden.get("ci");
    const HostDm12 rdm = l->builder.dm12(ci);
    const HostDm12 dot = l->builder.dm12Dot(ci, ci);
    const double tol = 1e-11 * std::max(1.0, maxAbs(rdm.dm2));
    for (int64_t i = 0; i < rdm.dm1.size(); ++i)
      ASSERT_NEAR(dot.dm1.flat(i), 2.0 * rdm.dm1.flat(i), tol) << "dm1 element " << i;
    for (int64_t i = 0; i < rdm.dm2.size(); ++i)
      ASSERT_NEAR(dot.dm2.flat(i), 2.0 * rdm.dm2.flat(i), tol) << "dm2 element " << i;
  }
}

TEST(HostDm12TangentTests, IsLinearInTheDirection) {
  const std::optional<Loaded> l = load(kGoldenCas44);
  ASSERT_TRUE(l.has_value());
  const Tensor& ci = l->golden.get("ci");
  const Tensor u1 = direction(ci, 0x11u);
  const Tensor u2 = direction(ci, 0x22u);
  const HostDm12 d1 = l->builder.dm12Dot(ci, u1);
  const HostDm12 d2 = l->builder.dm12Dot(ci, u2);
  // 3 u1 - 2 u2, as one direction.
  Tensor mixed(ci.dims());
  for (int64_t i = 0; i < mixed.size(); ++i)
    mixed.flatRef(i) = 3.0 * u1.flat(i) - 2.0 * u2.flat(i);
  const HostDm12 dm = l->builder.dm12Dot(ci, mixed);
  for (int64_t i = 0; i < dm.dm1.size(); ++i)
    ASSERT_NEAR(dm.dm1.flat(i), 3.0 * d1.dm1.flat(i) - 2.0 * d2.dm1.flat(i), 1e-12)
        << "dm1 element " << i;
  for (int64_t i = 0; i < dm.dm2.size(); ++i)
    ASSERT_NEAR(dm.dm2.flat(i), 3.0 * d1.dm2.flat(i) - 2.0 * d2.dm2.flat(i), 1e-12)
        << "dm2 element " << i;
}

// --- HostDm12RichardsonTests ---------------------------------------------------

TEST(HostDm12RichardsonTests, NormalizedPathErrorFallsFourfoldPerHalving) {
  // h from 1/8 down to 1/128, so the relative error runs over h^2 = 1.6e-2
  // down to 6.1e-5 -- every value far above the roundoff floor of the
  // difference itself (eps |dm| / 2h).
  constexpr std::array<double, 5> kSteps{0.125, 0.0625, 0.03125, 0.015625, 0.0078125};
  for (const TangentCase& c : tangentCases()) {
    SCOPED_TRACE(c.path);
    const std::optional<Loaded> l = load(c.path);
    ASSERT_TRUE(l.has_value());
    const Tensor ci = normalized(l->golden.get("ci"));
    const Tensor u = direction(ci, c.seed);
    const HostDm12 dot = l->builder.dm12Dot(ci, u);

    std::array<double, kSteps.size()> err1{};
    std::array<double, kSteps.size()> err2{};
    for (std::size_t i = 0; i < kSteps.size(); ++i) {
      const HostDm12 cd = centralDifference(l->builder, ci, u, kSteps[i], /*normalize=*/true);
      err1[i] = maxDiff(cd.dm1, dot.dm1);
      err2[i] = maxDiff(cd.dm2, dot.dm2);
    }
    // The error is -h^2/(1+h^2) dm_dot elementwise, so each ratio is
    // 4 (1 + h^2/4)/(1 + h^2) -- just under 4 at the first halving, within a
    // thousandth of it at the last.
    for (std::size_t i = 0; i + 1 < kSteps.size(); ++i) {
      SCOPED_TRACE(std::format("h {} -> {}", kSteps[i], kSteps[i + 1]));
      ASSERT_GT(err1[i + 1], 0.0);
      ASSERT_GT(err2[i + 1], 0.0);
      EXPECT_NEAR(err1[i] / err1[i + 1], 4.0, 0.2) << "dm1 Richardson ratio";
      EXPECT_NEAR(err2[i] / err2[i + 1], 4.0, 0.2) << "dm2 Richardson ratio";
    }
    // And the h^2 coefficient is the tangent itself: err(h) ~ h^2 |dm_dot|.
    const double h = kSteps.back();
    EXPECT_NEAR(err1.back(), h * h * maxAbs(dot.dm1), 0.05 * h * h * maxAbs(dot.dm1));
    EXPECT_NEAR(err2.back(), h * h * maxAbs(dot.dm2), 0.05 * h * h * maxAbs(dot.dm2));
  }
}

// --- HostDm12CreateTests -------------------------------------------------------

void expectConfigError(const Result<HostDm12Builder>& r, std::string_view fragment) {
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
  EXPECT_NE(r.error().message.find(fragment), std::string::npos) << r.error().message;
}

TEST(HostDm12CreateTests, AnOutOfRangeActiveSpaceIsAnInvalidConfigError) {
  expectConfigError(HostDm12Builder::create(0, 0, 0), "outside [1, 32]");
  expectConfigError(HostDm12Builder::create(-4, 2, 2), "outside [1, 32]");
  expectConfigError(HostDm12Builder::create(33, 2, 2), "outside [1, 32]");
  expectConfigError(HostDm12Builder::create(4, 5, 2), "must lie in [0, 4]");
  expectConfigError(HostDm12Builder::create(4, 2, -1), "must lie in [0, 4]");
  // CAS(32,32): C(32,16)^2 determinants, n^2 times which is past the working
  // set the builder allows.
  expectConfigError(HostDm12Builder::create(32, 16, 16), "working set");
}

TEST(HostDm12CreateTests, AGoodActiveSpaceReportsItsShape) {
  const Result<HostDm12Builder> b = HostDm12Builder::create(10, 5, 5);
  ASSERT_TRUE(b.has_value()) << b.error().message;
  EXPECT_EQ(b->norb(), 10);
  EXPECT_EQ(b->nelecA(), 5);
  EXPECT_EQ(b->nelecB(), 5);
  EXPECT_EQ(b->nstringsA(), 252);  // C(10,5)
  EXPECT_EQ(b->nstringsB(), 252);
  EXPECT_EQ(b->ndet(), 63504);
  // An empty spin channel is a legal active space, not a refusal.
  const Result<HostDm12Builder> empty = HostDm12Builder::create(4, 2, 0);
  ASSERT_TRUE(empty.has_value()) << empty.error().message;
  EXPECT_EQ(empty->nstringsB(), 1);
  EXPECT_EQ(empty->ndet(), 6);
}

TEST(HostDm12CreateTests, ApplyAllEIsOneVectorPerExcitationOperator) {
  const Result<HostDm12Builder> b = HostDm12Builder::create(4, 2, 2);
  ASSERT_TRUE(b.has_value()) << b.error().message;
  const Tensor ci({b->nstringsA(), b->nstringsB()});  // zero-filled
  const std::vector<double> v = b->applyAllE(ci);
  EXPECT_EQ(std::ssize(v), b->norb() * b->norb() * b->ndet());
  for (const double x : v) EXPECT_EQ(x, 0.0);
}

// --- HostDm12DeathTest --------------------------------------------------------

TEST(HostDm12DeathTest, ACiOfTheWrongShapeAborts) {
  const Result<HostDm12Builder> b = HostDm12Builder::create(4, 2, 2);
  ASSERT_TRUE(b.has_value()) << b.error().message;
  const Tensor flat({36});
  const Tensor wrong({6, 5});
  const Tensor right({6, 6});
  EXPECT_DEATH((void)b->dm12(flat), "host dm1/dm2: ci must be the");
  EXPECT_DEATH((void)b->dm12(wrong), "host dm1/dm2: ci must be the");
  EXPECT_DEATH((void)b->applyAllE(wrong), "host dm1/dm2: ci must be the");
  EXPECT_DEATH((void)b->dm12Dot(right, wrong), "host dm1/dm2: dir must be the");
}

}  // namespace nevpt2::test::rdm_host_dm
