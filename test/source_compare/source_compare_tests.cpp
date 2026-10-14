// Suites for nevpt2.source_compare, --source-oracle's block-by-block
// comparison (apps/sc_pseudodensity/source_compare.cppm).
//
// Host-only ON PURPOSE: the comparison is arithmetic on two host tensors, so
// this binary is not REQUIRES_GPU and runs with no card visible. The "GPU
// side" here is an oracle itself -- every committed golden/*.srcgold, SC (#92)
// and PC (#95) -- so what is proved is the comparator's logic, which #74's PC
// entry will rely on before any GPU PC term exists:
//
//   SourceCompareSelfTests      every committed oracle against itself passes
//                               with every block compared; a copy with ONE
//                               block perturbed past the tolerance fails and
//                               names exactly that block, for every block of
//                               every oracle, P y included; a perturbation
//                               inside the tolerance does not
//   SourceCompareKnownGapTests  the pinning works both ways: a perturbed block
//                               on the known-gap list passes, an agreeing
//                               block on it fails ("now AGREES"), an entry
//                               naming no block fails; an unclassified oracle
//                               is a configuration Error, every committed one
//                               is classified, and the list is empty
//   SourceCompareBasisTests     a Y built from an oracle's A, carried back by
//                               oracleBasisA, and a y built from its P y,
//                               projected by projectOffState, pass -- the path
//                               a GPU's terms take
//
// The committed oracles are only ever read; their paths come from CMake
// (NEVPT2_TEST_COMMITTED_SRCGOLDS).
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.source_compare;

#ifndef NEVPT2_TEST_COMMITTED_SRCGOLDS
#error "NEVPT2_TEST_COMMITTED_SRCGOLDS must be defined by test/source_compare/CMakeLists.txt"
#endif

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::source_compare {

std::vector<std::string> committedOracles() {
  std::vector<std::string> out;
  const std::string_view joined = NEVPT2_TEST_COMMITTED_SRCGOLDS;
  for (const auto part : std::views::split(joined, '|'))
    if (!std::ranges::empty(part)) out.emplace_back(std::string_view(part));
  return out;
}

SrcGoldFile load(const std::string& path) {
  Result<SrcGoldFile> o = loadSrcGold(path);
  EXPECT_TRUE(o.has_value()) << o.error().message;
  return o ? *std::move(o) : SrcGoldFile{};
}

std::string join(const std::vector<std::string>& v) {
  std::string s;
  for (const std::string& x : v) s += (s.empty() ? "" : ", ") + x;
  return s;
}

// The oracle's own two arrays, as a GPU's would arrive at compareSourceTerms.
struct Terms {
  Tensor a;
  Tensor py;
};
Terms ownTerms(const SrcGoldFile& o) { return {o.get("a_corr"), o.get("py")}; }

// Moves element (p, q) of A (and (q, p), keeping it antisymmetric) or, for
// q < 0, determinant p of P y, by `by`.
void perturb(Terms& t, std::int64_t p, std::int64_t q, double by) {
  if (q < 0) {
    t.py.flatRef(p) += by;
  } else {
    t.a(p, q) += by;
    t.a(q, p) -= by;
  }
}

// --- SourceCompareSelfTests ---------------------------------------------------

TEST(SourceCompareSelfTests, EveryCommittedOracleAgreesWithItself) {
  const std::vector<std::string> paths = committedOracles();
  ASSERT_FALSE(paths.empty()) << "test/source_compare/CMakeLists.txt globbed no oracle";
  std::set<SourceScheme> schemes;
  for (const std::string& path : paths) {
    SCOPED_TRACE(path);
    const SrcGoldFile o = load(path);
    schemes.insert(o.scheme);
    const Terms t = ownTerms(o);
    const SourceComparison c =
        compareSourceTerms(o, t.a, t.py, {}, sourceOracleTol(o.scheme));
    EXPECT_TRUE(c.ok) << "blocks out of place: " << join(c.failures);
    ASSERT_EQ(std::ssize(c.rows), 9) << "eight blocks of two spaces and P y";
    for (const SourceBlockRow& r : c.rows) {
      EXPECT_EQ(r.maxDelta, 0.0) << r.name;
      EXPECT_FALSE(r.knownGap) << r.name;
    }
    EXPECT_EQ(c.rows.back().name, kSourceCiBlock);
    EXPECT_GT(c.rows.back().n, 0) << "P y compared on no determinant";
    EXPECT_TRUE(c.stillGaps.empty());
  }
  EXPECT_TRUE(schemes.contains(SourceScheme::Sc)) << "no SC oracle was compared";
  EXPECT_TRUE(schemes.contains(SourceScheme::Pc)) << "no PC oracle was compared";
}

TEST(SourceCompareSelfTests, ACopyWithOneBlockPerturbedFailsNamingIt) {
  for (const std::string& path : committedOracles()) {
    SCOPED_TRACE(path);
    const SrcGoldFile o = load(path);
    const double tol = sourceOracleTol(o.scheme);
    const Terms own = ownTerms(o);
    const SourceComparison self = compareSourceTerms(o, own.a, own.py, {}, tol);
    int exercised = 0;
    for (const SourceBlockRow& r : self.rows) {
      if (r.n == 0) continue;  // a block this partition has no pair in
      SCOPED_TRACE(r.name);
      // One element of this block, ten tolerances off: the comparison fails,
      // and names this block and nothing else.
      Terms t = ownTerms(o);
      perturb(t, r.atP, r.atQ, 10.0 * tol);
      const SourceComparison c = compareSourceTerms(o, t.a, t.py, {}, tol);
      EXPECT_FALSE(c.ok);
      EXPECT_EQ(c.failures, std::vector<std::string>{r.name});
      for (const SourceBlockRow& row : c.rows)
        if (row.name == r.name) {
          EXPECT_EQ(row.verdict, "FAILED: a block that agreed no longer does");
          EXPECT_EQ(row.atP, r.atP);
          EXPECT_EQ(row.atQ, r.atQ);
          EXPECT_NEAR(row.maxDelta, 10.0 * tol, 1e-3 * tol);
        }
      // Half a tolerance off is still agreement: the gate is the tolerance.
      Terms inside = ownTerms(o);
      perturb(inside, r.atP, r.atQ, 0.5 * tol);
      EXPECT_TRUE(compareSourceTerms(o, inside.a, inside.py, {}, tol).ok);
      ++exercised;
    }
    // Every block N2's and LiF's partitions have: core-core, core-virt,
    // act_O-act_V, act_O-virt, act_V-virt, virt-virt, both core-act blocks
    // and P y.
    EXPECT_EQ(exercised, 9);
  }
}

// --- SourceCompareKnownGapTests -------------------------------------------------

TEST(SourceCompareKnownGapTests, AGapIsPinnedBothWays) {
  for (const std::string& path : committedOracles()) {
    SCOPED_TRACE(path);
    const SrcGoldFile o = load(path);
    const double tol = sourceOracleTol(o.scheme);
    const Terms own = ownTerms(o);
    const SourceComparison self = compareSourceTerms(o, own.a, own.py, {}, tol);
    const SourceBlockRow& virtVirt = *std::ranges::find(self.rows, "virt-virt",
                                                        &SourceBlockRow::name);
    ASSERT_GT(virtVirt.n, 0);
    const std::array<std::string_view, 1> gap{"virt-virt"};

    // Known wrong, and still wrong: passes, reported as a gap still off.
    Terms off = ownTerms(o);
    perturb(off, virtVirt.atP, virtVirt.atQ, 10.0 * tol);
    const SourceComparison stillOff = compareSourceTerms(o, off.a, off.py, gap, tol);
    EXPECT_TRUE(stillOff.ok) << join(stillOff.failures);
    EXPECT_EQ(stillOff.stillGaps, std::vector<std::string>{"virt-virt"});

    // Known wrong, but now agrees: fails, so the list moves with the fix.
    const SourceComparison fixed = compareSourceTerms(o, own.a, own.py, gap, tol);
    EXPECT_FALSE(fixed.ok);
    EXPECT_EQ(fixed.failures, std::vector<std::string>{"virt-virt"});
    for (const SourceBlockRow& r : fixed.rows)
      if (r.name == "virt-virt") EXPECT_TRUE(r.verdict.starts_with("FAILED: now AGREES")) << r.verdict;

    // A list entry naming no block pins nothing, so it fails.
    const std::array<std::string_view, 1> typo{"virt_virt"};
    const SourceComparison t = compareSourceTerms(o, own.a, own.py, typo, tol);
    EXPECT_FALSE(t.ok);
    ASSERT_EQ(std::ssize(t.failures), 1);
    EXPECT_TRUE(t.failures.front().starts_with("virt_virt")) << t.failures.front();
  }
}

TEST(SourceCompareKnownGapTests, EveryCommittedOracleIsClassifiedAndTheListIsEmpty) {
  // Empty since #89 for SC; the PC oracle starts empty, so #74's GPU PC terms
  // are held to every block of it from their first run.
  EXPECT_TRUE(kKnownGaps.empty());
  for (const std::string& path : committedOracles()) {
    SCOPED_TRACE(path);
    const Result<std::vector<std::string_view>> gaps =
        knownSourceGaps(sourceOracleStem(path));
    ASSERT_TRUE(gaps.has_value()) << gaps.error().message;
    EXPECT_TRUE(gaps->empty());
  }
  EXPECT_EQ(sourceOracleStem("/x/golden/n2_ccpvdz_cas44_pc.srcgold"), "n2_ccpvdz_cas44_pc");
  const Result<std::vector<std::string_view>> none = knownSourceGaps("n2_ccpvdz_cas88");
  ASSERT_FALSE(none.has_value());
  EXPECT_EQ(none.error().kind, ErrorKind::InvalidConfig);
  EXPECT_NE(none.error().message.find("classify its blocks"), std::string::npos)
      << none.error().message;
}

// --- SourceCompareBasisTests ----------------------------------------------------

TEST(SourceCompareBasisTests, TermsCarriedFromTheGoldensBasisLandOnTheOracle) {
  for (const std::string& path : committedOracles()) {
    SCOPED_TRACE(path);
    const SrcGoldFile o = load(path);
    const std::int64_t nmo = o.nmo, ncore = o.ncore, nact = o.nact;
    const Tensor& w = o.get("act_rot");
    const Tensor& aRef = o.get("a_corr");
    // Wf: W on the active block, 1 elsewhere. A Y in the golden's basis whose
    // Y - Y^T is Wf A Wf^T: Y = (1/2) Wf A Wf^T (A antisymmetric).
    Tensor wf({nmo, nmo});
    for (std::int64_t p = 0; p < nmo; ++p) wf(p, p) = 1.0;
    for (std::int64_t r = 0; r < nact; ++r)
      for (std::int64_t c = 0; c < nact; ++c) wf(ncore + r, ncore + c) = w(r, c);
    Tensor y({nmo, nmo});
    for (std::int64_t p = 0; p < nmo; ++p)
      for (std::int64_t q = 0; q < nmo; ++q) {
        double s = 0.0;
        for (std::int64_t k = 0; k < nmo; ++k)
          for (std::int64_t l = 0; l < nmo; ++l) s += wf(p, k) * aRef(k, l) * wf(q, l);
        y(p, q) = 0.5 * s;
      }
    const Tensor a = oracleBasisA(o, y);
    double worst = 0.0;
    for (std::int64_t i = 0; i < a.size(); ++i)
      worst = std::max(worst, std::fabs(a.flat(i) - aRef.flat(i)));
    EXPECT_LT(worst, 1e-13);

    // A y with a component along the state: the projection drops it.
    const Tensor& ci = o.get("ci");
    const Tensor& pyRef = o.get("py");
    Tensor yc({o.ndet});
    for (std::int64_t i = 0; i < o.ndet; ++i) yc.flatRef(i) = pyRef.flat(i) + 3.7 * ci.flat(i);
    double cy = 0.0;
    const Tensor py = projectOffState(ci, yc, cy);
    EXPECT_NEAR(cy, 3.7, 1e-8);

    const SourceComparison c = compareSourceTerms(o, a, py, {}, sourceOracleTol(o.scheme));
    EXPECT_TRUE(c.ok) << join(c.failures);
  }
}

}  // namespace nevpt2::test::source_compare
