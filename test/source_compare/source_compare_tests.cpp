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
//   SourceCompareDirectionalTests
//                               the same three properties for every committed
//                               DIRECTIONAL oracle (#105): terms built to meet
//                               its stored projections pass with every row the
//                               partition has compared (all nine on CASCI
//                               orbitals; act_V's three empty by construction
//                               on CASSCF ones, #117, and asserted empty
//                               exactly there), moving one block (or P y) along one
//                               of its directions fails naming that block, a
//                               block probed by fewer than two directions
//                               fails, and a direction labelled with a block it
//                               does not lie in is refused
//
// The first three suites read the FULL oracles; the directional ones have no
// element to perturb, and are the fourth suite's.
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

// The rows of `path`'s comparison that its partition leaves EMPTY BY
// CONSTRUCTION, named from which reference the oracle was generated on rather
// than read off the partition the file carries -- so a block that comes out
// empty for any other reason is still a failure. On CASSCF orbitals (a
// `<case>_casscf` golden's oracle, #117) there is no RHF partition to split
// the active space (docs/gradient-theory.md, section 1.4): act_O is all of
// it and act_V is empty, so its three blocks have no pair. A CASCI oracle
// has every space, and no empty row. The partition is held to that too.
std::set<std::string> expectedEmptyRows(const std::string& path, const SrcGoldFile& o) {
  const bool casscf = sourceOracleStem(path).contains("_casscf");
  if (casscf) {
    EXPECT_EQ(o.nActOcc, o.nact) << "a CASSCF oracle's active space is one range";
  } else {
    EXPECT_GT(o.nActOcc, 0) << "a CASCI oracle's act_O is empty";
    EXPECT_LT(o.nActOcc, o.nact) << "a CASCI oracle's act_V is empty";
  }
  std::set<std::string> out;
  if (!casscf) return out;
  constexpr std::size_t actV = 2;  // kSourceSpaceNames' act_V
  for (std::size_t s = 0; s < kSourceSpaceNames.size(); ++s)
    for (std::size_t t = s; t < kSourceSpaceNames.size(); ++t)
      if ((s == actV || t == actV) && s != t)
        out.insert(std::format("{}-{}", kSourceSpaceNames[s], kSourceSpaceNames[t]));
  EXPECT_EQ(std::ssize(out), 3) << "core-act_V, act_O-act_V, act_V-virt";
  return out;
}

// Every row of `c` is empty exactly when `empty` names it; the count of the
// rest, which is what a test then exercises.
int nonEmptyRows(const SourceComparison& c, const std::set<std::string>& empty) {
  int n = 0;
  for (const SourceBlockRow& r : c.rows) {
    if (empty.contains(r.name)) {
      EXPECT_EQ(r.n, 0) << r.name << " should be empty in this partition";
    } else {
      EXPECT_GT(r.n, 0) << r.name << " is empty, and this partition has pairs in it";
      ++n;
    }
  }
  return n;
}

// --- SourceCompareSelfTests ---------------------------------------------------

TEST(SourceCompareSelfTests, EveryCommittedOracleAgreesWithItself) {
  const std::vector<std::string> paths = committedOracles();
  ASSERT_FALSE(paths.empty()) << "test/source_compare/CMakeLists.txt globbed no oracle";
  std::set<SourceScheme> schemes;
  for (const std::string& path : paths) {
    SCOPED_TRACE(path);
    const SrcGoldFile o = load(path);
    if (o.flavour != SourceFlavour::Full) continue;
    schemes.insert(o.scheme);
    const Terms t = ownTerms(o);
    const SourceComparison c =
        compareSourceTerms(o, t.a, t.py, {}, sourceOracleTol(o.scheme));
    EXPECT_TRUE(c.ok) << "blocks out of place: " << join(c.failures);
    ASSERT_EQ(std::ssize(c.rows), 9) << "eight blocks of two spaces and P y";
    const std::set<std::string> empty = expectedEmptyRows(path, o);
    EXPECT_EQ(nonEmptyRows(c, empty), 9 - std::ssize(empty));
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
    if (o.flavour != SourceFlavour::Full) continue;
    const double tol = sourceOracleTol(o.scheme);
    const Terms own = ownTerms(o);
    const SourceComparison self = compareSourceTerms(o, own.a, own.py, {}, tol);
    // Every row the partition has a pair in: all nine for a CASCI oracle,
    // the six without act_V for a CASSCF one -- and none empty by accident.
    const std::set<std::string> empty = expectedEmptyRows(path, o);
    const int expected = nonEmptyRows(self, empty);
    EXPECT_EQ(expected, 9 - std::ssize(empty));
    std::int64_t exercised = 0;
    for (const SourceBlockRow& r : self.rows) {
      // Empty: by construction (skipped), or by accident -- already a failure
      // in nonEmptyRows above, and the count below stays short of `expected`.
      // Either way there is no element to perturb.
      if (r.n == 0) continue;
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
    // Every block N2's and LiF's CASCI partitions have: core-core, core-virt,
    // act_O-act_V, act_O-virt, act_V-virt, virt-virt, both core-act blocks
    // and P y -- nine; the CASSCF ones, those less the three act_V blocks.
    EXPECT_EQ(exercised, 9 - std::ssize(empty));
  }
}

// --- SourceCompareKnownGapTests -------------------------------------------------

TEST(SourceCompareKnownGapTests, AGapIsPinnedBothWays) {
  for (const std::string& path : committedOracles()) {
    SCOPED_TRACE(path);
    const SrcGoldFile o = load(path);
    if (o.flavour != SourceFlavour::Full) continue;
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
    if (o.flavour != SourceFlavour::Full) continue;
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

// --- SourceCompareDirectionalTests ----------------------------------------------

// The x of G x = b, G symmetric positive definite and small (a block's
// directions, or the CI ones): Gaussian elimination, no pivoting needed.
std::vector<double> solveSmall(std::vector<std::vector<double>> g, std::vector<double> b) {
  const std::size_t n = b.size();
  for (std::size_t k = 0; k < n; ++k)
    for (std::size_t i = k + 1; i < n; ++i) {
      const double f = g[i][k] / g[k][k];
      for (std::size_t j = k; j < n; ++j) g[i][j] -= f * g[k][j];
      b[i] -= f * b[k];
    }
  std::vector<double> x(n);
  for (std::size_t i = n; i-- > 0;) {
    double r = b[i];
    for (std::size_t j = i + 1; j < n; ++j) r -= g[i][j] * x[j];
    x[i] = r / g[i][i];
  }
  return x;
}

// Terms whose projections ARE a directional oracle's values: A (in the
// oracle's basis) a combination of each block's own directions and P y one
// of the CI directions, the coefficients solving each block's Gram system --
// what a GPU that agrees with the oracle exactly would hand compareSourceOracle.
Terms directionalTerms(const SrcGoldFile& o) {
  const Tensor& dirs = o.get("kappa_dirs");
  const Tensor& block = o.get("kappa_block");
  const Tensor& val = o.get("kappa_val");
  const std::int64_t nmo = o.nmo, nk = dirs.dims()[0];
  const auto dotK = [&](std::int64_t a, std::int64_t b) {
    double s = 0.0;
    for (std::int64_t p = 0; p < nmo; ++p)
      for (std::int64_t q = p + 1; q < nmo; ++q) s += dirs(a, p, q) * dirs(b, p, q);
    return s;
  };
  Terms t{Tensor({nmo, nmo}), Tensor({o.ndet})};
  for (int b = 0; b < 8; ++b) {
    std::vector<std::int64_t> idx;
    for (std::int64_t d = 0; d < nk; ++d)
      if (block.flat(d) == b) idx.push_back(d);
    if (idx.empty()) continue;
    std::vector<std::vector<double>> g(idx.size(), std::vector<double>(idx.size()));
    std::vector<double> rhs(idx.size());
    for (std::size_t i = 0; i < idx.size(); ++i) {
      rhs[i] = val.flat(idx[i]);
      for (std::size_t j = 0; j < idx.size(); ++j) g[i][j] = dotK(idx[i], idx[j]);
    }
    const std::vector<double> c = solveSmall(g, rhs);
    for (std::size_t i = 0; i < idx.size(); ++i)
      for (std::int64_t p = 0; p < nmo; ++p)
        for (std::int64_t q = 0; q < nmo; ++q) t.a(p, q) += c[i] * dirs(idx[i], p, q);
  }
  const Tensor& u = o.get("ci_dirs");
  const Tensor& cval = o.get("ci_val");
  const std::int64_t nc = u.dims()[0];
  std::vector<std::vector<double>> g(static_cast<std::size_t>(nc),
                                     std::vector<double>(static_cast<std::size_t>(nc)));
  std::vector<double> rhs(static_cast<std::size_t>(nc));
  for (std::int64_t i = 0; i < nc; ++i) {
    rhs[static_cast<std::size_t>(i)] = cval.flat(i);
    for (std::int64_t j = 0; j < nc; ++j) {
      double s = 0.0;
      for (std::int64_t k = 0; k < o.ndet; ++k) s += u(i, k) * u(j, k);
      g[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = s;
    }
  }
  const std::vector<double> c = solveSmall(g, rhs);
  for (std::int64_t i = 0; i < nc; ++i)
    for (std::int64_t k = 0; k < o.ndet; ++k)
      t.py.flatRef(k) += c[static_cast<std::size_t>(i)] * u(i, k);
  return t;
}

// Each committed directional oracle, with its path (which names the
// reference it was generated on: expectedEmptyRows).
std::vector<std::pair<std::string, SrcGoldFile>> directionalOracles() {
  std::vector<std::pair<std::string, SrcGoldFile>> out;
  for (const std::string& path : committedOracles()) {
    SrcGoldFile o = load(path);
    if (o.flavour == SourceFlavour::Directional) out.emplace_back(path, std::move(o));
  }
  return out;
}

TEST(SourceCompareDirectionalTests, TermsMeetingTheStoredProjectionsPass) {
  const std::vector<std::pair<std::string, SrcGoldFile>> oracles = directionalOracles();
  ASSERT_FALSE(oracles.empty()) << "no committed directional oracle";
  std::set<SourceScheme> schemes;
  for (const auto& [path, o] : oracles) {
    SCOPED_TRACE(path);
    schemes.insert(o.scheme);
    EXPECT_TRUE(checkDirectionBlocks(o).has_value());
    const Terms t = directionalTerms(o);
    const SourceComparison c = compareSourceOracle(o, t.a, t.py, {}, sourceOracleTol(o));
    EXPECT_TRUE(c.ok) << "blocks out of place: " << join(c.failures);
    EXPECT_TRUE(c.directional);
    ASSERT_EQ(std::ssize(c.rows), 9) << "eight blocks of two spaces and P y";
    const std::set<std::string> empty = expectedEmptyRows(path, o);
    EXPECT_EQ(nonEmptyRows(c, empty), 9 - std::ssize(empty));
    for (const SourceBlockRow& r : c.rows) {
      EXPECT_LT(r.maxDelta, 1e-3 * sourceOracleTol(o)) << r.name;
      // Every block the partition has pairs in is probed; only the ones it
      // leaves empty by construction (act_V's, on CASSCF orbitals) are not.
      if (!empty.contains(r.name)) EXPECT_GE(r.n, kMinBlockDirections) << r.name;
    }
  }
  EXPECT_TRUE(schemes.contains(SourceScheme::Sc)) << "no SC directional oracle";
  EXPECT_TRUE(schemes.contains(SourceScheme::Pc)) << "no PC directional oracle";
}

TEST(SourceCompareDirectionalTests, OneBlockMovedAlongItsDirectionFailsNamingIt) {
  for (const auto& [path, o] : directionalOracles()) {
    SCOPED_TRACE(path);
    const double tol = sourceOracleTol(o);
    const Terms own = directionalTerms(o);
    const SourceComparison self = compareSourceOracle(o, own.a, own.py, {}, tol);
    const Tensor& dirs = o.get("kappa_dirs");
    const Tensor& u = o.get("ci_dirs");
    const std::set<std::string> empty = expectedEmptyRows(path, o);
    EXPECT_EQ(nonEmptyRows(self, empty), 9 - std::ssize(empty));
    int exercised = 0;
    for (const SourceBlockRow& r : self.rows) {
      // Empty: by construction (skipped), or by accident -- already a failure
      // in nonEmptyRows above, and `exercised` stays short. No direction to move.
      if (r.n == 0) continue;
      SCOPED_TRACE(r.name);
      // Along the direction this row's worst sits at, by `by`: that one
      // projection moves by exactly `by` (a unit direction), the block's
      // others by at most that, and no other block's at all.
      const auto moved = [&](double by) {
        Terms t = directionalTerms(o);
        if (r.name == kSourceCiBlock) {
          for (std::int64_t k = 0; k < o.ndet; ++k) t.py.flatRef(k) += by * u(r.atP, k);
        } else {
          for (std::int64_t p = 0; p < o.nmo; ++p)
            for (std::int64_t q = 0; q < o.nmo; ++q) t.a(p, q) += by * dirs(r.atP, p, q);
        }
        return compareSourceOracle(o, t.a, t.py, {}, tol);
      };
      const SourceComparison c = moved(10.0 * tol);
      EXPECT_FALSE(c.ok);
      EXPECT_EQ(c.failures, std::vector<std::string>{r.name});
      EXPECT_TRUE(moved(0.5 * tol).ok);
      ++exercised;
    }
    EXPECT_EQ(exercised, 9 - std::ssize(empty));
  }
}

TEST(SourceCompareDirectionalTests, AnUnderProbedOrMislabelledBlockIsCaught) {
  for (auto [path, o] : directionalOracles()) {
    SCOPED_TRACE(path);
    const double tol = sourceOracleTol(o);
    const Terms own = directionalTerms(o);
    // Relabel direction 0 (core-core: every partition here has core pairs)
    // as virt-virt: the support check refuses the file, and the comparison
    // alone fails core-core for being probed by too few directions.
    Tensor& block = o.arrays.at("kappa_block");
    ASSERT_EQ(block.flat(0), 0.0);
    block.flatRef(0) = 7.0;
    const Status st = checkDirectionBlocks(o);
    ASSERT_FALSE(st.has_value());
    EXPECT_EQ(st.error().kind, ErrorKind::IO);
    EXPECT_NE(st.error().message.find("kappa direction 0 is labelled virt-virt"),
              std::string::npos)
        << st.error().message;
    const SourceComparison c = compareSourceOracle(o, own.a, own.py, {}, tol);
    EXPECT_FALSE(c.ok);
    ASSERT_FALSE(c.failures.empty());
    EXPECT_EQ(c.failures.front(), "core-core");
    EXPECT_NE(c.rows.front().verdict.find("only 1 direction(s) probe the block"),
              std::string::npos)
        << c.rows.front().verdict;
  }
}

}  // namespace nevpt2::test::source_compare
