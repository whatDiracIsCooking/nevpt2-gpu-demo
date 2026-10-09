// Suites for nevpt2.energy_finish (src/energy/energy_finish.cppm) -- the class
// energies' pure host arithmetic.
//
// Host-only ON PURPOSE: no device, no kernel, no golden -- so this binary is
// not REQUIRES_GPU and runs with no card visible. Each hand case is small
// enough to work on paper; the energies themselves stay with the golden tier.
//
//   ForBatchesTests        the slab walk: covers [0, n) in order, each piece
//                          at most `batch`, the uneven tail; batch <= 0 is one
//                          piece; n == 0 calls nothing
//   NormToEnergyTests      PySCF's _norm_to_energy on a hand case; |N| at or
//                          below NUMERICAL_ZERO adds to the norm, not the energy
//   CheckDenominatorTests  records minDenominator; refuses a non-positive and
//                          a NaN minimum (!(minDen > 0)), accepts a positive one
//   FinishSingleTests      E -= sum_t sum_k y[t,k]^2 / (lambda_k + Delta_t) on a
//                          hand case, accumulating into e and tracking the
//                          smallest denominator into minDen
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.energy_finish;

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
// One binary per suite file, so a name clash here is a link error.
namespace nevpt2::test::energy_finish {

using Pieces = std::vector<std::pair<int64_t, int64_t>>;

Pieces walk(int64_t n, int64_t batch) {
  Pieces pieces;
  forBatches(n, batch, [&](int64_t b0, int64_t b1) { pieces.emplace_back(b0, b1); });
  return pieces;
}

// --- ForBatchesTests -------------------------------------------------------------

TEST(ForBatchesTests, UnevenTail) {
  EXPECT_EQ(walk(10, 4), (Pieces{{0, 4}, {4, 8}, {8, 10}}));
}

TEST(ForBatchesTests, ExactMultiple) { EXPECT_EQ(walk(9, 3), (Pieces{{0, 3}, {3, 6}, {6, 9}})); }

TEST(ForBatchesTests, BatchLargerThanNIsOnePiece) { EXPECT_EQ(walk(5, 100), (Pieces{{0, 5}})); }

TEST(ForBatchesTests, NonPositiveBatchIsOnePiece) {
  EXPECT_EQ(walk(7, 0), (Pieces{{0, 7}}));
  EXPECT_EQ(walk(7, -3), (Pieces{{0, 7}}));
}

TEST(ForBatchesTests, EmptyRangeCallsNothing) {
  EXPECT_TRUE(walk(0, 4).empty());
  EXPECT_TRUE(walk(0, 0).empty());
  EXPECT_TRUE(walk(0, -1).empty());
}

TEST(ForBatchesTests, CoversInOrderWithinTheBound) {
  for (int64_t n : {1, 2, 13, 64, 100})
    for (int64_t batch : {1, 2, 5, 16, 64, 1000}) {
      SCOPED_TRACE(std::format("n={} batch={}", n, batch));
      const Pieces pieces = walk(n, batch);
      ASSERT_EQ(std::ssize(pieces), (n + batch - 1) / batch);
      int64_t next = 0;
      for (const auto [b0, b1] : pieces) {
        EXPECT_EQ(b0, next);
        EXPECT_GT(b1, b0);
        EXPECT_LE(b1 - b0, batch);
        next = b1;
      }
      EXPECT_EQ(next, n);
    }
}

// --- NormToEnergyTests -------------------------------------------------------------

// PySCF's pyscf.mrpt.nevpt2._norm_to_energy, by hand:
//   idx = abs(norm) > NUMERICAL_ZERO
//   ener_t = -(norm[idx] / (diff[idx] + h[idx] / norm[idx])).sum()
//   norm_t = norm.sum()
// norm = (0.5, 2), h = (1, 3), diff = (1, 0.5):
//   0.5 / (1 + 1/0.5)  = 0.5 / 3 = 1/6
//   2 / (0.5 + 3/2)    = 2 / 2   = 1
// E = -(1/6 + 1) = -7/6, N = 2.5.
TEST(NormToEnergyTests, MatchesPySCFOnAHandCase) {
  const auto [n, e] = normToEnergy({0.5, 2.0}, {1.0, 3.0}, {1.0, 0.5});
  EXPECT_DOUBLE_EQ(n, 2.5);
  EXPECT_DOUBLE_EQ(e, -7.0 / 6.0);
}

TEST(NormToEnergyTests, VanishingNormsAddToTheNormOnly) {
  // The middle three are at or below NUMERICAL_ZERO in magnitude, with an h
  // that would blow up h / N (or divide 0 / 0) if they were not skipped.
  const std::vector<double> norm = {0.5, NUMERICAL_ZERO, -NUMERICAL_ZERO, 0.0, 2.0};
  const std::vector<double> h = {1.0, 5.0, 5.0, 5.0, 3.0};
  const std::vector<double> diff = {1.0, 1.0, 1.0, 1.0, 0.5};
  const auto [n, e] = normToEnergy(norm, h, diff);
  EXPECT_DOUBLE_EQ(n, 2.5);  // +Z - Z + 0 cancel exactly
  EXPECT_TRUE(std::isfinite(e));
  EXPECT_DOUBLE_EQ(e, -7.0 / 6.0);
}

TEST(NormToEnergyTests, NormJustAboveTheGuardContributes) {
  // |N| = 2 * NUMERICAL_ZERO is kept: -N / (diff + h / N) with h = 0 is -N / diff.
  const double tiny = 2.0 * NUMERICAL_ZERO;
  const auto [n, e] = normToEnergy({tiny}, {0.0}, {4.0});
  EXPECT_DOUBLE_EQ(n, tiny);
  EXPECT_DOUBLE_EQ(e, -tiny / 4.0);
}

TEST(NormToEnergyTests, EmptyIsZero) {
  const auto [n, e] = normToEnergy({}, {}, {});
  EXPECT_EQ(n, 0.0);
  EXPECT_EQ(e, 0.0);
}

// --- CheckDenominatorTests -----------------------------------------------------------

TEST(CheckDenominatorTests, PositiveIsRecordedAndAccepted) {
  PcClassResult r;
  r.status = PcStatus::Done;
  checkDenominator(r, 0.83);
  EXPECT_EQ(r.status, PcStatus::Done);
  EXPECT_EQ(r.spectrum.minDenominator, 0.83);
  EXPECT_TRUE(r.why.empty());
}

TEST(CheckDenominatorTests, ZeroIsRefused) {
  PcClassResult r;
  r.status = PcStatus::Done;
  checkDenominator(r, 0.0);
  EXPECT_EQ(r.status, PcStatus::Refused);
  EXPECT_EQ(r.spectrum.minDenominator, 0.0);
  EXPECT_NE(r.why.find("is not positive"), std::string::npos) << r.why;
}

TEST(CheckDenominatorTests, NegativeIsRefused) {
  PcClassResult r;
  r.status = PcStatus::Done;
  checkDenominator(r, -1.5e-3);
  EXPECT_EQ(r.status, PcStatus::Refused);
  EXPECT_EQ(r.spectrum.minDenominator, -1.5e-3);
  EXPECT_EQ(r.why, "a denominator lambda_k + Delta_t = -1.500e-03 is not positive");
}

TEST(CheckDenominatorTests, NaNIsRefused) {
  // !(minDen > 0), not minDen <= 0: a NaN fails both comparisons and must
  // still refuse.
  PcClassResult r;
  r.status = PcStatus::Done;
  checkDenominator(r, std::numeric_limits<double>::quiet_NaN());
  EXPECT_EQ(r.status, PcStatus::Refused);
  EXPECT_TRUE(std::isnan(r.spectrum.minDenominator));
  EXPECT_NE(r.why.find("is not positive"), std::string::npos) << r.why;
}

TEST(CheckDenominatorTests, AcceptingLeavesTheStatusAlone) {
  // It only ever refuses: a class not yet marked Done stays as it was.
  PcClassResult r;
  checkDenominator(r, 1.0);
  EXPECT_EQ(r.status, PcStatus::NotYet);
  EXPECT_EQ(r.spectrum.minDenominator, 1.0);
}

// --- FinishSingleTests ---------------------------------------------------------------

// y = [[1, 2], [3, 4]] (t, k), lambda = (1, 2), Delta = (0.5, 1.5):
//   denominators   t=0: 1.5, 2.5    t=1: 2.5, 3.5
//   E = -(1/1.5 + 4/2.5 + 9/2.5 + 16/3.5)
TEST(FinishSingleTests, HandCase) {
  const Tensor y = Tensor::fromFlat({2, 2}, {1.0, 2.0, 3.0, 4.0});
  double e = 0.0;
  double minDen = std::numeric_limits<double>::infinity();
  finishSingle(y, {1.0, 2.0}, {0.5, 1.5}, e, minDen);
  EXPECT_DOUBLE_EQ(e, -(1.0 / 1.5 + 4.0 / 2.5 + 9.0 / 2.5 + 16.0 / 3.5));
  EXPECT_DOUBLE_EQ(minDen, 1.5);
}

TEST(FinishSingleTests, AccumulatesAcrossSlabs) {
  // Two slabs of the same walk (one tuple each) add up to the one-slab answer,
  // and minDen carries over: the second slab's smallest (2.5) does not replace
  // the first's (1.5).
  double e = 0.0;
  double minDen = std::numeric_limits<double>::infinity();
  finishSingle(Tensor::fromFlat({1, 2}, {1.0, 2.0}), {1.0, 2.0}, {0.5}, e, minDen);
  EXPECT_DOUBLE_EQ(minDen, 1.5);
  finishSingle(Tensor::fromFlat({1, 2}, {3.0, 4.0}), {1.0, 2.0}, {1.5}, e, minDen);
  EXPECT_DOUBLE_EQ(e, -(1.0 / 1.5 + 4.0 / 2.5 + 9.0 / 2.5 + 16.0 / 3.5));
  EXPECT_DOUBLE_EQ(minDen, 1.5);
}

TEST(FinishSingleTests, TracksANonPositiveDenominator) {
  // lambda_0 + Delta_0 = -0.25: minDen records it (for checkDenominator to
  // refuse), and an earlier larger minDen is lowered to it.
  const Tensor y = Tensor::fromFlat({1, 2}, {1.0, 1.0});
  double e = 0.0;
  double minDen = 10.0;
  finishSingle(y, {-0.75, 1.0}, {0.5}, e, minDen);
  EXPECT_DOUBLE_EQ(minDen, -0.25);
  EXPECT_DOUBLE_EQ(e, -(1.0 / -0.25 + 1.0 / 1.5));
}

TEST(FinishSingleTests, EmptySlabChangesNothing) {
  double e = -3.0, minDen = 0.9;
  finishSingle(Tensor::fromFlat({0, 2}, {}), {1.0, 2.0}, {}, e, minDen);
  EXPECT_EQ(e, -3.0);
  EXPECT_EQ(minDen, 0.9);
}

}  // namespace nevpt2::test::energy_finish
