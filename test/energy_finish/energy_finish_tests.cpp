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
//   NormToEnergyDerivativeTests  the SC finish's derivative {dE/dN, dE/dH}
//                          (Park Eqs. 36-37) on the same hand case, Park
//                          Eq. 40's identity P N + Q H = E over a spread of
//                          perturbers, the vanishing-norm guard, and a
//                          central difference in N and in H whose error falls
//                          fourfold per halving
//   NormToEnergyTangentTests  the whole class at once: the sum of the
//                          per-perturber derivatives, the norm tangent as the
//                          plain sum, linearity in the direction, the
//                          degree-1 homogeneity (the tangent along (N, H) IS
//                          the energy) and a Richardson table along a line in
//                          (N, H) space
//   NormToEnergyTangentDeathTest  five vectors of different lengths abort
//
// THE TANGENT'S ONLY CALCULUS IS HERE. The class contractions are linear in
// the RDMs, so their tangents are the same einsum chains with the tangent
// RDMs substituted (nevpt2.energy's :sc_tangent partition, checked against a
// finite difference of the energy itself on a card); the derivative of the
// scalar finish is the one piece that had to be differentiated, and it is
// pure host arithmetic, so it is checked here on hand cases.
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

// --- the SC finish's derivative ------------------------------------------------
//
// THE HAND CASE is NormToEnergyTests' own, norm = (0.5, 2), h = (1, 3),
// diff = (1, 0.5), so the two suites below can be read against it:
//
//   k = 0:  T = -1 / (1 + 1/0.5) = -1/3,  P = 2T + Delta T^2 = -5/9,  Q = 1/9
//   k = 1:  T = -1 / (0.5 + 3/2) = -1/2,  P = -1 + 1/8 = -7/8,        Q = 1/4
//
// and each satisfies Park Eq. 40, P_k N_k + Q_k H_k = E_k (-1/6 and -1).
constexpr double kP0 = -5.0 / 9.0, kQ0 = 1.0 / 9.0;
constexpr double kP1 = -7.0 / 8.0, kQ1 = 1.0 / 4.0;
const std::vector<double> kHandNorm = {0.5, 2.0};
const std::vector<double> kHandH = {1.0, 3.0};
const std::vector<double> kHandDiff = {1.0, 0.5};

// normToEnergy's energy at (norm + t normDot, h + t hDot) -- the line the
// tangent is the derivative along.
double energyAlong(const std::vector<double>& norm, const std::vector<double>& h,
                   const std::vector<double>& diff, const std::vector<double>& normDot,
                   const std::vector<double>& hDot, double t) {
  std::vector<double> n = norm, hh = h;
  for (std::size_t k = 0; k < n.size(); ++k) {
    n[k] += t * normDot[k];
    hh[k] += t * hDot[k];
  }
  return normToEnergy(n, hh, diff).second;
}

// (E(t = +step) - E(t = -step)) / 2 step along that line.
double centralDifferenceAlong(const std::vector<double>& norm, const std::vector<double>& h,
                              const std::vector<double>& diff, const std::vector<double>& normDot,
                              const std::vector<double>& hDot, double step) {
  return (energyAlong(norm, h, diff, normDot, hDot, step) -
          energyAlong(norm, h, diff, normDot, hDot, -step)) /
         (2.0 * step);
}

// --- NormToEnergyDerivativeTests -----------------------------------------------

TEST(NormToEnergyDerivativeTests, MatchesTheHandCase) {
  const auto [p0, q0] = normToEnergyDerivative(kHandNorm[0], kHandH[0], kHandDiff[0]);
  EXPECT_DOUBLE_EQ(p0, kP0);
  EXPECT_DOUBLE_EQ(q0, kQ0);
  const auto [p1, q1] = normToEnergyDerivative(kHandNorm[1], kHandH[1], kHandDiff[1]);
  EXPECT_DOUBLE_EQ(p1, kP1);
  EXPECT_DOUBLE_EQ(q1, kQ1);
}

TEST(NormToEnergyDerivativeTests, SatisfiesParkEq40) {
  // P_k N_k + Q_k H_k = E_k: the finish is homogeneous of degree 1 in
  // (N, H), which is the identity the adjoint side is checked against too
  // (nevpt2.sc_amplitudes, docs/gradient-theory.md). Over a spread of
  // perturbers, including negative norms and a negative Delta.
  for (const double n : {0.5, 2.0, 13.0, -0.75, 1e-6})
    for (const double h : {1.0, 3.0, -2.5, 0.0})
      for (const double d : {1.0, 0.5, 7.0, -0.25}) {
        SCOPED_TRACE(std::format("N={} H={} Delta={}", n, h, d));
        const double den = d + h / n;
        if (std::fabs(den) < 1e-6) continue;  // the pole: no finite derivative
        const auto [p, q] = normToEnergyDerivative(n, h, d);
        const double energy = -n / den;
        EXPECT_NEAR(p * n + q * h, energy, 1e-12 * std::max(1.0, std::fabs(energy)));
      }
}

TEST(NormToEnergyDerivativeTests, AVanishingNormHasNoDerivative) {
  // normToEnergy's own guard: |N| <= NUMERICAL_ZERO is an empty perturber
  // space, contributing no energy -- and so no derivative either, with an H
  // that would blow up H / N if it were not skipped.
  for (const double n : {0.0, NUMERICAL_ZERO, -NUMERICAL_ZERO}) {
    SCOPED_TRACE(std::format("N={}", n));
    const auto [p, q] = normToEnergyDerivative(n, 5.0, 1.0);
    EXPECT_EQ(p, 0.0);
    EXPECT_EQ(q, 0.0);
  }
  // Just above the guard it contributes, as normToEnergy's own test has it.
  const auto [p, q] = normToEnergyDerivative(2.0 * NUMERICAL_ZERO, 0.0, 4.0);
  EXPECT_DOUBLE_EQ(p, -2.0 / 4.0 + 4.0 / 16.0);  // 2T + Delta T^2, T = -1/4
  EXPECT_DOUBLE_EQ(q, 1.0 / 16.0);
}

TEST(NormToEnergyDerivativeTests, MatchesACentralDifferenceInNAndInH) {
  // One perturber at a time, differenced in N alone and in H alone. The
  // finish is a smooth rational function here, so the error is O(step^2) and
  // falls fourfold per halving (the Richardson check below).
  for (const std::size_t k : {std::size_t{0}, std::size_t{1}}) {
    SCOPED_TRACE(std::format("perturber {}", k));
    const std::vector<double> n{kHandNorm[k]}, h{kHandH[k]}, d{kHandDiff[k]};
    const auto [p, q] = normToEnergyDerivative(n[0], h[0], d[0]);
    double prevN = 0.0, prevH = 0.0;
    for (const double step : {0.0625, 0.03125, 0.015625}) {
      const double errN = std::fabs(centralDifferenceAlong(n, h, d, {1.0}, {0.0}, step) - p);
      const double errH = std::fabs(centralDifferenceAlong(n, h, d, {0.0}, {1.0}, step) - q);
      if (prevN > 0.0) {
        EXPECT_NEAR(prevN / errN, 4.0, 0.4) << "dE/dN Richardson ratio at step " << step;
        EXPECT_NEAR(prevH / errH, 4.0, 0.4) << "dE/dH Richardson ratio at step " << step;
      }
      prevN = errN;
      prevH = errH;
    }
    EXPECT_LT(prevN, 1e-3);
    EXPECT_LT(prevH, 1e-3);
  }
}

// --- NormToEnergyTangentTests --------------------------------------------------

TEST(NormToEnergyTangentTests, IsTheSumOfThePerPerturberDerivatives) {
  const std::vector<double> normDot = {0.25, -1.5};
  const std::vector<double> hDot = {2.0, 0.5};
  const auto [nDot, eDot] =
      normToEnergyTangent(kHandNorm, kHandH, kHandDiff, normDot, hDot);
  EXPECT_DOUBLE_EQ(nDot, 0.25 - 1.5);
  EXPECT_DOUBLE_EQ(eDot, kP0 * 0.25 + kQ0 * 2.0 + kP1 * -1.5 + kQ1 * 0.5);
}

TEST(NormToEnergyTangentTests, TheNormTangentIsThePlainSum) {
  // The norm is summed unconditionally, exactly as normToEnergy sums it: a
  // perturber below the guard still carries its norm tangent, but no energy
  // tangent.
  const std::vector<double> norm = {0.5, 0.0, 2.0};
  const std::vector<double> h = {1.0, 5.0, 3.0};
  const std::vector<double> diff = {1.0, 1.0, 0.5};
  const std::vector<double> normDot = {0.25, 7.0, -1.5};
  const std::vector<double> hDot = {2.0, 9.0, 0.5};
  const auto [nDot, eDot] = normToEnergyTangent(norm, h, diff, normDot, hDot);
  EXPECT_DOUBLE_EQ(nDot, 0.25 + 7.0 - 1.5);
  EXPECT_DOUBLE_EQ(eDot, kP0 * 0.25 + kQ0 * 2.0 + kP1 * -1.5 + kQ1 * 0.5);
}

TEST(NormToEnergyTangentTests, IsLinearInTheDirection) {
  const std::vector<double> n1 = {0.25, -1.5}, h1 = {2.0, 0.5};
  const std::vector<double> n2 = {-3.0, 0.125}, h2 = {1.0, -4.0};
  std::vector<double> nMix(2), hMix(2);
  for (std::size_t k = 0; k < 2; ++k) {
    nMix[k] = 3.0 * n1[k] - 2.0 * n2[k];
    hMix[k] = 3.0 * h1[k] - 2.0 * h2[k];
  }
  const auto [nd1, ed1] = normToEnergyTangent(kHandNorm, kHandH, kHandDiff, n1, h1);
  const auto [nd2, ed2] = normToEnergyTangent(kHandNorm, kHandH, kHandDiff, n2, h2);
  const auto [ndM, edM] = normToEnergyTangent(kHandNorm, kHandH, kHandDiff, nMix, hMix);
  EXPECT_NEAR(ndM, 3.0 * nd1 - 2.0 * nd2, 1e-14);
  EXPECT_NEAR(edM, 3.0 * ed1 - 2.0 * ed2, 1e-14);
}

TEST(NormToEnergyTangentTests, AlongTheFinishsOwnInputsItIsTheEnergyItself) {
  // Park Eq. 40 summed: the finish is homogeneous of degree 1 in (N, H), so
  // the tangent in the direction (N, H) reproduces normToEnergy's own pair.
  const auto [n, e] = normToEnergy(kHandNorm, kHandH, kHandDiff);
  const auto [nDot, eDot] =
      normToEnergyTangent(kHandNorm, kHandH, kHandDiff, kHandNorm, kHandH);
  EXPECT_DOUBLE_EQ(nDot, n);
  EXPECT_NEAR(eDot, e, 1e-15);
}

TEST(NormToEnergyTangentTests, MatchesACentralDifferenceAlongALine) {
  // The whole class at once, along a direction in (N, H) space: the error
  // falls fourfold per halving down to the roundoff floor, which is what the
  // device-side cross-check does to the class energies themselves
  // (docs/testing.md, "The SC tangent's finite-difference cross-check").
  const std::vector<double> normDot = {0.25, -1.5};
  const std::vector<double> hDot = {2.0, 0.5};
  const auto [nDot, eDot] =
      normToEnergyTangent(kHandNorm, kHandH, kHandDiff, normDot, hDot);
  EXPECT_NE(eDot, 0.0);
  double prev = 0.0;
  for (const double step : {0.0625, 0.03125, 0.015625, 0.0078125}) {
    const double err = std::fabs(
        centralDifferenceAlong(kHandNorm, kHandH, kHandDiff, normDot, hDot, step) - eDot);
    if (prev > 0.0) EXPECT_NEAR(prev / err, 4.0, 0.4) << "Richardson ratio at step " << step;
    prev = err;
  }
  EXPECT_LT(prev, 1e-4);
}

TEST(NormToEnergyTangentTests, EmptyIsZero) {
  const auto [nDot, eDot] = normToEnergyTangent({}, {}, {}, {}, {});
  EXPECT_EQ(nDot, 0.0);
  EXPECT_EQ(eDot, 0.0);
}

// --- NormToEnergyTangentDeathTest ----------------------------------------------

TEST(NormToEnergyTangentDeathTest, MismatchedLengthsAbort) {
  // Every caller builds the five vectors side by side over one class's
  // perturbers, so a length mismatch is a broken invariant: check(), the
  // abort tier.
  EXPECT_DEATH((void)normToEnergyTangent({1.0, 2.0}, {1.0}, {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}),
               "normToEnergyTangent: norm/h/diff/normDot/hDot lengths differ");
  EXPECT_DEATH((void)normToEnergyTangent({1.0}, {1.0}, {1.0}, {}, {1.0}),
               "normToEnergyTangent: norm/h/diff/normDot/hDot lengths differ");
}

}  // namespace nevpt2::test::energy_finish
