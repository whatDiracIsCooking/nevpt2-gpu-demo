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
//   PcRankCutTests         PC's rank decision (#94), the one piece of the PC
//                          solve the tangent and adjoint stand on, on hand
//                          spectra: nothing dropped (PC_TAU is the floor), a
//                          clean null space, the cut relative to s_max, a gap
//                          exactly at PC_MIN_GAP (accepted: strictly <) and
//                          just below, s_max <= 0 / empty / NaN refused, a
//                          negative dropped mode counted by |s_k|, and m == 1
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
//   NormToEnergyDeltaTests  the finish's derivative in Delta, R = N T^2
//                          (Park Eq. 38; #87, the orbital energies' response):
//                          the hand case, the guard, the weighted sum, a
//                          Richardson table in Delta, and the degree -1
//                          homogeneity in (H, Delta), Q H + R Delta = -E
//   NormToEnergyDeltaDeathTest  four vectors of different lengths abort
//
// THE TANGENT'S ONLY CALCULUS IS HERE. The class contractions are linear in
// the RDMs, so their tangents are the same einsum chains with the tangent
// RDMs substituted (nevpt2.energy's :sc_tangent partition, checked against a
// finite difference of the energy itself on a card); the derivative of the
// scalar finish is the one piece that had to be differentiated, and it is
// pure host arithmetic, so it is checked here on hand cases.
//   PcCoefficientsTests    g_k = y_k / (lambda_k + Delta), the solve in the
//                          eigenbasis; the buffer is resized, not appended to,
//                          and a non-positive denominator is recorded
//   PcQuadFormTests        g1^T (mK + delta nS) g2 on a hand case, its
//                          linearity in each argument, and its symmetry when
//                          the blocks are symmetric (what the two-spin classes
//                          rely on)
//   TangentSingleTests     the PC tangent's Sr/Si finish: the energy
//                          finishSingle gives AND
//                          Edot -= 2 g . ydot - g^T (mK + Delta nS) g, on the
//                          same hand case; a zero tangent gives exactly zero
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

// --- PcRankCutTests ------------------------------------------------------------------

PcClassResult cut(std::vector<double> s) { return pcRankCut(s); }

TEST(PcRankCutTests, NoDroppedModeUsesTauAsTheFloor) {
  const PcClassResult r = cut({0.5, 1.0});
  EXPECT_EQ(r.status, PcStatus::NotYet);  // accepted: the solve marks Done
  EXPECT_TRUE(r.hasSpectrum);
  EXPECT_TRUE(r.why.empty());
  EXPECT_EQ(r.spectrum.d, 2);
  EXPECT_EQ(r.spectrum.dropped, 0);
  EXPECT_EQ(r.spectrum.largestDropped, 0.0);
  EXPECT_EQ(r.spectrum.smallestKept, 0.5);
  EXPECT_DOUBLE_EQ(r.spectrum.gap, 0.5 / PC_TAU);
}

TEST(PcRankCutTests, NoDroppedModeStillRefusesASmallKeptMode) {
  // 2e-13 is above the cut, so kept -- but only 2 x PC_TAU above zero, and
  // with nothing dropped that is the gap: 2 < PC_MIN_GAP.
  const PcClassResult r = cut({2e-13, 1.0});
  EXPECT_EQ(r.status, PcStatus::Refused);
  EXPECT_EQ(r.spectrum.dropped, 0);
  EXPECT_DOUBLE_EQ(r.spectrum.gap, 2.0);
  EXPECT_EQ(r.why, "S's gap 2.0e+00 < PC_MIN_GAP 1e+03");
}

TEST(PcRankCutTests, ACleanStructuralNullSpaceIsDropped) {
  // Two null modes at noise level (one each side of zero), two real ones.
  const PcClassResult r = cut({-1e-16, 1e-16, 0.1, 1.0});
  EXPECT_EQ(r.status, PcStatus::NotYet);
  EXPECT_EQ(r.spectrum.d, 4);
  EXPECT_EQ(r.spectrum.dropped, 2);
  EXPECT_EQ(r.spectrum.largestDropped, 1e-16);
  EXPECT_EQ(r.spectrum.smallestKept, 0.1);
  EXPECT_DOUBLE_EQ(r.spectrum.gap, 1e15);
}

TEST(PcRankCutTests, TheCutIsRelativeToSMax) {
  // The same spectrum scaled by 4: the same decision, the same ratios.
  const PcClassResult r = cut({-4e-16, 4e-16, 0.4, 4.0});
  EXPECT_EQ(r.status, PcStatus::NotYet);
  EXPECT_EQ(r.spectrum.dropped, 2);
  EXPECT_DOUBLE_EQ(r.spectrum.largestDropped, 1e-16);
  EXPECT_DOUBLE_EQ(r.spectrum.smallestKept, 0.1);
  // A mode exactly at PC_TAU * s_max is dropped (<=, not <).
  const PcClassResult at = cut({PC_TAU * 4.0, 4.0});
  EXPECT_EQ(at.spectrum.dropped, 1);
}

TEST(PcRankCutTests, AGapExactlyAtPcMinGapIsAccepted) {
  // Powers of two, so the ratio is exactly 1000 in floating point: the
  // refusal is gap < PC_MIN_GAP, strictly. x = 2^-50 ~ 8.9e-16 is under the
  // cut (PC_TAU * s_max = 1e-13); 1000 x ~ 8.9e-13 is above it.
  const double x = std::ldexp(1.0, -50);
  const PcClassResult r = cut({x, 1000.0 * x, 1.0});
  EXPECT_EQ(r.spectrum.dropped, 1);
  EXPECT_EQ(r.spectrum.gap, PC_MIN_GAP);
  EXPECT_EQ(r.status, PcStatus::NotYet) << r.why;
}

TEST(PcRankCutTests, AGapJustBelowPcMinGapIsRefused) {
  const double x = std::ldexp(1.0, -50);
  const PcClassResult r = cut({x, 999.0 * x, 1.0});
  EXPECT_EQ(r.spectrum.gap, 999.0);
  EXPECT_EQ(r.status, PcStatus::Refused);
  EXPECT_EQ(r.why, "S's gap 1.0e+03 < PC_MIN_GAP 1e+03");
}

TEST(PcRankCutTests, NoPositiveEigenvalueIsRefused) {
  for (const auto& s : {std::vector<double>{-1.0, 0.0}, std::vector<double>{-2.0, -1.0},
                        std::vector<double>{}}) {
    const PcClassResult r = cut(s);
    EXPECT_EQ(r.status, PcStatus::Refused);
    EXPECT_EQ(r.why, "S has no positive eigenvalue");
    EXPECT_TRUE(r.hasSpectrum);
    EXPECT_EQ(r.spectrum.d, std::ssize(s));
    EXPECT_EQ(r.spectrum.dropped, 0);  // refused before the cut
  }
}

TEST(PcRankCutTests, NaNSMaxIsRefused) {
  // !(s_max > 0), not s_max <= 0: a NaN fails both comparisons and must
  // still refuse.
  const PcClassResult r = cut({0.1, std::numeric_limits<double>::quiet_NaN()});
  EXPECT_EQ(r.status, PcStatus::Refused);
  EXPECT_EQ(r.why, "S has no positive eigenvalue");
}

TEST(PcRankCutTests, ANegativeDroppedModeCountsByItsMagnitude) {
  // largest |dropped| is 1e-15 (the negative one); without the fabs it would
  // be 1e-17 and the gap 5e4, accepted. With it the gap is 500: refused.
  const PcClassResult r = cut({-1e-15, 1e-17, 5e-13, 1.0});
  EXPECT_EQ(r.spectrum.dropped, 2);
  EXPECT_EQ(r.spectrum.largestDropped, 1e-15);
  EXPECT_DOUBLE_EQ(r.spectrum.gap, 500.0);
  EXPECT_EQ(r.status, PcStatus::Refused);
}

TEST(PcRankCutTests, OnlySMaxKeptGivesRankOne) {
  const PcClassResult r = cut({1e-17, 2e-17, 3.0});
  EXPECT_EQ(r.status, PcStatus::NotYet);
  EXPECT_EQ(r.spectrum.dropped, 2);
  EXPECT_EQ(r.spectrum.d - r.spectrum.dropped, 1);  // m == 1
  EXPECT_EQ(r.spectrum.smallestKept, 1.0);
  EXPECT_DOUBLE_EQ(r.spectrum.largestDropped, 2e-17 / 3.0);
  EXPECT_DOUBLE_EQ(r.spectrum.gap, 1.5e17);
}

TEST(PcRankCutTests, LeavesTheDenominatorForCheckDenominator) {
  const PcClassResult r = cut({0.5, 1.0});
  EXPECT_EQ(r.spectrum.minDenominator, 0.0);
  EXPECT_EQ(r.energy, 0.0);
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

// --- NormToEnergyDeltaTests ----------------------------------------------------
//
// The finish's derivative in its THIRD input, Delta (#87): the orbital
// energies move with the active density, so the complete tangent carries
// R_k Deltadot_k too, with R_k = dE_k/dDelta_k = N_k T_k^2 (Park Eq. 38).

// normToEnergy's energy with Delta displaced by t deltaDot.
double energyAlongDelta(const std::vector<double>& norm, const std::vector<double>& h,
                        const std::vector<double>& diff, const std::vector<double>& diffDot,
                        double t) {
  std::vector<double> d = diff;
  for (std::size_t k = 0; k < d.size(); ++k) d[k] += t * diffDot[k];
  return normToEnergy(norm, h, d).second;
}

TEST(NormToEnergyDeltaTests, MatchesTheHandCase) {
  // R = N T^2 = N Q on the hand case: 0.5 / 9 and 2 / 4.
  EXPECT_DOUBLE_EQ(normToEnergyDeltaDerivative(kHandNorm[0], kHandH[0], kHandDiff[0]),
                   kHandNorm[0] * kQ0);
  EXPECT_DOUBLE_EQ(normToEnergyDeltaDerivative(kHandNorm[1], kHandH[1], kHandDiff[1]),
                   kHandNorm[1] * kQ1);
}

TEST(NormToEnergyDeltaTests, AVanishingNormHasNoDerivative) {
  for (const double n : {0.0, NUMERICAL_ZERO, -NUMERICAL_ZERO})
    EXPECT_EQ(normToEnergyDeltaDerivative(n, 5.0, 1.0), 0.0) << "N=" << n;
}

TEST(NormToEnergyDeltaTests, TheTangentIsTheWeightedSum) {
  const std::vector<double> diffDot = {0.25, -1.5};
  EXPECT_DOUBLE_EQ(normToEnergyDeltaTangent(kHandNorm, kHandH, kHandDiff, diffDot),
                   kHandNorm[0] * kQ0 * 0.25 + kHandNorm[1] * kQ1 * -1.5);
  EXPECT_EQ(normToEnergyDeltaTangent({}, {}, {}, {}), 0.0);
}

TEST(NormToEnergyDeltaTests, MatchesACentralDifferenceInDelta) {
  // Along a line in Delta alone; the error falls fourfold per halving.
  const std::vector<double> diffDot = {0.25, -1.5};
  const double r = normToEnergyDeltaTangent(kHandNorm, kHandH, kHandDiff, diffDot);
  EXPECT_NE(r, 0.0);
  double prev = 0.0;
  for (const double step : {0.0625, 0.03125, 0.015625, 0.0078125}) {
    const double cd = (energyAlongDelta(kHandNorm, kHandH, kHandDiff, diffDot, step) -
                       energyAlongDelta(kHandNorm, kHandH, kHandDiff, diffDot, -step)) /
                      (2.0 * step);
    const double err = std::fabs(cd - r);
    if (prev > 0.0) EXPECT_NEAR(prev / err, 4.0, 0.4) << "Richardson ratio at step " << step;
    prev = err;
  }
  EXPECT_LT(prev, 1e-4);
}

TEST(NormToEnergyDeltaTests, TheEnergyIsHomogeneousOfDegreeMinusOneInDeltaAndH) {
  // E_k(N, sH, sDelta) = E_k(N, H, Delta) / s, so d/ds at s = 1 gives
  // Q H + R Delta = -E: an identity tying R to the same finish, independent
  // of the difference quotient above.
  for (std::size_t k = 0; k < kHandNorm.size(); ++k) {
    const auto [p, q] = normToEnergyDerivative(kHandNorm[k], kHandH[k], kHandDiff[k]);
    const double r = normToEnergyDeltaDerivative(kHandNorm[k], kHandH[k], kHandDiff[k]);
    const double e = normToEnergy({kHandNorm[k]}, {kHandH[k]}, {kHandDiff[k]}).second;
    EXPECT_NEAR(q * kHandH[k] + r * kHandDiff[k], -e, 1e-15) << "perturber " << k;
    (void)p;
  }
}

TEST(NormToEnergyDeltaDeathTest, MismatchedLengthsAbort) {
  EXPECT_DEATH((void)normToEnergyDeltaTangent({1.0, 2.0}, {1.0, 1.0}, {1.0, 1.0}, {1.0}),
               "normToEnergyDeltaTangent: norm/h/diff/diffDot lengths differ");
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

// --- PcCoefficientsTests -------------------------------------------------------

TEST(PcCoefficientsTests, DividesByTheDenominatorsAndTracksTheSmallest) {
  // lambda = (1, 2), Delta = 0.5: denominators 1.5 and 2.5.
  const std::array<double, 2> y{3.0, 5.0};
  std::vector<double> g{42.0};  // overwritten, not appended to
  double minDen = 10.0;
  pcCoefficients(y, {1.0, 2.0}, 0.5, g, minDen);
  ASSERT_EQ(g.size(), 2u);
  EXPECT_DOUBLE_EQ(g[0], 3.0 / 1.5);
  EXPECT_DOUBLE_EQ(g[1], 5.0 / 2.5);
  EXPECT_DOUBLE_EQ(minDen, 1.5);
}

TEST(PcCoefficientsTests, ANonPositiveDenominatorIsRecordedNotHidden) {
  // The class is refused by checkDenominator afterwards; these coefficients
  // are computed all the same, and minDen is what says so.
  const std::array<double, 2> y{1.0, 1.0};
  std::vector<double> g;
  double minDen = std::numeric_limits<double>::infinity();
  pcCoefficients(y, {-0.75, 1.0}, 0.5, g, minDen);
  EXPECT_DOUBLE_EQ(minDen, -0.25);
  EXPECT_DOUBLE_EQ(g[0], 1.0 / -0.25);
}

// --- PcQuadFormTests -----------------------------------------------------------

TEST(PcQuadFormTests, HandCase) {
  // mK = [[1, 2], [3, 4]], nS = [[10, 0], [0, 10]], delta = 0.5, so
  // mK + delta nS = [[6, 2], [3, 9]]; g1 = (1, 2), g2 = (3, 4):
  //   g1^T M g2 = 1*(6*3 + 2*4) + 2*(3*3 + 9*4) = 26 + 90 = 116
  const std::vector<double> mK{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> nS{10.0, 0.0, 0.0, 10.0};
  EXPECT_DOUBLE_EQ(pcQuadForm({1.0, 2.0}, {3.0, 4.0}, mK, nS, 0.5), 116.0);
  // Linear in each argument, as the derivative of a bilinear form must be.
  EXPECT_DOUBLE_EQ(pcQuadForm({2.0, 4.0}, {3.0, 4.0}, mK, nS, 0.5), 232.0);
  EXPECT_DOUBLE_EQ(pcQuadForm({0.0, 0.0}, {3.0, 4.0}, mK, nS, 0.5), 0.0);
}

TEST(PcQuadFormTests, SymmetricBlocksGiveASymmetricForm) {
  // The blocks the solve builds are U^T sym(Xdot) U, so symmetric: then
  // g1^T M g2 == g2^T M g1, which the two-spin classes rely on.
  const std::vector<double> mK{1.0, 2.0, 2.0, 5.0};
  const std::vector<double> nS{3.0, -1.0, -1.0, 4.0};
  EXPECT_DOUBLE_EQ(pcQuadForm({1.0, -2.0}, {0.5, 3.0}, mK, nS, 0.25),
                   pcQuadForm({0.5, 3.0}, {1.0, -2.0}, mK, nS, 0.25));
}

// --- TangentSingleTests --------------------------------------------------------

// The same hand case as FinishSingleTests, with a tangent on top:
//   y = [[1, 2], [3, 4]], lambda = (1, 2), Delta = (0.5, 1.5)
//   g_0 = (1/1.5, 2/2.5), g_1 = (3/2.5, 4/3.5)
//   Edot = -sum_t (2 g_t . ydot_t - g_t^T (mK + Delta_t nS) g_t)
TEST(TangentSingleTests, HandCaseAndTheEnergyBeside) {
  const Tensor y = Tensor::fromFlat({2, 2}, {1.0, 2.0, 3.0, 4.0});
  const Tensor ydot = Tensor::fromFlat({2, 2}, {0.5, -1.0, 2.0, 0.25});
  const std::vector<double> lambda{1.0, 2.0}, delta{0.5, 1.5};
  const std::vector<double> mK{1.0, 0.5, 0.5, 2.0}, nS{4.0, 0.0, 0.0, 1.0};
  double e = 0.0, edot = 0.0;
  double minDen = std::numeric_limits<double>::infinity();
  tangentSingle(y, ydot, lambda, delta, mK, nS, e, edot, minDen);

  // The energy is finishSingle's, from the same pass.
  EXPECT_DOUBLE_EQ(e, -(1.0 / 1.5 + 4.0 / 2.5 + 9.0 / 2.5 + 16.0 / 3.5));
  EXPECT_DOUBLE_EQ(minDen, 1.5);

  double want = 0.0;
  for (int t = 0; t < 2; ++t) {
    const std::vector<double> g{y(t, 0) / (lambda[0] + delta[static_cast<std::size_t>(t)]),
                                y(t, 1) / (lambda[1] + delta[static_cast<std::size_t>(t)])};
    const double dot = g[0] * ydot(t, 0) + g[1] * ydot(t, 1);
    want -= 2.0 * dot - pcQuadForm(g, g, mK, nS, delta[static_cast<std::size_t>(t)]);
  }
  EXPECT_DOUBLE_EQ(edot, want);
}

TEST(TangentSingleTests, IsLinearInYdotPlusTheQuadraticPart) {
  // Doubling ydot alone doubles the 2 g . ydot half and leaves the
  // g^T (mK + Delta nS) g half alone, so the difference of the two runs is
  // exactly that half.
  const Tensor y = Tensor::fromFlat({1, 2}, {1.0, 2.0});
  const Tensor ydot = Tensor::fromFlat({1, 2}, {0.5, -1.0});
  const Tensor ydot2 = Tensor::fromFlat({1, 2}, {1.0, -2.0});
  const std::vector<double> lambda{1.0, 2.0}, delta{0.5};
  const std::vector<double> mK{1.0, 0.5, 0.5, 2.0}, nS{4.0, 0.0, 0.0, 1.0};
  double e1 = 0.0, d1 = 0.0, e2 = 0.0, d2 = 0.0;
  double min1 = std::numeric_limits<double>::infinity(), min2 = min1;
  tangentSingle(y, ydot, lambda, delta, mK, nS, e1, d1, min1);
  tangentSingle(y, ydot2, lambda, delta, mK, nS, e2, d2, min2);
  const std::vector<double> g{1.0 / 1.5, 2.0 / 2.5};
  const double dot = g[0] * 0.5 + g[1] * -1.0;
  EXPECT_DOUBLE_EQ(d2 - d1, -2.0 * dot);
  EXPECT_DOUBLE_EQ(e1, e2);
}

TEST(TangentSingleTests, ZeroTangentBlocksLeaveOnlyTheDotTerm) {
  // With mK = nS = 0 (an RDM tangent of zero -- a direction that moves no
  // RDM) and ydot = 0 too, the tangent is exactly 0: never a stray value.
  const Tensor y = Tensor::fromFlat({2, 2}, {1.0, 2.0, 3.0, 4.0});
  const Tensor ydot = Tensor::fromFlat({2, 2}, {0.0, 0.0, 0.0, 0.0});
  double e = 0.0, edot = 0.0;
  double minDen = std::numeric_limits<double>::infinity();
  tangentSingle(y, ydot, {1.0, 2.0}, {0.5, 1.5}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, e,
                edot, minDen);
  EXPECT_EQ(edot, 0.0);
  EXPECT_LT(e, 0.0);
}

TEST(TangentSingleTests, EmptySlabChangesNothing) {
  double e = -3.0, edot = 7.0, minDen = 0.9;
  tangentSingle(Tensor::fromFlat({0, 2}, {}), Tensor::fromFlat({0, 2}, {}), {1.0, 2.0}, {},
                {1.0, 0.0, 0.0, 1.0}, {1.0, 0.0, 0.0, 1.0}, e, edot, minDen);
  EXPECT_EQ(e, -3.0);
  EXPECT_EQ(edot, 7.0);
  EXPECT_EQ(minDen, 0.9);
}

}  // namespace nevpt2::test::energy_finish
