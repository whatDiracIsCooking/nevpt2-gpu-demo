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
//   NormToEnergyHessianTests  the finish's 3x3 second derivative in the d form
//                          (#150, docs/hessian-theory.md Sec. 2): the hand
//                          case, agreement with Sec. 2.1's T forms at N != 0,
//                          Sec. 2.3's two exact zeros (the Hessian annihilates
//                          the ray (N, H)), Sec. 2.1-2.2's structural relations
//                          (E_ND = N E_NH + Q, E_DD = N^2 E_HH, the rank-one
//                          (H, Delta) block), N = 0 exactly giving zero where
//                          the T form gives NaN, and a Richardson table
//                          against a central difference of (P, Q, R)
//   NormToEnergyMultiplierTangentTests  Pdot / Qdot / Rdot (Sec. 2.4): the
//                          3x3 applied to the direction, still along the ray
//                          (Rdot = R there), a Richardson table along a line,
//                          and the class form per perturber with the guard
//   NormToEnergyMultiplierTangentDeathTest  six vectors of different lengths
//                          abort
//   NormToEnergyHessianSandwichTests  Eddot's second sum over a class (#153):
//                          symmetric in (u, v), the central difference of
//                          normToEnergyTangent's base point at fixed Delta,
//                          and the guard
//   NormToEnergyHessianSandwichDeathTest  nine vectors of different lengths
//                          abort
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
//
// THE PC ADJOINT'S PER-TUPLE ARITHMETIC (#74), held to an INDEPENDENT host
// energy that inverts the full (m, m) matrix D_t = diag(lambda + Delta_t) +
// h M by Gaussian elimination -- so nothing below re-derives the formula it
// checks:
//   PcAdjointTests         pcAdjointSingle / pcAdjointPair (both exchange
//                          layouts): the energy, ybar = dE/dy and xbar =
//                          dE/dx against central differences (exact: E is
//                          quadratic in the rows), R = dE/dDelta and
//                          <mgg, M> = dE/dh along a symmetric M against
//                          Richardson-checked differences, mggDelta as the
//                          Delta-weighted sum of the per-tuple weights, and a
//                          partner-layout pair as the two-tensor one with the
//                          exchange folded back
//   PcRotationTangentTests pcRotationTangentSingle / Pair, the forward side's
//                          own arithmetic: against a central difference of the
//                          same independent energy along (ydot, mK), and
//                          against the adjoint's ybar . ydot + <mgg, mK> -- the
//                          host-level duality
//   PcDeltaTangentTests    pcDeltaTangentSingle (the PC CI tangent's
//                          orbital-energy term) is sum_t R_t Deltadot_t
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

// --- NormToEnergyHessianTests ----------------------------------------------------
//
// The finish's 3x3 second derivative in (N, H, Delta), docs/hessian-theory.md
// Sec. 2 (#150). On the hand case, from Sec. 2.1's T forms worked on paper:
//
//   k = 0 (T = -1/3):  E_NN = -16/27, E_NH = 8/27, E_ND = 7/27,
//                      E_HH = -4/27,  E_HD = -2/27, E_DD = -1/27
//   k = 1 (T = -1/2):  E_NN = -9/32,  E_NH = 3/16, E_ND = 5/8,
//                      E_HH = -1/8,   E_HD = -1/4,  E_DD = -1/2

// Sec. 2.1's T forms, transcribed independently of the d form the module
// implements, so the two are compared rather than one re-derived. These are
// the forms with the removable 1/N singularities: NaN at N == 0 exactly.
FinishHessian hessianInTheTForm(double n, double h, double delta) {
  const double t = -1.0 / (delta + h / n);
  const double u = 1.0 + delta * t;
  return {.nn = 2.0 * t / n * u * u,
          .nh = 2.0 * t * t / n * u,
          .nDelta = t * t * (3.0 + 2.0 * delta * t),
          .dd = 2.0 * t * t * t / n,
          .hh = 2.0 * t * t * t / n,
          .hDelta = 2.0 * t * t * t,
          .deltaDelta = 2.0 * n * t * t * t};
}

// The finish's gradient (P, Q, R) at one perturber -- the first derivatives
// the Hessian is checked against by difference.
std::array<double, 3> finishGradient(double n, double h, double delta) {
  const auto [p, q] = normToEnergyDerivative(n, h, delta);
  return {p, q, normToEnergyDeltaDerivative(n, h, delta)};
}

std::array<double, 9> asMatrix(const FinishHessian& e) {
  return {e.nn, e.nh,     e.nDelta, e.nh,       e.hh, e.hDelta,
          e.nDelta, e.hDelta, e.deltaDelta};
}

// The spread every identity below runs over: negative norms, a negative
// Delta, H = 0, and a small norm (the T form's 1/N is large there).
template <class F>
void forEachPerturber(F&& f) {
  for (const double n : {0.5, 2.0, 13.0, -0.75, 1e-6})
    for (const double h : {1.0, 3.0, -2.5, 0.0})
      for (const double d : {1.0, 0.5, 7.0, -0.25}) {
        if (std::fabs(n * d + h) < 1e-6) continue;  // the finish's pole
        SCOPED_TRACE(std::format("N={} H={} Delta={}", n, h, d));
        f(n, h, d);
      }
}

TEST(NormToEnergyHessianTests, MatchesTheHandCase) {
  const std::array<std::array<double, 6>, 2> want = {{
      {-16.0 / 27.0, 8.0 / 27.0, 7.0 / 27.0, -4.0 / 27.0, -2.0 / 27.0, -1.0 / 27.0},
      {-9.0 / 32.0, 3.0 / 16.0, 5.0 / 8.0, -1.0 / 8.0, -1.0 / 4.0, -1.0 / 2.0},
  }};
  for (std::size_t k = 0; k < 2; ++k) {
    SCOPED_TRACE(std::format("perturber {}", k));
    const FinishHessian e = normToEnergyHessian(kHandNorm[k], kHandH[k], kHandDiff[k]);
    EXPECT_DOUBLE_EQ(e.nn, want[k][0]);
    EXPECT_DOUBLE_EQ(e.nh, want[k][1]);
    EXPECT_DOUBLE_EQ(e.nDelta, want[k][2]);
    EXPECT_DOUBLE_EQ(e.hh, want[k][3]);
    EXPECT_DOUBLE_EQ(e.hDelta, want[k][4]);
    EXPECT_DOUBLE_EQ(e.deltaDelta, want[k][5]);
    EXPECT_DOUBLE_EQ(e.dd, want[k][3]);
  }
}

TEST(NormToEnergyHessianTests, AgreesWithTheTFormAwayFromNZero) {
  // The two forms are the same function at every N != 0, small N included.
  forEachPerturber([](double n, double h, double d) {
    const auto got = asMatrix(normToEnergyHessian(n, h, d));
    const auto ref = asMatrix(hessianInTheTForm(n, h, d));
    for (std::size_t i = 0; i < 9; ++i)
      EXPECT_NEAR(got[i], ref[i], 1e-12 * std::max(1.0, std::fabs(ref[i]))) << "entry " << i;
  });
}

TEST(NormToEnergyHessianTests, TheRayIsAnnihilatedExactly) {
  // Sec. 2.3, Eq. 40's second-order partner: E is degree-1 homogeneous in
  // (N, H), so its Hessian annihilates the ray (N, H):
  //   N E_NN + H E_NH == 0,   N E_NH + H E_HH == 0.
  // Both are T (Delta + H/N) = -1 in disguise -- exact zeros, so the bound is
  // round-off relative to the terms that cancel.
  forEachPerturber([](double n, double h, double d) {
    const FinishHessian e = normToEnergyHessian(n, h, d);
    EXPECT_NEAR(n * e.nn + h * e.nh, 0.0,
                1e-14 * std::max(1.0, std::fabs(n * e.nn) + std::fabs(h * e.nh)));
    EXPECT_NEAR(n * e.nh + h * e.hh, 0.0,
                1e-14 * std::max(1.0, std::fabs(n * e.nh) + std::fabs(h * e.hh)));
  });
}

TEST(NormToEnergyHessianTests, SatisfiesTheStructuralRelations) {
  // Sec. 2.1-2.2: E_ND == N E_NH + Q (Q = T^2 = E_H, Park Eq. 37),
  // E_DD == N^2 E_HH, and the (H, Delta) block is E_dd (1, N) (x) (1, N),
  // so rank one: E_HD == N E_HH and det == 0.
  forEachPerturber([](double n, double h, double d) {
    const FinishHessian e = normToEnergyHessian(n, h, d);
    const double q = normToEnergyDerivative(n, h, d).second;
    const auto tol = [](double x) { return 1e-13 * std::max(1.0, std::fabs(x)); };
    EXPECT_NEAR(e.nDelta, n * e.nh + q, tol(e.nDelta));
    EXPECT_NEAR(e.deltaDelta, n * n * e.hh, tol(e.deltaDelta));
    EXPECT_NEAR(e.hDelta, n * e.hh, tol(e.hDelta));
    EXPECT_EQ(e.hh, e.dd);
    EXPECT_NEAR(e.hh * e.deltaDelta - e.hDelta * e.hDelta, 0.0,
                tol(e.hh * e.deltaDelta) + tol(e.hDelta * e.hDelta));
  });
}

TEST(NormToEnergyHessianTests, NZeroIsZeroNotNaN) {
  // N == 0 exactly is the case normToEnergy's NUMERICAL_ZERO guard exists
  // for; there the T form evaluates 0/0. The module returns zeros (the empty
  // perturber space's energy is identically zero), as normToEnergyDerivative
  // does, and so at the guard's boundary.
  EXPECT_TRUE(std::isnan(hessianInTheTForm(0.0, 5.0, 1.0).hh));
  for (const double n : {0.0, -0.0, NUMERICAL_ZERO, -NUMERICAL_ZERO}) {
    SCOPED_TRACE(std::format("N={}", n));
    for (const double v : asMatrix(normToEnergyHessian(n, 5.0, 1.0))) EXPECT_EQ(v, 0.0);
    EXPECT_EQ(normToEnergyHessian(n, 5.0, 1.0).dd, 0.0);
  }
  // Just above the guard it is finite, and E_HH = -2 N^2/d^3 is O(N^2): the
  // removable singularity is removed, not merely survived.
  const FinishHessian e = normToEnergyHessian(2.0 * NUMERICAL_ZERO, 5.0, 1.0);
  for (const double v : asMatrix(e)) EXPECT_TRUE(std::isfinite(v));
  EXPECT_LT(std::fabs(e.hh), 1e-27);
  EXPECT_NEAR(e.nn, -2.0 / 5.0, 1e-12);  // -2/d + O(N), d -> H
}

TEST(NormToEnergyHessianTests, IsTheGradientsCentralDifference) {
  // Column j of the 3x3 against a central difference of (P, Q, R) in input j;
  // the summed error falls fourfold per halving.
  for (std::size_t k = 0; k < 2; ++k) {
    const double n = kHandNorm[k], h = kHandH[k], d = kHandDiff[k];
    const auto hess = asMatrix(normToEnergyHessian(n, h, d));
    for (std::size_t j = 0; j < 3; ++j) {
      SCOPED_TRACE(std::format("perturber {} input {}", k, j));
      double prev = 0.0;
      for (const double step : {0.0625, 0.03125, 0.015625, 0.0078125}) {
        std::array<double, 3> plus{n, h, d}, minus{n, h, d};
        plus[j] += step;
        minus[j] -= step;
        const auto gp = finishGradient(plus[0], plus[1], plus[2]);
        const auto gm = finishGradient(minus[0], minus[1], minus[2]);
        double err = 0.0;
        for (std::size_t i = 0; i < 3; ++i)
          err += std::fabs((gp[i] - gm[i]) / (2.0 * step) - hess[i * 3 + j]);
        if (prev > 0.0) EXPECT_NEAR(prev / err, 4.0, 0.5) << "Richardson ratio at " << step;
        prev = err;
      }
      EXPECT_LT(prev, 1e-3);
    }
  }
}

// --- NormToEnergyMultiplierTangentTests ------------------------------------------

TEST(NormToEnergyMultiplierTangentTests, IsTheHessianAppliedToTheDirection) {
  const FinishHessian e = normToEnergyHessian(kHandNorm[0], kHandH[0], kHandDiff[0]);
  const MultiplierTangent t = applyFinishHessian(e, 0.25, -1.5, 0.75);
  EXPECT_DOUBLE_EQ(t.p, e.nn * 0.25 + e.nh * -1.5 + e.nDelta * 0.75);
  EXPECT_DOUBLE_EQ(t.q, e.nh * 0.25 + e.hh * -1.5 + e.hDelta * 0.75);
  EXPECT_DOUBLE_EQ(t.r, e.nDelta * 0.25 + e.hDelta * -1.5 + e.deltaDelta * 0.75);
}

TEST(NormToEnergyMultiplierTangentTests, AlongTheRayPAndQAreStillAndRIsItself) {
  // Sec. 2.3 again, as the multipliers see it: P and Q are degree 0 in (N, H)
  // and R = N T^2 is degree 1, so along (Ndot, Hdot, Deltadot) = (N, H, 0)
  // Pdot = Qdot = 0 and Rdot = R.
  forEachPerturber([](double n, double h, double d) {
    const MultiplierTangent t = applyFinishHessian(normToEnergyHessian(n, h, d), n, h, 0.0);
    const auto g = finishGradient(n, h, d);
    EXPECT_NEAR(t.p, 0.0, 1e-13 * std::max(1.0, std::fabs(g[0])));
    EXPECT_NEAR(t.q, 0.0, 1e-13 * std::max(1.0, std::fabs(g[1])));
    EXPECT_NEAR(t.r, g[2], 1e-13 * std::max(1.0, std::fabs(g[2])));
  });
}

TEST(NormToEnergyMultiplierTangentTests, MatchesACentralDifferenceAlongALine) {
  // P, Q, R differenced along one line in (N, H, Delta) per perturber.
  const std::vector<double> normDot = {0.25, -1.5}, hDot = {-0.5, 0.75}, diffDot = {0.3, -0.2};
  const ClassMultiplierTangents t = normToEnergyMultiplierTangent(kHandNorm, kHandH, kHandDiff,
                                                                  normDot, hDot, diffDot);
  for (std::size_t k = 0; k < 2; ++k) {
    SCOPED_TRACE(std::format("perturber {}", k));
    const std::array<double, 3> want{t.p[k], t.q[k], t.r[k]};
    double prev = 0.0;
    for (const double step : {0.0625, 0.03125, 0.015625, 0.0078125}) {
      const auto at = [&](double s) {
        return finishGradient(kHandNorm[k] + s * normDot[k], kHandH[k] + s * hDot[k],
                              kHandDiff[k] + s * diffDot[k]);
      };
      const auto gp = at(step), gm = at(-step);
      double err = 0.0;
      for (std::size_t i = 0; i < 3; ++i)
        err += std::fabs((gp[i] - gm[i]) / (2.0 * step) - want[i]);
      if (prev > 0.0) EXPECT_NEAR(prev / err, 4.0, 0.5) << "Richardson ratio at " << step;
      prev = err;
    }
    EXPECT_LT(prev, 1e-3);
  }
}

TEST(NormToEnergyMultiplierTangentTests, TheClassFormIsPerPerturberAndGuarded) {
  // A guarded perturber (N = 0, with an H that would blow up H/N) has zero
  // tangents, and every other one is applyFinishHessian's.
  const std::vector<double> norm = {0.5, 0.0, 2.0}, h = {1.0, 5.0, 3.0}, diff = {1.0, 1.0, 0.5};
  const std::vector<double> normDot = {0.25, 1.0, -1.5}, hDot = {-0.5, 2.0, 0.75},
                            diffDot = {0.3, 4.0, -0.2};
  const ClassMultiplierTangents t =
      normToEnergyMultiplierTangent(norm, h, diff, normDot, hDot, diffDot);
  ASSERT_EQ(t.p.size(), 3u);
  ASSERT_EQ(t.q.size(), 3u);
  ASSERT_EQ(t.r.size(), 3u);
  for (std::size_t k = 0; k < 3; ++k) {
    const MultiplierTangent w =
        applyFinishHessian(normToEnergyHessian(norm[k], h[k], diff[k]), normDot[k], hDot[k],
                           diffDot[k]);
    EXPECT_EQ(t.p[k], w.p);
    EXPECT_EQ(t.q[k], w.q);
    EXPECT_EQ(t.r[k], w.r);
  }
  EXPECT_EQ(t.p[1], 0.0);
  EXPECT_EQ(t.q[1], 0.0);
  EXPECT_EQ(t.r[1], 0.0);
  const ClassMultiplierTangents empty = normToEnergyMultiplierTangent({}, {}, {}, {}, {}, {});
  EXPECT_TRUE(empty.p.empty() && empty.q.empty() && empty.r.empty());
}

TEST(NormToEnergyMultiplierTangentDeathTest, MismatchedLengthsAbort) {
  EXPECT_DEATH((void)normToEnergyMultiplierTangent({1.0, 2.0}, {1.0, 1.0}, {1.0, 1.0},
                                                   {1.0, 1.0}, {1.0, 1.0}, {1.0}),
               "normToEnergyMultiplierTangent: norm/h/diff/normDot/hDot/diffDot lengths differ");
}

// --- NormToEnergyHessianSandwichTests --------------------------------------------
//
// Eddot's second sum over one class (#153, docs/hessian-theory.md Sec. 4.1):
// the 3x3 between two directions' first-order tangents.

TEST(NormToEnergyHessianSandwichTests, IsSymmetricInTheTwoDirections) {
  const std::vector<double> nu = {0.25, -1.5}, hu = {-0.5, 0.75}, du = {0.3, -0.2};
  const std::vector<double> nv = {1.0, 0.5}, hv = {0.125, -2.0}, dv = {-0.4, 0.1};
  const double uv = normToEnergyHessianSandwich(kHandNorm, kHandH, kHandDiff, nu, hu, du, nv,
                                                hv, dv);
  const double vu = normToEnergyHessianSandwich(kHandNorm, kHandH, kHandDiff, nv, hv, dv, nu,
                                                hu, du);
  EXPECT_NEAR(uv, vu, 1e-14);
  EXPECT_GT(std::fabs(uv), 1e-3) << "a vacuous comparison";
}

TEST(NormToEnergyHessianSandwichTests, IsTheTangentsCentralDifferenceAtFixedDelta) {
  // With Delta frozen, d/ds of normToEnergyTangent(N + s Nv, H + s Hv; Nu, Hu)
  // is exactly the sandwich at zero Deltadot: the tangent is linear in its
  // direction, so only the base point moves. The central difference's error
  // falls fourfold per halving.
  const std::vector<double> nu = {0.25, -1.5}, hu = {-0.5, 0.75};
  const std::vector<double> nv = {1.0, 0.5}, hv = {0.125, -2.0};
  const std::vector<double> zeros = {0.0, 0.0};
  const double want = normToEnergyHessianSandwich(kHandNorm, kHandH, kHandDiff, nu, hu, zeros,
                                                  nv, hv, zeros);
  const auto at = [&](double s) {
    std::vector<double> n = kHandNorm, h = kHandH;
    for (std::size_t k = 0; k < n.size(); ++k) {
      n[k] += s * nv[k];
      h[k] += s * hv[k];
    }
    return normToEnergyTangent(n, h, kHandDiff, nu, hu).second;
  };
  double prev = 0.0;
  for (const double step : {0.0625, 0.03125, 0.015625, 0.0078125}) {
    const double err = std::fabs((at(step) - at(-step)) / (2.0 * step) - want);
    if (prev > 0.0) EXPECT_NEAR(prev / err, 4.0, 0.5) << "Richardson ratio at " << step;
    prev = err;
  }
  EXPECT_LT(prev, 1e-3);
}

TEST(NormToEnergyHessianSandwichTests, AGuardedPerturberContributesNothing) {
  // N = 0 is normToEnergy's empty perturber space: the sandwich over it alone
  // is zero, and adding it to a class changes nothing.
  const double alone =
      normToEnergyHessianSandwich({0.0}, {5.0}, {1.0}, {1.0}, {2.0}, {4.0}, {3.0}, {-1.0}, {0.5});
  EXPECT_EQ(alone, 0.0);
  const double without = normToEnergyHessianSandwich({0.5}, {1.0}, {1.0}, {0.25}, {-0.5}, {0.3},
                                                     {1.0}, {0.125}, {-0.4});
  const double with = normToEnergyHessianSandwich({0.5, 0.0}, {1.0, 5.0}, {1.0, 1.0},
                                                  {0.25, 1.0}, {-0.5, 2.0}, {0.3, 4.0},
                                                  {1.0, 3.0}, {0.125, -1.0}, {-0.4, 0.5});
  EXPECT_EQ(with, without);
  EXPECT_EQ(normToEnergyHessianSandwich({}, {}, {}, {}, {}, {}, {}, {}, {}), 0.0);
}

TEST(NormToEnergyHessianSandwichDeathTest, MismatchedLengthsAbort) {
  EXPECT_DEATH((void)normToEnergyHessianSandwich({1.0, 2.0}, {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0},
                                                 {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {1.0},
                                                 {1.0, 1.0}),
               "normToEnergyHessianSandwich: the nine vectors' lengths differ");
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

// --- the PC adjoint (#74) ---------------------------------------------------------

// The independent reference: E for one slab, with the per-tuple (m, m) matrix
// D_t = diag(lambda + delta_t) + h * mm, inverted by elimination. `x` empty: a
// single-coupling class (factor f); `x` given: the pair form
// -(2 y D^-1 y - y D^-1 x).
std::vector<double> solveDense(std::vector<double> a, std::vector<double> b, std::size_t m) {
  for (std::size_t c = 0; c < m; ++c) {
    std::size_t piv = c;
    for (std::size_t r = c + 1; r < m; ++r)
      if (std::fabs(a[r * m + c]) > std::fabs(a[piv * m + c])) piv = r;
    for (std::size_t k = 0; k < m; ++k) std::swap(a[c * m + k], a[piv * m + k]);
    std::swap(b[c], b[piv]);
    for (std::size_t r = c + 1; r < m; ++r) {
      const double f = a[r * m + c] / a[c * m + c];
      for (std::size_t k = c; k < m; ++k) a[r * m + k] -= f * a[c * m + k];
      b[r] -= f * b[c];
    }
  }
  std::vector<double> x(m);
  for (std::size_t r = m; r-- > 0;) {
    double v = b[r];
    for (std::size_t k = r + 1; k < m; ++k) v -= a[r * m + k] * x[k];
    x[r] = v / a[r * m + r];
  }
  return x;
}

double referenceEnergy(const std::vector<double>& y, const std::vector<double>& x,
                       const std::vector<double>& lambda, const std::vector<double>& delta,
                       double factor, const std::vector<double>& mm, double h) {
  const std::size_t m = lambda.size(), nt = delta.size();
  double e = 0.0;
  for (std::size_t t = 0; t < nt; ++t) {
    std::vector<double> a(m * m);
    for (std::size_t k = 0; k < m; ++k)
      for (std::size_t l = 0; l < m; ++l)
        a[k * m + l] = (k == l ? lambda[k] + delta[t] : 0.0) + h * mm[k * m + l];
    const std::vector<double> row(y.begin() + static_cast<std::ptrdiff_t>(t * m),
                                  y.begin() + static_cast<std::ptrdiff_t>((t + 1) * m));
    const std::vector<double> g = solveDense(a, row, m);
    double yy = 0.0, yx = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
      yy += g[k] * row[k];
      if (!x.empty()) yx += g[k] * x[t * m + k];
    }
    e -= x.empty() ? factor * yy : 2.0 * yy - yx;
  }
  return e;
}

// A seeded case: nt tuples, m modes, positive denominators.
struct PcCase {
  std::size_t nt, m;
  std::vector<double> y, x, lambda, delta, mm;
};

PcCase pcCase(std::size_t nt, std::size_t m, std::uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  PcCase c{nt, m, {}, {}, {}, {}, {}};
  for (std::size_t i = 0; i < nt * m; ++i) c.y.push_back(u(gen));
  for (std::size_t i = 0; i < nt * m; ++i) c.x.push_back(u(gen));
  for (std::size_t k = 0; k < m; ++k) c.lambda.push_back(0.5 + 0.3 * static_cast<double>(k));
  for (std::size_t t = 0; t < nt; ++t) c.delta.push_back(1.0 + 0.25 * u(gen));
  c.mm.assign(m * m, 0.0);
  for (std::size_t k = 0; k < m; ++k)
    for (std::size_t l = k; l < m; ++l) c.mm[k * m + l] = c.mm[l * m + k] = u(gen);
  return c;
}

Tensor rows(const std::vector<double>& v, std::size_t nt, std::size_t m) {
  return Tensor::fromFlat({static_cast<int64_t>(nt), static_cast<int64_t>(m)}, v);
}

double frob(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

// The derivative of the reference energy in one input by a central
// difference at h and h/2, Richardson-combined: the inputs other than y are
// rational in the energy, so a single step would carry h^2 truncation.
template <class F>
double richardson(F&& f, double h) {
  const double d1 = (f(h) - f(-h)) / (2.0 * h);
  const double d2 = (f(h / 2) - f(-h / 2)) / h;
  return (4.0 * d2 - d1) / 3.0;
}

constexpr double kPcAdjTol = 1e-9;

TEST(PcAdjointTests, SingleEnergyYbarRAndWeights) {
  const PcCase c = pcCase(3, 3, 0x7401u);
  for (const double f : {1.0, 0.5}) {
    SCOPED_TRACE(f);
    std::vector<double> mgg(9, 0.0), mggD(9, 0.0);
    double minDen = std::numeric_limits<double>::infinity();
    const PcSlabAdjoint a =
        pcAdjointSingle(rows(c.y, 3, 3), c.lambda, c.delta, f, mgg, mggD, minDen);
    EXPECT_NEAR(a.energy, referenceEnergy(c.y, {}, c.lambda, c.delta, f, c.mm, 0.0), 1e-14);
    EXPECT_TRUE(a.xbar.empty());
    // ybar: E is quadratic in y, so the central difference is exact.
    for (std::size_t i = 0; i < c.y.size(); ++i) {
      std::vector<double> yp = c.y, ym = c.y;
      yp[i] += 1e-3;
      ym[i] -= 1e-3;
      const double fd = (referenceEnergy(yp, {}, c.lambda, c.delta, f, c.mm, 0.0) -
                         referenceEnergy(ym, {}, c.lambda, c.delta, f, c.mm, 0.0)) /
                        2e-3;
      EXPECT_NEAR(a.ybar[i], fd, kPcAdjTol) << "y[" << i << "]";
    }
    // R_t = dE/dDelta_t.
    for (std::size_t t = 0; t < 3; ++t) {
      const double fd = richardson(
          [&](double h) {
            std::vector<double> d = c.delta;
            d[t] += h;
            return referenceEnergy(c.y, {}, c.lambda, d, f, c.mm, 0.0);
          },
          1e-3);
      EXPECT_NEAR(a.r[t], fd, 1e-8) << "R[" << t << "]";
    }
    // <mgg, M> = dE/dh along D + h M.
    const double fdM = richardson(
        [&](double h) { return referenceEnergy(c.y, {}, c.lambda, c.delta, f, c.mm, h); }, 1e-3);
    EXPECT_NEAR(frob(mgg, c.mm), fdM, 1e-8);
    EXPECT_EQ(minDen, *std::ranges::min_element(c.delta) + c.lambda[0]);
  }
}

TEST(PcAdjointTests, WeightedByDeltaIsTheSumOfThePerTupleWeights) {
  const PcCase c = pcCase(4, 2, 0x7402u);
  std::vector<double> mgg(4, 0.0), mggD(4, 0.0), want(4, 0.0);
  double minDen = std::numeric_limits<double>::infinity();
  (void)pcAdjointSingle(rows(c.y, 4, 2), c.lambda, c.delta, 1.0, mgg, mggD, minDen);
  for (std::size_t t = 0; t < 4; ++t) {
    std::vector<double> one(4, 0.0), oneD(4, 0.0);
    const std::vector<double> row(c.y.begin() + static_cast<std::ptrdiff_t>(2 * t),
                                  c.y.begin() + static_cast<std::ptrdiff_t>(2 * t + 2));
    (void)pcAdjointSingle(rows(row, 1, 2), c.lambda, {c.delta[t]}, 1.0, one, oneD, minDen);
    for (std::size_t k = 0; k < 4; ++k) want[k] += c.delta[t] * one[k];
  }
  for (std::size_t k = 0; k < 4; ++k) EXPECT_NEAR(mggD[k], want[k], 1e-15);
}

TEST(PcAdjointTests, PairTwoTensorEnergyYbarXbarRAndWeights) {
  const PcCase c = pcCase(3, 3, 0x7403u);
  std::vector<double> mgg(9, 0.0), mggD(9, 0.0);
  double minDen = std::numeric_limits<double>::infinity();
  const Tensor x = rows(c.x, 3, 3);
  const PcSlabAdjoint a =
      pcAdjointPair(rows(c.y, 3, 3), &x, {}, c.lambda, c.delta, mgg, mggD, minDen);
  EXPECT_NEAR(a.energy, referenceEnergy(c.y, c.x, c.lambda, c.delta, 1.0, c.mm, 0.0), 1e-14);
  ASSERT_EQ(a.xbar.size(), c.x.size());
  for (std::size_t i = 0; i < c.y.size(); ++i) {
    std::vector<double> p = c.y, q = c.y;
    p[i] += 1e-3;
    q[i] -= 1e-3;
    EXPECT_NEAR(a.ybar[i],
                (referenceEnergy(p, c.x, c.lambda, c.delta, 1.0, c.mm, 0.0) -
                 referenceEnergy(q, c.x, c.lambda, c.delta, 1.0, c.mm, 0.0)) /
                    2e-3,
                kPcAdjTol)
        << "y[" << i << "]";
    p = c.x;
    q = c.x;
    p[i] += 1e-3;
    q[i] -= 1e-3;
    EXPECT_NEAR(a.xbar[i],
                (referenceEnergy(c.y, p, c.lambda, c.delta, 1.0, c.mm, 0.0) -
                 referenceEnergy(c.y, q, c.lambda, c.delta, 1.0, c.mm, 0.0)) /
                    2e-3,
                kPcAdjTol)
        << "x[" << i << "]";
  }
  for (std::size_t t = 0; t < 3; ++t) {
    const double fd = richardson(
        [&](double h) {
          std::vector<double> d = c.delta;
          d[t] += h;
          return referenceEnergy(c.y, c.x, c.lambda, d, 1.0, c.mm, 0.0);
        },
        1e-3);
    EXPECT_NEAR(a.r[t], fd, 1e-8) << "R[" << t << "]";
  }
  const double fdM = richardson(
      [&](double h) { return referenceEnergy(c.y, c.x, c.lambda, c.delta, 1.0, c.mm, h); }, 1e-3);
  // Only sym(mgg) is defined against a symmetric M; the caller takes sym.
  EXPECT_NEAR(frob(mgg, c.mm), fdM, 1e-8);
}

TEST(PcAdjointTests, PairPartnerFoldsTheExchangeBackOntoY) {
  // Four tuples, partner (0 1)(2 3) -- an involution, as Sijr's (j, i) swap.
  const PcCase c = pcCase(4, 2, 0x7404u);
  const std::vector<int64_t> partner{1, 0, 3, 2};
  // The pair's Delta is the tuple's and its partner's alike; E as a function
  // of y alone, x_t being y's partner row.
  PcCase cc = c;
  cc.delta = {c.delta[0], c.delta[0], c.delta[2], c.delta[2]};
  const auto energyOfYcc = [&](const std::vector<double>& y) {
    std::vector<double> x(y.size());
    for (std::size_t t = 0; t < 4; ++t)
      for (std::size_t k = 0; k < 2; ++k)
        x[t * 2 + k] = y[static_cast<std::size_t>(partner[t]) * 2 + k];
    return referenceEnergy(y, x, cc.lambda, cc.delta, 1.0, cc.mm, 0.0);
  };
  std::vector<double> mgg(4, 0.0), mggD(4, 0.0);
  double minDen = std::numeric_limits<double>::infinity();
  const PcSlabAdjoint a =
      pcAdjointPair(rows(cc.y, 4, 2), nullptr, partner, cc.lambda, cc.delta, mgg, mggD, minDen);
  EXPECT_TRUE(a.xbar.empty());
  EXPECT_NEAR(a.energy, energyOfYcc(cc.y), 1e-14);
  for (std::size_t i = 0; i < cc.y.size(); ++i) {
    std::vector<double> p = cc.y, q = cc.y;
    p[i] += 1e-3;
    q[i] -= 1e-3;
    EXPECT_NEAR(a.ybar[i], (energyOfYcc(p) - energyOfYcc(q)) / 2e-3, kPcAdjTol)
        << "y[" << i << "]";
  }
  // And the same numbers as the two-tensor layout with x read off the partner.
  std::vector<double> x(cc.y.size());
  for (std::size_t t = 0; t < 4; ++t)
    for (std::size_t k = 0; k < 2; ++k)
      x[t * 2 + k] = cc.y[static_cast<std::size_t>(partner[t]) * 2 + k];
  std::vector<double> mgg2(4, 0.0), mggD2(4, 0.0);
  const Tensor xt = rows(x, 4, 2);
  const PcSlabAdjoint b =
      pcAdjointPair(rows(cc.y, 4, 2), &xt, {}, cc.lambda, cc.delta, mgg2, mggD2, minDen);
  EXPECT_NEAR(a.energy, b.energy, 1e-15);
  for (std::size_t t = 0; t < 4; ++t) {
    EXPECT_NEAR(a.r[t], b.r[t], 1e-15);
    for (std::size_t k = 0; k < 2; ++k)
      EXPECT_NEAR(a.ybar[t * 2 + k],
                  b.ybar[t * 2 + k] + b.xbar[static_cast<std::size_t>(partner[t]) * 2 + k],
                  1e-14);
  }
  for (std::size_t k = 0; k < 4; ++k) EXPECT_NEAR(mgg[k], mgg2[k], 1e-15);
}

TEST(PcAdjointTests, AnEmptySlabIsNothing) {
  std::vector<double> mgg(4, 0.0), mggD(4, 0.0);
  double minDen = 0.7;
  const PcSlabAdjoint a = pcAdjointSingle(Tensor::fromFlat({0, 2}, {}), {1.0, 2.0}, {}, 1.0, mgg,
                                          mggD, minDen);
  EXPECT_EQ(a.energy, 0.0);
  EXPECT_TRUE(a.ybar.empty());
  EXPECT_TRUE(a.r.empty());
  EXPECT_EQ(minDen, 0.7);
  for (const double w : mgg) EXPECT_EQ(w, 0.0);
}

// --- the forward rotation tangent, per tuple (#74) ------------------------------------

TEST(PcRotationTangentTests, SingleIsTheDerivativeAlongYdotAndMk) {
  const PcCase c = pcCase(3, 3, 0x7405u);
  std::mt19937_64 gen(0x7406u);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> ydot(9);
  for (double& v : ydot) v = u(gen);
  for (const double f : {1.0, 0.5}) {
    SCOPED_TRACE(f);
    const double edot =
        pcRotationTangentSingle(rows(c.y, 3, 3), rows(ydot, 3, 3), c.lambda, c.delta, f, c.mm);
    const double fd = richardson(
        [&](double h) {
          std::vector<double> y = c.y;
          for (std::size_t i = 0; i < y.size(); ++i) y[i] += h * ydot[i];
          return referenceEnergy(y, {}, c.lambda, c.delta, f, c.mm, h);
        },
        1e-3);
    EXPECT_NEAR(edot, fd, 1e-8);
    // The host-level duality: the adjoint's ybar . ydot + <mgg, mK>.
    std::vector<double> mgg(9, 0.0), mggD(9, 0.0);
    double minDen = std::numeric_limits<double>::infinity();
    const PcSlabAdjoint a =
        pcAdjointSingle(rows(c.y, 3, 3), c.lambda, c.delta, f, mgg, mggD, minDen);
    EXPECT_NEAR(edot, frob(a.ybar, ydot) + frob(mgg, c.mm), 1e-14);
  }
}

TEST(PcRotationTangentTests, PairBothLayoutsAreTheDerivative) {
  const PcCase c = pcCase(4, 2, 0x7407u);
  std::mt19937_64 gen(0x7408u);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> ydot(8), xdot(8);
  for (double& v : ydot) v = u(gen);
  for (double& v : xdot) v = u(gen);
  // Two tensors.
  const Tensor x = rows(c.x, 4, 2), xd = rows(xdot, 4, 2);
  const double edot = pcRotationTangentPair(rows(c.y, 4, 2), &x, rows(ydot, 4, 2), &xd, {},
                                            c.lambda, c.delta, c.mm);
  const double fd = richardson(
      [&](double h) {
        std::vector<double> y = c.y, xx = c.x;
        for (std::size_t i = 0; i < y.size(); ++i) {
          y[i] += h * ydot[i];
          xx[i] += h * xdot[i];
        }
        return referenceEnergy(y, xx, c.lambda, c.delta, 1.0, c.mm, h);
      },
      1e-3);
  EXPECT_NEAR(edot, fd, 1e-8);
  std::vector<double> mgg(4, 0.0), mggD(4, 0.0);
  double minDen = std::numeric_limits<double>::infinity();
  const PcSlabAdjoint a =
      pcAdjointPair(rows(c.y, 4, 2), &x, {}, c.lambda, c.delta, mgg, mggD, minDen);
  EXPECT_NEAR(edot, frob(a.ybar, ydot) + frob(a.xbar, xdot) + frob(mgg, c.mm), 1e-14);

  // Partner layout: x and xdot ARE y's and ydot's partner rows.
  const std::vector<int64_t> partner{1, 0, 3, 2};
  std::vector<double> delta = {c.delta[0], c.delta[0], c.delta[2], c.delta[2]};
  std::vector<double> xp(8), xdp(8);
  for (std::size_t t = 0; t < 4; ++t)
    for (std::size_t k = 0; k < 2; ++k) {
      xp[t * 2 + k] = c.y[static_cast<std::size_t>(partner[t]) * 2 + k];
      xdp[t * 2 + k] = ydot[static_cast<std::size_t>(partner[t]) * 2 + k];
    }
  const Tensor xpt = rows(xp, 4, 2), xdpt = rows(xdp, 4, 2);
  EXPECT_NEAR(pcRotationTangentPair(rows(c.y, 4, 2), nullptr, rows(ydot, 4, 2), nullptr, partner,
                                    c.lambda, delta, c.mm),
              pcRotationTangentPair(rows(c.y, 4, 2), &xpt, rows(ydot, 4, 2), &xdpt, {}, c.lambda,
                                    delta, c.mm),
              1e-15);
}

// --- the PC CI tangent's orbital-energy term (#74) --------------------------------------

TEST(PcDeltaTangentTests, IsTheAdjointsROnDeltadot) {
  const PcCase c = pcCase(3, 3, 0x7409u);
  const std::vector<double> deltaDot{0.3, -1.2, 0.7};
  for (const double f : {1.0, 0.5}) {
    std::vector<double> mgg(9, 0.0), mggD(9, 0.0);
    double minDen = std::numeric_limits<double>::infinity();
    const PcSlabAdjoint a =
        pcAdjointSingle(rows(c.y, 3, 3), c.lambda, c.delta, f, mgg, mggD, minDen);
    EXPECT_NEAR(pcDeltaTangentSingle(rows(c.y, 3, 3), c.lambda, c.delta, deltaDot, f),
                frob(a.r, deltaDot), 1e-15);
  }
}

}  // namespace nevpt2::test::energy_finish
