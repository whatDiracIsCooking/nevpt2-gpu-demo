// Suites for nevpt2.kappa_rotation (src/gradient/kappa_rotation.cppm) -- the
// orbital-rotation parameterisation's host side: a seeded antisymmetric
// kappa, U = exp(kappa), and the transformation U induces on one array's
// active axes.
//
// Host-only ON PURPOSE: no device, no kernel, no golden -- so this binary is
// not REQUIRES_GPU and runs with no card visible. What the SOURCE TERM
// assembled from these satisfies is a numerical claim and stays with the
// golden tier (the `nevpt2_sc_pseudodensity_kappa_*` entries in the root
// CMakeLists.txt, all `gpu`). What is checkable here is the algebra:
//
//   AntisymmetricKappaTests  the direction is exactly antisymmetric, has a
//                            zero diagonal, is reproducible from its seed and
//                            different between seeds, and has unit Frobenius
//                            norm
//   ExpKappaTests            U is ORTHOGONAL to 1e-14 (the acceptance's
//                            number) at several sizes and magnitudes, exp(0)
//                            is the identity exactly, det > 0 (a rotation,
//                            not a reflection -- checked through 2x2's closed
//                            form), exp(-k) is exp(k)^T, and the series'
//                            scaling-and-squaring branch agrees with the
//                            group law exp(k) = exp(k/2)^2
//   RotateAxesTests          the convention (out[a] = sum_b u(b,a) x[b]) on
//                            hand cases, several axes as the composition of
//                            one-axis passes, an empty axis list as a copy,
//                            orthogonality preserved as an invariant of the
//                            data (sum of squares along a rotated axis), and
//                            the FIRST-ORDER action: rotating by exp(t k)
//                            moves the data by t (kappa action) + O(t^2)
//   RotateByZeroTests        a rotation by kappa = 0 leaves every block
//                            BIT-IDENTICAL -- compared as bit patterns, with
//                            a -0.0 and a denormal in the data, on blocks
//                            shaped like the real integral blocks
//   KappaRotationDeathTest   the abort tier: a non-square kappa, a kappa that
//                            is not antisymmetric, a duplicate axis, an axis
//                            that is not an axis, and an axis whose extent
//                            does not match u
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.kappa_rotation;

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::kappa_rotation {

// max |a - b| over two same-shaped tensors.
double maxDiff(const Tensor& a, const Tensor& b) {
  if (a.dims() != b.dims()) return std::numeric_limits<double>::infinity();
  double m = 0.0;
  for (std::int64_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a.flat(i) - b.flat(i)));
  return m;
}

// Every element identical as a BIT PATTERN -- what "a rotation by zero
// changes nothing" has to mean for -0.0 and for a denormal, where plain
// equality would pass on a changed value.
bool bitIdentical(const Tensor& a, const Tensor& b) {
  if (a.dims() != b.dims()) return false;
  for (std::int64_t i = 0; i < a.size(); ++i)
    if (std::bit_cast<std::uint64_t>(a.flat(i)) != std::bit_cast<std::uint64_t>(b.flat(i)))
      return false;
  return true;
}

// A tensor of the given shape filled with distinct, awkward values: a -0.0, a
// denormal, a huge and a tiny magnitude among ordinary ones.
Tensor awkward(std::vector<std::int64_t> dims) {
  Tensor t(std::move(dims));
  for (std::int64_t i = 0; i < t.size(); ++i) {
    switch (i % 5) {
      case 0: t.flatRef(i) = -0.0; break;
      case 1: t.flatRef(i) = std::numeric_limits<double>::denorm_min(); break;
      case 2: t.flatRef(i) = 1e-300; break;
      case 3: t.flatRef(i) = 1e300; break;
      default: t.flatRef(i) = 0.5 + static_cast<double>(i); break;
    }
  }
  return t;
}

// A seeded dense tensor -- the ordinary operand of the rotation tests.
Tensor filled(std::vector<std::int64_t> dims, std::uint64_t seed) {
  Tensor t(std::move(dims));
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::int64_t i = 0; i < t.size(); ++i) t.flatRef(i) = dist(gen);
  return t;
}

// The first-order action, written out independently of the module (which
// takes it on the device): Xdot[... a ...] = sum_b kappa_ba X[... b ...],
// summed over `axes`. It is what rotateAxes(x, exp(t k)) must agree with to
// O(t^2), and it pins the SIGN of the convention.
Tensor kappaAction(const Tensor& x, const std::vector<int>& axes, const Tensor& kappa) {
  const std::int64_t n = kappa.dim(0);
  Tensor out(x.dims());
  for (const int axis : axes) {
    const std::int64_t stride = x.stride(axis);
    const std::int64_t outer = x.size() / (n * stride);
    for (std::int64_t o = 0; o < outer; ++o)
      for (std::int64_t inner = 0; inner < stride; ++inner) {
        const std::int64_t base = o * n * stride + inner;
        for (std::int64_t a = 0; a < n; ++a) {
          double acc = 0.0;
          for (std::int64_t b = 0; b < n; ++b) acc += kappa(b, a) * x.flat(base + b * stride);
          out.flatRef(base + a * stride) += acc;
        }
      }
  }
  return out;
}

// --- AntisymmetricKappaTests -----------------------------------------------

TEST(AntisymmetricKappaTests, ExactlyAntisymmetricWithZeroDiagonal) {
  const Tensor k = antisymmetricKappa(6, 0x1234);
  EXPECT_EQ(k.rank(), 2);
  EXPECT_EQ(k.dim(0), 6);
  EXPECT_EQ(antisymmetryResidual(k), 0.0);  // mirrored, not symmetrized
  for (std::int64_t p = 0; p < 6; ++p) EXPECT_EQ(k(p, p), 0.0);
}

TEST(AntisymmetricKappaTests, ReproducibleFromItsSeedAndDifferentBetweenSeeds) {
  EXPECT_TRUE(bitIdentical(antisymmetricKappa(5, 7), antisymmetricKappa(5, 7)));
  EXPECT_GT(maxDiff(antisymmetricKappa(5, 7), antisymmetricKappa(5, 8)), 1e-3);
}

TEST(AntisymmetricKappaTests, UnitFrobeniusNorm) {
  const Tensor k = antisymmetricKappa(7, 0xabc);
  double norm2 = 0.0;
  for (std::int64_t i = 0; i < k.size(); ++i) norm2 += k.flat(i) * k.flat(i);
  EXPECT_NEAR(norm2, 1.0, 1e-14);
}

// n = 1 has no strict upper triangle at all: the direction is the zero
// matrix, and must not be scaled by 1/0.
TEST(AntisymmetricKappaTests, OneOrbitalIsTheZeroDirection) {
  const Tensor k = antisymmetricKappa(1, 3);
  EXPECT_EQ(k.size(), 1);
  EXPECT_EQ(k.flat(0), 0.0);
}

// --- ExpKappaTests ---------------------------------------------------------

TEST(ExpKappaTests, OrthogonalToMachinePrecision) {
  for (const std::int64_t n : {2, 3, 5, 10}) {
    for (const double scale : {1e-8, 1e-3, 1.0, 7.5}) {
      Tensor k = antisymmetricKappa(n, static_cast<std::uint64_t>(n) * 31 + 5);
      for (std::int64_t i = 0; i < k.size(); ++i) k.flatRef(i) *= scale;
      const Tensor u = expKappa(k);
      EXPECT_LT(orthogonalityResidual(u), 1e-14) << "n = " << n << ", scale = " << scale;
    }
  }
}

TEST(ExpKappaTests, ExpOfZeroIsExactlyTheIdentity) {
  const Tensor u = expKappa(Tensor({4, 4}));
  EXPECT_TRUE(bitIdentical(u, Tensor::eye(4)));
}

// exp(-k) = exp(k)^T for an antisymmetric k (the inverse of a rotation).
TEST(ExpKappaTests, NegatedKappaGivesTheTranspose) {
  const Tensor k = antisymmetricKappa(5, 0x55);
  Tensor neg = k;
  for (std::int64_t i = 0; i < neg.size(); ++i) neg.flatRef(i) = -neg.flat(i);
  const Tensor u = expKappa(k);
  const Tensor v = expKappa(neg);
  double worst = 0.0;
  for (std::int64_t p = 0; p < 5; ++p)
    for (std::int64_t q = 0; q < 5; ++q) worst = std::max(worst, std::fabs(v(p, q) - u(q, p)));
  EXPECT_LT(worst, 1e-14);
}

// The 2x2 closed form: kappa = [[0, t], [-t, 0]] gives the rotation by t,
// which pins both the magnitude and the sign of the exponential.
TEST(ExpKappaTests, TwoByTwoIsTheClosedFormRotation) {
  const double t = 0.7;
  Tensor k({2, 2});
  k(0, 1) = t;
  k(1, 0) = -t;
  const Tensor u = expKappa(k);
  EXPECT_NEAR(u(0, 0), std::cos(t), 1e-15);
  EXPECT_NEAR(u(0, 1), std::sin(t), 1e-15);
  EXPECT_NEAR(u(1, 0), -std::sin(t), 1e-15);
  EXPECT_NEAR(u(1, 1), std::cos(t), 1e-15);
}

// The group law, which is also the scaling-and-squaring branch's own
// identity: a large kappa takes the squarings, its half does not.
TEST(ExpKappaTests, GroupLawAcrossTheScalingBranch) {
  Tensor k = antisymmetricKappa(6, 0x99);
  for (std::int64_t i = 0; i < k.size(); ++i) k.flatRef(i) *= 9.0;  // several squarings
  Tensor half = k;
  for (std::int64_t i = 0; i < half.size(); ++i) half.flatRef(i) *= 0.5;
  const Tensor u = expKappa(k);
  const Tensor h = expKappa(half);
  Tensor sq({6, 6});
  for (std::int64_t p = 0; p < 6; ++p)
    for (std::int64_t q = 0; q < 6; ++q) {
      double s = 0.0;
      for (std::int64_t i = 0; i < 6; ++i) s += h(p, i) * h(i, q);
      sq(p, q) = s;
    }
  EXPECT_LT(maxDiff(u, sq), 1e-13);
}

// --- RotateAxesTests -------------------------------------------------------

// The convention, on a case small enough to write out: out[a] = sum_b u(b, a)
// x[b] -- u's FIRST index is the one contracted with the data.
TEST(RotateAxesTests, ContractsUsFirstIndex) {
  Tensor x({2});
  x(0) = 1.0;
  x(1) = 10.0;
  Tensor u({2, 2});
  u(0, 0) = 2.0;
  u(0, 1) = 3.0;
  u(1, 0) = 5.0;
  u(1, 1) = 7.0;
  const Tensor out = rotateAxes(x, {0}, u);
  EXPECT_EQ(out(0), 2.0 * 1.0 + 5.0 * 10.0);
  EXPECT_EQ(out(1), 3.0 * 1.0 + 7.0 * 10.0);
}

// Two axes are the composition of the one-axis passes, in either order.
TEST(RotateAxesTests, SeveralAxesCompose) {
  const Tensor x = filled({3, 3, 2}, 0x4242);
  const Tensor u = expKappa(antisymmetricKappa(3, 0x4243));
  const Tensor both = rotateAxes(x, {0, 1}, u);
  const Tensor stepwise = rotateAxes(rotateAxes(x, {1}, u), {0}, u);
  const Tensor reversed = rotateAxes(x, {1, 0}, u);
  EXPECT_LT(maxDiff(both, stepwise), 1e-14);
  EXPECT_LT(maxDiff(both, reversed), 1e-14);
}

TEST(RotateAxesTests, NoAxesIsACopy) {
  const Tensor x = awkward({4, 2});
  EXPECT_TRUE(bitIdentical(rotateAxes(x, {}, Tensor::eye(4)), x));
}

// An orthogonal transformation preserves the sum of squares along the axis it
// acts on -- an invariant of the data, independent of the convention.
TEST(RotateAxesTests, OrthogonalRotationPreservesTheNorm) {
  const Tensor x = filled({4, 5}, 0x7777);
  const Tensor u = expKappa(antisymmetricKappa(4, 0x7778));
  const Tensor out = rotateAxes(x, {0}, u);
  double before = 0.0, after = 0.0;
  for (std::int64_t i = 0; i < x.size(); ++i) {
    before += x.flat(i) * x.flat(i);
    after += out.flat(i) * out.flat(i);
  }
  EXPECT_NEAR(after, before, 1e-12 * before);
}

// The first-order action IS the derivative of the finite rotation at t = 0:
// halving t must quarter the residual |x(t) - x - t Xdot|, which is what ties
// the device-side kappa action (nevpt2.gradient:rotation) to this
// exponential map.
TEST(RotateAxesTests, FirstOrderActionIsTheDerivativeOfTheRotation) {
  const Tensor x = filled({4, 4, 3}, 0xbeef);
  const std::vector<int> axes{0, 1};
  const Tensor kappa = antisymmetricKappa(4, 0xbef0);
  const Tensor dot = kappaAction(x, axes, kappa);
  double prev = 0.0;
  for (int step = 0; step < 4; ++step) {
    const double t = std::ldexp(1.0, -6 - step);
    Tensor scaled = kappa;
    for (std::int64_t i = 0; i < scaled.size(); ++i) scaled.flatRef(i) *= t;
    const Tensor moved = rotateAxes(x, axes, expKappa(scaled));
    double resid = 0.0;
    for (std::int64_t i = 0; i < x.size(); ++i)
      resid = std::max(resid, std::fabs(moved.flat(i) - x.flat(i) - t * dot.flat(i)));
    if (step > 0) EXPECT_NEAR(prev / resid, 4.0, 0.3) << "step " << step;
    prev = resid;
  }
}

// --- RotateByZeroTests -----------------------------------------------------

// The acceptance's own statement, on blocks shaped like the real integral
// blocks (nevpt2.energy's layouts) and with awkward values in them: a
// rotation by kappa = 0 is the identity on the data, bit for bit.
TEST(RotateByZeroTests, EveryBlockShapeIsBitIdentical) {
  constexpr std::int64_t n = 3;  // the active extent
  const Tensor u = expKappa(Tensor({n, n}));
  const std::vector<std::pair<std::vector<std::int64_t>, std::vector<int>>> blocks{
      {{n, n}, {0, 1}},                // h1e
      {{n, n, n, n}, {0, 1, 2, 3}},    // h2e
      {{4, n, n, n}, {1, 2, 3}},       // h2e_v_Sr
      {{4, n}, {1}},                   // h1e_v_Sr
      {{n, n, 2, n}, {0, 1, 3}},       // h2e_v_Si
      {{n, 2}, {0}},                   // h1e_v_Si
      {{4, n, 2, 2}, {1}},             // h2e_v_Sijr
      {{4, 4, 2, n}, {3}},             // h2e_v_Srsi
      {{4, 4, n, n}, {2, 3}},          // h2e_v_Srs
      {{n, n, 2, 2}, {0, 1}},          // h2e_v_Sij
      {{4, n, 2, n}, {1, 3}},          // h2e_v1_Sir
      {{4, n, n, 2}, {1, 2}},          // h2e_v2_Sir
  };
  for (const auto& [dims, axes] : blocks) {
    const Tensor x = awkward(dims);
    EXPECT_TRUE(bitIdentical(rotateAxes(x, axes, u), x))
        << "rank " << dims.size() << ", first axis " << axes.front();
  }
}

// The same through the whole helper chain a caller uses: a zero direction's
// exponential, applied to a block.
TEST(RotateByZeroTests, ZeroDirectionFromAntisymmetricKappa) {
  const Tensor k = antisymmetricKappa(1, 11);  // n = 1: the zero direction
  const Tensor x = awkward({1, 1, 1, 1});
  EXPECT_TRUE(bitIdentical(rotateAxes(x, {0, 1, 2, 3}, expKappa(k)), x));
}

// --- KappaRotationDeathTest ------------------------------------------------

TEST(KappaRotationDeathTest, NonSquareKappaAborts) {
  EXPECT_DEATH(expKappa(Tensor({2, 3})), "kappa must be a square matrix");
}

TEST(KappaRotationDeathTest, NotAntisymmetricAborts) {
  Tensor k({2, 2});
  k(0, 1) = 1.0;  // k(1, 0) left at zero
  EXPECT_DEATH(expKappa(k), "not antisymmetric");
}

TEST(KappaRotationDeathTest, DuplicateAxisAborts) {
  EXPECT_DEATH(rotateAxes(Tensor({2, 2}), {0, 0}, Tensor::eye(2)), "listed twice");
}

TEST(KappaRotationDeathTest, AxisOutOfRangeAborts) {
  EXPECT_DEATH(rotateAxes(Tensor({2, 2}), {2}, Tensor::eye(2)), "is not an axis");
}

TEST(KappaRotationDeathTest, AxisExtentMismatchAborts) {
  EXPECT_DEATH(rotateAxes(Tensor({2, 3}), {1}, Tensor::eye(2)), "has extent");
}

TEST(KappaRotationDeathTest, NonSquareUAborts) {
  EXPECT_DEATH(rotateAxes(Tensor({2, 2}), {0}, Tensor({2, 3})),
               "u must be a square matrix");
}

}  // namespace nevpt2::test::kappa_rotation
