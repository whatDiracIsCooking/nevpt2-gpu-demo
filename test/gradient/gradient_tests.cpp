// Suites for nevpt2.sc_amplitudes (src/gradient/sc_amplitudes.cppm) -- the
// strongly-contracted amplitude T, the gradient multipliers P / Q / R, and the
// subscript rewriting that differentiates one of nevpt2.energy's class einsums.
//
// Host-only ON PURPOSE: no device, no kernel, no golden -- so this binary is
// not REQUIRES_GPU and runs with no card visible. What the pseudodensities
// assembled FROM these satisfy is a numerical claim, and it stays with the
// golden tier (the `nevpt2_sc_pseudodensity_*` entries in the root
// CMakeLists.txt, all `gpu`).
//
//   ScMultipliersTests        Park Eqs. 12 and 36-38 on hand cases whose
//                             values are exact in binary; the pair (norm,
//                             energy) reproduces nevpt2.energy_finish's
//                             normToEnergy, including its vanishing-norm
//                             guard, and P_k N_k + Q_k H_k == T_k N_k (the
//                             identity Eq. 40's assembly is checked against)
//   DotHostTests              sum_k a_k b_k, and the empty case
//   DerivativeSubscriptsTests the rewriting, on the real strings the eight
//                             classes use: each operand slot in turn, the
//                             transposed-core case that makes it worth doing
//                             mechanically, a two-operand term, a
//                             one-operand term, and lhs whitespace
//   ScAmplitudesDeathTest     the abort tier: mismatched lengths, a slot that
//                             is not an operand, a subscript string with no
//                             '->', and an operand whose derivative would be
//                             a diagonal
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.sc_amplitudes;

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::gradient {

// The amplitude's definition, written out independently of the module: the
// reciprocal of normToEnergy's denominator.
double amplitude(double norm, double h, double delta) { return -1.0 / (delta + h / norm); }

// --- ScMultipliersTests ----------------------------------------------------------

// One perturber, H = 0: T = -1/Delta, and every multiplier follows. All four
// values are exact in binary, so EXPECT_EQ is the right assertion.
TEST(ScMultipliersTests, OnePerturberNoHamiltonian) {
  const ScMultipliers m = scMultipliers({1.0}, {0.0}, {2.0});
  EXPECT_EQ(m.t[0], -0.5);
  EXPECT_EQ(m.p[0], -0.5);  // 2 T + Delta T^2 = -1 + 0.5
  EXPECT_EQ(m.q[0], 0.25);
  EXPECT_EQ(m.r[0], 0.25);  // N T^2
  EXPECT_EQ(m.norm, 1.0);
  EXPECT_EQ(m.energy, -0.5);  // T N
}

// H != 0, and N != 1 so the N-scaling of each multiplier is pinned too:
// H/N = 1, Delta + H/N = 2, T = -1/2.
TEST(ScMultipliersTests, OnePerturberWithHamiltonian) {
  const ScMultipliers m = scMultipliers({4.0}, {4.0}, {1.0});
  EXPECT_EQ(m.t[0], -0.5);
  EXPECT_EQ(m.p[0], -0.75);  // -1 + 1 * 0.25
  EXPECT_EQ(m.q[0], 0.25);
  EXPECT_EQ(m.r[0], 1.0);
  EXPECT_EQ(m.energy, -2.0);
  // Eq. 40's scalar form, the identity every assembled pseudodensity is
  // contracted back against.
  EXPECT_DOUBLE_EQ(m.p[0] * 4.0 + m.q[0] * 4.0, m.energy);
}

// Several perturbers, mixed signs: the multipliers are elementwise, the sums
// are sums, and the identity holds term by term and in total.
TEST(ScMultipliersTests, ManyPerturbersMatchTheDefinitionAndTheIdentity) {
  const std::vector<double> norm{2.0, 4.0, -0.5, 8.0};
  const std::vector<double> h{1.0, -2.0, 0.25, 3.0};
  const std::vector<double> delta{3.0, 5.0, 2.0, 7.0};
  const ScMultipliers m = scMultipliers(norm, h, delta);
  double identity = 0.0;
  for (std::size_t k = 0; k < norm.size(); ++k) {
    const double t = amplitude(norm[k], h[k], delta[k]);
    EXPECT_DOUBLE_EQ(m.t[k], t) << "k=" << k;
    EXPECT_DOUBLE_EQ(m.p[k], 2.0 * t + delta[k] * t * t) << "k=" << k;
    EXPECT_DOUBLE_EQ(m.q[k], t * t) << "k=" << k;
    EXPECT_DOUBLE_EQ(m.r[k], norm[k] * t * t) << "k=" << k;
    EXPECT_NEAR(m.p[k] * norm[k] + m.q[k] * h[k], t * norm[k], 1e-14) << "k=" << k;
    identity += m.p[k] * norm[k] + m.q[k] * h[k];
  }
  EXPECT_EQ(m.norm, 2.0 + 4.0 - 0.5 + 8.0);
  EXPECT_NEAR(identity, m.energy, 1e-14);
}

// The pair (norm, energy) IS normToEnergy's, which is what makes these the
// amplitudes already implicit in the class energies rather than a second
// formula.
TEST(ScMultipliersTests, ReproducesNormToEnergy) {
  const std::vector<double> norm{2.0, 4.0, -0.5, 8.0};
  const std::vector<double> h{1.0, -2.0, 0.25, 3.0};
  const std::vector<double> delta{3.0, 5.0, 2.0, 7.0};
  const auto [refNorm, refEnergy] = normToEnergy(norm, h, delta);
  const ScMultipliers m = scMultipliers(norm, h, delta);
  EXPECT_EQ(m.norm, refNorm);
  EXPECT_NEAR(m.energy, refEnergy, 1e-15);
}

// normToEnergy's guard, verbatim: a norm at or below NUMERICAL_ZERO is an
// empty perturber space -- it counts towards the norm and nothing else, so its
// multipliers are zero and its pseudodensity contribution vanishes.
TEST(ScMultipliersTests, VanishingNormHasNoAmplitude) {
  const std::vector<double> norm{NUMERICAL_ZERO, -NUMERICAL_ZERO, 0.0, 2.0};
  const std::vector<double> h{1.0, 1.0, 1.0, 1.0};
  const std::vector<double> delta{3.0, 3.0, 3.0, 3.0};
  const ScMultipliers m = scMultipliers(norm, h, delta);
  for (std::size_t k = 0; k < 3; ++k) {
    EXPECT_EQ(m.t[k], 0.0) << "k=" << k;
    EXPECT_EQ(m.p[k], 0.0) << "k=" << k;
    EXPECT_EQ(m.q[k], 0.0) << "k=" << k;
    EXPECT_EQ(m.r[k], 0.0) << "k=" << k;
  }
  EXPECT_NE(m.t[3], 0.0);
  EXPECT_NEAR(m.energy, m.t[3] * 2.0, 1e-15);
  const auto [refNorm, refEnergy] = normToEnergy(norm, h, delta);
  EXPECT_EQ(m.norm, refNorm);
  EXPECT_NEAR(m.energy, refEnergy, 1e-15);
}

TEST(ScMultipliersTests, EmptyClassIsZero) {
  const ScMultipliers m = scMultipliers({}, {}, {});
  EXPECT_TRUE(m.t.empty());
  EXPECT_EQ(m.norm, 0.0);
  EXPECT_EQ(m.energy, 0.0);
}

// --- DotHostTests ----------------------------------------------------------------

TEST(DotHostTests, SumsTheProducts) {
  EXPECT_EQ(dotHost({1.0, 2.0, -3.0}, {4.0, 0.5, 2.0}), 4.0 + 1.0 - 6.0);
}

TEST(DotHostTests, EmptyIsZero) { EXPECT_EQ(dotHost({}, {}), 0.0); }

// --- DerivativeSubscriptsTests ---------------------------------------------------

// Sr's three-operand H term: each slot in turn. The dropped operand's labels
// become the output, and the multiplier -- carrying the term's original output
// labels -- is appended last.
TEST(DerivativeSubscriptsTests, EachSlotOfAThreeOperandTerm) {
  const std::string subs = "ipqr,pqrabc,iabc->i";
  EXPECT_EQ(derivativeSubscripts(subs, 0), "pqrabc,iabc,i->ipqr");
  EXPECT_EQ(derivativeSubscripts(subs, 1), "ipqr,iabc,i->pqrabc");
  EXPECT_EQ(derivativeSubscripts(subs, 2), "ipqr,pqrabc,i->iabc");
}

// Sijr's second norm term: the dropped operand reads the core pair in the
// opposite order from the output, and the rewritten string carries that
// transposition -- the case this exists for.
TEST(DerivativeSubscriptsTests, TransposedCorePair) {
  EXPECT_EQ(derivativeSubscripts("rpji,raij,pa->rji", 1), "rpji,pa,rji->raij");
  EXPECT_EQ(derivativeSubscripts("rpji,raij,pa->rji", 0), "raij,pa,rji->rpji");
}

// The active-block slot of Srs' norm term: its derivative is Park Eq. 42's
// M^(-2), the integrals contracted with the multiplier.
TEST(DerivativeSubscriptsTests, ActiveSlotIsTheEq42Intermediate) {
  EXPECT_EQ(derivativeSubscripts("rsqp,rsba,pqba->rs", 2), "rsqp,rsba,rs->pqba");
}

// Sir's one term with no active block: two operands, both the same block.
TEST(DerivativeSubscriptsTests, TwoOperandTerm) {
  EXPECT_EQ(derivativeSubscripts("ri,ri->ir", 0), "ri,ir->ri");
  EXPECT_EQ(derivativeSubscripts("ri,ri->ir", 1), "ri,ir->ri");
}

// One operand: the multiplier replaces it outright.
TEST(DerivativeSubscriptsTests, OneOperandTerm) {
  EXPECT_EQ(derivativeSubscripts("iajb->iajb", 0), "iajb->iajb");
}

// Multi-index output (Sij's), and lhs whitespace, which the parse drops as
// the einsum planner's does.
TEST(DerivativeSubscriptsTests, MultiIndexOutputAndWhitespace) {
  EXPECT_EQ(derivativeSubscripts("qpij,baij,pqab->ij", 2), "qpij,baij,ij->pqab");
  EXPECT_EQ(derivativeSubscripts("ip, pa,ia->i", 1), "ip,ia,i->pa");
}

// --- ScAmplitudesDeathTest -------------------------------------------------------

TEST(ScAmplitudesDeathTest, MismatchedLengthsAbort) {
  EXPECT_DEATH(scMultipliers({1.0, 2.0}, {1.0}, {1.0, 2.0}), "norm/h/delta lengths differ");
  EXPECT_DEATH(dotHost({1.0, 2.0}, {1.0}), "lengths differ");
}

TEST(ScAmplitudesDeathTest, SlotMustBeAnOperand) {
  EXPECT_DEATH(derivativeSubscripts("ipqr,pqrabc,iabc->i", 3), "is not an operand");
  EXPECT_DEATH(derivativeSubscripts("ipqr,pqrabc,iabc->i", -1), "is not an operand");
}

TEST(ScAmplitudesDeathTest, MalformedSubscriptsAbort) {
  EXPECT_DEATH(derivativeSubscripts("ipqr,pqrabc,iabc", 0), "no '->'");
}

// energy_Sir's diagonal integral read: an einsum can read a repeated label but
// cannot write one, so this one is refused here and the class hoists the read
// into its own packed block instead.
TEST(ScAmplitudesDeathTest, DiagonalOperandIsRefused) {
  EXPECT_DEATH(derivativeSubscripts("rpqi,raai,qp->ir", 1), "repeats label");
}

}  // namespace nevpt2::test::gradient
