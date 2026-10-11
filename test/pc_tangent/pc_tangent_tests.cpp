// Suites for nevpt2.pc_tangent's report (src/gradient/pc_tangent.cpp) -- the
// gate over a PcTangentCheck.
//
// Host-only ON PURPOSE: printPcTangentReport is pure host arithmetic over a
// struct of numbers, so this binary is not REQUIRES_GPU and runs with no card
// visible. It asserts nothing about an energy: the numbers below are made up,
// and what is checked is which combinations the gate accepts. The real
// comparison -- the analytic tangent against a finite difference of a real
// state -- is the golden tier's `nevpt2_pc_tangent_*` entries.
//
//   PcTangentReportTests  a clean check passes; a class REFUSED by the gap or
//                         denominator check fails the gate (and never
//                         contributes a derivative); a class energy that
//                         disagrees with pcEnergiesDevice's fails; Euler's
//                         identity is gated on the homogeneous classes and
//                         only reported on the other three; a Richardson ratio
//                         away from 4, or an extrapolation away from the
//                         analytic tangent, fails
//   PcTangentLadderTests  the finite difference is gated on a RUN of fourfold
//                         halvings (kPcMinFourfoldRun), so no steps or a
//                         single halving fails; and the PC conditioning guard
//                         must have run, and passed, at every displaced state
//                         -- a state with no gap recorded fails (#70)
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.pc_tangent;

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one. One binary
// per suite file, so a name clash here is a link error.
namespace nevpt2::test::pc_tangent {

constexpr double kAtol = 1e-8;
constexpr double kIdTol = 1e-9;
constexpr double kRatioTol = 0.5;

// A check that passes: eight computed classes whose energies match the
// reference exactly, Euler's identity exact where it applies, and a finite
// difference whose error falls exactly 4x per halving and whose Richardson
// extrapolation is the analytic tangent.
PcTangentCheck cleanCheck() {
  PcTangentCheck c;
  for (int i = 0; i < 8; ++i) {
    PcClassTangent& k = c.classes[static_cast<std::size_t>(i)];
    k.result.status = PcStatus::Done;
    k.result.hasSpectrum = i != 2;
    k.result.energy = -0.01 * static_cast<double>(i + 1);
    k.edot = 0.001 * static_cast<double>(i + 1);
    c.energyRef[static_cast<std::size_t>(i)] = k.result.energy;
    // Exact for every class here; the gate only reads the homogeneous ones.
    c.eulerEdot[static_cast<std::size_t>(i)] = 2.0 * k.result.energy;
    c.total += k.result.energy;
    c.totalRef += k.result.energy;
    c.edotTotal += k.edot;
  }
  // Three steps whose errors sit at 16 e, 4 e and e -- a run of two fourfold
  // halvings, the fewest the gate accepts (kPcMinFourfoldRun) -- so the
  // Richardson combination of the two finest is the tangent exactly:
  // fd = edot + err with err = 4e then e gives
  // (4 (edot + e) - (edot + 4 e)) / 3 = edot. Every displaced state carries
  // the conditioning guard's record of a healthy gap.
  const double e = 1e-7;
  const auto step = [&](double h, double err, double ratio) {
    PcFdStep s;
    s.h = h;
    s.fdTotal = c.edotTotal + err;
    s.err = err;
    s.ratio = ratio;
    s.displacedGap = {1e6, 2e6};
    for (int i = 0; i < 8; ++i)
      s.fdClass[static_cast<std::size_t>(i)] = c.classes[static_cast<std::size_t>(i)].edot;
    return s;
  };
  const PcFdStep coarsest = step(1.0 / 512.0, 16.0 * e, 0.0);
  const PcFdStep coarse = step(1.0 / 1024.0, 4.0 * e, 4.0);
  const PcFdStep fine = step(1.0 / 2048.0, e, 4.0);
  c.steps = {coarsest, coarse, fine};
  c.richardson = (4.0 * fine.fdTotal - coarse.fdTotal) / 3.0;
  c.richardsonErr = std::fabs(c.richardson - c.edotTotal);
  c.allDone = true;
  return c;
}

bool gate(const PcTangentCheck& c) { return printPcTangentReport(c, kAtol, kIdTol, kRatioTol); }

// --- PcTangentReportTests ------------------------------------------------------

TEST(PcTangentReportTests, ACleanCheckPasses) { EXPECT_TRUE(gate(cleanCheck())); }

TEST(PcTangentReportTests, ARefusedClassFailsTheGate) {
  // What the gap check does to a class, the tangent inherits: the class comes
  // back Refused and the report must fail rather than treat its derivative as
  // zero.
  for (int i = 0; i < 8; ++i) {
    PcTangentCheck c = cleanCheck();
    PcClassTangent& k = c.classes[static_cast<std::size_t>(i)];
    k.result.status = PcStatus::Refused;
    k.result.why = "S's gap 7.9e-01 < PC_MIN_GAP 1e+03";
    k.edot = 0.0;  // meaningless, and the report must not accept it as an answer
    c.allDone = false;
    EXPECT_FALSE(gate(c)) << "class index " << i;
  }
}

TEST(PcTangentReportTests, AClassNeverComputedFailsTheGate) {
  PcTangentCheck c = cleanCheck();
  c.classes[5].result.status = PcStatus::NotYet;
  c.allDone = false;
  EXPECT_FALSE(gate(c));
}

TEST(PcTangentReportTests, AnEnergyThatDisagreesWithTheEnergyPathFails) {
  // The tangent path builds each class's S and K itself, so this is the check
  // that the two transcriptions did not drift.
  PcTangentCheck c = cleanCheck();
  c.energyRef[3] += 1e-7;
  EXPECT_FALSE(gate(c));
  // ... and a difference well inside the tolerance is accepted.
  PcTangentCheck ok = cleanCheck();
  ok.energyRef[3] += 1e-13;
  EXPECT_TRUE(gate(ok));
}

TEST(PcTangentReportTests, EulerIsGatedOnTheHomogeneousClassesOnly) {
  for (int i = 0; i < 8; ++i) {
    PcTangentCheck c = cleanCheck();
    // Break Euler's identity for class i alone.
    c.eulerEdot[static_cast<std::size_t>(i)] = 2.0 * c.classes[static_cast<std::size_t>(i)].result.energy + 0.5;
    const bool passed = gate(c);
    EXPECT_EQ(passed, !kPcHomogeneousInCi[static_cast<std::size_t>(i)])
        << "class index " << i;
  }
}

TEST(PcTangentReportTests, ARatioAwayFromFourFails) {
  PcTangentCheck c = cleanCheck();
  c.steps.back().ratio = 2.0;  // first order, or noise-dominated
  EXPECT_FALSE(gate(c));
  // The first step has no predecessor, so its ratio is 0 and not gated.
  PcTangentCheck ok = cleanCheck();
  ok.steps.front().ratio = 0.0;
  EXPECT_TRUE(gate(ok));
}

TEST(PcTangentReportTests, AnExtrapolationAwayFromTheTangentFails) {
  PcTangentCheck c = cleanCheck();
  c.richardsonErr = 1e-6;
  EXPECT_FALSE(gate(c));
}

// --- PcTangentLadderTests ------------------------------------------------------

TEST(PcTangentLadderTests, NoStepsIsNotALadder) {
  // A finite difference gated on no step at all has measured nothing: the
  // exact checks passing is not enough.
  PcTangentCheck c = cleanCheck();
  c.steps.clear();
  EXPECT_FALSE(gate(c));
}

TEST(PcTangentLadderTests, OneHalvingIsNotARun) {
  // Two steps make one ratio -- a single halving that happens to be 4 is not
  // an order of convergence (#41's error was invisible at any one step).
  PcTangentCheck c = cleanCheck();
  c.steps.erase(c.steps.begin());
  c.steps.front().ratio = 0.0;
  ASSERT_EQ(c.steps.size(), 2u);
  EXPECT_FALSE(gate(c));
}

TEST(PcTangentLadderTests, TheRunIsReadOffTheErrorsNotTheRatioField) {
  // The run is recomputed from the errors themselves, so a step whose stored
  // ratio says 4 while its error did not fall fourfold still breaks it.
  PcTangentCheck c = cleanCheck();
  c.steps[1].err = c.steps[0].err;  // no fall across the first halving
  EXPECT_FALSE(gate(c));
}

TEST(PcTangentLadderTests, ADisplacedStateWithNoGuardRecordFails) {
  // The conditioning guard must have run at EVERY displaced state; a state
  // that skipped it (no gap recorded) fails the gate, whichever step and
  // whichever side of it.
  for (std::size_t s = 0; s < 3; ++s)
    for (std::size_t side = 0; side < 2; ++side) {
      PcTangentCheck c = cleanCheck();
      c.steps[s].displacedGap[side] = 0.0;
      EXPECT_FALSE(gate(c)) << "step " << s << " side " << side;
    }
}

TEST(PcTangentLadderTests, ADisplacedGapBelowTheFloorFails) {
  PcTangentCheck c = cleanCheck();
  c.steps[2].displacedGap[1] = 0.5 * PC_MIN_GAP;
  EXPECT_FALSE(gate(c));
  PcTangentCheck ok = cleanCheck();
  ok.steps[2].displacedGap[1] = PC_MIN_GAP;
  EXPECT_TRUE(gate(ok));
}

}  // namespace nevpt2::test::pc_tangent
