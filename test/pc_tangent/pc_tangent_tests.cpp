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
//   PcHessianReportTests  the --pc-hessian gate (#152): a clean check passes;
//                         a class refused along (u, v) -- gap, denominator or
//                         lift -- fails it; Euler is gated on the homogeneous
//                         classes only; the three-term tangent must match the
//                         two-term one; a negative control that answers
//                         instead of refusing, or refuses for another reason,
//                         fails; the ladder and the extrapolation as the
//                         tangent's
//   PcHessianVectorReportTests  the --pc-hessian-vector gate (#155): a clean
//                         check passes; a class refused along u fails; the
//                         walk's own Edot(u) and <u, y>, the duality per class
//                         (a pair pcHessianDevice refused, NaN, never passes)
//                         and the symmetry are gated; Euler on the homogeneous
//                         classes only; the control as --pc-hessian's; a
//                         plateau in ANY class's difference fails, as does a
//                         class with H_cc u = 0 that does not difference to 0
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

// --- PcHessianReportTests (#152) --------------------------------------------------

// A Hessian check that passes: every class computed, the transcription and
// Euler exact, the control refusing Sr, Si and Sir for their lift, and the
// tangent check's own clean ladder, re-centred on Eddot.
PcHessianCheck cleanHessian() {
  PcHessianCheck c;
  const PcTangentCheck t = cleanCheck();
  for (std::size_t i = 0; i < 8; ++i) {
    PcClassHessian& k = c.classes[i];
    k.result = t.classes[i].result;
    k.result.spectrum.smallestKept = 1e-3;
    k.result.spectrum.gap = 1e9;
    k.edotU = 0.002 * static_cast<double>(i + 1);
    k.edotV = -0.001 * static_cast<double>(i + 1);
    k.eddot = t.classes[i].edot;
    c.energyRef[i] = k.result.energy;
    c.edotURef[i] = k.edotU;
    c.edotVRef[i] = k.edotV;
    c.eulerEddot[i] = k.edotU;
    c.eddotTotal += k.eddot;
    PcClassHessian& w = c.control[i];
    w.result.status = PcStatus::Done;
    if (i == 0 || i == 1 || i == 7) {
      w.result.status = PcStatus::Refused;
      w.result.why = "S's dropped modes lift along this direction pair";
      w.lift = 0.4;
    }
  }
  c.steps = t.steps;
  c.richardson = t.richardson;
  c.richardsonErr = t.richardsonErr;
  c.allDone = true;
  return c;
}

bool hessGate(const PcHessianCheck& c) {
  return printPcHessianReport(c, kAtol, kIdTol, kIdTol, kRatioTol);
}

TEST(PcHessianReportTests, ACleanCheckPasses) { EXPECT_TRUE(hessGate(cleanHessian())); }

TEST(PcHessianReportTests, ARefusedClassFailsTheGate) {
  // A gap, denominator or lift refusal along (u, v) is the answer: the gate
  // fails rather than reading the class's (meaningless) Eddot as zero.
  for (const char* why : {"S's gap 7.9e-01 < PC_MIN_GAP 1e+03",
                          "S's dropped modes lift along this direction pair"})
    for (std::size_t i = 0; i < 8; ++i) {
      PcHessianCheck c = cleanHessian();
      c.classes[i].result.status = PcStatus::Refused;
      c.classes[i].result.why = why;
      c.classes[i].eddot = 0.0;
      c.allDone = false;
      EXPECT_FALSE(hessGate(c)) << "class index " << i << ": " << why;
    }
}

TEST(PcHessianReportTests, EulerIsGatedOnTheHomogeneousClassesOnly) {
  for (std::size_t i = 0; i < 8; ++i) {
    PcHessianCheck c = cleanHessian();
    c.eulerEddot[i] += 0.5;
    EXPECT_EQ(hessGate(c), !kPcHomogeneousInCi[i]) << "class index " << i;
  }
}

TEST(PcHessianReportTests, AThreeTermTangentAwayFromTheTwoTermOneFails) {
  PcHessianCheck c = cleanHessian();
  c.edotVRef[4] += 1e-6;
  EXPECT_FALSE(hessGate(c));
}

TEST(PcHessianReportTests, AControlThatAnswersInsteadOfRefusingFails) {
  // Nothing refused along the symmetry-breaking direction: the lift test is
  // off, or not reaching the classes it must.
  PcHessianCheck none = cleanHessian();
  for (PcClassHessian& w : none.control) {
    w.result.status = PcStatus::Done;
    w.lift = 0.0;
  }
  EXPECT_FALSE(hessGate(none));
  // A control refused for some OTHER reason is not the lift test working.
  PcHessianCheck other = cleanHessian();
  other.control[3].result.status = PcStatus::Refused;
  other.control[3].lift = 0.0;
  EXPECT_FALSE(hessGate(other));
}

TEST(PcHessianReportTests, TheLadderAndTheExtrapolationAreGatedAsTheTangents) {
  PcHessianCheck ratio = cleanHessian();
  ratio.steps.back().ratio = 1.0;  // the plateau #152's first ladder showed
  EXPECT_FALSE(hessGate(ratio));
  PcHessianCheck far = cleanHessian();
  far.richardsonErr = 1e-6;
  EXPECT_FALSE(hessGate(far));
  PcHessianCheck unguarded = cleanHessian();
  unguarded.steps[1].displacedGap[0] = 0.0;
  EXPECT_FALSE(hessGate(unguarded));
}

// --- PcHessianVectorReportTests (#155) ------------------------------------------

constexpr double kDualTol = 1e-9;
constexpr double kFdTol = 1e-8;

// A Hessian-vector check that passes: every class computed, the walk's
// tangent and <u, y> exact, two duality rows and the symmetry exact, Euler
// exact, the control refusing Sr, Si and Sir for their lift, and four steps
// whose per-class errors fall fourfold, with Sijrs's H_cc u exactly zero.
PcHessianVectorCheck cleanVector() {
  PcHessianVectorCheck c;
  const PcTangentCheck t = cleanCheck();
  c.allDone = true;
  for (std::size_t i = 0; i < 8; ++i) {
    c.result[i] = t.classes[i].result;
    c.result[i].spectrum.smallestKept = 1e-3;
    c.walkEdot[i] = c.edotRef[i] = 0.003 * static_cast<double>(i + 1);
    c.uy += c.edotRef[i];
    c.control[i].status = PcStatus::Done;
    if (i == 0 || i == 1 || i == 7) {
      c.control[i].status = PcStatus::Refused;
      c.control[i].why = "S's dropped modes lift along this direction pair";
      c.controlLift[i] = 0.4;
    }
  }
  for (const char* what : {"seed 0x5eed0152", "the state itself"}) {
    PcHessianVectorCheck::Dual row;
    row.what = what;
    for (std::size_t i = 0; i < 8; ++i) row.lhs[i] = row.rhs[i] = 0.1 * static_cast<double>(i);
    c.duals.push_back(row);
  }
  c.vHu = c.uHv = 5.3;
  for (std::size_t k = 0; k < 9; ++k) c.huNorm[k] = k == 2 ? 0.0 : 1.0 + static_cast<double>(k);
  double h = 1.0 / 4096.0, scale = 1e-5;
  for (int s = 0; s < 4; ++s) {
    PcHessianVectorCheck::Step st;
    st.h = h;
    st.displacedGap = {1e9, 1e9};
    for (std::size_t k = 0; k < 9; ++k) st.err[k] = k == 2 ? 0.0 : scale * c.huNorm[k];
    c.steps.push_back(st);
    h /= 2.0;
    scale /= 4.0;
  }
  for (std::size_t k = 0; k < 9; ++k) c.richardsonErr[k] = k == 2 ? 0.0 : 1e-11 * c.huNorm[k];
  return c;
}

bool vectorGate(const PcHessianVectorCheck& c) {
  return printPcHessianVectorReport(c, kIdTol, kDualTol, kFdTol, kRatioTol);
}

TEST(PcHessianVectorReportTests, ACleanCheckPasses) { EXPECT_TRUE(vectorGate(cleanVector())); }

TEST(PcHessianVectorReportTests, ARefusedClassFailsTheGate) {
  // A gap, denominator or lift refusal along u is the answer: no vector.
  for (const char* why : {"S's gap 7.9e-01 < PC_MIN_GAP 1e+03",
                          "S's dropped modes lift along this direction pair"})
    for (std::size_t i = 0; i < 8; ++i) {
      PcHessianVectorCheck c = cleanVector();
      c.result[i].status = PcStatus::Refused;
      c.result[i].why = why;
      c.allDone = false;
      EXPECT_FALSE(vectorGate(c)) << "class index " << i << ": " << why;
    }
}

TEST(PcHessianVectorReportTests, TheWalksTangentAndItsYAreGated) {
  PcHessianVectorCheck edot = cleanVector();
  edot.walkEdot[3] += 1e-6;
  EXPECT_FALSE(vectorGate(edot));
  PcHessianVectorCheck uy = cleanVector();
  uy.uy += 1e-6;
  EXPECT_FALSE(vectorGate(uy));
}

TEST(PcHessianVectorReportTests, TheDualityIsGatedPerClassAndOnARefusedPair) {
  for (std::size_t i = 0; i < 8; ++i) {
    PcHessianVectorCheck c = cleanVector();
    c.duals[1].lhs[i] += 1e-6;
    EXPECT_FALSE(vectorGate(c)) << "class index " << i;
  }
  // pcHessianDevice refusing the pair leaves no Eddot (NaN): never a pass.
  PcHessianVectorCheck refused = cleanVector();
  refused.duals[0].rhs[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(vectorGate(refused));
  PcHessianVectorCheck sym = cleanVector();
  sym.uHv += 1e-6;
  EXPECT_FALSE(vectorGate(sym));
}

TEST(PcHessianVectorReportTests, EulerIsGatedOnTheHomogeneousClassesOnly) {
  for (std::size_t i = 0; i < 8; ++i) {
    PcHessianVectorCheck c = cleanVector();
    c.euler[i] = 0.5;
    EXPECT_EQ(vectorGate(c), !kPcHomogeneousInCi[i]) << "class index " << i;
  }
}

TEST(PcHessianVectorReportTests, AControlThatAnswersInsteadOfRefusingFails) {
  PcHessianVectorCheck none = cleanVector();
  for (std::size_t i = 0; i < 8; ++i) {
    none.control[i].status = PcStatus::Done;
    none.controlLift[i] = 0.0;
  }
  EXPECT_FALSE(vectorGate(none));
  PcHessianVectorCheck other = cleanVector();
  other.control[3].status = PcStatus::Refused;
  EXPECT_FALSE(vectorGate(other));
}

TEST(PcHessianVectorReportTests, APlateauInAnyClassFails) {
  // The plateau the unrepaired formula leaves in Sr and Sir
  // (docs/pc-nevpt2.md, "What the cut drops, and how it moves"), in any class
  // with a nonzero H_cc u -- even when the total converges.
  for (const std::size_t k : {0u, 1u, 3u, 4u, 5u, 6u, 7u, 8u}) {
    PcHessianVectorCheck c = cleanVector();
    c.steps[3].err[k] = c.steps[2].err[k];
    EXPECT_FALSE(vectorGate(c)) << "index " << k;
  }
  PcHessianVectorCheck far = cleanVector();
  far.richardsonErr[7] = 1e-6 * far.huNorm[7];
  EXPECT_FALSE(vectorGate(far));
  // A class whose H_cc u is exactly zero must difference to exactly zero.
  PcHessianVectorCheck zero = cleanVector();
  zero.richardsonErr[2] = 1e-15;
  EXPECT_FALSE(vectorGate(zero));
  PcHessianVectorCheck unguarded = cleanVector();
  unguarded.steps[1].displacedGap[1] = 0.0;
  EXPECT_FALSE(vectorGate(unguarded));
}

}  // namespace nevpt2::test::pc_tangent
