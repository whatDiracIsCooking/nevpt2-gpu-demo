// Suites for nevpt2.gradient_plan (src/gradient/gradient_plan.cppm) -- the
// gradient path's pure host arithmetic, which the whole-path `gpu` duality
// entries check only GLOBALLY (a wrong array choice there either fails without
// saying where, or cancels on both sides of the dot product).
//
// Host-only ON PURPOSE: no device, no kernel, no golden -- so this binary is
// not REQUIRES_GPU and runs with no card visible. Every expected value below
// is written out by hand from docs/gradient-theory.md, Sec. 3 (the promotion
// rule and the seventeen-array inventory) or from small exactly-representable
// inputs, never recomputed by the code under test, and every floating-point
// comparison is EXACT (dyadic inputs, so the arithmetic is exact too).
//
//   RotBlockTableTests        rotBlockInfo's roles per block (space and storage
//                             axis per slot, X[p,q,r,s] == (p r | q s)),
//                             activeAxesOf, rotBlockName, spanOf
//   GeneralPickTests          all 27 (P, Q, R) space triples against the
//                             inventory: the five stored (Q, R) orderings, the
//                             swap that normalises onto them, and the two that
//                             are refused -- (v,v) in one distribution, the
//                             absent (xv|vv), and the untied (xc|cc)
//   PromotionPlanTests        every two-electron block's four slots: the array
//                             and swap, the einsum strings for Sr / Sijrs /
//                             Sijr, and that no slot ever asks for (v, v)
//   TermClassificationTests   termHasActive
//   DeltaTests                each class's Delta vector, in its tuple order,
//                             and its linearity and homogeneity in the orbital
//                             energies (what makes Deltadot the same formula)
//   SlabMultiplierTests       plainMultipliers; sijrMultipliers and
//                             srsiMultipliers on hand triangles, their
//                             symmetric scatter and perturber-order R; the
//                             homogeneous-vs-affine distinction (#32)
//   CiSourceHostTests         ciSourceAddInto / ciSourceSymmetrized /
//                             ciSourceFolded's basic shape (#63 has the rest)
//   FourfoldRunTests          longestFourfoldRun, the step-ladder rule: a run
//                             of halvings inside the open window, broken by a
//                             wild row or a zero, the longest (then coarsest)
//                             reported (#70)
//   GradientPlanDeathTest     a promotion of a non-two-electron slot, and a
//                             shape mismatch in ciSourceAddInto, abort
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.gradient_plan;

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::gradient_plan {

using S = RotSpace;
using G = GeneralArray;

Tensor vec(std::vector<double> v) {
  const int64_t n = std::ssize(v);
  return Tensor::fromFlat({n}, std::move(v));
}

// ---------------------------------------------------------------------------
// RotBlockTableTests
// ---------------------------------------------------------------------------

struct ExpectedBlock {
  RotBlock b;
  int rank;
  std::vector<std::pair<S, int>> roles;  // (space, axis) for p, q[, r, s]
  bool trace;
  std::vector<int> activeAxes;
  const char* name;
};

// Read off docs/gradient-theory.md Sec. 3.2's table (the chemists' form of
// each golden field) under X[p,q,r,s] == (p r | q s): h2e_v_Sr is (rb|tu) at
// storage (r, t, b, u), so (p, q, r, s) = (v, a, a, a) on axes 0..3; cvcv is
// stored chemists' (i r | j s), so p = i (axis 0), r = r (axis 1), q = j
// (axis 2), s = s (axis 3).
std::vector<ExpectedBlock> expectedBlocks() {
  return {
      {RotBlock::Sr, 4, {{S::Virt, 0}, {S::Act, 1}, {S::Act, 2}, {S::Act, 3}}, false,
       {1, 2, 3}, "h2e_v_Sr"},
      {RotBlock::Si, 4, {{S::Act, 0}, {S::Act, 1}, {S::Core, 2}, {S::Act, 3}}, false,
       {0, 1, 3}, "h2e_v_Si"},
      {RotBlock::Sijrs, 4, {{S::Core, 0}, {S::Core, 2}, {S::Virt, 1}, {S::Virt, 3}}, false,
       {}, "cvcv"},
      {RotBlock::Sijr, 4, {{S::Virt, 0}, {S::Act, 1}, {S::Core, 2}, {S::Core, 3}}, false,
       {1}, "h2e_v_Sijr"},
      {RotBlock::Srsi, 4, {{S::Virt, 0}, {S::Virt, 1}, {S::Core, 2}, {S::Act, 3}}, false,
       {3}, "h2e_v_Srsi"},
      {RotBlock::SrsiT, 4, {{S::Virt, 0}, {S::Virt, 1}, {S::Core, 2}, {S::Act, 3}}, false,
       {3}, "h2e_v_Srsi(T)"},
      {RotBlock::Srs, 4, {{S::Virt, 0}, {S::Virt, 1}, {S::Act, 2}, {S::Act, 3}}, false,
       {2, 3}, "h2e_v_Srs"},
      {RotBlock::Sij, 4, {{S::Act, 0}, {S::Act, 1}, {S::Core, 2}, {S::Core, 3}}, false,
       {0, 1}, "h2e_v_Sij"},
      {RotBlock::Sir1, 4, {{S::Virt, 0}, {S::Act, 1}, {S::Core, 2}, {S::Act, 3}}, false,
       {1, 3}, "h2e_v1_Sir"},
      {RotBlock::Sir2, 4, {{S::Virt, 0}, {S::Act, 1}, {S::Act, 2}, {S::Core, 3}}, false,
       {1, 2}, "h2e_v2_Sir"},
      {RotBlock::H1eSr, 2, {{S::Virt, 0}, {S::Act, 1}}, true, {1}, "h1e_v_Sr"},
      {RotBlock::H1eSi, 2, {{S::Act, 0}, {S::Core, 1}}, false, {0}, "h1e_v_Si"},
      {RotBlock::H1eSir, 2, {{S::Virt, 0}, {S::Core, 1}}, false, {}, "h1e_v_Sir"},
      {RotBlock::V2tr, 0, {}, false, {}, "v2tr"},
      {RotBlock::H1e, 2, {{S::Act, 0}, {S::Act, 1}}, false, {0, 1}, "h1e"},
      {RotBlock::H2e, 4, {{S::Act, 0}, {S::Act, 1}, {S::Act, 2}, {S::Act, 3}}, false,
       {0, 1, 2, 3}, "h2e"},
  };
}

TEST(RotBlockTableTests, EveryBlockIsCoveredOnce) {
  const std::vector<ExpectedBlock> exp = expectedBlocks();
  ASSERT_EQ(exp.size(), kAllRotBlocks.size());
  for (std::size_t i = 0; i < exp.size(); ++i) EXPECT_EQ(exp[i].b, kAllRotBlocks[i]);
}

TEST(RotBlockTableTests, RolesRankAndTraceMatchTheInventory) {
  for (const ExpectedBlock& e : expectedBlocks()) {
    SCOPED_TRACE(e.name);
    const RotBlockInfo info = rotBlockInfo(e.b);
    EXPECT_EQ(info.rank, e.rank);
    EXPECT_EQ(info.trace, e.trace);
    ASSERT_EQ(std::ssize(e.roles), info.rank);
    for (int k = 0; k < info.rank; ++k) {
      SCOPED_TRACE(k);
      EXPECT_EQ(info.roles[static_cast<std::size_t>(k)].space,
                e.roles[static_cast<std::size_t>(k)].first);
      EXPECT_EQ(info.roles[static_cast<std::size_t>(k)].axis,
                e.roles[static_cast<std::size_t>(k)].second);
    }
  }
}

TEST(RotBlockTableTests, RolesAreAPermutationOfTheStorageAxes) {
  for (const RotBlock b : kAllRotBlocks) {
    SCOPED_TRACE(rotBlockName(b));
    const RotBlockInfo info = rotBlockInfo(b);
    std::vector<int> axes;
    for (int k = 0; k < info.rank; ++k) axes.push_back(info.roles[static_cast<std::size_t>(k)].axis);
    std::ranges::sort(axes);
    for (int k = 0; k < info.rank; ++k) EXPECT_EQ(axes[static_cast<std::size_t>(k)], k);
  }
}

TEST(RotBlockTableTests, ActiveAxesOf) {
  for (const ExpectedBlock& e : expectedBlocks()) {
    SCOPED_TRACE(e.name);
    EXPECT_EQ(activeAxesOf(e.b), e.activeAxes);
  }
}

TEST(RotBlockTableTests, RotBlockName) {
  for (const ExpectedBlock& e : expectedBlocks()) EXPECT_STREQ(rotBlockName(e.b), e.name);
}

TEST(RotBlockTableTests, SpanOfRunsCoreThenActiveThenVirtual) {
  using P = std::pair<int64_t, int64_t>;
  EXPECT_EQ(spanOf(2, 3, 4, S::Core), (P{0, 2}));
  EXPECT_EQ(spanOf(2, 3, 4, S::Act), (P{2, 5}));
  EXPECT_EQ(spanOf(2, 3, 4, S::Virt), (P{5, 9}));
  // An empty core (CAS over all occupied orbitals) is an empty span at 0.
  EXPECT_EQ(spanOf(0, 4, 6, S::Core), (P{0, 0}));
  EXPECT_EQ(spanOf(0, 4, 6, S::Act), (P{0, 4}));
  EXPECT_EQ(spanOf(0, 4, 6, S::Virt), (P{4, 10}));
}

// ---------------------------------------------------------------------------
// GeneralPickTests
// ---------------------------------------------------------------------------

struct ExpectedPick {
  S p, q, r;
  std::optional<GeneralPickPlan> want;
};

std::optional<GeneralPickPlan> pick(G a, bool swap) { return GeneralPickPlan{a, swap}; }

// All 27 triples, by hand. The stored (Q, R) orderings are (a,a), (v,a),
// (c,a), (c,v) and (c,c) (Sec. 3.3's inventory); the reversed ones read the
// same array swapped. (v,v) is the absent (xv|vv) family and (c,c) with P = c
// is (xc|cc), which is stored only with two labels tied -- both refused.
std::vector<ExpectedPick> expectedPicks() {
  return {
      // P = active
      {S::Act, S::Act, S::Act, pick(G::XaAa, false)},
      {S::Act, S::Virt, S::Act, pick(G::XaVa, false)},
      {S::Act, S::Act, S::Virt, pick(G::XaVa, true)},
      {S::Act, S::Core, S::Act, pick(G::XaCa, false)},
      {S::Act, S::Act, S::Core, pick(G::XaCa, true)},
      {S::Act, S::Core, S::Virt, pick(G::XaCv, false)},
      {S::Act, S::Virt, S::Core, pick(G::XaCv, true)},
      {S::Act, S::Core, S::Core, pick(G::XaCc, false)},
      {S::Act, S::Virt, S::Virt, std::nullopt},
      // P = virtual
      {S::Virt, S::Act, S::Act, pick(G::XvAa, false)},
      {S::Virt, S::Virt, S::Act, pick(G::XvVa, false)},
      {S::Virt, S::Act, S::Virt, pick(G::XvVa, true)},
      {S::Virt, S::Core, S::Act, pick(G::XvCa, false)},
      {S::Virt, S::Act, S::Core, pick(G::XvCa, true)},
      {S::Virt, S::Core, S::Virt, pick(G::XvCv, false)},
      {S::Virt, S::Virt, S::Core, pick(G::XvCv, true)},
      {S::Virt, S::Core, S::Core, pick(G::XvCc, false)},
      {S::Virt, S::Virt, S::Virt, std::nullopt},  // (xv|vv): Sec. 3.4's absent array
      // P = core
      {S::Core, S::Act, S::Act, pick(G::XcAa, false)},
      {S::Core, S::Virt, S::Act, pick(G::XcVa, false)},
      {S::Core, S::Act, S::Virt, pick(G::XcVa, true)},
      {S::Core, S::Core, S::Act, pick(G::XcCa, false)},
      {S::Core, S::Act, S::Core, pick(G::XcCa, true)},
      {S::Core, S::Core, S::Virt, pick(G::XcCv, false)},
      {S::Core, S::Virt, S::Core, pick(G::XcCv, true)},
      {S::Core, S::Core, S::Core, std::nullopt},  // (xc|cc): only ever tied
      {S::Core, S::Virt, S::Virt, std::nullopt},
  };
}

const char* spaceName(S s) { return s == S::Core ? "c" : (s == S::Act ? "a" : "v"); }

TEST(GeneralPickTests, AllTwentySevenTriplesMatchTheInventory) {
  const std::vector<ExpectedPick> exp = expectedPicks();
  ASSERT_EQ(exp.size(), 27u);
  std::set<std::tuple<S, S, S>> seen;
  for (const ExpectedPick& e : exp) {
    SCOPED_TRACE(std::format("(x{}|{}{})", spaceName(e.p), spaceName(e.q), spaceName(e.r)));
    seen.insert({e.p, e.q, e.r});
    const std::optional<GeneralPickPlan> got = generalPickPlan(e.p, e.q, e.r);
    ASSERT_EQ(got.has_value(), e.want.has_value());
    if (!got) continue;
    EXPECT_STREQ(generalArrayName(got->array), generalArrayName(e.want->array));
    EXPECT_EQ(got->swap, e.want->swap);
  }
  EXPECT_EQ(seen.size(), 27u);  // the table above is every triple, once
}

TEST(GeneralPickTests, NoTwoVirtualsInOneDistribution) {
  for (const S p : {S::Core, S::Act, S::Virt})
    EXPECT_FALSE(generalPickPlan(p, S::Virt, S::Virt).has_value()) << spaceName(p);
}

TEST(GeneralPickTests, FourteenPickableArraysEachReachedOnceUnswapped) {
  std::map<std::string, int> unswapped;
  for (const ExpectedPick& e : expectedPicks())
    if (e.want && !e.want->swap) ++unswapped[generalArrayName(e.want->array)];
  EXPECT_EQ(unswapped.size(), 14u);
  for (const auto& [name, count] : unswapped) EXPECT_EQ(count, 1) << name;
}

TEST(GeneralPickTests, ArrayNames) {
  EXPECT_STREQ(generalArrayName(G::XaAa), "g_xa_aa");
  EXPECT_STREQ(generalArrayName(G::XvVa), "g_xv_va");
  EXPECT_STREQ(generalArrayName(G::XcCa), "g_xc_ca");
  EXPECT_STREQ(generalArrayName(G::XvCv), "g_xv_cv");
  EXPECT_STREQ(generalArrayName(G::XaCc), "g_xa_cc");
  EXPECT_STREQ(generalArrayName(G::XvCc), "g_xv_cc");
}

// ---------------------------------------------------------------------------
// PromotionPlanTests
// ---------------------------------------------------------------------------

// The array (and swap) each slot of each two-electron block promotes into,
// derived by hand: slot p's source is (x P | Q R) with P = space(r) and
// (Q, R) = (space(q), space(s)); q -> P = space(s), (space(p), space(r));
// r -> P = space(p), (space(q), space(s)); s -> P = space(q), (space(p),
// space(r)) -- then normalised onto a stored ordering.
struct ExpectedSlots {
  RotBlock b;
  std::array<std::pair<G, bool>, 4> slots;  // p, q, r, s
};

std::vector<ExpectedSlots> expectedSlots() {
  return {
      // Sr (v, a, a, a)
      {RotBlock::Sr,
       {{{G::XaAa, false}, {G::XaVa, false}, {G::XvAa, false}, {G::XaVa, false}}}},
      // Si (a, a, c, a)
      {RotBlock::Si, {{{G::XcAa, false}, {G::XaCa, true}, {G::XaAa, false}, {G::XaCa, true}}}},
      // Sijrs: p = c, q = c, r = v, s = v
      {RotBlock::Sijrs,
       {{{G::XvCv, false}, {G::XvCv, false}, {G::XcCv, false}, {G::XcCv, false}}}},
      // Sijr (v, a, c, c)
      {RotBlock::Sijr, {{{G::XcCa, true}, {G::XcCv, true}, {G::XvCa, true}, {G::XaCv, true}}}},
      // Srsi and its batched twin (v, v, c, a)
      {RotBlock::Srsi,
       {{{G::XcVa, false}, {G::XaCv, true}, {G::XvVa, false}, {G::XvCv, true}}}},
      {RotBlock::SrsiT,
       {{{G::XcVa, false}, {G::XaCv, true}, {G::XvVa, false}, {G::XvCv, true}}}},
      // Srs (v, v, a, a)
      {RotBlock::Srs, {{{G::XaVa, false}, {G::XaVa, false}, {G::XvVa, false}, {G::XvVa, false}}}},
      // Sij (a, a, c, c)
      {RotBlock::Sij, {{{G::XcCa, true}, {G::XcCa, true}, {G::XaCa, true}, {G::XaCa, true}}}},
      // Sir1 (v, a, c, a)
      {RotBlock::Sir1, {{{G::XcAa, false}, {G::XaCv, true}, {G::XvAa, false}, {G::XaCv, true}}}},
      // Sir2 (v, a, a, c)
      {RotBlock::Sir2, {{{G::XaCa, true}, {G::XcVa, false}, {G::XvCa, true}, {G::XaVa, false}}}},
      // h2e (a, a, a, a)
      {RotBlock::H2e, {{{G::XaAa, false}, {G::XaAa, false}, {G::XaAa, false}, {G::XaAa, false}}}},
  };
}

TEST(PromotionPlanTests, EveryTwoElectronSlotPicksTheHandDerivedArray) {
  const std::vector<ExpectedSlots> exp = expectedSlots();
  // Every rank-4 block is in the table.
  int rank4 = 0;
  for (const RotBlock b : kAllRotBlocks) rank4 += rotBlockInfo(b).rank == 4 ? 1 : 0;
  ASSERT_EQ(std::ssize(exp), rank4);
  for (const ExpectedSlots& e : exp) {
    for (int slot = 0; slot < 4; ++slot) {
      SCOPED_TRACE(std::format("{} slot {}", rotBlockName(e.b), "pqrs"[slot]));
      const PromotionPlan pl = promotionPlan(e.b, slot);
      const auto [arr, swap] = e.slots[static_cast<std::size_t>(slot)];
      EXPECT_STREQ(generalArrayName(pl.pick.array), generalArrayName(arr));
      EXPECT_EQ(pl.pick.swap, swap);
      // The promoted role is the slot's own.
      EXPECT_EQ(pl.promoted.axis,
                rotBlockInfo(e.b).roles[static_cast<std::size_t>(slot)].axis);
    }
  }
}

// The einsum each rule IS, written out by hand. Sr: X[p,q,r,s] = (p r | q s)
// on axes (i, j, k, l), so promoting p reads (x r | q s) = g_xa_aa[z, k, j, l]
// against kappa[z, i], and so on.
TEST(PromotionPlanTests, SubscriptsForSr) {
  EXPECT_EQ(promotionSubscripts(promotionPlan(RotBlock::Sr, 0)), "zkjl,zi->ijkl");
  EXPECT_EQ(promotionSubscripts(promotionPlan(RotBlock::Sr, 1)), "zlik,zj->ijkl");
  EXPECT_EQ(promotionSubscripts(promotionPlan(RotBlock::Sr, 2)), "zijl,zk->ijkl");
  EXPECT_EQ(promotionSubscripts(promotionPlan(RotBlock::Sr, 3)), "zjik,zl->ijkl");
}

// cvcv's roles are NOT on their storage axes in order (q on axis 2, r on
// axis 1): (i r | j s) at storage (i, r, j, s).
TEST(PromotionPlanTests, SubscriptsForCvcvFollowItsChemistsLayout) {
  EXPECT_EQ(promotionSubscripts(promotionPlan(RotBlock::Sijrs, 0)), "zjkl,zi->ijkl");
  EXPECT_EQ(promotionSubscripts(promotionPlan(RotBlock::Sijrs, 1)), "zlij,zk->ijkl");
  EXPECT_EQ(promotionSubscripts(promotionPlan(RotBlock::Sijrs, 2)), "zikl,zj->ijkl");
  EXPECT_EQ(promotionSubscripts(promotionPlan(RotBlock::Sijrs, 3)), "zkij,zl->ijkl");
}

// A swapped pick puts the second distribution's labels in stored order: Sijr
// slot p reads (x r | q s) with (Q, R) = (a, c), stored as g_xc_ca[x, r, s, q].
TEST(PromotionPlanTests, SubscriptsForASwappedPick) {
  const PromotionPlan pl = promotionPlan(RotBlock::Sijr, 0);
  EXPECT_TRUE(pl.pick.swap);
  EXPECT_EQ(pl.partner.axis, 2);
  EXPECT_EQ(pl.firstAxis, 3);
  EXPECT_EQ(pl.lastAxis, 1);
  EXPECT_EQ(promotionSubscripts(pl), "zklj,zi->ijkl");
}

// #30 Sec. 3.4: no promotion of any block the energy reads ever needs
// (x? | v v) -- the eighteenth array, and the largest.
TEST(PromotionPlanTests, NoSlotEverAsksForTwoVirtualsInOneDistribution) {
  for (const RotBlock b : kAllRotBlocks) {
    const RotBlockInfo info = rotBlockInfo(b);
    if (info.rank != 4) continue;
    static constexpr int kSecond[4][2] = {{1, 3}, {0, 2}, {1, 3}, {0, 2}};
    for (int slot = 0; slot < 4; ++slot) {
      const S q = info.roles[static_cast<std::size_t>(kSecond[slot][0])].space;
      const S r = info.roles[static_cast<std::size_t>(kSecond[slot][1])].space;
      EXPECT_FALSE(q == S::Virt && r == S::Virt) << rotBlockName(b) << " slot " << slot;
    }
  }
}

// ---------------------------------------------------------------------------
// TermClassificationTests
// ---------------------------------------------------------------------------

struct FakeOperand {
  int block = kScActive;
};
struct FakeTerm {
  std::vector<FakeOperand> ops;
};

TEST(TermClassificationTests, TermHasActive) {
  EXPECT_TRUE(termHasActive(FakeTerm{{{0}, {0}, {kScActive}}}));
  EXPECT_TRUE(termHasActive(FakeTerm{{{kScActive}}}));
  EXPECT_FALSE(termHasActive(FakeTerm{{{0}, {1}}}));
  EXPECT_FALSE(termHasActive(FakeTerm{{}}));
  EXPECT_EQ(kScActive, -1);
}

// ---------------------------------------------------------------------------
// DeltaTests
// ---------------------------------------------------------------------------

// Each class's Delta_t, by hand, in its tuple vector's order.
TEST(DeltaTests, SrIsEpsROverTheSlab) {
  EXPECT_EQ(srDelta(1, 3, vec({10, 20, 30, 40})), (std::vector<double>{20, 30}));
}

TEST(DeltaTests, SiIsMinusEpsI) {
  EXPECT_EQ(siDelta(0, 2, vec({-1, -2, -3})), (std::vector<double>{1, 2}));
}

TEST(DeltaTests, SijrsIsEpsAPlusEpsBMinusEpsIMinusEpsJInCvcvOrder) {
  // i = 1 only; then (a, j, b) with ec = {-1, -2}, ev = {10, 20}.
  EXPECT_EQ(sijrsDelta(1, 2, 2, 2, vec({-1, -2}), vec({10, 20})),
            (std::vector<double>{23, 33, 24, 34, 33, 43, 34, 44}));
}

TEST(DeltaTests, SijrRunsTheJLeqITriangle) {
  // r = b0 + 0 = 1 (eps 3); (j, i) = (0,0), (0,1), (1,1).
  EXPECT_EQ(sijrDelta(1, 2, 1, vec({-1, -2}), vec({5, 3})), (std::vector<double>{5, 6, 7}));
}

TEST(DeltaTests, SrsiRunsTheRLeqSTriangle) {
  // (r, s) = (0,0), (0,1), (1,1), one core index.
  EXPECT_EQ(srsiDelta(2, 1, vec({-1}), vec({1, 2})), (std::vector<double>{3, 4, 5}));
}

TEST(DeltaTests, SrsIsEpsRPlusEpsS) {
  EXPECT_EQ(srsDelta(1, 2, 3, vec({1, 2, 4})), (std::vector<double>{3, 4, 6}));
}

TEST(DeltaTests, SijIsMinusEpsIMinusEpsJ) {
  EXPECT_EQ(sijDelta(0, 2, 2, vec({-1, -2})), (std::vector<double>{2, 3, 3, 4}));
}

TEST(DeltaTests, SirIsEpsRMinusEpsICoreFirst) {
  EXPECT_EQ(sirDelta(1, 3, 2, vec({-1, -2}), vec({10, 20, 30})),
            (std::vector<double>{21, 31, 22, 32}));
}

// Delta_t is a signed sum of orbital energies, so every class's Delta is
// LINEAR AND HOMOGENEOUS in (e_core, e_virt): zero at zero, additive, and
// scaled by a scalar. That is what lets assembleSlab read the same function at
// the response deps to get Deltadot -- an affine Delta would need its constant
// subtracted. Integer inputs keep every sum exact.
TEST(DeltaTests, EveryClassIsLinearAndHomogeneousInTheOrbitalEnergies) {
  const int64_t nc = 3, nv = 4;
  const Tensor ec1 = vec({-3, -5, -7}), ev1 = vec({2, 4, 8, 16});
  const Tensor ec2 = vec({-1, 6, -2}), ev2 = vec({3, -9, 5, 1});
  const Tensor ecSum = vec({-4, 1, -9}), evSum = vec({5, -5, 13, 17});
  const Tensor ec3 = vec({-9, -15, -21}), ev3 = vec({6, 12, 24, 48});  // 3 * (ec1, ev1)
  const Tensor ec0 = vec({0, 0, 0}), ev0 = vec({0, 0, 0, 0});
  using Fn = std::function<std::vector<double>(const Tensor&, const Tensor&)>;
  const std::vector<std::pair<const char*, Fn>> classes{
      {"Sr", [&](const Tensor&, const Tensor& ev) { return srDelta(1, 3, ev); }},
      {"Si", [&](const Tensor& ec, const Tensor&) { return siDelta(0, 2, ec); }},
      {"Sijrs",
       [&](const Tensor& ec, const Tensor& ev) { return sijrsDelta(0, 2, nc, nv, ec, ev); }},
      {"Sijr", [&](const Tensor& ec, const Tensor& ev) { return sijrDelta(2, nc, 1, ec, ev); }},
      {"Srsi", [&](const Tensor& ec, const Tensor& ev) { return srsiDelta(nv, nc, ec, ev); }},
      {"Srs", [&](const Tensor&, const Tensor& ev) { return srsDelta(1, 4, nv, ev); }},
      {"Sij", [&](const Tensor& ec, const Tensor&) { return sijDelta(1, 3, nc, ec); }},
      {"Sir", [&](const Tensor& ec, const Tensor& ev) { return sirDelta(0, 3, nc, ec, ev); }},
  };
  for (const auto& [name, f] : classes) {
    SCOPED_TRACE(name);
    const std::vector<double> d1 = f(ec1, ev1), d2 = f(ec2, ev2), dSum = f(ecSum, evSum);
    const std::vector<double> d3 = f(ec3, ev3), d0 = f(ec0, ev0);
    ASSERT_FALSE(d1.empty());
    ASSERT_EQ(d1.size(), d2.size());
    ASSERT_EQ(d1.size(), dSum.size());
    for (std::size_t k = 0; k < d1.size(); ++k) {
      EXPECT_EQ(d0[k], 0.0);
      EXPECT_EQ(dSum[k], d1[k] + d2[k]);
      EXPECT_EQ(d3[k], 3.0 * d1[k]);
    }
  }
}

// The triangle classes' lengths: nr * ncore (ncore + 1) / 2 and
// nvirt (nvirt + 1) / 2 * ncore -- one entry per PERTURBER, not per tuple.
TEST(DeltaTests, TriangleClassesHaveOneEntryPerPerturber) {
  const Tensor ec = vec({1, 2, 3, 4}), ev = vec({5, 6, 7});
  EXPECT_EQ(sijrDelta(3, 4, 0, ec, ev).size(), 3u * 10u);
  EXPECT_EQ(srsiDelta(3, 4, ec, ev).size(), 6u * 4u);
}

// ---------------------------------------------------------------------------
// SlabMultiplierTests
// ---------------------------------------------------------------------------

// One perturber, N = 2, H = 2, Delta = 1: T = -1/(1 + 1) = -1/2, P = 2T +
// Delta T^2 = -3/4, Q = T^2 = 1/4, R = N T^2 = 1/2, E = T N = -1.
TEST(SlabMultiplierTests, PlainMultipliersOnOnePerturber) {
  const ScSlabMultipliers m = plainMultipliers(vec({2}), vec({2}), {1});
  EXPECT_EQ(m.p.dims(), (std::vector<int64_t>{1}));
  EXPECT_EQ(m.p(0), -0.75);
  EXPECT_EQ(m.q(0), 0.25);
  EXPECT_EQ(m.r, (std::vector<double>{0.5}));
  EXPECT_EQ(m.norm, 2.0);
  EXPECT_EQ(m.energy, -1.0);
}

TEST(SlabMultiplierTests, PlainMultipliersKeepTheTupleShape) {
  const Tensor n = Tensor::fromFlat({2, 1}, {2, 2});
  const Tensor h = Tensor::fromFlat({2, 1}, {2, -6});
  const ScSlabMultipliers m = plainMultipliers(n, h, {1, 5});
  EXPECT_EQ(m.p.dims(), (std::vector<int64_t>{2, 1}));
  EXPECT_EQ(m.q.dims(), (std::vector<int64_t>{2, 1}));
  EXPECT_EQ(m.p(0, 0), -0.75);
  EXPECT_EQ(m.p(1, 0), 0.25);  // T = -1/(5 - 3) = -1/2; P = -1 + 5/4
}

// Sijr on one virtual and two core orbitals: three perturbers (r, j <= i).
// Delta = {5, 6, 7} (DeltaTests.SijrRunsTheJLeqITriangle). The raw normT /
// hT and the per-perturber values worked by hand:
//   (0,0): N = (2 + 2)/2 = 2,  H = -6:  T = -1/(5 - 3) = -1/2,
//          P = -1 + 5/4 = 1/4,  Q = 1/4,  R = 1/2,  E = -1
//   (0,1): N = 1 + 1 = 2,      H = -10: T = -1/(6 - 5) = -1,
//          P = -2 + 6 = 4,      Q = 1,    R = 2,    E = -2
//   (1,1): N = 4,              H = -12: T = -1/(7 - 3) = -1/4,
//          P = -1/2 + 7/16 = -1/16, Q = 1/16, R = 1/4, E = -1
Tensor sijrNorm(double a, double b) {
  Tensor t({1, 2, 2});
  t(0, 0, 0) = 2;
  t(0, 0, 1) = a;
  t(0, 1, 0) = b;
  t(0, 1, 1) = 4;
  return t;
}

Tensor sijrH(double a, double b) {
  Tensor t({1, 2, 2});
  t(0, 0, 0) = -6;
  t(0, 0, 1) = a;
  t(0, 1, 0) = b;
  t(0, 1, 1) = -12;
  return t;
}

TEST(SlabMultiplierTests, SijrOnAHandTriangle) {
  const ScSlabMultipliers m =
      sijrMultipliers(sijrNorm(1, 1), sijrH(-5, -5), 1, vec({-1, -2}), vec({5, 3}));
  EXPECT_EQ(m.p.dims(), (std::vector<int64_t>{1, 2, 2}));
  EXPECT_EQ(m.p(0, 0, 0), 0.25);
  EXPECT_EQ(m.p(0, 0, 1), 4.0);
  EXPECT_EQ(m.p(0, 1, 0), 4.0);  // scattered back symmetric in (j, i)
  EXPECT_EQ(m.p(0, 1, 1), -1.0 / 16);
  EXPECT_EQ(m.q(0, 0, 0), 0.25);
  EXPECT_EQ(m.q(0, 0, 1), 1.0);
  EXPECT_EQ(m.q(0, 1, 0), 1.0);
  EXPECT_EQ(m.q(0, 1, 1), 1.0 / 16);
  // R stays in PERTURBER (triangle) order, the order sijrDelta packs.
  EXPECT_EQ(m.r, (std::vector<double>{0.5, 2.0, 0.25}));
  EXPECT_EQ(m.norm, 8.0);
  EXPECT_EQ(m.energy, -4.0);
}

// (r, j, i) and (r, i, j) are the same perturber, so only their SUM enters:
// splitting N = 2 as 1/2 + 3/2 instead of 1 + 1 (and H likewise) changes
// nothing -- and the scattered P against the RAW normT is the perturber sum
// of P against the symmetrized N, the identity sijrMultipliers' comment
// states.
TEST(SlabMultiplierTests, SijrSeesOnlyTheSymmetrizedPair) {
  const Tensor ec = vec({-1, -2}), ev = vec({5, 3});
  const ScSlabMultipliers even = sijrMultipliers(sijrNorm(1, 1), sijrH(-5, -5), 1, ec, ev);
  const Tensor nT = sijrNorm(0.5, 1.5);
  const ScSlabMultipliers odd = sijrMultipliers(nT, sijrH(-2, -8), 1, ec, ev);
  EXPECT_EQ(odd.p.data(), even.p.data());
  EXPECT_EQ(odd.q.data(), even.q.data());
  EXPECT_EQ(odd.r, even.r);
  EXPECT_EQ(odd.energy, even.energy);
  double raw = 0.0;
  for (int64_t k = 0; k < nT.size(); ++k) raw += odd.p.flat(k) * nT.flat(k);
  // Perturber side: 1/4 * 2 + 4 * 2 + (-1/16) * 4.
  EXPECT_EQ(raw, 0.25 * 2 + 4.0 * 2 + (-1.0 / 16) * 4);
}

// Srsi on two virtuals and one core orbital: perturbers (r <= s, i), Delta
// = {3, 4, 5} (DeltaTests.SrsiRunsTheRLeqSTriangle):
//   (0,0): N = 2, H = -2:  T = -1/(3 - 1) = -1/2, P = -1 + 3/4 = -1/4,
//          Q = 1/4,  R = 1/2,  E = -1
//   (0,1): N = 2, H = -6:  T = -1/(4 - 3) = -1,   P = -2 + 4 = 2,
//          Q = 1,    R = 2,    E = -2
//   (1,1): N = 4, H = -4:  T = -1/(5 - 1) = -1/4, P = -1/2 + 5/16 = -3/16,
//          Q = 1/16, R = 1/4,  E = -1
TEST(SlabMultiplierTests, SrsiOnAHandTriangle) {
  Tensor n({2, 2, 1}), h({2, 2, 1});
  n(0, 0, 0) = 2;
  n(0, 1, 0) = 0.5;
  n(1, 0, 0) = 1.5;
  n(1, 1, 0) = 4;
  h(0, 0, 0) = -2;
  h(0, 1, 0) = -1;
  h(1, 0, 0) = -5;
  h(1, 1, 0) = -4;
  const ScSlabMultipliers m = srsiMultipliers(n, h, vec({-1}), vec({1, 2}));
  EXPECT_EQ(m.p.dims(), (std::vector<int64_t>{2, 2, 1}));
  EXPECT_EQ(m.p(0, 0, 0), -0.25);
  EXPECT_EQ(m.p(0, 1, 0), 2.0);
  EXPECT_EQ(m.p(1, 0, 0), 2.0);  // symmetric in (r, s)
  EXPECT_EQ(m.p(1, 1, 0), -3.0 / 16);
  EXPECT_EQ(m.q(0, 1, 0), 1.0);
  EXPECT_EQ(m.q(1, 0, 0), 1.0);
  EXPECT_EQ(m.r, (std::vector<double>{0.5, 2.0, 0.25}));
  EXPECT_EQ(m.norm, 8.0);
  EXPECT_EQ(m.energy, -4.0);
}

// #32's distinction, at the level these multipliers see it. For the three
// HOMOGENEOUS classes (Sr, Srsi, Srs: metric and Dyall blocks linear in the
// RDMs) scaling the state by s scales every N and H by lambda = s^2, and the
// slab's energy scales with it exactly while T, P and Q do not move -- Euler,
// degree 2 in the CI vector. Srsi is the symmetrized one of the three.
TEST(SlabMultiplierTests, AHomogeneousClassScalesItsEnergyWithItsInputs) {
  Tensor n({2, 2, 1}), h({2, 2, 1});
  n(0, 0, 0) = 2;
  n(0, 1, 0) = 1;
  n(1, 0, 0) = 1;
  n(1, 1, 0) = 4;
  h(0, 0, 0) = -2;
  h(0, 1, 0) = -3;
  h(1, 0, 0) = -3;
  h(1, 1, 0) = -4;
  Tensor n4 = n, h4 = h;
  for (int64_t k = 0; k < n.size(); ++k) {
    n4.flatRef(k) *= 4;
    h4.flatRef(k) *= 4;
  }
  const Tensor ec = vec({-1}), ev = vec({1, 2});
  const ScSlabMultipliers m1 = srsiMultipliers(n, h, ec, ev);
  const ScSlabMultipliers m4 = srsiMultipliers(n4, h4, ec, ev);
  EXPECT_EQ(m4.energy, 4.0 * m1.energy);
  EXPECT_EQ(m4.norm, 4.0 * m1.norm);
  EXPECT_EQ(m4.p.data(), m1.p.data());
  EXPECT_EQ(m4.q.data(), m1.q.data());
  for (std::size_t k = 0; k < m1.r.size(); ++k) EXPECT_EQ(m4.r[k], 4.0 * m1.r[k]);
}

// The other five are AFFINE: an RDM-free constant rides in the metric (Sijr's
// hdm1 = 2 delta - dm1), so N = N0 + lambda N1 and no Euler identity holds --
// the energy at lambda = 2 is NOT twice the energy at lambda = 1. What does
// hold, exactly, is the NORM INCREMENT: a class norm is affine in its inputs,
// so displacing them by X moves it by exactly sum X (docs/testing.md, "The two
// exact checks"). Sijr is the symmetrized affine class.
TEST(SlabMultiplierTests, AnAffineClassHasTheNormIncrementButNoEuler) {
  const Tensor ec = vec({-1, -2}), ev = vec({5, 3});
  // N0: the RDM-free part, here 2 on every raw tuple; N1: the RDM part.
  const auto normAt = [](double lambda) {
    Tensor t({1, 2, 2});
    const double n1[4] = {1, 0.5, 0.5, 2};
    for (int64_t k = 0; k < 4; ++k) t.flatRef(k) = 2.0 + lambda * n1[k];
    return t;
  };
  const auto hAt = [](double lambda) {
    Tensor t({1, 2, 2});
    const double h1[4] = {-1, -1, -1, -2};
    for (int64_t k = 0; k < 4; ++k) t.flatRef(k) = -4.0 + lambda * h1[k];
    return t;
  };
  const ScSlabMultipliers m1 = sijrMultipliers(normAt(1), hAt(1), 1, ec, ev);
  const ScSlabMultipliers m2 = sijrMultipliers(normAt(2), hAt(2), 1, ec, ev);
  // The norm increment: sum of the raw displacement (lambda N1, with
  // sum N1 = 4), exactly.
  EXPECT_EQ(m2.norm - m1.norm, 4.0);
  EXPECT_EQ(m1.norm, 4 * 2.0 + 4.0);
  // No Euler: the energy is not homogeneous in lambda.
  EXPECT_NE(m2.energy, 2.0 * m1.energy);
  // ...while the SAME inputs with the constant dropped are homogeneous.
  const auto scaled = [](const Tensor& t, double c, double lambda) {
    Tensor out = t;
    for (int64_t k = 0; k < out.size(); ++k) out.flatRef(k) = lambda * (t.flat(k) - c);
    return out;
  };
  const ScSlabMultipliers l1 =
      sijrMultipliers(scaled(normAt(1), 2.0, 1), scaled(hAt(1), -4.0, 1), 1, ec, ev);
  const ScSlabMultipliers l2 =
      sijrMultipliers(scaled(normAt(1), 2.0, 2), scaled(hAt(1), -4.0, 2), 1, ec, ev);
  EXPECT_EQ(l2.energy, 2.0 * l1.energy);
}

// ---------------------------------------------------------------------------
// CiSourceHostTests (the shape and accumulation half; #63 pins the fold)
// ---------------------------------------------------------------------------

Tensor sixIndex(int64_t n, double seed) {
  Tensor t({n, n, n, n, n, n});
  for (int64_t k = 0; k < t.size(); ++k) t.flatRef(k) = seed + static_cast<double>(k);
  return t;
}

TEST(CiSourceHostTests, AddIntoIsPureAccumulation) {
  const Tensor x = sixIndex(2, 1.0);
  Tensor y(x.dims());
  ciSourceAddInto(x, y);
  EXPECT_EQ(y.data(), x.data());
  ciSourceAddInto(x, y);
  for (int64_t k = 0; k < x.size(); ++k) EXPECT_EQ(y.flat(k), 2.0 * x.flat(k));
}

TEST(CiSourceHostTests, SymmetrizedIsInvariantUnderReversal) {
  const Tensor d = sixIndex(2, 0.5);
  const Tensor s = ciSourceSymmetrized(d);
  ASSERT_EQ(s.dims(), d.dims());
  for (int64_t p = 0; p < 2; ++p)
    for (int64_t q = 0; q < 2; ++q)
      for (int64_t r = 0; r < 2; ++r)
        for (int64_t ss = 0; ss < 2; ++ss)
          for (int64_t t = 0; t < 2; ++t)
            for (int64_t u = 0; u < 2; ++u) {
              EXPECT_EQ(s(p, q, r, ss, t, u), s(u, t, ss, r, q, p));
              EXPECT_EQ(s(p, q, r, ss, t, u), d(p, q, r, ss, t, u) + d(u, t, ss, r, q, p));
            }
}

TEST(CiSourceHostTests, FoldWithNoElectronsIsTheIdentity) {
  const Tensor g3 = sixIndex(2, 3.0);
  const Tensor g1 = Tensor::fromFlat({2, 2}, {1, 2, 3, 4});
  Tensor g2({2, 2, 2, 2});
  EXPECT_EQ(ciSourceFolded(g1, g2, g3, 0).data(), g3.data());
}

// One orbital: d3 = g3 + g2 / N + g1 / N^2 on the single element.
TEST(CiSourceHostTests, FoldOnOneOrbital) {
  const Tensor g1 = Tensor::fromFlat({1, 1}, {8});
  const Tensor g2 = Tensor::fromFlat({1, 1, 1, 1}, {4});
  const Tensor g3 = Tensor::fromFlat({1, 1, 1, 1, 1, 1}, {1});
  EXPECT_EQ(ciSourceFolded(g1, g2, g3, 2)(0, 0, 0, 0, 0, 0), 1.0 + 4.0 / 2 + 8.0 / 4);
}

// ---------------------------------------------------------------------------
// FourfoldRunTests -- the step-ladder rule every finite-difference check of
// the gradient path gates on (and reference_data/nevpt2_grad.py's
// fourfold_runs by the same definition). Dyadic inputs, so every ratio is
// exact.
// ---------------------------------------------------------------------------

TEST(FourfoldRunTests, AnExactSecondOrderLadderIsOneRun) {
  const std::vector<double> err{1.0, 0.25, 0.0625, 0.015625};
  const FourfoldRun r = longestFourfoldRun(err);
  EXPECT_EQ(r.length, 3);
  EXPECT_EQ(r.end, 3u);
}

TEST(FourfoldRunTests, NoLadderHasNoRun) {
  EXPECT_EQ(longestFourfoldRun(std::vector<double>{}).length, 0);
  EXPECT_EQ(longestFourfoldRun(std::vector<double>{1.0}).length, 0);
}

TEST(FourfoldRunTests, FirstOrderAndFourthOrderAreNotFourfold) {
  // 2 per halving is a first-order error -- #41's plateau, or worse, a
  // constant -- and 16 is fourth order; neither is the truncation regime the
  // window recognises.
  EXPECT_EQ(longestFourfoldRun(std::vector<double>{8.0, 4.0, 2.0, 1.0}).length, 0);
  EXPECT_EQ(longestFourfoldRun(std::vector<double>{1.0, 1.0, 1.0}).length, 0);
  EXPECT_EQ(longestFourfoldRun(std::vector<double>{256.0, 16.0, 1.0}).length, 0);
}

TEST(FourfoldRunTests, AWildRowBreaksTheRunAndTheLongestRunWins) {
  // A run of two, a wild row (#32's cancelling denominator), then a run of
  // three: the longest is reported, ending on its finest rung.
  const std::vector<double> err{64.0, 16.0, 4.0, 512.0, 128.0, 32.0, 8.0, 8.0};
  const FourfoldRun r = longestFourfoldRun(err);
  EXPECT_EQ(r.length, 3);
  EXPECT_EQ(r.end, 6u);
}

TEST(FourfoldRunTests, TiesGoToTheCoarserRun) {
  const std::vector<double> err{16.0, 4.0, 1.0, 100.0, 25.0, 6.25};
  const FourfoldRun r = longestFourfoldRun(err);
  EXPECT_EQ(r.length, 2);
  EXPECT_EQ(r.end, 2u);
}

TEST(FourfoldRunTests, AZeroErrorHasNoRatioAndBreaksTheRun) {
  const std::vector<double> err{16.0, 4.0, 0.0, 0.0};
  EXPECT_EQ(longestFourfoldRun(err).length, 1);
}

TEST(FourfoldRunTests, TheWindowIsOpenAndAdjustable) {
  // The default window is (3.2, 4.8), open at both ends: a ratio of exactly
  // 5 is out, 4.5 is in; the PC tangent's narrower 4 +- 0.5 excludes 4.5 too.
  const std::vector<double> five{5.0, 1.0};
  const std::vector<double> fourHalf{4.5, 1.0};
  EXPECT_EQ(longestFourfoldRun(five).length, 0);
  EXPECT_EQ(longestFourfoldRun(fourHalf).length, 1);
  EXPECT_EQ(longestFourfoldRun(fourHalf, 3.5, 4.5).length, 0);
  EXPECT_DOUBLE_EQ(kFourfoldLow, 3.2);
  EXPECT_DOUBLE_EQ(kFourfoldHigh, 4.8);
}

// ---------------------------------------------------------------------------
// GradientPlanDeathTest -- the abort tier (our bug, not the caller's input)
// ---------------------------------------------------------------------------

TEST(GradientPlanDeathTest, PromotionPlanNeedsATwoElectronSlot) {
  EXPECT_DEATH(promotionPlan(RotBlock::H1e, 0), "is not a two-electron slot");
  EXPECT_DEATH(promotionPlan(RotBlock::V2tr, 0), "is not a two-electron slot");
  EXPECT_DEATH(promotionPlan(RotBlock::Sr, 4), "is not a two-electron slot");
}

TEST(GradientPlanDeathTest, CiSourceAddIntoNeedsMatchingShapes) {
  Tensor y({2});
  EXPECT_DEATH(ciSourceAddInto(Tensor({3}), y), "shapes differ");
}

}  // namespace nevpt2::test::gradient_plan
