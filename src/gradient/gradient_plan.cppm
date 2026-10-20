// nevpt2.gradient_plan -- the gradient path's PURE HOST arithmetic: which
// integral block a promotion reads and over what span, the per-class Delta
// vectors and slab multipliers, the term classification, and the CI source
// term's host-side folding. No device, no kernel, no golden -- so the unit
// tier pins it card-free (test/gradient_plan/), the way nevpt2.rdm_plan pins
// the RDM build's index arithmetic. nevpt2.gradient re-exports it, and its
// internal partitions are the only callers:
//
//   :general     rotBlockInfo / activeAxesOf / rotBlockName / spanOf, and
//                generalPick, which is generalPickPlan + the array lookup;
//                promotionPlan / promotionSubscripts are the four-einsum
//                promotion rule's index bookkeeping, and heffDressingPlan
//                the one-electron blocks' core-dressing terms;
//   :terms       kScActive / termHasActive, ScSlabMultipliers, sliceHost,
//                plainMultipliers;
//   :sc_classes  the eight classes' Delta constructions and the two
//                symmetrized classes' multipliers (sijrMultipliers,
//                srsiMultipliers);
//   :ci_source   ciSourceFolded / ciSourceSymmetrized / ciSourceAddInto;
//   gradient.cpp orbitalEnergyResponse / orbitalEnergyDm1Coefficient and,
//                through :terms and :sc_classes, deltaAdjointInto -- the
//                orbital energies' CI response (#87), whose forward half
//                (orbitalEnergyDot, orbitalEnergiesFrom) the binaries call.
//
// And one rule shared beyond nevpt2.gradient: longestFourfoldRun, the step
// ladder every finite-difference check of the gradient path is gated on --
// the SC tangent, the two orbital source-term checks and the PC tangent
// (nevpt2.pc_tangent links this module for it).
//
// Everything here is moved, not rewritten: the same loops in the same order,
// so the device path's numbers are bit-for-bit what they were.
export module nevpt2.gradient_plan;

import std;
// Re-exported: Tensor (the host arrays in and out), ScMultipliers /
// scMultipliers (what the slab multipliers finish with) and check().
export import nevpt2.tensor;
export import nevpt2.sc_amplitudes;
export import nevpt2.error_handling;

export namespace nevpt2 {

// --- the promotion table (docs/gradient-theory.md, Sec. 3) -------------------

// The three orbital spaces of the NEVPT2 partition, in the order the
// sidecar's general index runs: core, then active, then virtual.
enum class RotSpace { Core, Act, Virt };

// Every array a class hands to assembleSlab as one of its `blocks`, plus the
// two active-space integral blocks every Dyall block is built from. The
// rotation's whole per-block knowledge is keyed by this.
enum class RotBlock {
  Sr,       // h2e_v_Sr      (v, a, a, a)
  Si,       // h2e_v_Si      (a, a, c, a)
  Sijrs,    // cvcv          (c, v, c, v) -- chemists' order, roles on other axes
  Sijr,     // h2e_v_Sijr    (v, a, c, c)
  Srsi,     // h2e_v_Srsi    (v, v, c, a)
  SrsiT,    // the same array, batched along its second axis
  Srs,      // h2e_v_Srs     (v, v, a, a)
  Sij,      // h2e_v_Sij     (a, a, c, c)
  Sir1,     // h2e_v1_Sir    (v, a, c, a)
  Sir2,     // h2e_v2_Sir    (v, a, a, c)
  H1eSr,    // h1e_v_Sr      (v, c) -- h^eff[v,a] - sum_b (rb|bt)
  H1eSi,    // h1e_v_Si      (a, c)
  H1eSir,   // h1e_v_Sir     (v, c)
  V2tr,     // the hoisted trace sum_a h2e_v2_Sir[r,a,a,i] -- DERIVED, see below
  H1e,      // h1e           (a, a) = h^eff[a,a]
  H2e,      // h2e           (a, a, a, a)
};

// Every RotBlock, in declaration order -- what a caller (or a test) walks to
// cover the whole table.
inline constexpr std::array<RotBlock, 16> kAllRotBlocks{
    RotBlock::Sr,   RotBlock::Si,    RotBlock::Sijrs,  RotBlock::Sijr,
    RotBlock::Srsi, RotBlock::SrsiT, RotBlock::Srs,    RotBlock::Sij,
    RotBlock::Sir1, RotBlock::Sir2,  RotBlock::H1eSr,  RotBlock::H1eSi,
    RotBlock::H1eSir, RotBlock::V2tr, RotBlock::H1e,   RotBlock::H2e};

// One slot of a two-electron block: which space its index runs over, and
// which storage axis holds it.
struct RotRole {
  RotSpace space = RotSpace::Act;
  int axis = 0;
};

// A block's rotation recipe. `rank` is 4 for a two-electron block, 2 for a
// one-electron (h^eff) one and 0 for V2tr, which is DERIVED from Sir2's own
// tangent (sum_a Xdot[r,a,a,i]) rather than promoted: an einsum cannot write
// the repeated label its promotion would need, and the trace of the
// intra-active promotions against an antisymmetric kappa cancels exactly, so
// what is left is exactly the trace of the general part (:terms' slabDots
// takes it there).
struct RotBlockInfo {
  int rank = 4;
  std::array<RotRole, 4> roles{};  // p, q, r, s; only p, q when rank == 2
  bool trace = false;              // h1e_v_Sr's -sum_b (rb|bt)
};

// The table. Every two-electron row is read straight off nevpt2.energy's
// layout comment and generate_golden.py's `_integral_blocks` slice: the roles
// are assigned so that X[p,q,r,s] == (p r | q s), which holds for every block
// on its storage axes in order -- and for `cvcv`, stored as the chemists'
// (i r | j s), with q and r swapped onto axes 2 and 1.
RotBlockInfo rotBlockInfo(RotBlock b) {
  using S = RotSpace;
  const auto four = [](S p, S q, S r, S s) {
    return RotBlockInfo{4, {RotRole{p, 0}, RotRole{q, 1}, RotRole{r, 2}, RotRole{s, 3}}, false};
  };
  const auto one = [](S p, S q, bool trace) {
    return RotBlockInfo{2, {RotRole{p, 0}, RotRole{q, 1}, RotRole{}, RotRole{}}, trace};
  };
  switch (b) {
    case RotBlock::Sr: return four(S::Virt, S::Act, S::Act, S::Act);
    case RotBlock::Si: return four(S::Act, S::Act, S::Core, S::Act);
    // (i r | j s): p and r are the first distribution (axes 0 and 1), q and s
    // the second (axes 2 and 3).
    case RotBlock::Sijrs:
      return RotBlockInfo{4,
                          {RotRole{S::Core, 0}, RotRole{S::Core, 2}, RotRole{S::Virt, 1},
                           RotRole{S::Virt, 3}},
                          false};
    case RotBlock::Sijr: return four(S::Virt, S::Act, S::Core, S::Core);
    case RotBlock::Srsi:
    case RotBlock::SrsiT: return four(S::Virt, S::Virt, S::Core, S::Act);
    case RotBlock::Srs: return four(S::Virt, S::Virt, S::Act, S::Act);
    case RotBlock::Sij: return four(S::Act, S::Act, S::Core, S::Core);
    case RotBlock::Sir1: return four(S::Virt, S::Act, S::Core, S::Act);
    case RotBlock::Sir2: return four(S::Virt, S::Act, S::Act, S::Core);
    case RotBlock::H1eSr: return one(S::Virt, S::Act, /*trace=*/true);
    case RotBlock::H1eSi: return one(S::Act, S::Core, false);
    case RotBlock::H1eSir: return one(S::Virt, S::Core, false);
    case RotBlock::V2tr: return RotBlockInfo{0, {}, false};
    case RotBlock::H1e: return one(S::Act, S::Act, false);
    case RotBlock::H2e: return four(S::Act, S::Act, S::Act, S::Act);
  }
  return RotBlockInfo{};
}

// The axes of `b` that carry an ACTIVE index, ascending -- the ones #36's
// intra-active action (:rotation's kappaActionDevice) covers, and the ones a
// purely active rotation is the whole story for. Read off the same table, so
// the two halves of the action cannot disagree about which axis is which.
std::vector<int> activeAxesOf(RotBlock b) {
  const RotBlockInfo info = rotBlockInfo(b);
  std::vector<int> axes;
  for (int k = 0; k < info.rank; ++k)
    if (info.roles[static_cast<std::size_t>(k)].space == RotSpace::Act)
      axes.push_back(info.roles[static_cast<std::size_t>(k)].axis);
  std::sort(axes.begin(), axes.end());
  return axes;
}

// The ten two-electron rows above are nevpt2.energy's ExtBlock values one for
// one, by name; a class names its blocks by RotBlock directly (the kinds list
// it hands assembleSlab), so no ExtBlock -> RotBlock conversion is needed
// anywhere.
const char* rotBlockName(RotBlock b) {
  switch (b) {
    case RotBlock::Sr: return "h2e_v_Sr";
    case RotBlock::Si: return "h2e_v_Si";
    case RotBlock::Sijrs: return "cvcv";
    case RotBlock::Sijr: return "h2e_v_Sijr";
    case RotBlock::Srsi: return "h2e_v_Srsi";
    case RotBlock::SrsiT: return "h2e_v_Srsi(T)";
    case RotBlock::Srs: return "h2e_v_Srs";
    case RotBlock::Sij: return "h2e_v_Sij";
    case RotBlock::Sir1: return "h2e_v1_Sir";
    case RotBlock::Sir2: return "h2e_v2_Sir";
    case RotBlock::H1eSr: return "h1e_v_Sr";
    case RotBlock::H1eSi: return "h1e_v_Si";
    case RotBlock::H1eSir: return "h1e_v_Sir";
    case RotBlock::V2tr: return "v2tr";
    case RotBlock::H1e: return "h1e";
    case RotBlock::H2e: return "h2e";
  }
  return "?";
}

// `[lo, hi)` of the sidecar's general index for one space, given the three
// space sizes (the order is core, active, virtual).
std::pair<int64_t, int64_t> spanOf(int64_t ncore, int64_t nact, int64_t nvirt, RotSpace s) {
  switch (s) {
    case RotSpace::Core: return {0, ncore};
    case RotSpace::Act: return {ncore, ncore + nact};
    case RotSpace::Virt: return {ncore + nact, ncore + nact + nvirt};
  }
  return {0, 0};
}

// The fourteen one-general-index sidecar arrays a promotion can PICK --
// `g_xP_QR[x, p, q, r] = (x p | q r)` with (Q, R) in one of the five stored
// orderings. The inventory's other three rows (`g_xc_cc_j` / `g_xc_cc_k`,
// the tied (xc|cc), and the tied `g_xa_vv` / `g_xc_vv`) are read only by the
// orbital-energy and h^eff rules on their tied labels, never picked.
enum class GeneralArray {
  XaAa, XvAa, XcAa,
  XaVa, XvVa, XcVa,
  XaCa, XvCa, XcCa,
  XaCv, XvCv, XcCv,
  XaCc, XvCc,
};

// The array's name, as GeneralIntegralsDevice and the sidecar spell it.
const char* generalArrayName(GeneralArray a) {
  switch (a) {
    case GeneralArray::XaAa: return "g_xa_aa";
    case GeneralArray::XvAa: return "g_xv_aa";
    case GeneralArray::XcAa: return "g_xc_aa";
    case GeneralArray::XaVa: return "g_xa_va";
    case GeneralArray::XvVa: return "g_xv_va";
    case GeneralArray::XcVa: return "g_xc_va";
    case GeneralArray::XaCa: return "g_xa_ca";
    case GeneralArray::XvCa: return "g_xv_ca";
    case GeneralArray::XcCa: return "g_xc_ca";
    case GeneralArray::XaCv: return "g_xa_cv";
    case GeneralArray::XvCv: return "g_xv_cv";
    case GeneralArray::XcCv: return "g_xc_cv";
    case GeneralArray::XaCc: return "g_xa_cc";
    case GeneralArray::XvCc: return "g_xv_cc";
  }
  return "?";
}

// One promotion's source: which array, and whether its last two axes hold
// (R, Q) rather than (Q, R) -- the integral's (q r) <-> (r q) symmetry, used
// to normalise the second charge distribution onto one of the five stored
// orderings.
struct GeneralPickPlan {
  GeneralArray array = GeneralArray::XaAa;
  bool swap = false;
};

// `(x p | q r)` for the three spaces asked for. The five stored (Q, R) groups
// are (a,a), (v,a), (c,a), (c,v) and (c,c); (v,v) is never asked for (no
// class puts two virtuals in one charge distribution, which is also why
// Sec. 3.4's `(xv|vv)` is absent), and (c,c) is stored for P in {a, v} only
// (Sec. 3.3: every route to a third free core label runs through a tied
// pair). Those two come back empty; :general's generalPick turns that into
// an abort, because it is OUR bug (the table above, not the caller's input).
std::optional<GeneralPickPlan> generalPickPlan(RotSpace p, RotSpace q, RotSpace r) {
  using S = RotSpace;
  using G = GeneralArray;
  // Normalise the second distribution.
  bool swap = false;
  if ((q == S::Act && r == S::Virt) || (q == S::Act && r == S::Core) ||
      (q == S::Virt && r == S::Core)) {
    std::swap(q, r);
    swap = true;
  }
  const auto byP = [p](G a, G v, G c) { return p == S::Act ? a : (p == S::Virt ? v : c); };
  std::optional<G> g;
  if (q == S::Act && r == S::Act)
    g = byP(G::XaAa, G::XvAa, G::XcAa);
  else if (q == S::Virt && r == S::Act)
    g = byP(G::XaVa, G::XvVa, G::XcVa);
  else if (q == S::Core && r == S::Act)
    g = byP(G::XaCa, G::XvCa, G::XcCa);
  else if (q == S::Core && r == S::Virt)
    g = byP(G::XaCv, G::XvCv, G::XcCv);
  else if (q == S::Core && r == S::Core && p != S::Core)
    g = p == S::Act ? G::XaCc : G::XvCc;
  if (!g) return std::nullopt;
  return GeneralPickPlan{*g, swap};
}

// The promotion rule's bookkeeping for one slot of a two-electron block:
// which role is the promoted slot's partner in its charge distribution, and
// which two roles are the other distribution -- p -> (r; q, s), q -> (s; p,
// r), r -> (p; q, s), s -> (q; p, r) -- plus the array that rule reads and the
// storage axes of the source's last three indices, in the order the source
// holds them.
struct PromotionPlan {
  RotRole promoted;     // the slot being replaced by the general index
  RotRole partner;      // its partner in the same charge distribution
  GeneralPickPlan pick;
  int firstAxis = 0;    // the block axis the source's third index carries
  int lastAxis = 0;     // the block axis the source's fourth index carries
};

// The label each storage axis of a block carries; `z` is the promoted
// (general) index and `m` the one a dressing term contracts over.
inline constexpr std::string_view kAxisLabels = "ijkl";

PromotionPlan promotionPlan(RotBlock b, int slot,
                            std::source_location loc = std::source_location::current()) {
  static constexpr int kPartner[4] = {2, 3, 0, 1};
  static constexpr int kSecond[4][2] = {{1, 3}, {0, 2}, {1, 3}, {0, 2}};
  const RotBlockInfo info = rotBlockInfo(b);
  if (info.rank != 4 || slot < 0 || slot >= 4)
    check(false,
          std::format("promotionPlan: {} slot {} is not a two-electron slot", rotBlockName(b),
                      slot),
          loc);
  const auto role = [&](int k) { return info.roles[static_cast<std::size_t>(k)]; };
  const RotRole prom = role(slot);
  const RotRole partner = role(kPartner[slot]);
  const RotRole second0 = role(kSecond[slot][0]);
  const RotRole second1 = role(kSecond[slot][1]);
  const std::optional<GeneralPickPlan> pick =
      generalPickPlan(partner.space, second0.space, second1.space);
  if (!pick)
    check(false,
          "generalPick: the gradient sidecar carries no such one-general-index array "
          "(docs/gradient-theory.md, Sec. 3.3) -- the promotion table asked for one of the "
          "two arrays no class can reach",
          loc);
  return PromotionPlan{prom, partner, *pick, pick->swap ? second1.axis : second0.axis,
                       pick->swap ? second0.axis : second1.axis};
}

// The einsum that rule is: source labelled (z, partner, first, last), kappa's
// column block labelled (z, promoted), out the block's own `ijkl`.
std::string promotionSubscripts(const PromotionPlan& pl) {
  std::string subs = "z";
  subs += kAxisLabels[static_cast<std::size_t>(pl.partner.axis)];
  subs += kAxisLabels[static_cast<std::size_t>(pl.firstAxis)];
  subs += kAxisLabels[static_cast<std::size_t>(pl.lastAxis)];
  subs += ",z";
  subs += kAxisLabels[static_cast<std::size_t>(pl.promoted.axis)];
  subs += "->";
  subs += kAxisLabels;
  return subs;
}

// --- h^eff's core dressing ----------------------------------------------------

// One term of the core dressing's response on a one-electron (h^eff) block:
// `coeff * einsum(subscripts, source, kappa[:, core])`, the source labelled
// with `z` for its general index and `m` for the core one the term
// contracts, kappa's core columns labelled (z, m), out the block's own `ij`.
struct HeffDressingTerm {
  GeneralPickPlan pick;
  std::string subscripts;
  double coeff = 0.0;
};

// The core dressing h^eff_pq = h_pq + sum_j [ 2 (pq|jj) - (pj|jq) ] has its
// two core labels TIED, so its derivative promotes each in turn:
//
//   4 sum_{x, j in core} kappa_xj (pq|xj)
//     - sum_{x, j in core} kappa_xj [ (px|jq) + (pj|xq) ]
//
// -- three terms, in this order: the Coulomb, whose two promotions collapse
// into one because (pq|xj) is symmetric in (x, j), and then the EXCHANGE
// PAIR, whose two promotions do NOT collapse: (px|jq) and (pj|xq) are
// different integrals. Writing the exchange as twice either one is the
// natural simplification and it is WRONG (#38): it leaves a term that
// survives even a core-internal rotation, where the true dressing response is
// zero. Neither the duality nor the sidecar's restriction identities can see
// that mistake; only a core-internal finite difference did (docs/testing.md,
// "The full orbital source term"), and test/gradient_plan's
// HeffDressingTests pins it card-free. :general's accumGeneralOneElectron
// runs exactly these terms; `p` and `q` are the block's two spaces.
std::array<HeffDressingTerm, 3> heffDressingPlan(
    RotSpace p, RotSpace q, std::source_location loc = std::source_location::current()) {
  const auto pickOf = [&](RotSpace a, RotSpace b, RotSpace c) {
    const std::optional<GeneralPickPlan> pl = generalPickPlan(a, b, c);
    if (!pl)
      check(false,
            "heffDressingPlan: the gradient sidecar carries no such one-general-index array "
            "(docs/gradient-theory.md, Sec. 3.3)",
            loc);
    return *pl;
  };
  const char i = kAxisLabels[0], j = kAxisLabels[1];
  // (pq|xj) = (x j | p q): P = core, the second distribution (S_p, S_q).
  const GeneralPickPlan cou = pickOf(RotSpace::Core, p, q);
  std::string couSubs = "zm";
  couSubs += cou.swap ? j : i;
  couSubs += cou.swap ? i : j;
  couSubs += ",zm->ij";
  // (px|jq) = (x p | j q): P = space(p), and (C, S_q) is stored in that order
  // for every S_q, so this source is never swapped -- its axes are
  // (x, p, j, q).
  const GeneralPickPlan exc1 = pickOf(p, RotSpace::Core, q);
  check(!exc1.swap, "heffDressingPlan: the first exchange source is not swapped", loc);
  // (pj|xq) = (x q | p j): P = space(q), and the second distribution is
  // (S_p, C), which IS swapped whenever p is not core.
  const GeneralPickPlan exc2 = pickOf(q, p, RotSpace::Core);
  std::string excSubs = "zj";
  excSubs += exc2.swap ? 'm' : i;
  excSubs += exc2.swap ? i : 'm';
  excSubs += ",zm->ij";
  return {HeffDressingTerm{cou, std::move(couSubs), 4.0},
          HeffDressingTerm{exc1, "zimj,zm->ij", -1.0},
          HeffDressingTerm{exc2, std::move(excSubs), -1.0}};
}

// --- term classification -----------------------------------------------------

// An ScTerm operand's role: one of the class's external integral blocks (the
// index into the class's block list) or the active-space block the term
// contracts them against.
inline constexpr int kScActive = -1;

// Whether any of a term's operands is the active block -- the split between
// the term half Eqs. 42-43 see and the half they cannot. A template over the
// term type so the classification is testable without a DeviceTensor:
// anything with an `ops` range whose elements carry an `int block` will do
// (:terms' ScTerm is the one real caller).
template <class Term>
bool termHasActive(const Term& term) {
  for (const auto& o : term.ops)
    if (o.block == kScActive) return true;
  return false;
}

// --- the slab multipliers ----------------------------------------------------

// One slab's multipliers, shaped for the pseudodensity assembly. For plain
// perturbers this is scMultipliers applied elementwise (plainMultipliers
// below); Sijr and Srsi first symmetrize and pack a triangle, then scatter
// the triangle's multipliers back over the full tuple shape.
// `p` and `q` are shaped as the tuple vector -- scattered back over it for a
// class whose perturbers are symmetrized pairs. `r` (Park Eq. 38, the
// multiplier of the orbital-energy sum) is NOT: it stays in PERTURBER order,
// the order the class's `delta` vector was built in, because that is the
// vector it is contracted against (`deltaOf` in assembleSlab).
struct ScSlabMultipliers {
  Tensor p;
  Tensor q;
  std::vector<double> r;
  double norm = 0.0;
  double energy = 0.0;
};

// `v[b0:b1)` of a rank-1 host tensor (an orbital-energy vector).
std::vector<double> sliceHost(const Tensor& v, int64_t b0, int64_t b1) {
  return std::vector<double>(v.data().begin() + b0, v.data().begin() + b1);
}

// The tuple-is-perturber case: `delta` in the tuple vector's flat order.
ScSlabMultipliers plainMultipliers(const Tensor& norm, const Tensor& h,
                                   const std::vector<double>& delta) {
  const ScMultipliers m = scMultipliers(norm.data(), h.data(), delta);
  return ScSlabMultipliers{Tensor::fromFlat(norm.dims(), m.p),
                           Tensor::fromFlat(norm.dims(), m.q), m.r, m.norm, m.energy};
}

// --- the eight classes' Delta vectors ----------------------------------------
//
// Each class's Delta_t, as a FUNCTION of the orbital-energy vectors `ec`
// (e_core) and `ev` (e_virt), in exactly the order the class's per-tuple
// vector runs -- so that the same function read at the response `deps` is
// Deltadot (Delta_t is a signed sum of orbital energies, linear and
// homogeneous in them). `[b0, b1)` is the slab's range of the class's
// batched axis.

// Sr: Delta_t = eps_r, over the slab's r.
std::vector<double> srDelta(int64_t b0, int64_t b1, const Tensor& ev) {
  return sliceHost(ev, b0, b1);
}

// Si: Delta_t = -eps_i, over the slab's i.
std::vector<double> siDelta(int64_t b0, int64_t b1, const Tensor& ec) {
  std::vector<double> d = sliceHost(ec, b0, b1);
  for (double& x : d) x = -x;
  return d;
}

// Sijrs: Delta_t = eps_a + eps_b - eps_i - eps_j over (i in slab, a, j, b),
// cvcv's own axis order.
std::vector<double> sijrsDelta(int64_t b0, int64_t b1, int64_t ncore, int64_t nvirt,
                               const Tensor& ec, const Tensor& ev) {
  std::vector<double> d;
  d.reserve(static_cast<std::size_t>((b1 - b0) * nvirt * ncore * nvirt));
  for (int64_t i = b0; i < b1; ++i)
    for (int64_t a = 0; a < nvirt; ++a)
      for (int64_t j = 0; j < ncore; ++j)
        for (int64_t b = 0; b < nvirt; ++b) d.push_back(ev(a) + ev(b) - ec(i) - ec(j));
  return d;
}

// Sijr: Delta_t = eps_r - eps_i - eps_j over the (r, j <= i) triangle, in the
// order sijrMultipliers packs it (`nr` virtuals from `b0`), so that the same
// formula at the orbital energies' response is Deltadot and lines up with the
// triangle's own R multipliers.
std::vector<double> sijrDelta(int64_t nr, int64_t ncore, int64_t b0, const Tensor& ec,
                              const Tensor& ev) {
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(nr * ncore * (ncore + 1) / 2));
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t j = 0; j < ncore; ++j)
      for (int64_t i = j; i < ncore; ++i) out.push_back(ev(b0 + r) - ec(j) - ec(i));
  return out;
}

// Srsi: Delta_t = eps_r + eps_s - eps_i over the (r <= s, i) triangle, in the
// order srsiMultipliers packs it -- sijrDelta's counterpart.
std::vector<double> srsiDelta(int64_t nvirt, int64_t ncore, const Tensor& ec,
                              const Tensor& ev) {
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(nvirt * (nvirt + 1) / 2 * ncore));
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t ss = r; ss < nvirt; ++ss)
      for (int64_t i = 0; i < ncore; ++i) out.push_back(ev(r) + ev(ss) - ec(i));
  return out;
}

// Srs: Delta_t = eps_r + eps_s over (r in slab, s).
std::vector<double> srsDelta(int64_t b0, int64_t b1, int64_t nvirt, const Tensor& ev) {
  std::vector<double> d;
  d.reserve(static_cast<std::size_t>((b1 - b0) * nvirt));
  for (int64_t r = b0; r < b1; ++r)
    for (int64_t ss = 0; ss < nvirt; ++ss) d.push_back(ev(r) + ev(ss));
  return d;
}

// Sij: Delta_t = -(eps_i + eps_j) over (i in slab, j).
std::vector<double> sijDelta(int64_t b0, int64_t b1, int64_t ncore, const Tensor& ec) {
  std::vector<double> d;
  d.reserve(static_cast<std::size_t>((b1 - b0) * ncore));
  for (int64_t i = b0; i < b1; ++i)
    for (int64_t j = 0; j < ncore; ++j) d.push_back(-(ec(i) + ec(j)));
  return d;
}

// Sir: Delta_t = eps_r - eps_i over (i, r in slab) -- the terms' own output
// order, core first.
std::vector<double> sirDelta(int64_t b0, int64_t b1, int64_t ncore, const Tensor& ec,
                             const Tensor& ev) {
  std::vector<double> d;
  d.reserve(static_cast<std::size_t>(ncore * (b1 - b0)));
  for (int64_t i = 0; i < ncore; ++i)
    for (int64_t r = b0; r < b1; ++r) d.push_back(ev(r) - ec(i));
  return d;
}

// --- PC's Delta vectors, in the PC classes' own tuple order (#74) -------------
//
// PC walks three classes in a different tuple order from SC's term lists --
// over the RAW (unsymmetrized) tuples, with the external index of the slab
// first -- so the PC adjoint's R_t lines up with these. Sr, Si, Srs and Sij
// share SC's order and use srDelta / siDelta / srsDelta / sijDelta above.

// PC Sijr: Delta_t = eps_r - eps_j - eps_i over (r in slab, j, i), every
// ORDERED core pair (energy_pc_classes.cppm's pc_Sijr).
std::vector<double> pcSijrDelta(int64_t b0, int64_t b1, int64_t ncore, const Tensor& ec,
                                const Tensor& ev) {
  std::vector<double> d;
  d.reserve(static_cast<std::size_t>((b1 - b0) * ncore * ncore));
  for (int64_t r = b0; r < b1; ++r)
    for (int64_t j = 0; j < ncore; ++j)
      for (int64_t i = 0; i < ncore; ++i) d.push_back(ev(r) - ec(j) - ec(i));
  return d;
}

// PC Srsi: Delta_t = eps_r + eps_s - eps_i over (r in slab, s, i), every
// ORDERED virtual pair (pc_Srsi).
std::vector<double> pcSrsiDelta(int64_t b0, int64_t b1, int64_t nvirt, int64_t ncore,
                                const Tensor& ec, const Tensor& ev) {
  std::vector<double> d;
  d.reserve(static_cast<std::size_t>((b1 - b0) * nvirt * ncore));
  for (int64_t r = b0; r < b1; ++r)
    for (int64_t ss = 0; ss < nvirt; ++ss)
      for (int64_t i = 0; i < ncore; ++i) d.push_back(ev(r) + ev(ss) - ec(i));
  return d;
}

// PC Sir: Delta_t = eps_r - eps_i over (r in slab, i) -- the slab's own order,
// virtual first (pc_Sir), where SC's sirDelta runs core first.
std::vector<double> pcSirDelta(int64_t b0, int64_t b1, int64_t ncore, const Tensor& ec,
                               const Tensor& ev) {
  std::vector<double> d;
  d.reserve(static_cast<std::size_t>((b1 - b0) * ncore));
  for (int64_t r = b0; r < b1; ++r)
    for (int64_t i = 0; i < ncore; ++i) d.push_back(ev(r) - ec(i));
  return d;
}

// PC Sijr's exchange partner within one slab of (nr, ncore, ncore) tuples:
// (r, j, i) -> (r, i, j), as flat tuple indices -- an involution, and the map
// pcAdjointPair / pcRotationTangentPair read the exchange row through.
std::vector<int64_t> pcSijrPartner(int64_t nr, int64_t ncore) {
  std::vector<int64_t> p(static_cast<std::size_t>(nr * ncore * ncore));
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t j = 0; j < ncore; ++j)
      for (int64_t i = 0; i < ncore; ++i)
        p[static_cast<std::size_t>((r * ncore + j) * ncore + i)] = (r * ncore + i) * ncore + j;
  return p;
}

// --- the two symmetrized classes' multipliers --------------------------------


// The symmetrization and triangular packing of nevpt2.energy's energy_Sijr,
// verbatim: (r, j, i) and (r, i, j) summed, the diagonal halved, and the
// (r, j <= i) triangle read out in the order sijrDelta builds. Linear in
// `t`, so the same packing of a TANGENT table is the tangent of the packed one
// (sijrMultiplierTangents below).
std::vector<double> sijrTriangle(const Tensor& t) {
  const int64_t nr = t.dim(0), ncore = t.dim(1);
  Tensor sym({nr, ncore, ncore});
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t j = 0; j < ncore; ++j)
      for (int64_t i = 0; i < ncore; ++i) sym(r, j, i) = t(r, j, i) + t(r, i, j);
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t kk = 0; kk < ncore; ++kk) sym(r, kk, kk) *= 0.5;
  std::vector<double> tri;
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t j = 0; j < ncore; ++j)
      for (int64_t i = j; i < ncore; ++i) tri.push_back(sym(r, j, i));
  return tri;
}

// One weight per (r, j <= i) perturber scattered back over the raw
// (nr, ncore, ncore) tuples: (r, j, i) and (r, i, j) both get it.
Tensor sijrScatter(const std::vector<double>& tri, int64_t nr, int64_t ncore) {
  Tensor full({nr, ncore, ncore});
  std::size_t k = 0;
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t j = 0; j < ncore; ++j)
      for (int64_t i = j; i < ncore; ++i, ++k) {
        full(r, j, i) = tri[k];
        full(r, i, j) = tri[k];
      }
  return full;
}

// The symmetrization and triangular packing of nevpt2.energy's energy_Sijr,
// verbatim, plus the scatter back. The perturbers are (r, j <= i); the weight
// that multiplies the RAW normT(r, j, i) is that perturber's, and (r, j, i)
// and (r, i, j) are the same perturber, so the scattered array is symmetric
// in (j, i) -- which is exactly what makes
//   sum_{r,j,i} Pfull(r,j,i) normT(r,j,i) == sum_perturbers P normSym
// hold, the diagonal included (normSym(r,i,i) is normT(r,i,i) after its 1/2).
ScSlabMultipliers sijrMultipliers(const Tensor& normT, const Tensor& hT, int64_t b0,
                                  const Tensor& hostECore, const Tensor& hostEVirt) {
  const int64_t nr = normT.dim(0), ncore = normT.dim(1);
  const ScMultipliers m = scMultipliers(sijrTriangle(normT), sijrTriangle(hT),
                                        sijrDelta(nr, ncore, b0, hostECore, hostEVirt));
  // `r` stays in the triangle's own order: it multiplies Deltadot, which
  // sijrDelta builds there too.
  return ScSlabMultipliers{sijrScatter(m.p, nr, ncore), sijrScatter(m.q, nr, ncore), m.r,
                           m.norm, m.energy};
}

// energy_Srsi's symmetrization and packing, verbatim: (r, s, i) and (s, r, i)
// summed, the diagonal halved, the (r <= s, i) triangle read out in the order
// srsiDelta builds. Linear in `t`, as sijrTriangle is.
std::vector<double> srsiTriangle(const Tensor& t) {
  const int64_t nvirt = t.dim(0), ncore = t.dim(2);
  Tensor sym({nvirt, nvirt, ncore});
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t ss = 0; ss < nvirt; ++ss)
      for (int64_t i = 0; i < ncore; ++i) sym(r, ss, i) = t(r, ss, i) + t(ss, r, i);
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t i = 0; i < ncore; ++i) sym(r, r, i) *= 0.5;
  std::vector<double> tri;
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t ss = r; ss < nvirt; ++ss)
      for (int64_t i = 0; i < ncore; ++i) tri.push_back(sym(r, ss, i));
  return tri;
}

// One weight per (r <= s, i) perturber scattered back over (nvirt, nvirt,
// ncore), symmetric in (r, s).
Tensor srsiScatter(const std::vector<double>& tri, int64_t nvirt, int64_t ncore) {
  Tensor full({nvirt, nvirt, ncore});
  std::size_t k = 0;
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t ss = r; ss < nvirt; ++ss)
      for (int64_t i = 0; i < ncore; ++i, ++k) {
        full(r, ss, i) = tri[k];
        full(ss, r, i) = tri[k];
      }
  return full;
}

// energy_Srsi's symmetrization and packing, verbatim, plus the scatter back --
// as sijrMultipliers, but over the whole (r, s, i) and symmetric in (r, s).
ScSlabMultipliers srsiMultipliers(const Tensor& normT, const Tensor& hT,
                                  const Tensor& hostECore, const Tensor& hostEVirt) {
  const int64_t nvirt = normT.dim(0), ncore = normT.dim(2);
  const ScMultipliers m = scMultipliers(srsiTriangle(normT), srsiTriangle(hT),
                                        srsiDelta(nvirt, ncore, hostECore, hostEVirt));
  return ScSlabMultipliers{srsiScatter(m.p, nvirt, ncore), srsiScatter(m.q, nvirt, ncore),
                           m.r, m.norm, m.energy};
}

// --- the slab multipliers' tangents along a CI direction (#156) ---------------
//
// Reverse-over-forward (docs/hessian-theory.md, Sec. 4.2): differentiating the
// pseudodensity walk along a CI direction u replaces each multiplier by its
// tangent, Pdot = E_NN Ndot + E_NH Hdot and Qdot = E_NH Ndot + E_HH Hdot --
// nevpt2.energy_finish's normToEnergyMultiplierTangent, Sec. 2.4's 3x3 -- at
// FIXED orbital energies (Deltadot = 0: a CI direction moves Delta only
// through the orbital energies' response, which this walk holds fixed, as
// energiesHessianDevice does). `normDot` / `hDot` are the per-tuple tangents
// along u, shaped as `norm` / `h`. A guarded (empty) perturber's tangents are
// zero, as its multipliers are. The two symmetrized classes pack the tangents
// through the SAME triangle as the values (the packing is linear) and scatter
// the triangle's tangents back exactly as their multipliers are.
//
// ALONG AN ORBITAL DIRECTION w (#184, docs/hessian-theory.md Sec. 5.5's R1) the
// same 3x3 applies, and Delta moves too when the orbital energies respond to
// kappa: `diffDot` is the class's Delta construction read at deps(w), in
// PERTURBER order (the order `delta` and R are in), and Rdot -- the third
// output, in that order too -- is what carries dE/de's tangent. The
// five-argument forms are the six-argument ones at diffDot = 0.
struct ScSlabMultiplierTangents {
  Tensor p;               // Pdot, shaped as the tuple vector
  Tensor q;               // Qdot
  std::vector<double> r;  // Rdot, in perturber order (zero when diffDot and Ndot, Hdot are)
};

ScSlabMultiplierTangents plainMultiplierTangents(const Tensor& norm, const Tensor& h,
                                                 const std::vector<double>& delta,
                                                 const Tensor& normDot, const Tensor& hDot,
                                                 const std::vector<double>& diffDot) {
  const ClassMultiplierTangents t = normToEnergyMultiplierTangent(
      norm.data(), h.data(), delta, normDot.data(), hDot.data(), diffDot);
  return {Tensor::fromFlat(norm.dims(), t.p), Tensor::fromFlat(norm.dims(), t.q), t.r};
}

ScSlabMultiplierTangents plainMultiplierTangents(const Tensor& norm, const Tensor& h,
                                                 const std::vector<double>& delta,
                                                 const Tensor& normDot, const Tensor& hDot) {
  return plainMultiplierTangents(norm, h, delta, normDot, hDot,
                                 std::vector<double>(delta.size(), 0.0));
}

ScSlabMultiplierTangents sijrMultiplierTangents(const Tensor& normT, const Tensor& hT,
                                                const Tensor& normTDot, const Tensor& hTDot,
                                                const std::vector<double>& diffDot, int64_t b0,
                                                const Tensor& hostECore,
                                                const Tensor& hostEVirt) {
  const int64_t nr = normT.dim(0), ncore = normT.dim(1);
  const std::vector<double> delta = sijrDelta(nr, ncore, b0, hostECore, hostEVirt);
  const ClassMultiplierTangents t =
      normToEnergyMultiplierTangent(sijrTriangle(normT), sijrTriangle(hT), delta,
                                    sijrTriangle(normTDot), sijrTriangle(hTDot), diffDot);
  return {sijrScatter(t.p, nr, ncore), sijrScatter(t.q, nr, ncore), t.r};
}

ScSlabMultiplierTangents sijrMultiplierTangents(const Tensor& normT, const Tensor& hT,
                                                const Tensor& normTDot, const Tensor& hTDot,
                                                int64_t b0, const Tensor& hostECore,
                                                const Tensor& hostEVirt) {
  const int64_t nr = normT.dim(0), ncore = normT.dim(1);
  return sijrMultiplierTangents(
      normT, hT, normTDot, hTDot,
      std::vector<double>(static_cast<std::size_t>(nr * ncore * (ncore + 1) / 2), 0.0), b0,
      hostECore, hostEVirt);
}

ScSlabMultiplierTangents srsiMultiplierTangents(const Tensor& normT, const Tensor& hT,
                                                const Tensor& normTDot, const Tensor& hTDot,
                                                const std::vector<double>& diffDot,
                                                const Tensor& hostECore,
                                                const Tensor& hostEVirt) {
  const int64_t nvirt = normT.dim(0), ncore = normT.dim(2);
  const std::vector<double> delta = srsiDelta(nvirt, ncore, hostECore, hostEVirt);
  const ClassMultiplierTangents t =
      normToEnergyMultiplierTangent(srsiTriangle(normT), srsiTriangle(hT), delta,
                                    srsiTriangle(normTDot), srsiTriangle(hTDot), diffDot);
  return {srsiScatter(t.p, nvirt, ncore), srsiScatter(t.q, nvirt, ncore), t.r};
}

ScSlabMultiplierTangents srsiMultiplierTangents(const Tensor& normT, const Tensor& hT,
                                                const Tensor& normTDot, const Tensor& hTDot,
                                                const Tensor& hostECore,
                                                const Tensor& hostEVirt) {
  const int64_t nvirt = normT.dim(0), ncore = normT.dim(2);
  return srsiMultiplierTangents(
      normT, hT, normTDot, hTDot,
      std::vector<double>(static_cast<std::size_t>(nvirt * (nvirt + 1) / 2 * ncore), 0.0),
      hostECore, hostEVirt);
}

// --- the mixed block's per-perturber finish (#184) ---------------------------
//
// docs/hessian-theory.md, Sec. 5.5: along an orbital direction w and a CI
// direction u, per perturber k,
//
//   Eddot_kc,k(w, u) = (Ndot, Hdot, Ddot)_k[w] . Hess f_k . (Ndot, Hdot, Ddot)_k[u]   C1
//                    + P_k N_k^{wu} + Q_k H_k^{wu,ext}                              C2
//                    + Q_k H_k^{wu,int}                                             C3
//                    + Q_k H_k^{wu,dig}                                             C4
//                    + R_k Delta_k(eps^{wu})                                        C5
//
// -- the five terms, each reported on its own: an exact zero is a property of
// ONE of them (C3, C4 and C1's digest share along an intra-space w), and
// Euler and the duality read their sum. C1's digest share is the same
// sandwich with only the digests' part of Hdot[w] in the orbital triple: the
// sandwich is linear in each triple, so it is exactly that part of C1.
struct ScMixedTerms {
  double c1 = 0.0;        // the finish's curvature (the 3x3 sandwich)
  double c1Digest = 0.0;  // the part of c1 the digests' share of Hdot[w] carries
  double c2 = 0.0;        // one external slot along w, the block along u
  double c3 = 0.0;        // the Dyall blocks at (h1e'[w], h2e'[w]) and the tangent RDMs
  double c4 = 0.0;        // the Dyall blocks at the digests' mixed tangent f3^{wu}
  double c5 = 0.0;        // the orbital energies' mixed response, R . Delta(eps^{wu})
  [[nodiscard]] double total() const { return c1 + c2 + c3 + c4 + c5; }
  ScMixedTerms& operator+=(const ScMixedTerms& o) {
    c1 += o.c1;
    c1Digest += o.c1Digest;
    c2 += o.c2;
    c3 += o.c3;
    c4 += o.c4;
    c5 += o.c5;
    return *this;
  }
};

// One slab's (or, for Srsi, one class's) per-tuple tables, every one shaped
// as the class's raw tuple vector: the value pair, the two first-order pairs
// and the four mixed pieces of the H side's and the norm's second derivative.
struct ScMixedTables {
  Tensor norm, h;          // N, H
  Tensor normDotW, hDotW;  // along w -- the digests' share merged into hDotW
  Tensor hDotWDigest;      // the digests' share of hDotW alone
  Tensor normDotU, hDotU;  // along u
  Tensor normWU, hWU;      // C2: one external slot along w, the block along u
  Tensor hWUInt;           // C3
  Tensor hWUDigest;        // C4
};

// The five terms over perturbers already in the class's own packed order:
// every vector one per perturber, `delta` the class's Delta and the three
// `diff*` its Delta construction at deps(w), e'(u) and eps^{wu}. A guarded
// (empty) perturber contributes zero, through scMultipliers and the 3x3 alike.
ScMixedTerms mixedTermsPacked(const std::vector<double>& norm, const std::vector<double>& h,
                              const std::vector<double>& delta,
                              const std::vector<double>& normDotW,
                              const std::vector<double>& hDotW,
                              const std::vector<double>& hDotWDigest,
                              const std::vector<double>& diffDotW,
                              const std::vector<double>& normDotU,
                              const std::vector<double>& hDotU,
                              const std::vector<double>& diffDotU,
                              const std::vector<double>& normWU, const std::vector<double>& hWU,
                              const std::vector<double>& hWUInt,
                              const std::vector<double>& hWUDigest,
                              const std::vector<double>& diffWU) {
  const std::size_t k = norm.size();
  check(h.size() == k && delta.size() == k && normDotW.size() == k && hDotW.size() == k &&
            hDotWDigest.size() == k && diffDotW.size() == k && normDotU.size() == k &&
            hDotU.size() == k && diffDotU.size() == k && normWU.size() == k &&
            hWU.size() == k && hWUInt.size() == k && hWUDigest.size() == k &&
            diffWU.size() == k,
        "mixedTermsPacked: the per-perturber vectors' lengths differ");
  const std::vector<double> zero(k, 0.0);
  const ScMultipliers m = scMultipliers(norm, h, delta);
  ScMixedTerms t;
  t.c1 = normToEnergyHessianSandwich(norm, h, delta, normDotW, hDotW, diffDotW, normDotU, hDotU,
                                     diffDotU);
  t.c1Digest = normToEnergyHessianSandwich(norm, h, delta, zero, hDotWDigest, zero, normDotU,
                                           hDotU, diffDotU);
  for (std::size_t i = 0; i < k; ++i) {
    t.c2 += m.p[i] * normWU[i] + m.q[i] * hWU[i];
    t.c3 += m.q[i] * hWUInt[i];
    t.c4 += m.q[i] * hWUDigest[i];
    t.c5 += m.r[i] * diffWU[i];
  }
  return t;
}

// The tuple-is-perturber case: every table in the tuple vector's flat order,
// as `delta` and the three Delta tangents are.
ScMixedTerms plainMixedTerms(const ScMixedTables& t, const std::vector<double>& delta,
                             const std::vector<double>& diffDotW,
                             const std::vector<double>& diffDotU,
                             const std::vector<double>& diffWU) {
  return mixedTermsPacked(t.norm.data(), t.h.data(), delta, t.normDotW.data(), t.hDotW.data(),
                          t.hDotWDigest.data(), diffDotW, t.normDotU.data(), t.hDotU.data(),
                          diffDotU, t.normWU.data(), t.hWU.data(), t.hWUInt.data(),
                          t.hWUDigest.data(), diffWU);
}

// Sijr: every raw (nr, ncore, ncore) table packed through the SAME triangle
// as its multipliers (the packing is linear, so a packed tangent is the
// tangent of the packed value), the Delta tangents already in that order.
ScMixedTerms sijrMixedTerms(const ScMixedTables& t, const std::vector<double>& diffDotW,
                            const std::vector<double>& diffDotU,
                            const std::vector<double>& diffWU, int64_t b0,
                            const Tensor& hostECore, const Tensor& hostEVirt) {
  const int64_t nr = t.norm.dim(0), ncore = t.norm.dim(1);
  return mixedTermsPacked(sijrTriangle(t.norm), sijrTriangle(t.h),
                          sijrDelta(nr, ncore, b0, hostECore, hostEVirt),
                          sijrTriangle(t.normDotW), sijrTriangle(t.hDotW),
                          sijrTriangle(t.hDotWDigest), diffDotW, sijrTriangle(t.normDotU),
                          sijrTriangle(t.hDotU), diffDotU, sijrTriangle(t.normWU),
                          sijrTriangle(t.hWU), sijrTriangle(t.hWUInt),
                          sijrTriangle(t.hWUDigest), diffWU);
}

// Srsi: the same over the whole (nvirt, nvirt, ncore) and srsiTriangle.
ScMixedTerms srsiMixedTerms(const ScMixedTables& t, const std::vector<double>& diffDotW,
                            const std::vector<double>& diffDotU,
                            const std::vector<double>& diffWU, const Tensor& hostECore,
                            const Tensor& hostEVirt) {
  const int64_t nvirt = t.norm.dim(0), ncore = t.norm.dim(2);
  return mixedTermsPacked(srsiTriangle(t.norm), srsiTriangle(t.h),
                          srsiDelta(nvirt, ncore, hostECore, hostEVirt),
                          srsiTriangle(t.normDotW), srsiTriangle(t.hDotW),
                          srsiTriangle(t.hDotWDigest), diffDotW, srsiTriangle(t.normDotU),
                          srsiTriangle(t.hDotU), diffDotU, srsiTriangle(t.normWU),
                          srsiTriangle(t.hWU), srsiTriangle(t.hWUInt),
                          srsiTriangle(t.hWUDigest), diffWU);
}

// --- the orbital block's sandwich, per perturber (#158) ----------------------
//
// docs/hessian-theory.md, Sec. 5.1's second term, I'[u]^T Ehat'' I'[v], along
// two ORBITAL directions u and v, per perturber k:
//
//   K3 = (Ndot, Hdot, Ddot)_k[u] . Hess f_k . (Ndot, Hdot, Ddot)_k[v]   the finish's curvature
//   K4 = P_k N_k^{uv} + Q_k H_k^{uv}       both external slots, one per direction, K undotted
//   K5 = Q_k H_k^{u,K'[v]} + Q_k H_k^{v,K'[u]}   one external slot and the Dyall block
//
// every Hdot carrying the digests' share (merged per tuple), and every Dyall
// block tangent K' its own. There is no Delta^{uv} here: the orbital energies'
// second order is <Ehat', I''>'s (the curvature's K1 / K2), not this term's.
//
// The shape is the mixed block's (Sec. 5.5's C1 / C2 / C3 / C4 with the CI
// direction moved to the integrals), so the finish IS mixedTermsPacked, read
// through each class's own packing (plainMixedTerms, sijrMixedTerms,
// srsiMixedTerms -- `mixedFn` below), three times over remapped tables:
//
//   W = u, U = v: c1 = K3, c1Digest = K3's share of the digests along u,
//                 c2 = K4, c3 = K5's (x'[u], K'[v]) half, c4 its digests' part;
//   W = v, U = u: the same with the directions exchanged -- K3 and K4 again
//                 (the 3x3 is symmetric, and K4's table is the same table),
//                 K3's share of the digests along v, the (x'[v], K'[u]) half;
//   W = u, U = v, both triples the digests' share alone: c1Digest is the
//                 digest x digest part of K3 -- the energy's own curvature in
//                 the digests, which the energy stage's Hessian computes too
//                 (energiesHessianDevice at f3'[u], f3'[v]).
//
// c5 is the mixed Delta term, zero here (dwu = 0). Every table is the class's
// raw tuple vector, as ScMixedTables' are.
struct ScPairTerms {
  double k3 = 0.0;              // the finish's curvature (the 3x3 sandwich)
  double k3DigestU = 0.0;       // the part of k3 the digests' share of Hdot[u] carries
  double k3DigestV = 0.0;       // ... of Hdot[v]
  double k3DigestDigest = 0.0;  // the part both digests' shares carry (in both of those)
  double k4 = 0.0;              // both external slots, one along u and one along v
  double k5uv = 0.0;            // one external slot along u, the Dyall block along v
  double k5uvDigest = 0.0;      // the part of k5uv the digests' share of K'[v] carries
  double k5vu = 0.0;            // one external slot along v, the Dyall block along u
  double k5vuDigest = 0.0;      // the part of k5vu the digests' share of K'[u] carries
  double k3Swap = 0.0;          // k3 from the exchanged finish: equal by the 3x3's symmetry
  double k4Swap = 0.0;          // k4 likewise: the same table, the same sum
  // K6 (#190; docs/hessian-theory.md, Sec. 5.6): the guarded tuples' share,
  // K4's tables there at the ray-limit multipliers -- on an H class the empty
  // tuples both directions populate (emptyTermPacked below), on Sijrs every
  // guarded tuple at its exact P = -1/Delta (k4 above is then the unguarded
  // tuples' share alone, as for every other class).
  double k6 = 0.0;
  [[nodiscard]] double total() const { return k3 + k4 + k5uv + k5vu + k6; }
  ScPairTerms& operator+=(const ScPairTerms& o) {
    k6 += o.k6;
    k3 += o.k3;
    k3DigestU += o.k3DigestU;
    k3DigestV += o.k3DigestV;
    k3DigestDigest += o.k3DigestDigest;
    k4 += o.k4;
    k5uv += o.k5uv;
    k5uvDigest += o.k5uvDigest;
    k5vu += o.k5vu;
    k5vuDigest += o.k5vuDigest;
    k3Swap += o.k3Swap;
    k4Swap += o.k4Swap;
    return *this;
  }
};

// --- the guarded tuples along the pair: K6 and the rho_t refusal (#190) -------
//
// docs/hessian-theory.md, Sec. 5.6 and Sec. 9.1. The energy stage skips every
// perturber with |N_t| <= NUMERICAL_ZERO, and K3-K5 skip it with it. Three
// kinds of guarded perturber, three answers:
//
//   SMALL               a real perturber under the guard (x_t != 0): the
//                       guard holds its energy at zero on a neighbourhood of
//                       the state, so it contributes zero and is never scored.
//   EMPTY BY THE        x_t = 0 by symmetry. Populated at first order by both
//   INTEGRALS           directions (N^{uu}, N^{vv} above the population floor)
//                       its energy along the plane is -Q_N^2 / (Q_N Delta +
//                       Q_H), a quadratic form exactly when the H tables are
//                       proportional to the N tables, H^{ww} = eta N^{ww}. It
//                       is scored by the scale-free defect
//                         rho_t = max(|eta_u - eta_v|, 2 |k - c eta|) / |Delta + eta|
//                       (c = N^{uv}, k = H^{uv}, both over sqrt(N^{uu} N^{vv});
//                       eta_w = H^{ww}/N^{ww}, eta their mean). At round-off
//                       it adds K6 = -N^{uv}/(Delta + eta), which is K4 at
//                       the ray-limit multipliers Pbar = 2T + Delta T^2,
//                       Qbar = T^2, T = -1/(Delta + eta) -- the form computed,
//                       since it is not 0/0 for orthogonal perturbers. Above
//                       tolerance the (class, pair) has NO Hessian (the
//                       directional second derivatives exist, their
//                       polarisation is not bilinear) and is REFUSED, by the
//                       caller, as err_numerical. Populated by one direction
//                       or neither, it contributes nothing to the mixed term.
//   EMPTY BY THE STATE  x_t != 0, N_t round-off against |x_t|^2: Sec. 5.6
//                       does not derive it, so it is refused (no committed
//                       golden has one).
//
// SIJRS IS EXEMPT (`hZero`): its H is identically zero and its energy,
// -N_t/Delta_t, is linear in N_t with NO guard (energy_sijrs), so every
// guarded tuple -- small ones too -- takes the class's own P = -1/Delta,
// with no score. Its walk finishes with that exact multiplier throughout
// (sijrsMultipliers below), so its K6 is the guarded tuples' share of K4.
//
// The four thresholds, from the measured separations:
//   kEmptyIntegralTol  a perturber's largest integral (for a symmetrized
//                      perturber the sum of its two tuples' -- an upper bound
//                      on the largest within a factor two). Empty ones
//                      measure <= 2.7e-15 on every committed golden, small
//                      ones >= 2.6e-9 (acrolein): docs/testing.md, "The
//                      empty-perturber term, offline" -- Sec. 5.6's census's
//                      1e-12.
//   kPopulatedTol      the population floor, on N^{ww} / |w|^2 (|w| the
//                      direction's Frobenius norm, so the test is
//                      scale-free) on an empty tuple. A direction that does
//                      not populate the tuple leaves its dot at the
//                      integrals' symmetry noise, ~1e-15, and N^{ww} at its
//                      square; a real population is >= 1.0e-10 on the host
//                      (LiF; >= 4.5e-8 on N2). The card's own, N2 CAS(4,4),
//                      its DF twin and CAS(8,8) (docs/testing.md, "The
//                      orbital block's sandwich"), over the H classes (Sijrs
//                      is exempt, so its floor decides nothing): <= 1.9e-32
//                      below, >= 2.5e-7 above. NUMERICAL_ZERO's 1e-14, the
//                      host probe's choice, sits more than six decades from
//                      either side.
//   kRhoTol            rho_t's tolerance. Measured on the card (the same
//                      cases): <= 1.4e-13 on every computed pair, the
//                      proportional `virtual x (2 virtual + core)` included,
//                      and a (class, pair)'s worst >= 2.9e-2 on every refused
//                      one; the host's separation is <= 2.3e-12 / >= 2.1e-2.
//                      1e-7 -- the host probe's RHO_TOL -- sits more than five
//                      decades from either side, as PC_LIFT_TOL sits between
//                      its round-off and its real lifts.
//   kEmptyStateRel     N_t / max|x_t|^2 on a guarded tuple that is not empty
//                      by the integrals: acrolein's small tuples measure
//                      >= 1e-1 (N_t 4.5e-16 against |x_t| 5.8e-8), and a
//                      perturber zero by the state would be round-off, ~1e-16.
inline constexpr double kEmptyIntegralTol = 1e-12;
inline constexpr double kPopulatedTol = NUMERICAL_ZERO;
inline constexpr double kRhoTol = 1e-7;
inline constexpr double kEmptyStateRel = 1e-10;

// One class's guarded perturbers along one orbital pair. Counts accumulate
// over slabs.
struct ScGuardedCensus {
  int64_t guarded = 0;       // |N_t| <= NUMERICAL_ZERO
  int64_t empty = 0;         // ... and empty by the integrals
  int64_t populatedU = 0;    // ... and populated at first order along u
  int64_t populatedV = 0;    // ... along v
  int64_t populated = 0;     // ... along both: K6's perturbers (an H class's scored by rho)
  int64_t rhoAbove = 0;      // ... with rho_t > kRhoTol: the (class, pair) is refused
  int64_t emptyByState = 0;  // guarded, not empty by the integrals, N_t round-off: refused
  double rhoMax = 0.0;       // the worst rho_t over `populated` (0 for Sijrs: not scored)
  // The separation the population floor sits in: the largest N^{ww} / |w|^2
  // on an empty perturber at or under kPopulatedTol, and the smallest above.
  double unpopulatedMax = 0.0;
  double populatedMin = std::numeric_limits<double>::infinity();
  [[nodiscard]] bool refused() const { return rhoAbove > 0 || emptyByState > 0; }
  ScGuardedCensus& operator+=(const ScGuardedCensus& o) {
    guarded += o.guarded;
    empty += o.empty;
    populatedU += o.populatedU;
    populatedV += o.populatedV;
    populated += o.populated;
    rhoAbove += o.rhoAbove;
    emptyByState += o.emptyByState;
    rhoMax = std::max(rhoMax, o.rhoMax);
    unpopulatedMax = std::max(unpopulatedMax, o.unpopulatedMax);
    populatedMin = std::min(populatedMin, o.populatedMin);
    return *this;
  }
};

// The guarded perturbers' term, over perturbers in the class's own packed
// order: `norm`, `xMax` (the largest |integral| each reads), `delta`, and
// K4's cross tables at (u, u), (v, v) and (u, v) for the norm and H sides
// (both external slots dotted, the Dyall block undotted, summed over ordered
// slot pairs -- so N^{ww} is N_t's second derivative along w). `scaleU` /
// `scaleV` are |u|^2, |v|^2. Returns the census, K6, and the ray-limit
// multipliers (zero off K6's perturbers) in the same order -- what the
// reverse pairing differentiates with. `hZero`: Sijrs (see above).
struct ScEmptyPacked {
  ScGuardedCensus census;
  double k6 = 0.0;
  std::vector<double> pBar, qBar;
};

ScEmptyPacked emptyTermPacked(const std::vector<double>& norm, const std::vector<double>& xMax,
                              const std::vector<double>& delta,
                              const std::vector<double>& normUU,
                              const std::vector<double>& normVV,
                              const std::vector<double>& normUV, const std::vector<double>& hUU,
                              const std::vector<double>& hVV, const std::vector<double>& hUV,
                              double scaleU, double scaleV, bool hZero,
                              std::source_location loc = std::source_location::current()) {
  const std::size_t n = norm.size();
  for (const std::vector<double>* x : {&xMax, &delta, &normUU, &normVV, &normUV, &hUU, &hVV, &hUV})
    if (x->size() != n)
      check(false,
            std::format("emptyTermPacked: the per-perturber vectors' lengths differ ({} against "
                        "{})",
                        x->size(), n),
            loc);
  if (!(scaleU > 0.0) || !(scaleV > 0.0))
    check(false, std::format("emptyTermPacked: |u|^2 = {}, |v|^2 = {}", scaleU, scaleV), loc);
  ScEmptyPacked out;
  out.pBar.assign(n, 0.0);
  out.qBar.assign(n, 0.0);
  ScGuardedCensus& c = out.census;
  for (std::size_t k = 0; k < n; ++k) {
    const double nk = std::fabs(norm[k]);
    if (nk > NUMERICAL_ZERO) continue;
    ++c.guarded;
    const double x = xMax[k];
    const bool empty = x <= kEmptyIntegralTol;
    if (empty) {
      ++c.empty;
    } else if (nk <= kEmptyStateRel * x * x) {
      ++c.emptyByState;
    }
    // A small tuple is the guard's zero, except on Sijrs, which has no guard.
    if (!empty && !hZero) continue;
    const double popU = std::fabs(normUU[k]) / scaleU;
    const double popV = std::fabs(normVV[k]) / scaleV;
    const bool bu = popU > kPopulatedTol, bv = popV > kPopulatedTol;
    if (empty)
      for (const double p : {popU, popV}) {
        if (p > kPopulatedTol) c.populatedMin = std::min(c.populatedMin, p);
        else c.unpopulatedMax = std::max(c.unpopulatedMax, p);
      }
    if (empty && bu) ++c.populatedU;
    if (empty && bv) ++c.populatedV;
    if (empty && bu && bv) ++c.populated;
    if (hZero) {
      // Sijrs: the class's own exact multiplier, on every guarded tuple.
      out.pBar[k] = -1.0 / delta[k];
      out.k6 += out.pBar[k] * normUV[k];
      continue;
    }
    if (!(bu && bv)) continue;
    const double s = std::sqrt(std::fabs(normUU[k] * normVV[k]));
    const double cc = normUV[k] / s, kk = hUV[k] / s;
    const double etaU = hUU[k] / normUU[k], etaV = hVV[k] / normVV[k];
    const double eta = 0.5 * (etaU + etaV);
    const double den = delta[k] + eta;
    const double rho =
        std::max(std::fabs(etaU - etaV), 2.0 * std::fabs(kk - cc * eta)) / std::fabs(den);
    c.rhoMax = std::max(c.rhoMax, rho);
    if (!(rho <= kRhoTol)) ++c.rhoAbove;
    const double t = -1.0 / den;
    out.pBar[k] = 2.0 * t + delta[k] * t * t;
    out.qBar[k] = t * t;
    out.k6 += out.pBar[k] * normUV[k] + out.qBar[k] * hUV[k];
  }
  return out;
}

// One slab's (or, for Srsi, one class's) tables for the guarded term, each
// shaped as the class's raw tuple vector, and the two directions' |w|^2.
struct ScEmptyTables {
  Tensor norm, xMax;
  Tensor normUU, normVV, normUV;
  Tensor hUU, hVV, hUV;
  double scaleU = 1.0, scaleV = 1.0;
};

// The guarded term with its multipliers scattered back over the raw tuples
// (as ScSlabMultipliers' are), so sum_raw Pbar N^{uv}_raw is K6's norm half.
struct ScEmptySlab {
  ScGuardedCensus census;
  double k6 = 0.0;
  Tensor pBar, qBar;
  bool any = false;  // a nonzero multiplier anywhere: the reverse pairing has work
};

ScEmptySlab emptySlabOf(ScEmptyPacked&& e, Tensor pBar, Tensor qBar) {
  ScEmptySlab s{e.census, e.k6, std::move(pBar), std::move(qBar), false};
  for (std::size_t k = 0; k < e.pBar.size(); ++k)
    if (e.pBar[k] != 0.0 || e.qBar[k] != 0.0) s.any = true;
  return s;
}

// The tuple-is-perturber case: `delta` in the tuple vector's flat order.
ScEmptySlab plainEmptyTerm(const ScEmptyTables& t, const std::vector<double>& delta,
                           bool hZero = false) {
  ScEmptyPacked e = emptyTermPacked(t.norm.data(), t.xMax.data(), delta, t.normUU.data(),
                                    t.normVV.data(), t.normUV.data(), t.hUU.data(), t.hVV.data(),
                                    t.hUV.data(), t.scaleU, t.scaleV, hZero);
  Tensor p = Tensor::fromFlat(t.norm.dims(), e.pBar);
  Tensor q = Tensor::fromFlat(t.norm.dims(), e.qBar);
  return emptySlabOf(std::move(e), std::move(p), std::move(q));
}

// Sijr: every raw (nr, ncore, ncore) table through the SAME triangle as its
// multipliers -- the packing is linear, so a packed cross table is the packed
// perturber's -- and xMax through it too (a sum of two maxima, so an upper
// bound on the perturber's largest integral within a factor two).
ScEmptySlab sijrEmptyTerm(const ScEmptyTables& t, int64_t b0, const Tensor& hostECore,
                          const Tensor& hostEVirt) {
  const int64_t nr = t.norm.dim(0), ncore = t.norm.dim(1);
  ScEmptyPacked e = emptyTermPacked(
      sijrTriangle(t.norm), sijrTriangle(t.xMax), sijrDelta(nr, ncore, b0, hostECore, hostEVirt),
      sijrTriangle(t.normUU), sijrTriangle(t.normVV), sijrTriangle(t.normUV), sijrTriangle(t.hUU),
      sijrTriangle(t.hVV), sijrTriangle(t.hUV), t.scaleU, t.scaleV, /*hZero=*/false);
  Tensor p = sijrScatter(e.pBar, nr, ncore);
  Tensor q = sijrScatter(e.qBar, nr, ncore);
  return emptySlabOf(std::move(e), std::move(p), std::move(q));
}

// Srsi: the same over the whole (nvirt, nvirt, ncore) and srsiTriangle.
ScEmptySlab srsiEmptyTerm(const ScEmptyTables& t, const Tensor& hostECore,
                          const Tensor& hostEVirt) {
  const int64_t nvirt = t.norm.dim(0), ncore = t.norm.dim(2);
  ScEmptyPacked e = emptyTermPacked(
      srsiTriangle(t.norm), srsiTriangle(t.xMax), srsiDelta(nvirt, ncore, hostECore, hostEVirt),
      srsiTriangle(t.normUU), srsiTriangle(t.normVV), srsiTriangle(t.normUV), srsiTriangle(t.hUU),
      srsiTriangle(t.hVV), srsiTriangle(t.hUV), t.scaleU, t.scaleV, /*hZero=*/false);
  Tensor p = srsiScatter(e.pBar, nvirt, ncore);
  Tensor q = srsiScatter(e.qBar, nvirt, ncore);
  return emptySlabOf(std::move(e), std::move(p), std::move(q));
}

// --- Sijrs's own finish, exact (#190) ------------------------------------------
//
// H_t is identically zero, so E_t = -N_t / Delta_t: linear in N_t, and
// energy_sijrs applies NO guard. scMultipliers' guard (and the finish
// Hessian's, which at H = 0 carries E_HH = -2/(N Delta^3) and so cannot just
// be unguarded) zeroes the multipliers of a tuple the energy does not skip --
// harmless at first order (a guarded tuple's Ndot is the guard's size), but
// at second order an empty tuple's N^{uv} is not, and that is the whole of the
// core x core gap #158's ladders measured (docs/hessian-theory.md, Sec. 5.6).
// So Sijrs finishes with its own closed forms on every tuple:
//
//   T = P = -1/Delta,  Q = 1/Delta^2,  R = N/Delta^2
//   Pdot = Deltadot/Delta^2,  Qdot = -2 Deltadot/Delta^3,
//   Rdot = Ndot/Delta^2 - 2 N Deltadot/Delta^3
//
// -- scMultipliers' and normToEnergyMultiplierTangent's own values wherever
// those do not guard (f_NN = 0, f_NDelta = 1/Delta^2, f_DeltaDelta =
// -2N/Delta^3; Hdot is zero, so no H entry is ever read).
ScMultipliers sijrsMultipliers(const std::vector<double>& norm, const std::vector<double>& delta,
                               std::source_location loc = std::source_location::current()) {
  const std::size_t n = norm.size();
  if (delta.size() != n)
    check(false, std::format("sijrsMultipliers: norm/delta lengths differ ({}, {})", n,
                             delta.size()),
          loc);
  ScMultipliers m;
  m.t.resize(n);
  m.p.resize(n);
  m.q.resize(n);
  m.r.resize(n);
  for (std::size_t k = 0; k < n; ++k) {
    const double t = -1.0 / delta[k];
    m.t[k] = t;
    m.p[k] = t;
    m.q[k] = t * t;
    m.r[k] = norm[k] * t * t;
    m.norm += norm[k];
    m.energy += t * norm[k];
  }
  return m;
}

ScSlabMultiplierTangents sijrsMultiplierTangents(const Tensor& norm,
                                                 const std::vector<double>& delta,
                                                 const Tensor& normDot,
                                                 const std::vector<double>& diffDot,
                                                 std::source_location loc =
                                                     std::source_location::current()) {
  const std::size_t n = static_cast<std::size_t>(norm.size());
  if (delta.size() != n || static_cast<std::size_t>(normDot.size()) != n || diffDot.size() != n)
    check(false, "sijrsMultiplierTangents: the per-tuple vectors' lengths differ", loc);
  std::vector<double> p(n), q(n), r(n);
  for (std::size_t k = 0; k < n; ++k) {
    const double inv = 1.0 / delta[k];
    const double inv2 = inv * inv;
    p[k] = diffDot[k] * inv2;
    q[k] = -2.0 * diffDot[k] * inv2 * inv;
    r[k] = normDot.flat(static_cast<int64_t>(k)) * inv2 -
           2.0 * norm.flat(static_cast<int64_t>(k)) * diffDot[k] * inv2 * inv;
  }
  return {Tensor::fromFlat(norm.dims(), p), Tensor::fromFlat(norm.dims(), q), r};
}

// mixedTermsPacked's five terms for Sijrs, at its exact finish: C1 the 3x3
// between (Ndot, 0, Ddot)[w] and (Ndot, 0, Ddot)[u], C2 = P N^{wu}, C5 =
// R Delta(eps^{wu}); C1's digest share, C3 and C4 have no H to live in.
ScMixedTerms sijrsMixedTerms(const ScMixedTables& t, const std::vector<double>& delta,
                             const std::vector<double>& diffDotW,
                             const std::vector<double>& diffDotU,
                             const std::vector<double>& diffWU,
                             std::source_location loc = std::source_location::current()) {
  const std::size_t n = delta.size();
  if (static_cast<std::size_t>(t.norm.size()) != n || diffDotW.size() != n ||
      diffDotU.size() != n || diffWU.size() != n ||
      static_cast<std::size_t>(t.normDotW.size()) != n ||
      static_cast<std::size_t>(t.normDotU.size()) != n ||
      static_cast<std::size_t>(t.normWU.size()) != n)
    check(false, "sijrsMixedTerms: the per-tuple vectors' lengths differ", loc);
  ScMixedTerms m;
  for (std::size_t k = 0; k < n; ++k) {
    const int64_t q = static_cast<int64_t>(k);
    const double inv = 1.0 / delta[k];
    const double inv2 = inv * inv;
    m.c1 += (t.normDotW.flat(q) * diffDotU[k] + diffDotW[k] * t.normDotU.flat(q)) * inv2 -
            2.0 * t.norm.flat(q) * diffDotW[k] * diffDotU[k] * inv2 * inv;
    m.c2 += -inv * t.normWU.flat(q);
    m.c5 += t.norm.flat(q) * inv2 * diffWU[k];
  }
  return m;
}

// One slab's (or, for Srsi, one class's) tables for the pair, each shaped as
// the class's raw tuple vector.
struct ScPairTables {
  Tensor norm, h;                   // N, H
  Tensor normDotU, hDotU;           // along u -- the digests' share merged into hDotU
  Tensor hDotUDigest;               // the digests' share of hDotU alone
  Tensor normDotV, hDotV;           // along v, likewise
  Tensor hDotVDigest;
  Tensor normUV, hUV;               // K4: two external slots, one per direction
  Tensor hUBlockV, hUBlockVDigest;  // K5: x'[u] against K'[v], and against its digests' part
  Tensor hVBlockU, hVBlockUDigest;  // K5: x'[v] against K'[u], likewise
};

// The pair's terms through the class's own mixed finish `mixedFn(tables, dw,
// du, dwu)` (its plainMixedTerms / sijrMixedTerms / srsiMixedTerms lambda),
// the three remappings above. `diffDotU` / `diffDotV` are the class's Delta
// construction at deps(u) / deps(v), in perturber order.
template <class MixedFn>
ScPairTerms pairTermsVia(MixedFn&& mixedFn, const ScPairTables& t,
                         const std::vector<double>& diffDotU,
                         const std::vector<double>& diffDotV) {
  const std::vector<double> zero(diffDotU.size(), 0.0);
  const Tensor zeroT(t.norm.dims());
  ScMixedTables a;
  a.norm = t.norm;
  a.h = t.h;
  a.normDotW = t.normDotU;
  a.hDotW = t.hDotU;
  a.hDotWDigest = t.hDotUDigest;
  a.normDotU = t.normDotV;
  a.hDotU = t.hDotV;
  a.normWU = t.normUV;
  a.hWU = t.hUV;
  a.hWUInt = t.hUBlockV;
  a.hWUDigest = t.hUBlockVDigest;
  const ScMixedTerms uv = mixedFn(a, diffDotU, diffDotV, zero);

  ScMixedTables b = a;
  b.normDotW = t.normDotV;
  b.hDotW = t.hDotV;
  b.hDotWDigest = t.hDotVDigest;
  b.normDotU = t.normDotU;
  b.hDotU = t.hDotU;
  b.hWUInt = t.hVBlockU;
  b.hWUDigest = t.hVBlockUDigest;
  const ScMixedTerms vu = mixedFn(b, diffDotV, diffDotU, zero);

  // Both triples the digests' share alone: c1Digest reads (0, hDotWDigest, 0)
  // against (normDotU, hDotU, du), so hand it (0, hDotVDigest, 0) there.
  ScMixedTables c = a;
  c.normDotU = zeroT;
  c.hDotU = t.hDotVDigest;
  const ScMixedTerms dd = mixedFn(c, zero, zero, zero);

  ScPairTerms p;
  p.k3 = uv.c1;
  p.k3DigestU = uv.c1Digest;
  p.k4 = uv.c2;
  p.k5uv = uv.c3;
  p.k5uvDigest = uv.c4;
  p.k3Swap = vu.c1;
  p.k3DigestV = vu.c1Digest;
  p.k4Swap = vu.c2;
  p.k5vu = vu.c3;
  p.k5vuDigest = vu.c4;
  p.k3DigestDigest = dd.c1Digest;
  return p;
}


// --- the CI source term's host half (Park Eq. 48) ----------------------------
//
// :ci_source's header has the derivation: the fixed-particle-number trace
// identities let dm1's and dm2's coefficient tensors ride on dm3's diagonals,
// so one 3-body sigma carries all three.

// g3 with the dm2 and dm1 coefficients folded onto its diagonals. `nelec` is
// the active space's electron count; at zero there are no electrons to
// differentiate and dm1/dm2/dm3 are identically zero, so the fold is skipped
// rather than divided by.
Tensor ciSourceFolded(const Tensor& g1, const Tensor& g2, const Tensor& g3,
                      const int64_t nelec) {
  Tensor d3 = g3;
  if (nelec <= 0) return d3;
  const int64_t n = g1.dim(0);
  // `inv1` and `inv2` are named for the RDM each scales: g1 rides on TWO of
  // d3's diagonals (r == s and t == u), so it is divided by N_elec twice; g2
  // on one (t == u), so once.
  const double inv1 = 1.0 / (static_cast<double>(nelec) * static_cast<double>(nelec));
  const double inv2 = 1.0 / static_cast<double>(nelec);
  for (int64_t p = 0; p < n; ++p)
    for (int64_t q = 0; q < n; ++q)
      for (int64_t r = 0; r < n; ++r)
        for (int64_t s = 0; s < n; ++s)
          for (int64_t t = 0; t < n; ++t) {
            d3(p, q, r, s, t, t) += inv2 * g2(p, q, r, s);
            if (r == s) d3(p, q, r, r, t, t) += inv1 * g1(p, q);
          }
  return d3;
}

// d3 + d3 with its operator string reversed: the two halves of the 3-body
// RDM derivative, which the one-sided sigma kernel takes as a single
// coefficient tensor (it is linear in it). test/sigma_vector/'s
// SigmaVector3GammaTests pins this convention against an exact central
// difference.
Tensor ciSourceSymmetrized(const Tensor& d3) {
  const int64_t n = d3.dim(0);
  Tensor out(d3.dims());
  for (int64_t p = 0; p < n; ++p)
    for (int64_t q = 0; q < n; ++q)
      for (int64_t r = 0; r < n; ++r)
        for (int64_t s = 0; s < n; ++s)
          for (int64_t t = 0; t < n; ++t)
            for (int64_t u = 0; u < n; ++u)
              out(p, q, r, s, t, u) = d3(p, q, r, s, t, u) + d3(u, t, s, r, q, p);
  return out;
}

// y += x, elementwise, over two vectors of the same shape.
void ciSourceAddInto(const Tensor& x, Tensor& y) {
  check(x.dims() == y.dims(), "ciSourceAddInto: shapes differ");
  for (int64_t i = 0; i < y.size(); ++i) y.flatRef(i) += x.flat(i);
}

// --- the orbital energies' CI response (#87) ---------------------------------
//
// e_core / e_virt are the generalized Fock's diagonal, and f is built from the
// active 1-RDM:
//
//   e_p = h^eff_pp + sum_ut gamma_ut W_p[u, t],
//   W_p[u, t] = (pp|ut) - (1/2) (pu|tp)                  p core or virtual
//
// -- the active-density line of docs/gradients.md, Sec. 6's deps formula with
// the rotation taken out, and what :general's fockDiagResidual rebuilds the
// sidecar's f diagonal from. So the orbital energies are AFFINE in dm1, and
// the CI source term's missing piece (Sec. 5.5) is one more dm1 coefficient,
//
//   dE/d(gamma_ut) += sum_p (dE/de_p) W_p[u, t],
//
// with dE/de_p = sum_t R_t dDelta_t/de_p (Park Eq. 38's multiplier, scattered
// onto the orbital energies each Delta reads: deltaAdjoint below). Forward
// mode is the transpose: e_p_dot = sum_ut gamma_dot_ut W_p[u, t], and every
// class's Delta read at it is its Deltadot.

// W over every core and every virtual orbital, (nact, nact, ncore) and
// (nact, nact, nvirt), from the four gradient-sidecar arrays the
// orbital-energy rules read on their tied pairs (docs/gradient-theory.md,
// Sec. 3.3; `g_xP_QR[x, p, q, r] = (x p | q r)`, x over the whole MO range):
//
//   core i:     (ii|ut) = g_xa_cc[u, t, i, i]   (iu|ti) = (ui|it) = g_xc_ca[u, i, i, t]
//   virtual r:  (rr|ut) = g_xa_vv[u, t, r]      (ru|tr) = (ur|rt) = g_xv_va[u, r, r, t]
//
// with u, t read at their general index ncore + u. Pure host arithmetic;
// test/gradient_plan/'s OrbitalEnergyResponseTests rebuild the golden's
// e_core / e_virt from it and the sidecar's h^eff, which pins every index.
struct OrbitalEnergyResponse {
  Tensor core;  // (nact, nact, ncore): W_i[u, t]
  Tensor virt;  // (nact, nact, nvirt): W_r[u, t]
};

OrbitalEnergyResponse orbitalEnergyResponse(const int64_t ncore, const int64_t nact,
                                            const int64_t nvirt, const Tensor& gXaCc,
                                            const Tensor& gXcCa, const Tensor& gXaVv,
                                            const Tensor& gXvVa) {
  const int64_t nmo = ncore + nact + nvirt;
  check(gXaCc.dims() == std::vector<int64_t>{nmo, nact, ncore, ncore} &&
            gXcCa.dims() == std::vector<int64_t>{nmo, ncore, ncore, nact} &&
            gXaVv.dims() == std::vector<int64_t>{nmo, nact, nvirt} &&
            gXvVa.dims() == std::vector<int64_t>{nmo, nvirt, nvirt, nact},
        "orbitalEnergyResponse: a sidecar array's shape is not its partition's");
  OrbitalEnergyResponse w{Tensor({nact, nact, ncore}), Tensor({nact, nact, nvirt})};
  for (int64_t u = 0; u < nact; ++u)
    for (int64_t t = 0; t < nact; ++t) {
      for (int64_t i = 0; i < ncore; ++i)
        w.core(u, t, i) = gXaCc(ncore + u, t, i, i) - 0.5 * gXcCa(ncore + u, i, i, t);
      for (int64_t r = 0; r < nvirt; ++r)
        w.virt(u, t, r) = gXaVv(ncore + u, t, r) - 0.5 * gXvVa(ncore + u, r, r, t);
    }
  return w;
}

// The forward half: the orbital energies' tangent along a 1-RDM tangent,
// e_p_dot = sum_ut gammaDot[u, t] W_p[u, t].
OrbitalEnergyDot orbitalEnergyDot(const OrbitalEnergyResponse& w, const Tensor& gammaDot) {
  const int64_t nact = w.core.dim(0), ncore = w.core.dim(2), nvirt = w.virt.dim(2);
  check(gammaDot.dims() == std::vector<int64_t>{nact, nact},
        "orbitalEnergyDot: the 1-RDM tangent is not (nact, nact)");
  OrbitalEnergyDot d{Tensor({ncore}), Tensor({nvirt})};
  for (int64_t u = 0; u < nact; ++u)
    for (int64_t t = 0; t < nact; ++t) {
      const double g = gammaDot(u, t);
      for (int64_t i = 0; i < ncore; ++i) d.core(i) += g * w.core(u, t, i);
      for (int64_t r = 0; r < nvirt; ++r) d.virt(r) += g * w.virt(u, t, r);
    }
  return d;
}

// The orbital energies themselves, rebuilt: e_p = h^eff_pp + sum_ut gamma W_p
// -- the check that a sidecar's W belongs to the golden beside it (its
// e_core / e_virt must come back out), returned in OrbitalEnergyDot's two
// fields because it is the same pair of vectors. `heff` is the sidecar's
// (nmo, nmo) h^eff, `gamma` the active 1-RDM.
OrbitalEnergyDot orbitalEnergiesFrom(const OrbitalEnergyResponse& w, const Tensor& heff,
                                     const Tensor& gamma) {
  const int64_t nact = w.core.dim(0), ncore = w.core.dim(2), nvirt = w.virt.dim(2);
  check(heff.dims() == std::vector<int64_t>{ncore + nact + nvirt, ncore + nact + nvirt},
        "orbitalEnergiesFrom: h^eff is not (nmo, nmo)");
  OrbitalEnergyDot e = orbitalEnergyDot(w, gamma);
  for (int64_t i = 0; i < ncore; ++i) e.core(i) += heff(i, i);
  for (int64_t r = 0; r < nvirt; ++r) e.virt(r) += heff(ncore + nact + r, ncore + nact + r);
  return e;
}

// The reverse half: the dm1 coefficient sum_p (dE/de_p) W_p, (nact, nact),
// which the CI source term adds to dE/d(dm1) before the fold. `dEdCore` /
// `dEdVirt` are dE/de over the core and virtual orbitals (deltaAdjoint,
// summed over the eight classes).
Tensor orbitalEnergyDm1Coefficient(const OrbitalEnergyResponse& w,
                                   const std::vector<double>& dEdCore,
                                   const std::vector<double>& dEdVirt) {
  const int64_t nact = w.core.dim(0), ncore = w.core.dim(2), nvirt = w.virt.dim(2);
  check(std::ssize(dEdCore) == ncore && std::ssize(dEdVirt) == nvirt,
        "orbitalEnergyDm1Coefficient: dE/de is not (ncore) + (nvirt)");
  Tensor g1({nact, nact});
  for (int64_t u = 0; u < nact; ++u)
    for (int64_t t = 0; t < nact; ++t) {
      double s = 0.0;
      for (int64_t i = 0; i < ncore; ++i) s += dEdCore[static_cast<std::size_t>(i)] * w.core(u, t, i);
      for (int64_t r = 0; r < nvirt; ++r) s += dEdVirt[static_cast<std::size_t>(r)] * w.virt(u, t, r);
      g1(u, t) = s;
    }
  return g1;
}

// W with each orbital's (u, t) pair symmetrized, (W_p[u, t] + W_p[t, u]) / 2:
// the part of a response any symmetric 1-RDM tangent reads, and all a dm1
// coefficient can carry (the 1-RDM is symmetric identically in the CI vector,
// so the antisymmetric part of its coefficient has no CI derivative). W itself
// is symmetric already; its tangent along a rotation (#184,
// generalEpsResponseTangent) is not until it passes through here.
OrbitalEnergyResponse symmetrizedResponse(const OrbitalEnergyResponse& w) {
  OrbitalEnergyResponse s{w.core, w.virt};
  const int64_t nact = w.core.dim(0), ncore = w.core.dim(2), nvirt = w.virt.dim(2);
  for (int64_t u = 0; u < nact; ++u)
    for (int64_t t = 0; t < nact; ++t) {
      for (int64_t i = 0; i < ncore; ++i)
        s.core(u, t, i) = 0.5 * (w.core(u, t, i) + w.core(t, u, i));
      for (int64_t r = 0; r < nvirt; ++r)
        s.virt(u, t, r) = 0.5 * (w.virt(u, t, r) + w.virt(t, u, r));
    }
  return s;
}

// dE/de_p from one class's slab: the TRANSPOSE of its Delta construction
// applied to that slab's R multipliers,
//
//   dEdCore[i] += sum_t R_t dDelta_t/de_i,    dEdVirt[r] likewise,
//
// read off by PROBING `deltaOf(ec, ev)` -- the very function the slab's
// multipliers and Deltadot are built from -- with one unit vector per
// orbital. Exact, because Delta_t is a signed sum of orbital energies (linear
// and homogeneous, DeltaTests' EveryClassIsLinearAndHomogeneous...), and no
// per-class transpose is transcribed. Costs (ncore + nvirt) Delta builds per
// slab, which is nothing beside the slab's einsums at every size a sidecar
// exists for.
template <class DeltaFn>
void deltaAdjointInto(DeltaFn&& deltaOf, const std::vector<double>& r, std::vector<double>& dEdCore,
                      std::vector<double>& dEdVirt) {
  const int64_t ncore = std::ssize(dEdCore), nvirt = std::ssize(dEdVirt);
  Tensor ec({ncore}), ev({nvirt});
  const auto pair = [&]() {
    const std::vector<double> d = deltaOf(ec, ev);
    check(d.size() == r.size(), "deltaAdjointInto: Delta and R have different lengths");
    double s = 0.0;
    for (std::size_t k = 0; k < d.size(); ++k) s += r[k] * d[k];
    return s;
  };
  for (int64_t i = 0; i < ncore; ++i) {
    ec(i) = 1.0;
    dEdCore[static_cast<std::size_t>(i)] += pair();
    ec(i) = 0.0;
  }
  for (int64_t v = 0; v < nvirt; ++v) {
    ev(v) = 1.0;
    dEdVirt[static_cast<std::size_t>(v)] += pair();
    ev(v) = 0.0;
  }
}

// --- the step ladder ---------------------------------------------------------
//
// Every finite difference in the gradient path is gated on a RUN of halvings
// that each cut its error fourfold -- never on one step (docs/testing.md,
// "Residual-independent checks of the gradient path"). One step cannot tell a
// converged quotient from one converged to the wrong thing: #41's PC source
// term was wrong only at a step size, by a first-order term no single step
// shows, and #32's lone step went wild where a symmetry-empty perturber's
// Delta N + H cancelled to round-off. A run of fourfold halvings is the
// signature of an O(h^2) truncation error being walked down, and nothing else
// produces one by accident.
//
// The rule lives here so that every check applies the SAME one: the SC
// tangent (apps/integral_direct/), the active-rotation and full orbital source
// terms (apps/sc_pseudodensity/), the PC tangent (nevpt2.pc_tangent) and --
// in Python, by the same definition (`fourfold_runs`) --
// reference_data/nevpt2_grad.py's ladders. test/gradient_plan/'s
// FourfoldRunTests pin it.

// The Richardson window: while truncation dominates, halving h cuts a central
// difference's error fourfold. A little slack each way -- the coarsest steps
// still carry an h^4 term, and the finest approach the round-off floor.
inline constexpr double kFourfoldLow = 3.2;
inline constexpr double kFourfoldHigh = 4.8;

// The longest run of consecutive halvings over a ladder.
struct FourfoldRun {
  int length = 0;        // how many consecutive ratios fell inside the window
  std::size_t end = 0;   // the index of the finer rung of the run's last halving
};

// `err[i]` is the ladder's error (or its change between rungs) at the i-th
// step, each step half the previous one; the ratio across halving i is
// err[i-1] / err[i], and a run counts the consecutive ratios strictly inside
// (lo, hi). A rung whose error is exactly zero has no ratio and breaks the
// run, as does a wild one. Ties go to the coarser run: the first run of the
// greatest length is the one reported.
FourfoldRun longestFourfoldRun(std::span<const double> err, double lo = kFourfoldLow,
                               double hi = kFourfoldHigh) {
  FourfoldRun best;
  int run = 0;
  for (std::size_t i = 1; i < err.size(); ++i) {
    const double ratio = err[i] > 0.0 ? err[i - 1] / err[i] : 0.0;
    run = (ratio > lo && ratio < hi) ? run + 1 : 0;
    if (run > best.length) best = FourfoldRun{run, i};
  }
  return best;
}

}  // namespace nevpt2
