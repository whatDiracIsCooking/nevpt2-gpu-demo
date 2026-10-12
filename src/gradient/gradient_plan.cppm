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

// --- the two symmetrized classes' multipliers --------------------------------

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
  Tensor normSym({nr, ncore, ncore}), hSym({nr, ncore, ncore});
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t j = 0; j < ncore; ++j)
      for (int64_t i = 0; i < ncore; ++i) {
        normSym(r, j, i) = normT(r, j, i) + normT(r, i, j);
        hSym(r, j, i) = hT(r, j, i) + hT(r, i, j);
      }
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t kk = 0; kk < ncore; ++kk) {
      normSym(r, kk, kk) *= 0.5;
      hSym(r, kk, kk) *= 0.5;
    }
  std::vector<double> normTri, hTri;
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t j = 0; j < ncore; ++j)
      for (int64_t i = j; i < ncore; ++i) {
        normTri.push_back(normSym(r, j, i));
        hTri.push_back(hSym(r, j, i));
      }
  const ScMultipliers m =
      scMultipliers(normTri, hTri, sijrDelta(nr, ncore, b0, hostECore, hostEVirt));
  Tensor p({nr, ncore, ncore}), q({nr, ncore, ncore});
  std::size_t k = 0;
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t j = 0; j < ncore; ++j)
      for (int64_t i = j; i < ncore; ++i, ++k) {
        p(r, j, i) = m.p[k];
        p(r, i, j) = m.p[k];
        q(r, j, i) = m.q[k];
        q(r, i, j) = m.q[k];
      }
  // `r` stays in the triangle's own order: it multiplies Deltadot, which
  // sijrDelta builds there too.
  return ScSlabMultipliers{std::move(p), std::move(q), m.r, m.norm, m.energy};
}

// energy_Srsi's symmetrization and packing, verbatim, plus the scatter back --
// as sijrMultipliers, but over the whole (r, s, i) and symmetric in (r, s).
ScSlabMultipliers srsiMultipliers(const Tensor& normT, const Tensor& hT,
                                  const Tensor& hostECore, const Tensor& hostEVirt) {
  const int64_t nvirt = normT.dim(0), ncore = normT.dim(2);
  Tensor normSym({nvirt, nvirt, ncore}), hSym({nvirt, nvirt, ncore});
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t ss = 0; ss < nvirt; ++ss)
      for (int64_t i = 0; i < ncore; ++i) {
        normSym(r, ss, i) = normT(r, ss, i) + normT(ss, r, i);
        hSym(r, ss, i) = hT(r, ss, i) + hT(ss, r, i);
      }
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t i = 0; i < ncore; ++i) {
      normSym(r, r, i) *= 0.5;
      hSym(r, r, i) *= 0.5;
    }
  std::vector<double> normTri, hTri;
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t ss = r; ss < nvirt; ++ss)
      for (int64_t i = 0; i < ncore; ++i) {
        normTri.push_back(normSym(r, ss, i));
        hTri.push_back(hSym(r, ss, i));
      }
  const ScMultipliers m =
      scMultipliers(normTri, hTri, srsiDelta(nvirt, ncore, hostECore, hostEVirt));
  Tensor p({nvirt, nvirt, ncore}), q({nvirt, nvirt, ncore});
  std::size_t k = 0;
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t ss = r; ss < nvirt; ++ss)
      for (int64_t i = 0; i < ncore; ++i, ++k) {
        p(r, ss, i) = m.p[k];
        p(ss, r, i) = m.p[k];
        q(r, ss, i) = m.q[k];
        q(ss, r, i) = m.q[k];
      }
  return ScSlabMultipliers{std::move(p), std::move(q), m.r, m.norm, m.energy};
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
