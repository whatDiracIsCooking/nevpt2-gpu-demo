// nevpt2.gradient:general -- the GENERAL-RANGE half of the orbital rotation:
// what a kappa whose indices leave the active space does to every array the
// class energies read, out of the gradient sidecar's one-general-index
// integral blocks (docs/gradient-theory.md, Sec. 3.3) and its full-MO-range
// one-electron and Fock matrices (Sec. 3.5).
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :general;` and from no importer of nevpt2.gradient.
// (Used by gradient_rotation.cppm, gradient_terms.cppm,
// gradient_sc_classes.cppm and gradient.cpp.)
//
// ONE RULE, NOT A TABLE OF DERIVATIVES. Every stored two-electron block is a
// slice of the full MO integral tensor, written in nevpt2.energy's physicist
// layout
//
//   X[p, q, r, s] = (p r | q s)                       (chemists' notation)
//
// -- which is `cvcv` too, with its roles on other axes (`rotBlockInfo`). The
// chain rule on one such slice is Sec. 3.1's promotion rule: for each of the
// four slots, replace that index by a general one and contract it with kappa.
// Writing the promoted slot's partner and the other charge distribution out,
//
//   dX[p,q,r,s]/dkappa, slot p:  sum_x kappa_xp (x r | q s)
//                       slot q:  sum_x kappa_xq (x s | p r)
//                       slot r:  sum_x kappa_xr (x p | q s)
//                       slot s:  sum_x kappa_xs (x q | p r)
//
// and every right-hand side is ONE sidecar array -- `g_xP_QR[x, p, q, r] =
// (x p | q r)`, where P is the space of the promoted slot's PARTNER and (Q,R)
// the other distribution's two spaces -- read with its axes labelled by the
// roles they carry (`generalPick` picks the array, and normalises (Q,R) to
// one of the five stored orderings by the integral's own (q r) <-> (r q)
// symmetry). So the whole core/virtual/mixed half of the source term is four
// einsums per block, generated from the block's space signature, and no
// derivative is transcribed per class.
//
// Sec. 3.4's missing eighteenth array never comes up: a promotion would need
// `(xv|vv)` only for a block with two virtual labels in ONE charge
// distribution, and no class has one.
//
// THE ONE-ELECTRON BLOCKS ARE h^eff SLICES, so they carry one term the rule
// above does not reach. With d^(0)_core = 2 delta_ij on the core,
//
//   h^eff_pq = h_pq + sum_j [ 2 (pq|jj) - (pj|jq) ],
//
// the core density is FIXED as an MO matrix (the rotation moves the orbitals,
// not which of them are core), so differentiating promotes its two tied
// indices as well:
//
//   dh^eff_pq = sum_x [ kappa_xp h^eff_xq + kappa_xq h^eff_px ]
//             + 4 sum_{x, j in core} kappa_xj (pq|xj)
//             -   sum_{x, j in core} kappa_xj [ (px|jq) + (pj|xq) ]
//
// where the COULOMB pair's two promotions collapse into one term, because
// `(pq|xj)` is symmetric in (x, j), and the EXCHANGE pair's do NOT: `(px|jq)`
// and `(pj|xq)` are different integrals, and writing the exchange as twice
// either one is wrong -- wrong in a way that survives even a core-internal
// rotation, where the true dressing response is zero. Both lines keep x over
// the WHOLE MO range: their own x-in-core parts are a symmetric matrix and a
// matched pair against an antisymmetric kappa, so they vanish on their own
// and nothing has to be excluded. `h1e_v_Sr` carries one more: it is
// `h^eff[v,a]` minus `sum_b (rb|bt)`, whose promotion is the same rule over a
// traced ACTIVE pair -- two different integrals again, and there the x-in-
// active parts are the pair that cancels.
//
// AND THE ORBITAL ENERGIES MOVE -- which is the half of Park Eq. 28 the
// active-block work (#36) could not reach and this one can. `e_core` /
// `e_virt` are the generalized Fock's diagonals, and f reads the SAME fixed
// MO densities (2 on the core diagonal, gamma^(1) on the active block), so
// the same argument gives
//
//   deps_p = 2 sum_x kappa_xp f_xp
//          + 4 sum_{x, j in core} kappa_xj [ (pp|xj) - (1/2) (px|jp) ]
//          + 2 sum_{x, u, t in act} kappa_xu gamma_ut [ (pp|xt) - (1/2) (px|tp) ]
//
// -- and HERE the exchange pair does collapse, unlike h^eff's above: with the
// two free indices equal, `(px|jp)` and `(pj|xp)` are the same integral
// `(xp|pj)`, so the factor 2 is right. (The active line's collapse is the
// commutator one: `[kappa, gamma]` against an integral pair that is symmetric
// in its two promoted labels.)
//
// for p in core and in virtual -- Eq. 28's `f d^Fock` and `g(d^Fock) d^(0)`
// terms, in the form the energy stage can consume them: a tangent of the
// `delta` vector each class divides by. Sec. 3.1's second structural fact is
// what makes the integrals cheap here: `Delta_t` reads f only on its
// diagonal, so the two labels of the promoted density are TIED to p, and the
// arrays wanted are the restricted rows of the inventory (`g_xa_vv`,
// `g_xc_vv`, `g_xc_cc_j`, `g_xc_cc_k`) or tied slices of a full one.
//
// TWO CONSEQUENCES WORTH STATING, both exact:
//
//   * a CORE-INTERNAL or VIRTUAL-INTERNAL rotation moves no orbital energy.
//     The first line above vanishes because f is diagonal inside core and
//     inside virtual ((O2), which the sidecar asserts); the second and third
//     vanish because kappa has no column outside its own space. So the
//     core-core and virtual-virtual blocks of Y -- the two Park Eqs. 56-57
//     read -- are the integral half alone, and are EXACT: nothing is held
//     fixed there that moves (h2e carries no core or virtual index either, so
//     the f3 digests do not move, and `scPseudocanonicalMultipliers` divides
//     an exact numerator).
//   * `(x i | j k)` with three free core labels is never wanted: every route
//     to it runs through the tied pairs above (Sec. 3.3), which is why the
//     sidecar stores only `g_xc_cc_j` / `g_xc_cc_k` and why `generalPick`
//     aborts rather than inventing a (C, C, C) array.
//
// WHAT CHECKS THE PROMOTION TABLE, since neither mode of the duality can
// (both read these tangents, so a wrong one cancels): `scGeneralIntegralCheck`
// restricts every rule's source array to the promoted slot's OWN space, where
// the promoted index is the block's own, and holds it against the stored
// block -- which pins the array choice and all four index permutations, per
// block, per slot. The same function rebuilds h^eff and the generalized
// Fock's diagonal out of the dressing rules' own sources and holds them
// against the sidecar's `heff` and `f`. On top of that, a core-internal and a
// virtual-internal rotation ARE finite-differenceable from the stored blocks
// alone (nothing leaves its space), and
// apps/sc_pseudodensity/:kappa_full_check does exactly that.
module nevpt2.gradient:general;

import std;
import nevpt2.gradient;

namespace nevpt2 {

// out += x, same shape: a no-permutation transpose, the one way this tree
// adds two device tensors. An empty `x` is a no-op -- what a block the
// general half does not reach hands back.
void addTensorInto(const DeviceTensor& x, const DeviceTensor& out, wwrStream_t stream) {
  if (x.rank() == 0) return;
  std::vector<int> axes(static_cast<std::size_t>(x.rank()));
  for (std::size_t a = 0; a < axes.size(); ++a) axes[a] = static_cast<int>(a);
  deviceTransposeAccum(x, axes, out, 1.0, stream);
}

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

// --- the sidecar's arrays, by what a promotion asks for ----------------------

// `[lo, hi)` of the sidecar's general index for one space.
std::pair<int64_t, int64_t> spanOf(const GeneralIntegralsDevice& gen, RotSpace s) {
  switch (s) {
    case RotSpace::Core: return {0, gen.ncore};
    case RotSpace::Act: return {gen.ncore, gen.ncore + gen.nact};
    case RotSpace::Virt: return {gen.ncore + gen.nact, gen.nmo()};
  }
  return {0, 0};
}

// One promotion's source: `g_xP_QR` and whether its last two axes hold (R, Q)
// rather than (Q, R) -- the integral's (q r) <-> (r q) symmetry, used to
// normalise the second charge distribution onto one of the five stored
// orderings.
struct GeneralPick {
  const DeviceTensor* g = nullptr;
  bool swap = false;
};

// `(x p | q r)` for the three spaces asked for. The five stored (Q, R) groups
// are (a,a), (v,a), (c,a), (c,v) and (c,c); (v,v) is never asked for (no
// class puts two virtuals in one charge distribution, which is also why
// Sec. 3.4's `(xv|vv)` is absent), and (c,c) is stored for P in {a, v} only
// (Sec. 3.3: every route to a third free core label runs through a tied pair).
// Both of those are OUR bug if they are ever reached -- the table above, not
// the caller's input -- so they abort.
GeneralPick generalPick(const GeneralIntegralsDevice& gen, RotSpace p, RotSpace q, RotSpace r,
                        std::source_location loc = std::source_location::current()) {
  using S = RotSpace;
  // Normalise the second distribution.
  bool swap = false;
  if ((q == S::Act && r == S::Virt) || (q == S::Act && r == S::Core) ||
      (q == S::Virt && r == S::Core)) {
    std::swap(q, r);
    swap = true;
  }
  const DeviceTensor* g = nullptr;
  if (q == S::Act && r == S::Act)
    g = p == S::Act ? &gen.g_xa_aa : (p == S::Virt ? &gen.g_xv_aa : &gen.g_xc_aa);
  else if (q == S::Virt && r == S::Act)
    g = p == S::Act ? &gen.g_xa_va : (p == S::Virt ? &gen.g_xv_va : &gen.g_xc_va);
  else if (q == S::Core && r == S::Act)
    g = p == S::Act ? &gen.g_xa_ca : (p == S::Virt ? &gen.g_xv_ca : &gen.g_xc_ca);
  else if (q == S::Core && r == S::Virt)
    g = p == S::Act ? &gen.g_xa_cv : (p == S::Virt ? &gen.g_xv_cv : &gen.g_xc_cv);
  else if (q == S::Core && r == S::Core && p != S::Core)
    g = p == S::Act ? &gen.g_xa_cc : &gen.g_xv_cc;
  if (g == nullptr)
    check(false,
          "generalPick: the gradient sidecar carries no such one-general-index array "
          "(docs/gradient-theory.md, Sec. 3.3) -- the promotion table asked for one of the "
          "two arrays no class can reach",
          loc);
  return GeneralPick{g, swap};
}

// --- the general action on one block ----------------------------------------

// The label each storage axis of a block carries; `z` is the promoted
// (general) index and `m` the one a dressing term contracts over.
constexpr std::string_view kAxisLabels = "ijkl";

// Accumulates the general-range part of `dkappa`'s action on a two-electron
// block into `out` (shaped like the block): one einsum per slot, over the
// WHOLE general index, with kappa's columns for that slot's space.
//
// `kappaFor` hands back the column block kappa[:, space] to use -- and for an
// ACTIVE slot that block has its active ROWS zeroed, because the x in active
// part of that slot's sum is exactly what #36's intra-active action already
// did. So the two halves partition the sum with no overlap, and an
// active-restricted kappa makes every einsum here an exactly-zero tensor
// (which is what keeps the active block's answer bit-for-bit #36's).
template <class KappaFn>
void accumGeneralTwoElectron(RotBlock b, const GeneralIntegralsDevice& gen, KappaFn&& kappaFor,
                             const DeviceTensor& out, wwrStream_t stream) {
  // Which role is the promoted slot's partner in its charge distribution, and
  // which two roles are the other distribution: p -> (r; q, s), q -> (s; p,
  // r), r -> (p; q, s), s -> (q; p, r).
  static constexpr int kPartner[4] = {2, 3, 0, 1};
  static constexpr int kSecond[4][2] = {{1, 3}, {0, 2}, {1, 3}, {0, 2}};
  const RotBlockInfo info = rotBlockInfo(b);
  for (int slot = 0; slot < 4; ++slot) {
    const RotRole prom = info.roles[static_cast<std::size_t>(slot)];
    const RotRole partner = info.roles[static_cast<std::size_t>(kPartner[slot])];
    const RotRole second0 = info.roles[static_cast<std::size_t>(kSecond[slot][0])];
    const RotRole second1 = info.roles[static_cast<std::size_t>(kSecond[slot][1])];
    const GeneralPick pick = generalPick(gen, partner.space, second0.space, second1.space);
    std::string subs = "z";
    subs += kAxisLabels[static_cast<std::size_t>(partner.axis)];
    const int firstAxis = pick.swap ? second1.axis : second0.axis;
    const int lastAxis = pick.swap ? second0.axis : second1.axis;
    subs += kAxisLabels[static_cast<std::size_t>(firstAxis)];
    subs += kAxisLabels[static_cast<std::size_t>(lastAxis)];
    subs += ",z";
    subs += kAxisLabels[static_cast<std::size_t>(prom.axis)];
    subs += "->";
    subs += kAxisLabels;
    deviceEinsumAccum(subs, {pick.g, kappaFor(prom.space)}, out, 1.0, stream);
  }
}

// The same for a one-electron (h^eff) block: the two index transformations of
// h^eff itself, the core dressing's response, and -- for h1e_v_Sr -- the four
// promotions of its subtracted active trace. See the header for the algebra.
template <class KappaFn>
void accumGeneralOneElectron(RotBlock b, const GeneralIntegralsDevice& gen, KappaFn&& kappaFor,
                             const DeviceTensor& out, const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  const RotBlockInfo info = rotBlockInfo(b);
  const RotRole p = info.roles[0], q = info.roles[1];
  const auto [pLo, pHi] = spanOf(gen, p.space);
  const auto [qLo, qHi] = spanOf(gen, q.space);

  // (1) sum_x [ kappa_xp heff_xq + kappa_xq heff_px ], using heff's symmetry
  // to read both halves as columns of the same matrix.
  const DeviceTensor heffQ = sliceAxis(gen.heff, 1, qLo, qHi);  // (nmo, |space(q)|)
  const DeviceTensor heffP = sliceAxis(gen.heff, 1, pLo, pHi);  // (nmo, |space(p)|)
  deviceEinsumAccum("zj,zi->ij", {&heffQ, kappaFor(p.space)}, out, 1.0, stream);
  deviceEinsumAccum("zi,zj->ij", {&heffP, kappaFor(q.space)}, out, 1.0, stream);

  // (2) the core dressing. Its two core labels are TIED, so each of them is
  // promoted in turn and the four resulting terms are
  //
  //   4 sum_{x, j in core} kappa_xj (pq|xj)
  //     - sum_{x, j in core} kappa_xj [ (px|jq) + (pj|xq) ],
  //
  // the Coulomb pair collapsing because (pq|xj) is symmetric in (x, j) and
  // the exchange pair NOT collapsing because (px|jq) and (pj|xq) are
  // different integrals. Both lines keep x over the WHOLE general range:
  // their own x-in-core parts are a symmetric matrix (the Coulomb) and a
  // matched pair (the exchange) against an antisymmetric kappa, hence zero,
  // so nothing has to be excluded -- and a core-INTERNAL rotation, for which
  // every surviving row would be a core one, correctly gets nothing from the
  // dressing at all (which is what the core-internal finite difference in
  // apps/sc_pseudodensity/:kappa_full_check measures).
  const GeneralPick cou = generalPick(gen, RotSpace::Core, p.space, q.space);
  std::string couSubs = "zm";
  couSubs += kAxisLabels[cou.swap ? 1 : 0];
  couSubs += kAxisLabels[cou.swap ? 0 : 1];
  couSubs += ",zm->ij";
  deviceEinsumAccum(couSubs, {cou.g, kappaFor(RotSpace::Core)}, out, 4.0, stream);
  // (px|jq) = (x p | j q): P = space(p), and (C, S_q) is stored in that order
  // for every S_q, so this source is never swapped -- its axes are
  // (x, p, j, q).
  const GeneralPick exc1 = generalPick(gen, p.space, RotSpace::Core, q.space);
  check(!exc1.swap, "accumGeneralOneElectron: the first exchange source is not swapped");
  deviceEinsumAccum("zimj,zm->ij", {exc1.g, kappaFor(RotSpace::Core)}, out, -1.0, stream);
  // (pj|xq) = (x q | p j): P = space(q), and the second distribution is
  // (S_p, C), which IS swapped whenever p is not core.
  const GeneralPick exc2 = generalPick(gen, q.space, p.space, RotSpace::Core);
  std::string excSubs = "zj";
  excSubs += exc2.swap ? 'm' : kAxisLabels[0];
  excSubs += exc2.swap ? kAxisLabels[0] : 'm';
  excSubs += ",zm->ij";
  deviceEinsumAccum(excSubs, {exc2.g, kappaFor(RotSpace::Core)}, out, -1.0, stream);

  if (!info.trace) return;
  // (3) h1e_v_Sr is h^eff[v,a] - T with T_rt = sum_b (r b | b t), so every
  // promotion of T comes in with a minus sign. Two of them are traces of a
  // sidecar array over its active pair; the other two are the plain rule over
  // the contracted active index, whose own x in active part cancels inside
  // the trace (an antisymmetric kappa against a pair the trace symmetrises)
  // and so uses the same active columns as every other active slot.
  const DeviceTensor tTr = deviceEinsumNew("zmmj->zj", {&gen.g_xa_aa}, 1.0, res);
  const DeviceTensor tVr = deviceEinsumNew("zmim->zi", {&gen.g_xa_va}, 1.0, res);
  deviceEinsumAccum("zj,zi->ij", {&tTr, kappaFor(p.space)}, out, -1.0, stream);
  deviceEinsumAccum("zi,zj->ij", {&tVr, kappaFor(q.space)}, out, -1.0, stream);
  deviceEinsumAccum("zimj,zm->ij", {&gen.g_xv_aa, kappaFor(RotSpace::Act)}, out, -1.0, stream);
  deviceEinsumAccum("zjim,zm->ij", {&gen.g_xa_va, kappaFor(RotSpace::Act)}, out, -1.0, stream);
}

// `kappa`'s general-range action on `block`, as a fresh tensor from res's
// pool, or an empty tensor for a block the general half does not reach
// (V2tr, whose tangent :terms derives from Sir2's). `dims` is the block's own
// shape: the general action is written for WHOLE blocks (the entry points
// that supply a sidecar walk one slab per block), and a shape that is not the
// block's full one is our bug.
template <class KappaFn>
DeviceTensor generalBlockAction(RotBlock b, const GeneralIntegralsDevice& gen,
                                KappaFn&& kappaFor, const std::vector<int64_t>& dims,
                                const DeviceResources& res,
                                std::source_location loc = std::source_location::current()) {
  const RotBlockInfo info = rotBlockInfo(b);
  if (info.rank == 0) return DeviceTensor();
  if (std::ssize(dims) != info.rank)
    check(false,
          std::format("generalBlockAction: {} is a rank-{} block, given rank {}",
                      rotBlockName(b), info.rank, std::ssize(dims)),
          loc);
  for (int k = 0; k < info.rank; ++k) {
    const RotRole role = info.roles[static_cast<std::size_t>(k)];
    const auto [lo, hi] = spanOf(gen, role.space);
    if (dims[static_cast<std::size_t>(role.axis)] != hi - lo)
      check(false,
            std::format("generalBlockAction: {} axis {} has extent {}, the sidecar's space has "
                        "{} -- the general action needs whole blocks, not slabs",
                        rotBlockName(b), role.axis, dims[static_cast<std::size_t>(role.axis)],
                        hi - lo),
            loc);
  }
  DeviceTensor out = DeviceTensor::zeros(dims, res);
  if (info.rank == 4)
    accumGeneralTwoElectron(b, gen, kappaFor, out, res.stream());
  else
    accumGeneralOneElectron(b, gen, kappaFor, out, res);
  return out;
}

// --- the orbital energies' response -----------------------------------------

// `deps_p` for p in core and in virtual, along one direction: the three terms
// of the header's formula, downloaded to the host because that is where the
// `delta` vector each class divides by is assembled.
//
// `gamma` is the active 1-RDM (dm1) -- the density f is built from, held
// fixed with the rest of the RDM side.
//
// The kappa column blocks come in explicitly, and `kA` is the FULL active
// column block, active rows included: there is no intra-active half of this
// term for it to avoid double-counting (#36 held the orbital energies fixed
// altogether, which is the gap this closes), so every general row counts.
struct ScEpsDot {
  Tensor core;  // (ncore)
  Tensor virt;  // (nvirt)
};

ScEpsDot generalEpsResponse(const GeneralIntegralsDevice& gen, const DeviceTensor& gamma,
                            const DeviceTensor& kC, const DeviceTensor& kA,
                            const DeviceTensor& kV, const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  const auto [cLo, cHi] = spanOf(gen, RotSpace::Core);
  const auto [vLo, vHi] = spanOf(gen, RotSpace::Virt);
  ScEpsDot out;

  if (cHi > cLo) {
    const DeviceTensor fC = sliceAxis(gen.f, 1, cLo, cHi);  // f[x, i]
    DeviceTensor d = DeviceTensor::zeros({cHi - cLo}, res);
    // 2 sum_x kappa_xi f_xi
    deviceEinsumAccum("zi,zi->i", {&fC, &kC}, d, 2.0, stream);
    // the core dressing: (ii|xj) = g_xc_cc_j[x, j, i], (i x|j i) = (x i|i j)
    // = g_xc_cc_k[x, i, j]
    deviceEinsumAccum("zmi,zm->i", {&gen.g_xc_cc_j, &kC}, d, 4.0, stream);
    deviceEinsumAccum("zim,zm->i", {&gen.g_xc_cc_k, &kC}, d, -2.0, stream);
    // the active density: (ii|xt) = g_xa_cc[x, t, i, i], (i x|t i) = (x i|i t)
    // = g_xc_ca[x, i, i, t] -- both read on their tied core pair.
    const DeviceTensor cou = deviceEinsumNew("ztii->zti", {&gen.g_xa_cc}, 1.0, res);
    const DeviceTensor exc = deviceEinsumNew("zmmt->zmt", {&gen.g_xc_ca}, 1.0, res);
    deviceEinsumAccum("zti,ut,zu->i", {&cou, &gamma, &kA}, d, 2.0, stream);
    deviceEinsumAccum("zit,ut,zu->i", {&exc, &gamma, &kA}, d, -1.0, stream);
    out.core = downloadTensor(d, stream);
  } else {
    out.core = Tensor({0});
  }

  if (vHi > vLo) {
    const DeviceTensor fV = sliceAxis(gen.f, 1, vLo, vHi);  // f[x, r]
    DeviceTensor d = DeviceTensor::zeros({vHi - vLo}, res);
    deviceEinsumAccum("zr,zr->r", {&fV, &kV}, d, 2.0, stream);
    // (rr|xj) = g_xc_vv[x, j, r], (r x|j r) = (x r|j r) = g_xv_cv[x, r, j, r]
    const DeviceTensor excC = deviceEinsumNew("zrmr->zrm", {&gen.g_xv_cv}, 1.0, res);
    deviceEinsumAccum("zmr,zm->r", {&gen.g_xc_vv, &kC}, d, 4.0, stream);
    deviceEinsumAccum("zrm,zm->r", {&excC, &kC}, d, -2.0, stream);
    // (rr|xt) = g_xa_vv[x, t, r], (r x|t r) = (x r|r t) = g_xv_va[x, r, r, t]
    const DeviceTensor excA = deviceEinsumNew("zrrt->zrt", {&gen.g_xv_va}, 1.0, res);
    deviceEinsumAccum("ztr,ut,zu->r", {&gen.g_xa_vv, &gamma, &kA}, d, 2.0, stream);
    deviceEinsumAccum("zrt,ut,zu->r", {&excA, &gamma, &kA}, d, -1.0, stream);
    out.virt = downloadTensor(d, stream);
  } else {
    out.virt = Tensor({0});
  }
  return out;
}

// --- what checks the promotion table ----------------------------------------

// sum of x's squares, as one einsum reduced to x's LEADING axis (never to a
// rank-0 output) and finished on the host. :terms' dotTensors is the
// two-operand version and lives where the pairings are.
double sumSquares(const DeviceTensor& x, const DeviceResources& res,
                  std::source_location loc = std::source_location::current()) {
  if (x.rank() < 1) check(false, "sumSquares: needs a rank >= 1", loc);
  static constexpr std::string_view kLabels = "abcdef";
  const std::string lab(kLabels.substr(0, static_cast<std::size_t>(x.rank())));
  const DeviceTensor reduced =
      deviceEinsumNew(lab + "," + lab + "->" + lab.substr(0, 1), {&x, &x}, 1.0, res);
  const Tensor host = downloadTensor(reduced, res.stream());
  double sum = 0.0;
  for (const double v : host.data()) sum += v;
  return sum;
}

// ||a - b|| / ||b||, the residual every check below reports: `a` is what a
// rule's own source array rebuilds, `b` what the golden or the sidecar
// already holds. `a` is consumed (it is a fresh accumulator).
double relativeResidual(const DeviceTensor& a, const DeviceTensor& b,
                        const DeviceResources& res) {
  const double scale = sumSquares(b, res);
  std::vector<int> axes(static_cast<std::size_t>(b.rank()));
  for (std::size_t k = 0; k < axes.size(); ++k) axes[k] = static_cast<int>(k);
  deviceTransposeAccum(b, axes, a, -1.0, res.stream());
  return std::sqrt(sumSquares(a, res) / std::max(scale, 1e-300));
}

// Every promotion rule of one two-electron block, restricted to the promoted
// slot's OWN space: there the general index IS the block's own index, so the
// source array must be the stored block under exactly the permutation the
// rule uses -- which is what pins the array choice and the labels. The worst
// of the four slots, and `where` names it.
double restrictionResidual4(RotBlock b, const GeneralIntegralsDevice& gen,
                            const DeviceTensor& stored, const DeviceResources& res,
                            std::string& where) {
  static constexpr int kPartner[4] = {2, 3, 0, 1};
  static constexpr int kSecond[4][2] = {{1, 3}, {0, 2}, {1, 3}, {0, 2}};
  static constexpr std::string_view kSlotName = "pqrs";
  const RotBlockInfo info = rotBlockInfo(b);
  double worst = 0.0;
  for (int slot = 0; slot < 4; ++slot) {
    const RotRole prom = info.roles[static_cast<std::size_t>(slot)];
    const RotRole partner = info.roles[static_cast<std::size_t>(kPartner[slot])];
    const RotRole second0 = info.roles[static_cast<std::size_t>(kSecond[slot][0])];
    const RotRole second1 = info.roles[static_cast<std::size_t>(kSecond[slot][1])];
    const GeneralPick pick = generalPick(gen, partner.space, second0.space, second1.space);
    const auto [lo, hi] = spanOf(gen, prom.space);
    const DeviceTensor rows = sliceAxis(*pick.g, 0, lo, hi);
    // The same labels the rule's einsum gives the source, with the general
    // index now carrying the promoted slot's own label.
    std::string subs(1, kAxisLabels[static_cast<std::size_t>(prom.axis)]);
    subs += kAxisLabels[static_cast<std::size_t>(partner.axis)];
    subs += kAxisLabels[static_cast<std::size_t>(pick.swap ? second1.axis : second0.axis)];
    subs += kAxisLabels[static_cast<std::size_t>(pick.swap ? second0.axis : second1.axis)];
    subs += "->";
    subs += kAxisLabels;
    DeviceTensor rebuilt = deviceEinsumNew(subs, {&rows}, 1.0, res);
    const double resid = relativeResidual(rebuilt, stored, res);
    if (resid > worst) {
      worst = resid;
      where = std::format("{} slot {}", rotBlockName(b), kSlotName[static_cast<std::size_t>(
                                                             slot)]);
    }
  }
  return worst;
}

// The same for a one-electron block: h^eff's own columns, restricted to the
// other index's space, are the stored block -- which also says the sidecar's
// `heff` is the matrix the golden's one-electron slices came from. For
// h1e_v_Sr the subtracted active trace is checked twice over, once through
// each of the two arrays its promotions read.
double restrictionResidual2(RotBlock b, const GeneralIntegralsDevice& gen,
                            const DeviceTensor& stored, const DeviceResources& res,
                            std::string& where) {
  const RotBlockInfo info = rotBlockInfo(b);
  const auto [pLo, pHi] = spanOf(gen, info.roles[0].space);
  const auto [qLo, qHi] = spanOf(gen, info.roles[1].space);
  const DeviceTensor rows = sliceAxis(sliceAxis(gen.heff, 0, pLo, pHi), 1, qLo, qHi);
  double worst = 0.0;
  if (!info.trace) {
    DeviceTensor rebuilt = deviceEinsumNew("ij->ij", {&rows}, 1.0, res);
    worst = relativeResidual(rebuilt, stored, res);
    where = rotBlockName(b);
    return worst;
  }
  // T_rt = sum_b (r b | b t), reachable as either trace array's own rows.
  const DeviceTensor tTr = deviceEinsumNew("zmmj->zj", {&gen.g_xa_aa}, 1.0, res);
  const DeviceTensor tVr = deviceEinsumNew("zmim->zi", {&gen.g_xa_va}, 1.0, res);
  const DeviceTensor tTrRows = sliceAxis(tTr, 0, pLo, pHi);   // (v, a)
  const DeviceTensor tVrRows = sliceAxis(tVr, 0, qLo, qHi);   // (a, v)
  for (int which = 0; which < 2; ++which) {
    DeviceTensor rebuilt = deviceEinsumNew("ij->ij", {&rows}, 1.0, res);
    if (which == 0)
      deviceEinsumAccum("ij->ij", {&tTrRows}, rebuilt, -1.0, res.stream());
    else
      deviceEinsumAccum("ji->ij", {&tVrRows}, rebuilt, -1.0, res.stream());
    const double resid = relativeResidual(rebuilt, stored, res);
    if (resid > worst) {
      worst = resid;
      where = std::format("{} ({})", rotBlockName(b), which == 0 ? "trace g_xa_aa"
                                                                 : "trace g_xa_va");
    }
  }
  return worst;
}

// h^eff rebuilt from `h` and the DRESSING RULES' own sources, on one
// one-electron block's space pair: the Coulomb source read on its tied core
// pair is sum_j (pq|jj), the exchange source likewise is sum_j (pj|jq), and
// the two must put `h` back onto the sidecar's `heff`. That is the only check
// those two arrays get -- a derivative-only term has no stored block to be
// the restriction of.
double heffResidual(RotBlock b, const GeneralIntegralsDevice& gen,
                    const DeviceResources& res) {
  const RotBlockInfo info = rotBlockInfo(b);
  const auto [pLo, pHi] = spanOf(gen, info.roles[0].space);
  const auto [qLo, qHi] = spanOf(gen, info.roles[1].space);
  const GeneralPick cou = generalPick(gen, RotSpace::Core, info.roles[0].space,
                                      info.roles[1].space);
  const GeneralPick exc = generalPick(gen, info.roles[0].space, RotSpace::Core,
                                      info.roles[1].space);
  const DeviceTensor hRows = sliceAxis(sliceAxis(gen.h, 0, pLo, pHi), 1, qLo, qHi);
  const DeviceTensor heffRows = sliceAxis(sliceAxis(gen.heff, 0, pLo, pHi), 1, qLo, qHi);
  // Both sources are read with their general index tied to the core label
  // they already carry, so the general axis is first cut down to the core
  // span -- otherwise the repeated label would span two different extents.
  const auto [cLo, cHi] = spanOf(gen, RotSpace::Core);
  const DeviceTensor couCore = sliceAxis(*cou.g, 0, cLo, cHi);
  const DeviceTensor excCore = sliceAxis(*exc.g, 0, cLo, cHi);
  DeviceTensor rebuilt = deviceEinsumNew("ij->ij", {&hRows}, 1.0, res);
  deviceEinsumAccum(cou.swap ? "mmji->ij" : "mmij->ij", {&couCore}, rebuilt, 2.0, res.stream());
  deviceEinsumAccum("mimj->ij", {&excCore}, rebuilt, -1.0, res.stream());
  return relativeResidual(rebuilt, heffRows, res);
}

// The generalized Fock's core and virtual DIAGONALS, rebuilt from `heff` and
// the orbital-energy rules' own tied arrays against gamma -- the only check
// those arrays get, for the same reason. `core` picks which of the two.
double fockDiagResidual(bool core, const GeneralIntegralsDevice& gen, const DeviceTensor& gamma,
                        const DeviceResources& res) {
  const auto [lo, hi] = spanOf(gen, core ? RotSpace::Core : RotSpace::Virt);
  if (hi <= lo) return 0.0;
  const DeviceTensor heffPP =
      sliceAxis(sliceAxis(gen.heff, 0, lo, hi), 1, lo, hi);  // (n, n)
  const DeviceTensor fPP = sliceAxis(sliceAxis(gen.f, 0, lo, hi), 1, lo, hi);
  DeviceTensor rebuilt = deviceEinsumNew("ii->i", {&heffPP}, 1.0, res);
  const DeviceTensor fDiag = deviceEinsumNew("ii->i", {&fPP}, 1.0, res);
  const auto [aLo, aHi] = spanOf(gen, RotSpace::Act);
  // Each tied array is named before it is sliced: a view of a temporary would
  // outlive the buffer it views (the free is stream-ordered, so the pool
  // could hand the memory to the very einsum that reads it).
  if (core) {
    const DeviceTensor couAll = deviceEinsumNew("ztii->zti", {&gen.g_xa_cc}, 1.0, res);
    const DeviceTensor excAll = deviceEinsumNew("zmmt->zmt", {&gen.g_xc_ca}, 1.0, res);
    const DeviceTensor cou = sliceAxis(couAll, 0, aLo, aHi);
    const DeviceTensor exc = sliceAxis(excAll, 0, aLo, aHi);
    deviceEinsumAccum("ut,uti->i", {&gamma, &cou}, rebuilt, 1.0, res.stream());
    deviceEinsumAccum("ut,uit->i", {&gamma, &exc}, rebuilt, -0.5, res.stream());
  } else {
    const DeviceTensor excAll = deviceEinsumNew("zrrt->zrt", {&gen.g_xv_va}, 1.0, res);
    const DeviceTensor cou = sliceAxis(gen.g_xa_vv, 0, aLo, aHi);
    const DeviceTensor exc = sliceAxis(excAll, 0, aLo, aHi);
    deviceEinsumAccum("ut,utr->r", {&gamma, &cou}, rebuilt, 1.0, res.stream());
    deviceEinsumAccum("ut,urt->r", {&gamma, &exc}, rebuilt, -0.5, res.stream());
  }
  return relativeResidual(rebuilt, fDiag, res);
}

// f's core and virtual diagonals against the golden's e_core / e_virt: (O2)
// (docs/gradient-theory.md, Sec. 3.6) seen from this side, and what says the
// sidecar belongs to this golden.
double epsResidual(bool core, const GeneralIntegralsDevice& gen, const DeviceTensor& eps,
                   const DeviceResources& res) {
  const auto [lo, hi] = spanOf(gen, core ? RotSpace::Core : RotSpace::Virt);
  if (hi <= lo) return 0.0;
  const DeviceTensor fPP = sliceAxis(sliceAxis(gen.f, 0, lo, hi), 1, lo, hi);
  DeviceTensor diag = deviceEinsumNew("ii->i", {&fPP}, 1.0, res);
  return relativeResidual(diag, eps, res);
}

}  // namespace nevpt2
