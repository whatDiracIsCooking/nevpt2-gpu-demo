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

// The promotion table itself -- RotSpace, RotBlock, rotBlockInfo,
// activeAxesOf, rotBlockName, generalPickPlan, promotionPlan -- is pure host
// arithmetic and lives in nevpt2.gradient_plan (gradient_plan.cppm), where
// the unit tier pins it card-free (test/gradient_plan/). This partition is
// what reads it against the sidecar's device arrays.

// --- the sidecar's arrays, by what a promotion asks for ----------------------

// `[lo, hi)` of the sidecar's general index for one space.
std::pair<int64_t, int64_t> spanOf(const GeneralIntegralsDevice& gen, RotSpace s) {
  return spanOf(gen.ncore, gen.nact, gen.nvirt, s);
}

// The sidecar array a GeneralPickPlan names.
const DeviceTensor& generalArray(const GeneralIntegralsDevice& gen, GeneralArray a) {
  switch (a) {
    case GeneralArray::XaAa: return gen.g_xa_aa;
    case GeneralArray::XvAa: return gen.g_xv_aa;
    case GeneralArray::XcAa: return gen.g_xc_aa;
    case GeneralArray::XaVa: return gen.g_xa_va;
    case GeneralArray::XvVa: return gen.g_xv_va;
    case GeneralArray::XcVa: return gen.g_xc_va;
    case GeneralArray::XaCa: return gen.g_xa_ca;
    case GeneralArray::XvCa: return gen.g_xv_ca;
    case GeneralArray::XcCa: return gen.g_xc_ca;
    case GeneralArray::XaCv: return gen.g_xa_cv;
    case GeneralArray::XvCv: return gen.g_xv_cv;
    case GeneralArray::XcCv: return gen.g_xc_cv;
    case GeneralArray::XaCc: return gen.g_xa_cc;
    case GeneralArray::XvCc: return gen.g_xv_cc;
  }
  return gen.g_xa_aa;
}

// One promotion's source: the array generalPickPlan picks, and whether its
// last two axes hold (R, Q) rather than (Q, R).
struct GeneralPick {
  const DeviceTensor* g = nullptr;
  bool swap = false;
};

// `(x p | q r)` for the three spaces asked for (generalPickPlan has the
// rule). The two combinations no class can reach -- (v,v) in one
// distribution, and (c,c) with P = c -- are OUR bug if they are ever asked
// for (the table, not the caller's input), so they abort.
GeneralPick generalPick(const GeneralIntegralsDevice& gen, RotSpace p, RotSpace q, RotSpace r,
                        std::source_location loc = std::source_location::current()) {
  const std::optional<GeneralPickPlan> pl = generalPickPlan(p, q, r);
  if (!pl)
    check(false,
          "generalPick: the gradient sidecar carries no such one-general-index array "
          "(docs/gradient-theory.md, Sec. 3.3) -- the promotion table asked for one of the "
          "two arrays no class can reach",
          loc);
  return GeneralPick{&generalArray(gen, pl->array), pl->swap};
}

// --- the general action on one block ----------------------------------------

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
  // promotionPlan (nevpt2.gradient_plan) has the slot bookkeeping: the
  // promoted slot's partner, the other distribution, the array, its labels.
  for (int slot = 0; slot < 4; ++slot) {
    const PromotionPlan pl = promotionPlan(b, slot);
    deviceEinsumAccum(promotionSubscripts(pl),
                      {&generalArray(gen, pl.pick.array), kappaFor(pl.promoted.space)}, out, 1.0,
                      stream);
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
  //
  // The three terms -- which array, which subscripts, which coefficient -- are
  // nevpt2.gradient_plan's heffDressingPlan, so the unit tier can hold the
  // exchange pair's non-collapse card-free (test/gradient_plan's
  // HeffDressingTests, #64). Do not "simplify" the pair to twice one term.
  for (const HeffDressingTerm& t : heffDressingPlan(p.space, q.space))
    deviceEinsumAccum(t.subscripts, {&generalArray(gen, t.pick.array), kappaFor(RotSpace::Core)},
                      out, t.coeff, stream);

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
  static constexpr std::string_view kSlotName = "pqrs";
  double worst = 0.0;
  for (int slot = 0; slot < 4; ++slot) {
    const PromotionPlan pl = promotionPlan(b, slot);
    const auto [lo, hi] = spanOf(gen, pl.promoted.space);
    const DeviceTensor rows = sliceAxis(generalArray(gen, pl.pick.array), 0, lo, hi);
    // The same labels the rule's einsum gives the source, with the general
    // index now carrying the promoted slot's own label.
    std::string subs(1, kAxisLabels[static_cast<std::size_t>(pl.promoted.axis)]);
    subs += kAxisLabels[static_cast<std::size_t>(pl.partner.axis)];
    subs += kAxisLabels[static_cast<std::size_t>(pl.firstAxis)];
    subs += kAxisLabels[static_cast<std::size_t>(pl.lastAxis)];
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
