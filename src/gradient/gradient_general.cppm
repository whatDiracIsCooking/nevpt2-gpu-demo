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

// THE KAPPA-LINEAR EINSUMS, WRITTEN ONCE (#240). Every einsum of the general
// action -- and of the orbital energies' response below -- is linear in ONE
// kappa column block, which it reads as its LAST operand. Each is spelled
// once, as `emit(subs, operands-before-kappa, space, coef)`, and two emitters
// read the same list: the FORWARD one appends kappa[:, space] and accumulates
// into the block's tangent (what every entry point before #240 ran: the same
// einsums, in the same order); the TRANSPOSE one appends the tangent's
// adjoint instead and accumulates d<bar, tangent>/dkappa[:, space] -- the
// einsum rewritten for its kappa operand by derivativeSubscripts, as
// :rdm_adjoint rewrites a block's chain for an RDM operand. So the one-pass Y
// (gradient.cppm, "Y in one reverse pass") and the probed one read the SAME
// transcription of the promotion rule, and no derivative is written twice.
using KappaOperands = std::vector<const DeviceTensor*>;

// The forward emitter: kappa[:, space] appended, `out` accumulated.
template <class KappaFn>
auto forwardEmitter(KappaFn& kappaFor, const DeviceTensor& out, wwrStream_t stream) {
  return [&kappaFor, &out, stream](const std::string& subs, KappaOperands ops, RotSpace s,
                                   double coef) {
    ops.push_back(kappaFor(s));
    deviceEinsumAccum(subs, ops, out, coef, stream);
  };
}

// The transpose emitter (#240): `bar` -- the adjoint of what the forward one
// accumulates into -- appended in kappa's place, and the einsum rewritten for
// its kappa operand (the LAST one), accumulated into *kbarFor(space), the
// adjoint of that column block. A column block of an empty space has nothing
// to receive.
template <class KbarFn>
auto transposeEmitter(KbarFn& kbarFor, const DeviceTensor& bar, wwrStream_t stream) {
  return [&kbarFor, &bar, stream](const std::string& subs, KappaOperands ops, RotSpace s,
                                  double coef) {
    const DeviceTensor& dst = *kbarFor(s);
    if (dst.size() == 0) return;
    const int slot = static_cast<int>(ops.size());
    ops.push_back(&bar);
    deviceEinsumAccum(derivativeSubscripts(subs, slot), ops, dst, coef, stream);
  };
}

// The general-range part of `dkappa`'s action on a two-electron block: one
// einsum per slot, over the WHOLE general index, with kappa's columns for
// that slot's space.
//
// `kappaFor` hands back the column block kappa[:, space] to use -- and for an
// ACTIVE slot that block has its active ROWS zeroed, because the x in active
// part of that slot's sum is exactly what #36's intra-active action already
// did. So the two halves partition the sum with no overlap, and an
// active-restricted kappa makes every einsum here an exactly-zero tensor
// (which is what keeps the active block's answer bit-for-bit #36's).
template <class EmitFn>
void emitGeneralTwoElectron(RotBlock b, const GeneralIntegralsDevice& gen, EmitFn&& emit) {
  // promotionPlan (nevpt2.gradient_plan) has the slot bookkeeping: the
  // promoted slot's partner, the other distribution, the array, its labels.
  for (int slot = 0; slot < 4; ++slot) {
    const PromotionPlan pl = promotionPlan(b, slot);
    emit(promotionSubscripts(pl), KappaOperands{&generalArray(gen, pl.pick.array)},
         pl.promoted.space, 1.0);
  }
}

template <class KappaFn>
void accumGeneralTwoElectron(RotBlock b, const GeneralIntegralsDevice& gen, KappaFn&& kappaFor,
                             const DeviceTensor& out, wwrStream_t stream) {
  emitGeneralTwoElectron(b, gen, forwardEmitter(kappaFor, out, stream));
}

// The same for a one-electron (h^eff) block: the two index transformations of
// h^eff itself, the core dressing's response, and -- for h1e_v_Sr -- the four
// promotions of its subtracted active trace. See the header for the algebra.
template <class EmitFn>
void emitGeneralOneElectron(RotBlock b, const GeneralIntegralsDevice& gen, EmitFn&& emit,
                            const DeviceResources& res) {
  const RotBlockInfo info = rotBlockInfo(b);
  const RotRole p = info.roles[0], q = info.roles[1];
  const auto [pLo, pHi] = spanOf(gen, p.space);
  const auto [qLo, qHi] = spanOf(gen, q.space);

  // (1) sum_x [ kappa_xp heff_xq + kappa_xq heff_px ], using heff's symmetry
  // to read both halves as columns of the same matrix.
  const DeviceTensor heffQ = sliceAxis(gen.heff, 1, qLo, qHi);  // (nmo, |space(q)|)
  const DeviceTensor heffP = sliceAxis(gen.heff, 1, pLo, pHi);  // (nmo, |space(p)|)
  emit("zj,zi->ij", KappaOperands{&heffQ}, p.space, 1.0);
  emit("zi,zj->ij", KappaOperands{&heffP}, q.space, 1.0);

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
    emit(t.subscripts, KappaOperands{&generalArray(gen, t.pick.array)}, RotSpace::Core,
         t.coeff);

  if (!info.trace) return;
  // (3) h1e_v_Sr is h^eff[v,a] - T with T_rt = sum_b (r b | b t), so every
  // promotion of T comes in with a minus sign. Two of them are traces of a
  // sidecar array over its active pair; the other two are the plain rule over
  // the contracted active index, whose own x in active part cancels inside
  // the trace (an antisymmetric kappa against a pair the trace symmetrises)
  // and so uses the same active columns as every other active slot.
  const DeviceTensor tTr = deviceEinsumNew("zmmj->zj", {&gen.g_xa_aa}, 1.0, res);
  const DeviceTensor tVr = deviceEinsumNew("zmim->zi", {&gen.g_xa_va}, 1.0, res);
  emit("zj,zi->ij", KappaOperands{&tTr}, p.space, -1.0);
  emit("zi,zj->ij", KappaOperands{&tVr}, q.space, -1.0);
  emit("zimj,zm->ij", KappaOperands{&gen.g_xv_aa}, RotSpace::Act, -1.0);
  emit("zjim,zm->ij", KappaOperands{&gen.g_xa_va}, RotSpace::Act, -1.0);
}

template <class KappaFn>
void accumGeneralOneElectron(RotBlock b, const GeneralIntegralsDevice& gen, KappaFn&& kappaFor,
                             const DeviceTensor& out, const DeviceResources& res) {
  emitGeneralOneElectron(b, gen, forwardEmitter(kappaFor, out, res.stream()), res);
}

// A block the general action is asked about must be WHOLE: its extents the
// sidecar's spaces (a slab is our bug -- the entry points that supply a
// sidecar walk one slab per block). Returns the block's rank, 0 for one the
// general half does not reach (V2tr).
int checkWholeBlock(RotBlock b, const GeneralIntegralsDevice& gen,
                    const std::vector<int64_t>& dims, const char* who,
                    std::source_location loc) {
  const RotBlockInfo info = rotBlockInfo(b);
  if (info.rank == 0) return 0;
  if (std::ssize(dims) != info.rank)
    check(false,
          std::format("{}: {} is a rank-{} block, given rank {}", who, rotBlockName(b),
                      info.rank, std::ssize(dims)),
          loc);
  for (int k = 0; k < info.rank; ++k) {
    const RotRole role = info.roles[static_cast<std::size_t>(k)];
    const auto [lo, hi] = spanOf(gen, role.space);
    if (dims[static_cast<std::size_t>(role.axis)] != hi - lo)
      check(false,
            std::format("{}: {} axis {} has extent {}, the sidecar's space has "
                        "{} -- the general action needs whole blocks, not slabs",
                        who, rotBlockName(b), role.axis,
                        dims[static_cast<std::size_t>(role.axis)], hi - lo),
            loc);
  }
  return info.rank;
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
  const int rank = checkWholeBlock(b, gen, dims, "generalBlockAction", loc);
  if (rank == 0) return DeviceTensor();
  DeviceTensor out = DeviceTensor::zeros(dims, res);
  if (rank == 4)
    accumGeneralTwoElectron(b, gen, kappaFor, out, res.stream());
  else
    accumGeneralOneElectron(b, gen, kappaFor, out, res);
  return out;
}

// ITS TRANSPOSE (#240): given `bar`, the adjoint of block `b`'s general-range
// tangent (shaped like the whole block), accumulates d<bar, action(kappa)>/
// dkappa into the column-block adjoints *kbarFor(space) -- (n_mo, |space|)
// each, the same index order as the column blocks the forward reads. The
// ACTIVE one receives every row, including the active rows the forward's
// column block has zeroed: the caller drops those rows when it assembles Y
// (that half is the intra-active action's, transposed on its own). Nothing
// for a block the general half does not reach.
template <class KbarFn>
void generalBlockActionAdjoint(RotBlock b, const GeneralIntegralsDevice& gen, KbarFn&& kbarFor,
                               const DeviceTensor& bar, const DeviceResources& res,
                               std::source_location loc = std::source_location::current()) {
  const int rank = checkWholeBlock(b, gen, bar.dims, "generalBlockActionAdjoint", loc);
  if (rank == 0) return;
  if (rank == 4)
    emitGeneralTwoElectron(b, gen, transposeEmitter(kbarFor, bar, res.stream()));
  else
    emitGeneralOneElectron(b, gen, transposeEmitter(kbarFor, bar, res.stream()), res);
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

// The three terms, once per space, through the emitters above (#240): every
// einsum reads one kappa column block LAST -- here the active one is the FULL
// block, so RotSpace::Act names kA and not the row-dropped kAct. `emitCore`
// receives the core energies' einsums, `emitVirt` the virtual ones'; a space
// with no orbital emits nothing.
template <class EmitCore, class EmitVirt>
void emitEpsResponse(const GeneralIntegralsDevice& gen, const DeviceTensor& gamma,
                     EmitCore&& emitCore, EmitVirt&& emitVirt, const DeviceResources& res) {
  const auto [cLo, cHi] = spanOf(gen, RotSpace::Core);
  const auto [vLo, vHi] = spanOf(gen, RotSpace::Virt);
  if (cHi > cLo) {
    const DeviceTensor fC = sliceAxis(gen.f, 1, cLo, cHi);  // f[x, i]
    // 2 sum_x kappa_xi f_xi
    emitCore("zi,zi->i", KappaOperands{&fC}, RotSpace::Core, 2.0);
    // the core dressing: (ii|xj) = g_xc_cc_j[x, j, i], (i x|j i) = (x i|i j)
    // = g_xc_cc_k[x, i, j]
    emitCore("zmi,zm->i", KappaOperands{&gen.g_xc_cc_j}, RotSpace::Core, 4.0);
    emitCore("zim,zm->i", KappaOperands{&gen.g_xc_cc_k}, RotSpace::Core, -2.0);
    // the active density: (ii|xt) = g_xa_cc[x, t, i, i], (i x|t i) = (x i|i t)
    // = g_xc_ca[x, i, i, t] -- both read on their tied core pair.
    const DeviceTensor cou = deviceEinsumNew("ztii->zti", {&gen.g_xa_cc}, 1.0, res);
    const DeviceTensor exc = deviceEinsumNew("zmmt->zmt", {&gen.g_xc_ca}, 1.0, res);
    emitCore("zti,ut,zu->i", KappaOperands{&cou, &gamma}, RotSpace::Act, 2.0);
    emitCore("zit,ut,zu->i", KappaOperands{&exc, &gamma}, RotSpace::Act, -1.0);
  }
  if (vHi > vLo) {
    const DeviceTensor fV = sliceAxis(gen.f, 1, vLo, vHi);  // f[x, r]
    emitVirt("zr,zr->r", KappaOperands{&fV}, RotSpace::Virt, 2.0);
    // (rr|xj) = g_xc_vv[x, j, r], (r x|j r) = (x r|j r) = g_xv_cv[x, r, j, r]
    const DeviceTensor excC = deviceEinsumNew("zrmr->zrm", {&gen.g_xv_cv}, 1.0, res);
    emitVirt("zmr,zm->r", KappaOperands{&gen.g_xc_vv}, RotSpace::Core, 4.0);
    emitVirt("zrm,zm->r", KappaOperands{&excC}, RotSpace::Core, -2.0);
    // (rr|xt) = g_xa_vv[x, t, r], (r x|t r) = (x r|r t) = g_xv_va[x, r, r, t]
    const DeviceTensor excA = deviceEinsumNew("zrrt->zrt", {&gen.g_xv_va}, 1.0, res);
    emitVirt("ztr,ut,zu->r", KappaOperands{&gen.g_xa_vv, &gamma}, RotSpace::Act, 2.0);
    emitVirt("zrt,ut,zu->r", KappaOperands{&excA, &gamma}, RotSpace::Act, -1.0);
  }
}

ScEpsDot generalEpsResponse(const GeneralIntegralsDevice& gen, const DeviceTensor& gamma,
                            const DeviceTensor& kC, const DeviceTensor& kA,
                            const DeviceTensor& kV, const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  const auto kappaFor = [&](RotSpace s) -> const DeviceTensor* {
    return s == RotSpace::Core ? &kC : s == RotSpace::Act ? &kA : &kV;
  };
  const DeviceTensor dC = DeviceTensor::zeros({gen.ncore}, res);
  const DeviceTensor dV = DeviceTensor::zeros({gen.nvirt}, res);
  emitEpsResponse(gen, gamma, forwardEmitter(kappaFor, dC, stream),
                  forwardEmitter(kappaFor, dV, stream), res);
  ScEpsDot out;
  out.core = gen.ncore > 0 ? downloadTensor(dC, stream) : Tensor({0});
  out.virt = gen.nvirt > 0 ? downloadTensor(dV, stream) : Tensor({0});
  return out;
}

// ITS TRANSPOSE (#240): given the orbital energies' adjoints `barCore` (n_core)
// and `barVirt` (n_virt) -- dE/de, summed over the classes -- accumulates
// d(<barCore, deps_core> + <barVirt, deps_virt>)/dkappa into the column-block
// adjoints *kbarFor(space), the active one being the FULL block's (every row
// counts here, as in the forward).
template <class KbarFn>
void generalEpsResponseAdjoint(const GeneralIntegralsDevice& gen, const DeviceTensor& gamma,
                               KbarFn&& kbarFor, const DeviceTensor& barCore,
                               const DeviceTensor& barVirt, const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  emitEpsResponse(gen, gamma, transposeEmitter(kbarFor, barCore, stream),
                  transposeEmitter(kbarFor, barVirt, stream), res);
}

// --- the generalized Fock's response to gamma, off the diagonal (#184) -------
//
// docs/hessian-theory.md, Sec. 5.5's C5. The orbital energies are the
// generalized Fock's diagonal, f_pp = h^eff_pp + f^gamma[gamma]_pp, with
//
//   f^gamma[g]_xp = sum_ab g_ab [ (xp|ab) - (1/2) (xa|bp) ]
//
// over the WHOLE MO range of x and p -- and generalEpsResponse above reads
// f's off-diagonal columns STORED (its first line, 2 sum_x kappa_xp f_xp), so
// the first-order path never forms f^gamma at all. The mixed derivative
// d^2 e_p / dkappa dgamma does: differentiating that first line along a 1-RDM
// tangent gamma' leaves 2 sum_x kappa_xp f^gamma[gamma']_xp, the FOCK LINE
// below, beside generalEpsResponse's own active-density line read at gamma'
// (the GAMMA LINE: it is linear in gamma, so its derivative is itself). The
// core-density line and h^eff carry no gamma, and drop.
//
// One column block of the generator per orbital space P, as the four-index
// (nmo, |P|, nact, nact) array G_P[x, p, a, b] = (xp|ab) - (1/2)(xa|bp): read
// against a gamma it is f^gamma's column block, and against kappa's columns
// it is the Fock line's coefficient on gamma'. The arrays are the sidecar's
// untied rows (Sec. 5.4's row: g_xc_aa / g_xa_ca, g_xa_aa, g_xv_aa / g_xa_va),
// `g_xP_QR[x, p, q, r] = (x p | q r)`:
//
//   P core:    (xi|ab) = g_xc_aa[x, i, a, b]   (xa|bi) = (xa|ib) = g_xa_ca[x, a, i, b]
//   P active:  (xt|ab) = g_xa_aa[x, t, a, b]   (xa|bt)           = g_xa_aa[x, a, b, t]
//   P virtual: (xr|ab) = g_xv_aa[x, r, a, b]   (xa|br) = (xa|rb) = g_xa_va[x, a, r, b]
DeviceTensor fockGammaColumns(const GeneralIntegralsDevice& gen, RotSpace p,
                              const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  switch (p) {
    case RotSpace::Core: {
      DeviceTensor g = deviceEinsumNew("xpab->xpab", {&gen.g_xc_aa}, 1.0, res);
      deviceEinsumAccum("xapb->xpab", {&gen.g_xa_ca}, g, -0.5, stream);
      return g;
    }
    case RotSpace::Act: {
      DeviceTensor g = deviceEinsumNew("xpab->xpab", {&gen.g_xa_aa}, 1.0, res);
      deviceEinsumAccum("xabp->xpab", {&gen.g_xa_aa}, g, -0.5, stream);
      return g;
    }
    case RotSpace::Virt: {
      DeviceTensor g = deviceEinsumNew("xpab->xpab", {&gen.g_xv_aa}, 1.0, res);
      deviceEinsumAccum("xapb->xpab", {&gen.g_xa_va}, g, -0.5, stream);
      return g;
    }
  }
  return DeviceTensor();
}

// The full-range identity C5's generator carries (Sec. 5.5): h^eff +
// f^gamma[gamma] at the state's own gamma IS the sidecar's f, every element
// of it -- the off-diagonal extension of fockDiagResidual below, which
// rebuilds f's core and virtual diagonals only, and the one check the Fock
// line's arrays get beside the finite difference that sees C5. Assembled on
// the host one column block at a time (the blocks are small at every size a
// sidecar exists for), as a relative Frobenius residual.
double fockFullResidual(const GeneralIntegralsDevice& gen, const DeviceTensor& gamma,
                        const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  const int64_t nmo = gen.nmo();
  Tensor rebuilt = downloadTensor(gen.heff, stream);
  const Tensor f = downloadTensor(gen.f, stream);
  for (const RotSpace p : {RotSpace::Core, RotSpace::Act, RotSpace::Virt}) {
    const auto [lo, hi] = spanOf(gen, p);
    if (hi <= lo) continue;
    const DeviceTensor g = fockGammaColumns(gen, p, res);
    const DeviceTensor cols = deviceEinsumNew("xpab,ab->xp", {&g, &gamma}, 1.0, res);
    const Tensor host = downloadTensor(cols, stream);
    for (int64_t x = 0; x < nmo; ++x)
      for (int64_t q = lo; q < hi; ++q) rebuilt(x, q) += host(x, q - lo);
  }
  double num = 0.0, den = 0.0;
  for (int64_t i = 0; i < f.size(); ++i) {
    const double d = rebuilt.flat(i) - f.flat(i);
    num += d * d;
    den += f.flat(i) * f.flat(i);
  }
  return std::sqrt(num / std::max(den, 1e-300));
}

// The orbital energies' response to gamma, DIFFERENTIATED ALONG kappa: the
// tangent W'[kappa] of nevpt2.gradient_plan's W (e_p = h^eff_pp + <gamma, W_p>),
// so that the mixed second derivative of the orbital energies is
//
//   eps^{wu}_p = d^2 e_p / dkappa dgamma [w, gamma'] = <gamma', W'_p[w]>,
//
//   W'_p[w][a, b] = 2 sum_x w_xp [ (xp|ab) - (1/2)(xa|bp) ]        the Fock line
//                 + 2 sum_x w_xa [ (pp|xb) - (1/2)(px|bp) ]        the gamma line
//
// for p core and virtual (docs/hessian-theory.md, Sec. 5.5's C5). In W's own
// (nact, nact, ncore) / (nact, nact, nvirt) layout, symmetrized over (a, b)
// (symmetrizedResponse) since only a symmetric gamma' ever reads it -- so ONE
// tensor serves both orientations: forward, orbitalEnergyDot(W', gamma'[u])
// is eps^{wu}; reverse, orbitalEnergyDm1Coefficient(W', dE/de) is the dm1
// coefficient d/dw of the CI source term's orbital-energy fold acquires.
// The gamma line is generalEpsResponse's own active-density line with gamma
// left open (the same two tied arrays, the same einsum labels); the Fock line
// is fockGammaColumns against kappa's own columns. `kC`, `kA`, `kV` are
// kappa's FULL column blocks (no rows zeroed), as generalEpsResponse takes
// them. Downloaded, as W is host-side.
OrbitalEnergyResponse generalEpsResponseTangent(const GeneralIntegralsDevice& gen,
                                                const DeviceTensor& kC, const DeviceTensor& kA,
                                                const DeviceTensor& kV,
                                                const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  const int64_t nact = gen.nact;
  OrbitalEnergyResponse out{Tensor({nact, nact, gen.ncore}), Tensor({nact, nact, gen.nvirt})};
  if (gen.ncore > 0) {
    DeviceTensor d = DeviceTensor::zeros({nact, nact, gen.ncore}, res);
    const DeviceTensor g = fockGammaColumns(gen, RotSpace::Core, res);
    deviceEinsumAccum("zi,ziab->abi", {&kC, &g}, d, 2.0, stream);
    // (ii|xb) = g_xa_cc[x, b, i, i], (ix|bi) = (xi|ib) = g_xc_ca[x, i, i, b]
    const DeviceTensor cou = deviceEinsumNew("ztii->zti", {&gen.g_xa_cc}, 1.0, res);
    const DeviceTensor exc = deviceEinsumNew("zmmt->zmt", {&gen.g_xc_ca}, 1.0, res);
    deviceEinsumAccum("zu,zti->uti", {&kA, &cou}, d, 2.0, stream);
    deviceEinsumAccum("zu,zit->uti", {&kA, &exc}, d, -1.0, stream);
    out.core = downloadTensor(d, stream);
  }
  if (gen.nvirt > 0) {
    DeviceTensor d = DeviceTensor::zeros({nact, nact, gen.nvirt}, res);
    const DeviceTensor g = fockGammaColumns(gen, RotSpace::Virt, res);
    deviceEinsumAccum("zr,zrab->abr", {&kV, &g}, d, 2.0, stream);
    // (rr|xb) = g_xa_vv[x, b, r], (rx|br) = (xr|rb) = g_xv_va[x, r, r, b]
    const DeviceTensor excA = deviceEinsumNew("zrrt->zrt", {&gen.g_xv_va}, 1.0, res);
    deviceEinsumAccum("zu,ztr->utr", {&kA, &gen.g_xa_vv}, d, 2.0, stream);
    deviceEinsumAccum("zu,zrt->utr", {&kA, &excA}, d, -1.0, stream);
    out.virt = downloadTensor(d, stream);
  }
  return symmetrizedResponse(out);
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

// --- the stored blocks OUT OF the inventory (#220) ---------------------------
//
// The same restrictions read the other way round: the stored block a rule's
// source array restricts to, BUILT from the inventory instead of held against
// it. For a two-electron block that is the slot-p rule's array on the
// promoted slot's own space; a one-electron block is h^eff's slice (less the
// active trace T_rt = sum_b (r b | b t) for h1e_v_Sr, read off g_xa_aa); and
// an orbital-energy vector is f's diagonal on its space. Every one is a
// slice of a slice -- linear in the inventory -- so handed a DOTTED
// inventory (an integral perturbation's) it returns the stored block's own
// dot, which is how an integral perturbation sidecar carries no block of the
// golden's: the reverse of what scGeneralIntegralCheck holds, on the same
// tables, so a block it would pass is the block this builds.
DeviceTensor storedBlockFrom(RotBlock b, const GeneralIntegralsDevice& gen,
                             const DeviceResources& res) {
  const RotBlockInfo info = rotBlockInfo(b);
  if (info.rank == 4) {
    const PromotionPlan pl = promotionPlan(b, 0);
    const auto [lo, hi] = spanOf(gen, pl.promoted.space);
    const DeviceTensor rows = sliceAxis(generalArray(gen, pl.pick.array), 0, lo, hi);
    std::string subs(1, kAxisLabels[static_cast<std::size_t>(pl.promoted.axis)]);
    subs += kAxisLabels[static_cast<std::size_t>(pl.partner.axis)];
    subs += kAxisLabels[static_cast<std::size_t>(pl.firstAxis)];
    subs += kAxisLabels[static_cast<std::size_t>(pl.lastAxis)];
    subs += "->";
    subs += kAxisLabels;
    return deviceEinsumNew(subs, {&rows}, 1.0, res);
  }
  check(info.rank == 2, std::format("storedBlockFrom: {} is not a stored block", rotBlockName(b)));
  const auto [pLo, pHi] = spanOf(gen, info.roles[0].space);
  const auto [qLo, qHi] = spanOf(gen, info.roles[1].space);
  const DeviceTensor rows = sliceAxis(sliceAxis(gen.heff, 0, pLo, pHi), 1, qLo, qHi);
  DeviceTensor out = deviceEinsumNew("ij->ij", {&rows}, 1.0, res);
  if (info.trace) {
    const DeviceTensor tTr = deviceEinsumNew("zmmj->zj", {&gen.g_xa_aa}, 1.0, res);
    const DeviceTensor tTrRows = sliceAxis(tTr, 0, pLo, pHi);  // (v, a)
    deviceEinsumAccum("ij->ij", {&tTrRows}, out, -1.0, res.stream());
  }
  return out;
}

// f's diagonal on the core (`core`) or the virtual space: e_core / e_virt,
// as (O2) says the golden's are.
DeviceTensor storedEpsFrom(bool core, const GeneralIntegralsDevice& gen,
                           const DeviceResources& res) {
  const auto [lo, hi] = spanOf(gen, core ? RotSpace::Core : RotSpace::Virt);
  if (hi <= lo) return DeviceTensor::zeros({0}, res);
  const DeviceTensor fPP = sliceAxis(sliceAxis(gen.f, 0, lo, hi), 1, lo, hi);
  return deviceEinsumNew("ii->i", {&fPP}, 1.0, res);
}

// --- and the same two maps TRANSPOSED (#246) ---------------------------------
//
// The integral adjoint's pullback onto the inventory: `bar` is dE/d(stored
// block), and what lands in `out` (an inventory of zeros-or-partial sums,
// shaped as the sidecar's) is d<bar, storedBlockFrom(b, I)>/dI -- the slice
// read the other way. Every write lands in an axis-0 slice of an inventory
// array, which is contiguous, so the einsum's row-major output is honoured;
// a column restriction is a selector matrix's contraction. `inventoryAdjoint
// Pairing` against `integralAdjointPairing` is the duality that holds it.

// (n, n_mo) with E[j, lo + j] = 1: column j of a block placed at lo.
DeviceTensor columnSelector(int64_t n, int64_t lo, int64_t nmo, const DeviceResources& res) {
  Tensor e({n, nmo});
  for (int64_t j = 0; j < n; ++j) e(j, lo + j) = 1.0;
  return uploadTensor(e, res);
}

void storedBlockAdjoint(RotBlock b, const DeviceTensor& bar, const GeneralIntegralsDevice& out,
                        const DeviceResources& res) {
  const RotBlockInfo info = rotBlockInfo(b);
  if (info.rank == 4) {
    const PromotionPlan pl = promotionPlan(b, 0);
    const auto [lo, hi] = spanOf(out, pl.promoted.space);
    const DeviceTensor rows = sliceAxis(generalArray(out, pl.pick.array), 0, lo, hi);
    std::string subs(kAxisLabels);
    subs += "->";
    subs += kAxisLabels[static_cast<std::size_t>(pl.promoted.axis)];
    subs += kAxisLabels[static_cast<std::size_t>(pl.partner.axis)];
    subs += kAxisLabels[static_cast<std::size_t>(pl.firstAxis)];
    subs += kAxisLabels[static_cast<std::size_t>(pl.lastAxis)];
    deviceEinsumAccum(subs, {&bar}, rows, 1.0, res.stream());
    return;
  }
  check(info.rank == 2,
        std::format("storedBlockAdjoint: {} is not a stored block", rotBlockName(b)));
  const auto [pLo, pHi] = spanOf(out, info.roles[0].space);
  const auto [qLo, qHi] = spanOf(out, info.roles[1].space);
  if (pHi <= pLo || qHi <= qLo) return;
  const DeviceTensor sel = columnSelector(qHi - qLo, qLo, out.nmo(), res);
  const DeviceTensor heffRows = sliceAxis(out.heff, 0, pLo, pHi);
  deviceEinsumAccum("ij,jx->ix", {&bar, &sel}, heffRows, 1.0, res.stream());
  if (info.trace) {
    // out -= sum_m g_xa_aa[z, m, m, j] on rows z of the p span.
    const DeviceTensor eye = deviceEye(out.nact, res);
    const DeviceTensor tRows = sliceAxis(out.g_xa_aa, 0, pLo, pHi);
    deviceEinsumAccum("zj,mn->zmnj", {&bar, &eye}, tRows, -1.0, res.stream());
  }
}

void storedEpsAdjoint(bool core, const DeviceTensor& bar, const GeneralIntegralsDevice& out,
                      const DeviceResources& res) {
  const auto [lo, hi] = spanOf(out, core ? RotSpace::Core : RotSpace::Virt);
  if (hi <= lo) return;
  const DeviceTensor sel = columnSelector(hi - lo, lo, out.nmo(), res);
  const DeviceTensor fRows = sliceAxis(out.f, 0, lo, hi);
  deviceEinsumAccum("i,ix->ix", {&bar, &sel}, fRows, 1.0, res.stream());
}

}  // namespace nevpt2
