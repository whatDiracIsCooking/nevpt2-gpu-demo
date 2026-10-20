// nevpt2.gradient:terms -- the generic machinery every SC class's
// pseudodensity assembly is written in: one bilinear term of a class's norm
// or Dyall-Hamiltonian expectation value (ScTerm), evaluating a list of them
// into the per-tuple vectors the multipliers need, differentiating the same
// list with respect to each of its operands, and contracting a pseudodensity
// back against the operand it is conjugate to.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :terms;` and from no importer of nevpt2.gradient.
// (Used by gradient_sc_classes.cppm and gradient.cpp.)
//
// Every einsum here is issued on res.stream() and every tensor it allocates
// is drawn from res's pool, freed when it goes out of scope -- the slab's
// pseudodensities at the end of the slab's iteration, a class's M / E when
// the class returns.
module nevpt2.gradient:terms;

import std;
import nevpt2.gradient;
// The kappa directions, the device action that forms a tensor's tangent along
// one, and the ScGDot pairing of an active block with its own tangent: the
// rotation machinery assembleSlab pairs and differentiates along.
import :rotation;
// The general-range action a slab's tangent adds when a direction came with a
// sidecar (read off nevpt2.gradient_plan's promotion table), and
// addTensorInto.
import :general;

namespace nevpt2 {

// An ScTerm operand: a tensor and its role, one of the class's external
// integral blocks (the index into the class's block list) or the active-space
// block (nevpt2.gradient_plan's kScActive) the term contracts them against.
struct ScOperand {
  const DeviceTensor* t = nullptr;
  int block = kScActive;
};

// One term of a class's norm or H: `coef * einsum(subs, ops...)` accumulated
// into the class's per-tuple vector, whose indices are `subs`' output labels.
// Copied verbatim from the matching deviceEinsum* call in nevpt2.energy's
// energy_<class> -- same subscripts, same coefficient, same operands.
struct ScTerm {
  std::string subs;
  std::vector<ScOperand> ops;
  double coef = 1.0;
};

// termHasActive -- whether a term has an active operand -- is
// nevpt2.gradient_plan's, where the unit tier pins it card-free.

// Which terms of a list a pass covers. A class's per-tuple norm (or H) is the
// sum of both passes; only the WithActive half has an Eqs. 42-43 counterpart,
// so the two are accumulated separately and added.
enum class ScTermFilter { WithActive, WithoutActive };

// Accumulates `sum coef * einsum(term)` over the matching terms into `out`
// (shaped as the class's per-tuple vector).
void evalTerms(const std::vector<ScTerm>& terms, const DeviceTensor& out, ScTermFilter filter,
               wwrStream_t stream) {
  for (const ScTerm& term : terms) {
    if ((filter == ScTermFilter::WithActive) != termHasActive(term)) continue;
    std::vector<const DeviceTensor*> ops;
    ops.reserve(term.ops.size());
    for (const ScOperand& o : term.ops) ops.push_back(o.t);
    deviceEinsumAccum(term.subs, ops, out, term.coef, stream);
  }
}

// <x, d>: the elementwise product of two equal-shaped tensors, summed. Issued
// as an einsum reduced to x's LEADING axis -- not to a scalar -- and finished
// on the host, so the kernel never writes a rank-0 output.
double dotTensors(const DeviceTensor& x, const DeviceTensor& d, const DeviceResources& res,
                  std::source_location loc = std::source_location::current()) {
  if (x.dims != d.dims || x.rank() < 1)
    check(false, "dotTensors: operands must have the same shape, of rank >= 1", loc);
  static constexpr std::string_view kLabels = "abcdef";
  const std::string lab(kLabels.substr(0, static_cast<std::size_t>(x.rank())));
  const DeviceTensor reduced =
      deviceEinsumNew(lab + "," + lab + "->" + lab.substr(0, 1), {&x, &d}, 1.0, res);
  const Tensor host = downloadTensor(reduced, res.stream());
  double sum = 0.0;
  for (const double v : host.data()) sum += v;
  return sum;
}

// One active-space pseudodensity under assembly (Park Eqs. 42-43's M / E),
// keyed by the metric or Dyall-Hamiltonian block it is conjugate to.
struct ScActiveAccum {
  const DeviceTensor* g = nullptr;
  DeviceTensor d;
};

// The accumulator for `g`, created on first use (shaped like `g` -- its
// conjugate). A class contracts each of its blocks in several terms, and a
// block that appeared on the norm side and on the H side would share one
// accumulator, which is what the Eqs. 42-43 identity sums over anyway.
DeviceTensor& activeAccumFor(std::vector<ScActiveAccum>& accums, const DeviceTensor* g,
                             const DeviceResources& res) {
  for (ScActiveAccum& a : accums)
    if (a.g == g) return a.d;
  accums.push_back(ScActiveAccum{g, DeviceTensor::zeros(g->dims, res)});
  return accums.back().d;
}

// Accumulates the multiplier-weighted derivative of every term with respect
// to EVERY one of its operands: an external one into dBlocks[its block
// index] (Eq. 41's D), an active one into its ScActiveAccum (Eqs. 42-43's M
// or E). `mult` is the class's per-tuple multiplier -- P for a norm term
// list, Q for an H one -- shaped as the terms' output labels.
//
// The derivative subscripts are REWRITTEN from the term's own
// (nevpt2.sc_amplitudes' derivativeSubscripts), never hand-written: each is
// the same contraction with one operand replaced by `mult` and the term's
// output labels and the dropped operand's swapping roles.
//
// `dBlocks` null skips the external slots altogether (accumActiveDerivatives
// below): the reverse-over-forward walk wants only the M_G tangents.
void accumDerivativesInto(const std::vector<ScTerm>& terms, const DeviceTensor& mult,
                          std::vector<DeviceTensor>* dBlocks,
                          std::vector<ScActiveAccum>& dActive, const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  for (const ScTerm& term : terms) {
    for (std::size_t k = 0; k < term.ops.size(); ++k) {
      if (dBlocks == nullptr && term.ops[k].block != kScActive) continue;
      const std::string subs = derivativeSubscripts(term.subs, static_cast<int>(k));
      std::vector<const DeviceTensor*> ops;
      ops.reserve(term.ops.size());
      for (std::size_t j = 0; j < term.ops.size(); ++j)
        if (j != k) ops.push_back(term.ops[j].t);
      ops.push_back(&mult);
      if (term.ops[k].block == kScActive) {
        const DeviceTensor& out = activeAccumFor(dActive, term.ops[k].t, res);
        deviceEinsumAccum(subs, ops, out, term.coef, stream);
      } else {
        deviceEinsumAccum(subs, ops, (*dBlocks)[static_cast<std::size_t>(term.ops[k].block)],
                          term.coef, stream);
      }
    }
  }
}

void accumDerivatives(const std::vector<ScTerm>& terms, const DeviceTensor& mult,
                      std::vector<DeviceTensor>& dBlocks,
                      std::vector<ScActiveAccum>& dActive, const DeviceResources& res) {
  accumDerivativesInto(terms, mult, &dBlocks, dActive, res);
}

// The active slots alone: Eqs. 42-43's M / E weighted by `mult`, no D.
void accumActiveDerivatives(const std::vector<ScTerm>& terms, const DeviceTensor& mult,
                            std::vector<ScActiveAccum>& dActive, const DeviceResources& res) {
  accumDerivativesInto(terms, mult, nullptr, dActive, res);
}

// --- the walk's tangent along a CI direction (#156) ---------------------------
//
// Reverse-over-forward (docs/hessian-theory.md, Sec. 4.2). Every active
// accumulator is M_G = sum_t mult_t * dTerm_t/dG, and in every term the
// active block G is the ONLY operand that moves with the CI vector -- the
// other two are external integrals, constant at fixed orbitals -- so
// dTerm_t/dG is constant and
//
//   Mdot_G = the same accumulation with mult_t replaced by its tangent,
//
// Pdot / Qdot of nevpt2.gradient_plan's *MultiplierTangents (the 3x3 against
// this slab's Ndot, Hdot). Those per-tuple tangents are the SAME term lists'
// product rule with each active block replaced by its tangent along u -- the
// same evalTermsTangent the rotation's forward tangent runs, with no external
// block moving. A class carrying a CI direction holds one of these: the
// tangents of its active blocks, keyed by the base block each belongs to
// (ScGDot, which may VIEW a caller's tensor -- a raw RDM's tangent), and the
// Mdot_G accumulators under the same keys, so the class's own RDM sweep reads
// them exactly as it reads M_G.
struct ScCiWalk {
  std::vector<ScGDot> gDots;
  std::vector<ScActiveAccum> dActive;
};

// The FORWARD TANGENT of a term list: the product rule, each operand
// replaced in turn by its own tangent, with the term's OWN subscripts (no
// rewriting -- that is the adjoint's business, accumDerivatives above). An
// operand with no tangent does not move along the direction and is skipped:
// a purely RDM-built metric (absent from `gDots`), and a block with no active
// index at all (a nullptr in `blockDots` -- Sir's h1e_v and its hoisted
// diagonal trace, whose active pair is contracted away and so cannot move;
// see gradient.cppm, "The active-rotation source term").
//
// `blockDots` is indexed as the class's block list, `gDots` keyed by the base
// active block each tangent belongs to.
void evalTermsTangent(const std::vector<ScTerm>& terms,
                      const std::vector<const DeviceTensor*>& blockDots,
                      const std::vector<ScGDot>& gDots, const DeviceTensor& out,
                      wwrStream_t stream) {
  for (const ScTerm& term : terms) {
    for (std::size_t k = 0; k < term.ops.size(); ++k) {
      const ScOperand& op = term.ops[k];
      const DeviceTensor* dot = op.block == kScActive
                                    ? gDotOf(gDots, op.t)
                                    : blockDots[static_cast<std::size_t>(op.block)];
      if (dot == nullptr) continue;
      std::vector<const DeviceTensor*> ops;
      ops.reserve(term.ops.size());
      for (std::size_t j = 0; j < term.ops.size(); ++j)
        ops.push_back(j == k ? dot : term.ops[j].t);
      deviceEinsumAccum(term.subs, ops, out, term.coef, stream);
    }
  }
}

// --- the mixed block H_kc: an orbital direction w beside a CI direction u (#184) ---
//
// docs/hessian-theory.md, Sec. 5.5. Along w the external blocks AND the
// Dyall blocks move (the rotation's slab dots and tangent blocks); along u
// only the active blocks do (ScCiWalk's). So a term x . G . x has, at second
// order in (w, u), exactly two kinds of piece:
//
//   * an external slot along w and the active block along u -- C2, the
//     product rule's cross terms (evalTermsMixed below), and
//   * the active block's own mixed tangent G^{wu}: every make_* at the dotted
//     integrals AND the tangent RDMs (C3), and at the digests' mixed tangent
//     f3^{wu} (C4) -- evalTermsTangent with those blocks in place.
//
// A class carrying the mixed direction holds one of these. FORWARD (the
// scalar Eddot_kc(w, u), orientation F) it holds u's tangent blocks and the
// two mixed sets; REVERSE (the vector H_kc^T w, orientation R1) it holds the
// Mdot_G accumulators along w, which the class's own RDM sweep reads as it
// reads M_G. Both hold `wDigest`, the digests' share of w's tangent blocks on
// its own (the f3-reading blocks at f3'[w] alone), which is C1's digest part.
// The orbital direction itself is rot.dirs[0], tangent blocks built and the
// digests' share merged (ScRotationDir::f3acDot).
struct ScMixedWalk {
  bool forward = false;
  bool reverse = false;
  // Whether the orbital energies respond to kappa (Deltadot[w] from deps) and
  // to the CI vector (Deltadot[u], and with both, C5).
  bool epsKappa = false;
  bool epsCi = false;
  std::vector<ScGDot> wDigest;   // the digests' share of G'[w] alone
  // forward
  std::vector<ScGDot> uDots;     // G'[u]: ScCiWalk's tangent blocks
  std::vector<ScGDot> wuInt;     // C3: G at (h1e'[w], h2e'[w]) and the tangent RDMs
  std::vector<ScGDot> wuDigest;  // C4: the f3-reading blocks at f3^{wu} alone
  const Tensor* epsCoreU = nullptr;   // e'[u], zero unless epsCi
  const Tensor* epsVirtU = nullptr;
  const Tensor* epsCoreWU = nullptr;  // eps^{wu}, zero unless both
  const Tensor* epsVirtWU = nullptr;
  // reverse
  std::vector<ScActiveAccum> dActive;  // Mdot_G along w
};

// The SECOND-ORDER cross terms of a term list: every term with one EXTERNAL
// operand replaced by its tangent along w (`blockDotsW`, as slabDots hands
// it) and its ACTIVE operand by its tangent along u (`gDotsU`), summed over
// the external slots -- C2. The pairs a CI direction cannot reach (two
// external slots, or the active block along both) are not here: an external
// block does not move with the CI vector, and the active block's own mixed
// tangent is a block of its own (ScMixedWalk::wuInt / wuDigest).
void evalTermsMixed(const std::vector<ScTerm>& terms,
                    const std::vector<const DeviceTensor*>& blockDotsW,
                    const std::vector<ScGDot>& gDotsU, const DeviceTensor& out,
                    wwrStream_t stream) {
  for (const ScTerm& term : terms) {
    for (std::size_t j = 0; j < term.ops.size(); ++j) {
      if (term.ops[j].block != kScActive) continue;
      const DeviceTensor* dotU = gDotOf(gDotsU, term.ops[j].t);
      if (dotU == nullptr) continue;
      for (std::size_t k = 0; k < term.ops.size(); ++k) {
        if (k == j || term.ops[k].block == kScActive) continue;
        const DeviceTensor* dotW = blockDotsW[static_cast<std::size_t>(term.ops[k].block)];
        if (dotW == nullptr) continue;
        std::vector<const DeviceTensor*> ops;
        ops.reserve(term.ops.size());
        for (std::size_t i = 0; i < term.ops.size(); ++i)
          ops.push_back(i == j ? dotU : (i == k ? dotW : term.ops[i].t));
        deviceEinsumAccum(term.subs, ops, out, term.coef, stream);
      }
    }
  }
}

// The Eqs. 42-43 accumulators' EXTERNAL-SLOT tangent along w: M_G =
// sum_t mult_t dTerm_t/dG, and dTerm_t/dG carries the term's external
// operands, so along w each of them is replaced in turn by its tangent
// (`blockDots`) with `mult` -- the multiplier itself, not its tangent -- held.
// The other half of Mdot_G, the multiplier's tangent against the plain
// derivative, is accumActiveDerivatives at Pdot / Qdot. Together they are
// the walk's tangent along w (R1's gdot, docs/hessian-theory.md Sec. 5.5).
void accumActiveDerivativesAlong(const std::vector<ScTerm>& terms, const DeviceTensor& mult,
                                 const std::vector<const DeviceTensor*>& blockDots,
                                 std::vector<ScActiveAccum>& dActive,
                                 const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  for (const ScTerm& term : terms) {
    for (std::size_t k = 0; k < term.ops.size(); ++k) {
      if (term.ops[k].block != kScActive) continue;
      const std::string subs = derivativeSubscripts(term.subs, static_cast<int>(k));
      for (std::size_t j = 0; j < term.ops.size(); ++j) {
        if (j == k || term.ops[j].block == kScActive) continue;
        const DeviceTensor* dot = blockDots[static_cast<std::size_t>(term.ops[j].block)];
        if (dot == nullptr) continue;
        std::vector<const DeviceTensor*> ops;
        ops.reserve(term.ops.size());
        for (std::size_t i = 0; i < term.ops.size(); ++i)
          if (i != k) ops.push_back(i == j ? dot : term.ops[i].t);
        ops.push_back(&mult);
        deviceEinsumAccum(subs, ops, activeAccumFor(dActive, term.ops[k].t, res), term.coef,
                          stream);
      }
    }
  }
}

// One per-tuple table of a term list's tangent, on the host: evalTermsTangent
// into a fresh zero vector of the tuple shape, read back (downloadTensor
// synchronizes the stream the einsums were issued on).
Tensor tangentTable(const std::vector<ScTerm>& terms,
                    const std::vector<const DeviceTensor*>& blockDots,
                    const std::vector<ScGDot>& gDots, const std::vector<int64_t>& tupleDims,
                    const DeviceResources& res) {
  const DeviceTensor out = DeviceTensor::zeros(tupleDims, res);
  evalTermsTangent(terms, blockDots, gDots, out, res.stream());
  return downloadTensor(out, res.stream());
}

// The same for evalTermsMixed (C2).
Tensor mixedTable(const std::vector<ScTerm>& terms,
                  const std::vector<const DeviceTensor*>& blockDotsW,
                  const std::vector<ScGDot>& gDotsU, const std::vector<int64_t>& tupleDims,
                  const DeviceResources& res) {
  const DeviceTensor out = DeviceTensor::zeros(tupleDims, res);
  evalTermsMixed(terms, blockDotsW, gDotsU, out, res.stream());
  return downloadTensor(out, res.stream());
}

// One slab's blocks dotted along ONE direction: the tangent of each block
// something in the direction moves, and a nullptr for each block nothing
// does -- which is what the product rule skips. The tensors own their memory
// and are freed with the returned value, so a direction's slab tangents do
// not outlive its iteration.
//
// Each tangent is the two halves of the chain rule added: the INTRA-ACTIVE
// action on the stored block (its active axes, from `kinds[i]`), and -- for a
// direction built with a sidecar (`gen`) -- the GENERAL part, every slot's
// promotion whose index leaves its own space (:general). With an
// active-restricted kappa the second is an exactly-zero tensor, so the sum is
// bit for bit the first.
struct ScSlabDots {
  std::vector<DeviceTensor> owned;
  std::vector<const DeviceTensor*> ptr;  // &owned[i], or nullptr
};

ScSlabDots slabDots(const std::vector<const DeviceTensor*>& blocks,
                    const std::vector<RotBlock>& kinds, const ScRotationDir& dir,
                    const GeneralIntegralsDevice* gen, const DeviceResources& res,
                    std::source_location loc = std::source_location::current()) {
  if (kinds.size() != blocks.size()) check(false, "slabDots: one kind per block", loc);
  ScSlabDots out;
  out.owned.resize(blocks.size());
  out.ptr.assign(blocks.size(), nullptr);
  // A dotted input set (#154): every block's dot is the set's own field, a
  // view, with no promotion run. The fields are whole blocks, so the slab
  // must be one too -- anything else is our bug (only the integral-direct
  // entry points build such a direction).
  if (dir.dotted != nullptr) {
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      const DeviceTensor& src =
          kinds[i] == RotBlock::V2tr ? dir.dotted->v2tr : dottedField(dir.dotted->d, kinds[i]);
      if (src.dims != blocks[i]->dims)
        check(false,
              std::format("slabDots: the dotted set's {} is not shaped like the slab -- a dotted "
                          "direction needs one slab per block",
                          rotBlockName(kinds[i])),
              loc);
      out.owned[i] = src.view();
      out.ptr[i] = &out.owned[i];
    }
    return out;
  }
  // Each block's GENERAL part, kept alive until the walk over `blocks` is
  // done: V2tr's tangent is the trace of Sir2's.
  std::vector<DeviceTensor> general(blocks.size());
  const auto columns = [&dir](RotSpace s) { return dir.columns(s); };
  for (std::size_t i = 0; i < blocks.size(); ++i)
    if (gen != nullptr && kinds[i] != RotBlock::V2tr)
      general[i] = generalBlockAction(kinds[i], *gen, columns, blocks[i]->dims, res);

  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const std::vector<int> axes = activeAxesOf(kinds[i]);
    if (kinds[i] == RotBlock::V2tr) {
      // v2tr[r, i] = sum_a h2e_v2_Sir[r, a, a, i], so its tangent is the
      // trace of that block's. The INTRA-ACTIVE half of that trace cancels
      // exactly -- the two active promotions map onto each other under the
      // traced pair, against an antisymmetric kappa -- so what is left is
      // the trace of the general half, and it is exactly zero (not zero to
      // rounding) for an active-restricted direction.
      if (gen == nullptr) continue;
      std::size_t src = blocks.size();
      for (std::size_t j = 0; j < kinds.size(); ++j)
        if (kinds[j] == RotBlock::Sir2) src = j;
      if (src == blocks.size())
        check(false, "slabDots: v2tr's tangent needs the Sir2 block beside it", loc);
      out.owned[i] = deviceEinsumNew("raai->ri", {&general[src]}, 1.0, res);
      out.ptr[i] = &out.owned[i];
      continue;
    }
    if (axes.empty() && general[i].rank() == 0) continue;  // nothing moves it
    out.owned[i] = axes.empty() ? DeviceTensor::zeros(blocks[i]->dims, res)
                                : kappaActionDevice(*blocks[i], axes, dir.kappa, res);
    addTensorInto(general[i], out.owned[i], res.stream());
    out.ptr[i] = &out.owned[i];
  }
  return out;
}

// The EXTERNAL half of the adjoint pairing for one slab and one direction:
// sum_blocks <D, Xdot>, Eq. 41's pseudodensity against the direction's action
// on the very block it is conjugate to.
double pairSlabDots(const ScSlabDots& dots, const std::vector<DeviceTensor>& dBlocks,
                    const DeviceResources& res,
                    std::source_location loc = std::source_location::current()) {
  if (dots.ptr.size() != dBlocks.size())
    check(false, "pairSlabDots: one pseudodensity per block", loc);
  double ext = 0.0;
  for (std::size_t i = 0; i < dBlocks.size(); ++i)
    if (dots.ptr[i] != nullptr) ext += dotTensors(dBlocks[i], *dots.ptr[i], res);
  return ext;
}

// ScSlabMultipliers (one slab's per-tuple P and Q, its perturber-order R and
// its (norm, energy)), sliceHost and plainMultipliers are
// nevpt2.gradient_plan's, where the unit tier pins them card-free.

// The per-slab core every class but Sijrs and Srsi runs:
//
//   1. evaluate the class's norm and H term lists into per-tuple vectors
//      (splitting off the terms with no active block, which Eqs. 42-43 cannot
//      see);
//   2. turn them into the per-tuple multipliers P and Q (`multiplierFn`);
//   3. assemble this slab's pseudodensities -- Eq. 41's D for each external
//      block, Eqs. 42-43's M / E for each active one;
//   4. contract the D family back against its own slab, accumulating Eq. 40's
//      sum into `acc`.
//
// `blocks` lists the class's external integral blocks, in the order ScTerm's
// `block` indices name them; a D accumulator is allocated per block, shaped
// like it, and freed when this returns -- D is a per-slab object.
//
// `dActive` is the CALLER's: the M / E of Eqs. 42-43 have no external index
// left, so they accumulate over the whole class's walk, and the caller
// contracts them back (and frees them) once the walk is done --
// activeContraction() below.
//
// `rot` adds the ROTATION work, and nothing happens for a caller with no
// directions. Per direction: this slab's own blocks dotted along it (`kinds`
// names each block, in `blocks`' order, so :general's one table supplies both
// its active axes and its promotion rule), the EXTERNAL half of the adjoint
// pairing <D, Xdot> accumulated into acc.rotExt, and -- for a direction that
// asked for it -- the forward tangent of the same terms under the product
// rule, finished with THIS slab's own multipliers as Park Eqs. 36-37
// prescribe, P.Ndot + Q.Hdot. `gDots[d]` is the class's active-block tangents
// along direction d (rotationTangentBlocks), and is empty for a direction
// that wants the adjoint alone.
//
// `deltaOf(e_core, e_virt)` is the class's own `delta` construction as a
// FUNCTION of the orbital-energy vectors -- `multiplierFn` reads it once at
// the real ones, and a general direction reads it again at the response
// `deps` (Park Eq. 38's multiplier R against Deltadot, the orbital-energy
// half of the source term). That works because Delta_t is a signed sum of
// orbital energies, linear and homogeneous in them, so its tangent is the
// same formula at the dotted vectors -- the same reason a Dyall block's
// tangent is its own make_* at the dotted integrals.
//
// `ci` non-null adds the walk's TANGENT along a CI direction (#156, ScCiWalk
// above): this slab's per-tuple Ndot / Hdot from ci->gDots, Edot(u) =
// P.Ndot + Q.Hdot into acc.ciTangent (the free check that the tangent blocks
// are energiesTangentDevice's), and the multipliers' tangents from
// `multiplierDotFn(norm, h, normDot, hDot)` -- the class's own *MultiplierTangents,
// at fixed orbital energies -- accumulated into ci->dActive. Null runs none of
// it. `multiplierDotFn(norm, h, normDot, hDot, diffDot)` takes the Delta
// tangent in perturber order (zeros along a CI direction: the walk holds the
// orbital energies fixed).
//
// `mix` non-null adds the MIXED direction (#184, ScMixedWalk above): w is
// rot.dirs[0]. Every slab forms its per-tuple tangents along w (the digests'
// share merged, and on its own), the orbital energies' Delta tangent along w
// when they respond to kappa, and acc.mixedEdotW = P.Ndot + Q.Hdot + R.Ddot
// along w -- the free check that the merge is #88's term. Forward, it adds
// u's tangents and the four mixed tables and finishes them with
// `mixedFn(tables, diffDotW, diffDotU, diffWU)` -- plainMixedTerms or the
// class's packed one -- into acc.mixed. Reverse, it accumulates Mdot_G along
// w into mix->dActive (the multipliers' tangents against the plain
// derivative, plus the multipliers against the external slots' tangents) and,
// with the orbital energies responding to the CI vector, Rdot into
// acc.dEdCoreDot / dEdVirtDot. Null runs none of it, and `mixedFn` is never
// called.
template <class MultiplierFn, class DeltaFn, class MultiplierDotFn, class MixedFn>
void assembleSlab(const std::vector<ScTerm>& normTerms, const std::vector<ScTerm>& hTerms,
                  const std::vector<int64_t>& tupleDims,
                  const std::vector<const DeviceTensor*>& blocks,
                  std::vector<ScActiveAccum>& dActive, MultiplierFn&& multiplierFn,
                  DeltaFn&& deltaOf, const ScRotation& rot,
                  const std::vector<RotBlock>& kinds,
                  const std::vector<std::vector<ScGDot>>& gDots, ScCiWalk* ci,
                  MultiplierDotFn&& multiplierDotFn, ScMixedWalk* mix, MixedFn&& mixedFn,
                  ScClassGradient& acc, const DeviceResources& res) {
  const wwrStream_t stream = res.stream();

  // The with-active halves kept apart, so actReference can exclude the terms
  // Eqs. 42-43 have no counterpart for, then added in for the multipliers.
  DeviceTensor normActive = DeviceTensor::zeros(tupleDims, res);
  DeviceTensor hActive = DeviceTensor::zeros(tupleDims, res);
  DeviceTensor norm = DeviceTensor::zeros(tupleDims, res);
  DeviceTensor h = DeviceTensor::zeros(tupleDims, res);
  evalTerms(normTerms, normActive, ScTermFilter::WithActive, stream);
  evalTerms(hTerms, hActive, ScTermFilter::WithActive, stream);
  evalTerms(normTerms, norm, ScTermFilter::WithoutActive, stream);
  evalTerms(hTerms, h, ScTermFilter::WithoutActive, stream);
  addTensorInto(normActive, norm, stream);
  addTensorInto(hActive, h, stream);

  // No sync before the downloads: each is enqueued on the same stream as the
  // einsums that wrote its source, behind them, and downloadTensor
  // synchronizes that stream before the host reads (as nevpt2.energy's
  // per-slab finish does).
  const Tensor hostNorm = downloadTensor(norm, stream);
  const Tensor hostH = downloadTensor(h, stream);
  const Tensor hostNormActive = downloadTensor(normActive, stream);
  const Tensor hostHActive = downloadTensor(hActive, stream);

  const ScSlabMultipliers m = multiplierFn(hostNorm, hostH);
  acc.norm += m.norm;
  acc.energy += m.energy;
  // dE/de: this slab's R_t scattered onto the orbital energies its Delta_t
  // reads, through the same `deltaOf` (#87; the CI source term's orbital-energy
  // response).
  if (acc.epsAdjoint) deltaAdjointInto(deltaOf, m.r, acc.dEdCore, acc.dEdVirt);
  acc.actReference +=
      dotHost(m.p.data(), hostNormActive.data()) + dotHost(m.q.data(), hostHActive.data());

  const DeviceTensor mp = uploadTensor(m.p, res);
  const DeviceTensor mq = uploadTensor(m.q, res);
  std::vector<DeviceTensor> dBlocks;
  dBlocks.reserve(blocks.size());
  for (const DeviceTensor* b : blocks) dBlocks.push_back(DeviceTensor::zeros(b->dims, res));
  accumDerivatives(normTerms, mp, dBlocks, dActive, res);
  accumDerivatives(hTerms, mq, dBlocks, dActive, res);

  double ext = 0.0;
  for (std::size_t i = 0; i < blocks.size(); ++i) ext += dotTensors(*blocks[i], dBlocks[i], res);
  acc.extContraction += 0.5 * ext;

  // --- the tangent along a CI direction (#156) ------------------------------
  if (ci != nullptr) {
    // A CI direction moves no integral: every external block's dot is null,
    // so the product rule reaches the active blocks alone.
    const std::vector<const DeviceTensor*> noBlockMoves(blocks.size(), nullptr);
    const DeviceTensor normDot = DeviceTensor::zeros(tupleDims, res);
    const DeviceTensor hDot = DeviceTensor::zeros(tupleDims, res);
    evalTermsTangent(normTerms, noBlockMoves, ci->gDots, normDot, stream);
    evalTermsTangent(hTerms, noBlockMoves, ci->gDots, hDot, stream);
    // No sync before the downloads, for the reason the base pair's give.
    const Tensor hostNormDot = downloadTensor(normDot, stream);
    const Tensor hostHDot = downloadTensor(hDot, stream);
    acc.ciTangent +=
        dotHost(m.p.data(), hostNormDot.data()) + dotHost(m.q.data(), hostHDot.data());
    const ScSlabMultiplierTangents t = multiplierDotFn(
        hostNorm, hostH, hostNormDot, hostHDot, std::vector<double>(m.r.size(), 0.0));
    const DeviceTensor pDot = uploadTensor(t.p, res);
    const DeviceTensor qDot = uploadTensor(t.q, res);
    accumActiveDerivatives(normTerms, pDot, ci->dActive, res);
    accumActiveDerivatives(hTerms, qDot, ci->dActive, res);
  }

  // --- the mixed direction (#184) -------------------------------------------
  if (mix != nullptr) {
    check(rot.size() == 1 && rot.dirs[0].tangent,
          "assembleSlab: the mixed direction is rot.dirs[0], with its tangent blocks");
    const ScRotationDir& w = rot.dirs[0];
    const ScSlabDots dots = slabDots(blocks, kinds, w, rot.gen, res);
    const std::vector<const DeviceTensor*> none(blocks.size(), nullptr);
    ScMixedTables tab;
    tab.norm = hostNorm;
    tab.h = hostH;
    tab.normDotW = tangentTable(normTerms, dots.ptr, gDots[0], tupleDims, res);
    tab.hDotW = tangentTable(hTerms, dots.ptr, gDots[0], tupleDims, res);
    tab.hDotWDigest = tangentTable(hTerms, none, mix->wDigest, tupleDims, res);
    const std::vector<double> zero(m.r.size(), 0.0);
    const std::vector<double> diffDotW =
        mix->epsKappa ? deltaOf(w.epsCoreDot, w.epsVirtDot) : zero;
    acc.mixedEdotW += dotHost(m.p.data(), tab.normDotW.data()) +
                      dotHost(m.q.data(), tab.hDotW.data()) + dotHost(m.r, diffDotW);
    if (mix->forward) {
      tab.normDotU = tangentTable(normTerms, none, mix->uDots, tupleDims, res);
      tab.hDotU = tangentTable(hTerms, none, mix->uDots, tupleDims, res);
      tab.normWU = mixedTable(normTerms, dots.ptr, mix->uDots, tupleDims, res);
      tab.hWU = mixedTable(hTerms, dots.ptr, mix->uDots, tupleDims, res);
      tab.hWUInt = tangentTable(hTerms, none, mix->wuInt, tupleDims, res);
      tab.hWUDigest = tangentTable(hTerms, none, mix->wuDigest, tupleDims, res);
      const std::vector<double> diffDotU = deltaOf(*mix->epsCoreU, *mix->epsVirtU);
      const std::vector<double> diffWU = deltaOf(*mix->epsCoreWU, *mix->epsVirtWU);
      acc.mixedEdotU += dotHost(m.p.data(), tab.normDotU.data()) +
                        dotHost(m.q.data(), tab.hDotU.data()) + dotHost(m.r, diffDotU);
      acc.mixed += mixedFn(tab, diffDotW, diffDotU, diffWU);
    }
    if (mix->reverse) {
      const ScSlabMultiplierTangents t =
          multiplierDotFn(hostNorm, hostH, tab.normDotW, tab.hDotW, diffDotW);
      const DeviceTensor pDot = uploadTensor(t.p, res);
      const DeviceTensor qDot = uploadTensor(t.q, res);
      accumActiveDerivatives(normTerms, pDot, mix->dActive, res);
      accumActiveDerivatives(hTerms, qDot, mix->dActive, res);
      accumActiveDerivativesAlong(normTerms, mp, dots.ptr, mix->dActive, res);
      accumActiveDerivativesAlong(hTerms, mq, dots.ptr, mix->dActive, res);
      if (mix->epsCi) deltaAdjointInto(deltaOf, t.r, acc.dEdCoreDot, acc.dEdVirtDot);
    }
  }

  // --- the rotation, one direction at a time --------------------------------
  for (std::size_t d = 0; d < rot.size(); ++d) {
    const ScSlabDots dots = slabDots(blocks, kinds, rot.dirs[d], rot.gen, res);
    acc.rotExt[d] += pairSlabDots(dots, dBlocks, res);
    // The orbital-energy half, R_t against this slab's Deltadot. One
    // contraction, and the adjoint's and the tangent's alike -- it is
    // reported once (acc.rotEps) and added to both.
    if (rot.dirs[d].general)
      acc.rotEps[d] +=
          dotHost(m.r, deltaOf(rot.dirs[d].epsCoreDot, rot.dirs[d].epsVirtDot));
    if (!rot.dirs[d].tangent) continue;
    const DeviceTensor normDot = DeviceTensor::zeros(tupleDims, res);
    const DeviceTensor hDot = DeviceTensor::zeros(tupleDims, res);
    evalTermsTangent(normTerms, dots.ptr, gDots[d], normDot, stream);
    evalTermsTangent(hTerms, dots.ptr, gDots[d], hDot, stream);
    // No sync before the downloads, for the reason the base pair's give.
    const Tensor hostNormDot = downloadTensor(normDot, stream);
    const Tensor hostHDot = downloadTensor(hDot, stream);
    acc.rotTangent[d] +=
        dotHost(m.p.data(), hostNormDot.data()) + dotHost(m.q.data(), hostHDot.data());
  }
}

// Eqs. 42-43 contracted back, once a class's walk is over: sum_G <G, M_G>
// over the active-space pseudodensities it accumulated.
double activeContraction(const std::vector<ScActiveAccum>& dActive, const DeviceResources& res) {
  double sum = 0.0;
  for (const ScActiveAccum& a : dActive) sum += dotTensors(*a.g, a.d, res);
  return sum;
}

// The accumulator `g` got, or nullptr -- how a class reaches one of its own M
// / E after the walk (Srs, which differentiates its Dyall block in turn).
const DeviceTensor* activeAccumOf(const std::vector<ScActiveAccum>& dActive,
                                  const DeviceTensor* g) {
  for (const ScActiveAccum& a : dActive)
    if (a.g == g) return &a.d;
  return nullptr;
}

// The Dyall-block tangents built up front, for the directions that asked for
// the forward tangent -- those are needed INSIDE the slab walk (the product
// rule runs per slab), while a direction that only wants the adjoint pairing
// has its set built one at a time afterwards, in rotationPairActive. That is
// what keeps the peak at one extra set of a class's blocks per TANGENT
// direction rather than per direction.
//
// `gdotFn(dir)` is the class's own one-liner: its make_* calls with that
// direction's dotted integrals (gradient_sc_classes.cppm).
template <class GDotFn>
std::vector<std::vector<ScGDot>> rotationTangentBlocks(const ScRotation& rot, GDotFn&& gdotFn) {
  std::vector<std::vector<ScGDot>> out(rot.size());
  for (std::size_t d = 0; d < rot.size(); ++d)
    if (rot.dirs[d].tangent) out[d] = gdotFn(rot.dirs[d]);
  return out;
}

// The ACTIVE half of the adjoint pairing, once a class's walk is over and its
// M_G accumulators are complete: sum_G <M_G, Gdot>, into `out[d]`. A
// direction whose tangent blocks were built up front reuses them; the rest
// build, pair and free one direction at a time, so a long direction list
// costs no extra peak.
template <class GDotFn>
void rotationPairActive(const ScRotation& rot, const std::vector<std::vector<ScGDot>>& pre,
                        const std::vector<ScActiveAccum>& dActive, GDotFn&& gdotFn,
                        std::vector<double>& out, const DeviceResources& res) {
  for (std::size_t d = 0; d < rot.size(); ++d) {
    std::vector<ScGDot> own;
    if (pre[d].empty()) own = gdotFn(rot.dirs[d]);
    const std::vector<ScGDot>& gDots = pre[d].empty() ? own : pre[d];
    for (const ScGDot& g : gDots) {
      const DeviceTensor* m = activeAccumOf(dActive, g.g);
      if (m != nullptr) out[d] += dotTensors(g.dot, *m, res);
    }
  }
}

}  // namespace nevpt2
