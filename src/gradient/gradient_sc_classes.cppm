// nevpt2.gradient:sc_classes -- the eight SC classes' pseudodensity
// assemblies. Each is the class's norm and Dyall-Hamiltonian terms written out
// as an ScTerm list -- the SAME subscript strings, coefficients and operands as
// the matching deviceEinsum* calls in nevpt2.energy's energy_<class>, so the
// two read side by side -- handed to nevpt2.gradient:terms' assembleSlab,
// which evaluates them, forms the multipliers and differentiates them.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :sc_classes;` and from no importer of nevpt2.gradient.
// (Used by gradient.cpp.)
//
// The metric and Dyall-Hamiltonian blocks are nevpt2.energy's own make_*
// (exported from energy.cppm), built once per class before its slab walk and
// freed when the class returns, exactly as the class energies do.
//
// Three classes need more than a term list, and only these three:
//   Sijrs  has no active space at all -- its norm is a quadratic form in the
//          integrals alone (nevpt2.energy runs it as one reduction kernel, not
//          as einsums), so its pseudodensity is written out by hand;
//   Sijr   and Srsi have perturbers that are SYMMETRIZED pairs of tuples, so
//          their multipliers come from a triangle and are scattered back over
//          the raw tuple shape (sijrMultipliers / srsiMultipliers);
//   Srsi   additionally needs two slab walks, because the pair (r, s) it
//          symmetrizes over can straddle two batches;
//   Sir    reads one integral block on its DIAGONAL (h2e_v2_Sir[r, a, a, i]),
//          whose derivative an einsum output cannot spell, so that read is
//          hoisted into a block of its own.
// EACH CLASS ALSO CARRIES THE ORBITAL ROTATION, in four lines: WHICH blocks
// its external operands are (`:general`'s RotBlock, from which one table
// gives both their active axes and their general-range promotion rule), a
// `gdot` lambda that is its own make_* calls with a direction's DOTTED
// integrals, a `delta` construction written as a FUNCTION of the orbital
// energies (so the same formula at the response `deps` is its tangent), and
// `rot`/`gDots` handed to assembleSlab. Everything else -- the per-slab
// pairing, the product rule, the per-direction bookkeeping -- is generic
// (gradient_terms.cppm, gradient_rotation.cppm, gradient_general.cppm). An
// RDM-only metric has no tangent and is simply absent from the `gdot` list.
//
// AND EACH TAKES AN OPTIONAL `adj`: non-null, the class also sweeps its
// Eqs. 42-43 accumulators one chain rule further in, down to the five RDM
// coefficient tensors Park Eq. 48 needs (nevpt2.gradient:rdm_adjoint, which
// has the derivation). That has to happen HERE, inside the class, because the
// accumulators are keyed by the block tensor each is conjugate to and those
// blocks are built and freed per class -- the sweep is a few lines at the end
// of each walk, in reverse-topological order where a class has nested blocks
// (Sij's a9 -> hdm3 -> hdm2, Srs's a7 -> rm3 -> rm2, Sijr's a3 -> hdm1).
// Null, nothing below the Eqs. 42-43 accumulators runs at all. The two are
// independent of each other: the rotation reads the Eqs. 42-43 accumulators
// forwards, the sweep reads them backwards, and a caller asks for either,
// both or neither.
//
// AND AN OPTIONAL `ci` (#156, :rdm_adjoint's ScCiDirection): non-null, the
// walk is also differentiated along a CI direction -- the class's active
// blocks rebuilt at the RDMs' tangents (the same builders, and
// nevpt2.energy's tangent-only ones where a block carries an RDM-free term,
// exactly as nevpt2.energy's *Blocks do at BlockOrder::Tangent), the
// multipliers' tangents accumulated into a second set of Eqs. 42-43
// accumulators (:terms' ScCiWalk), and the class's own sweep, written once as
// a lambda, run on that set too. Sijrs reads no RDM and takes none.
//
// AND AN OPTIONAL `mix` (#184, :rdm_adjoint's ScMixedDirection): the mixed
// block H_kc along the orbital direction rot.dirs[0] and, forward, a CI
// direction. Each class builds the blocks only it knows -- u's tangent blocks
// (the same `ciBlocks` its CI walk builds), the C3 blocks (its integral-
// carrying make_* at w's dotted integrals and u's tangent RDMs, tangent-only
// builders where a block has an RDM-free term) and, for the two f3-reading
// blocks, the digest-only builds -- and :terms' assembleSlab does the rest.
// Reverse, the class's sweep runs three more times (mixedReverseSweeps), at
// the integrals it is handed: that is why each sweep takes them as arguments.
module nevpt2.gradient:sc_classes;

import std;
import nevpt2.gradient;
import :rdm_adjoint;  // ScRdmAdjoint and the per-block adjoint sweeps
import :rotation;
import :general;
import :terms;
// The Dyall blocks' transposes in the integrals: the one-pass Y (#240).
import :block_transpose;

namespace nevpt2 {

// --- Sr: one virtual index --------------------------------------------------

void gradientSr(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                const Tensor& hostECore, const Tensor& hostEVirt, const DeviceTensor& dm1,
                const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceTensor& f3ac,
                const DeviceTensor& f3ca, int64_t norb, const ScRotation& rot,
                ScClassGradient& acc, ScRdmAdjoint* adj, const ScCiDirection* ci,
                const ScMixedDirection* mix, const DeviceResources& dr) {
  const DeviceTensor& h1e = ai.h1e;
  const DeviceTensor& h2e = ai.h2e;
  const DeviceTensor a16 = make_a16(h1e, h2e, dm3, f3ac, f3ca, norb, dr);
  const DeviceTensor a17 = make_a17(h1e, h2e, dm2, dm3, dr);
  const DeviceTensor a19 = make_a19(h1e, h2e, dm1, dm2, dr);
  acc.hasActive = true;
  // Along a CI direction every block is linear in the RDMs: the raw ones'
  // tangents are the caller's, viewed; a16/a17/a19 their make_* at them.
  const auto ciBlocks = [&](const RdmTensors& t) {
    std::vector<ScGDot> v;
    v.push_back(ScGDot{&dm1, t.dm1.view()});
    v.push_back(ScGDot{&dm2, t.dm2.view()});
    v.push_back(ScGDot{&dm3, t.dm3.view()});
    v.push_back(ScGDot{&a16, make_a16(h1e, h2e, t.dm3, t.f3ac, t.f3ca, norb, dr)});
    v.push_back(ScGDot{&a17, make_a17(h1e, h2e, t.dm2, t.dm3, dr)});
    v.push_back(ScGDot{&a19, make_a19(h1e, h2e, t.dm1, t.dm2, dr)});
    return v;
  };
  std::optional<ScCiWalk> cw;
  if (ci != nullptr) {
    cw.emplace(ciWalkOf(*ci));
    cw->gDots = ciBlocks(ci->dot);
  }
  // The M / E of Eqs. 42-43 have no external index left, so one set
  // serves the whole walk; the class contracts them back at the end.
  std::vector<ScActiveAccum> dActive;
  // a16's f3 addends carry no integral factor, so the zeros isolate the
  // h1e/h2e part of its tangent (gradient_rotation.cppm) -- unless the
  // direction carries the digests' own tangent (#184), which then takes their
  // place: the digests' share merged per tuple. dm1/dm2/dm3 are this class's
  // other blocks and do not move at all.
  const DeviceTensor z3ac = DeviceTensor::zeros(f3ac.dims, dr);
  const DeviceTensor z3ca = DeviceTensor::zeros(f3ca.dims, dr);
  const auto gdot = [&](const ScRotationDir& d) {
    const DeviceTensor& fac = d.f3acDot != nullptr ? *d.f3acDot : z3ac;
    const DeviceTensor& fca = d.f3caDot != nullptr ? *d.f3caDot : z3ca;
    std::vector<ScGDot> v;
    v.push_back(ScGDot{&a16, make_a16(d.h1eDot, d.h2eDot, dm3, fac, fca, norb, dr)});
    v.push_back(ScGDot{&a17, make_a17(d.h1eDot, d.h2eDot, dm2, dm3, dr)});
    v.push_back(ScGDot{&a19, make_a19(d.h1eDot, d.h2eDot, dm1, dm2, dr)});
    return v;
  };
  const std::vector<std::vector<ScGDot>> gDots = rotationTangentBlocks(rot, gdot);
  // The mixed direction (#184): a16 is the class's one f3-reading block, so
  // its digest-only builds (zero integrals) are the digests' share of G'[w]
  // and C4's block; C3's are the tangent-RDM builders at w's dotted integrals.
  std::optional<ScMixedWalk> mw;
  if (mix != nullptr) {
    const ScRotationDir& w = rot.dirs[0];
    const DeviceTensor& z1 = *mix->zeroH1e;
    const DeviceTensor& z2 = *mix->zeroH2e;
    mw.emplace(mixedWalkOf(*mix));
    mw->wDigest.push_back(
        ScGDot{&a16, make_a16(z1, z2, dm3, *w.f3acDot, *w.f3caDot, norb, dr)});
    // The orbital pair (#158): v's digest-only build, the same at rot.dirs[1].
    if (mix->pairForward) {
      const ScRotationDir& v = rot.dirs[1];
      mw->vDigest.push_back(
          ScGDot{&a16, make_a16(z1, z2, dm3, *v.f3acDot, *v.f3caDot, norb, dr)});
    }
    if (mix->forward) {
      const RdmTensors& t = *mix->dotU;
      mw->uDots = ciBlocks(t);
      mw->wuInt.push_back(
          ScGDot{&a16, make_a16(w.h1eDot, w.h2eDot, t.dm3, z3ac, z3ca, norb, dr)});
      mw->wuInt.push_back(ScGDot{&a17, make_a17(w.h1eDot, w.h2eDot, t.dm2, t.dm3, dr)});
      mw->wuInt.push_back(ScGDot{&a19, make_a19(w.h1eDot, w.h2eDot, t.dm1, t.dm2, dr)});
      mw->wuDigest.push_back(
          ScGDot{&a16, make_a16(z1, z2, t.dm3, *mix->f3acWU, *mix->f3caWU, norb, dr)});
    }
  }

  forBatches(src.extent(ExtBlock::Sr), batch, [&](int64_t b0, int64_t b1) {
    const DeviceTensor slab = src.slab(ExtBlock::Sr, b0, b1, dr);
    const DeviceTensor h1e_v = sliceAxis(ai.h1e_v_Sr, 0, b0, b1);
    const ScOperand g2{&slab, 0}, g1{&h1e_v, 1};
    const std::vector<ScTerm> normTerms{
        {"ipqr,rpqbac,iabc->i", {g2, {&dm3, kScActive}, g2}, 1.0},
        {"ipqr,rpqa,ia->i", {g2, {&dm2, kScActive}, g1}, 2.0},
        {"ip,pa,ia->i", {g1, {&dm1, kScActive}, g1}, 1.0},
    };
    const std::vector<ScTerm> hTerms{
        {"ipqr,pqrabc,iabc->i", {g2, {&a16, kScActive}, g2}, 1.0},
        {"ipqr,pqra,ia->i", {g2, {&a17, kScActive}, g1}, 2.0},
        {"ip,pa,ia->i", {g1, {&a19, kScActive}, g1}, 1.0},
    };
    // Delta_t = eps_r, as a function of the orbital energies: read at the
    // real ones for the multipliers, and at `deps` for their response.
    const auto deltaOf = [&](const Tensor&, const Tensor& ev) { return srDelta(b0, b1, ev); };
    const std::vector<double> delta = deltaOf(hostECore, hostEVirt);
    assembleSlab(
        normTerms, hTerms, {b1 - b0}, {&slab, &h1e_v}, dActive,
        [&](const Tensor& n, const Tensor& h) { return plainMultipliers(n, h, delta); },
        deltaOf, rot, {RotBlock::Sr, RotBlock::H1eSr}, gDots, cw ? &*cw : nullptr,
        [&](const Tensor& n, const Tensor& h, const Tensor& nd, const Tensor& hd,
            const std::vector<double>& dd) {
          return plainMultiplierTangents(n, h, delta, nd, hd, dd);
        },
        mw ? &*mw : nullptr,
        [&](const ScMixedTables& t, const std::vector<double>& dw, const std::vector<double>& du,
            const std::vector<double>& dwu) { return plainMixedTerms(t, delta, dw, du, dwu); },
        [&](const ScEmptyTables& t) { return plainEmptyTerm(t, delta); }, acc, dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
  rotationPairActive(rot, gDots, dActive, gdot, acc.rotAct, dr);
  // The one-pass Y (#240): the Dyall blocks' accumulators through their
  // builders' transposes, onto the active integrals' adjoint.
  if (rot.bars != nullptr) {
    const DeviceTensor eyeT = deviceEye(dm1.dims[0], dr);
    scClassIntegralSweep(dActive, {{&a16, DyallBlock::A16}, {&a17, DyallBlock::A17}, {&a19, DyallBlock::A19}},
                         DyallOperands{&dm1, &dm2, &dm3, nullptr, nullptr, nullptr, nullptr, nullptr, &eyeT},
                         ScIntegralBars{rot.bars->h1e, rot.bars->h2e}, dr);
  }

  // The RDM sweep: dm1/dm2/dm3 are contracted RAW on the norm side, so their
  // accumulators ARE their Eq. 48 contributions; a16/a17/a19 are one chain
  // rule away, and a16 is one of the only two blocks in the eight classes
  // that read the f3 digests at all. Linear in the accumulators, so the CI
  // direction's tangent is the same sweep on its own (#156); affine in the
  // integrals it is handed, which is what the mixed direction's reverse
  // sweeps read (#184, :rdm_adjoint's ScMixedDirection).
  const auto sweep = [&](const std::vector<ScActiveAccum>& acts, const DeviceTensor& i1,
                         const DeviceTensor& i2, ScRdmAdjoint& out) {
    const wwrStream_t s = dr.stream();
    adjointAddRaw(acts, &dm1, out.dm1, s);
    adjointAddRaw(acts, &dm2, out.dm2, s);
    adjointAddRaw(acts, &dm3, out.dm3, s);
    const DeviceTensor delta = deviceEye(norb, dr);
    if (const DeviceTensor* m = activeAccumOf(acts, &a16); m != nullptr)
      adjointA16(*m, i1, i2, delta, out, dr);
    if (const DeviceTensor* m = activeAccumOf(acts, &a17); m != nullptr)
      adjointA17(*m, i1, i2, out, dr);
    if (const DeviceTensor* m = activeAccumOf(acts, &a19); m != nullptr)
      adjointA19(*m, i1, i2, out, dr);
  };
  if (adj != nullptr) sweep(dActive, h1e, h2e, *adj);
  if (ci != nullptr) sweep(cw->dActive, h1e, h2e, ci->adjDot);
  if (mix != nullptr && mix->reverse)
    mixedReverseSweeps(sweep, dActive, *mw, h1e, h2e, rot.dirs[0].h1eDot, rot.dirs[0].h2eDot,
                       *mix);
}

// --- Si: one core index (the hole-side metrics, and -e_core as Delta) -------

void gradientSi(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                const Tensor& hostECore, const Tensor& hostEVirt, const DeviceTensor& dm1,
                const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceTensor& f3ac,
                const DeviceTensor& f3ca, int64_t norb, const ScRotation& rot,
                ScClassGradient& acc, ScRdmAdjoint* adj, const ScCiDirection* ci,
                const ScMixedDirection* mix, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const DeviceTensor& h1e = ai.h1e;
  const DeviceTensor& h2e = ai.h2e;
  const DeviceTensor a22 = make_a22(h1e, h2e, dm2, dm3, f3ac, f3ca, norb, dr);
  const DeviceTensor a23 = make_a23(h1e, h2e, dm1, dm2, dm3, dr);
  const DeviceTensor a25 = make_a25(h1e, h2e, dm1, dm2, dr);
  const DeviceTensor dm3_h = make_dm3_h(dm2, dm3, dr);
  const DeviceTensor dm2_h = make_dm2_h(dm1, dm2, dr);
  // dm1_h = 2 delta - dm1^T, as energy_Si builds it.
  const DeviceTensor eye = deviceEye(norb, dr);
  const DeviceTensor dm1_h = deviceEinsumNew("ij->ij", {&eye}, 2.0, dr);
  deviceTransposeAccum(dm1, {1, 0}, dm1_h, -1.0, s);
  acc.hasActive = true;
  // Along a CI direction: a25 and dm1_h carry an RDM-free term (2 h1e^T,
  // 2 delta), so theirs are the tangent-only builders; the rest are their
  // make_* at the tangent RDMs (nevpt2.energy's siBlocks, Tangent order).
  const auto ciBlocks = [&](const RdmTensors& t) {
    std::vector<ScGDot> v;
    v.push_back(ScGDot{&a22, make_a22(h1e, h2e, t.dm2, t.dm3, t.f3ac, t.f3ca, norb, dr)});
    v.push_back(ScGDot{&a23, make_a23(h1e, h2e, t.dm1, t.dm2, t.dm3, dr)});
    v.push_back(ScGDot{&a25, make_a25_dot(h1e, h2e, t.dm1, t.dm2, dr)});
    v.push_back(ScGDot{&dm3_h, make_dm3_h(t.dm2, t.dm3, dr)});
    v.push_back(ScGDot{&dm2_h, make_dm2_h(t.dm1, t.dm2, dr)});
    v.push_back(ScGDot{&dm1_h, make_hdm1_dot(t.dm1, dr)});
    return v;
  };
  std::optional<ScCiWalk> cw;
  if (ci != nullptr) {
    cw.emplace(ciWalkOf(*ci));
    cw->gDots = ciBlocks(ci->dot);
  }
  // The M / E of Eqs. 42-43 have no external index left, so one set
  // serves the whole walk; the class contracts them back at the end.
  std::vector<ScActiveAccum> dActive;
  // a22 as a16: the f3 addends are integral-free, so they are zeroed in the
  // tangent -- or carry the direction's own digests' tangent when it has one
  // (#184). a25's `2 h1e^T` term, by contrast, IS an integral term and so
  // belongs in it -- make_a25 called with the dotted integrals carries it.
  const DeviceTensor z3ac = DeviceTensor::zeros(f3ac.dims, dr);
  const DeviceTensor z3ca = DeviceTensor::zeros(f3ca.dims, dr);
  const auto gdot = [&](const ScRotationDir& d) {
    const DeviceTensor& fac = d.f3acDot != nullptr ? *d.f3acDot : z3ac;
    const DeviceTensor& fca = d.f3caDot != nullptr ? *d.f3caDot : z3ca;
    std::vector<ScGDot> v;
    v.push_back(ScGDot{&a22, make_a22(d.h1eDot, d.h2eDot, dm2, dm3, fac, fca, norb, dr)});
    v.push_back(ScGDot{&a23, make_a23(d.h1eDot, d.h2eDot, dm1, dm2, dm3, dr)});
    v.push_back(ScGDot{&a25, make_a25(d.h1eDot, d.h2eDot, dm1, dm2, dr)});
    return v;
  };
  const std::vector<std::vector<ScGDot>> gDots = rotationTangentBlocks(rot, gdot);
  // The mixed direction (#184), as Sr's with a22 the f3-reading block. C3's
  // a25 is the TANGENT builder at w's dotted integrals: its `2 h1e^T` term
  // reads no RDM, so it has no CI derivative and no mixed one.
  std::optional<ScMixedWalk> mw;
  if (mix != nullptr) {
    const ScRotationDir& w = rot.dirs[0];
    const DeviceTensor& z1 = *mix->zeroH1e;
    const DeviceTensor& z2 = *mix->zeroH2e;
    mw.emplace(mixedWalkOf(*mix));
    mw->wDigest.push_back(
        ScGDot{&a22, make_a22(z1, z2, dm2, dm3, *w.f3acDot, *w.f3caDot, norb, dr)});
    if (mix->pairForward) {  // the orbital pair (#158), as Sr's
      const ScRotationDir& v = rot.dirs[1];
      mw->vDigest.push_back(
          ScGDot{&a22, make_a22(z1, z2, dm2, dm3, *v.f3acDot, *v.f3caDot, norb, dr)});
    }
    if (mix->forward) {
      const RdmTensors& t = *mix->dotU;
      mw->uDots = ciBlocks(t);
      mw->wuInt.push_back(
          ScGDot{&a22, make_a22(w.h1eDot, w.h2eDot, t.dm2, t.dm3, z3ac, z3ca, norb, dr)});
      mw->wuInt.push_back(
          ScGDot{&a23, make_a23(w.h1eDot, w.h2eDot, t.dm1, t.dm2, t.dm3, dr)});
      mw->wuInt.push_back(ScGDot{&a25, make_a25_dot(w.h1eDot, w.h2eDot, t.dm1, t.dm2, dr)});
      mw->wuDigest.push_back(ScGDot{
          &a22, make_a22(z1, z2, t.dm2, t.dm3, *mix->f3acWU, *mix->f3caWU, norb, dr)});
    }
  }

  forBatches(src.extent(ExtBlock::Si), batch, [&](int64_t b0, int64_t b1) {
    const DeviceTensor slab = src.slab(ExtBlock::Si, b0, b1, dr);
    const DeviceTensor h1e_v = sliceAxis(ai.h1e_v_Si, 1, b0, b1);
    const ScOperand g2{&slab, 0}, g1{&h1e_v, 1};
    const std::vector<ScTerm> normTerms{
        {"qpir,rpqbac,baic->i", {g2, {&dm3_h, kScActive}, g2}, 1.0},
        {"qpir,rpqa,ai->i", {g2, {&dm2_h, kScActive}, g1}, 2.0},
        {"pi,pa,ai->i", {g1, {&dm1_h, kScActive}, g1}, 1.0},
    };
    const std::vector<ScTerm> hTerms{
        {"qpir,pqrabc,baic->i", {g2, {&a22, kScActive}, g2}, 1.0},
        {"qpir,pqra,ai->i", {g2, {&a23, kScActive}, g1}, 2.0},
        {"pi,pa,ai->i", {g1, {&a25, kScActive}, g1}, 1.0},
    };
    // Delta_t = -eps_i.
    const auto deltaOf = [&](const Tensor& ec, const Tensor&) { return siDelta(b0, b1, ec); };
    const std::vector<double> delta = deltaOf(hostECore, hostEVirt);
    assembleSlab(
        normTerms, hTerms, {b1 - b0}, {&slab, &h1e_v}, dActive,
        [&](const Tensor& n, const Tensor& h) { return plainMultipliers(n, h, delta); },
        deltaOf, rot, {RotBlock::Si, RotBlock::H1eSi}, gDots, cw ? &*cw : nullptr,
        [&](const Tensor& n, const Tensor& h, const Tensor& nd, const Tensor& hd,
            const std::vector<double>& dd) {
          return plainMultiplierTangents(n, h, delta, nd, hd, dd);
        },
        mw ? &*mw : nullptr,
        [&](const ScMixedTables& t, const std::vector<double>& dw, const std::vector<double>& du,
            const std::vector<double>& dwu) { return plainMixedTerms(t, delta, dw, du, dwu); },
        [&](const ScEmptyTables& t) { return plainEmptyTerm(t, delta); }, acc, dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
  rotationPairActive(rot, gDots, dActive, gdot, acc.rotAct, dr);
  // The one-pass Y (#240): the Dyall blocks' accumulators through their
  // builders' transposes, onto the active integrals' adjoint.
  if (rot.bars != nullptr) {
    const DeviceTensor eyeT = deviceEye(dm1.dims[0], dr);
    scClassIntegralSweep(dActive, {{&a22, DyallBlock::A22}, {&a23, DyallBlock::A23}, {&a25, DyallBlock::A25}},
                         DyallOperands{&dm1, &dm2, &dm3, nullptr, nullptr, nullptr, nullptr, nullptr, &eyeT},
                         ScIntegralBars{rot.bars->h1e, rot.bars->h2e}, dr);
  }

  // The RDM sweep: six blocks, none of which reads another, so no ordering
  // question. a22 is the second (and last) block that reads the f3 digests.
  const auto sweep = [&](const std::vector<ScActiveAccum>& acts, const DeviceTensor& i1,
                         const DeviceTensor& i2, ScRdmAdjoint& out) {
    if (const DeviceTensor* m = activeAccumOf(acts, &a22); m != nullptr)
      adjointA22(*m, i1, i2, eye, out, dr);
    if (const DeviceTensor* m = activeAccumOf(acts, &a23); m != nullptr)
      adjointA23(*m, i1, i2, out, s);
    if (const DeviceTensor* m = activeAccumOf(acts, &a25); m != nullptr)
      adjointA25(*m, i1, i2, out, s);
    if (const DeviceTensor* m = activeAccumOf(acts, &dm3_h); m != nullptr)
      adjointDm3H(*m, eye, out, s);
    if (const DeviceTensor* m = activeAccumOf(acts, &dm2_h); m != nullptr)
      adjointDm2H(*m, eye, out, s);
    if (const DeviceTensor* m = activeAccumOf(acts, &dm1_h); m != nullptr)
      adjointHoleDm1(*m, out, s);
  };
  if (adj != nullptr) sweep(dActive, h1e, h2e, *adj);
  if (ci != nullptr) sweep(cw->dActive, h1e, h2e, ci->adjDot);
  if (mix != nullptr && mix->reverse)
    mixedReverseSweeps(sweep, dActive, *mw, h1e, h2e, rot.dirs[0].h1eDot, rot.dirs[0].h2eDot,
                       *mix);
}

// --- Sijrs: no active space -------------------------------------------------
//
// The MP2-like class. nevpt2.energy runs it as one reduction kernel
// (device_einsum.cu's energy_sijrs) over
//
//   N_t = g[i,a,j,b] theta[i,a,j,b],   theta = 2 g[i,a,j,b] - g[i,b,j,a],
//   Delta_t = eps_a + eps_b - eps_i - eps_j,   H_t = 0
//
// so there is no term list to differentiate and no M / E to assemble. By hand,
// from d/dg of sum_t P_t N_t (the third piece is the theta of the tuple with
// the two virtuals exchanged, which contributes to THIS integral):
//
//   D[i,a,j,b] = P[i,a,j,b] (theta[i,a,j,b] + 2 g[i,a,j,b]) - P[i,b,j,a] g[i,b,j,a],
//
// and (1/2) <g, D> = sum_t P_t N_t = E, Eq. 40 again (the exchanged piece
// relabels into the straight one).
//
// It has NO ACTIVE-ACTIVE rotation source term at all, in either direction:
// cvcv carries no active index, so an active rotation moves nothing it reads.
// A GENERAL direction is a different matter -- all four of its labels are
// core or virtual, and its Delta_t is four orbital energies -- so it carries
// both halves of the full source term:
//
//   the adjoint   <D, gdot>, the pseudodensity above against the direction's
//                 action on cvcv, plus R_t Deltadot_t;
//   the tangent   sum_t P_t Ndot_t + R_t Deltadot_t, with
//                 Ndot = gdot theta + g thetadot and thetadot the same
//                 2 gdot - gdot(swapped) the norm's own theta is.
//
// H_t is identically zero here, so there is no Q term in either.
//
// THE MIXED DIRECTION (#184) reaches it only through Delta: N_t moves along w
// and not along u, H_t is zero, so of the five terms only C1 (the 3x3 between
// (Ndot, 0, Ddot)[w] and (0, 0, Ddot)[u]) and C5 (R . Delta(eps^{wu})) are
// there, and both need the orbital energies to respond to the CI vector. In
// reverse, likewise, it reaches the CI vector only through dE/de and its
// tangent along w.
//
// A CI DIRECTION IN FULL MODE (#214) reaches it the same way: `epsCoreU` /
// `epsVirtU` non-null (the orbital energies' tangent along u), N_t and H_t
// stay put and only Delta moves, so the walk's Edot(u) gains R . Deltadot[u]
// and dE/de's tangent Rdot = E_DeltaDelta Deltadot[u] -- through the same
// unguarded finish as the rest of the class -- joins acc.dEdCoreDot /
// dEdVirtDot. Null: nothing, as before (its CI derivative at fixed orbital
// energies is exactly zero).
void gradientSijrs(IntegralSource& src, int64_t batch, const Tensor& hostECore,
                   const Tensor& hostEVirt, const ScRotation& rot, ScClassGradient& acc,
                   const ScMixedDirection* mix, const Tensor* epsCoreU, const Tensor* epsVirtU,
                   const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t ncore = hostECore.size(), nvirt = hostEVirt.size();
  if (ncore == 0 || nvirt == 0) return;
  acc.hasActive = false;

  forBatches(src.extent(ExtBlock::Sijrs), batch, [&](int64_t b0, int64_t b1) {
    const DeviceTensor slab = src.slab(ExtBlock::Sijrs, b0, b1, dr);  // (ni, v, c, v)
    const DeviceTensor theta = deviceTransposeNew(slab, {0, 3, 2, 1}, -1.0, dr);
    deviceEinsumAccum("iajb->iajb", {&slab}, theta, 2.0, s);
    const DeviceTensor norm = deviceEinsumNew("iajb,iajb->iajb", {&slab, &theta}, 1.0, dr);

    const Tensor hostNorm = downloadTensor(norm, s);
    // Delta_t = eps_a + eps_b - eps_i - eps_j, as a function of the orbital
    // energies (assembleSlab's `deltaOf`, written out here because this class
    // does not go through it).
    const auto deltaOf = [&](const Tensor& ec, const Tensor& ev) {
      return sijrsDelta(b0, b1, ncore, nvirt, ec, ev);
    };
    const std::vector<double> delta = deltaOf(hostECore, hostEVirt);
    // Sijrs's exact finish (#190): energy_sijrs applies no guard, so neither
    // do its multipliers, here or in any of the tangents below.
    const ScMultipliers m = sijrsMultipliers(hostNorm.data(), delta);
    acc.norm += m.norm;
    acc.energy += m.energy;
    // dE/de, as assembleSlab takes it (#87).
    if (acc.epsAdjoint) deltaAdjointInto(deltaOf, m.r, acc.dEdCore, acc.dEdVirt);

    const DeviceTensor p = uploadTensor(Tensor::fromFlat(hostNorm.dims(), m.p), dr);
    const DeviceTensor d = deviceEinsumNew("iajb,iajb->iajb", {&p, &theta}, 1.0, dr);
    deviceEinsumAccum("iajb,iajb->iajb", {&p, &slab}, d, 2.0, s);
    deviceEinsumAccum("ibja,ibja->iajb", {&p, &slab}, d, -1.0, s);
    acc.extContraction += 0.5 * dotTensors(slab, d, dr);
    // The one-pass Y (#240): cvcv's pseudodensity through its dot, transposed.
    if (rot.bars != nullptr) {
      std::vector<DeviceTensor> dv;
      dv.push_back(d.view());
      addExternalBars(*rot.bars, {&slab}, {RotBlock::Sijrs}, dv, dr);
    }

    for (std::size_t k = 0; k < rot.size(); ++k) {
      const ScRotationDir& dir = rot.dirs[k];
      if (!dir.general) continue;  // an active rotation moves nothing here
      acc.rotEps[k] += dotHost(m.r, deltaOf(dir.epsCoreDot, dir.epsVirtDot));
      const ScSlabDots dots = slabDots({&slab}, {RotBlock::Sijrs}, dir, rot.gen, dr);
      if (dots.ptr[0] == nullptr) continue;
      const DeviceTensor& gdot = *dots.ptr[0];
      acc.rotExt[k] += dotTensors(d, gdot, dr);
      if (!dir.tangent) continue;
      const DeviceTensor thetaDot = deviceTransposeNew(gdot, {0, 3, 2, 1}, -1.0, dr);
      deviceEinsumAccum("iajb->iajb", {&gdot}, thetaDot, 2.0, s);
      const DeviceTensor nDot =
          deviceEinsumNew("iajb,iajb->iajb", {&gdot, &theta}, 1.0, dr);
      deviceEinsumAccum("iajb,iajb->iajb", {&slab, &thetaDot}, nDot, 1.0, s);
      acc.rotTangent[k] += dotHost(m.p, downloadTensor(nDot, s).data());
    }

    // A CI direction, the orbital energies responding to it (#214).
    if (epsCoreU != nullptr) {
      const std::vector<double> diffDotU = deltaOf(*epsCoreU, *epsVirtU);
      acc.ciTangent += dotHost(m.r, diffDotU);
      const Tensor noNormDot =
          Tensor::fromFlat(hostNorm.dims(), std::vector<double>(delta.size(), 0.0));
      const ScSlabMultiplierTangents t =
          sijrsMultiplierTangents(hostNorm, delta, noNormDot, diffDotU);
      deltaAdjointInto(deltaOf, t.r, acc.dEdCoreDot, acc.dEdVirtDot);
    }

    if (mix == nullptr) return;
    // The mixed direction, w = rot.dirs[0]: Ndot[w] the same product rule as
    // the tangent above (zero for an active w: cvcv carries no active index).
    const ScRotationDir& w = rot.dirs[0];
    const std::vector<double> zero(delta.size(), 0.0);
    Tensor normDotW = Tensor::fromFlat(hostNorm.dims(), zero);
    if (w.general) {
      const ScSlabDots dots = slabDots({&slab}, {RotBlock::Sijrs}, w, rot.gen, dr);
      if (dots.ptr[0] != nullptr) {
        const DeviceTensor& gdot = *dots.ptr[0];
        const DeviceTensor thetaDot = deviceTransposeNew(gdot, {0, 3, 2, 1}, -1.0, dr);
        deviceEinsumAccum("iajb->iajb", {&gdot}, thetaDot, 2.0, s);
        const DeviceTensor nDot = deviceEinsumNew("iajb,iajb->iajb", {&gdot, &theta}, 1.0, dr);
        deviceEinsumAccum("iajb,iajb->iajb", {&slab, &thetaDot}, nDot, 1.0, s);
        normDotW = downloadTensor(nDot, s);
      }
    }
    const std::vector<double> diffDotW =
        mix->epsKappa ? deltaOf(w.epsCoreDot, w.epsVirtDot) : zero;
    acc.mixedEdotW += dotHost(m.p, normDotW.data()) + dotHost(m.r, diffDotW);
    const Tensor zeroT = Tensor::fromFlat(hostNorm.dims(), zero);
    if (mix->forward) {
      ScMixedTables tab{hostNorm, zeroT, normDotW, zeroT, zeroT, zeroT, zeroT,
                        zeroT,    zeroT, zeroT,    zeroT};
      const std::vector<double> diffDotU = deltaOf(mix->epsCoreU, mix->epsVirtU);
      const std::vector<double> diffWU = deltaOf(mix->epsCoreWU, mix->epsVirtWU);
      acc.mixedEdotU += dotHost(m.r, diffDotU);
      acc.mixed += sijrsMixedTerms(tab, delta, diffDotW, diffDotU, diffWU);
    }
    if (mix->reverse && mix->epsCi) {
      const ScSlabMultiplierTangents t = sijrsMultiplierTangents(hostNorm, delta, normDotW, diffDotW);
      deltaAdjointInto(deltaOf, t.r, acc.dEdCoreDot, acc.dEdVirtDot);
    }

    // THE ORBITAL PAIR (#158), u = w and v = rot.dirs[1]: H_t is zero, so of
    // K3-K5 only K3 (the 3x3 between (Ndot, 0, Ddot)[u] and [v]) and K4
    // (N^{uv} = g'[u] theta'[v] + g'[v] theta'[u], theta linear in g) are here.
    // Reverse, D's tangent along v -- D is linear in g at fixed P, so it is
    // D's own formula at Pdot plus the same at g'[v] -- paired with g'[u].
    if (!mix->pairForward && !mix->pairReverse) return;
    const ScRotationDir& v = rot.dirs[1];
    const ScSlabDots dotsU = slabDots({&slab}, {RotBlock::Sijrs}, w, rot.gen, dr);
    const ScSlabDots dotsV = slabDots({&slab}, {RotBlock::Sijrs}, v, rot.gen, dr);
    const DeviceTensor zeroSlab = DeviceTensor::zeros(slab.dims, dr);
    const DeviceTensor& gu = dotsU.ptr[0] != nullptr ? *dotsU.ptr[0] : zeroSlab;
    const DeviceTensor& gv = dotsV.ptr[0] != nullptr ? *dotsV.ptr[0] : zeroSlab;
    const auto thetaOf = [&](const DeviceTensor& x) {
      DeviceTensor t = deviceTransposeNew(x, {0, 3, 2, 1}, -1.0, dr);
      deviceEinsumAccum("iajb->iajb", {&x}, t, 2.0, s);
      return t;
    };
    const DeviceTensor thetaU = thetaOf(gu);
    const DeviceTensor thetaV = thetaOf(gv);
    const DeviceTensor nDotV = deviceEinsumNew("iajb,iajb->iajb", {&gv, &theta}, 1.0, dr);
    deviceEinsumAccum("iajb,iajb->iajb", {&slab, &thetaV}, nDotV, 1.0, s);
    const Tensor normDotV = downloadTensor(nDotV, s);
    const std::vector<double> diffDotV = deltaOf(v.epsCoreDot, v.epsVirtDot);
    acc.pairEdotV += dotHost(m.p, normDotV.data()) + dotHost(m.r, diffDotV);
    // N^{ww'} = g'[w] theta'[w'] + g'[w'] theta'[w], both slots one per direction.
    const auto crossOf = [&](const DeviceTensor& a, const DeviceTensor& ta, const DeviceTensor& b,
                             const DeviceTensor& tb) {
      const DeviceTensor x = deviceEinsumNew("iajb,iajb->iajb", {&a, &tb}, 1.0, dr);
      deviceEinsumAccum("iajb,iajb->iajb", {&b, &ta}, x, 1.0, s);
      return downloadTensor(x, s);
    };
    std::optional<ScEmptySlab> empty;
    if (mix->pairForward) {
      const Tensor hostUV = crossOf(gu, thetaU, gv, thetaV);
      ScPairTables pt{hostNorm, zeroT, normDotW, zeroT, zeroT, normDotV, zeroT,
                      zeroT,    hostUV,   zeroT, zeroT, zeroT,
                      zeroT,    zeroT};
      ScPairTerms terms = pairTermsVia(
          [&](const ScMixedTables& t, const std::vector<double>& dw,
              const std::vector<double>& du, const std::vector<double>& dwu) {
            return sijrsMixedTerms(t, delta, dw, du, dwu);
          },
          pt, diffDotW, diffDotV);
      // The guarded tuples (#190): a tuple reads g[i,a,j,b] and, through
      // theta, g[i,b,j,a]. Sijrs's own P = -1/Delta is already K4's on every
      // one of them (sijrsMultipliers), so their share is moved from k4 to k6,
      // where every class reports it, and no score is taken (H = 0).
      ScEmptyTables et;
      et.norm = hostNorm;
      {
        const Tensor hg = downloadTensor(slab, s);
        et.xMax = Tensor(hg.dims());
        for (int64_t i = 0; i < hg.dim(0); ++i)
          for (int64_t a = 0; a < hg.dim(1); ++a)
            for (int64_t j = 0; j < hg.dim(2); ++j)
              for (int64_t bb = 0; bb < hg.dim(3); ++bb)
                et.xMax(i, a, j, bb) =
                    std::max(std::fabs(hg(i, a, j, bb)), std::fabs(hg(i, bb, j, a)));
      }
      et.normUU = crossOf(gu, thetaU, gu, thetaU);
      et.normVV = crossOf(gv, thetaV, gv, thetaV);
      et.normUV = hostUV;
      et.hUU = et.hVV = et.hUV = zeroT;
      et.scaleU = mix->pairScaleU;
      et.scaleV = mix->pairScaleV;
      empty.emplace(plainEmptyTerm(et, delta, /*hZero=*/true));
      acc.guarded += empty->census;
      terms.k4 -= empty->k6;
      terms.k4Swap -= empty->k6;
      terms.k6 = empty->k6;
      acc.pair += terms;
    }
    if (mix->pairReverse) {
      check(mix->pairForward,
            "gradientSijrs: the orbital pair's reverse reads the forward finish's guarded share");
      const ScSlabMultiplierTangents t =
          sijrsMultiplierTangents(hostNorm, delta, normDotV, diffDotV);
      const DeviceTensor pDot = uploadTensor(t.p, dr);
      const DeviceTensor dDot = deviceEinsumNew("iajb,iajb->iajb", {&pDot, &theta}, 1.0, dr);
      deviceEinsumAccum("iajb,iajb->iajb", {&pDot, &slab}, dDot, 2.0, s);
      deviceEinsumAccum("ibja,ibja->iajb", {&pDot, &slab}, dDot, -1.0, s);
      deviceEinsumAccum("iajb,iajb->iajb", {&p, &thetaV}, dDot, 1.0, s);
      deviceEinsumAccum("iajb,iajb->iajb", {&p, &gv}, dDot, 2.0, s);
      deviceEinsumAccum("ibja,ibja->iajb", {&p, &gv}, dDot, -1.0, s);
      acc.pairReverse += dotTensors(dDot, gu, dr) + dotHost(t.r, diffDotW);
      // K6's reverse half on its own (in the line above already): D's
      // tangent along v at the guarded tuples' multiplier alone.
      if (empty->any) {
        const DeviceTensor pBar = uploadTensor(empty->pBar, dr);
        const DeviceTensor d6 = deviceEinsumNew("iajb,iajb->iajb", {&pBar, &thetaV}, 1.0, dr);
        deviceEinsumAccum("iajb,iajb->iajb", {&pBar, &gv}, d6, 2.0, s);
        deviceEinsumAccum("ibja,ibja->iajb", {&pBar, &gv}, d6, -1.0, s);
        acc.pairReverseK6 += dotTensors(d6, gu, dr);
      }
    }
  });
}

// --- Sijr: a virtual and a symmetrized core pair ----------------------------

// Its Delta vector and its symmetrized multipliers -- energy_Sijr's
// symmetrization and triangular packing plus the scatter back (sijrDelta,
// sijrMultipliers) -- are nevpt2.gradient_plan's, pinned card-free there.

void gradientSijr(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                  const Tensor& hostECore, const Tensor& hostEVirt, const DeviceTensor& dm1,
                  const DeviceTensor& dm2, const ScRotation& rot, ScClassGradient& acc,
                  ScRdmAdjoint* adj, const ScCiDirection* ci, const ScMixedDirection* mix,
                  const DeviceResources& dr) {
  const int64_t ncore = hostECore.size();
  if (ncore == 0) return;
  const DeviceTensor hdm1 = make_hdm1(dm1, dr);
  const DeviceTensor a3 = make_a3(ai.h1e, ai.h2e, dm1, dm2, hdm1, dr);
  acc.hasActive = true;
  // Along a CI direction: hdm1's 2 delta drops (make_hdm1_dot), and make_a3
  // is linear in (dm1, dm2, hdm1), so the tangent hdm1 carries it -- at the
  // state's integrals for the CI tangent, at w's dotted ones for the mixed
  // direction's C3 (#184).
  const auto ciBlocksAt = [&](const DeviceTensor& i1, const DeviceTensor& i2,
                              const RdmTensors& t, bool withMetric) {
    DeviceTensor hdm1Dot = make_hdm1_dot(t.dm1, dr);
    std::vector<ScGDot> v;
    v.push_back(ScGDot{&a3, make_a3(i1, i2, t.dm1, t.dm2, hdm1Dot, dr)});
    if (withMetric) v.push_back(ScGDot{&hdm1, std::move(hdm1Dot)});
    return v;
  };
  std::optional<ScCiWalk> cw;
  if (ci != nullptr) {
    cw.emplace(ciWalkOf(*ci));
    cw->gDots = ciBlocksAt(ai.h1e, ai.h2e, ci->dot, /*withMetric=*/true);
  }
  std::optional<ScMixedWalk> mw;
  if (mix != nullptr) {
    mw.emplace(mixedWalkOf(*mix));
    if (mix->forward) {
      const ScRotationDir& w = rot.dirs[0];
      mw->uDots = ciBlocksAt(ai.h1e, ai.h2e, *mix->dotU, /*withMetric=*/true);
      mw->wuInt = ciBlocksAt(w.h1eDot, w.h2eDot, *mix->dotU, /*withMetric=*/false);
    }
  }
  // The M / E of Eqs. 42-43 have no external index left, so one set
  // serves the whole walk; the class contracts them back at the end.
  std::vector<ScActiveAccum> dActive;
  // hdm1 is pure RDM (its `2 delta` included: a rotation leaves delta alone),
  // so a3 is the one block of this class that moves.
  const auto gdot = [&](const ScRotationDir& d) {
    std::vector<ScGDot> v;
    v.push_back(ScGDot{&a3, make_a3(d.h1eDot, d.h2eDot, dm1, dm2, hdm1, dr)});
    return v;
  };
  const std::vector<std::vector<ScGDot>> gDots = rotationTangentBlocks(rot, gdot);

  forBatches(src.extent(ExtBlock::Sijr), batch, [&](int64_t b0, int64_t b1) {
    const DeviceTensor slab = src.slab(ExtBlock::Sijr, b0, b1, dr);  // (nr, a, c, c)
    const ScOperand g{&slab, 0};
    const std::vector<ScTerm> normTerms{
        {"rpji,raji,pa->rji", {g, g, {&hdm1, kScActive}}, 2.0},
        {"rpji,raij,pa->rji", {g, g, {&hdm1, kScActive}}, -1.0},
    };
    const std::vector<ScTerm> hTerms{
        {"rpji,raji,pa->rji", {g, g, {&a3, kScActive}}, 2.0},
        {"rpji,raij,pa->rji", {g, g, {&a3, kScActive}}, -1.0},
    };
    assembleSlab(
        normTerms, hTerms, {b1 - b0, ncore, ncore}, {&slab}, dActive,
        [&](const Tensor& n, const Tensor& h) {
          return sijrMultipliers(n, h, b0, hostECore, hostEVirt);
        },
        [&](const Tensor& ec, const Tensor& ev) {
          return sijrDelta(b1 - b0, ncore, b0, ec, ev);
        },
        rot, {RotBlock::Sijr}, gDots, cw ? &*cw : nullptr,
        [&](const Tensor& n, const Tensor& h, const Tensor& nd, const Tensor& hd,
            const std::vector<double>& dd) {
          return sijrMultiplierTangents(n, h, nd, hd, dd, b0, hostECore, hostEVirt);
        },
        mw ? &*mw : nullptr,
        [&](const ScMixedTables& t, const std::vector<double>& dw, const std::vector<double>& du,
            const std::vector<double>& dwu) {
          return sijrMixedTerms(t, dw, du, dwu, b0, hostECore, hostEVirt);
        },
        [&](const ScEmptyTables& t) { return sijrEmptyTerm(t, b0, hostECore, hostEVirt); }, acc,
        dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
  rotationPairActive(rot, gDots, dActive, gdot, acc.rotAct, dr);
  // The one-pass Y (#240): the Dyall blocks' accumulators through their
  // builders' transposes, onto the active integrals' adjoint.
  if (rot.bars != nullptr) {
    const DeviceTensor eyeT = deviceEye(dm1.dims[0], dr);
    scClassIntegralSweep(dActive, {{&a3, DyallBlock::A3}},
                         DyallOperands{&dm1, &dm2, nullptr, &hdm1, nullptr, nullptr, nullptr, nullptr, &eyeT},
                         ScIntegralBars{rot.bars->h1e, rot.bars->h2e}, dr);
  }

  // The RDM sweep: hdm1 is both this class's own metric and an operand of a3,
  // so its accumulator is seeded from the walk and then added to by a3's
  // sweep, before hdm1 itself is swept down to dm1.
  const auto sweep = [&](const std::vector<ScActiveAccum>& acts, const DeviceTensor& i1,
                         const DeviceTensor& i2, ScRdmAdjoint& out) {
    const wwrStream_t s = dr.stream();
    const int64_t norb = dm1.dims[0];
    const DeviceTensor delta = deviceEye(norb, dr);
    const DeviceTensor mHdm1 = DeviceTensor::zeros(hdm1.dims, dr);
    if (const DeviceTensor* m = activeAccumOf(acts, &hdm1); m != nullptr)
      addTensorInto(*m, mHdm1, s);
    if (const DeviceTensor* m = activeAccumOf(acts, &a3); m != nullptr)
      adjointA3(*m, i1, i2, delta, mHdm1, out, s);
    adjointHoleDm1(mHdm1, out, s);
  };
  if (adj != nullptr) sweep(dActive, ai.h1e, ai.h2e, *adj);
  if (ci != nullptr) sweep(cw->dActive, ai.h1e, ai.h2e, ci->adjDot);
  if (mix != nullptr && mix->reverse)
    mixedReverseSweeps(sweep, dActive, *mw, ai.h1e, ai.h2e, rot.dirs[0].h1eDot,
                       rot.dirs[0].h2eDot, *mix);
}

// --- Srsi: a symmetrized virtual pair and a core index ----------------------

// Its Delta vector and its symmetrized multipliers -- energy_Srsi's
// symmetrization and packing plus the scatter back, symmetric in (r, s)
// (srsiDelta, srsiMultipliers) -- are nevpt2.gradient_plan's, pinned
// card-free there.

// Its own term list, built per slab from the two slabs: the operands' roles
// (`ga` the straight block, `gb` the one read with (r, s) swapped) are what
// the derivative and the product rule both index by, and both of this class's
// passes evaluate the SAME list, so no subscript is written twice.
std::vector<ScTerm> srsiTerms(const ScOperand& ga, const ScOperand& gb, const DeviceTensor& g) {
  return {
      {"rsip,rsia,pa->rsi", {ga, ga, {&g, kScActive}}, 2.0},
      {"rsip,sria,pa->rsi", {ga, gb, {&g, kScActive}}, -1.0},
  };
}

void gradientSrsi(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                  const Tensor& hostECore, const Tensor& hostEVirt, const DeviceTensor& dm1,
                  const DeviceTensor& dm2, const ScRotation& rot, ScClassGradient& acc,
                  ScRdmAdjoint* adj, const ScCiDirection* ci, const ScMixedDirection* mix,
                  const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t nvirt = hostEVirt.size(), ncore = hostECore.size();
  if (nvirt == 0 || ncore == 0) return;
  const DeviceTensor k27 = make_k27(ai.h1e, ai.h2e, dm1, dm2, dr);
  acc.hasActive = true;
  // Along a CI direction: dm1's tangent is the caller's, and k27 is linear in
  // the RDMs. The walk is two-pass here, so the tangent is too (below).
  const auto ciBlocks = [&](const RdmTensors& t) {
    std::vector<ScGDot> v;
    v.push_back(ScGDot{&dm1, t.dm1.view()});
    v.push_back(ScGDot{&k27, make_k27(ai.h1e, ai.h2e, t.dm1, t.dm2, dr)});
    return v;
  };
  std::optional<ScCiWalk> cw;
  if (ci != nullptr) {
    cw.emplace(ciWalkOf(*ci));
    cw->gDots = ciBlocks(ci->dot);
  }
  // dm1 does not move; k27 is this class's one Dyall block.
  const auto gdot = [&](const ScRotationDir& d) {
    std::vector<ScGDot> v;
    v.push_back(ScGDot{&k27, make_k27(d.h1eDot, d.h2eDot, dm1, dm2, dr)});
    return v;
  };
  const std::vector<std::vector<ScGDot>> gDots = rotationTangentBlocks(rot, gdot);
  const std::vector<RotBlock> kinds{RotBlock::Srsi, RotBlock::SrsiT};
  // The mixed direction (#184): k27 reads no digest, so C4 and C1's digest
  // share are absent here; C3's block is make_k27 at w's dotted integrals and
  // u's tangent RDMs.
  std::optional<ScMixedWalk> mw;
  if (mix != nullptr) {
    mw.emplace(mixedWalkOf(*mix));
    if (mix->forward) {
      const ScRotationDir& w = rot.dirs[0];
      const RdmTensors& t = *mix->dotU;
      mw->uDots = ciBlocks(t);
      mw->wuInt.push_back(ScGDot{&k27, make_k27(w.h1eDot, w.h2eDot, t.dm1, t.dm2, dr)});
    }
  }

  // Pass 1: the raw per-tuple (r, s, i) norm and H over the WHOLE virtual
  // pair, exactly as energy_Srsi accumulates them -- the symmetrization pairs
  // (r, s) with (s, r), which can sit in different batches, so no multiplier
  // is known until every slab has been walked. v^2 c doubles; the slabs, not
  // this, are what scale with the active space. The per-direction TANGENT of
  // the same tables rides along, for the same reason: the multipliers it is
  // finished with are not known until the walk is over.
  const std::vector<int64_t> rawDims{nvirt, nvirt, ncore};
  const DeviceTensor normT = DeviceTensor::zeros(rawDims, dr);
  const DeviceTensor hT = DeviceTensor::zeros(rawDims, dr);
  std::vector<DeviceTensor> normTDot, hTDot;
  for (std::size_t d = 0; d < rot.size(); ++d) {
    normTDot.push_back(rot.dirs[d].tangent ? DeviceTensor::zeros(rawDims, dr) : DeviceTensor());
    hTDot.push_back(rot.dirs[d].tangent ? DeviceTensor::zeros(rawDims, dr) : DeviceTensor());
  }
  // The CI direction's raw tangent tables (#156), for the same reason: its
  // multiplier tangents need the whole symmetrized pair too.
  std::optional<DeviceTensor> normTCi, hTCi;
  if (cw) {
    normTCi.emplace(DeviceTensor::zeros(rawDims, dr));
    hTCi.emplace(DeviceTensor::zeros(rawDims, dr));
  }
  // The mixed direction's forward raw tables (#184): u's tangent pair and the
  // mixed pieces C2 and C3. w's own pair is normTDot[0] / hTDot[0] above.
  std::optional<DeviceTensor> normTU, hTU, normTWU, hTWU, hTWUInt;
  if (mw && mw->forward) {
    normTU.emplace(DeviceTensor::zeros(rawDims, dr));
    hTU.emplace(DeviceTensor::zeros(rawDims, dr));
    normTWU.emplace(DeviceTensor::zeros(rawDims, dr));
    hTWU.emplace(DeviceTensor::zeros(rawDims, dr));
    hTWUInt.emplace(DeviceTensor::zeros(rawDims, dr));
  }
  // The orbital pair's forward raw tables (#158): K4's two-slot pieces and
  // K5's two halves. v's own pair is normTDot[1] / hTDot[1] above; k27 reads
  // no digest, so every digests' share is zero here.
  const bool pairForward = mw && mw->pairForward;
  const bool pairReverse = mw && mw->pairReverse;
  // The guarded tuples' (u, u) and (v, v) cross tables too (#190): which
  // perturbers are guarded is not known until the whole pair is walked.
  std::optional<DeviceTensor> normTUV, hTUV, hTUBlockV, hTVBlockU, normTUU, normTVV, hTUU, hTVV;
  Tensor xMaxT(pairForward ? rawDims : std::vector<int64_t>{0, 0, 0});
  if (pairForward) {
    normTUV.emplace(DeviceTensor::zeros(rawDims, dr));
    hTUV.emplace(DeviceTensor::zeros(rawDims, dr));
    hTUBlockV.emplace(DeviceTensor::zeros(rawDims, dr));
    hTVBlockU.emplace(DeviceTensor::zeros(rawDims, dr));
    normTUU.emplace(DeviceTensor::zeros(rawDims, dr));
    normTVV.emplace(DeviceTensor::zeros(rawDims, dr));
    hTUU.emplace(DeviceTensor::zeros(rawDims, dr));
    hTVV.emplace(DeviceTensor::zeros(rawDims, dr));
  }
  forBatches(src.extent(ExtBlock::Srsi), batch, [&](int64_t b0, int64_t b1) {
    const DeviceTensor slab = src.slab(ExtBlock::Srsi, b0, b1, dr);
    const DeviceTensor slabT = src.slab(ExtBlock::SrsiT, b0, b1, dr);
    const ScOperand ga{&slab, 0}, gb{&slabT, 1};
    const DeviceTensor normRows = sliceAxis(normT, 0, b0, b1);
    const DeviceTensor hRows = sliceAxis(hT, 0, b0, b1);
    evalTerms(srsiTerms(ga, gb, dm1), normRows, ScTermFilter::WithActive, s);
    evalTerms(srsiTerms(ga, gb, k27), hRows, ScTermFilter::WithActive, s);
    for (std::size_t d = 0; d < rot.size(); ++d) {
      if (!rot.dirs[d].tangent) continue;
      // No pseudodensity yet in this pass, so no pairing: just the dotted
      // slabs the product rule reads (pass 2 does the pairing).
      const ScSlabDots dots =
          slabDots({&slab, &slabT}, kinds, rot.dirs[d], rot.gen, dr);
      const DeviceTensor normRowsDot = sliceAxis(normTDot[d], 0, b0, b1);
      const DeviceTensor hRowsDot = sliceAxis(hTDot[d], 0, b0, b1);
      evalTermsTangent(srsiTerms(ga, gb, dm1), dots.ptr, gDots[d], normRowsDot, s);
      evalTermsTangent(srsiTerms(ga, gb, k27), dots.ptr, gDots[d], hRowsDot, s);
      if (d != 0 || !mw || !mw->forward) continue;
      const std::vector<const DeviceTensor*> none{nullptr, nullptr};
      evalTermsTangent(srsiTerms(ga, gb, dm1), none, mw->uDots, sliceAxis(*normTU, 0, b0, b1),
                       s);
      evalTermsTangent(srsiTerms(ga, gb, k27), none, mw->uDots, sliceAxis(*hTU, 0, b0, b1), s);
      evalTermsMixed(srsiTerms(ga, gb, dm1), dots.ptr, mw->uDots,
                     sliceAxis(*normTWU, 0, b0, b1), s);
      evalTermsMixed(srsiTerms(ga, gb, k27), dots.ptr, mw->uDots, sliceAxis(*hTWU, 0, b0, b1),
                     s);
      evalTermsTangent(srsiTerms(ga, gb, k27), none, mw->wuInt,
                       sliceAxis(*hTWUInt, 0, b0, b1), s);
    }
    if (pairForward) {
      // The guarded tuples' integrals (#190), this slab's rows.
      const std::vector<ScTerm> terms = srsiTerms(ga, gb, dm1);
      const Tensor rows = tupleIntegralMax({&terms}, {b1 - b0, nvirt, ncore}, s);
      for (int64_t r = 0; r < b1 - b0; ++r)
        for (int64_t q = 0; q < nvirt; ++q)
          for (int64_t i = 0; i < ncore; ++i) xMaxT(b0 + r, q, i) = rows(r, q, i);
      const ScSlabDots du = slabDots({&slab, &slabT}, kinds, rot.dirs[0], rot.gen, dr);
      const ScSlabDots dv = slabDots({&slab, &slabT}, kinds, rot.dirs[1], rot.gen, dr);
      evalTermsCross(srsiTerms(ga, gb, dm1), du.ptr, dv.ptr, sliceAxis(*normTUV, 0, b0, b1), s);
      evalTermsCross(srsiTerms(ga, gb, k27), du.ptr, dv.ptr, sliceAxis(*hTUV, 0, b0, b1), s);
      evalTermsCross(srsiTerms(ga, gb, dm1), du.ptr, du.ptr, sliceAxis(*normTUU, 0, b0, b1), s);
      evalTermsCross(srsiTerms(ga, gb, dm1), dv.ptr, dv.ptr, sliceAxis(*normTVV, 0, b0, b1), s);
      evalTermsCross(srsiTerms(ga, gb, k27), du.ptr, du.ptr, sliceAxis(*hTUU, 0, b0, b1), s);
      evalTermsCross(srsiTerms(ga, gb, k27), dv.ptr, dv.ptr, sliceAxis(*hTVV, 0, b0, b1), s);
      evalTermsMixed(srsiTerms(ga, gb, k27), du.ptr, gDots[1],
                     sliceAxis(*hTUBlockV, 0, b0, b1), s);
      evalTermsMixed(srsiTerms(ga, gb, k27), dv.ptr, gDots[0],
                     sliceAxis(*hTVBlockU, 0, b0, b1), s);
    }
    if (cw) {
      // A CI direction moves neither slab: the active blocks' tangents alone.
      const std::vector<const DeviceTensor*> noBlockMoves{nullptr, nullptr};
      const DeviceTensor normRowsCi = sliceAxis(*normTCi, 0, b0, b1);
      const DeviceTensor hRowsCi = sliceAxis(*hTCi, 0, b0, b1);
      evalTermsTangent(srsiTerms(ga, gb, dm1), noBlockMoves, cw->gDots, normRowsCi, s);
      evalTermsTangent(srsiTerms(ga, gb, k27), noBlockMoves, cw->gDots, hRowsCi, s);
    }
  });

  const Tensor hostNormT = downloadTensor(normT, s);
  const Tensor hostHT = downloadTensor(hT, s);
  const ScSlabMultipliers m = srsiMultipliers(hostNormT, hostHT, hostECore, hostEVirt);
  acc.norm += m.norm;
  acc.energy += m.energy;
  // Delta over the SAME triangle srsiDelta packs, as a function of the
  // orbital energies -- what dE/de and every Delta tangent below read.
  const auto deltaOf = [&](const Tensor& ec, const Tensor& ev) {
    return srsiDelta(nvirt, ncore, ec, ev);
  };
  // dE/de, as assembleSlab takes it for the other classes (#87).
  if (acc.epsAdjoint) deltaAdjointInto(deltaOf, m.r, acc.dEdCore, acc.dEdVirt);
  // The tangent, finished with the scattered multipliers over the RAW tables
  // -- the same identity that lets actReference be taken over them (a
  // perturber's weight multiplies both of its tuples, and the diagonal's
  // halving is already in the packing).
  for (std::size_t d = 0; d < rot.size(); ++d) {
    // The orbital-energy half, R_t against Deltadot over the SAME triangle
    // srsiDelta packs -- as assembleSlab does it for the other classes.
    if (rot.dirs[d].general)
      acc.rotEps[d] += dotHost(m.r, deltaOf(rot.dirs[d].epsCoreDot, rot.dirs[d].epsVirtDot));
    if (!rot.dirs[d].tangent) continue;
    acc.rotTangent[d] += dotHost(m.p.data(), downloadTensor(normTDot[d], s).data()) +
                         dotHost(m.q.data(), downloadTensor(hTDot[d], s).data());
  }
  // Every Srsi term carries an active block, so Eqs. 42-43's reference is the
  // whole P/Q-weighted sum.
  acc.actReference +=
      dotHost(m.p.data(), hostNormT.data()) + dotHost(m.q.data(), hostHT.data());
  const DeviceTensor mp = uploadTensor(m.p, dr);
  const DeviceTensor mq = uploadTensor(m.q, dr);
  // The CI direction's Edot(u) and multiplier tangents, over the same raw
  // tables and the same triangle (#156).
  std::optional<DeviceTensor> mpDot, mqDot;
  if (cw) {
    const Tensor hostNormTCi = downloadTensor(*normTCi, s);
    const Tensor hostHTCi = downloadTensor(*hTCi, s);
    // FULL mode (#214): Delta's tangent along u over the same triangle.
    const bool epsCi = cw->epsCoreU != nullptr;
    const std::vector<double> diffDotU =
        epsCi ? deltaOf(*cw->epsCoreU, *cw->epsVirtU) : std::vector<double>(m.r.size(), 0.0);
    acc.ciTangent += dotHost(m.p.data(), hostNormTCi.data()) +
                     dotHost(m.q.data(), hostHTCi.data()) + dotHost(m.r, diffDotU);
    const ScSlabMultiplierTangents t = srsiMultiplierTangents(
        hostNormT, hostHT, hostNormTCi, hostHTCi, diffDotU, hostECore, hostEVirt);
    mpDot.emplace(uploadTensor(t.p, dr));
    mqDot.emplace(uploadTensor(t.q, dr));
    if (epsCi) deltaAdjointInto(deltaOf, t.r, acc.dEdCoreDot, acc.dEdVirtDot);
  }
  // The mixed direction (#184), over the same raw tables and the same
  // triangle: w's pair is the tangent tables of direction 0 (the digests'
  // share is zero here: k27 reads none), and its multiplier tangents are what
  // pass 2 accumulates Mdot_G with.
  std::optional<DeviceTensor> mpDotW, mqDotW;
  if (mw) {
    const ScRotationDir& w = rot.dirs[0];
    const Tensor zeroT(rawDims);
    const std::vector<double> zero(m.r.size(), 0.0);
    ScMixedTables tab;
    tab.norm = hostNormT;
    tab.h = hostHT;
    tab.normDotW = downloadTensor(normTDot[0], s);
    tab.hDotW = downloadTensor(hTDot[0], s);
    tab.hDotWDigest = zeroT;
    const std::vector<double> diffDotW =
        mw->epsKappa ? deltaOf(w.epsCoreDot, w.epsVirtDot) : zero;
    acc.mixedEdotW += dotHost(m.p.data(), tab.normDotW.data()) +
                      dotHost(m.q.data(), tab.hDotW.data()) + dotHost(m.r, diffDotW);
    if (mw->forward) {
      tab.normDotU = downloadTensor(*normTU, s);
      tab.hDotU = downloadTensor(*hTU, s);
      tab.normWU = downloadTensor(*normTWU, s);
      tab.hWU = downloadTensor(*hTWU, s);
      tab.hWUInt = downloadTensor(*hTWUInt, s);
      tab.hWUDigest = zeroT;
      const std::vector<double> diffDotU = deltaOf(*mw->epsCoreU, *mw->epsVirtU);
      const std::vector<double> diffWU = deltaOf(*mw->epsCoreWU, *mw->epsVirtWU);
      acc.mixedEdotU += dotHost(m.p.data(), tab.normDotU.data()) +
                        dotHost(m.q.data(), tab.hDotU.data()) + dotHost(m.r, diffDotU);
      acc.mixed += srsiMixedTerms(tab, diffDotW, diffDotU, diffWU, hostECore, hostEVirt);
    }
    if (mw->reverse) {
      const ScSlabMultiplierTangents t = srsiMultiplierTangents(
          hostNormT, hostHT, tab.normDotW, tab.hDotW, diffDotW, hostECore, hostEVirt);
      mpDotW.emplace(uploadTensor(t.p, dr));
      mqDotW.emplace(uploadTensor(t.q, dr));
      if (mw->epsCi) deltaAdjointInto(deltaOf, t.r, acc.dEdCoreDot, acc.dEdVirtDot);
    }
  }
  // The orbital pair (#158), over the same raw tables and the same triangle:
  // v's pair is the tangent tables of direction 1. Reverse, its multiplier
  // tangents are what pass 2 differentiates the pseudodensities with, and
  // Rdot meets Ddot[u] here, once for the class.
  std::optional<DeviceTensor> mpDotV, mqDotV, pBarT, qBarT;
  check(!pairReverse || pairForward,
        "gradientSrsi: the orbital pair's reverse reads the forward finish's guarded-tuple "
        "multipliers");
  if (pairForward || pairReverse) {
    const ScRotationDir& u = rot.dirs[0];
    const ScRotationDir& v = rot.dirs[1];
    const Tensor zeroT(rawDims);
    const Tensor normDotV = downloadTensor(normTDot[1], s);
    const Tensor hDotV = downloadTensor(hTDot[1], s);
    const std::vector<double> diffDotU = deltaOf(u.epsCoreDot, u.epsVirtDot);
    const std::vector<double> diffDotV = deltaOf(v.epsCoreDot, v.epsVirtDot);
    acc.pairEdotV += dotHost(m.p.data(), normDotV.data()) + dotHost(m.q.data(), hDotV.data()) +
                     dotHost(m.r, diffDotV);
    if (pairForward) {
      const Tensor hostNormUV = downloadTensor(*normTUV, s);
      const Tensor hostHUV = downloadTensor(*hTUV, s);
      const ScPairTables pt{hostNormT,
                            hostHT,
                            downloadTensor(normTDot[0], s),
                            downloadTensor(hTDot[0], s),
                            zeroT,
                            normDotV,
                            hDotV,
                            zeroT,
                            hostNormUV,
                            hostHUV,
                            downloadTensor(*hTUBlockV, s),
                            zeroT,
                            downloadTensor(*hTVBlockU, s),
                            zeroT};
      ScPairTerms terms = pairTermsVia(
          [&](const ScMixedTables& t, const std::vector<double>& dw,
              const std::vector<double>& du, const std::vector<double>& dwu) {
            return srsiMixedTerms(t, dw, du, dwu, hostECore, hostEVirt);
          },
          pt, diffDotU, diffDotV);
      // The guarded perturbers (#190), over the same triangle.
      ScEmptyTables et;
      et.norm = hostNormT;
      et.xMax = xMaxT;
      et.normUU = downloadTensor(*normTUU, s);
      et.normVV = downloadTensor(*normTVV, s);
      et.normUV = hostNormUV;
      et.hUU = downloadTensor(*hTUU, s);
      et.hVV = downloadTensor(*hTVV, s);
      et.hUV = hostHUV;
      et.scaleU = mw->pairScaleU;
      et.scaleV = mw->pairScaleV;
      const ScEmptySlab e = srsiEmptyTerm(et, hostECore, hostEVirt);
      acc.guarded += e.census;
      terms.k6 = e.k6;
      acc.pair += terms;
      if (pairReverse && e.any) {
        pBarT.emplace(uploadTensor(e.pBar, dr));
        qBarT.emplace(uploadTensor(e.qBar, dr));
      }
    }
    if (pairReverse) {
      const ScSlabMultiplierTangents t = srsiMultiplierTangents(
          hostNormT, hostHT, normDotV, hDotV, diffDotV, hostECore, hostEVirt);
      mpDotV.emplace(uploadTensor(t.p, dr));
      mqDotV.emplace(uploadTensor(t.q, dr));
      acc.pairReverse += dotHost(t.r, diffDotU);
    }
  }

  // Pass 2: the pseudodensities, now that the multipliers exist. The M / E
  // accumulators are class-wide (shaped like dm1 and k27), so they are built
  // on the first batch and accumulated across all of them.
  double ext = 0.0;
  std::vector<ScActiveAccum> dActive;
  forBatches(src.extent(ExtBlock::Srsi), batch, [&](int64_t b0, int64_t b1) {
    const DeviceTensor slab = src.slab(ExtBlock::Srsi, b0, b1, dr);
    const DeviceTensor slabT = src.slab(ExtBlock::SrsiT, b0, b1, dr);
    const ScOperand ga{&slab, 0}, gb{&slabT, 1};
    const std::vector<ScTerm> normTerms = srsiTerms(ga, gb, dm1);
    const std::vector<ScTerm> hTerms = srsiTerms(ga, gb, k27);
    const DeviceTensor pRows = sliceAxis(mp, 0, b0, b1);
    const DeviceTensor qRows = sliceAxis(mq, 0, b0, b1);
    std::vector<DeviceTensor> dBlocks;
    dBlocks.push_back(DeviceTensor::zeros(slab.dims, dr));
    dBlocks.push_back(DeviceTensor::zeros(slabT.dims, dr));
    accumDerivatives(normTerms, pRows, dBlocks, dActive, dr);
    accumDerivatives(hTerms, qRows, dBlocks, dActive, dr);
    if (cw) {
      accumActiveDerivatives(normTerms, sliceAxis(*mpDot, 0, b0, b1), cw->dActive, dr);
      accumActiveDerivatives(hTerms, sliceAxis(*mqDot, 0, b0, b1), cw->dActive, dr);
    }
    if (mw && mw->reverse) {
      // Mdot_G along w: the multipliers' tangents against the plain
      // derivative, and the multipliers against the slabs' tangents.
      const ScSlabDots dots = slabDots({&slab, &slabT}, kinds, rot.dirs[0], rot.gen, dr);
      accumActiveDerivatives(normTerms, sliceAxis(*mpDotW, 0, b0, b1), mw->dActive, dr);
      accumActiveDerivatives(hTerms, sliceAxis(*mqDotW, 0, b0, b1), mw->dActive, dr);
      accumActiveDerivativesAlong(normTerms, pRows, dots.ptr, mw->dActive, dr);
      accumActiveDerivativesAlong(hTerms, qRows, dots.ptr, mw->dActive, dr);
    }
    if (pairReverse) {
      // The orbital pair (#158): this slab's share of D's and M_G's tangents
      // along v, paired with u's dotted slabs and tangent block.
      const ScSlabDots du = slabDots({&slab, &slabT}, kinds, rot.dirs[0], rot.gen, dr);
      const ScSlabDots dv = slabDots({&slab, &slabT}, kinds, rot.dirs[1], rot.gen, dr);
      std::vector<DeviceTensor> dDot;
      dDot.push_back(DeviceTensor::zeros(slab.dims, dr));
      dDot.push_back(DeviceTensor::zeros(slabT.dims, dr));
      std::vector<ScActiveAccum> mDot;
      accumDerivatives(normTerms, sliceAxis(*mpDotV, 0, b0, b1), dDot, mDot, dr);
      accumDerivatives(hTerms, sliceAxis(*mqDotV, 0, b0, b1), dDot, mDot, dr);
      accumDerivativesAlongAll(normTerms, pRows, dv.ptr, gDots[1], dDot, mDot, dr);
      accumDerivativesAlongAll(hTerms, qRows, dv.ptr, gDots[1], dDot, mDot, dr);
      acc.pairReverse += pairSlabDots(du, dDot, dr) + pairActiveAccums(mDot, gDots[0], dr);
      // K6's reverse half (#190), this slab's share.
      if (pBarT) {
        std::vector<DeviceTensor> d6;
        d6.push_back(DeviceTensor::zeros(slab.dims, dr));
        d6.push_back(DeviceTensor::zeros(slabT.dims, dr));
        accumExternalDerivativesAlong(normTerms, sliceAxis(*pBarT, 0, b0, b1), dv.ptr, d6, s);
        accumExternalDerivativesAlong(hTerms, sliceAxis(*qBarT, 0, b0, b1), dv.ptr, d6, s);
        const double k6 = pairSlabDots(du, d6, dr);
        acc.pairReverse += k6;
        acc.pairReverseK6 += k6;
      }
    }
    ext += dotTensors(slab, dBlocks[0], dr) + dotTensors(slabT, dBlocks[1], dr);
    // The one-pass Y (#240): both slabs' pseudodensities, transposed once.
    if (rot.bars != nullptr) addExternalBars(*rot.bars, {&slab, &slabT}, kinds, dBlocks, dr);
    // The external half of the pairing, now that this slab's D exists.
    for (std::size_t d = 0; d < rot.size(); ++d)
      acc.rotExt[d] +=
          pairSlabDots(slabDots({&slab, &slabT}, kinds, rot.dirs[d], rot.gen, dr),
                       dBlocks, dr);
  });
  acc.extContraction += 0.5 * ext;
  acc.actContraction += activeContraction(dActive, dr);
  rotationPairActive(rot, gDots, dActive, gdot, acc.rotAct, dr);
  // The one-pass Y (#240): the Dyall blocks' accumulators through their
  // builders' transposes, onto the active integrals' adjoint.
  if (rot.bars != nullptr) {
    const DeviceTensor eyeT = deviceEye(dm1.dims[0], dr);
    scClassIntegralSweep(dActive, {{&k27, DyallBlock::K27}},
                         DyallOperands{&dm1, &dm2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &eyeT},
                         ScIntegralBars{rot.bars->h1e, rot.bars->h2e}, dr);
  }

  // The RDM sweep: dm1 raw on the norm side, k27 one chain rule away.
  const auto sweep = [&](const std::vector<ScActiveAccum>& acts, const DeviceTensor& i1,
                         const DeviceTensor& i2, ScRdmAdjoint& out) {
    adjointAddRaw(acts, &dm1, out.dm1, s);
    if (const DeviceTensor* m = activeAccumOf(acts, &k27); m != nullptr)
      adjointK27(*m, i1, i2, out, s);
  };
  if (adj != nullptr) sweep(dActive, ai.h1e, ai.h2e, *adj);
  if (ci != nullptr) sweep(cw->dActive, ai.h1e, ai.h2e, ci->adjDot);
  if (mix != nullptr && mix->reverse)
    mixedReverseSweeps(sweep, dActive, *mw, ai.h1e, ai.h2e, rot.dirs[0].h1eDot,
                       rot.dirs[0].h2eDot, *mix);
}

// --- Srs: a virtual pair (Park's own worked example, Eqs. 41-47) ------------
//
// The class the paper writes the pseudodensities out for: rm2 is its metric
// (Gamma^(2) there), a7 its extended-Koopmans Dyall block (K), so the
// external pseudodensity assembled here IS Eq. 41's D_{ar,bs} and the two
// active-space accumulators ARE Eqs. 42-43's M^(-2) and E^(-2) -- each half
// of ours, because our norm and H einsums carry Eqs. 32-33's 1/2 inside the
// contraction rather than outside M and E.
void gradientSrs(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                 const Tensor& hostECore, const Tensor& hostEVirt, const DeviceTensor& dm1,
                 const DeviceTensor& dm2, const DeviceTensor& dm3, const ScRotation& rot,
                 ScClassGradient& acc, ScRdmAdjoint* adj, const ScCiDirection* ci,
                 const ScMixedDirection* mix, const DeviceResources& dr) {
  const int64_t nvirt = hostEVirt.size();
  if (nvirt == 0) return;
  const auto [rm2, a7] = make_a7(ai.h1e, ai.h2e, dm1, dm2, dm3, dr);
  acc.hasActive = true;
  // Along a CI direction: make_a7 is linear in the RDMs (rm2 and a7 both, no
  // RDM-free term -- Srs is one of the three homogeneous classes), so its
  // pair at the tangent RDMs is the pair of tangents. At w's dotted
  // integrals (#184's C3) only a7 carries an integral, so rm2 is dropped.
  const auto ciBlocksAt = [&](const DeviceTensor& i1, const DeviceTensor& i2,
                              const RdmTensors& t, bool withMetric) {
    std::pair<DeviceTensor, DeviceTensor> dotted = make_a7(i1, i2, t.dm1, t.dm2, t.dm3, dr);
    std::vector<ScGDot> v;
    if (withMetric) v.push_back(ScGDot{&rm2, std::move(dotted.first)});
    v.push_back(ScGDot{&a7, std::move(dotted.second)});
    return v;
  };
  std::optional<ScCiWalk> cw;
  if (ci != nullptr) {
    cw.emplace(ciWalkOf(*ci));
    cw->gDots = ciBlocksAt(ai.h1e, ai.h2e, ci->dot, /*withMetric=*/true);
  }
  std::optional<ScMixedWalk> mw;
  if (mix != nullptr) {
    mw.emplace(mixedWalkOf(*mix));
    if (mix->forward) {
      const ScRotationDir& w = rot.dirs[0];
      mw->uDots = ciBlocksAt(ai.h1e, ai.h2e, *mix->dotU, /*withMetric=*/true);
      mw->wuInt = ciBlocksAt(w.h1eDot, w.h2eDot, *mix->dotU, /*withMetric=*/false);
    }
  }
  // The M / E of Eqs. 42-43 have no external index left, so one set
  // serves the whole walk; the class contracts them back at the end.
  std::vector<ScActiveAccum> dActive;
  // rm2 is pure RDM; a7 is the Dyall block, and make_a7's rm2/rm3 (which it
  // rebuilds from the RDMs either way) carry no integral, so the dotted call
  // returns exactly a7's tangent in its second slot.
  const auto gdot = [&](const ScRotationDir& d) {
    std::vector<ScGDot> v;
    std::pair<DeviceTensor, DeviceTensor> dotted =
        make_a7(d.h1eDot, d.h2eDot, dm1, dm2, dm3, dr);
    v.push_back(ScGDot{&a7, std::move(dotted.second)});
    return v;
  };
  const std::vector<std::vector<ScGDot>> gDots = rotationTangentBlocks(rot, gdot);

  forBatches(src.extent(ExtBlock::Srs), batch, [&](int64_t b0, int64_t b1) {
    const DeviceTensor slab = src.slab(ExtBlock::Srs, b0, b1, dr);  // (nr, v, a, a)
    const ScOperand g{&slab, 0};
    const std::vector<ScTerm> normTerms{
        {"rsqp,rsba,pqba->rs", {g, g, {&rm2, kScActive}}, 0.5},
    };
    const std::vector<ScTerm> hTerms{
        {"rsqp,rsba,pqab->rs", {g, g, {&a7, kScActive}}, 0.5},
    };
    // Delta_t = eps_r + eps_s.
    const auto deltaOf = [&](const Tensor&, const Tensor& ev) {
      return srsDelta(b0, b1, nvirt, ev);
    };
    const std::vector<double> delta = deltaOf(hostECore, hostEVirt);
    assembleSlab(
        normTerms, hTerms, {b1 - b0, nvirt}, {&slab}, dActive,
        [&](const Tensor& n, const Tensor& h) { return plainMultipliers(n, h, delta); },
        deltaOf, rot, {RotBlock::Srs}, gDots, cw ? &*cw : nullptr,
        [&](const Tensor& n, const Tensor& h, const Tensor& nd, const Tensor& hd,
            const std::vector<double>& dd) {
          return plainMultiplierTangents(n, h, delta, nd, hd, dd);
        },
        mw ? &*mw : nullptr,
        [&](const ScMixedTables& t, const std::vector<double>& dw, const std::vector<double>& du,
            const std::vector<double>& dwu) { return plainMixedTerms(t, delta, dw, du, dwu); },
        [&](const ScEmptyTables& t) { return plainEmptyTerm(t, delta); }, acc, dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
  rotationPairActive(rot, gDots, dActive, gdot, acc.rotAct, dr);
  // The one-pass Y (#240): a7's accumulator through make_a7's transpose,
  // against its hole RDMs rm2 and rm3, onto the active integrals' adjoint.
  if (rot.bars != nullptr) {
    const DeviceTensor eyeT = deviceEye(dm1.dims[0], dr);
    const DeviceTensor rm3T = make_rm3(dm2, dm3, rm2, dr);
    scClassIntegralSweep(dActive, {{&a7, DyallBlock::A7}},
                         DyallOperands{&dm1, &dm2, &dm3, nullptr, nullptr, nullptr, &rm2,
                                       &rm3T, &eyeT},
                         ScIntegralBars{rot.bars->h1e, rot.bars->h2e}, dr);
  }

  // Eqs. 44-47: one more derivative, now of the Dyall block itself. a7 is
  // linear in the active integrals (h1e, h2e) AND in the hole RDMs make_a7
  // contracts them against (rm2, rm3), so differentiating its own term list
  // against E -- the accumulator the walk just filled for it -- gives the
  // one-electron active-space pseudodensity d^eff (Eqs. 44-45's role), its
  // two-electron partner, and the RDM-conjugate densities (Eqs. 46-47's).
  // Each family is degree 1, so contracting either back against its own
  // operands must give <a7, E>, which is the H side of this class's energy.
  //
  // Ours are in make_a7's form, not Park's Eq. 35 decomposition of K, so they
  // are one chain rule short of his Gamma^(2)/Gamma^(3)-conjugate
  // expressions; that step goes with the orbital response that consumes them.
  const DeviceTensor* eA7 = activeAccumOf(dActive, &a7);
  if (eA7 != nullptr) {
    const DeviceTensor rm3 = make_rm3(dm2, dm3, rm2, dr);
    const ScOperand i1{&ai.h1e, 0}, i2{&ai.h2e, 1};
    const ScOperand srm2{&rm2, kScActive}, srm3{&rm3, kScActive};
    // make_a7's five terms, verbatim.
    const std::vector<ScTerm> a7Terms{
        {"bi,pqia->pqab", {i1, srm2}, -1.0},
        {"ai,pqbi->pqab", {i1, srm2}, -1.0},
        {"kbij,pqkija->pqab", {i2, srm3}, -1.0},
        {"kaij,pqkibj->pqab", {i2, srm3}, -1.0},
        {"baij,pqij->pqab", {i2, srm2}, -1.0},
    };
    std::vector<DeviceTensor> dInts;
    dInts.push_back(DeviceTensor::zeros(ai.h1e.dims, dr));
    dInts.push_back(DeviceTensor::zeros(ai.h2e.dims, dr));
    std::vector<ScActiveAccum> dRdm;
    accumDerivatives(a7Terms, *eA7, dInts, dRdm, dr);
    acc.hasActiveIntegrals = true;
    acc.intReference = dotTensors(a7, *eA7, dr);
    acc.intContraction = dotTensors(ai.h1e, dInts[0], dr) + dotTensors(ai.h2e, dInts[1], dr);
    acc.rdmContraction = activeContraction(dRdm, dr);
  }

  // The RDM sweep, reverse-topologically: a7 reads rm2 and rm3, rm3 reads rm2
  // as well, so a7, then rm3, then rm2 -- and rm2's accumulator is seeded
  // from the walk, where it is this class's own metric.
  const auto sweep = [&](const std::vector<ScActiveAccum>& acts, const DeviceTensor& i1,
                         const DeviceTensor& i2, ScRdmAdjoint& out) {
    const wwrStream_t s = dr.stream();
    const int64_t norb = dm1.dims[0];
    const DeviceTensor delta = deviceEye(norb, dr);
    const DeviceTensor mRm2 = DeviceTensor::zeros(rm2.dims, dr);
    const DeviceTensor mRm3 =
        DeviceTensor::zeros({norb, norb, norb, norb, norb, norb}, dr);
    if (const DeviceTensor* m = activeAccumOf(acts, &rm2); m != nullptr)
      addTensorInto(*m, mRm2, s);
    if (const DeviceTensor* m = activeAccumOf(acts, &a7); m != nullptr)
      adjointA7(*m, i1, i2, mRm2, mRm3, s);
    adjointRm3(mRm3, delta, mRm2, out, s);
    adjointRm2(mRm2, delta, out, s);
  };
  if (adj != nullptr) sweep(dActive, ai.h1e, ai.h2e, *adj);
  if (ci != nullptr) sweep(cw->dActive, ai.h1e, ai.h2e, ci->adjDot);
  if (mix != nullptr && mix->reverse)
    mixedReverseSweeps(sweep, dActive, *mw, ai.h1e, ai.h2e, rot.dirs[0].h1eDot,
                       rot.dirs[0].h2eDot, *mix);
}

// --- Sij: a core pair -------------------------------------------------------

void gradientSij(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                 const Tensor& hostECore, const Tensor& hostEVirt, const DeviceTensor& dm1,
                 const DeviceTensor& dm2, const DeviceTensor& dm3, const ScRotation& rot,
                 ScClassGradient& acc, ScRdmAdjoint* adj, const ScCiDirection* ci,
                 const ScMixedDirection* mix, const DeviceResources& dr) {
  const int64_t ncore = hostECore.size();
  if (ncore == 0) return;
  const DeviceTensor hdm1 = make_hdm1(dm1, dr);
  const DeviceTensor hdm2 = make_hdm2(dm1, dm2, dr);
  const DeviceTensor hdm3 = make_hdm3(dm1, dm2, dm3, hdm1, hdm2, dr);
  const DeviceTensor a9 = make_a9(ai.h1e, ai.h2e, hdm1, hdm2, hdm3, dr);
  acc.hasActive = true;
  // Along a CI direction (nevpt2.energy's sijBlocks, Tangent order): hdm1's
  // and hdm2's RDM-free terms drop, and make_hdm3 / make_a9 are linear in the
  // hole RDMs they take, so the tangent hdm1/hdm2 carry them. The terms read
  // hdm2 and a9. At w's dotted integrals (#184's C3) only a9 carries one.
  const auto ciBlocksAt = [&](const DeviceTensor& i1, const DeviceTensor& i2,
                              const RdmTensors& t, bool withMetric) {
    const DeviceTensor hdm1Dot = make_hdm1_dot(t.dm1, dr);
    DeviceTensor hdm2Dot = make_hdm2_dot(t.dm1, t.dm2, dr);
    const DeviceTensor hdm3Dot = make_hdm3(t.dm1, t.dm2, t.dm3, hdm1Dot, hdm2Dot, dr);
    std::vector<ScGDot> v;
    v.push_back(ScGDot{&a9, make_a9(i1, i2, hdm1Dot, hdm2Dot, hdm3Dot, dr)});
    if (withMetric) v.push_back(ScGDot{&hdm2, std::move(hdm2Dot)});
    return v;
  };
  std::optional<ScCiWalk> cw;
  if (ci != nullptr) {
    cw.emplace(ciWalkOf(*ci));
    cw->gDots = ciBlocksAt(ai.h1e, ai.h2e, ci->dot, /*withMetric=*/true);
  }
  std::optional<ScMixedWalk> mw;
  if (mix != nullptr) {
    mw.emplace(mixedWalkOf(*mix));
    if (mix->forward) {
      const ScRotationDir& w = rot.dirs[0];
      mw->uDots = ciBlocksAt(ai.h1e, ai.h2e, *mix->dotU, /*withMetric=*/true);
      mw->wuInt = ciBlocksAt(w.h1eDot, w.h2eDot, *mix->dotU, /*withMetric=*/false);
    }
  }
  // The M / E of Eqs. 42-43 have no external index left, so one set
  // serves the whole walk; the class contracts them back at the end.
  std::vector<ScActiveAccum> dActive;
  // The hole RDMs hdm1/hdm2/hdm3 read no integral, so a9 is the one block
  // that moves -- built with the dotted integrals and the SAME hole RDMs.
  const auto gdot = [&](const ScRotationDir& d) {
    std::vector<ScGDot> v;
    v.push_back(ScGDot{&a9, make_a9(d.h1eDot, d.h2eDot, hdm1, hdm2, hdm3, dr)});
    return v;
  };
  const std::vector<std::vector<ScGDot>> gDots = rotationTangentBlocks(rot, gdot);

  forBatches(src.extent(ExtBlock::Sij), batch, [&](int64_t b0, int64_t b1) {
    const DeviceTensor slab = src.slab(ExtBlock::Sij, b0, b1, dr);  // (a, a, ni, c)
    const ScOperand g{&slab, 0};
    const std::vector<ScTerm> normTerms{
        {"qpij,baij,pqab->ij", {g, g, {&hdm2, kScActive}}, 0.5},
    };
    const std::vector<ScTerm> hTerms{
        {"qpij,baij,pqab->ij", {g, g, {&a9, kScActive}}, 0.5},
    };
    // Delta_t = -(eps_i + eps_j).
    const auto deltaOf = [&](const Tensor& ec, const Tensor&) {
      return sijDelta(b0, b1, ncore, ec);
    };
    const std::vector<double> delta = deltaOf(hostECore, hostEVirt);
    assembleSlab(
        normTerms, hTerms, {b1 - b0, ncore}, {&slab}, dActive,
        [&](const Tensor& n, const Tensor& h) { return plainMultipliers(n, h, delta); },
        deltaOf, rot, {RotBlock::Sij}, gDots, cw ? &*cw : nullptr,
        [&](const Tensor& n, const Tensor& h, const Tensor& nd, const Tensor& hd,
            const std::vector<double>& dd) {
          return plainMultiplierTangents(n, h, delta, nd, hd, dd);
        },
        mw ? &*mw : nullptr,
        [&](const ScMixedTables& t, const std::vector<double>& dw, const std::vector<double>& du,
            const std::vector<double>& dwu) { return plainMixedTerms(t, delta, dw, du, dwu); },
        [&](const ScEmptyTables& t) { return plainEmptyTerm(t, delta); }, acc, dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
  rotationPairActive(rot, gDots, dActive, gdot, acc.rotAct, dr);
  // The one-pass Y (#240): the Dyall blocks' accumulators through their
  // builders' transposes, onto the active integrals' adjoint.
  if (rot.bars != nullptr) {
    const DeviceTensor eyeT = deviceEye(dm1.dims[0], dr);
    scClassIntegralSweep(dActive, {{&a9, DyallBlock::A9}},
                         DyallOperands{&dm1, &dm2, &dm3, &hdm1, &hdm2, &hdm3, nullptr, nullptr, &eyeT},
                         ScIntegralBars{rot.bars->h1e, rot.bars->h2e}, dr);
  }

  // The RDM sweep, reverse-topologically: a9 reads hdm2 and hdm3, hdm3 reads
  // hdm2 as well, so a9, then hdm3, then hdm2 -- whose accumulator is seeded
  // from the walk, where it is this class's own metric. (hdm1 is passed to
  // make_hdm3 and make_a9 and read by neither, so no term contracts it and it
  // gets no accumulator.)
  const auto sweep = [&](const std::vector<ScActiveAccum>& acts, const DeviceTensor& i1,
                         const DeviceTensor& i2, ScRdmAdjoint& out) {
    const wwrStream_t s = dr.stream();
    const int64_t norb = dm1.dims[0];
    const DeviceTensor delta = deviceEye(norb, dr);
    const DeviceTensor mHdm2 = DeviceTensor::zeros(hdm2.dims, dr);
    const DeviceTensor mHdm3 = DeviceTensor::zeros(hdm3.dims, dr);
    if (const DeviceTensor* m = activeAccumOf(acts, &hdm2); m != nullptr)
      addTensorInto(*m, mHdm2, s);
    if (const DeviceTensor* m = activeAccumOf(acts, &a9); m != nullptr)
      adjointA9(*m, i1, i2, mHdm2, mHdm3, s);
    adjointHdm3(mHdm3, delta, mHdm2, out, s);
    adjointHdm2(mHdm2, delta, out, dr);
  };
  if (adj != nullptr) sweep(dActive, ai.h1e, ai.h2e, *adj);
  if (ci != nullptr) sweep(cw->dActive, ai.h1e, ai.h2e, ci->adjDot);
  if (mix != nullptr && mix->reverse)
    mixedReverseSweeps(sweep, dActive, *mw, ai.h1e, ai.h2e, rot.dirs[0].h1eDot,
                       rot.dirs[0].h2eDot, *mix);
}

// --- Sir: a core and a virtual index ----------------------------------------
//
// The class with the most terms, two integral blocks and the one diagonal
// read. energy_Sir's sixth norm term contracts h2e_v2_Sir[r, a, a, i] against
// a free active index -- an einsum can READ a repeated label but cannot WRITE
// one, so the derivative with respect to that operand has nowhere to go. The
// read is therefore hoisted into its own block, the TRACE
// v2tr[r, i] = sum_a h2e_v2_Sir[r, a, a, i] (block 3), against which the term
// is the same shape as the two one-electron terms below it. v2tr is linear in
// the Sir2 block, so Eq. 40 is unaffected: <v2tr, D_v2tr> is exactly what
// that diagonal contributes, and a consumer wanting it as part of the Sir2
// block scatters it back with delta[a, b].
void gradientSir(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                 const Tensor& hostECore, const Tensor& hostEVirt, const DeviceTensor& dm1,
                 const DeviceTensor& dm2, const DeviceTensor& dm3, const ScRotation& rot,
                 ScClassGradient& acc, ScRdmAdjoint* adj, const ScCiDirection* ci,
                 const ScMixedDirection* mix, const DeviceResources& dr) {
  const int64_t ncore = hostECore.size();
  if (ncore == 0) return;
  const DeviceTensor a12 = make_a12(ai.h1e, ai.h2e, dm1, dm2, dm3, dr);
  const DeviceTensor a13 = make_a13(ai.h1e, ai.h2e, dm1, dm2, dm3, dr);
  acc.hasActive = true;
  // Along a CI direction: dm1/dm2's tangents are the caller's, and a12/a13
  // carry no RDM-free block term (the one-electron norm term `ri,ri` has no
  // active operand, so the product rule skips it by itself). At w's dotted
  // integrals (#184's C3) the raw RDMs carry none, and drop.
  const auto ciBlocksAt = [&](const DeviceTensor& i1, const DeviceTensor& i2,
                              const RdmTensors& t, bool withRaw) {
    std::vector<ScGDot> v;
    if (withRaw) {
      v.push_back(ScGDot{&dm1, t.dm1.view()});
      v.push_back(ScGDot{&dm2, t.dm2.view()});
    }
    v.push_back(ScGDot{&a12, make_a12(i1, i2, t.dm1, t.dm2, t.dm3, dr)});
    v.push_back(ScGDot{&a13, make_a13(i1, i2, t.dm1, t.dm2, t.dm3, dr)});
    return v;
  };
  std::optional<ScCiWalk> cw;
  if (ci != nullptr) {
    cw.emplace(ciWalkOf(*ci));
    cw->gDots = ciBlocksAt(ai.h1e, ai.h2e, ci->dot, /*withRaw=*/true);
  }
  std::optional<ScMixedWalk> mw;
  if (mix != nullptr) {
    mw.emplace(mixedWalkOf(*mix));
    if (mix->forward) {
      const ScRotationDir& w = rot.dirs[0];
      mw->uDots = ciBlocksAt(ai.h1e, ai.h2e, *mix->dotU, /*withRaw=*/true);
      mw->wuInt = ciBlocksAt(w.h1eDot, w.h2eDot, *mix->dotU, /*withRaw=*/false);
    }
  }
  // The M / E of Eqs. 42-43 have no external index left, so one set
  // serves the whole walk; the class contracts them back at the end.
  std::vector<ScActiveAccum> dActive;
  const auto gdot = [&](const ScRotationDir& d) {
    std::vector<ScGDot> v;
    v.push_back(ScGDot{&a12, make_a12(d.h1eDot, d.h2eDot, dm1, dm2, dm3, dr)});
    v.push_back(ScGDot{&a13, make_a13(d.h1eDot, d.h2eDot, dm1, dm2, dm3, dr)});
    return v;
  };
  const std::vector<std::vector<ScGDot>> gDots = rotationTangentBlocks(rot, gdot);

  forBatches(src.extent(ExtBlock::Sir1), batch, [&](int64_t b0, int64_t b1) {
    const DeviceTensor slab1 = src.slab(ExtBlock::Sir1, b0, b1, dr);  // (nr, a, c, a)
    const DeviceTensor slab2 = src.slab(ExtBlock::Sir2, b0, b1, dr);  // (nr, a, a, c)
    const DeviceTensor h1e_v = sliceAxis(ai.h1e_v_Sir, 0, b0, b1);    // (nr, c)
    const DeviceTensor v2tr = deviceEinsumNew("raai->ri", {&slab2}, 1.0, dr);  // (nr, c)
    const ScOperand g1{&slab1, 0}, g2{&slab2, 1}, e1{&h1e_v, 2}, gtr{&v2tr, 3};
    const ScOperand sdm1{&dm1, kScActive}, sdm2{&dm2, kScActive};
    const std::vector<ScTerm> normTerms{
        {"rpiq,raib,qpab->ir", {g1, g1, sdm2}, 2.0},
        {"rpiq,rabi,qpab->ir", {g1, g2, sdm2}, -1.0},
        {"rpqi,raib,qpab->ir", {g2, g1, sdm2}, -1.0},
        {"raqi,rabi,qb->ir", {g2, g2, sdm1}, 2.0},
        {"rpqi,rabi,qbap->ir", {g2, g2, sdm2}, -1.0},
        // energy_Sir's "rpqi,raai,qp->ir", against the hoisted trace.
        {"rpqi,ri,qp->ir", {g2, gtr, sdm1}, 1.0},
        {"rpiq,ri,qp->ir", {g1, e1, sdm1}, 4.0},
        {"rpqi,ri,qp->ir", {g2, e1, sdm1}, -2.0},
        // The one term in any class with no active block at all: it is
        // excluded from Eqs. 42-43's reference (assembleSlab's split).
        {"ri,ri->ir", {e1, e1}, 2.0},
    };
    const std::vector<ScTerm> hTerms{
        {"rpiq,raib,pqab->ir", {g1, g1, {&a12, kScActive}}, 2.0},
        {"rpiq,rabi,pqab->ir", {g1, g2, {&a12, kScActive}}, -1.0},
        {"rpqi,raib,pqab->ir", {g2, g1, {&a12, kScActive}}, -1.0},
        {"rpqi,rabi,pqab->ir", {g2, g2, {&a13, kScActive}}, 1.0},
    };
    // Delta_t = eps_r - eps_i, over (c, nr) -- the terms' own output order.
    const auto deltaOf = [&](const Tensor& ec, const Tensor& ev) {
      return sirDelta(b0, b1, ncore, ec, ev);
    };
    const std::vector<double> delta = deltaOf(hostECore, hostEVirt);
    // h1e_v (v, c) carries no active index, and neither does the hoisted
    // trace v2tr (r, i): its active pair IS the contraction, and a trace of
    // two rotated axes against an ANTISYMMETRIC kappa cancels exactly, so
    // v2tr does not move. Both are `{}` here, which is what makes the
    // hoisting free for the source term as well as for Eq. 40.
    assembleSlab(
        normTerms, hTerms, {ncore, b1 - b0}, {&slab1, &slab2, &h1e_v, &v2tr}, dActive,
        [&](const Tensor& n, const Tensor& h) { return plainMultipliers(n, h, delta); },
        deltaOf, rot,
        {RotBlock::Sir1, RotBlock::Sir2, RotBlock::H1eSir, RotBlock::V2tr}, gDots,
        cw ? &*cw : nullptr,
        [&](const Tensor& n, const Tensor& h, const Tensor& nd, const Tensor& hd,
            const std::vector<double>& dd) {
          return plainMultiplierTangents(n, h, delta, nd, hd, dd);
        },
        mw ? &*mw : nullptr,
        [&](const ScMixedTables& t, const std::vector<double>& dw, const std::vector<double>& du,
            const std::vector<double>& dwu) { return plainMixedTerms(t, delta, dw, du, dwu); },
        [&](const ScEmptyTables& t) { return plainEmptyTerm(t, delta); }, acc, dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
  rotationPairActive(rot, gDots, dActive, gdot, acc.rotAct, dr);
  // The one-pass Y (#240): the Dyall blocks' accumulators through their
  // builders' transposes, onto the active integrals' adjoint.
  if (rot.bars != nullptr) {
    const DeviceTensor eyeT = deviceEye(dm1.dims[0], dr);
    scClassIntegralSweep(dActive, {{&a12, DyallBlock::A12}, {&a13, DyallBlock::A13}},
                         DyallOperands{&dm1, &dm2, &dm3, nullptr, nullptr, nullptr, nullptr, nullptr, &eyeT},
                         ScIntegralBars{rot.bars->h1e, rot.bars->h2e}, dr);
  }

  // The RDM sweep: dm1 and dm2 raw on the norm side, a12 and a13 one chain
  // rule away. The hoisted v2tr block is an INTEGRAL trace, not an RDM, so
  // nothing of it reaches here.
  const auto sweep = [&](const std::vector<ScActiveAccum>& acts, const DeviceTensor& i1,
                         const DeviceTensor& i2, ScRdmAdjoint& out) {
    const wwrStream_t s = dr.stream();
    const int64_t norb = dm1.dims[0];
    const DeviceTensor delta = deviceEye(norb, dr);
    adjointAddRaw(acts, &dm1, out.dm1, s);
    adjointAddRaw(acts, &dm2, out.dm2, s);
    if (const DeviceTensor* m = activeAccumOf(acts, &a12); m != nullptr)
      adjointA12(*m, i1, i2, out, s);
    if (const DeviceTensor* m = activeAccumOf(acts, &a13); m != nullptr)
      adjointA13(*m, i1, i2, delta, out, s);
  };
  if (adj != nullptr) sweep(dActive, ai.h1e, ai.h2e, *adj);
  if (ci != nullptr) sweep(cw->dActive, ai.h1e, ai.h2e, ci->adjDot);
  if (mix != nullptr && mix->reverse)
    mixedReverseSweeps(sweep, dActive, *mw, ai.h1e, ai.h2e, rot.dirs[0].h1eDot,
                       rot.dirs[0].h2eDot, *mix);
}

}  // namespace nevpt2
