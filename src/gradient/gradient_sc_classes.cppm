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
module nevpt2.gradient:sc_classes;

import std;
import nevpt2.gradient;
import :rdm_adjoint;  // ScRdmAdjoint and the per-block adjoint sweeps
import :rotation;
import :general;
import :terms;

namespace nevpt2 {

// --- Sr: one virtual index --------------------------------------------------

void gradientSr(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                const Tensor& hostECore, const Tensor& hostEVirt, const DeviceTensor& dm1,
                const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceTensor& f3ac,
                const DeviceTensor& f3ca, int64_t norb, const ScRotation& rot,
                ScClassGradient& acc, ScRdmAdjoint* adj, const DeviceResources& dr) {
  const DeviceTensor& h1e = ai.h1e;
  const DeviceTensor& h2e = ai.h2e;
  const DeviceTensor a16 = make_a16(h1e, h2e, dm3, f3ac, f3ca, norb, dr);
  const DeviceTensor a17 = make_a17(h1e, h2e, dm2, dm3, dr);
  const DeviceTensor a19 = make_a19(h1e, h2e, dm1, dm2, dr);
  acc.hasActive = true;
  // The M / E of Eqs. 42-43 have no external index left, so one set
  // serves the whole walk; the class contracts them back at the end.
  std::vector<ScActiveAccum> dActive;
  // a16's f3 addends carry no integral factor, so the zeros isolate the
  // h1e/h2e part of its tangent (gradient_rotation.cppm); dm1/dm2/dm3 are
  // this class's other blocks and do not move at all.
  const DeviceTensor z3ac = DeviceTensor::zeros(f3ac.dims, dr);
  const DeviceTensor z3ca = DeviceTensor::zeros(f3ca.dims, dr);
  const auto gdot = [&](const ScRotationDir& d) {
    std::vector<ScGDot> v;
    v.push_back(ScGDot{&a16, make_a16(d.h1eDot, d.h2eDot, dm3, z3ac, z3ca, norb, dr)});
    v.push_back(ScGDot{&a17, make_a17(d.h1eDot, d.h2eDot, dm2, dm3, dr)});
    v.push_back(ScGDot{&a19, make_a19(d.h1eDot, d.h2eDot, dm1, dm2, dr)});
    return v;
  };
  const std::vector<std::vector<ScGDot>> gDots = rotationTangentBlocks(rot, gdot);

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
    const auto deltaOf = [&](const Tensor&, const Tensor& ev) {
      return sliceHost(ev, b0, b1);
    };
    const std::vector<double> delta = deltaOf(hostECore, hostEVirt);
    assembleSlab(
        normTerms, hTerms, {b1 - b0}, {&slab, &h1e_v}, dActive,
        [&](const Tensor& n, const Tensor& h) { return plainMultipliers(n, h, delta); },
        deltaOf, rot, {RotBlock::Sr, RotBlock::H1eSr}, gDots, acc, dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
  rotationPairActive(rot, gDots, dActive, gdot, acc.rotAct, dr);

  // The RDM sweep: dm1/dm2/dm3 are contracted RAW on the norm side, so their
  // accumulators ARE their Eq. 48 contributions; a16/a17/a19 are one chain
  // rule away, and a16 is one of the only two blocks in the eight classes
  // that read the f3 digests at all.
  if (adj != nullptr) {
    const wwrStream_t s = dr.stream();
    adjointAddRaw(dActive, &dm1, adj->dm1, s);
    adjointAddRaw(dActive, &dm2, adj->dm2, s);
    adjointAddRaw(dActive, &dm3, adj->dm3, s);
    const DeviceTensor delta = deviceEye(norb, dr);
    if (const DeviceTensor* m = activeAccumOf(dActive, &a16); m != nullptr)
      adjointA16(*m, h1e, h2e, delta, *adj, dr);
    if (const DeviceTensor* m = activeAccumOf(dActive, &a17); m != nullptr)
      adjointA17(*m, h1e, h2e, *adj, dr);
    if (const DeviceTensor* m = activeAccumOf(dActive, &a19); m != nullptr)
      adjointA19(*m, h1e, h2e, *adj, dr);
  }
}

// --- Si: one core index (the hole-side metrics, and -e_core as Delta) -------

void gradientSi(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                const Tensor& hostECore, const Tensor& hostEVirt, const DeviceTensor& dm1,
                const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceTensor& f3ac,
                const DeviceTensor& f3ca, int64_t norb, const ScRotation& rot,
                ScClassGradient& acc, ScRdmAdjoint* adj, const DeviceResources& dr) {
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
  // The M / E of Eqs. 42-43 have no external index left, so one set
  // serves the whole walk; the class contracts them back at the end.
  std::vector<ScActiveAccum> dActive;
  // a22 as a16: the f3 addends are integral-free, so they are zeroed in the
  // tangent. a25's `2 h1e^T` term, by contrast, IS an integral term and so
  // belongs in it -- make_a25 called with the dotted integrals carries it.
  const DeviceTensor z3ac = DeviceTensor::zeros(f3ac.dims, dr);
  const DeviceTensor z3ca = DeviceTensor::zeros(f3ca.dims, dr);
  const auto gdot = [&](const ScRotationDir& d) {
    std::vector<ScGDot> v;
    v.push_back(ScGDot{&a22, make_a22(d.h1eDot, d.h2eDot, dm2, dm3, z3ac, z3ca, norb, dr)});
    v.push_back(ScGDot{&a23, make_a23(d.h1eDot, d.h2eDot, dm1, dm2, dm3, dr)});
    v.push_back(ScGDot{&a25, make_a25(d.h1eDot, d.h2eDot, dm1, dm2, dr)});
    return v;
  };
  const std::vector<std::vector<ScGDot>> gDots = rotationTangentBlocks(rot, gdot);

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
    const auto deltaOf = [&](const Tensor& ec, const Tensor&) {
      std::vector<double> d = sliceHost(ec, b0, b1);
      for (double& x : d) x = -x;
      return d;
    };
    const std::vector<double> delta = deltaOf(hostECore, hostEVirt);
    assembleSlab(
        normTerms, hTerms, {b1 - b0}, {&slab, &h1e_v}, dActive,
        [&](const Tensor& n, const Tensor& h) { return plainMultipliers(n, h, delta); },
        deltaOf, rot, {RotBlock::Si, RotBlock::H1eSi}, gDots, acc, dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
  rotationPairActive(rot, gDots, dActive, gdot, acc.rotAct, dr);

  // The RDM sweep: six blocks, none of which reads another, so no ordering
  // question. a22 is the second (and last) block that reads the f3 digests.
  if (adj != nullptr) {
    if (const DeviceTensor* m = activeAccumOf(dActive, &a22); m != nullptr)
      adjointA22(*m, h1e, h2e, eye, *adj, dr);
    if (const DeviceTensor* m = activeAccumOf(dActive, &a23); m != nullptr)
      adjointA23(*m, h1e, h2e, *adj, s);
    if (const DeviceTensor* m = activeAccumOf(dActive, &a25); m != nullptr)
      adjointA25(*m, h1e, h2e, *adj, s);
    if (const DeviceTensor* m = activeAccumOf(dActive, &dm3_h); m != nullptr)
      adjointDm3H(*m, eye, *adj, s);
    if (const DeviceTensor* m = activeAccumOf(dActive, &dm2_h); m != nullptr)
      adjointDm2H(*m, eye, *adj, s);
    if (const DeviceTensor* m = activeAccumOf(dActive, &dm1_h); m != nullptr)
      adjointHoleDm1(*m, *adj, s);
  }
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
void gradientSijrs(IntegralSource& src, int64_t batch, const Tensor& hostECore,
                   const Tensor& hostEVirt, const ScRotation& rot, ScClassGradient& acc,
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
      std::vector<double> d;
      d.reserve(static_cast<std::size_t>(hostNorm.size()));
      for (int64_t i = b0; i < b1; ++i)
        for (int64_t a = 0; a < nvirt; ++a)
          for (int64_t j = 0; j < ncore; ++j)
            for (int64_t b = 0; b < nvirt; ++b)
              d.push_back(ev(a) + ev(b) - ec(i) - ec(j));
      return d;
    };
    const std::vector<double> delta = deltaOf(hostECore, hostEVirt);
    const std::vector<double> noH(delta.size(), 0.0);
    const ScMultipliers m = scMultipliers(hostNorm.data(), noH, delta);
    acc.norm += m.norm;
    acc.energy += m.energy;

    const DeviceTensor p = uploadTensor(Tensor::fromFlat(hostNorm.dims(), m.p), dr);
    const DeviceTensor d = deviceEinsumNew("iajb,iajb->iajb", {&p, &theta}, 1.0, dr);
    deviceEinsumAccum("iajb,iajb->iajb", {&p, &slab}, d, 2.0, s);
    deviceEinsumAccum("ibja,ibja->iajb", {&p, &slab}, d, -1.0, s);
    acc.extContraction += 0.5 * dotTensors(slab, d, dr);

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
  });
}

// --- Sijr: a virtual and a symmetrized core pair ----------------------------

// The symmetrization and triangular packing of nevpt2.energy's energy_Sijr,
// verbatim, plus the scatter back. The perturbers are (r, j <= i); the weight
// that multiplies the RAW normT(r, j, i) is that perturber's, and (r, j, i)
// and (r, i, j) are the same perturber, so the scattered array is symmetric
// in (j, i) -- which is exactly what makes
//   sum_{r,j,i} Pfull(r,j,i) normT(r,j,i) == sum_perturbers P normSym
// hold, the diagonal included (normSym(r,i,i) is normT(r,i,i) after its 1/2).
// Delta_t = eps_r - eps_i - eps_j over the (r, j <= i) triangle, in the order
// the packing below walks it: the class's `deltaOf`, so that the same formula
// at the orbital energies' response is Deltadot and lines up with the
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

void gradientSijr(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                  const Tensor& hostECore, const Tensor& hostEVirt, const DeviceTensor& dm1,
                  const DeviceTensor& dm2, const ScRotation& rot, ScClassGradient& acc,
                  ScRdmAdjoint* adj,
                  const DeviceResources& dr) {
  const int64_t ncore = hostECore.size();
  if (ncore == 0) return;
  const DeviceTensor hdm1 = make_hdm1(dm1, dr);
  const DeviceTensor a3 = make_a3(ai.h1e, ai.h2e, dm1, dm2, hdm1, dr);
  acc.hasActive = true;
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
        rot, {RotBlock::Sijr}, gDots, acc, dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
  rotationPairActive(rot, gDots, dActive, gdot, acc.rotAct, dr);

  // The RDM sweep: hdm1 is both this class's own metric and an operand of a3,
  // so its accumulator is seeded from the walk and then added to by a3's
  // sweep, before hdm1 itself is swept down to dm1.
  if (adj != nullptr) {
    const wwrStream_t s = dr.stream();
    const int64_t norb = dm1.dims[0];
    const DeviceTensor delta = deviceEye(norb, dr);
    const DeviceTensor mHdm1 = DeviceTensor::zeros(hdm1.dims, dr);
    if (const DeviceTensor* m = activeAccumOf(dActive, &hdm1); m != nullptr)
      addTensorInto(*m, mHdm1, s);
    if (const DeviceTensor* m = activeAccumOf(dActive, &a3); m != nullptr)
      adjointA3(*m, ai.h1e, ai.h2e, delta, mHdm1, *adj, s);
    adjointHoleDm1(mHdm1, *adj, s);
  }
}

// --- Srsi: a symmetrized virtual pair and a core index ----------------------

// energy_Srsi's symmetrization and packing, verbatim, plus the scatter back --
// as sijrMultipliers, but over the whole (r, s, i) and symmetric in (r, s).
// Delta_t = eps_r + eps_s - eps_i over the (r <= s, i) triangle, in the
// packing order below -- sijrDelta's counterpart, and the class's `deltaOf`.
std::vector<double> srsiDelta(int64_t nvirt, int64_t ncore, const Tensor& ec,
                              const Tensor& ev) {
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(nvirt * (nvirt + 1) / 2 * ncore));
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t ss = r; ss < nvirt; ++ss)
      for (int64_t i = 0; i < ncore; ++i) out.push_back(ev(r) + ev(ss) - ec(i));
  return out;
}

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
                  ScRdmAdjoint* adj,
                  const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t nvirt = hostEVirt.size(), ncore = hostECore.size();
  if (nvirt == 0 || ncore == 0) return;
  const DeviceTensor k27 = make_k27(ai.h1e, ai.h2e, dm1, dm2, dr);
  acc.hasActive = true;
  // dm1 does not move; k27 is this class's one Dyall block.
  const auto gdot = [&](const ScRotationDir& d) {
    std::vector<ScGDot> v;
    v.push_back(ScGDot{&k27, make_k27(d.h1eDot, d.h2eDot, dm1, dm2, dr)});
    return v;
  };
  const std::vector<std::vector<ScGDot>> gDots = rotationTangentBlocks(rot, gdot);
  const std::vector<RotBlock> kinds{RotBlock::Srsi, RotBlock::SrsiT};

  // Pass 1: the raw per-tuple (r, s, i) norm and H over the WHOLE virtual
  // pair, exactly as energy_Srsi accumulates them -- the symmetrization pairs
  // (r, s) with (s, r), which can sit in different batches, so no multiplier
  // is known until every slab has been walked. v^2 c doubles; the slabs, not
  // this, are what scale with the active space. The per-direction TANGENT of
  // the same tables rides along, for the same reason: the multipliers it is
  // finished with are not known until the walk is over.
  const DeviceTensor normT = DeviceTensor::zeros({nvirt, nvirt, ncore}, dr);
  const DeviceTensor hT = DeviceTensor::zeros({nvirt, nvirt, ncore}, dr);
  std::vector<DeviceTensor> normTDot, hTDot;
  for (std::size_t d = 0; d < rot.size(); ++d) {
    normTDot.push_back(rot.dirs[d].tangent ? DeviceTensor::zeros({nvirt, nvirt, ncore}, dr)
                                           : DeviceTensor());
    hTDot.push_back(rot.dirs[d].tangent ? DeviceTensor::zeros({nvirt, nvirt, ncore}, dr)
                                        : DeviceTensor());
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
    }
  });

  const Tensor hostNormT = downloadTensor(normT, s);
  const Tensor hostHT = downloadTensor(hT, s);
  const ScSlabMultipliers m = srsiMultipliers(hostNormT, hostHT, hostECore, hostEVirt);
  acc.norm += m.norm;
  acc.energy += m.energy;
  // The tangent, finished with the scattered multipliers over the RAW tables
  // -- the same identity that lets actReference be taken over them (a
  // perturber's weight multiplies both of its tuples, and the diagonal's
  // halving is already in the packing).
  for (std::size_t d = 0; d < rot.size(); ++d) {
    // The orbital-energy half, R_t against Deltadot over the SAME triangle
    // srsiDelta packs -- as assembleSlab does it for the other classes.
    if (rot.dirs[d].general)
      acc.rotEps[d] += dotHost(
          m.r, srsiDelta(nvirt, ncore, rot.dirs[d].epsCoreDot, rot.dirs[d].epsVirtDot));
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
    ext += dotTensors(slab, dBlocks[0], dr) + dotTensors(slabT, dBlocks[1], dr);
    // The external half of the pairing, now that this slab's D exists.
    for (std::size_t d = 0; d < rot.size(); ++d)
      acc.rotExt[d] +=
          pairSlabDots(slabDots({&slab, &slabT}, kinds, rot.dirs[d], rot.gen, dr),
                       dBlocks, dr);
  });
  acc.extContraction += 0.5 * ext;
  acc.actContraction += activeContraction(dActive, dr);
  rotationPairActive(rot, gDots, dActive, gdot, acc.rotAct, dr);

  // The RDM sweep: dm1 raw on the norm side, k27 one chain rule away.
  if (adj != nullptr) {
    adjointAddRaw(dActive, &dm1, adj->dm1, s);
    if (const DeviceTensor* m = activeAccumOf(dActive, &k27); m != nullptr)
      adjointK27(*m, ai.h1e, ai.h2e, *adj, s);
  }
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
                 ScClassGradient& acc, ScRdmAdjoint* adj, const DeviceResources& dr) {
  const int64_t nvirt = hostEVirt.size();
  if (nvirt == 0) return;
  const auto [rm2, a7] = make_a7(ai.h1e, ai.h2e, dm1, dm2, dm3, dr);
  acc.hasActive = true;
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
      std::vector<double> d;
      d.reserve(static_cast<std::size_t>((b1 - b0) * nvirt));
      for (int64_t r = b0; r < b1; ++r)
        for (int64_t ss = 0; ss < nvirt; ++ss) d.push_back(ev(r) + ev(ss));
      return d;
    };
    const std::vector<double> delta = deltaOf(hostECore, hostEVirt);
    assembleSlab(
        normTerms, hTerms, {b1 - b0, nvirt}, {&slab}, dActive,
        [&](const Tensor& n, const Tensor& h) { return plainMultipliers(n, h, delta); },
        deltaOf, rot, {RotBlock::Srs}, gDots, acc, dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
  rotationPairActive(rot, gDots, dActive, gdot, acc.rotAct, dr);

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
  if (adj != nullptr) {
    const wwrStream_t s = dr.stream();
    const int64_t norb = dm1.dims[0];
    const DeviceTensor delta = deviceEye(norb, dr);
    const DeviceTensor mRm2 = DeviceTensor::zeros(rm2.dims, dr);
    const DeviceTensor mRm3 =
        DeviceTensor::zeros({norb, norb, norb, norb, norb, norb}, dr);
    if (const DeviceTensor* m = activeAccumOf(dActive, &rm2); m != nullptr)
      addTensorInto(*m, mRm2, s);
    if (const DeviceTensor* m = activeAccumOf(dActive, &a7); m != nullptr)
      adjointA7(*m, ai.h1e, ai.h2e, mRm2, mRm3, s);
    adjointRm3(mRm3, delta, mRm2, *adj, s);
    adjointRm2(mRm2, delta, *adj, s);
  }
}

// --- Sij: a core pair -------------------------------------------------------

void gradientSij(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                 const Tensor& hostECore, const Tensor& hostEVirt, const DeviceTensor& dm1,
                 const DeviceTensor& dm2, const DeviceTensor& dm3, const ScRotation& rot,
                 ScClassGradient& acc, ScRdmAdjoint* adj, const DeviceResources& dr) {
  const int64_t ncore = hostECore.size();
  if (ncore == 0) return;
  const DeviceTensor hdm1 = make_hdm1(dm1, dr);
  const DeviceTensor hdm2 = make_hdm2(dm1, dm2, dr);
  const DeviceTensor hdm3 = make_hdm3(dm1, dm2, dm3, hdm1, hdm2, dr);
  const DeviceTensor a9 = make_a9(ai.h1e, ai.h2e, hdm1, hdm2, hdm3, dr);
  acc.hasActive = true;
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
      std::vector<double> d;
      d.reserve(static_cast<std::size_t>((b1 - b0) * ncore));
      for (int64_t i = b0; i < b1; ++i)
        for (int64_t j = 0; j < ncore; ++j) d.push_back(-(ec(i) + ec(j)));
      return d;
    };
    const std::vector<double> delta = deltaOf(hostECore, hostEVirt);
    assembleSlab(
        normTerms, hTerms, {b1 - b0, ncore}, {&slab}, dActive,
        [&](const Tensor& n, const Tensor& h) { return plainMultipliers(n, h, delta); },
        deltaOf, rot, {RotBlock::Sij}, gDots, acc, dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
  rotationPairActive(rot, gDots, dActive, gdot, acc.rotAct, dr);

  // The RDM sweep, reverse-topologically: a9 reads hdm2 and hdm3, hdm3 reads
  // hdm2 as well, so a9, then hdm3, then hdm2 -- whose accumulator is seeded
  // from the walk, where it is this class's own metric. (hdm1 is passed to
  // make_hdm3 and make_a9 and read by neither, so no term contracts it and it
  // gets no accumulator.)
  if (adj != nullptr) {
    const wwrStream_t s = dr.stream();
    const int64_t norb = dm1.dims[0];
    const DeviceTensor delta = deviceEye(norb, dr);
    const DeviceTensor mHdm2 = DeviceTensor::zeros(hdm2.dims, dr);
    const DeviceTensor mHdm3 = DeviceTensor::zeros(hdm3.dims, dr);
    if (const DeviceTensor* m = activeAccumOf(dActive, &hdm2); m != nullptr)
      addTensorInto(*m, mHdm2, s);
    if (const DeviceTensor* m = activeAccumOf(dActive, &a9); m != nullptr)
      adjointA9(*m, ai.h1e, ai.h2e, mHdm2, mHdm3, s);
    adjointHdm3(mHdm3, delta, mHdm2, *adj, s);
    adjointHdm2(mHdm2, delta, *adj, dr);
  }
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
                 ScClassGradient& acc, ScRdmAdjoint* adj,
                 const DeviceResources& dr) {
  const int64_t ncore = hostECore.size();
  if (ncore == 0) return;
  const DeviceTensor a12 = make_a12(ai.h1e, ai.h2e, dm1, dm2, dm3, dr);
  const DeviceTensor a13 = make_a13(ai.h1e, ai.h2e, dm1, dm2, dm3, dr);
  acc.hasActive = true;
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
      std::vector<double> d;
      d.reserve(static_cast<std::size_t>(ncore * (b1 - b0)));
      for (int64_t i = 0; i < ncore; ++i)
        for (int64_t r = b0; r < b1; ++r) d.push_back(ev(r) - ec(i));
      return d;
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
        {RotBlock::Sir1, RotBlock::Sir2, RotBlock::H1eSir, RotBlock::V2tr}, gDots, acc, dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
  rotationPairActive(rot, gDots, dActive, gdot, acc.rotAct, dr);

  // The RDM sweep: dm1 and dm2 raw on the norm side, a12 and a13 one chain
  // rule away. The hoisted v2tr block is an INTEGRAL trace, not an RDM, so
  // nothing of it reaches here.
  if (adj != nullptr) {
    const wwrStream_t s = dr.stream();
    const int64_t norb = dm1.dims[0];
    const DeviceTensor delta = deviceEye(norb, dr);
    adjointAddRaw(dActive, &dm1, adj->dm1, s);
    adjointAddRaw(dActive, &dm2, adj->dm2, s);
    if (const DeviceTensor* m = activeAccumOf(dActive, &a12); m != nullptr)
      adjointA12(*m, ai.h1e, ai.h2e, *adj, s);
    if (const DeviceTensor* m = activeAccumOf(dActive, &a13); m != nullptr)
      adjointA13(*m, ai.h1e, ai.h2e, delta, *adj, s);
  }
}

}  // namespace nevpt2
