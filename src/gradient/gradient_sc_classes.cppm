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
module nevpt2.gradient:sc_classes;

import std;
import nevpt2.gradient;
import :terms;

namespace nevpt2 {

// --- Sr: one virtual index --------------------------------------------------

void gradientSr(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                const Tensor& hostEVirt, const DeviceTensor& dm1, const DeviceTensor& dm2,
                const DeviceTensor& dm3, const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                int64_t norb, ScClassGradient& acc, const DeviceResources& dr) {
  const DeviceTensor& h1e = ai.h1e;
  const DeviceTensor& h2e = ai.h2e;
  const DeviceTensor a16 = make_a16(h1e, h2e, dm3, f3ac, f3ca, norb, dr);
  const DeviceTensor a17 = make_a17(h1e, h2e, dm2, dm3, dr);
  const DeviceTensor a19 = make_a19(h1e, h2e, dm1, dm2, dr);
  acc.hasActive = true;
  // The M / E of Eqs. 42-43 have no external index left, so one set
  // serves the whole walk; the class contracts them back at the end.
  std::vector<ScActiveAccum> dActive;

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
    const std::vector<double> delta = sliceHost(hostEVirt, b0, b1);
    assembleSlab(
        normTerms, hTerms, {b1 - b0}, {&slab, &h1e_v}, dActive,
        [&](const Tensor& n, const Tensor& h) { return plainMultipliers(n, h, delta); }, acc,
        dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
}

// --- Si: one core index (the hole-side metrics, and -e_core as Delta) -------

void gradientSi(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                const Tensor& hostECore, const DeviceTensor& dm1, const DeviceTensor& dm2,
                const DeviceTensor& dm3, const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                int64_t norb, ScClassGradient& acc, const DeviceResources& dr) {
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
    std::vector<double> delta = sliceHost(hostECore, b0, b1);
    for (double& x : delta) x = -x;
    assembleSlab(
        normTerms, hTerms, {b1 - b0}, {&slab, &h1e_v}, dActive,
        [&](const Tensor& n, const Tensor& h) { return plainMultipliers(n, h, delta); }, acc,
        dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
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
void gradientSijrs(IntegralSource& src, int64_t batch, const Tensor& hostECore,
                   const Tensor& hostEVirt, ScClassGradient& acc, const DeviceResources& dr) {
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
    std::vector<double> delta;
    delta.reserve(static_cast<std::size_t>(hostNorm.size()));
    for (int64_t i = b0; i < b1; ++i)
      for (int64_t a = 0; a < nvirt; ++a)
        for (int64_t j = 0; j < ncore; ++j)
          for (int64_t b = 0; b < nvirt; ++b)
            delta.push_back(hostEVirt(a) + hostEVirt(b) - hostECore(i) - hostECore(j));
    const std::vector<double> noH(delta.size(), 0.0);
    const ScMultipliers m = scMultipliers(hostNorm.data(), noH, delta);
    acc.norm += m.norm;
    acc.energy += m.energy;

    const DeviceTensor p = uploadTensor(Tensor::fromFlat(hostNorm.dims(), m.p), dr);
    const DeviceTensor d = deviceEinsumNew("iajb,iajb->iajb", {&p, &theta}, 1.0, dr);
    deviceEinsumAccum("iajb,iajb->iajb", {&p, &slab}, d, 2.0, s);
    deviceEinsumAccum("ibja,ibja->iajb", {&p, &slab}, d, -1.0, s);
    acc.extContraction += 0.5 * dotTensors(slab, d, dr);
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
  std::vector<double> normTri, hTri, diffTri;
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t j = 0; j < ncore; ++j)
      for (int64_t i = j; i < ncore; ++i) {
        normTri.push_back(normSym(r, j, i));
        hTri.push_back(hSym(r, j, i));
        diffTri.push_back(hostEVirt(b0 + r) - hostECore(j) - hostECore(i));
      }
  const ScMultipliers m = scMultipliers(normTri, hTri, diffTri);
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
  return ScSlabMultipliers{std::move(p), std::move(q), m.norm, m.energy};
}

void gradientSijr(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                  const Tensor& hostECore, const Tensor& hostEVirt, const DeviceTensor& dm1,
                  const DeviceTensor& dm2, ScClassGradient& acc, const DeviceResources& dr) {
  const int64_t ncore = hostECore.size();
  if (ncore == 0) return;
  const DeviceTensor hdm1 = make_hdm1(dm1, dr);
  const DeviceTensor a3 = make_a3(ai.h1e, ai.h2e, dm1, dm2, hdm1, dr);
  acc.hasActive = true;
  // The M / E of Eqs. 42-43 have no external index left, so one set
  // serves the whole walk; the class contracts them back at the end.
  std::vector<ScActiveAccum> dActive;

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
        acc, dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
}

// --- Srsi: a symmetrized virtual pair and a core index ----------------------

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
  std::vector<double> normTri, hTri, diffTri;
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t ss = r; ss < nvirt; ++ss)
      for (int64_t i = 0; i < ncore; ++i) {
        normTri.push_back(normSym(r, ss, i));
        hTri.push_back(hSym(r, ss, i));
        diffTri.push_back(hostEVirt(r) + hostEVirt(ss) - hostECore(i));
      }
  const ScMultipliers m = scMultipliers(normTri, hTri, diffTri);
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
  return ScSlabMultipliers{std::move(p), std::move(q), m.norm, m.energy};
}

void gradientSrsi(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                  const Tensor& hostECore, const Tensor& hostEVirt, const DeviceTensor& dm1,
                  const DeviceTensor& dm2, ScClassGradient& acc, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t nvirt = hostEVirt.size(), ncore = hostECore.size();
  if (nvirt == 0 || ncore == 0) return;
  const DeviceTensor k27 = make_k27(ai.h1e, ai.h2e, dm1, dm2, dr);
  acc.hasActive = true;

  // Pass 1: the raw per-tuple (r, s, i) norm and H over the WHOLE virtual
  // pair, exactly as energy_Srsi accumulates them -- the symmetrization pairs
  // (r, s) with (s, r), which can sit in different batches, so no multiplier
  // is known until every slab has been walked. v^2 c doubles; the slabs, not
  // this, are what scale with the active space.
  const DeviceTensor normT = DeviceTensor::zeros({nvirt, nvirt, ncore}, dr);
  const DeviceTensor hT = DeviceTensor::zeros({nvirt, nvirt, ncore}, dr);
  forBatches(src.extent(ExtBlock::Srsi), batch, [&](int64_t b0, int64_t b1) {
    const DeviceTensor slab = src.slab(ExtBlock::Srsi, b0, b1, dr);
    const DeviceTensor slabT = src.slab(ExtBlock::SrsiT, b0, b1, dr);
    const DeviceTensor normRows = sliceAxis(normT, 0, b0, b1);
    const DeviceTensor hRows = sliceAxis(hT, 0, b0, b1);
    deviceEinsumAccum("rsip,rsia,pa->rsi", {&slab, &slab, &dm1}, normRows, 2.0, s);
    deviceEinsumAccum("rsip,sria,pa->rsi", {&slab, &slabT, &dm1}, normRows, -1.0, s);
    deviceEinsumAccum("rsip,rsia,pa->rsi", {&slab, &slab, &k27}, hRows, 2.0, s);
    deviceEinsumAccum("rsip,sria,pa->rsi", {&slab, &slabT, &k27}, hRows, -1.0, s);
  });

  const Tensor hostNormT = downloadTensor(normT, s);
  const Tensor hostHT = downloadTensor(hT, s);
  const ScSlabMultipliers m = srsiMultipliers(hostNormT, hostHT, hostECore, hostEVirt);
  acc.norm += m.norm;
  acc.energy += m.energy;
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
    const std::vector<ScTerm> normTerms{
        {"rsip,rsia,pa->rsi", {ga, ga, {&dm1, kScActive}}, 2.0},
        {"rsip,sria,pa->rsi", {ga, gb, {&dm1, kScActive}}, -1.0},
    };
    const std::vector<ScTerm> hTerms{
        {"rsip,rsia,pa->rsi", {ga, ga, {&k27, kScActive}}, 2.0},
        {"rsip,sria,pa->rsi", {ga, gb, {&k27, kScActive}}, -1.0},
    };
    const DeviceTensor pRows = sliceAxis(mp, 0, b0, b1);
    const DeviceTensor qRows = sliceAxis(mq, 0, b0, b1);
    std::vector<DeviceTensor> dBlocks;
    dBlocks.push_back(DeviceTensor::zeros(slab.dims, dr));
    dBlocks.push_back(DeviceTensor::zeros(slabT.dims, dr));
    accumDerivatives(normTerms, pRows, dBlocks, dActive, dr);
    accumDerivatives(hTerms, qRows, dBlocks, dActive, dr);
    ext += dotTensors(slab, dBlocks[0], dr) + dotTensors(slabT, dBlocks[1], dr);
  });
  acc.extContraction += 0.5 * ext;
  acc.actContraction += activeContraction(dActive, dr);
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
                 const Tensor& hostEVirt, const DeviceTensor& dm1, const DeviceTensor& dm2,
                 const DeviceTensor& dm3, ScClassGradient& acc, const DeviceResources& dr) {
  const int64_t nvirt = hostEVirt.size();
  if (nvirt == 0) return;
  const auto [rm2, a7] = make_a7(ai.h1e, ai.h2e, dm1, dm2, dm3, dr);
  acc.hasActive = true;
  // The M / E of Eqs. 42-43 have no external index left, so one set
  // serves the whole walk; the class contracts them back at the end.
  std::vector<ScActiveAccum> dActive;

  forBatches(src.extent(ExtBlock::Srs), batch, [&](int64_t b0, int64_t b1) {
    const DeviceTensor slab = src.slab(ExtBlock::Srs, b0, b1, dr);  // (nr, v, a, a)
    const ScOperand g{&slab, 0};
    const std::vector<ScTerm> normTerms{
        {"rsqp,rsba,pqba->rs", {g, g, {&rm2, kScActive}}, 0.5},
    };
    const std::vector<ScTerm> hTerms{
        {"rsqp,rsba,pqab->rs", {g, g, {&a7, kScActive}}, 0.5},
    };
    std::vector<double> delta;
    delta.reserve(static_cast<std::size_t>((b1 - b0) * nvirt));
    for (int64_t r = b0; r < b1; ++r)
      for (int64_t ss = 0; ss < nvirt; ++ss) delta.push_back(hostEVirt(r) + hostEVirt(ss));
    assembleSlab(
        normTerms, hTerms, {b1 - b0, nvirt}, {&slab}, dActive,
        [&](const Tensor& n, const Tensor& h) { return plainMultipliers(n, h, delta); }, acc,
        dr);
  });
  acc.actContraction += activeContraction(dActive, dr);

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
}

// --- Sij: a core pair -------------------------------------------------------

void gradientSij(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                 const Tensor& hostECore, const DeviceTensor& dm1, const DeviceTensor& dm2,
                 const DeviceTensor& dm3, ScClassGradient& acc, const DeviceResources& dr) {
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

  forBatches(src.extent(ExtBlock::Sij), batch, [&](int64_t b0, int64_t b1) {
    const DeviceTensor slab = src.slab(ExtBlock::Sij, b0, b1, dr);  // (a, a, ni, c)
    const ScOperand g{&slab, 0};
    const std::vector<ScTerm> normTerms{
        {"qpij,baij,pqab->ij", {g, g, {&hdm2, kScActive}}, 0.5},
    };
    const std::vector<ScTerm> hTerms{
        {"qpij,baij,pqab->ij", {g, g, {&a9, kScActive}}, 0.5},
    };
    std::vector<double> delta;
    delta.reserve(static_cast<std::size_t>((b1 - b0) * ncore));
    for (int64_t i = b0; i < b1; ++i)
      for (int64_t j = 0; j < ncore; ++j) delta.push_back(-(hostECore(i) + hostECore(j)));
    assembleSlab(
        normTerms, hTerms, {b1 - b0, ncore}, {&slab}, dActive,
        [&](const Tensor& n, const Tensor& h) { return plainMultipliers(n, h, delta); }, acc,
        dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
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
                 const DeviceTensor& dm2, const DeviceTensor& dm3, ScClassGradient& acc,
                 const DeviceResources& dr) {
  const int64_t ncore = hostECore.size();
  if (ncore == 0) return;
  const DeviceTensor a12 = make_a12(ai.h1e, ai.h2e, dm1, dm2, dm3, dr);
  const DeviceTensor a13 = make_a13(ai.h1e, ai.h2e, dm1, dm2, dm3, dr);
  acc.hasActive = true;
  // The M / E of Eqs. 42-43 have no external index left, so one set
  // serves the whole walk; the class contracts them back at the end.
  std::vector<ScActiveAccum> dActive;

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
    std::vector<double> delta;  // (c, nr), the terms' own output order
    delta.reserve(static_cast<std::size_t>(ncore * (b1 - b0)));
    for (int64_t i = 0; i < ncore; ++i)
      for (int64_t r = b0; r < b1; ++r) delta.push_back(hostEVirt(r) - hostECore(i));
    assembleSlab(
        normTerms, hTerms, {ncore, b1 - b0}, {&slab1, &slab2, &h1e_v, &v2tr}, dActive,
        [&](const Tensor& n, const Tensor& h) { return plainMultipliers(n, h, delta); }, acc,
        dr);
  });
  acc.actContraction += activeContraction(dActive, dr);
}

}  // namespace nevpt2
