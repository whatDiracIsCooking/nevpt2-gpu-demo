// nevpt2.energy:sc_hessian -- the SECOND directional derivative of the eight
// SC class energies with respect to the CI vector, Eddot(u, v)
// (energiesHessianDevice, declared in energy.cppm; docs/hessian-theory.md,
// Sec. 4.1). An internal partition: nothing here is exported, so no importer
// of nevpt2.energy sees anything but energiesHessianDevice.
//
// NO NEW CONTRACTION LOGIC, BY CONSTRUCTION -- the forward tangent's argument
// (:sc_tangent) one order up. At fixed integrals each class's per-perturber
// N_k and H_k is the external slab contracted against ONE block per term, and
// every block is linear in the RDMs up to its RDM-free terms. The RDMs are
// exactly quadratic in the CI vector (Sec. 3), so along (u, v)
//
//   N''_k[u, v] = the same chain with the block built at d^2(dm)[u, v]
//
// by the SAME tangent builders a first derivative uses -- an RDM-free term
// is constant, so it is absent from the second derivative exactly as from the
// first. So this file is :sc_tangent's per-class walks carrying FOUR RDM sets
// instead of two -- the plain one, u's tangent, v's tangent and the
// second-order one -- through the shared block builders (srBlocks,
// siBlocks, ...) and :sc_classes' own per-slab chains. Nothing here
// transcribes a subscript string.
//
// THE ONE NEW EXPRESSION is the finish's, per perturber (Sec. 4.1):
//
//   Eddot_k = P_k N''_k + Q_k H''_k
//           + (Ndot_k[v], Hdot_k[v], 0) . Hess f_k . (Ndot_k[u], Hdot_k[u], 0)
//
// the first sum normToEnergyTangent at the second-order vectors, the second
// nevpt2.energy_finish's normToEnergyHessianSandwich over #150's 3x3. The
// second needs u's and v's per-perturber tangents AT THE SAME TIME, which is
// why the walk carries all three tangent sets rather than being run twice.
//
// FROZEN ORBITAL ENERGIES ONLY: every Deltadot and Delta'' is zero here. That
// is the functional --sc-tangent licenses and the one the Euler identity
// Eddot(c, u) == Edot(u) holds in (Sec. 4.3); with the orbital energies
// responding, Delta is affine in dm1 and so has a second derivative of its
// own, which this file does not carry.
//
// COST: four copies of every class's blocks (the tangent holds two), each
// built once before the walk and freed when the class returns; four chains
// per term per slab, each slab's outputs freed at the end of its iteration,
// every free a wwrFreeAsync on the one stream behind its readers.
module nevpt2.energy:sc_hessian;

import std;
import nevpt2.energy;
import nevpt2.profile;
import :shared;      // energy_Sijrs: the one class with no RDM, hence no derivative
import :sc_classes;  // the per-slab chains, the Delta vectors and the packings
import :sc_finish;   // sliceHost
import :sc_tangent;  // ScRdms, BlockOrder and the shared block builders

namespace nevpt2 {

// One class's value, both first derivatives and the second.
struct ScHessianSums {
  double norm = 0.0;
  double energy = 0.0;
  double normDotU = 0.0;
  double energyDotU = 0.0;
  double normDotV = 0.0;
  double energyDotV = 0.0;
  double normDDot = 0.0;
  double energyDDot = 0.0;
};

void addHessianSlab(ScHessianSums& acc, const ScHessianSums& slab) {
  acc.norm += slab.norm;
  acc.energy += slab.energy;
  acc.normDotU += slab.normDotU;
  acc.energyDotU += slab.energyDotU;
  acc.normDotV += slab.normDotV;
  acc.energyDotV += slab.energyDotV;
  acc.normDDot += slab.normDDot;
  acc.energyDDot += slab.energyDDot;
}

// The four RDM sets, in the order every walk below indexes them.
constexpr std::size_t kBase = 0, kU = 1, kV = 2, kUV = 3;
constexpr std::size_t kSets = 4;
using ScRdmSets = std::array<const ScRdms*, kSets>;

// Which builders set `i` takes: the plain ones at the RDMs themselves, the
// tangent ones (RDM-free terms dropped) at every derivative set.
constexpr BlockOrder orderOf(std::size_t i) {
  return i == kBase ? BlockOrder::Value : BlockOrder::Tangent;
}

// One set's per-perturber N and H, host-side, in the class's own order.
struct PerturberVectors {
  std::vector<double> norm;
  std::vector<double> h;
};

// The finish, from one slab's (or one packed class's) four sets of vectors
// and the base Delta: normToEnergy, both first-order tangents, and Eddot's
// two sums. Delta is held fixed, so every Deltadot is zero.
ScHessianSums hessianFinish(const std::array<PerturberVectors, kSets>& v,
                            const std::vector<double>& diff) {
  const PerturberVectors& b = v[kBase];
  const std::vector<double> zeros(b.norm.size(), 0.0);
  const auto [n, e] = normToEnergy(b.norm, b.h, diff);
  const auto [nU, eU] = normToEnergyTangent(b.norm, b.h, diff, v[kU].norm, v[kU].h);
  const auto [nV, eV] = normToEnergyTangent(b.norm, b.h, diff, v[kV].norm, v[kV].h);
  // The first sum, P.N'' + Q.H'', is the tangent's expression at the
  // second-order vectors; the second is the 3x3 between u's and v's.
  const auto [nUV, eFirst] = normToEnergyTangent(b.norm, b.h, diff, v[kUV].norm, v[kUV].h);
  const double eSecond = normToEnergyHessianSandwich(b.norm, b.h, diff, v[kU].norm, v[kU].h,
                                                     zeros, v[kV].norm, v[kV].h, zeros);
  return {n, e, nU, eU, nV, eV, nUV, eFirst + eSecond};
}

// One slab's four (norm, H) device vectors read back. No synchronization
// before the downloads, for the reason nevpt2.energy:sc_finish's `finish`
// gives: each is enqueued behind the kernels that wrote it, and
// downloadTensor synchronizes the stream itself.
PerturberVectors downloadPerturbers(const DeviceTensor& norm, const DeviceTensor& h,
                          const DeviceResources& dr) {
  return {downloadTensor(norm, dr.stream()).data(), downloadTensor(h, dr.stream()).data()};
}

// --- the eight classes --------------------------------------------------------
//
// Each is its tangent_<class> twin (:sc_tangent) with four block sets in
// place of two. The slab walk, the slab lifetimes and the frees are
// unchanged.

ScHessianSums hessian_Sr(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                         const Tensor& hostEVirt, const ScRdmSets& sets, int64_t norb,
                         const DeviceResources& dr) {
  const std::array<SrBlocks, kSets> blk = {
      srBlocks(ai, *sets[0], norb, dr), srBlocks(ai, *sets[1], norb, dr),
      srBlocks(ai, *sets[2], norb, dr), srBlocks(ai, *sets[3], norb, dr)};

  ScHessianSums acc;
  forBatches(src.extent(ExtBlock::Sr), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sr, b0, b1, dr);
    const DeviceTensor& h2e_v = slab;
    DeviceTensor h1e_v = sliceAxis(ai.h1e_v_Sr, 0, b0, b1);
    std::array<PerturberVectors, kSets> v;
    for (std::size_t i = 0; i < kSets; ++i) {
      const ScRdms& r = *sets[i];
      const DeviceTensor norm = srNormSlab(h2e_v, h1e_v, r.dm1, r.dm2, r.dm3, dr);
      const DeviceTensor h = srHSlab(h2e_v, h1e_v, blk[i].a16, blk[i].a17, blk[i].a19, dr);
      v[i] = downloadPerturbers(norm, h, dr);
    }
    addHessianSlab(acc, hessianFinish(v, sliceHost(hostEVirt, b0, b1)));
  });
  return acc;
}

ScHessianSums hessian_Si(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                         const Tensor& hostECore, const ScRdmSets& sets, int64_t norb,
                         const DeviceResources& dr) {
  const std::array<SiBlocks, kSets> blk = {siBlocks(ai, *sets[0], norb, orderOf(0), dr),
                                           siBlocks(ai, *sets[1], norb, orderOf(1), dr),
                                           siBlocks(ai, *sets[2], norb, orderOf(2), dr),
                                           siBlocks(ai, *sets[3], norb, orderOf(3), dr)};

  ScHessianSums acc;
  forBatches(src.extent(ExtBlock::Si), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Si, b0, b1, dr);
    const DeviceTensor& h2e_v = slab;
    DeviceTensor h1e_v = sliceAxis(ai.h1e_v_Si, 1, b0, b1);
    std::array<PerturberVectors, kSets> v;
    for (std::size_t i = 0; i < kSets; ++i) {
      const SiBlocks& b = blk[i];
      const DeviceTensor norm = siNormSlab(h2e_v, h1e_v, b.dm1_h, b.dm2_h, b.dm3_h, dr);
      const DeviceTensor h = siHSlab(h2e_v, h1e_v, b.a22, b.a23, b.a25, dr);
      v[i] = downloadPerturbers(norm, h, dr);
    }
    addHessianSlab(acc, hessianFinish(v, siDiff(hostECore, b0, b1)));
  });
  return acc;
}

// Sijrs reads no RDM: its value is energy_Sijrs', and at fixed orbital
// energies every derivative of it is exactly zero.
ScHessianSums hessian_Sijrs(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                            const DeviceResources& dr) {
  const auto [n, e] = energy_Sijrs(ai, src, batch, dr);
  ScHessianSums out;
  out.norm = n;
  out.energy = e;
  return out;
}

// The packed classes: one TriVectors per set out of the same (linear)
// packing, whose base `diff` is the one Delta.
ScHessianSums hessianFinishPacked(const std::array<TriVectors, kSets>& t) {
  std::array<PerturberVectors, kSets> v;
  for (std::size_t i = 0; i < kSets; ++i) v[i] = {t[i].norm, t[i].h};
  return hessianFinish(v, t[kBase].diff);
}

ScHessianSums hessian_Sijr(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                           const Tensor& hostECore, const Tensor& hostEVirt,
                           const ScRdmSets& sets, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const std::array<SijrBlocks, kSets> blk = {
      sijrBlocks(ai, *sets[0], orderOf(0), dr), sijrBlocks(ai, *sets[1], orderOf(1), dr),
      sijrBlocks(ai, *sets[2], orderOf(2), dr), sijrBlocks(ai, *sets[3], orderOf(3), dr)};

  ScHessianSums acc;
  forBatches(src.extent(ExtBlock::Sijr), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sijr, b0, b1, dr);  // (nr, a, c, c)
    const DeviceTensor& h2e_v = slab;
    std::array<TriVectors, kSets> t;
    for (std::size_t i = 0; i < kSets; ++i) {
      const DeviceTensor normT = sijrSlab(h2e_v, blk[i].hdm1, dr);
      const DeviceTensor hT = sijrSlab(h2e_v, blk[i].a3, dr);
      t[i] = sijrPack(downloadTensor(normT, s), downloadTensor(hT, s), hostECore, hostEVirt, b0);
    }
    addHessianSlab(acc, hessianFinishPacked(t));
  });
  return acc;
}

ScHessianSums hessian_Srsi(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                           const Tensor& hostECore, const Tensor& hostEVirt,
                           const ScRdmSets& sets, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t nvirt = hostEVirt.size(), ncore = hostECore.size();
  // k27 has no RDM-free term: the same builder at every set.
  const std::array<DeviceTensor, kSets> k27 = {
      make_k27(ai.h1e, ai.h2e, sets[0]->dm1, sets[0]->dm2, dr),
      make_k27(ai.h1e, ai.h2e, sets[1]->dm1, sets[1]->dm2, dr),
      make_k27(ai.h1e, ai.h2e, sets[2]->dm1, sets[2]->dm2, dr),
      make_k27(ai.h1e, ai.h2e, sets[3]->dm1, sets[3]->dm2, dr)};

  // Eight whole (v, v, c) outputs, for the reason energy_Srsi has two:
  // srsiPack pairs (r, s) with (s, r), which can sit in different slabs.
  std::array<DeviceTensor, kSets> normT, hT;
  for (std::size_t i = 0; i < kSets; ++i) {
    normT[i] = DeviceTensor::zeros({nvirt, nvirt, ncore}, dr);
    hT[i] = DeviceTensor::zeros({nvirt, nvirt, ncore}, dr);
  }
  forBatches(src.extent(ExtBlock::Srsi), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Srsi, b0, b1, dr);    // h[r_batch, s, i, a]
    DeviceTensor slabT = src.slab(ExtBlock::SrsiT, b0, b1, dr);  // h[s, r_batch, i, a]
    for (std::size_t i = 0; i < kSets; ++i) {
      DeviceTensor normRows = sliceAxis(normT[i], 0, b0, b1);
      DeviceTensor hRows = sliceAxis(hT[i], 0, b0, b1);
      srsiAccumSlab(slab, slabT, sets[i]->dm1, normRows, s);
      srsiAccumSlab(slab, slabT, k27[i], hRows, s);
    }
  });

  std::array<TriVectors, kSets> t;
  for (std::size_t i = 0; i < kSets; ++i)
    t[i] = srsiPack(downloadTensor(normT[i], s), downloadTensor(hT[i], s), hostECore, hostEVirt);
  return hessianFinishPacked(t);
}

ScHessianSums hessian_Srs(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                          const Tensor& hostEVirt, const ScRdmSets& sets,
                          const DeviceResources& dr) {
  if (hostEVirt.size() == 0) return {};
  // make_a7 has no RDM-free term: the same builder at every set.
  std::array<std::pair<DeviceTensor, DeviceTensor>, kSets> blk;
  for (std::size_t i = 0; i < kSets; ++i)
    blk[i] = make_a7(ai.h1e, ai.h2e, sets[i]->dm1, sets[i]->dm2, sets[i]->dm3, dr);

  ScHessianSums acc;
  forBatches(src.extent(ExtBlock::Srs), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Srs, b0, b1, dr);  // (nr, v, a, a)
    const DeviceTensor& h2e_v = slab;
    std::array<PerturberVectors, kSets> v;
    for (std::size_t i = 0; i < kSets; ++i) {
      const DeviceTensor norm = srsNormSlab(h2e_v, blk[i].first, dr);
      const DeviceTensor h = srsHSlab(h2e_v, blk[i].second, dr);
      v[i] = downloadPerturbers(norm, h, dr);
    }
    addHessianSlab(acc, hessianFinish(v, srsDiff(hostEVirt, b0, b1)));
  });
  return acc;
}

ScHessianSums hessian_Sij(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                          const Tensor& hostECore, const ScRdmSets& sets,
                          const DeviceResources& dr) {
  if (hostECore.size() == 0) return {};
  const std::array<SijBlocks, kSets> blk = {
      sijBlocks(ai, *sets[0], orderOf(0), dr), sijBlocks(ai, *sets[1], orderOf(1), dr),
      sijBlocks(ai, *sets[2], orderOf(2), dr), sijBlocks(ai, *sets[3], orderOf(3), dr)};

  ScHessianSums acc;
  forBatches(src.extent(ExtBlock::Sij), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sij, b0, b1, dr);  // (a, a, ni, c)
    const DeviceTensor& h2e_v = slab;
    std::array<PerturberVectors, kSets> v;
    for (std::size_t i = 0; i < kSets; ++i) {
      const DeviceTensor norm = sijSlab(h2e_v, blk[i].hdm2, dr);
      const DeviceTensor h = sijSlab(h2e_v, blk[i].a9, dr);
      v[i] = downloadPerturbers(norm, h, dr);
    }
    addHessianSlab(acc, hessianFinish(v, sijDiff(hostECore, b0, b1)));
  });
  return acc;
}

ScHessianSums hessian_Sir(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                          const Tensor& hostECore, const Tensor& hostEVirt,
                          const ScRdmSets& sets, const DeviceResources& dr) {
  const std::array<SirBlocks, kSets> blk = {sirBlocks(ai, *sets[0], dr),
                                            sirBlocks(ai, *sets[1], dr),
                                            sirBlocks(ai, *sets[2], dr),
                                            sirBlocks(ai, *sets[3], dr)};

  ScHessianSums acc;
  forBatches(src.extent(ExtBlock::Sir1), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab1 = src.slab(ExtBlock::Sir1, b0, b1, dr);  // (nr, a, c, a)
    DeviceTensor slab2 = src.slab(ExtBlock::Sir2, b0, b1, dr);  // (nr, a, a, c)
    DeviceTensor h1e_v = sliceAxis(ai.h1e_v_Sir, 0, b0, b1);   // (nr, c)
    std::array<PerturberVectors, kSets> v;
    for (std::size_t i = 0; i < kSets; ++i) {
      const ScRdms& r = *sets[i];
      // The one-electron norm term reads no RDM: in the value, out of every
      // derivative.
      const DeviceTensor norm = sirNormSlab(slab1, slab2, h1e_v, r.dm1, r.dm2,
                                            /*withOneElectronTerm=*/i == kBase, dr);
      const DeviceTensor h = sirHSlab(slab1, slab2, blk[i].a12, blk[i].a13, dr);
      v[i] = downloadPerturbers(norm, h, dr);
    }
    addHessianSlab(acc, hessianFinish(v, sirDiff(hostECore, hostEVirt, b0, b1)));
  });
  return acc;
}

// --- entry point ----------------------------------------------------------------

EnergyHessianResult energiesHessianDevice(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                          int64_t batch, const RdmTensors& rdm,
                                          const RdmTensors& dotU, const RdmTensors& dotV,
                                          const RdmTensors& ddot, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  // --profile: the launches below report in the energy table, like
  // energiesDevice's (the same einsums, run four times).
  const profile::Section profileSection(profile::kEnergy);
  const int64_t norb = rdm.dm1.dims[0];
  const Tensor hostECore = downloadTensor(ai.e_core, s);
  const Tensor hostEVirt = downloadTensor(ai.e_virt, s);
  const ScRdmSets sets = {&rdm, &dotU, &dotV, &ddot};

  // CLASSES order, as energiesDevice.
  const std::array<ScHessianSums, 8> cls = {
      hessian_Sr(ai, src, batch, hostEVirt, sets, norb, dr),
      hessian_Si(ai, src, batch, hostECore, sets, norb, dr),
      hessian_Sijrs(ai, src, batch, dr),
      hessian_Sijr(ai, src, batch, hostECore, hostEVirt, sets, dr),
      hessian_Srsi(ai, src, batch, hostECore, hostEVirt, sets, dr),
      hessian_Srs(ai, src, batch, hostEVirt, sets, dr),
      hessian_Sij(ai, src, batch, hostECore, sets, dr),
      hessian_Sir(ai, src, batch, hostECore, hostEVirt, sets, dr),
  };

  EnergyHessianResult out;
  for (std::size_t c = 0; c < cls.size(); ++c) {
    out.norms[c] = cls[c].norm;
    out.energies[c] = cls[c].energy;
    out.normDotsU[c] = cls[c].normDotU;
    out.energyDotsU[c] = cls[c].energyDotU;
    out.normDotsV[c] = cls[c].normDotV;
    out.energyDotsV[c] = cls[c].energyDotV;
    out.normDDots[c] = cls[c].normDDot;
    out.energyDDots[c] = cls[c].energyDDot;
    out.total += cls[c].energy;
    out.totalDotU += cls[c].energyDotU;
    out.totalDotV += cls[c].energyDotV;
    out.totalDDot += cls[c].energyDDot;
  }
  return out;
}

}  // namespace nevpt2
