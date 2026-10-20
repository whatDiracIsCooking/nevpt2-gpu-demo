// nevpt2.energy:sc_tangent -- the FORWARD TANGENT of the eight SC class
// energies with respect to the RDMs (energiesTangentDevice, declared in
// energy.cppm), and the three tangent-only metric builders it needs.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :sc_tangent;` and from no importer of nevpt2.energy --
// which sees only energiesTangentDevice, this partition's one deliverable.
//
// NO NEW CONTRACTION LOGIC, BY CONSTRUCTION. At fixed integrals each class's
// per-perturber norm and Dyall-Hamiltonian expectation value is a quadratic
// form in that class's external integrals and LINEAR in the one active-space
// block per term it contracts them against (nevpt2.energy's make_*). So, with
// a direction (dm1_dot, dm2_dot, dm3_dot, f3ac_dot, f3ca_dot) in RDM space,
//
//   N_dot = the same einsum chain with the metric replaced by its tangent
//   H_dot = the same einsum chain with the Dyall block replaced by its tangent
//
// and each tangent block is the same make_* called with the tangent RDMs --
// every one of them is linear in the RDMs too. This file therefore reuses
// nevpt2.energy:sc_classes' per-slab chains (`*NormSlab` / `*HSlab`,
// `sijrSlab`, `srsiAccumSlab`, `sijSlab`), its Delta vectors (`*Diff`, which
// carry no RDM and so are shared unchanged) and its host packings (`sijrPack`
// / `srsiPack`, linear in what they read). Nothing here transcribes a
// subscript string.
//
// THREE KINDS OF TERM NEED CARE, and all three are RDM-FREE terms -- constant
// as the RDMs move, so absent from the derivative:
//
//   * make_hdm1's `2 delta` (and energy_Si's dm1_h, the same expression),
//     make_hdm2's two delta-delta terms and make_a25's `2 h1e^T`: dropped by
//     make_hdm1_dot / make_hdm2_dot / make_a25_dot below, the only builders
//     this file writes. Every other make_* is called with tangent arguments
//     as it stands -- including make_hdm3, make_a3 and make_a9, which are
//     linear in the (affine) hdm1/hdm2/hdm3 they take, so handing them the
//     tangent intermediates is exactly right.
//   * Sir's purely one-electron norm term (`ri,ri->ir`): sirNormSlab's
//     `withOneElectronTerm` is false in the tangent.
//   * Sijrs, the MP2-like class, contracts no RDM at all, so its norm and
//     energy tangents are exactly zero -- not approximately.
//
// THE ONE ACTUAL DERIVATIVE is the scalar finish's, nevpt2.energy_finish's
// normToEnergyTangent (Park, JCTC 2019, Eqs. 36-37): with the base N, H and
// Delta from the same walk, Edot = sum_k (P_k Ndot_k + Q_k Hdot_k). And, when
// the caller hands in the orbital energies' tangent (`epsDot`, #87), its Delta
// derivative too: + sum_k R_k Deltadot_k (Eq. 38, normToEnergyDeltaTangent),
// Deltadot being each class's own Delta vector read at that tangent -- the
// same *Diff / pack functions, which are linear and homogeneous in the
// orbital energies. Sijrs, whose tangent is otherwise zero, then needs its
// per-tuple norms, which tangent_Sijrs builds for that case alone.
//
// COST: one extra tangent block per metric and Dyall block (held beside its
// base, so the peak is about twice a class's intermediates) and one extra
// contraction per class term. The base N and H are computed here too, not
// taken from energiesDevice: the finish's derivative needs them per perturber.
module nevpt2.energy:sc_tangent;

import std;
import nevpt2.energy;
import nevpt2.profile;
import :shared;      // energy_Sijrs: the one class with no RDM, hence no tangent
import :sc_classes;  // the per-slab chains, the Delta vectors and the packings
import :sc_finish;   // finish, sliceHost

namespace nevpt2 {

// --- the tangent-only builders ------------------------------------------------
//
// Declared in energy.cppm and exported: nevpt2.gradient's reverse-over-forward
// walk (#156) builds the same blocks' tangents along a CI direction and must
// drop exactly these terms.

// The tangent of make_hdm1 (and of energy_Si's dm1_h, which is the same
// expression 2 delta - dm1^T): the constant 2 delta is gone, so this is
// just -dm1_dot^T.
DeviceTensor make_hdm1_dot(const DeviceTensor& dm1Dot, const DeviceResources& dr) {
  return deviceTransposeNew(dm1Dot, {1, 0}, -1.0, dr);
}

// make_hdm2's chain with its two RDM-free terms (`il,jk` and `ik,jl`, both
// delta against delta) left out; everything else is linear in dm1/dm2 and is
// evaluated here on their tangents. Read side by side with make_hdm2.
DeviceTensor make_hdm2_dot(const DeviceTensor& dm1Dot, const DeviceTensor& dm2Dot,
                           const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = dm1Dot.dims[0];
  DeviceTensor delta = deviceEye(n, dr);
  DeviceTensor dm2b = deviceEinsumNew("ikjl->ijkl", {&dm2Dot}, 1.0, dr);
  deviceEinsumAccum("jk,il->ijkl", {&delta, &dm1Dot}, dm2b, -1.0, s);
  DeviceTensor hdm2 = deviceEinsumNew("klij->ijkl", {&dm2b}, 1.0, dr);
  deviceEinsumAccum("il,kj->ijkl", {&delta, &dm1Dot}, hdm2, 1.0, s);
  deviceEinsumAccum("jk,li->ijkl", {&delta, &dm1Dot}, hdm2, 1.0, s);
  deviceEinsumAccum("ik,lj->ijkl", {&delta, &dm1Dot}, hdm2, -2.0, s);
  deviceEinsumAccum("jl,ki->ijkl", {&delta, &dm1Dot}, hdm2, -2.0, s);
  return hdm2;
}

// make_a25's chain without its `2 h1e^T` term (the one term of any make_*
// here that reads no RDM). Read side by side with make_a25.
DeviceTensor make_a25_dot(const DeviceTensor& h1e, const DeviceTensor& h2e,
                          const DeviceTensor& dm1Dot, const DeviceTensor& dm2Dot,
                          const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor a25 = deviceEinsumNew("pi,ai->ap", {&h1e, &dm1Dot}, -1.0, dr);
  deviceEinsumAccum("pijk,jaik->ap", {&h2e, &dm2Dot}, a25, -1.0, s);
  deviceEinsumAccum("piaj,ij->ap", {&h2e, &dm1Dot}, a25, 2.0, s);
  return a25;
}

// --- the shared plumbing ------------------------------------------------------

// One set of RDMs, as every class below reads them: the plain ones, or the
// tangent ones in their place -- nevpt2.energy's exported RdmTensors, which
// energiesHessianDevice takes four of. References to tensors the caller owns.
using ScRdms = RdmTensors;

// --- the class blocks, at an RDM set or at a tangent one ----------------------
//
// Every class's metric and Dyall blocks, built once before its slab walk. At
// the plain RDMs (`BlockOrder::Value`) they are the make_* builders as
// energy_<class> calls them. At ANY tangent RDM set (`BlockOrder::Tangent`)
// -- a first derivative's, or the second derivative's (:sc_hessian) -- every
// builder is linear in the RDMs, so it is the same make_* on the tangent
// RDMs, except the three tangent-only builders above for the blocks that
// carry an RDM-free term. Shared by :sc_tangent and :sc_hessian, so the two
// cannot drift apart.
enum class BlockOrder { Value, Tangent };

struct SrBlocks {
  DeviceTensor a16, a17, a19;
};
// No RDM-free term: the same three builders at either order.
SrBlocks srBlocks(const ActiveIntegralsDevice& ai, const ScRdms& r, int64_t norb,
                  const DeviceResources& dr) {
  return {make_a16(ai.h1e, ai.h2e, r.dm3, r.f3ac, r.f3ca, norb, dr),
          make_a17(ai.h1e, ai.h2e, r.dm2, r.dm3, dr), make_a19(ai.h1e, ai.h2e, r.dm1, r.dm2, dr)};
}

struct SiBlocks {
  DeviceTensor a22, a23, a25, dm3_h, dm2_h, dm1_h;
};
// a25 is the one builder whose chain the tangent cannot reuse as it stands
// (its 2 h1e^T term reads no RDM); dm1_h's constant 2 delta is the other.
SiBlocks siBlocks(const ActiveIntegralsDevice& ai, const ScRdms& r, int64_t norb,
                  BlockOrder order, const DeviceResources& dr) {
  const bool dot = order == BlockOrder::Tangent;
  DeviceTensor dm1_h;
  if (dot) {
    dm1_h = make_hdm1_dot(r.dm1, dr);
  } else {
    const DeviceTensor delta = deviceEye(norb, dr);
    dm1_h = deviceEinsumNew("ij->ij", {&delta}, 2.0, dr);
    deviceTransposeAccum(r.dm1, {1, 0}, dm1_h, -1.0, dr.stream());
  }
  return {make_a22(ai.h1e, ai.h2e, r.dm2, r.dm3, r.f3ac, r.f3ca, norb, dr),
          make_a23(ai.h1e, ai.h2e, r.dm1, r.dm2, r.dm3, dr),
          dot ? make_a25_dot(ai.h1e, ai.h2e, r.dm1, r.dm2, dr)
              : make_a25(ai.h1e, ai.h2e, r.dm1, r.dm2, dr),
          make_dm3_h(r.dm2, r.dm3, dr),
          make_dm2_h(r.dm1, r.dm2, dr),
          std::move(dm1_h)};
}

struct SijrBlocks {
  DeviceTensor hdm1, a3;
};
// make_a3 is linear in (hdm1, dm1, dm2), so the tangent hdm1 carries it.
SijrBlocks sijrBlocks(const ActiveIntegralsDevice& ai, const ScRdms& r, BlockOrder order,
                      const DeviceResources& dr) {
  DeviceTensor hdm1 =
      order == BlockOrder::Tangent ? make_hdm1_dot(r.dm1, dr) : make_hdm1(r.dm1, dr);
  DeviceTensor a3 = make_a3(ai.h1e, ai.h2e, r.dm1, r.dm2, hdm1, dr);
  return {std::move(hdm1), std::move(a3)};
}

struct SijBlocks {
  DeviceTensor hdm1, hdm2, hdm3, a9;
};
// make_hdm3 and make_a9 are linear in the hole RDMs they take, so the tangent
// hdm1/hdm2 carry the two builders that needed a tangent of their own.
SijBlocks sijBlocks(const ActiveIntegralsDevice& ai, const ScRdms& r, BlockOrder order,
                    const DeviceResources& dr) {
  const bool dot = order == BlockOrder::Tangent;
  DeviceTensor hdm1 = dot ? make_hdm1_dot(r.dm1, dr) : make_hdm1(r.dm1, dr);
  DeviceTensor hdm2 = dot ? make_hdm2_dot(r.dm1, r.dm2, dr) : make_hdm2(r.dm1, r.dm2, dr);
  DeviceTensor hdm3 = make_hdm3(r.dm1, r.dm2, r.dm3, hdm1, hdm2, dr);
  DeviceTensor a9 = make_a9(ai.h1e, ai.h2e, hdm1, hdm2, hdm3, dr);
  return {std::move(hdm1), std::move(hdm2), std::move(hdm3), std::move(a9)};
}

struct SirBlocks {
  DeviceTensor a12, a13;
};
// No RDM-free block (Sir's one RDM-free term is in its norm chain, which
// sirNormSlab's `withOneElectronTerm` drops): the same two at either order.
SirBlocks sirBlocks(const ActiveIntegralsDevice& ai, const ScRdms& r,
                    const DeviceResources& dr) {
  return {make_a12(ai.h1e, ai.h2e, r.dm1, r.dm2, r.dm3, dr),
          make_a13(ai.h1e, ai.h2e, r.dm1, r.dm2, r.dm3, dr)};
}

// One class's value and tangent: normToEnergy's pair, and the derivative of
// each along the direction.
struct ScTangentSums {
  double norm = 0.0;
  double energy = 0.0;
  double normDot = 0.0;
  double energyDot = 0.0;
};

void addSlab(ScTangentSums& acc, const ScTangentSums& slab) {
  acc.norm += slab.norm;
  acc.energy += slab.energy;
  acc.normDot += slab.normDot;
  acc.energyDot += slab.energyDot;
}

// The finish every class but Sijr/Srsi runs: a slab's four device vectors read
// back, then normToEnergy and its tangent against the same Delta. No
// synchronization before the downloads, for the reason
// nevpt2.energy:sc_finish's `finish` gives -- each download is enqueued on
// the stream that wrote its source, behind it, and downloadTensor
// synchronizes that stream itself.
ScTangentSums tangentFinish(const DeviceTensor& norm, const DeviceTensor& h,
                            const DeviceTensor& normDot, const DeviceTensor& hDot,
                            const std::vector<double>& diff, const std::vector<double>* diffDot,
                            const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const Tensor hostNorm = downloadTensor(norm, s);
  const Tensor hostH = downloadTensor(h, s);
  const Tensor hostNormDot = downloadTensor(normDot, s);
  const Tensor hostHDot = downloadTensor(hDot, s);
  const auto [n, e] = normToEnergy(hostNorm.data(), hostH.data(), diff);
  const auto [nDot, eDot] =
      normToEnergyTangent(hostNorm.data(), hostH.data(), diff, hostNormDot.data(),
                          hostHDot.data());
  const double eDotEps =
      diffDot ? normToEnergyDeltaTangent(hostNorm.data(), hostH.data(), diff, *diffDot) : 0.0;
  return {n, e, nDot, eDot + eDotEps};
}

// The same for a class that packs its perturbers on the host: the base pair
// and the tangent pair come out of the SAME packing (it is linear), so the
// tangent's own Delta -- identical by construction -- is discarded. `eps`,
// when the orbital energies move, is the same packing read at their tangent:
// its `diff` is Deltadot, in the perturbers' own order.
ScTangentSums tangentFinishPacked(const TriVectors& base, const TriVectors& dot,
                                  const TriVectors* eps) {
  const auto [n, e] = normToEnergy(base.norm, base.h, base.diff);
  const auto [nDot, eDot] =
      normToEnergyTangent(base.norm, base.h, base.diff, dot.norm, dot.h);
  const double eDotEps =
      eps ? normToEnergyDeltaTangent(base.norm, base.h, base.diff, eps->diff) : 0.0;
  return {n, e, nDot, eDot + eDotEps};
}

// A class's Delta read at the orbital energies' tangent, or nothing when they
// are held fixed: `fn(core, virt)` is the class's own Delta construction, and
// Delta is linear and homogeneous in the orbital energies, so that IS
// Deltadot.
template <class DiffFn>
std::optional<std::vector<double>> diffDotOf(const OrbitalEnergyDot* eps, DiffFn&& fn) {
  if (eps == nullptr) return std::nullopt;
  return fn(eps->core, eps->virt);
}

// The pointer tangentFinish takes.
const std::vector<double>* ptrOf(const std::optional<std::vector<double>>& v) {
  return v ? &*v : nullptr;
}

// --- the eight classes --------------------------------------------------------
//
// Each is its energy_<class> twin with a second set of blocks built from the
// tangent RDMs and a second contraction per term. The slab walk, the slab
// lifetimes and the frees are unchanged: a class's blocks (both sets) are
// built once before the walk and freed when it returns, a slab's outputs at
// the end of its iteration, every free a wwrFreeAsync on the one stream
// behind the kernels that read it.

ScTangentSums tangent_Sr(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                         const Tensor& hostEVirt, const ScRdms& r, const ScRdms& d,
                         const OrbitalEnergyDot* eps, int64_t norb, const DeviceResources& dr) {
  // The same three builders on the tangent RDMs: each is linear in them.
  const SrBlocks base = srBlocks(ai, r, norb, dr);
  const SrBlocks dot = srBlocks(ai, d, norb, dr);

  ScTangentSums acc;
  forBatches(src.extent(ExtBlock::Sr), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sr, b0, b1, dr);
    const DeviceTensor& h2e_v = slab;
    DeviceTensor h1e_v = sliceAxis(ai.h1e_v_Sr, 0, b0, b1);

    DeviceTensor ener = srHSlab(h2e_v, h1e_v, base.a16, base.a17, base.a19, dr);
    DeviceTensor norm = srNormSlab(h2e_v, h1e_v, r.dm1, r.dm2, r.dm3, dr);
    DeviceTensor enerDot = srHSlab(h2e_v, h1e_v, dot.a16, dot.a17, dot.a19, dr);
    DeviceTensor normDot = srNormSlab(h2e_v, h1e_v, d.dm1, d.dm2, d.dm3, dr);

    const auto diffDot =
        diffDotOf(eps, [&](const Tensor&, const Tensor& ev) { return sliceHost(ev, b0, b1); });
    addSlab(acc, tangentFinish(norm, ener, normDot, enerDot, sliceHost(hostEVirt, b0, b1),
                               ptrOf(diffDot), dr));
  });
  return acc;
}

ScTangentSums tangent_Si(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                         const Tensor& hostECore, const ScRdms& r, const ScRdms& d,
                         const OrbitalEnergyDot* eps, int64_t norb, const DeviceResources& dr) {
  // siBlocks says which two blocks the tangent cannot build as they stand.
  const SiBlocks base = siBlocks(ai, r, norb, BlockOrder::Value, dr);
  const SiBlocks dot = siBlocks(ai, d, norb, BlockOrder::Tangent, dr);

  ScTangentSums acc;
  forBatches(src.extent(ExtBlock::Si), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Si, b0, b1, dr);
    const DeviceTensor& h2e_v = slab;
    DeviceTensor h1e_v = sliceAxis(ai.h1e_v_Si, 1, b0, b1);

    DeviceTensor ener = siHSlab(h2e_v, h1e_v, base.a22, base.a23, base.a25, dr);
    DeviceTensor norm = siNormSlab(h2e_v, h1e_v, base.dm1_h, base.dm2_h, base.dm3_h, dr);
    DeviceTensor enerDot = siHSlab(h2e_v, h1e_v, dot.a22, dot.a23, dot.a25, dr);
    DeviceTensor normDot = siNormSlab(h2e_v, h1e_v, dot.dm1_h, dot.dm2_h, dot.dm3_h, dr);

    const auto diffDot =
        diffDotOf(eps, [&](const Tensor& ec, const Tensor&) { return siDiff(ec, b0, b1); });
    addSlab(acc, tangentFinish(norm, ener, normDot, enerDot, siDiff(hostECore, b0, b1),
                               ptrOf(diffDot), dr));
  });
  return acc;
}

// Sijrs reads no RDM: the base pair is energy_Sijrs', and at fixed orbital
// energies both tangents are exactly zero. Its Delta is four of them, though,
// so when they move (`eps`) its energy tangent is sum_t R_t Deltadot_t -- the
// one place this class needs its per-tuple norms on the host, which
// energy_Sijrs' device reduction never forms: theta = 2 g - g(a <-> b) and
// N = g theta, the kernel's own two lines, as einsums over the same slab
// (the adjoint side, nevpt2.gradient's gradientSijrs, builds them the same
// way). H is identically zero here.
ScTangentSums tangent_Sijrs(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                            const OrbitalEnergyDot* eps, const DeviceResources& dr) {
  const auto [n, e] = energy_Sijrs(ai, src, batch, dr);
  double eDot = 0.0;
  const int64_t ncore = ai.e_core.size(), nvirt = ai.e_virt.size();
  if (eps != nullptr && ncore > 0 && nvirt > 0) {
    const wwrStream_t s = dr.stream();
    const Tensor hostECore = downloadTensor(ai.e_core, s);
    const Tensor hostEVirt = downloadTensor(ai.e_virt, s);
    forBatches(src.extent(ExtBlock::Sijrs), batch, [&](int64_t b0, int64_t b1) {
      DeviceTensor slab = src.slab(ExtBlock::Sijrs, b0, b1, dr);  // (ni, v, c, v)
      const DeviceTensor theta = deviceTransposeNew(slab, {0, 3, 2, 1}, -1.0, dr);
      deviceEinsumAccum("iajb->iajb", {&slab}, theta, 2.0, s);
      const DeviceTensor norm = deviceEinsumNew("iajb,iajb->iajb", {&slab, &theta}, 1.0, dr);
      // No sync before the download: it is enqueued behind the einsums that
      // wrote `norm`, and downloadTensor synchronizes the stream itself.
      const Tensor hostNorm = downloadTensor(norm, s);
      const std::vector<double> noH(static_cast<std::size_t>(hostNorm.size()), 0.0);
      eDot += normToEnergyDeltaTangent(hostNorm.data(), noH,
                                       sijrsDiff(hostECore, hostEVirt, b0, b1),
                                       sijrsDiff(eps->core, eps->virt, b0, b1));
    });
  }
  return {n, e, 0.0, eDot};
}

ScTangentSums tangent_Sijr(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                           const Tensor& hostECore, const Tensor& hostEVirt, const ScRdms& r,
                           const ScRdms& d, const OrbitalEnergyDot* eps,
                           const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const SijrBlocks base = sijrBlocks(ai, r, BlockOrder::Value, dr);
  const SijrBlocks dot = sijrBlocks(ai, d, BlockOrder::Tangent, dr);

  ScTangentSums acc;
  forBatches(src.extent(ExtBlock::Sijr), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sijr, b0, b1, dr);  // (nr, a, c, c)
    const DeviceTensor& h2e_v = slab;

    DeviceTensor normT = sijrSlab(h2e_v, base.hdm1, dr);
    DeviceTensor hT = sijrSlab(h2e_v, base.a3, dr);
    DeviceTensor normTDot = sijrSlab(h2e_v, dot.hdm1, dr);
    DeviceTensor hTDot = sijrSlab(h2e_v, dot.a3, dr);

    const Tensor hostNormT = downloadTensor(normT, s);
    const Tensor hostHT = downloadTensor(hT, s);
    const TriVectors base = sijrPack(hostNormT, hostHT, hostECore, hostEVirt, b0);
    const TriVectors dot = sijrPack(downloadTensor(normTDot, s), downloadTensor(hTDot, s),
                                    hostECore, hostEVirt, b0);
    const std::optional<TriVectors> epsPack =
        eps ? std::optional(sijrPack(hostNormT, hostHT, eps->core, eps->virt, b0))
            : std::nullopt;
    addSlab(acc, tangentFinishPacked(base, dot, epsPack ? &*epsPack : nullptr));
  });
  return acc;
}

ScTangentSums tangent_Srsi(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                           const Tensor& hostECore, const Tensor& hostEVirt, const ScRdms& r,
                           const ScRdms& d, const OrbitalEnergyDot* eps,
                           const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const DeviceTensor& h1e = ai.h1e;
  const DeviceTensor& h2e = ai.h2e;
  const int64_t nvirt = hostEVirt.size(), ncore = hostECore.size();
  DeviceTensor k27 = make_k27(h1e, h2e, r.dm1, r.dm2, dr);
  DeviceTensor k27Dot = make_k27(h1e, h2e, d.dm1, d.dm2, dr);

  // Four whole (v, v, c) outputs now, for the same reason energy_Srsi has
  // two: srsiPack's symmetrization pairs (r, s) with (s, r), which can sit in
  // different slabs.
  DeviceTensor normT = DeviceTensor::zeros({nvirt, nvirt, ncore}, dr);
  DeviceTensor hT = DeviceTensor::zeros({nvirt, nvirt, ncore}, dr);
  DeviceTensor normTDot = DeviceTensor::zeros({nvirt, nvirt, ncore}, dr);
  DeviceTensor hTDot = DeviceTensor::zeros({nvirt, nvirt, ncore}, dr);
  forBatches(src.extent(ExtBlock::Srsi), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Srsi, b0, b1, dr);    // h[r_batch, s, i, a]
    DeviceTensor slabT = src.slab(ExtBlock::SrsiT, b0, b1, dr);  // h[s, r_batch, i, a]
    const DeviceTensor& hA = slab;
    const DeviceTensor& hB = slabT;
    DeviceTensor normRows = sliceAxis(normT, 0, b0, b1);
    DeviceTensor hRows = sliceAxis(hT, 0, b0, b1);
    DeviceTensor normRowsDot = sliceAxis(normTDot, 0, b0, b1);
    DeviceTensor hRowsDot = sliceAxis(hTDot, 0, b0, b1);
    srsiAccumSlab(hA, hB, r.dm1, normRows, s);
    srsiAccumSlab(hA, hB, k27, hRows, s);
    srsiAccumSlab(hA, hB, d.dm1, normRowsDot, s);
    srsiAccumSlab(hA, hB, k27Dot, hRowsDot, s);
  });

  const Tensor hostNormT = downloadTensor(normT, s);
  const Tensor hostHT = downloadTensor(hT, s);
  const TriVectors base = srsiPack(hostNormT, hostHT, hostECore, hostEVirt);
  const TriVectors dot =
      srsiPack(downloadTensor(normTDot, s), downloadTensor(hTDot, s), hostECore, hostEVirt);
  const std::optional<TriVectors> epsPack =
      eps ? std::optional(srsiPack(hostNormT, hostHT, eps->core, eps->virt)) : std::nullopt;
  return tangentFinishPacked(base, dot, epsPack ? &*epsPack : nullptr);
}

ScTangentSums tangent_Srs(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                          const Tensor& hostEVirt, const ScRdms& r, const ScRdms& d,
                          const OrbitalEnergyDot* eps, const DeviceResources& dr) {
  const DeviceTensor& h1e = ai.h1e;
  const DeviceTensor& h2e = ai.h2e;
  if (hostEVirt.size() == 0) return {};
  auto [rm2, a7] = make_a7(h1e, h2e, r.dm1, r.dm2, r.dm3, dr);
  auto [rm2Dot, a7Dot] = make_a7(h1e, h2e, d.dm1, d.dm2, d.dm3, dr);

  ScTangentSums acc;
  forBatches(src.extent(ExtBlock::Srs), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Srs, b0, b1, dr);  // (nr, v, a, a)
    const DeviceTensor& h2e_v = slab;
    DeviceTensor norm = srsNormSlab(h2e_v, rm2, dr);
    DeviceTensor h = srsHSlab(h2e_v, a7, dr);
    DeviceTensor normDot = srsNormSlab(h2e_v, rm2Dot, dr);
    DeviceTensor hDot = srsHSlab(h2e_v, a7Dot, dr);

    const auto diffDot =
        diffDotOf(eps, [&](const Tensor&, const Tensor& ev) { return srsDiff(ev, b0, b1); });
    addSlab(acc, tangentFinish(norm, h, normDot, hDot, srsDiff(hostEVirt, b0, b1),
                               ptrOf(diffDot), dr));
  });
  return acc;
}

ScTangentSums tangent_Sij(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                          const Tensor& hostECore, const ScRdms& r, const ScRdms& d,
                          const OrbitalEnergyDot* eps, const DeviceResources& dr) {
  if (hostECore.size() == 0) return {};
  const SijBlocks base = sijBlocks(ai, r, BlockOrder::Value, dr);
  const SijBlocks dot = sijBlocks(ai, d, BlockOrder::Tangent, dr);

  ScTangentSums acc;
  forBatches(src.extent(ExtBlock::Sij), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sij, b0, b1, dr);  // (a, a, ni, c)
    const DeviceTensor& h2e_v = slab;
    DeviceTensor norm = sijSlab(h2e_v, base.hdm2, dr);
    DeviceTensor h = sijSlab(h2e_v, base.a9, dr);
    DeviceTensor normDot = sijSlab(h2e_v, dot.hdm2, dr);
    DeviceTensor hDot = sijSlab(h2e_v, dot.a9, dr);

    const auto diffDot =
        diffDotOf(eps, [&](const Tensor& ec, const Tensor&) { return sijDiff(ec, b0, b1); });
    addSlab(acc, tangentFinish(norm, h, normDot, hDot, sijDiff(hostECore, b0, b1),
                               ptrOf(diffDot), dr));
  });
  return acc;
}

ScTangentSums tangent_Sir(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                          const Tensor& hostECore, const Tensor& hostEVirt, const ScRdms& r,
                          const ScRdms& d, const OrbitalEnergyDot* eps,
                          const DeviceResources& dr) {
  const SirBlocks base = sirBlocks(ai, r, dr);
  const SirBlocks dot = sirBlocks(ai, d, dr);

  ScTangentSums acc;
  forBatches(src.extent(ExtBlock::Sir1), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab1 = src.slab(ExtBlock::Sir1, b0, b1, dr);  // (nr, a, c, a)
    DeviceTensor slab2 = src.slab(ExtBlock::Sir2, b0, b1, dr);  // (nr, a, a, c)
    const DeviceTensor& h2e_v1 = slab1;
    const DeviceTensor& h2e_v2 = slab2;
    DeviceTensor h1e_v = sliceAxis(ai.h1e_v_Sir, 0, b0, b1);  // (nr, c)

    DeviceTensor norm = sirNormSlab(h2e_v1, h2e_v2, h1e_v, r.dm1, r.dm2,
                                    /*withOneElectronTerm=*/true, dr);
    DeviceTensor h = sirHSlab(h2e_v1, h2e_v2, base.a12, base.a13, dr);
    // The one-electron term reads no RDM: constant, so out of the tangent.
    DeviceTensor normDot = sirNormSlab(h2e_v1, h2e_v2, h1e_v, d.dm1, d.dm2,
                                       /*withOneElectronTerm=*/false, dr);
    DeviceTensor hDot = sirHSlab(h2e_v1, h2e_v2, dot.a12, dot.a13, dr);

    const auto diffDot = diffDotOf(eps, [&](const Tensor& ec, const Tensor& ev) {
      return sirDiff(ec, ev, b0, b1);
    });
    addSlab(acc, tangentFinish(norm, h, normDot, hDot, sirDiff(hostECore, hostEVirt, b0, b1),
                               ptrOf(diffDot), dr));
  });
  return acc;
}

// --- entry points ---------------------------------------------------------------

EnergyTangentResult energiesTangentDevice(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                          int64_t batch, const DeviceTensor& dm1,
                                          const DeviceTensor& dm2, const DeviceTensor& dm3,
                                          const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                          const DeviceTensor& dm1Dot, const DeviceTensor& dm2Dot,
                                          const DeviceTensor& dm3Dot,
                                          const DeviceTensor& f3acDot,
                                          const DeviceTensor& f3caDot,
                                          const OrbitalEnergyDot* epsDot,
                                          const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  // --profile: the launches below report in the energy table, like
  // energiesDevice's (they are the same einsums, run twice).
  const profile::Section profileSection(profile::kEnergy);
  const int64_t norb = dm1.dims[0];
  const Tensor hostECore = downloadTensor(ai.e_core, s);
  const Tensor hostEVirt = downloadTensor(ai.e_virt, s);

  const ScRdms r{dm1, dm2, dm3, f3ac, f3ca};
  const ScRdms d{dm1Dot, dm2Dot, dm3Dot, f3acDot, f3caDot};
  // The orbital energies' tangent is the caller's to give or withhold: it is
  // what makes Edot the derivative of the COMPLETE functional, in which
  // e_core / e_virt move with the active density (nevpt2.energy_finish's
  // normToEnergyDeltaTangent). Its lengths are the golden's, so a mismatch
  // is our bug.
  if (epsDot != nullptr)
    check(epsDot->core.size() == hostECore.size() && epsDot->virt.size() == hostEVirt.size(),
          "energiesTangentDevice: the orbital energies' tangent has the wrong lengths");

  // CLASSES order, as energiesDevice.
  const std::array<ScTangentSums, 8> cls = {
      tangent_Sr(ai, src, batch, hostEVirt, r, d, epsDot, norb, dr),
      tangent_Si(ai, src, batch, hostECore, r, d, epsDot, norb, dr),
      tangent_Sijrs(ai, src, batch, epsDot, dr),
      tangent_Sijr(ai, src, batch, hostECore, hostEVirt, r, d, epsDot, dr),
      tangent_Srsi(ai, src, batch, hostECore, hostEVirt, r, d, epsDot, dr),
      tangent_Srs(ai, src, batch, hostEVirt, r, d, epsDot, dr),
      tangent_Sij(ai, src, batch, hostECore, r, d, epsDot, dr),
      tangent_Sir(ai, src, batch, hostECore, hostEVirt, r, d, epsDot, dr),
  };

  EnergyTangentResult out;
  for (std::size_t c = 0; c < cls.size(); ++c) {
    out.norms[c] = cls[c].norm;
    out.energies[c] = cls[c].energy;
    out.normDots[c] = cls[c].normDot;
    out.energyDots[c] = cls[c].energyDot;
    out.total += cls[c].energy;
    out.totalDot += cls[c].energyDot;
  }
  return out;
}

EnergyTangentResult energiesTangentDevice(const NevptIntegralsDevice& ints,
                                          const DeviceTensor& dm1, const DeviceTensor& dm2,
                                          const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                          const DeviceTensor& f3ca, const DeviceTensor& dm1Dot,
                                          const DeviceTensor& dm2Dot, const DeviceTensor& dm3Dot,
                                          const DeviceTensor& f3acDot,
                                          const DeviceTensor& f3caDot,
                                          const OrbitalEnergyDot* epsDot,
                                          const DeviceResources& res) {
  FullBlockSource src(ints);
  // Views: `ints` keeps owning everything (energiesDevice's arrangement).
  ActiveIntegralsDevice ai{ints.h1e.view(),      ints.h2e.view(),      ints.e_core.view(),
                           ints.e_virt.view(),   ints.h1e_v_Sr.view(), ints.h1e_v_Si.view(),
                           ints.h1e_v_Sir.view()};
  return energiesTangentDevice(ai, src, /*batch=*/0, dm1, dm2, dm3, f3ac, f3ca, dm1Dot, dm2Dot,
                               dm3Dot, f3acDot, f3caDot, epsDot, res);
}

}  // namespace nevpt2
