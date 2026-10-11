// The eight SC-NEVPT2 class energies on the GPU (each over slabs of its
// external integral block -- see energy.cppm) -- a direct transcription of
// a NumPy reference's ~140 np.einsum calls, each issued here as the generic
// device contraction kernel (device_einsum.cu) with the *same* subscript
// string, rather than hand-derived index loops. Only the *finishing*
// arithmetic (norm/energy triangular extraction, symmetrization, the final
// -sum(N/(Delta+H/N)) reduction) stays on the host: those operate on tensors
// no bigger than (nvirt,)/(ncore,)/(nvirt,ncore) -- a handful to a few
// hundred elements -- so there is nothing for the GPU to accelerate there.
// (An intermediate all-host reference, an earlier version of this file, was
// used to validate this port call-by-call and has since been replaced; see energy/energy.cppm's header
// comment and docs/implementation.md, "Porting the class energies".)
module;

// What einsum/device_einsum_bridge.h pulls in, textually in the GMF; the
// bridge itself is included after the imports (see rdm/rdm_accumulate_bridge.h).
#include <runtime.h>
#include "einsum/einsum_plan.h"

module nevpt2.energy;

import std;
import :shared;
import :sc_finish;
import :sc_classes;

import nevpt2.profile;
import wwr.runtime_api;

// energy_Sijrs's slab reduction, linked in from einsum/device_einsum.cu.
// extern "C++": attached to the global module, so it binds to the .cu's
// definition (see rdm/rdm_accumulate_bridge.h).
extern "C++" {
#include "einsum/device_einsum_bridge.h"
}

namespace nevpt2 {

const std::array<std::string, 8> CLASSES = {
    "Sr", "Si", "Sijrs", "Sijr", "Srsi", "Srs", "Sij", "Sir",
};

// --- the class metrics and Dyall Hamiltonians ---------------------------------
//
// make_a16, make_a22, make_dm3_h/make_dm2_h, make_hdm1/2/3, make_a3, make_k27,
// make_a7, make_a9, make_a12 and make_a13 are declared -- and exported -- in
// energy.cppm: energy_pc.cpp builds its PC metrics and Hamiltonians from these
// same intermediates, and so does nevpt2.gradient's pseudodensity assembly.
// energy_Sijrs, which PC reuses whole, stays in the nevpt2.energy:shared
// partition (not exported).

DeviceTensor make_a16(const DeviceTensor& h1e,
                       const DeviceTensor& h2e, const DeviceTensor& dm3,
                       const DeviceTensor& f3ac, const DeviceTensor& f3ca, int64_t norb,
                       const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor a16 = deviceEinsumNew("ib,rpqiac->pqrabc", {&h1e, &dm3}, -1.0, dr);
  deviceEinsumAccum("ia,rpqbic->pqrabc", {&h1e, &dm3}, a16, 1.0, s);
  deviceEinsumAccum("ci,rpqbai->pqrabc", {&h1e, &dm3}, a16, -1.0, s);
  deviceTransposeAccum(f3ca, {1, 4, 0, 2, 5, 3}, a16, -1.0, s);
  deviceEinsumAccum("kbia,rpqcki->pqrabc", {&h2e, &dm3}, a16, -1.0, s);
  deviceEinsumAccum("kbaj,rpqjkc->pqrabc", {&h2e, &dm3}, a16, -1.0, s);
  deviceEinsumAccum("cbij,rpqjai->pqrabc", {&h2e, &dm3}, a16, 1.0, s);

  DeviceTensor fdm2 = deviceEinsumNew("kbij,rpajki->prab", {&h2e, &dm3}, 1.0, dr);
  launchDiagA16(fdm2, a16, norb, s);

  deviceTransposeAccum(f3ac, {1, 2, 0, 4, 3, 5}, a16, 1.0, s);
  deviceTransposeAccum(f3ca, {1, 2, 0, 4, 3, 5}, a16, -1.0, s);
  deviceEinsumAccum("jbij,rpqiac->pqrabc", {&h2e, &dm3}, a16, 1.0, s);
  deviceEinsumAccum("cjka,rpqbjk->pqrabc", {&h2e, &dm3}, a16, -1.0, s);
  deviceEinsumAccum("jcij,rpqbai->pqrabc", {&h2e, &dm3}, a16, 1.0, s);
  return a16;
}

DeviceTensor make_a22(const DeviceTensor& h1e,
                       const DeviceTensor& h2e, const DeviceTensor& dm2,
                       const DeviceTensor& dm3, const DeviceTensor& f3ac,
                       const DeviceTensor& f3ca, int64_t norb, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor a22 = deviceEinsumNew("pb,kipjac->ijkabc", {&h1e, &dm3}, -1.0, dr);
  deviceEinsumAccum("pa,kibjpc->ijkabc", {&h1e, &dm3}, a22, -1.0, s);
  deviceEinsumAccum("cp,kibjap->ijkabc", {&h1e, &dm3}, a22, 1.0, s);
  deviceEinsumAccum("cqra,kibjqr->ijkabc", {&h2e, &dm3}, a22, 1.0, s);
  deviceEinsumAccum("qcpq,kibjap->ijkabc", {&h2e, &dm3}, a22, -1.0, s);
  deviceTransposeAccum(f3ac, {1, 5, 0, 2, 4, 3}, a22, -1.0, s);

  DeviceTensor fdm2a = deviceEinsumNew("pqrb,kiqcpr->ikbc", {&h2e, &dm3}, 1.0, dr);
  launchDiagA22a(fdm2a, a22, norb, s);

  deviceEinsumAccum("pqab,kiqjpc->ijkabc", {&h2e, &dm3}, a22, -1.0, s);
  deviceEinsumAccum("pcrb,kiajpr->ijkabc", {&h2e, &dm3}, a22, 1.0, s);
  deviceEinsumAccum("cqrb,kiqjar->ijkabc", {&h2e, &dm3}, a22, 1.0, s);
  deviceTransposeAccum(f3ac, {1, 3, 0, 4, 2, 5}, a22, -1.0, s);
  deviceTransposeAccum(f3ca, {1, 3, 0, 4, 2, 5}, a22, 1.0, s);
  deviceEinsumAccum("jb,kiac->ijkabc", {&h1e, &dm2}, a22, 2.0, s);
  deviceEinsumAccum("pjrb,kiprac->ijkabc", {&h2e, &dm3}, a22, 2.0, s);

  DeviceTensor fdm2b = deviceEinsumNew("pa,kipc->ikac", {&h1e, &dm2}, 1.0, dr);
  deviceEinsumAccum("cp,kiap->ikac", {&h1e, &dm2}, fdm2b, -1.0, s);
  deviceEinsumAccum("cqra,kiqr->ikac", {&h2e, &dm2}, fdm2b, -1.0, s);
  deviceEinsumAccum("qcpq,kiap->ikac", {&h2e, &dm2}, fdm2b, 1.0, s);
  deviceEinsumAccum("pqra,kiqcpr->ikac", {&h2e, &dm3}, fdm2b, 1.0, s);
  deviceEinsumAccum("rcpq,kiaqrp->ikac", {&h2e, &dm3}, fdm2b, -1.0, s);
  launchDiagA22b(fdm2b, a22, norb, s);

  return a22;
}

DeviceTensor make_dm3_h(const DeviceTensor& dm2, const DeviceTensor& dm3,
                        const DeviceResources& dr) {
  const int64_t n = dm2.dims[0];
  DeviceTensor delta = deviceEye(n, dr);
  DeviceTensor dm3_h = deviceEinsumNew("abef,cd->abcdef", {&dm2, &delta}, 2.0, dr);
  deviceTransposeAccum(dm3, {0, 1, 3, 2, 4, 5}, dm3_h, -1.0, dr.stream());
  return dm3_h;
}

DeviceTensor make_dm2_h(const DeviceTensor& dm1, const DeviceTensor& dm2,
                        const DeviceResources& dr) {
  const int64_t n = dm1.dims[0];
  DeviceTensor delta = deviceEye(n, dr);
  DeviceTensor dm2_h = deviceEinsumNew("ab,cd->abcd", {&dm1, &delta}, 2.0, dr);
  deviceTransposeAccum(dm2, {0, 1, 3, 2}, dm2_h, -1.0, dr.stream());
  return dm2_h;
}

DeviceTensor make_hdm1(const DeviceTensor& dm1, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  int64_t n = dm1.dims[0];
  DeviceTensor delta = deviceEye(n, dr);
  DeviceTensor hdm1 = deviceEinsumNew("ij->ij", {&delta}, 2.0, dr);
  deviceTransposeAccum(dm1, {1, 0}, hdm1, -1.0, s);
  return hdm1;
}

DeviceTensor make_hdm2(const DeviceTensor& dm1,
                        const DeviceTensor& dm2, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  int64_t n = dm1.dims[0];
  DeviceTensor delta = deviceEye(n, dr);
  DeviceTensor dm2b = deviceEinsumNew("ikjl->ijkl", {&dm2}, 1.0, dr);
  deviceEinsumAccum("jk,il->ijkl", {&delta, &dm1}, dm2b, -1.0, s);
  DeviceTensor hdm2 = deviceEinsumNew("klij->ijkl", {&dm2b}, 1.0, dr);
  deviceEinsumAccum("il,kj->ijkl", {&delta, &dm1}, hdm2, 1.0, s);
  deviceEinsumAccum("jk,li->ijkl", {&delta, &dm1}, hdm2, 1.0, s);
  deviceEinsumAccum("ik,lj->ijkl", {&delta, &dm1}, hdm2, -2.0, s);
  deviceEinsumAccum("jl,ki->ijkl", {&delta, &dm1}, hdm2, -2.0, s);
  deviceEinsumAccum("il,jk->ijkl", {&delta, &delta}, hdm2, -2.0, s);
  deviceEinsumAccum("ik,jl->ijkl", {&delta, &delta}, hdm2, 4.0, s);
  return hdm2;
}

DeviceTensor make_hdm3(const DeviceTensor& dm1,
                        const DeviceTensor& dm2, const DeviceTensor& dm3,
                        const DeviceTensor& hdm1, const DeviceTensor& hdm2, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  (void)hdm1;
  int64_t n = dm3.dims[0];
  DeviceTensor delta = deviceEye(n, dr);
  DeviceTensor hdm3 = deviceEinsumNew("pb,qrac->pqrabc", {&delta, &hdm2}, -1.0, dr);
  deviceEinsumAccum("br,pqac->pqrabc", {&delta, &hdm2}, hdm3, -1.0, s);
  deviceEinsumAccum("bq,prac->pqrabc", {&delta, &hdm2}, hdm3, 2.0, s);
  deviceEinsumAccum("ap,bqcr->pqrabc", {&delta, &dm2}, hdm3, 2.0, s);
  deviceEinsumAccum("ap,cr,bq->pqrabc", {&delta, &delta, &dm1}, hdm3, -4.0, s);
  deviceEinsumAccum("cr,bqap->pqrabc", {&delta, &dm2}, hdm3, 2.0, s);
  deviceEinsumAccum("bqapcr->pqrabc", {&dm3}, hdm3, -1.0, s);
  deviceEinsumAccum("ar,pc,bq->pqrabc", {&delta, &delta, &dm1}, hdm3, 2.0, s);
  deviceEinsumAccum("ar,bqcp->pqrabc", {&delta, &dm2}, hdm3, -1.0, s);
  return hdm3;
}

DeviceTensor make_a3(const DeviceTensor& h1e,
                      const DeviceTensor& h2e, const DeviceTensor& dm1,
                      const DeviceTensor& dm2, const DeviceTensor& hdm1, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  int64_t n = dm2.dims[0];
  DeviceTensor delta = deviceEye(n, dr);
  DeviceTensor a3 = deviceEinsumNew("ia,ip->pa", {&h1e, &hdm1}, 1.0, dr);
  deviceEinsumAccum("ijka,pj,ik->pa", {&h2e, &delta, &dm1}, a3, 2.0, s);
  deviceEinsumAccum("ijka,jpik->pa", {&h2e, &dm2}, a3, -1.0, s);
  return a3;
}

DeviceTensor make_k27(const DeviceTensor& h1e,
                       const DeviceTensor& h2e, const DeviceTensor& dm1,
                       const DeviceTensor& dm2, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor k27 = deviceEinsumNew("ai,pi->pa", {&h1e, &dm1}, -1.0, dr);
  deviceEinsumAccum("iajk,pkij->pa", {&h2e, &dm2}, k27, -1.0, s);
  deviceEinsumAccum("iaji,pj->pa", {&h2e, &dm1}, k27, 1.0, s);
  return k27;
}

DeviceTensor make_rm3(const DeviceTensor& dm2, const DeviceTensor& dm3,
                      const DeviceTensor& rm2, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  int64_t n = dm2.dims[0];
  DeviceTensor delta = deviceEye(n, dr);
  DeviceTensor rm3 = deviceEinsumNew("injmkl->ijklmn", {&dm3}, 1.0, dr);
  deviceEinsumAccum("jn,imkl->ijklmn", {&delta, &dm2}, rm3, -1.0, s);
  deviceEinsumAccum("km,ijln->ijklmn", {&delta, &rm2}, rm3, -1.0, s);
  deviceEinsumAccum("kn,ijml->ijklmn", {&delta, &rm2}, rm3, -1.0, s);
  return rm3;
}

std::pair<DeviceTensor, DeviceTensor> make_a7(const DeviceTensor& h1e,
                                               const DeviceTensor& h2e,
                                               const DeviceTensor& dm1,
                                               const DeviceTensor& dm2,
                                               const DeviceTensor& dm3, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  int64_t n = dm2.dims[0];
  DeviceTensor delta = deviceEye(n, dr);
  DeviceTensor rm2 = deviceEinsumNew("iljk->ijkl", {&dm2}, 1.0, dr);
  deviceEinsumAccum("ik,jl->ijkl", {&dm1, &delta}, rm2, -1.0, s);

  // rm3 is a16's hole-side partner for the virtual pair; it is its own
  // function because nevpt2.gradient needs it to differentiate a7 (Park Eqs.
  // 44-47) and a7 does not hand it back.
  DeviceTensor rm3 = make_rm3(dm2, dm3, rm2, dr);

  DeviceTensor a7 = deviceEinsumNew("bi,pqia->pqab", {&h1e, &rm2}, -1.0, dr);
  deviceEinsumAccum("ai,pqbi->pqab", {&h1e, &rm2}, a7, -1.0, s);
  deviceEinsumAccum("kbij,pqkija->pqab", {&h2e, &rm3}, a7, -1.0, s);
  deviceEinsumAccum("kaij,pqkibj->pqab", {&h2e, &rm3}, a7, -1.0, s);
  deviceEinsumAccum("baij,pqij->pqab", {&h2e, &rm2}, a7, -1.0, s);
  return {std::move(rm2), std::move(a7)};
}

DeviceTensor make_a9(const DeviceTensor& h1e,
                      const DeviceTensor& h2e, const DeviceTensor& hdm1,
                      const DeviceTensor& hdm2, const DeviceTensor& hdm3, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  (void)hdm1;
  DeviceTensor a9 = deviceEinsumNew("ib,pqai->pqab", {&h1e, &hdm2}, 1.0, dr);
  deviceEinsumAccum("ijib,pqaj->pqab", {&h2e, &hdm2}, a9, 2.0, s);
  deviceEinsumAccum("ijjb,pqai->pqab", {&h2e, &hdm2}, a9, -1.0, s);
  deviceEinsumAccum("ijkb,pkqaij->pqab", {&h2e, &hdm3}, a9, -1.0, s);
  deviceEinsumAccum("ia,pqib->pqab", {&h1e, &hdm2}, a9, 1.0, s);
  deviceEinsumAccum("ijja,pqib->pqab", {&h2e, &hdm2}, a9, -1.0, s);
  deviceEinsumAccum("ijba,pqji->pqab", {&h2e, &hdm2}, a9, -1.0, s);
  deviceEinsumAccum("ijia,pqjb->pqab", {&h2e, &hdm2}, a9, 2.0, s);
  deviceEinsumAccum("ijka,pqkjbi->pqab", {&h2e, &hdm3}, a9, -1.0, s);
  return a9;
}

DeviceTensor make_a12(const DeviceTensor& h1e,
                       const DeviceTensor& h2e, const DeviceTensor& dm1,
                       const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  (void)dm1;
  DeviceTensor a12 = deviceEinsumNew("ia,qpib->pqab", {&h1e, &dm2}, 1.0, dr);
  deviceEinsumAccum("bi,qpai->pqab", {&h1e, &dm2}, a12, -1.0, s);
  deviceEinsumAccum("ijka,qpjbik->pqab", {&h2e, &dm3}, a12, 1.0, s);
  deviceEinsumAccum("kbij,qpajki->pqab", {&h2e, &dm3}, a12, -1.0, s);
  deviceEinsumAccum("bjka,qpjk->pqab", {&h2e, &dm2}, a12, -1.0, s);
  deviceEinsumAccum("jbij,qpai->pqab", {&h2e, &dm2}, a12, 1.0, s);
  return a12;
}

DeviceTensor make_a13(const DeviceTensor& h1e,
                       const DeviceTensor& h2e, const DeviceTensor& dm1,
                       const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  int64_t n = dm3.dims[0];
  DeviceTensor delta = deviceEye(n, dr);
  DeviceTensor a13 = deviceEinsumNew("ia,qbip->pqab", {&h1e, &dm2}, -1.0, dr);
  deviceEinsumAccum("pa,qb->pqab", {&h1e, &dm1}, a13, 2.0, s);
  deviceEinsumAccum("bi,qiap->pqab", {&h1e, &dm2}, a13, 1.0, s);
  deviceEinsumAccum("pa,bi,qi->pqab", {&delta, &h1e, &dm1}, a13, -2.0, s);
  deviceEinsumAccum("ijka,qbjpik->pqab", {&h2e, &dm3}, a13, -1.0, s);
  deviceEinsumAccum("kbij,qjapki->pqab", {&h2e, &dm3}, a13, 1.0, s);
  deviceEinsumAccum("blma,qmlp->pqab", {&h2e, &dm2}, a13, 1.0, s);
  deviceEinsumAccum("kpma,qbkm->pqab", {&h2e, &dm2}, a13, 2.0, s);
  deviceEinsumAccum("bpma,qm->pqab", {&h2e, &dm1}, a13, -2.0, s);
  deviceEinsumAccum("lbkl,qkap->pqab", {&h2e, &dm2}, a13, -1.0, s);
  deviceEinsumAccum("ap,mbkl,qlmk->pqab", {&delta, &h2e, &dm2}, a13, -2.0, s);
  deviceEinsumAccum("ap,lbkl,qk->pqab", {&delta, &h2e, &dm1}, a13, 2.0, s);
  return a13;
}

// No RDM contraction at all: the MP2-like class, reduced on the device one
// core-orbital slab at a time (device_einsum.cu's energy_sijrs) -- the
// slab is (ni, nvirt, ncore, nvirt), so the full (ncore*nvirt)^2 block never
// has to exist anywhere when the source builds slabs on the fly.
std::pair<double, double> energy_Sijrs(const ActiveIntegralsDevice& ai,
                                        IntegralSource& src, int64_t batch,
                                        const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  int64_t ncore = ai.e_core.size(), nvirt = ai.e_virt.size();
  if (ncore == 0 || nvirt == 0) return {0.0, 0.0};
  DeviceTensor acc = DeviceTensor::zeros({2}, dr);
  // The reduction kernel's launcher takes int: narrowed once, checked.
  const int ncoreK = narrowTo<int>(ncore), nvirtK = narrowTo<int>(nvirt);
  forBatches(src.extent(ExtBlock::Sijrs), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sijrs, b0, b1, dr);
    const DeviceTensor& g = slab;
    // The kernel reads the slab as dense rows; every IntegralSource hands out
    // a contiguous one, so anything else is our bug.
    check(isContiguous(g), "energy_Sijrs: the Sijrs slab must be contiguous");
    gpuCheck(device::energySijrs(s, g.data(), narrowTo<int>(b1 - b0), narrowTo<int>(b0), ncoreK,
                                 nvirtK, ai.e_core.data(), ai.e_virt.data(), acc.data()));
  });
  Tensor host = downloadTensor(acc, s);
  return {host.flat(0), host.flat(1)};
}

// --- slabs and sources ----------------------------------------------------------

int64_t batchAxis(ExtBlock b) {
  switch (b) {
    case ExtBlock::Si:
    case ExtBlock::Sij:
      return 2;
    case ExtBlock::SrsiT:
      return 1;
    default:
      return 0;
  }
}

const char* extBlockName(ExtBlock b) {
  switch (b) {
    case ExtBlock::Sr: return "h2e_v_Sr";
    case ExtBlock::Si: return "h2e_v_Si";
    case ExtBlock::Sijrs: return "cvcv";
    case ExtBlock::Sijr: return "h2e_v_Sijr";
    case ExtBlock::Srsi: return "h2e_v_Srsi";
    case ExtBlock::SrsiT: return "h2e_v_Srsi";
    case ExtBlock::Srs: return "h2e_v_Srs";
    case ExtBlock::Sij: return "h2e_v_Sij";
    case ExtBlock::Sir1: return "h2e_v1_Sir";
    case ExtBlock::Sir2: return "h2e_v2_Sir";
  }
  return "?";
}

const DeviceTensor& FullBlockSource::block(ExtBlock b) const {
  switch (b) {
    case ExtBlock::Sr: return ints_.h2e_v_Sr;
    case ExtBlock::Si: return ints_.h2e_v_Si;
    case ExtBlock::Sijrs: return ints_.cvcv;
    case ExtBlock::Sijr: return ints_.h2e_v_Sijr;
    case ExtBlock::Srsi:
    case ExtBlock::SrsiT: return ints_.h2e_v_Srsi;
    case ExtBlock::Srs: return ints_.h2e_v_Srs;
    case ExtBlock::Sij: return ints_.h2e_v_Sij;
    case ExtBlock::Sir1: return ints_.h2e_v1_Sir;
    case ExtBlock::Sir2: return ints_.h2e_v2_Sir;
  }
  std::abort();
}

int64_t FullBlockSource::extent(ExtBlock b) const { return block(b).dims[batchAxis(b)]; }

DeviceTensor FullBlockSource::slab(ExtBlock b, int64_t b0, int64_t b1, const DeviceResources&) {
  return sliceAxis(block(b), batchAxis(b), b0, b1);  // a view: ints_ owns the block
}

// --- entry points ----------------------------------------------------------------

EnergyResult energiesDevice(const ActiveIntegralsDevice& ai,
                             IntegralSource& src, int64_t batch, const DeviceTensor& dm1,
                             const DeviceTensor& dm2, const DeviceTensor& dm3,
                             const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                             const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  // --profile: the einsum launches below report in the energy table. A
  // density-fitted slab build opens its own section (DF integrals) inside.
  const profile::Section profileSection(profile::kEnergy);
  int64_t norb = dm1.dims[0];
  Tensor hostECore = downloadTensor(ai.e_core, s);
  Tensor hostEVirt = downloadTensor(ai.e_virt, s);

  EnergyResult r;
  std::tie(r.norms[0], r.energies[0]) =
      energy_Sr(ai, src, batch, hostEVirt, dm1, dm2, dm3, f3ac, f3ca, norb, dr);
  std::tie(r.norms[1], r.energies[1]) =
      energy_Si(ai, src, batch, hostECore, dm1, dm2, dm3, f3ac, f3ca, norb, dr);
  std::tie(r.norms[2], r.energies[2]) = energy_Sijrs(ai, src, batch, dr);
  std::tie(r.norms[3], r.energies[3]) =
      energy_Sijr(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dr);
  std::tie(r.norms[4], r.energies[4]) =
      energy_Srsi(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dr);
  std::tie(r.norms[5], r.energies[5]) =
      energy_Srs(ai, src, batch, hostEVirt, dm1, dm2, dm3, dr);
  std::tie(r.norms[6], r.energies[6]) =
      energy_Sij(ai, src, batch, hostECore, dm1, dm2, dm3, dr);
  std::tie(r.norms[7], r.energies[7]) =
      energy_Sir(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dm3, dr);
  r.total = 0.0;
  for (double e : r.energies) r.total += e;
  return r;
}

EnergyResult energiesDevice(const NevptIntegralsDevice& ints,
                             const DeviceTensor& dm1, const DeviceTensor& dm2,
                             const DeviceTensor& dm3, const DeviceTensor& f3ac,
                             const DeviceTensor& f3ca, const DeviceResources& res) {
  FullBlockSource src(ints);
  // Views: `ints` keeps owning everything.
  ActiveIntegralsDevice ai{ints.h1e.view(),      ints.h2e.view(),      ints.e_core.view(),
                           ints.e_virt.view(),   ints.h1e_v_Sr.view(), ints.h1e_v_Si.view(),
                           ints.h1e_v_Sir.view()};
  return energiesDevice(ai, src, /*batch=*/0, dm1, dm2, dm3, f3ac, f3ca, res);
}

}  // namespace nevpt2
