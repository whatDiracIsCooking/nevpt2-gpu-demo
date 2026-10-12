// nevpt2.energy:sc_classes -- seven of the eight SC class energies and the
// SC-only intermediates they build (make_a17/a19/a23/a25). Sijrs, which
// PC reuses whole, is in nevpt2.energy:shared.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :sc_classes;` and from no importer of nevpt2.energy.
// (Used by energy.cpp and by nevpt2.energy:sc_tangent.)
//
// EACH CLASS IS WRITTEN IN THREE PIECES, so the forward tangent can reuse all
// three rather than re-transcribe them (energy_sc_tangent.cppm):
//
//   * the per-slab CONTRACTION of that class's norm or H -- the slab's
//     integrals against ONE metric or Dyall-Hamiltonian block per term
//     (`*NormSlab` / `*HSlab`, or one function where the two sides share
//     their subscripts). Every one is LINEAR in those blocks, which is the
//     whole reason the tangent needs no contraction of its own: it calls the
//     same function with the tangent blocks.
//   * the class's Delta vector, the orbital-energy denominator the finish
//     takes the pair against (`*Diff`) -- no RDM in it, so it is the same for
//     the energy and its tangent.
//   * the host packing, where a class has one: Sijr's and Srsi's symmetrized,
//     triangle-packed perturbers (`sijrPack` / `srsiPack`), also linear in
//     what they read.
//
// The class energy itself is then the slab walk plus nevpt2.energy_finish's
// normToEnergy, and the tangent is the same walk with each contraction run
// twice.
module nevpt2.energy:sc_classes;

import std;
import nevpt2.energy;
import :shared;
import :sc_finish;

namespace nevpt2 {

// --- make_a* intermediates, device-resident ----------------------------------

// make_a17/make_a19 are Sr's (with make_a16), make_a23/make_a25 Si's (with
// make_a22). All four are DECLARED in energy.cppm ("the class metrics and
// Dyall Hamiltonians") and so exported, like the ones energy.cpp defines: the
// SC gradient (nevpt2.gradient) differentiates the same class einsums and
// needs the same blocks.

DeviceTensor make_a17(const DeviceTensor& h1e_in,
                       const DeviceTensor& h2e, const DeviceTensor& dm2,
                       const DeviceTensor& dm3, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor h1e = deviceEinsumNew("mn->mn", {&h1e_in}, 1.0, dr);
  deviceEinsumAccum("mjjn->mn", {&h2e}, h1e, -1.0, s);
  DeviceTensor a17 = deviceEinsumNew("pi,cabi->abcp", {&h1e, &dm2}, -1.0, dr);
  deviceEinsumAccum("kpij,cabjki->abcp", {&h2e, &dm3}, a17, -1.0, s);
  return a17;
}

DeviceTensor make_a19(const DeviceTensor& h1e_in,
                       const DeviceTensor& h2e, const DeviceTensor& dm1,
                       const DeviceTensor& dm2, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor h1e = deviceEinsumNew("mn->mn", {&h1e_in}, 1.0, dr);
  deviceEinsumAccum("mjjn->mn", {&h2e}, h1e, -1.0, s);
  DeviceTensor a19 = deviceEinsumNew("pi,ai->ap", {&h1e, &dm1}, -1.0, dr);
  deviceEinsumAccum("kpij,ajki->ap", {&h2e, &dm2}, a19, -1.0, s);
  return a19;
}

DeviceTensor make_a23(const DeviceTensor& h1e,
                       const DeviceTensor& h2e, const DeviceTensor& dm1,
                       const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor a23 = deviceEinsumNew("ip,caib->abcp", {&h1e, &dm2}, -1.0, dr);
  deviceEinsumAccum("pijk,cajbik->abcp", {&h2e, &dm3}, a23, -1.0, s);
  deviceEinsumAccum("bp,ca->abcp", {&h1e, &dm1}, a23, 2.0, s);
  deviceEinsumAccum("pibk,caik->abcp", {&h2e, &dm2}, a23, 2.0, s);
  return a23;
}

DeviceTensor make_a25(const DeviceTensor& h1e,
                       const DeviceTensor& h2e, const DeviceTensor& dm1,
                       const DeviceTensor& dm2, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor a25 = deviceEinsumNew("pi,ai->ap", {&h1e, &dm1}, -1.0, dr);
  deviceEinsumAccum("pijk,jaik->ap", {&h2e, &dm2}, a25, -1.0, s);
  // The one RDM-FREE term of any make_* here: constant as the RDMs move, so
  // the tangent's a25 drops it (make_a25_dot, energy_sc_tangent.cppm).
  deviceEinsumAccum("ap->pa", {&h1e}, a25, 2.0, s);
  deviceEinsumAccum("piaj,ij->ap", {&h2e, &dm1}, a25, 2.0, s);
  return a25;
}

// --- the per-slab contractions ------------------------------------------------
//
// One function per class and side, each the verbatim chain the class energy
// below used to inline. `g`, `dm*`, `a*` are the RDM-only blocks the slab is
// contracted against; the tangent passes the tangent ones.

// Sr: the norm side (dm1/dm2/dm3) and the H side (a16/a17/a19), over a slab
// of h2e_v_Sr and the matching rows of h1e_v_Sr.
DeviceTensor srNormSlab(const DeviceTensor& h2e_v, const DeviceTensor& h1e_v,
                        const DeviceTensor& dm1, const DeviceTensor& dm2,
                        const DeviceTensor& dm3, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor norm = deviceEinsumNew("ipqr,rpqbac,iabc->i", {&h2e_v, &dm3, &h2e_v}, 1.0, dr);
  deviceEinsumAccum("ipqr,rpqa,ia->i", {&h2e_v, &dm2, &h1e_v}, norm, 2.0, s);
  deviceEinsumAccum("ip,pa,ia->i", {&h1e_v, &dm1, &h1e_v}, norm, 1.0, s);
  return norm;
}

DeviceTensor srHSlab(const DeviceTensor& h2e_v, const DeviceTensor& h1e_v,
                     const DeviceTensor& a16, const DeviceTensor& a17,
                     const DeviceTensor& a19, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor ener = deviceEinsumNew("ipqr,pqrabc,iabc->i", {&h2e_v, &a16, &h2e_v}, 1.0, dr);
  deviceEinsumAccum("ipqr,pqra,ia->i", {&h2e_v, &a17, &h1e_v}, ener, 2.0, s);
  deviceEinsumAccum("ip,pa,ia->i", {&h1e_v, &a19, &h1e_v}, ener, 1.0, s);
  return ener;
}

// Si: the hole-side metrics (dm3_h/dm2_h/dm1_h) and a22/a23/a25.
DeviceTensor siNormSlab(const DeviceTensor& h2e_v, const DeviceTensor& h1e_v,
                        const DeviceTensor& dm1_h, const DeviceTensor& dm2_h,
                        const DeviceTensor& dm3_h, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor norm = deviceEinsumNew("qpir,rpqbac,baic->i", {&h2e_v, &dm3_h, &h2e_v}, 1.0, dr);
  deviceEinsumAccum("qpir,rpqa,ai->i", {&h2e_v, &dm2_h, &h1e_v}, norm, 2.0, s);
  deviceEinsumAccum("pi,pa,ai->i", {&h1e_v, &dm1_h, &h1e_v}, norm, 1.0, s);
  return norm;
}

DeviceTensor siHSlab(const DeviceTensor& h2e_v, const DeviceTensor& h1e_v,
                     const DeviceTensor& a22, const DeviceTensor& a23,
                     const DeviceTensor& a25, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor ener = deviceEinsumNew("qpir,pqrabc,baic->i", {&h2e_v, &a22, &h2e_v}, 1.0, dr);
  deviceEinsumAccum("qpir,pqra,ai->i", {&h2e_v, &a23, &h1e_v}, ener, 2.0, s);
  deviceEinsumAccum("pi,pa,ai->i", {&h1e_v, &a25, &h1e_v}, ener, 1.0, s);
  return ener;
}

// Sijr: both sides share their subscripts, so ONE function -- `g` is hdm1 for
// the norm and a3 for H. (r, j, i), before the symmetrization sijrPack does.
DeviceTensor sijrSlab(const DeviceTensor& h2e_v, const DeviceTensor& g,
                      const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor out = deviceEinsumNew("rpji,raji,pa->rji", {&h2e_v, &h2e_v, &g}, 2.0, dr);
  deviceEinsumAccum("rpji,raij,pa->rji", {&h2e_v, &h2e_v, &g}, out, -1.0, s);
  return out;
}

// Srsi: likewise one function -- `g` is dm1 for the norm and k27 for H --
// accumulating into rows [b0, b1) of the class's whole (v, v, c) output.
void srsiAccumSlab(const DeviceTensor& hA, const DeviceTensor& hB, const DeviceTensor& g,
                   const DeviceTensor& rows, wwrStream_t stream) {
  deviceEinsumAccum("rsip,rsia,pa->rsi", {&hA, &hA, &g}, rows, 2.0, stream);
  deviceEinsumAccum("rsip,sria,pa->rsi", {&hA, &hB, &g}, rows, -1.0, stream);
}

// Srs: one term each side, but with different subscripts (the metric is read
// `pqba`, the Dyall block `pqab`), so two functions.
DeviceTensor srsNormSlab(const DeviceTensor& h2e_v, const DeviceTensor& rm2,
                         const DeviceResources& dr) {
  return deviceEinsumNew("rsqp,rsba,pqba->rs", {&h2e_v, &h2e_v, &rm2}, 0.5, dr);
}

DeviceTensor srsHSlab(const DeviceTensor& h2e_v, const DeviceTensor& a7,
                      const DeviceResources& dr) {
  return deviceEinsumNew("rsqp,rsba,pqab->rs", {&h2e_v, &h2e_v, &a7}, 0.5, dr);
}

// Sij: one term, same subscripts both sides -- `g` is hdm2 for the norm and
// a9 for H.
DeviceTensor sijSlab(const DeviceTensor& h2e_v, const DeviceTensor& g,
                     const DeviceResources& dr) {
  return deviceEinsumNew("qpij,baij,pqab->ij", {&h2e_v, &h2e_v, &g}, 0.5, dr);
}

// Sir's norm, over both of its slabs and the h1e_v_Sir rows. Its LAST term
// contracts no RDM at all (two one-electron integrals alone), which is why
// `withOneElectronTerm` is here: it is constant as the RDMs move, so the
// tangent leaves it out -- the only class term in the eight that needs that.
DeviceTensor sirNormSlab(const DeviceTensor& h2e_v1, const DeviceTensor& h2e_v2,
                         const DeviceTensor& h1e_v, const DeviceTensor& dm1,
                         const DeviceTensor& dm2, bool withOneElectronTerm,
                         const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor norm = deviceEinsumNew("rpiq,raib,qpab->ir", {&h2e_v1, &h2e_v1, &dm2}, 2.0, dr);
  deviceEinsumAccum("rpiq,rabi,qpab->ir", {&h2e_v1, &h2e_v2, &dm2}, norm, -1.0, s);
  deviceEinsumAccum("rpqi,raib,qpab->ir", {&h2e_v2, &h2e_v1, &dm2}, norm, -1.0, s);
  deviceEinsumAccum("raqi,rabi,qb->ir", {&h2e_v2, &h2e_v2, &dm1}, norm, 2.0, s);
  deviceEinsumAccum("rpqi,rabi,qbap->ir", {&h2e_v2, &h2e_v2, &dm2}, norm, -1.0, s);
  deviceEinsumAccum("rpqi,raai,qp->ir", {&h2e_v2, &h2e_v2, &dm1}, norm, 1.0, s);
  deviceEinsumAccum("rpiq,ri,qp->ir", {&h2e_v1, &h1e_v, &dm1}, norm, 4.0, s);
  deviceEinsumAccum("rpqi,ri,qp->ir", {&h2e_v2, &h1e_v, &dm1}, norm, -2.0, s);
  if (withOneElectronTerm) deviceEinsumAccum("ri,ri->ir", {&h1e_v, &h1e_v}, norm, 2.0, s);
  return norm;
}

DeviceTensor sirHSlab(const DeviceTensor& h2e_v1, const DeviceTensor& h2e_v2,
                      const DeviceTensor& a12, const DeviceTensor& a13,
                      const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor h = deviceEinsumNew("rpiq,raib,pqab->ir", {&h2e_v1, &h2e_v1, &a12}, 2.0, dr);
  deviceEinsumAccum("rpiq,rabi,pqab->ir", {&h2e_v1, &h2e_v2, &a12}, h, -1.0, s);
  deviceEinsumAccum("rpqi,raib,pqab->ir", {&h2e_v2, &h2e_v1, &a12}, h, -1.0, s);
  deviceEinsumAccum("rpqi,rabi,pqab->ir", {&h2e_v2, &h2e_v2, &a13}, h, 1.0, s);
  return h;
}

// --- the Delta vectors --------------------------------------------------------
//
// The orbital-energy denominator each class's finish takes its (N, H) pair
// against, in the class output's own index order. No RDM enters, so the
// tangent takes the very same vector.

// Si: minus the core slice (Sr's is sliceHost(e_virt, b0, b1), unchanged).
std::vector<double> siDiff(const Tensor& hostECore, int64_t b0, int64_t b1) {
  std::vector<double> diff = sliceHost(hostECore, b0, b1);
  for (double& x : diff) x = -x;
  return diff;
}

// Srs: eps_r + eps_s over the slab's r and every s.
std::vector<double> srsDiff(const Tensor& hostEVirt, int64_t b0, int64_t b1) {
  const int64_t nvirt = hostEVirt.size();
  std::vector<double> diff;
  for (int64_t r = b0; r < b1; ++r)
    for (int64_t ss = 0; ss < nvirt; ++ss) diff.push_back(hostEVirt(r) + hostEVirt(ss));
  return diff;
}

// Sij: -(eps_i + eps_j) over the slab's i and every j.
std::vector<double> sijDiff(const Tensor& hostECore, int64_t b0, int64_t b1) {
  const int64_t ncore = hostECore.size();
  std::vector<double> diff;
  for (int64_t i = b0; i < b1; ++i)
    for (int64_t j = 0; j < ncore; ++j) diff.push_back(-(hostECore(i) + hostECore(j)));
  return diff;
}

// Sir: eps_r - eps_i, in the output's (c, nr) order.
std::vector<double> sirDiff(const Tensor& hostECore, const Tensor& hostEVirt, int64_t b0,
                            int64_t b1) {
  const int64_t ncore = hostECore.size();
  std::vector<double> diff;
  for (int64_t i = 0; i < ncore; ++i)
    for (int64_t r = b0; r < b1; ++r) diff.push_back(hostEVirt(r) - hostECore(i));
  return diff;
}

// Sijrs: eps_a + eps_b - eps_i - eps_j over the slab's i and every (a, j, b),
// cvcv's own (i, a, j, b) order -- the negative of energy_sijrs' `djba`. The
// class's energy runs on the device (energy_Sijrs) and never needs this; the
// orbital energies' tangent does (:sc_tangent's tangent_Sijrs).
std::vector<double> sijrsDiff(const Tensor& hostECore, const Tensor& hostEVirt, int64_t b0,
                              int64_t b1) {
  const int64_t ncore = hostECore.size(), nvirt = hostEVirt.size();
  std::vector<double> diff;
  diff.reserve(static_cast<std::size_t>((b1 - b0) * nvirt * ncore * nvirt));
  for (int64_t i = b0; i < b1; ++i)
    for (int64_t a = 0; a < nvirt; ++a)
      for (int64_t j = 0; j < ncore; ++j)
        for (int64_t b = 0; b < nvirt; ++b)
          diff.push_back(hostEVirt(a) + hostEVirt(b) - hostECore(i) - hostECore(j));
  return diff;
}

// --- the two host packings ----------------------------------------------------

// One class's perturber vectors as normToEnergy takes them: N, H and Delta,
// same length, same order.
struct TriVectors {
  std::vector<double> norm;
  std::vector<double> h;
  std::vector<double> diff;
};

// Sijr's host finish for one slab: the downloaded (r, j, i) tables
// symmetrized in (i, j) with the diagonal halved, then packed over j <= i
// with Delta = eps_r - eps_j - eps_i. LINEAR in `normTh`/`hTh`, so the
// tangent packs its own pair with the same call and keeps this `diff`.
TriVectors sijrPack(const Tensor& normTh, const Tensor& hTh, const Tensor& hostECore,
                    const Tensor& hostEVirt, int64_t b0) {
  const int64_t nr = normTh.dim(0), ncore = hostECore.size();
  Tensor normSym({nr, ncore, ncore}), hSym({nr, ncore, ncore});
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t j = 0; j < ncore; ++j)
      for (int64_t i = 0; i < ncore; ++i) {
        normSym(r, j, i) = normTh(r, j, i) + normTh(r, i, j);
        hSym(r, j, i) = hTh(r, j, i) + hTh(r, i, j);
      }
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t kk = 0; kk < ncore; ++kk) {
      normSym(r, kk, kk) *= 0.5;
      hSym(r, kk, kk) *= 0.5;
    }

  TriVectors out;
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t j = 0; j < ncore; ++j)
      for (int64_t i = j; i < ncore; ++i) {
        out.norm.push_back(normSym(r, j, i));
        out.h.push_back(hSym(r, j, i));
        out.diff.push_back(hostEVirt(b0 + r) - hostECore(j) - hostECore(i));
      }
  return out;
}

// Srsi's host finish, over the whole (v, v, c) tables (the symmetrization
// pairs (r, s) with (s, r), which can sit in different slabs, so this runs
// once after the walk): symmetrized in (r, s) with the diagonal halved, then
// packed over r <= s with Delta = eps_r + eps_s - eps_i. Linear, like
// sijrPack.
TriVectors srsiPack(const Tensor& normTh, const Tensor& hTh, const Tensor& hostECore,
                    const Tensor& hostEVirt) {
  const int64_t nvirt = hostEVirt.size(), ncore = hostECore.size();
  Tensor normSym({nvirt, nvirt, ncore}), hSym({nvirt, nvirt, ncore});
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t ss = 0; ss < nvirt; ++ss)
      for (int64_t i = 0; i < ncore; ++i) {
        normSym(r, ss, i) = normTh(r, ss, i) + normTh(ss, r, i);
        hSym(r, ss, i) = hTh(r, ss, i) + hTh(ss, r, i);
      }
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t i = 0; i < ncore; ++i) {
      normSym(r, r, i) *= 0.5;
      hSym(r, r, i) *= 0.5;
    }

  TriVectors out;
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t ss = r; ss < nvirt; ++ss)
      for (int64_t i = 0; i < ncore; ++i) {
        out.norm.push_back(normSym(r, ss, i));
        out.h.push_back(hSym(r, ss, i));
        out.diff.push_back(hostEVirt(r) + hostEVirt(ss) - hostECore(i));
      }
  return out;
}

// --- the eight class energies -----------------------------------------------
//
// Each walks its batch index (ExtBlock's batchAxis) and accumulates the
// per-slab -sum N/(Delta+H/N) -- an elementwise sum, so splitting it by slab
// only reassociates it. The slab and the per-slab outputs are freed at the end
// of their iteration, once read back; the RDM-only intermediates (a16, ...) are
// built once per class, before the slab loop, exactly as before slabs existed,
// and freed when the class returns. Every free is a wwrFreeAsync on the one
// stream, behind the kernels that read the buffer.
//
// Sijrs, which PC reuses whole, is with the shared definitions in energy.cpp.

std::pair<double, double> energy_Sr(const ActiveIntegralsDevice& ai,
                                     IntegralSource& src, int64_t batch,
                                     const Tensor& hostEVirt, const DeviceTensor& dm1,
                                     const DeviceTensor& dm2, const DeviceTensor& dm3,
                                     const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                     int64_t norb, const DeviceResources& dr) {
  const DeviceTensor& h1e = ai.h1e;
  const DeviceTensor& h2e = ai.h2e;

  DeviceTensor a16 = make_a16(h1e, h2e, dm3, f3ac, f3ca, norb, dr);
  DeviceTensor a17 = make_a17(h1e, h2e, dm2, dm3, dr);
  DeviceTensor a19 = make_a19(h1e, h2e, dm1, dm2, dr);

  double normSum = 0.0, enerSum = 0.0;
  forBatches(src.extent(ExtBlock::Sr), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sr, b0, b1, dr);
    const DeviceTensor& h2e_v = slab;
    DeviceTensor h1e_v = sliceAxis(ai.h1e_v_Sr, 0, b0, b1);

    DeviceTensor ener = srHSlab(h2e_v, h1e_v, a16, a17, a19, dr);
    DeviceTensor norm = srNormSlab(h2e_v, h1e_v, dm1, dm2, dm3, dr);

    // No sync first: see finish.
    auto [n, e] = finish(norm, ener, sliceHost(hostEVirt, b0, b1), dr);
    normSum += n;
    enerSum += e;
  });
  return {normSum, enerSum};
}

std::pair<double, double> energy_Si(const ActiveIntegralsDevice& ai,
                                     IntegralSource& src, int64_t batch,
                                     const Tensor& hostECore, const DeviceTensor& dm1,
                                     const DeviceTensor& dm2, const DeviceTensor& dm3,
                                     const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                     int64_t norb, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const DeviceTensor& h1e = ai.h1e;
  const DeviceTensor& h2e = ai.h2e;

  DeviceTensor a22 = make_a22(h1e, h2e, dm2, dm3, f3ac, f3ca, norb, dr);
  DeviceTensor a23 = make_a23(h1e, h2e, dm1, dm2, dm3, dr);
  DeviceTensor a25 = make_a25(h1e, h2e, dm1, dm2, dr);
  DeviceTensor delta = deviceEye(norb, dr);

  DeviceTensor dm3_h = make_dm3_h(dm2, dm3, dr);
  DeviceTensor dm2_h = make_dm2_h(dm1, dm2, dr);
  // 2 delta - dm1^T, which is make_hdm1's expression; the 2 delta is the
  // constant the tangent's dm1_h drops (make_hdm1_dot).
  DeviceTensor dm1_h = deviceEinsumNew("ij->ij", {&delta}, 2.0, dr);
  deviceTransposeAccum(dm1, {1, 0}, dm1_h, -1.0, s);

  double normSum = 0.0, enerSum = 0.0;
  forBatches(src.extent(ExtBlock::Si), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Si, b0, b1, dr);
    const DeviceTensor& h2e_v = slab;
    DeviceTensor h1e_v = sliceAxis(ai.h1e_v_Si, 1, b0, b1);

    DeviceTensor ener = siHSlab(h2e_v, h1e_v, a22, a23, a25, dr);
    DeviceTensor norm = siNormSlab(h2e_v, h1e_v, dm1_h, dm2_h, dm3_h, dr);

    auto [n, e] = finish(norm, ener, siDiff(hostECore, b0, b1), dr);
    normSum += n;
    enerSum += e;
  });
  return {normSum, enerSum};
}

std::pair<double, double> energy_Sijr(const ActiveIntegralsDevice& ai,
                                       IntegralSource& src, int64_t batch,
                                       const Tensor& hostECore, const Tensor& hostEVirt,
                                       const DeviceTensor& dm1, const DeviceTensor& dm2,
                                       const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const DeviceTensor& h1e = ai.h1e;
  const DeviceTensor& h2e = ai.h2e;
  DeviceTensor hdm1 = make_hdm1(dm1, dr);
  DeviceTensor a3 = make_a3(h1e, h2e, dm1, dm2, hdm1, dr);

  double normSum = 0.0, enerSum = 0.0;
  forBatches(src.extent(ExtBlock::Sijr), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sijr, b0, b1, dr);  // (nr, a, c, c)
    const DeviceTensor& h2e_v = slab;

    DeviceTensor normT = sijrSlab(h2e_v, hdm1, dr);
    DeviceTensor hT = sijrSlab(h2e_v, a3, dr);

    const TriVectors v = sijrPack(downloadTensor(normT, s), downloadTensor(hT, s), hostECore,
                                  hostEVirt, b0);
    auto [n, e] = normToEnergy(v.norm, v.h, v.diff);
    normSum += n;
    enerSum += e;
  });
  return {normSum, enerSum};
}

std::pair<double, double> energy_Srsi(const ActiveIntegralsDevice& ai,
                                       IntegralSource& src, int64_t batch,
                                       const Tensor& hostECore, const Tensor& hostEVirt,
                                       const DeviceTensor& dm1, const DeviceTensor& dm2,
                                       const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const DeviceTensor& h1e = ai.h1e;
  const DeviceTensor& h2e = ai.h2e;
  int64_t nvirt = hostEVirt.size(), ncore = hostECore.size();
  DeviceTensor k27 = make_k27(h1e, h2e, dm1, dm2, dr);

  // normT/hT are (v, v, c) -- the whole output, filled one r-batch of rows
  // at a time, because the symmetrization in srsiPack pairs (r, s) with
  // (s, r), which can sit in different batches. That is v^2 c doubles: the
  // slabs, not the output, are what scale with n_act.
  DeviceTensor normT = DeviceTensor::zeros({nvirt, nvirt, ncore}, dr);
  DeviceTensor hT = DeviceTensor::zeros({nvirt, nvirt, ncore}, dr);
  forBatches(src.extent(ExtBlock::Srsi), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Srsi, b0, b1, dr);     // h[r_batch, s, i, a]
    DeviceTensor slabT = src.slab(ExtBlock::SrsiT, b0, b1, dr);   // h[s, r_batch, i, a]
    const DeviceTensor& hA = slab;
    const DeviceTensor& hB = slabT;
    // Rows [b0, b1) of a row-major (v, v, c) tensor: contiguous.
    DeviceTensor normRows = sliceAxis(normT, 0, b0, b1);
    DeviceTensor hRows = sliceAxis(hT, 0, b0, b1);
    srsiAccumSlab(hA, hB, dm1, normRows, s);
    srsiAccumSlab(hA, hB, k27, hRows, s);
  });

  const TriVectors v =
      srsiPack(downloadTensor(normT, s), downloadTensor(hT, s), hostECore, hostEVirt);
  return normToEnergy(v.norm, v.h, v.diff);
}

std::pair<double, double> energy_Srs(const ActiveIntegralsDevice& ai,
                                      IntegralSource& src, int64_t batch,
                                      const Tensor& hostEVirt, const DeviceTensor& dm1,
                                      const DeviceTensor& dm2, const DeviceTensor& dm3,
                                      const DeviceResources& dr) {
  const DeviceTensor& h1e = ai.h1e;
  const DeviceTensor& h2e = ai.h2e;
  int64_t nvirt = hostEVirt.size();
  if (nvirt == 0) return {0.0, 0.0};
  auto [rm2, a7] = make_a7(h1e, h2e, dm1, dm2, dm3, dr);

  double normSum = 0.0, enerSum = 0.0;
  forBatches(src.extent(ExtBlock::Srs), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Srs, b0, b1, dr);  // (nr, v, a, a)
    const DeviceTensor& h2e_v = slab;
    DeviceTensor norm = srsNormSlab(h2e_v, rm2, dr);
    DeviceTensor h = srsHSlab(h2e_v, a7, dr);

    auto [n, e] = finish(norm, h, srsDiff(hostEVirt, b0, b1), dr);
    normSum += n;
    enerSum += e;
  });
  return {normSum, enerSum};
}

std::pair<double, double> energy_Sij(const ActiveIntegralsDevice& ai,
                                      IntegralSource& src, int64_t batch,
                                      const Tensor& hostECore, const DeviceTensor& dm1,
                                      const DeviceTensor& dm2, const DeviceTensor& dm3,
                                      const DeviceResources& dr) {
  const DeviceTensor& h1e = ai.h1e;
  const DeviceTensor& h2e = ai.h2e;
  int64_t ncore = hostECore.size();
  if (ncore == 0) return {0.0, 0.0};
  DeviceTensor hdm1 = make_hdm1(dm1, dr);
  DeviceTensor hdm2 = make_hdm2(dm1, dm2, dr);
  DeviceTensor hdm3 = make_hdm3(dm1, dm2, dm3, hdm1, hdm2, dr);
  DeviceTensor a9 = make_a9(h1e, h2e, hdm1, hdm2, hdm3, dr);

  double normSum = 0.0, enerSum = 0.0;
  forBatches(src.extent(ExtBlock::Sij), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sij, b0, b1, dr);  // (a, a, ni, c)
    const DeviceTensor& h2e_v = slab;
    DeviceTensor norm = sijSlab(h2e_v, hdm2, dr);
    DeviceTensor h = sijSlab(h2e_v, a9, dr);

    auto [n, e] = finish(norm, h, sijDiff(hostECore, b0, b1), dr);
    normSum += n;
    enerSum += e;
  });
  return {normSum, enerSum};
}

std::pair<double, double> energy_Sir(const ActiveIntegralsDevice& ai,
                                      IntegralSource& src, int64_t batch,
                                      const Tensor& hostECore, const Tensor& hostEVirt,
                                      const DeviceTensor& dm1, const DeviceTensor& dm2,
                                      const DeviceTensor& dm3, const DeviceResources& dr) {
  const DeviceTensor& h1e = ai.h1e;
  const DeviceTensor& h2e = ai.h2e;

  DeviceTensor a12 = make_a12(h1e, h2e, dm1, dm2, dm3, dr);
  DeviceTensor a13 = make_a13(h1e, h2e, dm1, dm2, dm3, dr);

  double normSum = 0.0, enerSum = 0.0;
  forBatches(src.extent(ExtBlock::Sir1), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab1 = src.slab(ExtBlock::Sir1, b0, b1, dr);  // (nr, a, c, a)
    DeviceTensor slab2 = src.slab(ExtBlock::Sir2, b0, b1, dr);  // (nr, a, a, c)
    const DeviceTensor& h2e_v1 = slab1;
    const DeviceTensor& h2e_v2 = slab2;
    DeviceTensor h1e_v = sliceAxis(ai.h1e_v_Sir, 0, b0, b1);  // (nr, c)

    DeviceTensor norm = sirNormSlab(h2e_v1, h2e_v2, h1e_v, dm1, dm2,
                                    /*withOneElectronTerm=*/true, dr);
    DeviceTensor h = sirHSlab(h2e_v1, h2e_v2, a12, a13, dr);

    auto [n, e] = finish(norm, h, sirDiff(hostECore, hostEVirt, b0, b1), dr);
    normSum += n;
    enerSum += e;
  });
  return {normSum, enerSum};
}

}  // namespace nevpt2
