// nevpt2.energy:sc_classes -- seven of the eight SC class energies and the
// SC-only intermediates they build (make_a17/a19/a23/a25). Sijrs, which
// PC reuses whole, is in nevpt2.energy:shared.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :sc_classes;` and from no importer of nevpt2.energy.
// (Used by energy.cpp.)
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
  deviceEinsumAccum("ap->pa", {&h1e}, a25, 2.0, s);
  deviceEinsumAccum("piaj,ij->ap", {&h2e, &dm1}, a25, 2.0, s);
  return a25;
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
  const wwrStream_t s = dr.stream();
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

    DeviceTensor ener = deviceEinsumNew("ipqr,pqrabc,iabc->i", {&h2e_v, &a16, &h2e_v}, 1.0, dr);
    deviceEinsumAccum("ipqr,pqra,ia->i", {&h2e_v, &a17, &h1e_v}, ener, 2.0, s);
    deviceEinsumAccum("ip,pa,ia->i", {&h1e_v, &a19, &h1e_v}, ener, 1.0, s);

    DeviceTensor norm = deviceEinsumNew("ipqr,rpqbac,iabc->i", {&h2e_v, &dm3, &h2e_v}, 1.0, dr);
    deviceEinsumAccum("ipqr,rpqa,ia->i", {&h2e_v, &dm2, &h1e_v}, norm, 2.0, s);
    deviceEinsumAccum("ip,pa,ia->i", {&h1e_v, &dm1, &h1e_v}, norm, 1.0, s);

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
  DeviceTensor dm1_h = deviceEinsumNew("ij->ij", {&delta}, 2.0, dr);
  deviceTransposeAccum(dm1, {1, 0}, dm1_h, -1.0, s);

  double normSum = 0.0, enerSum = 0.0;
  forBatches(src.extent(ExtBlock::Si), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Si, b0, b1, dr);
    const DeviceTensor& h2e_v = slab;
    DeviceTensor h1e_v = sliceAxis(ai.h1e_v_Si, 1, b0, b1);

    DeviceTensor ener = deviceEinsumNew("qpir,pqrabc,baic->i", {&h2e_v, &a22, &h2e_v}, 1.0, dr);
    deviceEinsumAccum("qpir,pqra,ai->i", {&h2e_v, &a23, &h1e_v}, ener, 2.0, s);
    deviceEinsumAccum("pi,pa,ai->i", {&h1e_v, &a25, &h1e_v}, ener, 1.0, s);

    DeviceTensor norm = deviceEinsumNew("qpir,rpqbac,baic->i", {&h2e_v, &dm3_h, &h2e_v}, 1.0, dr);
    deviceEinsumAccum("qpir,rpqa,ai->i", {&h2e_v, &dm2_h, &h1e_v}, norm, 2.0, s);
    deviceEinsumAccum("pi,pa,ai->i", {&h1e_v, &dm1_h, &h1e_v}, norm, 1.0, s);

    std::vector<double> negECore = sliceHost(hostECore, b0, b1);
    for (double& x : negECore) x = -x;
    auto [n, e] = finish(norm, ener, negECore, dr);
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
  int64_t ncore = hostECore.size();
  DeviceTensor hdm1 = make_hdm1(dm1, dr);
  DeviceTensor a3 = make_a3(h1e, h2e, dm1, dm2, hdm1, dr);

  double normSum = 0.0, enerSum = 0.0;
  forBatches(src.extent(ExtBlock::Sijr), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sijr, b0, b1, dr);  // (nr, a, c, c)
    const DeviceTensor& h2e_v = slab;
    int64_t nr = b1 - b0;

    DeviceTensor normT = deviceEinsumNew("rpji,raji,pa->rji", {&h2e_v, &h2e_v, &hdm1}, 2.0, dr);
    deviceEinsumAccum("rpji,raij,pa->rji", {&h2e_v, &h2e_v, &hdm1}, normT, -1.0, s);
    DeviceTensor hT = deviceEinsumNew("rpji,raji,pa->rji", {&h2e_v, &h2e_v, &a3}, 2.0, dr);
    deviceEinsumAccum("rpji,raij,pa->rji", {&h2e_v, &h2e_v, &a3}, hT, -1.0, s);

    Tensor normTh = downloadTensor(normT, s), hTh = downloadTensor(hT, s);
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

    std::vector<double> normTri, hTri, diffTri;
    for (int64_t r = 0; r < nr; ++r)
      for (int64_t j = 0; j < ncore; ++j)
        for (int64_t i = j; i < ncore; ++i) {
          normTri.push_back(normSym(r, j, i));
          hTri.push_back(hSym(r, j, i));
          diffTri.push_back(hostEVirt(b0 + r) - hostECore(j) - hostECore(i));
        }
    auto [n, e] = normToEnergy(normTri, hTri, diffTri);
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
  // at a time, because the symmetrization below pairs (r, s) with (s, r),
  // which can sit in different batches. That is v^2 c doubles: the slabs,
  // not the output, are what scale with n_act.
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
    deviceEinsumAccum("rsip,rsia,pa->rsi", {&hA, &hA, &dm1}, normRows, 2.0, s);
    deviceEinsumAccum("rsip,sria,pa->rsi", {&hA, &hB, &dm1}, normRows, -1.0, s);
    deviceEinsumAccum("rsip,rsia,pa->rsi", {&hA, &hA, &k27}, hRows, 2.0, s);
    deviceEinsumAccum("rsip,sria,pa->rsi", {&hA, &hB, &k27}, hRows, -1.0, s);
  });

  Tensor normTh = downloadTensor(normT, s), hTh = downloadTensor(hT, s);
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

  std::vector<double> normTri, hTri, diffTri;
  for (int64_t r = 0; r < nvirt; ++r)
    for (int64_t ss = r; ss < nvirt; ++ss)
      for (int64_t i = 0; i < ncore; ++i) {
        normTri.push_back(normSym(r, ss, i));
        hTri.push_back(hSym(r, ss, i));
        diffTri.push_back(hostEVirt(r) + hostEVirt(ss) - hostECore(i));
      }
  return normToEnergy(normTri, hTri, diffTri);
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
    DeviceTensor norm = deviceEinsumNew("rsqp,rsba,pqba->rs", {&h2e_v, &h2e_v, &rm2}, 0.5, dr);
    DeviceTensor h = deviceEinsumNew("rsqp,rsba,pqab->rs", {&h2e_v, &h2e_v, &a7}, 0.5, dr);

    std::vector<double> diff;
    for (int64_t r = b0; r < b1; ++r)
      for (int64_t ss = 0; ss < nvirt; ++ss) diff.push_back(hostEVirt(r) + hostEVirt(ss));
    auto [n, e] = finish(norm, h, diff, dr);
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
    DeviceTensor norm = deviceEinsumNew("qpij,baij,pqab->ij", {&h2e_v, &h2e_v, &hdm2}, 0.5, dr);
    DeviceTensor h = deviceEinsumNew("qpij,baij,pqab->ij", {&h2e_v, &h2e_v, &a9}, 0.5, dr);

    std::vector<double> diff;
    for (int64_t i = b0; i < b1; ++i)
      for (int64_t j = 0; j < ncore; ++j) diff.push_back(-(hostECore(i) + hostECore(j)));
    auto [n, e] = finish(norm, h, diff, dr);
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
  const wwrStream_t s = dr.stream();
  const DeviceTensor& h1e = ai.h1e;
  const DeviceTensor& h2e = ai.h2e;
  int64_t ncore = hostECore.size();

  DeviceTensor a12 = make_a12(h1e, h2e, dm1, dm2, dm3, dr);
  DeviceTensor a13 = make_a13(h1e, h2e, dm1, dm2, dm3, dr);

  double normSum = 0.0, enerSum = 0.0;
  forBatches(src.extent(ExtBlock::Sir1), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab1 = src.slab(ExtBlock::Sir1, b0, b1, dr);  // (nr, a, c, a)
    DeviceTensor slab2 = src.slab(ExtBlock::Sir2, b0, b1, dr);  // (nr, a, a, c)
    const DeviceTensor& h2e_v1 = slab1;
    const DeviceTensor& h2e_v2 = slab2;
    DeviceTensor h1e_v = sliceAxis(ai.h1e_v_Sir, 0, b0, b1);  // (nr, c)

    DeviceTensor norm = deviceEinsumNew("rpiq,raib,qpab->ir", {&h2e_v1, &h2e_v1, &dm2}, 2.0, dr);
    deviceEinsumAccum("rpiq,rabi,qpab->ir", {&h2e_v1, &h2e_v2, &dm2}, norm, -1.0, s);
    deviceEinsumAccum("rpqi,raib,qpab->ir", {&h2e_v2, &h2e_v1, &dm2}, norm, -1.0, s);
    deviceEinsumAccum("raqi,rabi,qb->ir", {&h2e_v2, &h2e_v2, &dm1}, norm, 2.0, s);
    deviceEinsumAccum("rpqi,rabi,qbap->ir", {&h2e_v2, &h2e_v2, &dm2}, norm, -1.0, s);
    deviceEinsumAccum("rpqi,raai,qp->ir", {&h2e_v2, &h2e_v2, &dm1}, norm, 1.0, s);
    deviceEinsumAccum("rpiq,ri,qp->ir", {&h2e_v1, &h1e_v, &dm1}, norm, 4.0, s);
    deviceEinsumAccum("rpqi,ri,qp->ir", {&h2e_v2, &h1e_v, &dm1}, norm, -2.0, s);
    deviceEinsumAccum("ri,ri->ir", {&h1e_v, &h1e_v}, norm, 2.0, s);

    DeviceTensor h = deviceEinsumNew("rpiq,raib,pqab->ir", {&h2e_v1, &h2e_v1, &a12}, 2.0, dr);
    deviceEinsumAccum("rpiq,rabi,pqab->ir", {&h2e_v1, &h2e_v2, &a12}, h, -1.0, s);
    deviceEinsumAccum("rpqi,raib,pqab->ir", {&h2e_v2, &h2e_v1, &a12}, h, -1.0, s);
    deviceEinsumAccum("rpqi,rabi,pqab->ir", {&h2e_v2, &h2e_v2, &a13}, h, 1.0, s);

    std::vector<double> diff;  // (c, nr), the output's own order
    for (int64_t i = 0; i < ncore; ++i)
      for (int64_t r = b0; r < b1; ++r) diff.push_back(hostEVirt(r) - hostECore(i));
    auto [n, e] = finish(norm, h, diff, dr);
    normSum += n;
    enerSum += e;
  });
  return {normSum, enerSum};
}

}  // namespace nevpt2
