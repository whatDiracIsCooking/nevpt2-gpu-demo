// dfGeneralIntegralsDevice (#165): the density-fitted case's general-range
// inventory, built on the device from the whole three-index tensor -- see
// gradient.cppm for why every one-general-index array is one contraction over
// the auxiliary index, and docs/gradient-theory.md, Sec. 3.8 for which blocks
// of B that reads. And dfDigestSourceDevice / dfDigestPairing (#168): the
// digests' transpose carried from h2e back onto the three-index tensor (see
// gradient.cppm, "the digests' transpose onto the THREE-index tensor").
module nevpt2.gradient;

import std;

namespace nevpt2 {

Result<GeneralIntegralsDevice> dfGeneralIntegralsDevice(
    const Tensor& bAA, const Tensor& bCA, const Tensor& bVA, const Tensor& bCV, const Tensor& bCC,
    const Tensor& bVV, const Tensor& h, const Tensor& heff, const Tensor& f,
    const DeviceResources& res) {
  // Ranks first: dim(1)/dim(2) are only meaningful on rank-3 tensors.
  for (const auto& [name, b] : {std::pair<const char*, const Tensor*>{"B_aa", &bAA},
                                {"B_ca", &bCA},
                                {"B_va", &bVA},
                                {"B_cv", &bCV},
                                {"B_cc", &bCC},
                                {"B_vv", &bVV}})
    if (b->rank() != 3)
      return err_io(std::format(
          "density-fitted gradient: {} must be rank 3 [naux, p, q], it is rank {}", name,
          b->rank()));
  const int64_t naux = bAA.dim(0), na = bAA.dim(1), nc = bCA.dim(1), nv = bVA.dim(1);
  const int64_t nmo = nc + na + nv;
  const std::vector<std::pair<const char*, std::pair<const Tensor*, std::vector<int64_t>>>>
      expect{
          {"B_aa", {&bAA, {naux, na, na}}}, {"B_ca", {&bCA, {naux, nc, na}}},
          {"B_va", {&bVA, {naux, nv, na}}}, {"B_cv", {&bCV, {naux, nc, nv}}},
          {"B_cc", {&bCC, {naux, nc, nc}}}, {"B_vv", {&bVV, {naux, nv, nv}}},
          {"h", {&h, {nmo, nmo}}},          {"heff", {&heff, {nmo, nmo}}},
          {"f", {&f, {nmo, nmo}}},
      };
  for (const auto& [name, spec] : expect) {
    if (spec.first->dims() == spec.second) continue;
    std::string got;
    for (const int64_t d : spec.first->dims()) got += (got.empty() ? "" : ", ") + std::to_string(d);
    std::string want;
    for (const int64_t d : spec.second) want += (want.empty() ? "" : ", ") + std::to_string(d);
    return err_io(std::format(
        "density-fitted gradient: {} has shape [{}], but naux = {} and a {} core / {} active / "
        "{} virtual partition want [{}] -- the golden's B blocks and the sidecar's do not "
        "describe one fitted calculation",
        name, got, naux, nc, na, nv, want));
  }

  // The whole B[L, p, q], on the host: the six space-pair blocks, each
  // off-diagonal one written at both (P, Q) and (Q, P) -- B is symmetric in
  // its two MO indices (Sec. 3.8) -- and the three diagonal ones as stored.
  Tensor b({naux, nmo, nmo});
  const int64_t c0 = 0, a0 = nc, v0 = nc + na;
  const auto put = [&](const Tensor& blk, int64_t rowLo, int64_t colLo, bool mirror) {
    for (int64_t l = 0; l < naux; ++l)
      for (int64_t p = 0; p < blk.dim(1); ++p)
        for (int64_t q = 0; q < blk.dim(2); ++q) {
          b(l, rowLo + p, colLo + q) = blk(l, p, q);
          if (mirror) b(l, colLo + q, rowLo + p) = blk(l, p, q);
        }
  };
  put(bCC, c0, c0, false);
  put(bAA, a0, a0, false);
  put(bVV, v0, v0, false);
  put(bCA, c0, a0, true);
  put(bVA, v0, a0, true);
  put(bCV, c0, v0, true);
  DeviceTensor dB = uploadTensor(b, res);

  // [lo, hi) of each space along an MO axis.
  enum Sp { C, A, V };
  const auto lo = [&](Sp s) { return s == C ? c0 : (s == A ? a0 : v0); };
  const auto hi = [&](Sp s) { return s == C ? a0 : (s == A ? v0 : nmo); };
  // The promoted strip B[L, x, P], x over the whole MO range: a view.
  const auto strip = [&](Sp p) { return sliceAxis(dB, 2, lo(p), hi(p)); };
  // (x p | q r) for every x: the strip against the block B_QR.
  const auto full = [&](Sp p, Sp q, Sp r) {
    const DeviceTensor s = strip(p);
    const DeviceTensor blk = sliceAxis(sliceAxis(dB, 1, lo(q), hi(q)), 2, lo(r), hi(r));
    return deviceEinsumNew("Lxp,Lqr->xpqr", {&s, &blk}, 1.0, res);
  };

  GeneralIntegralsDevice g;
  g.ncore = nc;
  g.nact = na;
  g.nvirt = nv;
  g.h = uploadTensor(h, res);
  g.heff = uploadTensor(heff, res);
  g.f = uploadTensor(f, res);
  g.g_xa_aa = full(A, A, A);
  g.g_xv_aa = full(V, A, A);
  g.g_xc_aa = full(C, A, A);
  g.g_xa_va = full(A, V, A);
  g.g_xv_va = full(V, V, A);
  g.g_xc_va = full(C, V, A);
  g.g_xa_ca = full(A, C, A);
  g.g_xv_ca = full(V, C, A);
  g.g_xc_ca = full(C, C, A);
  g.g_xa_cv = full(A, C, V);
  g.g_xv_cv = full(V, C, V);
  g.g_xc_cv = full(C, C, V);
  g.g_xa_cc = full(A, C, C);
  g.g_xv_cc = full(V, C, C);

  // The three tied rows (Sec. 3.3's restricted ones), each reading a
  // DIAGONAL of B_cc or B_vv -- the only place the energy's own economy, no
  // (L|vv) off its diagonal, would have sufficed; Y's virtual rows above are
  // where B_vv is read in full.
  const DeviceTensor sC = strip(C);
  const DeviceTensor sA = strip(A);
  const DeviceTensor cc = sliceAxis(sliceAxis(dB, 1, c0, a0), 2, c0, a0);
  const DeviceTensor vv = sliceAxis(sliceAxis(dB, 1, v0, nmo), 2, v0, nmo);
  const DeviceTensor diagC = deviceEinsumNew("Ljj->Lj", {&cc}, 1.0, res);
  const DeviceTensor diagV = deviceEinsumNew("Lrr->Lr", {&vv}, 1.0, res);
  // g_xc_cc_j[x, i, j] = (x i | j j); g_xc_cc_k[x, j, i] = (x j | j i).
  g.g_xc_cc_j = deviceEinsumNew("Lxi,Lj->xij", {&sC, &diagC}, 1.0, res);
  g.g_xc_cc_k = deviceEinsumNew("Lxj,Lji->xji", {&sC, &cc}, 1.0, res);
  // g_xa_vv[x, t, r] = (x t | r r); g_xc_vv[x, i, r] = (x i | r r).
  g.g_xa_vv = deviceEinsumNew("Lxt,Lr->xtr", {&sA, &diagV}, 1.0, res);
  g.g_xc_vv = deviceEinsumNew("Lxi,Lr->xir", {&sC, &diagV}, 1.0, res);
  // The whole B stays with the inventory (#154: the cross-slot curvature's
  // three-index route reads it). Moving the owner moves the handle, not the
  // device memory, so the contractions above -- queued against views of it --
  // still read live memory; the views themselves go out of scope here.
  g.b = std::move(dB);
  return g;
}

Result<Tensor> dfDigestSourceDevice(const DeviceTensor& hbar, const Tensor& bAA,
                                    const Tensor& bCA, const Tensor& bVA,
                                    const DeviceResources& res) {
  for (const auto& [name, b] : {std::pair<const char*, const Tensor*>{"B_aa", &bAA},
                                {"B_ca", &bCA},
                                {"B_va", &bVA}})
    if (b->rank() != 3)
      return err_io(std::format(
          "density-fitted digest term: {} must be rank 3 [naux, p, q], it is rank {}", name,
          b->rank()));
  const int64_t naux = bAA.dim(0), na = bAA.dim(1), nc = bCA.dim(1), nv = bVA.dim(1);
  if (bAA.dim(2) != na || bCA.dim(0) != naux || bCA.dim(2) != na || bVA.dim(0) != naux ||
      bVA.dim(2) != na)
    return err_io(std::format(
        "density-fitted digest term: B_aa, B_ca and B_va do not describe one fitted "
        "calculation (naux {} / {} / {}, active extent {} / {} / {})",
        naux, bCA.dim(0), bVA.dim(0), na, bCA.dim(2), bVA.dim(2)));
  if (hbar.dims != std::vector<int64_t>{na, na, na, na})
    return err_io(std::format(
        "density-fitted digest term: hbar is not (n_act)^4 for the golden's {} active orbitals",
        na));
  const int64_t nmo = nc + na + nv;

  // The active strip B[L, x, a], x over the whole MO range: the three blocks
  // with an active index stacked along x in core / active / virtual order,
  // the MO order the rest of the tree uses.
  Tensor strip({naux, nmo, na});
  const auto put = [&](const Tensor& blk, int64_t rowLo) {
    for (int64_t l = 0; l < naux; ++l)
      for (int64_t p = 0; p < blk.dim(1); ++p)
        for (int64_t t = 0; t < na; ++t) strip(l, rowLo + p, t) = blk(l, p, t);
  };
  put(bCA, 0);
  put(bAA, nc);
  put(bVA, nc + na);
  const DeviceTensor dStrip = uploadTensor(strip, res);
  const DeviceTensor dBaa = sliceAxis(dStrip, 1, nc, nc + na);

  // Bbar[L,t,u] = sum_{v,w} (hbar[t,v,u,w] + hbar[v,t,w,u]) B_aa[L,v,w].
  const DeviceTensor bbar = deviceEinsumNew("tvuw,Lvw->Ltu", {&hbar, &dBaa}, 1.0, res);
  deviceEinsumAccum("vtwu,Lvw->Ltu", {&hbar, &dBaa}, bbar, 1.0, res.stream());
  // D[x,t] = sum_{L,u} B[L,x,u] (Bbar[L,t,u] + Bbar[L,u,t]).
  const DeviceTensor d = deviceEinsumNew("Lxu,Ltu->xt", {&dStrip, &bbar}, 1.0, res);
  deviceEinsumAccum("Lxu,Lut->xt", {&dStrip, &bbar}, d, 1.0, res.stream());
  // downloadTensor synchronizes the stream itself, behind every einsum above;
  // dStrip and bbar are freed (stream-ordered) as they go out of scope.
  return downloadTensor(d, res.stream());
}

double dfDigestPairing(const Tensor& d, const Tensor& kappa, int64_t ncore) {
  check(kappa.rank() == 2 && d.rank() == 2 && kappa.dim(0) == d.dim(0) &&
            kappa.dim(1) == d.dim(0) && ncore + d.dim(1) <= kappa.dim(1),
        "dfDigestPairing: kappa must be (n_mo, n_mo) over the rows of d");
  double s = 0.0;
  for (int64_t x = 0; x < d.dim(0); ++x)
    for (int64_t t = 0; t < d.dim(1); ++t) s += kappa(x, ncore + t) * d(x, t);
  return s;
}

}  // namespace nevpt2
