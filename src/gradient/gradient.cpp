// The SC-NEVPT2 pseudodensity assembly's entry points and report -- see
// gradient.cppm for what is assembled and which of Park's equations each
// piece is, and gradient_sc_classes.cppm for the eight classes themselves.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries -- the rotation entry points
// propagate a malformed kappa (the caller's input) unchanged.
#include "error_handling/error_macros.h"

module nevpt2.gradient;

import std;
import :ci_source;    // Park Eq. 48's sigma assembly
import :general;      // the promotion table the general-range rotation reads
import :rdm_adjoint;  // the block -> RDM chain rule the CI source needs
import :rotation;
import :sc_classes;
import :terms;

namespace nevpt2 {

// The whole assembly, with or without rotation directions and with or without
// the RDM sweep: the eight classes in CLASSES order, each building its own
// metric and Dyall-Hamiltonian blocks, walking its slabs and freeing
// everything when it returns, exactly as energiesDevice's classes do.
//
// `adj` non-null ALSO sweeps each class's Eqs. 42-43 accumulators one chain
// rule further in, into the five RDM coefficient tensors Park Eq. 48 needs
// (nevpt2.gradient:rdm_adjoint); null leaves the walk exactly as the
// pseudodensity identities and the rotation pairing need it. The two are
// independent: a caller asks for either, both or neither. `epsAdjoint` also
// accumulates each class's dE/de (ScClassGradient::dEdCore / dEdVirt), which
// the CI source term folds onto dm1 when the orbital energies move (#87).
ScGradientResult assembleClasses(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                 int64_t batch, const DeviceTensor& dm1,
                                 const DeviceTensor& dm2, const DeviceTensor& dm3,
                                 const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                 const ScRotation& rot, ScRdmAdjoint* adj, bool epsAdjoint,
                                 const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t norb = dm1.dims[0];
  const Tensor hostECore = downloadTensor(ai.e_core, s);
  const Tensor hostEVirt = downloadTensor(ai.e_virt, s);

  ScGradientResult r;
  // One accumulator per direction per class, so a caller can read the
  // source term class by class (Sijrs' integral half stays zero for an
  // ACTIVE direction: it reads no active index).
  for (ScClassGradient& c : r.classes) {
    c.rotExt.assign(rot.size(), 0.0);
    c.rotAct.assign(rot.size(), 0.0);
    c.rotTangent.assign(rot.size(), 0.0);
    c.rotEps.assign(rot.size(), 0.0);
    // dE/de, for the CI source term's orbital-energy response (#87).
    c.epsAdjoint = epsAdjoint;
    if (epsAdjoint) {
      c.dEdCore.assign(static_cast<std::size_t>(hostECore.size()), 0.0);
      c.dEdVirt.assign(static_cast<std::size_t>(hostEVirt.size()), 0.0);
    }
  }
  gradientSr(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dm3, f3ac, f3ca, norb, rot,
             r.classes[0], adj, dr);
  gradientSi(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dm3, f3ac, f3ca, norb, rot,
             r.classes[1], adj, dr);
  // Sijrs contracts no RDM at all, so it has nothing to sweep and takes no
  // `adj`. It DOES take `rot`: every one of cvcv's four labels is core or
  // virtual, so an ACTIVE rotation moves nothing it reads while a GENERAL one
  // moves all of it, and its Delta_t is four orbital energies
  // (gradient_sc_classes.cppm).
  gradientSijrs(src, batch, hostECore, hostEVirt, rot, r.classes[2], dr);
  gradientSijr(ai, src, batch, hostECore, hostEVirt, dm1, dm2, rot, r.classes[3], adj, dr);
  gradientSrsi(ai, src, batch, hostECore, hostEVirt, dm1, dm2, rot, r.classes[4], adj, dr);
  gradientSrs(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dm3, rot, r.classes[5], adj,
              dr);
  gradientSij(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dm3, rot, r.classes[6], adj,
              dr);
  gradientSir(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dm3, rot, r.classes[7], adj,
              dr);
  for (const ScClassGradient& c : r.classes) r.total += c.energy;
  return r;
}

ScGradientResult scPseudodensityDevice(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                       int64_t batch, const DeviceTensor& dm1,
                                       const DeviceTensor& dm2, const DeviceTensor& dm3,
                                       const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                       const DeviceResources& dr) {
  return assembleClasses(ai, src, batch, dm1, dm2, dm3, f3ac, f3ca, ScRotation{},
                         /*adj=*/nullptr, /*epsAdjoint=*/false, dr);
}

ScCiSourceResult scCiSourceDevice(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                  int64_t batch, const Tensor& ci, int64_t nelecA,
                                  int64_t nelecB, const DeviceTensor& dm1,
                                  const DeviceTensor& dm2, const DeviceTensor& dm3,
                                  const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                  const SigmaVectorOptions& sigma,
                                  const OrbitalEnergyResponse* eps, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t norb = dm1.dims[0];
  // The five Eq. 48 coefficient tensors, zero-filled by their allocation and
  // accumulated over all eight classes. n^2 + n^4 + 3 n^6 doubles, held for
  // the whole walk -- they are what the sigma calls below read.
  ScRdmAdjoint adj = ScRdmAdjoint::zeros(norb, dr);
  ScCiSourceResult out;
  out.pseudodensities = assembleClasses(ai, src, batch, dm1, dm2, dm3, f3ac, f3ca,
                                        ScRotation{}, &adj, /*epsAdjoint=*/eps != nullptr, dr);

  // Read back, then the sigma vectors. downloadTensor synchronizes the stream
  // the sweep's einsums were issued on, so each tensor is complete when it
  // lands; the sigma calls that follow are the only GPU work left.
  Tensor g1 = downloadTensor(adj.dm1, s);
  // The orbital energies' response (#87): dE/de summed over the classes, then
  // carried onto dm1 through e_p's own dependence on it -- one more dm1
  // coefficient, which the fold below hands to the same 3-body sigma.
  if (eps != nullptr) {
    out.orbitalEnergiesMove = true;
    out.dEdCore.assign(static_cast<std::size_t>(eps->core.dim(2)), 0.0);
    out.dEdVirt.assign(static_cast<std::size_t>(eps->virt.dim(2)), 0.0);
    for (const ScClassGradient& c : out.pseudodensities.classes) {
      check(c.dEdCore.size() == out.dEdCore.size() && c.dEdVirt.size() == out.dEdVirt.size(),
            "scCiSourceDevice: a class's dE/de does not match the orbital-energy response");
      for (std::size_t i = 0; i < c.dEdCore.size(); ++i) out.dEdCore[i] += c.dEdCore[i];
      for (std::size_t r = 0; r < c.dEdVirt.size(); ++r) out.dEdVirt[r] += c.dEdVirt[r];
    }
    ciSourceAddInto(orbitalEnergyDm1Coefficient(*eps, out.dEdCore, out.dEdVirt), g1);
  }
  const Tensor g2 = downloadTensor(adj.dm2, s);
  const Tensor g3 = downloadTensor(adj.dm3, s);
  const Tensor gac = downloadTensor(adj.f3ac, s);
  const Tensor gca = downloadTensor(adj.f3ca, s);
  const Tensor h2e = downloadTensor(ai.h2e, s);
  out.y = ciSourceVector(sigma, ci, norb, nelecA, nelecB, g1, g2, g3, gac, gca, h2e, dr);
  return out;
}

ScGradientResult scPseudodensityDevice(const NevptIntegralsDevice& ints, const DeviceTensor& dm1,
                                       const DeviceTensor& dm2, const DeviceTensor& dm3,
                                       const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                       const DeviceResources& res) {
  FullBlockSource src(ints);
  // Views: `ints` keeps owning everything (as energiesDevice's own
  // integral-direct entry point does).
  ActiveIntegralsDevice ai{ints.h1e.view(),      ints.h2e.view(),      ints.e_core.view(),
                           ints.e_virt.view(),   ints.h1e_v_Sr.view(), ints.h1e_v_Si.view(),
                           ints.h1e_v_Sir.view()};
  return scPseudodensityDevice(ai, src, /*batch=*/0, dm1, dm2, dm3, f3ac, f3ca, res);
}

Result<ScRotationResult> scActiveRotationDevice(const ActiveIntegralsDevice& ai,
                                                IntegralSource& src, int64_t batch,
                                                const DeviceTensor& dm1, const DeviceTensor& dm2,
                                                const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                                const DeviceTensor& f3ca,
                                                const std::vector<ScRotationRequest>& dirs,
                                                const DeviceResources& dr) {
  const ScRotation rot = NEVPT2_TRY(makeRotation(dirs, ai.h1e, ai.h2e, dr));
  ScRotationResult out;
  out.grad =
      assembleClasses(ai, src, batch, dm1, dm2, dm3, f3ac, f3ca, rot, /*adj=*/nullptr,
                      /*epsAdjoint=*/false, dr);
  out.dirs.assign(dirs.size(), ScRotationPairing{});
  for (std::size_t d = 0; d < dirs.size(); ++d) {
    ScRotationPairing& p = out.dirs[d];
    for (const ScClassGradient& c : out.grad.classes) {
      p.ext += c.rotExt[d];
      p.act += c.rotAct[d];
      p.tangent += c.rotTangent[d];
    }
    p.adjoint = p.ext + p.act;
    p.hasTangent = dirs[d].tangent;
  }
  return out;
}

Result<ScRotationResult> scActiveRotationDevice(const NevptIntegralsDevice& ints,
                                                const DeviceTensor& dm1, const DeviceTensor& dm2,
                                                const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                                const DeviceTensor& f3ca,
                                                const std::vector<ScRotationRequest>& dirs,
                                                const DeviceResources& res) {
  FullBlockSource src(ints);
  // Views: `ints` keeps owning everything (scPseudodensityDevice's arrangement).
  ActiveIntegralsDevice ai{ints.h1e.view(),      ints.h2e.view(),      ints.e_core.view(),
                           ints.e_virt.view(),   ints.h1e_v_Sr.view(), ints.h1e_v_Si.view(),
                           ints.h1e_v_Sir.view()};
  return scActiveRotationDevice(ai, src, /*batch=*/0, dm1, dm2, dm3, f3ac, f3ca, dirs, res);
}

Result<Tensor> scActiveRotationMatrix(const NevptIntegralsDevice& ints, const DeviceTensor& dm1,
                                      const DeviceTensor& dm2, const DeviceTensor& dm3,
                                      const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                      int64_t chunk, const DeviceResources& res) {
  const int64_t n = dm1.dims[0];
  if (chunk < 1) return err_config("scActiveRotationMatrix: chunk must be at least 1");
  Tensor y({n, n});
  // The basis of antisymmetric directions, E_pq - E_qp for p < q, in `chunk`
  // sized batches: one slab walk per batch, the adjoint pairing only (no
  // tangent, so no second set of Dyall blocks is held).
  std::vector<ScRotationRequest> reqs;
  std::vector<std::pair<int64_t, int64_t>> which;
  const auto flush = [&]() -> Status {
    if (reqs.empty()) return {};
    const ScRotationResult r =
        NEVPT2_TRY(scActiveRotationDevice(ints, dm1, dm2, dm3, f3ac, f3ca, reqs, res));
    for (std::size_t k = 0; k < which.size(); ++k) {
      // <kappa, Y> = sum_pq kappa_pq Y_pq = 2 Y_pq for this direction.
      const double v = 0.5 * r.dirs[k].adjoint;
      y(which[k].first, which[k].second) = v;
      y(which[k].second, which[k].first) = -v;
    }
    reqs.clear();
    which.clear();
    return {};
  };
  for (int64_t p = 0; p < n; ++p)
    for (int64_t q = p + 1; q < n; ++q) {
      Tensor k({n, n});
      k(p, q) = 1.0;
      k(q, p) = -1.0;
      reqs.push_back(ScRotationRequest{std::move(k), /*tangent=*/false});
      which.emplace_back(p, q);
      if (std::ssize(reqs) == chunk) NEVPT2_TRY(flush());
    }
  NEVPT2_TRY(flush());
  return y;
}

// --- the full source term ----------------------------------------------------

// The eight-field integral set and the sidecar beside it, as views: the
// entry points below all build their own FullBlockSource and ActiveIntegrals
// view over the caller's `ints`, exactly as the active-block ones do.
ActiveIntegralsDevice activeViewOf(const NevptIntegralsDevice& ints) {
  return ActiveIntegralsDevice{ints.h1e.view(),      ints.h2e.view(),      ints.e_core.view(),
                               ints.e_virt.view(),   ints.h1e_v_Sr.view(), ints.h1e_v_Si.view(),
                               ints.h1e_v_Sir.view()};
}

// Every sidecar array's expected shape, against the integrals it sits beside.
// A sidecar paired with the wrong golden, or one whose header does not match
// its arrays, is the CALLER's input: an Error naming the first array that is
// off, before any device work.
Status checkGeneralShapes(const NevptIntegralsDevice& ints,
                          const GeneralIntegralsDevice& gen) {
  const int64_t nmo = gen.nmo(), nc = gen.ncore, na = gen.nact, nv = gen.nvirt;
  if (na != ints.h1e.dims[0])
    return err_config(std::format(
        "the gradient sidecar's active space is {} orbitals, the golden's is {}", na,
        ints.h1e.dims[0]));
  if (nc != ints.e_core.dims[0] || nv != ints.e_virt.dims[0])
    return err_config(std::format(
        "the gradient sidecar's partition is {} core / {} virtual, the golden's is {} / {}", nc,
        nv, ints.e_core.dims[0], ints.e_virt.dims[0]));
  const std::vector<std::pair<const char*, std::vector<int64_t>>> expect{
      {"h", {nmo, nmo}},
      {"heff", {nmo, nmo}},
      {"f", {nmo, nmo}},
      {"g_xa_aa", {nmo, na, na, na}},
      {"g_xv_aa", {nmo, nv, na, na}},
      {"g_xc_aa", {nmo, nc, na, na}},
      {"g_xa_va", {nmo, na, nv, na}},
      {"g_xv_va", {nmo, nv, nv, na}},
      {"g_xc_va", {nmo, nc, nv, na}},
      {"g_xa_ca", {nmo, na, nc, na}},
      {"g_xv_ca", {nmo, nv, nc, na}},
      {"g_xc_ca", {nmo, nc, nc, na}},
      {"g_xa_cv", {nmo, na, nc, nv}},
      {"g_xv_cv", {nmo, nv, nc, nv}},
      {"g_xc_cv", {nmo, nc, nc, nv}},
      {"g_xa_cc", {nmo, na, nc, nc}},
      {"g_xv_cc", {nmo, nv, nc, nc}},
      {"g_xc_cc_j", {nmo, nc, nc}},
      {"g_xc_cc_k", {nmo, nc, nc}},
      {"g_xa_vv", {nmo, na, nv}},
      {"g_xc_vv", {nmo, nc, nv}},
  };
  const std::vector<const DeviceTensor*> got{
      &gen.h,         &gen.heff,      &gen.f,         &gen.g_xa_aa,   &gen.g_xv_aa,
      &gen.g_xc_aa,   &gen.g_xa_va,   &gen.g_xv_va,   &gen.g_xc_va,   &gen.g_xa_ca,
      &gen.g_xv_ca,   &gen.g_xc_ca,   &gen.g_xa_cv,   &gen.g_xv_cv,   &gen.g_xc_cv,
      &gen.g_xa_cc,   &gen.g_xv_cc,   &gen.g_xc_cc_j, &gen.g_xc_cc_k, &gen.g_xa_vv,
      &gen.g_xc_vv};
  for (std::size_t k = 0; k < expect.size(); ++k) {
    if (got[k]->dims == expect[k].second) continue;
    std::string shape;
    for (const int64_t d : got[k]->dims) shape += (shape.empty() ? "" : ", ") + std::to_string(d);
    std::string want;
    for (const int64_t d : expect[k].second) want += (want.empty() ? "" : ", ") + std::to_string(d);
    return err_config(std::format(
        "the gradient sidecar's '{}' has shape [{}], but a {} core / {} active / {} virtual "
        "partition wants [{}]",
        expect[k].first, shape, nc, na, nv, want));
  }
  return {};
}

Result<ScFullRotationResult> scFullRotationDevice(const NevptIntegralsDevice& ints,
                                                  const GeneralIntegralsDevice& gen,
                                                  const DeviceTensor& dm1,
                                                  const DeviceTensor& dm2,
                                                  const DeviceTensor& dm3,
                                                  const DeviceTensor& f3ac,
                                                  const DeviceTensor& f3ca,
                                                  const DeviceTensor* digestAdjoint,
                                                  const std::vector<ScRotationRequest>& dirs,
                                                  const DeviceResources& dr) {
  NEVPT2_TRY(checkGeneralShapes(ints, gen));
  if (digestAdjoint != nullptr && digestAdjoint->dims != ints.h2e.dims)
    return err_config("scFullRotationDevice: the digests' adjoint is not shaped like h2e");
  FullBlockSource src(ints);
  const ActiveIntegralsDevice ai = activeViewOf(ints);
  const ScRotation rot =
      NEVPT2_TRY(makeGeneralRotation(dirs, ai.h1e, ai.h2e, gen, dm1, dr));
  ScFullRotationResult out;
  out.grad = assembleClasses(ai, src, /*batch=*/0, dm1, dm2, dm3, f3ac, f3ca, rot,
                             /*adj=*/nullptr, /*epsAdjoint=*/false, dr);
  out.dirs.assign(dirs.size(), ScFullRotationPairing{});
  for (std::size_t d = 0; d < dirs.size(); ++d) {
    ScFullRotationPairing& p = out.dirs[d];
    for (const ScClassGradient& c : out.grad.classes) {
      p.ext += c.rotExt[d];
      p.act += c.rotAct[d];
      p.eps += c.rotEps[d];
      p.tangentIntegral += c.rotTangent[d];
    }
    p.integral = p.ext + p.act;
    // The digests' term (#89): hbar against the same h2e_dot the Dyall
    // blocks read -- the intra-active action plus the sidecar's general part.
    if (digestAdjoint != nullptr) {
      p.digest = dotTensors(*digestAdjoint, rot.dirs[d].h2eDot, dr);
      p.hasDigest = true;
    }
    p.adjoint = p.integral + p.eps + p.digest;
    p.tangent = p.tangentIntegral + p.eps;
    p.hasTangent = dirs[d].tangent;
  }
  return out;
}

Result<Tensor> scFullRotationMatrix(const NevptIntegralsDevice& ints,
                                    const GeneralIntegralsDevice& gen, const DeviceTensor& dm1,
                                    const DeviceTensor& dm2, const DeviceTensor& dm3,
                                    const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                    const DeviceTensor* digestAdjoint,
                                    const std::vector<std::pair<int64_t, int64_t>>& pairs,
                                    int64_t chunk, const DeviceResources& res) {
  const int64_t n = gen.nmo();
  if (chunk < 1) return err_config("scFullRotationMatrix: chunk must be at least 1");
  Tensor y({n, n});
  std::vector<ScRotationRequest> reqs;
  std::vector<std::pair<int64_t, int64_t>> which;
  const auto flush = [&]() -> Status {
    if (reqs.empty()) return {};
    const ScFullRotationResult r =
        NEVPT2_TRY(scFullRotationDevice(ints, gen, dm1, dm2, dm3, f3ac, f3ca, digestAdjoint,
                                        reqs, res));
    for (std::size_t k = 0; k < which.size(); ++k) {
      // <kappa, Y> = sum_pq kappa_pq Y_pq = 2 Y_pq for this direction.
      const double v = 0.5 * r.dirs[k].adjoint;
      y(which[k].first, which[k].second) = v;
      y(which[k].second, which[k].first) = -v;
    }
    reqs.clear();
    which.clear();
    return {};
  };
  for (const auto& [p, q] : pairs) {
    if (p < 0 || q < 0 || p >= n || q >= n || p == q)
      return err_config(std::format(
          "scFullRotationMatrix: ({}, {}) is not a pair of distinct orbitals of {}", p, q, n));
    Tensor k({n, n});
    k(p, q) = 1.0;
    k(q, p) = -1.0;
    reqs.push_back(ScRotationRequest{std::move(k), /*tangent=*/false});
    which.emplace_back(p, q);
    if (std::ssize(reqs) == chunk) NEVPT2_TRY(flush());
  }
  NEVPT2_TRY(flush());
  return y;
}

Result<ScGeneralCheck> scGeneralIntegralCheck(const NevptIntegralsDevice& ints,
                                              const GeneralIntegralsDevice& gen,
                                              const DeviceTensor& dm1,
                                              const DeviceResources& res) {
  NEVPT2_TRY(checkGeneralShapes(ints, gen));
  // Every block the promotion table covers, against the stored array it is a
  // slice of. h1e and h2e are in the list too: their rules are the active
  // block's own, and the same restriction pins them.
  const std::vector<std::pair<RotBlock, const DeviceTensor*>> blocks{
      {RotBlock::Sr, &ints.h2e_v_Sr},       {RotBlock::Si, &ints.h2e_v_Si},
      {RotBlock::Sijrs, &ints.cvcv},        {RotBlock::Sijr, &ints.h2e_v_Sijr},
      {RotBlock::Srsi, &ints.h2e_v_Srsi},   {RotBlock::Srs, &ints.h2e_v_Srs},
      {RotBlock::Sij, &ints.h2e_v_Sij},     {RotBlock::Sir1, &ints.h2e_v1_Sir},
      {RotBlock::Sir2, &ints.h2e_v2_Sir},   {RotBlock::H2e, &ints.h2e},
      {RotBlock::H1e, &ints.h1e},           {RotBlock::H1eSr, &ints.h1e_v_Sr},
      {RotBlock::H1eSi, &ints.h1e_v_Si},    {RotBlock::H1eSir, &ints.h1e_v_Sir},
  };
  ScGeneralCheck out;
  for (const auto& [kind, stored] : blocks) {
    std::string where;
    const bool oneElectron = rotBlockInfo(kind).rank == 2;
    const double resid = oneElectron ? restrictionResidual2(kind, gen, *stored, res, where)
                                     : restrictionResidual4(kind, gen, *stored, res, where);
    if (resid > out.blocks) {
      out.blocks = resid;
      out.worstBlock = where;
    }
    if (oneElectron) out.heff = std::max(out.heff, heffResidual(kind, gen, res));
  }
  out.fock = std::max(fockDiagResidual(/*core=*/true, gen, dm1, res),
                      fockDiagResidual(/*core=*/false, gen, dm1, res));
  out.eps = std::max(epsResidual(/*core=*/true, gen, ints.e_core, res),
                     epsResidual(/*core=*/false, gen, ints.e_virt, res));
  return out;
}

bool printScFullRotationTable(const ScFullRotationResult& r, double atol) {
  std::printf(
      "\nThe full orbital source term Y_xy (Park Eqs. 26, 28), per direction\n"
      "  <kappa, Y>   the ADJOINT pairing, integral half + orbital energies\n"
      "  ext / act    the integral half's two pieces -- Eq. 41's D, Eqs. 42-43's M_G\n"
      "  eps          sum_t R_t Deltadot_t (Eqs. 38-39): shared by both modes\n"
      "  Edot         the forward TANGENT, the same classes' product rule + eps\n"
      "  |delta|      the INTEGRAL halves against each other: one derivative, two modes\n");
  std::printf("%-5s %22s %22s %22s %22s %22s %11s\n", "dir", "<kappa, Y>", "ext", "act", "eps",
              "Edot", "|delta|");
  bool ok = true;
  for (std::size_t d = 0; d < r.dirs.size(); ++d) {
    const ScFullRotationPairing& p = r.dirs[d];
    const double delta = std::fabs(p.integral - p.tangentIntegral);
    if (p.hasTangent) ok = ok && delta < atol;
    std::printf("%-5zu %22.14e %22.14e %22.14e %22.14e %22s %11s\n", d, p.adjoint, p.ext, p.act,
                p.eps, p.hasTangent ? std::format("{:.14e}", p.tangent).c_str() : "-",
                p.hasTangent ? std::format("{:.2e}", delta).c_str() : "-");
  }
  return ok;
}

// --- the digests' orbital response, forward mode (#88) ------------------------

Result<std::vector<ScDigestTangent>> scDigestRotationTangentDevice(
    const RdmBuildOptions& opt, const Tensor& ci, const int64_t nelecA, const int64_t nelecB,
    const NevptIntegralsDevice& ints, const GeneralIntegralsDevice& gen, const DeviceTensor& dm1,
    const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceTensor& f3ac,
    const DeviceTensor& f3ca, const std::vector<ScRotationRequest>& dirs,
    const DeviceResources& res) {
  // A tangent built from an approximate digest is not the derivative of
  // anything the energy stage computed; nevpt2.cli refuses the combination,
  // so reaching here with one is our bug.
  check(!opt.cublas && !opt.ozaki && !opt.fusedDigest,
        "scDigestRotationTangentDevice runs the emitted or the native-fp64 BLAS digest "
        "(cli refuses --cublas / --ozaki / --fused-digest with --kappa-full and "
        "--dump-source-terms)");
  NEVPT2_TRY(checkGeneralShapes(ints, gen));
  const wwrStream_t s = res.stream();
  const int64_t norb = dm1.dims[0];
  const ActiveIntegralsDevice ai = activeViewOf(ints);
  // h2e_dot(kappa) per direction, exactly as the frozen-digest tangent's Dyall
  // blocks read it: the intra-active action plus the sidecar's general part.
  const ScRotation rot = NEVPT2_TRY(makeGeneralRotation(dirs, ai.h1e, ai.h2e, gen, dm1, res));

  // The other RDMs' tangents are zero: a rotation at fixed CI moves no RDM,
  // and only the digests' h2e dependence is differentiated here. Allocated
  // once, zero-filled by their allocation, and only ever read.
  const DeviceTensor dm1Zero = DeviceTensor::zeros(dm1.dims, res);
  const DeviceTensor dm2Zero = DeviceTensor::zeros(dm2.dims, res);
  const DeviceTensor dm3Zero = DeviceTensor::zeros(dm3.dims, res);

  std::vector<ScDigestTangent> out(dirs.size());
  for (std::size_t d = 0; d < rot.dirs.size(); ++d) {
    // downloadTensor synchronizes the stream, so the rotation's einsums are
    // complete; the build takes the integrals on the host, as the plain
    // build's caller hands it the golden's.
    const Tensor h2eDot = downloadTensor(rot.dirs[d].h2eDot, s);
    for (int64_t i = 0; i < h2eDot.size(); ++i)
      out[d].h2eDotMax = std::max(out[d].h2eDotMax, std::fabs(h2eDot.flat(i)));
    std::printf("\ndigest tangent, direction %zu: the RDM build at h2e_dot (max |h2e_dot| = "
                "%.3e)\n",
                d, out[d].h2eDotMax);
    // f3ac_dot / f3ca_dot: the build is linear in h2e at a fixed CI vector.
    // Its dm3 reads no integral and is not used.
    const RdmBuildResult dot = buildRdmsDevice(opt, ci, norb, nelecA, nelecB, h2eDot, res);
    const EnergyTangentResult t = energiesTangentDevice(
        ints, dm1, dm2, dm3, f3ac, f3ca, dm1Zero, dm2Zero, dm3Zero, dot.f3ac, dot.f3ca,
        /*epsDot=*/nullptr, res);
    out[d].classes = t.energyDots;
    out[d].total = t.totalDot;
  }
  return out;
}

bool printScDigestTangentTable(const ScFullRotationResult& full,
                               const std::vector<ScDigestTangent>& digest, const double atol) {
  check(full.dirs.size() == digest.size(),
        "printScDigestTangentTable: the two answers are for different direction lists");
  std::printf(
      "\nEdot(kappa) in its two terms, per direction (#88), and the full duality (#89)\n"
      "  frozen       the tangent above: integral half + eps, f3ac/f3ca held fixed\n"
      "  digest       the f3ac/f3ca digests' own response: the RDM build at h2e_dot,\n"
      "               through the classes' H side (energiesTangentDevice)\n"
      "  Edot         frozen + digest: the exact directional derivative at fixed CI\n"
      "  rev digest   the same term in REVERSE mode: <hbar, h2e_dot>, hbar the digests'\n"
      "               transpose (scDigestAdjointDevice) -- part of <kappa, Y>\n"
      "  |full delta| <kappa, Y> (integral + eps + rev digest) against Edot\n");
  std::printf("%-5s %22s %22s %22s %22s %12s %12s\n", "dir", "frozen", "digest", "Edot",
              "rev digest", "|full delta|", "max|h2e_dot|");
  bool ok = true;
  for (std::size_t d = 0; d < digest.size(); ++d) {
    const ScFullRotationPairing& p = full.dirs[d];
    const double frozen = p.tangent;
    const double edot = frozen + digest[d].total;
    std::string rev = "-";
    std::string gap = "-";
    if (p.hasDigest) {
      const double delta = std::fabs(p.adjoint - edot);
      ok = ok && delta < atol;
      rev = std::format("{:.14e}", p.digest);
      gap = std::format("{:.2e}", delta);
    }
    std::printf("%-5zu %22.14e %22.14e %22.14e %22s %12s %12.3e\n", d, frozen, digest[d].total,
                edot, rev.c_str(), gap.c_str(), digest[d].h2eDotMax);
  }
  return ok;
}

// --- the digests' orbital response, reverse mode (#89) ------------------------

DeviceTensor scDigestAdjointDevice(const RdmBuildOptions& opt, const Tensor& ci,
                                   const int64_t nelecA, const int64_t nelecB,
                                   const NevptIntegralsDevice& ints, const DeviceTensor& dm1,
                                   const DeviceTensor& dm2, const DeviceTensor& dm3,
                                   const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                   const DeviceResources& res) {
  const int64_t norb = dm1.dims[0];
  FullBlockSource src(ints);
  const ActiveIntegralsDevice ai = activeViewOf(ints);
  // The walk and its RDM sweep, exactly as scCiSourceDevice runs them: of
  // the five coefficient tensors only the two digests' are read here.
  ScRdmAdjoint adj = ScRdmAdjoint::zeros(norb, res);
  (void)assembleClasses(ai, src, /*batch=*/0, dm1, dm2, dm3, f3ac, f3ca, ScRotation{}, &adj,
                        /*epsAdjoint=*/false, res);
  return buildDigestAdjointDevice(opt, ci, norb, nelecA, nelecB, dm3, adj.f3ac, adj.f3ca, res);
}

bool printScRotationTable(const ScRotationResult& r, double atol) {
  std::printf(
      "\nThe active-rotation source term Y_act (Park Eq. 26), per direction\n"
      "  <kappa, Y>   the ADJOINT pairing: sum_blocks <D, Xdot> + sum_G <M_G, Gdot>\n"
      "  ext / act    its two halves -- Eq. 41's external D, Eqs. 42-43's M_G\n"
      "  Edot         the forward TANGENT of the same classes along kappa\n"
      "  |delta|      the two against each other: one derivative, two modes\n");
  // Scientific notation throughout: an active rotation that mixes two
  // DIFFERENT irreps moves the energy not at all (the source term's entry is
  // zero by symmetry, not small), and a fixed-point column could not tell
  // 1e-16 from 1e-6.
  std::printf("%-5s %22s %22s %22s %22s %11s\n", "dir", "<kappa, Y>", "ext", "act", "Edot",
              "|delta|");
  bool ok = true;
  for (std::size_t d = 0; d < r.dirs.size(); ++d) {
    const ScRotationPairing& p = r.dirs[d];
    const double delta = std::fabs(p.adjoint - p.tangent);
    if (p.hasTangent) ok = ok && delta < atol;
    std::printf("%-5zu %22.14e %22.14e %22.14e %22s %11s\n", d, p.adjoint, p.ext, p.act,
                p.hasTangent ? std::format("{:.14e}", p.tangent).c_str() : "-",
                p.hasTangent ? std::format("{:.2e}", delta).c_str() : "-");
  }
  return ok;
}

ScCiSourceResult scCiSourceDevice(const NevptIntegralsDevice& ints, const Tensor& ci,
                                  const int64_t nelecA, const int64_t nelecB,
                                  const DeviceTensor& dm1, const DeviceTensor& dm2,
                                  const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                  const DeviceTensor& f3ca, const SigmaVectorOptions& sigma,
                                  const OrbitalEnergyResponse* eps, const DeviceResources& res) {
  FullBlockSource src(ints);
  ActiveIntegralsDevice ai{ints.h1e.view(),      ints.h2e.view(),      ints.e_core.view(),
                           ints.e_virt.view(),   ints.h1e_v_Sr.view(), ints.h1e_v_Si.view(),
                           ints.h1e_v_Sir.view()};
  return scCiSourceDevice(ai, src, /*batch=*/0, ci, nelecA, nelecB, dm1, dm2, dm3, f3ac, f3ca,
                          sigma, eps, res);
}

bool printScGradientReport(const ScGradientResult& g, const EnergyResult& energies,
                           const Tensor& ref, double atol, double idtol) {
  std::printf(
      "\nSC-NEVPT2 pseudodensities (Park Eqs. 41-47), per class\n"
      "  E(T)        the class energy from the amplitudes, sum_t T_t N_t (Eq. 15)\n"
      "  |d E|       against nevpt2.energy's own answer for the same state\n"
      "  |d ref|     against the golden PySCF per-class energy\n"
      "  ext resid   E(T) - (1/2) sum_blocks <x, D>        (Eq. 40, Eq. 41's D)\n"
      "  act resid   sum_G <G, M_G> - its reference         (Eqs. 42-43)\n"
      "  int resid   the active integrals' and hole RDMs' own pseudodensities\n"
      "              contracted back, against <K, E>        (Eqs. 44-47, Srs)\n");
  std::printf("%-7s %16s %16s %10s %10s %11s %11s %11s\n", "class", "E(T) (Ha)",
              "ref energy (Ha)", "|d E|", "|d ref|", "ext resid", "act resid", "int resid");
  bool ok = true;
  for (int c = 0; c < 8; ++c) {
    const ScClassGradient& k = g.classes[c];
    const double refE = ref.flat(c);
    const double dE = std::fabs(k.energy - energies.energies[c]);
    const double dRef = std::fabs(k.energy - refE);
    const double extResid = std::fabs(k.energy - k.extContraction);
    const double actResid = std::fabs(k.actContraction - k.actReference);
    const double intResid = std::max(std::fabs(k.intContraction - k.intReference),
                                     std::fabs(k.rdmContraction - k.intReference));
    ok = ok && dE < atol && dRef < 1e-7 && extResid < idtol;
    if (k.hasActive) ok = ok && actResid < idtol;
    if (k.hasActiveIntegrals) ok = ok && intResid < idtol;
    const std::string act = k.hasActive ? std::format("{:.2e}", actResid) : std::string("-");
    const std::string integ =
        k.hasActiveIntegrals ? std::format("{:.2e}", intResid) : std::string("-");
    std::printf("%-7s %16.10f %16.10f %10.2e %10.2e %11.2e %11s %11s\n", CLASSES[c].c_str(),
                k.energy, refE, dE, dRef, extResid, act.c_str(), integ.c_str());
  }
  std::printf("\nE_corr (from the amplitudes) = %.10f Ha\n", g.total);
  std::printf("E_corr (nevpt2.energy)       = %.10f Ha  |delta|=%.2e\n", energies.total,
              std::fabs(g.total - energies.total));
  std::printf(
      "\n%s: SC pseudodensity identities hold to %.0e, and every class energy "
      "matches nevpt2.energy to %.0e and the golden to 1e-07\n",
      ok ? "PASS" : "FAIL", idtol, atol);
  return ok;
}

}  // namespace nevpt2
