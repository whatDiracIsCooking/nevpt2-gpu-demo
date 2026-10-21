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
// The host dm1/dm2 tangents along a CI direction: the mixed block's forward
// orientation builds u's itself (#184).
import nevpt2.rdm_host_dm;
import :ci_source;    // Park Eq. 48's sigma assembly
import :curvature;    // the orbital block's second-order input generators (#154)
import :general;      // the promotion table the general-range rotation reads
import :rdm_adjoint;  // the block -> RDM chain rule the CI source needs
import :rotation;
import :sc_classes;
import :terms;
import :pc_classes;   // PC-NEVPT2's reverse mode (#74)

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
// `ci` non-null ALSO differentiates the walk and its sweep along a CI
// direction, accumulating the coefficient tensors' tangent into ci->adjDot
// and each class's Edot(u) into ScClassGradient::ciTangent (#156; the
// :rdm_adjoint partition's ScCiDirection). Independent of the other three.
// `mix` non-null carries the mixed block's direction (#184, :rdm_adjoint's
// ScMixedDirection): the orbital direction is rot.dirs[0], and each class adds
// its share of Eddot_kc(w, u) (forward) or of the walk's tangent along w
// (reverse) -- see scMixedHessianDevice / scMixedHessianVectorDevice below.
ScGradientResult assembleClasses(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                 int64_t batch, const DeviceTensor& dm1,
                                 const DeviceTensor& dm2, const DeviceTensor& dm3,
                                 const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                 const ScRotation& rot, ScRdmAdjoint* adj, bool epsAdjoint,
                                 const ScCiDirection* ci, const ScMixedDirection* mix,
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
    // dE/de's tangent along the mixed direction's w (#184).
    if (mix != nullptr && mix->reverse && mix->epsCi) {
      c.dEdCoreDot.assign(static_cast<std::size_t>(hostECore.size()), 0.0);
      c.dEdVirtDot.assign(static_cast<std::size_t>(hostEVirt.size()), 0.0);
    }
  }
  gradientSr(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dm3, f3ac, f3ca, norb, rot,
             r.classes[0], adj, ci, mix, dr);
  gradientSi(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dm3, f3ac, f3ca, norb, rot,
             r.classes[1], adj, ci, mix, dr);
  // Sijrs contracts no RDM at all, so it has nothing to sweep and takes no
  // `adj` -- nor `ci`: its every CI derivative is exactly zero at fixed
  // orbital energies. It DOES take `rot`: every one of cvcv's four labels is
  // core or virtual, so an ACTIVE rotation moves nothing it reads while a
  // GENERAL one moves all of it, and its Delta_t is four orbital energies
  // (gradient_sc_classes.cppm).
  gradientSijrs(src, batch, hostECore, hostEVirt, rot, r.classes[2], mix, dr);
  gradientSijr(ai, src, batch, hostECore, hostEVirt, dm1, dm2, rot, r.classes[3], adj, ci,
               mix, dr);
  gradientSrsi(ai, src, batch, hostECore, hostEVirt, dm1, dm2, rot, r.classes[4], adj, ci,
               mix, dr);
  gradientSrs(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dm3, rot, r.classes[5], adj,
              ci, mix, dr);
  gradientSij(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dm3, rot, r.classes[6], adj,
              ci, mix, dr);
  gradientSir(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dm3, rot, r.classes[7], adj,
              ci, mix, dr);
  for (const ScClassGradient& c : r.classes) r.total += c.energy;
  return r;
}

ScGradientResult scPseudodensityDevice(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                       int64_t batch, const DeviceTensor& dm1,
                                       const DeviceTensor& dm2, const DeviceTensor& dm3,
                                       const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                       const DeviceResources& dr) {
  return assembleClasses(ai, src, batch, dm1, dm2, dm3, f3ac, f3ca, ScRotation{},
                         /*adj=*/nullptr, /*epsAdjoint=*/false, /*ci=*/nullptr,
                         /*mix=*/nullptr, dr);
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
                                        ScRotation{}, &adj, /*epsAdjoint=*/eps != nullptr,
                                        /*ci=*/nullptr, /*mix=*/nullptr, dr);

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

ScCiHessianResult scCiHessianVectorDevice(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                          int64_t batch, const Tensor& ci, const Tensor& u,
                                          int64_t nelecA, int64_t nelecB, const RdmTensors& rdm,
                                          const RdmTensors& dotU,
                                          const SigmaVectorOptions& sigma,
                                          const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t norb = rdm.dm1.dims[0];
  check(u.dims() == ci.dims(), "scCiHessianVectorDevice: u is not shaped like the CI vector");
  // g and its tangent along u, zero-filled and accumulated over the eight
  // classes by ONE walk: the plain sweep into `adj`, the tangent's into
  // `adjDot`. The orbital energies are held fixed (epsAdjoint off, and the
  // multiplier tangents' Deltadot = 0).
  ScRdmAdjoint adj = ScRdmAdjoint::zeros(norb, dr);
  ScRdmAdjoint adjDot = ScRdmAdjoint::zeros(norb, dr);
  const ScCiDirection dir{dotU, adjDot};
  ScCiHessianResult out;
  out.pseudodensities =
      assembleClasses(ai, src, batch, rdm.dm1, rdm.dm2, rdm.dm3, rdm.f3ac, rdm.f3ca,
                      ScRotation{}, &adj, /*epsAdjoint=*/false, &dir, /*mix=*/nullptr, dr);
  for (std::size_t c = 0; c < out.edotU.size(); ++c) {
    out.edotU[c] = out.pseudodensities.classes[c].ciTangent;
    out.edotUTotal += out.edotU[c];
  }

  // Read back (downloadTensor synchronizes the stream the sweeps' einsums
  // were issued on), then the three sigma calls -- the only GPU work left.
  const Tensor g1 = downloadTensor(adj.dm1, s);
  const Tensor g2 = downloadTensor(adj.dm2, s);
  const Tensor g3 = downloadTensor(adj.dm3, s);
  const Tensor gac = downloadTensor(adj.f3ac, s);
  const Tensor gca = downloadTensor(adj.f3ca, s);
  out.gDot = ScRdmCoefficients{downloadTensor(adjDot.dm1, s), downloadTensor(adjDot.dm2, s),
                               downloadTensor(adjDot.dm3, s), downloadTensor(adjDot.f3ac, s),
                               downloadTensor(adjDot.f3ca, s)};
  const Tensor h2e = downloadTensor(ai.h2e, s);
  const ScRdmCoefficients& gd = out.gDot;
  out.y = ciSourceVector(sigma, ci, norb, nelecA, nelecB, g1, g2, g3, gac, gca, h2e, dr);
  out.fromWalkTangent = ciSourceVector(sigma, ci, norb, nelecA, nelecB, gd.dm1, gd.dm2,
                                       gd.dm3, gd.f3ac, gd.f3ca, h2e, dr);
  out.fromDirection =
      ciSourceVector(sigma, u, norb, nelecA, nelecB, g1, g2, g3, gac, gca, h2e, dr);
  out.hu = out.fromWalkTangent;
  ciSourceAddInto(out.fromDirection, out.hu);
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
                      /*epsAdjoint=*/false, /*ci=*/nullptr, /*mix=*/nullptr, dr);
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
                             /*adj=*/nullptr, /*epsAdjoint=*/false, /*ci=*/nullptr,
                             /*mix=*/nullptr, dr);
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
  out.fockFull = fockFullResidual(gen, dm1, res);
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

Result<std::vector<ScDigestTangent>> digestRotationTangents(
    const RdmBuildOptions& opt, const Tensor& ci, int64_t nelecA, int64_t nelecB,
    const NevptIntegralsDevice& ints, const ScRotation& rot, const DeviceTensor& dm1,
    const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceTensor& f3ac,
    const DeviceTensor& f3ca, const DeviceResources& res);

ScDigestTangent digestTangentAt(const RdmBuildOptions& opt, const Tensor& ci, int64_t nelecA,
                                int64_t nelecB, const NevptIntegralsDevice& ints,
                                const DeviceTensor& h2eDotDevice, const DeviceTensor& dm1,
                                const DeviceTensor& dm2, const DeviceTensor& dm3,
                                const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                const DeviceTensor& dm1Zero, const DeviceTensor& dm2Zero,
                                const DeviceTensor& dm3Zero, std::optional<std::size_t> announce,
                                const DeviceResources& res);

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
  const ActiveIntegralsDevice ai = activeViewOf(ints);
  // h2e_dot(kappa) per direction, exactly as the frozen-digest tangent's Dyall
  // blocks read it: the intra-active action plus the sidecar's general part.
  const ScRotation rot = NEVPT2_TRY(makeGeneralRotation(dirs, ai.h1e, ai.h2e, gen, dm1, res));
  return digestRotationTangents(opt, ci, nelecA, nelecB, ints, rot, dm1, dm2, dm3, f3ac, f3ca,
                                res);
}

Result<std::vector<ScDigestTangent>> scDigestRotationTangentDevice(
    const RdmBuildOptions& opt, const Tensor& ci, const int64_t nelecA, const int64_t nelecB,
    const NevptIntegralsDevice& ints, const DeviceTensor& dm1, const DeviceTensor& dm2,
    const DeviceTensor& dm3, const DeviceTensor& f3ac, const DeviceTensor& f3ca,
    const std::vector<ScRotationRequest>& dirs, const DeviceResources& res) {
  check(!opt.cublas && !opt.ozaki && !opt.fusedDigest,
        "scDigestRotationTangentDevice runs the emitted or the native-fp64 BLAS digest "
        "(cli refuses --cublas / --ozaki / --fused-digest with --sc-hessian-kc)");
  // The active-active block alone: h2e_dot is the intra-active action, which
  // needs nothing the golden lacks.
  const ScRotation rot = NEVPT2_TRY(makeRotation(dirs, ints.h1e, ints.h2e, res));
  return digestRotationTangents(opt, ci, nelecA, nelecB, ints, rot, dm1, dm2, dm3, f3ac, f3ca,
                                res);
}

// The body both overloads share: per direction, the RDM build at its h2e_dot
// and the energy's response to that digest displacement alone.
Result<std::vector<ScDigestTangent>> digestRotationTangents(
    const RdmBuildOptions& opt, const Tensor& ci, const int64_t nelecA, const int64_t nelecB,
    const NevptIntegralsDevice& ints, const ScRotation& rot, const DeviceTensor& dm1,
    const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceTensor& f3ac,
    const DeviceTensor& f3ca, const DeviceResources& res) {
  // The other RDMs' tangents are zero: a rotation at fixed CI moves no RDM,
  // and only the digests' h2e dependence is differentiated here. Allocated
  // once, zero-filled by their allocation, and only ever read.
  const DeviceTensor dm1Zero = DeviceTensor::zeros(dm1.dims, res);
  const DeviceTensor dm2Zero = DeviceTensor::zeros(dm2.dims, res);
  const DeviceTensor dm3Zero = DeviceTensor::zeros(dm3.dims, res);

  std::vector<ScDigestTangent> out(rot.size());
  for (std::size_t d = 0; d < rot.dirs.size(); ++d)
    out[d] = digestTangentAt(opt, ci, nelecA, nelecB, ints, rot.dirs[d].h2eDot, dm1, dm2, dm3,
                             f3ac, f3ca, dm1Zero, dm2Zero, dm3Zero, d, res);
  return out;
}

// One h2e increment's share of the above: the RDM build at `h2eDot` and the
// energy's response to that digest displacement alone. `announce` labels the
// progress line (#88's walks print one per direction); the orbital
// curvature's terms (#154) pass none. dm*Zero are the zero RDM tangents, the
// caller's so a loop allocates them once.
ScDigestTangent digestTangentAt(const RdmBuildOptions& opt, const Tensor& ci, const int64_t nelecA,
                                const int64_t nelecB, const NevptIntegralsDevice& ints,
                                const DeviceTensor& h2eDotDevice, const DeviceTensor& dm1,
                                const DeviceTensor& dm2, const DeviceTensor& dm3,
                                const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                const DeviceTensor& dm1Zero, const DeviceTensor& dm2Zero,
                                const DeviceTensor& dm3Zero, std::optional<std::size_t> announce,
                                const DeviceResources& res) {
  const int64_t norb = dm1.dims[0];
  ScDigestTangent out;
  // downloadTensor synchronizes the stream, so the einsums that wrote the
  // increment are complete; the build takes the integrals on the host, as the
  // plain build's caller hands it the golden's.
  const Tensor h2eDot = downloadTensor(h2eDotDevice, res.stream());
  for (int64_t i = 0; i < h2eDot.size(); ++i)
    out.h2eDotMax = std::max(out.h2eDotMax, std::fabs(h2eDot.flat(i)));
  if (announce)
    std::printf("\ndigest tangent, direction %zu: the RDM build at h2e_dot (max |h2e_dot| = "
                "%.3e)\n",
                *announce, out.h2eDotMax);
  // f3ac_dot / f3ca_dot: the build is linear in h2e at a fixed CI vector.
  // Its dm3 reads no integral and is not used.
  const RdmBuildResult dot = buildRdmsDevice(opt, ci, norb, nelecA, nelecB, h2eDot, res);
  const EnergyTangentResult t =
      energiesTangentDevice(ints, dm1, dm2, dm3, f3ac, f3ca, dm1Zero, dm2Zero, dm3Zero, dot.f3ac,
                            dot.f3ca, /*epsDot=*/nullptr, res);
  out.classes = t.energyDots;
  out.total = t.totalDot;
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
                        /*epsAdjoint=*/false, /*ci=*/nullptr, /*mix=*/nullptr, res);
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

// --- the mixed orbital-CI block, H_kc (#184) -----------------------------------

// What both orientations build once per orbital direction w: the direction
// itself (its dotted integrals, slab-dot column blocks and deps; its tangent
// blocks are built per class, with the digests' share merged), the digests'
// tangent f3'[w] that merge reads, h2e'[w] on the host (C4's and R1's 4-body
// sigmas take it there), W'[w] when the orbital energies respond to both
// kappa and the CI vector (C5's generator), and the two zero integral blocks
// the digest-only builds and R1's integral-free sweep read.
struct ScMixedPrep {
  ScRotation rot;
  std::vector<RdmBuildResult> digests;  // owns rot.dirs[0]'s f3acDot / f3caDot
  Tensor h2eDot;                        // h2e'[w], host
  double h1eDotMax = 0.0;
  double h2eDotMax = 0.0;
  bool wDotSet = false;
  OrbitalEnergyResponse wDot;           // W'[w], when epsKappa and epsCi
  DeviceTensor zeroH1e, zeroH2e;
};

double maxAbs(const Tensor& t) {
  double m = 0.0;
  for (int64_t i = 0; i < t.size(); ++i) m = std::max(m, std::fabs(t.flat(i)));
  return m;
}

// The setup's own consistency and the refusals, then the prep. Everything a
// caller can get wrong -- w's shape or antisymmetry, a sidecar beside the
// wrong integrals, a mode or a w that needs a sidecar it did not hand in, a
// missing or misshapen W -- is an Error; an approximate digest is our bug
// (cli refuses the flags that reach here).
Result<ScMixedPrep> prepareMixed(const ScMixedHessianSetup& s, const Tensor& w,
                                 const DeviceResources& res) {
  check(!s.opt.cublas && !s.opt.ozaki && !s.opt.fusedDigest,
        "the mixed block H_kc runs the emitted or the native-fp64 BLAS digest (cli refuses "
        "--cublas / --ozaki / --fused-digest with --sc-hessian-kc)");
  const NevptIntegralsDevice& ints = s.ints;
  const int64_t nc = ints.e_core.dims[0], na = ints.h1e.dims[0], nv = ints.e_virt.dims[0];
  const int64_t nmo = nc + na + nv;
  if (w.rank() != 2 || w.dim(0) != nmo || w.dim(1) != nmo)
    return err_config(std::format("the mixed block's orbital direction is not a {} x {} kappa "
                                  "(n_mo = {} core + {} active + {} virtual)",
                                  nmo, nmo, nc, na, nv));
  if (s.epsCi && s.epsResponse == nullptr)
    return err_config(
        "the mixed block with the orbital energies responding to the CI vector needs their "
        "response W (an OrbitalEnergyResponse, out of the gradient sidecar)");
  if (s.epsResponse != nullptr &&
      (s.epsResponse->core.dims() != std::vector<int64_t>{na, na, nc} ||
       s.epsResponse->virt.dims() != std::vector<int64_t>{na, na, nv}))
    return err_config("the orbital energies' response W does not fit this partition");

  ScMixedPrep p;
  const ScRotationRequest req{w, /*tangent=*/true};
  if (s.gen == nullptr) {
    // No sidecar: the active-active block alone, the orbital energies frozen.
    // Anything else is REFUSED, not answered with the part that can be formed
    // (docs/hessian-theory.md, Secs. 5.5 and 9.1).
    if (s.epsKappa || s.epsCi)
      return err_unsupported(
          "the mixed block H_kc with the orbital energies responding needs the gradient "
          "sidecar (deps and W read its integral rows); without one only the "
          "frozen-orbital-energy functional along an active-active w can be formed");
    double outside = 0.0;
    for (int64_t x = 0; x < nmo; ++x)
      for (int64_t y = 0; y < nmo; ++y) {
        const bool act = x >= nc && x < nc + na && y >= nc && y < nc + na;
        if (!act) outside = std::max(outside, std::fabs(w(x, y)));
      }
    if (outside > 0.0)
      return err_unsupported(std::format(
          "the mixed block H_kc along an orbital direction that leaves the active space (max "
          "|w| outside the active-active block = {:.3e}) needs the gradient sidecar's "
          "one-general-index integrals; without one only an active-active w can be formed",
          outside));
    Tensor wAA({na, na});
    for (int64_t x = 0; x < na; ++x)
      for (int64_t y = 0; y < na; ++y) wAA(x, y) = w(nc + x, nc + y);
    p.rot = NEVPT2_TRY(makeRotation({ScRotationRequest{wAA, /*tangent=*/true}}, ints.h1e,
                                    ints.h2e, res));
  } else {
    NEVPT2_TRY(checkGeneralShapes(ints, *s.gen));
    p.rot = NEVPT2_TRY(makeGeneralRotation({req}, ints.h1e, ints.h2e, *s.gen, s.rdm.dm1, res));
  }
  ScRotationDir& d = p.rot.dirs[0];
  p.digests = attachDigestTangents(p.rot, s.opt, s.ci, s.nelecA, s.nelecB, res);
  p.h2eDot = downloadTensor(d.h2eDot, res.stream());
  p.h2eDotMax = maxAbs(p.h2eDot);
  p.h1eDotMax = maxAbs(downloadTensor(d.h1eDot, res.stream()));
  if (s.epsKappa && s.epsCi) {
    p.wDot = generalEpsResponseTangent(*s.gen, d.kCore, d.kActFull, d.kVirt, res);
    p.wDotSet = true;
  }
  p.zeroH1e = DeviceTensor::zeros(ints.h1e.dims, res);
  p.zeroH2e = DeviceTensor::zeros(ints.h2e.dims, res);
  return p;
}

// The class-independent half of the mixed direction both orientations hand
// every class.
ScMixedDirection mixedDirectionOf(const ScMixedHessianSetup& s, const ScMixedPrep& p) {
  ScMixedDirection m;
  m.epsKappa = s.epsKappa;
  m.epsCi = s.epsCi;
  m.zeroH1e = &p.zeroH1e;
  m.zeroH2e = &p.zeroH2e;
  const int64_t nc = s.ints.e_core.dims[0], nv = s.ints.e_virt.dims[0];
  m.epsCoreU = Tensor({nc});
  m.epsVirtU = Tensor({nv});
  m.epsCoreWU = Tensor({nc});
  m.epsVirtWU = Tensor({nv});
  return m;
}

Result<std::vector<ScMixedHessianResult>> scMixedHessianDevice(const ScMixedHessianSetup& s,
                                                               const Tensor& w,
                                                               const std::vector<Tensor>& us,
                                                               const DeviceResources& res) {
  ScMixedPrep p = NEVPT2_TRY(prepareMixed(s, w, res));
  const NevptIntegralsDevice& ints = s.ints;
  const int64_t norb = ints.h1e.dims[0];
  Result<HostDm12Builder> builder = HostDm12Builder::create(norb, s.nelecA, s.nelecB);
  if (!builder) return std::unexpected(builder.error());
  const Tensor h2e = downloadTensor(ints.h2e, res.stream());
  FullBlockSource src(ints);
  const ActiveIntegralsDevice ai = activeViewOf(ints);

  std::vector<ScMixedHessianResult> out;
  out.reserve(us.size());
  for (const Tensor& u : us) {
    if (u.dims() != s.ci.dims())
      return err_config("scMixedHessianDevice: a CI direction is not shaped like the state");
    // u's tangent RDMs: dm1/dm2 on the host, dm3/f3ac/f3ca the tangent build
    // at (c, u) and h2e -- and the digests' MIXED tangent f3^{wu}, the same
    // build handed h2e'[w] (linear in h2e; its dm3 reads no integral and is
    // the first build's again, so it is not used).
    const HostDm12 dot = builder->dm12Dot(s.ci, u);
    const DeviceTensor dm1U = uploadTensor(dot.dm1, res);
    const DeviceTensor dm2U = uploadTensor(dot.dm2, res);
    const RdmBuildResult tU =
        buildRdmTangentsDevice(s.opt, s.ci, u, norb, s.nelecA, s.nelecB, h2e, res);
    const RdmBuildResult tWU =
        buildRdmTangentsDevice(s.opt, s.ci, u, norb, s.nelecA, s.nelecB, p.h2eDot, res);
    const RdmTensors dotU{dm1U, dm2U, tU.dm3, tU.f3ac, tU.f3ca};

    ScMixedDirection mix = mixedDirectionOf(s, p);
    mix.forward = true;
    mix.dotU = &dotU;
    mix.f3acWU = &tWU.f3ac;
    mix.f3caWU = &tWU.f3ca;
    if (s.epsCi) {
      const OrbitalEnergyDot eU = orbitalEnergyDot(*s.epsResponse, dot.dm1);
      mix.epsCoreU = eU.core;
      mix.epsVirtU = eU.virt;
    }
    if (p.wDotSet) {
      const OrbitalEnergyDot eWU = orbitalEnergyDot(p.wDot, dot.dm1);
      mix.epsCoreWU = eWU.core;
      mix.epsVirtWU = eWU.virt;
    }
    const ScGradientResult g =
        assembleClasses(ai, src, /*batch=*/0, s.rdm.dm1, s.rdm.dm2, s.rdm.dm3, s.rdm.f3ac,
                        s.rdm.f3ca, p.rot, /*adj=*/nullptr, /*epsAdjoint=*/false,
                        /*ci=*/nullptr, &mix, res);
    ScMixedHessianResult r;
    for (std::size_t c = 0; c < 8; ++c) {
      r.classes[c] = g.classes[c].mixed;
      r.total += g.classes[c].mixed;
      r.edotW[c] = g.classes[c].mixedEdotW;
      r.edotU[c] = g.classes[c].mixedEdotU;
      r.edotWTotal += r.edotW[c];
      r.edotUTotal += r.edotU[c];
    }
    r.eddot = r.total.total();
    r.h1eDotMax = p.h1eDotMax;
    r.h2eDotMax = p.h2eDotMax;
    out.push_back(r);
  }
  return out;
}

Result<ScMixedHessianVectorResult> scMixedHessianVectorDevice(const ScMixedHessianSetup& s,
                                                              const Tensor& w,
                                                              const SigmaVectorOptions& sigma,
                                                              const DeviceResources& res) {
  ScMixedPrep p = NEVPT2_TRY(prepareMixed(s, w, res));
  const wwrStream_t st = res.stream();
  const NevptIntegralsDevice& ints = s.ints;
  const int64_t norb = ints.h1e.dims[0];
  FullBlockSource src(ints);
  const ActiveIntegralsDevice ai = activeViewOf(ints);

  // The plain coefficient tensors g (for the digests' sigma at h2e'[w]) and
  // dE/de (for the orbital-energy fold's tangent), and the walk's tangent
  // along w in two accumulators: sweep(Mdot; h) + sweep(M; h'[w]) and the
  // integral-free sweep(M; 0) that the second over-counts (:rdm_adjoint's
  // ScMixedDirection). n^2 + n^4 + 3 n^6 doubles each.
  ScRdmAdjoint adj = ScRdmAdjoint::zeros(norb, res);
  ScRdmAdjoint adjDot = ScRdmAdjoint::zeros(norb, res);
  ScRdmAdjoint adjFree = ScRdmAdjoint::zeros(norb, res);
  ScMixedDirection mix = mixedDirectionOf(s, p);
  mix.reverse = true;
  mix.adjDot = &adjDot;
  mix.adjFree = &adjFree;
  const ScGradientResult g =
      assembleClasses(ai, src, /*batch=*/0, s.rdm.dm1, s.rdm.dm2, s.rdm.dm3, s.rdm.f3ac,
                      s.rdm.f3ca, p.rot, &adj, /*epsAdjoint=*/s.epsCi, /*ci=*/nullptr, &mix,
                      res);
  ScMixedHessianVectorResult out;
  for (std::size_t c = 0; c < 8; ++c) {
    out.edotW[c] = g.classes[c].mixedEdotW;
    out.edotWTotal += out.edotW[c];
  }
  out.h2eDotMax = p.h2eDotMax;

  // Read back (each downloadTensor synchronizes the stream the sweeps were
  // issued on), the integral-free part taken off.
  const auto net = [&](const DeviceTensor& a, const DeviceTensor& f) {
    Tensor t = downloadTensor(a, st);
    const Tensor free = downloadTensor(f, st);
    for (int64_t i = 0; i < t.size(); ++i) t.flatRef(i) -= free.flat(i);
    return t;
  };
  Tensor g1 = net(adjDot.dm1, adjFree.dm1);
  const Tensor g2 = net(adjDot.dm2, adjFree.dm2);
  const Tensor g3 = net(adjDot.dm3, adjFree.dm3);
  const Tensor gac = net(adjDot.f3ac, adjFree.f3ac);
  const Tensor gca = net(adjDot.f3ca, adjFree.f3ca);
  // The orbital-energy fold, differentiated along w: y's dm1 coefficient
  // carries W^T dE/de, so its tangent carries W^T (dE/de)'[w] -- Rdot
  // scattered -- and, with the orbital energies responding to kappa too,
  // W'[w]^T dE/de (C5's generator, transposed: the same tensor F reads
  // forward).
  if (s.epsCi) {
    const int64_t nc = ints.e_core.dims[0], nv = ints.e_virt.dims[0];
    std::vector<double> dEdCore(static_cast<std::size_t>(nc), 0.0);
    std::vector<double> dEdVirt(static_cast<std::size_t>(nv), 0.0);
    std::vector<double> dEdCoreDot(static_cast<std::size_t>(nc), 0.0);
    std::vector<double> dEdVirtDot(static_cast<std::size_t>(nv), 0.0);
    for (const ScClassGradient& c : g.classes) {
      check(c.dEdCore.size() == dEdCore.size() && c.dEdVirt.size() == dEdVirt.size() &&
                c.dEdCoreDot.size() == dEdCore.size() && c.dEdVirtDot.size() == dEdVirt.size(),
            "scMixedHessianVectorDevice: a class's dE/de does not match the partition");
      for (std::size_t i = 0; i < dEdCore.size(); ++i) {
        dEdCore[i] += c.dEdCore[i];
        dEdCoreDot[i] += c.dEdCoreDot[i];
      }
      for (std::size_t r = 0; r < dEdVirt.size(); ++r) {
        dEdVirt[r] += c.dEdVirt[r];
        dEdVirtDot[r] += c.dEdVirtDot[r];
      }
    }
    ciSourceAddInto(orbitalEnergyDm1Coefficient(*s.epsResponse, dEdCoreDot, dEdVirtDot), g1);
    if (p.wDotSet) ciSourceAddInto(orbitalEnergyDm1Coefficient(p.wDot, dEdCore, dEdVirt), g1);
  }
  const Tensor h2e = downloadTensor(ints.h2e, st);
  out.fromWalkTangent =
      ciSourceVector(sigma, s.ci, norb, s.nelecA, s.nelecB, g1, g2, g3, gac, gca, h2e, res);
  // The digests' own orbital response: y's 4-body sigmas read h2e, linearly,
  // so along w they are the same sigmas at h2e'[w] with the plain (gac, gca).
  const Tensor zero1({norb, norb});
  const Tensor zero2({norb, norb, norb, norb});
  const Tensor zero3({norb, norb, norb, norb, norb, norb});
  out.fromDigests = ciSourceVector(sigma, s.ci, norb, s.nelecA, s.nelecB, zero1, zero2, zero3,
                                   downloadTensor(adj.f3ac, st), downloadTensor(adj.f3ca, st),
                                   p.h2eDot, res);
  out.hkcTw = out.fromWalkTangent;
  ciSourceAddInto(out.fromDigests, out.hkcTw);
  return out;
}

// --- the orbital block's integral curvature, <Ehat', I''> (#154) ---------------

const char* scCurvatureRouteName(const ScCurvatureRoute r) {
  switch (r) {
    case ScCurvatureRoute::Auto: return "auto";
    case ScCurvatureRoute::Sidecar: return "sidecar";
    case ScCurvatureRoute::ThreeIndex: return "three-index";
  }
  return "?";
}

// One walk over DOTTED INPUT SETS (:rotation's ScDotted): per set, the
// pairing in both modes, in ScFullRotationPairing's fields -- the integral
// halves, eps, and the digests' reverse term <hbar, h2e increment> when
// `hbar` is non-null. The sets are borrowed for the walk.
ScFullRotationResult pairDottedSets(const NevptIntegralsDevice& ints, const RdmTensors& rdm,
                                    const DeviceTensor* hbar,
                                    const std::vector<const ScDotted*>& sets,
                                    const std::vector<bool>& tangent, const DeviceResources& res) {
  FullBlockSource src(ints);
  const ActiveIntegralsDevice ai = activeViewOf(ints);
  const ScRotation rot = makeDottedRotation(sets, tangent, res);
  ScFullRotationResult out;
  out.grad = assembleClasses(ai, src, /*batch=*/0, rdm.dm1, rdm.dm2, rdm.dm3, rdm.f3ac, rdm.f3ca,
                             rot, /*adj=*/nullptr, /*epsAdjoint=*/false, /*ci=*/nullptr,
                             /*mix=*/nullptr, res);
  out.dirs.assign(sets.size(), ScFullRotationPairing{});
  for (std::size_t d = 0; d < sets.size(); ++d) {
    ScFullRotationPairing& p = out.dirs[d];
    for (const ScClassGradient& c : out.grad.classes) {
      p.ext += c.rotExt[d];
      p.act += c.rotAct[d];
      p.eps += c.rotEps[d];
      p.tangentIntegral += c.rotTangent[d];
    }
    p.integral = p.ext + p.act;
    if (hbar != nullptr) {
      p.digest = dotTensors(*hbar, sets[d]->d.h2e, res);
      p.hasDigest = true;
    }
    p.adjoint = p.integral + p.eps + p.digest;
    p.tangent = p.tangentIntegral + p.eps;
    p.hasTangent = tangent[d];
  }
  return out;
}

// max |x|, on the host (downloadTensor synchronizes the stream).
double maxAbsDevice(const DeviceTensor& x, const DeviceResources& res) {
  if (x.size() == 0) return 0.0;
  const Tensor h = downloadTensor(x, res.stream());
  double m = 0.0;
  for (int64_t i = 0; i < h.size(); ++i) m = std::max(m, std::fabs(h.flat(i)));
  return m;
}

// A direction of the orbital curvature: an (n_mo, n_mo) antisymmetric kappa,
// or the caller's Error naming which one and why.
Status checkCurvatureDirection(const Tensor& k, int64_t nmo, std::size_t pair, const char* name) {
  if (k.rank() != 2 || k.dim(0) != nmo || k.dim(1) != nmo)
    return err_config(std::format("pair {}: {} is not a {} x {} matrix", pair, name, nmo, nmo));
  double anti = 0.0;
  for (int64_t p = 0; p < nmo; ++p)
    for (int64_t q = 0; q < nmo; ++q) anti = std::max(anti, std::fabs(k(p, q) + k(q, p)));
  if (anti > 1e-14)
    return err_config(std::format(
        "pair {}: {} is not antisymmetric (max |k + k^T| = {:.3e}) -- an orbital direction is "
        "a kappa; the symmetric W = (uv + vu)/2 is formed here",
        pair, name, anti));
  return {};
}

// Whether every nonzero entry of `k` sits in its active-active block.
bool isActiveActive(const Tensor& k, int64_t actLo, int64_t actHi) {
  for (int64_t p = 0; p < k.dim(0); ++p)
    for (int64_t q = 0; q < k.dim(1); ++q) {
      const bool inside = p >= actLo && p < actHi && q >= actLo && q < actHi;
      if (!inside && k(p, q) != 0.0) return false;
    }
  return true;
}

// PC's one walk over dotted input sets (#200; docs/hessian-theory.md, Sec.
// 6.7): pairDottedSets' pairing with PC's integral adjoint Ehat'_pc --
// :pc_classes' dE/dslab against every slab's dot, dE/dK against the class's
// own K builder at the set's h1e / h2e (pcClassMatrices), R_t against the
// set's orbital energies -- and, with `hbar` (pcDigestAdjointDevice), the
// digests' <hbar, h2e increment>. PC reads no v2tr (its Sir pairs Sir2's own
// dotted block), so a set's v2tr field is never read here. A class the
// solve refuses leaves `ok` false and `dirs` empty.
Result<PcFullRotationResult> pairDottedSetsPc(const NevptIntegralsDevice& ints,
                                              const RdmTensors& rdm, const DeviceTensor* hbar,
                                              const std::vector<const ScDotted*>& sets,
                                              const std::vector<bool>& tangent,
                                              const DeviceResources& res) {
  FullBlockSource src(ints);
  const ActiveIntegralsDevice ai = activeViewOf(ints);
  const ScRotation rot = makeDottedRotation(sets, tangent, res);
  PcFullRotationResult out;
  out.walk = NEVPT2_TRY(assemblePcClasses(ai, src, /*batch=*/0, rdm.dm1, rdm.dm2, rdm.dm3,
                                          rdm.f3ac, rdm.f3ca, rot, /*adj=*/nullptr,
                                          /*epsAdjoint=*/false, res));
  out.ok = out.walk.allDone();
  if (!out.ok) return out;
  out.dirs.assign(sets.size(), ScFullRotationPairing{});
  for (std::size_t d = 0; d < sets.size(); ++d) {
    ScFullRotationPairing& p = out.dirs[d];
    for (const PcClassGradient& c : out.walk.classes) {
      p.ext += c.rotExt[d];
      p.act += c.rotAct[d];
      p.eps += c.rotEps[d];
      p.tangentIntegral += c.rotTangent[d];
    }
    p.integral = p.ext + p.act;
    if (hbar != nullptr) {
      p.digest = dotTensors(*hbar, sets[d]->d.h2e, res);
      p.hasDigest = true;
    }
    p.adjoint = p.integral + p.eps + p.digest;
    p.tangent = p.tangentIntegral + p.eps;
    p.hasTangent = tangent[d];
  }
  return out;
}

// The PC energy's response to one h2e increment through the digests alone:
// the RDM build at `h2eDotDevice` (f3ac / f3ca are linear in h2e at a fixed
// CI vector) and pcTangentsDevice with every other RDM tangent zero --
// digestTangentAt's construction for PC. A class the solve refuses is a
// Numerical Error naming it.
Result<double> pcDigestTangentAt(const RdmBuildOptions& opt, const Tensor& ci,
                                 const int64_t nelecA, const int64_t nelecB,
                                 const NevptIntegralsDevice& ints,
                                 const DeviceTensor& h2eDotDevice, const RdmTensors& rdm,
                                 const DeviceTensor& dm1Zero, const DeviceTensor& dm2Zero,
                                 const DeviceTensor& dm3Zero, const DeviceResources& res) {
  const int64_t norb = rdm.dm1.dims[0];
  // downloadTensor synchronizes: the einsums that wrote the increment are done.
  const Tensor h2eDot = downloadTensor(h2eDotDevice, res.stream());
  const RdmBuildResult dot = buildRdmsDevice(opt, ci, norb, nelecA, nelecB, h2eDot, res);
  const PcTangentResult t = NEVPT2_TRY(pcTangentsDevice(
      ints, rdm.dm1, rdm.dm2, rdm.dm3, rdm.f3ac, rdm.f3ca, dm1Zero, dm2Zero, dm3Zero, dot.f3ac,
      dot.f3ca, /*epsDot=*/nullptr, res));
  double total = 0.0;
  for (std::size_t c = 0; c < 8; ++c) {
    if (t.classes[c].result.status != PcStatus::Done)
      return err_numerical(std::format(
          "the PC digests' tangent at a second-order h2e increment: {} refused by the solve ({})",
          CLASSES[c], t.classes[c].result.why));
    total += t.classes[c].edot;
  }
  return total;
}

// One walk over dotted sets in either variant, as the pairings' list: SC's
// pairDottedSets, or PC's, whose refused class is a Numerical Error naming it
// (a pairing with a class missing is not the term).
Result<std::vector<ScFullRotationPairing>> pairDottedSetsAny(
    const bool pc, const NevptIntegralsDevice& ints, const RdmTensors& rdm,
    const DeviceTensor* hbar, const std::vector<const ScDotted*>& sets,
    const std::vector<bool>& tangent, const DeviceResources& res) {
  if (!pc) return pairDottedSets(ints, rdm, hbar, sets, tangent, res).dirs;
  PcFullRotationResult w = NEVPT2_TRY(pairDottedSetsPc(ints, rdm, hbar, sets, tangent, res));
  if (!w.ok)
    return err_numerical(std::format(
        "the PC orbital curvature has no value: {} refused by the solve the energy runs "
        "(docs/pc-nevpt2.md, \"The singular-metric convention\")",
        w.walk.refusals()));
  return std::move(w.dirs);
}

// One pair's settled plan: which route serves each term, and which member is
// active-active.
struct CurvaturePlan {
  ScCurvatureRoute k1, k2;
  bool uAct, vAct;
};

// Shape, antisymmetry and the routes of pair `q`, or the caller's Error --
// Unsupported for a route the data cannot serve or a conventional pair with
// neither member active-active (the header's refusals), before any device work.
Result<CurvaturePlan> planCurvaturePair(const ScOrbitalCurvatureSetup& s,
                                        const ScOrbitalCurvatureRequest& r, const std::size_t q) {
  const int64_t nmo = s.gen.nmo();
  const int64_t actLo = s.gen.ncore, actHi = s.gen.ncore + s.gen.nact;
  const bool fitted = s.gen.densityFitted();
  NEVPT2_TRY(checkCurvatureDirection(r.u, nmo, q, "u"));
  NEVPT2_TRY(checkCurvatureDirection(r.v, nmo, q, "v"));
  CurvaturePlan p{ScCurvatureRoute::Sidecar, ScCurvatureRoute::Sidecar,
                  isActiveActive(r.u, actLo, actHi), isActiveActive(r.v, actLo, actHi)};
  if (r.k1 == ScCurvatureRoute::ThreeIndex) {
    if (!fitted)
      return err_unsupported(std::format(
          "pair {}: K1's three-index route reads the whole B[L, p, q], which only a "
          "density-fitted case carries",
          q));
    p.k1 = ScCurvatureRoute::ThreeIndex;
  }
  const bool oneActive = p.uAct || p.vAct;
  const ScCurvatureRoute want =
      r.k2 != ScCurvatureRoute::Auto
          ? r.k2
          : (fitted ? ScCurvatureRoute::ThreeIndex : ScCurvatureRoute::Sidecar);
  if (want == ScCurvatureRoute::ThreeIndex && !fitted)
    return err_unsupported(std::format(
        "pair {}: K2's three-index route reads the whole B[L, p, q], which only a "
        "density-fitted case carries",
        q));
  if (want == ScCurvatureRoute::Sidecar && !oneActive)
    return err_unsupported(std::format(
        "pair {}: the orbital Hessian's cross-slot curvature K2 reads two-general-index "
        "integrals -- for a pair with neither direction active-active, the whole MO-integral "
        "tensor, which the conventional gradient sidecar does not carry "
        "(docs/hessian-theory.md, Sec. 5.4). Refused rather than answered without K2 "
        "(Sec. 9.1); the full-tensor K2 is #180",
        q));
  p.k2 = want;
  return p;
}

// W = (u v + v u) / 2: symmetric, the parameterisation's own second order.
Tensor symmetricProduct(const Tensor& u, const Tensor& v) {
  const Tensor uv = matmulHost(u, v);
  const Tensor vu = matmulHost(v, u);
  Tensor w(uv.dims());
  for (int64_t i = 0; i < w.size(); ++i) w.flatRef(i) = 0.5 * (uv.flat(i) + vu.flat(i));
  return w;
}

// K1's and K2's dotted input sets for one planned pair, by the plan's routes.
Result<std::pair<ScDotted, ScDotted>> curvatureSets(const ScOrbitalCurvatureSetup& s,
                                                    const ScOrbitalCurvatureRequest& r,
                                                    const CurvaturePlan& p, const Tensor& w,
                                                    const DeviceResources& res) {
  ScDotted k1Set = p.k1 == ScCurvatureRoute::ThreeIndex
                       ? dfDottedFirst(s.ints, s.gen, s.rdm.dm1, w, res)
                       : NEVPT2_TRY(dottedAtMatrix(s.ints, s.gen, s.rdm.dm1, w, res));
  // The composition's outer direction is the active-active one; z^x is
  // symmetric in (u, v), so either way round is the same term.
  ScDotted k2Set = p.k2 == ScCurvatureRoute::ThreeIndex
                       ? dfDottedCross(s.ints, s.gen, s.rdm.dm1, r.u, r.v, res)
                       : NEVPT2_TRY(composedCross(s.ints, s.gen, s.rdm.dm1, p.uAct ? r.u : r.v,
                                                  p.uAct ? r.v : r.u, res));
  return std::pair<ScDotted, ScDotted>{std::move(k1Set), std::move(k2Set)};
}

Result<NevptIntegralsDevice> orbitalSecondInputsDevice(const ScOrbitalCurvatureSetup& s,
                                                       const ScOrbitalCurvatureRequest& r,
                                                       const DeviceResources& res) {
  NEVPT2_TRY(checkGeneralShapes(s.ints, s.gen));
  const CurvaturePlan p = NEVPT2_TRY(planCurvaturePair(s, r, 0));
  const auto [k1, k2] = NEVPT2_TRY(curvatureSets(s, r, p, symmetricProduct(r.u, r.v), res));
  NevptIntegralsDevice out;
  const auto sum = [&](DeviceTensor NevptIntegralsDevice::* member) {
    out.*member = DeviceTensor::zeros((k1.d.*member).dims, res);
    addScaledInto(k1.d.*member, out.*member, 1.0, res.stream());
    addScaledInto(k2.d.*member, out.*member, 1.0, res.stream());
  };
  for (const ScDottedField& f : kDottedFields) sum(f.member);
  sum(&NevptIntegralsDevice::e_core);
  sum(&NevptIntegralsDevice::e_virt);
  return out;
}

Result<NevptIntegralsDevice> orbitalMatrixInputsDevice(const ScOrbitalCurvatureSetup& s,
                                                       const Tensor& m,
                                                       const DeviceResources& res) {
  NEVPT2_TRY(checkGeneralShapes(s.ints, s.gen));
  const int64_t nmo = s.gen.nmo();
  if (m.rank() != 2 || m.dim(0) != nmo || m.dim(1) != nmo)
    return err_config(std::format("orbitalMatrixInputsDevice: M is not a {} x {} matrix", nmo, nmo));
  ScDotted set = NEVPT2_TRY(dottedAtMatrix(s.ints, s.gen, s.rdm.dm1, m, res));
  return std::move(set.d);
}

// scOrbitalCurvatureDevice and pcOrbitalCurvatureDevice: one body, the
// variant choosing the walk that pairs the dotted sets and the digests'
// forward term. The input generators (:curvature) and every refusal are the
// two variants' alike (docs/hessian-theory.md, Sec. 6.7).
Result<std::vector<ScOrbitalCurvatureResult>> orbitalCurvature(
    const bool pc, const ScOrbitalCurvatureSetup& s,
    const std::vector<ScOrbitalCurvatureRequest>& pairs, const DeviceResources& res) {
  const char* const who = pc ? "pcOrbitalCurvatureDevice" : "scOrbitalCurvatureDevice";
  NEVPT2_TRY(checkGeneralShapes(s.ints, s.gen));
  if (s.digestAdjoint != nullptr && s.digestAdjoint->dims != s.ints.h2e.dims)
    return err_config(std::format("{}: the digests' adjoint is not shaped like h2e", who));

  // Every pair is settled -- shape, antisymmetry, which route serves each
  // term -- before any device work, so a refusal is never a partial answer.
  std::vector<CurvaturePlan> plans;
  bool anyTangent = false;
  for (std::size_t q = 0; q < pairs.size(); ++q) {
    plans.push_back(NEVPT2_TRY(planCurvaturePair(s, pairs[q], q)));
    anyTangent = anyTangent || pairs[q].tangent;
  }
  if (anyTangent && s.ci != nullptr)
    check(!s.opt.cublas && !s.opt.ozaki && !s.opt.fusedDigest,
          "the orbital curvature's digests' forward term runs the emitted or the native-fp64 "
          "BLAS digest (cli refuses --cublas / --ozaki / --fused-digest with --sc-hessian-kk-int "
          "and --sc-hessian-kk)");

  // The zero RDM tangents the digests' forward term reads, allocated once.
  std::optional<DeviceTensor> dm1Zero, dm2Zero, dm3Zero;
  if (anyTangent && s.ci != nullptr) {
    dm1Zero.emplace(DeviceTensor::zeros(s.rdm.dm1.dims, res));
    dm2Zero.emplace(DeviceTensor::zeros(s.rdm.dm2.dims, res));
    dm3Zero.emplace(DeviceTensor::zeros(s.rdm.dm3.dims, res));
  }

  std::vector<ScOrbitalCurvatureResult> out;
  out.reserve(pairs.size());
  for (std::size_t q = 0; q < pairs.size(); ++q) {
    const ScOrbitalCurvatureRequest& r = pairs[q];
    const CurvaturePlan& p = plans[q];
    const Tensor w = symmetricProduct(r.u, r.v);
    const Tensor vu = matmulHost(r.v, r.u);
    const auto [k1Set, k2Set] = NEVPT2_TRY(curvatureSets(s, r, p, w, res));
    std::vector<const ScDotted*> sets{&k1Set, &k2Set};
    std::vector<bool> tangent{r.tangent, r.tangent};
    // Both active-active: the traced pairs' shares, K1's at W and K2's at v u
    // (with K2's minus sign, below).
    std::optional<ScDotted> tracedW, tracedVu;
    if (p.uAct && p.vAct) {
      tracedW.emplace(restoredTraces(s.ints, s.gen, w, res));
      tracedVu.emplace(restoredTraces(s.ints, s.gen, vu, res));
      sets.push_back(&*tracedW);
      sets.push_back(&*tracedVu);
      tangent.push_back(false);
      tangent.push_back(false);
    }
    const std::vector<ScFullRotationPairing> walk =
        NEVPT2_TRY(pairDottedSetsAny(pc, s.ints, s.rdm, s.digestAdjoint, sets, tangent, res));

    ScOrbitalCurvatureResult o;
    o.k1Route = p.k1;
    o.k2Route = p.k2;
    for (int t = 0; t < 2; ++t) {
      const ScFullRotationPairing& w2 = walk[static_cast<std::size_t>(t)];
      const ScDotted& set = *sets[static_cast<std::size_t>(t)];
      ScCurvatureTerm& term = t == 0 ? o.k1 : o.k2;
      term.ext = w2.ext;
      term.act = w2.act;
      term.integral = w2.integral;
      term.eps = w2.eps;
      term.digest = w2.digest;
      term.hasDigest = w2.hasDigest;
      term.adjoint = w2.adjoint;
      term.tangentIntegral = w2.tangentIntegral;
      term.hasTangent = w2.hasTangent;
      term.h1eMax = maxAbsDevice(set.d.h1e, res);
      term.h2eMax = maxAbsDevice(set.d.h2e, res);
      if (r.tangent && s.ci != nullptr)
        term.digestForward =
            pc ? NEVPT2_TRY(pcDigestTangentAt(s.opt, *s.ci, s.nelecA, s.nelecB, s.ints, set.d.h2e,
                                              s.rdm, *dm1Zero, *dm2Zero, *dm3Zero, res))
               : digestTangentAt(s.opt, *s.ci, s.nelecA, s.nelecB, s.ints, set.d.h2e, s.rdm.dm1,
                                 s.rdm.dm2, s.rdm.dm3, s.rdm.f3ac, s.rdm.f3ca, *dm1Zero,
                                 *dm2Zero, *dm3Zero, std::nullopt, res)
                     .total;
      term.tangent = term.tangentIntegral + term.eps + term.digestForward;
    }
    if (p.uAct && p.vAct) {
      o.hasTraced = true;
      o.k1Traced = walk[2].adjoint;
      o.k2Traced = -walk[3].adjoint;
    }
    out.push_back(o);
  }
  return out;
}

Result<std::vector<ScOrbitalCurvatureResult>> scOrbitalCurvatureDevice(
    const ScOrbitalCurvatureSetup& s, const std::vector<ScOrbitalCurvatureRequest>& pairs,
    const DeviceResources& res) {
  return orbitalCurvature(/*pc=*/false, s, pairs, res);
}

Result<std::vector<ScOrbitalCurvatureResult>> pcOrbitalCurvatureDevice(
    const ScOrbitalCurvatureSetup& s, const std::vector<ScOrbitalCurvatureRequest>& pairs,
    const DeviceResources& res) {
  return orbitalCurvature(/*pc=*/true, s, pairs, res);
}

// The matrix path's dotted sets, one per direction (K1's machinery at each
// direction's matrix), and their tangent flags: scMatrixRotationDevice's and
// pcMatrixRotationDevice's shared front half.
struct MatrixSets {
  std::vector<ScDotted> owned;
  std::vector<bool> tangent;
  std::vector<const ScDotted*> ptrs() const {
    std::vector<const ScDotted*> out;
    for (const ScDotted& d : owned) out.push_back(&d);
    return out;
  }
};

Result<MatrixSets> matrixSets(const char* who, const NevptIntegralsDevice& ints,
                              const GeneralIntegralsDevice& gen, const RdmTensors& rdm,
                              const DeviceTensor* digestAdjoint,
                              const std::vector<ScRotationRequest>& dirs,
                              const DeviceResources& res) {
  NEVPT2_TRY(checkGeneralShapes(ints, gen));
  if (digestAdjoint != nullptr && digestAdjoint->dims != ints.h2e.dims)
    return err_config(std::format("{}: the digests' adjoint is not shaped like h2e", who));
  MatrixSets out;
  out.owned.reserve(dirs.size());
  for (const ScRotationRequest& d : dirs) {
    out.owned.push_back(NEVPT2_TRY(dottedAtMatrix(ints, gen, rdm.dm1, d.kappa, res)));
    out.tangent.push_back(d.tangent);
  }
  return out;
}

Result<ScFullRotationResult> scMatrixRotationDevice(const NevptIntegralsDevice& ints,
                                                    const GeneralIntegralsDevice& gen,
                                                    const RdmTensors& rdm,
                                                    const DeviceTensor* digestAdjoint,
                                                    const std::vector<ScRotationRequest>& dirs,
                                                    const DeviceResources& res) {
  const MatrixSets m = NEVPT2_TRY(
      matrixSets("scMatrixRotationDevice", ints, gen, rdm, digestAdjoint, dirs, res));
  return pairDottedSets(ints, rdm, digestAdjoint, m.ptrs(), m.tangent, res);
}

Result<PcFullRotationResult> pcMatrixRotationDevice(const NevptIntegralsDevice& ints,
                                                    const GeneralIntegralsDevice& gen,
                                                    const RdmTensors& rdm,
                                                    const DeviceTensor* digestAdjoint,
                                                    const std::vector<ScRotationRequest>& dirs,
                                                    const DeviceResources& res) {
  const MatrixSets m = NEVPT2_TRY(
      matrixSets("pcMatrixRotationDevice", ints, gen, rdm, digestAdjoint, dirs, res));
  return pairDottedSetsPc(ints, rdm, digestAdjoint, m.ptrs(), m.tangent, res);
}

// Caller-supplied increments as dotted sets, each field checked against the
// integrals' own shape: scDottedPairingDevice's and pcDottedPairingDevice's
// shared front half. The sets VIEW the increments, which outlive the walk.
Result<std::vector<ScDotted>> incrementSets(const char* who, const NevptIntegralsDevice& ints,
                                            const DeviceTensor* digestAdjoint,
                                            const std::vector<const NevptIntegralsDevice*>& dotted,
                                            const DeviceResources& res) {
  if (digestAdjoint != nullptr && digestAdjoint->dims != ints.h2e.dims)
    return err_config(std::format("{}: the digests' adjoint is not shaped like h2e", who));
  const auto fields = [](const NevptIntegralsDevice& n) {
    return std::array<std::pair<const char*, const DeviceTensor*>, 16>{{
        {"h1e", &n.h1e},           {"h2e", &n.h2e},
        {"e_core", &n.e_core},     {"e_virt", &n.e_virt},
        {"h2e_v_Sr", &n.h2e_v_Sr}, {"h1e_v_Sr", &n.h1e_v_Sr},
        {"h2e_v_Si", &n.h2e_v_Si}, {"h1e_v_Si", &n.h1e_v_Si},
        {"cvcv", &n.cvcv},         {"h2e_v_Sijr", &n.h2e_v_Sijr},
        {"h2e_v_Srsi", &n.h2e_v_Srsi}, {"h2e_v_Srs", &n.h2e_v_Srs},
        {"h2e_v_Sij", &n.h2e_v_Sij},   {"h2e_v1_Sir", &n.h2e_v1_Sir},
        {"h2e_v2_Sir", &n.h2e_v2_Sir}, {"h1e_v_Sir", &n.h1e_v_Sir},
    }};
  };
  const auto want = fields(ints);
  std::vector<ScDotted> owned;
  owned.reserve(dotted.size());
  for (std::size_t k = 0; k < dotted.size(); ++k) {
    const auto got = fields(*dotted[k]);
    for (std::size_t f = 0; f < want.size(); ++f)
      if (got[f].second->dims != want[f].second->dims)
        return err_config(std::format(
            "{}: increment {}'s {} is not shaped like the integrals' own", who, k, want[f].first));
    ScDotted d;
    d.d.h1e = dotted[k]->h1e.view();
    d.d.h2e = dotted[k]->h2e.view();
    d.d.e_core = dotted[k]->e_core.view();
    d.d.e_virt = dotted[k]->e_virt.view();
    d.d.h2e_v_Sr = dotted[k]->h2e_v_Sr.view();
    d.d.h1e_v_Sr = dotted[k]->h1e_v_Sr.view();
    d.d.h2e_v_Si = dotted[k]->h2e_v_Si.view();
    d.d.h1e_v_Si = dotted[k]->h1e_v_Si.view();
    d.d.cvcv = dotted[k]->cvcv.view();
    d.d.h2e_v_Sijr = dotted[k]->h2e_v_Sijr.view();
    d.d.h2e_v_Srsi = dotted[k]->h2e_v_Srsi.view();
    d.d.h2e_v_Srs = dotted[k]->h2e_v_Srs.view();
    d.d.h2e_v_Sij = dotted[k]->h2e_v_Sij.view();
    d.d.h2e_v1_Sir = dotted[k]->h2e_v1_Sir.view();
    d.d.h2e_v2_Sir = dotted[k]->h2e_v2_Sir.view();
    d.d.h1e_v_Sir = dotted[k]->h1e_v_Sir.view();
    d.v2tr = traceV2tr(d.d.h2e_v2_Sir, res);
    owned.push_back(std::move(d));
  }
  return owned;
}

Result<ScFullRotationResult> scDottedPairingDevice(
    const NevptIntegralsDevice& ints, const RdmTensors& rdm, const DeviceTensor* digestAdjoint,
    const std::vector<const NevptIntegralsDevice*>& dotted, const bool tangent,
    const DeviceResources& res) {
  const std::vector<ScDotted> owned =
      NEVPT2_TRY(incrementSets("scDottedPairingDevice", ints, digestAdjoint, dotted, res));
  std::vector<const ScDotted*> sets;
  for (const ScDotted& d : owned) sets.push_back(&d);
  return pairDottedSets(ints, rdm, digestAdjoint, sets, std::vector<bool>(sets.size(), tangent),
                        res);
}

Result<PcFullRotationResult> pcDottedPairingDevice(
    const NevptIntegralsDevice& ints, const RdmTensors& rdm, const DeviceTensor* digestAdjoint,
    const std::vector<const NevptIntegralsDevice*>& dotted, const bool tangent,
    const DeviceResources& res) {
  const std::vector<ScDotted> owned =
      NEVPT2_TRY(incrementSets("pcDottedPairingDevice", ints, digestAdjoint, dotted, res));
  std::vector<const ScDotted*> sets;
  for (const ScDotted& d : owned) sets.push_back(&d);
  return pairDottedSetsPc(ints, rdm, digestAdjoint, sets,
                          std::vector<bool>(sets.size(), tangent), res);
}

Result<double> scRestoredTracePairingDevice(const NevptIntegralsDevice& ints,
                                            const GeneralIntegralsDevice& gen,
                                            const RdmTensors& rdm, const Tensor& m,
                                            const DeviceResources& res) {
  NEVPT2_TRY(checkGeneralShapes(ints, gen));
  const int64_t nmo = gen.nmo();
  if (m.rank() != 2 || m.dim(0) != nmo || m.dim(1) != nmo)
    return err_config(std::format("scRestoredTracePairingDevice: M is not a {} x {} matrix", nmo, nmo));
  const ScDotted traced = restoredTraces(ints, gen, m, res);
  const ScFullRotationResult r =
      pairDottedSets(ints, rdm, /*hbar=*/nullptr, {&traced}, {false}, res);
  return r.dirs[0].adjoint;
}

Result<double> pcRestoredTracePairingDevice(const NevptIntegralsDevice& ints,
                                            const GeneralIntegralsDevice& gen,
                                            const RdmTensors& rdm, const Tensor& m,
                                            const DeviceResources& res) {
  NEVPT2_TRY(checkGeneralShapes(ints, gen));
  const int64_t nmo = gen.nmo();
  if (m.rank() != 2 || m.dim(0) != nmo || m.dim(1) != nmo)
    return err_config(std::format("pcRestoredTracePairingDevice: M is not a {} x {} matrix", nmo, nmo));
  const ScDotted traced = restoredTraces(ints, gen, m, res);
  const std::vector<ScFullRotationPairing> r = NEVPT2_TRY(
      pairDottedSetsAny(/*pc=*/true, ints, rdm, /*hbar=*/nullptr, {&traced}, {false}, res));
  return r[0].adjoint;
}

// --- the orbital block's sandwich, I'^T Ehat'' I' (#158) -----------------------

Result<std::vector<ScOrbitalSandwichResult>> scOrbitalSandwichDevice(
    const ScOrbitalSandwichSetup& s, const std::vector<ScOrbitalSandwichRequest>& pairs,
    const DeviceResources& res) {
  // A tangent built from an approximate digest is not the derivative of
  // anything the energy stage computed; nevpt2.cli refuses the combination.
  check(!s.opt.cublas && !s.opt.ozaki && !s.opt.fusedDigest,
        "scOrbitalSandwichDevice runs the emitted or the native-fp64 BLAS digest (cli refuses "
        "--cublas / --ozaki / --fused-digest with --sc-hessian-kk)");
  NEVPT2_TRY(checkGeneralShapes(s.ints, s.gen));
  const int64_t nmo = s.gen.nmo();
  // Every pair is checked before any device work.
  for (std::size_t q = 0; q < pairs.size(); ++q) {
    NEVPT2_TRY(checkCurvatureDirection(pairs[q].u, nmo, q, "u"));
    NEVPT2_TRY(checkCurvatureDirection(pairs[q].v, nmo, q, "v"));
  }
  if (s.ci.rank() != 2)
    return err_config("scOrbitalSandwichDevice: the state is not an (nstringsA, nstringsB) CI vector");

  FullBlockSource src(s.ints);
  const ActiveIntegralsDevice ai = activeViewOf(s.ints);
  const int64_t nc = s.gen.ncore, nv = s.gen.nvirt;
  // The integrals the digest-only blocks are built at (:sc_classes' vDigest /
  // wDigest), and the zero RDM sets the digests' reference walk reads.
  const DeviceTensor zeroH1e = DeviceTensor::zeros(s.ints.h1e.dims, res);
  const DeviceTensor zeroH2e = DeviceTensor::zeros(s.ints.h2e.dims, res);
  std::optional<DeviceTensor> z1, z2, z3, zac, zca;
  for (const ScOrbitalSandwichRequest& r : pairs)
    if (r.digestReference && !z1) {
      z1.emplace(DeviceTensor::zeros(s.rdm.dm1.dims, res));
      z2.emplace(DeviceTensor::zeros(s.rdm.dm2.dims, res));
      z3.emplace(DeviceTensor::zeros(s.rdm.dm3.dims, res));
      zac.emplace(DeviceTensor::zeros(s.rdm.f3ac.dims, res));
      zca.emplace(DeviceTensor::zeros(s.rdm.f3ca.dims, res));
    }

  std::vector<ScOrbitalSandwichResult> out;
  out.reserve(pairs.size());
  for (const ScOrbitalSandwichRequest& r : pairs) {
    // u and v as the walk's two tangent directions, each with its digests'
    // tangent attached (the RDM build at its own h2e'), so every tuple's Hdot
    // carries the digests' share before the 3x3 meets it.
    ScRotation rot = NEVPT2_TRY(makeGeneralRotation(
        {ScRotationRequest{r.u, /*tangent=*/true}, ScRotationRequest{r.v, /*tangent=*/true}},
        ai.h1e, ai.h2e, s.gen, s.rdm.dm1, res));
    const std::vector<RdmBuildResult> digests =
        attachDigestTangents(rot, s.opt, s.ci, s.nelecA, s.nelecB, res);

    ScMixedDirection mix;
    mix.epsKappa = true;
    mix.pairForward = true;
    mix.pairReverse = r.reverse;
    // |u|^2, |v|^2: the guarded tuples' population floor is relative to them.
    const auto norm2 = [](const Tensor& k) {
      double n = 0.0;
      for (int64_t i = 0; i < k.size(); ++i) n += k.flat(i) * k.flat(i);
      return n;
    };
    mix.pairScaleU = norm2(r.u);
    mix.pairScaleV = norm2(r.v);
    if (!(mix.pairScaleU > 0.0) || !(mix.pairScaleV > 0.0))
      return err_config(std::format("scOrbitalSandwichDevice: pair {} has a zero direction",
                                    out.size()));
    mix.zeroH1e = &zeroH1e;
    mix.zeroH2e = &zeroH2e;
    // The CI side's orbital-energy tangents, which the walk state points at
    // and the orbital pair never reads.
    mix.epsCoreU = Tensor({nc});
    mix.epsVirtU = Tensor({nv});
    mix.epsCoreWU = Tensor({nc});
    mix.epsVirtWU = Tensor({nv});
    const ScGradientResult g =
        assembleClasses(ai, src, /*batch=*/0, s.rdm.dm1, s.rdm.dm2, s.rdm.dm3, s.rdm.f3ac,
                        s.rdm.f3ca, rot, /*adj=*/nullptr, /*epsAdjoint=*/false, /*ci=*/nullptr,
                        &mix, res);

    // The guarded tuples (docs/hessian-theory.md, Sec. 5.6 and Sec. 9.1;
    // #190): K6 is in the walk's totals wherever the energy is a quadratic
    // form along the pair. Where an H class's empty tuples both directions
    // populate carry non-proportional tables (rho_t above kRhoTol) the energy
    // is not twice differentiable along the pair -- there is no Hessian
    // element to return -- and a guarded tuple empty by the state is not
    // derived; either refuses the pair, never answered with a zero.
    std::string noHessian, byState;
    for (std::size_t c = 0; c < 8; ++c) {
      const ScGuardedCensus& gc = g.classes[c].guarded;
      if (gc.rhoAbove > 0)
        noHessian += std::format(
            "{}{}: {} empty tuples both directions populate, {} of them with rho_t above {:.0e}, "
            "the worst {:.3e}",
            noHessian.empty() ? "" : "; ", CLASSES[c], gc.populated, gc.rhoAbove, kRhoTol,
            gc.rhoMax);
      if (gc.emptyByState > 0)
        byState += std::format("{}{}: {} guarded tuples empty by the state",
                               byState.empty() ? "" : "; ", CLASSES[c], gc.emptyByState);
    }
    if (!noHessian.empty())
      return err_numerical(std::format(
          "pair {}: the SC energy is not twice differentiable along this pair -- {} -- so its "
          "empty perturbers' energy is not a quadratic form on the plane and the orbital "
          "Hessian has no element here (docs/hessian-theory.md, Sec. 5.6); refused",
          out.size(), noHessian));
    if (!byState.empty())
      return err_numerical(std::format(
          "pair {}: {} -- a guarded perturber empty by the state, whose second-order term "
          "docs/hessian-theory.md, Sec. 5.6 does not derive; refused",
          out.size(), byState));

    ScOrbitalSandwichResult o;
    for (std::size_t c = 0; c < 8; ++c) {
      o.guarded[c] = g.classes[c].guarded;
      o.classes[c] = g.classes[c].pair;
      o.total += g.classes[c].pair;
      o.edotU[c] = g.classes[c].mixedEdotW;
      o.edotV[c] = g.classes[c].pairEdotV;
      o.edotUTotal += o.edotU[c];
      o.edotVTotal += o.edotV[c];
      o.reverseClasses[c] = g.classes[c].pairReverse;
      if (r.reverse) o.reverse += o.reverseClasses[c];
      o.reverseK6Classes[c] = g.classes[c].pairReverseK6;
      if (r.reverse) o.reverseK6 += o.reverseK6Classes[c];
    }
    o.forward = o.total.total();
    o.hasReverse = r.reverse;
    o.h2eDotMaxU = maxAbsDevice(rot.dirs[0].h2eDot, res);
    o.h2eDotMaxV = maxAbsDevice(rot.dirs[1].h2eDot, res);
    o.f3DotMaxU = std::max(maxAbsDevice(digests[0].f3ac, res), maxAbsDevice(digests[0].f3ca, res));
    o.f3DotMaxV = std::max(maxAbsDevice(digests[1].f3ac, res), maxAbsDevice(digests[1].f3ca, res));
    if (r.digestReference) {
      // The energy stage's own second derivative along the digests' tangents
      // alone -- f3 enters only the H side, linearly, so this is the 3x3
      // between the two digest-only triples, off the plain digest path's
      // f3'[u], f3'[v] and the energy's own Hessian walk.
      const RdmTensors zeroSet{*z1, *z2, *z3, *zac, *zca};
      const RdmTensors dotU{*z1, *z2, *z3, digests[0].f3ac, digests[0].f3ca};
      const RdmTensors dotV{*z1, *z2, *z3, digests[1].f3ac, digests[1].f3ca};
      const EnergyHessianResult e =
          energiesHessianDevice(ai, src, /*batch=*/0, s.rdm, dotU, dotV, zeroSet, res);
      o.hasDigestReference = true;
      o.digestReference = e.totalDDot;
    }
    out.push_back(o);
  }
  return out;
}

Result<std::vector<ScOrbitalHessianResult>> scOrbitalHessianDevice(
    const ScOrbitalCurvatureSetup& s, const std::vector<ScOrbitalCurvatureRequest>& pairs,
    const DeviceResources& res) {
  if (s.ci == nullptr)
    return err_config(
        "scOrbitalHessianDevice: the sandwich's digests' tangents need the state (setup.ci)");
  // The curvature settles every pair -- and refuses -- before its device work.
  const std::vector<ScOrbitalCurvatureResult> curv =
      NEVPT2_TRY(scOrbitalCurvatureDevice(s, pairs, res));
  const ScOrbitalSandwichSetup ss{s.ints, s.gen, s.rdm, *s.ci, s.nelecA, s.nelecB, s.opt};
  std::vector<ScOrbitalSandwichRequest> reqs;
  reqs.reserve(pairs.size());
  for (const ScOrbitalCurvatureRequest& p : pairs)
    reqs.push_back(ScOrbitalSandwichRequest{p.u, p.v, p.tangent, false});
  const std::vector<ScOrbitalSandwichResult> sand =
      NEVPT2_TRY(scOrbitalSandwichDevice(ss, reqs, res));
  std::vector<ScOrbitalHessianResult> out;
  out.reserve(pairs.size());
  for (std::size_t q = 0; q < pairs.size(); ++q)
    out.push_back(ScOrbitalHessianResult{curv[q], sand[q]});
  return out;
}

// --- PC-NEVPT2's reverse mode (#74) -------------------------------------------

std::string PcGradientResult::refusals() const {
  std::string out;
  for (std::size_t c = 0; c < classes.size(); ++c) {
    const PcClassResult& r = classes[c].result;
    if (r.status == PcStatus::Done) continue;
    out += std::format("{}{} ({})", out.empty() ? "" : ", ", CLASSES[c],
                       r.status == PcStatus::Refused ? r.why : std::string("not computed"));
  }
  return out;
}

void printPcGradientClasses(const PcGradientResult& r) {
  std::printf("  %-6s %-8s %18s  %s\n", "class", "status", "PC energy (Ha)", "S's gap / why");
  for (std::size_t c = 0; c < r.classes.size(); ++c) {
    const PcClassResult& k = r.classes[c].result;
    const std::string gap =
        k.hasSpectrum ? std::format("{:.1e}", k.spectrum.gap) : std::string("-");
    if (k.status == PcStatus::Done)
      std::printf("  %-6s %-8s %18.10f  %s\n", CLASSES[c].c_str(), "done", k.energy, gap.c_str());
    else
      std::printf("  %-6s %-8s %18s  %s -- %s\n", CLASSES[c].c_str(), "REFUSED", "-", gap.c_str(),
                  k.status == PcStatus::Refused ? k.why.c_str() : "not computed");
  }
  std::printf("  E_corr (PC, the adjoint's own walk) = %.10f Ha%s\n", r.total,
              r.allDone() ? "" : "  (NOT a PC energy: a class was refused)");
}

Result<PcCiSourceResult> pcCiSourceDevice(const NevptIntegralsDevice& ints, const Tensor& ci,
                                          const int64_t nelecA, const int64_t nelecB,
                                          const DeviceTensor& dm1, const DeviceTensor& dm2,
                                          const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                          const DeviceTensor& f3ca,
                                          const SigmaVectorOptions& sigma,
                                          const OrbitalEnergyResponse* eps,
                                          const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t norb = dm1.dims[0];
  FullBlockSource src(ints);
  const ActiveIntegralsDevice ai = activeViewOf(ints);
  // The five Eq. 48 coefficient tensors, PC's: accumulated over the classes
  // by the same per-block sweeps SC's are (n^2 + n^4 + 3 n^6 doubles).
  ScRdmAdjoint adj = ScRdmAdjoint::zeros(norb, dr);
  PcCiSourceResult out;
  out.walk = NEVPT2_TRY(assemblePcClasses(ai, src, /*batch=*/0, dm1, dm2, dm3, f3ac, f3ca,
                                          ScRotation{}, &adj, /*epsAdjoint=*/eps != nullptr, dr));
  out.ok = out.walk.allDone();
  // A refused class has no derivative: y is NOT formed (never a sum with a
  // zero in it), and the caller reads the refusal off `walk`.
  if (!out.ok) return out;

  Tensor g1 = downloadTensor(adj.dm1, s);
  if (eps != nullptr) {
    out.orbitalEnergiesMove = true;
    out.dEdCore.assign(static_cast<std::size_t>(eps->core.dim(2)), 0.0);
    out.dEdVirt.assign(static_cast<std::size_t>(eps->virt.dim(2)), 0.0);
    for (const PcClassGradient& c : out.walk.classes) {
      check(c.dEdCore.size() == out.dEdCore.size() && c.dEdVirt.size() == out.dEdVirt.size(),
            "pcCiSourceDevice: a class's dE/de does not match the orbital-energy response");
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

// a + b as a fresh tensor from the pool, same shape.
DeviceTensor sumOfTensors(const DeviceTensor& a, const DeviceTensor& b,
                          const DeviceResources& dr) {
  std::vector<int> axes(a.dims.size());
  std::iota(axes.begin(), axes.end(), 0);
  DeviceTensor out = deviceTransposeNew(a, axes, 1.0, dr);
  addTensorInto(b, out, dr.stream());
  return out;
}

// The five coefficient tensors of one ScRdmAdjoint, host-resident (each
// downloadTensor synchronizes the stream the sweeps were issued on).
ScRdmCoefficients coefficientsOf(const ScRdmAdjoint& a, const wwrStream_t s) {
  return ScRdmCoefficients{downloadTensor(a.dm1, s), downloadTensor(a.dm2, s),
                           downloadTensor(a.dm3, s), downloadTensor(a.f3ac, s),
                           downloadTensor(a.f3ca, s)};
}

// pcCiHessianVectorDevice and pcCiSourceThreeTermDevice: one walk, `u` null
// for y alone.
Result<PcCiHessianResult> pcCiWalkDevice(const NevptIntegralsDevice& ints, const Tensor& ci,
                                         const Tensor* u, const int64_t nelecA,
                                         const int64_t nelecB, const RdmTensors& rdm,
                                         const RdmTensors* dotU, const RdmTensors* ddotUU,
                                         const bool perClass, const SigmaVectorOptions& sigma,
                                         const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t norb = rdm.dm1.dims[0];
  check(u == nullptr || (u->dims() == ci.dims() && dotU != nullptr && ddotUU != nullptr),
        "pcCiHessianVectorDevice: u must be shaped like the CI vector, with its two RDM sets");
  FullBlockSource src(ints);
  const ActiveIntegralsDevice ai = activeViewOf(ints);
  const PcRdms base{&rdm.dm1, &rdm.dm2, &rdm.dm3, &rdm.f3ac, &rdm.f3ca};

  // The RDMs plus each set: every class's matrix tangent is one subtraction
  // of two builds (nevpt2.energy's pcClassDirection), as pcHessianDevice's.
  struct Sum {
    DeviceTensor dm1, dm2, dm3, f3ac, f3ca;
    PcRdms view() const { return {&dm1, &dm2, &dm3, &f3ac, &f3ca}; }
  };
  const auto plus = [&](const RdmTensors& d) {
    return Sum{sumOfTensors(rdm.dm1, d.dm1, dr), sumOfTensors(rdm.dm2, d.dm2, dr),
               sumOfTensors(rdm.dm3, d.dm3, dr), sumOfTensors(rdm.f3ac, d.f3ac, dr),
               sumOfTensors(rdm.f3ca, d.f3ca, dr)};
  };
  std::optional<Sum> sumU, sumUU;
  PcRdms viewU, viewUU;
  if (u != nullptr) {
    sumU.emplace(plus(*dotU));
    sumUU.emplace(plus(*ddotUU));
    viewU = sumU->view();
    viewUU = sumUU->view();
  }

  // One accumulator pair shared by every class, or one per RDM class.
  const std::size_t nAdj = perClass ? 8 : 1;
  std::vector<ScRdmAdjoint> adj, adjDot;
  for (std::size_t k = 0; k < nAdj; ++k) {
    adj.push_back(ScRdmAdjoint::zeros(norb, dr));
    if (u != nullptr) adjDot.push_back(ScRdmAdjoint::zeros(norb, dr));
  }
  std::array<ScRdmAdjoint*, 8> pAdj{}, pAdjDot{};
  for (std::size_t c = 0; c < 8; ++c) {
    const std::size_t k = perClass ? c : 0;
    pAdj[c] = &adj[k];
    pAdjDot[c] = u != nullptr ? &adjDot[k] : nullptr;
  }

  PcCiHessianResult out;
  out.walk = NEVPT2_TRY(assemblePcCiClasses(ai, src, /*batch=*/0, base,
                                            u != nullptr ? &viewU : nullptr,
                                            u != nullptr ? &viewUU : nullptr, pAdj, pAdjDot, dr));
  out.ok = out.walk.allDone();
  // A refused class has no derivative: no vector is formed.
  if (!out.ok) return out;

  const Tensor h2e = downloadTensor(ai.h2e, s);
  const auto sigmaOf = [&](const ScRdmCoefficients& g, const Tensor& at) {
    return ciSourceVector(sigma, at, norb, nelecA, nelecB, g.dm1, g.dm2, g.dm3, g.f3ac, g.f3ca,
                          h2e, dr);
  };
  out.y = Tensor(ci.dims());
  if (u != nullptr) {
    out.hu = Tensor(ci.dims());
    out.fromWalkTangent = Tensor(ci.dims());
    out.fromDirection = Tensor(ci.dims());
  }
  for (std::size_t k = 0; k < nAdj; ++k) {
    // Sijrs's accumulators (class 2, per class) were never swept into: its
    // share of y and of H_cc u is exactly zero, and no sigma is run for it.
    if (perClass && k == 2) {
      out.yClass[k] = Tensor(ci.dims());
      if (u != nullptr) out.huClass[k] = Tensor(ci.dims());
      continue;
    }
    const ScRdmCoefficients g = coefficientsOf(adj[k], s);
    const Tensor yk = sigmaOf(g, ci);
    ciSourceAddInto(yk, out.y);
    if (perClass) out.yClass[k] = yk;
    if (u == nullptr) continue;
    const Tensor walkTangent = sigmaOf(coefficientsOf(adjDot[k], s), ci);
    const Tensor direction = sigmaOf(g, *u);
    ciSourceAddInto(walkTangent, out.fromWalkTangent);
    ciSourceAddInto(direction, out.fromDirection);
    Tensor hk = walkTangent;
    ciSourceAddInto(direction, hk);
    ciSourceAddInto(hk, out.hu);
    if (perClass) out.huClass[k] = std::move(hk);
  }
  return out;
}

Result<PcCiHessianResult> pcCiHessianVectorDevice(const NevptIntegralsDevice& ints,
                                                  const Tensor& ci, const Tensor& u,
                                                  const int64_t nelecA, const int64_t nelecB,
                                                  const RdmTensors& rdm, const RdmTensors& dotU,
                                                  const RdmTensors& ddotUU, const bool perClass,
                                                  const SigmaVectorOptions& sigma,
                                                  const DeviceResources& res) {
  return pcCiWalkDevice(ints, ci, &u, nelecA, nelecB, rdm, &dotU, &ddotUU, perClass, sigma, res);
}

Result<PcCiHessianResult> pcCiSourceThreeTermDevice(const NevptIntegralsDevice& ints,
                                                    const Tensor& ci, const int64_t nelecA,
                                                    const int64_t nelecB, const RdmTensors& rdm,
                                                    const bool perClass,
                                                    const SigmaVectorOptions& sigma,
                                                    const DeviceResources& res) {
  return pcCiWalkDevice(ints, ci, /*u=*/nullptr, nelecA, nelecB, rdm, /*dotU=*/nullptr,
                        /*ddotUU=*/nullptr, perClass, sigma, res);
}

// --- PC's mixed orbital-CI block, H_kc (#201) -----------------------------------

// The PC setup's own refusal, then SC's prep (the direction, its digests'
// tangent, h2e'[w]): PC's CI side holds the orbital energies fixed in c.
Result<ScMixedPrep> preparePcMixed(const ScMixedHessianSetup& s, const Tensor& w,
                                   const DeviceResources& res) {
  if (s.epsCi)
    return err_unsupported(
        "PC's mixed block H_kc with the orbital energies responding to the CI vector: PC's CI "
        "side (pcHessianDevice, pcCiHessianVectorDevice, the three-term y) holds them fixed in "
        "c, so there is no CI leg to pair that mode with -- run it with them fixed in c "
        "(responding to kappa or not)");
  return prepareMixed(s, w, res);
}

// The RDMs plus a tangent set, owning (pcCiWalkDevice's `plus`).
struct PcSumRdms {
  DeviceTensor dm1, dm2, dm3, f3ac, f3ca;
  PcRdms view() const { return {&dm1, &dm2, &dm3, &f3ac, &f3ca}; }
};

PcSumRdms pcSumOf(const RdmTensors& rdm, const DeviceTensor& dm1, const DeviceTensor& dm2,
                  const DeviceTensor& dm3, const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                  const DeviceResources& dr) {
  return PcSumRdms{sumOfTensors(rdm.dm1, dm1, dr), sumOfTensors(rdm.dm2, dm2, dr),
                   sumOfTensors(rdm.dm3, dm3, dr), sumOfTensors(rdm.f3ac, f3ac, dr),
                   sumOfTensors(rdm.f3ca, f3ca, dr)};
}

Result<std::vector<PcMixedHessianResult>> pcMixedHessianDevice(const ScMixedHessianSetup& s,
                                                               const Tensor& w,
                                                               const std::vector<Tensor>& us,
                                                               const DeviceResources& res) {
  ScMixedPrep p = NEVPT2_TRY(preparePcMixed(s, w, res));
  const NevptIntegralsDevice& ints = s.ints;
  const int64_t norb = ints.h1e.dims[0];
  Result<HostDm12Builder> builder = HostDm12Builder::create(norb, s.nelecA, s.nelecB);
  if (!builder) return std::unexpected(builder.error());
  for (const Tensor& u : us)
    if (u.dims() != s.ci.dims())
      return err_config("pcMixedHessianDevice: a CI direction is not shaped like the state");
  const Tensor h2e = downloadTensor(ints.h2e, res.stream());
  FullBlockSource src(ints);
  const ActiveIntegralsDevice ai = activeViewOf(ints);
  const PcRdms rdm{&s.rdm.dm1, &s.rdm.dm2, &s.rdm.dm3, &s.rdm.f3ac, &s.rdm.f3ca};

  std::vector<PcMixedHessianResult> out;
  out.reserve(us.size());
  for (std::size_t q = 0; q < us.size(); ++q) {
    const Tensor& u = us[q];
    // u's tangent RDMs (dm1/dm2 on the host, the rest the tangent build at
    // (c, u)); the (u, u) second-order set the lift test reads; and the
    // digests' MIXED tangent f3^{wu}, the same build at (c, u) handed h2e'[w]
    // (linear in h2e). Each as "the RDMs plus", the form every PC builder's
    // one-subtraction tangent takes; the mixed set's f3 slots hold f3^{wu}
    // alone (K_wu's subtraction base has zero digests).
    const HostDm12 dotU = builder->dm12Dot(s.ci, u);
    const HostDm12 dotUU = builder->dm12Dot(u, u);
    const DeviceTensor dm1U = uploadTensor(dotU.dm1, res);
    const DeviceTensor dm2U = uploadTensor(dotU.dm2, res);
    const DeviceTensor dm1UU = uploadTensor(dotUU.dm1, res);
    const DeviceTensor dm2UU = uploadTensor(dotUU.dm2, res);
    const RdmBuildResult tU =
        buildRdmTangentsDevice(s.opt, s.ci, u, norb, s.nelecA, s.nelecB, h2e, res);
    const RdmBuildResult tUU =
        buildRdmTangentsDevice(s.opt, u, u, norb, s.nelecA, s.nelecB, h2e, res);
    const RdmBuildResult tWU =
        buildRdmTangentsDevice(s.opt, s.ci, u, norb, s.nelecA, s.nelecB, p.h2eDot, res);
    const PcSumRdms sumU = pcSumOf(s.rdm, dm1U, dm2U, tU.dm3, tU.f3ac, tU.f3ca, res);
    const PcSumRdms sumUU = pcSumOf(s.rdm, dm1UU, dm2UU, tUU.dm3, tUU.f3ac, tUU.f3ca, res);
    PcSumRdms sumWU{sumOfTensors(s.rdm.dm1, dm1U, res), sumOfTensors(s.rdm.dm2, dm2U, res),
                    sumOfTensors(s.rdm.dm3, tU.dm3, res), tWU.f3ac.view(), tWU.f3ca.view()};
    const PcRdms vU = sumU.view(), vUU = sumUU.view(), vWU = sumWU.view();

    PcMixedWalk mw;
    mw.rot = &p.rot;
    mw.epsKappa = s.epsKappa;
    mw.sumU = &vU;
    mw.sumUU = &vUU;
    mw.sumWU = &vWU;
    const std::array<ScRdmAdjoint*, 8> none{};
    PcMixedHessianResult r;
    r.walk = NEVPT2_TRY(assemblePcMixedClasses(ai, src, rdm, mw, none, none, none, none, res));
    // A refused class has no mixed derivative along this (w, u): the
    // element is not formed, never one with a zero in it.
    if (!r.walk.allDone())
      return err_numerical(std::format(
          "PC's mixed block H_kc has no element along CI direction {}: {} refused", q,
          r.walk.refusals()));
    for (const PcClassGradient& c : r.walk.classes) {
      r.eddot += c.mixedEddot;
      r.rotation += c.mixedRotation;
      r.edotW += c.mixedEdotW;
      r.edotU += c.mixedEdotU;
    }
    r.h1eDotMax = p.h1eDotMax;
    r.h2eDotMax = p.h2eDotMax;
    out.push_back(std::move(r));
  }
  return out;
}

Result<PcMixedHessianVectorResult> pcMixedHessianVectorDevice(const ScMixedHessianSetup& s,
                                                              const Tensor& w,
                                                              const bool perClass,
                                                              const SigmaVectorOptions& sigma,
                                                              const DeviceResources& res) {
  ScMixedPrep p = NEVPT2_TRY(preparePcMixed(s, w, res));
  const wwrStream_t st = res.stream();
  const NevptIntegralsDevice& ints = s.ints;
  const int64_t norb = ints.h1e.dims[0];
  FullBlockSource src(ints);
  const ActiveIntegralsDevice ai = activeViewOf(ints);
  const PcRdms rdm{&s.rdm.dm1, &s.rdm.dm2, &s.rdm.dm3, &s.rdm.f3ac, &s.rdm.f3ca};
  const ScRotationDir& dir = p.rot.dirs[0];
  // The class sweeps' integrals: w's dotted ones and zero ones (only h1e /
  // h2e are read by a sweep; the rest is the state's).
  const ActiveIntegralsDevice aiW{dir.h1eDot.view(),   dir.h2eDot.view(),   ints.e_core.view(),
                                  ints.e_virt.view(),  ints.h1e_v_Sr.view(), ints.h1e_v_Si.view(),
                                  ints.h1e_v_Sir.view()};
  const ActiveIntegralsDevice aiZero{p.zeroH1e.view(),    p.zeroH2e.view(),
                                     ints.e_core.view(),  ints.e_virt.view(),
                                     ints.h1e_v_Sr.view(), ints.h1e_v_Si.view(),
                                     ints.h1e_v_Sir.view()};

  // Four accumulator sets, shared by every class or one per RDM class.
  const std::size_t nAdj = perClass ? 8 : 1;
  std::vector<ScRdmAdjoint> adj, adjDot, adjW, adjFree;
  for (std::size_t k = 0; k < nAdj; ++k) {
    adj.push_back(ScRdmAdjoint::zeros(norb, res));
    adjDot.push_back(ScRdmAdjoint::zeros(norb, res));
    adjW.push_back(ScRdmAdjoint::zeros(norb, res));
    adjFree.push_back(ScRdmAdjoint::zeros(norb, res));
  }
  std::array<ScRdmAdjoint*, 8> pAdj{}, pDot{}, pW{}, pFree{};
  for (std::size_t c = 0; c < 8; ++c) {
    const std::size_t k = perClass ? c : 0;
    pAdj[c] = &adj[k];
    pDot[c] = &adjDot[k];
    pW[c] = &adjW[k];
    pFree[c] = &adjFree[k];
  }
  PcMixedWalk mw;
  mw.rot = &p.rot;
  mw.epsKappa = s.epsKappa;
  mw.aiW = &aiW;
  mw.aiZero = &aiZero;
  PcMixedHessianVectorResult out;
  out.walk = NEVPT2_TRY(assemblePcMixedClasses(ai, src, rdm, mw, pAdj, pDot, pW, pFree, res));
  if (!out.walk.allDone())
    return err_numerical(std::format("PC's H_kc^T w has no value: {} refused",
                                     out.walk.refusals()));
  for (const PcClassGradient& c : out.walk.classes) out.edotW += c.mixedEdotW;
  out.h2eDotMax = p.h2eDotMax;

  // Per accumulator set: sigma(adjDot + adjW - adjFree, c; h2e) and the 4-body
  // sigmas at h2e'[w] with the plain digest coefficients (each downloadTensor
  // synchronizes the stream the sweeps were issued on).
  const Tensor h2e = downloadTensor(ints.h2e, st);
  const Tensor zero1({norb, norb});
  const Tensor zero2({norb, norb, norb, norb});
  const Tensor zero3({norb, norb, norb, norb, norb, norb});
  const auto net = [&](const DeviceTensor& a, const DeviceTensor& b, const DeviceTensor& f) {
    Tensor t = downloadTensor(a, st);
    const Tensor tb = downloadTensor(b, st);
    const Tensor tf = downloadTensor(f, st);
    for (int64_t i = 0; i < t.size(); ++i) t.flatRef(i) += tb.flat(i) - tf.flat(i);
    return t;
  };
  out.hkcTw = Tensor(s.ci.dims());
  out.fromWalkTangent = Tensor(s.ci.dims());
  out.fromDigests = Tensor(s.ci.dims());
  for (std::size_t k = 0; k < nAdj; ++k) {
    // Sijrs (class 2, per class) was never swept into: its share is zero.
    if (perClass && k == 2) {
      out.hkcTwClass[k] = Tensor(s.ci.dims());
      continue;
    }
    const Tensor walk = ciSourceVector(
        sigma, s.ci, norb, s.nelecA, s.nelecB, net(adjDot[k].dm1, adjW[k].dm1, adjFree[k].dm1),
        net(adjDot[k].dm2, adjW[k].dm2, adjFree[k].dm2),
        net(adjDot[k].dm3, adjW[k].dm3, adjFree[k].dm3),
        net(adjDot[k].f3ac, adjW[k].f3ac, adjFree[k].f3ac),
        net(adjDot[k].f3ca, adjW[k].f3ca, adjFree[k].f3ca), h2e, res);
    const Tensor digests =
        ciSourceVector(sigma, s.ci, norb, s.nelecA, s.nelecB, zero1, zero2, zero3,
                       downloadTensor(adj[k].f3ac, st), downloadTensor(adj[k].f3ca, st),
                       p.h2eDot, res);
    ciSourceAddInto(walk, out.fromWalkTangent);
    ciSourceAddInto(digests, out.fromDigests);
    Tensor hk = walk;
    ciSourceAddInto(digests, hk);
    ciSourceAddInto(hk, out.hkcTw);
    if (perClass) out.hkcTwClass[k] = std::move(hk);
  }
  return out;
}

Result<PcFullRotationResult> pcFullRotationDevice(
    const NevptIntegralsDevice& ints, const GeneralIntegralsDevice& gen, const DeviceTensor& dm1,
    const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceTensor& f3ac,
    const DeviceTensor& f3ca, const DeviceTensor* digestAdjoint,
    const std::vector<ScRotationRequest>& dirs, const DeviceResources& dr) {
  NEVPT2_TRY(checkGeneralShapes(ints, gen));
  if (digestAdjoint != nullptr && digestAdjoint->dims != ints.h2e.dims)
    return err_config("pcFullRotationDevice: the digests' adjoint is not shaped like h2e");
  FullBlockSource src(ints);
  const ActiveIntegralsDevice ai = activeViewOf(ints);
  const ScRotation rot = NEVPT2_TRY(makeGeneralRotation(dirs, ai.h1e, ai.h2e, gen, dm1, dr));
  PcFullRotationResult out;
  out.walk = NEVPT2_TRY(assemblePcClasses(ai, src, /*batch=*/0, dm1, dm2, dm3, f3ac, f3ca, rot,
                                          /*adj=*/nullptr, /*epsAdjoint=*/false, dr));
  out.ok = out.walk.allDone();
  if (!out.ok) return out;
  out.dirs.assign(dirs.size(), ScFullRotationPairing{});
  for (std::size_t d = 0; d < dirs.size(); ++d) {
    ScFullRotationPairing& p = out.dirs[d];
    for (const PcClassGradient& c : out.walk.classes) {
      p.ext += c.rotExt[d];
      p.act += c.rotAct[d];
      p.eps += c.rotEps[d];
      p.tangentIntegral += c.rotTangent[d];
    }
    p.integral = p.ext + p.act;
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

Result<Tensor> pcFullRotationMatrix(const NevptIntegralsDevice& ints,
                                    const GeneralIntegralsDevice& gen, const DeviceTensor& dm1,
                                    const DeviceTensor& dm2, const DeviceTensor& dm3,
                                    const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                    const DeviceTensor* digestAdjoint,
                                    const std::vector<std::pair<int64_t, int64_t>>& pairs,
                                    int64_t chunk, const DeviceResources& res) {
  const int64_t n = gen.nmo();
  if (chunk < 1) return err_config("pcFullRotationMatrix: chunk must be at least 1");
  Tensor y({n, n});
  std::vector<ScRotationRequest> reqs;
  std::vector<std::pair<int64_t, int64_t>> which;
  const auto flush = [&]() -> Status {
    if (reqs.empty()) return {};
    const PcFullRotationResult r = NEVPT2_TRY(pcFullRotationDevice(
        ints, gen, dm1, dm2, dm3, f3ac, f3ca, digestAdjoint, reqs, res));
    if (!r.ok)
      return err_numerical(std::format(
          "the PC orbital source term has no value: {} refused by the solve the energy runs "
          "(docs/pc-nevpt2.md, \"The singular-metric convention\")",
          r.walk.refusals()));
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
          "pcFullRotationMatrix: ({}, {}) is not a pair of distinct orbitals of {}", p, q, n));
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

Result<DeviceTensor> pcDigestAdjointDevice(const RdmBuildOptions& opt, const Tensor& ci,
                                           const int64_t nelecA, const int64_t nelecB,
                                           const NevptIntegralsDevice& ints,
                                           const DeviceTensor& dm1, const DeviceTensor& dm2,
                                           const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                           const DeviceTensor& f3ca, const DeviceResources& res) {
  const int64_t norb = dm1.dims[0];
  FullBlockSource src(ints);
  const ActiveIntegralsDevice ai = activeViewOf(ints);
  ScRdmAdjoint adj = ScRdmAdjoint::zeros(norb, res);
  const PcGradientResult walk =
      NEVPT2_TRY(assemblePcClasses(ai, src, /*batch=*/0, dm1, dm2, dm3, f3ac, f3ca, ScRotation{},
                                   &adj, /*epsAdjoint=*/false, res));
  if (!walk.allDone())
    return err_numerical(std::format(
        "the PC digests' adjoint has no value: {} refused by the solve the energy runs",
        walk.refusals()));
  return buildDigestAdjointDevice(opt, ci, norb, nelecA, nelecB, dm3, adj.f3ac, adj.f3ca, res);
}

Result<std::vector<ScDigestTangent>> pcDigestRotationTangentDevice(
    const RdmBuildOptions& opt, const Tensor& ci, const int64_t nelecA, const int64_t nelecB,
    const NevptIntegralsDevice& ints, const GeneralIntegralsDevice& gen, const DeviceTensor& dm1,
    const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceTensor& f3ac,
    const DeviceTensor& f3ca, const std::vector<ScRotationRequest>& dirs,
    const DeviceResources& res) {
  check(!opt.cublas && !opt.ozaki && !opt.fusedDigest,
        "pcDigestRotationTangentDevice runs the emitted or the native-fp64 BLAS digest "
        "(cli refuses --cublas / --ozaki / --fused-digest with --dump-source-terms)");
  NEVPT2_TRY(checkGeneralShapes(ints, gen));
  const wwrStream_t s = res.stream();
  const int64_t norb = dm1.dims[0];
  const ActiveIntegralsDevice ai = activeViewOf(ints);
  const ScRotation rot = NEVPT2_TRY(makeGeneralRotation(dirs, ai.h1e, ai.h2e, gen, dm1, res));
  const DeviceTensor dm1Zero = DeviceTensor::zeros(dm1.dims, res);
  const DeviceTensor dm2Zero = DeviceTensor::zeros(dm2.dims, res);
  const DeviceTensor dm3Zero = DeviceTensor::zeros(dm3.dims, res);
  std::vector<ScDigestTangent> out(dirs.size());
  for (std::size_t d = 0; d < rot.dirs.size(); ++d) {
    // downloadTensor synchronizes: the rotation's einsums are complete.
    const Tensor h2eDot = downloadTensor(rot.dirs[d].h2eDot, s);
    for (int64_t i = 0; i < h2eDot.size(); ++i)
      out[d].h2eDotMax = std::max(out[d].h2eDotMax, std::fabs(h2eDot.flat(i)));
    const RdmBuildResult dot = buildRdmsDevice(opt, ci, norb, nelecA, nelecB, h2eDot, res);
    const PcTangentResult t = NEVPT2_TRY(pcTangentsDevice(
        ints, dm1, dm2, dm3, f3ac, f3ca, dm1Zero, dm2Zero, dm3Zero, dot.f3ac, dot.f3ca,
        /*epsDot=*/nullptr, res));
    for (std::size_t c = 0; c < 8; ++c) {
      if (t.classes[c].result.status != PcStatus::Done)
        return err_numerical(std::format(
            "the PC digests' tangent along direction {}: {} refused by the solve ({})", d,
            CLASSES[c], t.classes[c].result.why));
      out[d].classes[c] = t.classes[c].edot;
      out[d].total += t.classes[c].edot;
    }
  }
  return out;
}

}  // namespace nevpt2
