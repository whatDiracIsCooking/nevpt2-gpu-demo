// scParameterHessianDevice: the level-1 parameter Hessian as a bilinear form
// over supplied directions (#222) -- see gradient.cppm, "the parameter
// Hessian as a bilinear form over supplied directions", for what it is, why
// two blocks are formed as vectors and one per pair, and what it refuses.
// An implementation unit of nevpt2.gradient: it composes the entry points the
// module already exports and adds no contraction of its own.
module;

// NEVPT2_TRY: a macro, which no import carries -- a malformed direction or a
// refused pair (the caller's input, or the state's) propagates unchanged.
#include "error_handling/error_macros.h"

module nevpt2.gradient;

import std;
// The host dm1/dm2 tangents along each CI direction, H_cc u's inputs.
import nevpt2.rdm_host_dm;

namespace nevpt2 {

// The integral set's active view, for the CI block's walk (gradient.cpp's
// activeViewOf, which lives in that unit).
ActiveIntegralsDevice parameterHessianActiveView(const NevptIntegralsDevice& ints) {
  return ActiveIntegralsDevice{ints.h1e.view(),      ints.h2e.view(),      ints.e_core.view(),
                               ints.e_virt.view(),   ints.h1e_v_Sr.view(), ints.h1e_v_Si.view(),
                               ints.h1e_v_Sir.view()};
}

double parameterHessianDot(const Tensor& x, const Tensor& y) {
  double s = 0.0;
  for (int64_t i = 0; i < x.size(); ++i) s += x.flat(i) * y.flat(i);
  return s;
}

bool parameterHessianIsZero(const Tensor& t) {
  for (int64_t i = 0; i < t.size(); ++i)
    if (t.flat(i) != 0.0) return false;
  return true;
}

Result<ScParameterHessianResult> scParameterHessianDevice(
    const ScParameterHessianSetup& s, const std::vector<ScParameterDirection>& dirs,
    const SigmaVectorOptions& sigma, const DeviceResources& res) {
  const int64_t n = static_cast<int64_t>(dirs.size());
  const int64_t nmo = s.gen.nmo();
  if (n == 0) return err_config("the parameter Hessian needs at least one direction");
  if (s.gen.densityFitted())
    return err_unsupported(
        "the parameter Hessian on a density-fitted case: its mixed block is integral-direct "
        "(scMixedHessianDevice's Data), and the response sidecars it reads are the "
        "conventional _casscf states' (docs/hessian-theory.md, Sec. 8)");
  for (int64_t a = 0; a < n; ++a) {
    const ScParameterDirection& d = dirs[static_cast<std::size_t>(a)];
    if (d.kappa.dims() != std::vector<int64_t>{nmo, nmo})
      return err_config(std::format(
          "parameter direction {}: kappa is not a {} x {} matrix", a, nmo, nmo));
    double asym = 0.0;
    for (int64_t p = 0; p < nmo; ++p)
      for (int64_t q = 0; q < nmo; ++q)
        asym = std::max(asym, std::fabs(d.kappa(p, q) + d.kappa(q, p)));
    if (asym > 1e-14)
      return err_config(std::format(
          "parameter direction {}: kappa is not antisymmetric (max |k + k^T| = {:.3e})", a,
          asym));
    if (d.ci.dims() != s.ci.dims())
      return err_config(
          std::format("parameter direction {}: its CI part is not shaped like the state", a));
  }

  ScParameterHessianResult out;
  out.hessian = Tensor({n, n});
  out.kk = Tensor({n, n});
  out.kc = Tensor({n, n});
  out.cc = Tensor({n, n});

  // --- the orbital block, per pair A <= B, refusals counted ------------------
  // One call per pair, so a refusal names its entry; every pair is tried, so
  // the count is the whole block's. Eddot_kk(u, v) is a symmetric bilinear
  // form, so (B, A) is (A, B)'s value; the caller holds that to swapped walks.
  const ScOrbitalCurvatureSetup cs{s.ints, s.gen, s.rdm, &s.hbar, &s.ci,
                                   s.nelecA, s.nelecB, s.opt};
  out.kkPairs.resize(static_cast<std::size_t>(n * n));
  const auto since = [](const std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
  };
  auto t0 = std::chrono::steady_clock::now();
  std::optional<Error> firstRefusal;
  std::string firstEntry;
  int64_t refused = 0, tried = 0;
  for (int64_t a = 0; a < n; ++a)
    for (int64_t b = a; b < n; ++b) {
      const Tensor& ka = dirs[static_cast<std::size_t>(a)].kappa;
      const Tensor& kb = dirs[static_cast<std::size_t>(b)].kappa;
      if (parameterHessianIsZero(ka) || parameterHessianIsZero(kb)) continue;
      ++tried;
      Result<std::vector<ScOrbitalHessianResult>> r =
          scOrbitalHessianDevice(cs, {ScOrbitalCurvatureRequest{ka, kb}}, res);
      if (!r) {
        ++refused;
        if (!firstRefusal) {
          firstRefusal = r.error();
          firstEntry = std::format("({}, {})", a, b);
        }
        continue;
      }
      out.kk(a, b) = out.kk(b, a) = r->front().value();
      out.kkPairs[static_cast<std::size_t>(a * n + b)] = std::move(r->front());
    }
  if (firstRefusal) {
    Error e = *firstRefusal;
    e.message = std::format(
        "the parameter Hessian is refused whole -- {} of its {} orbital-block pairs A <= B have "
        "no second derivative here, and a matrix is never returned with a zero in their place "
        "(docs/hessian-theory.md, Sec. 9.1); the first, entry {}: {}",
        refused, tried, firstEntry, e.message);
    return std::unexpected(std::move(e));
  }
  out.secondsKk = since(t0);

  // --- the mixed block, as 3N vectors H_kc^T kappa_A ----------------------------
  t0 = std::chrono::steady_clock::now();
  const ScMixedHessianSetup ms{s.ints, &s.gen, s.rdm,  s.ci, s.nelecA, s.nelecB,
                               s.opt,  true,   true,   &s.eps};
  out.hkcT.reserve(static_cast<std::size_t>(n));
  for (int64_t a = 0; a < n; ++a) {
    const Tensor& ka = dirs[static_cast<std::size_t>(a)].kappa;
    if (parameterHessianIsZero(ka)) {
      out.hkcT.emplace_back(s.ci.dims());
      continue;
    }
    ScMixedHessianVectorResult r =
        NEVPT2_TRY(scMixedHessianVectorDevice(ms, ka, sigma, res));
    out.hkcT.push_back(std::move(r.hkcTw));
  }
  out.secondsKc = since(t0);

  // --- the CI block, as 3N vectors H_cc c_A, FULL mode ---------------------------
  t0 = std::chrono::steady_clock::now();
  Result<HostDm12Builder> builder = HostDm12Builder::create(
      s.rdm.dm1.dims[0], s.nelecA, s.nelecB);
  if (!builder) return std::unexpected(builder.error());
  FullBlockSource src(s.ints);
  const ActiveIntegralsDevice ai = parameterHessianActiveView(s.ints);
  out.hcc.reserve(static_cast<std::size_t>(n));
  for (int64_t a = 0; a < n; ++a) {
    const Tensor& u = dirs[static_cast<std::size_t>(a)].ci;
    // Scoped: each direction's tangent RDMs are freed before the next is built.
    const HostDm12 dot = builder->dm12Dot(s.ci, u);
    const DeviceTensor dm1U = uploadTensor(dot.dm1, res);
    const DeviceTensor dm2U = uploadTensor(dot.dm2, res);
    const RdmBuildResult t = buildRdmTangentsDevice(s.opt, s.ci, u, s.rdm.dm1.dims[0],
                                                    s.nelecA, s.nelecB, s.h2e, res);
    const RdmTensors dotU{dm1U, dm2U, t.dm3, t.f3ac, t.f3ca};
    ScCiHessianResult r = scCiHessianVectorDevice(ai, src, /*batch=*/0, s.ci, u, s.nelecA,
                                                  s.nelecB, s.rdm, dotU, sigma, &s.eps, res);
    if (a == 0) {
      out.y = r.y;
    } else {
      for (int64_t i = 0; i < r.y.size(); ++i)
        out.ySpread = std::max(out.ySpread, std::fabs(r.y.flat(i) - out.y.flat(i)));
    }
    out.hcc.push_back(std::move(r.hu));
  }
  out.secondsCc = since(t0);

  // --- the dots, and the assembly ------------------------------------------------
  for (int64_t a = 0; a < n; ++a)
    for (int64_t b = 0; b < n; ++b) {
      const Tensor& cb = dirs[static_cast<std::size_t>(b)].ci;
      out.kc(a, b) = parameterHessianDot(out.hkcT[static_cast<std::size_t>(a)], cb);
      out.cc(a, b) = parameterHessianDot(cb, out.hcc[static_cast<std::size_t>(a)]);
    }
  for (int64_t a = 0; a < n; ++a)
    for (int64_t b = 0; b < n; ++b)
      out.hessian(a, b) = out.kk(a, b) + out.kc(a, b) + out.kc(b, a) + out.cc(a, b);
  return out;
}

}  // namespace nevpt2
