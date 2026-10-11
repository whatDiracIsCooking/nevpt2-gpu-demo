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
import :rotation;
import :terms;
import :sc_classes;

namespace nevpt2 {

// The whole assembly, with or without rotation directions: the eight classes
// in CLASSES order, each building its own metric and Dyall-Hamiltonian
// blocks, walking its slabs and freeing everything when it returns, exactly
// as energiesDevice's classes do.
ScGradientResult assembleClasses(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                 int64_t batch, const DeviceTensor& dm1,
                                 const DeviceTensor& dm2, const DeviceTensor& dm3,
                                 const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                 const ScRotation& rot, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t norb = dm1.dims[0];
  const Tensor hostECore = downloadTensor(ai.e_core, s);
  const Tensor hostEVirt = downloadTensor(ai.e_virt, s);

  ScGradientResult r;
  // One accumulator per direction per class, so a caller can read the
  // source term class by class (Sijrs' stay zero: it reads no active index).
  for (ScClassGradient& c : r.classes) {
    c.rotExt.assign(rot.size(), 0.0);
    c.rotAct.assign(rot.size(), 0.0);
    c.rotTangent.assign(rot.size(), 0.0);
  }
  gradientSr(ai, src, batch, hostEVirt, dm1, dm2, dm3, f3ac, f3ca, norb, rot, r.classes[0], dr);
  gradientSi(ai, src, batch, hostECore, dm1, dm2, dm3, f3ac, f3ca, norb, rot, r.classes[1], dr);
  gradientSijrs(src, batch, hostECore, hostEVirt, r.classes[2], dr);
  gradientSijr(ai, src, batch, hostECore, hostEVirt, dm1, dm2, rot, r.classes[3], dr);
  gradientSrsi(ai, src, batch, hostECore, hostEVirt, dm1, dm2, rot, r.classes[4], dr);
  gradientSrs(ai, src, batch, hostEVirt, dm1, dm2, dm3, rot, r.classes[5], dr);
  gradientSij(ai, src, batch, hostECore, dm1, dm2, dm3, rot, r.classes[6], dr);
  gradientSir(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dm3, rot, r.classes[7], dr);
  for (const ScClassGradient& c : r.classes) r.total += c.energy;
  return r;
}

ScGradientResult scPseudodensityDevice(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                       int64_t batch, const DeviceTensor& dm1,
                                       const DeviceTensor& dm2, const DeviceTensor& dm3,
                                       const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                       const DeviceResources& dr) {
  return assembleClasses(ai, src, batch, dm1, dm2, dm3, f3ac, f3ca, ScRotation{}, dr);
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
  out.grad = assembleClasses(ai, src, batch, dm1, dm2, dm3, f3ac, f3ca, rot, dr);
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
