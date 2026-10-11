// The SC-NEVPT2 pseudodensity assembly's entry points and report -- see
// gradient.cppm for what is assembled and which of Park's equations each
// piece is, and gradient_sc_classes.cppm for the eight classes themselves.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry

module nevpt2.gradient;

import std;
import :terms;
import :sc_classes;

namespace nevpt2 {

ScGradientResult scPseudodensityDevice(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                       int64_t batch, const DeviceTensor& dm1,
                                       const DeviceTensor& dm2, const DeviceTensor& dm3,
                                       const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                       const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t norb = dm1.dims[0];
  const Tensor hostECore = downloadTensor(ai.e_core, s);
  const Tensor hostEVirt = downloadTensor(ai.e_virt, s);

  // CLASSES order, one class at a time: each builds its own metric and
  // Dyall-Hamiltonian blocks, walks its slabs and frees everything when it
  // returns, exactly as energiesDevice's classes do.
  ScGradientResult r;
  gradientSr(ai, src, batch, hostEVirt, dm1, dm2, dm3, f3ac, f3ca, norb, r.classes[0], dr);
  gradientSi(ai, src, batch, hostECore, dm1, dm2, dm3, f3ac, f3ca, norb, r.classes[1], dr);
  gradientSijrs(src, batch, hostECore, hostEVirt, r.classes[2], dr);
  gradientSijr(ai, src, batch, hostECore, hostEVirt, dm1, dm2, r.classes[3], dr);
  gradientSrsi(ai, src, batch, hostECore, hostEVirt, dm1, dm2, r.classes[4], dr);
  gradientSrs(ai, src, batch, hostEVirt, dm1, dm2, dm3, r.classes[5], dr);
  gradientSij(ai, src, batch, hostECore, dm1, dm2, dm3, r.classes[6], dr);
  gradientSir(ai, src, batch, hostECore, hostEVirt, dm1, dm2, dm3, r.classes[7], dr);
  for (const ScClassGradient& c : r.classes) r.total += c.energy;
  return r;
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
