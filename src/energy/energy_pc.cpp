// The PC-NEVPT2 class energies on the GPU, and their tangent with respect to
// the CI vector (pcTangentsDevice, whose per-class work is the :pc_tangent
// partition) -- see energy.cppm for the method, why differentiating the solve
// rather than the eigendecomposition is legitimate, and
// docs/pc-nevpt2.md, "PC-NEVPT2 on the device: the design" for the derivation
// and the measured cut/gap figures.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.energy;

import std;
import :shared;
import :pc_solve;
import :pc_classes;
import :pc_tangent;
import :sc_tangent;  // tangent_Sijrs: PC's Sijrs IS SC's, Delta term included (#74)
import nevpt2.profile;
// The wwrsolverStatus_t error specializations, which make gpuCheck accept a
// solver status (the calling unit imports them; nevpt2.wwr does not re-export them).
import wwr.extension.solver;

namespace nevpt2 {

Result<PcEnergyResult> pcEnergiesDevice(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                        int64_t batch, const DeviceTensor& dm1,
                                        const DeviceTensor& dm2, const DeviceTensor& dm3,
                                        const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                        const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  // --profile: the einsum launches below report in the energy table, as the
  // SC stage's do (energiesDevice).
  const profile::Section profileSection(profile::kEnergy);
  Tensor ec = downloadTensor(ai.e_core, s);
  Tensor ev = downloadTensor(ai.e_virt, s);

  PcEnergyResult r;
  r.classes[0] = NEVPT2_TRY(pc_Sr(ai, src, batch, ev, dm2, dm3, f3ac, f3ca, dr));
  r.classes[1] = NEVPT2_TRY(pc_Si(ai, src, batch, ec, dm1, dm2, dm3, f3ac, f3ca, dr));
  // Sijrs: no active index, PC = SC (the same function, not a copy).
  r.classes[2].status = PcStatus::Done;
  r.classes[2].energy = energy_Sijrs(ai, src, batch, dr).second;
  r.classes[3] = NEVPT2_TRY(pc_Sijr(ai, src, batch, ec, ev, dm1, dm2, dr));
  r.classes[4] = NEVPT2_TRY(pc_Srsi(ai, src, batch, ec, ev, dm1, dm2, dr));
  r.classes[5] = NEVPT2_TRY(pc_Srs(ai, src, batch, ev, dm1, dm2, dm3, dr));
  r.classes[6] = NEVPT2_TRY(pc_Sij(ai, src, batch, ec, dm1, dm2, dm3, dr));
  r.classes[7] = NEVPT2_TRY(pc_Sir(ai, src, batch, ec, ev, dm1, dm2, dm3, dr));
  return r;
}

Result<PcEnergyResult> pcEnergiesDevice(const NevptIntegralsDevice& ints, const DeviceTensor& dm1,
                                        const DeviceTensor& dm2, const DeviceTensor& dm3,
                                        const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                        const DeviceResources& res) {
  FullBlockSource src(ints);
  // Views: `ints` keeps owning everything.
  ActiveIntegralsDevice ai{ints.h1e.view(),      ints.h2e.view(),      ints.e_core.view(),
                           ints.e_virt.view(),   ints.h1e_v_Sr.view(), ints.h1e_v_Si.view(),
                           ints.h1e_v_Sir.view()};
  return pcEnergiesDevice(ai, src, /*batch=*/0, dm1, dm2, dm3, f3ac, f3ca, res);
}

Result<PcTangentResult> pcTangentsDevice(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                         int64_t batch, const DeviceTensor& dm1,
                                         const DeviceTensor& dm2, const DeviceTensor& dm3,
                                         const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                         const DeviceTensor& dm1Dot, const DeviceTensor& dm2Dot,
                                         const DeviceTensor& dm3Dot, const DeviceTensor& f3acDot,
                                         const DeviceTensor& f3caDot,
                                         const OrbitalEnergyDot* epsDot,
                                         const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  // --profile: the tangent's einsums land in the energy table beside the
  // energy stage's, as the SC and PC stages' do.
  const profile::Section profileSection(profile::kEnergy);
  Tensor ec = downloadTensor(ai.e_core, s);
  Tensor ev = downloadTensor(ai.e_virt, s);

  // The two bundles every class is built from: the RDMs themselves, and the
  // RDMs plus their tangents. Each class's matrix tangent is then ONE
  // subtraction of the two builds, which is exact because the matrices are
  // affine in the RDMs (energy_pc_tangent.cppm, `tangentMatrices`). Five
  // extra n^2..n^6 tensors, freed when this returns.
  const DeviceTensor dm1Sum = sumTensors(dm1, dm1Dot, dr);
  const DeviceTensor dm2Sum = sumTensors(dm2, dm2Dot, dr);
  const DeviceTensor dm3Sum = sumTensors(dm3, dm3Dot, dr);
  const DeviceTensor f3acSum = sumTensors(f3ac, f3acDot, dr);
  const DeviceTensor f3caSum = sumTensors(f3ca, f3caDot, dr);
  const PcRdms rdm{&dm1, &dm2, &dm3, &f3ac, &f3ca};
  const PcRdms sum{&dm1Sum, &dm2Sum, &dm3Sum, &f3acSum, &f3caSum};

  PcTangentResult r;
  r.classes[0] = NEVPT2_TRY(pcDotSr(ai, src, batch, ev, rdm, sum, epsDot, dr));
  r.classes[1] = NEVPT2_TRY(pcDotSi(ai, src, batch, ec, rdm, sum, epsDot, dr));
  // Sijrs contracts no RDM (no active index at all), so its energy is the SC
  // one -- the same function, not a copy -- and at fixed orbital energies its
  // tangent is EXACTLY zero, not a number too small to see. With `epsDot` its
  // Delta (four orbital energies) moves, and the SC tangent's own Sijrs term
  // is the answer: PC's Sijrs IS SC's (#74).
  r.classes[2].result.status = PcStatus::Done;
  {
    const ScTangentSums sijrs = tangent_Sijrs(ai, src, batch, epsDot, dr);
    r.classes[2].result.energy = sijrs.energy;
    r.classes[2].edot = sijrs.energyDot;
  }
  r.classes[3] = NEVPT2_TRY(pcDotSijr(ai, src, batch, ec, ev, rdm, sum, epsDot, dr));
  r.classes[4] = NEVPT2_TRY(pcDotSrsi(ai, src, batch, ec, ev, rdm, sum, epsDot, dr));
  r.classes[5] = NEVPT2_TRY(pcDotSrs(ai, src, batch, ev, rdm, sum, epsDot, dr));
  r.classes[6] = NEVPT2_TRY(pcDotSij(ai, src, batch, ec, rdm, sum, epsDot, dr));
  r.classes[7] = NEVPT2_TRY(pcDotSir(ai, src, batch, ec, ev, rdm, sum, epsDot, dr));
  return r;
}

Result<PcTangentResult> pcTangentsDevice(const NevptIntegralsDevice& ints,
                                         const DeviceTensor& dm1, const DeviceTensor& dm2,
                                         const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                         const DeviceTensor& f3ca, const DeviceTensor& dm1Dot,
                                         const DeviceTensor& dm2Dot, const DeviceTensor& dm3Dot,
                                         const DeviceTensor& f3acDot,
                                         const DeviceTensor& f3caDot,
                                         const OrbitalEnergyDot* epsDot,
                                         const DeviceResources& res) {
  FullBlockSource src(ints);
  // Views: `ints` keeps owning everything.
  ActiveIntegralsDevice ai{ints.h1e.view(),      ints.h2e.view(),      ints.e_core.view(),
                           ints.e_virt.view(),   ints.h1e_v_Sr.view(), ints.h1e_v_Si.view(),
                           ints.h1e_v_Sir.view()};
  return pcTangentsDevice(ai, src, /*batch=*/0, dm1, dm2, dm3, f3ac, f3ca, dm1Dot, dm2Dot,
                          dm3Dot, f3acDot, f3caDot, epsDot, res);
}

PcMatrices pcClassMatrices(const int cls, const DeviceTensor& h1e, const DeviceTensor& h2e,
                           const PcRdms& rdm, const DeviceResources& res) {
  // The class builders read the active integrals alone out of the bundle.
  const ActiveIntegralsDevice ai{h1e.view(), h2e.view(), {}, {}, {}, {}, {}};
  switch (cls) {
    case 0: return pcMatricesSr(ai, rdm, res);
    case 1: return pcMatricesSi(ai, rdm, res);
    case 3: return pcMatricesSijr(ai, rdm, res);
    case 4: return pcMatricesSrsi(ai, rdm, res);
    case 5: return pcMatricesSrs(ai, rdm, res);
    case 6: return pcMatricesSij(ai, rdm, res);
    case 7: return pcMatricesSir(ai, rdm, res);
    default: break;
  }
  check(false, std::format("pcClassMatrices: class {} has no PC matrices (Sijrs is SC's)", cls));
  return {};
}

Result<PcClassSolve> pcSolveClass(const PcMatrices& mx, const char* name,
                                  const DeviceResources& res) {
  const DeviceTensor s1 = mx.hasS1 ? mx.matS1() : DeviceTensor();
  ClassSolve cs = NEVPT2_TRY(solveClass(mx.matS(), mx.matK(), name, res,
                                        mx.hasS1 ? &s1 : nullptr, /*keepUt=*/true));
  PcClassSolve out;
  out.result = std::move(cs.result);
  out.lambda = std::move(cs.lambda);
  out.T = std::move(cs.T);
  out.t1 = std::move(cs.t1);
  out.Ut = std::move(cs.Ut);
  return out;
}

bool printPcReport(const PcEnergyResult& r, const Tensor& ref, double refTotal, double atol) {
  std::printf(
      "\nPC-NEVPT2 (all eight classes)\n"
      "S cut at tau=%.0e * s_max; a class is refused below a gap of %.0e "
      "(smallest kept / largest dropped, or / tau if nothing is dropped)\n",
      PC_TAU, PC_MIN_GAP);
  std::printf("%-6s %5s %7s %12s %12s %10s %10s %16s %16s %10s\n", "class", "d", "dropped",
              "max dropped", "min kept", "gap", "min den", "PC energy (Ha)",
              "block2 ref (Ha)", "|delta|");
  bool ok = true;
  int computed = 0, refused = 0;
  double maxDelta = 0.0;
  for (int c = 0; c < 8; ++c) {
    const PcClassResult& k = r.classes[c];
    const char* name = CLASSES[c].c_str();
    const double refE = ref.flat(c);
    std::string spec;
    if (k.hasSpectrum) {
      const PcSpectrum& sp = k.spectrum;
      spec = std::format("{:5} {:7} {:>12} {:12.1e} {:10.1e} {:>10}", sp.d, sp.dropped,
                         sp.dropped ? std::format("{:.1e}", sp.largestDropped) : "-",
                         sp.smallestKept, sp.gap,
                         k.status == PcStatus::Done ? std::format("{:.3f}", sp.minDenominator)
                                                    : "-");
    } else {
      spec = std::format("{:>5} {:>7} {:>12} {:>12} {:>10} {:>10}", "-", "-", "-", "-", "-", "-");
    }
    switch (k.status) {
      case PcStatus::NotYet:
        std::printf("%-6s %s %16s %16.10f %10s\n", name, spec.c_str(), "not yet", refE, "-");
        break;
      case PcStatus::Refused:
        ok = false;
        ++refused;
        std::printf("%-6s %s %16s %16.10f %10s  (%s)\n", name, spec.c_str(), "REFUSED", refE,
                    "-", k.why.c_str());
        break;
      case PcStatus::Done: {
        ++computed;
        const double dE = std::fabs(k.energy - refE);
        maxDelta = std::max(maxDelta, dE);
        ok = ok && dE < atol;
        std::printf("%-6s %s %16.10f %16.10f %10.2e%s\n", name, spec.c_str(), k.energy, refE, dE,
                    c == 2 ? "  (PC = SC)" : "");
        break;
      }
    }
  }
  // The gate: every class computed -- a NotYet or Refused class
  // fails it -- each within atol of block2, and the total within atol of
  // block2's e_pc_total, checked on its own rather than taken as implied by
  // the classes (eight deltas just under atol could sum past it).
  ok = ok && computed == 8;
  double total = 0.0;
  for (const PcClassResult& k : r.classes) total += k.energy;
  const double dTotal = std::fabs(total - refTotal);
  if (computed == 8) {
    ok = ok && dTotal < atol;
    std::printf("\nE_corr (PC-NEVPT2, ours)   = %.10f Ha\n", total);
    std::printf("E_corr (PC-NEVPT2, block2) = %.10f Ha  |delta|=%.2e\n", refTotal, dTotal);
  } else {
    std::printf("\nE_corr (PC-NEVPT2): not summed, %d of 8 classes computed\n", computed);
  }
  std::printf("PC check: %d of 8 classes computed, %d refused, max class |delta| vs block2 "
              "%.2e\n",
              computed, refused, maxDelta);
  // A different wording from the SC line, so a ctest entry for one path can
  // never pass on the other's output.
  std::printf("\n%s: PC-NEVPT2 per-class energies and the total match the golden block2 "
              "reference to %.0e\n",
              ok ? "PASS" : "FAIL", atol);
  return ok;
}

}  // namespace nevpt2
