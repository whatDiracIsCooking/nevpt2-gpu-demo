// nevpt2.app.sc_pseudodensity:source_oracle -- --source-oracle PATH: the
// GPU's own SC-NEVPT2 source terms held, block by block, to PySCF's EXACT
// ones, committed as a `.srcgold` beside the golden (#92). An interface
// partition that exports nothing, as :source_dump is.
//
// WHY. Every other on-card check of Y_xy and y_I compares the device with
// itself -- forward against reverse, at round-off -- so a term BOTH modes
// leave out cancels, and none of them can see it (#73 found two that way, by
// the offline nevpt2_grad.py --source both). This is that offline comparison
// with PySCF's side computed once, by reference_data/sc_source_oracle.py, and
// committed: the dump's own assembly (:source_dump's assembleSourceTerms),
// carried into the oracle's basis and compared by nevpt2.source_compare
// (source_compare.cppm: what is compared, the tolerances, and THE known-gap
// list, kKnownGaps -- EMPTY since #89).
//
// EITHER SCHEME, AND THE RUN SAYS WHICH. With --pc (#74) the terms assembled
// are PC-NEVPT2's (:source_dump's assembleSourceTerms, the PC solve's
// adjoint) and the oracle must be a PC one (`*_pc.srcgold`, "NEVPT2P1", #95:
// block2's exact PC source terms); without it, SC's against an SC oracle. A
// mismatch either way is a configuration error, never a comparison of one
// scheme's terms with the other's.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries. An oracle that does not
// describe this golden is the caller's input and propagates unchanged to
// main, which reports it once.
#include "error_handling/error_macros.h"

export module nevpt2.app.sc_pseudodensity:source_oracle;

import std;
import nevpt2.gradient;
import nevpt2.golden;
import nevpt2.source_compare;
// requireGradientArrays / uploadGeneral: the sidecar as --kappa-full reads it.
import :kappa_full_check;
// assembleSourceTerms: the dump's own Y and y.
import :source_dump;

// Compares the GPU's source terms for the golden's state with the oracle at
// `oraclePath`, prints the per-block table, and returns whether every block
// is where kKnownGaps says it is. A sidecar or an oracle that does not
// describe this golden, a PC oracle, or an oracle with no known-gap
// classification, is an Error the caller reports once.
nevpt2::Result<bool> checkSourceOracle(const nevpt2::GoldenFile& g,
                                       const nevpt2::GradGoldFile& gg,
                                       const std::string& oraclePath, const bool pc,
                                       const double energyTotal,
                                       const nevpt2::NevptIntegralsDevice& ints,
                                       const nevpt2::DeviceTensor& dm1,
                                       const nevpt2::DeviceTensor& dm2,
                                       const nevpt2::DeviceTensor& dm3,
                                       const nevpt2::DeviceTensor& f3ac,
                                       const nevpt2::DeviceTensor& f3ca,
                                       const nevpt2::RdmBuildOptions& rdmOpt,
                                       const nevpt2::DeviceResources& res) {
  NEVPT2_TRY(gg.checkAgainst(g));
  NEVPT2_TRY(requireGradientArrays(gg));
  std::printf("source-term oracle: %s\n", oraclePath.c_str());
  const nevpt2::SrcGoldFile o = NEVPT2_TRY(nevpt2::loadSrcGold(oraclePath));
  const nevpt2::SourceScheme want = pc ? nevpt2::SourceScheme::Pc : nevpt2::SourceScheme::Sc;
  if (o.scheme != want)
    return nevpt2::err_config(std::format(
        "--source-oracle: {} is a {} oracle, and this run assembles the {} source terms -- {}",
        oraclePath, nevpt2::sourceSchemeName(o.scheme), nevpt2::sourceSchemeName(want),
        pc ? "drop --pc for an SC oracle" : "pass --pc for a PC oracle"));
  NEVPT2_TRY(o.checkAgainst(g, gg));
  NEVPT2_TRY(nevpt2::checkDirectionBlocks(o));
  const bool directional = o.flavour == nevpt2::SourceFlavour::Directional;
  const std::string stem = nevpt2::sourceOracleStem(oraclePath);
  const std::vector<std::string_view> gaps = NEVPT2_TRY(nevpt2::knownSourceGaps(stem));
  const double tol = nevpt2::sourceOracleTol(o);

  std::printf("\n--source-oracle: the GPU's %s Y_xy and y_I against %s exact ones (#92%s%s)%s\n",
              pc ? "PC" : "SC", pc ? "block2's" : "PySCF's", pc ? ", #95, #74" : "",
              directional ? ", #105" : "",
              directional ? ", projected onto the oracle's seeded directions per block" : "");
  std::printf("  partition: n_mo = %lld = %lld core + %lld active (%lld act_O) + %lld virtual; "
              "PySCF's steps: kappa %.6g, CI %.6g%s\n",
              static_cast<long long>(o.nmo), static_cast<long long>(o.ncore),
              static_cast<long long>(o.nact), static_cast<long long>(o.nActOcc),
              static_cast<long long>(o.nvirt), o.kappaStep, o.ciStep,
              directional ? " (each ladder's first rung; Richardson of the first two)" : "");
  const nevpt2::GeneralIntegralsDevice gen = uploadGeneral(gg, res);
  // The digests' transpose (#89), so the Y compared is the exact one.
  const nevpt2::DeviceTensor hbar =
      pc ? NEVPT2_TRY(nevpt2::pcDigestAdjointDevice(rdmOpt, g.get("ci"), g.nelecA, g.nelecB,
                                                    ints, dm1, dm2, dm3, f3ac, f3ca, res))
         : nevpt2::scDigestAdjointDevice(rdmOpt, g.get("ci"), g.nelecA, g.nelecB, ints, dm1,
                                         dm2, dm3, f3ac, f3ca, res);
  const GpuSourceTerms terms = NEVPT2_TRY(assembleSourceTerms(
      g, gg, pc, energyTotal, ints, gen, dm1, dm2, dm3, f3ac, f3ca, hbar, rdmOpt.nTiles, res));

  double cy = 0.0;
  const nevpt2::Tensor a = nevpt2::oracleBasisA(o, terms.kappa);
  const nevpt2::Tensor py = nevpt2::projectOffState(g.get("ci"), terms.ci, cy);
  const nevpt2::SourceComparison cmp = nevpt2::compareSourceOracle(o, a, py, gaps, tol);

  // --- the verdict, block by block ------------------------------------------------
  std::printf("\n  per block, max over its %s (Eh; tolerance %.0e, docs/testing.md, \"%s\"):\n",
              directional ? "directions" : "pairs", tol,
              directional ? "The directional source-term oracle"
              : pc        ? "The PC source terms' oracle"
                          : "The SC source terms against PySCF's, on a card");
  std::printf("    %-12s %6s %12s %12s %11s  %-12s %s\n", "block", directional ? "dirs" : "pairs",
              "max|ref|",
              "max|GPU|", "max|delta|", "expected", "verdict");
  for (const nevpt2::SourceBlockRow& r : cmp.rows) {
    const std::string at =
        cmp.directional ? std::format("direction {}", r.atP)
        : r.atQ >= 0    ? std::format("({}, {})", r.atP, r.atQ)
                        : std::format("det {}", r.atP);
    std::printf("    %-12s %6lld %12.4e %12.4e %11.3e  %-12s %s  at %s\n", r.name.c_str(),
                static_cast<long long>(r.n), r.maxRef, r.maxGot, r.maxDelta,
                r.knownGap ? "known gap" : "agree", r.verdict.c_str(), at.c_str());
  }
  for (const std::string& f : cmp.failures)
    if (std::ranges::none_of(cmp.rows, [&](const nevpt2::SourceBlockRow& r) { return r.name == f; }))
      std::printf("    %s -- FAILED\n", f.c_str());
  std::printf("    (the GPU's <c, y> = %+.6e, dropped by the projection)\n", cy);
  const bool ok = cmp.ok && terms.walkOk && terms.epsOk;

  const auto join = [](const std::vector<std::string>& v) {
    std::string s;
    for (const std::string& x : v) s += (s.empty() ? "" : ", ") + x;
    return s.empty() ? std::string("none") : s;
  };
  const char* const scheme = pc ? "PC" : "SC";
  const char* const provider = pc ? "block2's" : "PySCF's";
  if (ok)
    std::printf("\nPASS: the GPU's %s source terms against %s committed ones, %s: %zu "
                "blocks agree within %.0e, and the %zu known gaps still disagree (%s)\n",
                scheme, provider, stem.c_str(), cmp.agreed.size(), tol, cmp.stillGaps.size(),
                join(cmp.stillGaps).c_str());
  else
    std::printf("\nFAIL: the GPU's %s source terms against %s committed ones, %s: %s\n",
                scheme, provider, stem.c_str(),
                cmp.failures.empty() ? "y's walk or the orbital energies it reads did not check out"
                                     : ("blocks out of place: " + join(cmp.failures)).c_str());
  return ok;
}
