// SC-NEVPT2, density-fitted: the sibling of apps/integral_direct/main.cppm
// that never reads a four-index two-electron integral from the golden file.
//
// Reads a DENSITY-FITTED golden file (`generate_golden.py --df`: the whole
// CASCI and the NEVPT2 integrals density-fitted, the golden answer PySCF's
// DF-NEVPT2 -- not the conventional one, which differs by the fitting error)
// and, from its three-index B_aa/B_ca/B_va/B_cv tensors:
//
//   1. rebuilds the active block h2e on the device (one DGEMM over B_aa) and
//      runs the SAME tiled dm3/f3ac/f3ca build as the integral-direct demo
//      (nevpt2.rdm_build -- it contracts only that active block);
//   2. runs the SAME eight class energies (nevpt2.energy), with a
//      DfIntegralSource behind them: each class walks its external index in
//      `--batch`-sized slabs, and each slab is built on the fly from B by
//      wwrblasDgemm + a permutation (src/df_integrals), used, and freed. No full
//      external block (cvcv, h2e_v_Srsi, ...) is ever allocated.
//
// The one-electron inputs (h1e, h1e_v_*), the orbital energies and dm1/dm2
// still come from the golden file: they are the CASCI's, not a density-fitting
// target. `--check-blocks` (a diagnostic) additionally builds every external
// block whole and compares it with the four-index arrays a --df golden file
// also carries; the PASS line is about the energies alone.
//
// A module unit nothing imports; see apps/integral_direct/main.cppm.
module;

#include <cstdio>  // stderr, a macro `import std` does not carry

export module nevpt2.app.density_fit;

import std;
// This file's helper, in an interface partition that exports nothing (see
// its header): `export import` only because [module.unit] requires a
// primary interface to re-export its interface partitions.
export import :block_check;

// nevpt2.cli: every flag this demo takes, which flags it implements, and the
// checks that span them (--cublas is refused there on a build without it).
import nevpt2.cli;
import nevpt2.df_integrals;   // re-exports nevpt2.energy, .einsum, .device_tensor, .wwr
import nevpt2.golden;
import nevpt2.link_tables;
import nevpt2.profile;
import nevpt2.rdm_build;

int main(int argc, char** argv) {
  const std::vector<std::string_view> args(argv + 1, argv + argc);

  // Every flag, the common ones and this demo's own, and every check that
  // spans them (nevpt2.cli's parseArgs): a flag this demo does not implement
  // is refused, a bad value or combination is a config Error -- reported
  // once, here.
  nevpt2::Result<nevpt2::cli::Options> parsed =
      nevpt2::cli::parseArgs(nevpt2::cli::Binary::DfDemo, args);
  if (!parsed) {
    nevpt2::report(parsed.error());
    return 1;
  }
  const nevpt2::cli::Options parsedOpts = *std::move(parsed);
  const nevpt2::cli::CommonOptions& opt = parsedOpts.common;
  // --batch: slab length along each block's batch index, a MEMORY lever
  // (nevpt2::cli::kDefaultBatch says why 8); 0 = one slab per block.
  const std::int64_t batchArg = parsedOpts.batch;
  const bool checkBlocksArg = parsedOpts.checkBlocks;

  // A missing, unreadable or malformed golden file -- or one without an
  // array this demo reads -- is the user's input: reported once, here, with a
  // nonzero exit.
  nevpt2::Result<nevpt2::GoldenFile> loaded = nevpt2::loadGolden(opt.goldenPath);
  if (!loaded) {
    nevpt2::report(loaded.error());
    return 1;
  }
  const nevpt2::GoldenFile g = *std::move(loaded);
  for (const char* name : {"B_aa", "B_ca", "B_va", "B_cv"}) {
    if (!g.arrays.contains(name)) {
      std::fprintf(stderr,
                   "error: %s has no %s -- not a density-fitted golden file "
                   "(regenerate it with `generate_golden.py --df`)\n",
                   opt.goldenPath.c_str(), name);
      return 1;
    }
  }
  if (opt.pc && !(g.arrays.contains("pc_class_energies") && g.arrays.contains("e_pc_total"))) {
    std::fprintf(stderr,
                 "error: --pc needs block2's pc_class_energies and e_pc_total, and %s has none "
                 "(generated with --no-pc: N2 CAS(12,12)'s metric has no gap)\n",
                 opt.goldenPath.c_str());
    return 1;
  }
  if (const nevpt2::Status st =
          g.require({"ci", "h1e", "e_core", "e_virt", "h1e_v_Sr", "h1e_v_Si", "h1e_v_Sir", "dm1",
                     "dm2", "class_norms", "class_energies"});
      !st) {
    nevpt2::report(st.error());
    return 1;
  }
  if (checkBlocksArg) {
    using nevpt2::ExtBlock;
    using nevpt2::extBlockName;
    if (const nevpt2::Status st = g.require(
            {"h2e", extBlockName(ExtBlock::Sr), extBlockName(ExtBlock::Si),
             extBlockName(ExtBlock::Sijrs), extBlockName(ExtBlock::Sijr),
             extBlockName(ExtBlock::Srsi), extBlockName(ExtBlock::Srs), extBlockName(ExtBlock::Sij),
             extBlockName(ExtBlock::Sir1), extBlockName(ExtBlock::Sir2), "h2e_v_Srsi"});
        !st) {
      nevpt2::report(st.error());
      return 1;
    }
  }
  const std::int64_t norb = g.ncas;
  const std::int64_t ndet =
      nevpt2::link_tables::num_strings(norb, g.nelecA) * nevpt2::link_tables::num_strings(norb, g.nelecB);
  if (const nevpt2::Status st = g.checkNdet(ndet); !st) {
    nevpt2::report(st.error());
    return 1;
  }

  // `reference=`: which reference built the orbitals, as the golden's header
  // records it (docs/reference-data.md, "The file format"). Every committed
  // golden says CASCI.
  std::print(
      "=== SC-NEVPT2, density-fitted ({}/C++): CAS({},{})  ncore={}  n_det={}  "
      "naux={}  reference={} ===\n",
      nevpt2::kGpuBackendName, norb, g.nelecA + g.nelecB, g.ncore, ndet, g.get("B_aa").dim(0),
      nevpt2::referenceKindName(g.reference));
  std::printf("golden reference (PySCF DF-NEVPT2): %s\n", opt.goldenPath.c_str());
  std::printf("golden E_CASCI=%.10f  golden E_corr=%.10f\n", g.eCasci, g.eNevpt2Total);

  // Device first: the DF source shapes its slabs with the einsum transpose,
  // and the active h2e it builds feeds the RDM build. The one DeviceResources:
  // device 0, its non-blocking stream, memory pool, BLAS handle
  // and dense-solver handle (--pc's eigensolves). Every GPU call below is
  // issued on its stream and every allocation drawn from its pool, passed
  // down explicitly. Creating it brings the device context up. Every buffer
  // co-owns it, so it outlives the last free whatever the
  // declaration order.
  // A device that cannot be selected, or has no memory pools, is the
  // environment's failure: reported once, here.
  nevpt2::Result<std::shared_ptr<nevpt2::DeviceResources>> created =
      nevpt2::DeviceResources::create(0, opt.poolThreshold);
  if (!created) {
    nevpt2::report(created.error());
    return 1;
  }
  const std::shared_ptr<nevpt2::DeviceResources> res = *std::move(created);
  // --profile: every stage's launches and GEMMs, each stage
  // reported in its own table after the energy stage below.
  nevpt2::profile::setEnabled(opt.profile);

  // B blocks whose shapes disagree are a malformed golden file: reported
  // once, here.
  nevpt2::Result<nevpt2::DfIntegralSource> made = nevpt2::DfIntegralSource::create(
      g.get("B_aa"), g.get("B_ca"), g.get("B_va"), g.get("B_cv"), *res);
  if (!made) {
    nevpt2::report(made.error());
    return 1;
  }
  nevpt2::DfIntegralSource src = *std::move(made);
  nevpt2::DeviceTensor dH2e = src.activeH2e(*res);
  // downloadTensor synchronizes the stream before the host reads h2eHost.
  nevpt2::Tensor h2eHost = nevpt2::downloadTensor(dH2e, res->stream());

  bool blocksOk = true;
  if (checkBlocksArg) blocksOk = checkBlocks(src, g, h2eHost, 1e-10, *res);
  // The memory line below reports the energy stage's slabs, not this setup:
  // the active h2e and --check-blocks' whole blocks are built unbatched.
  src.resetPeak();

  nevpt2::RdmBuildResult rdm =
      nevpt2::buildRdmsDevice(opt.rdm, g.get("ci"), norb, g.nelecA, g.nelecB, h2eHost, *res);
  double rdmSeconds = rdm.seconds;

  // --rdm-tangent: the tangent of that build, on the same DF-rebuilt active
  // h2e, checked against twice the build (Euler; see the integral-direct
  // demo and docs/performance.md, "Tile floors").
  if (opt.rdmTangent) {
    if (const nevpt2::Status st = nevpt2::checkRdmTangentsDevice(
            opt.rdm, rdm, g.get("ci"), norb, g.nelecA, g.nelecB, h2eHost, *res);
        !st) {
      nevpt2::report(st.error());
      return 1;
    }
  }

  // Every device tensor from here on (and the RDM build's results) is owning,
  // and is freed on res's stream when main returns.
  nevpt2::ActiveIntegralsDevice active;
  active.h1e = nevpt2::uploadTensor(g.get("h1e"), *res);
  active.h2e = std::move(dH2e);
  active.e_core = nevpt2::uploadTensor(g.get("e_core"), *res);
  active.e_virt = nevpt2::uploadTensor(g.get("e_virt"), *res);
  active.h1e_v_Sr = nevpt2::uploadTensor(g.get("h1e_v_Sr"), *res);
  active.h1e_v_Si = nevpt2::uploadTensor(g.get("h1e_v_Si"), *res);
  active.h1e_v_Sir = nevpt2::uploadTensor(g.get("h1e_v_Sir"), *res);

  nevpt2::DeviceTensor dDm1 = nevpt2::uploadTensor(g.get("dm1"), *res);
  nevpt2::DeviceTensor dDm2 = nevpt2::uploadTensor(g.get("dm2"), *res);

  // --pc: the PC-NEVPT2 classes instead of the SC ones, over the
  // same DF slabs; see apps/integral_direct/main.cppm.
  if (opt.pc) {
    std::print("computing the PC-NEVPT2 class energies on the GPU from B (batch={}{})...\n",
               batchArg, batchArg == 0 ? " = whole blocks" : "");
    auto t0 = std::chrono::steady_clock::now();
    nevpt2::Result<nevpt2::PcEnergyResult> pc = nevpt2::pcEnergiesDevice(
        active, src, batchArg, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, *res);
    if (!pc) {  // an eigensolve's devInfo != 0
      nevpt2::report(pc.error());
      return 1;
    }
    double pcSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("GPU PC energy stage (incl. slab builds) done in %.2fs\n", pcSeconds);
    std::printf("GPU RDM build: %.2fs   GPU PC energy stage: %.2fs   total: %.2fs\n",
                rdmSeconds, pcSeconds, rdmSeconds + pcSeconds);
    // --profile: as for the SC stage below; every PC class ends in a
    // download (a stream sync), so every recorded event pair is complete.
    nevpt2::profile::printReport(nevpt2::profile::kRdmBuild, rdmSeconds);
    nevpt2::profile::printReport(nevpt2::profile::kDfIntegrals);
    nevpt2::profile::printReport(nevpt2::profile::kEnergy, pcSeconds);
    nevpt2::printPoolHighWater("for the whole run", *res);
    bool pcOk = nevpt2::printPcReport(*pc, g.get("pc_class_energies"),
                                      g.get("e_pc_total").flat(0), 1e-7);
    return pcOk ? 0 : 1;
  }

  // On res's stream, like every upload above (see apps/integral_direct/main.cppm).
  std::print(
      "computing the eight SC-NEVPT2 class energies on the GPU from B "
      "(batch={}{})...\n",
      batchArg, batchArg == 0 ? " = whole blocks" : "");
  auto t2 = std::chrono::steady_clock::now();
  nevpt2::EnergyResult r = nevpt2::energiesDevice(active, src, batchArg, dDm1, dDm2, rdm.dm3,
                                                  rdm.f3ac, rdm.f3ca, *res);
  auto t3 = std::chrono::steady_clock::now();
  double energySeconds = std::chrono::duration<double>(t3 - t2).count();
  std::printf("GPU energy contraction (incl. slab builds) done in %.2fs\n", energySeconds);

  // Memory bookkeeping, counted not measured: the largest single slab-build
  // allocation vs. the largest full external block the integral-direct path
  // would hold resident (and their sum, which it holds all at once).
  {
    using nevpt2::ExtBlock;
    std::int64_t fullMax = 0, fullSum = 0;
    for (ExtBlock b : {ExtBlock::Sr, ExtBlock::Si, ExtBlock::Sijrs, ExtBlock::Sijr,
                       ExtBlock::Srsi, ExtBlock::Srs, ExtBlock::Sij, ExtBlock::Sir1,
                       ExtBlock::Sir2}) {
      fullMax = std::max(fullMax, src.fullBlockDoubles(b));
      fullSum += src.fullBlockDoubles(b);
    }
    auto mb = [](std::int64_t doubles) { return doubles * 8.0 / (1024.0 * 1024.0); };

    std::printf(
        "external integrals: largest slab allocation %.3f MB; integral-direct would hold "
        "%.3f MB (largest block %.3f MB)\n",
        mb(src.peakSlabDoubles()), mb(fullSum), mb(fullMax));
  }

  // --profile: one table per stage, after syncs that cover them all (see
  // apps/integral_direct/main.cppm). The slab builds run inside the energy
  // stage, so the DF-integrals table has no wall-clock of its own (the
  // energy stage's includes it); it also holds the setup builds (B_vc, the
  // active h2e, --check-blocks' whole blocks).
  nevpt2::profile::printReport(nevpt2::profile::kRdmBuild, rdmSeconds);
  nevpt2::profile::printReport(nevpt2::profile::kDfIntegrals);
  nevpt2::profile::printReport(nevpt2::profile::kEnergy, energySeconds);

  const nevpt2::Tensor& refNorms = g.get("class_norms");
  const nevpt2::Tensor& refEnergies = g.get("class_energies");
  constexpr double kAtol = 1e-7;
  bool ok = true;

  std::printf("\n%-7s %14s %16s %16s %10s\n", "class", "norm", "energy (Ha)",
              "ref energy (Ha)", "|delta|");
  for (int c = 0; c < 8; ++c) {
    double refN = refNorms.flat(c), refE = refEnergies.flat(c);
    double dE = std::fabs(r.energies[c] - refE);
    ok = ok && dE < kAtol && std::fabs(r.norms[c] - refN) < kAtol && r.norms[c] >= -kAtol;
    std::printf("%-7s %14.8f %16.10f %16.10f %10.2e\n", nevpt2::CLASSES[c].c_str(),
                r.norms[c], r.energies[c], refE, dE);
  }

  double dTotal = std::fabs(r.total - g.eNevpt2Total);
  ok = ok && dTotal < kAtol;
  std::printf("\nE_corr (DF-NEVPT2, ours)  = %.10f Ha\n", r.total);
  std::printf("E_corr (DF-NEVPT2, golden)= %.10f Ha  |delta|=%.2e\n", g.eNevpt2Total, dTotal);
  // Full double precision, beside the 10-decimal line: the SC twin of
  // printPcReport's PC line (docs/pc-nevpt2.md, "The PC total's
  // run-to-run spread").
  std::printf("E_corr full precision (DF-NEVPT2) = %.17g Ha\n", r.total);
  std::printf("E_total (E_CASCI + E_corr) = %.10f Ha\n", g.eCasci + r.total);
  std::printf("GPU RDM build: %.2fs   GPU energy contraction: %.2fs   total: %.2fs\n",
              rdmSeconds, energySeconds, rdmSeconds + energySeconds);
  nevpt2::printPoolHighWater("for the whole run", *res);
  if (!blocksOk) std::printf("\nFAIL: --check-blocks found a DF-built block off the golden\n");
  ok = ok && blocksOk;
  std::printf("\n%s: per-class norms/energies and the total match the "
              "golden PySCF DF-NEVPT2 reference to %.0e\n",
              ok ? "PASS" : "FAIL", kAtol);
  return ok ? 0 : 1;
}
