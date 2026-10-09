// SC-NEVPT2 demo, for any committed golden (CAS(10,10) is the default case):
// a proper GPU C++ program -- CUDA or HIP, one source tree -- no Python at
// runtime.
//
// Reads a golden CASSCF-input file (`generate_golden.py`'s output -- a CI
// vector, the cheap 1-/2-RDMs, and every MO-integral block, from a
// deterministic CASCI state on a real molecule -- see
// docs/reference-data.md for why CASCI and not a fresh CASSCF), then:
//
//   1-4. builds dm3/f3ac/f3ca on the device: nevpt2.rdm_build
//      (rdm/rdm_build.cpp), shared with the density-fitted demo --
//      Knowles-Handy link tables on the host, `kernels.cu` (the
//      CAS-independent generic kernel set) compiled
//      ONCE at build time and linked in (rdm/rdm_launch.cu's launchers),
//      the determinant axis tiled (`--tiles`, default 3 -- see docs/performance.md,
//      "Why CAS(10,10) needs tiling" for why this is not left to an
//      auto-sized budget), each tile's partial dm3/f3ac/f3ca accumulated *on
//      the device*, then the fdm2 correction + wedge reconstruction once on
//      the fully-summed result;
//   5. assembles the eight SC-NEVPT2 class energies **on the GPU**
//      (device_einsum.cu's generic contraction kernel + energy/energy.cpp,
//      itself a mechanical port of a NumPy reference) from the device-
//      resident RDMs plus the golden MO integrals -- no host round-trip for
//      dm3/f3ac/f3ca at all -- and reports them against the golden PySCF
//      answer. See docs/implementation.md, "Porting the class energies"
//      section for why this needs a *generic* device kernel rather than ~140
//      hand-specialized ones.
//
// "Integral-direct" names what this demo consumes: the full four-index MO
// blocks, read straight from the golden file. apps/density_fit/main.cppm is
// its sibling that builds them from three-index tensors instead.

// A module unit, so each binary is apps/<app>/{main.cppm,CMakeLists.txt}.
// Nothing imports it. main() is written plainly: clang attaches main to the
// global module even inside a named module's purview (CWG 2811), so it links
// as the ordinary `main` -- and spelling that `extern "C++"` draws -Wmain.
module;

// stderr is a macro, which `import std` does not carry, so <cstdio> comes in
// through the global module fragment.
#include <cstdio>

export module nevpt2.app.integral_direct;

import std;
// nevpt2.cli: the flags both demos take, and the checks that span them
// (--cublas is refused there on a build without it).
import nevpt2.cli;
import nevpt2.energy;  // re-exports nevpt2.einsum, .device_tensor, .wwr, .tensor
import nevpt2.golden;
import nevpt2.link_tables;
import nevpt2.profile;
import nevpt2.rdm_build;

int main(int argc, char** argv) {
  constexpr std::string_view kGoldenHint = "path.nevpt2gold";
  const std::vector<std::string_view> args(argv + 1, argv + argc);
  nevpt2::cli::CommonOptions opt;

  // Every flag this demo takes is a common one; anything else is a usage
  // error. A bad value, or a bad combination, is a config Error reported once.
  for (std::size_t i = 0; i < args.size(); ++i) {
    const nevpt2::Result<bool> used = nevpt2::cli::parseCommonFlag(args, i, opt);
    if (!used) {
      nevpt2::report(used.error());
      return 1;
    }
    if (!*used) {
      std::fputs(nevpt2::cli::usage(argv[0], kGoldenHint, "").c_str(), stderr);
      return 1;
    }
  }
  if (const nevpt2::Status st = nevpt2::cli::finalize(opt, kGoldenHint); !st) {
    nevpt2::report(st.error());
    return 1;
  }

  // A missing, unreadable or malformed golden file -- or one without an
  // array this demo reads -- is the user's input: reported once, here, with a
  // nonzero exit.
  nevpt2::Result<nevpt2::GoldenFile> loaded = nevpt2::loadGolden(opt.goldenPath);
  if (!loaded) {
    nevpt2::report(loaded.error());
    return 1;
  }
  const nevpt2::GoldenFile g = *std::move(loaded);
  if (const nevpt2::Status st = g.require(
          {"ci", "h1e", "h2e", "e_core", "e_virt", "h2e_v_Sr", "h1e_v_Sr", "h2e_v_Si",
           "h1e_v_Si", "cvcv", "h2e_v_Sijr", "h2e_v_Srsi", "h2e_v_Srs", "h2e_v_Sij",
           "h2e_v1_Sir", "h2e_v2_Sir", "h1e_v_Sir", "dm1", "dm2", "class_norms",
           "class_energies"});
      !st) {
    nevpt2::report(st.error());
    return 1;
  }
  if (opt.pc && !(g.arrays.contains("pc_class_energies") && g.arrays.contains("e_pc_total"))) {
    std::fprintf(stderr,
                 "error: --pc needs block2's pc_class_energies and e_pc_total, and %s has none "
                 "(generated with --no-pc: N2 CAS(12,12)'s metric has no gap -- the "
                 "12-orbital PC case is n2_ccpvdz_cas1012)\n",
                 opt.goldenPath.c_str());
    return 1;
  }
  const std::int64_t norb = g.ncas;
  const std::int64_t ndet =
      nevpt2::link_tables::num_strings(norb, g.nelecA) * nevpt2::link_tables::num_strings(norb, g.nelecB);

  std::print("=== SC-NEVPT2 demo ({}/C++): CAS({},{})  ncore={}  n_det={} ===\n",
             nevpt2::kGpuBackendName, norb, g.nelecA + g.nelecB, g.ncore, ndet);

  std::printf("golden reference: %s\n", opt.goldenPath.c_str());
  std::printf("golden E_CASCI=%.10f  golden E_corr=%.10f\n", g.eCasci,
              g.eNevpt2Total);

  // The one DeviceResources: device 0, its non-blocking stream,
  // memory pool, BLAS handle and dense-solver handle (--pc's eigensolves).
  // Every GPU call below is issued on its stream and every allocation drawn
  // from its pool, passed down explicitly. Creating it brings the device
  // context up.
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

  nevpt2::RdmBuildResult rdm = nevpt2::buildRdmsDevice(opt.rdm, g.get("ci"), norb, g.nelecA,
                                                       g.nelecB, g.get("h2e"), *res);
  double rdmSeconds = rdm.seconds;

  // --- assemble the eight class energies, GPU-resident end to end: the RDM
  // build's own output DeviceTensors go straight in (no host round trip for
  // dm3/f3ac/f3ca at all), the golden MO integrals are uploaded once, and
  // energy/energy.cpp's contraction runs. Every device tensor here is owning
  // and is freed (on res's stream) when main returns, so the
  // run ends with nothing allocated. ---
  nevpt2::NevptIntegralsDevice dInts;
  dInts.h1e = nevpt2::uploadTensor(g.get("h1e"), *res);
  dInts.h2e = nevpt2::uploadTensor(g.get("h2e"), *res);
  dInts.e_core = nevpt2::uploadTensor(g.get("e_core"), *res);
  dInts.e_virt = nevpt2::uploadTensor(g.get("e_virt"), *res);
  dInts.h2e_v_Sr = nevpt2::uploadTensor(g.get("h2e_v_Sr"), *res);
  dInts.h1e_v_Sr = nevpt2::uploadTensor(g.get("h1e_v_Sr"), *res);
  dInts.h2e_v_Si = nevpt2::uploadTensor(g.get("h2e_v_Si"), *res);
  dInts.h1e_v_Si = nevpt2::uploadTensor(g.get("h1e_v_Si"), *res);
  dInts.cvcv = nevpt2::uploadTensor(g.get("cvcv"), *res);
  dInts.h2e_v_Sijr = nevpt2::uploadTensor(g.get("h2e_v_Sijr"), *res);
  dInts.h2e_v_Srsi = nevpt2::uploadTensor(g.get("h2e_v_Srsi"), *res);
  dInts.h2e_v_Srs = nevpt2::uploadTensor(g.get("h2e_v_Srs"), *res);
  dInts.h2e_v_Sij = nevpt2::uploadTensor(g.get("h2e_v_Sij"), *res);
  dInts.h2e_v1_Sir = nevpt2::uploadTensor(g.get("h2e_v1_Sir"), *res);
  dInts.h2e_v2_Sir = nevpt2::uploadTensor(g.get("h2e_v2_Sir"), *res);
  dInts.h1e_v_Sir = nevpt2::uploadTensor(g.get("h1e_v_Sir"), *res);

  nevpt2::DeviceTensor dDm1 = nevpt2::uploadTensor(g.get("dm1"), *res);
  nevpt2::DeviceTensor dDm2 = nevpt2::uploadTensor(g.get("dm2"), *res);

  // --pc: the PC-NEVPT2 classes instead of the SC ones (all
  // eight), checked against block2's pc_class_energies and e_pc_total.
  // printPcReport prints the PASS/FAIL line: PASS, and exit 0,
  // only when all eight classes and the total are within 1e-7 and none was
  // refused.
  if (opt.pc) {
    std::printf("computing the PC-NEVPT2 class energies on the GPU...\n");
    auto t0 = std::chrono::steady_clock::now();
    nevpt2::Result<nevpt2::PcEnergyResult> pc =
        nevpt2::pcEnergiesDevice(dInts, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, *res);
    if (!pc) {  // an eigensolve's devInfo != 0
      nevpt2::report(pc.error());
      return 1;
    }
    double pcSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("GPU PC energy stage done in %.2fs\n", pcSeconds);
    std::printf("GPU RDM build: %.2fs   GPU PC energy stage: %.2fs   total: %.2fs\n",
                rdmSeconds, pcSeconds, rdmSeconds + pcSeconds);
    nevpt2::printPoolHighWater("for the whole run", *res);
    bool pcOk = nevpt2::printPcReport(*pc, g.get("pc_class_energies"),
                                      g.get("e_pc_total").flat(0), 1e-7);
    return pcOk ? 0 : 1;
  }

  // On res's stream, like every upload above, so all of it is ordered with
  // no events.
  std::printf("computing the eight SC-NEVPT2 class energies on the GPU...\n");
  auto t2 = std::chrono::steady_clock::now();
  nevpt2::EnergyResult r =
      nevpt2::energiesDevice(dInts, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, *res);
  auto t3 = std::chrono::steady_clock::now();
  double energySeconds = std::chrono::duration<double>(t3 - t2).count();
  std::printf("GPU energy contraction done in %.2fs\n", energySeconds);

  // --profile: one table per stage. Safe with no extra sync here: the RDM
  // build synchronized its stream before returning, and every energy_S*
  // function ends by reading its result back with downloadTensor, which
  // synchronizes the stream every kernel (and event) was issued on (see
  // energy/energy.cpp), so every recorded event pair is complete.
  nevpt2::profile::printReport(nevpt2::profile::kRdmBuild, rdmSeconds);
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
    ok = ok && dE < kAtol && std::fabs(r.norms[c] - refN) < kAtol &&
         r.norms[c] >= -kAtol;
    std::printf("%-7s %14.8f %16.10f %16.10f %10.2e\n",
                nevpt2::CLASSES[c].c_str(), r.norms[c], r.energies[c], refE, dE);
  }

  double dTotal = std::fabs(r.total - g.eNevpt2Total);
  ok = ok && dTotal < kAtol;
  std::printf("\nE_corr (NEVPT2, ours)  = %.10f Ha\n", r.total);
  std::printf("E_corr (NEVPT2, golden)= %.10f Ha  |delta|=%.2e\n",
              g.eNevpt2Total, dTotal);
  std::printf("E_total (E_CASCI + E_corr) = %.10f Ha\n", g.eCasci + r.total);
  std::printf("GPU RDM build: %.2fs   GPU energy contraction: %.2fs   total: %.2fs\n",
              rdmSeconds, energySeconds, rdmSeconds + energySeconds);
  nevpt2::printPoolHighWater("for the whole run", *res);
  std::printf("\n%s: per-class norms/energies and the total match the "
              "golden PySCF reference to %.0e\n",
              ok ? "PASS" : "FAIL", kAtol);
  return ok ? 0 : 1;
}
