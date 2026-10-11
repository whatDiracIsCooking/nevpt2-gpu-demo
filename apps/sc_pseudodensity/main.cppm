// The SC-NEVPT2 pseudodensity check, for any committed golden: the same
// CASCI state and the same GPU RDM build as apps/integral_direct/, then
// nevpt2.gradient's assembly of the gradient's density-like quantities (Park,
// JCTC 2019 / arXiv:1907.10180, Eqs. 41-47) instead of -- well, alongside --
// the class energies.
//
// WHAT IT ASSERTS, and why there is no finite difference anywhere. A class's
// norm and Dyall-Hamiltonian expectation value are quadratic forms in that
// class's external integrals and linear in its active-space blocks, so
// contracting the assembled pseudodensities back against their own operands
// must reproduce the class energy EXACTLY -- Park Eq. 40. The identities are
// algebraic, so the tolerance is numerical noise (1e-10), not a
// discretization error, and a single mis-transcribed index breaks them. On
// top of that each class energy, computed here through the amplitudes T
// (Eq. 15), is checked against nevpt2.energy's own answer for the same state
// to 1e-10 and against the golden PySCF per-class energy to 1e-7.
//
// It is the only caller of nevpt2.gradient; the demos do not link it.

// A module unit, so each binary is apps/<app>/{main.cppm,CMakeLists.txt}.
// Nothing imports it. main() is written plainly: clang attaches main to the
// global module even inside a named module's purview (CWG 2811), so it links
// as the ordinary `main`.
module;

// stderr is a macro, which `import std` does not carry, so <cstdio> comes in
// through the global module fragment.
#include <cstdio>

export module nevpt2.app.sc_pseudodensity;

import std;
// nevpt2.cli: --golden, --tiles and the rest of the flags the demos take.
import nevpt2.cli;
import nevpt2.gradient;  // re-exports nevpt2.energy (and so .einsum, .device_tensor, .wwr, .tensor)
import nevpt2.golden;
import nevpt2.link_tables;
import nevpt2.rdm_build;

int main(int argc, char** argv) {
  constexpr std::string_view kGoldenHint = "path.nevpt2gold";
  const std::vector<std::string_view> args(argv + 1, argv + argc);
  nevpt2::cli::CommonOptions opt;

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
  if (opt.pc) {
    std::fputs("error: --pc has no pseudodensity assembly yet (this checks the "
               "strongly-contracted one)\n",
               stderr);
    return 1;
  }

  // A missing, unreadable or malformed golden file -- or one without an array
  // this reads -- is the user's input: reported once, here, with a nonzero
  // exit.
  nevpt2::Result<nevpt2::GoldenFile> loaded = nevpt2::loadGolden(opt.goldenPath);
  if (!loaded) {
    nevpt2::report(loaded.error());
    return 1;
  }
  const nevpt2::GoldenFile g = *std::move(loaded);
  if (const nevpt2::Status st = g.require(
          {"ci", "h1e", "h2e", "e_core", "e_virt", "h2e_v_Sr", "h1e_v_Sr", "h2e_v_Si",
           "h1e_v_Si", "cvcv", "h2e_v_Sijr", "h2e_v_Srsi", "h2e_v_Srs", "h2e_v_Sij",
           "h2e_v1_Sir", "h2e_v2_Sir", "h1e_v_Sir", "dm1", "dm2", "class_energies"});
      !st) {
    nevpt2::report(st.error());
    return 1;
  }
  const std::int64_t norb = g.ncas;
  const std::int64_t ndet = nevpt2::link_tables::num_strings(norb, g.nelecA) *
                            nevpt2::link_tables::num_strings(norb, g.nelecB);
  if (const nevpt2::Status st = g.checkNdet(ndet); !st) {
    nevpt2::report(st.error());
    return 1;
  }

  std::print("=== SC-NEVPT2 pseudodensities ({}/C++): CAS({},{})  ncore={}  n_det={} ===\n",
             nevpt2::kGpuBackendName, norb, g.nelecA + g.nelecB, g.ncore, ndet);
  std::printf("golden reference: %s\n", opt.goldenPath.c_str());

  // The one DeviceResources: device 0, its non-blocking stream, memory pool,
  // BLAS handle and dense-solver handle. Every GPU call below is issued on its
  // stream and every allocation drawn from its pool.
  nevpt2::Result<std::shared_ptr<nevpt2::DeviceResources>> created =
      nevpt2::DeviceResources::create(0, opt.poolThreshold);
  if (!created) {
    nevpt2::report(created.error());
    return 1;
  }
  const std::shared_ptr<nevpt2::DeviceResources> res = *std::move(created);

  nevpt2::RdmBuildResult rdm = nevpt2::buildRdmsDevice(opt.rdm, g.get("ci"), norb, g.nelecA,
                                                       g.nelecB, g.get("h2e"), *res);

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

  // The reference the identities' class energies are held to at 1e-10: the
  // production path's own answer, from the same RDMs and the same integrals.
  std::printf("computing the eight SC-NEVPT2 class energies on the GPU...\n");
  const nevpt2::EnergyResult energies =
      nevpt2::energiesDevice(dInts, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, *res);

  std::printf("assembling the SC pseudodensities on the GPU...\n");
  const auto t0 = std::chrono::steady_clock::now();
  const nevpt2::ScGradientResult grad =
      nevpt2::scPseudodensityDevice(dInts, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, *res);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("pseudodensity assembly done in %.2fs\n", seconds);

  const bool ok = nevpt2::printScGradientReport(grad, energies, g.get("class_energies"),
                                                /*atol=*/1e-10, /*idtol=*/1e-10);
  nevpt2::printPoolHighWater("for the whole run", *res);
  return ok ? 0 : 1;
}
