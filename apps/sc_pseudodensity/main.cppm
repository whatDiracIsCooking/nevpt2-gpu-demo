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
// WITH --kappa-act IT CHECKS THE ORBITAL SOURCE TERM INSTEAD, the
// active-active block of Park's Eq. 26: the adjoint pairing <kappa, Y_act>
// out of those same pseudodensities against the forward tangent of the same
// classes along kappa, for several seeded antisymmetric directions, and both
// against the one finite difference in sight -- energiesDevice rerun at
// rotated integrals, which is what pins the rotation convention. The
// partition :kappa_check is where that lives, and it prints its own PASS
// line.
//
// WITH --kappa-full IT CHECKS THE WHOLE Y_xy -- core-core, virtual-virtual
// and mixed as well -- reading the gradient sidecar beside the golden
// (`<case>.gradgold`, or --gradgold PATH) for the one-general-index integral
// blocks and the full-MO-range Fock matrices a rotation that leaves the
// active space needs. Five checks, two finite differences and Park Eqs. 56-57
// come out of it; the partition :kappa_full_check is where that lives, and it
// prints its own PASS line too.
//
// WITH --dump-source-terms PATH IT WRITES THEM INSTEAD: the whole Y_xy (out of
// the same sidecar) and the CI source term y_I, as a `.gradsrc` that
// reference_data/nevpt2_grad.py --source dump builds a nuclear gradient from
// (#73). The partition :source_dump is where that lives; its PASS line says
// the file was written and read back unchanged. The writer is nevpt2.gradsrc,
// beside this file: the library computes, the binary decides to write.
//
// WITH --source-oracle PATH IT COMPARES THEM, block by block, with PySCF's
// exact source terms committed as a `.srcgold` beside the golden (#92): the
// blocks that agree must stay within the measured tolerance, and the blocks
// on the recorded known-gap list must still miss it. The partition
// :source_oracle is where that lives, and the list; it prints its own PASS
// line.

// A module unit, so each binary is apps/<app>/{main.cppm,CMakeLists.txt}.
// Nothing imports it. main() is written plainly: clang attaches main to the
// global module even inside a named module's purview (CWG 2811), so it links
// as the ordinary `main`.
export module nevpt2.app.sc_pseudodensity;

// --kappa-act: the active-active block of the orbital source term, checked
// against the forward tangent and against a finite difference of the energy
// itself. This binary's own flag (nothing else assembles pseudodensities),
// and its own PASS line.
export import :kappa_check;
// --kappa-full: the same for the WHOLE kappa space, out of the gradient
// sidecar beside the golden. Its own PASS line as well.
export import :kappa_full_check;
// --dump-source-terms: Y_xy and y_I written to a .gradsrc, and read back.
export import :source_dump;
// --source-oracle: the same two terms against PySCF's committed ones.
export import :source_oracle;

import std;
// nevpt2.cli: every flag this binary takes, and the ones it refuses.
import nevpt2.cli;
import nevpt2.gradient;  // re-exports nevpt2.energy (and so .einsum, .device_tensor, .wwr, .tensor)
import nevpt2.golden;
import nevpt2.link_tables;
import nevpt2.rdm_build;

int main(int argc, char** argv) {
  const std::vector<std::string_view> args(argv + 1, argv + argc);

  // Every flag, the common ones and this binary's own, and every check that
  // spans them (nevpt2.cli's parseArgs). A flag this binary does not
  // implement -- --pc (there is no PC pseudodensity assembly), --profile,
  // --rdm-tangent -- is refused as Unsupported rather than ignored; a bad
  // value or combination is a config Error. Reported once, here.
  nevpt2::Result<nevpt2::cli::Options> parsed =
      nevpt2::cli::parseArgs(nevpt2::cli::Binary::ScPseudodensity, args);
  if (!parsed) {
    nevpt2::report(parsed.error());
    return 1;
  }
  const nevpt2::cli::Options parsedOpts = *std::move(parsed);
  const nevpt2::cli::CommonOptions& opt = parsedOpts.common;
  const bool kappaActArg = parsedOpts.kappaAct;
  const bool kappaFullArg = parsedOpts.kappaFull;
  const std::string& dumpPath = parsedOpts.dumpSourceTermsPath;
  const std::string& oraclePath = parsedOpts.sourceOraclePath;
  std::string gradgoldPath = parsedOpts.gradgoldPath;

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

  // --kappa-full and --dump-source-terms read the gradient sidecar beside the
  // golden -- the path derived from the golden's unless --gradgold gave one.
  // A missing or malformed sidecar is the user's input, reported once here.
  const auto loadSidecar = [&]() -> nevpt2::Result<nevpt2::GradGoldFile> {
    if (gradgoldPath.empty()) {
      static constexpr std::string_view kGoldenSuffix = ".nevpt2gold";
      gradgoldPath = opt.goldenPath;
      if (gradgoldPath.ends_with(kGoldenSuffix))
        gradgoldPath.resize(gradgoldPath.size() - kGoldenSuffix.size());
      gradgoldPath += ".gradgold";
    }
    std::printf("gradient sidecar:  %s\n", gradgoldPath.c_str());
    return nevpt2::loadGradGold(gradgoldPath);
  };

  // --dump-source-terms: Y_xy and y_I, written (:source_dump). They feed the
  // CASCI-on-RHF Lagrangian of reference_data/nevpt2_grad.py, so a CASSCF
  // golden is refused as Unsupported, as nevpt2_demo's derivative paths
  // refuse one.
  if (!dumpPath.empty()) {
    if (const nevpt2::Status st = g.requireCasciReference("--dump-source-terms"); !st) {
      nevpt2::report(st.error());
      return 1;
    }
    nevpt2::Result<nevpt2::GradGoldFile> sidecar = loadSidecar();
    if (!sidecar) {
      nevpt2::report(sidecar.error());
      return 1;
    }
    const nevpt2::GradGoldFile gg = *std::move(sidecar);
    const nevpt2::Result<bool> ok =
        dumpSourceTerms(g, gg, energies, dInts, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca,
                        opt.rdm.nTiles, dumpPath, *res);
    if (!ok) {
      nevpt2::report(ok.error());
      return 1;
    }
    nevpt2::printPoolHighWater("for the whole run", *res);
    return *ok ? 0 : 1;
  }

  // --source-oracle: the same two terms against PySCF's exact ones
  // (:source_oracle). CASCI-on-RHF only, as the dump is.
  if (!oraclePath.empty()) {
    if (const nevpt2::Status st = g.requireCasciReference("--source-oracle"); !st) {
      nevpt2::report(st.error());
      return 1;
    }
    nevpt2::Result<nevpt2::GradGoldFile> sidecar = loadSidecar();
    if (!sidecar) {
      nevpt2::report(sidecar.error());
      return 1;
    }
    const nevpt2::GradGoldFile gg = *std::move(sidecar);
    const nevpt2::Result<bool> ok =
        checkSourceOracle(g, gg, oraclePath, energies, dInts, dDm1, dDm2, rdm.dm3, rdm.f3ac,
                          rdm.f3ca, opt.rdm.nTiles, *res);
    if (!ok) {
      nevpt2::report(ok.error());
      return 1;
    }
    nevpt2::printPoolHighWater("for the whole run", *res);
    return *ok ? 0 : 1;
  }

  // --kappa-full: the WHOLE source term, checked (:kappa_full_check).
  if (kappaFullArg) {
    nevpt2::Result<nevpt2::GradGoldFile> sidecar = loadSidecar();
    if (!sidecar) {
      nevpt2::report(sidecar.error());
      return 1;
    }
    const nevpt2::GradGoldFile gg = *std::move(sidecar);
    const nevpt2::Result<bool> ok = checkKappaFull(g, gg, energies, dInts, dDm1, dDm2, rdm.dm3,
                                                   rdm.f3ac, rdm.f3ca, *res);
    if (!ok) {
      nevpt2::report(ok.error());
      return 1;
    }
    nevpt2::printPoolHighWater("for the whole run", *res);
    return *ok ? 0 : 1;
  }

  // --kappa-act: the active-active orbital source term instead of the
  // identity report -- the adjoint pairing <kappa, Y_act> against the
  // forward tangent along the same direction, and both against a central
  // finite difference of energiesDevice at rotated integrals
  // (:kappa_check). Its own PASS line is the gate, and the identity report
  // below does not run.
  if (kappaActArg) {
    const nevpt2::Result<bool> ok = checkKappaAct(g, energies, dInts, dDm1, dDm2, rdm.dm3,
                                                  rdm.f3ac, rdm.f3ca, *res);
    if (!ok) {
      nevpt2::report(ok.error());
      return 1;
    }
    nevpt2::printPoolHighWater("for the whole run", *res);
    return *ok ? 0 : 1;
  }

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
