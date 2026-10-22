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
// active space needs. Six checks, three finite differences and Park Eqs.
// 56-57 come out of it, with the f3ac/f3ca digests' own term printed beside
// the duality (#88); the partition :kappa_full_check is where that lives, and
// it prints its own PASS line too.
//
// WITH --sc-hessian-kc IT CHECKS ONE ORDER UP (#184): the energy's mixed
// orbital-CI Hessian block H_kc, the scalar Eddot_kc(w, u) and the vector
// H_kc^T w against each other by duality, with Euler in the CI vector, the
// intra-space exact zeros and finite differences of y along kappa
// (docs/hessian-theory.md, Sec. 5.5). It reads the sidecar when there is one
// and runs the active-active block alone, with every other w refused, when
// there is none; the partition :hessian_kc_check is where that lives, with its
// own PASS line. Not on a density-fitted case.
//
// WITH --sc-hessian-kk-int IT CHECKS THE ORBITAL BLOCK'S FIRST TERM (#154):
// the integrals' own curvature <Ehat', I''[u, v]> = K1 + K2 of the orbital
// Hessian (docs/hessian-theory.md, Sec. 5.1 and Sec. 5.4), each term's adjoint
// pairing against its forward tangent, the traced pairs' and the intra-space
// exact zeros, and Richardson-verified ladders of the fixed-pseudodensity
// pairing along exp(s u + t v), out of the sidecar (required). K2 for a pair
// with neither direction active-active is refused on a conventional case and
// computed in three-index form on a density-fitted one; the partition
// :hessian_kk_check is where that lives, with its own PASS line. With --pc
// the same checks run on PC-NEVPT2's curvature <Ehat'_pc, I''> (#200),
// after gating PC's first-order orbital source term on seeded directions --
// on a density-fitted case the only gate of it in tree.
//
// WITH --sc-hessian-kk IT CHECKS THE WHOLE ORBITAL BLOCK (#158): the second
// term, the energy's own curvature in its inputs I'^T Ehat'' I' = K3 + K4 +
// K5, forward against reverse, its intra-space exact zeros and its digests'
// curvature against the energy stage's at the plain digest path's tangents --
// and Eddot_kk(u, v) = <Ehat', I''> + the sandwich against Richardson-verified
// ladders of the SC total itself along exp(s u + t v), out of the sidecar
// (required); the partition :hessian_kk_sandwich_check, with its own PASS
// line. With --pc it is PC-NEVPT2's block (#202): the solve's sandwich
// forward against reverse in both orbital-energy modes, the ladders of the PC
// total, and the step-free gauge against #201's H_kc (:hessian_kk_pc_check,
// its own PASS line).
//
// WITH --sc-source-tangent PATH IT CHECKS THE SOURCE TERMS' TANGENT (#220):
// Y and y differentiated along every nuclear coordinate's perturbed
// Hamiltonian, the integral perturbation sidecar at PATH
// (`reference_data/integral_perturbation.py`), against the forward second
// derivative by duality and against ladders of the gated Y and y recomputed
// at the displaced integrals, out of the gradient sidecar (required); a
// coordinate whose perturbation populates empty perturbers is refused and
// counted. The partition :source_tangent_check, its own PASS line. With --pc
// it is PC-NEVPT2's (#234): Y_pc and the three-term y_pc, the orbital
// energies responding to c through W (PC's mixed block's FULL mode), no
// coordinate refused (PC has no empty-perturber term); the partition
// :source_tangent_pc_check, its own PASS line.
//
// A DENSITY-FITTED golden (`generate_golden.py --df`) runs here too (#165),
// with no flag of its own: it carries its fitted four-index blocks beside the
// B blocks, so the identities and --kappa-act read it as they read any
// golden, and --kappa-full knows the case by its sidecar's NEVPT2F1 magic,
// contracts the general-range inventory out of the whole B tensor and gates
// both of Y's terms, the digests' own included (#168). --dump-source-terms
// writes the fitted Y and y; --source-oracle (no fitted oracle exists) and
// --pc --dump-source-terms refuse a fitted sidecar. docs/gradients.md,
// Sec. 6.2, has the decisions.
//
// WITH --dump-source-terms PATH IT WRITES THEM INSTEAD: the whole Y_xy (out of
// the same sidecar), the CI source term y_I and Edot(kappa) along seeded
// directions in its two terms, as a `.gradsrc` that
// reference_data/nevpt2_grad.py --source dump builds a nuclear gradient from
// (#73) and --edot-check holds against PySCF's exact Edot (#88). The partition :source_dump is where that lives; its PASS line says
// the file was written and read back unchanged. The writer is nevpt2.gradsrc,
// beside this file: the library computes, the binary decides to write.
//
// WITH --source-oracle PATH IT COMPARES THEM, block by block, with PySCF's
// exact source terms committed as a `.srcgold` beside the golden (#92): the
// blocks that agree must stay within the measured tolerance, and the blocks
// on the recorded known-gap list must still miss it. The partition
// :source_oracle is where that lives, and the list; it prints its own PASS
// line.
//
// AND WITH --pc, BOTH OF THOSE ARE PC-NEVPT2'S (#74): --dump-source-terms and
// --source-oracle assemble the PC source terms -- nevpt2.gradient's PC reverse
// mode -- instead of SC's, against the PC energy (pcEnergiesDevice, itself held
// to the golden's block2 e_pc_total) and, for the oracle, a PC `.srcgold`.
// --pc means nothing to the identities or --kappa-act / --kappa-full, which
// are SC's, and nevpt2.cli refuses it there; with --sc-hessian-kk-int it makes
// the orbital curvature PC's (#200), with --sc-hessian-kc the mixed block
// PC's (#201, :hessian_kc_pc_check, its own PASS line), and with
// --sc-hessian-kk the sandwich and Eddot_kk PC's (#202, :hessian_kk_pc_check).

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
// --sc-hessian-kc: the energy's mixed orbital-CI Hessian block H_kc, its two
// orientations against each other (#184). Its own PASS line.
export import :hessian_kc_check;
// --pc --sc-hessian-kc: the same block of the PC energy (#201). Its own PASS
// line.
export import :hessian_kc_pc_check;
// --sc-hessian-kk-int: the orbital Hessian block's integral curvature
// <Ehat', I''> = K1 + K2 (#154). Its own PASS line.
export import :hessian_kk_check;
// --sc-hessian-kk: the orbital Hessian block's sandwich and the whole block
// Eddot_kk (#158). Its own PASS line.
export import :hessian_kk_sandwich_check;
// --pc --sc-hessian-kk: the same block of the PC energy (#202). Its own PASS
// line.
export import :hessian_kk_pc_check;
// --sc-source-tangent PATH: the SC source terms' tangent along a supplied
// integral perturbation (#220). Its own PASS line.
export import :source_tangent_check;
// --pc --sc-source-tangent PATH: the PC source terms' (#234). Its own PASS
// line.
export import :source_tangent_pc_check;

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
  // implement -- --profile, --rdm-tangent -- is refused as Unsupported rather
  // than ignored; a bad value or combination (--pc with anything but
  // --dump-source-terms, --source-oracle, --sc-hessian-kc,
  // --sc-hessian-kk-int or --sc-hessian-kk) is a config Error. Reported once,
  // here.
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
  // --pc: the state must carry block2's PC answer, which is also the
  // generator's word that its PC metric is gapped (docs/reference-data.md,
  // "PC fields") -- a PC derivative is never taken where the energy is not.
  if (opt.pc) {
    if (const nevpt2::Status st = g.require({"pc_class_energies", "e_pc_total"}); !st) {
      nevpt2::report(st.error());
      return 1;
    }
  }

  // --sc-hessian-kc / --sc-hessian-kk-int / --sc-hessian-kk (#157): level 1
  // differentiates the energy functional, which does not know how its
  // orbitals were made, so a Hessian sits behind the SOURCE-TERM gate -- which
  // takes a CASSCF golden as readily as a CASCI one -- and never behind the
  // CASCI-only Lagrangian gate (docs/hessian-theory.md, Sec. 9.1). With --pc
  // or without: PC's orbital blocks sit behind the same gate (#203). Before
  // DeviceResources::create, so the gate touches no card.
  if (parsedOpts.scHessianKc || parsedOpts.scHessianKkInt || parsedOpts.scHessianKk) {
    if (const nevpt2::Status st = g.requireSourceTermReference(
            parsedOpts.scHessianKc      ? "--sc-hessian-kc"
            : parsedOpts.scHessianKkInt ? "--sc-hessian-kk-int"
                                        : "--sc-hessian-kk");
        !st) {
      nevpt2::report(st.error());
      return 1;
    }
  }

  std::print(
      "=== SC-NEVPT2 pseudodensities ({}/C++): CAS({},{})  ncore={}  n_det={}  reference={} ===\n",
      nevpt2::kGpuBackendName, norb, g.nelecA + g.nelecB, g.ncore, ndet,
      nevpt2::referenceKindName(g.reference));
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

  // --pc (#74): the PC energy the PC source terms differentiate, held to the
  // golden's block2 total first -- a state the PC stage disagrees on, or
  // refuses a class of, has no PC derivative to write or compare.
  double energyTotal = energies.total;
  if (opt.pc) {
    std::printf("computing the PC-NEVPT2 class energies on the GPU...\n");
    nevpt2::Result<nevpt2::PcEnergyResult> pc =
        nevpt2::pcEnergiesDevice(dInts, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, *res);
    if (!pc) {
      nevpt2::report(pc.error());
      return 1;
    }
    if (!nevpt2::printPcReport(*pc, g.get("pc_class_energies"), g.get("e_pc_total").flat(0),
                               1e-7)) {
      std::printf("FAIL: the PC energy stage does not reproduce the golden on this state, so "
                  "there is no PC source term to take\n");
      return 1;
    }
    energyTotal = 0.0;
    for (const nevpt2::PcClassResult& c : pc->classes) energyTotal += c.energy;
  }

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

  // A density-fitted case (#165, #168) carries both of Y's terms, so the SC
  // dump writes it (docs/gradients.md, Sec. 6.2). Two things stay refused as
  // Unsupported:
  //   --pc on a fitted case: PC's orbital source term over the fitted
  //     inventory is gated only as --pc --sc-hessian-kk-int's first check
  //     (#200), which writes nothing; a PC dump over the fitted inventory has
  //     not been built, and a dump is never written ungated;
  //   --source-oracle on a fitted case: there is no oracle to compare with.
  //     reference_data/sc_source_oracle.py writes PySCF's / block2's exact
  //     source terms of the CONVENTIONAL calculation only, so no fitted
  //     .srcgold exists; the fitted Y's independent check is the nuclear
  //     gradient built from the dump against the fitted grad_num (#166, #167).
  const auto refuseFitted = [&](const nevpt2::GradGoldFile& gg,
                                std::string_view flag) -> nevpt2::Status {
    if (!gg.densityFitted()) return {};
    if (flag == "--source-oracle")
      return nevpt2::err_unsupported(
          "--source-oracle on a density-fitted case: there is no fitted oracle to compare "
          "against -- reference_data/sc_source_oracle.py writes the conventional "
          "calculation's exact source terms only; the fitted Y is checked by --kappa-full and "
          "by the nuclear gradient built from --dump-source-terms (docs/gradients.md, Sec. "
          "6.2)");
    if (opt.pc)
      return nevpt2::err_unsupported(std::format(
          "--pc {} on a density-fitted case: the PC dump over the fitted inventory is not "
          "built -- PC's fitted orbital source term is gated only inside --pc "
          "--sc-hessian-kk-int (docs/gradients.md, Sec. 6.2); without --pc the SC dump runs",
          flag));
    return {};
  };

  // --dump-source-terms: Y_xy and y_I, written (:source_dump). The terms
  // themselves are derivatives at fixed orbitals and CI vector, with no
  // reference-kind dependence, so this is the SOURCE-TERM gate; the
  // CASCI-only Lagrangian they feed is reference_data/nevpt2_grad.py's, and
  // refuses a CASSCF golden there.
  if (!dumpPath.empty()) {
    if (const nevpt2::Status st = g.requireSourceTermReference("--dump-source-terms"); !st) {
      nevpt2::report(st.error());
      return 1;
    }
    nevpt2::Result<nevpt2::GradGoldFile> sidecar = loadSidecar();
    if (!sidecar) {
      nevpt2::report(sidecar.error());
      return 1;
    }
    const nevpt2::GradGoldFile gg = *std::move(sidecar);
    if (const nevpt2::Status st = refuseFitted(gg, "--dump-source-terms"); !st) {
      nevpt2::report(st.error());
      return 1;
    }
    const nevpt2::Result<bool> ok =
        dumpSourceTerms(g, gg, opt.pc, energyTotal, dInts, dDm1, dDm2, rdm.dm3, rdm.f3ac,
                        rdm.f3ca, opt.rdm, dumpPath, parsedOpts.probedY, *res);
    if (!ok) {
      nevpt2::report(ok.error());
      return 1;
    }
    nevpt2::printPoolHighWater("for the whole run", *res);
    return *ok ? 0 : 1;
  }

  // --source-oracle: the same two terms against PySCF's exact ones
  // (:source_oracle). The source-term gate, as the dump's.
  if (!oraclePath.empty()) {
    if (const nevpt2::Status st = g.requireSourceTermReference("--source-oracle"); !st) {
      nevpt2::report(st.error());
      return 1;
    }
    nevpt2::Result<nevpt2::GradGoldFile> sidecar = loadSidecar();
    if (!sidecar) {
      nevpt2::report(sidecar.error());
      return 1;
    }
    const nevpt2::GradGoldFile gg = *std::move(sidecar);
    if (const nevpt2::Status st = refuseFitted(gg, "--source-oracle"); !st) {
      nevpt2::report(st.error());
      return 1;
    }
    const nevpt2::Result<bool> ok =
        checkSourceOracle(g, gg, oraclePath, opt.pc, energyTotal, dInts, dDm1, dDm2, rdm.dm3,
                          rdm.f3ac, rdm.f3ca, opt.rdm, *res);
    if (!ok) {
      nevpt2::report(ok.error());
      return 1;
    }
    nevpt2::printPoolHighWater("for the whole run", *res);
    return *ok ? 0 : 1;
  }

  // --sc-hessian-kc (#184): the mixed orbital-CI Hessian block
  // (:hessian_kc_check). It reads the sidecar when there is one -- --gradgold,
  // or `<case>.gradgold` if that file exists -- and with none it runs the one
  // thing a golden alone supports, an active-active w with the orbital
  // energies frozen, and says so on its PASS line. A --gradgold that cannot be
  // read is still the user's error, reported once here.
  if (parsedOpts.scHessianKc) {
    std::optional<nevpt2::GradGoldFile> gg;
    bool haveSidecar = !gradgoldPath.empty();
    if (!haveSidecar) {
      static constexpr std::string_view kGoldenSuffix = ".nevpt2gold";
      std::string derived = opt.goldenPath;
      if (derived.ends_with(kGoldenSuffix)) derived.resize(derived.size() - kGoldenSuffix.size());
      derived += ".gradgold";
      haveSidecar = std::filesystem::exists(derived);
      if (!haveSidecar)
        std::printf("gradient sidecar:  none (%s does not exist)\n", derived.c_str());
    }
    if (haveSidecar) {
      nevpt2::Result<nevpt2::GradGoldFile> sidecar = loadSidecar();
      if (!sidecar) {
        nevpt2::report(sidecar.error());
        return 1;
      }
      gg.emplace(*std::move(sidecar));
    }
    // With --pc it is PC's block (#201), on a state the PC energy stage above
    // has already held to the golden's block2 total; it runs on a
    // density-fitted sidecar too, where SC's refuses.
    const nevpt2::Result<bool> ok =
        opt.pc ? checkPcHessianKc(g, gg ? &*gg : nullptr, dInts, dDm1, dDm2, rdm, opt.rdm, *res)
               : checkScHessianKc(g, gg ? &*gg : nullptr, dInts, dDm1, dDm2, rdm, opt.rdm, *res);
    if (!ok) {
      nevpt2::report(ok.error());
      return 1;
    }
    nevpt2::printPoolHighWater("for the whole run", *res);
    return *ok ? 0 : 1;
  }

  // --sc-hessian-kk-int (#154): the orbital Hessian block's integral curvature
  // <Ehat', I''> (:hessian_kk_check). K1 reads the sidecar's arrays at a
  // symmetric argument, so the sidecar is required, as --kappa-full's is.
  // With --pc it is PC's (#200), on a state the PC energy stage above has
  // already held to the golden's block2 total.
  if (parsedOpts.scHessianKkInt) {
    nevpt2::Result<nevpt2::GradGoldFile> sidecar = loadSidecar();
    if (!sidecar) {
      nevpt2::report(sidecar.error());
      return 1;
    }
    const nevpt2::GradGoldFile gg = *std::move(sidecar);
    const nevpt2::Result<bool> ok =
        checkScHessianKkIntegral(g, gg, dInts, dDm1, dDm2, rdm, opt.rdm, opt.pc, *res);
    if (!ok) {
      nevpt2::report(ok.error());
      return 1;
    }
    nevpt2::printPoolHighWater("for the whole run", *res);
    return *ok ? 0 : 1;
  }

  // --sc-hessian-kk (#158): the orbital Hessian block's sandwich and the
  // whole block (:hessian_kk_sandwich_check). The sidecar is required, as
  // --sc-hessian-kk-int's is. With --pc it is PC's (#202,
  // :hessian_kk_pc_check), on a state the PC energy stage above has already
  // held to the golden's block2 total.
  if (parsedOpts.scHessianKk) {
    nevpt2::Result<nevpt2::GradGoldFile> sidecar = loadSidecar();
    if (!sidecar) {
      nevpt2::report(sidecar.error());
      return 1;
    }
    const nevpt2::GradGoldFile gg = *std::move(sidecar);
    const nevpt2::Result<bool> ok =
        opt.pc ? checkPcHessianKk(g, gg, dInts, dDm1, dDm2, rdm, opt.rdm, *res)
               : checkScHessianKk(g, gg, dInts, dDm1, dDm2, rdm, opt.rdm, *res);
    if (!ok) {
      nevpt2::report(ok.error());
      return 1;
    }
    nevpt2::printPoolHighWater("for the whole run", *res);
    return *ok ? 0 : 1;
  }

  // --sc-source-tangent PATH (#220): the SC source terms' tangent along the
  // integral perturbations of the `.pertgold` at PATH
  // (:source_tangent_check). Y is read out of the gradient sidecar, so it is
  // required; the perturbation is held to it (its frame, its calculation).
  // With --pc it is PC's (#234, :source_tangent_pc_check), on a state the PC
  // energy stage above has held to block2's.
  if (!parsedOpts.scSourceTangentPath.empty()) {
    if (const nevpt2::Status st = g.requireSourceTermReference("--sc-source-tangent"); !st) {
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
        opt.pc ? checkPcSourceTangent(g, gg, parsedOpts.scSourceTangentPath, dInts, dDm1, dDm2,
                                      rdm, opt.rdm, *res)
               : checkScSourceTangent(g, gg, parsedOpts.scSourceTangentPath, dInts, dDm1, dDm2,
                                      rdm, opt.rdm, *res);
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
                                                   rdm.f3ac, rdm.f3ca, opt.rdm, *res);
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
