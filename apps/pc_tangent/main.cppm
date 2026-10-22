// The PC-NEVPT2 tangent check, for any committed golden that carries PC
// fields: the same CASCI state and the same GPU RDM build as
// apps/integral_direct --pc, then nevpt2.energy's analytic derivative of the
// PC solve with respect to the CI vector (pcTangentsDevice), held against the
// exact Euler identity and against a Richardson-verified central finite
// difference of the PC total itself (nevpt2.pc_tangent).
//
// WHAT IT ASSERTS, and why there IS a finite difference here (the SC
// pseudodensity check, apps/sc_pseudodensity/, has none). The SC
// pseudodensities satisfy algebraic identities -- contract them back against
// their own operands and the class energy comes out -- but the PC derivative
// is the derivative of a SOLVE, and no such identity exists for it short of
// differentiating the eigendecomposition, which is precisely what this
// avoids. Two independent checks stand in:
//
//   * Euler, exact and step-free: every PC class energy is homogeneous of
//     degree 2 in the CI vector, so the tangent along the state itself must
//     be twice the energy.
//   * a central difference along c + h u, whose O(h^2) error must fall 4x per
//     halving of h and whose Richardson extrapolation must land on the
//     analytic tangent. Each step costs two more states, so two more full RDM
//     builds and PC stages.
//
// A class the gap or denominator check refuses is REPORTED as refused and
// fails the gate -- it is never given a zero derivative (nevpt2.energy's
// module header, "PC'S TANGENT DIFFERENTIATES THE SOLVE";
// docs/pc-nevpt2.md, "The singular-metric convention").
//
// With --pc-hessian (#152) it checks the SECOND directional derivative
// instead, nevpt2.energy's pcHessianDevice, at the golden's fixed orbital
// energies: Euler at second order (Eddot(c, u) == Edot(u), exact, on the same
// five classes) and a Richardson-verified central difference of the tangent
// Edot(v) along u -- the Hessian's one finite difference
// (docs/hessian-theory.md, Sec. 6 and 9).
//
// With --pc-hessian-vector (#155) it checks the VECTOR H_cc u instead,
// nevpt2.gradient's pcCiHessianVectorDevice, per class: the duality
// <v, H_cc u> == Eddot(u, v) against --pc-hessian's pcHessianDevice, the
// symmetry, Euler as a vector, the same negative control, and one
// Richardson-verified difference of the GPU's own three-term y
// (docs/hessian-theory.md, Sec. 4.2 and 6.2).
//
// With --gradgold PATH beside either Hessian flag (#214), the FULL mode: the
// orbital energies respond to the CI vector, W read from that sidecar (the
// reader --gradient-source uses, nevpt2.ci_checks' loadOrbitalEnergyResponse),
// on every side of every check, and Euler is skipped (not an identity there,
// docs/hessian-theory.md, Sec. 4.3). The PASS line names the mode.
//
// It is the only caller of nevpt2.pc_tangent; the demos do not link it.

// A module unit, so each binary is apps/<app>/{main.cppm,CMakeLists.txt}.
// Nothing imports it. main() is written plainly: clang attaches main to the
// global module even inside a named module's purview (CWG 2811), so it links
// as the ordinary `main`.
export module nevpt2.app.pc_tangent;

import std;
// nevpt2.cli: every flag this binary takes, and the ones it refuses.
import nevpt2.cli;
import nevpt2.golden;
// loadOrbitalEnergyResponse: the gradient sidecar's W, for the FULL mode.
import nevpt2.ci_checks;
import nevpt2.link_tables;
// nevpt2.pc_tangent re-exports nevpt2.energy (and so .einsum, .device_tensor,
// .wwr, .tensor), nevpt2.rdm_build and nevpt2.rdm_host_dm.
import nevpt2.pc_tangent;

int main(int argc, char** argv) {
  // The finite-difference steps: --fd-steps successive halvings of
  // 2^-(--fd-start-pow). Why the first step is 2^-12 by default, and why
  // never fewer than three steps, is beside their defaults in nevpt2.cli
  // (kDefaultFdStartPow, kMinFdSteps). nevpt2.cli does not link this
  // binary's nevpt2.pc_tangent, so the floor it checks is pinned to the
  // report's here: three steps make a run of kPcMinFourfoldRun = 2 fourfold
  // halvings, the fewest printPcTangentReport accepts.
  static_assert(nevpt2::cli::kMinFdSteps == nevpt2::kPcMinFourfoldRun + 1,
                "--fd-steps' floor must be the report's minimum fourfold run plus one");
  // The direction's seed: fixed, so the whole check is reproducible run to
  // run and card to card.
  constexpr std::uint64_t kSeed = 0x5eed'0033u;
  // The tolerances the report gates on. `kIdTol` covers the two exact
  // checks -- the tangent path's class energies against pcEnergiesDevice's
  // own, and Euler's identity on the homogeneous classes -- which are
  // algebra, so only rounding separates them. `kAtol` and the ratio band are
  // what the finite difference can deliver, measured
  // (docs/pc-nevpt2.md, "The tangent with respect to the CI vector").
  constexpr double kIdTol = 1e-9;
  constexpr double kAtol = 1e-8;
  constexpr double kRatioTol = 0.5;

  const std::vector<std::string_view> args(argv + 1, argv + argc);

  // Every flag, the common ones and this binary's own, and every check that
  // spans them (nevpt2.cli's parseArgs). A flag this binary does not
  // implement is refused as Unsupported rather than ignored: --pc (it is PC
  // throughout), --profile, --rdm-tangent, and the --cublas / --ozaki family
  // and --fused-digest (the tangent RDM build is the emitted or the
  // native-fp64 BLAS digest only). A bad value or combination is a config
  // Error. Reported once, here.
  nevpt2::Result<nevpt2::cli::Options> parsed =
      nevpt2::cli::parseArgs(nevpt2::cli::Binary::PcTangent, args);
  if (!parsed) {
    nevpt2::report(parsed.error());
    return 1;
  }
  const nevpt2::cli::Options parsedOpts = *std::move(parsed);
  const nevpt2::cli::CommonOptions& opt = parsedOpts.common;
  const int fdSteps = parsedOpts.fdSteps;
  const int startPow = parsedOpts.fdStartPow;

  // A missing, unreadable or malformed golden file -- or one without an array
  // this reads -- is the user's input: reported once, here, with a nonzero
  // exit.
  nevpt2::Result<nevpt2::GoldenFile> loaded = nevpt2::loadGolden(opt.goldenPath);
  if (!loaded) {
    nevpt2::report(loaded.error());
    return 1;
  }
  const nevpt2::GoldenFile g = *std::move(loaded);
  // `pc_class_energies` is not compared against here -- this check is
  // internal consistency plus a finite difference, not a block2 comparison --
  // but a file that carries it is one whose PC answer the generator's
  // _check_pc_conditioning vetted (docs/reference-data.md, "PC fields"), and
  // a PC derivative on an ill-conditioned metric is exactly what must not be
  // computed. dm1/dm2 are NOT required: nevpt2.rdm_host_dm rebuilds them from
  // `ci`, as the displaced states need anyway.
  if (const nevpt2::Status st = g.require(
          {"ci", "h1e", "h2e", "e_core", "e_virt", "h2e_v_Sr", "h1e_v_Sr", "h2e_v_Si",
           "h1e_v_Si", "cvcv", "h2e_v_Sijr", "h2e_v_Srsi", "h2e_v_Srs", "h2e_v_Sij",
           "h2e_v1_Sir", "h2e_v2_Sir", "h1e_v_Sir", "pc_class_energies"});
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
  // --pc-hessian / --pc-hessian-vector (#157): level 1 differentiates the
  // energy functional, which does not know how its orbitals were made, so a
  // Hessian sits behind the SOURCE-TERM gate -- which takes a CASSCF golden as
  // readily as a CASCI one -- and never behind the CASCI-only Lagrangian gate
  // (docs/hessian-theory.md, Sec. 9.1). Before DeviceResources::create, so the
  // gate touches no card.
  if (parsedOpts.pcHessian || parsedOpts.pcHessianVector) {
    if (const nevpt2::Status st = g.requireSourceTermReference(
            parsedOpts.pcHessian ? "--pc-hessian" : "--pc-hessian-vector");
        !st) {
      nevpt2::report(st.error());
      return 1;
    }
  }

  std::print("=== PC-NEVPT2 CI tangent ({}/C++): CAS({},{})  ncore={}  n_det={}  reference={} ===\n",
             nevpt2::kGpuBackendName, norb, g.nelecA + g.nelecB, g.ncore, ndet,
             nevpt2::referenceKindName(g.reference));
  std::printf("golden reference: %s\n", opt.goldenPath.c_str());

  // --gradgold (#214; nevpt2.cli has already refused it without a Hessian
  // flag): the FULL mode's W, read and held to the golden's orbital energies
  // before any card is touched. A sidecar that does not fit is the user's
  // input, reported once here.
  std::optional<nevpt2::OrbitalEnergyResponse> epsResponse;
  if (!parsedOpts.gradgoldPath.empty()) {
    nevpt2::Result<nevpt2::OrbitalEnergyResponse> w =
        nevpt2::loadOrbitalEnergyResponse(g, parsedOpts.gradgoldPath);
    if (!w) {
      nevpt2::report(w.error());
      return 1;
    }
    epsResponse = *std::move(w);
  }
  const nevpt2::OrbitalEnergyResponse* eps = epsResponse ? &*epsResponse : nullptr;

  std::vector<double> steps;
  steps.reserve(static_cast<std::size_t>(fdSteps));
  const double firstStep = std::ldexp(1.0, -startPow);
  for (int i = 0; i < fdSteps; ++i) steps.push_back(std::ldexp(firstStep, -i));
  std::printf("direction: seed=0x%llx, projected against ci and normalized; %d finite-"
              "difference steps from 2^-%d = %g\n",
              static_cast<unsigned long long>(kSeed), fdSteps, startPow, firstStep);

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

  // --pc-hessian (#152): the second directional derivative Eddot(u, v)
  // instead, at the golden's fixed orbital energies, against Euler at second
  // order and a Richardson-verified central difference of Edot(v) along u
  // (nevpt2.pc_tangent's pcHessianFdCheck). u is the tangent check's own
  // direction (kSeed), so its Euler half differentiates the very Edot(u) this
  // binary's tangent run gates.
  if (parsedOpts.pcHessian) {
    constexpr std::uint64_t kSeedV = 0x5eed'0152u;
    constexpr std::uint64_t kSeedControl = 0x5eed'c152u;
    // The tolerances, each with its reason (the measured residuals are in
    // docs/pc-nevpt2.md, "The second derivative, Eddot(u, v)"):
    //   kHessIdTol    the exact checks -- the Hessian walk's energies against
    //                 pcEnergiesDevice's, and Euler -- which are algebra;
    //   kHessEdotTol  the three-term Edot against pcTangentsDevice's two-term
    //                 one: equal only to the state's convergence, the third
    //                 term carrying P K (zero at an exact eigenstate,
    //                 |P K|/|K| <= 1.2e-12 on the N2 CAS(4,4) golden,
    //                 amplified by S^+), so looser than algebra;
    //   kHessAtol     the Richardson extrapolation against Eddot: what the
    //                 difference can deliver from the tangent's own 2^-12
    //                 start, Edot's round-off divided by 2h;
    //   the ratio band, as the tangent's.
    constexpr double kHessIdTol = 1e-9;
    constexpr double kHessEdotTol = 1e-8;
    constexpr double kHessAtol = 1e-8;
    constexpr double kHessRatioTol = 0.5;
    std::printf("second direction: seed=0x%llx, built the same way; negative control seed=0x%llx\n",
                static_cast<unsigned long long>(kSeedV),
                static_cast<unsigned long long>(kSeedControl));
    const nevpt2::Result<nevpt2::PcHessianCheck> hess =
        nevpt2::pcHessianFdCheck(opt.rdm, dInts, g.get("ci"), norb, g.nelecA, g.nelecB,
                                 g.get("h2e"), steps, kSeed, kSeedV, kSeedControl, eps, *res);
    if (!hess) {
      nevpt2::report(hess.error());
      return 1;
    }
    const bool ok = nevpt2::printPcHessianReport(*hess, kHessAtol, kHessIdTol, kHessEdotTol,
                                                 kHessRatioTol);
    nevpt2::printPoolHighWater("for the whole run", *res);
    return ok ? 0 : 1;
  }

  // --pc-hessian-vector (#155): the VECTOR H_cc u by reverse over forward
  // (nevpt2.pc_tangent's pcHessianVectorCheck), against --pc-hessian's
  // Eddot(u, v) by duality and a per-class central difference of the GPU's
  // own three-term y along u. u is --pc-hessian's (kSeed), and its first v
  // is too, so the duality's first row is that entry's Eddot.
  if (parsedOpts.pcHessianVector) {
    constexpr std::array<std::uint64_t, 3> kSeedsV = {0x5eed'0152u, 0x5eed'1155u, 0x5eed'2155u};
    constexpr std::uint64_t kSeedControl = 0x5eed'c155u;
    // The tolerances, each with its reason (the measured residuals are in
    // docs/pc-nevpt2.md, "The Hessian-vector product, H_cc u"):
    //   kHvIdTol    the walk's own Edot(u) / <u, y> against pcHessianDevice's,
    //               and Euler -- algebra, so rounding only;
    //   kHvDualTol  the duality and the symmetry: two computations of the
    //               same algebra that share only the per-class blocks;
    //   kHvFdTol    the per-class Richardson extrapolation of the difference
    //               of y against H_cc u, relative;
    //   the ratio band, as the other two checks'.
    constexpr double kHvIdTol = 1e-9;
    constexpr double kHvDualTol = 1e-9;
    constexpr double kHvFdTol = 1e-8;
    constexpr double kHvRatioTol = 0.5;
    std::printf("duality directions v: seeds 0x%llx, 0x%llx, 0x%llx and the state; negative "
                "control seed=0x%llx\n",
                static_cast<unsigned long long>(kSeedsV[0]),
                static_cast<unsigned long long>(kSeedsV[1]),
                static_cast<unsigned long long>(kSeedsV[2]),
                static_cast<unsigned long long>(kSeedControl));
    const nevpt2::Result<nevpt2::PcHessianVectorCheck> hv = nevpt2::pcHessianVectorCheck(
        opt.rdm, dInts, g.get("ci"), norb, g.nelecA, g.nelecB, g.get("h2e"), steps, kSeed,
        kSeedsV, kSeedControl, eps, *res);
    if (!hv) {
      nevpt2::report(hv.error());
      return 1;
    }
    const bool ok =
        nevpt2::printPcHessianVectorReport(*hv, kHvIdTol, kHvDualTol, kHvFdTol, kHvRatioTol);
    nevpt2::printPoolHighWater("for the whole run", *res);
    return ok ? 0 : 1;
  }

  const nevpt2::Result<nevpt2::PcTangentCheck> check =
      nevpt2::pcTangentFdCheck(opt.rdm, dInts, g.get("ci"), norb, g.nelecA, g.nelecB,
                               g.get("h2e"), steps, kSeed, *res);
  if (!check) {
    nevpt2::report(check.error());
    return 1;
  }
  const bool ok = nevpt2::printPcTangentReport(*check, kAtol, kIdTol, kRatioTol);
  nevpt2::printPoolHighWater("for the whole run", *res);
  return ok ? 0 : 1;
}
