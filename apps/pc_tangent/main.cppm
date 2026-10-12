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

  std::print("=== PC-NEVPT2 CI tangent ({}/C++): CAS({},{})  ncore={}  n_det={} ===\n",
             nevpt2::kGpuBackendName, norb, g.nelecA + g.nelecB, g.ncore, ndet);
  std::printf("golden reference: %s\n", opt.goldenPath.c_str());

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
