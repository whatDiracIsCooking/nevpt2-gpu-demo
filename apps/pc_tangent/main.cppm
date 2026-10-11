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
module;

// stderr is a macro, which `import std` does not carry, so <cstdio> comes in
// through the global module fragment.
#include <cstdio>

export module nevpt2.app.pc_tangent;

import std;
// nevpt2.cli: --golden, --tiles and the rest of the flags the demos take.
import nevpt2.cli;
import nevpt2.golden;
import nevpt2.link_tables;
// nevpt2.pc_tangent re-exports nevpt2.energy (and so .einsum, .device_tensor,
// .wwr, .tensor), nevpt2.rdm_build and nevpt2.rdm_host_dm.
import nevpt2.pc_tangent;

int main(int argc, char** argv) {
  constexpr std::string_view kGoldenHint = "path.nevpt2gold";
  constexpr std::string_view kExtraFlags = "[--fd-steps N] [--fd-start-pow N]";
  // The finite-difference steps: successive halvings of 2^-kStartPow, as many
  // as --fd-steps asks for.
  //
  // WHY THE LARGEST STEP IS SMALL. The displaced state is not a CASCI
  // eigenstate, and its PC metric grows modes the state's own does not have.
  // The direction is chosen so that the ones scaling as h^2 never appear at
  // all (nevpt2.pc_tangent's `pcTangentDirection`), but Sir still grows one
  // at ~h^4 relative, with a Dyall eigenvalue that makes
  // lambda_k + Delta_t negative -- so the PC stage refuses the class, rightly:
  // the energy at that state does not exist and neither does a difference
  // quotient built from it. Below h ~ 2^-10 that mode falls under
  // PC_TAU = 1e-13 * s_max and is dropped like the structural ones, so every
  // class comes back with the state's own dropped count. 2^-12 is where it is
  // run, comfortably inside that and still leaving the h^2 truncation error
  // ~3e-6 Eh -- five decades above the difference's own noise floor (~1e-11
  // Eh, the RDM build's reassociation divided by 2h), so the Richardson
  // ratios measure truncation and nothing else
  // (docs/pc-nevpt2.md, "The tangent with respect to the CI vector").
  //
  // Four steps is the default; each one costs two more states -- two more RDM
  // builds and PC stages -- which at 12 orbitals is minutes, so a caller that
  // wants fewer can say so (never one: the Richardson extrapolation needs
  // two).
  constexpr int kStartPow = 12;
  constexpr int kDefaultFdSteps = 4;
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
  nevpt2::cli::CommonOptions opt;
  int fdSteps = kDefaultFdSteps;
  int startPow = kStartPow;

  for (std::size_t i = 0; i < args.size(); ++i) {
    const nevpt2::Result<bool> used = nevpt2::cli::parseCommonFlag(args, i, opt);
    if (!used) {
      nevpt2::report(used.error());
      return 1;
    }
    if (*used) continue;
    if (args[i] == "--fd-steps" && i + 1 < args.size()) {
      const nevpt2::Result<int> v = nevpt2::cli::parseInteger<int>("--fd-steps", args[++i]);
      if (!v) {
        nevpt2::report(v.error());
        return 1;
      }
      fdSteps = *v;
      continue;
    }
    if (args[i] == "--fd-start-pow" && i + 1 < args.size()) {
      const nevpt2::Result<int> v =
          nevpt2::cli::parseInteger<int>("--fd-start-pow", args[++i]);
      if (!v) {
        nevpt2::report(v.error());
        return 1;
      }
      startPow = *v;
      continue;
    }
    std::fputs(nevpt2::cli::usage(argv[0], kGoldenHint, kExtraFlags).c_str(), stderr);
    return 1;
  }
  if (const nevpt2::Status st = nevpt2::cli::finalize(opt, kGoldenHint); !st) {
    nevpt2::report(st.error());
    return 1;
  }
  // Two steps make a Richardson table; the upper bound is only there so a
  // typo cannot queue hours of RDM builds.
  if (fdSteps < 2 || fdSteps > 8) {
    const nevpt2::Status st = nevpt2::err_config("--fd-steps must be in 2..8");
    nevpt2::report(st.error());
    return 1;
  }
  // 2^-1 is already far past what the denominator condition allows; past
  // 2^-30 the difference quotient is all rounding.
  if (startPow < 1 || startPow > 30) {
    const nevpt2::Status st = nevpt2::err_config("--fd-start-pow must be in 1..30");
    nevpt2::report(st.error());
    return 1;
  }
  // The tangent RDM build runs the emitted digest or the native-fp64 BLAS
  // one; the approximate digests and the fused GEMM have no tangent path
  // (cli::finalize refuses them with --rdm-tangent for the same reason).
  if (opt.rdm.cublas || opt.rdm.ozaki || opt.rdm.fusedDigest) {
    const nevpt2::Status st = nevpt2::err_config(
        "the PC tangent runs the tangent RDM build, which needs the emitted digest or "
        "--blas-digest; not --cublas, --ozaki or --fused-digest");
    nevpt2::report(st.error());
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
