// nevpt2.app.sc_pseudodensity:kappa_check -- --kappa-act: the ACTIVE-ACTIVE
// block of the orbital source term Y_xy = dE/dkappa_xy (Park, JCTC 2019 /
// arXiv:1907.10180, Eq. 26), checked three ways against each other.
// An interface partition that exports nothing: main.cppm is the primary
// interface, which may import only interface partitions and must re-export
// them ([module.unit]), so it does `export import :kappa_check;` -- and since
// no declaration here is `export`ed, every one keeps module linkage,
// reachable from main.cppm and from no importer.
//
// WHAT IS BEING DIFFERENTIATED, and what it is not: nevpt2.gradient's
// "The active-rotation source term" says it in full. In one line -- the SC
// total as a function of the MO INTEGRALS, every array carrying an active
// index rotated along an antisymmetric kappa, with the RDM-side inputs
// (dm1/dm2/dm3 and the f3ac/f3ca digests) and the orbital energies held
// fixed. That is the integral half of the chain rule; the two pieces it
// leaves out (the digests' own h2e dependence, which reverse mode reaches
// through the 4-RDM contraction's transpose, and the orbital energies'
// response, which needs integral blocks the golden does not carry --
// docs/gradient-theory.md, Sec. 1.3) are --kappa-full's, out of the sidecar
// (#88, #89, and the orbital energies' own work).
//
// THE THREE NUMBERS, per seeded direction:
//
//   A  <kappa, Y_act>, the ADJOINT: the pseudodensities of Eqs. 41-43 paired
//      with the direction's action on the very arrays they are conjugate to.
//   B  Edot(kappa), the forward TANGENT: the same class terms rerun under
//      the product rule, finished with Eqs. 36-37's multipliers.
//   C  a central finite difference of energiesDevice itself along
//      exp(t kappa) -- the energy stage, unmodified, at rotated integrals.
//
// A == B is the duality the assembly is held to (1e-7, the tree's usual; it
// comes out near machine precision, and it is EXACT arithmetic, not a step
// size). What A and B cannot check between them is the convention they
// share -- the device kappa action, and the claim that a Dyall block's
// tangent is its own builder called with the dotted integrals -- because a
// wrong tangent of an operand cancels between the two modes. C is what pins
// that: it rotates every stored block with nevpt2.kappa_rotation's finite
// exp(t kappa) on the host, uploads, and reruns the energy. So the finite
// difference is here for the same reason it is in
// --sc-tangent (nevpt2.ci_checks) -- one cross-check that licenses the
// exact test, with the plateau and the Richardson ratio printed.
//
// AND Y_act AS A MATRIX, where the probe count is affordable: the same
// pairing over the n_act (n_act - 1) / 2 basis directions
// (scActiveRotationMatrix). Its pairing with each seeded kappa must
// reproduce A, which is what makes the matrix -- the object the core/virtual
// work compares against -- as checked as the pairing is.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries. A malformed direction is the
// caller's input and propagates unchanged to main, which reports it once.
#include "error_handling/error_macros.h"

export module nevpt2.app.sc_pseudodensity:kappa_check;

import std;
import nevpt2.gradient;         // the assembly, the pairing and the matrix
import nevpt2.kappa_rotation;   // expKappa / rotateAxes: the finite rotation
import nevpt2.golden;

// How many seeded directions the duality is checked along. Four is "several"
// (the acceptance wording) and costs four extra sets of a class's Dyall
// blocks, held while its slabs are walked.
constexpr std::size_t kDirections = 4;
// Their seeds, fixed so the check is reproducible run to run.
constexpr std::uint64_t kKappaSeedBase = 0x5eed'0036u;

// The adjoint-against-tangent tolerance: this tree's usual 1e-7, on an
// identity that is exact in exact arithmetic (docs/testing.md, "The
// active-rotation source term", has what is measured).
constexpr double kRotTol = 1e-7;
// The amplitude energies against nevpt2.energy's own, as the identity report
// holds them.
constexpr double kEnergyTol = 1e-10;
// The matrix's pairing against the direct one: the same functional summed in
// a different order, so this is reassociation noise.
constexpr double kMatrixTol = 1e-9;
// Above this many basis directions the matrix is not assembled (it is one
// Dyall-block build per direction per class): CAS(10,10)'s 45 are in,
// CAS(12,12)'s 66 are not.
constexpr std::int64_t kMatrixMaxPairs = 64;
// Chunk: how many basis directions share one slab walk.
constexpr std::int64_t kMatrixChunk = 16;

// The finite-difference sweep, as --sc-tangent's (src/gradient/ci_checks_tangent.cpp):
// h = 2^-k, halving, powers of two so h is exact and the ratio across a
// halving is exactly the Richardson factor.
constexpr int kFirstStepExp = 6;
constexpr int kLastStepExp = 19;
constexpr std::size_t kStepCount = kLastStepExp - kFirstStepExp + 1;
// The difference's best error over the sweep, RELATIVE TO THE ENERGY SCALE --
// max(|Edot|, |E_corr|), not |Edot| alone. Two reasons, and they are the same
// reason: differencing E(+h) - E(-h) carries round-off of order eps |E| / h,
// so the floor is set by the energy's magnitude and not by the derivative's;
// and the derivative here is EXACTLY ZERO whenever every active rotation
// mixes two different irreps (N2's CAS(4,4): four active orbitals, four
// distinct irreps), where a relative-to-|Edot| criterion would divide by
// nothing. docs/testing.md, "The active-rotation source term", has the
// measured plateaus.
constexpr double kFdTol = 1e-6;
// The run length -- a RUN, not the whole table, because differencing the
// energy at a small displacement has an intrinsic hazard: a perturber that is
// symmetry-empty at the base point has N and H growing together as h^2, and
// at a step where Delta N + H cancels to round-off the term is numerically
// enormous (docs/testing.md, "The SC tangent's finite-difference
// cross-check", where it was first seen). The window is nevpt2.gradient_plan's
// kFourfoldLow / kFourfoldHigh, shared by every such check.
constexpr int kRichardsonRun = 3;

// The golden's integral blocks with every ACTIVE axis transformed by `u`,
// uploaded -- the arrays an active rotation moves, and (empty axis lists) the
// ones it does not. The layouts are nevpt2.energy's own (energy.cppm: the
// NevptIntegralsDevice fields and the ExtBlock table), the same ones
// nevpt2.gradient:rotation's activeAxesOf encodes for the slab side; a
// disagreement between the two shows up as C disagreeing with A and B.
// Everything allocated is owned by the returned struct and freed (on res's
// stream) when the caller drops it.
nevpt2::NevptIntegralsDevice uploadRotated(const nevpt2::GoldenFile& g, const nevpt2::Tensor& u,
                                           const nevpt2::DeviceResources& res) {
  const auto up = [&](const char* name, const std::vector<int>& axes) {
    const nevpt2::Tensor& host = g.get(name);
    return axes.empty() ? nevpt2::uploadTensor(host, res)
                        : nevpt2::uploadTensor(nevpt2::rotateAxes(host, axes, u), res);
  };
  nevpt2::NevptIntegralsDevice d;
  d.h1e = up("h1e", {0, 1});                   // (a, a)
  d.h2e = up("h2e", {0, 1, 2, 3});             // (a, a, a, a)
  d.e_core = up("e_core", {});                 // (c) -- held fixed: see the header
  d.e_virt = up("e_virt", {});                 // (v)
  d.h2e_v_Sr = up("h2e_v_Sr", {1, 2, 3});      // (v, a, a, a)
  d.h1e_v_Sr = up("h1e_v_Sr", {1});            // (v, a)
  d.h2e_v_Si = up("h2e_v_Si", {0, 1, 3});      // (a, a, c, a)
  d.h1e_v_Si = up("h1e_v_Si", {0});            // (a, c)
  d.cvcv = up("cvcv", {});                     // (c, v, c, v) -- no active index at all
  d.h2e_v_Sijr = up("h2e_v_Sijr", {1});        // (v, a, c, c)
  d.h2e_v_Srsi = up("h2e_v_Srsi", {3});        // (v, v, c, a)
  d.h2e_v_Srs = up("h2e_v_Srs", {2, 3});       // (v, v, a, a)
  d.h2e_v_Sij = up("h2e_v_Sij", {0, 1});       // (a, a, c, c)
  d.h2e_v1_Sir = up("h2e_v1_Sir", {1, 3});     // (v, a, c, a)
  d.h2e_v2_Sir = up("h2e_v2_Sir", {1, 2});     // (v, a, a, c)
  d.h1e_v_Sir = up("h1e_v_Sir", {});           // (v, c)
  return d;
}

// t * kappa.
nevpt2::Tensor kappaScaled(const nevpt2::Tensor& kappa, double t) {
  nevpt2::Tensor out = kappa;
  for (std::int64_t i = 0; i < out.size(); ++i) out.flatRef(i) *= t;
  return out;
}

// The SC total at the rotation exp(t kappa), the unmodified energy stage at
// rotated integrals and the SAME RDMs and digests -- no gradient code is
// involved in it, which is what makes it a reference for one.
double scTotalRotated(const nevpt2::GoldenFile& g, const nevpt2::Tensor& kappa, double t,
                      const nevpt2::DeviceTensor& dm1, const nevpt2::DeviceTensor& dm2,
                      const nevpt2::DeviceTensor& dm3, const nevpt2::DeviceTensor& f3ac,
                      const nevpt2::DeviceTensor& f3ca, const nevpt2::DeviceResources& res) {
  const nevpt2::Tensor u = nevpt2::expKappa(kappaScaled(kappa, t));
  const nevpt2::NevptIntegralsDevice ints = uploadRotated(g, u, res);
  return nevpt2::energiesDevice(ints, dm1, dm2, dm3, f3ac, f3ca, res).total;
}

// Runs the check for one golden state: prints the per-direction adjoint /
// tangent table, the matrix cross-check where it is affordable, and the
// finite-difference sweep along the first direction, and returns whether
// every criterion above holds. A malformed direction, or a matrix request
// this cannot honour, is a configuration Error (the caller reports it once).
nevpt2::Result<bool> checkKappaAct(const nevpt2::GoldenFile& g,
                                   const nevpt2::EnergyResult& energies,
                                   const nevpt2::NevptIntegralsDevice& ints,
                                   const nevpt2::DeviceTensor& dm1,
                                   const nevpt2::DeviceTensor& dm2,
                                   const nevpt2::DeviceTensor& dm3,
                                   const nevpt2::DeviceTensor& f3ac,
                                   const nevpt2::DeviceTensor& f3ca,
                                   const nevpt2::DeviceResources& res) {
  const std::int64_t norb = g.ncas;
  std::printf("\n--kappa-act: the active-active block of the orbital source term\n");
  std::printf("  kappa: %zu seeded antisymmetric directions, unit Frobenius norm, seeds "
              "0x%llx..\n",
              kDirections, static_cast<unsigned long long>(kKappaSeedBase));

  std::vector<nevpt2::Tensor> kappas;
  std::vector<nevpt2::ScRotationRequest> reqs;
  for (std::size_t d = 0; d < kDirections; ++d) {
    nevpt2::Tensor k =
        nevpt2::antisymmetricKappa(norb, kKappaSeedBase + static_cast<std::uint64_t>(d));
    // The two properties the whole construction rests on, on this very
    // direction: kappa + kappa^T is zero, and exp(kappa) is orthogonal. The
    // unit tier proves them generally (test/gradient/); this is the state's
    // own.
    const nevpt2::Tensor u = nevpt2::expKappa(k);
    std::printf("    dir %zu: max|k + k^T| = %.1e   max|U^T U - 1| = %.1e\n", d,
                nevpt2::antisymmetryResidual(k), nevpt2::orthogonalityResidual(u));
    kappas.push_back(k);
    reqs.push_back(nevpt2::ScRotationRequest{std::move(k), /*tangent=*/true});
  }

  const auto t0 = std::chrono::steady_clock::now();
  const nevpt2::ScRotationResult rot =
      NEVPT2_TRY(nevpt2::scActiveRotationDevice(ints, dm1, dm2, dm3, f3ac, f3ca, reqs, res));
  std::printf("  assembly + %zu directions: %.2fs\n", kDirections,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

  bool ok = nevpt2::printScRotationTable(rot, kRotTol);
  std::printf("  adjoint vs tangent, every direction within %.0e: %s\n", kRotTol,
              ok ? "ok" : "FAILED");

  // The class energies the same walk reports, against nevpt2.energy's own:
  // the identity entries hold these to 1e-10 too, and a rotation check on a
  // state whose energies have drifted would mean nothing.
  double worstE = 0.0;
  for (int c = 0; c < 8; ++c)
    worstE = std::max(worstE, std::fabs(rot.grad.classes[c].energy - energies.energies[c]));
  ok = ok && worstE < kEnergyTol;
  std::printf("  class energies vs energiesDevice: max |delta| = %.2e (< %.0e: %s)\n", worstE,
              kEnergyTol, worstE < kEnergyTol ? "ok" : "FAILED");

  // --- Y_act as a matrix, where the probe count is affordable ---------------
  const std::int64_t pairs = norb * (norb - 1) / 2;
  if (pairs <= kMatrixMaxPairs) {
    const auto tm = std::chrono::steady_clock::now();
    const nevpt2::Tensor y = NEVPT2_TRY(
        nevpt2::scActiveRotationMatrix(ints, dm1, dm2, dm3, f3ac, f3ca, kMatrixChunk, res));
    double worst = 0.0, ymax = 0.0;
    for (std::int64_t i = 0; i < y.size(); ++i) ymax = std::max(ymax, std::fabs(y.flat(i)));
    for (std::size_t d = 0; d < kappas.size(); ++d) {
      double pair = 0.0;
      for (std::int64_t p = 0; p < norb; ++p)
        for (std::int64_t q = 0; q < norb; ++q) pair += kappas[d](p, q) * y(p, q);
      worst = std::max(worst, std::fabs(pair - rot.dirs[d].adjoint));
    }
    ok = ok && worst < kMatrixTol;
    std::printf("\n  Y_act as a matrix: %lld basis directions in chunks of %lld, %.2fs, "
                "max|Y| = %.3e\n",
                static_cast<long long>(pairs), static_cast<long long>(kMatrixChunk),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tm).count(),
                ymax);
    std::printf("  <kappa, Y> from the matrix vs the direct pairing: max |delta| = %.2e "
                "(< %.0e: %s)\n",
                worst, kMatrixTol, worst < kMatrixTol ? "ok" : "FAILED");
  } else {
    std::printf("\n  Y_act as a matrix: skipped, %lld basis directions is above the %lld this "
                "check assembles\n",
                static_cast<long long>(pairs), static_cast<long long>(kMatrixMaxPairs));
  }

  // --- the finite difference, along the first direction ---------------------
  std::printf("\n  central differences of energiesDevice along exp(h kappa), direction 0 --\n"
              "  every stored block rotated on the host by nevpt2.kappa_rotation's exp, the\n"
              "  RDMs and the digests untouched:\n");
  std::printf("  %5s %14s %20s %12s %12s %8s\n", "k", "h = 2^-k", "(E(+h)-E(-h))/2h", "|err|",
              "rel err", "ratio");
  const double edot = rot.dirs[0].tangent;
  // The scale the difference's error is judged against: see kFdTol.
  const double scale = std::max(std::fabs(edot), std::fabs(energies.total));
  std::array<double, kStepCount> step{};
  std::array<double, kStepCount> err{};
  for (std::size_t i = 0; i < kStepCount; ++i) {
    const int k = kFirstStepExp + static_cast<int>(i);
    const double h = std::ldexp(1.0, -k);
    step[i] = h;
    const double plus = scTotalRotated(g, kappas[0], h, dm1, dm2, dm3, f3ac, f3ca, res);
    const double minus = scTotalRotated(g, kappas[0], -h, dm1, dm2, dm3, f3ac, f3ca, res);
    const double d = (plus - minus) / (2.0 * h);
    err[i] = std::fabs(d - edot);
    const double rel = err[i] / std::max(1e-30, scale);
    if (i == 0) {
      std::printf("  %5d %14.3e %20.12f %12.3e %12.3e %8s\n", k, h, d, err[i], rel, "-");
    } else {
      std::printf("  %5d %14.3e %20.12f %12.3e %12.3e %8.2f\n", k, h, d, err[i], rel,
                  err[i] > 0.0 ? err[i - 1] / err[i] : 0.0);
    }
  }

  std::size_t bestAt = 0;
  for (std::size_t i = 1; i < kStepCount; ++i)
    if (err[i] < err[bestAt]) bestAt = i;
  const double bestRel = err[bestAt] / std::max(1e-30, scale);
  ok = ok && bestRel < kFdTol;
  std::printf("\n  Edot(kappa_0) = %.14e, <kappa_0, Y_act> = %.14e\n", edot,
              rot.dirs[0].adjoint);
  std::printf("  the error scale (max of |Edot| and |E_corr| = %.6f): %.3e\n",
              std::fabs(energies.total), scale);
  std::printf("  plateau: |err| = %.3e (relative %.3e) at h = %.3e (2^-%d); round-off takes "
              "over below it\n",
              err[bestAt], bestRel, step[bestAt], kFirstStepExp + static_cast<int>(bestAt));

  const nevpt2::FourfoldRun run = nevpt2::longestFourfoldRun(err);
  const bool richardson = run.length >= kRichardsonRun;
  ok = ok && richardson;
  std::printf("  Richardson: %d consecutive halvings cut the error fourfold (ratio in [%.1f, "
              "%.1f]), ending at h = %.3e -- %s, %d needed\n",
              run.length, nevpt2::kFourfoldLow, nevpt2::kFourfoldHigh, step[run.end],
              richardson ? "ok" : "NOT ENOUGH", kRichardsonRun);

  std::printf("\n%s: the active-rotation source term's adjoint and tangent agree to %.0e for "
              "every direction, and the finite difference to %.0e of the energy scale\n",
              ok ? "PASS" : "FAIL", kRotTol, kFdTol);
  return ok;
}
