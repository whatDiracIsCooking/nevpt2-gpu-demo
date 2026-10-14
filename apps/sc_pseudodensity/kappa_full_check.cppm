// nevpt2.app.sc_pseudodensity:kappa_full_check -- --kappa-full: the FULL
// orbital source term Y_xy = dE/dkappa_xy (Park, JCTC 2019 /
// arXiv:1907.10180, Eqs. 26 and 28) over the whole kappa space -- core-core,
// virtual-virtual and mixed as well as active-active -- read out of the
// gradient sidecar beside the golden (docs/reference-data.md, "The gradient
// sidecar") and checked five ways.
// An interface partition that exports nothing, as :kappa_check is.
//
// WHAT IS BEING DIFFERENTIATED. The SC total as a function of every MO
// integral array the energy stage reads AND of the orbital-energy vectors
// `e_core` / `e_virt`, with the RDM side (dm1/dm2/dm3) fixed -- the
// derivative's own definition. The integral halves the duality compares hold
// the `f3ac`/`f3ca` digests fixed as well (nevpt2.gradient, "The full source
// term Y_xy"); the digests' own response is a term of its own in each mode --
// forward, the rebuilt digests (#88, scDigestRotationTangentDevice), reverse,
// their transpose (#89, scDigestAdjointDevice) -- and with it the FULL
// duality, the exact <kappa, Y> against the exact Edot, is gated beside the
// like-for-like one. The orbital energies were the active-block work's other
// gap, and they move here: that is what the sidecar's generalized Fock is for.
//
// THE SIX CHECKS, in the order they print:
//
//   1  THE SIDECAR AGAINST ITSELF AND THE GOLDEN (scGeneralIntegralCheck).
//      Every promotion rule's own source array, restricted to the promoted
//      slot's own space, must be the stored block under exactly the
//      permutation that rule uses; h^eff must come back out of `h` and the
//      dressing rules' sources; the generalized Fock's core and virtual
//      diagonals must come back out of `heff` and the orbital-energy rules'
//      tied arrays, and must BE the golden's e_core / e_virt ((O2)). This is
//      what pins the promotion table, which neither mode of check 2 can see:
//      both read the same tangents, so a wrong one cancels between them.
//   2  THE DUALITY, on four seeded FULL antisymmetric directions:
//      <kappa, Y>'s integral half against the forward tangent's. Exact
//      arithmetic, no step size, gated at this tree's 1e-7. Then the FULL
//      duality (#89), gated at the same 1e-7: the exact <kappa, Y> --
//      integral half, orbital energies and the digests' transpose -- against
//      the exact Edot, the frozen-digest tangent plus the rebuilt digests'
//      term. The forward digests' term has an INDEPENDENT provider too,
//      PySCF's exact directional derivative, which no binary can reach:
//      `--dump-source-terms` writes the directions and both terms, and
//      reference_data/nevpt2_grad.py --edot-check holds them against it
//      (docs/testing.md, "The digests' orbital response, forward mode").
//   3  THE ACTIVE BLOCK, BIT FOR BIT AS FAR AS THE KERNELS GO. The same
//      seeded active kappa run through scActiveRotationDevice (the
//      active-block entry point) and, embedded in a full (n_mo, n_mo) kappa,
//      through scFullRotationDevice: the integral halves must agree. By
//      construction they agree EXACTLY -- the general part of every slot's
//      sum is an exactly-zero tensor there, and adding zero to a double
//      changes no bit -- but the two numbers come out of two separate runs of
//      kernels that accumulate with atomicAdd, whose order the hardware
//      picks, so exact equality of two such runs is not a property this tree
//      has (and the convention in CLAUDE.md says as much). So the active
//      entry point is ALSO run twice against itself, and the comparison is
//      held to that same-run reproducibility: the construction adds no error
//      above the noise the kernels already have. That noise is itself held
//      to a recorded envelope (#64), so it cannot grow unnoticed.
//   4  TWO FINITE DIFFERENCES, and they are the only ones available for the
//      new blocks. A rotation INSIDE the core space, or inside the virtual
//      space, keeps every index in its own space, so the displaced integrals
//      are the stored blocks with those axes transformed on the host -- and
//      the displaced orbital energies are the rotated generalized Fock's
//      diagonal, which the sidecar carries. So the energy stage itself can be
//      rerun at exp(h kappa) and differenced, exactly as the active block's
//      check does, which pins the core and virtual promotion rules to
//      nevpt2.kappa_rotation's exponential map. A MIXED rotation cannot be
//      differenced this way (its indices leave their spaces, which needs
//      integrals the sidecar does not carry in full), and does not need to
//      be: the rule is ONE einsum over the whole general index, so the rows
//      checks 3 and 4 exercise are the same code the mixed rows run.
//      The same two directions also assert two derived exactnesses: an
//      intra-space rotation moves NO orbital energy (f is diagonal there),
//      so the `eps` term must vanish, and it moves no active integral, so
//      h2e_dot and with it the digests' term are EXACTLY zero, forward (#88)
//      and reverse (#89).
//   5  PARK EQS. 56-57, the pseudocanonical multipliers, off the assembled
//      Y: one division per core and per virtual pair. Those two blocks of Y
//      are exact (neither the digests nor the orbital energies move under an
//      intra-space rotation), and a degenerate pair -- which an abelian
//      subgroup leaves in different irreps, so its numerator vanishes too --
//      is accepted as zero or refused, never divided through
//      (scPseudocanonicalMultipliers). The matrix those divisions read is
//      itself held against the direct pairing on an inactive-restricted
//      direction first, so it is as checked as the pairing is.
//   6  THE DIGESTS' TERM AGAINST ITS OWN FINITE DIFFERENCE (#88), on check
//      3's active-restricted direction: the energy stage rerun with every
//      integral, RDM and orbital energy the state's own and ONLY the digests
//      rebuilt -- buildRdmsDevice at h2e rotated on the host by exp(h kappa)
//      -- differenced and held to the digests' term, Richardson-verified as
//      check 4 is. An active rotation's h2e_dot is the intra-active action
//      alone; the sidecar's general part of it, which a mixed rotation adds,
//      is the promotion table check 1 pins, and the whole term is held to
//      PySCF by the offline provider gate above. Beside it, the same term in
//      reverse mode (#89) along the same direction, held to the forward one
//      at 1e-7 -- exact arithmetic, as the full duality is.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries. A malformed direction, a
// sidecar that does not fit the golden, or an undetermined multiplier is the
// caller's input and propagates unchanged to main, which reports it once.
#include "error_handling/error_macros.h"

export module nevpt2.app.sc_pseudodensity:kappa_full_check;

import std;
import nevpt2.gradient;        // the full source term, and Eqs. 56-57
import nevpt2.kappa_rotation;  // expKappa / rotateAxes: the finite rotations
import nevpt2.golden;

// How many seeded full-space directions the duality is checked along. Four,
// as --kappa-act's, and the seed base differs so the two checks cannot share
// a direction by accident.
constexpr std::size_t kFullDirections = 4;
constexpr std::uint64_t kFullSeedBase = 0x5eed'0038u;

// The duality's tolerance: this tree's usual 1e-7, on an identity that is
// exact in exact arithmetic (docs/testing.md, "The full orbital source term",
// has what is measured).
constexpr double kFullRotTol = 1e-7;
// The sidecar self-checks, as relative Frobenius norms. They compare two
// PySCF transforms of the same integrals, so what is left is reassociation.
constexpr double kSelfCheckTol = 1e-10;
// The orbital-energy response of an intra-space rotation, which is zero by
// (O2) -- and inherits (O2)'s own residual, the canonicalisation's 1e-13
// rather than machine precision (docs/reference-data.md, "The gradient
// sidecar"), so it is held four decades above that.
constexpr double kEpsZeroTol = 1e-9;
// The class energies against nevpt2.energy's own, as the identity report
// holds them.
constexpr double kFullEnergyTol = 1e-10;
// What "bit for bit" can mean on this device. The einsum kernel accumulates
// with atomicAdd, whose order within a launch the hardware picks, so two
// identical runs of the SAME entry point need not agree to the last bit --
// and measurably do not (docs/testing.md, "The full orbital source term").
// So check 3 holds the two paths to the larger of what the active path
// reproduces from ITSELF in the same run and this relative floor; the rerun
// gap alone would be a coin flip between two draws of the same noise, and
// anything a wrong construction could do is many decades above either.
constexpr double kBitFloorRel = 1e-11;
// And that same-run spread is itself held to a RECORDED envelope (#64): the
// floor above follows whatever spread the run measures, so on its own it
// would widen silently if a kernel change made the nondeterminism worse. The
// envelope -- absolute, Eh, on both the pairing and the tangent -- is chosen
// from a measurement, so it lives in docs/testing.md alone ("The kernels'
// same-run self-variation" has the number, the card, the flags, the run
// count and how it was chosen); this binary's CMakeLists.txt reads it from
// that section's table at configure time.
#ifndef NEVPT2_SELF_VARIATION_ENVELOPE
#error "NEVPT2_SELF_VARIATION_ENVELOPE must be defined by apps/sc_pseudodensity/CMakeLists.txt"
#endif
constexpr double kSelfVariationEnvelope = NEVPT2_SELF_VARIATION_ENVELOPE;

// The finite-difference sweep, as :kappa_check's: h = 2^-k, halving, powers
// of two so h is exact and the ratio across a halving is the Richardson
// factor.
constexpr int kFdFirstExp = 6;
constexpr int kFdLastExp = 18;
constexpr std::size_t kFdSteps = kFdLastExp - kFdFirstExp + 1;
// The best error over the sweep, relative to max(|Edot|, |E_corr|) -- the
// scale :kappa_check's header argues for, and for the same reasons.
constexpr double kFullFdTol = 1e-6;
// The run length; the window is nevpt2.gradient_plan's kFourfoldLow /
// kFourfoldHigh, shared by every finite-difference check of the gradient path.
constexpr int kFdRichardsonRun = 3;

// How many basis directions share one slab walk when Y's inactive blocks are
// probed for Eqs. 56-57.
constexpr std::int64_t kYChunk = 16;
// The probed matrix's pairing against the direct one: the same functional
// summed in a different order, so this is reassociation noise (:kappa_check's
// own matrix cross-check uses the same number).
constexpr double kFullMatrixTol = 1e-9;
// Below this gap (Eh) a core or virtual pair counts as degenerate and its
// multiplier is not divided out. Symmetry-degenerate partners split by ~1e-9
// (the golden generator's measured Fock noise) and the smallest real
// same-irrep gap it will write is 1e-4 (docs/reference-data.md, "The
// degeneracy refusal"), so 1e-6 sits between them with decades to spare.
constexpr double kPcGapFloor = 1e-6;
// What a degenerate pair's numerator has to be under: it is zero by symmetry
// (the two partners are in different irreps), and the pairing's own noise is
// ~1e-15 on an |Y| of order 1e-1.
constexpr double kPcZeroTol = 1e-10;

// The sidecar's arrays the source term reads -- `require`d before any of them
// is get()'d, so a sidecar from an older generator is an IO Error naming the
// missing array rather than an abort.
nevpt2::Status requireGradientArrays(const nevpt2::GradGoldFile& gg) {
  return gg.require({"h", "heff", "f", "g_xa_aa", "g_xv_aa", "g_xc_aa", "g_xa_va", "g_xv_va",
                     "g_xc_va", "g_xa_ca", "g_xv_ca", "g_xc_ca", "g_xa_cv", "g_xv_cv",
                     "g_xc_cv", "g_xa_cc", "g_xv_cc", "g_xc_cc_j", "g_xc_cc_k", "g_xa_vv",
                     "g_xc_vv"});
}

// The sidecar, uploaded. Everything it allocates is owned by the returned
// struct and freed (on res's stream) when the caller drops it.
nevpt2::GeneralIntegralsDevice uploadGeneral(const nevpt2::GradGoldFile& gg,
                                             const nevpt2::DeviceResources& res) {
  const auto up = [&](const char* name) { return nevpt2::uploadTensor(gg.get(name), res); };
  nevpt2::GeneralIntegralsDevice d;
  d.ncore = gg.ncore;
  d.nact = gg.nact;
  d.nvirt = gg.nvirt;
  d.h = up("h");
  d.heff = up("heff");
  d.f = up("f");
  d.g_xa_aa = up("g_xa_aa");
  d.g_xv_aa = up("g_xv_aa");
  d.g_xc_aa = up("g_xc_aa");
  d.g_xa_va = up("g_xa_va");
  d.g_xv_va = up("g_xv_va");
  d.g_xc_va = up("g_xc_va");
  d.g_xa_ca = up("g_xa_ca");
  d.g_xv_ca = up("g_xv_ca");
  d.g_xc_ca = up("g_xc_ca");
  d.g_xa_cv = up("g_xa_cv");
  d.g_xv_cv = up("g_xv_cv");
  d.g_xc_cv = up("g_xc_cv");
  d.g_xa_cc = up("g_xa_cc");
  d.g_xv_cc = up("g_xv_cc");
  d.g_xc_cc_j = up("g_xc_cc_j");
  d.g_xc_cc_k = up("g_xc_cc_k");
  d.g_xa_vv = up("g_xa_vv");
  d.g_xc_vv = up("g_xc_vv");
  return d;
}

// A full (n_mo, n_mo) kappa with `k` written into one diagonal block --
// [lo, lo + k.dim(0)) -- and zero everywhere else. How an active-, core- or
// virtual-restricted direction is handed to the full-space entry point, and
// (for the active one) what makes check 3 a comparison of the same numbers.
nevpt2::Tensor embedBlock(const nevpt2::Tensor& k, std::int64_t lo, std::int64_t nmo) {
  nevpt2::Tensor out({nmo, nmo});
  for (std::int64_t p = 0; p < k.dim(0); ++p)
    for (std::int64_t q = 0; q < k.dim(1); ++q) out(lo + p, lo + q) = k(p, q);
  return out;
}

// t * kappa.
nevpt2::Tensor scaled(const nevpt2::Tensor& kappa, double t) {
  nevpt2::Tensor out = kappa;
  for (std::int64_t i = 0; i < out.size(); ++i) out.flatRef(i) *= t;
  return out;
}

// One space's diagonal block of a full-MO-range matrix, as a host tensor.
nevpt2::Tensor diagonalBlock(const nevpt2::Tensor& m, std::int64_t lo, std::int64_t n) {
  nevpt2::Tensor out({n, n});
  for (std::int64_t p = 0; p < n; ++p)
    for (std::int64_t q = 0; q < n; ++q) out(p, q) = m(lo + p, lo + q);
  return out;
}

// The golden's blocks with every axis over ONE INACTIVE SPACE transformed by
// `u`, uploaded, and that space's orbital energies replaced by the rotated
// generalized Fock's diagonal -- the whole displaced input of check 4.
//
// The axis lists are read off nevpt2.energy's layouts (energy.cppm's
// NevptIntegralsDevice fields and the ExtBlock table), the same ones
// nevpt2.gradient:general's table encodes for the promotion rules; a
// disagreement between the two shows up as the finite difference disagreeing
// with both modes, which is what it is here for.
//
// Nothing an ACTIVE index touches moves: h1e and h2e are untouched (so every
// Dyall block, and the f3 digests, are the state's own), and h^eff is
// untouched as a matrix because an intra-space rotation leaves the core
// density -- and the virtual space carries none -- so each one-electron block
// only has its own inactive axis transformed.
nevpt2::NevptIntegralsDevice uploadSpaceRotated(const nevpt2::GoldenFile& g,
                                                const nevpt2::GradGoldFile& gg, bool core,
                                                const nevpt2::Tensor& u,
                                                const nevpt2::DeviceResources& res) {
  const auto up = [&](const char* name, const std::vector<int>& coreAxes,
                      const std::vector<int>& virtAxes) {
    const std::vector<int>& axes = core ? coreAxes : virtAxes;
    const nevpt2::Tensor& host = g.get(name);
    return axes.empty() ? nevpt2::uploadTensor(host, res)
                        : nevpt2::uploadTensor(nevpt2::rotateAxes(host, axes, u), res);
  };
  nevpt2::NevptIntegralsDevice d;
  d.h1e = up("h1e", {}, {});                          // (a, a)
  d.h2e = up("h2e", {}, {});                          // (a, a, a, a)
  d.h2e_v_Sr = up("h2e_v_Sr", {}, {0});               // (v, a, a, a)
  d.h1e_v_Sr = up("h1e_v_Sr", {}, {0});               // (v, a)
  d.h2e_v_Si = up("h2e_v_Si", {2}, {});               // (a, a, c, a)
  d.h1e_v_Si = up("h1e_v_Si", {1}, {});               // (a, c)
  d.cvcv = up("cvcv", {0, 2}, {1, 3});                // (c, v, c, v)
  d.h2e_v_Sijr = up("h2e_v_Sijr", {2, 3}, {0});       // (v, a, c, c)
  d.h2e_v_Srsi = up("h2e_v_Srsi", {2}, {0, 1});       // (v, v, c, a)
  d.h2e_v_Srs = up("h2e_v_Srs", {}, {0, 1});          // (v, v, a, a)
  d.h2e_v_Sij = up("h2e_v_Sij", {2, 3}, {});          // (a, a, c, c)
  d.h2e_v1_Sir = up("h2e_v1_Sir", {2}, {0});          // (v, a, c, a)
  d.h2e_v2_Sir = up("h2e_v2_Sir", {3}, {0});          // (v, a, a, c)
  d.h1e_v_Sir = up("h1e_v_Sir", {1}, {0});            // (v, c)
  // The orbital energies ARE the generalized Fock's diagonal, so at the
  // displaced orbitals they are the rotated block's: diag(U^T f U). The other
  // space's are untouched. (At h = 0 this reproduces e_core / e_virt exactly,
  // because f is diagonal there -- which the self-checks assert.)
  const nevpt2::Tensor& f = gg.get("f");
  const std::int64_t lo = core ? 0 : gg.ncore + gg.nact;
  const std::int64_t n = core ? gg.ncore : gg.nvirt;
  const nevpt2::Tensor rotated = nevpt2::rotateAxes(diagonalBlock(f, lo, n), {0, 1}, u);
  nevpt2::Tensor eps({n});
  for (std::int64_t p = 0; p < n; ++p) eps(p) = rotated(p, p);
  d.e_core = core ? nevpt2::uploadTensor(eps, res) : nevpt2::uploadTensor(g.get("e_core"), res);
  d.e_virt = core ? nevpt2::uploadTensor(g.get("e_virt"), res) : nevpt2::uploadTensor(eps, res);
  return d;
}

// The SC total at the intra-space rotation exp(t kappa): the unmodified
// energy stage at displaced integrals and displaced orbital energies, with
// the same RDMs and digests -- no gradient code in it, which is what makes it
// a reference for one.
double scTotalSpaceRotated(const nevpt2::GoldenFile& g, const nevpt2::GradGoldFile& gg,
                           bool core, const nevpt2::Tensor& kappaBlock, double t,
                           const nevpt2::DeviceTensor& dm1, const nevpt2::DeviceTensor& dm2,
                           const nevpt2::DeviceTensor& dm3, const nevpt2::DeviceTensor& f3ac,
                           const nevpt2::DeviceTensor& f3ca,
                           const nevpt2::DeviceResources& res) {
  const nevpt2::Tensor u = nevpt2::expKappa(scaled(kappaBlock, t));
  const nevpt2::NevptIntegralsDevice ints = uploadSpaceRotated(g, gg, core, u, res);
  return nevpt2::energiesDevice(ints, dm1, dm2, dm3, f3ac, f3ca, res).total;
}

// One finite-difference sweep of `energyAt` (the SC total at exp(t kappa),
// however the caller displaces it) against `edot`, printed: the plateau and
// the Richardson run, as --kappa-act's and --sc-tangent's. Returns whether
// both criteria hold. The caller prints what is being displaced first.
bool sweepAgainst(double edot, double scale, const std::function<double(double)>& energyAt) {
  std::printf("  %5s %14s %20s %12s %12s %8s\n", "k", "h = 2^-k", "(E(+h)-E(-h))/2h", "|err|",
              "rel err", "ratio");
  std::array<double, kFdSteps> step{};
  std::array<double, kFdSteps> err{};
  for (std::size_t i = 0; i < kFdSteps; ++i) {
    const int k = kFdFirstExp + static_cast<int>(i);
    const double h = std::ldexp(1.0, -k);
    step[i] = h;
    const double plus = energyAt(h);
    const double minus = energyAt(-h);
    const double d = (plus - minus) / (2.0 * h);
    err[i] = std::fabs(d - edot);
    const double rel = err[i] / std::max(1e-30, scale);
    if (i == 0)
      std::printf("  %5d %14.3e %20.12f %12.3e %12.3e %8s\n", k, h, d, err[i], rel, "-");
    else
      std::printf("  %5d %14.3e %20.12f %12.3e %12.3e %8.2f\n", k, h, d, err[i], rel,
                  err[i] > 0.0 ? err[i - 1] / err[i] : 0.0);
  }
  std::size_t bestAt = 0;
  for (std::size_t i = 1; i < kFdSteps; ++i)
    if (err[i] < err[bestAt]) bestAt = i;
  const double bestRel = err[bestAt] / std::max(1e-30, scale);
  const nevpt2::FourfoldRun run = nevpt2::longestFourfoldRun(err);
  const bool ok = bestRel < kFullFdTol && run.length >= kFdRichardsonRun;
  std::printf("  Edot(kappa) = %.14e; plateau |err| = %.3e (relative %.3e) at h = %.3e\n", edot,
              err[bestAt], bestRel, step[bestAt]);
  std::printf("  Richardson: %d consecutive fourfold halvings (ratio in [%.1f, %.1f]) ending "
              "at h = %.3e -- %s, %d needed\n",
              run.length, nevpt2::kFourfoldLow, nevpt2::kFourfoldHigh, step[run.end],
              ok ? "ok" : "NOT ENOUGH", kFdRichardsonRun);
  return ok;
}

// One intra-space direction's finite-difference sweep (check 4).
bool finiteDifferenceSweep(const char* name, const nevpt2::GoldenFile& g,
                           const nevpt2::GradGoldFile& gg, bool core,
                           const nevpt2::Tensor& kappaBlock, double edot, double scale,
                           const nevpt2::DeviceTensor& dm1, const nevpt2::DeviceTensor& dm2,
                           const nevpt2::DeviceTensor& dm3, const nevpt2::DeviceTensor& f3ac,
                           const nevpt2::DeviceTensor& f3ca,
                           const nevpt2::DeviceResources& res) {
  std::printf("\n  central differences of energiesDevice along exp(h kappa), %s-internal --\n"
              "  every stored block's %s axes rotated on the host and that space's orbital\n"
              "  energies taken from the rotated generalized Fock's diagonal:\n",
              name, name);
  return sweepAgainst(edot, scale, [&](double t) {
    return scTotalSpaceRotated(g, gg, core, kappaBlock, t, dm1, dm2, dm3, f3ac, f3ca, res);
  });
}

// The SC total with ONLY the digests displaced (check 6): every integral, RDM
// and orbital energy the state's own, and f3ac/f3ca rebuilt from h2e rotated
// on the host by exp(t kappaAct) -- so its derivative at t = 0 is exactly the
// digests' term along the embedded active direction, and nothing else.
double scTotalDigestsRotated(const nevpt2::GoldenFile& g, const nevpt2::Tensor& kappaAct,
                             double t, const nevpt2::NevptIntegralsDevice& ints,
                             const nevpt2::DeviceTensor& dm1, const nevpt2::DeviceTensor& dm2,
                             const nevpt2::DeviceTensor& dm3,
                             const nevpt2::RdmBuildOptions& rdmOpt,
                             const nevpt2::DeviceResources& res) {
  const nevpt2::Tensor h2e =
      nevpt2::rotateAxes(g.get("h2e"), {0, 1, 2, 3}, nevpt2::expKappa(scaled(kappaAct, t)));
  const nevpt2::RdmBuildResult b =
      nevpt2::buildRdmsDevice(rdmOpt, g.get("ci"), g.ncas, g.nelecA, g.nelecB, h2e, res);
  return nevpt2::energiesDevice(ints, dm1, dm2, dm3, b.f3ac, b.f3ca, res).total;
}

// A seeded antisymmetric (n_mo, n_mo) kappa with only the (a, b) and (b, a)
// blocks of the two index ranges kept, renormalised to unit Frobenius norm --
// how a direction that is purely active-virtual, or purely core-active, is
// drawn for the provider gate (#88).
nevpt2::Tensor maskedKappa(std::int64_t nmo, std::uint64_t seed, std::int64_t aLo,
                           std::int64_t aHi, std::int64_t bLo, std::int64_t bHi) {
  const nevpt2::Tensor full = nevpt2::antisymmetricKappa(nmo, seed);
  nevpt2::Tensor out({nmo, nmo});
  double norm2 = 0.0;
  for (std::int64_t p = aLo; p < aHi; ++p)
    for (std::int64_t q = bLo; q < bHi; ++q) {
      out(p, q) = full(p, q);
      out(q, p) = -full(p, q);
      norm2 += 2.0 * full(p, q) * full(p, q);
    }
  if (norm2 > 0.0)
    for (std::int64_t i = 0; i < out.size(); ++i) out.flatRef(i) /= std::sqrt(norm2);
  return out;
}

// The directions the forward Edot(kappa) is exported along (#88) and held to
// PySCF's exact directional derivative offline: --kappa-full's own four
// seeded FULL directions -- every block, mixed ones included -- and one purely
// active-virtual and one purely core-active, so the two mixed blocks the
// digests move under are each probed on their own. A partition with no core
// (or no virtual) orbital has no such block and skips that direction.
std::vector<nevpt2::Tensor> edotDirections(std::int64_t nc, std::int64_t na, std::int64_t nv) {
  const std::int64_t nmo = nc + na + nv;
  std::vector<nevpt2::Tensor> out;
  for (std::size_t d = 0; d < kFullDirections; ++d)
    out.push_back(
        nevpt2::antisymmetricKappa(nmo, kFullSeedBase + static_cast<std::uint64_t>(d)));
  if (nv > 0) out.push_back(maskedKappa(nmo, kFullSeedBase + 0x400u, nc, nc + na, nc + na, nmo));
  if (nc > 0) out.push_back(maskedKappa(nmo, kFullSeedBase + 0x500u, 0, nc, nc, nc + na));
  return out;
}

// Runs the whole --kappa-full check for one golden state and its sidecar, and
// returns whether every criterion holds. Anything malformed in either file,
// or a pseudocanonical multiplier the state does not determine, is an Error
// the caller reports once.
nevpt2::Result<bool> checkKappaFull(const nevpt2::GoldenFile& g, const nevpt2::GradGoldFile& gg,
                                    const nevpt2::EnergyResult& energies,
                                    const nevpt2::NevptIntegralsDevice& ints,
                                    const nevpt2::DeviceTensor& dm1,
                                    const nevpt2::DeviceTensor& dm2,
                                    const nevpt2::DeviceTensor& dm3,
                                    const nevpt2::DeviceTensor& f3ac,
                                    const nevpt2::DeviceTensor& f3ca,
                                    const nevpt2::RdmBuildOptions& rdmOpt,
                                    const nevpt2::DeviceResources& res) {
  // The sidecar is found beside the golden by name, so the first thing to
  // settle is that the two describe the same state -- and then that it
  // carries every array the source term reads.
  NEVPT2_TRY(gg.checkAgainst(g));
  NEVPT2_TRY(requireGradientArrays(gg));
  const std::int64_t nc = gg.ncore, na = gg.nact, nv = gg.nvirt, nmo = gg.nmo;
  std::printf("\n--kappa-full: the full orbital source term Y_xy (Park Eqs. 26, 28)\n");
  std::printf("  sidecar partition: n_mo = %lld = %lld core + %lld active + %lld virtual "
              "(n_ao = %lld, n_elec(RHF) = %lld)\n",
              static_cast<long long>(nmo), static_cast<long long>(nc),
              static_cast<long long>(na), static_cast<long long>(nv),
              static_cast<long long>(gg.nao), static_cast<long long>(gg.nelecRhf));
  const nevpt2::GeneralIntegralsDevice gen = uploadGeneral(gg, res);

  // --- 1. the sidecar against itself and the golden -------------------------
  const nevpt2::ScGeneralCheck self =
      NEVPT2_TRY(nevpt2::scGeneralIntegralCheck(ints, gen, dm1, res));
  const double worstSelf =
      std::max(std::max(self.blocks, self.heff), std::max(self.fock, self.eps));
  bool ok = worstSelf < kSelfCheckTol;
  std::printf("\n  the promotion table against the arrays it reads (relative, worst of each "
              "family):\n");
  std::printf("    every rule's source restricted to its own space, vs the stored block: "
              "%.2e  (%s)\n",
              self.blocks, self.worstBlock.c_str());
  std::printf("    h^eff rebuilt from h + the dressing rules' sources:                   "
              "%.2e\n",
              self.heff);
  std::printf("    the generalized Fock's core/virtual diagonal, rebuilt from its rules:  "
              "%.2e\n",
              self.fock);
  std::printf("    that diagonal against the golden's e_core / e_virt ((O2)):            "
              "%.2e\n",
              self.eps);
  std::printf("  worst %.2e (< %.0e: %s)\n", worstSelf, kSelfCheckTol, ok ? "ok" : "FAILED");

  // --- 2. the duality on four seeded full directions ------------------------
  std::vector<nevpt2::Tensor> kappas;
  std::vector<nevpt2::ScRotationRequest> reqs;
  for (std::size_t d = 0; d < kFullDirections; ++d) {
    nevpt2::Tensor k =
        nevpt2::antisymmetricKappa(nmo, kFullSeedBase + static_cast<std::uint64_t>(d));
    std::printf("    dir %zu: max|k + k^T| = %.1e   max|U^T U - 1| = %.1e\n", d,
                nevpt2::antisymmetryResidual(k), nevpt2::orthogonalityResidual(
                                                     nevpt2::expKappa(k)));
    kappas.push_back(k);
    reqs.push_back(nevpt2::ScRotationRequest{std::move(k), /*tangent=*/true});
  }
  // The digests' transpose (#89): hbar = dE/d(h2e) through f3ac/f3ca, once,
  // and with it every pairing below is the EXACT <kappa, Y> at fixed CI.
  const auto tAdj = std::chrono::steady_clock::now();
  const nevpt2::DeviceTensor hbar = nevpt2::scDigestAdjointDevice(
      rdmOpt, g.get("ci"), g.nelecA, g.nelecB, ints, dm1, dm2, dm3, f3ac, f3ca, res);
  std::printf("  the digests' transpose hbar = dE/dh2e through f3ac/f3ca: %.2fs\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - tAdj).count());
  const auto t0 = std::chrono::steady_clock::now();
  const nevpt2::ScFullRotationResult rot = NEVPT2_TRY(
      nevpt2::scFullRotationDevice(ints, gen, dm1, dm2, dm3, f3ac, f3ca, &hbar, reqs, res));
  std::printf("  assembly + %zu full directions: %.2fs\n", kFullDirections,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  const bool dual = nevpt2::printScFullRotationTable(rot, kFullRotTol);
  ok = ok && dual;
  std::printf("  adjoint vs tangent (integral halves), every direction within %.0e: %s\n",
              kFullRotTol, dual ? "ok" : "FAILED");

  // The digests' own term along the same four directions, forward (#88) and
  // reverse (#89), and with them the FULL duality: <kappa, Y> -- integral
  // half, orbital energies and the transpose's term -- against the exact
  // Edot, frozen-digest tangent plus the rebuilt digests' term. Exact
  // arithmetic again, gated at the same 1e-7. The forward term's independent
  // gate is PySCF's exact directional derivative, offline (--dump-source-terms,
  // then nevpt2_grad.py --edot-check); check 6 below is its on-card one.
  const auto tDig = std::chrono::steady_clock::now();
  const std::vector<nevpt2::ScDigestTangent> digest =
      NEVPT2_TRY(nevpt2::scDigestRotationTangentDevice(rdmOpt, g.get("ci"), g.nelecA, g.nelecB,
                                                       ints, gen, dm1, dm2, dm3, f3ac, f3ca,
                                                       reqs, res));
  const bool fullDual = nevpt2::printScDigestTangentTable(rot, digest, kFullRotTol);
  ok = ok && fullDual;
  std::printf("  the digests' term along %zu full directions: %.2fs (forward gated offline "
              "against PySCF's exact Edot,\n  and on the card by check 6)\n",
              kFullDirections,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - tDig).count());
  std::printf("  FULL duality, <kappa, Y> (with the digests' transpose) vs exact Edot, every "
              "direction within %.0e: %s\n",
              kFullRotTol, fullDual ? "ok" : "FAILED");

  double worstE = 0.0;
  for (int c = 0; c < 8; ++c)
    worstE = std::max(worstE, std::fabs(rot.grad.classes[c].energy - energies.energies[c]));
  ok = ok && worstE < kFullEnergyTol;
  std::printf("  class energies vs energiesDevice: max |delta| = %.2e (< %.0e: %s)\n", worstE,
              kFullEnergyTol, worstE < kFullEnergyTol ? "ok" : "FAILED");

  // --- 3. bit-for-bit against the active block ------------------------------
  const nevpt2::Tensor kAct = nevpt2::antisymmetricKappa(na, kFullSeedBase + 0x100u);
  const std::vector<nevpt2::ScRotationRequest> actReq{
      nevpt2::ScRotationRequest{kAct, /*tangent=*/true}};
  const nevpt2::ScRotationResult actOnly = NEVPT2_TRY(
      nevpt2::scActiveRotationDevice(ints, dm1, dm2, dm3, f3ac, f3ca, actReq, res));
  // The SAME request a second time, through the same entry point: the einsum
  // kernel accumulates with atomicAdd, whose order within a launch the
  // hardware picks, so what "bit for bit" can mean at all is bounded by what
  // the kernels reproduce run to run. This measures that bound, and the
  // comparison below is held to it.
  const nevpt2::ScRotationResult actTwice = NEVPT2_TRY(
      nevpt2::scActiveRotationDevice(ints, dm1, dm2, dm3, f3ac, f3ca, actReq, res));
  const std::vector<nevpt2::ScRotationRequest> embedded{
      nevpt2::ScRotationRequest{embedBlock(kAct, nc, nmo), /*tangent=*/true}};
  // No hbar: this compares integral halves, which the digests' term is not in.
  const nevpt2::ScFullRotationResult viaFull = NEVPT2_TRY(nevpt2::scFullRotationDevice(
      ints, gen, dm1, dm2, dm3, f3ac, f3ca, /*digestAdjoint=*/nullptr, embedded, res));
  const double adjointGap = std::fabs(viaFull.dirs[0].integral - actOnly.dirs[0].adjoint);
  const double tangentGap =
      std::fabs(viaFull.dirs[0].tangentIntegral - actOnly.dirs[0].tangent);
  const double rerunAdjointGap = std::fabs(actTwice.dirs[0].adjoint - actOnly.dirs[0].adjoint);
  const double rerunTangentGap = std::fabs(actTwice.dirs[0].tangent - actOnly.dirs[0].tangent);
  const double bitFloor =
      std::max(rerunAdjointGap, kBitFloorRel * std::fabs(actOnly.dirs[0].adjoint));
  const double bitFloorTangent =
      std::max(rerunTangentGap, kBitFloorRel * std::fabs(actOnly.dirs[0].tangent));
  const bool sameAdjoint = adjointGap <= bitFloor;
  const bool sameTangent = tangentGap <= bitFloorTangent;
  ok = ok && sameAdjoint && sameTangent;
  std::printf("\n  the active block, bit for bit: an active-restricted kappa through the full\n"
              "  entry point against scActiveRotationDevice's own answer, and against what\n"
              "  that entry point reproduces when it is simply run twice --\n");
  std::printf("    <kappa, Y_act>  full  %.17e\n                    act   %.17e\n"
              "                    rerun %.17e\n"
              "                    full - act = %.2e, rerun - act = %.2e  %s\n",
              viaFull.dirs[0].integral, actOnly.dirs[0].adjoint, actTwice.dirs[0].adjoint,
              adjointGap, rerunAdjointGap,
              adjointGap == 0.0 ? "EQUAL (bit for bit)"
                                : (sameAdjoint ? "within the kernels' own reproducibility"
                                               : "DIFFER BY MORE THAN THAT"));
  std::printf("    Edot(kappa)     full  %.17e\n                    act   %.17e\n"
              "                    rerun %.17e\n"
              "                    full - act = %.2e, rerun - act = %.2e  %s\n",
              viaFull.dirs[0].tangentIntegral, actOnly.dirs[0].tangent,
              actTwice.dirs[0].tangent, tangentGap, rerunTangentGap,
              tangentGap == 0.0 ? "EQUAL (bit for bit)"
                                : (sameTangent ? "within the kernels' own reproducibility"
                                               : "DIFFER BY MORE THAN THAT"));
  std::printf("    (and the orbital energies' response the active block holds fixed, which "
              "this adds: %.6e)\n",
              viaFull.dirs[0].eps);
  // The floor those two comparisons used follows the spread this run
  // measured; the spread itself must stay inside its recorded envelope.
  const bool withinEnvelope = rerunAdjointGap <= kSelfVariationEnvelope &&
                              rerunTangentGap <= kSelfVariationEnvelope;
  ok = ok && withinEnvelope;
  std::printf("    same-run self-variation: rerun - act = %.2e (pairing), %.2e (tangent),\n"
              "    against the recorded envelope %.1e (docs/testing.md, \"The kernels' "
              "same-run self-variation\"): %s\n",
              rerunAdjointGap, rerunTangentGap, kSelfVariationEnvelope,
              withinEnvelope ? "ok" : "FAILED -- the kernels' nondeterminism has grown");

  // --- 4. the two intra-space finite differences ----------------------------
  for (int which = 0; which < 2; ++which) {
    const bool core = which == 0;
    const char* name = core ? "core" : "virtual";
    const std::int64_t n = core ? nc : nv;
    if (n < 2) {
      std::printf("\n  %s-internal: skipped, %lld orbital(s) leave no rotation\n", name,
                  static_cast<long long>(n));
      continue;
    }
    const nevpt2::Tensor block =
        nevpt2::antisymmetricKappa(n, kFullSeedBase + (core ? 0x200u : 0x300u));
    const std::vector<nevpt2::ScRotationRequest> one{nevpt2::ScRotationRequest{
        embedBlock(block, core ? 0 : nc + na, nmo), /*tangent=*/true}};
    const nevpt2::ScFullRotationResult r = NEVPT2_TRY(
        nevpt2::scFullRotationDevice(ints, gen, dm1, dm2, dm3, f3ac, f3ca, &hbar, one, res));
    const bool epsZero = std::fabs(r.dirs[0].eps) < kEpsZeroTol;
    const bool dualHere = std::fabs(r.dirs[0].integral - r.dirs[0].tangentIntegral) <
                          kFullRotTol;
    // No active index moves, so h2e_dot is an exactly-zero tensor and so is
    // every digest built from it: the digests' term is 0.0, not small (#88).
    const std::vector<nevpt2::ScDigestTangent> dig0 =
        NEVPT2_TRY(nevpt2::scDigestRotationTangentDevice(rdmOpt, g.get("ci"), g.nelecA,
                                                         g.nelecB, ints, gen, dm1, dm2, dm3,
                                                         f3ac, f3ca, one, res));
    // And in reverse mode (#89): hbar paired with that zero tensor.
    const bool digestZero =
        dig0[0].total == 0.0 && dig0[0].h2eDotMax == 0.0 && r.dirs[0].digest == 0.0;
    ok = ok && epsZero && dualHere && digestZero;
    std::printf("\n  %s-internal direction: <kappa, Y> = %.14e, adjoint vs tangent %.2e\n",
                name, r.dirs[0].adjoint,
                std::fabs(r.dirs[0].integral - r.dirs[0].tangentIntegral));
    std::printf("  its orbital-energy response is zero by (O2): eps = %.2e (< %.0e: %s)\n",
                r.dirs[0].eps, kEpsZeroTol, epsZero ? "ok" : "FAILED");
    std::printf("  it moves no active integral: max |h2e_dot| = %.1e, the digests' term = %.1e "
                "forward, %.1e reverse (exactly 0: %s)\n",
                dig0[0].h2eDotMax, dig0[0].total, r.dirs[0].digest,
                digestZero ? "ok" : "FAILED");
    const double scale = std::max(std::fabs(r.dirs[0].tangent), std::fabs(energies.total));
    ok = finiteDifferenceSweep(name, g, gg, core, block, r.dirs[0].tangent, scale, dm1, dm2,
                               dm3, f3ac, f3ca, res) &&
         ok;
  }

  // --- 5. Park Eqs. 56-57 off the assembled Y -------------------------------
  std::vector<std::pair<std::int64_t, std::int64_t>> pairs;
  for (std::int64_t p = 0; p < nc; ++p)
    for (std::int64_t q = p + 1; q < nc; ++q) pairs.emplace_back(p, q);
  for (std::int64_t p = nc + na; p < nmo; ++p)
    for (std::int64_t q = p + 1; q < nmo; ++q) pairs.emplace_back(p, q);
  const auto tY = std::chrono::steady_clock::now();
  const nevpt2::Tensor y = NEVPT2_TRY(nevpt2::scFullRotationMatrix(
      ints, gen, dm1, dm2, dm3, f3ac, f3ca, &hbar, pairs, kYChunk, res));
  double yMax = 0.0;
  for (const auto& [p, q] : pairs) yMax = std::max(yMax, std::fabs(y(p, q)));
  std::printf("\n  Y's inactive blocks: %zu basis directions in chunks of %lld, %.2fs, "
              "max|Y| = %.3e\n",
              pairs.size(), static_cast<long long>(kYChunk),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - tY).count(),
              yMax);
  // The matrix against the pairing, on exactly the directions it covers: the
  // first seeded kappa masked down to those two blocks, paired with the
  // probed Y, against the direct answer for that same masked direction. The
  // same functional summed in a different order, so what is left is
  // reassociation -- and it is what makes the MATRIX as checked as the
  // pairing is, which matters because Eqs. 56-57 read the matrix.
  nevpt2::Tensor kInactive({nmo, nmo});
  for (const auto& [p, q] : pairs) {
    kInactive(p, q) = kappas[0](p, q);
    kInactive(q, p) = kappas[0](q, p);
  }
  const std::vector<nevpt2::ScRotationRequest> inactiveReq{
      nevpt2::ScRotationRequest{kInactive, /*tangent=*/false}};
  const nevpt2::ScFullRotationResult rInactive = NEVPT2_TRY(nevpt2::scFullRotationDevice(
      ints, gen, dm1, dm2, dm3, f3ac, f3ca, &hbar, inactiveReq, res));
  double fromMatrix = 0.0;
  for (std::int64_t p = 0; p < nmo; ++p)
    for (std::int64_t q = 0; q < nmo; ++q) fromMatrix += kInactive(p, q) * y(p, q);
  const double matrixGap = std::fabs(fromMatrix - rInactive.dirs[0].adjoint);
  ok = ok && matrixGap < kFullMatrixTol;
  std::printf("  <kappa, Y> from the matrix vs the direct pairing, on an inactive-restricted\n"
              "  direction: %.14e vs %.14e, |delta| = %.2e (< %.0e: %s)\n",
              fromMatrix, rInactive.dirs[0].adjoint, matrixGap, kFullMatrixTol,
              matrixGap < kFullMatrixTol ? "ok" : "FAILED");

  const nevpt2::ScPseudocanonical pc = NEVPT2_TRY(nevpt2::scPseudocanonicalMultipliers(
      y, g.get("e_core"), g.get("e_virt"), na, kPcGapFloor, kPcZeroTol));
  std::printf("  the pseudocanonical multipliers z^c (Park Eqs. 56-57): %lld pair(s) divided "
              "through, %lld degenerate and left zero\n",
              static_cast<long long>(pc.divided), static_cast<long long>(pc.degenerate));
  std::printf("  max |z^c| = %.6e at the smallest gap divided by, %.3e Eh\n", pc.maxZ,
              pc.smallestGap);

  // --- 6. the digests' term against its own finite difference (#88) -------
  const std::vector<nevpt2::ScRotationRequest> actEmbedded{
      nevpt2::ScRotationRequest{embedBlock(kAct, nc, nmo), /*tangent=*/false}};
  const std::vector<nevpt2::ScDigestTangent> digAct =
      NEVPT2_TRY(nevpt2::scDigestRotationTangentDevice(rdmOpt, g.get("ci"), g.nelecA, g.nelecB,
                                                       ints, gen, dm1, dm2, dm3, f3ac, f3ca,
                                                       actEmbedded, res));
  std::printf("\n  the digests' term along check 3's active direction: %.14e\n"
              "  central differences of energiesDevice with ONLY f3ac/f3ca rebuilt, from h2e\n"
              "  rotated on the host by exp(h kappa) -- every integral, RDM and orbital energy\n"
              "  the state's own:\n",
              digAct[0].total);
  const double digScale = std::max(std::fabs(digAct[0].total), std::fabs(energies.total));
  const bool digestFd = sweepAgainst(digAct[0].total, digScale, [&](double t) {
    return scTotalDigestsRotated(g, kAct, t, ints, dm1, dm2, dm3, rdmOpt, res);
  });
  ok = ok && digestFd;
  // The same term in reverse mode (#89), along the same direction: the
  // transpose's pairing against the rebuilt digests' tangent, exact arithmetic.
  const nevpt2::ScFullRotationResult revAct = NEVPT2_TRY(nevpt2::scFullRotationDevice(
      ints, gen, dm1, dm2, dm3, f3ac, f3ca, &hbar, actEmbedded, res));
  const double revGap = std::fabs(revAct.dirs[0].digest - digAct[0].total);
  const bool revOk = revGap < kFullRotTol;
  ok = ok && revOk;
  std::printf("  and in reverse mode, <hbar, h2e_dot> = %.14e, against the forward term "
              "|delta| = %.2e (< %.0e: %s)\n",
              revAct.dirs[0].digest, revGap, kFullRotTol, revOk ? "ok" : "FAILED");

  std::printf("\n%s: the full orbital source term's adjoint and tangent agree to %.0e on every "
              "direction -- integral halves, and in full with the digests' transpose against "
              "the exact Edot -- the sidecar's promotion table holds to %.0e, the active block "
              "is unchanged to the kernels' own reproducibility (itself inside its recorded "
              "envelope), the two intra-space finite "
              "differences converge, Park Eqs. 56-57 are determined, and the digests' term "
              "converges onto its own finite difference and vanishes on the intra-space "
              "rotations\n",
              ok ? "PASS" : "FAIL", kFullRotTol, kSelfCheckTol);
  return ok;
}
