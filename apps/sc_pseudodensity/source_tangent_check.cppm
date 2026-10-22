// nevpt2.app.sc_pseudodensity:source_tangent_check -- --sc-source-tangent
// PATH: the SC source terms' TANGENT along a supplied integral perturbation
// (#220; docs/hessian-theory.md, Sec. 8, the NEVPT2 half of L_{R Theta}).
// An interface partition that exports nothing, as :hessian_kc_check is.
//
// WHAT IS DIFFERENTIATED. The gated source terms the dump writes
// (:source_dump) -- Y with the digests' transpose and the orbital energies
// responding to kappa, y with them responding to c through W -- along every
// coordinate's perturbed Hamiltonian I -> I + h Idot that PATH, an integral
// perturbation sidecar (`*.pertgold`, reference_data/integral_perturbation.py),
// carries: nevpt2.gradient's scSourceTangentDevice, Ydot by probing and ydot
// by reverse over forward. Its header says how.
//
// THE CHECKS, per coordinate, in the order they print:
//
//   0  THE SIDECARS (once, then per coordinate): the perturbation is in the
//      gradient sidecar's frame (mo_coeff bit for bit, the same calculation),
//      the gradient sidecar against its golden (--kappa-full's self-check),
//      and each coordinate's dotted h^eff and f rebuilt from its dotted h and
//      rows (h^eff's core dressing; f's diagonal and, over the whole range,
//      C5's generator) -- the same identities, linear, at the dot.
//   1  THE EMPTY PERTURBERS: a perturbation that populates an H class's
//      empty tuple is refused whole (Numerical), counted on the PASS line --
//      a displacement that leaves the point group, which this milestone does
//      not take.
//   2  EVERY PROBE'S TWO MODES: each basis direction's integrals term
//      (adjoint against forward tangent, the integral half) and sandwich
//      (reverse against forward), the worst over the n_mo (n_mo - 1) / 2
//      probes, exact arithmetic, 1e-7.
//   3  THE DUALITY: <Ydot, kappa> + <ydot, u> -- the matrix and the vector,
//      reverse -- against d^2 E / dh dtheta forward over forward
//      (scSourceTangentForwardDevice), along seeded (kappa, u), (kappa, 0)
//      and (0, u); 1e-7.
//   4  THE LADDERS: Y and y recomputed in full at I +- h Idot -- every stored
//      block, the inventory and the orbital energies displaced, the digests
//      rebuilt at the displaced h2e and their transpose with them, W at the
//      displaced rows -- central-differenced against Ydot and ydot, the worst
//      element of each relative to its largest; three fourfold halvings and
//      the best under 1e-6, as the other finite differences of the gradient
//      path (longestFourfoldRun).
//
// And one sum rule, printed: the frozen path's MO integrals do not move under
// a rigid translation, so the coordinates' Ydot (and ydot) along each
// Cartesian axis sum to the sidecar's own round-off.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries. A sidecar that does not fit
// the golden is the caller's input and propagates unchanged to main.
#include "error_handling/error_macros.h"

export module nevpt2.app.sc_pseudodensity:source_tangent_check;

import std;
import nevpt2.gradient;
import nevpt2.kappa_rotation;  // antisymmetricKappa: the duality's directions
import nevpt2.golden;
// generalIntegrals: the gradient sidecar as --kappa-full reads it, either
// flavour.
import :kappa_full_check;
// kcDirection / kcDot: the CI directions --sc-hessian-kc seeds, the same
// construction.
import :hessian_kc_check;

// Every exact identity's gate: two computations of one number, the tree's
// 1e-7 (docs/testing.md, "The source terms' tangent along an integral
// perturbation", records what is measured).
constexpr double kStTol = 1e-7;
// The sidecars' self-checks, relative: two transforms of the same integrals.
constexpr double kStSelfTol = 1e-10;
// How many probes of Ydot's integrals term share one walk: --dump-source-terms'
// chunk. And how many basis directions of Y's own share one in a ladder rung,
// where Y is assembled twice per rung: a walk's fixed cost dominates at these
// sizes, so fewer, wider walks (a launch-granularity knob, not a memory one:
// every direction is a whole-block set either way).
constexpr std::int64_t kStChunk = 16;
constexpr std::int64_t kStLadderChunk = 64;
// Seeds, distinct from every other check's.
constexpr std::uint64_t kStSeedBase = 0x5eed'0220u;
// The ladders: h = 2^-k, the best error relative to the tangent's largest
// element under kStFdTol, and a run of kStFdRun fourfold halvings -- the
// rule every finite difference of the gradient path is held to. They start
// fine on purpose: along a nuclear perturbation the gated Y has structure on
// a scale of h ~ 1e-3 (the error is not yet O(h^2) above it -- measured,
// docs/testing.md, "The source terms' tangent along an integral
// perturbation"), so the coarse rungs would only add wall time.
constexpr int kStFdFirstExp = 10;
constexpr int kStFdLastExp = 20;
constexpr std::size_t kStFdSteps = kStFdLastExp - kStFdFirstExp + 1;
constexpr double kStFdTol = 1e-6;
constexpr int kStFdRun = 3;

double stMaxAbs(const nevpt2::Tensor& t) {
  double m = 0.0;
  for (std::int64_t i = 0; i < t.size(); ++i) m = std::max(m, std::fabs(t.flat(i)));
  return m;
}

// The gradient sidecar's dotted inventory for coordinate k, uploaded.
nevpt2::GeneralIntegralsDevice uploadPerturbation(const nevpt2::PertGoldFile& pg,
                                                  const std::int64_t k,
                                                  const nevpt2::DeviceResources& res) {
  const auto up = [&](const char* name) {
    return nevpt2::uploadTensor(pg.coordinate(name, k), res);
  };
  nevpt2::GeneralIntegralsDevice d;
  d.ncore = pg.ncore;
  d.nact = pg.nact;
  d.nvirt = pg.nvirt;
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

// a + h b, host, one shape.
nevpt2::Tensor stAxpy(const nevpt2::Tensor& a, const nevpt2::Tensor& b, const double h) {
  nevpt2::Tensor out = a;
  for (std::int64_t i = 0; i < out.size(); ++i) out.flatRef(i) += h * b.flat(i);
  return out;
}

// The gated source terms at I + h Idot, as --dump-source-terms assembles them
// at I: Y over every pair (the digests rebuilt at the displaced h2e, their
// transpose with them) and y with W at the displaced rows.
struct StSourceTerms {
  nevpt2::Tensor kappa;
  nevpt2::Tensor ci;
};

nevpt2::Result<StSourceTerms> stSourceTermsAt(
    const nevpt2::GoldenFile& g, const nevpt2::GradGoldFile& gg, const nevpt2::PertGoldFile& pg,
    const std::int64_t k, const double h, const nevpt2::NevptIntegralsDevice& ints,
    const nevpt2::GeneralIntegralsDevice& gen, const nevpt2::IntegralPerturbationDevice& pert,
    const nevpt2::DeviceTensor& dm1, const nevpt2::DeviceTensor& dm2,
    const nevpt2::Tensor& h2eDot, const nevpt2::RdmBuildOptions& opt,
    const nevpt2::SigmaVectorOptions& sigma, const nevpt2::DeviceResources& res) {
  const nevpt2::DisplacedIntegralsDevice at =
      NEVPT2_TRY(nevpt2::displacedIntegralsDevice(ints, gen, pert, h, res));
  const nevpt2::Tensor& ci = g.get("ci");
  const nevpt2::RdmBuildResult b = nevpt2::buildRdmsDevice(
      opt, ci, g.ncas, g.nelecA, g.nelecB, stAxpy(g.get("h2e"), h2eDot, h), res);
  const nevpt2::DeviceTensor hbar = nevpt2::scDigestAdjointDevice(
      opt, ci, g.nelecA, g.nelecB, at.ints, dm1, dm2, b.dm3, b.f3ac, b.f3ca, res);
  std::vector<std::pair<std::int64_t, std::int64_t>> pairs;
  for (std::int64_t p = 0; p < gg.nmo; ++p)
    for (std::int64_t q = p + 1; q < gg.nmo; ++q) pairs.emplace_back(p, q);
  StSourceTerms out;
  out.kappa = NEVPT2_TRY(nevpt2::scFullRotationMatrix(at.ints, at.gen, dm1, dm2, b.dm3, b.f3ac,
                                                      b.f3ca, &hbar, pairs, kStLadderChunk, res));
  const auto row = [&](const char* name) {
    return stAxpy(gg.get(name), pg.coordinate(name, k), h);
  };
  const nevpt2::OrbitalEnergyResponse w = nevpt2::orbitalEnergyResponse(
      gg.ncore, gg.nact, gg.nvirt, row("g_xa_cc"), row("g_xc_ca"), row("g_xa_vv"),
      row("g_xv_va"));
  out.ci = nevpt2::scCiSourceDevice(at.ints, ci, g.nelecA, g.nelecB, dm1, dm2, b.dm3, b.f3ac,
                                    b.f3ca, sigma, &w, res)
               .y;
  return out;
}

// One ladder's verdict, printed: err[i] is the worst element of the central
// difference at step[i] against `want`, relative to want's largest.
bool stLadderVerdict(const char* what, const std::array<double, kStFdSteps>& step,
                     const std::array<double, kStFdSteps>& err) {
  std::size_t bestAt = 0;
  for (std::size_t i = 1; i < kStFdSteps; ++i)
    if (err[i] < err[bestAt]) bestAt = i;
  const nevpt2::FourfoldRun run = nevpt2::longestFourfoldRun(err);
  const bool ok = err[bestAt] < kStFdTol && run.length >= kStFdRun;
  std::printf("  %s: best %.3e at h = %.3e; %d consecutive fourfold halvings ending at h = "
              "%.3e (%d needed) -- %s\n",
              what, err[bestAt], step[bestAt], run.length, step[run.end], kStFdRun,
              ok ? "ok" : "FAILED");
  return ok;
}

// What one coordinate came to: computed, or refused with the reason.
struct StCoordinate {
  bool refused = false;
  bool ok = true;
  nevpt2::Tensor kappaDot;
  nevpt2::Tensor ciDot;
};

nevpt2::Result<StCoordinate> stCoordinate(
    const nevpt2::GoldenFile& g, const nevpt2::GradGoldFile& gg, const nevpt2::PertGoldFile& pg,
    const std::int64_t k, const nevpt2::NevptIntegralsDevice& ints,
    const nevpt2::GeneralIntegralsDevice& gen, const nevpt2::RdmTensors& rdm,
    const nevpt2::OrbitalEnergyResponse& wResp, const nevpt2::DeviceTensor& hbar,
    const nevpt2::RdmBuildOptions& opt, const nevpt2::SigmaVectorOptions& sigma,
    const nevpt2::DeviceResources& res) {
  static constexpr std::string_view kAxis = "xyz";
  const nevpt2::Tensor& ci = g.get("ci");
  StCoordinate out;
  std::printf("\n  === coordinate %lld: atom %lld, %c ===\n", static_cast<long long>(k),
              static_cast<long long>(k / 3), kAxis[static_cast<std::size_t>(k % 3)]);

  // --- 0. the dotted inventory against itself ---------------------------------
  const nevpt2::IntegralPerturbationDevice pert = NEVPT2_TRY(
      nevpt2::integralPerturbationDevice(ints, uploadPerturbation(pg, k, res), res));
  {
    const nevpt2::ScGeneralCheck self =
        NEVPT2_TRY(nevpt2::scGeneralIntegralCheck(pert.d, pert.gen, rdm.dm1, res));
    const double worst = std::max({self.heff, self.fock, self.fockFull});
    out.ok = out.ok && worst < kStSelfTol;
    std::printf("  [0] the dot against itself (relative): h^eff's dressing %.2e, f's diagonal "
                "%.2e, f over the whole range %.2e (< %.0e: %s)\n",
                self.heff, self.fock, self.fockFull, kStSelfTol,
                worst < kStSelfTol ? "ok" : "FAILED");
  }

  const nevpt2::ScSourceTangentSetup setup{ints, gen,      rdm,   ci,  g.nelecA,
                                           g.nelecB, opt, wResp, hbar};
  const auto t0 = std::chrono::steady_clock::now();
  const nevpt2::Result<nevpt2::ScSourceTangentResult> tangent =
      nevpt2::scSourceTangentDevice(setup, pert, sigma, kStChunk, res);
  // --- 1. the empty perturbers --------------------------------------------------
  if (!tangent && tangent.error().kind == nevpt2::ErrorKind::Numerical) {
    std::printf("  [1] REFUSED, Numerical: %s\n", tangent.error().message.c_str());
    out.refused = true;
    return out;
  }
  if (!tangent) return std::unexpected(tangent.error());
  const nevpt2::ScSourceTangentResult& t = *tangent;
  {
    std::int64_t empty = 0, populated = 0;
    for (const nevpt2::ScGuardedCensus& c : t.guarded) {
      empty += c.empty;
      populated += c.populated;
    }
    std::printf("  [1] the empty perturbers: %lld guarded tuples empty by the integrals, %lld "
                "of them populated by Idot (an H class's would refuse it) -- ok\n",
                static_cast<long long>(empty), static_cast<long long>(populated));
  }
  std::printf("  Ydot and ydot in %.2fs: %lld probes; max |hdot2e| = %.3e, max |f3'[Idot]| = "
              "%.3e\n  |Ydot| max %.6e (integrals term %.6e, sandwich %.6e); |ydot| max %.6e "
              "(walk %.6e, digests %.6e)\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
              static_cast<long long>(t.probes), t.h2eDotMax, t.f3DotMax, stMaxAbs(t.kappaDot),
              stMaxAbs(t.kappaDotIntegrals), stMaxAbs(t.kappaDotSandwich), stMaxAbs(t.ciDot),
              stMaxAbs(t.ciDotWalk), stMaxAbs(t.ciDotDigests));

  // --- 2. every probe's two modes ----------------------------------------------
  {
    const bool ok = t.integralsDuality < kStTol && t.sandwichDuality < kStTol;
    out.ok = out.ok && ok;
    std::printf("  [2] every probe's two modes, worst: the integrals term's integral half %.2e, "
                "the sandwich %.2e (< %.0e: %s)\n",
                t.integralsDuality, t.sandwichDuality, kStTol, ok ? "ok" : "FAILED");
  }

  // --- 3. the duality ------------------------------------------------------------
  {
    const std::int64_t nmo = gg.nmo;
    const nevpt2::Tensor k1 = nevpt2::antisymmetricKappa(nmo, kStSeedBase + 1);
    const nevpt2::Tensor k2 = nevpt2::antisymmetricKappa(nmo, kStSeedBase + 2);
    const nevpt2::Tensor u1 = kcDirection(ci, kStSeedBase + 3);
    const nevpt2::Tensor u2 = kcDirection(ci, kStSeedBase + 4);
    const std::vector<std::pair<std::string, std::pair<nevpt2::Tensor, nevpt2::Tensor>>> dirs{
        {"(kappa, u)", {k1, u1}},
        {"(kappa, 0)", {k2, nevpt2::Tensor()}},
        {"(0, u)", {nevpt2::Tensor(), u2}},
        {"(0, c)", {nevpt2::Tensor(), ci}},
    };
    std::vector<std::pair<nevpt2::Tensor, nevpt2::Tensor>> req;
    for (const auto& [name, d] : dirs) req.push_back(d);
    const std::vector<nevpt2::ScSourceTangentForward> f =
        NEVPT2_TRY(nevpt2::scSourceTangentForwardDevice(setup, pert, req, res));
    std::printf("  [3] the duality <Ydot, kappa> + <ydot, u> against d^2E/dh dtheta, forward "
                "over forward:\n");
    std::printf("  %-11s %22s %22s %11s   %s\n", "theta", "reverse", "forward", "|delta|",
                "forward's terms: integrals + digests + sandwich + mixed");
    double worst = 0.0;
    for (std::size_t i = 0; i < dirs.size(); ++i) {
      const auto& [kappa, u] = dirs[i].second;
      double lhs = 0.0;
      if (kappa.rank() != 0)
        for (std::int64_t e = 0; e < kappa.size(); ++e)
          lhs += kappa.flat(e) * t.kappaDot.flat(e);
      if (u.rank() != 0) lhs += kcDot(u, t.ciDot);
      const double d = std::fabs(lhs - f[i].total());
      worst = std::max(worst, d);
      std::printf("  %-11s %22.14e %22.14e %11.2e   %.4e + %.4e + %.4e + %.4e\n",
                  dirs[i].first.c_str(), lhs, f[i].total(), d, f[i].integrals,
                  f[i].integralsDigest, f[i].sandwich, f[i].mixed);
    }
    const bool ok = worst < kStTol;
    out.ok = out.ok && ok;
    std::printf("  duality, worst |delta| = %.2e (< %.0e: %s)\n", worst, kStTol,
                ok ? "ok" : "FAILED");
  }

  // --- 4. the ladders ----------------------------------------------------------
  {
    const nevpt2::Tensor h2eDot = nevpt2::downloadTensor(pert.d.h2e, res.stream());
    std::array<double, kStFdSteps> step{}, errY{}, errC{};
    const double scaleY = std::max(1e-300, stMaxAbs(t.kappaDot));
    const double scaleC = std::max(1e-300, stMaxAbs(t.ciDot));
    const auto tl = std::chrono::steady_clock::now();
    std::printf("  [4] central differences of Y and y at I +- h Idot, worst element relative to "
                "the tangent's largest:\n");
    std::printf("  %5s %12s %12s %8s %12s %8s\n", "k", "h = 2^-k", "Y", "ratio", "y", "ratio");
    for (std::size_t i = 0; i < kStFdSteps; ++i) {
      const int e = kStFdFirstExp + static_cast<int>(i);
      const double h = std::ldexp(1.0, -e);
      step[i] = h;
      const StSourceTerms plus = NEVPT2_TRY(stSourceTermsAt(
          g, gg, pg, k, h, ints, gen, pert, rdm.dm1, rdm.dm2, h2eDot, opt, sigma, res));
      const StSourceTerms minus = NEVPT2_TRY(stSourceTermsAt(
          g, gg, pg, k, -h, ints, gen, pert, rdm.dm1, rdm.dm2, h2eDot, opt, sigma, res));
      double ey = 0.0, ec = 0.0;
      for (std::int64_t j = 0; j < t.kappaDot.size(); ++j)
        ey = std::max(ey, std::fabs((plus.kappa.flat(j) - minus.kappa.flat(j)) / (2.0 * h) -
                                    t.kappaDot.flat(j)));
      for (std::int64_t j = 0; j < t.ciDot.size(); ++j)
        ec = std::max(ec, std::fabs((plus.ci.flat(j) - minus.ci.flat(j)) / (2.0 * h) -
                                    t.ciDot.flat(j)));
      errY[i] = ey / scaleY;
      errC[i] = ec / scaleC;
      if (i == 0)
        std::printf("  %5d %12.3e %12.3e %8s %12.3e %8s\n", e, h, errY[i], "-", errC[i], "-");
      else
        std::printf("  %5d %12.3e %12.3e %8.2f %12.3e %8.2f\n", e, h, errY[i],
                    errY[i] > 0.0 ? errY[i - 1] / errY[i] : 0.0, errC[i],
                    errC[i] > 0.0 ? errC[i - 1] / errC[i] : 0.0);
    }
    const bool okY = stLadderVerdict("Y", step, errY);
    const bool okC = stLadderVerdict("y", step, errC);
    out.ok = out.ok && okY && okC;
    std::printf("  ladders in %.2fs\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count());
  }
  out.kappaDot = t.kappaDot;
  out.ciDot = t.ciDot;
  return out;
}

// Runs every check above on every coordinate of the sidecar at `pertPath`.
// Returns whether every computed coordinate held them and at least one was
// computed; a sidecar that does not fit the golden is an Error for main.
nevpt2::Result<bool> checkScSourceTangent(const nevpt2::GoldenFile& g,
                                          const nevpt2::GradGoldFile& gg,
                                          const std::string& pertPath,
                                          const nevpt2::NevptIntegralsDevice& ints,
                                          const nevpt2::DeviceTensor& dm1,
                                          const nevpt2::DeviceTensor& dm2,
                                          const nevpt2::RdmBuildResult& built,
                                          const nevpt2::RdmBuildOptions& opt,
                                          const nevpt2::DeviceResources& res) {
  const nevpt2::RdmTensors rdm{dm1, dm2, built.dm3, built.f3ac, built.f3ca};
  nevpt2::SigmaVectorOptions sigma;
  sigma.nTiles = opt.nTiles;
  std::printf("\n--sc-source-tangent: the SC source terms' tangent along a supplied integral "
              "perturbation (docs/hessian-theory.md, Sec. 8)\n");
  std::printf("  integral perturbation sidecar: %s\n", pertPath.c_str());
  const nevpt2::PertGoldFile pg = NEVPT2_TRY(nevpt2::loadPertGold(pertPath));
  NEVPT2_TRY(pg.checkAgainst(g, gg));
  const bool fitted = gg.densityFitted();
  std::printf("  %lld coordinates of %lld atoms, n_mo = %lld = %lld core + %lld active + %lld "
              "virtual%s; its mo_coeff is the gradient sidecar's, bit for bit\n",
              static_cast<long long>(pg.ncoord()), static_cast<long long>(pg.natm),
              static_cast<long long>(pg.nmo), static_cast<long long>(pg.ncore),
              static_cast<long long>(pg.nact), static_cast<long long>(pg.nvirt),
              fitted ? std::format(", density-fitted (naux = {})", pg.naux).c_str() : "");

  // --- 0. the gradient sidecar against its golden ---------------------------------
  const nevpt2::GeneralIntegralsDevice gen = NEVPT2_TRY(generalIntegrals(g, gg, res));
  bool ok = true;
  {
    const nevpt2::ScGeneralCheck self =
        NEVPT2_TRY(nevpt2::scGeneralIntegralCheck(ints, gen, dm1, res));
    const double worst =
        std::max({self.blocks, self.heff, self.fock, self.eps, self.fockFull});
    ok = ok && worst < kStSelfTol;
    std::printf("  [0] the gradient sidecar against its golden (relative): restrictions %.2e "
                "(%s), h^eff %.2e, f's diagonal %.2e, (O2) %.2e, f over the whole range %.2e "
                "(< %.0e: %s)\n",
                self.blocks, self.worstBlock.c_str(), self.heff, self.fock, self.eps,
                self.fockFull, kStSelfTol, worst < kStSelfTol ? "ok" : "FAILED");
  }
  const nevpt2::OrbitalEnergyResponse wResp = nevpt2::orbitalEnergyResponse(
      gg.ncore, gg.nact, gg.nvirt, gg.get("g_xa_cc"), gg.get("g_xc_ca"), gg.get("g_xa_vv"),
      gg.get("g_xv_va"));
  const nevpt2::DeviceTensor hbar = nevpt2::scDigestAdjointDevice(
      opt, g.get("ci"), g.nelecA, g.nelecB, ints, dm1, dm2, built.dm3, built.f3ac, built.f3ca,
      res);

  std::int64_t computed = 0, refused = 0;
  std::vector<std::optional<StCoordinate>> coords;
  for (std::int64_t k = 0; k < pg.ncoord(); ++k) {
    StCoordinate c = NEVPT2_TRY(
        stCoordinate(g, gg, pg, k, ints, gen, rdm, wResp, hbar, opt, sigma, res));
    if (c.refused) {
      ++refused;
      coords.emplace_back();
      continue;
    }
    ++computed;
    ok = ok && c.ok;
    coords.emplace_back(std::move(c));
  }

  // The translational sum rule, per Cartesian axis where every atom's
  // coordinate was computed: printed, not gated -- it checks the sidecar's
  // own derivative, which reference_data/integral_perturbation.py ladders.
  static constexpr std::string_view kAxis = "xyz";
  for (std::int64_t axis = 0; axis < 3; ++axis) {
    bool all = true;
    for (std::int64_t a = 0; a < pg.natm; ++a)
      all = all && coords[static_cast<std::size_t>(3 * a + axis)].has_value();
    if (!all) continue;
    nevpt2::Tensor sy = coords[static_cast<std::size_t>(axis)]->kappaDot;
    nevpt2::Tensor sc = coords[static_cast<std::size_t>(axis)]->ciDot;
    double my = stMaxAbs(sy), mc = stMaxAbs(sc);
    for (std::int64_t a = 1; a < pg.natm; ++a) {
      const StCoordinate& c = *coords[static_cast<std::size_t>(3 * a + axis)];
      sy = stAxpy(sy, c.kappaDot, 1.0);
      sc = stAxpy(sc, c.ciDot, 1.0);
      my = std::max(my, stMaxAbs(c.kappaDot));
      mc = std::max(mc, stMaxAbs(c.ciDot));
    }
    std::printf("\n  the translational sum rule along %c: max |sum_A Ydot_A| = %.2e (of %.3e), "
                "max |sum_A ydot_A| = %.2e (of %.3e)\n",
                kAxis[static_cast<std::size_t>(axis)], stMaxAbs(sy), my, stMaxAbs(sc), mc);
  }

  ok = ok && computed > 0;
  std::printf("\n%s: the SC source terms' tangent along the integral perturbation agrees with "
              "itself by duality to %.0e -- every probe's two modes, and <Ydot, kappa> + "
              "<ydot, u> against d^2E/dh dtheta forward over forward -- and the ladders of the "
              "gated Y and y at I +- h Idot converge to it, on %lld of %lld coordinates; %lld "
              "refused, populating empty perturbers (Sec. 5.6)%s\n",
              ok ? "PASS" : "FAIL", kStTol, static_cast<long long>(computed),
              static_cast<long long>(pg.ncoord()), static_cast<long long>(refused),
              fitted ? " (density-fitted)" : "");
  return ok;
}
