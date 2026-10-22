// nevpt2.app.sc_pseudodensity:source_tangent_pc_check -- --pc
// --sc-source-tangent PATH: the PC source terms' TANGENT along a supplied
// integral perturbation (#234; docs/hessian-theory.md, Sec. 6.4 / 6.7 and 8,
// the PC half of L_{R Theta}), the twin of #220's :source_tangent_check.
// An interface partition that exports nothing, as :source_tangent_check is.
//
// WHAT IS DIFFERENTIATED. PC's gated source terms: Y_pc with hbar_pc and the
// orbital energies responding to kappa (pcFullRotationMatrixOnePass, the
// dump's), and the THREE-TERM y_pc with them responding to c through W
// (pcCiSourceThreeTermDevice with the sidecar's W) -- along every
// coordinate's perturbed Hamiltonian I -> I + h Idot that PATH carries:
// nevpt2.gradient's pcSourceTangentDevice, Ydot probed and ydot PC's FULL
// mixed block in reverse. Its header says how, and why Ydot is probed.
//
// WHAT IS DIFFERENT FROM SC's. S is RDM-only and does not move along an
// integral perturbation, so no lift test runs on Idot's leg; PC has no
// empty-perturber term (Sec. 6.6), so no coordinate is censused or refused
// for one, symmetry-breaking ones included. A forward CI direction u is
// tested for its lift as PC's mixed block tests it, so the u here are the
// ones --pc --sc-hessian-kc uses (:hessian_kc_pc_check's `Epq` and `rot`
// kinds, which do not lift) and the state; a random amplitude vector would
// lift and be refused.
//
// THE CHECKS, per coordinate, in the order they print:
//
//   0  THE SIDECARS (once, then per coordinate): as SC's -- the perturbation
//      in the gradient sidecar's frame, the gradient sidecar against its
//      golden, and each coordinate's dotted h^eff and f against themselves.
//   1  THE FIRST DERIVATIVE ALONG Idot, three ways: Ehat'_pc paired with
//      Idot's own set (hbar_pc in), its forward tangent plus the digests'
//      forward term, and the FULL mixed walk's own Edot(Idot) (digests
//      merged in K', Sijrs's included). 1e-7.
//   2  EVERY PROBE'S TWO MODES: each basis direction's integrals term
//      (adjoint against forward tangent, the integral half) and sandwich
//      (reverse against forward), the worst over the n_mo (n_mo - 1) / 2
//      probes. 1e-7.
//   3  THE DUALITY: <Ydot, kappa> + <ydot, u> against d^2 E_pc / dh dtheta
//      forward over forward (pcSourceTangentForwardDevice), along (kappa,
//      Epq), (kappa, 0), (0, rot) and (0, c); the mixed block's rotation term
//      printed per direction. 1e-7.
//   4  THE LADDERS: Y_pc and the three-term y_pc recomputed in full at
//      I +- h Idot -- every stored block, the inventory and the orbital
//      energies displaced, the digests and hbar_pc rebuilt at the displaced
//      h2e, W at the displaced rows -- central-differenced against Ydot and
//      ydot, the worst element relative to the tangent's largest; three
//      fourfold halvings and the best under 1e-6. PC's ladders start at
//      h = 2^-4 (the PC gate convention). A SYMMETRY ZERO -- a tangent whose
//      largest element is round-off of the term's own (on N2, ydot along x
//      and y: no active pair carries the displacement's irrep, so the
//      perturbation cannot reach the CI source term at first order) -- has
//      no error to halve: its rows are measured against the term's own
//      largest element instead, and must sit below the same 1e-6 at every
//      rung, not merely the best (the difference is round-off over h).
//
// And the translational sum rule, printed, as SC's.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries. A sidecar that does not fit
// the golden is the caller's input and propagates unchanged to main.
#include "error_handling/error_macros.h"

export module nevpt2.app.sc_pseudodensity:source_tangent_pc_check;

import std;
import nevpt2.gradient;
import nevpt2.kappa_rotation;  // antisymmetricKappa: the duality's orbital direction
import nevpt2.golden;
import nevpt2.rdm_host_dm;     // HostDm12Builder: the non-lifting CI directions
// generalIntegrals: the gradient sidecar as --kappa-full reads it.
import :kappa_full_check;
// kcDot: the determinant-space dot.
import :hessian_kc_check;
// pkcEpqDirection / pkcRotDirection: PC's mixed check's non-lifting CI
// directions.
import :hessian_kc_pc_check;
// uploadPerturbation / stAxpy / stMaxAbs: SC's tangent check's helpers.
import :source_tangent_check;

// Every exact identity's gate, SC's: the tree's 1e-7 (docs/testing.md, "PC's
// source terms' tangent along an integral perturbation", records what is
// measured).
constexpr double kPstTol = 1e-7;
constexpr double kPstSelfTol = 1e-10;
// Probes of Ydot's integrals term per walk: SC's chunk.
constexpr std::int64_t kPstChunk = 16;
// Seeds, distinct from every other check's.
constexpr std::uint64_t kPstSeedBase = 0x5eed'0234u;
// The ladders: h = 2^-k from the PC gate's 2^-4, the best error relative to
// the tangent's largest element under kPstFdTol, and a run of kPstFdRun
// fourfold halvings -- the rule every finite difference of the gradient path
// is held to.
constexpr int kPstFdFirstExp = 4;
constexpr int kPstFdLastExp = 18;
constexpr std::size_t kPstFdSteps = kPstFdLastExp - kPstFdFirstExp + 1;
constexpr double kPstFdTol = 1e-6;
constexpr int kPstFdRun = 3;
// A tangent is a symmetry zero when its largest element is below this times
// the term's own largest at I (the duality has already held it to its
// forward twin, which is the same zero).
constexpr double kPstZeroRel = 1e-12;

// PC's gated source terms at I + h Idot, as the PC dump assembles them at I:
// Y_pc in one pass (the digests and hbar_pc rebuilt at the displaced h2e)
// and the three-term y_pc with W at the displaced rows.
struct PstSourceTerms {
  nevpt2::Tensor kappa;
  nevpt2::Tensor ci;
};

nevpt2::Result<PstSourceTerms> pstSourceTermsAt(
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
  const nevpt2::DeviceTensor hbar = NEVPT2_TRY(nevpt2::pcDigestAdjointDevice(
      opt, ci, g.nelecA, g.nelecB, at.ints, dm1, dm2, b.dm3, b.f3ac, b.f3ca, res));
  PstSourceTerms out;
  out.kappa = NEVPT2_TRY(nevpt2::pcFullRotationMatrixOnePass(at.ints, at.gen, dm1, dm2, b.dm3,
                                                             b.f3ac, b.f3ca, &hbar, res));
  const auto row = [&](const char* name) {
    return stAxpy(gg.get(name), pg.coordinate(name, k), h);
  };
  const nevpt2::OrbitalEnergyResponse w = nevpt2::orbitalEnergyResponse(
      gg.ncore, gg.nact, gg.nvirt, row("g_xa_cc"), row("g_xc_ca"), row("g_xa_vv"),
      row("g_xv_va"));
  const nevpt2::RdmTensors rdm{dm1, dm2, b.dm3, b.f3ac, b.f3ca};
  const nevpt2::PcCiHessianResult y = NEVPT2_TRY(nevpt2::pcCiSourceThreeTermDevice(
      at.ints, ci, g.nelecA, g.nelecB, rdm, /*perClass=*/false, sigma, res, &w));
  if (!y.ok)
    return nevpt2::err_numerical(std::format(
        "the three-term PC y at I + h Idot (h = {:.3e}) has no value: {} refused", h,
        y.walk.refusals()));
  out.ci = y.y;
  return out;
}

// One ladder's verdict, printed, as SC's (stLadderVerdict) on PC's rungs; a
// symmetry zero's (`zero`, err relative to the term's own largest) holds
// every rung under the tolerance instead.
bool pstLadderVerdict(const char* what, const std::array<double, kPstFdSteps>& step,
                      const std::array<double, kPstFdSteps>& err, const bool zero) {
  if (zero) {
    double worst = 0.0;
    for (const double e : err) worst = std::max(worst, e);
    const bool ok = worst < kPstFdTol;
    std::printf("  %s: a symmetry zero -- every rung's central difference relative to the term's "
                "own largest element, worst %.3e (< %.0e) -- %s\n",
                what, worst, kPstFdTol, ok ? "ok" : "FAILED");
    return ok;
  }
  std::size_t bestAt = 0;
  for (std::size_t i = 1; i < kPstFdSteps; ++i)
    if (err[i] < err[bestAt]) bestAt = i;
  const nevpt2::FourfoldRun run = nevpt2::longestFourfoldRun(err);
  const bool ok = err[bestAt] < kPstFdTol && run.length >= kPstFdRun;
  std::printf("  %s: best %.3e at h = %.3e; %d consecutive fourfold halvings ending at h = "
              "%.3e (%d needed) -- %s\n",
              what, err[bestAt], step[bestAt], run.length, step[run.end], kPstFdRun,
              ok ? "ok" : "FAILED");
  return ok;
}

struct PstCoordinate {
  bool ok = true;
  nevpt2::Tensor kappaDot;
  nevpt2::Tensor ciDot;
};

nevpt2::Result<PstCoordinate> pstCoordinate(
    const nevpt2::GoldenFile& g, const nevpt2::GradGoldFile& gg, const nevpt2::PertGoldFile& pg,
    const std::int64_t k, const nevpt2::NevptIntegralsDevice& ints,
    const nevpt2::GeneralIntegralsDevice& gen, const nevpt2::RdmTensors& rdm,
    const nevpt2::OrbitalEnergyResponse& wResp, const nevpt2::DeviceTensor& hbar,
    const std::vector<std::pair<std::string, nevpt2::Tensor>>& us,
    const nevpt2::RdmBuildOptions& opt, const nevpt2::SigmaVectorOptions& sigma,
    const nevpt2::DeviceResources& res) {
  static constexpr std::string_view kAxis = "xyz";
  const nevpt2::Tensor& ci = g.get("ci");
  PstCoordinate out;
  std::printf("\n  === coordinate %lld: atom %lld, %c ===\n", static_cast<long long>(k),
              static_cast<long long>(k / 3), kAxis[static_cast<std::size_t>(k % 3)]);

  // --- 0. the dotted inventory against itself ---------------------------------
  const nevpt2::IntegralPerturbationDevice pert = NEVPT2_TRY(
      nevpt2::integralPerturbationDevice(ints, uploadPerturbation(pg, k, res), res));
  {
    const nevpt2::ScGeneralCheck self =
        NEVPT2_TRY(nevpt2::scGeneralIntegralCheck(pert.d, pert.gen, rdm.dm1, res));
    const double worst = std::max({self.heff, self.fock, self.fockFull});
    out.ok = out.ok && worst < kPstSelfTol;
    std::printf("  [0] the dot against itself (relative): h^eff's dressing %.2e, f's diagonal "
                "%.2e, f over the whole range %.2e (< %.0e: %s)\n",
                self.heff, self.fock, self.fockFull, kPstSelfTol,
                worst < kPstSelfTol ? "ok" : "FAILED");
  }

  const nevpt2::ScSourceTangentSetup setup{ints, gen,      rdm,   ci,  g.nelecA,
                                           g.nelecB, opt, wResp, hbar};
  const auto t0 = std::chrono::steady_clock::now();
  const nevpt2::PcSourceTangentResult t =
      NEVPT2_TRY(nevpt2::pcSourceTangentDevice(setup, pert, sigma, kPstChunk, res));
  std::printf("  Ydot and ydot in %.2fs: %lld probes; max |hdot2e| = %.3e, max |f3'[Idot]| = "
              "%.3e\n  |Ydot| max %.6e (integrals term %.6e, sandwich %.6e); |ydot| max %.6e "
              "(walk %.6e, digests %.6e)\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
              static_cast<long long>(t.probes), t.h2eDotMax, t.f3DotMax, stMaxAbs(t.kappaDot),
              stMaxAbs(t.kappaDotIntegrals), stMaxAbs(t.kappaDotSandwich), stMaxAbs(t.ciDot),
              stMaxAbs(t.ciDotWalk), stMaxAbs(t.ciDotDigests));

  // --- 1. the first derivative along Idot -------------------------------------
  {
    const double dFwd = std::fabs(t.edotAdjoint - t.edotForward);
    const double dWalk = std::fabs(t.edotAdjoint - t.edotWalk);
    const bool ok = dFwd < kPstTol && dWalk < kPstTol;
    out.ok = out.ok && ok;
    std::printf("  [1] Edot(Idot): the pairing (hbar_pc in) %.14e, forward %.14e (|delta| "
                "%.2e), the FULL mixed walk's own %.14e (|delta| %.2e) (< %.0e: %s)\n",
                t.edotAdjoint, t.edotForward, dFwd, t.edotWalk, dWalk, kPstTol,
                ok ? "ok" : "FAILED");
  }

  // --- 2. every probe's two modes ----------------------------------------------
  {
    const bool ok = t.integralsDuality < kPstTol && t.sandwichDuality < kPstTol;
    out.ok = out.ok && ok;
    std::printf("  [2] every probe's two modes, worst: the integrals term's integral half %.2e, "
                "the sandwich %.2e (< %.0e: %s)\n",
                t.integralsDuality, t.sandwichDuality, kPstTol, ok ? "ok" : "FAILED");
  }

  // --- 3. the duality ------------------------------------------------------------
  {
    const std::int64_t nmo = gg.nmo;
    const nevpt2::Tensor k1 = nevpt2::antisymmetricKappa(nmo, kPstSeedBase + 1);
    const nevpt2::Tensor k2 = nevpt2::antisymmetricKappa(nmo, kPstSeedBase + 2);
    std::vector<std::pair<std::string, std::pair<nevpt2::Tensor, nevpt2::Tensor>>> dirs;
    dirs.push_back({std::format("(kappa, {})", us[0].first), {k1, us[0].second}});
    dirs.push_back({"(kappa, 0)", {k2, nevpt2::Tensor()}});
    for (std::size_t i = 1; i < us.size(); ++i)
      dirs.push_back({std::format("(0, {})", us[i].first), {nevpt2::Tensor(), us[i].second}});
    std::vector<std::pair<nevpt2::Tensor, nevpt2::Tensor>> req;
    for (const auto& [name, d] : dirs) req.push_back(d);
    const std::vector<nevpt2::PcSourceTangentForward> f =
        NEVPT2_TRY(nevpt2::pcSourceTangentForwardDevice(setup, pert, req, res));
    std::printf("  [3] the duality <Ydot, kappa> + <ydot, u> against d^2E_pc/dh dtheta, "
                "forward over forward:\n");
    std::printf("  %-22s %22s %22s %11s   %s\n", "theta", "reverse", "forward", "|delta|",
                "forward's terms: integrals + digests + sandwich + mixed (rotation)");
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
      std::printf("  %-22s %22.14e %22.14e %11.2e   %.4e + %.4e + %.4e + %.4e (%.2e)\n",
                  dirs[i].first.c_str(), lhs, f[i].total(), d, f[i].integrals,
                  f[i].integralsDigest, f[i].sandwich, f[i].mixed, f[i].mixedRotation);
    }
    const bool ok = worst < kPstTol;
    out.ok = out.ok && ok;
    std::printf("  duality, worst |delta| = %.2e (< %.0e: %s)\n", worst, kPstTol,
                ok ? "ok" : "FAILED");
  }

  // --- 4. the ladders ----------------------------------------------------------
  {
    const nevpt2::Tensor h2eDot = nevpt2::downloadTensor(pert.d.h2e, res.stream());
    std::array<double, kPstFdSteps> step{}, errY{}, errC{};
    const auto tl = std::chrono::steady_clock::now();
    // The terms at I itself: the scale a symmetry zero is measured against.
    const PstSourceTerms base = NEVPT2_TRY(pstSourceTermsAt(
        g, gg, pg, k, 0.0, ints, gen, pert, rdm.dm1, rdm.dm2, h2eDot, opt, sigma, res));
    const double baseY = stMaxAbs(base.kappa), baseC = stMaxAbs(base.ci);
    const bool zeroY = stMaxAbs(t.kappaDot) <= kPstZeroRel * baseY;
    const bool zeroC = stMaxAbs(t.ciDot) <= kPstZeroRel * baseC;
    const double scaleY = std::max(1e-300, zeroY ? baseY : stMaxAbs(t.kappaDot));
    const double scaleC = std::max(1e-300, zeroC ? baseC : stMaxAbs(t.ciDot));
    std::printf("  [4] central differences of Y_pc and the three-term y_pc at I +- h Idot, "
                "worst element relative to the tangent's largest%s:\n",
                zeroY || zeroC ? std::format(" (a symmetry zero, {}{}{}: to the term's own, "
                                             "|Y| {:.3e}, |y| {:.3e})",
                                             zeroY ? "Ydot" : "", zeroY && zeroC ? " and " : "",
                                             zeroC ? "ydot" : "", baseY, baseC)
                                     .c_str()
                               : "");
    std::printf("  %5s %12s %12s %8s %12s %8s\n", "k", "h = 2^-k", "Y", "ratio", "y", "ratio");
    for (std::size_t i = 0; i < kPstFdSteps; ++i) {
      const int e = kPstFdFirstExp + static_cast<int>(i);
      const double h = std::ldexp(1.0, -e);
      step[i] = h;
      const PstSourceTerms plus = NEVPT2_TRY(pstSourceTermsAt(
          g, gg, pg, k, h, ints, gen, pert, rdm.dm1, rdm.dm2, h2eDot, opt, sigma, res));
      const PstSourceTerms minus = NEVPT2_TRY(pstSourceTermsAt(
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
    const bool okY = pstLadderVerdict("Y_pc", step, errY, zeroY);
    const bool okC = pstLadderVerdict("y_pc", step, errC, zeroC);
    out.ok = out.ok && okY && okC;
    std::printf("  ladders in %.2fs\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count());
  }
  out.kappaDot = t.kappaDot;
  out.ciDot = t.ciDot;
  return out;
}

// Runs every check above on every coordinate of the sidecar at `pertPath`.
// Returns whether every coordinate held them; a sidecar that does not fit the
// golden, or a class the solve refuses, is an Error for main.
nevpt2::Result<bool> checkPcSourceTangent(const nevpt2::GoldenFile& g,
                                          const nevpt2::GradGoldFile& gg,
                                          const std::string& pertPath,
                                          const nevpt2::NevptIntegralsDevice& ints,
                                          const nevpt2::DeviceTensor& dm1,
                                          const nevpt2::DeviceTensor& dm2,
                                          const nevpt2::RdmBuildResult& built,
                                          const nevpt2::RdmBuildOptions& opt,
                                          const nevpt2::DeviceResources& res) {
  const nevpt2::RdmTensors rdm{dm1, dm2, built.dm3, built.f3ac, built.f3ca};
  const nevpt2::Tensor& ci = g.get("ci");
  nevpt2::SigmaVectorOptions sigma;
  sigma.nTiles = opt.nTiles;
  std::printf("\n--pc --sc-source-tangent: the PC source terms' tangent along a supplied "
              "integral perturbation (docs/hessian-theory.md, Sec. 6.4, 6.7 and 8)\n");
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
    ok = ok && worst < kPstSelfTol;
    std::printf("  [0] the gradient sidecar against its golden (relative): restrictions %.2e "
                "(%s), h^eff %.2e, f's diagonal %.2e, (O2) %.2e, f over the whole range %.2e "
                "(< %.0e: %s)\n",
                self.blocks, self.worstBlock.c_str(), self.heff, self.fock, self.eps,
                self.fockFull, kPstSelfTol, worst < kPstSelfTol ? "ok" : "FAILED");
  }
  const nevpt2::OrbitalEnergyResponse wResp = nevpt2::orbitalEnergyResponse(
      gg.ncore, gg.nact, gg.nvirt, gg.get("g_xa_cc"), gg.get("g_xc_ca"), gg.get("g_xa_vv"),
      gg.get("g_xv_va"));
  const nevpt2::DeviceTensor hbar =
      NEVPT2_TRY(nevpt2::pcDigestAdjointDevice(opt, ci, g.nelecA, g.nelecB, ints, dm1, dm2,
                                               built.dm3, built.f3ac, built.f3ca, res));
  // The CI directions: PC's mixed check's non-lifting kinds and the state.
  const nevpt2::HostDm12Builder builder =
      NEVPT2_TRY(nevpt2::HostDm12Builder::create(g.ncas, g.nelecA, g.nelecB));
  std::vector<std::pair<std::string, nevpt2::Tensor>> us;
  us.emplace_back(std::format("Epq 0x{:x}", kPstSeedBase + 3),
                  pkcEpqDirection(builder, ci, g.get("dm1"), kPstSeedBase + 3));
  us.emplace_back(std::format("rot 0x{:x}", kPstSeedBase + 4),
                  pkcRotDirection(builder, ci, kPstSeedBase + 4));
  us.emplace_back("c", ci);
  std::printf("  u: %s (sum f_pq E_pq |c>, f on the state's symmetry mask), %s (f "
              "antisymmetric over every active pair) and the state; the orbital energies "
              "respond to kappa, to Idot and to c (FULL)\n",
              us[0].first.c_str(), us[1].first.c_str());

  std::vector<PstCoordinate> coords;
  for (std::int64_t k = 0; k < pg.ncoord(); ++k) {
    PstCoordinate c = NEVPT2_TRY(
        pstCoordinate(g, gg, pg, k, ints, gen, rdm, wResp, hbar, us, opt, sigma, res));
    ok = ok && c.ok;
    coords.push_back(std::move(c));
  }

  // The translational sum rule, per Cartesian axis: printed, not gated.
  static constexpr std::string_view kAxis = "xyz";
  for (std::int64_t axis = 0; axis < 3; ++axis) {
    nevpt2::Tensor sy = coords[static_cast<std::size_t>(axis)].kappaDot;
    nevpt2::Tensor sc = coords[static_cast<std::size_t>(axis)].ciDot;
    double my = stMaxAbs(sy), mc = stMaxAbs(sc);
    for (std::int64_t a = 1; a < pg.natm; ++a) {
      const PstCoordinate& c = coords[static_cast<std::size_t>(3 * a + axis)];
      sy = stAxpy(sy, c.kappaDot, 1.0);
      sc = stAxpy(sc, c.ciDot, 1.0);
      my = std::max(my, stMaxAbs(c.kappaDot));
      mc = std::max(mc, stMaxAbs(c.ciDot));
    }
    std::printf("\n  the translational sum rule along %c: max |sum_A Ydot_A| = %.2e (of %.3e), "
                "max |sum_A ydot_A| = %.2e (of %.3e)\n",
                kAxis[static_cast<std::size_t>(axis)], stMaxAbs(sy), my, stMaxAbs(sc), mc);
  }

  ok = ok && !coords.empty();
  std::printf("\n%s: the PC source terms' tangent along the integral perturbation agrees with "
              "itself by duality to %.0e -- Edot(Idot) three ways, every probe's two modes, and "
              "<Ydot, kappa> + <ydot, u> against d^2E_pc/dh dtheta forward over forward -- and "
              "the ladders of the gated Y_pc and three-term y_pc at I +- h Idot converge to it, "
              "on %lld of %lld coordinates (orbital energies FULL)%s\n",
              ok ? "PASS" : "FAIL", kPstTol, static_cast<long long>(coords.size()),
              static_cast<long long>(pg.ncoord()), fitted ? " (density-fitted)" : "");
  return ok;
}
