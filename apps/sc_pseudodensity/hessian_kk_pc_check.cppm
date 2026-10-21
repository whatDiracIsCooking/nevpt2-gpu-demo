// nevpt2.app.sc_pseudodensity:hessian_kk_pc_check -- --pc --sc-hessian-kk:
// the second of PC-NEVPT2's orbital Hessian block's two terms, the solve's
// sandwich I'[u]^T Ehat''_pc I'[v] (docs/hessian-theory.md, Sec. 6.4-6.8;
// #202), checked by duality, and the whole block Eddot_kk(u, v) =
// <Ehat'_pc, I''> + the sandwich checked against ladders of the PC total
// itself along exp(s u + t v) and, step-free, against #201's H_kc.
// An interface partition that exports nothing, as :hessian_kk_sandwich_check is.
//
// WHAT IS DIFFERENT FROM SC's --sc-hessian-kk. PC's metric S is RDM-only and
// does not move along any kappa, so the sandwich has no projector, rotation
// or lift term (Sec. 6.5), and its finish is one bilinear form per tuple --
// the two-leg cross term of Sec. 6.4 -- not SC's K3-K5 split. PC's finish
// divides by no norm, so there is no empty-perturber term, no rho_t refusal
// and no guarded-tuple census (Sec. 6.6): every pair SC refuses is formed here,
// and its ladders must converge. The tables print the term's shares instead:
// each leg's Dyall block alone and its digests' share alone.
//
// THE ORBITAL ENERGIES' TWO MODES. EULER: they respond to kappa (Delta'_w per
// tuple out of the sidecar's deps, the curvature's eps term at second order)
// -- the functional the ladders displace. FROZEN: held fixed on both legs and
// in the curvature -- the functional the gauge against H_kc is taken in, since
// PC's H_kc refuses the orbital energies responding to the CI vector.
//
// THE CHECKS, in the order they print:
//
//   1  THE MERGE: each walk's own Edot(u), Edot(v) -- every tuple's K' carrying
//      the digests' share, Delta' in where the orbital energies respond --
//      against the first-order <k, Y_pc> with the digests' term (--pc
//      --sc-hessian-kk-int's gated adjoint, hbar_pc in; less its eps term in
//      FROZEN mode), for every direction the pairs use, both modes.
//   2  THE DUALITY: the forward finish against the reverse pairing
//      <(dE/dslab)'[v], x'[u]> + <(dE/dK)'[v], K'[u]> + R'[v] . Delta'[u], along
//      every pair, both modes -- exact arithmetic, the tree's 1e-7 -- and the
//      swap (u and v exchanged) against its original.
//   3  THE INTRA-SPACE ZEROS: along a core-core or virtual-virtual direction no
//      active integral moves, so h2e' and the digests' tangent f3' are
//      exactly-zero tensors and every share that direction's digests carry
//      vanishes IDENTICALLY; its Dyall block's share is round-off (h1e'
//      carries the core dressing's cancelling pair) -- while the term does not
//      vanish (the slabs move).
//   4  THE DIGESTS AGAINST THE PLAIN DIGEST PATH: the share both legs'
//      digests carry against the energy stage's own PC Hessian
//      (pcHessianDevice) at the digests' tangents f3'[u], f3'[v] alone; and
//      where the curvature is formed, the whole digests' contribution to
//      Eddot_kk, that plus the curvature's digests' term reverse
//      (<hbar_pc, h2e''>), against the same plus the plain build at h2e''.
//   5  THE LADDERS (EULER): E_pc(U) along exp(s u + t v), the four corners each
//      their own PC energy (the digests rebuilt per corner), against Eddot_kk,
//      and E - L against the sandwich alone, along every block-diagonal pair
//      -- virtual x virtual and core x core included, which SC refuses or
//      closes with K6. --sc-hessian-kk's ladders, unchanged: the 1e-6
//      relative gate on the Richardson value and #190's floor-aware rule.
//   6  THE GAUGE AGAINST H_kc (FROZEN, step-free; Sec. 6.8): w active-active,
//      any v, Eddot_kk(v, w) + (1/2) <[v, w], Y_pc> == Eddot_kc(v, what c),
//      what = sum_pq w_pq E_pq -- the assembled block against #201's mixed
//      block, whose rotation term the right side carries and the left does
//      not. Two separate PC walks: the tree's 1e-7.
//   7  THE REFUSALS: on a conventional case Eddot_kk along full x full has no
//      K2 and is Unsupported before any device work (Sec. 6.7); on a fitted
//      case it is formed (PC has no rho_t refusal) and its sandwich is check
//      2's.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries. A malformed direction or a
// sidecar that does not fit the golden is the caller's input and propagates
// unchanged to main, which reports it once.
#include "error_handling/error_macros.h"

export module nevpt2.app.sc_pseudodensity:hessian_kk_pc_check;

import std;
import nevpt2.gradient;
import nevpt2.kappa_rotation;
import nevpt2.golden;
import nevpt2.rdm_host_dm;
// --kappa-full's sidecar upload and its direction helpers.
import :kappa_full_check;
// --sc-hessian-kk-int's displaced integral sets, first-order pairings and
// ladder constants; --sc-hessian-kk's ladders; --sc-hessian-kc's setup and
// --pc --sc-hessian-kc's one-body operator on a CI vector.
import :hessian_kk_check;
import :hessian_kk_sandwich_check;
import :hessian_kc_check;
import :hessian_kc_pc_check;

// Every exact identity between two computations, and the gauge between two
// separate PC walks: the tree's 1e-7 (docs/testing.md, "PC's orbital block's
// sandwich", records what is measured).
constexpr double kPkkTol = 1e-7;
// An intra-space direction's Dyall-block share: h2e' is an exactly-zero
// tensor there, but h1e' carries the core dressing's response, whose two
// promotions of a tied core pair cancel only to round-off -- absolute, decades
// above what is measured and decades below any real term (as --sc-hessian-kk
// holds K5's half and --pc --sc-hessian-kc its rotation term).
constexpr double kPkkZeroTol = 1e-13;
// Seeds, fixed so a run is reproducible, distinct from every other check's.
constexpr std::uint64_t kPkkSeedBase = 0x5eed'0202u;
// The ladders' first step: h = 2^-4, --pc --sc-hessian-kk-int's (kKkFdFirstExp),
// not SC's 2^-2. A quarter-radian rotation can carry a PC class out of the
// region where its solve is defined: at 2^-2 along active x active on N2
// CAS(4,4) a corner's denominator lambda_k + Delta_t goes negative and the
// corner's PC energy is refused. The gate and the floor-aware shape rule are
// --sc-hessian-kk's, unchanged; the finest step stays 2^-16.
constexpr int kPkkFdFirstExp = kKkFdFirstExp;

// One direction with its name, and its first-order pairings.
struct PkkDir {
  std::string name;
  nevpt2::Tensor k;
  double y = 0.0;        // <k, Y_pc>, the digests' term in (EULER)
  double yFrozen = 0.0;  // the same less the orbital energies' share (FROZEN)
};

// One pair, by index into the direction list.
struct PkkPair {
  std::string name;
  std::size_t u, v;
  bool blockDiagonal = false;  // displaceable from stored blocks: the ladders run
  bool oneActive = false;      // one member active-active: K2 from the sidecar
  bool intraU = false, intraV = false;  // a core-core or virtual-virtual member
};

const char* pkkModeName(bool epsKappa) {
  return epsKappa ? "EULER: the orbital energies respond to kappa"
                  : "FROZEN: the orbital energies held fixed";
}

// One pair's sandwich, as a table row.
void pkkPrintPair(const std::string& name, const nevpt2::PcOrbitalSandwichResult& r) {
  const nevpt2::PcSandwichTerms& t = r.total;
  std::printf("    %-30s %22.14e %22.14e %9.2e %12.4e %12.4e %12.4e %12.4e %12.4e\n",
              name.c_str(), r.forward, r.reverse, std::fabs(r.forward - r.reverse), t.kU, t.kV,
              t.digestU, t.digestV, t.digestDigest);
}

nevpt2::Result<bool> checkPcHessianKk(const nevpt2::GoldenFile& g, const nevpt2::GradGoldFile& gg,
                                      const nevpt2::NevptIntegralsDevice& ints,
                                      const nevpt2::DeviceTensor& dm1,
                                      const nevpt2::DeviceTensor& dm2,
                                      const nevpt2::RdmBuildResult& built,
                                      const nevpt2::RdmBuildOptions& opt,
                                      const nevpt2::DeviceResources& res) {
  const nevpt2::RdmTensors rdm{dm1, dm2, built.dm3, built.f3ac, built.f3ca};
  const nevpt2::Tensor& ci = g.get("ci");
  const bool fitted = gg.densityFitted();
  const nevpt2::GeneralIntegralsDevice gen = NEVPT2_TRY(generalIntegrals(g, gg, res));
  const std::int64_t nc = gg.ncore, na = gg.nact, nv = gg.nvirt, nmo = gg.nmo;
  std::printf("\n--pc --sc-hessian-kk: the PC orbital Hessian block Eddot_kk(u, v) = <Ehat'_pc, "
              "I''> + I'[u]^T Ehat''_pc I'[v] (docs/hessian-theory.md, Sec. 6.4)%s\n",
              fitted ? " -- density-fitted" : "");
  std::printf("  the sandwich = sum_t -2 nu gammahat_u^T D^-1 gammahat_v, gammahat = yhat' - "
              "Ahat' xhat, the digests' share merged in K' per class;\n  S does not move along "
              "kappa (Sec. 6.5): no rotation, projector or lift term, and no guarded tuple "
              "(Sec. 6.6); both modes\n");
  bool ok = true;

  const nevpt2::DeviceTensor hbar =
      NEVPT2_TRY(nevpt2::pcDigestAdjointDevice(opt, ci, g.nelecA, g.nelecB, ints, dm1, dm2,
                                               built.dm3, built.f3ac, built.f3ca, res));
  const nevpt2::PcOrbitalSandwichSetup euler{ints, gen, rdm, ci, g.nelecA, g.nelecB, opt, true};
  const nevpt2::PcOrbitalSandwichSetup frozen{ints, gen, rdm, ci, g.nelecA, g.nelecB, opt, false};
  const nevpt2::ScOrbitalCurvatureSetup cs{ints, gen, rdm, &hbar, &ci, g.nelecA, g.nelecB, opt};

  // The directions, and their first-order pairings <k, Y_pc> in one walk.
  const nevpt2::Tensor kAA = nevpt2::antisymmetricKappa(na, kPkkSeedBase + 1);
  std::vector<PkkDir> dirs{
      {"active", embedBlock(kAA, nc, nmo)},
      {"active'", embedBlock(nevpt2::antisymmetricKappa(na, kPkkSeedBase + 2), nc, nmo)},
      {"full", nevpt2::antisymmetricKappa(nmo, kPkkSeedBase + 3)},
      {"full'", nevpt2::antisymmetricKappa(nmo, kPkkSeedBase + 4)},
  };
  const std::size_t iAct = 0, iAct2 = 1, iFull = 2, iFull2 = 3;
  std::optional<std::size_t> iCore, iCore2, iVirt, iVirt2;
  if (nc >= 2) {
    iCore = dirs.size();
    dirs.push_back({"core", embedBlock(nevpt2::antisymmetricKappa(nc, kPkkSeedBase + 5), 0, nmo)});
    iCore2 = dirs.size();
    dirs.push_back({"core'", embedBlock(nevpt2::antisymmetricKappa(nc, kPkkSeedBase + 6), 0, nmo)});
  }
  if (nv >= 2) {
    iVirt = dirs.size();
    dirs.push_back(
        {"virtual", embedBlock(nevpt2::antisymmetricKappa(nv, kPkkSeedBase + 7), nc + na, nmo)});
    iVirt2 = dirs.size();
    dirs.push_back(
        {"virtual'", embedBlock(nevpt2::antisymmetricKappa(nv, kPkkSeedBase + 8), nc + na, nmo)});
  }
  {
    std::vector<nevpt2::ScRotationRequest> req;
    for (const PkkDir& d : dirs) req.push_back(nevpt2::ScRotationRequest{d.k, false});
    const std::vector<nevpt2::ScFullRotationPairing> first =
        NEVPT2_TRY(kkFirstOrderPairings(true, ints, gen, rdm, hbar, req, res));
    for (std::size_t k = 0; k < dirs.size(); ++k) {
      dirs[k].y = first[k].adjoint;
      dirs[k].yFrozen = first[k].adjoint - first[k].eps;
    }
  }

  // SC's pair list (--sc-hessian-kk), so the pairs SC refuses for rho_t --
  // full x full and virtual x virtual on N2 -- run here too.
  std::vector<PkkPair> pairs{
      {"active x full", iAct, iFull, false, true},
      {"full x active (the swap)", iFull, iAct, false, true},
      {"full x full", iFull, iFull2, false, false},
      {"active x active", iAct, iAct2, true, true},
  };
  if (iCore) pairs.push_back({"active x core", iAct, *iCore, true, true, false, true});
  if (iVirt) pairs.push_back({"active x virtual", iAct, *iVirt, true, true, false, true});
  if (iCore && iVirt) pairs.push_back({"core x virtual", *iCore, *iVirt, true, false, true, true});
  if (iCore) pairs.push_back({"core x core", *iCore, *iCore2, true, false, true, true});
  if (iVirt) pairs.push_back({"virtual x virtual", *iVirt, *iVirt2, true, false, true, true});

  // The sandwich along every pair, both modes, one call each, the reverse
  // pairing in both and the digests' reference (EULER) where both directions
  // move h2e. PC refuses no pair for its empty perturbers (Sec. 6.6): any
  // error here is a FAIL, reported once at main.
  std::array<std::vector<nevpt2::PcOrbitalSandwichResult>, 2> sand;  // [EULER, FROZEN]
  const auto t0 = std::chrono::steady_clock::now();
  for (const PkkPair& p : pairs) {
    const bool digests = !p.intraU && !p.intraV;
    sand[0].push_back(NEVPT2_TRY(nevpt2::pcOrbitalSandwichDevice(
        euler, {nevpt2::PcOrbitalSandwichRequest{dirs[p.u].k, dirs[p.v].k, true, digests}},
        res))[0]);
    sand[1].push_back(NEVPT2_TRY(nevpt2::pcOrbitalSandwichDevice(
        frozen, {nevpt2::PcOrbitalSandwichRequest{dirs[p.u].k, dirs[p.v].k, true, false}},
        res))[0]);
  }
  const double sandSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  // --- 1. the merge -----------------------------------------------------------------
  {
    double worst = 0.0;
    std::printf("\n  [1] the merge: each walk's own Edot against the first-order <k, Y_pc> with "
                "the digests' term\n");
    for (int mode = 0; mode < 2; ++mode) {
      std::printf("  mode %s\n", pkkModeName(mode == 0));
      for (std::size_t k = 0; k < pairs.size(); ++k) {
        const nevpt2::PcOrbitalSandwichResult& r = sand[static_cast<std::size_t>(mode)][k];
        const double yu = mode == 0 ? dirs[pairs[k].u].y : dirs[pairs[k].u].yFrozen;
        const double yv = mode == 0 ? dirs[pairs[k].v].y : dirs[pairs[k].v].yFrozen;
        const double du = std::fabs(r.total.edotU - yu), dv = std::fabs(r.total.edotV - yv);
        worst = std::max({worst, du, dv});
        std::printf("    %-30s Edot(u) %22.14e |delta| %.2e   Edot(v) %22.14e |delta| %.2e\n",
                    pairs[k].name.c_str(), r.total.edotU, du, r.total.edotV, dv);
      }
    }
    ok = ok && worst < kPkkTol;
    std::printf("  worst |delta| = %.2e (< %.0e: %s)\n", worst, kPkkTol,
                worst < kPkkTol ? "ok" : "FAILED");
  }

  // --- 2. the duality ---------------------------------------------------------------
  {
    std::printf("\n  [2] the duality (%zu pairs, both modes, in %.2fs): forward against the "
                "reverse pairing\n",
                pairs.size(), sandSeconds);
    double worst = 0.0, swap = 0.0;
    for (int mode = 0; mode < 2; ++mode) {
      const std::vector<nevpt2::PcOrbitalSandwichResult>& s = sand[static_cast<std::size_t>(mode)];
      std::printf("  mode %s\n", pkkModeName(mode == 0));
      std::printf("    %-30s %22s %22s %9s %12s %12s %12s %12s %12s\n", "pair", "forward",
                  "reverse", "|delta|", "K'u alone", "K'v alone", "f3'u alone", "f3'v alone",
                  "f3' x f3'");
      for (std::size_t k = 0; k < pairs.size(); ++k) {
        pkkPrintPair(pairs[k].name, s[k]);
        worst = std::max(worst, std::fabs(s[k].forward - s[k].reverse));
      }
      swap = std::max(swap, std::fabs(s[0].forward - s[1].forward));
    }
    const bool good = worst < kPkkTol && swap < kPkkTol;
    ok = ok && good;
    std::printf("  duality, worst |delta| = %.2e; the swap against its original, worst |delta| = "
                "%.2e (both < %.0e: %s)\n",
                worst, swap, kPkkTol, good ? "ok" : "FAILED");
  }

  // --- 3. the intra-space zeros -----------------------------------------------------
  {
    std::printf("\n  [3] the intra-space directions (EULER): h2e' and f3' exactly zero, so the "
                "digests' shares vanish identically\n");
    bool good = true;
    std::size_t seen = 0;
    for (std::size_t k = 0; k < pairs.size(); ++k) {
      const PkkPair& p = pairs[k];
      if (!p.intraU && !p.intraV) continue;
      ++seen;
      const nevpt2::PcOrbitalSandwichResult& r = sand[0][k];
      const nevpt2::PcSandwichTerms& t = r.total;
      bool zeros = t.digestDigest == 0.0;
      if (p.intraU)
        zeros = zeros && r.h2eDotMaxU == 0.0 && r.f3DotMaxU == 0.0 && t.digestU == 0.0 &&
                std::fabs(t.kU) < kPkkZeroTol;
      if (p.intraV)
        zeros = zeros && r.h2eDotMaxV == 0.0 && r.f3DotMaxV == 0.0 && t.digestV == 0.0 &&
                std::fabs(t.kV) < kPkkZeroTol;
      // ... while the term does not: the slabs move along every direction.
      const bool alive = std::fabs(r.forward) > 1e3 * kPkkZeroTol;
      good = good && zeros && alive;
      std::printf("    %-30s max|h2e'| %.1e / %.1e, max|f3'| %.1e / %.1e; the digests' shares "
                  "%.1e / %.1e / %.1e (u / v / both),\n      |K'u alone| %.1e, |K'v alone| %.1e "
                  "(an intra-space leg's < %.0e); the term %.6e: %s\n",
                  p.name.c_str(), r.h2eDotMaxU, r.h2eDotMaxV, r.f3DotMaxU, r.f3DotMaxV, t.digestU,
                  t.digestV, t.digestDigest, std::fabs(t.kU), std::fabs(t.kV), kPkkZeroTol,
                  r.forward, zeros && alive ? "ok" : "FAILED");
    }
    good = good && seen > 0;
    ok = ok && good;
    std::printf("  %s\n", good ? "ok" : "FAILED");
  }

  // The whole block where the curvature can be formed: every pair whose K2 the
  // data serves. Its own walks, and the digests' forward term at h2e''.
  std::vector<nevpt2::ScOrbitalCurvatureRequest> creq;
  std::vector<std::size_t> cwhich;
  for (std::size_t k = 0; k < pairs.size(); ++k)
    if (fitted || pairs[k].oneActive) {
      creq.push_back(
          nevpt2::ScOrbitalCurvatureRequest{dirs[pairs[k].u].k, dirs[pairs[k].v].k, true});
      cwhich.push_back(k);
    }
  const std::vector<nevpt2::ScOrbitalCurvatureResult> curv =
      NEVPT2_TRY(nevpt2::pcOrbitalCurvatureDevice(cs, creq, res));

  // --- 4. the digests against the plain digest path --------------------------------
  {
    std::printf("\n  [4] the digests alone: the operator's shares against the energy stage at the "
                "plain digest path's builds\n");
    std::printf("    %-30s %22s %22s %9s %22s %22s %9s\n", "pair", "f3' x f3'",
                "pcHessianDevice", "|delta|", "digests alone, operator", "energy stage",
                "|delta|");
    double worst = 0.0;
    std::size_t seen = 0;
    for (std::size_t k = 0; k < pairs.size(); ++k) {
      const nevpt2::PcOrbitalSandwichResult& r = sand[0][k];
      if (!r.hasDigestReference) continue;
      ++seen;
      const double d1 = std::fabs(r.total.digestDigest - r.digestReference);
      worst = std::max(worst, d1);
      std::optional<std::size_t> c;
      for (std::size_t i = 0; i < cwhich.size(); ++i)
        if (cwhich[i] == k) c = i;
      if (!c) {
        std::printf("    %-30s %22.14e %22.14e %9.2e %22s %22s %9s\n", pairs[k].name.c_str(),
                    r.total.digestDigest, r.digestReference, d1, "(no K2 here)", "", "");
        continue;
      }
      const nevpt2::ScOrbitalCurvatureResult& q = curv[*c];
      const double walk = r.total.digestDigest + q.k1.digest + q.k2.digest;
      const double plain = r.digestReference + q.k1.digestForward + q.k2.digestForward;
      const double d2 = std::fabs(walk - plain);
      worst = std::max(worst, d2);
      std::printf("    %-30s %22.14e %22.14e %9.2e %22.14e %22.14e %9.2e\n", pairs[k].name.c_str(),
                  r.total.digestDigest, r.digestReference, d1, walk, plain, d2);
    }
    const bool good = seen > 0 && worst < kPkkTol;
    ok = ok && good;
    std::printf("  worst |delta| = %.2e (< %.0e: %s)\n", worst, kPkkTol, good ? "ok" : "FAILED");
  }

  // --- 5. the ladders ---------------------------------------------------------------
  const KkFockParts fock = kkFockParts(gen, g.get("dm1"), res);
  std::printf("\n  Eddot_kk(u, v) = <Ehat'_pc, I''> + the sandwich (EULER), where the curvature is "
              "formed:\n");
  for (std::size_t c = 0; c < cwhich.size(); ++c)
    std::printf("    %-30s %22.14e = %22.14e + %22.14e\n", pairs[cwhich[c]].name.c_str(),
                curv[c].value() + sand[0][cwhich[c]].forward, curv[c].value(),
                sand[0][cwhich[c]].forward);
  {
    const double swap = std::fabs(curv[0].value() + sand[0][0].forward -
                                  (curv[1].value() + sand[0][1].forward));
    ok = ok && swap < kPkkTol;
    std::printf("  Eddot_kk's swap, (full, active) against (active, full): |delta| %.2e (< %.0e: "
                "%s)\n",
                swap, kPkkTol, swap < kPkkTol ? "ok" : "FAILED");
  }
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    const PkkPair& p = pairs[k];
    if (!p.blockDiagonal) continue;
    std::optional<double> want;
    for (std::size_t c = 0; c < cwhich.size(); ++c)
      if (cwhich[c] == k) want = curv[c].value() + sand[0][k].forward;
    const double scale = std::max({want ? std::fabs(*want) : std::fabs(sand[0][k].forward),
                                   std::fabs(dirs[p.u].y), std::fabs(dirs[p.v].y)});
    std::printf("\n  [5] the ladders along exp(s u + t v), %s (<u, Y_pc> = %.6e, <v, Y_pc> = "
                "%.6e)%s:\n",
                p.name.c_str(), dirs[p.u].y, dirs[p.v].y,
                want ? "" : " -- the sandwich's alone: K2, so Eddot_kk, is refused here");
    // A corner the PC solve refuses (a class's gap or denominator check) has
    // no energy: the ladder FAILS, says why, and the run goes on to the
    // other checks -- never a corner with a zero in it.
    const nevpt2::Result<bool> fd =
        swLadders(true, kPkkFdFirstExp, g, fock, dirs[p.u].k, dirs[p.v].k, want,
                  sand[0][k].forward, scale, ints, rdm, hbar, opt, nc, na, nv, res);
    if (!fd) {
      if (fd.error().kind != nevpt2::ErrorKind::Numerical) return std::unexpected(fd.error());
      std::printf("  the ladder has no value -- NOT CONVERGED: %s\n", fd.error().message.c_str());
      ok = false;
      continue;
    }
    ok = ok && *fd;
  }

  // --- 6. the gauge against H_kc (FROZEN) -------------------------------------------
  {
    std::printf("\n  [6] the gauge against H_kc, %s, step-free: w = active (active-active), "
                "what = sum_pq w_pq E_pq,\n      Eddot_kk(v, w) + (1/2) <[v, w], Y_pc> == "
                "Eddot_kc(v, what c)\n",
                pkkModeName(false));
    const nevpt2::HostDm12Builder builder =
        NEVPT2_TRY(nevpt2::HostDm12Builder::create(g.ncas, g.nelecA, g.nelecB));
    const nevpt2::Tensor wc = pkcOneBody(builder, kAA, ci);
    const nevpt2::ScMixedHessianSetup kc =
        kcSetup(g, ints, &gen, rdm, opt, /*epsKappa=*/false, /*epsCi=*/false, nullptr);
    const nevpt2::Tensor& w = dirs[iAct].k;
    std::vector<std::size_t> vs{iFull, iAct2};
    if (iCore) vs.push_back(*iCore);
    if (iVirt) vs.push_back(*iVirt);
    std::printf("    %-10s %22s %22s %22s %22s %9s %13s\n", "v", "Eddot_kk(v, w)",
                "(1/2) <[v, w], Y_pc>", "Eddot_kc(v, what c)", "of it, rotation", "|delta|",
                "Eddot_kk's sandwich");
    double worst = 0.0;
    for (const std::size_t iv : vs) {
      const nevpt2::Tensor& v = dirs[iv].k;
      const nevpt2::PcOrbitalHessianResult h = NEVPT2_TRY(nevpt2::pcOrbitalHessianDevice(
          cs, {nevpt2::ScOrbitalCurvatureRequest{v, w}}, /*epsKappa=*/false, res))[0];
      nevpt2::Tensor comm({nmo, nmo});
      const nevpt2::Tensor vw = kkMatmul(v, w), wv = kkMatmul(w, v);
      for (std::int64_t i = 0; i < comm.size(); ++i) comm.flatRef(i) = vw.flat(i) - wv.flat(i);
      const std::vector<nevpt2::ScFullRotationPairing> yc = NEVPT2_TRY(kkFirstOrderPairings(
          true, ints, gen, rdm, hbar, {nevpt2::ScRotationRequest{comm, false}}, res));
      const double half = 0.5 * (yc[0].adjoint - yc[0].eps);
      const nevpt2::PcMixedHessianResult m =
          NEVPT2_TRY(nevpt2::pcMixedHessianDevice(kc, v, {wc}, res))[0];
      const double d = std::fabs(h.value() + half - m.eddot);
      worst = std::max(worst, d);
      std::printf("    %-10s %22.14e %22.14e %22.14e %22.14e %9.2e %13.5e\n",
                  dirs[iv].name.c_str(), h.value(), half, m.eddot, m.rotation, d,
                  h.sandwich.forward);
    }
    ok = ok && worst < kPkkTol;
    std::printf("  the gauge, worst |delta| = %.2e (< %.0e: %s)\n", worst, kPkkTol,
                worst < kPkkTol ? "ok" : "FAILED");
  }

  // --- 7. the assembled block's refusals --------------------------------------------
  {
    // Conventional: the curvature's K2 refusal comes first, before any device
    // work. Fitted: K2 is formed, and PC refuses nothing for its empty
    // perturbers -- the block is formed, its sandwich check 2's.
    const nevpt2::Result<std::vector<nevpt2::PcOrbitalHessianResult>> r =
        nevpt2::pcOrbitalHessianDevice(
            cs, {nevpt2::ScOrbitalCurvatureRequest{dirs[iFull].k, dirs[iFull2].k}}, true, res);
    bool good = false;
    std::string what;
    if (!r) {
      good = !fitted && r.error().kind == nevpt2::ErrorKind::Unsupported;
      what = std::format("refused, {}: \"{}\"",
                         r.error().kind == nevpt2::ErrorKind::Unsupported ? "Unsupported"
                                                                          : "WRONG KIND",
                         r.error().message);
    } else {
      const double gap = std::fabs(r->front().sandwich.forward - sand[0][2].forward);
      good = fitted && gap < kPkkTol;
      what = std::format("{:.14e}, its sandwich against check 2's |delta| {:.2e}",
                         r->front().value(), gap);
    }
    ok = ok && good;
    std::printf("\n  [7] Eddot_kk along full x full on a %s case: %s: %s\n",
                fitted ? "density-fitted" : "conventional", what.c_str(), good ? "ok" : "FAILED");
  }

  std::printf("\n%s: the PC orbital Hessian's sandwich I'^T Ehat''_pc I' agrees with itself by "
              "duality to %.0e in both orbital-energy modes, its digests' share is the "
              "first-order term's and its curvature the energy stage's, the intra-space zeros "
              "hold, the ladders of the PC total along exp(s u + t v) converge to Eddot_kk = "
              "<Ehat'_pc, I''> + the sandwich on every block-diagonal pair, and the gauge "
              "against H_kc holds; no pair refused for its empty perturbers (Sec. 6.6)%s\n",
              ok ? "PASS" : "FAIL", kPkkTol, fitted ? " (density-fitted)" : "");
  return ok;
}
