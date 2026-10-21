// nevpt2.app.sc_pseudodensity:hessian_kk_sandwich_check -- --sc-hessian-kk:
// the second of the orbital Hessian block's two terms, the energy's own
// curvature in its inputs I'[u]^T Ehat'' I'[v] = K3 + K4 + K5 + K6
// (docs/hessian-theory.md, Sec. 5.1 and Sec. 5.6; #158, #190), checked by
// duality, and the whole
// block Eddot_kk(u, v) = <Ehat', I''> + the sandwich checked against ladders
// of the SC total itself along exp(s u + t v).
// An interface partition that exports nothing, as :hessian_kk_check is.
//
// WHAT IS BEING DIFFERENTIATED. The SC total E(U) at fixed CI vector, U =
// exp(s u + t v): every stored integral block rotated, the orbital energies
// the rotated generalized Fock's diagonal, and the f3ac/f3ca digests REBUILT
// at the rotated h2e. Its mixed second derivative is the whole orbital block,
// the curvature's K1 + K2 (#154, scOrbitalCurvatureDevice) plus this term; the
// curvature alone is the mixed second derivative of the fixed-pseudodensity
// pairing L(U) = <Ehat'_0, I(U)>, so E(U) - L(U) differenced is this term
// alone.
//
// THE CHECKS, in the order they print:
//
//   0  THE GUARDED TUPLES (docs/hessian-theory.md, Sec. 5.6; #190): the
//      tuples the vanishing-norm guard skips and the integrals leave empty
//      add K6 where both directions populate them. Sijrs (H = 0) always
//      does, at its own P = -1/Delta, which is what closes core x core and
//      core x virtual. An H class does where its tables are proportional
//      (rho_t at round-off): `virtual x (2 virtual + core)`, whose core part
//      populates no H class's empty tuple, is the proportional pair, and its
//      H-class K6 must be nonzero. Where they are not, there is no Hessian
//      along the pair and it is REFUSED, Numerical, naming the class, the
//      tuple count and the worst rho_t: on N2 exactly full x full and
//      virtual x virtual. Printed: each computed pair's K6 per class, the
//      worst rho_t and the population floor's separation over the computed
//      pairs, and each refusal's message.
//   1  THE MERGE: the walk's own Edot(u), Edot(v) -- every tuple's Hdot
//      carrying the digests' share -- against the first-order <u, Y>, <v, Y>
//      with the digests' term (--kappa-full's gated adjoint, hbar in), for
//      every direction the pairs use.
//   2  THE DUALITY: the forward finish K3 + ... + K6 against the reverse
//      pairing <Ddot[v], X'[u]> + <Mdot_G[v], G'[u]> + Rdot[v] . Ddot[u],
//      along active x full and its swap and every computed block-diagonal
//      pair -- exact arithmetic, the tree's 1e-7 -- K6 against its own
//      reverse half on its own, and the swap against its original (the walk
//      with u and v exchanged).
//   3  THE INTRA-SPACE ZEROS: along a core-core or virtual-virtual direction
//      no active integral moves, so h2e' and the digests' tangent f3' are
//      exactly-zero tensors and the digests' shares of K3 and of K5 with
//      that direction's Dyall block vanish IDENTICALLY; that K5 half itself
//      is round-off (h1e' carries the core dressing's cancelling pair) --
//      while K3 and K4 do not vanish.
//   4  THE DIGESTS AGAINST THE PLAIN DIGEST PATH: the digests' own curvature
//      (K3's digest x digest share, out of the walk's merge) against the
//      energy stage's own Hessian walk (energiesHessianDevice) at the digests'
//      tangents f3'[u], f3'[v] -- the plain build at h2e'[u], h2e'[v] -- and
//      the whole digests' contribution to Eddot_kk, that plus the
//      curvature's digests' term reverse (<hbar, h2e''>), against the same
//      plus the plain build run at the second-order dotted integrals h2e''
//      (the curvature's forward digests' term).
//   5  THE LADDERS: E(U) along exp(s u + t v), the four corners (+-h, +-h)
//      each its own energy (the digests rebuilt per corner), against Eddot_kk,
//      and E(U) - L(U) against the sandwich alone, along every computed
//      block-diagonal pair. Each is verified by the raw differences' halvings
//      above the round-off floor (the shape rule below) and gated on its
//      Richardson extrapolation
//      (4 D(h/2) - D(h)) / 3 at the tree's 1e-6 relative: the raw ladder of a
//      total energy reaches its round-off floor, ~ eps |E| / h^2, before its
//      O(h^2) error is that small when Eddot is small against |E|.
//   6  THE REFUSALS: on a conventional case Eddot_kk along full x full has no
//      K2 and is Unsupported before any device work (Sec. 9.1); on a fitted
//      case the guarded tuples refuse it, Numerical.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries. A malformed direction or a
// sidecar that does not fit the golden is the caller's input and propagates
// unchanged to main, which reports it once.
#include "error_handling/error_macros.h"

export module nevpt2.app.sc_pseudodensity:hessian_kk_sandwich_check;

import std;
import nevpt2.gradient;
import nevpt2.kappa_rotation;
import nevpt2.golden;
// --kappa-full's sidecar upload and its direction helpers.
import :kappa_full_check;
// --sc-hessian-kk-int's block-diagonal rotations, displaced integral sets,
// orbital energies and first-order pairing: the ladders here displace
// exactly as that check's do.
import :hessian_kk_check;

// Every exact identity's gate: two computations of one number, the tree's
// usual 1e-7 (docs/testing.md, "The orbital block's sandwich", records what
// is measured).
constexpr double kSwTol = 1e-7;
// K5's half against an intra-space direction's Dyall block: h2e' is an
// exactly-zero tensor there, but h1e' (h^eff's active block) carries the core
// dressing's response, whose two promotions of a tied core pair cancel only
// to round-off -- so that half is round-off, held absolutely to this, decades
// above what is measured and decades below any real term (as --sc-hessian-kc
// holds its C3; docs/testing.md, "The orbital block's sandwich").
constexpr double kSwZeroTol = 1e-13;
// Seeds, fixed so a run is reproducible, distinct from every other check's.
constexpr std::uint64_t kSwSeedBase = 0x5eed'0158u;

// THE LADDERS' SHAPE RULE (#190): FLOOR-AWARE. The value gate is #158's,
// unchanged: the Richardson value's best error relative to max(|target|,
// |<u, Y>|, |<v, Y>|) under kKkFdTol (1e-6). The shape check that verifies
// it differs from --sc-hessian-kk-int's ladders, measured on the pairs K6
// closes (docs/testing.md, "The orbital block's sandwich"):
//
//   * Each rung's round-off floor is eps * max_c |E_c| / h^2, E_c the four
//     corner SC totals that rung differences (energiesDevice(...).total --
//     the energies actually differenced; L is paired from the corners'
//     integral combination, formed before the pairing, so it adds no
//     cancellation of its own). A rung whose error is within kSwFloorMargin
//     (10x) of its floor is EXCLUDED from the ratio test: a ratio measured
//     there is noise -- the k = 11 ratio of core x virtual on CAS(4,4) came
//     out anywhere from 3.07 to 4.41 run to run, which made a fixed-length
//     run flaky.
//   * Above the floor, kSwFdRun (2) consecutive halvings must each cut the
//     error by at least kFourfoldLow (3.2), with NO upper bound. The lower
//     bound is what catches the failure this check exists for: a ladder
//     converging to a different number from its target sits flat (ratio
//     1.00 per halving -- #158's first run of core x core, the polarisation
//     of a pair with no Hessian). A ratio ABOVE 4 while the error falls to
//     the target is a large h^4 term in the energy along the pair, not a
//     bug: core x core on N2 CAS(4,4) halves by 5.9, 8.7, 10.0 on its way to
//     1.6e-9 of the target.
//   * The steps start at h = 2^-2 rather than 2^-4, for every pair: a pair
//     whose truncation error is tiny -- the proportional virtual x (2
//     virtual + core) on CAS(8,8), 8e-6 already at 2^-4 -- otherwise reaches
//     the floor before it shows any halvings. 2^-2 is still a small rotation
//     of a unit-norm kappa (an angle a quarter of a radian at most), and the
//     finest step stays 2^-16.
constexpr int kSwFdFirstExp = 2;
constexpr std::size_t kSwFdSteps = kKkFdLastExp - kSwFdFirstExp + 1;
constexpr double kSwFourfoldHigh = std::numeric_limits<double>::infinity();
constexpr double kSwFloorMargin = 10.0;
constexpr int kSwFdRun = 2;

// One direction with its name, and its first-order pairing.
struct SwDir {
  std::string name;
  nevpt2::Tensor k;
  double y = 0.0;  // <k, Y>, the digests' term in
};

// One pair, by index into the direction list.
struct SwPair {
  std::string name;
  std::size_t u, v;
  bool blockDiagonal = false;  // displaceable from stored blocks: the ladders run
  bool oneActive = false;      // one member active-active: K2 from the sidecar
  bool intraU = false, intraV = false;  // a core-core or virtual-virtual member
};

// Both ladders along one pair: per step h, the four corners' SC totals (the
// digests rebuilt at each corner's h2e) and, combined before the pairing, the
// fixed-pseudodensity pairing L. `want` is Eddot_kk -- absent where the
// curvature is refused, and then E's ladder is printed and not gated --
// `wantSandwich` this term; `scale` both gates' denominator -- the magnitudes
// the differenced values carry. Returns whether the gated ones converge.
// `pc`: the corners are PC totals (pcEnergiesDevice, a refused class a
// Numerical Error) and L is PC's fixed-adjoint pairing (--pc --sc-hessian-kk,
// #202); `hbar` is then PC's. The steps are h = 2^-k for k from `firstExp`
// (SC's kSwFdFirstExp) to kKkFdLastExp.
nevpt2::Result<bool> swLadders(bool pc, int firstExp, const nevpt2::GoldenFile& g,
                               const KkFockParts& fock,
                               const nevpt2::Tensor& u, const nevpt2::Tensor& v,
                               std::optional<double> want,
                               double wantSandwich, double scale,
                               const nevpt2::NevptIntegralsDevice& ints,
                               const nevpt2::RdmTensors& rdm, const nevpt2::DeviceTensor& hbar,
                               const nevpt2::RdmBuildOptions& opt, std::int64_t nc,
                               std::int64_t na, std::int64_t nv,
                               const nevpt2::DeviceResources& res) {
  nevpt2::check(firstExp >= kSwFdFirstExp && firstExp < kKkFdLastExp,
                "swLadders: the first step lies inside SC's ladder");
  const std::size_t nSteps = static_cast<std::size_t>(kKkFdLastExp - firstExp + 1);
  std::vector<double> step(nSteps), second(nSteps), floor(nSteps);
  std::vector<nevpt2::NevptIntegralsDevice> combos;
  for (std::size_t i = 0; i < nSteps; ++i) {
    const double h = std::ldexp(1.0, -(firstExp + static_cast<int>(i)));
    step[i] = h;
    std::vector<KkHostInts> corners;
    double e = 0.0, eMax = 0.0;
    const std::array<std::pair<double, double>, 4> at{{{h, h}, {h, -h}, {-h, h}, {-h, -h}}};
    const std::array<double, 4> sign{1.0, -1.0, -1.0, 1.0};
    for (std::size_t c = 0; c < at.size(); ++c) {
      KkHostInts ic =
          kkRotatedInts(g, fock, kkRotation(u, v, at[c].first, at[c].second, false, nc, na, nv),
                        nc, na);
      const nevpt2::NevptIntegralsDevice d = kkUploadCombination({ic}, {1.0}, res);
      const nevpt2::Tensor* h2e = nullptr;
      for (const auto& [name, t] : ic.fields)
        if (name == "h2e") h2e = &t;
      nevpt2::check(h2e != nullptr, "swLadders: the displaced set carries h2e");
      const nevpt2::RdmBuildResult b =
          nevpt2::buildRdmsDevice(opt, g.get("ci"), g.ncas, g.nelecA, g.nelecB, *h2e, res);
      double ec = 0.0;
      if (pc) {
        const nevpt2::PcEnergyResult pe = NEVPT2_TRY(
            nevpt2::pcEnergiesDevice(d, rdm.dm1, rdm.dm2, rdm.dm3, b.f3ac, b.f3ca, res));
        for (std::size_t k = 0; k < pe.classes.size(); ++k) {
          const nevpt2::PcClassResult& r = pe.classes[k];
          if (r.status != nevpt2::PcStatus::Done)
            return nevpt2::err_numerical(std::format(
                "the ladder's corner (s, t) = ({:+g}, {:+g}) has no PC energy: {} is refused ({})",
                at[c].first, at[c].second, nevpt2::CLASSES[k], r.why));
          ec += r.energy;
        }
      } else {
        ec = nevpt2::energiesDevice(d, rdm.dm1, rdm.dm2, rdm.dm3, b.f3ac, b.f3ca, res).total;
      }
      e += sign[c] * ec;
      eMax = std::max(eMax, std::fabs(ec));
      corners.push_back(std::move(ic));
    }
    second[i] = e / (4.0 * h * h);
    floor[i] = std::numeric_limits<double>::epsilon() * eMax / (h * h);
    combos.push_back(kkUploadCombination(corners, {1.0, -1.0, -1.0, 1.0}, res));
  }
  std::vector<const nevpt2::NevptIntegralsDevice*> ptr;
  for (const nevpt2::NevptIntegralsDevice& c : combos) ptr.push_back(&c);
  const std::vector<nevpt2::ScFullRotationPairing> paired =
      NEVPT2_TRY(kkPairIncrements(pc, ints, rdm, hbar, ptr, res));

  // The second differences of E and of E - L, raw; each is O(h^2) off its
  // limit (the fourfold run on the raw errors is what verifies that), and its
  // Richardson extrapolation (4 D(h/2) - D(h)) / 3 is O(h^4) off, which is
  // what the 1e-6 relative gate holds: the raw ladder of a total energy
  // bottoms out at its round-off floor, ~ eps |E| / h^2, before its O(h^2)
  // error is that small for a pair whose Eddot is small against |E|.
  std::vector<double> dE(nSteps), dS(nSteps);
  for (std::size_t i = 0; i < nSteps; ++i) {
    dE[i] = second[i];
    dS[i] = second[i] - paired[i].adjoint / (4.0 * step[i] * step[i]);
  }
  struct Gate {
    const char* label;
    const std::vector<double>* d;
    double target;
  };
  std::vector<Gate> gates;
  if (want) gates.push_back(Gate{"E against Eddot_kk", &dE, *want});
  gates.push_back(Gate{"E - L against the sandwich", &dS, wantSandwich});
  bool ok = true;
  for (const Gate& gt : gates) {
    std::vector<double> err(nSteps);
    for (std::size_t i = 0; i < nSteps; ++i) err[i] = std::fabs((*gt.d)[i] - gt.target);
    std::printf("  %s (%.14e):\n", gt.label, gt.target);
    // The ratio test's errors: a rung within kSwFloorMargin of its floor
    // reads as 0, which longestFourfoldRun never counts (no ratio across it).
    std::vector<double> above(nSteps);
    for (std::size_t i = 0; i < nSteps; ++i)
      above[i] = err[i] > kSwFloorMargin * floor[i] ? err[i] : 0.0;
    std::printf("  %5s %12s %22s %10s %6s %22s %10s %10s\n", "k", "h = 2^-k",
                "second diff / 4h^2", "|err|", "ratio", "Richardson (4D(h/2)-D(h))/3", "rel err",
                "floor");
    double bestRel = std::numeric_limits<double>::infinity();
    std::size_t bestAt = 0;
    for (std::size_t i = 0; i < nSteps; ++i) {
      const int k = firstExp + static_cast<int>(i);
      const std::string ratio =
          i == 0 ? std::string("-") : std::format("{:.2f}", err[i] > 0.0 ? err[i - 1] / err[i] : 0.0);
      if (i + 1 < nSteps) {
        const double r = (4.0 * (*gt.d)[i + 1] - (*gt.d)[i]) / 3.0;
        const double rel = std::fabs(r - gt.target) / std::max(1e-30, scale);
        if (rel < bestRel) {
          bestRel = rel;
          bestAt = i;
        }
        std::printf("  %5d %12.3e %22.14e %10.3e %6s %22.14e %10.3e %10.3e%s\n", k, step[i],
                    (*gt.d)[i], err[i], ratio.c_str(), r, rel, floor[i],
                    above[i] > 0.0 ? "" : " (at the floor)");
      } else {
        std::printf("  %5d %12.3e %22.14e %10.3e %6s %22s %10s %10.3e%s\n", k, step[i],
                    (*gt.d)[i], err[i], ratio.c_str(), "", "", floor[i],
                    above[i] > 0.0 ? "" : " (at the floor)");
      }
    }
    const nevpt2::FourfoldRun run =
        nevpt2::longestFourfoldRun(above, nevpt2::kFourfoldLow, kSwFourfoldHigh);
    const bool good = bestRel < kKkFdTol && run.length >= kSwFdRun;
    ok = ok && good;
    std::printf("  extrapolated: best relative error %.3e (against %.3e) from h = %.3e; raw, above "
                "%.0fx the floor: %d halvings by >= %.1f ending at h = %.3e -- %s\n",
                bestRel, scale, step[bestAt], kSwFloorMargin, run.length, nevpt2::kFourfoldLow,
                step[run.end], good ? "ok" : "NOT CONVERGED");
  }
  return ok;
}

// One pair's sandwich, as a table row.
void swPrintPair(const std::string& name, const nevpt2::ScOrbitalSandwichResult& r) {
  const nevpt2::ScPairTerms& t = r.total;
  std::printf("    %-30s %22.14e %22.14e %9.2e %12.4e %12.4e %12.4e %12.4e %12.4e %9.2e\n",
              name.c_str(), r.forward, r.reverse, std::fabs(r.forward - r.reverse), t.k3, t.k4,
              t.k5uv, t.k5vu, t.k6, std::fabs(t.k6 - r.reverseK6));
}

nevpt2::Result<bool> checkScHessianKk(const nevpt2::GoldenFile& g, const nevpt2::GradGoldFile& gg,
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
  std::printf("\n--sc-hessian-kk: the orbital Hessian block Eddot_kk(u, v) = <Ehat', I''> + "
              "I'[u]^T Ehat'' I'[v] (docs/hessian-theory.md, Sec. 5.1)%s\n",
              fitted ? " -- density-fitted" : "");
  std::printf("  the sandwich = K3 (the finish's curvature) + K4 (both external slots) + K5 (an "
              "external slot against the Dyall block),\n  the digests' share merged per tuple; "
              "both modes\n");
  bool ok = true;

  const nevpt2::DeviceTensor hbar = nevpt2::scDigestAdjointDevice(
      opt, ci, g.nelecA, g.nelecB, ints, dm1, dm2, built.dm3, built.f3ac, built.f3ca, res);
  const nevpt2::ScOrbitalSandwichSetup sw{ints, gen, rdm, ci, g.nelecA, g.nelecB, opt};
  const nevpt2::ScOrbitalCurvatureSetup cs{ints, gen, rdm, &hbar, &ci, g.nelecA, g.nelecB, opt};

  // The directions, and their first-order pairings <k, Y> in one walk.
  std::vector<SwDir> dirs{
      {"active", embedBlock(nevpt2::antisymmetricKappa(na, kSwSeedBase + 1), nc, nmo)},
      {"active'", embedBlock(nevpt2::antisymmetricKappa(na, kSwSeedBase + 2), nc, nmo)},
      {"full", nevpt2::antisymmetricKappa(nmo, kSwSeedBase + 3)},
      {"full'", nevpt2::antisymmetricKappa(nmo, kSwSeedBase + 4)},
  };
  const std::size_t iAct = 0, iAct2 = 1, iFull = 2, iFull2 = 3;
  std::optional<std::size_t> iCore, iCore2, iVirt, iVirt2;
  if (nc >= 2) {
    iCore = dirs.size();
    dirs.push_back({"core", embedBlock(nevpt2::antisymmetricKappa(nc, kSwSeedBase + 5), 0, nmo)});
    iCore2 = dirs.size();
    dirs.push_back({"core'", embedBlock(nevpt2::antisymmetricKappa(nc, kSwSeedBase + 6), 0, nmo)});
  }
  if (nv >= 2) {
    iVirt = dirs.size();
    dirs.push_back(
        {"virtual", embedBlock(nevpt2::antisymmetricKappa(nv, kSwSeedBase + 7), nc + na, nmo)});
    iVirt2 = dirs.size();
    dirs.push_back(
        {"virtual'", embedBlock(nevpt2::antisymmetricKappa(nv, kSwSeedBase + 8), nc + na, nmo)});
  }
  // The proportional pair's second member (#190): 2 virtual + core. On every
  // empty tuple of an H class its first-order perturber is twice virtual's
  // -- a core rotation populates none of them (Sec. 5.6's census: core x core
  // populates Sijrs alone) -- so its tables are proportional to virtual's and
  // rho_t is round-off, while it is not a multiple of virtual.
  std::optional<std::size_t> iProp;
  if (iCore && iVirt) {
    nevpt2::Tensor k = dirs[*iCore].k;
    for (std::int64_t i = 0; i < k.size(); ++i) k.flatRef(i) += 2.0 * dirs[*iVirt].k.flat(i);
    iProp = dirs.size();
    dirs.push_back({"2 virtual + core", std::move(k)});
  }
  {
    std::vector<nevpt2::ScRotationRequest> req;
    for (const SwDir& d : dirs) req.push_back(nevpt2::ScRotationRequest{d.k, false});
    const nevpt2::ScFullRotationResult first = NEVPT2_TRY(nevpt2::scFullRotationDevice(
        ints, gen, dm1, dm2, built.dm3, built.f3ac, built.f3ca, &hbar, req, res));
    for (std::size_t k = 0; k < dirs.size(); ++k) dirs[k].y = first.dirs[k].adjoint;
  }

  std::vector<SwPair> pairs{
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
  if (iProp)
    pairs.push_back({"virtual x (2 virtual + core)", *iVirt, *iProp, true, false, true, true});

  // The sandwich along every pair, one call each, both modes, the digests'
  // reference where both directions move h2e. A pair along which the energy
  // has no Hessian (Sec. 5.6, #190) is refused as a Numerical Error; every
  // other error is the caller's and propagates.
  const std::vector<SwPair> candidates = pairs;
  pairs.clear();
  std::vector<nevpt2::ScOrbitalSandwichResult> sand;
  std::vector<std::pair<std::string, std::string>> refusedPairs;
  const auto t0 = std::chrono::steady_clock::now();
  for (const SwPair& p : candidates) {
    const bool digests = !p.intraU && !p.intraV && p.name != "full x full";
    nevpt2::Result<std::vector<nevpt2::ScOrbitalSandwichResult>> r =
        nevpt2::scOrbitalSandwichDevice(
            sw, {nevpt2::ScOrbitalSandwichRequest{dirs[p.u].k, dirs[p.v].k, true, digests}}, res);
    if (!r) {
      if (r.error().kind != nevpt2::ErrorKind::Numerical) return std::unexpected(r.error());
      refusedPairs.emplace_back(p.name, r.error().message);
      continue;
    }
    pairs.push_back(p);
    sand.push_back(r->front());
  }
  const double sandSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  // --- 0. the guarded tuples ------------------------------------------------------
  {
    std::printf("\n  [0] the guarded tuples (docs/hessian-theory.md, Sec. 5.6): K6, and the "
                "pairs with no Hessian\n");
    // CLASSES order: Sijrs is the one class with H = 0, exempt from rho.
    std::size_t iSijrs = 0;
    for (std::size_t c = 0; c < 8; ++c)
      if (nevpt2::CLASSES[c] == "Sijrs") iSijrs = c;
    double worstRho = 0.0, popNoise = 0.0;
    double popMin = std::numeric_limits<double>::infinity();
    bool computedRight = true;
    std::optional<double> propK6H, coreK6;
    for (std::size_t k = 0; k < pairs.size(); ++k) {
      const nevpt2::ScOrbitalSandwichResult& r = sand[k];
      std::string row;
      double k6H = 0.0;
      for (std::size_t c = 0; c < 8; ++c) {
        const nevpt2::ScGuardedCensus& g = r.guarded[c];
        worstRho = std::max(worstRho, g.rhoMax);
        // The floor decides only for the H classes (Sijrs is exempt, and its
        // sign-indefinite N^{ww} can sit anywhere near zero).
        if (c != iSijrs) {
          popNoise = std::max(popNoise, g.unpopulatedMax);
          popMin = std::min(popMin, g.populatedMin);
        }
        computedRight = computedRight && !g.refused();
        if (c != iSijrs) k6H += r.classes[c].k6;
        if (g.populated == 0 && r.classes[c].k6 == 0.0) continue;
        row += std::format("\n      {:<6} {:5d} of {:5d} empty tuples populated by both "
                           "(u {:5d}, v {:5d}), max rho_t {:.2e}, K6 {:+.10e}",
                           nevpt2::CLASSES[c], g.populated, g.empty, g.populatedU, g.populatedV,
                           g.rhoMax, r.classes[c].k6);
      }
      if (pairs[k].name.starts_with("virtual x (2 virtual")) propK6H = k6H;
      if (pairs[k].name == "core x core") coreK6 = r.classes[iSijrs].k6;
      std::printf("    %-30s computed, K6 %+.10e (the H classes' %+.10e)%s\n",
                  pairs[k].name.c_str(), r.total.k6, k6H,
                  row.empty() ? " -- no guarded tuple populated" : row.c_str());
    }
    std::vector<std::string> refusedNames;
    for (const auto& [name, why] : refusedPairs) {
      std::printf("    %-30s refused, Numerical: %s\n", name.c_str(), why.c_str());
      refusedNames.push_back(name);
    }
    // On N2 the two pairs with no Hessian are exactly the ones whose two
    // directions both carry a random virtual rotation (Sec. 5.6's census);
    // every other pair is computed, the proportional one included, and its
    // H-class K6 is far from zero. Sijrs closes core x core.
    std::vector<std::string> wantRefused;
    if (iVirt) wantRefused = {"full x full", "virtual x virtual"};
    else wantRefused = {"full x full"};
    const bool refusedRight = refusedNames == wantRefused;
    const bool propRight = !iProp || (propK6H && std::fabs(*propK6H) > 1e3 * kSwZeroTol);
    const bool coreRight = !iCore || (coreK6 && std::fabs(*coreK6) > 1e3 * kSwZeroTol);
    const bool good = computedRight && refusedRight && propRight && coreRight &&
                      worstRho <= nevpt2::kRhoTol && pairs.size() >= 2 &&
                      pairs[0].name == "active x full" && pairs[1].name.starts_with("full x active");
    ok = ok && good;
    std::printf("  over the computed pairs: worst rho_t %.2e (<= %.0e); the H classes' population "
                "N^ww/|w|^2 on an empty tuple <= %.2e unpopulated, >= %.2e populated (floor "
                "%.0e); the "
                "proportional pair's H-class K6 %s; Sijrs's K6 on core x core %s; %zu pairs "
                "refused (%s): %s\n",
                worstRho, nevpt2::kRhoTol, popNoise, popMin, nevpt2::kPopulatedTol,
                propK6H ? std::format("{:+.6e}", *propK6H).c_str() : "-",
                coreK6 ? std::format("{:+.6e}", *coreK6).c_str() : "-", refusedPairs.size(),
                refusedRight ? "the expected ones" : "NOT the expected ones",
                good ? "ok" : "FAILED");
  }

  // --- 1. the merge ---------------------------------------------------------------
  {
    double worst = 0.0;
    std::printf("\n  [1] the merge: the walk's own Edot against the first-order <k, Y> with the "
                "digests' term\n");
    for (std::size_t k = 0; k < pairs.size(); ++k) {
      const double du = std::fabs(sand[k].edotUTotal - dirs[pairs[k].u].y);
      const double dv = std::fabs(sand[k].edotVTotal - dirs[pairs[k].v].y);
      worst = std::max({worst, du, dv});
      std::printf("    %-30s Edot(u) %22.14e |delta| %.2e   Edot(v) %22.14e |delta| %.2e\n",
                  pairs[k].name.c_str(), sand[k].edotUTotal, du, sand[k].edotVTotal, dv);
    }
    ok = ok && worst < kSwTol;
    std::printf("  worst |delta| = %.2e (< %.0e: %s)\n", worst, kSwTol,
                worst < kSwTol ? "ok" : "FAILED");
  }

  // --- 2. the duality -------------------------------------------------------------
  {
    std::printf("\n  [2] the duality (%zu pairs in %.2fs): forward K3 + K4 + K5 + K6 against "
                "the reverse pairing, and K6 against its own reverse half\n",
                pairs.size(), sandSeconds);
    std::printf("    %-30s %22s %22s %9s %12s %12s %12s %12s %12s %9s\n", "pair", "forward",
                "reverse", "|delta|", "K3", "K4", "K5 (u, K'v)", "K5 (v, K'u)", "K6",
                "K6 |delta|");
    double worst = 0.0, worstSelf = 0.0, worstK6 = 0.0;
    for (std::size_t k = 0; k < pairs.size(); ++k) {
      swPrintPair(pairs[k].name, sand[k]);
      worst = std::max(worst, std::fabs(sand[k].forward - sand[k].reverse));
      worstK6 = std::max(worstK6, std::fabs(sand[k].total.k6 - sand[k].reverseK6));
      worstSelf = std::max({worstSelf, std::fabs(sand[k].total.k3 - sand[k].total.k3Swap),
                            std::fabs(sand[k].total.k4 - sand[k].total.k4Swap)});
    }
    const double swap = std::fabs(sand[0].forward - sand[1].forward);
    const bool good = worst < kSwTol && swap < kSwTol && worstSelf < kSwTol && worstK6 < kSwTol;
    ok = ok && good;
    std::printf("  duality, worst |delta| = %.2e; K6 against its reverse half, worst |delta| = "
                "%.2e; the swap against its original |delta| = %.2e; K3 and K4 from the "
                "exchanged finish |delta| = %.2e (all < %.0e: %s)\n",
                worst, worstK6, swap, worstSelf, kSwTol, good ? "ok" : "FAILED");
  }

  // --- 3. the intra-space zeros ---------------------------------------------------
  {
    std::printf("\n  [3] the intra-space directions: h2e' and f3' exactly zero, so the digests' "
                "shares vanish identically\n");
    bool good = true;
    std::size_t seen = 0;
    for (std::size_t k = 0; k < pairs.size(); ++k) {
      const SwPair& p = pairs[k];
      if (!p.intraU && !p.intraV) continue;
      ++seen;
      const nevpt2::ScOrbitalSandwichResult& r = sand[k];
      const nevpt2::ScPairTerms& t = r.total;
      bool zeros = true;
      // Along an intra-space v: K'[v]'s digests' share, and K5's (x'[u], K'[v]) half.
      if (p.intraV)
        zeros = zeros && r.h2eDotMaxV == 0.0 && r.f3DotMaxV == 0.0 && t.k3DigestV == 0.0 &&
                t.k5uvDigest == 0.0 && std::fabs(t.k5uv) < kSwZeroTol;
      if (p.intraU)
        zeros = zeros && r.h2eDotMaxU == 0.0 && r.f3DotMaxU == 0.0 && t.k3DigestU == 0.0 &&
                t.k5vuDigest == 0.0 && std::fabs(t.k5vu) < kSwZeroTol;
      zeros = zeros && t.k3DigestDigest == 0.0;
      // ... while K4 does not, and neither does K5's other half where the
      // other direction leaves the inactive spaces. (K3 is not held nonzero
      // for an active member: along an active direction that mixes irreps
      // every tuple's triple vanishes by symmetry -- what makes <u, Y_aa>
      // vanish at N2 CAS(4,4) -- and K3 with it.)
      const double otherHalf = p.intraU && p.intraV ? 1.0 : (p.intraV ? t.k5vu : t.k5uv);
      const bool alive =
          std::fabs(t.k4) > 1e3 * kSwZeroTol && std::fabs(otherHalf) > 1e3 * kSwZeroTol;
      good = good && zeros && alive;
      std::printf("    %-30s max|h2e'| %.1e / %.1e, max|f3'| %.1e / %.1e; K3's digests' shares "
                  "%.1e / %.1e,\n      K5's %.1e / %.1e; |K5 (u, K'v)| %.1e, |K5 (v, K'u)| %.1e "
                  "(the intra-space halves < %.0e); K3 %.6e, K4 %.6e: %s\n",
                  p.name.c_str(), r.h2eDotMaxU, r.h2eDotMaxV, r.f3DotMaxU, r.f3DotMaxV,
                  t.k3DigestU, t.k3DigestV, t.k5vuDigest, t.k5uvDigest, std::fabs(t.k5uv),
                  std::fabs(t.k5vu), kSwZeroTol, t.k3, t.k4, zeros && alive ? "ok" : "FAILED");
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
      creq.push_back(nevpt2::ScOrbitalCurvatureRequest{dirs[pairs[k].u].k, dirs[pairs[k].v].k, true});
      cwhich.push_back(k);
    }
  const std::vector<nevpt2::ScOrbitalCurvatureResult> curv =
      NEVPT2_TRY(nevpt2::scOrbitalCurvatureDevice(cs, creq, res));

  // --- 4. the digests against the plain digest path --------------------------------
  //
  // The functional with ONLY the digests moving, f3(h2e(U)) and everything
  // else the state's, has Eddot = E_f3 . f3(h2e''[u, v]) + E_f3f3[f3'[u],
  // f3'[v]] (f3 is linear in h2e: Sec. 5.2). The energy stage computes both
  // off the plain digest path -- the RDM build at h2e'' (the curvature's
  // forward digests' term) and at h2e'[u], h2e'[v] (energiesHessianDevice at
  // those tangents) -- and the operator carries them as the curvature's
  // reverse digests' term (#89's transpose) and K3's digest x digest share
  // (the walk's per-tuple merge).
  {
    std::printf("\n  [4] the digests alone: the operator's shares against the energy stage at the "
                "plain digest path's builds\n");
    std::printf("    %-30s %22s %22s %9s %22s %22s %9s\n", "pair", "K3 digest x digest",
                "energy stage", "|delta|", "digests alone, operator", "energy stage", "|delta|");
    double worst = 0.0;
    std::size_t seen = 0;
    for (std::size_t c = 0; c < cwhich.size(); ++c) {
      const nevpt2::ScOrbitalSandwichResult& r = sand[cwhich[c]];
      if (!r.hasDigestReference) continue;
      ++seen;
      const nevpt2::ScOrbitalCurvatureResult& k = curv[c];
      const double d1 = std::fabs(r.total.k3DigestDigest - r.digestReference);
      const double walk = r.total.k3DigestDigest + k.k1.digest + k.k2.digest;
      const double plain = r.digestReference + k.k1.digestForward + k.k2.digestForward;
      const double d2 = std::fabs(walk - plain);
      worst = std::max({worst, d1, d2});
      std::printf("    %-30s %22.14e %22.14e %9.2e %22.14e %22.14e %9.2e\n",
                  pairs[cwhich[c]].name.c_str(), r.total.k3DigestDigest, r.digestReference, d1,
                  walk, plain, d2);
    }
    const bool good = seen > 0 && worst < kSwTol;
    ok = ok && good;
    std::printf("  worst |delta| = %.2e (< %.0e: %s)\n", worst, kSwTol, good ? "ok" : "FAILED");
  }

  // --- 5. the ladders -------------------------------------------------------------
  const KkFockParts fock = kkFockParts(gen, g.get("dm1"), res);
  std::printf("\n  Eddot_kk(u, v) = <Ehat', I''> + the sandwich, where the curvature is formed:\n");
  for (std::size_t c = 0; c < cwhich.size(); ++c)
    std::printf("    %-30s %22.14e = %22.14e + %22.14e\n", pairs[cwhich[c]].name.c_str(),
                curv[c].value() + sand[cwhich[c]].forward, curv[c].value(),
                sand[cwhich[c]].forward);
  {
    const double swap = std::fabs(curv[0].value() + sand[0].forward -
                                  (curv[1].value() + sand[1].forward));
    ok = ok && swap < kSwTol;
    std::printf("  Eddot_kk's swap, (full, active) against (active, full): |delta| %.2e (< %.0e: "
                "%s)\n",
                swap, kSwTol, swap < kSwTol ? "ok" : "FAILED");
  }
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    const SwPair& p = pairs[k];
    if (!p.blockDiagonal) continue;
    std::optional<double> want;
    for (std::size_t c = 0; c < cwhich.size(); ++c)
      if (cwhich[c] == k) want = curv[c].value() + sand[k].forward;
    const double scale = std::max({want ? std::fabs(*want) : std::fabs(sand[k].forward),
                                   std::fabs(dirs[p.u].y), std::fabs(dirs[p.v].y)});
    std::printf("\n  [5] the ladders along exp(s u + t v), %s (<u, Y> = %.6e, <v, Y> = %.6e)%s:\n",
                p.name.c_str(), dirs[p.u].y, dirs[p.v].y,
                want ? "" : " -- the sandwich's alone: K2, so Eddot_kk, is refused here");
    const bool fd = NEVPT2_TRY(swLadders(false, kSwFdFirstExp, g, fock, dirs[p.u].k, dirs[p.v].k, want,
                                         sand[k].forward, scale, ints, rdm, hbar, opt, nc, na, nv,
                                         res));
    ok = ok && fd;
  }

  // --- 6. the assembled block's refusals ------------------------------------------
  {
    // Conventional: #154's K2 refusal comes first, before any device work.
    // Fitted: K2 is formed, and the sandwich's guarded-tuple refusal (or, if
    // the pair populates none, the sandwich itself) decides.
    const nevpt2::Result<std::vector<nevpt2::ScOrbitalHessianResult>> r =
        nevpt2::scOrbitalHessianDevice(
            cs, {nevpt2::ScOrbitalCurvatureRequest{dirs[iFull].k, dirs[iFull2].k}}, res);
    const nevpt2::ErrorKind want =
        fitted ? nevpt2::ErrorKind::Numerical : nevpt2::ErrorKind::Unsupported;
    bool good = false;
    std::string what;
    if (!r) {
      good = r.error().kind == want;
      what = std::format("refused, {}", good ? (fitted ? "Numerical" : "Unsupported") : "WRONG KIND");
    } else {
      // Not refused: only right where the sandwich along full x full was computed.
      for (std::size_t k = 0; k < pairs.size(); ++k)
        if (pairs[k].name == "full x full") {
          const double gap = std::fabs(r->front().sandwich.forward - sand[k].forward);
          good = gap < kSwTol;
          what = std::format("{:.14e}, its sandwich against check 2's |delta| {:.2e}",
                             r->front().value(), gap);
        }
      if (what.empty()) what = "NOT REFUSED";
    }
    ok = ok && good;
    std::printf("\n  [6] Eddot_kk along full x full on a %s case: %s: %s\n",
                fitted ? "density-fitted" : "conventional", what.c_str(), good ? "ok" : "FAILED");
  }

  std::printf("\n%s: the SC orbital Hessian's sandwich I'^T Ehat'' I' = K3 + K4 + K5 + K6 agrees "
              "with itself by duality to %.0e, its digests' share is the first-order term's and "
              "its curvature the energy stage's, the intra-space zeros hold, and the ladders of "
              "the SC total along exp(s u + t v) converge to Eddot_kk = <Ehat', I''> + the "
              "sandwich; %zu pairs refused, with no Hessian along them (rho_t, Sec. 5.6)%s\n",
              ok ? "PASS" : "FAIL", kSwTol, refusedPairs.size(),
              fitted ? " (density-fitted)" : "");
  return ok;
}
