// nevpt2.app.sc_pseudodensity:hessian_kc_pc_check -- --pc --sc-hessian-kc:
// the PC-NEVPT2 energy's MIXED orbital-CI second derivative, the block H_kc
// of its parameter Hessian (docs/hessian-theory.md, Sec. 6.4-6.8; #201), in
// the two orientations #184 built for SC -- the scalar Eddot_kc(w, u)
// (pcMixedHessianDevice) and the determinant vector H_kc^T w, the derivative
// along w of the THREE-TERM y (pcMixedHessianVectorDevice) -- checked against
// each other and against what neither of them shares.
// An interface partition that exports nothing, as :hessian_kc_check is.
//
// WHAT IS DIFFERENT FROM SC's --sc-hessian-kc. PC's metric S is RDM-only, so
// it does not move along kappa; it moves along u, and its cut's dropped space
// turns with it. That leaves ONE rotation term in H_kc, -2 rho^T S_u P A_w x
// (Sec. 6.5), present along every w that moves the Dyall block K -- every
// block except core-core and virtual-virtual. The duality and Euler cannot
// see it (both orientations carry it; u = c has P S_c = 0); the step-free
// active-rotation gauge and the finite difference of the three-term y can,
// and both run. Every table prints the term per class.
//
// THE CI DIRECTIONS. A random amplitude vector lifts S's dropped modes and
// the class is REFUSED (PC_LIFT_TOL; check 6 is that refusal). The u here
// are of two kinds that do not lift: `Epq`, sum_pq f_pq E_pq |c> over the
// state's symmetry mask (pcTangentDirection's construction -- on N2's
// CAS(4,4), whose four active orbitals are four irreps, these are number
// operators, and every active-active rotation's mixed element along them is
// a symmetry zero), and `rot`, sum_pq f_pq E_pq |c> for a random
// ANTISYMMETRIC f over every active pair: an orbital rotation of the state,
// any irrep, a singlet, along which the dropped modes turn (so the rotation
// term is there) but do not lift (Sec. 6.8: an orthogonal similarity). Each
// is projected against c and normalized; the state itself is the fourth.
//
// THE CHECKS, in the order they print. Each names its orbital-energy mode:
// EULER (they respond to kappa, held fixed in c) or FROZEN (held fixed in
// both). FULL -- responding to c too -- is refused for PC (check 6): PC's
// mixed block holds them fixed in c (only PC's CI block H_cc has a FULL mode,
// #214).
//
//   With the gradient sidecar (conventional or density-fitted), EULER mode,
//   along a seeded fully mixed w:
//   1  PC's FIRST-ORDER SOURCE TERM along w, which the next checks rest on:
//      the full duality, <w, Y_pc> (pcFullRotationDevice with hbar_pc)
//      against Edot (its tangent + pcDigestRotationTangentDevice), and both
//      walks' own Edot(w) -- the digests' tangent merged in K'_w -- against
//      that first-order term, class by class. On a fitted case #200's
//      --pc --sc-hessian-kk-int is this term's gate in tree; this re-holds it
//      along this run's own w.
//   2  THE DUALITY <H_kc^T w, u> == Eddot_kc(w, u), the three CI directions
//      and the state, with the five-way per-class table along the first
//      `rot` direction (the rotation term apart). The tree's 1e-7.
//   3  EULER IN c: Eddot_kc(w, c) == 2 <w, Y_pc> per class on PC's five
//      homogeneous classes (Sr, Si, Srsi, Srs, Sir), against the already-gated
//      first-order term of check 1. Sijrs is exactly zero (no RDM, orbital
//      energies fixed in c); Sijr and Sij are affine and printed.
//   4  THE INTRA-SPACE ZEROS: along a core-core and a virtual-virtual w,
//      h2e'[w] is an exactly-zero tensor and K'[w] vanishes (up to h1e'[w]'s
//      core-dressing round-off along core-core), so the rotation term does;
//      the slabs still move. The duality along both.
//   5  ONE FINITE DIFFERENCE of the GPU's own three-term y
//      (pcCiSourceThreeTermDevice), PER CLASS, along exp(s w) for the
//      core-core and the virtual-virtual w (--kappa-full's displacement: the
//      blocks' inactive axes and that space's orbital energies rotated on
//      the host), dotted with a `rot` u, against <H_kc^T w, u>'s class share
//      -- for every class the space indexes; a class it does not index sees
//      no input move. Richardson-verified; a plateau is a failure.
//   6  THE REFUSALS: a random amplitude u, along which S's dropped modes
//      lift, is a Numerical Error naming PC_LIFT_TOL -- never an element;
//      the orbital energies responding to c is Unsupported.
//
//   Always (FROZEN mode, no sidecar handed in), along a seeded active-active w:
//   7  the duality (three u and the state); THE ACTIVE-ROTATION GAUGE,
//      Eddot_kc(w, u) == Eddot_cc(what c, u) + <y, what u> per class, with
//      what = sum_pq w_pq E_pq, Eddot_cc pcHessianDevice's and y the
//      three-term one -- step-free, and the only step-free check that sees
//      the rotation term (on the right it is Eddot_cc's own rotation terms
//      along what c) -- and its first-order twin Edot(w) == <y, what c>;
//      the finite difference of the three-term y PER CLASS along exp(s w)
//      (the active axes rotated, the digests rebuilt, --kappa-act's
//      displacement) against <H_kc^T w, u>'s class share, for every class --
//      the difference that sees the rotation term; and the refusals of a w
//      leaving the active space and of a responding mode without a sidecar.
//
// With no sidecar beside the golden only check 7 runs, and the PASS line
// says so. CAS(8,8) is a diagnostic, not a gate, for #152's reason (#194).
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries. A malformed direction or a
// sidecar that does not fit the golden is the caller's input and propagates
// unchanged to main, which reports it once.
#include "error_handling/error_macros.h"

export module nevpt2.app.sc_pseudodensity:hessian_kc_pc_check;

import std;
import nevpt2.gradient;
import nevpt2.kappa_rotation;
import nevpt2.golden;
import nevpt2.rdm_host_dm;
// The displacements of the finite differences are --kappa-full's and
// --kappa-act's, and the seeded-direction helpers SC's mixed check's.
import :kappa_full_check;
import :kappa_check;
import :hessian_kc_check;

// Every exact identity between two computations, SC's mixed check's gate:
// the tree's 1e-7 (docs/testing.md, "PC's mixed Hessian block's checks").
constexpr double kPkcTol = 1e-7;
// The rotation term along a core-core w, where K'[w] is h1e'[w]'s core
// dressing round-off alone (exactly 0 along virtual-virtual): absolute,
// decades above what is measured and decades below any real term
// (docs/testing.md, "PC's mixed Hessian block's checks").
constexpr double kPkcRotZeroTol = 1e-13;
// The finite difference of y: --kappa-full's ladder, run rule and gate, as
// SC's mixed check's (h = 2^-k, k in [6, 18]; the best error relative to
// max(|<H_kc^T w, u>_c|, |<u, y_c>|) under 1e-6; a run of three fourfold
// halvings).
constexpr int kPkcFdFirstExp = 6;
constexpr int kPkcFdLastExp = 18;
constexpr std::size_t kPkcFdSteps = kPkcFdLastExp - kPkcFdFirstExp + 1;
constexpr double kPkcFdTol = 1e-6;
constexpr int kPkcFdRun = 3;
// Seeds, distinct from every other check's.
constexpr std::uint64_t kPkcSeedBase = 0x5eed'0201u;
constexpr std::uint64_t kPkcEpqSeed = 0x5eed'1201u;
constexpr std::array<std::uint64_t, 2> kPkcRotSeeds = {0x5eed'2201u, 0x5eed'3201u};
constexpr std::uint64_t kPkcControlSeed = 0x5eed'4201u;

// PC's five classes homogeneous of degree 2 in c (Sec. 6.8): Sr, Si, Srsi,
// Srs, Sir.
constexpr bool pkcHomogeneous(std::size_t c) {
  return c == 0 || c == 1 || c == 4 || c == 5 || c == 7;
}

// sum_pq f_pq E_pq x, the one-body operator `f` (norb, norb) on a CI vector,
// from the builder's E_pq |x> table.
nevpt2::Tensor pkcOneBody(const nevpt2::HostDm12Builder& b, const nevpt2::Tensor& f,
                          const nevpt2::Tensor& x) {
  const std::int64_t norb = b.norb(), ndet = b.ndet();
  const std::vector<double> v = b.applyAllE(x);
  nevpt2::Tensor out(x.dims());
  for (std::int64_t p = 0; p < norb; ++p)
    for (std::int64_t q = 0; q < norb; ++q) {
      const double fpq = f(p, q);
      if (fpq == 0.0) continue;
      const std::int64_t off = (p * norb + q) * ndet;
      for (std::int64_t k = 0; k < ndet; ++k)
        out.flatRef(k) += fpq * v[static_cast<std::size_t>(off + k)];
    }
  return out;
}

// `x` projected against `ci` and normalized.
nevpt2::Tensor pkcProjected(const nevpt2::Tensor& ci, nevpt2::Tensor x) {
  const double overlap = kcDot(ci, x), ciNorm2 = kcDot(ci, ci);
  for (std::int64_t i = 0; i < x.size(); ++i) x.flatRef(i) -= (overlap / ciNorm2) * ci.flat(i);
  const double norm = std::sqrt(kcDot(x, x));
  for (std::int64_t i = 0; i < x.size(); ++i) x.flatRef(i) /= norm;
  return x;
}

// The `Epq` direction: f random on the pairs the state's dm1 allows.
nevpt2::Tensor pkcEpqDirection(const nevpt2::HostDm12Builder& b, const nevpt2::Tensor& ci,
                               const nevpt2::Tensor& dm1, std::uint64_t seed) {
  const std::int64_t n = b.norb();
  double dm1Max = 0.0;
  for (std::int64_t i = 0; i < dm1.size(); ++i) dm1Max = std::max(dm1Max, std::fabs(dm1.flat(i)));
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  nevpt2::Tensor f({n, n});
  for (std::int64_t p = 0; p < n; ++p)
    for (std::int64_t q = 0; q < n; ++q) {
      const double r = dist(gen);
      if (std::fabs(dm1(p, q)) > 1e-12 * dm1Max) f(p, q) = r;
    }
  return pkcProjected(ci, pkcOneBody(b, f, ci));
}

// The `rot` direction: f a random antisymmetric matrix over every active pair.
nevpt2::Tensor pkcRotDirection(const nevpt2::HostDm12Builder& b, const nevpt2::Tensor& ci,
                               std::uint64_t seed) {
  return pkcProjected(ci, pkcOneBody(b, nevpt2::antisymmetricKappa(b.norb(), seed), ci));
}

// One RDM set, owning: dm1/dm2 uploaded from the host's dm12Dot and the rest
// the device tangent build at the vector pair (a, b).
struct PkcSet {
  nevpt2::DeviceTensor dm1, dm2;
  nevpt2::RdmBuildResult rdm;
  nevpt2::PcRdms view() const { return {&dm1, &dm2, &rdm.dm3, &rdm.f3ac, &rdm.f3ca}; }
};

PkcSet pkcTangentSet(const nevpt2::RdmBuildOptions& opt, const nevpt2::HostDm12Builder& b,
                     const nevpt2::Tensor& x, const nevpt2::Tensor& y, const nevpt2::Tensor& h2e,
                     const nevpt2::DeviceResources& res) {
  const nevpt2::HostDm12 host = b.dm12Dot(x, y);
  return PkcSet{nevpt2::uploadTensor(host.dm1, res), nevpt2::uploadTensor(host.dm2, res),
                nevpt2::buildRdmTangentsDevice(opt, x, y, b.norb(), b.nelecA(), b.nelecB(), h2e,
                                               res)};
}

// The duality along w for each of `us`, printed; the worst |delta|.
nevpt2::Result<double> pkcDuality(const nevpt2::ScMixedHessianSetup& s, const nevpt2::Tensor& w,
                                  const nevpt2::Tensor& hkcTw,
                                  const std::vector<std::pair<std::string, nevpt2::Tensor>>& us,
                                  const nevpt2::DeviceResources& res) {
  std::vector<nevpt2::Tensor> dirs;
  for (const auto& [name, u] : us) dirs.push_back(u);
  const std::vector<nevpt2::PcMixedHessianResult> f =
      NEVPT2_TRY(nevpt2::pcMixedHessianDevice(s, w, dirs, res));
  std::printf("  %-18s %22s %22s %11s %13s\n", "u", "<H_kc^T w, u>", "Eddot_kc(w, u)", "|delta|",
              "rotation");
  double worst = 0.0;
  for (std::size_t k = 0; k < us.size(); ++k) {
    const double lhs = kcDot(us[k].second, hkcTw);
    const double d = std::fabs(lhs - f[k].eddot);
    worst = std::max(worst, d);
    std::printf("  %-18s %22.14e %22.14e %11.2e %13.5e\n", us[k].first.c_str(), lhs, f[k].eddot,
                d, f[k].rotation);
  }
  return worst;
}

// One forward result per class: Eddot_kc, the rotation term, the walk's own
// Edot(w) and three-term Edot(u), and the CI leg's lift relative to s_max.
void pkcPrintClasses(const nevpt2::PcMixedHessianResult& r) {
  std::printf("  %-7s %22s %14s %16s %16s %10s\n", "class", "Eddot_kc(w, u)", "rotation term",
              "Edot(w)", "Edot(u)", "lift");
  for (std::size_t c = 0; c < 8; ++c) {
    const nevpt2::PcClassGradient& k = r.walk.classes[c];
    std::printf("  %-7s %22.14e %14.5e %16.8e %16.8e %10.1e\n", nevpt2::CLASSES[c].c_str(),
                k.mixedEddot, k.mixedRotation, k.mixedEdotW, k.mixedEdotU, k.lift);
  }
  std::printf("  %-7s %22.14e %14.5e %16.8e %16.8e\n", "total", r.eddot, r.rotation, r.edotW,
              r.edotU);
}

// One class's central-difference ladder, from values precomputed at +-h and
// at 0: `want` the class's <H_kc^T w, u>. A class whose share and every
// difference are exactly 0.0 (its inputs vanish by the case's symmetry) is
// reported as such and passes; anything else must converge by --kappa-full's
// rule. Prints one row; returns whether it holds.
bool pkcClassSweep(const char* name, double want, double base,
                   const std::array<double, kPkcFdSteps>& plus,
                   const std::array<double, kPkcFdSteps>& minus) {
  std::array<double, kPkcFdSteps> err{};
  bool allZero = want == 0.0;
  for (std::size_t i = 0; i < kPkcFdSteps; ++i) {
    const double h = std::ldexp(1.0, -(kPkcFdFirstExp + static_cast<int>(i)));
    const double d = (plus[i] - minus[i]) / (2.0 * h);
    allZero = allZero && d == 0.0;
    err[i] = std::fabs(d - want);
  }
  if (allZero) {
    std::printf("  %-7s %22s   exactly 0 on both sides, at every step\n", name, "0");
    return true;
  }
  const double scale = std::max(std::fabs(want), std::fabs(base));
  std::size_t bestAt = 0;
  for (std::size_t i = 1; i < kPkcFdSteps; ++i)
    if (err[i] < err[bestAt]) bestAt = i;
  const double bestRel = err[bestAt] / std::max(1e-30, scale);
  const nevpt2::FourfoldRun run = nevpt2::longestFourfoldRun(err);
  const bool ok = bestRel < kPkcFdTol && run.length >= kPkcFdRun;
  std::printf("  %-7s %22.14e %12.3e %10.3e %6d %6d  %s\n", name, want, bestRel,
              std::ldexp(1.0, -(kPkcFdFirstExp + static_cast<int>(bestAt))), run.length,
              kPkcFdFirstExp + static_cast<int>(run.end), ok ? "ok" : "FAILED");
  return ok;
}

// The per-class finite differences of <u, y_c(exp(s w))> against `share` (the
// class shares of H_kc^T w dotted with u), `yAt(s)` the three-term y's class
// shares at the displaced orbitals; `classes` the ones to difference (the
// others are printed as not displaced). Also the total. Returns whether every
// differenced class and the total converge.
nevpt2::Result<bool> pkcFdClasses(
    const std::array<double, 8>& share, const nevpt2::Tensor& u, const std::array<bool, 8>& classes,
    const std::function<nevpt2::Result<std::array<nevpt2::Tensor, 8>>(double)>& yAt) {
  const auto dotted = [&](const std::array<nevpt2::Tensor, 8>& y) {
    std::array<double, 9> out{};
    for (std::size_t c = 0; c < 8; ++c) {
      out[c] = kcDot(u, y[c]);
      out[8] += out[c];
    }
    return out;
  };
  const std::array<double, 9> base = dotted(NEVPT2_TRY(yAt(0.0)));
  std::array<std::array<double, kPkcFdSteps>, 9> plus{}, minus{};
  for (std::size_t i = 0; i < kPkcFdSteps; ++i) {
    const double h = std::ldexp(1.0, -(kPkcFdFirstExp + static_cast<int>(i)));
    const std::array<double, 9> p = dotted(NEVPT2_TRY(yAt(h)));
    const std::array<double, 9> m = dotted(NEVPT2_TRY(yAt(-h)));
    for (std::size_t c = 0; c < 9; ++c) {
      plus[c][i] = p[c];
      minus[c][i] = m[c];
    }
  }
  std::printf("  %-7s %22s %12s %10s %6s %6s\n", "class", "<H_kc^T w, u>_c", "best rel", "at h",
              "run", "ends k");
  bool ok = true;
  double total = 0.0;
  for (std::size_t c = 0; c < 8; ++c) {
    total += share[c];
    if (c == 2) {
      std::printf("  %-7s %22.14e   reads no RDM: no CI source term to difference\n",
                  nevpt2::CLASSES[c].c_str(), share[c]);
      ok = ok && share[c] == 0.0;
      continue;
    }
    if (!classes[c]) {
      std::printf("  %-7s %22.14e   not differenced: this space indexes none of its inputs\n",
                  nevpt2::CLASSES[c].c_str(), share[c]);
      continue;
    }
    ok = pkcClassSweep(nevpt2::CLASSES[c].c_str(), share[c], base[c], plus[c], minus[c]) && ok;
  }
  ok = pkcClassSweep("total", total, base[8], plus[8], minus[8]) && ok;
  return ok;
}

// <u, H_kc^T w>'s class shares.
std::array<double, 8> pkcShares(const nevpt2::PcMixedHessianVectorResult& r,
                                const nevpt2::Tensor& u) {
  std::array<double, 8> out{};
  for (std::size_t c = 0; c < 8; ++c) out[c] = kcDot(u, r.hkcTwClass[c]);
  return out;
}

// The three-term y's class shares, or a Numerical Error naming the refusal.
nevpt2::Result<std::array<nevpt2::Tensor, 8>> pkcThreeTermY(
    const nevpt2::NevptIntegralsDevice& ints, const nevpt2::GoldenFile& g,
    const nevpt2::RdmTensors& rdm, const nevpt2::SigmaVectorOptions& sigma,
    const nevpt2::DeviceResources& res) {
  const nevpt2::PcCiHessianResult r = NEVPT2_TRY(nevpt2::pcCiSourceThreeTermDevice(
      ints, g.get("ci"), g.nelecA, g.nelecB, rdm, /*perClass=*/true, sigma, res));
  if (!r.ok)
    return nevpt2::err_numerical(
        std::format("the three-term PC y has no value: {} refused", r.walk.refusals()));
  return r.yClass;
}

// Check 7: FROZEN, no sidecar, an active-active w.
nevpt2::Result<bool> pkcWithoutSidecar(const nevpt2::GoldenFile& g,
                                       const nevpt2::NevptIntegralsDevice& ints,
                                       const nevpt2::RdmTensors& rdm,
                                       const nevpt2::RdmBuildOptions& opt,
                                       const nevpt2::SigmaVectorOptions& sigma,
                                       const nevpt2::HostDm12Builder& builder,
                                       const std::vector<std::pair<std::string, nevpt2::Tensor>>& us,
                                       const nevpt2::DeviceResources& res) {
  const std::int64_t nc = g.ncore, na = g.ncas;
  const std::int64_t nmo = nc + na + g.get("e_virt").size();
  const nevpt2::Tensor& ci = g.get("ci");
  const nevpt2::ScMixedHessianSetup frozen =
      kcSetup(g, ints, /*gen=*/nullptr, rdm, opt, false, false, nullptr);
  std::printf("\n  [7] WITHOUT THE SIDECAR -- mode %s\n", kcModeName(false, false));
  bool ok = true;

  // The refusals first: a w leaving the active space, and a responding mode.
  {
    const nevpt2::Tensor wMixed = nevpt2::antisymmetricKappa(nmo, kPkcSeedBase + 0x700u);
    const nevpt2::Result<nevpt2::PcMixedHessianVectorResult> r =
        nevpt2::pcMixedHessianVectorDevice(frozen, wMixed, false, sigma, res);
    const bool refused = !r && r.error().kind == nevpt2::ErrorKind::Unsupported;
    const nevpt2::ScMixedHessianSetup responding =
        kcSetup(g, ints, /*gen=*/nullptr, rdm, opt, true, false, nullptr);
    const nevpt2::Tensor wAct =
        embedBlock(nevpt2::antisymmetricKappa(na, kPkcSeedBase + 0x701u), nc, nmo);
    const nevpt2::Result<std::vector<nevpt2::PcMixedHessianResult>> r2 =
        nevpt2::pcMixedHessianDevice(responding, wAct, {us[1].second}, res);
    const bool refused2 = !r2 && r2.error().kind == nevpt2::ErrorKind::Unsupported;
    ok = ok && refused && refused2;
    std::printf("  a w leaving the active space: %s\n",
                refused ? "refused, Unsupported" : "NOT REFUSED");
    std::printf("  the orbital energies responding to kappa: %s\n",
                refused2 ? "refused, Unsupported" : "NOT REFUSED");
  }

  const nevpt2::Tensor kAA = nevpt2::antisymmetricKappa(na, kPkcSeedBase + 0x702u);
  const nevpt2::Tensor w = embedBlock(kAA, nc, nmo);
  const nevpt2::PcMixedHessianVectorResult hv =
      NEVPT2_TRY(nevpt2::pcMixedHessianVectorDevice(frozen, w, /*perClass=*/true, sigma, res));
  std::printf("  H_kc^T w along an active-active w (seed 0x%llx; max |h2e'[w]| = %.3e): "
              "|H_kc^T w| = %.10e\n",
              static_cast<unsigned long long>(kPkcSeedBase + 0x702u), hv.h2eDotMax,
              std::sqrt(kcDot(hv.hkcTw, hv.hkcTw)));
  {
    const nevpt2::PcMixedHessianResult f1 =
        NEVPT2_TRY(nevpt2::pcMixedHessianDevice(frozen, w, {us[1].second}, res))[0];
    std::printf("  Eddot_kc(w, u) along %s, per class:\n", us[1].first.c_str());
    pkcPrintClasses(f1);
  }
  const double dual = NEVPT2_TRY(pkcDuality(frozen, w, hv.hkcTw, us, res));
  ok = ok && dual < kPkcTol;
  std::printf("  duality, worst |delta| = %.2e (< %.0e: %s)\n", dual, kPkcTol,
              dual < kPkcTol ? "ok" : "FAILED");

  // THE ACTIVE-ROTATION GAUGE (Sec. 6.8), per class, both sides FROZEN:
  // Eddot_kc(w, u) == Eddot_cc(what c, u) + <y, what u>, and at first order
  // Edot(w) == <y, what c>, what = sum_pq w_pq E_pq.
  {
    const nevpt2::Tensor h2e = g.get("h2e");
    const std::array<nevpt2::Tensor, 8> y = NEVPT2_TRY(pkcThreeTermY(ints, g, rdm, sigma, res));
    const nevpt2::Tensor wc = pkcOneBody(builder, kAA, ci);
    const nevpt2::PcRdms base{&rdm.dm1, &rdm.dm2, &rdm.dm3, &rdm.f3ac, &rdm.f3ca};
    const PkcSet dotW = pkcTangentSet(opt, builder, ci, wc, h2e, res);
    const PkcSet ddotWW = pkcTangentSet(opt, builder, wc, wc, h2e, res);
    std::printf("\n  the active-rotation gauge, FROZEN on both sides: Eddot_kc(w, u) == "
                "Eddot_cc(what c, u) + <y, what u>\n");
    // First order, from the forward walk's own Edot(w).
    const nevpt2::PcMixedHessianResult f0 =
        NEVPT2_TRY(nevpt2::pcMixedHessianDevice(frozen, w, {us[1].second}, res))[0];
    double worst1 = 0.0;
    for (std::size_t c = 0; c < 8; ++c)
      worst1 = std::max(worst1, std::fabs(f0.walk.classes[c].mixedEdotW - kcDot(y[c], wc)));
    std::printf("  first order, Edot(w) == <y, what c>, worst class |delta| = %.2e "
                "(Edot(w) = %.10e)\n",
                worst1, f0.edotW);
    ok = ok && worst1 < kPkcTol;
    double worst = 0.0;
    for (std::size_t k = 0; k + 1 < us.size(); ++k) {
      const nevpt2::Tensor& u = us[k].second;
      const PkcSet dotU = pkcTangentSet(opt, builder, ci, u, h2e, res);
      const PkcSet ddotWU = pkcTangentSet(opt, builder, wc, u, h2e, res);
      const PkcSet ddotUU = pkcTangentSet(opt, builder, u, u, h2e, res);
      const nevpt2::PcRdms vWU = ddotWU.view(), vWW = ddotWW.view(), vUU = ddotUU.view();
      const nevpt2::PcHessianResult cc = NEVPT2_TRY(nevpt2::pcHessianDevice(
          ints, nevpt2::PcHessianSets{base, dotW.view(), dotU.view(), &vWU, &vWW, &vUU}, res));
      for (std::size_t c = 0; c < 8; ++c)
        if (cc.classes[c].result.status != nevpt2::PcStatus::Done)
          return nevpt2::err_numerical(
              std::format("the gauge's Eddot_cc(what c, u) has no value: {} refused ({})",
                          nevpt2::CLASSES[c], cc.classes[c].result.why));
      const nevpt2::PcMixedHessianResult f =
          NEVPT2_TRY(nevpt2::pcMixedHessianDevice(frozen, w, {u}, res))[0];
      const nevpt2::Tensor wu = pkcOneBody(builder, kAA, u);
      std::printf("  u = %s:\n  %-7s %22s %22s %22s %11s %13s\n", us[k].first.c_str(), "class",
                  "Eddot_kc(w, u)", "Eddot_cc(what c, u)", "<y, what u>", "|delta|", "rotation");
      for (std::size_t c = 0; c < 8; ++c) {
        const double rhs = cc.classes[c].eddot + kcDot(y[c], wu);
        const double d = std::fabs(f.walk.classes[c].mixedEddot - rhs);
        worst = std::max(worst, d);
        std::printf("  %-7s %22.14e %22.14e %22.14e %11.2e %13.5e\n", nevpt2::CLASSES[c].c_str(),
                    f.walk.classes[c].mixedEddot, cc.classes[c].eddot, kcDot(y[c], wu), d,
                    f.walk.classes[c].mixedRotation);
      }
    }
    ok = ok && worst < kPkcTol;
    std::printf("  the gauge, worst class |delta| = %.2e (< %.0e: %s)\n", worst, kPkcTol,
                worst < kPkcTol ? "ok" : "FAILED");
  }

  // The finite difference of the three-term y, per class, along exp(s w).
  {
    const nevpt2::Tensor& u = us[1].second;
    std::printf("\n  central differences of <u, y_c> along exp(s w), u = %s, the active axes "
                "rotated and the\n  digests rebuilt, the orbital energies frozen (every class "
                "differenced):\n",
                us[1].first.c_str());
    const std::array<bool, 8> all{true, true, true, true, true, true, true, true};
    const bool fd = NEVPT2_TRY(pkcFdClasses(
        pkcShares(hv, u), u, all,
        [&](double s) -> nevpt2::Result<std::array<nevpt2::Tensor, 8>> {
          const nevpt2::Tensor rot = nevpt2::expKappa(scaled(kAA, s));
          const nevpt2::NevptIntegralsDevice displaced = uploadRotated(g, rot, res);
          const nevpt2::RdmBuildResult b = nevpt2::buildRdmsDevice(
              opt, ci, g.ncas, g.nelecA, g.nelecB,
              nevpt2::rotateAxes(g.get("h2e"), {0, 1, 2, 3}, rot), res);
          const nevpt2::RdmTensors r{rdm.dm1, rdm.dm2, rdm.dm3, b.f3ac, b.f3ca};
          return pkcThreeTermY(displaced, g, r, sigma, res);
        }));
    ok = ok && fd;
  }
  return ok;
}

// Runs every check above; `gg` null runs check 7 alone. Returns whether every
// criterion held; anything malformed is an Error for main to report once.
nevpt2::Result<bool> checkPcHessianKc(const nevpt2::GoldenFile& g,
                                      const nevpt2::GradGoldFile* gg,
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
  const bool fitted = gg != nullptr && gg->densityFitted();
  std::printf("\n--pc --sc-hessian-kc: the PC energy's mixed orbital-CI Hessian block H_kc "
              "(docs/hessian-theory.md, Sec. 6.4-6.8)%s\n",
              fitted ? " -- density-fitted" : "");
  std::printf("  F  = Eddot_kc(w, u), forward over forward; R1 = H_kc^T w, the three-term y "
              "differentiated along w\n");
  const nevpt2::HostDm12Builder builder =
      NEVPT2_TRY(nevpt2::HostDm12Builder::create(g.ncas, g.nelecA, g.nelecB));
  std::vector<std::pair<std::string, nevpt2::Tensor>> us;
  us.emplace_back(std::format("Epq 0x{:x}", kPkcEpqSeed),
                  pkcEpqDirection(builder, ci, g.get("dm1"), kPkcEpqSeed));
  for (const std::uint64_t seed : kPkcRotSeeds)
    us.emplace_back(std::format("rot 0x{:x}", seed), pkcRotDirection(builder, ci, seed));
  us.emplace_back("the state itself", ci);
  std::printf("  u: %s (sum f_pq E_pq |c>, f on the state's symmetry mask), %s and %s (f "
              "antisymmetric over every active pair), and the state\n",
              us[0].first.c_str(), us[1].first.c_str(), us[2].first.c_str());

  if (gg == nullptr) {
    std::printf("  no gradient sidecar: the active-active block alone, the orbital energies "
                "frozen\n");
    const bool ok = NEVPT2_TRY(pkcWithoutSidecar(g, ints, rdm, opt, sigma, builder, us, res));
    std::printf("\n%s: the PC mixed Hessian block H_kc's two orientations agree by duality to "
                "%.0e (no sidecar: an active-active w, the orbital energies frozen on both legs, "
                "every other w refused), the active-rotation gauge holds, and the finite "
                "difference of the three-term y converges in every class (orbital-energy mode: "
                "FROZEN)\n",
                ok ? "PASS" : "FAIL", kPkcTol);
    return ok;
  }

  const nevpt2::GeneralIntegralsDevice gen = NEVPT2_TRY(generalIntegrals(g, *gg, res));
  const std::int64_t nc = gg->ncore, na = gg->nact, nv = gg->nvirt, nmo = gg->nmo;
  bool ok = true;
  const nevpt2::ScMixedHessianSetup euler =
      kcSetup(g, ints, &gen, rdm, opt, /*epsKappa=*/true, /*epsCi=*/false, nullptr);
  const nevpt2::Tensor w = nevpt2::antisymmetricKappa(nmo, kPkcSeedBase);
  std::printf("\n  mode %s\n", kcModeName(true, false));
  std::printf("  w: a seeded fully mixed antisymmetric kappa (max|w + w^T| = %.1e), seed 0x%llx\n",
              nevpt2::antisymmetryResidual(w), static_cast<unsigned long long>(kPkcSeedBase));

  // --- 1. PC's first-order source term along w -------------------------------------
  const nevpt2::DeviceTensor hbar =
      NEVPT2_TRY(nevpt2::pcDigestAdjointDevice(opt, ci, g.nelecA, g.nelecB, ints, dm1, dm2,
                                               built.dm3, built.f3ac, built.f3ca, res));
  const std::vector<nevpt2::ScRotationRequest> req{nevpt2::ScRotationRequest{w, true}};
  const nevpt2::PcFullRotationResult y1 = NEVPT2_TRY(nevpt2::pcFullRotationDevice(
      ints, gen, dm1, dm2, built.dm3, built.f3ac, built.f3ca, &hbar, req, res));
  if (!y1.ok)
    return nevpt2::err_numerical(
        std::format("the PC orbital source term has no value: {} refused", y1.walk.refusals()));
  const std::vector<nevpt2::ScDigestTangent> dig =
      NEVPT2_TRY(nevpt2::pcDigestRotationTangentDevice(opt, ci, g.nelecA, g.nelecB, ints, gen,
                                                       dm1, dm2, built.dm3, built.f3ac,
                                                       built.f3ca, req, res));
  std::array<double, 8> yW{};  // <w, Y_pc> per class, the forward tangent's
  for (std::size_t c = 0; c < 8; ++c)
    yW[c] = y1.walk.classes[c].rotTangent[0] + y1.walk.classes[c].rotEps[0] + dig[0].classes[c];
  const auto t0 = std::chrono::steady_clock::now();
  const nevpt2::PcMixedHessianVectorResult hv =
      NEVPT2_TRY(nevpt2::pcMixedHessianVectorDevice(euler, w, /*perClass=*/false, sigma, res));
  std::printf("  R1 built in %.2fs: |H_kc^T w| = %.10e  (|walk tangent| = %.10e, |digests at "
              "h2e'[w]| = %.10e)\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
              std::sqrt(kcDot(hv.hkcTw, hv.hkcTw)),
              std::sqrt(kcDot(hv.fromWalkTangent, hv.fromWalkTangent)),
              std::sqrt(kcDot(hv.fromDigests, hv.fromDigests)));
  const std::vector<nevpt2::PcMixedHessianResult> f1 =
      NEVPT2_TRY(nevpt2::pcMixedHessianDevice(euler, w, {us[1].second}, res));
  {
    const nevpt2::ScFullRotationPairing& p = y1.dirs[0];
    const double edot = p.tangent + dig[0].total;
    const double dFull = std::fabs(p.adjoint - edot);
    std::printf("\n  [1] PC's first-order orbital source term along w%s: <w, Y_pc> = %.14e, "
                "Edot = %.14e, |delta| %.2e\n",
                fitted ? ", over the fitted inventory" : "", p.adjoint, edot, dFull);
    std::printf("  %-7s %22s %22s %11s %22s %11s\n", "class", "first order", "F walk's Edot(w)",
                "|delta|", "R1 walk's Edot(w)", "|delta|");
    double worst = dFull;
    for (std::size_t c = 0; c < 8; ++c) {
      if (c == 2) {
        // Sijrs is SC's class: no RDM, so no mixed walk -- its Edot(w) is the
        // orbital energies' response alone, and its H_kc share exactly zero.
        std::printf("  %-7s %22.14e   SC's class, reads no RDM: not walked (H_kc share 0)\n",
                    nevpt2::CLASSES[c].c_str(), yW[c]);
        continue;
      }
      const double df = std::fabs(f1[0].walk.classes[c].mixedEdotW - yW[c]);
      const double dr = std::fabs(hv.walk.classes[c].mixedEdotW - yW[c]);
      worst = std::max({worst, df, dr});
      std::printf("  %-7s %22.14e %22.14e %11.2e %22.14e %11.2e\n", nevpt2::CLASSES[c].c_str(),
                  yW[c], f1[0].walk.classes[c].mixedEdotW, df, hv.walk.classes[c].mixedEdotW, dr);
    }
    ok = ok && worst < kPkcTol;
    std::printf("  worst |delta| = %.2e (< %.0e: %s)\n", worst, kPkcTol,
                worst < kPkcTol ? "ok" : "FAILED");
  }

  // --- 2. the duality ------------------------------------------------------------
  {
    std::printf("\n  [2] the duality <H_kc^T w, u> == Eddot_kc(w, u)\n");
    std::printf("  Eddot_kc(w, u) along %s, per class:\n", us[1].first.c_str());
    pkcPrintClasses(f1[0]);
    const double dual = NEVPT2_TRY(pkcDuality(euler, w, hv.hkcTw, us, res));
    ok = ok && dual < kPkcTol;
    std::printf("  duality, worst |delta| = %.2e (< %.0e: %s)\n", dual, kPkcTol,
                dual < kPkcTol ? "ok" : "FAILED");
  }

  // --- 3. Euler in c ---------------------------------------------------------------
  {
    std::printf("\n  [3] Euler in c, Eddot_kc(w, c) == 2 <w, Y_pc> -- mode %s\n",
                kcModeName(true, false));
    const nevpt2::PcMixedHessianResult fc =
        NEVPT2_TRY(nevpt2::pcMixedHessianDevice(euler, w, {ci}, res))[0];
    std::printf("  %-7s %22s %22s %11s %13s\n", "class", "Eddot_kc(w, c)", "2 <w, Y_pc>",
                "|delta|", "rotation");
    double worst = 0.0;
    for (std::size_t c = 0; c < 8; ++c) {
      const double got = fc.walk.classes[c].mixedEddot;
      const double d = std::fabs(got - 2.0 * yW[c]);
      std::string verdict = "(affine)";
      if (pkcHomogeneous(c)) {
        worst = std::max(worst, d);
        verdict = std::format("{:.2e}", d);
      } else if (c == 2) {
        ok = ok && got == 0.0;
        verdict = got == 0.0 ? "0 exactly" : "NOT ZERO";
      }
      std::printf("  %-7s %22.14e %22.14e %11s %13.5e\n", nevpt2::CLASSES[c].c_str(), got,
                  2.0 * yW[c], verdict.c_str(), fc.walk.classes[c].mixedRotation);
    }
    ok = ok && worst < kPkcTol;
    std::printf("  Euler, worst homogeneous class |delta| = %.2e (< %.0e: %s)\n", worst, kPkcTol,
                worst < kPkcTol ? "ok" : "FAILED");
  }

  // --- 4 and 5. the intra-space zeros and the finite differences ------------------
  const nevpt2::Tensor& uFd = us[1].second;
  for (int which = 0; which < 2; ++which) {
    const bool core = which == 0;
    const char* name = core ? "core" : "virtual";
    const std::int64_t n = core ? nc : nv;
    if (n < 2) {
      std::printf("\n  [4/5] %s-internal: skipped, %lld orbital(s) leave no rotation\n", name,
                  static_cast<long long>(n));
      continue;
    }
    const nevpt2::Tensor block =
        nevpt2::antisymmetricKappa(n, kPkcSeedBase + (core ? 0x500u : 0x600u));
    const nevpt2::Tensor wSpace = embedBlock(block, core ? 0 : nc + na, nmo);
    const nevpt2::PcMixedHessianVectorResult hvS = NEVPT2_TRY(
        nevpt2::pcMixedHessianVectorDevice(euler, wSpace, /*perClass=*/true, sigma, res));
    const nevpt2::PcMixedHessianResult f =
        NEVPT2_TRY(nevpt2::pcMixedHessianDevice(euler, wSpace, {uFd}, res))[0];
    double rotMax = 0.0;
    for (const nevpt2::PcClassGradient& k : f.walk.classes)
      rotMax = std::max(rotMax, std::fabs(k.mixedRotation));
    const bool zeros = f.h2eDotMax == 0.0 && rotMax < kPkcRotZeroTol;
    const double dual = std::fabs(kcDot(uFd, hvS.hkcTw) - f.eddot);
    ok = ok && zeros && dual < kPkcTol;
    std::printf("\n  [4] a %s-internal w: max |h2e'[w]| = %.1e (exactly 0), max |h1e'[w]| = "
                "%.1e; u = %s\n",
                name, f.h2eDotMax, f.h1eDotMax, us[1].first.c_str());
    pkcPrintClasses(f);
    std::printf("  the rotation term, worst class %.1e (< %.0e): %s; the duality |delta| = %.2e "
                "(< %.0e: %s)\n",
                rotMax, kPkcRotZeroTol, zeros ? "ok" : "FAILED", dual, kPkcTol,
                dual < kPkcTol ? "ok" : "FAILED");
    std::printf("  [5] central differences of <u, y_c> along exp(s w), %s-internal -- the "
                "stored blocks' %s axes\n  and that space's orbital energies rotated on the "
                "host, the RDMs and digests the state's own:\n",
                name, name);
    // The classes whose inputs carry this space's index.
    const std::array<bool, 8> indexed =
        core ? std::array<bool, 8>{false, true, true, true, true, false, true, true}
             : std::array<bool, 8>{true, false, true, true, true, true, false, true};
    const bool fd = NEVPT2_TRY(pkcFdClasses(
        pkcShares(hvS, uFd), uFd, indexed,
        [&](double s) -> nevpt2::Result<std::array<nevpt2::Tensor, 8>> {
          const nevpt2::Tensor rot = nevpt2::expKappa(scaled(block, s));
          const nevpt2::NevptIntegralsDevice displaced = uploadSpaceRotated(g, *gg, core, rot, res);
          return pkcThreeTermY(displaced, g, rdm, sigma, res);
        }));
    ok = ok && fd;
  }

  // --- 6. the refusals ---------------------------------------------------------------
  {
    std::printf("\n  [6] the refusals\n");
    // A random amplitude vector, projected against c: neither spin-adapted nor
    // in the state's symmetry sector, so S's dropped modes lift along it.
    const nevpt2::Tensor uRand = kcDirection(ci, kPkcControlSeed);
    const nevpt2::Result<std::vector<nevpt2::PcMixedHessianResult>> r =
        nevpt2::pcMixedHessianDevice(euler, w, {uRand}, res);
    const bool lifted = !r && r.error().kind == nevpt2::ErrorKind::Numerical &&
                        r.error().message.find("PC_LIFT_TOL") != std::string::npos;
    ok = ok && lifted;
    std::printf("  a random amplitude u (seed 0x%llx): %s\n",
                static_cast<unsigned long long>(kPkcControlSeed),
                lifted ? std::format("refused, Numerical (\"{}\")", r.error().message).c_str()
                       : "NOT REFUSED for its lift");
    const nevpt2::OrbitalEnergyResponse wResp =
        fitted ? nevpt2::OrbitalEnergyResponse{}
               : nevpt2::orbitalEnergyResponse(nc, na, nv, gg->get("g_xa_cc"),
                                               gg->get("g_xc_ca"), gg->get("g_xa_vv"),
                                               gg->get("g_xv_va"));
    const nevpt2::ScMixedHessianSetup full =
        kcSetup(g, ints, &gen, rdm, opt, true, true, fitted ? nullptr : &wResp);
    const nevpt2::Result<std::vector<nevpt2::PcMixedHessianResult>> r2 =
        nevpt2::pcMixedHessianDevice(full, w, {us[1].second}, res);
    const nevpt2::Result<nevpt2::PcMixedHessianVectorResult> r3 =
        nevpt2::pcMixedHessianVectorDevice(full, w, false, sigma, res);
    const bool refused = !r2 && r2.error().kind == nevpt2::ErrorKind::Unsupported && !r3 &&
                         r3.error().kind == nevpt2::ErrorKind::Unsupported;
    ok = ok && refused;
    std::printf("  the orbital energies responding to c (FULL), both orientations: %s\n",
                refused ? "refused, Unsupported" : "NOT REFUSED");
  }

  // --- 7. without the sidecar ----------------------------------------------------------
  ok = NEVPT2_TRY(pkcWithoutSidecar(g, ints, rdm, opt, sigma, builder, us, res)) && ok;

  std::printf("\n%s: the PC mixed Hessian block H_kc's two orientations agree by duality to %.0e "
              "(EULER mode along a mixed w, the intra-space w and with no sidecar), Euler in c "
              "holds on the five homogeneous classes against the gated Y_pc, the "
              "active-rotation gauge holds, the intra-space rotation term vanishes, the finite "
              "differences of the three-term y converge in every class, and every refusal "
              "holds%s\n",
              ok ? "PASS" : "FAIL", kPkcTol, fitted ? " (density-fitted)" : "");
  return ok;
}
