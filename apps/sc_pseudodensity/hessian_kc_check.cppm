// nevpt2.app.sc_pseudodensity:hessian_kc_check -- --sc-hessian-kc: the SC
// energy's MIXED orbital-CI second derivative, the block H_kc of its
// parameter Hessian (docs/hessian-theory.md, Sec. 5.5; #184), in the two
// orientations Sec. 5.5 builds -- the scalar Eddot_kc(w, u), forward over
// forward (F), and the determinant vector H_kc^T w, reverse in the CI vector
// and forward in kappa (R1) -- checked against each other and against what
// neither of them shares.
// An interface partition that exports nothing, as :kappa_full_check is.
//
// WHY THIS BINARY. H_kc pairs an orbital direction with a CI one, and the
// orbital directions -- and the gradient sidecar that a w leaving the active
// space reads -- are this binary's (--kappa-act, --kappa-full); so are the
// displaced orbitals its one finite difference runs at. nevpt2_demo owns the
// CI directions, but a CI direction needs nothing a golden lacks.
//
// THE CHECKS, in the order they print. Every one names the orbital-energy
// mode it runs in (Sec. 5.5: "the mode is part of the check"): the FULL mode
// lets them respond to kappa and to the CI vector (C5 on); the EULER mode to
// kappa alone; the FROZEN mode -- the only one with no sidecar -- to neither.
//
//   1  THE FULL-RANGE f IDENTITY for C5's generator: h^eff + f^gamma[gamma]
//      at the state's gamma is the sidecar's f over the WHOLE MO range, the
//      off-diagonal extension of --kappa-full's diagonal-only check, beside
//      that check's own families (scGeneralIntegralCheck).
//   2  THE DIGESTS' SHARE, MERGED PER TUPLE (full mode): the walk's own
//      Edot(w) -- P.Ndot + Q.Hdot + R.Ddot with every tuple's Hdot carrying
//      the digests' tangent -- against the first-order source term the tree
//      already gates, the frozen-digest tangent plus #88's separate digest
//      walk, class by class.
//   3  THE DUALITY (full mode): <H_kc^T w, u> == Eddot_kc(w, u) along a
//      seeded fully mixed w, for several seeded u and the state itself --
//      and, where ndet allows, COLUMN BY COLUMN, u = e_I for every
//      determinant. Exact arithmetic, the tree's 1e-7.
//   4  EULER IN c (Euler mode): a class homogeneous of degree 2 in c (Sr,
//      Srsi, Srs) has Eddot_kc(w, c) == 2 <w, Y>, against the already-gated
//      first-order Y (the forward tangent plus #88's term) -- the check that
//      sees a term both orientations omit, C4 included (u = c turns it into
//      the --rdm-tangent identity at dotted integrals). Sijrs, whose only
//      route to c is the orbital energies, is exactly zero in this mode. The
//      duality is held in this mode too.
//   5  THE INTRA-SPACE ZEROS (full mode): along a core-core and a
//      virtual-virtual w no active integral moves, so C4 and C1's digest
//      share vanish IDENTICALLY (h2e'[w] is an exactly-zero tensor), and C3
//      with them up to the core dressing's round-off in h1e'[w] -- while C1,
//      C2 and C5 do not. The duality is held along both.
//   6  ONE FINITE DIFFERENCE, of the gated y: a Richardson-verified central
//      difference of scCiSourceDevice along exp(s w), dotted with u, against
//      <H_kc^T w, u> -- along a core-core and a virtual-virtual w (the blocks,
//      the orbital energies and W rotated on the host out of stored arrays,
//      as --kappa-full displaces) and an active-active one (the active axes
//      rotated, as --kappa-act displaces; W rotated on its active pair and
//      the digests rebuilt). Both legs in the full mode: the only check that
//      sees C5.
//   7  WITHOUT THE SIDECAR (frozen mode, the sidecar not handed in): an
//      active-active w, the duality, Euler against scActiveRotationDevice's
//      tangent plus the digests' term, the finite difference of y with the
//      orbital energies frozen on both legs -- and the REFUSALS: a w that
//      leaves the active space, or a responding mode, is Unsupported.
//
// With no sidecar beside the golden (and no --gradgold) only check 7 runs,
// and the PASS line says so.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries. A malformed direction or a
// sidecar that does not fit the golden is the caller's input and propagates
// unchanged to main, which reports it once.
#include "error_handling/error_macros.h"

export module nevpt2.app.sc_pseudodensity:hessian_kc_check;

import std;
import nevpt2.gradient;
import nevpt2.kappa_rotation;
import nevpt2.golden;
// --kappa-full's sidecar upload and its intra-space displacement, and
// --kappa-act's active one: the displaced orbitals the finite difference
// runs at are theirs, not a copy.
import :kappa_full_check;
import :kappa_check;

// The duality's and every exact identity's gate: an identity between two
// computations, the tree's usual 1e-7 (docs/testing.md, "The SC mixed
// Hessian block's checks", records what is measured).
constexpr double kKcTol = 1e-7;
// The sidecar self-check, as --kappa-full's: two PySCF transforms of the same
// integrals, reassociation only.
constexpr double kKcSelfTol = 1e-10;
// The column-by-column check's bound on ndet: --gradient-source's Jacobian's
// and --sc-hessian-vector's, for the same reason (one F walk per column).
constexpr std::int64_t kKcColumnMaxDet = 256;
// Seeds, fixed so a run is reproducible, distinct from every other check's.
constexpr std::uint64_t kKcSeedBase = 0x5eed'0184u;
// The CI directions of the duality.
constexpr std::array<std::uint64_t, 3> kKcUSeeds = {0x5eed'1184u, 0x5eed'2184u, 0x5eed'3184u};
// C3 along an intra-space w: h2e'[w] is an exactly-zero tensor, but h1e'[w]
// (h^eff's active block) carries the core dressing's response, whose two
// promotions of a tied core pair cancel against an antisymmetric kappa only
// to round-off in the einsum's accumulation order. So C3 is round-off, not
// 0.0, there: held absolutely to this, decades above what is measured and
// decades below any real term (docs/testing.md, "The SC mixed Hessian
// block's checks").
constexpr double kKcC3ZeroTol = 1e-13;
// The finite difference of y: the same ladder, run rule and relative gate as
// --kappa-full's sweeps (h = 2^-k for k in [6, 18], the best error relative
// to max(|<H_kc^T w, u>|, |<u, y>|) -- the round-off floor of differencing
// <u, y> is set by its own magnitude -- under 1e-6, and a run of three
// fourfold halvings).
constexpr int kKcFdFirstExp = 6;
constexpr int kKcFdLastExp = 18;
constexpr std::size_t kKcFdSteps = kKcFdLastExp - kKcFdFirstExp + 1;
constexpr double kKcFdTol = 1e-6;
constexpr int kKcFdRun = 3;

// A seeded direction in ci's space, projected against ci and normalized --
// nevpt2.ci_checks' tangentDirection, the same construction.
nevpt2::Tensor kcDirection(const nevpt2::Tensor& ci, std::uint64_t seed) {
  nevpt2::Tensor u(ci.dims());
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::int64_t i = 0; i < u.size(); ++i) u.flatRef(i) = dist(gen);
  double overlap = 0.0;
  for (std::int64_t i = 0; i < u.size(); ++i) overlap += ci.flat(i) * u.flat(i);
  for (std::int64_t i = 0; i < u.size(); ++i) u.flatRef(i) -= overlap * ci.flat(i);
  double norm2 = 0.0;
  for (std::int64_t i = 0; i < u.size(); ++i) norm2 += u.flat(i) * u.flat(i);
  for (std::int64_t i = 0; i < u.size(); ++i) u.flatRef(i) /= std::sqrt(norm2);
  return u;
}

double kcDot(const nevpt2::Tensor& x, const nevpt2::Tensor& y) {
  double s = 0.0;
  for (std::int64_t i = 0; i < x.size(); ++i) s += x.flat(i) * y.flat(i);
  return s;
}

// The orbital-energy mode, as the PASS line and every check's heading say it.
const char* kcModeName(bool epsKappa, bool epsCi) {
  if (epsKappa && epsCi) return "FULL: the orbital energies respond to kappa and to c";
  if (epsKappa) return "EULER: the orbital energies respond to kappa, held fixed in c";
  return "FROZEN: the orbital energies held fixed in kappa and in c";
}

// One mode's setup.
nevpt2::ScMixedHessianSetup kcSetup(const nevpt2::GoldenFile& g,
                                    const nevpt2::NevptIntegralsDevice& ints,
                                    const nevpt2::GeneralIntegralsDevice* gen,
                                    const nevpt2::RdmTensors& rdm,
                                    const nevpt2::RdmBuildOptions& opt, bool epsKappa,
                                    bool epsCi, const nevpt2::OrbitalEnergyResponse* w) {
  return nevpt2::ScMixedHessianSetup{ints,   gen,      rdm,   g.get("ci"), g.nelecA,
                                     g.nelecB, opt,    epsKappa, epsCi,     w};
}

// The duality along w for each of `us`, printed: <H_kc^T w, u> against F's
// Eddot_kc(w, u). Returns the worst |delta|.
nevpt2::Result<double> kcDuality(const nevpt2::ScMixedHessianSetup& s, const nevpt2::Tensor& w,
                                 const nevpt2::Tensor& hkcTw,
                                 const std::vector<std::pair<std::string, nevpt2::Tensor>>& us,
                                 const nevpt2::DeviceResources& res) {
  std::vector<nevpt2::Tensor> dirs;
  for (const auto& [name, u] : us) dirs.push_back(u);
  const std::vector<nevpt2::ScMixedHessianResult> f =
      NEVPT2_TRY(nevpt2::scMixedHessianDevice(s, w, dirs, res));
  std::printf("  %-18s %22s %22s %11s\n", "u", "<H_kc^T w, u>", "Eddot_kc(w, u)", "|delta|");
  double worst = 0.0;
  for (std::size_t k = 0; k < us.size(); ++k) {
    const double lhs = kcDot(us[k].second, hkcTw);
    const double d = std::fabs(lhs - f[k].eddot);
    worst = std::max(worst, d);
    std::printf("  %-18s %22.14e %22.14e %11.2e\n", us[k].first.c_str(), lhs, f[k].eddot, d);
  }
  return worst;
}

// The five terms of one F result, per class, printed.
void kcPrintTerms(const nevpt2::ScMixedHessianResult& r) {
  std::printf("  %-7s %13s %13s %13s %13s %13s %13s %15s\n", "class", "C1", "(C1 digest)",
              "C2", "C3", "C4", "C5", "Eddot_kc");
  for (std::size_t c = 0; c < 8; ++c) {
    const nevpt2::ScMixedTerms& t = r.classes[c];
    std::printf("  %-7s %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %15.7e\n",
                nevpt2::CLASSES[c].c_str(), t.c1, t.c1Digest, t.c2, t.c3, t.c4, t.c5,
                t.total());
  }
  const nevpt2::ScMixedTerms& t = r.total;
  std::printf("  %-7s %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %15.7e\n", "total", t.c1,
              t.c1Digest, t.c2, t.c3, t.c4, t.c5, t.total());
}

// One central-difference sweep of `at(s)` = <u, y(exp(s w))> against `want`,
// printed, with --kappa-full's run rule. Returns whether both criteria hold.
bool kcSweep(double want, const std::function<double(double)>& at) {
  std::printf("  %5s %14s %22s %12s %12s %8s\n", "k", "h = 2^-k", "(<u,y+> - <u,y->)/2h",
              "|err|", "rel err", "ratio");
  const double base = at(0.0);
  const double scale = std::max(std::fabs(want), std::fabs(base));
  std::array<double, kKcFdSteps> step{};
  std::array<double, kKcFdSteps> err{};
  for (std::size_t i = 0; i < kKcFdSteps; ++i) {
    const int k = kKcFdFirstExp + static_cast<int>(i);
    const double h = std::ldexp(1.0, -k);
    step[i] = h;
    const double d = (at(h) - at(-h)) / (2.0 * h);
    err[i] = std::fabs(d - want);
    const double rel = err[i] / std::max(1e-30, scale);
    if (i == 0)
      std::printf("  %5d %14.3e %22.14e %12.3e %12.3e %8s\n", k, h, d, err[i], rel, "-");
    else
      std::printf("  %5d %14.3e %22.14e %12.3e %12.3e %8.2f\n", k, h, d, err[i], rel,
                  err[i] > 0.0 ? err[i - 1] / err[i] : 0.0);
  }
  std::size_t bestAt = 0;
  for (std::size_t i = 1; i < kKcFdSteps; ++i)
    if (err[i] < err[bestAt]) bestAt = i;
  const double bestRel = err[bestAt] / std::max(1e-30, scale);
  const nevpt2::FourfoldRun run = nevpt2::longestFourfoldRun(err);
  const bool ok = bestRel < kKcFdTol && run.length >= kKcFdRun;
  std::printf("  <H_kc^T w, u> = %.14e, <u, y> = %.14e; plateau |err| = %.3e (relative %.3e) "
              "at h = %.3e\n",
              want, base, err[bestAt], bestRel, step[bestAt]);
  std::printf("  Richardson: %d consecutive fourfold halvings (ratio in [%.1f, %.1f]) ending at "
              "h = %.3e -- %s, %d needed\n",
              run.length, nevpt2::kFourfoldLow, nevpt2::kFourfoldHigh, step[run.end],
              ok ? "ok" : "NOT ENOUGH", kKcFdRun);
  return ok;
}

// W -- the orbital energies' response to gamma -- at the orbitals rotated by
// `u` within ONE inactive space, out of the sidecar's untied rows on that
// space's own slice: W_p[a, b] = (pp|ab) - (1/2)(pa|bp) with p = sum_q U_qp q,
// so W(U)_p = sum_{q,q'} U_qp U_q'p A[q, q', a, b] over the space's full
// (q q'|ab) - (1/2)(q a|b q') block -- (cc|aa) / (ca|ca) for the core,
// (vv|aa) / (va|va) for the virtuals, both rows the sidecar stores in full.
// The other space's W is the state's own. In W's (nact, nact, n) layout.
nevpt2::OrbitalEnergyResponse kcRotatedResponse(const nevpt2::GradGoldFile& gg,
                                                const nevpt2::OrbitalEnergyResponse& base,
                                                bool core, const nevpt2::Tensor& u) {
  const std::int64_t nc = gg.ncore, na = gg.nact, nv = gg.nvirt;
  const std::int64_t n = core ? nc : nv;
  const std::int64_t lo = core ? 0 : nc + na;
  const nevpt2::Tensor& cou = gg.get(core ? "g_xc_aa" : "g_xv_aa");  // (x p | a b)
  const nevpt2::Tensor& exc = gg.get(core ? "g_xa_ca" : "g_xa_va");  // (x a | p b)
  nevpt2::Tensor a({n, n, na, na});
  for (std::int64_t q = 0; q < n; ++q)
    for (std::int64_t r = 0; r < n; ++r)
      for (std::int64_t s = 0; s < na; ++s)
        for (std::int64_t t = 0; t < na; ++t)
          a(q, r, s, t) = cou(lo + q, r, s, t) - 0.5 * exc(lo + q, s, r, t);
  const nevpt2::Tensor rotated = nevpt2::rotateAxes(a, {0, 1}, u);
  nevpt2::OrbitalEnergyResponse out = base;
  nevpt2::Tensor& w = core ? out.core : out.virt;
  for (std::int64_t s = 0; s < na; ++s)
    for (std::int64_t t = 0; t < na; ++t)
      for (std::int64_t p = 0; p < n; ++p) w(s, t, p) = rotated(p, p, s, t);
  return nevpt2::symmetrizedResponse(out);
}

// <u, y> at the orbitals exp(s w) for an intra-space w, the full mode: the
// stored blocks and the orbital energies as --kappa-full displaces them
// (uploadSpaceRotated), W rotated on the same space, the RDMs and the digests
// the state's own (h2e carries no core or virtual index).
double kcSourceSpaceRotated(const nevpt2::GoldenFile& g, const nevpt2::GradGoldFile& gg,
                            const nevpt2::OrbitalEnergyResponse& base, bool core,
                            const nevpt2::Tensor& block, double s, const nevpt2::Tensor& u,
                            const nevpt2::RdmTensors& rdm,
                            const nevpt2::SigmaVectorOptions& sigma,
                            const nevpt2::DeviceResources& res) {
  const nevpt2::Tensor rot = nevpt2::expKappa(scaled(block, s));
  const nevpt2::NevptIntegralsDevice ints = uploadSpaceRotated(g, gg, core, rot, res);
  const nevpt2::OrbitalEnergyResponse w = kcRotatedResponse(gg, base, core, rot);
  const nevpt2::Tensor y =
      nevpt2::scCiSourceDevice(ints, g.get("ci"), g.nelecA, g.nelecB, rdm.dm1, rdm.dm2,
                               rdm.dm3, rdm.f3ac, rdm.f3ca, sigma, &w, res)
          .y;
  return kcDot(u, y);
}

// <u, y> at exp(s w) for an active-active w: every stored block's active axes
// rotated (--kappa-act's uploadRotated), the digests rebuilt at the rotated
// h2e, and -- in the full mode, `base` non-null -- W rotated on its active
// pair and the orbital energies re-formed from it, e_p = h^eff_pp + <gamma,
// W_p> (h^eff_pp reads no active index). Null `base`: the frozen mode, the
// orbital energies the golden's and no response on this leg.
double kcSourceActiveRotated(const nevpt2::GoldenFile& g, const nevpt2::GradGoldFile* gg,
                             const nevpt2::OrbitalEnergyResponse* base,
                             const nevpt2::Tensor& kAA, double s, const nevpt2::Tensor& u,
                             const nevpt2::RdmTensors& rdm, const nevpt2::RdmBuildOptions& opt,
                             const nevpt2::SigmaVectorOptions& sigma,
                             const nevpt2::DeviceResources& res) {
  const nevpt2::Tensor rot = nevpt2::expKappa(scaled(kAA, s));
  nevpt2::NevptIntegralsDevice ints = uploadRotated(g, rot, res);
  const nevpt2::RdmBuildResult b = nevpt2::buildRdmsDevice(
      opt, g.get("ci"), g.ncas, g.nelecA, g.nelecB,
      nevpt2::rotateAxes(g.get("h2e"), {0, 1, 2, 3}, rot), res);
  std::optional<nevpt2::OrbitalEnergyResponse> w;
  if (base != nullptr) {
    w.emplace(*base);
    w->core = nevpt2::rotateAxes(base->core, {0, 1}, rot);
    w->virt = nevpt2::rotateAxes(base->virt, {0, 1}, rot);
    const nevpt2::OrbitalEnergyDot e =
        nevpt2::orbitalEnergiesFrom(*w, gg->get("heff"), g.get("dm1"));
    ints.e_core = nevpt2::uploadTensor(e.core, res);
    ints.e_virt = nevpt2::uploadTensor(e.virt, res);
  }
  const nevpt2::Tensor y =
      nevpt2::scCiSourceDevice(ints, g.get("ci"), g.nelecA, g.nelecB, rdm.dm1, rdm.dm2,
                               rdm.dm3, b.f3ac, b.f3ca, sigma, w ? &*w : nullptr, res)
          .y;
  return kcDot(u, y);
}

// Check 7: no sidecar, the frozen mode, along an active-active w. `gen` is
// the sidecar when there is one -- used ONLY to show the refusals hold even
// with a sidecar in hand elsewhere; every computation here passes null.
nevpt2::Result<bool> kcWithoutSidecar(const nevpt2::GoldenFile& g,
                                      const nevpt2::NevptIntegralsDevice& ints,
                                      const nevpt2::RdmTensors& rdm,
                                      const nevpt2::RdmBuildOptions& opt,
                                      const nevpt2::SigmaVectorOptions& sigma,
                                      const nevpt2::DeviceResources& res) {
  const std::int64_t nc = g.ncore, na = g.ncas;
  const std::int64_t nmo = nc + na + g.get("e_virt").size();
  const nevpt2::Tensor& ci = g.get("ci");
  const nevpt2::ScMixedHessianSetup frozen =
      kcSetup(g, ints, /*gen=*/nullptr, rdm, opt, false, false, nullptr);
  std::printf("\n  [7] WITHOUT THE SIDECAR -- mode %s\n", kcModeName(false, false));
  bool ok = true;

  // The refusals first: a w that leaves the active space, and a responding
  // mode, each Unsupported.
  {
    const nevpt2::Tensor wMixed = nevpt2::antisymmetricKappa(nmo, kKcSeedBase + 0x700u);
    const nevpt2::Result<nevpt2::ScMixedHessianVectorResult> r =
        nevpt2::scMixedHessianVectorDevice(frozen, wMixed, sigma, res);
    const bool refused = !r && r.error().kind == nevpt2::ErrorKind::Unsupported;
    const nevpt2::ScMixedHessianSetup responding =
        kcSetup(g, ints, /*gen=*/nullptr, rdm, opt, true, false, nullptr);
    const nevpt2::Tensor wAct = embedBlock(nevpt2::antisymmetricKappa(na, kKcSeedBase + 0x701u),
                                           nc, nmo);
    const nevpt2::Result<std::vector<nevpt2::ScMixedHessianResult>> r2 =
        nevpt2::scMixedHessianDevice(responding, wAct, {kcDirection(ci, kKcUSeeds[0])}, res);
    const bool refused2 = !r2 && r2.error().kind == nevpt2::ErrorKind::Unsupported;
    ok = ok && refused && refused2;
    std::printf("  a w leaving the active space: %s\n",
                refused ? std::format("refused, Unsupported (\"{:.60}...\")", r.error().message)
                              .c_str()
                        : "NOT REFUSED");
    std::printf("  the orbital energies responding to kappa: %s\n",
                refused2 ? "refused, Unsupported" : "NOT REFUSED");
  }

  const nevpt2::Tensor kAA = nevpt2::antisymmetricKappa(na, kKcSeedBase + 0x702u);
  const nevpt2::Tensor w = embedBlock(kAA, nc, nmo);
  const nevpt2::ScMixedHessianVectorResult hv =
      NEVPT2_TRY(nevpt2::scMixedHessianVectorDevice(frozen, w, sigma, res));
  std::printf("  H_kc^T w along an active-active w (max |h2e'[w]| = %.3e): |H_kc^T w| = %.10e\n",
              hv.h2eDotMax, std::sqrt(kcDot(hv.hkcTw, hv.hkcTw)));
  std::vector<std::pair<std::string, nevpt2::Tensor>> us;
  for (const std::uint64_t seed : kKcUSeeds)
    us.emplace_back(std::format("seed 0x{:x}", seed), kcDirection(ci, seed));
  us.emplace_back("the state itself", ci);
  const double dual = NEVPT2_TRY(kcDuality(frozen, w, hv.hkcTw, us, res));
  ok = ok && dual < kKcTol;
  std::printf("  duality, worst |delta| = %.2e (< %.0e: %s)\n", dual, kKcTol,
              dual < kKcTol ? "ok" : "FAILED");

  // Euler against the active block's own first-order Y: its forward tangent
  // (frozen digests, frozen orbital energies) plus the digests' term along
  // the same active direction -- the frozen-orbital-energy source term in full.
  const std::vector<nevpt2::ScRotationRequest> req{nevpt2::ScRotationRequest{kAA, true}};
  const nevpt2::ScRotationResult y1 = NEVPT2_TRY(nevpt2::scActiveRotationDevice(
      ints, rdm.dm1, rdm.dm2, rdm.dm3, rdm.f3ac, rdm.f3ca, req, res));
  const std::vector<nevpt2::ScDigestTangent> dig =
      NEVPT2_TRY(nevpt2::scDigestRotationTangentDevice(opt, ci, g.nelecA, g.nelecB, ints,
                                                       rdm.dm1, rdm.dm2, rdm.dm3, rdm.f3ac,
                                                       rdm.f3ca, req, res));
  const std::vector<nevpt2::ScMixedHessianResult> fc =
      NEVPT2_TRY(nevpt2::scMixedHessianDevice(frozen, w, {ci}, res));
  std::printf("  Euler, Eddot_kc(w, c) == 2 <w, Y> on the homogeneous classes:\n");
  std::printf("  %-7s %22s %22s %11s %22s %11s\n", "class", "Eddot_kc(w, c)", "2 <w, Y>",
              "|delta|", "walk's Edot(w)", "|delta|");
  for (std::size_t c = 0; c < 8; ++c) {
    const double y = y1.grad.classes[c].rotTangent[0] + dig[0].classes[c];
    const bool homogeneous = c == 0 || c == 4 || c == 5;
    const double d = std::fabs(fc[0].classes[c].total() - 2.0 * y);
    const double dw = std::fabs(fc[0].edotW[c] - y);
    if (homogeneous) ok = ok && d < kKcTol;
    ok = ok && dw < kKcTol;
    std::printf("  %-7s %22.14e %22.14e %11s %22.14e %11.2e\n", nevpt2::CLASSES[c].c_str(),
                fc[0].classes[c].total(), 2.0 * y,
                homogeneous ? std::format("{:.2e}", d).c_str() : "(affine)", fc[0].edotW[c], dw);
  }

  // The finite difference of y, the orbital energies frozen on both legs.
  const nevpt2::Tensor& u = us[0].second;
  std::printf("  central differences of <u, y> along exp(s w), the active axes rotated and the "
              "digests rebuilt,\n  the orbital energies frozen and no response on either leg:\n");
  const bool fd = kcSweep(kcDot(u, hv.hkcTw), [&](double s) {
    return kcSourceActiveRotated(g, nullptr, nullptr, kAA, s, u, rdm, opt, sigma, res);
  });
  ok = ok && fd;
  return ok;
}

// Runs every check above; `gg` null runs check 7 alone. Returns whether every
// criterion held; anything malformed is an Error for main to report once.
nevpt2::Result<bool> checkScHessianKc(const nevpt2::GoldenFile& g,
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
  std::printf("\n--sc-hessian-kc: the SC energy's mixed orbital-CI Hessian block H_kc "
              "(docs/hessian-theory.md, Sec. 5.5)\n");
  std::printf("  F  = Eddot_kc(w, u), forward over forward; R1 = H_kc^T w, reverse in c, "
              "forward in kappa\n");

  if (gg == nullptr) {
    std::printf("  no gradient sidecar: the active-active block alone, the orbital energies "
                "frozen\n");
    const bool ok = NEVPT2_TRY(kcWithoutSidecar(g, ints, rdm, opt, sigma, res));
    std::printf("\n%s: the SC mixed Hessian block H_kc's two orientations agree by duality to "
                "%.0e (no sidecar: an active-active w, the orbital energies frozen on both legs, "
                "every other w refused), Euler in c holds on the homogeneous classes, and the "
                "finite difference of y converges\n",
                ok ? "PASS" : "FAIL", kKcTol);
    return ok;
  }

  if (gg->densityFitted())
    return nevpt2::err_unsupported(
        "--sc-hessian-kc on a density-fitted case: the mixed block is integral-direct, as "
        "scFullRotationDevice is (docs/hessian-theory.md, Sec. 5.5's Data), and its checks "
        "are not gated on a fitted sidecar");
  const nevpt2::GeneralIntegralsDevice gen = NEVPT2_TRY(generalIntegrals(g, *gg, res));
  const std::int64_t nc = gg->ncore, na = gg->nact, nv = gg->nvirt, nmo = gg->nmo;
  const nevpt2::OrbitalEnergyResponse wResp = nevpt2::orbitalEnergyResponse(
      nc, na, nv, gg->get("g_xa_cc"), gg->get("g_xc_ca"), gg->get("g_xa_vv"),
      gg->get("g_xv_va"));
  bool ok = true;

  // --- 1. the full-range f identity ---------------------------------------------
  const nevpt2::ScGeneralCheck self =
      NEVPT2_TRY(nevpt2::scGeneralIntegralCheck(ints, gen, dm1, res));
  const double worstSelf = std::max({self.blocks, self.heff, self.fock, self.eps, self.fockFull});
  ok = ok && worstSelf < kKcSelfTol;
  std::printf("\n  [1] the sidecar against itself (relative): restrictions %.2e (%s), h^eff "
              "%.2e,\n      f's diagonal %.2e, (O2) %.2e, and C5's generator over the WHOLE range,"
              "\n      h^eff + f^gamma[gamma] vs the sidecar's f: %.2e  (< %.0e: %s)\n",
              self.blocks, self.worstBlock.c_str(), self.heff, self.fock, self.eps,
              self.fockFull, kKcSelfTol, worstSelf < kKcSelfTol ? "ok" : "FAILED");

  // --- 2 and 3. the merge, the duality and the columns, full mode ----------------
  const nevpt2::ScMixedHessianSetup full = kcSetup(g, ints, &gen, rdm, opt, true, true, &wResp);
  const nevpt2::Tensor w = nevpt2::antisymmetricKappa(nmo, kKcSeedBase);
  std::printf("\n  mode %s\n", kcModeName(true, true));
  std::printf("  w: a seeded fully mixed antisymmetric kappa (max|w + w^T| = %.1e), seed 0x%llx\n",
              nevpt2::antisymmetryResidual(w), static_cast<unsigned long long>(kKcSeedBase));
  const auto t0 = std::chrono::steady_clock::now();
  const nevpt2::ScMixedHessianVectorResult hv =
      NEVPT2_TRY(nevpt2::scMixedHessianVectorDevice(full, w, sigma, res));
  std::printf("  R1 built in %.2fs: |H_kc^T w| = %.10e  (|walk tangent| = %.10e, |digests at "
              "h2e'[w]| = %.10e)\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
              std::sqrt(kcDot(hv.hkcTw, hv.hkcTw)),
              std::sqrt(kcDot(hv.fromWalkTangent, hv.fromWalkTangent)),
              std::sqrt(kcDot(hv.fromDigests, hv.fromDigests)));
  {
    const std::vector<nevpt2::ScRotationRequest> req{nevpt2::ScRotationRequest{w, true}};
    const nevpt2::ScFullRotationResult y1 = NEVPT2_TRY(nevpt2::scFullRotationDevice(
        ints, gen, dm1, dm2, built.dm3, built.f3ac, built.f3ca, nullptr, req, res));
    const std::vector<nevpt2::ScDigestTangent> dig =
        NEVPT2_TRY(nevpt2::scDigestRotationTangentDevice(opt, ci, g.nelecA, g.nelecB, ints, gen,
                                                         dm1, dm2, built.dm3, built.f3ac,
                                                         built.f3ca, req, res));
    std::printf("\n  [2] the digests' share merged per tuple: the walk's own Edot(w) against "
                "the first-order\n      source term (the frozen-digest tangent + eps + #88's "
                "digest term), per class\n");
    std::printf("  %-7s %22s %22s %11s\n", "class", "walk (merged)", "first order", "|delta|");
    double worst = 0.0;
    for (std::size_t c = 0; c < 8; ++c) {
      const double ref =
          y1.grad.classes[c].rotTangent[0] + y1.grad.classes[c].rotEps[0] + dig[0].classes[c];
      const double d = std::fabs(hv.edotW[c] - ref);
      worst = std::max(worst, d);
      std::printf("  %-7s %22.14e %22.14e %11.2e\n", nevpt2::CLASSES[c].c_str(), hv.edotW[c],
                  ref, d);
    }
    ok = ok && worst < kKcTol;
    std::printf("  worst |delta| = %.2e (< %.0e: %s)\n", worst, kKcTol,
                worst < kKcTol ? "ok" : "FAILED");
  }

  std::printf("\n  [3] the duality <H_kc^T w, u> == Eddot_kc(w, u)\n");
  std::vector<std::pair<std::string, nevpt2::Tensor>> us;
  for (const std::uint64_t seed : kKcUSeeds)
    us.emplace_back(std::format("seed 0x{:x}", seed), kcDirection(ci, seed));
  us.emplace_back("the state itself", ci);
  {
    const std::vector<nevpt2::ScMixedHessianResult> f1 =
        NEVPT2_TRY(nevpt2::scMixedHessianDevice(full, w, {us[0].second}, res));
    std::printf("  Eddot_kc(w, u) along the first u, in its five terms:\n");
    kcPrintTerms(f1[0]);
    const double dual = NEVPT2_TRY(kcDuality(full, w, hv.hkcTw, us, res));
    ok = ok && dual < kKcTol;
    std::printf("  duality, worst |delta| = %.2e (< %.0e: %s)\n", dual, kKcTol,
                dual < kKcTol ? "ok" : "FAILED");
  }
  const std::int64_t ndet = ci.size();
  if (ndet <= kKcColumnMaxDet) {
    std::vector<nevpt2::Tensor> cols;
    for (std::int64_t I = 0; I < ndet; ++I) {
      nevpt2::Tensor e(ci.dims());
      e.flatRef(I) = 1.0;
      cols.push_back(std::move(e));
    }
    const std::vector<nevpt2::ScMixedHessianResult> f =
        NEVPT2_TRY(nevpt2::scMixedHessianDevice(full, w, cols, res));
    double worst = 0.0, hMax = 0.0;
    std::int64_t at = 0;
    for (std::int64_t I = 0; I < ndet; ++I) {
      const double d = std::fabs(hv.hkcTw.flat(I) - f[static_cast<std::size_t>(I)].eddot);
      hMax = std::max(hMax, std::fabs(hv.hkcTw.flat(I)));
      if (d > worst) {
        worst = d;
        at = I;
      }
    }
    ok = ok && worst < kKcTol;
    std::printf("  column by column, (H_kc^T w)_I against Eddot_kc(w, e_I) for all %lld "
                "determinants (max |(H_kc^T w)_I| = %.6e):\n  worst |delta| = %.2e at "
                "determinant %lld (< %.0e: %s)\n",
                static_cast<long long>(ndet), hMax, worst, static_cast<long long>(at), kKcTol,
                worst < kKcTol ? "ok" : "FAILED");
  } else {
    std::printf("  column-by-column check skipped: ndet = %lld is above %lld\n",
                static_cast<long long>(ndet), static_cast<long long>(kKcColumnMaxDet));
  }

  // --- 4. Euler in c, the Euler mode ----------------------------------------------
  {
    const nevpt2::ScMixedHessianSetup euler =
        kcSetup(g, ints, &gen, rdm, opt, true, false, &wResp);
    std::printf("\n  [4] Euler in c -- mode %s\n", kcModeName(true, false));
    const std::vector<nevpt2::ScRotationRequest> req{nevpt2::ScRotationRequest{w, true}};
    const nevpt2::ScFullRotationResult y1 = NEVPT2_TRY(nevpt2::scFullRotationDevice(
        ints, gen, dm1, dm2, built.dm3, built.f3ac, built.f3ca, nullptr, req, res));
    const std::vector<nevpt2::ScDigestTangent> dig =
        NEVPT2_TRY(nevpt2::scDigestRotationTangentDevice(opt, ci, g.nelecA, g.nelecB, ints, gen,
                                                         dm1, dm2, built.dm3, built.f3ac,
                                                         built.f3ca, req, res));
    const std::vector<nevpt2::ScMixedHessianResult> fc =
        NEVPT2_TRY(nevpt2::scMixedHessianDevice(euler, w, {ci}, res));
    std::printf("  %-7s %22s %22s %11s\n", "class", "Eddot_kc(w, c)", "2 <w, Y>", "|delta|");
    for (std::size_t c = 0; c < 8; ++c) {
      const double y =
          y1.grad.classes[c].rotTangent[0] + y1.grad.classes[c].rotEps[0] + dig[0].classes[c];
      const bool homogeneous = c == 0 || c == 4 || c == 5;
      const bool sijrs = c == 2;
      const double got = fc[0].classes[c].total();
      const double d = std::fabs(got - 2.0 * y);
      if (homogeneous) ok = ok && d < kKcTol;
      if (sijrs) ok = ok && got == 0.0;
      std::printf("  %-7s %22.14e %22.14e %11s\n", nevpt2::CLASSES[c].c_str(), got, 2.0 * y,
                  homogeneous ? std::format("{:.2e}", d).c_str()
                              : (sijrs ? (got == 0.0 ? "0 exactly" : "NOT ZERO") : "(affine)"));
    }
    const nevpt2::ScMixedHessianVectorResult hvE =
        NEVPT2_TRY(nevpt2::scMixedHessianVectorDevice(euler, w, sigma, res));
    const double dual = NEVPT2_TRY(kcDuality(euler, w, hvE.hkcTw, {us[0], us.back()}, res));
    ok = ok && dual < kKcTol;
    std::printf("  the duality in this mode, worst |delta| = %.2e (< %.0e: %s)\n", dual, kKcTol,
                dual < kKcTol ? "ok" : "FAILED");
  }

  // --- 5 and 6. the intra-space zeros and the finite differences --------------------
  const nevpt2::Tensor& u = us[0].second;
  for (int which = 0; which < 2; ++which) {
    const bool core = which == 0;
    const char* name = core ? "core" : "virtual";
    const std::int64_t n = core ? nc : nv;
    if (n < 2) {
      std::printf("\n  [5/6] %s-internal: skipped, %lld orbital(s) leave no rotation\n", name,
                  static_cast<long long>(n));
      continue;
    }
    const nevpt2::Tensor block =
        nevpt2::antisymmetricKappa(n, kKcSeedBase + (core ? 0x500u : 0x600u));
    const nevpt2::Tensor wSpace = embedBlock(block, core ? 0 : nc + na, nmo);
    const nevpt2::ScMixedHessianVectorResult hvS =
        NEVPT2_TRY(nevpt2::scMixedHessianVectorDevice(full, wSpace, sigma, res));
    const std::vector<nevpt2::ScMixedHessianResult> f =
        NEVPT2_TRY(nevpt2::scMixedHessianDevice(full, wSpace, {u}, res));
    const nevpt2::ScMixedTerms& t = f[0].total;
    const bool zeros = f[0].h2eDotMax == 0.0 && t.c4 == 0.0 && t.c1Digest == 0.0 &&
                       std::fabs(t.c3) < kKcC3ZeroTol;
    const double dual = std::fabs(kcDot(u, hvS.hkcTw) - f[0].eddot);
    ok = ok && zeros && dual < kKcTol;
    std::printf("\n  [5] a %s-internal w: max |h2e'[w]| = %.1e, max |h1e'[w]| = %.1e\n", name,
                f[0].h2eDotMax, f[0].h1eDotMax);
    kcPrintTerms(f[0]);
    std::printf("  C4 = %.1e and C1's digest share = %.1e (exactly 0), |C3| = %.1e (< %.0e): %s; "
                "the duality |delta| = %.2e (< %.0e: %s)\n",
                t.c4, t.c1Digest, std::fabs(t.c3), kKcC3ZeroTol, zeros ? "ok" : "FAILED", dual,
                kKcTol, dual < kKcTol ? "ok" : "FAILED");
    std::printf("  [6] central differences of <u, y> along exp(s w), %s-internal -- the stored "
                "blocks'\n  %s axes, the orbital energies and W rotated on the host, both legs "
                "in the full mode:\n",
                name, name);
    const nevpt2::Tensor rot0 = nevpt2::expKappa(scaled(block, 0.0));
    const nevpt2::OrbitalEnergyResponse w0 = kcRotatedResponse(*gg, wResp, core, rot0);
    double wGap = 0.0;
    const nevpt2::Tensor& a = core ? w0.core : w0.virt;
    const nevpt2::Tensor& b = core ? wResp.core : wResp.virt;
    for (std::int64_t i = 0; i < a.size(); ++i)
      wGap = std::max(wGap, std::fabs(a.flat(i) - b.flat(i)));
    std::printf("  (the rotated W at s = 0 against the tied rows' W: max |delta| = %.1e)\n", wGap);
    ok = ok && wGap < kKcSelfTol;
    const bool fd = kcSweep(kcDot(u, hvS.hkcTw), [&](double s) {
      return kcSourceSpaceRotated(g, *gg, wResp, core, block, s, u, rdm, sigma, res);
    });
    ok = ok && fd;
  }
  {
    const nevpt2::Tensor kAA = nevpt2::antisymmetricKappa(na, kKcSeedBase + 0x400u);
    const nevpt2::Tensor wAct = embedBlock(kAA, nc, nmo);
    const nevpt2::ScMixedHessianVectorResult hvA =
        NEVPT2_TRY(nevpt2::scMixedHessianVectorDevice(full, wAct, sigma, res));
    std::printf("\n  [6] central differences of <u, y> along exp(s w), active-active -- the "
                "active axes rotated,\n  the digests rebuilt, W rotated on its active pair and "
                "the orbital energies re-formed from it,\n  both legs in the full mode:\n");
    const bool fd = kcSweep(kcDot(u, hvA.hkcTw), [&](double s) {
      return kcSourceActiveRotated(g, gg, &wResp, kAA, s, u, rdm, opt, sigma, res);
    });
    ok = ok && fd;
  }

  // --- 7. without the sidecar ----------------------------------------------------
  ok = NEVPT2_TRY(kcWithoutSidecar(g, ints, rdm, opt, sigma, res)) && ok;

  std::printf("\n%s: the SC mixed Hessian block H_kc's two orientations agree by duality to %.0e "
              "(full mode, column by column where ndet allows; the Euler mode; and with no "
              "sidecar, every other w refused), Euler in c holds on the homogeneous classes, "
              "the digests' share merged per tuple is the first-order term, the intra-space "
              "zeros hold, C5's generator rebuilds f over the whole range, and the finite "
              "differences of y converge\n",
              ok ? "PASS" : "FAIL", kKcTol);
  return ok;
}
