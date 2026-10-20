// nevpt2.app.sc_pseudodensity:hessian_kk_check -- --sc-hessian-kk-int: the
// first of the orbital Hessian block's two terms, the integrals' own
// curvature <Ehat', I''[u, v]> (docs/hessian-theory.md, Sec. 5.1 and Sec.
// 5.4; #154), as K1 (the first-order rule at W = (uv + vu)/2) and K2 (the
// cross-slot term), checked against each other's modes and against a ladder
// of the displaced integrals themselves.
// An interface partition that exports nothing, as :hessian_kc_check is.
//
// WHAT IS BEING DIFFERENTIATED. The energy's first derivative in its inputs
// -- the pseudodensities, R_t, and the digests' transpose hbar -- held FIXED
// at the state, paired with the inputs' second derivative along
// U = exp(s u + t v). That pairing, L(U) = <Ehat'_0, I(U)>, is a function of
// the orbitals alone whose mixed second derivative is exactly this term, so
// a finite difference of L itself checks it on its own -- the other half of
// the orbital block (K3-K5, the energy's curvature in the integrals) is #158's
// and is not in L. The ladder displaces along exp(s u + t v), never
// exp(s u) exp(t v): the latter's mixed derivative is the term plus
// (1/2) <[u, v], Y> (Baker-Campbell-Hausdorff), which check 7 shows.
//
// THE CHECKS, in the order they print:
//
//   1  THE SIDECAR AGAINST ITSELF (scGeneralIntegralCheck), as --kappa-full's.
//   2  K1's MACHINERY AT AN ANTISYMMETRIC ARGUMENT: the first-order rule at a
//      seeded full kappa through the matrix path (traced pairs restored) is
//      --kappa-full's gated first-order pairing, both modes, and the restored
//      terms alone vanish there -- why the first-order path may drop them --
//      while they do not at a symmetric W.
//   3  THE DUALITY, per term: K1's and K2's adjoint pairings (ext + act + eps
//      + <hbar, h2e''>) against their forward tangents (the class terms'
//      product rule at the same inputs + eps + the RDM build at h2e''), along
//      an active x full pair and its swap, an active x active one, and active
//      x core / active x virtual ones -- exact arithmetic, the tree's 1e-7.
//      The swap is held to its original (Eddot is symmetric; the swap runs
//      the composition the other way round, so it is not a tautology here).
//      A fitted case adds full x full and inactive pairs (three-index K2).
//   4  THE TRACED PAIRS' ZERO (Sec. 5.1): along the active x active pair,
//      K1's and K2's traced shares -- each nonzero -- sum to zero.
//   5  THE INTRA-SPACE ZEROS: along a core x core and a virtual x virtual
//      pair no active integral moves at second order, so h2e'' is an
//      exactly-zero tensor and the digests' term vanishes IDENTICALLY. On a
//      conventional case K2 is refused there (neither direction is
//      active-active) and K1 is run alone; on a fitted case both run, and
//      their active-block pairings cancel (the core dressing's tied pair is
//      invariant under a core rotation) to round-off.
//   6  THE LADDERS: L(U) differenced along exp(s u + t v), the four corners
//      (+-h, +-h) combined BEFORE the pairing -- displaced blocks rotated on
//      the host, orbital energies from the rotated generalized Fock with
//      gamma rotated on its active pair -- Richardson-verified against the
//      term, along every block-diagonal pair check 3 runs (a direction that
//      leaves its space cannot be displaced from stored blocks; each
//      promotion is one contraction over the whole general index, so the
//      rows the block-diagonal pairs exercise are the code the mixed rows run).
//   7  THE PARAMETERISATION: the active x active ladder rerun along
//      exp(s u) exp(t v) converges to the term plus (1/2) <[u, v], Y>.
//   8  ON A FITTED CASE, THE TWO ROUTES: K1 through the sidecar's promotion
//      table against its three-index jet, and K2 through the composition
//      against its three-index jet wherever one direction is active-active --
//      constructions that share nothing past B.
//   9  THE REFUSALS: on a conventional case, a pair with neither direction
//      active-active and a three-index route are Unsupported, never answered
//      with K1 alone (Sec. 9.1).
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries. A malformed direction or a
// sidecar that does not fit the golden is the caller's input and propagates
// unchanged to main, which reports it once.
#include "error_handling/error_macros.h"

export module nevpt2.app.sc_pseudodensity:hessian_kk_check;

import std;
import nevpt2.gradient;
import nevpt2.kappa_rotation;
import nevpt2.golden;
// --kappa-full's sidecar upload (either flavour) and its direction helpers.
import :kappa_full_check;

// Every exact identity's gate: two computations of one number, the tree's
// usual 1e-7 (docs/testing.md, "The orbital block's integral curvature",
// records what is measured).
constexpr double kKkTol = 1e-7;
// The sidecar self-check, and K1's matrix path against the first-order
// pairing at an antisymmetric argument: the same functional summed in a
// different order, so reassociation only.
constexpr double kKkSelfTol = 1e-10;
// The restored traced terms at an antisymmetric argument, and the traced
// pairs' K1 + K2 along an active x active pair: two promotions that cancel
// in exact arithmetic, each einsum accumulating in its own order -- absolute,
// decades above what is measured and decades below either share
// (docs/testing.md, "The orbital block's integral curvature").
constexpr double kKkCancelTol = 1e-12;
// The fitted intra-space pair's active-block cancellation: h1e'[W] and h1e^x
// carry the core dressing's tied pair, each nonzero, summing to round-off in
// the einsums' accumulation order.
constexpr double kKkActZeroTol = 1e-12;
// Seeds, fixed so a run is reproducible, distinct from every other check's.
constexpr std::uint64_t kKkSeedBase = 0x5eed'0154u;
// The ladder: h = 2^-k for k in [4, 16]. A SECOND difference's round-off
// floor grows as 1/h^2, so the sweep starts four halvings earlier than the
// first-order checks' and stops two earlier; the gate is theirs -- the best
// error relative to max(|term|, |<u, Y>|, |<v, Y>|) (the first derivatives set
// the magnitude L's displaced values carry) under 1e-6, and a run of three
// fourfold halvings.
constexpr int kKkFdFirstExp = 4;
constexpr int kKkFdLastExp = 16;
constexpr std::size_t kKkFdSteps = kKkFdLastExp - kKkFdFirstExp + 1;
constexpr double kKkFdTol = 1e-6;
constexpr int kKkFdRun = 3;

// One named pair of directions.
struct KkPair {
  std::string name;
  nevpt2::Tensor u, v;
  bool blockDiagonal = false;  // displaceable from stored blocks: the ladder runs
  bool oneActive = false;      // one member active-active: K2 has the sidecar route too
};

// The three diagonal blocks of a block-diagonal orthogonal U.
struct KkBlocks {
  nevpt2::Tensor c, a, v;
};

// One space's diagonal block of `m`.
nevpt2::Tensor kkBlock(const nevpt2::Tensor& m, std::int64_t lo, std::int64_t n) {
  nevpt2::Tensor out({n, n});
  for (std::int64_t p = 0; p < n; ++p)
    for (std::int64_t q = 0; q < n; ++q) out(p, q) = m(lo + p, lo + q);
  return out;
}

nevpt2::Tensor kkMatmul(const nevpt2::Tensor& a, const nevpt2::Tensor& b) {
  const std::int64_t n = a.dim(0);
  nevpt2::Tensor out({n, n});
  for (std::int64_t p = 0; p < n; ++p)
    for (std::int64_t k = 0; k < n; ++k)
      for (std::int64_t q = 0; q < n; ++q) out(p, q) += a(p, k) * b(k, q);
  return out;
}

// U = exp(s u + t v) (or, `product`, exp(s u) exp(t v)) for two
// BLOCK-DIAGONAL directions, block by block: each space's block is the
// exponential of that space's block of the argument, so the off-diagonal
// blocks are exactly zero.
KkBlocks kkRotation(const nevpt2::Tensor& u, const nevpt2::Tensor& v, double s, double t,
                    bool product, std::int64_t nc, std::int64_t na, std::int64_t nv) {
  const auto one = [&](std::int64_t lo, std::int64_t n) {
    const nevpt2::Tensor bu = kkBlock(u, lo, n);
    const nevpt2::Tensor bv = kkBlock(v, lo, n);
    if (product) return kkMatmul(nevpt2::expKappa(scaled(bu, s)), nevpt2::expKappa(scaled(bv, t)));
    nevpt2::Tensor sum({n, n});
    for (std::int64_t i = 0; i < sum.size(); ++i) sum.flatRef(i) = s * bu.flat(i) + t * bv.flat(i);
    return nevpt2::expKappa(sum);
  };
  return KkBlocks{one(0, nc), one(nc, na), one(nc + na, nv)};
}

// The integral set the energy reads, as host tensors, by field.
struct KkHostInts {
  std::vector<std::pair<std::string, nevpt2::Tensor>> fields;  // the golden's names
};

// Each stored array's axes by space -- nevpt2.energy's layouts, as
// :kappa_full_check's uploadSpaceRotated lists them.
struct KkLayout {
  const char* name;
  std::vector<int> core, act, virt;
};
const std::vector<KkLayout>& kkLayouts() {
  static const std::vector<KkLayout> layouts{
      {"h1e", {}, {0, 1}, {}},
      {"h2e", {}, {0, 1, 2, 3}, {}},
      {"h2e_v_Sr", {}, {1, 2, 3}, {0}},
      {"h1e_v_Sr", {}, {1}, {0}},
      {"h2e_v_Si", {2}, {0, 1, 3}, {}},
      {"h1e_v_Si", {1}, {0}, {}},
      {"cvcv", {0, 2}, {}, {1, 3}},
      {"h2e_v_Sijr", {2, 3}, {1}, {0}},
      {"h2e_v_Srsi", {2}, {3}, {0, 1}},
      {"h2e_v_Srs", {}, {2, 3}, {0, 1}},
      {"h2e_v_Sij", {2, 3}, {0, 1}, {}},
      {"h2e_v1_Sir", {2}, {1, 3}, {0}},
      {"h2e_v2_Sir", {3}, {1, 2}, {0}},
      {"h1e_v_Sir", {1}, {}, {0}},
  };
  return layouts;
}

// What the generalized Fock's core and virtual blocks are built from, read
// once: h^eff, gamma, and per inactive space the (pp'|ab) - (1/2)(pa|bp')
// block over that space's full (p, p') range -- the sidecar's untied rows,
// downloaded from the device inventory so a fitted case (whose sidecar does
// not carry them) reads the ones built from B.
struct KkFockParts {
  nevpt2::Tensor heff, gamma;
  nevpt2::Tensor core, virt;  // (n, n, nact, nact)
};

KkFockParts kkFockParts(const nevpt2::GeneralIntegralsDevice& gen, const nevpt2::Tensor& gamma,
                        const nevpt2::DeviceResources& res) {
  const std::int64_t nc = gen.ncore, na = gen.nact, nv = gen.nvirt;
  KkFockParts p;
  p.heff = nevpt2::downloadTensor(gen.heff, res.stream());
  p.gamma = gamma;
  const auto block = [&](const nevpt2::DeviceTensor& cou, const nevpt2::DeviceTensor& exc,
                         std::int64_t lo, std::int64_t n) {
    const nevpt2::Tensor hc = nevpt2::downloadTensor(cou, res.stream());  // (x p | a b)
    const nevpt2::Tensor he = nevpt2::downloadTensor(exc, res.stream());  // (x a | p b)
    nevpt2::Tensor a({n, n, na, na});
    for (std::int64_t q = 0; q < n; ++q)
      for (std::int64_t r = 0; r < n; ++r)
        for (std::int64_t s = 0; s < na; ++s)
          for (std::int64_t t = 0; t < na; ++t)
            a(q, r, s, t) = hc(lo + q, r, s, t) - 0.5 * he(lo + q, s, r, t);
    return a;
  };
  p.core = nc > 0 ? block(gen.g_xc_aa, gen.g_xa_ca, 0, nc) : nevpt2::Tensor({0, 0, na, na});
  p.virt = nv > 0 ? block(gen.g_xv_aa, gen.g_xa_va, nc + na, nv) : nevpt2::Tensor({0, 0, na, na});
  return p;
}

// One inactive space's orbital energies at U: diag(U_S^T [h^eff_SS +
// F^gamma[U_A gamma U_A^T]_SS] U_S), F^gamma[g]_xy = sum_ab g_ab A[x, y, a, b].
nevpt2::Tensor kkOrbitalEnergies(const KkFockParts& p, const nevpt2::Tensor& block,
                                 std::int64_t lo, const nevpt2::Tensor& us,
                                 const nevpt2::Tensor& ua) {
  const std::int64_t n = us.dim(0), na = ua.dim(0);
  // gamma_U = U_A gamma U_A^T.
  nevpt2::Tensor g({na, na});
  for (std::int64_t a = 0; a < na; ++a)
    for (std::int64_t b = 0; b < na; ++b) {
      double v = 0.0;
      for (std::int64_t c = 0; c < na; ++c)
        for (std::int64_t d = 0; d < na; ++d) v += ua(a, c) * p.gamma(c, d) * ua(b, d);
      g(a, b) = v;
    }
  nevpt2::Tensor f({n, n});
  for (std::int64_t x = 0; x < n; ++x)
    for (std::int64_t y = 0; y < n; ++y) {
      double v = p.heff(lo + x, lo + y);
      for (std::int64_t a = 0; a < na; ++a)
        for (std::int64_t b = 0; b < na; ++b) v += g(a, b) * block(x, y, a, b);
      f(x, y) = v;
    }
  const nevpt2::Tensor rotated = nevpt2::rotateAxes(f, {0, 1}, us);
  nevpt2::Tensor e({n});
  for (std::int64_t q = 0; q < n; ++q) e(q) = rotated(q, q);
  return e;
}

// The whole input set at the block-diagonal U, on the host.
KkHostInts kkRotatedInts(const nevpt2::GoldenFile& g, const KkFockParts& fock, const KkBlocks& u,
                         std::int64_t nc, std::int64_t na) {
  KkHostInts out;
  for (const KkLayout& l : kkLayouts()) {
    nevpt2::Tensor x = g.get(l.name);
    if (!l.core.empty()) x = nevpt2::rotateAxes(x, l.core, u.c);
    if (!l.act.empty()) x = nevpt2::rotateAxes(x, l.act, u.a);
    if (!l.virt.empty()) x = nevpt2::rotateAxes(x, l.virt, u.v);
    out.fields.emplace_back(l.name, std::move(x));
  }
  out.fields.emplace_back("e_core", kkOrbitalEnergies(fock, fock.core, 0, u.c, u.a));
  out.fields.emplace_back("e_virt", kkOrbitalEnergies(fock, fock.virt, nc + na, u.v, u.a));
  return out;
}

// sum_k coef_k x_k, field by field, uploaded as a NevptIntegralsDevice.
nevpt2::NevptIntegralsDevice kkUploadCombination(const std::vector<KkHostInts>& sets,
                                                 const std::vector<double>& coef,
                                                 const nevpt2::DeviceResources& res) {
  nevpt2::NevptIntegralsDevice d;
  const auto field = [&](const std::string& name) -> nevpt2::DeviceTensor& {
    if (name == "h1e") return d.h1e;
    if (name == "h2e") return d.h2e;
    if (name == "e_core") return d.e_core;
    if (name == "e_virt") return d.e_virt;
    if (name == "h2e_v_Sr") return d.h2e_v_Sr;
    if (name == "h1e_v_Sr") return d.h1e_v_Sr;
    if (name == "h2e_v_Si") return d.h2e_v_Si;
    if (name == "h1e_v_Si") return d.h1e_v_Si;
    if (name == "cvcv") return d.cvcv;
    if (name == "h2e_v_Sijr") return d.h2e_v_Sijr;
    if (name == "h2e_v_Srsi") return d.h2e_v_Srsi;
    if (name == "h2e_v_Srs") return d.h2e_v_Srs;
    if (name == "h2e_v_Sij") return d.h2e_v_Sij;
    if (name == "h2e_v1_Sir") return d.h2e_v1_Sir;
    if (name == "h2e_v2_Sir") return d.h2e_v2_Sir;
    return d.h1e_v_Sir;
  };
  for (std::size_t f = 0; f < sets[0].fields.size(); ++f) {
    nevpt2::Tensor sum(sets[0].fields[f].second.dims());
    for (std::size_t k = 0; k < sets.size(); ++k)
      for (std::int64_t i = 0; i < sum.size(); ++i)
        sum.flatRef(i) += coef[k] * sets[k].fields[f].second.flat(i);
    field(sets[0].fields[f].first) = nevpt2::uploadTensor(sum, res);
  }
  return d;
}

// The ladder of L along `pair`: per step h, the four corners' integral sets
// combined as I(h,h) - I(h,-h) - I(-h,h) + I(-h,-h) and paired once, over
// 4 h^2. `want` is what it must converge to; `scale` its relative gate's
// denominator. Returns whether both criteria hold.
nevpt2::Result<bool> kkLadder(const nevpt2::GoldenFile& g, const KkFockParts& fock,
                              const KkPair& pair, bool product, double want, double scale,
                              const nevpt2::NevptIntegralsDevice& ints,
                              const nevpt2::RdmTensors& rdm, const nevpt2::DeviceTensor& hbar,
                              std::int64_t nc, std::int64_t na, std::int64_t nv,
                              const nevpt2::DeviceResources& res) {
  std::vector<nevpt2::NevptIntegralsDevice> combos;
  std::array<double, kKkFdSteps> step{};
  for (std::size_t i = 0; i < kKkFdSteps; ++i) {
    const double h = std::ldexp(1.0, -(kKkFdFirstExp + static_cast<int>(i)));
    step[i] = h;
    std::vector<KkHostInts> corners;
    for (const auto& [ss, tt] : std::array<std::pair<double, double>, 4>{
             {{h, h}, {h, -h}, {-h, h}, {-h, -h}}})
      corners.push_back(
          kkRotatedInts(g, fock, kkRotation(pair.u, pair.v, ss, tt, product, nc, na, nv), nc, na));
    combos.push_back(kkUploadCombination(corners, {1.0, -1.0, -1.0, 1.0}, res));
  }
  std::vector<const nevpt2::NevptIntegralsDevice*> ptr;
  for (const nevpt2::NevptIntegralsDevice& c : combos) ptr.push_back(&c);
  const nevpt2::ScFullRotationResult paired =
      NEVPT2_TRY(nevpt2::scDottedPairingDevice(ints, rdm, &hbar, ptr, /*tangent=*/false, res));

  std::printf("  %5s %14s %22s %12s %12s %8s\n", "k", "h = 2^-k", "second difference / 4h^2",
              "|err|", "rel err", "ratio");
  std::array<double, kKkFdSteps> err{};
  for (std::size_t i = 0; i < kKkFdSteps; ++i) {
    const double d = paired.dirs[i].adjoint / (4.0 * step[i] * step[i]);
    err[i] = std::fabs(d - want);
    const double rel = err[i] / std::max(1e-30, scale);
    const int k = kKkFdFirstExp + static_cast<int>(i);
    if (i == 0)
      std::printf("  %5d %14.3e %22.14e %12.3e %12.3e %8s\n", k, step[i], d, err[i], rel, "-");
    else
      std::printf("  %5d %14.3e %22.14e %12.3e %12.3e %8.2f\n", k, step[i], d, err[i], rel,
                  err[i] > 0.0 ? err[i - 1] / err[i] : 0.0);
  }
  std::size_t bestAt = 0;
  for (std::size_t i = 1; i < kKkFdSteps; ++i)
    if (err[i] < err[bestAt]) bestAt = i;
  const double bestRel = err[bestAt] / std::max(1e-30, scale);
  const nevpt2::FourfoldRun run = nevpt2::longestFourfoldRun(err);
  const bool ok = bestRel < kKkFdTol && run.length >= kKkFdRun;
  std::printf("  want %.14e; plateau |err| = %.3e (relative %.3e against %.3e) at h = %.3e\n",
              want, err[bestAt], bestRel, scale, step[bestAt]);
  std::printf("  Richardson: %d consecutive fourfold halvings (ratio in [%.1f, %.1f]) ending at "
              "h = %.3e -- %s, %d needed\n",
              run.length, nevpt2::kFourfoldLow, nevpt2::kFourfoldHigh, step[run.end],
              ok ? "ok" : "NOT ENOUGH", kKkFdRun);
  return ok;
}

// The first-order pairing <kappa, Y> along an antisymmetric kappa, the
// digests' transpose included: the scale of the ladders' gate, and check 7's
// commutator term.
nevpt2::Result<double> kkFirstOrder(const nevpt2::NevptIntegralsDevice& ints,
                                    const nevpt2::GeneralIntegralsDevice& gen,
                                    const nevpt2::RdmTensors& rdm,
                                    const nevpt2::DeviceTensor& hbar, const nevpt2::Tensor& k,
                                    const nevpt2::DeviceResources& res) {
  const nevpt2::ScFullRotationResult r = NEVPT2_TRY(nevpt2::scFullRotationDevice(
      ints, gen, rdm.dm1, rdm.dm2, rdm.dm3, rdm.f3ac, rdm.f3ca, &hbar,
      {nevpt2::ScRotationRequest{k, /*tangent=*/false}}, res));
  return r.dirs[0].adjoint;
}

// One pair's two terms, printed as a table row pair.
void kkPrintPair(const std::string& name, const nevpt2::ScOrbitalCurvatureResult& r) {
  const auto row = [](const char* term, const nevpt2::ScCurvatureTerm& t,
                      nevpt2::ScCurvatureRoute route) {
    std::printf("    %-3s %-11s %22.14e %22.14e %10.2e %12.4e %12.4e %12.4e %12.4e\n", term,
                nevpt2::scCurvatureRouteName(route), t.adjoint, t.tangent,
                std::fabs(t.adjoint - t.tangent), t.ext, t.act, t.eps, t.digest);
  };
  std::printf("  %s: <Ehat', I''> = %.14e (tangent %.14e)\n", name.c_str(), r.value(),
              r.valueTangent());
  row("K1", r.k1, r.k1Route);
  row("K2", r.k2, r.k2Route);
}

nevpt2::Result<bool> checkScHessianKkIntegral(const nevpt2::GoldenFile& g,
                                              const nevpt2::GradGoldFile& gg,
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
  std::printf("\n--sc-hessian-kk-int: the orbital Hessian's integral curvature <Ehat', I''[u, v]> "
              "= K1 + K2 (docs/hessian-theory.md, Sec. 5.1)%s\n",
              fitted ? " -- density-fitted" : "");
  std::printf("  K1 = <Ehat', I'[W]>, W = (uv + vu)/2; K2 = <Ehat', I^x[u, v]>; both modes, "
              "the digests' term in each\n");
  bool ok = true;

  // --- 1. the sidecar against itself ---------------------------------------------
  const nevpt2::ScGeneralCheck self =
      NEVPT2_TRY(nevpt2::scGeneralIntegralCheck(ints, gen, dm1, res));
  const double worstSelf = std::max({self.blocks, self.heff, self.fock, self.eps, self.fockFull});
  ok = ok && worstSelf < kKkSelfTol;
  std::printf("\n  [1] the sidecar against itself (relative): restrictions %.2e (%s), h^eff %.2e, "
              "f's diagonal %.2e,\n      (O2) %.2e, f over the whole range %.2e  (< %.0e: %s)\n",
              self.blocks, self.worstBlock.c_str(), self.heff, self.fock, self.eps, self.fockFull,
              kKkSelfTol, worstSelf < kKkSelfTol ? "ok" : "FAILED");

  // The digests' transpose: what the adjoint pairings' digest term reads.
  const nevpt2::DeviceTensor hbar = nevpt2::scDigestAdjointDevice(
      opt, ci, g.nelecA, g.nelecB, ints, dm1, dm2, built.dm3, built.f3ac, built.f3ca, res);
  const nevpt2::ScOrbitalCurvatureSetup setup{ints,     gen,      rdm, &hbar, &ci,
                                              g.nelecA, g.nelecB, opt};

  // The directions.
  const nevpt2::Tensor uAct = embedBlock(nevpt2::antisymmetricKappa(na, kKkSeedBase + 1), nc, nmo);
  const nevpt2::Tensor vAct = embedBlock(nevpt2::antisymmetricKappa(na, kKkSeedBase + 2), nc, nmo);
  const nevpt2::Tensor full1 = nevpt2::antisymmetricKappa(nmo, kKkSeedBase + 3);
  const nevpt2::Tensor full2 = nevpt2::antisymmetricKappa(nmo, kKkSeedBase + 4);
  std::optional<nevpt2::Tensor> core1, core2, virt1, virt2;
  if (nc >= 2) {
    core1 = embedBlock(nevpt2::antisymmetricKappa(nc, kKkSeedBase + 5), 0, nmo);
    core2 = embedBlock(nevpt2::antisymmetricKappa(nc, kKkSeedBase + 6), 0, nmo);
  }
  if (nv >= 2) {
    virt1 = embedBlock(nevpt2::antisymmetricKappa(nv, kKkSeedBase + 7), nc + na, nmo);
    virt2 = embedBlock(nevpt2::antisymmetricKappa(nv, kKkSeedBase + 8), nc + na, nmo);
  }

  // --- 2. K1's machinery at an antisymmetric argument ------------------------------
  {
    const std::vector<nevpt2::ScRotationRequest> req{nevpt2::ScRotationRequest{full1, true}};
    const nevpt2::ScFullRotationResult first = NEVPT2_TRY(nevpt2::scFullRotationDevice(
        ints, gen, dm1, dm2, built.dm3, built.f3ac, built.f3ca, &hbar, req, res));
    const nevpt2::ScFullRotationResult mat =
        NEVPT2_TRY(nevpt2::scMatrixRotationDevice(ints, gen, rdm, &hbar, req, res));
    const double dAdj = std::fabs(first.dirs[0].adjoint - mat.dirs[0].adjoint);
    const double dTan = std::fabs(first.dirs[0].tangentIntegral - mat.dirs[0].tangentIntegral);
    const double restoredFull =
        NEVPT2_TRY(nevpt2::scRestoredTracePairingDevice(ints, gen, rdm, full1, res));
    const double restoredAct =
        NEVPT2_TRY(nevpt2::scRestoredTracePairingDevice(ints, gen, rdm, uAct, res));
    // A symmetric argument: the active x active pair's W.
    nevpt2::Tensor wAct({nmo, nmo});
    {
      const nevpt2::Tensor uv = kkMatmul(uAct, vAct), vu = kkMatmul(vAct, uAct);
      for (std::int64_t i = 0; i < wAct.size(); ++i)
        wAct.flatRef(i) = 0.5 * (uv.flat(i) + vu.flat(i));
    }
    const double restoredW =
        NEVPT2_TRY(nevpt2::scRestoredTracePairingDevice(ints, gen, rdm, wAct, res));
    const bool good = dAdj < kKkSelfTol && dTan < kKkSelfTol &&
                      std::fabs(restoredFull) < kKkCancelTol &&
                      std::fabs(restoredAct) < kKkCancelTol;
    ok = ok && good;
    std::printf("\n  [2] K1's machinery at an antisymmetric argument (a seeded full kappa, seed "
                "0x%llx):\n",
                static_cast<unsigned long long>(kKkSeedBase + 3));
    std::printf("      <kappa, Y> first-order %.14e, matrix path %.14e: |delta| %.2e\n",
                first.dirs[0].adjoint, mat.dirs[0].adjoint, dAdj);
    std::printf("      the tangents' integral halves: |delta| %.2e  (both < %.0e)\n", dTan,
                kKkSelfTol);
    std::printf("      the restored traced terms alone: %.2e at that kappa, %.2e at an active "
                "kappa (< %.0e), %.6e at a symmetric W: %s\n",
                restoredFull, restoredAct, kKkCancelTol, restoredW, good ? "ok" : "FAILED");
  }

  // --- 3, 4, 8. the pairs, the duality, the traced zero, the routes ---------------
  std::vector<KkPair> pairs{
      KkPair{"active x full", uAct, full1, false, true},
      KkPair{"full x active (the swap)", full1, uAct, false, true},
      KkPair{"active x active", uAct, vAct, true, true},
  };
  if (core1) pairs.push_back(KkPair{"active x core", uAct, *core1, true, true});
  if (virt1) pairs.push_back(KkPair{"active x virtual", uAct, *virt1, true, true});
  if (fitted) {
    pairs.push_back(KkPair{"full x full", full1, full2, false, false});
    if (core1 && virt1) pairs.push_back(KkPair{"core x virtual", *core1, *virt1, true, false});
    if (core1) pairs.push_back(KkPair{"core x core", *core1, *core2, true, false});
    if (virt1) pairs.push_back(KkPair{"virtual x virtual", *virt1, *virt2, true, false});
  }
  std::vector<nevpt2::ScOrbitalCurvatureRequest> reqs;
  for (const KkPair& p : pairs) reqs.push_back(nevpt2::ScOrbitalCurvatureRequest{p.u, p.v, true});
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<nevpt2::ScOrbitalCurvatureResult> res3 =
      NEVPT2_TRY(nevpt2::scOrbitalCurvatureDevice(setup, reqs, res));
  std::printf("\n  [3] the duality, per term (%zu pairs in %.2fs): adjoint against tangent\n",
              pairs.size(),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  std::printf("    %-3s %-11s %22s %22s %10s %12s %12s %12s %12s\n", "", "route", "adjoint",
              "tangent", "|delta|", "ext", "act", "eps", "digest");
  double worstDual = 0.0;
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    kkPrintPair(pairs[k].name, res3[k]);
    worstDual = std::max({worstDual, std::fabs(res3[k].k1.adjoint - res3[k].k1.tangent),
                          std::fabs(res3[k].k2.adjoint - res3[k].k2.tangent)});
  }
  ok = ok && worstDual < kKkTol;
  const double swapGap = std::fabs(res3[0].value() - res3[1].value());
  ok = ok && swapGap < kKkTol;
  std::printf("  duality, worst |delta| = %.2e; the swap against its original |delta| = %.2e "
              "(both < %.0e: %s)\n",
              worstDual, swapGap, kKkTol, worstDual < kKkTol && swapGap < kKkTol ? "ok" : "FAILED");

  {
    const nevpt2::ScOrbitalCurvatureResult& aa = res3[2];
    const double sum = aa.k1Traced + aa.k2Traced;
    const bool good = aa.hasTraced && std::fabs(sum) < kKkCancelTol &&
                      std::fabs(aa.k1Traced) > 1e3 * kKkCancelTol;
    ok = ok && good;
    std::printf("\n  [4] the traced pairs (h1e_v_Sr's trace, v2tr) along the active x active "
                "pair: K1's share %.6e,\n      K2's %.6e, sum %.2e (< %.0e, each well above "
                "it): %s\n",
                aa.k1Traced, aa.k2Traced, sum, kKkCancelTol, good ? "ok" : "FAILED");
  }

  // --- 5. the intra-space zeros ---------------------------------------------------
  {
    std::printf("\n  [5] the intra-space pairs: h2e'' is exactly zero, so the digests' term is\n");
    struct Intra {
      const char* name;
      const std::optional<nevpt2::Tensor>* a;
      const std::optional<nevpt2::Tensor>* b;
    };
    for (const Intra& in : {Intra{"core x core", &core1, &core2},
                            Intra{"virtual x virtual", &virt1, &virt2}}) {
      if (!*in.a) {
        std::printf("  %s: skipped, the space has fewer than two orbitals\n", in.name);
        continue;
      }
      const nevpt2::Tensor& a = **in.a;
      const nevpt2::Tensor& b = **in.b;
      if (!fitted) {
        // K2 refused; K1 alone through its own machinery.
        const nevpt2::Result<std::vector<nevpt2::ScOrbitalCurvatureResult>> refused =
            nevpt2::scOrbitalCurvatureDevice(setup, {nevpt2::ScOrbitalCurvatureRequest{a, b}},
                                             res);
        const bool isRefused = !refused && refused.error().kind == nevpt2::ErrorKind::Unsupported;
        nevpt2::Tensor w({nmo, nmo});
        const nevpt2::Tensor ab = kkMatmul(a, b), ba = kkMatmul(b, a);
        for (std::int64_t i = 0; i < w.size(); ++i) w.flatRef(i) = 0.5 * (ab.flat(i) + ba.flat(i));
        const nevpt2::ScFullRotationResult k1 = NEVPT2_TRY(nevpt2::scMatrixRotationDevice(
            ints, gen, rdm, &hbar, {nevpt2::ScRotationRequest{w, true}}, res));
        const bool good = isRefused && k1.dirs[0].hasDigest && k1.dirs[0].digest == 0.0 &&
                          std::fabs(k1.dirs[0].adjoint - k1.dirs[0].tangent) < kKkTol;
        ok = ok && good;
        std::printf("  %s: K2 %s; K1 alone = %.14e, its digest term %.1e (exactly 0), its "
                    "duality |delta| %.2e: %s\n",
                    in.name, isRefused ? "refused, Unsupported" : "NOT REFUSED",
                    k1.dirs[0].adjoint, k1.dirs[0].digest,
                    std::fabs(k1.dirs[0].adjoint - k1.dirs[0].tangent),
                    good ? "ok" : "FAILED");
        continue;
      }
      // Fitted: the pair is among check 3's; find it.
      for (std::size_t k = 0; k < pairs.size(); ++k) {
        if (pairs[k].name != in.name) continue;
        const nevpt2::ScOrbitalCurvatureResult& r = res3[k];
        const bool zeros = r.k1.h2eMax == 0.0 && r.k2.h2eMax == 0.0 && r.k1.digest == 0.0 &&
                           r.k2.digest == 0.0 && r.k1.digestForward == 0.0 &&
                           r.k2.digestForward == 0.0;
        const double actSum = r.k1.act + r.k2.act;
        const bool good = zeros && std::fabs(actSum) < kKkActZeroTol;
        ok = ok && good;
        std::printf("  %s: max|h2e''| %.1e / %.1e, the digests' term %.1e / %.1e reverse and "
                    "%.1e / %.1e forward (exactly 0);\n      the active blocks' pairing K1 "
                    "%.6e + K2 %.6e = %.2e (< %.0e): %s\n",
                    in.name, r.k1.h2eMax, r.k2.h2eMax, r.k1.digest, r.k2.digest,
                    r.k1.digestForward, r.k2.digestForward, r.k1.act, r.k2.act, actSum,
                    kKkActZeroTol, good ? "ok" : "FAILED");
      }
    }
  }

  // --- 6 and 7. the ladders -------------------------------------------------------
  const KkFockParts fock = kkFockParts(gen, g.get("dm1"), res);
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    if (!pairs[k].blockDiagonal) continue;
    const double yu = NEVPT2_TRY(kkFirstOrder(ints, gen, rdm, hbar, pairs[k].u, res));
    const double yv = NEVPT2_TRY(kkFirstOrder(ints, gen, rdm, hbar, pairs[k].v, res));
    const double want = res3[k].value();
    const double scale = std::max({std::fabs(want), std::fabs(yu), std::fabs(yv)});
    std::printf("\n  [6] the ladder along exp(s u + t v), %s (<u, Y> = %.6e, <v, Y> = %.6e):\n",
                pairs[k].name.c_str(), yu, yv);
    const bool fd = NEVPT2_TRY(
        kkLadder(g, fock, pairs[k], false, want, scale, ints, rdm, hbar, nc, na, nv, res));
    ok = ok && fd;
  }
  {
    // The pair whose commutator the gradient sees most: [u, v] is nonzero only
    // for two directions in one space, and <[u, v], Y> can still vanish by
    // symmetry (an active space of one orbital per irrep has Y_aa = 0), in
    // which case the two parameterisations agree here and the run says so.
    std::size_t best = pairs.size();
    double bestComm = 0.0;
    nevpt2::Tensor bestKappa;
    for (std::size_t k = 0; k < pairs.size(); ++k) {
      const bool sameSpace = pairs[k].name == "active x active" ||
                             pairs[k].name == "core x core" ||
                             pairs[k].name == "virtual x virtual";
      if (!sameSpace) continue;
      nevpt2::Tensor comm({nmo, nmo});
      const nevpt2::Tensor uv = kkMatmul(pairs[k].u, pairs[k].v);
      const nevpt2::Tensor vu = kkMatmul(pairs[k].v, pairs[k].u);
      for (std::int64_t i = 0; i < comm.size(); ++i) comm.flatRef(i) = uv.flat(i) - vu.flat(i);
      const double yComm = NEVPT2_TRY(kkFirstOrder(ints, gen, rdm, hbar, comm, res));
      if (best == pairs.size() || std::fabs(yComm) > std::fabs(bestComm)) {
        best = k;
        bestComm = yComm;
      }
    }
    const KkPair& pp = pairs[best];
    const double want = res3[best].value() + 0.5 * bestComm;
    const double yu = NEVPT2_TRY(kkFirstOrder(ints, gen, rdm, hbar, pp.u, res));
    const double yv = NEVPT2_TRY(kkFirstOrder(ints, gen, rdm, hbar, pp.v, res));
    const double scale = std::max({std::fabs(want), std::fabs(yu), std::fabs(yv)});
    std::printf("\n  [7] the ladder along exp(s u) exp(t v), %s: the term %.10e + (1/2) <[u, v], "
                "Y> = %.10e\n      (the product parameterisation's own mixed derivative, %.2e "
                "away from the term%s):\n",
                pp.name.c_str(), res3[best].value(), want, 0.5 * std::fabs(bestComm),
                std::fabs(bestComm) < 1e-12 ? " -- <[u, v], Y> vanishes by symmetry on this case"
                                            : "");
    const bool fd =
        NEVPT2_TRY(kkLadder(g, fock, pp, true, want, scale, ints, rdm, hbar, nc, na, nv, res));
    ok = ok && fd;
  }

  // --- 8. the two routes on a fitted case -----------------------------------------
  if (fitted) {
    std::printf("\n  [8] the two routes: the sidecar's (promotion table; composition) against "
                "the three-index jets of B\n");
    std::vector<nevpt2::ScOrbitalCurvatureRequest> side, three;
    std::vector<std::size_t> which;
    for (std::size_t k = 0; k < pairs.size(); ++k) {
      const bool oneActive = pairs[k].oneActive;
      nevpt2::ScOrbitalCurvatureRequest a{pairs[k].u, pairs[k].v, false};
      a.k1 = nevpt2::ScCurvatureRoute::Sidecar;
      a.k2 = oneActive ? nevpt2::ScCurvatureRoute::Sidecar : nevpt2::ScCurvatureRoute::ThreeIndex;
      nevpt2::ScOrbitalCurvatureRequest b{pairs[k].u, pairs[k].v, false};
      b.k1 = nevpt2::ScCurvatureRoute::ThreeIndex;
      b.k2 = nevpt2::ScCurvatureRoute::ThreeIndex;
      side.push_back(a);
      three.push_back(b);
      which.push_back(k);
    }
    const std::vector<nevpt2::ScOrbitalCurvatureResult> rs =
        NEVPT2_TRY(nevpt2::scOrbitalCurvatureDevice(setup, side, res));
    const std::vector<nevpt2::ScOrbitalCurvatureResult> rt =
        NEVPT2_TRY(nevpt2::scOrbitalCurvatureDevice(setup, three, res));
    std::printf("    %-26s %22s %22s %10s %22s %22s %10s\n", "pair", "K1 sidecar",
                "K1 three-index", "|delta|", "K2 sidecar", "K2 three-index", "|delta|");
    double worst = 0.0;
    for (std::size_t i = 0; i < which.size(); ++i) {
      const double d1 = std::fabs(rs[i].k1.adjoint - rt[i].k1.adjoint);
      const bool k2Both = rs[i].k2Route == nevpt2::ScCurvatureRoute::Sidecar;
      const double d2 = k2Both ? std::fabs(rs[i].k2.adjoint - rt[i].k2.adjoint) : 0.0;
      worst = std::max({worst, d1, d2});
      std::printf("    %-26s %22.14e %22.14e %10.2e %22s %22.14e %10s\n",
                  pairs[which[i]].name.c_str(), rs[i].k1.adjoint, rt[i].k1.adjoint, d1,
                  k2Both ? std::format("{:.14e}", rs[i].k2.adjoint).c_str() : "(no active member)",
                  rt[i].k2.adjoint, k2Both ? std::format("{:.2e}", d2).c_str() : "-");
    }
    ok = ok && worst < kKkTol;
    std::printf("  worst |delta| = %.2e (< %.0e: %s)\n", worst, kKkTol,
                worst < kKkTol ? "ok" : "FAILED");
  }

  // --- 9. the refusals ------------------------------------------------------------
  {
    std::printf("\n  [9] the refusals:\n");
    // A pair with neither direction active-active, by the composition.
    nevpt2::ScOrbitalCurvatureRequest neither{full1, full2, false};
    if (fitted) neither.k2 = nevpt2::ScCurvatureRoute::Sidecar;
    const nevpt2::Result<std::vector<nevpt2::ScOrbitalCurvatureResult>> r1 =
        nevpt2::scOrbitalCurvatureDevice(setup, {neither}, res);
    const bool refused1 = !r1 && r1.error().kind == nevpt2::ErrorKind::Unsupported;
    ok = ok && refused1;
    std::printf("  full x full, K2 from the sidecar: %s\n",
                refused1 ? std::format("refused, Unsupported (\"{:.90}...\")", r1.error().message)
                               .c_str()
                         : "NOT REFUSED");
    if (!fitted) {
      nevpt2::ScOrbitalCurvatureRequest three{uAct, vAct, false};
      three.k2 = nevpt2::ScCurvatureRoute::ThreeIndex;
      const nevpt2::Result<std::vector<nevpt2::ScOrbitalCurvatureResult>> r2 =
          nevpt2::scOrbitalCurvatureDevice(setup, {three}, res);
      const bool refused2 = !r2 && r2.error().kind == nevpt2::ErrorKind::Unsupported;
      ok = ok && refused2;
      std::printf("  the three-index route on a conventional case: %s\n",
                  refused2 ? "refused, Unsupported" : "NOT REFUSED");
    }
  }

  std::printf("\n%s: the SC orbital Hessian's integral curvature <Ehat', I''> = K1 + K2 agrees "
              "with itself by duality to %.0e in each term%s, the traced pairs' zero and the "
              "intra-space zeros hold, every refusal holds, and the ladders of the pairing along "
              "exp(s u + t v) converge to it\n",
              ok ? "PASS" : "FAIL", kKkTol,
              fitted ? " (density-fitted: K2 by the three-index jets, both routes agree)" : "");
  return ok;
}
