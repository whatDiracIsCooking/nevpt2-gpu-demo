// nevpt2.app.sc_pseudodensity:parameter_hessian_check -- --sc-parameter-hessian
// PATH: the SC level-1 parameter Hessian as a BILINEAR FORM over the response
// directions of a `.respgold` (#222; docs/hessian-theory.md, Sec. 8: level 1
// is consumed, not solved). An interface partition that exports nothing, as
// :source_tangent_check is.
//
// WHAT IS COMPUTED. With Theta_A = (kappa_A, c_A) the sidecar's first-order
// response to nuclear coordinate A, the matrix
//
//   E_thth[A, B] = Eddot_kk(kappa_A, kappa_B) + Eddot_kc(kappa_A, c_B)
//                + Eddot_kc(kappa_B, c_A)     + Eddot_cc(c_A, c_B)
//
// by nevpt2.gradient's scParameterHessianDevice, with the orbital energies
// responding throughout (its header says how each block is formed, and why
// the CI and mixed blocks are 3N vector products and the orbital block per
// pair). With --parameter-hessian-out PATH the matrix and its three blocks are
// written as a `.parhess` and read back.
//
// THE CHECKS, in the order they print:
//
//   0  THE SIDECARS: the responses in the golden's frame (ci and mo_coeff bit
//      for bit, nevpt2.golden's checkAgainst), each kappa antisymmetric and
//      zero on its three diagonal blocks, each CI response orthogonal to ci;
//      the gradient sidecar against itself (--kappa-full's self-check, f over
//      the whole range included); and the ladders' host builder at U = 1 --
//      every stored block, h^eff, f, the eighteen rows and the orbital
//      energies rebuilt from the full tensor -- against the golden's and the
//      sidecar's own arrays.
//   1  THE MATRIX, and the free check that every H_cc walk ran the gated y.
//   2  EVERY ENTRY AGAINST THE PER-BLOCK FLAGS' SCALARS, on the same pair:
//      the orbital block is --sc-hessian-kk's scalar itself (A <= B, the
//      symmetric form's mirror for A > B); the mixed block
//      against --sc-hessian-kc's forward Eddot_kc(kappa_A, c_B)
//      (scMixedHessianDevice, FULL mode) on every ordered pair; the CI block
//      against --sc-hessian's Eddot(c_A, c_B) (energiesHessianDevice, FULL
//      mode) on every pair A <= B, held to both [A][B] and [B][A]. Exact
//      arithmetic, the tree's 1e-7.
//   3  SYMMETRY, 1e-7: the CI block's <c_B, H_cc c_A> against <c_A, H_cc c_B>
//      on every entry (two walks); the mixed block's two orientations sum to
//      a symmetric matrix by construction; the orbital block, which the
//      matrix holds once per pair A <= B, walked again with the pair's
//      directions swapped on seeded off-diagonal pairs (every one where there
//      are at most kPhSwappedPairs).
//   4  THE LADDERS: the gated Y and y (--dump-source-terms' pair, the
//      orbital energies responding to kappa in Y and to c through W in y)
//      recomputed in full at (C exp(h kappa_B), c + h c_B) -- the full tensor
//      rotated on all four slots, every stored block, h^eff, f, the sidecar's
//      rows, W and the orbital energies rebuilt from it at the displaced
//      state's own active density, the RDMs and the digests rebuilt, their
//      transpose with them -- and <Y, kappa_A> + <y, c_A> central-differenced
//      along h. Along exp(h B) exp(s A) that converges to E_thth[A, B] +
//      (1/2) Edot([kappa_B, kappa_A]) (Baker-Campbell-Hausdorff; Sec. 5.1's
//      parameterisation term), the second term the gated first-order pairing
//      along the commutator, at the state. Seeded columns B, every A in each
//      (the column's walk carries every kappa_A at once); h = 2^-k, the best
//      error relative to max(|target|, |Edot(Theta_A)|) under 1e-6 and a run
//      of three fourfold halvings, the gradient path's rule.
//   5  THE REFUSALS: the same matrix with no full tensor uploaded is refused
//      WHOLE, Unsupported (K2 for every pair, none of whose members is
//      active-active); and the written `.parhess` read back bit for bit.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries. A sidecar that does not fit
// the golden, or a refused matrix, is the caller's input (or the state's) and
// propagates unchanged to main, which reports it once.
#include "error_handling/error_macros.h"

export module nevpt2.app.sc_pseudodensity:parameter_hessian_check;

import std;
import nevpt2.gradient;
import nevpt2.kappa_rotation;
import nevpt2.golden;
import nevpt2.rdm_host_dm;
// generalIntegrals / uploadGeneral: the gradient sidecar as --kappa-full reads
// it; scaled.
import :kappa_full_check;
// The full tensor's host builder: --sc-hessian-kk-int's mixed ladders' corners.
import :hessian_kk_check;
// kcDot.
import :hessian_kc_check;

// Every exact identity's gate: two computations of one number, the tree's 1e-7
// (docs/testing.md, "The parameter Hessian over the responses", records what
// is measured).
constexpr double kPhTol = 1e-7;
// The sidecars against themselves and the host builder against the stored
// arrays: separate transforms of the same integrals, reassociation only.
constexpr double kPhSelfTol = 1e-10;
// The ladders: h = 2^-k for k in [first, last], the best error relative to
// max(|target|, |Edot(Theta_A)|) under kPhFdTol, and a run of kPhFdRun
// fourfold halvings -- the rule every finite difference of the gradient path
// is held to. They start at 2^-10, as #220's ladders of the same Y and y do:
// along a response direction the gated source terms have structure on a
// scale of h ~ 1e-3, above which the error is not yet O(h^2) (docs/testing.md,
// "The parameter Hessian over the responses", has the rungs).
constexpr int kPhFdFirstExp = 10;
constexpr int kPhFdLastExp = 20;
constexpr std::size_t kPhFdSteps = kPhFdLastExp - kPhFdFirstExp + 1;
constexpr double kPhFdTol = 1e-6;
constexpr int kPhFdRun = 3;
// How many columns B the ladders run (testing-cost rule: seeded columns, every
// A in each, never all (3N)^2), and the seed that picks them.
constexpr std::size_t kPhLadderColumns = 2;
constexpr std::uint64_t kPhSeed = 0x5eed'0222u;
// How many of the orbital block's off-diagonal pairs the symmetry check walks
// a second time with the directions swapped (the matrix holds one walk per
// pair A <= B; a pair walk is the block's whole cost).
constexpr std::size_t kPhSwappedPairs = 4;

double phMaxAbs(const nevpt2::Tensor& t) {
  double m = 0.0;
  for (std::int64_t i = 0; i < t.size(); ++i) m = std::max(m, std::fabs(t.flat(i)));
  return m;
}

// max |a - b| / max |b|, two host tensors of one shape (1.0 if the shapes
// differ, so a mis-shaped rebuild cannot pass).
double phRelGap(const nevpt2::Tensor& a, const nevpt2::Tensor& b) {
  if (a.dims() != b.dims()) return 1.0;
  double gap = 0.0;
  for (std::int64_t i = 0; i < a.size(); ++i) gap = std::max(gap, std::fabs(a.flat(i) - b.flat(i)));
  return gap / std::max(1e-300, phMaxAbs(b));
}

// The gradient sidecar's inventory at the orbitals the ROTATED tensor `e`
// describes, at the active density `gamma`: h and h^eff as given, f =
// h^eff + sum_tu gamma_tu [(pq|tu) - (1/2)(pt|qu)] over the whole range, the
// fourteen full rows g_xP_QR[x, p, q, r] = (x p | q r) and the four tied ones
// as generate_golden.py's _gradient_arrays stores them. Into a copy of `gg`.
nevpt2::GradGoldFile phRotatedSidecar(const nevpt2::GradGoldFile& gg, const nevpt2::Tensor& e,
                                      const nevpt2::Tensor& h, const nevpt2::Tensor& heff,
                                      const nevpt2::Tensor& gamma) {
  const std::int64_t nc = gg.ncore, na = gg.nact, nv = gg.nvirt, n = gg.nmo;
  const auto lo = [&](char sp) { return sp == 'c' ? 0 : (sp == 'a' ? nc : nc + na); };
  const auto ext = [&](char sp) { return sp == 'c' ? nc : (sp == 'a' ? na : nv); };
  nevpt2::GradGoldFile out = gg;
  out.arrays["h"] = h;
  out.arrays["heff"] = heff;
  nevpt2::Tensor f = heff;
  for (std::int64_t p = 0; p < n; ++p)
    for (std::int64_t q = 0; q < n; ++q) {
      double v = 0.0;
      for (std::int64_t t = 0; t < na; ++t)
        for (std::int64_t u = 0; u < na; ++u)
          v += gamma(t, u) * (e(p, q, nc + t, nc + u) - 0.5 * e(p, nc + t, q, nc + u));
      f(p, q) += v;
    }
  out.arrays["f"] = std::move(f);
  for (const char* name : {"g_xa_aa", "g_xv_aa", "g_xc_aa", "g_xa_va", "g_xv_va", "g_xc_va",
                           "g_xa_ca", "g_xv_ca", "g_xc_ca", "g_xa_cv", "g_xv_cv", "g_xc_cv",
                           "g_xa_cc", "g_xv_cc"}) {
    const char sp = name[3], sq = name[5], sr = name[6];
    nevpt2::Tensor x({n, ext(sp), ext(sq), ext(sr)});
    for (std::int64_t a = 0; a < n; ++a)
      for (std::int64_t p = 0; p < ext(sp); ++p)
        for (std::int64_t q = 0; q < ext(sq); ++q)
          for (std::int64_t r = 0; r < ext(sr); ++r)
            x(a, p, q, r) = e(a, lo(sp) + p, lo(sq) + q, lo(sr) + r);
    out.arrays[name] = std::move(x);
  }
  nevpt2::Tensor jj({n, nc, nc}), kk({n, nc, nc}), avv({n, na, nv}), cvv({n, nc, nv});
  for (std::int64_t a = 0; a < n; ++a) {
    for (std::int64_t i = 0; i < nc; ++i)
      for (std::int64_t j = 0; j < nc; ++j) {
        jj(a, i, j) = e(a, i, j, j);  // (x i | j j)
        kk(a, j, i) = e(a, j, j, i);  // (x j | j i)
      }
    for (std::int64_t r = 0; r < nv; ++r) {
      const std::int64_t rr = nc + na + r;
      for (std::int64_t t = 0; t < na; ++t) avv(a, t, r) = e(a, nc + t, rr, rr);
      for (std::int64_t i = 0; i < nc; ++i) cvv(a, i, r) = e(a, i, rr, rr);
    }
  }
  out.arrays["g_xc_cc_j"] = std::move(jj);
  out.arrays["g_xc_cc_k"] = std::move(kk);
  out.arrays["g_xa_vv"] = std::move(avv);
  out.arrays["g_xc_vv"] = std::move(cvv);
  return out;
}

// The arrays phRotatedSidecar rebuilds, which check 0 holds to the stored ones.
constexpr std::array<std::string_view, 21> kPhSidecarNames{
    "h",       "heff",    "f",       "g_xa_aa", "g_xv_aa", "g_xc_aa",   "g_xa_va",
    "g_xv_va", "g_xc_va", "g_xa_ca", "g_xv_ca", "g_xc_ca", "g_xa_cv",   "g_xv_cv",
    "g_xc_cv", "g_xa_cc", "g_xv_cc", "g_xc_cc_j", "g_xc_cc_k", "g_xa_vv", "g_xc_vv"};

// What a ladder point needs that does not move with it.
struct PhLadder {
  const nevpt2::GoldenFile& g;
  const nevpt2::GradGoldFile& gg;
  const nevpt2::Tensor& eri;
  const nevpt2::HostDm12Builder& builder;
  const nevpt2::RdmBuildOptions& opt;
  const nevpt2::SigmaVectorOptions& sigma;
  const std::vector<nevpt2::ScParameterDirection>& dirs;
};

// One displaced state's inputs, built and uploaded: the orbitals C U, the CI
// vector `ch`. `fields` (the stored blocks) and `side` (the sidecar) are what
// check 0 compares at U = 1.
struct PhPointInputs {
  KkHostInts fields;
  nevpt2::GradGoldFile side;
  nevpt2::Tensor dm1, dm2;
};

PhPointInputs phInputsAt(const PhLadder& l, const nevpt2::Tensor& u, const nevpt2::Tensor& ch) {
  const nevpt2::Tensor e = nevpt2::rotateAxes(l.eri, {0, 1, 2, 3}, u);
  const nevpt2::Tensor h = nevpt2::rotateAxes(l.gg.get("h"), {0, 1}, u);
  const nevpt2::HostDm12 dm = l.builder.dm12(ch);
  // kkIntsFromRotatedFull reads the partition and gamma, never p.eri / p.h.
  const KkFullParts p{nevpt2::Tensor{}, nevpt2::Tensor{}, dm.dm1, l.gg.ncore, l.gg.nact,
                      l.gg.nvirt};
  const nevpt2::Tensor heff = kkHeffFromRotated(p, e, h);
  return PhPointInputs{kkIntsFromRotatedFull(p, e, heff), phRotatedSidecar(l.gg, e, h, heff, dm.dm1),
                       dm.dm1, dm.dm2};
}

// <Y, kappa_A> + <y, c_A> for every A in `as`, at (C U, ch): the gated source
// terms recomputed in full there (the digests rebuilt at the displaced h2e,
// their transpose with them, W at the displaced rows).
std::vector<double> phGradientAt(const PhLadder& l, const nevpt2::Tensor& u,
                                 const nevpt2::Tensor& ch, const std::vector<std::size_t>& as,
                                 const nevpt2::DeviceResources& res) {
  const PhPointInputs in = phInputsAt(l, u, ch);
  const nevpt2::NevptIntegralsDevice ints = kkUploadCombination({in.fields}, {1.0}, res);
  const nevpt2::GeneralIntegralsDevice gen = uploadGeneral(in.side, res);
  const nevpt2::OrbitalEnergyResponse w = nevpt2::orbitalEnergyResponse(
      l.gg.ncore, l.gg.nact, l.gg.nvirt, in.side.get("g_xa_cc"), in.side.get("g_xc_ca"),
      in.side.get("g_xa_vv"), in.side.get("g_xv_va"));
  const nevpt2::Tensor* h2e = nullptr;
  for (const auto& [name, t] : in.fields.fields)
    if (name == "h2e") h2e = &t;
  const nevpt2::DeviceTensor dm1 = nevpt2::uploadTensor(in.dm1, res);
  const nevpt2::DeviceTensor dm2 = nevpt2::uploadTensor(in.dm2, res);
  const nevpt2::RdmBuildResult b =
      nevpt2::buildRdmsDevice(l.opt, ch, l.g.ncas, l.g.nelecA, l.g.nelecB, *h2e, res);
  const nevpt2::DeviceTensor hbar = nevpt2::scDigestAdjointDevice(
      l.opt, ch, l.g.nelecA, l.g.nelecB, ints, dm1, dm2, b.dm3, b.f3ac, b.f3ca, res);
  std::vector<nevpt2::ScRotationRequest> reqs;
  for (const std::size_t a : as) reqs.push_back(nevpt2::ScRotationRequest{l.dirs[a].kappa, false});
  // A ladder point's sidecar is built to fit its integrals, so an Error here
  // is our bug.
  const nevpt2::Result<nevpt2::ScFullRotationResult> y1 = nevpt2::scFullRotationDevice(
      ints, gen, dm1, dm2, b.dm3, b.f3ac, b.f3ca, &hbar, reqs, res);
  nevpt2::check(y1.has_value(), "phGradientAt: the displaced sidecar does not fit its integrals");
  const nevpt2::Tensor y = nevpt2::scCiSourceDevice(ints, ch, l.g.nelecA, l.g.nelecB, dm1, dm2,
                                                    b.dm3, b.f3ac, b.f3ca, l.sigma, &w, res)
                               .y;
  std::vector<double> out;
  for (std::size_t i = 0; i < as.size(); ++i)
    out.push_back(y1->dirs[i].adjoint + kcDot(l.dirs[as[i]].ci, y));
  return out;
}

// Coordinate k = 3 atom + axis as "atom.axis".
std::string phLabel(const std::int64_t k) { return std::format("{}.{}", k / 3, "xyz"[k % 3]); }

// One block of the matrix, printed, its rows labelled by coordinate.
void phPrintMatrix(const char* name, const nevpt2::Tensor& m,
                   const std::vector<std::int64_t>& sel) {
  const std::int64_t n = m.dim(0);
  std::printf("  %s:\n", name);
  for (std::int64_t a = 0; a < n; ++a) {
    std::printf("    %5s", phLabel(sel[static_cast<std::size_t>(a)]).c_str());
    for (std::int64_t b = 0; b < n; ++b) std::printf(" %11.6f", m(a, b));
    std::printf("\n");
  }
}

// The columns the ladders run: kPhLadderColumns coordinates, seeded.
std::vector<std::size_t> phLadderColumns(std::size_t n) {
  std::vector<std::size_t> all(n);
  std::iota(all.begin(), all.end(), std::size_t{0});
  std::mt19937_64 gen(kPhSeed);
  std::shuffle(all.begin(), all.end(), gen);
  all.resize(std::min(n, kPhLadderColumns));
  std::sort(all.begin(), all.end());
  return all;
}

nevpt2::Result<bool> checkScParameterHessian(
    const nevpt2::GoldenFile& g, const nevpt2::GradGoldFile& gg, const std::string& respPath,
    const nevpt2::EriGoldFile* eri, const std::string& outPath,
    const std::vector<std::int64_t>& coords, const nevpt2::NevptIntegralsDevice& ints,
    const nevpt2::DeviceTensor& dm1,
    const nevpt2::DeviceTensor& dm2, const nevpt2::RdmBuildResult& built,
    const nevpt2::RdmBuildOptions& opt, const nevpt2::DeviceResources& res) {
  const nevpt2::RdmTensors rdm{dm1, dm2, built.dm3, built.f3ac, built.f3ca};
  const nevpt2::Tensor& ci = g.get("ci");
  nevpt2::SigmaVectorOptions sigma;
  sigma.nTiles = opt.nTiles;
  std::printf("\n--sc-parameter-hessian: the SC parameter Hessian as a bilinear form over the "
              "responses, E_thth[Theta_A, Theta_B]\n  (docs/hessian-theory.md, Sec. 8), the "
              "orbital energies responding throughout\n");
  std::printf("response sidecar:  %s\n", respPath.c_str());
  const nevpt2::RespGoldFile rg = NEVPT2_TRY(nevpt2::loadRespGold(respPath));
  NEVPT2_TRY(rg.checkAgainst(g, gg));
  if (gg.densityFitted())
    return nevpt2::err_unsupported(
        "--sc-parameter-hessian on a density-fitted case: the mixed block is integral-direct "
        "and the response sidecars are the conventional _casscf states'");
  nevpt2::GeneralIntegralsDevice gen = NEVPT2_TRY(generalIntegrals(g, gg, res));
  if (eri != nullptr) gen.eri = nevpt2::uploadTensor(eri->get("eri"), res);
  const std::int64_t nc = gg.ncore, na = gg.nact, nv = gg.nvirt, nmo = gg.nmo;
  // The coordinates the direction set is restricted to: every one, or the
  // block --parameter-hessian-coords chose (nevpt2.cli has held it strictly
  // increasing; its range is this sidecar's).
  std::vector<std::int64_t> sel = coords;
  if (sel.empty())
    for (std::int64_t k = 0; k < rg.ncoord(); ++k) sel.push_back(k);
  if (sel.back() >= rg.ncoord())
    return nevpt2::err_config(std::format(
        "--parameter-hessian-coords: coordinate {} is past the response sidecar's {} (3 x {} "
        "atoms)",
        sel.back(), rg.ncoord(), rg.natm));
  const auto n3 = static_cast<std::int64_t>(sel.size());
  const std::size_t n = sel.size();
  bool ok = true;
  std::string selText;
  for (const std::int64_t k : sel) selText += " " + phLabel(k);
  std::printf("  coordinates (atom.axis):%s%s\n", selText.c_str(),
              coords.empty() ? " -- every one" : " -- the block --parameter-hessian-coords chose");

  // --- 0. the sidecars --------------------------------------------------------------
  std::vector<nevpt2::ScParameterDirection> dirs;
  double asym = 0.0, gauge = 0.0, overlap = 0.0;
  for (const std::int64_t k : sel) {
    nevpt2::ScParameterDirection d{rg.kappa(k), rg.ciResponse(k)};
    asym = std::max(asym, nevpt2::antisymmetryResidual(d.kappa));
    for (std::int64_t p = 0; p < nmo; ++p)
      for (std::int64_t q = 0; q < nmo; ++q) {
        const auto space = [&](std::int64_t x) { return x < nc ? 0 : (x < nc + na ? 1 : 2); };
        if (space(p) == space(q)) gauge = std::max(gauge, std::fabs(d.kappa(p, q)));
      }
    overlap = std::max(overlap, std::fabs(kcDot(ci, d.ci)));
    dirs.push_back(std::move(d));
  }
  const bool dirsOk = asym < 1e-14 && gauge == 0.0 && overlap < kPhSelfTol;
  ok = ok && dirsOk;
  std::printf("\n  [0] %lld directions (of 3 x %lld atoms), in the golden's frame (ci and mo_coeff "
              "bit for bit):\n      max |kappa + kappa^T| = %.1e, max |kappa| on the three "
              "diagonal blocks = %.1e (0 exactly),\n      max |<c, c_A>| = %.1e (< %.0e): %s\n",
              static_cast<long long>(n3), static_cast<long long>(rg.natm), asym, gauge, overlap,
              kPhSelfTol, dirsOk ? "ok" : "FAILED");
  const nevpt2::ScGeneralCheck self =
      NEVPT2_TRY(nevpt2::scGeneralIntegralCheck(ints, gen, dm1, res));
  const double worstSelf = std::max({self.blocks, self.heff, self.fock, self.eps, self.fockFull});
  ok = ok && worstSelf < kPhSelfTol;
  std::printf("      the gradient sidecar against itself (relative): restrictions %.2e (%s), "
              "h^eff %.2e, f's diagonal %.2e,\n      (O2) %.2e, f over the whole range %.2e "
              "(< %.0e: %s)\n",
              self.blocks, self.worstBlock.c_str(), self.heff, self.fock, self.eps, self.fockFull,
              kPhSelfTol, worstSelf < kPhSelfTol ? "ok" : "FAILED");

  const nevpt2::OrbitalEnergyResponse wResp = nevpt2::orbitalEnergyResponse(
      nc, na, nv, gg.get("g_xa_cc"), gg.get("g_xc_ca"), gg.get("g_xa_vv"), gg.get("g_xv_va"));
  const nevpt2::DeviceTensor hbar = nevpt2::scDigestAdjointDevice(
      opt, ci, g.nelecA, g.nelecB, ints, dm1, dm2, built.dm3, built.f3ac, built.f3ca, res);

  // --- 1. the matrix -------------------------------------------------------------------
  const nevpt2::ScParameterHessianSetup setup{ints,  gen, rdm,   ci,    g.nelecA,
                                              g.nelecB, g.get("h2e"), opt, wResp, hbar};
  const auto t0 = std::chrono::steady_clock::now();
  const nevpt2::ScParameterHessianResult m =
      NEVPT2_TRY(nevpt2::scParameterHessianDevice(setup, dirs, sigma, res));
  const double tMatrix =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("\n  [1] the matrix in %.1fs: %lld orbital-block pairs (A <= B) in "
              "%.1fs, %lld H_kc^T kappa_A in %.1fs and %lld H_cc c_A vectors in %.1fs\n",
              tMatrix, static_cast<long long>(n3 * (n3 + 1) / 2), m.secondsKk,
              static_cast<long long>(n3), m.secondsKc, static_cast<long long>(n3), m.secondsCc);
  std::fflush(stdout);
  phPrintMatrix("E_thth[A, B] (Eh/bohr^2)", m.hessian, sel);
  std::printf("  block magnitudes: max |h_kk| = %.4e, max |h_kc| = %.4e, max |h_cc| = %.4e\n",
              phMaxAbs(m.kk), phMaxAbs(m.kc), phMaxAbs(m.cc));
  const nevpt2::Tensor y0 = nevpt2::scCiSourceDevice(ints, ci, g.nelecA, g.nelecB, dm1, dm2,
                                                     built.dm3, built.f3ac, built.f3ca, sigma,
                                                     &wResp, res)
                                .y;
  double yGap = m.ySpread;
  for (std::int64_t i = 0; i < y0.size(); ++i)
    yGap = std::max(yGap, std::fabs(y0.flat(i) - m.y.flat(i)));
  ok = ok && yGap < kPhTol;
  std::printf("  every H_cc walk's y against the gated y (scCiSourceDevice, FULL mode): max "
              "|delta| = %.2e (< %.0e: %s)\n",
              yGap, kPhTol, yGap < kPhTol ? "ok" : "FAILED");

  // --- 2. every entry against the per-block scalars -------------------------------------
  {
    nevpt2::Tensor kcF({n3, n3}), ccF({n3, n3});
    const nevpt2::ScMixedHessianSetup ms{ints, &gen, rdm, ci, g.nelecA, g.nelecB,
                                         opt,  true, true, &wResp};
    std::vector<nevpt2::Tensor> cis;
    for (const auto& d : dirs) cis.push_back(d.ci);
    for (std::size_t a = 0; a < n; ++a) {
      const std::vector<nevpt2::ScMixedHessianResult> f =
          NEVPT2_TRY(nevpt2::scMixedHessianDevice(ms, dirs[a].kappa, cis, res));
      for (std::size_t b = 0; b < n; ++b)
        kcF(static_cast<std::int64_t>(a), static_cast<std::int64_t>(b)) = f[b].eddot;
    }
    nevpt2::Result<nevpt2::HostDm12Builder> builder =
        nevpt2::HostDm12Builder::create(g.ncas, g.nelecA, g.nelecB);
    if (!builder) return std::unexpected(builder.error());
    nevpt2::FullBlockSource src(ints);
    const nevpt2::ActiveIntegralsDevice ai{ints.h1e.view(),      ints.h2e.view(),
                                           ints.e_core.view(),   ints.e_virt.view(),
                                           ints.h1e_v_Sr.view(), ints.h1e_v_Si.view(),
                                           ints.h1e_v_Sir.view()};
    for (std::size_t a = 0; a < n; ++a)
      for (std::size_t b = a; b < n; ++b) {
        const nevpt2::Tensor& u = dirs[a].ci;
        const nevpt2::Tensor& v = dirs[b].ci;
        const nevpt2::HostDm12 dotU = builder->dm12Dot(ci, u);
        const nevpt2::HostDm12 dotV = builder->dm12Dot(ci, v);
        const nevpt2::HostDm12 ddot = builder->dm12Dot(u, v);
        const nevpt2::DeviceTensor d1U = nevpt2::uploadTensor(dotU.dm1, res);
        const nevpt2::DeviceTensor d2U = nevpt2::uploadTensor(dotU.dm2, res);
        const nevpt2::DeviceTensor d1V = nevpt2::uploadTensor(dotV.dm1, res);
        const nevpt2::DeviceTensor d2V = nevpt2::uploadTensor(dotV.dm2, res);
        const nevpt2::DeviceTensor d1UV = nevpt2::uploadTensor(ddot.dm1, res);
        const nevpt2::DeviceTensor d2UV = nevpt2::uploadTensor(ddot.dm2, res);
        const nevpt2::RdmHessianTangents t3 = nevpt2::buildRdmHessianTangentsDevice(
            opt, ci, u, v, g.ncas, g.nelecA, g.nelecB, g.get("h2e"), res);
        const nevpt2::RdmTensors setU{d1U, d2U, t3.cu.dm3, t3.cu.f3ac, t3.cu.f3ca};
        const nevpt2::RdmTensors setV{d1V, d2V, t3.cv.dm3, t3.cv.f3ac, t3.cv.f3ca};
        const nevpt2::RdmTensors setUV{d1UV, d2UV, t3.uv.dm3, t3.uv.f3ac, t3.uv.f3ca};
        const nevpt2::OrbitalEnergyHessianDots eps{nevpt2::orbitalEnergyDot(wResp, dotU.dm1),
                                                   nevpt2::orbitalEnergyDot(wResp, dotV.dm1),
                                                   nevpt2::orbitalEnergyDot(wResp, ddot.dm1)};
        const double e = nevpt2::energiesHessianDevice(ai, src, /*batch=*/0, rdm, setU, setV,
                                                       setUV, &eps, res)
                             .totalDDot;
        ccF(static_cast<std::int64_t>(a), static_cast<std::int64_t>(b)) = e;
        ccF(static_cast<std::int64_t>(b), static_cast<std::int64_t>(a)) = e;
      }
    double worstKc = 0.0, worstCc = 0.0, worstAll = 0.0;
    for (std::int64_t a = 0; a < n3; ++a)
      for (std::int64_t b = 0; b < n3; ++b) {
        worstKc = std::max(worstKc, std::fabs(m.kc(a, b) - kcF(a, b)));
        worstCc = std::max(worstCc, std::fabs(m.cc(a, b) - ccF(a, b)));
        const double scalar = m.kk(a, b) + kcF(a, b) + kcF(b, a) + ccF(a, b);
        worstAll = std::max(worstAll, std::fabs(m.hessian(a, b) - scalar));
      }
    const bool good = worstKc < kPhTol && worstCc < kPhTol && worstAll < kPhTol;
    ok = ok && good;
    std::printf("\n  [2] every entry against the per-block flags' scalars on the same pair "
                "(%lld entries):\n      h_kc vs --sc-hessian-kc's Eddot_kc(kappa_A, c_B), every "
                "ordered pair: worst |delta| = %.2e\n      h_cc vs --sc-hessian's Eddot(c_A, "
                "c_B) (FULL mode), every pair, both orders: worst |delta| = %.2e\n      h_kk IS "
                "--sc-hessian-kk's Eddot_kk(kappa_A, kappa_B), every pair A <= B\n      the "
                "entry assembled from the scalars: worst |delta| = %.2e (< %.0e: %s)\n",
                static_cast<long long>(n3 * n3), worstKc, worstCc, worstAll, kPhTol,
                good ? "ok" : "FAILED");
  }

  // --- 3. symmetry ------------------------------------------------------------------------
  // The CI block on every entry (two different walks, <c_B, H_cc c_A> and
  // <c_A, H_cc c_B>); the mixed block's two orientations sum to a symmetric
  // matrix by construction; the orbital block's (B, A) walk -- the pair's
  // directions swapped -- against the (A, B) the matrix holds, on every
  // off-diagonal pair up to kPhSwappedPairs, seeded beyond that (one pair walk
  // is the block's whole cost).
  {
    double sCc = 0.0, sAll = 0.0;
    for (std::int64_t a = 0; a < n3; ++a)
      for (std::int64_t b = 0; b < n3; ++b) {
        sCc = std::max(sCc, std::fabs(m.cc(a, b) - m.cc(b, a)));
        sAll = std::max(sAll, std::fabs(m.hessian(a, b) - m.hessian(b, a)));
      }
    std::vector<std::pair<std::size_t, std::size_t>> offDiagonal;
    for (std::size_t a = 0; a < n; ++a)
      for (std::size_t b = a + 1; b < n; ++b) offDiagonal.emplace_back(a, b);
    std::mt19937_64 pick(kPhSeed + 1);
    std::shuffle(offDiagonal.begin(), offDiagonal.end(), pick);
    const std::size_t total = offDiagonal.size();
    offDiagonal.resize(std::min(total, kPhSwappedPairs));
    std::sort(offDiagonal.begin(), offDiagonal.end());
    const nevpt2::ScOrbitalCurvatureSetup cs{ints, gen, rdm, &hbar, &ci, g.nelecA, g.nelecB, opt};
    double sKk = 0.0;
    std::string swapped;
    for (const auto& [a, b] : offDiagonal) {
      const std::vector<nevpt2::ScOrbitalHessianResult> r = NEVPT2_TRY(
          nevpt2::scOrbitalHessianDevice(cs, {{dirs[b].kappa, dirs[a].kappa}}, res));
      sKk = std::max(sKk, std::fabs(r.front().value() - m.kk(static_cast<std::int64_t>(a),
                                                             static_cast<std::int64_t>(b))));
      swapped += " (" + phLabel(sel[b]) + ", " + phLabel(sel[a]) + ")";
    }
    const bool good = sAll < kPhTol && sKk < kPhTol && sCc < kPhTol;
    ok = ok && good;
    std::printf("\n  [3] symmetry: max |h_cc - h_cc^T| = %.2e on every entry; h_kc + h_kc^T is "
                "symmetric by construction;\n      h_kk's swapped walks Eddot_kk(kappa_B, "
                "kappa_A) against Eddot_kk(kappa_A, kappa_B) on %zu of the %zu pairs A < B "
                "(seed 0x%llx):%s\n      max |delta| = %.2e; max |E_thth - E_thth^T| = %.2e "
                "(relative to max |E_thth| = %.4e: %.2e) (< %.0e: %s)\n",
                sCc, offDiagonal.size(), total, static_cast<unsigned long long>(kPhSeed + 1),
                swapped.c_str(), sKk, sAll, phMaxAbs(m.hessian),
                sAll / std::max(1e-300, phMaxAbs(m.hessian)), kPhTol, good ? "ok" : "FAILED");
  }

  // --- 4. the ladders ---------------------------------------------------------------------
  if (eri == nullptr) {
    std::printf("\n  [4] the ladders: not run -- they displace the orbitals out of the full "
                "tensor, and none was given\n");
  } else {
    nevpt2::Result<nevpt2::HostDm12Builder> built12 =
        nevpt2::HostDm12Builder::create(g.ncas, g.nelecA, g.nelecB);
    if (!built12) return std::unexpected(built12.error());
    const PhLadder lad{g, gg, eri->get("eri"), *built12, opt, sigma, dirs};
    std::vector<std::size_t> every(n);
    std::iota(every.begin(), every.end(), std::size_t{0});
    nevpt2::Tensor one({nmo, nmo});
    for (std::int64_t q = 0; q < nmo; ++q) one(q, q) = 1.0;

    // The host builder at U = 1 against the stored arrays, and the
    // first-order pairings it gives against the state's own.
    {
      const PhPointInputs at = phInputsAt(lad, one, ci);
      double worst = 0.0;
      std::string worstName;
      for (const auto& [name, mine] : at.fields.fields) {
        const double r = phRelGap(mine, g.get(name));
        if (r >= worst) {
          worst = r;
          worstName = name;
        }
      }
      for (const std::string_view name : kPhSidecarNames) {
        const double r = phRelGap(at.side.get(std::string(name)), gg.get(std::string(name)));
        if (r >= worst) {
          worst = r;
          worstName = name;
        }
      }
      ok = ok && worst < kPhSelfTol;
      std::printf("\n  [4] the ladders' host builder at U = 1 against the golden's blocks and "
                  "orbital energies and the sidecar's\n      h, h^eff, f and eighteen rows "
                  "(relative, per array): worst %.2e (%s) (< %.0e: %s)\n",
                  worst, worstName.c_str(), kPhSelfTol, worst < kPhSelfTol ? "ok" : "FAILED");
    }
    std::vector<nevpt2::ScRotationRequest> reqs;
    for (const auto& d : dirs) reqs.push_back(nevpt2::ScRotationRequest{d.kappa, false});
    const nevpt2::ScFullRotationResult y1 = NEVPT2_TRY(nevpt2::scFullRotationDevice(
        ints, gen, dm1, dm2, built.dm3, built.f3ac, built.f3ca, &hbar, reqs, res));
    std::vector<double> edot(n);
    for (std::size_t a = 0; a < n; ++a) edot[a] = y1.dirs[a].adjoint + kcDot(dirs[a].ci, y0);
    {
      const std::vector<double> base = phGradientAt(lad, one, ci, every, res);
      double gap = 0.0, scale = 0.0;
      for (std::size_t a = 0; a < n; ++a) {
        gap = std::max(gap, std::fabs(base[a] - edot[a]));
        scale = std::max(scale, std::fabs(edot[a]));
      }
      const double rel = gap / std::max(1e-300, scale);
      ok = ok && rel < kPhSelfTol;
      std::printf("      Edot(Theta_A) = <Y, kappa_A> + <y, c_A> from the rebuilt inputs at U = "
                  "1 against the state's: %.2e relative (< %.0e: %s)\n",
                  rel, kPhSelfTol, rel < kPhSelfTol ? "ok" : "FAILED");
    }
    const std::vector<std::size_t> cols = phLadderColumns(n);
    std::string colText;
    for (const std::size_t b : cols) colText += " " + phLabel(sel[b]);
    std::printf("      columns B (seed 0x%llx):%s; every A in each -- %zu of the %zu entries, "
                "each against its own ladder\n",
                static_cast<unsigned long long>(kPhSeed), colText.c_str(), cols.size() * n,
                n * n);
    std::printf("      along exp(h kappa_B) exp(s kappa_A): the target is E_thth[A, B] + (1/2) "
                "Edot([kappa_B, kappa_A]); h = 2^-k, k = %d..%d\n",
                kPhFdFirstExp, kPhFdLastExp);
    int laddered = 0, passed = 0;
    for (const std::size_t b : cols) {
      // The BCH term: the gated first-order pairing along each commutator.
      std::vector<nevpt2::ScRotationRequest> comm;
      for (std::size_t a = 0; a < n; ++a) {
        const nevpt2::Tensor ba = kkMatmul(dirs[b].kappa, dirs[a].kappa);
        const nevpt2::Tensor ab = kkMatmul(dirs[a].kappa, dirs[b].kappa);
        nevpt2::Tensor c(ba.dims());
        for (std::int64_t i = 0; i < c.size(); ++i) c.flatRef(i) = ba.flat(i) - ab.flat(i);
        comm.push_back(nevpt2::ScRotationRequest{std::move(c), false});
      }
      const nevpt2::ScFullRotationResult bch = NEVPT2_TRY(nevpt2::scFullRotationDevice(
          ints, gen, dm1, dm2, built.dm3, built.f3ac, built.f3ca, &hbar, comm, res));
      std::vector<std::array<double, kPhFdSteps>> err(n);
      std::array<double, kPhFdSteps> step{};
      for (std::size_t i = 0; i < kPhFdSteps; ++i) {
        const double h = std::ldexp(1.0, -(kPhFdFirstExp + static_cast<int>(i)));
        step[i] = h;
        const auto at = [&](double s) {
          nevpt2::Tensor ch = ci;
          for (std::int64_t q = 0; q < ch.size(); ++q) ch.flatRef(q) += s * dirs[b].ci.flat(q);
          return phGradientAt(lad, nevpt2::expKappa(scaled(dirs[b].kappa, s)), ch, every, res);
        };
        const std::vector<double> plus = at(h), minus = at(-h);
        for (std::size_t a = 0; a < n; ++a) {
          const double target =
              m.hessian(static_cast<std::int64_t>(a), static_cast<std::int64_t>(b)) +
              0.5 * bch.dirs[a].adjoint;
          err[a][i] = std::fabs((plus[a] - minus[a]) / (2.0 * h) - target);
        }
      }
      std::printf("\n      column B = %s:\n", phLabel(sel[b]).c_str());
      std::printf("      %6s %20s %16s %10s %8s %7s %s\n", "A", "E_thth[A, B]", "(1/2) Edot([B,A])",
                  "best rel", "at k", "run", "");
      for (std::size_t a = 0; a < n; ++a) {
        const double target =
            m.hessian(static_cast<std::int64_t>(a), static_cast<std::int64_t>(b)) +
            0.5 * bch.dirs[a].adjoint;
        const double scale = std::max(std::fabs(target), std::fabs(edot[a]));
        std::size_t bestAt = 0;
        for (std::size_t i = 1; i < kPhFdSteps; ++i)
          if (err[a][i] < err[a][bestAt]) bestAt = i;
        const double best = err[a][bestAt] / std::max(1e-300, scale);
        const nevpt2::FourfoldRun run = nevpt2::longestFourfoldRun(err[a]);
        const bool good = best < kPhFdTol && run.length >= kPhFdRun;
        ++laddered;
        if (good) ++passed;
        ok = ok && good;
        std::printf("      %6s %20.12e %16.6e %10.2e %8d %7d %s\n", phLabel(sel[a]).c_str(),
                    m.hessian(static_cast<std::int64_t>(a), static_cast<std::int64_t>(b)),
                    0.5 * bch.dirs[a].adjoint, best, kPhFdFirstExp + static_cast<int>(bestAt),
                    run.length, good ? "ok" : "FAILED");
      }
      // The diagonal entry's rungs in full: the one a nuclear Hessian's
      // diagonal rests on.
      std::printf("      the diagonal entry's rungs, |err| relative:");
      const double dscale = std::max(
          std::fabs(m.hessian(static_cast<std::int64_t>(b), static_cast<std::int64_t>(b)) +
                    0.5 * bch.dirs[b].adjoint),
          std::fabs(edot[b]));
      for (std::size_t i = 0; i < kPhFdSteps; ++i)
        std::printf(" %.1e", err[b][i] / std::max(1e-300, dscale));
      std::printf("\n");
    }
    std::printf("\n      ladders: %d of %d entries converge (best < %.0e relative, %d fourfold "
                "halvings)\n",
                passed, laddered, kPhFdTol, kPhFdRun);
  }

  // --- 5. the refusals, and the written matrix ------------------------------------------------
  {
    nevpt2::GeneralIntegralsDevice noEri = NEVPT2_TRY(generalIntegrals(g, gg, res));
    const nevpt2::ScParameterHessianSetup bare{ints,     noEri,    rdm,          ci,  g.nelecA,
                                               g.nelecB, g.get("h2e"), opt, wResp, hbar};
    const nevpt2::Result<nevpt2::ScParameterHessianResult> r =
        nevpt2::scParameterHessianDevice(bare, dirs, sigma, res);
    const bool refused = !r && r.error().kind == nevpt2::ErrorKind::Unsupported;
    ok = ok && refused;
    std::printf("\n  [5] the same matrix with no full tensor uploaded: %s\n",
                refused ? std::format("refused whole, Unsupported (\"{:.150}...\")",
                                      r.error().message)
                              .c_str()
                        : "NOT REFUSED");
  }
  if (!outPath.empty()) {
    nevpt2::ParHessFile p{rg.natm, nmo, nc, na, nv, rg.ndet, sel, m.hessian, m.kk, m.kc, m.cc};
    NEVPT2_TRY(nevpt2::writeParHess(outPath, p));
    const nevpt2::ParHessFile back = NEVPT2_TRY(nevpt2::loadParHess(outPath));
    bool same = back.coords == p.coords && back.natm == p.natm && back.nmo == p.nmo &&
                back.ncore == p.ncore &&
                back.nact == p.nact && back.nvirt == p.nvirt && back.ndet == p.ndet;
    for (const auto& [x, y] : {std::pair{&back.hessian, &p.hessian}, std::pair{&back.kk, &p.kk},
                               std::pair{&back.kc, &p.kc}, std::pair{&back.cc, &p.cc}})
      for (std::int64_t i = 0; same && i < x->size(); ++i)
        same = std::bit_cast<std::uint64_t>(x->flat(i)) == std::bit_cast<std::uint64_t>(y->flat(i));
    ok = ok && same;
    std::printf("      written to %s and read back: %s\n", outPath.c_str(),
                same ? "bit for bit" : "DIFFERENT");
  }

  std::printf("\n%s: the SC parameter Hessian over the %lld response directions matches the "
              "per-block flags' scalars on every entry to %.0e, is symmetric to %.0e, and the "
              "ladders of the gated Y and y converge to it%s (orbital energies responding "
              "throughout: to kappa, and to c through W)\n",
              ok ? "PASS" : "FAIL", static_cast<long long>(n3), kPhTol, kPhTol,
              eri == nullptr ? " -- NOT RUN, no full tensor" : "");
  return ok;
}
