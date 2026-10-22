// nevpt2.app.sc_pseudodensity:integral_adjoint_check -- the checks the
// integral adjoint --dump-source-terms writes (#246) is held to before it is
// written: dE_corr/d(integrals), SC or (--pc) PC, at fixed orbitals, CI
// vector and RDMs (nevpt2.gradient, "The integral adjoint"). An interface
// partition that exports nothing, as :kappa_full_check is.
//
// WHAT IT IS FOR. reference_data/nevpt2_grad.py's nuclear term differences
// Phi along the frozen orthonormal path (NUC_STEP); its E_corr half is the
// adjoint written here contracted with each coordinate's frozen-path integral
// derivative, an integral perturbation sidecar's dotted inventory (#220,
// reference_data/integral_perturbation.py) -- dE_corr/dR_k = <Ibar, Idot_k>,
// summed over the 21 inventory arrays, with no step (#247).
//
// THE PERTURBATION. A seeded integral perturbation (hdot, (pq|rs)dot) over the
// WHOLE MO range: each integral of the state scaled by a seeded factor in
// +-kAdjScale, a hash of its canonical index, so hdot is symmetric and
// (pq|rs)dot keeps the eight-fold symmetry of a real two-electron integral,
// and no n_mo^4 tensor is ever held (each integral is read off the sidecar's
// rows by that symmetry, InventoryLookup). Scaled rather than drawn whole: an
// integral that is zero at the state stays zero, as it does along a nuclear
// coordinate that keeps the point group, so the perturbation populates no
// perturber the integrals leave empty -- a draw over every element did, and
// SC's energy along it is smooth only on a scale too small for a ladder to
// resolve reliably (docs/testing.md, "The integral adjoint"). Its dotted inventory is built on the host by
// integral_perturbation.py's definitions (`inventory`: h, h^eff with the core
// dressing, f at the golden's dm1, the eighteen rows), at fixed orbitals and
// fixed active density exactly as the sidecar's -- so it is a perturbation of
// the kind #247 contracts, not an arbitrary set of blocks.
//
// THE CHECKS, in the order they print:
//
//   a  the seeded dot against itself: h^eff's dressing and f rebuilt from its
//      rows (scGeneralIntegralCheck on the derived blocks), 1e-10 relative --
//      what says the host builder is the sidecar's definitions;
//   b  the scatter's duality: <Ibar, Idot> on the inventory against the same
//      pairing on the stored blocks integralPerturbationDevice derives from
//      it, 1e-10 relative -- the pullback onto the inventory is the
//      restrictions' transpose;
//   c  THE PAIRING against a ladder of the GPU's own E_corr (energiesDevice,
//      or pcEnergiesDevice with --pc) along I + h Idot -- every stored block,
//      e_core / e_virt displaced, the digests rebuilt at the displaced h2e --
//      central differences, the best within kAdjTol (the tree's 1e-7) of the
//      pairing relative to max(1, |pairing|), on a run of kAdjRun fourfold
//      halvings (longestFourfoldRun: the rule every finite difference of the
//      gradient path is held to, Python's fourfold_runs). A rung at which the
//      energy refuses (a PC denominator turned negative at a coarse step) has
//      no quotient: printed REFUSED, an infinite error that breaks any run.
//      The measured ladders are docs/testing.md, "The integral adjoint".
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries.
#include "error_handling/error_macros.h"

export module nevpt2.app.sc_pseudodensity:integral_adjoint_check;

import std;
import nevpt2.gradient;
import nevpt2.golden;

// The pairing's gate: the tree's 1e-7, relative to max(1, |pairing|).
constexpr double kAdjTol = 1e-7;
// Checks a and b: exact identities, reassociation only.
constexpr double kAdjSelfTol = 1e-10;
// The seed of the perturbation, distinct from every other seed in this binary.
constexpr std::uint64_t kAdjSeed = 0x5eed'0246u;
// Each integral's factor is uniform in [-kAdjScale, kAdjScale).
constexpr double kAdjScale = 0.1;
// The ladder: h = 2^-k for k in [kAdjFirstExp, kAdjLastExp], and the run of
// fourfold halvings it needs.
constexpr int kAdjFirstExp = 6;
constexpr int kAdjLastExp = 22;
constexpr std::size_t kAdjSteps = kAdjLastExp - kAdjFirstExp + 1;
constexpr int kAdjRun = 3;

// splitmix64: one well-mixed 64-bit word per key.
std::uint64_t adjMix(std::uint64_t x) {
  x += 0x9e37'79b9'7f4a'7c15u;
  x = (x ^ (x >> 30)) * 0xbf58'476d'1ce4'e5b9u;
  x = (x ^ (x >> 27)) * 0x94d0'49bb'1331'11ebu;
  return x ^ (x >> 31);
}

// Uniform in [-kAdjScale, kAdjScale) from a key.
double adjUniform(const std::uint64_t key) {
  const double u = static_cast<double>(adjMix(kAdjSeed ^ adjMix(key)) >> 11) * 0x1.0p-53;
  return kAdjScale * (2.0 * u - 1.0);
}

std::uint64_t adjPair(std::int64_t p, std::int64_t q) {
  if (p < q) std::swap(p, q);
  return static_cast<std::uint64_t>(p * (p + 1) / 2 + q);
}

// The seeded factors: one per symmetric pair (p, q) of h, one per canonical
// index of (pq|rs) -- so the perturbation keeps both symmetries.
double adjFactorH(std::int64_t p, std::int64_t q) { return adjUniform(2 * adjPair(p, q)); }
double adjFactorG(std::int64_t p, std::int64_t q, std::int64_t r, std::int64_t s) {
  return adjUniform(2 * adjPair(static_cast<std::int64_t>(adjPair(p, q)),
                                static_cast<std::int64_t>(adjPair(r, s))) +
                    1);
}

// Any (pq|rs) over the whole MO range, read off the sidecar's inventory (host
// copies): the eight-fold symmetry brings every index pattern the dot's
// definitions read to one stored row, g_xP_QR[x, p, q, r] = (x p | q r), or to
// a tied row. A pattern no row holds is our bug.
class InventoryLookup {
 public:
  InventoryLookup(const nevpt2::GeneralIntegralsDevice& gen, const nevpt2::DeviceResources& res)
      : nc_(gen.ncore), na_(gen.nact) {
    for (const nevpt2::InventoryArray& a : nevpt2::kInventoryArrays)
      arrays_.emplace(a.name, nevpt2::downloadTensor(gen.*a.member, res.stream()));
    static constexpr std::string_view kSpaces = "cav";
    for (std::size_t b = 0; b < 3; ++b)
      for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t d = 0; d < 3; ++d) {
          const auto it =
              arrays_.find(std::format("g_x{}_{}{}", kSpaces[b], kSpaces[c], kSpaces[d]));
          // Four-index rows only: g_xa_vv / g_xc_vv share the pattern's name
          // but are tied (rank 3), read below.
          rows_[b][c][d] =
              it != arrays_.end() && it->second.rank() == 4 && it->second.size() > 0
                  ? &it->second
                  : nullptr;
        }
  }
  const nevpt2::Tensor& array(const std::string& name) const { return arrays_.at(name); }

  double g(std::int64_t p, std::int64_t q, std::int64_t r, std::int64_t s) const {
    const std::array<std::array<std::int64_t, 4>, 8> perms{{{p, q, r, s},
                                                            {q, p, r, s},
                                                            {p, q, s, r},
                                                            {q, p, s, r},
                                                            {r, s, p, q},
                                                            {s, r, p, q},
                                                            {r, s, q, p},
                                                            {s, r, q, p}}};
    for (const auto& [a, b, c, d] : perms)
      if (const nevpt2::Tensor* t = rows_[code(b)][code(c)][code(d)])
        return (*t)(a, local(b), local(c), local(d));
    for (const auto& [a, b, c, d] : perms) {
      const char sb = space(b), sc = space(c), sd = space(d);
      if (sb == 'c' && sc == 'c' && sd == 'c' && c == d)
        return arrays_.at("g_xc_cc_j")(a, local(b), local(c));
      if (sb == 'c' && sc == 'c' && sd == 'c' && b == c)
        return arrays_.at("g_xc_cc_k")(a, local(b), local(d));
      if (sb == 'a' && sc == 'v' && sd == 'v' && c == d)
        return arrays_.at("g_xa_vv")(a, local(b), local(c));
      if (sb == 'c' && sc == 'v' && sd == 'v' && c == d)
        return arrays_.at("g_xc_vv")(a, local(b), local(c));
    }
    nevpt2::check(false, std::format("InventoryLookup: no inventory row holds ({} {} | {} {})",
                                     p, q, r, s));
    return 0.0;
  }

 private:
  char space(std::int64_t i) const { return i < nc_ ? 'c' : i < nc_ + na_ ? 'a' : 'v'; }
  std::size_t code(std::int64_t i) const { return i < nc_ ? 0 : i < nc_ + na_ ? 1 : 2; }
  std::int64_t local(std::int64_t i) const {
    return i < nc_ ? i : i < nc_ + na_ ? i - nc_ : i - nc_ - na_;
  }
  std::int64_t nc_, na_;
  std::map<std::string, nevpt2::Tensor> arrays_;
  std::array<std::array<std::array<const nevpt2::Tensor*, 3>, 3>, 3> rows_{};
};

// The perturbation's dotted inventory, by integral_perturbation.py's
// `inventory`, shaped as `gen`'s arrays; `gamma` is the golden's dm1.
nevpt2::Result<nevpt2::GeneralIntegralsDevice> seededInventoryDot(
    const nevpt2::GeneralIntegralsDevice& gen, const nevpt2::Tensor& gamma,
    const nevpt2::DeviceResources& res) {
  const std::int64_t nc = gen.ncore, na = gen.nact, nv = gen.nvirt, n = gen.nmo();
  const std::int64_t cLo = 0, aLo = nc, vLo = nc + na;
  const InventoryLookup at(gen, res);
  const nevpt2::Tensor& h0 = at.array("h");
  // Each integral scaled by its seeded factor: an integral that is zero at
  // the state stays zero along the perturbation.
  const auto adjH = [&](std::int64_t p, std::int64_t q) { return h0(p, q) * adjFactorH(p, q); };
  const auto adjG = [&](std::int64_t p, std::int64_t q, std::int64_t r, std::int64_t s) {
    return at.g(p, q, r, s) * adjFactorG(p, q, r, s);
  };
  nevpt2::GeneralIntegralsDevice out;
  out.ncore = nc;
  out.nact = na;
  out.nvirt = nv;
  nevpt2::Tensor h({n, n}), heff({n, n}), f({n, n});
  for (std::int64_t p = 0; p < n; ++p)
    for (std::int64_t q = 0; q < n; ++q) {
      h(p, q) = adjH(p, q);
      double e = h(p, q);
      for (std::int64_t j = 0; j < nc; ++j) e += 2.0 * adjG(p, q, j, j) - adjG(p, j, j, q);
      heff(p, q) = e;
      double fpq = e;
      for (std::int64_t t = 0; t < na; ++t)
        for (std::int64_t u = 0; u < na; ++u)
          fpq += gamma(t, u) *
                 (adjG(p, q, aLo + t, aLo + u) - 0.5 * adjG(p, aLo + t, q, aLo + u));
      f(p, q) = fpq;
    }
  const auto row = [&](std::int64_t pLo, std::int64_t np, std::int64_t qLo, std::int64_t nq,
                       std::int64_t rLo, std::int64_t nr) {
    nevpt2::Tensor t({n, np, nq, nr});
    for (std::int64_t x = 0; x < n; ++x)
      for (std::int64_t i = 0; i < np; ++i)
        for (std::int64_t j = 0; j < nq; ++j)
          for (std::int64_t k = 0; k < nr; ++k) t(x, i, j, k) = adjG(x, pLo + i, qLo + j, rLo + k);
    return t;
  };
  const auto tied = [&](std::int64_t pLo, std::int64_t np, std::int64_t qLo, std::int64_t nq,
                        bool exchange) {
    nevpt2::Tensor t({n, np, nq});
    for (std::int64_t x = 0; x < n; ++x)
      for (std::int64_t i = 0; i < np; ++i)
        for (std::int64_t j = 0; j < nq; ++j)
          t(x, i, j) = exchange ? adjG(x, pLo + i, pLo + i, qLo + j)  // (x j | j i)
                                : adjG(x, pLo + i, qLo + j, qLo + j);  // (x p | q q)
    return t;
  };
  std::vector<std::pair<const char*, nevpt2::Tensor>> host;
  host.emplace_back("h", std::move(h));
  host.emplace_back("heff", std::move(heff));
  host.emplace_back("f", std::move(f));
  host.emplace_back("g_xa_aa", row(aLo, na, aLo, na, aLo, na));
  host.emplace_back("g_xv_aa", row(vLo, nv, aLo, na, aLo, na));
  host.emplace_back("g_xc_aa", row(cLo, nc, aLo, na, aLo, na));
  host.emplace_back("g_xa_va", row(aLo, na, vLo, nv, aLo, na));
  host.emplace_back("g_xv_va", row(vLo, nv, vLo, nv, aLo, na));
  host.emplace_back("g_xc_va", row(cLo, nc, vLo, nv, aLo, na));
  host.emplace_back("g_xa_ca", row(aLo, na, cLo, nc, aLo, na));
  host.emplace_back("g_xv_ca", row(vLo, nv, cLo, nc, aLo, na));
  host.emplace_back("g_xc_ca", row(cLo, nc, cLo, nc, aLo, na));
  host.emplace_back("g_xa_cv", row(aLo, na, cLo, nc, vLo, nv));
  host.emplace_back("g_xv_cv", row(vLo, nv, cLo, nc, vLo, nv));
  host.emplace_back("g_xc_cv", row(cLo, nc, cLo, nc, vLo, nv));
  host.emplace_back("g_xa_cc", row(aLo, na, cLo, nc, cLo, nc));
  host.emplace_back("g_xv_cc", row(vLo, nv, cLo, nc, cLo, nc));
  host.emplace_back("g_xc_cc_j", tied(cLo, nc, cLo, nc, /*exchange=*/false));
  host.emplace_back("g_xc_cc_k", tied(cLo, nc, cLo, nc, /*exchange=*/true));
  host.emplace_back("g_xa_vv", tied(aLo, na, vLo, nv, /*exchange=*/false));
  host.emplace_back("g_xc_vv", tied(cLo, nc, vLo, nv, /*exchange=*/false));
  for (std::size_t i = 0; i < nevpt2::kInventoryArrays.size(); ++i) {
    const nevpt2::InventoryArray& a = nevpt2::kInventoryArrays[i];
    if (std::string_view(a.name) != host[i].first || host[i].second.dims() != (gen.*a.member).dims)
      return nevpt2::err_config(std::format(
          "the integral adjoint's seeded perturbation: {} does not fit the sidecar's inventory",
          a.name));
    out.*a.member = nevpt2::uploadTensor(host[i].second, res);
  }
  return out;
}

// E_corr (SC, or the PC total with `pc`) at I + h Idot: the stored blocks and
// orbital energies displaced, the digests rebuilt at the displaced h2e.
nevpt2::Result<double> adjEnergyAt(const nevpt2::GoldenFile& g, const bool pc, const double h,
                                   const nevpt2::NevptIntegralsDevice& ints,
                                   const nevpt2::GeneralIntegralsDevice& gen,
                                   const nevpt2::IntegralPerturbationDevice& pert,
                                   const nevpt2::Tensor& h2eDot, const nevpt2::DeviceTensor& dm1,
                                   const nevpt2::DeviceTensor& dm2,
                                   const nevpt2::RdmBuildOptions& opt,
                                   const nevpt2::DeviceResources& res) {
  const nevpt2::DisplacedIntegralsDevice at =
      NEVPT2_TRY(nevpt2::displacedIntegralsDevice(ints, gen, pert, h, res));
  nevpt2::Tensor h2e = g.get("h2e");
  for (std::int64_t i = 0; i < h2e.size(); ++i) h2e.flatRef(i) += h * h2eDot.flat(i);
  const nevpt2::RdmBuildResult b =
      nevpt2::buildRdmsDevice(opt, g.get("ci"), g.ncas, g.nelecA, g.nelecB, h2e, res);
  if (!pc) return nevpt2::energiesDevice(at.ints, dm1, dm2, b.dm3, b.f3ac, b.f3ca, res).total;
  const nevpt2::PcEnergyResult e =
      NEVPT2_TRY(nevpt2::pcEnergiesDevice(at.ints, dm1, dm2, b.dm3, b.f3ac, b.f3ca, res));
  double total = 0.0;
  for (std::size_t c = 0; c < e.classes.size(); ++c) {
    if (e.classes[c].status != nevpt2::PcStatus::Done)
      return nevpt2::err_numerical(std::format(
          "the integral adjoint's ladder: PC class {} refused at h = {:.3e} ({})",
          nevpt2::CLASSES[c], h, e.classes[c].why));
    total += e.classes[c].energy;
  }
  return total;
}

// Runs checks a-c on `adj` (scOnePassSourceTerms' or pcOnePassSourceTerms'
// adjoint at this state). Returns whether all three hold; an input that does
// not fit is an Error for main.
nevpt2::Result<bool> checkIntegralAdjoint(const nevpt2::GoldenFile& g, const bool pc,
                                          const nevpt2::NevptIntegralsDevice& ints,
                                          const nevpt2::GeneralIntegralsDevice& gen,
                                          const nevpt2::IntegralAdjointDevice& adj,
                                          const nevpt2::DeviceTensor& dm1,
                                          const nevpt2::DeviceTensor& dm2,
                                          const nevpt2::RdmBuildOptions& opt,
                                          const nevpt2::DeviceResources& res) {
  const auto t0 = std::chrono::steady_clock::now();
  const nevpt2::IntegralPerturbationDevice pert = NEVPT2_TRY(nevpt2::integralPerturbationDevice(
      ints, NEVPT2_TRY(seededInventoryDot(gen, g.get("dm1"), res)), res));

  // a. the seeded dot against itself.
  const nevpt2::ScGeneralCheck self =
      NEVPT2_TRY(nevpt2::scGeneralIntegralCheck(pert.d, pert.gen, dm1, res));
  const double selfWorst = std::max({self.heff, self.fock, self.fockFull});
  const bool selfOk = selfWorst < kAdjSelfTol;
  std::printf("  9a. the seeded integral perturbation against itself (relative): h^eff's "
              "dressing %.2e, f's diagonal %.2e, f over the whole range %.2e (< %.0e: %s)\n",
              self.heff, self.fock, self.fockFull, kAdjSelfTol, selfOk ? "ok" : "FAILED");

  // b. the scatter's duality.
  const double onStored = nevpt2::integralAdjointPairing(adj.stored, pert.d, res);
  const double onInventory = nevpt2::inventoryAdjointPairing(adj.inventory, pert.gen, res);
  const double scatterGap =
      std::fabs(onStored - onInventory) / std::max(1.0, std::fabs(onInventory));
  const bool scatterOk = scatterGap <= kAdjSelfTol;
  std::printf("  9b. <Ibar, Idot> on the inventory vs on the stored blocks: %.14e vs %.14e, "
              "relative |delta| = %.2e (<= %.0e: %s)\n",
              onInventory, onStored, scatterGap, kAdjSelfTol, scatterOk ? "ok" : "FAILED");

  // c. the pairing against a ladder of the GPU's own energy.
  const nevpt2::Tensor h2eDot = nevpt2::downloadTensor(pert.d.h2e, res.stream());
  const double scale = std::max(1.0, std::fabs(onInventory));
  std::array<double, kAdjSteps> step{}, err{};
  std::printf("  9c. %s along I + h Idot, central differences against <Ibar, Idot> = %.14e:\n",
              pc ? "E_corr(PC)" : "E_corr(SC)", onInventory);
  std::printf("  %5s %12s %22s %12s %8s\n", "k", "h = 2^-k", "quotient", "|q - pair|", "ratio");
  for (std::size_t i = 0; i < kAdjSteps; ++i) {
    const int e = kAdjFirstExp + static_cast<int>(i);
    const double hh = std::ldexp(1.0, -e);
    step[i] = hh;
    const nevpt2::Result<double> plus =
        adjEnergyAt(g, pc, hh, ints, gen, pert, h2eDot, dm1, dm2, opt, res);
    const nevpt2::Result<double> minus =
        adjEnergyAt(g, pc, -hh, ints, gen, pert, h2eDot, dm1, dm2, opt, res);
    // A coarse rung may leave the energy's domain (a PC denominator turned
    // negative): it has no quotient, and an infinite error breaks any run.
    for (const nevpt2::Result<double>* r : {&plus, &minus})
      if (!*r && r->error().kind != nevpt2::ErrorKind::Numerical)
        return std::unexpected(r->error());
    if (!plus || !minus) {
      err[i] = std::numeric_limits<double>::infinity();
      std::printf("  %5d %12.3e %22s   REFUSED: %s\n", e, hh, "-",
                  (!plus ? plus : minus).error().message.c_str());
      continue;
    }
    const double q = (*plus - *minus) / (2.0 * hh);
    err[i] = std::fabs(q - onInventory) / scale;
    if (i == 0 || !std::isfinite(err[i - 1]))
      std::printf("  %5d %12.3e %22.14e %12.3e %8s\n", e, hh, q, err[i], "-");
    else
      std::printf("  %5d %12.3e %22.14e %12.3e %8.2f\n", e, hh, q, err[i],
                  err[i] > 0.0 ? err[i - 1] / err[i] : 0.0);
  }
  std::size_t bestAt = 0;
  for (std::size_t i = 1; i < kAdjSteps; ++i)
    if (err[i] < err[bestAt]) bestAt = i;
  const nevpt2::FourfoldRun run = nevpt2::longestFourfoldRun(err);
  const bool ladderOk = err[bestAt] <= kAdjTol && run.length >= kAdjRun;
  std::printf("  9c. best %.3e at h = %.3e (<= %.0e); %d consecutive fourfold halvings ending "
              "at h = %.3e (%d needed) -- %s\n",
              err[bestAt], step[bestAt], kAdjTol, run.length, step[run.end], kAdjRun,
              ladderOk ? "ok" : "FAILED");
  std::printf("  the integral adjoint's checks in %.2fs\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  return selfOk && scatterOk && ladderOk;
}
