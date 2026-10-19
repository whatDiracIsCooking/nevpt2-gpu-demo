// nevpt2.ci_checks, --gradient-source: the SC-NEVPT2 CI SOURCE TERM
// y_I = dE/dc_I (Park Eq. 48), checked against the forward tangent by
// DUALITY -- and, integral-direct only, PC-NEVPT2's (#74). Both demos run the
// SC check -- nevpt2_demo over FullBlockSource, nevpt2_df_demo over its
// DfIntegralSource slabs (#162); ci_checks.cppm has the interface.
//
// WHAT IS BEING CHECKED, and why it is exact. Forward and reverse mode are
// the same derivative computed two ways:
//
//   forward   nevpt2.energy's energiesTangentDevice, given the RDM tangents
//             along a CI direction u, returns Edot(u) = sum_I (dE/dc_I) u_I;
//   reverse   nevpt2.gradient's scCiSourceDevice returns the whole vector
//             y_I = dE/dc_I, by sweeping the Eqs. 42-43 pseudodensities down
//             to the five RDM coefficient tensors and contracting those with
//             the sigma-vector kernels.
//
// So for EVERY direction u,
//
//   <u, y> == Edot(u),
//
// WHICH FUNCTIONAL, and the sidecar (#87). e_core / e_virt are the
// generalized Fock's diagonal, built from the active 1-RDM, so they move with
// the CI vector. Given the gradient sidecar (--gradgold PATH), BOTH sides
// carry that: the reverse sweep scatters each class's R_t onto the orbital
// energies its Delta_t reads and folds dE/de into dE/d(dm1) through
// W_p = (pp|ut) - (1/2)(pu|tp), and the forward tangent reads every class's
// Delta at e_dot = <dm1_dot, W>. Without one, both hold them fixed at the
// golden's. Either way the identity is exact on the functional both sides
// differentiate -- which is exactly why it could not see the orbital
// energies missing from BOTH (docs/testing.md, "The GPU as the source-term
// provider"): that takes an independent provider, PySCF's.
//
// an algebraic identity between two independent computations -- no step size,
// no truncation error, nothing but floating-point reassociation. That is why
// this is a PERMANENT 1e-7 gate and not a finite-difference plateau: the
// difference quotient was the one-time cross-check that licensed the tangent
// itself (ci_checks_tangent.cpp, and docs/testing.md, "The SC tangent's
// finite-difference cross-check"), and this check inherits its authority
// without inheriting its noise.
//
// THREE THINGS IT RUNS.
//
//   1. SEVERAL SEEDED DIRECTIONS. Random u, projected against the state and
//      normalized -- the same construction checkScTangent uses. A single
//      direction can hide an error that happens to be orthogonal to it; four
//      independent ones at three active-space sizes do not.
//   2. THE EULER DIRECTION, u = the state itself. Free (the tangent build
//      along `ci` is the one --rdm-tangent already runs) and it probes the
//      one direction no random vector reaches, since the random ones are
//      projected against `ci` precisely to avoid it.
//   3. THE FULL JACOBIAN, where the determinant space is small enough
//      (CAS(4,4): ndet = 36). y_I against Edot(e_I) for EVERY I, one forward
//      tangent per determinant. A dot product detects an error; it does not
//      localise it -- a column-by-column comparison names the determinant,
//      and with it the half of the sweep that is wrong. It costs ndet tangent
//      builds, which is why it is bounded by kJacobianMaxDet rather than run
//      at every size.
//
// WITH --pc (#74) THE SAME THREE CHECKS HOLD PC-NEVPT2'S y. The reverse side
// is nevpt2.gradient's pcCiSourceDevice -- the PC solve's adjoint, swept
// through SC's own block adjoints to the same five coefficient tensors and
// the same sigma assembly -- and the forward side nevpt2.energy's
// pcTangentsDevice, the tangent of that solve (both with the orbital
// energies' response when the sidecar is given). A class the gap or
// denominator check refuses is refused by both and FAILS the run, named in
// the per-class table: neither side has a number for it, and none is made up.
// The PC tangent's own finite difference is nevpt2_pc_tangent's, the one-time
// license this duality inherits, as --sc-tangent's is for SC.
//
// Everything runs on the caller's one DeviceResources: every tangent build,
// every energy and every sigma launch is issued on its stream, and every
// device tensor here is freed (on that stream) when it goes out of scope.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries -- a sidecar that cannot be
// read or does not fit the golden is the caller's input.
#include "error_handling/error_macros.h"

module nevpt2.ci_checks;

import std;
import nevpt2.rdm_host_dm;

namespace nevpt2 {

// The gate. Both sides are the same algebra, so the only difference is
// floating-point reassociation (the sigma kernels walk the reverse link
// tables, the tangent contracts einsums); this is the issue's own bound and
// the tree's usual one, decades above what is measured
// (docs/testing.md, "The CI source term's forward-reverse duality").
constexpr double kCiSourceTol = 1e-7;

// The sidecar's orbital energies rebuilt from h^eff and W against dm1, held
// to the golden's e_core / e_virt (relative Frobenius): what says the sidecar
// belongs to this golden, as --kappa-full's own self-check bound has it.
constexpr double kEpsRebuildTol = 1e-10;

// How many determinants the column-by-column Jacobian is run up to. CAS(4,4)
// is 36 and CAS(8,8) is 4900; one forward tangent per determinant is minutes
// at the second, so the Jacobian is the small case's check and the dot
// products are every case's.
constexpr std::int64_t kJacobianMaxDet = 256;

// The directions the dot-product check uses, fixed so a run is reproducible.
constexpr std::array<std::uint64_t, 4> kCiSourceSeeds = {0x5eed'0035u, 0x5eed'1035u,
                                                         0x5eed'2035u, 0x5eed'3035u};

// x scaled so <x|x> = 1 (ci_checks_tangent.cpp's tangentNormalized, which
// has module linkage there and no declaration here).
nevpt2::Tensor ciSourceNormalized(nevpt2::Tensor x) {
  double norm2 = 0.0;
  for (std::int64_t i = 0; i < x.size(); ++i) norm2 += x.flat(i) * x.flat(i);
  const double inv = 1.0 / std::sqrt(norm2);
  for (std::int64_t i = 0; i < x.size(); ++i) x.flatRef(i) *= inv;
  return x;
}

// A seeded random direction, projected against `ci` and normalized -- the
// same construction checkScTangent and test/rdm_host_dm/ use. The projection
// is not needed for the identity (which holds for any direction at all); it
// keeps the direction a rotation of the state, which is the physically
// meaningful one, and leaves the state's own direction to the Euler case.
nevpt2::Tensor ciSourceDirection(const nevpt2::Tensor& ci, std::uint64_t seed) {
  nevpt2::Tensor u(ci.dims());
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::int64_t i = 0; i < u.size(); ++i) u.flatRef(i) = dist(gen);
  double overlap = 0.0;
  for (std::int64_t i = 0; i < u.size(); ++i) overlap += ci.flat(i) * u.flat(i);
  for (std::int64_t i = 0; i < u.size(); ++i) u.flatRef(i) -= overlap * ci.flat(i);
  return ciSourceNormalized(std::move(u));
}

// The unit direction along determinant `I`: one Jacobian column's input.
nevpt2::Tensor ciSourceUnit(const std::vector<std::int64_t>& dims, std::int64_t index) {
  nevpt2::Tensor e(dims);
  e.flatRef(index) = 1.0;
  return e;
}

// <x, y> over two same-shaped vectors.
double ciSourceDot(const nevpt2::Tensor& x, const nevpt2::Tensor& y) {
  double s = 0.0;
  for (std::int64_t i = 0; i < x.size(); ++i) s += x.flat(i) * y.flat(i);
  return s;
}

// Edot(u): the forward tangent of the SC total along `u`, the whole way from
// the direction -- dm1/dm2 tangents on the host (exact, both forms being
// quadratic), dm3/f3ac/f3ca tangents on the device, then the eight classes'
// tangent. Every device tensor it allocates is freed when it returns.
//
// `eps` non-null: the orbital energies move too, e_dot = <dm1_dot, W> from
// the same host dm1 tangent the RDM side reads.
double ciSourceTangentAlong(const nevpt2::HostDm12Builder& builder, const nevpt2::Tensor& ci,
                            const nevpt2::Tensor& u, const nevpt2::RdmBuildOptions& rdmOpt,
                            const CiCheckIntegrals& ints,
                            const nevpt2::DeviceTensor& dDm1, const nevpt2::DeviceTensor& dDm2,
                            const nevpt2::RdmBuildResult& rdm,
                            const nevpt2::OrbitalEnergyResponse* eps,
                            const nevpt2::DeviceResources& res) {
  const nevpt2::HostDm12 dot = builder.dm12Dot(ci, u);
  const nevpt2::DeviceTensor dDm1Dot = nevpt2::uploadTensor(dot.dm1, res);
  const nevpt2::DeviceTensor dDm2Dot = nevpt2::uploadTensor(dot.dm2, res);
  const nevpt2::RdmBuildResult rdmDot = nevpt2::buildRdmTangentsDevice(
      rdmOpt, ci, u, builder.norb(), builder.nelecA(), builder.nelecB(), ints.h2e, res);
  const std::optional<nevpt2::OrbitalEnergyDot> epsDot =
      eps ? std::optional(nevpt2::orbitalEnergyDot(*eps, dot.dm1)) : std::nullopt;
  return nevpt2::energiesTangentDevice(ints.active, ints.source, ints.batch, dDm1, dDm2,
                                       rdm.dm3, rdm.f3ac, rdm.f3ca, dDm1Dot, dDm2Dot,
                                       rdmDot.dm3, rdmDot.f3ac, rdmDot.f3ca,
                                       epsDot ? &*epsDot : nullptr, res)
      .totalDot;
}

// || a - b || / || b || over the core and virtual halves together.
double relativeEpsGap(const nevpt2::OrbitalEnergyDot& a, const nevpt2::Tensor& eCore,
                      const nevpt2::Tensor& eVirt) {
  double num = 0.0, den = 0.0;
  for (std::int64_t i = 0; i < eCore.size(); ++i) {
    num += (a.core.flat(i) - eCore.flat(i)) * (a.core.flat(i) - eCore.flat(i));
    den += eCore.flat(i) * eCore.flat(i);
  }
  for (std::int64_t r = 0; r < eVirt.size(); ++r) {
    num += (a.virt.flat(r) - eVirt.flat(r)) * (a.virt.flat(r) - eVirt.flat(r));
    den += eVirt.flat(r) * eVirt.flat(r);
  }
  return std::sqrt(num / std::max(den, 1e-300));
}

// The orbital energies' response W out of the gradient sidecar at `path`,
// after checking the sidecar describes `g`: its partition, its four arrays,
// and e_core / e_virt rebuilt from its h^eff and W against the golden's dm1.
// Every failure is the caller's input, an Error reported once at main.
nevpt2::Result<nevpt2::OrbitalEnergyResponse> loadOrbitalEnergyResponse(
    const nevpt2::GoldenFile& g, const std::string& path) {
  std::printf("  gradient sidecar: %s\n", path.c_str());
  const nevpt2::GradGoldFile gg = NEVPT2_TRY(nevpt2::loadGradGold(path));
  NEVPT2_TRY(gg.checkAgainst(g));
  NEVPT2_TRY(gg.require({"heff", "g_xa_cc", "g_xc_ca", "g_xa_vv", "g_xv_va"}));
  nevpt2::OrbitalEnergyResponse w =
      nevpt2::orbitalEnergyResponse(gg.ncore, gg.nact, gg.nvirt, gg.get("g_xa_cc"),
                                    gg.get("g_xc_ca"), gg.get("g_xa_vv"), gg.get("g_xv_va"));
  const double gap = relativeEpsGap(nevpt2::orbitalEnergiesFrom(w, gg.get("heff"), g.get("dm1")),
                                    g.get("e_core"), g.get("e_virt"));
  std::printf("  e_core / e_virt rebuilt as h^eff_pp + <dm1, W_p>, against the golden's: "
              "%.2e relative (< %.0e)\n",
              gap, kEpsRebuildTol);
  if (!(gap < kEpsRebuildTol))
    return nevpt2::err_config(std::format(
        "{}: its h^eff and orbital-energy integrals rebuild the golden's e_core / e_virt only "
        "to {:.3e} relative (need {:.0e}): this sidecar does not belong to the golden",
        path, gap, kEpsRebuildTol));
  return w;
}

// Runs the three checks above for one golden state, prints every number, and
// returns whether all of them are within kCiSourceTol. An active space the
// host RDM builder refuses is a configuration Error (the caller reports it
// once); everything else here is a number.
//
// `gradgoldPath` empty: the orbital energies are held fixed on both sides.
// Otherwise both sides carry their response, read from that sidecar.
nevpt2::Result<bool> checkCiSource(const nevpt2::GoldenFile& g,
                                   const nevpt2::RdmBuildOptions& rdmOpt,
                                   const CiCheckIntegrals& ints,
                                   const nevpt2::RdmBuildResult& baseRdm,
                                   const nevpt2::EnergyResult& energies,
                                   const std::string& gradgoldPath,
                                   std::string_view fixedReason,
                                   const nevpt2::DeviceResources& res) {
  nevpt2::Result<nevpt2::HostDm12Builder> builder =
      nevpt2::HostDm12Builder::create(g.ncas, g.nelecA, g.nelecB);
  if (!builder) return std::unexpected(builder.error());
  const nevpt2::Tensor& ci = g.get("ci");

  std::printf("\n--gradient-source: y_I = dE/dc_I (Park Eq. 48) against the forward tangent\n");
  std::optional<nevpt2::OrbitalEnergyResponse> epsResponse;
  if (!gradgoldPath.empty())
    epsResponse = NEVPT2_TRY(loadOrbitalEnergyResponse(g, gradgoldPath));
  const nevpt2::OrbitalEnergyResponse* eps = epsResponse ? &*epsResponse : nullptr;
  const std::string functional =
      eps ? std::string("the orbital energies moving with the active density")
          : std::format("the orbital energies held fixed ({})", fixedReason);
  std::printf("  functional: %s\n", functional.c_str());

  // The host rebuild IS the dm1/dm2 both sides read, so the whole check is
  // internally consistent at one state; the unit tier is what holds that
  // rebuild to the golden's own arrays.
  const nevpt2::HostDm12 base = builder->dm12(ci);
  const nevpt2::DeviceTensor dDm1 = nevpt2::uploadTensor(base.dm1, res);
  const nevpt2::DeviceTensor dDm2 = nevpt2::uploadTensor(base.dm2, res);
  // dm3/f3ac/f3ca at the base point are the caller's own build on this very CI
  // vector with these very options.
  const nevpt2::RdmBuildResult& rdm = baseRdm;

  // The reverse side, once: y is the whole vector, so every direction below
  // is a dot product against it.
  nevpt2::SigmaVectorOptions sigmaOpt;
  sigmaOpt.nTiles = rdmOpt.nTiles;
  const auto t0 = std::chrono::steady_clock::now();
  const nevpt2::ScCiSourceResult src = nevpt2::scCiSourceDevice(
      ints.active, ints.source, ints.batch, ci, g.nelecA, g.nelecB, dDm1, dDm2, rdm.dm3,
      rdm.f3ac, rdm.f3ca, sigmaOpt, eps, res);
  const double ySeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  double yNorm = 0.0;
  for (std::int64_t i = 0; i < src.y.size(); ++i) yNorm += src.y.flat(i) * src.y.flat(i);
  yNorm = std::sqrt(yNorm);
  std::printf("  y built in %.2fs (sigma tiles=%lld, ndet=%lld); |y| = %.8f\n", ySeconds,
              static_cast<long long>(sigmaOpt.nTiles),
              static_cast<long long>(src.y.size()), yNorm);
  if (src.orbitalEnergiesMove) {
    double dEdeNorm = 0.0;
    for (const double v : src.dEdCore) dEdeNorm += v * v;
    for (const double v : src.dEdVirt) dEdeNorm += v * v;
    std::printf("  dE/de over %zu core + %zu virtual orbitals, folded onto dm1 through W: "
                "|dE/de| = %.8f\n",
                src.dEdCore.size(), src.dEdVirt.size(), std::sqrt(dEdeNorm));
  }

  bool ok = true;

  // The pseudodensity walk y is built on is the one apps/sc_pseudodensity/
  // gates; its class energies must still be nevpt2.energy's, which is the
  // cheapest possible check that the sweep did not disturb the walk.
  {
    const double d = std::fabs(src.pseudodensities.total - energies.total);
    ok = ok && d < kCiSourceTol;
    std::printf("  the walk y rides on: E_corr from the amplitudes %.10f vs energiesDevice "
                "%.10f  |delta| = %.2e\n",
                src.pseudodensities.total, energies.total, d);
  }

  std::printf("\n  <u, y> against Edot(u), for several seeded directions and for the state\n");
  std::printf("  %-18s %18s %18s %11s %11s\n", "direction", "<u, y>", "Edot(u)", "|delta|",
              "relative");
  double worst = 0.0;
  const auto compare = [&](const std::string& what, const nevpt2::Tensor& u) {
    const double dotY = ciSourceDot(u, src.y);
    const double edot = ciSourceTangentAlong(*builder, ci, u, rdmOpt, ints, dDm1, dDm2,
                                             rdm, eps, res);
    const double d = std::fabs(dotY - edot);
    worst = std::max(worst, d);
    ok = ok && d < kCiSourceTol;
    std::printf("  %-18s %18.12f %18.12f %11.2e %11.2e\n", what.c_str(), dotY, edot, d,
                d / std::max(1e-30, std::fabs(edot)));
  };
  for (const std::uint64_t seed : kCiSourceSeeds)
    compare(std::format("seed 0x{:x}", seed), ciSourceDirection(ci, seed));
  // Euler: the one direction the projected random ones never reach.
  compare("the state itself", ci);

  // The full Jacobian, where it is affordable: every column of dE/dc against
  // forward mode, so a disagreement names its determinant.
  const std::int64_t ndet = src.y.size();
  if (ndet <= kJacobianMaxDet) {
    std::printf("\n  the FULL Jacobian: y_I against Edot(e_I) for all %lld determinants\n",
                static_cast<long long>(ndet));
    double jWorst = 0.0;
    std::int64_t jAt = 0;
    double yMax = 0.0;
    for (std::int64_t I = 0; I < ndet; ++I) {
      const nevpt2::Tensor e = ciSourceUnit(src.y.dims(), I);
      const double edot =
          ciSourceTangentAlong(*builder, ci, e, rdmOpt, ints, dDm1, dDm2, rdm, eps, res);
      const double d = std::fabs(src.y.flat(I) - edot);
      yMax = std::max(yMax, std::fabs(src.y.flat(I)));
      if (d > jWorst) {
        jWorst = d;
        jAt = I;
      }
      ok = ok && d < kCiSourceTol;
    }
    worst = std::max(worst, jWorst);
    std::printf("  worst column: |y_I - Edot(e_I)| = %.3e at I = %lld "
                "(max |y_I| over the vector = %.8f)\n",
                jWorst, static_cast<long long>(jAt), yMax);
  } else {
    std::printf("\n  the full Jacobian is skipped: ndet = %lld is above %lld, and it costs one "
                "forward tangent per determinant\n",
                static_cast<long long>(ndet), static_cast<long long>(kJacobianMaxDet));
  }

  std::printf("\n%s: the CI source term and the forward tangent agree to %.0e "
              "(worst |<u,y> - Edot(u)| = %.3e), %s\n",
              ok ? "PASS" : "FAIL", kCiSourceTol, worst, functional.c_str());
  return ok;
}

// --- PC-NEVPT2 (#74) -------------------------------------------------------------

// Edot_pc(u): pcTangentsDevice along `u` -- the same RDM tangents as
// ciSourceTangentAlong -- summed over the classes. A class the PC solve
// refuses comes back as nullopt: there is no derivative to compare.
nevpt2::Result<std::optional<double>> pcCiSourceTangentAlong(
    const nevpt2::HostDm12Builder& builder, const nevpt2::Tensor& ci, const nevpt2::Tensor& u,
    const nevpt2::RdmBuildOptions& rdmOpt, const nevpt2::Tensor& h2e,
    const nevpt2::NevptIntegralsDevice& ints, const nevpt2::DeviceTensor& dDm1,
    const nevpt2::DeviceTensor& dDm2, const nevpt2::RdmBuildResult& rdm,
    const nevpt2::OrbitalEnergyResponse* eps, const nevpt2::DeviceResources& res) {
  const nevpt2::HostDm12 dot = builder.dm12Dot(ci, u);
  const nevpt2::DeviceTensor dDm1Dot = nevpt2::uploadTensor(dot.dm1, res);
  const nevpt2::DeviceTensor dDm2Dot = nevpt2::uploadTensor(dot.dm2, res);
  const nevpt2::RdmBuildResult rdmDot = nevpt2::buildRdmTangentsDevice(
      rdmOpt, ci, u, builder.norb(), builder.nelecA(), builder.nelecB(), h2e, res);
  const std::optional<nevpt2::OrbitalEnergyDot> epsDot =
      eps ? std::optional(nevpt2::orbitalEnergyDot(*eps, dot.dm1)) : std::nullopt;
  const nevpt2::PcTangentResult t = NEVPT2_TRY(nevpt2::pcTangentsDevice(
      ints, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, dDm1Dot, dDm2Dot, rdmDot.dm3, rdmDot.f3ac,
      rdmDot.f3ca, epsDot ? &*epsDot : nullptr, res));
  double total = 0.0;
  for (const nevpt2::PcClassTangent& c : t.classes) {
    if (c.result.status != nevpt2::PcStatus::Done) return std::optional<double>{};
    total += c.edot;
  }
  return std::optional<double>{total};
}

// --gradient-source --pc: checkCiSource's three checks for PC-NEVPT2's y,
// against the golden's block2 PC energy (whose total the adjoint's own walk
// must reproduce, 1e-7) and the PC forward tangent. Same tolerance, same
// directions, same Jacobian bound, same functional switch.
nevpt2::Result<bool> checkPcCiSource(const nevpt2::GoldenFile& g,
                                     const nevpt2::RdmBuildOptions& rdmOpt,
                                     const nevpt2::NevptIntegralsDevice& ints,
                                     const nevpt2::RdmBuildResult& baseRdm,
                                     const std::string& gradgoldPath,
                                     const nevpt2::DeviceResources& res) {
  nevpt2::Result<nevpt2::HostDm12Builder> builder =
      nevpt2::HostDm12Builder::create(g.ncas, g.nelecA, g.nelecB);
  if (!builder) return std::unexpected(builder.error());
  const nevpt2::Tensor& ci = g.get("ci");
  const nevpt2::Tensor& h2e = g.get("h2e");

  std::printf("\n--gradient-source --pc: the PC-NEVPT2 y_I = dE_pc/dc_I against the PC forward "
              "tangent (#74)\n");
  std::optional<nevpt2::OrbitalEnergyResponse> epsResponse;
  if (!gradgoldPath.empty())
    epsResponse = NEVPT2_TRY(loadOrbitalEnergyResponse(g, gradgoldPath));
  const nevpt2::OrbitalEnergyResponse* eps = epsResponse ? &*epsResponse : nullptr;
  const char* const functional =
      eps ? "the orbital energies moving with the active density"
          : "the orbital energies held fixed (no --gradgold)";
  std::printf("  functional: %s\n", functional);

  const nevpt2::HostDm12 base = builder->dm12(ci);
  const nevpt2::DeviceTensor dDm1 = nevpt2::uploadTensor(base.dm1, res);
  const nevpt2::DeviceTensor dDm2 = nevpt2::uploadTensor(base.dm2, res);
  const nevpt2::RdmBuildResult& rdm = baseRdm;

  nevpt2::SigmaVectorOptions sigmaOpt;
  sigmaOpt.nTiles = rdmOpt.nTiles;
  const auto t0 = std::chrono::steady_clock::now();
  const nevpt2::PcCiSourceResult src = NEVPT2_TRY(nevpt2::pcCiSourceDevice(
      ints, ci, g.nelecA, g.nelecB, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, sigmaOpt, eps, res));
  const double ySeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  the PC classes as the adjoint's walk solved them:\n");
  nevpt2::printPcGradientClasses(src.walk);
  if (!src.ok) {
    std::printf("\nFAIL: the PC CI source term is REFUSED on this state -- %s -- so there is "
                "no derivative to hold against the forward tangent (which refuses the same "
                "class)\n",
                src.walk.refusals().c_str());
    return false;
  }
  double yNorm = 0.0;
  for (std::int64_t i = 0; i < src.y.size(); ++i) yNorm += src.y.flat(i) * src.y.flat(i);
  std::printf("  y built in %.2fs (sigma tiles=%lld, ndet=%lld); |y| = %.8f\n", ySeconds,
              static_cast<long long>(sigmaOpt.nTiles), static_cast<long long>(src.y.size()),
              std::sqrt(yNorm));

  bool ok = true;
  {
    // The walk's PC total against block2's: the state this y differentiates.
    const double ref = g.get("e_pc_total").flat(0);
    const double d = std::fabs(src.walk.total - ref);
    ok = ok && d < kCiSourceTol;
    std::printf("  the walk y rides on: E_corr(PC) %.10f vs the golden's block2 %.10f  "
                "|delta| = %.2e\n",
                src.walk.total, ref, d);
  }

  std::printf("\n  <u, y> against Edot_pc(u), for several seeded directions and for the state\n");
  std::printf("  %-18s %18s %18s %11s %11s\n", "direction", "<u, y>", "Edot(u)", "|delta|",
              "relative");
  double worst = 0.0;
  bool refused = false;
  const auto compare = [&](const std::string& what, const nevpt2::Tensor& u) -> nevpt2::Status {
    const double dotY = ciSourceDot(u, src.y);
    const std::optional<double> edot = NEVPT2_TRY(pcCiSourceTangentAlong(
        *builder, ci, u, rdmOpt, h2e, ints, dDm1, dDm2, rdm, eps, res));
    if (!edot) {
      refused = true;
      ok = false;
      std::printf("  %-18s %18.12f %18s\n", what.c_str(), dotY, "REFUSED");
      return {};
    }
    const double d = std::fabs(dotY - *edot);
    worst = std::max(worst, d);
    ok = ok && d < kCiSourceTol;
    std::printf("  %-18s %18.12f %18.12f %11.2e %11.2e\n", what.c_str(), dotY, *edot, d,
                d / std::max(1e-30, std::fabs(*edot)));
    return {};
  };
  for (const std::uint64_t seed : kCiSourceSeeds)
    NEVPT2_TRY(compare(std::format("seed 0x{:x}", seed), ciSourceDirection(ci, seed)));
  NEVPT2_TRY(compare("the state itself", ci));

  const std::int64_t ndet = src.y.size();
  if (ndet <= kJacobianMaxDet) {
    std::printf("\n  the FULL Jacobian: y_I against Edot_pc(e_I) for all %lld determinants\n",
                static_cast<long long>(ndet));
    double jWorst = 0.0, yMax = 0.0;
    std::int64_t jAt = 0;
    for (std::int64_t I = 0; I < ndet; ++I) {
      const nevpt2::Tensor e = ciSourceUnit(src.y.dims(), I);
      const std::optional<double> edot = NEVPT2_TRY(pcCiSourceTangentAlong(
          *builder, ci, e, rdmOpt, h2e, ints, dDm1, dDm2, rdm, eps, res));
      if (!edot) {
        refused = true;
        ok = false;
        continue;
      }
      const double d = std::fabs(src.y.flat(I) - *edot);
      yMax = std::max(yMax, std::fabs(src.y.flat(I)));
      if (d > jWorst) {
        jWorst = d;
        jAt = I;
      }
      ok = ok && d < kCiSourceTol;
    }
    worst = std::max(worst, jWorst);
    std::printf("  worst column: |y_I - Edot(e_I)| = %.3e at I = %lld "
                "(max |y_I| over the vector = %.8f)\n",
                jWorst, static_cast<long long>(jAt), yMax);
  } else {
    std::printf("\n  the full Jacobian is skipped: ndet = %lld is above %lld, and it costs one "
                "forward tangent per determinant\n",
                static_cast<long long>(ndet), static_cast<long long>(kJacobianMaxDet));
  }
  if (refused)
    std::printf("  a forward tangent was REFUSED by the PC solve: no number was compared there\n");

  std::printf("\n%s: the PC CI source term and the PC forward tangent agree to %.0e "
              "(worst |<u,y> - Edot_pc(u)| = %.3e), %s\n",
              ok ? "PASS" : "FAIL", kCiSourceTol, worst, functional);
  return ok;
}

}  // namespace nevpt2
