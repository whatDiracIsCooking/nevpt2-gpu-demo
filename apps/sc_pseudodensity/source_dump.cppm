// nevpt2.app.sc_pseudodensity:source_dump -- --dump-source-terms PATH: the
// GPU's own SC-NEVPT2 gradient source terms, written as a `.gradsrc` for
// reference_data/nevpt2_grad.py --source dump to build a nuclear gradient
// from (#73). An interface partition that exports nothing, as :kappa_check
// and :kappa_full_check are.
//
// WHAT IS WRITTEN. The two derivatives nevpt2.gradient already computes and
// the other modes of this binary and nevpt2_demo already gate, assembled whole:
//
//   Y_xy  over EVERY pair of MOs -- scFullRotationMatrix, one basis direction
//         per pair, out of the gradient sidecar (as --kappa-full reads it).
//         Its forward/reverse duality, the sidecar's promotion table and the
//         two intra-space finite differences are --kappa-full's checks
//         (docs/testing.md, "The full orbital source term");
//   y_I   over every determinant -- scCiSourceDevice, the reverse sweep whose
//         duality against the forward tangent is nevpt2_demo
//         --gradient-source's check (docs/testing.md, "The CI source term's
//         forward-reverse duality");
//   Edot  along seeded directions (#88) -- :kappa_full_check's
//         edotDirections, --kappa-full's four full ones plus one purely
//         active-virtual and one purely core-active -- in its two terms: the
//         frozen-digest tangent (scFullRotationDevice) and the f3ac/f3ca
//         digests' own (scDigestRotationTangentDevice). That is what
//         reference_data/nevpt2_grad.py --edot-check holds against PySCF's
//         exact directional derivative, the independent provider for the
//         term no pair of modes here can check.
//
// AND WHAT THEY ARE NOT. Both are derivatives of the functional this tree
// differentiates, which holds one thing fixed that the exact energy does not
// (docs/gradients.md, sections 1 and 6): Y holds the f3ac/f3ca digests fixed
// (the Edot directions carry that term beside it, forward mode only, #88).
// y is the complete functional's: the orbital energies e_core/e_virt move
// with the active density, read from the same sidecar Y is (#87). What the
// remaining omission costs a nuclear gradient is measured, against PySCF's
// exact source terms, in docs/testing.md, "The GPU as the source-term
// provider".
//
// WHAT IS CHECKED HERE, since the derivatives themselves are checked
// elsewhere: that what is written is what was computed, and that the matrix
// is the pairing.
//
//   1  the probed matrix against the direct pairing, on one seeded FULL
//      antisymmetric direction: sum_pq kappa_pq Y_pq vs <kappa, Y> from one
//      scFullRotationDevice call -- the same functional summed in a different
//      order, so reassociation noise;
//   2  y's walk: its class-energy total against energiesDevice's, the cheapest
//      check that the sweep ran on the state this binary loaded -- and the
//      sidecar's orbital-energy response W, which y now reads, rebuilding
//      the golden's e_core / e_virt as h^eff_pp + <dm1, W_p>;
//   3  the file read back (readGradSrc) and compared with what was written,
//      exactly: the doubles are stored raw, so any difference is a bug;
//   4  each direction's frozen-digest tangent against sum_pq kappa_pq Y_pq off
//      the matrix being written -- the duality again, across the file's own
//      members, so the edot_frozen the reader adds the digests' term to is
//      the derivative the dumped Y pairs to.
//
// Everything runs on the caller's one DeviceResources and its one stream.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries. A sidecar that does not fit
// the golden, or a file that cannot be written, is the caller's input and
// propagates unchanged to main, which reports it once.
#include "error_handling/error_macros.h"

export module nevpt2.app.sc_pseudodensity:source_dump;

import std;
import nevpt2.gradient;        // scFullRotationMatrix, scCiSourceDevice
import nevpt2.kappa_rotation;  // antisymmetricKappa: check 1's direction
import nevpt2.golden;
import nevpt2.gradsrc;         // the .gradsrc writer and its reader
// requireGradientArrays / uploadGeneral: the sidecar as --kappa-full reads it.
import :kappa_full_check;

// How many basis directions share one slab walk while Y is probed -- the
// number --kappa-full's Eqs. 56-57 probe uses.
constexpr std::int64_t kDumpChunk = 16;
// Check 1's tolerance, relative to max(1, |<kappa, Y>|): reassociation over
// n_mo^2 terms, the bound --kappa-full holds its inactive-block matrix to.
constexpr double kDumpMatrixTol = 1e-9;
// Check 2's: the tree's usual 1e-7.
constexpr double kDumpWalkTol = 1e-7;
// Check 2's orbital-energy rebuild, relative Frobenius: --kappa-full's own
// self-check bound, which its `eps` row (the same diagonal) is held to.
constexpr double kDumpEpsTol = 1e-10;
// Check 1's direction, distinct from every seed --kappa-full and --kappa-act use.
constexpr std::uint64_t kDumpSeed = 0x5eed'0073u;
// Check 4's: the duality's own 1e-7 (exact arithmetic), across the probed
// matrix and the direct tangent -- reassociation over n_mo^2 terms on top.
constexpr double kDumpEdotTol = 1e-7;

// Every element of `a` and `b` identical, bit for bit, and the shapes equal.
bool sameTensor(const nevpt2::Tensor& a, const nevpt2::Tensor& b) {
  if (a.dims() != b.dims()) return false;
  for (std::int64_t i = 0; i < a.size(); ++i)
    if (std::bit_cast<std::uint64_t>(a.flat(i)) != std::bit_cast<std::uint64_t>(b.flat(i)))
      return false;
  return true;
}

double frobenius(const nevpt2::Tensor& t) {
  double s = 0.0;
  for (std::int64_t i = 0; i < t.size(); ++i) s += t.flat(i) * t.flat(i);
  return std::sqrt(s);
}

// The GPU's two source terms for the golden's state, as --dump-source-terms
// writes them and --source-oracle (:source_oracle) compares them: Y over every
// MO pair, and y over every determinant with the orbital energies' response
// (#87) out of the sidecar -- plus the two cheap checks that they were taken
// on this state (check 2 above).
struct GpuSourceTerms {
  nevpt2::Tensor kappa;   // Y_xy, [n_mo, n_mo], in the golden's basis
  nevpt2::Tensor ci;      // y_I, [n_det]
  double walkGap = 0.0;   // |E_corr from y's walk - energiesDevice's|
  double epsGap = 0.0;    // e_core / e_virt rebuilt from the sidecar, relative
  bool walkOk = false;
  bool epsOk = false;
};

// Assembles GpuSourceTerms. `gen` is the sidecar's uploaded general integrals
// (uploadGeneral), which the caller keeps for whatever else it probes.
nevpt2::Result<GpuSourceTerms> assembleSourceTerms(
    const nevpt2::GoldenFile& g, const nevpt2::GradGoldFile& gg,
    const nevpt2::EnergyResult& energies, const nevpt2::NevptIntegralsDevice& ints,
    const nevpt2::GeneralIntegralsDevice& gen, const nevpt2::DeviceTensor& dm1,
    const nevpt2::DeviceTensor& dm2, const nevpt2::DeviceTensor& dm3,
    const nevpt2::DeviceTensor& f3ac, const nevpt2::DeviceTensor& f3ca, std::int64_t sigmaTiles,
    const nevpt2::DeviceResources& res) {
  const std::int64_t nmo = gg.nmo;
  GpuSourceTerms out;

  // --- Y over every pair ------------------------------------------------------
  std::vector<std::pair<std::int64_t, std::int64_t>> pairs;
  for (std::int64_t p = 0; p < nmo; ++p)
    for (std::int64_t q = p + 1; q < nmo; ++q) pairs.emplace_back(p, q);
  const auto tY = std::chrono::steady_clock::now();
  out.kappa = NEVPT2_TRY(nevpt2::scFullRotationMatrix(ints, gen, dm1, dm2, dm3, f3ac, f3ca,
                                                      pairs, kDumpChunk, res));
  double yMax = 0.0;
  for (std::int64_t i = 0; i < out.kappa.size(); ++i)
    yMax = std::max(yMax, std::fabs(out.kappa.flat(i)));
  std::printf("  Y: %zu basis directions in chunks of %lld, %.2fs; max|Y| = %.6e, |Y| = %.6e\n",
              pairs.size(), static_cast<long long>(kDumpChunk),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - tY).count(),
              yMax, frobenius(out.kappa));

  // --- y over every determinant ----------------------------------------------
  const nevpt2::Tensor& ci = g.get("ci");
  nevpt2::SigmaVectorOptions sigmaOpt;
  sigmaOpt.nTiles = sigmaTiles;
  // The orbital energies' response to the active density, out of the same
  // sidecar: y is the complete functional's derivative (#87).
  const nevpt2::OrbitalEnergyResponse eps =
      nevpt2::orbitalEnergyResponse(gg.ncore, gg.nact, gg.nvirt, gg.get("g_xa_cc"),
                                    gg.get("g_xc_ca"), gg.get("g_xa_vv"), gg.get("g_xv_va"));
  const nevpt2::OrbitalEnergyDot rebuilt =
      nevpt2::orbitalEnergiesFrom(eps, gg.get("heff"), g.get("dm1"));
  double epsNum = 0.0, epsDen = 0.0;
  for (const auto& [mine, gold] :
       {std::pair{&rebuilt.core, &g.get("e_core")}, std::pair{&rebuilt.virt, &g.get("e_virt")}})
    for (std::int64_t p = 0; p < gold->size(); ++p) {
      epsNum += (mine->flat(p) - gold->flat(p)) * (mine->flat(p) - gold->flat(p));
      epsDen += gold->flat(p) * gold->flat(p);
    }
  out.epsGap = std::sqrt(epsNum / std::max(epsDen, 1e-300));
  out.epsOk = out.epsGap < kDumpEpsTol;
  const auto tCi = std::chrono::steady_clock::now();
  nevpt2::ScCiSourceResult ciSrc = nevpt2::scCiSourceDevice(
      ints, ci, g.nelecA, g.nelecB, dm1, dm2, dm3, f3ac, f3ca, sigmaOpt, &eps, res);
  double cy = 0.0;
  for (std::int64_t i = 0; i < ci.size(); ++i) cy += ci.flat(i) * ciSrc.y.flat(i);
  std::printf("  y: %lld determinants, %.2fs; |y| = %.10f, <c, y> = %.10f\n",
              static_cast<long long>(ciSrc.y.size()),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - tCi).count(),
              frobenius(ciSrc.y), cy);

  // The walk y rode on, against the energy stage.
  out.walkGap = std::fabs(ciSrc.pseudodensities.total - energies.total);
  out.walkOk = out.walkGap < kDumpWalkTol;
  std::printf("  E_corr from y's walk %.10f vs energiesDevice %.10f, |delta| = %.2e "
              "(< %.0e: %s)\n",
              ciSrc.pseudodensities.total, energies.total, out.walkGap, kDumpWalkTol,
              out.walkOk ? "ok" : "FAILED");
  std::printf("  y carries the orbital energies' response: e_core / e_virt rebuilt as "
              "h^eff_pp + <dm1, W_p> vs the golden's, %.2e relative (< %.0e: %s)\n",
              out.epsGap, kDumpEpsTol, out.epsOk ? "ok" : "FAILED");
  out.ci = std::move(ciSrc.y);
  return out;
}

// Assembles Y and y for the golden's state, writes them to `path`, reads the
// file back, and returns whether the three checks above hold. A sidecar that
// does not describe this golden, or a file that cannot be written or read, is
// an Error the caller reports once.
nevpt2::Result<bool> dumpSourceTerms(const nevpt2::GoldenFile& g, const nevpt2::GradGoldFile& gg,
                                     const nevpt2::EnergyResult& energies,
                                     const nevpt2::NevptIntegralsDevice& ints,
                                     const nevpt2::DeviceTensor& dm1,
                                     const nevpt2::DeviceTensor& dm2,
                                     const nevpt2::DeviceTensor& dm3,
                                     const nevpt2::DeviceTensor& f3ac,
                                     const nevpt2::DeviceTensor& f3ca,
                                     const nevpt2::RdmBuildOptions& rdmOpt,
                                     const std::string& path,
                                     const nevpt2::DeviceResources& res) {
  NEVPT2_TRY(gg.checkAgainst(g));
  NEVPT2_TRY(requireGradientArrays(gg));
  const std::int64_t nmo = gg.nmo;
  std::printf("\n--dump-source-terms: the GPU's Y_xy and y_I (Park Eqs. 26-27) -> %s\n",
              path.c_str());
  std::printf("  partition: n_mo = %lld = %lld core + %lld active + %lld virtual\n",
              static_cast<long long>(nmo), static_cast<long long>(gg.ncore),
              static_cast<long long>(gg.nact), static_cast<long long>(gg.nvirt));
  const nevpt2::GeneralIntegralsDevice gen = uploadGeneral(gg, res);
  const GpuSourceTerms terms = NEVPT2_TRY(assembleSourceTerms(
      g, gg, energies, ints, gen, dm1, dm2, dm3, f3ac, f3ca, rdmOpt.nTiles, res));
  const nevpt2::Tensor& y = terms.kappa;

  // 1. the matrix against the direct pairing on one seeded full direction.
  const nevpt2::Tensor k = nevpt2::antisymmetricKappa(nmo, kDumpSeed);
  const std::vector<nevpt2::ScRotationRequest> req{
      nevpt2::ScRotationRequest{k, /*tangent=*/false}};
  const nevpt2::ScFullRotationResult direct =
      NEVPT2_TRY(nevpt2::scFullRotationDevice(ints, gen, dm1, dm2, dm3, f3ac, f3ca, req, res));
  double fromMatrix = 0.0;
  for (std::int64_t p = 0; p < nmo; ++p)
    for (std::int64_t q = 0; q < nmo; ++q) fromMatrix += k(p, q) * y(p, q);
  const double matrixGap = std::fabs(fromMatrix - direct.dirs[0].adjoint);
  const double matrixTol = kDumpMatrixTol * std::max(1.0, std::fabs(direct.dirs[0].adjoint));
  const bool matrixOk = matrixGap <= matrixTol;
  std::printf("  1. sum kappa.Y from the matrix vs <kappa, Y> direct, one seeded full "
              "direction:\n     %.14e vs %.14e, |delta| = %.2e (<= %.1e: %s)\n",
              fromMatrix, direct.dirs[0].adjoint, matrixGap, matrixTol,
              matrixOk ? "ok" : "FAILED");

  // 2. the walk y rode on, and the orbital energies it reads -- printed by
  // assembleSourceTerms above.
  const bool walkOk = terms.walkOk;
  const bool epsOk = terms.epsOk;

  // --- Edot along the seeded directions, in its two terms (#88) -------------
  const std::vector<nevpt2::Tensor> kappas = edotDirections(gg.ncore, gg.nact, gg.nvirt);
  std::vector<nevpt2::ScRotationRequest> edotReq;
  for (const nevpt2::Tensor& kd : kappas)
    edotReq.push_back(nevpt2::ScRotationRequest{kd, /*tangent=*/true});
  const nevpt2::ScFullRotationResult frozen =
      NEVPT2_TRY(nevpt2::scFullRotationDevice(ints, gen, dm1, dm2, dm3, f3ac, f3ca, edotReq, res));
  const std::vector<nevpt2::ScDigestTangent> digest =
      NEVPT2_TRY(nevpt2::scDigestRotationTangentDevice(rdmOpt, g.get("ci"), g.nelecA, g.nelecB,
                                                       ints, gen, dm1, dm2, dm3, f3ac, f3ca,
                                                       edotReq, res));
  nevpt2::printScDigestTangentTable(frozen, digest);
  // 4. the frozen-digest tangent against the matrix being written.
  bool edotOk = true;
  double edotGapMax = 0.0;
  for (std::size_t d = 0; d < kappas.size(); ++d) {
    double fromY = 0.0;
    for (std::int64_t p = 0; p < nmo; ++p)
      for (std::int64_t q = 0; q < nmo; ++q) fromY += kappas[d](p, q) * y(p, q);
    const double gap = std::fabs(fromY - frozen.dirs[d].tangent);
    edotGapMax = std::max(edotGapMax, gap);
    edotOk = edotOk && gap <= kDumpEdotTol;
  }
  std::printf("  4. the %zu directions' frozen-digest Edot vs sum kappa.Y off the matrix: max "
              "|delta| = %.2e (<= %.0e: %s)\n",
              kappas.size(), edotGapMax, kDumpEdotTol, edotOk ? "ok" : "FAILED");

  // --- the file, and the file read back ----------------------------------------
  nevpt2::GradSrc out;
  out.kappaSource = y;
  out.ciSource = terms.ci;
  out.ci = g.get("ci");
  out.ncore = gg.ncore;
  out.nact = gg.nact;
  out.nvirt = gg.nvirt;
  out.eCorr = energies.total;
  const std::int64_t ndir = std::ssize(kappas);
  out.edotKappa = nevpt2::Tensor({ndir, nmo, nmo});
  out.edotFrozen = nevpt2::Tensor({ndir});
  out.edotDigest = nevpt2::Tensor({ndir});
  for (std::int64_t d = 0; d < ndir; ++d) {
    const std::size_t du = static_cast<std::size_t>(d);
    for (std::int64_t i = 0; i < nmo * nmo; ++i)
      out.edotKappa.flatRef(d * nmo * nmo + i) = kappas[du].flat(i);
    out.edotFrozen(d) = frozen.dirs[du].tangent;
    out.edotDigest(d) = digest[du].total;
  }
  NEVPT2_TRY(nevpt2::writeGradSrc(path, out));
  const nevpt2::GradSrc back = NEVPT2_TRY(nevpt2::readGradSrc(path));
  // 3. the round trip, bit for bit.
  const bool roundTrip = sameTensor(back.kappaSource, out.kappaSource) &&
                         sameTensor(back.ciSource, out.ciSource) &&
                         sameTensor(back.ci, out.ci) && back.ncore == out.ncore &&
                         back.nact == out.nact && back.nvirt == out.nvirt &&
                         std::bit_cast<std::uint64_t>(back.eCorr) ==
                             std::bit_cast<std::uint64_t>(out.eCorr) &&
                         sameTensor(back.edotKappa, out.edotKappa) &&
                         sameTensor(back.edotFrozen, out.edotFrozen) &&
                         sameTensor(back.edotDigest, out.edotDigest);
  std::printf("  3. %s read back: %s\n", path.c_str(),
              roundTrip ? "every member identical, bit for bit" : "DIFFERS from what was written");

  const bool ok = matrixOk && walkOk && epsOk && roundTrip && edotOk;
  std::printf("\n%s: the GPU's source terms Y_xy and y_I were assembled, written to %s and "
              "read back unchanged, with Edot(kappa) along %zu seeded directions in its "
              "frozen-digest and digest terms\n",
              ok ? "PASS" : "FAIL", path.c_str(), kappas.size());
  return ok;
}
