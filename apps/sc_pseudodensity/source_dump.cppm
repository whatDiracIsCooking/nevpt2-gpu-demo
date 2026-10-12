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
//         forward-reverse duality").
//
// AND WHAT THEY ARE NOT. Both are derivatives of the functional this tree
// differentiates, which holds two things fixed that the exact energy does not
// (docs/gradients.md, sections 1 and 6): Y holds the f3ac/f3ca digests fixed,
// and y holds the orbital energies e_core/e_virt fixed. Writing the file
// changes neither; what that costs a nuclear gradient is measured, against
// PySCF's exact source terms, in docs/testing.md, "The GPU as the source-term
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
//      check that the sweep ran on the state this binary loaded;
//   3  the file read back (readGradSrc) and compared with what was written,
//      exactly: the doubles are stored raw, so any difference is a bug.
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
// Check 1's direction, distinct from every seed --kappa-full and --kappa-act use.
constexpr std::uint64_t kDumpSeed = 0x5eed'0073u;

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
                                     const nevpt2::DeviceTensor& f3ca, std::int64_t sigmaTiles,
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

  // --- Y over every pair ------------------------------------------------------
  std::vector<std::pair<std::int64_t, std::int64_t>> pairs;
  for (std::int64_t p = 0; p < nmo; ++p)
    for (std::int64_t q = p + 1; q < nmo; ++q) pairs.emplace_back(p, q);
  const auto tY = std::chrono::steady_clock::now();
  const nevpt2::Tensor y = NEVPT2_TRY(nevpt2::scFullRotationMatrix(
      ints, gen, dm1, dm2, dm3, f3ac, f3ca, pairs, kDumpChunk, res));
  double yMax = 0.0;
  for (std::int64_t i = 0; i < y.size(); ++i) yMax = std::max(yMax, std::fabs(y.flat(i)));
  std::printf("  Y: %zu basis directions in chunks of %lld, %.2fs; max|Y| = %.6e, |Y| = %.6e\n",
              pairs.size(), static_cast<long long>(kDumpChunk),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - tY).count(),
              yMax, frobenius(y));

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

  // --- y over every determinant ----------------------------------------------
  const nevpt2::Tensor& ci = g.get("ci");
  nevpt2::SigmaVectorOptions sigmaOpt;
  sigmaOpt.nTiles = sigmaTiles;
  const auto tCi = std::chrono::steady_clock::now();
  const nevpt2::ScCiSourceResult ciSrc = nevpt2::scCiSourceDevice(
      ints, ci, g.nelecA, g.nelecB, dm1, dm2, dm3, f3ac, f3ca, sigmaOpt, res);
  double cy = 0.0;
  for (std::int64_t i = 0; i < ci.size(); ++i) cy += ci.flat(i) * ciSrc.y.flat(i);
  std::printf("  y: %lld determinants, %.2fs; |y| = %.10f, <c, y> = %.10f\n",
              static_cast<long long>(ciSrc.y.size()),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - tCi).count(),
              frobenius(ciSrc.y), cy);

  // 2. the walk y rode on, against the energy stage.
  const double walkGap = std::fabs(ciSrc.pseudodensities.total - energies.total);
  const bool walkOk = walkGap < kDumpWalkTol;
  std::printf("  2. E_corr from y's walk %.10f vs energiesDevice %.10f, |delta| = %.2e "
              "(< %.0e: %s)\n",
              ciSrc.pseudodensities.total, energies.total, walkGap, kDumpWalkTol,
              walkOk ? "ok" : "FAILED");

  // --- the file, and the file read back ----------------------------------------
  nevpt2::GradSrc out;
  out.kappaSource = y;
  out.ciSource = ciSrc.y;
  out.ci = ci;
  out.ncore = gg.ncore;
  out.nact = gg.nact;
  out.nvirt = gg.nvirt;
  out.eCorr = energies.total;
  NEVPT2_TRY(nevpt2::writeGradSrc(path, out));
  const nevpt2::GradSrc back = NEVPT2_TRY(nevpt2::readGradSrc(path));
  // 3. the round trip, bit for bit.
  const bool roundTrip = sameTensor(back.kappaSource, out.kappaSource) &&
                         sameTensor(back.ciSource, out.ciSource) &&
                         sameTensor(back.ci, out.ci) && back.ncore == out.ncore &&
                         back.nact == out.nact && back.nvirt == out.nvirt &&
                         std::bit_cast<std::uint64_t>(back.eCorr) ==
                             std::bit_cast<std::uint64_t>(out.eCorr);
  std::printf("  3. %s read back: %s\n", path.c_str(),
              roundTrip ? "every member identical, bit for bit" : "DIFFERS from what was written");

  const bool ok = matrixOk && walkOk && roundTrip;
  std::printf("\n%s: the GPU's source terms Y_xy and y_I were assembled, written to %s and "
              "read back unchanged\n",
              ok ? "PASS" : "FAIL", path.c_str());
  return ok;
}
