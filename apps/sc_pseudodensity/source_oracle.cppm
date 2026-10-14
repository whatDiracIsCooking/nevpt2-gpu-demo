// nevpt2.app.sc_pseudodensity:source_oracle -- --source-oracle PATH: the
// GPU's own SC-NEVPT2 source terms held, block by block, to PySCF's EXACT
// ones, committed as a `.srcgold` beside the golden (#92). An interface
// partition that exports nothing, as :source_dump is.
//
// WHY. Every other on-card check of Y_xy and y_I compares the device with
// itself -- forward against reverse, at round-off -- so a term BOTH modes
// leave out cancels, and none of them can see it (#73 found two that way, by
// the offline nevpt2_grad.py --source both). This is that offline comparison
// with PySCF's side computed once, by reference_data/sc_source_oracle.py, and
// committed: the dump's own assembly (:source_dump's assembleSourceTerms),
// then
//
//   A = Y - Y^T, carried into the oracle's basis (Y' = W^T Y W on the active
//         block, W = act_rot) and compared with PySCF's A_corr on every
//         non-gauge pair, grouped into the eight blocks of the four spaces
//         core / act_O / act_V / virt;
//   P y = y - <c, y> c, compared with PySCF's on the determinants of the
//         state's irrep (ci_mask), in the golden's own basis.
//
// THE TOLERANCE IS NOT 1e-7: PySCF's side is a central difference, and its
// truncation at the committed steps is what the agreement can be read to.
// Measured from the step ladders, its one home is docs/testing.md, "The SC
// source terms against PySCF's, on a card" -- read from that table row at
// configure time (NEVPT2_SOURCE_ORACLE_TOL, apps/sc_pseudodensity/
// CMakeLists.txt), as the self-variation envelope is.
//
// THE KNOWN GAPS ARE PINNED, NOT HIDDEN. kKnownGaps below is the ONE list of
// blocks the device is known to get wrong. A block on it must still DISAGREE
// by more than the tolerance and every other block must AGREE within it, so
// the entry fails if a correct block breaks AND if a fix lands without the
// list being updated. It is EMPTY: #87 took P y off it, and #89 -- the
// digests' transpose carried into Y (docs/gradients.md, section 6) -- took
// off every Y block the f3ac/f3ca digests had kept on it.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry
// NEVPT2_TRY: a macro, which no import carries. An oracle that does not
// describe this golden is the caller's input and propagates unchanged to
// main, which reports it once.
#include "error_handling/error_macros.h"

export module nevpt2.app.sc_pseudodensity:source_oracle;

import std;
import nevpt2.gradient;
import nevpt2.golden;
// requireGradientArrays / uploadGeneral: the sidecar as --kappa-full reads it.
import :kappa_full_check;
// assembleSourceTerms: the dump's own Y and y.
import :source_dump;

#ifndef NEVPT2_SOURCE_ORACLE_TOL
#error "NEVPT2_SOURCE_ORACLE_TOL must be defined by apps/sc_pseudodensity/CMakeLists.txt"
#endif
// Per element, absolute (Eh): what every block not on kKnownGaps must agree
// to, and what every block on it must still miss by.
constexpr double kSourceOracleTol = NEVPT2_SOURCE_ORACLE_TOL;

// The four spaces, in nevpt2_grad.py's SPACE_NAMES order.
constexpr std::array<std::string_view, 4> kSpaceNames{"core", "act_O", "act_V", "virt"};
// The CI half's row name, as the known-gap list spells it.
constexpr std::string_view kCiBlock = "P y";

// THE known-gap list: per committed oracle (its file name's stem), the blocks
// the device is known to get wrong today. EMPTY since #89, and kept as an
// explicit "nothing is known wrong": every block of both oracles is held to
// agree. Until then it held every Y block an active two-electron integral
// moves under (N2 CAS(4,4): act_O-virt, act_V-virt; LiF: act_O-act_V,
// act_O-virt, act_V-virt), for the one omission #89 closed -- Y held the
// f3ac/f3ca digests fixed (docs/testing.md, "The GPU as the source-term
// provider" keeps that record). `P y` left it with #87.
struct KnownGap {
  std::string_view oracle;
  std::string_view block;
};
constexpr std::array<KnownGap, 0> kKnownGaps{};
// The oracles that have a list at all -- one with no known gap is named here
// too, so an oracle nobody has classified is refused rather than read as
// "everything must agree".
constexpr std::array<std::string_view, 2> kClassifiedOracles{"n2_ccpvdz_cas44",
                                                             "lif_def2svp_cas46"};

struct BlockRow {
  std::string name;
  std::int64_t n = 0;
  double maxRef = 0.0;
  double maxGpu = 0.0;
  double maxDelta = 0.0;
  std::int64_t atP = -1, atQ = -1;  // where (for P y: the determinant, in atP)
};

void account(BlockRow& r, double gpu, double ref, std::int64_t p, std::int64_t q) {
  const double d = std::fabs(gpu - ref);
  ++r.n;
  r.maxRef = std::max(r.maxRef, std::fabs(ref));
  r.maxGpu = std::max(r.maxGpu, std::fabs(gpu));
  if (d >= r.maxDelta) {
    r.maxDelta = d;
    r.atP = p;
    r.atQ = q;
  }
}

// The oracle file's case name: its file name without the directory and the
// `.srcgold` suffix.
std::string oracleStem(const std::string& path) {
  std::string stem = std::filesystem::path(path).filename().string();
  static constexpr std::string_view kSuffix = ".srcgold";
  if (stem.ends_with(kSuffix)) stem.resize(stem.size() - kSuffix.size());
  return stem;
}

// Compares the GPU's source terms for the golden's state with the oracle at
// `oraclePath`, prints the per-block table, and returns whether every block
// is where kKnownGaps says it is. A sidecar or an oracle that does not
// describe this golden, or an oracle with no known-gap classification, is an
// Error the caller reports once.
nevpt2::Result<bool> checkSourceOracle(const nevpt2::GoldenFile& g,
                                       const nevpt2::GradGoldFile& gg,
                                       const std::string& oraclePath,
                                       const nevpt2::EnergyResult& energies,
                                       const nevpt2::NevptIntegralsDevice& ints,
                                       const nevpt2::DeviceTensor& dm1,
                                       const nevpt2::DeviceTensor& dm2,
                                       const nevpt2::DeviceTensor& dm3,
                                       const nevpt2::DeviceTensor& f3ac,
                                       const nevpt2::DeviceTensor& f3ca,
                                       const nevpt2::RdmBuildOptions& rdmOpt,
                                       const nevpt2::DeviceResources& res) {
  NEVPT2_TRY(gg.checkAgainst(g));
  NEVPT2_TRY(requireGradientArrays(gg));
  std::printf("source-term oracle: %s\n", oraclePath.c_str());
  const nevpt2::SrcGoldFile o = NEVPT2_TRY(nevpt2::loadSrcGold(oraclePath));
  NEVPT2_TRY(o.checkAgainst(g, gg));
  const std::string stem = oracleStem(oraclePath);
  if (std::ranges::find(kClassifiedOracles, stem) == kClassifiedOracles.end())
    return nevpt2::err_config(std::format(
        "--source-oracle: no known-gap list is recorded for '{}' (source_oracle.cppm's "
        "kKnownGaps / kClassifiedOracles) -- classify its blocks before gating on it",
        stem));
  std::vector<std::string_view> gaps;
  for (const KnownGap& k : kKnownGaps)
    if (k.oracle == stem) gaps.push_back(k.block);

  const std::int64_t nmo = o.nmo, ncore = o.ncore, nact = o.nact, nOcc = o.nActOcc;
  std::printf("\n--source-oracle: the GPU's Y_xy and y_I against PySCF's exact ones (#92)\n");
  std::printf("  partition: n_mo = %lld = %lld core + %lld active (%lld act_O) + %lld virtual; "
              "PySCF's steps: kappa %.6g, CI %.6g\n",
              static_cast<long long>(nmo), static_cast<long long>(ncore),
              static_cast<long long>(nact), static_cast<long long>(nOcc),
              static_cast<long long>(o.nvirt), o.kappaStep, o.ciStep);
  const nevpt2::GeneralIntegralsDevice gen = uploadGeneral(gg, res);
  // The digests' transpose (#89), so the Y compared is the exact one.
  const nevpt2::DeviceTensor hbar = nevpt2::scDigestAdjointDevice(
      rdmOpt, g.get("ci"), g.nelecA, g.nelecB, ints, dm1, dm2, dm3, f3ac, f3ca, res);
  const GpuSourceTerms terms = NEVPT2_TRY(assembleSourceTerms(
      g, gg, energies, ints, gen, dm1, dm2, dm3, f3ac, f3ca, hbar, rdmOpt.nTiles, res));

  // --- A = Y - Y^T, carried into the oracle's basis ----------------------------
  // Wf = 1 off the active block and W on it, so A' = Wf^T A Wf -- the
  // transform nevpt2_grad.load_source_terms applies to a dump.
  const nevpt2::Tensor& yk = terms.kappa;
  const nevpt2::Tensor& w = o.get("act_rot");
  const auto wf = [&](std::int64_t r, std::int64_t c) -> double {
    const bool ra = r >= ncore && r < ncore + nact, ca = c >= ncore && c < ncore + nact;
    if (ra && ca) return w(r - ncore, c - ncore);
    return (r == c && !ra && !ca) ? 1.0 : 0.0;
  };
  std::vector<double> a(static_cast<std::size_t>(nmo * nmo));
  for (std::int64_t p = 0; p < nmo; ++p)
    for (std::int64_t q = 0; q < nmo; ++q) a[p * nmo + q] = yk(p, q) - yk(q, p);
  // tmp = A Wf, then A' = Wf^T tmp. n_mo is tens here, so plain loops.
  std::vector<double> tmp(a.size(), 0.0), ap(a.size(), 0.0);
  for (std::int64_t p = 0; p < nmo; ++p)
    for (std::int64_t k = 0; k < nmo; ++k) {
      const double apk = a[p * nmo + k];
      if (apk == 0.0) continue;
      for (std::int64_t q = 0; q < nmo; ++q) tmp[p * nmo + q] += apk * wf(k, q);
    }
  for (std::int64_t k = 0; k < nmo; ++k)
    for (std::int64_t p = 0; p < nmo; ++p) {
      const double wkp = wf(k, p);
      if (wkp == 0.0) continue;
      for (std::int64_t q = 0; q < nmo; ++q) ap[p * nmo + q] += wkp * tmp[k * nmo + q];
    }

  const auto spaceOf = [&](std::int64_t p) -> int {
    if (p < ncore) return 0;
    if (p < ncore + nOcc) return 1;
    if (p < ncore + nact) return 2;
    return 3;
  };
  // Every block of two spaces, in nevpt2_grad's order, then P y.
  std::vector<BlockRow> rows;
  std::array<std::array<int, 4>, 4> rowOf{};
  for (int s = 0; s < 4; ++s)
    for (int t = s; t < 4; ++t) {
      // act_O-act_O and act_V-act_V are the gauges: no pair there is compared.
      if (s == t && (s == 1 || s == 2)) continue;
      rowOf[s][t] = static_cast<int>(rows.size());
      rows.push_back(BlockRow{std::format("{}-{}", kSpaceNames[s], kSpaceNames[t])});
    }
  const nevpt2::Tensor& aRef = o.get("a_corr");
  for (std::int64_t p = 0; p < nmo; ++p)
    for (std::int64_t q = p + 1; q < nmo; ++q) {
      const int s = spaceOf(p), t = spaceOf(q);
      if (s == t && (s == 1 || s == 2)) continue;
      account(rows[rowOf[s][t]], ap[p * nmo + q], aRef(p, q), p, q);
    }

  // --- P y, in the golden's basis ----------------------------------------------
  const nevpt2::Tensor& ci = g.get("ci");
  const nevpt2::Tensor& yc = terms.ci;
  double cy = 0.0;
  for (std::int64_t i = 0; i < ci.size(); ++i) cy += ci.flat(i) * yc.flat(i);
  const nevpt2::Tensor& pyRef = o.get("py");
  const nevpt2::Tensor& mask = o.get("ci_mask");
  BlockRow ciRow{std::string(kCiBlock)};
  for (std::int64_t i = 0; i < ci.size(); ++i)
    if (mask.flat(i) != 0.0) account(ciRow, yc.flat(i) - cy * ci.flat(i), pyRef.flat(i), i, -1);
  rows.push_back(ciRow);

  // --- the verdict, block by block ------------------------------------------------
  std::printf("\n  per block, max over its pairs (Eh; tolerance %.0e, docs/testing.md, \"The SC "
              "source terms against PySCF's, on a card\"):\n",
              kSourceOracleTol);
  std::printf("    %-12s %6s %12s %12s %11s  %-12s %s\n", "block", "pairs", "max|ref|",
              "max|GPU|", "max|delta|", "expected", "verdict");
  bool ok = true;
  std::vector<std::string> agreed, stillGaps, failures;
  for (const BlockRow& r : rows) {
    const bool known = std::ranges::find(gaps, std::string_view(r.name)) != gaps.end();
    std::string verdict;
    bool rowOk = true;
    if (r.n == 0) {
      verdict = "empty";
      if (known) {
        rowOk = false;
        verdict = "FAILED: on the known-gap list, but this case has no such pair";
      }
    } else if (known) {
      rowOk = r.maxDelta > kSourceOracleTol;
      verdict = rowOk ? "ok, still off"
                      : "FAILED: now AGREES -- take it off kKnownGaps (source_oracle.cppm)";
    } else {
      rowOk = r.maxDelta <= kSourceOracleTol;
      verdict = rowOk ? "ok" : "FAILED: a block that agreed no longer does";
    }
    const std::string at =
        r.atQ >= 0 ? std::format("({}, {})", r.atP, r.atQ) : std::format("det {}", r.atP);
    std::printf("    %-12s %6lld %12.4e %12.4e %11.3e  %-12s %s  at %s\n", r.name.c_str(),
                static_cast<long long>(r.n), r.maxRef, r.maxGpu, r.maxDelta,
                known ? "known gap" : "agree", verdict.c_str(), at.c_str());
    if (!rowOk) failures.push_back(r.name);
    if (r.n > 0) (known ? stillGaps : agreed).push_back(r.name);
    ok = ok && rowOk;
  }
  // A list entry naming no block at all (a typo) fails too: it would pin nothing.
  for (const std::string_view gname : gaps)
    if (std::ranges::none_of(rows, [&](const BlockRow& r) { return r.name == gname; })) {
      std::printf("    known-gap entry '%.*s' names no block -- FAILED\n",
                  static_cast<int>(gname.size()), gname.data());
      failures.emplace_back(gname);
      ok = false;
    }
  std::printf("    (the GPU's <c, y> = %+.6e, dropped by the projection)\n", cy);
  ok = ok && terms.walkOk && terms.epsOk;

  const auto join = [](const std::vector<std::string>& v) {
    std::string s;
    for (const std::string& x : v) s += (s.empty() ? "" : ", ") + x;
    return s.empty() ? std::string("none") : s;
  };
  if (ok)
    std::printf("\nPASS: the GPU's SC source terms against PySCF's committed ones, %s: %zu "
                "blocks agree within %.0e, and the %zu known gaps still disagree (%s)\n",
                stem.c_str(), agreed.size(), kSourceOracleTol, stillGaps.size(),
                join(stillGaps).c_str());
  else
    std::printf("\nFAIL: the GPU's SC source terms against PySCF's committed ones, %s: %s\n",
                stem.c_str(),
                failures.empty() ? "y's walk or the orbital energies it reads did not check out"
                                 : ("blocks out of place: " + join(failures)).c_str());
  return ok;
}
