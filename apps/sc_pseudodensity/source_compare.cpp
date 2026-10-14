// nevpt2.source_compare's bodies: the basis carry, the projection, and the
// block-by-block verdict (source_compare.cppm has the why).
module nevpt2.source_compare;

import std;

namespace nevpt2 {

std::string sourceOracleStem(const std::string& path) {
  std::string stem = std::filesystem::path(path).filename().string();
  static constexpr std::string_view kSuffix = ".srcgold";
  if (stem.ends_with(kSuffix)) stem.resize(stem.size() - kSuffix.size());
  return stem;
}

Result<std::vector<std::string_view>> knownSourceGaps(std::string_view stem) {
  if (std::ranges::find(kClassifiedOracles, stem) == kClassifiedOracles.end())
    return err_config(std::format(
        "--source-oracle: no known-gap list is recorded for '{}' (source_compare.cppm's "
        "kKnownGaps / kClassifiedOracles) -- classify its blocks before gating on it",
        stem));
  std::vector<std::string_view> gaps;
  for (const KnownGap& k : kKnownGaps)
    if (k.oracle == stem) gaps.push_back(k.block);
  return gaps;
}

Tensor oracleBasisA(const SrcGoldFile& o, const Tensor& y) {
  const std::int64_t nmo = o.nmo, ncore = o.ncore, nact = o.nact;
  check(y.dims() == std::vector<int64_t>{nmo, nmo},
        std::format("oracleBasisA: Y is not [{}, {}]", nmo, nmo));
  // Wf = 1 off the active block and W on it, so A' = Wf^T A Wf -- the
  // transform nevpt2_grad.load_source_terms applies to a dump.
  const Tensor& w = o.get("act_rot");
  const auto wf = [&](std::int64_t r, std::int64_t c) -> double {
    const bool ra = r >= ncore && r < ncore + nact, ca = c >= ncore && c < ncore + nact;
    if (ra && ca) return w(r - ncore, c - ncore);
    return (r == c && !ra && !ca) ? 1.0 : 0.0;
  };
  std::vector<double> a(static_cast<std::size_t>(nmo * nmo));
  for (std::int64_t p = 0; p < nmo; ++p)
    for (std::int64_t q = 0; q < nmo; ++q) a[p * nmo + q] = y(p, q) - y(q, p);
  // tmp = A Wf, then A' = Wf^T tmp. n_mo is tens here, so plain loops.
  std::vector<double> tmp(a.size(), 0.0);
  Tensor ap({nmo, nmo});
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
      for (std::int64_t q = 0; q < nmo; ++q) ap(p, q) += wkp * tmp[k * nmo + q];
    }
  return ap;
}

Tensor projectOffState(const Tensor& ci, const Tensor& y, double& cy) {
  check(ci.size() == y.size(),
        std::format("projectOffState: c has {} elements, y {}", ci.size(), y.size()));
  cy = 0.0;
  for (std::int64_t i = 0; i < ci.size(); ++i) cy += ci.flat(i) * y.flat(i);
  Tensor py({ci.size()});
  for (std::int64_t i = 0; i < ci.size(); ++i) py.flatRef(i) = y.flat(i) - cy * ci.flat(i);
  return py;
}

// One compared element into its row: the running maxima, and where the worst
// difference sits.
void accountSourceRow(SourceBlockRow& r, double got, double ref, std::int64_t p,
                      std::int64_t q) {
  const double d = std::fabs(got - ref);
  ++r.n;
  r.maxRef = std::max(r.maxRef, std::fabs(ref));
  r.maxGot = std::max(r.maxGot, std::fabs(got));
  if (d >= r.maxDelta) {
    r.maxDelta = d;
    r.atP = p;
    r.atQ = q;
  }
}

SourceComparison compareSourceTerms(const SrcGoldFile& o, const Tensor& a, const Tensor& py,
                                    std::span<const std::string_view> gaps, double tol) {
  const std::int64_t nmo = o.nmo, ncore = o.ncore, nact = o.nact, nOcc = o.nActOcc;
  check(a.dims() == std::vector<int64_t>{nmo, nmo},
        std::format("compareSourceTerms: A is not [{}, {}]", nmo, nmo));
  check(py.size() == o.ndet,
        std::format("compareSourceTerms: P y has {} elements, the oracle {}", py.size(),
                    o.ndet));
  const auto spaceOf = [&](std::int64_t p) -> int {
    if (p < ncore) return 0;
    if (p < ncore + nOcc) return 1;
    if (p < ncore + nact) return 2;
    return 3;
  };
  SourceComparison out;
  out.tol = tol;
  // Every block of two spaces, in nevpt2_grad's order, then P y.
  std::array<std::array<int, 4>, 4> rowOf{};
  for (int s = 0; s < 4; ++s)
    for (int t = s; t < 4; ++t) {
      // act_O-act_O and act_V-act_V are the gauges: no pair there is compared.
      if (s == t && (s == 1 || s == 2)) continue;
      rowOf[s][t] = static_cast<int>(out.rows.size());
      out.rows.push_back(
          SourceBlockRow{std::format("{}-{}", kSourceSpaceNames[s], kSourceSpaceNames[t])});
    }
  const Tensor& aRef = o.get("a_corr");
  for (std::int64_t p = 0; p < nmo; ++p)
    for (std::int64_t q = p + 1; q < nmo; ++q) {
      const int s = spaceOf(p), t = spaceOf(q);
      if (s == t && (s == 1 || s == 2)) continue;
      accountSourceRow(out.rows[rowOf[s][t]], a(p, q), aRef(p, q), p, q);
    }
  const Tensor& pyRef = o.get("py");
  const Tensor& mask = o.get("ci_mask");
  SourceBlockRow ciRow{std::string(kSourceCiBlock)};
  for (std::int64_t i = 0; i < o.ndet; ++i)
    if (mask.flat(i) != 0.0) accountSourceRow(ciRow, py.flat(i), pyRef.flat(i), i, -1);
  out.rows.push_back(ciRow);

  out.ok = true;
  for (SourceBlockRow& r : out.rows) {
    r.knownGap = std::ranges::find(gaps, std::string_view(r.name)) != gaps.end();
    if (r.n == 0) {
      r.verdict = "empty";
      if (r.knownGap) {
        r.ok = false;
        r.verdict = "FAILED: on the known-gap list, but this case has no such pair";
      }
    } else if (r.knownGap) {
      r.ok = r.maxDelta > tol;
      r.verdict = r.ok ? "ok, still off"
                       : "FAILED: now AGREES -- take it off kKnownGaps "
                         "(source_compare.cppm)";
    } else {
      r.ok = r.maxDelta <= tol;
      r.verdict = r.ok ? "ok" : "FAILED: a block that agreed no longer does";
    }
    if (!r.ok) out.failures.push_back(r.name);
    if (r.n > 0) (r.knownGap ? out.stillGaps : out.agreed).push_back(r.name);
    out.ok = out.ok && r.ok;
  }
  for (const std::string_view gname : gaps)
    if (std::ranges::none_of(out.rows, [&](const SourceBlockRow& r) { return r.name == gname; })) {
      out.failures.push_back(std::format("{} (a known-gap entry naming no block)", gname));
      out.ok = false;
    }
  return out;
}

}  // namespace nevpt2
