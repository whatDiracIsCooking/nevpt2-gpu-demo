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

// Which of the four spaces orbital p is in (nevpt2_grad's State.space_of).
int sourceSpaceOf(const SrcGoldFile& o, std::int64_t p) {
  if (p < o.ncore) return 0;
  if (p < o.ncore + o.nActOcc) return 1;
  if (p < o.ncore + o.nact) return 2;
  return 3;
}

// act_O-act_O and act_V-act_V are the gauges: no pair there is compared.
bool isGaugeBlock(int s, int t) { return s == t && (s == 1 || s == 2); }

// The eight rows of two spaces, in nevpt2_grad's order (every s <= t but the
// gauges), each with the index of the block it is -- what a directional
// oracle's kappa_block counts -- and rowOf[s][t] pointing at it.
std::vector<SourceBlockRow> sourceBlockRows(std::array<std::array<int, 4>, 4>& rowOf) {
  std::vector<SourceBlockRow> rows;
  for (int s = 0; s < 4; ++s)
    for (int t = s; t < 4; ++t) {
      if (isGaugeBlock(s, t)) continue;
      rowOf[s][t] = static_cast<int>(rows.size());
      rows.push_back(
          SourceBlockRow{std::format("{}-{}", kSourceSpaceNames[s], kSourceSpaceNames[t])});
    }
  return rows;
}

// How many non-gauge pairs block (s, t) has in this oracle's partition.
std::int64_t sourceBlockPairs(const SrcGoldFile& o, int s, int t) {
  const std::array<std::int64_t, 4> size{o.ncore, o.nActOcc, o.nact - o.nActOcc, o.nvirt};
  return s == t ? size[s] * (size[s] - 1) / 2 : size[s] * size[t];
}

// Every row held to where `gaps` puts it at `tol` -- the one verdict both
// flavours give. `minProbes` > 0 also fails a row the partition has pairs in
// that fewer directions than that probed (pairs[i] is row i's pair count; the
// P y row is last and always has some).
void finishSourceComparison(SourceComparison& out, std::span<const std::string_view> gaps,
                            std::span<const std::int64_t> pairs, std::int64_t minProbes) {
  out.ok = true;
  for (std::size_t i = 0; i < out.rows.size(); ++i) {
    SourceBlockRow& r = out.rows[i];
    r.knownGap = std::ranges::find(gaps, std::string_view(r.name)) != gaps.end();
    if (minProbes > 0 && pairs[i] > 0 && r.n < minProbes) {
      r.ok = false;
      r.verdict = std::format("FAILED: {} pairs in this partition, but only {} direction(s) "
                              "probe the block (need {})",
                              pairs[i], r.n, minProbes);
    } else if (r.n == 0) {
      r.verdict = "empty";
      if (r.knownGap) {
        r.ok = false;
        r.verdict = "FAILED: on the known-gap list, but this case has no such pair";
      }
    } else if (r.knownGap) {
      r.ok = r.maxDelta > out.tol;
      r.verdict = r.ok ? "ok, still off"
                       : "FAILED: now AGREES -- take it off kKnownGaps "
                         "(source_compare.cppm)";
    } else {
      r.ok = r.maxDelta <= out.tol;
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
}

SourceComparison compareSourceTerms(const SrcGoldFile& o, const Tensor& a, const Tensor& py,
                                    std::span<const std::string_view> gaps, double tol) {
  const std::int64_t nmo = o.nmo;
  check(o.flavour == SourceFlavour::Full,
        "compareSourceTerms: a directional oracle is compareSourceDirections'");
  check(a.dims() == std::vector<int64_t>{nmo, nmo},
        std::format("compareSourceTerms: A is not [{}, {}]", nmo, nmo));
  check(py.size() == o.ndet,
        std::format("compareSourceTerms: P y has {} elements, the oracle {}", py.size(),
                    o.ndet));
  SourceComparison out;
  out.tol = tol;
  std::array<std::array<int, 4>, 4> rowOf{};
  out.rows = sourceBlockRows(rowOf);
  const Tensor& aRef = o.get("a_corr");
  for (std::int64_t p = 0; p < nmo; ++p)
    for (std::int64_t q = p + 1; q < nmo; ++q) {
      const int s = sourceSpaceOf(o, p), t = sourceSpaceOf(o, q);
      if (isGaugeBlock(s, t)) continue;
      accountSourceRow(out.rows[rowOf[s][t]], a(p, q), aRef(p, q), p, q);
    }
  const Tensor& pyRef = o.get("py");
  const Tensor& mask = o.get("ci_mask");
  SourceBlockRow ciRow{std::string(kSourceCiBlock)};
  for (std::int64_t i = 0; i < o.ndet; ++i)
    if (mask.flat(i) != 0.0) accountSourceRow(ciRow, py.flat(i), pyRef.flat(i), i, -1);
  out.rows.push_back(ciRow);
  finishSourceComparison(out, gaps, {}, 0);
  return out;
}

Status checkDirectionBlocks(const SrcGoldFile& o) {
  if (o.flavour != SourceFlavour::Directional) return {};
  std::array<std::array<int, 4>, 4> rowOf{};
  const std::vector<SourceBlockRow> rows = sourceBlockRows(rowOf);
  const Tensor& dirs = o.get("kappa_dirs");
  const Tensor& block = o.get("kappa_block");
  for (std::int64_t d = 0; d < dirs.dims()[0]; ++d) {
    const int b = static_cast<int>(block.flat(d));
    for (std::int64_t p = 0; p < o.nmo; ++p)
      for (std::int64_t q = p + 1; q < o.nmo; ++q) {
        if (dirs(d, p, q) == 0.0) continue;
        const int s = sourceSpaceOf(o, p), t = sourceSpaceOf(o, q);
        if (isGaugeBlock(s, t) || rowOf[s][t] != b)
          return err_io(std::format(
              "source-term oracle: kappa direction {} is labelled {} but is nonzero at ({}, "
              "{}), in {} -- its projection would not be that block's",
              d, rows[static_cast<std::size_t>(b)].name, p, q,
              isGaugeBlock(s, t) ? std::format("the {0}-{0} gauge", kSourceSpaceNames[s])
                                 : rows[static_cast<std::size_t>(rowOf[s][t])].name));
      }
  }
  return {};
}

SourceComparison compareSourceDirections(const SrcGoldFile& o, const Tensor& a,
                                         const Tensor& py,
                                         std::span<const std::string_view> gaps, double tol) {
  const std::int64_t nmo = o.nmo;
  check(o.flavour == SourceFlavour::Directional,
        "compareSourceDirections: a full oracle is compareSourceTerms'");
  check(a.dims() == std::vector<int64_t>{nmo, nmo},
        std::format("compareSourceDirections: A is not [{}, {}]", nmo, nmo));
  check(py.size() == o.ndet,
        std::format("compareSourceDirections: P y has {} elements, the oracle {}", py.size(),
                    o.ndet));
  SourceComparison out;
  out.directional = true;
  out.tol = tol;
  std::array<std::array<int, 4>, 4> rowOf{};
  out.rows = sourceBlockRows(rowOf);
  std::vector<std::int64_t> pairs(out.rows.size() + 1, 1);
  for (int s = 0; s < 4; ++s)
    for (int t = s; t < 4; ++t)
      if (!isGaugeBlock(s, t)) pairs[static_cast<std::size_t>(rowOf[s][t])] = sourceBlockPairs(o, s, t);

  const Tensor& dirs = o.get("kappa_dirs");
  const Tensor& block = o.get("kappa_block");
  const Tensor& val = o.get("kappa_val");
  for (std::int64_t d = 0; d < dirs.dims()[0]; ++d) {
    double got = 0.0;
    for (std::int64_t p = 0; p < nmo; ++p)
      for (std::int64_t q = p + 1; q < nmo; ++q) got += dirs(d, p, q) * a(p, q);
    accountSourceRow(out.rows[static_cast<std::size_t>(block.flat(d))], got, val.flat(d), d, -1);
  }
  const Tensor& u = o.get("ci_dirs");
  const Tensor& cval = o.get("ci_val");
  SourceBlockRow ciRow{std::string(kSourceCiBlock)};
  for (std::int64_t d = 0; d < u.dims()[0]; ++d) {
    double got = 0.0;
    for (std::int64_t i = 0; i < o.ndet; ++i) got += u(d, i) * py.flat(i);
    accountSourceRow(ciRow, got, cval.flat(d), d, -1);
  }
  out.rows.push_back(ciRow);
  finishSourceComparison(out, gaps, pairs, kMinBlockDirections);
  return out;
}

SourceComparison compareSourceOracle(const SrcGoldFile& o, const Tensor& a, const Tensor& py,
                                     std::span<const std::string_view> gaps, double tol) {
  return o.flavour == SourceFlavour::Directional ? compareSourceDirections(o, a, py, gaps, tol)
                                                 : compareSourceTerms(o, a, py, gaps, tol);
}

}  // namespace nevpt2
