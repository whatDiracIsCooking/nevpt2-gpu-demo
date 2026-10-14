// loadSrcGold: the `*.srcgold` source-term oracle reader (#92), both
// flavours (SC "NEVPT2S1", PC "NEVPT2P1", #95: one layout, the magic is the
// scheme). An implementation unit of nevpt2.golden, so it shares the one
// byte-level cursor and array parser with loadGolden and loadGradGold
// (`import :reader;`); the formats differ only in their magic and their
// scalar header (reference_data/golden_format.py's SRCGOLD_HEADERS is the
// layout).
module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.golden;

import std;
import :reader;

namespace nevpt2 {

std::string_view sourceSchemeName(SourceScheme scheme) {
  return scheme == SourceScheme::Pc ? "PC" : "SC";
}

Result<std::string> srcGoldCase(const std::string& path, SourceScheme scheme) {
  std::string stem = std::filesystem::path(path).filename().string();
  static constexpr std::string_view kSuffix = ".srcgold";
  if (stem.ends_with(kSuffix)) stem.resize(stem.size() - kSuffix.size());
  if (scheme == SourceScheme::Pc) {
    static constexpr std::string_view kPc = "_pc";
    if (!stem.ends_with(kPc))
      return err_io("source-term oracle: a PC oracle's name must end in '_pc.srcgold' -- " +
                    path);
    stem.resize(stem.size() - kPc.size());
  }
  return stem;
}

Status SrcGoldFile::checkAgainst(const GoldenFile& golden, const GradGoldFile& sidecar) const {
  if (nact != golden.ncas || ncore != golden.ncore || ndet != golden.ndet)
    return err_io(std::format(
        "source-term oracle: written for {} core / {} active orbitals and {} determinants, but "
        "the golden it was given with has {} / {} / {} -- the two files do not describe the "
        "same state",
        ncore, nact, ndet, golden.ncore, golden.ncas, golden.ndet));
  if (nmo != sidecar.nmo || nvirt != sidecar.nvirt)
    return err_io(std::format(
        "source-term oracle: n_mo = {} ({} virtual), but the gradient sidecar's is {} ({})", nmo,
        nvirt, sidecar.nmo, sidecar.nvirt));
  const Tensor& mine = get("ci");
  const Tensor& gold = golden.get("ci");
  for (int64_t i = 0; i < mine.size(); ++i)
    if (std::bit_cast<std::uint64_t>(mine.flat(i)) != std::bit_cast<std::uint64_t>(gold.flat(i)))
      return err_io(std::format(
          "source-term oracle: its ci differs from the golden's at determinant {} ({:.17g} vs "
          "{:.17g}) -- it was written for a different state",
          i, mine.flat(i), gold.flat(i)));
  return {};
}

const Tensor& SrcGoldFile::get(const std::string& name) const {
  auto it = arrays.find(name);
  if (it == arrays.end())
    check(false, std::format("source-term oracle: array '{}' is not one loadSrcGold requires",
                             name));
  return it->second;
}

Result<SrcGoldFile> loadSrcGold(const std::string& path) {
  Reader r = NEVPT2_TRY(Reader::open(path, "source-term oracle"));
  const std::string magic = NEVPT2_TRY(r.str(8));
  SrcGoldFile s;
  if (magic == "NEVPT2S1")
    s.scheme = SourceScheme::Sc;
  else if (magic == "NEVPT2P1")
    s.scheme = SourceScheme::Pc;
  else
    return err_io("source-term oracle: bad magic (got '" + magic +
                  "', expected 'NEVPT2S1' or 'NEVPT2P1') -- " + path);
  s.nmo = NEVPT2_TRY(r.i32());
  s.ncore = NEVPT2_TRY(r.i32());
  s.nact = NEVPT2_TRY(r.i32());
  s.nvirt = NEVPT2_TRY(r.i32());
  s.nActOcc = NEVPT2_TRY(r.i32());
  s.ndet = NEVPT2_TRY(r.i64());
  s.kappaStep = NEVPT2_TRY(r.f64());
  s.ciStep = NEVPT2_TRY(r.f64());
  if (s.nmo < 1 || s.ncore < 0 || s.nact < 1 || s.nvirt < 0 ||
      s.ncore + s.nact + s.nvirt != s.nmo || s.nActOcc < 0 || s.nActOcc > s.nact || s.ndet < 1)
    return err_io(std::format(
        "source-term oracle: header nmo = {}, ncore = {}, nact = {}, nvirt = {}, n_act_occ = {}, "
        "ndet = {} is not a usable partition -- {}",
        s.nmo, s.ncore, s.nact, s.nvirt, s.nActOcc, s.ndet, path));

  NEVPT2_TRY(readArraySection(r, s.arrays));
  // A fixed format: every array is read, so every one is required here, at
  // the shape the header says, and nothing downstream re-checks.
  const std::array<std::pair<std::string_view, std::vector<int64_t>>, 5> want{{
      {"a_corr", {s.nmo, s.nmo}},
      {"act_rot", {s.nact, s.nact}},
      {"py", {s.ndet}},
      {"ci_mask", {s.ndet}},
      {"ci", {s.ndet}},
  }};
  for (const auto& [name, dims] : want) {
    const auto it = s.arrays.find(std::string(name));
    if (it == s.arrays.end())
      return err_io(std::format("source-term oracle: missing required array '{}' -- {}", name,
                                path));
    if (it->second.dims() != dims) {
      std::string got, exp;
      for (const int64_t d : it->second.dims()) got += (got.empty() ? "" : ", ") + std::to_string(d);
      for (const int64_t d : dims) exp += (exp.empty() ? "" : ", ") + std::to_string(d);
      return err_io(std::format("source-term oracle: '{}' has shape [{}], expected [{}] -- {}",
                                name, got, exp, path));
    }
  }
  return s;
}

}  // namespace nevpt2
