// loadSrcGold: the `*.srcgold` source-term oracle reader (#92), all four kinds
// (SC "NEVPT2S1", PC "NEVPT2P1", #95, and their directional flavours
// "NEVPT2SD" / "NEVPT2PD", #105: one layout, the magic is the kind). An
// implementation unit of nevpt2.golden, so it shares the one byte-level
// cursor and array parser with loadGolden and loadGradGold (`import
// :reader;`); the formats differ only in their magic and their scalar header
// (reference_data/golden_format.py's SRCGOLD_HEADERS is the layout).
module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.golden;

import std;
import :reader;

namespace nevpt2 {

std::string_view sourceSchemeName(SourceScheme scheme) {
  return scheme == SourceScheme::Pc ? "PC" : "SC";
}

std::string_view sourceFlavourName(SourceFlavour flavour) {
  return flavour == SourceFlavour::Directional ? "directional" : "full";
}

Result<std::string> srcGoldCase(const std::string& path, SourceScheme scheme,
                                SourceFlavour flavour) {
  std::string stem = std::filesystem::path(path).filename().string();
  static constexpr std::string_view kSuffix = ".srcgold";
  if (stem.ends_with(kSuffix)) stem.resize(stem.size() - kSuffix.size());
  if (flavour == SourceFlavour::Directional) {
    static constexpr std::string_view kDir = "_dir";
    if (!stem.ends_with(kDir))
      return err_io(
          "source-term oracle: a directional oracle's name must end in '_dir.srcgold' -- " +
          path);
    stem.resize(stem.size() - kDir.size());
  }
  if (scheme == SourceScheme::Pc) {
    static constexpr std::string_view kPc = "_pc";
    if (!stem.ends_with(kPc))
      return err_io(std::format(
          "source-term oracle: a PC oracle's name must end in '{}.srcgold' -- {}",
          flavour == SourceFlavour::Directional ? "_pc_dir" : "_pc", path));
    stem.resize(stem.size() - kPc.size());
  }
  return stem;
}

Status SrcGoldFile::checkAgainstGolden(const GoldenFile& golden) const {
  if (nact != golden.ncas || ncore != golden.ncore || ndet != golden.ndet)
    return err_io(std::format(
        "source-term oracle: written for {} core / {} active orbitals and {} determinants, but "
        "the golden it was given with has {} / {} / {} -- the two files do not describe the "
        "same state",
        ncore, nact, ndet, golden.ncore, golden.ncas, golden.ndet));
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

Status SrcGoldFile::checkAgainst(const GoldenFile& golden, const GradGoldFile& sidecar) const {
  NEVPT2_TRY(checkAgainstGolden(golden));
  if (nmo != sidecar.nmo || nvirt != sidecar.nvirt)
    return err_io(std::format(
        "source-term oracle: n_mo = {} ({} virtual), but the gradient sidecar's is {} ({})", nmo,
        nvirt, sidecar.nmo, sidecar.nvirt));
  return {};
}

const Tensor& SrcGoldFile::get(const std::string& name) const {
  auto it = arrays.find(name);
  if (it == arrays.end())
    check(false, std::format("source-term oracle: array '{}' is not one loadSrcGold requires",
                             name));
  return it->second;
}

// Every array of the flavour present at its shape, or an IO Error naming the
// first that is not. A shape entry of -1 is free (a direction or rung count)
// and is bound by its first appearance, so every array that shares it agrees.
Status requireSrcGoldArrays(
    const SrcGoldFile& s,
    std::span<const std::pair<std::string_view, std::vector<int64_t>>> want,
    const std::string& path) {
  std::map<int, int64_t> bound;  // -1 - k -> the extent free dimension k took
  for (const auto& [name, dims] : want) {
    const auto it = s.arrays.find(std::string(name));
    if (it == s.arrays.end())
      return err_io(std::format("source-term oracle: missing required array '{}' -- {}", name,
                                path));
    const std::vector<int64_t>& got = it->second.dims();
    bool ok = got.size() == dims.size();
    for (std::size_t i = 0; ok && i < dims.size(); ++i) {
      if (dims[i] >= 0) {
        ok = got[i] == dims[i];
      } else {
        const auto [b, fresh] = bound.try_emplace(static_cast<int>(dims[i]), got[i]);
        ok = got[i] == b->second;
      }
    }
    if (!ok) {
      std::string g, e;
      for (const int64_t d : got) g += (g.empty() ? "" : ", ") + std::to_string(d);
      for (const int64_t d : dims) e += (e.empty() ? "" : ", ") + (d < 0 ? "n" : std::to_string(d));
      return err_io(std::format("source-term oracle: '{}' has shape [{}], expected [{}] -- {}",
                                name, g, e, path));
    }
  }
  return {};
}

// The directional flavour's own invariants (loadSrcGold's comment lists them).
Status checkDirectionalArrays(const SrcGoldFile& s, const std::string& path) {
  const Tensor& dirs = s.get("kappa_dirs");
  const Tensor& block = s.get("kappa_block");
  const std::int64_t nk = dirs.dims()[0], nmo = s.nmo;
  if (nk < 1 || s.get("ci_dirs").dims()[0] < 1)
    return err_io("source-term oracle: a directional oracle with no kappa or no CI direction -- " +
                  path);
  if (s.get("kappa_fd").dims()[1] < 2 || s.get("ci_fd").dims()[1] < 2)
    return err_io("source-term oracle: a ladder of fewer than two rungs has no Richardson "
                  "value -- " + path);
  for (std::int64_t d = 0; d < nk; ++d) {
    const double b = block.flat(d);
    if (!(b >= 0.0 && b < 8.0) || b != std::floor(b))
      return err_io(std::format("source-term oracle: direction {}'s block index {} is not one "
                                "of the eight -- {}", d, b, path));
    for (std::int64_t p = 0; p < nmo; ++p)
      for (std::int64_t q = p; q < nmo; ++q)
        if (dirs(d, p, q) != -dirs(d, q, p))
          return err_io(std::format("source-term oracle: kappa direction {} is not "
                                    "antisymmetric at ({}, {}) -- {}", d, p, q, path));
  }
  const Tensor& ci = s.get("ci");
  const Tensor& u = s.get("ci_dirs");
  for (std::int64_t d = 0; d < u.dims()[0]; ++d) {
    double dot = 0.0;
    for (std::int64_t i = 0; i < s.ndet; ++i) dot += u(d, i) * ci.flat(i);
    if (std::fabs(dot) > 1e-10)
      return err_io(std::format("source-term oracle: CI direction {} is not orthogonal to ci "
                                "(<u, c> = {:.2e}) -- {}", d, dot, path));
  }
  return {};
}

Result<SrcGoldFile> loadSrcGold(const std::string& path) {
  Reader r = NEVPT2_TRY(Reader::open(path, "source-term oracle"));
  const std::string magic = NEVPT2_TRY(r.str(8));
  SrcGoldFile s;
  if (magic == "NEVPT2S1" || magic == "NEVPT2SD")
    s.scheme = SourceScheme::Sc;
  else if (magic == "NEVPT2P1" || magic == "NEVPT2PD")
    s.scheme = SourceScheme::Pc;
  else
    return err_io("source-term oracle: bad magic (got '" + magic +
                  "', expected 'NEVPT2S1', 'NEVPT2P1', 'NEVPT2SD' or 'NEVPT2PD') -- " + path);
  s.flavour = magic.ends_with('D') ? SourceFlavour::Directional : SourceFlavour::Full;
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
  // A fixed format per flavour: every array is read, so every one is
  // required here, at the shape the header says, and nothing downstream
  // re-checks. -1 / -2 / -3 / -4 are the free direction and rung counts.
  if (s.flavour == SourceFlavour::Full) {
    const std::array<std::pair<std::string_view, std::vector<int64_t>>, 5> want{{
        {"a_corr", {s.nmo, s.nmo}},
        {"act_rot", {s.nact, s.nact}},
        {"py", {s.ndet}},
        {"ci_mask", {s.ndet}},
        {"ci", {s.ndet}},
    }};
    NEVPT2_TRY(requireSrcGoldArrays(s, want, path));
  } else {
    const std::array<std::pair<std::string_view, std::vector<int64_t>>, 9> want{{
        {"kappa_dirs", {-1, s.nmo, s.nmo}},
        {"kappa_block", {-1}},
        {"kappa_fd", {-1, -2}},
        {"kappa_val", {-1}},
        {"ci_dirs", {-3, s.ndet}},
        {"ci_fd", {-3, -4}},
        {"ci_val", {-3}},
        {"act_rot", {s.nact, s.nact}},
        {"ci", {s.ndet}},
    }};
    NEVPT2_TRY(requireSrcGoldArrays(s, want, path));
    NEVPT2_TRY(checkDirectionalArrays(s, path));
  }
  return s;
}

}  // namespace nevpt2
