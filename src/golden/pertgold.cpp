// loadPertGold: the `*.pertgold` integral perturbation sidecar reader (#220).
// An implementation unit of nevpt2.golden, so it shares the one byte-level
// cursor and array parser with loadGolden (`import :reader;`) -- the formats
// differ only in their magic, their scalar header and which arrays they hold
// (reference_data/golden_format.py's PERTGOLD_HEADERS / PERTGOLD_ARRAYS).
module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.golden;

import std;
import :reader;

namespace nevpt2 {

std::span<const std::string_view> PertGoldFile::inventoryNames() {
  static constexpr std::array<std::string_view, 21> kNames{
      "h",       "heff",    "f",       "g_xa_aa", "g_xv_aa",   "g_xc_aa",   "g_xa_va",
      "g_xv_va", "g_xc_va", "g_xa_ca", "g_xv_ca", "g_xc_ca",   "g_xa_cv",   "g_xv_cv",
      "g_xc_cv", "g_xa_cc", "g_xv_cc", "g_xc_cc_j", "g_xc_cc_k", "g_xa_vv", "g_xc_vv"};
  return kNames;
}

// Every array's shape, the coordinate axis included: the name's letters after
// `g_x` spell its spaces (GradGoldFile's convention), the tied rows have one
// label fewer.
std::vector<int64_t> pertGoldShape(const PertGoldFile& p, const std::string_view name) {
  if (name == "mo_coeff") return {p.nao, p.nmo};
  const int64_t nc = p.ncoord();
  if (name == "h" || name == "heff" || name == "f") return {nc, p.nmo, p.nmo};
  const auto ext = [&p](const char s) {
    return s == 'c' ? p.ncore : (s == 'a' ? p.nact : p.nvirt);
  };
  if (name == "g_xc_cc_j" || name == "g_xc_cc_k") return {nc, p.nmo, p.ncore, p.ncore};
  if (name == "g_xa_vv") return {nc, p.nmo, p.nact, p.nvirt};
  if (name == "g_xc_vv") return {nc, p.nmo, p.ncore, p.nvirt};
  // g_xP_QR: x, then P, Q, R.
  return {nc, p.nmo, ext(name[3]), ext(name[5]), ext(name[6])};
}

std::string pertGoldShapeText(const std::vector<int64_t>& dims) {
  std::string s;
  for (const int64_t d : dims) s += (s.empty() ? "" : ", ") + std::to_string(d);
  return s;
}

Status PertGoldFile::checkAgainst(const GoldenFile& golden, const GradGoldFile& sidecar) const {
  if (nact != golden.ncas || ncore != golden.ncore)
    return err_io(std::format(
        "integral perturbation sidecar: its partition is {} core / {} active orbitals, but the "
        "golden it was given with has {} / {} -- the two files do not describe the same state",
        ncore, nact, golden.ncore, golden.ncas));
  if (nao != sidecar.nao || nmo != sidecar.nmo || nvirt != sidecar.nvirt)
    return err_io(std::format(
        "integral perturbation sidecar: nao / nmo / nvirt = {} / {} / {}, but the gradient "
        "sidecar beside it has {} / {} / {}",
        nao, nmo, nvirt, sidecar.nao, sidecar.nmo, sidecar.nvirt));
  if (densityFitted() != sidecar.densityFitted() || naux != sidecar.naux)
    return err_io(std::format(
        "integral perturbation sidecar: it perturbs the {} integrals (naux = {}), but the "
        "gradient sidecar beside it is {} (naux = {}) -- a derivative needs both from one "
        "calculation",
        densityFitted() ? "density-fitted" : "conventional", naux,
        sidecar.densityFitted() ? "density-fitted" : "conventional", sidecar.naux));
  const auto side = sidecar.arrays.find("mo_coeff");
  if (side == sidecar.arrays.end())
    return err_io("integral perturbation sidecar: the gradient sidecar beside it carries no "
                  "mo_coeff to hold its frame to");
  const Tensor& mine = get("mo_coeff");
  bool same = side->second.dims() == mine.dims();
  for (int64_t i = 0; same && i < mine.size(); ++i)
    same = std::bit_cast<std::uint64_t>(mine.flat(i)) ==
           std::bit_cast<std::uint64_t>(side->second.flat(i));
  if (!same)
    return err_io("integral perturbation sidecar: its mo_coeff is not the gradient sidecar's, "
                  "bit for bit -- the perturbation is written in another orbital basis");
  return {};
}

Tensor PertGoldFile::coordinate(const std::string& name, const int64_t k) const {
  const Tensor& all = get(name);
  check(name != "mo_coeff" && k >= 0 && k < ncoord() && all.rank() >= 2 &&
            all.dim(0) == ncoord(),
        std::format("PertGoldFile::coordinate: '{}' has no coordinate {}", name, k));
  std::vector<int64_t> dims(all.dims().begin() + 1, all.dims().end());
  const int64_t n = all.size() / ncoord();
  std::vector<double> data(all.data().begin() + k * n, all.data().begin() + (k + 1) * n);
  return Tensor::fromFlat(dims, std::move(data));
}

const Tensor& PertGoldFile::get(const std::string& name) const {
  auto it = arrays.find(name);
  if (it == arrays.end())
    check(false,
          std::format("integral perturbation sidecar: array '{}' read but not carried", name));
  return it->second;
}

Result<PertGoldFile> loadPertGold(const std::string& path) {
  Reader r = NEVPT2_TRY(Reader::open(path, "integral perturbation sidecar"));
  const std::string magic = NEVPT2_TRY(r.str(8));
  if (magic != "NEVPT2I1")
    return err_io("integral perturbation sidecar: bad magic (got '" + magic +
                  "', expected 'NEVPT2I1') -- " + path);
  PertGoldFile p;
  p.natm = NEVPT2_TRY(r.i32());
  p.nao = NEVPT2_TRY(r.i32());
  p.nmo = NEVPT2_TRY(r.i32());
  p.ncore = NEVPT2_TRY(r.i32());
  p.nact = NEVPT2_TRY(r.i32());
  p.nvirt = NEVPT2_TRY(r.i32());
  p.naux = NEVPT2_TRY(r.i32());
  if (p.natm < 1 || p.nao < 1 || p.nmo < 1 || p.ncore < 0 || p.nact < 1 || p.nvirt < 0 ||
      p.naux < 0)
    return err_io(std::format(
        "integral perturbation sidecar: header natm = {}, nao = {}, nmo = {}, ncore = {}, nact "
        "= {}, nvirt = {}, naux = {} is not a usable partition -- {}",
        p.natm, p.nao, p.nmo, p.ncore, p.nact, p.nvirt, p.naux, path));
  if (p.ncore + p.nact + p.nvirt != p.nmo)
    return err_io(std::format(
        "integral perturbation sidecar: header ncore + nact + nvirt = {} + {} + {} = {}, but "
        "nmo = {} -- {}",
        p.ncore, p.nact, p.nvirt, p.ncore + p.nact + p.nvirt, p.nmo, path));
  NEVPT2_TRY(readArraySection(r, p.arrays));

  // The whole sidecar or none of it: every array present, nothing else.
  std::vector<std::string_view> names{"mo_coeff"};
  for (const std::string_view n : PertGoldFile::inventoryNames()) names.push_back(n);
  for (const std::string_view n : names)
    if (!p.arrays.contains(std::string(n)))
      return err_io(std::format(
          "integral perturbation sidecar: missing required array '{}'; a perturbation missing "
          "any array is refused whole, never read in part -- {}",
          n, path));
  for (const auto& [name, t] : p.arrays) {
    if (std::ranges::find(names, std::string_view(name)) == names.end())
      return err_io(std::format(
          "integral perturbation sidecar: array '{}' is not one of the format's -- {}", name,
          path));
    const std::vector<int64_t> want = pertGoldShape(p, name);
    if (t.dims() != want)
      return err_io(std::format(
          "integral perturbation sidecar: {} has shape [{}] but the header says [{}] -- {}",
          name, pertGoldShapeText(t.dims()), pertGoldShapeText(want), path));
  }
  return p;
}

}  // namespace nevpt2
