// loadEriGold: the `*.erigold` full MO-integral tensor sidecar reader (#180).
// An implementation unit of nevpt2.golden, so it shares the one byte-level
// cursor and array parser with loadGolden (`import :reader;`) -- the formats
// differ only in their magic, their scalar header and which arrays they hold
// (reference_data/golden_format.py's ERIGOLD_HEADERS / ERIGOLD_ARRAYS).
module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.golden;

import std;
import :reader;

namespace nevpt2 {

std::string eriGoldShapeText(const std::vector<int64_t>& dims) {
  std::string s;
  for (const int64_t d : dims) s += (s.empty() ? "" : ", ") + std::to_string(d);
  return s;
}

Status EriGoldFile::checkAgainst(const GoldenFile& golden, const GradGoldFile& sidecar) const {
  if (nact != golden.ncas || ncore != golden.ncore)
    return err_io(std::format(
        "full MO-integral tensor sidecar: its partition is {} core / {} active orbitals, but "
        "the golden it was given with has {} / {} -- the two files do not describe the same "
        "state",
        ncore, nact, golden.ncore, golden.ncas));
  if (sidecar.densityFitted() || golden.arrays.contains("B_aa"))
    return err_io(
        "full MO-integral tensor sidecar: it holds the CONVENTIONAL integrals, but the golden "
        "or gradient sidecar beside it is density-fitted -- a fitted case's B[L, p, q] is "
        "already its whole tensor, and the two calculations' integrals do not mix");
  if (nao != sidecar.nao || nmo != sidecar.nmo || nvirt != sidecar.nvirt ||
      ncore != sidecar.ncore || nact != sidecar.nact)
    return err_io(std::format(
        "full MO-integral tensor sidecar: nao / nmo / ncore / nact / nvirt = {} / {} / {} / {} "
        "/ {}, but the gradient sidecar beside it has {} / {} / {} / {} / {}",
        nao, nmo, ncore, nact, nvirt, sidecar.nao, sidecar.nmo, sidecar.ncore, sidecar.nact,
        sidecar.nvirt));
  const auto side = sidecar.arrays.find("mo_coeff");
  if (side == sidecar.arrays.end())
    return err_io("full MO-integral tensor sidecar: the gradient sidecar beside it carries no "
                  "mo_coeff to hold its frame to");
  const Tensor& mine = get("mo_coeff");
  bool same = side->second.dims() == mine.dims();
  for (int64_t i = 0; same && i < mine.size(); ++i)
    same = std::bit_cast<std::uint64_t>(mine.flat(i)) ==
           std::bit_cast<std::uint64_t>(side->second.flat(i));
  if (!same)
    return err_io("full MO-integral tensor sidecar: its mo_coeff is not the gradient sidecar's, "
                  "bit for bit -- the integrals are written in another orbital basis");
  return {};
}

const Tensor& EriGoldFile::get(const std::string& name) const {
  auto it = arrays.find(name);
  if (it == arrays.end())
    check(false,
          std::format("full MO-integral tensor sidecar: array '{}' read but not carried", name));
  return it->second;
}

Result<EriGoldFile> loadEriGold(const std::string& path) {
  Reader r = NEVPT2_TRY(Reader::open(path, "full MO-integral tensor sidecar"));
  const std::string magic = NEVPT2_TRY(r.str(8));
  if (magic != "NEVPT2E1")
    return err_io("full MO-integral tensor sidecar: bad magic (got '" + magic +
                  "', expected 'NEVPT2E1') -- " + path);
  EriGoldFile e;
  e.nao = NEVPT2_TRY(r.i32());
  e.nmo = NEVPT2_TRY(r.i32());
  e.ncore = NEVPT2_TRY(r.i32());
  e.nact = NEVPT2_TRY(r.i32());
  e.nvirt = NEVPT2_TRY(r.i32());
  if (e.nao < 1 || e.nmo < 1 || e.ncore < 0 || e.nact < 1 || e.nvirt < 0)
    return err_io(std::format(
        "full MO-integral tensor sidecar: header nao = {}, nmo = {}, ncore = {}, nact = {}, "
        "nvirt = {} is not a usable partition -- {}",
        e.nao, e.nmo, e.ncore, e.nact, e.nvirt, path));
  if (e.ncore + e.nact + e.nvirt != e.nmo)
    return err_io(std::format(
        "full MO-integral tensor sidecar: header ncore + nact + nvirt = {} + {} + {} = {}, but "
        "nmo = {} -- {}",
        e.ncore, e.nact, e.nvirt, e.ncore + e.nact + e.nvirt, e.nmo, path));
  NEVPT2_TRY(readArraySection(r, e.arrays));

  const std::array<std::pair<std::string_view, std::vector<int64_t>>, 2> want{{
      {"mo_coeff", {e.nao, e.nmo}},
      {"eri", {e.nmo, e.nmo, e.nmo, e.nmo}},
  }};
  for (const auto& [name, dims] : want) {
    const auto it = e.arrays.find(std::string(name));
    if (it == e.arrays.end())
      return err_io(std::format(
          "full MO-integral tensor sidecar: missing required array '{}'; refused whole -- {}",
          name, path));
    if (it->second.dims() != dims)
      return err_io(std::format(
          "full MO-integral tensor sidecar: {} has shape [{}] but the header says [{}] -- {}",
          name, eriGoldShapeText(it->second.dims()), eriGoldShapeText(dims), path));
  }
  if (e.arrays.size() != want.size())
    for (const auto& [name, t] : e.arrays)
      if (name != "mo_coeff" && name != "eri")
        return err_io(std::format(
            "full MO-integral tensor sidecar: array '{}' is not one of the format's -- {}", name,
            path));
  return e;
}

}  // namespace nevpt2
