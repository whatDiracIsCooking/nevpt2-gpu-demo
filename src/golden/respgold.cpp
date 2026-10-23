// loadRespGold: the `*.respgold` CASSCF response sidecar reader (#219's file,
// read on the GPU side by #222). An implementation unit of nevpt2.golden, so
// it shares the one byte-level cursor and array parser with loadGolden
// (`import :reader;`) -- the formats differ only in their magic, their scalar
// header and which arrays they hold (reference_data/golden_format.py's
// RESPGOLD_HEADERS / RESPGOLD_ARRAYS).
module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.golden;

import std;
import :reader;

namespace nevpt2 {

std::string respGoldShapeText(const std::vector<int64_t>& dims) {
  std::string s;
  for (const int64_t d : dims) s += (s.empty() ? "" : ", ") + std::to_string(d);
  return s;
}

// Bit-for-bit equality of two host tensors of one shape.
bool respGoldBitEqual(const Tensor& a, const Tensor& b) {
  if (a.dims() != b.dims()) return false;
  for (int64_t i = 0; i < a.size(); ++i)
    if (std::bit_cast<std::uint64_t>(a.flat(i)) != std::bit_cast<std::uint64_t>(b.flat(i)))
      return false;
  return true;
}

Status RespGoldFile::checkAgainst(const GoldenFile& golden, const GradGoldFile& sidecar) const {
  if (nact != golden.ncas || ncore != golden.ncore || nelecA != golden.nelecA ||
      nelecB != golden.nelecB)
    return err_io(std::format(
        "response sidecar: its state is {} core / {} active orbitals with ({}, {}) active "
        "electrons, but the golden it was given with has {} / {} with ({}, {}) -- the two files "
        "do not describe the same state",
        ncore, nact, nelecA, nelecB, golden.ncore, golden.ncas, golden.nelecA, golden.nelecB));
  if (sidecar.densityFitted())
    return err_io(
        "response sidecar: the gradient sidecar beside it is density-fitted, but the responses "
        "are a conventional CASSCF state's");
  if (nao != sidecar.nao || nmo != sidecar.nmo || nvirt != sidecar.nvirt)
    return err_io(std::format(
        "response sidecar: nao / nmo / nvirt = {} / {} / {}, but the gradient sidecar beside it "
        "has {} / {} / {}",
        nao, nmo, nvirt, sidecar.nao, sidecar.nmo, sidecar.nvirt));
  const auto ci = golden.arrays.find("ci");
  if (ci == golden.arrays.end() || !respGoldBitEqual(get("ci"), ci->second))
    return err_io("response sidecar: its ci is not the golden's, bit for bit -- the responses "
                  "are written about another CI vector");
  const auto side = sidecar.arrays.find("mo_coeff");
  if (side == sidecar.arrays.end() || !respGoldBitEqual(get("mo_coeff"), side->second))
    return err_io("response sidecar: its mo_coeff is not the gradient sidecar's, bit for bit -- "
                  "the responses are written in another orbital basis");
  return {};
}

Tensor RespGoldFile::kappa(const int64_t k) const {
  check(k >= 0 && k < ncoord(), std::format("RespGoldFile::kappa: no coordinate {}", k));
  const Tensor& all = get("kappa");
  const int64_t n = nmo * nmo;
  std::vector<double> data(all.data().begin() + k * n, all.data().begin() + (k + 1) * n);
  return Tensor::fromFlat({nmo, nmo}, std::move(data));
}

Tensor RespGoldFile::ciResponse(const int64_t k) const {
  check(k >= 0 && k < ncoord(), std::format("RespGoldFile::ciResponse: no coordinate {}", k));
  const Tensor& all = get("ci_response");
  std::vector<double> data(all.data().begin() + k * ndet, all.data().begin() + (k + 1) * ndet);
  return Tensor::fromFlat(get("ci").dims(), std::move(data));
}

const Tensor& RespGoldFile::get(const std::string& name) const {
  auto it = arrays.find(name);
  if (it == arrays.end())
    check(false, std::format("response sidecar: array '{}' read but not carried", name));
  return it->second;
}

Result<RespGoldFile> loadRespGold(const std::string& path) {
  Reader r = NEVPT2_TRY(Reader::open(path, "response sidecar"));
  const std::string magic = NEVPT2_TRY(r.str(8));
  if (magic != "NEVPT2R1")
    return err_io("response sidecar: bad magic (got '" + magic + "', expected 'NEVPT2R1') -- " +
                  path);
  RespGoldFile p;
  p.natm = NEVPT2_TRY(r.i32());
  p.nao = NEVPT2_TRY(r.i32());
  p.nmo = NEVPT2_TRY(r.i32());
  p.ncore = NEVPT2_TRY(r.i32());
  p.nact = NEVPT2_TRY(r.i32());
  p.nvirt = NEVPT2_TRY(r.i32());
  p.nelecA = NEVPT2_TRY(r.i32());
  p.nelecB = NEVPT2_TRY(r.i32());
  p.ndet = NEVPT2_TRY(r.i64());
  p.eCasscf = NEVPT2_TRY(r.f64());
  if (p.natm < 1 || p.nao < 1 || p.nmo < 1 || p.ncore < 0 || p.nact < 1 || p.nvirt < 0 ||
      p.nelecA < 0 || p.nelecB < 0 || p.ndet < 1)
    return err_io(std::format(
        "response sidecar: header natm = {}, nao = {}, nmo = {}, ncore = {}, nact = {}, nvirt = "
        "{}, nelec = ({}, {}), ndet = {} is not a usable state -- {}",
        p.natm, p.nao, p.nmo, p.ncore, p.nact, p.nvirt, p.nelecA, p.nelecB, p.ndet, path));
  if (p.ncore + p.nact + p.nvirt != p.nmo)
    return err_io(std::format(
        "response sidecar: header ncore + nact + nvirt = {} + {} + {} = {}, but nmo = {} -- {}",
        p.ncore, p.nact, p.nvirt, p.ncore + p.nact + p.nvirt, p.nmo, path));
  NEVPT2_TRY(readArraySection(r, p.arrays));

  const int64_t nc = p.ncoord();
  const std::array<std::pair<std::string_view, std::vector<int64_t>>, 8> want{{
      {"mo_coeff", {p.nao, p.nmo}},
      {"ci", {}},  // the golden's (nstringsA, nstringsB): its size is checked
      {"kappa", {nc, p.nmo, p.nmo}},
      {"ci_response", {nc, p.ndet}},
      {"hessian", {nc, nc}},
      {"l_rr", {nc, nc}},
      {"gradient", {nc}},
      {"kept", {nc}},
  }};
  for (const auto& [name, dims] : want) {
    const auto it = p.arrays.find(std::string(name));
    if (it == p.arrays.end())
      return err_io(std::format(
          "response sidecar: missing required array '{}'; refused whole -- {}", name, path));
    const bool ok = name == "ci" ? (it->second.rank() == 2 && it->second.size() == p.ndet)
                                 : it->second.dims() == dims;
    if (!ok)
      return err_io(std::format(
          "response sidecar: {} has shape [{}], which the header (ndet = {}) does not allow -- {}",
          name, respGoldShapeText(it->second.dims()), p.ndet, path));
  }
  if (p.arrays.size() != want.size())
    for (const auto& [name, t] : p.arrays)
      if (std::ranges::find(want, std::string_view(name),
                            &std::pair<std::string_view, std::vector<int64_t>>::first) ==
          want.end())
        return err_io(std::format(
            "response sidecar: array '{}' is not one of the format's -- {}", name, path));
  return p;
}

}  // namespace nevpt2
