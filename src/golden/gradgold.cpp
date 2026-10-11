// loadGradGold: the `*.gradgold` sidecar reader. An implementation unit of
// nevpt2.golden, so it shares the one byte-level cursor and array parser with
// loadGolden (`import :reader;`) -- the two formats differ only in their magic
// and their scalar header.
module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.golden;

import std;
import :reader;

namespace nevpt2 {

Status GradGoldFile::require(const std::initializer_list<std::string_view> names) const {
  for (const std::string_view name : names) {
    if (!arrays.contains(std::string(name)))
      return err_io(std::format("gradient sidecar: missing required array '{}'", name));
  }
  return {};
}

const Tensor& GradGoldFile::get(const std::string& name) const {
  auto it = arrays.find(name);
  if (it == arrays.end())
    check(false, std::format("gradient sidecar: array '{}' read without being require()d", name));
  return it->second;
}

Result<GradGoldFile> loadGradGold(const std::string& path) {
  Reader r = NEVPT2_TRY(Reader::open(path, "gradient sidecar"));
  const std::string magic = NEVPT2_TRY(r.str(8));
  if (magic != "NEVPT2D1")
    return err_io("gradient sidecar: bad magic (got '" + magic +
                  "', expected 'NEVPT2D1') -- " + path);

  GradGoldFile g;
  g.nao = NEVPT2_TRY(r.i32());
  g.nmo = NEVPT2_TRY(r.i32());
  g.ncore = NEVPT2_TRY(r.i32());
  g.nact = NEVPT2_TRY(r.i32());
  g.nvirt = NEVPT2_TRY(r.i32());
  g.nelecRhf = NEVPT2_TRY(r.i32());
  // Every array's shape is written in terms of this partition, so a header
  // that does not add up would size the shape checks against nothing.
  if (g.nao < 1 || g.nmo < 1 || g.ncore < 0 || g.nact < 1 || g.nvirt < 0)
    return err_io(std::format(
        "gradient sidecar: header nao = {}, nmo = {}, ncore = {}, nact = {}, nvirt = {} is not a "
        "usable orbital-space partition -- {}",
        g.nao, g.nmo, g.ncore, g.nact, g.nvirt, path));
  if (g.ncore + g.nact + g.nvirt != g.nmo)
    return err_io(std::format(
        "gradient sidecar: header ncore + nact + nvirt = {} + {} + {} = {}, but nmo = {} -- {}",
        g.ncore, g.nact, g.nvirt, g.ncore + g.nact + g.nvirt, g.nmo, path));
  // The reference is a closed-shell RHF, and its core is doubly occupied, so
  // the electron count is even and fills at least the core. It is what says
  // which half of F's active spectrum is act_O (gradient-theory.md, §2.6), so
  // a count that cannot mean that is a file-format mismatch.
  if (g.nelecRhf < 2 * g.ncore || g.nelecRhf % 2 != 0 || g.nelecRhf > 2 * (g.ncore + g.nact))
    return err_io(std::format(
        "gradient sidecar: header nelec_rhf = {} is not an even count filling the {} core and at "
        "most the {} active orbitals -- {}",
        g.nelecRhf, g.ncore, g.nact, path));

  NEVPT2_TRY(readArraySection(r, g.arrays));
  // mo_coeff is the one array shaped by nao, so it is the one the header's
  // AO count can be checked against.
  if (const auto mo = g.arrays.find("mo_coeff"); mo != g.arrays.end()) {
    const std::vector<int64_t>& dims = mo->second.dims();
    if (dims != std::vector<int64_t>{g.nao, g.nmo}) {
      std::string got;
      for (const int64_t d : dims) got += (got.empty() ? "" : ", ") + std::to_string(d);
      return err_io(std::format(
          "gradient sidecar: mo_coeff has shape [{}] but the header says [{}, {}] -- {}", got,
          g.nao, g.nmo, path));
    }
  }
  return g;
}

}  // namespace nevpt2
