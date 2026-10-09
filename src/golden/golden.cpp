module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.golden;

import std;
import :reader;

namespace nevpt2 {

Status GoldenFile::require(const std::initializer_list<std::string_view> names) const {
  for (const std::string_view name : names) {
    if (!arrays.contains(std::string(name)))
      return err_io(std::format("golden file: missing required array '{}'", name));
  }
  return {};
}

Status GoldenFile::checkNdet(const int64_t expected) const {
  if (ndet != expected)
    return err_io(std::format(
        "golden file: header ndet = {}, but CAS({},{}) with {} alpha / {} beta electrons has {} "
        "determinants",
        ndet, nelecA + nelecB, ncas, nelecA, nelecB, expected));
  return {};
}

const Tensor& GoldenFile::get(const std::string& name) const {
  auto it = arrays.find(name);
  if (it == arrays.end())
    check(false, std::format("golden file: array '{}' read without being require()d", name));
  return it->second;
}

Result<GoldenFile> loadGolden(const std::string& path) {
  Reader r = NEVPT2_TRY(Reader::open(path));
  const std::string magic = NEVPT2_TRY(r.str(8));
  if (magic != "NEVPT2G1")
    return err_io("golden file: bad magic (got '" + magic + "', expected 'NEVPT2G1') -- " + path);

  GoldenFile g;
  g.ncas = NEVPT2_TRY(r.i32());
  g.nelecA = NEVPT2_TRY(r.i32());
  g.nelecB = NEVPT2_TRY(r.i32());
  g.ncore = NEVPT2_TRY(r.i32());
  g.ndet = NEVPT2_TRY(r.i64());
  g.eCasci = NEVPT2_TRY(r.f64());
  g.eNevpt2Total = NEVPT2_TRY(r.f64());

  const int64_t nArrays = NEVPT2_TRY(r.i32());
  if (nArrays < 0)
    return err_io(std::format("golden file: negative array count {} -- {}", nArrays, path));
  for (int64_t a = 0; a < nArrays; ++a) NEVPT2_TRY(readArray(r, g));
  // The writer stops after the last array, so anything left is a truncated
  // or miscounted header, not padding.
  if (r.remaining() != 0)
    return err_io(std::format("golden file: {} bytes after the last of its {} arrays -- {}",
                              r.remaining(), nArrays, path));
  // The header's determinant count is the CI vector's length.
  if (const auto ci = g.arrays.find("ci"); ci != g.arrays.end() && ci->second.size() != g.ndet)
    return err_io(std::format("golden file: ci has {} elements but the header says ndet = {} -- {}",
                              ci->second.size(), g.ndet, path));
  return g;
}

}  // namespace nevpt2
