// writeParHess / loadParHess: the `*.parhess` parameter Hessian sidecar (#222),
// the GPU's bilinear form E_thth[Theta_A, Theta_B] over a response sidecar's
// directions. The one format in golden/'s family the C++ side WRITES: the
// same magic + scalar header + named-array section as every other
// (reference_data/golden_format.py's PARHESS_HEADERS / PARHESS_ARRAYS, whose
// write_parhess writes the same bytes and whose read_parhess is the format's
// Python reader). An implementation unit of nevpt2.golden, so the reader
// shares the one byte-level cursor and array parser (`import :reader;`).
module;

#include <cstdio>  // FILE*, fopen/fwrite/fclose: the writer's

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.golden;

import std;
import :reader;

namespace nevpt2 {

// The four matrices, in file order after `coords`, and which field of
// ParHessFile each is.
constexpr std::array<std::string_view, 4> kParHessMatrices{"hessian", "h_kk", "h_kc", "h_cc"};

const Tensor& parHessMatrix(const ParHessFile& p, const std::size_t i) {
  return i == 0 ? p.hessian : i == 1 ? p.kk : i == 2 ? p.kc : p.cc;
}

Tensor& parHessMatrix(ParHessFile& p, const std::size_t i) {
  return i == 0 ? p.hessian : i == 1 ? p.kk : i == 2 ? p.kc : p.cc;
}

// Whether `coords` is a strictly increasing list within [0, 3 natm).
bool parHessCoordsValid(const std::vector<int64_t>& coords, const int64_t natm) {
  for (std::size_t i = 0; i < coords.size(); ++i)
    if (coords[i] < 0 || coords[i] >= 3 * natm || (i > 0 && coords[i] <= coords[i - 1]))
      return false;
  return !coords.empty();
}

Status writeParHess(const std::string& path, const ParHessFile& p) {
  const auto n = static_cast<int64_t>(p.coords.size());
  check(p.natm >= 1 && p.ncore >= 0 && p.nact >= 1 && p.nvirt >= 0 && p.ndet >= 1 &&
            p.ncore + p.nact + p.nvirt == p.nmo,
        "writeParHess: the header's partition does not add up");
  check(parHessCoordsValid(p.coords, p.natm),
        "writeParHess: the coordinates are not strictly increasing within [0, 3 natm)");
  for (std::size_t i = 0; i < kParHessMatrices.size(); ++i)
    check(parHessMatrix(p, i).dims() == std::vector<int64_t>{n, n},
          std::format("writeParHess: {} is not ({}, {})", kParHessMatrices[i], n, n));

  // Built whole in memory first: the file is tiny, and one fwrite either
  // writes all of it or reports the failure.
  std::vector<unsigned char> buf;
  const auto put = [&buf](const void* src, const std::size_t bytes) {
    const auto* b = static_cast<const unsigned char*>(src);
    buf.insert(buf.end(), b, b + bytes);
  };
  const auto putI32 = [&put](const int64_t v) {
    const auto x = static_cast<std::int32_t>(v);
    put(&x, 4);
  };
  const auto putI64 = [&put](const int64_t v) {
    const auto x = static_cast<std::int64_t>(v);
    put(&x, 8);
  };
  const auto putArray = [&](const std::string_view name, const std::vector<int64_t>& dims,
                            const double* data, const int64_t count) {
    putI32(static_cast<int64_t>(name.size()));
    put(name.data(), name.size());
    putI32(static_cast<int64_t>(dims.size()));
    for (const int64_t d : dims) putI64(d);
    put(data, static_cast<std::size_t>(count) * sizeof(double));
  };
  put("NEVPT2H1", 8);
  for (const int64_t v : {p.natm, p.nmo, p.ncore, p.nact, p.nvirt}) putI32(v);
  putI64(p.ndet);
  putI32(static_cast<int64_t>(1 + kParHessMatrices.size()));
  std::vector<double> coords(p.coords.begin(), p.coords.end());
  putArray("coords", {n}, coords.data(), n);
  for (std::size_t i = 0; i < kParHessMatrices.size(); ++i) {
    const Tensor& t = parHessMatrix(p, i);
    putArray(kParHessMatrices[i], t.dims(), t.data().data(), t.size());
  }

  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return err_io("could not open the parameter Hessian sidecar for writing: " + path);
  const bool wrote = std::fwrite(buf.data(), 1, buf.size(), f) == buf.size();
  const bool closed = std::fclose(f) == 0;
  if (!wrote || !closed)
    return err_io("could not write the parameter Hessian sidecar: " + path);
  return {};
}

Result<ParHessFile> loadParHess(const std::string& path) {
  Reader r = NEVPT2_TRY(Reader::open(path, "parameter Hessian sidecar"));
  const std::string magic = NEVPT2_TRY(r.str(8));
  if (magic != "NEVPT2H1")
    return err_io("parameter Hessian sidecar: bad magic (got '" + magic +
                  "', expected 'NEVPT2H1') -- " + path);
  ParHessFile p;
  p.natm = NEVPT2_TRY(r.i32());
  p.nmo = NEVPT2_TRY(r.i32());
  p.ncore = NEVPT2_TRY(r.i32());
  p.nact = NEVPT2_TRY(r.i32());
  p.nvirt = NEVPT2_TRY(r.i32());
  p.ndet = NEVPT2_TRY(r.i64());
  if (p.natm < 1 || p.nmo < 1 || p.ncore < 0 || p.nact < 1 || p.nvirt < 0 || p.ndet < 1 ||
      p.ncore + p.nact + p.nvirt != p.nmo)
    return err_io(std::format(
        "parameter Hessian sidecar: header natm = {}, nmo = {}, ncore = {}, nact = {}, nvirt = "
        "{}, ndet = {} is not a usable partition -- {}",
        p.natm, p.nmo, p.ncore, p.nact, p.nvirt, p.ndet, path));
  std::unordered_map<std::string, Tensor> arrays;
  NEVPT2_TRY(readArraySection(r, arrays));
  const auto c = arrays.find("coords");
  if (c == arrays.end() || c->second.rank() != 1)
    return err_io("parameter Hessian sidecar: no rank-1 'coords' array; refused whole -- " +
                  path);
  for (int64_t i = 0; i < c->second.size(); ++i) {
    const double v = c->second.flat(i);
    if (v != std::floor(v))
      return err_io("parameter Hessian sidecar: a coordinate index is not an integer -- " + path);
    p.coords.push_back(static_cast<int64_t>(v));
  }
  if (!parHessCoordsValid(p.coords, p.natm))
    return err_io(std::format(
        "parameter Hessian sidecar: its coordinates are not strictly increasing within [0, {}) "
        "-- {}",
        3 * p.natm, path));
  const auto n = static_cast<int64_t>(p.coords.size());
  for (std::size_t i = 0; i < kParHessMatrices.size(); ++i) {
    const auto it = arrays.find(std::string(kParHessMatrices[i]));
    if (it == arrays.end())
      return err_io(std::format(
          "parameter Hessian sidecar: missing required array '{}'; refused whole -- {}",
          kParHessMatrices[i], path));
    if (it->second.dims() != std::vector<int64_t>{n, n})
      return err_io(std::format(
          "parameter Hessian sidecar: {} is not ({}, {}), as its {} coordinates say -- {}",
          kParHessMatrices[i], n, n, n, path));
    parHessMatrix(p, i) = it->second;
  }
  if (arrays.size() != 1 + kParHessMatrices.size())
    for (const auto& [name, t] : arrays)
      if (name != "coords" &&
          std::ranges::find(kParHessMatrices, std::string_view(name)) == kParHessMatrices.end())
        return err_io(std::format(
            "parameter Hessian sidecar: array '{}' is not one of the format's -- {}", name, path));
  return p;
}

}  // namespace nevpt2
