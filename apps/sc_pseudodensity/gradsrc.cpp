module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.gradsrc;

import std;
import :archive;  // the .npz bytes: NPY members in a stored zip archive

namespace nevpt2 {

// A member from a tensor: its shape and its row-major data, which is C order.
NpyMember floatMember(std::string name, const Tensor& t) {
  NpyMember m;
  m.name = std::move(name);
  m.shape = t.dims();
  m.f64 = t.data();
  return m;
}

NpyMember intMember(std::string name, const int64_t v) {
  NpyMember m;
  m.name = std::move(name);
  m.isInt = true;
  m.i64 = v;
  return m;
}

NpyMember scalarMember(std::string name, const double v) {
  NpyMember m;
  m.name = std::move(name);
  m.f64 = {v};
  return m;
}

Status writeGradSrc(const std::string& path, const GradSrc& src) {
  const int64_t nmo = src.ncore + src.nact + src.nvirt;
  check(src.kappaSource.rank() == 2 && src.kappaSource.dim(0) == nmo &&
            src.kappaSource.dim(1) == nmo,
        "writeGradSrc: kappaSource is not (n_mo, n_mo) for the partition");
  check(src.ciSource.dims() == src.ci.dims(),
        "writeGradSrc: ciSource and ci are not the same shape");
  if (src.hasDirections()) {
    const int64_t ndir = src.edotKappa.rank() == 3 ? src.edotKappa.dim(0) : -1;
    check(src.edotKappa.dims() == std::vector<int64_t>{ndir, nmo, nmo} &&
              src.edotFrozen.dims() == std::vector<int64_t>{ndir} &&
              src.edotDigest.dims() == std::vector<int64_t>{ndir},
          "writeGradSrc: the direction members are not (n_dir, n_mo, n_mo), (n_dir), (n_dir)");
  } else {
    check(src.edotFrozen.size() == 0 && src.edotDigest.size() == 0,
          "writeGradSrc: edot_frozen / edot_digest without edot_kappa");
  }
  std::vector<NpyMember> members{
      floatMember("kappa_source", src.kappaSource),
      floatMember("ci_source", src.ciSource),
      floatMember("ci", src.ci),
      intMember("ncore", src.ncore),
      intMember("nact", src.nact),
      intMember("nvirt", src.nvirt),
      scalarMember("e_corr", src.eCorr),
  };
  if (src.hasDirections()) {
    members.push_back(floatMember("edot_kappa", src.edotKappa));
    members.push_back(floatMember("edot_frozen", src.edotFrozen));
    members.push_back(floatMember("edot_digest", src.edotDigest));
  }
  const std::vector<std::uint8_t> bytes =
      NEVPT2_TRY(npzBytes(members, std::format("gradient source dump {}", path)));
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return err_io(std::format("could not open gradient source dump for writing: {}", path));
  out.write(reinterpret_cast<const char*>(bytes.data()), std::ssize(bytes));
  out.close();
  if (!out) return err_io(std::format("could not write gradient source dump: {}", path));
  return {};
}

// The member called `name`, or an IO Error naming it.
Result<const NpyMember*> findMember(const std::vector<NpyMember>& members,
                                    const std::string_view name, const std::string_view where) {
  const auto it = std::ranges::find(members, name, &NpyMember::name);
  if (it == members.end()) return err_io(std::format("{}: no '{}' member", where, name));
  return &*it;
}

Result<Tensor> floatArray(const std::vector<NpyMember>& members, const std::string_view name,
                          const std::string_view where) {
  const NpyMember* m = NEVPT2_TRY(findMember(members, name, where));
  if (m->isInt || m->shape.empty())
    return err_io(std::format("{}: '{}' is not a float array", where, name));
  return Tensor::fromFlat(m->shape, m->f64);
}

Result<int64_t> intScalar(const std::vector<NpyMember>& members, const std::string_view name,
                          const std::string_view where) {
  const NpyMember* m = NEVPT2_TRY(findMember(members, name, where));
  if (!m->isInt) return err_io(std::format("{}: '{}' is not an integer scalar", where, name));
  return m->i64;
}

Result<double> floatScalar(const std::vector<NpyMember>& members, const std::string_view name,
                           const std::string_view where) {
  const NpyMember* m = NEVPT2_TRY(findMember(members, name, where));
  if (m->isInt || !m->shape.empty())
    return err_io(std::format("{}: '{}' is not a float scalar", where, name));
  return m->f64[0];
}

Result<GradSrc> readGradSrc(const std::string& path) {
  const std::string where = std::format("gradient source dump {}", path);
  std::ifstream in(path, std::ios::binary);
  if (!in) return err_io(std::format("could not open {}", where));
  const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(in),
                                        std::istreambuf_iterator<char>()};
  if (in.bad()) return err_io(std::format("could not read {}", where));
  const std::vector<NpyMember> members = NEVPT2_TRY(parseNpz(bytes, where));
  GradSrc src;
  src.kappaSource = NEVPT2_TRY(floatArray(members, "kappa_source", where));
  src.ciSource = NEVPT2_TRY(floatArray(members, "ci_source", where));
  src.ci = NEVPT2_TRY(floatArray(members, "ci", where));
  src.ncore = NEVPT2_TRY(intScalar(members, "ncore", where));
  src.nact = NEVPT2_TRY(intScalar(members, "nact", where));
  src.nvirt = NEVPT2_TRY(intScalar(members, "nvirt", where));
  src.eCorr = NEVPT2_TRY(floatScalar(members, "e_corr", where));
  const int64_t nmo = src.ncore + src.nact + src.nvirt;
  if (src.kappaSource.rank() != 2 || src.kappaSource.dim(0) != nmo ||
      src.kappaSource.dim(1) != nmo)
    return err_io(std::format("{}: kappa_source is not ({}, {})", where, nmo, nmo));
  if (src.ciSource.dims() != src.ci.dims())
    return err_io(std::format("{}: ci_source and ci are not the same shape", where));
  // The forward Edot(kappa) directions (#88): all three members or none.
  const auto has = [&](std::string_view name) {
    return std::ranges::find(members, name, &NpyMember::name) != members.end();
  };
  const int present = int{has("edot_kappa")} + int{has("edot_frozen")} + int{has("edot_digest")};
  if (present != 0 && present != 3)
    return err_io(std::format(
        "{}: edot_kappa / edot_frozen / edot_digest are a set, and only {} of them is here",
        where, present));
  if (present == 3) {
    src.edotKappa = NEVPT2_TRY(floatArray(members, "edot_kappa", where));
    src.edotFrozen = NEVPT2_TRY(floatArray(members, "edot_frozen", where));
    src.edotDigest = NEVPT2_TRY(floatArray(members, "edot_digest", where));
    const int64_t ndir = src.edotKappa.rank() == 3 ? src.edotKappa.dim(0) : -1;
    if (ndir < 1 || src.edotKappa.dims() != std::vector<int64_t>{ndir, nmo, nmo} ||
        src.edotFrozen.dims() != std::vector<int64_t>{ndir} ||
        src.edotDigest.dims() != std::vector<int64_t>{ndir})
      return err_io(std::format(
          "{}: edot_kappa / edot_frozen / edot_digest are not (n_dir, {}, {}), (n_dir), (n_dir)",
          where, nmo, nmo));
  }
  return src;
}

}  // namespace nevpt2
