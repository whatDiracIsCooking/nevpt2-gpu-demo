module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.golden;

import std;
import :reader;

namespace nevpt2 {

// The two magics loadGolden accepts. "NEVPT2G2" carries the reference_kind
// field; "NEVPT2G1" is the layout from before it, which every committed
// golden is written in and which means CASCI -- so the older one stays
// readable and no committed file has to be rewritten
// (docs/reference-data.md, "The file format").
constexpr std::string_view kGoldenMagic = "NEVPT2G2";
constexpr std::string_view kGoldenMagicV1 = "NEVPT2G1";

std::string_view referenceKindName(const ReferenceKind kind) {
  switch (kind) {
    case ReferenceKind::Casscf:
      return "CASSCF";
    case ReferenceKind::Casci:
      break;
  }
  // loadGolden refuses every other code as an IO Error, so a GoldenFile
  // holding one could only have been built by us: our bug.
  check(kind == ReferenceKind::Casci, "referenceKindName: unknown ReferenceKind");
  return "CASCI";
}

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

Status GoldenFile::requireCasciReference(const std::string_view what) const {
  if (reference == ReferenceKind::Casci) return {};
  return err_unsupported(std::format(
      "{}: this golden's reference is {}, and every derivative in this tree is derived for a "
      "CASCI diagonalisation on converged RHF orbitals -- its orbital constraint is RHF "
      "Brillouin/CPHF plus an active-space selection, not CASSCF stationarity, so the "
      "Lagrangian is a different one (docs/gradient-theory.md, §1.1, §1.2 and §5: CPHF, not "
      "CP-CASSCF). A CASSCF reference needs CP-CASSCF, and differentiating one numerically "
      "needs a different strategy as well, because a displaced CASSCF lands on a "
      "differently-rotated solution (§4.4). Not supported; use a CASCI golden.",
      what, referenceKindName(reference)));
}

const Tensor& GoldenFile::get(const std::string& name) const {
  auto it = arrays.find(name);
  if (it == arrays.end())
    check(false, std::format("golden file: array '{}' read without being require()d", name));
  return it->second;
}

Result<GoldenFile> loadGolden(const std::string& path) {
  Reader r = NEVPT2_TRY(Reader::open(path, "golden file"));
  const std::string magic = NEVPT2_TRY(r.str(8));
  // Two layouts, differing only in the reference_kind field at the end of the
  // scalar header: G2 carries it, G1 predates it and is a CASCI.
  const bool hasReferenceKind = (magic == kGoldenMagic);
  if (!hasReferenceKind && magic != kGoldenMagicV1)
    return err_io(std::format("golden file: bad magic (got '{}', expected '{}' or the pre-kind "
                              "'{}') -- {}",
                              magic, kGoldenMagic, kGoldenMagicV1, path));

  GoldenFile g;
  g.ncas = NEVPT2_TRY(r.i32());
  g.nelecA = NEVPT2_TRY(r.i32());
  g.nelecB = NEVPT2_TRY(r.i32());
  g.ncore = NEVPT2_TRY(r.i32());
  g.ndet = NEVPT2_TRY(r.i64());
  g.eCasci = NEVPT2_TRY(r.f64());
  g.eNevpt2Total = NEVPT2_TRY(r.f64());
  if (hasReferenceKind) {
    // The one enumerated field either format has, so an unknown code is a
    // file this reader cannot interpret -- an IO Error like a bad magic, not
    // an Unsupported one: the gate on the kinds it *does* know is
    // requireCasciReference's job.
    const int64_t kind = NEVPT2_TRY(r.i32());
    if (kind != static_cast<int64_t>(ReferenceKind::Casci) &&
        kind != static_cast<int64_t>(ReferenceKind::Casscf))
      return err_io(std::format(
          "golden file: header reference_kind = {} is not a kind this reader knows ({} = CASCI, "
          "{} = CASSCF) -- {}",
          kind, static_cast<int>(ReferenceKind::Casci), static_cast<int>(ReferenceKind::Casscf),
          path));
    g.reference = static_cast<ReferenceKind>(kind);
  }

  // Wherever the header read above stopped -- derived, so a layout change is
  // made in the reads above and nowhere else.
  g.arraysAt = r.position();
  NEVPT2_TRY(readArraySection(r, g.arrays));
  // The header's determinant count is the CI vector's length.
  if (const auto ci = g.arrays.find("ci"); ci != g.arrays.end() && ci->second.size() != g.ndet)
    return err_io(std::format("golden file: ci has {} elements but the header says ndet = {} -- {}",
                              ci->second.size(), g.ndet, path));
  return g;
}

}  // namespace nevpt2
