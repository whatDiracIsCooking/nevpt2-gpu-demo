// nevpt2.golden -- readers for the two reference files `generate_golden.py`
// writes (see that script's `write_golden` / `write_gradgold` docstrings for
// the exact binary layouts: a magic, a scalar header, then a sequence of named
// N-d f64 arrays).
// Pure host C++, no GPU -- these only need to run once at program start.
//
//   loadGolden     `*.nevpt2gold`: the CASSCF/CASCI input and the golden
//                  SC/PC-NEVPT2 answer every demo checks itself against.
//   loadGradGold   `*.gradgold`: the analytic-gradient SIDECAR beside a
//                  golden, carrying the one-general-index MO-integral
//                  inventory a gradient reads plus the full-MO-range Fock
//                  matrices and mo_coeff.
//
// They are two files and not one on purpose: the golden format is extensible
// (magic + array count), but adding arrays would change every committed
// golden's bytes, and a regenerated golden that differs from the committed one
// is this project's signal that something changed
// (docs/reference-data.md, "Regenerating the golden file"). A sidecar keeps
// that intact and keeps the big goldens from growing.
//
// Both files are the user's input, so every way either can be missing or
// malformed is a value-tier Error (nevpt2.error_handling, ErrorKind::IO)
// handed back to the demo's main() to report once -- never an abort.
export module nevpt2.golden;

import std;
// Re-exported: GoldenFile's arrays are Tensors, and loadGolden/require hand
// back a Result/Status the caller reports.
export import nevpt2.tensor;
export import nevpt2.error_handling;

export namespace nevpt2 {

struct GoldenFile {
  // Stored as int32 (ndet as int64) in the file; held as int64_t like every
  // host size.
  int64_t ncas = 0;
  int64_t nelecA = 0;
  int64_t nelecB = 0;
  int64_t ncore = 0;
  int64_t ndet = 0;
  double eCasci = 0.0;
  double eNevpt2Total = 0.0;
  std::unordered_map<std::string, Tensor> arrays;

  // An IO Error naming the first of `names` the file does not carry. Each
  // demo calls it right after loading, on every array it will get() (the
  // density-fitted demo again for --check-blocks' arrays; --pc's
  // pc_class_energies/e_pc_total are checked separately, in main): a missing
  // one is a file-format mismatch (the user's input), reported before any GPU
  // work.
  [[nodiscard]] Status require(std::initializer_list<std::string_view> names) const;

  // An IO Error unless the header's ndet is `expected`, the determinant count
  // the demo derives from ncas/nelecA/nelecB (it cannot be derived here: the
  // string counts live in link_tables). loadGolden has already checked that
  // ci, when present, holds ndet elements, so passing this ties the CI
  // vector's length to the active space the kernels index it with.
  [[nodiscard]] Status checkNdet(int64_t expected) const;

  // The array `name`. Aborts (check) if it is absent: the demo should have
  // require()d it, so reaching here without it is our bug, not the file's.
  const Tensor& get(const std::string& name) const;
};

// The parsed file, or an IO Error if it cannot be opened, ends early, has the
// wrong magic, or carries a malformed array header (a negative or oversized
// rank, a negative extent, or an element count that overflows or runs past
// the end of the file), has bytes after its last array, or carries a ci
// whose length is not the header's ndet.
[[nodiscard]] Result<GoldenFile> loadGolden(const std::string& path);

// ---------------------------------------------------------------------------
// The gradient sidecar (`*.gradgold`).
// ---------------------------------------------------------------------------

// A golden's gradient sidecar: the orbital-space partition its array shapes
// are written in terms of, and the arrays themselves.
//
// The scalar header is a partition and not a CI state -- the state is the
// golden's job. Every integral array has `nmo` as its leading extent (the one
// index of each that runs over the full MO range), and the rest of its extents
// are `ncore`, `nact` and `nvirt` in the order its name spells:
// `g_xa_cv`, say, is `(x t | i r)` with shape [nmo, nact, ncore, nvirt].
// docs/gradient-theory.md, §3.3 is the inventory, §3.5 the matrices, and
// `generate_golden.py`'s `_gradient_arrays` writes them.
struct GradGoldFile {
  // Stored as int32 in the file; held as int64_t like every host size.
  int64_t nao = 0;       // AO basis functions: mo_coeff's leading extent
  int64_t nmo = 0;       // = ncore + nact + nvirt, checked on load
  int64_t ncore = 0;
  int64_t nact = 0;
  int64_t nvirt = 0;
  int64_t nelecRhf = 0;  // the RHF electron count: which side of F's active
                         // spectrum is occupied (gradient-theory.md, §2.6)
  std::unordered_map<std::string, Tensor> arrays;

  // An IO Error naming the first of `names` the sidecar does not carry --
  // GoldenFile::require's contract, on the sidecar's arrays.
  [[nodiscard]] Status require(std::initializer_list<std::string_view> names) const;

  // The array `name`. Aborts (check) if it is absent: the caller should have
  // require()d it, so reaching here without it is our bug, not the file's.
  const Tensor& get(const std::string& name) const;
};

// The parsed sidecar, or an IO Error if it cannot be opened, ends early, has
// the wrong magic, carries a space partition that does not add up to nmo or a
// non-even RHF electron count that cannot fill its core, carries a malformed
// array header (as loadGolden refuses one), has bytes after its last array, or
// carries an `mo_coeff` whose shape is not [nao, nmo].
[[nodiscard]] Result<GradGoldFile> loadGradGold(const std::string& path);

}  // namespace nevpt2
