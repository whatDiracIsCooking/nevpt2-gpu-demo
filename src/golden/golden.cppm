// nevpt2.golden -- reader for the golden CASSCF-input reference file `generate_golden.py`
// writes (see that script's `write_golden` docstring for the exact binary
// layout: magic, scalar header, then a sequence of named N-d f64 arrays).
// Pure host C++, no GPU -- this only needs to run once at program start.
//
// The file is the user's input, so every way it can be missing or malformed
// is a value-tier Error (nevpt2.error_handling, ErrorKind::IO) handed back to
// the demo's main() to report once -- never an abort.
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

  // The array `name`. Aborts (check) if it is absent: the demo should have
  // require()d it, so reaching here without it is our bug, not the file's.
  const Tensor& get(const std::string& name) const;
};

// The parsed file, or an IO Error if it cannot be opened, ends early, has the
// wrong magic, or carries a malformed array header (a negative or oversized
// rank, a negative extent, or an element count that overflows or runs past
// the end of the file).
[[nodiscard]] Result<GoldenFile> loadGolden(const std::string& path);

}  // namespace nevpt2
