// nevpt2.golden:reader -- the byte-level reader behind loadGolden: a bounded
// FILE* cursor and the one-array parser. An internal partition: nothing here is
// exported, so it is reachable from golden.cpp (`import :reader;`) and from no
// importer of nevpt2.golden.
module;

#include <cstdio>  // SEEK_END / SEEK_SET: macros, which `import std` does not carry

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.golden:reader;

import std;
import nevpt2.golden;

namespace nevpt2 {

// No array generate_golden.py writes has more than a handful of axes; a rank
// past this is a corrupt header, refused before it sizes a shape vector.
constexpr int64_t kMaxRank = 16;

class Reader {
 public:
  static Result<Reader> open(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return err_io("could not open golden file: " + path);
    Reader r(f, path);
    // The file's size bounds every later read, so a corrupt extent is refused
    // as running past the end instead of sizing a huge allocation.
    if (std::fseek(f, 0, SEEK_END) != 0) return err_io("could not seek golden file: " + path);
    const long size = std::ftell(f);
    if (size < 0 || std::fseek(f, 0, SEEK_SET) != 0)
      return err_io("could not size golden file: " + path);
    r.remaining_ = size;
    return r;
  }

  const std::string& path() const { return path_; }
  int64_t remaining() const { return remaining_; }

  Status bytes(void* dst, std::size_t n) {
    if (std::cmp_greater(n, remaining_) || std::fread(dst, 1, n, f_.get()) != n)
      return err_io("golden file: unexpected EOF -- " + path_);
    remaining_ -= static_cast<int64_t>(n);
    return {};
  }
  Result<int64_t> i32() {
    std::int32_t v;
    NEVPT2_TRY(bytes(&v, 4));
    return v;
  }
  Result<int64_t> i64() {
    int64_t v;
    NEVPT2_TRY(bytes(&v, 8));
    return v;
  }
  Result<double> f64() {
    double v;
    NEVPT2_TRY(bytes(&v, 8));
    return v;
  }
  Result<std::string> str(int64_t n) {
    if (n < 0 || n > remaining_)
      return err_io(std::format("golden file: bad string length {} -- {}", n, path_));
    std::string s(static_cast<std::size_t>(n), '\0');
    NEVPT2_TRY(bytes(s.data(), s.size()));
    return s;
  }

 private:
  struct Closer {
    void operator()(std::FILE* f) const { std::fclose(f); }
  };
  Reader(std::FILE* f, std::string path) : f_(f), path_(std::move(path)) {}

  std::unique_ptr<std::FILE, Closer> f_;
  std::string path_;
  int64_t remaining_ = 0;
};

// One named array, its header validated before it sizes anything.
Status readArray(Reader& r, GoldenFile& g) {
  const int64_t nameLen = NEVPT2_TRY(r.i32());
  std::string name = NEVPT2_TRY(r.str(nameLen));
  const int64_t ndim = NEVPT2_TRY(r.i32());
  if (ndim < 0 || ndim > kMaxRank)
    return err_io(std::format("golden file: array '{}' has rank {} (expected 0..{}) -- {}", name,
                              ndim, kMaxRank, r.path()));
  // The file stores each extent as an int64 -- the int64_t held here.
  std::vector<int64_t> shape(static_cast<std::size_t>(ndim));
  int64_t total = 1;
  for (int64_t& s : shape) {
    s = NEVPT2_TRY(r.i64());
    if (s < 0)
      return err_io(std::format("golden file: array '{}' has a negative extent {} -- {}", name,
                                s, r.path()));
    // Bounded by the bytes left in the file, so neither the element count
    // nor its byte count can overflow.
    if (s > 0 && total > r.remaining() / static_cast<int64_t>(sizeof(double)) / s)
      return err_io(std::format("golden file: array '{}' runs past the end of the file -- {}",
                                name, r.path()));
    total *= s;
  }
  std::vector<double> data(static_cast<std::size_t>(total));
  NEVPT2_TRY(r.bytes(data.data(), data.size() * sizeof(double)));
  g.arrays.emplace(std::move(name), Tensor::fromFlat(shape, std::move(data)));
  return {};
}

}  // namespace nevpt2
