// nevpt2.cublas_emul:unavailable -- the abort every stub body falls into in a
// build without the emulated digest. An internal partition: nothing here is
// exported, so it is reachable from cublas_emul.cpp (`import :unavailable;`)
// and from no importer of nevpt2.cublas_emul. Non-CUDA builds only: the CUDA
// build lists nevpt2.cublas_emul:raw_handle instead (cublas/CMakeLists.txt).
module nevpt2.cublas_emul:unavailable;

import std;
import nevpt2.cublas_emul;

namespace nevpt2 {

// Never reached: every caller of the stubs is behind a --cublas that main()
// refused through requireCublasEmul(). Reaching one is our bug.
[[noreturn]] void no_cublas_emul(
    const std::source_location loc = std::source_location::current()) {
  check(false, "the --cublas digest was reached in a build without it", loc);
  std::abort();  // check(false, ...) has already aborted; this tells the compiler
}

}  // namespace nevpt2
