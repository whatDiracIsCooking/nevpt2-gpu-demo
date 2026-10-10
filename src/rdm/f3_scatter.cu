// f3 permute-scatter for the GEMM digest paths (--cublas and --blas-digest).
//
// A GEMM digest computes the f3 GEMM in its *winning* orientation, an
// (n^4, n^2) row-major result -- the M=n^4, N=n^2 aspect ratio the emulation
// accelerates ~5-6x, vs the as-written (n^2, n^4) that is a wash. The
// scatter folds that temp into the
// NEVPTkern-permuted n^6 accumulator, accumulating (+=) so the caller zeroes
// the accumulator once.
//
// The GEMM reads L2 directly -- no path builds the R2 copy the emitted digest
// once used: R2[tuvw, K] = L2[wvut, K], so
//     C[r, af] = sum_K L2[r, K] * W[af, K],   r = ((w*n + v)*n + u)*n + t
// is the same result as R2's C[tuvw, af], with each row already sitting at
// its reversed index. The NEVPTkern target ((((w,v),u),t),fr,a) for ca and
// ((((w,v),u),t),a,fr) for ac then starts at r*n^2. So the scatter is a
// transpose of the last two axes within each row (ca) and a plain += (ac) --
// the latter is rdm_accumulate.cu's accumulateInplace, so only ca lives here.
//
// LINKED IN, not loaded as a code object: a STATIC device library
// (wwr_add_gpu_device_library, see CMakeLists.txt) whose launcher
// rdm_build.cpp calls through f3_scatter_bridge.h, launched through WarpWraps'
// parallel_for. Built on BOTH backends: the kernel itself is backend-neutral,
// and besides the CUDA-only --cublas path the native-fp64 --blas-digest
// calls it on either backend.
#include "rdm/f3_scatter_bridge.h"

#include "common/device_index.h"  // idx2, splitIdx2, int64_t
#include <extension/parallel_for/parallel_for.cuh>

namespace nevpt2::device {

namespace {

// Const scalar members: parallel_for's device_functor requires the functor
// not be copy-assignable (it is passed as a grid constant).
struct F3ScatterCaFunctor {
  const double* const c;
  double* const f3;
  const int norb;

  __device__ void operator()(const int64_t i) const {
    const int n2 = norb * norb;
    const auto [r, af] = splitIdx2(i, n2);  // r = wvut, af = a*norb + fr
    const int a = af / norb;
    const int fr = af % norb;
    f3[idx2(r, fr * norb + a, n2)] += c[i];
  }
};

// --fused-digest: the three digest GEMMs as one, C[r, j] = sum_K L2[r, K] *
// B[j, K] with B = [R; W_ca; W_ac] stacked (j in [0, 3 n^2)), so C is
// (n^4, 3 n^2) row-major. One thread per (r, col) of an n^2 block folds all
// three: dm3 is a plain +=, ca the last-two-axes transpose above, ac a +=.
struct FusedSplitFunctor {
  const double* const c;
  double* const dm3;
  double* const f3ca;
  double* const f3ac;
  const int norb;

  __device__ void operator()(const int64_t i) const {
    const int n2 = norb * norb;
    const auto [r, col] = splitIdx2(i, n2);
    const int a = col / norb;
    const int fr = col % norb;
    const double* row = c + r * 3 * n2;
    dm3[i] += row[col];
    f3ca[idx2(r, fr * norb + a, n2)] += row[n2 + col];
    f3ac[i] += row[2 * n2 + col];
  }
};

}  // namespace

wwrError_t fusedDigestSplit(const wwrStream_t stream, const double* c, double* dm3, double* f3ca,
                            double* f3ac, const int norb) {
  const int64_t n = norb;
  const int64_t n2 = n * n;
  const int64_t n6 = n2 * n2 * n2;
  return ::wwr::extension::parallel_for(stream, n6,
                                        FusedSplitFunctor{c, dm3, f3ca, f3ac, norb});
}

wwrError_t f3ScatterCa(const wwrStream_t stream, const double* c, double* f3, const int norb) {
  const int64_t n = norb;
  const int64_t n2 = n * n;
  const int64_t n6 = n2 * n2 * n2;
  return ::wwr::extension::parallel_for(stream, n6, F3ScatterCaFunctor{c, f3, norb});
}

}  // namespace nevpt2::device
