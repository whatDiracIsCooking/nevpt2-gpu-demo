// The sigma-vector kernels: a coefficient tensor over active indices
// contracted with the excited wavefunctions it multiplies, giving one
// amplitude per determinant. See sigma_vector_bridge.h for what that is and
// why it matters (the n^6 / n^8 RDM derivative never exists); this file is
// the device half.
//
// Its own file and its own device library, like rdm_accumulate.cu,
// f3_digest.cu and f3_scatter.cu -- not a patch to kernels.cu, which
// rdm_launch.cu #includes and which serves the RDM build alone. LINKED IN,
// like every kernel here: nevpt2.sigma_vector (sigma_vector.cpp) calls the
// launchers at the bottom through sigma_vector_bridge.h. The kernels and
// their launchers share this one TU, so nothing has to be #included to be
// launched.
//
// BACKEND-NEUTRAL, deliberately, exactly like kernels.cu: one thread owns one
// output determinant and the whole sum for it, so there is no atomic, no warp
// shuffle, no shared memory and nothing to synchronize -- WarpWraps'
// runtime.h, which picks the vendor header off the compiler's own device
// macro, is the whole prelude, and this file builds for CUDA and HIP alike.
// (On HIP it is compiled as a CXX source with -x hip; see
// cmake/add_device_library.cmake.)
//
// EVERY WALK IS A REVERSE WALK. (E_pq x)[K] = sum_{K'} <K|E_pq|K'> x[K'] is a
// row of the reverse link table (link_tables::reverse_link) -- the same
// rlink_a/rlink_b produce_generic's R gathers over. A row [cre, des, source,
// sign] of target string A says <A|E_{cre,des}|source> = sign for one spin,
// with the other spin's string riding along. So a determinant (A, B) has
// nla + nlb single-excitation sources, and a b-body sigma is b nested walks
// over them. The FIRST step is the LEFTMOST (last applied) operator, which is
// what puts d3's index pairs in the dm3 layout's order.
#include <runtime.h>  // WarpWraps: <cuda_runtime.h> or <hip/hip_runtime.h>

#include "rdm/sigma_vector_bridge.h"

// idx2: the one 64-bit widening these kernels need -- the n^4 coefficient-row
// and the ndet vector-row offsets. The kernel-local ints (thread index,
// link-table entries, index pairs, n2) stay int, as in kernels.cu.
#include "common/device_index.h"

namespace nevpt2::device {

namespace {

constexpr int kBlock = 128;

// Everything a walk step reads that does not change along it: the two reverse
// tables, the active-space shape, and nl = nla + nlb, the sources per
// determinant. Passed by value into __forceinline__ device code, so it costs
// nothing.
struct Links {
  const int* rlinkA;
  const int* rlinkB;
  int norb;
  int nb;
  int nla;
  int nlb;
  int nl;
};

// One reverse single-excitation step out of determinant (A, B): the l-th of
// its nl sources, with the operator's flat index pair p * norb + q and the
// parity. Alpha rows come first (l < nla), then beta; the order is immaterial,
// since every walk sums over all of them.
struct RevStep {
  int pq;       // p * norb + q of the operator E_pq this step applies
  int a;        // the source determinant's alpha string
  int b;        // ... and its beta string
  double sign;  // <A,B|E_pq|a,b>
};

__device__ __forceinline__ RevStep revStep(const Links lk, const int A, const int B, const int l) {
  if (l < lk.nla) {
    const int* const row = lk.rlinkA + (A * lk.nla + l) * 4;
    return RevStep{row[0] * lk.norb + row[1], row[2], B, static_cast<double>(row[3])};
  }
  const int* const row = lk.rlinkB + (B * lk.nlb + (l - lk.nla)) * 4;
  return RevStep{row[0] * lk.norb + row[1], A, row[2], static_cast<double>(row[3])};
}

// The 2-body sigma of ONE coefficient tensor at ONE determinant:
//
//   sum_{pq,rs} c[pq * n^2 + rs] (E_pq E_rs |x>)[A, B]
//
// Two nested reverse walks: the outer step is E_pq (leftmost), the inner
// E_rs, and the innermost factor is x at the determinant the second step
// landed on. Both sigma kernels below are this function -- the 3-body one
// with one more walk wrapped around it.
__device__ __forceinline__ double sigma2At(const Links lk, const double* __restrict__ x,
                                           const double* __restrict__ c, const int A, const int B) {
  const int n2 = lk.norb * lk.norb;
  double acc = 0.0;
  for (int l1 = 0; l1 < lk.nl; ++l1) {
    const RevStep s1 = revStep(lk, A, B, l1);
    const double* const crow = c + idx2(s1.pq, 0, n2);  // c[pq, :], the rs row
    for (int l2 = 0; l2 < lk.nl; ++l2) {
      const RevStep s2 = revStep(lk, s1.a, s1.b, l2);
      acc += s1.sign * s2.sign * crow[s2.pq] * x[idx2(s2.a, s2.b, lk.nb)];
    }
  }
  return acc;
}

// sigma[K] = sum_{pqrstu} d3[p,q,r,s,t,u] (E_pq E_rs E_tu |ci>)[K]: one
// reverse step for the leftmost pair, then the 2-body sigma of that pair's
// n^4 row at the determinant it landed on.
__global__ void sigma3Kernel(const double* __restrict__ ci, const double* __restrict__ d3,
                             double* __restrict__ sigma, const Links lk, const int k0,
                             const int width) {
  const int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= width) return;
  const int K = k0 + kl;  // global determinant index
  const int A = K / lk.nb;
  const int B = K % lk.nb;
  const int n2 = lk.norb * lk.norb;
  double acc = 0.0;
  for (int l = 0; l < lk.nl; ++l) {
    const RevStep s = revStep(lk, A, B, l);
    acc += s.sign * sigma2At(lk, ci, d3 + idx2(s.pq, 0, n2 * n2), s.a, s.b);
  }
  sigma[K] = acc;
}

// out[b, K] = sum_{pqrs} c2[b, p,q,r,s] (E_pq E_rs |x>)[K], over the n^2
// batches and the tile's determinants: one thread per (b, K) pair, the batch
// the slow index (idx / width), as consume_ca_generic's (af, K) is.
__global__ void sigma2BatchedKernel(const double* __restrict__ x, const double* __restrict__ c2,
                                    double* __restrict__ out, const Links lk, const int ndet,
                                    const int k0, const int width) {
  const int n2 = lk.norb * lk.norb;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n2 * width) return;
  const int b = idx / width;
  const int K = k0 + idx % width;
  out[idx2(b, K, ndet)] = sigma2At(lk, x, c2 + idx2(b, 0, n2 * n2), K / lk.nb, K % lk.nb);
}

// sigma[K] = sum_b sum_{pqrs} c2[b, p,q,r,s] (E_pq E_rs |in_b>)[K]: the same
// n^2 2-body sigmas, each against its own input row, summed over the batch by
// the one thread that owns K.
__global__ void sigma2ReduceKernel(const double* __restrict__ in, const double* __restrict__ c2,
                                   double* __restrict__ sigma, const Links lk, const int ndet,
                                   const int k0, const int width) {
  const int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= width) return;
  const int K = k0 + kl;
  const int A = K / lk.nb;
  const int B = K % lk.nb;
  const int n2 = lk.norb * lk.norb;
  double acc = 0.0;
  for (int b = 0; b < n2; ++b)
    acc += sigma2At(lk, in + idx2(b, 0, ndet), c2 + idx2(b, 0, n2 * n2), A, B);
  sigma[K] = acc;
}

}  // namespace

wwrError_t sigmaVector3(const wwrStream_t stream, const double* ci, const int* rlinkA,
                        const int* rlinkB, const double* d3, double* sigma, const int norb,
                        const int nb, const int nla, const int nlb, const int k0,
                        const int width) {
  const Links lk{rlinkA, rlinkB, norb, nb, nla, nlb, nla + nlb};
  sigma3Kernel<<<gridFor(width, kBlock), kBlock, 0, stream>>>(ci, d3, sigma, lk, k0, width);
  return wwrGetLastError();
}

wwrError_t sigmaVector2Batched(const wwrStream_t stream, const double* x, const int* rlinkA,
                               const int* rlinkB, const double* c2, double* out, const int norb,
                               const int nb, const int nla, const int nlb, const int ndet,
                               const int k0, const int width) {
  const Links lk{rlinkA, rlinkB, norb, nb, nla, nlb, nla + nlb};
  const int64_t threads = static_cast<int64_t>(norb) * norb * width;
  sigma2BatchedKernel<<<gridFor(threads, kBlock), kBlock, 0, stream>>>(x, c2, out, lk, ndet, k0,
                                                                       width);
  return wwrGetLastError();
}

wwrError_t sigmaVector2Reduce(const wwrStream_t stream, const double* in, const int* rlinkA,
                              const int* rlinkB, const double* c2, double* sigma, const int norb,
                              const int nb, const int nla, const int nlb, const int ndet,
                              const int k0, const int width) {
  const Links lk{rlinkA, rlinkB, norb, nb, nla, nlb, nla + nlb};
  sigma2ReduceKernel<<<gridFor(width, kBlock), kBlock, 0, stream>>>(in, c2, sigma, lk, ndet, k0,
                                                                    width);
  return wwrGetLastError();
}

}  // namespace nevpt2::device
