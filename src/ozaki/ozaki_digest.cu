// --ozaki: the digest GEMM
//     C[M,N] = sum_K A[M,K] * B[N,K]  (+ beta * C)
// emulated on int8 tensor cores by the Ozaki scheme (Ozaki et al. 2012;
// docs/references.md, "Floating-point emulation"). The same scheme --cublas
// gets from cuBLAS's CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, written by hand
// so it is not tied to one vendor: the WMMA calls go through WarpWraps'
// <wmma.h>, whose wwr::wwrwmma is nvcuda::wmma on CUDA and rocwmma on HIP, and
// 16x16x16 -- the one shape both share -- is the only one used.
//
// Every step but the final double sum is exact:
//
//  1. Power-of-two scaling and digits. Each row of A and of B is scaled by
//     s = 2^(e+1), where max|row| < 2^e, so |x / s| <= 1/2 exactly; each digit
//     is then d = rint(r * 128), r <- r * 128 - d. Multiplying by 2^7 and
//     subtracting the nearest integer are exact in fp64, so |d| <= 64 and
//     digit p carries bits 7p+1 .. 7p+7 below the row's scale. The only
//     representation loss is truncation: an element 2^-j below its row's
//     scale keeps 56 - j bits.
//  2. Pairs grouped by level g = p + q, which all carry the same weight
//     2^-7(g+2): one int32 accumulator per level, and each digit tile loaded
//     once per K step. Exact while K <= kOzakiMaxK; a wider K is cut into
//     chunks of at most that, each scaled and split on its own and added in
//     with beta = 1 (the product is linear in K).
//
// The levels are then weighted (exact, powers of two) and summed in double,
// smallest first, and scaled by sA[i] * sB[j] (exact). So the result differs
// from the exact product only by the dropped pairs (maxPairSum < 14), the
// digit truncation of small elements, and the final double sum.
//
// Launched on the caller's stream; the scratch is the caller's (a
// DeviceBuffer in rdm_build.cpp), so nothing here allocates.
#include <cstddef>
#include <cstdint>

#include <cooperative_groups.h>
#include <runtime.h>
#include <wmma.h>

#include "ozaki/ozaki_digest_bridge.h"

namespace nevpt2::device {

namespace {

namespace wmma = ::wwr::wwrwmma;
namespace cg = ::cooperative_groups;

constexpr int kSplits = 8;
constexpr int kLevels = 2 * kSplits - 1;  // g = p + q in 0..14
constexpr int kDigitBits = 7;
constexpr int kT = 16;             // WMMA m = n = k: the portable shape
constexpr int kWarpsPerBlock = 4;  // along M; one N tile per block
constexpr int kWarp = WWR_WARP_SIZE;
constexpr int kPerLane = kT * kT / kWarp;  // C elements each lane folds
static_assert(kT * kT % kWarp == 0);

int pad16(int x) { return (x + kT - 1) / kT * kT; }

struct Layout {
  int Mp, Np, Kp;
  std::size_t aPlane, bPlane;  // bytes per digit plane
  std::size_t aBytes, bBytes, sBytes;
};

Layout layoutFor(int M, int N, int K) {
  Layout l{};
  l.Mp = pad16(M);
  l.Np = pad16(N);
  l.Kp = pad16(K);
  l.aPlane = static_cast<std::size_t>(l.Mp) * l.Kp;
  l.bPlane = static_cast<std::size_t>(l.Np) * l.Kp;
  auto up256 = [](std::size_t b) { return (b + 255) / 256 * 256; };
  l.aBytes = up256(l.aPlane * kSplits);
  l.bBytes = up256(l.bPlane * kSplits);
  l.sBytes = up256(static_cast<std::size_t>(l.Mp + l.Np) * sizeof(double));
  return l;
}

constexpr int kScaleThreads = 256;

// s[i] = 2^(e+1) with max_k |X[i,k]| < 2^e (1 for an all-zero row), over the
// K columns of a chunk whose rows are ld apart. One block per row.
__global__ void rowScaleKernel(const double* __restrict__ X, double* __restrict__ s, int rows,
                               int K, int ld) {
  const int i = blockIdx.x;
  if (i >= rows) return;  // whole block: rows == gridDim.x
  __shared__ double red[kScaleThreads];
  double m = 0.0;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    m = fmax(m, fabs(X[static_cast<std::size_t>(i) * ld + k]));
  red[threadIdx.x] = m;
  __syncthreads();
  for (int w = blockDim.x / 2; w > 0; w >>= 1) {
    if (threadIdx.x < w) red[threadIdx.x] = fmax(red[threadIdx.x], red[threadIdx.x + w]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    int e = 0;
    frexp(red[0], &e);  // red[0] = f * 2^e, f in [0.5, 1)  =>  red[0] < 2^e
    s[i] = red[0] > 0.0 ? ldexp(1.0, e + 1) : 1.0;
  }
}

// Digit planes of X (rows x K, row-major, rows ld apart) into
// out[p][row * Kp + k], zero in the padding. One thread per padded element.
__global__ void splitKernel(const double* __restrict__ X, const double* __restrict__ s,
                            std::int8_t* __restrict__ out, int rows, int K, int ld, int Kp,
                            std::size_t plane) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= plane) return;
  const int row = static_cast<int>(idx / Kp);
  const int k = static_cast<int>(idx % Kp);
  double r = (row < rows && k < K) ? X[static_cast<std::size_t>(row) * ld + k] / s[row] : 0.0;
#pragma unroll
  for (int p = 0; p < kSplits; ++p) {
    const double t = r * 128.0;  // exact
    const double d = rint(t);    // |d| <= 64
    out[p * plane + idx] = static_cast<std::int8_t>(d);
    r = t - d;  // exact
  }
}

// One warp per 16x16 C tile: every level's int32 accumulator over the whole K,
// then the weighted double sum and the scale-back, written straight to C.
// Each level is folded through the warp's own shared-memory stage under a
// warp-sized cooperative-groups tile barrier (WarpWraps' portable spelling
// of __syncwarp), so warps never wait on each other and an idle one exits.
// A block-wide __syncthreads there measured ~2x slower on the RTX 3080
// (docs/performance.md, "The int8 Ozaki digest").
__global__ void ozakiGemmKernel(const std::int8_t* __restrict__ Ad,
                                const std::int8_t* __restrict__ Bd, std::size_t aPlane,
                                std::size_t bPlane, const double* __restrict__ sA,
                                const double* __restrict__ sB, double* __restrict__ C, int M,
                                int N, int Kp, double beta, int maxPairSum) {
  __shared__ std::int32_t stage[kWarpsPerBlock][kT * kT];
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int row = (blockIdx.x * kWarpsPerBlock + warp) * kT;
  const int col = blockIdx.y * kT;
  if (row >= M) return;  // whole warp; nothing below waits on other warps
  const auto warpTile = cg::tiled_partition<kWarp>(cg::this_thread_block());

  wmma::fragment<wmma::matrix_a, kT, kT, kT, std::int8_t, wmma::row_major> a[kSplits];
  wmma::fragment<wmma::matrix_b, kT, kT, kT, std::int8_t, wmma::col_major> b[kSplits];
  wmma::fragment<wmma::accumulator, kT, kT, kT, std::int32_t> acc[kLevels];
#pragma unroll
  for (int g = 0; g < kLevels; ++g) wmma::fill_fragment(acc[g], 0);

  for (int k0 = 0; k0 < Kp; k0 += kT) {
#pragma unroll
    for (int p = 0; p < kSplits; ++p)
      if (p <= maxPairSum) {
        wmma::load_matrix_sync(a[p], Ad + p * aPlane + static_cast<std::size_t>(row) * Kp + k0,
                               Kp);
        // B is [N, K] row-major = [K, N] column-major with leading dim Kp.
        wmma::load_matrix_sync(b[p], Bd + p * bPlane + static_cast<std::size_t>(col) * Kp + k0,
                               Kp);
      }
#pragma unroll
    for (int p = 0; p < kSplits; ++p)
#pragma unroll
      for (int q = 0; q < kSplits; ++q)
        if (p + q <= maxPairSum) wmma::mma_sync(acc[p + q], a[p], b[q], acc[p + q]);
  }

  double sum[kPerLane];
#pragma unroll
  for (int t = 0; t < kPerLane; ++t) sum[t] = 0.0;
  std::int32_t* my = stage[warp];
#pragma unroll
  for (int g = kLevels - 1; g >= 0; --g) {  // smallest weight first
    if (g > maxPairSum) continue;           // uniform across the warp
    wmma::store_matrix_sync(my, acc[g], kT, wmma::mem_row_major);
    warpTile.sync();
    const double w = ldexp(1.0, -kDigitBits * (g + 2));
#pragma unroll
    for (int t = 0; t < kPerLane; ++t) sum[t] += w * static_cast<double>(my[lane + kWarp * t]);
    warpTile.sync();  // before the next level overwrites the stage
  }
#pragma unroll
  for (int t = 0; t < kPerLane; ++t) {
    const int e = lane + kWarp * t;
    const int i = row + e / kT, j = col + e % kT;
    if (i < M && j < N) {
      const double v = sA[i] * sB[j] * sum[t];
      double* c = C + static_cast<std::size_t>(i) * N + j;
      *c = beta == 0.0 ? v : v + beta * *c;
    }
  }
}

}  // namespace

std::size_t ozakiScratchBytes(int M, int N, int K) {
  const Layout l = layoutFor(M, N, K < kOzakiMaxK ? K : kOzakiMaxK);
  return l.aBytes + l.bBytes + l.sBytes;
}

wwrError_t ozakiDigestGemm(wwrStream_t stream, const double* A, const double* B, double* C, int M,
                           int N, int K, double beta, int maxPairSum, void* scratch) {
  // K chunks of at most kOzakiMaxK, near-equal. The scratch is reused by
  // each chunk in turn: stream order serializes them.
  const int nChunks = (K + kOzakiMaxK - 1) / kOzakiMaxK;
  for (int c = 0; c < nChunks; ++c) {
    const int k0 = static_cast<int>(static_cast<std::int64_t>(K) * c / nChunks);
    const int k1 = static_cast<int>(static_cast<std::int64_t>(K) * (c + 1) / nChunks);
    const int kc = k1 - k0;
    const Layout l = layoutFor(M, N, kc);
    auto* base = static_cast<std::int8_t*>(scratch);
    std::int8_t* Ad = base;
    std::int8_t* Bd = base + l.aBytes;
    auto* sA = reinterpret_cast<double*>(base + l.aBytes + l.bBytes);
    double* sB = sA + l.Mp;

    rowScaleKernel<<<M, kScaleThreads, 0, stream>>>(A + k0, sA, M, kc, K);
    rowScaleKernel<<<N, kScaleThreads, 0, stream>>>(B + k0, sB, N, kc, K);
    constexpr int kBlock = 256;
    splitKernel<<<static_cast<unsigned>((l.aPlane + kBlock - 1) / kBlock), kBlock, 0, stream>>>(
        A + k0, sA, Ad, M, kc, K, l.Kp, l.aPlane);
    splitKernel<<<static_cast<unsigned>((l.bPlane + kBlock - 1) / kBlock), kBlock, 0, stream>>>(
        B + k0, sB, Bd, N, kc, K, l.Kp, l.bPlane);
    const dim3 grid((l.Mp / kT + kWarpsPerBlock - 1) / kWarpsPerBlock, l.Np / kT);
    ozakiGemmKernel<<<grid, kWarp * kWarpsPerBlock, 0, stream>>>(
        Ad, Bd, l.aPlane, l.bPlane, sA, sB, C, M, N, l.Kp, c == 0 ? beta : 1.0, maxPairSum);
    if (const wwrError_t e = wwrGetLastError(); e != wwrSuccess) return e;
  }
  return wwrSuccess;
}

}  // namespace nevpt2::device
