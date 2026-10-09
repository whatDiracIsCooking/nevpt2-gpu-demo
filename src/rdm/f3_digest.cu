// The emitted path's f3 digest GEMM, reading L2 directly instead of R2.
//
// kernels.cu's former nevpt2_digest_f3_{ca,ac}_generic took R2, a full
// permuted copy of L2 -- R2[tuvw, K] = L2[wvut, K], the same n^4 * width --
// built per tile by nevpt2_r2_transpose_generic only so that the digest's N
// index runs in tuvw order. But the digest then writes its (af, tuvw) result
// to the NEVPTkern target f3[((((w,v),u),t),fr,a)] (ca) or
// f3[((((w,v),u),t),a,fr)] (ac): the target row IS r = wvut, the L2 row R2's
// tuvw row was copied from. So letting the N index run over L2's own rows
// gives
//     f3[r*n^2 + fr*n + a] (ca) / f3[r*n^2 + a*n + fr] (ac)
//         = sum_K W[a*n + fr, K] * L2[r, K]
// with no permutation anywhere: no R2, no r2_transpose. The same observation
// let f3_scatter.cu drop R2 from the --cublas path.
//
// Otherwise kernels.cu's digest GEMM, unchanged: the same 16x16 tile
// (TILE = TK = 16, __shared__, __syncthreads), the same loads, the same
// K-ordered accumulation, so each output element is the same sum, in the
// same order, as on the R2 path. One more change:
// the result is added (+=) into the tile-sweep accumulator instead of
// overwriting a per-tile n^6 output that rdm_accumulate.cu then added in.
// Each element has exactly one writer, so the += needs no atomics, and
// acc_final + acc is what the accumulate kernel computed. That drops the two
// per-tile f3 outputs (dFaccTile) and their accumulate launches.
//
// Its own file and device library, not a patch to kernels.cu (whose R2
// transpose and f3 digest kernels, left uncalled, were later deleted). LINKED
// IN like rdm_accumulate.cu;
// rdm_build.cpp calls it through f3_digest_bridge.h. The launch is a
// hand-written <<<dim3(gx, gy), dim3(16, 16)>>>, not parallel_for, because
// the tile is baked into the kernel. Shared unchanged between backends: under
// HIP the .cu is compiled with -x hip.
#include "rdm/f3_digest_bridge.h"

#include "common/device_index.h"  // idx2, gridFor, int64_t

namespace nevpt2::device {

namespace {

constexpr int kTile = 16;

// kAc selects the output's last two axes: (fr, a) for ca, (a, fr) for ac.
template <bool kAc>
__global__ void f3DigestKernel(const double* __restrict__ W, const double* __restrict__ L2,
                               double* __restrict__ f3, const int norb, const int width) {
  const int n2 = norb * norb;
  const int n4 = n2 * n2;
  const int M = n2;  // af = a*norb + free (A = W rows)
  const int N = n4;  // r = wvut           (B = L2 rows)
  const int Kdim = width;
  const int TILE = kTile, TK = kTile;
  const int nthreads = blockDim.x * blockDim.y;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * TILE + threadIdx.y;  // M index (af)
  const int col = blockIdx.x * TILE + threadIdx.x;  // N index (wvut)
  __shared__ double As[kTile][kTile];  // W  tile: rows af,   cols K
  __shared__ double Bs[kTile][kTile];  // L2 tile: rows wvut, cols K
  double acc = 0.0;
  for (int k0 = 0; k0 < Kdim; k0 += TK) {
    for (int i = tid; i < TILE * TK; i += nthreads) {
      const int r = i / TK, c = i % TK;
      const int gm = blockIdx.y * TILE + r;
      const int gn = blockIdx.x * TILE + r;
      const int gk = k0 + c;
      As[r][c] = (gm < M && gk < Kdim) ? W[idx2(gm, gk, Kdim)] : 0.0;
      Bs[r][c] = (gn < N && gk < Kdim) ? L2[idx2(gn, gk, Kdim)] : 0.0;
    }
    __syncthreads();
    const int kk = (Kdim - k0 < TK) ? (Kdim - k0) : TK;
    if (row < M && col < N)
      for (int c = 0; c < kk; ++c) acc += As[threadIdx.y][c] * Bs[threadIdx.x][c];
    __syncthreads();
  }
  if (row < M && col < N) {
    const int a = row / norb;
    const int fr = row % norb;
    const int last2 = kAc ? a * norb + fr : fr * norb + a;
    f3[idx2(col, last2, n2)] += acc;
  }
}

}  // namespace

wwrError_t f3DigestAccumulate(const wwrStream_t stream, const int order, const double* w,
                              const double* l2, double* f3, const int norb, const int width) {
  const int64_t n = norb;
  const int64_t n2 = n * n;
  const int64_t n4 = n2 * n2;
  // x over the n4 columns, y over the n2 rows, one 16x16 tile per block: the
  // same gridFor as rdm_launch.cu's dm3 digest.
  const dim3 block(kTile, kTile);
  const dim3 grid = gridFor(n4, n2, block);
  if (order == 0)
    f3DigestKernel<false><<<grid, block, 0, stream>>>(w, l2, f3, norb, width);
  else
    f3DigestKernel<true><<<grid, block, 0, stream>>>(w, l2, f3, norb, width);
  return wwrGetLastError();
}

}  // namespace nevpt2::device
