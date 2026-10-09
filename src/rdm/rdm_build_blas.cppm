// nevpt2.rdm_build:blas -- the BLAS GEMMs of the RDM build: the f3 consume step
// (and the permuted integrals and per-backend chunking it is issued
// with), and the native-fp64 --blas-digest GEMM.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :blas;` and from no importer of nevpt2.rdm_build.
// (Used by rdm_build.cpp.)
module nevpt2.rdm_build:blas;

import std;
import nevpt2.rdm_build;
import wwr.blas;
// wwrblasStatus_t's error_type specializations: what lets gpuCheck take a
// BLAS status.
import wwr.extension.blas;

namespace nevpt2 {

// The f3 consume step through backend-neutral BLAS (cuBLAS / hipBLAS via
// wwr.blas), the default, instead of the emitted loop kernels
// (--consume-emitted). The emitted kernels compute, per tile of width w,
//   ca: W[a,f,K] = sum_{pqx} eri[a,x,q,p] * L2[p,q,x,f,K]
//   ac: W[a,f,K] = sum_{pqx} eri[a,x,q,p] * L2[p,q,f,x,K]
// with L2 row-major [pqrs, K] and W row-major [af, K], both at row stride w.
// The orders differ only in which L2 slot the third contracted index x and
// the free f occupy, so both run the same loop: for each x, one
// strided-batched GEMM over f, accumulating with beta=1 after the first,
//   W[a, f, K] += E_x[a, pq] . L2[pq, (x,f), K],  E[x,a,p,q] = eri[a,x,q,p]
// (k = n^2, batch stride = f's L2 stride). One n^4 permuted integral copy
// serves both orders, built once on the host.
//
// Measured shape choice (docs/performance.md, "Consume as a DGEMM"): the product is skinny,
// only n output columns (a) per call, and the two BLAS libraries tile it
// differently, so each per-x GEMM is issued in column chunks and k chunks
// (contiguous row ranges of pq, accumulated with beta=1 like the x loop)
// whose limits are per backend -- tuning to each library's heuristic, not a
// property of the math:
//  * cuBLAS 13.0, RTX 3080: with <= 10 columns and k <= 128 it picks a skinny
//    kernel that runs at the card's fp64 peak; at 11+ columns or k = 144 it
//    falls back to a 32-column tile and wastes the padding (~150 GFLOP/s,
//    flat in the column count up to 32). CAS(10,10) (10 columns, k = 100) is
//    in the fast regime as is; CAS(12,12) (12, 144) is not.
//  * hipBLAS / rocBLAS, RX 9060 XT: one 16-column tile, time flat in the
//    column count up to 16 and indifferent to k. Splitting 12 columns into
//    two halves there doubles the time, so the limit is 16 and k is whole.
// For the same reason ca does not use the one plain GEMM it also fits (L2 as
// [pqr, (f,K)] is contiguous in (f,K), k = n^3): 3.5x slower on cuBLAS at
// CAS(10,10).
//
// BLAS is column-major, so the row-major C = A.B is issued as C^T = B^T.A^T:
// the L2 operand goes first.
//
// Issued through the 64-bit-integer API (wwrblasDgemmStridedBatched_64:
// cublasDgemmStridedBatched_64 / hipblasDgemmStridedBatched_64), so no
// dimension or leading dimension is narrowed to int (until then
// a checked int narrowing of lda = n^2 * width asked for more --tiles).

// eri is chemists'-order eriF3, row-major [a, x, q, p]; out is [x, a, p, q].
std::vector<double> permuteEriConsume(const std::vector<double>& eri, int64_t n) {
  std::vector<double> out(eri.size());
  int64_t i = 0;
  for (int64_t x = 0; x < n; ++x)
    for (int64_t a = 0; a < n; ++a)
      for (int64_t p = 0; p < n; ++p)
        for (int64_t q = 0; q < n; ++q) out[i++] = eri[((a * n + x) * n + q) * n + p];
  return out;
}

struct ConsumeChunking {
  int64_t maxCols;
  int64_t maxK;
};

ConsumeChunking consumeChunking() {
  if (std::string_view(kGpuBackendName) == "CUDA") return {10, 128};
  return {16, std::numeric_limits<int64_t>::max()};
}

// Split [0, total) into the fewest near-equal chunks of at most `maxLen`.
// total * (c + 1) stays far inside int64_t: total is n or n^2 here.
std::vector<std::pair<int64_t, int64_t>> evenChunks(int64_t total, int64_t maxLen) {
  int64_t count = total / maxLen + (total % maxLen != 0);  // ceil, no overflow at the max
  std::vector<std::pair<int64_t, int64_t>> out;
  for (int64_t c = 0; c < count; ++c) out.emplace_back(total * c / count, total * (c + 1) / count);
  return out;
}

// order 0 (ca): x is L2's third slot (stride n*w), f its fourth (stride w).
// order 1 (ac): the other way round.
void consumeGemm(wwrblasHandle_t h, const double* dE, const double* dL2, double* dW, int order,
                 int64_t n, int64_t width) {
  const double one = 1.0, zero = 0.0;
  int64_t n2 = n * n;
  int64_t nw = n * width;
  int64_t xStride = order == 0 ? nw : width;
  int64_t fStride = order == 0 ? width : nw;
  int64_t lda = n2 * width, ldb = n2, ldc = nw;
  const ConsumeChunking lim = consumeChunking();
  auto colChunks = evenChunks(n, lim.maxCols);
  auto kChunks = evenChunks(n2, lim.maxK);
  for (auto [a0, a1] : colChunks) {
    double* c = dW + a0 * nw;
    for (int64_t x = 0; x < n; ++x) {
      for (auto [k0, k1] : kChunks) {
        // Row k0 of the L2 operand (pq = k0) sits k0 * lda further on.
        const double* l2 = dL2 + x * xStride + k0 * n2 * width;
        const double* e = dE + x * n2 * n + a0 * n2 + k0;
        bool first = x == 0 && k0 == 0;
        gpuCheck(wwrblasDgemmStridedBatched_64(h, WWRBLAS_OP_N, WWRBLAS_OP_N, width, a1 - a0,
                                               k1 - k0, &one, l2, lda, fStride, e, ldb, 0,
                                               first ? &zero : &one, c, ldc, width, n));
      }
    }
  }
}

// --blas-digest: the --cublas digest GEMM's exact shape and
// operand mapping (cublas/cublas_emul.cpp's digest_gemm), issued as a plain
// native-fp64 wwrblasDgemm -- cuBLAS or rocBLAS, so it runs on both
// backends. Row-major C[M,N] = sum_K A[M,K] * B[N,K] (+ beta * C) is the
// column-major C^T = op_T(B) . A, dims (m=N, n=M, k=K), leading dims K/K/N.
// Through the 64-bit-integer API (wwrblasDgemm_64: cublasDgemm_v2_64 /
// hipblasDgemm_64), so nothing is narrowed.
void digestGemmNative(wwrblasHandle_t h, const double* dA, const double* dB, double* dC, int64_t M,
                      int64_t N, int64_t K, double beta) {
  const double one = 1.0;
  gpuCheck(
      wwrblasDgemm_64(h, WWRBLAS_OP_T, WWRBLAS_OP_N, N, M, K, &one, dB, K, dA, K, &beta, dC, N));
}

}  // namespace nevpt2
