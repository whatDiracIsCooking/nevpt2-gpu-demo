// ozaki_digest_bridge.h -- the declaration shared between rdm/rdm_build.cpp
// (host C++23 module unit) and ozaki_digest.cu (device-compiled, linked in as
// the nevpt2.ozaki.device static library). Same shape as
// rdm/rdm_accumulate_bridge.h; see that header for why a host unit includes it
// after its imports, inside extern "C++".
//
// --ozaki: the digest GEMM
//     C[M,N] = sum_K A[M,K] * B[N,K]  (+ beta * C),
// both operands row-major with K contiguous, emulated on int8 tensor cores by
// the Ozaki scheme -- through WarpWraps' <wmma.h> (nvcuda::wmma or rocWMMA),
// so on both backends. docs/performance.md, "The int8 Ozaki
// digest", has what it measured.
#pragma once

#include "wwr/wwr.h"  // wwrStream_t (+ wwrError_t in a device pass)

// std::size_t: from `import std;` in the host unit, <cstddef> in the .cu.

namespace nevpt2::device {

// The widest K one kernel pass takes: every int32 accumulator stays exact
// while 8 * 64 * 64 * K < 2^31 (at most 8 digit pairs share a level,
// |digit| <= 64). A wider call is cut into K chunks of at most this.
inline constexpr int kOzakiMaxK = 65535;

// Bytes of device scratch ozakiDigestGemm needs for an (M, N, K) call: the
// int8 digit planes of both operands (padded to 16) for one K chunk, plus the
// two per-row scale vectors. The caller allocates it once, at the widest K.
std::size_t ozakiScratchBytes(int M, int N, int K);

// `maxPairSum` keeps the digit pairs (p, q) with p + q <= maxPairSum; 14 keeps
// all 64 of the 8x8. `scratch` is at least ozakiScratchBytes(M, N, K) bytes,
// 256-byte aligned. Returns the LAUNCH status, wwrSuccess on success.
// Precondition (the caller's to check): 0 <= maxPairSum <= 14. Any K > 0.
wwrError_t ozakiDigestGemm(wwrStream_t stream, const double* A, const double* B, double* C, int M,
                           int N, int K, double beta, int maxPairSum, void* scratch);

}  // namespace nevpt2::device
