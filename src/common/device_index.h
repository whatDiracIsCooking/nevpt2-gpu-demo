// Device index helpers: row-major flattening in int64_t, the global
// thread index, and one gridFor for every hand-written <<<>>> launch.
//
// Every n^4 / n^6 offset a kernel file computes is widened to 64 bits once,
// here, instead of by a `(size_t)` / `(long long)` cast at the head of each
// chain. Kernel-local `int` indices stay where they are deliberate (32-bit is
// cheaper on the GPU; link tables are int): the helpers take int64_t, so an
// `int` argument widens at the call and nothing narrower than the offset is
// touched. The flattening is integer arithmetic on non-negative values far
// below 2^63, so writing a chain through these helpers changes no value and
// no floating-point order.
//
// gridFor builds on idivup (align_up.h); the launch shape stays the caller's
// -- the block size is an argument, never a default.
//
// Device-only in practice, like block_params.h: dim3 and blockIdx come from
// WarpWraps' runtime.h, so this header has no module face. Reached
// root-relative as "common/device_index.h"; the target links
// nevpt2::common::device.
#pragma once

#include "align_up.h"     // idivup
#include "host_device.h"  // NEVPT2_HOST_DEVICE
#include "int64.h"        // nevpt2::int64_t
#include <runtime.h>      // dim3, blockIdx, blockDim, threadIdx, gridDim

namespace nevpt2 {

// Row-major offset of (i, j) in a matrix with nj columns.
NEVPT2_HOST_DEVICE constexpr int64_t idx2(const int64_t i, const int64_t j, const int64_t nj) {
  return i * nj + j;
}

// Row-major offset of (p, q, r, s) in an (n, n, n, n) tensor.
NEVPT2_HOST_DEVICE constexpr int64_t idx4(const int64_t p, const int64_t q, const int64_t r,
                                          const int64_t s, const int64_t n) {
  return ((p * n + q) * n + r) * n + s;
}

// Row-major offset of (p, q, r, s) in an (np, nq, nr, ns) tensor; the leading
// extent np does not enter.
NEVPT2_HOST_DEVICE constexpr int64_t idx4(const int64_t p, const int64_t q, const int64_t r,
                                          const int64_t s, const int64_t nq, const int64_t nr,
                                          const int64_t ns) {
  return ((p * nq + q) * nr + r) * ns + s;
}

// Row-major offset of (p, q, r, s, t, u) in an (n,)^6 tensor.
NEVPT2_HOST_DEVICE constexpr int64_t idx6(const int64_t p, const int64_t q, const int64_t r,
                                          const int64_t s, const int64_t t, const int64_t u,
                                          const int64_t n) {
  return ((((p * n + q) * n + r) * n + s) * n + t) * n + u;
}

// The inverse of idx2 for a row width nj that fits an int: the row stays
// int64_t, the column (< nj) is narrowed to int here, once, so a kernel can
// keep its column arithmetic 32-bit.
struct RowCol {
  int64_t row;
  int col;
};
NEVPT2_HOST_DEVICE constexpr RowCol splitIdx2(const int64_t i, const int nj) {
  return {i / nj, static_cast<int>(i % nj)};
}

// The blocks of `block` threads that cover n elements, as a 1-D grid size.
// Unchecked narrowing to unsigned, as every grid here always was: the callers
// stay far below a grid's 2^31 - 1 x-limit.
NEVPT2_HOST_DEVICE constexpr unsigned gridFor(const int64_t n, const int block) {
  return static_cast<unsigned>(idivup<int64_t>(n, block));
}

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

// The 2-D grid of `block`-shaped blocks that covers an (nx, ny) index space:
// x over nx, y over ny.
inline dim3 gridFor(const int64_t nx, const int64_t ny, const dim3 block) {
  return dim3(gridFor(nx, static_cast<int>(block.x)), gridFor(ny, static_cast<int>(block.y)), 1);
}

// This thread's index in a 1-D grid, widened before the multiply so it
// cannot wrap at 2^32.
__device__ __forceinline__ int64_t globalThreadIndex() {
  return static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

// The thread count of a 1-D grid: a grid-stride loop's step.
__device__ __forceinline__ int64_t gridStride() {
  return static_cast<int64_t>(gridDim.x) * blockDim.x;
}

#endif

}  // namespace nevpt2
