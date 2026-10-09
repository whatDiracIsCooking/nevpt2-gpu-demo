// block_reduce_bridge.h -- the launchers block_reduce_tests.cpp (host, a plain
// TU that imports modules) calls into block_reduce_kernels.cu (device-compiled,
// linked in as the nevpt2.test.block_reduce.device static library). Same shape
// and rules as warp_reduce_bridge.h: included after the host TU's imports,
// plain types only.
//
// Every launcher runs `nblocks` independent block folds of `threads` threads
// each: block b folds in[b * threads, (b + 1) * threads) with nactive[b]
// active threads and writes its result from EVERY thread to
// out[b * threads + t]. Each returns the launch status, for the caller's
// gpuCheck; a shape the kernels were not instantiated for returns
// wwrErrorInvalidValue, which gpuCheck turns into an abort (a test's bug).
//
//   blockReduceTree*   block_reduce<kBlock>(v, op, nactive): the shared-memory
//                      tree. threads = kBlock, any power of two in [1, 1024].
//   blockReduceWarps*  block_reduce(block, partials, v, op, nactive): warp
//                      shuffles, then across warps. threads = numWarps * W,
//                      numWarps a power of two with numWarps * W <= 1024,
//                      W = blockReduceWarpSize().
#pragma once

#include "wwr/wwr.h"  // wwrStream_t (+ wwrError_t in a device pass)

namespace nevpt2::device {

// kWarpSize as the device pass that built block_reduce compiled it.
unsigned int blockReduceWarpSize();

// --- the tree overload ---

wwrError_t blockReduceTreeSum(wwrStream_t stream, const double* in, double* out,
                              const unsigned int* nactive, unsigned int nblocks,
                              unsigned int threads);

wwrError_t blockReduceTreeMaxNan(wwrStream_t stream, const double* in, double* out,
                                 const unsigned int* nactive, unsigned int nblocks,
                                 unsigned int threads);

// The interval probe folded as the Interval STRUCT (the class-type path only
// the tree carries); encoded at load and store.
wwrError_t blockReduceTreeInterval(wwrStream_t stream, const unsigned long long* in,
                                   unsigned long long* out, const unsigned int* nactive,
                                   unsigned int nblocks, unsigned int threads);

// Two back-to-back sums in one kernel through the tree's one static buffer:
// out = fold(in), outNeg = fold(-in). Proves the buffer is reusable without a
// barrier at the call site.
wwrError_t blockReduceTreeSumTwice(wwrStream_t stream, const double* in, double* out,
                                   double* outNeg, const unsigned int* nactive,
                                   unsigned int nblocks, unsigned int threads);

// --- the warp-shuffle overload ---

wwrError_t blockReduceWarpsSum(wwrStream_t stream, const double* in, double* out,
                               const unsigned int* nactive, unsigned int nblocks,
                               unsigned int numWarps);

wwrError_t blockReduceWarpsMaxNan(wwrStream_t stream, const double* in, double* out,
                                  const unsigned int* nactive, unsigned int nblocks,
                                  unsigned int numWarps);

wwrError_t blockReduceWarpsInterval(wwrStream_t stream, const unsigned long long* in,
                                    unsigned long long* out, const unsigned int* nactive,
                                    unsigned int nblocks, unsigned int numWarps);

}  // namespace nevpt2::device
