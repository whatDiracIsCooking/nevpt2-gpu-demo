// The kernels behind the BlockReduceTreeTests / BlockReduceWarpsTests suites:
// both nevpt2::device::block_reduce overloads
// (src/common/block_reduce.cuh) under a sum, the NaN-winning maximum and the
// non-commutative interval probe, one fold per block. Launched through
// block_reduce_bridge.h, as a linked-in device library (add_device_library).
//
// The block shape is a template argument of both overloads, so each launcher
// dispatches the runtime `threads` / `numWarps` to an instantiation, walking
// the powers of two down from the largest.
#include "block_reduce_bridge.h"

#include "common/block_reduce.cuh"
#include "interval_probe.h"

namespace nevpt2::device {

namespace {

// --- the tree overload ---

template <unsigned int kBlock, typename T, typename Op>
__global__ void treeKernel(const T* in, T* out, const unsigned int* nactive, const Op op) {
  const unsigned int i = blockIdx.x * kBlock + threadIdx.x;
  out[i] = block_reduce<kBlock>(in[i], op, nactive[blockIdx.x]);
}

template <unsigned int kBlock>
__global__ void treeIntervalKernel(const unsigned long long* in, unsigned long long* out,
                                   const unsigned int* nactive) {
  const unsigned int i = blockIdx.x * kBlock + threadIdx.x;
  const Interval r =
      block_reduce<kBlock>(decodeInterval(in[i]), IntervalStructOp{}, nactive[blockIdx.x]);
  out[i] = encodeInterval(r);
}

template <unsigned int kBlock>
__global__ void treeSumTwiceKernel(const double* in, double* out, double* outNeg,
                                   const unsigned int* nactive) {
  const unsigned int i = blockIdx.x * kBlock + threadIdx.x;
  const unsigned int n = nactive[blockIdx.x];
  // No barrier between the two: the same instantiation, so the same static
  // buffer, and block_reduce's own leading/trailing barriers are what make
  // the second call safe.
  out[i] = block_reduce<kBlock>(in[i], AddOp{}, n);
  outNeg[i] = block_reduce<kBlock>(-in[i], AddOp{}, n);
}

// Launch `kernel<kBlock>` for the power of two kBlock == threads, searching
// from 1024 down; threads not a power of two in [1, 1024] is InvalidValue.
template <unsigned int kBlock, typename Launch>
wwrError_t dispatchTree(const unsigned int threads, const Launch& launch) {
  if (threads == kBlock) {
    launch.template operator()<kBlock>();
    return wwrGetLastError();
  }
  if constexpr (kBlock > 1) {
    return dispatchTree<kBlock / 2>(threads, launch);
  } else {
    return ::wwr::wwrErrorInvalidValue;
  }
}

// --- the warp-shuffle overload ---

template <unsigned int kNumWarps, typename T, typename Op>
__global__ void warpsKernel(const T* in, T* out, const unsigned int* nactive, const Op op) {
  __shared__ T partials[kNumWarps];
  const cg::thread_block block = cg::this_thread_block();
  const unsigned int i = blockIdx.x * kNumWarps * kWarpSize + threadIdx.x;
  out[i] = block_reduce(block, partials, in[i], op, nactive[blockIdx.x]);
}

// kNumWarps == numWarps among the powers of two num_warps admits, from 1024 /
// kWarpSize down.
template <unsigned int kNumWarps, typename T, typename Op>
wwrError_t dispatchWarps(const wwrStream_t stream, const T* in, T* out,
                         const unsigned int* nactive, const unsigned int nblocks,
                         const unsigned int numWarps, const Op op) {
  static_assert(num_warps<kNumWarps>);
  if (numWarps == kNumWarps) {
    warpsKernel<kNumWarps><<<nblocks, kNumWarps * kWarpSize, 0, stream>>>(in, out, nactive, op);
    return wwrGetLastError();
  }
  if constexpr (kNumWarps > 1) {
    return dispatchWarps<kNumWarps / 2>(stream, in, out, nactive, nblocks, numWarps, op);
  } else {
    return ::wwr::wwrErrorInvalidValue;
  }
}

template <typename T, typename Op>
wwrError_t launchWarps(const wwrStream_t stream, const T* in, T* out,
                       const unsigned int* nactive, const unsigned int nblocks,
                       const unsigned int numWarps, const Op op) {
  if (nblocks == 0) return wwrSuccess;
  return dispatchWarps<1024 / kWarpSize>(stream, in, out, nactive, nblocks, numWarps, op);
}

template <typename T, typename Op>
wwrError_t launchTree(const wwrStream_t stream, const T* in, T* out,
                      const unsigned int* nactive, const unsigned int nblocks,
                      const unsigned int threads, const Op op) {
  if (nblocks == 0) return wwrSuccess;
  return dispatchTree<1024>(threads, [&]<unsigned int kBlock>() {
    treeKernel<kBlock><<<nblocks, kBlock, 0, stream>>>(in, out, nactive, op);
  });
}

}  // namespace

unsigned int blockReduceWarpSize() { return kWarpSize; }

wwrError_t blockReduceTreeSum(const wwrStream_t stream, const double* in, double* out,
                              const unsigned int* nactive, const unsigned int nblocks,
                              const unsigned int threads) {
  return launchTree(stream, in, out, nactive, nblocks, threads, AddOp{});
}

wwrError_t blockReduceTreeMaxNan(const wwrStream_t stream, const double* in, double* out,
                                 const unsigned int* nactive, const unsigned int nblocks,
                                 const unsigned int threads) {
  return launchTree(stream, in, out, nactive, nblocks, threads, MaxNanOp{});
}

wwrError_t blockReduceTreeInterval(const wwrStream_t stream, const unsigned long long* in,
                                   unsigned long long* out, const unsigned int* nactive,
                                   const unsigned int nblocks, const unsigned int threads) {
  if (nblocks == 0) return wwrSuccess;
  return dispatchTree<1024>(threads, [&]<unsigned int kBlock>() {
    treeIntervalKernel<kBlock><<<nblocks, kBlock, 0, stream>>>(in, out, nactive);
  });
}

wwrError_t blockReduceTreeSumTwice(const wwrStream_t stream, const double* in, double* out,
                                   double* outNeg, const unsigned int* nactive,
                                   const unsigned int nblocks, const unsigned int threads) {
  if (nblocks == 0) return wwrSuccess;
  return dispatchTree<1024>(threads, [&]<unsigned int kBlock>() {
    treeSumTwiceKernel<kBlock><<<nblocks, kBlock, 0, stream>>>(in, out, outNeg, nactive);
  });
}

wwrError_t blockReduceWarpsSum(const wwrStream_t stream, const double* in, double* out,
                               const unsigned int* nactive, const unsigned int nblocks,
                               const unsigned int numWarps) {
  return launchWarps(stream, in, out, nactive, nblocks, numWarps, AddOp{});
}

wwrError_t blockReduceWarpsMaxNan(const wwrStream_t stream, const double* in, double* out,
                                  const unsigned int* nactive, const unsigned int nblocks,
                                  const unsigned int numWarps) {
  return launchWarps(stream, in, out, nactive, nblocks, numWarps, MaxNanOp{});
}

wwrError_t blockReduceWarpsInterval(const wwrStream_t stream, const unsigned long long* in,
                                    unsigned long long* out, const unsigned int* nactive,
                                    const unsigned int nblocks, const unsigned int numWarps) {
  return launchWarps(stream, in, out, nactive, nblocks, numWarps, IntervalOp{});
}

}  // namespace nevpt2::device
