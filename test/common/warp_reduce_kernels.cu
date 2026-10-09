// The kernels behind the WarpReduceTests suite: nevpt2::device::
// warp_reduce (src/common/warp_reduce.cuh) under a sum, NaN-winning maximum and
// the non-commutative interval probe, one warp fold per tile. Launched through
// warp_reduce_bridge.h, as a linked-in device library (add_device_library).
#include "warp_reduce_bridge.h"

#include "common/align_up.h"       // idivup
#include "common/block_reduce.cuh"  // AddOp, MaxNanOp
#include "common/warp_reduce.cuh"
#include "interval_probe.h"

namespace nevpt2::device {

namespace {

constexpr unsigned int kBlockThreads = kWarpReduceWarpsPerBlock * kWarpSize;

template <typename T, typename Op>
__global__ void warpReduceKernel(const T* in, T* out, const unsigned int* nactive,
                                 const unsigned int nwarps, const Op op) {
  const cg::thread_block block = cg::this_thread_block();
  auto warp_tile = cg::tiled_partition<kWarpSize>(block);
  const unsigned int warp = blockIdx.x * kWarpReduceWarpsPerBlock + warp_tile.meta_group_rank();
  // Warp-uniform: a warp past the last one leaves as a whole, so every
  // shuffle below still sees its full tile.
  if (warp >= nwarps) return;
  const unsigned int i = warp * kWarpSize + warp_tile.thread_rank();
  out[i] = warp_reduce(warp_tile, in[i], op, nactive[warp]);
}

template <typename T, typename Op>
wwrError_t launch(const wwrStream_t stream, const T* in, T* out, const unsigned int* nactive,
                  const unsigned int nwarps, const Op op) {
  if (nwarps == 0) return wwrSuccess;
  warpReduceKernel<<<idivup(nwarps, kWarpReduceWarpsPerBlock), kBlockThreads, 0, stream>>>(in, out, nactive, nwarps, op);
  return wwrGetLastError();
}

}  // namespace

unsigned int warpReduceWarpSize() { return kWarpSize; }

wwrError_t warpReduceSum(const wwrStream_t stream, const double* in, double* out,
                         const unsigned int* nactive, const unsigned int nwarps) {
  return launch(stream, in, out, nactive, nwarps, AddOp{});
}

wwrError_t warpReduceMaxNan(const wwrStream_t stream, const double* in, double* out,
                            const unsigned int* nactive, const unsigned int nwarps) {
  return launch(stream, in, out, nactive, nwarps, MaxNanOp{});
}

wwrError_t warpReduceInterval(const wwrStream_t stream, const unsigned long long* in,
                              unsigned long long* out, const unsigned int* nactive,
                              const unsigned int nwarps) {
  return launch(stream, in, out, nactive, nwarps, IntervalOp{});
}

}  // namespace nevpt2::device
