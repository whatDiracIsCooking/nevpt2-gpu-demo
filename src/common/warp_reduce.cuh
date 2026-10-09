// Warp-wide register reduction over a cooperative-groups tile.
//
// nevpt2::device::warp_reduce folds one value per lane across a
// kWarpSize-wide thread_block_tile under an associative binary op and returns
// the fold in every lane: a tile.shfl_down ladder into lane 0, then one
// tile.shfl broadcast. No shared memory and no barriers -- a shuffle is a
// warp-level collective, so the lanes stay in step by construction.
//
// The fold is identity-free and order-preserving: only the first `nactive`
// lanes' values enter, combined left to right in lane order, so the op needs
// associativity only -- no identity, no commutativity. (Floating-point `+` is
// not associative, so the fold order is a fixed reassociation of the serial
// sum, like every other device sum here: compared to the golden to 1e-7.)
//
// The value type is arithmetic: HIP's tile shuffles resolve only for vendor
// scalar types, so a class type cannot ride a shuffle portably --
// block_reduce.cuh's shared-memory tree carries those.
//
// #include'd into a kernel file only: no module face. Reached root-relative as
// "common/warp_reduce.cuh"; the target links nevpt2::common::device for the
// src/ root and, through it, wwr::device (cooperative_groups.h,
// device_guard.h).
#pragma once

#include <type_traits>

#include "block_params.h"
#include "device_functor.h"

// The device-pass gate: #errors outside a CUDA or HIP device compile.
#include <device_guard.h>

// The vendor cooperative-groups header, resolved per backend by WarpWraps.
#include <cooperative_groups.h>

namespace nevpt2::device {

namespace cg = cooperative_groups;

// Fold v across warp_tile under op, result valid in every lane.
//
// Lanes with thread_rank() >= nactive contribute nothing (their v is ignored,
// so no identity is needed) but must still call: every shuffle is a tile-wide
// collective.
//
//   ParentT  deduced, so tiled_partition's result binds as-is (HIP only
//            converts a parented tile to the <kWarpSize, void> one)
//   Op       `__device__ T operator()(T, T) const`, associative
//   nactive  lanes whose v enters the fold; 1 <= nactive <= kWarpSize
template <typename ParentT, typename T, device_functor Op>
  requires std::is_arithmetic_v<T>
__device__ __forceinline__ T warp_reduce(cg::thread_block_tile<kWarpSize, ParentT>& warp_tile,
                                         T v, const Op op,
                                         const unsigned int nactive = kWarpSize) {
  const unsigned int lane = warp_tile.thread_rank();
  // Offsets grow, not halve: before each step lane t < nactive holds the fold
  // of the contiguous [t, min(t + offset, nactive)), and appends its
  // neighbour's [t + offset, ...) on the right. A halving ladder would pair
  // strided lanes (v0 with v16) and silently need commutativity. Every lane
  // shuffles; only the combine is guarded, so lanes >= nactive are never
  // folded in.
  for (unsigned int offset = 1; offset < kWarpSize; offset <<= 1) {
    const T other = warp_tile.shfl_down(v, offset);
    if (lane + offset < nactive) {
      v = op(v, other);
    }
  }
  return warp_tile.shfl(v, 0);
}

}  // namespace nevpt2::device
