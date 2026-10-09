// Launch-shape constants and constraints (kWarpSize, num_warps) for kernel
// files.
//
// Device-only in practice: WWR_WARP_SIZE comes from WarpWraps' runtime.h,
// which defines it only in a device-compile pass (32 by default, 64 on a CDNA
// build), so this header has no module face.
//
// Reached root-relative as "common/block_params.h".
#pragma once

#include <concepts>

#include <runtime.h>  // WWR_WARP_SIZE

namespace nevpt2 {

// The warp/wavefront size of the build's target.
inline constexpr unsigned int kWarpSize = static_cast<unsigned int>(WWR_WARP_SIZE);

// A warps-per-block count N: integral, >= 1, a power of two, and at most 1024
// threads once multiplied by kWarpSize. The bound is checked as
// N <= 1024 / kWarpSize so a huge N cannot overflow the product. Constrains a
// value, not a type: `template <auto W> requires num_warps<W>`.
template <auto N>
concept num_warps = std::integral<decltype(N)> && (N >= 1) && ((N & (N - 1)) == 0) &&
                    // The widest unsigned type, so a 64-bit N (e.g. 1ull << 32)
                    // is not truncated to 0 and let through; N >= 1 already
                    // ruled out negatives.
                    (static_cast<unsigned long long>(N) <= 1024 / kWarpSize);

}  // namespace nevpt2
