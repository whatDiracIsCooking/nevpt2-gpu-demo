// warp_reduce_bridge.h -- the launchers warp_reduce_tests.cpp (host, a plain
// TU that imports modules) calls into warp_reduce_kernels.cu (device-compiled,
// linked in as the nevpt2.test.warp_reduce.device static library). Same shape
// as src/rdm/rdm_accumulate_bridge.h, and for the same reasons: the host TU
// includes it AFTER its imports (wwrError_t comes from nevpt2.wwr there), and
// only plain types cross it -- nvcc's host pass is libstdc++, the tests libc++.
// (A plain TU has no module purview, so no extern "C++" is needed.)
//
// Every launcher runs `nwarps` independent warp folds, kWarpReduceWarpsPerBlock
// warps to a block (so tiles other than a block's first are exercised): warp w
// folds in[w * W, (w + 1) * W) with nactive[w] active lanes and writes its
// result from EVERY lane to out[w * W + lane], W = warpReduceWarpSize(). Each
// returns the launch status, for the caller's gpuCheck.
#pragma once

#include "wwr/wwr.h"  // wwrStream_t (+ wwrError_t in a device pass)

namespace nevpt2::device {

inline constexpr unsigned int kWarpReduceWarpsPerBlock = 4;

// kWarpSize as the device pass that built warp_reduce compiled it.
unsigned int warpReduceWarpSize();

// warp_reduce under AddOp.
wwrError_t warpReduceSum(wwrStream_t stream, const double* in, double* out,
                         const unsigned int* nactive, unsigned int nwarps);

// warp_reduce under MaxNanOp.
wwrError_t warpReduceMaxNan(wwrStream_t stream, const double* in, double* out,
                            const unsigned int* nactive, unsigned int nwarps);

// warp_reduce under IntervalOp (interval_probe.h), on the encoded interval.
wwrError_t warpReduceInterval(wwrStream_t stream, const unsigned long long* in,
                              unsigned long long* out, const unsigned int* nactive,
                              unsigned int nwarps);

}  // namespace nevpt2::device
