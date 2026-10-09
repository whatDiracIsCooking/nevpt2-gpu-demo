// A plain POD description of one einsum contraction, shared between the host
// planner (einsum/einsum.cpp, which parses a subscript string and fills
// this in) and the device kernel (device_einsum.cu, which only ever reads it).
// No vendor types here so this header includes cleanly from both the device
// compiler (C++20) and a host module unit's global module fragment (C++23).
//
// It crosses device_einsum_bridge.h BY VALUE between clang + libc++ and the
// .cu's host half (nvcc's own host compiler), so it MUST stay a plain POD: no
// std:: members, nothing with a library-defined layout. The static_assert
// below holds it to that.
// Reached root-relative, "einsum/einsum_plan.h" (nevpt2::einsum::plan).
//
// Sizing: `kEinsumMaxOperandRank`=6, `kEinsumMaxContracted`=6 and
// `kEinsumMaxLabels`=8 are the exact maxima measured across every call in
// energy/energy.cpp; the later call sites (energy_pc.cpp, df_integrals.cpp,
// device_tensor.cppm) stay within them (the largest
// operand/output rank is 6, the most contracted
// labels in one call is 6, the most *distinct* labels total in one call is 8
// -- einsum.cpp's buildPlan aborts (check()) if a call ever exceeds them). Not a
// round-number guess.
#pragma once

#include <type_traits>

#include "common/int64.h"

namespace nevpt2 {

constexpr int kEinsumMaxOperands = 3;
constexpr int kEinsumMaxOperandRank = 6;
constexpr int kEinsumMaxContracted = 6;
constexpr int kEinsumMaxLabels = 8;  // output labels + contracted labels

struct EinsumPlan {
  int nOperands = 0;
  int opRank[kEinsumMaxOperands] = {};
  // opAxisLabel[o][a] is the label-slot index (into the 0..outRank-1 output
  // labels, then outRank..outRank+nContractedLabels-1 contracted labels)
  // that operand o's axis a reads.
  int opAxisLabel[kEinsumMaxOperands][kEinsumMaxOperandRank] = {};
  int64_t opAxisStride[kEinsumMaxOperands][kEinsumMaxOperandRank] = {};

  int outRank = 0;
  int64_t outDims[kEinsumMaxOperandRank] = {};  // to decode a flat out index

  int nContractedLabels = 0;
  int64_t contractedDims[kEinsumMaxContracted] = {};

  int64_t outSize = 1;
  int64_t contractedSize = 1;
};

static_assert(std::is_trivially_copyable_v<EinsumPlan> && std::is_standard_layout_v<EinsumPlan>,
              "EinsumPlan crosses device_einsum_bridge.h by value: keep it a POD");

}  // namespace nevpt2
