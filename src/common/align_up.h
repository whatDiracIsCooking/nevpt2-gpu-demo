// Integer ceiling-divide and round-up-to-a-multiple (idivup, align_up), as a
// header both a kernel file and a module's global module fragment can reach.
//
// A plain header, not a module unit: grid sizing straddles the host/device
// line, and a kernel file (C++20, a plain TU) cannot `import`. common.cppm
// includes this in its global module fragment and re-exports the names, so
// `import nevpt2.common;` and `#include "common/align_up.h"` see the same
// `nevpt2::` spellings. Both helpers are NEVPT2_HOST_DEVICE, so kernels call
// them too.
//
// Reached root-relative as "common/align_up.h".
#pragma once

#include <concepts>

#include "host_device.h"

namespace nevpt2 {

// Ceiling of a / b: the number of b-sized chunks needed to cover a. Assumes
// non-negative operands; undefined for b == 0, and a + b may overflow near the
// top of T's range (rdm_build.cpp's maxLen split, which must not, spells its
// own).
template <std::integral T>
NEVPT2_HOST_DEVICE constexpr T idivup(const T a, const T b) {
  return (a + b - 1) / b;
}

// Round a up to the next multiple of b. Same domain as idivup.
template <std::integral T>
NEVPT2_HOST_DEVICE constexpr T align_up(const T a, const T b) {
  return idivup(a, b) * b;
}

}  // namespace nevpt2
