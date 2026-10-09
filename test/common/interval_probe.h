// interval_probe.h -- the non-commutative lane-order probe the warp_reduce /
// block_reduce suites fold.
//
// Both reductions promise an identity-free, ORDER-PRESERVING fold: only the
// first `nactive` values enter, combined left to right in lane/thread order, so
// the op needs associativity only (common/warp_reduce.cuh). A sum or a maximum
// cannot see a broken order -- both are commutative. This op can:
//
//   an interval [first, last] of lane indices; op(a, b) appends b to a, and is
//   the interval [a.first, b.last] when b starts right where a ends
//   (a.last + 1 == b.first) and BAD otherwise. BAD absorbs.
//
// That is associative (concatenation of contiguous ranges, with an absorbing
// error) and not commutative (op([0,0], [1,1]) is [0,1], op([1,1], [0,0]) is
// BAD). Lane t starts as [t, t], so a correct fold of nactive lanes is exactly
// [0, nactive - 1]: a lane folded out of order -- a halving ladder pairing lane
// 0 with lane 16 -- gives BAD, and an inactive lane folded in either gives BAD
// (the suites poison inactive lanes with it) or pushes `last` past nactive - 1.
//
// Two carriers, one encoding. warp_reduce and block_reduce's warp overload need
// an arithmetic type (HIP's tile shuffles take only vendor scalars), so the
// interval rides one unsigned long long: bit 63 BAD, bits 32..62 `first`, bits
// 0..31 `last`. block_reduce's shared-memory tree also carries class types, so
// its probe folds the Interval struct below and encodes only at the boundary.
//
// Plain types and NEVPT2_HOST_DEVICE only: included by the .cu kernel files
// (device pass) and by the host test TUs, which build inputs and expected
// values with the same functions.
#pragma once

#include "common/host_device.h"

namespace nevpt2::device {

inline constexpr unsigned long long kIntervalBad = 1ull << 63;

// The encoded interval [first, last]; first < 2^31.
NEVPT2_HOST_DEVICE constexpr unsigned long long encodeInterval(const unsigned int first,
                                                               const unsigned int last) {
  return (static_cast<unsigned long long>(first) << 32) | last;
}

// op on the encoding: [a.first, b.last] when b continues a, else kIntervalBad.
struct IntervalOp {
  NEVPT2_HOST_DEVICE unsigned long long operator()(const unsigned long long a,
                                                   const unsigned long long b) const {
    if (((a | b) & kIntervalBad) != 0) return kIntervalBad;
    const unsigned long long aLast = a & 0xffffffffull;
    const unsigned long long bFirst = b >> 32;
    if (aLast + 1 != bFirst) return kIntervalBad;
    return (a & ~0xffffffffull) | (b & 0xffffffffull);
  }
};

// The same interval as a class type, for block_reduce's tree.
struct Interval {
  unsigned int first;
  unsigned int last;
  bool bad;
};

NEVPT2_HOST_DEVICE constexpr Interval decodeInterval(const unsigned long long e) {
  return Interval{static_cast<unsigned int>((e >> 32) & 0x7fffffffull),
                  static_cast<unsigned int>(e & 0xffffffffull), (e & kIntervalBad) != 0};
}

NEVPT2_HOST_DEVICE constexpr unsigned long long encodeInterval(const Interval i) {
  return i.bad ? kIntervalBad : encodeInterval(i.first, i.last);
}

// op on the struct, the same rule.
struct IntervalStructOp {
  NEVPT2_HOST_DEVICE Interval operator()(const Interval a, const Interval b) const {
    if (a.bad || b.bad || a.last + 1 != b.first) return Interval{0, 0, true};
    return Interval{a.first, b.last, false};
  }
};

}  // namespace nevpt2::device
