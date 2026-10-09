// device_functor: the trivially-copyable functor concept the reductions
// (warp_reduce.cuh, block_reduce.cuh) constrain their fold op with.
//
// It names only the host-checkable half of a device functor's contract: a
// trivially-copyable class type. The callability probe (`{ f(a, b) } -> T`)
// is deliberately NOT here -- in a host context it would reject a
// `__device__`-only operator(), so each consuming device function carries its
// own `requires`. The same name and split as WarpWraps' parallel_for
// device_functor, minus its !is_copy_assignable conjunct: these functors are
// plain arguments, never WWR_GRID_CONSTANT, so a const member is not required.
//
// A plain trait, so it #includes anywhere. Reached root-relative as
// "common/device_functor.h".
#pragma once

#include <type_traits>

namespace nevpt2::device {

template <typename F>
concept device_functor = std::is_trivially_copyable_v<F> && std::is_class_v<F>;

}  // namespace nevpt2::device
