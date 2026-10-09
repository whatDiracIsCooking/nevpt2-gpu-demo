// nevpt2.common -- the module face of the shared building blocks a host unit
// needs: idivup and align_up (align_up.h), the int64_t size type (int64.h),
// and narrowTo, the one checked way a host int64_t becomes a narrower or unsigned
// integer (a 32-bit BLAS dimension, a kernel launcher's int argument, a
// std::size_t extent, ...).
//
// The header definitions live in headers so a kernel file can share them; a
// global module fragment cannot `import`, so the headers arrive by #include
// here and `export using` republishes each name under its original `nevpt2::`
// spelling (a declaration in the global module fragment is never implicitly
// exported). Importers and #includers therefore see identical names. Add a
// shared helper in ONE place -- the header -- then add its `using` below.
//
// narrowTo is host-only, so it is defined here rather than in a header. It is
// a function rather than a macro because a macro cannot cross a module
// boundary; like nevpt2::gpuCheck it takes the caller's std::source_location,
// so a failure names the call site, not this file. An out-of-range value is a
// broken invariant (our bug), so the failure goes through nevpt2.error_handling's
// abort tier, check(): "error at file:line in function: ..." on stderr, then
// std::abort(). That module is vendor-free (it imports only std), so importing
// it keeps nevpt2.common GPU-free.
//
// The rest of common/ is device-only and has no module face: block_params.h
// (WWR_WARP_SIZE exists only in a device pass), device_index.h (the idx*
// flattening helpers and gridFor), device_functor.h, and the
// reductions warp_reduce.cuh / block_reduce.cuh. Kernel files reach those
// root-relative ("common/block_reduce.cuh") through nevpt2::common::device.
module;

#include "align_up.h"
#include "int64.h"

export module nevpt2.common;

import std;
import nevpt2.error_handling;  // check(): narrowTo's failure path

// The tree spells a 32-bit integer `int` -- the link tables, every kernel
// launcher argument narrowTo<int> produces, the golden file's 4-byte header
// fields -- and never std::int32_t. That is only right where int is 32 bits,
// which every platform a CUDA or ROCm toolchain targets (LP64, LLP64) gives;
// this makes any other one a compile error rather than a silent misread.
static_assert(sizeof(int) == 4, "nevpt2 spells a 32-bit integer `int`");

// Not exported: narrowTo's cold path, visible inside this module only. It
// formats the message and hands it to check(false, ...), which aborts.
namespace nevpt2 {
[[noreturn]] inline void narrowFailed(int64_t v, std::size_t bits, bool isSigned,
                                      std::string_view hint, const std::source_location& loc) {
  check(false,
        std::format("{} does not fit in a {}-bit {} integer{}{}", v, bits,
                    isSigned ? "signed" : "unsigned", hint.empty() ? "" : "; ", hint),
        loc);
  std::unreachable();  // check(false, ...) aborts
}
}  // namespace nevpt2

export namespace nevpt2 {

using nevpt2::align_up;
using nevpt2::idivup;
using nevpt2::int64_t;

// v as a To, or -- when v is outside To's range -- a message with the
// caller's file:line (and enclosing function) on stderr and std::abort(),
// through check(). `hint` is appended to the message
// (e.g. "use more --tiles" where a tile width sizes the value).
template <std::integral To>
To narrowTo(int64_t v, std::string_view hint,
            std::source_location loc = std::source_location::current()) {
  if (!std::in_range<To>(v))
    narrowFailed(v, sizeof(To) * 8, std::is_signed_v<To>, hint, loc);
  return static_cast<To>(v);
}

template <std::integral To>
To narrowTo(int64_t v, std::source_location loc = std::source_location::current()) {
  return narrowTo<To>(v, std::string_view{}, loc);
}

}  // namespace nevpt2
