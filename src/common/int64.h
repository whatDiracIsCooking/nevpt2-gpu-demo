// nevpt2::int64_t -- std::int64_t, the one signed 64-bit type every host size,
// extent and element count is held in, on the host and the device alike.
// Narrowing it to a 32-bit API argument is never a bare static_cast: host
// code goes through nevpt2::narrowTo<int>
// (nevpt2.common), which fails loudly, with the caller's location, when the
// value does not fit.
//
// The one using-declaration of a std name in the tree, on purpose: everywhere
// else std names are qualified, but this type is in nearly every signature, so
// code inside namespace nevpt2 spells it bare. Code outside the namespace
// writes std::int64_t. (Under `import std` the global ::int64_t is not
// guaranteed, so a bare spelling outside nevpt2 would not compile.)
//
// Signed: extents are subtracted and compared, and a negative intermediate
// must stay visible as negative rather than wrap. 64-bit: the n^6 RDM
// accumulators and the per-tile L2 block (n^4 * tile width) pass 2^31
// elements well inside the active spaces this project runs.
//
// A plain header, not only a module, because the kernel files (*.cu) are
// C++20 and cannot `import`. Nothing past <cstdint>, so it includes cleanly
// from the device compiler and from a host module unit's global module
// fragment alike. Host code reaches it through `import nevpt2.common;`, which
// re-exports it; kernel files through device_index.h ("common/int64.h").
#pragma once

#include <cstdint>

namespace nevpt2 {

using std::int64_t;

}  // namespace nevpt2
