// rdm_accumulate_bridge.h -- the declaration shared between rdm_build.cpp
// (host C++23 module unit) and rdm_accumulate.cu (device-compiled, linked in
// as the nevpt2.rdm.accumulate.device static library).
//
// Linked in, like every kernel in the tree, after WarpWraps'
// wwr.extension.init_state pattern, and the bridge the others point to: the
// launch is an ordinary,
// type-checked function call, not a look-up-by-name + void** args.
//
// Included by rdm_accumulate.cu directly, and by rdm_build.cpp in its
// purview, AFTER its imports and inside `extern "C++" { ... }`. Both halves of
// that are load-bearing:
//   - extern "C++": a name declared in a module's purview is otherwise
//     attached to the module and gets module linkage, so it could never bind
//     to the definition in a plain TU, which is what rdm_accumulate.cu is. A
//     declaration inside a linkage-specification is attached to the GLOBAL
//     module instead -- the same attachment the GMF gives, and the same
//     mangled name.
//   - after the imports: the launchers return wwrError_t, which a HOST
//     include of wwr/wwr.h does not declare (runtime.h's always-on section
//     carries wwrStream_t only; the rest of the runtime surface is in its
//     device-pass section, and wwr.h gates its usings the same way). In the
//     GMF nevpt2::wwrError_t would not be declared yet; after
//     `import nevpt2.wwr;` it is. The includer
//     puts runtime.h itself in its GMF, so the vendor headers stay out of the
//     purview and wwr.h's own #include of it is a #pragma once no-op there.
// A .cu gets wwrError_t from wwr.h's device-pass usings.
//
// Plain types only across this boundary (pointers, integers, the vendor
// stream handle, the vendor error enum by its wwr* name): nvcc compiles the
// .cu's host half with its own host compiler, which need not share libc++'s
// ABI with the module units.
#pragma once

#include "wwr/wwr.h"  // wwrStream_t (+ wwrError_t in a device pass)

namespace nevpt2::device {

// acc[i] += part[i] for i in [0, n), on `stream`. Returns the LAUNCH status
// (bad configuration, resource limits), wwrSuccess on success, for the
// caller's gpuCheck; a kernel's own execution errors surface at the next
// synchronization, as with any launch.
wwrError_t accumulateInplace(wwrStream_t stream, double* acc, const double* part, long long n);

}  // namespace nevpt2::device
