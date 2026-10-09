// nevpt2's spelling of WarpWraps' runtime names: `using ::wwr::wwrX;` into
// namespace nevpt2, so code inside the namespace writes wwrStream_t,
// wwrError_t, wwrGetLastError() bare. Every one already carries the wwr
// prefix, so the `wwr::` scope in front of it said the same thing twice.
//
// The kernel-file half. A *.cu is C++20 and cannot `import`, so it reaches the
// runtime surface through WarpWraps' runtime.h -- whose wwrStream_t is
// declared in every compile, but whose wwrError_t, wwrSuccess,
// wwrGetLastError, ... only in a device pass (its section gated on
// __CUDACC__/__HIP__). The usings below are split along the same gate.
//
// Host code reaches these names through `import nevpt2.wwr;` (wwr/wwr.cppm,
// the one door host code imports), which #includes this header in its global module
// fragment, republishes wwrStream_t, and adds the device-gated runtime names
// plus the BLAS and solver ones from its imports. A *_bridge.h includes this
// header and is reached both ways: in a device pass it supplies the names
// itself; in a host unit -- which includes the bridge after its imports --
// wwrError_t arrives as nevpt2.wwr's export.
//
// Code outside namespace nevpt2 (a main(), an anonymous namespace at global
// scope) still writes wwr::wwrX. Add a runtime name a kernel file needs below,
// inside the gate; add it to wwr.cppm's list too when host code names it.
#pragma once

#include <runtime.h>

namespace nevpt2 {

using ::wwr::wwrStream_t;

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
using ::wwr::wwrError_t;
using ::wwr::wwrGetLastError;
using ::wwr::wwrSuccess;
#endif

}  // namespace nevpt2
