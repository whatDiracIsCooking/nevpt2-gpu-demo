// nevpt2.wwr -- the one door every host unit reaches the GPU through:
// WarpWraps' wwr-prefixed names republished into namespace nevpt2, so host
// code inside the namespace writes wwrStream_t, wwrMemcpyAsync, wwrblasDgemm,
// WWRBLAS_OP_N bare instead of behind a second `wwr::`, plus the error checks
// that go with them.
//
// `import nevpt2.wwr;` alone names the runtime surface (wwr.runtime_api is
// re-exported, so code outside namespace nevpt2 -- the main()s -- writes
// wwr::wwrX), the bare usings below, gpuCheck (defined here), and
// nevpt2.error_handling's vendor-free Error / Result / Status / check
// (re-exported). The wwr* names bind to ::cuda* or ::hip* by
// whichever backend the build selected, so host code is written ONCE and
// builds unchanged for either. (This was nevpt2.gpu, a separate re-exporting
// module, until it was folded in here; nevpt2.common stays GPU-free.)
//
// Every kernel is a device library linked into the binary and launched
// through the runtime API (a *_bridge.h per kernel file), so nothing here
// loads code at run time.
//
// wwr.h is the kernel-file half and the single source of what the two
// share: it arrives by #include in the global module fragment and the purview
// republishes it with `export using` (a GMF declaration is never implicitly
// exported -- the same shape as nevpt2.common over align_up.h / int64.h).
// What the header gates to a device pass (wwrError_t, wwrSuccess,
// wwrGetLastError) and the BLAS / solver surfaces, which no kernel file uses,
// are bound here directly from the imports.
//
// Only names the tree uses are listed: a new wwr* call that does not compile
// bare needs its `using` added below, not a `wwr::` in front of it.
// wwr::extension::* is a namespace, not a prefixed name, and stays qualified.
//
// Below the names: WarpWraps' RAII wrappers (wwr.extension.*) instantiated
// with nevpt2's error policy, Abort, in every slot -- the stream, pool, BLAS
// and solver handle wrappers DeviceResources holds, and the device-buffer
// suite. The suite is a template over the handle type because the handle,
// DeviceResources, sits above this module (nevpt2.device_resources), which
// closes it into DeviceBuffer<T> / DeviceBufferView<T>.
module;

// WWR_SELECTED_CUDA / _HIP: the same switch the wwr* names are bound by, so
// kGpuBackendName can never disagree with the backend the runtime calls go to.
#include <selected_backend.h>

#include "wwr.h"

export module nevpt2.wwr;

import std;
export import wwr.runtime_api;
export import nevpt2.error_handling;
import wwr.blas;
import wwr.solver;
import wwr.extension.common;  // gpu_check, error_type, kit::AbortPolicy + the wwrError_t trait
// The wwrblasStatus_t / wwrsolverStatus_t error_type traits, and
// BlasHandleWrapper / SolverDnHandleWrapper.
import wwr.extension.blas;
import wwr.extension.solver;
import wwr.extension.runtime;  // StreamWrapper, MemPoolWrapper
// Re-exported: a DeviceBuffer<T> is a DeviceBufferWrapper, whose interface
// (data(), num_elements(), ...) every holder of one calls.
export import wwr.extension.memory_buffer;

export namespace nevpt2 {

// ---------------------------------------------------------------------------
// The GPU-status check. (It is here, not in the vendor-free
// nevpt2.error_handling, because this is the module that knows the wwr*
// status types.)
//
// This demo's GPU error model is "abort with the caller's file:line": gpuCheck
// hands any status WarpWraps' error layer knows to its gpu_check with
// kit::AbortPolicy, which prints "GPU error at file:line in function: NAME
// (description)" to stderr and calls std::abort(). A linked-in launcher's
// status goes through it too: every *_bridge.h launcher returns wwrError_t
// (see rdm/rdm_accumulate_bridge.h). There is deliberately no separate
// out-of-memory policy: an allocation failure aborts like any other. During
// the RDM build that means raise --tiles (docs/performance.md, "What too
// few tiles looks like").
// ---------------------------------------------------------------------------

// The abort-on-failure policy gpuCheck uses: WarpWraps' opt-in kit policy
// (wwr.extension.common). The RAII wrappers below carry it in every slot.
template <class E>
using Abort = ::wwr::extension::kit::AbortPolicy<E>;

// Aborts with the caller's file:line (and enclosing function) on any
// non-success status. A function rather than a macro because a macro cannot
// cross a module boundary. Generic over every status type WarpWraps' error
// layer knows (wwr::extension::error_type): wwrError_t, wwrblasStatus_t and
// wwrsolverStatus_t all come with this module.
template <::wwr::extension::error_type E>
void gpuCheck(E res, std::source_location loc = std::source_location::current()) {
  ::wwr::extension::gpu_check(res, Abort<E>{}, loc);
}

// There is no stream here. Every GPU call in both demos -- kernel launches,
// copies, memsets, allocations, frees, BLAS calls, timing events, syncs -- is
// issued on the one non-blocking stream of the demo's nevpt2::DeviceResources
// (nevpt2.device_resources), passed down explicitly from main(). Nothing
// names the legacy default stream; devtools/stream-lint.sh --strict fails the
// local gate on a stream-less call or a literal nullptr/0 stream
// (docs/performance.md, "One stream").

// "CUDA" or "HIP" -- for the banner, the --cublas refusal and the per-backend
// RDM-build defaults.
#if defined(WWR_SELECTED_CUDA)
inline constexpr const char* kGpuBackendName = "CUDA";
#else
inline constexpr const char* kGpuBackendName = "HIP";
#endif

// From wwr.h.
using nevpt2::wwrStream_t;

// Runtime (wwr.runtime_api).
using ::wwr::wwrDeviceProp;
using ::wwr::wwrError_t;
using ::wwr::wwrEvent_t;
using ::wwr::wwrEventCreateWithFlags;
using ::wwr::wwrEventDefault;
using ::wwr::wwrEventDestroy;
using ::wwr::wwrEventElapsedTime;
using ::wwr::wwrEventRecord;
using ::wwr::wwrFreeAsync;
using ::wwr::wwrGetDeviceProperties;
using ::wwr::wwrGetErrorName;
using ::wwr::wwrGetErrorString;
using ::wwr::wwrGetLastError;
using ::wwr::wwrMallocAsync;
using ::wwr::wwrMemcpyAsync;
using ::wwr::wwrMemcpyDeviceToHost;
using ::wwr::wwrMemcpyHostToDevice;
using ::wwr::wwrMemPool_t;
using ::wwr::wwrMemPoolAttrReservedMemHigh;
using ::wwr::wwrMemPoolAttrUsedMemCurrent;
using ::wwr::wwrMemPoolAttrUsedMemHigh;
using ::wwr::wwrMemPoolGetAttribute;
using ::wwr::wwrMemsetAsync;
using ::wwr::wwrSetDevice;
using ::wwr::wwrStreamCreateWithFlags;
using ::wwr::wwrStreamDestroy;
using ::wwr::wwrStreamNonBlocking;
using ::wwr::wwrStreamSynchronize;
using ::wwr::wwrSuccess;

// BLAS (wwr.blas).
using ::wwr::WWRBLAS_FILL_MODE_LOWER;
using ::wwr::WWRBLAS_OP_N;
using ::wwr::WWRBLAS_OP_T;
using ::wwr::wwrblasDgemm;
using ::wwr::wwrblasDgemm_64;
using ::wwr::wwrblasDgemmStridedBatched_64;
using ::wwr::wwrblasHandle_t;
using ::wwr::wwrblasStatus_t;

// Dense solver (wwr.solver).
using ::wwr::WWRSOLVER_EIG_MODE_VECTOR;
using ::wwr::wwrsolverDnDsyevd;
using ::wwr::wwrsolverDnDsyevd_bufferSize;
using ::wwr::wwrsolverDnHandle_t;
using ::wwr::wwrsolverStatus_t;

// ---------------------------------------------------------------------------
// WarpWraps' RAII wrappers, with Abort (above) in every policy slot.
// ---------------------------------------------------------------------------

// The wrappers DeviceResources owns.
// Both handle wrappers are bound to a StreamWrapper, which they co-own.
using StreamWrapper =
    ::wwr::extension::StreamWrapper<Abort<wwrError_t>, Abort<wwrError_t>, Abort<wwrError_t>>;
using MemPoolWrapper =
    ::wwr::extension::MemPoolWrapper<Abort<wwrError_t>, Abort<wwrError_t>, Abort<wwrError_t>>;
using BlasHandleWrapper =
    ::wwr::extension::BlasHandleWrapper<Abort<wwrblasStatus_t>, Abort<wwrblasStatus_t>,
                                        StreamWrapper, Abort<wwrError_t>>;
using SolverDnHandleWrapper =
    ::wwr::extension::SolverDnHandleWrapper<Abort<wwrsolverStatus_t>, Abort<wwrsolverStatus_t>,
                                            StreamWrapper, Abort<wwrError_t>>;

// The device buffers: WarpWraps' buffer suite, Abort in every
// policy slot, backed by `Handles` -- a device_handle_pool (dev_idx(),
// stream(), pool()). nevpt2.device_resources instantiates it over
// `const DeviceResources`; nothing else should.
template <class Handles>
using DeviceBufferSuite =
    ::wwr::extension::kit::device_buffer_suite<::wwr::extension::kit::single_policy_map<Abort>,
                                               Handles>;

}  // namespace nevpt2
