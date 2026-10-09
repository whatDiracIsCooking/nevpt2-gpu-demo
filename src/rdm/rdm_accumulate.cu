// The kernel the device-resident RDM build added: an on-device tile-partial
// accumulate, replacing the download-host_add-discard round trip the
// non-resident launcher paid once per tile (see rdm_build.cpp and the
// docs/performance.md, "Device-resident RDM build" section). One fixed kernel for any
// n^6-sized accumulator: the emitted path's dm3, and the f3 ac scatter on the
// GEMM digests (the emitted f3 digest adds into its accumulator itself,
// f3_digest.cu).
//
// LINKED IN, like every kernel here:
// a STATIC device library (add_device_library over wwr_add_gpu_device_library, see CMakeLists.txt)
// whose launcher rdm_build.cpp calls through rdm_accumulate_bridge.h. The
// launch goes through WarpWraps' parallel_for (index-per-thread, 4 warps per
// block). Shared unchanged between backends: under HIP the .cu is compiled
// with -x hip.
#include "rdm/rdm_accumulate_bridge.h"

#include <extension/parallel_for/parallel_for.cuh>

namespace nevpt2::device {

namespace {

// Const scalar members: parallel_for's device_functor requires the functor
// not be copy-assignable (it is passed as a grid constant). The pointers are
// const, not what they point at.
struct AccumulateFunctor {
  double* const acc;
  const double* const part;

  __device__ void operator()(const long long i) const { acc[i] += part[i]; }
};

}  // namespace

wwrError_t accumulateInplace(const wwrStream_t stream, double* acc, const double* part,
                             const long long n) {
  return ::wwr::extension::parallel_for(stream, n, AccumulateFunctor{acc, part});
}

}  // namespace nevpt2::device
