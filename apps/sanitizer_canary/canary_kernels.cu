// The sanitizer canary's kernels: each one wrong on purpose (canary_bridge.h).
// A device library like every other kernel file, so it builds for both
// backends; only the CUDA build has a tool (compute-sanitizer) that runs it.
#include "canary_bridge.h"
#include "common/device_index.h"  // gridFor

namespace nevpt2::canary {

namespace {

__global__ void copyKernel(const double* in, double* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i];
}

__global__ void readPastEndKernel(const double* in, double* out, int n) {
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = in[n];
}

// Thread 0 writes the slot, every thread reads it, no barrier: a read-after-
// write hazard between warp 0 and warp 1. volatile, so the compiler cannot
// keep the value in a register and hide the shared-memory traffic. (The form
// where EVERY thread writes and then reads its own value is not reported:
// each thread's read is forwarded from its own write.)
__global__ void sharedRaceKernel(double* out) {
  __shared__ volatile double slot;
  if (threadIdx.x == 0) slot = 1.0;
  out[threadIdx.x] = slot;
}

// Warp 0 and warp 1 wait at two DIFFERENT __syncthreads() calls. Runs to
// completion on sm_70+ without the tool (the hardware barrier is the same
// one), which is exactly why it needs synccheck. (Measured on the RTX 3080: a
// __syncthreads() that only some threads reach, with the rest skipping it, is
// NOT reported by synccheck there -- the form below is.)
__global__ void divergentBarrierKernel(double* out) {
  if (threadIdx.x < 32) {
    __syncthreads();
    out[threadIdx.x] = 1.0;
  } else {
    __syncthreads();
    out[threadIdx.x] = 2.0;
  }
}

}  // namespace

wwrError_t launchCopy(const wwrStream_t stream, const double* in, double* out, const int n) {
  copyKernel<<<gridFor(n, 128), 128, 0, stream>>>(in, out, n);
  return wwrGetLastError();
}

wwrError_t launchReadPastEnd(const wwrStream_t stream, const double* in, double* out, const int n) {
  readPastEndKernel<<<1, 1, 0, stream>>>(in, out, n);
  return wwrGetLastError();
}

wwrError_t launchSharedRace(const wwrStream_t stream, double* out) {
  sharedRaceKernel<<<1, 64, 0, stream>>>(out);
  return wwrGetLastError();
}

wwrError_t launchDivergentBarrier(const wwrStream_t stream, double* out) {
  divergentBarrierKernel<<<1, 64, 0, stream>>>(out);
  return wwrGetLastError();
}

}  // namespace nevpt2::canary
