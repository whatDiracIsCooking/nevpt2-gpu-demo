// The bodies behind cublas_emul.cppm -- real on CUDA (NEVPT2_HAVE_CUBLAS_EMUL),
// refusing stubs otherwise.
//
// cuBLAS arrives by #include in the global module fragment, not through
// WarpWraps' wwr.cuda.cublas_v2 module: that module does not export the
// fixed-point emulation surface this file exists for
// (cublasSetFixedPointEmulation*, CUDA_EMULATION_MANTISSA_CONTROL_FIXED, ...).
module;

#if defined(NEVPT2_HAVE_CUBLAS_EMUL)
#include <cublas_v2.h>
#endif

module nevpt2.cublas_emul;

import std;
import nevpt2.device_resources;
// wwrblasStatus_t's error_type specializations (cublasStatus_t on CUDA), so
// gpuCheck takes a cuBLAS status. Imported on both backends to keep the
// module preamble free of #if; only the CUDA half below uses it.
import wwr.extension.blas;
// The one partition per backend (cublas/CMakeLists.txt lists only that one):
// the cuBLAS handle cast on CUDA, the stubs' abort otherwise.
#if defined(NEVPT2_HAVE_CUBLAS_EMUL)
import :raw_handle;
#else
import :unavailable;
#endif

#if defined(NEVPT2_HAVE_CUBLAS_EMUL)

namespace nevpt2 {

Status requireCublasEmul() { return {}; }

EmulHandle make_emul_handle(int max_mantissa_bits, wwrStream_t stream) {
  cublasHandle_t h;
  gpuCheck(cublasCreate(&h));
  // wwrStream_t IS cudaStream_t on CUDA (WarpWraps binds the names), so the
  // demo's stream goes straight in: DeviceResources' non-blocking one. A
  // handle left without a stream runs on the legacy default one -- which
  // would race everything else, silently.
  gpuCheck(cublasSetStream(h, stream));
  gpuCheck(cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_EAGER));
  gpuCheck(cublasSetFixedPointEmulationMantissaControl(
      h, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
  gpuCheck(cublasSetFixedPointEmulationMaxMantissaBitCount(h, max_mantissa_bits));
  return h;
}

// cuBLAS is column-major, so we compute C^T = (N,M) directly: op(A_cublas=B)=
// OP_T, op(B_cublas=A)=OP_N, dims (m=N, n=M, k=K), leading dims K/K/N.
void digest_gemm(EmulHandle h, const double* dA, const double* dB, double* dC, int M, int N,
                 int K, double beta) {
  const double alpha = 1.0;
  gpuCheck(cublasGemmEx(raw(h), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB,
                        CUDA_R_64F, K, dA, CUDA_R_64F, K, &beta, dC, CUDA_R_64F, N,
                        CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT));
}

// The pointer must be un-registered before its device buffer is freed (else
// the handle writes through freed memory on every later emulated call -- a
// use-after-free seen as a later segfault).
int probe_emul_bits(EmulHandle h, const DeviceResources& res, const double* dA,
                    const double* dB, double* dC, int M, int N, int K) {
  const wwrStream_t stream = res.stream();
  const int sentinel = -1;
  // Freed (wwrFreeAsync on `stream`) when it goes out of scope at the return,
  // after the pointer is un-registered below.
  DeviceBuffer<int> d_bits(1, res.shared_from_this());
  gpuCheck(wwrMemcpyAsync(d_bits.data(), &sentinel, sizeof(int), wwrMemcpyHostToDevice, stream));
  gpuCheck(cublasSetFixedPointEmulationMantissaBitCountPointer(raw(h), d_bits.data()));
  digest_gemm(h, dA, dB, dC, M, N, K, /*beta=*/0.0);
  int bits = -1;
  gpuCheck(wwrMemcpyAsync(&bits, d_bits.data(), sizeof(int), wwrMemcpyDeviceToHost, stream));
  // The GEMM wrote d_bits on `stream` (the handle's), and the host reads
  // `bits` next.
  gpuCheck(wwrStreamSynchronize(stream));
  gpuCheck(cublasSetFixedPointEmulationMantissaBitCountPointer(raw(h), nullptr));
  return bits;
}

void destroy_emul_handle(EmulHandle h) {
  if (h) cublasDestroy(raw(h));
}

}  // namespace nevpt2

#else  // !NEVPT2_HAVE_CUBLAS_EMUL

namespace nevpt2 {

// A HIP binary given --cublas: the user's request, which this build cannot
// serve -- reported by main() like any other bad flag.
Status requireCublasEmul() {
  return err_unsupported(std::format(
      "the --cublas digest is CUDA-only (CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT has no {} "
      "counterpart); this build has no emulated digest",
      kGpuBackendName));
}

// Each stub aborts through no_cublas_emul (nevpt2.cublas_emul:unavailable).
EmulHandle make_emul_handle(int, wwrStream_t) { no_cublas_emul(); }
void digest_gemm(EmulHandle, const double*, const double*, double*, int, int, int, double) {
  no_cublas_emul();
}
int probe_emul_bits(EmulHandle, const DeviceResources&, const double*, const double*, double*,
                    int, int, int) {
  no_cublas_emul();
}
void destroy_emul_handle(EmulHandle) {}

}  // namespace nevpt2

#endif  // NEVPT2_HAVE_CUBLAS_EMUL
