// nevpt2.cublas_emul -- cuBLAS fixed-point-emulation DGEMM for the RDM-build
// digest (CUDA 13.0).
//
// The digests are ~80% of the compute floor and are genuine GEMMs, and the
// RTX 3080 has no fp64 tensor cores -- so CUDA 13.0's cuBLAS fixed-point
// emulation (a DGEMM decomposed into lower-precision tensor-core passes) is
// the lever. `max_mantissa_bits=53` is bit-identical to native fp64; lower is
// faster and less accurate.
//
// THE TRAP (it cost real time to find): the
// emulation knobs are scoped to the dedicated CUBLAS_COMPUTE_64F_EMULATED_
// FIXEDPOINT compute type requested via cublasGemmEx -- they do NOTHING to
// cublasDgemm, which always runs CUBLAS_COMPUTE_64F. And only the EAGER
// strategy actually engages emulation on this GPU (PERFORMANT's heuristic
// never picks it here). Both are set in cublas_emul.cpp; `digest_gemm` always
// calls cublasGemmEx with the emulated compute type.
//
// CUDA-ONLY. There is no ROCm counterpart to the emulated compute type, so
// cublas/CMakeLists.txt defines NEVPT2_HAVE_CUBLAS_EMUL on the CUDA backend
// alone. Without it cublas_emul.cpp supplies same-signature stubs: a
// requireCublasEmul() that returns an Unsupported Error (each demo's main.cppm
// calls it up front when --cublas is given and reports that Error like any
// other bad flag), and handle/GEMM stubs that are therefore never
// reached and abort through check if they are -- which keeps
// rdm/rdm_build.cpp's tile loop free of backend #ifs.
//
// The handle is opaque here (cublasHandle_t is a plain pointer), so this
// interface names no vendor type and importing it pulls in no cuBLAS header.
export module nevpt2.cublas_emul;

import std;
// wwrStream_t, which the handle is bound to, and DeviceResources, whose
// pool the probe's one int comes from. Not re-exported: every caller already
// has both.
import nevpt2.device_resources;

export namespace nevpt2 {

// The same fact at compile time, fixed when this interface is compiled (the
// define is PRIVATE; cmake/README.md). The demos ask requireCublasEmul()
// instead, which carries the message.
#if defined(NEVPT2_HAVE_CUBLAS_EMUL)
inline constexpr bool kHaveCublasEmul = true;
#else
inline constexpr bool kHaveCublasEmul = false;
#endif

// Success when this build has the emulated digest (CUDA); otherwise an
// Unsupported Error saying --cublas is CUDA-only. What each main() asks
// before accepting --cublas.
Status requireCublasEmul();

using EmulHandle = void*;  // a cublasHandle_t on CUDA

// Create a handle bound to the CURRENT context (cuBLAS attaches to the context
// current on the calling thread -- the runtime's primary context, which
// DeviceResources::create's wwrSetDevice made current before this is called),
// configured for
// fixed-point emulation at `max_mantissa_bits` (53 = bit-identical fp64), and
// bound to `stream` (cublasSetStream): every GEMM it runs is enqueued there.
EmulHandle make_emul_handle(int max_mantissa_bits, wwrStream_t stream);

// Row-major digest GEMM: C[M,N] = alpha * sum_K A[M,K] * B[N,K] + beta * C,
// where A, B are row-major with K minor (the produce/consume block layout),
// and C is row-major with N minor.
void digest_gemm(EmulHandle h, const double* dA, const double* dB, double* dC, int M, int N,
                 int K, double beta);

// Run one digest_gemm with the introspection pointer set, and return the
// mantissa bits cuBLAS actually used (53 = engaged bit-identical; -1 = it
// declined and ran native fp64). The handle must be bound to res.stream(): the
// probe's int is allocated from res's pool there, and the read-back is
// enqueued and synchronized there.
int probe_emul_bits(EmulHandle h, const DeviceResources& res, const double* dA,
                    const double* dB, double* dC, int M, int N, int K);

void destroy_emul_handle(EmulHandle h);

}  // namespace nevpt2
