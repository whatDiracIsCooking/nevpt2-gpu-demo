// nevpt2.cublas_emul:raw_handle -- the EmulHandle -> cublasHandle_t cast the
// real (CUDA) bodies call cuBLAS through. An internal partition: nothing here
// is exported, so it is reachable from cublas_emul.cpp (`import :raw_handle;`)
// and from no importer of nevpt2.cublas_emul. CUDA builds only: the HIP build
// lists nevpt2.cublas_emul:unavailable instead (cublas/CMakeLists.txt).
module;

#include <cublas_v2.h>

module nevpt2.cublas_emul:raw_handle;

import std;
import nevpt2.cublas_emul;

namespace nevpt2 {

// cuBLAS statuses go through gpuCheck like every other: cublasStatus_t IS
// wwrblasStatus_t on CUDA (an alias), so wwr.extension.blas's
// specializations cover it.
cublasHandle_t raw(EmulHandle h) { return static_cast<cublasHandle_t>(h); }

}  // namespace nevpt2
