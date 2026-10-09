# The --cublas digest: CUDA ONLY, on purpose. CUDA 13.0's
# CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT has no ROCm equivalent.
#
# nevpt2.cublas_emul is built on BOTH backends -- real under
# NEVPT2_HAVE_CUBLAS_EMUL (CUDA), refusing stubs otherwise -- so
# rdm/rdm_build.cpp and both demos' main.cppm import it unconditionally and the
# tile loop carries no backend #if. The
# define is PRIVATE: importers get requireCublasEmul() (and
# kHaveCublasEmul, the same fact as a constant, which nothing reads yet) from
# the compiled module, not the macro.
if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
  add_cxx_module_library(
    NAME nevpt2.cublas_emul
    PRIMARY_INTERFACE cublas_emul.cppm
    # nevpt2.cublas_emul:raw_handle -- the EmulHandle -> cublasHandle_t cast.
    PARTITIONS cublas_emul_raw_handle.cppm
    IMPLEMENTATION cublas_emul.cpp
    # nevpt2::device_resources PUBLIC: the interface names wwrStream_t and
    # DeviceResources (the probe's allocation).
    # wwr::extension::blas: gpuCheck on a cublasStatus_t (= wwrblasStatus_t).
    LINK_PUBLIC nevpt2::device_resources
    LINK_PRIVATE CUDA::cublas wwr::extension::blas
    DEFINES_PRIVATE NEVPT2_HAVE_CUBLAS_EMUL
  )
else()
  add_cxx_module_library(
    NAME nevpt2.cublas_emul
    PRIMARY_INTERFACE cublas_emul.cppm
    # nevpt2.cublas_emul:unavailable -- the abort behind every stub.
    PARTITIONS cublas_emul_unavailable.cppm
    IMPLEMENTATION cublas_emul.cpp
    LINK_PUBLIC nevpt2::device_resources
    LINK_PRIVATE wwr::extension::blas
  )
endif()
