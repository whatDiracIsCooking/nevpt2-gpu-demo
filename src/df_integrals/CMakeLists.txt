# nevpt2.df_integrals -- the density-fitted IntegralSource: external
# SC-NEVPT2 integral slabs built on the device from three-index B tensors, one
# DGEMM each. wwr::blas is WarpWraps' backend-neutral BLAS (cuBLAS on CUDA,
# hipBLAS on HIP), so unlike --cublas this path exists on both backends.
add_cxx_module_library(
  NAME nevpt2.df_integrals
  PRIMARY_INTERFACE df_integrals.cppm
  PARTITIONS df_integrals_permutation.cppm
  IMPLEMENTATION df_integrals.cpp
  LINK_PUBLIC nevpt2::energy
  # wwr::extension::blas: the wwrblasStatus_t specializations that let
  # nevpt2::gpuCheck check a BLAS status.
  LINK_PRIVATE wwr::blas wwr::extension::blas nevpt2::profile nevpt2::common
)
