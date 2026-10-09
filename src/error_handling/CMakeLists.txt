# nevpt2.error_handling -- the vendor-free error model: the value tier (Error,
# Result<T>, Status, err_*, report) and the abort tier (check), plus
# error_macros.h (NEVPT2_TRY). Imports only std and links nothing, so any
# module -- nevpt2.common included -- can import it without the GPU runtime.
# gpuCheck, which judges a WarpWraps status, lives in nevpt2.wwr.
add_cxx_module_library(
  NAME nevpt2.error_handling
  PRIMARY_INTERFACE error_handling.cppm
)
# The src/ root, so a consumer #includes "error_handling/error_macros.h" from
# its global module fragment (a macro cannot ride an import).
target_include_directories(nevpt2.error_handling PUBLIC "${NEVPT2_SRC_ROOT}")
