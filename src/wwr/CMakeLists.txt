# nevpt2.wwr -- the one door host code reaches the GPU through: WarpWraps'
# wwr*-prefixed runtime, BLAS and solver names, `using`'d into namespace nevpt2
# so they are written without `wwr::`, with wwr.runtime_api and
# nevpt2.error_handling (Error / Result / Status, check) re-exported, plus
# gpuCheck (and its Abort policy alias) and kGpuBackendName.
# It absorbed nevpt2.gpu, which did only that re-exporting.
#
#   wwr.h     the kernel-file half: wwrStream_t always, wwrError_t /
#             wwrSuccess / wwrGetLastError in a device pass (runtime.h's
#             gate). Reached root-relative ("wwr/wwr.h") by every *_bridge.h.
#   wwr.cppm  the host module face: re-exports wwr.h and adds the host-only
#             runtime, BLAS and solver names, and WarpWraps' RAII wrappers
#             with nevpt2::Abort in every slot (the stream / pool / BLAS /
#             solver handle wrappers, the device-buffer suite).
#
# Its own component rather than part of nevpt2.common, so a unit needing only
# idivup / narrowTo does not pull in the GPU runtime.
add_cxx_module_library(
  NAME nevpt2.wwr
  PRIMARY_INTERFACE wwr.cppm
  # PUBLIC: the interface's exported usings name entities of all three, and it
  # re-exports wwr.runtime_api and nevpt2.error_handling. wwr.extension.*:
  # gpuCheck is a template callers instantiate over WarpWraps' error layer and
  # the blas / solver status traits; runtime / blas / solver also carry the
  # RAII wrappers it instantiates, and memory_buffer is re-exported
  # (DeviceBuffer<T> is its DeviceBufferWrapper).
  LINK_PUBLIC wwr::runtime_api wwr::blas wwr::solver nevpt2::error_handling
              wwr::extension::common wwr::extension::runtime wwr::extension::blas
              wwr::extension::solver wwr::extension::memory_buffer
  # wwr::backend for the WWR_GPU_BACKEND_* define runtime.h (included by wwr.h,
  # in the GMF) selects on in a host compile, and for selected_backend.h.
  LINK_PRIVATE wwr::backend
)
# The src/ root, so a host unit including a *_bridge.h resolves "wwr/wwr.h".
target_include_directories(nevpt2.wwr PUBLIC "${NEVPT2_SRC_ROOT}")
