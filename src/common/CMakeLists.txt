# The shared building blocks, in two targets:
#
#   nevpt2.common          (nevpt2::common)          the host module face:
#                          idivup / align_up and int64_t, re-exported from
#                          align_up.h and int64.h, plus the checked narrowTo<To>
#   nevpt2.common.device   (nevpt2::common::device)  INTERFACE: the src/ include
#                          root kernel files reach these headers through
#                          ("common/block_reduce.cuh"), plus wwr::device for the
#                          WarpWraps headers they include (runtime.h,
#                          device_guard.h, cooperative_groups.h)
#
# Headers: host_device.h, align_up.h, int64.h (host and device); block_params.h,
# device_index.h, device_functor.h, warp_reduce.cuh, block_reduce.cuh (device
# only -- see common.cppm).
add_cxx_module_library(
  NAME nevpt2.common
  PRIMARY_INTERFACE common.cppm
  # PUBLIC: narrowTo, a template importers instantiate, fails through
  # nevpt2.error_handling's check(). That module imports only std.
  LINK_PUBLIC nevpt2::error_handling
)

add_library(nevpt2.common.device INTERFACE)
add_library(nevpt2::common::device ALIAS nevpt2.common.device)
target_include_directories(nevpt2.common.device INTERFACE "${NEVPT2_SRC_ROOT}")
target_link_libraries(nevpt2.common.device INTERFACE wwr::device)
