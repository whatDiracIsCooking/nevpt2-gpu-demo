# GPU suites for nevpt2.common's device half: warp_reduce.cuh and
# block_reduce.cuh, which have no module face, run on the card.
#
# The first .cu files under test/. Each suite's kernels are a device library
# of their own (add_device_library, so they join nevpt2_device_libraries and
# are compiled C++20 on both backends), launched through a plain-type
# *_bridge.h -- nvcc's host pass is libstdc++, the test binaries libc++. Like
# every device library here they are built with separable compilation OFF and
# linked into an ordinary CXX executable, which therefore needs no CUDA
# device-link step (nor does add_gtest_executable grow one).
#
# The test binaries include the bridges, so they need the src/ root (for
# "wwr/wwr.h", "common/host_device.h") and wwr::backend (runtime.h's include
# path and the WWR_GPU_BACKEND_* define selected_backend.h reads in a
# host-only pass), as nevpt2.rdm_build does.
#
# REQUIRES_GPU: every case launches a kernel; the drift guards are `unit`
# only and run card-free.
include(add_device_library)

# hip found HERE too, as src/CMakeLists.txt does for itself: an IMPORTED target
# is visible only in the directory that found it and below, test/ is src/'s
# sibling, and wwr_add_gpu_device_library reads hip::amdhip64 and links
# hip::device. A cached no-op beyond the target import. (CUDA needs no
# counterpart: the CUDA::* targets already resolve here.)
if(NOT NEVPT2_GPU_BACKEND STREQUAL "CUDA")
  find_package(hip CONFIG REQUIRED)
endif()

# --- warp_reduce --------------------------------------------------------------

add_device_library(
  NAME nevpt2.test.warp_reduce.device
  SOURCES warp_reduce_kernels.cu
  LINK_PRIVATE wwr::device nevpt2::common::device
)

add_gtest_executable(
  NAME warp_reduce_tests
  SOURCES warp_reduce_tests.cpp
  LINK nevpt2::wwr nevpt2::test::shared_resources nevpt2.test.warp_reduce.device wwr::backend)
target_include_directories(warp_reduce_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")

add_gtest_suite_tests(
  TARGET warp_reduce_tests
  SUITES WarpReduceTests
  REQUIRES_GPU)

# --- block_reduce -------------------------------------------------------------

add_device_library(
  NAME nevpt2.test.block_reduce.device
  SOURCES block_reduce_kernels.cu
  LINK_PRIVATE wwr::device nevpt2::common::device
)

add_gtest_executable(
  NAME block_reduce_tests
  SOURCES block_reduce_tests.cpp
  LINK nevpt2::wwr nevpt2::test::shared_resources nevpt2.test.block_reduce.device wwr::backend)
target_include_directories(block_reduce_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")

add_gtest_suite_tests(
  TARGET block_reduce_tests
  SUITES BlockReduceTreeTests BlockReduceWarpsTests
  REQUIRES_GPU)
