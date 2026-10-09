# nevpt2_sanitizer_canary -- the deliberately buggy runs the sanitizer tier's
# canaries are made of (see main.cppm for the modes). A tiny binary of
# its own rather than hidden demo flags, so neither demo carries planted bugs.
#
# Built in EVERY configuration, not just the sanitizer presets: that is what
# lets a canary be shown to go red with its sanitizer off (run it from the
# default build), and what keeps devtools/cross-backend-check.sh compiling it
# for the other backend. It is registered with ctest only under its own
# sanitizer's preset (the root CMakeLists.txt).
add_device_library(
  NAME nevpt2.sanitizer_canary.device
  SOURCES canary_kernels.cu
  # nevpt2::common::device: the src/ root, for canary_bridge.h's "wwr/wwr.h".
  LINK_PRIVATE wwr::device nevpt2::common::device
)
target_include_directories(nevpt2.sanitizer_canary.device PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(nevpt2_sanitizer_canary)  # main.cppm: see apps/integral_direct/CMakeLists.txt
target_sources(nevpt2_sanitizer_canary PRIVATE FILE_SET CXX_MODULES FILES main.cppm canaries.cppm)
set_target_properties(nevpt2_sanitizer_canary PROPERTIES
  CXX_MODULE_STD ON
  RUNTIME_OUTPUT_DIRECTORY "${NEVPT2_SRC_BINARY_DIR}")
target_compile_features(nevpt2_sanitizer_canary PRIVATE cxx_std_23)
target_compile_options(nevpt2_sanitizer_canary PRIVATE -Wall)
target_include_directories(nevpt2_sanitizer_canary PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# wwr::backend: selected_backend.h, which the bridge's runtime.h reads on the
# host side (as for rdm_build.cpp).
target_link_libraries(nevpt2_sanitizer_canary PRIVATE
  nevpt2::wwr nevpt2::device_resources nevpt2.sanitizer_canary.device wwr::backend)
