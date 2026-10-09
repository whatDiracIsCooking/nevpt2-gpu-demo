# test/ -- the GoogleTest unit tier. One directory per component,
# mirroring src/: test/<x>/ tests nevpt2.<x>. Added from the root
# CMakeLists.txt only, under NEVPT2_BUILD_TESTING, so the standalone
# `cmake -S src` build stays gtest-free.
#
# What a suite here proves is LOGIC -- an error kind's name, a propagation
# path, an index helper -- never an energy. The numerical claims stay with the
# golden tier (the demo entries in the root CMakeLists.txt, all `gpu`). Every
# entry here is labelled `unit`; one that needs a card is labelled `gpu` too
# (add_gtest_suite_tests REQUIRES_GPU), so a host-only suite runs card-free.
# A GPU suite takes its device from nevpt2.test.shared_resources (test/utils/):
# one DeviceResources per process, as each demo's main() makes one.
#
# Each suite is registered with add_gtest_suite_tests (one ctest entry per
# suite + the <target>.SuiteListIsComplete drift guard); see cmake/README.md,
# "Tests".
include(add_gtest_executable)
include(add_gtest_suite_tests)

# src/ sets its compile and link options as DIRECTORY properties (the libc++
# link flag in src/CMakeLists.txt; the host ASan/UBSan flags in
# nevpt2_sanitizers.cmake), and test/ is src/'s sibling, not its child, so it
# would inherit none of them. Copy them, so a test binary is compiled and
# linked exactly as the module libraries it imports were -- an ASan-built
# nevpt2.common linked into an uninstrumented test would not link.
get_directory_property(_nevpt2_src_compile_options DIRECTORY "${PROJECT_SOURCE_DIR}/src"
                       COMPILE_OPTIONS)
get_directory_property(_nevpt2_src_link_options DIRECTORY "${PROJECT_SOURCE_DIR}/src"
                       LINK_OPTIONS)
set_directory_properties(PROPERTIES
  COMPILE_OPTIONS "${_nevpt2_src_compile_options}"
  LINK_OPTIONS "${_nevpt2_src_link_options}")
unset(_nevpt2_src_compile_options)
unset(_nevpt2_src_link_options)

# compute-sanitizer's launcher, likewise from src/: nevpt2_sanitizers.cmake sets
# CMAKE_TEST_LAUNCHER in src/'s scope only, so without this no gtest entry
# would run under it and the compute-sanitizer build's unit entries would pass
# unsanitized. Copied under ANOTHER name, not as CMAKE_TEST_LAUNCHER: that
# would wrap every binary here, and compute-sanitizer FAILS one that never
# calls the GPU runtime ("Target application terminated before first
# instrumented API call"), which is every host-only suite. So
# add_gtest_suite_tests prepends it to the REQUIRES_GPU entries only. Empty,
# wrapping nothing, unless NEVPT2_COMPUTE_SANITIZER is ON.
get_directory_property(NEVPT2_GPU_TEST_LAUNCHER DIRECTORY "${PROJECT_SOURCE_DIR}/src"
                       DEFINITION CMAKE_TEST_LAUNCHER)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)  # must match libc++'s std module, as in src/

# Test-support libraries first, so the suites below can link them.
add_subdirectory(utils)

add_subdirectory(error_handling)
add_subdirectory(link_tables)
add_subdirectory(rdm_plan)
add_subdirectory(energy_finish)
add_subdirectory(rdm)
add_subdirectory(tensor)
add_subdirectory(device_tensor)
add_subdirectory(device_resources)
add_subdirectory(einsum)
add_subdirectory(ozaki)
add_subdirectory(golden)
add_subdirectory(df_integrals)
add_subdirectory(common)
add_subdirectory(cli)
