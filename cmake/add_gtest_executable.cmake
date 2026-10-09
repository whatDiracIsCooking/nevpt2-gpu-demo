# add_gtest_executable -- one GoogleTest binary under test/.
#
#   add_gtest_executable(
#     NAME    <target>                  # e.g. error_handling_tests
#     SOURCES <file.cpp ...>            # the binary's own test sources; no main()
#     [LINK   <lib ...>])               # the module libraries those sources import
#
# Always: GTest::gtest_main (it supplies main()), cxx_std_23, CXX_MODULE_STD ON
# (every test TU does `import std;`), CXX_SCAN_FOR_MODULES ON, -Wall. A test TU
# is a plain translation unit, not a module unit: `#include <gtest/gtest.h>`
# FIRST, then `import std;` and `import nevpt2.<x>;` (cmake/README.md, "Tests").
#
# Ported from WarpWraps' wwr_add_gtest_executable, minus its CUDA
# device-linking switch. Not needed even for a suite with kernels
# (test/common/): those are an add_device_library built with
# separable compilation OFF, so the CXX binary linking it has no device-link
# step.
include_guard(GLOBAL)

include(nevpt2_internal_helpers)

function(add_gtest_executable)
  cmake_parse_arguments(_GTE "" "NAME" "SOURCES;LINK" ${ARGN})
  _nevpt2_require_args("add_gtest_executable" _GTE NAME SOURCES)

  add_executable(${_GTE_NAME} ${_GTE_SOURCES})
  target_link_libraries(${_GTE_NAME} PRIVATE GTest::gtest_main ${_GTE_LINK})
  target_compile_features(${_GTE_NAME} PRIVATE cxx_std_23)
  target_compile_options(${_GTE_NAME} PRIVATE -Wall)
  set_target_properties(${_GTE_NAME} PROPERTIES CXX_MODULE_STD ON CXX_SCAN_FOR_MODULES ON)
endfunction()
