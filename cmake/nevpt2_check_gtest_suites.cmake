# Script mode, for add_gtest_suite_tests' drift guard: fail if the SUITES a
# CMakeLists names have drifted from the suites the binary actually registers.
#
#   cmake -DEXE=<gtest binary> -DEXPECTED=<a;b;c> -P nevpt2_check_gtest_suites.cmake
#
# The suite names live in CMake, by hand. Without this check a test file whose
# suite was never listed builds, never runs, and leaves ctest green -- a suite
# that passes vacuously. So the list is verified rather than trusted, in both
# directions: a registered suite missing from the list, and a listed name the
# binary no longer registers. Ported from WarpWraps' wwr_check_gtest_suites.cmake.

if(NOT EXE)
  message(FATAL_ERROR "nevpt2_check_gtest_suites: EXE not set")
endif()

execute_process(
  COMMAND "${EXE}" --gtest_list_tests
  OUTPUT_VARIABLE _listing
  ERROR_VARIABLE _stderr
  RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "'${EXE} --gtest_list_tests' failed (${_rc}):\n${_stderr}")
endif()

# A suite header starts in column 0 and ends in '.'; a typed or parameterized
# suite's '/N' suffix is collapsed, so `Foo/0.` and `Foo/1.` both report `Foo`.
set(_actual "")
string(REPLACE "\n" ";" _lines "${_listing}")
foreach(_line IN LISTS _lines)
  if(_line MATCHES "^([A-Za-z_][A-Za-z0-9_]*)(/[0-9]+)?\\.")
    list(APPEND _actual "${CMAKE_MATCH_1}")
  endif()
endforeach()
list(REMOVE_DUPLICATES _actual)
list(SORT _actual)

set(_expected ${EXPECTED})
list(REMOVE_DUPLICATES _expected)
list(SORT _expected)

if(NOT _actual STREQUAL _expected)
  set(_missing ${_actual})
  if(_expected)
    list(REMOVE_ITEM _missing ${_expected})
  endif()
  set(_stale ${_expected})
  if(_actual)
    list(REMOVE_ITEM _stale ${_actual})
  endif()
  message(FATAL_ERROR
    "GoogleTest suite list is out of date for ${EXE}.\n"
    "  registered but NOT listed in CMake (these tests never run): ${_missing}\n"
    "  listed in CMake but not registered (stale entries): ${_stale}\n"
    "Update the SUITES argument of add_gtest_suite_tests() in that target's CMakeLists.txt.")
endif()
message(STATUS "suite list complete: ${_actual}")
