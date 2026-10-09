# add_gtest_suite_tests -- register a GoogleTest binary with ctest as one entry
# per suite, plus a drift guard.
#
#   add_gtest_suite_tests(
#     TARGET   <gtest executable>
#     SUITES   <Suite ...>               # TEST() / TEST_F() suites
#     [TIMEOUT <seconds>]                # per entry, default 120 (x the multiplier)
#     [REQUIRES_GPU]                     # label every entry `gpu` too
#     [DEATH])                           # label every entry `death` too
#
# Every entry is labelled `unit`, so `ctest --preset unit` (`-L unit`) selects
# the whole gtest tier and nothing of the golden tier. Each suite becomes the
# ctest entry `<Suite>`, running `<target> --gtest_filter=<Suite>.*`.
#
# REQUIRES_GPU marks a binary that needs a live device -- it allocates device
# memory, launches a kernel or creates a library handle. Nothing in test/
# calls GTEST_SKIP, so on a box with no card such a suite ERRORS rather than
# skips; the label is what lets `-LE gpu` leave it out by name. A host-only
# suite leaves it off and runs card-free -- which proves logic, never an
# energy.
#
# Under the compute-sanitizer preset a REQUIRES_GPU entry also runs wrapped in
# NEVPT2_GPU_TEST_LAUNCHER -- src/'s CMAKE_TEST_LAUNCHER, which
# test/CMakeLists.txt copies under that name -- and a host-only one does not
# (compute-sanitizer fails a process that never calls the GPU runtime).
#
# DEATH marks EXPECT_DEATH suites (the abort tier: check, narrowTo). gtest
# forks per death assertion, so they are slower and noisier than the rest; the
# label lets a run pick them out or drop them. Name such a suite `...DeathTest`
# (gtest runs those first, before any thread exists).
#
# THE DRIFT GUARD, `<target>.SuiteListIsComplete`: the suite names here are
# written by hand, so a new suite left off the list would build, never run,
# and leave ctest green. The guard runs `<target> --gtest_list_tests` through
# nevpt2_check_gtest_suites.cmake and fails if the binary's suites and the
# union of every SUITES list given for TARGET differ, in either direction. It
# is registered once per target, deferred to the end of the directory, so a
# target split over several calls (plain suites in one, DEATH suites in
# another) is checked against the union. It is labelled `unit` but never `gpu`
# or `death`: listing tests constructs no fixture and touches no device.
#
# Ported from WarpWraps' wwr_add_gtest_suite_tests, NOT called: that one finds
# its script via ${PROJECT_SOURCE_DIR}/cmake/, which is this project's cmake/
# when WarpWraps is embedded. This one finds its script beside itself. Dropped:
# TYPED_SUITES / TYPES -- no typed suite here yet; bring them back with the
# first one.
include_guard(GLOBAL)

include(nevpt2_internal_helpers)

# Internal: the deferred drift guard for one target. Not called directly.
# `script` is resolved by the caller from CMAKE_CURRENT_FUNCTION_LIST_DIR,
# which a deferred call cannot rely on.
function(_nevpt2_register_gtest_suite_guard target script)
  get_property(_suites GLOBAL PROPERTY _nevpt2_gtest_suites_${target})
  add_test(NAME ${target}.SuiteListIsComplete
           COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:${target}> "-DEXPECTED=${_suites}"
                   -P "${script}")
  math(EXPR _timeout "60 * ${NEVPT2_TEST_TIMEOUT_MULTIPLIER}")
  set_tests_properties(${target}.SuiteListIsComplete PROPERTIES
    LABELS unit
    TIMEOUT ${_timeout})
endfunction()

function(add_gtest_suite_tests)
  cmake_parse_arguments(_GST "REQUIRES_GPU;DEATH" "TARGET;TIMEOUT" "SUITES" ${ARGN})
  _nevpt2_require_args("add_gtest_suite_tests" _GST TARGET SUITES)
  if(NOT _GST_TIMEOUT)
    set(_GST_TIMEOUT 120)
  endif()
  math(EXPR _timeout "${_GST_TIMEOUT} * ${NEVPT2_TEST_TIMEOUT_MULTIPLIER}")

  set(_labels unit)
  if(_GST_REQUIRES_GPU)
    list(APPEND _labels gpu)
  endif()
  if(_GST_DEATH)
    list(APPEND _labels death)
  endif()

  # A REQUIRES_GPU entry runs under NEVPT2_GPU_TEST_LAUNCHER (compute-sanitizer
  # in that preset, set by test/CMakeLists.txt; empty otherwise). Host-only
  # entries are never wrapped: compute-sanitizer fails a process that makes no
  # GPU runtime call.
  set(_launcher "")
  if(_GST_REQUIRES_GPU)
    set(_launcher ${NEVPT2_GPU_TEST_LAUNCHER})
  endif()

  foreach(_suite IN LISTS _GST_SUITES)
    add_test(NAME ${_suite}
             COMMAND ${_launcher} $<TARGET_FILE:${_GST_TARGET}> --gtest_filter=${_suite}.*)
    set_tests_properties(${_suite} PROPERTIES LABELS "${_labels}" TIMEOUT ${_timeout})
  endforeach()

  # The union every call for this target contributes; the guard reads it at
  # the end of the directory.
  set_property(GLOBAL APPEND PROPERTY _nevpt2_gtest_suites_${_GST_TARGET} ${_GST_SUITES})

  get_property(_scheduled GLOBAL PROPERTY _nevpt2_gtest_guard_${_GST_TARGET})
  if(NOT _scheduled)
    set_property(GLOBAL PROPERTY _nevpt2_gtest_guard_${_GST_TARGET} TRUE)
    set(_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/nevpt2_check_gtest_suites.cmake")
    # EVAL CODE bakes the arguments into the deferred call as literals: a plain
    # `DEFER CALL f("${_GST_TARGET}")` re-evaluates them when the call runs,
    # after this scope is gone, and `$<TARGET_FILE:>` then fails to parse.
    cmake_language(EVAL CODE
      "cmake_language(DEFER CALL _nevpt2_register_gtest_suite_guard \"${_GST_TARGET}\" \"${_script}\")")
  endif()
endfunction()
