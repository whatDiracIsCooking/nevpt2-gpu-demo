# add_sanitizer_canary -- register a deliberately buggy run as a ctest
# entry that passes only when the sanitizer catches the bug. A canary that goes
# red means the check it guards went dark (docs/testing.md, "Sanitizers").
#
#   add_sanitizer_canary(
#     NAME     <ctest name>
#     TARGET   <executable target>
#     ARGS     <arg ...>                 # selects the bug to run
#     EXPECT   <regex>                   # the sanitizer's report line
#     [REQUIRES_GPU])                    # label `gpu` too
#
# Not WILL_FAIL: that credits any non-zero exit, so a crash for the wrong
# reason passes, and a report printed with exit 0 (the tool not failing the
# run) is invisible. nevpt2_check_sanitizer_canary.cmake demands both a
# non-zero exit AND the EXPECT report. Every entry is labelled
# `sanitizer_canary`; the caller registers it only under its own preset.
#
# The launcher is read off TARGET's own TEST_LAUNCHER property -- the very
# property ctest wraps the demo entries with (cmake/nevpt2_sanitizers.cmake
# sets CMAKE_TEST_LAUNCHER, which initializes it). So a device canary proves
# the demo entries really run under compute-sanitizer, not merely that the
# tool works when invoked by hand. (A test whose COMMAND is cmake, as here,
# does not get TEST_LAUNCHER applied by ctest; the script runs it instead.)
include_guard(GLOBAL)

include(nevpt2_internal_helpers)

function(add_sanitizer_canary)
  cmake_parse_arguments(_CAN "REQUIRES_GPU" "NAME;TARGET;EXPECT" "ARGS" ${ARGN})
  _nevpt2_require_args("add_sanitizer_canary" _CAN NAME TARGET EXPECT)

  # A ';' would split the -D argument; the script turns '|' back into ';'.
  string(JOIN "|" _args ${_CAN_ARGS})
  set(_launcher "$<JOIN:$<TARGET_PROPERTY:${_CAN_TARGET},TEST_LAUNCHER>,|>")

  add_test(NAME ${_CAN_NAME}
           COMMAND ${CMAKE_COMMAND} "-DLAUNCHER=${_launcher}"
                   -DEXE=$<TARGET_FILE:${_CAN_TARGET}> "-DARGS=${_args}"
                   "-DEXPECT=${_CAN_EXPECT}" -P
                   "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/nevpt2_check_sanitizer_canary.cmake")

  set(_labels sanitizer_canary)
  if(_CAN_REQUIRES_GPU)
    list(APPEND _labels gpu)
  endif()
  math(EXPR _timeout "60 * ${NEVPT2_TEST_TIMEOUT_MULTIPLIER}")
  set_tests_properties(${_CAN_NAME} PROPERTIES LABELS "${_labels}" TIMEOUT ${_timeout})
endfunction()
