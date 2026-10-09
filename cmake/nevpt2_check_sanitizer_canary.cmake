# Verdict for add_sanitizer_canary(): run the canary and pass only when
# it exited non-zero AND its output carries the sanitizer report EXPECT names.
#
# Run via (LAUNCHER may be empty; '|' separates list items):
#   cmake -DLAUNCHER=<a|b|c> -DEXE=<binary> -DARGS=<x|y> -DEXPECT=<regex> \
#     -P nevpt2_check_sanitizer_canary.cmake
#
# Run by hand against an UNSANITIZED binary (e.g. build/src/nevpt2_sanitizer_canary
# with no LAUNCHER) it must FAIL: that is the "goes red with its sanitizer off"
# check docs/testing.md ("Sanitizers") records.

if(NOT EXE OR NOT EXPECT)
  message(FATAL_ERROR "EXE and EXPECT must be set")
endif()
string(REPLACE "|" ";" _launcher "${LAUNCHER}")
string(REPLACE "|" ";" _args "${ARGS}")

execute_process(
  COMMAND ${_launcher} "${EXE}" ${_args}
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE _out
  RESULT_VARIABLE _rc)
message("${_out}")

if(_rc EQUAL 0)
  message(FATAL_ERROR
    "Canary exited 0: the sanitizer did not fail the run, so the check it "
    "guards is off or is not failing runs (docs/testing.md, \"Sanitizers\").")
endif()
if(NOT _out MATCHES "${EXPECT}")
  message(FATAL_ERROR
    "Canary exited ${_rc} but without the expected report '${EXPECT}': it "
    "failed for some other reason, which proves nothing about the check.")
endif()
message(STATUS "Canary tripped as expected (exit ${_rc}): ${EXPECT}")
