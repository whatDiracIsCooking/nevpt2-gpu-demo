# nevpt2_coverage.cmake -- clang source-based code coverage of HOST code.
# Default OFF, so the normal presets build exactly as before; the `coverage`
# and `hip-coverage` presets turn it on, each in its own build-* directory,
# and devtools/coverage.sh turns their profiles into a report.
#
# Included once by src/CMakeLists.txt, next to nevpt2_sanitizers.cmake and for
# the same reason: BEFORE nevpt2_warpwraps.cmake, because add_compile_options
# is a directory property, and test/CMakeLists.txt copies src/'s directory
# options so the gtest binaries are built and linked the same way.
#
# What is instrumented: every CXX unit EXCEPT those of a device library
# (cmake/add_device_library.cmake marks them NEVPT2_DEVICE_LIBRARY). That is
# our module libraries, the demos, the gtest binaries and the WarpWraps
# targets we link; devtools/coverage.sh drops deps/, test/ and _deps/ from the
# report, so what it counts is src/ and apps/.
#
# What is NOT, on purpose -- the `.cu` files, on BOTH backends, kernels and
# launchers alike. Device code has no llvm-cov support on either vendor, and
# instrumenting the host half only would be backend-asymmetric: under CUDA
# nvcc's host compiler is not our clang, and under HIP a `.cu` is a `-x hip`
# CXX unit whose device pass would get the flags too. So the report is "host
# C++ module code", the same on both cards, and a kernel path is measured by
# the golden tier's numbers, never by this (docs/testing.md, "Coverage").
#
# Profiles land wherever LLVM_PROFILE_FILE says; the coverage test presets
# point it into the build dir (one file per process). A process that ends in
# std::abort() -- our abort tier, and every EXPECT_DEATH child -- writes no
# profile, so a line reached only on the way to an abort counts as unexecuted.
include_guard(GLOBAL)

option(NEVPT2_ENABLE_COVERAGE
       "clang source-based coverage on host CXX units (device libraries excluded)" OFF)

if(NEVPT2_ENABLE_COVERAGE)
  if(NEVPT2_ENABLE_ASAN OR NEVPT2_ENABLE_UBSAN OR NEVPT2_COMPUTE_SANITIZER)
    message(FATAL_ERROR
      "NEVPT2_ENABLE_COVERAGE does not combine with the sanitizer tier: "
      "use the `coverage` preset, which turns every sanitizer off.")
  endif()
  message(STATUS "nevpt2: source-based coverage on host CXX units (device libraries excluded)")
  set(_nevpt2_cov_host
      "$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<BOOL:$<TARGET_PROPERTY:NEVPT2_DEVICE_LIBRARY>>>>")
  add_compile_options("$<${_nevpt2_cov_host}:-fprofile-instr-generate>"
                      "$<${_nevpt2_cov_host}:-fcoverage-mapping>")
  unset(_nevpt2_cov_host)
  # Every binary links an instrumented library, so every link needs the
  # profile runtime -- device libraries are STATIC, so this reaches no
  # device link step of theirs.
  add_link_options(-fprofile-instr-generate)
endif()
