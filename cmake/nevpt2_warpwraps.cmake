# nevpt2_warpwraps.cmake -- embed deps/WarpWraps with add_subdirectory, so the
# host modules can `import wwr.runtime_api;` and link `wwr::runtime_api`.
#
# Included once, after project(), by src/CMakeLists.txt. (The device libraries
# reach WarpWraps' runtime.h through the wwr::device target, not a path.)
#
# What embedding costs, and how each cost is contained:
#   - WarpWraps' CMake finds its whole vendor stack as REQUIRED (CCCL on CUDA;
#     hipblas, hipsolver, rocthrust, ... on HIP). Nothing to contain -- those
#     packages ship with the toolkit / the ROCm image.
#   - It adds its example/ and test/ directories unconditionally. EXCLUDE_FROM_ALL
#     keeps them out of the default build (only what our targets link gets
#     compiled), and WWR_COMPILE_TIME_ONLY skips its GoogleTest fetch, so a
#     configure needs no network. Checked: `ctest -N` in a top-level build
#     lists only this project's nevpt2_* entries -- none of WarpWraps' tests
#     register into ours.
include_guard(GLOBAL)

set(_nevpt2_wwr_root "${CMAKE_CURRENT_LIST_DIR}/../deps/WarpWraps")
cmake_path(NORMAL_PATH _nevpt2_wwr_root)

# A fresh clone without `git submodule update --init` has an empty
# deps/WarpWraps, which would otherwise surface as a confusing add_subdirectory
# error naming no CMakeLists.txt.
if(NOT EXISTS "${_nevpt2_wwr_root}/CMakeLists.txt")
  message(FATAL_ERROR
    "nevpt2: WarpWraps not found at ${_nevpt2_wwr_root}. It is a git submodule "
    "-- run `git submodule update --init --recursive`.")
endif()

# One knob: the backend is ours, forwarded. WarpWraps refuses to default it.
set(WWR_GPU_BACKEND "${NEVPT2_GPU_BACKEND}" CACHE STRING "" FORCE)
set(WWR_COMPILE_TIME_ONLY ON CACHE BOOL "nevpt2 embeds WarpWraps: no GoogleTest fetch" FORCE)
set(WWR_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory("${_nevpt2_wwr_root}" "${CMAKE_BINARY_DIR}/deps/WarpWraps" EXCLUDE_FROM_ALL)
unset(_nevpt2_wwr_root)
