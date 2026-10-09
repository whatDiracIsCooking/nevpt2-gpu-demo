# add_device_library -- one kernel file (or a few) as a STATIC device
# library linked into the binary: WarpWraps' wwr_add_gpu_device_library, plus
# the two things every device library in this project needs on top of it.
#
#   add_device_library(
#     NAME    <target, dotted: nevpt2.<component>[.<what>].device>
#     SOURCES <file.cu ...>
#     [LINK_PRIVATE <lib ...>])   # wwr::device, or wwr::extension::parallel_for
#
# On top of wwr_add_gpu_device_library it:
#   - pins device code to C++20 on BOTH backends. Under CUDA WarpWraps already
#     sets CUDA_STANDARD 20; under HIP it compiles the .cu as a CXX source, which
#     would otherwise inherit the host's CMAKE_CXX_STANDARD 23. A target
#     property, not a -std flag, so CMake emits exactly one -std;
#   - joins the `nevpt2_device_libraries` umbrella, which is what
#     `devtools/cross-backend-check.sh --device-only` builds -- so a new kernel
#     library is covered by being declared, not by someone editing a list.
#
# The caller adds include directories itself (most want NEVPT2_SRC_ROOT, for a
# root-relative "rdm/..._bridge.h"); the kernel launches through the runtime
# API from a *_bridge.h the host unit includes after its imports, inside
# extern "C++" (docs/architecture.md, "Kernels are linked in, never loaded").
include_guard(GLOBAL)

include(nevpt2_internal_helpers)

if(NOT TARGET nevpt2_device_libraries)
  add_custom_target(nevpt2_device_libraries COMMENT "Building the kernel device libraries")
endif()

function(add_device_library)
  cmake_parse_arguments(ARG "" "NAME" "SOURCES;LINK_PRIVATE" ${ARGN})
  _nevpt2_require_args("add_device_library" ARG NAME SOURCES)

  # wwr_add_gpu_device_library is a macro (no scope of its own), so calling it
  # from this function resolves relative SOURCES -- and its HIP-side
  # set_source_files_properties(LANGUAGE CXX) -- in the caller's directory, as
  # it would at a direct call site.
  wwr_add_gpu_device_library(
    NAME ${ARG_NAME}
    SOURCES ${ARG_SOURCES}
    LINK_PRIVATE ${ARG_LINK_PRIVATE}
  )
  if(NOT NEVPT2_GPU_BACKEND STREQUAL "CUDA")
    set_target_properties(${ARG_NAME} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    # Under the host sanitizers (cmake/nevpt2_sanitizers.cmake) the device
    # pass stays uninstrumented, explicitly: gfx1200 cannot take device ASan.
    if(NEVPT2_ENABLE_ASAN OR NEVPT2_ENABLE_UBSAN)
      target_compile_options(${ARG_NAME} PRIVATE -fno-gpu-sanitize)
    endif()
  endif()
  add_dependencies(nevpt2_device_libraries ${ARG_NAME})
endfunction()
