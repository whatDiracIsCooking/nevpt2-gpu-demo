# add_cxx_module_library -- a STATIC C++23 named-module library.
#
#   add_cxx_module_library(
#     NAME              <target, dotted: nevpt2.<component>>
#     PRIMARY_INTERFACE <file.cppm>               # export module nevpt2.<component>;
#     [PARTITIONS       <file.cppm ...>]          # module nevpt2.<component>:<part>;
#     [IMPLEMENTATION   <file.cpp ...>]           # module nevpt2.<component>;
#     [LINK_PUBLIC      <lib ...>]                # what the exported surface names
#     [LINK_PRIVATE     <lib ...>]                # what only the bodies need
#     [DEFINES_PRIVATE  <def ...>])
#
# PARTITIONS are module partition units -- in practice internal ones
# (`module nevpt2.<component>:<part>;`, no `export`), which share declarations
# among the component's implementation units without exporting them. A
# partition provides a module unit that other units import, so it goes in the
# CXX_MODULES file set beside the primary interface, internal or not.
#
# Always: STATIC, the interface in a CXX_MODULES file set, cxx_std_23 PUBLIC,
# CXX_MODULE_STD ON (every module here does `import std;`), and the `::` alias
# (`nevpt2.einsum` -> `nevpt2::einsum`).
#
# No IMPORT_STD switch (always on), no INCLUDE_CUDA_TOOLKIT (the toolkit
# headers arrive with the WarpWraps / CUDA:: targets that need them) and no
# install-interface include dirs (nothing is installed). Add a keyword when a
# real call site needs it.
include_guard(GLOBAL)
include(nevpt2_internal_helpers)

function(add_cxx_module_library)
  cmake_parse_arguments(
    ARG
    ""
    "NAME;PRIMARY_INTERFACE"
    "PARTITIONS;IMPLEMENTATION;LINK_PUBLIC;LINK_PRIVATE;DEFINES_PRIVATE"
    ${ARGN}
  )
  _nevpt2_require_args("add_cxx_module_library" ARG NAME PRIMARY_INTERFACE)

  add_library(${ARG_NAME} STATIC)
  target_sources(${ARG_NAME} PUBLIC FILE_SET CXX_MODULES FILES ${ARG_PRIMARY_INTERFACE}
                                                            ${ARG_PARTITIONS})
  if(ARG_IMPLEMENTATION)
    target_sources(${ARG_NAME} PRIVATE ${ARG_IMPLEMENTATION})
  endif()

  target_compile_features(${ARG_NAME} PUBLIC cxx_std_23)
  set_target_properties(${ARG_NAME} PROPERTIES CXX_MODULE_STD ON)
  target_compile_options(${ARG_NAME} PRIVATE -Wall)

  if(ARG_LINK_PUBLIC)
    target_link_libraries(${ARG_NAME} PUBLIC ${ARG_LINK_PUBLIC})
  endif()
  if(ARG_LINK_PRIVATE)
    target_link_libraries(${ARG_NAME} PRIVATE ${ARG_LINK_PRIVATE})
  endif()
  if(ARG_DEFINES_PRIVATE)
    target_compile_definitions(${ARG_NAME} PRIVATE ${ARG_DEFINES_PRIVATE})
  endif()

  _nevpt2_create_alias(${ARG_NAME})
endfunction()
