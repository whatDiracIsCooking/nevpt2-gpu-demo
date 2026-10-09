# Internal helpers for the add_* macros (underscore-prefixed -- not part
# of the public API). No export-name helper: this project installs nothing.
include_guard(GLOBAL)

# FATAL_ERROR naming the first required keyword argument that is missing.
function(_nevpt2_require_args label prefix)
  foreach(_arg IN LISTS ARGN)
    if(NOT ${prefix}_${_arg})
      message(FATAL_ERROR "${label}: ${_arg} is required")
    endif()
  endforeach()
endfunction()

# `nevpt2.einsum` -> ALIAS `nevpt2::einsum`. A target is DECLARED by its dotted
# name and LINKED by its `::` alias: a mistyped `::` name is a configure error,
# where a mistyped plain name silently degrades to a `-l` flag.
function(_nevpt2_create_alias target)
  string(FIND "${target}" "." _has_dot)
  if(_has_dot EQUAL -1)
    return()
  endif()
  string(REPLACE "." "::" _alias "${target}")
  add_library(${_alias} ALIAS ${target})
endfunction()
