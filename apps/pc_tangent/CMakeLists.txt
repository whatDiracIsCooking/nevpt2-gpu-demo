# nevpt2_pc_tangent -- the PC-NEVPT2 CI-tangent check (the exact Euler
# identity plus a Richardson-verified central finite difference), the only
# binary that links nevpt2::pc_tangent. Same shape as nevpt2_demo: it imports
# the component modules, carries no device code of its own, and main.cppm is a
# module unit, so it goes in a CXX_MODULES file set for the dependency scan
# rather than in plain sources.
add_executable(nevpt2_pc_tangent)
target_sources(nevpt2_pc_tangent PRIVATE FILE_SET CXX_MODULES FILES main.cppm)
set_target_properties(nevpt2_pc_tangent PROPERTIES CXX_MODULE_STD ON)
target_compile_features(nevpt2_pc_tangent PRIVATE cxx_std_23)
target_compile_options(nevpt2_pc_tangent PRIVATE -Wall)
# Into src/'s binary dir, beside the two demos.
set_target_properties(nevpt2_pc_tangent
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${NEVPT2_SRC_BINARY_DIR}")
# nevpt2::pc_tangent re-exports nevpt2::energy, nevpt2::rdm_build and
# nevpt2::rdm_host_dm; the golden reader, the string tables (for the
# determinant count) and the shared flags come in beside it.
target_link_libraries(nevpt2_pc_tangent PRIVATE
  nevpt2::pc_tangent
  nevpt2::golden
  nevpt2::link_tables
  nevpt2::cli
)
