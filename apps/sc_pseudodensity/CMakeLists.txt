# nevpt2_sc_pseudodensity -- the SC gradient's pseudodensity identity checks
# (Park Eqs. 40-47), the only binary that links nevpt2::gradient. Same shape
# as nevpt2_demo: imports the component modules, carries no device code of its
# own, and main.cppm is a module unit, so it goes in a CXX_MODULES file set
# for the dependency scan rather than in plain sources.
add_executable(nevpt2_sc_pseudodensity)
target_sources(nevpt2_sc_pseudodensity PRIVATE FILE_SET CXX_MODULES FILES main.cppm)
set_target_properties(nevpt2_sc_pseudodensity PROPERTIES CXX_MODULE_STD ON)
target_compile_features(nevpt2_sc_pseudodensity PRIVATE cxx_std_23)
target_compile_options(nevpt2_sc_pseudodensity PRIVATE -Wall)
# Into src/'s binary dir, beside the two demos.
set_target_properties(nevpt2_sc_pseudodensity
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${NEVPT2_SRC_BINARY_DIR}")
target_link_libraries(nevpt2_sc_pseudodensity PRIVATE
  nevpt2::gradient
  nevpt2::golden
  nevpt2::link_tables
  nevpt2::cli
  nevpt2::rdm_build
)
