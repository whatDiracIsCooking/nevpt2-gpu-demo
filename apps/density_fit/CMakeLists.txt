# nevpt2_df_demo -- the density-fitted SC-NEVPT2 binary: the same RDM build
# and class energies as nevpt2_demo, with the external integrals built in
# slabs from the golden file's three-index B tensors (src/df_integrals) instead
# of read whole. Checks itself against a `generate_golden.py --df` file.
add_executable(nevpt2_df_demo)  # main.cppm: see apps/integral_direct/CMakeLists.txt
target_sources(nevpt2_df_demo PRIVATE FILE_SET CXX_MODULES FILES main.cppm block_check.cppm)
set_target_properties(nevpt2_df_demo PROPERTIES CXX_MODULE_STD ON)
target_compile_features(nevpt2_df_demo PRIVATE cxx_std_23)
target_compile_options(nevpt2_df_demo PRIVATE -Wall)
# Beside nevpt2_demo: build/src/nevpt2_df_demo, src/build/nevpt2_df_demo standalone.
set_target_properties(nevpt2_df_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${NEVPT2_SRC_BINARY_DIR}")
target_link_libraries(nevpt2_df_demo PRIVATE
  nevpt2::df_integrals
  nevpt2::golden
  nevpt2::link_tables
  nevpt2::profile
  nevpt2::cli
  nevpt2::rdm_build
)
