# nevpt2_demo -- the host binary. Imports the component modules; carries no
# device code of its own (the kernels are device libraries the components link).
# main.cppm is a module unit (export module nevpt2.app.integral_direct), so it
# goes in a CXX_MODULES file set for the dependency scan, not in plain sources.
add_executable(nevpt2_demo)
target_sources(nevpt2_demo PRIVATE FILE_SET CXX_MODULES FILES main.cppm)
set_target_properties(nevpt2_demo PROPERTIES CXX_MODULE_STD ON)
target_compile_features(nevpt2_demo PRIVATE cxx_std_23)
target_compile_options(nevpt2_demo PRIVATE -Wall)
# Into src/'s binary dir, not apps/integral_direct/'s, so the binary stays where it always was:
# build/src/nevpt2_demo from the top level, src/build/nevpt2_demo standalone
# (README.md's path).
set_target_properties(nevpt2_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${NEVPT2_SRC_BINARY_DIR}")
target_link_libraries(nevpt2_demo PRIVATE
  nevpt2::energy
  nevpt2::golden
  nevpt2::link_tables
  nevpt2::profile
  nevpt2::cli
  nevpt2::rdm_build
)
