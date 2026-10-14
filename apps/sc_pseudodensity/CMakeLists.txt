# nevpt2_sc_pseudodensity -- the SC gradient's pseudodensity identity checks
# (Park Eqs. 40-47), the only binary that links nevpt2::gradient. Same shape
# as nevpt2_demo: imports the component modules, carries no device code of its
# own, and main.cppm is a module unit, so it goes in a CXX_MODULES file set
# for the dependency scan rather than in plain sources.
#
# nevpt2.gradsrc -- the `.gradsrc` writer (and the reader its round trip
# needs) that --dump-source-terms uses: the GPU's own Y_xy and y_I as a NumPy
# .npz for reference_data/nevpt2_grad.py --source dump (#73). The one module
# library under apps/, and on purpose: nevpt2.gradient computes, and writing
# what it computed is this binary's decision, not the component's. A library
# rather than a partition only so test/gradsrc/ can round-trip it with no
# card. Host code alone -- no device, no GPU runtime.
add_cxx_module_library(
  NAME nevpt2.gradsrc
  PRIMARY_INTERFACE gradsrc.cppm
  PARTITIONS gradsrc_archive.cppm
  IMPLEMENTATION gradsrc.cpp
  LINK_PUBLIC nevpt2::tensor nevpt2::error_handling
)

add_executable(nevpt2_sc_pseudodensity)
target_sources(nevpt2_sc_pseudodensity PRIVATE FILE_SET CXX_MODULES
               FILES main.cppm kappa_check.cppm kappa_full_check.cppm source_dump.cppm
                     source_oracle.cppm)
set_target_properties(nevpt2_sc_pseudodensity PROPERTIES CXX_MODULE_STD ON)
target_compile_features(nevpt2_sc_pseudodensity PRIVATE cxx_std_23)
target_compile_options(nevpt2_sc_pseudodensity PRIVATE -Wall)
# Into src/'s binary dir, beside the two demos.
set_target_properties(nevpt2_sc_pseudodensity
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${NEVPT2_SRC_BINARY_DIR}")
# The kernels' same-run self-variation envelope (#64). --kappa-full's check 3
# holds scActiveRotationDevice, run twice on one input, to it, so a kernel
# change that makes the atomicAdd nondeterminism materially worse FAILS rather
# than silently widening the tolerance check 3 derives from that same spread.
# The envelope is chosen from a measurement, so its ONE home is
# docs/testing.md ("The kernels' same-run self-variation") -- read from that
# table row here, at configure time, rather than restated in this file or in
# the source. Editing the row re-runs the configure (CMAKE_CONFIGURE_DEPENDS).
set(_nevpt2_testing_md "${CMAKE_CURRENT_SOURCE_DIR}/../../docs/testing.md")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_nevpt2_testing_md}")
file(STRINGS "${_nevpt2_testing_md}" _nevpt2_env_rows
     REGEX "^\\| \\*\\*the recorded envelope\\*\\*")
list(LENGTH _nevpt2_env_rows _nevpt2_env_count)
if(NOT _nevpt2_env_count EQUAL 1)
  message(FATAL_ERROR "nevpt2_sc_pseudodensity: docs/testing.md must carry exactly one "
                      "'| **the recorded envelope** ... | **<x>e-<n>** |' row (\"The kernels' "
                      "same-run self-variation\"); found ${_nevpt2_env_count}")
endif()
if(NOT _nevpt2_env_rows MATCHES "\\| \\*\\*([0-9.]+e-[0-9]+)\\*\\* \\|$")
  message(FATAL_ERROR "nevpt2_sc_pseudodensity: docs/testing.md's recorded-envelope row "
                      "does not end in '| **<x>e-<n>** |': ${_nevpt2_env_rows}")
endif()
target_compile_definitions(nevpt2_sc_pseudodensity
                           PRIVATE NEVPT2_SELF_VARIATION_ENVELOPE=${CMAKE_MATCH_1})
# --source-oracle's tolerance (#92), the same way and for the same reason: it
# is measured -- PySCF's side is a central difference, so the agreement can be
# read only to that quotient's truncation -- and its ONE home is
# docs/testing.md, "The SC source terms against PySCF's, on a card".
file(STRINGS "${_nevpt2_testing_md}" _nevpt2_oracle_rows
     REGEX "^\\| \\*\\*the source-oracle tolerance\\*\\*")
list(LENGTH _nevpt2_oracle_rows _nevpt2_oracle_count)
if(NOT _nevpt2_oracle_count EQUAL 1)
  message(FATAL_ERROR "nevpt2_sc_pseudodensity: docs/testing.md must carry exactly one "
                      "'| **the source-oracle tolerance** ... | **<x>e-<n>** |' row (\"The SC "
                      "source terms against PySCF's, on a card\"); found ${_nevpt2_oracle_count}")
endif()
if(NOT _nevpt2_oracle_rows MATCHES "\\| \\*\\*([0-9.]+e-[0-9]+)\\*\\* \\|$")
  message(FATAL_ERROR "nevpt2_sc_pseudodensity: docs/testing.md's source-oracle tolerance "
                      "row does not end in '| **<x>e-<n>** |': ${_nevpt2_oracle_rows}")
endif()
target_compile_definitions(nevpt2_sc_pseudodensity
                           PRIVATE NEVPT2_SOURCE_ORACLE_TOL=${CMAKE_MATCH_1})
unset(_nevpt2_oracle_rows)
unset(_nevpt2_oracle_count)
unset(_nevpt2_testing_md)
unset(_nevpt2_env_rows)
unset(_nevpt2_env_count)

target_link_libraries(nevpt2_sc_pseudodensity PRIVATE
  nevpt2::gradient
  # --kappa-act's and --kappa-full's host side: expKappa and the finite
  # rotation of every stored block, which their finite differences rerun the
  # energy at.
  nevpt2::kappa_rotation
  # --dump-source-terms: the .gradsrc writer and reader.
  nevpt2::gradsrc
  nevpt2::golden
  nevpt2::link_tables
  nevpt2::cli
  nevpt2::rdm_build
)
