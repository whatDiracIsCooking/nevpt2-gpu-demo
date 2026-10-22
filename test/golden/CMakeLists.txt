# Host-only suites for nevpt2.golden: loadGolden's refusals on a
# missing, truncated or wrongly-tagged file (temp files the test writes) come
# back as an IO Result, never an abort; GoldenFile::require names a missing
# array; the header's reference_kind field round-trips in the NEVPT2G2 layout,
# reads as CASCI from the pre-kind NEVPT2G1 one, and splits the two gates
# (requireSourceTermReference accepts a CASSCF golden,
# requireLagrangianReference refuses it) -- on a spliced file and on the
# committed CASSCF-orbital golden; and the committed CAS(4,4) golden parses with the
# expected header and array shapes. Pure host file parsing, no device: NOT
# REQUIRES_GPU, so these run with no card visible. Nothing here checks an
# energy.
add_gtest_executable(
  NAME golden_tests
  SOURCES golden_tests.cpp
  LINK nevpt2::golden)

# The committed golden the happy path reads, passed in rather than hard-coded
# in the source. Read only, never written.
set(_golden_cas44 "${PROJECT_SOURCE_DIR}/golden/n2_ccpvdz_cas44.nevpt2gold")
if(NOT EXISTS "${_golden_cas44}")
  message(FATAL_ERROR "test/golden: committed golden missing: ${_golden_cas44}")
endif()
target_compile_definitions(golden_tests PRIVATE NEVPT2_TEST_GOLDEN_CAS44="${_golden_cas44}")
unset(_golden_cas44)

# Its CASSCF-orbital twin: the committed NEVPT2G2 file, the one reference_kind
# = CASSCF reference in the tree that is not spliced together by a test
# (docs/reference-data.md, "The MO sidecar: committed CASSCF orbitals"). Read
# only, like the one above.
set(_golden_cas44_casscf "${PROJECT_SOURCE_DIR}/golden/n2_ccpvdz_cas44_casscf.nevpt2gold")
if(NOT EXISTS "${_golden_cas44_casscf}")
  message(FATAL_ERROR "test/golden: committed golden missing: ${_golden_cas44_casscf}")
endif()
target_compile_definitions(
  golden_tests PRIVATE NEVPT2_TEST_GOLDEN_CAS44_CASSCF="${_golden_cas44_casscf}")
unset(_golden_cas44_casscf)

# The layout contract's C++ half (GoldenLayoutContractTests): EVERY golden and
# gradient sidecar present in golden/ at configure time, so both header layouts
# -- NEVPT2G1 (the CASCI goldens) and NEVPT2G2 (the *_casscf* ones) -- go
# through loadGolden / loadGradGold, and a header change the reader does not
# follow fails here instead of in a golden entry. Gitignored files are picked
# up when they have been regenerated, and re-globbed on the next build
# (CONFIGURE_DEPENDS). '|'-joined: a ';' would split the definition. The
# Python half is reference_data/golden_layout_check.py (docs/reference-data.md,
# "The file format").
file(GLOB _contract_goldens CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/golden/*.nevpt2gold")
file(GLOB _contract_gradgolds CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/golden/*.gradgold")
# And the SC source-term oracles (#92; loadSrcGold), each held to the golden
# and gradient sidecar beside it.
file(GLOB _contract_srcgolds CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/golden/*.srcgold")
list(SORT _contract_goldens)
list(SORT _contract_gradgolds)
list(SORT _contract_srcgolds)
list(JOIN _contract_goldens "|" _contract_goldens)
list(JOIN _contract_gradgolds "|" _contract_gradgolds)
list(JOIN _contract_srcgolds "|" _contract_srcgolds)
target_compile_definitions(
  golden_tests PRIVATE NEVPT2_TEST_COMMITTED_GOLDENS="${_contract_goldens}"
                       NEVPT2_TEST_COMMITTED_GRADGOLDS="${_contract_gradgolds}"
                       NEVPT2_TEST_COMMITTED_SRCGOLDS="${_contract_srcgolds}")
unset(_contract_goldens)
unset(_contract_gradgolds)
unset(_contract_srcgolds)

add_gtest_suite_tests(
  TARGET golden_tests
  SUITES GoldenLoadErrorTests GoldenRequireTests GoldenReferenceKindTests
         GoldenReferenceKindGateTests GoldenCommittedCasscfTests GoldenCas44Tests
         GoldenLayoutContractTests SrcGoldLoadErrorTests)

# Host-only suites for the OTHER reader in nevpt2.golden: loadGradGold, the
# `*.gradgold` gradient sidecar (docs/gradient-theory.md, section 3). Same
# contract as loadGolden -- a malformed sidecar is a value-tier IO Error, never
# an abort -- plus a byte-level round trip and the committed CAS(4,4) sidecar
# checked against the golden beside it. No device, so NOT REQUIRES_GPU.
add_gtest_executable(
  NAME gradgold_tests
  SOURCES gradgold_tests.cpp
  LINK nevpt2::golden)

# The committed CAS(4,4) sidecar and the golden it describes, both read only.
set(_gradgold_cas44 "${PROJECT_SOURCE_DIR}/golden/n2_ccpvdz_cas44.gradgold")
if(NOT EXISTS "${_gradgold_cas44}")
  message(FATAL_ERROR "test/golden: committed gradient sidecar missing: ${_gradgold_cas44}")
endif()
# And its density-fitted twin (NEVPT2F1, #164) beside the *_df golden.
set(_gradgold_cas44_df "${PROJECT_SOURCE_DIR}/golden/n2_ccpvdz_cas44_df.gradgold")
if(NOT EXISTS "${_gradgold_cas44_df}")
  message(FATAL_ERROR "test/golden: committed gradient sidecar missing: ${_gradgold_cas44_df}")
endif()
target_compile_definitions(
  gradgold_tests
  PRIVATE NEVPT2_TEST_GRADGOLD_CAS44="${_gradgold_cas44}"
          NEVPT2_TEST_GOLDEN_CAS44="${PROJECT_SOURCE_DIR}/golden/n2_ccpvdz_cas44.nevpt2gold"
          NEVPT2_TEST_GRADGOLD_CAS44_DF="${_gradgold_cas44_df}"
          NEVPT2_TEST_GOLDEN_CAS44_DF="${PROJECT_SOURCE_DIR}/golden/n2_ccpvdz_cas44_df.nevpt2gold")
unset(_gradgold_cas44)
unset(_gradgold_cas44_df)

add_gtest_suite_tests(
  TARGET gradgold_tests
  SUITES GradGoldRoundTripTests GradGoldLoadErrorTests GradGoldCas44Tests GradGoldDfCas44Tests)

# Host-only suites for the integral perturbation sidecar reader (#220):
# loadPertGold, `*.pertgold`. The same contract -- a malformed sidecar is a
# value-tier IO Error, never an abort -- and the one #220 adds: a sidecar
# missing ANY ONE of its arrays is refused whole. Plus the committed N2
# CAS(4,4) CASSCF pair (conventional and density-fitted) against the golden
# and gradient sidecar each belongs to, and refused beside the others. No
# device, so NOT REQUIRES_GPU.
add_gtest_executable(
  NAME pertgold_tests
  SOURCES pertgold_tests.cpp
  LINK nevpt2::golden)

# The committed files, read only: each perturbation sidecar, the golden and
# gradient sidecar it pairs with, and the CASCI pair it must refuse.
set(_pert_defs "")
foreach(_spec "CASSCF|n2_ccpvdz_cas44_casscf" "CASSCF_DF|n2_ccpvdz_cas44_casscf_df"
              "CAS44|n2_ccpvdz_cas44")
  string(REPLACE "|" ";" _spec "${_spec}")
  list(GET _spec 0 _tag)
  list(GET _spec 1 _stem)
  set(_exts nevpt2gold gradgold)
  if(NOT _tag STREQUAL "CAS44")
    list(APPEND _exts pertgold)
  endif()
  foreach(_ext ${_exts})
    set(_file "${PROJECT_SOURCE_DIR}/golden/${_stem}.${_ext}")
    if(NOT EXISTS "${_file}")
      message(FATAL_ERROR "test/golden: committed file missing: ${_file}")
    endif()
    if(_ext STREQUAL "nevpt2gold")
      set(_kind GOLDEN)
    else()
      string(TOUPPER "${_ext}" _kind)
    endif()
    list(APPEND _pert_defs "NEVPT2_TEST_${_kind}_${_tag}=\"${_file}\"")
  endforeach()
endforeach()
target_compile_definitions(pertgold_tests PRIVATE ${_pert_defs})
unset(_pert_defs)
unset(_spec)
unset(_tag)
unset(_stem)
unset(_exts)
unset(_ext)
unset(_file)
unset(_kind)

add_gtest_suite_tests(
  TARGET pertgold_tests
  SUITES PertGoldRoundTripTests PertGoldLoadErrorTests PertGoldCommittedTests)
