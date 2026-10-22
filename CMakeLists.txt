# The SC-NEVPT2 GPU demo: one source tree (src/), two backends, C++23 modules.
#
# A BUILD TARGETS EXACTLY ONE BACKEND. NEVPT2_GPU_BACKEND is CUDA or HIP and is
# read BEFORE project() (cmake/nevpt2_toolchain.cmake), because it decides
# which toolchain has to exist: the CUDA build needs nvcc and never looks for
# ROCm, the HIP build needs hipcc and never looks for CUDA. Both build the SAME
# src/ files -- WarpWraps (deps/WarpWraps) supplies the backend-neutral wwr*
# names that make that possible; see src/CMakeLists.txt.
#
# This file only wires up testing; the real build lives in src/CMakeLists.txt,
# which is still a self-contained project you can configure directly:
#
#   cmake -S src -B src/build && cmake --build src/build -j
#
# That standalone path is what README.md documents and is deliberately left
# working: both entry points include the same cmake/nevpt2_toolchain.cmake
# before their project() call. What the top level adds is the ctest
# registration: the golden-file entries and the GoogleTest unit tier (test/).
cmake_minimum_required(VERSION 4.2)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(nevpt2_toolchain)

project(nevpt2 LANGUAGES ${NEVPT2_LANGUAGES})

message(STATUS "nevpt2: GPU backend = ${NEVPT2_GPU_BACKEND}")
add_subdirectory(src)

# --- tests ------------------------------------------------------------------
#
# The demo IS the test: it checks its own per-class norms/energies and the
# total against the golden PySCF answer baked into the .nevpt2gold file and
# prints `PASS:` only when every one matches to 1e-7. So a ctest entry is the
# binary plus a PASS_REGULAR_EXPRESSION -- there is no separate assertion tier
# to write, and a run that merely exits 0 without printing PASS (an output
# format change, say) still fails.
#
# EVERY GOLDEN ENTRY NEEDS A REAL CARD and is labelled `gpu` for it. The only
# entries that do not are the GoogleTest unit tier at the bottom (test/,
# labelled `unit`) and a host-sanitizer build's one host canary.
# So `ctest -LE gpu` runs the host-only unit suites -- logic (error kinds,
# propagation, index helpers), never an energy. A green `-LE gpu` run says
# nothing about a number this project computes.
#
# NEVPT2_BUILD_TESTING ON (the default) registers both tiers and FETCHES
# GoogleTest at configure time (FetchContent, below), so a first configure
# needs the network. OFF -- the compile-cuda / compile-hip compile-only presets --
# registers nothing and fetches nothing.
option(NEVPT2_BUILD_TESTING
       "Register the golden-file ctest entries and the GoogleTest unit tier (fetches googletest)"
       ON)

if(NEVPT2_BUILD_TESTING)
  enable_testing()

  # --- GoogleTest ---------------------------------------------------------------
  #
  # Fetched and built from source rather than found: everything here compiles
  # and links against libc++ (cmake/nevpt2_toolchain.cmake), so a distro
  # GoogleTest built against libstdc++ would not link. Built here, it inherits
  # CMAKE_CXX_FLAGS_INIT's -stdlib=libc++. Only under NEVPT2_BUILD_TESTING, so
  # a compile-only configure needs no network. No gmock (nothing mocks).
  # INSTALL_GTEST OFF: this project installs nothing.
  include(FetchContent)
  set(NEVPT2_GOOGLETEST_TAG "v1.17.0" CACHE STRING "googletest git tag to build against")
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG ${NEVPT2_GOOGLETEST_TAG}
    GIT_SHALLOW TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  # GoogleTest's own sources are plain C++ -- no named modules, no `import
  # std` -- so skip clang-scan-deps on every one of its TUs.
  foreach(_gt_target IN ITEMS gtest gtest_main)
    if(TARGET ${_gt_target})
      set_target_properties(${_gt_target} PROPERTIES CXX_SCAN_FOR_MODULES OFF)
    endif()
  endforeach()
  unset(_gt_target)

  # The sanitizer tier (cmake/nevpt2_sanitizers.cmake, all OFF by default).
  # Under the compute-sanitizer preset every demo entry below runs wrapped in
  # compute-sanitizer: the demo targets carry a TEST_LAUNCHER, set in src/
  # when they were created, which ctest applies to add_test(COMMAND <target>).
  # Nothing here sets it. The multiplier scales every TIMEOUT below.
  math(EXPR _nevpt2_timeout "1800 * ${NEVPT2_TEST_TIMEOUT_MULTIPLIER}")

  set(_gold_1010 "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas1010.nevpt2gold")
  set(_gold_1212 "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas1212.nevpt2gold")

  # A missing golden is a WARNING, not an error, and the test is simply not
  # registered -- the same convention the CAS(14,14) case relies on, since that
  # one is gitignored and absent in a fresh clone. Registering a test whose
  # input does not exist would turn "the data is missing" into a failure that
  # reads like "the demo is broken".
  #
  # Both binaries print the same PASS line prefix; add_golden_test runs
  # the integral-direct nevpt2_demo, add_df_golden_test the
  # density-fitted nevpt2_df_demo.
  function(_add_golden_test_with binary name gold)
    if(NOT EXISTS "${gold}")
      message(WARNING
        "nevpt2: golden file missing, test '${name}' not registered: ${gold}")
      return()
    endif()
    add_test(NAME ${name}
             COMMAND ${binary} --golden "${gold}" ${ARGN})
    # PASS_REGULAR_EXPRESSION makes ctest IGNORE the exit code, so a
    # sanitizer finding that does not stop the run -- compute-sanitizer
    # reports an error and lets the process carry on to print PASS; LSan
    # reports at exit, after it -- would pass on the regex alone, whatever
    # --error-exitcode says. FAIL_REGULAR_EXPRESSION wins over it: these are
    # the summary/report lines each tool prints only when it found something
    # (compute-sanitizer's ERROR/RACECHECK SUMMARY with a non-zero count,
    # ASan/LSan's "ERROR: ...Sanitizer", UBSan's "runtime error:"). None of
    # them can appear in an unsanitized run, so the normal presets are
    # unaffected.
    set_tests_properties(${name} PROPERTIES
      FAIL_REGULAR_EXPRESSION
        "ERROR SUMMARY: [1-9];RACECHECK SUMMARY: [1-9];ERROR: [A-Za-z]+Sanitizer;runtime error:"
      PASS_REGULAR_EXPRESSION "PASS: per-class norms/energies and the total match"
      LABELS "gpu"
      TIMEOUT ${_nevpt2_timeout})
  endfunction()
  function(add_golden_test name gold)
    _add_golden_test_with(nevpt2_demo ${name} "${gold}" ${ARGN})
  endfunction()
  function(add_df_golden_test name gold)
    _add_golden_test_with(nevpt2_df_demo ${name} "${gold}" ${ARGN})
  endfunction()

  # --pc: the same binaries checked against block2's
  # partially-contracted answer (`pc_class_energies`/`e_pc_total`, which only
  # some goldens carry -- docs/reference-data.md, "PC fields"). The demo prints
  # `PASS: PC-NEVPT2 ...` only when all eight classes AND the total match to
  # 1e-7 and no class was refused, and that PASS line is worded differently
  # from the SC one, so neither kind of entry can pass on the other's output.
  set(_nevpt2_pc_pass "PASS: PC-NEVPT2 per-class energies and the total match")
  function(add_pc_golden_test name gold)
    _add_golden_test_with(nevpt2_demo ${name} "${gold}" --pc ${ARGN})
    if(TEST ${name})
      set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "${_nevpt2_pc_pass}")
    endif()
  endfunction()
  function(add_df_pc_golden_test name gold)
    _add_golden_test_with(nevpt2_df_demo ${name} "${gold}" --pc ${ARGN})
    if(TEST ${name})
      set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "${_nevpt2_pc_pass}")
    endif()
  endfunction()

  # --tiles 3 is the fewest MEASURED to run correctly at CAS(10,10) on a 10 GB
  # card -- not a number picked for looks. docs/performance.md ("Why CAS(10,10)
  # needs tiling") has why an auto-sized budget picks a split that corrupts device
  # memory instead of failing cleanly.
  add_golden_test(nevpt2_cas1010 "${_gold_1010}" --tiles 3)

  # CAS(12,12) is ~47s on an RTX 3080 against CAS(10,10)'s ~1.8s, so it is
  # labelled `slow` as well and `ctest -LE slow` drops it. --tiles 40 is
  # comfortably above the measured floor: 17 now that the path allocates no
  # R2 (33 with it), with ~1 GB of the card held externally.
  add_golden_test(nevpt2_cas1212 "${_gold_1212}" --tiles 40)
  if(TEST nevpt2_cas1212)
    set_tests_properties(nevpt2_cas1212 PROPERTIES LABELS "gpu;slow")
  endif()

  # The cuBLAS fixed-point-emulation digest is a CUDA-only path (the
  # tensor-core trick has no ROCm equivalent; a HIP build rejects --cublas),
  # and it needs a SMALLER tile than the emitted path: cuBLAS silently
  # declines to emulate above a per-tile-K threshold and runs native fp64
  # instead. --tiles 3 at CAS(10,10) engages; see docs/performance.md, "The cuBLAS
  # fixed-point-emulation digest".
  if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
    add_golden_test(nevpt2_cas1010_cublas "${_gold_1010}" --tiles 3 --cublas)
  endif()

  # --- the gradient-validation molecules: LiF and acrolein -----------------------
  #
  # The two molecules the analytic-gradient work is validated against, each with
  # the active-space SIZE Park used (docs/reference-data.md, "LiF and acrolein,
  # the gradient-validation pair"). LiF is the two-atom smoke test: linear, so
  # Coov mapped to C2v, CAS(4,6) taken as the pure pi space by irrep. Its golden
  # carries NO PC fields -- the generator refuses its PC metric, three blocks'
  # lstsq cut sitting in a continuum -- so there is deliberately no
  # nevpt2_lif_cas46_pc entry. Acrolein is planar Cs, CAS(6,5) = the oxygen lone
  # pair plus 2 pi + 2 pi*, and carries both SC and PC.
  #
  # --tiles 3 is not a measured floor at these active-space sizes (one tile fits,
  # as for the N2 CAS(8,8) entries); it is the count every other entry here uses.
  # Not `small`: the sanitizer tier stays on N2. Both goldens are committed
  # (0.20 MB and 7.1 MB); only acrolein's 57 MB gradient sidecar is gitignored.
  #
  # These three entries check the two molecules' ENERGIES. The end-to-end
  # gradient check -- the assembled analytic gradient against a finite-
  # difference one, SC and PC, on these two plus N2 CAS(4,4) -- registers NO
  # entry here, deliberately: both sides of it are offline Python
  # (reference_data/nevpt2_grad.py and nevpt2_numgrad.py), so an entry would
  # put PySCF and block2 in the test path, it needs no card while the golden
  # tier is 100% gpu-labelled, and acrolein's PC pair runs for hours. It is a
  # manual gate instead, like the sanitizer tier: the commands, what they
  # print and why there is no entry are in docs/testing.md, "The analytic
  # gradient against the numerical one".
  set(_gold_lif46 "${CMAKE_CURRENT_SOURCE_DIR}/golden/lif_def2svp_cas46.nevpt2gold")
  set(_gold_acr65 "${CMAKE_CURRENT_SOURCE_DIR}/golden/acrolein_def2svp_cas65.nevpt2gold")
  add_golden_test(nevpt2_lif_cas46 "${_gold_lif46}" --tiles 3)
  # Its CASSCF-orbital twin (#123's golden; #125, which builds LiF's CASSCF SC
  # gradient from this state's GPU dump, registers it): SC only, like the twin.
  add_golden_test(nevpt2_lif_cas46_casscf
                  "${CMAKE_CURRENT_SOURCE_DIR}/golden/lif_def2svp_cas46_casscf.nevpt2gold"
                  --tiles 3)
  add_golden_test(nevpt2_acrolein_cas65 "${_gold_acr65}" --tiles 3)
  add_pc_golden_test(nevpt2_acrolein_cas65_pc "${_gold_acr65}" --tiles 3)

  # --- the SC gradient's pseudodensity identities --------------------------------
  #
  # nevpt2_sc_pseudodensity (apps/sc_pseudodensity/, the only binary that links
  # nevpt2::gradient): the same RDM build and the same integrals as the entries
  # above, then nevpt2.gradient's assembly of Park Eqs. 41-47 and the EXACT
  # consistency identity of his Eq. 40 -- contracting each class's assembled
  # pseudodensities back against their own operands reproduces that class's
  # energy. Algebraic, so the tolerance is noise (1e-10) and no finite
  # difference is involved; the class energies are also held to 1e-10 against
  # nevpt2.energy's own answer and to 1e-7 against the golden. Its PASS line is
  # worded differently from both demos', so no entry can pass on another's
  # output. CAS(4,4) and CAS(8,8) are `small` too, so the sanitizer tier covers
  # the new device work; CAS(10,10) is not (compute-sanitizer multiplies a
  # run's time -- see "small active spaces" below).
  foreach(_case cas44 cas88 cas1010)
    set(_sc_pd_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.nevpt2gold")
    _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_${_case}
                          "${_sc_pd_gold}" --tiles 3)
    if(TEST nevpt2_sc_pseudodensity_${_case})
      set_tests_properties(nevpt2_sc_pseudodensity_${_case} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: SC pseudodensity identities hold")
      if(NOT _case STREQUAL "cas1010")
        set_tests_properties(nevpt2_sc_pseudodensity_${_case} PROPERTIES LABELS "gpu;small")
      endif()
    endif()
    unset(_sc_pd_gold)
  endforeach()

  # --- the active-active orbital source term, --kappa-act -------------------------
  #
  # nevpt2_sc_pseudodensity --kappa-act (apps/sc_pseudodensity/kappa_check.cppm):
  # the active-active block of Park Eq. 26's Y_xy, taken BOTH WAYS and held
  # against itself -- the adjoint pairing <kappa, Y_act> out of the same
  # pseudodensities the entries above check, and the forward tangent Edot
  # along the same direction, for four seeded antisymmetric directions, to
  # 1e-7 (exact arithmetic, not a step size: it comes out near machine
  # precision). Y_act is also assembled as a MATRIX by probing the basis
  # directions and its pairing checked back, and the whole convention is
  # pinned to nevpt2.kappa_rotation's exp(t kappa) by one central finite
  # difference of energiesDevice at rotated integrals -- the same
  # plateau-and-Richardson analysis as --sc-tangent's, and for the same
  # reason (docs/testing.md, "The active-rotation source term").
  #
  # The sweep reruns the energy stage at 28 displaced integral sets, but NOT
  # the RDM build (the rotation moves integrals only), so these are seconds at
  # every size here. Its PASS line is its own, so no entry can pass on
  # another's output.
  #
  # ONLY CAS(4,4) is `small`. The device work the sanitizer tier is here to
  # watch -- the kappa action, the per-direction pairings, the product rule --
  # is the same code at every size, and the CAS(8,8) entry costs 103 s under
  # memcheck against CAS(4,4)'s 1.4 s (the 28-point sweep and the 28 basis
  # probes, each a full pass), which would be half that preset's whole runtime
  # for no path this does not already cover (docs/testing.md, "The
  # active-rotation source term", and "Wall times" for what the tier is kept
  # under).
  set(_nevpt2_kappa_pass "PASS: the active-rotation source term's adjoint and tangent agree")
  foreach(_case cas44 cas88 cas1010)
    set(_kappa_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.nevpt2gold")
    _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_kappa_${_case}
                          "${_kappa_gold}" --tiles 3 --kappa-act)
    if(TEST nevpt2_sc_pseudodensity_kappa_${_case})
      set_tests_properties(nevpt2_sc_pseudodensity_kappa_${_case} PROPERTIES
        PASS_REGULAR_EXPRESSION "${_nevpt2_kappa_pass}")
      if(_case STREQUAL "cas44")
        set_tests_properties(nevpt2_sc_pseudodensity_kappa_${_case}
                             PROPERTIES LABELS "gpu;small")
      endif()
    endif()
    unset(_kappa_gold)
  endforeach()

  # Its wave-0 canary twin (#118): the same check on CASSCF ORBITALS,
  # golden/n2_ccpvdz_cas44_casscf.nevpt2gold. The source terms are derivatives
  # at fixed orbitals and CI vector with no reference-kind dependence (Park
  # Eqs. 26-28, 36-38, 41-48), so the source-term gate lets this golden
  # through, and the adjoint-vs-tangent duality must hold on it exactly as on
  # the CASCI twin -- with no sidecar, no oracle and no new reference data.
  # #118 named --kappa-full here; that check reads a `<case>.gradgold`
  # sidecar, which no CASSCF golden had until #120, so this sidecar-free check
  # was the canary instead (docs/testing.md, "The source terms on CASSCF
  # orbitals: the wave-0 canary"). `small`, like the CASCI twin. Since #121
  # --kappa-full runs on this golden too, in its loop below, and so do
  # --gradient-source, --dump-source-terms and --source-oracle
  # (docs/testing.md, "The source terms on CASSCF orbitals, measured").
  _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_kappa_cas44_casscf
                        "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44_casscf.nevpt2gold"
                        --tiles 3 --kappa-act)
  if(TEST nevpt2_sc_pseudodensity_kappa_cas44_casscf)
    set_tests_properties(nevpt2_sc_pseudodensity_kappa_cas44_casscf PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_kappa_pass}" LABELS "gpu;small")
  endif()

  # --- the SC gradient's CI source term: forward against reverse ----------------
  #
  # nevpt2_demo --gradient-source: nevpt2.gradient's scCiSourceDevice assembles
  # y_I = dE/dc_I (Park Eq. 48) by sweeping those same pseudodensities one chain
  # rule further in -- down to the five RDM coefficient tensors -- and
  # contracting them with nevpt2.sigma_vector's kernels, then checks it against
  # nevpt2.energy's FORWARD tangent by duality: <u, y> = Edot(u), for four
  # seeded directions and for the state itself, plus the FULL Jacobian column by
  # column where ndet allows it (CAS(4,4): all 36).
  #
  # Two computations of one number with no step size anywhere, so the tolerance
  # is 1e-7 like every other numerical gate here and nothing is differenced --
  # the finite-difference cross-check that licenses this is the tangent's own,
  # one-time (--sc-tangent; docs/testing.md, "The SC tangent's
  # finite-difference cross-check"). What is measured: docs/testing.md, "The CI
  # source term's forward-reverse duality".
  #
  # --tiles is the tangent RDM build's memory lever here, exactly as elsewhere
  # (docs/performance.md, "Tile floors" and "--tiles on the CI source term"):
  # 3 at CAS(10,10) is the shipped count and what this is measured with. Its
  # PASS line is worded unlike either demo's and unlike the pseudodensity one,
  # so no entry can pass on another's output. CAS(4,4) and CAS(8,8) are `small`,
  # so the sanitizer tier covers the sweep and the sigma launches.
  #
  # WHICH FUNCTIONAL (#87). CAS(4,4) and CAS(8,8) have a committed gradient
  # sidecar, so they pass --gradgold and both sides carry the orbital energies'
  # response to the active density -- the COMPLETE functional, and the regex
  # requires the run to say so. CAS(10,10) has no sidecar (none is committed
  # at that size), so it differentiates the functional with e_core / e_virt
  # held at the golden's, and says that instead; the duality is exact on
  # either. What checks the response itself -- which no duality can, since
  # both modes share it -- is docs/testing.md, "The GPU as the source-term
  # provider".
  foreach(_case cas44 cas44_casscf cas88 cas1010)
    set(_ci_src_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.nevpt2gold")
    set(_ci_src_side "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.gradgold")
    if(_case STREQUAL "cas1010")
      set(_ci_src_args)
      set(_ci_src_functional "the orbital energies held fixed")
    else()
      if(NOT EXISTS "${_ci_src_side}")
        message(FATAL_ERROR "nevpt2: committed gradient sidecar missing: ${_ci_src_side}")
      endif()
      set(_ci_src_args --gradgold "${_ci_src_side}")
      set(_ci_src_functional "the orbital energies moving with the active density")
    endif()
    _add_golden_test_with(nevpt2_demo nevpt2_gradient_source_${_case}
                          "${_ci_src_gold}" --tiles 3 --gradient-source ${_ci_src_args})
    if(TEST nevpt2_gradient_source_${_case})
      set_tests_properties(nevpt2_gradient_source_${_case} PROPERTIES
        PASS_REGULAR_EXPRESSION
          "PASS: the CI source term and the forward tangent agree .*, ${_ci_src_functional}")
      if(NOT _case STREQUAL "cas1010")
        set_tests_properties(nevpt2_gradient_source_${_case} PROPERTIES LABELS "gpu;small")
      endif()
    endif()
    unset(_ci_src_gold)
    unset(_ci_src_side)
    unset(_ci_src_args)
    unset(_ci_src_functional)
  endforeach()

  # --- PC-NEVPT2's CI source term: --gradient-source --pc (#74) -------------------
  #
  # The same three checks for PC-NEVPT2's y_I (src/gradient/
  # ci_checks_source.cpp, checkPcCiSource): nevpt2.gradient's
  # pcCiSourceDevice -- the PC solve's adjoint, swept through SC's own block
  # adjoints and the same sigma assembly -- against nevpt2.energy's
  # pcTangentsDevice, the solve's tangent, along four seeded directions and
  # the state itself and, at CAS(4,4), column by column over the Jacobian.
  # Exact arithmetic, 1e-7, both with the orbital energies' response read from
  # the committed sidecar. The adjoint's own walk is held to the golden's block2
  # e_pc_total too. The tangent's one-time license is nevpt2_pc_tangent's
  # finite difference (below), as --sc-tangent's is SC's. Both `small`, so the
  # sanitizer tier runs the PC adjoint's device work. What the duality cannot
  # see -- a term both modes leave out -- is the oracle entry's job below.
  foreach(_case cas44 cas44_casscf cas88)
    set(_pc_src_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.nevpt2gold")
    set(_pc_src_side "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.gradgold")
    if(NOT EXISTS "${_pc_src_side}")
      message(FATAL_ERROR "nevpt2: committed gradient sidecar missing: ${_pc_src_side}")
    endif()
    _add_golden_test_with(nevpt2_demo nevpt2_gradient_source_pc_${_case}
                          "${_pc_src_gold}" --tiles 3 --pc --gradient-source
                          --gradgold "${_pc_src_side}")
    if(TEST nevpt2_gradient_source_pc_${_case})
      set_tests_properties(nevpt2_gradient_source_pc_${_case} PROPERTIES
        PASS_REGULAR_EXPRESSION
          "PASS: the PC CI source term and the PC forward tangent agree .*, the orbital energies moving with the active density"
        LABELS "gpu;small")
    endif()
    unset(_pc_src_gold)
    unset(_pc_src_side)
  endforeach()

  # --- the FULL orbital source term, --kappa-full ---------------------------------
  #
  # nevpt2_sc_pseudodensity --kappa-full
  # (apps/sc_pseudodensity/kappa_full_check.cppm): Park Eq. 26's Y_xy over the
  # WHOLE kappa space -- the core-core, virtual-virtual and mixed blocks on top
  # of the active-active one the entries above check -- assembled from the
  # gradient SIDECAR beside each golden (`<case>.gradgold`, so only the cases
  # that have one can run this). Five checks, and the gate is their conjunction:
  # the duality on four seeded full directions (1e-7, exact arithmetic), the
  # sidecar's promotion table against the arrays it reads (1e-10), the active
  # block unchanged bit for bit when kappa is active-restricted (to the
  # kernels' same-run spread, which must itself stay inside the envelope
  # docs/testing.md records, "The kernels' same-run self-variation"), two
  # finite differences -- a core-internal and a virtual-internal rotation,
  # which are the only ones the stored blocks can displace -- and Park
  # Eqs. 56-57's pseudocanonical multipliers coming out determined -- and, for
  # the f3ac/f3ca digests' own term, forward (#88) and reverse (#89): the FULL
  # duality on the same four directions (the exact <kappa, Y>, the digests'
  # transpose in it, against the exact Edot; 1e-7), an exact zero along the
  # two intra-space rotations in both modes and a third finite difference, the
  # energy with only the digests rebuilt along an active direction, with the
  # reverse term held to it too. Its PASS line is its own, so no entry can
  # pass on another's output (docs/testing.md, "The full orbital source term"
  # and "The digests' orbital response, forward mode" / "reverse mode").
  #
  # LiF is here for the Eqs. 56-57 half: its four core orbital energies are all
  # distinct, the smallest gap 0.85 Eh against the 3.3e-3 Eh 1s pair in N2's
  # core, so its core multipliers are the well-separated case those divisions
  # want. N2's exactly degenerate virtual pairs are the other half of that
  # test -- they sit in different irreps, so their numerators vanish with the
  # gap and the multiplier is left zero rather than divided through.
  #
  # Not `small`: the two 26-point finite-difference sweeps and the ~180 basis
  # probes are each a full pass over everything, which is what made the
  # CAS(8,8) --kappa-act entry too slow for the sanitizer tier
  # (docs/testing.md, "The active-rotation source term"); CAS(4,4)'s --kappa-act
  # entry already covers this code's device work there.
  set(_nevpt2_kappa_full_pass
      "PASS: the full orbital source term's adjoint and tangent agree")
  foreach(_case n2_ccpvdz_cas44 n2_ccpvdz_cas44_casscf n2_ccpvdz_cas88
                lif_def2svp_cas46)
    set(_kf_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.nevpt2gold")
    set(_kf_side "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.gradgold")
    # Both files are committed for these four cases; the sidecar is checked
    # for anyway, as the golden is, so a tree missing one warns instead of
    # registering an entry that cannot run (acrolein's 57 MB sidecar, for
    # instance, is gitignored -- docs/reference-data.md, "The gradient
    # sidecar").
    if(NOT EXISTS "${_kf_side}")
      message(WARNING "nevpt2: gradient sidecar missing, test "
                      "'nevpt2_sc_pseudodensity_kappa_full_${_case}' not registered: "
                      "${_kf_side}")
    else()
      _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_kappa_full_${_case}
                            "${_kf_gold}" --tiles 3 --kappa-full)
      if(TEST nevpt2_sc_pseudodensity_kappa_full_${_case})
        set_tests_properties(nevpt2_sc_pseudodensity_kappa_full_${_case} PROPERTIES
          PASS_REGULAR_EXPRESSION "${_nevpt2_kappa_full_pass}")
      endif()
    endif()
    unset(_kf_gold)
    unset(_kf_side)
  endforeach()

  # --- the GPU's own source terms, written: --dump-source-terms -------------------
  #
  # nevpt2_sc_pseudodensity --dump-source-terms (apps/sc_pseudodensity/
  # source_dump.cppm, #73): the whole Y_xy (out of the same sidecar --kappa-full
  # reads) and y_I, written to a `.gradsrc` -- a NumPy .npz -- in this build
  # directory, then read back. The derivatives themselves are the --kappa-full
  # and --gradient-source entries' to check; this one checks that the probed
  # matrix IS the pairing (one seeded full direction, reassociation noise), that
  # y's walk ran on this state (its E_corr against energiesDevice, 1e-7), that
  # each written Edot direction's exact value -- frozen-digest tangent plus the
  # digests' term -- is <kappa, Y> off the written matrix, which carries the
  # digests' transpose (#89, 1e-7), and that the file reads back bit for bit. Its
  # PASS line is its own. The written Edot -- frozen-digest tangent plus the
  # digests' term -- against PySCF's exact directional derivative is
  # nevpt2_grad.py --edot-check, the same manual gate as below.
  #
  # What a NUCLEAR gradient built from the file comes out at is NOT an entry
  # here: that comparison runs reference_data/nevpt2_grad.py --source both, so
  # it would put PySCF in the test path and it needs no card -- the same three
  # reasons the PySCF-sourced gradient has none (docs/testing.md, "The GPU as
  # the source-term provider" and "Why there is no ctest entry for it"). It is
  # the same manual gate.
  #
  # Not `small`: the dump's edot directions and digests' terms are full passes
  # each (and until #240 Y was one basis direction per MO pair, 378 at N2, 253
  # at LiF), --kappa-full's own reason for staying out of the sanitizer tier;
  # the writer's host code is the unit tier's (test/gradsrc/), which every
  # sanitizer preset runs. Acrolein is not here because its sidecar is
  # gitignored (its --probed-y entries below register where it exists).
  # lif_def2svp_cas46_casscf (#125) sits beside its CASCI twin: its dump is the
  # GPU-sourced half of LiF's CASSCF SC gradient verdict (docs/testing.md, "SC
  # on a CASSCF reference: LiF, and the N2 degeneracy probe").
  set(_nevpt2_dump_dir "${CMAKE_CURRENT_BINARY_DIR}/gradsrc")
  file(MAKE_DIRECTORY "${_nevpt2_dump_dir}")
  foreach(_case n2_ccpvdz_cas44 n2_ccpvdz_cas44_casscf n2_ccpvdz_cas88
                lif_def2svp_cas46 lif_def2svp_cas46_casscf)
    set(_dump_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.nevpt2gold")
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.gradgold")
      message(WARNING "nevpt2: gradient sidecar missing, test "
                      "'nevpt2_sc_pseudodensity_dump_${_case}' not registered")
    else()
      _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_dump_${_case}
                            "${_dump_gold}" --tiles 3
                            --dump-source-terms "${_nevpt2_dump_dir}/${_case}.gradsrc")
      if(TEST nevpt2_sc_pseudodensity_dump_${_case})
        set_tests_properties(nevpt2_sc_pseudodensity_dump_${_case} PROPERTIES
          PASS_REGULAR_EXPRESSION
            "PASS: the GPU's source terms Y_xy and y_I were assembled, written to")
      endif()
    endif()
    unset(_dump_gold)
  endforeach()

  # --- the one-pass Y against the probed one: --dump-source-terms --probed-y ------
  #
  # Since #240 the dump writes Y from ONE reverse pass (nevpt2.gradient's
  # scFullRotationMatrixOnePass / pcFullRotationMatrixOnePass: the walk's
  # adjoints pulled back through the Dyall blocks' transposes and the kappa
  # action, docs/gradients.md, Sec. 6), and every dump entry above and below
  # gates check 7, those transposes against their builders by duality. These
  # entries add check 8: the probed Y -- one basis direction per MO pair, the
  # route the one-pass one replaced, kept as its oracle -- computed beside it
  # and held to it element by element within the probed matrix's own bound
  # (kDumpMatrixTol, 1e-9 of max(1, |Y|)). SC and PC on N2 CAS(4,4)'s CASSCF
  # state, SC on LiF CAS(4,6) (its golden carries no PC reference), SC and PC
  # on the C1 case HSOH CAS(6,6), SC on the density-fitted twin, and acrolein
  # CAS(6,5) on CASSCF orbitals where its gitignored sidecar has been
  # regenerated (docs/reference-data.md, "The gradient sidecar"). HSOH and
  # acrolein are `slow`: their probe is the hundreds-of-directions cost the
  # one-pass route exists to avoid (docs/performance.md, "The one-pass `Y`").
  foreach(_spec "n2_ccpvdz_cas44_casscf|sc" "n2_ccpvdz_cas44_casscf|pc"
                "lif_def2svp_cas46|sc" "hsoh_def2svp_cas66_casscf|sc"
                "hsoh_def2svp_cas66_casscf|pc" "n2_ccpvdz_cas44_df|sc"
                "acrolein_def2svp_cas65_casscf|sc" "acrolein_def2svp_cas65_casscf|pc")
    string(REPLACE "|" ";" _spec "${_spec}")
    list(GET _spec 0 _case)
    list(GET _spec 1 _scheme)
    if(_scheme STREQUAL "pc")
      set(_flags --pc)
      set(_name nevpt2_sc_pseudodensity_dump_probed_pc_${_case})
      set(_out "${_nevpt2_dump_dir}/${_case}_probed_pc.gradsrc")
      set(_pass "PASS: the GPU's PC source terms Y_xy and y_I were assembled, written to")
    else()
      set(_flags "")
      set(_name nevpt2_sc_pseudodensity_dump_probed_${_case})
      set(_out "${_nevpt2_dump_dir}/${_case}_probed.gradsrc")
      set(_pass "PASS: the GPU's source terms Y_xy and y_I were assembled, written to")
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.gradgold")
      message(WARNING "nevpt2: gradient sidecar missing, test '${_name}' not registered")
    else()
      _add_golden_test_with(nevpt2_sc_pseudodensity ${_name}
                            "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.nevpt2gold" --tiles 3
                            ${_flags} --dump-source-terms "${_out}" --probed-y)
      if(TEST ${_name})
        set_tests_properties(${_name} PROPERTIES PASS_REGULAR_EXPRESSION "${_pass}")
        if(_case MATCHES "^(hsoh|acrolein)_")
          set_tests_properties(${_name} PROPERTIES LABELS "gpu;slow")
        endif()
      endif()
    endif()
  endforeach()
  unset(_spec)
  unset(_case)
  unset(_scheme)
  unset(_flags)
  unset(_name)
  unset(_out)
  unset(_pass)
  unset(_nevpt2_dump_dir)

  # --- the GPU's source terms against PySCF's: --source-oracle ------------------
  #
  # nevpt2_sc_pseudodensity --source-oracle (apps/sc_pseudodensity/
  # source_oracle.cppm, #92): the dump's own Y_xy and y_I, compared BLOCK BY
  # BLOCK with PySCF's exact source terms, committed as golden/<case>.srcgold
  # (reference_data/sc_source_oracle.py; docs/reference-data.md, "The SC
  # source-term oracle"). Every other check of those two terms compares the
  # device with itself, so a term both its modes leave out cancels there; this
  # one is independent, and it is the offline `nevpt2_grad.py --source both`
  # comparison of #73 with PySCF's side committed rather than recomputed.
  #
  # The known gaps are PINNED: a block on source_compare.cppm's kKnownGaps list
  # must still disagree by more than the tolerance, every other block must
  # agree within it -- so the entry fails when a correct block breaks AND when
  # a fix lands without the list being updated. The list is empty since #89:
  # every block of both oracles is held to agree. The tolerance
  # is not 1e-7, because PySCF's side is a central difference: it is measured
  # from the step ladders, and docs/testing.md, "The SC source terms against
  # PySCF's, on a card" is its one home (read into the binary at configure
  # time). The PASS line is its own.
  #
  # Not `small`, for the dump's reason: Y is one basis direction per MO pair,
  # each a full slab walk.
  foreach(_case n2_ccpvdz_cas44 n2_ccpvdz_cas44_casscf lif_def2svp_cas46)
    set(_oracle_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.nevpt2gold")
    set(_oracle_file "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.srcgold")
    foreach(_need "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.gradgold" "${_oracle_file}")
      if(NOT EXISTS "${_need}")
        message(FATAL_ERROR "nevpt2: committed source-oracle input missing: ${_need}")
      endif()
    endforeach()
    _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_oracle_${_case}
                          "${_oracle_gold}" --tiles 3 --source-oracle "${_oracle_file}")
    if(TEST nevpt2_sc_pseudodensity_oracle_${_case})
      set_tests_properties(nevpt2_sc_pseudodensity_oracle_${_case} PROPERTIES
        PASS_REGULAR_EXPRESSION
          "PASS: the GPU's SC source terms against PySCF's committed ones, ${_case}:")
    endif()
    unset(_oracle_gold)
    unset(_oracle_file)
  endforeach()

  # --- PC-NEVPT2's source terms, written and checked (#74) -----------------------
  #
  # nevpt2_sc_pseudodensity --pc --dump-source-terms: PC-NEVPT2's Y_xy and y_I
  # by the PC solve's adjoint (nevpt2.gradient's PC reverse mode, with the
  # f3ac/f3ca digests' transpose in Y and the orbital energies' response in
  # y), written to a `.gradsrc` marked `pc` and read back; its gates are the
  # SC dump's four plus a fifth -- the PC integral halves' duality, the adjoint
  # pairing against the forward tangent along six seeded full directions
  # (1e-7) -- with check 4 the FULL duality (the digests' transpose against the
  # rebuilt digests' forward term). Not `small`, for the SC dump's reason.
  #
  # --pc --source-oracle: the same terms against block2's EXACT PC source
  # terms, committed as golden/n2_ccpvdz_cas44_pc.srcgold (#95), block by
  # block, with an EMPTY known-gap list and the PC oracle tolerance
  # (docs/testing.md, "The PC source terms' oracle" -- its one home). The
  # independent provider: what catches a term the adjoint and the tangent
  # both leave out, which no duality can.
  set(_nevpt2_pc_dump_dir "${CMAKE_CURRENT_BINARY_DIR}/gradsrc")
  file(MAKE_DIRECTORY "${_nevpt2_pc_dump_dir}")
  foreach(_case n2_ccpvdz_cas44 n2_ccpvdz_cas44_casscf n2_ccpvdz_cas88)
    set(_pc_dump_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.nevpt2gold")
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.gradgold")
      message(FATAL_ERROR "nevpt2: committed gradient sidecar missing for ${_case}")
    endif()
    _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_dump_pc_${_case}
                          "${_pc_dump_gold}" --tiles 3 --pc
                          --dump-source-terms "${_nevpt2_pc_dump_dir}/${_case}_pc.gradsrc")
    if(TEST nevpt2_sc_pseudodensity_dump_pc_${_case})
      set_tests_properties(nevpt2_sc_pseudodensity_dump_pc_${_case} PROPERTIES
        PASS_REGULAR_EXPRESSION
          "PASS: the GPU's PC source terms Y_xy and y_I were assembled, written to")
      # CAS(8,8) is `slow`: its Y probe rebuilds every class's n^6 Dyall block
      # per direction, minutes on the RX 9060 XT (docs/pc-nevpt2.md, "The
      # source terms by reverse mode").
      if(_case STREQUAL "n2_ccpvdz_cas88")
        set_tests_properties(nevpt2_sc_pseudodensity_dump_pc_${_case} PROPERTIES
                             LABELS "gpu;slow")
      endif()
    endif()
    unset(_pc_dump_gold)
  endforeach()
  unset(_nevpt2_pc_dump_dir)
  #
  # n2_ccpvdz_cas44_casscf (#121) is the same state's CASSCF-orbital twin
  # (#117's oracle): the source terms are derivatives at fixed orbitals and CI
  # vector, so nothing in them depends on how the orbitals were made, and the
  # same tolerance holds (docs/pc-nevpt2.md, "On a CASSCF reference", under
  # "The source terms by reverse mode", #160: the PC dump and the PC full
  # oracle on this golden, block by block, kKnownGaps empty).
  foreach(_case n2_ccpvdz_cas44 n2_ccpvdz_cas44_casscf)
    foreach(_need "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.gradgold"
                  "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}_pc.srcgold")
      if(NOT EXISTS "${_need}")
        message(FATAL_ERROR "nevpt2: committed source-oracle input missing: ${_need}")
      endif()
    endforeach()
    _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_oracle_pc_${_case}
                          "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.nevpt2gold"
                          --tiles 3 --pc --source-oracle
                          "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}_pc.srcgold")
    if(TEST nevpt2_sc_pseudodensity_oracle_pc_${_case})
      set_tests_properties(nevpt2_sc_pseudodensity_oracle_pc_${_case} PROPERTIES
        PASS_REGULAR_EXPRESSION
          "PASS: the GPU's PC source terms against block2's committed ones, ${_case}_pc:")
    endif()
  endforeach()

  # --- the directional source-term oracles (#105) --------------------------------
  #
  # The same --source-oracle comparison against a DIRECTIONAL oracle,
  # golden/<case>[_pc]_dir.srcgold (reference_data/sc_source_oracle.py
  # --directional): the GPU's full Y and y projected onto seeded directions per
  # block, block by block, empty known-gap lists, and the directional
  # tolerances (docs/testing.md, "The directional source-term oracle" -- their
  # one home). Acrolein CAS(6,5) is why the flavour exists: its full oracle
  # would take hours to generate, its directional ones take minutes. N2
  # CAS(4,4) and LiF sit beside their full oracles as the cross-check of the
  # two flavours. Acrolein on its CASSCF orbitals (#135) has the same pair,
  # golden/acrolein_def2svp_cas65_casscf[_pc]_dir.srcgold.
  #
  # Acrolein's gradient sidecars, CASCI and CASSCF, are gitignored (57 MB
  # each; docs/reference-data.md, "The gradient sidecar"), so its four entries
  # are registered only where the sidecar has been regenerated, with a warning
  # otherwise -- the --kappa-full entries' rule. They are `slow`: the
  # comparison assembles the whole Y, one basis direction per MO pair
  # (docs/testing.md, "The directional source-term oracle", has their wall
  # time).
  foreach(_spec "n2_ccpvdz_cas44|sc" "n2_ccpvdz_cas44|pc"
                "n2_ccpvdz_cas44_casscf|sc" "n2_ccpvdz_cas44_casscf|pc" "lif_def2svp_cas46|sc"
                "acrolein_def2svp_cas65|sc" "acrolein_def2svp_cas65|pc"
                "acrolein_def2svp_cas65_casscf|sc" "acrolein_def2svp_cas65_casscf|pc")
    string(REPLACE "|" ";" _spec "${_spec}")
    list(GET _spec 0 _case)
    list(GET _spec 1 _scheme)
    if(_scheme STREQUAL "pc")
      set(_stem "${_case}_pc_dir")
      set(_flags --pc)
      set(_pass "PASS: the GPU's PC source terms against block2's committed ones, ${_stem}:")
      set(_name nevpt2_sc_pseudodensity_oracle_dir_pc_${_case})
    else()
      set(_stem "${_case}_dir")
      set(_flags "")
      set(_pass "PASS: the GPU's SC source terms against PySCF's committed ones, ${_stem}:")
      set(_name nevpt2_sc_pseudodensity_oracle_dir_${_case})
    endif()
    set(_dir_oracle "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_stem}.srcgold")
    set(_dir_side "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.gradgold")
    if(NOT EXISTS "${_dir_oracle}")
      message(FATAL_ERROR "nevpt2: committed source-oracle input missing: ${_dir_oracle}")
    endif()
    if(NOT EXISTS "${_dir_side}")
      message(WARNING "nevpt2: gradient sidecar missing, test '${_name}' not registered: "
                      "${_dir_side}")
    else()
      _add_golden_test_with(nevpt2_sc_pseudodensity ${_name}
                            "${CMAKE_CURRENT_SOURCE_DIR}/golden/${_case}.nevpt2gold"
                            --tiles 3 ${_flags} --source-oracle "${_dir_oracle}")
      if(TEST ${_name})
        set_tests_properties(${_name} PROPERTIES PASS_REGULAR_EXPRESSION "${_pass}")
        if(_case MATCHES "^acrolein_def2svp_cas65")
          set_tests_properties(${_name} PROPERTIES LABELS "gpu;slow")
        endif()
      endif()
    endif()
  endforeach()
  unset(_spec)
  unset(_case)
  unset(_scheme)
  unset(_stem)
  unset(_flags)
  unset(_pass)
  unset(_name)
  unset(_dir_oracle)
  unset(_dir_side)

  # --- the density-fitted orbital source term (#165, #168) ---------------------
  #
  # nevpt2_sc_pseudodensity on a `*_df` golden, beside its NEVPT2F1 sidecar
  # (golden/n2_ccpvdz_cas44_df.gradgold, #164). No --df flag: the golden
  # carries its fitted four-index blocks, so the identities and --kappa-act
  # read it unchanged, and --kappa-full knows the case by the sidecar's magic,
  # contracts the one-general-index inventory out of the whole B[L,p,q] (the
  # golden's four blocks and the sidecar's B_cc / B_vv) and runs the same
  # checks with BOTH of Y's terms (#168) -- the integral halves' duality, the
  # frozen-digest one and the FULL one with the digests' transpose against
  # the rebuilt digests' term (1e-7), the promotion table and the sidecar's
  # carried tied arrays against the built inventory (1e-10), the active block
  # against --kappa-act's entry point, the two intra-space finite differences
  # with the digests' term exactly zero on both, Eqs. 56-57, and the digests'
  # term against its own finite difference, forward and reverse -- plus the
  # transpose carried onto the three-index B (dfDigestSourceDevice) against
  # the four-index pairing (1e-7). Its PASS line names the fitted case.
  # --dump-source-terms writes the fitted Y and y with both terms and Edot's
  # two along the seeded directions, read back bit for bit (#166's input).
  # Two refusals stay, each an entry whose regex is the error: --pc on a
  # fitted case (no PC dump over the fitted inventory; its source term is gated
  # only inside --pc --sc-hessian-kk-int, #200), and --source-oracle
  # (no fitted oracle exists; docs/gradients.md, Sec. 6.2).
  #
  # The identity and --kappa-act entries are `small`, as their integral-direct
  # CAS(4,4) twins are; --kappa-full and the dump are not, for the twins'
  # reason (their sweeps and probes are each a full pass).
  set(_df_y_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44_df.nevpt2gold")
  set(_df_y_side "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44_df.gradgold")
  if(NOT EXISTS "${_df_y_side}")
    message(FATAL_ERROR "nevpt2: committed gradient sidecar missing: ${_df_y_side}")
  endif()
  _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_cas44_df
                        "${_df_y_gold}" --tiles 3)
  if(TEST nevpt2_sc_pseudodensity_cas44_df)
    set_tests_properties(nevpt2_sc_pseudodensity_cas44_df PROPERTIES
      PASS_REGULAR_EXPRESSION "PASS: SC pseudodensity identities hold" LABELS "gpu;small")
  endif()
  _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_kappa_cas44_df
                        "${_df_y_gold}" --tiles 3 --kappa-act)
  if(TEST nevpt2_sc_pseudodensity_kappa_cas44_df)
    set_tests_properties(nevpt2_sc_pseudodensity_kappa_cas44_df PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_kappa_pass}" LABELS "gpu;small")
  endif()
  # CAS(8,8)'s fitted sidecar is committed too (#164), so --kappa-full runs
  # there as well, as its integral-direct twin does.
  foreach(_case cas44_df cas88_df)
    set(_df_kf_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.nevpt2gold")
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.gradgold")
      message(FATAL_ERROR "nevpt2: committed gradient sidecar missing for n2_ccpvdz_${_case}")
    endif()
    _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_kappa_full_${_case}
                          "${_df_kf_gold}" --tiles 3 --kappa-full)
    if(TEST nevpt2_sc_pseudodensity_kappa_full_${_case})
      set_tests_properties(nevpt2_sc_pseudodensity_kappa_full_${_case} PROPERTIES
        PASS_REGULAR_EXPRESSION
          "PASS: the full orbital source term's adjoint and tangent agree to [^ ]+ on every direction \\(density-fitted: both Y terms")
    endif()
    unset(_df_kf_gold)
  endforeach()
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gradsrc")
  _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_dump_cas44_df
                        "${_df_y_gold}" --tiles 3
                        --dump-source-terms "${CMAKE_CURRENT_BINARY_DIR}/gradsrc/n2_ccpvdz_cas44_df.gradsrc")
  if(TEST nevpt2_sc_pseudodensity_dump_cas44_df)
    set_tests_properties(nevpt2_sc_pseudodensity_dump_cas44_df PROPERTIES
      PASS_REGULAR_EXPRESSION
        "PASS: the GPU's source terms Y_xy and y_I were assembled, written to .* \\(density-fitted: both Y terms")
  endif()
  _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_dump_pc_cas44_df_refused
                        "${_df_y_gold}" --tiles 3 --pc
                        --dump-source-terms "${CMAKE_CURRENT_BINARY_DIR}/gradsrc/refused_df_pc.gradsrc")
  if(TEST nevpt2_sc_pseudodensity_dump_pc_cas44_df_refused)
    set_tests_properties(nevpt2_sc_pseudodensity_dump_pc_cas44_df_refused PROPERTIES
      PASS_REGULAR_EXPRESSION
        "--pc --dump-source-terms on a density-fitted case: the PC dump over the fitted inventory is not built")
  endif()
  _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_oracle_cas44_df_refused
                        "${_df_y_gold}" --tiles 3
                        --source-oracle "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44_dir.srcgold")
  if(TEST nevpt2_sc_pseudodensity_oracle_cas44_df_refused)
    set_tests_properties(nevpt2_sc_pseudodensity_oracle_cas44_df_refused PROPERTIES
      PASS_REGULAR_EXPRESSION
        "--source-oracle on a density-fitted case: there is no fitted oracle to compare against")
  endif()

  # --- the mixed orbital-CI Hessian block H_kc, --sc-hessian-kc (#184) -----------
  #
  # nevpt2_sc_pseudodensity --sc-hessian-kc (apps/sc_pseudodensity/
  # hessian_kc_check.cppm; docs/hessian-theory.md, Sec. 5.5): the SC energy's
  # mixed second derivative in its two orientations -- the scalar
  # Eddot_kc(w, u) forward over forward, the vector H_kc^T w reverse in c --
  # held to each other by DUALITY (1e-7; column by column at CAS(4,4)), with
  # Euler in c on the homogeneous classes, the digests' share merged per tuple
  # against the first-order source term, the intra-space exact zeros, the
  # full-range f identity for the orbital-energy term's generator, three
  # Richardson-verified finite differences of y (core-core, virtual-virtual,
  # active-active, the orbital energies responding on both legs), and the
  # no-sidecar functional with its refusals. Both cases read the committed
  # sidecar beside the golden. CAS(4,4) is `small`, so the sanitizer tier runs
  # its device work; CAS(8,8)'s four ladders are a full y each per rung, which
  # keeps it out (docs/testing.md, "The SC mixed Hessian block's checks").
  set(_nevpt2_sc_hessian_kc_pass "PASS: the SC mixed Hessian block H_kc's two orientations agree")
  foreach(_case cas44 cas88)
    set(_kc_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.nevpt2gold")
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.gradgold")
      message(FATAL_ERROR "nevpt2: committed gradient sidecar missing for n2_ccpvdz_${_case}")
    endif()
    _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_${_case}_sc_hessian_kc
                          "${_kc_gold}" --tiles 3 --sc-hessian-kc)
    unset(_kc_gold)
  endforeach()
  if(TEST nevpt2_sc_pseudodensity_cas44_sc_hessian_kc)
    set_tests_properties(nevpt2_sc_pseudodensity_cas44_sc_hessian_kc PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_sc_hessian_kc_pass}" LABELS "gpu;small")
  endif()
  if(TEST nevpt2_sc_pseudodensity_cas88_sc_hessian_kc)
    set_tests_properties(nevpt2_sc_pseudodensity_cas88_sc_hessian_kc PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_sc_hessian_kc_pass}")
  endif()

  # --- the source terms' tangent along an integral perturbation (#220) ----------
  #
  # nevpt2_sc_pseudodensity --sc-source-tangent PATH (apps/sc_pseudodensity/
  # source_tangent_check.cppm; docs/hessian-theory.md, Sec. 8): the gated SC
  # source terms Y and y differentiated along each nuclear coordinate's
  # perturbed Hamiltonian, the committed `.pertgold` beside the golden
  # (reference_data/integral_perturbation.py). Per coordinate: the dotted
  # inventory against itself, every probe of Ydot in both modes, the DUALITY
  # <Ydot, kappa> + <ydot, u> against d^2E/dh dtheta forward over forward
  # (1e-7), and Richardson-verified ladders of the whole Y and y recomputed at
  # I +- h Idot. On N2 the x and y coordinates leave the point group and
  # populate empty perturbers, so they are refused (Numerical) and the PASS
  # line counts them: 2 of 6 computed, 4 refused. N2 CAS(4,4) on CASSCF
  # orbitals and its density-fitted twin (its NEVPT2F1 sidecar committed for
  # this). Not `small`: each is about a minute of probes and ladders
  # (docs/testing.md, "The source terms' tangent along an integral
  # perturbation").
  set(_nevpt2_sc_source_tangent_pass
      "PASS: the SC source terms' tangent along the integral perturbation agrees with itself by duality to 1e-07 .* on 2 of 6 coordinates; 4 refused, populating empty perturbers \\(Sec. 5.6\\)")
  foreach(_case cas44_casscf cas44_casscf_df)
    set(_st_stem "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}")
    foreach(_st_ext gradgold pertgold)
      if(NOT EXISTS "${_st_stem}.${_st_ext}")
        message(FATAL_ERROR "nevpt2: committed sidecar missing: ${_st_stem}.${_st_ext}")
      endif()
    endforeach()
    _add_golden_test_with(nevpt2_sc_pseudodensity
                          nevpt2_sc_pseudodensity_${_case}_sc_source_tangent
                          "${_st_stem}.nevpt2gold" --tiles 3
                          --sc-source-tangent "${_st_stem}.pertgold")
    if(TEST nevpt2_sc_pseudodensity_${_case}_sc_source_tangent)
      set_tests_properties(nevpt2_sc_pseudodensity_${_case}_sc_source_tangent PROPERTIES
        PASS_REGULAR_EXPRESSION "${_nevpt2_sc_source_tangent_pass}")
    endif()
    unset(_st_stem)
  endforeach()
  if(TEST nevpt2_sc_pseudodensity_cas44_casscf_df_sc_source_tangent)
    set_tests_properties(nevpt2_sc_pseudodensity_cas44_casscf_df_sc_source_tangent PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_sc_source_tangent_pass} \\(density-fitted\\)")
  endif()

  # --- PC's mixed orbital-CI block H_kc, --pc --sc-hessian-kc (#201) --------------
  #
  # The same binary and flag with --pc (apps/sc_pseudodensity/
  # hessian_kc_pc_check.cppm; docs/hessian-theory.md, Sec. 6.4-6.8): PC's
  # Eddot_kc(w, u) and H_kc^T w (the three-term y differentiated along w),
  # held to each other by DUALITY (1e-7), with PC's first-order source term
  # along w re-held and both walks' own Edot(w) against it, Euler in c on PC's
  # five homogeneous classes against the gated Y_pc, the intra-space zeros of
  # the one rotation term, Richardson-verified finite differences of the
  # three-term y PER CLASS (core-core and virtual-virtual with the orbital
  # energies responding to kappa; active-active frozen, where the rotation
  # term lives), the step-free active-rotation gauge against pcHessianDevice,
  # and the refusals (a lifting u, Numerical; the orbital energies responding
  # to c, Unsupported; no sidecar with a w leaving the active space or a
  # responding mode, Unsupported). The state is held to block2's PC total
  # first. CAS(4,4), conventional and density-fitted -- the fitted one resting
  # on --pc --sc-hessian-kk-int's gate of PC's first-order source term over
  # the fitted inventory (#200) -- both `small`. CAS(8,8) has no entry: it is
  # recorded as a diagnostic, as #152's and #155's are, until #194 (docs/
  # pc-nevpt2.md, "The mixed block H_kc"; docs/testing.md, "PC's mixed Hessian
  # block's checks").
  set(_nevpt2_pc_hessian_kc_pass
      "PASS: the PC mixed Hessian block H_kc's two orientations agree by duality")
  foreach(_case cas44 cas44_df)
    set(_pkc_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.nevpt2gold")
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.gradgold")
      message(FATAL_ERROR "nevpt2: committed gradient sidecar missing for n2_ccpvdz_${_case}")
    endif()
    _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_${_case}_pc_hessian_kc
                          "${_pkc_gold}" --tiles 3 --pc --sc-hessian-kc)
    if(TEST nevpt2_sc_pseudodensity_${_case}_pc_hessian_kc)
      set_tests_properties(nevpt2_sc_pseudodensity_${_case}_pc_hessian_kc PROPERTIES
        PASS_REGULAR_EXPRESSION "${_nevpt2_pc_hessian_kc_pass}" LABELS "gpu;small")
    endif()
    unset(_pkc_gold)
  endforeach()
  if(TEST nevpt2_sc_pseudodensity_cas44_df_pc_hessian_kc)
    set_tests_properties(nevpt2_sc_pseudodensity_cas44_df_pc_hessian_kc PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_pc_hessian_kc_pass}.* \\(density-fitted\\)")
  endif()

  # --- the orbital block's integral curvature, --sc-hessian-kk-int (#154) --------
  #
  # nevpt2_sc_pseudodensity --sc-hessian-kk-int (apps/sc_pseudodensity/
  # hessian_kk_check.cppm; docs/hessian-theory.md, Sec. 5.1 and Sec. 5.4): the
  # first term of the SC energy's orbital Hessian, <Ehat', I''[u, v]> = K1 + K2
  # -- K1 the first-order pairing at W = (uv + vu)/2 with the three traced-pair
  # promotions restored, K2 the cross-slot term (from the sidecar when one of
  # u, v is active-active; on a fitted case from the three-index B for any
  # pair). Each term's adjoint against its forward tangent (1e-7, the digests'
  # term in both modes), the swap, K1's matrix path against the first-order
  # pairing at an antisymmetric kappa, the traced pairs' zero, the intra-space
  # exact zeros, Richardson-verified ladders of the fixed-pseudodensity
  # pairing along exp(s u + t v) for every block-diagonal pair, the product
  # parameterisation's (1/2) <[u, v], Y> offset, and the refusals (a
  # conventional pair with neither member active-active is Unsupported, never
  # answered with K1 alone). The fitted case adds full x full and inactive
  # pairs and holds the two routes to each other. CAS(4,4) -- both flavours --
  # is `small`, so the sanitizer tier runs the device work; CAS(8,8)'s
  # forward digests are a plain RDM build per term per pair, which keeps it
  # out (docs/testing.md, "The orbital block's integral curvature").
  set(_nevpt2_sc_hessian_kk_pass
      "PASS: the SC orbital Hessian's integral curvature <Ehat', I''> = K1 \\+ K2 agrees")
  foreach(_case cas44 cas88 cas44_df)
    set(_kk_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.nevpt2gold")
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.gradgold")
      message(FATAL_ERROR "nevpt2: committed gradient sidecar missing for n2_ccpvdz_${_case}")
    endif()
    _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_${_case}_sc_hessian_kk_int
                          "${_kk_gold}" --tiles 3 --sc-hessian-kk-int)
    if(TEST nevpt2_sc_pseudodensity_${_case}_sc_hessian_kk_int)
      set_tests_properties(nevpt2_sc_pseudodensity_${_case}_sc_hessian_kk_int PROPERTIES
        PASS_REGULAR_EXPRESSION "${_nevpt2_sc_hessian_kk_pass}")
    endif()
    unset(_kk_gold)
  endforeach()
  foreach(_case cas44 cas44_df)
    if(TEST nevpt2_sc_pseudodensity_${_case}_sc_hessian_kk_int)
      set_tests_properties(nevpt2_sc_pseudodensity_${_case}_sc_hessian_kk_int PROPERTIES
        LABELS "gpu;small")
    endif()
  endforeach()
  # The fitted case's PASS line names its route.
  if(TEST nevpt2_sc_pseudodensity_cas44_df_sc_hessian_kk_int)
    set_tests_properties(nevpt2_sc_pseudodensity_cas44_df_sc_hessian_kk_int PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_sc_hessian_kk_pass} .*\\(density-fitted: K2 by the three-index jets, both routes agree\\)")
  endif()

  # --- PC's orbital curvature, --pc --sc-hessian-kk-int (#200) -------------------
  #
  # The same binary and checks with --pc (apps/sc_pseudodensity/
  # hessian_kk_check.cppm; docs/hessian-theory.md, Sec. 6.7): PC-NEVPT2's
  # <Ehat'_pc, I''[u, v]> = K1 + K2 (pcOrbitalCurvatureDevice) -- #154's input
  # generators against PC's integral adjoint (#74's dE/dslab, dE/dK, R_t and
  # the digests' hbar_pc). First, check 0 gates PC's first-order orbital source
  # term itself on seeded full kappas, the integral half's duality and the full
  # one with the digests' forward term (1e-7) -- on a fitted case the only gate
  # of it in tree, which the curvature there rests on. Then every SC check:
  # duality per term, the swap, K1's matrix path at an antisymmetric kappa,
  # the traced pairs' zero (h1e_v_Sr's alone -- PC reads no v2tr), the
  # intra-space zeros, the Richardson-verified ladders of the fixed-adjoint
  # pairing along exp(s u + t v) (the term paired in the ladder's own walk),
  # the product parameterisation, the routes on a fitted case and the
  # refusals. The state is held to block2's PC total first. CAS(4,4), both
  # flavours, is `small`; CAS(8,8) runs too, both flavours, outside it, for
  # the same reason as SC's (docs/testing.md, "PC's orbital curvature").
  set(_nevpt2_pc_hessian_kk_pass
      "PASS: the PC orbital Hessian's integral curvature <Ehat', I''> = K1 \\+ K2 agrees .*\\(PC's first-order source term gated first, both dualities\\)")
  foreach(_case cas44 cas88 cas44_df cas88_df)
    set(_pkk_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.nevpt2gold")
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.gradgold")
      message(FATAL_ERROR "nevpt2: committed gradient sidecar missing for n2_ccpvdz_${_case}")
    endif()
    _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_${_case}_pc_hessian_kk_int
                          "${_pkk_gold}" --tiles 3 --pc --sc-hessian-kk-int)
    if(TEST nevpt2_sc_pseudodensity_${_case}_pc_hessian_kk_int)
      set_tests_properties(nevpt2_sc_pseudodensity_${_case}_pc_hessian_kk_int PROPERTIES
        PASS_REGULAR_EXPRESSION "${_nevpt2_pc_hessian_kk_pass}")
    endif()
    unset(_pkk_gold)
  endforeach()
  foreach(_case cas44 cas44_df)
    if(TEST nevpt2_sc_pseudodensity_${_case}_pc_hessian_kk_int)
      set_tests_properties(nevpt2_sc_pseudodensity_${_case}_pc_hessian_kk_int PROPERTIES
        LABELS "gpu;small")
    endif()
  endforeach()
  foreach(_case cas44_df cas88_df)
    if(TEST nevpt2_sc_pseudodensity_${_case}_pc_hessian_kk_int)
      set_tests_properties(nevpt2_sc_pseudodensity_${_case}_pc_hessian_kk_int PROPERTIES
        PASS_REGULAR_EXPRESSION "${_nevpt2_pc_hessian_kk_pass}.* \\(density-fitted: K2 by the three-index jets, both routes agree\\)")
    endif()
  endforeach()

  # --- the orbital block's sandwich and Eddot_kk, --sc-hessian-kk (#158, #190) ----
  #
  # nevpt2_sc_pseudodensity --sc-hessian-kk (apps/sc_pseudodensity/
  # hessian_kk_sandwich_check.cppm; docs/hessian-theory.md, Sec. 5.1 and Sec.
  # 5.6): the second term of the SC energy's orbital Hessian, the sandwich
  # I'^T Ehat'' I' = K3 + K4 + K5 + K6, forward against reverse (1e-7), the
  # per-tuple merge of the digests' share against the first-order <k, Y>, the
  # intra-space exact zeros, the digests' curvature against the energy
  # stage's own Hessian at the plain digest path's tangents, and
  # Richardson-extrapolated ladders of the SC total along exp(s u + t v)
  # against Eddot_kk = <Ehat', I''> + the sandwich (and of E - L against the
  # sandwich alone), both held to 1e-6 relative -- core x core, core x
  # virtual and the proportional virtual x (2 virtual + core) included, which
  # K6 closes. A pair along which the energy has no Hessian (an H class's
  # empty perturbers with non-proportional tables, rho_t above kRhoTol) is
  # refused as Numerical and the PASS line counts them; on N2 those are
  # exactly full x full and virtual x virtual. CAS(4,4)
  # -- both flavours -- is `small`, so the sanitizer tier runs the device work;
  # CAS(8,8)'s ladders are a plain RDM build and an energy per corner, which
  # keeps it out (docs/testing.md, "The orbital block's sandwich").
  set(_nevpt2_sc_hessian_kk_full_pass
      "PASS: the SC orbital Hessian's sandwich I'\\^T Ehat'' I' = K3 \\+ K4 \\+ K5 \\+ K6 agrees with itself by duality .* 2 pairs refused, with no Hessian along them \\(rho_t, Sec. 5.6\\)")
  foreach(_case cas44 cas88 cas44_df)
    set(_kk_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.nevpt2gold")
    _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_${_case}_sc_hessian_kk
                          "${_kk_gold}" --tiles 3 --sc-hessian-kk)
    if(TEST nevpt2_sc_pseudodensity_${_case}_sc_hessian_kk)
      set_tests_properties(nevpt2_sc_pseudodensity_${_case}_sc_hessian_kk PROPERTIES
        PASS_REGULAR_EXPRESSION "${_nevpt2_sc_hessian_kk_full_pass}")
    endif()
    unset(_kk_gold)
  endforeach()
  foreach(_case cas44 cas44_df)
    if(TEST nevpt2_sc_pseudodensity_${_case}_sc_hessian_kk)
      set_tests_properties(nevpt2_sc_pseudodensity_${_case}_sc_hessian_kk PROPERTIES
        LABELS "gpu;small")
    endif()
  endforeach()
  if(TEST nevpt2_sc_pseudodensity_cas44_df_sc_hessian_kk)
    set_tests_properties(nevpt2_sc_pseudodensity_cas44_df_sc_hessian_kk PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_sc_hessian_kk_full_pass} \\(density-fitted\\)")
  endif()

  # --- PC's orbital sandwich and Eddot_kk, --pc --sc-hessian-kk (#202) -----------
  #
  # The same binary and flag with --pc (apps/sc_pseudodensity/
  # hessian_kk_pc_check.cppm; docs/hessian-theory.md, Sec. 6.4-6.8): PC's
  # sandwich I'^T Ehat''_pc I' -- the solve's two-leg cross term per tuple, S
  # fixed along kappa so no rotation term, no guarded tuple -- forward against
  # reverse (1e-7) with the orbital energies responding to kappa and frozen,
  # each walk's own Edot against the gated first-order Y_pc, the intra-space
  # exact zeros, the digests' share against pcHessianDevice at the plain
  # digest path's tangents, Richardson-verified ladders of the PC total along
  # exp(s u + t v) against Eddot_kk = <Ehat'_pc, I''> + the sandwich (and of
  # E - L against the sandwich) at --sc-hessian-kk's gate and floor-aware rule
  # -- virtual x virtual included, which SC refuses -- the step-free gauge
  # Eddot_kk(v, w) + (1/2) <[v, w], Y_pc> == Eddot_kc(v, what c) against #201's
  # H_kc, and the refusal of a conventional pair with no K2. The state is held
  # to block2's PC total first. CAS(4,4), conventional and density-fitted --
  # the fitted one resting on --pc --sc-hessian-kk-int's gate of PC's
  # first-order source term over the fitted inventory (#200) -- both `small`.
  # CAS(8,8) has no entry: it is recorded as a diagnostic, as #152's, #155's
  # and #201's are, until #194 (docs/pc-nevpt2.md, "The orbital block's
  # sandwich"; docs/testing.md, "PC's orbital block's sandwich").
  set(_nevpt2_pc_hessian_kk_full_pass
      "PASS: the PC orbital Hessian's sandwich I'\\^T Ehat''_pc I' agrees with itself by duality to 1e-07 in both orbital-energy modes.* no pair refused for its empty perturbers \\(Sec. 6.6\\)")
  foreach(_case cas44 cas44_df)
    set(_pkkf_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.nevpt2gold")
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.gradgold")
      message(FATAL_ERROR "nevpt2: committed gradient sidecar missing for n2_ccpvdz_${_case}")
    endif()
    _add_golden_test_with(nevpt2_sc_pseudodensity nevpt2_sc_pseudodensity_${_case}_pc_hessian_kk
                          "${_pkkf_gold}" --tiles 3 --pc --sc-hessian-kk)
    if(TEST nevpt2_sc_pseudodensity_${_case}_pc_hessian_kk)
      set_tests_properties(nevpt2_sc_pseudodensity_${_case}_pc_hessian_kk PROPERTIES
        PASS_REGULAR_EXPRESSION "${_nevpt2_pc_hessian_kk_full_pass}" LABELS "gpu;small")
    endif()
    unset(_pkkf_gold)
  endforeach()
  if(TEST nevpt2_sc_pseudodensity_cas44_df_pc_hessian_kk)
    set_tests_properties(nevpt2_sc_pseudodensity_cas44_df_pc_hessian_kk PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_pc_hessian_kk_full_pass} \\(density-fitted\\)")
  endif()
  unset(_df_y_gold)
  unset(_df_y_side)

  # --- *_pc: PC-NEVPT2 ----------------------------------------------
  #
  # The same RDM build as the SC entry at the same --tiles; the PC stage runs
  # after the tile blocks are freed and does not raise the pool's used
  # high-water (docs/pc-nevpt2.md, "Cost"). The 12-orbital PC case is
  # CAS(10,12), not CAS(12,12): the cas1212 goldens carry no PC fields (the
  # generator refuses their ungapped metric). Its measured --tiles floor is 14
  # on the RTX 3080 and 7 on the RX 9060 XT (for --pc and SC alike); 40 is
  # the count PC was measured with, the same as the CAS(12,12) entries'.
  set(_gold_1012 "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas1012.nevpt2gold")
  add_pc_golden_test(nevpt2_cas1010_pc "${_gold_1010}" --tiles 3)
  add_pc_golden_test(nevpt2_cas1012_pc "${_gold_1012}" --tiles 40)
  if(TEST nevpt2_cas1012_pc)
    set_tests_properties(nevpt2_cas1012_pc PROPERTIES LABELS "gpu;slow")
  endif()

  # --- density-fitted ---------------------------------------------------------
  #
  # The *_df golden files are `generate_golden.py --df`: the whole calculation
  # density-fitted, the answer PySCF's DF-NEVPT2. They are a DIFFERENT
  # reference from the conventional files above (different integrals, so a
  # different CASCI state and E_corr, ~5e-5 Eh apart at CAS(10,10)) -- each
  # binary is only ever checked against a file of its own kind.
  set(_gold_1010_df "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas1010_df.nevpt2gold")
  set(_gold_1212_df "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas1212_df.nevpt2gold")
  set(_gold_tz1010_df "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvtz_cas1010_df.nevpt2gold")

  # A DF file also carries the four-index blocks rebuilt from its B tensors,
  # so the UNMODIFIED integral-direct binary must pass on it too: this checks
  # the DF golden file is self-consistent, independently of any DF code.
  add_golden_test(nevpt2_cas1010_df_blocks "${_gold_1010_df}" --tiles 3)

  # The DF binary, from B alone. --check-blocks also compares every DF-built
  # block against the file's four-index arrays (1e-10). The default --batch 8
  # splits every virtual-indexed block into two slabs here (nvirt=16).
  add_df_golden_test(nevpt2_df_cas1010 "${_gold_1010_df}" --tiles 3 --check-blocks)

  # cc-pVTZ: nvirt=48, so --batch 5 leaves a ragged last slab (48 = 9*5 + 3)
  # -- the slab boundaries are what this case is for, not memory pressure
  # (every external block here is < 2 MB).
  add_df_golden_test(nevpt2_df_ccpvtz_cas1010 "${_gold_tz1010_df}"
                     --tiles 3 --batch 5 --check-blocks)

  # The same RDM build as nevpt2_cas1212, so the same measured tile floor (17)
  # and the same --tiles 40.
  add_df_golden_test(nevpt2_df_cas1212 "${_gold_1212_df}" --tiles 40)
  if(TEST nevpt2_df_cas1212)
    set_tests_properties(nevpt2_df_cas1212 PROPERTIES LABELS "gpu;slow")
  endif()

  if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
    add_df_golden_test(nevpt2_df_cas1010_cublas "${_gold_1010_df}" --tiles 3 --cublas)
  endif()

  # --- *_pc, density-fitted -------------------------------------------
  #
  # The DF goldens that carry PC fields at these sizes; n2_ccpvdz_cas1212_df
  # has none. Same --tiles/--batch as the SC entries above (cc-pVTZ's --batch 5
  # for the ragged last slab), so the PC slab walk crosses the same boundaries.
  add_df_pc_golden_test(nevpt2_df_cas1010_pc "${_gold_1010_df}" --tiles 3)
  add_df_pc_golden_test(nevpt2_df_ccpvtz_cas1010_pc "${_gold_tz1010_df}"
                        --tiles 3 --batch 5)

  # --- --gradient-source / --sc-tangent, density-fitted (#162) --------------------
  #
  # The integral-direct demo's two checks of the energy's CI-vector derivative
  # (nevpt2.ci_checks, src/gradient/ci_checks_*.cpp), run by nevpt2_df_demo
  # over its DfIntegralSource: y_I = dE/dc_I against the forward tangent by
  # the duality <u, y> = Edot(u) -- four seeded directions, the state itself
  # and, at CAS(4,4), the full Jacobian -- and the SC tangent against Euler,
  # the norms' exact affine increment and its one-time finite difference. Both
  # differentiate the CI vector at FIXED integrals, so the B tensors are never
  # differentiated; the default --batch 8 leaves a ragged last slab (19
  # virtuals = 8 + 8 + 3 at CAS(4,4), 17 = 8 + 8 + 1 at CAS(8,8)), so the
  # gradient walk crosses owning, freed-per-slab DF slabs, which no
  # integral-direct entry does. FROZEN orbital energies: this demo reads no
  # gradient sidecar, and the --gradient-source regex requires the PASS line
  # to say so. Same tolerances, same --tiles as the integral-direct twins
  # below; the same `small` split (CAS(8,8)'s --sc-tangent is not).
  set(_nevpt2_df_ci_src_pass
      "PASS: the CI source term and the forward tangent agree .*, the orbital energies held fixed \\(frozen-orbital-energy mode")
  foreach(_case cas44 cas88)
    set(_df_ci_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}_df.nevpt2gold")
    add_df_golden_test(nevpt2_df_gradient_source_${_case} "${_df_ci_gold}"
                       --tiles 3 --gradient-source)
    if(TEST nevpt2_df_gradient_source_${_case})
      set_tests_properties(nevpt2_df_gradient_source_${_case} PROPERTIES
        PASS_REGULAR_EXPRESSION "${_nevpt2_df_ci_src_pass}" LABELS "gpu;small")
    endif()
    add_df_golden_test(nevpt2_df_${_case}_sc_tangent "${_df_ci_gold}" --tiles 3 --sc-tangent)
    if(TEST nevpt2_df_${_case}_sc_tangent)
      set_tests_properties(nevpt2_df_${_case}_sc_tangent PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: the SC tangent matches Euler")
      if(_case STREQUAL "cas44")
        set_tests_properties(nevpt2_df_${_case}_sc_tangent PROPERTIES LABELS "gpu;small")
      endif()
    endif()
    unset(_df_ci_gold)
  endforeach()

  # --- --consume-emitted -------------------------------------------------------
  #
  # The f3 consume step runs through wwr.blas by default; the
  # generated loop kernels stay reachable for the A/B, and these entries keep
  # them PASSing on both backends (and next to the --cublas digest on CUDA).
  add_golden_test(nevpt2_cas1010_consume_emitted "${_gold_1010}"
                  --tiles 3 --consume-emitted)
  if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
    add_golden_test(nevpt2_cas1010_cublas_consume_emitted "${_gold_1010}"
                    --tiles 3 --cublas --consume-emitted)
  endif()

  # --- --blas-digest / --digest-emitted ----------------------------------------
  #
  # The --cublas digest's GEMM shape as plain native-fp64 wwrblasDgemm (cuBLAS
  # / rocBLAS), so -- unlike --cublas -- on both backends. The
  # default digest differs by backend because the A/B did (docs/performance.md,
  # "Native-fp64 BLAS digest"): BLAS on HIP, the emitted kernels on CUDA. So
  # each backend's entries run the digest that is NOT its default -- every
  # other entry above already runs the default one.
  if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
    add_golden_test(nevpt2_cas1010_blas_digest "${_gold_1010}" --tiles 3 --blas-digest)
    add_df_golden_test(nevpt2_df_cas1010_blas_digest "${_gold_1010_df}"
                       --tiles 3 --blas-digest)
  else()
    add_golden_test(nevpt2_cas1010_digest_emitted "${_gold_1010}"
                    --tiles 3 --digest-emitted)
    add_df_golden_test(nevpt2_df_cas1010_digest_emitted "${_gold_1010_df}"
                       --tiles 3 --digest-emitted)
  endif()

  # --- --ozaki --------------------------------------------------------------------
  #
  # The three digest GEMMs on int8 tensor cores by the Ozaki scheme
  # (src/ozaki/), through WarpWraps' <wmma.h>, so on both backends. All 64
  # digit pairs (the default --ozaki-pairs 14). The cc-pVTZ --pc entry is the
  # gate for any lossy digest: the tightest PC metric gap of the committed
  # goldens, where a truncated digest (--ozaki-pairs 5) gets Sr/Si refused
  # rather than drifting (docs/performance.md, "The int8 Ozaki
  # digest"). The small entries put the kernel under the sanitizer tier.
  add_golden_test(nevpt2_cas1010_ozaki "${_gold_1010}" --tiles 3 --ozaki)
  add_df_pc_golden_test(nevpt2_df_ccpvtz_cas1010_pc_ozaki "${_gold_tz1010_df}"
                        --tiles 3 --batch 5 --ozaki)

  # --- --profile -----------------------------------------------------------------
  #
  # --profile brackets every RDM-build launch and GEMM, every energy-stage
  # einsum and (df demo) every slab DGEMM with an event pair (nevpt2.profile).
  # These entries keep the
  # instrumented paths exercised and still PASS, i.e. the wrapping did not
  # reorder or drop a launch. The tables themselves are diagnostic output, not
  # asserted on.
  add_golden_test(nevpt2_cas1010_profile "${_gold_1010}" --tiles 3 --profile)
  add_df_golden_test(nevpt2_df_cas1010_profile "${_gold_1010_df}" --tiles 3 --profile)
  if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
    add_golden_test(nevpt2_cas1010_cublas_profile "${_gold_1010}"
                    --tiles 3 --cublas --profile)
  endif()

  # --- small active spaces: CAS(4,4) and CAS(8,8) --------------------------------
  #
  # The cases the sanitizer tier runs: compute-sanitizer multiplies
  # a run's time, so its test preset selects ONLY the `small` label (plus the
  # canaries). They are ordinary golden checks too -- `ctest --preset fast`
  # runs them on every backend. Same molecule and basis as the production
  # files (N2/cc-pVDZ), from the same generate_golden.py settings;
  # docs/testing.md ("The small cases") has why these two sizes.
  #
  # Every RDM-build and energy path is covered at least once: the default
  # digest and consume, the other digest (--blas-digest on CUDA,
  # --digest-emitted on HIP), --consume-emitted, --cublas (CUDA; engaged
  # bits=53 at CAS(8,8)), the --profile events, and density fitting with
  # --check-blocks. --tiles 3 is NOT a measured floor here (one tile fits
  # easily): it is there so the tile loop and the device-resident accumulate
  # run more than once. Density fitting's default --batch 8 leaves a ragged
  # last slab on both: 19 virtuals = 8 + 8 + 3 at CAS(4,4), 17 = 8 + 8 + 1 at
  # CAS(8,8).
  set(_gold_44 "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44.nevpt2gold")
  set(_gold_88 "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas88.nevpt2gold")
  set(_gold_44_df "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44_df.nevpt2gold")
  set(_gold_88_df "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas88_df.nevpt2gold")

  add_golden_test(nevpt2_cas44 "${_gold_44}" --tiles 3)
  add_golden_test(nevpt2_cas88 "${_gold_88}" --tiles 3)
  add_golden_test(nevpt2_cas88_consume_emitted "${_gold_88}" --tiles 3 --consume-emitted)
  add_golden_test(nevpt2_cas88_profile "${_gold_88}" --tiles 3 --profile)
  if(NEVPT2_GPU_BACKEND STREQUAL "CUDA")
    add_golden_test(nevpt2_cas88_blas_digest "${_gold_88}" --tiles 3 --blas-digest)
    add_golden_test(nevpt2_cas88_cublas "${_gold_88}" --tiles 3 --cublas)
  else()
    add_golden_test(nevpt2_cas88_digest_emitted "${_gold_88}" --tiles 3 --digest-emitted)
  endif()
  add_df_golden_test(nevpt2_df_cas44 "${_gold_44_df}" --tiles 3 --check-blocks)
  add_df_golden_test(nevpt2_df_cas88 "${_gold_88_df}" --tiles 3 --check-blocks)
  add_df_golden_test(nevpt2_df_cas88_profile "${_gold_88_df}" --tiles 3 --profile)
  add_golden_test(nevpt2_cas88_ozaki "${_gold_88}" --tiles 3 --ozaki)
  add_df_pc_golden_test(nevpt2_df_cas88_pc_ozaki "${_gold_88_df}" --tiles 3 --ozaki)
  foreach(_t nevpt2_cas44 nevpt2_cas88 nevpt2_cas88_consume_emitted nevpt2_cas88_profile
             nevpt2_cas88_blas_digest nevpt2_cas88_cublas nevpt2_cas88_digest_emitted
             nevpt2_df_cas44 nevpt2_df_cas88 nevpt2_df_cas88_profile
             nevpt2_cas88_ozaki nevpt2_df_cas88_pc_ozaki)
    if(TEST ${_t})
      set_tests_properties(${_t} PROPERTIES LABELS "gpu;small")
    endif()
  endforeach()

  # --- the density-fitted gradient sidecar, read end to end (#164) --------------
  #
  # golden/n2_ccpvdz_cas44_df.gradgold is the NEVPT2F1 sidecar beside the DF
  # golden (docs/reference-data.md, "The density-fitted gradient sidecar"). Its
  # four tied orbital-energy arrays and h^eff are what --gradient-source's
  # eps-response mode reads, so this runs that mode on the DF case: the
  # unmodified nevpt2_demo (which runs a *_df golden off its four-index arrays,
  # as nevpt2_cas1010_df_blocks does) loads the fitted sidecar, checks it against
  # the fitted golden (flavour, naux, and e_core / e_virt rebuilt from it), and
  # gates on the same exact forward-reverse duality as nevpt2_gradient_source_*.
  # The regex requires the response to be carried. nevpt2_df_demo's own
  # --gradient-source is #162.
  set(_ci_src_df_side "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44_df.gradgold")
  if(NOT EXISTS "${_ci_src_df_side}")
    message(FATAL_ERROR "nevpt2: committed gradient sidecar missing: ${_ci_src_df_side}")
  endif()
  _add_golden_test_with(nevpt2_demo nevpt2_gradient_source_cas44_df "${_gold_44_df}"
                        --tiles 3 --gradient-source --gradgold "${_ci_src_df_side}")
  if(TEST nevpt2_gradient_source_cas44_df)
    set_tests_properties(nevpt2_gradient_source_cas44_df PROPERTIES
      PASS_REGULAR_EXPRESSION
        "PASS: the CI source term and the forward tangent agree .*, the orbital energies moving with the active density"
      LABELS "gpu;small")
  endif()
  unset(_ci_src_df_side)

  # --- --rdm-tangent, small ------------------------------------------------------
  #
  # The tangent RDM build (nevpt2.rdm_build's :tangent partition) beside the
  # plain one, checked against twice it -- Euler's identity, since every output
  # is quadratic in the CI vector. The entry still PASSes on the SC energies,
  # so a broken tangent build shows up as the demo's nonzero exit (the
  # self-check reports a Numerical Error) rather than as a changed number.
  # `small`, so the tangent build's allocations and launches run under the
  # sanitizer tier; the derivative itself is checked against a finite
  # difference in the unit tier (test/rdm/rdm_tangent_tests.cpp). The tangent
  # build holds about twice the per-tile intermediates, so its tile floors are
  # its own (docs/performance.md, "Tile floors"); --tiles 3 is far above them
  # at CAS(8,8).
  add_golden_test(nevpt2_cas88_rdm_tangent "${_gold_88}" --tiles 3 --rdm-tangent)
  if(TEST nevpt2_cas88_rdm_tangent)
    set_tests_properties(nevpt2_cas88_rdm_tangent PROPERTIES LABELS "gpu;small")
  endif()

  # --- --sc-tangent, small -------------------------------------------------------
  #
  # The SC class energies' forward tangent with respect to the RDMs
  # (nevpt2.energy's :sc_tangent partition, energiesTangentDevice), checked
  # three ways by src/gradient/ci_checks_tangent.cpp: the class norms'
  # exact affine increment (every class), Euler's identity for the three
  # classes whose blocks are purely linear in the RDMs, and a central finite
  # difference of energiesDevice along a seeded CI direction, Richardson
  # verified. THE SC ENERGY STAGE'S OWN FINITE DIFFERENCE, and a one-time one
  # (docs/gradients.md, Sec. 1.2, lists the tier's others): it
  # licenses the exact dot-product tests the adjoint will be held to
  # (docs/testing.md, "The SC tangent's finite-difference cross-check", has
  # the measured plateau and the h where round-off takes over). Its PASS line
  # is its own, so neither entry can pass on the golden comparison's -- which
  # --sc-tangent does not run.
  #
  # The sweep reruns the whole RDM build and energy stage at 28 displaced CI
  # vectors, which is still seconds at these sizes (CAS(8,8): ~4 s on an RTX
  # 3080). CAS(4,4) is `small` as well, so the tangent's own einsums run under
  # the sanitizer tier; CAS(8,8) is not, to keep that tier's 28 extra builds
  # down to the smallest case.
  set(_nevpt2_sc_tangent_pass "PASS: the SC tangent matches Euler")
  add_golden_test(nevpt2_cas44_sc_tangent "${_gold_44}" --tiles 3 --sc-tangent)
  add_golden_test(nevpt2_cas88_sc_tangent "${_gold_88}" --tiles 3 --sc-tangent)
  if(TEST nevpt2_cas44_sc_tangent)
    set_tests_properties(nevpt2_cas44_sc_tangent PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_sc_tangent_pass}" LABELS "gpu;small")
  endif()
  if(TEST nevpt2_cas88_sc_tangent)
    set_tests_properties(nevpt2_cas88_sc_tangent PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_sc_tangent_pass}" LABELS "gpu")
  endif()
  # The wave-0 canary (#118): the same tangent on CASSCF ORBITALS. --sc-tangent
  # sits behind the source-term gate, which takes a CASSCF golden, and its
  # checks -- Euler, the exact norm increment, the finite difference -- are of
  # the same functional either way (docs/testing.md, "The source terms on
  # CASSCF orbitals: the wave-0 canary"). `small`, like the CAS(4,4) twin.
  add_golden_test(nevpt2_cas44_casscf_sc_tangent
                  "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44_casscf.nevpt2gold"
                  --tiles 3 --sc-tangent)
  if(TEST nevpt2_cas44_casscf_sc_tangent)
    set_tests_properties(nevpt2_cas44_casscf_sc_tangent PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_sc_tangent_pass}" LABELS "gpu;small")
  endif()

  # --- --sc-hessian (#153) --------------------------------------------------------
  #
  # The SC energy's SECOND directional derivative Eddot(u, v) (nevpt2.energy's
  # :sc_hessian partition, energiesHessianDevice, over nevpt2.rdm_build's
  # three-argument tangent build), checked by src/gradient/ci_checks_tangent.cpp's
  # checkScHessian at the golden's fixed orbital energies: Euler at second
  # order, Eddot(c, u) == Edot(u), on the three homogeneous classes; the norms'
  # exact increment at the second-order RDMs (every class); and ONE
  # Richardson-verified central difference of Edot along v -- the Hessian's one
  # finite difference (docs/hessian-theory.md, Sec. 9; docs/testing.md, "The SC
  # Hessian's finite-difference cross-check"). Its own PASS line, as
  # --sc-tangent's. Each sweep point is a plain AND a tangent build, so the
  # same `small` split as --sc-tangent: CAS(4,4) under the sanitizer tier,
  # CAS(8,8) not. --tiles 3 is far above the three-argument build's floor at
  # both sizes (docs/performance.md, "Tile floors").
  set(_nevpt2_sc_hessian_pass "PASS: the SC Hessian matches Euler")
  add_golden_test(nevpt2_cas44_sc_hessian "${_gold_44}" --tiles 3 --sc-hessian)
  add_golden_test(nevpt2_cas88_sc_hessian "${_gold_88}" --tiles 3 --sc-hessian)
  if(TEST nevpt2_cas44_sc_hessian)
    set_tests_properties(nevpt2_cas44_sc_hessian PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_sc_hessian_pass}" LABELS "gpu;small")
  endif()
  if(TEST nevpt2_cas88_sc_hessian)
    set_tests_properties(nevpt2_cas88_sc_hessian PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_sc_hessian_pass}" LABELS "gpu")
  endif()

  # --- --sc-hessian-vector (#156) -------------------------------------------------
  #
  # The VECTOR H_cc u by reverse over forward (nevpt2.gradient's
  # scCiHessianVectorDevice: the pseudodensity walk differentiated along u,
  # then sigma(gdot, c) + sigma(g, u)), checked by
  # src/gradient/ci_checks_tangent.cpp's checkScHessianVector at the golden's
  # fixed orbital energies: the exact duality <v, H_cc u> == Eddot(u, v)
  # against --sc-hessian's energiesHessianDevice; at CAS(4,4) (ndet = 36) H_cc
  # materialised column by column against Eddot(e_I, v), and its symmetry; and
  # ONE Richardson-verified central difference of the GPU's own y, no
  # orbital-energy response on either leg (docs/hessian-theory.md, Sec. 4.2;
  # docs/testing.md, "The SC Hessian-vector product's checks"). The same
  # `small` split and --tiles 3 as --sc-hessian, whose three-argument build
  # the duality runs.
  set(_nevpt2_sc_hessian_vector_pass "PASS: the SC Hessian-vector product matches")
  add_golden_test(nevpt2_cas44_sc_hessian_vector "${_gold_44}" --tiles 3 --sc-hessian-vector)
  add_golden_test(nevpt2_cas88_sc_hessian_vector "${_gold_88}" --tiles 3 --sc-hessian-vector)
  if(TEST nevpt2_cas44_sc_hessian_vector)
    set_tests_properties(nevpt2_cas44_sc_hessian_vector PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_sc_hessian_vector_pass}" LABELS "gpu;small")
  endif()
  if(TEST nevpt2_cas88_sc_hessian_vector)
    set_tests_properties(nevpt2_cas88_sc_hessian_vector PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_sc_hessian_vector_pass}" LABELS "gpu")
  endif()

  # --- the CI block in FULL mode, --gradgold beside the H_cc flags (#214) ----------
  #
  # The same two checks with the committed gradient sidecar beside the golden:
  # the orbital energies respond to the CI vector (e = h^eff + <dm1, W>, W read
  # by the --gradient-source reader) on every side of every check, every
  # displaced state at its own orbital energies, and Euler -- not an identity
  # there -- skipped and said to be (docs/hessian-theory.md, Sec. 4.3 and
  # 9.1). --sc-hessian keeps its norm increment and its difference of Edot;
  # --sc-hessian-vector its duality against the FULL Eddot, the columns and a
  # Richardson ladder of the gated y. Each PASS line names the mode, which the
  # regex holds it to. CAS(4,4) (its CASSCF twin is with the #157 entries
  # below), `small`, at --sc-hessian's --tiles 3.
  set(_nevpt2_sc_hessian_full_pass
      "PASS: the SC Hessian matches the exact norm increment .*\\(orbital-energy mode: FULL, the orbital energies responding to c, Euler skipped there\\)")
  set(_nevpt2_sc_hessian_vector_full_pass
      "${_nevpt2_sc_hessian_vector_pass} .*\\(orbital-energy mode: FULL, the orbital energies responding to c on both legs\\)")
  set(_side_44 "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44.gradgold")
  if(NOT EXISTS "${_side_44}")
    message(FATAL_ERROR "nevpt2: committed gradient sidecar missing: ${_side_44}")
  endif()
  add_golden_test(nevpt2_cas44_sc_hessian_full "${_gold_44}" --tiles 3 --sc-hessian
                  --gradgold "${_side_44}")
  add_golden_test(nevpt2_cas44_sc_hessian_vector_full "${_gold_44}" --tiles 3
                  --sc-hessian-vector --gradgold "${_side_44}")
  if(TEST nevpt2_cas44_sc_hessian_full)
    set_tests_properties(nevpt2_cas44_sc_hessian_full PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_sc_hessian_full_pass}" LABELS "gpu;small")
  endif()
  if(TEST nevpt2_cas44_sc_hessian_vector_full)
    set_tests_properties(nevpt2_cas44_sc_hessian_vector_full PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_sc_hessian_vector_full_pass}" LABELS "gpu;small")
  endif()

  # --- *_pc, small ----------------------------------------------------
  #
  # --pc at CAS(4,4)/CAS(8,8), both demos, labelled `small` so the sanitizer
  # tier runs the PC stage (its eigensolves on the solver handle, the BLAS
  # GEMMs, the slab einsums) too. All four goldens carry PC fields.
  add_pc_golden_test(nevpt2_cas44_pc "${_gold_44}" --tiles 3)
  add_pc_golden_test(nevpt2_cas88_pc "${_gold_88}" --tiles 3)
  add_df_pc_golden_test(nevpt2_df_cas44_pc "${_gold_44_df}" --tiles 3)
  add_df_pc_golden_test(nevpt2_df_cas88_pc "${_gold_88_df}" --tiles 3)
  foreach(_t nevpt2_cas44_pc nevpt2_cas88_pc nevpt2_df_cas44_pc nevpt2_df_cas88_pc)
    if(TEST ${_t})
      set_tests_properties(${_t} PROPERTIES LABELS "gpu;small")
    endif()
  endforeach()

  # --- CASSCF-orbital references -------------------------------------------------
  #
  # The same molecule, basis and active spaces as the CASCI entries above, on
  # converged CASSCF orbitals instead of RHF's: `generate_golden.py --casscf
  # --mo-out` commits the optimised orbitals to a `*.mogold` sidecar once, and
  # `--mo-from` runs the same deterministic CASCI on them, so no golden's state
  # depends on an optimiser (docs/reference-data.md, "The MO sidecar: committed
  # CASSCF orbitals"). These files carry `reference_kind = CASSCF` and a
  # DIFFERENT E_corr from their CASCI twins -- a different reference, not a
  # noisier one; how far apart is in docs/reference-data.md, "What 'CASSCF
  # input' means here".
  #
  # Nothing in src/ branches on the kind except the two gates in
  # nevpt2.golden: the SOURCE-TERM gate (--sc-tangent, --gradient-source,
  # --dump-source-terms, --source-oracle) accepts a CASSCF golden, and the
  # LAGRANGIAN gate a full gradient sits behind refuses one as Unsupported.
  # The two source-term canaries on n2_ccpvdz_cas44_casscf sit beside their
  # CASCI twins above (nevpt2_cas44_casscf_sc_tangent,
  # nevpt2_sc_pseudodensity_kappa_cas44_casscf); test/golden's
  # GoldenCommittedCasscfTests is the gates' split checked against one of
  # these committed files rather than a spliced one.
  # Every other path is the unmodified binary on the unmodified flags: a
  # CASSCF-orbital golden is checked the same way, to the same 1e-7.
  #
  # CAS(4,4), conventional and DF, is `small` like its CASCI twins, so the
  # sanitizer tier reads a NEVPT2G2 file too; CAS(10,10) is the production size
  # at the shipped --tiles 3. All three goldens carry PC fields -- their metric
  # gaps, like their CASCI twins', are decades inside the refusal
  # (docs/reference-data.md, "PC fields").
  set(_gold_44_casscf "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44_casscf.nevpt2gold")
  set(_gold_44_casscf_df
      "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44_casscf_df.nevpt2gold")
  set(_gold_1010_casscf "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas1010_casscf.nevpt2gold")

  add_golden_test(nevpt2_cas44_casscf "${_gold_44_casscf}" --tiles 3)
  add_pc_golden_test(nevpt2_cas44_casscf_pc "${_gold_44_casscf}" --tiles 3)
  add_df_golden_test(nevpt2_df_cas44_casscf "${_gold_44_casscf_df}" --tiles 3 --check-blocks)
  add_df_pc_golden_test(nevpt2_df_cas44_casscf_pc "${_gold_44_casscf_df}" --tiles 3)
  foreach(_t nevpt2_cas44_casscf nevpt2_cas44_casscf_pc
             nevpt2_df_cas44_casscf nevpt2_df_cas44_casscf_pc)
    if(TEST ${_t})
      set_tests_properties(${_t} PROPERTIES LABELS "gpu;small")
    endif()
  endforeach()

  add_golden_test(nevpt2_cas1010_casscf "${_gold_1010_casscf}" --tiles 3)
  add_pc_golden_test(nevpt2_cas1010_casscf_pc "${_gold_1010_casscf}" --tiles 3)

  # The C1 case (#213): HSOH CAS(6,6)/def2-SVP at its own CASSCF minimum, with
  # no point-group symmetry, so every Cartesian displacement keeps the group
  # (docs/reference-data.md, "HSOH, the C1 case"). CASSCF orbitals only -- it
  # has no CASCI twin. SC, PC (its metric is gapped decades inside the
  # refusal), and the CI source term's forward-reverse duality with the
  # orbital energies moving, read from the committed sidecar beside it (which
  # also carries the re-optimised states' grad_num; the binary reads only the
  # 24 --gradgold arrays). --tiles 3 is the count every entry here uses, not a
  # measured floor. Not `small`: the sanitizer tier stays on N2.
  set(_gold_hsoh66_casscf
      "${CMAKE_CURRENT_SOURCE_DIR}/golden/hsoh_def2svp_cas66_casscf.nevpt2gold")
  set(_side_hsoh66_casscf
      "${CMAKE_CURRENT_SOURCE_DIR}/golden/hsoh_def2svp_cas66_casscf.gradgold")
  if(NOT EXISTS "${_side_hsoh66_casscf}")
    message(FATAL_ERROR "nevpt2: committed gradient sidecar missing: ${_side_hsoh66_casscf}")
  endif()
  add_golden_test(nevpt2_hsoh_cas66_casscf "${_gold_hsoh66_casscf}" --tiles 3)
  add_pc_golden_test(nevpt2_hsoh_cas66_casscf_pc "${_gold_hsoh66_casscf}" --tiles 3)
  _add_golden_test_with(nevpt2_demo nevpt2_gradient_source_hsoh_cas66_casscf
                        "${_gold_hsoh66_casscf}" --tiles 3 --gradient-source
                        --gradgold "${_side_hsoh66_casscf}")
  if(TEST nevpt2_gradient_source_hsoh_cas66_casscf)
    set_tests_properties(nevpt2_gradient_source_hsoh_cas66_casscf PROPERTIES
      PASS_REGULAR_EXPRESSION
        "PASS: the CI source term and the forward tangent agree .*, the orbital energies moving with the active density")
  endif()
  unset(_gold_hsoh66_casscf)
  unset(_side_hsoh66_casscf)

  # --- every Hessian flag on CASSCF orbitals (#157) -------------------------------
  #
  # Level 1 differentiates the energy functional, which does not know how its
  # orbitals were made, so every Hessian flag sits behind the SOURCE-TERM gate
  # and takes a CASSCF golden -- the opposite of the Lagrangian gate a full
  # gradient sits behind (docs/hessian-theory.md, Sec. 9.1). One entry per
  # flag on n2_ccpvdz_cas44_casscf, the same binary and flags as its CASCI
  # twin above; where the PASS line names the orbital-energy mode, the regex
  # holds it to that mode (fixed for the CI block and PC; FULL with Euler
  # skipped, and EULER with it gated, for H_kc, whose committed sidecar sits
  # beside this golden). All `small`, as their CAS(4,4) twins are.
  # A regex here never spells a literal ';' (a PASS_REGULAR_EXPRESSION is a
  # list, and one would split it into two alternatives): '.' stands in.
  function(_add_casscf_hessian_test binary name flag pass)
    _add_golden_test_with(${binary} ${name} "${_gold_44_casscf}" --tiles 3 ${flag})
    if(TEST ${name})
      set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "${pass}"
                                              LABELS "gpu;small")
    endif()
  endfunction()
  _add_casscf_hessian_test(nevpt2_demo nevpt2_cas44_casscf_sc_hessian --sc-hessian
    "${_nevpt2_sc_hessian_pass} .*\\(the orbital energies held fixed\\)")
  _add_casscf_hessian_test(nevpt2_demo nevpt2_cas44_casscf_sc_hessian_vector --sc-hessian-vector
    "${_nevpt2_sc_hessian_vector_pass} .*\\(the orbital energies held fixed on both legs\\)")
  _add_casscf_hessian_test(nevpt2_sc_pseudodensity
    nevpt2_sc_pseudodensity_cas44_casscf_sc_hessian_kc --sc-hessian-kc
    "${_nevpt2_sc_hessian_kc_pass} .*\\(orbital-energy modes: FULL, the orbital energies responding to kappa and to c, Euler skipped there. EULER, held fixed in c, Euler gated\\)")
  _add_casscf_hessian_test(nevpt2_sc_pseudodensity
    nevpt2_sc_pseudodensity_cas44_casscf_sc_hessian_kk_int --sc-hessian-kk-int
    "${_nevpt2_sc_hessian_kk_pass}")
  _add_casscf_hessian_test(nevpt2_sc_pseudodensity
    nevpt2_sc_pseudodensity_cas44_casscf_sc_hessian_kk --sc-hessian-kk
    "${_nevpt2_sc_hessian_kk_full_pass}")
  _add_casscf_hessian_test(nevpt2_pc_tangent nevpt2_pc_hessian_cas44_casscf --pc-hessian
    "PASS: the PC-NEVPT2 Hessian Eddot\\(u, v\\) matches Euler .*\\(orbital energies held fixed. smallest kept s/s_max [^ ]+ \\([A-Za-z]+\\), smallest gap [^ ]+ against PC_MIN_GAP")
  _add_casscf_hessian_test(nevpt2_pc_tangent nevpt2_pc_hessian_vector_cas44_casscf
    --pc-hessian-vector
    "PASS: the PC-NEVPT2 Hessian-vector product H_cc u matches .*\\(orbital energies held fixed. smallest kept s/s_max [^ ]+ \\([A-Za-z]+\\), smallest gap [^ ]+ against PC_MIN_GAP")
  # The CI block's FULL mode (#214): the four H_cc flags again with the
  # committed sidecar beside this golden, each regex holding its line to FULL
  # (Euler skipped). Their CASCI twins sit beside the frozen-mode entries.
  set(_side_44_casscf "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44_casscf.gradgold")
  if(NOT EXISTS "${_side_44_casscf}")
    message(FATAL_ERROR "nevpt2: committed gradient sidecar missing: ${_side_44_casscf}")
  endif()
  set(_nevpt2_pc_hessian_full_pass
      "PASS: the PC-NEVPT2 Hessian Eddot\\(u, v\\) matches the central difference of Edot \\(orbital-energy mode: FULL, the orbital energies responding to c, Euler skipped there. smallest kept s/s_max [^ ]+ \\([A-Za-z]+\\), smallest gap [^ ]+ against PC_MIN_GAP")
  set(_nevpt2_pc_hessian_vector_full_pass
      "PASS: the PC-NEVPT2 Hessian-vector product H_cc u matches .*\\(orbital-energy mode: FULL, the orbital energies responding to c, Euler skipped there. smallest kept s/s_max [^ ]+ \\([A-Za-z]+\\), smallest gap [^ ]+ against PC_MIN_GAP")
  _add_casscf_hessian_test(nevpt2_demo nevpt2_cas44_casscf_sc_hessian_full
    "--sc-hessian;--gradgold;${_side_44_casscf}" "${_nevpt2_sc_hessian_full_pass}")
  _add_casscf_hessian_test(nevpt2_demo nevpt2_cas44_casscf_sc_hessian_vector_full
    "--sc-hessian-vector;--gradgold;${_side_44_casscf}" "${_nevpt2_sc_hessian_vector_full_pass}")
  _add_casscf_hessian_test(nevpt2_pc_tangent nevpt2_pc_hessian_cas44_casscf_full
    "--pc-hessian;--gradgold;${_side_44_casscf}" "${_nevpt2_pc_hessian_full_pass}")
  _add_casscf_hessian_test(nevpt2_pc_tangent nevpt2_pc_hessian_vector_cas44_casscf_full
    "--pc-hessian-vector;--gradgold;${_side_44_casscf}" "${_nevpt2_pc_hessian_vector_full_pass}")
  unset(_side_44_casscf)
  # PC's orbital blocks (#200-#202; #203): the same binary and flags with --pc
  # as their CAS(4,4) twins, the committed sidecar beside this golden read.
  # `flag` is a list here ("--pc;<flag>"), expanded unquoted above. Each regex
  # holds the line to its orbital-energy mode: EULER for the integral
  # curvature, EULER along a mixed w for H_kc (its FROZEN no-sidecar walk
  # runs inside, check 7), both modes for the sandwich.
  _add_casscf_hessian_test(nevpt2_sc_pseudodensity
    nevpt2_sc_pseudodensity_cas44_casscf_pc_hessian_kc "--pc;--sc-hessian-kc"
    "${_nevpt2_pc_hessian_kc_pass} to 1e-07 \\(EULER mode along a mixed w")
  _add_casscf_hessian_test(nevpt2_sc_pseudodensity
    nevpt2_sc_pseudodensity_cas44_casscf_pc_hessian_kk_int "--pc;--sc-hessian-kk-int"
    "${_nevpt2_pc_hessian_kk_pass}.*\\(orbital-energy mode: EULER, the orbital energies responding to kappa\\)")
  _add_casscf_hessian_test(nevpt2_sc_pseudodensity
    nevpt2_sc_pseudodensity_cas44_casscf_pc_hessian_kk "--pc;--sc-hessian-kk"
    "${_nevpt2_pc_hessian_kk_full_pass}")

  # --- the PC tangent with respect to the CI vector -------------------------------
  #
  # nevpt2_pc_tangent (apps/pc_tangent/, the only binary that links
  # nevpt2::pc_tangent): nevpt2.energy's analytic derivative of the PC solve
  # (pcTangentsDevice) against Euler's identity -- exact on the five classes
  # whose matrices are linear in the RDMs -- and against a central finite
  # difference of the PC total on the unit sphere, Richardson-verified: the
  # error must fall 4x per halving and the extrapolation must land on the
  # analytic tangent (docs/pc-nevpt2.md, "The tangent with respect to the CI
  # vector"). A finite difference like the *_sc_tangent entries above, and for
  # a sharper reason: the derivative of a solve has no algebraic identity at
  # all short of differentiating the eigendecomposition, which is what this
  # avoids. Its PASS line is worded differently from every other entry's, so
  # none can pass on another's output.
  #
  # Each step costs TWO more states, each a full RDM build (host dm1/dm2 too:
  # the golden's arrays are only the state's own) and a full PC stage, so these
  # are the slowest entries per case in the tree. CAS(4,4) and CAS(8,8) are
  # `small` as well -- the sanitizer tier then covers the tangent's einsums,
  # GEMMs and the second solve -- and the 12-orbital PC case is `slow`.
  foreach(_case cas44 cas88)
    set(_pc_tan_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_${_case}.nevpt2gold")
    _add_golden_test_with(nevpt2_pc_tangent nevpt2_pc_tangent_${_case}
                          "${_pc_tan_gold}" --tiles 3)
    if(TEST nevpt2_pc_tangent_${_case})
      set_tests_properties(nevpt2_pc_tangent_${_case} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: the PC-NEVPT2 tangent matches the central finite difference"
        LABELS "gpu;small")
    endif()
    unset(_pc_tan_gold)
  endforeach()

  # CAS(10,12), the 12-orbital PC case (the cas1212 goldens carry no PC
  # fields). --tiles 40 is above BOTH measured floors that matter here: the
  # plain build's 14 and the TANGENT build's 25, which is its own because it
  # holds about twice the per-tile intermediates (docs/performance.md, "Tile
  # floors"). Three steps rather than four: every step is two more 12-orbital
  # RDM builds.
  _add_golden_test_with(nevpt2_pc_tangent nevpt2_pc_tangent_cas1012 "${_gold_1012}"
                        --tiles 40 --fd-steps 3)
  if(TEST nevpt2_pc_tangent_cas1012)
    set_tests_properties(nevpt2_pc_tangent_cas1012 PROPERTIES
      PASS_REGULAR_EXPRESSION "PASS: the PC-NEVPT2 tangent matches the central finite difference"
      LABELS "gpu;slow")
  endif()

  # --- the PC second derivative, Eddot(u, v) (#152) -------------------------------
  #
  # nevpt2_pc_tangent --pc-hessian: nevpt2.energy's pcHessianDevice -- the PC
  # solve differentiated twice, with docs/hessian-theory.md Sec. 6.2's
  # rotation terms (S's dropped space turns with the state) -- at the golden's
  # fixed orbital energies, against Euler at second order (Eddot(c, u) ==
  # Edot(u), exact, on the five homogeneous classes), a negative control (a
  # symmetry-breaking direction, along which the dropped modes lift, must be
  # REFUSED), and a Richardson-verified central difference of the three-term
  # Edot along the unit sphere (docs/pc-nevpt2.md, "The second derivative,
  # Eddot(u, v)"). CAS(4,4), whose Sr null space is a rank bound that turns
  # with the state -- the case the rotation terms exist for; `small`, so the
  # sanitizer tier covers the dropped-space projections. --tiles 3 is far
  # above the three-argument build's floor (docs/performance.md, "Tile
  # floors").
  set(_pc_hess_gold "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44.nevpt2gold")
  _add_golden_test_with(nevpt2_pc_tangent nevpt2_pc_hessian_cas44 "${_pc_hess_gold}"
                        --tiles 3 --pc-hessian)
  if(TEST nevpt2_pc_hessian_cas44)
    set_tests_properties(nevpt2_pc_hessian_cas44 PROPERTIES
      PASS_REGULAR_EXPRESSION "PASS: the PC-NEVPT2 Hessian Eddot\\(u, v\\) matches Euler"
      LABELS "gpu;small")
  endif()

  # --- the PC Hessian-vector product, H_cc u (#155) -------------------------------
  #
  # nevpt2_pc_tangent --pc-hessian-vector: nevpt2.gradient's
  # pcCiHessianVectorDevice -- PC's adjoint walk differentiated along u, then
  # two sigma calls per class -- at the golden's fixed orbital energies,
  # against the entry above's Eddot(u, v) by duality (per class, three seeded
  # v and the state), H_cc's symmetry, Euler as a vector (H_cc c == y on the
  # homogeneous classes), a negative control that must be REFUSED for its
  # lift, and ONE finite difference: of the GPU's own three-term y, per class,
  # Richardson-verified, which must converge fourfold for every class -- the
  # only check that sees the rotation terms (docs/pc-nevpt2.md, "The
  # Hessian-vector product, H_cc u"). CAS(4,4), as above, and `small`. No
  # CAS(8,8) entry, for the same conditioning reason as --pc-hessian's (same
  # section).
  _add_golden_test_with(nevpt2_pc_tangent nevpt2_pc_hessian_vector_cas44 "${_pc_hess_gold}"
                        --tiles 3 --pc-hessian-vector)
  if(TEST nevpt2_pc_hessian_vector_cas44)
    set_tests_properties(nevpt2_pc_hessian_vector_cas44 PROPERTIES
      PASS_REGULAR_EXPRESSION "PASS: the PC-NEVPT2 Hessian-vector product H_cc u matches"
      LABELS "gpu;small")
  endif()

  # --- PC's CI block in FULL mode, --gradgold beside the PC H_cc flags (#214) -----
  #
  # The two entries above with the committed gradient sidecar beside the
  # golden: the orbital energies respond to the CI vector on every side of
  # every check (every tuple's Delta'[u], Delta'[v], Delta''[u, v] in the
  # solve's second derivative; the rows' Rdot folded through W in H_cc u),
  # every displaced state at its own orbital energies, Euler skipped and said
  # to be. --pc-hessian keeps its control and its difference of Edot;
  # --pc-hessian-vector its duality, symmetry, control and per-class ladder of
  # the gated three-term y. Each PASS line names the mode. CAS(4,4) (its
  # CASSCF twins are with the #157 entries above, as are the two regexes),
  # `small`, --tiles 3.
  set(_pc_hess_side "${CMAKE_CURRENT_SOURCE_DIR}/golden/n2_ccpvdz_cas44.gradgold")
  _add_golden_test_with(nevpt2_pc_tangent nevpt2_pc_hessian_cas44_full "${_pc_hess_gold}"
                        --tiles 3 --pc-hessian --gradgold "${_pc_hess_side}")
  _add_golden_test_with(nevpt2_pc_tangent nevpt2_pc_hessian_vector_cas44_full "${_pc_hess_gold}"
                        --tiles 3 --pc-hessian-vector --gradgold "${_pc_hess_side}")
  if(TEST nevpt2_pc_hessian_cas44_full)
    set_tests_properties(nevpt2_pc_hessian_cas44_full PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_pc_hessian_full_pass}" LABELS "gpu;small")
  endif()
  if(TEST nevpt2_pc_hessian_vector_cas44_full)
    set_tests_properties(nevpt2_pc_hessian_vector_cas44_full PROPERTIES
      PASS_REGULAR_EXPRESSION "${_nevpt2_pc_hessian_vector_full_pass}" LABELS "gpu;small")
  endif()
  unset(_pc_hess_side)
  unset(_pc_hess_gold)

  # --- salicylaldimine CAS(8,8)/6-31G** -------------------------------------------
  #
  # The first molecule that is not N2: salicylaldimine (C7H7NO, PubChem CID
  # 260739, flattened to Cs), its 4 pi + 4 pi* orbitals as the active space
  # (generate_golden.py --cas-irreps; docs/reference-data.md, "Salicylaldimine
  # CAS(8,8)"). 161 MOs, ncore=28, nvirt=125 -- the external spaces are far
  # larger than N2's, the active space the same size as n2_ccpvdz_cas88. The
  # DF file is fitted with 6-31G**-RIFIT (reference_data/basis/), not a JK
  # basis. --tiles 3 is not a measured floor (one tile fits, as for the N2
  # CAS(8,8)); the default --batch 8 leaves a ragged last slab (125 = 15*8 + 5).
  # Not `small`: the sanitizer tier stays on N2. The two SC entries are `slow`
  # because of HIP: ~35s on the RX 9060 XT, nearly all of it a few
  # small-output einsums in the SC energy stage, against ~1.2s on the RTX 3080
  # (the PC entries take ~2s / ~0.8s). The goldens are gitignored (145/166 MB) -- generate
  # them first, or these four are not registered.
  set(_gold_sal88 "${CMAKE_CURRENT_SOURCE_DIR}/golden/salicylaldimine_631gss_cas88.nevpt2gold")
  set(_gold_sal88_df "${CMAKE_CURRENT_SOURCE_DIR}/golden/salicylaldimine_631gss_cas88_df.nevpt2gold")
  add_golden_test(nevpt2_salicylaldimine_cas88 "${_gold_sal88}" --tiles 3)
  add_pc_golden_test(nevpt2_salicylaldimine_cas88_pc "${_gold_sal88}" --tiles 3)
  add_df_golden_test(nevpt2_df_salicylaldimine_cas88 "${_gold_sal88_df}"
                     --tiles 3 --check-blocks)
  add_df_pc_golden_test(nevpt2_df_salicylaldimine_cas88_pc "${_gold_sal88_df}" --tiles 3)
  foreach(_t nevpt2_salicylaldimine_cas88 nevpt2_df_salicylaldimine_cas88)
    if(TEST ${_t})
      set_tests_properties(${_t} PROPERTIES LABELS "gpu;slow")
    endif()
  endforeach()

  # --- sanitizer canaries ---------------------------------------------------------
  #
  # A deliberately buggy run (apps/sanitizer_canary/) that passes only when its
  # sanitizer reports the bug AND fails the run (add_sanitizer_canary).
  # Registered only under the sanitizer that must catch it, so a green canary
  # proves that sanitizer is on in this build. docs/testing.md ("Canaries") records
  # each one going red with its sanitizer off.
  include(add_sanitizer_canary)
  if(NEVPT2_ENABLE_ASAN)
    add_sanitizer_canary(
      NAME nevpt2_canary_asan_heap_overflow
      TARGET nevpt2_sanitizer_canary
      ARGS host-heap-overflow
      EXPECT "AddressSanitizer: heap-buffer-overflow")
  endif()
  if(NEVPT2_ENABLE_UBSAN)
    add_sanitizer_canary(
      NAME nevpt2_canary_ubsan_signed_overflow
      TARGET nevpt2_sanitizer_canary
      ARGS host-signed-overflow
      EXPECT "runtime error: signed integer overflow")
  endif()
  if(NEVPT2_COMPUTE_SANITIZER)
    set(_tool "${NEVPT2_COMPUTE_SANITIZER_TOOL}")
    if(_tool STREQUAL "memcheck")
      add_sanitizer_canary(
        NAME nevpt2_canary_memcheck_device_oob
        TARGET nevpt2_sanitizer_canary
        ARGS device-oob
        EXPECT "Invalid __global__ read of size 8"
        REQUIRES_GPU)
      # The bug stream-ordered allocation can introduce: a free that runs
      # before the kernel still reading the buffer.
      add_sanitizer_canary(
        NAME nevpt2_canary_memcheck_free_before_read
        TARGET nevpt2_sanitizer_canary
        ARGS device-free-before-read
        EXPECT "Invalid __global__ read of size 8"
        REQUIRES_GPU)
      # The same bug through the demos' own allocator: a pool-backed
      # DeviceBuffer, whose freed block the pool keeps mapped.
      add_sanitizer_canary(
        NAME nevpt2_canary_memcheck_pool_free_before_read
        TARGET nevpt2_sanitizer_canary
        ARGS device-pool-free-before-read
        EXPECT "Invalid __global__ read of size 8"
        REQUIRES_GPU)
      # A device allocation never freed: proves --leak-check full
      # is really on, so the demo entries' 0 leaks is a measurement.
      if(NEVPT2_COMPUTE_SANITIZER_LEAK_CHECK)
        add_sanitizer_canary(
          NAME nevpt2_canary_memcheck_leak
          TARGET nevpt2_sanitizer_canary
          ARGS device-leak
          EXPECT "Leaked 512 bytes at"
          REQUIRES_GPU)
      endif()
    elseif(_tool STREQUAL "initcheck")
      add_sanitizer_canary(
        NAME nevpt2_canary_initcheck_uninit_read
        TARGET nevpt2_sanitizer_canary
        ARGS device-uninit-read
        EXPECT "Uninitialized __global__ memory read of size 8"
        REQUIRES_GPU)
    elseif(_tool STREQUAL "racecheck")
      add_sanitizer_canary(
        NAME nevpt2_canary_racecheck_shared_race
        TARGET nevpt2_sanitizer_canary
        ARGS device-shared-race
        EXPECT "Race reported between"
        REQUIRES_GPU)
    elseif(_tool STREQUAL "synccheck")
      add_sanitizer_canary(
        NAME nevpt2_canary_synccheck_divergent_barrier
        TARGET nevpt2_sanitizer_canary
        ARGS device-divergent-barrier
        EXPECT "Barrier error detected"
        REQUIRES_GPU)
    endif()
    unset(_tool)
  endif()

  # --- the GoogleTest unit tier ------------------------------------
  #
  # One directory per component under test/, mirroring src/; every entry is
  # labelled `unit` (`ctest --preset unit`), and `gpu` too only if it needs a
  # card. Added from here only: the standalone `cmake -S src` build stays
  # gtest-free. cmake/README.md, "Tests".
  add_subdirectory(test)
endif()
