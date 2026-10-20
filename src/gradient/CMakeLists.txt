# The gradient work: the SC half's amplitudes, multipliers, pseudodensities
# and CI source term, the PC half's source terms (#74) and its tangent check. ALL LEAVES -- nothing
# else in src/ imports any of these modules. apps/sc_pseudodensity/ and
# apps/pc_tangent/ drive most of it; both demos link nevpt2::ci_checks
# (below) for --sc-tangent and --gradient-source, the CI source term's
# forward-reverse duality check.

# nevpt2.sc_amplitudes -- the amplitude T, the multipliers P/Q/R (Park Eqs.
# 12, 36-38), the pseudocanonical multipliers z^c (Eqs. 56-57) and the
# einsum-subscript rewriting that turns a class term into its own derivative.
# Pure host arithmetic and string handling, so the unit tier tests it
# card-free (test/gradient/); nevpt2.gradient re-exports it.
# nevpt2::energy_finish is where NUMERICAL_ZERO and normToEnergy (whose
# division the amplitude IS) live -- and, like it, this needs no GPU.
# error_handling is PUBLIC because Eqs. 56-57 hand back a Result: a
# degeneracy their division cannot resolve is a numerical refusal.
add_cxx_module_library(
  NAME nevpt2.sc_amplitudes
  PRIMARY_INTERFACE sc_amplitudes.cppm
  LINK_PUBLIC nevpt2::energy_finish nevpt2::error_handling
)

# nevpt2.gradient_plan -- the device assembly's pure host arithmetic, pulled
# out of nevpt2.gradient's internal partitions (which no test can import) the
# way nevpt2.rdm_plan was pulled out of the RDM build: the promotion table
# (rotBlockInfo, activeAxesOf, spanOf, generalPickPlan, promotionPlan,
# heffDressingPlan), the
# eight classes' Delta vectors and the two symmetrized classes' slab
# multipliers, termHasActive, the CI source term's host folding, and the
# orbital energies' CI response (W out of the sidecar, dE/de, #87). No GPU,
# so the unit tier tests it card-free (test/gradient_plan/); nevpt2.gradient
# re-exports it.
add_cxx_module_library(
  NAME nevpt2.gradient_plan
  PRIMARY_INTERFACE gradient_plan.cppm
  LINK_PUBLIC nevpt2::tensor nevpt2::sc_amplitudes nevpt2::error_handling
)

# nevpt2.kappa_rotation -- the orbital-rotation parameterisation, host side:
# an antisymmetric kappa, exp(kappa), and the transformation an orthogonal U
# induces on one array's active axes. No device either, so the unit tier
# checks it card-free (test/gradient/): that U is orthogonal, and that a
# rotation by kappa = 0 leaves a block bit-identical. nevpt2.gradient does NOT
# link it -- the device twin of the kappa action is its own (:rotation) --
# apps/sc_pseudodensity/ does, for the finite-difference side of the check.
add_cxx_module_library(
  NAME nevpt2.kappa_rotation
  PRIMARY_INTERFACE kappa_rotation.cppm
  LINK_PUBLIC nevpt2::tensor
  LINK_PRIVATE nevpt2::error_handling
)

# nevpt2.gradient -- the device assembly. It imports nevpt2.energy (and
# re-exports it: the entry points take its ActiveIntegralsDevice and
# IntegralSource, and build the metric / Dyall-Hamiltonian blocks with its own
# exported make_* functions), and runs entirely through nevpt2.einsum's
# generic contraction kernel, which nevpt2.energy re-exports -- so there is no
# device library of its own here. The SIX internal partitions, which no
# importer sees, are nevpt2.gradient:general (gradient_general.cppm -- the
# promotion table, from nevpt2.gradient_plan, read against the sidecar's
# device arrays, and the general-range action a core/virtual/mixed
# rotation needs, which :rotation builds on), :rotation
# (gradient_rotation.cppm, the rotation machinery), :terms (the generic
# term/derivative machinery), :sc_classes (the eight classes), :rdm_adjoint
# (the chain rule from the metric / Dyall blocks down to the five RDMs),
# :ci_source (Park Eq. 48's sigma assembly) and :pc_classes (PC-NEVPT2's
# reverse mode, #74: the PC solve's adjoint over SC's own rotation machinery
# and block sweeps). PARTITIONS is in dependency order.
#
# nevpt2::sigma_vector is PUBLIC: scCiSourceDevice takes its
# SigmaVectorOptions, and the CI source term IS its three sigma entry points
# fed the RDM coefficient tensors. It brings in the sigma kernels' own device
# library. nevpt2::rdm_build is PUBLIC too (#88): the digests' tangent along a
# rotation is buildRdmsDevice run at the dotted active integrals, so
# scDigestRotationTangentDevice takes its RdmBuildOptions and calls it. That
# brings the RDM kernels' device libraries in as well; no kernel is new.
#
# gradient_df.cpp (#165) is dfGeneralIntegralsDevice: the density-fitted
# case's general-range inventory, contracted on the device from the whole
# three-index tensor (the golden's four B blocks plus the NEVPT2F1 sidecar's
# B_cc / B_vv) into the same GeneralIntegralsDevice the sidecar fills for a
# conventional case -- and (#168) dfDigestSourceDevice, the digests'
# transpose carried from h2e back onto the three-index tensor. Einsums only,
# so still no device library here.
add_cxx_module_library(
  NAME nevpt2.gradient
  PRIMARY_INTERFACE gradient.cppm
  PARTITIONS gradient_general.cppm gradient_rotation.cppm gradient_terms.cppm
             gradient_sc_classes.cppm gradient_rdm_adjoint.cppm gradient_ci_source.cppm
             gradient_pc_classes.cppm
  IMPLEMENTATION gradient.cpp gradient_df.cpp
  LINK_PUBLIC nevpt2::energy nevpt2::sc_amplitudes nevpt2::gradient_plan
              nevpt2::sigma_vector nevpt2::rdm_build
  # The mixed block's forward orientation (#184) builds its CI direction's
  # dm1/dm2 tangents itself, on the host, as nevpt2.ci_checks does.
  LINK_PRIVATE nevpt2::rdm_host_dm
)

# nevpt2.pc_tangent -- the finite-difference check of PC-NEVPT2's CI-vector
# tangent: it drives nevpt2.energy's pcTangentsDevice at one state and its
# pcEnergiesDevice at the displaced states c +- h u. It links the RDM build
# AND the energy stage (as nevpt2.gradient now does, for the digests' orbital
# response), because that is what a displaced state costs: nevpt2::rdm_host_dm rebuilds its dm1/dm2 on the host
# (the golden file's arrays are only the state's own) and nevpt2::rdm_build
# its dm3/f3ac/f3ca on the device. nevpt2::gradient_plan (host-only) is for
# longestFourfoldRun, the step-ladder rule its report gates on.
# apps/pc_tangent/ is the only binary that links it.
add_cxx_module_library(
  NAME nevpt2.pc_tangent
  PRIMARY_INTERFACE pc_tangent.cppm
  IMPLEMENTATION pc_tangent.cpp
  LINK_PUBLIC nevpt2::energy nevpt2::rdm_build nevpt2::rdm_host_dm nevpt2::gradient_plan
)

# nevpt2.ci_checks -- the two checks of the energy's CI-vector derivative the
# demos run in place of the golden comparison: --sc-tangent (the SC forward
# tangent against the norms' affine increment, Euler and a finite difference)
# and --gradient-source (y_I against that tangent by the forward-reverse
# duality, SC and, integral-direct only, PC). A library rather than one
# demo's partitions because BOTH demos run the SC pair (#162): each takes an
# IntegralSource, FullBlockSource from nevpt2_demo and DfIntegralSource from
# nevpt2_df_demo, so one body of checks serves both. A leaf, like
# nevpt2.pc_tangent: it links the RDM build (nevpt2::gradient re-exports it)
# and nevpt2::rdm_host_dm for the same reason -- every displaced state and
# every tangent needs its own RDMs -- and nevpt2::golden for the state and
# the gradient sidecar.
add_cxx_module_library(
  NAME nevpt2.ci_checks
  PRIMARY_INTERFACE ci_checks.cppm
  IMPLEMENTATION ci_checks_tangent.cpp ci_checks_source.cpp
  LINK_PUBLIC nevpt2::gradient nevpt2::golden
  LINK_PRIVATE nevpt2::rdm_host_dm
)
