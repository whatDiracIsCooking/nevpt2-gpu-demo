# The gradient work: the SC half's amplitudes, multipliers, pseudodensities
# and CI source term, and the PC half's tangent check. ALL LEAVES -- nothing
# else in src/ imports any of these modules. apps/sc_pseudodensity/ and
# apps/pc_tangent/ drive most of it; nevpt2_demo links nevpt2::gradient for
# --gradient-source, the CI source term's forward-reverse duality check.

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
# multipliers, termHasActive, and the CI source term's host folding. No GPU,
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
# (the chain rule from the metric / Dyall blocks down to the five RDMs) and
# :ci_source (Park Eq. 48's sigma assembly). PARTITIONS is in dependency
# order.
#
# nevpt2::sigma_vector is PUBLIC: scCiSourceDevice takes its
# SigmaVectorOptions, and the CI source term IS its three sigma entry points
# fed the RDM coefficient tensors. It brings in the sigma kernels' own device
# library, which is the only device code reaching this component.
add_cxx_module_library(
  NAME nevpt2.gradient
  PRIMARY_INTERFACE gradient.cppm
  PARTITIONS gradient_general.cppm gradient_rotation.cppm gradient_terms.cppm
             gradient_sc_classes.cppm gradient_rdm_adjoint.cppm gradient_ci_source.cppm
  IMPLEMENTATION gradient.cpp
  LINK_PUBLIC nevpt2::energy nevpt2::sc_amplitudes nevpt2::gradient_plan
              nevpt2::sigma_vector
)

# nevpt2.pc_tangent -- the finite-difference check of PC-NEVPT2's CI-vector
# tangent: it drives nevpt2.energy's pcTangentsDevice at one state and its
# pcEnergiesDevice at the displaced states c +- h u. The only module in src/
# that links the RDM build AND the energy stage, because that is what a
# displaced state costs: nevpt2::rdm_host_dm rebuilds its dm1/dm2 on the host
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
