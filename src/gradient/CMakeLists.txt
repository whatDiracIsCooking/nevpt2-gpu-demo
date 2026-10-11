# The gradient work: the SC half's amplitudes, multipliers and
# pseudodensities, and the PC half's tangent check. ALL LEAVES -- nothing else
# in src/ imports any of these modules and neither demo links them;
# apps/sc_pseudodensity/ and apps/pc_tangent/ are what drive them.

# nevpt2.sc_amplitudes -- the amplitude T, the multipliers P/Q/R (Park Eqs.
# 12, 36-38) and the einsum-subscript rewriting that turns a class term into
# its own derivative. Pure host arithmetic and string handling, so the unit
# tier tests it card-free (test/gradient/); nevpt2.gradient re-exports it.
# nevpt2::energy_finish is where NUMERICAL_ZERO and normToEnergy (whose
# division the amplitude IS) live -- and, like it, this needs no GPU.
add_cxx_module_library(
  NAME nevpt2.sc_amplitudes
  PRIMARY_INTERFACE sc_amplitudes.cppm
  LINK_PUBLIC nevpt2::energy_finish
  LINK_PRIVATE nevpt2::error_handling
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
# device library of its own here. gradient_terms.cppm and
# gradient_sc_classes.cppm are the internal partitions nevpt2.gradient:terms
# (the generic term/derivative machinery) and nevpt2.gradient:sc_classes (the
# eight classes), which no importer sees; gradient_rotation.cppm is
# nevpt2.gradient:rotation, the active-rotation machinery both of them use
# (:terms imports it, so it is listed first).
add_cxx_module_library(
  NAME nevpt2.gradient
  PRIMARY_INTERFACE gradient.cppm
  PARTITIONS gradient_rotation.cppm gradient_terms.cppm gradient_sc_classes.cppm
  IMPLEMENTATION gradient.cpp
  LINK_PUBLIC nevpt2::energy nevpt2::sc_amplitudes
)

# nevpt2.pc_tangent -- the finite-difference check of PC-NEVPT2's CI-vector
# tangent: it drives nevpt2.energy's pcTangentsDevice at one state and its
# pcEnergiesDevice at the displaced states c +- h u. The only module in src/
# that links the RDM build AND the energy stage, because that is what a
# displaced state costs: nevpt2::rdm_host_dm rebuilds its dm1/dm2 on the host
# (the golden file's arrays are only the state's own) and nevpt2::rdm_build
# its dm3/f3ac/f3ca on the device. apps/pc_tangent/ is the only binary that
# links it.
add_cxx_module_library(
  NAME nevpt2.pc_tangent
  PRIMARY_INTERFACE pc_tangent.cppm
  IMPLEMENTATION pc_tangent.cpp
  LINK_PUBLIC nevpt2::energy nevpt2::rdm_build nevpt2::rdm_host_dm
)
