# The SC-NEVPT2 gradient's first half: the amplitudes and multipliers, and the
# pseudodensities assembled from them over the class energies' own slab walk.
# A LEAF -- nothing else in src/ imports either module and neither demo links
# them; apps/sc_pseudodensity/ is what drives it.

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

# nevpt2.gradient -- the device assembly. It imports nevpt2.energy (and
# re-exports it: the entry points take its ActiveIntegralsDevice and
# IntegralSource, and build the metric / Dyall-Hamiltonian blocks with its own
# exported make_* functions), and runs entirely through nevpt2.einsum's
# generic contraction kernel, which nevpt2.energy re-exports -- so there is no
# device library of its own here. gradient_terms.cppm and
# gradient_sc_classes.cppm are the internal partitions nevpt2.gradient:terms
# (the generic term/derivative machinery) and nevpt2.gradient:sc_classes (the
# eight classes), which no importer sees.
add_cxx_module_library(
  NAME nevpt2.gradient
  PRIMARY_INTERFACE gradient.cppm
  PARTITIONS gradient_terms.cppm gradient_sc_classes.cppm
  IMPLEMENTATION gradient.cpp
  LINK_PUBLIC nevpt2::energy nevpt2::sc_amplitudes
)
