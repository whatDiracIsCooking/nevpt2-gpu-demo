# nevpt2.energy_finish -- the class energies' pure host arithmetic: the slab
# walk (forBatches), the SC finish (normToEnergy) and its derivative
# (normToEnergyDerivative, normToEnergyTangent -- the SC forward tangent's
# only calculus), the PC finish and denominator check (finishSingle,
# checkDenominator), the PC tangent's per-tuple arithmetic (pcCoefficients,
# pcQuadForm, tangentSingle) and the result types they fill. No GPU, so the
# unit tier tests it card-free (test/energy_finish/); nevpt2.energy
# re-exports it.
add_cxx_module_library(
  NAME nevpt2.energy_finish
  PRIMARY_INTERFACE energy_finish.cppm
  LINK_PUBLIC nevpt2::tensor
  # error_handling: check(), which normToEnergyTangent (inline, in the
  # interface) calls on its five vector lengths, and the PC tangent's helpers
  # on their (m, m) block shapes -- the abort tier.
  LINK_PRIVATE nevpt2::error_handling
)

# nevpt2.energy -- the eight NEVPT2 class energies on the GPU, strongly
# contracted (energy.cpp) through nevpt2.einsum (which it re-exports: the
# entry points' signatures name DeviceTensor and DeviceResources), and
# partially contracted (energy_pc.cpp, until the merge its own
# module, nevpt2.energy_pc) from the same intermediates. PC's per-class
# eigensolves go through DeviceResources' solver handle (wwr.solver --
# cuSOLVER / hipSOLVER, linked PUBLIC by nevpt2.device_resources) and its
# d x d GEMMs through the BLAS handle beside it (wwr.blas, likewise from
# nevpt2.device_resources);
# wwr.extension.solver supplies the wwrsolverStatus_t error specializations
# gpuCheck needs. energy_shared.cppm is the internal partition
# nevpt2.energy:shared -- the Sijrs class both units use, which no importer
# sees. energy_sc_tangent.cppm is nevpt2.energy:sc_tangent, the SC classes'
# forward tangent with respect to the RDMs (energiesTangentDevice): it reruns
# :sc_classes' own per-slab chains with the tangent RDMs substituted, so it
# needs no kernel and no subscript of its own. energy_pc_tangent.cppm is
# nevpt2.energy:pc_tangent, the per-class work behind pcTangentsDevice -- the
# PC class energies' derivative with respect to the CI vector, which
# differentiates the SOLVE and not the eigendecomposition.
# energy_sc_hessian.cppm is nevpt2.energy:sc_hessian, the SC classes' second
# directional derivative Eddot(u, v) (energiesHessianDevice, #153): the
# tangent's walks carrying four RDM sets, through :sc_tangent's shared block
# builders. energy_pc_hessian.cppm is nevpt2.energy:pc_hessian, the PC
# classes' second directional derivative behind pcHessianDevice (#152): the
# solve differentiated twice, over the same builders. No importer sees any of
# them. (The metric / Dyall-Hamiltonian intermediates ARE exported, from
# energy.cppm: nevpt2.gradient differentiates the same class einsums.)
add_cxx_module_library(
  NAME nevpt2.energy
  PRIMARY_INTERFACE energy.cppm
  PARTITIONS energy_shared.cppm energy_sc_finish.cppm energy_sc_classes.cppm
             energy_sc_tangent.cppm energy_sc_hessian.cppm
             energy_pc_solve.cppm energy_pc_classes.cppm energy_pc_tangent.cppm
             energy_pc_hessian.cppm
  IMPLEMENTATION energy.cpp energy_pc.cpp
  LINK_PUBLIC nevpt2::einsum nevpt2::energy_finish
  LINK_PRIVATE nevpt2::profile wwr::solver wwr::extension::solver
)

# energy_Sijrs launches device_einsum.cu's slab reduction directly through
# einsum/device_einsum_bridge.h (after its imports, in extern "C++"), so it links
# the device library and the bridge's include root itself.
target_link_libraries(nevpt2.energy PRIVATE nevpt2.einsum.device nevpt2::einsum::plan
                                            wwr::backend)
