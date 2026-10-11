# nevpt2.energy_finish -- the class energies' pure host arithmetic: the slab
# walk (forBatches), the SC finish (normToEnergy), the PC finish and
# denominator check (finishSingle, checkDenominator) and the PC result types
# they fill. No GPU, so the unit tier tests it card-free
# (test/energy_finish/); nevpt2.energy re-exports it.
add_cxx_module_library(
  NAME nevpt2.energy_finish
  PRIMARY_INTERFACE energy_finish.cppm
  LINK_PUBLIC nevpt2::tensor
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
# sees. (The metric / Dyall-Hamiltonian intermediates ARE exported, from
# energy.cppm: nevpt2.gradient differentiates the same class einsums.)
add_cxx_module_library(
  NAME nevpt2.energy
  PRIMARY_INTERFACE energy.cppm
  PARTITIONS energy_shared.cppm energy_sc_finish.cppm energy_sc_classes.cppm
             energy_pc_solve.cppm energy_pc_classes.cppm
  IMPLEMENTATION energy.cpp energy_pc.cpp
  LINK_PUBLIC nevpt2::einsum nevpt2::energy_finish
  LINK_PRIVATE nevpt2::profile wwr::solver wwr::extension::solver
)

# energy_Sijrs launches device_einsum.cu's slab reduction directly through
# einsum/device_einsum_bridge.h (after its imports, in extern "C++"), so it links
# the device library and the bridge's include root itself.
target_link_libraries(nevpt2.energy PRIVATE nevpt2.einsum.device nevpt2::einsum::plan
                                            wwr::backend)
