# nevpt2.device_resources -- DeviceResources: the one non-blocking stream,
# memory pool, BLAS handle and dense-solver handle both demos issue
# every GPU call through, and DeviceBuffer<T>, the RAII device
# allocation backed by it. The WarpWraps wrapper types it is built
# from, with nevpt2::Abort in every slot, are nevpt2.wwr's.
add_cxx_module_library(
  NAME nevpt2.device_resources
  PRIMARY_INTERFACE device_resources.cppm
  PARTITIONS device_select.cppm
  IMPLEMENTATION device_resources.cpp
  # PUBLIC: the interface re-exports nevpt2.wwr (which carries the wrapper
  # instantiations DeviceResources' private members are, and re-exports
  # memory_buffer for DeviceBuffer<T>), wwr.blas (blas() returns a
  # wwrblasHandle_t) and wwr.solver (solver(), a wwrsolverDnHandle_t); every
  # importer needs the wrappers' module interfaces to see the class's layout.
  LINK_PUBLIC nevpt2::wwr wwr::blas wwr::solver wwr::extension::runtime
              wwr::extension::blas wwr::extension::solver wwr::extension::memory_buffer
)
