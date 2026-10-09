# The generic device einsum: a host planner/launcher module plus the linked-in
# kernel library, sharing one POD (einsum_plan.h) across the host/device line.

# nevpt2.einsum.plan -- einsum_plan.h's own target: an INTERFACE library
# carrying only NEVPT2_SRC_ROOT (src/), so both sides reach it
# root-relative as "einsum/einsum_plan.h".
add_library(nevpt2.einsum.plan INTERFACE)
add_library(nevpt2::einsum::plan ALIAS nevpt2.einsum.plan)
target_include_directories(nevpt2.einsum.plan INTERFACE "${NEVPT2_SRC_ROOT}")

# nevpt2.einsum -- parses numpy-einsum subscripts into an EinsumPlan and
# launches the kernel. Re-exports nevpt2.wwr (every call takes a wwrStream_t)
# and nevpt2.device_tensor (every operand).
add_cxx_module_library(
  NAME nevpt2.einsum
  PRIMARY_INTERFACE einsum.cppm
  PARTITIONS einsum_planner.cppm einsum_launch.cppm
  IMPLEMENTATION einsum.cpp
  LINK_PUBLIC nevpt2::wwr nevpt2::device_tensor
  LINK_PRIVATE nevpt2::einsum::plan nevpt2::profile
)

# device_einsum.cu -- this demo's generic device contraction kernel, the
# three diagonal-slice kernels and the Sijrs slab reduction; energy/ drives
# them. LINKED IN as a device library, like rdm/rdm_accumulate.cu: einsum.cpp
# and energy.cpp call it through device_einsum_bridge.h. No parallel_for (the
# launchers keep their own 256-thread <<<>>>; see the .cu), so it links
# wwr::device for runtime.h directly, plus the plan's include root.
add_device_library(
  NAME nevpt2.einsum.device
  SOURCES device_einsum.cu
  LINK_PRIVATE wwr::device nevpt2::einsum::plan nevpt2::common::device
)

# einsum.cpp includes the bridge (runtime.h in its global module fragment, the
# bridge after its imports in extern "C++"): the same
# include root (from nevpt2::einsum::plan, above) plus wwr::backend for the
# WWR_GPU_BACKEND_* define selected_backend.h reads when there is no
# device-compile pass to go by.
target_link_libraries(nevpt2.einsum PRIVATE nevpt2.einsum.device wwr::backend)
