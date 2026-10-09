# nevpt2.tensor -- the host N-d Tensor. No GPU: golden/ and the host-side
# bookkeeping link it without pulling in a runtime.
add_cxx_module_library(
  NAME nevpt2.tensor
  PRIMARY_INTERFACE tensor.cppm
  IMPLEMENTATION tensor.cpp
  # error_handling: check(), which fromFlat (inline, in the interface) and
  # transpose call -- the abort tier, imports only std.
  LINK_PUBLIC nevpt2::common nevpt2::error_handling
)

# nevpt2.device_tensor -- the device-resident twin. Re-exports nevpt2.tensor
# (its upload/download signatures name Tensor) and nevpt2.device_resources
# (allocation takes the DeviceResources whose pool it draws on;
# copies take a wwrStream_t), so both are PUBLIC.
add_cxx_module_library(
  NAME nevpt2.device_tensor
  PRIMARY_INTERFACE device_tensor.cppm
  PARTITIONS device_tensor_row_major.cppm
  IMPLEMENTATION device_tensor.cpp
  LINK_PUBLIC nevpt2::tensor nevpt2::device_resources
)
