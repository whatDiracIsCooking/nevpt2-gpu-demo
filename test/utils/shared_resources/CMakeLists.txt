# nevpt2.test.shared_resources -- one nevpt2::DeviceResources per test process,
# created on first use and released by a registered GoogleTest global
# environment.
#
# PUBLIC nevpt2::device_resources: the interface re-exports it -- what
# sharedResources() hands back is a Result<std::shared_ptr<DeviceResources>>.
# PRIVATE GTest::gtest: only the implementation unit includes <gtest/gtest.h>
# (to register the environment); the interface names none of gtest.
add_cxx_module_library(
  NAME nevpt2.test.shared_resources
  PRIMARY_INTERFACE shared_resources.cppm
  PARTITIONS environment.cppm
  IMPLEMENTATION shared_resources.cpp
  LINK_PUBLIC nevpt2::device_resources
  LINK_PRIVATE GTest::gtest
)
