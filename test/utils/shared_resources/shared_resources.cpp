// The implementation of nevpt2.test.shared_resources: the lazily-created
// DeviceResources and the GoogleTest global environment that releases it.
// gtest is a textual header, not a module, so it comes in through the global
// module fragment.
module;

#include <gtest/gtest.h>

module nevpt2.test.shared_resources;

import std;
import :environment;

namespace nevpt2::test {

const Result<std::shared_ptr<DeviceResources>>& sharedResources() {
  auto& resources = slot();
  if (!resources) resources = DeviceResources::create(0, kDefaultPoolReleaseThreshold);
  return *resources;
}

}  // namespace nevpt2::test
