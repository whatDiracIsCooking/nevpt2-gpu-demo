// nevpt2.test.shared_resources:environment -- the lazily-filled DeviceResources slot and the
// GoogleTest global environment that empties it.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :environment;` and from no importer of nevpt2.test.shared_resources.
// (Used by shared_resources.cpp.)
module;

#include <gtest/gtest.h>

module nevpt2.test.shared_resources:environment;

import std;
import nevpt2.test.shared_resources;

namespace nevpt2::test {

// Empty until the first sharedResources() call, and again after TearDown().
// Function-local, so it is constructed on first use and never touched by a
// binary that does not call sharedResources().
std::optional<Result<std::shared_ptr<DeviceResources>>>& slot() {
  static std::optional<Result<std::shared_ptr<DeviceResources>>> resources;
  return resources;
}

// SetUp() is the base no-op on purpose: creation is lazy (see the interface).
// TearDown() runs inside RUN_ALL_TESTS after the last test, while the runtime
// is still up. DeviceResources' destructor synchronizes its stream first, so
// every queued wwrFreeAsync has landed before the pool and stream go.
class SharedResourcesEnvironment : public ::testing::Environment {
 public:
  void TearDown() override { slot().reset(); }
};

// Registered during static initialization, before RUN_ALL_TESTS. This unit is
// a static-library member, linked into a binary only if the binary references
// sharedResources(), so a host-only binary neither links it nor registers the
// environment. gtest takes ownership of the Environment; its API has no
// non-owning overload, hence the raw new.
[[maybe_unused]] const ::testing::Environment* const kEnvironmentRegistered =
    ::testing::AddGlobalTestEnvironment(new SharedResourcesEnvironment);

}  // namespace nevpt2::test
