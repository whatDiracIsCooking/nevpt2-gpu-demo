// nevpt2.test.shared_resources -- the one nevpt2::DeviceResources every GPU
// suite in a test process shares.
//
// A demo's main() creates exactly one DeviceResources and passes it down
// (docs/performance.md, "One stream"). A test binary has no main() of its
// own (GTest::gtest_main), so this is where its one lives: every case in the
// process gets the SAME one -- the same non-blocking stream, pool, BLAS and
// solver handles -- so everything a suite issues is on that one stream.
//
// Two properties matter, and neither is incidental:
//
//  - Lazy. It is created on the FIRST sharedResources() call, not at process
//    start, so a host-only suite in the same binary never touches a device
//    and `ctest -LE gpu` stays card-free.
//
//  - Released inside RUN_ALL_TESTS. A function-local static alone would be
//    destroyed at exit, after main returns and possibly after the GPU runtime
//    has gone -- destroying a stream and pool against a dead runtime is the
//    classic shutdown crash. The implementation unit registers a
//    ::testing::Environment whose TearDown() drops this process's reference
//    after the last test, before main returns. (A DeviceBuffer co-owns the
//    DeviceResources it was drawn from, so a buffer a test still held would
//    keep it alive past TearDown; none does -- every buffer is a test-local.)
//
// Usage, in a REQUIRES_GPU suite:
//
//   const auto& res = nevpt2::test::sharedResources();
//   ASSERT_TRUE(res.has_value()) << res.error().message;
//   const nevpt2::DeviceResources& dev = **res;
//
// create() returns a Result (no card, a bad index or no pool
// support is the environment's fault, not ours), so a card-less run FAILS the
// test with that message instead of aborting the binary. The failure is cached
// like a success: one create() per process, not one per case.
export module nevpt2.test.shared_resources;

import std;
// Re-exported: sharedResources() returns a Result over DeviceResources, and a
// caller that can name the function can use what it returns (stream(), the
// DeviceBuffers it backs) without a second import -- import is not transitive.
export import nevpt2.device_resources;

export namespace nevpt2::test {

// The process's one DeviceResources on device 0 (the demos' default release
// threshold), created on the first call, or the Error creation returned. The
// same object -- or the same Error -- on every call until the environment's
// TearDown(). Do not call it from a host-only suite: it brings the device up.
const Result<std::shared_ptr<DeviceResources>>& sharedResources();

}  // namespace nevpt2::test
