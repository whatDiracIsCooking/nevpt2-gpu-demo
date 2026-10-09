// GPU suites for nevpt2::DeviceResources itself -- the one stream, pool, BLAS
// handle and dense-solver handle every GPU call goes through. The host-only
// helpers (parsePoolThreshold, toMB) are pool_threshold_tests' job.
//
// REQUIRES_GPU: every case reads the process's one DeviceResources
// (nevpt2.test.shared_resources). With no card visible, each case fails on
// sharedResources()'s Error, naming it; none skips.
//
//   PoolHighWaterTests       a DeviceBuffer raises the pool's used-memory
//                            high-water by at least its size; freeing it and
//                            synchronizing the stream returns `used current`
//                            to where it was, and leaves the high-water put
//   ReleaseThresholdTests    releaseThreshold() returns what create() was
//                            given -- the `max` default and a byte count --
//                            and the pool itself carries that threshold
//   StreamBindingTests       stream() is non-blocking; blas() and solver()
//                            are bound to stream()
//   PrintPoolHighWaterTests  printPoolHighWater's two formats: "release
//                            threshold max" and "release threshold N bytes"
//
// The shared DeviceResources may already have served other cases in this
// process, so the pool checks assert on DELTAS from what the pool reports at
// the start of the case, never on absolute values.
//
// A second DeviceResources: create() has no once-per-process guard (it
// selects the device and builds a fresh stream, pool and handles each call),
// so the non-default-threshold cases make their own, with a byte-count
// threshold, scoped to the case. It allocates nothing, so nothing is ever
// issued on its stream, and the one-stream rule for the shared one is
// untouched.
//
// TU shape: gtest's header FIRST, then `import std;` and the modules.
#include <gtest/gtest.h>

import std;
import nevpt2.test.shared_resources;

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::device_resources {

// A non-default threshold for the byte-count branch: not a power of two, so a
// truncation or unit conversion anywhere would show.
inline constexpr std::uint64_t kByteThreshold = 123'456'789;

// The process's DeviceResources, or nullptr after recording the creation
// Error as this test's failure (callers ASSERT_NE on it).
const DeviceResources* resourcesOrFail() {
  const auto& res = test::sharedResources();
  if (!res.has_value()) {
    ADD_FAILURE() << "DeviceResources::create failed: " << kindName(res.error().kind) << ": "
                  << res.error().message;
    return nullptr;
  }
  return res->get();
}

// A DeviceResources of this case's own on device 0 with `threshold`, or
// nullptr after recording the creation Error as this test's failure.
std::shared_ptr<DeviceResources> ownResourcesOrFail(std::uint64_t threshold) {
  auto res = DeviceResources::create(0, threshold);
  if (!res.has_value()) {
    ADD_FAILURE() << "DeviceResources::create failed: " << kindName(res.error().kind) << ": "
                  << res.error().message;
    return nullptr;
  }
  return std::move(*res);
}

// A 64-bit pool attribute (used current, release threshold), read the same
// way DeviceResources::highWater() reads its two.
std::uint64_t poolAttr(const DeviceResources& res, wwrMemPoolAttr attr) {
  std::uint64_t v = 0;
  gpuCheck(wwrMemPoolGetAttribute(res.pool(), attr, &v));
  return v;
}

// --- PoolHighWaterTests --------------------------------------------------------

TEST(PoolHighWaterTests, AllocationRaisesHighWaterAndFreeLowersUsed) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  // Every free a previous case queued must have landed before `used current`
  // is a stable baseline.
  gpuCheck(wwrStreamSynchronize(res->stream()));
  const PoolHighWater before = res->highWater();
  const std::uint64_t usedBefore = poolAttr(*res, wwrMemPoolAttrUsedMemCurrent);
  ASSERT_GE(before.used, usedBefore);

  // Big enough to push `used` past the old high-water whatever earlier cases
  // left it at, so the high-water must strictly rise.
  constexpr std::uint64_t kHeadroom = 8ull << 20;
  const std::uint64_t bytes = (before.used - usedBefore) + kHeadroom;
  const std::size_t n = static_cast<std::size_t>((bytes + sizeof(double) - 1) / sizeof(double));
  {
    const DeviceBuffer<double> buf(n, res->shared_from_this());
    ASSERT_EQ(buf.num_elements(), n);
    // The allocation is stream-ordered; let it (and its zero fill) run before
    // reading the pool's accounting of it.
    gpuCheck(wwrStreamSynchronize(res->stream()));

    const std::uint64_t usedWith = poolAttr(*res, wwrMemPoolAttrUsedMemCurrent);
    EXPECT_GE(usedWith - usedBefore, n * sizeof(double));
    const PoolHighWater during = res->highWater();
    EXPECT_GE(during.used, usedWith);
    EXPECT_GT(during.used, before.used);
    EXPECT_GE(during.used - before.used, kHeadroom);
    // Reserved covers everything allocated.
    EXPECT_GE(during.reserved, during.used);
  }
  // buf's destructor queued wwrFreeAsync on the stream; the pool sees the free
  // only once the stream reaches it.
  gpuCheck(wwrStreamSynchronize(res->stream()));

  EXPECT_EQ(poolAttr(*res, wwrMemPoolAttrUsedMemCurrent), usedBefore);
  // A high-water mark does not fall when memory is freed.
  const PoolHighWater after = res->highWater();
  EXPECT_GT(after.used, before.used);
  EXPECT_GE(after.reserved, before.reserved);
}

// --- ReleaseThresholdTests -----------------------------------------------------

TEST(ReleaseThresholdTests, DefaultIsMaxAndPoolCarriesIt) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  // sharedResources() creates with the demos' default.
  EXPECT_EQ(res->releaseThreshold(), kDefaultPoolReleaseThreshold);
  EXPECT_EQ(res->releaseThreshold(), std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(poolAttr(*res, wwrMemPoolAttrReleaseThreshold), kDefaultPoolReleaseThreshold);
}

TEST(ReleaseThresholdTests, ByteCountRoundTripsAndPoolCarriesIt) {
  const auto res = ownResourcesOrFail(kByteThreshold);
  ASSERT_NE(res, nullptr);

  EXPECT_EQ(res->releaseThreshold(), kByteThreshold);
  EXPECT_EQ(poolAttr(*res, wwrMemPoolAttrReleaseThreshold), kByteThreshold);
  EXPECT_EQ(res->dev_idx(), 0);
}

// --- StreamBindingTests --------------------------------------------------------

TEST(StreamBindingTests, StreamIsNonBlocking) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  ASSERT_NE(res->stream(), wwrStream_t{});
  unsigned int flags = 0;
  gpuCheck(wwrStreamGetFlags(res->stream(), &flags));
  EXPECT_EQ(flags & wwrStreamNonBlocking, wwrStreamNonBlocking) << "flags = " << flags;
}

TEST(StreamBindingTests, BlasHandleIsBoundToTheStream) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  wwrStream_t bound{};
  gpuCheck(wwrblasGetStream(res->blas(), &bound));
  EXPECT_EQ(bound, res->stream());
}

TEST(StreamBindingTests, SolverHandleIsBoundToTheStream) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  wwrStream_t bound{};
  gpuCheck(wwrsolverDnGetStream(res->solver(), &bound));
  EXPECT_EQ(bound, res->stream());
}

// --- PrintPoolHighWaterTests ---------------------------------------------------

// The high-water is read before printing; nothing allocates in between, so
// the line must carry exactly these values.
TEST(PrintPoolHighWaterTests, MaxThresholdBranch) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  ASSERT_EQ(res->releaseThreshold(), kDefaultPoolReleaseThreshold);

  const PoolHighWater hw = res->highWater();
  testing::internal::CaptureStdout();
  printPoolHighWater("at the test", *res);
  const std::string out = testing::internal::GetCapturedStdout();

  EXPECT_EQ(out, std::format("device pool high-water at the test: used {:.1f} MB, reserved "
                             "{:.1f} MB (release threshold max)\n",
                             toMB(hw.used), toMB(hw.reserved)));
}

TEST(PrintPoolHighWaterTests, ByteThresholdBranch) {
  const auto res = ownResourcesOrFail(kByteThreshold);
  ASSERT_NE(res, nullptr);

  const PoolHighWater hw = res->highWater();
  testing::internal::CaptureStdout();
  printPoolHighWater("after the RDM build", *res);
  const std::string out = testing::internal::GetCapturedStdout();

  EXPECT_EQ(out, std::format("device pool high-water after the RDM build: used {:.1f} MB, "
                             "reserved {:.1f} MB (release threshold 123456789 bytes)\n",
                             toMB(hw.used), toMB(hw.reserved)));
}

}  // namespace nevpt2::test::device_resources
