// Suites for nevpt2.device_tensor and the DeviceBuffer it holds -- the first
// GPU suites in the unit tier.
//
// REQUIRES_GPU: every case draws device memory from the process's one
// DeviceResources (nevpt2.test.shared_resources) and copies on its one
// non-blocking stream. With no card visible, each case fails on
// sharedResources()'s Error, naming it; none skips.
//
//   DeviceTensorRoundTripTests  a host Tensor uploaded through DeviceTensor
//                               (and through a bare DeviceBuffer) and
//                               downloaded comes back with the same shape and
//                               the same bits
//   DeviceBufferZeroFillTests   a fresh DeviceBuffer / DeviceTensor::zeros
//                               reads back as all +0.0, even when the pool
//                               hands back memory a previous buffer dirtied
//
// The comparisons here are EXACT, on purpose. "Against a tolerance, never
// bit-identical" (docs/implementation.md, "One generic einsum, 146 call
// sites") is about computed values, which kernels reassociate; nothing here
// computes -- a copy, or a memset to the all-zero
// bit pattern of +0.0, that changes one bit is a bug, not rounding.
//
// TU shape: gtest's header FIRST, then `import std;` and the modules.
#include <gtest/gtest.h>

import std;
import nevpt2.device_tensor;
import nevpt2.test.shared_resources;

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
// One binary per suite file, so a name clash here is a link error.
namespace nevpt2::test::device_tensor {

// A host Tensor whose every element is distinct and not exactly representable
// in fewer bits than a double has (so a float round trip, a dropped element
// or a transposed copy would all show).
Tensor patterned(std::vector<int64_t> dims) {
  Tensor t(std::move(dims));
  for (int64_t i = 0; i < t.size(); ++i) t.flatRef(i) = 0.1 * static_cast<double>(i) - 3.0;
  return t;
}

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

// Reads a whole DeviceBuffer<double> back on res.stream(), synchronizing it
// before the host reads the copy.
std::vector<double> readBack(const DeviceBuffer<double>& buf, const DeviceResources& res) {
  std::vector<double> host(buf.num_elements(), -1.0);
  if (host.empty()) return host;
  gpuCheck(wwrMemcpyAsync(host.data(), buf.data(), buf.size_bytes(), wwrMemcpyDeviceToHost,
                          res.stream()));
  // The host reads `host` next; an async copy into pageable memory may return
  // before it has landed.
  gpuCheck(wwrStreamSynchronize(res.stream()));
  return host;
}

// --- DeviceTensorRoundTripTests -----------------------------------------------

TEST(DeviceTensorRoundTripTests, UploadDownloadKeepsShapeAndBits) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  const Tensor host = patterned({3, 4, 5});
  const DeviceTensor dev = uploadTensor(host, *res);
  EXPECT_TRUE(dev.owning());
  EXPECT_TRUE(isContiguous(dev));
  EXPECT_EQ(dev.dims, host.dims());
  EXPECT_EQ(dev.strides, (std::vector<int64_t>{20, 5, 1}));
  EXPECT_EQ(dev.size(), host.size());

  const Tensor back = downloadTensor(dev, res->stream());
  EXPECT_EQ(back.dims(), host.dims());
  EXPECT_EQ(back.data(), host.data());
}

TEST(DeviceTensorRoundTripTests, ViewDownloadsTheOwnersData) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  const Tensor host = patterned({6, 7});
  const DeviceTensor dev = uploadTensor(host, *res);
  const DeviceTensor view = dev.view();
  EXPECT_FALSE(view.owning());
  EXPECT_EQ(view.data(), dev.data());
  EXPECT_EQ(downloadTensor(view, res->stream()).data(), host.data());
}

TEST(DeviceTensorRoundTripTests, EmptyTensorRoundTrips) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  const Tensor host({4, 0});
  const DeviceTensor dev = uploadTensor(host, *res);
  EXPECT_EQ(dev.size(), 0);
  const Tensor back = downloadTensor(dev, res->stream());
  EXPECT_EQ(back.dims(), host.dims());
  EXPECT_TRUE(back.data().empty());
}

TEST(DeviceTensorRoundTripTests, DeviceEyeIsTheHostIdentity) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  EXPECT_EQ(downloadTensor(deviceEye(5, *res), res->stream()).data(), Tensor::eye(5).data());
}

TEST(DeviceTensorRoundTripTests, BareDeviceBufferRoundTrips) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  const Tensor host = patterned({257});  // not a multiple of any warp/block size
  DeviceBuffer<double> buf(host.data().size(), res->shared_from_this());
  ASSERT_EQ(buf.num_elements(), host.data().size());
  gpuCheck(wwrMemcpyAsync(buf.data(), host.data().data(), buf.size_bytes(),
                          wwrMemcpyHostToDevice, res->stream()));
  EXPECT_EQ(readBack(buf, *res), host.data());

  // A DeviceTensor that takes the buffer over reads back the same data.
  const DeviceTensor dev(std::move(buf), {257});
  EXPECT_TRUE(dev.owning());
  EXPECT_EQ(downloadTensor(dev, res->stream()).data(), host.data());
}

// --- DeviceBufferZeroFillTests ------------------------------------------------

TEST(DeviceBufferZeroFillTests, FreshBufferIsAllZero) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  const DeviceBuffer<double> buf(1000, res->shared_from_this());
  const std::vector<double> host = readBack(buf, *res);
  ASSERT_EQ(host.size(), 1000u);
  for (std::size_t i = 0; i < host.size(); ++i) {
    // +0.0 exactly: all-zero bytes, so not -0.0 either.
    ASSERT_EQ(std::bit_cast<std::uint64_t>(host[i]), 0u) << "element " << i;
  }
}

TEST(DeviceBufferZeroFillTests, ZerosIsAllZero) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  const DeviceTensor dev = DeviceTensor::zeros({8, 9, 10}, *res);
  EXPECT_TRUE(dev.owning());
  EXPECT_EQ(dev.dims, (std::vector<int64_t>{8, 9, 10}));
  const Tensor back = downloadTensor(dev, res->stream());
  EXPECT_EQ(back.data(), std::vector<double>(720, 0.0));
}

// The pool keeps freed memory (kDefaultPoolReleaseThreshold: never release),
// so a buffer allocated right after a same-sized one was freed is likely to
// get that memory back. It must still read as zero: the fill is the buffer's
// doing, not the pool's or the driver's.
TEST(DeviceBufferZeroFillTests, ReusedPoolMemoryIsZeroedAgain) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  constexpr int64_t n = 4096;
  {
    const DeviceTensor dirty = uploadTensor(patterned({n}), *res);
    ASSERT_EQ(downloadTensor(dirty, res->stream()).flat(1), 0.1 - 3.0);
  }  // freed with wwrFreeAsync on res->stream(), back into the pool
  const DeviceTensor fresh = DeviceTensor::zeros({n}, *res);
  EXPECT_EQ(downloadTensor(fresh, res->stream()).data(), std::vector<double>(n, 0.0));
}

}  // namespace nevpt2::test::device_tensor
