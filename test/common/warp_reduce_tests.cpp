// WarpReduceTests -- nevpt2::device::warp_reduce (src/common/warp_reduce.cuh)
// on the card.
//
// REQUIRES_GPU: every case launches warp_reduce_kernels.cu on the process's one
// DeviceResources stream (nevpt2.test.shared_resources).
//
// What each case pins down, for every partial tile nactive = 1 .. W (W the
// build's kWarpSize, taken from the device pass, never assumed):
//   - every lane holds the fold (the closing broadcast), not just lane 0;
//   - inactive lanes are ignored: they hold NaN, which survives + and max_nan,
//     or the interval probe's BAD, which absorbs -- so one folded in shows;
//   - the fold is in lane order: the interval probe (interval_probe.h) is
//     associative but not commutative, and comes out [0, nactive - 1] only if
//     every combine appended the right neighbour;
//   - warps other than a block's first fold independently (four to a block).
//
// The comparisons are EXACT, on purpose: the sums' inputs are small integers
// (patternValue), whose sum is exact in any order, and max / the probe compute
// nothing inexact. A tolerance here would hide a dropped or doubled lane.
//
// TU shape: gtest's header and runtime.h FIRST (runtime.h before the imports,
// as a module's global module fragment would have it, so the bridge's own
// #include of it below is a #pragma once no-op), then `import`s, then the
// bridge, which needs nevpt2.wwr's wwrError_t.
#include <gtest/gtest.h>

#include <runtime.h>

import std;
import nevpt2.wwr;
import nevpt2.test.shared_resources;

#include "interval_probe.h"
#include "reduce_test_support.h"
#include "warp_reduce_bridge.h"

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
// One binary per suite file, so a name clash here is a link error.
namespace nevpt2::test::warp_reduce {

using test::download;
using test::expectEveryLane;
using test::upload;

// The serial references, on the host.
const auto kAdd = [](const double a, const double b) { return a + b; };
const auto kMaxNan = [](const double a, const double b) {
  return (std::isnan(a) || a > b) ? a : b;
};

// The width every fold here has: the device pass's kWarpSize.
unsigned int warpSize() { return device::warpReduceWarpSize(); }

// The compiled warp size is the card's: warp_reduce's ladder and the HIP
// tile-shuffle widths are only right if they agree (32 on the RTX 3080 and on
// the RX 9060 XT, gfx1200 being wave32).
TEST(WarpReduceTests, CompiledWarpSizeIsTheDevices) {
  const DeviceResources* res = test::resourcesOrFail();
  ASSERT_NE(res, nullptr);
  wwrDeviceProp prop{};
  gpuCheck(wwrGetDeviceProperties(&prop, 0));  // sharedResources() is device 0
  EXPECT_EQ(static_cast<unsigned int>(prop.warpSize), warpSize());
}

TEST(WarpReduceTests, SumEveryPartialTile) {
  const DeviceResources* res = test::resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const unsigned int w = warpSize();
  const std::vector<unsigned int> nactive = test::everyPartialTile(w);
  const std::vector<double> in = test::foldInputs(nactive, w, test::kPoison, test::patternValue);

  const auto dIn = upload(in, *res);
  const auto dN = upload(nactive, *res);
  DeviceBuffer<double> dOut(in.size(), res->shared_from_this());
  gpuCheck(device::warpReduceSum(res->stream(), dIn.data(), dOut.data(), dN.data(), w));
  expectEveryLane(download(dOut, *res), test::serialFolds(in, nactive, w, kAdd), w, "sum");
}

TEST(WarpReduceTests, MaxNanEveryPartialTile) {
  const DeviceResources* res = test::resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const unsigned int w = warpSize();
  const std::vector<unsigned int> nactive = test::everyPartialTile(w);
  const std::vector<double> in = test::foldInputs(nactive, w, test::kPoison, test::patternValue);

  const auto dIn = upload(in, *res);
  const auto dN = upload(nactive, *res);
  DeviceBuffer<double> dOut(in.size(), res->shared_from_this());
  gpuCheck(device::warpReduceMaxNan(res->stream(), dIn.data(), dOut.data(), dN.data(), w));
  const std::vector<double> got = download(dOut, *res);
  for (const double v : got) ASSERT_FALSE(std::isnan(v)) << "an inactive lane's NaN got in";
  expectEveryLane(got, test::serialFolds(in, nactive, w, kMaxNan), w, "max_nan");
}

// A NaN in an ACTIVE lane must win wherever it sits: warp f carries it in lane
// f, every lane active.
TEST(WarpReduceTests, MaxNanKeepsAnActiveNaNInAnyLane) {
  const DeviceResources* res = test::resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const unsigned int w = warpSize();
  const std::vector<unsigned int> nactive(w, w);
  const std::vector<double> in = test::foldInputs(
      nactive, w, 0.0, [](const unsigned int f, const unsigned int t) {
        return t == f ? test::kPoison : test::patternValue(f, t);
      });

  const auto dIn = upload(in, *res);
  const auto dN = upload(nactive, *res);
  DeviceBuffer<double> dOut(in.size(), res->shared_from_this());
  gpuCheck(device::warpReduceMaxNan(res->stream(), dIn.data(), dOut.data(), dN.data(), w));
  const std::vector<double> got = download(dOut, *res);
  for (std::size_t i = 0; i < got.size(); ++i) {
    ASSERT_TRUE(std::isnan(got[i])) << "fold " << i / w << " lane " << i % w;
  }
}

// The probe itself, on the host: what makes it a lane-order test.
TEST(WarpReduceTests, IntervalProbeIsOrderSensitive) {
  const device::IntervalOp op{};
  EXPECT_EQ(op(device::encodeInterval(0, 0), device::encodeInterval(1, 1)),
            device::encodeInterval(0, 1));
  EXPECT_EQ(op(device::encodeInterval(1, 1), device::encodeInterval(0, 0)), device::kIntervalBad);
  EXPECT_EQ(op(device::encodeInterval(0, 1), device::kIntervalBad), device::kIntervalBad);
  // Associative on a run of three: both bracketings give [3, 5].
  const auto a = device::encodeInterval(3, 3);
  const auto b = device::encodeInterval(4, 4);
  const auto c = device::encodeInterval(5, 5);
  EXPECT_EQ(op(op(a, b), c), op(a, op(b, c)));
  EXPECT_EQ(op(op(a, b), c), device::encodeInterval(3, 5));
}

// Lane t holds [t, t], inactive lanes BAD: only an in-order fold of exactly
// the active lanes gives [0, nactive - 1].
TEST(WarpReduceTests, FoldsInLaneOrderEveryPartialTile) {
  const DeviceResources* res = test::resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const unsigned int w = warpSize();
  const std::vector<unsigned int> nactive = test::everyPartialTile(w);
  const std::vector<unsigned long long> in = test::foldInputs(
      nactive, w, device::kIntervalBad,
      [](unsigned int, const unsigned int t) { return device::encodeInterval(t, t); });

  std::vector<unsigned long long> want(in.size());
  for (unsigned int f = 0; f < w; ++f) {
    for (unsigned int t = 0; t < w; ++t) want[f * w + t] = device::encodeInterval(0, nactive[f] - 1);
  }
  // The host's serial fold agrees, so the expectation is the op's, not ours.
  ASSERT_EQ(test::serialFolds(in, nactive, w, device::IntervalOp{}), want);

  const auto dIn = upload(in, *res);
  const auto dN = upload(nactive, *res);
  DeviceBuffer<unsigned long long> dOut(in.size(), res->shared_from_this());
  gpuCheck(device::warpReduceInterval(res->stream(), dIn.data(), dOut.data(), dN.data(), w));
  expectEveryLane(download(dOut, *res), want, w, "interval");
}

}  // namespace nevpt2::test::warp_reduce
