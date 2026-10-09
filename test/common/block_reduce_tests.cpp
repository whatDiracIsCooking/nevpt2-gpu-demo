// BlockReduceTreeTests / BlockReduceWarpsTests -- the two
// nevpt2::device::block_reduce overloads (src/common/block_reduce.cuh) on the
// card.
//
// REQUIRES_GPU: every case launches block_reduce_kernels.cu on the process's
// one DeviceResources stream (nevpt2.test.shared_resources).
//
//   BlockReduceTreeTests   block_reduce<kBlock>(v, op, nactive), the
//                          shared-memory tree, for blocks of 1 to 1024
//                          threads; its interval probe folds a class type
//   BlockReduceWarpsTests  block_reduce(block, partials, v, op, nactive),
//                          warp_reduce then across warps, for 1, 2, 4 and the
//                          most (1024 / W) warps a block can hold
//
// Each shape runs one block per partial tile, nactive = 1 .. threads, with
// every thread's result checked (the fold is broadcast), the inactive threads
// poisoned (NaN, or the interval probe's BAD), and the non-commutative
// interval probe (interval_probe.h) proving thread order. W is the device
// pass's kWarpSize, read back through the bridge, never assumed.
//
// The comparisons are EXACT, on purpose -- see warp_reduce_tests.cpp: the
// sums' inputs are small integers, exact in any order.
//
// TU shape as in warp_reduce_tests.cpp.
#include <gtest/gtest.h>

#include <runtime.h>

import std;
import nevpt2.wwr;
import nevpt2.test.shared_resources;

#include "block_reduce_bridge.h"
#include "interval_probe.h"
#include "reduce_test_support.h"

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
// One binary per suite file, so a name clash here is a link error.
namespace nevpt2::test::block_reduce {

using test::download;
using test::expectEveryLane;
using test::upload;

const auto kAdd = [](const double a, const double b) { return a + b; };
const auto kMaxNan = [](const double a, const double b) {
  return (std::isnan(a) || a > b) ? a : b;
};

// The bridge launchers all share one signature shape:
// (stream, in, out, nactive, nblocks, shape).
using DoubleLaunch = wwrError_t (*)(wwrStream_t, const double*, double*, const unsigned int*,
                                    unsigned int, unsigned int);
using IntervalLaunch = wwrError_t (*)(wwrStream_t, const unsigned long long*,
                                      unsigned long long*, const unsigned int*, unsigned int,
                                      unsigned int);

// The tree's block sizes: the degenerate 1 and 2, one and several warps' worth,
// and the 1024-thread cap.
const std::vector<unsigned int> kTreeBlocks = {1, 2, 32, 64, 256, 1024};

// The warp overload's warps per block: one (the kNumWarps == 1 branch), a few,
// and the most a 1024-thread block holds.
std::vector<unsigned int> warpCounts() {
  const unsigned int most = 1024 / device::blockReduceWarpSize();
  std::vector<unsigned int> n = {1, 2, 4};
  if (most > 4) n.push_back(most);
  return n;
}

// One block per partial tile of `threads`, under a double op; compares every
// thread with the host's serial fold.
template <typename HostOp>
void runDouble(const DoubleLaunch launch, const unsigned int threads, const unsigned int shape,
               const HostOp& hostOp, const std::string& what) {
  const DeviceResources* res = test::resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const std::vector<unsigned int> nactive = test::everyPartialTile(threads);
  const std::vector<double> in =
      test::foldInputs(nactive, threads, test::kPoison, test::patternValue);

  const auto dIn = upload(in, *res);
  const auto dN = upload(nactive, *res);
  DeviceBuffer<double> dOut(in.size(), res->shared_from_this());
  gpuCheck(launch(res->stream(), dIn.data(), dOut.data(), dN.data(),
                  static_cast<unsigned int>(nactive.size()), shape));
  expectEveryLane(download(dOut, *res), test::serialFolds(in, nactive, threads, hostOp), threads,
                  what);
}

// One block per partial tile; thread t holds [t, t], inactive threads BAD.
void runInterval(const IntervalLaunch launch, const unsigned int threads,
                 const unsigned int shape, const std::string& what) {
  const DeviceResources* res = test::resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const std::vector<unsigned int> nactive = test::everyPartialTile(threads);
  const std::vector<unsigned long long> in = test::foldInputs(
      nactive, threads, device::kIntervalBad,
      [](unsigned int, const unsigned int t) { return device::encodeInterval(t, t); });
  std::vector<unsigned long long> want(in.size());
  for (unsigned int f = 0; f < nactive.size(); ++f) {
    for (unsigned int t = 0; t < threads; ++t) {
      want[f * threads + t] = device::encodeInterval(0, nactive[f] - 1);
    }
  }

  const auto dIn = upload(in, *res);
  const auto dN = upload(nactive, *res);
  DeviceBuffer<unsigned long long> dOut(in.size(), res->shared_from_this());
  gpuCheck(launch(res->stream(), dIn.data(), dOut.data(), dN.data(),
                  static_cast<unsigned int>(nactive.size()), shape));
  expectEveryLane(download(dOut, *res), want, threads, what);
}

// Every block full, block f carrying a NaN in thread f (f < threads): the NaN
// must win wherever it sits. `nblocks` <= threads.
void runActiveNaN(const DoubleLaunch launch, const unsigned int threads,
                  const unsigned int shape, const std::string& what) {
  const DeviceResources* res = test::resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const std::vector<unsigned int> nactive(threads, threads);
  const std::vector<double> in = test::foldInputs(
      nactive, threads, 0.0, [](const unsigned int f, const unsigned int t) {
        return t == f ? test::kPoison : test::patternValue(f, t);
      });

  const auto dIn = upload(in, *res);
  const auto dN = upload(nactive, *res);
  DeviceBuffer<double> dOut(in.size(), res->shared_from_this());
  gpuCheck(launch(res->stream(), dIn.data(), dOut.data(), dN.data(), threads, shape));
  const std::vector<double> got = download(dOut, *res);
  std::size_t notNaN = 0;
  for (const double v : got) notNaN += std::isnan(v) ? 0 : 1;
  EXPECT_EQ(notNaN, 0u) << what << ": threads whose fold dropped the NaN";
}

// --- BlockReduceTreeTests -----------------------------------------------------

TEST(BlockReduceTreeTests, SumEveryPartialBlock) {
  for (const unsigned int b : kTreeBlocks) {
    runDouble(device::blockReduceTreeSum, b, b, kAdd, "tree sum, block " + std::to_string(b));
  }
}

TEST(BlockReduceTreeTests, MaxNanEveryPartialBlock) {
  for (const unsigned int b : kTreeBlocks) {
    runDouble(device::blockReduceTreeMaxNan, b, b, kMaxNan,
              "tree max_nan, block " + std::to_string(b));
  }
}

TEST(BlockReduceTreeTests, MaxNanKeepsAnActiveNaNInAnyThread) {
  for (const unsigned int b : kTreeBlocks) {
    runActiveNaN(device::blockReduceTreeMaxNan, b, b,
                 "tree max_nan, block " + std::to_string(b));
  }
}

// The class-type path: the probe folded as the Interval struct.
TEST(BlockReduceTreeTests, FoldsAStructInThreadOrderEveryPartialBlock) {
  for (const unsigned int b : kTreeBlocks) {
    runInterval(device::blockReduceTreeInterval, b, b,
                "tree interval, block " + std::to_string(b));
  }
}

// Two calls back to back through the tree's one static buffer: the second
// (of -v) must not see the first's values.
TEST(BlockReduceTreeTests, BackToBackCallsReuseTheBuffer) {
  const DeviceResources* res = test::resourcesOrFail();
  ASSERT_NE(res, nullptr);
  for (const unsigned int b : kTreeBlocks) {
    const std::vector<unsigned int> nactive = test::everyPartialTile(b);
    const std::vector<double> in = test::foldInputs(nactive, b, test::kPoison, test::patternValue);
    const std::vector<double> want = test::serialFolds(in, nactive, b, kAdd);
    std::vector<double> wantNeg(want.size());
    for (std::size_t i = 0; i < want.size(); ++i) wantNeg[i] = -want[i];

    const auto dIn = upload(in, *res);
    const auto dN = upload(nactive, *res);
    DeviceBuffer<double> dOut(in.size(), res->shared_from_this());
    DeviceBuffer<double> dNeg(in.size(), res->shared_from_this());
    gpuCheck(device::blockReduceTreeSumTwice(res->stream(), dIn.data(), dOut.data(), dNeg.data(),
                                             dN.data(), b, b));
    expectEveryLane(download(dOut, *res), want, b, "first sum, block " + std::to_string(b));
    expectEveryLane(download(dNeg, *res), wantNeg, b, "second sum, block " + std::to_string(b));
  }
}

// --- BlockReduceWarpsTests ----------------------------------------------------

TEST(BlockReduceWarpsTests, SumEveryPartialBlock) {
  const unsigned int w = device::blockReduceWarpSize();
  for (const unsigned int nw : warpCounts()) {
    runDouble(device::blockReduceWarpsSum, nw * w, nw, kAdd,
              "warps sum, " + std::to_string(nw) + " warps");
  }
}

TEST(BlockReduceWarpsTests, MaxNanEveryPartialBlock) {
  const unsigned int w = device::blockReduceWarpSize();
  for (const unsigned int nw : warpCounts()) {
    runDouble(device::blockReduceWarpsMaxNan, nw * w, nw, kMaxNan,
              "warps max_nan, " + std::to_string(nw) + " warps");
  }
}

TEST(BlockReduceWarpsTests, MaxNanKeepsAnActiveNaNInAnyThread) {
  const unsigned int w = device::blockReduceWarpSize();
  for (const unsigned int nw : warpCounts()) {
    runActiveNaN(device::blockReduceWarpsMaxNan, nw * w, nw,
                 "warps max_nan, " + std::to_string(nw) + " warps");
  }
}

TEST(BlockReduceWarpsTests, FoldsInThreadOrderEveryPartialBlock) {
  const unsigned int w = device::blockReduceWarpSize();
  for (const unsigned int nw : warpCounts()) {
    runInterval(device::blockReduceWarpsInterval, nw * w, nw,
                "warps interval, " + std::to_string(nw) + " warps");
  }
}

}  // namespace nevpt2::test::block_reduce
