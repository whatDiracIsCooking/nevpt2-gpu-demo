// Suites for nevpt2.profile -- the --profile recorder: time(), Section,
// setEnabled() and report().
//
// REQUIRES_GPU: a recorded span is an event pair on a stream, here the
// process's one non-blocking stream (nevpt2.test.shared_resources). With no
// card visible, each case fails on sharedResources()'s Error, naming it; none
// skips.
//
//   ProfileDisabledTests     off (the default): time() calls f exactly once
//                            and records nothing in the open Section
//   ProfileSectionTests      on: a span lands in the innermost open Section,
//                            and a closed inner Section restores the outer
//   ProfileReportTests       report(section) groups by label, counts calls,
//                            keeps outSize / contractedSize from the LAST
//                            call, and forgets what it reported
//   ProfileSetEnabledTests   setEnabled(false) after use stops recording
//   ProfileDeathTest         a span with no Section open is our bug: check()
//                            aborts naming it
//
// nevpt2.profile's state is process-global (g_enabled, the span list, the
// innermost Section's name). Every fixture case starts from what the process
// starts with -- disabled, no Section open -- asserts it, and leaves it so:
// TearDown disables profiling and drains (report()s) every section the case
// used, after a stream sync, so no span or event outlives the case whatever
// order the cases run in or wherever one stops. Sections are RAII, so none
// outlives its case. Each case uses section names of its own.
//
// report() reads events back, so every case synchronizes the stream before
// it (nevpt2.profile's header): an unfinished event makes the elapsed time
// undefined. The spans here time no GPU work, so totalMs is only checked to
// be finite and non-negative -- the timings are not this suite's claim.
//
// TU shape: gtest's header FIRST (then <cstdio>, for stderr), then
// `import std;` and the modules.
#include <gtest/gtest.h>

#include <cstdio>  // stderr, a macro `import std` does not carry

import std;
import nevpt2.profile;
import nevpt2.test.shared_resources;

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::profile_suite {

namespace prof = nevpt2::profile;

// The row labelled `label`, or nullptr.
const ProfileRow* findRow(const std::vector<ProfileRow>& rows, std::string_view label) {
  const auto it = std::ranges::find(rows, label, &ProfileRow::label);
  return it == rows.end() ? nullptr : &*it;
}

class ProfileFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto& res = test::sharedResources();
    ASSERT_TRUE(res.has_value()) << "DeviceResources::create failed: "
                                 << kindName(res.error().kind) << ": " << res.error().message;
    res_ = res->get();
    // The process's starting state, which every case leaves behind.
    ASSERT_FALSE(prof::enabled());
  }

  void TearDown() override {
    prof::setEnabled(false);
    if (res_ == nullptr) return;
    // Every span's stop event must have completed before report() reads it.
    gpuCheck(wwrStreamSynchronize(res_->stream()));
    for (const std::string& section : used_) (void)prof::report(section);
  }

  // Registers `name` for TearDown's drain and returns it.
  std::string_view section(std::string_view name) { return used_.emplace_back(name); }

  wwrStream_t stream() const { return res_->stream(); }

  // Synchronizes the stream, then report(section).
  std::vector<ProfileRow> syncAndReport(std::string_view name) {
    gpuCheck(wwrStreamSynchronize(stream()));
    return prof::report(name);
  }

 private:
  const DeviceResources* res_ = nullptr;
  std::deque<std::string> used_;  // deque: section() hands out views into it
};

using ProfileDisabledTests = ProfileFixture;
using ProfileSectionTests = ProfileFixture;
using ProfileReportTests = ProfileFixture;
using ProfileSetEnabledTests = ProfileFixture;

// --- ProfileDisabledTests --------------------------------------------------------

TEST_F(ProfileDisabledTests, TimeCallsFOnceAndRecordsNothing) {
  const std::string_view name = section("disabled");
  const prof::Section s(name);

  int calls = 0;
  prof::time("off", stream(), [&] { ++calls; }, 7, 3);
  EXPECT_EQ(calls, 1);

  EXPECT_TRUE(syncAndReport(name).empty());
}

// --- ProfileSectionTests ---------------------------------------------------------

TEST_F(ProfileSectionTests, SpanLandsInInnermostOpenSection) {
  const std::string_view outerName = section("outer");
  const std::string_view innerName = section("inner");
  prof::setEnabled(true);

  int calls = 0;
  const auto f = [&] { ++calls; };
  {
    const prof::Section outer(outerName);
    prof::time("before", stream(), f);
    {
      const prof::Section inner(innerName);
      prof::time("nested", stream(), f);
      prof::time("nested", stream(), f);
    }
    // The inner Section's destructor restored the outer one.
    prof::time("after", stream(), f);
  }
  EXPECT_EQ(calls, 4);

  const std::vector<ProfileRow> inner = syncAndReport(innerName);
  ASSERT_EQ(inner.size(), 1u);
  EXPECT_EQ(inner[0].label, "nested");
  EXPECT_EQ(inner[0].calls, 2);

  const std::vector<ProfileRow> outer = syncAndReport(outerName);
  ASSERT_EQ(outer.size(), 2u);
  const ProfileRow* before = findRow(outer, "before");
  const ProfileRow* after = findRow(outer, "after");
  ASSERT_NE(before, nullptr);
  ASSERT_NE(after, nullptr);
  EXPECT_EQ(before->calls, 1);
  EXPECT_EQ(after->calls, 1);
  EXPECT_EQ(findRow(outer, "nested"), nullptr);
}

TEST_F(ProfileSectionTests, SectionIsOpenWhileProfilingIsOff) {
  // A Section is host-only bookkeeping, constructed whether or not profiling
  // is on: one opened while off still catches spans recorded once it is on.
  const std::string_view name = section("opened-while-off");
  const prof::Section s(name);
  prof::setEnabled(true);

  prof::time("span", stream(), [] {});

  const std::vector<ProfileRow> rows = syncAndReport(name);
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_EQ(rows[0].label, "span");
  EXPECT_EQ(rows[0].calls, 1);
}

// --- ProfileReportTests ----------------------------------------------------------

TEST_F(ProfileReportTests, GroupsByLabelCountsCallsKeepsLastSizes) {
  const std::string_view name = section("report");
  const prof::Section s(name);
  prof::setEnabled(true);

  int calls = 0;
  const auto f = [&] { ++calls; };
  prof::time("gemm", stream(), f, 10, 2);
  prof::time("other", stream(), f, 99, 98);
  prof::time("gemm", stream(), f, 20, 3);
  prof::time("gemm", stream(), f, 30, 4);
  prof::time("sizeless", stream(), f);
  EXPECT_EQ(calls, 5);

  const std::vector<ProfileRow> rows = syncAndReport(name);
  ASSERT_EQ(rows.size(), 3u);

  const ProfileRow* gemm = findRow(rows, "gemm");
  ASSERT_NE(gemm, nullptr);
  EXPECT_EQ(gemm->calls, 3);
  EXPECT_EQ(gemm->outSize, 30);
  EXPECT_EQ(gemm->contractedSize, 4);

  const ProfileRow* other = findRow(rows, "other");
  ASSERT_NE(other, nullptr);
  EXPECT_EQ(other->calls, 1);
  EXPECT_EQ(other->outSize, 99);
  EXPECT_EQ(other->contractedSize, 98);

  const ProfileRow* sizeless = findRow(rows, "sizeless");
  ASSERT_NE(sizeless, nullptr);
  EXPECT_EQ(sizeless->calls, 1);
  EXPECT_EQ(sizeless->outSize, 0);
  EXPECT_EQ(sizeless->contractedSize, 0);

  for (const ProfileRow& row : rows) {
    EXPECT_TRUE(std::isfinite(row.totalMs)) << row.label;
    EXPECT_GE(row.totalMs, 0.0) << row.label;
    EXPECT_DOUBLE_EQ(row.meanUs, row.totalMs * 1000.0 / row.calls) << row.label;
  }
  // Sorted by totalMs, descending.
  EXPECT_TRUE(std::ranges::is_sorted(rows, std::ranges::greater{}, &ProfileRow::totalMs));
}

TEST_F(ProfileReportTests, ReportForgetsWhatItReportedAndOnlyThatSection) {
  const std::string_view first = section("report-first");
  const std::string_view second = section("report-second");
  prof::setEnabled(true);
  {
    const prof::Section a(first);
    prof::time("a", stream(), [] {});
  }
  {
    const prof::Section b(second);
    prof::time("b", stream(), [] {});
  }

  ASSERT_EQ(syncAndReport(first).size(), 1u);
  // A second report of the same section does not double-count.
  EXPECT_TRUE(syncAndReport(first).empty());

  // Reporting `first` left `second`'s span in place.
  const std::vector<ProfileRow> rows = syncAndReport(second);
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_EQ(rows[0].label, "b");
  EXPECT_EQ(rows[0].calls, 1);
}

// --- ProfileSetEnabledTests ------------------------------------------------------

TEST_F(ProfileSetEnabledTests, DisablingAfterUseStopsRecording) {
  const std::string_view name = section("toggle");
  const prof::Section s(name);

  int calls = 0;
  const auto f = [&] { ++calls; };
  prof::setEnabled(true);
  EXPECT_TRUE(prof::enabled());
  prof::time("span", stream(), f);
  prof::setEnabled(false);
  EXPECT_FALSE(prof::enabled());
  prof::time("span", stream(), f);
  prof::time("span", stream(), f);
  EXPECT_EQ(calls, 3);

  const std::vector<ProfileRow> rows = syncAndReport(name);
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_EQ(rows[0].label, "span");
  EXPECT_EQ(rows[0].calls, 1);
}

// --- ProfileDeathTest ------------------------------------------------------------
//
// A span with no Section open would land in "", which no caller reports, so
// its events would never be read or destroyed: begin() check()s first, which
// prints "error at file:line in function: <what>" and std::abort()s before
// any event is created. The child enables profiling and gets the stream
// itself (the forked child's state is its own: the parent stays disabled).

// Records one span with no Section open, in the death-test child. If the
// child cannot bring the device up, it exits with a message no regex below
// matches, so the death test fails rather than passing on the wrong abort.
void spanOutsideSectionInChild() {
  const auto& res = test::sharedResources();
  if (!res.has_value()) {
    std::fputs("no device in the death-test child\n", stderr);
    std::_Exit(3);
  }
  prof::setEnabled(true);
  prof::time("orphan", (*res)->stream(), [] {});
}

TEST(ProfileDeathTest, SpanWithNoSectionOpenAborts) {
  ASSERT_FALSE(prof::enabled());
  EXPECT_DEATH(spanOutsideSectionInChild(),
               "error at .*profile\\.cpp:[0-9]+ .*profile span 'orphan' recorded outside any "
               "profile::Section");
  // The child's setEnabled(true) did not reach this process.
  EXPECT_FALSE(prof::enabled());
}

}  // namespace nevpt2::test::profile_suite
