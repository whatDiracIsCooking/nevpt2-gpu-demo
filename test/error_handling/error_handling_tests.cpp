// Suites for nevpt2.error_handling, and for nevpt2.common's integer helpers.
//
// Host-only ON PURPOSE: the two error tiers are host control flow -- they
// build values, format text and abort, and never touch a device -- so this
// binary is not REQUIRES_GPU and runs with no card visible. It checks logic,
// not energies: the golden tier owns every numerical claim.
//
//   ErrorKindTests      every ErrorKind's kindName, the text report() prints
//   ErrorValueTests     err_* -> Result<T> / Status: kind, message, origin
//   TryMacroTests       NEVPT2_TRY returns the value, or the Error UNCHANGED
//   ReportTests         report()'s exact stderr line
//   CommonIntegerTests  idivup / align_up, and narrowTo in range
//   AbortTierDeathTest  narrowTo out of range, and check(false), abort with
//                       the caller's file:line and message
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, so not importable

import std;
import nevpt2.error_handling;
import nevpt2.common;

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
// One binary per suite file, so a name clash here is a link error.
namespace nevpt2::test::error_handling {

// --- ErrorKindTests -----------------------------------------------------------

TEST(ErrorKindTests, KindNameIsTheTextReportPrints) {
  EXPECT_EQ(kindName(ErrorKind::IO), "I/O error");
  EXPECT_EQ(kindName(ErrorKind::InvalidConfig), "invalid configuration");
  EXPECT_EQ(kindName(ErrorKind::Unsupported), "unsupported");
  EXPECT_EQ(kindName(ErrorKind::Numerical), "numerical error");
}

TEST(ErrorKindTests, KindNameIsConstexpr) {
  static_assert(kindName(ErrorKind::IO) == "I/O error");
  static_assert(kindName(ErrorKind::Numerical) == "numerical error");
  SUCCEED();
}

TEST(ErrorKindTests, FactoriesSetTheirKind) {
  EXPECT_EQ(Error::io("m").kind, ErrorKind::IO);
  EXPECT_EQ(Error::config("m").kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(Error::unsupported("m").kind, ErrorKind::Unsupported);
  EXPECT_EQ(Error::numerical("m").kind, ErrorKind::Numerical);
}

// --- ErrorValueTests ----------------------------------------------------------

Result<int> failsWithIo() { return err_io("golden file missing"); }
Status failsWithConfig() { return err_config("--tiles must be positive"); }

TEST(ErrorValueTests, ErrIntoResultCarriesKindAndMessage) {
  const Result<int> r = failsWithIo();
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::IO);
  EXPECT_EQ(r.error().message, "golden file missing");
}

TEST(ErrorValueTests, ErrIntoStatus) {
  const Status s = failsWithConfig();
  ASSERT_FALSE(s.has_value());
  EXPECT_EQ(s.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(s.error().message, "--tiles must be positive");
}

TEST(ErrorValueTests, EachErrFactoryMapsToItsKind) {
  const Status u = err_unsupported("u");
  const Status n = err_numerical("n");
  ASSERT_FALSE(u.has_value());
  ASSERT_FALSE(n.has_value());
  EXPECT_EQ(u.error().kind, ErrorKind::Unsupported);
  EXPECT_EQ(n.error().kind, ErrorKind::Numerical);
}

TEST(ErrorValueTests, OriginIsTheCreatingCallSite) {
  const std::uint_least32_t line = std::source_location::current().line() + 1;
  const Status s = err_numerical("not converged");
  ASSERT_FALSE(s.has_value());
  EXPECT_EQ(s.error().origin.line(), line);
  EXPECT_NE(std::string_view{s.error().origin.file_name()}.find("error_handling_tests.cpp"),
            std::string_view::npos);
}

TEST(ErrorValueTests, SuccessHoldsTheValue) {
  const Result<int> r = 42;
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, 42);
  const Status s{};
  EXPECT_TRUE(s.has_value());
}

// --- TryMacroTests ------------------------------------------------------------

// The line every propagated Error must still name: where it was created.
std::uint_least32_t g_innerLine = 0;

Result<int> innerFails() {
  g_innerLine = std::source_location::current().line() + 1;
  return err_numerical("metric not gapped");
}
Result<int> innerSucceeds() { return 7; }
Status innerStatusFails() { return err_unsupported("--cublas on HIP"); }

// Two levels of NEVPT2_TRY between the Error's creation and the test.
Result<int> middle(const bool fail) {
  const int v = NEVPT2_TRY(fail ? innerFails() : innerSucceeds());
  return v + 1;
}
Result<int> outer(const bool fail) {
  const int v = NEVPT2_TRY(middle(fail));
  return v * 10;
}
Status statusChain() {
  NEVPT2_TRY(innerStatusFails());
  return {};
}

TEST(TryMacroTests, SuccessYieldsTheValue) {
  const Result<int> r = outer(false);
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, 80);
}

TEST(TryMacroTests, FailurePropagatesTheErrorUnchanged) {
  const Result<int> r = outer(true);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::Numerical);
  EXPECT_EQ(r.error().message, "metric not gapped");
  // The origin is innerFails()'s, not middle()'s or outer()'s.
  EXPECT_EQ(r.error().origin.line(), g_innerLine);
  EXPECT_NE(std::string_view{r.error().origin.function_name()}.find("innerFails"),
            std::string_view::npos);
}

TEST(TryMacroTests, StatusBareUsePropagates) {
  const Status s = statusChain();
  ASSERT_FALSE(s.has_value());
  EXPECT_EQ(s.error().kind, ErrorKind::Unsupported);
  EXPECT_EQ(s.error().message, "--cublas on HIP");
}

TEST(TryMacroTests, ArgumentIsEvaluatedOnce) {
  int calls = 0;
  const auto counted = [&]() -> Result<int> {
    ++calls;
    return 1;
  };
  const auto wrapper = [&]() -> Result<int> { return NEVPT2_TRY(counted()) + 1; };
  const Result<int> r = wrapper();
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, 2);
  EXPECT_EQ(calls, 1);
}

// --- ReportTests --------------------------------------------------------------

TEST(ReportTests, PrintsOneLineWithOriginKindAndMessage) {
  const Error e = Error::io("cannot open golden/x.nevpt2gold");
  const std::string expected =
      std::format("error at {}:{} in {}: I/O error: cannot open golden/x.nevpt2gold\n",
                  e.origin.file_name(), e.origin.line(), e.origin.function_name());
  testing::internal::CaptureStderr();
  report(e);
  EXPECT_EQ(testing::internal::GetCapturedStderr(), expected);
}

TEST(ReportTests, UsesTheKindName) {
  const Status s = err_config("bad --batch");
  ASSERT_FALSE(s.has_value());
  testing::internal::CaptureStderr();
  report(s.error());
  const std::string out = testing::internal::GetCapturedStderr();
  EXPECT_NE(out.find(": invalid configuration: bad --batch\n"), std::string::npos) << out;
  EXPECT_EQ(out.rfind("error at ", 0), 0u) << out;
}

// --- CommonIntegerTests -------------------------------------------------------

TEST(CommonIntegerTests, IdivupIsTheCeiling) {
  EXPECT_EQ(idivup(0, 4), 0);
  EXPECT_EQ(idivup(1, 4), 1);
  EXPECT_EQ(idivup(4, 4), 1);
  EXPECT_EQ(idivup(5, 4), 2);
  EXPECT_EQ(idivup(int64_t{1} << 40, int64_t{256}), int64_t{1} << 32);
  static_assert(idivup(7, 2) == 4);
}

TEST(CommonIntegerTests, AlignUpRoundsToAMultiple) {
  EXPECT_EQ(align_up(0, 8), 0);
  EXPECT_EQ(align_up(1, 8), 8);
  EXPECT_EQ(align_up(8, 8), 8);
  EXPECT_EQ(align_up(9, 8), 16);
  EXPECT_EQ(align_up(std::size_t{1000}, std::size_t{256}), std::size_t{1024});
  static_assert(align_up(5, 3) == 6);
}

TEST(CommonIntegerTests, NarrowToInRangeReturnsTheValue) {
  EXPECT_EQ(narrowTo<int>(int64_t{0}), 0);
  EXPECT_EQ(narrowTo<int>(int64_t{std::numeric_limits<int>::max()}),
            std::numeric_limits<int>::max());
  EXPECT_EQ(narrowTo<int>(int64_t{std::numeric_limits<int>::min()}),
            std::numeric_limits<int>::min());
  EXPECT_EQ(narrowTo<unsigned>(int64_t{std::numeric_limits<unsigned>::max()}),
            std::numeric_limits<unsigned>::max());
}

// --- AbortTierDeathTest --------------------------------------------------------
//
// The abort tier: out of range is our bug, so narrowTo goes through check(),
// which prints "error at file:line in function: <what>" and std::abort()s.

TEST(AbortTierDeathTest, AboveIntMaxAborts) {
  const int64_t big = int64_t{1} << 40;
  EXPECT_DEATH((void)narrowTo<int>(big),
               "error at .*error_handling_tests\\.cpp:[0-9]+ .*1099511627776 does not fit "
               "in a 32-bit signed integer");
}

TEST(AbortTierDeathTest, NegativeIntoUnsignedAborts) {
  EXPECT_DEATH((void)narrowTo<unsigned>(int64_t{-1}),
               "-1 does not fit in a 32-bit unsigned integer");
}

TEST(AbortTierDeathTest, HintIsAppended) {
  EXPECT_DEATH((void)narrowTo<int>(int64_t{1} << 33, "use more --tiles"),
               "8589934592 does not fit in a 32-bit signed integer; use more --tiles");
}

TEST(AbortTierDeathTest, CheckFalseAborts) {
  EXPECT_DEATH(check(false, "broken invariant"), "error at .*: broken invariant");
}

}  // namespace nevpt2::test::error_handling
