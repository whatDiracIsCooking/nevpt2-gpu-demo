// Host-only suites for nevpt2.device_resources' free parsing/formatting
// helpers: parsePoolThreshold (--pool-threshold's argument) and toMB.
//
// Host-only ON PURPOSE: neither function touches a device, so this binary is
// not REQUIRES_GPU and runs with no card visible. It imports
// nevpt2.device_resources, which links the GPU runtime, but loading that
// runtime needs no card -- the GPU is reached only through
// DeviceResources::create, which nothing here calls (cli_tests, also
// host-only, links the same library through nevpt2.cli).
//
//   PoolThresholdTests  parsePoolThreshold: "max", "0", byte counts up to
//                       UINT64_MAX; refuses junk, empty, signs, whitespace
//                       and overflow with an InvalidConfig Error naming
//                       --pool-threshold
//   ToMBTests           toMB: bytes to binary megabytes
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.device_resources;

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::device_resources {

// The one message every refusal carries, naming the flag and the argument.
std::string refusal(const std::string_view arg) {
  return std::format("--pool-threshold takes a byte count or \"max\", not \"{}\"", arg);
}

void expectRefused(const std::string_view arg) {
  const Result<std::uint64_t> r = parsePoolThreshold(arg);
  ASSERT_FALSE(r.has_value()) << "accepted \"" << arg << "\" as " << *r;
  EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig) << arg;
  EXPECT_EQ(r.error().message, refusal(arg));
  EXPECT_TRUE(r.error().message.contains("--pool-threshold")) << arg;
}

// --- PoolThresholdTests -------------------------------------------------------

TEST(PoolThresholdTests, MaxIsTheDefault) {
  const Result<std::uint64_t> r = parsePoolThreshold("max");
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, kDefaultPoolReleaseThreshold);
}

TEST(PoolThresholdTests, ZeroIsAccepted) {
  const Result<std::uint64_t> r = parsePoolThreshold("0");
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, 0u);
}

TEST(PoolThresholdTests, PlainByteCount) {
  const Result<std::uint64_t> r = parsePoolThreshold("1073741824");
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, std::uint64_t{1} << 30);
}

TEST(PoolThresholdTests, LargestByteCountFitsAndEqualsMax) {
  // UINT64_MAX spelled out is the same threshold as "max".
  const Result<std::uint64_t> r = parsePoolThreshold("18446744073709551615");
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, kDefaultPoolReleaseThreshold);
}

TEST(PoolThresholdTests, LeadingZerosAreDecimal) {
  // Recorded behaviour, not a design choice: std::from_chars base 10 reads
  // leading zeros as decimal digits (no octal), so "01024" is 1024.
  const Result<std::uint64_t> r = parsePoolThreshold("01024");
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, 1024u);
}

TEST(PoolThresholdTests, RefusesTrailingJunk) {
  for (const std::string_view bad : {"1024x", "1024 ", "1e9", "1.5", "0x10", "maxx"})
    expectRefused(bad);
}

TEST(PoolThresholdTests, RefusesEmpty) { expectRefused(""); }

TEST(PoolThresholdTests, RefusesLeadingSign) {
  for (const std::string_view bad : {"-1", "+1", "-0"}) expectRefused(bad);
}

TEST(PoolThresholdTests, RefusesWhitespace) {
  for (const std::string_view bad : {" 1", "\t1", "1\n", " ", " max", "max "})
    expectRefused(bad);
}

TEST(PoolThresholdTests, RefusesOverflowPastUint64) {
  for (const std::string_view bad : {"18446744073709551616", "99999999999999999999999"})
    expectRefused(bad);
}

// --- ToMBTests ----------------------------------------------------------------

TEST(ToMBTests, ConvertsBinaryMegabytes) {
  EXPECT_EQ(toMB(0), 0.0);
  EXPECT_EQ(toMB(std::uint64_t{1} << 20), 1.0);
  EXPECT_EQ(toMB(std::uint64_t{3} << 19), 1.5);
  EXPECT_EQ(toMB(std::uint64_t{1} << 30), 1024.0);
}

}  // namespace nevpt2::test::device_resources
