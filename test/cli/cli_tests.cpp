// Suites for nevpt2.cli, the command-line front end both demos share.
//
// Host-only ON PURPOSE: parsing builds values and Errors and never touches a
// device, so this binary is not REQUIRES_GPU and runs with no card visible.
//
//   CliIntegerTests   parseInteger: whole integers that fit T, only
//   CliFlagTests      parseCommonFlag: which flags it takes, what they set,
//                     and what it leaves for the app
//   CliFinalizeTests  finalize: --golden, --tiles >= 1, --ozaki-pairs 0..14,
//                     the digest conflicts
//   CliUsageTests     usage(): the app's own flags in their place
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.cli;
import nevpt2.cublas_emul;  // kHaveCublasEmul: what finalize does with --cublas

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
// One binary per suite file, so a name clash here is a link error.
namespace nevpt2::test::cli {

using namespace ::nevpt2::cli;

// Feeds `args` through parseCommonFlag as an app's loop would. The Error, if
// one is returned; false (not consumed) is an InvalidConfig Error here too,
// naming the argument, so a test can tell the two apart by message.
Result<CommonOptions> parseAll(const std::vector<std::string_view>& args) {
  CommonOptions opt;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const Result<bool> used = parseCommonFlag(args, i, opt);
    if (!used) return std::unexpected(used.error());
    if (!*used) return err_config(std::format("not consumed: {}", args[i]));
  }
  return opt;
}

// --- CliIntegerTests ----------------------------------------------------------

TEST(CliIntegerTests, ParsesAWholeInteger) {
  EXPECT_EQ(parseInteger<int64_t>("--tiles", "40"), 40);
  EXPECT_EQ(parseInteger<int>("--mantissa-bits", "-3"), -3);
}

TEST(CliIntegerTests, RefusesGarbageAndTrailingText) {
  for (const std::string_view bad : {"", "abc", "4x", "4.0", " 4"}) {
    const Result<int64_t> r = parseInteger<int64_t>("--tiles", bad);
    ASSERT_FALSE(r.has_value()) << bad;
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message, std::format("--tiles takes an integer, not \"{}\"", bad));
  }
}

TEST(CliIntegerTests, RefusesOutOfRangeForT) {
  EXPECT_FALSE(parseInteger<int>("--mantissa-bits", "99999999999").has_value());
}


// --- CliFlagTests -------------------------------------------------------------

TEST(CliFlagTests, DefaultsAreRdmBuildOptionsDefaults) {
  const Result<CommonOptions> r = parseAll({});
  ASSERT_TRUE(r.has_value());
  const RdmBuildOptions defaults;
  EXPECT_TRUE(r->goldenPath.empty());
  EXPECT_EQ(r->rdm.nTiles, defaults.nTiles);
  EXPECT_EQ(r->rdm.blasDigest, defaults.blasDigest);
  EXPECT_FALSE(r->pc);
  EXPECT_FALSE(r->profile);
  EXPECT_EQ(r->poolThreshold, kDefaultPoolReleaseThreshold);
}

TEST(CliFlagTests, EveryCommonFlagSetsItsField) {
  const Result<CommonOptions> r = parseAll(
      {"--golden", "g.nevpt2gold", "--tiles", "40", "--pc", "--profile", "--consume-emitted",
       "--blas-digest", "--ozaki", "--ozaki-pairs", "5", "--ozaki-check", "--mantissa-bits",
       "40", "--pool-threshold", "1024"});
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->goldenPath, "g.nevpt2gold");
  EXPECT_EQ(r->rdm.nTiles, 40);
  EXPECT_TRUE(r->pc);
  EXPECT_TRUE(r->profile);
  EXPECT_FALSE(r->rdm.consumeGemm);
  EXPECT_TRUE(r->rdm.blasDigest);
  EXPECT_TRUE(r->rdm.ozaki);
  EXPECT_EQ(r->rdm.ozakiMaxPairSum, 5);
  EXPECT_TRUE(r->rdm.ozakiCheck);
  EXPECT_EQ(r->rdm.mantissaBits, 40);
  EXPECT_EQ(r->poolThreshold, 1024u);
}

TEST(CliFlagTests, CublasAndDigestEmittedSetTheirFields) {
  const Result<CommonOptions> r = parseAll({"--cublas", "--digest-emitted"});
  ASSERT_TRUE(r.has_value());
  EXPECT_TRUE(r->rdm.cublas);
  EXPECT_FALSE(r->rdm.blasDigest);
}

TEST(CliFlagTests, LeavesAnAppFlagUnconsumed) {
  CommonOptions opt;
  const std::vector<std::string_view> args{"--batch", "8"};
  std::size_t i = 0;
  const Result<bool> used = parseCommonFlag(args, i, opt);
  ASSERT_TRUE(used.has_value());
  EXPECT_FALSE(*used);
  EXPECT_EQ(i, 0u);
}

TEST(CliFlagTests, AValuedFlagWithNoValueIsLeftForUsage) {
  const Result<CommonOptions> r = parseAll({"--tiles"});
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().message, "not consumed: --tiles");
}

TEST(CliFlagTests, ABadValueIsAConfigErrorNamingTheFlag) {
  const Result<CommonOptions> r = parseAll({"--tiles", "three"});
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(r.error().message, "--tiles takes an integer, not \"three\"");
}

TEST(CliFlagTests, ABadPoolThresholdIsAConfigError) {
  const Result<CommonOptions> r = parseAll({"--pool-threshold", "lots"});
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
}

// --- CliFinalizeTests ---------------------------------------------------------

CommonOptions withGolden() {
  CommonOptions opt;
  opt.goldenPath = "g.nevpt2gold";
  return opt;
}

TEST(CliFinalizeTests, AcceptsTheDefaults) {
  CommonOptions opt = withGolden();
  EXPECT_TRUE(finalize(opt, "path.nevpt2gold").has_value());
}

TEST(CliFinalizeTests, RequiresGoldenNamingTheHint) {
  CommonOptions opt;
  const Status st = finalize(opt, "path_df.nevpt2gold");
  ASSERT_FALSE(st.has_value());
  EXPECT_EQ(st.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(st.error().message, "--golden <path_df.nevpt2gold> is required");
}

TEST(CliFinalizeTests, RefusesTilesBelowOne) {
  for (const int64_t tiles : {0, -1}) {
    CommonOptions opt = withGolden();
    opt.rdm.nTiles = tiles;
    const Status st = finalize(opt, "path.nevpt2gold");
    ASSERT_FALSE(st.has_value()) << tiles;
    EXPECT_EQ(st.error().message, "--tiles must be >= 1");
  }
}

TEST(CliFinalizeTests, OzakiPairsIsZeroToFourteen) {
  for (const int pairs : {0, 14}) {
    CommonOptions opt = withGolden();
    opt.rdm.ozakiMaxPairSum = pairs;
    EXPECT_TRUE(finalize(opt, "path.nevpt2gold").has_value()) << pairs;
  }
  for (const int pairs : {-1, 15}) {
    CommonOptions opt = withGolden();
    opt.rdm.ozakiMaxPairSum = pairs;
    const Status st = finalize(opt, "path.nevpt2gold");
    ASSERT_FALSE(st.has_value()) << pairs;
    EXPECT_EQ(st.error().message, "--ozaki-pairs must be in 0..14");
  }
}

TEST(CliFinalizeTests, CublasIsRefusedWhereItIsNotBuilt) {
  CommonOptions opt = withGolden();
  opt.rdm.cublas = true;
  opt.rdm.blasDigest = false;
  const Status st = finalize(opt, "path.nevpt2gold");
  EXPECT_EQ(st.has_value(), kHaveCublasEmul);
  if (!kHaveCublasEmul) EXPECT_EQ(st.error().kind, ErrorKind::Unsupported);
}

TEST(CliFinalizeTests, CublasWithBlasDigestIsRefused) {
  if (!kHaveCublasEmul) GTEST_SKIP() << "--cublas is refused first on this backend";
  CommonOptions opt = withGolden();
  opt.rdm.cublas = true;
  opt.rdm.blasDigest = true;
  const Status st = finalize(opt, "path.nevpt2gold");
  ASSERT_FALSE(st.has_value());
  EXPECT_EQ(st.error().message, "--cublas and --blas-digest are two digests; pick one");
}

TEST(CliFinalizeTests, CublasWithOzakiIsRefused) {
  if (!kHaveCublasEmul) GTEST_SKIP() << "--cublas is refused first on this backend";
  CommonOptions opt = withGolden();
  opt.rdm.cublas = true;
  opt.rdm.blasDigest = false;
  opt.rdm.ozaki = true;
  const Status st = finalize(opt, "path.nevpt2gold");
  ASSERT_FALSE(st.has_value());
  EXPECT_EQ(st.error().message, "--cublas and --ozaki are two digests; pick one");
}

TEST(CliFinalizeTests, OzakiTurnsTheBlasDigestOff) {
  CommonOptions opt = withGolden();
  opt.rdm.ozaki = true;
  opt.rdm.blasDigest = true;
  ASSERT_TRUE(finalize(opt, "path.nevpt2gold").has_value());
  EXPECT_FALSE(opt.rdm.blasDigest);
}

// --- CliUsageTests ------------------------------------------------------------

TEST(CliUsageTests, NoExtraFlags) {
  const std::string u = usage("nevpt2_demo", "path.nevpt2gold", "");
  EXPECT_TRUE(u.starts_with("usage: nevpt2_demo --golden <path.nevpt2gold> [--tiles N]"));
  EXPECT_TRUE(u.ends_with("(the --cublas family is CUDA-only)\n"));
}

TEST(CliUsageTests, ExtraFlagsComeBeforeTheCommonOnes) {
  const std::string u = usage("nevpt2_df_demo", "path_df.nevpt2gold", "[--check-blocks]");
  EXPECT_TRUE(
      u.starts_with("usage: nevpt2_df_demo --golden <path_df.nevpt2gold> [--check-blocks] "
                    "[--tiles N]"));
}

}  // namespace nevpt2::test::cli
