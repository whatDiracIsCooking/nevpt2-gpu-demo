// Suites for nevpt2.cli, the command line of all four golden-reading binaries.
//
// Host-only ON PURPOSE: parsing builds values and Errors and never touches a
// device, so this binary is not REQUIRES_GPU and runs with no card visible.
//
//   CliIntegerTests      parseInteger: whole integers that fit T, only
//   CliFlagTests         parseCommonFlag: which flags it takes, what they set,
//                        and what it leaves for the binary
//   CliFinalizeTests     finalize: --golden, --tiles >= 1, --ozaki-pairs
//                        0..14 and only with --ozaki (so --ozaki-check),
//                        --mantissa-bits 1..53 and only with --cublas, the
//                        digest conflicts
//   CliUsageTests        usage(b): each binary's own flags, and only its own
//   CliMatrixTests       the binary/flag matrix: exactly which binary
//                        implements which flag, and that every flag a binary
//                        does not implement is refused as Unsupported
//   CliParseArgsTests    parseArgs' shared behaviour: an unknown argument, a
//                        missing value, finalize() applied, and every flag
//                        combination a registered ctest entry passes accepted
//   CliDemoFlagTests     nevpt2_demo's --sc-tangent / --gradient-source /
//                        --sc-hessian / --sc-hessian-vector
//   CliDfDemoFlagTests   nevpt2_df_demo's --batch / --check-blocks, and its
//                        --sc-tangent / --gradient-source (#162)
//   CliScPseudodensityFlagTests  --kappa-act / --kappa-full / --gradgold /
//                                --dump-source-terms / --source-oracle /
//                                --sc-hessian-kc / --sc-hessian-kk-int /
//                                --sc-hessian-kk
//   CliPcTangentFlagTests        --fd-steps / --fd-start-pow / --pc-hessian /
//                                --pc-hessian-vector
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
  EXPECT_FALSE(r->rdmTangent);
  EXPECT_FALSE(r->mantissaBitsGiven);
  EXPECT_EQ(r->poolThreshold, kDefaultPoolReleaseThreshold);
}

TEST(CliFlagTests, EveryCommonFlagSetsItsField) {
  const Result<CommonOptions> r = parseAll(
      {"--golden", "g.nevpt2gold", "--tiles", "40", "--pc", "--profile", "--consume-emitted",
       "--blas-digest", "--ozaki", "--ozaki-pairs", "5", "--ozaki-check", "--mantissa-bits",
       "40", "--pool-threshold", "1024", "--fused-digest", "--rdm-tangent"});
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->goldenPath, "g.nevpt2gold");
  EXPECT_EQ(r->rdm.nTiles, 40);
  EXPECT_TRUE(r->pc);
  EXPECT_TRUE(r->profile);
  EXPECT_FALSE(r->rdm.consumeGemm);
  EXPECT_TRUE(r->rdm.blasDigest);
  EXPECT_TRUE(r->rdm.ozaki);
  EXPECT_EQ(r->rdm.ozakiMaxPairSum, 5);
  EXPECT_TRUE(r->ozakiPairsGiven);
  EXPECT_TRUE(r->rdm.ozakiCheck);
  EXPECT_TRUE(r->rdm.fusedDigest);
  EXPECT_TRUE(r->rdmTangent);
  EXPECT_EQ(r->rdm.mantissaBits, 40);
  EXPECT_TRUE(r->mantissaBitsGiven);
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

TEST(CliFinalizeTests, OzakiPairsAndOzakiCheckNeedOzaki) {
  // Only the --ozaki digest reads either, so without it each would be parsed
  // and ignored.
  {
    CommonOptions opt = withGolden();
    opt.rdm.ozakiMaxPairSum = 5;
    opt.ozakiPairsGiven = true;
    const Status st = finalize(opt, "path.nevpt2gold");
    ASSERT_FALSE(st.has_value());
    EXPECT_EQ(st.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(st.error().message,
              "--ozaki-pairs sets the --ozaki digest's truncation; it needs --ozaki");
    opt.rdm.ozaki = true;
    EXPECT_TRUE(finalize(opt, "path.nevpt2gold").has_value());
  }
  {
    CommonOptions opt = withGolden();
    opt.rdm.ozakiCheck = true;
    const Status st = finalize(opt, "path.nevpt2gold");
    ASSERT_FALSE(st.has_value());
    EXPECT_EQ(st.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(st.error().message,
              "--ozaki-check checks the --ozaki digest's GEMMs; it needs --ozaki");
    opt.rdm.ozaki = true;
    EXPECT_TRUE(finalize(opt, "path.nevpt2gold").has_value());
  }
}

TEST(CliFinalizeTests, MantissaBitsNeedsCublas) {
  CommonOptions opt = withGolden();
  opt.rdm.mantissaBits = 40;
  opt.mantissaBitsGiven = true;
  const Status st = finalize(opt, "path.nevpt2gold");
  ASSERT_FALSE(st.has_value());
  EXPECT_EQ(st.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(st.error().message,
            "--mantissa-bits sets the --cublas digest's precision; it needs --cublas");
}

TEST(CliFinalizeTests, MantissaBitsIsOneToFiftyThree) {
  if (!kHaveCublasEmul) GTEST_SKIP() << "--cublas is refused on this backend";
  for (const int bits : {1, 53}) {
    CommonOptions opt = withGolden();
    opt.rdm.cublas = true;
    opt.rdm.blasDigest = false;
    opt.rdm.mantissaBits = bits;
    opt.mantissaBitsGiven = true;
    EXPECT_TRUE(finalize(opt, "path.nevpt2gold").has_value()) << bits;
  }
  // Refused on range before --cublas is looked at, so on every backend.
  for (const int bits : {0, -1, 54}) {
    CommonOptions opt = withGolden();
    opt.rdm.cublas = true;
    opt.rdm.mantissaBits = bits;
    opt.mantissaBitsGiven = true;
    const Status st = finalize(opt, "path.nevpt2gold");
    ASSERT_FALSE(st.has_value()) << bits;
    EXPECT_EQ(st.error().message, "--mantissa-bits must be in 1..53");
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

TEST(CliFinalizeTests, FusedDigestNeedsAGemmDigest) {
  CommonOptions opt = withGolden();
  opt.rdm.fusedDigest = true;
  opt.rdm.blasDigest = false;
  const Status st = finalize(opt, "path.nevpt2gold");
  ASSERT_FALSE(st.has_value());
  EXPECT_EQ(st.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(st.error().message,
            "--fused-digest needs a GEMM digest (--blas-digest, --cublas or --ozaki)");
  // Either native-fp64 GEMM digest is enough (--cublas is refused first on HIP).
  opt.rdm.blasDigest = true;
  EXPECT_TRUE(finalize(opt, "path.nevpt2gold").has_value());
  opt.rdm.blasDigest = false;
  opt.rdm.ozaki = true;
  EXPECT_TRUE(finalize(opt, "path.nevpt2gold").has_value());
}

TEST(CliFinalizeTests, RdmTangentRefusesTheDigestsItHasNoTangentFor) {
  // The tangent build runs the emitted digest or the native-fp64 BLAS one;
  // --cublas, --ozaki and --fused-digest have no tangent path, so each is a
  // config Error rather than a silently ignored flag.
  constexpr std::string_view kMessage =
      "--rdm-tangent runs the emitted digest or --blas-digest; not --cublas, --ozaki or "
      "--fused-digest";
  {
    CommonOptions opt = withGolden();
    opt.rdmTangent = true;
    opt.rdm.ozaki = true;
    const Status st = finalize(opt, "path.nevpt2gold");
    ASSERT_FALSE(st.has_value());
    EXPECT_EQ(st.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(st.error().message, kMessage);
  }
  {
    CommonOptions opt = withGolden();
    opt.rdmTangent = true;
    opt.rdm.blasDigest = true;
    opt.rdm.fusedDigest = true;
    const Status st = finalize(opt, "path.nevpt2gold");
    ASSERT_FALSE(st.has_value());
    EXPECT_EQ(st.error().message, kMessage);
  }
  // Both digests it does have are accepted.
  for (const bool blasDigest : {false, true}) {
    CommonOptions opt = withGolden();
    opt.rdmTangent = true;
    opt.rdm.blasDigest = blasDigest;
    EXPECT_TRUE(finalize(opt, "path.nevpt2gold").has_value());
  }
}

// --- CliUsageTests ------------------------------------------------------------

TEST(CliUsageTests, NamesTheBinaryAndItsGolden) {
  const std::string u = usage(Binary::Demo);
  EXPECT_TRUE(u.starts_with("usage: nevpt2_demo --golden <path.nevpt2gold> [--tiles N]"));
  EXPECT_TRUE(u.ends_with("(the --cublas family is CUDA-only)\n"));
  EXPECT_TRUE(usage(Binary::DfDemo)
                  .starts_with("usage: nevpt2_df_demo --golden <path_df.nevpt2gold> [--tiles N]"));
}

TEST(CliUsageTests, ListsExactlyTheFlagsTheBinaryImplements) {
  for (const Binary b : kBinaries) {
    const std::string u = usage(b);
    for (const FlagSpec& f : flagTable()) {
      // "[--ozaki]" is a prefix of "[--ozaki-pairs P]"'s text but not of its
      // bracket, so match the bracketed spelling.
      const std::string spelled =
          f.valueName.empty() ? std::format("[{}]", f.name)
                              : std::format("[{} {}]", f.name, f.valueName);
      if (f.name == "--golden") continue;
      EXPECT_EQ(u.contains(spelled), implements(b, f.name)) << binaryName(b) << " " << f.name;
    }
  }
  // nevpt2_pc_tangent takes no --cublas, so no note about it either.
  EXPECT_FALSE(usage(Binary::PcTangent).contains("CUDA-only"));
}

// --- CliMatrixTests -----------------------------------------------------------

// parseArgs(b, --golden g.nevpt2gold, extra...).
Result<Options> parse(const Binary b, const std::vector<std::string_view>& extra) {
  std::vector<std::string_view> args{"--golden", "g.nevpt2gold"};
  args.insert(args.end(), extra.begin(), extra.end());
  return parseArgs(b, args);
}

// The matrix, written out by hand: the contract README.md ("Which binary
// takes which flag") records, held against flagTable(). D = nevpt2_demo,
// F = nevpt2_df_demo, S = nevpt2_sc_pseudodensity, P = nevpt2_pc_tangent.
struct MatrixRow {
  std::string_view flag;
  std::string_view binaries;
};
constexpr std::array kExpectedMatrix{
    MatrixRow{"--golden", "DFSP"},          MatrixRow{"--tiles", "DFSP"},
    MatrixRow{"--pc", "DFS"},               MatrixRow{"--profile", "DF"},
    MatrixRow{"--rdm-tangent", "DF"},       MatrixRow{"--consume-emitted", "DFSP"},
    MatrixRow{"--blas-digest", "DFSP"},     MatrixRow{"--digest-emitted", "DFSP"},
    MatrixRow{"--fused-digest", "DFS"},     MatrixRow{"--cublas", "DFS"},
    MatrixRow{"--mantissa-bits", "DFS"},    MatrixRow{"--ozaki", "DFS"},
    MatrixRow{"--ozaki-pairs", "DFS"},      MatrixRow{"--ozaki-check", "DFS"},
    MatrixRow{"--pool-threshold", "DFSP"},  MatrixRow{"--sc-tangent", "DF"},
    MatrixRow{"--gradient-source", "DF"},   MatrixRow{"--sc-hessian", "D"},
    MatrixRow{"--sc-hessian-vector", "D"},  MatrixRow{"--batch", "F"},
    MatrixRow{"--check-blocks", "F"},       MatrixRow{"--kappa-act", "S"},
    MatrixRow{"--kappa-full", "S"},         MatrixRow{"--sc-hessian-kc", "S"},
    MatrixRow{"--sc-hessian-kk-int", "S"},  MatrixRow{"--sc-hessian-kk", "S"},
    MatrixRow{"--gradgold", "DS"},          MatrixRow{"--dump-source-terms", "S"},
    MatrixRow{"--source-oracle", "S"},      MatrixRow{"--fd-steps", "P"},
    MatrixRow{"--fd-start-pow", "P"},       MatrixRow{"--pc-hessian", "P"},
    MatrixRow{"--pc-hessian-vector", "P"},
};

char letterOf(const Binary b) {
  switch (b) {
    case Binary::Demo:
      return 'D';
    case Binary::DfDemo:
      return 'F';
    case Binary::ScPseudodensity:
      return 'S';
    case Binary::PcTangent:
      return 'P';
  }
  return '?';
}

TEST(CliMatrixTests, TableIsTheDocumentedMatrix) {
  ASSERT_EQ(flagTable().size(), kExpectedMatrix.size());
  for (std::size_t r = 0; r < kExpectedMatrix.size(); ++r) {
    EXPECT_EQ(flagTable()[r].name, kExpectedMatrix[r].flag) << r;
    for (const Binary b : kBinaries)
      EXPECT_EQ(implements(b, kExpectedMatrix[r].flag),
                kExpectedMatrix[r].binaries.contains(letterOf(b)))
          << binaryName(b) << " " << kExpectedMatrix[r].flag;
  }
}

TEST(CliMatrixTests, EveryRowIsAFlagSomeBinaryTakes) {
  std::set<std::string_view> seen;
  for (const FlagSpec& f : flagTable()) {
    EXPECT_TRUE(f.name.starts_with("--")) << f.name;
    EXPECT_NE(f.binaries, 0) << f.name;
    EXPECT_TRUE(seen.insert(f.name).second) << "duplicate row " << f.name;
    EXPECT_EQ(findFlag(f.name), &f);
  }
  EXPECT_EQ(findFlag("--no-such-flag"), nullptr);
  for (const Binary b : kBinaries) EXPECT_FALSE(implements(b, "--no-such-flag"));
}

// The arguments that hand `flag` to parseArgs: the flag, and a value
// parseInteger accepts if it takes one. Prefixed with --golden.
std::vector<std::string_view> withFlag(const FlagSpec& f) {
  std::vector<std::string_view> args{"--golden", "g.nevpt2gold", f.name};
  if (!f.valueName.empty()) args.push_back("1");
  return args;
}

TEST(CliMatrixTests, EveryFlagABinaryLacksIsRefusedAsUnsupported) {
  for (const Binary b : kBinaries) {
    for (const FlagSpec& f : flagTable()) {
      if (implements(b, f.name)) continue;
      const Result<Options> r = parseArgs(b, withFlag(f));
      ASSERT_FALSE(r.has_value()) << binaryName(b) << " accepted " << f.name;
      EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << binaryName(b) << " " << f.name;
      EXPECT_TRUE(r.error().message.starts_with(
          std::format("{} does not implement {} (only ", binaryName(b), f.name)))
          << r.error().message;
    }
  }
}

TEST(CliMatrixTests, RdmTangentIsRefusedByTheGradientBinaries) {
  // #39's finding: both gradient binaries used to parse --rdm-tangent and do
  // nothing with it.
  for (const Binary b : {Binary::ScPseudodensity, Binary::PcTangent}) {
    const Result<Options> r = parse(b, {"--rdm-tangent"});
    ASSERT_FALSE(r.has_value()) << binaryName(b);
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported);
    EXPECT_EQ(r.error().message,
              std::format("{} does not implement --rdm-tangent (only nevpt2_demo, "
                          "nevpt2_df_demo); a flag a binary would not act on is refused, "
                          "not ignored",
                          binaryName(b)));
  }
  // ...and the demos, which run it, still take it.
  for (const Binary b : {Binary::Demo, Binary::DfDemo}) {
    const Result<Options> r = parse(b, {"--rdm-tangent"});
    ASSERT_TRUE(r.has_value()) << binaryName(b) << ": " << r.error().message;
    EXPECT_TRUE(r->common.rdmTangent);
  }
}

TEST(CliMatrixTests, ProfileIsTheDemosAloneAndPcTheirsAndTheSourceTerms) {
  for (const Binary b : {Binary::ScPseudodensity, Binary::PcTangent}) {
    const Result<Options> r = parse(b, {"--profile"});
    ASSERT_FALSE(r.has_value()) << binaryName(b);
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported);
  }
  // nevpt2_pc_tangent is PC throughout: --pc would change nothing there.
  const Result<Options> r = parse(Binary::PcTangent, {"--pc"});
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::Unsupported);
  // nevpt2_sc_pseudodensity takes it -- for its two source-term jobs alone
  // (#74; CliScPseudodensityFlagTests has the combinations).
  EXPECT_TRUE(implements(Binary::ScPseudodensity, "--pc"));
}

TEST(CliMatrixTests, PcTangentRefusesTheDigestsWithNoTangentPath) {
  for (const std::string_view flag : {"--cublas", "--ozaki", "--fused-digest", "--ozaki-check"}) {
    const Result<Options> r = parse(Binary::PcTangent, {flag});
    ASSERT_FALSE(r.has_value()) << flag;
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << flag;
  }
  // The two exact digests and the consume switch are its to take.
  for (const std::string_view flag : {"--blas-digest", "--digest-emitted", "--consume-emitted"}) {
    const Result<Options> r = parse(Binary::PcTangent, {flag});
    EXPECT_TRUE(r.has_value()) << flag << ": " << r.error().message;
  }
}

TEST(CliMatrixTests, RefusedBeforeItsValueIsRead) {
  // Another binary's valued flag is refused as Unsupported even with a bad
  // value or none: the binary never gets as far as parsing it.
  for (const std::vector<std::string_view>& args :
       {std::vector<std::string_view>{"--golden", "g.nevpt2gold", "--fd-steps", "lots"},
        std::vector<std::string_view>{"--golden", "g.nevpt2gold", "--fd-steps"}}) {
    const Result<Options> r = parseArgs(Binary::Demo, args);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported);
  }
}

// --- CliParseArgsTests --------------------------------------------------------

TEST(CliParseArgsTests, DefaultsForEveryBinary) {
  for (const Binary b : kBinaries) {
    const Result<Options> r = parse(b, {});
    ASSERT_TRUE(r.has_value()) << binaryName(b) << ": " << r.error().message;
    EXPECT_EQ(r->common.goldenPath, "g.nevpt2gold");
    EXPECT_FALSE(r->scTangent);
    EXPECT_FALSE(r->gradientSource);
    EXPECT_EQ(r->batch, kDefaultBatch);
    EXPECT_FALSE(r->checkBlocks);
    EXPECT_FALSE(r->kappaAct);
    EXPECT_FALSE(r->kappaFull);
    EXPECT_TRUE(r->gradgoldPath.empty());
    EXPECT_EQ(r->fdSteps, kDefaultFdSteps);
    EXPECT_EQ(r->fdStartPow, kDefaultFdStartPow);
  }
}

TEST(CliParseArgsTests, AnUnknownArgumentIsAConfigErrorCarryingTheUsage) {
  for (const Binary b : kBinaries) {
    for (const std::string_view bad : {"--sc_tangent", "--kappa-ful", "--fd-step", "-v", "x"}) {
      const Result<Options> r = parse(b, {bad});
      ASSERT_FALSE(r.has_value()) << binaryName(b) << " " << bad;
      EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
      EXPECT_TRUE(r.error().message.starts_with(std::format(
          "unrecognized argument \"{}\"; usage: {} --golden", bad, binaryName(b))))
          << r.error().message;
      EXPECT_FALSE(r.error().message.ends_with("\n"));
    }
  }
}

TEST(CliParseArgsTests, AValuedFlagAtTheEndIsAConfigError) {
  for (const Binary b : kBinaries) {
    for (const FlagSpec& f : flagTable()) {
      if (f.valueName.empty() || !implements(b, f.name)) continue;
      const Result<Options> r = parse(b, {f.name});
      ASSERT_FALSE(r.has_value()) << binaryName(b) << " " << f.name;
      EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
      EXPECT_TRUE(r.error().message.starts_with(
          std::format("{} takes a value ({}); usage: ", f.name, f.valueName)))
          << r.error().message;
    }
  }
}

TEST(CliParseArgsTests, MissingGoldenNamesTheBinarysHint) {
  const Result<Options> demo = parseArgs(Binary::PcTangent, std::vector<std::string_view>{});
  ASSERT_FALSE(demo.has_value());
  EXPECT_EQ(demo.error().message, "--golden <path.nevpt2gold> is required");
  const Result<Options> df = parseArgs(Binary::DfDemo, std::vector<std::string_view>{});
  ASSERT_FALSE(df.has_value());
  EXPECT_EQ(df.error().message, "--golden <path_df.nevpt2gold> is required");
}

TEST(CliParseArgsTests, AppliesFinalize) {
  const Result<Options> r = parse(Binary::ScPseudodensity, {"--tiles", "0"});
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().message, "--tiles must be >= 1");
}

TEST(CliParseArgsTests, EveryRegisteredCtestCombinationIsAccepted) {
  // The flags each registered ctest entry passes (root CMakeLists.txt), less
  // --golden: a refusal must never reject one of these.
  struct Entry {
    Binary b;
    std::vector<std::string_view> args;
    bool cublas = false;
  };
  const std::vector<Entry> entries{
      {Binary::Demo, {"--tiles", "3"}},
      {Binary::Demo, {"--tiles", "40"}},
      {Binary::Demo, {"--tiles", "3", "--cublas"}, true},
      {Binary::Demo, {"--pc", "--tiles", "3"}},
      {Binary::Demo, {"--pc", "--tiles", "40"}},
      {Binary::Demo, {"--tiles", "3", "--consume-emitted"}},
      {Binary::Demo, {"--tiles", "3", "--cublas", "--consume-emitted"}, true},
      {Binary::Demo, {"--tiles", "3", "--blas-digest"}},
      {Binary::Demo, {"--pc", "--tiles", "3", "--blas-digest"}},
      {Binary::Demo, {"--tiles", "3", "--digest-emitted"}},
      {Binary::Demo, {"--tiles", "3", "--ozaki"}},
      {Binary::Demo, {"--tiles", "3", "--profile"}},
      {Binary::Demo, {"--tiles", "3", "--cublas", "--profile"}, true},
      {Binary::Demo, {"--tiles", "3", "--rdm-tangent"}},
      {Binary::Demo, {"--tiles", "3", "--sc-tangent"}},
      {Binary::Demo, {"--tiles", "3", "--gradient-source"}},
      {Binary::Demo, {"--tiles", "3", "--sc-hessian"}},
      {Binary::Demo, {"--tiles", "3", "--sc-hessian-vector"}},
      {Binary::Demo, {"--tiles", "3", "--gradient-source", "--gradgold", "x.gradgold"}},
      {Binary::DfDemo, {"--tiles", "3", "--check-blocks"}},
      {Binary::DfDemo, {"--tiles", "3", "--batch", "5", "--check-blocks"}},
      {Binary::DfDemo, {"--tiles", "40"}},
      {Binary::DfDemo, {"--tiles", "3", "--cublas"}, true},
      {Binary::DfDemo, {"--pc", "--tiles", "3"}},
      {Binary::DfDemo, {"--pc", "--tiles", "3", "--batch", "5"}},
      {Binary::DfDemo, {"--pc", "--tiles", "3", "--batch", "5", "--ozaki"}},
      {Binary::DfDemo, {"--pc", "--tiles", "3", "--ozaki"}},
      {Binary::DfDemo, {"--tiles", "3", "--consume-emitted"}},
      {Binary::DfDemo, {"--tiles", "3", "--blas-digest"}},
      {Binary::DfDemo, {"--tiles", "3", "--digest-emitted"}},
      {Binary::DfDemo, {"--tiles", "3", "--profile"}},
      {Binary::DfDemo, {"--tiles", "3", "--sc-tangent"}},
      {Binary::DfDemo, {"--tiles", "3", "--gradient-source"}},
      {Binary::ScPseudodensity, {"--tiles", "3"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--kappa-act"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--kappa-full"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--dump-source-terms", "x.gradsrc"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--source-oracle", "x.srcgold"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--sc-hessian-kc"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--sc-hessian-kk-int"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--sc-hessian-kk"}},
      {Binary::PcTangent, {"--tiles", "3"}},
      {Binary::PcTangent, {"--tiles", "40", "--fd-steps", "3"}},
      {Binary::PcTangent, {"--tiles", "3", "--pc-hessian"}},
      {Binary::PcTangent, {"--tiles", "3", "--pc-hessian-vector"}},
  };
  for (const Entry& e : entries) {
    if (e.cublas && !kHaveCublasEmul) continue;  // a HIP build registers none of these
    const Result<Options> r = parse(e.b, e.args);
    EXPECT_TRUE(r.has_value()) << binaryName(e.b) << ": " << r.error().message;
  }
}

// --- CliDemoFlagTests ---------------------------------------------------------

TEST(CliDemoFlagTests, EachCheckSetsItsField) {
  const Result<Options> sc = parse(Binary::Demo, {"--sc-tangent"});
  ASSERT_TRUE(sc.has_value()) << sc.error().message;
  EXPECT_TRUE(sc->scTangent);
  EXPECT_FALSE(sc->gradientSource);
  const Result<Options> gs = parse(Binary::Demo, {"--gradient-source", "--blas-digest"});
  ASSERT_TRUE(gs.has_value()) << gs.error().message;
  EXPECT_TRUE(gs->gradientSource);
  EXPECT_FALSE(gs->scTangent);
}

TEST(CliDemoFlagTests, GradgoldRidesOnGradientSource) {
  // #87: the CI source term reads the orbital energies' response from the
  // sidecar, so --gradgold is accepted beside --gradient-source ...
  const Result<Options> gs =
      parse(Binary::Demo, {"--gradient-source", "--gradgold", "x.gradgold"});
  ASSERT_TRUE(gs.has_value()) << gs.error().message;
  EXPECT_TRUE(gs->gradientSource);
  EXPECT_EQ(gs->gradgoldPath, "x.gradgold");
  // ... and refused without it: nothing else in this demo reads it.
  for (const std::vector<std::string_view>& args :
       {std::vector<std::string_view>{"--gradgold", "x.gradgold"},
        std::vector<std::string_view>{"--sc-tangent", "--gradgold", "x.gradgold"}}) {
    const Result<Options> r = parse(Binary::Demo, args);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message,
              "--gradgold is the sidecar --gradient-source reads the orbital energies' "
              "response from; it does nothing on its own");
  }
}

TEST(CliDemoFlagTests, TheTwoChecksRefuseEachOther) {
  const Result<Options> r = parse(Binary::Demo, {"--sc-tangent", "--gradient-source"});
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(r.error().message,
            "--sc-tangent and --gradient-source are two different checks of the same "
            "derivative; run one at a time");
}

TEST(CliDemoFlagTests, EachCheckRefusesTheDigestsWithNoTangentPath) {
  for (const std::string_view which : {"--sc-tangent", "--gradient-source"}) {
    for (const std::vector<std::string_view>& digest :
         {std::vector<std::string_view>{"--ozaki"},
          std::vector<std::string_view>{"--blas-digest", "--fused-digest"}}) {
      std::vector<std::string_view> args{which};
      args.insert(args.end(), digest.begin(), digest.end());
      const Result<Options> r = parse(Binary::Demo, args);
      ASSERT_FALSE(r.has_value()) << which;
      EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
      EXPECT_EQ(r.error().message,
                std::format("{} runs the tangent RDM build, which is the emitted digest or "
                            "--blas-digest; not --cublas, --ozaki or --fused-digest",
                            which));
    }
  }
}

TEST(CliDemoFlagTests, ScTangentRefusesPcAndGradientSourceTakesIt) {
  // --sc-tangent is the SC tangent's finite difference and nothing else...
  const Result<Options> r = parse(Binary::Demo, {"--sc-tangent", "--pc"});
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(r.error().message,
            "--sc-tangent differentiates the strongly-contracted energy; it has no --pc path "
            "(the PC tangent's finite difference is nevpt2_pc_tangent)");
  // ...while --gradient-source --pc is the PC CI source term's duality (#74),
  // with or without the sidecar.
  for (const std::vector<std::string_view>& args :
       {std::vector<std::string_view>{"--gradient-source", "--pc"},
        std::vector<std::string_view>{"--gradient-source", "--pc", "--gradgold", "x.gradgold"}}) {
    const Result<Options> ok = parse(Binary::Demo, args);
    ASSERT_TRUE(ok.has_value()) << ok.error().message;
    EXPECT_TRUE(ok->gradientSource);
    EXPECT_TRUE(ok->common.pc);
  }
}

TEST(CliDemoFlagTests, EachCheckRefusesProfileItWouldNotPrint) {
  for (const std::string_view which : {"--sc-tangent", "--gradient-source"}) {
    const Result<Options> r = parse(Binary::Demo, {which, "--profile"});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported);
    EXPECT_EQ(r.error().message,
              std::format("{} prints no profile tables; --profile times the energy run", which));
  }
}

// --sc-hessian (#153), nevpt2_demo alone: one check at a time, the tangent
// RDM build's digests, SC only, no --profile, no --gradgold.
TEST(CliDemoFlagTests, ScHessianSetsItsFieldAndRefusesTheOtherChecks) {
  const Result<Options> h = parse(Binary::Demo, {"--sc-hessian", "--blas-digest"});
  ASSERT_TRUE(h.has_value()) << h.error().message;
  EXPECT_TRUE(h->scHessian);
  EXPECT_FALSE(h->scTangent);
  EXPECT_FALSE(h->gradientSource);
  for (const std::string_view other : {"--sc-tangent", "--gradient-source"}) {
    const Result<Options> r = parse(Binary::Demo, {"--sc-hessian", other});
    ASSERT_FALSE(r.has_value()) << other;
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message,
              std::format("--sc-hessian and {} are two different checks of the energy's CI "
                          "derivatives; run one at a time",
                          other));
  }
}

TEST(CliDemoFlagTests, ScHessianRefusesWhatTheTangentRefuses) {
  for (const std::vector<std::string_view>& digest :
       {std::vector<std::string_view>{"--ozaki"},
        std::vector<std::string_view>{"--blas-digest", "--fused-digest"}}) {
    std::vector<std::string_view> args{"--sc-hessian"};
    args.insert(args.end(), digest.begin(), digest.end());
    const Result<Options> r = parse(Binary::Demo, args);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message,
              "--sc-hessian runs the tangent RDM build, which is the emitted digest or "
              "--blas-digest; not --cublas, --ozaki or --fused-digest");
  }
  const Result<Options> pc = parse(Binary::Demo, {"--sc-hessian", "--pc"});
  ASSERT_FALSE(pc.has_value());
  EXPECT_EQ(pc.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(pc.error().message,
            "--sc-hessian differentiates the strongly-contracted energy twice; it has no --pc "
            "path");
  const Result<Options> prof = parse(Binary::Demo, {"--sc-hessian", "--profile"});
  ASSERT_FALSE(prof.has_value());
  EXPECT_EQ(prof.error().kind, ErrorKind::Unsupported);
  const Result<Options> gg = parse(Binary::Demo, {"--sc-hessian", "--gradgold", "x.gradgold"});
  ASSERT_FALSE(gg.has_value());
  EXPECT_EQ(gg.error().kind, ErrorKind::InvalidConfig);
}

// --sc-hessian-vector (#156), nevpt2_demo alone: one check at a time with all
// three others, and the same refusals as --sc-hessian.
TEST(CliDemoFlagTests, ScHessianVectorSetsItsFieldAndRefusesTheOtherChecks) {
  const Result<Options> h = parse(Binary::Demo, {"--sc-hessian-vector", "--blas-digest"});
  ASSERT_TRUE(h.has_value()) << h.error().message;
  EXPECT_TRUE(h->scHessianVector);
  EXPECT_FALSE(h->scHessian);
  EXPECT_FALSE(h->scTangent);
  EXPECT_FALSE(h->gradientSource);
  for (const std::string_view other : {"--sc-tangent", "--gradient-source", "--sc-hessian"}) {
    const Result<Options> r = parse(Binary::Demo, {"--sc-hessian-vector", other});
    ASSERT_FALSE(r.has_value()) << other;
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message,
              std::format("--sc-hessian-vector and {} are two different checks of the "
                          "energy's CI derivatives; run one at a time",
                          other));
  }
}

TEST(CliDemoFlagTests, ScHessianVectorRefusesWhatTheHessianRefuses) {
  for (const std::vector<std::string_view>& digest :
       {std::vector<std::string_view>{"--ozaki"},
        std::vector<std::string_view>{"--blas-digest", "--fused-digest"}}) {
    std::vector<std::string_view> args{"--sc-hessian-vector"};
    args.insert(args.end(), digest.begin(), digest.end());
    const Result<Options> r = parse(Binary::Demo, args);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message,
              "--sc-hessian-vector runs the tangent RDM build, which is the emitted digest or "
              "--blas-digest; not --cublas, --ozaki or --fused-digest");
  }
  const Result<Options> pc = parse(Binary::Demo, {"--sc-hessian-vector", "--pc"});
  ASSERT_FALSE(pc.has_value());
  EXPECT_EQ(pc.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(pc.error().message,
            "--sc-hessian-vector differentiates the strongly-contracted energy twice; it has "
            "no --pc path");
  const Result<Options> prof = parse(Binary::Demo, {"--sc-hessian-vector", "--profile"});
  ASSERT_FALSE(prof.has_value());
  EXPECT_EQ(prof.error().kind, ErrorKind::Unsupported);
  EXPECT_EQ(prof.error().message,
            "--sc-hessian-vector prints no profile tables; --profile times the energy run");
  const Result<Options> gg =
      parse(Binary::Demo, {"--sc-hessian-vector", "--gradgold", "x.gradgold"});
  ASSERT_FALSE(gg.has_value());
  EXPECT_EQ(gg.error().kind, ErrorKind::InvalidConfig);
}

TEST(CliDemoFlagTests, RdmTangentRunsBesideEitherCheck) {
  for (const std::string_view which : {"--sc-tangent", "--gradient-source"}) {
    const Result<Options> r = parse(Binary::Demo, {which, "--rdm-tangent"});
    EXPECT_TRUE(r.has_value()) << which << ": " << r.error().message;
  }
}

// --- CliDfDemoFlagTests -------------------------------------------------------

TEST(CliDfDemoFlagTests, BatchAndCheckBlocksSetTheirFields) {
  const Result<Options> r = parse(Binary::DfDemo, {"--batch", "0", "--check-blocks"});
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->batch, 0);
  EXPECT_TRUE(r->checkBlocks);
}

TEST(CliDfDemoFlagTests, BatchIsAWholeNonNegativeInteger) {
  const Result<Options> neg = parse(Binary::DfDemo, {"--batch", "-1"});
  ASSERT_FALSE(neg.has_value());
  EXPECT_EQ(neg.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(neg.error().message, "--batch must be >= 0");
  const Result<Options> bad = parse(Binary::DfDemo, {"--batch", "eight"});
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error().message, "--batch takes an integer, not \"eight\"");
}

// #162: the CI derivative's two checks run over the DF slabs too, under the
// integral-direct demo's shared rules, plus the density-fitted demo's own:
// no --pc on --gradient-source, no --check-blocks beside either, and no
// --gradgold (it reads no sidecar: the matrix refuses it, below).
TEST(CliDfDemoFlagTests, EachCheckSetsItsField) {
  const Result<Options> sc = parse(Binary::DfDemo, {"--sc-tangent", "--batch", "5"});
  ASSERT_TRUE(sc.has_value()) << sc.error().message;
  EXPECT_TRUE(sc->scTangent);
  EXPECT_FALSE(sc->gradientSource);
  EXPECT_EQ(sc->batch, 5);
  const Result<Options> gs = parse(Binary::DfDemo, {"--gradient-source", "--blas-digest"});
  ASSERT_TRUE(gs.has_value()) << gs.error().message;
  EXPECT_TRUE(gs->gradientSource);
  EXPECT_FALSE(gs->scTangent);
}

TEST(CliDfDemoFlagTests, TheTwoChecksRefuseEachOther) {
  const Result<Options> r = parse(Binary::DfDemo, {"--sc-tangent", "--gradient-source"});
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(r.error().message,
            "--sc-tangent and --gradient-source are two different checks of the same "
            "derivative; run one at a time");
}

TEST(CliDfDemoFlagTests, EachCheckRefusesTheDigestsWithNoTangentPath) {
  for (const std::string_view which : {"--sc-tangent", "--gradient-source"}) {
    for (const std::vector<std::string_view>& digest :
         {std::vector<std::string_view>{"--ozaki"},
          std::vector<std::string_view>{"--blas-digest", "--fused-digest"}}) {
      std::vector<std::string_view> args{which};
      args.insert(args.end(), digest.begin(), digest.end());
      const Result<Options> r = parse(Binary::DfDemo, args);
      ASSERT_FALSE(r.has_value()) << which;
      EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
      EXPECT_EQ(r.error().message,
                std::format("{} runs the tangent RDM build, which is the emitted digest or "
                            "--blas-digest; not --cublas, --ozaki or --fused-digest",
                            which));
    }
  }
  if (kHaveCublasEmul) {
    const Result<Options> r = parse(Binary::DfDemo, {"--gradient-source", "--cublas"});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
  }
}

TEST(CliDfDemoFlagTests, NeitherCheckHasAPcPath) {
  const Result<Options> sc = parse(Binary::DfDemo, {"--sc-tangent", "--pc"});
  ASSERT_FALSE(sc.has_value());
  EXPECT_EQ(sc.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(sc.error().message,
            "--sc-tangent differentiates the strongly-contracted energy; it has no --pc path "
            "(the PC tangent's finite difference is nevpt2_pc_tangent)");
  const Result<Options> gs = parse(Binary::DfDemo, {"--gradient-source", "--pc"});
  ASSERT_FALSE(gs.has_value());
  EXPECT_EQ(gs.error().kind, ErrorKind::Unsupported);
  EXPECT_EQ(gs.error().message,
            "--gradient-source --pc is the integral-direct demo's alone: the PC CI source term "
            "has no density-fitted slab path (nevpt2_demo --gradient-source --pc)");
}

TEST(CliDfDemoFlagTests, EachCheckRefusesProfileAndCheckBlocks) {
  for (const std::string_view which : {"--sc-tangent", "--gradient-source"}) {
    const Result<Options> p = parse(Binary::DfDemo, {which, "--profile"});
    ASSERT_FALSE(p.has_value());
    EXPECT_EQ(p.error().kind, ErrorKind::Unsupported);
    EXPECT_EQ(p.error().message,
              std::format("{} prints no profile tables; --profile times the energy run", which));
    const Result<Options> b = parse(Binary::DfDemo, {which, "--check-blocks"});
    ASSERT_FALSE(b.has_value());
    EXPECT_EQ(b.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(b.error().message,
              std::format("--check-blocks is the energy run's diagnostic; {} replaces that run",
                          which));
  }
}

TEST(CliDfDemoFlagTests, NoSidecarAndRdmTangentRunsBeside) {
  const Result<Options> g =
      parse(Binary::DfDemo, {"--gradient-source", "--gradgold", "x.gradgold"});
  ASSERT_FALSE(g.has_value());
  EXPECT_EQ(g.error().kind, ErrorKind::Unsupported);
  for (const std::string_view which : {"--sc-tangent", "--gradient-source"}) {
    const Result<Options> r = parse(Binary::DfDemo, {which, "--rdm-tangent"});
    EXPECT_TRUE(r.has_value()) << which << ": " << r.error().message;
  }
}

// --- CliScPseudodensityFlagTests ----------------------------------------------

TEST(CliScPseudodensityFlagTests, EachCheckSetsItsField) {
  const Result<Options> act = parse(Binary::ScPseudodensity, {"--kappa-act"});
  ASSERT_TRUE(act.has_value()) << act.error().message;
  EXPECT_TRUE(act->kappaAct);
  EXPECT_FALSE(act->kappaFull);
  const Result<Options> full =
      parse(Binary::ScPseudodensity, {"--kappa-full", "--gradgold", "x.gradgold"});
  ASSERT_TRUE(full.has_value()) << full.error().message;
  EXPECT_TRUE(full->kappaFull);
  EXPECT_EQ(full->gradgoldPath, "x.gradgold");
}

TEST(CliScPseudodensityFlagTests, KappaActAndKappaFullRefuseEachOther) {
  const Result<Options> r = parse(Binary::ScPseudodensity, {"--kappa-act", "--kappa-full"});
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(r.error().message,
            "--kappa-act and --kappa-full are two checks of the same source term; pass one");
}

TEST(CliScPseudodensityFlagTests, GradgoldNeedsKappaFull) {
  for (const std::vector<std::string_view>& args :
       {std::vector<std::string_view>{"--gradgold", "x.gradgold"},
        std::vector<std::string_view>{"--kappa-act", "--gradgold", "x.gradgold"}}) {
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message,
              "--gradgold is the sidecar --kappa-full, --sc-hessian-kc, --sc-hessian-kk-int, "
              "--sc-hessian-kk, --dump-source-terms and --source-oracle read; it does nothing on "
              "its own");
  }
}

TEST(CliScPseudodensityFlagTests, DumpSourceTermsSetsItsPath) {
  const Result<Options> r = parse(Binary::ScPseudodensity, {"--dump-source-terms", "x.gradsrc"});
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->dumpSourceTermsPath, "x.gradsrc");
  EXPECT_FALSE(r->kappaAct);
  EXPECT_FALSE(r->kappaFull);
  // It reads the sidecar too, so --gradgold is accepted beside it.
  const Result<Options> g = parse(Binary::ScPseudodensity,
                                  {"--dump-source-terms", "x.gradsrc", "--gradgold", "s.gradgold"});
  ASSERT_TRUE(g.has_value()) << g.error().message;
  EXPECT_EQ(g->gradgoldPath, "s.gradgold");
}

TEST(CliScPseudodensityFlagTests, DumpSourceTermsRefusesTheChecks) {
  for (const std::string_view check : {"--kappa-act", "--kappa-full"}) {
    const Result<Options> r =
        parse(Binary::ScPseudodensity, {"--dump-source-terms", "x.gradsrc", check});
    ASSERT_FALSE(r.has_value()) << check;
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message,
              "--dump-source-terms writes the source terms --kappa-act and --kappa-full check; "
              "run one at a time");
  }
}

TEST(CliScPseudodensityFlagTests, DumpSourceTermsNeedsAPath) {
  const Result<Options> r = parse(Binary::ScPseudodensity, {"--dump-source-terms"});
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
}

TEST(CliScPseudodensityFlagTests, OtherBinariesRefuseDumpSourceTerms) {
  // nevpt2_demo computes y_I as well (--gradient-source), but not Y_xy, which
  // needs the sidecar only nevpt2_sc_pseudodensity reads: half a dump is
  // refused rather than written.
  for (const Binary b : {Binary::Demo, Binary::DfDemo, Binary::PcTangent}) {
    const Result<Options> r = parse(b, {"--dump-source-terms", "x.gradsrc"});
    ASSERT_FALSE(r.has_value()) << binaryName(b);
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << binaryName(b);
    EXPECT_NE(r.error().message.find("only nevpt2_sc_pseudodensity"), std::string::npos)
        << r.error().message;
  }
}

TEST(CliScPseudodensityFlagTests, SourceOracleSetsItsPath) {
  const Result<Options> r = parse(Binary::ScPseudodensity, {"--source-oracle", "x.srcgold"});
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->sourceOraclePath, "x.srcgold");
  EXPECT_TRUE(r->dumpSourceTermsPath.empty());
  EXPECT_FALSE(r->kappaAct);
  EXPECT_FALSE(r->kappaFull);
  // It reads the sidecar too, so --gradgold is accepted beside it.
  const Result<Options> g = parse(Binary::ScPseudodensity,
                                  {"--source-oracle", "x.srcgold", "--gradgold", "s.gradgold"});
  ASSERT_TRUE(g.has_value()) << g.error().message;
  EXPECT_EQ(g->gradgoldPath, "s.gradgold");
}

// --pc (#74): PC-NEVPT2's source terms, so only the two jobs that assemble
// them take it; the identities and the two kappa checks are SC's.
TEST(CliScPseudodensityFlagTests, PcIsTheSourceTermJobsAlone) {
  for (const std::vector<std::string_view>& args :
       {std::vector<std::string_view>{"--pc", "--dump-source-terms", "x.gradsrc"},
        std::vector<std::string_view>{"--pc", "--source-oracle", "x_pc.srcgold"}}) {
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    EXPECT_TRUE(r->common.pc);
  }
  for (const std::vector<std::string_view>& args :
       {std::vector<std::string_view>{"--pc"}, std::vector<std::string_view>{"--pc", "--kappa-act"},
        std::vector<std::string_view>{"--pc", "--kappa-full"}}) {
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message,
              "--pc makes --dump-source-terms and --source-oracle PC-NEVPT2's; the identities, "
              "--kappa-act and --kappa-full are the SC energy's and have no --pc path");
  }
  // The dump builds the digests' tangent with --pc as without: exact digests.
  const Result<Options> r =
      parse(Binary::ScPseudodensity, {"--pc", "--dump-source-terms", "x.gradsrc", "--ozaki"});
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
}

TEST(CliScPseudodensityFlagTests, SourceOracleRefusesTheOtherJobs) {
  for (const std::vector<std::string_view>& other :
       {std::vector<std::string_view>{"--kappa-act"}, std::vector<std::string_view>{"--kappa-full"},
        std::vector<std::string_view>{"--dump-source-terms", "x.gradsrc"}}) {
    std::vector<std::string_view> args{"--source-oracle", "x.srcgold"};
    args.insert(args.end(), other.begin(), other.end());
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value()) << other.front();
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message,
              "--source-oracle compares the source terms --dump-source-terms writes and "
              "--kappa-act / --kappa-full check; run one at a time");
  }
}

// --sc-hessian-kc (#184), nevpt2_sc_pseudodensity alone: the mixed block
// H_kc, one job at a time, SC only, exact digests only, with or without the
// sidecar (--gradgold rides on it).
TEST(CliScPseudodensityFlagTests, ScHessianKcSetsItsFieldAndTakesGradgold) {
  const Result<Options> r = parse(Binary::ScPseudodensity, {"--sc-hessian-kc"});
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_TRUE(r->scHessianKc);
  EXPECT_FALSE(r->kappaAct);
  EXPECT_FALSE(r->kappaFull);
  const Result<Options> g = parse(
      Binary::ScPseudodensity, {"--sc-hessian-kc", "--gradgold", "s.gradgold", "--blas-digest"});
  ASSERT_TRUE(g.has_value()) << g.error().message;
  EXPECT_EQ(g->gradgoldPath, "s.gradgold");
}

TEST(CliScPseudodensityFlagTests, ScHessianKcRefusesTheOtherJobs) {
  for (const std::vector<std::string_view>& other :
       {std::vector<std::string_view>{"--kappa-act"}, std::vector<std::string_view>{"--kappa-full"},
        std::vector<std::string_view>{"--dump-source-terms", "x.gradsrc"},
        std::vector<std::string_view>{"--source-oracle", "x.srcgold"}}) {
    std::vector<std::string_view> args{"--sc-hessian-kc"};
    args.insert(args.end(), other.begin(), other.end());
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value()) << other.front();
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message,
              "--sc-hessian-kc checks the energy's mixed orbital-CI second derivative; "
              "--kappa-act, --kappa-full, --dump-source-terms and --source-oracle are "
              "first-order jobs -- run one at a time");
  }
}

TEST(CliScPseudodensityFlagTests, ScHessianKcRefusesPcAndTheApproximateDigests) {
  const Result<Options> pc = parse(Binary::ScPseudodensity, {"--sc-hessian-kc", "--pc"});
  ASSERT_FALSE(pc.has_value());
  EXPECT_EQ(pc.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(pc.error().message,
            "--sc-hessian-kc differentiates the strongly-contracted energy twice; PC's half of "
            "H_kc has no path (it waits on the PC projector terms, #152 / #155)");
  for (const std::vector<std::string_view>& digest :
       {std::vector<std::string_view>{"--cublas"}, std::vector<std::string_view>{"--ozaki"},
        std::vector<std::string_view>{"--fused-digest", "--blas-digest"}}) {
    if (digest.front() == "--cublas" && !kHaveCublasEmul) continue;
    std::vector<std::string_view> args{"--sc-hessian-kc"};
    args.insert(args.end(), digest.begin(), digest.end());
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value()) << digest.front();
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig) << digest.front();
    EXPECT_EQ(r.error().message,
              "--sc-hessian-kc runs the tangent RDM build and the digests' tangent, which are "
              "the emitted digest or --blas-digest; not --cublas, --ozaki or --fused-digest");
  }
}

// --sc-hessian-kk-int (#154), nevpt2_sc_pseudodensity alone: the orbital
// block's integral curvature, one job at a time, SC only, exact digests only;
// --gradgold rides on it.
TEST(CliScPseudodensityFlagTests, ScHessianKkIntSetsItsFieldAndTakesGradgold) {
  const Result<Options> r = parse(Binary::ScPseudodensity, {"--sc-hessian-kk-int"});
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_TRUE(r->scHessianKkInt);
  EXPECT_FALSE(r->scHessianKc);
  EXPECT_FALSE(r->kappaFull);
  const Result<Options> g = parse(Binary::ScPseudodensity,
                                  {"--sc-hessian-kk-int", "--gradgold", "s.gradgold",
                                   "--blas-digest"});
  ASSERT_TRUE(g.has_value()) << g.error().message;
  EXPECT_EQ(g->gradgoldPath, "s.gradgold");
}

TEST(CliScPseudodensityFlagTests, ScHessianKkIntRefusesTheOtherJobs) {
  for (const std::vector<std::string_view>& other :
       {std::vector<std::string_view>{"--kappa-act"}, std::vector<std::string_view>{"--kappa-full"},
        std::vector<std::string_view>{"--sc-hessian-kc"},
        std::vector<std::string_view>{"--dump-source-terms", "x.gradsrc"},
        std::vector<std::string_view>{"--source-oracle", "x.srcgold"}}) {
    std::vector<std::string_view> args{"--sc-hessian-kk-int"};
    args.insert(args.end(), other.begin(), other.end());
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value()) << other.front();
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message,
              "--sc-hessian-kk-int checks the orbital Hessian's integral curvature; "
              "--kappa-act, --kappa-full, --sc-hessian-kc, --dump-source-terms and "
              "--source-oracle are jobs of their own -- run one at a time");
  }
}

TEST(CliScPseudodensityFlagTests, ScHessianKkIntRefusesPcAndTheApproximateDigests) {
  const Result<Options> pc = parse(Binary::ScPseudodensity, {"--sc-hessian-kk-int", "--pc"});
  ASSERT_FALSE(pc.has_value());
  EXPECT_EQ(pc.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(pc.error().message,
            "--sc-hessian-kk-int differentiates the strongly-contracted energy twice; PC's "
            "orbital Hessian has no path");
  for (const std::vector<std::string_view>& digest :
       {std::vector<std::string_view>{"--cublas"}, std::vector<std::string_view>{"--ozaki"},
        std::vector<std::string_view>{"--fused-digest", "--blas-digest"}}) {
    if (digest.front() == "--cublas" && !kHaveCublasEmul) continue;
    std::vector<std::string_view> args{"--sc-hessian-kk-int"};
    args.insert(args.end(), digest.begin(), digest.end());
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value()) << digest.front();
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig) << digest.front();
    EXPECT_EQ(r.error().message,
              "--sc-hessian-kk-int builds the digests' tangent at the second-order integrals, "
              "which is the emitted digest or --blas-digest; not --cublas, --ozaki or "
              "--fused-digest");
  }
}

// --sc-hessian-kk (#158), nevpt2_sc_pseudodensity alone: the orbital block's
// sandwich and the whole Eddot_kk, one job at a time, SC only, exact digests
// only; --gradgold rides on it.
TEST(CliScPseudodensityFlagTests, ScHessianKkSetsItsFieldAndTakesGradgold) {
  const Result<Options> r = parse(Binary::ScPseudodensity, {"--sc-hessian-kk"});
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_TRUE(r->scHessianKk);
  EXPECT_FALSE(r->scHessianKkInt);
  const Result<Options> g = parse(Binary::ScPseudodensity,
                                  {"--sc-hessian-kk", "--gradgold", "s.gradgold", "--blas-digest"});
  ASSERT_TRUE(g.has_value()) << g.error().message;
  EXPECT_EQ(g->gradgoldPath, "s.gradgold");
}

TEST(CliScPseudodensityFlagTests, ScHessianKkRefusesTheOtherJobs) {
  for (const std::vector<std::string_view>& other :
       {std::vector<std::string_view>{"--kappa-act"}, std::vector<std::string_view>{"--kappa-full"},
        std::vector<std::string_view>{"--sc-hessian-kc"},
        std::vector<std::string_view>{"--sc-hessian-kk-int"},
        std::vector<std::string_view>{"--dump-source-terms", "x.gradsrc"},
        std::vector<std::string_view>{"--source-oracle", "x.srcgold"}}) {
    std::vector<std::string_view> args{"--sc-hessian-kk"};
    args.insert(args.end(), other.begin(), other.end());
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value()) << other.front();
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message,
              "--sc-hessian-kk checks the orbital Hessian block; --kappa-act, --kappa-full, "
              "--sc-hessian-kc, --sc-hessian-kk-int, --dump-source-terms and --source-oracle "
              "are jobs of their own -- run one at a time");
  }
}

TEST(CliScPseudodensityFlagTests, ScHessianKkRefusesPcAndTheApproximateDigests) {
  const Result<Options> pc = parse(Binary::ScPseudodensity, {"--sc-hessian-kk", "--pc"});
  ASSERT_FALSE(pc.has_value());
  EXPECT_EQ(pc.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(pc.error().message,
            "--sc-hessian-kk differentiates the strongly-contracted energy twice; PC's orbital "
            "Hessian has no path");
  for (const std::vector<std::string_view>& digest :
       {std::vector<std::string_view>{"--cublas"}, std::vector<std::string_view>{"--ozaki"},
        std::vector<std::string_view>{"--fused-digest", "--blas-digest"}}) {
    if (digest.front() == "--cublas" && !kHaveCublasEmul) continue;
    std::vector<std::string_view> args{"--sc-hessian-kk"};
    args.insert(args.end(), digest.begin(), digest.end());
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value()) << digest.front();
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig) << digest.front();
    EXPECT_EQ(r.error().message,
              "--sc-hessian-kk builds the digests' tangents along each direction and at the "
              "second-order integrals, which is the emitted digest or --blas-digest; not "
              "--cublas, --ozaki or --fused-digest");
  }
}

TEST(CliScPseudodensityFlagTests, OtherBinariesRefuseScHessianKk) {
  for (const Binary b : {Binary::Demo, Binary::DfDemo, Binary::PcTangent}) {
    const Result<Options> r = parse(b, {"--sc-hessian-kk"});
    ASSERT_FALSE(r.has_value()) << binaryName(b);
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << binaryName(b);
    EXPECT_NE(r.error().message.find("only nevpt2_sc_pseudodensity"), std::string::npos)
        << r.error().message;
  }
}

TEST(CliScPseudodensityFlagTests, OtherBinariesRefuseScHessianKkInt) {
  for (const Binary b : {Binary::Demo, Binary::DfDemo, Binary::PcTangent}) {
    const Result<Options> r = parse(b, {"--sc-hessian-kk-int"});
    ASSERT_FALSE(r.has_value()) << binaryName(b);
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << binaryName(b);
    EXPECT_NE(r.error().message.find("only nevpt2_sc_pseudodensity"), std::string::npos)
        << r.error().message;
  }
}

TEST(CliScPseudodensityFlagTests, OtherBinariesRefuseScHessianKc) {
  for (const Binary b : {Binary::Demo, Binary::DfDemo, Binary::PcTangent}) {
    const Result<Options> r = parse(b, {"--sc-hessian-kc"});
    ASSERT_FALSE(r.has_value()) << binaryName(b);
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << binaryName(b);
    EXPECT_NE(r.error().message.find("only nevpt2_sc_pseudodensity"), std::string::npos)
        << r.error().message;
  }
}

TEST(CliScPseudodensityFlagTests, OtherBinariesRefuseSourceOracle) {
  for (const Binary b : {Binary::Demo, Binary::DfDemo, Binary::PcTangent}) {
    const Result<Options> r = parse(b, {"--source-oracle", "x.srcgold"});
    ASSERT_FALSE(r.has_value()) << binaryName(b);
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << binaryName(b);
  }
}

// The density-fitted orbital source term (#165; docs/gradients.md, Sec. 6.2):
// nevpt2_sc_pseudodensity takes a `*_df` golden with no flag of its own --
// the sidecar's NEVPT2F1 magic says the case is fitted -- so there is no
// --df anywhere, and the orbital-source-term flags stay that binary's alone:
// nevpt2_df_demo refuses them, naming it, rather than growing a second copy.
TEST(CliScPseudodensityFlagTests, ADensityFittedCaseTakesNoFlag) {
  for (const Binary b : kBinaries) {
    const Result<Options> r = parse(b, {"--df"});
    ASSERT_FALSE(r.has_value()) << binaryName(b);
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig) << binaryName(b);
    EXPECT_TRUE(r.error().message.starts_with("unrecognized argument \"--df\""))
        << r.error().message;
  }
  for (const std::vector<std::string_view>& args :
       {std::vector<std::string_view>{"--kappa-act"}, std::vector<std::string_view>{"--kappa-full"},
        std::vector<std::string_view>{"--dump-source-terms", "x.gradsrc"}}) {
    const Result<Options> df = parse(Binary::DfDemo, args);
    ASSERT_FALSE(df.has_value()) << args.front();
    EXPECT_EQ(df.error().kind, ErrorKind::Unsupported) << args.front();
    EXPECT_NE(df.error().message.find("nevpt2_sc_pseudodensity"), std::string::npos)
        << df.error().message;
    // And the binary that does take them takes them for a fitted golden
    // exactly as for any other: the flags carry no flavour.
    const Result<Options> sc = parse(Binary::ScPseudodensity, args);
    EXPECT_TRUE(sc.has_value()) << args.front() << ": " << sc.error().message;
  }
}

TEST(CliScPseudodensityFlagTests, TakesAnyDigestTheDemosTake) {
  // The identities, --kappa-act and --source-oracle build no tangent RDMs, so
  // the approximate digests are theirs to run.
  for (const std::vector<std::string_view>& mode :
       {std::vector<std::string_view>{}, std::vector<std::string_view>{"--kappa-act"},
        std::vector<std::string_view>{"--source-oracle", "x.srcgold"}}) {
    for (const std::string_view digest : {"--ozaki", "--cublas", "--fused-digest"}) {
      // --cublas is refused outright on a build without the emulated digest (HIP).
      if (digest == "--cublas" && !kHaveCublasEmul) continue;
      std::vector<std::string_view> args = mode;
      args.push_back(digest);
      // --fused-digest needs a GEMM digest to fuse.
      if (digest == "--fused-digest") args.push_back("--blas-digest");
      const Result<Options> r = parse(Binary::ScPseudodensity, args);
      EXPECT_TRUE(r.has_value()) << digest << ": " << r.error().message;
    }
  }
}

TEST(CliScPseudodensityFlagTests, DigestTangentModesRefuseAnApproximateDigest) {
  // --kappa-full and --dump-source-terms build the digests' tangent along a
  // rotation (#88), which is formed from an exact digest only.
  for (const std::vector<std::string_view>& mode :
       {std::vector<std::string_view>{"--kappa-full"},
        std::vector<std::string_view>{"--dump-source-terms", "x.gradsrc"}}) {
    for (const std::string_view digest : {"--ozaki", "--cublas", "--fused-digest"}) {
      // --cublas is refused outright on a build without the emulated digest (HIP).
      if (digest == "--cublas" && !kHaveCublasEmul) continue;
      std::vector<std::string_view> args = mode;
      args.push_back(digest);
      if (digest == "--fused-digest") args.push_back("--blas-digest");
      const Result<Options> r = parse(Binary::ScPseudodensity, args);
      ASSERT_FALSE(r.has_value()) << mode[0] << " " << digest;
      EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
      EXPECT_EQ(r.error().message,
                std::format("{} builds the f3ac/f3ca digests' tangent, which is the emitted "
                            "digest or --blas-digest; not --cublas, --ozaki or --fused-digest",
                            mode[0]));
    }
    // And the exact ones are accepted.
    for (const std::string_view digest : {"--blas-digest", "--digest-emitted"}) {
      std::vector<std::string_view> args = mode;
      args.push_back(digest);
      const Result<Options> r = parse(Binary::ScPseudodensity, args);
      EXPECT_TRUE(r.has_value()) << mode[0] << " " << digest << ": " << r.error().message;
    }
  }
}

// --- CliPcTangentFlagTests ----------------------------------------------------

TEST(CliPcTangentFlagTests, FdStepsIsThreeToEight) {
  for (const int steps : {kMinFdSteps, kDefaultFdSteps, kMaxFdSteps}) {
    const std::string v = std::to_string(steps);
    const Result<Options> r = parse(Binary::PcTangent, {"--fd-steps", v});
    ASSERT_TRUE(r.has_value()) << steps << ": " << r.error().message;
    EXPECT_EQ(r->fdSteps, steps);
  }
  EXPECT_EQ(kMinFdSteps, 3);
  EXPECT_EQ(kMaxFdSteps, 8);
  for (const std::string_view steps : {"2", "9", "0", "-3"}) {
    const Result<Options> r = parse(Binary::PcTangent, {"--fd-steps", steps});
    ASSERT_FALSE(r.has_value()) << steps;
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message, "--fd-steps must be in 3..8");
  }
}

TEST(CliPcTangentFlagTests, FdStartPowIsOneToThirty) {
  for (const std::string_view pow : {"1", "12", "30"}) {
    const Result<Options> r = parse(Binary::PcTangent, {"--fd-start-pow", pow});
    ASSERT_TRUE(r.has_value()) << pow << ": " << r.error().message;
    EXPECT_EQ(std::to_string(r->fdStartPow), pow);
  }
  for (const std::string_view pow : {"0", "31", "-12"}) {
    const Result<Options> r = parse(Binary::PcTangent, {"--fd-start-pow", pow});
    ASSERT_FALSE(r.has_value()) << pow;
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message, "--fd-start-pow must be in 1..30");
  }
}

TEST(CliPcTangentFlagTests, PcHessianIsAPlainSwitchWithTheLadderFlags) {
  const Result<Options> off = parse(Binary::PcTangent, {});
  ASSERT_TRUE(off.has_value()) << off.error().message;
  EXPECT_FALSE(off->pcHessian);
  const Result<Options> on =
      parse(Binary::PcTangent, {"--pc-hessian", "--fd-steps", "5", "--fd-start-pow", "13"});
  ASSERT_TRUE(on.has_value()) << on.error().message;
  EXPECT_TRUE(on->pcHessian);
  EXPECT_EQ(on->fdSteps, 5);
  EXPECT_EQ(on->fdStartPow, 13);
  // Like the rest of the binary, it runs no approximate digest.
  for (const std::string_view flag : {"--cublas", "--ozaki", "--fused-digest"}) {
    const Result<Options> r = parse(Binary::PcTangent, {"--pc-hessian", flag});
    ASSERT_FALSE(r.has_value()) << flag;
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << flag;
  }
}

TEST(CliPcTangentFlagTests, PcHessianVectorIsAPlainSwitchAndNotWithPcHessian) {
  const Result<Options> off = parse(Binary::PcTangent, {});
  ASSERT_TRUE(off.has_value()) << off.error().message;
  EXPECT_FALSE(off->pcHessianVector);
  const Result<Options> on = parse(
      Binary::PcTangent, {"--pc-hessian-vector", "--fd-steps", "5", "--fd-start-pow", "13"});
  ASSERT_TRUE(on.has_value()) << on.error().message;
  EXPECT_TRUE(on->pcHessianVector);
  EXPECT_FALSE(on->pcHessian);
  EXPECT_EQ(on->fdSteps, 5);
  EXPECT_EQ(on->fdStartPow, 13);
  // Two checks of the same second derivative: one at a time.
  const Result<Options> both = parse(Binary::PcTangent, {"--pc-hessian", "--pc-hessian-vector"});
  ASSERT_FALSE(both.has_value());
  EXPECT_EQ(both.error().kind, ErrorKind::InvalidConfig);
  EXPECT_NE(both.error().message.find("one at a time"), std::string::npos)
      << both.error().message;
  // Like the rest of the binary, it runs no approximate digest.
  for (const std::string_view flag : {"--cublas", "--ozaki", "--fused-digest"}) {
    const Result<Options> r = parse(Binary::PcTangent, {"--pc-hessian-vector", flag});
    ASSERT_FALSE(r.has_value()) << flag;
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << flag;
  }
}

TEST(CliPcTangentFlagTests, ABadValueIsAConfigErrorNamingTheFlag) {
  for (const std::string_view flag : {"--fd-steps", "--fd-start-pow"}) {
    const Result<Options> r = parse(Binary::PcTangent, {flag, "4.5"});
    ASSERT_FALSE(r.has_value()) << flag;
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message, std::format("{} takes an integer, not \"4.5\"", flag));
  }
}

}  // namespace nevpt2::test::cli
