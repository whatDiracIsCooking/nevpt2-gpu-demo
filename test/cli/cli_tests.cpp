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
//                                --sc-hessian-kk / --sc-source-tangent /
//                                --sc-parameter-hessian and its
//                                --parameter-hessian-out /
//                                --parameter-hessian-coords (#222)
//   CliPcTangentFlagTests        --fd-steps / --fd-start-pow / --pc-hessian /
//                                --pc-hessian-vector
//   CliHessianRefusalTests       every Hessian flag of every binary at once
//                                (#157): one binary each, refused by the
//                                others naming it, every approximate digest
//                                Unsupported beside it, the exact ones taken;
//                                and the same for each orbital-block flag
//                                with --pc (#203)
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

// The Unsupported message any Hessian flag gets beside an approximate digest
// (#157; docs/hessian-theory.md, Sec. 9.1), `digest` the one it names.
std::string hessianDigestMessage(const std::string_view hessian, const std::string_view digest) {
  return std::format(
      "{} with {}: a second derivative built from an approximate digest is not the derivative "
      "of anything the energy stage computed, so every Hessian runs the emitted digest or "
      "--blas-digest alone; not --cublas, --ozaki or --fused-digest (docs/hessian-theory.md, "
      "Sec. 9.1)",
      hessian, digest);
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
    MatrixRow{"--sc-source-tangent", "S"},
    MatrixRow{"--sc-parameter-hessian", "S"},
    MatrixRow{"--parameter-hessian-out", "S"},
    MatrixRow{"--parameter-hessian-coords", "S"},
    MatrixRow{"--gradgold", "DSP"},         MatrixRow{"--erigold", "S"},
    MatrixRow{"--dump-source-terms", "S"},
    MatrixRow{"--probed-y", "S"},
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
      {Binary::Demo, {"--tiles", "3", "--sc-hessian", "--gradgold", "x.gradgold"}},
      {Binary::Demo, {"--tiles", "3", "--sc-hessian-vector", "--gradgold", "x.gradgold"}},
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
      {Binary::ScPseudodensity,
       {"--tiles", "3", "--dump-source-terms", "x.gradsrc", "--probed-y"}},
      {Binary::ScPseudodensity,
       {"--tiles", "3", "--pc", "--dump-source-terms", "x.gradsrc", "--probed-y"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--source-oracle", "x.srcgold"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--sc-hessian-kc"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--sc-hessian-kk-int"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--sc-hessian-kk"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--pc", "--sc-hessian-kc"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--pc", "--sc-hessian-kk-int"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--pc", "--sc-hessian-kk"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--sc-source-tangent", "x.pertgold"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--sc-hessian-kk-int", "--erigold", "x.erigold"}},
      {Binary::ScPseudodensity,
       {"--tiles", "3", "--pc", "--sc-hessian-kk-int", "--erigold", "x.erigold"}},
      {Binary::ScPseudodensity,
       {"--tiles", "3", "--sc-parameter-hessian", "x.respgold", "--erigold", "x.erigold",
        "--parameter-hessian-out", "x.parhess"}},
      {Binary::ScPseudodensity,
       {"--tiles", "3", "--sc-parameter-hessian", "x.respgold", "--erigold", "x.erigold",
        "--parameter-hessian-coords", "2,5", "--parameter-hessian-out", "x.parhess"}},
      {Binary::ScPseudodensity,
       {"--tiles", "3", "--sc-parameter-hessian", "x.respgold", "--erigold", "x.erigold"}},
      {Binary::ScPseudodensity, {"--tiles", "3", "--sc-parameter-hessian", "x.respgold"}},
      {Binary::PcTangent, {"--tiles", "3"}},
      {Binary::PcTangent, {"--tiles", "40", "--fd-steps", "3"}},
      {Binary::PcTangent, {"--tiles", "3", "--pc-hessian"}},
      {Binary::PcTangent, {"--tiles", "3", "--pc-hessian-vector"}},
      {Binary::PcTangent, {"--tiles", "3", "--pc-hessian", "--gradgold", "x.gradgold"}},
      {Binary::PcTangent, {"--tiles", "3", "--pc-hessian-vector", "--gradgold", "x.gradgold"}},
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

TEST(CliDemoFlagTests, GradgoldRidesOnGradientSourceAndTheCiHessian) {
  // #87: the CI source term reads the orbital energies' response from the
  // sidecar, so --gradgold is accepted beside --gradient-source -- and, #214,
  // beside --sc-hessian / --sc-hessian-vector, whose FULL mode it selects ...
  for (const std::string_view which :
       {"--gradient-source", "--sc-hessian", "--sc-hessian-vector"}) {
    const Result<Options> r = parse(Binary::Demo, {which, "--gradgold", "x.gradgold"});
    ASSERT_TRUE(r.has_value()) << which << ": " << r.error().message;
    EXPECT_EQ(r->gradientSource, which == "--gradient-source");
    EXPECT_EQ(r->scHessian, which == "--sc-hessian");
    EXPECT_EQ(r->scHessianVector, which == "--sc-hessian-vector");
    EXPECT_EQ(r->gradgoldPath, "x.gradgold");
  }
  // ... and refused without one of them: nothing else in this demo reads it.
  for (const std::vector<std::string_view>& args :
       {std::vector<std::string_view>{"--gradgold", "x.gradgold"},
        std::vector<std::string_view>{"--sc-tangent", "--gradgold", "x.gradgold"}}) {
    const Result<Options> r = parse(Binary::Demo, args);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message,
              "--gradgold is the sidecar --gradient-source, --sc-hessian and "
              "--sc-hessian-vector read the orbital energies' response from; it does nothing "
              "on its own");
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
// RDM build's digests, SC only, no --profile; --gradgold beside it selects
// the FULL mode (#214).
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
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported);
    EXPECT_EQ(r.error().message, hessianDigestMessage("--sc-hessian", digest.back()));
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
  // #214: --gradgold beside it is no longer an invalid configuration -- it
  // selects the FULL mode (CliDemoFlagTests.GradgoldRidesOnGradientSourceAndTheCiHessian).
  const Result<Options> gg = parse(Binary::Demo, {"--sc-hessian", "--gradgold", "x.gradgold"});
  ASSERT_TRUE(gg.has_value()) << gg.error().message;
  EXPECT_EQ(gg->gradgoldPath, "x.gradgold");
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
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported);
    EXPECT_EQ(r.error().message, hessianDigestMessage("--sc-hessian-vector", digest.back()));
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
  // #214: the FULL mode, as --sc-hessian's.
  const Result<Options> gg =
      parse(Binary::Demo, {"--sc-hessian-vector", "--gradgold", "x.gradgold"});
  ASSERT_TRUE(gg.has_value()) << gg.error().message;
  EXPECT_EQ(gg->gradgoldPath, "x.gradgold");
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
              "--sc-hessian-kk, --sc-source-tangent, --sc-parameter-hessian, --dump-source-terms "
              "and --source-oracle read; it does nothing on its own");
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

TEST(CliScPseudodensityFlagTests, ProbedYRidesOnTheDump) {
  // #240: the probed Y is the one-pass Y's oracle, compared inside the dump.
  const Result<Options> r =
      parse(Binary::ScPseudodensity, {"--dump-source-terms", "x.gradsrc", "--probed-y"});
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_TRUE(r->probedY);
  const Result<Options> off = parse(Binary::ScPseudodensity, {"--dump-source-terms", "x.gradsrc"});
  ASSERT_TRUE(off.has_value()) << off.error().message;
  EXPECT_FALSE(off->probedY);
  for (const std::vector<std::string_view>& args :
       {std::vector<std::string_view>{"--probed-y"},
        std::vector<std::string_view>{"--probed-y", "--kappa-full"},
        std::vector<std::string_view>{"--probed-y", "--source-oracle", "x.srcgold"}}) {
    const Result<Options> bad = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(bad.has_value()) << args.size();
    EXPECT_EQ(bad.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(bad.error().message,
              "--probed-y holds the one-pass Y --dump-source-terms writes to the probed one; it "
              "does nothing without --dump-source-terms");
  }
  // The other binaries refuse it, naming this one.
  for (const Binary b : {Binary::Demo, Binary::DfDemo, Binary::PcTangent}) {
    const Result<Options> o = parse(b, {"--probed-y"});
    ASSERT_FALSE(o.has_value()) << binaryName(b);
    EXPECT_EQ(o.error().kind, ErrorKind::Unsupported) << binaryName(b);
  }
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

// --pc (#74, #200, #201, #202): PC-NEVPT2's source terms, orbital
// curvature, mixed block and orbital sandwich, so only the five jobs that
// assemble them take it; the identities and the two kappa checks are SC's.
TEST(CliScPseudodensityFlagTests, PcIsTheSourceTermAndHessianJobsAlone) {
  for (const std::vector<std::string_view>& args :
       {std::vector<std::string_view>{"--pc", "--dump-source-terms", "x.gradsrc"},
        std::vector<std::string_view>{"--pc", "--source-oracle", "x_pc.srcgold"},
        std::vector<std::string_view>{"--pc", "--sc-hessian-kk-int"},
        std::vector<std::string_view>{"--pc", "--sc-hessian-kc"},
        std::vector<std::string_view>{"--pc", "--sc-hessian-kk"},
        std::vector<std::string_view>{"--pc", "--sc-source-tangent", "p.pertgold"}}) {
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
              "--pc makes --dump-source-terms, --source-oracle, --sc-hessian-kc, "
              "--sc-hessian-kk-int, --sc-hessian-kk and --sc-source-tangent PC-NEVPT2's; the "
              "identities, --kappa-act and --kappa-full are the SC energy's and have no --pc "
              "path");
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
// H_kc, one job at a time, exact digests only (with --pc, PC's, #201), with
// or without the sidecar (--gradgold rides on it).
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

// With --pc it is PC's mixed block (#201), under the same digest rule and the
// same one-job rule.
TEST(CliScPseudodensityFlagTests, ScHessianKcTakesPcAndRefusesTheApproximateDigests) {
  const Result<Options> pc = parse(Binary::ScPseudodensity, {"--sc-hessian-kc", "--pc"});
  ASSERT_TRUE(pc.has_value()) << pc.error().message;
  EXPECT_TRUE(pc->scHessianKc);
  EXPECT_TRUE(pc->common.pc);
  for (const bool withPc : {false, true}) {
    for (const std::vector<std::string_view>& digest :
         {std::vector<std::string_view>{"--cublas"}, std::vector<std::string_view>{"--ozaki"},
          std::vector<std::string_view>{"--fused-digest", "--blas-digest"}}) {
      if (digest.front() == "--cublas" && !kHaveCublasEmul) continue;
      std::vector<std::string_view> args{"--sc-hessian-kc"};
      if (withPc) args.push_back("--pc");
      args.insert(args.end(), digest.begin(), digest.end());
      const Result<Options> r = parse(Binary::ScPseudodensity, args);
      ASSERT_FALSE(r.has_value()) << digest.front();
      EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << digest.front();
      EXPECT_EQ(r.error().message, hessianDigestMessage("--sc-hessian-kc", digest.front()));
    }
  }
  const Result<Options> both =
      parse(Binary::ScPseudodensity, {"--sc-hessian-kc", "--pc", "--sc-hessian-kk-int"});
  ASSERT_FALSE(both.has_value());
  EXPECT_EQ(both.error().kind, ErrorKind::InvalidConfig);
}

// --sc-hessian-kk-int (#154), nevpt2_sc_pseudodensity alone: the orbital
// block's integral curvature, one job at a time, exact digests only (with
// --pc, PC's, #200); --gradgold rides on it.
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

// --erigold (#180): the full MO-integral tensor --sc-hessian-kk-int's K2
// reads, with or without --pc, and nothing else's.
TEST(CliScPseudodensityFlagTests, ErigoldRidesOnScHessianKkIntAndScParameterHessianAlone) {
  for (const std::vector<std::string_view>& args :
       {std::vector<std::string_view>{"--sc-hessian-kk-int", "--erigold", "x.erigold"},
        std::vector<std::string_view>{"--erigold", "x.erigold", "--pc", "--sc-hessian-kk-int",
                                      "--gradgold", "s.gradgold"}}) {
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    EXPECT_TRUE(r->scHessianKkInt);
    EXPECT_EQ(r->erigoldPath, "x.erigold");
  }
  // #222: the parameter Hessian's K2 reads it too.
  const Result<Options> ph = parse(Binary::ScPseudodensity, {"--erigold", "x.erigold",
                                                             "--sc-parameter-hessian", "r"});
  ASSERT_TRUE(ph.has_value()) << ph.error().message;
  EXPECT_EQ(ph->erigoldPath, "x.erigold");
  for (const std::vector<std::string_view>& args :
       {std::vector<std::string_view>{"--erigold", "x.erigold"},
        std::vector<std::string_view>{"--sc-hessian-kk", "--erigold", "x.erigold"},
        std::vector<std::string_view>{"--kappa-full", "--erigold", "x.erigold"},
        std::vector<std::string_view>{"--sc-hessian-kc", "--erigold", "x.erigold"}}) {
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value()) << args[0];
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message,
              "--erigold is the full MO-integral tensor --sc-hessian-kk-int's and "
              "--sc-parameter-hessian's K2 reads for a pair with no active-active member; it "
              "does nothing on its own or beside another job");
  }
  for (const Binary b : {Binary::Demo, Binary::DfDemo, Binary::PcTangent}) {
    const Result<Options> r = parse(b, {"--erigold", "x.erigold"});
    ASSERT_FALSE(r.has_value()) << binaryName(b);
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << binaryName(b);
    EXPECT_NE(r.error().message.find("only nevpt2_sc_pseudodensity"), std::string::npos)
        << r.error().message;
  }
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

// With --pc it is PC's orbital curvature (#200), under the same digest rule.
TEST(CliScPseudodensityFlagTests, ScHessianKkIntTakesPcAndRefusesTheApproximateDigests) {
  const Result<Options> pc = parse(Binary::ScPseudodensity, {"--sc-hessian-kk-int", "--pc"});
  ASSERT_TRUE(pc.has_value()) << pc.error().message;
  EXPECT_TRUE(pc->scHessianKkInt);
  EXPECT_TRUE(pc->common.pc);
  for (const std::vector<std::string_view>& digest :
       {std::vector<std::string_view>{"--cublas"}, std::vector<std::string_view>{"--ozaki"},
        std::vector<std::string_view>{"--fused-digest", "--blas-digest"}}) {
    if (digest.front() == "--cublas" && !kHaveCublasEmul) continue;
    std::vector<std::string_view> args{"--sc-hessian-kk-int"};
    args.insert(args.end(), digest.begin(), digest.end());
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value()) << digest.front();
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << digest.front();
    EXPECT_EQ(r.error().message, hessianDigestMessage("--sc-hessian-kk-int", digest.front()));
  }
}

// --sc-hessian-kk (#158), nevpt2_sc_pseudodensity alone: the orbital block's
// sandwich and the whole Eddot_kk, one job at a time, exact digests only
// (with --pc, PC's, #202); --gradgold rides on it.
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

TEST(CliScPseudodensityFlagTests, ScHessianKkTakesPcAndRefusesTheApproximateDigests) {
  // With --pc it is PC's sandwich and Eddot_kk (#202).
  const Result<Options> pc = parse(Binary::ScPseudodensity, {"--sc-hessian-kk", "--pc"});
  ASSERT_TRUE(pc.has_value()) << pc.error().message;
  EXPECT_TRUE(pc->scHessianKk);
  EXPECT_TRUE(pc->common.pc);
  for (const std::vector<std::string_view>& digest :
       {std::vector<std::string_view>{"--cublas"}, std::vector<std::string_view>{"--ozaki"},
        std::vector<std::string_view>{"--fused-digest", "--blas-digest"}}) {
    if (digest.front() == "--cublas" && !kHaveCublasEmul) continue;
    std::vector<std::string_view> args{"--sc-hessian-kk"};
    args.insert(args.end(), digest.begin(), digest.end());
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value()) << digest.front();
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << digest.front();
    EXPECT_EQ(r.error().message, hessianDigestMessage("--sc-hessian-kk", digest.front()));
  }
}

// --sc-source-tangent PATH (#220), nevpt2_sc_pseudodensity alone: the SC
// source terms' tangent along a `.pertgold`'s integral perturbations -- with
// --pc PC's (#234) -- one job at a time, exact digests only (refused the
// approximate ones as every Hessian flag is, with --pc too); --gradgold rides
// on it.
TEST(CliScPseudodensityFlagTests, ScSourceTangentSetsItsPathAndTakesGradgold) {
  const Result<Options> r = parse(Binary::ScPseudodensity, {"--sc-source-tangent", "p.pertgold"});
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->scSourceTangentPath, "p.pertgold");
  EXPECT_FALSE(r->scHessianKk);
  const Result<Options> g =
      parse(Binary::ScPseudodensity,
            {"--sc-source-tangent", "p.pertgold", "--gradgold", "s.gradgold", "--blas-digest"});
  ASSERT_TRUE(g.has_value()) << g.error().message;
  EXPECT_EQ(g->gradgoldPath, "s.gradgold");
  for (const std::vector<std::string_view>& digest :
       {std::vector<std::string_view>{"--digest-emitted"},
        std::vector<std::string_view>{"--consume-emitted"}}) {
    std::vector<std::string_view> args{"--sc-source-tangent", "p.pertgold"};
    args.insert(args.end(), digest.begin(), digest.end());
    const Result<Options> e = parse(Binary::ScPseudodensity, args);
    EXPECT_TRUE(e.has_value()) << digest.front() << ": " << e.error().message;
  }
}

TEST(CliScPseudodensityFlagTests, ScSourceTangentNeedsAPath) {
  const Result<Options> r = parse(Binary::ScPseudodensity, {"--sc-source-tangent"});
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
}

TEST(CliScPseudodensityFlagTests, ScSourceTangentRefusesTheOtherJobs) {
  for (const std::vector<std::string_view>& other :
       {std::vector<std::string_view>{"--kappa-act"}, std::vector<std::string_view>{"--kappa-full"},
        std::vector<std::string_view>{"--sc-hessian-kc"},
        std::vector<std::string_view>{"--sc-hessian-kk-int"},
        std::vector<std::string_view>{"--sc-hessian-kk"},
        std::vector<std::string_view>{"--dump-source-terms", "x.gradsrc"},
        std::vector<std::string_view>{"--source-oracle", "x.srcgold"}}) {
    std::vector<std::string_view> args{"--sc-source-tangent", "p.pertgold"};
    args.insert(args.end(), other.begin(), other.end());
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value()) << other.front();
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig) << other.front();
    // --sc-hessian-kk's and the others' own one-job refusals name themselves
    // first where they see the pair; either way it is the one-job rule.
    EXPECT_NE(r.error().message.find("run one at a time"), std::string::npos)
        << r.error().message;
    // With --pc (#234) the same one-job rule.
    args.push_back("--pc");
    const Result<Options> pc = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(pc.has_value()) << other.front();
    EXPECT_EQ(pc.error().kind, ErrorKind::InvalidConfig) << other.front();
    EXPECT_NE(pc.error().message.find("run one at a time"), std::string::npos)
        << pc.error().message;
  }
}

// --pc --sc-source-tangent (#234): PC's tangent, the same flag, its path and
// --gradgold kept, the exact digests taken.
TEST(CliScPseudodensityFlagTests, ScSourceTangentTakesPc) {
  const Result<Options> r = parse(
      Binary::ScPseudodensity,
      {"--sc-source-tangent", "p.pertgold", "--pc", "--gradgold", "s.gradgold", "--blas-digest"});
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_TRUE(r->common.pc);
  EXPECT_EQ(r->scSourceTangentPath, "p.pertgold");
  EXPECT_EQ(r->gradgoldPath, "s.gradgold");
}

TEST(CliScPseudodensityFlagTests, ScSourceTangentRefusesTheApproximateDigests) {
  for (const std::vector<std::string_view>& digest :
       {std::vector<std::string_view>{"--cublas"}, std::vector<std::string_view>{"--ozaki"},
        std::vector<std::string_view>{"--fused-digest", "--blas-digest"}}) {
    if (digest.front() == "--cublas" && !kHaveCublasEmul) continue;
    std::vector<std::string_view> args{"--sc-source-tangent", "p.pertgold"};
    args.insert(args.end(), digest.begin(), digest.end());
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value()) << digest.front();
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << digest.front();
    EXPECT_EQ(r.error().message, hessianDigestMessage("--sc-source-tangent", digest.front()));
    // With --pc (#234) the same refusal.
    args.push_back("--pc");
    const Result<Options> pc = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(pc.has_value()) << digest.front();
    EXPECT_EQ(pc.error().kind, ErrorKind::Unsupported) << digest.front();
    EXPECT_EQ(pc.error().message, hessianDigestMessage("--sc-source-tangent", digest.front()));
  }
}

TEST(CliScPseudodensityFlagTests, OtherBinariesRefuseScSourceTangent) {
  for (const Binary b : {Binary::Demo, Binary::DfDemo, Binary::PcTangent}) {
    const Result<Options> r = parse(b, {"--sc-source-tangent", "p.pertgold"});
    ASSERT_FALSE(r.has_value()) << binaryName(b);
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << binaryName(b);
    EXPECT_NE(r.error().message.find("only nevpt2_sc_pseudodensity"), std::string::npos)
        << r.error().message;
  }
}

// --sc-parameter-hessian PATH (#222): the parameter Hessian over a response
// sidecar's directions, with its sidecars beside it -- the gradient sidecar,
// the full tensor, the output and the coordinate block -- the exact digests
// taken.
TEST(CliScPseudodensityFlagTests, ScParameterHessianSetsItsPathsAndTakesItsSidecars) {
  const Result<Options> r =
      parse(Binary::ScPseudodensity, {"--sc-parameter-hessian", "r.respgold"});
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->scParameterHessianPath, "r.respgold");
  EXPECT_TRUE(r->parameterHessianOutPath.empty());
  EXPECT_TRUE(r->parameterHessianCoords.empty());
  const Result<Options> all = parse(
      Binary::ScPseudodensity,
      {"--sc-parameter-hessian", "r.respgold", "--gradgold", "s.gradgold", "--erigold",
       "x.erigold", "--parameter-hessian-out", "o.parhess", "--parameter-hessian-coords", "2,5",
       "--blas-digest"});
  ASSERT_TRUE(all.has_value()) << all.error().message;
  EXPECT_EQ(all->gradgoldPath, "s.gradgold");
  EXPECT_EQ(all->erigoldPath, "x.erigold");
  EXPECT_EQ(all->parameterHessianOutPath, "o.parhess");
  EXPECT_EQ(all->parameterHessianCoords, (std::vector<std::int64_t>{2, 5}));
  for (const std::vector<std::string_view>& digest :
       {std::vector<std::string_view>{"--digest-emitted"},
        std::vector<std::string_view>{"--consume-emitted"}}) {
    std::vector<std::string_view> args{"--sc-parameter-hessian", "r.respgold"};
    args.insert(args.end(), digest.begin(), digest.end());
    const Result<Options> e = parse(Binary::ScPseudodensity, args);
    EXPECT_TRUE(e.has_value()) << digest.front() << ": " << e.error().message;
  }
}

TEST(CliScPseudodensityFlagTests, ScParameterHessianCoordsAreAnIncreasingList) {
  const Result<Options> one = parse(Binary::ScPseudodensity, {"--sc-parameter-hessian", "r",
                                                              "--parameter-hessian-coords", "0"});
  ASSERT_TRUE(one.has_value()) << one.error().message;
  EXPECT_EQ(one->parameterHessianCoords, (std::vector<std::int64_t>{0}));
  for (const std::string_view bad : {"", "1,", ",1", "2,1", "1,1", "-1", "a", "1;2", "1, 2"}) {
    const Result<Options> r = parse(
        Binary::ScPseudodensity, {"--sc-parameter-hessian", "r", "--parameter-hessian-coords", bad});
    ASSERT_FALSE(r.has_value()) << "\"" << bad << "\"";
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig) << "\"" << bad << "\"";
    EXPECT_NE(r.error().message.find("--parameter-hessian-coords"), std::string::npos)
        << r.error().message;
  }
}

TEST(CliScPseudodensityFlagTests, ScParameterHessianOutAndCoordsRideOnIt) {
  const Result<Options> out = parse(Binary::ScPseudodensity, {"--parameter-hessian-out", "o"});
  ASSERT_FALSE(out.has_value());
  EXPECT_EQ(out.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(out.error().message,
            "--parameter-hessian-out is where --sc-parameter-hessian writes its matrix; it does "
            "nothing on its own");
  const Result<Options> coords =
      parse(Binary::ScPseudodensity, {"--sc-hessian-kk", "--parameter-hessian-coords", "1"});
  ASSERT_FALSE(coords.has_value());
  EXPECT_EQ(coords.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(coords.error().message,
            "--parameter-hessian-coords restricts --sc-parameter-hessian's direction set; it "
            "does nothing on its own");
  for (const std::string_view flag : {"--parameter-hessian-out", "--parameter-hessian-coords"})
    for (const Binary b : {Binary::Demo, Binary::DfDemo, Binary::PcTangent}) {
      const Result<Options> r = parse(b, {flag, "1"});
      ASSERT_FALSE(r.has_value()) << binaryName(b) << " " << flag;
      EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << binaryName(b) << " " << flag;
    }
}

TEST(CliScPseudodensityFlagTests, ScParameterHessianRefusesTheOtherJobsAndPc) {
  for (const std::vector<std::string_view>& other :
       {std::vector<std::string_view>{"--kappa-act"}, std::vector<std::string_view>{"--kappa-full"},
        std::vector<std::string_view>{"--sc-hessian-kc"},
        std::vector<std::string_view>{"--sc-hessian-kk-int"},
        std::vector<std::string_view>{"--sc-hessian-kk"},
        std::vector<std::string_view>{"--sc-source-tangent", "p.pertgold"},
        std::vector<std::string_view>{"--dump-source-terms", "x.gradsrc"},
        std::vector<std::string_view>{"--source-oracle", "x.srcgold"}}) {
    std::vector<std::string_view> args{"--sc-parameter-hessian", "r.respgold"};
    args.insert(args.end(), other.begin(), other.end());
    const Result<Options> r = parse(Binary::ScPseudodensity, args);
    ASSERT_FALSE(r.has_value()) << other.front();
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig) << other.front();
    EXPECT_NE(r.error().message.find("run one at a time"), std::string::npos)
        << r.error().message;
  }
  // SC only: PC's bilinear form is not built.
  const Result<Options> pc =
      parse(Binary::ScPseudodensity, {"--sc-parameter-hessian", "r.respgold", "--pc"});
  ASSERT_FALSE(pc.has_value());
  EXPECT_EQ(pc.error().kind, ErrorKind::Unsupported);
  EXPECT_EQ(pc.error().message,
            "--sc-parameter-hessian --pc: the parameter Hessian over the responses is SC's "
            "here; PC's bilinear form is not built");
}

TEST(CliScPseudodensityFlagTests, ScParameterHessianRefusesTheApproximateDigests) {
  for (const std::vector<std::string_view>& digest :
       {std::vector<std::string_view>{"--cublas"}, std::vector<std::string_view>{"--ozaki"},
        std::vector<std::string_view>{"--fused-digest", "--blas-digest"}}) {
    if (digest.front() == "--cublas" && !kHaveCublasEmul) continue;
    for (const bool flagFirst : {true, false}) {
      std::vector<std::string_view> args;
      if (flagFirst) args.insert(args.end(), {"--sc-parameter-hessian", "r.respgold"});
      args.insert(args.end(), digest.begin(), digest.end());
      if (!flagFirst) args.insert(args.end(), {"--sc-parameter-hessian", "r.respgold"});
      const Result<Options> r = parse(Binary::ScPseudodensity, args);
      ASSERT_FALSE(r.has_value()) << digest.front();
      EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << digest.front();
      EXPECT_EQ(r.error().message, hessianDigestMessage("--sc-parameter-hessian", digest.front()));
    }
  }
}

TEST(CliScPseudodensityFlagTests, OtherBinariesRefuseScParameterHessian) {
  for (const Binary b : {Binary::Demo, Binary::DfDemo, Binary::PcTangent}) {
    const Result<Options> r = parse(b, {"--sc-parameter-hessian", "r.respgold"});
    ASSERT_FALSE(r.has_value()) << binaryName(b);
    EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << binaryName(b);
    EXPECT_NE(r.error().message.find("only nevpt2_sc_pseudodensity"), std::string::npos)
        << r.error().message;
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

// #214: --gradgold selects the FULL mode of either PC CI Hessian check, and
// is refused without one (the tangent check reads no sidecar).
TEST(CliPcTangentFlagTests, GradgoldRidesOnThePcHessianFlags) {
  for (const std::string_view which : {"--pc-hessian", "--pc-hessian-vector"}) {
    const Result<Options> r = parse(Binary::PcTangent, {which, "--gradgold", "x.gradgold"});
    ASSERT_TRUE(r.has_value()) << which << ": " << r.error().message;
    EXPECT_EQ(r->gradgoldPath, "x.gradgold");
  }
  const Result<Options> r = parse(Binary::PcTangent, {"--gradgold", "x.gradgold"});
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
  EXPECT_EQ(r.error().message,
            "--gradgold is the sidecar --pc-hessian and --pc-hessian-vector read the orbital "
            "energies' response from; it does nothing on its own");
}

TEST(CliPcTangentFlagTests, ABadValueIsAConfigErrorNamingTheFlag) {
  for (const std::string_view flag : {"--fd-steps", "--fd-start-pow"}) {
    const Result<Options> r = parse(Binary::PcTangent, {flag, "4.5"});
    ASSERT_FALSE(r.has_value()) << flag;
    EXPECT_EQ(r.error().kind, ErrorKind::InvalidConfig);
    EXPECT_EQ(r.error().message, std::format("{} takes an integer, not \"4.5\"", flag));
  }
}

// --- CliHessianRefusalTests ---------------------------------------------------
//
// docs/hessian-theory.md, Sec. 9.1's flag-parsing rows, across every binary
// and every Hessian flag at once (#157). The per-flag suites above hold each
// flag's own combinations; these hold the matrix.

// Every Hessian flag in the tree, and the one binary that implements it.
struct HessianFlag {
  std::string_view flag;
  Binary binary;
};
constexpr std::array kHessianFlags{
    HessianFlag{"--sc-hessian", Binary::Demo},
    HessianFlag{"--sc-hessian-vector", Binary::Demo},
    HessianFlag{"--sc-hessian-kc", Binary::ScPseudodensity},
    HessianFlag{"--sc-hessian-kk-int", Binary::ScPseudodensity},
    HessianFlag{"--sc-hessian-kk", Binary::ScPseudodensity},
    HessianFlag{"--pc-hessian", Binary::PcTangent},
    HessianFlag{"--pc-hessian-vector", Binary::PcTangent},
};

TEST(CliHessianRefusalTests, EachHessianFlagIsOneBinarysAlone) {
  for (const HessianFlag& h : kHessianFlags)
    for (const Binary b : kBinaries)
      EXPECT_EQ(implements(b, h.flag), b == h.binary) << binaryName(b) << " " << h.flag;
}

// A Hessian flag on a binary that lacks it: Unsupported, naming the one that
// has it -- never parsed and ignored.
TEST(CliHessianRefusalTests, ABinaryThatLacksAHessianFlagRefusesItNamingTheOneThatHasIt) {
  for (const HessianFlag& h : kHessianFlags) {
    for (const Binary b : kBinaries) {
      if (b == h.binary) continue;
      const Result<Options> r = parse(b, {h.flag});
      ASSERT_FALSE(r.has_value()) << binaryName(b) << " accepted " << h.flag;
      EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << binaryName(b) << " " << h.flag;
      EXPECT_TRUE(r.error().message.starts_with(std::format(
          "{} does not implement {} (only {})", binaryName(b), h.flag, binaryName(h.binary))))
          << r.error().message;
    }
  }
}

// Every approximate digest, alone and with the flags that ride on it, beside
// every Hessian flag on its own binary: Unsupported, whatever finalize() would
// have said of the digest flags (--fused-digest alone has no GEMM digest to
// fuse; --cublas is refused outright on HIP) and whichever order they come in.
// nevpt2_pc_tangent implements none of these, so it refuses the digest flag
// itself -- Unsupported too, naming the binaries that take it.
TEST(CliHessianRefusalTests, EveryHessianFlagRefusesEveryApproximateDigestAsUnsupported) {
  struct Digest {
    std::vector<std::string_view> args;
    std::string_view named;  // the digest the message names
  };
  const std::vector<Digest> digests{
      {{"--cublas"}, "--cublas"},
      {{"--cublas", "--mantissa-bits", "40"}, "--cublas"},
      {{"--cublas", "--fused-digest"}, "--cublas"},
      {{"--ozaki"}, "--ozaki"},
      {{"--ozaki", "--ozaki-pairs", "5"}, "--ozaki"},
      {{"--ozaki", "--fused-digest"}, "--ozaki"},
      {{"--fused-digest"}, "--fused-digest"},
      {{"--fused-digest", "--blas-digest"}, "--fused-digest"},
  };
  for (const HessianFlag& h : kHessianFlags) {
    for (const Digest& d : digests) {
      for (const bool flagFirst : {true, false}) {
        std::vector<std::string_view> args;
        if (flagFirst) args.push_back(h.flag);
        args.insert(args.end(), d.args.begin(), d.args.end());
        if (!flagFirst) args.push_back(h.flag);
        const Result<Options> r = parse(h.binary, args);
        ASSERT_FALSE(r.has_value()) << h.flag << " " << d.args.front();
        EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << h.flag << " " << d.args.front();
        if (h.binary == Binary::PcTangent)
          EXPECT_TRUE(r.error().message.starts_with(
              std::format("nevpt2_pc_tangent does not implement {}", d.args.front())))
              << r.error().message;
        else
          EXPECT_EQ(r.error().message, hessianDigestMessage(h.flag, d.named));
      }
    }
  }
}

// And the exact digests are every Hessian flag's to run.
TEST(CliHessianRefusalTests, EveryHessianFlagTakesTheExactDigests) {
  for (const HessianFlag& h : kHessianFlags) {
    for (const std::vector<std::string_view>& digest :
         {std::vector<std::string_view>{}, std::vector<std::string_view>{"--blas-digest"},
          std::vector<std::string_view>{"--digest-emitted"},
          std::vector<std::string_view>{"--consume-emitted"}}) {
      std::vector<std::string_view> args{h.flag};
      args.insert(args.end(), digest.begin(), digest.end());
      const Result<Options> r = parse(h.binary, args);
      EXPECT_TRUE(r.has_value()) << h.flag << ": " << r.error().message;
    }
  }
}

// --- the PC orbital Hessian flags (#200-#203) ------------------------------------
//
// --pc with an orbital Hessian flag is PC-NEVPT2's block (#200, #201, #202),
// on nevpt2_sc_pseudodensity alone; #157's matrix carries over to it whole.
struct PcOrbitalHessianFlag {
  std::string_view flag;
  bool Options::* field;
};
const std::array kPcOrbitalHessianFlags{
    PcOrbitalHessianFlag{"--sc-hessian-kc", &Options::scHessianKc},
    PcOrbitalHessianFlag{"--sc-hessian-kk-int", &Options::scHessianKkInt},
    PcOrbitalHessianFlag{"--sc-hessian-kk", &Options::scHessianKk},
};

// Each one takes --pc, in either order, with every exact digest, and sets
// both fields.
TEST(CliHessianRefusalTests, EveryPcOrbitalHessianFlagTakesPcAndTheExactDigests) {
  for (const PcOrbitalHessianFlag& h : kPcOrbitalHessianFlags) {
    for (const std::vector<std::string_view>& digest :
         {std::vector<std::string_view>{}, std::vector<std::string_view>{"--blas-digest"},
          std::vector<std::string_view>{"--digest-emitted"},
          std::vector<std::string_view>{"--consume-emitted"}}) {
      for (const bool pcFirst : {true, false}) {
        std::vector<std::string_view> args;
        if (pcFirst) args.push_back("--pc");
        args.push_back(h.flag);
        if (!pcFirst) args.push_back("--pc");
        args.insert(args.end(), digest.begin(), digest.end());
        const Result<Options> r = parse(Binary::ScPseudodensity, args);
        ASSERT_TRUE(r.has_value()) << h.flag << ": " << r.error().message;
        EXPECT_TRUE(r->common.pc) << h.flag;
        EXPECT_TRUE((*r).*(h.field)) << h.flag;
      }
    }
  }
}

// With --pc, every approximate digest is still Unsupported, with #157's
// message, whichever order the three flags come in.
TEST(CliHessianRefusalTests, EveryPcOrbitalHessianFlagRefusesEveryApproximateDigestAsUnsupported) {
  const std::vector<std::vector<std::string_view>> digests{
      {"--cublas"},       {"--cublas", "--mantissa-bits", "40"},
      {"--ozaki"},        {"--ozaki", "--ozaki-pairs", "5"},
      {"--fused-digest"}, {"--fused-digest", "--blas-digest"},
  };
  for (const PcOrbitalHessianFlag& h : kPcOrbitalHessianFlags) {
    for (const std::vector<std::string_view>& d : digests) {
      for (const int order : {0, 1, 2}) {
        std::vector<std::string_view> args;
        if (order == 0) args = {"--pc", h.flag};
        if (order == 1) args = {h.flag, "--pc"};
        args.insert(args.end(), d.begin(), d.end());
        if (order == 2) {
          args.push_back(h.flag);
          args.push_back("--pc");
        }
        const Result<Options> r = parse(Binary::ScPseudodensity, args);
        ASSERT_FALSE(r.has_value()) << h.flag << " " << d.front();
        EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << h.flag << " " << d.front();
        EXPECT_EQ(r.error().message, hessianDigestMessage(h.flag, d.front()));
      }
    }
  }
}

// On a binary that lacks the flag, --pc does not change the refusal: still
// Unsupported, naming nevpt2_sc_pseudodensity. The demos take --pc, so it
// may come first there; nevpt2_pc_tangent refuses --pc itself, so the flag
// comes first on every binary.
TEST(CliHessianRefusalTests, ABinaryThatLacksAPcOrbitalHessianFlagRefusesItWithPc) {
  for (const PcOrbitalHessianFlag& h : kPcOrbitalHessianFlags) {
    for (const Binary b : kBinaries) {
      if (b == Binary::ScPseudodensity) continue;
      std::vector<std::vector<std::string_view>> orders{{h.flag, "--pc"}};
      if (b != Binary::PcTangent) orders.push_back({"--pc", h.flag});
      for (const std::vector<std::string_view>& args : orders) {
        const Result<Options> r = parse(b, args);
        ASSERT_FALSE(r.has_value()) << binaryName(b) << " accepted " << h.flag;
        EXPECT_EQ(r.error().kind, ErrorKind::Unsupported) << binaryName(b) << " " << h.flag;
        EXPECT_TRUE(r.error().message.starts_with(std::format(
            "{} does not implement {} (only nevpt2_sc_pseudodensity)", binaryName(b), h.flag)))
            << r.error().message;
      }
    }
  }
}

}  // namespace nevpt2::test::cli
