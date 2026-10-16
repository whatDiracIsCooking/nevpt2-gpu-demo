// Suites for nevpt2.golden, the .nevpt2gold reader.
//
// Host-only ON PURPOSE: loadGolden is file parsing, so this binary is not
// REQUIRES_GPU and runs with no card visible. It checks the loader's logic --
// a refusal's kind, a parsed header, an array's shape -- never an energy: the
// golden tier owns every numerical claim, and the energies read below are only
// compared with the values the file was written with.
//
//   GoldenLoadErrorTests  a missing path, a truncated file, a bad magic,
//                         trailing bytes and a ci that is not ndet long each
//                         come back as an IO Error -- never an abort
//   GoldenRequireTests    require() names the first array the file lacks;
//                         checkNdet() refuses a header ndet it does not match
//   GoldenReferenceKindTests
//                         the header's reference_kind field: a "NEVPT2G2"
//                         file round-trips both kinds, the pre-kind
//                         "NEVPT2G1" layout reads as CASCI, an unknown code
//                         is an IO Error
//   GoldenReferenceKindGateTests
//                         the two reference-kind gates split: the
//                         source-term gate (--sc-tangent, --gradient-source,
//                         --dump-source-terms, --source-oracle) accepts a
//                         CASSCF golden, the Lagrangian gate refuses one as
//                         Unsupported
//   GoldenCommittedCasscfTests
//                         the committed golden/n2_ccpvdz_cas44_casscf.nevpt2gold:
//                         a real NEVPT2G2 file whose reference_kind is CASSCF,
//                         so the split above is checked against a committed
//                         file and not only a spliced one
//   GoldenCas44Tests      the committed golden/n2_ccpvdz_cas44.nevpt2gold:
//                         header fields and array shapes
//   GoldenLayoutContractTests
//                         the layout contract's C++ half: every golden and
//                         gradient sidecar in golden/, of both header
//                         layouts, loads, with its array table found where
//                         that file's header ends; every source-term oracle
//                         (*.srcgold, #92), of all four kinds (SC "NEVPT2S1",
//                         PC "NEVPT2P1" as <case>_pc.srcgold, #95, and their
//                         directional flavours "NEVPT2SD" / "NEVPT2PD" as
//                         <case>[_pc]_dir.srcgold, #105), loads and belongs
//                         to its golden (and to its gradient sidecar, where
//                         that is on disk: acrolein's is gitignored)
//   SrcGoldLoadErrorTests loadSrcGold's refusals: a golden handed to it, a
//                         truncated or over-long oracle, and an oracle held
//                         to a golden it was not written for; the magic is
//                         what sets the scheme and the flavour, a directional
//                         oracle with a direction that is not antisymmetric is
//                         refused, and srcGoldCase refuses an oracle not
//                         named for its kind
//
// The malformed files are written to a per-test temp directory, removed in
// TearDown. The committed goldens are only ever read; their paths come from
// CMake (NEVPT2_TEST_GOLDEN_CAS44, NEVPT2_TEST_GOLDEN_CAS44_CASSCF,
// NEVPT2_TEST_COMMITTED_GOLDENS, NEVPT2_TEST_COMMITTED_GRADGOLDS,
// NEVPT2_TEST_COMMITTED_SRCGOLDS), not from
// this source.
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.golden;

#ifndef NEVPT2_TEST_GOLDEN_CAS44
#error "NEVPT2_TEST_GOLDEN_CAS44 must be defined by test/golden/CMakeLists.txt"
#endif
#ifndef NEVPT2_TEST_GOLDEN_CAS44_CASSCF
#error "NEVPT2_TEST_GOLDEN_CAS44_CASSCF must be defined by test/golden/CMakeLists.txt"
#endif
#if !defined(NEVPT2_TEST_COMMITTED_GOLDENS) || !defined(NEVPT2_TEST_COMMITTED_GRADGOLDS) || \
    !defined(NEVPT2_TEST_COMMITTED_SRCGOLDS)
#error "NEVPT2_TEST_COMMITTED_GOLDENS/_GRADGOLDS/_SRCGOLDS must be defined by test/golden/CMakeLists.txt"
#endif

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
// One binary per suite file, so a name clash here is a link error.
namespace nevpt2::test::golden {

constexpr std::string_view kGoldenCas44 = NEVPT2_TEST_GOLDEN_CAS44;
constexpr std::string_view kGoldenCas44Casscf = NEVPT2_TEST_GOLDEN_CAS44_CASSCF;

std::string readAll(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// A fixture owning one fresh temp directory per test, removed afterwards.
class GoldenTempDir : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::random_device rd;
    dir_ = std::filesystem::temp_directory_path() /
           std::format("nevpt2_golden_tests_{}_{}_{:08x}", info->test_suite_name(), info->name(),
                       rd());
    std::filesystem::create_directories(dir_);
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }

  // Writes `bytes` to `<dir>/<name>` and returns its path.
  std::string write(const std::string& name, std::string_view bytes) const {
    const std::filesystem::path p = dir_ / name;
    std::ofstream out(p, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    EXPECT_TRUE(out.good()) << "could not write " << p;
    return p.string();
  }

  std::filesystem::path dir_;
};

void expectIoError(const Result<GoldenFile>& r, std::string_view fragment) {
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::IO);
  EXPECT_NE(r.error().message.find(fragment), std::string::npos) << r.error().message;
}

// --- GoldenLoadErrorTests -----------------------------------------------------

using GoldenLoadErrorTests = GoldenTempDir;

TEST_F(GoldenLoadErrorTests, MissingPathIsAnIoError) {
  const std::string path = (dir_ / "does_not_exist.nevpt2gold").string();
  const Result<GoldenFile> r = loadGolden(path);
  expectIoError(r, "could not open golden file");
  if (!r.has_value()) {
    EXPECT_NE(r.error().message.find(path), std::string::npos);
  }
}

TEST_F(GoldenLoadErrorTests, EmptyFileIsAnIoError) {
  // Too short even for the 8-byte magic, refused before any read.
  expectIoError(loadGolden(write("empty.nevpt2gold", "")), "bad string length 8");
}

TEST_F(GoldenLoadErrorTests, BadMagicIsAnIoError) {
  // The real file with its magic overwritten: everything after it is intact.
  std::string bytes = readAll(std::string(kGoldenCas44));
  ASSERT_GT(bytes.size(), 8u);
  bytes.replace(0, 8, "NOTGOLD!");
  const Result<GoldenFile> r = loadGolden(write("bad_magic.nevpt2gold", bytes));
  expectIoError(r, "bad magic");
  if (!r.has_value()) {
    EXPECT_NE(r.error().message.find("NOTGOLD!"), std::string::npos);
  }
}

TEST_F(GoldenLoadErrorTests, TruncatedHeaderIsAnIoError) {
  // The magic and part of the scalar header only.
  const std::string bytes = readAll(std::string(kGoldenCas44));
  ASSERT_GT(bytes.size(), 20u);
  expectIoError(loadGolden(write("trunc_header.nevpt2gold", bytes.substr(0, 20))),
                "unexpected EOF");
}

TEST_F(GoldenLoadErrorTests, TruncatedArrayDataIsAnIoError) {
  // The real file minus its last byte: the final array's data runs short.
  const std::string bytes = readAll(std::string(kGoldenCas44));
  ASSERT_GT(bytes.size(), 1u);
  expectIoError(loadGolden(write("trunc_tail.nevpt2gold", bytes.substr(0, bytes.size() - 1))),
                "golden file");
}

TEST_F(GoldenLoadErrorTests, TruncatedMidFileIsAnIoError) {
  // Cut halfway: an array's extents now promise more data than the file holds.
  const std::string bytes = readAll(std::string(kGoldenCas44));
  ASSERT_GT(bytes.size(), 2u);
  expectIoError(loadGolden(write("trunc_mid.nevpt2gold", bytes.substr(0, bytes.size() / 2))),
                "golden file");
}

TEST_F(GoldenLoadErrorTests, TrailingBytesAreAnIoError) {
  // The real file plus one byte past its last array.
  const std::string bytes = readAll(std::string(kGoldenCas44)) + '\0';
  expectIoError(loadGolden(write("trailing.nevpt2gold", bytes)), "1 bytes after the last");
}

TEST_F(GoldenLoadErrorTests, NdetNotTheCiLengthIsAnIoError) {
  // The header's ndet bumped by one: ci still holds the real count. Found by
  // its value inside the scalar header (which ends at arraysAt) rather than
  // at a position written out here; the other header fields of this file
  // (4, 2, 2, 5) cannot hold the int64 36's byte pattern, which the
  // uniqueness check below confirms.
  const Result<GoldenFile> g = loadGolden(std::string(kGoldenCas44));
  ASSERT_TRUE(g.has_value()) << g.error().message;
  std::string bytes = readAll(std::string(kGoldenCas44));
  std::int64_t ndet = g->ndet;
  const std::string pattern(reinterpret_cast<const char*>(&ndet), 8);
  const std::string header = bytes.substr(0, static_cast<std::size_t>(g->arraysAt));
  const std::size_t at = header.find(pattern);
  ASSERT_NE(at, std::string::npos);
  ASSERT_EQ(header.find(pattern, at + 1), std::string::npos) << "ndet's bytes are not unique";
  ++ndet;
  std::memcpy(bytes.data() + at, &ndet, 8);
  expectIoError(loadGolden(write("bad_ndet.nevpt2gold", bytes)),
                std::format("but the header says ndet = {}", ndet));
}

// --- GoldenRequireTests -------------------------------------------------------

TEST(GoldenRequireTests, PresentArraysAreOk) {
  GoldenFile g;
  g.arrays.emplace("dm1", Tensor({2, 2}));
  g.arrays.emplace("h1e", Tensor({2, 2}));
  const Status s = g.require({"dm1", "h1e"});
  EXPECT_TRUE(s.has_value());
}

TEST(GoldenRequireTests, MissingArrayIsAnIoErrorNamingIt) {
  GoldenFile g;
  g.arrays.emplace("dm1", Tensor({2, 2}));
  const Status s = g.require({"dm1", "dm2", "h1e"});
  ASSERT_FALSE(s.has_value());
  EXPECT_EQ(s.error().kind, ErrorKind::IO);
  // The FIRST missing name, not a later one.
  EXPECT_NE(s.error().message.find("'dm2'"), std::string::npos) << s.error().message;
  EXPECT_EQ(s.error().message.find("h1e"), std::string::npos) << s.error().message;
}

TEST(GoldenRequireTests, MissingArrayInTheCommittedGolden) {
  Result<GoldenFile> r = loadGolden(std::string(kGoldenCas44));
  ASSERT_TRUE(r.has_value()) << r.error().message;
  const Status s = r->require({"dm1", "no_such_array"});
  ASSERT_FALSE(s.has_value());
  EXPECT_EQ(s.error().kind, ErrorKind::IO);
  EXPECT_NE(s.error().message.find("no_such_array"), std::string::npos) << s.error().message;
}

TEST(GoldenRequireTests, CheckNdetRefusesAMismatch) {
  GoldenFile g;
  g.ncas = 4;
  g.nelecA = 2;
  g.nelecB = 2;
  g.ndet = 36;
  EXPECT_TRUE(g.checkNdet(36).has_value());
  const Status s = g.checkNdet(35);
  ASSERT_FALSE(s.has_value());
  EXPECT_EQ(s.error().kind, ErrorKind::IO);
  EXPECT_NE(s.error().message.find("header ndet = 36"), std::string::npos) << s.error().message;
}

// --- GoldenReferenceKindTests -------------------------------------------------

// The committed golden rewritten in the "NEVPT2G2" layout: the magic bumped
// and a four-byte reference_kind spliced in at the end of the scalar header,
// every array byte left exactly where it was. That is the whole difference
// between the two layouts, and the transformation regenerating a committed
// golden performs today (docs/reference-data.md, "Regenerating the golden
// file").
//
// The field goes where the G1 file's scalar header ends, which is where
// loadGolden found its array table (GoldenFile::arraysAt) -- read off the
// file, not a header length written out here, so this splice cannot drift
// from the reader it tests. Empty if the committed file does not load.
std::string cas44AsG2(const std::int32_t kind) {
  const Result<GoldenFile> g1 = loadGolden(std::string(kGoldenCas44));
  EXPECT_TRUE(g1.has_value()) << g1.error().message;
  if (!g1.has_value()) return {};
  std::string out = readAll(std::string(kGoldenCas44));
  out.replace(0, 8, "NEVPT2G2");
  std::string field(4, '\0');
  std::memcpy(field.data(), &kind, 4);
  out.insert(static_cast<std::size_t>(g1->arraysAt), field);
  return out;
}

using GoldenReferenceKindTests = GoldenTempDir;

TEST_F(GoldenReferenceKindTests, TheCommittedGoldenIsAPreKindFileAndReadsAsCasci) {
  // Every committed CASCI golden is still the pre-kind layout, which is why
  // bumping the magic did not have to rewrite one; only the CASSCF-orbital
  // files, written after the field existed, are G2
  // (GoldenCommittedCasscfTests below).
  const std::string bytes = readAll(std::string(kGoldenCas44));
  ASSERT_GE(bytes.size(), 8u);
  EXPECT_EQ(bytes.substr(0, 8), "NEVPT2G1");

  Result<GoldenFile> r = loadGolden(std::string(kGoldenCas44));
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->reference, ReferenceKind::Casci);
  EXPECT_EQ(referenceKindName(r->reference), "CASCI");
}

TEST_F(GoldenReferenceKindTests, G2RoundTripsBothKinds) {
  const Result<GoldenFile> reference = loadGolden(std::string(kGoldenCas44));
  ASSERT_TRUE(reference.has_value()) << reference.error().message;

  const std::vector<std::pair<std::int32_t, ReferenceKind>> kinds = {
      {0, ReferenceKind::Casci}, {1, ReferenceKind::Casscf}};
  for (const auto& [code, kind] : kinds) {
    const std::string name = std::format("g2_kind{}.nevpt2gold", code);
    Result<GoldenFile> r = loadGolden(write(name, cas44AsG2(code)));
    ASSERT_TRUE(r.has_value()) << r.error().message;
    EXPECT_EQ(r->reference, kind) << code;
    // The four inserted bytes are the ONLY difference: the array table
    // starts that much later, and the rest of the header, and every array,
    // parse exactly as the G1 file's did.
    EXPECT_EQ(r->arraysAt, reference->arraysAt + 4);
    EXPECT_EQ(r->ncas, reference->ncas);
    EXPECT_EQ(r->nelecA, reference->nelecA);
    EXPECT_EQ(r->nelecB, reference->nelecB);
    EXPECT_EQ(r->ncore, reference->ncore);
    EXPECT_EQ(r->ndet, reference->ndet);
    EXPECT_EQ(r->eCasci, reference->eCasci);
    EXPECT_EQ(r->eNevpt2Total, reference->eNevpt2Total);
    ASSERT_EQ(std::ssize(r->arrays), std::ssize(reference->arrays));
    for (const auto& [arrayName, t] : reference->arrays) {
      ASSERT_TRUE(r->require({arrayName}).has_value()) << arrayName;
      const Tensor& got = r->get(arrayName);
      EXPECT_EQ(got.dims(), t.dims()) << arrayName;
      ASSERT_EQ(got.size(), t.size()) << arrayName;
      for (int64_t k = 0; k < t.size(); ++k) ASSERT_EQ(got.flat(k), t.flat(k)) << arrayName << k;
    }
  }
  EXPECT_EQ(referenceKindName(ReferenceKind::Casscf), "CASSCF");
}

TEST_F(GoldenReferenceKindTests, AnUnknownReferenceKindIsAnIoError) {
  // A well-formed G2 file in every other respect, so what is refused is the
  // kind code and nothing else.
  for (const std::int32_t code : {-1, 2, 7}) {
    const Result<GoldenFile> r =
        loadGolden(write(std::format("bad_kind{}.nevpt2gold", code), cas44AsG2(code)));
    expectIoError(r, std::format("reference_kind = {} is not a kind this reader knows", code));
  }
}

TEST_F(GoldenReferenceKindTests, BadMagicNamesBothAcceptedMagics) {
  std::string bytes = readAll(std::string(kGoldenCas44));
  ASSERT_GT(bytes.size(), 8u);
  bytes.replace(0, 8, "NEVPT2G9");
  const Result<GoldenFile> r = loadGolden(write("bad_magic_g9.nevpt2gold", bytes));
  expectIoError(r, "bad magic");
  if (!r.has_value()) {
    EXPECT_NE(r.error().message.find("NEVPT2G2"), std::string::npos) << r.error().message;
    EXPECT_NE(r.error().message.find("NEVPT2G1"), std::string::npos) << r.error().message;
  }
}

// The flags behind each gate. The source-term ones are the four real call
// sites (apps/integral_direct, apps/sc_pseudodensity); nothing in this tree
// builds a full gradient yet, so the Lagrangian names are what will ask.
constexpr std::array<std::string_view, 4> kSourceTermFlags = {
    "--sc-tangent", "--gradient-source", "--dump-source-terms", "--source-oracle"};
constexpr std::array<std::string_view, 2> kLagrangianConsumers = {"the numerical gradient",
                                                                   "the analytic gradient"};

// A CASSCF golden through both gates: every source-term flag accepted, every
// Lagrangian consumer refused as Unsupported with a message that names it,
// the kind, and the derivation that does not cover it.
void expectTheGatesSplit(const GoldenFile& g) {
  ASSERT_EQ(g.reference, ReferenceKind::Casscf);
  for (const std::string_view flag : kSourceTermFlags) {
    const Status s = g.requireSourceTermReference(flag);
    EXPECT_TRUE(s.has_value()) << flag << ": " << s.error().message;
  }
  for (const std::string_view who : kLagrangianConsumers) {
    const Status s = g.requireLagrangianReference(who);
    ASSERT_FALSE(s.has_value()) << who;
    // Unsupported, not IO: a CASSCF-orbital golden is a well-formed file
    // carrying a reference this code's Lagrangian does not cover.
    EXPECT_EQ(s.error().kind, ErrorKind::Unsupported) << who;
    EXPECT_NE(s.error().message.find(who), std::string::npos) << s.error().message;
    EXPECT_NE(s.error().message.find("CASSCF"), std::string::npos) << s.error().message;
    // It says which half refuses, and why, citing the theory.
    EXPECT_NE(s.error().message.find("Lagrangian"), std::string::npos) << s.error().message;
    EXPECT_NE(s.error().message.find("docs/gradient-theory.md"), std::string::npos)
        << s.error().message;
    EXPECT_NE(s.error().message.find("CP-CASSCF"), std::string::npos) << s.error().message;
  }
}

TEST(GoldenReferenceKindGateTests, ACasciReferencePassesBothGates) {
  GoldenFile g;  // the default, which is what a pre-kind file loads as
  EXPECT_EQ(g.reference, ReferenceKind::Casci);
  for (const std::string_view flag : kSourceTermFlags)
    EXPECT_TRUE(g.requireSourceTermReference(flag).has_value()) << flag;
  for (const std::string_view who : kLagrangianConsumers)
    EXPECT_TRUE(g.requireLagrangianReference(who).has_value()) << who;
}

TEST(GoldenReferenceKindGateTests, ACasscfReferenceSplitsTheGates) {
  GoldenFile g;
  g.reference = ReferenceKind::Casscf;
  expectTheGatesSplit(g);
}

TEST_F(GoldenReferenceKindTests, TheGatesSplitOnALoadedCasscfGolden) {
  // End to end: a G2 file whose kind is CASSCF loads fine, and then the gates
  // split on it, which is the order the demos do it in -- all host-side, with
  // no card.
  Result<GoldenFile> r = loadGolden(write("casscf.nevpt2gold", cas44AsG2(1)));
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(referenceKindName(r->reference), "CASSCF");
  expectTheGatesSplit(*r);
}

// --- GoldenCommittedCasscfTests -----------------------------------------------
//
// The suite above spliced its G2 files together byte by byte, because when the
// field was added no committed golden carried it. One does now:
// golden/n2_ccpvdz_cas44_casscf.nevpt2gold, written by `generate_golden.py
// --mo-from` on committed CASSCF orbitals (docs/reference-data.md, "The MO
// sidecar: committed CASSCF orbitals"). These three tests are the same
// contract read off that real file -- the layout, the kind, and the two
// gates' split on it -- so the split is checked against a reference that
// actually exists. Host-only like the rest of this binary: nothing here
// computes or compares an energy, only whether two files are the same
// reference.
TEST(GoldenCommittedCasscfTests, TheCommittedCasscfGoldenIsAG2CasscfFile) {
  const std::string bytes = readAll(std::string(kGoldenCas44Casscf));
  ASSERT_GE(bytes.size(), 8u);
  EXPECT_EQ(bytes.substr(0, 8), "NEVPT2G2");

  const Result<GoldenFile> r = loadGolden(std::string(kGoldenCas44Casscf));
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->reference, ReferenceKind::Casscf);
  EXPECT_EQ(referenceKindName(r->reference), "CASSCF");
  // The same case as the CASCI twin beside it: the orbitals differ, the
  // active space does not.
  EXPECT_EQ(r->ncas, 4);
  EXPECT_EQ(r->nelecA, 2);
  EXPECT_EQ(r->nelecB, 2);
  EXPECT_EQ(r->ncore, 5);
  EXPECT_EQ(r->ndet, 36);
  // It carries the PC fields its --pc ctest entry needs.
  EXPECT_TRUE(r->require({"pc_class_energies", "e_pc_total"}).has_value());
}

TEST(GoldenCommittedCasscfTests, ItIsADifferentReferenceFromItsCasciTwin) {
  const Result<GoldenFile> casscf = loadGolden(std::string(kGoldenCas44Casscf));
  ASSERT_TRUE(casscf.has_value()) << casscf.error().message;
  const Result<GoldenFile> casci = loadGolden(std::string(kGoldenCas44));
  ASSERT_TRUE(casci.has_value()) << casci.error().message;

  // Same header shape, different state: optimised orbitals are a different
  // reference, which is the whole reason the kind is recorded. How far apart
  // the two energies are is a measurement, and it lives in
  // docs/reference-data.md, "What 'CASSCF input' means here" -- not here.
  EXPECT_EQ(casscf->ncas, casci->ncas);
  EXPECT_EQ(casscf->ndet, casci->ndet);
  EXPECT_NE(casscf->reference, casci->reference);
  EXPECT_NE(casscf->eCasci, casci->eCasci);
  EXPECT_NE(casscf->eNevpt2Total, casci->eNevpt2Total);
}

TEST(GoldenCommittedCasscfTests, TheGatesSplitOnTheCommittedCasscfGolden) {
  const Result<GoldenFile> r = loadGolden(std::string(kGoldenCas44Casscf));
  ASSERT_TRUE(r.has_value()) << r.error().message;
  expectTheGatesSplit(*r);
  // And the CASCI twin beside it passes both gates, so the Lagrangian's
  // refusal is the kind's doing and not the gate's.
  const Result<GoldenFile> casci = loadGolden(std::string(kGoldenCas44));
  ASSERT_TRUE(casci.has_value()) << casci.error().message;
  EXPECT_TRUE(casci->requireSourceTermReference("--sc-tangent").has_value());
  EXPECT_TRUE(casci->requireLagrangianReference("the analytic gradient").has_value());
}

// --- GoldenCas44Tests ---------------------------------------------------------

class GoldenCas44Tests : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    Result<GoldenFile> r = loadGolden(std::string(kGoldenCas44));
    ASSERT_TRUE(r.has_value()) << r.error().message;
    golden_ = std::make_unique<GoldenFile>(std::move(*r));
  }
  static void TearDownTestSuite() { golden_.reset(); }

  void SetUp() override { ASSERT_NE(golden_, nullptr) << "loadGolden failed in SetUpTestSuite"; }

  static std::unique_ptr<GoldenFile> golden_;
};

std::unique_ptr<GoldenFile> GoldenCas44Tests::golden_;

TEST_F(GoldenCas44Tests, HeaderFields) {
  const GoldenFile& g = *golden_;
  EXPECT_EQ(g.ncas, 4);
  EXPECT_EQ(g.nelecA, 2);
  EXPECT_EQ(g.nelecB, 2);
  EXPECT_EQ(g.ncore, 5);
  EXPECT_EQ(g.ndet, 36);
  // Only that the scalars were read from the right offsets -- not a
  // numerical claim; the golden tier checks energies.
  EXPECT_NEAR(g.eCasci, -109.01797129075752, 1e-7);
  EXPECT_NEAR(g.eNevpt2Total, -0.23492101637383625, 1e-7);
}

TEST_F(GoldenCas44Tests, ArrayShapes) {
  const GoldenFile& g = *golden_;
  const int64_t n = g.ncas, c = g.ncore, v = 19;
  const std::vector<std::pair<std::string, std::vector<int64_t>>> expected = {
      {"ci", {6, 6}},
      {"dm1", {n, n}},
      {"dm2", {n, n, n, n}},
      {"class_norms", {8}},
      {"class_energies", {8}},
      {"h1e", {n, n}},
      {"h2e", {n, n, n, n}},
      {"e_core", {c}},
      {"e_virt", {v}},
      {"h2e_v_Sr", {v, n, n, n}},
      {"h1e_v_Sr", {v, n}},
      {"h2e_v_Si", {n, n, c, n}},
      {"h1e_v_Si", {n, c}},
      {"cvcv", {c, v, c, v}},
      {"h2e_v_Sijr", {v, n, c, c}},
      {"h2e_v_Srsi", {v, v, c, n}},
      {"h2e_v_Srs", {v, v, n, n}},
      {"h2e_v_Sij", {n, n, c, c}},
      {"h2e_v1_Sir", {v, n, c, n}},
      {"h2e_v2_Sir", {v, n, n, c}},
      {"h1e_v_Sir", {v, c}},
      {"pc_class_energies", {8}},
      {"e_pc_total", {1}},
  };
  // Exactly these arrays: none missing, none extra.
  EXPECT_EQ(g.arrays.size(), expected.size());
  for (const auto& [name, dims] : expected) {
    ASSERT_TRUE(g.require({name}).has_value()) << name;
    const Tensor& t = g.get(name);
    EXPECT_EQ(t.dims(), dims) << name;
    int64_t product = 1;
    for (const int64_t d : dims) product *= d;
    EXPECT_EQ(t.size(), product) << name;
  }
  // The CI vector is one coefficient per determinant.
  EXPECT_EQ(g.get("ci").size(), g.ndet);
}

TEST_F(GoldenCas44Tests, RequireEveryArrayTheDemoReads) {
  EXPECT_TRUE(golden_->require({"ci", "dm1", "dm2", "h1e", "h2e", "e_core", "e_virt",
                                "class_energies", "pc_class_energies", "e_pc_total"})
                  .has_value());
}

TEST_F(GoldenCas44Tests, ArraysAreRowMajor) {
  // The file is written in C order; the loaded Tensor's strides say so.
  const Tensor& h2e = golden_->get("h2e");
  const int64_t n = golden_->ncas;
  EXPECT_EQ(h2e.stride(0), n * n * n);
  EXPECT_EQ(h2e.stride(1), n * n);
  EXPECT_EQ(h2e.stride(2), n);
  EXPECT_EQ(h2e.stride(3), 1);
}

// --- GoldenLayoutContractTests ------------------------------------------------
//
// The C++ half of the golden layout contract (docs/reference-data.md, "The
// file format"; the Python half is reference_data/golden_layout_check.py).
// One header change -- NEVPT2G1 -> NEVPT2G2 -- broke three readers, so every
// committed file of BOTH layouts goes through loadGolden here, and a change to
// the header that the reader does not follow fails this suite rather than
// misreading a golden. A misread header cannot pass quietly: loadGolden walks
// the whole array table and refuses a malformed array header, a short file or
// a byte left over, and a ci that is not ndet long.

std::vector<std::string> splitPaths(const std::string_view joined) {
  std::vector<std::string> out;
  for (const auto part : std::views::split(joined, '|'))
    if (!part.empty()) out.emplace_back(std::string_view(part));
  return out;
}

TEST(GoldenLayoutContractTests, EveryCommittedGoldenOfBothLayoutsLoads) {
  const std::vector<std::string> paths = splitPaths(NEVPT2_TEST_COMMITTED_GOLDENS);
  ASSERT_FALSE(paths.empty()) << "test/golden/CMakeLists.txt globbed no golden";
  std::map<std::string, int> perMagic;
  std::map<std::string, std::set<int64_t>> arraysAtPerMagic;
  for (const std::string& path : paths) {
    SCOPED_TRACE(path);
    const std::string magic = readAll(path).substr(0, 8);
    const Result<GoldenFile> r = loadGolden(path);
    ASSERT_TRUE(r.has_value()) << r.error().message;
    ++perMagic[magic];
    arraysAtPerMagic[magic].insert(r->arraysAt);
    // Every demo reads these, and only a correctly-placed array table yields
    // them under their own names.
    EXPECT_TRUE(r->require({"ci", "class_energies", "h1e", "h2e"}).has_value());
    EXPECT_EQ(r->get("ci").size(), r->ndet);
    EXPECT_GT(r->ncas, 0);
    EXPECT_EQ(r->get("h1e").dims(), (std::vector<int64_t>{r->ncas, r->ncas}));
    // The layout says the kind: G1 is a CASCI by definition.
    if (magic == "NEVPT2G1") EXPECT_EQ(r->reference, ReferenceKind::Casci);
  }
  // Both layouts on disk, or this suite is not testing what it claims to.
  EXPECT_GT(perMagic["NEVPT2G1"], 0) << "no committed NEVPT2G1 golden was loaded";
  EXPECT_GT(perMagic["NEVPT2G2"], 0) << "no committed NEVPT2G2 golden was loaded";
  EXPECT_EQ(std::ssize(perMagic), 2) << "a golden in a third layout: extend this suite";
  // Within one layout the header has one length, and the two layouts' differ:
  // the reader found each file's array table where that file put it.
  for (const auto& [magic, at] : arraysAtPerMagic) EXPECT_EQ(std::ssize(at), 1) << magic;
  EXPECT_NE(arraysAtPerMagic["NEVPT2G1"], arraysAtPerMagic["NEVPT2G2"]);
}

TEST(GoldenLayoutContractTests, EveryCommittedGradientSidecarLoadsAgainstItsGolden) {
  const std::vector<std::string> paths = splitPaths(NEVPT2_TEST_COMMITTED_GRADGOLDS);
  ASSERT_FALSE(paths.empty()) << "test/golden/CMakeLists.txt globbed no gradient sidecar";
  for (const std::string& path : paths) {
    SCOPED_TRACE(path);
    const Result<GradGoldFile> side = loadGradGold(path);
    ASSERT_TRUE(side.has_value()) << side.error().message;
    EXPECT_TRUE(side->require({"mo_coeff"}).has_value());
    std::filesystem::path twin(path);
    twin.replace_extension(".nevpt2gold");
    if (!std::filesystem::exists(twin)) continue;
    const Result<GoldenFile> golden = loadGolden(twin.string());
    ASSERT_TRUE(golden.has_value()) << golden.error().message;
    EXPECT_TRUE(side->checkAgainst(*golden).has_value());
  }
}

// The golden and gradient sidecar beside an oracle path: <case>.<ext>, where
// an oracle is <case>[_pc][_dir].srcgold (srcGoldCase).
std::filesystem::path besideOracle(const std::string& oracle, SourceScheme scheme,
                                   SourceFlavour flavour, std::string_view ext) {
  const Result<std::string> c = srcGoldCase(oracle, scheme, flavour);
  EXPECT_TRUE(c.has_value()) << c.error().message;
  return std::filesystem::path(oracle).parent_path() /
         (c.value_or("") + std::string(ext));
}

TEST(GoldenLayoutContractTests, EveryCommittedSourceOracleLoadsAgainstItsGolden) {
  const std::vector<std::string> paths = splitPaths(NEVPT2_TEST_COMMITTED_SRCGOLDS);
  ASSERT_FALSE(paths.empty()) << "test/golden/CMakeLists.txt globbed no source-term oracle";
  std::map<std::pair<SourceScheme, SourceFlavour>, int> perKind;
  int withSidecar = 0;
  for (const std::string& path : paths) {
    SCOPED_TRACE(path);
    const Result<SrcGoldFile> o = loadSrcGold(path);
    ASSERT_TRUE(o.has_value()) << o.error().message;
    ++perKind[{o->scheme, o->flavour}];
    EXPECT_GT(o->kappaStep, 0.0);
    EXPECT_GT(o->ciStep, 0.0);
    const Result<GoldenFile> golden =
        loadGolden(besideOracle(path, o->scheme, o->flavour, ".nevpt2gold").string());
    ASSERT_TRUE(golden.has_value()) << golden.error().message;
    const Status mine = o->checkAgainstGolden(*golden);
    EXPECT_TRUE(mine.has_value()) << mine.error().message;
    // Acrolein's 57 MB sidecar is gitignored (docs/reference-data.md, "The
    // gradient sidecar"): held to it when it has been regenerated.
    const std::filesystem::path sidePath = besideOracle(path, o->scheme, o->flavour, ".gradgold");
    if (!std::filesystem::exists(sidePath)) continue;
    const Result<GradGoldFile> side = loadGradGold(sidePath.string());
    ASSERT_TRUE(side.has_value()) << side.error().message;
    const Status st = o->checkAgainst(*golden, *side);
    EXPECT_TRUE(st.has_value()) << st.error().message;
    ++withSidecar;
  }
  // All four kinds on disk, or this suite is not testing what it claims to.
  for (const SourceScheme sch : {SourceScheme::Sc, SourceScheme::Pc})
    for (const SourceFlavour fl : {SourceFlavour::Full, SourceFlavour::Directional})
      EXPECT_GT((perKind[{sch, fl}]), 0)
          << "no committed " << sourceSchemeName(sch) << " " << sourceFlavourName(fl)
          << " oracle was loaded";
  EXPECT_GT(withSidecar, 0) << "no oracle was held to its gradient sidecar";
}

// --- SrcGoldLoadErrorTests ----------------------------------------------------

using SrcGoldLoadErrorTests = GoldenTempDir;

// The first committed oracle of `flavour` whose gradient sidecar is
// committed too (acrolein's is not), or "".
std::string firstCommittedOracle(SourceFlavour flavour = SourceFlavour::Full) {
  for (const std::string& path : splitPaths(NEVPT2_TEST_COMMITTED_SRCGOLDS)) {
    const Result<SrcGoldFile> o = loadSrcGold(path);
    if (!o || o->flavour != flavour) continue;
    if (std::filesystem::exists(besideOracle(path, o->scheme, o->flavour, ".gradgold")))
      return path;
  }
  return {};
}

void expectOracleIoError(const Result<SrcGoldFile>& r, std::string_view fragment) {
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::IO);
  EXPECT_NE(r.error().message.find(fragment), std::string::npos) << r.error().message;
}

TEST_F(SrcGoldLoadErrorTests, AGoldenIsNotAnOracle) {
  expectOracleIoError(loadSrcGold(std::string(kGoldenCas44)), "expected 'NEVPT2S1'");
}

TEST_F(SrcGoldLoadErrorTests, TruncatedAndOverlongOraclesAreIoErrors) {
  const std::string oracle = firstCommittedOracle();
  ASSERT_FALSE(oracle.empty());
  const std::string bytes = readAll(oracle);
  ASSERT_GT(bytes.size(), 40u);
  expectOracleIoError(loadSrcGold(write("trunc_header.srcgold", bytes.substr(0, 20))),
                      "unexpected EOF");
  expectOracleIoError(loadSrcGold(write("trunc_tail.srcgold", bytes.substr(0, bytes.size() - 1))),
                      "source-term oracle");
  expectOracleIoError(loadSrcGold(write("trailing.srcgold", bytes + '\0')),
                      "1 bytes after the last");
}

TEST_F(SrcGoldLoadErrorTests, TheMagicIsTheScheme) {
  // One layout, two magics: an SC oracle's bytes under the PC magic load as a
  // PC oracle with every number unchanged, and under a third magic do not
  // load at all.
  const std::string oracle = firstCommittedOracle();
  ASSERT_FALSE(oracle.empty());
  const Result<SrcGoldFile> sc = loadSrcGold(oracle);
  ASSERT_TRUE(sc.has_value()) << sc.error().message;
  std::string bytes = readAll(oracle);
  const std::string_view was(bytes.data(), 8);
  ASSERT_TRUE(was == "NEVPT2S1" || was == "NEVPT2P1") << was;
  const bool wasPc = was == "NEVPT2P1";
  bytes.replace(0, 8, wasPc ? "NEVPT2S1" : "NEVPT2P1");
  const Result<SrcGoldFile> flipped = loadSrcGold(write("flipped_pc.srcgold", bytes));
  ASSERT_TRUE(flipped.has_value()) << flipped.error().message;
  EXPECT_EQ(sc->scheme, wasPc ? SourceScheme::Pc : SourceScheme::Sc);
  EXPECT_EQ(flipped->scheme, wasPc ? SourceScheme::Sc : SourceScheme::Pc);
  EXPECT_EQ(flipped->nmo, sc->nmo);
  EXPECT_EQ(flipped->ndet, sc->ndet);
  EXPECT_EQ(flipped->get("a_corr").data(), sc->get("a_corr").data());
  bytes.replace(0, 8, "NEVPT2X1");
  expectOracleIoError(loadSrcGold(write("third.srcgold", bytes)),
                      "expected 'NEVPT2S1', 'NEVPT2P1', 'NEVPT2SD' or 'NEVPT2PD'");
}

TEST_F(SrcGoldLoadErrorTests, TheMagicIsTheFlavourToo) {
  // A directional oracle's bytes under its scheme's other directional magic
  // load with the scheme flipped and every array unchanged; under a FULL
  // magic they are refused, naming the array a full oracle needs.
  const std::string oracle = firstCommittedOracle(SourceFlavour::Directional);
  ASSERT_FALSE(oracle.empty());
  const Result<SrcGoldFile> o = loadSrcGold(oracle);
  ASSERT_TRUE(o.has_value()) << o.error().message;
  EXPECT_EQ(o->flavour, SourceFlavour::Directional);
  std::string bytes = readAll(oracle);
  const bool wasPc = std::string_view(bytes.data(), 8) == "NEVPT2PD";
  bytes.replace(0, 8, wasPc ? "NEVPT2SD" : "NEVPT2PD");
  const Result<SrcGoldFile> flipped = loadSrcGold(write("flipped_dir.srcgold", bytes));
  ASSERT_TRUE(flipped.has_value()) << flipped.error().message;
  EXPECT_EQ(flipped->scheme, wasPc ? SourceScheme::Sc : SourceScheme::Pc);
  EXPECT_EQ(flipped->flavour, SourceFlavour::Directional);
  EXPECT_EQ(flipped->get("kappa_val").data(), o->get("kappa_val").data());
  bytes.replace(0, 8, "NEVPT2S1");
  expectOracleIoError(loadSrcGold(write("full_magic.srcgold", bytes)),
                      "missing required array 'a_corr'");
}

TEST_F(SrcGoldLoadErrorTests, ADirectionThatIsNotAntisymmetricIsRefused) {
  const std::string oracle = firstCommittedOracle(SourceFlavour::Directional);
  ASSERT_FALSE(oracle.empty());
  std::string bytes = readAll(oracle);
  // The array section names each array before its rank, its extents and its
  // data: kappa_dirs[0, 0, 1] is the second double after the three extents.
  const std::string name = "kappa_dirs";
  const std::size_t at = bytes.find(name);
  ASSERT_NE(at, std::string::npos);
  const std::size_t data = at + name.size() + 4 + 3 * 8;
  double v = 0.0;
  std::memcpy(&v, bytes.data() + data + 8, sizeof v);
  v += 1.0;  // (0, 1) no longer the negative of (1, 0)
  std::memcpy(bytes.data() + data + 8, &v, sizeof v);
  expectOracleIoError(loadSrcGold(write("asym.srcgold", bytes)),
                      "kappa direction 0 is not antisymmetric at (0, 1)");
}

TEST_F(SrcGoldLoadErrorTests, ADirectionalOracleIsNamedForItsCase) {
  const Result<std::string> pc = srcGoldCase("/x/golden/acrolein_def2svp_cas65_pc_dir.srcgold",
                                             SourceScheme::Pc, SourceFlavour::Directional);
  ASSERT_TRUE(pc.has_value()) << pc.error().message;
  EXPECT_EQ(*pc, "acrolein_def2svp_cas65");
  const Result<std::string> sc = srcGoldCase("/x/golden/acrolein_def2svp_cas65_dir.srcgold",
                                             SourceScheme::Sc, SourceFlavour::Directional);
  ASSERT_TRUE(sc.has_value()) << sc.error().message;
  EXPECT_EQ(*sc, "acrolein_def2svp_cas65");
  const Result<std::string> noDir = srcGoldCase("/x/golden/n2_ccpvdz_cas44.srcgold",
                                                SourceScheme::Sc, SourceFlavour::Directional);
  ASSERT_FALSE(noDir.has_value());
  EXPECT_NE(noDir.error().message.find("must end in '_dir.srcgold'"), std::string::npos)
      << noDir.error().message;
  const Result<std::string> noPc = srcGoldCase("/x/golden/n2_ccpvdz_cas44_dir.srcgold",
                                               SourceScheme::Pc, SourceFlavour::Directional);
  ASSERT_FALSE(noPc.has_value());
  EXPECT_NE(noPc.error().message.find("must end in '_pc_dir.srcgold'"), std::string::npos)
      << noPc.error().message;
}

TEST_F(SrcGoldLoadErrorTests, APcOracleIsNamedForItsCase) {
  const Result<std::string> pc = srcGoldCase("/x/golden/n2_ccpvdz_cas44_pc.srcgold",
                                             SourceScheme::Pc);
  ASSERT_TRUE(pc.has_value()) << pc.error().message;
  EXPECT_EQ(*pc, "n2_ccpvdz_cas44");
  const Result<std::string> sc = srcGoldCase("/x/golden/n2_ccpvdz_cas44.srcgold",
                                             SourceScheme::Sc);
  ASSERT_TRUE(sc.has_value()) << sc.error().message;
  EXPECT_EQ(*sc, "n2_ccpvdz_cas44");
  // A PC oracle without the suffix would claim the SC oracle's place.
  const Result<std::string> bad = srcGoldCase("/x/golden/n2_ccpvdz_cas44.srcgold",
                                              SourceScheme::Pc);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error().kind, ErrorKind::IO);
  EXPECT_NE(bad.error().message.find("must end in '_pc.srcgold'"), std::string::npos)
      << bad.error().message;
}

TEST_F(SrcGoldLoadErrorTests, AnOracleForAnotherStateIsRefused) {
  const std::string oracle = firstCommittedOracle();
  ASSERT_FALSE(oracle.empty());
  Result<SrcGoldFile> o = loadSrcGold(oracle);
  ASSERT_TRUE(o.has_value()) << o.error().message;
  Result<GoldenFile> golden =
      loadGolden(besideOracle(oracle, o->scheme, o->flavour, ".nevpt2gold").string());
  ASSERT_TRUE(golden.has_value()) << golden.error().message;
  const Result<GradGoldFile> side =
      loadGradGold(besideOracle(oracle, o->scheme, o->flavour, ".gradgold").string());
  ASSERT_TRUE(side.has_value()) << side.error().message;
  // One CI coefficient moved by one ulp: a different state, bit for bit.
  Tensor& ci = golden->arrays.at("ci");
  ci.flatRef(0) = std::nextafter(ci.flat(0), 2.0);
  const Status st = o->checkAgainst(*golden, *side);
  ASSERT_FALSE(st.has_value());
  EXPECT_EQ(st.error().kind, ErrorKind::IO);
  EXPECT_NE(st.error().message.find("its ci differs from the golden's at determinant 0"),
            std::string::npos)
      << st.error().message;
  // And a golden of another active space.
  golden->ncas += 1;
  const Status st2 = o->checkAgainst(*golden, *side);
  ASSERT_FALSE(st2.has_value());
  EXPECT_NE(st2.error().message.find("do not describe the same state"), std::string::npos)
      << st2.error().message;
}

}  // namespace nevpt2::test::golden
