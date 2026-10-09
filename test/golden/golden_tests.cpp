// Suites for nevpt2.golden, the .nevpt2gold reader.
//
// Host-only ON PURPOSE: loadGolden is file parsing, so this binary is not
// REQUIRES_GPU and runs with no card visible. It checks the loader's logic --
// a refusal's kind, a parsed header, an array's shape -- never an energy: the
// golden tier owns every numerical claim, and the energies read below are only
// compared with the values the file was written with.
//
//   GoldenLoadErrorTests  a missing path, a truncated file and a bad magic
//                         each come back as an IO Error -- never an abort
//   GoldenRequireTests    require() names the first array the file lacks
//   GoldenCas44Tests      the committed golden/n2_ccpvdz_cas44.nevpt2gold:
//                         header fields and array shapes
//
// The malformed files are written to a per-test temp directory, removed in
// TearDown. The committed golden is only ever read; its path comes from CMake
// (NEVPT2_TEST_GOLDEN_CAS44), not from this source.
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.golden;

#ifndef NEVPT2_TEST_GOLDEN_CAS44
#error "NEVPT2_TEST_GOLDEN_CAS44 must be defined by test/golden/CMakeLists.txt"
#endif

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
// One binary per suite file, so a name clash here is a link error.
namespace nevpt2::test::golden {

constexpr std::string_view kGoldenCas44 = NEVPT2_TEST_GOLDEN_CAS44;

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

}  // namespace nevpt2::test::golden
