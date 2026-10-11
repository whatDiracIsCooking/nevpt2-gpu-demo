// Suites for loadGradGold, the `*.gradgold` gradient-sidecar reader.
//
// Host-only ON PURPOSE, like golden_tests.cpp: this is file parsing, so the
// binary is not REQUIRES_GPU and runs with no card visible. It checks the
// loader's logic -- a refusal's kind, a parsed header, an array's shape and
// its values -- never an energy.
//
//   GradGoldRoundTripTests  a sidecar this test writes byte by byte loads
//                           back with the same header and the same array
//                           values, to 1e-12
//   GradGoldLoadErrorTests  a missing path, an empty file, a bad magic, a
//                           truncated header, truncated array data, trailing
//                           bytes, a space partition that does not add up to
//                           nmo, an RHF electron count that cannot fill the
//                           core, a negative array count, an oversized rank
//                           and an mo_coeff of the wrong shape each come back
//                           as a value-tier IO Error -- never an abort, the
//                           same contract loadGolden has
//   GradGoldCas44Tests      the committed golden/n2_ccpvdz_cas44.gradgold:
//                           header, every array's shape against
//                           docs/gradient-theory.md's inventory, the (O1)
//                           invariant of its §3.6, and the one-electron
//                           matrices against the slices of them the committed
//                           golden beside it carries (§3.5)
//
// The malformed and synthetic files are written to a per-test temp directory,
// removed in TearDown. The two committed files are only ever read; their paths
// come from CMake, not from this source.
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.golden;

#ifndef NEVPT2_TEST_GRADGOLD_CAS44
#error "NEVPT2_TEST_GRADGOLD_CAS44 must be defined by test/golden/CMakeLists.txt"
#endif
#ifndef NEVPT2_TEST_GOLDEN_CAS44
#error "NEVPT2_TEST_GOLDEN_CAS44 must be defined by test/golden/CMakeLists.txt"
#endif

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::gradgold {

constexpr std::string_view kGradGoldCas44 = NEVPT2_TEST_GRADGOLD_CAS44;
constexpr std::string_view kGoldenCas44 = NEVPT2_TEST_GOLDEN_CAS44;

// The committed CAS(4,4) sidecar's partition. Not a measured number: the
// active space is the case's definition and the rest follows from cc-pVDZ on
// N2 (docs/reference-data.md, "The small cases").
constexpr int64_t kNao = 28, kNmo = 28, kNcore = 5, kNact = 4, kNvirt = 19, kNelecRhf = 14;

// How far the RHF Fock's core/active/virtual off-blocks may be from zero:
// generate_golden.py's O1_FOCK_TOL, which the sidecar was written under. (O1)
// holds only up to RHF convergence, not to machine precision
// (docs/gradient-theory.md, §3.6; docs/reference-data.md, "The gradient
// sidecar").
constexpr double kO1Tol = 1e-5;

// The sidecar is written from the same matrices the golden's one-electron
// fields are slices of, so the two agree to the f64 round trip and nothing
// looser.
constexpr double kExactTol = 1e-12;

// --- a byte-level writer, so a test can build a sidecar it knows the contents
// --- of and a malformed one the reader must refuse ---------------------------

void appendRaw(std::string& out, const void* p, const std::size_t n) {
  const char* bytes = static_cast<const char*>(p);
  out.append(bytes, n);
}
void appendI32(std::string& out, const std::int32_t v) { appendRaw(out, &v, 4); }
void appendI64(std::string& out, const std::int64_t v) { appendRaw(out, &v, 8); }

// One named array record: name length, name, rank, int64 extents, C-order f64
// data (generate_golden.py's _write_arrays).
void appendArray(std::string& out, const std::string_view name, const std::vector<int64_t>& dims,
                 const std::vector<double>& data) {
  appendI32(out, static_cast<std::int32_t>(name.size()));
  out.append(name);
  appendI32(out, static_cast<std::int32_t>(dims.size()));
  for (const int64_t d : dims) appendI64(out, d);
  appendRaw(out, data.data(), data.size() * sizeof(double));
}

// The 8-byte magic and the six-int32 scalar header.
std::string sidecarHeader(const std::string_view magic, const int64_t nao, const int64_t nmo,
                          const int64_t ncore, const int64_t nact, const int64_t nvirt,
                          const int64_t nelecRhf) {
  std::string out;
  out.append(magic);
  for (const int64_t v : {nao, nmo, ncore, nact, nvirt, nelecRhf})
    appendI32(out, static_cast<std::int32_t>(v));
  return out;
}

// Deterministic, irrational-looking values with a full f64 mantissa, so a
// 1e-12 round-trip comparison is a real check and not a comparison of zeros.
double fill(const int64_t k) { return std::cos(static_cast<double>(k) * 0.37) * 3.5; }

std::vector<double> filled(const int64_t n, const int64_t seed) {
  std::vector<double> v(static_cast<std::size_t>(n));
  for (int64_t i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = fill(seed + i);
  return v;
}

int64_t product(const std::vector<int64_t>& dims) {
  int64_t p = 1;
  for (const int64_t d : dims) p *= d;
  return p;
}

std::string readAll(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// A fixture owning one fresh temp directory per test, removed afterwards.
class GradGoldTempDir : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::random_device rd;
    dir_ = std::filesystem::temp_directory_path() /
           std::format("nevpt2_gradgold_tests_{}_{}_{:08x}", info->test_suite_name(), info->name(),
                       rd());
    std::filesystem::create_directories(dir_);
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }

  // Writes `bytes` to `<dir>/<name>` and returns its path.
  std::string write(const std::string& name, const std::string_view bytes) const {
    const std::filesystem::path p = dir_ / name;
    std::ofstream out(p, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    EXPECT_TRUE(out.good()) << "could not write " << p;
    return p.string();
  }

  std::filesystem::path dir_;
};

void expectIoError(const Result<GradGoldFile>& r, const std::string_view fragment) {
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::IO);
  EXPECT_NE(r.error().message.find(fragment), std::string::npos) << r.error().message;
}

// --- GradGoldRoundTripTests ---------------------------------------------------

using GradGoldRoundTripTests = GradGoldTempDir;

TEST_F(GradGoldRoundTripTests, HeaderAndEveryArrayValueSurviveTheRoundTrip) {
  // A miniature partition, shaped like a real sidecar's: one matrix over the
  // AO range, one over the MO range, one rank-4 integral block and one of the
  // rank-3 restricted blocks.
  constexpr int64_t nao = 7, ncore = 2, nact = 3, nvirt = 4, nmo = ncore + nact + nvirt;
  const std::vector<std::pair<std::string, std::vector<int64_t>>> spec = {
      {"mo_coeff", {nao, nmo}},
      {"fock_rhf", {nmo, nmo}},
      {"g_xa_cv", {nmo, nact, ncore, nvirt}},
      {"g_xc_vv", {nmo, ncore, nvirt}},
  };

  std::string bytes = sidecarHeader("NEVPT2D1", nao, nmo, ncore, nact, nvirt, 2 * ncore + 2);
  appendI32(bytes, static_cast<std::int32_t>(spec.size()));
  std::vector<std::vector<double>> written;
  int64_t seed = 0;
  for (const auto& [name, dims] : spec) {
    written.push_back(filled(product(dims), seed));
    appendArray(bytes, name, dims, written.back());
    seed += 1000;
  }

  const Result<GradGoldFile> r = loadGradGold(write("mini.gradgold", bytes));
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->nao, nao);
  EXPECT_EQ(r->nmo, nmo);
  EXPECT_EQ(r->ncore, ncore);
  EXPECT_EQ(r->nact, nact);
  EXPECT_EQ(r->nvirt, nvirt);
  EXPECT_EQ(r->nelecRhf, 2 * ncore + 2);
  EXPECT_EQ(std::ssize(r->arrays), std::ssize(spec));

  for (std::size_t i = 0; i < spec.size(); ++i) {
    const auto& [name, dims] = spec[i];
    ASSERT_TRUE(r->require({name}).has_value()) << name;
    const Tensor& t = r->get(name);
    EXPECT_EQ(t.dims(), dims) << name;
    ASSERT_EQ(t.size(), std::ssize(written[i])) << name;
    double worst = 0.0;
    for (int64_t k = 0; k < t.size(); ++k)
      worst = std::max(worst, std::abs(t.flat(k) - written[i][static_cast<std::size_t>(k)]));
    EXPECT_LE(worst, kExactTol) << name;
  }
}

TEST_F(GradGoldRoundTripTests, RowMajorOrderSurvivesTheRoundTrip) {
  // The writer emits C order; the loaded Tensor's strides and element order
  // must say so, or every index into a sidecar block would be transposed.
  constexpr int64_t nmo = 4, ncore = 1, nact = 1, nvirt = 2;
  const std::vector<int64_t> dims{nmo, nact, ncore, nvirt};
  const std::vector<double> data = filled(product(dims), 17);
  std::string bytes = sidecarHeader("NEVPT2D1", 4, nmo, ncore, nact, nvirt, 2);
  appendI32(bytes, 1);
  appendArray(bytes, "g_xa_cv", dims, data);

  const Result<GradGoldFile> r = loadGradGold(write("order.gradgold", bytes));
  ASSERT_TRUE(r.has_value()) << r.error().message;
  const Tensor& t = r->get("g_xa_cv");
  EXPECT_EQ(t.stride(0), nact * ncore * nvirt);
  EXPECT_EQ(t.stride(1), ncore * nvirt);
  EXPECT_EQ(t.stride(2), nvirt);
  EXPECT_EQ(t.stride(3), 1);
  int64_t k = 0;
  for (int64_t x = 0; x < nmo; ++x)
    for (int64_t p = 0; p < nact; ++p)
      for (int64_t i = 0; i < ncore; ++i)
        for (int64_t v = 0; v < nvirt; ++v, ++k)
          EXPECT_LE(std::abs(t(x, p, i, v) - data[static_cast<std::size_t>(k)]), kExactTol);
}

// --- GradGoldLoadErrorTests ---------------------------------------------------

using GradGoldLoadErrorTests = GradGoldTempDir;

// A well-formed one-array sidecar, as the base for each corruption below.
std::string goodSidecar() {
  std::string bytes = sidecarHeader("NEVPT2D1", 3, 4, 1, 1, 2, 2);
  appendI32(bytes, 1);
  appendArray(bytes, "mo_coeff", {3, 4}, filled(12, 5));
  return bytes;
}

TEST_F(GradGoldLoadErrorTests, MissingPathIsAnIoError) {
  const std::string path = (dir_ / "does_not_exist.gradgold").string();
  const Result<GradGoldFile> r = loadGradGold(path);
  expectIoError(r, "could not open gradient sidecar");
  if (!r.has_value()) EXPECT_NE(r.error().message.find(path), std::string::npos);
}

TEST_F(GradGoldLoadErrorTests, EmptyFileIsAnIoError) {
  // Too short even for the 8-byte magic, refused before any read.
  expectIoError(loadGradGold(write("empty.gradgold", "")), "bad string length 8");
}

TEST_F(GradGoldLoadErrorTests, BadMagicIsAnIoError) {
  // Notably: a GOLDEN handed to loadGradGold is refused here, not parsed as a
  // sidecar whose header happens to read.
  const std::string bytes = readAll(std::string(kGoldenCas44));
  ASSERT_GT(bytes.size(), 8u);
  const Result<GradGoldFile> r = loadGradGold(write("is_a_golden.gradgold", bytes));
  expectIoError(r, "bad magic");
  if (!r.has_value()) EXPECT_NE(r.error().message.find("NEVPT2G1"), std::string::npos);
}

TEST_F(GradGoldLoadErrorTests, SidecarHandedToLoadGoldenIsAnIoError) {
  // The converse, so neither loader can be fed the other's file silently.
  const Result<GoldenFile> r = loadGolden(std::string(kGradGoldCas44));
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::IO);
  EXPECT_NE(r.error().message.find("bad magic"), std::string::npos) << r.error().message;
}

TEST_F(GradGoldLoadErrorTests, TruncatedHeaderIsAnIoError) {
  // The magic and part of the six-int32 scalar header only.
  const std::string bytes = goodSidecar();
  expectIoError(loadGradGold(write("trunc_header.gradgold", bytes.substr(0, 8 + 12))),
                "unexpected EOF");
}

TEST_F(GradGoldLoadErrorTests, TruncatedArrayDataIsAnIoError) {
  const std::string bytes = goodSidecar();
  expectIoError(loadGradGold(write("trunc_tail.gradgold", bytes.substr(0, bytes.size() - 1))),
                "gradient sidecar");
}

TEST_F(GradGoldLoadErrorTests, TruncatedCommittedSidecarIsAnIoError) {
  // Cut halfway: an array's extents now promise more data than the file holds.
  const std::string bytes = readAll(std::string(kGradGoldCas44));
  ASSERT_GT(bytes.size(), 2u);
  expectIoError(loadGradGold(write("trunc_mid.gradgold", bytes.substr(0, bytes.size() / 2))),
                "gradient sidecar");
}

TEST_F(GradGoldLoadErrorTests, TrailingBytesAreAnIoError) {
  const std::string bytes = goodSidecar() + '\0';
  expectIoError(loadGradGold(write("trailing.gradgold", bytes)), "1 bytes after the last");
}

TEST_F(GradGoldLoadErrorTests, PartitionThatDoesNotSumToNmoIsAnIoError) {
  // Every array's shape is written in terms of ncore/nact/nvirt, so a
  // partition that does not add up to nmo describes no file.
  std::string bytes = sidecarHeader("NEVPT2D1", 3, 4, 1, 1, 1, 2);
  appendI32(bytes, 0);
  expectIoError(loadGradGold(write("bad_partition.gradgold", bytes)),
                "ncore + nact + nvirt = 1 + 1 + 1 = 3, but nmo = 4");
}

TEST_F(GradGoldLoadErrorTests, AnEmptyActiveSpaceIsAnIoError) {
  std::string bytes = sidecarHeader("NEVPT2D1", 3, 4, 1, 0, 3, 2);
  appendI32(bytes, 0);
  expectIoError(loadGradGold(write("no_active.gradgold", bytes)),
                "is not a usable orbital-space partition");
}

TEST_F(GradGoldLoadErrorTests, AnOddRhfElectronCountIsAnIoError) {
  // The reference is a closed-shell RHF: an odd count cannot say which half
  // of F's active spectrum is occupied.
  std::string bytes = sidecarHeader("NEVPT2D1", 3, 4, 1, 1, 2, 3);
  appendI32(bytes, 0);
  expectIoError(loadGradGold(write("odd_nelec.gradgold", bytes)), "nelec_rhf = 3");
}

TEST_F(GradGoldLoadErrorTests, AnRhfElectronCountThatCannotFillTheCoreIsAnIoError) {
  std::string bytes = sidecarHeader("NEVPT2D1", 6, 6, 2, 2, 2, 2);
  appendI32(bytes, 0);
  expectIoError(loadGradGold(write("too_few_nelec.gradgold", bytes)), "nelec_rhf = 2");
}

TEST_F(GradGoldLoadErrorTests, NegativeArrayCountIsAnIoError) {
  std::string bytes = sidecarHeader("NEVPT2D1", 3, 4, 1, 1, 2, 2);
  appendI32(bytes, -1);
  expectIoError(loadGradGold(write("neg_count.gradgold", bytes)), "negative array count -1");
}

TEST_F(GradGoldLoadErrorTests, AnOversizedRankIsAnIoError) {
  std::string bytes = sidecarHeader("NEVPT2D1", 3, 4, 1, 1, 2, 2);
  appendI32(bytes, 1);
  appendI32(bytes, 4);
  bytes.append("junk");
  appendI32(bytes, 999);  // the rank
  expectIoError(loadGradGold(write("bad_rank.gradgold", bytes)), "has rank 999");
}

TEST_F(GradGoldLoadErrorTests, AnMoCoeffOfTheWrongShapeIsAnIoError) {
  // The one array the header's AO count can be checked against.
  std::string bytes = sidecarHeader("NEVPT2D1", 3, 4, 1, 1, 2, 2);
  appendI32(bytes, 1);
  appendArray(bytes, "mo_coeff", {4, 3}, filled(12, 5));
  expectIoError(loadGradGold(write("bad_mo.gradgold", bytes)),
                "mo_coeff has shape [4, 3] but the header says [3, 4]");
}

// --- GradGoldCas44Tests -------------------------------------------------------

class GradGoldCas44Tests : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    Result<GradGoldFile> g = loadGradGold(std::string(kGradGoldCas44));
    ASSERT_TRUE(g.has_value()) << g.error().message;
    sidecar_ = std::make_unique<GradGoldFile>(std::move(*g));
    Result<GoldenFile> r = loadGolden(std::string(kGoldenCas44));
    ASSERT_TRUE(r.has_value()) << r.error().message;
    golden_ = std::make_unique<GoldenFile>(std::move(*r));
  }
  static void TearDownTestSuite() {
    sidecar_.reset();
    golden_.reset();
  }
  void SetUp() override {
    ASSERT_NE(sidecar_, nullptr) << "loadGradGold failed in SetUpTestSuite";
    ASSERT_NE(golden_, nullptr) << "loadGolden failed in SetUpTestSuite";
  }

  static std::unique_ptr<GradGoldFile> sidecar_;
  static std::unique_ptr<GoldenFile> golden_;
};

std::unique_ptr<GradGoldFile> GradGoldCas44Tests::sidecar_;
std::unique_ptr<GoldenFile> GradGoldCas44Tests::golden_;

TEST_F(GradGoldCas44Tests, HeaderFields) {
  const GradGoldFile& g = *sidecar_;
  EXPECT_EQ(g.nao, kNao);
  EXPECT_EQ(g.nmo, kNmo);
  EXPECT_EQ(g.ncore, kNcore);
  EXPECT_EQ(g.nact, kNact);
  EXPECT_EQ(g.nvirt, kNvirt);
  EXPECT_EQ(g.nelecRhf, kNelecRhf);
  // The sidecar describes the golden beside it, so their partitions agree.
  EXPECT_EQ(g.nact, golden_->ncas);
  EXPECT_EQ(g.ncore, golden_->ncore);
}

TEST_F(GradGoldCas44Tests, ArrayShapesAreTheInventory) {
  const GradGoldFile& g = *sidecar_;
  const int64_t m = g.nmo, c = g.ncore, a = g.nact, v = g.nvirt;
  // docs/gradient-theory.md: §3.3's seventeen one-general-index blocks
  // (`(xc|cc)` on its two independent restricted slices, so eighteen arrays),
  // §3.5's matrices, and mo_coeff. `(xv|vv)` -- the eighteenth block and the
  // largest -- is deliberately absent (§3.4).
  const std::vector<std::pair<std::string, std::vector<int64_t>>> expected = {
      {"mo_coeff", {g.nao, m}},
      {"h", {m, m}},
      {"heff", {m, m}},
      {"f", {m, m}},
      {"f_h", {m, m}},
      {"fock_rhf", {m, m}},
      {"g_xa_aa", {m, a, a, a}},
      {"g_xv_aa", {m, v, a, a}},
      {"g_xc_aa", {m, c, a, a}},
      {"g_xa_va", {m, a, v, a}},
      {"g_xv_va", {m, v, v, a}},
      {"g_xc_va", {m, c, v, a}},
      {"g_xa_ca", {m, a, c, a}},
      {"g_xv_ca", {m, v, c, a}},
      {"g_xc_ca", {m, c, c, a}},
      {"g_xa_cv", {m, a, c, v}},
      {"g_xv_cv", {m, v, c, v}},
      {"g_xc_cv", {m, c, c, v}},
      {"g_xa_cc", {m, a, c, c}},
      {"g_xv_cc", {m, v, c, c}},
      {"g_xc_cc_j", {m, c, c}},
      {"g_xc_cc_k", {m, c, c}},
      {"g_xa_vv", {m, a, v}},
      {"g_xc_vv", {m, c, v}},
  };
  // Exactly these arrays: none missing, none extra. In particular no
  // n_mo*n_virt^3 block.
  EXPECT_EQ(std::ssize(g.arrays), std::ssize(expected));
  EXPECT_FALSE(g.arrays.contains("g_xv_vv"));
  for (const auto& [name, dims] : expected) {
    ASSERT_TRUE(g.require({name}).has_value()) << name;
    const Tensor& t = g.get(name);
    EXPECT_EQ(t.dims(), dims) << name;
    EXPECT_EQ(t.size(), product(dims)) << name;
  }
}

TEST_F(GradGoldCas44Tests, TheRhfFockIsBlockDiagonal) {
  // Invariant (O1) of docs/gradient-theory.md, §3.6: core, active and virtual
  // are each invariant subspaces of the RHF Fock operator, which is the one
  // statement that the active space is a set of RHF canonical orbitals. Only
  // up to RHF convergence, hence kO1Tol and not kExactTol.
  const GradGoldFile& g = *sidecar_;
  const Tensor& F = g.get("fock_rhf");
  const int64_t nocc = g.ncore + g.nact;
  // Which of the three spaces an index is in: 0 core, 1 active, 2 virtual.
  const auto space = [&](const int64_t i) { return i < g.ncore ? 0 : (i < nocc ? 1 : 2); };
  double worst = 0.0;
  for (int64_t p = 0; p < g.nmo; ++p)
    for (int64_t q = 0; q < g.nmo; ++q)
      if (space(p) != space(q)) worst = std::max(worst, std::abs(F(p, q)));
  EXPECT_LE(worst, kO1Tol);
}

TEST_F(GradGoldCas44Tests, TheGeneralizedFockIsPseudocanonicalOnTheGoldensDiagonals) {
  // Invariant (O2) of §3.6: f is diagonal inside core and inside virtual, and
  // those diagonals ARE the golden's e_core / e_virt. Exact by construction --
  // the sidecar's f and the golden's orbital energies come from one
  // canonicalisation -- so kExactTol.
  const GradGoldFile& g = *sidecar_;
  ASSERT_TRUE(golden_->require({"e_core", "e_virt"}).has_value());
  const Tensor& f = g.get("f");
  const Tensor& eCore = golden_->get("e_core");
  const Tensor& eVirt = golden_->get("e_virt");
  const int64_t nocc = g.ncore + g.nact;

  for (int64_t i = 0; i < g.ncore; ++i) {
    EXPECT_LE(std::abs(f(i, i) - eCore(i)), kExactTol) << "core " << i;
    for (int64_t j = 0; j < g.ncore; ++j)
      if (i != j) EXPECT_LE(std::abs(f(i, j)), kExactTol) << "core " << i << "," << j;
  }
  for (int64_t r = 0; r < g.nvirt; ++r) {
    EXPECT_LE(std::abs(f(nocc + r, nocc + r) - eVirt(r)), kExactTol) << "virt " << r;
    for (int64_t s = 0; s < g.nvirt; ++s)
      if (r != s)
        EXPECT_LE(std::abs(f(nocc + r, nocc + s)), kExactTol) << "virt " << r << "," << s;
  }
}

TEST_F(GradGoldCas44Tests, HeffRestrictsToTheGoldensOneElectronFields) {
  // §3.5: the golden's one-electron fields are RESTRICTIONS of the sidecar's
  // full-MO-range matrices, and the sidecar's job on that side is to carry the
  // same matrices whole. h1e is heff's active-active block, h1e_v_Si its
  // active-core block and h1e_v_Sir its virtual-core block. The one golden
  // field that is not a plain slice -- h1e_v_Sr, which subtracts an active
  // exchange sum -- is checked by rebuilding it from the sidecar's g_xa_aa.
  const GradGoldFile& g = *sidecar_;
  ASSERT_TRUE(golden_->require({"h1e", "h1e_v_Si", "h1e_v_Sir", "h1e_v_Sr"}).has_value());
  const Tensor& heff = g.get("heff");
  const int64_t nocc = g.ncore + g.nact;

  const Tensor& h1e = golden_->get("h1e");
  for (int64_t t = 0; t < g.nact; ++t)
    for (int64_t u = 0; u < g.nact; ++u)
      EXPECT_LE(std::abs(heff(g.ncore + t, g.ncore + u) - h1e(t, u)), kExactTol);

  const Tensor& hSi = golden_->get("h1e_v_Si");
  for (int64_t t = 0; t < g.nact; ++t)
    for (int64_t i = 0; i < g.ncore; ++i)
      EXPECT_LE(std::abs(heff(g.ncore + t, i) - hSi(t, i)), kExactTol);

  const Tensor& hSir = golden_->get("h1e_v_Sir");
  for (int64_t r = 0; r < g.nvirt; ++r)
    for (int64_t i = 0; i < g.ncore; ++i)
      EXPECT_LE(std::abs(heff(nocc + r, i) - hSir(r, i)), kExactTol);

  // h1e_v_Sr[r,t] = heff[r, t] - sum_b (r b | b t), and g_xa_aa[x,b,b,t] is
  // exactly that (x b | b t) with x taken over the virtual range.
  const Tensor& hSr = golden_->get("h1e_v_Sr");
  const Tensor& gAA = g.get("g_xa_aa");
  for (int64_t r = 0; r < g.nvirt; ++r) {
    for (int64_t t = 0; t < g.nact; ++t) {
      double exch = 0.0;
      for (int64_t b = 0; b < g.nact; ++b) exch += gAA(nocc + r, b, b, t);
      EXPECT_LE(std::abs(heff(nocc + r, g.ncore + t) - exch - hSr(r, t)), kExactTol)
          << "r=" << r << " t=" << t;
    }
  }
}

TEST_F(GradGoldCas44Tests, TheActiveIntegralBlockRestrictsToTheGoldensH2e) {
  // The same tie on the two-electron side: the golden's active Hamiltonian is
  // g_xa_aa's active slice, re-paired into the golden's index order
  // (h2e[t,u,b,w] = (t b | u w), generate_golden.py's _integral_blocks).
  const GradGoldFile& g = *sidecar_;
  ASSERT_TRUE(golden_->require({"h2e"}).has_value());
  const Tensor& h2e = golden_->get("h2e");
  const Tensor& gAA = g.get("g_xa_aa");
  double worst = 0.0;
  for (int64_t t = 0; t < g.nact; ++t)
    for (int64_t u = 0; u < g.nact; ++u)
      for (int64_t b = 0; b < g.nact; ++b)
        for (int64_t w = 0; w < g.nact; ++w)
          worst = std::max(worst, std::abs(gAA(g.ncore + t, b, u, w) - h2e(t, u, b, w)));
  EXPECT_LE(worst, kExactTol);
}

}  // namespace nevpt2::test::gradgold
