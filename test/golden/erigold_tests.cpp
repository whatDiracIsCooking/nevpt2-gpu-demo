// Suites for loadEriGold, the `*.erigold` full MO-integral tensor sidecar
// reader (#180).
//
// Host-only ON PURPOSE, like pertgold_tests.cpp: this is file parsing, so the
// binary is not REQUIRES_GPU and runs with no card visible. It checks the
// loader's logic -- a refusal's kind, a parsed header, an array's shape --
// never an energy.
//
//   EriGoldRoundTripTests   a sidecar this test writes byte by byte loads back
//                           with its header and every value
//   EriGoldLoadErrorTests   a missing path, a bad magic, a truncated header,
//                           a partition that does not add up, an array of the
//                           wrong shape, an array that is not the format's,
//                           and either array missing -- each a value-tier IO
//                           Error, never an abort
//   EriGoldCommittedTests   the committed golden/n2_ccpvdz_cas44.erigold:
//                           header and shapes, its tensor's (x a | a a) rows
//                           against the gradient sidecar's g_xa_aa, the
//                           pairing with its own golden and gradient sidecar
//                           (the frame bit for bit), and refused beside the
//                           CASSCF-orbital pair (another frame) and the
//                           density-fitted one (another calculation)
//
// The malformed and synthetic files are written to a per-test temp directory,
// removed in TearDown. The committed files are only ever read; their paths
// come from CMake, not from this source.
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.golden;

#if !defined(NEVPT2_TEST_ERIGOLD_CAS44) || !defined(NEVPT2_TEST_GOLDEN_CAS44) || \
    !defined(NEVPT2_TEST_GRADGOLD_CAS44)
#error "NEVPT2_TEST_{ERIGOLD,GOLDEN,GRADGOLD}_CAS44 must be defined by test/golden/CMakeLists.txt"
#endif
#if !defined(NEVPT2_TEST_GOLDEN_CASSCF) || !defined(NEVPT2_TEST_GRADGOLD_CASSCF) || \
    !defined(NEVPT2_TEST_GOLDEN_CAS44_DF) || !defined(NEVPT2_TEST_GRADGOLD_CAS44_DF)
#error "NEVPT2_TEST_{GOLDEN,GRADGOLD}_{CASSCF,CAS44_DF} must be defined by test/golden/CMakeLists.txt"
#endif

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::erigold {

constexpr std::string_view kEriCas44 = NEVPT2_TEST_ERIGOLD_CAS44;
constexpr std::string_view kGoldenCas44 = NEVPT2_TEST_GOLDEN_CAS44;
constexpr std::string_view kGradCas44 = NEVPT2_TEST_GRADGOLD_CAS44;
constexpr std::string_view kGoldenCasscf = NEVPT2_TEST_GOLDEN_CASSCF;
constexpr std::string_view kGradCasscf = NEVPT2_TEST_GRADGOLD_CASSCF;
constexpr std::string_view kGoldenCas44Df = NEVPT2_TEST_GOLDEN_CAS44_DF;
constexpr std::string_view kGradCas44Df = NEVPT2_TEST_GRADGOLD_CAS44_DF;

// The committed N2 CAS(4,4) partition: cc-pVDZ on N2 (docs/reference-data.md,
// "The small cases").
constexpr int64_t kNao = 28, kNmo = 28, kNcore = 5, kNact = 4, kNvirt = 19;

// A written double read back: the f64 round trip, nothing looser.
constexpr double kExactTol = 1e-12;
// The committed tensor's (x a | a a) rows against the gradient sidecar's
// g_xa_aa: two transforms of the same AO integrals (generate_golden.py's
// _full_eri holds the two to 1e-12 relative before it writes).
constexpr double kOverlapTol = 1e-10;

void appendRaw(std::string& out, const void* p, const std::size_t n) {
  out.append(static_cast<const char*>(p), n);
}
void appendI32(std::string& out, const std::int32_t v) { appendRaw(out, &v, 4); }
void appendI64(std::string& out, const std::int64_t v) { appendRaw(out, &v, 8); }

// One named array record (reference_data/golden_format.py's write_arrays).
void appendArray(std::string& out, const std::string_view name, const std::vector<int64_t>& dims,
                 const std::vector<double>& data) {
  appendI32(out, static_cast<std::int32_t>(name.size()));
  out.append(name);
  appendI32(out, static_cast<std::int32_t>(dims.size()));
  for (const int64_t d : dims) appendI64(out, d);
  appendRaw(out, data.data(), data.size() * sizeof(double));
}

// A miniature partition.
struct Mini {
  int64_t nao = 6, ncore = 1, nact = 2, nvirt = 2;
  int64_t nmo() const { return ncore + nact + nvirt; }
};

// The format's two arrays and their shapes for `m` (ERIGOLD_ARRAYS), in order.
std::vector<std::pair<std::string, std::vector<int64_t>>> arraySpec(const Mini& m) {
  const int64_t n = m.nmo();
  return {{"mo_coeff", {m.nao, n}}, {"eri", {n, n, n, n}}};
}

int64_t product(const std::vector<int64_t>& dims) {
  int64_t p = 1;
  for (const int64_t d : dims) p *= d;
  return p;
}

// Deterministic values with a full mantissa.
double fill(const int64_t i) { return std::sin(0.37 * static_cast<double>(i) + 0.1) * 1.5; }

// The sidecar's bytes: `skip` leaves that array out, `extra` appends one the
// format does not have, `reshape` gives that array one extent too many.
std::string sidecarBytes(const Mini& m, const std::string_view skip = {},
                         const std::string_view extra = {},
                         const std::string_view reshape = {},
                         const std::string_view magic = "NEVPT2E1") {
  std::string out(magic);
  for (const int64_t v : {m.nao, m.nmo(), m.ncore, m.nact, m.nvirt})
    appendI32(out, static_cast<std::int32_t>(v));
  const auto spec = arraySpec(m);
  std::int32_t count = 0;
  for (const auto& [name, dims] : spec) count += name != skip;
  if (!extra.empty()) ++count;
  appendI32(out, count);
  int64_t seed = 0;
  for (auto [name, dims] : spec) {
    if (name == skip) continue;
    if (name == reshape) dims.push_back(1);
    std::vector<double> data(static_cast<std::size_t>(product(dims)));
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = fill(seed + static_cast<int64_t>(i));
    appendArray(out, name, dims, data);
    seed += 1000;
  }
  if (!extra.empty()) appendArray(out, extra, {1}, {1.0});
  return out;
}

class EriGoldTempDir : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::random_device rd;
    dir_ = std::filesystem::temp_directory_path() /
           std::format("nevpt2_erigold_tests_{}_{}_{:08x}", info->test_suite_name(),
                       info->name(), rd());
    std::filesystem::create_directories(dir_);
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
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

void expectIoError(const Result<EriGoldFile>& r, const std::string_view fragment) {
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::IO);
  EXPECT_NE(r.error().message.find(fragment), std::string::npos) << r.error().message;
}

// --- EriGoldRoundTripTests -----------------------------------------------------

using EriGoldRoundTripTests = EriGoldTempDir;

TEST_F(EriGoldRoundTripTests, HeaderAndEveryValueSurvive) {
  const Mini m;
  const Result<EriGoldFile> r = loadEriGold(write("mini.erigold", sidecarBytes(m)));
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->nao, m.nao);
  EXPECT_EQ(r->nmo, m.nmo());
  EXPECT_EQ(r->ncore, m.ncore);
  EXPECT_EQ(r->nact, m.nact);
  EXPECT_EQ(r->nvirt, m.nvirt);
  EXPECT_EQ(std::ssize(r->arrays), 2);
  int64_t seed = 0;
  for (const auto& [name, dims] : arraySpec(m)) {
    const Tensor& t = r->get(name);
    ASSERT_EQ(t.dims(), dims) << name;
    double worst = 0.0;
    for (int64_t i = 0; i < t.size(); ++i)
      worst = std::max(worst, std::abs(t.flat(i) - fill(seed + i)));
    EXPECT_LE(worst, kExactTol) << name;
    seed += 1000;
  }
}

// --- EriGoldLoadErrorTests -----------------------------------------------------

using EriGoldLoadErrorTests = EriGoldTempDir;

TEST_F(EriGoldLoadErrorTests, AMissingPathIsAnIoError) {
  expectIoError(loadEriGold((dir_ / "nope.erigold").string()), "could not open");
}

TEST_F(EriGoldLoadErrorTests, ABadMagicIsAnIoError) {
  expectIoError(loadEriGold(write("magic.erigold", sidecarBytes(Mini{}, {}, {}, {}, "NEVPT2I1"))),
                "bad magic");
}

TEST_F(EriGoldLoadErrorTests, ATruncatedHeaderIsAnIoError) {
  const std::string bytes = sidecarBytes(Mini{}).substr(0, 8 + 2 * 4);
  expectIoError(loadEriGold(write("short.erigold", bytes)), "unexpected EOF");
}

TEST_F(EriGoldLoadErrorTests, APartitionThatDoesNotAddUpIsAnIoError) {
  const Mini m;
  std::string bytes = sidecarBytes(m);
  const std::int32_t wrong = static_cast<std::int32_t>(m.nmo() + 1);
  std::memcpy(bytes.data() + 8 + 1 * 4, &wrong, 4);  // nmo
  expectIoError(loadEriGold(write("partition.erigold", bytes)), "nmo");
  Mini noAct;
  noAct.nact = 0;
  expectIoError(loadEriGold(write("nact.erigold", sidecarBytes(noAct))), "usable partition");
}

TEST_F(EriGoldLoadErrorTests, AnArrayOfTheWrongShapeIsAnIoError) {
  const Mini m;
  expectIoError(loadEriGold(write("shape.erigold", sidecarBytes(m, {}, {}, "eri"))),
                "eri has shape");
  expectIoError(loadEriGold(write("mo.erigold", sidecarBytes(m, {}, {}, "mo_coeff"))),
                "mo_coeff has shape");
}

TEST_F(EriGoldLoadErrorTests, AnArrayThatIsNotTheFormatsIsAnIoError) {
  expectIoError(loadEriGold(write("extra.erigold", sidecarBytes(Mini{}, {}, "h"))),
                "'h' is not one of the format's");
}

TEST_F(EriGoldLoadErrorTests, EitherMissingArrayRefusesTheWholeSidecar) {
  const Mini m;
  for (const auto& [name, dims] : arraySpec(m)) {
    SCOPED_TRACE(name);
    expectIoError(loadEriGold(write(name + ".erigold", sidecarBytes(m, name))),
                  std::format("missing required array '{}'", name));
  }
}

// --- EriGoldCommittedTests -----------------------------------------------------

TEST(EriGoldCommittedTests, TheCommittedTensorCarriesThePartitionAndItsSidecarsRows) {
  const Result<EriGoldFile> r = loadEriGold(std::string(kEriCas44));
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->nao, kNao);
  EXPECT_EQ(r->nmo, kNmo);
  EXPECT_EQ(r->ncore, kNcore);
  EXPECT_EQ(r->nact, kNact);
  EXPECT_EQ(r->nvirt, kNvirt);
  const Tensor& eri = r->get("eri");
  ASSERT_EQ(eri.dims(), (std::vector<int64_t>{kNmo, kNmo, kNmo, kNmo}));
  const Result<GradGoldFile> side = loadGradGold(std::string(kGradCas44));
  ASSERT_TRUE(side.has_value()) << side.error().message;
  const Tensor& row = side->get("g_xa_aa");  // (x t | u w)
  double worst = 0.0;
  for (int64_t x = 0; x < kNmo; ++x)
    for (int64_t t = 0; t < kNact; ++t)
      for (int64_t u = 0; u < kNact; ++u)
        for (int64_t w = 0; w < kNact; ++w)
          worst = std::max(worst, std::abs(eri(x, kNcore + t, kNcore + u, kNcore + w) -
                                           row(x, t, u, w)));
  EXPECT_LE(worst, kOverlapTol);
}

TEST(EriGoldCommittedTests, ItPairsWithItsOwnGoldenAndSidecarOnly) {
  const Result<EriGoldFile> eri = loadEriGold(std::string(kEriCas44));
  ASSERT_TRUE(eri.has_value()) << eri.error().message;
  const auto pairOf = [](const std::string_view g, const std::string_view s) {
    Result<GoldenFile> golden = loadGolden(std::string(g));
    Result<GradGoldFile> side = loadGradGold(std::string(s));
    EXPECT_TRUE(golden.has_value() && side.has_value());
    return std::pair{*std::move(golden), *std::move(side)};
  };
  const auto [golden, side] = pairOf(kGoldenCas44, kGradCas44);
  EXPECT_TRUE(eri->checkAgainst(golden, side).has_value());
  // Another frame: the CASSCF-orbital state of the same partition.
  const auto [casscf, casscfSide] = pairOf(kGoldenCasscf, kGradCasscf);
  const Status frame = eri->checkAgainst(casscf, casscfSide);
  ASSERT_FALSE(frame.has_value());
  EXPECT_EQ(frame.error().kind, ErrorKind::IO);
  EXPECT_NE(frame.error().message.find("bit for bit"), std::string::npos)
      << frame.error().message;
  // The fitted calculation: its B is already the whole tensor.
  const auto [fitted, fittedSide] = pairOf(kGoldenCas44Df, kGradCas44Df);
  const Status df = eri->checkAgainst(fitted, fittedSide);
  ASSERT_FALSE(df.has_value());
  EXPECT_EQ(df.error().kind, ErrorKind::IO);
  EXPECT_NE(df.error().message.find("density-fitted"), std::string::npos) << df.error().message;
}

}  // namespace nevpt2::test::erigold
