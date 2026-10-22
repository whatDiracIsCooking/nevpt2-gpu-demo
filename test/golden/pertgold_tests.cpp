// Suites for loadPertGold, the `*.pertgold` integral perturbation sidecar
// reader (#220).
//
// Host-only ON PURPOSE, like gradgold_tests.cpp: this is file parsing, so the
// binary is not REQUIRES_GPU and runs with no card visible. It checks the
// loader's logic -- a refusal's kind, a parsed header, an array's shape, a
// coordinate's slice -- never an energy.
//
//   PertGoldRoundTripTests   a sidecar this test writes byte by byte loads
//                            back with its header, every array value, and
//                            each coordinate's slice of a stacked array
//   PertGoldLoadErrorTests   a missing path, a bad magic, a truncated header,
//                            a partition that does not add up, no atom, a
//                            negative naux, an array of the wrong shape, an
//                            array that is not the format's -- and ANY ONE
//                            array missing, each of the twenty-two in turn:
//                            the whole sidecar refused, never read in part
//                            -- each a value-tier IO Error, never an abort
//   PertGoldCommittedTests   the committed golden/n2_ccpvdz_cas44_casscf
//                            [_df].pertgold: header and shapes, the pairing
//                            with its own golden and gradient sidecar (the
//                            frame bit for bit), and refused beside the
//                            CASCI golden's (another frame) and beside the
//                            other calculation's (fitted against exact)
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

#if !defined(NEVPT2_TEST_PERTGOLD_CASSCF) || !defined(NEVPT2_TEST_GOLDEN_CASSCF) || \
    !defined(NEVPT2_TEST_GRADGOLD_CASSCF)
#error "NEVPT2_TEST_{PERTGOLD,GOLDEN,GRADGOLD}_CASSCF must be defined by test/golden/CMakeLists.txt"
#endif
#if !defined(NEVPT2_TEST_PERTGOLD_CASSCF_DF) || !defined(NEVPT2_TEST_GOLDEN_CASSCF_DF) || \
    !defined(NEVPT2_TEST_GRADGOLD_CASSCF_DF)
#error "NEVPT2_TEST_{PERTGOLD,GOLDEN,GRADGOLD}_CASSCF_DF must be defined by test/golden/CMakeLists.txt"
#endif
#if !defined(NEVPT2_TEST_GOLDEN_CAS44) || !defined(NEVPT2_TEST_GRADGOLD_CAS44)
#error "NEVPT2_TEST_{GOLDEN,GRADGOLD}_CAS44 must be defined by test/golden/CMakeLists.txt"
#endif

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::pertgold {

constexpr std::string_view kPertCasscf = NEVPT2_TEST_PERTGOLD_CASSCF;
constexpr std::string_view kGoldenCasscf = NEVPT2_TEST_GOLDEN_CASSCF;
constexpr std::string_view kGradCasscf = NEVPT2_TEST_GRADGOLD_CASSCF;
constexpr std::string_view kPertCasscfDf = NEVPT2_TEST_PERTGOLD_CASSCF_DF;
constexpr std::string_view kGoldenCasscfDf = NEVPT2_TEST_GOLDEN_CASSCF_DF;
constexpr std::string_view kGradCasscfDf = NEVPT2_TEST_GRADGOLD_CASSCF_DF;
constexpr std::string_view kGoldenCas44 = NEVPT2_TEST_GOLDEN_CAS44;
constexpr std::string_view kGradCas44 = NEVPT2_TEST_GRADGOLD_CAS44;

// The committed N2 CAS(4,4) partition: the case's definition, cc-pVDZ on N2
// (docs/reference-data.md, "The small cases"), and its fitting basis.
constexpr int64_t kNatm = 2, kNao = 28, kNmo = 28, kNcore = 5, kNact = 4, kNvirt = 19;
constexpr int64_t kNaux = 140;

// A written double read back: the f64 round trip, nothing looser.
constexpr double kExactTol = 1e-12;

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

// A miniature partition, every array present, written by name.
struct Mini {
  int64_t natm = 1, nao = 6, ncore = 1, nact = 2, nvirt = 3, naux = 0;
  int64_t nmo() const { return ncore + nact + nvirt; }
  int64_t ncoord() const { return 3 * natm; }
};

// Every array of the format and its shape for `m` (golden_format.py's
// PERTGOLD_ARRAYS), in file order.
std::vector<std::pair<std::string, std::vector<int64_t>>> arraySpec(const Mini& m) {
  const int64_t c = m.ncore, a = m.nact, v = m.nvirt, n = m.nmo(), k = m.ncoord();
  const auto ext = [&](const char s) { return s == 'c' ? c : (s == 'a' ? a : v); };
  std::vector<std::pair<std::string, std::vector<int64_t>>> out{{"mo_coeff", {m.nao, n}}};
  for (const char* name : {"h", "heff", "f"}) out.push_back({name, {k, n, n}});
  for (const char* name : {"g_xa_aa", "g_xv_aa", "g_xc_aa", "g_xa_va", "g_xv_va", "g_xc_va",
                           "g_xa_ca", "g_xv_ca", "g_xc_ca", "g_xa_cv", "g_xv_cv", "g_xc_cv",
                           "g_xa_cc", "g_xv_cc"}) {
    const std::string_view s(name);
    out.push_back({name, {k, n, ext(s[3]), ext(s[5]), ext(s[6])}});
  }
  out.push_back({"g_xc_cc_j", {k, n, c, c}});
  out.push_back({"g_xc_cc_k", {k, n, c, c}});
  out.push_back({"g_xa_vv", {k, n, a, v}});
  out.push_back({"g_xc_vv", {k, n, c, v}});
  return out;
}

int64_t product(const std::vector<int64_t>& dims) {
  int64_t p = 1;
  for (const int64_t d : dims) p *= d;
  return p;
}

// Deterministic values with a full mantissa: a round trip that compares
// zeros would check nothing.
double fill(const int64_t i) { return std::sin(0.31 * static_cast<double>(i) + 0.2) * 2.5; }

// The sidecar's bytes: `skip` leaves that array out, `extra` appends one the
// format does not have, `reshape` gives that array one extent too many.
std::string sidecarBytes(const Mini& m, const std::string_view skip = {},
                         const std::string_view extra = {},
                         const std::string_view reshape = {},
                         const std::string_view magic = "NEVPT2I1") {
  std::string out(magic);
  for (const int64_t v : {m.natm, m.nao, m.nmo(), m.ncore, m.nact, m.nvirt, m.naux})
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
    for (std::size_t i = 0; i < data.size(); ++i)
      data[i] = fill(seed + static_cast<int64_t>(i));
    appendArray(out, name, dims, data);
    seed += 1000;
  }
  if (!extra.empty()) appendArray(out, extra, {1}, {1.0});
  return out;
}

class PertGoldTempDir : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::random_device rd;
    dir_ = std::filesystem::temp_directory_path() /
           std::format("nevpt2_pertgold_tests_{}_{}_{:08x}", info->test_suite_name(),
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

void expectIoError(const Result<PertGoldFile>& r, const std::string_view fragment) {
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::IO);
  EXPECT_NE(r.error().message.find(fragment), std::string::npos) << r.error().message;
}

// --- PertGoldRoundTripTests ----------------------------------------------------

using PertGoldRoundTripTests = PertGoldTempDir;

TEST_F(PertGoldRoundTripTests, HeaderEveryArrayAndEveryCoordinateSurvive) {
  const Mini m;
  const Result<PertGoldFile> r = loadPertGold(write("mini.pertgold", sidecarBytes(m)));
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->natm, m.natm);
  EXPECT_EQ(r->ncoord(), 3);
  EXPECT_EQ(r->nao, m.nao);
  EXPECT_EQ(r->nmo, m.nmo());
  EXPECT_EQ(r->ncore, m.ncore);
  EXPECT_EQ(r->nact, m.nact);
  EXPECT_EQ(r->nvirt, m.nvirt);
  EXPECT_FALSE(r->densityFitted());
  EXPECT_EQ(std::ssize(r->arrays), 22);
  EXPECT_EQ(std::ssize(PertGoldFile::inventoryNames()), 21);
  int64_t seed = 0;
  for (const auto& [name, dims] : arraySpec(m)) {
    const Tensor& t = r->get(name);
    ASSERT_EQ(t.dims(), dims) << name;
    double worst = 0.0;
    for (int64_t i = 0; i < t.size(); ++i)
      worst = std::max(worst, std::abs(t.flat(i) - fill(seed + i)));
    EXPECT_LE(worst, kExactTol) << name;
    if (name != "mo_coeff") {
      // Each coordinate's slice is the stacked array's k-th block, its
      // leading axis dropped.
      const int64_t per = t.size() / m.ncoord();
      for (int64_t k = 0; k < m.ncoord(); ++k) {
        const Tensor s = r->coordinate(name, k);
        EXPECT_EQ(s.dims(), std::vector<int64_t>(dims.begin() + 1, dims.end())) << name;
        double w = 0.0;
        for (int64_t i = 0; i < s.size(); ++i)
          w = std::max(w, std::abs(s.flat(i) - fill(seed + k * per + i)));
        EXPECT_LE(w, kExactTol) << name << " coordinate " << k;
      }
    }
    seed += 1000;
  }
}

TEST_F(PertGoldRoundTripTests, TheFittedFlavourReadsItsNaux) {
  Mini m;
  m.naux = 9;
  const Result<PertGoldFile> r = loadPertGold(write("mini_df.pertgold", sidecarBytes(m)));
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_TRUE(r->densityFitted());
  EXPECT_EQ(r->naux, 9);
}

// --- PertGoldLoadErrorTests ----------------------------------------------------

using PertGoldLoadErrorTests = PertGoldTempDir;

TEST_F(PertGoldLoadErrorTests, AMissingPathIsAnIoError) {
  expectIoError(loadPertGold((dir_ / "nope.pertgold").string()), "could not open");
}

TEST_F(PertGoldLoadErrorTests, ABadMagicIsAnIoError) {
  const Mini m;
  expectIoError(loadPertGold(write("magic.pertgold", sidecarBytes(m, {}, {}, {}, "NEVPT2D1"))),
                "bad magic");
}

TEST_F(PertGoldLoadErrorTests, ATruncatedHeaderIsAnIoError) {
  const std::string bytes = sidecarBytes(Mini{}).substr(0, 8 + 3 * 4);
  expectIoError(loadPertGold(write("short.pertgold", bytes)), "unexpected EOF");
}

TEST_F(PertGoldLoadErrorTests, APartitionThatDoesNotAddUpIsAnIoError) {
  const Mini m;
  std::string bytes = sidecarBytes(m);
  const std::int32_t wrong = static_cast<std::int32_t>(m.nmo() + 1);
  std::memcpy(bytes.data() + 8 + 2 * 4, &wrong, 4);  // nmo
  expectIoError(loadPertGold(write("partition.pertgold", bytes)), "nmo");
}

TEST_F(PertGoldLoadErrorTests, NoAtomOrANegativeNauxIsAnIoError) {
  Mini none;
  none.natm = 0;
  expectIoError(loadPertGold(write("natm.pertgold", sidecarBytes(none))), "usable partition");
  Mini neg;
  neg.naux = -1;
  expectIoError(loadPertGold(write("naux.pertgold", sidecarBytes(neg))), "usable partition");
}

TEST_F(PertGoldLoadErrorTests, AnArrayOfTheWrongShapeIsAnIoError) {
  const Mini m;
  expectIoError(loadPertGold(write("shape.pertgold", sidecarBytes(m, {}, {}, "g_xv_cv"))),
                "g_xv_cv has shape");
  expectIoError(loadPertGold(write("mo.pertgold", sidecarBytes(m, {}, {}, "mo_coeff"))),
                "mo_coeff has shape");
}

TEST_F(PertGoldLoadErrorTests, AnArrayThatIsNotTheFormatsIsAnIoError) {
  expectIoError(loadPertGold(write("extra.pertgold", sidecarBytes(Mini{}, {}, "B_cc"))),
                "'B_cc' is not one of the format's");
}

TEST_F(PertGoldLoadErrorTests, AnyOneMissingArrayRefusesTheWholeSidecar) {
  // Each of the twenty-two in turn: a perturbation with one input held fixed
  // would pass every identity its two modes share, so it is never read.
  const Mini m;
  for (const auto& [name, dims] : arraySpec(m)) {
    SCOPED_TRACE(name);
    expectIoError(loadPertGold(write(name + ".pertgold", sidecarBytes(m, name))),
                  std::format("missing required array '{}'", name));
  }
}

// --- PertGoldCommittedTests ----------------------------------------------------

TEST(PertGoldCommittedTests, TheCommittedSidecarsCarryThePartitionAndShapes) {
  for (const auto& [path, naux] :
       {std::pair{kPertCasscf, int64_t{0}}, std::pair{kPertCasscfDf, kNaux}}) {
    SCOPED_TRACE(path);
    const Result<PertGoldFile> r = loadPertGold(std::string(path));
    ASSERT_TRUE(r.has_value()) << r.error().message;
    EXPECT_EQ(r->natm, kNatm);
    EXPECT_EQ(r->nao, kNao);
    EXPECT_EQ(r->nmo, kNmo);
    EXPECT_EQ(r->ncore, kNcore);
    EXPECT_EQ(r->nact, kNact);
    EXPECT_EQ(r->nvirt, kNvirt);
    EXPECT_EQ(r->naux, naux);
    EXPECT_EQ(r->get("g_xv_cv").dims(),
              (std::vector<int64_t>{3 * kNatm, kNmo, kNvirt, kNcore, kNvirt}));
    EXPECT_EQ(r->coordinate("f", 2).dims(), (std::vector<int64_t>{kNmo, kNmo}));
  }
}

TEST(PertGoldCommittedTests, EachPairsWithItsOwnGoldenAndSidecarOnly) {
  const auto load = [](const std::string_view p, const std::string_view g,
                       const std::string_view s) {
    Result<PertGoldFile> pert = loadPertGold(std::string(p));
    Result<GoldenFile> golden = loadGolden(std::string(g));
    Result<GradGoldFile> side = loadGradGold(std::string(s));
    EXPECT_TRUE(pert.has_value() && golden.has_value() && side.has_value());
    return std::tuple{*std::move(pert), *std::move(golden), *std::move(side)};
  };
  const auto [pert, golden, side] = load(kPertCasscf, kGoldenCasscf, kGradCasscf);
  const auto [pertDf, goldenDf, sideDf] = load(kPertCasscfDf, kGoldenCasscfDf, kGradCasscfDf);
  EXPECT_TRUE(pert.checkAgainst(golden, side).has_value());
  EXPECT_TRUE(pertDf.checkAgainst(goldenDf, sideDf).has_value());
  // Another frame: the CASCI state's orbitals are RHF's, not the CASSCF ones.
  const Result<GoldenFile> casci = loadGolden(std::string(kGoldenCas44));
  const Result<GradGoldFile> casciSide = loadGradGold(std::string(kGradCas44));
  ASSERT_TRUE(casci.has_value() && casciSide.has_value());
  const Status frame = pert.checkAgainst(*casci, *casciSide);
  ASSERT_FALSE(frame.has_value());
  EXPECT_EQ(frame.error().kind, ErrorKind::IO);
  EXPECT_NE(frame.error().message.find("bit for bit"), std::string::npos)
      << frame.error().message;
  // The other calculation: fitted against exact, both ways.
  for (const Status st : {pert.checkAgainst(goldenDf, sideDf), pertDf.checkAgainst(golden, side)}) {
    ASSERT_FALSE(st.has_value());
    EXPECT_EQ(st.error().kind, ErrorKind::IO);
    EXPECT_NE(st.error().message.find("one calculation"), std::string::npos)
        << st.error().message;
  }
}

}  // namespace nevpt2::test::pertgold
