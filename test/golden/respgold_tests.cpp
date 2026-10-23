// Suites for loadRespGold, the `*.respgold` CASSCF response sidecar reader
// (#219's file, read on the GPU side by #222), and for writeParHess /
// loadParHess, the `*.parhess` parameter Hessian sidecar (#222) -- the one
// format in golden/'s family the C++ side writes.
//
// Host-only ON PURPOSE, like erigold_tests.cpp: this is file parsing and
// writing, so the binary is not REQUIRES_GPU and runs with no card visible.
// It checks the reader's and writer's logic -- a refusal's kind, a parsed
// header, an array's shape -- never an energy.
//
//   RespGoldLoadErrorTests   a missing path, a bad magic, a partition that
//                            does not add up, an array of the wrong shape, an
//                            array that is not the format's and each array
//                            missing -- each a value-tier IO Error
//   RespGoldCommittedTests   the committed N2 CAS(4,4) and HSOH CAS(6,6)
//                            response sidecars: header, each coordinate's
//                            kappa antisymmetric and zero on its three
//                            diagonal blocks, each CI response orthogonal to
//                            ci; paired with its own golden and gradient
//                            sidecar, and refused beside the CASCI state (its
//                            ci is not the responses')
//   ParHessRoundTripTests    writeParHess's file loads back bit for bit, and
//                            its bytes are the format's (magic, header, the
//                            five arrays in order)
//   ParHessLoadErrorTests    a bad magic, coordinates out of order or past
//                            3 natm, a matrix of the wrong shape and a missing
//                            one -- each an IO Error
//
// The synthetic files are written to a per-test temp directory, removed in
// TearDown. The committed files are only ever read; their paths come from
// CMake, not from this source.
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.golden;

#if !defined(NEVPT2_TEST_RESPGOLD_CASSCF) || !defined(NEVPT2_TEST_GOLDEN_CASSCF) || \
    !defined(NEVPT2_TEST_GRADGOLD_CASSCF) || !defined(NEVPT2_TEST_GOLDEN_CAS44) ||   \
    !defined(NEVPT2_TEST_RESPGOLD_HSOH) || !defined(NEVPT2_TEST_GOLDEN_HSOH) ||       \
    !defined(NEVPT2_TEST_GRADGOLD_HSOH)
#error "the NEVPT2_TEST_* paths must be defined by test/golden/CMakeLists.txt"
#endif

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::respgold {

constexpr std::string_view kRespCasscf = NEVPT2_TEST_RESPGOLD_CASSCF;
constexpr std::string_view kGoldenCasscf = NEVPT2_TEST_GOLDEN_CASSCF;
constexpr std::string_view kGradCasscf = NEVPT2_TEST_GRADGOLD_CASSCF;
constexpr std::string_view kGoldenCas44 = NEVPT2_TEST_GOLDEN_CAS44;
constexpr std::string_view kRespHsoh = NEVPT2_TEST_RESPGOLD_HSOH;
constexpr std::string_view kGoldenHsoh = NEVPT2_TEST_GOLDEN_HSOH;
constexpr std::string_view kGradHsoh = NEVPT2_TEST_GRADGOLD_HSOH;

// A response orthogonal to ci: the generator's projection, round-off only.
constexpr double kOrthoTol = 1e-12;

void appendRaw(std::string& out, const void* p, const std::size_t n) {
  out.append(static_cast<const char*>(p), n);
}
void appendI32(std::string& out, const std::int32_t v) { appendRaw(out, &v, 4); }
void appendI64(std::string& out, const std::int64_t v) { appendRaw(out, &v, 8); }
void appendF64(std::string& out, const double v) { appendRaw(out, &v, 8); }

// One named array record (reference_data/golden_format.py's write_arrays).
void appendArray(std::string& out, const std::string_view name, const std::vector<int64_t>& dims,
                 const std::vector<double>& data) {
  appendI32(out, static_cast<std::int32_t>(name.size()));
  out.append(name);
  appendI32(out, static_cast<std::int32_t>(dims.size()));
  for (const int64_t d : dims) appendI64(out, d);
  appendRaw(out, data.data(), data.size() * sizeof(double));
}

int64_t product(const std::vector<int64_t>& dims) {
  int64_t p = 1;
  for (const int64_t d : dims) p *= d;
  return p;
}

// A miniature response sidecar: one atom, 5 orbitals, a (2, 2) CI vector.
struct Mini {
  int64_t natm = 1, nao = 6, ncore = 1, nact = 2, nvirt = 2, nelecA = 1, nelecB = 1, ndet = 4;
  int64_t nmo() const { return ncore + nact + nvirt; }
  int64_t ncoord() const { return 3 * natm; }
};

std::vector<std::pair<std::string, std::vector<int64_t>>> respSpec(const Mini& m) {
  const int64_t n = m.nmo(), c = m.ncoord();
  return {{"mo_coeff", {m.nao, n}}, {"ci", {2, 2}},    {"kappa", {c, n, n}},
          {"ci_response", {c, m.ndet}}, {"hessian", {c, c}}, {"l_rr", {c, c}},
          {"gradient", {c}},           {"kept", {c}}};
}

std::string respBytes(const Mini& m, const std::string_view skip = {},
                      const std::string_view extra = {}, const std::string_view reshape = {},
                      const std::string_view magic = "NEVPT2R1") {
  std::string out(magic);
  for (const int64_t v : {m.natm, m.nao, m.nmo(), m.ncore, m.nact, m.nvirt, m.nelecA, m.nelecB})
    appendI32(out, static_cast<std::int32_t>(v));
  appendI64(out, m.ndet);
  appendF64(out, -1.25);
  const auto spec = respSpec(m);
  std::int32_t count = 0;
  for (const auto& [name, dims] : spec) count += name != skip;
  if (!extra.empty()) ++count;
  appendI32(out, count);
  for (auto [name, dims] : spec) {
    if (name == skip) continue;
    if (name == reshape) dims.push_back(1);
    std::vector<double> data(static_cast<std::size_t>(product(dims)));
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = 0.25 * static_cast<double>(i);
    appendArray(out, name, dims, data);
  }
  if (!extra.empty()) appendArray(out, extra, {1}, {1.0});
  return out;
}

class GoldenTempDir : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::random_device rd;
    dir_ = std::filesystem::temp_directory_path() /
           std::format("nevpt2_respgold_tests_{}_{}_{:08x}", info->test_suite_name(),
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
  std::string read(const std::string& path) const {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  }
  std::filesystem::path dir_;
};

template <class T>
void expectIoError(const Result<T>& r, const std::string_view fragment) {
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, ErrorKind::IO);
  EXPECT_NE(r.error().message.find(fragment), std::string::npos) << r.error().message;
}

// --- RespGoldLoadErrorTests -------------------------------------------------------

using RespGoldLoadErrorTests = GoldenTempDir;

TEST_F(RespGoldLoadErrorTests, TheMiniatureLoads) {
  const Mini m;
  const Result<RespGoldFile> r = loadRespGold(write("mini.respgold", respBytes(m)));
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->natm, 1);
  EXPECT_EQ(r->nmo, m.nmo());
  EXPECT_EQ(r->ndet, m.ndet);
  EXPECT_EQ(r->eCasscf, -1.25);
  EXPECT_EQ(r->kappa(2).dims(), (std::vector<int64_t>{m.nmo(), m.nmo()}));
  EXPECT_EQ(r->ciResponse(1).dims(), (std::vector<int64_t>{2, 2}));
  // Coordinate 1's CI response is the second row of ci_response.
  EXPECT_EQ(r->ciResponse(1).flat(0), 0.25 * static_cast<double>(m.ndet));
}

TEST_F(RespGoldLoadErrorTests, AMissingPathIsAnIoError) {
  expectIoError(loadRespGold((dir_ / "nope.respgold").string()), "could not open");
}

TEST_F(RespGoldLoadErrorTests, ABadMagicIsAnIoError) {
  expectIoError(loadRespGold(write("magic.respgold", respBytes(Mini{}, {}, {}, {}, "NEVPT2E1"))),
                "bad magic");
}

TEST_F(RespGoldLoadErrorTests, APartitionThatDoesNotAddUpIsAnIoError) {
  const Mini m;
  std::string bytes = respBytes(m);
  const std::int32_t wrong = static_cast<std::int32_t>(m.nmo() + 1);
  std::memcpy(bytes.data() + 8 + 2 * 4, &wrong, 4);  // nmo
  expectIoError(loadRespGold(write("partition.respgold", bytes)), "nmo");
}

TEST_F(RespGoldLoadErrorTests, AnArrayOfTheWrongShapeIsAnIoError) {
  expectIoError(loadRespGold(write("shape.respgold", respBytes(Mini{}, {}, {}, "kappa"))),
                "kappa has shape");
}

TEST_F(RespGoldLoadErrorTests, AnArrayThatIsNotTheFormatsIsAnIoError) {
  expectIoError(loadRespGold(write("extra.respgold", respBytes(Mini{}, {}, "eri"))),
                "'eri' is not one of the format's");
}

TEST_F(RespGoldLoadErrorTests, EachMissingArrayRefusesTheWholeSidecar) {
  const Mini m;
  for (const auto& [name, dims] : respSpec(m)) {
    SCOPED_TRACE(name);
    expectIoError(loadRespGold(write(name + ".respgold", respBytes(m, name))),
                  std::format("missing required array '{}'", name));
  }
}

// --- RespGoldCommittedTests -------------------------------------------------------

void checkCommitted(const std::string_view resp, const std::string_view golden,
                    const std::string_view grad) {
  const Result<RespGoldFile> r = loadRespGold(std::string(resp));
  ASSERT_TRUE(r.has_value()) << r.error().message;
  const Result<GoldenFile> g = loadGolden(std::string(golden));
  const Result<GradGoldFile> s = loadGradGold(std::string(grad));
  ASSERT_TRUE(g.has_value() && s.has_value());
  EXPECT_TRUE(r->checkAgainst(*g, *s).has_value());
  const Tensor& ci = g->get("ci");
  const auto space = [&](int64_t x) { return x < r->ncore ? 0 : (x < r->ncore + r->nact ? 1 : 2); };
  for (int64_t k = 0; k < r->ncoord(); ++k) {
    const Tensor kappa = r->kappa(k);
    for (int64_t p = 0; p < r->nmo; ++p)
      for (int64_t q = 0; q < r->nmo; ++q) {
        EXPECT_EQ(kappa(p, q), -kappa(q, p)) << k << " " << p << " " << q;
        if (space(p) == space(q)) EXPECT_EQ(kappa(p, q), 0.0) << k << " " << p << " " << q;
      }
    const Tensor c = r->ciResponse(k);
    double overlap = 0.0;
    for (int64_t i = 0; i < c.size(); ++i) overlap += c.flat(i) * ci.flat(i);
    EXPECT_LE(std::abs(overlap), kOrthoTol) << k;
  }
}

TEST(RespGoldCommittedTests, TheN2SidecarFitsItsStateAndItsDirectionsAreTheGauges) {
  checkCommitted(kRespCasscf, kGoldenCasscf, kGradCasscf);
}

TEST(RespGoldCommittedTests, TheHsohSidecarFitsItsStateAndItsDirectionsAreTheGauges) {
  checkCommitted(kRespHsoh, kGoldenHsoh, kGradHsoh);
}

TEST(RespGoldCommittedTests, ItIsRefusedBesideTheCasciState) {
  const Result<RespGoldFile> r = loadRespGold(std::string(kRespCasscf));
  const Result<GoldenFile> g = loadGolden(std::string(kGoldenCas44));
  const Result<GradGoldFile> s = loadGradGold(std::string(kGradCasscf));
  ASSERT_TRUE(r.has_value() && g.has_value() && s.has_value());
  const Status st = r->checkAgainst(*g, *s);
  ASSERT_FALSE(st.has_value());
  EXPECT_EQ(st.error().kind, ErrorKind::IO);
  EXPECT_NE(st.error().message.find("ci is not the golden's"), std::string::npos)
      << st.error().message;
}

// --- ParHessRoundTripTests ----------------------------------------------------------

ParHessFile miniParHess() {
  ParHessFile p;
  p.natm = 2;
  p.nmo = 5;
  p.ncore = 1;
  p.nact = 2;
  p.nvirt = 2;
  p.ndet = 4;
  p.coords = {1, 4, 5};
  Tensor* m[] = {&p.hessian, &p.kk, &p.kc, &p.cc};
  for (std::size_t k = 0; k < 4; ++k) {
    *m[k] = Tensor({3, 3});
    for (int64_t i = 0; i < 9; ++i)
      m[k]->flatRef(i) = std::sin(0.71 * static_cast<double>(i + 10 * static_cast<int64_t>(k)));
  }
  return p;
}

using ParHessRoundTripTests = GoldenTempDir;

TEST_F(ParHessRoundTripTests, WhatIsWrittenLoadsBackBitForBit) {
  const ParHessFile p = miniParHess();
  const std::string path = (dir_ / "mini.parhess").string();
  ASSERT_TRUE(writeParHess(path, p).has_value());
  const Result<ParHessFile> r = loadParHess(path);
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->natm, p.natm);
  EXPECT_EQ(r->nmo, p.nmo);
  EXPECT_EQ(r->ncore, p.ncore);
  EXPECT_EQ(r->nact, p.nact);
  EXPECT_EQ(r->nvirt, p.nvirt);
  EXPECT_EQ(r->ndet, p.ndet);
  EXPECT_EQ(r->coords, p.coords);
  for (const auto& [a, b] : {std::pair{&r->hessian, &p.hessian}, std::pair{&r->kk, &p.kk},
                             std::pair{&r->kc, &p.kc}, std::pair{&r->cc, &p.cc}}) {
    ASSERT_EQ(a->dims(), b->dims());
    for (int64_t i = 0; i < a->size(); ++i) EXPECT_EQ(a->flat(i), b->flat(i));
  }
}

TEST_F(ParHessRoundTripTests, TheBytesAreTheFormats) {
  const ParHessFile p = miniParHess();
  const std::string path = (dir_ / "mini.parhess").string();
  ASSERT_TRUE(writeParHess(path, p).has_value());
  std::string want("NEVPT2H1");
  for (const int64_t v : {p.natm, p.nmo, p.ncore, p.nact, p.nvirt})
    appendI32(want, static_cast<std::int32_t>(v));
  appendI64(want, p.ndet);
  appendI32(want, 5);
  appendArray(want, "coords", {3}, {1.0, 4.0, 5.0});
  appendArray(want, "hessian", {3, 3}, p.hessian.data());
  appendArray(want, "h_kk", {3, 3}, p.kk.data());
  appendArray(want, "h_kc", {3, 3}, p.kc.data());
  appendArray(want, "h_cc", {3, 3}, p.cc.data());
  EXPECT_EQ(read(path), want);
}

// --- ParHessLoadErrorTests ----------------------------------------------------------

using ParHessLoadErrorTests = GoldenTempDir;

// The mini file's bytes with `edit` applied to its arrays first: coords and
// the four matrices, by name.
std::string parHessBytes(const std::string_view magic, const std::vector<double>& coords,
                         const std::string_view skip = {}, const int64_t rows = 3) {
  const ParHessFile p = miniParHess();
  std::string out(magic);
  for (const int64_t v : {p.natm, p.nmo, p.ncore, p.nact, p.nvirt})
    appendI32(out, static_cast<std::int32_t>(v));
  appendI64(out, p.ndet);
  appendI32(out, skip.empty() ? 5 : 4);
  appendArray(out, "coords", {static_cast<int64_t>(coords.size())}, coords);
  for (const std::string_view name : {"hessian", "h_kk", "h_kc", "h_cc"}) {
    if (name == skip) continue;
    appendArray(out, name, {rows, 3}, std::vector<double>(static_cast<std::size_t>(rows * 3), 0.5));
  }
  return out;
}

TEST_F(ParHessLoadErrorTests, EachMalformedFileIsAnIoError) {
  expectIoError(loadParHess(write("magic.parhess", parHessBytes("NEVPT2R1", {1, 4, 5}))),
                "bad magic");
  expectIoError(loadParHess(write("order.parhess", parHessBytes("NEVPT2H1", {4, 1, 5}))),
                "strictly increasing");
  expectIoError(loadParHess(write("past.parhess", parHessBytes("NEVPT2H1", {1, 4, 6}))),
                "strictly increasing");
  expectIoError(loadParHess(write("half.parhess", parHessBytes("NEVPT2H1", {1, 4.5, 5}))),
                "not an integer");
  expectIoError(loadParHess(write("shape.parhess", parHessBytes("NEVPT2H1", {1, 4, 5}, {}, 2))),
                "is not (3, 3)");
  expectIoError(loadParHess(write("skip.parhess", parHessBytes("NEVPT2H1", {1, 4, 5}, "h_kc"))),
                "missing required array 'h_kc'");
}

}  // namespace nevpt2::test::respgold
