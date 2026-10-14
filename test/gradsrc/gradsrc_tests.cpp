// Suites for nevpt2.gradsrc (apps/sc_pseudodensity/gradsrc.cppm), the
// `.gradsrc` writer nevpt2_sc_pseudodensity --dump-source-terms uses (#73).
//
// Host-only ON PURPOSE: this is file writing and parsing, so the binary is not
// REQUIRES_GPU and runs with no card visible. It checks the bytes and the
// round trip -- never a derivative (the derivatives are the golden tier's and
// docs/testing.md, "The GPU as the source-term provider").
//
//   GradSrcRoundTripTests  writeGradSrc then readGradSrc gives every member
//                          back bit for bit -- the doubles stored raw, the
//                          shapes, the partition, a -0.0 and a denormal
//                          included -- and the same contents always give the
//                          same bytes; the optional forward-Edot direction
//                          members (#88) round-trip too, and a dump without
//                          them reads them back empty
//   GradSrcFormatTests     the bytes, read by an INDEPENDENT parser written
//                          here: a zip archive of stored members whose CRC-32s
//                          (recomputed bitwise, not by the writer's table) are
//                          right, whose names are the seven nevpt2_grad.py's
//                          load_source_terms reads, each an NPY v1.0 file with
//                          a 64-byte-aligned header naming '<f8' / '<i8', C
//                          order and the right shape -- what numpy.load reads
//   GradSrcErrorTests      a missing file, an empty one, a truncated one, a
//                          flipped data byte (the CRC catches it) and an
//                          unwritable path are each a value-tier IO Error,
//                          never an abort
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.gradsrc;

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::gradsrc {

using ::nevpt2::GradSrc;
using ::nevpt2::Tensor;

// A deterministic tensor of `dims`, with a -0.0 and a denormal in it so a
// round trip that reformatted the doubles would show.
Tensor patterned(const std::vector<std::int64_t>& dims, const double scale) {
  Tensor t(dims);
  for (std::int64_t i = 0; i < t.size(); ++i)
    t.flatRef(i) = scale * std::sin(0.37 * static_cast<double>(i) + 0.1);
  if (t.size() > 2) {
    t.flatRef(1) = -0.0;
    t.flatRef(2) = std::numeric_limits<double>::denorm_min();
  }
  return t;
}

// What a CAS(4,4) dump looks like: 28 MOs, a 6 x 6 CI vector.
GradSrc sample() {
  GradSrc s;
  s.ncore = 5;
  s.nact = 4;
  s.nvirt = 19;
  s.kappaSource = patterned({28, 28}, 0.25);
  s.ciSource = patterned({6, 6}, 0.5);
  s.ci = patterned({6, 6}, 1.0);
  s.eCorr = -0.2349210163738361;
  return s;
}

bool sameBits(const Tensor& a, const Tensor& b) {
  if (a.dims() != b.dims()) return false;
  for (std::int64_t i = 0; i < a.size(); ++i)
    if (std::bit_cast<std::uint64_t>(a.flat(i)) != std::bit_cast<std::uint64_t>(b.flat(i)))
      return false;
  return true;
}

std::vector<std::uint8_t> readBytes(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void writeBytes(const std::filesystem::path& p, const std::vector<std::uint8_t>& b) {
  std::ofstream out(p, std::ios::binary);
  out.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}

std::uint64_t le(const std::vector<std::uint8_t>& b, std::size_t at, int n) {
  std::uint64_t v = 0;
  for (int i = 0; i < n; ++i) v |= static_cast<std::uint64_t>(b[at + i]) << (8 * i);
  return v;
}

// CRC-32 the slow way, bit by bit: independent of the writer's table.
std::uint32_t crcBitwise(const std::uint8_t* p, std::size_t n) {
  std::uint32_t c = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < n; ++i) {
    c ^= p[i];
    for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
  }
  return ~c;
}

// One stored member as the independent parser below sees it.
struct Member {
  std::string name;
  std::uint32_t crc = 0;
  std::size_t dataAt = 0;
  std::size_t size = 0;
};

// Walks the local headers from the front -- the other direction from the
// reader under test, which goes through the central directory.
std::vector<Member> walkLocalHeaders(const std::vector<std::uint8_t>& b) {
  std::vector<Member> out;
  std::size_t at = 0;
  while (at + 30 <= b.size() && le(b, at, 4) == 0x04034b50u) {
    Member m;
    m.crc = static_cast<std::uint32_t>(le(b, at + 14, 4));
    m.size = le(b, at + 18, 4);
    const std::size_t nlen = le(b, at + 26, 2);
    const std::size_t xlen = le(b, at + 28, 2);
    EXPECT_EQ(le(b, at + 8, 2), 0u) << "member not stored";
    EXPECT_EQ(le(b, at + 22, 4), m.size) << "compressed and uncompressed sizes differ";
    m.name.assign(reinterpret_cast<const char*>(b.data()) + at + 30, nlen);
    m.dataAt = at + 30 + nlen + xlen;
    out.push_back(m);
    at = m.dataAt + m.size;
  }
  return out;
}

class GradSrcTempDir : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::random_device rd;
    dir_ = std::filesystem::temp_directory_path() /
           std::format("nevpt2_gradsrc_tests_{}_{}_{:08x}", info->test_suite_name(), info->name(),
                       rd());
    std::filesystem::create_directories(dir_);
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  std::string path(const std::string& name) const { return (dir_ / name).string(); }

  std::filesystem::path dir_;
};

// --- GradSrcRoundTripTests ------------------------------------------------------

class GradSrcRoundTripTests : public GradSrcTempDir {};

TEST_F(GradSrcRoundTripTests, EveryMemberComesBackBitForBit) {
  const GradSrc s = sample();
  const std::string p = path("x.gradsrc");
  ASSERT_TRUE(writeGradSrc(p, s).has_value());
  const nevpt2::Result<GradSrc> back = readGradSrc(p);
  ASSERT_TRUE(back.has_value()) << back.error().message;
  EXPECT_TRUE(sameBits(back->kappaSource, s.kappaSource));
  EXPECT_TRUE(sameBits(back->ciSource, s.ciSource));
  EXPECT_TRUE(sameBits(back->ci, s.ci));
  EXPECT_EQ(back->ncore, s.ncore);
  EXPECT_EQ(back->nact, s.nact);
  EXPECT_EQ(back->nvirt, s.nvirt);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(back->eCorr), std::bit_cast<std::uint64_t>(s.eCorr));
}

// What a dump carrying the forward Edot(kappa) directions (#88) looks like:
// three directions on sample()'s 28 MOs.
GradSrc sampleWithDirections() {
  GradSrc s = sample();
  s.edotKappa = patterned({3, 28, 28}, 0.125);
  s.edotFrozen = patterned({3}, -0.03);
  s.edotDigest = patterned({3}, 0.007);
  return s;
}

TEST_F(GradSrcRoundTripTests, TheDirectionMembersComeBackBitForBit) {
  const GradSrc s = sampleWithDirections();
  const std::string p = path("d.gradsrc");
  ASSERT_TRUE(writeGradSrc(p, s).has_value());
  const nevpt2::Result<GradSrc> back = readGradSrc(p);
  ASSERT_TRUE(back.has_value()) << back.error().message;
  ASSERT_TRUE(back->hasDirections());
  EXPECT_TRUE(sameBits(back->edotKappa, s.edotKappa));
  EXPECT_TRUE(sameBits(back->edotFrozen, s.edotFrozen));
  EXPECT_TRUE(sameBits(back->edotDigest, s.edotDigest));
  EXPECT_TRUE(sameBits(back->kappaSource, s.kappaSource));
}

TEST_F(GradSrcRoundTripTests, ADumpWithoutDirectionsReadsThemBackEmpty) {
  // The members are optional as a set: a dump written before #88, or without
  // them, still reads, with the three left empty.
  ASSERT_TRUE(writeGradSrc(path("x.gradsrc"), sample()).has_value());
  const nevpt2::Result<GradSrc> back = readGradSrc(path("x.gradsrc"));
  ASSERT_TRUE(back.has_value()) << back.error().message;
  EXPECT_FALSE(back->hasDirections());
  EXPECT_EQ(back->edotFrozen.size(), 0);
  EXPECT_EQ(back->edotDigest.size(), 0);
}

TEST_F(GradSrcRoundTripTests, SameContentsSameBytes) {
  const GradSrc s = sample();
  ASSERT_TRUE(writeGradSrc(path("a.gradsrc"), s).has_value());
  ASSERT_TRUE(writeGradSrc(path("b.gradsrc"), s).has_value());
  EXPECT_EQ(readBytes(path("a.gradsrc")), readBytes(path("b.gradsrc")));
}

TEST_F(GradSrcRoundTripTests, AVectorShapedCiVectorKeepsItsShape) {
  // A 1-D CI vector (the "(n,)" tuple spelling) round-trips too.
  GradSrc s = sample();
  s.ciSource = patterned({36}, 0.5);
  s.ci = patterned({36}, 1.0);
  ASSERT_TRUE(writeGradSrc(path("v.gradsrc"), s).has_value());
  const nevpt2::Result<GradSrc> back = readGradSrc(path("v.gradsrc"));
  ASSERT_TRUE(back.has_value()) << back.error().message;
  EXPECT_EQ(back->ci.dims(), std::vector<std::int64_t>{36});
  EXPECT_TRUE(sameBits(back->ciSource, s.ciSource));
}

// --- GradSrcFormatTests ---------------------------------------------------------

class GradSrcFormatTests : public GradSrcTempDir {};

TEST_F(GradSrcFormatTests, IsAStoredZipOfTheSevenNpyMembersWithGoodCrcs) {
  ASSERT_TRUE(writeGradSrc(path("x.gradsrc"), sample()).has_value());
  const std::vector<std::uint8_t> b = readBytes(path("x.gradsrc"));
  const std::vector<Member> members = walkLocalHeaders(b);
  std::vector<std::string> names;
  for (const Member& m : members) {
    names.push_back(m.name);
    EXPECT_EQ(crcBitwise(b.data() + m.dataAt, m.size), m.crc) << m.name;
  }
  EXPECT_EQ(names, (std::vector<std::string>{"kappa_source.npy", "ci_source.npy", "ci.npy",
                                             "ncore.npy", "nact.npy", "nvirt.npy",
                                             "e_corr.npy"}));
  // The end-of-central-directory record closes the file and counts them all.
  ASSERT_GE(b.size(), 22u);
  EXPECT_EQ(le(b, b.size() - 22, 4), 0x06054b50u);
  EXPECT_EQ(le(b, b.size() - 22 + 10, 2), members.size());
}

TEST_F(GradSrcFormatTests, TheDirectionMembersFollowTheSevenAsNpyFiles) {
  ASSERT_TRUE(writeGradSrc(path("d.gradsrc"), sampleWithDirections()).has_value());
  const std::vector<std::uint8_t> b = readBytes(path("d.gradsrc"));
  const std::map<std::string, std::string> want{{"edot_kappa.npy", "(3, 28, 28)"},
                                                {"edot_frozen.npy", "(3,)"},
                                                {"edot_digest.npy", "(3,)"}};
  std::vector<std::string> names;
  for (const Member& m : walkLocalHeaders(b)) {
    names.push_back(m.name);
    EXPECT_EQ(crcBitwise(b.data() + m.dataAt, m.size), m.crc) << m.name;
    if (!want.contains(m.name)) continue;
    const std::size_t hlen = le(b, m.dataAt + 8, 2);
    const std::string header(reinterpret_cast<const char*>(b.data()) + m.dataAt + 10, hlen);
    EXPECT_NE(header.find("'descr': '<f8'"), std::string::npos) << header;
    EXPECT_NE(header.find("'shape': " + want.at(m.name)), std::string::npos) << header;
  }
  EXPECT_EQ(names, (std::vector<std::string>{"kappa_source.npy", "ci_source.npy", "ci.npy",
                                             "ncore.npy", "nact.npy", "nvirt.npy",
                                             "e_corr.npy", "edot_kappa.npy", "edot_frozen.npy",
                                             "edot_digest.npy"}));
}

TEST_F(GradSrcFormatTests, EachMemberIsAnNpyV1FileNumpyCanRead) {
  ASSERT_TRUE(writeGradSrc(path("x.gradsrc"), sample()).has_value());
  const std::vector<std::uint8_t> b = readBytes(path("x.gradsrc"));
  const std::map<std::string, std::pair<std::string, std::string>> want{
      {"kappa_source.npy", {"'<f8'", "(28, 28)"}}, {"ci_source.npy", {"'<f8'", "(6, 6)"}},
      {"ci.npy", {"'<f8'", "(6, 6)"}},             {"ncore.npy", {"'<i8'", "()"}},
      {"nact.npy", {"'<i8'", "()"}},               {"nvirt.npy", {"'<i8'", "()"}},
      {"e_corr.npy", {"'<f8'", "()"}},
  };
  for (const Member& m : walkLocalHeaders(b)) {
    const std::uint8_t* p = b.data() + m.dataAt;
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(p), 6), "\x93NUMPY") << m.name;
    EXPECT_EQ(p[6], 1) << m.name;
    EXPECT_EQ(p[7], 0) << m.name;
    const std::size_t hlen = static_cast<std::size_t>(p[8]) | (static_cast<std::size_t>(p[9]) << 8);
    EXPECT_EQ((10 + hlen) % 64, 0u) << m.name << ": the data offset is not 64-aligned";
    const std::string header(reinterpret_cast<const char*>(p) + 10, hlen);
    EXPECT_EQ(header.back(), '\n') << m.name;
    const auto& [descr, shape] = want.at(m.name);
    EXPECT_NE(header.find("'descr': " + descr), std::string::npos) << header;
    EXPECT_NE(header.find("'fortran_order': False"), std::string::npos) << header;
    EXPECT_NE(header.find("'shape': " + shape), std::string::npos) << header;
  }
}

TEST_F(GradSrcFormatTests, DataIsLittleEndianCOrder) {
  const GradSrc s = sample();
  ASSERT_TRUE(writeGradSrc(path("x.gradsrc"), s).has_value());
  const std::vector<std::uint8_t> b = readBytes(path("x.gradsrc"));
  for (const Member& m : walkLocalHeaders(b)) {
    if (m.name != "kappa_source.npy" && m.name != "ncore.npy") continue;
    const std::size_t hlen = le(b, m.dataAt + 8, 2);
    const std::size_t data = m.dataAt + 10 + hlen;
    if (m.name == "ncore.npy") {
      EXPECT_EQ(le(b, data, 8), 5u);
    } else {
      // Element (2, 3) of a (28, 28) row-major matrix is element 2*28+3.
      EXPECT_EQ(le(b, data + 8 * (2 * 28 + 3), 8),
                std::bit_cast<std::uint64_t>(s.kappaSource(2, 3)));
    }
  }
}

// --- GradSrcErrorTests ----------------------------------------------------------

class GradSrcErrorTests : public GradSrcTempDir {};

TEST_F(GradSrcErrorTests, AMissingFileIsAnIoError) {
  const nevpt2::Result<GradSrc> r = readGradSrc(path("absent.gradsrc"));
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, nevpt2::ErrorKind::IO);
}

TEST_F(GradSrcErrorTests, AnEmptyOrTruncatedFileIsAnIoError) {
  ASSERT_TRUE(writeGradSrc(path("x.gradsrc"), sample()).has_value());
  std::vector<std::uint8_t> b = readBytes(path("x.gradsrc"));
  for (const std::size_t keep : {std::size_t{0}, std::size_t{100}, b.size() / 2, b.size() - 1}) {
    writeBytes(path("t.gradsrc"), std::vector<std::uint8_t>(b.begin(), b.begin() + keep));
    const nevpt2::Result<GradSrc> r = readGradSrc(path("t.gradsrc"));
    ASSERT_FALSE(r.has_value()) << "kept " << keep << " bytes";
    EXPECT_EQ(r.error().kind, nevpt2::ErrorKind::IO) << "kept " << keep << " bytes";
  }
}

TEST_F(GradSrcErrorTests, AFlippedDataByteFailsTheCrc) {
  ASSERT_TRUE(writeGradSrc(path("x.gradsrc"), sample()).has_value());
  std::vector<std::uint8_t> b = readBytes(path("x.gradsrc"));
  const std::vector<Member> members = walkLocalHeaders(b);
  ASSERT_FALSE(members.empty());
  b[members[0].dataAt + members[0].size - 1] ^= 0x01u;  // kappa_source's last data byte
  writeBytes(path("f.gradsrc"), b);
  const nevpt2::Result<GradSrc> r = readGradSrc(path("f.gradsrc"));
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, nevpt2::ErrorKind::IO);
  EXPECT_NE(r.error().message.find("CRC"), std::string::npos) << r.error().message;
}

TEST_F(GradSrcErrorTests, AnUnwritablePathIsAnIoError) {
  const nevpt2::Status st = writeGradSrc(path("no/such/dir/x.gradsrc"), sample());
  ASSERT_FALSE(st.has_value());
  EXPECT_EQ(st.error().kind, nevpt2::ErrorKind::IO);
}

}  // namespace nevpt2::test::gradsrc
