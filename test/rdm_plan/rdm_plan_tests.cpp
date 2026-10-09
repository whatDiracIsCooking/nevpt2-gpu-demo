// Suites for nevpt2.rdm_plan (src/rdm/rdm_plan.cppm) -- the RDM build's pure
// host index arithmetic.
//
// Host-only ON PURPOSE: no device, no kernel, no golden -- so this binary is
// not REQUIRES_GPU and runs with no card visible. Every comparison is EXACT
// (integer logic, or a permutation that moves doubles without arithmetic).
//
//   TilePlanTests           tileWidthForCount is ceil(ndet / nTiles); the
//                           plan's tiles cover [0, ndet) exactly, in order,
//                           every one but the last at the full width
//   TilePlanDeathTest       a width or a k0 past int aborts (narrowTo), the
//                           width naming "use more --tiles"
//   EvenChunksTests         covers [0, total), each chunk at most maxLen and
//                           the fewest of them, near-equal; total % maxLen
//                           == 0, maxLen > total, maxLen = INT64_MAX
//   PermuteEriConsumeTests  [a,x,q,p] -> [x,a,p,q] on small n, against a
//                           value that encodes its own source index
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.rdm_plan;

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
// One binary per suite file, so a name clash here is a link error.
namespace nevpt2::test::rdm_plan {

constexpr int64_t kIntMax = std::numeric_limits<int>::max();
constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();

// Checks `tiles` covers [0, ndet) exactly: consecutive, starting at 0, each
// nK wide but the last, which is the remainder (or nK when nK divides ndet).
void expectExactCover(const std::vector<std::pair<int, int>>& tiles, int64_t ndet, int64_t nK) {
  ASSERT_EQ(std::ssize(tiles), ndet == 0 ? 0 : (ndet + nK - 1) / nK);
  int64_t next = 0;
  for (std::size_t i = 0; i < tiles.size(); ++i) {
    const auto [k0, width] = tiles[i];
    EXPECT_EQ(k0, next) << "tile " << i;
    if (i + 1 < tiles.size()) EXPECT_EQ(width, nK) << "tile " << i;
    EXPECT_GT(width, 0) << "tile " << i;
    next += width;
  }
  EXPECT_EQ(next, ndet);
  if (!tiles.empty()) {
    const int64_t lastWidth = ndet - (std::ssize(tiles) - 1) * nK;
    EXPECT_EQ(tiles.back().second, lastWidth);
  }
}

// --- TilePlanTests -------------------------------------------------------------

TEST(TilePlanTests, WidthForCountIsTheCeiling) {
  EXPECT_EQ(tileWidthForCount(100, 1), 100);
  EXPECT_EQ(tileWidthForCount(100, 3), 34);
  EXPECT_EQ(tileWidthForCount(100, 4), 25);
  EXPECT_EQ(tileWidthForCount(100, 100), 1);
  EXPECT_EQ(tileWidthForCount(100, 150), 1);
  EXPECT_EQ(tileWidthForCount(1, 7), 1);
}

TEST(TilePlanTests, TilesCoverTheDeterminantAxisExactly) {
  for (int64_t ndet : {1, 2, 7, 36, 63504, 853776})
    for (int64_t nTiles : {1, 2, 3, 5, 40, 150}) {
      SCOPED_TRACE(std::format("ndet={} nTiles={}", ndet, nTiles));
      const int64_t nK = tileWidthForCount(ndet, nTiles);
      const auto tiles = planTiles(ndet, nK);
      expectExactCover(tiles, ndet, nK);
      // Never more tiles than asked for (the ceiling can make fewer).
      EXPECT_LE(std::ssize(tiles), nTiles);
    }
}

TEST(TilePlanTests, LastTileIsTheRemainder) {
  const auto tiles = planTiles(10, 4);
  ASSERT_EQ(tiles.size(), 3u);
  EXPECT_EQ(tiles[0], (std::pair<int, int>{0, 4}));
  EXPECT_EQ(tiles[1], (std::pair<int, int>{4, 4}));
  EXPECT_EQ(tiles[2], (std::pair<int, int>{8, 2}));
}

TEST(TilePlanTests, EvenSplitHasAFullLastTile) {
  const auto tiles = planTiles(12, 4);
  ASSERT_EQ(tiles.size(), 3u);
  EXPECT_EQ(tiles.back(), (std::pair<int, int>{8, 4}));
}

TEST(TilePlanTests, EmptyAxisHasNoTiles) { EXPECT_TRUE(planTiles(0, 4).empty()); }

TEST(TilePlanTests, WidestIntTileIsAccepted) {
  // A width of exactly INT_MAX still fits: one tile, no abort.
  const auto tiles = planTiles(kIntMax, kIntMax);
  ASSERT_EQ(tiles.size(), 1u);
  EXPECT_EQ(tiles[0], (std::pair<int, int>{0, std::numeric_limits<int>::max()}));
}

// --- TilePlanDeathTest -----------------------------------------------------------
//
// narrowTo is the abort tier: check(false, ...) prints "error at file:line in
// function: <what>" and std::abort()s, before any tile is stored (so the huge
// ndet here allocates nothing).

TEST(TilePlanDeathTest, WidthPastIntAbortsAskingForMoreTiles) {
  EXPECT_DEATH((void)planTiles(kIntMax + 1, kIntMax + 1),
               "2147483648 does not fit in a 32-bit signed integer; use more --tiles");
}

TEST(TilePlanDeathTest, OneTileOfAWideAxisAbortsAskingForMoreTiles) {
  const int64_t ndet = int64_t{3} << 31;
  EXPECT_DEATH((void)planTiles(ndet, tileWidthForCount(ndet, 1)), "use more --tiles");
}

TEST(TilePlanDeathTest, StartPastIntAborts) {
  // Every width fits (2^30), but the third tile starts at 2^31: a determinant
  // space too large for the kernels' 32-bit indexing at any tile count.
  EXPECT_DEATH((void)planTiles((int64_t{1} << 31) + 5, int64_t{1} << 30),
               "2147483648 does not fit in a 32-bit signed integer");
}

// --- EvenChunksTests -------------------------------------------------------------

// Checks `chunks` covers [0, total) in order, each at most maxLen, as few as
// possible (ceil(total / maxLen)), and near-equal (lengths differ by <= 1).
void expectEvenChunks(const std::vector<std::pair<int64_t, int64_t>>& chunks, int64_t total,
                      int64_t maxLen) {
  const int64_t fewest = total / maxLen + (total % maxLen != 0);
  ASSERT_EQ(std::ssize(chunks), fewest);
  int64_t next = 0, shortest = kInt64Max, longest = 0;
  for (const auto [b, e] : chunks) {
    EXPECT_EQ(b, next);
    EXPECT_GT(e, b);
    EXPECT_LE(e - b, maxLen);
    shortest = std::min(shortest, e - b);
    longest = std::max(longest, e - b);
    next = e;
  }
  EXPECT_EQ(next, total);
  if (!chunks.empty()) EXPECT_LE(longest - shortest, 1);
}

TEST(EvenChunksTests, CoversWithTheFewestChunksOfAtMostMaxLen) {
  for (int64_t total : {1, 10, 12, 100, 144, 1000})
    for (int64_t maxLen : {1, 3, 10, 16, 128}) {
      SCOPED_TRACE(std::format("total={} maxLen={}", total, maxLen));
      expectEvenChunks(evenChunks(total, maxLen), total, maxLen);
    }
}

TEST(EvenChunksTests, ExactMultipleGivesEqualChunks) {
  const auto chunks = evenChunks(12, 4);
  ASSERT_EQ(chunks.size(), 3u);
  EXPECT_EQ(chunks[0], (std::pair<int64_t, int64_t>{0, 4}));
  EXPECT_EQ(chunks[1], (std::pair<int64_t, int64_t>{4, 8}));
  EXPECT_EQ(chunks[2], (std::pair<int64_t, int64_t>{8, 12}));
}

TEST(EvenChunksTests, UnevenSplitIsNearEqualNotGreedy) {
  // 12 columns at <= 10 (cuBLAS's limit at CAS(12,12)): two halves of 6, not
  // 10 + 2.
  const auto chunks = evenChunks(12, 10);
  ASSERT_EQ(chunks.size(), 2u);
  EXPECT_EQ(chunks[0], (std::pair<int64_t, int64_t>{0, 6}));
  EXPECT_EQ(chunks[1], (std::pair<int64_t, int64_t>{6, 12}));
}

TEST(EvenChunksTests, MaxLenLargerThanTotalIsOneChunk) {
  const auto chunks = evenChunks(10, 16);
  ASSERT_EQ(chunks.size(), 1u);
  EXPECT_EQ(chunks[0], (std::pair<int64_t, int64_t>{0, 10}));
}

TEST(EvenChunksTests, MaxLenAtInt64MaxIsOneChunk) {
  // hipBLAS's k limit: whole. The ceiling must not overflow here.
  for (int64_t total : {1, 144, 20736}) {
    const auto chunks = evenChunks(total, kInt64Max);
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0], (std::pair<int64_t, int64_t>{0, total}));
  }
}

TEST(EvenChunksTests, ZeroTotalIsNoChunks) { EXPECT_TRUE(evenChunks(0, 10).empty()); }

// --- PermuteEriConsumeTests --------------------------------------------------------

// A value that names its own source index: eri[a,x,q,p] = 1000a + 100x + 10q + p
// (n <= 10), so a misplaced element shows which index went wrong.
std::vector<double> encodedEri(int64_t n) {
  std::vector<double> eri(static_cast<std::size_t>(n * n * n * n));
  for (int64_t a = 0; a < n; ++a)
    for (int64_t x = 0; x < n; ++x)
      for (int64_t q = 0; q < n; ++q)
        for (int64_t p = 0; p < n; ++p)
          eri[static_cast<std::size_t>(((a * n + x) * n + q) * n + p)] =
              static_cast<double>(1000 * a + 100 * x + 10 * q + p);
  return eri;
}

TEST(PermuteEriConsumeTests, PermutesAXQPToXAPQ) {
  for (int64_t n : {1, 2, 3, 4}) {
    SCOPED_TRACE(std::format("n={}", n));
    const std::vector<double> out = permuteEriConsume(encodedEri(n), n);
    ASSERT_EQ(std::ssize(out), n * n * n * n);
    for (int64_t x = 0; x < n; ++x)
      for (int64_t a = 0; a < n; ++a)
        for (int64_t p = 0; p < n; ++p)
          for (int64_t q = 0; q < n; ++q)
            EXPECT_EQ(out[static_cast<std::size_t>(((x * n + a) * n + p) * n + q)],
                      static_cast<double>(1000 * a + 100 * x + 10 * q + p))
                << "x=" << x << " a=" << a << " p=" << p << " q=" << q;
  }
}

TEST(PermuteEriConsumeTests, IsAPermutation) {
  // Every source element lands exactly once: the sorted output is the input.
  std::vector<double> in = encodedEri(3);
  std::vector<double> out = permuteEriConsume(in, 3);
  std::ranges::sort(in);
  std::ranges::sort(out);
  EXPECT_EQ(out, in);
}

}  // namespace nevpt2::test::rdm_plan
