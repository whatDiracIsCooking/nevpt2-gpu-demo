// Suites for nevpt2.df_integrals' DfIntegralSource -- the density-fitted
// IntegralSource -- on tiny synthetic B blocks, against a host reference built
// without any of its code. Until these, the source was checked only through
// the DF golden energies, where a wrong axis permutation or a bad B_vc
// transpose would show only as an energy that is off, with no pointer to the
// block. What they check is logic, not an energy:
//
//   DfSlabTests              for every ExtBlock and four [b0, b1) ranges
//                            (full, a single index, a middle run, the tail),
//                            slab() has the block's physicist dims with
//                            batchAxis(b) shortened, and equals the host
//                            reference restricted along that axis
//   DfActiveH2eTests         activeH2e() is h2e[t,u,v,w] = (tv|uw)
//   DfSizeAccountingTests    extent(), fullBlockDoubles(), naux(), and
//                            peakSlabDoubles()/resetPeak() as the interface
//                            documents them
//   DfCreateValidationTests  create() with disagreeing ranks or shapes returns
//                            an IO Error -- no abort -- and uploads nothing
//
// THE HOST REFERENCE. One symmetric three-index tensor B[L,p,q] = B[L,q,p]
// over ALL norb = ncore + ncas + nvirt orbitals (core first, then active, then
// virtual, as PySCF orders them) is drawn from a seeded generator; the four
// blocks create() takes are cut from it, and the host builds the full
// four-index (pq|rs) = sum_L B[L,p,q] B[L,r,s] from it directly. Each block's
// reference is then the PySCF slice generate_golden.py's _integral_blocks
// takes for it -- an orbital range per index of (pq|rs) and a numpy transpose
// -- copied here as data (kPyscfSlices), NOT the X/Y/axes table in
// df_integrals.cpp's slab(). So the two sides share nothing but the
// definition of (pq|rs): a wrong permutation, a B_va where B_vc belongs or a
// batch offset on the wrong axis all land on the wrong element.
//
// Symmetric B is what a real-orbital density fit gives, and what the source
// relies on (its B_vc is B_cv transposed). The four extents are all distinct
// (naux 6, ncore 3, ncas 4, nvirt 5), so a block built with a core where a
// virtual belongs has the wrong dims rather than the right ones by accident.
//
// Tolerance. Every element is a dot product of naux = 6 products of numbers in
// [-1, 1], so |element| <= 6 and the device's reassociation moves it by a few
// ulps; |got - want| <= kTol * max(1, max|want|) with kTol = 1e-12. A
// misplaced index moves an element by O(1), twelve orders above that.
//
// REQUIRES_GPU: every case builds on the process's one DeviceResources
// (nevpt2.test.shared_resources), each slab on its one stream.
//
// The fixture (SyntheticDf, makeSyntheticDf, hostBlock, hostSlab, makeSource)
// is written to be reused by a sibling suite comparing FullBlockSource's
// slabs, built from hostBlock's full blocks, against these.
//
// TU shape: gtest's header FIRST, then `import std;` and the modules.
#include <gtest/gtest.h>

import std;
import nevpt2.df_integrals;
import nevpt2.test.shared_resources;

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one. One binary
// per suite file, so a name clash here is a link error.
namespace nevpt2::test::df_integrals {

// 1e-12, scaled by max(1, max|reference|) -- see the header comment.
constexpr double kTol = 1e-12;

// Seed for the synthetic B. Any value works; a fixed one makes a failure
// reproducible.
constexpr std::uint64_t kSeed = 0x5eedDF;

// --- the synthetic system --------------------------------------------------------

// Orbital ranges of the full space: core [0, ncore), active [ncore, nocc),
// virtual [nocc, norb).
enum class Space { Core, Active, Virtual };

struct SyntheticDf {
  int64_t naux = 0, ncore = 0, ncas = 0, nvirt = 0;
  Tensor b;    // B[L, p, q] over all norb orbitals, symmetric in (p, q)
  Tensor eri;  // (pq|rs) = sum_L B[L,p,q] B[L,r,s], (norb, norb, norb, norb)
  // The four blocks DfIntegralSource::create takes, cut from `b`.
  Tensor bAA, bCA, bVA, bCV;

  int64_t norb() const { return ncore + ncas + nvirt; }
  int64_t offset(Space s) const {
    switch (s) {
      case Space::Core: return 0;
      case Space::Active: return ncore;
      case Space::Virtual: return ncore + ncas;
    }
    return 0;
  }
  int64_t length(Space s) const {
    switch (s) {
      case Space::Core: return ncore;
      case Space::Active: return ncas;
      case Space::Virtual: return nvirt;
    }
    return 0;
  }
};

// B[L, p, q] restricted to p in `ps`, q in `qs`: what _df_blocks writes as
// lpq[:, ps, qs].
Tensor cutB(const SyntheticDf& sys, Space ps, Space qs) {
  const int64_t p0 = sys.offset(ps), q0 = sys.offset(qs);
  Tensor out({sys.naux, sys.length(ps), sys.length(qs)});
  for (int64_t l = 0; l < out.dim(0); ++l)
    for (int64_t p = 0; p < out.dim(1); ++p)
      for (int64_t q = 0; q < out.dim(2); ++q) out(l, p, q) = sys.b(l, p0 + p, q0 + q);
  return out;
}

// The seeded synthetic system: B uniform in [-1, 1] (symmetrized by drawing
// the upper triangle and mirroring it), the full (pq|rs), and the four blocks.
SyntheticDf makeSyntheticDf(int64_t naux, int64_t ncore, int64_t ncas, int64_t nvirt,
                            std::uint64_t seed = kSeed) {
  SyntheticDf sys;
  sys.naux = naux;
  sys.ncore = ncore;
  sys.ncas = ncas;
  sys.nvirt = nvirt;
  const int64_t n = sys.norb();

  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  sys.b = Tensor({naux, n, n});
  for (int64_t l = 0; l < naux; ++l)
    for (int64_t p = 0; p < n; ++p)
      for (int64_t q = p; q < n; ++q) {
        const double v = dist(gen);
        sys.b(l, p, q) = v;
        sys.b(l, q, p) = v;
      }

  sys.eri = Tensor({n, n, n, n});
  for (int64_t p = 0; p < n; ++p)
    for (int64_t q = 0; q < n; ++q)
      for (int64_t r = 0; r < n; ++r)
        for (int64_t s = 0; s < n; ++s) {
          double acc = 0.0;
          for (int64_t l = 0; l < naux; ++l) acc += sys.b(l, p, q) * sys.b(l, r, s);
          sys.eri(p, q, r, s) = acc;
        }

  sys.bAA = cutB(sys, Space::Active, Space::Active);
  sys.bCA = cutB(sys, Space::Core, Space::Active);
  sys.bVA = cutB(sys, Space::Virtual, Space::Active);
  sys.bCV = cutB(sys, Space::Core, Space::Virtual);
  return sys;
}

// The system every suite here uses: all four extents distinct.
const SyntheticDf& standardSystem() {
  static const SyntheticDf sys = makeSyntheticDf(6, 3, 4, 5);
  return sys;
}

// --- the host reference ----------------------------------------------------------

// One block as PySCF builds it: the slice of (pq|rs) with index k in
// spaces[k], then numpy's transpose(axes) (output axis i = input axis
// axes[i]).
struct PyscfSlice {
  std::array<Space, 4> spaces;
  std::array<int, 4> axes;
};

// generate_golden.py's _integral_blocks, transcribed. PySCF's eris arrays are
// (pq|rs) slices: ppaa = (p p|a a), papa = (p a|p a), pacv = (p a|c v),
// cvcv = (c v|c v), with p the full range.
PyscfSlice pyscfSlice(ExtBlock b) {
  constexpr Space C = Space::Core, A = Space::Active, V = Space::Virtual;
  switch (b) {
    case ExtBlock::Sr: return {{V, A, A, A}, {0, 2, 1, 3}};     // ppaa[nocc:, ncore:nocc]
    case ExtBlock::Si: return {{A, C, A, A}, {0, 2, 1, 3}};     // ppaa[ncore:nocc, :ncore]
    case ExtBlock::Sijrs: return {{C, V, C, V}, {0, 1, 2, 3}};  // cvcv
    case ExtBlock::Sijr: return {{C, A, C, V}, {3, 1, 2, 0}};   // pacv[:ncore]
    case ExtBlock::Srsi:                                        // pacv[nocc:]
    case ExtBlock::SrsiT: return {{V, A, C, V}, {3, 0, 2, 1}};  //   (the same block)
    case ExtBlock::Srs: return {{V, A, V, A}, {0, 2, 1, 3}};    // papa[nocc:, :, nocc:]
    case ExtBlock::Sij: return {{C, A, C, A}, {1, 3, 0, 2}};    // papa[:ncore, :, :ncore]
    case ExtBlock::Sir1: return {{V, C, A, A}, {0, 2, 1, 3}};   // ppaa[nocc:, :ncore]
    case ExtBlock::Sir2: return {{V, A, C, A}, {0, 3, 1, 2}};   // papa[nocc:, :, :ncore]
  }
  return {};
}

// The active h2e: ppaa[ncore:nocc, ncore:nocc].transpose(0, 2, 1, 3).
constexpr PyscfSlice kActiveH2e{{Space::Active, Space::Active, Space::Active, Space::Active},
                                {0, 2, 1, 3}};

Tensor hostSliceTranspose(const SyntheticDf& sys, const PyscfSlice& sl) {
  std::array<int64_t, 4> inDims{}, inOff{};
  for (int k = 0; k < 4; ++k) {
    inDims[k] = sys.length(sl.spaces[k]);
    inOff[k] = sys.offset(sl.spaces[k]);
  }
  Tensor out({inDims[sl.axes[0]], inDims[sl.axes[1]], inDims[sl.axes[2]], inDims[sl.axes[3]]});
  std::array<int64_t, 4> o{}, in{};
  for (o[0] = 0; o[0] < out.dim(0); ++o[0])
    for (o[1] = 0; o[1] < out.dim(1); ++o[1])
      for (o[2] = 0; o[2] < out.dim(2); ++o[2])
        for (o[3] = 0; o[3] < out.dim(3); ++o[3]) {
          for (int k = 0; k < 4; ++k) in[sl.axes[k]] = o[k];
          out(o[0], o[1], o[2], o[3]) =
              sys.eri(inOff[0] + in[0], inOff[1] + in[1], inOff[2] + in[2], inOff[3] + in[3]);
        }
  return out;
}

// The whole block `b`, in its physicist layout.
Tensor hostBlock(const SyntheticDf& sys, ExtBlock b) {
  return hostSliceTranspose(sys, pyscfSlice(b));
}

Tensor hostActiveH2e(const SyntheticDf& sys) { return hostSliceTranspose(sys, kActiveH2e); }

// `full` with `axis` restricted to [b0, b1).
Tensor restrictAxis(const Tensor& full, int64_t axis, int64_t b0, int64_t b1) {
  std::vector<int64_t> dims = full.dims();
  dims[axis] = b1 - b0;
  Tensor out(dims);
  std::array<int64_t, 4> o{};
  for (o[0] = 0; o[0] < out.dim(0); ++o[0])
    for (o[1] = 0; o[1] < out.dim(1); ++o[1])
      for (o[2] = 0; o[2] < out.dim(2); ++o[2])
        for (o[3] = 0; o[3] < out.dim(3); ++o[3]) {
          std::array<int64_t, 4> f = o;
          f[axis] += b0;
          out(o[0], o[1], o[2], o[3]) = full(f[0], f[1], f[2], f[3]);
        }
  return out;
}

// What slab(b, b0, b1) must return.
Tensor hostSlab(const SyntheticDf& sys, ExtBlock b, int64_t b0, int64_t b1) {
  return restrictAxis(hostBlock(sys, b), batchAxis(b), b0, b1);
}

// --- device plumbing -------------------------------------------------------------

constexpr std::array<ExtBlock, 10> kAllBlocks{
    ExtBlock::Sr,   ExtBlock::Si,  ExtBlock::Sijrs, ExtBlock::Sijr, ExtBlock::Srsi,
    ExtBlock::SrsiT, ExtBlock::Srs, ExtBlock::Sij,   ExtBlock::Sir1, ExtBlock::Sir2};

// SrsiT and Srsi share a name (the same block); the trace says which.
std::string blockLabel(ExtBlock b) {
  return b == ExtBlock::SrsiT ? std::string("h2e_v_Srsi (SrsiT)") : std::string(extBlockName(b));
}

// The process's DeviceResources, or nullptr after recording the creation
// Error as this test's failure (callers ASSERT_NE on it).
const DeviceResources* resourcesOrFail() {
  const auto& res = test::sharedResources();
  if (!res.has_value()) {
    ADD_FAILURE() << "DeviceResources::create failed: " << kindName(res.error().kind) << ": "
                  << res.error().message;
    return nullptr;
  }
  return res->get();
}

// A DfIntegralSource over `sys`'s blocks, or the Error create() returned.
Result<DfIntegralSource> makeSource(const SyntheticDf& sys, const DeviceResources& res) {
  return DfIntegralSource::create(sys.bAA, sys.bCA, sys.bVA, sys.bCV, res);
}

// The [b0, b1) ranges each block is sliced over, for an extent n >= 3: the
// full range, a single index, a middle run, and the tail.
std::vector<std::pair<int64_t, int64_t>> rangesFor(int64_t n) {
  return {{0, n}, {1, 2}, {1, n - 1}, {n - 1, n}};
}

// Shape, then every element within kTol * max(1, max|want|).
void expectMatches(const Tensor& got, const Tensor& want) {
  ASSERT_EQ(got.dims(), want.dims());
  double scale = 1.0;
  for (const double v : want.data()) scale = std::max(scale, std::abs(v));
  int64_t bad = 0;
  for (int64_t i = 0; i < want.size(); ++i) {
    if (std::abs(got.flat(i) - want.flat(i)) > kTol * scale) {
      if (++bad <= 5)
        ADD_FAILURE() << "flat element " << i << ": got " << got.flat(i) << ", want "
                      << want.flat(i);
    }
  }
  EXPECT_EQ(bad, 0) << "elements off by more than " << kTol * scale;
}

// The pool's currently allocated bytes, after every free already queued on
// res.stream() has landed.
std::uint64_t poolUsedNow(const DeviceResources& res) {
  // Earlier cases' stream-ordered frees may still be queued; the count read
  // below must not move under them, so wait for the stream first.
  gpuCheck(wwrStreamSynchronize(res.stream()));
  std::uint64_t used = 0;
  gpuCheck(wwrMemPoolGetAttribute(res.pool(), wwrMemPoolAttrUsedMemCurrent, &used));
  return used;
}

// --- DfSlabTests -----------------------------------------------------------------

TEST(DfSlabTests, HostReferenceIsTheDefiningSum) {
  // The reference itself, spot-checked against the definition, so a slip in
  // the transcription above cannot make both sides agree on a wrong block.
  // Sr[r,a,b,c] = (rb|ac), with the orbital offsets spelled out.
  const SyntheticDf& sys = standardSystem();
  const Tensor sr = hostBlock(sys, ExtBlock::Sr);
  ASSERT_EQ(sr.dims(), (std::vector<int64_t>{sys.nvirt, sys.ncas, sys.ncas, sys.ncas}));
  const int64_t v0 = sys.ncore + sys.ncas, a0 = sys.ncore;
  const int64_t r = 2, a = 1, b = 3, c = 0;
  double want = 0.0;
  for (int64_t l = 0; l < sys.naux; ++l)
    want += sys.b(l, v0 + r, a0 + b) * sys.b(l, a0 + a, a0 + c);
  EXPECT_NEAR(sr(r, a, b, c), want, kTol);
  // Sijr[r,p,j,i] = (rj|ip): a virtual, an active and two cores.
  const Tensor sijr = hostBlock(sys, ExtBlock::Sijr);
  ASSERT_EQ(sijr.dims(), (std::vector<int64_t>{sys.nvirt, sys.ncas, sys.ncore, sys.ncore}));
  EXPECT_NEAR(sijr(4, 2, 1, 0), sys.eri(v0 + 4, 1, 0, a0 + 2), kTol);
}

TEST(DfSlabTests, EveryBlockEveryRangeMatchesTheHost) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const SyntheticDf& sys = standardSystem();
  auto src = makeSource(sys, *res);
  ASSERT_TRUE(src.has_value()) << src.error().message;

  for (const ExtBlock b : kAllBlocks) {
    const int64_t n = src->extent(b);
    for (const auto& [b0, b1] : rangesFor(n)) {
      SCOPED_TRACE(std::format("{} [{}, {}) along axis {}", blockLabel(b), b0, b1, batchAxis(b)));
      const DeviceTensor slab = src->slab(b, b0, b1, *res);
      expectMatches(downloadTensor(slab, res->stream()), hostSlab(sys, b, b0, b1));
    }
  }
}

TEST(DfSlabTests, SrsiTIsSrsiBatchedAlongItsSecondAxis) {
  // The same block, two batch axes: a full SrsiT slab equals a full Srsi
  // slab, and an SrsiT slab of r is Srsi's [:, r, :, :].
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const SyntheticDf& sys = standardSystem();
  auto src = makeSource(sys, *res);
  ASSERT_TRUE(src.has_value()) << src.error().message;

  const Tensor srsi = downloadTensor(src->slab(ExtBlock::Srsi, 0, sys.nvirt, *res), res->stream());
  expectMatches(downloadTensor(src->slab(ExtBlock::SrsiT, 0, sys.nvirt, *res), res->stream()),
                srsi);
  expectMatches(downloadTensor(src->slab(ExtBlock::SrsiT, 2, 4, *res), res->stream()),
                restrictAxis(srsi, 1, 2, 4));
}

// --- DfActiveH2eTests ------------------------------------------------------------

TEST(DfActiveH2eTests, IsTvUwFromBaa) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const SyntheticDf& sys = standardSystem();
  auto src = makeSource(sys, *res);
  ASSERT_TRUE(src.has_value()) << src.error().message;

  const Tensor got = downloadTensor(src->activeH2e(*res), res->stream());
  const Tensor want = hostActiveH2e(sys);
  expectMatches(got, want);
  // h2e[t,u,v,w] = (tv|uw), spelled out once against the defining sum.
  const int64_t a0 = sys.ncore;
  EXPECT_NEAR(got(0, 1, 2, 3), sys.eri(a0 + 0, a0 + 2, a0 + 1, a0 + 3), kTol);
}

// --- DfSizeAccountingTests -------------------------------------------------------

TEST(DfSizeAccountingTests, ExtentIsTheBatchAxisLength) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const SyntheticDf& sys = standardSystem();
  auto src = makeSource(sys, *res);
  ASSERT_TRUE(src.has_value()) << src.error().message;

  EXPECT_EQ(src->naux(), sys.naux);
  for (const ExtBlock b : kAllBlocks) {
    SCOPED_TRACE(blockLabel(b));
    EXPECT_EQ(src->extent(b), hostBlock(sys, b).dim(batchAxis(b)));
  }
  // The core-batched blocks, by name, so the reference cannot drift with them.
  EXPECT_EQ(src->extent(ExtBlock::Si), sys.ncore);
  EXPECT_EQ(src->extent(ExtBlock::Sij), sys.ncore);
  EXPECT_EQ(src->extent(ExtBlock::Sijrs), sys.ncore);
  EXPECT_EQ(src->extent(ExtBlock::Sr), sys.nvirt);
  EXPECT_EQ(src->extent(ExtBlock::SrsiT), sys.nvirt);
}

TEST(DfSizeAccountingTests, FullBlockDoublesIsTheUnbatchedBlockSize) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const SyntheticDf& sys = standardSystem();
  auto src = makeSource(sys, *res);
  ASSERT_TRUE(src.has_value()) << src.error().message;

  for (const ExtBlock b : kAllBlocks) {
    SCOPED_TRACE(blockLabel(b));
    EXPECT_EQ(src->fullBlockDoubles(b), hostBlock(sys, b).size());
    // And a full-range slab is exactly that many doubles.
    EXPECT_EQ(src->slab(b, 0, src->extent(b), *res).size(), src->fullBlockDoubles(b));
  }
}

TEST(DfSizeAccountingTests, PeakTracksTheLargestSlabUntilReset) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const SyntheticDf& sys = standardSystem();
  auto src = makeSource(sys, *res);
  ASSERT_TRUE(src.has_value()) << src.error().message;

  // create() builds B_vc, which is not a slab build: nothing counted yet.
  EXPECT_EQ(src->peakSlabDoubles(), 0);

  // Every slab build allocates the GEMM result and (unless the block needs no
  // permute) the permuted slab, both exactly the slab's size, so a fresh
  // peak is the slab's size.
  for (const ExtBlock b : kAllBlocks) {
    SCOPED_TRACE(blockLabel(b));
    src->resetPeak();
    EXPECT_EQ(src->peakSlabDoubles(), 0);
    const DeviceTensor one = src->slab(b, 1, 2, *res);
    EXPECT_EQ(src->peakSlabDoubles(), one.size());
  }

  // A high-water mark: a smaller build after a larger one leaves it alone...
  // (Sijrs, c*v*c*v = 225 doubles, then Srs, v*v*a*a = 400, at these extents.)
  src->resetPeak();
  const int64_t full = src->slab(ExtBlock::Sijrs, 0, sys.ncore, *res).size();
  const int64_t single = src->slab(ExtBlock::Sijrs, 2, 3, *res).size();
  ASSERT_LT(single, full);
  EXPECT_EQ(src->peakSlabDoubles(), full);
  EXPECT_EQ(src->peakSlabDoubles(), src->fullBlockDoubles(ExtBlock::Sijrs));
  // ...and a larger one raises it.
  const int64_t larger = src->slab(ExtBlock::Srs, 0, sys.nvirt, *res).size();
  ASSERT_GT(larger, full);
  EXPECT_EQ(src->peakSlabDoubles(), larger);

  // activeH2e() is a build too, and counts.
  src->resetPeak();
  EXPECT_EQ(src->activeH2e(*res).size(), src->peakSlabDoubles());
  EXPECT_EQ(src->peakSlabDoubles(), sys.ncas * sys.ncas * sys.ncas * sys.ncas);
}

// --- DfCreateValidationTests -----------------------------------------------------

// create() over the given blocks must return an IO Error and leave the pool's
// allocated bytes exactly where they were.
void expectIoErrorAndNothingUploaded(const Tensor& bAA, const Tensor& bCA, const Tensor& bVA,
                                     const Tensor& bCV, const DeviceResources& res) {
  const std::uint64_t before = poolUsedNow(res);
  auto src = DfIntegralSource::create(bAA, bCA, bVA, bCV, res);
  ASSERT_FALSE(src.has_value()) << "create() accepted inconsistent B blocks";
  EXPECT_EQ(src.error().kind, ErrorKind::IO) << kindName(src.error().kind);
  EXPECT_NE(src.error().message.find("inconsistent B_* shapes"), std::string::npos)
      << src.error().message;
  EXPECT_EQ(poolUsedNow(res), before) << "create() allocated device memory before refusing";
}

TEST(DfCreateValidationTests, ConsistentBlocksAreAccepted) {
  // The baseline the refusals below differ from by one block.
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const SyntheticDf& sys = standardSystem();
  EXPECT_TRUE(makeSource(sys, *res).has_value());
}

TEST(DfCreateValidationTests, WrongRankIsAnIoError) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const SyntheticDf& sys = standardSystem();
  const Tensor rank2({sys.naux, sys.ncas * sys.ncas});
  const Tensor rank4({sys.naux, sys.ncore, sys.nvirt, 1});
  {
    SCOPED_TRACE("B_aa rank 2");
    expectIoErrorAndNothingUploaded(rank2, sys.bCA, sys.bVA, sys.bCV, *res);
  }
  {
    SCOPED_TRACE("B_cv rank 4");
    expectIoErrorAndNothingUploaded(sys.bAA, sys.bCA, sys.bVA, rank4, *res);
  }
}

TEST(DfCreateValidationTests, DisagreeingShapesAreAnIoError) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const SyntheticDf& sys = standardSystem();
  const int64_t L = sys.naux, c = sys.ncore, a = sys.ncas, v = sys.nvirt;
  {
    SCOPED_TRACE("B_aa not square in its orbital axes");
    expectIoErrorAndNothingUploaded(Tensor({L, a, a + 1}), sys.bCA, sys.bVA, sys.bCV, *res);
  }
  {
    SCOPED_TRACE("B_ca with a different naux");
    expectIoErrorAndNothingUploaded(sys.bAA, Tensor({L + 1, c, a}), sys.bVA, sys.bCV, *res);
  }
  {
    SCOPED_TRACE("B_ca with a different ncas");
    expectIoErrorAndNothingUploaded(sys.bAA, Tensor({L, c, a - 1}), sys.bVA, sys.bCV, *res);
  }
  {
    SCOPED_TRACE("B_va with a different naux");
    expectIoErrorAndNothingUploaded(sys.bAA, sys.bCA, Tensor({L - 1, v, a}), sys.bCV, *res);
  }
  {
    SCOPED_TRACE("B_va with a different ncas");
    expectIoErrorAndNothingUploaded(sys.bAA, sys.bCA, Tensor({L, v, a + 1}), sys.bCV, *res);
  }
  {
    SCOPED_TRACE("B_cv stored transposed, as (L, v, c)");
    expectIoErrorAndNothingUploaded(sys.bAA, sys.bCA, sys.bVA, Tensor({L, v, c}), *res);
  }
  {
    SCOPED_TRACE("B_cv with a different naux");
    expectIoErrorAndNothingUploaded(sys.bAA, sys.bCA, sys.bVA, Tensor({L + 1, c, v}), *res);
  }
}

}  // namespace nevpt2::test::df_integrals
