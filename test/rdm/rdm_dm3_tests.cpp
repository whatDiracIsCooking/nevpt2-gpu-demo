// Suites for the dm3 build in src/rdm/kernels.cu (produce_generic +
// digest_dm3_generic, launched from rdm_launch.cu), driven end to end through
// nevpt2.rdm_build's buildRdmsDevice on a seeded random normalized CI vector
// at CAS(4,4) and CAS(6,6). What they check is wavefunction-independent
// algebra, true of ANY N-electron state -- logic, not an energy:
//
//   Dm3HermiticityTests       dm3[p,q,r,s,t,u] = dm3[u,t,s,r,q,p], and the
//                             normal-ordered G3 (below) is Hermitian too
//   Dm3PairSymmetryTests      swapping two index pairs of dm3 moves it by the
//                             commutator [E_rs, E_tu] against the host dm2;
//                             G3 is symmetric under all 6 pair permutations
//   Dm3PartialTraceTests      sum_t dm3[pqrstt] = N dm2 (and each other pair),
//                             sum_t G3[pqrstt] = (N-2) G2, and
//                             sum_r G2[pqrr] = (N-1) dm1, against host-built
//                             dm1/dm2
//   Dm3TilingInvarianceTests  the same CI vector at nTiles 1, 2, 3 and a count
//                             that leaves a short last tile gives one dm3
//
// Convention. buildRdmsDevice's dm3 is PySCF's make_dm123 one (what the golden
// files were made with): dm3[p,q,r,s,t,u] = <0|E_pq E_rs E_tu|0>, with the
// spin-summed E_pq = sum_s a+_ps a_qs, NOT normal ordered. The host references
// here are dm1[p,q] = <0|E_pq|0> and dm2[p,q,r,s] = <0|E_pq E_rs|0>, built from
// the CI vector with nevpt2.link_tables alone (no kernel). The (N-2)/(N-1)
// trace identities hold for the normal-ordered densities
//   G2[p,q,r,s]     = dm2 - d_qr dm1[p,s]
//   G3[p,q,r,s,t,u] = dm3 - d_qr G2[p,s,t,u] - d_qt G2[p,u,r,s]
//                         - d_st G2[p,q,r,u] - d_qr d_st dm1[p,u]
// (from E_pq E_rs E_tu = e_pqrstu + d_qr e_pstu + d_qt e_purs
//  + d_st (e_pqru + d_qr E_pu)), which the host derives from the device dm3.
//
// Tolerances. Every comparison is |got - want| <= kTol * scale, scale =
// max(1, max|dm3|) -- the device reassociates each dm3 element's K-sum
// differently per tile plan, so nothing here is compared bit for bit. Each
// dm3 element is a dot product over at most 400 determinants of O(1) terms
// (the CI vector is normalized), so its rounding sits a few ulps of the
// element; an identity adds at most n = 6 of them. A misplaced index, a wrong
// sign or a dropped tile moves an element by a sizeable fraction of the CI
// weights, many orders above kTol.
//
// REQUIRES_GPU: every case builds on the process's one DeviceResources
// (nevpt2.test.shared_resources).
//
// TU shape: gtest's header FIRST, then `import std;` and the modules.
#include <gtest/gtest.h>

import std;
import nevpt2.link_tables;
import nevpt2.rdm_build;
import nevpt2.test.shared_resources;

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one. One binary
// per suite file, so a name clash here is a link error.
namespace nevpt2::test::rdm_dm3 {

// 1e-12, as the issue's tiling-invariance bound, scaled by max(1, max|dm3|).
constexpr double kTol = 1e-12;

struct CasCase {
  int64_t norb;
  int64_t nelecA;
  int64_t nelecB;
  std::uint64_t seed;
  // Tile counts for the invariance suite: 1, 2, 3 and one whose last tile is
  // short (checked in the test, not assumed).
  std::vector<int64_t> tileCounts;
};

// CAS(4,4): 36 determinants; 5 tiles of width 8 leave a last tile of 4.
// CAS(6,6): 400 determinants; 3 tiles are 134+134+132 and 7 are 6x58+52.
const std::vector<CasCase>& cases() {
  static const std::vector<CasCase> kCases = {
      {4, 2, 2, 0x5eed'0404u, {1, 2, 3, 5}},
      {6, 3, 3, 0x5eed'0606u, {1, 2, 3, 7}},
  };
  return kCases;
}

std::string caseName(const CasCase& c) {
  return std::format("CAS({},{})", c.nelecA + c.nelecB, c.norb);
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

// A seeded random CI vector of shape (na, nb), normalized to 1.
Tensor randomCi(const CasCase& c) {
  const int64_t na = link_tables::num_strings(c.norb, c.nelecA);
  const int64_t nb = link_tables::num_strings(c.norb, c.nelecB);
  Tensor ci({na, nb});
  std::mt19937_64 gen(c.seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  double norm2 = 0.0;
  for (int64_t i = 0; i < ci.size(); ++i) {
    ci.flatRef(i) = dist(gen);
    norm2 += ci.flat(i) * ci.flat(i);
  }
  const double inv = 1.0 / std::sqrt(norm2);
  for (int64_t i = 0; i < ci.size(); ++i) ci.flatRef(i) *= inv;
  return ci;
}

// dm3 from the device build at `nTiles`, read back flat (n^6, row-major).
// The emitted digest (digest_dm3_generic) on both backends: the issue is
// kernels.cu, and HIP would otherwise default to the BLAS digest. The h2e is
// zero: it feeds only f3ac/f3ca, which this file does not read.
std::vector<double> buildDm3(const CasCase& c, const Tensor& ci, int64_t nTiles,
                             const DeviceResources& res) {
  RdmBuildOptions opt;
  opt.nTiles = nTiles;
  opt.cublas = false;
  opt.blasDigest = false;
  opt.ozaki = false;
  const Tensor h2e({c.norb, c.norb, c.norb, c.norb});
  RdmBuildResult built = buildRdmsDevice(opt, ci, c.norb, c.nelecA, c.nelecB, h2e, res);
  // downloadTensor synchronizes the stream itself.
  return downloadTensor(built.dm3, res.stream()).data();
}

// v[pq] = E_pq |ci>, for every (p, q), each a flat (na*nb) vector, by the
// forward link tables: a row [cre, des, target, sign] of string `addr` says
// E_{cre,des} |addr> = sign |target>, per spin.
std::vector<std::vector<double>> applyAllE(const CasCase& c, const Tensor& ci) {
  const int64_t n = c.norb;
  const int64_t na = link_tables::num_strings(n, c.nelecA);
  const int64_t nb = link_tables::num_strings(n, c.nelecB);
  const int64_t nla = link_tables::nlink(n, c.nelecA);
  const int64_t nlb = link_tables::nlink(n, c.nelecB);
  const std::vector<int> fa = link_tables::gen_linkstr_index(n, c.nelecA);
  const std::vector<int> fb = link_tables::gen_linkstr_index(n, c.nelecB);
  std::vector<std::vector<double>> v(static_cast<std::size_t>(n * n),
                                     std::vector<double>(static_cast<std::size_t>(na * nb)));
  for (int64_t a = 0; a < na; ++a) {
    for (int64_t l = 0; l < nla; ++l) {
      const int* row = &fa[static_cast<std::size_t>((a * nla + l) * 4)];
      auto& out = v[static_cast<std::size_t>(row[0] * n + row[1])];
      for (int64_t b = 0; b < nb; ++b) {
        out[static_cast<std::size_t>(row[2] * nb + b)] += row[3] * ci(a, b);
      }
    }
  }
  for (int64_t b = 0; b < nb; ++b) {
    for (int64_t l = 0; l < nlb; ++l) {
      const int* row = &fb[static_cast<std::size_t>((b * nlb + l) * 4)];
      auto& out = v[static_cast<std::size_t>(row[0] * n + row[1])];
      for (int64_t a = 0; a < na; ++a) {
        out[static_cast<std::size_t>(a * nb + row[2])] += row[3] * ci(a, b);
      }
    }
  }
  return v;
}

double dot(const std::vector<double>& x, const std::vector<double>& y) {
  double s = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) s += x[i] * y[i];
  return s;
}

// Host dm1[p,q] = <ci|E_pq|ci> and dm2[p,q,r,s] = <ci|E_pq E_rs|ci>
// = <E_qp ci|E_rs ci> (real CI vector), flat row-major.
struct HostRdms {
  std::vector<double> dm1;
  std::vector<double> dm2;
};

HostRdms hostRdms(const CasCase& c, const Tensor& ci) {
  const int64_t n = c.norb;
  const auto v = applyAllE(c, ci);
  const std::vector<double>& c0 = ci.data();
  HostRdms h;
  h.dm1.resize(static_cast<std::size_t>(n * n));
  h.dm2.resize(static_cast<std::size_t>(n * n * n * n));
  for (int64_t p = 0; p < n; ++p)
    for (int64_t q = 0; q < n; ++q) {
      h.dm1[static_cast<std::size_t>(p * n + q)] = dot(c0, v[static_cast<std::size_t>(p * n + q)]);
      for (int64_t r = 0; r < n; ++r)
        for (int64_t s = 0; s < n; ++s)
          h.dm2[static_cast<std::size_t>(((p * n + q) * n + r) * n + s)] =
              dot(v[static_cast<std::size_t>(q * n + p)], v[static_cast<std::size_t>(r * n + s)]);
    }
  return h;
}

// Flat row-major offsets.
struct Idx {
  int64_t n;
  std::size_t operator()(int64_t p, int64_t q) const { return static_cast<std::size_t>(p * n + q); }
  std::size_t operator()(int64_t p, int64_t q, int64_t r, int64_t s) const {
    return static_cast<std::size_t>(((p * n + q) * n + r) * n + s);
  }
  std::size_t operator()(int64_t p, int64_t q, int64_t r, int64_t s, int64_t t, int64_t u) const {
    return static_cast<std::size_t>(((((p * n + q) * n + r) * n + s) * n + t) * n + u);
  }
};

double kd(int64_t a, int64_t b) { return a == b ? 1.0 : 0.0; }

// G2 from dm1/dm2 (see the header).
std::vector<double> normalOrdered2(int64_t n, const HostRdms& h) {
  const Idx ix{n};
  std::vector<double> g2(h.dm2.size());
  for (int64_t p = 0; p < n; ++p)
    for (int64_t q = 0; q < n; ++q)
      for (int64_t r = 0; r < n; ++r)
        for (int64_t s = 0; s < n; ++s)
          g2[ix(p, q, r, s)] = h.dm2[ix(p, q, r, s)] - kd(q, r) * h.dm1[ix(p, s)];
  return g2;
}

// G3 from the device dm3 and the host G2/dm1 (see the header).
std::vector<double> normalOrdered3(int64_t n, const std::vector<double>& dm3,
                                   const std::vector<double>& g2, const HostRdms& h) {
  const Idx ix{n};
  std::vector<double> g3(dm3.size());
  for (int64_t p = 0; p < n; ++p)
    for (int64_t q = 0; q < n; ++q)
      for (int64_t r = 0; r < n; ++r)
        for (int64_t s = 0; s < n; ++s)
          for (int64_t t = 0; t < n; ++t)
            for (int64_t u = 0; u < n; ++u)
              g3[ix(p, q, r, s, t, u)] =
                  dm3[ix(p, q, r, s, t, u)] - kd(q, r) * g2[ix(p, s, t, u)] -
                  kd(q, t) * g2[ix(p, u, r, s)] - kd(s, t) * g2[ix(p, q, r, u)] -
                  kd(q, r) * kd(s, t) * h.dm1[ix(p, u)];
  return g3;
}

double maxAbs(const std::vector<double>& x) {
  double m = 0.0;
  for (double v : x) m = std::max(m, std::abs(v));
  return m;
}

// Counts |got - want| > tol, reporting the first few, so a wholly wrong n^6
// tensor does not print 46656 lines. Returns the count for EXPECT_EQ.
class Mismatches {
 public:
  Mismatches(std::string what, double tol) : what_(std::move(what)), tol_(tol) {}
  void check(double got, double want, std::string_view where) {
    const double d = std::abs(got - want);
    worst_ = std::max(worst_, d);
    if (d <= tol_) return;
    if (++bad_ <= 5) {
      ADD_FAILURE() << what_ << " at " << where << ": got " << got << ", want " << want
                    << " (|diff| " << d << " > " << tol_ << ")";
    }
  }
  void expectNone() const {
    EXPECT_EQ(bad_, 0u) << what_ << ": mismatched elements (worst |diff| " << worst_ << ")";
  }

 private:
  std::string what_;
  double tol_;
  double worst_ = 0.0;
  std::size_t bad_ = 0;
};

std::string at6(int64_t p, int64_t q, int64_t r, int64_t s, int64_t t, int64_t u) {
  return std::format("[{},{},{},{},{},{}]", p, q, r, s, t, u);
}

// The host references are themselves checked first: Tr dm1 = N, and G2's
// trace -- a wrong host reference would make every device check below
// meaningless.
TEST(Dm3PartialTraceTests, HostDm2TracesToNMinusOneDm1) {
  for (const CasCase& c : cases()) {
    SCOPED_TRACE(caseName(c));
    const int64_t n = c.norb;
    const double nElec = static_cast<double>(c.nelecA + c.nelecB);
    const Idx ix{n};
    const Tensor ci = randomCi(c);
    const HostRdms h = hostRdms(c, ci);
    double tr = 0.0;
    for (int64_t p = 0; p < n; ++p) tr += h.dm1[ix(p, p)];
    EXPECT_NEAR(tr, nElec, kTol * nElec) << "Tr dm1";
    const std::vector<double> g2 = normalOrdered2(n, h);
    Mismatches m("sum_r G2[p,q,r,r] vs (N-1) dm1", kTol * std::max(1.0, maxAbs(h.dm2)));
    for (int64_t p = 0; p < n; ++p)
      for (int64_t q = 0; q < n; ++q) {
        double s = 0.0;
        for (int64_t r = 0; r < n; ++r) s += g2[ix(p, q, r, r)];
        m.check(s, (nElec - 1.0) * h.dm1[ix(p, q)], std::format("[{},{}]", p, q));
      }
    m.expectNone();
  }
}

TEST(Dm3PartialTraceTests, Dm3TracesToNDm2AndG3ToNMinusTwoG2) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  for (const CasCase& c : cases()) {
    SCOPED_TRACE(caseName(c));
    const int64_t n = c.norb;
    const double nElec = static_cast<double>(c.nelecA + c.nelecB);
    const Idx ix{n};
    const Tensor ci = randomCi(c);
    const HostRdms h = hostRdms(c, ci);
    const std::vector<double> dm3 = buildDm3(c, ci, 1, *res);
    const std::vector<double> g2 = normalOrdered2(n, h);
    const std::vector<double> g3 = normalOrdered3(n, dm3, g2, h);
    const double tol = kTol * std::max(1.0, maxAbs(dm3));

    // Raw: sum_x E_xx = N on an N-electron state, in any of the three slots.
    Mismatches raw("sum over one pair of dm3 vs N dm2", tol);
    // Normal-ordered: sum_t G3[p,q,r,s,t,t] = (N-2) G2[p,q,r,s].
    Mismatches no("sum_t G3[p,q,r,s,t,t] vs (N-2) G2", tol);
    for (int64_t p = 0; p < n; ++p)
      for (int64_t q = 0; q < n; ++q)
        for (int64_t r = 0; r < n; ++r)
          for (int64_t s = 0; s < n; ++s) {
            double first = 0.0, middle = 0.0, last = 0.0, g = 0.0;
            for (int64_t x = 0; x < n; ++x) {
              first += dm3[ix(x, x, p, q, r, s)];
              middle += dm3[ix(p, q, x, x, r, s)];
              last += dm3[ix(p, q, r, s, x, x)];
              g += g3[ix(p, q, r, s, x, x)];
            }
            const double want = nElec * h.dm2[ix(p, q, r, s)];
            const std::string where = std::format("[{},{},{},{}]", p, q, r, s);
            raw.check(first, want, "first pair " + where);
            raw.check(middle, want, "middle pair " + where);
            raw.check(last, want, "last pair " + where);
            no.check(g, (nElec - 2.0) * g2[ix(p, q, r, s)], where);
          }
    raw.expectNone();
    no.expectNone();
  }
}

TEST(Dm3HermiticityTests, Dm3AndG3AreHermitian) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  for (const CasCase& c : cases()) {
    SCOPED_TRACE(caseName(c));
    const int64_t n = c.norb;
    const Idx ix{n};
    const Tensor ci = randomCi(c);
    const HostRdms h = hostRdms(c, ci);
    const std::vector<double> dm3 = buildDm3(c, ci, 1, *res);
    const std::vector<double> g3 = normalOrdered3(n, dm3, normalOrdered2(n, h), h);
    const double tol = kTol * std::max(1.0, maxAbs(dm3));
    // <E_pq E_rs E_tu>* = <E_ut E_sr E_qp>, real; G3's adjoint swaps each
    // pair's creator and annihilator in place.
    Mismatches raw("dm3[p,q,r,s,t,u] vs dm3[u,t,s,r,q,p]", tol);
    Mismatches no("G3[p,q,r,s,t,u] vs G3[q,p,s,r,u,t]", tol);
    for (int64_t p = 0; p < n; ++p)
      for (int64_t q = 0; q < n; ++q)
        for (int64_t r = 0; r < n; ++r)
          for (int64_t s = 0; s < n; ++s)
            for (int64_t t = 0; t < n; ++t)
              for (int64_t u = 0; u < n; ++u) {
                const std::string where = at6(p, q, r, s, t, u);
                raw.check(dm3[ix(p, q, r, s, t, u)], dm3[ix(u, t, s, r, q, p)], where);
                no.check(g3[ix(p, q, r, s, t, u)], g3[ix(q, p, s, r, u, t)], where);
              }
    raw.expectNone();
    no.expectNone();
  }
}

TEST(Dm3PairSymmetryTests, PairSwapsMoveByTheCommutatorAndG3IsSymmetric) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  for (const CasCase& c : cases()) {
    SCOPED_TRACE(caseName(c));
    const int64_t n = c.norb;
    const Idx ix{n};
    const Tensor ci = randomCi(c);
    const HostRdms h = hostRdms(c, ci);
    const std::vector<double> dm3 = buildDm3(c, ci, 1, *res);
    const std::vector<double> g3 = normalOrdered3(n, dm3, normalOrdered2(n, h), h);
    const double tol = kTol * std::max(1.0, maxAbs(dm3));
    // [E_ab, E_cd] = d_bc E_ad - d_ad E_cb, so against the HOST dm2:
    //   dm3[pq,rs,tu] - dm3[pq,tu,rs] = d_st dm2[p,q,r,u] - d_ur dm2[p,q,t,s]
    //   dm3[pq,rs,tu] - dm3[rs,pq,tu] = d_qr dm2[p,s,t,u] - d_sp dm2[r,q,t,u]
    Mismatches swapLast("dm3 last-two-pair swap vs commutator", tol);
    Mismatches swapFirst("dm3 first-two-pair swap vs commutator", tol);
    Mismatches perm("G3 under the 6 pair permutations", tol);
    for (int64_t p = 0; p < n; ++p)
      for (int64_t q = 0; q < n; ++q)
        for (int64_t r = 0; r < n; ++r)
          for (int64_t s = 0; s < n; ++s)
            for (int64_t t = 0; t < n; ++t)
              for (int64_t u = 0; u < n; ++u) {
                const std::string where = at6(p, q, r, s, t, u);
                const double d = dm3[ix(p, q, r, s, t, u)];
                swapLast.check(d - dm3[ix(p, q, t, u, r, s)],
                               kd(s, t) * h.dm2[ix(p, q, r, u)] - kd(u, r) * h.dm2[ix(p, q, t, s)],
                               where);
                swapFirst.check(d - dm3[ix(r, s, p, q, t, u)],
                                kd(q, r) * h.dm2[ix(p, s, t, u)] -
                                    kd(s, p) * h.dm2[ix(r, q, t, u)],
                                where);
                const double g = g3[ix(p, q, r, s, t, u)];
                perm.check(g3[ix(p, q, t, u, r, s)], g, where);
                perm.check(g3[ix(r, s, p, q, t, u)], g, where);
                perm.check(g3[ix(r, s, t, u, p, q)], g, where);
                perm.check(g3[ix(t, u, p, q, r, s)], g, where);
                perm.check(g3[ix(t, u, r, s, p, q)], g, where);
              }
    swapLast.expectNone();
    swapFirst.expectNone();
    perm.expectNone();
  }
}

TEST(Dm3TilingInvarianceTests, EveryTileCountGivesTheSameDm3) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  for (const CasCase& c : cases()) {
    SCOPED_TRACE(caseName(c));
    const int64_t ndet =
        link_tables::num_strings(c.norb, c.nelecA) * link_tables::num_strings(c.norb, c.nelecB);
    // The plan rdm_build:tiles makes (width = ceil(ndet / nTiles)): at least
    // one count here must leave a short last tile, or that path goes untested.
    bool shortLastTile = false;
    for (int64_t nt : c.tileCounts) shortLastTile |= (ndet % ((ndet + nt - 1) / nt)) != 0;
    EXPECT_TRUE(shortLastTile) << "no tile count leaves a short last tile";

    const Tensor ci = randomCi(c);
    const std::vector<double> ref = buildDm3(c, ci, 1, *res);
    const double tol = kTol * std::max(1.0, maxAbs(ref));
    for (int64_t nt : c.tileCounts) {
      if (nt == 1) continue;
      const std::vector<double> got = buildDm3(c, ci, nt, *res);
      ASSERT_EQ(got.size(), ref.size());
      Mismatches m(std::format("dm3 at nTiles={} vs nTiles=1", nt), tol);
      for (std::size_t i = 0; i < got.size(); ++i) m.check(got[i], ref[i], std::to_string(i));
      m.expectNone();
    }
  }
}

}  // namespace nevpt2::test::rdm_dm3
