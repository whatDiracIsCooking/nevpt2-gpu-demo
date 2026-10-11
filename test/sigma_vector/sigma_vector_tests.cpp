// Suites for nevpt2.sigma_vector -- the 3-body and 4-body-contracted sigma
// vectors (src/rdm/sigma_vector.cu, launched by src/rdm/sigma_vector.cpp) on
// seeded random coefficient tensors and a seeded random normalized CI vector
// at CAS(4,4) and CAS(6,6). Logic, not an energy: no golden file is read and
// the integrals are random.
//
// WHAT A SIGMA VECTOR IS, and what is checked. The device computes
//
//   sigma3[K] = sum_{pqrstu} d3[p,q,r,s,t,u] (E_pq E_rs E_tu |ci>)[K]
//
// (and the 4-body form with the ERI contraction folded in; see
// src/rdm/sigma_vector.cppm) by walking the REVERSE single-excitation link
// tables backwards from the determinant each thread owns. Every reference
// here goes the other way round -- repeated FORWARD application of E_pq to a
// vector, `applyAll` below, the same scatter nevpt2.rdm_host_dm's applyAllE
// is -- so a sign convention or a transposed index pair does not cancel
// between the two:
//
//   SigmaVector3Tests
//     MatchesHostConstruction    the full vector against the host triple
//                                forward application, CAS(4,4) and CAS(6,6)
//     ContractsToDeviceDm3       sum_K ci[K] sigma3[K] = sum d3 . dm3, with
//                                dm3 from buildRdmsDevice -- which pins the
//                                index LAYOUT to the RDM build's own
//     IsTilingInvariant          one vector at several --tiles-style counts,
//                                including one with a short last tile
//   SigmaVector3GammaTests
//     MatchesHostRdm3Derivative  the acceptance test of issue #29:
//                                sum_{pqrstu} d3 Gamma^I with
//                                Gamma^I = d dm3 / d ci[I], built on the host
//                                as a CENTRAL DIFFERENCE of the host dm3
//                                contraction -- exact, with no truncation
//                                error at all, because that contraction is
//                                quadratic in ci. Affordable only at
//                                ndet = 36: it is two host builds per
//                                determinant.
//   SigmaVector4Tests
//     MatchesHostConstruction    both f3 orders against the host QUADRUPLE
//                                forward application, CAS(4,4) (n^8 operator
//                                strings: the reason this one is CAS(4,4)
//                                alone)
//     ContractsToDeviceF3        sum_K ci[K] sigma4[K] = sum d6 . f3{ca,ac}
//                                against buildRdmsDevice's own digests, so
//                                the 4-body convention is pinned to the
//                                tensors the energy actually reads
//     IsTilingInvariant          as above, across the two kernel passes
//
// The one-sided sigma and the two-sided Gamma. sigmaVector3Device computes
// the ONE-SIDED contraction sum d3 (E E E|ci>), while the RDM derivative is
// Gamma^I = <I|E E E|ci> + <ci|E E E|I>; the second term is the first with
// the operator string reversed, so
//   sum_{pqrstu} d3 Gamma^I = sigma3[I] with d3 + d3 reversed
// (`reversed` below), which is what the Gamma suite feeds the device.
//
// Tolerances. 1e-10 on |device - host| / max(1, max|host|) -- the issue's
// bound. The device sums each determinant's terms in a different order than
// the host does (a reverse walk against a forward scatter), so nothing here
// is compared bit for bit; a wrong sign, a transposed pair or a dropped walk
// moves an element by a sizeable fraction of its own size, many orders above
// that.
//
// REQUIRES_GPU: every case launches the sigma kernels on the process's one
// DeviceResources (nevpt2.test.shared_resources).
//
// TU shape: gtest's header FIRST, then `import std;` and the modules.
#include <gtest/gtest.h>

import std;
import nevpt2.link_tables;
import nevpt2.rdm_build;
import nevpt2.rdm_plan;  // tileWidthForCount: the tile plan the sigma launches reuse
import nevpt2.sigma_vector;
import nevpt2.test.shared_resources;

// Not a module unit (gtest is a textual header), so no partition: the helpers
// and suites live in a named namespace instead of an anonymous one.
namespace nevpt2::test::sigma_vector {

constexpr double kTol = 1e-10;

struct CasCase {
  int64_t norb;
  int64_t nelecA;
  int64_t nelecB;
  std::uint64_t seed;
  // Tile counts for the invariance cases: 1, 2 and one whose last tile is
  // short (checked in the test, not assumed).
  std::vector<int64_t> tileCounts;
};

// CAS(4,4): 36 determinants, 6 reverse-table rows per string per spin;
// 5 tiles of width 8 leave a last tile of 4. CAS(6,6): 400 determinants,
// 7 tiles are 6x58 + 52.
const CasCase& cas44() {
  static const CasCase kCase{4, 2, 2, 0x51'6d'a0'44u, {1, 2, 5}};
  return kCase;
}
const CasCase& cas66() {
  static const CasCase kCase{6, 3, 3, 0x51'6d'a0'66u, {1, 3, 7}};
  return kCase;
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

int64_t nstringsA(const CasCase& c) { return link_tables::num_strings(c.norb, c.nelecA); }
int64_t nstringsB(const CasCase& c) { return link_tables::num_strings(c.norb, c.nelecB); }

// A seeded random CI vector of shape (na, nb), normalized to 1.
Tensor randomCi(const CasCase& c) {
  Tensor ci({nstringsA(c), nstringsB(c)});
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

// A seeded random tensor of the given shape: a coefficient tensor (n^6), or
// the chemists'-ordered active ERI (n^4). It need not be a physical integral
// array -- every identity checked here is linear in it.
Tensor randomTensor(std::vector<int64_t> dims, std::uint64_t seed) {
  Tensor t(std::move(dims));
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int64_t i = 0; i < t.size(); ++i) t.flatRef(i) = dist(gen);
  return t;
}

// d3 with its operator string reversed: the adjoint half of the RDM
// derivative (see the header).
Tensor reversed(const Tensor& d3, int64_t n) {
  Tensor out(d3.dims());
  for (int64_t p = 0; p < n; ++p)
    for (int64_t q = 0; q < n; ++q)
      for (int64_t r = 0; r < n; ++r)
        for (int64_t s = 0; s < n; ++s)
          for (int64_t t = 0; t < n; ++t)
            for (int64_t u = 0; u < n; ++u) out(p, q, r, s, t, u) = d3(u, t, s, r, q, p);
  return out;
}

Tensor plus(const Tensor& x, const Tensor& y) {
  Tensor out(x.dims());
  for (int64_t i = 0; i < out.size(); ++i) out.flatRef(i) = x.flat(i) + y.flat(i);
  return out;
}

// --- the host reference: forward application of every E_pq ------------------

// The forward single-excitation tables of both spins, cached: the references
// below call applyAll up to n^6 times.
class Forward {
 public:
  explicit Forward(const CasCase& c)
      : n_(c.norb),
        na_(nstringsA(c)),
        nb_(nstringsB(c)),
        nla_(link_tables::nlink(c.norb, c.nelecA)),
        nlb_(link_tables::nlink(c.norb, c.nelecB)),
        fa_(link_tables::gen_linkstr_index(c.norb, c.nelecA)),
        fb_(link_tables::gen_linkstr_index(c.norb, c.nelecB)) {}

  int64_t norb() const { return n_; }
  int64_t ndet() const { return na_ * nb_; }
  int64_t npair() const { return n_ * n_; }

  // v[(p * n + q) * ndet + K] = (E_pq |x>)[K], for every (p, q) at once: a
  // row [cre, des, target, sign] of string `addr` says
  // E_{cre,des} |addr> = sign |target> for that spin, with the other spin's
  // string riding along. nevpt2.rdm_host_dm's applyAllE, over a flat vector.
  std::vector<double> applyAll(const std::vector<double>& x) const {
    std::vector<double> v(static_cast<std::size_t>(npair() * ndet()), 0.0);
    double* const vp = v.data();
    const double* const xp = x.data();
    for (int64_t a = 0; a < na_; ++a)
      for (int64_t l = 0; l < nla_; ++l) {
        const int* const row = &fa_[static_cast<std::size_t>((a * nla_ + l) * 4)];
        double* const out = vp + (row[0] * n_ + row[1]) * ndet() + row[2] * nb_;
        const double* const in = xp + a * nb_;
        const double sign = row[3];
        for (int64_t b = 0; b < nb_; ++b) out[b] += sign * in[b];
      }
    for (int64_t b = 0; b < nb_; ++b)
      for (int64_t l = 0; l < nlb_; ++l) {
        const int* const row = &fb_[static_cast<std::size_t>((b * nlb_ + l) * 4)];
        double* const out = vp + (row[0] * n_ + row[1]) * ndet() + row[2];
        const double* const in = xp + b;
        const double sign = row[3];
        for (int64_t a = 0; a < na_; ++a) out[a * nb_] += sign * in[a * nb_];
      }
    return v;
  }

  // One (p, q) block of an applyAll result, as a vector of its own.
  std::vector<double> row(const std::vector<double>& v, int64_t pq) const {
    const auto first = v.begin() + static_cast<std::ptrdiff_t>(pq * ndet());
    return std::vector<double>(first, first + static_cast<std::ptrdiff_t>(ndet()));
  }

 private:
  int64_t n_, na_, nb_, nla_, nlb_;
  std::vector<int> fa_, fb_;
};

// sigma3[K] = sum_{pqrstu} d3[p,q,r,s,t,u] (E_pq E_rs E_tu |ci>)[K], by three
// nested forward applications: the LAST index pair is applied first.
std::vector<double> hostSigma3(const Forward& f, const std::vector<double>& ci,
                               const Tensor& d3) {
  const int64_t n2 = f.npair();
  const int64_t ndet = f.ndet();
  std::vector<double> sigma(static_cast<std::size_t>(ndet), 0.0);
  const std::vector<double> vTu = f.applyAll(ci);
  for (int64_t tu = 0; tu < n2; ++tu) {
    const std::vector<double> vRs = f.applyAll(f.row(vTu, tu));
    for (int64_t rs = 0; rs < n2; ++rs) {
      const std::vector<double> vPq = f.applyAll(f.row(vRs, rs));
      for (int64_t pq = 0; pq < n2; ++pq) {
        const double d = d3.flat((pq * n2 + rs) * n2 + tu);
        if (d == 0.0) continue;
        const double* const w = vPq.data() + pq * ndet;
        for (int64_t K = 0; K < ndet; ++K) sigma[static_cast<std::size_t>(K)] += d * w[K];
      }
    }
  }
  return sigma;
}

// The 4-body contracted sigma on the host, by FOUR nested forward
// applications (n^8 operator strings -- CAS(4,4) only):
//
//   sigma[K] = sum_{wv,ut,mid,qp} coef (E_wv E_ut E_mid E_qp |ci>)[K]
//   coef = sum_a d6[w,v,u,t,s,a] eri[a,x,q,p], mid = (s,x)   (order 0, ca)
//   coef = sum_a d6[w,v,u,t,a,s] eri[a,x,q,p], mid = (x,s)   (order 1, ac)
std::vector<double> hostSigma4(const Forward& f, const std::vector<double>& ci, int order,
                               const Tensor& d6, const Tensor& eri) {
  const int64_t n = f.norb();
  const int64_t n2 = f.npair();
  const int64_t ndet = f.ndet();
  std::vector<double> sigma(static_cast<std::size_t>(ndet), 0.0);
  const std::vector<double> vQp = f.applyAll(ci);
  for (int64_t qp = 0; qp < n2; ++qp) {
    const int64_t q = qp / n, p = qp % n;
    const std::vector<double> vMid = f.applyAll(f.row(vQp, qp));
    for (int64_t mid = 0; mid < n2; ++mid) {
      // The middle operator is E_sx (ca) or E_xs (ac): `s` is d6's free
      // index, `x` the one the ERI contracts.
      const int64_t s = order == 0 ? mid / n : mid % n;
      const int64_t x = order == 0 ? mid % n : mid / n;
      const std::vector<double> vUt = f.applyAll(f.row(vMid, mid));
      for (int64_t ut = 0; ut < n2; ++ut) {
        const int64_t u = ut / n, t = ut % n;
        const std::vector<double> vWv = f.applyAll(f.row(vUt, ut));
        for (int64_t wv = 0; wv < n2; ++wv) {
          const int64_t w = wv / n, v = wv % n;
          double coef = 0.0;
          for (int64_t a = 0; a < n; ++a) {
            const double d = order == 0 ? d6(w, v, u, t, s, a) : d6(w, v, u, t, a, s);
            coef += d * eri(a, x, q, p);
          }
          if (coef == 0.0) continue;
          const double* const z = vWv.data() + wv * ndet;
          for (int64_t K = 0; K < ndet; ++K) sigma[static_cast<std::size_t>(K)] += coef * z[K];
        }
      }
    }
  }
  return sigma;
}

// --- comparison helpers -----------------------------------------------------

double maxAbs(const std::vector<double>& x) {
  double m = 0.0;
  for (const double v : x) m = std::max(m, std::abs(v));
  return m;
}

std::vector<double> flat(const Tensor& t) { return t.data(); }

double dot(const std::vector<double>& x, const std::vector<double>& y) {
  double s = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) s += x[i] * y[i];
  return s;
}

// |got - want| over the determinant axis, reported per element so a sign
// error names the determinant it broke on, and compared against
// kTol * max(1, max|want|) -- not bit for bit (the two walks reassociate).
void expectVectorsAgree(const std::vector<double>& got, const std::vector<double>& want,
                        std::string_view what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  const double tol = kTol * std::max(1.0, maxAbs(want));
  double worst = 0.0;
  std::size_t bad = 0;
  for (std::size_t K = 0; K < want.size(); ++K) {
    const double d = std::abs(got[K] - want[K]);
    worst = std::max(worst, d);
    if (d > tol && ++bad <= 5)
      ADD_FAILURE() << what << " at determinant " << K << ": got " << got[K] << ", want "
                    << want[K] << " (|diff| " << d << " > " << tol << ")";
  }
  EXPECT_EQ(bad, 0u) << what << ": mismatched determinants (worst |diff| " << worst << ")";
  // A reference of its own size: a comparison against ~0 would be vacuous.
  EXPECT_GT(maxAbs(want), 1e-6) << what << ": the reference vector is ~0";
}

// dm3, f3ca and f3ac from the device build, read back flat (n^6 each). The
// emitted digest on both backends (HIP would otherwise default to the BLAS
// one), one tile: this is a reference, not a performance path.
struct DeviceRdms {
  std::vector<double> dm3, f3ca, f3ac;
};

DeviceRdms buildRdms(const CasCase& c, const Tensor& ci, const Tensor& h2e,
                     const DeviceResources& res) {
  RdmBuildOptions opt;
  opt.nTiles = 1;
  opt.cublas = false;
  opt.blasDigest = false;
  opt.ozaki = false;
  RdmBuildResult built = buildRdmsDevice(opt, ci, c.norb, c.nelecA, c.nelecB, h2e, res);
  // downloadTensor synchronizes the stream itself.
  return {downloadTensor(built.dm3, res.stream()).data(),
          downloadTensor(built.f3ca, res.stream()).data(),
          downloadTensor(built.f3ac, res.stream()).data()};
}

SigmaVectorOptions tiles(int64_t nTiles) {
  SigmaVectorOptions opt;
  opt.nTiles = nTiles;
  return opt;
}

// --- SigmaVector3Tests ------------------------------------------------------

TEST(SigmaVector3Tests, MatchesHostConstruction) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  for (const CasCase& c : {cas44(), cas66()}) {
    SCOPED_TRACE(caseName(c));
    const int64_t n = c.norb;
    const Tensor ci = randomCi(c);
    const Tensor d3 = randomTensor({n, n, n, n, n, n}, c.seed ^ 0xd3u);
    const Tensor got = sigmaVector3Device(tiles(1), ci, n, c.nelecA, c.nelecB, d3, *res);
    const std::vector<double> want = hostSigma3(Forward(c), flat(ci), d3);
    expectVectorsAgree(flat(got), want, "sigma3");
  }
}

TEST(SigmaVector3Tests, ContractsToDeviceDm3) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  for (const CasCase& c : {cas44(), cas66()}) {
    SCOPED_TRACE(caseName(c));
    const int64_t n = c.norb;
    const Tensor ci = randomCi(c);
    const Tensor d3 = randomTensor({n, n, n, n, n, n}, c.seed ^ 0xd3u);
    const Tensor sigma = sigmaVector3Device(tiles(1), ci, n, c.nelecA, c.nelecB, d3, *res);
    // h2e feeds f3ac/f3ca only; dm3 is what this case reads.
    const DeviceRdms rdms = buildRdms(c, ci, Tensor({n, n, n, n}), *res);
    // sum_K ci[K] (E_pq E_rs E_tu|ci>)[K] = <ci|E_pq E_rs E_tu|ci> = dm3.
    const double want = dot(flat(d3), rdms.dm3);
    const double got = dot(flat(ci), flat(sigma));
    EXPECT_NEAR(got, want, kTol * std::max(1.0, std::abs(want)));
    EXPECT_GT(std::abs(want), 1e-6) << "the dm3 contraction is ~0";
  }
}

TEST(SigmaVector3Tests, IsTilingInvariant) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  for (const CasCase& c : {cas44(), cas66()}) {
    SCOPED_TRACE(caseName(c));
    const int64_t n = c.norb;
    const int64_t ndet = nstringsA(c) * nstringsB(c);
    const Tensor ci = randomCi(c);
    const Tensor d3 = randomTensor({n, n, n, n, n, n}, c.seed ^ 0xd3u);
    const std::vector<double> ref =
        flat(sigmaVector3Device(tiles(1), ci, n, c.nelecA, c.nelecB, d3, *res));
    bool sawShortLastTile = false;
    for (const int64_t nTiles : c.tileCounts) {
      SCOPED_TRACE(std::format("nTiles={}", nTiles));
      const int64_t width = tileWidthForCount(ndet, nTiles);
      sawShortLastTile = sawShortLastTile || (ndet % width != 0);
      expectVectorsAgree(flat(sigmaVector3Device(tiles(nTiles), ci, n, c.nelecA, c.nelecB, d3,
                                                 *res)),
                         ref, "sigma3 across tile counts");
    }
    EXPECT_TRUE(sawShortLastTile) << "no tile count here leaves a short last tile";
  }
}

// --- SigmaVector3GammaTests -------------------------------------------------

TEST(SigmaVector3GammaTests, MatchesHostRdm3Derivative) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const CasCase& c = cas44();
  const int64_t n = c.norb;
  const Tensor ci = randomCi(c);
  const Tensor d3 = randomTensor({n, n, n, n, n, n}, c.seed ^ 0x6a'22'a0u);
  const Forward f(c);
  const int64_t ndet = f.ndet();

  // f(x) = sum_{pqrstu} d3 <x|E_pq E_rs E_tu|x> = <x | sigma3(d3, x)>, the
  // host contraction whose CI gradient is sum d3 Gamma^I.
  const auto contraction = [&](const std::vector<double>& x) {
    return dot(x, hostSigma3(f, x, d3));
  };
  // A central difference of it, which is EXACT: the contraction is quadratic
  // in x, so the h^2 terms cancel identically and only roundoff is left.
  // h is a power of two, so x +- h is exact too.
  constexpr double kH = 1.0 / 64.0;
  std::vector<double> want(static_cast<std::size_t>(ndet), 0.0);
  for (int64_t I = 0; I < ndet; ++I) {
    std::vector<double> plusI = flat(ci), minusI = flat(ci);
    plusI[static_cast<std::size_t>(I)] += kH;
    minusI[static_cast<std::size_t>(I)] -= kH;
    want[static_cast<std::size_t>(I)] = (contraction(plusI) - contraction(minusI)) / (2.0 * kH);
  }

  // sum d3 Gamma^I is the one-sided sigma of d3 + d3 reversed (see the
  // header), which is what the device is handed.
  const Tensor symmetric = plus(d3, reversed(d3, n));
  const Tensor got = sigmaVector3Device(tiles(1), ci, n, c.nelecA, c.nelecB, symmetric, *res);
  expectVectorsAgree(flat(got), want, "sum d3 Gamma^I");
}

// --- SigmaVector4Tests ------------------------------------------------------

TEST(SigmaVector4Tests, MatchesHostConstruction) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const CasCase& c = cas44();
  const int64_t n = c.norb;
  const Tensor ci = randomCi(c);
  const Tensor eri = randomTensor({n, n, n, n}, c.seed ^ 0xe1u);
  const Tensor d6 = randomTensor({n, n, n, n, n, n}, c.seed ^ 0xd6u);
  const Forward f(c);
  for (const int order : {0, 1}) {
    SCOPED_TRACE(order == 0 ? "ca" : "ac");
    const Tensor got =
        sigmaVector4Device(tiles(1), order, ci, n, c.nelecA, c.nelecB, d6, eri, *res);
    const std::vector<double> want = hostSigma4(f, flat(ci), order, d6, eri);
    expectVectorsAgree(flat(got), want, "sigma4");
  }
}

TEST(SigmaVector4Tests, ContractsToDeviceF3) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  for (const CasCase& c : {cas44(), cas66()}) {
    SCOPED_TRACE(caseName(c));
    const int64_t n = c.norb;
    const Tensor ci = randomCi(c);
    const Tensor d6 = randomTensor({n, n, n, n, n, n}, c.seed ^ 0xd6u);
    // The build takes the PHYSICIST-ordered h2e and derives its chemists'
    // eriF3 as transpose(h2e, {0,2,1,3}), which is its own inverse -- so this
    // is the h2e whose eriF3 is `eri`.
    const Tensor eri = randomTensor({n, n, n, n}, c.seed ^ 0xe1u);
    const Tensor h2e = transpose(eri, {0, 2, 1, 3});
    const DeviceRdms rdms = buildRdms(c, ci, h2e, *res);
    for (const int order : {0, 1}) {
      SCOPED_TRACE(order == 0 ? "ca" : "ac");
      const Tensor sigma =
          sigmaVector4Device(tiles(1), order, ci, n, c.nelecA, c.nelecB, d6, eri, *res);
      const double want = dot(flat(d6), order == 0 ? rdms.f3ca : rdms.f3ac);
      const double got = dot(flat(ci), flat(sigma));
      EXPECT_NEAR(got, want, kTol * std::max(1.0, std::abs(want)));
      EXPECT_GT(std::abs(want), 1e-6) << "the f3 contraction is ~0";
    }
  }
}

TEST(SigmaVector4Tests, IsTilingInvariant) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const CasCase& c = cas44();
  const int64_t n = c.norb;
  const int64_t ndet = nstringsA(c) * nstringsB(c);
  const Tensor ci = randomCi(c);
  const Tensor eri = randomTensor({n, n, n, n}, c.seed ^ 0xe1u);
  const Tensor d6 = randomTensor({n, n, n, n, n, n}, c.seed ^ 0xd6u);
  const std::vector<double> ref =
      flat(sigmaVector4Device(tiles(1), 0, ci, n, c.nelecA, c.nelecB, d6, eri, *res));
  bool sawShortLastTile = false;
  for (const int64_t nTiles : c.tileCounts) {
    SCOPED_TRACE(std::format("nTiles={}", nTiles));
    const int64_t width = tileWidthForCount(ndet, nTiles);
    sawShortLastTile = sawShortLastTile || (ndet % width != 0);
    expectVectorsAgree(
        flat(sigmaVector4Device(tiles(nTiles), 0, ci, n, c.nelecA, c.nelecB, d6, eri, *res)), ref,
        "sigma4 across tile counts");
  }
  EXPECT_TRUE(sawShortLastTile) << "no tile count here leaves a short last tile";
}

}  // namespace nevpt2::test::sigma_vector
