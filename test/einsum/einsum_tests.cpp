// Suites for nevpt2.einsum -- the generic device einsum (device_einsum.cu),
// called through its host module API only.
//
// REQUIRES_GPU: every case uploads operands into the process's one
// DeviceResources (nevpt2.test.shared_resources) and launches on its one
// non-blocking stream. With no card visible, each case fails on
// sharedResources()'s Error, naming it; none skips.
//
//   EinsumNewTests             deviceEinsumNew / deviceTransposeNew: a matrix
//                              product, a trace (a repeated label to a rank-0
//                              output), a 3-operand 4-index contraction of a
//                              shape the SC energy uses, a strided (sliced)
//                              operand, `scale` applied
//   EinsumAccumTests           deviceEinsumAccum adds into what `out` already
//                              holds (twice, two scales), and scale = -1
//                              subtracts
//   TransposeAccumTests        deviceTransposeAccum: a 2-index transpose and
//                              the 6-index {1,4,0,2,5,3} permutation make_a16
//                              uses, both accumulated with a scale
//   EinsumDeathTest            a malformed subscript string (no "->"), a
//                              label count that does not match an operand's
//                              rank, an operand count that does not match the
//                              label groups, an inconsistent label extent and
//                              an output label in no operand -- all our bugs,
//                              so check() aborts naming the caller
//
// Each device result is compared with a plain host loop over nevpt2.tensor's
// Tensor -- never `==`: the kernel sums a contraction in its own order (one
// thread per (output, contracted) pair, atomicAdd), so it reassociates. The
// tolerance, per output element, is
//     |device - host| <= kRelTol * sum_k |term_k|  (+ kAbsFloor)
// where sum_k |term_k| is the host's sum of the absolute values of every term
// added into that element (the accumulated prior value and each product).
// Any summation order of n terms is within (n - 1) * eps * sum|terms| of the
// exact sum (the standard bound), and so is the host's; the largest
// contraction here has n = 7 * 7 * 8 = 392 terms, so 2 * 392 * eps ~ 1.7e-13
// covers both. kRelTol = 1e-12 leaves ~6x headroom over that bound and is
// still tight enough that a dropped, doubled or transposed term (each moves
// an element by O(1) * a term) fails it by orders of magnitude. kAbsFloor
// only guards an element whose every term is exactly zero. This is THIS
// suite's tolerance, for small contractions of O(1) data -- not the golden
// tier's 1e-7 on energies.
//
// The death tests build no device data: buildPlan validates the subscripts
// before it touches an operand's memory, so empty DeviceTensors with their
// dims set are enough. deviceEinsumNew still takes a DeviceResources, so the
// death statement creates it in the forked child (the parent never touches
// the device in a death-only run); hence the suite is REQUIRES_GPU too.
//
// TU shape: gtest's header FIRST (then <cstdio>, for stderr), then
// `import std;` and the modules.
#include <gtest/gtest.h>

#include <cstdio>  // stderr, a macro `import std` does not carry

import std;
import nevpt2.einsum;
import nevpt2.test.shared_resources;

namespace nevpt2 {
namespace {

constexpr double kRelTol = 1e-12;
constexpr double kAbsFloor = 1e-300;

// A host Tensor of deterministic, distinct, O(1), mixed-sign values; `seed`
// decorrelates two operands of the same shape.
Tensor patterned(std::vector<int64_t> dims, int seed) {
  Tensor t(std::move(dims));
  for (int64_t i = 0; i < t.size(); ++i) {
    t.flatRef(i) = std::sin(0.37 * static_cast<double>(i) + 1.3 * seed) + 0.01 * seed;
  }
  return t;
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

// The host reference for one output: its value and the sum of |term| that
// built it (the scale for the tolerance).
struct Expected {
  Tensor value;
  Tensor absSum;
};

Expected expectedOfShape(std::vector<int64_t> dims) {
  return Expected{Tensor(dims), Tensor(dims)};
}

void expectNear(const Tensor& device, const Expected& host) {
  ASSERT_EQ(device.dims(), host.value.dims());
  for (int64_t i = 0; i < device.size(); ++i) {
    const double tol = kRelTol * host.absSum.flat(i) + kAbsFloor;
    EXPECT_NEAR(device.flat(i), host.value.flat(i), tol) << "flat element " << i;
  }
}

// --- EinsumNewTests -------------------------------------------------------------

TEST(EinsumNewTests, MatrixProduct) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  constexpr int64_t ni = 5, nj = 7, nk = 3;  // nothing square, nothing a warp
  const Tensor a = patterned({ni, nj}, 1);
  const Tensor b = patterned({nj, nk}, 2);
  const DeviceTensor da = uploadTensor(a, *res);
  const DeviceTensor db = uploadTensor(b, *res);

  const double scale = 0.75;
  const DeviceTensor dc = deviceEinsumNew("ij,jk->ik", {&da, &db}, scale, *res);
  EXPECT_EQ(dc.dims, (std::vector<int64_t>{ni, nk}));

  Expected want = expectedOfShape({ni, nk});
  for (int64_t i = 0; i < ni; ++i)
    for (int64_t k = 0; k < nk; ++k)
      for (int64_t j = 0; j < nj; ++j) {
        const double term = scale * a(i, j) * b(j, k);
        want.value(i, k) += term;
        want.absSum(i, k) += std::abs(term);
      }
  expectNear(downloadTensor(dc, res->stream()), want);
}

TEST(EinsumNewTests, TraceToRankZero) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  constexpr int64_t n = 9;
  const Tensor a = patterned({n, n}, 3);
  const DeviceTensor da = uploadTensor(a, *res);

  const DeviceTensor dt = deviceEinsumNew("ii->", {&da}, 1.0, *res);
  EXPECT_EQ(dt.rank(), 0);
  EXPECT_EQ(dt.size(), 1);

  Expected want = expectedOfShape({});
  for (int64_t i = 0; i < n; ++i) {
    want.value.flatRef(0) += a(i, i);
    want.absSum.flatRef(0) += std::abs(a(i, i));
  }
  expectNear(downloadTensor(dt, res->stream()), want);
}

// energy.cpp's "qpij,baij,pqab->ij": three rank-4 operands, four labels
// contracted (p, q, a, b), each operand reading them in a different order.
TEST(EinsumNewTests, FourIndexThreeOperandContraction) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  constexpr int64_t np = 4, nq = 3, ni = 5, nj = 2, na = 6, nb = 4;
  const Tensor x = patterned({nq, np, ni, nj}, 4);  // qpij
  const Tensor y = patterned({nb, na, ni, nj}, 5);  // baij
  const Tensor z = patterned({np, nq, na, nb}, 6);  // pqab
  const DeviceTensor dx = uploadTensor(x, *res);
  const DeviceTensor dy = uploadTensor(y, *res);
  const DeviceTensor dz = uploadTensor(z, *res);

  const double scale = -1.5;
  const DeviceTensor dout = deviceEinsumNew("qpij,baij,pqab->ij", {&dx, &dy, &dz}, scale, *res);
  EXPECT_EQ(dout.dims, (std::vector<int64_t>{ni, nj}));

  Expected want = expectedOfShape({ni, nj});
  for (int64_t i = 0; i < ni; ++i)
    for (int64_t j = 0; j < nj; ++j)
      for (int64_t p = 0; p < np; ++p)
        for (int64_t q = 0; q < nq; ++q)
          for (int64_t a = 0; a < na; ++a)
            for (int64_t b = 0; b < nb; ++b) {
              const double term = scale * x(q, p, i, j) * y(b, a, i, j) * z(p, q, a, b);
              want.value(i, j) += term;
              want.absSum(i, j) += std::abs(term);
            }
  expectNear(downloadTensor(dout, res->stream()), want);
}

// energy.cpp's "rsqp,rsba,pqba->rs": two output labels kept from two
// operands, four contracted -- the slab-walk shape, at 7 x 7 x 8 = 392 terms
// per element the largest contraction in this file.
TEST(EinsumNewTests, FourIndexSlabShape) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  constexpr int64_t nr = 3, ns = 5, np = 7, nq = 7, na = 8, nb = 1;
  const Tensor x = patterned({nr, ns, nq, np}, 7);  // rsqp
  const Tensor y = patterned({nr, ns, nb, na}, 8);  // rsba
  const Tensor z = patterned({np, nq, nb, na}, 9);  // pqba
  const DeviceTensor dx = uploadTensor(x, *res);
  const DeviceTensor dy = uploadTensor(y, *res);
  const DeviceTensor dz = uploadTensor(z, *res);

  const DeviceTensor dout = deviceEinsumNew("rsqp,rsba,pqba->rs", {&dx, &dy, &dz}, 1.0, *res);

  Expected want = expectedOfShape({nr, ns});
  for (int64_t r = 0; r < nr; ++r)
    for (int64_t s = 0; s < ns; ++s)
      for (int64_t p = 0; p < np; ++p)
        for (int64_t q = 0; q < nq; ++q)
          for (int64_t a = 0; a < na; ++a)
            for (int64_t b = 0; b < nb; ++b) {
              const double term = x(r, s, q, p) * y(r, s, b, a) * z(p, q, b, a);
              want.value(r, s) += term;
              want.absSum(r, s) += std::abs(term);
            }
  expectNear(downloadTensor(dout, res->stream()), want);
}

// The kernel honours strides, so a sliceAxis view along a non-leading axis is
// a valid operand (energy.cpp's tiles are such views).
TEST(EinsumNewTests, StridedSliceOperand) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  constexpr int64_t ni = 4, nj = 9, b0 = 2, b1 = 7, nk = 3;
  const Tensor a = patterned({ni, nj}, 10);
  const Tensor b = patterned({b1 - b0, nk}, 11);
  const DeviceTensor da = uploadTensor(a, *res);
  const DeviceTensor db = uploadTensor(b, *res);
  const DeviceTensor slice = sliceAxis(da, 1, b0, b1);  // a[:, 2:7]
  ASSERT_FALSE(isContiguous(slice));

  const DeviceTensor dc = deviceEinsumNew("ij,jk->ik", {&slice, &db}, 1.0, *res);

  Expected want = expectedOfShape({ni, nk});
  for (int64_t i = 0; i < ni; ++i)
    for (int64_t k = 0; k < nk; ++k)
      for (int64_t j = 0; j < b1 - b0; ++j) {
        const double term = a(i, b0 + j) * b(j, k);
        want.value(i, k) += term;
        want.absSum(i, k) += std::abs(term);
      }
  expectNear(downloadTensor(dc, res->stream()), want);
}

TEST(EinsumNewTests, TransposeNewPermutesAndScales) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  const Tensor x = patterned({2, 3, 4}, 12);
  const DeviceTensor dx = uploadTensor(x, *res);
  const DeviceTensor dt = deviceTransposeNew(dx, {2, 0, 1}, 2.0, *res);
  EXPECT_EQ(dt.dims, (std::vector<int64_t>{4, 2, 3}));

  Expected want = expectedOfShape({4, 2, 3});
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j)
      for (int64_t k = 0; k < 4; ++k) {
        want.value(k, i, j) = 2.0 * x(i, j, k);
        want.absSum(k, i, j) = std::abs(2.0 * x(i, j, k));
      }
  expectNear(downloadTensor(dt, res->stream()), want);
}

// --- EinsumAccumTests -----------------------------------------------------------

// `out` starts non-zero, then takes two accumulations with different scales:
// the result is prior + s1 * A.B + s2 * A.B, so a kernel that overwrote, or
// ignored `scale`, fails.
TEST(EinsumAccumTests, AccumulatesOntoPriorContentsWithScale) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  constexpr int64_t ni = 6, nj = 5, nk = 4;
  const Tensor a = patterned({ni, nj}, 13);
  const Tensor b = patterned({nj, nk}, 14);
  const Tensor prior = patterned({ni, nk}, 15);
  const DeviceTensor da = uploadTensor(a, *res);
  const DeviceTensor db = uploadTensor(b, *res);
  const DeviceTensor dout = uploadTensor(prior, *res);

  constexpr double s1 = 0.5, s2 = -3.0;
  deviceEinsumAccum("ij,jk->ik", {&da, &db}, dout, s1, res->stream());
  deviceEinsumAccum("ij,jk->ik", {&da, &db}, dout, s2, res->stream());

  Expected want = expectedOfShape({ni, nk});
  for (int64_t i = 0; i < ni; ++i)
    for (int64_t k = 0; k < nk; ++k) {
      want.value(i, k) = prior(i, k);
      want.absSum(i, k) = std::abs(prior(i, k));
      for (double s : {s1, s2})
        for (int64_t j = 0; j < nj; ++j) {
          const double term = s * a(i, j) * b(j, k);
          want.value(i, k) += term;
          want.absSum(i, k) += std::abs(term);
        }
    }
  expectNear(downloadTensor(dout, res->stream()), want);
}

// energy.cpp's `x -= einsum(...)` idiom: scale = -1 on a 1-operand,
// no-contraction "mn->mn" subtracts the operand. Accumulating a tensor's own
// copy into itself this way leaves (to rounding) zero.
TEST(EinsumAccumTests, ScaleMinusOneSubtracts) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  const Tensor x = patterned({7, 3}, 16);
  const Tensor y = patterned({7, 3}, 17);
  const DeviceTensor dx = uploadTensor(x, *res);
  const DeviceTensor dy = uploadTensor(y, *res);
  deviceEinsumAccum("mn->mn", {&dy}, dx, -1.0, res->stream());

  Expected want = expectedOfShape({7, 3});
  for (int64_t i = 0; i < x.size(); ++i) {
    want.value.flatRef(i) = x.flat(i) - y.flat(i);
    want.absSum.flatRef(i) = std::abs(x.flat(i)) + std::abs(y.flat(i));
  }
  expectNear(downloadTensor(dx, res->stream()), want);
}

// energy.cpp's "mjjn->mn": a label repeated within one operand (a diagonal)
// that is contracted, accumulated.
TEST(EinsumAccumTests, RepeatedLabelDiagonalContraction) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  constexpr int64_t nm = 3, nj = 5, nn = 4;
  const Tensor x = patterned({nm, nj, nj, nn}, 18);
  const Tensor prior = patterned({nm, nn}, 19);
  const DeviceTensor dx = uploadTensor(x, *res);
  const DeviceTensor dout = uploadTensor(prior, *res);
  deviceEinsumAccum("mjjn->mn", {&dx}, dout, 1.25, res->stream());

  Expected want = expectedOfShape({nm, nn});
  for (int64_t m = 0; m < nm; ++m)
    for (int64_t n = 0; n < nn; ++n) {
      want.value(m, n) = prior(m, n);
      want.absSum(m, n) = std::abs(prior(m, n));
      for (int64_t j = 0; j < nj; ++j) {
        const double term = 1.25 * x(m, j, j, n);
        want.value(m, n) += term;
        want.absSum(m, n) += std::abs(term);
      }
    }
  expectNear(downloadTensor(dout, res->stream()), want);
}

// --- TransposeAccumTests --------------------------------------------------------

// energy.cpp:199's dm1_h -= dm1^T, here with a non-trivial scale onto a
// non-square prior (so a missed transpose is a shape abort, and a transpose
// of the wrong axes reads the wrong element).
TEST(TransposeAccumTests, MatrixTransposeAccumulates) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  const Tensor x = patterned({4, 6}, 20);
  const Tensor prior = patterned({6, 4}, 21);
  const DeviceTensor dx = uploadTensor(x, *res);
  const DeviceTensor dout = uploadTensor(prior, *res);
  deviceTransposeAccum(dx, {1, 0}, dout, -0.5, res->stream());

  Expected want = expectedOfShape({6, 4});
  for (int64_t i = 0; i < 4; ++i)
    for (int64_t j = 0; j < 6; ++j) {
      want.value(j, i) = prior(j, i) - 0.5 * x(i, j);
      want.absSum(j, i) = std::abs(prior(j, i)) + std::abs(0.5 * x(i, j));
    }
  expectNear(downloadTensor(dout, res->stream()), want);
}

// make_a16's 6-index permutation {1,4,0,2,5,3} (energy.cpp:455), checked
// against the numpy convention the module documents: out.dim(i) ==
// x.dim(axes[i]) and out[j0..j5] == x[j with j_{axes[i]} = out index i].
// Distinct extents per axis, so any wrong axis is also a wrong shape.
TEST(TransposeAccumTests, SixIndexA16Permutation) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);

  const std::vector<int64_t> xd{2, 3, 4, 1, 5, 2};
  const std::vector<int> axes{1, 4, 0, 2, 5, 3};
  std::vector<int64_t> od(6);
  for (int i = 0; i < 6; ++i) od[i] = xd[axes[i]];

  const Tensor x = patterned(xd, 22);
  const Tensor prior = patterned(od, 23);
  const DeviceTensor dx = uploadTensor(x, *res);
  const DeviceTensor dout = uploadTensor(prior, *res);
  deviceTransposeAccum(dx, axes, dout, 1.0, res->stream());
  deviceTransposeAccum(dx, axes, dout, -2.0, res->stream());  // net scale -1

  Expected want = expectedOfShape(od);
  std::array<int64_t, 6> j{};
  for (j[0] = 0; j[0] < xd[0]; ++j[0])
    for (j[1] = 0; j[1] < xd[1]; ++j[1])
      for (j[2] = 0; j[2] < xd[2]; ++j[2])
        for (j[3] = 0; j[3] < xd[3]; ++j[3])
          for (j[4] = 0; j[4] < xd[4]; ++j[4])
            for (j[5] = 0; j[5] < xd[5]; ++j[5]) {
              const double v = x(j[0], j[1], j[2], j[3], j[4], j[5]);
              const auto o = [&](int i) { return j[axes[i]]; };
              const double p = prior(o(0), o(1), o(2), o(3), o(4), o(5));
              want.value(o(0), o(1), o(2), o(3), o(4), o(5)) = p + v - 2.0 * v;
              want.absSum(o(0), o(1), o(2), o(3), o(4), o(5)) =
                  std::abs(p) + std::abs(v) + std::abs(2.0 * v);
            }
  expectNear(downloadTensor(dout, res->stream()), want);
}

// --- EinsumDeathTest ------------------------------------------------------------
//
// A subscript string is a literal at every call site, so a malformed one -- or
// one that does not fit its operands -- is our bug: buildPlan goes through
// check(), which prints "error at file:line in function: <what>" for the
// CALLER's file:line (deviceEinsumNew's defaulted source_location) and
// std::abort()s. It does so before reading any operand's memory, so the
// operands are empty DeviceTensors with only their dims set.

DeviceTensor shapeOnly(std::vector<int64_t> dims) {
  DeviceTensor t;
  t.dims = std::move(dims);
  return t;
}

// Runs deviceEinsumNew in the death-test child, which creates the device
// resources itself. If it cannot, it exits with a message no regex below
// matches, so the death test fails rather than passing on the wrong abort.
void einsumInChild(const std::string& subscripts, std::vector<const DeviceTensor*> operands) {
  const auto& res = test::sharedResources();
  if (!res.has_value()) {
    std::fputs("no device in the death-test child\n", stderr);
    std::_Exit(3);
  }
  (void)deviceEinsumNew(subscripts, std::move(operands), 1.0, **res);
}

TEST(EinsumDeathTest, MissingArrowAborts) {
  const DeviceTensor a = shapeOnly({2, 3});
  const DeviceTensor b = shapeOnly({3, 4});
  EXPECT_DEATH(einsumInChild("ij,jk", {&a, &b}),
               "error at .*einsum_tests\\.cpp:[0-9]+ .*deviceEinsum: subscripts must use "
               "'->': ij,jk");
}

TEST(EinsumDeathTest, LabelCountNotOperandRankAborts) {
  const DeviceTensor a = shapeOnly({2, 3});
  EXPECT_DEATH(einsumInChild("ijk->i", {&a}),
               "deviceEinsum: operand 0 rank 2 != label length 3 \\(ijk->i\\)");
}

TEST(EinsumDeathTest, OperandCountNotLabelGroupsAborts) {
  const DeviceTensor a = shapeOnly({2, 3});
  EXPECT_DEATH(einsumInChild("ij,jk->ik", {&a}),
               "deviceEinsum: 2 operand label-groups but 1 operands given: ij,jk->ik");
}

TEST(EinsumDeathTest, InconsistentLabelExtentAborts) {
  const DeviceTensor a = shapeOnly({2, 3});
  const DeviceTensor b = shapeOnly({4, 5});
  EXPECT_DEATH(einsumInChild("ij,jk->ik", {&a, &b}),
               "deviceEinsum: inconsistent dim for label 'j' in ij,jk->ik");
}

TEST(EinsumDeathTest, OutputLabelInNoOperandAborts) {
  const DeviceTensor a = shapeOnly({2, 3});
  EXPECT_DEATH(einsumInChild("ij->iz", {&a}),
               "deviceEinsum: output label 'z' not in any operand \\(ij->iz\\)");
}

}  // namespace
}  // namespace nevpt2
