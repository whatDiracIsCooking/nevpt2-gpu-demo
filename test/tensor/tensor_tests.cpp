// Suites for nevpt2.tensor's host Tensor.
//
// Host-only ON PURPOSE: Tensor is a host container -- shape, strides and a
// std::vector<double> -- so this binary is not REQUIRES_GPU and runs with no
// card visible. It checks index bookkeeping, not energies: the golden tier owns
// every numerical claim.
//
//   TensorShapeTests  fromFlat keeps the data in order; rank / dim / dims /
//                     size; row-major strides (last axis fastest)
//   TensorIndexTests  operator() (both overloads), flat and the strides agree
//                     on every element's offset
//   TensorDeathTest   fromFlat with a data size that does not match the shape
//                     aborts with the caller's file:line and both sizes
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.tensor;

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
// One binary per suite file, so a name clash here is a link error.
namespace nevpt2::test::tensor {

// 0, 1, 2, ... n-1: each element's value is its own flat offset, so an index
// test can read the expected offset straight back out of the tensor.
std::vector<double> iota(int64_t n) {
  std::vector<double> v(narrowTo<std::size_t>(n));
  std::iota(v.begin(), v.end(), 0.0);
  return v;
}

// --- TensorShapeTests ---------------------------------------------------------

TEST(TensorShapeTests, FromFlatKeepsShapeAndData) {
  const std::vector<double> data = iota(24);
  const Tensor t = Tensor::fromFlat({2, 3, 4}, data);
  EXPECT_EQ(t.rank(), 3);
  EXPECT_EQ(t.dims(), (std::vector<int64_t>{2, 3, 4}));
  EXPECT_EQ(t.dim(0), 2);
  EXPECT_EQ(t.dim(1), 3);
  EXPECT_EQ(t.dim(2), 4);
  EXPECT_EQ(t.size(), 24);
  EXPECT_EQ(t.data(), data);
}

TEST(TensorShapeTests, StridesAreRowMajor) {
  const Tensor t = Tensor::fromFlat({2, 3, 4, 5}, iota(120));
  EXPECT_EQ(t.stride(3), 1);
  EXPECT_EQ(t.stride(2), 5);
  EXPECT_EQ(t.stride(1), 20);
  EXPECT_EQ(t.stride(0), 60);
}

TEST(TensorShapeTests, RankOneAndRankZero) {
  const Tensor v = Tensor::fromFlat({7}, iota(7));
  EXPECT_EQ(v.rank(), 1);
  EXPECT_EQ(v.dim(0), 7);
  EXPECT_EQ(v.stride(0), 1);
  EXPECT_EQ(v.size(), 7);

  // An empty shape is a scalar: the shape product is 1.
  const Tensor s = Tensor::fromFlat({}, {3.5});
  EXPECT_EQ(s.rank(), 0);
  EXPECT_EQ(s.size(), 1);
  EXPECT_EQ(s.flat(0), 3.5);
}

TEST(TensorShapeTests, ZeroExtentHoldsNoData) {
  const Tensor t = Tensor::fromFlat({3, 0, 2}, {});
  EXPECT_EQ(t.rank(), 3);
  EXPECT_EQ(t.dim(1), 0);
  EXPECT_EQ(t.size(), 0);
}

TEST(TensorShapeTests, ShapeConstructorZeroFills) {
  const Tensor t({2, 3});
  EXPECT_EQ(t.size(), 6);
  EXPECT_EQ(t.data(), std::vector<double>(6, 0.0));
}

// --- TensorIndexTests ---------------------------------------------------------

TEST(TensorIndexTests, CallOperatorFlatAndStridesAgree) {
  const Tensor t = Tensor::fromFlat({2, 3, 4}, iota(24));
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j)
      for (int64_t k = 0; k < 4; ++k) {
        const int64_t off = i * t.stride(0) + j * t.stride(1) + k * t.stride(2);
        EXPECT_EQ(off, (i * 3 + j) * 4 + k);
        EXPECT_EQ(t(i, j, k), static_cast<double>(off)) << i << "," << j << "," << k;
        EXPECT_EQ(t.flat(off), t(i, j, k));
      }
}

TEST(TensorIndexTests, NarrowerSignedIndicesWiden) {
  const Tensor t = Tensor::fromFlat({3, 5}, iota(15));
  const int i = 2;
  const short j = 4;
  EXPECT_EQ(t(i, j), 14.0);
  EXPECT_EQ(t(int64_t{1}, 0), 5.0);
}

TEST(TensorIndexTests, MutableCallOperatorWritesTheSameOffset) {
  Tensor t({2, 3});
  t(1, 2) = 42.0;
  EXPECT_EQ(t.flat(1 * 3 + 2), 42.0);
  t.flatRef(0 * 3 + 1) = -1.0;
  EXPECT_EQ(t(0, 1), -1.0);
  // Nothing else moved.
  EXPECT_EQ(t.data(), (std::vector<double>{0.0, -1.0, 0.0, 0.0, 0.0, 42.0}));
}

// --- TensorDeathTest ----------------------------------------------------------
//
// A data size that does not match the shape product is our bug: fromFlat goes
// through check(), which prints "error at file:line in function: <what>" for
// the CALLER's file:line and std::abort()s.

TEST(TensorDeathTest, TooFewElementsAborts) {
  EXPECT_DEATH((void)Tensor::fromFlat({2, 3}, iota(5)),
               "error at .*tensor_tests\\.cpp:[0-9]+ .*Tensor::fromFlat: data size 5 does not "
               "match shape product 6");
}

TEST(TensorDeathTest, TooManyElementsAborts) {
  EXPECT_DEATH((void)Tensor::fromFlat({4}, iota(5)),
               "Tensor::fromFlat: data size 5 does not match shape product 4");
}

TEST(TensorDeathTest, DataForAZeroExtentShapeAborts) {
  EXPECT_DEATH((void)Tensor::fromFlat({3, 0}, {1.0}),
               "Tensor::fromFlat: data size 1 does not match shape product 0");
}

}  // namespace nevpt2::test::tensor
