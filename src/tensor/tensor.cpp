module nevpt2.tensor;

import std;
import nevpt2.error_handling;  // check()

namespace nevpt2 {

Tensor transpose(const Tensor& x, const std::vector<int>& axes, std::source_location loc) {
  const int64_t n = x.rank();
  check(std::ssize(axes) == n, "transpose: axes length != rank", loc);
  std::vector<int64_t> outDims(n);
  for (int64_t i = 0; i < n; ++i) outDims[i] = x.dim(axes[i]);
  Tensor out(outDims);

  // inv[axes[i]] = i, so that in_idx[m] = out_idx[inv[m]] (see tensor/tensor.cppm's
  // numpy-transpose convention docstring).
  std::vector<int64_t> inv(n);
  for (int64_t i = 0; i < n; ++i) inv[axes[i]] = i;

  std::vector<int64_t> xStrides(n);
  for (int64_t i = 0; i < n; ++i) xStrides[i] = x.stride(i);

  std::vector<int64_t> outIdx(n, 0);
  const int64_t total = out.size();
  for (int64_t flat = 0; flat < total; ++flat) {
    // decode flat -> outIdx (row-major, matching Tensor's own layout)
    int64_t rem = flat;
    for (int64_t i = n - 1; i >= 0; --i) {
      outIdx[i] = rem % outDims[i];
      rem /= outDims[i];
    }
    int64_t xOff = 0;
    for (int64_t m = 0; m < n; ++m) xOff += outIdx[inv[m]] * xStrides[m];
    out.flatRef(flat) = x.flat(xOff);
  }
  return out;
}

}  // namespace nevpt2
