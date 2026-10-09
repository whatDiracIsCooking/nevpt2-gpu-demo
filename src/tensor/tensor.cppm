// nevpt2.tensor -- a small dense host N-d tensor: the container the golden file's arrays are
// read into (golden/golden.cppm), the host side of tensor/device_tensor.cppm's
// upload/download (and the shape/stride bookkeeping DeviceTensor mirrors), and
// the operand type for the few host-side index manipulations still done off the
// GPU (rdm/rdm_build.cpp's one h2e transpose to chemists' order;
// energy/energy.cpp symmetrizing and triangle-packing downloaded per-slab
// results before normToEnergy).
// The generic host `einsum` this type was originally built around -- to port the
// NumPy reference's np.einsum calls into a checkable all-host reference -- is
// gone with that reference, and with it the whole-tensor arithmetic operators
// it used; the device contraction (device_einsum.cu) is now the only einsum.
// Only `transpose` remains here, still needed by rdm_build.cpp.
export module nevpt2.tensor;

import std;
// Re-exported: every extent, stride, rank and element count here is an int64_t,
// and narrowTo is how a caller hands one to a narrower API.
export import nevpt2.common;
import nevpt2.error_handling;  // check(): fromFlat's and transpose's invariants

export namespace nevpt2 {

class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(std::vector<int64_t> dims) : dims_(std::move(dims)) {
    strides_.resize(dims_.size());
    int64_t s = 1;
    for (int64_t i = std::ssize(dims_) - 1; i >= 0; --i) {
      strides_[i] = s;
      s *= dims_[i];
    }
    data_.assign(narrowTo<std::size_t>(s, "a negative tensor extent"), 0.0);
  }

  // A data size that does not match the shape is our bug (check(), the abort
  // tier, at the caller's file:line): golden/golden.cpp and downloadTensor
  // size `data` from the very shape they pass, and the other caller
  // (energy_pc.cpp) passes literals.
  static Tensor fromFlat(std::vector<int64_t> dims, std::vector<double> data,
                         std::source_location loc = std::source_location::current()) {
    Tensor t(std::move(dims));
    if (data.size() != t.data_.size())
      check(false,
            std::format("Tensor::fromFlat: data size {} does not match shape product {}",
                        data.size(), t.data_.size()),
            loc);
    t.data_ = std::move(data);
    return t;
  }

  // n x n identity -- what deviceEye uploads as the `delta` the class-energy
  // formulas use throughout.
  static Tensor eye(int64_t n) {
    Tensor t({n, n});
    for (int64_t i = 0; i < n; ++i) t(i, i) = 1.0;
    return t;
  }

  int64_t rank() const { return std::ssize(dims_); }
  int64_t dim(int64_t axis) const { return dims_[axis]; }
  const std::vector<int64_t>& dims() const { return dims_; }
  int64_t stride(int64_t axis) const { return strides_[axis]; }
  int64_t size() const { return std::ssize(data_); }
  const std::vector<double>& data() const { return data_; }

  double flat(int64_t off) const { return data_[off]; }
  double& flatRef(int64_t off) { return data_[off]; }

  template <typename... Idx>
  double& operator()(Idx... idx) {
    return data_[offset(idx...)];
  }
  template <typename... Idx>
  double operator()(Idx... idx) const {
    return data_[offset(idx...)];
  }

 private:
  // Each index is braced into an int64_t: a signed index of any width up to 64
  // bits widens, and an unsigned one is a compile error (a narrowing
  // conversion) rather than a silent reinterpretation.
  template <typename... Idx>
  int64_t offset(Idx... idx) const {
    const int64_t a[] = {int64_t{idx}...};
    int64_t off = 0;
    for (std::size_t k = 0; k < sizeof...(idx); ++k) off += a[k] * strides_[k];
    return off;
  }

  std::vector<int64_t> dims_;
  std::vector<int64_t> strides_;
  std::vector<double> data_;
};

// A copy-and-permute transpose, numpy-`X.transpose(axes)`-alike:
// result.dim(i) == X.dim(axes[i]), and result[j0,...] == X[axes-permuted j].
// `axes` is a literal at every call site, so a length != x.rank() is our bug:
// check() aborts, naming the caller's file:line.
Tensor transpose(const Tensor& x, const std::vector<int>& axes,
                 std::source_location loc = std::source_location::current());

}  // namespace nevpt2
