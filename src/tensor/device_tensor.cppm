// nevpt2.device_tensor -- a device-resident twin of nevpt2.tensor's Tensor: same shape/stride bookkeeping,
// but the data lives in device memory instead of a host std::vector. The
// currency of every device stage -- the RDM build, the einsums, the SC and PC
// energies, the DF slabs -- so dm3/f3ac/f3ca and the MO integrals stay
// GPU-resident end to end.
//
// A DeviceTensor OWNS its memory or VIEWS someone else's: it holds
// either an owning DeviceBuffer<double> (nevpt2.device_resources -- WarpWraps'
// buffer suite over DeviceResources) or a DeviceBufferView<double> into one. An owning
// tensor frees its buffer when it goes out of scope -- wwrFreeAsync on the
// demo's one stream, so after every kernel already queued there that reads it,
// with no host sync. A view frees nothing and must not outlive the tensor it
// views (every view in this tree is a slice used inside the owner's scope, or
// FullBlockSource's view of a block main() holds).
export module nevpt2.device_tensor;

import std;
// Re-exported: upload/download take and return a host Tensor.
export import nevpt2.tensor;
// Re-exported: allocation takes the DeviceResources whose pool and stream it
// draws on (and which the buffer co-owns); every copy takes the wwrStream_t it
// is issued on; DeviceBuffer / DeviceBufferView are what a tensor holds.
export import nevpt2.device_resources;

export namespace nevpt2 {

class DeviceTensor {
 public:
  std::vector<int64_t> dims;
  std::vector<int64_t> strides;  // in elements (not bytes); row-major unless a slice

  // Empty: no elements, nothing to free (an unset NevptIntegralsDevice field).
  DeviceTensor() = default;
  // Takes ownership of `buf`, laid out row-major as `dims`.
  DeviceTensor(DeviceBuffer<double> buf, std::vector<int64_t> dims);

  // A fresh, zeroed, owning tensor of shape `dims`, drawn from res's pool on
  // res.stream() (an output accumulator with no host-side initial value).
  // Aborts through narrowTo (naming `loc`) on a negative extent -- our bug.
  static DeviceTensor zeros(std::vector<int64_t> dims, const DeviceResources& res,
                            std::source_location loc = std::source_location::current());

  // Move-only, like the buffer it may own. A non-owning copy is view().
  DeviceTensor(DeviceTensor&&) noexcept = default;
  DeviceTensor& operator=(DeviceTensor&&) noexcept = default;
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  ~DeviceTensor() = default;

  // A non-owning DeviceTensor over the same memory, same dims and strides.
  DeviceTensor view() const;
  bool owning() const { return std::holds_alternative<DeviceBuffer<double>>(storage_); }

  // The device address of element 0. Constness is shallow, as it was for the
  // raw pointer this replaced: kernels write through a `const DeviceTensor&`
  // output (deviceEinsumAccum's `out`) -- const protects the shape, not the
  // data.
  double* data() const {
    return std::visit([](auto& b) -> double* { return b.data(); }, storage_);
  }

  int64_t rank() const { return std::ssize(dims); }
  int64_t size() const {
    int64_t n = 1;
    for (int64_t d : dims) n *= d;
    return n;
  }

 private:
  friend DeviceTensor sliceAxis(const DeviceTensor& t, int64_t axis, int64_t b0, int64_t b1,
                                std::source_location loc);
  DeviceTensor(DeviceBufferView<double> v, std::vector<int64_t> dims, std::vector<int64_t> strides);

  // A view first: it is what a default-constructed (empty) tensor holds.
  // `mutable` for the shallow constness above, and because WarpWraps' view
  // constructor takes its source by non-const reference.
  mutable std::variant<DeviceBufferView<double>, DeviceBuffer<double>> storage_;
};

// Everything below is stream-ordered on the demo's one non-blocking stream:
// an allocation draws from DeviceResources' pool on its stream and
// zero-fills there, a free (an owning tensor's destructor) returns it on the
// same stream, and every copy is enqueued on the caller's `stream` -- always
// res.stream() in this tree. So a tensor may be destroyed while kernels that
// read it are still queued: the free runs after them.

// Allocates a fresh device buffer and enqueues the upload of `host`'s data
// into it on res.stream(). `host` is pageable memory, which the runtime
// stages before the call returns, so the caller may release it straight away.
// (The buffer is zero-filled first and then overwritten -- the suite always
// zero-fills; docs/performance.md, "The memory pool", measures what that costs.)
DeviceTensor uploadTensor(const Tensor& host, const DeviceResources& res,
                          std::source_location loc = std::source_location::current());

// A view of `t` restricted to [b0, b1) along `axis`: same memory (offset),
// same strides, one dim shortened. No allocation, no ownership. The einsum
// kernel honours strides, so a view along a non-leading axis is a valid
// operand; but it is not contiguous, so downloadTensor (a flat memcpy) must
// not be called on one -- see isContiguous. WarpWraps bounds-checks the view
// against `t`'s memory and aborts (naming `loc`) if it would run past it.
DeviceTensor sliceAxis(const DeviceTensor& t, int64_t axis, int64_t b0, int64_t b1,
                       std::source_location loc = std::source_location::current());

// True when `t`'s strides are exactly row-major for its dims.
bool isContiguous(const DeviceTensor& t);

// A view of contiguous `t` read as row-major `dims` (same element count): no
// allocation, no ownership, no copy -- numpy's reshape of a C-contiguous
// array. Aborts through check() (naming `loc`) if `t` is not contiguous or
// the counts differ -- our bug, never bad input.
// Used by the PC solve, which works on (d, d) matrices whose rows the slab
// einsums address by their basis indices, (p, q).
DeviceTensor reshapeView(const DeviceTensor& t, std::vector<int64_t> dims,
                         std::source_location loc = std::source_location::current());

// Reads a device tensor back into a host Tensor of the same shape. The copy
// is enqueued on `stream` behind whatever wrote `t`, and the stream is
// synchronized before the host reads the result.
Tensor downloadTensor(const DeviceTensor& t, wwrStream_t stream);

// n x n identity, uploaded -- the device twin of Tensor::eye. Built host-side
// and uploaded rather than a dedicated kernel: it's at most norb x norb
// (<=256 elements here), negligible next to the norb^6-scale contractions
// this whole file exists for.
DeviceTensor deviceEye(int64_t n, const DeviceResources& res,
                       std::source_location loc = std::source_location::current());

}  // namespace nevpt2
