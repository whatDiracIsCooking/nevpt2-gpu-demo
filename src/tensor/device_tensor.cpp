module nevpt2.device_tensor;

import std;
import :row_major;
import nevpt2.wwr;

namespace nevpt2 {

DeviceTensor::DeviceTensor(DeviceBuffer<double> buf, std::vector<int64_t> d)
    : dims(std::move(d)), strides(rowMajorStrides(dims)), storage_(std::move(buf)) {}

DeviceTensor::DeviceTensor(DeviceBufferView<double> v, std::vector<int64_t> d,
                           std::vector<int64_t> s)
    : dims(std::move(d)), strides(std::move(s)), storage_(std::move(v)) {}

DeviceTensor DeviceTensor::view() const {
  return DeviceTensor(
      std::visit([](auto& b) { return DeviceBufferView<double>(b); }, storage_), dims,
      strides);
}

DeviceTensor uploadTensor(const Tensor& host, const DeviceResources& res,
                          std::source_location loc) {
  // A host Tensor's size is its std::vector's, so it is never negative.
  const std::size_t n = host.data().size();
  DeviceTensor t(DeviceBuffer<double>(n, res.shared_from_this(), loc), host.dims());
  if (n > 0) {
    gpuCheck(wwrMemcpyAsync(t.data(), host.data().data(), n * sizeof(double), wwrMemcpyHostToDevice,
                            res.stream()));
  }
  return t;
}

DeviceTensor sliceAxis(const DeviceTensor& t, int64_t axis, int64_t b0, int64_t b1,
                       std::source_location loc) {
  std::vector<int64_t> dims = t.dims;
  dims[axis] = b1 - b0;
  // The view spans from the slice's first element to its last: 1 + sum over
  // axes of (extent - 1) * stride. Empty if any extent is 0. Computed in
  // int64_t and narrowed to the view's std::size_t once, so a b0 or b1 out of
  // order (a negative offset or extent) aborts here, naming `loc`, instead of
  // wrapping to a huge unsigned count.
  int64_t offset = 0, count = 0;
  if (elementCount(dims) > 0) {
    offset = b0 * t.strides[axis];
    count = 1;
    for (int64_t i = 0; i < std::ssize(dims); ++i) count += (dims[i] - 1) * t.strides[i];
  }
  const std::size_t viewOffset = narrowTo<std::size_t>(offset, "sliceAxis offset", loc);
  const std::size_t viewCount = narrowTo<std::size_t>(count, "sliceAxis extent", loc);
  DeviceBufferView<double> v = std::visit(
      [&](auto& b) { return DeviceBufferView<double>(b, viewOffset, viewCount, loc); },
      t.storage_);
  return DeviceTensor(std::move(v), std::move(dims), t.strides);
}

bool isContiguous(const DeviceTensor& t) { return t.strides == rowMajorStrides(t.dims); }

DeviceTensor reshapeView(const DeviceTensor& t, std::vector<int64_t> dims,
                         std::source_location loc) {
  if (!isContiguous(t) || elementCount(dims) != t.size())
    check(false,
          std::format("reshapeView: the tensor is {} and has {} elements, the new shape {}",
                      isContiguous(t) ? "contiguous" : "NOT contiguous", t.size(),
                      elementCount(dims)),
          loc);
  DeviceTensor v = t.view();
  v.strides = rowMajorStrides(dims);
  v.dims = std::move(dims);
  return v;
}

DeviceTensor DeviceTensor::zeros(std::vector<int64_t> dims, const DeviceResources& res,
                                 std::source_location loc) {
  // Zeroed by the buffer itself (wwrMemsetAsync on res.stream()): exact, since
  // all-zero bytes is the IEEE-754 bit pattern for +0.0.
  const std::size_t n = narrowTo<std::size_t>(elementCount(dims), "a negative extent", loc);
  return DeviceTensor(DeviceBuffer<double>(n, res.shared_from_this(), loc), std::move(dims));
}

Tensor downloadTensor(const DeviceTensor& t, wwrStream_t stream) {
  std::vector<double> host(narrowTo<std::size_t>(t.size(), "a negative extent"));
  if (host.empty()) return Tensor::fromFlat(t.dims, std::move(host));
  gpuCheck(wwrMemcpyAsync(host.data(), t.data(), host.size() * sizeof(double),
                          wwrMemcpyDeviceToHost, stream));
  // The host reads `host` next, so wait for the copy (and everything queued
  // before it on `stream`) to land. An async copy into pageable memory may
  // return before it has.
  gpuCheck(wwrStreamSynchronize(stream));
  return Tensor::fromFlat(t.dims, std::move(host));
}

DeviceTensor deviceEye(int64_t n, const DeviceResources& res, std::source_location loc) {

  return uploadTensor(Tensor::eye(n), res, loc);
}

}  // namespace nevpt2
