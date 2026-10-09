// nevpt2.device_tensor:row_major -- row-major strides and element counts for a
// DeviceTensor shape.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :row_major;` and from no importer of nevpt2.device_tensor.
// (Used by device_tensor.cpp.)
module nevpt2.device_tensor:row_major;

import std;
import nevpt2.device_tensor;

namespace nevpt2 {

std::vector<int64_t> rowMajorStrides(const std::vector<int64_t>& dims) {
  std::vector<int64_t> strides(dims.size());
  int64_t s = 1;
  for (int64_t i = std::ssize(dims) - 1; i >= 0; --i) {
    strides[i] = s;
    s *= dims[i];
  }
  return strides;
}

int64_t elementCount(const std::vector<int64_t>& dims) {
  int64_t n = 1;
  for (int64_t d : dims) n *= d;
  return n;
}

}  // namespace nevpt2
