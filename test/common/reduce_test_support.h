// reduce_test_support.h -- what the warp_reduce and block_reduce suites share:
// the process's DeviceResources, host <-> device copies on its
// one stream, the exact input pattern and its poison, and the every-lane
// comparison.
//
// Included by the host test TUs AFTER their imports (it uses DeviceBuffer,
// gpuCheck and the wwr* names nevpt2.wwr exports), so it is plain text pasted
// into a non-module TU -- no module face of its own.
#pragma once

namespace nevpt2::test {

// The process's DeviceResources, or nullptr after recording the creation
// Error as this test's failure (callers ASSERT_NE on it).
inline const DeviceResources* resourcesOrFail() {
  const auto& res = sharedResources();
  if (!res.has_value()) {
    ADD_FAILURE() << "DeviceResources::create failed: " << kindName(res.error().kind) << ": "
                  << res.error().message;
    return nullptr;
  }
  return res->get();
}

// A device copy of `host`, drawn from res's pool and filled on its stream.
template <typename T>
DeviceBuffer<T> upload(const std::vector<T>& host, const DeviceResources& res) {
  DeviceBuffer<T> buf(host.size(), res.shared_from_this());
  gpuCheck(wwrMemcpyAsync(buf.data(), host.data(), buf.size_bytes(), wwrMemcpyHostToDevice,
                          res.stream()));
  return buf;
}

// A whole DeviceBuffer read back on res's stream.
template <typename T>
std::vector<T> download(const DeviceBuffer<T>& buf, const DeviceResources& res) {
  std::vector<T> host(buf.num_elements());
  gpuCheck(wwrMemcpyAsync(host.data(), buf.data(), buf.size_bytes(), wwrMemcpyDeviceToHost,
                          res.stream()));
  // The host reads `host` next; an async copy into pageable memory may return
  // before it has landed (and the kernels before it must have finished).
  gpuCheck(wwrStreamSynchronize(res.stream()));
  return host;
}

// The active input of fold `f`, element `t`: a small integer, so any order of
// summing a fold's at most 1024 of them is EXACT in double, and the suites can
// compare sums with ==. Varies with f so neighbouring folds differ.
inline double patternValue(const unsigned int f, const unsigned int t) {
  return static_cast<double>(static_cast<int>((f * 37u + t * 11u) % 29u) - 14);
}

// What an inactive lane / thread holds: a value that, folded in, shows. NaN
// survives both + and max_nan.
inline constexpr double kPoison = std::numeric_limits<double>::quiet_NaN();

// `nfolds` folds of `width` elements; fold f has nactive[f] active elements
// set by value(f, t), the rest kPoison.
template <typename T, typename Value>
std::vector<T> foldInputs(const std::vector<unsigned int>& nactive, const unsigned int width,
                          const T poison, const Value& value) {
  std::vector<T> in(nactive.size() * width, poison);
  for (unsigned int f = 0; f < nactive.size(); ++f) {
    for (unsigned int t = 0; t < nactive[f]; ++t) in[f * width + t] = value(f, t);
  }
  return in;
}

// The serial left-to-right fold of each fold's active inputs, broadcast to
// every element of the fold: what every lane / thread must hold.
template <typename T, typename Op>
std::vector<T> serialFolds(const std::vector<T>& in, const std::vector<unsigned int>& nactive,
                           const unsigned int width, const Op& op) {
  std::vector<T> want(in.size());
  for (unsigned int f = 0; f < nactive.size(); ++f) {
    T acc = in[f * width];
    for (unsigned int t = 1; t < nactive[f]; ++t) acc = op(acc, in[f * width + t]);
    for (unsigned int t = 0; t < width; ++t) want[f * width + t] = acc;
  }
  return want;
}

// nactive = 1, 2, ..., width: one fold per partial (and the full) tile.
inline std::vector<unsigned int> everyPartialTile(const unsigned int width) {
  std::vector<unsigned int> n(width);
  for (unsigned int f = 0; f < width; ++f) n[f] = f + 1;
  return n;
}

// Exact per-element equality (two NaNs count as equal), reported as the first
// few mismatches by fold and lane plus a count, so a broken 1024 x 1024 run
// does not print a million lines.
template <typename T>
void expectEveryLane(const std::vector<T>& got, const std::vector<T>& want,
                     const unsigned int width, const std::string_view what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  std::size_t bad = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    bool same = got[i] == want[i];
    if constexpr (std::is_floating_point_v<T>) {
      same = same || (std::isnan(got[i]) && std::isnan(want[i]));
    }
    if (same) continue;
    if (++bad <= 5) {
      ADD_FAILURE() << what << ": fold " << i / width << " lane " << i % width << " got "
                    << got[i] << ", want " << want[i];
    }
  }
  EXPECT_EQ(bad, 0u) << what << ": mismatched lanes";
}

}  // namespace nevpt2::test
