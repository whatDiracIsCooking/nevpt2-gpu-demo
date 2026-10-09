// nevpt2.rdm_build:staging -- host-to-device staging and the per-tile device
// accumulate, both on the build's one stream.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :staging;` and from no importer of nevpt2.rdm_build.
// (Used by rdm_build.cpp.)
module;

// What rdm/rdm_accumulate_bridge.h pulls in, textually in the GMF; the
// bridge itself is included after the imports.
#include <runtime.h>

module nevpt2.rdm_build:staging;

import std;
import nevpt2.rdm_build;
import wwr.runtime_api;

// The kernel launchers it calls: extern "C++" attaches them to the global
// module, so they bind to the .cu's definitions (see
// rdm/rdm_accumulate_bridge.h).
extern "C++" {
#include "rdm/rdm_accumulate_bridge.h"
}

namespace nevpt2 {

// Every allocation, copy, zero-fill and launch in buildRdmsDevice is enqueued
// on the build's one stream, res.stream(). Every device buffer is a
// DeviceBuffer: drawn from res's pool on that stream, zero-filled
// there (exact for f64: all-zero bytes is +0.0), and freed there when it goes
// out of scope. The host vectors are pageable, so the runtime stages them
// before an upload returns. `loc` is the caller's line, so an out-of-memory
// abort names the upload in buildRdmsDevice that asked.
template <class T>
DeviceBuffer<T> upload(const std::vector<T>& h, const DeviceResources& res,
                       std::source_location loc = std::source_location::current()) {
  DeviceBuffer<T> d(h.size(), res.shared_from_this(), loc);
  if (!h.empty()) {
    gpuCheck(wwrMemcpyAsync(d.data(), h.data(), h.size() * sizeof(T), wwrMemcpyHostToDevice,
                            res.stream()));
  }
  return d;
}

// out[i] += part[i], on device -- the device-resident RDM build's per-tile
// accumulate, replacing a download+host-add+discard round trip. A linked-in
// kernel (rdm_accumulate.cu), called through its bridge like every kernel here.
// On the build's stream, like every other launch here.
void accumulateInto(double* acc, const double* part, int64_t n, wwrStream_t s) {
  gpuCheck(device::accumulateInplace(s, acc, part, n));
}

}  // namespace nevpt2
