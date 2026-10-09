// nevpt2.app.sanitizer_canary:canaries -- the planted bugs, one per mode: two host ones for
// ASan/UBSan and the device ones for compute-sanitizer.
// An interface partition that exports nothing: main.cppm is the primary
// interface, which may import only interface partitions and must re-export
// them ([module.unit]), so it does `export import :canaries;` -- and since no
// declaration here is `export`ed, every one keeps module linkage, reachable
// from main.cppm and from no importer (nothing imports a demo anyway).
module;

#include <cstdio>  // stderr: a macro, which `import std` does not carry

#include <runtime.h>  // what canary_bridge.h pulls in, textually before the imports

export module nevpt2.app.sanitizer_canary:canaries;

import std;
import nevpt2.wwr;
import nevpt2.device_resources;  // device-pool-free-before-read: the demos' pool

// The kernel launchers it calls: extern "C++" attaches them to the global
// module, so they bind to the .cu's definitions (see
// rdm/rdm_accumulate_bridge.h).
extern "C++" {
#include "canary_bridge.h"
}

// Doubles per device buffer (512 bytes): one per thread of the 64-thread
// race/barrier kernels, so their only bug is the planted one.
constexpr int kN = 64;

constexpr std::size_t kBytes = kN * sizeof(double);

double* deviceDoubles(wwr::wwrStream_t s) {
  void* p = nullptr;
  nevpt2::gpuCheck(wwr::wwrMallocAsync(&p, kBytes, s));
  return static_cast<double*>(p);
}

int hostHeapOverflow() {
  volatile int past = kN;  // volatile: the compiler cannot see the overflow
  const int i = past;      // one volatile read; std::format takes no volatile
  double* a = new double[kN]();
  double v = a[i];
  delete[] a;
  std::print("read a[{}] = {:g}\n", i, v);
  return 0;
}

int hostSignedOverflow() {
  volatile int big = std::numeric_limits<int>::max();
  int r = big + 1;
  std::printf("INT_MAX + 1 = %d\n", r);
  return 0;
}

// device-free-before-read again, but through the demos' own allocator: a
// DeviceBuffer drawn from DeviceResources' pool (wwrMallocFromPoolAsync) at
// the demos' release threshold, which keeps every freed byte in the pool, so
// the freed range stays mapped. `out` is drawn first, so the pool cannot hand
// the freed block straight back to it.
int devicePoolFreeBeforeRead() {
  nevpt2::Result<std::shared_ptr<nevpt2::DeviceResources>> created =
      nevpt2::DeviceResources::create(0, nevpt2::kDefaultPoolReleaseThreshold);
  if (!created) {
    nevpt2::report(created.error());
    return 2;
  }
  const std::shared_ptr<nevpt2::DeviceResources> res = *std::move(created);
  nevpt2::DeviceBuffer<double> out(kN, res);
  const double* stale = nullptr;
  {
    const nevpt2::DeviceBuffer<double> in(kN, res);  // zero-filled on the stream
    stale = in.data();
  }  // ~DeviceBuffer: wwrFreeAsync on res->stream(), before the read below
  nevpt2::gpuCheck(nevpt2::canary::launchCopy(res->stream(), stale, out.data(), kN));
  // Kernel faults surface here (and abort through gpuCheck).
  nevpt2::gpuCheck(wwr::wwrStreamSynchronize(res->stream()));
  return 0;
}

int device(std::string_view mode) {
  if (mode == "device-pool-free-before-read") return devicePoolFreeBeforeRead();

  wwr::wwrStream_t s = nullptr;
  nevpt2::gpuCheck(wwr::wwrStreamCreateWithFlags(&s, wwr::wwrStreamNonBlocking));
  double* out = deviceDoubles(s);
  nevpt2::gpuCheck(wwr::wwrMemsetAsync(out, 0, kBytes, s));

  if (mode == "device-oob") {
    double* in = deviceDoubles(s);
    nevpt2::gpuCheck(wwr::wwrMemsetAsync(in, 0, kBytes, s));
    nevpt2::gpuCheck(nevpt2::canary::launchReadPastEnd(s, in, out, kN));
    nevpt2::gpuCheck(wwr::wwrFreeAsync(in, s));
  } else if (mode == "device-free-before-read") {
    // The buffer is written and then freed on S, and only THEN is the kernel
    // that reads it enqueued on S: in stream order the read follows the free.
    // That is the deterministic form of the bug -- the racy form (a free on S
    // while a kernel on another stream still reads) is caught only when the
    // interleaving happens under the tool (docs/testing.md, "Sanitizers").
    double* in = deviceDoubles(s);
    nevpt2::gpuCheck(wwr::wwrMemsetAsync(in, 0, kBytes, s));
    nevpt2::gpuCheck(wwr::wwrFreeAsync(in, s));
    nevpt2::gpuCheck(nevpt2::canary::launchCopy(s, in, out, kN));
  } else if (mode == "device-leak") {
    // Allocated on S and never freed: a leak memcheck reports only with
    // --leak-check full (on by default under memcheck).
    (void)deviceDoubles(s);
  } else if (mode == "device-uninit-read") {
    double* in = deviceDoubles(s);  // never written
    nevpt2::gpuCheck(nevpt2::canary::launchCopy(s, in, out, kN));
    nevpt2::gpuCheck(wwr::wwrFreeAsync(in, s));
  } else if (mode == "device-shared-race") {
    nevpt2::gpuCheck(nevpt2::canary::launchSharedRace(s, out));
  } else if (mode == "device-divergent-barrier") {
    nevpt2::gpuCheck(nevpt2::canary::launchDivergentBarrier(s, out));
  } else {
    std::print(stderr, "nevpt2_sanitizer_canary: unknown mode '{}'\n", mode);
    return 2;
  }

  nevpt2::gpuCheck(wwr::wwrFreeAsync(out, s));
  // Kernel faults surface here (and abort through gpuCheck).
  nevpt2::gpuCheck(wwr::wwrStreamSynchronize(s));
  nevpt2::gpuCheck(wwr::wwrStreamDestroy(s));
  return 0;
}
