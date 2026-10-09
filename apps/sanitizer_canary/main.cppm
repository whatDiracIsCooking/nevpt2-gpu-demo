// nevpt2_sanitizer_canary -- one deliberately buggy run per mode, each the
// smallest program a particular sanitizer must report.
//
// A canary is registered as a ctest entry ONLY under its own sanitizer's preset
// (add_sanitizer_canary in the root CMakeLists.txt), and passes only when
// the run fails AND prints the tool's report. So a green canary proves the tool
// is on and failing runs; a red one means the check it guards went dark. With
// the sanitizer off, every mode runs to completion, prints "nothing caught it"
// and exits 0, which the canary check turns into a failure -- that is how each
// one is shown to go red (docs/testing.md, "Sanitizers").
//
//   nevpt2_sanitizer_canary <mode>
//
//   host-heap-overflow      ASan       reads one element past a heap array
//   host-signed-overflow    UBSan      INT_MAX + 1
//   device-oob              memcheck   a kernel reads one element past its buffer
//   device-free-before-read memcheck   wwrFreeAsync on S, THEN a kernel on S reads
//                                      the buffer: the stream-ordered
//                                      use-after-free the demos could introduce
//   device-leak             memcheck   a device allocation never freed (needs
//                                      --leak-check full)
//   device-uninit-read      initcheck  a kernel reads device memory nothing wrote
//   device-shared-race      racecheck  a __shared__ write/read with no barrier
//   device-divergent-barrier synccheck  two warps at two different __syncthreads()
//
// Every GPU call is on one non-blocking stream with stream-ordered allocation,
// the same shape as the demos, so the canary has nothing for
// devtools/stream-lint.sh to report. Unlike the demos it allocates with plain
// wwrMallocAsync/wwrFreeAsync, not a pool-backed nevpt2::DeviceBuffer.
//
// A module unit nothing imports; see apps/integral_direct/main.cppm.
module;

#include <cstdio>  // stderr: a macro, which `import std` does not carry

export module nevpt2.app.sanitizer_canary;

import std;
// The canaries themselves, and the kernel launchers they call, in an interface
// partition that exports nothing (see its header): `export import` only
// because [module.unit] requires a primary interface to re-export its
// interface partitions.
export import :canaries;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr,
                 "usage: %s <host-heap-overflow | host-signed-overflow | device-oob | "
                 "device-free-before-read | device-leak | device-uninit-read | "
                 "device-shared-race | "
                 "device-divergent-barrier>\n",
                 argv[0]);
    return 2;
  }
  const std::string_view mode = argv[1];
  int rc = 0;
  if (mode == "host-heap-overflow") {
    rc = hostHeapOverflow();
  } else if (mode == "host-signed-overflow") {
    rc = hostSignedOverflow();
  } else {
    rc = device(mode);
  }
  if (rc == 0) {
    std::printf("canary %s: ran to completion, nothing caught it\n", argv[1]);
  }
  return rc;
}
