module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry

module nevpt2.device_resources;

import std;
import :device_select;
import wwr.extension.runtime;
import wwr.extension.blas;
import wwr.extension.solver;

namespace nevpt2 {

DeviceResources::DeviceResources(int idx, std::uint64_t releaseThreshold,
                                 const wwrDeviceProp& props)
    : idx_(selectDevice(idx)),
      releaseThreshold_(releaseThreshold),
      props_(props),
      // Non-blocking, spelled out rather than left to StreamWrapper's default
      // (which is also non-blocking): the one-stream design rests on this flag.
      stream_(std::make_shared<StreamWrapper>(idx, wwrStreamNonBlocking)),
      pool_(idx, releaseThreshold),
      blas_(stream_),
      solver_(stream_) {
  // The wrappers select their device and may leave another current; the raw
  // calls (kernel launches, copies) go to whichever is current, so make sure
  // it is this one.
  gpuCheck(wwrSetDevice(idx));
}

DeviceResources::~DeviceResources() {
  // Every wwrFreeAsync was queued on the stream; let them land before the pool
  // goes. (The members' own destructors then run: solver_, blas_, pool_, and stream_ --
  // whose wrapper synchronizes again before destroying it.) No abort on the
  // destructor path: a failure here is past the point it could matter.
  (void)wwrStreamSynchronize(stream());
}

Result<std::shared_ptr<DeviceResources>> DeviceResources::create(int idx,
                                                                 std::uint64_t releaseThreshold) {
  // Selecting the device brings its context up, before any stream exists. A
  // failure is the environment's (no card visible, a bad index), so it is an
  // Error, not a gpuCheck abort.
  if (const wwrError_t e = wwrSetDevice(idx); e != wwrSuccess)
    return err_config(std::format("cannot select GPU device {}: {} ({})", idx,
                                  wwrGetErrorName(e), wwrGetErrorString(e)));
  // Checked for memory-pool support before anything is created on it: every
  // allocation here is wwrMallocFromPoolAsync, which a device without pools
  // cannot serve. (Both backends' property structs carry
  // memoryPoolsSupported; WarpWraps exports no backend-neutral name for the
  // device-attribute enum, so the struct is the portable way to ask.)
  wwrDeviceProp props{};
  gpuCheck(wwrGetDeviceProperties(&props, idx));
  if (!props.memoryPoolsSupported)
    return err_config(std::format(
        "device {} ({}) reports no stream-ordered memory-pool support; every allocation in "
        "this demo comes from a pool",
        idx, static_cast<const char*>(props.name)));
  // Not make_shared: the constructor is private.
  return std::shared_ptr<DeviceResources>(new DeviceResources(idx, releaseThreshold, props));
}

Result<std::uint64_t> parsePoolThreshold(const std::string_view arg) {
  if (arg == "max") return kDefaultPoolReleaseThreshold;
  std::uint64_t v = 0;
  const auto [end, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), v);
  if (ec != std::errc{} || end != arg.data() + arg.size())
    return err_config(
        std::format("--pool-threshold takes a byte count or \"max\", not \"{}\"", arg));
  return v;
}

void printPoolHighWater(const char* when, const DeviceResources& res) {
  PoolHighWater hw = res.highWater();
  if (res.releaseThreshold() == kDefaultPoolReleaseThreshold) {
    std::printf(
        "device pool high-water %s: used %.1f MB, reserved %.1f MB (release threshold max)\n",
        when, toMB(hw.used), toMB(hw.reserved));
  } else {
    std::print(
        "device pool high-water {}: used {:.1f} MB, reserved {:.1f} MB (release threshold {} "
        "bytes)\n",
        when, toMB(hw.used), toMB(hw.reserved), res.releaseThreshold());
  }
}

PoolHighWater DeviceResources::highWater() const {
  PoolHighWater hw;
  // 64-bit attributes on both backends (cuuint64_t / uint64_t), read through
  // the void* -- an std::uint64_t is exactly what they write
  // (https://github.com/whatDiracIsCooking/WarpWraps/issues/316).
  gpuCheck(wwrMemPoolGetAttribute(pool(), wwrMemPoolAttrUsedMemHigh, &hw.used));
  gpuCheck(wwrMemPoolGetAttribute(pool(), wwrMemPoolAttrReservedMemHigh, &hw.reserved));
  return hw;
}

}  // namespace nevpt2
