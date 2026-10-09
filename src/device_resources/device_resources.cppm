// nevpt2.device_resources -- DeviceResources: the ONE stream, memory pool, BLAS handle and
// dense-solver handle every GPU call in both demos goes through (the
// solver handle for the PC-NEVPT2 per-class eigensolves).
//
// Each demo's main() creates exactly one (DeviceResources::create) and passes
// it -- or its stream() -- down. Everything is issued on stream(), a
// NON-BLOCKING stream: it does not synchronize with the legacy default stream
// in either direction, so a call that slipped onto the legacy stream would race
// the work on it silently (docs/performance.md, "One stream"). Nothing in src/ names the legacy stream; the
// devtools/stream-lint.sh --strict gate enforces it.
//
// Every device allocation is stream-ordered: wwrMallocFromPoolAsync from pool()
// on stream(), and wwrFreeAsync on stream(). A free is therefore ordered after
// every kernel queued before it on the same stream -- which is what lets the
// energy stage free a slab without first synchronizing.
//
// The members are WarpWraps' RAII wrappers with nevpt2::Abort in every policy
// slot, as nevpt2.wwr instantiates them (StreamWrapper, MemPoolWrapper,
// BlasHandleWrapper, SolverDnHandleWrapper). The BLAS handle
// is a BlasHandleWrapper over the stream, bound to it once at creation: the
// consume GEMMs, the --blas-digest GEMMs and the density-fitting slab GEMMs all
// use it (the --cublas emulation handle is separate -- cublas_emul.cpp -- and
// bound to the same stream). The solver handle is its sibling: a
// SolverDnHandleWrapper over the same stream, bound once at
// creation, used by nevpt2.energy's PC Dsyevd calls (energy_pc.cpp). Their workspace and devInfo
// are DeviceBuffers from pool(), like every other allocation.
//
// It satisfies WarpWraps' device_handle_pool concept (dev_idx(), stream(),
// pool(), all const noexcept), and backs every device buffer in the tree
// (DeviceBuffer below): WarpWraps' DeviceBufferWrapper allocates
// from pool() on stream(), zero-fills on stream(), frees with wwrFreeAsync on
// stream() in its destructor, and co-owns the DeviceResources through a
// shared_ptr so the pool and stream outlive the last free. create() returns a
// shared_ptr and the constructor is private, so every DeviceResources is owned
// by one -- which is what makes shared_from_this() (how a buffer gets its
// co-owning pointer from the `const DeviceResources&` every function here
// takes) always valid.
export module nevpt2.device_resources;

import std;
// Re-exported: callers name wwrStream_t / wwrMemPool_t / wwrDeviceProp,
// check what they get with gpuCheck, and call DeviceBuffer<T>'s interface
// (wwr.extension.memory_buffer, which nevpt2.wwr re-exports).
export import nevpt2.wwr;
// wwrblasHandle_t, which blas() returns.
export import wwr.blas;
// wwrsolverDnHandle_t, which solver() returns, and the wwrsolver* surface its
// users call.
export import wwr.solver;

export namespace nevpt2 {

// The pool's high-water marks since creation, in bytes (the pool attributes
// UsedMemHigh / ReservedMemHigh). `used` is the most the demo held allocated
// at once; `reserved` is the most the pool held from the device at once
// (allocated plus freed-but-kept, which the release threshold governs).
struct PoolHighWater {
  std::uint64_t used = 0;
  std::uint64_t reserved = 0;
};

// The release threshold both demos use unless --pool-threshold overrides it:
// keep everything freed in the pool, never hand it back mid-run. Chosen by
// measurement (docs/performance.md, "The memory pool").
inline constexpr std::uint64_t kDefaultPoolReleaseThreshold =
    std::numeric_limits<std::uint64_t>::max();

// enable_shared_from_this rather than a shared_ptr parameter on every function
// that allocates: callees need only the reference (stream(), blas()), and a
// buffer gets its co-owning pointer where it is built
// (`DeviceBuffer<T>(n, res.shared_from_this())`). Safe because create() is the
// only way to make one.
class DeviceResources : public std::enable_shared_from_this<DeviceResources> {
 public:
  // Selects device `idx`, creates the non-blocking stream, a pool on that
  // device with `releaseThreshold` (bytes the pool may keep across a
  // synchronization before trimming back to the OS), and the BLAS handle bound
  // to the stream, and the dense-solver handle bound to the same stream.
  //
  // Device selection is the environment's to fail, not ours: a
  // device that cannot be selected (no card, a bad index) or that reports no
  // memory-pool support is an InvalidConfig Error, created here and reported
  // by main(), and nothing is created on it. A failure after that -- creating
  // the stream, pool or handles on a device that passed -- still aborts.
  static Result<std::shared_ptr<DeviceResources>> create(int idx, std::uint64_t releaseThreshold);

  DeviceResources(const DeviceResources&) = delete;
  DeviceResources& operator=(const DeviceResources&) = delete;
  DeviceResources(DeviceResources&&) = delete;
  DeviceResources& operator=(DeviceResources&&) = delete;
  // Synchronizes the stream first, so every queued free has landed in the pool
  // before the pool (and then the stream) is destroyed.
  ~DeviceResources();

  int dev_idx() const noexcept { return idx_; }
  wwrStream_t stream() const noexcept { return stream_->get(); }
  wwrMemPool_t pool() const noexcept { return pool_.get(); }
  const wwrDeviceProp& props() const noexcept { return props_; }
  wwrblasHandle_t blas() const noexcept { return blas_.get(); }
  wwrsolverDnHandle_t solver() const noexcept { return solver_.get(); }
  std::uint64_t releaseThreshold() const noexcept { return releaseThreshold_; }

  PoolHighWater highWater() const;

 private:
  DeviceResources(int idx, std::uint64_t releaseThreshold, const wwrDeviceProp& props);

  int idx_;
  std::uint64_t releaseThreshold_;
  wwrDeviceProp props_{};
  // Its own shared_ptr so blas_ and solver_ can co-own it (both handle wrappers
  // retain their stream owner): ownership runs blas_/solver_ -> stream_ only,
  // so there is no cycle.
  std::shared_ptr<StreamWrapper> stream_;
  MemPoolWrapper pool_;
  BlasHandleWrapper blas_;
  SolverDnHandleWrapper solver_;
};

// The device buffers: nevpt2.wwr's DeviceBufferSuite -- WarpWraps'
// buffer suite, nevpt2::Abort in every policy slot -- closed over
// DeviceResources. `const`: a buffer needs only
// the handle's const accessors, and shared_from_this() on the
// `const DeviceResources&` callers hold yields a pointer to const.
//
// A DeviceBuffer<T> is the one spelling of a device allocation in this tree:
// its constructor draws from pool() on stream() (wwrMallocFromPoolAsync) and
// zero-fills it there (wwrMemsetAsync -- the suite has no uninitialized
// allocation; the cost is measured in docs/performance.md, "The memory pool"), and
// its destructor returns it with wwrFreeAsync on stream(). The free is
// therefore ordered after every kernel already queued on stream(), so a buffer
// may go out of scope while kernels that read it are still queued -- the
// stream-ordered free, by construction. Move-only. A
// DeviceBufferView<T> is a copyable, non-owning view of one; it frees nothing.
// Built directly -- `DeviceBuffer<T>(n, res.shared_from_this())` -- the buffer
// co-owning res; its constructor's `source_location` defaults to the caller's
// line, so an out-of-memory abort names the function that asked.
using DeviceBuffers = DeviceBufferSuite<const DeviceResources>;
template <class T>
using DeviceBuffer = DeviceBuffers::device<T>;
template <class T>
using DeviceBufferView = DeviceBuffers::device_view<T>;

// "1234.5 MB" -- the unit every memory line in both demos prints.
inline double toMB(std::uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

// --pool-threshold's argument: "max" (kDefaultPoolReleaseThreshold) or a byte
// count; anything else is an InvalidConfig Error naming it.
Result<std::uint64_t> parsePoolThreshold(std::string_view arg);

// "device pool high-water <when>: used X MB, reserved Y MB (release
// threshold Z)" on stdout -- measured by the pool, unlike the demos'
// counted-not-measured per-tile and slab lines.
void printPoolHighWater(const char* when, const DeviceResources& res);

}  // namespace nevpt2
