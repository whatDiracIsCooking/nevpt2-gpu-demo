// nevpt2.profile -- the --profile recorder: one switch and one list
// of labelled GPU event pairs for the whole run.
//
// One --profile flag covers einsum's contractions (in practice the energy
// stage), rdm_build's phases and the density-fitted demo's slab DGEMMs.
//
// A recorded span is an event pair around one piece of enqueued GPU work,
// both events on the stream that work runs on -- res.stream(), the one
// non-blocking stream; nothing here makes a stream of its own.
// Spans land in the innermost open Section, so one report() per section gives
// one table per stage; a span with no Section open is our bug (check). Off (the default), time() costs one bool check and
// calls through.
//
// Events are read back only by report(), which must be called after a
// synchronization that covers every span in that section: an unfinished
// event makes wwrEventElapsedTime undefined. The RDM build synchronizes its
// stream before returning, and each energy class ends in downloadTensor (a
// stream sync), so the demos report after the energy stage with no sync of
// their own.
export module nevpt2.profile;

import std;
export import nevpt2.wwr;
// Re-exported: a span's sizes are int64_t, the one host size type.
export import nevpt2.common;

export namespace nevpt2 {

struct ProfileRow {
  std::string label;    // the caller's: an einsum subscript string or "diagA16"/...,
                        // an RDM-build phase ("produce", ...), a DF slab DGEMM
  int calls = 0;
  double totalMs = 0.0;
  double meanUs = 0.0;
  int64_t outSize = 0;         // from the last call with this label
  int64_t contractedSize = 0;  // from the last call with this label
};

namespace profile {

// The sections the demos report, in the order they print them.
inline constexpr std::string_view kRdmBuild = "RDM build";
inline constexpr std::string_view kDfIntegrals = "DF integrals";
inline constexpr std::string_view kEnergy = "energy";

}  // namespace profile
}  // namespace nevpt2

// Not exported: what enabled() and time() reach inline, visible inside this
// module only.
namespace nevpt2::profile {
// Read inline by time(), so "off" stays one load and branch per launch.
inline bool g_enabled = false;
// Records a span's start event in the current section and returns its index;
// end() records its stop event. Called only when enabled.
std::size_t begin(std::string_view label, wwrStream_t stream, int64_t outSize,
                  int64_t contractedSize);
void end(std::size_t span, wwrStream_t stream);
}  // namespace nevpt2::profile

export namespace nevpt2::profile {

void setEnabled(bool enabled);
inline bool enabled() { return g_enabled; }

// Names the section every span recorded while it is alive lands in; nests
// (the destructor restores the enclosing one). Host-only bookkeeping, no GPU
// call, so it is constructed whether or not profiling is on.
class Section {
 public:
  explicit Section(std::string_view name);
  ~Section();
  Section(const Section&) = delete;
  Section& operator=(const Section&) = delete;

 private:
  std::string previous_;
};

// Runs `f`, whose GPU work is enqueued on `stream`, between an event pair on
// that same stream, labelled `label` in the current section. outSize /
// contractedSize ride along into the report (einsum's launch shape; 0 prints
// as "-"). Off: just f().
template <class F>
void time(std::string_view label, wwrStream_t stream, F&& f, int64_t outSize = 0,
          int64_t contractedSize = 0) {
  if (!enabled()) {
    std::forward<F>(f)();
    return;
  }
  const std::size_t span = begin(label, stream, outSize, contractedSize);
  std::forward<F>(f)();
  end(span, stream);
}

// Aggregates `section`'s spans by label, sorted by totalMs descending, then
// destroys their events and forgets them (a second report() of the same
// section does not double-count). Call only after a synchronization that
// covers every span in it -- see the top of this file.
std::vector<ProfileRow> report(std::string_view section);

// report(section) as one printed table. `wallSeconds` is the stage's
// wall-clock where it has one of its own (0 leaves it out). Prints nothing
// when profiling is off.
void printReport(std::string_view section, double wallSeconds = 0.0);

}  // namespace nevpt2::profile
