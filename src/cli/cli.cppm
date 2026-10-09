// nevpt2.cli -- the command-line flags every demo takes, parsed in one place.
//
// Each app keeps its own loop over argv and offers every argument to
// parseCommonFlag first; what that does not consume is the app's own flag or
// a usage error. finalize() then applies the checks that span flags (the
// digest conflicts, --cublas on a build without it). Every bad argument is a
// value-tier Error (nevpt2.error_handling) for main() to report once;
// nothing here prints or exits.
export module nevpt2.cli;

import std;
// Re-exported: every entry point hands back a Result/Status, and
// CommonOptions holds an RdmBuildOptions.
export import nevpt2.error_handling;
export import nevpt2.rdm_build;
import nevpt2.device_resources;

export namespace nevpt2::cli {

// The flags both demos take.
struct CommonOptions {
  std::string goldenPath;
  RdmBuildOptions rdm;
  bool profile = false;
  bool pc = false;
  std::uint64_t poolThreshold = kDefaultPoolReleaseThreshold;
  // --mantissa-bits was given: finalize() refuses it without --cublas, the
  // only digest that reads it (rdm.mantissaBits alone cannot tell an
  // explicit 53 from the default).
  bool mantissaBitsGiven = false;
};

// An integer flag's value (--tiles, --ozaki-pairs, --mantissa-bits, --batch,
// ...): a whole integer that fits T, or a config Error naming the flag and the
// argument (std::atoi/atoll silently read garbage as 0). A range
// is checked after parsing, in finalize() (or the app, for its own flags).
template <class T>
Result<T> parseInteger(const std::string_view flag, const std::string_view arg) {
  T v{};
  const auto [end, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), v);
  if (ec != std::errc{} || end != arg.data() + arg.size())
    return err_config(std::format("{} takes an integer, not \"{}\"", flag, arg));
  return v;
}

// Offers args[i] to the common flags. true: it was one, and i now indexes its
// last argument (the value, for a flag that takes one). false: not one of
// them -- the app's own flag, or a usage error; a flag that takes a value but
// has none after it is false too, so it lands in the app's usage message. An
// Error: it was one, and its value is bad.
Result<bool> parseCommonFlag(std::span<const std::string_view> args, std::size_t& i,
                             CommonOptions& opt);

// The checks that span flags, once every argument is in: --golden given,
// --tiles >= 1, --ozaki-pairs in 0..14, --mantissa-bits in 1..53 and only
// with --cublas, --cublas only where the emulated digest is built, and at most
// one of --cublas / --blas-digest / --ozaki. Also makes --ozaki take
// precedence over blasDigest's HIP default. `goldenHint` names the file in
// the missing-golden message ("path.nevpt2gold").
Status finalize(CommonOptions& opt, std::string_view goldenHint);

// The usage line: the program, its golden file, `extraFlags` (the app's own,
// bracketed; may be empty), then the common flags.
std::string usage(std::string_view prog, std::string_view goldenHint,
                  std::string_view extraFlags);

}  // namespace nevpt2::cli
