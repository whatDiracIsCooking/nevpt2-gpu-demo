// nevpt2.error_handling -- the vendor-free error model, in two tiers. It
// imports only std, so every module can import it, nevpt2.common (narrowTo)
// included, without pulling in the GPU runtime.
//
// THE VALUE TIER is for failures that are not our bug: a missing or malformed
// golden file, a bad flag combination, a case the code does not handle, a
// numerical refusal. A routine that can hit one returns a Result<T> (or a
// Status, for no value) holding an Error; the caller either handles it or
// propagates it unchanged with NEVPT2_TRY (error_handling/error_macros.h), and
// each main() calls report() on the one that reaches it and returns nonzero.
// An Error records where it was CREATED (its factory's defaulted
// std::source_location), and propagation never touches it, so report() names
// the site that detected the failure, not the main() that printed it. There is
// no per-kind exit-code table: ctest only needs nonzero.
//
// THE ABORT TIER is for broken invariants -- our bugs, never bad input:
// check(cond, what) prints the caller's file:line and `what` to stderr and
// std::abort()s. narrowTo's failure path (nevpt2.common) goes through it.
//
// A GPU status is not judged here: gpuCheck, which aborts on any non-success
// wwrError_t / wwrblasStatus_t / wwrsolverStatus_t, lives in nevpt2.wwr, the
// module that knows those types.
module;

#include <cstdio>  // stderr: a macro, which `import std` does not carry

export module nevpt2.error_handling;

import std;

export namespace nevpt2 {

// ---------------------------------------------------------------------------
// The value tier.
// ---------------------------------------------------------------------------

// What kind of failure an Error is. Only what report() prints depends on it.
enum class ErrorKind {
  IO,             // a file could not be opened, read or parsed
  InvalidConfig,  // the flags / inputs are inconsistent or out of range
  Unsupported,    // a valid request this build or this code does not handle
  Numerical,      // a numerical refusal (ill-conditioned, not converged, ...)
};

// The name report() prints for a kind.
[[nodiscard]] constexpr std::string_view kindName(const ErrorKind kind) noexcept {
  switch (kind) {
    case ErrorKind::IO:
      return "I/O error";
    case ErrorKind::InvalidConfig:
      return "invalid configuration";
    case ErrorKind::Unsupported:
      return "unsupported";
    case ErrorKind::Numerical:
      return "numerical error";
  }
  return "error";  // unreachable: kind is always one of the four above
}

// A failure handed back to the caller. `origin` is where the Error was
// created: each factory defaults it to its call site.
struct Error {
  ErrorKind kind;
  std::string message;
  std::source_location origin;

  [[nodiscard]] static Error io(std::string message,
                                std::source_location loc = std::source_location::current()) {
    return Error{ErrorKind::IO, std::move(message), loc};
  }
  [[nodiscard]] static Error config(std::string message,
                                    std::source_location loc = std::source_location::current()) {
    return Error{ErrorKind::InvalidConfig, std::move(message), loc};
  }
  [[nodiscard]] static Error unsupported(
      std::string message, std::source_location loc = std::source_location::current()) {
    return Error{ErrorKind::Unsupported, std::move(message), loc};
  }
  [[nodiscard]] static Error numerical(
      std::string message, std::source_location loc = std::source_location::current()) {
    return Error{ErrorKind::Numerical, std::move(message), loc};
  }
};

// A value or the Error that prevented it.
template <class T>
using Result = std::expected<T, Error>;

// Success, or the Error that prevented it.
using Status = std::expected<void, Error>;

// `return err_io("...");` from a function returning any Result<T> or Status:
// std::unexpected<Error> converts to each. The location is the caller's.
[[nodiscard]] inline std::unexpected<Error> err_io(
    std::string message, std::source_location loc = std::source_location::current()) {
  return std::unexpected<Error>{Error::io(std::move(message), loc)};
}
[[nodiscard]] inline std::unexpected<Error> err_config(
    std::string message, std::source_location loc = std::source_location::current()) {
  return std::unexpected<Error>{Error::config(std::move(message), loc)};
}
[[nodiscard]] inline std::unexpected<Error> err_unsupported(
    std::string message, std::source_location loc = std::source_location::current()) {
  return std::unexpected<Error>{Error::unsupported(std::move(message), loc)};
}
[[nodiscard]] inline std::unexpected<Error> err_numerical(
    std::string message, std::source_location loc = std::source_location::current()) {
  return std::unexpected<Error>{Error::numerical(std::move(message), loc)};
}

// Prints `error at file:line in function: <kind>: <message>` to stderr, once.
// What each main() calls on the Error that reaches it, before returning
// nonzero.
inline void report(const Error& error) {
  std::print(stderr, "error at {}:{} in {}: {}: {}\n", error.origin.file_name(),
             error.origin.line(), error.origin.function_name(), kindName(error.kind),
             error.message);
}

// ---------------------------------------------------------------------------
// The abort tier.
// ---------------------------------------------------------------------------

// Unless `cond` holds, prints `error at file:line in function: <what>` (the
// caller's location) to stderr and std::abort()s. For a broken invariant --
// our bug -- never for bad input, which is an Error. A function rather than a
// macro because a macro cannot cross a module boundary.
inline void check(const bool cond, const std::string_view what,
                  const std::source_location loc = std::source_location::current()) {
  if (cond) [[likely]]
    return;
  std::print(stderr, "error at {}:{} in {}: {}\n", loc.file_name(), loc.line(),
             loc.function_name(), what);
  std::abort();
}

}  // namespace nevpt2
