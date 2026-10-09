// NEVPT2_TRY: the value tier's propagation guard (nevpt2.error_handling).
//
// A plain header, not a module unit, because this is a MACRO and a macro does
// not cross a module boundary: `import nevpt2.error_handling;` would not carry
// it. Pull it into a consumer's global module fragment by its root-relative
// path, "error_handling/error_macros.h" (nevpt2::error_handling exports the
// src/ root). Host modules only -- never in a .cu (it is a clang statement
// expression over std::expected, and kernel files cannot import the module).
//
//   auto tensor = NEVPT2_TRY(readTensor(path));  // Result<T>: the T, moved out
//   NEVPT2_TRY(checkFlags(opts));                // Status: used bare
//
// On failure it returns the Error from the enclosing function UNCHANGED --
// its kind, message and origin (where it was created) all preserved -- as a
// std::unexpected<Error>, so the enclosing function must return some
// nevpt2::Result<U> or nevpt2::Status. It evaluates its argument once, and
// needs `import std;` at the expansion site (it spells ::std::unexpected and
// ::std::move). Variadic so an argument with an unparenthesised comma in a
// template-argument list passes through as one.
#pragma once

#define NEVPT2_TRY(...)                                                   \
  ({                                                                      \
    auto&& nevpt2_try_result_ = (__VA_ARGS__);                            \
    if (!nevpt2_try_result_.has_value())                                  \
      return ::std::unexpected(::std::move(nevpt2_try_result_).error());  \
    *::std::move(nevpt2_try_result_);                                     \
  })
