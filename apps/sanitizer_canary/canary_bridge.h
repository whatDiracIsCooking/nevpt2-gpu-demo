// canary_bridge.h -- the declarations shared between the sanitizer canary's
// host unit (main.cppm) and its deliberately buggy kernels (canary_kernels.cu,
// linked in as the nevpt2.sanitizer_canary.device static library). Same shape
// as every other *_bridge.h here: plain types across the boundary, each
// launcher returns the LAUNCH status as wwrError_t, so main.cppm includes
// it after its imports, inside extern "C++" (see src/rdm/rdm_accumulate_bridge.h).
//
// Every kernel here is WRONG ON PURPOSE. Each is the smallest program that a
// particular compute-sanitizer tool must report; see main.cppm and docs/testing.md
// ("Sanitizers").
#pragma once

#include "wwr/wwr.h"  // wwrStream_t (+ wwrError_t in a device pass)

namespace nevpt2::canary {

// out[i] = in[i] for i in [0, n). Correct in itself: the canaries make it
// read something it must not (freed, never written).
wwrError_t launchCopy(wwrStream_t stream, const double* in, double* out, int n);

// out[0] = in[n]: one element past the end of an n-element buffer.
wwrError_t launchReadPastEnd(wwrStream_t stream, const double* in, double* out, int n);

// One block of 64 threads: thread 0 writes a __shared__ slot that every
// thread reads, with no barrier in between -- a shared-memory hazard.
// Writes out[0..63].
wwrError_t launchSharedRace(wwrStream_t stream, double* out);

// One block of 64 threads whose two warps wait at two different
// __syncthreads() calls. Writes out[0..63].
wwrError_t launchDivergentBarrier(wwrStream_t stream, double* out);

}  // namespace nevpt2::canary
