// f3_digest_bridge.h -- the declaration shared between rdm_build.cpp (host
// C++23 module unit) and f3_digest.cu (device-compiled, linked in as the
// nevpt2.rdm.f3_digest.device static library). Same shape as
// rdm_accumulate_bridge.h; see that header for why a host unit includes it
// after its imports, inside extern "C++".
#pragma once

#include "wwr/wwr.h"  // wwrStream_t (+ wwrError_t in a device pass)

namespace nevpt2::device {

// The emitted f3 digest GEMM for one tile, ACCUMULATING (+=) into the
// NEVPTkern-ordered n^6 accumulator f3: for every row r = wvut of L2,
//   ca: f3[r, fr, a] += sum_K W[a*n + fr, K] * L2[r, K]
//   ac: f3[r, a, fr] += sum_K W[a*n + fr, K] * L2[r, K]
// `order` 0 = ca, 1 = ac. Returns the LAUNCH status, wwrSuccess on success.
wwrError_t f3DigestAccumulate(wwrStream_t stream, int order, const double* w, const double* l2,
                              double* f3, int norb, int width);

}  // namespace nevpt2::device
