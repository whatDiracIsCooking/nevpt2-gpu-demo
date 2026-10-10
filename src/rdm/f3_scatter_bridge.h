// f3_scatter_bridge.h -- the declaration shared between rdm_build.cpp (host
// C++23 module unit) and f3_scatter.cu (device-compiled, linked in as the
// nevpt2.rdm.f3_scatter.device static library). Same shape as
// rdm/rdm_accumulate_bridge.h; see that header for why a host unit includes
// it after its imports, inside extern "C++", and why only plain types cross it.
//
// Only the ca order has a kernel of its own: the ac order's target index is
// the source index, so its scatter is accumulateInplace(f3, C, n^6).
#pragma once

#include "wwr/wwr.h"  // wwrStream_t (+ wwrError_t in a device pass)

namespace nevpt2::device {

// f3[r*n^2 + fr*norb + a] += C[r*n^2 + a*norb + fr] for every r in [0, n^4)
// and a, fr in [0, norb): fold the --cublas / --blas-digest f3 GEMM's (n^4, n^2) row-major
// result into the NEVPTkern-permuted ca accumulator. Returns the launch status,
// wwrSuccess on success.
wwrError_t f3ScatterCa(wwrStream_t stream, const double* c, double* f3, int norb);

// --fused-digest: fold the one (n^4, 3 n^2) row-major GEMM over the stacked
// [R; W_ca; W_ac] into all three accumulators -- columns [0, n^2) += into dm3,
// [n^2, 2 n^2) through f3ScatterCa's transpose into f3ca, [2 n^2, 3 n^2) +=
// into f3ac. Returns the launch status, wwrSuccess on success.
wwrError_t fusedDigestSplit(wwrStream_t stream, const double* c, double* dm3, double* f3ca,
                            double* f3ac, int norb);

}  // namespace nevpt2::device
