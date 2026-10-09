// rdm_kernels_bridge.h -- the declarations shared between rdm_build.cpp (host
// C++23 module unit) and rdm_launch.cu, the launcher TU for the RDM-build
// kernels (kernels.cu), linked in as the nevpt2.rdm.kernels.device static
// library. Same shape as rdm_accumulate_bridge.h; see that header for why a
// host unit includes it after its imports, inside extern "C++".
//
// Plain types only across this boundary (pointers, integers, the vendor
// stream handle). Every function launches with exactly the geometry the
// kernels ran at as a code object (see rdm_launch.cu) and returns the LAUNCH
// status, wwrSuccess on success. `order` is the f3 order the host loop
// indexes by: 0 = ca, 1 = ac.
#pragma once

#include "wwr/wwr.h"  // wwrStream_t (+ wwrError_t in a device pass)

namespace nevpt2::device {

// K1: R[tu, kl] and L2[pqrs, kl] for the determinants [k0, k0 + width).
wwrError_t rdmProduce(wwrStream_t stream, const double* ci, const int* flinkA, const int* flinkB,
                      const int* rlinkA, const int* rlinkB, double* r, double* l2, int norb, int na,
                      int nb, int nla, int nlb, int k0, int width);

// K3: dm3[pqrs, tu] = sum_K L2[pqrs, K] R[tu, K] for one tile (overwrites).
wwrError_t rdmDigestDm3(wwrStream_t stream, const double* r, const double* l2, double* dm3,
                        int norb, int width);

// The emitted f3 consume, W = eri . L2 (only under --consume-emitted).
wwrError_t rdmConsume(wwrStream_t stream, int order, const double* eri, const double* l2, double* w,
                      int norb, int width);

// No f3 digest here: the emitted f3 digest reads L2 directly and has its own
// bridge (f3_digest.cu).

// fdm2 and the wedge, once, on the fully-summed dm3 / f3.
wwrError_t rdmFdm2(wwrStream_t stream, int order, const double* eri, const double* dm3,
                   double* fdm2, int norb);
wwrError_t rdmWedge(wwrStream_t stream, int order, const double* fdm2, double* f3, int norb);

}  // namespace nevpt2::device
