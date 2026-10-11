// sigma_vector_bridge.h -- the declarations shared between sigma_vector.cpp
// (host C++23 module unit, nevpt2.sigma_vector) and sigma_vector.cu, the
// kernel TU linked in as the nevpt2.rdm.sigma_vector.device static library.
// Same shape as rdm_accumulate_bridge.h; see that header for why a host unit
// includes it after its imports, inside extern "C++".
//
// Plain types only across this boundary (pointers, integers, the vendor
// stream handle). Every function below launches over ONE determinant tile
// [k0, k0 + width) of the output and returns the LAUNCH status, wwrSuccess on
// success; a kernel's own execution errors surface at the next
// synchronization, as with any launch.
//
// WHAT A SIGMA VECTOR IS. Given a coefficient tensor over active indices, a
// sigma vector is that tensor contracted with the excited wavefunctions it
// multiplies, one amplitude per determinant:
//
//   sigma[K] = sum_{pqrstu} d3[p,q,r,s,t,u] (E_pq E_rs E_tu |ci>)[K]
//
// The point (the issue's, and Park's, JCTC 2019): the n^6 (or, for the 4-body
// term, n^8) RDM DERIVATIVE never has to exist. Contracted against its
// coefficient tensor on the fly, it is a vector of length ndet -- the same
// trick f3ac/f3ca already play for the energy, where the 4-RDM is never
// built either.
//
// HOW. Every kernel here walks the single-excitation link tables BACKWARDS
// from the determinant it owns: one thread per output determinant, and
// (E_pq x)[K] = sum_{K'} <K|E_pq|K'> x[K'] is exactly a row of the REVERSE
// table (link_tables::reverse_link), the same rlink_a/rlink_b
// produce_generic's R gathers over. A `b`-body sigma is then b nested reverse
// walks, each step picking up its operator's index pair and a sign. Nothing
// is shared between threads: like kernels.cu, this file uses no atomics, no
// warp shuffle and no shared memory, so it compiles unchanged for CUDA and
// HIP.
#pragma once

#include "wwr/wwr.h"  // wwrStream_t (+ wwrError_t in a device pass)

namespace nevpt2::device {

// The 3-body sigma, WRITING (not accumulating) sigma[K] for K in
// [k0, k0 + width):
//
//   sigma[K] = sum_{pqrstu} d3[p,q,r,s,t,u] (E_pq E_rs E_tu |ci>)[K]
//
// `d3` is n^6 row-major in the dm3 LAYOUT -- index pair (p,q) leftmost, so
// contracting this sigma back against `ci` gives sum d3 . dm3 with
// dm3[p,q,r,s,t,u] = <ci|E_pq E_rs E_tu|ci> (nevpt2.rdm_build's convention).
// `ci` is the (na, nb) CI vector, `sigma` has one element per determinant,
// and `rlinkA`/`rlinkB` are the REVERSE link tables of the two spins.
// Three nested walks: (nla + nlb)^3 terms per determinant.
wwrError_t sigmaVector3(wwrStream_t stream, const double* ci, const int* rlinkA,
                        const int* rlinkB, const double* d3, double* sigma, int norb, int nb,
                        int nla, int nlb, int k0, int width);

// The 2-body sigma of n^2 coefficient tensors at once, all against the SAME
// input vector -- stage 1 of the 4-body contracted sigma:
//
//   out[b, K] = sum_{pqrs} c2[b, p,q,r,s] (E_pq E_rs |x>)[K]
//
// `c2` is (n^2, n^4) row-major, `x` and each row of `out` are one element per
// determinant, and `out` has row stride `ndet` (the full determinant axis, not
// the tile width: stage 2 gathers from anywhere in it). Two nested walks:
// n^2 (nla + nlb)^2 terms per determinant.
wwrError_t sigmaVector2Batched(wwrStream_t stream, const double* x, const int* rlinkA,
                               const int* rlinkB, const double* c2, double* out, int norb, int nb,
                               int nla, int nlb, int ndet, int k0, int width);

// The same n^2 2-body sigmas, each against ITS OWN input vector and summed
// over the batch -- stage 2, which WRITES sigma[K] for K in [k0, k0 + width):
//
//   sigma[K] = sum_b sum_{pqrs} c2[b, p,q,r,s] (E_pq E_rs |in_b>)[K]
//
// `in` is (n^2, ndet) row-major -- stage 1's `out`. Same cost as stage 1.
wwrError_t sigmaVector2Reduce(wwrStream_t stream, const double* in, const int* rlinkA,
                              const int* rlinkB, const double* c2, double* sigma, int norb, int nb,
                              int nla, int nlb, int ndet, int k0, int width);

}  // namespace nevpt2::device
