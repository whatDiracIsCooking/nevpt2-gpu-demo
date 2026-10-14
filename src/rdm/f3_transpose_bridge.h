// f3_transpose_bridge.h -- the declarations shared between the RDM build's
// :adjoint partition (rdm_build_adjoint.cppm, host C++23 module unit) and
// f3_transpose.cu (device-compiled, linked in as the
// nevpt2.rdm.f3_transpose.device static library). Same shape as
// rdm/rdm_accumulate_bridge.h; see that header for why a host unit includes
// it after its imports, inside extern "C++", and why only plain types cross it.
//
// THE TRANSPOSE OF THE f3 DIGESTS' h2e DEPENDENCE. At a fixed CI vector the
// build's f3 path is LINEAR in the integrals: per tile, consume
// (W = eri . L2), digest (f3_pre += W . L2), and once at the end the fdm2
// correction (fdm2 = eri . dm3) folded in by the wedge. These four kernels
// are the adjoints of those steps, so that, given G_o = dE/d(f3_o) for the
// two orders, the host side assembles dE/d(h2e) through the digests without
// the 4-RDM ever existing -- L2 is re-produced tile by tile, exactly as the
// plain build produces it (docs/gradients.md, section 6).
//
// Every output written by `+=` has exactly one writer per launch, and the
// launches are ordered on the one stream, so nothing here needs an atomic.
// `hbar` is PHYSICIST-ordered, like the h2e the build takes: the chemists'
// index eri[i,j,k,l] the forward kernels read is h2e[i,k,j,l], and each
// adjoint writes straight to the physicist slot. `order` is the build's: 0 =
// ca, 1 = ac. Each returns the launch status, wwrSuccess on success.
#pragma once

#include "wwr/wwr.h"  // wwrStream_t (+ wwrError_t in a device pass)

namespace nevpt2::device {

// The wedge's adjoint onto the pre-wedge f3, in the digest GEMM's operand
// layout: gt[r*n^2 + a*norb + f] = Gpre[r*n^2 + last2(a, f)], with
// last2 = f*norb + a (ca) or a*norb + f (ac) and
//   Gpre[s] = [s0 >= s2] G[s] + [s0 > s2] G[s2, s3, s0, s1, s4, s5]
// -- the wedge overwrites every s0 < s2 element with its mirror, so those
// pre-wedge values never reach the output. Overwrites all n^6 of `gt`.
wwrError_t f3WedgeAdjoint(wwrStream_t stream, int order, const double* g, double* gt, int norb);

// The wedge's adjoint onto fdm2 (the same for both orders, as the wedge is):
//   fdm2bar[x, c, d, e] = sum_{i > x} G[x, i, i, c, d, e]
//                       - sum_{j < x} G[j, c, x, j, d, e].
// Overwrites all n^4 of `fdm2bar`.
wwrError_t f3Fdm2Adjoint(wwrStream_t stream, const double* g, double* fdm2bar, int norb);

// fdm2's adjoint onto the integrals, accumulated (+=) into `hbar`:
//   ca: hbar[l, B, C, A] += sum_{ijk} fdm2bar[i, j, k, l] dm3[A, B, C, k, j, i]
//   ac: hbar[k, E, C, F] += sum_{ijl} fdm2bar[i, j, k, l] dm3[i, j, C, l, E, F]
wwrError_t f3Fdm2EriAdjoint(wwrStream_t stream, int order, const double* dm3,
                            const double* fdm2bar, double* hbar, int norb);

// consume's adjoint onto the integrals for one tile, accumulated (+=) into
// `hbar`, from Wbar[a*norb + f, K] = dE/dW (row stride `width`) and the
// tile's L2[pqrs, K]:
//   ca: hbar[a, q, x, p] += sum_{f, K} Wbar[a*norb + f, K] L2[p, q, x, f, K]
//   ac: hbar[a, q, x, p] += sum_{f, K} Wbar[a*norb + f, K] L2[p, q, f, x, K]
wwrError_t f3ConsumeAdjoint(wwrStream_t stream, int order, const double* wbar, const double* l2,
                            double* hbar, int norb, int width);

}  // namespace nevpt2::device
