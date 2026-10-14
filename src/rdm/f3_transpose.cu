// The TRANSPOSE of the f3 digests' integral dependence: the kernels that take
// G_o = dE/d(f3_o) back onto the active two-electron integrals, through the
// 4-RDM, without the 4-RDM ever existing (#89; docs/gradients.md, section 6).
// The contracts, with the forward step each one undoes, are in
// f3_transpose_bridge.h; the host side that drives them tile by tile -- the
// same tile plan and the same produce kernel as the plain build, plus one
// native-fp64 GEMM per order for the digest's own adjoint -- is
// nevpt2.rdm_build's :adjoint partition (rdm_build_adjoint.cppm).
//
// One thread per OUTPUT element everywhere, each a plain loop over what its
// element sums: no atomic, no shuffle, no shared memory, so the result is
// fixed by the summation order written here and the same on both backends.
// The `+=` outputs have one writer per launch; launches are ordered on the one
// stream. LINKED IN, like every kernel here (add_device_library), launched
// through WarpWraps' parallel_for; under HIP the .cu is compiled with -x hip.
#include "rdm/f3_transpose_bridge.h"

#include "common/device_index.h"  // idx2, idx4, idx6, int64_t
#include <extension/parallel_for/parallel_for.cuh>

namespace nevpt2::device {

namespace {

// Const scalar members: parallel_for's device_functor requires the functor
// not be copy-assignable (it is passed as a grid constant).

// gt[r, a*n + f] = Gpre[r, last2(a, f)] -- the wedge's adjoint, read into the
// digest GEMM's operand layout. Gpre[s] = [s0 >= s2] G[s] + [s0 > s2]
// G[s2, s3, s0, s1, s4, s5] (f3_transpose_bridge.h).
struct WedgeAdjointFunctor {
  const double* const g;
  double* const gt;
  const int norb;
  const int order;

  __device__ void operator()(const int64_t i) const {
    const int64_t n = norb;
    const int64_t n2 = n * n;
    const int64_t r = i / n2;
    const int64_t af = i % n2;
    const int64_t a = af / n;
    const int64_t f = af % n;
    const int64_t last2 = order == 0 ? f * n + a : a * n + f;
    const int64_t s = r * n2 + last2;
    // s = (s0, s1, s2, s3, s4, s5), row-major.
    const int64_t s5 = s % n;
    const int64_t s4 = (s / n) % n;
    const int64_t s3 = (s / n2) % n;
    const int64_t s2 = (s / (n2 * n)) % n;
    const int64_t s1 = (s / (n2 * n2)) % n;
    const int64_t s0 = s / (n2 * n2 * n);
    double v = 0.0;
    if (s0 >= s2) v = g[s];
    if (s0 > s2) v += g[idx6(s2, s3, s0, s1, s4, s5, n)];
    gt[i] = v;
  }
};

// fdm2bar[x, c, d, e] = sum_{i > x} G[x, i, i, c, d, e]
//                     - sum_{j < x} G[j, c, x, j, d, e].
struct Fdm2AdjointFunctor {
  const double* const g;
  double* const fdm2bar;
  const int norb;

  __device__ void operator()(const int64_t o) const {
    const int64_t n = norb;
    const int64_t e = o % n;
    const int64_t d = (o / n) % n;
    const int64_t c = (o / (n * n)) % n;
    const int64_t x = o / (n * n * n);
    double acc = 0.0;
    for (int64_t i = x + 1; i < n; ++i) acc += g[idx6(x, i, i, c, d, e, n)];
    for (int64_t j = 0; j < x; ++j) acc -= g[idx6(j, c, x, j, d, e, n)];
    fdm2bar[o] = acc;
  }
};

// fdm2's adjoint, one thread per physicist-ordered integral h[w, y, z, v]:
//   ca: fdm2[i,j,k,l] = sum eri[l,C,B,A] dm3[A,B,C,k,j,i], eri[l,C,B,A] =
//       h[l,B,C,A], so h[w=l, y=B, z=C, v=A] += sum_{ijk} fdm2bar[i,j,k,l]
//       dm3[A,B,C,k,j,i];
//   ac: fdm2[i,j,k,l] = sum eri[k,C,E,F] dm3[i,j,C,l,E,F], eri[k,C,E,F] =
//       h[k,E,C,F], so h[w=k, y=E, z=C, v=F] += sum_{ijl} fdm2bar[i,j,k,l]
//       dm3[i,j,C,l,E,F].
struct Fdm2EriAdjointFunctor {
  const double* const dm3;
  const double* const fdm2bar;
  double* const hbar;
  const int norb;
  const int order;

  __device__ void operator()(const int64_t o) const {
    const int64_t n = norb;
    const int64_t v = o % n;
    const int64_t z = (o / n) % n;
    const int64_t y = (o / (n * n)) % n;
    const int64_t w = o / (n * n * n);
    double acc = 0.0;
    if (order == 0) {
      // l = w, B = y, C = z, A = v.
      for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
          for (int64_t k = 0; k < n; ++k)
            acc += fdm2bar[idx4(i, j, k, w, n)] * dm3[idx6(v, y, z, k, j, i, n)];
    } else {
      // k = w, E = y, C = z, F = v.
      for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
          for (int64_t l = 0; l < n; ++l)
            acc += fdm2bar[idx4(i, j, w, l, n)] * dm3[idx6(i, j, z, l, y, v, n)];
    }
    hbar[o] += acc;
  }
};

// consume's adjoint for one tile, one thread per physicist-ordered integral
// h[a, q, x, p] (= eri[a, x, q, p], what consume reads):
//   ca: += sum_{f, K} Wbar[a*n + f, K] L2[p, q, x, f, K]
//   ac: += sum_{f, K} Wbar[a*n + f, K] L2[p, q, f, x, K]
struct ConsumeAdjointFunctor {
  const double* const wbar;
  const double* const l2;
  double* const hbar;
  const int norb;
  const int width;
  const int order;

  __device__ void operator()(const int64_t o) const {
    const int64_t n = norb;
    const int64_t p = o % n;
    const int64_t x = (o / n) % n;
    const int64_t q = (o / (n * n)) % n;
    const int64_t a = o / (n * n * n);
    double acc = 0.0;
    for (int64_t f = 0; f < n; ++f) {
      const int64_t row = order == 0 ? idx4(p, q, x, f, n) : idx4(p, q, f, x, n);
      const double* const wRow = wbar + idx2(a * n + f, 0, width);
      const double* const lRow = l2 + idx2(row, 0, width);
      for (int64_t k = 0; k < width; ++k) acc += wRow[k] * lRow[k];
    }
    hbar[o] += acc;
  }
};

}  // namespace

wwrError_t f3WedgeAdjoint(const wwrStream_t stream, const int order, const double* g, double* gt,
                          const int norb) {
  const int64_t n = norb;
  const int64_t n6 = n * n * n * n * n * n;
  return ::wwr::extension::parallel_for(stream, n6, WedgeAdjointFunctor{g, gt, norb, order});
}

wwrError_t f3Fdm2Adjoint(const wwrStream_t stream, const double* g, double* fdm2bar,
                         const int norb) {
  const int64_t n = norb;
  return ::wwr::extension::parallel_for(stream, n * n * n * n,
                                        Fdm2AdjointFunctor{g, fdm2bar, norb});
}

wwrError_t f3Fdm2EriAdjoint(const wwrStream_t stream, const int order, const double* dm3,
                            const double* fdm2bar, double* hbar, const int norb) {
  const int64_t n = norb;
  return ::wwr::extension::parallel_for(stream, n * n * n * n,
                                        Fdm2EriAdjointFunctor{dm3, fdm2bar, hbar, norb, order});
}

wwrError_t f3ConsumeAdjoint(const wwrStream_t stream, const int order, const double* wbar,
                            const double* l2, double* hbar, const int norb, const int width) {
  const int64_t n = norb;
  return ::wwr::extension::parallel_for(
      stream, n * n * n * n, ConsumeAdjointFunctor{wbar, l2, hbar, norb, width, order});
}

}  // namespace nevpt2::device
