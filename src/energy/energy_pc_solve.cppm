// nevpt2.energy:pc_solve -- PC-NEVPT2's per-class solve: the d x d GEMM and
// symmetric eigendecomposition on DeviceResources' handles, the
// S^-1/2 K S^-1/2 solve built from them. (The denominator check and finish
// the classes share are nevpt2.energy_finish's.)
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :pc_solve;` and from no importer of nevpt2.energy.
// (Used by energy_pc_classes.cppm.)
module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.energy:pc_solve;

import std;
import nevpt2.energy;
// The wwrsolverStatus_t error specializations, which make gpuCheck accept a
// solver status (nevpt2.wwr does not re-export them).
import wwr.extension.solver;

namespace nevpt2 {

// (A + A^T) / 2 of a square (d, d) tensor, as a fresh tensor.
DeviceTensor symmetrised(const DeviceTensor& a, const DeviceResources& dr) {
  DeviceTensor s = deviceEinsumNew("ij->ij", {&a}, 0.5, dr);
  deviceTransposeAccum(a, {1, 0}, s, 0.5, dr.stream());
  return s;
}

// Row-major C (M, N) = op(A) op(B), op(X) = X or X^T, as a fresh tensor from
// the pool, on res.blas() (bound to res.stream()). A and B are contiguous
// row-major; transA means A is stored (K, M), transB that B is stored (N, K).
// Read column-major, row-major C is C^T = op(B)^T op(A)^T, so B goes first and
// each op flag carries over unchanged; each leading dimension is the stored
// row length. The d x d GEMMs of solveClass go through here, not
// deviceEinsumNew: at d = n^3 (Sr, Si) the generic kernel's one atomicAdd per
// (output, contracted index) made them the bulk of the PC stage (docs/pc-nevpt2.md,
// "The solve: once per class, not once per tuple").
DeviceTensor gemm(const DeviceTensor& A, bool transA, const DeviceTensor& B, bool transB,
                  const DeviceResources& dr,
                  std::source_location loc = std::source_location::current()) {
  // Every operand is a tensor this file built to shape, so a non-matrix or a
  // mismatch is our bug: check() aborts, naming the gemm call site.
  check(A.rank() == 2 && B.rank() == 2 && isContiguous(A) && isContiguous(B),
        "gemm needs two contiguous matrices", loc);
  const int64_t M = transA ? A.dims[1] : A.dims[0], K = transA ? A.dims[0] : A.dims[1];
  const int64_t N = transB ? B.dims[0] : B.dims[1];
  check((transB ? B.dims[1] : B.dims[0]) == K, "gemm: inner dimensions differ", loc);
  DeviceTensor C = DeviceTensor::zeros({M, N}, dr);
  const double one = 1.0, zero = 0.0;
  // The 32-bit BLAS API: each dimension narrowed once, checked.
  const int m = narrowTo<int>(M), n = narrowTo<int>(N), k = narrowTo<int>(K);
  const int ldb = narrowTo<int>(B.dims[1]), lda = narrowTo<int>(A.dims[1]);
  gpuCheck(wwrblasDgemm(dr.blas(), transB ? WWRBLAS_OP_T : WWRBLAS_OP_N,
                        transA ? WWRBLAS_OP_T : WWRBLAS_OP_N, n, m, k, &one, B.data(), ldb,
                        A.data(), lda, &zero, C.data(), n));
  return C;
}

// In-place symmetric eigendecomposition on res.solver(): `a` (d, d), symmetric,
// contiguous, is overwritten by the eigenvectors and `w` (d) receives the
// eigenvalues, ascending. The solver is column-major, so read row-major the
// output buffer is V^T: row k is eigenvector k. The workspace and devInfo come
// from the pool as DeviceBuffers (freed on the stream when this returns, after
// the solve that uses them); devInfo is read back with downloadTensor, which
// synchronizes the stream -- the host needs it before using the eigenvalues
// anyway, so this is the one sync, not an extra one.
//
// A solver call that fails is gpuCheck's (abort tier); a solve that RUNS but
// reports devInfo != 0 (no convergence, or an illegal argument) is a
// numerical refusal, returned as a value-tier Numerical Error naming the
// class, d and devInfo, which each main() reports once.
Status syevd(const DeviceTensor& a, const DeviceTensor& w, const char* what,
             const DeviceResources& dr) {
  // The solver's API is 32-bit (n, lda, lwork): d narrowed once, checked.
  const int d = narrowTo<int>(a.dims[0]);
  int lwork = 0;
  gpuCheck(wwrsolverDnDsyevd_bufferSize(dr.solver(), WWRSOLVER_EIG_MODE_VECTOR,
                                        WWRBLAS_FILL_MODE_LOWER, d, a.data(), d, w.data(), &lwork));
  DeviceBuffer<double> work(narrowTo<std::size_t>(std::max(lwork, 1)), dr.shared_from_this());
  // One int, held in a one-double tensor (zero-filled) so the readback is a
  // downloadTensor like every other: the solver writes the int at the start of
  // the 8 bytes, and the memcpy below reads those same bytes back.
  DeviceTensor info = DeviceTensor::zeros({1}, dr);
  gpuCheck(wwrsolverDnDsyevd(dr.solver(), WWRSOLVER_EIG_MODE_VECTOR, WWRBLAS_FILL_MODE_LOWER, d,
                             a.data(), d, w.data(), work.data(), lwork,
                             reinterpret_cast<int*>(info.data())));
  Tensor infoHost = downloadTensor(info, dr.stream());
  int devInfo = 0;
  double raw = infoHost.flat(0);
  std::memcpy(&devInfo, &raw, sizeof devInfo);
  if (devInfo != 0)
    return err_numerical(std::format("Dsyevd on {} (d={}) failed: devInfo={}", what, d, devInfo));
  return {};
}

// One class's solve: S and K (d, d), PySCF index order flattened row-major.
struct ClassSolve {
  PcClassResult result;        // status Refused (with why) if the gap check failed
  std::vector<double> lambda;  // (m) eigenvalues of X^T K_sym X, ascending
  DeviceTensor T;              // (d, m) S_sym U, what every slab is GEMM'd against
  DeviceTensor t1;             // (n1, m) S_1^T U, when an S_1 was given (else empty)
};

// `S1` (d, n1), optional: the metric column(s) the one-body part of V|Psi_0>
// projects through (docs/pc-nevpt2.md, "One idea"), so b_t = S x_t + S_1 h1_t and
// y_t = T^T x_t + t1^T h1_t.
// A Dsyevd devInfo != 0 propagates out as its Numerical Error; a refusal
// (gap, no positive eigenvalue) is a value, cs.result.status.
Result<ClassSolve> solveClass(const DeviceTensor& S, const DeviceTensor& K, const char* name,
                              const DeviceResources& dr, const DeviceTensor* S1 = nullptr) {
  const wwrStream_t s = dr.stream();
  const int64_t d = S.dims[0];
  ClassSolve cs;
  cs.result.hasSpectrum = true;
  cs.result.spectrum.d = d;

  // Only a quadratic form's symmetric part is defined; K is symmetric only to
  // the CI vector's convergence (docs/pc-nevpt2.md, "Symmetrisation").
  DeviceTensor Ssym = symmetrised(S, dr);
  DeviceTensor Ksym = symmetrised(K, dr);

  // 1. S = V s V^T.
  DeviceTensor Vt = deviceEinsumNew("ij->ij", {&Ssym}, 1.0, dr);  // overwritten: V^T
  DeviceTensor sEig = DeviceTensor::zeros({d}, dr);
  NEVPT2_TRY(syevd(Vt, sEig, name, dr));
  Tensor sh = downloadTensor(sEig, s);

  // 2. The cut and the gap check. Ascending, so the dropped modes are a prefix.
  const double smax = sh.flat(d - 1);
  PcSpectrum& sp = cs.result.spectrum;
  if (!(smax > 0.0)) {
    cs.result.status = PcStatus::Refused;
    cs.result.why = "S has no positive eigenvalue";
    return cs;
  }
  int64_t nd = 0;
  while (nd < d && sh.flat(nd) <= PC_TAU * smax) ++nd;
  sp.dropped = nd;
  for (int64_t k = 0; k < nd; ++k)
    sp.largestDropped = std::max(sp.largestDropped, std::fabs(sh.flat(k)) / smax);
  sp.smallestKept = sh.flat(nd) / smax;  // nd < d: s_max itself is always kept
  const double floor = nd > 0 ? sp.largestDropped : PC_TAU;
  sp.gap = floor > 0.0 ? sp.smallestKept / floor : std::numeric_limits<double>::infinity();
  if (sp.gap < PC_MIN_GAP) {
    cs.result.status = PcStatus::Refused;
    cs.result.why = std::format("S's gap {:.1e} < PC_MIN_GAP {:.0e}", sp.gap, PC_MIN_GAP);
    return cs;
  }
  const int64_t m = d - nd;

  // X^T = s_kept^-1/2 V_kept^T, (m, d): the kept rows of Vt, each scaled.
  Tensor sInvHalf({m});
  for (int64_t k = 0; k < m; ++k) sInvHalf.flatRef(k) = 1.0 / std::sqrt(sh.flat(nd + k));
  DeviceTensor dSInvHalf = uploadTensor(sInvHalf, dr);
  DeviceTensor VtKept = sliceAxis(Vt, 0, nd, d);
  DeviceTensor Xt = deviceEinsumNew("k,ki->ki", {&dSInvHalf, &VtKept}, 1.0, dr);

  // 3. K' = X^T K_sym X (two GEMMs), then K' = W lambda W^T.
  DeviceTensor KXt = gemm(Xt, false, Ksym, true, dr);  // (K X)^T: ij,kj->ki
  DeviceTensor Wt = gemm(Xt, false, KXt, true, dr);    // ki,li->kl, overwritten: W^T
  DeviceTensor lam = DeviceTensor::zeros({m}, dr);
  NEVPT2_TRY(syevd(Wt, lam, name, dr));
  Tensor lh = downloadTensor(lam, s);
  cs.lambda.assign(lh.data().begin(), lh.data().end());

  // 4.-5. U^T = W^T X^T, then T = S_sym U.
  DeviceTensor Ut = gemm(Wt, false, Xt, false, dr);  // kl,li->ki
  cs.T = gemm(Ssym, false, Ut, true, dr);            // pq,kq->pk
  if (S1 != nullptr) cs.t1 = deviceEinsumNew("ia,ki->ak", {S1, &Ut}, 1.0, dr);
  cs.result.status = PcStatus::Done;
  return cs;
}

// The denominator check (checkDenominator) and the Sr/Si finish
// (finishSingle) the classes share are pure host arithmetic:
// nevpt2.energy_finish's, re-exported by nevpt2.energy.

}  // namespace nevpt2
