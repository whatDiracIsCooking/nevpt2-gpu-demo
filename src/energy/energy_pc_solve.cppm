// nevpt2.energy:pc_solve -- PC-NEVPT2's per-class solve: the d x d GEMM and
// symmetric eigendecomposition on DeviceResources' handles, the
// S^-1/2 K S^-1/2 solve built from them, and the TANGENT of that solve
// (tangentClass) -- which differentiates the solve, never the
// eigendecomposition; the argument for why that is legitimate is at
// tangentClass below. (The denominator check and finish the classes share are
// nevpt2.energy_finish's.)
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :pc_solve;` and from no importer of nevpt2.energy.
// (Used by energy_pc_classes.cppm and energy_pc_tangent.cppm.)
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
  // (m, d) U^T, only when solveClass was asked to keep it (`keepUt`): the
  // tangent needs the eigenvectors themselves, to turn a metric/Hamiltonian
  // derivative into the (m, m) blocks tangentClass builds below. The energy
  // path does not, and holding a second (d, m) matrix through its slab walk
  // would raise its peak for no reader, so it is opt-in.
  DeviceTensor Ut;
  // (nd, d) the DROPPED eigenvectors of S_sym, row k one of them, only when
  // solveClass was asked to keep them (`keepDropped`) and something was
  // dropped: the second derivative's rotation terms (energy_pc_hessian.cppm)
  // need the dropped-space projector P = Vn^T Vn. Empty otherwise.
  DeviceTensor Vn;
  double sMax = 0.0;  // S_sym's largest eigenvalue, what the cut is relative to
};

// `S1` (d, n1), optional: the metric column(s) the one-body part of V|Psi_0>
// projects through (docs/pc-nevpt2.md, "One idea"), so b_t = S x_t + S_1 h1_t and
// y_t = T^T x_t + t1^T h1_t.
// A Dsyevd devInfo != 0 propagates out as its Numerical Error; a refusal
// (gap, no positive eigenvalue) is a value, cs.result.status.
Result<ClassSolve> solveClass(const DeviceTensor& S, const DeviceTensor& K, const char* name,
                              const DeviceResources& dr, const DeviceTensor* S1 = nullptr,
                              bool keepUt = false, bool keepDropped = false) {
  const wwrStream_t s = dr.stream();
  const int64_t d = S.dims[0];
  ClassSolve cs;  // cs.result is filled by pcRankCut, step 2

  // Only a quadratic form's symmetric part is defined; K is symmetric only to
  // the CI vector's convergence (docs/pc-nevpt2.md, "Symmetrisation").
  DeviceTensor Ssym = symmetrised(S, dr);
  DeviceTensor Ksym = symmetrised(K, dr);

  // 1. S = V s V^T.
  DeviceTensor Vt = deviceEinsumNew("ij->ij", {&Ssym}, 1.0, dr);  // overwritten: V^T
  DeviceTensor sEig = DeviceTensor::zeros({d}, dr);
  NEVPT2_TRY(syevd(Vt, sEig, name, dr));
  Tensor sh = downloadTensor(sEig, s);

  // 2. The cut and the gap check: nevpt2.energy_finish's pcRankCut, unit-tested
  // card-free. Ascending, so the dropped modes are a prefix of sh.
  cs.result = pcRankCut(sh.data());
  if (cs.result.status == PcStatus::Refused) return cs;
  const int64_t nd = cs.result.spectrum.dropped;
  const int64_t m = d - nd;
  cs.sMax = sh.flat(d - 1);
  // A copy, not a view: Vt is freed when this returns.
  if (keepDropped && nd > 0) {
    const DeviceTensor VtDrop = sliceAxis(Vt, 0, 0, nd);
    cs.Vn = deviceEinsumNew("ij->ij", {&VtDrop}, 1.0, dr);
  }

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
  // After T and t1, which read it: a tangent caller asked to keep the
  // eigenvectors, so the buffer lives on in `cs` instead of being freed here.
  if (keepUt) cs.Ut = std::move(Ut);
  cs.result.status = PcStatus::Done;
  return cs;
}

// --- the tangent of that solve ------------------------------------------------
//
// WHAT IS DIFFERENTIATED, AND WHY IT IS LEGITIMATE. A class energy is
//
//   E_t = -c * b_t^T A_t^+ b_t,   A_t = K + Delta_t S,   b_t = S x_t + S_1 h1_t
//
// (docs/pc-nevpt2.md, "One idea"). Only S, K and S_1 carry the CI vector, so
// with x_vec = A_t^+ b_t the directional derivative is
//
//   Edot_t = -c * (2 x_vec^T bdot_t - x_vec^T Adot_t x_vec),
//
// which names NO eigenvector derivative: the solve already performed is all
// of it. The solve is not A^+ but S's cut restricted inverse
// M = Q (Q^T A Q)^-1 Q^T, differentiable where S's rank is locally constant
// -- PC_MIN_GAP's three decades between the dropped modes and the smallest
// kept one (docs/pc-nevpt2.md, "The singular-metric convention"). Its
// derivative (docs/hessian-theory.md, Sec. 6.2) is -M Adot M plus terms in
// P Sdot (P the dropped-space projector), because the dropped space is null
// for THIS state's metric and turns with the state, not null for every CI
// vector (docs/pc-nevpt2.md, "What the cut drops, and how it moves").
// Sandwiched on b_t (in S's range) those leave one term, -2 rho^T Sdot P A x
// with rho = S^+ x, and it vanishes where P A = 0: at the eigenstate, which
// is where this tangent is taken. The second derivative carries it
// (energy_pc_hessian.cppm). Where the gap is
// absent the class is refused, and the tangent is refused with it through the
// same PcStatus::Refused, rather than handing back a number whose rank
// assumption nobody checked.
//
// In the eigenbasis of the solve (U, lambda), with g_tk = y_tk /
// (lambda_k + Delta_t) so that x_vec = U g_t,
//
//   x_vec^T bdot_t     = g_t . ydot_t,   ydot_t = Tdot^T x_t + t1dot^T h1_t
//   x_vec^T Adot x_vec = g_t^T (mK + Delta_t nS) g_t
//
// so one extra slab einsum (against Tdot in place of T) and two (m, m) blocks
// are the whole tangent. nevpt2.energy_finish's pcCoefficients / pcQuadForm /
// tangentSingle are the host side of those two lines.
struct ClassTangent {
  DeviceTensor Tdot;       // (d, m) sym(Sdot) U -- T's counterpart in the slab einsum
  DeviceTensor t1dot;      // (n1, m) S_1dot^T U, when an S1dot was given (else empty)
  std::vector<double> mK;  // (m, m) row-major U^T sym(Kdot) U
  std::vector<double> nS;  // (m, m) row-major U^T sym(Sdot) U
};

// `cs` must come from solveClass(..., keepUt=true) with status Done;
// `Sdot`/`Kdot` are (d, d) and `S1dot` (d, n1) -- the tangents of the very
// matrices that solve was handed. A quadratic form defines only the symmetric
// part and the energy path symmetrises S and K, so this symmetrises their
// derivatives the same way (sym is linear, so sym(X)dot = sym(Xdot)).
// The two (m, m) blocks come back to the host: each tuple's finish needs a
// matrix-vector product with them, and that reduction stays where
// finishSingle is (docs/pc-nevpt2.md, "The solve: once per class, not once
// per tuple").
ClassTangent tangentClass(const ClassSolve& cs, const DeviceTensor& Sdot,
                          const DeviceTensor& Kdot, const DeviceResources& dr,
                          const DeviceTensor* S1dot = nullptr) {
  // Both are this file's own invariants -- the caller is the tangent class
  // code beside it -- so they abort rather than return an Error.
  check(cs.result.status == PcStatus::Done, "tangentClass: the class was not solved");
  check(cs.Ut.rank() == 2, "tangentClass needs solveClass(keepUt=true)'s eigenvectors");
  const wwrStream_t s = dr.stream();
  ClassTangent ct;

  DeviceTensor Sds = symmetrised(Sdot, dr);
  ct.Tdot = gemm(Sds, false, cs.Ut, true, dr);  // pq,kq->pk
  {
    DeviceTensor nS = gemm(cs.Ut, false, ct.Tdot, false, dr);  // kp,pl->kl
    const Tensor host = downloadTensor(nS, s);
    ct.nS.assign(host.data().begin(), host.data().end());
  }
  {
    // Scoped so K's (d, d) and (d, m) intermediates are freed before the
    // caller's slab walk begins.
    DeviceTensor Kds = symmetrised(Kdot, dr);
    DeviceTensor KdU = gemm(Kds, false, cs.Ut, true, dr);  // pq,kq->pk
    DeviceTensor mK = gemm(cs.Ut, false, KdU, false, dr);  // kp,pl->kl
    const Tensor host = downloadTensor(mK, s);
    ct.mK.assign(host.data().begin(), host.data().end());
  }
  if (S1dot != nullptr) ct.t1dot = deviceEinsumNew("ia,ki->ak", {S1dot, &cs.Ut}, 1.0, dr);
  return ct;
}

// The denominator check (checkDenominator) and the Sr/Si finish
// (finishSingle) the classes share are pure host arithmetic:
// nevpt2.energy_finish's, re-exported by nevpt2.energy.

}  // namespace nevpt2
