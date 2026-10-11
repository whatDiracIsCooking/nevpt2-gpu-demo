// nevpt2.energy:pc_tangent -- the PC-NEVPT2 class energies' DIRECTIONAL
// DERIVATIVE with respect to the CI vector: each class's metric S, Dyall
// Hamiltonian K and one-body metric column S_1 from one bundle of RDMs, the
// tangents of those three, and the slab walk that turns them into Edot
// beside E.
// An internal partition: nothing here is exported, so it is reachable from
// the units that `import :pc_tangent;` and from no importer of nevpt2.energy.
// (Used by energy_pc.cpp, which defines pcTangentsDevice over these.)
//
// WHAT IS AND IS NOT DIFFERENTIATED. The solve, never the
// eigendecomposition: the per-class derivative is
//
//   Edot_t = -c * (2 x_vec^T bdot_t - x_vec^T Adot_t x_vec),  x_vec = A_t^+ b_t
//
// with A_t = K + Delta_t S and b_t = S x_t + S_1 h1_t -- only the solve
// pcEnergiesDevice already ran, no dU/dh and no dlambda/dh, so no degeneracy
// case to handle. A^+ is a pseudo-inverse and is differentiable only where
// the rank is locally constant, which is exactly what PC_MIN_GAP enforces
// (docs/pc-nevpt2.md, "The singular-metric convention"): the dropped modes
// are STRUCTURAL -- linear dependencies among the class's operator strings,
// null for every CI vector -- and three decades separate them from the
// smallest kept mode, so the cut cannot move under a small change of the
// state. The tangent is therefore valid precisely on the states
// pcEnergiesDevice accepts, and on no others: a class refused by the gap or
// denominator check comes back Refused here too (PcClassTangent's `result`
// carries that status and the reason), never as a silent zero. The full
// derivation, including why the two extra terms of the general pseudo-inverse
// derivative drop, is at `tangentClass` in energy_pc_solve.cppm.
//
// IT IS THE SAME CLASSES, run twice. Each class builds its (S, K, S_1) from
// the RDM bundle through nevpt2.energy's exported make_* intermediates -- the
// same functions and the same subscripts energy_pc_classes.cppm uses, so the
// two can be read side by side -- then builds them a second time from the
// SUMMED bundle (dm + dm_dot) and subtracts. That subtraction IS the tangent,
// exactly: every term of every make_* contracts constants (the active
// integrals, a delta) with exactly ONE RDM factor, so S, K and S_1 are affine
// in the RDMs (see `tangentMatrices`). The report cross-checks this unit's
// class energies against pcEnergiesDevice's own to 1e-10, so a transcription
// that drifted from :pc_classes fails rather than passing quietly.
module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.energy:pc_tangent;

import std;
import nevpt2.energy;
import :pc_solve;

namespace nevpt2 {

// --- the RDM bundle and one class's matrices ----------------------------------

// The active-space RDMs a class's matrices are built from: non-owning, the
// caller (pcTangentsDevice) owns every tensor. dm1/dm2 are its uploads,
// dm3/f3ac/f3ca the RDM build's own output -- exactly energiesDevice's
// inputs.
struct PcRdms {
  const DeviceTensor* dm1 = nullptr;
  const DeviceTensor* dm2 = nullptr;
  const DeviceTensor* dm3 = nullptr;
  const DeviceTensor* f3ac = nullptr;
  const DeviceTensor* f3ca = nullptr;
};

// One class's three matrices, each in its natural PySCF index shape and
// owning its buffer; `matS`/`matK`/`matS1` are the flattened (d, d) /
// (d, n1) VIEWS solveClass and tangentClass take (reshapeView: no copy, and
// valid only while this struct lives).
struct PcMatrices {
  DeviceTensor S;
  DeviceTensor K;
  DeviceTensor S1;        // empty unless hasS1
  int64_t d = 0;          // the flattened basis dimension
  int64_t n1 = 0;         // S_1's column count (0 unless hasS1)
  bool hasS1 = false;

  DeviceTensor matS() const { return reshapeView(S, {d, d}); }
  DeviceTensor matK() const { return reshapeView(K, {d, d}); }
  DeviceTensor matS1() const { return reshapeView(S1, {d, n1}); }
};

// x + y as a fresh tensor from the pool: the summed RDM bundle the tangent
// matrices are built from (used by energy_pc.cpp's pcTangentsDevice). Same
// shape in, same shape out.
DeviceTensor sumTensors(const DeviceTensor& x, const DeviceTensor& y,
                        const DeviceResources& dr) {
  check(x.dims == y.dims, "sumTensors: a tensor and its tangent must have one shape");
  std::vector<int> axes(x.dims.size());
  std::iota(axes.begin(), axes.end(), 0);
  DeviceTensor out = deviceTransposeNew(x, axes, 1.0, dr);
  deviceTransposeAccum(y, axes, out, 1.0, dr.stream());
  return out;
}

// dot <- dot - base, elementwise, over all three matrices: the tangent of
// each, in place.
//
// WHY ONE SUBTRACTION IS THE DERIVATIVE, EXACTLY. Every term of every matrix
// built below contracts constants -- the active integrals h1e/h2e, or a
// Kronecker delta -- with exactly ONE RDM factor; none of nevpt2.energy's
// make_* multiplies two RDM-derived tensors together (make_hdm3 and make_rm3
// read hdm2 / rm2, each itself affine, against a delta; make_a7 reads rm2 and
// rm3 against the integrals). So each matrix is AFFINE in the RDM bundle,
// X(D) = L(D) + C with L linear, and
//
//   X(D + D_dot) - X(D) = L(D_dot),
//
// which is the directional derivative -- an algebraic identity, not a finite
// difference: no step size enters and there is no truncation error. The
// constant C (make_hdm1's 2 delta, make_hdm2's two delta-delta terms, and
// what make_a3 / make_a9 inherit from them) cancels in the subtraction, which
// is why the tangent RDMs cannot simply be pushed through the same builder
// alone.
void tangentMatrices(PcMatrices& dot, const PcMatrices& base, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  check(dot.d == base.d && dot.hasS1 == base.hasS1,
        "tangentMatrices: the two builds must be the same class");
  const DeviceTensor bS = base.matS(), dS = dot.matS();
  deviceEinsumAccum("ij->ij", {&bS}, dS, -1.0, s);
  const DeviceTensor bK = base.matK(), dK = dot.matK();
  deviceEinsumAccum("ij->ij", {&bK}, dK, -1.0, s);
  if (base.hasS1) {
    const DeviceTensor bS1 = base.matS1(), dS1 = dot.matS1();
    deviceEinsumAccum("ij->ij", {&bS1}, dS1, -1.0, s);
  }
}

// --- each class's S, K and S_1 ------------------------------------------------
//
// The same intermediates, subscripts and block patterns as
// energy_pc_classes.cppm's pc_<class>, read off docs/pc-nevpt2.md, "Per
// class", with the RDMs coming from a bundle so the tangent can build them
// twice.

// Sr: basis (p, q, r), d = n^3; S = dm3, K = a16, S_1 = dm2 as (pqr, a).
PcMatrices pcMatricesSr(const ActiveIntegralsDevice& ai, const PcRdms& r,
                        const DeviceResources& dr) {
  const int64_t n = r.dm2->dims[0];
  PcMatrices mx;
  mx.d = n * n * n;
  mx.n1 = n;
  mx.hasS1 = true;
  mx.K = make_a16(ai.h1e, ai.h2e, *r.dm3, *r.f3ac, *r.f3ca, n, dr);
  mx.S = deviceEinsumNew("rpqbac->pqrabc", {r.dm3}, 1.0, dr);
  mx.S1 = deviceEinsumNew("rpqa->pqra", {r.dm2}, 1.0, dr);
  return mx;
}

// Si: basis (p, q, r), d = n^3; S = dm3_h, K = a22, S_1 = dm2_h as (pqr, a).
PcMatrices pcMatricesSi(const ActiveIntegralsDevice& ai, const PcRdms& r,
                        const DeviceResources& dr) {
  const int64_t n = r.dm2->dims[0];
  PcMatrices mx;
  mx.d = n * n * n;
  mx.n1 = n;
  mx.hasS1 = true;
  mx.K = make_a22(ai.h1e, ai.h2e, *r.dm2, *r.dm3, *r.f3ac, *r.f3ca, n, dr);
  {
    DeviceTensor dm3_h = make_dm3_h(*r.dm2, *r.dm3, dr);
    mx.S = deviceEinsumNew("rpqbac->pqrabc", {&dm3_h}, 1.0, dr);
  }
  {
    DeviceTensor dm2_h = make_dm2_h(*r.dm1, *r.dm2, dr);
    mx.S1 = deviceEinsumNew("rpqa->pqra", {&dm2_h}, 1.0, dr);
  }
  return mx;
}

// Sijr: basis p, d = n; S = hdm1, K = a3, no S_1.
PcMatrices pcMatricesSijr(const ActiveIntegralsDevice& ai, const PcRdms& r,
                          const DeviceResources& dr) {
  PcMatrices mx;
  mx.d = r.dm1->dims[0];
  DeviceTensor hdm1 = make_hdm1(*r.dm1, dr);
  mx.K = make_a3(ai.h1e, ai.h2e, *r.dm1, *r.dm2, hdm1, dr);
  mx.S = std::move(hdm1);
  return mx;
}

// Srsi: basis p, d = n; S = dm1, K = k27, no S_1. S is a COPY of dm1, not a
// view of it: PcMatrices owns what it hands to tangentMatrices, which
// overwrites the summed build in place.
PcMatrices pcMatricesSrsi(const ActiveIntegralsDevice& ai, const PcRdms& r,
                          const DeviceResources& dr) {
  PcMatrices mx;
  mx.d = r.dm1->dims[0];
  mx.K = make_k27(ai.h1e, ai.h2e, *r.dm1, *r.dm2, dr);
  mx.S = deviceEinsumNew("pq->pq", {r.dm1}, 1.0, dr);
  return mx;
}

// Srs: basis (p, q), d = n^2; S = rm2 as pqab, K = a7, no S_1.
PcMatrices pcMatricesSrs(const ActiveIntegralsDevice& ai, const PcRdms& r,
                         const DeviceResources& dr) {
  const int64_t n = r.dm1->dims[0];
  PcMatrices mx;
  mx.d = n * n;
  auto [rm2, a7] = make_a7(ai.h1e, ai.h2e, *r.dm1, *r.dm2, *r.dm3, dr);
  mx.S = deviceEinsumNew("pqba->pqab", {&rm2}, 1.0, dr);
  mx.K = std::move(a7);
  return mx;
}

// Sij: basis (p, q), d = n^2; S = hdm2, K = a9, no S_1.
PcMatrices pcMatricesSij(const ActiveIntegralsDevice& ai, const PcRdms& r,
                         const DeviceResources& dr) {
  const int64_t n = r.dm1->dims[0];
  PcMatrices mx;
  mx.d = n * n;
  DeviceTensor hdm1 = make_hdm1(*r.dm1, dr);
  DeviceTensor hdm2 = make_hdm2(*r.dm1, *r.dm2, dr);
  {
    DeviceTensor hdm3 = make_hdm3(*r.dm1, *r.dm2, *r.dm3, hdm1, hdm2, dr);
    mx.K = make_a9(ai.h1e, ai.h2e, hdm1, hdm2, hdm3, dr);
  }
  mx.S = std::move(hdm2);
  return mx;
}

// Sir: basis (p, q) (+) (p, q), d = 2 n^2, held as (x, p, q, y, a, b) with
// x, y the spin-coupling block; S_1 is one column.
PcMatrices pcMatricesSir(const ActiveIntegralsDevice& ai, const PcRdms& r,
                         const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = r.dm1->dims[0];
  PcMatrices mx;
  mx.d = 2 * n * n;
  mx.n1 = 1;
  mx.hasS1 = true;

  DeviceTensor offDiag = uploadTensor(Tensor::fromFlat({2, 2}, {2.0, -1.0, -1.0, 0.0}), dr);
  DeviceTensor block22 = uploadTensor(Tensor::fromFlat({2, 2}, {0.0, 0.0, 0.0, 1.0}), dr);
  DeviceTensor oneBody = uploadTensor(Tensor::fromFlat({2}, {2.0, -1.0}), dr);
  DeviceTensor delta = deviceEye(n, dr);

  DeviceTensor S22 = deviceEinsumNew("pa,qb->pqab", {&delta, r.dm1}, 2.0, dr);
  deviceEinsumAccum("qbap->pqab", {r.dm2}, S22, -1.0, s);
  deviceEinsumAccum("ab,qp->pqab", {&delta, r.dm1}, S22, 1.0, s);
  mx.S = deviceEinsumNew("xy,qpab->xpqyab", {&offDiag, r.dm2}, 1.0, dr);
  deviceEinsumAccum("xy,pqab->xpqyab", {&block22, &S22}, mx.S, 1.0, s);

  {
    DeviceTensor a12 = make_a12(ai.h1e, ai.h2e, *r.dm1, *r.dm2, *r.dm3, dr);
    DeviceTensor a13 = make_a13(ai.h1e, ai.h2e, *r.dm1, *r.dm2, *r.dm3, dr);
    mx.K = deviceEinsumNew("xy,pqab->xpqyab", {&offDiag, &a12}, 1.0, dr);
    deviceEinsumAccum("xy,pqab->xpqyab", {&block22, &a13}, mx.K, 1.0, s);
  }
  mx.S1 = deviceEinsumNew("x,qp->xpq", {&oneBody, r.dm1}, 1.0, dr);
  return mx;
}

// --- the per-class tangents ---------------------------------------------------
//
// Each: build the class's matrices from `rdm`, solve (keeping the
// eigenvectors), build them again from `sum` = rdm + its tangent, subtract to
// get the matrix tangents, turn those into Tdot / t1dot / mK / nS
// (tangentClass), and walk the slabs once -- the energy's einsum against T
// and the same einsum against Tdot -- accumulating E and Edot together.
// A class the solve refuses returns that refusal with edot untouched (0 and
// meaningless, which `result.status` says).

// Sr: Delta = e_virt[r], factor 1.
Result<PcClassTangent> pcDotSr(const ActiveIntegralsDevice& ai, IntegralSource& src,
                               int64_t batch, const Tensor& ev, const PcRdms& rdm,
                               const PcRdms& sum, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = rdm.dm2->dims[0];
  PcClassTangent out;
  ClassSolve cs;
  ClassTangent ct;
  {
    PcMatrices base = pcMatricesSr(ai, rdm, dr);
    const DeviceTensor baseS1 = base.matS1();
    cs = NEVPT2_TRY(solveClass(base.matS(), base.matK(), "Sr", dr, &baseS1, /*keepUt=*/true));
    if (cs.result.status != PcStatus::Done) {
      out.result = cs.result;
      return out;
    }
    PcMatrices dot = pcMatricesSr(ai, sum, dr);
    tangentMatrices(dot, base, dr);
    const DeviceTensor dotS1 = dot.matS1();
    ct = tangentClass(cs, dot.matS(), dot.matK(), dr, &dotS1);
  }
  const int64_t m = std::ssize(cs.lambda);
  const DeviceTensor T = reshapeView(cs.T, {n, n, n, m});
  const DeviceTensor Tdot = reshapeView(ct.Tdot, {n, n, n, m});

  double e = 0.0, edot = 0.0, minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Sr), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sr, b0, b1, dr);  // h[r_batch, p, q, r]
    DeviceTensor h1 = sliceAxis(ai.h1e_v_Sr, 0, b0, b1);     // (nr, a)
    DeviceTensor Y = deviceEinsumNew("ipqr,pqrk->ik", {&slab, &T}, 1.0, dr);
    deviceEinsumAccum("ia,ak->ik", {&h1, &cs.t1}, Y, 1.0, s);
    DeviceTensor Yd = deviceEinsumNew("ipqr,pqrk->ik", {&slab, &Tdot}, 1.0, dr);
    deviceEinsumAccum("ia,ak->ik", {&h1, &ct.t1dot}, Yd, 1.0, s);
    const Tensor y = downloadTensor(Y, s);    // (nr, m)
    const Tensor yd = downloadTensor(Yd, s);  // (nr, m)
    std::vector<double> delta(ev.data().begin() + b0, ev.data().begin() + b1);
    tangentSingle(y, yd, cs.lambda, delta, ct.mK, ct.nS, e, edot, minDen);
  });
  out.result = cs.result;
  out.result.energy = e;
  out.edot = edot;
  checkDenominator(out.result, minDen);
  return out;
}

// Si: Delta = -e_core[i], factor 1.
Result<PcClassTangent> pcDotSi(const ActiveIntegralsDevice& ai, IntegralSource& src,
                               int64_t batch, const Tensor& ec, const PcRdms& rdm,
                               const PcRdms& sum, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = rdm.dm2->dims[0];
  PcClassTangent out;
  ClassSolve cs;
  ClassTangent ct;
  {
    PcMatrices base = pcMatricesSi(ai, rdm, dr);
    const DeviceTensor baseS1 = base.matS1();
    cs = NEVPT2_TRY(solveClass(base.matS(), base.matK(), "Si", dr, &baseS1, /*keepUt=*/true));
    if (cs.result.status != PcStatus::Done) {
      out.result = cs.result;
      return out;
    }
    PcMatrices dot = pcMatricesSi(ai, sum, dr);
    tangentMatrices(dot, base, dr);
    const DeviceTensor dotS1 = dot.matS1();
    ct = tangentClass(cs, dot.matS(), dot.matK(), dr, &dotS1);
  }
  const int64_t m = std::ssize(cs.lambda);
  const DeviceTensor T = reshapeView(cs.T, {n, n, n, m});
  const DeviceTensor Tdot = reshapeView(ct.Tdot, {n, n, n, m});

  double e = 0.0, edot = 0.0, minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Si), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Si, b0, b1, dr);  // h[q, p, i_batch, r]
    DeviceTensor h1 = sliceAxis(ai.h1e_v_Si, 1, b0, b1);     // (a, ni)
    DeviceTensor Y = deviceEinsumNew("qpir,pqrk->ik", {&slab, &T}, 1.0, dr);
    deviceEinsumAccum("ai,ak->ik", {&h1, &cs.t1}, Y, 1.0, s);
    DeviceTensor Yd = deviceEinsumNew("qpir,pqrk->ik", {&slab, &Tdot}, 1.0, dr);
    deviceEinsumAccum("ai,ak->ik", {&h1, &ct.t1dot}, Yd, 1.0, s);
    const Tensor y = downloadTensor(Y, s);
    const Tensor yd = downloadTensor(Yd, s);
    std::vector<double> delta(ec.data().begin() + b0, ec.data().begin() + b1);
    for (double& x : delta) x = -x;
    tangentSingle(y, yd, cs.lambda, delta, ct.mK, ct.nS, e, edot, minDen);
  });
  out.result = cs.result;
  out.result.energy = e;
  out.edot = edot;
  checkDenominator(out.result, minDen);
  return out;
}

// Sijr: Delta = e_v[r] - e_c[j] - e_c[i], the two spin couplings summed over
// ORDERED (j, i) -- so per tuple the energy is
// -(2 b_u^T A^+ b_u - b_u^T A^+ b_v) with v the (i, j)-swapped row, and the
// tangent is that bilinear's derivative:
//   d(b_1^T A^+ b_2) = g_1 . ydot_2 + g_2 . ydot_1 - g_1^T (mK + Delta nS) g_2.
Result<PcClassTangent> pcDotSijr(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                 int64_t batch, const Tensor& ec, const Tensor& ev,
                                 const PcRdms& rdm, const PcRdms& sum,
                                 const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  PcClassTangent out;
  ClassSolve cs;
  ClassTangent ct;
  {
    PcMatrices base = pcMatricesSijr(ai, rdm, dr);
    cs = NEVPT2_TRY(solveClass(base.matS(), base.matK(), "Sijr", dr, nullptr, /*keepUt=*/true));
    if (cs.result.status != PcStatus::Done) {
      out.result = cs.result;
      return out;
    }
    PcMatrices dot = pcMatricesSijr(ai, sum, dr);
    tangentMatrices(dot, base, dr);
    ct = tangentClass(cs, dot.matS(), dot.matK(), dr);
  }
  const int64_t m = std::ssize(cs.lambda);
  const int64_t nc = std::ssize(ec);

  double e = 0.0, edot = 0.0, minDen = std::numeric_limits<double>::infinity();
  std::vector<double> gu, gv;
  forBatches(src.extent(ExtBlock::Sijr), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sijr, b0, b1, dr);  // (nr, a, c, c)
    DeviceTensor Y = deviceEinsumNew("rpji,pk->rjik", {&slab, &cs.T}, 1.0, dr);
    DeviceTensor Yd = deviceEinsumNew("rpji,pk->rjik", {&slab, &ct.Tdot}, 1.0, dr);
    const Tensor y = downloadTensor(Y, s);    // (nr, c, c, m)
    const Tensor yd = downloadTensor(Yd, s);
    const int64_t nr = b1 - b0;
    for (int64_t r = 0; r < nr; ++r)
      for (int64_t j = 0; j < nc; ++j)
        for (int64_t i = 0; i < nc; ++i) {
          const double delta = ev.flat(b0 + r) - ec.flat(j) - ec.flat(i);
          // The direct row (r, j, i) and the exchange row (r, i, j); both
          // carry the same Delta, so the same denominators.
          const int64_t offU = ((r * nc + j) * nc + i) * m;
          const int64_t offV = ((r * nc + i) * nc + j) * m;
          const std::span<const double> yu(y.data().data() + offU, narrowTo<std::size_t>(m));
          const std::span<const double> yv(y.data().data() + offV, narrowTo<std::size_t>(m));
          const std::span<const double> ydu(yd.data().data() + offU, narrowTo<std::size_t>(m));
          const std::span<const double> ydv(yd.data().data() + offV, narrowTo<std::size_t>(m));
          pcCoefficients(yu, cs.lambda, delta, gu, minDen);
          pcCoefficients(yv, cs.lambda, delta, gv, minDen);
          double uu = 0.0, uv = 0.0, duu = 0.0, duv = 0.0, dvu = 0.0;
          for (int64_t k = 0; k < m; ++k) {
            const std::size_t uk = narrowTo<std::size_t>(k);
            uu += gu[uk] * yu[uk];
            uv += gu[uk] * yv[uk];
            duu += gu[uk] * ydu[uk];
            duv += gu[uk] * ydv[uk];
            dvu += gv[uk] * ydu[uk];
          }
          e -= 2.0 * uu - uv;
          const double qUU = pcQuadForm(gu, gu, ct.mK, ct.nS, delta);
          const double qUV = pcQuadForm(gu, gv, ct.mK, ct.nS, delta);
          edot -= 2.0 * (2.0 * duu - qUU) - (duv + dvu - qUV);
        }
  });
  out.result = cs.result;
  out.result.energy = e;
  out.edot = edot;
  checkDenominator(out.result, minDen);
  return out;
}

// Srsi: Delta = e_v[r] + e_v[s] - e_c[i], the same two couplings with the
// exchange row coming from the SrsiT slab instead of an index swap.
Result<PcClassTangent> pcDotSrsi(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                 int64_t batch, const Tensor& ec, const Tensor& ev,
                                 const PcRdms& rdm, const PcRdms& sum,
                                 const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  PcClassTangent out;
  ClassSolve cs;
  ClassTangent ct;
  {
    PcMatrices base = pcMatricesSrsi(ai, rdm, dr);
    cs = NEVPT2_TRY(solveClass(base.matS(), base.matK(), "Srsi", dr, nullptr, /*keepUt=*/true));
    if (cs.result.status != PcStatus::Done) {
      out.result = cs.result;
      return out;
    }
    PcMatrices dot = pcMatricesSrsi(ai, sum, dr);
    tangentMatrices(dot, base, dr);
    ct = tangentClass(cs, dot.matS(), dot.matK(), dr);
  }
  const int64_t m = std::ssize(cs.lambda);
  const int64_t nc = std::ssize(ec), nv = std::ssize(ev);

  double e = 0.0, edot = 0.0, minDen = std::numeric_limits<double>::infinity();
  std::vector<double> gu, gv;
  forBatches(src.extent(ExtBlock::Srsi), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Srsi, b0, b1, dr);    // h[r_batch, s, i, p]
    DeviceTensor slabT = src.slab(ExtBlock::SrsiT, b0, b1, dr);  // h[s, r_batch, i, p]
    DeviceTensor Y = deviceEinsumNew("rsip,pk->rsik", {&slab, &cs.T}, 1.0, dr);
    DeviceTensor YT = deviceEinsumNew("srip,pk->rsik", {&slabT, &cs.T}, 1.0, dr);
    DeviceTensor Yd = deviceEinsumNew("rsip,pk->rsik", {&slab, &ct.Tdot}, 1.0, dr);
    DeviceTensor YTd = deviceEinsumNew("srip,pk->rsik", {&slabT, &ct.Tdot}, 1.0, dr);
    const Tensor y = downloadTensor(Y, s);  // (nr, v, c, m)
    const Tensor yt = downloadTensor(YT, s);
    const Tensor yd = downloadTensor(Yd, s);
    const Tensor ytd = downloadTensor(YTd, s);
    const int64_t nr = b1 - b0;
    for (int64_t r = 0; r < nr; ++r)
      for (int64_t ss = 0; ss < nv; ++ss)
        for (int64_t i = 0; i < nc; ++i) {
          const double delta = ev.flat(b0 + r) + ev.flat(ss) - ec.flat(i);
          const int64_t off = ((r * nv + ss) * nc + i) * m;
          const std::span<const double> yu(y.data().data() + off, narrowTo<std::size_t>(m));
          const std::span<const double> yv(yt.data().data() + off, narrowTo<std::size_t>(m));
          const std::span<const double> ydu(yd.data().data() + off, narrowTo<std::size_t>(m));
          const std::span<const double> ydv(ytd.data().data() + off, narrowTo<std::size_t>(m));
          pcCoefficients(yu, cs.lambda, delta, gu, minDen);
          pcCoefficients(yv, cs.lambda, delta, gv, minDen);
          double uu = 0.0, uv = 0.0, duu = 0.0, duv = 0.0, dvu = 0.0;
          for (int64_t k = 0; k < m; ++k) {
            const std::size_t uk = narrowTo<std::size_t>(k);
            uu += gu[uk] * yu[uk];
            uv += gu[uk] * yv[uk];
            duu += gu[uk] * ydu[uk];
            duv += gu[uk] * ydv[uk];
            dvu += gv[uk] * ydu[uk];
          }
          e -= 2.0 * uu - uv;
          const double qUU = pcQuadForm(gu, gu, ct.mK, ct.nS, delta);
          const double qUV = pcQuadForm(gu, gv, ct.mK, ct.nS, delta);
          edot -= 2.0 * (2.0 * duu - qUU) - (duv + dvu - qUV);
        }
  });
  out.result = cs.result;
  out.result.energy = e;
  out.edot = edot;
  checkDenominator(out.result, minDen);
  return out;
}

// Srs: Delta = e_v[r] + e_v[s], factor 1/2 on every ordered (r, s).
Result<PcClassTangent> pcDotSrs(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                int64_t batch, const Tensor& ev, const PcRdms& rdm,
                                const PcRdms& sum, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = rdm.dm1->dims[0], nv = std::ssize(ev);
  PcClassTangent out;
  ClassSolve cs;
  ClassTangent ct;
  {
    PcMatrices base = pcMatricesSrs(ai, rdm, dr);
    cs = NEVPT2_TRY(solveClass(base.matS(), base.matK(), "Srs", dr, nullptr, /*keepUt=*/true));
    if (cs.result.status != PcStatus::Done) {
      out.result = cs.result;
      return out;
    }
    PcMatrices dot = pcMatricesSrs(ai, sum, dr);
    tangentMatrices(dot, base, dr);
    ct = tangentClass(cs, dot.matS(), dot.matK(), dr);
  }
  const int64_t m = std::ssize(cs.lambda);
  const DeviceTensor T = reshapeView(cs.T, {n, n, m});
  const DeviceTensor Tdot = reshapeView(ct.Tdot, {n, n, m});

  double e = 0.0, edot = 0.0, minDen = std::numeric_limits<double>::infinity();
  std::vector<double> g;
  forBatches(src.extent(ExtBlock::Srs), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Srs, b0, b1, dr);  // h[r_batch, s, q, p]
    DeviceTensor Y = deviceEinsumNew("rsqp,pqk->rsk", {&slab, &T}, 1.0, dr);
    DeviceTensor Yd = deviceEinsumNew("rsqp,pqk->rsk", {&slab, &Tdot}, 1.0, dr);
    const Tensor y = downloadTensor(Y, s);  // (nr, v, m)
    const Tensor yd = downloadTensor(Yd, s);
    const int64_t nr = b1 - b0;
    for (int64_t r = 0; r < nr; ++r)
      for (int64_t ss = 0; ss < nv; ++ss) {
        const double delta = ev.flat(b0 + r) + ev.flat(ss);
        const int64_t off = (r * nv + ss) * m;
        const std::span<const double> row(y.data().data() + off, narrowTo<std::size_t>(m));
        const std::span<const double> rowDot(yd.data().data() + off, narrowTo<std::size_t>(m));
        pcCoefficients(row, cs.lambda, delta, g, minDen);
        double norm = 0.0, dotTerm = 0.0;
        for (int64_t k = 0; k < m; ++k) {
          const std::size_t uk = narrowTo<std::size_t>(k);
          norm += g[uk] * row[uk];
          dotTerm += g[uk] * rowDot[uk];
        }
        e -= 0.5 * norm;
        edot -= 0.5 * (2.0 * dotTerm - pcQuadForm(g, g, ct.mK, ct.nS, delta));
      }
  });
  out.result = cs.result;
  out.result.energy = e;
  out.edot = edot;
  checkDenominator(out.result, minDen);
  return out;
}

// Sij: Delta = -(e_c[i] + e_c[j]), factor 1/2 on every ordered (i, j).
Result<PcClassTangent> pcDotSij(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                int64_t batch, const Tensor& ec, const PcRdms& rdm,
                                const PcRdms& sum, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = rdm.dm1->dims[0], nc = std::ssize(ec);
  PcClassTangent out;
  ClassSolve cs;
  ClassTangent ct;
  {
    PcMatrices base = pcMatricesSij(ai, rdm, dr);
    cs = NEVPT2_TRY(solveClass(base.matS(), base.matK(), "Sij", dr, nullptr, /*keepUt=*/true));
    if (cs.result.status != PcStatus::Done) {
      out.result = cs.result;
      return out;
    }
    PcMatrices dot = pcMatricesSij(ai, sum, dr);
    tangentMatrices(dot, base, dr);
    ct = tangentClass(cs, dot.matS(), dot.matK(), dr);
  }
  const int64_t m = std::ssize(cs.lambda);
  const DeviceTensor T = reshapeView(cs.T, {n, n, m});
  const DeviceTensor Tdot = reshapeView(ct.Tdot, {n, n, m});

  double e = 0.0, edot = 0.0, minDen = std::numeric_limits<double>::infinity();
  std::vector<double> g;
  forBatches(src.extent(ExtBlock::Sij), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sij, b0, b1, dr);  // h[q, p, i_batch, j]
    DeviceTensor Y = deviceEinsumNew("qpij,pqk->ijk", {&slab, &T}, 1.0, dr);
    DeviceTensor Yd = deviceEinsumNew("qpij,pqk->ijk", {&slab, &Tdot}, 1.0, dr);
    const Tensor y = downloadTensor(Y, s);  // (ni, c, m)
    const Tensor yd = downloadTensor(Yd, s);
    const int64_t ni = b1 - b0;
    for (int64_t i = 0; i < ni; ++i)
      for (int64_t j = 0; j < nc; ++j) {
        const double delta = -(ec.flat(b0 + i) + ec.flat(j));
        const int64_t off = (i * nc + j) * m;
        const std::span<const double> row(y.data().data() + off, narrowTo<std::size_t>(m));
        const std::span<const double> rowDot(yd.data().data() + off, narrowTo<std::size_t>(m));
        pcCoefficients(row, cs.lambda, delta, g, minDen);
        double norm = 0.0, dotTerm = 0.0;
        for (int64_t k = 0; k < m; ++k) {
          const std::size_t uk = narrowTo<std::size_t>(k);
          norm += g[uk] * row[uk];
          dotTerm += g[uk] * rowDot[uk];
        }
        e -= 0.5 * norm;
        edot -= 0.5 * (2.0 * dotTerm - pcQuadForm(g, g, ct.mK, ct.nS, delta));
      }
  });
  out.result = cs.result;
  out.result.energy = e;
  out.edot = edot;
  checkDenominator(out.result, minDen);
  return out;
}

// Sir: Delta = e_v[r] - e_c[i], factor 1, two slabs plus the one-body column.
Result<PcClassTangent> pcDotSir(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                int64_t batch, const Tensor& ec, const Tensor& ev,
                                const PcRdms& rdm, const PcRdms& sum,
                                const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = rdm.dm1->dims[0], nc = std::ssize(ec);
  const int64_t d = 2 * n * n;
  PcClassTangent out;
  ClassSolve cs;
  ClassTangent ct;
  {
    PcMatrices base = pcMatricesSir(ai, rdm, dr);
    const DeviceTensor baseS1 = base.matS1();
    cs = NEVPT2_TRY(solveClass(base.matS(), base.matK(), "Sir", dr, &baseS1, /*keepUt=*/true));
    if (cs.result.status != PcStatus::Done) {
      out.result = cs.result;
      return out;
    }
    PcMatrices dot = pcMatricesSir(ai, sum, dr);
    tangentMatrices(dot, base, dr);
    const DeviceTensor dotS1 = dot.matS1();
    ct = tangentClass(cs, dot.matS(), dot.matK(), dr, &dotS1);
  }
  const int64_t m = std::ssize(cs.lambda);
  const DeviceTensor T1 = reshapeView(sliceAxis(cs.T, 0, 0, n * n), {n, n, m});
  const DeviceTensor T2 = reshapeView(sliceAxis(cs.T, 0, n * n, d), {n, n, m});
  const DeviceTensor t1 = reshapeView(cs.t1, {m});
  const DeviceTensor T1d = reshapeView(sliceAxis(ct.Tdot, 0, 0, n * n), {n, n, m});
  const DeviceTensor T2d = reshapeView(sliceAxis(ct.Tdot, 0, n * n, d), {n, n, m});
  const DeviceTensor t1d = reshapeView(ct.t1dot, {m});

  double e = 0.0, edot = 0.0, minDen = std::numeric_limits<double>::infinity();
  std::vector<double> g;
  forBatches(src.extent(ExtBlock::Sir1), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab1 = src.slab(ExtBlock::Sir1, b0, b1, dr);  // h[r_batch, p, i, q]
    DeviceTensor slab2 = src.slab(ExtBlock::Sir2, b0, b1, dr);  // h[r_batch, p, q, i]
    DeviceTensor h1 = sliceAxis(ai.h1e_v_Sir, 0, b0, b1);       // (nr, c)
    DeviceTensor Y = deviceEinsumNew("rpiq,pqk->rik", {&slab1, &T1}, 1.0, dr);
    deviceEinsumAccum("rpqi,pqk->rik", {&slab2, &T2}, Y, 1.0, s);
    deviceEinsumAccum("ri,k->rik", {&h1, &t1}, Y, 1.0, s);
    DeviceTensor Yd = deviceEinsumNew("rpiq,pqk->rik", {&slab1, &T1d}, 1.0, dr);
    deviceEinsumAccum("rpqi,pqk->rik", {&slab2, &T2d}, Yd, 1.0, s);
    deviceEinsumAccum("ri,k->rik", {&h1, &t1d}, Yd, 1.0, s);
    const Tensor y = downloadTensor(Y, s);  // (nr, c, m)
    const Tensor yd = downloadTensor(Yd, s);
    const int64_t nr = b1 - b0;
    for (int64_t r = 0; r < nr; ++r)
      for (int64_t i = 0; i < nc; ++i) {
        const double delta = ev.flat(b0 + r) - ec.flat(i);
        const int64_t off = (r * nc + i) * m;
        const std::span<const double> row(y.data().data() + off, narrowTo<std::size_t>(m));
        const std::span<const double> rowDot(yd.data().data() + off, narrowTo<std::size_t>(m));
        pcCoefficients(row, cs.lambda, delta, g, minDen);
        double norm = 0.0, dotTerm = 0.0;
        for (int64_t k = 0; k < m; ++k) {
          const std::size_t uk = narrowTo<std::size_t>(k);
          norm += g[uk] * row[uk];
          dotTerm += g[uk] * rowDot[uk];
        }
        e -= norm;
        edot -= 2.0 * dotTerm - pcQuadForm(g, g, ct.mK, ct.nS, delta);
      }
  });
  out.result = cs.result;
  out.result.energy = e;
  out.edot = edot;
  checkDenominator(out.result, minDen);
  return out;
}

}  // namespace nevpt2
