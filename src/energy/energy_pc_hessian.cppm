// nevpt2.energy:pc_hessian -- the PC-NEVPT2 class energies' SECOND
// directional derivative with respect to the CI vector, Eddot(u, v) (#152):
// each class's solve, the tangents of its S, K and S_1 along u, along v and
// at the second-order RDMs (u, v), their projections on S's dropped space,
// and the slab walk that turns them into Eddot beside E, Edot(u) and Edot(v).
// An internal partition: nothing here is exported, so it is reachable from
// the units that `import :pc_hessian;` and from no importer of nevpt2.energy.
// (Used by energy_pc.cpp, which defines pcHessianDevice over these.)
//
// WHAT IS DIFFERENTIATED: THE SOLVE, TWICE (docs/hessian-theory.md, Sec. 6).
// E_t = -nu b_t^T M b_t with M = Q (Q^T A Q)^-1 Q^T -- S's cut restricted
// inverse, Q the kept eigenvectors of S, A_t = K + Delta_t S, b_t = S x_t +
// S_1 h1_t -- and only S, K and S_1 carry the CI vector. So the second
// derivative needs
//
//   b_uv, A_uv   the same S / K / S_1 builders at the second-order RDMs
//                d^2(dm)/dc^2 [u, v] -- the tangent build at (u, v), the
//                RDMs' exact second derivative (#151) -- through the same
//                one-subtraction trick as the tangent (`tangentMatrices`:
//                every matrix is affine in the RDMs, so X(D + D'') - X(D) IS
//                X'' exactly, and the RDM-free constants cancel);
//   b_u, A_u     the tangent's own, along each direction;
//   M            the factorisation the energy stage already computed, applied
//                to the tangent rows;
//   P S_u, P K_u the tangents' blocks on S's DROPPED space (P = Vn^T Vn, Vn
//                the dropped eigenvectors the solve already found). The
//                dropped space is null for THIS state's metric, not for every
//                CI vector, and it turns with the state (docs/pc-nevpt2.md,
//                "What the cut drops, and how it moves"), so dM carries terms
//                in P Sdot and S^+ = U U^T: the first derivative gains
//                -2 rho^T S_u P A x (zero at the eigenstate) and the second
//                three rotation terms (Sec. 6.2; nevpt2.energy_finish's
//                pcQuadSecond has them in this eigenbasis).
//
// NO EIGENSOLVE IS RERUN AND NO EIGENVECTOR DERIVATIVE APPEARS.
//
// WHAT IS REFUSED. A class the gap or denominator check refuses, as in the
// energy and the tangent. And, per direction pair, a class whose dropped
// eigenvalues leave zero at the order the Hessian probes: the first-order
// block P S_a P, or Sec. 6.2's second-order L_ab = P (S_ab - S_a S^+ S_b -
// S_b S^+ S_a) P, above PC_LIFT_TOL relative to s_max for some (a, b) in
// {u, v}. There the rank changes, and the energy's second derivative is the
// derivative of the PC_TAU threshold, not of a smooth function. Both come
// back as PcStatus::Refused with the reason, never as a zero.
//
// THE ORBITAL ENERGIES' TWO MODES (#214). Held fixed (no `eps`; Deltadot =
// Delta'' = 0): the functional in which Euler's identity Eddot(c, u) ==
// Edot(u) holds on the five homogeneous classes (Sec. 4.3). FULL (`eps`):
// each tuple's Delta'[u], Delta'[v], Delta''[u, v] are its own Delta formula
// read at e'[u], e'[v], e''[u, v], and enter pcQuadSecond as its
// PcDeltaTangents -- A_u = K_u + Delta_u S + Delta S_u and A_uv's three Delta
// terms; P S = 0 keeps every rotation block as it is. Euler does not hold
// there. And the second derivative is Sec. 6.2's at the EIGENSTATE (P A = 0); the
// first derivatives are exact at any state, which is what a finite
// difference of Edot at displaced states needs.
//
// IT IS THE SAME CLASSES. Each class's matrices come from :pc_tangent's
// pcMatrices* -- the builders pcTangentsDevice and the adjoint use -- and its
// slab einsums are :pc_classes' / :pc_tangent's subscripts, run against more
// (T, t1) pairs.
module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.energy:pc_hessian;

import std;
import nevpt2.energy;
import :shared;  // energy_Sijrs: no RDM, so every CI derivative is exactly zero
import :pc_solve;
import :pc_tangent;

namespace nevpt2 {

// The RDM bundles every class is built from: the RDMs, and the RDMs plus
// each tangent set. The three second-order sums are null when only the
// first derivatives are wanted. Non-owning; pcHessianDevice owns the sums.
struct PcHessianBundles {
  PcRdms rdm, sumU, sumV;
  const PcRdms* sumUV = nullptr;
  const PcRdms* sumUU = nullptr;
  const PcRdms* sumVV = nullptr;
  // The orbital energies' derivatives along (u, v), or null (held fixed).
  const OrbitalEnergyHessianDots* eps = nullptr;
};

// One slab's Delta derivatives, per tuple in the class's own order (#214):
// the class's Delta formula `fn(e_core, e_virt)` read at e'[u], e'[v] and
// e''[u, v]. All three empty (zeros, to hessianSingle) with the orbital
// energies held fixed.
struct PcDeltaDots {
  std::vector<double> u, v, uv;
};

template <class DeltaFn>
PcDeltaDots pcDeltaDots(const OrbitalEnergyHessianDots* eps, DeltaFn&& fn) {
  if (eps == nullptr) return {};
  return {fn(eps->u.core, eps->u.virt), fn(eps->v.core, eps->v.virt),
          fn(eps->uv.core, eps->uv.virt)};
}

// One tuple's three, for the pair classes (whose Delta is formed per tuple):
// `fn(e_core, e_virt)` the tuple's Delta formula, a double.
template <class DeltaFn>
std::array<double, 3> pcDeltaDotsAt(const OrbitalEnergyHessianDots* eps, DeltaFn&& fn) {
  if (eps == nullptr) return {0.0, 0.0, 0.0};
  return {fn(eps->u.core, eps->u.virt), fn(eps->v.core, eps->v.virt),
          fn(eps->uv.core, eps->uv.virt)};
}

// One tangent set's projections on S's dropped space (nd = the dropped
// count; everything empty when nd = 0).
struct RotationTangent {
  std::vector<double> G;    // (nd, m) Vn sym(Sdot) U
  std::vector<double> H;    // (nd, m) Vn sym(Kdot) U
  std::vector<double> pSp;  // (nd, nd) Vn sym(Sdot) Vn^T
  DeviceTensor N;           // (d, nd) sym(Sdot) Vn^T: the slab operand for r = Vn bdot
  DeviceTensor n1;          // (n1, nd) S_1dot^T Vn^T, when the class has an S_1
};

std::vector<double> hostOf(const DeviceTensor& t, const DeviceResources& dr) {
  const Tensor h = downloadTensor(t, dr.stream());
  return {h.data().begin(), h.data().end()};
}

// `full` false: pSp alone (what a second-order set needs for the lift).
RotationTangent rotationTangent(const ClassSolve& cs, const DeviceTensor& Sdot,
                                const DeviceTensor& Kdot, const DeviceTensor* S1dot, bool full,
                                const DeviceResources& dr) {
  RotationTangent rt;
  if (cs.Vn.rank() != 2) return rt;  // nothing dropped
  const DeviceTensor Sds = symmetrised(Sdot, dr);
  DeviceTensor N = gemm(Sds, false, cs.Vn, true, dr);  // pq,kq->pk
  rt.pSp = hostOf(gemm(cs.Vn, false, N, false, dr), dr);
  if (!full) return rt;
  {
    const DeviceTensor SdU = gemm(Sds, false, cs.Ut, true, dr);
    rt.G = hostOf(gemm(cs.Vn, false, SdU, false, dr), dr);
  }
  {
    const DeviceTensor Kds = symmetrised(Kdot, dr);
    const DeviceTensor KdU = gemm(Kds, false, cs.Ut, true, dr);
    rt.H = hostOf(gemm(cs.Vn, false, KdU, false, dr), dr);
  }
  if (S1dot != nullptr) rt.n1 = deviceEinsumNew("ia,ki->ak", {S1dot, &cs.Vn}, 1.0, dr);
  rt.N = std::move(N);
  return rt;
}

// One class's solve, its tangents and their dropped-space projections.
struct ClassHessianParts {
  ClassSolve cs;
  ClassTangent tu, tv, tuv;   // tuv only with a second-order set
  RotationTangent ru, rv;     // along u and v
  int64_t nd = 0;
  bool second = false;
  PcHessianBlocks blk;
  PcRotationBlocks rot;
  double lift = 0.0;          // relative to s_max (second derivative only)
};

double frobenius(const std::vector<double>& x) {
  double s = 0.0;
  for (const double v : x) s += v * v;
  return std::sqrt(s);
}

// Build the class from `b.rdm` and solve it (keeping U and the dropped
// eigenvectors); then, one bundle at a time so only one extra set of d x d
// matrices is alive, build it again from each sum, subtract
// (`tangentMatrices`) and project (`tangentClass`, `rotationTangent`). With a
// second-order set, the lift test: a class whose dropped modes rise along the
// pair comes back Refused (cs.result), like a gap refusal.
template <class Build>
Result<ClassHessianParts> hessianParts(const Build& build, const PcHessianBundles& b,
                                       const char* name, const DeviceResources& dr) {
  ClassHessianParts p;
  p.second = b.sumUV != nullptr;
  check(!p.second || (b.sumUU != nullptr && b.sumVV != nullptr),
        "hessianParts: the second derivative's lift test needs the (u, u) and (v, v) sets");
  const PcMatrices base = build(b.rdm);
  const DeviceTensor baseS1 = base.hasS1 ? base.matS1() : DeviceTensor();
  p.cs = NEVPT2_TRY(solveClass(base.matS(), base.matK(), name, dr,
                               base.hasS1 ? &baseS1 : nullptr, /*keepUt=*/true,
                               /*keepDropped=*/true));
  if (p.cs.result.status != PcStatus::Done) return p;
  const int64_t m = std::ssize(p.cs.lambda);
  p.nd = p.cs.result.spectrum.dropped;

  struct Dot {
    ClassTangent t;
    RotationTangent r;
  };
  const auto dotAt = [&](const PcRdms& sum, bool full) {
    PcMatrices dot = build(sum);
    tangentMatrices(dot, base, dr);
    const DeviceTensor dotS1 = dot.hasS1 ? dot.matS1() : DeviceTensor();
    const DeviceTensor* s1 = dot.hasS1 ? &dotS1 : nullptr;
    Dot out;
    out.t = tangentClass(p.cs, dot.matS(), dot.matK(), dr, s1);
    out.r = rotationTangent(p.cs, dot.matS(), dot.matK(), s1, full, dr);
    return out;
  };
  Dot du = dotAt(b.sumU, true), dv = dotAt(b.sumV, true);
  p.tu = std::move(du.t);
  p.tv = std::move(dv.t);
  p.ru = std::move(du.r);
  p.rv = std::move(dv.r);
  p.blk = {p.tu.mK, p.tu.nS, p.tv.mK, p.tv.nS, {}, {}};

  std::vector<double> H0, C;
  if (p.nd > 0) {
    const DeviceTensor Ks = symmetrised(base.matK(), dr);
    const DeviceTensor KU = gemm(Ks, false, p.cs.Ut, true, dr);
    H0 = hostOf(gemm(p.cs.Vn, false, KU, false, dr), dr);
    C = hostOf(gemm(p.cs.Ut, false, p.cs.Ut, true, dr), dr);  // U^T U
  }
  if (p.second) {
    Dot duv = dotAt(*b.sumUV, false);
    p.tuv = std::move(duv.t);
    p.blk.mKuv = p.tuv.mK;
    p.blk.nSuv = p.tuv.nS;
    if (p.nd > 0) {
      // The lift test. |.|_F bounds the largest |eigenvalue| the measured
      // figures behind PC_LIFT_TOL are, from above and by at most sqrt(nd),
      // which the tolerance's margin absorbs.
      const RotationTangent ruu = dotAt(*b.sumUU, false).r;
      const RotationTangent rvv = dotAt(*b.sumVV, false).r;
      const double lifts[] = {frobenius(p.ru.pSp),
                              frobenius(p.rv.pSp),
                              pcLiftNorm(p.nd, m, duv.r.pSp, p.ru.G, p.rv.G),
                              pcLiftNorm(p.nd, m, ruu.pSp, p.ru.G, p.ru.G),
                              pcLiftNorm(p.nd, m, rvv.pSp, p.rv.G, p.rv.G)};
      for (const double l : lifts) p.lift = std::max(p.lift, l / p.cs.sMax);
      if (!(p.lift <= PC_LIFT_TOL)) {
        p.cs.result.status = PcStatus::Refused;
        p.cs.result.why = pcLiftRefusalWhy(p.lift);
        return p;
      }
    }
  }
  p.rot = pcRotationBlocks(m, p.nd, C, p.ru.G, p.ru.H, p.rv.G, p.rv.H, H0, p.second);
  return p;
}

// The class's result from its sums: the energy and the three derivatives,
// then the denominator check -- which can still refuse the class.
PcClassHessian pcHessianResult(const ClassHessianParts& p, const PcHessianSums& acc,
                               double minDen) {
  PcClassHessian out;
  out.result = p.cs.result;
  out.result.energy = acc.energy;
  out.edotU = acc.edotU;
  out.edotV = acc.edotV;
  out.eddot = acc.eddot;
  out.lift = p.lift;
  checkDenominator(out.result, minDen);
  return out;
}

// A refused class: its refusal, every derivative 0 and meaningless (the
// status says so).
PcClassHessian pcHessianRefused(const ClassHessianParts& p) {
  PcClassHessian out;
  out.result = p.cs.result;
  out.lift = p.lift;
  return out;
}

// The (T, t1) pairs a class's slab einsum runs against: the energy's, the
// two tangents', the second-order one's, and N_v's for r_v = Vn b_v.
struct SlabOperands {
  struct Pair {
    const DeviceTensor* T;
    const DeviceTensor* t1;
    int64_t k;  // T's column count: m, or nd for N_v
  };
  std::vector<Pair> pairs;  // y, yu, yv[, yuv, rv]
};

SlabOperands slabOperands(const ClassHessianParts& p) {
  const int64_t m = std::ssize(p.cs.lambda);
  SlabOperands ops;
  ops.pairs = {{&p.cs.T, &p.cs.t1, m}, {&p.tu.Tdot, &p.tu.t1dot, m}, {&p.tv.Tdot, &p.tv.t1dot, m}};
  if (p.second) {
    ops.pairs.push_back({&p.tuv.Tdot, &p.tuv.t1dot, m});
    ops.pairs.push_back({&p.rv.N, &p.rv.n1, p.nd});
  }
  return ops;
}

// What every walk downloads: one host tensor per operand pair (rv's is
// empty when nothing is dropped).
using SlabRows = std::vector<Tensor>;

PcHessianSlab slabOf(const SlabRows& rows) {
  PcHessianSlab s{&rows[0], &rows[1], &rows[2], nullptr, nullptr};
  if (rows.size() == 5) {
    s.yuv = &rows[3];
    s.rv = &rows[4];
  }
  return s;
}

// `rows(T, t1, k)` for each pair; a k = 0 pair (rv with nothing dropped) is
// an empty host tensor, never an einsum.
template <class Rows>
SlabRows walkRows(const SlabOperands& ops, const Rows& rows) {
  SlabRows out;
  for (const SlabOperands::Pair& pr : ops.pairs)
    out.push_back(pr.k == 0 ? Tensor::fromFlat({0}, {}) : rows(*pr.T, *pr.t1, pr.k));
  return out;
}

// --- the per-class second derivatives -------------------------------------------
//
// Each: hessianParts, then the slab walk -- the energy's einsum against T and
// the same einsum against each other operand -- and the per-tuple finish.

// Sr: Delta = e_virt[r], factor 1.
Result<PcClassHessian> pcHessSr(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                int64_t batch, const Tensor& ev, const PcHessianBundles& b,
                                const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = b.rdm.dm2->dims[0];
  const ClassHessianParts p = NEVPT2_TRY(hessianParts(
      [&](const PcRdms& r) { return pcMatricesSr(ai, r, dr); }, b, "Sr", dr));
  if (p.cs.result.status != PcStatus::Done) return pcHessianRefused(p);
  const SlabOperands ops = slabOperands(p);

  PcHessianSums acc;
  double minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Sr), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sr, b0, b1, dr);  // h[r_batch, p, q, r]
    DeviceTensor h1 = sliceAxis(ai.h1e_v_Sr, 0, b0, b1);     // (nr, a)
    const SlabRows rows =
        walkRows(ops, [&](const DeviceTensor& T, const DeviceTensor& t1, int64_t k) {
          const DeviceTensor T4 = reshapeView(T, {n, n, n, k});
          DeviceTensor Y = deviceEinsumNew("ipqr,pqrk->ik", {&slab, &T4}, 1.0, dr);
          deviceEinsumAccum("ia,ak->ik", {&h1, &t1}, Y, 1.0, s);
          return downloadTensor(Y, s);  // (nr, k)
        });
    const PcDeltaDots dd =
        pcDeltaDots(b.eps, [&](const Tensor&, const Tensor& v) { return pcSlice(v, b0, b1); });
    hessianSingle(slabOf(rows), p.cs.lambda, pcSlice(ev, b0, b1), p.blk, p.rot, 1.0, acc,
                  minDen, dd.v, dd.u, dd.uv);
  });
  return pcHessianResult(p, acc, minDen);
}

// Si: Delta = -e_core[i], factor 1.
Result<PcClassHessian> pcHessSi(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                int64_t batch, const Tensor& ec, const PcHessianBundles& b,
                                const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = b.rdm.dm2->dims[0];
  const ClassHessianParts p = NEVPT2_TRY(hessianParts(
      [&](const PcRdms& r) { return pcMatricesSi(ai, r, dr); }, b, "Si", dr));
  if (p.cs.result.status != PcStatus::Done) return pcHessianRefused(p);
  const SlabOperands ops = slabOperands(p);

  PcHessianSums acc;
  double minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Si), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Si, b0, b1, dr);  // h[q, p, i_batch, r]
    DeviceTensor h1 = sliceAxis(ai.h1e_v_Si, 1, b0, b1);     // (a, ni)
    const SlabRows rows =
        walkRows(ops, [&](const DeviceTensor& T, const DeviceTensor& t1, int64_t k) {
          const DeviceTensor T4 = reshapeView(T, {n, n, n, k});
          DeviceTensor Y = deviceEinsumNew("qpir,pqrk->ik", {&slab, &T4}, 1.0, dr);
          deviceEinsumAccum("ai,ak->ik", {&h1, &t1}, Y, 1.0, s);
          return downloadTensor(Y, s);  // (ni, k)
        });
    const PcDeltaDots dd = pcDeltaDots(b.eps, [&](const Tensor& c, const Tensor&) {
      return pcSlice(c, b0, b1, /*negate=*/true);
    });
    hessianSingle(slabOf(rows), p.cs.lambda, pcSlice(ec, b0, b1, /*negate=*/true), p.blk, p.rot,
                  1.0, acc, minDen, dd.v, dd.u, dd.uv);
  });
  return pcHessianResult(p, acc, minDen);
}

// Tuple rows at flat row index `t` (m, and nd, long) of a two-coupling walk.
PcHessianRow pairRowAt(const SlabRows& rows, int64_t t, int64_t m, int64_t nd) {
  return pcHessianRowAt(slabOf(rows), t, m, nd);
}

// Sijr: Delta = e_v[r] - e_c[j] - e_c[i], the direct row (r, j, i) and the
// exchange row (r, i, j) of the same tensor, over ORDERED (j, i).
Result<PcClassHessian> pcHessSijr(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                  int64_t batch, const Tensor& ec, const Tensor& ev,
                                  const PcHessianBundles& b, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const ClassHessianParts p = NEVPT2_TRY(hessianParts(
      [&](const PcRdms& r) { return pcMatricesSijr(ai, r, dr); }, b, "Sijr", dr));
  if (p.cs.result.status != PcStatus::Done) return pcHessianRefused(p);
  const int64_t m = std::ssize(p.cs.lambda);
  const int64_t nc = std::ssize(ec);
  const SlabOperands ops = slabOperands(p);

  PcHessianSums acc;
  double minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Sijr), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sijr, b0, b1, dr);  // (nr, a, c, c)
    const SlabRows rows = walkRows(ops, [&](const DeviceTensor& T, const DeviceTensor&, int64_t) {
      DeviceTensor Y = deviceEinsumNew("rpji,pk->rjik", {&slab, &T}, 1.0, dr);
      return downloadTensor(Y, s);  // (nr, c, c, k)
    });
    const int64_t nr = b1 - b0;
    for (int64_t r = 0; r < nr; ++r)
      for (int64_t j = 0; j < nc; ++j)
        for (int64_t i = 0; i < nc; ++i) {
          const auto deltaOf = [&](const Tensor& c, const Tensor& v) {
            return v.flat(b0 + r) - c.flat(j) - c.flat(i);
          };
          const double delta = deltaOf(ec, ev);
          const std::array<double, 3> dd = pcDeltaDotsAt(b.eps, deltaOf);
          hessianPair(pairRowAt(rows, (r * nc + j) * nc + i, m, p.nd),
                      pairRowAt(rows, (r * nc + i) * nc + j, m, p.nd), p.cs.lambda, delta,
                      p.blk, p.rot, p.second, acc, minDen, dd[1], dd[0], dd[2]);
        }
  });
  return pcHessianResult(p, acc, minDen);
}

// Srsi: Delta = e_v[r] + e_v[s] - e_c[i], the exchange row from the SrsiT
// slab instead of an index swap.
Result<PcClassHessian> pcHessSrsi(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                  int64_t batch, const Tensor& ec, const Tensor& ev,
                                  const PcHessianBundles& b, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const ClassHessianParts p = NEVPT2_TRY(hessianParts(
      [&](const PcRdms& r) { return pcMatricesSrsi(ai, r, dr); }, b, "Srsi", dr));
  if (p.cs.result.status != PcStatus::Done) return pcHessianRefused(p);
  const int64_t m = std::ssize(p.cs.lambda);
  const int64_t nc = std::ssize(ec), nv = std::ssize(ev);
  const SlabOperands ops = slabOperands(p);

  PcHessianSums acc;
  double minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Srsi), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Srsi, b0, b1, dr);    // h[r_batch, s, i, p]
    DeviceTensor slabT = src.slab(ExtBlock::SrsiT, b0, b1, dr);  // h[s, r_batch, i, p]
    const SlabRows rows = walkRows(ops, [&](const DeviceTensor& T, const DeviceTensor&, int64_t) {
      DeviceTensor Y = deviceEinsumNew("rsip,pk->rsik", {&slab, &T}, 1.0, dr);
      return downloadTensor(Y, s);  // (nr, v, c, k)
    });
    const SlabRows rowsT = walkRows(ops, [&](const DeviceTensor& T, const DeviceTensor&, int64_t) {
      DeviceTensor Y = deviceEinsumNew("srip,pk->rsik", {&slabT, &T}, 1.0, dr);
      return downloadTensor(Y, s);
    });
    const int64_t nr = b1 - b0;
    for (int64_t r = 0; r < nr; ++r)
      for (int64_t ss = 0; ss < nv; ++ss)
        for (int64_t i = 0; i < nc; ++i) {
          const auto deltaOf = [&](const Tensor& c, const Tensor& v) {
            return v.flat(b0 + r) + v.flat(ss) - c.flat(i);
          };
          const double delta = deltaOf(ec, ev);
          const std::array<double, 3> dd = pcDeltaDotsAt(b.eps, deltaOf);
          const int64_t t = (r * nv + ss) * nc + i;
          hessianPair(pairRowAt(rows, t, m, p.nd), pairRowAt(rowsT, t, m, p.nd), p.cs.lambda,
                      delta, p.blk, p.rot, p.second, acc, minDen, dd[1], dd[0], dd[2]);
        }
  });
  return pcHessianResult(p, acc, minDen);
}

// Srs: Delta = e_v[r] + e_v[s], factor 1/2 on every ordered (r, s).
Result<PcClassHessian> pcHessSrs(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                 int64_t batch, const Tensor& ev, const PcHessianBundles& b,
                                 const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = b.rdm.dm1->dims[0], nv = std::ssize(ev);
  const ClassHessianParts p = NEVPT2_TRY(hessianParts(
      [&](const PcRdms& r) { return pcMatricesSrs(ai, r, dr); }, b, "Srs", dr));
  if (p.cs.result.status != PcStatus::Done) return pcHessianRefused(p);
  const SlabOperands ops = slabOperands(p);

  PcHessianSums acc;
  double minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Srs), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Srs, b0, b1, dr);  // h[r_batch, s, q, p]
    const SlabRows rows = walkRows(ops, [&](const DeviceTensor& T, const DeviceTensor&, int64_t k) {
      const DeviceTensor T3 = reshapeView(T, {n, n, k});
      DeviceTensor Y = deviceEinsumNew("rsqp,pqk->rsk", {&slab, &T3}, 1.0, dr);
      return downloadTensor(Y, s);  // (nr, v, k): (nr * v) tuples
    });
    const auto deltaOf = [&](const Tensor&, const Tensor& v) {
      std::vector<double> delta;
      for (int64_t r = b0; r < b1; ++r)
        for (int64_t ss = 0; ss < nv; ++ss) delta.push_back(v.flat(r) + v.flat(ss));
      return delta;
    };
    const PcDeltaDots dd = pcDeltaDots(b.eps, deltaOf);
    hessianSingle(slabOf(rows), p.cs.lambda, deltaOf(/*unread=*/ev, ev), p.blk, p.rot, 0.5,
                  acc, minDen, dd.v, dd.u, dd.uv);
  });
  return pcHessianResult(p, acc, minDen);
}

// Sij: Delta = -(e_c[i] + e_c[j]), factor 1/2 on every ordered (i, j).
Result<PcClassHessian> pcHessSij(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                 int64_t batch, const Tensor& ec, const PcHessianBundles& b,
                                 const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = b.rdm.dm1->dims[0], nc = std::ssize(ec);
  const ClassHessianParts p = NEVPT2_TRY(hessianParts(
      [&](const PcRdms& r) { return pcMatricesSij(ai, r, dr); }, b, "Sij", dr));
  if (p.cs.result.status != PcStatus::Done) return pcHessianRefused(p);
  const SlabOperands ops = slabOperands(p);

  PcHessianSums acc;
  double minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Sij), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sij, b0, b1, dr);  // h[q, p, i_batch, j]
    const SlabRows rows = walkRows(ops, [&](const DeviceTensor& T, const DeviceTensor&, int64_t k) {
      const DeviceTensor T3 = reshapeView(T, {n, n, k});
      DeviceTensor Y = deviceEinsumNew("qpij,pqk->ijk", {&slab, &T3}, 1.0, dr);
      return downloadTensor(Y, s);  // (ni, c, k): (ni * c) tuples
    });
    const auto deltaOf = [&](const Tensor& c, const Tensor&) {
      std::vector<double> delta;
      for (int64_t i = b0; i < b1; ++i)
        for (int64_t j = 0; j < nc; ++j) delta.push_back(-(c.flat(i) + c.flat(j)));
      return delta;
    };
    const PcDeltaDots dd = pcDeltaDots(b.eps, deltaOf);
    hessianSingle(slabOf(rows), p.cs.lambda, deltaOf(ec, /*unread=*/ec), p.blk, p.rot, 0.5,
                  acc, minDen, dd.v, dd.u, dd.uv);
  });
  return pcHessianResult(p, acc, minDen);
}

// Sir: Delta = e_v[r] - e_c[i], factor 1, two slabs plus the one-body column.
Result<PcClassHessian> pcHessSir(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                 int64_t batch, const Tensor& ec, const Tensor& ev,
                                 const PcHessianBundles& b, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = b.rdm.dm1->dims[0], nc = std::ssize(ec);
  const int64_t d = 2 * n * n;
  const ClassHessianParts p = NEVPT2_TRY(hessianParts(
      [&](const PcRdms& r) { return pcMatricesSir(ai, r, dr); }, b, "Sir", dr));
  if (p.cs.result.status != PcStatus::Done) return pcHessianRefused(p);
  const SlabOperands ops = slabOperands(p);

  PcHessianSums acc;
  double minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Sir1), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab1 = src.slab(ExtBlock::Sir1, b0, b1, dr);  // h[r_batch, p, i, q]
    DeviceTensor slab2 = src.slab(ExtBlock::Sir2, b0, b1, dr);  // h[r_batch, p, q, i]
    DeviceTensor h1 = sliceAxis(ai.h1e_v_Sir, 0, b0, b1);       // (nr, c)
    const SlabRows rows =
        walkRows(ops, [&](const DeviceTensor& T, const DeviceTensor& t1, int64_t k) {
          const DeviceTensor T1 = reshapeView(sliceAxis(T, 0, 0, n * n), {n, n, k});
          const DeviceTensor T2 = reshapeView(sliceAxis(T, 0, n * n, d), {n, n, k});
          const DeviceTensor t1v = reshapeView(t1, {k});
          DeviceTensor Y = deviceEinsumNew("rpiq,pqk->rik", {&slab1, &T1}, 1.0, dr);
          deviceEinsumAccum("rpqi,pqk->rik", {&slab2, &T2}, Y, 1.0, s);
          deviceEinsumAccum("ri,k->rik", {&h1, &t1v}, Y, 1.0, s);
          return downloadTensor(Y, s);  // (nr, c, k): (nr * c) tuples
        });
    const auto deltaOf = [&](const Tensor& c, const Tensor& v) {
      std::vector<double> delta;
      for (int64_t r = b0; r < b1; ++r)
        for (int64_t i = 0; i < nc; ++i) delta.push_back(v.flat(r) - c.flat(i));
      return delta;
    };
    const PcDeltaDots dd = pcDeltaDots(b.eps, deltaOf);
    hessianSingle(slabOf(rows), p.cs.lambda, deltaOf(ec, ev), p.blk, p.rot, 1.0, acc, minDen,
                  dd.v, dd.u, dd.uv);
  });
  return pcHessianResult(p, acc, minDen);
}

}  // namespace nevpt2
