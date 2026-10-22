// nevpt2.gradient:curvature -- the SECOND-ORDER INPUT GENERATORS of the
// orbital block's <Ehat', I''> term (#154; docs/hessian-theory.md, Sec. 5.1
// and Sec. 5.4): the dotted input sets (:rotation's ScDotted) whose pairing
// with the energy's first derivative in the integrals is K1 or K2.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :curvature;` and from no importer of nevpt2.gradient.
// (Used by gradient.cpp.)
//
// THE TERM. Along U = exp(s u + t v), every kappa-dependent input z the energy
// reads -- every stored block, the active h1e / h2e, the orbital energies --
// has the mixed second derivative
//
//   z''[u, v] = z'[W] + z^x[u, v],     W = (u v + v u) / 2,
//
// z'[W] the FIRST-order rule (every slot promoted once) at the symmetric W,
// and z^x the cross-slot term (two different slots, one by u and one by v,
// both ways round). With the pseudodensities fixed, the energy is linear in
// its inputs, so <Ehat', z''> splits into K1 = <Ehat', z'[W]> and
// K2 = <Ehat', z^x> -- each one walk of the existing pairing at a dotted set.
//
// K1 AT A MATRIX: dottedAtMatrix. The promotion machinery holds for any
// matrix (:rotation's makeGeneralRotation, `anyMatrix`) EXCEPT where it drops
// a traced pair's promotions because they cancel against an antisymmetric
// kappa. Those are restored here, and only here (`restoredTraces`):
//
//   h1e_v_Sr = h^eff[v,a] - T,  T_rt = sum_b (rb|bt): the two promotions of
//     the traced b, sum_{x, b} M_xb [ (rx|bt) + (rb|xt) ], are run by
//     :general over x OUTSIDE the active space only; the x-in-active rows are
//     the pair that cancels for M^T = -M and are added back;
//   v2tr = sum_a h2e_v2_Sir[r,a,a,i]: :terms' slabDots takes its tangent as
//     the trace of the GENERAL half of Sir2's alone; the intra-active half's
//     trace is added back, so v2tr's dot is the trace of Sir2's whole dot.
//
// (The third place Sec. 5.1 names, the Sir class's own "v2tr does not move",
// is that same trace.) The core dressing's tied pairs and the orbital
// energies' density labels need nothing: :general runs their general index
// over the whole MO range already.
//
// K2 FROM THE SIDECAR: composedCross, when ONE of the two directions is
// active-active (Sec. 5.4's restricted row). With `o` that direction and `i`
// the other, applying o's first-order rule to i's dotted inputs gives every
// pair of slots once, the same slot twice included -- the latter being the
// first-order rule at the matrix product i o:
//
//   A_o A_i z = z^x[o, i] + z'[i o].
//
// An active-active o acts only on ACTIVE slots and needs i's dotted inputs
// only at active values of them -- which is what i's dotted inputs are, on
// the visible axes of every array (kappaActionDevice). On the HIDDEN slots it
// acts as follows: a traced active pair is invariant (the projector commutes
// with o), the core dressing's pairs and the free core / virtual labels are
// untouched, and the orbital energies' density labels -- contracted with a
// gamma that does NOT commute with o -- carry gamma's own rotation
// [o, gamma] = o gamma - gamma o into i's orbital-energy response, which is
// generalEpsResponseTangent's W'[i] read at gamma' = [o, gamma] (#184's C5
// generator). So
//
//   z^x = A_o^vis(z'[i]) - z'[i o],   eps^x = <[o, gamma], W'[i]> - eps'[i o].
//
// K2 IN THREE-INDEX FORM: dfDottedCross, on a density-fitted case, for any
// pair. Every input is a sum over L of products of two factors B(U)_L =
// U^T B_L U (the integrals, h^eff's Coulomb and exchange dressing with their
// traced core pair, h1e_v_Sr's traced active pair, the orbital energies'
// density lines) plus U^T h U. In the multilinear view U = 1 + s u + t v, the
// st coefficient of a product F G is
//
//   F_st G + F G_st + F_s G_t + F_t G_s,   B_s = u^T B + B u, B_t likewise,
//                                          B_st = u^T B v + v^T B u,
//
// and that coefficient IS z^x (each slot holds one factor of U, so it picks
// one slot by u and a different one by v). Nothing of the twelve two-general-
// index arrays is formed. The same evaluator at first order (dfDottedFirst:
// F_M G + F G_M, B_M = M^T B + B M, h_M = M^T h + h M) is an independent
// construction of K1 that never sees the sidecar's promotion table or its
// restorations -- the fitted case's cross-check of both.
//
// K2 FROM THE FULL TENSOR: fullDottedCross, on a conventional case with its
// `.erigold` uploaded (#180), for any pair. Nothing above needs the two
// factors to be the same tensor, only that (pq|rs) = sum_L F_L[p,q] G_L[r,s]
// EXACTLY, with each factor rotating as U^T X_L U -- the first distribution's
// slots in F, the second's in G, which is how the evaluator places them. The
// exact four-index factorisation is F = the identity over L = (p', q'),
// F_L[p, q] = delta_{p p'} delta_{q q'}, and G_L[r, s] = (p'q'|rs), the
// tensor itself read as [n_mo^2, n_mo, n_mo]: sum_L F(U)_L[p,q] G(U)_L[r,s] =
// sum_{p'q'r's'} U_{p'p} U_{q'q} U_{r'r} U_{s's} (p'q'|r's'), the rotated
// integral. So the same jets, F_s G_t + ..., at L = n_mo^2: the whole
// n_mo^4 cost Sec. 5.4 says no smaller input avoids, and nothing else. Its
// first-order twin (fullDottedFirst) is an independent K1, as the fitted one.
//
// Every generator returns WHOLE blocks shaped like the integrals', from res's
// pool, on res.stream(): the walk that pairs them runs one slab per block.
module;

// NEVPT2_TRY: a macro, which no import carries. A matrix of the wrong shape is
// the caller's input and propagates unchanged.
#include "error_handling/error_macros.h"

module nevpt2.gradient:curvature;

import std;
import nevpt2.gradient;
// The promotion table, the sidecar's arrays by name, addTensorInto and the
// orbital energies' C5 generator (generalEpsResponseTangent).
import :general;
// ScDotted / dottedField, makeGeneralRotation (with `anyMatrix`),
// kappaActionDevice and kappaColumnBlock.
import :rotation;
// slabDots: the first-order rule on every block at once.
import :terms;

namespace nevpt2 {

// Every NevptIntegralsDevice field a dotted set carries, by the RotBlock that
// names it: the twelve stored external blocks, then h1e and h2e. (SrsiT and
// V2tr are not fields -- the first is h2e_v_Srsi again, the second
// ScDotted::v2tr.)
struct ScDottedField {
  RotBlock kind;
  DeviceTensor NevptIntegralsDevice::* member;
};

constexpr std::array<ScDottedField, 14> kDottedFields{{
    {RotBlock::Sr, &NevptIntegralsDevice::h2e_v_Sr},
    {RotBlock::Si, &NevptIntegralsDevice::h2e_v_Si},
    {RotBlock::Sijrs, &NevptIntegralsDevice::cvcv},
    {RotBlock::Sijr, &NevptIntegralsDevice::h2e_v_Sijr},
    {RotBlock::Srsi, &NevptIntegralsDevice::h2e_v_Srsi},
    {RotBlock::Srs, &NevptIntegralsDevice::h2e_v_Srs},
    {RotBlock::Sij, &NevptIntegralsDevice::h2e_v_Sij},
    {RotBlock::Sir1, &NevptIntegralsDevice::h2e_v1_Sir},
    {RotBlock::Sir2, &NevptIntegralsDevice::h2e_v2_Sir},
    {RotBlock::H1eSr, &NevptIntegralsDevice::h1e_v_Sr},
    {RotBlock::H1eSi, &NevptIntegralsDevice::h1e_v_Si},
    {RotBlock::H1eSir, &NevptIntegralsDevice::h1e_v_Sir},
    {RotBlock::H1e, &NevptIntegralsDevice::h1e},
    {RotBlock::H2e, &NevptIntegralsDevice::h2e},
}};

// out += coef * x, same shape: a no-permutation transpose, as addTensorInto.
void addScaledInto(const DeviceTensor& x, const DeviceTensor& out, double coef,
                   wwrStream_t stream) {
  if (x.size() == 0) return;
  std::vector<int> axes(static_cast<std::size_t>(x.rank()));
  for (std::size_t a = 0; a < axes.size(); ++a) axes[a] = static_cast<int>(a);
  deviceTransposeAccum(x, axes, out, coef, stream);
}

// v2tr's dot from Sir2's: the trace over its two active axes.
DeviceTensor traceV2tr(const DeviceTensor& sir2Dot, const DeviceResources& res) {
  return deviceEinsumNew("raai->ri", {&sir2Dot}, 1.0, res);
}

// A dotted set of zeros, every field shaped like `ints`' own.
ScDotted zeroDotted(const NevptIntegralsDevice& ints, const DeviceResources& res) {
  ScDotted out;
  for (const ScDottedField& f : kDottedFields)
    out.d.*f.member = DeviceTensor::zeros((ints.*f.member).dims, res);
  out.d.e_core = DeviceTensor::zeros(ints.e_core.dims, res);
  out.d.e_virt = DeviceTensor::zeros(ints.e_virt.dims, res);
  out.v2tr = DeviceTensor::zeros({ints.h2e_v2_Sir.dims[0], ints.h2e_v2_Sir.dims[3]}, res);
  return out;
}

// The host matrix product a b, both square.
Tensor matmulHost(const Tensor& a, const Tensor& b) {
  const int64_t n = a.dim(0);
  Tensor out({n, n});
  for (int64_t p = 0; p < n; ++p)
    for (int64_t k = 0; k < n; ++k) {
      const double x = a(p, k);
      if (x == 0.0) continue;
      for (int64_t q = 0; q < n; ++q) out(p, q) += x * b(k, q);
    }
  return out;
}

// A full (n_mo, n_mo) matrix's active-active block.
Tensor activeBlockOf(const Tensor& m, int64_t actLo, int64_t nact) {
  Tensor out({nact, nact});
  for (int64_t p = 0; p < nact; ++p)
    for (int64_t q = 0; q < nact; ++q) out(p, q) = m(actLo + p, actLo + q);
  return out;
}

// The traced pairs' promotions that the first-order machinery leaves out
// because they cancel against an antisymmetric kappa, at the matrix `m`
// (header), as a dotted set that is zero everywhere else: the h1e_v_Sr field
// carries the x-in-active rows of the traced b's two promotions (with T's
// minus sign), and v2tr the trace of Sir2's intra-active half. Zero to
// round-off for an antisymmetric `m`, and -- for an active-active m -- the
// whole of the traced pairs' dependence on it (Sec. 5.1's U P U^T).
ScDotted restoredTraces(const NevptIntegralsDevice& ints, const GeneralIntegralsDevice& gen,
                        const Tensor& m, const DeviceResources& res) {
  const wwrStream_t s = res.stream();
  const int64_t nmo = gen.nmo();
  const int64_t actLo = gen.ncore, actHi = gen.ncore + gen.nact;
  ScDotted out = zeroDotted(ints, res);
  // m[:, act] with ONLY its active rows: the complement of kAct's.
  Tensor rows({nmo, gen.nact});
  for (int64_t x = actLo; x < actHi; ++x)
    for (int64_t t = 0; t < gen.nact; ++t) rows(x, t) = m(x, actLo + t);
  const DeviceTensor kRows = uploadTensor(rows, res);
  // (r x | b t) = g_xv_aa[x, r, b, t] and (r b | x t) = g_xa_va[x, t, r, b], as
  // :general's accumGeneralOneElectron step (3) reads them.
  deviceEinsumAccum("zimj,zm->ij", {&gen.g_xv_aa, &kRows}, out.d.h1e_v_Sr, -1.0, s);
  deviceEinsumAccum("zjim,zm->ij", {&gen.g_xa_va, &kRows}, out.d.h1e_v_Sr, -1.0, s);
  const DeviceTensor mAA = uploadTensor(activeBlockOf(m, actLo, gen.nact), res);
  const DeviceTensor intra = kappaActionDevice(ints.h2e_v2_Sir, {1, 2}, mAA, res);
  out.v2tr = traceV2tr(intra, res);
  return out;
}

// K1's input, or any matrix's first-order dotted set: the promotion rule at
// `m` -- an (n_mo, n_mo) matrix, symmetric, antisymmetric or neither -- on
// every stored block, h1e, h2e and the orbital energies, with the traced
// pairs restored (header). The sidecar's shapes are the caller's to have
// checked; a matrix of the wrong shape is the caller's input.
Result<ScDotted> dottedAtMatrix(const NevptIntegralsDevice& ints,
                                const GeneralIntegralsDevice& gen, const DeviceTensor& dm1,
                                const Tensor& m, const DeviceResources& res) {
  ScRotation rot =
      NEVPT2_TRY(makeGeneralRotation({ScRotationRequest{m, /*tangent=*/false}}, ints.h1e,
                                     ints.h2e, gen, dm1, res, /*anyMatrix=*/true));
  ScRotationDir& dir = rot.dirs[0];
  std::vector<const DeviceTensor*> blocks;
  std::vector<RotBlock> kinds;
  for (const ScDottedField& f : kDottedFields) {
    if (f.kind == RotBlock::H1e || f.kind == RotBlock::H2e) continue;  // the direction's own
    blocks.push_back(&(ints.*f.member));
    kinds.push_back(f.kind);
  }
  ScSlabDots dots = slabDots(blocks, kinds, dir, &gen, res);
  ScDotted out;
  for (std::size_t i = 0; i < kinds.size(); ++i) {
    DeviceTensor& field = out.d.*kDottedFields[i].member;
    field = dots.ptr[i] != nullptr ? std::move(dots.owned[i])
                                   : DeviceTensor::zeros(blocks[i]->dims, res);
  }
  out.d.h1e = std::move(dir.h1eDot);
  out.d.h2e = std::move(dir.h2eDot);
  out.d.e_core = uploadTensor(dir.epsCoreDot, res);
  out.d.e_virt = uploadTensor(dir.epsVirtDot, res);
  // The restorations: h1e_v_Sr's traced b, and v2tr as the trace of Sir2's
  // WHOLE dot (which the h2e_v2_Sir field already is).
  const ScDotted traced = restoredTraces(ints, gen, m, res);
  addTensorInto(traced.d.h1e_v_Sr, out.d.h1e_v_Sr, res.stream());
  out.v2tr = traceV2tr(out.d.h2e_v2_Sir, res);
  return out;
}

// K2's input from the sidecar (header): `outer` the ACTIVE-ACTIVE direction,
// `inner` the other, both (n_mo, n_mo) and antisymmetric. That `outer` is
// active-active is the caller's to have checked.
Result<ScDotted> composedCross(const NevptIntegralsDevice& ints,
                               const GeneralIntegralsDevice& gen, const DeviceTensor& dm1,
                               const Tensor& outer, const Tensor& inner,
                               const DeviceResources& res) {
  const wwrStream_t s = res.stream();
  const int64_t nmo = gen.nmo();
  const int64_t actLo = gen.ncore, actHi = gen.ncore + gen.nact;
  const ScDotted di = NEVPT2_TRY(dottedAtMatrix(ints, gen, dm1, inner, res));
  const ScDotted dio = NEVPT2_TRY(dottedAtMatrix(ints, gen, dm1, matmulHost(inner, outer), res));
  const Tensor oHost = activeBlockOf(outer, actLo, gen.nact);
  const DeviceTensor oAA = uploadTensor(oHost, res);

  ScDotted out;
  for (const ScDottedField& f : kDottedFields) {
    const std::vector<int> axes = activeAxesOf(f.kind);
    const DeviceTensor& src = di.d.*f.member;
    DeviceTensor& field = out.d.*f.member;
    field = axes.empty() ? DeviceTensor::zeros(src.dims, res)
                         : kappaActionDevice(src, axes, oAA, res);
    addScaledInto(dio.d.*f.member, field, -1.0, s);
  }
  // The orbital energies: no visible active slot, and the density labels'
  // rotation [o, gamma] read through inner's C5 generator.
  const DeviceTensor kC = uploadTensor(kappaColumnBlock(inner, 0, gen.ncore, actLo, actHi, false), res);
  const DeviceTensor kA = uploadTensor(kappaColumnBlock(inner, actLo, actHi, actLo, actHi, false), res);
  const DeviceTensor kV = uploadTensor(kappaColumnBlock(inner, actHi, nmo, actLo, actHi, false), res);
  const OrbitalEnergyResponse wDot = generalEpsResponseTangent(gen, kC, kA, kV, res);
  const Tensor gamma = downloadTensor(dm1, s);
  Tensor gammaRot({gen.nact, gen.nact});
  for (int64_t a = 0; a < gen.nact; ++a)
    for (int64_t b = 0; b < gen.nact; ++b) {
      double v = 0.0;
      for (int64_t c = 0; c < gen.nact; ++c) v += oHost(a, c) * gamma(c, b) - gamma(a, c) * oHost(c, b);
      gammaRot(a, b) = v;
    }
  const OrbitalEnergyDot e = orbitalEnergyDot(wDot, gammaRot);
  out.d.e_core = uploadTensor(e.core, res);
  out.d.e_virt = uploadTensor(e.virt, res);
  addScaledInto(dio.d.e_core, out.d.e_core, -1.0, s);
  addScaledInto(dio.d.e_virt, out.d.e_virt, -1.0, s);
  out.v2tr = traceV2tr(out.d.h2e_v2_Sir, res);
  return out;
}

// --- the three-index route (density-fitted only) ------------------------------

// One factor pair of a product jet: sum over these of Phi(first, second).
using DfFactorPairs = std::vector<std::pair<const DeviceTensor*, const DeviceTensor*>>;

// B[L, P, Q] for two spaces, as a view.
DeviceTensor dfSlice(const GeneralIntegralsDevice& gen, const DeviceTensor& f, RotSpace p,
                     RotSpace q) {
  const auto [pLo, pHi] = spanOf(gen, p);
  const auto [qLo, qHi] = spanOf(gen, q);
  return sliceAxis(sliceAxis(f, 1, pLo, pHi), 2, qLo, qHi);
}

// A (n_mo, n_mo) matrix's (P, Q) block, as a view.
DeviceTensor matSlice(const GeneralIntegralsDevice& gen, const DeviceTensor& m, RotSpace p,
                      RotSpace q) {
  const auto [pLo, pHi] = spanOf(gen, p);
  const auto [qLo, qHi] = spanOf(gen, q);
  return sliceAxis(sliceAxis(m, 0, pLo, pHi), 1, qLo, qHi);
}

// sum_x m[x, p] f[L, x, q]: `m` on the first MO index of every B_L.
DeviceTensor dfLeft(const DeviceTensor& f, const DeviceTensor& m, const DeviceResources& res) {
  return deviceEinsumNew("Lxq,xp->Lpq", {&f, &m}, 1.0, res);
}

// The product jets of every input, from the factor pairs and the one-electron
// term `hTerm` (n_mo, n_mo): the evaluator both routes share (header). `gamma`
// is the active 1-RDM the orbital energies' density lines contract.
ScDotted dfDottedFromPairs(const NevptIntegralsDevice& ints, const GeneralIntegralsDevice& gen,
                           const DeviceTensor& gamma, const DfFactorPairs& pairs,
                           const DeviceTensor& hTerm, const DeviceResources& res) {
  using S = RotSpace;
  const wwrStream_t s = res.stream();
  ScDotted out;
  for (const ScDottedField& f : kDottedFields) {
    const RotBlockInfo info = rotBlockInfo(f.kind);
    DeviceTensor field = DeviceTensor::zeros((ints.*f.member).dims, res);
    if (info.rank == 4) {
      // X[axes] = (p r | q s) = sum_L B[L,p,r] B[L,q,s], the labels placed on
      // the axes rotBlockInfo assigns them.
      static constexpr std::string_view kRole = "pqrs";
      std::string outSubs(4, ' ');
      for (std::size_t k = 0; k < 4; ++k)
        outSubs[static_cast<std::size_t>(info.roles[k].axis)] = kRole[k];
      const std::string subs = "Lpr,Lqs->" + outSubs;
      for (const auto& [f1, f2] : pairs) {
        const DeviceTensor a = dfSlice(gen, *f1, info.roles[0].space, info.roles[2].space);
        const DeviceTensor b = dfSlice(gen, *f2, info.roles[1].space, info.roles[3].space);
        deviceEinsumAccum(subs, {&a, &b}, field, 1.0, s);
      }
    } else {
      // h^eff[p, q] = h + 2 (pq|jj) - (pj|jq), and h1e_v_Sr's - (rb|bt).
      const S p = info.roles[0].space, q = info.roles[1].space;
      const DeviceTensor hs = matSlice(gen, hTerm, p, q);
      deviceEinsumAccum("pq->pq", {&hs}, field, 1.0, s);
      for (const auto& [f1, f2] : pairs) {
        if (gen.ncore > 0) {
          const DeviceTensor fpq = dfSlice(gen, *f1, p, q);
          const DeviceTensor fcc = dfSlice(gen, *f2, S::Core, S::Core);
          const DeviceTensor tc = deviceEinsumNew("Ljj->L", {&fcc}, 1.0, res);
          deviceEinsumAccum("Lpq,L->pq", {&fpq, &tc}, field, 2.0, s);
          const DeviceTensor fpc = dfSlice(gen, *f1, p, S::Core);
          const DeviceTensor fcq = dfSlice(gen, *f2, S::Core, q);
          deviceEinsumAccum("Lpj,Ljq->pq", {&fpc, &fcq}, field, -1.0, s);
        }
        if (info.trace) {
          const DeviceTensor fpa = dfSlice(gen, *f1, p, S::Act);
          const DeviceTensor faq = dfSlice(gen, *f2, S::Act, q);
          deviceEinsumAccum("Lpb,Lbq->pq", {&fpa, &faq}, field, -1.0, s);
        }
      }
    }
    out.d.*f.member = std::move(field);
  }
  // e_p = h^eff_pp + sum_ab gamma_ab [ (pp|ab) - (1/2) (pa|bp) ], p core or virtual.
  for (const S sp : {S::Core, S::Virt}) {
    const DeviceTensor& dims = sp == S::Core ? ints.e_core : ints.e_virt;
    DeviceTensor e = DeviceTensor::zeros(dims.dims, res);
    if (dims.size() > 0) {
      const DeviceTensor hs = matSlice(gen, hTerm, sp, sp);
      deviceEinsumAccum("pp->p", {&hs}, e, 1.0, s);
      for (const auto& [f1, f2] : pairs) {
        const DeviceTensor fss = dfSlice(gen, *f1, sp, sp);
        if (gen.ncore > 0) {
          const DeviceTensor fcc = dfSlice(gen, *f2, S::Core, S::Core);
          const DeviceTensor tc = deviceEinsumNew("Ljj->L", {&fcc}, 1.0, res);
          deviceEinsumAccum("Lpp,L->p", {&fss, &tc}, e, 2.0, s);
          const DeviceTensor fsc = dfSlice(gen, *f1, sp, S::Core);
          const DeviceTensor fcs = dfSlice(gen, *f2, S::Core, sp);
          deviceEinsumAccum("Lpj,Ljp->p", {&fsc, &fcs}, e, -1.0, s);
        }
        const DeviceTensor faa = dfSlice(gen, *f2, S::Act, S::Act);
        const DeviceTensor g2 = deviceEinsumNew("Lab,ab->L", {&faa, &gamma}, 1.0, res);
        deviceEinsumAccum("Lpp,L->p", {&fss, &g2}, e, 1.0, s);
        const DeviceTensor fsa = dfSlice(gen, *f1, sp, S::Act);
        const DeviceTensor fas = dfSlice(gen, *f2, S::Act, sp);
        deviceEinsumAccum("Lpa,ab,Lbp->p", {&fsa, &gamma, &fas}, e, -0.5, s);
      }
    }
    (sp == S::Core ? out.d.e_core : out.d.e_virt) = std::move(e);
  }
  out.v2tr = traceV2tr(out.d.h2e_v2_Sir, res);
  return out;
}

// The first-order jet at any (n_mo, n_mo) `m` over an exact factorisation
// (pq|rs) = sum_L f1[L,p,q] f2[L,r,s] (header); the one-electron term is
// gen.h's.
ScDotted factorDottedFirst(const NevptIntegralsDevice& ints, const GeneralIntegralsDevice& gen,
                           const DeviceTensor& dm1, const Tensor& m, const DeviceTensor& f1,
                           const DeviceTensor& f2, const DeviceResources& res) {
  const wwrStream_t s = res.stream();
  const DeviceTensor dm = uploadTensor(m, res);
  // X_M[L,p,q] = sum_x m_xp X[L,x,q] + m_xq X[L,p,x], each factor.
  const auto jet = [&](const DeviceTensor& f) {
    DeviceTensor out = dfLeft(f, dm, res);
    deviceEinsumAccum("Lpx,xq->Lpq", {&f, &dm}, out, 1.0, s);
    return out;
  };
  const DeviceTensor f1m = jet(f1);
  // One tensor for both factors (the fitted B): its jet once.
  std::optional<DeviceTensor> f2own;
  if (&f2 != &f1) f2own.emplace(jet(f2));
  const DeviceTensor& f2m = f2own ? *f2own : f1m;
  // h_M = m^T h + h m.
  DeviceTensor hm = deviceEinsumNew("xp,xq->pq", {&dm, &gen.h}, 1.0, res);
  deviceEinsumAccum("px,xq->pq", {&gen.h, &dm}, hm, 1.0, s);
  const DfFactorPairs pairs{{&f1m, &f2}, {&f1, &f2m}};
  return dfDottedFromPairs(ints, gen, dm1, pairs, hm, res);
}

// The cross-slot jet along (u, v), both (n_mo, n_mo), over the same exact
// factorisation.
ScDotted factorDottedCross(const NevptIntegralsDevice& ints, const GeneralIntegralsDevice& gen,
                           const DeviceTensor& dm1, const Tensor& u, const Tensor& v,
                           const DeviceTensor& f1, const DeviceTensor& f2,
                           const DeviceResources& res) {
  const wwrStream_t s = res.stream();
  const DeviceTensor du = uploadTensor(u, res);
  const DeviceTensor dv = uploadTensor(v, res);
  // X_s = u^T X + X u, X_t likewise, and X_st = u^T X v + v^T X u: u on the
  // first index then v on the second, and the other way round.
  struct Jets {
    DeviceTensor s, t, st;
  };
  const auto jets = [&](const DeviceTensor& f) {
    DeviceTensor fu = dfLeft(f, du, res);
    deviceEinsumAccum("Lpx,xq->Lpq", {&f, &du}, fu, 1.0, s);
    DeviceTensor fv = dfLeft(f, dv, res);
    deviceEinsumAccum("Lpx,xq->Lpq", {&f, &dv}, fv, 1.0, s);
    const DeviceTensor leftU = dfLeft(f, du, res);
    const DeviceTensor leftV = dfLeft(f, dv, res);
    DeviceTensor fuv = deviceEinsumNew("Lpy,yq->Lpq", {&leftU, &dv}, 1.0, res);
    deviceEinsumAccum("Lpy,yq->Lpq", {&leftV, &du}, fuv, 1.0, s);
    return Jets{std::move(fu), std::move(fv), std::move(fuv)};
  };
  const Jets j1 = jets(f1);
  // One tensor for both factors (the fitted B): its jets once.
  std::optional<Jets> j2own;
  if (&f2 != &f1) j2own.emplace(jets(f2));
  const Jets& j2 = j2own ? *j2own : j1;
  // h_st = u^T h v + v^T h u.
  DeviceTensor huv = deviceEinsumNew("xp,xy,yq->pq", {&du, &gen.h, &dv}, 1.0, res);
  deviceEinsumAccum("xp,xy,yq->pq", {&dv, &gen.h, &du}, huv, 1.0, s);
  const DfFactorPairs pairs{{&j1.st, &f2}, {&f1, &j2.st}, {&j1.s, &j2.t}, {&j1.t, &j2.s}};
  return dfDottedFromPairs(ints, gen, dm1, pairs, huv, res);
}

// K1 in three-index form: the first-order jet at any (n_mo, n_mo) `m`.
ScDotted dfDottedFirst(const NevptIntegralsDevice& ints, const GeneralIntegralsDevice& gen,
                       const DeviceTensor& dm1, const Tensor& m, const DeviceResources& res) {
  return factorDottedFirst(ints, gen, dm1, m, gen.b, gen.b, res);
}

// K2 in three-index form: the cross-slot jet along (u, v), both (n_mo, n_mo).
ScDotted dfDottedCross(const NevptIntegralsDevice& ints, const GeneralIntegralsDevice& gen,
                       const DeviceTensor& dm1, const Tensor& u, const Tensor& v,
                       const DeviceResources& res) {
  return factorDottedCross(ints, gen, dm1, u, v, gen.b, gen.b, res);
}

// The full tensor's exact factorisation (header): the identity over
// L = (p', q') as the first factor, the tensor read as [n_mo^2, n_mo, n_mo]
// as the second. The identity is owned and returned; the second factor is a
// view of gen.eri.
struct FullFactors {
  DeviceTensor eye;     // owns the (n_mo^2, n_mo^2) identity
  DeviceTensor first;   // its view as [n_mo^2, n_mo, n_mo]
  DeviceTensor second;  // gen.eri's view as [n_mo^2, n_mo, n_mo]
};

FullFactors fullFactors(const GeneralIntegralsDevice& gen, const DeviceResources& res) {
  const int64_t n = gen.nmo();
  FullFactors f;
  f.eye = deviceEye(n * n, res);
  f.first = reshapeView(f.eye, {n * n, n, n});
  f.second = reshapeView(gen.eri, {n * n, n, n});
  return f;
}

// K1 from the full tensor: the first-order jet at any (n_mo, n_mo) `m`.
ScDotted fullDottedFirst(const NevptIntegralsDevice& ints, const GeneralIntegralsDevice& gen,
                         const DeviceTensor& dm1, const Tensor& m, const DeviceResources& res) {
  const FullFactors f = fullFactors(gen, res);
  return factorDottedFirst(ints, gen, dm1, m, f.first, f.second, res);
}

// K2 from the full tensor: the cross-slot jet along (u, v), any pair.
ScDotted fullDottedCross(const NevptIntegralsDevice& ints, const GeneralIntegralsDevice& gen,
                         const DeviceTensor& dm1, const Tensor& u, const Tensor& v,
                         const DeviceResources& res) {
  const FullFactors f = fullFactors(gen, res);
  return factorDottedCross(ints, gen, dm1, u, v, f.first, f.second, res);
}

}  // namespace nevpt2
