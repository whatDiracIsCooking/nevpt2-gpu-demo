// nevpt2.gradient:pc_classes -- PC-NEVPT2's REVERSE MODE (#74): the seven
// partially-contracted classes differentiated through the solve the energy
// already ran, for the CI source term y_I = dE_pc/dc_I and the orbital source
// term Y_xy = dE_pc/dkappa_xy. Sijrs is SC's, here as in the energy
// (nevpt2.gradient:sc_classes' gradientSijrs, called as is).
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :pc_classes;` and from no importer of nevpt2.gradient.
// (Used by gradient_pc.cpp.)
//
// WHAT IS DIFFERENTIATED -- the solve, never the eigendecomposition. A PC
// class energy is a sum over external tuples t of quadratic forms in
//
//   y_t = U^T b_t,   b_t = S x_t + S_1 h1_t,   D_t = lambda + Delta_t
//
// (docs/pc-nevpt2.md, "One idea"), with (U, lambda) the solve of
// A_t = K + Delta_t S on S's kept range. Reverse mode needs one identity: for
// any perturbation of S, K, S_1 and the integrals,
//
//   dE = sum_t ybar_t . (U^T db_t)  +  < U mgg U^T, dA_t >
//
// with ybar_t = dE/dy_t and mgg the (m, m) weight of the tuples' quadratic
// forms (nevpt2.energy_finish's pcAdjointSingle / pcAdjointPair compute both
// per tuple, and R_t = dE/dDelta_t beside them). It is the tangent's own
// formula -- Edot_t = -c (2 x^T bdot - x^T Adot x), x = A^+ b, at
// energy_pc_solve.cppm's `tangentClass` -- read the other way: dA^+ =
// -A^+ dA A^+, which needs A^+ differentiable, i.e. S's rank locally
// constant, and the two extra terms of the general pseudo-inverse derivative
// to drop, which they do because b_t lies in the kept range. PC_MIN_GAP is
// what makes the rank constant (docs/pc-nevpt2.md, "The singular-metric
// convention"; docs/gradients.md, Sec. 7), so the adjoint is valid exactly
// where pcEnergiesDevice accepts a class -- and a class the gap or the
// denominator check refuses is REFUSED here too, through the same
// PcClassResult status, never a zero contribution: its `result` is the
// solve's own verdict, and every entry point above this partition reports it.
// No eigenvector or eigenvalue derivative appears anywhere below.
//
// One further fact keeps that true along a direction that does NOT stay in
// the state's symmetry sector (the determinant axes the CI source term is
// read on): S's dropped modes are null modes of THIS state's metric -- they
// turn with the state, they are not null for every CI vector
// (docs/pc-nevpt2.md, "What the cut drops, and how it moves") -- and at the
// eigenstate K annihilates them too (K v = <Phi|H_D - E_0| sum_nu v_nu
// Phi_nu> = 0 there), so P A = 0 and the first-order rotation of S's kept
// range that such a direction causes contributes nothing at first order
// (the -2 rho^T Sdot P A x term of docs/hessian-theory.md, Sec. 6.2,
// vanishes). (It is why the
// PC CI step must be small in a finite difference -- the lifted modes are
// SECOND order -- and not a term of the derivative: docs/testing.md, "The PC
// CI source term's step is a first-order trap".) Off the eigenstate the term
// does not vanish, and a y without it is not the gradient there: the CI walk
// below (PcCiWalk, #155) carries it -- the three-term y a finite difference
// at displaced states needs -- and differentiates the whole walk along a CI
// direction for H_cc u.
//
// FROM THAT IDENTITY, everything the two source terms need, per class:
//
//   dE/dT       = sum_t x_t ybar_t^T       -- the slab einsum's adjoint in its
//   dE/dt_1     = sum_t h1_t ybar_t^T         basis operand (derivativeSubscripts)
//   dE/dslab    = (T ybar)  per tuple      -- in its external operand: Eq. 41's
//                                             pseudodensity, paired with a
//                                             rotation's dotted slab
//   dE/dK       = U sym(mgg) U^T
//   dE/dS       = sym(U (dE/dT)^T) + U sym(mggDelta) U^T
//   dE/dS_1     = U (dE/dt_1)^T
//   dE/de       = R_t scattered through the class's Delta (deltaAdjointInto)
//
// and those three matrices go one chain rule further in through the class's
// own builders -- K is a16 / a22 / a3 / k27 / a7 / a9 / a12+a13, exactly SC's
// blocks, so nevpt2.gradient:rdm_adjoint's per-block sweeps are reused
// unchanged, f3ac / f3ca coefficients included (they feed both the 4-body
// sigmas of y and the digests' transpose of Y, #89); S and S_1 are
// permutations of dm3 / dm2 / dm1 or of the hole-side metrics, whose own
// sweeps are rdm_adjoint's too. Sir's block-patterned S, K and S_1 are
// differentiated term by term with the same adjointTerm.
//
// THE ORBITAL ROTATION. Along kappa at fixed CI only the integrals move -- S
// and S_1 are RDM-only -- so the adjoint pairing is
//
//   <kappa, Y> = sum_slabs <dE/dslab, slabdot> + <dE/dK, Kdot> + sum_t R_t Deltadot_t
//                (+ the digests' term, <hbar, h2e_dot>, added by the caller)
//
// and the forward tangent, computed from the same rows by different code
// (nevpt2.energy_finish's pcRotationTangent*: ydot from the dotted slab
// through T, mK = U^T sym(Kdot) U), is held against it -- the PC counterpart
// of SC's integral-half duality. Kdot is the class's own K builder at the
// direction's dotted integrals with the digests zeroed (pcClassMatrices),
// the same "a Dyall block's tangent is its own builder" fact SC's rotation
// rests on; the slab and one-electron tangents are :terms' slabDots, SC's own.
//
// Every einsum on res.stream(), every tensor from res's pool and freed when
// its slab or class returns; every read-back a downloadTensor.
module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.gradient:pc_classes;

import std;
import nevpt2.gradient;
import :rdm_adjoint;  // ScRdmAdjoint, adjointTerm and the per-block sweeps
import :rotation;     // ScRotation, ScRotationDir
import :general;      // addTensorInto
import :terms;        // slabDots, pairSlabDots, dotTensors
import :sc_classes;   // gradientSijrs: PC's Sijrs IS SC's

namespace nevpt2 {

// One term of a PC class's slab einsum, copied from energy_pc_classes.cppm:
// target[t] += einsum(subs, external operand, basis operand). `ext` indexes
// the slab's external operands (and their RotBlock kinds), `basis` the
// class's basis views (T or t_1 reshaped), `target` 0 for y and 1 for Srsi's
// exchange tensor.
struct PcYTerm {
  std::string subs;
  int ext = 0;
  int basis = 0;
  int target = 0;
};

enum class PcPattern { Single, PairSwap, PairTwo };

// One slab's external operands: the tensors (owning, or views of the
// integrals), and the RotBlock each is, in the order PcYTerm::ext names them.
struct PcSlabOperands {
  std::vector<DeviceTensor> ext;
  std::vector<RotBlock> kinds;
};

// A class, described: everything pcClassAdjoint below needs that is not the
// same for all seven.
struct PcClassWalk {
  int cls = 0;
  ExtBlock block = ExtBlock::Sr;  // the slab walk's extent
  PcPattern pattern = PcPattern::Single;
  double factor = 1.0;            // Single only
  std::vector<PcYTerm> terms;
  // The basis views of (T, t1) -- also applied to dE/dT, dE/dt1's buffers.
  std::function<std::vector<DeviceTensor>(const DeviceTensor& T, const DeviceTensor& t1)> basis;
  std::function<PcSlabOperands(int64_t b0, int64_t b1)> slab;
  // Delta_t over the slab's tuples, as a FUNCTION of (e_core, e_virt).
  std::function<std::vector<double>(int64_t b0, int64_t b1, const Tensor& ec, const Tensor& ev)>
      delta;
  // PairSwap only: the exchange partner of each tuple of a slab of nr.
  std::function<std::vector<int64_t>(int64_t nr)> partner;
  // dE/dS (d, d), dE/dK (d, d), dE/dS_1 (d, n1) -> the RDM coefficients.
  std::function<void(const DeviceTensor& gS, const DeviceTensor& gK, const DeviceTensor& gS1,
                     ScRdmAdjoint& adj)>
      sweep;
};

// The per-class accumulators gradient.cppm's PcClassGradient exports, under
// construction. `result` starts as the solve's; a refusal leaves everything
// else untouched.

// A vector of move-only tensors, built by moving each in (an initializer list
// would copy).
template <class... Ts>
std::vector<DeviceTensor> tensorList(Ts&&... ts) {
  std::vector<DeviceTensor> v;
  v.reserve(sizeof...(Ts));
  (v.push_back(std::forward<Ts>(ts)), ...);
  return v;
}

// `a` sym'd in place: a <- (a + a^T) / 2, (m, m) row-major on the host.
std::vector<double> symHost(const std::vector<double>& a, int64_t m) {
  std::vector<double> s(a.size());
  for (int64_t k = 0; k < m; ++k)
    for (int64_t l = 0; l < m; ++l)
      s[static_cast<std::size_t>(k * m + l)] =
          0.5 * (a[static_cast<std::size_t>(k * m + l)] + a[static_cast<std::size_t>(l * m + k)]);
  return s;
}

// U M U^T for an (m, m) host M and the solve's Ut (m, d), as a fresh (d, d):
// two einsums, never the three-operand one (d^2 m^2 work).
DeviceTensor sandwich(const DeviceTensor& ut, const std::vector<double>& msym, int64_t m,
                      const DeviceResources& res) {
  const DeviceTensor dm = uploadTensor(Tensor::fromFlat({m, m}, msym), res);
  const DeviceTensor tmp = deviceEinsumNew("kl,lq->kq", {&dm, &ut}, 1.0, res);
  return deviceEinsumNew("kp,kq->pq", {&ut, &tmp}, 1.0, res);
}

// U^T sym(X) U for a (d, d) device X, read back as (m, m) row-major: the
// tangent's mK.
std::vector<double> projectSym(const DeviceTensor& x, const DeviceTensor& ut,
                               const DeviceResources& res) {
  const DeviceTensor tmp = deviceEinsumNew("pq,lq->pl", {&x, &ut}, 0.5, res);
  deviceEinsumAccum("qp,lq->pl", {&x, &ut}, tmp, 0.5, res.stream());
  const DeviceTensor mk = deviceEinsumNew("kp,pl->kl", {&ut, &tmp}, 1.0, res);
  const Tensor h = downloadTensor(mk, res.stream());
  return std::vector<double>(h.data().begin(), h.data().end());
}

// sum_t over a slab's targets: each target's einsum terms, the first
// allocating. A term whose external operand is null (a dotted operand nothing
// moves) is skipped; a target nothing contributes to comes back zero-filled,
// shaped like `shapeOf[target]`.
std::array<DeviceTensor, 2> evalTargets(const PcClassWalk& w,
                                        const std::vector<const DeviceTensor*>& ext,
                                        const std::vector<DeviceTensor>& basis,
                                        const std::array<std::vector<int64_t>, 2>& shapeOf,
                                        const DeviceResources& res) {
  std::array<DeviceTensor, 2> out;
  std::array<bool, 2> started{false, false};
  const int ntarget = w.pattern == PcPattern::PairTwo ? 2 : 1;
  for (const PcYTerm& t : w.terms) {
    const DeviceTensor* e = ext[static_cast<std::size_t>(t.ext)];
    if (e == nullptr) continue;
    const std::size_t tg = static_cast<std::size_t>(t.target);
    const DeviceTensor& b = basis[static_cast<std::size_t>(t.basis)];
    if (!started[tg]) {
      out[tg] = deviceEinsumNew(t.subs, {e, &b}, 1.0, res);
      started[tg] = true;
    } else {
      deviceEinsumAccum(t.subs, {e, &b}, out[tg], 1.0, res.stream());
    }
  }
  for (int tg = 0; tg < ntarget; ++tg)
    if (!started[static_cast<std::size_t>(tg)])
      out[static_cast<std::size_t>(tg)] =
          DeviceTensor::zeros(shapeOf[static_cast<std::size_t>(tg)], res);
  return out;
}

// THE CI WALK (#155): what a class's reverse walk adds for PC's H_cc u
// (docs/hessian-theory.md, Sec. 4.2 and 6.2), at the golden's fixed orbital
// energies. Null in pcClassAdjoint: the plain adjoint (#74), whose y carries
// the first derivative's two terms -- exact at the eigenstate, where it is
// evaluated.
//
// `threeTerm`: y carries the third, -2 rho^T S_v P A x, so it is the gradient
// of the energy at ANY CI vector -- what a finite difference of y at
// displaced states must difference (Sec. 6.2: it vanishes only where P K = 0).
// Its adjoint is one (nd, m) block, gbar3, taken onto dE/dS through the
// solve's own dropped eigenvectors.
//
// `sumU` non-null (implies threeTerm): the RDMs plus their tangent along u.
// The walk then also runs the class's slab einsum against T_u (the class's
// matrices' tangent along u, through pcClassDirection: the same tangentClass
// / rotationTangent pcHessianDevice uses), takes nevpt2.energy_finish's
// pcAdjointSecond* -- the reverse, in v, of pcQuadSecond's bilinear part --
// down to dE_dot/dS, dE_dot/dK, dE_dot/dS_1 through T_v = S_v U,
// N_v = S_v Vn^T, alpha_v and the dropped blocks G_v, H_v, and sweeps them
// into `adjDot` with the class's own sweep: gdot[u]. `sumUU` (the RDMs plus
// the second-order set at (u, u)) is the lift test's, as pcHessianDevice's
// along the pair (u, u): a class whose dropped modes leave zero at first
// order (|P S_u P|) or second (L_uu) above PC_LIFT_TOL relative to s_max is
// REFUSED, never zeroed.
struct PcCiWalk {
  bool threeTerm = false;
  const PcRdms* sumU = nullptr;
  const PcRdms* sumUU = nullptr;
  ScRdmAdjoint* adjDot = nullptr;
};

// ||x||_F of a host block.
double frobeniusHost(const std::vector<double>& x) {
  double s = 0.0;
  for (const double v : x) s += v * v;
  return std::sqrt(s);
}

// target += sym(Vn^T X U^T), X (nd, m) on the host, Vn (nd, d), Ut (m, d):
// a dropped-kept block's adjoint taken onto a (d, d) matrix gradient.
void addSymDroppedKept(const DeviceTensor& vn, const std::vector<double>& x, int64_t nd,
                       int64_t m, const DeviceTensor& ut, const DeviceTensor& target,
                       const DeviceResources& res) {
  const DeviceTensor dx = uploadTensor(Tensor::fromFlat({nd, m}, x), res);
  const DeviceTensor xu = deviceEinsumNew("kl,lq->kq", {&dx, &ut}, 1.0, res);   // (nd, d)
  const DeviceTensor full = deviceEinsumNew("kp,kq->pq", {&vn, &xu}, 0.5, res); // (d, d)
  addTensorInto(full, target, res.stream());
  deviceTransposeAccum(full, {1, 0}, target, 1.0, res.stream());
}

// One class, in reverse: the solve, the slab walk (energy, dE/dT, dE/dt1,
// mgg, R, and per direction the external pairing and the forward tangent),
// then dE/dK paired with every direction's Kdot and -- with `adj` -- dE/dS,
// dE/dK and dE/dS_1 swept onto the RDM coefficients.
//
// `zeroF3` is the RDM bundle with zero digests: what a rotation's Kdot is
// built from (the f3 addends of a16 / a22 carry no integral). A class the
// solve or the denominator check refuses comes back with that status and
// nothing accumulated that anyone may read. `ci` non-null: PcCiWalk's
// additions (it needs `adj` and no rotation).
Status pcClassAdjoint(const PcClassWalk& w, const ActiveIntegralsDevice& ai, IntegralSource& src,
                      int64_t batch, const Tensor& hostEC, const Tensor& hostEV,
                      const PcRdms& rdm, const PcRdms& zeroF3, const ScRotation& rot,
                      PcClassGradient& acc, ScRdmAdjoint* adj, const PcCiWalk* ci,
                      const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const char* name = CLASSES[static_cast<std::size_t>(w.cls)].c_str();
  const bool direction = ci != nullptr && ci->sumU != nullptr;
  const bool threeTerm = ci != nullptr && (ci->threeTerm || direction);
  check(ci == nullptr || (adj != nullptr && rot.empty() &&
                          (!direction || (ci->sumUU != nullptr && ci->adjDot != nullptr))),
        "pcClassAdjoint: the CI walk needs the RDM coefficients, no rotation, and with a "
        "direction its (u, u) set and its own accumulators");
  PcClassSolve cs;
  int64_t d = 0, n1 = 0;
  bool hasS1 = false;
  PcAdjointBlocks blk;
  PcClassDirection du;
  {
    const PcMatrices mx = pcClassMatrices(w.cls, ai.h1e, ai.h2e, rdm, dr);
    d = mx.d;
    n1 = mx.n1;
    hasS1 = mx.hasS1;
    cs = NEVPT2_TRY(pcSolveClass(mx, name, dr, /*keepDropped=*/threeTerm));
    if (threeTerm && cs.result.status == PcStatus::Done) {
      blk.m = std::ssize(cs.lambda);
      blk.nd = cs.Vn.rank() == 2 ? cs.Vn.dims[0] : 0;
      PcDroppedBase base = pcDroppedBase(cs, mx, dr);
      blk.C = std::move(base.C);
      blk.H0 = std::move(base.H0);
      if (direction) {
        PcMatrices dot = pcClassMatrices(w.cls, ai.h1e, ai.h2e, *ci->sumU, dr);
        du = pcClassDirection(cs, dot, mx, /*full=*/true, dr);
        blk.direction = true;
        blk.mKu = du.mK;
        blk.nSu = du.nS;
        blk.Gu = du.G;
        blk.Hu = du.H;
        if (blk.nd > 0) {
          // The lift test along (u, u), pcHessianDevice's: the first-order
          // P S_u P and the second-order L_uu, relative to s_max.
          PcMatrices dotUU = pcClassMatrices(w.cls, ai.h1e, ai.h2e, *ci->sumUU, dr);
          const PcClassDirection duu = pcClassDirection(cs, dotUU, mx, /*full=*/false, dr);
          acc.lift = std::max(frobeniusHost(du.pSp),
                              pcLiftNorm(blk.nd, blk.m, duu.pSp, du.G, du.G)) /
                     cs.sMax;
          if (!(acc.lift <= PC_LIFT_TOL)) {
            cs.result.status = PcStatus::Refused;
            cs.result.why = pcLiftRefusalWhy(acc.lift);
          }
        }
      }
    }
  }
  acc.result = cs.result;
  if (cs.result.status != PcStatus::Done) return {};
  const int64_t m = std::ssize(cs.lambda);
  const int64_t nd = blk.nd;
  PcAdjointSecond second = pcAdjointSecondZeros(blk);
  // The direction's adjoint accumulators: dE_dot/dT_v, dE_dot/dt1_v through
  // the same basis views, and their dropped-space twins dE_dot/dN_v,
  // dE_dot/dn1_v (N_v = sym(S_v) Vn^T, (d, nd)).
  DeviceTensor gTd, gt1d, gN, gn1;
  std::vector<DeviceTensor> basisU, gBasisD, gBasisN;
  if (direction) {
    gTd = DeviceTensor::zeros({d, m}, dr);
    if (hasS1) gt1d = DeviceTensor::zeros({n1, m}, dr);
    basisU = w.basis(du.Tdot, du.t1dot);
    gBasisD = w.basis(gTd, gt1d);
    if (nd > 0) {
      gN = DeviceTensor::zeros({d, nd}, dr);
      if (hasS1) gn1 = DeviceTensor::zeros({n1, nd}, dr);
      gBasisN = w.basis(gN, gn1);
    }
  }

  // dE/dT and dE/dt_1, accumulated through the basis views of zeroed buffers.
  const DeviceTensor gT = DeviceTensor::zeros({d, m}, dr);
  const DeviceTensor gt1 = hasS1 ? DeviceTensor::zeros({n1, m}, dr) : DeviceTensor();
  const std::vector<DeviceTensor> basis = w.basis(cs.T, cs.t1);
  const std::vector<DeviceTensor> gBasis = w.basis(gT, gt1);
  std::vector<double> mgg(static_cast<std::size_t>(m * m), 0.0);
  std::vector<double> mggDelta(static_cast<std::size_t>(m * m), 0.0);

  // Each direction's Kdot (kept for the pairing after the walk) and, for a
  // tangent direction, its mK (needed inside the walk).
  std::vector<DeviceTensor> kDot(rot.size());
  std::vector<std::vector<double>> mK(rot.size());
  for (std::size_t k = 0; k < rot.size(); ++k) {
    kDot[k] = pcClassMatrices(w.cls, rot.dirs[k].h1eDot, rot.dirs[k].h2eDot, zeroF3, dr).K;
    if (rot.dirs[k].tangent)
      mK[k] = projectSym(reshapeView(kDot[k], {d, d}), cs.Ut, dr);
  }

  double e = 0.0, minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(w.block), batch, [&](int64_t b0, int64_t b1) {
    const PcSlabOperands ops = w.slab(b0, b1);
    std::vector<const DeviceTensor*> extPtr;
    for (const DeviceTensor& t : ops.ext) extPtr.push_back(&t);
    const std::array<DeviceTensor, 2> yy = evalTargets(w, extPtr, basis, {}, dr);
    // No sync before the downloads: each is behind the einsums that wrote its
    // source on the one stream, and downloadTensor synchronizes it.
    const Tensor y = downloadTensor(yy[0], s);
    const bool two = w.pattern == PcPattern::PairTwo;
    const Tensor x = two ? downloadTensor(yy[1], s) : Tensor();
    const std::vector<double> delta = w.delta(b0, b1, hostEC, hostEV);
    const std::vector<int64_t> partner =
        w.pattern == PcPattern::PairSwap ? w.partner(b1 - b0) : std::vector<int64_t>{};
    const PcSlabAdjoint a =
        w.pattern == PcPattern::Single
            ? pcAdjointSingle(y, cs.lambda, delta, w.factor, mgg, mggDelta, minDen)
            : pcAdjointPair(y, two ? &x : nullptr, partner, cs.lambda, delta, mgg, mggDelta,
                            minDen);
    e += a.energy;
    // dE/de: R_t through the class's own Delta construction (#87's probe).
    if (!acc.dEdCore.empty() || !acc.dEdVirt.empty())
      deltaAdjointInto([&](const Tensor& ec, const Tensor& ev) { return w.delta(b0, b1, ec, ev); },
                       a.r, acc.dEdCore, acc.dEdVirt);

    std::array<DeviceTensor, 2> bar;
    bar[0] = uploadTensor(Tensor::fromFlat(y.dims(), a.ybar), dr);
    if (two) bar[1] = uploadTensor(Tensor::fromFlat(x.dims(), a.xbar), dr);
    for (const PcYTerm& t : w.terms) {
      const DeviceTensor& ext = ops.ext[static_cast<std::size_t>(t.ext)];
      deviceEinsumAccum(derivativeSubscripts(t.subs, 1),
                        {&ext, &bar[static_cast<std::size_t>(t.target)]},
                        gBasis[static_cast<std::size_t>(t.basis)], 1.0, s);
    }
    if (threeTerm) {
      // The third term, and along u the rows' tangent and their bars
      // (PcCiWalk). The direction's rows are the same einsum against T_u.
      Tensor yu, xu;
      if (direction) {
        const std::array<DeviceTensor, 2> yd =
            evalTargets(w, extPtr, basisU, {yy[0].dims, two ? yy[1].dims : yy[0].dims}, dr);
        yu = downloadTensor(yd[0], s);
        if (two) xu = downloadTensor(yd[1], s);
      }
      const PcSlabAdjointSecond sec =
          w.pattern == PcPattern::Single
              ? pcAdjointSecondSingle(y, direction ? &yu : nullptr, cs.lambda, delta, w.factor, blk,
                                      second)
              : pcAdjointSecondPair(y, two ? &x : nullptr, direction ? &yu : nullptr,
                                    direction && two ? &xu : nullptr, partner, cs.lambda, delta,
                                    blk, second);
      if (direction) {
        std::array<DeviceTensor, 2> barD, barN;
        barD[0] = uploadTensor(Tensor::fromFlat(y.dims(), sec.ybarDot), dr);
        if (two) barD[1] = uploadTensor(Tensor::fromFlat(x.dims(), sec.xbarDot), dr);
        // The r_v rows are (tuples, nd): y's leading extents with nd last.
        const auto rowsOf = [nd](const Tensor& t) {
          std::vector<int64_t> dims = t.dims();
          dims.back() = nd;
          return dims;
        };
        if (nd > 0) {
          barN[0] = uploadTensor(Tensor::fromFlat(rowsOf(y), sec.rbarDot), dr);
          if (two) barN[1] = uploadTensor(Tensor::fromFlat(rowsOf(x), sec.rxbarDot), dr);
        }
        for (const PcYTerm& t : w.terms) {
          const DeviceTensor& ext = ops.ext[static_cast<std::size_t>(t.ext)];
          const std::string subs = derivativeSubscripts(t.subs, 1);
          const std::size_t tg = static_cast<std::size_t>(t.target);
          const std::size_t bi = static_cast<std::size_t>(t.basis);
          deviceEinsumAccum(subs, {&ext, &barD[tg]}, gBasisD[bi], 1.0, s);
          if (nd > 0) deviceEinsumAccum(subs, {&ext, &barN[tg]}, gBasisN[bi], 1.0, s);
        }
      }
    }
    if (rot.empty()) return;

    // dE/dslab: this slab's pseudodensity in each external operand.
    std::vector<DeviceTensor> dExt;
    for (const DeviceTensor& t : ops.ext) dExt.push_back(DeviceTensor::zeros(t.dims, dr));
    for (const PcYTerm& t : w.terms)
      deviceEinsumAccum(derivativeSubscripts(t.subs, 0),
                        {&basis[static_cast<std::size_t>(t.basis)],
                         &bar[static_cast<std::size_t>(t.target)]},
                        dExt[static_cast<std::size_t>(t.ext)], 1.0, s);
    for (std::size_t k = 0; k < rot.size(); ++k) {
      const ScRotationDir& dir = rot.dirs[k];
      const ScSlabDots dots = slabDots(extPtr, ops.kinds, dir, rot.gen, dr);
      acc.rotExt[k] += pairSlabDots(dots, dExt, dr);
      if (dir.general)
        acc.rotEps[k] += dotHost(a.r, w.delta(b0, b1, dir.epsCoreDot, dir.epsVirtDot));
      if (!dir.tangent) continue;
      const std::array<DeviceTensor, 2> yd =
          evalTargets(w, dots.ptr, basis, {yy[0].dims, two ? yy[1].dims : yy[0].dims}, dr);
      const Tensor ydot = downloadTensor(yd[0], s);
      if (w.pattern == PcPattern::Single) {
        acc.rotTangent[k] +=
            pcRotationTangentSingle(y, ydot, cs.lambda, delta, w.factor, mK[k]);
      } else {
        const Tensor xdot = two ? downloadTensor(yd[1], s) : Tensor();
        acc.rotTangent[k] += pcRotationTangentPair(y, two ? &x : nullptr, ydot,
                                                   two ? &xdot : nullptr, partner, cs.lambda,
                                                   delta, mK[k]);
      }
    }
  });
  acc.result.energy = e;
  checkDenominator(acc.result, minDen);
  if (acc.result.status != PcStatus::Done) return {};

  // dE/dK, paired with each direction's Kdot.
  const DeviceTensor gK = sandwich(cs.Ut, symHost(mgg, m), m, dr);
  for (std::size_t k = 0; k < rot.size(); ++k)
    acc.rotAct[k] += dotTensors(gK, reshapeView(kDot[k], {d, d}), dr);

  if (adj == nullptr) return {};
  // dE/dS = sym(U (dE/dT)^T) + U sym(mggDelta) U^T, and dE/dS_1 = U (dE/dt_1)^T.
  const DeviceTensor uGt = deviceEinsumNew("kp,qk->pq", {&cs.Ut, &gT}, 0.5, dr);
  const DeviceTensor gS = sandwich(cs.Ut, symHost(mggDelta, m), m, dr);
  addTensorInto(uGt, gS, s);
  deviceTransposeAccum(uGt, {1, 0}, gS, 1.0, s);
  const DeviceTensor gS1 =
      hasS1 ? deviceEinsumNew("kp,ak->pa", {&cs.Ut, &gt1}, 1.0, dr) : DeviceTensor();
  // The third term's dE/dG_v = gbar3, through G_v = Vn sym(S_v) U.
  if (threeTerm && nd > 0) addSymDroppedKept(cs.Vn, second.gbar3, nd, m, cs.Ut, gS, dr);
  w.sweep(gS, gK, gS1, *adj);
  if (!direction) return {};

  // gdot[u]: the same three matrix adjoints, of Eddot(u, v)'s bilinear part
  // in v (energy_finish's pcAdjointSecond* header), then the same sweep.
  acc.ciEdot = second.edotU;
  // dE_dot/dK = U sym(mggDot) U^T + sym(Vn^T hbarDot U^T).
  const DeviceTensor gKd = sandwich(cs.Ut, symHost(second.mggDot, m), m, dr);
  // dE_dot/dS = sym(U (dE_dot/dT_v)^T) + U sym(mggDotDelta) U^T
  //           + sym(Vn^T (dE_dot/dN_v)^T) + sym(Vn^T gbarDot U^T).
  const DeviceTensor gSd = sandwich(cs.Ut, symHost(second.mggDotDelta, m), m, dr);
  {
    const DeviceTensor uGtd = deviceEinsumNew("kp,qk->pq", {&cs.Ut, &gTd}, 0.5, dr);
    addTensorInto(uGtd, gSd, s);
    deviceTransposeAccum(uGtd, {1, 0}, gSd, 1.0, s);
  }
  // dE_dot/dS_1 = U (dE_dot/dt1_v)^T + Vn^T (dE_dot/dn1_v)^T.
  const DeviceTensor gS1d =
      hasS1 ? deviceEinsumNew("kp,ak->pa", {&cs.Ut, &gt1d}, 1.0, dr) : DeviceTensor();
  if (nd > 0) {
    addSymDroppedKept(cs.Vn, second.hbarDot, nd, m, cs.Ut, gKd, dr);
    addSymDroppedKept(cs.Vn, second.gbarDot, nd, m, cs.Ut, gSd, dr);
    const DeviceTensor vGn = deviceEinsumNew("kp,qk->pq", {&cs.Vn, &gN}, 0.5, dr);
    addTensorInto(vGn, gSd, s);
    deviceTransposeAccum(vGn, {1, 0}, gSd, 1.0, s);
    if (hasS1) deviceEinsumAccum("kp,ak->pa", {&cs.Vn, &gn1}, gS1d, 1.0, s);
  }
  w.sweep(gSd, gKd, gS1d, *ci->adjDot);
  return {};
}

// --- the seven classes ----------------------------------------------------------
//
// Each is energy_pc_classes.cppm's pc_<class> read off as a PcClassWalk: the
// same slab einsums (subscripts verbatim), the same basis reshapes, its
// Delta, and the reverse of its S / K / S_1 construction (energy_pc_tangent's
// pcMatrices<class>, which pcClassMatrices dispatches to) term by term.

PcClassWalk pcWalk(int cls, const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t ncore,
                   int64_t nvirt, const DeviceResources& dr) {
  // The walk's lambdas outlive this call (assemblePcClasses runs them), so
  // they capture POINTERS to the caller's objects, by value -- the caller
  // keeps all three alive across the class.
  const ActiveIntegralsDevice* const aiP = &ai;
  IntegralSource* const srcP = &src;
  const DeviceResources* const drP = &dr;
  const int64_t n = ai.h1e.dims[0];
  PcClassWalk w;
  w.cls = cls;
  const auto plainBasis = [](const DeviceTensor& T, const DeviceTensor&) {
    return tensorList(T.view());
  };
  const auto slabOf = [srcP, drP](std::initializer_list<ExtBlock> blocks,
                                  std::initializer_list<RotBlock> kinds, int64_t b0, int64_t b1) {
    PcSlabOperands o;
    for (const ExtBlock b : blocks) o.ext.push_back(srcP->slab(b, b0, b1, *drP));
    o.kinds = kinds;
    return o;
  };
  switch (cls) {
    case 0: {  // Sr: S = dm3 (rpqbac->pqrabc), K = a16, S_1 = dm2 (rpqa->pqra)
      w.block = ExtBlock::Sr;
      w.terms = {{"ipqr,pqrk->ik", 0, 0, 0}, {"ia,ak->ik", 1, 1, 0}};
      w.basis = [n](const DeviceTensor& T, const DeviceTensor& t1) {
        return tensorList(reshapeView(T, {n, n, n, T.dims[1]}), t1.view());
      };
      w.slab = [slabOf, aiP](int64_t b0, int64_t b1) {
        PcSlabOperands o = slabOf({ExtBlock::Sr}, {RotBlock::Sr, RotBlock::H1eSr}, b0, b1);
        o.ext.push_back(sliceAxis(aiP->h1e_v_Sr, 0, b0, b1));
        return o;
      };
      w.delta = [](int64_t b0, int64_t b1, const Tensor&, const Tensor& ev) {
        return srDelta(b0, b1, ev);
      };
      w.sweep = [aiP, drP, n](const DeviceTensor& gS, const DeviceTensor& gK,
                              const DeviceTensor& gS1, ScRdmAdjoint& adj) {
        const DeviceResources& r = *drP;
        const wwrStream_t s = r.stream();
        const DeviceTensor delta = deviceEye(n, r);
        adjointTerm("rpqbac->pqrabc", 0, {}, reshapeView(gS, {n, n, n, n, n, n}), adj.dm3, 1.0,
                    s);
        adjointA16(reshapeView(gK, {n, n, n, n, n, n}), aiP->h1e, aiP->h2e, delta, adj, r);
        adjointTerm("rpqa->pqra", 0, {}, reshapeView(gS1, {n, n, n, n}), adj.dm2, 1.0, s);
      };
      break;
    }
    case 1: {  // Si: S = dm3_h, K = a22, S_1 = dm2_h -- the same shapes as Sr
      w.block = ExtBlock::Si;
      w.terms = {{"qpir,pqrk->ik", 0, 0, 0}, {"ai,ak->ik", 1, 1, 0}};
      w.basis = [n](const DeviceTensor& T, const DeviceTensor& t1) {
        return tensorList(reshapeView(T, {n, n, n, T.dims[1]}), t1.view());
      };
      w.slab = [slabOf, aiP](int64_t b0, int64_t b1) {
        PcSlabOperands o = slabOf({ExtBlock::Si}, {RotBlock::Si, RotBlock::H1eSi}, b0, b1);
        o.ext.push_back(sliceAxis(aiP->h1e_v_Si, 1, b0, b1));
        return o;
      };
      w.delta = [](int64_t b0, int64_t b1, const Tensor& ec, const Tensor&) {
        return siDelta(b0, b1, ec);
      };
      w.sweep = [aiP, drP, n](const DeviceTensor& gS, const DeviceTensor& gK,
                              const DeviceTensor& gS1, ScRdmAdjoint& adj) {
        const DeviceResources& r = *drP;
        const wwrStream_t s = r.stream();
        const DeviceTensor delta = deviceEye(n, r);
        adjointA22(reshapeView(gK, {n, n, n, n, n, n}), aiP->h1e, aiP->h2e, delta, adj, r);
        const DeviceTensor mDm3h = DeviceTensor::zeros({n, n, n, n, n, n}, r);
        adjointTerm("rpqbac->pqrabc", 0, {}, reshapeView(gS, {n, n, n, n, n, n}), mDm3h, 1.0, s);
        adjointDm3H(mDm3h, delta, adj, s);
        const DeviceTensor mDm2h = DeviceTensor::zeros({n, n, n, n}, r);
        adjointTerm("rpqa->pqra", 0, {}, reshapeView(gS1, {n, n, n, n}), mDm2h, 1.0, s);
        adjointDm2H(mDm2h, delta, adj, s);
      };
      break;
    }
    case 3: {  // Sijr: S = hdm1, K = a3(.., hdm1); the (j, i) exchange row
      w.block = ExtBlock::Sijr;
      w.pattern = PcPattern::PairSwap;
      w.terms = {{"rpji,pk->rjik", 0, 0, 0}};
      w.basis = plainBasis;
      w.slab = [slabOf](int64_t b0, int64_t b1) {
        return slabOf({ExtBlock::Sijr}, {RotBlock::Sijr}, b0, b1);
      };
      w.delta = [ncore](int64_t b0, int64_t b1, const Tensor& ec, const Tensor& ev) {
        return pcSijrDelta(b0, b1, ncore, ec, ev);
      };
      w.partner = [ncore](int64_t nr) { return pcSijrPartner(nr, ncore); };
      w.sweep = [aiP, drP, n](const DeviceTensor& gS, const DeviceTensor& gK, const DeviceTensor&,
                              ScRdmAdjoint& adj) {
        const DeviceResources& r = *drP;
        const wwrStream_t s = r.stream();
        const DeviceTensor delta = deviceEye(n, r);
        // hdm1 is both the metric and a3's operand: one accumulator for both,
        // swept after a3, as SC's Sijr does.
        const DeviceTensor mHdm1 = DeviceTensor::zeros({n, n}, r);
        addTensorInto(gS, mHdm1, s);
        adjointA3(gK, aiP->h1e, aiP->h2e, delta, mHdm1, adj, s);
        adjointHoleDm1(mHdm1, adj, s);
      };
      break;
    }
    case 4: {  // Srsi: S = dm1, K = k27; the exchange row from the SrsiT slab
      w.block = ExtBlock::Srsi;
      w.pattern = PcPattern::PairTwo;
      w.terms = {{"rsip,pk->rsik", 0, 0, 0}, {"srip,pk->rsik", 1, 0, 1}};
      w.basis = plainBasis;
      w.slab = [slabOf](int64_t b0, int64_t b1) {
        return slabOf({ExtBlock::Srsi, ExtBlock::SrsiT}, {RotBlock::Srsi, RotBlock::SrsiT}, b0,
                      b1);
      };
      w.delta = [ncore, nvirt](int64_t b0, int64_t b1, const Tensor& ec, const Tensor& ev) {
        return pcSrsiDelta(b0, b1, nvirt, ncore, ec, ev);
      };
      w.sweep = [aiP, drP](const DeviceTensor& gS, const DeviceTensor& gK, const DeviceTensor&,
                           ScRdmAdjoint& adj) {
        const wwrStream_t s = drP->stream();
        addTensorInto(gS, adj.dm1, s);
        adjointK27(gK, aiP->h1e, aiP->h2e, adj, s);
      };
      break;
    }
    case 5: {  // Srs: S = rm2 (pqba->pqab), K = a7, factor 1/2
      w.block = ExtBlock::Srs;
      w.factor = 0.5;
      w.terms = {{"rsqp,pqk->rsk", 0, 0, 0}};
      w.basis = [n](const DeviceTensor& T, const DeviceTensor&) {
        return tensorList(reshapeView(T, {n, n, T.dims[1]}));
      };
      w.slab = [slabOf](int64_t b0, int64_t b1) {
        return slabOf({ExtBlock::Srs}, {RotBlock::Srs}, b0, b1);
      };
      w.delta = [nvirt](int64_t b0, int64_t b1, const Tensor&, const Tensor& ev) {
        return srsDelta(b0, b1, nvirt, ev);
      };
      w.sweep = [aiP, drP, n](const DeviceTensor& gS, const DeviceTensor& gK, const DeviceTensor&,
                              ScRdmAdjoint& adj) {
        const DeviceResources& r = *drP;
        const wwrStream_t s = r.stream();
        const DeviceTensor delta = deviceEye(n, r);
        // a7 reads rm2 and rm3, rm3 reads rm2: a7, then rm3, then rm2, whose
        // accumulator is seeded with the metric's own.
        const DeviceTensor mRm2 = DeviceTensor::zeros({n, n, n, n}, r);
        const DeviceTensor mRm3 = DeviceTensor::zeros({n, n, n, n, n, n}, r);
        adjointTerm("pqba->pqab", 0, {}, reshapeView(gS, {n, n, n, n}), mRm2, 1.0, s);
        adjointA7(reshapeView(gK, {n, n, n, n}), aiP->h1e, aiP->h2e, mRm2, mRm3, s);
        adjointRm3(mRm3, delta, mRm2, adj, s);
        adjointRm2(mRm2, delta, adj, s);
      };
      break;
    }
    case 6: {  // Sij: S = hdm2, K = a9(hdm1, hdm2, hdm3), factor 1/2
      w.block = ExtBlock::Sij;
      w.factor = 0.5;
      w.terms = {{"qpij,pqk->ijk", 0, 0, 0}};
      w.basis = [n](const DeviceTensor& T, const DeviceTensor&) {
        return tensorList(reshapeView(T, {n, n, T.dims[1]}));
      };
      w.slab = [slabOf](int64_t b0, int64_t b1) {
        return slabOf({ExtBlock::Sij}, {RotBlock::Sij}, b0, b1);
      };
      w.delta = [ncore](int64_t b0, int64_t b1, const Tensor& ec, const Tensor&) {
        return sijDelta(b0, b1, ncore, ec);
      };
      w.sweep = [aiP, drP, n](const DeviceTensor& gS, const DeviceTensor& gK, const DeviceTensor&,
                              ScRdmAdjoint& adj) {
        const DeviceResources& r = *drP;
        const wwrStream_t s = r.stream();
        const DeviceTensor delta = deviceEye(n, r);
        // hdm2 is the metric and a9's operand; hdm3 feeds hdm2 -- so a9, then
        // hdm3, then hdm2, as SC's Sij sweeps them.
        const DeviceTensor mHdm2 = DeviceTensor::zeros({n, n, n, n}, r);
        const DeviceTensor mHdm3 = DeviceTensor::zeros({n, n, n, n, n, n}, r);
        addTensorInto(reshapeView(gS, {n, n, n, n}), mHdm2, s);
        adjointA9(reshapeView(gK, {n, n, n, n}), aiP->h1e, aiP->h2e, mHdm2, mHdm3, s);
        adjointHdm3(mHdm3, delta, mHdm2, adj, s);
        adjointHdm2(mHdm2, delta, adj, r);
      };
      break;
    }
    case 7: {  // Sir: the (x, p, q) (+) (y, a, b) block patterns, S_1 one column
      w.block = ExtBlock::Sir1;
      w.terms = {{"rpiq,pqk->rik", 0, 0, 0}, {"rpqi,pqk->rik", 1, 1, 0}, {"ri,k->rik", 2, 2, 0}};
      w.basis = [n](const DeviceTensor& T, const DeviceTensor& t1) {
        const int64_t m = T.dims[1], d = T.dims[0];
        return tensorList(reshapeView(sliceAxis(T, 0, 0, n * n), {n, n, m}),
                                         reshapeView(sliceAxis(T, 0, n * n, d), {n, n, m}),
                                         reshapeView(t1, {m}));
      };
      w.slab = [slabOf, aiP](int64_t b0, int64_t b1) {
        PcSlabOperands o = slabOf({ExtBlock::Sir1, ExtBlock::Sir2},
                                  {RotBlock::Sir1, RotBlock::Sir2, RotBlock::H1eSir}, b0, b1);
        o.ext.push_back(sliceAxis(aiP->h1e_v_Sir, 0, b0, b1));
        return o;
      };
      w.delta = [ncore](int64_t b0, int64_t b1, const Tensor& ec, const Tensor& ev) {
        return pcSirDelta(b0, b1, ncore, ec, ev);
      };
      w.sweep = [aiP, drP, n](const DeviceTensor& gS, const DeviceTensor& gK,
                              const DeviceTensor& gS1, ScRdmAdjoint& adj) {
        const DeviceResources& r = *drP;
        const wwrStream_t s = r.stream();
        const DeviceTensor offDiag =
            uploadTensor(Tensor::fromFlat({2, 2}, {2.0, -1.0, -1.0, 0.0}), r);
        const DeviceTensor block22 =
            uploadTensor(Tensor::fromFlat({2, 2}, {0.0, 0.0, 0.0, 1.0}), r);
        const DeviceTensor oneBody = uploadTensor(Tensor::fromFlat({2}, {2.0, -1.0}), r);
        const DeviceTensor delta = deviceEye(n, r);
        const DeviceTensor gS6 = reshapeView(gS, {2, n, n, 2, n, n});
        const DeviceTensor gK6 = reshapeView(gK, {2, n, n, 2, n, n});
        // S = offDiag (x) dm2[q,p,a,b] + block22 (x) S22, S22 = 2 delta_pa dm1[q,b]
        //     - dm2[q,b,a,p] + delta_ab dm1[q,p] -- pcMatricesSir, term by term.
        adjointTerm("xy,qpab->xpqyab", 1, {&offDiag}, gS6, adj.dm2, 1.0, s);
        const DeviceTensor mS22 = DeviceTensor::zeros({n, n, n, n}, r);
        adjointTerm("xy,pqab->xpqyab", 1, {&block22}, gS6, mS22, 1.0, s);
        adjointTerm("pa,qb->pqab", 1, {&delta}, mS22, adj.dm1, 2.0, s);
        adjointTerm("qbap->pqab", 0, {}, mS22, adj.dm2, -1.0, s);
        adjointTerm("ab,qp->pqab", 1, {&delta}, mS22, adj.dm1, 1.0, s);
        // K = offDiag (x) a12 + block22 (x) a13.
        const DeviceTensor mA12 = DeviceTensor::zeros({n, n, n, n}, r);
        const DeviceTensor mA13 = DeviceTensor::zeros({n, n, n, n}, r);
        adjointTerm("xy,pqab->xpqyab", 1, {&offDiag}, gK6, mA12, 1.0, s);
        adjointTerm("xy,pqab->xpqyab", 1, {&block22}, gK6, mA13, 1.0, s);
        adjointA12(mA12, aiP->h1e, aiP->h2e, adj, s);
        adjointA13(mA13, aiP->h1e, aiP->h2e, delta, adj, s);
        // S_1 = oneBody (x) dm1[q,p].
        adjointTerm("x,qp->xpq", 1, {&oneBody}, reshapeView(gS1, {2, n, n}), adj.dm1, 1.0, s);
      };
      break;
    }
    default:
      check(false, std::format("pcWalk: class {} has no PC walk (Sijrs is SC's)", cls));
  }
  return w;
}

// All eight classes, CLASSES order: the seven above, and Sijrs through SC's
// own gradientSijrs (no active index, so PC = SC -- energy, pseudodensity,
// rotation and Delta term alike). `epsAdjoint` sizes each class's dE/de, which
// the CI source term folds onto dm1 when the orbital energies move.
Result<PcGradientResult> assemblePcClasses(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                           int64_t batch, const DeviceTensor& dm1,
                                           const DeviceTensor& dm2, const DeviceTensor& dm3,
                                           const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                           const ScRotation& rot, ScRdmAdjoint* adj,
                                           bool epsAdjoint, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const Tensor hostEC = downloadTensor(ai.e_core, s);
  const Tensor hostEV = downloadTensor(ai.e_virt, s);
  const int64_t ncore = hostEC.size(), nvirt = hostEV.size();
  const PcRdms rdm{&dm1, &dm2, &dm3, &f3ac, &f3ca};
  const DeviceTensor z3ac = DeviceTensor::zeros(f3ac.dims, dr);
  const DeviceTensor z3ca = DeviceTensor::zeros(f3ca.dims, dr);
  const PcRdms zeroF3{&dm1, &dm2, &dm3, &z3ac, &z3ca};

  PcGradientResult r;
  for (PcClassGradient& c : r.classes) {
    c.rotExt.assign(rot.size(), 0.0);
    c.rotAct.assign(rot.size(), 0.0);
    c.rotTangent.assign(rot.size(), 0.0);
    c.rotEps.assign(rot.size(), 0.0);
    if (epsAdjoint) {
      c.dEdCore.assign(static_cast<std::size_t>(ncore), 0.0);
      c.dEdVirt.assign(static_cast<std::size_t>(nvirt), 0.0);
    }
  }
  for (const int cls : {0, 1, 3, 4, 5, 6, 7}) {
    PcClassGradient& c = r.classes[static_cast<std::size_t>(cls)];
    // A class with no tuple (no core, or no virtual) has no energy: Done, 0,
    // as the energy path's empty slab walk leaves it -- except that the
    // energy path still SOLVES it, and so does this, so a refusal is the
    // same refusal either way.
    const PcClassWalk w = pcWalk(cls, ai, src, ncore, nvirt, dr);
    NEVPT2_TRY(pcClassAdjoint(w, ai, src, batch, hostEC, hostEV, rdm, zeroF3, rot, c, adj,
                              /*ci=*/nullptr, dr));
  }
  {
    // Sijrs: SC's own class, its ScClassGradient copied across.
    ScClassGradient sc;
    sc.rotExt.assign(rot.size(), 0.0);
    sc.rotAct.assign(rot.size(), 0.0);
    sc.rotTangent.assign(rot.size(), 0.0);
    sc.rotEps.assign(rot.size(), 0.0);
    sc.epsAdjoint = epsAdjoint;
    if (epsAdjoint) {
      sc.dEdCore.assign(static_cast<std::size_t>(ncore), 0.0);
      sc.dEdVirt.assign(static_cast<std::size_t>(nvirt), 0.0);
    }
    gradientSijrs(src, batch, hostEC, hostEV, rot, sc, /*mix=*/nullptr, dr);
    PcClassGradient& c = r.classes[2];
    c.result.status = PcStatus::Done;
    c.result.energy = sc.energy;
    c.rotExt = sc.rotExt;
    c.rotAct = sc.rotAct;
    c.rotTangent = sc.rotTangent;
    c.rotEps = sc.rotEps;
    if (epsAdjoint) {
      c.dEdCore = sc.dEdCore;
      c.dEdVirt = sc.dEdVirt;
    }
  }
  for (const PcClassGradient& c : r.classes) r.total += c.result.energy;
  return r;
}

// The CI walk over all eight classes (#155): every RDM class through
// pcClassAdjoint with a PcCiWalk, class `cls` sweeping into adj[cls] and --
// along a direction -- adjDot[cls] (the caller may point several classes at
// one accumulator, or give each its own for a per-class y and H_cc u). No
// rotation and the orbital energies held fixed. Sijrs reads no RDM: its
// energy is SC's, and it adds nothing to either accumulator (its y and its
// H_cc u are exactly zero, as pcHessianDevice's Eddot of it is).
Result<PcGradientResult> assemblePcCiClasses(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                             int64_t batch, const PcRdms& rdm,
                                             const PcRdms* sumU, const PcRdms* sumUU,
                                             const std::array<ScRdmAdjoint*, 8>& adj,
                                             const std::array<ScRdmAdjoint*, 8>& adjDot,
                                             const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const Tensor hostEC = downloadTensor(ai.e_core, s);
  const Tensor hostEV = downloadTensor(ai.e_virt, s);
  const int64_t ncore = hostEC.size(), nvirt = hostEV.size();
  const DeviceTensor z3ac = DeviceTensor::zeros(rdm.f3ac->dims, dr);
  const DeviceTensor z3ca = DeviceTensor::zeros(rdm.f3ca->dims, dr);
  const PcRdms zeroF3{rdm.dm1, rdm.dm2, rdm.dm3, &z3ac, &z3ca};
  const ScRotation none;
  PcGradientResult r;
  for (const int cls : {0, 1, 3, 4, 5, 6, 7}) {
    const std::size_t k = static_cast<std::size_t>(cls);
    const PcCiWalk ci{/*threeTerm=*/true, sumU, sumUU, adjDot[k]};
    const PcClassWalk w = pcWalk(cls, ai, src, ncore, nvirt, dr);
    NEVPT2_TRY(pcClassAdjoint(w, ai, src, batch, hostEC, hostEV, rdm, zeroF3, none, r.classes[k],
                              adj[k], &ci, dr));
  }
  {
    ScClassGradient sc;
    gradientSijrs(src, batch, hostEC, hostEV, none, sc, /*mix=*/nullptr, dr);
    r.classes[2].result.status = PcStatus::Done;
    r.classes[2].result.energy = sc.energy;
  }
  for (const PcClassGradient& c : r.classes) r.total += c.result.energy;
  return r;
}

}  // namespace nevpt2
