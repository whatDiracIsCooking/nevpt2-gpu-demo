// nevpt2.energy_finish -- the class energies' pure host arithmetic: the slab
// walk every class runs (forBatches), the strongly-contracted finish
// (normToEnergy, PySCF's _norm_to_energy) and ITS DERIVATIVE
// (normToEnergyDerivative, normToEnergyTangent -- the only calculus the
// forward tangent of an SC class energy needs) and its SECOND derivative
// (normToEnergyHessian, the 3x3 in the d form, and the multiplier tangents
// applyFinishHessian / normToEnergyMultiplierTangent -- #150,
// docs/hessian-theory.md Sec. 2), the partially-contracted
// finish (pcRankCut, finishSingle, checkDenominator), the PC TANGENT's per-tuple
// arithmetic (pcCoefficients, pcQuadForm, tangentSingle) and its SECOND
// derivative's (pcQuadSecond, pcRotationBlocks, hessianSingle, hessianPair,
// pcLiftNorm -- #152) and that derivative's reverse in v, the per-tuple bars
// of PC's H_cc u and the three-term y (pcAdjointSecond* -- #155), with the PC result
// types they all fill. No device, no BLAS, no solver: nevpt2.energy
// re-exports this module, and its partitions (:sc_finish, :sc_tangent,
// :pc_solve, :pc_tangent, the class files) do the device work and call these.
// Exported so the unit tier can check them (test/energy_finish/).
// Carved out of the :shared, :sc_finish and :pc_solve partitions and
// energy.cppm unchanged.
export module nevpt2.energy_finish;

import std;
// Re-exported: finishSingle reads a host Tensor, and every extent here is an
// int64_t (nevpt2.tensor re-exports nevpt2.common).
export import nevpt2.tensor;
// check(): normToEnergyTangent's five lengths, and the PC tangent helpers'
// (m, m) block shapes -- all of them set by one solve, so a mismatch is our
// bug (nevpt2.tensor imports this privately).
import nevpt2.error_handling;

export namespace nevpt2 {

// Norms below this are treated as an empty perturber space (no energy
// contribution) -- pyscf.mrpt.nevpt2's own guard against dividing by a
// vanishing norm.
inline constexpr double NUMERICAL_ZERO = 1e-14;

// --- the slab walk -------------------------------------------------------------

// Calls f(b0, b1) for consecutive [b0, b1) covering [0, n), each at most
// `batch` long (batch <= 0: one piece). n == 0 calls nothing. Every class's
// slab walk, SC and PC.
template <class F>
void forBatches(int64_t n, int64_t batch, F&& f) {
  int64_t step = batch > 0 ? batch : std::max(n, int64_t{1});
  for (int64_t b0 = 0; b0 < n; b0 += step) f(b0, std::min(n, b0 + step));
}

// --- SC ------------------------------------------------------------------------

// pyscf.mrpt.nevpt2._norm_to_energy verbatim: the strongly-contracted class
// energy is -sum_k N_k / (Delta_k + H_k/N_k) over perturbers k with
// non-vanishing norm; the returned norm is the plain sum of N_k.
std::pair<double, double> normToEnergy(const std::vector<double>& norm,
                                        const std::vector<double>& h,
                                        const std::vector<double>& diff) {
  double normT = 0.0, enerT = 0.0;
  for (std::size_t k = 0; k < norm.size(); ++k) {
    normT += norm[k];
    if (std::fabs(norm[k]) > NUMERICAL_ZERO) {
      enerT -= norm[k] / (diff[k] + h[k] / norm[k]);
    }
  }
  return {normT, enerT};
}

// --- SC: the finish's derivative -----------------------------------------------
//
// The CLASS CONTRACTIONS ARE LINEAR IN THE RDMs at fixed integrals, so the
// tangent of every N_k and H_k is the same einsum chain run with the tangent
// RDMs substituted (nevpt2.energy's energiesTangentDevice, whose
// :sc_tangent partition does exactly that). The only part of the SC forward
// derivative that is actual calculus is the scalar finish above, and it is
// here: no device, no tensor, one perturber at a time. (PC's tangent has
// calculus of its own -- a pseudo-inverse's derivative, energy_pc_solve.cppm
// -- and its per-tuple half is the PC section below.)

// The derivative of normToEnergy's per-perturber energy with respect to its
// two RDM-dependent inputs: {dE_k/dN_k, dE_k/dH_k}. Delta_k is NOT one of
// them -- it is a sum of orbital energies, fixed as the RDMs move.
//
// With the amplitude T_k = -1 / (Delta_k + H_k / N_k), so that E_k = T_k N_k,
// eliminating H_k / N_k = -1/T_k - Delta_k gives
//
//   dE/dN = 2 T + Delta T^2,      dE/dH = T^2
//
// -- Park, JCTC 2019 (arXiv:1907.10180), Eqs. 36-37 (his P_k and Q_k).
// nevpt2.sc_amplitudes' scMultipliers reads the same two numbers out of the
// same division for the ADJOINT; they are defined here because the forward
// tangent's finish needs them and nevpt2.energy cannot depend on the gradient
// module. The identity that ties the two sides together, Park Eq. 40, is
// P_k N_k + Q_k H_k = E_k (the finish is homogeneous of degree 1 in (N, H)).
//
// A norm at or below NUMERICAL_ZERO is normToEnergy's empty perturber space:
// it contributes no energy, so it has no derivative either.
std::pair<double, double> normToEnergyDerivative(double norm, double h, double diff) {
  if (!(std::fabs(norm) > NUMERICAL_ZERO)) return {0.0, 0.0};
  const double t = -1.0 / (diff + h / norm);
  return {2.0 * t + diff * t * t, t * t};
}

// The tangent of normToEnergy: one class's perturber vectors and their
// tangents in, the derivatives of normToEnergy's own pair out --
//
//   normT_dot = sum_k Ndot_k           (the plain sum, every perturber)
//   enerT_dot = sum_k (P_k Ndot_k + Q_k Hdot_k)
//
// over the perturbers normToEnergy's guard keeps. All five vectors are one
// class's, same length and same order; a mismatch is the caller's broken
// invariant (every caller builds them side by side), so it aborts.
std::pair<double, double> normToEnergyTangent(
    const std::vector<double>& norm, const std::vector<double>& h,
    const std::vector<double>& diff, const std::vector<double>& normDot,
    const std::vector<double>& hDot,
    std::source_location loc = std::source_location::current()) {
  const std::size_t k = norm.size();
  if (h.size() != k || diff.size() != k || normDot.size() != k || hDot.size() != k)
    check(false,
          std::format("normToEnergyTangent: norm/h/diff/normDot/hDot lengths differ ({}, {}, "
                      "{}, {}, {})",
                      k, h.size(), diff.size(), normDot.size(), hDot.size()),
          loc);
  double normT = 0.0, enerT = 0.0;
  for (std::size_t i = 0; i < k; ++i) {
    normT += normDot[i];
    const auto [p, q] = normToEnergyDerivative(norm[i], h[i], diff[i]);
    enerT += p * normDot[i] + q * hDot[i];
  }
  return {normT, enerT};
}

// --- SC: the finish's derivative in Delta (#87) --------------------------------
//
// Delta_k is a signed sum of orbital energies, and those are NOT constants of
// the CI vector: e_core / e_virt are the generalized Fock's diagonal, built
// from the active 1-RDM (docs/gradients.md, Sec. 5.5). So the complete
// forward tangent carries one more term per perturber, its Delta's tangent
// against
//
//   dE_k/dDelta_k = N_k T_k^2                      (Park Eq. 38's R_k)
//
// -- E_k = -N_k / (Delta_k + H_k / N_k) differentiated in its third input.
// nevpt2.sc_amplitudes' scMultipliers reads the same R_k out of the same
// division for the ADJOINT; this is the forward side's own copy, for the
// reason normToEnergyDerivative above gives. Guarded the same way.
double normToEnergyDeltaDerivative(double norm, double h, double diff) {
  if (!(std::fabs(norm) > NUMERICAL_ZERO)) return 0.0;
  const double t = -1.0 / (diff + h / norm);
  return norm * t * t;
}

// sum_k R_k Deltadot_k over one class's perturbers: the orbital energies' half
// of the class energy's tangent, added to normToEnergyTangent's
// P.Ndot + Q.Hdot. `diffDot` is the class's own Delta construction read at
// the orbital energies' tangent (Delta is linear and homogeneous in them, so
// that IS Deltadot). Same lengths, same order, or it aborts.
double normToEnergyDeltaTangent(const std::vector<double>& norm, const std::vector<double>& h,
                                const std::vector<double>& diff,
                                const std::vector<double>& diffDot,
                                std::source_location loc = std::source_location::current()) {
  const std::size_t k = norm.size();
  if (h.size() != k || diff.size() != k || diffDot.size() != k)
    check(false,
          std::format("normToEnergyDeltaTangent: norm/h/diff/diffDot lengths differ ({}, {}, "
                      "{}, {})",
                      k, h.size(), diff.size(), diffDot.size()),
          loc);
  double enerT = 0.0;
  for (std::size_t i = 0; i < k; ++i)
    enerT += normToEnergyDeltaDerivative(norm[i], h[i], diff[i]) * diffDot[i];
  return enerT;
}

// --- SC: the finish's SECOND derivative (#150) ---------------------------------
//
// The 3x3 Hessian of the per-perturber finish E = -N^2 / d, d = N Delta + H,
// in its three inputs (N, H, Delta) -- docs/hessian-theory.md, Sec. 2. Like
// P, Q, R above it is reference-independent and is the only new calculus a
// second derivative of the SC energy needs (everything downstream is these
// numbers contracted back against integrals and densities).
//
// WRITTEN IN THE d FORM, NOT IN THE AMPLITUDE T, for two reasons (Sec. 2.1-2.2):
//
//   - E reads H and Delta only through d, so the (H, Delta) block is rank one,
//     E_dd (1, N) (x) (1, N) with E_dd = -2 N^2 / d^3: four independent
//     entries (E_NN, E_NH, E_NDelta, E_dd), and E_HDelta / E_DeltaDelta are
//     N and N^2 times E_HH BY CONSTRUCTION here.
//   - the T forms (E_NN = (2T/N)(1 + Delta T)^2, E_HH = 2T^3/N, ...) carry
//     removable 1/N singularities that none of P, Q, R has; in d all six are
//     1/N-free:
//
//       E_NN      = -2/d + 4 N Delta/d^2 - 2 N^2 Delta^2/d^3
//       E_NH      =  2 N/d^2 - 2 N^2 Delta/d^3
//       E_NDelta  =  3 N^2/d^2 - 2 N^3 Delta/d^3
//       E_HH      = -2 N^2/d^3,  E_HDelta = N E_HH,  E_DeltaDelta = N^2 E_HH
//
// A norm at or below NUMERICAL_ZERO is normToEnergy's empty perturber space,
// whose energy is identically zero, so all nine entries are zero there --
// exactly as normToEnergyDerivative returns {0, 0} -- and N = 0 is never a
// NaN. d = 0 (the finish's own pole) is not guarded, as it is not for P, Q.
struct FinishHessian {
  double nn = 0.0;          // E_NN
  double nh = 0.0;          // E_NH
  double nDelta = 0.0;      // E_NDelta
  double dd = 0.0;          // E_dd: the rank-one (H, Delta) block's coefficient
  double hh = 0.0;          // E_HH       = dd
  double hDelta = 0.0;      // E_HDelta   = N dd
  double deltaDelta = 0.0;  // E_DeltaDelta = N^2 dd
};

FinishHessian normToEnergyHessian(double norm, double h, double diff) {
  if (!(std::fabs(norm) > NUMERICAL_ZERO)) return {};
  const double d = norm * diff + h;
  const double a = norm / d;      // N / d
  const double b = a * diff;      // N Delta / d
  FinishHessian e;
  // Each line is Sec. 2.1's d form term by term, with a = N/d, b = N Delta/d.
  e.nn = (-2.0 + 4.0 * b - 2.0 * b * b) / d;
  e.nh = (2.0 * a - 2.0 * a * b) / d;
  e.nDelta = 3.0 * a * a - 2.0 * a * a * b;
  e.dd = -2.0 * a * a / d;
  e.hh = e.dd;
  e.hDelta = norm * e.dd;
  e.deltaDelta = norm * norm * e.dd;
  return e;
}

// One perturber's multiplier tangents: the derivatives of P, Q, R (Park
// Eqs. 36-38) along a direction whose first-order scalar tangents are
// (Ndot, Hdot, Deltadot) -- the 3x3 applied to them (Sec. 2.4). The same
// expression serves forward-over-forward (Eddot(u, v) = (Ndot_v, Hdot_v,
// Deltadot_v) . these, plus the first-order terms) and reverse-over-forward
// (the pseudodensity walk with P, Q, R replaced by their tangents).
struct MultiplierTangent {
  double p = 0.0;  // Pdot = E_NN Ndot + E_NH Hdot + E_NDelta Deltadot
  double q = 0.0;  // Qdot = E_NH Ndot + E_HH Hdot + E_HDelta Deltadot
  double r = 0.0;  // Rdot = E_NDelta Ndot + E_HDelta Hdot + E_DeltaDelta Deltadot
};

MultiplierTangent applyFinishHessian(const FinishHessian& e, double normDot, double hDot,
                                     double diffDot) {
  return {e.nn * normDot + e.nh * hDot + e.nDelta * diffDot,
          e.nh * normDot + e.hh * hDot + e.hDelta * diffDot,
          e.nDelta * normDot + e.hDelta * hDot + e.deltaDelta * diffDot};
}

// One class's multiplier tangents, perturber by perturber, in its own order:
// P, Q, R's derivatives along (normDot, hDot, diffDot). `diffDot` is the
// class's Delta construction read at the orbital energies' tangent (as for
// normToEnergyDeltaTangent; all zeros for frozen orbital energies). Six
// vectors of one class, same length and order, or it aborts. A guarded
// perturber's three tangents are zero.
struct ClassMultiplierTangents {
  std::vector<double> p, q, r;
};

ClassMultiplierTangents normToEnergyMultiplierTangent(
    const std::vector<double>& norm, const std::vector<double>& h,
    const std::vector<double>& diff, const std::vector<double>& normDot,
    const std::vector<double>& hDot, const std::vector<double>& diffDot,
    std::source_location loc = std::source_location::current()) {
  const std::size_t k = norm.size();
  if (h.size() != k || diff.size() != k || normDot.size() != k || hDot.size() != k ||
      diffDot.size() != k)
    check(false,
          std::format("normToEnergyMultiplierTangent: norm/h/diff/normDot/hDot/diffDot lengths "
                      "differ ({}, {}, {}, {}, {}, {})",
                      k, h.size(), diff.size(), normDot.size(), hDot.size(), diffDot.size()),
          loc);
  ClassMultiplierTangents out;
  out.p.resize(k);
  out.q.resize(k);
  out.r.resize(k);
  for (std::size_t i = 0; i < k; ++i) {
    const MultiplierTangent t = applyFinishHessian(normToEnergyHessian(norm[i], h[i], diff[i]),
                                                   normDot[i], hDot[i], diffDot[i]);
    out.p[i] = t.p;
    out.q[i] = t.q;
    out.r[i] = t.r;
  }
  return out;
}

// The second sum of docs/hessian-theory.md Sec. 4.1, over one class's
// perturbers: sum_k (Ndot_v, Hdot_v, Deltadot_v)_k . Hess f_k .
// (Ndot_u, Hdot_u, Deltadot_u)_k -- the finish's 3x3 sandwiched between two
// directions' first-order per-perturber tangents. With
// normToEnergyTangent(norm, h, diff, N''[u,v], H''[u,v]).second (the first
// sum, at frozen orbital energies) it is the class's Eddot(u, v). Symmetric
// in u <-> v, as the 3x3 is. `diffDot*` all zeros hold the orbital energies
// fixed. Nine vectors of one class, same length and order, or it aborts; a
// guarded perturber contributes zero.
double normToEnergyHessianSandwich(const std::vector<double>& norm, const std::vector<double>& h,
                                   const std::vector<double>& diff,
                                   const std::vector<double>& normDotU,
                                   const std::vector<double>& hDotU,
                                   const std::vector<double>& diffDotU,
                                   const std::vector<double>& normDotV,
                                   const std::vector<double>& hDotV,
                                   const std::vector<double>& diffDotV,
                                   std::source_location loc = std::source_location::current()) {
  const std::size_t k = norm.size();
  for (const std::vector<double>* x :
       {&h, &diff, &normDotU, &hDotU, &diffDotU, &normDotV, &hDotV, &diffDotV})
    if (x->size() != k)
      check(false,
            std::format("normToEnergyHessianSandwich: the nine vectors' lengths differ ({} "
                        "against {})",
                        x->size(), k),
            loc);
  double sum = 0.0;
  for (std::size_t i = 0; i < k; ++i) {
    const MultiplierTangent t = applyFinishHessian(normToEnergyHessian(norm[i], h[i], diff[i]),
                                                   normDotU[i], hDotU[i], diffDotU[i]);
    sum += t.p * normDotV[i] + t.q * hDotV[i] + t.r * diffDotV[i];
  }
  return sum;
}

// The orbital energies' tangent along one CI direction, host-resident: what
// energiesTangentDevice reads Deltadot from, in the golden's e_core / e_virt
// order. nevpt2.gradient_plan's orbitalEnergyDot builds it from the active
// 1-RDM's tangent and the gradient sidecar's integrals.
struct OrbitalEnergyDot {
  Tensor core;  // (ncore)
  Tensor virt;  // (nvirt)
};

// --- PC ------------------------------------------------------------------------

// S's cut, relative to its largest eigenvalue (docs/pc-nevpt2.md, "The
// singular-metric convention"): two decades above the structural null modes'
// noise, two below the first real mode on cc-pVTZ.
inline constexpr double PC_TAU = 1e-13;
// The smallest acceptable smallest-kept / largest-dropped ratio: the
// generator's PC_MIN_GAP.
inline constexpr double PC_MIN_GAP = 1e3;

enum class PcStatus {
  NotYet,   // not computed (the initial value; every class is computed)
  Done,     // energy computed
  Refused,  // gap below PC_MIN_GAP, or a non-positive denominator
};

// S's spectrum for one class, as the gap check saw it.
struct PcSpectrum {
  int64_t d = 0;                    // basis dimension
  int64_t dropped = 0;              // modes with s_k <= PC_TAU * s_max
  double largestDropped = 0.0;  // max |s_k| / s_max over dropped modes (0 if none)
  double smallestKept = 0.0;    // min s_k / s_max over kept modes
  double gap = 0.0;             // smallestKept / (dropped ? largestDropped : PC_TAU)
  double minDenominator = 0.0;  // min over tuples and modes of lambda_k + Delta_t
};

struct PcClassResult {
  PcStatus status = PcStatus::NotYet;
  double energy = 0.0;
  bool hasSpectrum = false;  // false for Sijrs (no active index, PC = SC)
  PcSpectrum spectrum;
  std::string why;  // for Refused
};

// One class's tangent beside the energy it differentiates: `result` is that
// class's own PcClassResult, from the SAME solve (so a class refused by the
// gap or denominator check is refused here too -- `edot` means nothing unless
// result.status is Done, and the report says REFUSED rather than printing a
// zero).
struct PcClassTangent {
  PcClassResult result;
  double edot = 0.0;  // d/dh of result.energy along the caller's CI direction
};

// One class's SECOND directional derivative beside the energy and the two
// first derivatives it is built from (#152, docs/hessian-theory.md Sec. 6):
// `result` is the class's own PcClassResult from the same solve, exactly as
// PcClassTangent's, so a class the gap or denominator check refuses -- or,
// here, whose dropped modes lift along the direction pair (`lift`) -- is
// refused and none of the three numbers means anything.
struct PcClassHessian {
  PcClassResult result;
  double edotU = 0.0;  // Edot(u), all three terms (Sec. 6.2): exact at any state
  double edotV = 0.0;  // Edot(v)
  double eddot = 0.0;  // Eddot(u, v), at the eigenstate (0 when not asked for)
  // The largest of |P S_a P|_F and |L_ab|_F over (a, b) in {u, v}, relative
  // to s_max: S's dropped eigenvalues leaving zero at first or second order
  // along the pair (0 when nothing is dropped, or no second derivative was
  // asked for).
  double lift = 0.0;
};

// The relative lift above which a (class, direction pair) is refused: S's
// dropped modes rise at the order the second derivative probes, so the
// energy is the derivative of the PC_TAU threshold, not a smooth function
// (docs/hessian-theory.md, Sec. 6.2). Measured (docs/pc-nevpt2.md, "What the
// cut drops, and how it moves"): where nothing lifts, |L| / s_max is
// round-off, at most 3.4e-8 (acrolein Sr, S^+ amplifying rounding); every
// real h^2 lift is at least 1.1e-2. 1e-5 sits between them with almost three
// decades of margin either way.
inline constexpr double PC_LIFT_TOL = 1e-5;

// The refusal's `why` for a relative lift above PC_LIFT_TOL: one wording for
// pcHessianDevice (#152) and the PC Hessian-vector product (#155), which
// refuse the same (class, direction) for the same reason.
std::string pcLiftRefusalWhy(double lift) {
  return std::format(
      "S's dropped modes lift along this direction pair (|P S P|, |L| / s_max = {:.1e} > "
      "PC_LIFT_TOL {:.0e}): the energy is not twice differentiable here",
      lift, PC_LIFT_TOL);
}

// The cut and the gap check, on S's eigenvalues `s` in ASCENDING order (as
// syevd returns them), so the dropped modes are a prefix: every s_k <=
// PC_TAU * s_max is dropped, the rest are kept, and the class is refused
// when S has no positive eigenvalue (!(s_max > 0): a NaN refuses too) or the
// gap -- smallest kept / largest |dropped|, both relative to s_max, with
// PC_TAU as the floor when nothing is dropped -- is below PC_MIN_GAP.
//
// The returned result has hasSpectrum set and its spectrum filled (d,
// dropped, largestDropped, smallestKept, gap; only d when s_max refuses).
// Its status is Refused, with `why`, or NotYet for an accepted cut: the solve
// that called this still has m = d - spectrum.dropped kept modes to work
// through, and marks the class Done itself. The rank decided here is what
// the PC tangent (and adjoint) need locally constant (energy_pc_solve.cppm,
// "the tangent of that solve"), which is the gap's whole job.
PcClassResult pcRankCut(std::span<const double> s) {
  PcClassResult r;
  r.hasSpectrum = true;
  PcSpectrum& sp = r.spectrum;
  const int64_t d = std::ssize(s);
  sp.d = d;
  const double smax = d > 0 ? s[narrowTo<std::size_t>(d - 1)] : 0.0;
  if (!(smax > 0.0)) {
    r.status = PcStatus::Refused;
    r.why = "S has no positive eigenvalue";
    return r;
  }
  int64_t nd = 0;
  while (nd < d && s[narrowTo<std::size_t>(nd)] <= PC_TAU * smax) ++nd;
  sp.dropped = nd;
  for (int64_t k = 0; k < nd; ++k)
    sp.largestDropped =
        std::max(sp.largestDropped, std::fabs(s[narrowTo<std::size_t>(k)]) / smax);
  sp.smallestKept = s[narrowTo<std::size_t>(nd)] / smax;  // nd < d: s_max itself is kept
  const double floor = nd > 0 ? sp.largestDropped : PC_TAU;
  sp.gap = floor > 0.0 ? sp.smallestKept / floor : std::numeric_limits<double>::infinity();
  if (sp.gap < PC_MIN_GAP) {
    r.status = PcStatus::Refused;
    r.why = std::format("S's gap {:.1e} < PC_MIN_GAP {:.0e}", sp.gap, PC_MIN_GAP);
  }
  return r;
}

// Marks `r` refused when a denominator is not positive (H_D must be positive
// on the kept range; the measured floor is 0.83 Eh -- docs/pc-nevpt2.md,
// "The singular-metric convention").
void checkDenominator(PcClassResult& r, double minDen) {
  r.spectrum.minDenominator = minDen;
  if (!(minDen > 0.0)) {
    r.status = PcStatus::Refused;
    r.why = std::format("a denominator lambda_k + Delta_t = {:.3e} is not positive", minDen);
  }
}

// The external-index finish Sr and Si share: E = -sum_t sum_k Y[t,k]^2 /
// (lambda_k + Delta_t), factor 1, over the slab's tuples t (b0 + row).
void finishSingle(const Tensor& y, const std::vector<double>& lambda,
                  const std::vector<double>& delta, double& e, double& minDen) {
  const int64_t nt = std::ssize(delta), m = std::ssize(lambda);
  for (int64_t t = 0; t < nt; ++t)
    for (int64_t k = 0; k < m; ++k) {
      const double den = lambda[k] + delta[t];
      minDen = std::min(minDen, den);
      const double yk = y(t, k);
      e -= yk * yk / den;
    }
}

// --- PC: the tangent's per-tuple arithmetic -----------------------------------
//
// The three lines every PC tangent class shares (the derivation is in
// energy_pc_solve.cppm, "the tangent of that solve"): with the solve's
// eigenbasis (U, lambda) and the tuple's Delta,
//
//   g_k        = y_k / (lambda_k + Delta)        (so x_vec = U g)
//   Edot_term  = 2 g . ydot - g^T (mK + Delta nS) g
//
// where `mK`/`nS` are the (m, m) row-major U^T sym(Kdot) U and
// U^T sym(Sdot) U. No device, no eigenvector derivative -- only the solve the
// energy already ran.

// g_k = y_k / (lambda_k + Delta), written into `g` (resized to lambda's
// length, so a tuple loop reuses one buffer); `minDen` is tracked exactly as
// finishSingle does, since the same denominators divide here.
void pcCoefficients(std::span<const double> y, const std::vector<double>& lambda, double delta,
                    std::vector<double>& g, double& minDen) {
  const int64_t m = std::ssize(lambda);
  g.resize(narrowTo<std::size_t>(m));
  for (int64_t k = 0; k < m; ++k) {
    const double den = lambda[k] + delta;
    minDen = std::min(minDen, den);
    g[narrowTo<std::size_t>(k)] = y[narrowTo<std::size_t>(k)] / den;
  }
}

// g1^T (mK + delta nS) g2 -- the x_1^T Adot x_2 of the tangent, in the
// eigenbasis. Both blocks are (m, m) row-major with m = g1.size().
double pcQuadForm(const std::vector<double>& g1, const std::vector<double>& g2,
                  const std::vector<double>& mK, const std::vector<double>& nS, double delta) {
  const std::size_t m = g1.size();
  // Shapes come from one solve, so a mismatch is a broken invariant here, not
  // bad input.
  check(g2.size() == m && mK.size() == m * m && nS.size() == m * m,
        "pcQuadForm: the (m, m) blocks and the two coefficient vectors disagree");
  double total = 0.0;
  for (std::size_t k = 0; k < m; ++k) {
    const double gk = g1[k];
    if (gk == 0.0) continue;
    double row = 0.0;
    for (std::size_t l = 0; l < m; ++l) row += (mK[k * m + l] + delta * nS[k * m + l]) * g2[l];
    total += gk * row;
  }
  return total;
}

// finishSingle's class energy AND its tangent in one pass over the slab's
// tuples (factor 1, Sr and Si): `ydot` is `y`'s counterpart from the Tdot /
// t1dot einsum, same (nt, m) shape. Both are accumulated, from the same
// solve, so the reported energy is the one the reported derivative
// differentiates.
void tangentSingle(const Tensor& y, const Tensor& ydot, const std::vector<double>& lambda,
                   const std::vector<double>& delta, const std::vector<double>& mK,
                   const std::vector<double>& nS, double& e, double& edot, double& minDen) {
  const int64_t nt = std::ssize(delta), m = std::ssize(lambda);
  check(y.size() == nt * m && ydot.size() == nt * m,
        "tangentSingle: y and ydot must be (tuples, m)");
  std::vector<double> g;
  for (int64_t t = 0; t < nt; ++t) {
    const std::span<const double> row(y.data().data() + t * m, narrowTo<std::size_t>(m));
    const std::span<const double> rowDot(ydot.data().data() + t * m, narrowTo<std::size_t>(m));
    pcCoefficients(row, lambda, delta[narrowTo<std::size_t>(t)], g, minDen);
    double dotTerm = 0.0, norm = 0.0;
    for (int64_t k = 0; k < m; ++k) {
      const std::size_t uk = narrowTo<std::size_t>(k);
      dotTerm += g[uk] * rowDot[uk];
      norm += g[uk] * row[uk];
    }
    e -= norm;  // sum_k y_k^2 / den == g . y
    edot -= 2.0 * dotTerm - pcQuadForm(g, g, mK, nS, delta[narrowTo<std::size_t>(t)]);
  }
}

// sum_t R_t Deltadot_t over a single-coupling class's tuples (Sr, Si, Srs,
// Sij, Sir), with R_t = dE_t/dDelta_t = factor |g_t|^2 -- E_t =
// -factor sum_k y_tk^2 / (lambda_k + Delta_t) differentiated in its Delta
// (#74). The orbital energies' half of the PC CI tangent, when they move with
// the active density (`deltaDot` is the class's own Delta construction read
// at the orbital energies' tangent); y is (nt, m), one row per tuple.
double pcDeltaTangentSingle(const Tensor& y, const std::vector<double>& lambda,
                            const std::vector<double>& delta,
                            const std::vector<double>& deltaDot, double factor) {
  const int64_t nt = std::ssize(delta), m = std::ssize(lambda);
  check(y.size() == nt * m && std::ssize(deltaDot) == nt,
        "pcDeltaTangentSingle: y must be (tuples, m) and deltaDot one per tuple");
  double sum = 0.0;
  for (int64_t t = 0; t < nt; ++t) {
    double gg = 0.0;
    for (int64_t k = 0; k < m; ++k) {
      const double g = y.flat(t * m + k) / (lambda[narrowTo<std::size_t>(k)] +
                                            delta[narrowTo<std::size_t>(t)]);
      gg += g * g;
    }
    sum += factor * gg * deltaDot[narrowTo<std::size_t>(t)];
  }
  return sum;
}

// --- PC: the second derivative's per-tuple arithmetic (#152) ------------------
//
// docs/hessian-theory.md Sec. 6.2, in the solve's eigenbasis. The solve is
// S's cut restricted inverse M = Q (Q^T A Q)^-1 Q^T = U D U^T, with U the
// solve's (d, m) eigenvectors (U^T S U = 1, U^T K U = lambda), D =
// diag(1 / (lambda + Delta)), S^+ = U U^T and P = Vn^T Vn the projector on
// S's dropped space (Vn its (nd, d) dropped eigenvectors). Every class energy
// is a sum of quadratic forms Q(r) = b_r^T M b_r over a tuple's rows (the
// single-coupling classes); Sijr / Srsi need the bilinear b_a^T M b_b, which
// is Q's polarisation, [Q(a + b) - Q(a - b)] / 4, exactly -- every row below
// is linear in the slab row, so a row of (a +- b) is the rows' sum or
// difference.
//
// With y = U^T b, y_u = U^T b_u, y_uv = U^T b_uv, g = D y (so x = M b = U g),
// alpha_u = U^T A_u U = mK_u + Delta nS_u, beta the same at (u, v),
// C = U^T U (rho = S^+ x = U C g, sigma = S^+ b = U y), and the dropped-space
// blocks G_u = Vn S_u U, H_u = Vn K_u U, H_0 = Vn K U (nd x m), r_v = Vn b_v:
//
//   Q     = g . y
//   Q_u   = 2 g . y_u - g^T alpha_u g - 2 g^T C G_u^T H_0 g
//
// The third term is -2 rho^T S_u P A x (P S = 0, so P A = P K): it vanishes at
// the eigenstate, where P K = 0, and is why the device's two-term tangent is
// exact there and nowhere else. At the eigenstate, Sec. 6.2's
//
//   Q_uv  = 2 g . y_uv - g^T beta g + 2 (y_u - alpha_u g)^T D (y_v - alpha_v g)
//         + 2 rho^T S_u P g_v - 2 rho^T S_v P A_u x + 2 rho^T S_v P S_u sigma
//
// with g_v = b_v - A_v x, i.e. in these blocks
//
//   R1 = 2 (G_u C g) . r_v - 2 g^T C G_u^T (H_v + Delta G_v) g
//   R2 = -2 g^T C G_v^T (H_u + Delta G_u) g
//   R3 = 2 g^T C G_v^T G_u y
//
// No eigenvector derivative and no second solve; the rotation terms carry S^+
// (through C) and so scale with 1 / the smallest kept eigenvalue of S, not
// with the gap PC_MIN_GAP checks (docs/pc-nevpt2.md, "What the cut drops, and
// how it moves"). With nd = 0 every rotation term is zero and Q_u is
// tangentSingle's own 2 g . ydot - g^T alpha g.

// The (m, m) eigenbasis blocks, each row-major U^T sym(Xdot) U as tangentClass
// builds it: along u, along v, and at (u, v) (empty when only the first
// derivatives are wanted).
struct PcHessianBlocks {
  std::vector<double> mKu, nSu, mKv, nSv, mKuv, nSuv;
};

// The rotation terms' blocks, assembled once per class by pcRotationBlocks.
// nd == 0: everything empty, every rotation term zero.
struct PcRotationBlocks {
  int64_t nd = 0;
  std::vector<double> Zu, Zv;          // (m, m) C G_a^T H_0: the tangent's third term
  std::vector<double> Ju;              // (nd, m) G_u C: R1's row term
  std::vector<double> A1, B1, A2, B2;  // (m, m) C G_u^T H_v, C G_u^T G_v, C G_v^T H_u, C G_v^T G_u
};

// Row-major (r, k) x (k, c).
std::vector<double> pcMatMul(const std::vector<double>& a, const std::vector<double>& b,
                             int64_t r, int64_t k, int64_t c) {
  check(std::ssize(a) == r * k && std::ssize(b) == k * c, "pcMatMul: the shapes disagree");
  std::vector<double> out(narrowTo<std::size_t>(r * c), 0.0);
  for (int64_t i = 0; i < r; ++i)
    for (int64_t l = 0; l < k; ++l) {
      const double ail = a[narrowTo<std::size_t>(i * k + l)];
      if (ail == 0.0) continue;
      for (int64_t j = 0; j < c; ++j)
        out[narrowTo<std::size_t>(i * c + j)] += ail * b[narrowTo<std::size_t>(l * c + j)];
    }
  return out;
}

// Row-major (k, r)^T x (k, c) = (r, c).
std::vector<double> pcMatTMul(const std::vector<double>& a, const std::vector<double>& b,
                              int64_t k, int64_t r, int64_t c) {
  check(std::ssize(a) == k * r && std::ssize(b) == k * c, "pcMatTMul: the shapes disagree");
  std::vector<double> out(narrowTo<std::size_t>(r * c), 0.0);
  for (int64_t l = 0; l < k; ++l)
    for (int64_t i = 0; i < r; ++i) {
      const double ali = a[narrowTo<std::size_t>(l * r + i)];
      if (ali == 0.0) continue;
      for (int64_t j = 0; j < c; ++j)
        out[narrowTo<std::size_t>(i * c + j)] += ali * b[narrowTo<std::size_t>(l * c + j)];
    }
  return out;
}

// The blocks above from C = U^T U (m, m) and the (nd, m) G_u, H_u, G_v, H_v,
// H_0. `second` false: only Zu and Zv (the first derivatives' third term).
PcRotationBlocks pcRotationBlocks(int64_t m, int64_t nd, const std::vector<double>& C,
                                  const std::vector<double>& Gu, const std::vector<double>& Hu,
                                  const std::vector<double>& Gv, const std::vector<double>& Hv,
                                  const std::vector<double>& H0, bool second) {
  PcRotationBlocks rot;
  rot.nd = nd;
  if (nd == 0) return rot;
  const auto cOf = [&](const std::vector<double>& x) { return pcMatMul(C, x, m, m, m); };
  rot.Zu = cOf(pcMatTMul(Gu, H0, nd, m, m));
  rot.Zv = cOf(pcMatTMul(Gv, H0, nd, m, m));
  if (!second) return rot;
  rot.Ju = pcMatMul(Gu, C, nd, m, m);
  rot.A1 = cOf(pcMatTMul(Gu, Hv, nd, m, m));
  rot.B1 = cOf(pcMatTMul(Gu, Gv, nd, m, m));
  rot.A2 = cOf(pcMatTMul(Gv, Hu, nd, m, m));
  rot.B2 = cOf(pcMatTMul(Gv, Gu, nd, m, m));
  return rot;
}

// || P (S_ab - S_a S^+ S_b - S_b S^+ S_a) P ||_F (Sec. 6.2's L_ab), from
// Vn S_ab Vn^T (nd, nd) and G_a, G_b (nd, m): S_a S^+ S_b restricted is
// G_a G_b^T. Zero when the rank is constant through second order along
// (a, b); the caller divides by s_max and refuses above a tolerance.
double pcLiftNorm(int64_t nd, int64_t m, const std::vector<double>& pSabP,
                  const std::vector<double>& Ga, const std::vector<double>& Gb) {
  check(std::ssize(pSabP) == nd * nd && std::ssize(Ga) == nd * m && std::ssize(Gb) == nd * m,
        "pcLiftNorm: the shapes disagree");
  double s = 0.0;
  for (int64_t i = 0; i < nd; ++i)
    for (int64_t j = 0; j < nd; ++j) {
      double l = pSabP[narrowTo<std::size_t>(i * nd + j)];
      for (int64_t k = 0; k < m; ++k) {
        const double gai = Ga[narrowTo<std::size_t>(i * m + k)];
        const double gbi = Gb[narrowTo<std::size_t>(i * m + k)];
        l -= gai * Gb[narrowTo<std::size_t>(j * m + k)] +
             gbi * Ga[narrowTo<std::size_t>(j * m + k)];
      }
      s += l * l;
    }
  return std::sqrt(s);
}

// One tuple's rows: the four slab einsums' (against T, Tdot_u, Tdot_v,
// Tdot_uv; m long) and r_v = Vn b_v (nd long). `yuv` and `rv` are empty when
// only the first derivatives are wanted.
struct PcHessianRow {
  std::span<const double> y, yu, yv, yuv, rv;
};

// Q and its derivatives.
struct PcQuadDerivatives {
  double f = 0.0, fu = 0.0, fv = 0.0, fuv = 0.0;
};

// The four sums every class accumulates over its tuples.
struct PcHessianSums {
  double energy = 0.0, edotU = 0.0, edotV = 0.0, eddot = 0.0;
};

// One tuple's Delta tangents (#201), for a leg whose orbital energies move:
// A_u = K_u + Delta_u S + Delta S_u, so alpha_u gains Delta_u I (U^T S U = I);
// A_uv gains Delta_uv S + Delta_u S_v + Delta_v S_u, so beta gains
// Delta_uv I + Delta_u nS_v + Delta_v nS_u. On S's dropped space P S = 0, so
// the rotation blocks gain nothing. All zero (the default) is #152's frozen
// orbital energies. The mixed block H_kc runs an orbital leg v with Delta_v
// = Delta'_w per tuple and the CI leg's Delta_u = Delta_uv = 0
// (docs/hessian-theory.md, Sec. 6.4 and 6.5); the CI block's FULL mode (#214)
// runs both legs CI with all three, the orbital energies responding to c.
struct PcDeltaTangents {
  double u = 0.0, v = 0.0, uv = 0.0;
};

// Q(r), Q_u, Q_v and -- when `second` -- Q_uv for one tuple of Delta `delta`,
// above, with the tuple's Delta tangents `dd` (zero: frozen orbital
// energies). `minDen` is tracked exactly as pcCoefficients does. Every length
// comes from one solve, so a mismatch is our bug and aborts.
PcQuadDerivatives pcQuadSecond(const PcHessianRow& r, const std::vector<double>& lambda,
                               double delta, const PcHessianBlocks& blk,
                               const PcRotationBlocks& rot, bool second, double& minDen,
                               const PcDeltaTangents& dd = {}) {
  const std::size_t m = lambda.size();
  const std::size_t nd = narrowTo<std::size_t>(rot.nd);
  check(r.y.size() == m && r.yu.size() == m && r.yv.size() == m,
        "pcQuadSecond: a row is not the solve's kept-mode count long");
  check(!second || (r.yuv.size() == m && r.rv.size() == nd),
        "pcQuadSecond: the second derivative needs y_uv and r_v rows");
  std::vector<double> den(m);
  for (std::size_t k = 0; k < m; ++k) {
    den[k] = lambda[k] + delta;
    minDen = std::min(minDen, den[k]);
  }
  const auto dot = [](std::span<const double> a, std::span<const double> b) {
    double s = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) s += a[k] * b[k];
    return s;
  };
  // (X + delta Y) x, both (m, m); Y may be empty (then X x).
  const auto apply = [&](const std::vector<double>& x0, const std::vector<double>& y0,
                         std::span<const double> x) {
    std::vector<double> out(m, 0.0);
    for (std::size_t k = 0; k < m; ++k) {
      double s = 0.0;
      for (std::size_t l = 0; l < m; ++l)
        s += (x0[k * m + l] + (y0.empty() ? 0.0 : delta * y0[k * m + l])) * x[l];
      out[k] = s;
    }
    return out;
  };
  std::vector<double> g(m);
  for (std::size_t k = 0; k < m; ++k) g[k] = r.y[k] / den[k];
  std::vector<double> au = apply(blk.mKu, blk.nSu, g), av = apply(blk.mKv, blk.nSv, g);
  for (std::size_t k = 0; k < m; ++k) {
    au[k] += dd.u * g[k];
    av[k] += dd.v * g[k];
  }

  PcQuadDerivatives q;
  q.f = dot(g, r.y);
  q.fu = 2.0 * dot(g, r.yu) - dot(g, au);
  q.fv = 2.0 * dot(g, r.yv) - dot(g, av);
  if (nd > 0) {
    q.fu -= 2.0 * dot(g, apply(rot.Zu, {}, g));
    q.fv -= 2.0 * dot(g, apply(rot.Zv, {}, g));
  }
  if (!second) return q;

  // Sec. 6.1's terms.
  double cross = 0.0;
  for (std::size_t k = 0; k < m; ++k) cross += (r.yu[k] - au[k]) * (r.yv[k] - av[k]) / den[k];
  q.fuv = 2.0 * dot(g, r.yuv) - dot(g, apply(blk.mKuv, blk.nSuv, g)) + 2.0 * cross;
  // The Delta tangents' share of beta (nothing unless the orbital energies move).
  const auto sandwichOf = [&](const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    double s = 0.0;
    for (std::size_t k = 0; k < m; ++k)
      for (std::size_t l = 0; l < m; ++l) s += g[k] * x[k * m + l] * g[l];
    return s;
  };
  if (dd.uv != 0.0) q.fuv -= dd.uv * dot(g, g);
  if (dd.u != 0.0) q.fuv -= dd.u * sandwichOf(blk.nSv);
  if (dd.v != 0.0) q.fuv -= dd.v * sandwichOf(blk.nSu);
  if (nd == 0) return q;
  // The rotation, R1 + R2 + R3.
  std::vector<double> jg(nd, 0.0);
  for (std::size_t i = 0; i < nd; ++i)
    for (std::size_t k = 0; k < m; ++k) jg[i] += rot.Ju[i * m + k] * g[k];
  q.fuv += 2.0 * dot(jg, r.rv) - 2.0 * dot(g, apply(rot.A1, rot.B1, g)) -
           2.0 * dot(g, apply(rot.A2, rot.B2, g)) + 2.0 * dot(g, apply(rot.B2, {}, r.y));
  return q;
}

// One slab's rows for the single-coupling classes (Sr, Si, Srs, Sij, Sir),
// each (nt, m) -- `rv` (nt, nd) -- and `yuv` / `rv` null when only the first
// derivatives are wanted.
struct PcHessianSlab {
  const Tensor* y = nullptr;
  const Tensor* yu = nullptr;
  const Tensor* yv = nullptr;
  const Tensor* yuv = nullptr;
  const Tensor* rv = nullptr;
};

// Tuple t's rows of `s`, m (and nd) long.
PcHessianRow pcHessianRowAt(const PcHessianSlab& s, int64_t t, int64_t m, int64_t nd) {
  const auto at = [](const Tensor* x, int64_t off, int64_t len) {
    if (x == nullptr) return std::span<const double>();
    return std::span<const double>(x->data().data() + off, narrowTo<std::size_t>(len));
  };
  return {at(s.y, t * m, m), at(s.yu, t * m, m), at(s.yv, t * m, m), at(s.yuv, t * m, m),
          at(s.rv, t * nd, nd)};
}

// The single-coupling classes' four sums over one slab: E_t = -factor Q(t).
// `deltaDotV` (one per tuple, or empty for zeros) is the v leg's Delta
// tangent, PcDeltaTangents::v (#201's orbital leg); `deltaDotU` the u leg's,
// PcDeltaTangents::u (#202: both legs orbital); `deltaDotUV` the pair's
// second-order Delta'', PcDeltaTangents::uv (#214: both legs CI, the orbital
// energies responding to the CI vector).
void hessianSingle(const PcHessianSlab& s, const std::vector<double>& lambda,
                   const std::vector<double>& delta, const PcHessianBlocks& blk,
                   const PcRotationBlocks& rot, double factor, PcHessianSums& acc,
                   double& minDen, std::span<const double> deltaDotV = {},
                   std::span<const double> deltaDotU = {},
                   std::span<const double> deltaDotUV = {}) {
  const int64_t nt = std::ssize(delta), m = std::ssize(lambda);
  const bool second = s.yuv != nullptr;
  check(s.y->size() == nt * m && s.yu->size() == nt * m && s.yv->size() == nt * m &&
            (!second || (s.yuv->size() == nt * m && s.rv != nullptr &&
                         s.rv->size() == nt * rot.nd)),
        "hessianSingle: the slabs must be (tuples, m) and (tuples, nd)");
  check((deltaDotV.empty() || std::ssize(deltaDotV) == nt) &&
            (deltaDotU.empty() || std::ssize(deltaDotU) == nt) &&
            (deltaDotUV.empty() || std::ssize(deltaDotUV) == nt),
        "hessianSingle: the Delta tangents are one per tuple");
  for (int64_t t = 0; t < nt; ++t) {
    const std::size_t ut = narrowTo<std::size_t>(t);
    const PcDeltaTangents dd{deltaDotU.empty() ? 0.0 : deltaDotU[ut],
                             deltaDotV.empty() ? 0.0 : deltaDotV[ut],
                             deltaDotUV.empty() ? 0.0 : deltaDotUV[ut]};
    const PcQuadDerivatives d = pcQuadSecond(pcHessianRowAt(s, t, m, rot.nd), lambda,
                                             delta[narrowTo<std::size_t>(t)], blk, rot, second,
                                             minDen, dd);
    acc.energy -= factor * d.f;
    acc.edotU -= factor * d.fu;
    acc.edotV -= factor * d.fv;
    acc.eddot -= factor * d.fuv;
  }
}

// The two-coupling finish (Sijr, Srsi) for one tuple: E_t = -(2 F(a, a) -
// F(a, b)) with F(a, b) = [Q(a + b) - Q(a - b)] / 4, a the direct row and b
// the exchange row at the same Delta.
// `deltaDotV` / `deltaDotU` / `deltaDotUV`: the tuple's v- and u-leg Delta
// tangents and its Delta'', as hessianSingle's.
void hessianPair(const PcHessianRow& a, const PcHessianRow& b, const std::vector<double>& lambda,
                 double delta, const PcHessianBlocks& blk, const PcRotationBlocks& rot,
                 bool second, PcHessianSums& acc, double& minDen, double deltaDotV = 0.0,
                 double deltaDotU = 0.0, double deltaDotUV = 0.0) {
  const PcDeltaTangents dd{deltaDotU, deltaDotV, deltaDotUV};
  const auto combine = [](std::span<const double> x, std::span<const double> y, double sign) {
    std::vector<double> out(x.size());
    for (std::size_t k = 0; k < x.size(); ++k) out[k] = x[k] + sign * y[k];
    return out;
  };
  const auto rowOf = [&](double sign, std::array<std::vector<double>, 5>& store) {
    store = {combine(a.y, b.y, sign), combine(a.yu, b.yu, sign), combine(a.yv, b.yv, sign),
             combine(a.yuv, b.yuv, sign), combine(a.rv, b.rv, sign)};
    return PcHessianRow{store[0], store[1], store[2], store[3], store[4]};
  };
  std::array<std::vector<double>, 5> plusStore, minusStore;
  const PcQuadDerivatives aa = pcQuadSecond(a, lambda, delta, blk, rot, second, minDen, dd);
  const PcQuadDerivatives p = pcQuadSecond(rowOf(1.0, plusStore), lambda, delta, blk, rot,
                                           second, minDen, dd);
  const PcQuadDerivatives n = pcQuadSecond(rowOf(-1.0, minusStore), lambda, delta, blk, rot,
                                           second, minDen, dd);
  const auto pair = [](double qa, double qp, double qn) { return 2.0 * qa - 0.25 * (qp - qn); };
  acc.energy -= pair(aa.f, p.f, n.f);
  acc.edotU -= pair(aa.fu, p.fu, n.fu);
  acc.edotV -= pair(aa.fv, p.fv, n.fv);
  acc.eddot -= pair(aa.fuv, p.fuv, n.fuv);
}

// --- PC: the ADJOINT's per-tuple arithmetic (#74) -----------------------------
//
// Reverse mode of the same finish. Every PC class energy is a sum over its
// tuples of quadratic forms in the eigenbasis rows y_t = U^T b_t,
//
//   single (Sr, Si, Srs, Sij, Sir)  E_t = -f y_t^T D_t^-1 y_t
//   pair   (Sijr, Srsi)             E_t = -(2 y_t^T D_t^-1 y_t - y_t^T D_t^-1 x_t)
//
// with D_t = diag(lambda + Delta_t) and x_t the exchange row: the partner
// tuple's y (Sijr: the (j, i)-swapped row of the same tensor) or the SrsiT
// slab's row (Srsi: a second tensor, which IS the partner's y). With
// g = D^-1 y, everything the adjoint needs comes out of the rows alone:
//
//   ybar_t  = dE/dy_t         the slab einsum's adjoint input: what the
//                             external integrals' pseudodensity (dE/dslab)
//                             and dE/dT, dE/dt_1 are contracted from
//   r_t     = dE/dDelta_t     Park Eq. 38's R for PC: what dE/de_core /
//                             dE/de_virt are scattered from
//   mgg    += the (m, m) weight of the solve's Hamiltonian: dE through
//             dA = U^T dA U is <mgg, U^T dA U> (row-major, NOT symmetrised --
//             the caller takes sym), and mggDelta the same weighted by Delta_t
//             (A = K + Delta S, so S's share of it)
//
// derived in nevpt2.gradient:pc_classes' header: dE = sum_t ybar_t . dy_t +
// <mgg, U^T dA U> + ..., the solve differentiated, no eigenvector derivative.
// The forward side's per-tuple arithmetic is tangentSingle / pcQuadForm above
// and pcRotationTangent* below -- separate code, so a transcription error in
// one does not cancel in the duality.
struct PcSlabAdjoint {
  double energy = 0.0;        // this slab's share of the class energy
  std::vector<double> ybar;   // (nt * m): dE/dy, same layout as y
  std::vector<double> xbar;   // (nt * m): dE/dx for a two-tensor pair; empty otherwise
  std::vector<double> r;      // (nt): dE/dDelta_t
};

// One single-coupling slab: y (nt, m), factor f. `mgg` / `mggDelta` are (m, m)
// row-major accumulators the caller owns (zero-filled once per class);
// `minDen` is tracked as finishSingle does.
PcSlabAdjoint pcAdjointSingle(const Tensor& y, const std::vector<double>& lambda,
                              const std::vector<double>& delta, double factor,
                              std::vector<double>& mgg, std::vector<double>& mggDelta,
                              double& minDen) {
  const int64_t nt = std::ssize(delta), m = std::ssize(lambda);
  check(y.size() == nt * m && std::ssize(mgg) == m * m && std::ssize(mggDelta) == m * m,
        "pcAdjointSingle: y must be (tuples, m) and the weights (m, m)");
  PcSlabAdjoint out;
  out.ybar.assign(narrowTo<std::size_t>(nt * m), 0.0);
  out.r.assign(narrowTo<std::size_t>(nt), 0.0);
  std::vector<double> g;
  for (int64_t t = 0; t < nt; ++t) {
    const std::span<const double> row(y.data().data() + t * m, narrowTo<std::size_t>(m));
    const double dt = delta[narrowTo<std::size_t>(t)];
    pcCoefficients(row, lambda, dt, g, minDen);
    double gy = 0.0, gg = 0.0;
    for (int64_t k = 0; k < m; ++k) {
      const std::size_t uk = narrowTo<std::size_t>(k);
      gy += g[uk] * row[uk];
      gg += g[uk] * g[uk];
      out.ybar[narrowTo<std::size_t>(t * m + k)] = -2.0 * factor * g[uk];
    }
    out.energy -= factor * gy;
    out.r[narrowTo<std::size_t>(t)] = factor * gg;
    for (int64_t k = 0; k < m; ++k) {
      const double gk = factor * g[narrowTo<std::size_t>(k)];
      if (gk == 0.0) continue;
      for (int64_t l = 0; l < m; ++l) {
        const double w = gk * g[narrowTo<std::size_t>(l)];
        mgg[narrowTo<std::size_t>(k * m + l)] += w;
        mggDelta[narrowTo<std::size_t>(k * m + l)] += dt * w;
      }
    }
  }
  return out;
}

// One two-coupling slab, E_t = -(2 y_t D^-1 y_t - y_t D^-1 x_t). The exchange
// row x_t is either `x` (Srsi: a second (nt, m) tensor, whose adjoint comes
// back in `xbar`) or, with `x` null, row partner[t] of `y` itself (Sijr:
// partner swaps j and i, an involution within the slab) -- then x's adjoint is
// folded back onto ybar's partner row and `xbar` stays empty, so ybar is the
// derivative of the class energy as a function of y alone.
PcSlabAdjoint pcAdjointPair(const Tensor& y, const Tensor* x, std::span<const int64_t> partner,
                            const std::vector<double>& lambda, const std::vector<double>& delta,
                            std::vector<double>& mgg, std::vector<double>& mggDelta,
                            double& minDen) {
  const int64_t nt = std::ssize(delta), m = std::ssize(lambda);
  check(y.size() == nt * m && std::ssize(mgg) == m * m && std::ssize(mggDelta) == m * m,
        "pcAdjointPair: y must be (tuples, m) and the weights (m, m)");
  check(x != nullptr ? x->size() == nt * m : std::ssize(partner) == nt,
        "pcAdjointPair: the exchange rows are a second (tuples, m) tensor or a partner map");
  PcSlabAdjoint out;
  out.ybar.assign(narrowTo<std::size_t>(nt * m), 0.0);
  if (x != nullptr) out.xbar.assign(narrowTo<std::size_t>(nt * m), 0.0);
  out.r.assign(narrowTo<std::size_t>(nt), 0.0);
  std::vector<double> g, gx;
  for (int64_t t = 0; t < nt; ++t) {
    const int64_t pt = x != nullptr ? t : partner[narrowTo<std::size_t>(t)];
    check(pt >= 0 && pt < nt, "pcAdjointPair: a partner outside the slab");
    const std::span<const double> row(y.data().data() + t * m, narrowTo<std::size_t>(m));
    const std::span<const double> xrow((x != nullptr ? *x : y).data().data() + pt * m,
                                       narrowTo<std::size_t>(m));
    const double dt = delta[narrowTo<std::size_t>(t)];
    pcCoefficients(row, lambda, dt, g, minDen);
    pcCoefficients(xrow, lambda, dt, gx, minDen);
    double gy = 0.0, gxv = 0.0, gg = 0.0, ggx = 0.0;
    for (int64_t k = 0; k < m; ++k) {
      const std::size_t uk = narrowTo<std::size_t>(k);
      gy += g[uk] * row[uk];
      gxv += g[uk] * xrow[uk];
      gg += g[uk] * g[uk];
      ggx += g[uk] * gx[uk];
      out.ybar[narrowTo<std::size_t>(t * m + k)] += -(4.0 * g[uk] - gx[uk]);
      if (x != nullptr)
        out.xbar[narrowTo<std::size_t>(t * m + k)] = g[uk];
      else
        out.ybar[narrowTo<std::size_t>(pt * m + k)] += g[uk];
    }
    out.energy -= 2.0 * gy - gxv;
    out.r[narrowTo<std::size_t>(t)] = 2.0 * gg - ggx;
    for (int64_t k = 0; k < m; ++k) {
      const double gk = g[narrowTo<std::size_t>(k)];
      if (gk == 0.0) continue;
      for (int64_t l = 0; l < m; ++l) {
        const double w = gk * (2.0 * g[narrowTo<std::size_t>(l)] - gx[narrowTo<std::size_t>(l)]);
        mgg[narrowTo<std::size_t>(k * m + l)] += w;
        mggDelta[narrowTo<std::size_t>(k * m + l)] += dt * w;
      }
    }
  }
  return out;
}

// --- PC: the FORWARD tangent along an orbital rotation, per tuple (#74) -------
//
// Along a rotation kappa at fixed CI vector only the integrals move: the slab
// rows (T and t_1 fixed, the slab and the one-electron block dotted:
// ydot = T^T xdot + t1^T h1dot) and the Dyall block K (mK = U^T sym(Kdot) U);
// S and S_1 are RDM-only and do not move. These are the class finishes' tangents in that
// setting -- tangentSingle / pcDotSijr's arithmetic with nS = 0 and a
// factor -- taken from y and ydot directly, NOT from pcAdjoint*'s ybar, so the
// rotation duality compares two computations. The orbital energies' share
// (sum_t R_t Deltadot_t) is reported apart, as for SC (gradient.cppm, "The
// full source term Y_xy").

// sum_t [ -f (2 g_t . ydot_t - g_t^T mK g_t) ] over a single-coupling slab.
double pcRotationTangentSingle(const Tensor& y, const Tensor& ydot,
                               const std::vector<double>& lambda,
                               const std::vector<double>& delta, double factor,
                               const std::vector<double>& mK) {
  const int64_t nt = std::ssize(delta), m = std::ssize(lambda);
  check(y.size() == nt * m && ydot.size() == nt * m && std::ssize(mK) == m * m,
        "pcRotationTangentSingle: y, ydot must be (tuples, m) and mK (m, m)");
  const std::vector<double> zero(narrowTo<std::size_t>(m * m), 0.0);
  std::vector<double> g;
  double minDen = std::numeric_limits<double>::infinity(), edot = 0.0;
  for (int64_t t = 0; t < nt; ++t) {
    const std::span<const double> row(y.data().data() + t * m, narrowTo<std::size_t>(m));
    const double dt = delta[narrowTo<std::size_t>(t)];
    pcCoefficients(row, lambda, dt, g, minDen);
    double dotTerm = 0.0;
    for (int64_t k = 0; k < m; ++k) dotTerm += g[narrowTo<std::size_t>(k)] * ydot.flat(t * m + k);
    edot -= factor * (2.0 * dotTerm - pcQuadForm(g, g, mK, zero, dt));
  }
  return edot;
}

// The same for a two-coupling slab (pcAdjointPair's two layouts): `x` /
// `xdot` the exchange rows and their tangent, or both null with `partner`.
double pcRotationTangentPair(const Tensor& y, const Tensor* x, const Tensor& ydot,
                             const Tensor* xdot, std::span<const int64_t> partner,
                             const std::vector<double>& lambda, const std::vector<double>& delta,
                             const std::vector<double>& mK) {
  const int64_t nt = std::ssize(delta), m = std::ssize(lambda);
  check(y.size() == nt * m && ydot.size() == nt * m && std::ssize(mK) == m * m,
        "pcRotationTangentPair: y, ydot must be (tuples, m) and mK (m, m)");
  check((x != nullptr) == (xdot != nullptr) &&
            (x != nullptr ? x->size() == nt * m && xdot->size() == nt * m
                          : std::ssize(partner) == nt),
        "pcRotationTangentPair: the exchange rows are a tensor pair or a partner map");
  const std::vector<double> zero(narrowTo<std::size_t>(m * m), 0.0);
  std::vector<double> gu, gv;
  double minDen = std::numeric_limits<double>::infinity(), edot = 0.0;
  for (int64_t t = 0; t < nt; ++t) {
    const int64_t pt = x != nullptr ? t : partner[narrowTo<std::size_t>(t)];
    const Tensor& xs = x != nullptr ? *x : y;
    const Tensor& xds = xdot != nullptr ? *xdot : ydot;
    const std::span<const double> yu(y.data().data() + t * m, narrowTo<std::size_t>(m));
    const std::span<const double> yv(xs.data().data() + pt * m, narrowTo<std::size_t>(m));
    const double dt = delta[narrowTo<std::size_t>(t)];
    pcCoefficients(yu, lambda, dt, gu, minDen);
    pcCoefficients(yv, lambda, dt, gv, minDen);
    double duu = 0.0, duv = 0.0, dvu = 0.0;
    for (int64_t k = 0; k < m; ++k) {
      const std::size_t uk = narrowTo<std::size_t>(k);
      duu += gu[uk] * ydot.flat(t * m + k);
      duv += gu[uk] * xds.flat(pt * m + k);
      dvu += gv[uk] * ydot.flat(t * m + k);
    }
    const double qUU = pcQuadForm(gu, gu, mK, zero, dt);
    const double qUV = pcQuadForm(gu, gv, mK, zero, dt);
    edot -= 2.0 * (2.0 * duu - qUU) - (duv + dvu - qUV);
  }
  return edot;
}

// --- PC: the adjoint's third term and its tangent along a CI direction (#155) --
//
// Reverse mode of pcQuadSecond, in v. The PC CI source term y is the gradient
// of Edot(v) with respect to everything v enters -- the rows y_v = U^T b_v
// and r_v = Vn b_v, the (m, m) block alpha_v = mK_v + Delta nS_v and the
// (nd, m) blocks G_v = Vn S_v U, H_v = Vn K_v U (docs/hessian-theory.md,
// Sec. 6.2; pcQuadSecond's header has the notation) -- and H_cc u is the
// gradient, in the same objects, of Eddot(u, v)'s part that is bilinear in
// the two first-order sets (the part linear in the second-order set is
// <y's adjoint, X_uv>, which is the second sigma call's). Per quadratic form
// Q of weight w (E = sum_rows w Q):
//
//   THE THIRD TERM (y at any CI vector): Q_v carries -2 g^T C G_v^T H_0 g, so
//     dE/dG_v += -2 w (H_0 g)(C g)^T                               (gbar3)
//   pcAdjointSingle / pcAdjointPair carry the other two (ybar, mgg).
//
//   ALONG u, with e = D (y_u - alpha_u g), the solve of the tangent's
//   residual, and Cg = C g, Gc = G_u C g:
//     dE/dy_v     += 2 w e                                         (ybarDot)
//     dE/dalpha_v += -2 w e g^T                                    (mggDot; Delta-weighted:
//                                                                   mggDotDelta)
//     dE/dr_v     += 2 w Gc                                        (rbarDot)
//     dE/dH_v     += -2 w Gc g^T                                   (hbarDot)
//     dE/dG_v     += -2 w Delta Gc g^T - 2 w ((H_u + Delta G_u) g)(Cg)^T
//                    + 2 w (G_u y)(Cg)^T                           (gbarDot)
//
// read off Q_uv's 6.1 cross term 2 (y_u - alpha_u g)^T D (y_v - alpha_v g)
// and its rotation R1 + R2 + R3. The pair classes are polarised exactly as
// hessianPair does it -- rows a (weight -2), a + b (+1/4), a - b (-1/4) -- and
// each row's bars flow back onto a and b. The caller (nevpt2.gradient's
// :pc_classes) takes these through T_v = S_v U, N_v = S_v Vn^T, t1_v, n1_v
// and U / Vn sandwiches onto dE/dS, dE/dK, dE/dS_1, then the class's own
// sweep. Separate code from pcQuadSecond, so the duality <v, H_cc u> ==
// Eddot(u, v) against it compares two computations (test/energy_finish's
// PcAdjointSecondTests holds the two to each other on the host).

// One class's blocks: `C` = U^T U (m, m) and `H0` = Vn sym(K) U (nd, m) for
// the third term (both empty when nd = 0); with `direction`, the tangent
// set's mK_u / nS_u (m, m) and G_u / H_u (nd, m; empty when nd = 0).
struct PcAdjointBlocks {
  int64_t m = 0, nd = 0;
  std::vector<double> C, H0;
  bool direction = false;
  std::vector<double> mKu, nSu, Gu, Hu;
};

// The per-class accumulators, (re)sized by pcAdjointSecondZeros.
struct PcAdjointSecond {
  std::vector<double> gbar3;                // (nd, m)
  std::vector<double> mggDot, mggDotDelta;  // (m, m), row-major, not symmetrised
  std::vector<double> gbarDot, hbarDot;     // (nd, m)
  double edotU = 0.0;  // the three-term Edot(u) over the rows seen: a free check
};

PcAdjointSecond pcAdjointSecondZeros(const PcAdjointBlocks& b) {
  const std::size_t mm = narrowTo<std::size_t>(b.m * b.m), nm = narrowTo<std::size_t>(b.nd * b.m);
  PcAdjointSecond a;
  a.gbar3.assign(nm, 0.0);
  if (b.direction) {
    a.mggDot.assign(mm, 0.0);
    a.mggDotDelta.assign(mm, 0.0);
    a.gbarDot.assign(nm, 0.0);
    a.hbarDot.assign(nm, 0.0);
  }
  return a;
}

// One slab's bars on its rows: target 0's (`ybarDot` (nt, m), `rbarDot`
// (nt, nd)) and, for a two-tensor pair, target 1's (`xbarDot`, `rxbarDot`);
// and `rDot` (nt), the tangent along the direction of R_t = dE/dDelta_t
// (pcAdjoint*'s `r`), which an orbital leg's Delta'_t pairs with (#202). All
// empty without a direction.
struct PcSlabAdjointSecond {
  std::vector<double> ybarDot, xbarDot, rbarDot, rxbarDot, rDot;
};

// One quadratic form of weight `w`: its row y (m) and, with a direction, y_u
// (m). The bars on this row are ADDED to `ybarDot` (m) / `rbarDot` (nd).
// `deltaDotDir` is the direction's Delta tangent for this tuple (#201: an
// orbital direction along which the orbital energies move): alpha_u gains
// Delta_u I and beta Delta_u nS_v (PcDeltaTangents), whose bar on nS_v,
// -w Delta_u g g^T, joins the Delta-weighted ones (mggDotDelta).
// `rDot` non-null: R = dE/dDelta = -w g^T (U^T S U) g differentiated along the
// direction, -w (2 g . e' + g^T nS_u g) with e' = D^-1 (y_u - alpha_u g), is
// ADDED to it (#202: an orbital leg's Delta'_t pairs with it).
void pcAdjointSecondRow(std::span<const double> y, std::span<const double> yu,
                        const std::vector<double>& lambda, double delta, double w,
                        const PcAdjointBlocks& b, PcAdjointSecond& acc, std::span<double> ybarDot,
                        std::span<double> rbarDot, double deltaDotDir = 0.0,
                        double* rDot = nullptr) {
  const std::size_t m = narrowTo<std::size_t>(b.m), nd = narrowTo<std::size_t>(b.nd);
  check(y.size() == m && lambda.size() == m, "pcAdjointSecondRow: a row is not m long");
  check(!b.direction || (yu.size() == m && ybarDot.size() == m && rbarDot.size() == nd),
        "pcAdjointSecondRow: the direction's row or bars are the wrong length");
  std::vector<double> g(m), cg(m, 0.0), h0g(nd, 0.0);
  for (std::size_t k = 0; k < m; ++k) g[k] = y[k] / (lambda[k] + delta);
  if (nd > 0) {
    for (std::size_t k = 0; k < m; ++k)
      for (std::size_t l = 0; l < m; ++l) cg[k] += b.C[k * m + l] * g[l];
    for (std::size_t i = 0; i < nd; ++i)
      for (std::size_t l = 0; l < m; ++l) h0g[i] += b.H0[i * m + l] * g[l];
    for (std::size_t i = 0; i < nd; ++i)
      for (std::size_t k = 0; k < m; ++k) acc.gbar3[i * m + k] += -2.0 * w * h0g[i] * cg[k];
  }
  if (!b.direction) return;

  std::vector<double> au(m, 0.0), e(m);
  double gyu = 0.0, gau = 0.0;
  for (std::size_t k = 0; k < m; ++k) {
    for (std::size_t l = 0; l < m; ++l) au[k] += (b.mKu[k * m + l] + delta * b.nSu[k * m + l]) * g[l];
    au[k] += deltaDotDir * g[k];
    e[k] = (yu[k] - au[k]) / (lambda[k] + delta);
    gyu += g[k] * yu[k];
    gau += g[k] * au[k];
    ybarDot[k] += 2.0 * w * e[k];
  }
  for (std::size_t k = 0; k < m; ++k)
    for (std::size_t l = 0; l < m; ++l) {
      const double c = -2.0 * w * e[k] * g[l];
      acc.mggDot[k * m + l] += c;
      acc.mggDotDelta[k * m + l] += delta * c - w * deltaDotDir * g[k] * g[l];
    }
  if (rDot != nullptr) {
    double ge = 0.0, gsg = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
      ge += g[k] * e[k];
      for (std::size_t l = 0; l < m; ++l) gsg += g[k] * b.nSu[k * m + l] * g[l];
    }
    *rDot += -w * (2.0 * ge + gsg);
  }
  double third = 0.0;
  if (nd > 0) {
    std::vector<double> gc(nd, 0.0), z(nd, 0.0), guy(nd, 0.0);
    for (std::size_t i = 0; i < nd; ++i)
      for (std::size_t k = 0; k < m; ++k) {
        const double gu = b.Gu[i * m + k];
        gc[i] += gu * cg[k];
        z[i] += (b.Hu[i * m + k] + delta * gu) * g[k];
        guy[i] += gu * y[k];
      }
    for (std::size_t i = 0; i < nd; ++i) {
      third += gc[i] * h0g[i];
      rbarDot[i] += 2.0 * w * gc[i];
      for (std::size_t k = 0; k < m; ++k) {
        acc.hbarDot[i * m + k] += -2.0 * w * gc[i] * g[k];
        acc.gbarDot[i * m + k] += -2.0 * w * delta * gc[i] * g[k] - 2.0 * w * z[i] * cg[k] +
                                  2.0 * w * guy[i] * cg[k];
      }
    }
  }
  acc.edotU += w * (2.0 * gyu - gau - 2.0 * third);
}

// The single-coupling classes (Sr, Si, Srs, Sij, Sir), E_t = -factor Q(t):
// `y` (nt, m) and, with a direction, `yu` (nt, m), else null.
// `deltaDotDir`: the direction's Delta tangent per tuple, or empty for zeros
// (pcAdjointSecondRow's).
PcSlabAdjointSecond pcAdjointSecondSingle(const Tensor& y, const Tensor* yu,
                                          const std::vector<double>& lambda,
                                          const std::vector<double>& delta, double factor,
                                          const PcAdjointBlocks& b, PcAdjointSecond& acc,
                                          std::span<const double> deltaDotDir = {}) {
  const int64_t nt = std::ssize(delta), m = b.m, nd = b.nd;
  check(y.size() == nt * m && (!b.direction || (yu != nullptr && yu->size() == nt * m)),
        "pcAdjointSecondSingle: the rows must be (tuples, m)");
  check(deltaDotDir.empty() || std::ssize(deltaDotDir) == nt,
        "pcAdjointSecondSingle: the Delta tangents are one per tuple");
  PcSlabAdjointSecond out;
  if (b.direction) {
    out.ybarDot.assign(narrowTo<std::size_t>(nt * m), 0.0);
    out.rbarDot.assign(narrowTo<std::size_t>(nt * nd), 0.0);
    out.rDot.assign(narrowTo<std::size_t>(nt), 0.0);
  }
  const auto rowOf = [&](const Tensor* x, int64_t t) {
    if (x == nullptr || !b.direction) return std::span<const double>();
    return std::span<const double>(x->data().data() + t * m, narrowTo<std::size_t>(m));
  };
  for (int64_t t = 0; t < nt; ++t) {
    std::span<double> yb, rb;
    if (b.direction) {
      yb = std::span<double>(out.ybarDot.data() + t * m, narrowTo<std::size_t>(m));
      rb = std::span<double>(out.rbarDot.data() + t * nd, narrowTo<std::size_t>(nd));
    }
    pcAdjointSecondRow(std::span<const double>(y.data().data() + t * m, narrowTo<std::size_t>(m)),
                       rowOf(yu, t), lambda, delta[narrowTo<std::size_t>(t)], -factor, b, acc, yb,
                       rb, deltaDotDir.empty() ? 0.0 : deltaDotDir[narrowTo<std::size_t>(t)],
                       b.direction ? &out.rDot[narrowTo<std::size_t>(t)] : nullptr);
  }
  return out;
}

// The two-coupling classes (Sijr, Srsi), pcAdjointPair's two layouts: the
// exchange row is `x`'s (with `xu` along the direction) or, with `x` null,
// row partner[t] of `y` itself, its bars folded back onto that row.
PcSlabAdjointSecond pcAdjointSecondPair(const Tensor& y, const Tensor* x, const Tensor* yu,
                                        const Tensor* xu, std::span<const int64_t> partner,
                                        const std::vector<double>& lambda,
                                        const std::vector<double>& delta, const PcAdjointBlocks& b,
                                        PcAdjointSecond& acc,
                                        std::span<const double> deltaDotDir = {}) {
  const int64_t nt = std::ssize(delta), m = b.m, nd = b.nd;
  check(y.size() == nt * m && (x != nullptr ? x->size() == nt * m : std::ssize(partner) == nt),
        "pcAdjointSecondPair: the exchange rows are a second (tuples, m) tensor or a partner map");
  check(!b.direction || (yu != nullptr && yu->size() == nt * m &&
                         (x == nullptr || (xu != nullptr && xu->size() == nt * m))),
        "pcAdjointSecondPair: the direction's rows must be shaped like the rows");
  check(deltaDotDir.empty() || std::ssize(deltaDotDir) == nt,
        "pcAdjointSecondPair: the Delta tangents are one per tuple");
  PcSlabAdjointSecond out;
  if (b.direction) {
    out.ybarDot.assign(narrowTo<std::size_t>(nt * m), 0.0);
    out.rbarDot.assign(narrowTo<std::size_t>(nt * nd), 0.0);
    out.rDot.assign(narrowTo<std::size_t>(nt), 0.0);
    if (x != nullptr) {
      out.xbarDot.assign(narrowTo<std::size_t>(nt * m), 0.0);
      out.rxbarDot.assign(narrowTo<std::size_t>(nt * nd), 0.0);
    }
  }
  const std::size_t um = narrowTo<std::size_t>(m), und = narrowTo<std::size_t>(nd);
  std::vector<double> row(um), rowU(um), ybs(um), rbs(und);
  for (int64_t t = 0; t < nt; ++t) {
    const int64_t pt = x != nullptr ? t : partner[narrowTo<std::size_t>(t)];
    check(pt >= 0 && pt < nt, "pcAdjointSecondPair: a partner outside the slab");
    const double* ya = y.data().data() + t * m;
    const double* yb = (x != nullptr ? *x : y).data().data() + pt * m;
    const double* ua = b.direction ? yu->data().data() + t * m : nullptr;
    const double* ub = b.direction ? (x != nullptr ? *xu : *yu).data().data() + pt * m : nullptr;
    double* barA = b.direction ? out.ybarDot.data() + t * m : nullptr;
    double* rA = b.direction ? out.rbarDot.data() + t * nd : nullptr;
    double* barB = !b.direction ? nullptr
                   : x != nullptr ? out.xbarDot.data() + t * m
                                  : out.ybarDot.data() + pt * m;
    double* rB = !b.direction ? nullptr
                 : x != nullptr ? out.rxbarDot.data() + t * nd
                                : out.rbarDot.data() + pt * nd;
    // (sign of b in the row, weight): a, a + b, a - b.
    for (const auto [sb, w] : {std::pair{0.0, -2.0}, std::pair{1.0, 0.25}, std::pair{-1.0, -0.25}}) {
      for (std::size_t k = 0; k < um; ++k) {
        row[k] = ya[k] + sb * yb[k];
        if (b.direction) rowU[k] = ua[k] + sb * ub[k];
      }
      std::fill(ybs.begin(), ybs.end(), 0.0);
      std::fill(rbs.begin(), rbs.end(), 0.0);
      pcAdjointSecondRow(row, b.direction ? std::span<const double>(rowU) : std::span<const double>(),
                         lambda, delta[narrowTo<std::size_t>(t)], w, b, acc,
                         b.direction ? std::span<double>(ybs) : std::span<double>(),
                         b.direction ? std::span<double>(rbs) : std::span<double>(),
                         deltaDotDir.empty() ? 0.0 : deltaDotDir[narrowTo<std::size_t>(t)],
                         b.direction ? &out.rDot[narrowTo<std::size_t>(t)] : nullptr);
      if (!b.direction) continue;
      for (std::size_t k = 0; k < um; ++k) {
        barA[k] += ybs[k];
        barB[k] += sb * ybs[k];
      }
      for (std::size_t i = 0; i < und; ++i) {
        rA[i] += rbs[i];
        rB[i] += sb * rbs[i];
      }
    }
  }
  return out;
}

}  // namespace nevpt2
