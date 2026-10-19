// nevpt2.energy_finish -- the class energies' pure host arithmetic: the slab
// walk every class runs (forBatches), the strongly-contracted finish
// (normToEnergy, PySCF's _norm_to_energy) and ITS DERIVATIVE
// (normToEnergyDerivative, normToEnergyTangent -- the only calculus the
// forward tangent of an SC class energy needs) and its SECOND derivative
// (normToEnergyHessian, the 3x3 in the d form, and the multiplier tangents
// applyFinishHessian / normToEnergyMultiplierTangent -- #150,
// docs/hessian-theory.md Sec. 2), the partially-contracted
// finish (pcRankCut, finishSingle, checkDenominator) and the PC TANGENT's per-tuple
// arithmetic (pcCoefficients, pcQuadForm, tangentSingle), with the PC result
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

}  // namespace nevpt2
