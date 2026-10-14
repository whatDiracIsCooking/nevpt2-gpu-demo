// nevpt2.energy_finish -- the class energies' pure host arithmetic: the slab
// walk every class runs (forBatches), the strongly-contracted finish
// (normToEnergy, PySCF's _norm_to_energy) and ITS DERIVATIVE
// (normToEnergyDerivative, normToEnergyTangent -- the only calculus the
// forward tangent of an SC class energy needs), the partially-contracted
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

}  // namespace nevpt2
