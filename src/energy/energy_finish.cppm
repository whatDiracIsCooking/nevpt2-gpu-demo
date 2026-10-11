// nevpt2.energy_finish -- the class energies' pure host arithmetic: the slab
// walk every class runs (forBatches), the strongly-contracted finish
// (normToEnergy, PySCF's _norm_to_energy) and ITS DERIVATIVE
// (normToEnergyDerivative, normToEnergyTangent -- the only calculus the
// forward tangent of an SC class energy needs) and the partially-contracted
// finish (finishSingle, checkDenominator), with the PC result types they fill. No
// device, no BLAS, no solver: nevpt2.energy re-exports this module, and its
// partitions (:sc_finish, :pc_solve, the class files) do the device work and
// call these. Exported so the unit tier can check them (test/energy_finish/).
// Carved out of the :shared, :sc_finish and :pc_solve partitions and
// energy.cppm unchanged.
export module nevpt2.energy_finish;

import std;
// Re-exported: finishSingle reads a host Tensor, and every extent here is an
// int64_t (nevpt2.tensor re-exports nevpt2.common).
export import nevpt2.tensor;
import nevpt2.error_handling;  // check(): normToEnergyTangent's five lengths

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
// :sc_tangent partition does exactly that). The only part of the forward
// derivative that is actual calculus is the scalar finish above, and it is
// here: no device, no tensor, one perturber at a time.

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

// --- PC ------------------------------------------------------------------------

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

}  // namespace nevpt2
