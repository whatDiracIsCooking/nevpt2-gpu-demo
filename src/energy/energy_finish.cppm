// nevpt2.energy_finish -- the class energies' pure host arithmetic: the slab
// walk every class runs (forBatches), the strongly-contracted finish
// (normToEnergy, PySCF's _norm_to_energy) and the partially-contracted one
// (finishSingle, checkDenominator), with the PC result types they fill. No
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
