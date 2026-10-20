// nevpt2.pc_tangent -- the finite-difference CHECK of PC-NEVPT2's CI-vector
// tangent: it drives nevpt2.energy's pcTangentsDevice at one state and
// nevpt2.energy's pcEnergiesDevice at the displaced states c +- h u, and
// compares. A leaf, like nevpt2.gradient beside it: nothing in src/ imports
// it and neither demo links it; apps/pc_tangent/ is what drives it.
//
// WHAT IS BEING CHECKED. pcTangentsDevice differentiates the PC solve
// analytically -- the derivation, and why a pseudo-inverse may be
// differentiated at all here, are in nevpt2.energy's module header ("PC'S
// TANGENT DIFFERENTIATES THE SOLVE") and at `tangentClass` in
// energy_pc_solve.cppm, resting on the gap PC_MIN_GAP enforces
// (docs/pc-nevpt2.md, "The singular-metric convention"). Nothing about that
// derivation is checked by an identity the way the SC pseudodensities are
// (nevpt2.gradient, Park Eq. 40): the only independent handle on d/dh of a
// SOLVE is to displace the state and re-solve. So this module does both:
//
//   1. EULER, exact, on the classes it applies to. Where a class's S, K and
//      S_1 are LINEAR in the RDMs, the class energy is homogeneous of degree
//      2 in the CI vector -- S, K, S_1 and b are degree 2 and A^+ is degree
//      -2 -- so the tangent along the state ITSELF is exactly twice the
//      energy: Edot(c, c) = 2 E. The RDM tangents along c are 2x the RDMs
//      (Euler again, as --rdm-tangent checks on the device build), so this
//      costs one extra tangent stage and no RDM build, and it tests the whole
//      chain -- the matrix tangents, U^T sym(Xdot) U, the per-tuple finish --
//      with no step size anywhere.
//
//      It does NOT apply to every class, and kPcHomogeneousInCi below says
//      which: the hole-side builders make_hdm1 / make_hdm2 carry RDM-FREE
//      Kronecker-delta terms (2 delta; the two delta-delta terms), which are
//      the hole RDMs only for a state of unit norm and do not scale with the
//      CI vector, so Sijr and Sij are affine rather than linear in the RDMs
//      and their energies are not homogeneous. Sijrs contracts no RDM at all,
//      so it is degree 0. Those same constants are why the finite difference
//      below walks the unit sphere (pc_tangent.cpp, `pcTangentStep`) and why
//      a matrix tangent is one SUBTRACTION of two builds rather than one
//      build from the tangent RDMs (energy_pc_tangent.cppm,
//      `tangentMatrices`).
//
//   2. A CENTRAL FINITE DIFFERENCE, Richardson-verified. Along the straight
//      line c + h u, (E(c + h u) - E(c - h u)) / 2h must approach Edot(c, u)
//      as O(h^2). Unlike the RDMs themselves -- quadratic in the CI vector,
//      so their straight-line central difference has NO truncation error
//      (test/rdm_host_dm/) -- the PC energy is NOT a polynomial in c: there
//      is a solve, A^+ is a ratio, and the h^2 term survives. So the
//      Richardson table is the evidence: the error must fall ~4x per
//      halving of h, and the extrapolation (4 fd(h) - fd(2h)) / 3 must land
//      on the analytic tangent.
//
// Each displaced state needs ITS OWN dm1/dm2 (nevpt2.rdm_host_dm rebuilds
// them from the CI vector -- the golden file's arrays are only the state's
// own) and its own dm3/f3ac/f3ca (a full nevpt2.rdm_build pass), which is why
// the FD is a check and not a method: 2 builds per step. The displaced state
// is NOT a CASCI eigenstate, and nothing here needs it to be -- the function
// differentiated is "the PC energy of the RDMs of this CI vector", which is
// exactly the object phase 2's source terms need.
//
// A class REFUSED by the gap or denominator check is reported as refused and
// fails the gate; it is never given a zero derivative. A refusal at a
// displaced state (where the metric is a different matrix) is a Numerical
// Error naming the step -- the finite difference would otherwise be formed
// from a number that does not exist.
export module nevpt2.pc_tangent;

import std;
// Re-exported: the entry point takes a NevptIntegralsDevice and the
// DeviceResources, and hands back nevpt2.energy_finish's PcClassTangent.
export import nevpt2.energy;
// Re-exported: it takes the RdmBuildOptions the caller parsed and runs the
// build and its tangent itself.
export import nevpt2.rdm_build;
// Re-exported: the host dm1/dm2 rebuild and their tangent, one per displaced
// state.
export import nevpt2.rdm_host_dm;
// Re-exported: longestFourfoldRun, the step-ladder rule the report gates on.
export import nevpt2.gradient_plan;

export namespace nevpt2 {

// Which classes' energies are homogeneous of degree 2 in the CI vector, so
// that Edot(c, c) = 2 E holds exactly (CLASSES order). A class qualifies when
// every one of its S, K and S_1 is LINEAR in the RDMs:
//
//   Sr    yes  S = dm3, K = a16                       -- linear
//   Si    yes  S = dm3_h = 2 dm2 delta - dm3, K = a22 -- linear (the delta
//              multiplies an RDM, so it scales with the state)
//   Sijrs NO   contracts no RDM at all: degree 0, and its tangent is 0
//   Sijr  NO   S = hdm1 = 2 delta - dm1: the 2 delta is RDM-FREE, so affine
//   Srsi  yes  S = dm1, K = k27                       -- linear
//   Srs   yes  S = rm2 = dm2 - dm1 delta, K = a7      -- linear
//   Sij   NO   S = hdm2 carries two delta-delta terms: affine
//   Sir   yes  the dm1 / dm2 blocks and a12 / a13     -- linear
inline constexpr std::array<bool, 8> kPcHomogeneousInCi{true,  true, false, false,
                                                        true,  true, false, true};

// The fewest consecutive fourfold halvings the finite difference must show
// (nevpt2.gradient_plan, longestFourfoldRun), on top of every ratio being 4:
// a RUN, never one step (docs/testing.md, "Residual-independent checks of the
// gradient path"). Two, so three steps at least -- what the slowest entry,
// nevpt2_pc_tangent_cas1012, runs; a check with fewer has measured no order
// of convergence at all and fails.
inline constexpr int kPcMinFourfoldRun = 2;

// One step of the central difference.
struct PcFdStep {
  double h = 0.0;
  std::array<double, 8> fdClass{};  // (E_+ - E_-) / 2h, per class, CLASSES order
  double fdTotal = 0.0;             // and of the total
  double err = 0.0;                 // |fdTotal - the analytic Edot|
  // err of the previous (larger) step divided by this one's: 4 for a
  // second-order difference, 0 for the first step, which has no predecessor.
  double ratio = 0.0;
  // The PC conditioning guard's record at the two displaced states c + h u
  // and c - h u: the smallest S gap over the classes that have a spectrum, as
  // the solve that produced fdClass measured it. 0 means the guard was never
  // run there -- and a displaced state is exactly where the metric's rank can
  // move (#41), so the report FAILS on it rather than trusting the energy.
  std::array<double, 2> displacedGap{};
};

struct PcTangentCheck {
  // The analytic tangent and the energies it differentiates, from one call to
  // pcTangentsDevice (so a refused class is refused here).
  std::array<PcClassTangent, 8> classes{};
  // pcEnergiesDevice's own class energies for the same state: the tangent
  // path builds each class's S and K itself, so this is what says the two
  // transcriptions agree.
  std::array<double, 8> energyRef{};
  double total = 0.0;     // sum of the tangent path's class energies
  double totalRef = 0.0;  // sum of pcEnergiesDevice's
  double edotTotal = 0.0;
  // Euler: the tangent along the state itself, per class. For the classes
  // kPcHomogeneousInCi marks, this must be exactly twice that class's energy;
  // for the other three it is reported and not gated on.
  std::array<double, 8> eulerEdot{};
  bool allDone = false;  // every class computed (none refused, none skipped)
  std::vector<PcFdStep> steps;
  // Richardson over the two finest steps: (4 fd(h) - fd(2h)) / 3, whose own
  // error is O(h^4), and |it - edotTotal|.
  double richardson = 0.0;
  double richardsonErr = 0.0;
  double seconds = 0.0;
};

// Runs the whole check on one state. `ci` is the CI vector (the golden's own),
// `h2e` the physicist-ordered active block the RDM build reads, `ints` the
// uploaded integral blocks every class contracts, and `steps` the h values to
// difference at -- at least two, each half the previous one if the Richardson
// ratios are to mean anything. The direction u is drawn from `seed`,
// projected against `ci` and normalized, so the check is deterministic.
//
// Every device allocation comes from res's pool and every launch, copy and
// BLAS call is on res.stream(); each step's RDM build and PC stage are scoped
// so the peak is one state's, not every state's at once.
//
// An out-of-range active space, a CI vector whose length is not the
// determinant count, fewer than two steps, or a refusal at a displaced state
// are returned Errors for main() to report once.
Result<PcTangentCheck> pcTangentFdCheck(const RdmBuildOptions& rdmOpt,
                                        const NevptIntegralsDevice& ints, const Tensor& ci,
                                        int64_t norb, int64_t nelecA, int64_t nelecB,
                                        const Tensor& h2e, std::span<const double> steps,
                                        std::uint64_t seed, const DeviceResources& res);

// Prints the per-class table (energy against pcEnergiesDevice's, the tangent,
// each class's own finite difference, the Euler identity where it applies),
// the Richardson table over `steps`, and the gate line the
// `nevpt2_pc_tangent_*` ctest entries match: "PASS: the PC-NEVPT2 tangent
// matches the central finite difference ..." or "FAIL: ...". Returns true --
// and prints PASS -- only when all eight classes were computed, each energy
// is within `idtol` of pcEnergiesDevice's, Euler's identity holds to `idtol`
// relative on every class kPcHomogeneousInCi marks, every Richardson ratio is
// within `ratioTol` of 4 AND they make a run of at least kPcMinFourfoldRun,
// every displaced state carries the conditioning guard's record of a gap of
// at least PC_MIN_GAP, and the extrapolated difference is within `atol` of
// the analytic tangent.
bool printPcTangentReport(const PcTangentCheck& c, double atol, double idtol, double ratioTol);

// --- the second derivative, Eddot(u, v) (#152; pc_hessian.cpp) -----------------
//
// The check of nevpt2.energy's pcHessianDevice, at the golden's FIXED orbital
// energies, along two directions built exactly as the tangent's
// (`pcTangentDirection`: spin-free generators over the state's symmetry
// mask, projected against the state and normalized) -- directions along
// which S's dropped space TURNS but does not lift (docs/pc-nevpt2.md, "What
// the cut drops, and how it moves"), so the rotation terms of
// docs/hessian-theory.md, Sec. 6.2 are exercised:
//
//   1. THE TRANSCRIPTION. pcHessianDevice's own class energies against
//      pcEnergiesDevice's, and its three-term Edot(u) / Edot(v) against
//      pcTangentsDevice's two-term one at the same RDM sets -- equal at the
//      eigenstate, to its convergence (the third term carries P K).
//   2. EULER AT SECOND ORDER, exact (Sec. 4.3): a class homogeneous of degree
//      2 has H c = grad E, so Eddot(c, u) == Edot(u) on the five classes
//      kPcHomogeneousInCi marks. It costs no RDM build: the first-order set
//      along c, and the (c, c) set, are 2 dm, and the second-order set at
//      (c, u) is the tangent along u itself. It holds only at fixed orbital
//      energies, which is the only mode there is.
//   3. ONE CENTRAL DIFFERENCE OF THE THREE-TERM Edot, Richardson-verified:
//      pcHessianDevice's first-order Edot(v) -- exact at ANY CI vector --
//      at c(+-h) = (c +- h u) / ||c +- h u||, the tangent check's own
//      unit-sphere path, so (Edot_v(c(h)) - Edot_v(c(-h))) / 2h -> Eddot(u, v)
//      as O(h^2). Not pcTangentsDevice's two-term Edot: off the eigenstate it
//      lacks -2 rho^T S_v P A x, and a difference of it converges to
//      something else (Sec. 6.2). The path's second derivative does not
//      enter: d/dh of Edot_v at h = 0 is v^T H c'(0) = v^T H u. The same
//      direction and path constraints as the tangent's (pc_tangent.cpp,
//      `pcTangentDirection` / `pcTangentStep`). Each step is two displaced
//      states, each a plain and a tangent RDM build.
//   4. A NEGATIVE CONTROL: the pair (u, w) with w a random vector of
//      amplitudes, which breaks the state's symmetry, so S's dropped modes
//      lift and the class must be REFUSED (PC_LIFT_TOL), never answered.
//
// A class refused by the gap or denominator check -- or for a lift along
// (u, v) -- is REFUSED here too, and fails the gate; a refusal at a displaced
// state is a Numerical Error naming the step, as the tangent's.
struct PcHessianCheck {
  // pcHessianDevice at (u, v): the energies, both first derivatives and the
  // second, per class.
  std::array<PcClassHessian, 8> classes{};
  std::array<double, 8> energyRef{};  // pcEnergiesDevice's class energies
  std::array<double, 8> edotURef{};   // pcTangentsDevice's Edot(u)
  std::array<double, 8> edotVRef{};   // pcTangentsDevice's Edot(v)
  // pcHessianDevice at (c, u): Eddot(c, u), per class, which must be
  // edotURef on the homogeneous classes.
  std::array<double, 8> eulerEddot{};
  // pcHessianDevice at (u, w), the negative control.
  std::array<PcClassHessian, 8> control{};
  double eddotTotal = 0.0;  // sum of classes[].eddot
  bool allDone = false;     // every class computed (none refused)
  // The difference of Edot(v) along the path; fdClass / fdTotal are its
  // quotients, err |fdTotal - eddotTotal|.
  std::vector<PcFdStep> steps;
  double richardson = 0.0;  // (4 fd(h) - fd(2h)) / 3 over the two finest steps
  double richardsonErr = 0.0;
  double seconds = 0.0;
};

// Runs the check on one state: as pcTangentFdCheck, with a second seed for v
// (`seedU` draws u, the direction of the difference; `seedV` draws v, the
// direction of the differenced tangent) and a third for the control's w.
Result<PcHessianCheck> pcHessianFdCheck(const RdmBuildOptions& rdmOpt,
                                        const NevptIntegralsDevice& ints, const Tensor& ci,
                                        int64_t norb, int64_t nelecA, int64_t nelecB,
                                        const Tensor& h2e, std::span<const double> steps,
                                        std::uint64_t seedU, std::uint64_t seedV,
                                        std::uint64_t seedControl, const DeviceResources& res);

// Prints the per-class table (with each class's smallest kept S eigenvalue,
// gap and lift), the control and the Richardson table, and the gate line
// the `nevpt2_pc_hessian_*` ctest entries match: "PASS: the PC-NEVPT2
// Hessian Eddot(u, v) matches Euler and the central difference of Edot ..."
// or "FAIL: ...", naming the orbital-energy mode and the smallest kept s and
// gap. Returns true -- and prints PASS -- only when all eight classes were
// computed; each energy is within `idtol` of pcEnergiesDevice's and each
// Edot within `edotTol` (relative, floor 1) of pcTangentsDevice's; Euler
// holds to `idtol` relative on every homogeneous class; the control refused
// at least one class for its lift and none otherwise; every ratio is within
// `ratioTol` of 4 and they make a run of at least kPcMinFourfoldRun; every
// displaced state carries a gap of at least PC_MIN_GAP; and the
// extrapolation is within `atol` of Eddot(u, v).
bool printPcHessianReport(const PcHessianCheck& c, double atol, double idtol, double edotTol,
                          double ratioTol);

}  // namespace nevpt2
