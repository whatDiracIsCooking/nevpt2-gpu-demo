// nevpt2.gradient -- the SC-NEVPT2 gradient's pseudodensities, assembled on
// the device over the SAME IntegralSource slab walk the class energies run
// (nevpt2.energy). The first half of the analytical gradient of Park, J. Chem.
// Theory Comput. 2019 (arXiv:1907.10180): the amplitudes and multipliers
// (nevpt2.sc_amplitudes, re-exported) and, from them, the density-like
// quantities that multiply every molecular integral the second-order energy
// reads -- Eqs. 41-47. Nothing else in the tree imports this, and neither
// demo links it; apps/sc_pseudodensity/ drives it, and the Z-vector source
// terms that will consume it are later work.
//
// WHAT A PSEUDODENSITY IS HERE. A strongly-contracted class energy is a sum
// over external-orbital tuples t of
//
//   E_t = -N_t / (Delta_t + H_t / N_t),
//
// where N_t and H_t are QUADRATIC FORMS in that tuple's vector of external
// integrals x_t -- each a sum of einsum terms contracting x_t twice against
// one active-space block (a metric on the norm side, a Dyall-Hamiltonian
// block on the H side; nevpt2.energy's make_* intermediates). So the
// derivative of E with respect to an integral is
//
//   dE / dx^P = sum over terms, over the term's two x slots, of
//               (multiplier_t) * (that term with the slot dropped),
//
// the multiplier being P_t (Eq. 36) for a norm term and Q_t (Eq. 37) for an H
// one. That object is Park Eq. 41's two-particle pseudodensity D_{ar,bs}:
// collect everything that multiplies the integral. Dropping the ACTIVE
// operand instead of an external one gives Eqs. 42-43's intermediates
// M^(-2) / E^(-2) -- the integrals contracted with the multiplier, external
// indices summed away, which is what the active-space pseudodensities d^eff
// (Eqs. 44-45) and D_{ec,ad} (Eqs. 46-47) are then built from.
//
// SO ONE ROUTINE DOES ALL OF IT, from the class einsums themselves. Each
// class is written here as the LIST of its norm and H terms -- the same
// subscript strings, coefficients and operands as nevpt2.energy's
// energy_<class>, so the two can be read side by side -- and
// nevpt2.gradient:terms both evaluates that list (giving the per-tuple N_t
// and H_t the multipliers need) and differentiates it, by rewriting each
// term's subscripts per operand (nevpt2.sc_amplitudes' derivativeSubscripts).
// No derivative is hand-transcribed, and the one place a class needs more
// than that -- Sir's diagonal integral read, Sijrs having no active space at
// all, Sijr's and Srsi's symmetrized perturbers -- says so where it happens.
//
// AND IT IS CHECKED EXACTLY, not by finite difference. N_t and H_t are
// homogeneous of degree 2 in x_t, so contracting the assembled
// pseudodensities back against their own integrals must give
//
//   (1/2) sum_blocks <x, D>                  == E_class      (Eq. 40)
//
// and, since they are degree 1 in the active blocks,
//
//   sum_G <G, M_G>                           == sum_t (P_t N_t + Q_t H_t)
//
// which is E_class again for every class but Sir (whose norm has one term
// with no active block at all, so that term is excluded from the second
// identity's reference and `actReference` carries it). Both are algebraic
// identities in exact arithmetic -- a mis-transcribed index does not satisfy
// them -- and the ctest entries hold them to 1e-10 (docs/testing.md, "The SC
// pseudodensity identities").
export module nevpt2.gradient;

import std;
// Re-exported: the entry points take an ActiveIntegralsDevice / IntegralSource
// and the DeviceTensors of the RDM build, and the report prints against an
// EnergyResult. nevpt2.energy re-exports nevpt2.einsum, .device_tensor, .wwr
// and nevpt2.energy_finish in turn.
export import nevpt2.energy;
// Re-exported: ScMultipliers and the amplitudes are this module's own first
// deliverable, and its host-only half.
export import nevpt2.sc_amplitudes;

export namespace nevpt2 {

// One SC class's assembled pseudodensities, as the numbers that check them.
// The pseudodensities themselves are per-slab (Eq. 41's D, which carries an
// external index) or per-class (Eqs. 42-47's, which do not), and each is
// contracted back against its own operands and freed as the walk reaches its
// end: nothing consumes them yet, and holding all eight classes' at once
// would raise the peak for no reader. The Z-vector source terms that will
// read them are later work, and the place they plug in is where these
// contractions are taken.
struct ScClassGradient {
  double norm = 0.0;    // sum_t N_t -- nevpt2.energy's class norm
  double energy = 0.0;  // Eq. 15: sum_t T_t N_t -- its class energy
  // Eq. 40's external half: (1/2) sum over the class's integral blocks of
  // <x, D>. Equals `energy` to rounding.
  double extContraction = 0.0;
  // Eqs. 42-43 contracted back: sum over the class's active blocks G of
  // <G, M_G>. Equals `actReference`.
  double actContraction = 0.0;
  // What actContraction must be: sum_t (P_t N_t + Q_t H_t) over the terms
  // that carry an active block (all of them except Sir's purely-external
  // one-electron term), so `energy` when the class has no such term.
  double actReference = 0.0;
  // false for Sijrs alone: the MP2-like class contracts no RDM, so it has no
  // active blocks and no M / E to check.
  bool hasActive = false;

  // Eqs. 44-47, one derivative further in and assembled for Srs alone --
  // Park's own worked subspace. Its Dyall block K is itself linear in the
  // active integrals and in the hole RDMs it contracts them against, so
  // differentiating K against E gives the one- and two-electron active-space
  // pseudodensities (d^eff and its two-particle partner) and the
  // RDM-conjugate ones. Each family is degree 1, so either contracted back
  // against its own operands must give <K, E>.
  bool hasActiveIntegrals = false;
  double intReference = 0.0;    // <K, E>
  double intContraction = 0.0;  // <h1e, d^eff> + <h2e, its two-electron partner>
  double rdmContraction = 0.0;  // <rm2, D_rm2> + <rm3, D_rm3>
};

struct ScGradientResult {
  std::array<ScClassGradient, 8> classes{};  // nevpt2.energy's CLASSES order
  double total = 0.0;                        // sum of the eight energies
};

// Assembles every SC class's pseudodensities from the same inputs as
// energiesDevice, over slabs of `source` bounded by `batch` (0: one slab per
// block), and contracts each back against its own operands. Everything runs
// on res.stream() and every device allocation is a DeviceBuffer from res's
// pool, freed when its slab or class returns; every read-back is a
// downloadTensor, which synchronizes that stream, so all work is complete
// when this returns.
//
// `dm3`/`f3ac`/`f3ca` are the RDM build's own device-resident output and
// `dm1`/`dm2` the caller's upload, exactly as for energiesDevice -- the
// metric and Dyall-Hamiltonian blocks are built with nevpt2.energy's own
// make_* functions, not a copy of them.
ScGradientResult scPseudodensityDevice(const ActiveIntegralsDevice& active,
                                       IntegralSource& source, int64_t batch,
                                       const DeviceTensor& dm1, const DeviceTensor& dm2,
                                       const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                       const DeviceTensor& f3ca, const DeviceResources& res);

// The integral-direct entry point: full blocks, one slab each.
ScGradientResult scPseudodensityDevice(const NevptIntegralsDevice& ints, const DeviceTensor& dm1,
                                       const DeviceTensor& dm2, const DeviceTensor& dm3,
                                       const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                       const DeviceResources& res);

// Prints the per-class table -- the amplitude energy against `energies`'
// (nevpt2.energy's own answer for the same state) and against the golden
// `ref`, and both identity residuals -- then the gate line the
// `nevpt2_sc_pseudodensity_*` ctest entries match: "PASS: SC pseudodensity
// identities hold ..." or "FAIL: ...". Returns true -- and prints PASS --
// only when, for all eight classes, the amplitude energy is within `atol` of
// `energies`' and of `ref`'s, and both identity residuals are within `idtol`.
bool printScGradientReport(const ScGradientResult& g, const EnergyResult& energies,
                           const Tensor& ref, double atol, double idtol);

}  // namespace nevpt2
