// nevpt2.gradient -- the SC-NEVPT2 gradient's pseudodensities, assembled on
// the device over the SAME IntegralSource slab walk the class energies run
// (nevpt2.energy). The first half of the analytical gradient of Park, J. Chem.
// Theory Comput. 2019 (arXiv:1907.10180): the amplitudes and multipliers
// (nevpt2.sc_amplitudes, re-exported) and, from them, the density-like
// quantities that multiply every molecular integral the second-order energy
// reads -- Eqs. 41-47 -- and, one chain rule further in, the CI SOURCE TERM
// y_I = dE/dc_I of Eq. 48 (scCiSourceDevice, at the bottom). Nothing else in
// the tree imports this; apps/sc_pseudodensity/ drives the pseudodensity
// identities and the active-rotation source term (--kappa-act) and
// nevpt2_demo's --gradient-source the CI source term. The orbital source
// term's core and virtual blocks are later work.
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
//
// AND TWO SOURCE TERMS COME OUT OF THE SAME WALK. The ORBITAL one's
// active-active block is "The active-rotation source term" below
// (scActiveRotationDevice); the CI one, one chain rule further in, is "The CI
// source term, Park Eq. 48" (scCiSourceDevice) at the bottom.
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
// Re-exported: scCiSourceDevice's SigmaVectorOptions, and the three sigma
// entry points the CI source term IS, once the pseudodensities are swept down
// to the RDMs (Park Eq. 48).
export import nevpt2.sigma_vector;

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

  // --- the active-rotation source term, one entry per requested direction ---
  // (empty when none was). See "The active-rotation source term" below.
  std::vector<double> rotExt{};      // sum_blocks <D, Xdot>: Eq. 41's half
  std::vector<double> rotAct{};      // sum_G <M_G, Gdot>:    Eqs. 42-43's half
  std::vector<double> rotTangent{};  // the forward tangent, P.Ndot + Q.Hdot
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

// --- The active-rotation source term ----------------------------------------
//
// Park Eq. 26's orbital source term Y_xy = dE/dkappa_xy, RESTRICTED TO THE
// ACTIVE-ACTIVE BLOCK -- and in both directions, which is the point:
//
//   the ADJOINT   <kappa, Y_act> = sum_blocks <D, Xdot> + sum_G <M_G, Gdot>,
//                 the pseudodensities above paired with the direction;
//   the TANGENT   Edot(kappa), the same class terms rerun under the product
//                 rule and finished with Eqs. 36-37's multipliers.
//
// They are one derivative taken two ways, so holding them equal (to 1e-7; it
// comes out near machine precision) is what checks the assembly. The third
// number, a central finite difference of energiesDevice itself along
// exp(t kappa), is apps/sc_pseudodensity/'s -- it is what pins the device
// kappa action to nevpt2.kappa_rotation's exponential map, which the two
// directions here SHARE and so cannot check against each other.
//
// WHAT MOVES, EXACTLY -- the scope of this entry point, and the reason it is
// the active-active block alone. An active rotation at fixed CI coefficients
// moves every stored array that carries an active index: the active-space
// integrals h1e and h2e, and every external integral block's active slots.
// Those are all here. Held FIXED are
//
//   * the RDMs dm1/dm2/dm3 -- correct: they are functions of the CI vector,
//     which this derivative holds fixed;
//   * the f3ac/f3ca digests -- NOT a property of the exact derivative, but
//     of this functional: the digests are h2e contracted with the 4-RDM, so
//     the exact dE/dkappa carries a term through them, and recovering it in
//     reverse mode needs that contraction's TRANSPOSE (the 4-RDM against a
//     6-index conjugate density), a kernel this tree does not have. Both
//     directions here therefore differentiate the energy stage's own
//     functional, with the digests among its fixed inputs;
//   * the orbital energies e_core/e_virt -- likewise this functional's, and
//     in the exact derivative they move too, through the active density
//     inside the generalized Fock matrix: that needs the (xa|cc) and (xa|vv)
//     integrals the golden does not carry (docs/gradient-theory.md, Sec. 1.3,
//     which is explicit that "the active block needs only what the golden
//     already carries" is the easy mistake here).
//
// So this is the INTEGRAL HALF of the chain rule, dE/dkappa at fixed RDMs,
// fixed digests and fixed orbital energies -- exact for that functional, in
// both modes, and the half that the core/virtual and mixed blocks' work
// builds on. docs/testing.md, "The active-rotation source term", records what
// is measured and names the two gaps again.
//
// THE CHAIN RULE THROUGH THE DYALL BLOCKS IS WHAT THIS ADDS to the
// pseudodensities above, and it needs no new term list: every make_* of
// nevpt2.energy is linear in the integrals it reads, so a block's tangent is
// that same builder called with the dotted integrals (the :rotation
// partition says it again, with the f3 = 0 argument that isolates the h1e/h2e
// part). Eqs. 44-47's one-electron and two-electron active-space
// pseudodensities -- the Srs-only, make_a7-shaped objects above -- are NOT on
// this path and are not completed into Park's Eq. 35 Gamma-conjugate form
// here: that form is a re-expression of the same numbers, with no consumer
// until the full Y (core and virtual included) needs the RDM side too.
struct ScRotationRequest {
  Tensor kappa;          // (n_act, n_act), antisymmetric (nevpt2.kappa_rotation)
  bool tangent = false;  // also take the forward tangent along it
};

// One direction's answer, summed over the eight classes.
struct ScRotationPairing {
  double ext = 0.0;         // sum_blocks <D, Xdot>
  double act = 0.0;         // sum_G <M_G, Gdot>
  double adjoint = 0.0;     // <kappa, Y_act> = ext + act
  double tangent = 0.0;     // Edot(kappa), 0 when it was not asked for
  bool hasTangent = false;
};

struct ScRotationResult {
  ScGradientResult grad;                   // the pseudodensity identities, as above
  std::vector<ScRotationPairing> dirs{};   // one per ScRotationRequest, in order
};

// The pseudodensity assembly of scPseudodensityDevice (which is this with no
// directions), plus the source term paired against each `dirs` entry. Same
// inputs, same slab walk, same stream and pool; a malformed kappa is the
// CALLER's input and comes back as an Error, reported once at main.
//
// Everything is accumulated per class in CLASSES order, so a caller can read
// the per-class breakdown off `grad.classes[c].rotExt/rotAct/rotTangent`
// (parallel to `dirs`). Sijrs contributes nothing to any of it: the MP2-like
// class reads no array with an active index at all.
Result<ScRotationResult> scActiveRotationDevice(const ActiveIntegralsDevice& active,
                                                IntegralSource& source, int64_t batch,
                                                const DeviceTensor& dm1, const DeviceTensor& dm2,
                                                const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                                const DeviceTensor& f3ca,
                                                const std::vector<ScRotationRequest>& dirs,
                                                const DeviceResources& res);

// The integral-direct entry point: full blocks, one slab each.
Result<ScRotationResult> scActiveRotationDevice(const NevptIntegralsDevice& ints,
                                                const DeviceTensor& dm1, const DeviceTensor& dm2,
                                                const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                                const DeviceTensor& f3ca,
                                                const std::vector<ScRotationRequest>& dirs,
                                                const DeviceResources& res);

// Y_act as a MATRIX: the antisymmetric (n_act, n_act) whose pairing with any
// antisymmetric kappa is scActiveRotationDevice's `adjoint`. Assembled by
// probing the same linear functional with the n_act (n_act - 1) / 2 basis
// directions E_pq - E_qp, `chunk` of them per slab walk (the pseudodensities
// are rebuilt once per chunk, so a bigger chunk is faster and holds more).
// Y_pq for p < q is half the pairing with E_pq - E_qp, so that
// sum_pq kappa_pq Y_pq is the pairing with kappa; Y_qp = -Y_pq, diagonal zero.
//
// There is no cheaper assembly available: the gauge invariance of the class
// energies under a covariant rotation of integrals AND RDMs together makes
// the directly-contractible part (<D, X> and <M_G, G> with one active index
// left open) exactly SYMMETRIC, hence invisible to an antisymmetric kappa --
// all of Y_act's content sits in the chain rule through the Dyall blocks,
// which is a map into the integrals, not out of them. Probing it is what a
// caller wanting the matrix pays; a caller wanting a pairing pays nothing.
Result<Tensor> scActiveRotationMatrix(const NevptIntegralsDevice& ints, const DeviceTensor& dm1,
                                      const DeviceTensor& dm2, const DeviceTensor& dm3,
                                      const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                      int64_t chunk, const DeviceResources& res);

// Prints the per-class table -- the amplitude energy against `energies`'
// (nevpt2.energy's own answer for the same state) and against the golden
// `ref`, and both identity residuals -- then the gate line the
// `nevpt2_sc_pseudodensity_*` ctest entries match: "PASS: SC pseudodensity
// identities hold ..." or "FAIL: ...". Returns true -- and prints PASS --
// only when, for all eight classes, the amplitude energy is within `atol` of
// `energies`' and of `ref`'s, and both identity residuals are within `idtol`.
bool printScGradientReport(const ScGradientResult& g, const EnergyResult& energies,
                           const Tensor& ref, double atol, double idtol);

// Prints the per-direction table of the active-rotation source term -- the
// adjoint pairing, its two halves, the forward tangent and the difference --
// and returns whether every direction that carries a tangent agrees to
// `atol`. The PASS/FAIL line is the caller's (apps/sc_pseudodensity/'s
// --kappa-act gate also weighs its finite difference), so this prints the
// table and the verdict, not a gate line.
bool printScRotationTable(const ScRotationResult& r, double atol);

// --- the CI source term, Park Eq. 48 ------------------------------------------
//
// y_I = dE/dc_I = <I| (O_D + O_D^dagger) |Psi>, the whole second-order
// energy's derivative with respect to the CI vector at fixed orbitals and
// fixed amplitudes -- the right-hand side of docs/gradient-theory.md's
// (decoupled) CI Z-vector equation, and the one derivative of E the
// Lagrangian needs in determinant space.
//
// HOW IT IS BUILT, in three steps, each already its own checked piece:
//
//   1. the same slab walk scPseudodensityDevice runs, which leaves one
//      accumulator M_G = dE/dG per metric / Dyall-Hamiltonian block G
//      (Eqs. 42-43);
//   2. ONE CHAIN RULE FURTHER IN (nevpt2.gradient:rdm_adjoint): every block
//      is linear in the RDMs, so sweeping each block's own einsum chain
//      backwards -- reverse-topologically, since a9 reads hdm3 reads hdm2 and
//      a7 reads rm3 reads rm2 -- turns the M_G into the five coefficient
//      tensors dE/d(dm1), dE/d(dm2), dE/d(dm3), dE/d(f3ac), dE/d(f3ca);
//   3. those contracted with the RDMs' own CI derivatives
//      (nevpt2.gradient:ci_source), which is what the sigma-vector kernels
//      compute without the n^6 * ndet or n^8 * ndet derivative ever
//      existing: nevpt2.sigma_vector's 3-body sigma for dm3 (dm1 and dm2 fold
//      onto its diagonals, so they need no sigma of their own) and the 4-body
//      contracted sigma, forward and dagger, for each f3 order.
//
// AND IT IS CHECKED AGAINST FORWARD MODE, exactly. nevpt2.energy's
// energiesTangentDevice computes the same derivative the other way round, so
// for ANY direction u in determinant space
//
//   <u, y> == Edot(u)                         to round-off, not to a step
//
// -- two different computations of one number, which is what
// nevpt2_demo's --gradient-source gates on (and, at CAS(4,4), column by
// column over the whole Jacobian; docs/testing.md, "The CI source term's
// forward-reverse duality").
struct ScCiSourceResult {
  // y_I over the determinant space, in the CI vector's own
  // (nstringsA, nstringsB) shape.
  Tensor y;
  // The same walk's Eq. 40-47 numbers: free, since the sweep runs on top of
  // exactly that walk, and printScGradientReport reports them unchanged.
  ScGradientResult pseudodensities;
};

// Assembles y_I from the same inputs as scPseudodensityDevice plus the state
// itself: `ci` is the (nstringsA, nstringsB) CI vector the RDMs were built
// from, and `nelecA`/`nelecB` its active space's electron counts. `sigma`
// tiles the sigma kernels' determinant axis -- a launch-granularity knob, not
// a memory one (nevpt2.sigma_vector's SigmaVectorOptions says why).
//
// Everything runs on res.stream() and every device allocation is a
// DeviceBuffer from res's pool; the five coefficient tensors live from the
// first class to the last sigma call and are freed when this returns.
ScCiSourceResult scCiSourceDevice(const ActiveIntegralsDevice& active, IntegralSource& source,
                                  int64_t batch, const Tensor& ci, int64_t nelecA,
                                  int64_t nelecB, const DeviceTensor& dm1,
                                  const DeviceTensor& dm2, const DeviceTensor& dm3,
                                  const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                  const SigmaVectorOptions& sigma, const DeviceResources& res);

// The integral-direct entry point: full blocks, one slab each.
ScCiSourceResult scCiSourceDevice(const NevptIntegralsDevice& ints, const Tensor& ci,
                                  int64_t nelecA, int64_t nelecB, const DeviceTensor& dm1,
                                  const DeviceTensor& dm2, const DeviceTensor& dm3,
                                  const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                  const SigmaVectorOptions& sigma, const DeviceResources& res);

}  // namespace nevpt2
