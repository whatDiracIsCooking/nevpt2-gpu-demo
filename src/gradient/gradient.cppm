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
// Re-exported: the promotion table, the per-class Delta vectors and slab
// multipliers, the term classification and the CI source term's host
// folding -- the device assembly's pure host arithmetic, its own module so
// the unit tier reaches it card-free. Every internal partition reads it.
export import nevpt2.gradient_plan;
// Re-exported: scCiSourceDevice's SigmaVectorOptions, and the three sigma
// entry points the CI source term IS, once the pseudodensities are swept down
// to the RDMs (Park Eq. 48).
export import nevpt2.sigma_vector;
// Re-exported: scDigestRotationTangentDevice takes the build's RdmBuildOptions
// and runs buildRdmsDevice itself -- the digests' tangent along a rotation IS
// that build at the dotted integrals (#88).
export import nevpt2.rdm_build;

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
  // Eq. 39's orbital-energy half, R_t . Deltadot_t -- zero unless the
  // direction came with the gradient sidecar (scFullRotationDevice), and
  // shared by both modes: see "The full source term Y_xy" below.
  std::vector<double> rotEps{};

  // --- the orbital energies' CI response (#87) ---
  // dE/de over the core and the virtual orbitals, this class's share: Park
  // Eq. 38's R_t scattered onto the orbital energies its Delta_t reads
  // (nevpt2.gradient_plan's deltaAdjointInto). Accumulated only when
  // `epsAdjoint` is set -- by scCiSourceDevice, when it is handed the
  // orbital energies' response -- and left empty otherwise.
  bool epsAdjoint = false;
  std::vector<double> dEdCore{};
  std::vector<double> dEdVirt{};

  // --- the walk's tangent along a CI direction u (#156) ---
  // sum_t (P_t Ndot_t + Q_t Hdot_t): this class's Edot(u), formed from the
  // walk's own tangent blocks while it accumulates the M_G tangents
  // (scCiHessianVectorDevice). Zero when the walk carried no CI direction.
  double ciTangent = 0.0;

  // --- the mixed block H_kc along an orbital w and a CI u (#184) ---
  // Forward (scMixedHessianDevice): this class's Eddot_kc(w, u) in its five
  // terms, and the walk's own Edot(u). Both orientations: the walk's own
  // Edot(w), P.Ndot + Q.Hdot + R.Ddot with the digests' share merged per
  // tuple -- which must be the first-order source term's (the frozen-digest
  // tangent plus #88's digest term). Reverse (scMixedHessianVectorDevice),
  // with the orbital energies responding to the CI vector: dE/de's tangent
  // along w, Rdot scattered as dEdCore / dEdVirt are R. Zero / empty when the
  // walk carried no mixed direction.
  ScMixedTerms mixed{};
  double mixedEdotW = 0.0;
  double mixedEdotU = 0.0;
  std::vector<double> dEdCoreDot{};
  std::vector<double> dEdVirtDot{};
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
//     the exact dE/dkappa carries a term through them, which reverse mode
//     recovers through that contraction's TRANSPOSE (the 4-RDM against a
//     6-index conjugate density; scDigestAdjointDevice, #89) -- read by the
//     FULL entry points below, not by this one. Both directions here
//     differentiate the energy stage's own functional, with the digests
//     among its fixed inputs;
//   * the orbital energies e_core/e_virt -- likewise this functional's, and
//     in the exact derivative they move too, through the active density
//     inside the generalized Fock matrix: that needs the (xa|cc) and (xa|vv)
//     integrals the golden does not carry (docs/gradient-theory.md, Sec. 1.3,
//     which is explicit that "the active block needs only what the golden
//     already carries" is the easy mistake here). The gradient SIDECAR does
//     carry them, and scFullRotationDevice below reads them -- so this entry
//     point's functional is kept as it was, which is what makes the two
//     comparable (see "The full source term Y_xy").
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

// --- The full source term Y_xy ----------------------------------------------
//
// Park Eq. 26's orbital source term over the WHOLE `kappa` space -- the
// core-core, virtual-virtual and mixed blocks on top of the active-active one
// above -- assembled from the same pseudodensities and the same slab walk,
// reading the gradient sidecar's one-general-index MO-integral blocks and its
// full-MO-range `heff` / `f` (docs/gradient-theory.md, Sec. 3.3 and Sec. 3.5;
// docs/reference-data.md, "The gradient sidecar"). Both directions again,
// and the adjoint is reported in the pieces it is assembled from:
//
//   <kappa, Y> = [ sum_blocks <D, Xdot> + sum_G <M_G, Gdot> ]   the INTEGRAL half
//              + [ sum_t R_t Deltadot_t ]                       the ORBITAL ENERGIES
//
// -- Eq. 28's `h d`, `h^eff d^eff`, `g(...) d` terms in the first bracket and
// its `f d^Fock` + `g(d^Fock) d^(0)` terms (through Eq. 39's diagonal
// pseudodensity, which is what R_t is) in the second.
//
// WHAT IS NEW HERE, against the active block's entry points above:
//
//   * every array the energy reads now moves on EVERY axis, not only its
//     active ones -- one promotion rule per slot, generated from the block's
//     space signature out of the sidecar's arrays (gradient_general.cppm);
//   * the one-electron blocks' core dressing responds (they are h^eff
//     slices, and the core density is fixed in the MO indices);
//   * THE ORBITAL ENERGIES RESPOND, which closes one of the two gaps the
//     active-block work left open: `e_core` / `e_virt` are the generalized
//     Fock's diagonals, and `deps_p` is computable from the sidecar's `f`
//     and the restricted integral rows (gradient_general.cppm's header has
//     the formula). Since `Delta_t` is LINEAR in those vectors, its tangent
//     is each class's own `delta` construction evaluated at `deps` -- the
//     same trick that makes a Dyall block's tangent its own `make_*` at the
//     dotted integrals.
//
// AND THE DIGESTS, the last dependence the integral half holds fixed: the
// `f3ac`/`f3ca` digests' own `h2e` dependence. It lives on the blocks `h2e`
// moves under -- the active-active and the two mixed blocks with an active
// index -- and is EXACTLY ABSENT from the core-core and virtual-virtual
// blocks (no core or virtual index enters `h2e`), which is why
// `scPseudocanonicalMultipliers` below divides an exact numerator. Forward,
// it is one digest rebuild per direction (the digests are linear in `h2e`, so
// `buildRdmsDevice` at the dotted `h2e` IS their tangent) --
// scDigestRotationTangentDevice below (#88). Reverse, it is that
// contraction's transpose, run once (scDigestAdjointDevice, #89): handed in
// as `digestAdjoint`, it makes `adjoint` the EXACT <kappa, Y> at fixed CI.
// It is kept out of the integral halves, so their duality stays the
// like-for-like one it always was, and the FULL duality -- `adjoint` against
// the frozen-digest tangent plus the forward digests' term -- is a second,
// equally exact one (docs/testing.md, "The full orbital source term" and
// "The digests' orbital response, reverse mode").
//
// THE INTEGRAL HALF IS BIT-FOR-BIT THE ACTIVE BLOCK'S when `kappa` is
// active-restricted, by construction and not by luck: the intra-active part
// of every slot's sum is `scActiveRotationDevice`'s own einsum on the stored
// block, and the general part is the x-outside-its-own-space remainder, which
// an active-restricted kappa makes an exactly-zero tensor. Adding zero to a
// double changes no bit. apps/sc_pseudodensity/:kappa_full_check asserts it.

// The gradient sidecar, device-resident: the seventeen one-general-index
// integral blocks of docs/gradient-theory.md, Sec. 3.3 (as eighteen arrays --
// `(xc|cc)` is stored on its two independent restricted slices) and the
// full-MO-range one-electron and Fock matrices of Sec. 3.5 that the source
// term reads. Names and layouts are the sidecar's own
// (`generate_golden.py`'s `_gradient_arrays`, `loadGradGold`):
// `g_xP_QR[x, p, q, r] = (x p | q r)` in chemists' order, with `x` over the
// full MO range and the rest over the spaces the name spells.
//
// `mo_coeff`, `f_h` and `fock_rhf` are in the sidecar and NOT here: they are
// the AO contraction's and the Z-vector equations' (Sec. 2), not the source
// term's.
struct GeneralIntegralsDevice {
  int64_t ncore = 0, nact = 0, nvirt = 0;
  int64_t nmo() const { return ncore + nact + nvirt; }

  DeviceTensor h;     // (nmo, nmo) the bare one-electron matrix
  DeviceTensor heff;  // (nmo, nmo) h + the core dressing: the one-electron blocks' parent
  DeviceTensor f;     // (nmo, nmo) the generalized Fock, whose diagonal is e_core/e_virt

  DeviceTensor g_xa_aa, g_xv_aa, g_xc_aa;
  DeviceTensor g_xa_va, g_xv_va, g_xc_va;
  DeviceTensor g_xa_ca, g_xv_ca, g_xc_ca;
  DeviceTensor g_xa_cv, g_xv_cv, g_xc_cv;
  DeviceTensor g_xa_cc, g_xv_cc;
  DeviceTensor g_xc_cc_j, g_xc_cc_k;  // (x i | j j) and (x j | j i)
  DeviceTensor g_xa_vv, g_xc_vv;      // (x t | r r) and (x i | r r)
};

// --- the density-fitted inventory (#165) -------------------------------------
//
// The SAME struct for a density-fitted case, BUILT on the device instead of
// read: a `*_df` golden carries four blocks of the fitted three-index tensor
// (B_aa, B_ca, B_va, B_cv) and its NEVPT2F1 sidecar the two it lacks (B_cc,
// B_vv), which together are the whole of B[L, p, q] over the MO range
// (docs/gradient-theory.md, Sec. 3.8). The aux index is inert under an
// orbital rotation (the metric is absorbed into B and never sees kappa), so
// Sec. 3.1's promotion rule acts on one B factor at a time, and every
// one-general-index array is
//
//   (x p | q r) = sum_L B[L, x, p] B[L, q, r]
//
// -- the promoted STRIP B[:, x, P] (x over the whole MO range) against the
// block B_QR, one contraction over L each, with the three tied rows reading a
// diagonal of B_cc or B_vv. So everything downstream -- the promotion table,
// the dressing and orbital-energy rules, the duality, the checks -- runs
// unchanged on a fitted case, and what is new is only where the arrays come
// from. The arrays are formed on the device at the conventional inventory's
// sizes (Sec. 3.3), which a validation case affords; the file read stays the
// three-index one, and the slab-wise contraction that would never form them
// is not in tree (docs/gradients.md, Sec. 6.2).
//
// `h`, `heff` and `f` are the sidecar's (n_mo, n_mo) matrices of the fitted
// calculation, uploaded as they are. Blocks whose ranks or extents do not
// agree with one another -- a malformed golden or sidecar, or a pair from two
// calculations -- are the caller's input: an IO Error, before any upload.
Result<GeneralIntegralsDevice> dfGeneralIntegralsDevice(
    const Tensor& bAA, const Tensor& bCA, const Tensor& bVA, const Tensor& bCV, const Tensor& bCC,
    const Tensor& bVV, const Tensor& h, const Tensor& heff, const Tensor& f,
    const DeviceResources& res);

// One direction's answer over the full kappa space, summed over the eight
// classes. `integral` and `tangentIntegral` are the two modes of the integral
// half -- the pair the duality gate reads, and the number that is bit-for-bit
// the active block's for an active-restricted kappa. `eps` is the
// orbital-energy response, which is ONE contraction (R_t against Deltadot_t)
// shared by both modes, so it is reported once and added to both.
//
// `digest` is the f3ac/f3ca digests' own term in REVERSE mode (#89):
// <hbar, h2e_dot(kappa)>, with hbar = dE/d(h2e) through the digests
// (scDigestAdjointDevice) -- zero, and `hasDigest` false, when the caller
// handed no hbar and so asked for the frozen-digest functional. It is in
// `adjoint` and NOT in `tangent`: forward mode's matching term is
// scDigestRotationTangentDevice's, a separate call, so the full duality is
// `adjoint` against `tangent + ScDigestTangent::total`.
struct ScFullRotationPairing {
  double ext = 0.0;              // sum_blocks <D, Xdot>
  double act = 0.0;              // sum_G <M_G, Gdot>
  double integral = 0.0;         // ext + act
  double eps = 0.0;              // sum_t R_t Deltadot_t
  double digest = 0.0;           // <hbar, h2e_dot>: the digests' term (#89)
  double adjoint = 0.0;          // <kappa, Y> = integral + eps + digest
  double tangentIntegral = 0.0;  // Edot's integral half, P.Ndot + Q.Hdot
  double tangent = 0.0;          // tangentIntegral + eps (the frozen-digest tangent)
  bool hasTangent = false;
  bool hasDigest = false;
};

struct ScFullRotationResult {
  ScGradientResult grad;                      // the pseudodensity identities, as above
  std::vector<ScFullRotationPairing> dirs{};  // one per request, in order
};

// The pseudodensity assembly plus the FULL source term paired against each
// direction. `dirs`' kappas are (n_mo, n_mo) and antisymmetric; a malformed
// one, or a sidecar whose shapes do not match the integrals it sits beside,
// is the CALLER's input and comes back as an Error, reported once at main.
//
// Integral-direct only -- full blocks, one slab per block: the promotion rule
// reads whole blocks (a slab would need the sidecar's arrays sliced along a
// batch axis too), and a sidecar exists only for cases small enough not to
// need tiling anyway (docs/reference-data.md, "The gradient sidecar").
//
// `digestAdjoint` is hbar = dE/d(h2e) through the digests
// (scDigestAdjointDevice, #89). Non-null, `adjoint` is the EXACT <kappa, Y>
// at fixed CI -- the digests' term paired with the very h2e_dot the Dyall
// blocks read; null, it is the frozen-digest functional's, as before #89.
// Either way the integral halves the duality gate reads are unchanged.
Result<ScFullRotationResult> scFullRotationDevice(const NevptIntegralsDevice& ints,
                                                  const GeneralIntegralsDevice& gen,
                                                  const DeviceTensor& dm1,
                                                  const DeviceTensor& dm2,
                                                  const DeviceTensor& dm3,
                                                  const DeviceTensor& f3ac,
                                                  const DeviceTensor& f3ca,
                                                  const DeviceTensor* digestAdjoint,
                                                  const std::vector<ScRotationRequest>& dirs,
                                                  const DeviceResources& res);

// Y as a MATRIX on the requested index pairs: `pairs` lists (x, y) with
// x < y, and the answer carries Y_xy = (1/2) <E_xy - E_yx, Y> there,
// Y_yx = -Y_xy, and zero everywhere else -- so that sum_xy kappa_xy Y_xy is
// the pairing with any antisymmetric kappa supported on those pairs. `chunk`
// pairs share one slab walk.
//
// Probing is what a matrix costs, for the reason scActiveRotationMatrix
// gives: the directly contractible part of the pairing is symmetric in
// (x, y) and so invisible to an antisymmetric kappa, and everything that is
// left is a map INTO the integrals. A caller that wants only Park Eqs. 56-57
// needs the core-core and virtual-virtual pairs alone. `digestAdjoint` as
// scFullRotationDevice's: non-null, Y carries the digests' response (#89).
Result<Tensor> scFullRotationMatrix(const NevptIntegralsDevice& ints,
                                    const GeneralIntegralsDevice& gen, const DeviceTensor& dm1,
                                    const DeviceTensor& dm2, const DeviceTensor& dm3,
                                    const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                    const DeviceTensor* digestAdjoint,
                                    const std::vector<std::pair<int64_t, int64_t>>& pairs,
                                    int64_t chunk, const DeviceResources& res);

// What checks the promotion table, which neither mode of the duality can
// (they share it). Three families of exact identities, every one of them a
// rule's OWN source array read back where it must reproduce something the
// golden or the sidecar already holds:
//
//   * every promotion's source array, restricted to the promoted slot's own
//     space, is the stored block under the permutation that rule uses --
//     `blocks`, over all sixteen blocks and their slots;
//   * h^eff rebuilt from `h` and the dressing rules' own Coulomb and exchange
//     sources is the sidecar's `heff` -- `heff`, per one-electron block;
//   * the generalized Fock's core and virtual diagonals, rebuilt from `heff`
//     and the orbital-energy rules' own tied arrays against `dm1`, are the
//     sidecar's `f` diagonal -- `fock`; and that diagonal is the golden's
//     `e_core` / `e_virt` -- `eps`, which is (O2)
//     (docs/gradient-theory.md, Sec. 3.6) seen from this side and is what
//     says the sidecar belongs to this golden.
//
// Each number is a worst case over everything in its family, as a relative
// Frobenius norm; `worstBlock` names where `blocks` peaked.
//
// And, for the mixed block's orbital-energy term (#184,
// docs/hessian-theory.md Sec. 5.5's C5), the FULL-RANGE identity its Fock-line
// generator carries: h^eff + f^gamma[gamma] at the state's gamma is the
// sidecar's f over the whole MO range, off-diagonal blocks included --
// `fockFull`, the extension of `fock`, which rebuilds the diagonal only.
struct ScGeneralCheck {
  double blocks = 0.0;
  std::string worstBlock{};
  double heff = 0.0;
  double fock = 0.0;
  double eps = 0.0;
  double fockFull = 0.0;
};

Result<ScGeneralCheck> scGeneralIntegralCheck(const NevptIntegralsDevice& ints,
                                              const GeneralIntegralsDevice& gen,
                                              const DeviceTensor& dm1,
                                              const DeviceResources& res);

// Prints the per-direction table of the full source term -- the adjoint
// pairing and its three pieces, the forward tangent, and the two against
// each other -- and returns whether every direction that carries a tangent
// agrees to `atol`. As printScRotationTable, the PASS/FAIL line is the
// caller's.
bool printScFullRotationTable(const ScFullRotationResult& r, double atol);

// --- the digests' orbital response, forward mode (#88) -----------------------
//
// The term the two modes above both leave out: the f3ac/f3ca digests are the
// active h2e contracted with the 4-RDM, so along a rotation they move with
// h2e, and the exact Edot(kappa) is
//
//   Edot(kappa) = [ the frozen-digest tangent above: tangentIntegral + eps ]
//               + sum_k Q_k Hdot_k |_{f3 -> f3_dot(kappa)}            (this)
//
// NO NEW KERNEL, by the pattern #27 set for the CI tangent ("a Dyall block's
// tangent is its own builder called with the dotted integrals"): the whole
// RDM build -- consume, digest, the fdm2 correction and the wedge -- is
// LINEAR in h2e at a fixed CI vector, so the digests' tangent along kappa is
// buildRdmsDevice itself run at h2e_dot(kappa), the very tangent the rotation
// machinery forms for the Dyall blocks (intra-active action plus the
// sidecar's general part; its dm3 output, which reads no integral, is
// discarded). And the energy's response to a digest displacement at fixed
// integrals is energiesTangentDevice with every other RDM tangent zero: the
// digests enter make_a16 / make_a22 as plain addends, so only the H side
// moves, and the orbital energies -- which no digest reaches -- are held.
//
// Reported as ITS OWN TERM beside the frozen-digest tangent rather than folded
// into it, so the integral halves' duality keeps comparing like with like;
// reverse mode's matching term is scDigestAdjointDevice's (#89, below), and
// the two meet in the FULL duality. What validates this one independently is
// outside the pair: PySCF's exact directional derivative minus its frozen-digest one
// (reference_data/nevpt2_grad.py --edot-check, reading the directions and
// this term out of a `.gradsrc`), and on the card a finite difference of the
// energy with only the digests rebuilt (apps/sc_pseudodensity/'s
// --kappa-full).
//
// Exact digests only: `opt` must name the emitted or the native-fp64 BLAS
// digest, as every tangent path's must (cli refuses --cublas / --ozaki /
// --fused-digest with the flags that reach this; reaching it anyway is our
// bug). `ci` is the CI vector the plain build ran on; `nelecA`/`nelecB` its
// strings. One RDM build and one tangent slab walk per direction, on
// res.stream(), at `opt.nTiles` -- the plain build's working set, so the
// plain build's tile floors (docs/performance.md, "Tile floors").
struct ScDigestTangent {
  std::array<double, 8> classes{};  // each class's d/dt through the digests alone
  double total = 0.0;               // their sum: Edot's digest term
  double h2eDotMax = 0.0;           // max |h2e_dot(kappa)|: did kappa move h2e at all
};

Result<std::vector<ScDigestTangent>> scDigestRotationTangentDevice(
    const RdmBuildOptions& opt, const Tensor& ci, int64_t nelecA, int64_t nelecB,
    const NevptIntegralsDevice& ints, const GeneralIntegralsDevice& gen, const DeviceTensor& dm1,
    const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceTensor& f3ac,
    const DeviceTensor& f3ca, const std::vector<ScRotationRequest>& dirs,
    const DeviceResources& res);

// The same along ACTIVE-ACTIVE directions with no sidecar (#184): `dirs`'
// kappas are (n_act, n_act), as scActiveRotationDevice's, and h2e_dot is the
// intra-active action alone -- which is all of it there, and needs nothing the
// golden lacks. What the mixed block's Euler identity is held to when no
// sidecar is read.
Result<std::vector<ScDigestTangent>> scDigestRotationTangentDevice(
    const RdmBuildOptions& opt, const Tensor& ci, int64_t nelecA, int64_t nelecB,
    const NevptIntegralsDevice& ints, const DeviceTensor& dm1, const DeviceTensor& dm2,
    const DeviceTensor& dm3, const DeviceTensor& f3ac, const DeviceTensor& f3ca,
    const std::vector<ScRotationRequest>& dirs, const DeviceResources& res);

// Prints Edot(kappa) in its two terms -- the frozen-digest tangent, the
// digest term -- and their sum, one line per direction. `full` and `digest`
// are the answers for the same direction list, in the same order. Where
// `full` carries the reverse-mode digest term too (#89), each line adds it
// and the FULL duality, |<kappa, Y> - Edot|, and the return is whether every
// such line is within `atol`; true when none carries it.
bool printScDigestTangentTable(const ScFullRotationResult& full,
                               const std::vector<ScDigestTangent>& digest, double atol);

// --- the digests' orbital response, reverse mode (#89) -----------------------
//
// The reverse half of the term above: hbar = dE/d(h2e) THROUGH THE DIGESTS,
// the n_act^4 tensor whose pairing with h2e_dot(kappa) is that term, so that
// Y -- probed basis direction by basis direction -- carries it.
//
// Two steps, both already their own checked pieces: the slab walk's RDM sweep
// (nevpt2.gradient:rdm_adjoint, the one scCiSourceDevice runs) leaves
// G_ac = dE/d(f3ac) and G_ca = dE/d(f3ca), and the build's transpose
// (nevpt2.rdm_build's buildDigestAdjointDevice: the 4-RDM against a six-index
// conjugate density, tiled, the 4-RDM never formed) takes them back onto
// h2e. Since the digests are linear in h2e at a fixed CI vector,
//
//   <hbar, h2e_dot(kappa)> == sum_k Q_k Hdot_k |_{f3 -> f3_dot(kappa)}
//
// -- scDigestRotationTangentDevice's term, to round-off, for every kappa: the
// full duality `--kappa-full` gates. `opt` supplies the tile count alone (the
// transpose has one route, whichever digest built the RDMs); `ci` is the
// state, `nelecA`/`nelecB` its strings. Everything on res.stream().
DeviceTensor scDigestAdjointDevice(const RdmBuildOptions& opt, const Tensor& ci, int64_t nelecA,
                                   int64_t nelecB, const NevptIntegralsDevice& ints,
                                   const DeviceTensor& dm1, const DeviceTensor& dm2,
                                   const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                   const DeviceTensor& f3ca, const DeviceResources& res);

// --- the digests' transpose onto the THREE-index tensor (#168) ---------------
//
// On a density-fitted case h2e is itself sum_L B_aa B_aa (docs/gradient-
// theory.md, Sec. 3.8), so the transpose's hbar can be carried one step
// further back, onto B, and the digests' term of <kappa, Y> read off B alone
// -- no one-general-index four-index array (g_xa_aa, which h2e_dot's general
// part reads) is needed for it. In nevpt2.energy's layout h2e[p,q,r,s] =
// (p r | q s) = sum_L B[L,p,r] B[L,q,s], so with
//
//   Bdot[L,t,u] = sum_x ( kappa_xt B[L,x,u] + kappa_xu B[L,t,x] )
//                                        (Sec. 3.1's rule on each B factor)
//   h2e_dot[p,q,r,s] = sum_L ( Bdot[L,p,r] B[L,q,s] + B[L,p,r] Bdot[L,q,s] )
//
// the pairing <hbar, h2e_dot> is <Bbar, Bdot> with
//
//   Bbar[L,t,u] = sum_{v,w} ( hbar[t,v,u,w] + hbar[v,t,w,u] ) B_aa[L,v,w]
//
// and, Bdot's two promotions folded onto the one strip B[L, x, a] (x over the
// whole MO range, a active -- B is symmetric in its MO pair),
//
//   <hbar, h2e_dot(kappa)> = sum_{x, t in act} kappa[x, t] D[x, t],
//   D[x, t] = sum_{L, u} B[L,x,u] ( Bbar[L,t,u] + Bbar[L,u,t] ).
//
// `D` is (n_mo, n_act), with t indexing the active space (column t is MO
// ncore + t). It reads hbar (n_act^4: the transpose's own output, the 4-RDM
// still never formed) and the golden's three blocks with an active index --
// B_ca, B_aa, B_va -- and nothing of the sidecar's: the digests' term never
// reaches B_cc, B_vv or B_cv. Returned on the host; the einsums run on
// res.stream(). Blocks whose extents disagree with one another or with hbar
// are the caller's input: an IO Error, before any upload.
Result<Tensor> dfDigestSourceDevice(const DeviceTensor& hbar, const Tensor& bAA,
                                    const Tensor& bCA, const Tensor& bVA,
                                    const DeviceResources& res);

// sum_{x, t} kappa[x, ncore + t] d[x, t]: the digests' term of <kappa, Y>
// along `kappa` ((n_mo, n_mo)) off dfDigestSourceDevice's `d`. On the host;
// exactly zero when kappa has no column in the active space.
double dfDigestPairing(const Tensor& d, const Tensor& kappa, int64_t ncore);

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
  // Whether `y` carries the orbital energies' response (an
  // OrbitalEnergyResponse was handed in), and if so dE/de over the core and
  // virtual orbitals, summed over the eight classes -- what was folded into
  // dE/d(dm1) through it. Empty when the orbital energies were held fixed.
  bool orbitalEnergiesMove = false;
  std::vector<double> dEdCore{};
  std::vector<double> dEdVirt{};
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
// `eps` is the orbital energies' response to the active density
// (nevpt2.gradient_plan's OrbitalEnergyResponse, built from the gradient
// sidecar). Non-null, y is the derivative of the COMPLETE functional, in
// which e_core / e_virt move with the CI vector: each class's R_t is
// scattered onto the orbital energies its Delta_t reads, and the resulting
// dE/de is folded into dE/d(dm1) through W before the sigma calls
// (docs/gradients.md, Sec. 5.5). Null holds them fixed at the golden's, the
// functional a case with no sidecar can differentiate. Forward mode's
// counterpart is energiesTangentDevice's `epsDot`.
//
// Everything runs on res.stream() and every device allocation is a
// DeviceBuffer from res's pool; the five coefficient tensors live from the
// first class to the last sigma call and are freed when this returns.
ScCiSourceResult scCiSourceDevice(const ActiveIntegralsDevice& active, IntegralSource& source,
                                  int64_t batch, const Tensor& ci, int64_t nelecA,
                                  int64_t nelecB, const DeviceTensor& dm1,
                                  const DeviceTensor& dm2, const DeviceTensor& dm3,
                                  const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                  const SigmaVectorOptions& sigma,
                                  const OrbitalEnergyResponse* eps, const DeviceResources& res);

// The integral-direct entry point: full blocks, one slab each.
ScCiSourceResult scCiSourceDevice(const NevptIntegralsDevice& ints, const Tensor& ci,
                                  int64_t nelecA, int64_t nelecB, const DeviceTensor& dm1,
                                  const DeviceTensor& dm2, const DeviceTensor& dm3,
                                  const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                  const SigmaVectorOptions& sigma,
                                  const OrbitalEnergyResponse* eps, const DeviceResources& res);

// Step 3 above on its own: y_I from the five RDM coefficient tensors
// g1 = dE/d(dm1), g2, g3, gac = dE/d(f3ac), gca = dE/d(f3ca) and the state
// `ci` they are differentiated at, as a Tensor of ci's own
// (nstringsA, nstringsB) shape -- one 3-body sigma over
// ciSourceSymmetrized(ciSourceFolded(g1, g2, g3, nelecA + nelecB)) plus the
// forward and dagger 4-body sigmas of each f3 order, summed by
// ciSourceAddInto. `h2e` is the PHYSICIST-ordered active block the golden
// carries. Defined in the :ci_source partition and exported so the unit tier
// can check its shape and accumulation, and the whole fold-symmetrize-sigma
// chain against an unfolded host reference (test/gradient_plan/'s
// CiSourceVectorTests); scCiSourceDevice is its one caller in src/.
Tensor ciSourceVector(const SigmaVectorOptions& opt, const Tensor& ci, int64_t norb,
                      int64_t nelecA, int64_t nelecB, const Tensor& g1, const Tensor& g2,
                      const Tensor& g3, const Tensor& gac, const Tensor& gca, const Tensor& h2e,
                      const DeviceResources& res);

// --- the CI block's Hessian-vector product, H_cc u (#156) ---------------------
//
// docs/hessian-theory.md, Sec. 4.2: REVERSE-OVER-FORWARD. y = sigma(g, c),
// with g = (g1, g2, g3, gac, gca) the five coefficient tensors of the sweep
// above and sigma(., .) ciSourceVector -- linear in g at fixed state, and
// linear in the state at fixed g (each RDM is a bilinear form in the CI
// vector, so dRDM/dc is linear in c). So along a CI direction u
//
//   H_cc u  =  sigma(gdot[u], c)  +  sigma(g, u),
//
// two calls of the SAME ciSourceVector at different arguments. gdot is the
// walk differentiated along u, and every step of the walk is linear or
// bilinear: the per-tuple norms and Dyall elements are linear in the blocks,
// the blocks in the RDMs, and the RDM sweep in the accumulators with
// constant coefficients -- so gdot is the same einsums at the tangent inputs
// (the blocks at the RDMs' tangents along u), with the multipliers replaced
// by their tangents Pdot / Qdot (nevpt2.energy_finish's
// normToEnergyMultiplierTangent: the 3x3 of Sec. 2.4, the only new
// ingredient). One walk carries both g and gdot.
//
// THE ORBITAL ENERGIES ARE HELD FIXED (the golden's e_core / e_virt;
// Deltadot = 0), the functional energiesHessianDevice differentiates: the
// exact gate is the duality <v, H_cc u> == Eddot(u, v), and the two sides
// must differentiate one functional. The orbital energies' response at
// second order is later work.
//
// Everything runs on res.stream(); every device allocation is a DeviceBuffer
// from res's pool. The ten coefficient tensors (g and gdot) live from the
// first class to the last sigma call, 2 (n^2 + n^4 + 3 n^6) doubles, beside
// the walk's own per-class blocks and their tangents.
struct ScRdmCoefficients {
  Tensor dm1, dm2, dm3, f3ac, f3ca;
};

struct ScCiHessianResult {
  // H_cc u over the determinant space, in the CI vector's (nstringsA,
  // nstringsB) shape, and its two halves.
  Tensor hu;
  Tensor fromWalkTangent;  // sigma(gdot[u], c)
  Tensor fromDirection;    // sigma(g, u)
  // y = sigma(g, c) from the same walk, at fixed orbital energies: free, and
  // what scCiSourceDevice returns with no OrbitalEnergyResponse.
  Tensor y;
  // gdot[u] itself, host-resident: the reverse side's walk tangent.
  ScRdmCoefficients gDot;
  // The walk's own Edot(u) per class (CLASSES order) and summed:
  // sum_t P_t Ndot_t + Q_t Hdot_t from the tangent blocks it built, which
  // must be energiesTangentDevice's Edot(u) -- the free check that the
  // tangent blocks are the forward tangent's.
  std::array<double, 8> edotU{};
  double edotUTotal = 0.0;
  // The walk's Eq. 40-47 numbers, as scCiSourceDevice reports them.
  ScGradientResult pseudodensities;
};

// `ci` and `u` are (nstringsA, nstringsB); `rdm` is the state's five RDMs
// (the caller's dm1/dm2 upload and dm3/f3ac/f3ca build, as for
// scCiSourceDevice) and `dotU` their tangents along u (nevpt2.rdm_host_dm's
// dm12Dot(ci, u) and nevpt2.rdm_build's buildRdmTangentsDevice(ci, u)).
// `source`/`batch` as for energiesDevice; `sigma` as for scCiSourceDevice.
ScCiHessianResult scCiHessianVectorDevice(const ActiveIntegralsDevice& active,
                                          IntegralSource& source, int64_t batch,
                                          const Tensor& ci, const Tensor& u, int64_t nelecA,
                                          int64_t nelecB, const RdmTensors& rdm,
                                          const RdmTensors& dotU,
                                          const SigmaVectorOptions& sigma,
                                          const DeviceResources& res);

// --- the mixed orbital-CI block, H_kc (#184) ----------------------------------
//
// docs/hessian-theory.md, Sec. 5.5. With w an orbital direction (an ordinary
// antisymmetric (n_mo, n_mo) kappa) and u a CI direction,
//
//   Eddot_kc(w, u) = d^2/ds dt E(exp(s w), c + t u) = <H_kc^T w, u>,
//
// built in BOTH of the orientations Sec. 5.5 settles on, so that each gates
// the other:
//
//   F   the SCALAR, forward over forward (scMixedHessianDevice): one slab walk
//       carrying the plain blocks, u's tangent blocks and w's dotted inputs,
//       finished per perturber in Sec. 5.5's five terms (ScMixedTerms):
//       C1 the 3x3 sandwich between w's and u's (Ndot, Hdot, Ddot), C2 one
//       external slot along w against the block along u, C3 the Dyall blocks
//       at w's dotted integrals and u's tangent RDMs, C4 the blocks at the
//       digests' mixed tangent f3^{wu} (the tangent RDM build at (c, u) handed
//       h2e'[w] -- #88's pattern one order up), and C5 R against the orbital
//       energies' mixed response eps^{wu};
//   R1  the VECTOR H_kc^T w, reverse in c and forward in kappa
//       (scMixedHessianVectorDevice): y = sigma(g; h2e) differentiated along
//       w, which is y's walk differentiated along w (the multipliers'
//       tangents from the orbital triple, the external slots' tangents, the
//       RDM sweep at the dotted integrals) through ONE sigma call, plus the
//       4-body sigmas with only (gac, gca) at h2e'[w] -- y is linear in h2e,
//       so no transpose and no tangent RDM build.
//
// Their gate is the DUALITY <H_kc^T w, u> == Eddot_kc(w, u), with no step
// size. They share the operand tangents (the promotion rule, the slab dots,
// the tangent blocks, deps), the 3x3, the multiplier tangents, h2e'[w], the
// digests' tangent f3'[w] and C5's generator, and nothing past those: the
// tangent RDM build against the sigma vectors, forward against reverse.
// R2, the orbital vector H_kc u, is not built (Sec. 5.5 says why).
//
// THE DIGESTS' SHARE IS MERGED PER TUPLE: w's tangent blocks are built with
// the digests' own tangent along w (the RDM build at h2e'[w]) in place of the
// zeros the first-order entry points use, so every tuple's Hdot[w] carries it
// before the 3x3 meets it -- C1's requirement, and #158's K3's. Both results
// report the walk's own Edot(w) per class, which must be the first-order
// source term's (the frozen-digest tangent plus #88's digest term).
//
// THE ORBITAL ENERGIES' MODE is the caller's and is part of every check:
// `epsKappa` lets them respond to kappa (deps, out of the sidecar), `epsCi`
// to the CI vector (through W, `epsResponse`); with both, C5 is on. Euler in
// c (Eddot_kc(w, c) == 2 <w, Y> on the homogeneous classes) is an identity
// with `epsCi` off; the duality and the finite difference of y hold in any
// mode, both legs in the same one.
//
// WITHOUT A SIDECAR (`gen` null) only an active-active w can be formed, with
// the orbital energies frozen in both legs: any other w, or either mode on,
// is refused as Unsupported -- never answered with the integral half alone
// (docs/hessian-theory.md, Sec. 9.1). The digests still move: along an
// active w, h2e'[w] needs nothing the golden lacks.
//
// Integral-direct, one slab per block (as scFullRotationDevice); every
// device tensor from res's pool, everything on res.stream(). Exact digests
// only (`opt`), at `opt.nTiles` -- F runs the tangent RDM build (its floors),
// R1 the plain build (its floors). A malformed w, a sidecar that does not fit
// the integrals, or a missing W is the caller's input: an Error.
struct ScMixedHessianSetup {
  const NevptIntegralsDevice& ints;
  const GeneralIntegralsDevice* gen;  // the gradient sidecar, or null
  const RdmTensors& rdm;              // the state's five RDMs
  const Tensor& ci;                   // (nstringsA, nstringsB)
  int64_t nelecA = 0;
  int64_t nelecB = 0;
  const RdmBuildOptions& opt;
  bool epsKappa = false;
  bool epsCi = false;
  const OrbitalEnergyResponse* epsResponse = nullptr;  // W: required when epsCi
};

// F along (w, u): per class and summed.
struct ScMixedHessianResult {
  std::array<ScMixedTerms, 8> classes{};  // CLASSES order
  ScMixedTerms total{};
  double eddot = 0.0;                     // total.total(): Eddot_kc(w, u)
  std::array<double, 8> edotW{};          // the walk's own Edot(w), digests merged
  std::array<double, 8> edotU{};          // the walk's own Edot(u)
  double edotWTotal = 0.0;
  double edotUTotal = 0.0;
  double h1eDotMax = 0.0;                 // max |h1e'[w]|, max |h2e'[w]|: did w
  double h2eDotMax = 0.0;                 // move the active integrals at all
};

// F along w and each of `us` (each shaped like `ci`): w's dotted inputs, its
// digests' tangent and C5's generator built once, then per u its tangent RDMs
// (dm1/dm2 on the host, dm3/f3ac/f3ca and f3^{wu} on the device) and one walk.
Result<std::vector<ScMixedHessianResult>> scMixedHessianDevice(const ScMixedHessianSetup& setup,
                                                               const Tensor& w,
                                                               const std::vector<Tensor>& us,
                                                               const DeviceResources& res);

// R1 along w.
struct ScMixedHessianVectorResult {
  Tensor hkcTw;            // H_kc^T w, (nstringsA, nstringsB)
  Tensor fromWalkTangent;  // sigma(gdot[w], c; h2e): the walk's tangent, eps fold included
  Tensor fromDigests;      // sigma(0, 0, 0, gac, gca; h2e'[w])
  std::array<double, 8> edotW{};  // as ScMixedHessianResult's
  double edotWTotal = 0.0;
  double h2eDotMax = 0.0;
};

Result<ScMixedHessianVectorResult> scMixedHessianVectorDevice(const ScMixedHessianSetup& setup,
                                                              const Tensor& w,
                                                              const SigmaVectorOptions& sigma,
                                                              const DeviceResources& res);

// --- PC-NEVPT2's reverse mode (#74) -------------------------------------------
//
// The PC source terms -- y_I = dE_pc/dc_I and Y_xy = dE_pc/dkappa_xy -- by the
// adjoint of the PC solve, the counterpart of everything above for the
// partially-contracted energy (nevpt2.energy's pcEnergiesDevice). The
// derivation, and why the solve may be differentiated at all, is the
// :pc_classes partition's header (it cites docs/pc-nevpt2.md, "The
// singular-metric convention"): the adjoint is valid exactly where the energy
// accepts a class, and A CLASS THE GAP OR DENOMINATOR CHECK REFUSES IS REFUSED
// HERE TOO -- reported through its PcClassResult, never as a zero
// contribution. Like SC's, these carry the digests' and the orbital energies'
// dependences (the #73 amendment): y the orbital energies' response through
// an OrbitalEnergyResponse, Y the f3ac/f3ca digests' transpose through
// `digestAdjoint` (pcDigestAdjointDevice) and the orbital energies' response
// through the sidecar. The forward modes they are held against are
// nevpt2.energy's pcTangentsDevice (with its `epsDot`) along a CI direction,
// and pcFullRotationDevice's own tangent plus pcDigestRotationTangentDevice
// along a rotation.

// One PC class's reverse-mode answer. `result` is the class's own solve's
// (status, energy, spectrum): nothing else here means anything unless it is
// Done. The rotation vectors parallel the request list, as ScClassGradient's;
// dE/de is filled when the orbital energies' response was asked for.
struct PcClassGradient {
  PcClassResult result;
  std::vector<double> rotExt{};      // sum_slabs <dE/dslab, slabdot>
  std::vector<double> rotAct{};      // <dE/dK, Kdot>
  std::vector<double> rotTangent{};  // the forward tangent's integral half
  std::vector<double> rotEps{};      // sum_t R_t Deltadot_t, shared by both modes
  std::vector<double> dEdCore{};
  std::vector<double> dEdVirt{};
};

struct PcGradientResult {
  std::array<PcClassGradient, 8> classes{};  // CLASSES order
  double total = 0.0;                        // sum of the classes' energies

  // Every class Done: what every consumer must check before reading a number.
  bool allDone() const {
    return std::ranges::all_of(
        classes, [](const PcClassGradient& c) { return c.result.status == PcStatus::Done; });
  }
  // "Sr (S's gap 1.2e+00 < PC_MIN_GAP 1e+03), ..." -- the refused classes and
  // why, for a report or an Error; empty when allDone().
  std::string refusals() const;
};

// The PC CI source term. `ok` is allDone(): only then is `y` the derivative
// (and filled); otherwise `walk` says which class was refused and why.
struct PcCiSourceResult {
  PcGradientResult walk;
  bool ok = false;
  Tensor y;  // (nstringsA, nstringsB), when ok
  bool orbitalEnergiesMove = false;
  std::vector<double> dEdCore{};
  std::vector<double> dEdVirt{};
};

// y_I = dE_pc/dc_I: every class's solve differentiated down to the five RDM
// coefficient tensors (the SC sweeps of nevpt2.gradient:rdm_adjoint, reused),
// then the same sigma assembly as scCiSourceDevice (ciSourceVector). `eps`
// non-null: the orbital energies move with the active density, R_t scattered
// onto dE/de and folded onto dm1 through W, exactly as SC's (#87). An
// eigensolve's devInfo != 0 is the Numerical Error pcEnergiesDevice returns.
// Integral-direct, on res.stream(), every tensor from res's pool.
Result<PcCiSourceResult> pcCiSourceDevice(const NevptIntegralsDevice& ints, const Tensor& ci,
                                          int64_t nelecA, int64_t nelecB,
                                          const DeviceTensor& dm1, const DeviceTensor& dm2,
                                          const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                          const DeviceTensor& f3ca,
                                          const SigmaVectorOptions& sigma,
                                          const OrbitalEnergyResponse* eps,
                                          const DeviceResources& res);

// One direction's PC answer over the full kappa space, summed over the eight
// classes -- ScFullRotationPairing's fields, PC's numbers: `integral` and
// `tangentIntegral` the two modes of the integral half (the pair the duality
// reads), `eps` the orbital energies' share both carry, `digest` the reverse
// digests' term (<hbar, h2e_dot>, zero unless `digestAdjoint` was handed in).
// The FULL duality is `adjoint` against `tangent` plus
// pcDigestRotationTangentDevice's term.
struct PcFullRotationResult {
  PcGradientResult walk;
  bool ok = false;  // walk.allDone(): only then is `dirs` meaningful
  std::vector<ScFullRotationPairing> dirs{};
};

// The PC pseudodensities paired with every direction (full (n_mo, n_mo)
// antisymmetric kappas) and, for a direction that asks, the forward tangent
// along it. Integral-direct only, as scFullRotationDevice (the general action
// reads whole blocks). A malformed kappa or a sidecar that does not fit is
// the caller's input: an Error.
Result<PcFullRotationResult> pcFullRotationDevice(
    const NevptIntegralsDevice& ints, const GeneralIntegralsDevice& gen, const DeviceTensor& dm1,
    const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceTensor& f3ac,
    const DeviceTensor& f3ca, const DeviceTensor* digestAdjoint,
    const std::vector<ScRotationRequest>& dirs, const DeviceResources& res);

// Y_pc as a MATRIX on the requested pairs, by probing, exactly as
// scFullRotationMatrix. A refused class is a Numerical Error naming it --
// there is no place in a matrix for "this class has no derivative".
Result<Tensor> pcFullRotationMatrix(const NevptIntegralsDevice& ints,
                                    const GeneralIntegralsDevice& gen, const DeviceTensor& dm1,
                                    const DeviceTensor& dm2, const DeviceTensor& dm3,
                                    const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                    const DeviceTensor* digestAdjoint,
                                    const std::vector<std::pair<int64_t, int64_t>>& pairs,
                                    int64_t chunk, const DeviceResources& res);

// hbar = dE_pc/d(h2e) THROUGH THE DIGESTS: the PC sweep's dE/d(f3ac),
// dE/d(f3ca) (Sr's a16 and Si's a22 read them) taken back onto h2e by
// nevpt2.rdm_build's transpose -- scDigestAdjointDevice's two steps, PC's
// first. A refused class is a Numerical Error naming it.
Result<DeviceTensor> pcDigestAdjointDevice(const RdmBuildOptions& opt, const Tensor& ci,
                                           int64_t nelecA, int64_t nelecB,
                                           const NevptIntegralsDevice& ints,
                                           const DeviceTensor& dm1, const DeviceTensor& dm2,
                                           const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                           const DeviceTensor& f3ca, const DeviceResources& res);

// The digests' term in FORWARD mode along each direction: the RDM build at
// h2e_dot(kappa) is (f3ac_dot, f3ca_dot), and pcTangentsDevice with every
// other RDM tangent zero is the PC energy's response to it --
// scDigestRotationTangentDevice's construction for PC. Exact digests only
// (the caller's cli refuses the rest); a refused class is a Numerical Error.
Result<std::vector<ScDigestTangent>> pcDigestRotationTangentDevice(
    const RdmBuildOptions& opt, const Tensor& ci, int64_t nelecA, int64_t nelecB,
    const NevptIntegralsDevice& ints, const GeneralIntegralsDevice& gen, const DeviceTensor& dm1,
    const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceTensor& f3ac,
    const DeviceTensor& f3ca, const std::vector<ScRotationRequest>& dirs,
    const DeviceResources& res);

// The per-class PC table every PC source-term report prints: status (REFUSED
// with its reason, or the energy), and the total.
void printPcGradientClasses(const PcGradientResult& r);

}  // namespace nevpt2
