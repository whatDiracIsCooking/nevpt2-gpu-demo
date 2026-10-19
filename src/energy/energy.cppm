// nevpt2.energy -- the eight NEVPT2 class energies, strongly contracted
// (energy.cpp, energiesDevice) and partially contracted (energy_pc.cpp,
// pcEnergiesDevice). SC: built from device-resident RDMs + MO
// integrals via the generic device einsum kernel (device_einsum.cu). This is
// the sole energy path both demos run; it was ported from a NumPy reference
// via an intermediate serial-host reference (an earlier, all-host energy.cpp,
// since replaced by this device one) that
// let each of its ~140 np.einsum calls be transcribed and checked one at a
// time before the CUDA index arithmetic was layered on. That host reference
// has done its job and been retired -- the device result is now validated
// directly, end to end, against the golden per-class PySCF energies
// (each demo's main.cppm, tolerance 1e-7); see docs/implementation.md, "Porting the class
// energies", for how the port was validated.
//
// THE EXTERNAL INTEGRALS ARRIVE IN SLABS. Every class's output is indexed by
// (at least) one external orbital -- Sr by a virtual, Si by a core, Srs by a
// virtual pair, ... -- and each output element needs only the integral
// entries carrying ITS external index. So each class walks its external index
// in batches and asks an IntegralSource for just that slab of its block(s):
//   - the integral-direct demo hands over FullBlockSource, whose slabs are
//     views into the full blocks it uploaded from the golden file (one batch
//     covering everything -- the same contraction as before slabs existed);
//   - the density-fitted demo builds each slab on the fly from the
//     three-index B tensors (src/df_integrals), so no full external
//     four-index block ever exists on the device.
// The einsum subscripts are unchanged either way; only their operand is a
// slab.
//
// PARTIALLY-contracted NEVPT2 (PC-NEVPT2), energy_pc.cpp -- the same eight
// classes, from the same inputs and intermediates as the SC ones above. The
// design is docs/pc-nevpt2.md, "PC-NEVPT2 on the device: the design";
// energy_pc.cpp implements it.
//
// PC IS SC WITHOUT THE FINAL CONTRACTION. Each SC class contracts every
// external tuple's integral vector x_t with a metric S and a Hamiltonian K
// (both active-only, the same for every tuple) down to two scalars. PC instead
// solves in the whole span of the class's internal functions:
//
//   E_t = -c * b_t^T (K + Delta_t S)^+ b_t,     b_t = S x_t
//
// Since only Delta_t and b_t vary with the tuple, each class is diagonalised
// ONCE (two Dsyevd on res.solver()): S = V s V^T, cut at s_k <= PC_TAU * s_max,
// X = V_kept s_kept^-1/2, X^T K_sym X = W lambda W^T, U = X W. Then
// y_t = U^T S x_t = T^T x_t with T = S U, so each slab is GEMM'd straight into
// the eigenbasis (one einsum against T), and
//   E_t = -c * sum_k y_tk^2 / (lambda_k + Delta_t)
// finishes on the host, like the SC classes' normToEnergy.
//
// THE GAP IS CHECKED, NOT ASSUMED. Per class the result carries S's spectrum
// (dropped count, largest dropped |s|, smallest kept, both relative to s_max)
// and the class is REFUSED -- no energy -- when smallest kept / largest dropped
// (or / PC_TAU when nothing is dropped) is below PC_MIN_GAP, generate_golden.py's
// _check_pc_conditioning rule applied to S. A non-positive denominator
// lambda_k + Delta_t refuses the class too, rather than dividing through a
// sign change.
//
// AND THE SC CLASSES CARRY A FORWARD TANGENT (energy_sc_tangent.cppm, the
// :sc_tangent partition): energiesTangentDevice, the derivative of the eight
// SC class energies with respect to the RDMs along a given direction. See
// "the forward tangent" below for why that needs no new contraction -- the
// class einsums are linear in the RDMs, so the tangent reruns them with the
// tangent RDMs substituted and differentiates only the scalar finish.
//
// PC'S TANGENT IS NOT LIKE THAT: IT DIFFERENTIATES THE SOLVE, NOT THE
// EIGENDECOMPOSITION -- and, unlike the SC tangent above, it cannot be a
// rerun of the energy's own contractions, because a pseudo-inverse sits
// between the RDMs and the energy
// (pcTangentsDevice, energy_pc_tangent.cppm). Only S, K and S_1 carry the CI
// vector, so with x_vec = A_t^+ b_t -- the solve the energy already ran --
// the derivative of E_t = -c b_t^T A_t^+ b_t along a CI direction is
//
//   Edot_t = -c * (2 x_vec^T bdot_t - x_vec^T Adot_t x_vec).
//
// No eigenvector derivative appears, so there is no degenerate-eigenvalue
// case to handle. WHY THAT IS LEGITIMATE, and where it stops being so: A^+ is
// a PSEUDO-inverse, and a pseudo-inverse is differentiable only where the
// rank is locally constant -- otherwise dA^+ does not exist and the line
// above is meaningless. Local rank constancy is exactly what PC_MIN_GAP
// already enforces: the modes the cut drops are STRUCTURAL -- linear
// dependencies among the class's active-space operator strings, null for
// every CI vector, measured at <=1.2e-15 of s_max -- and PC_MIN_GAP = 1e3
// demands three decades between them and the smallest kept mode, so no mode
// can cross the cut under a small change of the state
// (docs/pc-nevpt2.md, "The singular-metric convention"). The two extra terms
// of the general pseudo-inverse derivative, A^+ A^+ Adot (I - A A^+) and its
// transpose, drop because b_t lies in the kept range. So the tangent is valid
// precisely on the states pcEnergiesDevice already accepts -- and ONLY there:
// a class the gap or denominator check refuses is refused in the tangent too,
// through the same PcStatus::Refused (PcClassTangent carries that class's own
// PcClassResult), never reported as a zero derivative. The per-class algebra
// and the slab walk are in energy_pc_tangent.cppm; the step from dA^+ to the
// line above is at `tangentClass` in energy_pc_solve.cppm.
export module nevpt2.energy;

import std;
// Re-exported: energiesDevice and pcEnergiesDevice take DeviceTensors and the
// DeviceResources (nevpt2.device_resources, through nevpt2.einsum's nevpt2.device_tensor).
export import nevpt2.einsum;
// Re-exported: the class energies' host arithmetic and the PC result types
// (NUMERICAL_ZERO, PcClassResult, ...) that pcEnergiesDevice's result names.
export import nevpt2.energy_finish;

export namespace nevpt2 {

// NUMERICAL_ZERO, normToEnergy's vanishing-norm guard, is
// nevpt2.energy_finish's, re-exported above.

//: The eight class labels, fixed order -- the order
//: pyscf.mrpt.nevpt2.NEVPT.kernel evaluates them in, and generate_golden.py's
//: CLASSES.
extern const std::array<std::string, 8> CLASSES;

struct EnergyResult {
  std::array<double, 8> norms{};
  std::array<double, 8> energies{};
  double total = 0.0;
};

// The MO-integral half of the input contract, device-resident -- the fields
// and shapes are the golden file's integral blocks (generate_golden.py's
// _integral_blocks says which PySCF `eris[...]` slice each is), as DeviceTensors.
// All active blocks and external ("_v") blocks are physicist-ordered.
struct NevptIntegralsDevice {
  DeviceTensor h1e, h2e, e_core, e_virt;
  DeviceTensor h2e_v_Sr, h1e_v_Sr;
  DeviceTensor h2e_v_Si, h1e_v_Si;
  DeviceTensor cvcv;
  DeviceTensor h2e_v_Sijr, h2e_v_Srsi, h2e_v_Srs, h2e_v_Sij;
  DeviceTensor h2e_v1_Sir, h2e_v2_Sir, h1e_v_Sir;
};

// Everything except the external two-electron blocks: the active-space
// integrals, the orbital energies and the three one-electron external blocks
// (small -- n_ext * n_act at most -- and not a density-fitting target: they
// come from the CASCI's core Fock operator). Classes slice the h1e_v_* here
// along the same external index as their slabs.
struct ActiveIntegralsDevice {
  DeviceTensor h1e, h2e, e_core, e_virt;
  DeviceTensor h1e_v_Sr;   // (v, a)
  DeviceTensor h1e_v_Si;   // (a, c)
  DeviceTensor h1e_v_Sir;  // (v, c)
};

// The external two-electron blocks, by the NevptIntegralsDevice field each
// one is (same physicist layout). SrsiT is h2e_v_Srsi again, batched along
// its SECOND axis: Srsi contracts h[r,s,..] against h[s,r,..], so a batch of
// r needs both h[r_batch,:,..] and h[:,r_batch,..].
enum class ExtBlock { Sr, Si, Sijrs, Sijr, Srsi, SrsiT, Srs, Sij, Sir1, Sir2 };

// The axis of each block that a slab restricts. Chosen so that every output
// element of the class depends on exactly one index along it.
//   Sr   (v,a,a,a) 0   Si   (a,a,c,a) 2   Sijrs (c,v,c,v) 0
//   Sijr (v,a,c,c) 0   Srsi (v,v,c,a) 0   SrsiT (v,v,c,a) 1
//   Srs  (v,v,a,a) 0   Sij  (a,a,c,c) 2   Sir1 (v,a,c,a) 0   Sir2 (v,a,a,c) 0
int64_t batchAxis(ExtBlock b);
const char* extBlockName(ExtBlock b);

// Where the external two-electron blocks come from.
class IntegralSource {
 public:
  virtual ~IntegralSource() = default;
  // Full length of `b` along batchAxis(b).
  virtual int64_t extent(ExtBlock b) const = 0;
  // `b` with batchAxis(b) restricted to [b0, b1), in `b`'s physicist layout
  // (dims as in the table above, one of them shortened). Work it enqueues
  // goes on res.stream(); memory it allocates comes from res's pool.
  //
  // The slab is a DeviceTensor that either views memory the source owns
  // (FullBlockSource) or owns a buffer built for this slab alone (the
  // density-fitted source), freed when the slab goes out of scope -- on
  // res's stream, after the kernels queued there that read it, with no host
  // sync.
  virtual DeviceTensor slab(ExtBlock b, int64_t b0, int64_t b1, const DeviceResources& res) = 0;
};

// Slabs as views into already-uploaded full blocks (the integral-direct path).
// `ints` must outlive every slab.
class FullBlockSource final : public IntegralSource {
 public:
  explicit FullBlockSource(const NevptIntegralsDevice& ints) : ints_(ints) {}
  int64_t extent(ExtBlock b) const override;
  DeviceTensor slab(ExtBlock b, int64_t b0, int64_t b1, const DeviceResources& res) override;
  const DeviceTensor& block(ExtBlock b) const;

 private:
  const NevptIntegralsDevice& ints_;
};

// All eight SC-NEVPT2 class energies + the total, computed on the GPU.
// `dm3`/`f3ac`/`f3ca` are expected already device-resident (the tiled RDM
// build's own output -- see nevpt2.rdm_build); `dm1`/`dm2` are uploaded once
// by the caller (they're cheap host-computed inputs from the golden file, not
// part of the GPU RDM build). All eight classes run on res.stream(), one after
// another, and allocate their intermediates from res's pool; a class's
// intermediates are freed when it returns, so they do not pile up across
// all eight classes. (They used to
// take one stream each, `--streams`; the overlap that was for never happened
// -- see docs/performance.md, "What did not help".) Every class reads its result
// back with downloadTensor, which synchronizes the stream, so all work is
// complete when this returns.
//
// `batch` bounds every slab's length along its batch axis (0 = no bound: one
// slab per block, the whole thing).
EnergyResult energiesDevice(const ActiveIntegralsDevice& active,
                             IntegralSource& source, int64_t batch, const DeviceTensor& dm1,
                             const DeviceTensor& dm2, const DeviceTensor& dm3,
                             const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                             const DeviceResources& res);

// The integral-direct entry point: full blocks, one slab each.
EnergyResult energiesDevice(const NevptIntegralsDevice& ints,
                             const DeviceTensor& dm1, const DeviceTensor& dm2,
                             const DeviceTensor& dm3, const DeviceTensor& f3ac,
                             const DeviceTensor& f3ca, const DeviceResources& res);

// --- the forward tangent (energy_sc_tangent.cppm) ---------------------------
//
// The directional derivative of the SC class energies with respect to the
// RDMs: energiesTangentDevice takes the RDMs AND a second set of tensors of
// the same shapes -- the tangents dm1_dot / dm2_dot / dm3_dot / f3ac_dot /
// f3ca_dot -- and returns each class's norm and energy together with the
// derivative of each along that direction. With the RDM tangents from a CI
// direction u (nevpt2.rdm_host_dm's dm12Dot and nevpt2.rdm_build's
// buildRdmTangentsDevice), `totalDot` is Edot(u), the forward-mode derivative
// of the second-order energy along u -- which is what the reverse-mode
// (adjoint) assembly is checked against.
//
// NOTHING IS DIFFERENTIATED BY HAND HERE EXCEPT THE SCALAR FINISH. At fixed
// integrals every class's per-perturber N and H is LINEAR in the RDMs (each
// is the external integral slab contracted twice against ONE metric or
// Dyall-Hamiltonian block, and every make_* below is linear in the RDMs up to
// a handful of RDM-free terms), so N_dot and H_dot are the SAME einsum chains
// with the tangent blocks substituted -- and the tangent block is the same
// make_* called with the tangent RDMs. The three builders that carry an
// RDM-free term (make_hdm1's 2 delta, make_hdm2's two delta-delta terms and
// make_a25's 2 h1e^T), and the one class term that carries one (Sir's purely
// one-electron norm term), drop it in the tangent; Sijrs contracts no RDM at
// all, so its tangent is exactly zero. The only derivative taken is
// nevpt2.energy_finish's normToEnergyTangent, the scalar finish's.
//
// THE ORBITAL ENERGIES ARE AN OPTIONAL SECOND INPUT (#87). e_core / e_virt
// are the generalized Fock's diagonal, built from the active 1-RDM, so in the
// complete functional they move along a CI direction too. Handed their
// tangent (`epsDot`, an OrbitalEnergyDot -- nevpt2.gradient_plan's
// orbitalEnergyDot builds it from dm1_dot and the gradient sidecar), every
// class adds sum_k R_k Deltadot_k (normToEnergyDeltaTangent), Deltadot being
// the class's own Delta construction read at that tangent -- Sijrs included,
// whose Delta is four orbital energies. Null holds them fixed: the functional
// the golden's finished e_core / e_virt define, which is what a case with no
// sidecar can differentiate (docs/gradients.md, Sec. 5.5).
struct EnergyTangentResult {
  std::array<double, 8> norms{};        // as EnergyResult: sum_k N_k per class
  std::array<double, 8> energies{};     // as EnergyResult
  std::array<double, 8> normDots{};     // d/dt of each class norm
  std::array<double, 8> energyDots{};   // d/dt of each class energy
  double total = 0.0;                   // sum of `energies`
  double totalDot = 0.0;                // sum of `energyDots`: Edot
};

// Everything energiesDevice does, plus the tangent, from the same slab walk:
// one stream (res.stream()), every intermediate from res's pool and freed
// when its class returns, every read-back a downloadTensor (which
// synchronizes that stream), so all work is complete when this returns. The
// `*Dot` tensors have their plain counterparts' shapes and are read, never
// written. `batch` bounds every slab as in energiesDevice.
//
// `norms`/`energies`/`total` are energiesDevice's own numbers for the same
// state, recomputed here from the same chains -- so a caller can check the
// two against each other for free.
EnergyTangentResult energiesTangentDevice(const ActiveIntegralsDevice& active,
                                          IntegralSource& source, int64_t batch,
                                          const DeviceTensor& dm1, const DeviceTensor& dm2,
                                          const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                          const DeviceTensor& f3ca, const DeviceTensor& dm1Dot,
                                          const DeviceTensor& dm2Dot, const DeviceTensor& dm3Dot,
                                          const DeviceTensor& f3acDot,
                                          const DeviceTensor& f3caDot,
                                          const OrbitalEnergyDot* epsDot,
                                          const DeviceResources& res);

// The integral-direct entry point: full blocks, one slab each.
EnergyTangentResult energiesTangentDevice(const NevptIntegralsDevice& ints,
                                          const DeviceTensor& dm1, const DeviceTensor& dm2,
                                          const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                          const DeviceTensor& f3ca, const DeviceTensor& dm1Dot,
                                          const DeviceTensor& dm2Dot, const DeviceTensor& dm3Dot,
                                          const DeviceTensor& f3acDot,
                                          const DeviceTensor& f3caDot,
                                          const OrbitalEnergyDot* epsDot,
                                          const DeviceResources& res);

// --- the second directional derivative (energy_sc_hessian.cppm, #153) -------
//
// Eddot(u, v), the SC class energies' second derivative with respect to the
// CI vector along two directions, at the golden's FIXED orbital energies --
// docs/hessian-theory.md, Sec. 4.1's two terms, per perturber k:
//
//   Eddot_k = P_k N''_k + Q_k H''_k                        (first sum)
//           + (Ndot_k[v], Hdot_k[v], 0) . Hess f_k . (Ndot_k[u], Hdot_k[u], 0)
//
// N''/H'' are the same chains as N/H at the second-order RDMs
// d^2(dm)/dc^2 [u, v] (nevpt2.rdm_build's RdmHessianTangents::uv, and
// nevpt2.rdm_host_dm's dm12Dot(u, v) for dm1/dm2), built with the tangent
// builders: every make_* is linear in the RDMs, and the RDM-free terms have
// N'' = H'' = 0 as they have Ndot = Hdot = 0. The 3x3 is nevpt2.energy_finish's
// normToEnergyHessian; the two first-order tangents are energiesTangentDevice's
// own, formed here side by side because the second sum needs both at once.
// NO NEW CONTRACTION AND NO NEW KERNEL: four RDM sets through one slab walk.
//
// THE ORBITAL ENERGIES ARE HELD FIXED (Deltadot = Delta'' = 0): the
// functional --sc-tangent licenses, and the one in which the Euler identity
// Eddot(c, u) == Edot(u) holds for the three homogeneous classes
// (docs/hessian-theory.md, Sec. 4.3). Their response is later work.

// One RDM set, as the energy's class walks read it: the plain RDMs, a
// first-order tangent set, or the second-order one. References to tensors the
// caller owns, which must outlive the call they are passed to.
struct RdmTensors {
  const DeviceTensor& dm1;
  const DeviceTensor& dm2;
  const DeviceTensor& dm3;
  const DeviceTensor& f3ac;
  const DeviceTensor& f3ca;
};

struct EnergyHessianResult {
  std::array<double, 8> norms{};        // as EnergyResult
  std::array<double, 8> energies{};     // as EnergyResult
  std::array<double, 8> normDotsU{};    // Ndot along u, per class
  std::array<double, 8> energyDotsU{};  // Edot(u), per class
  std::array<double, 8> normDotsV{};    // Ndot along v
  std::array<double, 8> energyDotsV{};  // Edot(v)
  std::array<double, 8> normDDots{};    // sum_k N''_k[u, v]: the class norm's second derivative
  std::array<double, 8> energyDDots{};  // Eddot(u, v), per class
  double total = 0.0;                   // sum of `energies`
  double totalDotU = 0.0;               // Edot(u)
  double totalDotV = 0.0;               // Edot(v)
  double totalDDot = 0.0;               // Eddot(u, v)
};

// Everything energiesTangentDevice does along u AND along v, plus the second
// derivative, from ONE slab walk carrying the three tangent sets: `rdm` the
// plain RDMs, `dotU` / `dotV` the first-order tangents along u / v, `ddot`
// the second-order set at (u, v). One stream (res.stream()), every
// intermediate from res's pool and freed when its class returns (four block
// sets live per class, against the tangent's two), every read-back a
// downloadTensor, so all work is complete when this returns. `batch` bounds
// every slab as in energiesDevice. Sijrs reads no RDM, so at fixed orbital
// energies its every derivative is exactly zero.
EnergyHessianResult energiesHessianDevice(const ActiveIntegralsDevice& active,
                                          IntegralSource& source, int64_t batch,
                                          const RdmTensors& rdm, const RdmTensors& dotU,
                                          const RdmTensors& dotV, const RdmTensors& ddot,
                                          const DeviceResources& res);

// --- the class metrics and Dyall Hamiltonians -------------------------------
//
// The RDM-only intermediates every class energy is built from: each class
// contracts its external integral slab against ONE of these per term -- a
// metric (the norm side, pure RDM) or a Dyall-Hamiltonian block (the H side,
// RDMs against the active integrals), PySCF's make_* by the same names. SC
// (energy.cpp, energy_sc_classes.cppm) and PC (energy_pc.cpp) build them from
// these same functions rather than a copy, and so does the SC gradient's
// pseudodensity assembly (nevpt2.gradient), which differentiates exactly the
// einsums the class energies evaluate -- hence exported rather than hidden in
// the :shared partition they used to live in. All on res.stream(), each
// result drawn from res's pool and owned by the caller.
//
// These three are (n, n): PySCF's make_hdm1 / make_a3 / make_k27.
DeviceTensor make_hdm1(const DeviceTensor& dm1, const DeviceResources& res);
DeviceTensor make_a3(const DeviceTensor& h1e, const DeviceTensor& h2e, const DeviceTensor& dm1,
                     const DeviceTensor& dm2, const DeviceTensor& hdm1,
                     const DeviceResources& res);
DeviceTensor make_k27(const DeviceTensor& h1e, const DeviceTensor& h2e,
                      const DeviceTensor& dm1, const DeviceTensor& dm2,
                      const DeviceResources& res);
// For Srs, Sij, Sir: (n, n, n, n) unless noted, PySCF's make_hdm2,
// make_hdm3 (n^6), make_a7 ({rm2, a7}), make_a9, make_a12, make_a13.
DeviceTensor make_hdm2(const DeviceTensor& dm1, const DeviceTensor& dm2,
                       const DeviceResources& res);
DeviceTensor make_hdm3(const DeviceTensor& dm1, const DeviceTensor& dm2,
                       const DeviceTensor& dm3, const DeviceTensor& hdm1,
                       const DeviceTensor& hdm2, const DeviceResources& res);
std::pair<DeviceTensor, DeviceTensor> make_a7(const DeviceTensor& h1e, const DeviceTensor& h2e,
                                              const DeviceTensor& dm1, const DeviceTensor& dm2,
                                              const DeviceTensor& dm3,
                                              const DeviceResources& res);
// make_a7's own (n)^6 hole-side 3-RDM, which it builds and does not hand back:
// its own function because nevpt2.gradient differentiates a7 with respect to
// it (Park Eqs. 46-47). `rm2` is make_a7's first return value.
DeviceTensor make_rm3(const DeviceTensor& dm2, const DeviceTensor& dm3, const DeviceTensor& rm2,
                      const DeviceResources& res);
DeviceTensor make_a9(const DeviceTensor& h1e, const DeviceTensor& h2e,
                     const DeviceTensor& hdm1, const DeviceTensor& hdm2,
                     const DeviceTensor& hdm3, const DeviceResources& res);
DeviceTensor make_a12(const DeviceTensor& h1e, const DeviceTensor& h2e,
                      const DeviceTensor& dm1, const DeviceTensor& dm2,
                      const DeviceTensor& dm3, const DeviceResources& res);
DeviceTensor make_a13(const DeviceTensor& h1e, const DeviceTensor& h2e,
                      const DeviceTensor& dm1, const DeviceTensor& dm2,
                      const DeviceTensor& dm3, const DeviceResources& res);
// For Sr, Si: PySCF's make_a16 / make_a22 ((n)^6, `pqrabc` / `ijkabc`; they
// read the f3ac/f3ca digests) and the hole-side metrics energy_Si builds,
// dm3_h (n^6) = 2 dm2[a,b,e,f] delta[c,d] - dm3[a,b,d,c,e,f] and dm2_h (n^4)
// = 2 dm1[a,b] delta[c,d] - dm2[a,b,d,c].
DeviceTensor make_a16(const DeviceTensor& h1e, const DeviceTensor& h2e, const DeviceTensor& dm3,
                      const DeviceTensor& f3ac, const DeviceTensor& f3ca, int64_t norb,
                      const DeviceResources& res);
DeviceTensor make_a22(const DeviceTensor& h1e, const DeviceTensor& h2e, const DeviceTensor& dm2,
                      const DeviceTensor& dm3, const DeviceTensor& f3ac,
                      const DeviceTensor& f3ca, int64_t norb, const DeviceResources& res);
DeviceTensor make_dm3_h(const DeviceTensor& dm2, const DeviceTensor& dm3,
                        const DeviceResources& res);
DeviceTensor make_dm2_h(const DeviceTensor& dm1, const DeviceTensor& dm2,
                        const DeviceResources& res);
// SC-only, the rest of Sr's and Si's Dyall-Hamiltonian blocks (defined in
// energy_sc_classes.cppm): make_a17 (n^4, `abcp`) / make_a19 (n^2, `ap`) are
// Sr's, make_a23 / make_a25 Si's.
DeviceTensor make_a17(const DeviceTensor& h1e, const DeviceTensor& h2e, const DeviceTensor& dm2,
                      const DeviceTensor& dm3, const DeviceResources& res);
DeviceTensor make_a19(const DeviceTensor& h1e, const DeviceTensor& h2e, const DeviceTensor& dm1,
                      const DeviceTensor& dm2, const DeviceResources& res);
DeviceTensor make_a23(const DeviceTensor& h1e, const DeviceTensor& h2e, const DeviceTensor& dm1,
                      const DeviceTensor& dm2, const DeviceTensor& dm3,
                      const DeviceResources& res);
DeviceTensor make_a25(const DeviceTensor& h1e, const DeviceTensor& h2e, const DeviceTensor& dm1,
                      const DeviceTensor& dm2, const DeviceResources& res);

// --- PC-NEVPT2 (energy_pc.cpp) ----------------------------------------------

// PC_TAU and PC_MIN_GAP (S's cut and the gap it must clear), PcStatus,
// PcSpectrum, PcClassResult (one class's PC result) and pcRankCut (the cut
// and gap decision itself) are nevpt2.energy_finish's, re-exported above.

struct PcEnergyResult {
  std::array<PcClassResult, 8> classes;  // CLASSES order
};

// The PC class energies, from the same inputs as energiesDevice: everything on
// res.stream(), intermediates from res's pool and freed when each class
// returns, the eigensolves on res.solver(). Every class reads its result back
// with downloadTensor (which synchronizes the stream), so all work is complete
// when this returns. `batch` bounds every slab as in energiesDevice.
// f3ac/f3ca are read by Sr's and Si's K (make_a16 / make_a22).
// A class refused by the gap or denominator check is a value (its
// PcClassResult's status); an eigensolve that reports devInfo != 0 is a
// Numerical Error -- the first one stops the stage and is
// returned unchanged, for main() to report once.
Result<PcEnergyResult> pcEnergiesDevice(const ActiveIntegralsDevice& active,
                                        IntegralSource& source, int64_t batch,
                                        const DeviceTensor& dm1, const DeviceTensor& dm2,
                                        const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                        const DeviceTensor& f3ca, const DeviceResources& res);

// The integral-direct entry point: full blocks, one slab each.
Result<PcEnergyResult> pcEnergiesDevice(const NevptIntegralsDevice& ints, const DeviceTensor& dm1,
                                        const DeviceTensor& dm2, const DeviceTensor& dm3,
                                        const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                                        const DeviceResources& res);

// Prints the per-class PC table (S's spectrum and gap, energy, block2's
// `pc_class_energies` reference `ref` -- eight, CLASSES order -- and the
// difference), the summed total against block2's `e_pc_total` `refTotal`, and
// the gate line the `*_pc` ctest entries match: "PASS: PC-NEVPT2
// ..." or "FAIL: PC-NEVPT2 ...". Returns true -- and prints PASS -- only when
// all eight classes were computed (none refused, none not yet), each is within
// `atol` of `ref`, and the total is within `atol` of `refTotal`.
bool printPcReport(const PcEnergyResult& r, const Tensor& ref, double refTotal, double atol);

// --- PC-NEVPT2's tangent ------------------------------------------------------

// The eight classes' tangents, CLASSES order. PcClassTangent is
// nevpt2.energy_finish's: each carries the class's own PcClassResult (status,
// energy, spectrum) beside `edot`, so a refused class is refused here too and
// its `edot` is meaningless by construction rather than a zero that reads
// like an answer.
struct PcTangentResult {
  std::array<PcClassTangent, 8> classes;
};

// The directional derivative of pcEnergiesDevice with respect to the CI
// vector: d/dh of every class energy, along the direction whose RDM tangents
// are `dm1Dot`...`f3caDot` (nevpt2.rdm_host_dm's dm12Dot for the first two,
// nevpt2.rdm_build's buildRdmTangentsDevice for the other three -- the
// tangents of exactly the arrays passed as `dm1`...`f3ca`). The energies
// themselves come back too, from the same solves, so the derivative and the
// number it differentiates are never two different runs.
//
// Why this is legitimate -- and why a class the gap check refuses stays
// refused -- is the module header above, "PC'S TANGENT DIFFERENTIATES THE
// SOLVE". The integrals are held FIXED: this is the CI half of the
// derivative, the source term the Z-vector work needs, not a nuclear
// gradient.
//
// THE ORBITAL ENERGIES ARE AN OPTIONAL INPUT (#74), as for
// energiesTangentDevice (#87): handed their tangent `epsDot` (e_dot =
// <dm1_dot, W>, nevpt2.gradient_plan's orbitalEnergyDot), every class adds
// sum_t R_t Deltadot_t with R_t = dE_t/dDelta_t read off the same solve
// (pcDeltaTangentSingle for the single-coupling classes, the same sum inline
// for Sijr/Srsi), and Sijrs its SC Delta term; null holds them fixed at the
// golden's.
//
// Everything runs on res.stream(), each class's intermediates drawn from
// res's pool and freed when it returns, the eigensolves on res.solver() and
// the d x d products on res.blas(), exactly as pcEnergiesDevice does; every
// read-back is a downloadTensor (which synchronizes), so all work is complete
// when this returns. `batch` bounds every slab as in energiesDevice. An
// eigensolve that reports devInfo != 0 is a Numerical Error -- the first one
// stops the stage.
//
// Sijrs's tangent is EXACTLY zero (status Done): the MP2-like class contracts
// no RDM at all, so no CI direction can move it.
Result<PcTangentResult> pcTangentsDevice(const ActiveIntegralsDevice& active,
                                         IntegralSource& source, int64_t batch,
                                         const DeviceTensor& dm1, const DeviceTensor& dm2,
                                         const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                         const DeviceTensor& f3ca, const DeviceTensor& dm1Dot,
                                         const DeviceTensor& dm2Dot, const DeviceTensor& dm3Dot,
                                         const DeviceTensor& f3acDot,
                                         const DeviceTensor& f3caDot,
                                         const OrbitalEnergyDot* epsDot,
                                         const DeviceResources& res);

// The integral-direct entry point: full blocks, one slab each.
Result<PcTangentResult> pcTangentsDevice(const NevptIntegralsDevice& ints,
                                         const DeviceTensor& dm1, const DeviceTensor& dm2,
                                         const DeviceTensor& dm3, const DeviceTensor& f3ac,
                                         const DeviceTensor& f3ca, const DeviceTensor& dm1Dot,
                                         const DeviceTensor& dm2Dot, const DeviceTensor& dm3Dot,
                                         const DeviceTensor& f3acDot,
                                         const DeviceTensor& f3caDot,
                                         const OrbitalEnergyDot* epsDot,
                                         const DeviceResources& res);

// --- one PC class's matrices and solve, for the adjoint (#74) -----------------
//
// nevpt2.gradient's PC reverse mode differentiates the SAME solve the energy
// and the tangent run, so it needs the class's S, K and S_1 and the solve's
// eigenbasis -- built here by the very functions pcTangentsDevice builds them
// with (energy_pc_tangent.cppm's pcMatrices*, energy_pc_solve.cppm's
// solveClass), not by a copy. Exported for that one caller.

// The active-space RDMs a class's matrices are built from: non-owning, the
// caller owns every tensor (dm1/dm2 its uploads, dm3/f3ac/f3ca the RDM
// build's output -- energiesDevice's inputs).
struct PcRdms {
  const DeviceTensor* dm1 = nullptr;
  const DeviceTensor* dm2 = nullptr;
  const DeviceTensor* dm3 = nullptr;
  const DeviceTensor* f3ac = nullptr;
  const DeviceTensor* f3ca = nullptr;
};

// One class's three matrices, each in its natural PySCF index shape and
// owning its buffer; `matS`/`matK`/`matS1` are the flattened (d, d) /
// (d, n1) VIEWS the solve takes (reshapeView: no copy, valid only while this
// struct lives). docs/pc-nevpt2.md, "Per class", has every class's layout.
struct PcMatrices {
  DeviceTensor S;
  DeviceTensor K;
  DeviceTensor S1;        // empty unless hasS1
  int64_t d = 0;          // the flattened basis dimension
  int64_t n1 = 0;         // S_1's column count (0 unless hasS1)
  bool hasS1 = false;

  DeviceTensor matS() const { return reshapeView(S, {d, d}); }
  DeviceTensor matK() const { return reshapeView(K, {d, d}); }
  DeviceTensor matS1() const { return reshapeView(S1, {d, n1}); }
};

// Class `cls` (CLASSES order; Sijrs, which has no active index, is not one
// -- asking for it is our bug) built from `rdm` against the active integrals
// `h1e` / `h2e`. Every matrix is AFFINE in the RDMs and K LINEAR in
// (h1e, h2e) apart from Sr's and Si's f3ac/f3ca addends, so K at dotted
// integrals with zero digests IS K's tangent along an orbital rotation at
// fixed digests -- the use nevpt2.gradient makes of it.
PcMatrices pcClassMatrices(int cls, const DeviceTensor& h1e, const DeviceTensor& h2e,
                           const PcRdms& rdm, const DeviceResources& res);

// One class's solve, kept for differentiating: solveClass's own output with
// its eigenvectors. `result` is pcRankCut's verdict -- a class refused by the
// gap is a value (status Refused, `why`), exactly as pcEnergiesDevice reports
// it, and nothing else here is filled; otherwise `lambda` (m), `T` (d, m) =
// S_sym U, `t1` (n1, m) = S_1^T U when the class has an S_1, and `Ut` (m, d)
// = U^T. The denominator check is NOT made here: it needs the tuples'
// Delta, which is the caller's slab walk.
struct PcClassSolve {
  PcClassResult result;
  std::vector<double> lambda;
  DeviceTensor T;
  DeviceTensor t1;
  DeviceTensor Ut;
};

// solveClass on `mx` (its S, K and, when it has one, S_1), named `name` in a
// devInfo error. On res.stream(); an eigensolve's devInfo != 0 is the
// Numerical Error pcEnergiesDevice returns for it.
Result<PcClassSolve> pcSolveClass(const PcMatrices& mx, const char* name,
                                  const DeviceResources& res);

}  // namespace nevpt2
