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
export module nevpt2.energy;

import std;
// Re-exported: energiesDevice and pcEnergiesDevice take DeviceTensors and the
// DeviceResources (nevpt2.device_resources, through nevpt2.einsum's nevpt2.device_tensor).
export import nevpt2.einsum;

export namespace nevpt2 {

// Norms below this are treated as an empty perturber space (no energy
// contribution) -- pyscf.mrpt.nevpt2's own guard against dividing by a
// vanishing norm.
inline constexpr double NUMERICAL_ZERO = 1e-14;

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

// --- PC-NEVPT2 (energy_pc.cpp) ----------------------------------------------

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

}  // namespace nevpt2
