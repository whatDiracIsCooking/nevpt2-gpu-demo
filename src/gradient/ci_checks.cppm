// nevpt2.ci_checks -- the two checks of the energy's derivative with respect
// to the CI vector that the demos run in place of the golden comparison:
//
//   checkScTangent   --sc-tangent: nevpt2.energy's forward tangent of the SC
//                    total, against the class norms' exact affine increment,
//                    Euler's identity on the three homogeneous classes, and a
//                    one-time central finite difference of the energy
//                    (ci_checks_tangent.cpp has the derivation);
//   checkCiSource    --gradient-source: nevpt2.gradient's CI source term
//                    y_I = dE/dc_I (Park Eq. 48) against that tangent by the
//                    forward-reverse duality <u, y> == Edot(u), exact, plus
//                    the column-by-column Jacobian where ndet allows
//                    (ci_checks_source.cpp);
//   checkPcCiSource  --gradient-source --pc: the same duality for PC-NEVPT2's
//                    y (#74). Integral-direct only: pcCiSourceDevice and
//                    pcTangentsDevice have no slab entry point.
//
// WHY A LIBRARY AND NOT A DEMO'S PARTITIONS. Both demos run the SC pair
// (#162): y_I and Edot(u) differentiate the CI vector at FIXED integrals, so
// nothing in them differentiates an integral, and the slab entry points
// energiesDevice / energiesTangentDevice / scCiSourceDevice already take any
// IntegralSource. nevpt2_demo hands over FullBlockSource (one slab per block),
// nevpt2_df_demo its DfIntegralSource (B-built slabs, `--batch` long) -- one
// body of checks, two integral sources. A leaf like nevpt2.pc_tangent beside
// it: nothing in src/ imports it.
//
// Everything runs on the caller's one DeviceResources: every RDM and tangent
// build, every energy, every sigma launch is issued on res.stream(), and every
// device tensor is a DeviceBuffer from res's pool, freed (on that stream) when
// it goes out of scope.
export module nevpt2.ci_checks;

import std;
// Re-exported: the entry points take the IntegralSource / ActiveIntegralsDevice
// (nevpt2.energy, which nevpt2.gradient re-exports), the RdmBuildOptions and
// RdmBuildResult (nevpt2.rdm_build, likewise) and an EnergyResult.
export import nevpt2.gradient;
// Re-exported: every check reads its state out of a GoldenFile.
export import nevpt2.golden;
// The host dm1/dm2 rebuild and its tangent along a CI direction: the checks'
// own business, so not re-exported.
import nevpt2.rdm_host_dm;

export namespace nevpt2 {

// The integrals the SC checks differentiate the energy against, as one
// IntegralSource walk: `active` and `source` exactly as energiesDevice takes
// them, `batch` its slab bound (0 = one slab per block), and `h2e` the
// physicist-ordered active block the RDM build and its tangent contract --
// the golden's own on the integral-direct demo, the DF rebuild from B_aa on
// the density-fitted one. Whichever it is must be the block `baseRdm` below
// was built from, or the displaced builds would differentiate another
// functional. Every member is borrowed: the caller keeps them alive for the
// call.
struct CiCheckIntegrals {
  const ActiveIntegralsDevice& active;
  IntegralSource& source;
  int64_t batch;
  const Tensor& h2e;
};

// --sc-tangent on the golden state `g`, along `seed`'s direction: prints the
// per-class Euler table, the norm increments and the finite-difference sweep,
// and the gate line "PASS: the SC tangent matches Euler ..." / "FAIL: ...".
// `baseRdm` is the caller's dm3/f3ac/f3ca build of g's CI vector with
// `rdmOpt`. Returns whether every criterion held; an active space the host
// RDM builder refuses is a configuration Error, for main to report once.
Result<bool> checkScTangent(const GoldenFile& g, const RdmBuildOptions& rdmOpt,
                            const CiCheckIntegrals& ints, const RdmBuildResult& baseRdm,
                            std::uint64_t seed, const DeviceResources& res);

// --gradient-source on `g`: y_I against Edot(u) along four seeded directions
// and the state itself, plus the full Jacobian where ndet allows, all to 1e-7;
// `energies` is energiesDevice's answer for the same state, which the walk y
// rides on must reproduce. `gradgoldPath` non-empty reads the orbital
// energies' response from that sidecar and both sides carry it; empty holds
// them fixed, and `fixedReason` says why on the "functional:" and PASS lines
// ("the orbital energies held fixed (<fixedReason>)"). A sidecar that cannot
// be read or does not fit `g` is a returned Error.
Result<bool> checkCiSource(const GoldenFile& g, const RdmBuildOptions& rdmOpt,
                           const CiCheckIntegrals& ints, const RdmBuildResult& baseRdm,
                           const EnergyResult& energies, const std::string& gradgoldPath,
                           std::string_view fixedReason, const DeviceResources& res);

// --gradient-source --pc on `g`, integral-direct: the PC CI source term
// against the PC forward tangent, the same three checks; the adjoint's walk
// is held to the golden's block2 e_pc_total. A refused class FAILS the run.
Result<bool> checkPcCiSource(const GoldenFile& g, const RdmBuildOptions& rdmOpt,
                             const NevptIntegralsDevice& ints, const RdmBuildResult& baseRdm,
                             const std::string& gradgoldPath, const DeviceResources& res);

}  // namespace nevpt2
