// nevpt2.cli -- the command line of every binary that reads a golden file,
// parsed in one place.
//
// parseArgs(binary, args) is the entry point each main() calls. It walks argv
// against ONE table of every flag any of the four binaries takes and which of
// them implements it (flagTable(): the binary/flag matrix README.md records).
// A flag no binary has is a usage error (err_config); a flag another binary
// implements but this one does not is REFUSED (err_unsupported) -- never
// parsed and silently ignored, which would make a run look like it did
// something it did not. The common flags go through parseCommonFlag, the
// binary's own into its fields of Options; finalize() applies the checks that
// span the common flags (the digest conflicts, --cublas on a build without
// it), and parseArgs then the binary's own. Every bad argument is a
// value-tier Error (nevpt2.error_handling) for main() to report once; nothing
// here prints or exits.
export module nevpt2.cli;

import std;
// Re-exported: every entry point hands back a Result/Status, and
// CommonOptions holds an RdmBuildOptions.
export import nevpt2.error_handling;
export import nevpt2.rdm_build;
import nevpt2.device_resources;

export namespace nevpt2::cli {

// The flags every binary takes (flagTable() says which binary implements the
// few that are not every binary's: --pc, --profile, --rdm-tangent).
struct CommonOptions {
  std::string goldenPath;
  RdmBuildOptions rdm;
  bool profile = false;
  bool pc = false;
  // --rdm-tangent: after the RDM build, run the TANGENT build
  // (buildRdmTangentsDevice) along the state itself and check it against twice
  // the build -- Euler's identity for a form that is quadratic in the CI
  // vector (checkRdmTangentsDevice). It holds about twice the per-tile
  // intermediates, so it has tile floors of its own, and it is what they were
  // measured with (docs/performance.md, "Tile floors"). finalize() refuses it
  // with the digests it has no tangent path for.
  bool rdmTangent = false;
  std::uint64_t poolThreshold = kDefaultPoolReleaseThreshold;
  // --mantissa-bits was given: finalize() refuses it without --cublas, the
  // only digest that reads it (rdm.mantissaBits alone cannot tell an
  // explicit 53 from the default).
  bool mantissaBitsGiven = false;
  // --ozaki-pairs was given: finalize() refuses it without --ozaki, the only
  // digest that reads it (an explicit 14 is the default too).
  bool ozakiPairsGiven = false;
};

// An integer flag's value (--tiles, --ozaki-pairs, --mantissa-bits, --batch,
// ...): a whole integer that fits T, or a config Error naming the flag and the
// argument (std::atoi/atoll silently read garbage as 0). A range
// is checked after parsing, in finalize() (or parseArgs, for a binary's own).
template <class T>
Result<T> parseInteger(const std::string_view flag, const std::string_view arg) {
  T v{};
  const auto [end, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), v);
  if (ec != std::errc{} || end != arg.data() + arg.size())
    return err_config(std::format("{} takes an integer, not \"{}\"", flag, arg));
  return v;
}

// Offers args[i] to the common flags. true: it was one, and i now indexes its
// last argument (the value, for a flag that takes one). false: not one of
// them -- the binary's own flag, or a usage error; a flag that takes a value
// but has none after it is false too. An Error: it was one, and its value is
// bad. It does not consult the binary/flag matrix: parseArgs does, first.
Result<bool> parseCommonFlag(std::span<const std::string_view> args, std::size_t& i,
                             CommonOptions& opt);

// The checks that span flags, once every argument is in: --golden given,
// --tiles >= 1, --ozaki-pairs in 0..14, --ozaki-pairs and --ozaki-check only
// with --ozaki, --mantissa-bits in 1..53 and only with --cublas, --cublas
// only where the emulated digest is built, at most one of --cublas /
// --blas-digest / --ozaki, and --rdm-tangent only with a digest the tangent
// build has (the emitted one or --blas-digest). Also makes --ozaki take
// precedence over blasDigest's HIP default. `goldenHint` names the file in
// the missing-golden message ("path.nevpt2gold").
Status finalize(CommonOptions& opt, std::string_view goldenHint);

// --- the four binaries, and which flags each implements ---------------------

// The binaries whose command line parseArgs parses (the sanitizer canary
// reads no golden and parses nothing here).
enum class Binary : std::uint8_t { Demo, DfDemo, ScPseudodensity, PcTangent };
inline constexpr std::array kBinaries{Binary::Demo, Binary::DfDemo, Binary::ScPseudodensity,
                                      Binary::PcTangent};

// The executable's name: "nevpt2_demo", "nevpt2_df_demo",
// "nevpt2_sc_pseudodensity", "nevpt2_pc_tangent".
std::string_view binaryName(Binary b);
// The golden file the usage line and the missing-golden message name.
std::string_view goldenHint(Binary b);

// One row of the binary/flag matrix.
struct FlagSpec {
  std::string_view name;       // "--tiles"
  std::string_view valueName;  // "N"; empty for a switch
  std::uint8_t binaries;       // bit (1 << Binary) set: that binary implements it
};

// Every flag any binary takes, in usage-line order. --golden is in it (every
// binary's), and so is every digest flag: every binary builds the RDMs
// through RdmBuildOptions, and a tangent path refusing an approximate digest
// is a combination it cannot run (err_config), not a flag it lacks.
std::span<const FlagSpec> flagTable();
// The row for `name`, or nullptr when no binary takes it.
const FlagSpec* findFlag(std::string_view name);
// Whether `b` implements `flag`; false for a flag no binary takes.
bool implements(Binary b, std::string_view flag);

// --- each binary's own flags, and their ranges -------------------------------

// nevpt2_df_demo --batch: slab length along each block's batch index. A
// MEMORY lever, chosen to force several slabs on the blocks batched over a
// virtual (nvirt 15..48 on the committed cases); those batched over a core
// index (Si, Sijrs, Sij; ncore 1..5) fit in one -- not a measured speed
// optimum. 0 = one slab per block.
inline constexpr std::int64_t kDefaultBatch = 8;

// nevpt2_pc_tangent's finite-difference sweep: fdSteps successive halvings of
// 2^-fdStartPow.
//
// WHY THE LARGEST STEP IS SMALL. The displaced state is not a CASCI
// eigenstate, and its PC metric grows modes the state's own does not have.
// The direction is chosen so that the ones scaling as h^2 never appear at all
// (nevpt2.pc_tangent's `pcTangentDirection`), but Sir still grows one at ~h^4
// relative, with a Dyall eigenvalue that makes lambda_k + Delta_t negative --
// so the PC stage refuses the class, rightly: the energy at that state does
// not exist and neither does a difference quotient built from it. Below
// h ~ 2^-10 that mode falls under PC_TAU = 1e-13 * s_max and is dropped like
// the structural ones, so every class comes back with the state's own dropped
// count. 2^-12 is where it is run, comfortably inside that and still leaving
// the h^2 truncation error ~3e-6 Eh -- five decades above the difference's own
// noise floor (~1e-11 Eh, the RDM build's reassociation divided by 2h), so the
// Richardson ratios measure truncation and nothing else (docs/pc-nevpt2.md,
// "The tangent with respect to the CI vector"). 2^-1 is already far past what
// the denominator condition allows; past 2^-30 the quotient is all rounding.
inline constexpr int kDefaultFdStartPow = 12;
inline constexpr int kMinFdStartPow = 1;
inline constexpr int kMaxFdStartPow = 30;
// Four steps is the default; each costs two more states -- two more RDM
// builds and PC stages -- which at 12 orbitals is minutes. Never fewer than
// three: the report gates on a RUN of nevpt2::kPcMinFourfoldRun = 2 fourfold
// halvings, not on one step (docs/testing.md, "Residual-independent checks of
// the gradient path"); this module does not link nevpt2.pc_tangent, so
// apps/pc_tangent static_asserts that the two agree. The upper bound is only
// there so a typo cannot queue hours of RDM builds.
inline constexpr int kDefaultFdSteps = 4;
inline constexpr int kMinFdSteps = 3;
inline constexpr int kMaxFdSteps = 8;

// Everything a binary's command line can set. The common flags are in
// `common`; each binary reads only its own fields below, and parseArgs
// refuses another binary's flag before it could set one.
struct Options {
  CommonOptions common;
  // Both demos: --sc-tangent, --gradient-source (one at a time; SC only,
  // except nevpt2_demo's --gradient-source --pc); nevpt2_demo also takes
  // --gradgold PATH beside --gradient-source, --sc-hessian or
  // --sc-hessian-vector (the field below).
  bool scTangent = false;
  bool gradientSource = false;
  // nevpt2_demo alone: --sc-hessian, the SC energy's second directional
  // derivative Eddot(u, v) (#153); one at a time with the two above, SC only.
  bool scHessian = false;
  // nevpt2_demo alone: --sc-hessian-vector, the vector H_cc u by reverse over
  // forward (#156); one at a time with the three above, SC only.
  bool scHessianVector = false;
  // nevpt2_df_demo: --batch N (>= 0), --check-blocks.
  std::int64_t batch = kDefaultBatch;
  bool checkBlocks = false;
  // nevpt2_sc_pseudodensity: --kappa-act, --kappa-full, --dump-source-terms
  // PATH, --source-oracle PATH (one at a time), --gradgold PATH (only with
  // --kappa-full, --dump-source-terms or --source-oracle, the three that read
  // the sidecar). nevpt2_demo reads gradgoldPath too, with --gradient-source,
  // --sc-hessian or --sc-hessian-vector (the last two then in FULL mode,
  // #214), and nevpt2_pc_tangent with --pc-hessian or --pc-hessian-vector
  // (FULL mode).
  bool kappaAct = false;
  bool kappaFull = false;
  // nevpt2_sc_pseudodensity: --sc-hessian-kc, the SC energy's mixed
  // orbital-CI Hessian block H_kc in both its orientations (#184); one job at
  // a time with the four above, exact digests only; with --pc, PC's mixed
  // block (#201). It reads the sidecar when there is one (--gradgold PATH,
  // or `<case>.gradgold`).
  bool scHessianKc = false;
  // nevpt2_sc_pseudodensity: --sc-hessian-kk-int, the first term of the SC
  // energy's orbital Hessian block, the integrals' own curvature
  // <Ehat', I''> = K1 + K2 (#154); one job at a time with the five above,
  // exact digests only; with --pc, PC's <Ehat'_pc, I''> (#200). It reads the
  // sidecar (required).
  bool scHessianKkInt = false;
  // nevpt2_sc_pseudodensity: --sc-hessian-kk, the second term of the same
  // block, the energy's own curvature I'^T Ehat'' I' = K3 + K4 + K5, and the
  // whole block Eddot_kk(u, v) against ladders of the SC total (#158); one job
  // at a time with the six above, exact digests only; with --pc, PC's
  // sandwich and Eddot_kk (#202). It reads the sidecar (required).
  bool scHessianKk = false;
  // nevpt2_sc_pseudodensity: --sc-source-tangent PATH, the SC source terms'
  // tangent along the integral perturbations of the `.pertgold` at PATH
  // (#220; docs/hessian-theory.md, Sec. 8); one job at a time with the seven
  // above, exact digests only (a second derivative, so the Hessian flags'
  // refusal), SC only. It reads the gradient sidecar (required).
  std::string scSourceTangentPath;
  std::string gradgoldPath;
  std::string dumpSourceTermsPath;
  // nevpt2_sc_pseudodensity: --probed-y, only with --dump-source-terms (#240):
  // also probe Y basis direction by basis direction and hold the one-pass Y
  // the dump writes to it, element by element.
  bool probedY = false;
  std::string sourceOraclePath;
  // nevpt2_pc_tangent: --fd-steps N, --fd-start-pow N.
  int fdSteps = kDefaultFdSteps;
  int fdStartPow = kDefaultFdStartPow;
  // nevpt2_pc_tangent: --pc-hessian, the PC energy's SECOND directional
  // derivative Eddot(u, v) in place of the tangent check (#152); the same
  // finite-difference steps, now of Edot.
  bool pcHessian = false;
  // nevpt2_pc_tangent: --pc-hessian-vector, the VECTOR H_cc u by reverse
  // over forward (#155), against --pc-hessian's Eddot by duality and a
  // finite difference of the three-term y; the same steps, now of y.
  bool pcHessianVector = false;
};

// Parses `args` (argv without argv[0]) for binary `b`, then finalize()s the
// common flags and checks the binary's own. An argument no binary takes, or a
// valued flag with no value: err_config, the message carrying the usage line.
// A flag `b` does not implement: err_unsupported, naming the binaries that
// do. A bad value or combination: err_config (err_unsupported for --cublas on
// a build without it, for --profile where nothing would print it, and for any
// Hessian flag with --cublas, --ozaki or --fused-digest, which is checked
// before the digest flags themselves; docs/hessian-theory.md, Sec. 9.1).
Result<Options> parseArgs(Binary b, std::span<const std::string_view> args);

// The usage line for `b`: its name and golden file, then every other flag it
// implements (flagTable()'s order), ending in a newline.
std::string usage(Binary b);

}  // namespace nevpt2::cli
