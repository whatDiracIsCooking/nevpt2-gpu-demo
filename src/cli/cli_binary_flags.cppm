// nevpt2.cli:binary_flags -- the binary/flag matrix and each binary's own
// flags: the table flagTable() exposes, the column bits it is written in, and
// the parsing and checking of the flags that are one binary's alone
// (--sc-tangent, --batch, --kappa-full, --fd-steps, ...). An internal
// partition: nothing here is exported, so cli.cpp reaches it
// (`import :binary_flags;`) and no importer of nevpt2.cli does.
module nevpt2.cli:binary_flags;

import std;
import nevpt2.cli;

namespace nevpt2::cli {

// The matrix's column bit for `b`.
constexpr std::uint8_t bitOf(const Binary b) {
  return static_cast<std::uint8_t>(1u << static_cast<unsigned>(b));
}
constexpr std::uint8_t kDemo = bitOf(Binary::Demo);
constexpr std::uint8_t kDfDemo = bitOf(Binary::DfDemo);
constexpr std::uint8_t kScPd = bitOf(Binary::ScPseudodensity);
constexpr std::uint8_t kPcTangent = bitOf(Binary::PcTangent);
constexpr std::uint8_t kDemos = kDemo | kDfDemo;
constexpr std::uint8_t kEvery = kDemo | kDfDemo | kScPd | kPcTangent;
// The digests nevpt2_pc_tangent cannot run: it is a tangent path and nothing
// else, and the tangent RDM build is the emitted digest or the native-fp64
// BLAS one. (The demos' --rdm-tangent / --sc-tangent / --gradient-source
// refuse the same digests as a COMBINATION, err_config, since there they are
// one mode of several.)
constexpr std::uint8_t kNotPcTangent = kDemos | kScPd;

// What implements what, and why a binary that does not refuses it:
//   --pc            the demos, and nevpt2_sc_pseudodensity for its two
//                   source-term jobs (--dump-source-terms and --source-oracle
//                   write / compare PC-NEVPT2's with it, #74), for
//                   --sc-hessian-kk-int (PC's orbital curvature, #200),
//                   for --sc-hessian-kc (PC's mixed block H_kc, #201) and
//                   for --sc-hessian-kk (PC's sandwich and Eddot_kk, #202); the
//                   identities and --kappa-act / --kappa-full are SC's, so it
//                   is a config error there. nevpt2_pc_tangent is PC
//                   throughout, so the switch would change nothing there.
//   --profile       the demos only: nothing else prints a profile table.
//   --rdm-tangent   the demos only: the gradient binaries run no Euler check of
//                   the tangent RDM build beside their own (#38, #39).
//   the --cublas / --ozaki family and --fused-digest: every binary but
//                   nevpt2_pc_tangent (kNotPcTangent, above). With any
//                   Hessian flag, on any binary, they are Unsupported
//                   (checkHessianDigests, below; #157).
//   --sc-tangent, --gradient-source
//                   the demos (#162): both differentiate the CI vector at
//                   fixed integrals, so the density-fitted demo runs them over
//                   its slabs. It holds the orbital energies fixed (it reads
//                   no sidecar) and has no --pc path for --gradient-source
//                   (the PC source term has no slab entry point).
//   --sc-hessian    nevpt2_demo alone (#153): Eddot(u, v), checked against
//                   Euler, the norms' increment and a finite difference of
//                   Edot. The density-fitted demo could run it over its slabs
//                   as it runs --sc-tangent; nothing has asked for that yet.
//   --sc-hessian-vector
//                   nevpt2_demo alone (#156): the vector H_cc u, checked
//                   against --sc-hessian's Eddot by duality, column by column
//                   at small ndet, and a finite difference of y. The same
//                   "nothing has asked for it yet" holds for the DF demo.
//   --sc-hessian-kc nevpt2_sc_pseudodensity alone (#184): the mixed block
//                   H_kc needs the orbital directions and the sidecar this
//                   binary owns (--kappa-full's), and its one finite
//                   difference displaces along the directions --kappa-full and
//                   --kappa-act already displace.
//   --sc-hessian-kk-int
//                   nevpt2_sc_pseudodensity alone (#154): the orbital block's
//                   integral curvature <Ehat', I''> reads the sidecar and the
//                   orbital directions this binary owns, and its ladder
//                   displaces the stored blocks as --kappa-full does.
//   --sc-hessian-kk nevpt2_sc_pseudodensity alone (#158): the orbital block's
//                   sandwich and the whole Eddot_kk, for --sc-hessian-kk-int's
//                   reasons, its ladders displacing as that flag's do.
//   --sc-source-tangent PATH
//                   nevpt2_sc_pseudodensity alone (#220): the SC source terms'
//                   tangent along a `.pertgold`'s integral perturbations --
//                   the orbital source term Y and the sidecar it is read out
//                   of are this binary's, and its ladders displace the stored
//                   blocks and the sidecar's inventory as --kappa-full's do.
//                   With --pc, PC's (#234): the same flag rather than a twin
//                   on nevpt2_pc_tangent, because Y_pc, the sidecar and the
//                   PC orbital blocks it reuses are all this binary's.
//                   Refused the approximate digests with the Hessian flags
//                   (a mixed second derivative).
//   each binary's own flags: that binary alone. --dump-source-terms is
//                   nevpt2_sc_pseudodensity's because Y_xy needs the gradient
//                   sidecar's full-MO-range arrays; nevpt2_demo computes y_I
//                   too, but half a dump is not one.
//                   --source-oracle is its too, for the same reason: it
//                   compares the dump's two terms with a committed oracle.
//                   --probed-y is its too, and rides on --dump-source-terms
//                   (#240): the probed Y, one basis direction per MO pair,
//                   as the oracle of the one-pass Y the dump writes.
//                   --pc-hessian is nevpt2_pc_tangent's (#152): PC's second
//                   directional derivative, checked by a finite difference of
//                   the PC tangent that binary already owns. Like the rest of
//                   that binary it runs the emitted or the native-fp64 BLAS
//                   digest only, which the table refuses the others for.
//                   --pc-hessian-vector is its too (#155): PC's H_cc u,
//                   gated against --pc-hessian's Eddot by duality, with the
//                   same ladder flags; the two are one check at a time.
//   --sc-parameter-hessian PATH, --parameter-hessian-out PATH
//                   nevpt2_sc_pseudodensity alone (#222): the level-1
//                   parameter Hessian as a bilinear form over a `.respgold`'s
//                   response directions, assembled from the orbital, mixed
//                   and CI blocks the tree already gates; the orbital
//                   directions, the sidecar and the full tensor are this
//                   binary's, and its ladders recompute the gated Y and y at
//                   orbitals displaced out of the full tensor, as
//                   --sc-hessian-kk-int's mixed ladders displace. SC only:
//                   --pc is Unsupported. The second flag is where the matrix
//                   is written, and only beside the first; so is
//                   --parameter-hessian-coords LIST, the coordinates the
//                   direction set is restricted to (a point-group-keeping
//                   block). Refused the approximate digests with the Hessian
//                   flags.
//   --erigold       nevpt2_sc_pseudodensity alone, with --sc-hessian-kk-int
//                   (#180) or --sc-parameter-hessian (#222): the full
//                   MO-integral tensor K2 reads for a pair with no
//                   active-active member on a conventional case.
//                   --sc-hessian-kk's Eddot_kk keeps the refusal: its
//                   ladders of the whole energy displace stored blocks only.
//   --gradgold      nevpt2_sc_pseudodensity (--kappa-full, --sc-hessian-kc,
//                   --sc-hessian-kk-int, --sc-hessian-kk, --sc-source-tangent,
//                   --sc-parameter-hessian,
//                   --dump-source-terms,
//                   --source-oracle; each with or without --pc where it takes
//                   it), nevpt2_demo (--gradient-source, whose y_I and forward
//                   tangent then carry the orbital energies' CI response,
//                   #87, and --sc-hessian / --sc-hessian-vector, then in FULL
//                   mode, #214) and nevpt2_pc_tangent (--pc-hessian /
//                   --pc-hessian-vector in FULL mode, #214): the three
//                   binaries that read the sidecar.
// README.md, "Which binary takes which flag", is this table in prose; keep
// the two together.
constexpr std::array kFlagTable{
    FlagSpec{"--golden", "path", kEvery},
    FlagSpec{"--tiles", "N", kEvery},
    FlagSpec{"--pc", "", kDemos | kScPd},
    FlagSpec{"--profile", "", kDemos},
    FlagSpec{"--rdm-tangent", "", kDemos},
    FlagSpec{"--consume-emitted", "", kEvery},
    FlagSpec{"--blas-digest", "", kEvery},
    FlagSpec{"--digest-emitted", "", kEvery},
    FlagSpec{"--fused-digest", "", kNotPcTangent},
    FlagSpec{"--cublas", "", kNotPcTangent},
    FlagSpec{"--mantissa-bits", "N", kNotPcTangent},
    FlagSpec{"--ozaki", "", kNotPcTangent},
    FlagSpec{"--ozaki-pairs", "P", kNotPcTangent},
    FlagSpec{"--ozaki-check", "", kNotPcTangent},
    FlagSpec{"--pool-threshold", "BYTES|max", kEvery},
    FlagSpec{"--sc-tangent", "", kDemos},
    FlagSpec{"--gradient-source", "", kDemos},
    FlagSpec{"--sc-hessian", "", kDemo},
    FlagSpec{"--sc-hessian-vector", "", kDemo},
    FlagSpec{"--batch", "N", kDfDemo},
    FlagSpec{"--check-blocks", "", kDfDemo},
    FlagSpec{"--kappa-act", "", kScPd},
    FlagSpec{"--kappa-full", "", kScPd},
    FlagSpec{"--sc-hessian-kc", "", kScPd},
    FlagSpec{"--sc-hessian-kk-int", "", kScPd},
    FlagSpec{"--sc-hessian-kk", "", kScPd},
    FlagSpec{"--sc-source-tangent", "path", kScPd},
    FlagSpec{"--sc-parameter-hessian", "path", kScPd},
    FlagSpec{"--parameter-hessian-out", "path", kScPd},
    FlagSpec{"--parameter-hessian-coords", "LIST", kScPd},
    FlagSpec{"--gradgold", "path", kDemo | kScPd | kPcTangent},
    FlagSpec{"--erigold", "path", kScPd},
    FlagSpec{"--dump-source-terms", "path", kScPd},
    FlagSpec{"--probed-y", "", kScPd},
    FlagSpec{"--source-oracle", "path", kScPd},
    FlagSpec{"--fd-steps", "N", kPcTangent},
    FlagSpec{"--fd-start-pow", "N", kPcTangent},
    FlagSpec{"--pc-hessian", "", kPcTangent},
    FlagSpec{"--pc-hessian-vector", "", kPcTangent},
};

// The usage line for `b`, without usage()'s trailing newline (an Error's
// message carries it, and report() ends the line).
std::string usageLine(const Binary b) {
  std::string u = std::format("usage: {} --golden <{}>", binaryName(b), goldenHint(b));
  for (const FlagSpec& f : kFlagTable) {
    if (f.name == "--golden" || (f.binaries & bitOf(b)) == 0) continue;
    if (f.valueName.empty())
      u += std::format(" [{}]", f.name);
    else
      u += std::format(" [{} {}]", f.name, f.valueName);
  }
  const auto cublas = std::ranges::find(kFlagTable, "--cublas", &FlagSpec::name);
  if ((cublas->binaries & bitOf(b)) != 0) u += " (the --cublas family is CUDA-only)";
  return u;
}

// The binaries that implement `spec`, for an err_unsupported message.
std::string implementersOf(const FlagSpec& spec) {
  std::string names;
  for (const Binary b : kBinaries) {
    if ((spec.binaries & bitOf(b)) == 0) continue;
    if (!names.empty()) names += ", ";
    names += binaryName(b);
  }
  return names;
}

// One of a binary's own flags (not a common one) at args[i], already known to
// be that binary's and, if valued, to have a value. Advances i past the value.
Status parseOwnFlag(const std::span<const std::string_view> args, std::size_t& i,
                    Options& opt) {
  const std::string_view a = args[i];
  if (a == "--sc-tangent") {
    opt.scTangent = true;
  } else if (a == "--gradient-source") {
    opt.gradientSource = true;
  } else if (a == "--sc-hessian") {
    opt.scHessian = true;
  } else if (a == "--sc-hessian-vector") {
    opt.scHessianVector = true;
  } else if (a == "--check-blocks") {
    opt.checkBlocks = true;
  } else if (a == "--kappa-act") {
    opt.kappaAct = true;
  } else if (a == "--kappa-full") {
    opt.kappaFull = true;
  } else if (a == "--sc-hessian-kc") {
    opt.scHessianKc = true;
  } else if (a == "--sc-hessian-kk-int") {
    opt.scHessianKkInt = true;
  } else if (a == "--sc-hessian-kk") {
    opt.scHessianKk = true;
  } else if (a == "--sc-source-tangent") {
    opt.scSourceTangentPath = std::string(args[++i]);
  } else if (a == "--sc-parameter-hessian") {
    opt.scParameterHessianPath = std::string(args[++i]);
  } else if (a == "--parameter-hessian-out") {
    opt.parameterHessianOutPath = std::string(args[++i]);
  } else if (a == "--parameter-hessian-coords") {
    // A comma-separated, strictly increasing list of coordinate indices; the
    // range against 3 natm is the response sidecar's, checked where it is read.
    const std::string_view list = args[++i];
    opt.parameterHessianCoords.clear();
    std::size_t start = 0;
    while (true) {
      const std::size_t comma = list.find(',', start);
      const std::string_view item =
          list.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                             : comma - start);
      const Result<std::int64_t> v = parseInteger<std::int64_t>(a, item);
      if (!v) return std::unexpected(v.error());
      if (*v < 0 || (!opt.parameterHessianCoords.empty() && *v <= opt.parameterHessianCoords.back()))
        return err_config(std::format(
            "--parameter-hessian-coords takes non-negative coordinate indices in strictly "
            "increasing order, not \"{}\"",
            list));
      opt.parameterHessianCoords.push_back(*v);
      if (comma == std::string_view::npos) break;
      start = comma + 1;
    }
  } else if (a == "--gradgold") {
    opt.gradgoldPath = std::string(args[++i]);
  } else if (a == "--erigold") {
    opt.erigoldPath = std::string(args[++i]);
  } else if (a == "--dump-source-terms") {
    opt.dumpSourceTermsPath = std::string(args[++i]);
  } else if (a == "--probed-y") {
    opt.probedY = true;
  } else if (a == "--source-oracle") {
    opt.sourceOraclePath = std::string(args[++i]);
  } else if (a == "--batch") {
    const Result<std::int64_t> v = parseInteger<std::int64_t>(a, args[++i]);
    if (!v) return std::unexpected(v.error());
    opt.batch = *v;
  } else if (a == "--fd-steps") {
    const Result<int> v = parseInteger<int>(a, args[++i]);
    if (!v) return std::unexpected(v.error());
    opt.fdSteps = *v;
  } else if (a == "--fd-start-pow") {
    const Result<int> v = parseInteger<int>(a, args[++i]);
    if (!v) return std::unexpected(v.error());
    opt.fdStartPow = *v;
  } else if (a == "--pc-hessian") {
    opt.pcHessian = true;
  } else if (a == "--pc-hessian-vector") {
    opt.pcHessianVector = true;
  } else {
    // In the table, and neither parser takes it: our bug, not the user's.
    check(false, "nevpt2.cli: a flagTable() row that neither parser handles");
  }
  return {};
}

// Every Hessian flag, of every binary (docs/hessian-theory.md, Sec. 9.1) --
// and --sc-source-tangent (#220), a mixed second derivative of the energy in
// the integrals and the parameters, refused the approximate digests alike --
// and --sc-parameter-hessian (#222), the whole parameter Hessian as a
// bilinear form.
constexpr std::array<std::string_view, 9> kHessianFlags{
    "--sc-hessian",        "--sc-hessian-vector", "--sc-hessian-kc",
    "--sc-hessian-kk-int", "--sc-hessian-kk",     "--pc-hessian",
    "--pc-hessian-vector", "--sc-source-tangent", "--sc-parameter-hessian"};

// The first Hessian flag `opt` sets, in kHessianFlags' order; empty for none.
std::string_view hessianFlagOf(const Options& opt) {
  const std::array<bool, kHessianFlags.size()> set{
      opt.scHessian,       opt.scHessianVector,
      opt.scHessianKc,     opt.scHessianKkInt,
      opt.scHessianKk,     opt.pcHessian,
      opt.pcHessianVector, !opt.scSourceTangentPath.empty(),
      !opt.scParameterHessianPath.empty()};
  for (std::size_t k = 0; k < set.size(); ++k)
    if (set[k]) return kHessianFlags[k];
  return {};
}

// Any Hessian flag with --cublas, --ozaki or --fused-digest: Unsupported
// (#157; docs/hessian-theory.md, Sec. 9.1's first row). A second derivative
// built from an approximate digest is not the derivative of anything the
// energy stage computed, and the fused GEMM has no tangent path; the request
// is valid, and no binary has the code for it. Run before finalize(), so the
// kind does not depend on what finalize() would have said of the digest flags
// first. nevpt2_pc_tangent never gets here with one: it implements none of
// the three, so parseArgs has refused them already, as Unsupported too.
Status checkHessianDigests(const Options& opt) {
  const std::string_view hessian = hessianFlagOf(opt);
  if (hessian.empty()) return {};
  const RdmBuildOptions& r = opt.common.rdm;
  const std::string_view digest = r.cublas ? "--cublas"
                                  : r.ozaki ? "--ozaki"
                                  : r.fusedDigest ? "--fused-digest"
                                                  : "";
  if (digest.empty()) return {};
  return err_unsupported(std::format(
      "{} with {}: a second derivative built from an approximate digest is not the derivative "
      "of anything the energy stage computed, so every Hessian runs the emitted digest or "
      "--blas-digest alone; not --cublas, --ozaki or --fused-digest (docs/hessian-theory.md, "
      "Sec. 9.1)",
      hessian, digest));
}

// The checks the demos share on --sc-tangent / --gradient-source (and
// nevpt2_demo's --sc-hessian / --sc-hessian-vector, which only it can set):
// one at a time, the
// tangent RDM build's digests only, no --pc for the SC tangent or Hessian,
// no --profile. Each demo's own rules (the sidecar, --pc on the source term,
// --check-blocks) are its case below.
Status checkDifferentiationFlags(const Options& opt) {
  const CommonOptions& c = opt.common;
  if (opt.scTangent && opt.gradientSource)
    return err_config(
        "--sc-tangent and --gradient-source are two different checks of the same "
        "derivative; run one at a time");
  if (opt.scHessian && (opt.scTangent || opt.gradientSource))
    return err_config(std::format(
        "--sc-hessian and {} are two different checks of the energy's CI derivatives; run "
        "one at a time",
        opt.scTangent ? "--sc-tangent" : "--gradient-source"));
  if (opt.scHessianVector && (opt.scTangent || opt.gradientSource || opt.scHessian))
    return err_config(std::format(
        "--sc-hessian-vector and {} are two different checks of the energy's CI "
        "derivatives; run one at a time",
        opt.scTangent ? "--sc-tangent"
                      : (opt.scHessian ? "--sc-hessian" : "--gradient-source")));
  if (!(opt.scTangent || opt.gradientSource || opt.scHessian || opt.scHessianVector)) return {};
  const std::string_view which = opt.scTangent         ? "--sc-tangent"
                                 : opt.scHessian       ? "--sc-hessian"
                                 : opt.scHessianVector ? "--sc-hessian-vector"
                                                       : "--gradient-source";
  // Both differentiation checks run the tangent RDM build, which is the
  // emitted or the native-fp64 BLAS digest only -- the rule finalize()
  // applies to --rdm-tangent. (For --sc-hessian / --sc-hessian-vector this
  // never fires: checkHessianDigests has refused the pair as Unsupported.)
  if (c.rdm.cublas || c.rdm.ozaki || c.rdm.fusedDigest)
    return err_config(std::format(
        "{} runs the tangent RDM build, which is the emitted digest or --blas-digest; "
        "not --cublas, --ozaki or --fused-digest",
        which));
  // --gradient-source --pc is the PC CI source term's duality (#74, the
  // integral-direct demo's alone); --sc-tangent is the SC tangent's finite
  // difference and nothing else.
  if (c.pc && opt.scTangent)
    return err_config(
        "--sc-tangent differentiates the strongly-contracted energy; it has no --pc path "
        "(the PC tangent's finite difference is nevpt2_pc_tangent)");
  if (c.pc && opt.scHessian)
    return err_config(
        "--sc-hessian differentiates the strongly-contracted energy twice; it has no --pc "
        "path");
  if (c.pc && opt.scHessianVector)
    return err_config(
        "--sc-hessian-vector differentiates the strongly-contracted energy twice; it has no "
        "--pc path");
  // Each check prints its own report and returns before the profile tables
  // are printed, so --profile would be parsed and ignored.
  if (c.profile)
    return err_unsupported(
        std::format("{} prints no profile tables; --profile times the energy run", which));
  return {};
}

// The checks on `b`'s own flags, after finalize(). Only `b`'s fields can be
// set by now: another binary's flag was refused while parsing.
Status checkOwnFlags(const Binary b, const Options& opt) {
  const CommonOptions& c = opt.common;
  switch (b) {
    case Binary::Demo:
      // The sidecar's readers here are the CI source term's orbital-energy
      // response (#87) and the CI Hessian's FULL mode (#214). (With two checks
      // set, one of them is, so the one-at-a-time refusal below is what
      // reports it.)
      if (!opt.gradgoldPath.empty() &&
          !(opt.gradientSource || opt.scHessian || opt.scHessianVector))
        return err_config(
            "--gradgold is the sidecar --gradient-source, --sc-hessian and --sc-hessian-vector "
            "read the orbital energies' response from; it does nothing on its own");
      return checkDifferentiationFlags(opt);
    case Binary::DfDemo:
      if (opt.batch < 0) return err_config("--batch must be >= 0");
      if (const Status st = checkDifferentiationFlags(opt); !st) return st;
      // #162: the PC CI source term (pcCiSourceDevice) and the PC tangent it
      // is held against have no slab entry point, so they are the
      // integral-direct demo's alone.
      if (c.pc && opt.gradientSource)
        return err_unsupported(
            "--gradient-source --pc is the integral-direct demo's alone: the PC CI source term "
            "has no density-fitted slab path (nevpt2_demo --gradient-source --pc)");
      // --check-blocks is the energy run's diagnostic, and its verdict joins
      // that run's PASS line, which neither check prints.
      if (opt.checkBlocks && (opt.scTangent || opt.gradientSource))
        return err_config(std::format(
            "--check-blocks is the energy run's diagnostic; {} replaces that run",
            opt.scTangent ? "--sc-tangent" : "--gradient-source"));
      return {};
    case Binary::ScPseudodensity:
      if (opt.kappaAct && opt.kappaFull)
        return err_config(
            "--kappa-act and --kappa-full are two checks of the same source term; pass one");
      // The dump WRITES the source terms the two checks check: one job per
      // run, as the checks are one at a time.
      if (!opt.dumpSourceTermsPath.empty() && (opt.kappaAct || opt.kappaFull))
        return err_config(
            "--dump-source-terms writes the source terms --kappa-act and --kappa-full check; "
            "run one at a time");
      // --probed-y (#240) holds the dump's one-pass Y to the probed one --
      // the oracle the one-pass route replaced -- so it rides on the dump.
      if (opt.probedY && opt.dumpSourceTermsPath.empty())
        return err_config(
            "--probed-y holds the one-pass Y --dump-source-terms writes to the probed one; it "
            "does nothing without --dump-source-terms");
      // --source-oracle CHECKS the dump's terms against a committed oracle:
      // one job per run again.
      if (!opt.sourceOraclePath.empty() &&
          (opt.kappaAct || opt.kappaFull || !opt.dumpSourceTermsPath.empty()))
        return err_config(
            "--source-oracle compares the source terms --dump-source-terms writes and "
            "--kappa-act / --kappa-full check; run one at a time");
      // --sc-hessian-kc (#184) is a check of its own: one job per run.
      if (opt.scHessianKc && (opt.kappaAct || opt.kappaFull ||
                              !opt.dumpSourceTermsPath.empty() || !opt.sourceOraclePath.empty()))
        return err_config(
            "--sc-hessian-kc checks the energy's mixed orbital-CI second derivative; "
            "--kappa-act, --kappa-full, --dump-source-terms and --source-oracle are first-order "
            "jobs -- run one at a time");
      // With --pc it is PC's mixed block (#201).
      // --sc-hessian-kk-int (#154) likewise: one job per run.
      if (opt.scHessianKkInt && (opt.kappaAct || opt.kappaFull || opt.scHessianKc ||
                                 !opt.dumpSourceTermsPath.empty() ||
                                 !opt.sourceOraclePath.empty()))
        return err_config(
            "--sc-hessian-kk-int checks the orbital Hessian's integral curvature; --kappa-act, "
            "--kappa-full, --sc-hessian-kc, --dump-source-terms and --source-oracle are jobs of "
            "their own -- run one at a time");
      // --sc-hessian-kk (#158) likewise: one job per run. With --pc it is
      // PC's sandwich and Eddot_kk (#202).
      if (opt.scHessianKk && (opt.kappaAct || opt.kappaFull || opt.scHessianKc ||
                              opt.scHessianKkInt || !opt.dumpSourceTermsPath.empty() ||
                              !opt.sourceOraclePath.empty()))
        return err_config(
            "--sc-hessian-kk checks the orbital Hessian block; --kappa-act, --kappa-full, "
            "--sc-hessian-kc, --sc-hessian-kk-int, --dump-source-terms and --source-oracle are "
            "jobs of their own -- run one at a time");
      // --sc-source-tangent (#220) likewise: one job per run. With --pc it is
      // PC's source terms' tangent (#234).
      if (!opt.scSourceTangentPath.empty() &&
          (opt.kappaAct || opt.kappaFull || opt.scHessianKc || opt.scHessianKkInt ||
           opt.scHessianKk || !opt.dumpSourceTermsPath.empty() ||
           !opt.sourceOraclePath.empty()))
        return err_config(
            "--sc-source-tangent checks the source terms' tangent along an integral "
            "perturbation; --kappa-act, --kappa-full, --sc-hessian-kc, --sc-hessian-kk-int, "
            "--sc-hessian-kk, --dump-source-terms and --source-oracle are jobs of their own -- "
            "run one at a time");
      // --sc-parameter-hessian (#222) likewise: one job per run, SC only
      // (PC's bilinear form is not built), and --parameter-hessian-out is
      // where it writes its matrix.
      if (!opt.scParameterHessianPath.empty() &&
          (opt.kappaAct || opt.kappaFull || opt.scHessianKc || opt.scHessianKkInt ||
           opt.scHessianKk || !opt.scSourceTangentPath.empty() ||
           !opt.dumpSourceTermsPath.empty() || !opt.sourceOraclePath.empty()))
        return err_config(
            "--sc-parameter-hessian forms the whole parameter Hessian over a response "
            "sidecar's directions; --kappa-act, --kappa-full, --sc-hessian-kc, "
            "--sc-hessian-kk-int, --sc-hessian-kk, --sc-source-tangent, --dump-source-terms and "
            "--source-oracle are jobs of their own -- run one at a time");
      if (c.pc && !opt.scParameterHessianPath.empty())
        return err_unsupported(
            "--sc-parameter-hessian --pc: the parameter Hessian over the responses is SC's "
            "here; PC's bilinear form is not built");
      if (!opt.parameterHessianOutPath.empty() && opt.scParameterHessianPath.empty())
        return err_config(
            "--parameter-hessian-out is where --sc-parameter-hessian writes its matrix; it does "
            "nothing on its own");
      if (!opt.parameterHessianCoords.empty() && opt.scParameterHessianPath.empty())
        return err_config(
            "--parameter-hessian-coords restricts --sc-parameter-hessian's direction set; it "
            "does nothing on its own");
      // --pc picks PC-NEVPT2's source terms (#74), PC's orbital curvature
      // (#200), PC's mixed block (#201), PC's sandwich and Eddot_kk (#202) and
      // PC's source terms' tangent along an integral perturbation (#234): only
      // the six jobs that assemble them take it.
      if (c.pc && opt.dumpSourceTermsPath.empty() && opt.sourceOraclePath.empty() &&
          !opt.scHessianKkInt && !opt.scHessianKc && !opt.scHessianKk &&
          opt.scSourceTangentPath.empty())
        return err_config(
            "--pc makes --dump-source-terms, --source-oracle, --sc-hessian-kc, "
            "--sc-hessian-kk-int, --sc-hessian-kk and --sc-source-tangent PC-NEVPT2's; the "
            "identities, --kappa-act and --kappa-full are the SC energy's and have no --pc path");
      if (!opt.gradgoldPath.empty() && !opt.kappaFull && !opt.scHessianKc &&
          !opt.scHessianKkInt && !opt.scHessianKk && opt.dumpSourceTermsPath.empty() &&
          opt.sourceOraclePath.empty() && opt.scSourceTangentPath.empty() &&
          opt.scParameterHessianPath.empty())
        return err_config(
            "--gradgold is the sidecar --kappa-full, --sc-hessian-kc, --sc-hessian-kk-int, "
            "--sc-hessian-kk, --sc-source-tangent, --sc-parameter-hessian, --dump-source-terms "
            "and --source-oracle read; it does nothing on its own");
      // --erigold (#180) is the full tensor K2 reads: --sc-hessian-kk-int's,
      // and --sc-parameter-hessian's (#222), whose response directions have
      // no active-active member.
      if (!opt.erigoldPath.empty() && !opt.scHessianKkInt && opt.scParameterHessianPath.empty())
        return err_config(
            "--erigold is the full MO-integral tensor --sc-hessian-kk-int's and "
            "--sc-parameter-hessian's K2 reads for a pair with no active-active member; it does "
            "nothing on its own or beside another job");
      // --sc-hessian-kc / -kk-int / -kk with an approximate digest never get
      // here: checkHessianDigests refused them before finalize(), as
      // Unsupported (docs/hessian-theory.md, Sec. 9.1).
      // --kappa-full and --dump-source-terms build the f3ac/f3ca digests'
      // tangent along a rotation (#88) -- the RDM build rerun at the dotted
      // integrals -- and a tangent is only ever formed from an exact digest,
      // the rule finalize() applies to --rdm-tangent. The identities,
      // --kappa-act and --source-oracle build no tangent and take any digest.
      if ((opt.kappaFull || !opt.dumpSourceTermsPath.empty()) &&
          (c.rdm.cublas || c.rdm.ozaki || c.rdm.fusedDigest))
        return err_config(std::format(
            "{} builds the f3ac/f3ca digests' tangent, which is the emitted digest or "
            "--blas-digest; not --cublas, --ozaki or --fused-digest",
            opt.kappaFull ? "--kappa-full" : "--dump-source-terms"));
      return {};
    case Binary::PcTangent:
      if (opt.fdSteps < kMinFdSteps || opt.fdSteps > kMaxFdSteps)
        return err_config(
            std::format("--fd-steps must be in {}..{}", kMinFdSteps, kMaxFdSteps));
      if (opt.fdStartPow < kMinFdStartPow || opt.fdStartPow > kMaxFdStartPow)
        return err_config(
            std::format("--fd-start-pow must be in {}..{}", kMinFdStartPow, kMaxFdStartPow));
      if (opt.pcHessian && opt.pcHessianVector)
        return err_config(
            "--pc-hessian and --pc-hessian-vector are two different checks of the PC energy's "
            "second derivative; run one at a time");
      // The sidecar's reader here is the PC CI Hessian's FULL mode (#214).
      if (!opt.gradgoldPath.empty() && !opt.pcHessian && !opt.pcHessianVector)
        return err_config(
            "--gradgold is the sidecar --pc-hessian and --pc-hessian-vector read the orbital "
            "energies' response from; it does nothing on its own");
      return {};
  }
  check(false, "checkOwnFlags: not a Binary");
  return {};
}

}  // namespace nevpt2::cli
