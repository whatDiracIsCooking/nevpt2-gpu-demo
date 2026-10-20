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
//                   source-term jobs alone (--dump-source-terms and
//                   --source-oracle write / compare PC-NEVPT2's with it, #74;
//                   the identities and --kappa-act / --kappa-full are SC's,
//                   so it is a config error there). nevpt2_pc_tangent is PC
//                   throughout, so the switch would change nothing there.
//   --profile       the demos only: nothing else prints a profile table.
//   --rdm-tangent   the demos only: the gradient binaries run no Euler check of
//                   the tangent RDM build beside their own (#38, #39).
//   the --cublas / --ozaki family and --fused-digest: every binary but
//                   nevpt2_pc_tangent (kNotPcTangent, above).
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
//   each binary's own flags: that binary alone. --dump-source-terms is
//                   nevpt2_sc_pseudodensity's because Y_xy needs the gradient
//                   sidecar's full-MO-range arrays; nevpt2_demo computes y_I
//                   too, but half a dump is not one.
//                   --source-oracle is its too, for the same reason: it
//                   compares the dump's two terms with a committed oracle.
//                   --pc-hessian is nevpt2_pc_tangent's (#152): PC's second
//                   directional derivative, checked by a finite difference of
//                   the PC tangent that binary already owns. Like the rest of
//                   that binary it runs the emitted or the native-fp64 BLAS
//                   digest only, which the table refuses the others for.
//   --gradgold      nevpt2_sc_pseudodensity (--kappa-full, --dump-source-terms,
//                   --source-oracle)
//                   and nevpt2_demo (--gradient-source, whose y_I and forward
//                   tangent then carry the orbital energies' CI response,
//                   #87): the two binaries that read the sidecar.
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
    FlagSpec{"--gradgold", "path", kDemo | kScPd},
    FlagSpec{"--dump-source-terms", "path", kScPd},
    FlagSpec{"--source-oracle", "path", kScPd},
    FlagSpec{"--fd-steps", "N", kPcTangent},
    FlagSpec{"--fd-start-pow", "N", kPcTangent},
    FlagSpec{"--pc-hessian", "", kPcTangent},
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
  } else if (a == "--gradgold") {
    opt.gradgoldPath = std::string(args[++i]);
  } else if (a == "--dump-source-terms") {
    opt.dumpSourceTermsPath = std::string(args[++i]);
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
  } else {
    // In the table, and neither parser takes it: our bug, not the user's.
    check(false, "nevpt2.cli: a flagTable() row that neither parser handles");
  }
  return {};
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
  // applies to --rdm-tangent.
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
      // The sidecar's one reader here is the CI source term's orbital-energy
      // response (#87). (With both checks set, --gradient-source is, so the
      // one-at-a-time refusal below is what reports it.)
      if (!opt.gradgoldPath.empty() && !opt.gradientSource)
        return err_config(
            "--gradgold is the sidecar --gradient-source reads the orbital energies' "
            "response from; it does nothing on its own");
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
      if (c.pc && opt.scHessianKc)
        return err_config(
            "--sc-hessian-kc differentiates the strongly-contracted energy twice; PC's half of "
            "H_kc has no path (it waits on the PC projector terms, #152 / #155)");
      // --pc picks PC-NEVPT2's source terms (#74): only the two jobs that
      // assemble them take it.
      if (c.pc && opt.dumpSourceTermsPath.empty() && opt.sourceOraclePath.empty())
        return err_config(
            "--pc makes --dump-source-terms and --source-oracle PC-NEVPT2's; the identities, "
            "--kappa-act and --kappa-full are the SC energy's and have no --pc path");
      if (!opt.gradgoldPath.empty() && !opt.kappaFull && !opt.scHessianKc &&
          opt.dumpSourceTermsPath.empty() && opt.sourceOraclePath.empty())
        return err_config(
            "--gradgold is the sidecar --kappa-full, --sc-hessian-kc, --dump-source-terms and "
            "--source-oracle read; it does nothing on its own");
      // --sc-hessian-kc runs the tangent RDM build (its forward orientation)
      // and the digests' tangent: exact digests only, as every tangent path.
      if (opt.scHessianKc && (c.rdm.cublas || c.rdm.ozaki || c.rdm.fusedDigest))
        return err_config(
            "--sc-hessian-kc runs the tangent RDM build and the digests' tangent, which are "
            "the emitted digest or --blas-digest; not --cublas, --ozaki or --fused-digest");
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
      return {};
  }
  check(false, "checkOwnFlags: not a Binary");
  return {};
}

}  // namespace nevpt2::cli
