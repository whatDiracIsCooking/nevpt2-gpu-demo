module nevpt2.cli;

import std;
import nevpt2.cublas_emul;  // requireCublasEmul: a refusing stub on HIP
import :binary_flags;       // the binary/flag matrix, each binary's own flags

namespace nevpt2::cli {

Result<bool> parseCommonFlag(const std::span<const std::string_view> args, std::size_t& i,
                             CommonOptions& opt) {
  const std::string_view a = args[i];
  const bool hasValue = i + 1 < args.size();

  if (a == "--profile") {
    opt.profile = true;
  } else if (a == "--pc") {
    opt.pc = true;
  } else if (a == "--consume-emitted") {
    opt.rdm.consumeGemm = false;
  } else if (a == "--cublas") {
    opt.rdm.cublas = true;
  } else if (a == "--blas-digest") {
    opt.rdm.blasDigest = true;
  } else if (a == "--digest-emitted") {
    opt.rdm.blasDigest = false;
  } else if (a == "--ozaki") {
    opt.rdm.ozaki = true;
  } else if (a == "--fused-digest") {
    opt.rdm.fusedDigest = true;
  } else if (a == "--ozaki-check") {
    opt.rdm.ozakiCheck = true;
  } else if (a == "--rdm-tangent") {
    opt.rdmTangent = true;
  } else if (!hasValue) {
    return false;  // a valued flag at the end, or not ours: the app's usage
  } else if (a == "--golden") {
    opt.goldenPath = args[++i];
  } else if (a == "--pool-threshold") {
    const Result<std::uint64_t> threshold = parsePoolThreshold(args[++i]);
    if (!threshold) return std::unexpected(threshold.error());
    opt.poolThreshold = *threshold;
  } else if (a == "--tiles") {
    const Result<int64_t> tiles = parseInteger<int64_t>(a, args[++i]);
    if (!tiles) return std::unexpected(tiles.error());
    opt.rdm.nTiles = *tiles;
  } else if (a == "--ozaki-pairs") {
    const Result<int> pairs = parseInteger<int>(a, args[++i]);
    if (!pairs) return std::unexpected(pairs.error());
    opt.rdm.ozakiMaxPairSum = *pairs;
    opt.ozakiPairsGiven = true;
  } else if (a == "--mantissa-bits") {
    const Result<int> bits = parseInteger<int>(a, args[++i]);
    if (!bits) return std::unexpected(bits.error());
    opt.rdm.mantissaBits = *bits;
    opt.mantissaBitsGiven = true;
  } else {
    return false;
  }
  return true;
}

Status finalize(CommonOptions& opt, const std::string_view goldenHint) {
  if (opt.goldenPath.empty())
    return err_config(std::format("--golden <{}> is required", goldenHint));
  // 0 would divide by zero sizing the tiles, and a negative count would walk
  // the determinant axis backwards.
  if (opt.rdm.nTiles < 1) return err_config("--tiles must be >= 1");
  // The largest digit-pair level p + q the Ozaki digest keeps: 8 digits a
  // side, so 0..14 (14 = all 64 pairs).
  if (opt.rdm.ozakiMaxPairSum < 0 || opt.rdm.ozakiMaxPairSum > 14)
    return err_config("--ozaki-pairs must be in 0..14");
  // Only the --ozaki digest reads either; anywhere else they would be
  // silently ignored.
  if (opt.ozakiPairsGiven && !opt.rdm.ozaki)
    return err_config("--ozaki-pairs sets the --ozaki digest's truncation; it needs --ozaki");
  if (opt.rdm.ozakiCheck && !opt.rdm.ozaki)
    return err_config("--ozaki-check checks the --ozaki digest's GEMMs; it needs --ozaki");
  // The fp64 significand is 53 bits (the default, bit-identical to native
  // fp64); below 1 there is nothing left to emulate.
  if (opt.rdm.mantissaBits < 1 || opt.rdm.mantissaBits > 53)
    return err_config("--mantissa-bits must be in 1..53");
  // Only the --cublas digest reads it: anywhere else it would be silently
  // ignored. On HIP this fires before requireCublasEmul would refuse --cublas.
  if (opt.mantissaBitsGiven && !opt.rdm.cublas)
    return err_config("--mantissa-bits sets the --cublas digest's precision; it needs --cublas");
  // --cublas on a build without the emulated digest (HIP): an Unsupported
  // Error from nevpt2.cublas_emul's stub.
  if (opt.rdm.cublas) {
    if (Status st = requireCublasEmul(); !st) return st;
  }
  // blasDigest defaults on only on HIP, which rejected --cublas just above,
  // so this fires only for an explicit --blas-digest.
  if (opt.rdm.cublas && opt.rdm.blasDigest)
    return err_config("--cublas and --blas-digest are two digests; pick one");
  // --ozaki is a digest of its own. It takes precedence over blasDigest,
  // whose default (on under HIP) cannot be told apart from an explicit flag.
  if (opt.rdm.ozaki && opt.rdm.cublas)
    return err_config("--cublas and --ozaki are two digests; pick one");
  if (opt.rdm.ozaki) opt.rdm.blasDigest = false;
  // The fused GEMM replaces the GEMM digests' three calls; the emitted
  // digest has no GEMM to fuse.
  if (opt.rdm.fusedDigest && !(opt.rdm.cublas || opt.rdm.blasDigest || opt.rdm.ozaki))
    return err_config("--fused-digest needs a GEMM digest (--blas-digest, --cublas or --ozaki)");
  // The tangent build runs the emitted digest or the native-fp64 BLAS one; the
  // approximate digests (--cublas, --ozaki) and the fused GEMM have no tangent
  // path, so asking for both is a configuration error rather than something
  // silently ignored.
  if (opt.rdmTangent && (opt.rdm.cublas || opt.rdm.ozaki || opt.rdm.fusedDigest))
    return err_config(
        "--rdm-tangent runs the emitted digest or --blas-digest; not --cublas, --ozaki or "
        "--fused-digest");
  return {};
}

std::string_view binaryName(const Binary b) {
  switch (b) {
    case Binary::Demo:
      return "nevpt2_demo";
    case Binary::DfDemo:
      return "nevpt2_df_demo";
    case Binary::ScPseudodensity:
      return "nevpt2_sc_pseudodensity";
    case Binary::PcTangent:
      return "nevpt2_pc_tangent";
  }
  check(false, "binaryName: not a Binary");
  return {};
}

std::string_view goldenHint(const Binary b) {
  return b == Binary::DfDemo ? "path_df.nevpt2gold" : "path.nevpt2gold";
}

std::span<const FlagSpec> flagTable() { return kFlagTable; }

const FlagSpec* findFlag(const std::string_view name) {
  const auto* it = std::ranges::find(kFlagTable, name, &FlagSpec::name);
  return it == kFlagTable.end() ? nullptr : it;
}

bool implements(const Binary b, const std::string_view flag) {
  const FlagSpec* spec = findFlag(flag);
  return spec != nullptr && (spec->binaries & bitOf(b)) != 0;
}

std::string usage(const Binary b) { return usageLine(b) + '\n'; }

Result<Options> parseArgs(const Binary b, const std::span<const std::string_view> args) {
  Options opt;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view a = args[i];
    const FlagSpec* spec = findFlag(a);
    // Not a flag of any binary: a usage error.
    if (spec == nullptr)
      return err_config(std::format("unrecognized argument \"{}\"; {}", a, usageLine(b)));
    // Another binary's flag: refused, never parsed and ignored.
    if ((spec->binaries & bitOf(b)) == 0)
      return err_unsupported(
          std::format("{} does not implement {} (only {}); a flag a binary would not act on "
                      "is refused, not ignored",
                      binaryName(b), a, implementersOf(*spec)));
    if (!spec->valueName.empty() && i + 1 >= args.size())
      return err_config(std::format("{} takes a value ({}); {}", a, spec->valueName, usageLine(b)));
    const Result<bool> used = parseCommonFlag(args, i, opt.common);
    if (!used) return std::unexpected(used.error());
    if (*used) continue;
    if (Status st = parseOwnFlag(args, i, opt); !st) return std::unexpected(st.error());
  }
  if (Status st = finalize(opt.common, goldenHint(b)); !st) return std::unexpected(st.error());
  if (Status st = checkOwnFlags(b, opt); !st) return std::unexpected(st.error());
  return opt;
}

}  // namespace nevpt2::cli
