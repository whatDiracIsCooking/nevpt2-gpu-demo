module nevpt2.cli;

import std;
import nevpt2.cublas_emul;  // requireCublasEmul: a refusing stub on HIP

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
  } else if (a == "--ozaki-check") {
    opt.rdm.ozakiCheck = true;
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
  } else if (a == "--mantissa-bits") {
    const Result<int> bits = parseInteger<int>(a, args[++i]);
    if (!bits) return std::unexpected(bits.error());
    opt.rdm.mantissaBits = *bits;
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
  return {};
}

std::string usage(const std::string_view prog, const std::string_view goldenHint,
                  const std::string_view extraFlags) {
  return std::format(
      "usage: {} --golden <{}> {}{}"
      "[--tiles N] [--pc] [--profile] "
      "[--consume-emitted] [--blas-digest | --digest-emitted] "
      "[--cublas] [--mantissa-bits N] [--ozaki [--ozaki-pairs P] [--ozaki-check]] "
      "[--pool-threshold BYTES|max] "
      "(the --cublas family is CUDA-only)\n",
      prog, goldenHint, extraFlags, extraFlags.empty() ? "" : " ");
}

}  // namespace nevpt2::cli
