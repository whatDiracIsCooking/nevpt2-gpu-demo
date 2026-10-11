// nevpt2.rdm_build:tangent -- the tangent of the dm3/f3ac/f3ca build with
// respect to the CI vector (buildRdmTangentsDevice) and the --rdm-tangent
// self-check both demos run (checkRdmTangentsDevice). Both are declared in the
// module interface, which has the derivation; this partition is where they are
// defined, so rdm_build.cpp's plain build stays untouched by them.
// An internal partition: nothing here is exported, and no importer of
// nevpt2.rdm_build sees it -- it is reached only because the two functions it
// defines are the module's own.
//
// IT IS THE SAME PIPELINE, run with one copy of the CI vector replaced by the
// direction: per tile, produce twice (once from `ci`, once from `dir`), every
// digest twice over the swapped operands, and the fdm2/wedge step once at the
// end on the tangent accumulators -- that step is linear in what it reads, so
// it maps tangents to tangents unchanged. No kernel here is new: every launch
// below is one buildRdmsDevice already makes, through the same bridges.
//
// What it costs. Two L2 blocks and two R blocks per tile instead of one each
// (the two cross terms need both directions' produce output at once), against
// one W: the per-tile working set is 2 n^4 + 3 n^2 doubles per determinant
// against the plain build's n^4 + 2 n^2, so a little over twice as much, and
// the plain build's measured tile floors do not carry over
// (docs/performance.md, "Tile floors").
module;

#include <cstdio>  // stdout/stderr: macros, which `import std` does not carry

// The vendor headers the *_bridge.h below pull in, textually in the GMF; the
// bridges themselves are included after the imports (see
// rdm/rdm_accumulate_bridge.h).
#include <runtime.h>

module nevpt2.rdm_build:tangent;

import std;
import :blas;     // consumeGemm, digestGemmNative
import :staging;  // upload, accumulateInto

import nevpt2.common;
import nevpt2.link_tables;
import nevpt2.profile;
import nevpt2.rdm_build;
import nevpt2.rdm_plan;  // the tile plan and permuteEriConsume
import wwr.blas;
import wwr.runtime_api;

// The same launchers rdm_build.cpp calls, bound the same way: extern "C++"
// attaches them to the global module, so they keep external linkage and bind
// to the .cu's definitions (see rdm/rdm_accumulate_bridge.h).
extern "C++" {
#include "rdm/f3_digest_bridge.h"
#include "rdm/f3_scatter_bridge.h"
#include "rdm/rdm_kernels_bridge.h"
}

namespace nevpt2 {

RdmBuildResult buildRdmTangentsDevice(const RdmBuildOptions& opt, const Tensor& ci,
                                      const Tensor& dir, int64_t norb, int64_t nelecA,
                                      int64_t nelecB, const Tensor& h2e,
                                      const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  // --profile: the spans land in the plain build's table, each labelled
  // `*_dot` so the two builds' phases stay apart in it.
  const profile::Section profileSection(profile::kRdmBuild);

  check(opt.nTiles >= 1, "RdmBuildOptions::nTiles >= 1 (each demo's main refuses --tiles < 1)");
  // Only the two exact digests have a tangent path here: the approximate ones
  // (--cublas, --ozaki) and the fused GEMM are refused with --rdm-tangent at
  // flag parsing, so reaching this is our bug, not a user's.
  check(!opt.cublas && !opt.ozaki && !opt.fusedDigest,
        "the tangent RDM build runs the emitted or the native-fp64 BLAS digest "
        "(cli::finalize refuses --cublas / --ozaki / --fused-digest with --rdm-tangent)");

  const int64_t na = link_tables::num_strings(norb, nelecA);
  const int64_t nb = link_tables::num_strings(norb, nelecB);
  const int64_t ndet = na * nb;
  // Both vectors live on the same determinant axis: the caller derives them
  // from the very (norb, nelec) it passes here, so a mismatch is a broken
  // invariant, not bad input.
  check(ci.size() == ndet && dir.size() == ndet,
        "the tangent RDM build's ci and dir each hold one amplitude per determinant");

  const int64_t n2 = norb * norb;
  const int64_t n4 = n2 * n2;
  const int64_t n6 = n4 * n2;
  const int64_t nla = link_tables::nlink(norb, nelecA);
  const int64_t nlb = link_tables::nlink(norb, nelecB);
  // The RDM kernels' launchers (and their 32-bit indexing) take int: each size
  // they are handed is narrowed here, once, and checked.
  const int norbK = narrowTo<int>(norb);
  const int naK = narrowTo<int>(na), nbK = narrowTo<int>(nb);
  const int nlaK = narrowTo<int>(nla), nlbK = narrowTo<int>(nlb);
  // --blas-digest (HIP's default) against the emitted digest: the one branch
  // below, exactly as in the plain build.
  const bool gemmDigest = opt.blasDigest;

  const int64_t nK = tileWidthForCount(ndet, opt.nTiles);
  const auto tiles = planTiles(ndet, nK);
  const std::size_t n4Count = narrowTo<std::size_t>(n4);
  const std::size_t n6Count = narrowTo<std::size_t>(n6);
  const std::size_t n2KCount = narrowTo<std::size_t>(n2 * nK);
  const std::size_t n4KCount = narrowTo<std::size_t>(n4 * nK);
  const std::shared_ptr<const DeviceResources> owner = res.shared_from_this();
  std::print(
      "\nbuilding dm3_dot/f3ac_dot/f3ca_dot on the GPU (tiles={}, tile width={}, "
      "digest={}, consume={})...\n",
      tiles.size(), nK, gemmDigest ? "blas" : "emitted", opt.consumeGemm ? "blas" : "emitted");

  // --- host inputs: the same link tables and chemists'-order eri the plain
  // build uses, plus the direction beside the CI vector ---
  const auto fa = link_tables::gen_linkstr_index(norb, nelecA);
  const auto fb = link_tables::gen_linkstr_index(norb, nelecB);
  const auto ra = link_tables::reverse_link(fa, na, nla);
  const auto rb = link_tables::reverse_link(fb, nb, nlb);
  const Tensor eriF3 = transpose(h2e, {0, 2, 1, 3});

  const auto t0 = std::chrono::steady_clock::now();

  // Freed when the build returns, on `stream`, like every buffer here.
  DeviceBuffer<double> dCi = upload(ci.data(), res);
  DeviceBuffer<double> dDir = upload(dir.data(), res);
  DeviceBuffer<double> dEri = upload(eriF3.data(), res);
  DeviceBuffer<int> dFa = upload(fa, res);
  DeviceBuffer<int> dFb = upload(fb, res);
  DeviceBuffer<int> dRa = upload(ra, res);
  DeviceBuffer<int> dRb = upload(rb, res);

  // The BLAS consume's permuted integral copy and DeviceResources' handle,
  // bound to this stream at creation -- the plain build's arrangement, and the
  // --blas-digest GEMMs share the handle. Empty on --consume-emitted.
  const wwrblasHandle_t plainBlas = res.blas();
  DeviceBuffer<double> dEriConsume =
      opt.consumeGemm ? upload(permuteEriConsume(eriF3.data(), norb), res)
                      : DeviceBuffer<double>(0, owner);

  // The tangent accumulators: zeroed by their allocation, summed over tiles on
  // the device, and moved into the result.
  DeviceBuffer<double> dDm3Final(n6Count, owner);
  std::array<DeviceBuffer<double>, 2> dFaccFinal = {DeviceBuffer<double>(n6Count, owner),
                                                    DeviceBuffer<double>(n6Count, owner)};

  // --profile labels, indexed by order (0 = ca, 1 = ac), all suffixed `_dot`:
  // the spans share the plain build's table.
  constexpr const char* kConsumeLabel[2] = {"consume_ca_dot", "consume_ac_dot"};
  constexpr const char* kConsumeBlasLabel[2] = {"consume_ca_dot [blas]", "consume_ac_dot [blas]"};
  constexpr const char* kDigestF3Label[2] = {"digest_f3_ca_dot", "digest_f3_ac_dot"};
  constexpr const char* kGemmF3Label[2] = {"digest_f3_ca_dot [blas]", "digest_f3_ac_dot [blas]"};
  constexpr const char* kScatterLabel[2] = {"f3_scatter_ca_dot [blas]", "f3_scatter_ac_dot [blas]"};
  constexpr const char* kFdm2Label[2] = {"fdm2_ca_dot", "fdm2_ac_dot"};
  constexpr const char* kWedgeLabel[2] = {"wedge_ca_dot", "wedge_ac_dot"};

  // The tile sweep. Its buffers live in this block and are freed at its
  // closing brace, behind the last tile's kernels -- before the fdm2/wedge
  // step below allocates, for the reason the plain build's sweep gives.
  {
    // TWO copies of produce's output, one per direction: the cross terms
    // L2(c).R(u) and L2(u).R(c) both need the other direction's block, so
    // neither can be overwritten by the other's produce. W is single: each
    // order's two consumes run one after the other, the digest of the first
    // reading W before the second overwrites it (beta=0).
    DeviceBuffer<double> dRc(n2KCount, owner);
    DeviceBuffer<double> dRu(n2KCount, owner);
    DeviceBuffer<double> dL2c(n4KCount, owner);
    DeviceBuffer<double> dL2u(n4KCount, owner);
    DeviceBuffer<double> dW(n2KCount, owner);
    // The emitted dm3 digest's per-tile output (it overwrites, so both cross
    // terms reuse it and each is accumulated in turn); the BLAS digest
    // accumulates into dDm3Final directly and allocates none.
    DeviceBuffer<double> dDm3Tile(gemmDigest ? 0 : n6Count, owner);
    // The BLAS digest's f3 temp: both cross terms are summed into it
    // (beta = 0 then 1) and scattered once.
    DeviceBuffer<double> dF3C(gemmDigest ? n6Count : 0, owner);
    {
      // Counted, not measured: the per-tile working set the tile count sizes.
      const int64_t doubles = (2 * n4 + 3 * n2) * nK;
      std::printf("per-tile transition blocks (R x2, L2 x2, W): %.1f MB\n",
                  static_cast<double>(doubles) * 8.0 / (1024.0 * 1024.0));
    }

    for (auto [k0, width] : tiles) {
      profile::time("produce_c_dot", stream, [&] {
        gpuCheck(device::rdmProduce(stream, dCi.data(), dFa.data(), dFb.data(), dRa.data(),
                                    dRb.data(), dRc.data(), dL2c.data(), norbK, naK, nbK, nlaK,
                                    nlbK, k0, width));
      });
      profile::time("produce_u_dot", stream, [&] {
        gpuCheck(device::rdmProduce(stream, dDir.data(), dFa.data(), dFb.data(), dRa.data(),
                                    dRb.data(), dRu.data(), dL2u.data(), norbK, naK, nbK, nlaK,
                                    nlbK, k0, width));
      });

      // dm3_dot[pqrs,tu] += sum_K [ L2(c)[pqrs,K] R(u)[tu,K]
      //                           + L2(u)[pqrs,K] R(c)[tu,K] ].
      // `term` indexes the two: 0 is L2(c) against R(u), 1 the other way.
      for (int term = 0; term < 2; ++term) {
        const double* const l2 = term == 0 ? dL2c.data() : dL2u.data();
        const double* const r = term == 0 ? dRu.data() : dRc.data();
        if (gemmDigest) {
          profile::time("digest_dm3_dot [blas]", stream, [&] {
            digestGemmNative(plainBlas, l2, r, dDm3Final.data(), n4, n2, width, /*beta=*/1.0);
          });
        } else {
          profile::time("digest_dm3_dot", stream, [&] {
            gpuCheck(device::rdmDigestDm3(stream, r, l2, dDm3Tile.data(), norbK, width));
          });
          profile::time("accumulate_dm3_dot", stream,
                        [&] { accumulateInto(dDm3Final.data(), dDm3Tile.data(), n6, stream); });
        }
      }

      for (int o = 0; o < 2; ++o) {
        for (int term = 0; term < 2; ++term) {
          // W is the OTHER direction's consume: the digest pairs L2(c) with
          // W(u) and L2(u) with W(c).
          const double* const l2Digest = term == 0 ? dL2c.data() : dL2u.data();
          const double* const l2Consume = term == 0 ? dL2u.data() : dL2c.data();
          if (opt.consumeGemm) {
            profile::time(kConsumeBlasLabel[o], stream, [&] {
              consumeGemm(plainBlas, dEriConsume.data(), l2Consume, dW.data(), o, norb, width);
            });
          } else {
            profile::time(kConsumeLabel[o], stream, [&] {
              gpuCheck(device::rdmConsume(stream, o, dEri.data(), l2Consume, dW.data(), norbK,
                                          width));
            });
          }
          if (gemmDigest) {
            // Both terms into the f3 temp (beta=0 then 1), scattered once
            // below: the scatter accumulates, so folding each term separately
            // would be the same answer at twice the scatter.
            profile::time(kGemmF3Label[o], stream, [&] {
              digestGemmNative(plainBlas, l2Digest, dW.data(), dF3C.data(), n4, n2, width,
                               /*beta=*/term == 0 ? 0.0 : 1.0);
            });
          } else {
            // Reads L2 directly and adds into the accumulator itself, like the
            // plain build's emitted f3 digest.
            profile::time(kDigestF3Label[o], stream, [&] {
              gpuCheck(device::f3DigestAccumulate(stream, o, dW.data(), l2Digest,
                                                  dFaccFinal[o].data(), norbK, width));
            });
          }
        }
        if (gemmDigest) {
          profile::time(kScatterLabel[o], stream, [&] {
            if (o == 0) {
              gpuCheck(device::f3ScatterCa(stream, dF3C.data(), dFaccFinal[0].data(), norbK));
            } else {
              accumulateInto(dFaccFinal[1].data(), dF3C.data(), n6, stream);
            }
          });
        }
      }
    }
  }  // dRc, dRu, dL2c, dL2u, dW, dDm3Tile, dF3C freed here, behind the last tile

  // fdm2 + wedge, once, on the fully-summed tangent partials. Both steps are
  // LINEAR in what they read (fdm2 contracts dm3 with the integrals; the wedge
  // copies and adds), so running them on the tangents is the tangent of
  // running them on the plain build -- which is why the tangent needs no
  // derivative of its own here.
  DeviceBuffer<double> dFdm2(n4Count, owner);
  for (int o = 0; o < 2; ++o) {
    profile::time(kFdm2Label[o], stream, [&] {
      gpuCheck(device::rdmFdm2(stream, o, dEri.data(), dDm3Final.data(), dFdm2.data(), norbK));
    });
    profile::time(kWedgeLabel[o], stream, [&] {
      gpuCheck(device::rdmWedge(stream, o, dFdm2.data(), dFaccFinal[o].data(), norbK));
    });
  }
  // Nothing above blocks the host, so this sync is what makes `seconds` GPU
  // completion rather than host dispatch -- the plain build's reason, and its
  // one stream, so syncing it waits for all of the work.
  gpuCheck(wwrStreamSynchronize(stream));
  const auto t1 = std::chrono::steady_clock::now();

  const std::vector<int64_t> n6Shape(6, norb);
  RdmBuildResult r;
  r.dm3 = DeviceTensor(std::move(dDm3Final), n6Shape);
  r.f3ca = DeviceTensor(std::move(dFaccFinal[0]), n6Shape);  // order 0 == "ca"
  r.f3ac = DeviceTensor(std::move(dFaccFinal[1]), n6Shape);  // order 1 == "ac"
  r.seconds = std::chrono::duration<double>(t1 - t0).count();
  std::printf("GPU RDM tangent build (dm3_dot/f3ac_dot/f3ca_dot) done in %.2fs\n", r.seconds);
  printPoolHighWater("after the RDM tangent build", res);
  return r;
}

Status checkRdmTangentsDevice(const RdmBuildOptions& opt, const RdmBuildResult& built,
                              const Tensor& ci, int64_t norb, int64_t nelecA, int64_t nelecB,
                              const Tensor& h2e, const DeviceResources& res) {
  // The tier's one tolerance, relative to the tensor's own largest element
  // (the device reassociates each K-sum, so nothing here is bit-identical).
  constexpr double kTol = 1e-7;
  const RdmBuildResult dot =
      buildRdmTangentsDevice(opt, ci, ci, norb, nelecA, nelecB, h2e, res);

  const std::array<const char*, 3> names = {"dm3_dot", "f3ca_dot", "f3ac_dot"};
  const std::array<const DeviceTensor*, 3> plain = {&built.dm3, &built.f3ca, &built.f3ac};
  const std::array<const DeviceTensor*, 3> tangent = {&dot.dm3, &dot.f3ca, &dot.f3ac};
  std::print("\nrdm tangent self-check (Euler: the tangent along the state is twice the build)\n");
  double worstRel = 0.0;
  const char* worstName = names[0];
  for (std::size_t i = 0; i < names.size(); ++i) {
    // downloadTensor synchronizes the stream itself, so the tangent build's
    // last kernel is complete before these are read.
    const Tensor want = downloadTensor(*plain[i], res.stream());
    const Tensor got = downloadTensor(*tangent[i], res.stream());
    check(got.size() == want.size(), "the tangent build's output has the plain build's shape");
    double scale = 0.0, worst = 0.0;
    for (int64_t k = 0; k < want.size(); ++k) {
      const double w = 2.0 * want.flat(k);
      scale = std::max(scale, std::abs(w));
      worst = std::max(worst, std::abs(got.flat(k) - w));
    }
    const double rel = worst / std::max(1.0, scale);
    std::printf("  %-8s max |dot - 2 x build| = %.3e  (max |2 x build| = %.3e, relative %.3e)\n",
                names[i], worst, scale, rel);
    if (rel > worstRel) {
      worstRel = rel;
      worstName = names[i];
    }
  }
  if (worstRel > kTol) {
    return err_numerical(std::format(
        "the RDM tangent build disagrees with twice the plain build: {} is off by {:.3e} "
        "relative, past {:.1e}",
        worstName, worstRel, kTol));
  }
  std::printf("rdm tangent self-check: every tensor within %.1e relative\n", kTol);
  return {};
}

}  // namespace nevpt2
