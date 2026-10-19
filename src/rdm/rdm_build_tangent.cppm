// nevpt2.rdm_build:tangent -- the tangent of the dm3/f3ac/f3ca build with
// respect to the CI vector (buildRdmTangentsDevice), its three-argument form
// for the second derivative (buildRdmHessianTangentsDevice, #153) and the
// --rdm-tangent self-check both demos run (checkRdmTangentsDevice). All three
// are declared in the module interface, which has the derivation; this
// partition is where they are defined, so rdm_build.cpp's plain build stays
// untouched by them.
// An internal partition: nothing here is exported, and no importer of
// nevpt2.rdm_build sees it -- it is reached only because the functions it
// defines are the module's own.
//
// IT IS THE SAME PIPELINE, run on CROSS TERMS. Every output is a bilinear
// form B(x, y) in two copies of the CI vector, and what both entry points
// want is a set of symmetrised cross terms B(x_a, x_b) + B(x_b, x_a) over a
// few vectors x_0, x_1, ...: the tangent is the one pair (c, u); the Hessian's
// three RDM sets are (c, u), (c, v) and (u, v). So per tile, `produce` runs
// ONCE PER VECTOR (not once per pair: the three pairs share three vectors),
// every digest runs twice per pair over the swapped operands, and the
// fdm2/wedge step runs once per pair at the end on that pair's accumulators
// -- that step is linear in what it reads, so it maps cross terms to cross
// terms unchanged. No kernel here is new: every launch below is one
// buildRdmsDevice already makes, through the same bridges.
//
// What it costs. One L2 and one R block per VECTOR per tile, against one W:
// the per-tile working set is (V n^4 + (V + 1) n^2) doubles per determinant
// for V vectors -- 2 n^4 + 3 n^2 for the tangent against the plain build's
// n^4 + 2 n^2, and 3 n^4 + 4 n^2 for the Hessian's three
// (docs/hessian-theory.md, Sec. 4.4). So neither carries the plain build's
// measured tile floors, nor the Hessian the tangent's (docs/performance.md,
// "Tile floors").
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

// One cross term: the indices of its two vectors in the build's vector list.
struct RdmCrossPair {
  std::size_t a = 0;
  std::size_t b = 0;
};

// What a cross-term build prints, so each entry point keeps its own lines.
struct RdmCrossLabels {
  const char* building;  // what "building ... on the GPU" names
  const char* blocks;    // the per-tile line's parenthesis
  const char* done;      // the "GPU RDM ... done" line's subject
  const char* poolWhen;  // printPoolHighWater's "after ..."
};

// --profile labels for produce, one per vector (at most three vectors).
constexpr std::array<const char*, 3> kProduceLabel = {"produce_c_dot", "produce_u_dot",
                                                      "produce_v_dot"};

// The symmetrised cross terms B(x_a, x_b) + B(x_b, x_a) of every output, one
// RdmBuildResult per pair in `pairs`' order, from ONE tile sweep over the
// vectors `vecs` (each one amplitude per determinant). See the header for the
// pipeline and its working set.
std::vector<RdmBuildResult> buildRdmCrossTermsDevice(
    const RdmBuildOptions& opt, std::span<const Tensor* const> vecs,
    std::span<const RdmCrossPair> pairs, int64_t norb, int64_t nelecA, int64_t nelecB,
    const Tensor& h2e, const RdmCrossLabels& labels, const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  // --profile: the spans land in the plain build's table, each labelled
  // `*_dot` so the two builds' phases stay apart in it.
  const profile::Section profileSection(profile::kRdmBuild);

  check(opt.nTiles >= 1, "RdmBuildOptions::nTiles >= 1 (each demo's main refuses --tiles < 1)");
  // Only the two exact digests have a tangent path here: the approximate ones
  // (--cublas, --ozaki) and the fused GEMM are refused with --rdm-tangent,
  // --sc-tangent, --sc-hessian and --gradient-source at flag parsing, so
  // reaching this is our bug, not a user's.
  check(!opt.cublas && !opt.ozaki && !opt.fusedDigest,
        "the tangent RDM build runs the emitted or the native-fp64 BLAS digest "
        "(cli::finalize refuses --cublas / --ozaki / --fused-digest with --rdm-tangent)");
  check(!vecs.empty() && vecs.size() <= kProduceLabel.size() && !pairs.empty(),
        "a cross-term RDM build takes one to three vectors and at least one pair");
  for (const RdmCrossPair& p : pairs)
    check(p.a < vecs.size() && p.b < vecs.size(), "a cross pair names a vector it was handed");

  const int64_t na = link_tables::num_strings(norb, nelecA);
  const int64_t nb = link_tables::num_strings(norb, nelecB);
  const int64_t ndet = na * nb;
  // Every vector lives on the same determinant axis: the caller derives them
  // from the very (norb, nelec) it passes here, so a mismatch is a broken
  // invariant, not bad input.
  for (const Tensor* x : vecs)
    check(x->size() == ndet, "the tangent RDM build's vectors each hold one amplitude per "
                             "determinant");

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
  std::print("\nbuilding {} on the GPU (tiles={}, tile width={}, digest={}, consume={})...\n",
             labels.building, tiles.size(), nK, gemmDigest ? "blas" : "emitted",
             opt.consumeGemm ? "blas" : "emitted");

  // --- host inputs: the same link tables and chemists'-order eri the plain
  // build uses, plus the vectors ---
  const auto fa = link_tables::gen_linkstr_index(norb, nelecA);
  const auto fb = link_tables::gen_linkstr_index(norb, nelecB);
  const auto ra = link_tables::reverse_link(fa, na, nla);
  const auto rb = link_tables::reverse_link(fb, nb, nlb);
  const Tensor eriF3 = transpose(h2e, {0, 2, 1, 3});

  const auto t0 = std::chrono::steady_clock::now();

  // Freed when the build returns, on `stream`, like every buffer here.
  std::vector<DeviceBuffer<double>> dVec;
  dVec.reserve(vecs.size());
  for (const Tensor* x : vecs) dVec.push_back(upload(x->data(), res));
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

  // One pair's accumulators: zeroed by their allocation, summed over tiles on
  // the device, and moved into that pair's result. facc[0] is order 0 (ca).
  struct PairAcc {
    DeviceBuffer<double> dm3;
    std::array<DeviceBuffer<double>, 2> facc;
  };
  std::vector<PairAcc> acc;
  acc.reserve(pairs.size());
  for (std::size_t p = 0; p < pairs.size(); ++p)
    acc.push_back(PairAcc{DeviceBuffer<double>(n6Count, owner),
                          {DeviceBuffer<double>(n6Count, owner),
                           DeviceBuffer<double>(n6Count, owner)}});

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
    // ONE copy of produce's output PER VECTOR: a cross term L2(x_a).R(x_b)
    // needs both vectors' blocks at once, and with three vectors every block
    // is read by two pairs, so none can be overwritten by another's produce.
    // W is single: each consume is followed at once by the digest that reads
    // it, before the next consume overwrites it (beta=0).
    std::vector<DeviceBuffer<double>> dR;
    std::vector<DeviceBuffer<double>> dL2;
    dR.reserve(vecs.size());
    dL2.reserve(vecs.size());
    // Built here and moved in, not emplaced, so an out-of-memory abort names
    // this line (the constructor's source_location) rather than <vector>'s.
    for (std::size_t i = 0; i < vecs.size(); ++i) {
      dR.push_back(DeviceBuffer<double>(n2KCount, owner));
      dL2.push_back(DeviceBuffer<double>(n4KCount, owner));
    }
    DeviceBuffer<double> dW(n2KCount, owner);
    // The emitted dm3 digest's per-tile output (it overwrites, so every cross
    // term reuses it and each is accumulated in turn); the BLAS digest
    // accumulates into the pair's dm3 directly and allocates none.
    DeviceBuffer<double> dDm3Tile(gemmDigest ? 0 : n6Count, owner);
    // The BLAS digest's f3 temp: a pair's two terms are summed into it
    // (beta = 0 then 1) and scattered once.
    DeviceBuffer<double> dF3C(gemmDigest ? n6Count : 0, owner);
    {
      // Counted, not measured: the per-tile working set the tile count sizes.
      const int64_t nv = static_cast<int64_t>(vecs.size());
      const int64_t doubles = (nv * n4 + (nv + 1) * n2) * nK;
      std::printf("per-tile transition blocks (%s): %.1f MB\n", labels.blocks,
                  static_cast<double>(doubles) * 8.0 / (1024.0 * 1024.0));
    }

    for (auto [k0, width] : tiles) {
      for (std::size_t i = 0; i < vecs.size(); ++i) {
        profile::time(kProduceLabel[i], stream, [&] {
          gpuCheck(device::rdmProduce(stream, dVec[i].data(), dFa.data(), dFb.data(),
                                      dRa.data(), dRb.data(), dR[i].data(), dL2[i].data(), norbK,
                                      naK, nbK, nlaK, nlbK, k0, width));
        });
      }

      // dm3_ab[pqrs,tu] += sum_K [ L2(x_a)[pqrs,K] R(x_b)[tu,K]
      //                          + L2(x_b)[pqrs,K] R(x_a)[tu,K] ].
      // `term` indexes the two: 0 is L2(x_a) against R(x_b), 1 the other way.
      for (std::size_t p = 0; p < pairs.size(); ++p) {
        const auto [a, b] = pairs[p];
        for (int term = 0; term < 2; ++term) {
          const double* const l2 = term == 0 ? dL2[a].data() : dL2[b].data();
          const double* const r = term == 0 ? dR[b].data() : dR[a].data();
          if (gemmDigest) {
            profile::time("digest_dm3_dot [blas]", stream, [&] {
              digestGemmNative(plainBlas, l2, r, acc[p].dm3.data(), n4, n2, width,
                               /*beta=*/1.0);
            });
          } else {
            profile::time("digest_dm3_dot", stream, [&] {
              gpuCheck(device::rdmDigestDm3(stream, r, l2, dDm3Tile.data(), norbK, width));
            });
            profile::time("accumulate_dm3_dot", stream, [&] {
              accumulateInto(acc[p].dm3.data(), dDm3Tile.data(), n6, stream);
            });
          }
        }
      }

      for (int o = 0; o < 2; ++o) {
        for (std::size_t p = 0; p < pairs.size(); ++p) {
          const auto [a, b] = pairs[p];
          for (int term = 0; term < 2; ++term) {
            // W is the OTHER vector's consume: the digest pairs L2(x_a) with
            // W(x_b) and L2(x_b) with W(x_a).
            const double* const l2Digest = term == 0 ? dL2[a].data() : dL2[b].data();
            const double* const l2Consume = term == 0 ? dL2[b].data() : dL2[a].data();
            if (opt.consumeGemm) {
              profile::time(kConsumeBlasLabel[o], stream, [&] {
                consumeGemm(plainBlas, dEriConsume.data(), l2Consume, dW.data(), o, norb,
                            width);
              });
            } else {
              profile::time(kConsumeLabel[o], stream, [&] {
                gpuCheck(device::rdmConsume(stream, o, dEri.data(), l2Consume, dW.data(), norbK,
                                            width));
              });
            }
            if (gemmDigest) {
              // Both terms into the f3 temp (beta=0 then 1), scattered once
              // below: the scatter accumulates, so folding each term
              // separately would be the same answer at twice the scatter.
              profile::time(kGemmF3Label[o], stream, [&] {
                digestGemmNative(plainBlas, l2Digest, dW.data(), dF3C.data(), n4, n2, width,
                                 /*beta=*/term == 0 ? 0.0 : 1.0);
              });
            } else {
              // Reads L2 directly and adds into the accumulator itself, like
              // the plain build's emitted f3 digest.
              profile::time(kDigestF3Label[o], stream, [&] {
                gpuCheck(device::f3DigestAccumulate(stream, o, dW.data(), l2Digest,
                                                    acc[p].facc[o].data(), norbK, width));
              });
            }
          }
          if (gemmDigest) {
            profile::time(kScatterLabel[o], stream, [&] {
              if (o == 0) {
                gpuCheck(device::f3ScatterCa(stream, dF3C.data(), acc[p].facc[0].data(), norbK));
              } else {
                accumulateInto(acc[p].facc[1].data(), dF3C.data(), n6, stream);
              }
            });
          }
        }
      }
    }
  }  // dR, dL2, dW, dDm3Tile, dF3C freed here, behind the last tile

  // fdm2 + wedge, once per pair, on its fully-summed partials. Both steps are
  // LINEAR in what they read (fdm2 contracts dm3 with the integrals; the wedge
  // copies and adds), so running them on a cross term is the cross term of
  // running them on the plain build -- which is why the tangent needs no
  // derivative of its own here. One fdm2 scratch serves every pair: each
  // wedge reads it before the next fdm2 overwrites it, in stream order.
  DeviceBuffer<double> dFdm2(n4Count, owner);
  for (PairAcc& pa : acc) {
    for (int o = 0; o < 2; ++o) {
      profile::time(kFdm2Label[o], stream, [&] {
        gpuCheck(device::rdmFdm2(stream, o, dEri.data(), pa.dm3.data(), dFdm2.data(), norbK));
      });
      profile::time(kWedgeLabel[o], stream, [&] {
        gpuCheck(device::rdmWedge(stream, o, dFdm2.data(), pa.facc[o].data(), norbK));
      });
    }
  }
  // Nothing above blocks the host, so this sync is what makes `seconds` GPU
  // completion rather than host dispatch -- the plain build's reason, and its
  // one stream, so syncing it waits for all of the work.
  gpuCheck(wwrStreamSynchronize(stream));
  const auto t1 = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(t1 - t0).count();

  const std::vector<int64_t> n6Shape(6, norb);
  std::vector<RdmBuildResult> out;
  out.reserve(acc.size());
  for (PairAcc& pa : acc) {
    RdmBuildResult r;
    r.dm3 = DeviceTensor(std::move(pa.dm3), n6Shape);
    r.f3ca = DeviceTensor(std::move(pa.facc[0]), n6Shape);  // order 0 == "ca"
    r.f3ac = DeviceTensor(std::move(pa.facc[1]), n6Shape);  // order 1 == "ac"
    r.seconds = seconds;
    out.push_back(std::move(r));
  }
  std::printf("GPU RDM %s done in %.2fs\n", labels.done, seconds);
  printPoolHighWater(labels.poolWhen, res);
  return out;
}

RdmBuildResult buildRdmTangentsDevice(const RdmBuildOptions& opt, const Tensor& ci,
                                      const Tensor& dir, int64_t norb, int64_t nelecA,
                                      int64_t nelecB, const Tensor& h2e,
                                      const DeviceResources& res) {
  const std::array<const Tensor*, 2> vecs = {&ci, &dir};
  const std::array<RdmCrossPair, 1> pairs = {RdmCrossPair{0, 1}};
  const RdmCrossLabels labels{"dm3_dot/f3ac_dot/f3ca_dot", "R x2, L2 x2, W",
                              "tangent build (dm3_dot/f3ac_dot/f3ca_dot)",
                              "after the RDM tangent build"};
  std::vector<RdmBuildResult> r =
      buildRdmCrossTermsDevice(opt, vecs, pairs, norb, nelecA, nelecB, h2e, labels, res);
  return std::move(r.front());
}

RdmHessianTangents buildRdmHessianTangentsDevice(const RdmBuildOptions& opt, const Tensor& ci,
                                                 const Tensor& u, const Tensor& v, int64_t norb,
                                                 int64_t nelecA, int64_t nelecB,
                                                 const Tensor& h2e,
                                                 const DeviceResources& res) {
  const std::array<const Tensor*, 3> vecs = {&ci, &u, &v};
  // (c, u), (c, v), (u, v): the RdmHessianTangents fields, in order.
  const std::array<RdmCrossPair, 3> pairs = {RdmCrossPair{0, 1}, RdmCrossPair{0, 2},
                                             RdmCrossPair{1, 2}};
  const RdmCrossLabels labels{"the three-argument tangents (c,u), (c,v), (u,v)",
                              "R x3, L2 x3, W", "three-argument tangent build",
                              "after the three-argument RDM tangent build"};
  std::vector<RdmBuildResult> r =
      buildRdmCrossTermsDevice(opt, vecs, pairs, norb, nelecA, nelecB, h2e, labels, res);
  return RdmHessianTangents{std::move(r[0]), std::move(r[1]), std::move(r[2])};
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
