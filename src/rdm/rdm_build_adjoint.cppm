// nevpt2.rdm_build:adjoint -- the TRANSPOSE of the build's f3 path with
// respect to the integrals (buildDigestAdjointDevice): given the two digests'
// adjoints G_ca = dE/d(f3ca), G_ac = dE/d(f3ac), the one n^4 tensor
// dE/d(h2e) through the digests. Declared in the module interface, which has
// the contract; this partition is where it is defined, so rdm_build.cpp's
// plain build stays untouched by it.
// An internal partition: nothing here is exported, and no importer of
// nevpt2.rdm_build sees it -- it is reached only because the function it
// defines is the module's own.
//
// THE FORWARD MAP IT TRANSPOSES. At a fixed CI vector, per order o, the build
// computes (rdm_build.cpp)
//
//   per tile:  W_o[af, K]   = sum eri . L2[., K]                (consume)
//              f3pre_o[r, last2_o(a, f)] += sum_K W_o[af, K] L2[r, K]   (digest)
//   once:      fdm2_o       = eri . dm3                         (fdm2)
//              f3_o         = wedge(f3pre_o, fdm2_o)            (wedge)
//
// -- LINEAR in eri, which is why #88's forward tangent is the plain build run
// at h2e_dot. Its transpose, run backwards:
//
//   once:      Gt_o, fdm2bar_o = wedge^T(G_o)                   (f3WedgeAdjoint,
//                                                                f3Fdm2Adjoint)
//              hbar += fdm2^T(fdm2bar_o, dm3)                   (f3Fdm2EriAdjoint)
//   per tile:  Wbar_o[af, K] = sum_r Gt_o[r, af] L2[r, K]       (one DGEMM)
//              hbar += consume^T(Wbar_o, L2)                    (f3ConsumeAdjoint)
//
// L2 is re-produced tile by tile by the plain build's own produce kernel, so
// the 4-RDM never exists here either. The per-tile working set is the plain
// build's -- R, L2 and one n^2-by-width block, Wbar in W's place -- and on top
// of it the two n^6 pre-wedge adjoints Gt_o are held across the sweep; the
// floors that sets are measured, not assumed (docs/performance.md, "Tile
// floors").
//
// WHICH DIGEST IT RUNS. None of the build's: the transpose has one route on
// both backends -- the four f3_transpose.cu kernels and a native-fp64
// wwrblasDgemm -- so of `opt` only the tile count is read. An approximate
// forward digest (--cublas, --ozaki) changes the G it is handed, not this map.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry

// The vendor headers the *_bridge.h below pull in, textually in the GMF; the
// bridges themselves are included after the imports (see
// rdm/rdm_accumulate_bridge.h).
#include <runtime.h>

module nevpt2.rdm_build:adjoint;

import std;
import :staging;  // upload

import nevpt2.common;
import nevpt2.link_tables;
import nevpt2.profile;
import nevpt2.rdm_build;
import nevpt2.rdm_plan;  // the tile plan
import wwr.blas;
// wwrblasStatus_t's error_type specializations: what lets gpuCheck take a
// BLAS status.
import wwr.extension.blas;
import wwr.runtime_api;

// The produce launcher the plain build calls, and the transpose's own four,
// bound the same way: extern "C++" attaches them to the global module, so
// they keep external linkage and bind to the .cu's definitions (see
// rdm/rdm_accumulate_bridge.h).
extern "C++" {
#include "rdm/f3_transpose_bridge.h"
#include "rdm/rdm_kernels_bridge.h"
}

namespace nevpt2 {

DeviceTensor buildDigestAdjointDevice(const RdmBuildOptions& opt, const Tensor& ci,
                                      const int64_t norb, const int64_t nelecA,
                                      const int64_t nelecB, const DeviceTensor& dm3,
                                      const DeviceTensor& f3acBar, const DeviceTensor& f3caBar,
                                      const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  // --profile: the spans land in the plain build's table, each labelled
  // `*_adj` so the phases stay apart in it.
  const profile::Section profileSection(profile::kRdmBuild);
  check(opt.nTiles >= 1, "RdmBuildOptions::nTiles >= 1 (each demo's main refuses --tiles < 1)");

  const int64_t na = link_tables::num_strings(norb, nelecA);
  const int64_t nb = link_tables::num_strings(norb, nelecB);
  const int64_t ndet = na * nb;
  const int64_t n2 = norb * norb;
  const int64_t n4 = n2 * n2;
  const int64_t n6 = n4 * n2;
  const std::vector<int64_t> n6Shape(6, norb);
  // The caller hands the state's own build outputs and their adjoints; a
  // shape that is not (norb,)^6 is a broken invariant, not bad input.
  check(ci.size() == ndet, "buildDigestAdjointDevice: ci holds one amplitude per determinant");
  check(dm3.dims == n6Shape && f3acBar.dims == n6Shape && f3caBar.dims == n6Shape,
        "buildDigestAdjointDevice: dm3 and the two digest adjoints are (norb,)^6");
  const int64_t nla = link_tables::nlink(norb, nelecA);
  const int64_t nlb = link_tables::nlink(norb, nelecB);
  const int norbK = narrowTo<int>(norb);
  const int naK = narrowTo<int>(na), nbK = narrowTo<int>(nb);
  const int nlaK = narrowTo<int>(nla), nlbK = narrowTo<int>(nlb);

  const int64_t nK = tileWidthForCount(ndet, opt.nTiles);
  const auto tiles = planTiles(ndet, nK);
  const std::size_t n4Count = narrowTo<std::size_t>(n4);
  const std::size_t n6Count = narrowTo<std::size_t>(n6);
  const std::size_t n2KCount = narrowTo<std::size_t>(n2 * nK);
  const std::size_t n4KCount = narrowTo<std::size_t>(n4 * nK);
  const std::shared_ptr<const DeviceResources> owner = res.shared_from_this();
  std::print("\nbuilding dE/dh2e through f3ac/f3ca on the GPU, the digests' transpose "
             "(tiles={}, tile width={})...\n",
             tiles.size(), nK);

  const auto fa = link_tables::gen_linkstr_index(norb, nelecA);
  const auto fb = link_tables::gen_linkstr_index(norb, nelecB);
  const auto ra = link_tables::reverse_link(fa, na, nla);
  const auto rb = link_tables::reverse_link(fb, nb, nlb);

  const auto t0 = std::chrono::steady_clock::now();
  DeviceBuffer<double> dCi = upload(ci.data(), res);
  DeviceBuffer<int> dFa = upload(fa, res);
  DeviceBuffer<int> dFb = upload(fb, res);
  DeviceBuffer<int> dRa = upload(ra, res);
  DeviceBuffer<int> dRb = upload(rb, res);

  // The result, physicist-ordered like the h2e the build takes, zeroed by its
  // allocation and accumulated into by every adjoint below.
  DeviceBuffer<double> dHbar(n4Count, owner);
  // order 0 is ca, 1 is ac: the build's own indexing.
  const std::array<const double*, 2> gBar = {f3caBar.data(), f3acBar.data()};
  constexpr const char* kWedgeLabel[2] = {"wedge_ca_adj", "wedge_ac_adj"};
  constexpr const char* kFdm2Label[2] = {"fdm2_ca_adj", "fdm2_ac_adj"};
  constexpr const char* kDigestLabel[2] = {"digest_f3_ca_adj [blas]", "digest_f3_ac_adj [blas]"};
  constexpr const char* kConsumeLabel[2] = {"consume_ca_adj", "consume_ac_adj"};

  // The wedge's and fdm2's adjoints, once, before the sweep: the wedge is the
  // LAST forward step, so its transpose is the first, and the pre-wedge
  // adjoint Gt_o it leaves is what every tile's digest adjoint reads.
  std::array<DeviceBuffer<double>, 2> dGt = {DeviceBuffer<double>(n6Count, owner),
                                             DeviceBuffer<double>(n6Count, owner)};
  {
    DeviceBuffer<double> dFdm2Bar(n4Count, owner);
    for (int o = 0; o < 2; ++o) {
      profile::time(kWedgeLabel[o], stream, [&] {
        gpuCheck(device::f3WedgeAdjoint(stream, o, gBar[o], dGt[o].data(), norbK));
        gpuCheck(device::f3Fdm2Adjoint(stream, gBar[o], dFdm2Bar.data(), norbK));
      });
      // Reads fdm2bar before the next order's f3Fdm2Adjoint overwrites it:
      // the two are ordered on the one stream.
      profile::time(kFdm2Label[o], stream, [&] {
        gpuCheck(device::f3Fdm2EriAdjoint(stream, o, dm3.data(), dFdm2Bar.data(), dHbar.data(),
                                          norbK));
      });
    }
  }  // dFdm2Bar freed here, on `stream`, behind its last reader

  // The tile sweep: the plain build's produce, then per order the digest's
  // adjoint (a DGEMM into Wbar) and consume's (into hbar). Its buffers are
  // freed at the closing brace, behind the last tile's kernels.
  {
    DeviceBuffer<double> dR(n2KCount, owner);  // produce writes it; unread here
    DeviceBuffer<double> dL2(n4KCount, owner);
    DeviceBuffer<double> dWbar(n2KCount, owner);
    {
      // Counted, not measured: the per-tile working set the tile count sizes,
      // plus the two n^6 pre-wedge adjoints held across the sweep.
      const int64_t doubles = (n4 + 2 * n2) * nK;
      std::printf("per-tile transition blocks (R, L2, Wbar): %.1f MB, + 2 x n^6 Gt: %.1f MB\n",
                  static_cast<double>(doubles) * 8.0 / (1024.0 * 1024.0),
                  static_cast<double>(2 * n6) * 8.0 / (1024.0 * 1024.0));
    }
    const wwrblasHandle_t blas = res.blas();
    const double one = 1.0, zero = 0.0;
    for (auto [k0, width] : tiles) {
      profile::time("produce_adj", stream, [&] {
        gpuCheck(device::rdmProduce(stream, dCi.data(), dFa.data(), dFb.data(), dRa.data(),
                                    dRb.data(), dR.data(), dL2.data(), norbK, naK, nbK, nlaK,
                                    nlbK, k0, width));
      });
      for (int o = 0; o < 2; ++o) {
        // Wbar[af, K] = sum_r Gt[r, af] L2[r, K]: row-major (n2, width) out of
        // the row-major (n4, n2) Gt and (n4, width) L2. Column-major, that is
        // C(width, n2) = L2(width, n4) . Gt(n2, n4)^T, every leading dimension
        // the row length. The 64-bit-integer API, so nothing is narrowed.
        profile::time(kDigestLabel[o], stream, [&] {
          gpuCheck(wwrblasDgemm_64(blas, WWRBLAS_OP_N, WWRBLAS_OP_T, width, n2, n4, &one,
                                   dL2.data(), width, dGt[o].data(), n2, &zero, dWbar.data(),
                                   width));
        });
        profile::time(kConsumeLabel[o], stream, [&] {
          gpuCheck(device::f3ConsumeAdjoint(stream, o, dWbar.data(), dL2.data(), dHbar.data(),
                                            norbK, width));
        });
      }
    }
  }  // dR, dL2, dWbar freed here, on `stream`, behind the last tile

  // Nothing above blocks the host, so this sync is what makes the time below
  // GPU completion rather than host dispatch -- the plain build's reason.
  gpuCheck(wwrStreamSynchronize(stream));
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("GPU digest transpose (dE/dh2e through f3ac/f3ca) done in %.2fs\n", seconds);
  printPoolHighWater("after the digest transpose", res);
  return DeviceTensor(std::move(dHbar), {norb, norb, norb, norb});
}

}  // namespace nevpt2
