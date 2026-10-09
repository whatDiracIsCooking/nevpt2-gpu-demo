// The body of nevpt2.rdm_build: moved out of nevpt2_demo's main()
// unchanged in substance -- same kernels, same launch geometry, same tile
// plan, same device-resident accumulate, same --cublas branch. See the
// module interface for why it is shared.
module;

#include <cstdio>  // stdout/stderr: macros, which `import std` does not carry

// The vendor headers the *_bridge.h below pull in, textually in the GMF; the
// bridges themselves are included after the imports (see
// rdm/rdm_accumulate_bridge.h).
#include <runtime.h>

module nevpt2.rdm_build;

import std;
import :staging;
import :blas;

import nevpt2.common;
import nevpt2.cublas_emul;
import nevpt2.link_tables;
import nevpt2.profile;
import nevpt2.rdm_plan;  // the tile plan and permuteEriConsume
import wwr.blas;
// wwrblasStatus_t's error_type specializations: what lets gpuCheck take a
// BLAS status.
import wwr.extension.blas;
import wwr.runtime_api;

// The linked-in RDM kernels' launchers (rdm_launch.cu, over kernels.cu), the
// emitted f3 digest's (f3_digest.cu), the GEMM digests' f3 permute-scatter
// (f3_scatter.cu) and the Ozaki digest's; the accumulate kernel's
// (rdm_accumulate.cu) is nevpt2.rdm_build:staging's. extern "C++" attaches
// them to the global module, so they keep external linkage and bind to the
// .cu's definitions: see rdm/rdm_accumulate_bridge.h.
extern "C++" {
#include "rdm/f3_digest_bridge.h"
#include "rdm/f3_scatter_bridge.h"
#include "ozaki/ozaki_digest_bridge.h"
#include "rdm/rdm_kernels_bridge.h"
}

namespace nevpt2 {

RdmBuildResult buildRdmsDevice(const RdmBuildOptions& opt, const Tensor& ci, int64_t norb,
                               int64_t nelecA, int64_t nelecB, const Tensor& h2e,
                               const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  // --profile: every span below is reported in the RDM-build table, after the
  // stream synchronization at the end of the build.
  const profile::Section profileSection(profile::kRdmBuild);
  int64_t na = link_tables::num_strings(norb, nelecA);
  int64_t nb = link_tables::num_strings(norb, nelecB);
  int64_t ndet = na * nb;
  int64_t n2 = norb * norb;
  int64_t n4 = n2 * n2;
  int64_t n6 = n4 * n2;
  int64_t nla = link_tables::nlink(norb, nelecA);
  int64_t nlb = link_tables::nlink(norb, nelecB);
  // The RDM kernels' launchers (and their 32-bit indexing) take int: each
  // size they are handed is narrowed here, once, and checked.
  const int norbK = narrowTo<int>(norb);
  const int naK = narrowTo<int>(na), nbK = narrowTo<int>(nb);
  const int nlaK = narrowTo<int>(nla), nlbK = narrowTo<int>(nlb);
  bool cublasArg = opt.cublas;
  // Either GEMM digest (--cublas or --blas-digest) shares everything but the
  // GEMM call: no per-tile dm3 output, dm3 straight into its accumulator, f3
  // through dF3C and the permute-scatter.
  // --ozaki (src/ozaki/) is the third; the caller rejected it with --cublas,
  // and it takes precedence over blasDigest.
  bool gemmDigest = cublasArg || opt.blasDigest || opt.ozaki;
  const char* digestTag = cublasArg ? "[cublas]" : (opt.ozaki ? "[ozaki]" : "[blas]");

  check(opt.nTiles >= 1, "RdmBuildOptions::nTiles >= 1 (each demo's main refuses --tiles < 1)");
  int64_t nK = tileWidthForCount(ndet, opt.nTiles);
  auto tiles = planTiles(ndet, nK);
  // The device buffers' element counts (a DeviceBuffer takes std::size_t),
  // narrowed here, once, and checked, like the kernels' ints above. Every
  // buffer below co-owns res through `owner`.
  const std::size_t n4Count = narrowTo<std::size_t>(n4);
  const std::size_t n6Count = narrowTo<std::size_t>(n6);
  const std::size_t n2KCount = narrowTo<std::size_t>(n2 * nK);
  const std::size_t n4KCount = narrowTo<std::size_t>(n4 * nK);
  const std::shared_ptr<const DeviceResources> owner = res.shared_from_this();
  std::print(
      "\nbuilding dm3/f3ac/f3ca on the GPU (tiles={}, tile width={}, "
      "digest={}, consume={})...\n",
      tiles.size(), nK,
      cublasArg ? "cublas_emul" : (opt.ozaki ? "ozaki" : (opt.blasDigest ? "blas" : "emitted")),
      opt.consumeGemm ? "blas" : "emitted");

  // --- host inputs: link tables + the CI vector + chemists'-order eri ---
  auto fa = link_tables::gen_linkstr_index(norb, nelecA);
  auto fb = link_tables::gen_linkstr_index(norb, nelecB);
  auto ra = link_tables::reverse_link(fa, na, nla);
  auto rb = link_tables::reverse_link(fb, nb, nlb);

  // eri_f3 (chemists' order) is the pre-transpose of the physicist-ordered
  // h2e NevptIntegrals stores -- (0,2,1,3) is its own inverse (see
  // generate_golden.py's docstring for the same identity).
  Tensor eriF3 = transpose(h2e, {0, 2, 1, 3});

  // --- device setup: none here. The device is selected and its context
  // created by DeviceResources::create in main(), before this runs, so
  // `seconds` does not count it. ---
  auto t0 = std::chrono::steady_clock::now();

  // Freed when the build returns.
  DeviceBuffer<double> dCi = upload(ci.data(), res);
  DeviceBuffer<double> dEri = upload(eriF3.data(), res);
  DeviceBuffer<int> dFa = upload(fa, res);
  DeviceBuffer<int> dFb = upload(fb, res);
  DeviceBuffer<int> dRa = upload(ra, res);
  DeviceBuffer<int> dRb = upload(rb, res);

  // BLAS consume: the pre-permuted integral copy (n^4), and DeviceResources'
  // BLAS handle, bound to the build's stream at creation -- where every
  // kernel here runs. The --blas-digest GEMMs share the handle. Empty (no
  // allocation, null data()) on --consume-emitted.
  const wwrblasHandle_t plainBlas = res.blas();
  DeviceBuffer<double> dEriConsume =
      opt.consumeGemm ? upload(permuteEriConsume(eriF3.data(), norb), res)
                      : DeviceBuffer<double>(0, owner);

  // Device-resident: the tile-partial accumulators live on the GPU for the
  // whole sweep (zeroed once, by the allocation), and every tile's partial is
  // added into them by `accumulateInto` instead of a download+host-add+discard
  // round trip (see rdm_accumulate.cu and docs/performance.md, "Device-resident RDM
  // build" section). No host code touches dm3/f3ac/f3ca until the very end,
  // and nothing here blocks the host between tiles: kernels queued on the same
  // stream execute in issue order, so the host can enqueue the
  // *entire* tile sweep without ever waiting on the device mid-loop. (One
  // exception: --cublas's probe reads its verdict back once, inside the first
  // tile -- see probeOnce below.) They become the result's DeviceTensors.
  DeviceBuffer<double> dDm3Final(n6Count, owner);
  std::array<DeviceBuffer<double>, 2> dFaccFinal = {DeviceBuffer<double>(n6Count, owner),
                                                    DeviceBuffer<double>(n6Count, owner)};

  // Optional cuBLAS fixed-point-emulation digest (--cublas): the three digest
  // GEMMs -- ~80% of the compute floor -- routed through cuBLAS's tensor-core
  // emulation instead of the emitted GEMM kernels. dm3 accumulates straight
  // into its (already-zeroed) n^6 accumulator (beta=1); each f3 order runs the
  // winning M=n^4,N=n^2 orientation into dF3C, then a permute-scatter folds it
  // into the NEVPTkern accumulator. The handle binds to the current (primary)
  // context, which DeviceResources::create made current, and to the build's
  // stream. See cublas/cublas_emul.cppm. CUDA-only: on HIP,
  // the caller rejected --cublas at flag parsing and none of this runs.
  EmulHandle blas = nullptr;
  if (cublasArg) blas = make_emul_handle(opt.mantissaBits, stream);

  // --profile labels, indexed by order (0 = ca, 1 = ac).
  constexpr const char* kConsumeLabel[2] = {"consume_ca", "consume_ac"};
  constexpr const char* kConsumeBlasLabel[2] = {"consume_ca [blas]", "consume_ac [blas]"};
  constexpr const char* kDigestF3Label[2] = {"digest_f3_ca", "digest_f3_ac"};
  const std::string kGemmDm3Label = std::string("digest_dm3 ") + digestTag;
  const std::string kGemmF3Label[2] = {std::string("digest_f3_ca ") + digestTag,
                                       std::string("digest_f3_ac ") + digestTag};
  const std::string kScatterLabel[2] = {std::string("f3_scatter_ca ") + digestTag,
                                        std::string("f3_scatter_ac ") + digestTag};
  constexpr const char* kFdm2Label[2] = {"fdm2_ca", "fdm2_ac"};
  constexpr const char* kWedgeLabel[2] = {"wedge_ca", "wedge_ac"};

  // The tile sweep. Its per-tile buffers live in this block and are freed at
  // its closing brace. Their last use is the last tile -- free them there rather than let them
  // sit allocated (sized `n2*nK`/`n4*nK`, i.e. *larger* at *smaller* tile
  // counts) through the fdm2/wedge step and the entire energy-contraction
  // stage that follows. Found by a tile-count sweep: a moderate tile count
  // could fail with an out-of-memory during the *energy* contraction even
  // though the RDM build itself had already completed, because these buffers
  // were still holding memory hostage (docs/performance.md, "The memory
  // pool"). Stream-ordered: the frees are queued behind
  // the last tile's kernels, so the host need not wait for them, and the
  // memory goes back to the pool -- where the fdm2 buffer below and the
  // energy stage's allocations reuse it -- not to the device.
  {
    // Per-tile transition blocks, allocated once at the widest tile and
    // reused across tiles (each launch's `width` bounds how much it touches).
    // No R2 (the permuted copy of L2 kernels.cu's f3 digests read, the same
    // n4*nK) on any path: every f3 digest reads L2 directly
    // (f3_digest.cu on the emitted path, f3_scatter.cu absorbing the
    // permutation on the GEMM ones). The per-tile dm3 output exists only for
    // the emitted digest, which overwrites it; every other digest accumulates
    // straight into the accumulators (or dF3C), and the GEMM digests' f3 temp
    // dF3C only for them -- an unneeded one is empty (no allocation). Each is
    // zero-filled by its allocation, which none of them needs (every tile's
    // produce/consume/digest overwrites what it reads); the suite has no
    // uninitialized allocation, and docs/performance.md ("The memory pool") measures
    // the cost.
    DeviceBuffer<double> dR(n2KCount, owner);
    DeviceBuffer<double> dL2(n4KCount, owner);
    DeviceBuffer<double> dW(n2KCount, owner);
    DeviceBuffer<double> dDm3Tile(gemmDigest ? 0 : n6Count, owner);
    DeviceBuffer<double> dF3C(gemmDigest ? n6Count : 0, owner);
    // --ozaki: the digit planes + row scales, sized once for the widest tile
    // (every digest call is (n4, n2, width <= nK)). As large as L2 itself
    // (8 int8 planes of n4 * nK), so it doubles the per-tile working set.
    // --ozaki-check adds two (n4, n2) temps and their host copies.
    const std::size_t ozakiBytes =
        opt.ozaki ? device::ozakiScratchBytes(narrowTo<int>(n4), narrowTo<int>(n2),
                                              narrowTo<int>(nK))
                  : 0;
    DeviceBuffer<double> dOzaki((ozakiBytes + 7) / 8, owner);
    const bool ozakiCheck = opt.ozaki && opt.ozakiCheck;
    DeviceBuffer<double> dChkRef(ozakiCheck ? n6Count : 0, owner);
    DeviceBuffer<double> dChkOz(ozakiCheck ? n6Count : 0, owner);
    std::vector<double> hRef(ozakiCheck ? n6Count : 0), hOz(ozakiCheck ? n6Count : 0);
    double chkMaxAbsDiff = 0.0, chkMaxRef = 0.0, chkMaxRel = 0.0;
    int chkCalls = 0;
    if (opt.ozaki) {
      std::printf("ozaki int8 digest: 8 digits x 7 bits, pairs p+q <= %d (%s), scratch %.1f MB\n",
                  opt.ozakiMaxPairSum, opt.ozakiMaxPairSum >= 14 ? "all 64" : "truncated",
                  static_cast<double>(ozakiBytes) / (1024.0 * 1024.0));
    }
    {
      // Counted, not measured: the per-tile working set the tile count sizes.
      int64_t doubles = (2 * n2 + n4) * nK;
      std::printf("per-tile transition blocks (R, L2, W; no R2): %.1f MB\n",
                  static_cast<double>(doubles) * 8.0 / (1024.0 * 1024.0));
    }
    // The --cublas probe: one digest-shaped GEMM with the introspection pointer
    // set, reporting the mantissa bits cuBLAS actually engaged. Run on the FIRST
    // tile's operands, right after its produce wrote them (the first tile is
    // always the full width nK, so the probe's K = nK covers exactly what was
    // written). An earlier version ran it before the sweep, on dL2/dR as freshly
    // allocated -- compute-sanitizer initcheck reported every element as an
    // uninitialized read inside cuBLAS. The values never reached a result (the
    // probe's output is scratch), but the engaged/declined verdict printed below
    // was computed on whatever the allocation happened to hold, not on the data
    // the digest GEMMs then ran on. Its (n4, n2) output lands in dF3C: scratch
    // here (the tile's f3 GEMMs overwrite it with beta=0), and the only n6
    // buffer the --cublas path allocates that is not an accumulator.
    bool probed = false;
    auto probeOnce = [&] {
      if (!cublasArg || probed) return;
      probed = true;
      // cublasGemmEx-based and 32-bit, like digest_gemm below.
      int bits = probe_emul_bits(blas, res, dL2.data(), dR.data(), dF3C.data(),
                                 narrowTo<int>(n4), narrowTo<int>(n2),
                                 narrowTo<int>(nK, "use more --tiles"));
      std::printf(
          "cuBLAS fp-emulation digest: max_mantissa_bits=%d, engaged bits=%d%s\n",
          opt.mantissaBits, bits,
          bits < 0 ? "  (DECLINED -> ran native fp64!)"
                   : (bits == 53 ? "  (bit-identical to fp64)" : ""));
    };

    // The one call the two GEMM digests differ in. --cublas's digest_gemm is
    // cublasGemmEx-based and 32-bit (the emulation path has no _64), so its
    // dimensions are narrowed here; the native-fp64 one takes int64_t.
    auto digest = [&](const double* dA, const double* dB, double* dC, int64_t M, int64_t N,
                      int64_t K, double beta) {
      if (cublasArg) {
        digest_gemm(blas, dA, dB, dC, narrowTo<int>(M), narrowTo<int>(N),
                    narrowTo<int>(K, "use more --tiles"), beta);
      } else if (opt.ozaki) {
        auto oz = [&](double* out, double b) {
          gpuCheck(device::ozakiDigestGemm(stream, dA, dB, out, narrowTo<int>(M),
                                           narrowTo<int>(N), narrowTo<int>(K), b,
                                           opt.ozakiMaxPairSum, dOzaki.data()));
        };
        if (ozakiCheck) {
          // Diagnostic only: both products with beta=0, compared on the host.
          // The synchronize orders the two GEMMs and the downloads before the
          // host reads hRef/hOz.
          digestGemmNative(plainBlas, dA, dB, dChkRef.data(), M, N, K, 0.0);
          oz(dChkOz.data(), 0.0);
          const std::size_t bytes = static_cast<std::size_t>(M * N) * sizeof(double);
          gpuCheck(wwrMemcpyAsync(hRef.data(), dChkRef.data(), bytes, wwrMemcpyDeviceToHost,
                                  stream));
          gpuCheck(wwrMemcpyAsync(hOz.data(), dChkOz.data(), bytes, wwrMemcpyDeviceToHost,
                                  stream));
          gpuCheck(wwrStreamSynchronize(stream));
          double callMax = 0.0;
          for (int64_t i = 0; i < M * N; ++i) callMax = std::max(callMax, std::fabs(hRef[i]));
          for (int64_t i = 0; i < M * N; ++i) {
            const double d = std::fabs(hOz[i] - hRef[i]);
            chkMaxAbsDiff = std::max(chkMaxAbsDiff, d);
            if (callMax > 0.0) chkMaxRel = std::max(chkMaxRel, d / callMax);
          }
          chkMaxRef = std::max(chkMaxRef, callMax);
          ++chkCalls;
        }
        oz(dC, beta);
      } else {
        digestGemmNative(plainBlas, dA, dB, dC, M, N, K, beta);
      }
    };

    for (auto [k0, width] : tiles) {
      profile::time("produce", stream, [&] {
        gpuCheck(device::rdmProduce(stream, dCi.data(), dFa.data(), dFb.data(), dRa.data(),
                                       dRb.data(), dR.data(), dL2.data(), norbK, naK, nbK,
                                       nlaK, nlbK, k0, width));
      });
      probeOnce();  // --cublas only, first tile only; outside the profile timer
      if (gemmDigest) {
        // dm3[pqrs,tu] = sum_K L2[pqrs,K] R[tu,K], winning M=n4,N=n2; beta=1
        // accumulates across tiles into the zeroed final accumulator.
        profile::time(kGemmDm3Label, stream, [&] {
          digest(dL2.data(), dR.data(), dDm3Final.data(), n4, n2, width, /*beta=*/1.0);
        });
      } else {
        profile::time("digest_dm3", stream, [&] {
          gpuCheck(
              device::rdmDigestDm3(stream, dR.data(), dL2.data(), dDm3Tile.data(), norbK, width));
        });
        profile::time("accumulate_dm3", stream,
                      [&] { accumulateInto(dDm3Final.data(), dDm3Tile.data(), n6, stream); });
      }

      for (int o = 0; o < 2; ++o) {
        if (opt.consumeGemm) {
          profile::time(kConsumeBlasLabel[o], stream, [&] {
            consumeGemm(plainBlas, dEriConsume.data(), dL2.data(), dW.data(), o, norb, width);
          });
        } else {
          profile::time(kConsumeLabel[o], stream, [&] {
            gpuCheck(device::rdmConsume(stream, o, dEri.data(), dL2.data(), dW.data(), norbK,
                                           width));
          });
        }
        if (gemmDigest) {
          // f3: winning orientation C[r,af] = sum_K L2[r,K] W[af,K] into a temp
          // (beta=0), r = wvut -- R2's C[tuvw,af] with every row at its
          // reversed index -- then permute-scatter (+=) into the NEVPTkern
          // accumulator (zeroed once). The scatter reads dF3C as (n4,n2)
          // row-major -- exactly what this GEMM writes.
          profile::time(kGemmF3Label[o], stream, [&] {
            digest(dL2.data(), dW.data(), dF3C.data(), n4, n2, width, /*beta=*/0.0);
          });
          // ca transposes the last two axes of each row; ac's target index is
          // the source index, so its scatter is the tile accumulate itself.
          profile::time(kScatterLabel[o], stream, [&] {
            if (o == 0) {
              gpuCheck(
                  device::f3ScatterCa(stream, dF3C.data(), dFaccFinal[0].data(), norbK));
            } else {
              accumulateInto(dFaccFinal[1].data(), dF3C.data(), n6, stream);
            }
          });
        } else {
          // Reads L2 directly (no R2) and adds into the accumulator itself, so
          // there is no per-tile f3 output and no accumulate_f3 (f3_digest.cu).
          profile::time(kDigestF3Label[o], stream, [&] {
            gpuCheck(device::f3DigestAccumulate(stream, o, dW.data(), dL2.data(),
                                                   dFaccFinal[o].data(), norbK, width));
          });
        }
      }
    }
    if (ozakiCheck) {
      std::printf(
          "ozaki check over %d digest GEMMs: max |ozaki - native| = %.3e, max |native| = %.3e, "
          "max |diff| / max|C| per call = %.3e\n",
          chkCalls, chkMaxAbsDiff, chkMaxRef, chkMaxRel);
    }
  }  // dR, dL2, dW, dDm3Tile, dF3C freed here, on `stream`, behind the last tile

  // fdm2 + wedge run once, on the fully-summed dm3/f3 partials (they need
  // the complete tensors, not a per-tile slice). dDm3Final/dFaccFinal are already device-resident
  // (no upload -- they're the accumulators the tile loop just wrote into).
  // Zero-filled by its allocation, like the tile buffers.
  DeviceBuffer<double> dFdm2(n4Count, owner);
  for (int o = 0; o < 2; ++o) {
    profile::time(kFdm2Label[o], stream, [&] {
      gpuCheck(
          device::rdmFdm2(stream, o, dEri.data(), dDm3Final.data(), dFdm2.data(), norbK));
    });
    profile::time(kWedgeLabel[o], stream, [&] {
      gpuCheck(device::rdmWedge(stream, o, dFdm2.data(), dFaccFinal[o].data(), norbK));
    });
  }
  // Every RDM-build kernel above is enqueued on `stream` with no blocking
  // host call in between (the point of going device-resident) -- so an
  // explicit sync is needed here for `seconds` to measure actual GPU
  // completion instead of just host-side dispatch time. The energy
  // contraction synchronizes through each class's downloadTensor, so this isn't
  // needed for correctness, only for an honest timing split. (Not a
  // wwrDeviceSynchronize: with every launch, copy and BLAS call on the one
  // stream, syncing that stream waits for the same work.)
  gpuCheck(wwrStreamSynchronize(stream));
  auto t1 = std::chrono::steady_clock::now();

  destroy_emul_handle(blas);

  // The accumulators move into the result; every other buffer above (the
  // inputs, dEriConsume, dFdm2) is freed on `stream` as the build returns.
  const std::vector<int64_t> n6Shape(6, norb);

  RdmBuildResult r;
  r.dm3 = DeviceTensor(std::move(dDm3Final), n6Shape);
  r.f3ca = DeviceTensor(std::move(dFaccFinal[0]), n6Shape);  // order 0 == "ca"
  r.f3ac = DeviceTensor(std::move(dFaccFinal[1]), n6Shape);  // order 1 == "ac"
  r.seconds = std::chrono::duration<double>(t1 - t0).count();
  std::printf("GPU RDM build (dm3/f3ac/f3ca) done in %.2fs\n", r.seconds);
  // Measured, unlike the per-tile line above: the pool's own high-water marks
  // so far (everything since DeviceResources::create -- for the
  // density-fitted demo that includes its setup before the build).
  printPoolHighWater("after the RDM build", res);
  return r;
}

}  // namespace nevpt2
