// nevpt2.rdm_build -- the tiled, device-resident build of dm3 and the two
// integral-contracted 4-RDM digests f3ac/f3ca from a CI vector: steps 1-4 of
// nevpt2_demo's original main() header, lifted out of that main()
// verbatim so the density-fitted demo runs the very same build.
//
// Shared, not per-method, because the only integrals it touches are the
// ACTIVE-space h2e (n_act^4): f3ac/f3ca contract the 4-RDM with that block
// alone. The integral-direct demo reads h2e from the golden file; the
// density-fitted one rebuilds it from B_aa first. Either way the build itself
// does not know or care.
export module nevpt2.rdm_build;

import std;

// Re-exported: the build takes the caller's DeviceResources (and a host
// Tensor) and hands back owning DeviceTensors allocated from its pool.
export import nevpt2.device_tensor;

export namespace nevpt2 {

struct RdmBuildOptions {
  // Determinant-axis tile count. docs/performance.md, "Tile floors" and "Why CAS(10,10) needs tiling":
  // MEASURED floors, and too few corrupts device memory instead of failing.
  int64_t nTiles = 3;
  // Route the three digest GEMMs through cuBLAS fixed-point emulation
  // (CUDA-only; the caller rejects it on HIP before getting here).
  bool cublas = false;
  int mantissaBits = 53;
  // --blas-digest: the same three digest GEMMs (and the same f3 scatter) as
  // --cublas, but through plain native-fp64 wwrblasDgemm -- cuBLAS or
  // hipBLAS via wwr.blas, so both backends. Mutually exclusive
  // with `cublas`; the caller rejects both at once. The DEFAULT on HIP only,
  // because that is where it measured faster: 3.2x on the RX 9060 XT, slower
  // on the RTX 3080, whose emitted tiles already run near its fp64 peak
  // (docs/performance.md, "Native-fp64 BLAS digest"). --digest-emitted turns it off for
  // the A/B; --blas-digest turns it on on CUDA.
  bool blasDigest = std::string_view(kGpuBackendName) == "HIP";
  // Run the f3 consume step (W = eri . L2, both orders) as wwr.blas DGEMMs --
  // cuBLAS or hipBLAS, so both backends -- rather than the emitted loop
  // kernels. On by default because it measured faster on both cards (docs/performance.md,
  // "Consume as a DGEMM"); --consume-emitted turns it off for the A/B.
  bool consumeGemm = true;
  // --ozaki: the three digest GEMMs emulated on int8 tensor cores by the
  // Ozaki scheme (src/ozaki/), on both backends, through the GEMM digests'
  // f3 temp + scatter like --blas-digest. Keeps the digit pairs
  // p + q <= ozakiMaxPairSum (--ozaki-pairs; 14 = all 64, the default). Fewer
  // is faster in principle but not in practice, and P <= 5 makes the
  // cc-pVTZ PC metric refuse (docs/performance.md, "The int8
  // Ozaki digest"). The caller rejects it with --cublas and makes it
  // take precedence over blasDigest.
  bool ozaki = false;
  int ozakiMaxPairSum = 14;
  // --ozaki-check: also run every --ozaki digest GEMM through native
  // wwrblasDgemm and print the largest elementwise difference at the end of
  // the build. A diagnostic: it synchronizes the stream once per GEMM.
  bool ozakiCheck = false;
  // --fused-digest: run a GEMM digest's three GEMMs (dm3, f3 ca, f3 ac) as
  // one, over the stacked [R; W_ca; W_ac] (N = 3 n^2), so L2 is read (and,
  // for --cublas/--ozaki, split) once per tile instead of three times. Needs
  // a GEMM digest; the caller rejects it with the emitted one. Costs a second
  // W (n^2 * width) and an (n^4, 3 n^2) temp in place of the (n^4, n^2) one.
  bool fusedDigest = false;
};

// Device-resident results, each an owning (n_act,)*6 DeviceTensor allocated
// from the caller's DeviceResources pool: freed when the result (or whatever
// they are moved into) goes out of scope. `f3ca` is the build's order 0,
// `f3ac` order 1.
struct RdmBuildResult {
  DeviceTensor dm3;
  DeviceTensor f3ca;
  DeviceTensor f3ac;
  // Includes the uploads and allocations, not context setup: the context
  // comes up when main() creates the
  // DeviceResources, before the build starts.
  double seconds = 0.0;
};

// `h2e` is the physicist-ordered active block (n_act^4), as the golden file
// stores it. Prints the same progress lines the demo always printed (the tile
// plan, the --cublas engaged-bits line, the build time), plus the pool's
// high-water marks at the end of the build.
//
// Every allocation, free, launch, copy, zero-fill, BLAS call and
// --profile event is issued on res.stream(), which is
// synchronized before this returns. The plain BLAS calls (consume, and
// --blas-digest) use res.blas(); --cublas makes its own emulation handle,
// bound to the same stream. The --profile spans land in profile::kRdmBuild;
// the caller prints that table (profile::printReport) once this returns.
RdmBuildResult buildRdmsDevice(const RdmBuildOptions& opt, const Tensor& ci, int64_t norb,
                               int64_t nelecA, int64_t nelecB, const Tensor& h2e,
                               const DeviceResources& res);


}  // namespace nevpt2
