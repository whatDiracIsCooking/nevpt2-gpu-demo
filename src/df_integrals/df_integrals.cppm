// nevpt2.df_integrals -- the external SC-NEVPT2 integral blocks, built slab by
// slab from density-fitted three-index tensors, so no full four-index external
// block ever exists on the device.
//
// Input: the four MO blocks of the fitted (L|pq) tensor that SC-NEVPT2 needs
// (generate_golden.py --df writes them; symmetric-metric convention, so
// (pq|rs) = sum_L B[L,p,q] B[L,r,s] with no metric in between):
//     B_aa (L, a, a)   B_ca (L, c, a)   B_va (L, v, a)   B_cv (L, c, v)
// No (L|vv) block: no SC-NEVPT2 class has two virtual indices in one charge
// distribution, which is what keeps the density-fitted variant cheap.
//
// Every slab is the same two steps (the second skipped when the block's layout
// is already C's):
//   1. C[x, y, z, w] = sum_L X[L, x, y] Y[L, z, w] for x in the batch -- ONE
//      wwrblasDgemm (cuBLAS or hipBLAS through WarpWraps' wwr.blas, so this is
//      not a CUDA-only path), with the batch a contiguous run of X's rows;
//   2. a permutation of C into the block's physicist layout (the generic
//      einsum kernel's transpose), since energy.cpp's subscripts expect the
//      blocks exactly as the integral-direct golden file stores them.
// The batched index is always the FIRST index of X's pair, which is why B_cv
// is also kept transposed (B_vc): several blocks batch over a virtual that
// sits second in (L|cv).
export module nevpt2.df_integrals;

import std;

// Re-exported: the source IS an nevpt2::IntegralSource, built from host
// Tensors, and slabs are DeviceTensors.
export import nevpt2.energy;

export namespace nevpt2 {

class DfIntegralSource final : public IntegralSource {
 public:
  // Uploads the four B blocks (and builds B_vc on the device) into res's pool,
  // on res.stream(). The B blocks are owning DeviceTensors, freed with the
  // source; their buffers co-own `res`, so the source no longer has to be
  // destroyed before it (until then it kept a reference to `res`
  // and freed them by hand). Every slab GEMM runs on res.blas(), the handle
  // DeviceResources bound to its stream once (until then this
  // source created its own handle and re-bound it per slab).
  //
  // B blocks whose ranks or shapes disagree (a malformed golden file) are an
  // IO Error, returned before anything is uploaded.
  static Result<DfIntegralSource> create(const Tensor& bAA, const Tensor& bCA,
                                         const Tensor& bVA, const Tensor& bCV,
                                         const DeviceResources& res);

  int64_t extent(ExtBlock b) const override;
  // An owning slab, built for [b0, b1) alone.
  DeviceTensor slab(ExtBlock b, int64_t b0, int64_t b1, const DeviceResources& res) override;

  // The physicist-ordered active block h2e[t,u,v,w] = (tv|uw), from B_aa --
  // what the RDM build's f3 digests and every class's intermediates contract.
  // Caller owns the result (allocated from res's pool).
  DeviceTensor activeH2e(const DeviceResources& res);

  int64_t naux() const { return naux_; }
  // Largest single device allocation a slab build has made so far (the GEMM
  // result or the permuted slab), in doubles -- the high-water mark the
  // batch size controls.
  int64_t peakSlabDoubles() const { return peak_; }
  // Forget the high-water mark (after a diagnostic that built whole blocks).
  void resetPeak() { peak_ = 0; }
  // The same block, unbatched, in doubles: what FullBlockSource would hold.
  int64_t fullBlockDoubles(ExtBlock b) const;

 private:
  // create()'s upload, once the shapes are known to agree.
  DfIntegralSource(const Tensor& bAA, const Tensor& bCA, const Tensor& bVA,
                   const Tensor& bCV, const DeviceResources& res);

  // `label` names the build in the --profile DF-integrals table.
  DeviceTensor build(const char* label, const DeviceTensor& x, int64_t b0, int64_t b1,
                     const DeviceTensor& y, const std::array<int, 4>& axes,
                     const DeviceResources& res);

  DeviceTensor bAA_, bCA_, bVA_, bCV_, bVC_;
  int64_t naux_ = 0, ncas_ = 0, ncore_ = 0, nvirt_ = 0;
  int64_t peak_ = 0;

};

}  // namespace nevpt2
