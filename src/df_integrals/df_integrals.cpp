// The body of nevpt2.df_integrals -- see the interface for the method. The
// one table that matters is in slab(): for each external block, which B
// supplies the batched pair X, which supplies the other pair Y, and the
// permutation from C[x,y,z,w] = (xy|zw) into the block's physicist layout.
// Each row was derived from the PySCF `eris[...]` slice generate_golden.py's
// _integral_blocks takes for that block, and `nevpt2_df_demo --check-blocks`
// compares every one against the golden file's four-index arrays.
module nevpt2.df_integrals;

import std;
import :permutation;

import nevpt2.common;
import nevpt2.profile;

import wwr.blas;
// wwrblasStatus_t's error_type specializations: what lets gpuCheck take a
// BLAS status.
import wwr.extension.blas;

namespace nevpt2 {

Result<DfIntegralSource> DfIntegralSource::create(const Tensor& bAA, const Tensor& bCA,
                                                  const Tensor& bVA, const Tensor& bCV,
                                                  const DeviceResources& res) {
  // Ranks first: dim(1)/dim(2) below are only meaningful on rank-3 tensors.
  if (bAA.rank() != 3 || bCA.rank() != 3 || bVA.rank() != 3 || bCV.rank() != 3)
    return err_io("inconsistent B_* shapes in the golden file: every B block must be rank 3");
  const int64_t naux = bAA.dim(0), ncas = bAA.dim(1), ncore = bCA.dim(1), nvirt = bVA.dim(1);
  const bool shapesOk = bAA.dim(2) == ncas && bCA.dim(0) == naux && bCA.dim(2) == ncas &&
                        bVA.dim(0) == naux && bVA.dim(2) == ncas && bCV.dim(0) == naux &&
                        bCV.dim(1) == ncore && bCV.dim(2) == nvirt;
  if (!shapesOk)
    return err_io(std::format(
        "inconsistent B_* shapes in the golden file: B_aa ({},{},{}), B_ca ({},{},{}), "
        "B_va ({},{},{}), B_cv ({},{},{})",
        bAA.dim(0), bAA.dim(1), bAA.dim(2), bCA.dim(0), bCA.dim(1), bCA.dim(2), bVA.dim(0),
        bVA.dim(1), bVA.dim(2), bCV.dim(0), bCV.dim(1), bCV.dim(2)));
  return DfIntegralSource(bAA, bCA, bVA, bCV, res);
}

DfIntegralSource::DfIntegralSource(const Tensor& bAA, const Tensor& bCA, const Tensor& bVA,
                                   const Tensor& bCV, const DeviceResources& res) {
  naux_ = bAA.dim(0);
  ncas_ = bAA.dim(1);
  ncore_ = bCA.dim(1);
  nvirt_ = bVA.dim(1);
  bAA_ = uploadTensor(bAA, res);
  bCA_ = uploadTensor(bCA, res);
  bVA_ = uploadTensor(bVA, res);
  bCV_ = uploadTensor(bCV, res);
  // On res.stream(): ordered after the uploads above, and before any slab
  // work, which the demo issues on the same stream. --profile reports it with
  // the slab builds.
  const profile::Section profileSection(profile::kDfIntegrals);
  bVC_ = deviceTransposeNew(bCV_, {0, 2, 1}, 1.0, res);
}

int64_t DfIntegralSource::extent(ExtBlock b) const {
  switch (b) {
    case ExtBlock::Si:
    case ExtBlock::Sijrs:
    case ExtBlock::Sij:
      return ncore_;
    default:
      return nvirt_;
  }
}

int64_t DfIntegralSource::fullBlockDoubles(ExtBlock b) const {
  const int64_t a = ncas_, c = ncore_, v = nvirt_;
  switch (b) {
    case ExtBlock::Sr: return v * a * a * a;
    case ExtBlock::Si: return a * a * c * a;
    case ExtBlock::Sijrs: return c * v * c * v;
    case ExtBlock::Sijr: return v * a * c * c;
    case ExtBlock::Srsi:
    case ExtBlock::SrsiT: return v * v * c * a;
    case ExtBlock::Srs: return v * v * a * a;
    case ExtBlock::Sij: return a * a * c * c;
    case ExtBlock::Sir1: return v * a * c * a;
    case ExtBlock::Sir2: return v * a * a * c;
  }
  return 0;
}

// C[x, y, z, w] = sum_L X[L, b0 + x, y] Y[L, z, w], then transpose by `axes`
// (slab axis i = C axis axes[i], numpy's convention).
//
// cuBLAS/hipBLAS are column-major; a row-major (rows, cols) matrix is the
// column-major (cols, rows) one. So with M = nz*nw, N = nb*ny, K = naux:
//   A = Y as column-major (M x K), lda = M     -- Y[L, zw] at zw + L*M
//   B = X's batch rows as column-major (N x K), ldb = nx*ny, used transposed
//       -- X[L, b0+x, y] at (b0*ny) + (x*ny + y) + L*(nx*ny)
//   C = column-major (M x N), ldc = M          -- row-major C[(x,y), (z,w)]
DeviceTensor DfIntegralSource::build(const char* label, const DeviceTensor& x, int64_t b0,
                                     int64_t b1, const DeviceTensor& y,
                                     const std::array<int, 4>& axes, const DeviceResources& res) {
  // --profile: the DGEMM and the transpose after it (an einsum launch) both
  // report in the DF-integrals table, not the energy stage that asked for
  // the slab.
  const profile::Section profileSection(profile::kDfIntegrals);
  const int64_t nb = b1 - b0, nx = x.dims[1], ny = x.dims[2], nz = y.dims[1], nw = y.dims[2];
  DeviceTensor c = DeviceTensor::zeros({nb, ny, nz, nw}, res);  // zeroed: right even if K == 0
  peak_ = std::max(peak_, c.size());
  const int64_t m = nz * nw, n = nb * ny;
  if (m > 0 && n > 0 && naux_ > 0) {
    // res.blas() is bound to res.stream() once, at creation -- the stream
    // every other call here is issued on -- so there is no per-slab
    // wwrblasSetStream any more.
    const double one = 1.0, zero = 0.0;
    const double* xBatch = x.data() + b0 * ny;
    // The 32-bit BLAS API: each dimension narrowed once, checked.
    const int mK = narrowTo<int>(m), nK = narrowTo<int>(n), kK = narrowTo<int>(naux_);
    const int ldx = narrowTo<int>(nx * ny);
    profile::time(label, res.stream(), [&] {
      gpuCheck(wwrblasDgemm(res.blas(), WWRBLAS_OP_N, WWRBLAS_OP_T, mK, nK, kK, &one, y.data(), mK,
                            xBatch, ldx, &zero, c.data(), mK));
    });
  }
  if (isIdentity(axes)) return c;

  DeviceTensor out = deviceTransposeNew(c, {axes[0], axes[1], axes[2], axes[3]}, 1.0, res);
  peak_ = std::max(peak_, out.size());
  // `c` is freed as this returns, with no sync first. The transpose reading
  // it is queued on res.stream(), and so is the free (wwrFreeAsync), behind
  // it. A synchronous wwrFree would need the stream synchronized first --
  // once per slab build. A free that did run early is
  // exactly what compute-sanitizer memcheck's memcheck_free_before_read
  // canary shows it reports.
  return out;
}

DeviceTensor DfIntegralSource::slab(ExtBlock b, int64_t b0, int64_t b1,
                                    const DeviceResources& res) {

  // (X batched pair, Y pair, C -> slab axes); C[x,y,z,w] = (xy|zw).
  //   block  physicist slab              X      Y      C          axes
  //   Sr     [r,a,b,c] = (rb|ac)         B_va   B_aa   [r,b,a,c]  0 2 1 3
  //   Si     [t,u,i,v] = (it|uv)         B_ca   B_aa   [i,t,u,v]  1 2 0 3
  //   Sijrs  [i,a,j,b] = (ia|jb)         B_cv   B_cv   [i,a,j,b]  0 1 2 3
  //   Sijr   [r,p,j,i] = (rj|ip)         B_vc   B_ca   [r,j,i,p]  0 3 1 2
  //   Srsi   [x,y,i,t] = (xi|yt)         B_vc   B_va   [x,i,y,t]  0 2 1 3
  //   SrsiT  [y,x,i,t] = (xt|yi)         B_va   B_vc   [x,t,y,i]  2 0 3 1
  //   Srs    [r,s,t,u] = (rt|su)         B_va   B_va   [r,t,s,u]  0 2 1 3
  //   Sij    [t,u,i,j] = (it|ju)         B_ca   B_ca   [i,t,j,u]  1 3 0 2
  //   Sir1   [r,t,i,u] = (ri|tu)         B_vc   B_aa   [r,i,t,u]  0 2 1 3
  //   Sir2   [r,u,t,i] = (rt|iu)         B_va   B_ca   [r,t,i,u]  0 3 1 2
  DeviceTensor t;
  switch (b) {
    case ExtBlock::Sr: t = build("dgemm Sr", bVA_, b0, b1, bAA_, {0, 2, 1, 3}, res); break;
    case ExtBlock::Si: t = build("dgemm Si", bCA_, b0, b1, bAA_, {1, 2, 0, 3}, res); break;
    case ExtBlock::Sijrs:
      t = build("dgemm Sijrs", bCV_, b0, b1, bCV_, {0, 1, 2, 3}, res);
      break;
    case ExtBlock::Sijr: t = build("dgemm Sijr", bVC_, b0, b1, bCA_, {0, 3, 1, 2}, res); break;
    case ExtBlock::Srsi: t = build("dgemm Srsi", bVC_, b0, b1, bVA_, {0, 2, 1, 3}, res); break;
    case ExtBlock::SrsiT:
      t = build("dgemm SrsiT", bVA_, b0, b1, bVC_, {2, 0, 3, 1}, res);
      break;
    case ExtBlock::Srs: t = build("dgemm Srs", bVA_, b0, b1, bVA_, {0, 2, 1, 3}, res); break;
    case ExtBlock::Sij: t = build("dgemm Sij", bCA_, b0, b1, bCA_, {1, 3, 0, 2}, res); break;
    case ExtBlock::Sir1: t = build("dgemm Sir1", bVC_, b0, b1, bAA_, {0, 2, 1, 3}, res); break;
    case ExtBlock::Sir2: t = build("dgemm Sir2", bVA_, b0, b1, bCA_, {0, 3, 1, 2}, res); break;
  }
  return t;  // owning: freed when the caller's slab goes out of scope
}

DeviceTensor DfIntegralSource::activeH2e(const DeviceResources& res) {
  // h2e[t,u,v,w] = (tv|uw): C[t,v,u,w] from B_aa x B_aa, axes 0 2 1 3.
  return build("dgemm active h2e", bAA_, 0, ncas_, bAA_, {0, 2, 1, 3}, res);
}

}  // namespace nevpt2
