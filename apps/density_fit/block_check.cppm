// nevpt2.app.density_fit:block_check -- --check-blocks: every DF-built external block
// compared with the golden file's four-index array.
// An interface partition that exports nothing: main.cppm is the primary
// interface, which may import only interface partitions and must re-export
// them ([module.unit]), so it does `export import :block_check;` -- and since no
// declaration here is `export`ed, every one keeps module linkage, reachable
// from main.cppm and from no importer (nothing imports a demo anyway).
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry

export module nevpt2.app.density_fit:block_check;

import std;
import nevpt2.df_integrals;
import nevpt2.golden;

double maxAbsDiff(const nevpt2::Tensor& a, const nevpt2::Tensor& b) {
  if (a.dims() != b.dims()) return std::numeric_limits<double>::infinity();
  double m = 0.0;
  for (std::int64_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::fabs(a.flat(i) - b.flat(i)));
  return m;
}

// Builds every external block whole through `src`, downloads it and compares
// it with the golden file's four-index array of the same name. Returns false
// if any differs by more than `tol` (they are the same B contracted by DGEMM
// vs. NumPy, so ~1e-15 is expected).
bool checkBlocks(nevpt2::DfIntegralSource& src, const nevpt2::GoldenFile& g,
                 const nevpt2::Tensor& h2eOurs, double tol, const nevpt2::DeviceResources& res) {
  // Each download below synchronizes the stream before the host reads it.
  const wwr::wwrStream_t stream = res.stream();
  using nevpt2::ExtBlock;
  bool ok = true;
  std::printf("\n--check-blocks: DF-built blocks vs the golden four-index arrays\n");
  double d = maxAbsDiff(h2eOurs, g.get("h2e"));
  ok = ok && d < tol;
  std::printf("  %-12s max|delta|=%.2e\n", "h2e", d);
  for (ExtBlock b : {ExtBlock::Sr, ExtBlock::Si, ExtBlock::Sijrs, ExtBlock::Sijr, ExtBlock::Srsi,
                     ExtBlock::Srs, ExtBlock::Sij, ExtBlock::Sir1, ExtBlock::Sir2}) {
    nevpt2::DeviceTensor whole = src.slab(b, 0, src.extent(b), res);
    d = maxAbsDiff(nevpt2::downloadTensor(whole, stream),
                   g.get(nevpt2::extBlockName(b)));
    ok = ok && d < tol;
    std::printf("  %-12s max|delta|=%.2e\n", nevpt2::extBlockName(b), d);
  }
  // SrsiT is Srsi batched along its second axis; check a middle batch of it
  // against the same golden array, sliced the same way on the host.
  const std::int64_t nv = src.extent(ExtBlock::SrsiT);
  if (nv > 0) {
    const std::int64_t b0 = nv / 3, b1 = std::max(b0 + 1, 2 * nv / 3);
    nevpt2::DeviceTensor part = src.slab(ExtBlock::SrsiT, b0, b1, res);
    nevpt2::Tensor ours = nevpt2::downloadTensor(part, stream);
    const nevpt2::Tensor& ref = g.get("h2e_v_Srsi");
    double m = 0.0;
    for (std::int64_t s = 0; s < ref.dim(0); ++s)
      for (std::int64_t r = b0; r < b1; ++r)
        for (std::int64_t i = 0; i < ref.dim(2); ++i)
          for (std::int64_t a = 0; a < ref.dim(3); ++a)
            m = std::max(m, std::fabs(ours(s, r - b0, i, a) - ref(s, r, i, a)));
    ok = ok && m < tol;
    std::print("  {:<12} max|delta|={:.2e}  (rows {}..{} of axis 1)\n", "SrsiT", m, b0, b1 - 1);
  }
  std::printf("--check-blocks: %s (tolerance %.0e)\n", ok ? "ok" : "MISMATCH", tol);
  return ok;
}
