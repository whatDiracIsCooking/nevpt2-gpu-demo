// nevpt2.rdm_plan -- the RDM build's pure host index arithmetic: the
// determinant-axis tile plan (the width for a tile count, each tile's
// (k0, width)), the column / k chunking the BLAS consume GEMM is issued in, and
// the permuted integral copy it reads. No device, no BLAS: nevpt2.rdm_build
// (rdm_build.cpp, and its :blas partition's consumeGemm) does the launching and
// imports these; they are exported so the unit tier can check them
// (test/rdm_plan/). Carved out of the :tiles and :blas partitions unchanged.
export module nevpt2.rdm_plan;

import std;
// Re-exported: every extent here is an int64_t, and planTiles narrows with
// narrowTo.
export import nevpt2.common;

export namespace nevpt2 {

// The tile width that splits the determinant axis into exactly `nTiles`
// tiles. See
// docs/performance.md ("Why CAS(10,10) needs tiling") for why this is
// computed explicitly rather than asking the driver for "however much is
// free": at CAS(10,10) that under-counts real peak device usage and can
// corrupt device memory rather than failing cleanly.
int64_t tileWidthForCount(int64_t ndet, int64_t nTiles) {
  return idivup(ndet, nTiles);
}

// (k0, width) per tile, as the int the RDM-build launchers take. A width past
// int is a tile too wide -- more tiles narrow it; a k0 past int is a
// determinant space too large for the kernels' 32-bit indexing at any count.
// Either aborts (narrowTo, the abort tier).
std::vector<std::pair<int, int>> planTiles(int64_t ndet, int64_t nK) {
  std::vector<std::pair<int, int>> tiles;
  for (int64_t k0 = 0; k0 < ndet; k0 += nK) {
    int width = narrowTo<int>(std::min(nK, ndet - k0), "use more --tiles");
    tiles.emplace_back(narrowTo<int>(k0), width);
  }
  return tiles;
}

// The BLAS consume's permuted integral copy (see nevpt2.rdm_build:blas's
// consumeGemm, which reads it as E[x,a,p,q] = eri[a,x,q,p]).
// eri is chemists'-order eriF3, row-major [a, x, q, p]; out is [x, a, p, q].
std::vector<double> permuteEriConsume(const std::vector<double>& eri, int64_t n) {
  std::vector<double> out(eri.size());
  int64_t i = 0;
  for (int64_t x = 0; x < n; ++x)
    for (int64_t a = 0; a < n; ++a)
      for (int64_t p = 0; p < n; ++p)
        for (int64_t q = 0; q < n; ++q) out[i++] = eri[((a * n + x) * n + q) * n + p];
  return out;
}

// Split [0, total) into the fewest near-equal chunks of at most `maxLen`.
// total * (c + 1) stays far inside int64_t: total is n or n^2 here.
std::vector<std::pair<int64_t, int64_t>> evenChunks(int64_t total, int64_t maxLen) {
  int64_t count = total / maxLen + (total % maxLen != 0);  // ceil, no overflow at the max
  std::vector<std::pair<int64_t, int64_t>> out;
  for (int64_t c = 0; c < count; ++c) out.emplace_back(total * c / count, total * (c + 1) / count);
  return out;
}

}  // namespace nevpt2
