// nevpt2.rdm_build:tiles -- the determinant-axis tile plan: the width for a tile
// count, and each tile's (k0, width).
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :tiles;` and from no importer of nevpt2.rdm_build.
// (Used by rdm_build.cpp.)
module nevpt2.rdm_build:tiles;

import std;
import nevpt2.rdm_build;
import nevpt2.common;

namespace nevpt2 {

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
std::vector<std::pair<int, int>> planTiles(int64_t ndet, int64_t nK) {
  std::vector<std::pair<int, int>> tiles;
  for (int64_t k0 = 0; k0 < ndet; k0 += nK) {
    int width = narrowTo<int>(std::min(nK, ndet - k0), "use more --tiles");
    tiles.emplace_back(narrowTo<int>(k0), width);
  }
  return tiles;
}

}  // namespace nevpt2
