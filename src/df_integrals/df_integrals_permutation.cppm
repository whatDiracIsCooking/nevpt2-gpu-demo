// nevpt2.df_integrals:permutation -- isIdentity, which tells slab() a block needs
// no permute after its DGEMM.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :permutation;` and from no importer of nevpt2.df_integrals.
// (Used by df_integrals.cpp.)
module nevpt2.df_integrals:permutation;

import std;
import nevpt2.df_integrals;

namespace nevpt2 {

bool isIdentity(const std::array<int, 4>& axes) {
  return axes == std::array<int, 4>{0, 1, 2, 3};
}

}  // namespace nevpt2
