// nevpt2.link_tables:binomial -- binom(n, k), the count of n-orbital strings
// with k electrons that num_strings returns. An internal partition: nothing
// here is exported, so it is reachable from link_tables.cpp
// (`import :binomial;`) and from no importer of nevpt2.link_tables.
module nevpt2.link_tables:binomial;

import std;
import nevpt2.link_tables;

namespace nevpt2::link_tables {

int64_t binom(int64_t n, int64_t k) {
  if (k < 0 || k > n) return 0;
  if (k > n - k) k = n - k;
  int64_t r = 1;
  for (int64_t i = 0; i < k; ++i) {
    r = r * (n - i) / (i + 1);
  }
  return r;
}

}  // namespace nevpt2::link_tables
