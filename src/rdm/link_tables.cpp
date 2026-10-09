module nevpt2.link_tables;

import std;
import :binomial;

namespace nevpt2::link_tables {

int64_t num_strings(int64_t norb, int64_t nelec) {
  if (nelec < 0 || nelec > norb) return 0;
  return binom(norb, nelec);
}

int64_t str2addr(int64_t norb, int64_t nelec, std::uint64_t str_) {
  int64_t addr = 0;
  int64_t n = nelec;
  for (int64_t p = norb - 1; p >= 0; --p) {
    if ((str_ >> p) & 1ULL) {
      addr += num_strings(p, n);
      --n;
    }
  }
  return addr;
}

std::uint64_t addr2str(int64_t norb, int64_t nelec, int64_t addr) {
  std::uint64_t str_ = 0;
  int64_t n = nelec;
  for (int64_t p = norb - 1; p >= 0; --p) {
    if (n == 0) break;
    int64_t c = num_strings(p, n);
    if (c <= addr) {
      str_ |= (std::uint64_t(1) << p);
      addr -= c;
      --n;
    }
  }
  return str_;
}

int excit_sign(std::uint64_t str_, int64_t des, int64_t cre) {
  if (des == cre) return 1;
  int64_t lo = std::min(des, cre);
  int64_t hi = std::max(des, cre);
  std::uint64_t mask = ((std::uint64_t(1) << hi) - 1) & ~((std::uint64_t(1) << (lo + 1)) - 1);
  int popcount = std::popcount(str_ & mask);
  return (popcount & 1) ? -1 : 1;
}

std::vector<int> gen_linkstr_index(int64_t norb, int64_t nelec) {
  int64_t na = num_strings(norb, nelec);
  int64_t nvir = norb - nelec;
  int64_t nl = nelec + nelec * nvir;
  std::vector<int> link(narrowTo<std::size_t>(na * nl * 4));

  std::vector<int64_t> occ, vir;
  occ.reserve(narrowTo<std::size_t>(nelec));
  vir.reserve(narrowTo<std::size_t>(nvir));
  for (int64_t addr = 0; addr < na; ++addr) {
    std::uint64_t s0 = addr2str(norb, nelec, addr);
    occ.clear();
    vir.clear();
    for (int64_t p = 0; p < norb; ++p) {
      if ((s0 >> p) & 1ULL) occ.push_back(p);
      else vir.push_back(p);
    }
    int64_t row = 0;
    // The one place an entry meets the int storage the kernels read.
    auto put = [&](int64_t cre, int64_t des, int64_t target, int sign) {
      int64_t base = (addr * nl + row) * 4;
      link[base + 0] = narrowTo<int>(cre);
      link[base + 1] = narrowTo<int>(des);
      link[base + 2] = narrowTo<int>(target, "a string address past int");
      link[base + 3] = sign;
      ++row;
    };
    // diagonal number-operator terms: i^+ i | s0 > = | s0 >
    for (int64_t i : occ) put(i, i, addr, 1);
    // occupied -> virtual single excitations: a^+ i | s0 >
    for (int64_t i : occ) {
      for (int64_t a : vir) {
        std::uint64_t s1 = (s0 & ~(std::uint64_t(1) << i)) | (std::uint64_t(1) << a);
        int sign = excit_sign(s0, i, a);
        put(a, i, str2addr(norb, nelec, s1), sign);
      }
    }
  }
  return link;
}

std::vector<int> reverse_link(const std::vector<int>& forward, int64_t na, int64_t nl) {
  std::vector<int> rev(forward.size());
  std::vector<int64_t> fill(narrowTo<std::size_t>(na), 0);
  for (int64_t a = 0; a < na; ++a) {
    // Every source address a is a target address too, so it already fit
    // the int storage when gen_linkstr_index wrote it.
    const int src = narrowTo<int>(a);
    for (int64_t row = 0; row < nl; ++row) {
      int64_t base = (a * nl + row) * 4;
      int cre = forward[base + 0];
      int des = forward[base + 1];
      int64_t a1 = forward[base + 2];
      int sign = forward[base + 3];
      int64_t rbase = (a1 * nl + fill[a1]) * 4;
      rev[rbase + 0] = cre;
      rev[rbase + 1] = des;
      rev[rbase + 2] = src;
      rev[rbase + 3] = sign;
      ++fill[a1];
    }
  }
  return rev;
}

}  // namespace nevpt2::link_tables
