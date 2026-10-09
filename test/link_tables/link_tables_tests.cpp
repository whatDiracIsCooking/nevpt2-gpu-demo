// Suites for nevpt2.link_tables (src/rdm/link_tables.cppm).
//
// Host-only ON PURPOSE: the link tables are pure host integer combinatorics --
// no device, no kernel, no golden -- so this binary is not REQUIRES_GPU and
// runs with no card visible. Every comparison is EXACT (integer logic), each
// against a brute-force reference that shares no code with the unit under
// test: strings are enumerated by walking every bit pattern of `norb` bits in
// increasing integer value (the lexical order the module promises), and signs
// come from applying the annihilator, then the creator, one bit at a time.
//
//   NumStringsTests   num_strings against Pascal's triangle (0 out of range);
//                     nlink against nelec + nelec * (norb - nelec)
//   AddressingTests   str2addr / addr2str against the enumeration, and their
//                     round trip over every string
//   ExcitSignTests    excit_sign against the operator-by-operator parity
//   LinkTableTests    gen_linkstr_index against a brute-force table
//   ReverseLinkTests  reverse_link against a brute-force per-target regroup
//
// TU shape: gtest's header FIRST, then `import std;` and the modules -- every
// textual std #include before any import.
#include <gtest/gtest.h>

import std;
import nevpt2.link_tables;

// Not a module unit (gtest is a textual header), so no partition: the
// helpers and suites live in a named namespace instead of an anonymous one.
// One binary per suite file, so a name clash here is a link error.
namespace nevpt2::test::link_tables {

using namespace nevpt2::link_tables;

// Every (norb, nelec) with 0 <= nelec <= norb <= kMaxNorb: small enough that
// the whole 2^norb space is walked by brute force, and it covers the edges
// nelec = 0 (one empty string, no excitations) and nelec = norb (one full
// string, no virtuals).
constexpr int64_t kMaxNorb = 10;

// One 4-int row of a link table: [cre, des, target_addr, sign].
using Row = std::array<int, 4>;

// --- brute-force references ---------------------------------------------------

// C(n, k) from Pascal's triangle -- additions only, unlike the module's
// multiplicative binom.
int64_t pascal(int64_t n, int64_t k) {
  if (k < 0 || n < 0 || k > n) return 0;
  std::vector<std::vector<int64_t>> c(static_cast<std::size_t>(n + 1));
  for (std::size_t i = 0; i < c.size(); ++i) {
    c[i].assign(i + 1, 1);
    for (std::size_t j = 1; j < i; ++j) c[i][j] = c[i - 1][j - 1] + c[i - 1][j];
  }
  return c[static_cast<std::size_t>(n)][static_cast<std::size_t>(k)];
}

// Every `norb`-bit string with `nelec` set bits, in increasing integer value:
// element addr is the string at lexical address addr.
std::vector<std::uint64_t> enumerate(int64_t norb, int64_t nelec) {
  std::vector<std::uint64_t> out;
  const std::uint64_t end = std::uint64_t(1) << norb;
  for (std::uint64_t s = 0; s < end; ++s) {
    int count = 0;
    for (int64_t p = 0; p < norb; ++p) count += static_cast<int>((s >> p) & 1ULL);
    if (count == nelec) out.push_back(s);
  }
  return out;
}

// Occupied orbitals strictly below `p` in `s`, counted bit by bit.
int occupied_below(std::uint64_t s, int64_t p) {
  int n = 0;
  for (int64_t q = 0; q < p; ++q) n += static_cast<int>((s >> q) & 1ULL);
  return n;
}

// Sign of cre^+ des |s> for occupied `des`, applied operator by operator:
// des passes the electrons below it, then cre passes those below it in the
// string des left behind. The diagonal number operator is +1.
int reference_sign(std::uint64_t s, int64_t des, int64_t cre) {
  if (des == cre) return 1;
  int n = occupied_below(s, des);
  const std::uint64_t mid = s & ~(std::uint64_t(1) << des);
  n += occupied_below(mid, cre);
  return (n % 2 == 0) ? 1 : -1;
}

// The forward table, row by row, in the documented order: for each source
// string, its diagonal rows (ascending occupied orbital) first, then every
// occupied -> virtual excitation (occupied ascending, virtual ascending
// within it). Targets are found by searching the enumeration, not by str2addr.
std::vector<std::vector<Row>> reference_forward(int64_t norb, int64_t nelec) {
  const std::vector<std::uint64_t> strings = enumerate(norb, nelec);
  std::vector<std::vector<Row>> table(strings.size());
  for (std::size_t addr = 0; addr < strings.size(); ++addr) {
    const std::uint64_t s = strings[addr];
    std::vector<int64_t> occ, vir;
    for (int64_t p = 0; p < norb; ++p) (((s >> p) & 1ULL) ? occ : vir).push_back(p);
    for (int64_t i : occ) {
      table[addr].push_back({static_cast<int>(i), static_cast<int>(i), static_cast<int>(addr), 1});
    }
    for (int64_t i : occ) {
      for (int64_t a : vir) {
        const std::uint64_t t = (s & ~(std::uint64_t(1) << i)) | (std::uint64_t(1) << a);
        const auto it = std::ranges::find(strings, t);
        table[addr].push_back({static_cast<int>(a), static_cast<int>(i),
                               static_cast<int>(it - strings.begin()), reference_sign(s, i, a)});
      }
    }
  }
  return table;
}

// The flat (na, nl, 4) buffer, row (addr, row) of it.
Row row_of(const std::vector<int>& flat, int64_t nl, int64_t addr, int64_t row) {
  const auto base = static_cast<std::size_t>((addr * nl + row) * 4);
  return {flat[base + 0], flat[base + 1], flat[base + 2], flat[base + 3]};
}

// --- NumStringsTests ----------------------------------------------------------

TEST(NumStringsTests, MatchesPascalsTriangle) {
  for (int64_t norb = 0; norb <= 30; ++norb) {
    for (int64_t nelec = 0; nelec <= norb; ++nelec) {
      EXPECT_EQ(num_strings(norb, nelec), pascal(norb, nelec))
          << "norb=" << norb << " nelec=" << nelec;
    }
  }
}

TEST(NumStringsTests, MatchesTheEnumerationCount) {
  for (int64_t norb = 0; norb <= kMaxNorb; ++norb) {
    for (int64_t nelec = 0; nelec <= norb; ++nelec) {
      EXPECT_EQ(num_strings(norb, nelec), static_cast<int64_t>(enumerate(norb, nelec).size()))
          << "norb=" << norb << " nelec=" << nelec;
    }
  }
}

TEST(NumStringsTests, ZeroOutOfRange) {
  EXPECT_EQ(num_strings(4, -1), 0);
  EXPECT_EQ(num_strings(4, 5), 0);
  EXPECT_EQ(num_strings(0, 1), 0);
  EXPECT_EQ(num_strings(0, 0), 1);
}

TEST(NumStringsTests, NlinkIsDiagonalPlusSingles) {
  for (int64_t norb = 0; norb <= kMaxNorb; ++norb) {
    for (int64_t nelec = 0; nelec <= norb; ++nelec) {
      EXPECT_EQ(nlink(norb, nelec), nelec + nelec * (norb - nelec))
          << "norb=" << norb << " nelec=" << nelec;
    }
  }
}

// --- AddressingTests ----------------------------------------------------------

TEST(AddressingTests, Str2addrIsTheLexicalPosition) {
  for (int64_t norb = 0; norb <= kMaxNorb; ++norb) {
    for (int64_t nelec = 0; nelec <= norb; ++nelec) {
      const std::vector<std::uint64_t> strings = enumerate(norb, nelec);
      for (std::size_t addr = 0; addr < strings.size(); ++addr) {
        ASSERT_EQ(str2addr(norb, nelec, strings[addr]), static_cast<int64_t>(addr))
            << "norb=" << norb << " nelec=" << nelec << " str=" << strings[addr];
      }
    }
  }
}

TEST(AddressingTests, Addr2strIsTheLexicalString) {
  for (int64_t norb = 0; norb <= kMaxNorb; ++norb) {
    for (int64_t nelec = 0; nelec <= norb; ++nelec) {
      const std::vector<std::uint64_t> strings = enumerate(norb, nelec);
      for (std::size_t addr = 0; addr < strings.size(); ++addr) {
        ASSERT_EQ(addr2str(norb, nelec, static_cast<int64_t>(addr)), strings[addr])
            << "norb=" << norb << " nelec=" << nelec << " addr=" << addr;
      }
    }
  }
}

TEST(AddressingTests, RoundTripOverEveryString) {
  for (int64_t norb = 0; norb <= kMaxNorb; ++norb) {
    for (int64_t nelec = 0; nelec <= norb; ++nelec) {
      const int64_t na = num_strings(norb, nelec);
      for (int64_t addr = 0; addr < na; ++addr) {
        const std::uint64_t s = addr2str(norb, nelec, addr);
        ASSERT_EQ(std::popcount(s), nelec) << "norb=" << norb << " addr=" << addr;
        ASSERT_LT(s, std::uint64_t(1) << norb) << "norb=" << norb << " addr=" << addr;
        ASSERT_EQ(str2addr(norb, nelec, s), addr) << "norb=" << norb << " nelec=" << nelec;
      }
    }
  }
}

// --- ExcitSignTests -----------------------------------------------------------

TEST(ExcitSignTests, MatchesOperatorByOperatorParity) {
  // Every string of up to kMaxNorb bits, every occupied des, every cre that is
  // either des itself or unoccupied -- the excitations gen_linkstr_index asks for.
  for (int64_t norb = 0; norb <= kMaxNorb; ++norb) {
    const std::uint64_t end = std::uint64_t(1) << norb;
    for (std::uint64_t s = 0; s < end; ++s) {
      for (int64_t des = 0; des < norb; ++des) {
        if (((s >> des) & 1ULL) == 0) continue;
        for (int64_t cre = 0; cre < norb; ++cre) {
          if (cre != des && ((s >> cre) & 1ULL) != 0) continue;
          ASSERT_EQ(excit_sign(s, des, cre), reference_sign(s, des, cre))
              << "s=" << s << " des=" << des << " cre=" << cre;
        }
      }
    }
  }
}

TEST(ExcitSignTests, HandWorkedCases) {
  // 0b1011 (orbitals 0, 1, 3): 3 -> 2 passes nothing between them.
  EXPECT_EQ(excit_sign(0b1011, 3, 2), 1);
  // 0 -> 2 passes orbital 1.
  EXPECT_EQ(excit_sign(0b1011, 0, 2), -1);
  // 0 -> 4 in 0b1011 passes orbitals 1 and 3.
  EXPECT_EQ(excit_sign(0b1011, 0, 4), 1);
  // The diagonal is always +1.
  EXPECT_EQ(excit_sign(0b1011, 1, 1), 1);
}

// --- LinkTableTests -----------------------------------------------------------

TEST(LinkTableTests, ShapeIsNaByNlinkByFour) {
  for (int64_t norb = 0; norb <= kMaxNorb; ++norb) {
    for (int64_t nelec = 0; nelec <= norb; ++nelec) {
      const std::vector<int> link = gen_linkstr_index(norb, nelec);
      EXPECT_EQ(static_cast<int64_t>(link.size()),
                num_strings(norb, nelec) * nlink(norb, nelec) * 4)
          << "norb=" << norb << " nelec=" << nelec;
    }
  }
}

TEST(LinkTableTests, MatchesBruteForceTable) {
  for (int64_t norb = 0; norb <= kMaxNorb; ++norb) {
    for (int64_t nelec = 0; nelec <= norb; ++nelec) {
      const std::vector<int> link = gen_linkstr_index(norb, nelec);
      const std::vector<std::vector<Row>> ref = reference_forward(norb, nelec);
      const int64_t nl = nlink(norb, nelec);
      ASSERT_EQ(static_cast<int64_t>(ref.size()), num_strings(norb, nelec));
      for (std::size_t addr = 0; addr < ref.size(); ++addr) {
        ASSERT_EQ(static_cast<int64_t>(ref[addr].size()), nl);
        for (int64_t row = 0; row < nl; ++row) {
          ASSERT_EQ(row_of(link, nl, static_cast<int64_t>(addr), row),
                    ref[addr][static_cast<std::size_t>(row)])
              << "norb=" << norb << " nelec=" << nelec << " addr=" << addr << " row=" << row;
        }
      }
    }
  }
}

TEST(LinkTableTests, EveryRowAppliesToItsTarget) {
  // Independent of the row order: applying [cre, des] to the source string
  // gives exactly the target string, and the sign is the operator parity.
  for (int64_t norb = 0; norb <= kMaxNorb; ++norb) {
    for (int64_t nelec = 0; nelec <= norb; ++nelec) {
      const std::vector<std::uint64_t> strings = enumerate(norb, nelec);
      const std::vector<int> link = gen_linkstr_index(norb, nelec);
      const int64_t nl = nlink(norb, nelec);
      for (std::size_t addr = 0; addr < strings.size(); ++addr) {
        const std::uint64_t s = strings[addr];
        for (int64_t row = 0; row < nl; ++row) {
          const Row r = row_of(link, nl, static_cast<int64_t>(addr), row);
          const std::uint64_t t = (s & ~(std::uint64_t(1) << r[1])) | (std::uint64_t(1) << r[0]);
          ASSERT_GE(r[2], 0);
          ASSERT_LT(static_cast<std::size_t>(r[2]), strings.size());
          ASSERT_EQ(strings[static_cast<std::size_t>(r[2])], t)
              << "norb=" << norb << " nelec=" << nelec << " addr=" << addr << " row=" << row;
          ASSERT_EQ(r[3], reference_sign(s, r[1], r[0]));
        }
      }
    }
  }
}

// --- ReverseLinkTests ---------------------------------------------------------

TEST(ReverseLinkTests, MatchesBruteForceRegroup) {
  // For every target, the forward rows that reach it -- in (source, row)
  // order -- with the source address in the address slot.
  for (int64_t norb = 0; norb <= kMaxNorb; ++norb) {
    for (int64_t nelec = 0; nelec <= norb; ++nelec) {
      const int64_t na = num_strings(norb, nelec);
      const int64_t nl = nlink(norb, nelec);
      const std::vector<std::vector<Row>> fwd = reference_forward(norb, nelec);
      std::vector<std::vector<Row>> ref(fwd.size());
      for (std::size_t src = 0; src < fwd.size(); ++src) {
        for (const Row& r : fwd[src]) {
          ref[static_cast<std::size_t>(r[2])].push_back({r[0], r[1], static_cast<int>(src), r[3]});
        }
      }
      const std::vector<int> rev = reverse_link(gen_linkstr_index(norb, nelec), na, nl);
      ASSERT_EQ(static_cast<int64_t>(rev.size()), na * nl * 4);
      for (std::size_t tgt = 0; tgt < ref.size(); ++tgt) {
        // Every target is reached by exactly nl rows, so the inverse is the
        // same shape as the forward table.
        ASSERT_EQ(static_cast<int64_t>(ref[tgt].size()), nl);
        for (int64_t row = 0; row < nl; ++row) {
          ASSERT_EQ(row_of(rev, nl, static_cast<int64_t>(tgt), row),
                    ref[tgt][static_cast<std::size_t>(row)])
              << "norb=" << norb << " nelec=" << nelec << " tgt=" << tgt << " row=" << row;
        }
      }
    }
  }
}

TEST(ReverseLinkTests, ReversingTwiceIsTheIdentity) {
  // Single excitations are self-inverse as a relation, but row order is not
  // preserved; compare per string as multisets.
  for (int64_t norb = 0; norb <= kMaxNorb; ++norb) {
    for (int64_t nelec = 0; nelec <= norb; ++nelec) {
      const int64_t na = num_strings(norb, nelec);
      const int64_t nl = nlink(norb, nelec);
      const std::vector<int> fwd = gen_linkstr_index(norb, nelec);
      const std::vector<int> back = reverse_link(reverse_link(fwd, na, nl), na, nl);
      for (int64_t a = 0; a < na; ++a) {
        std::vector<Row> x, y;
        for (int64_t row = 0; row < nl; ++row) {
          x.push_back(row_of(fwd, nl, a, row));
          y.push_back(row_of(back, nl, a, row));
        }
        std::ranges::sort(x);
        std::ranges::sort(y);
        ASSERT_EQ(x, y) << "norb=" << norb << " nelec=" << nelec << " addr=" << a;
      }
    }
  }
}

}  // namespace nevpt2::test::link_tables
