// nevpt2.link_tables -- determinant strings, combinatorial-number-system addressing, and the
// Knowles-Handy single-excitation link table: num_strings/nlink/
// str2addr/addr2str/excit_sign/gen_linkstr_index/reverse_link. Pure host-side integer
// combinatorics, no GPU.
//
// Convention (matched to PySCF's fci.cistring, the
// normative spec the CI vector a producer hands over follows): a single-spin
// string is an unsigned integer whose set bits mark occupied orbitals (bit i
// = orbital i, lowest bit = lowest orbital). Strings are ordered lexically by
// integer value; str2addr/addr2str are exact inverses over
// [0, num_strings(norb, nelec)). link_index[addr] holds one row
// [cre, des, target_addr, sign] per single excitation cre^+ des that takes
// the string at addr to a nonzero string, diagonal (cre==des) rows first.
export module nevpt2.link_tables;

import std;
// Re-exported: every count and address here is an int64_t.
export import nevpt2.common;

export namespace nevpt2::link_tables {

// C(norb, nelec), or 0 if nelec is out of [0, norb].
int64_t num_strings(int64_t norb, int64_t nelec);

// Rows per string in the link table: nelec diagonal terms plus
// nelec * (norb - nelec) genuine single excitations.
inline int64_t nlink(int64_t norb, int64_t nelec) {
  return nelec + nelec * (norb - nelec);
}

// Lexical address of `str_` (nelec set bits) by the combinatorial number
// system -- the exact inverse of addr2str.
int64_t str2addr(int64_t norb, int64_t nelec, std::uint64_t str_);

// The string at lexical address `addr` -- the exact inverse of str2addr.
std::uint64_t addr2str(int64_t norb, int64_t nelec, int64_t addr);

// Parity of cre^+ des | string> (both orbitals already known distinct or
// equal, occupation not checked here -- gen_linkstr_index only calls this on
// valid excitations).
int excit_sign(std::uint64_t str_, int64_t des, int64_t cre);

// Knowles-Handy single-excitation link table for one spin, `nelec` electrons
// in `norb` orbitals. Returns a flat row-major buffer of shape
// (na, nlink, 4) with na = num_strings(norb, nelec), each row
// [cre, des, target_addr, sign] -- the exact layout
// pyscf.fci.cistring.gen_linkstr_index returns, and what the generic produce
// kernel's flink_a/flink_b/rlink_a/rlink_b arguments expect (rlink_* is this
// table's per-target reverse: reverse_link() below). The storage is int
// because the kernels read it so; each entry is narrowed into it, checked,
// exactly once, where it is written.
std::vector<int> gen_linkstr_index(int64_t norb, int64_t nelec);

// Sources-per-target inverse of a forward link table (same shape, same
// nlink rows per string). `na` is
// the string count and `nl` the per-string row count the flat `forward`
// buffer was built with.
std::vector<int> reverse_link(const std::vector<int>& forward, int64_t na, int64_t nl);


}  // namespace nevpt2::link_tables
