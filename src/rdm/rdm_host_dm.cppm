// nevpt2.rdm_host_dm -- the active-space 1- and 2-RDMs rebuilt on the HOST
// from a CI vector, and their tangents along a CI direction. Pure host C++
// over nevpt2.link_tables' single-excitation tables: no kernel, no device
// code, no BLAS.
//
// Why it exists: dm1 and dm2 are golden-file INPUTS -- generate_golden.py
// writes what PySCF's fci.rdm.make_dm123 returned -- so nothing in the tree
// computes them from the `ci` beside them, and no derivative with respect to
// the CI vector can be checked against the thing it differentiates. Building
// them here gives both: a free consistency check on the golden
// (test/rdm_host_dm/ rebuilds each committed dm1/dm2 to 1e-12), and the
// bilinear forms whose tangents the analytic-gradient work needs.
//
// CONVENTION, matched to the golden's arrays (PySCF's make_dm123, NOT
// reordered), with E_pq = sum_sigma a^+_{p sigma} a_{q sigma}:
//
//   dm1[p,q]     = <c|E_pq|c>
//   dm2[p,q,r,s] = <c|E_pq E_rs|c>          (so dm2 = <p^+ q r^+ s>, the
//                                            un-normal-ordered one)
//
// Both are quadratic forms in the CI vector, which is what makes the tangents
// trivial: with |v_pq> = E_pq|c> (applyAllE) and a real CI vector,
//
//   dm1[p,q]     = <c|v_pq>
//   dm2[p,q,r,s] = <v_qp|v_rs>
//   dm1_dot[p,q]     = <u|v^c_pq> + <c|v^u_pq>
//   dm2_dot[p,q,r,s] = <v^u_qp|v^c_rs> + <v^c_qp|v^u_rs>
//
// where v^c = applyAllE(c) and v^u = applyAllE(u). dm12Dot returns the
// second pair: the directional derivative d/dt dm12(c + t u) at t = 0.
//
// The CI vector is NOT normalized here, by design: these are the plain
// bilinear forms, so dm12(a c) = a^2 dm12(c) and dm12Dot(c, c) = 2 dm12(c).
// A caller that wants the derivative of the NORMALIZED state's RDMs supplies
// a direction already projected against `ci` (and divides by <c|c>).
//
// Cost: applyAllE materializes norb^2 * ndet doubles (tens of MB at
// CAS(10,10), and it grows with ndet, so this is a small- and medium-CAS
// tool), and dm12 contracts them into dm2 in n^2 (n^2 + 1) / 2 length-ndet
// dot products -- the Gram matrix is symmetric, so half the pairs. This is
// the host reference, not a production path; the device build is
// nevpt2.rdm_build.
export module nevpt2.rdm_host_dm;

import std;
// Re-exported: a build takes and returns Tensors, and create() hands back a
// Result the caller reports.
export import nevpt2.tensor;
export import nevpt2.error_handling;

export namespace nevpt2 {

// One build's pair of arrays: dm1 (norb, norb) and dm2 (norb, norb, norb,
// norb), row-major -- the golden file's layout for its `dm1` / `dm2`. From
// dm12Dot the same two shapes hold the tangents dm1_dot / dm2_dot.
struct HostDm12 {
  Tensor dm1;
  Tensor dm2;
};

// The active space and its forward link tables, built once and reused by
// every build (the tangent needs two applyAllE passes, and a finite-difference
// check many).
class HostDm12Builder {
 public:
  // The builder for CAS(nelecA + nelecB, norb). An out-of-range active space
  // is the caller's input, not our bug, so it comes back as an
  // InvalidConfig Error:
  //   norb outside [1, 32]       -- past 32 orbitals a string address no
  //                                 longer fits the `int` nevpt2.link_tables
  //                                 stores it in (and the working set below
  //                                 is hopeless anyway)
  //   nelecA / nelecB outside [0, norb]
  //   norb^2 * ndet past 2^31 doubles -- the applyAllE working set, capped so
  //                                 the index arithmetic cannot overflow
  //                                 rather than as a memory policy
  [[nodiscard]] static Result<HostDm12Builder> create(int64_t norb, int64_t nelecA,
                                                      int64_t nelecB);

  [[nodiscard]] int64_t norb() const { return norb_; }
  [[nodiscard]] int64_t nelecA() const { return nelecA_; }
  [[nodiscard]] int64_t nelecB() const { return nelecB_; }
  [[nodiscard]] int64_t nstringsA() const { return na_; }
  [[nodiscard]] int64_t nstringsB() const { return nb_; }
  [[nodiscard]] int64_t ndet() const { return na_ * nb_; }

  // v[p,q,:] = E_pq |ci>, flat row-major (norb, norb, ndet) -- the one
  // primitive dm12 and dm12Dot are built from, exported because it is also
  // the sigma-vector-shaped object a CI derivative needs. `ci` is the
  // (nstringsA, nstringsB) CI vector; a different shape is the caller's
  // broken invariant (check(), the abort tier), since it derives the shape
  // from the same norb / nelec this builder was created with.
  [[nodiscard]] std::vector<double> applyAllE(const Tensor& ci) const;

  // dm1[p,q] = <ci|E_pq|ci> and dm2[p,q,r,s] = <ci|E_pq E_rs|ci>.
  [[nodiscard]] HostDm12 dm12(const Tensor& ci) const;

  // The tangents: d/dt dm1(ci + t dir) and d/dt dm2(ci + t dir) at t = 0,
  // exactly (both forms are quadratic, so the derivative is the bilinear
  // cross term and no step size enters). `dir` has `ci`'s shape.
  [[nodiscard]] HostDm12 dm12Dot(const Tensor& ci, const Tensor& dir) const;

 private:
  HostDm12Builder(int64_t norb, int64_t nelecA, int64_t nelecB, int64_t na, int64_t nb);

  // ci's shape against this builder's, through the abort tier. `what` names
  // the argument in the message.
  void checkShape(const Tensor& ci, std::string_view what,
                  std::source_location loc = std::source_location::current()) const;

  int64_t norb_;
  int64_t nelecA_;
  int64_t nelecB_;
  int64_t na_;
  int64_t nb_;
  int64_t nlinkA_;
  int64_t nlinkB_;
  // The forward tables, one row [cre, des, target, sign] per single
  // excitation, exactly as nevpt2.link_tables lays them out.
  std::vector<int> linkA_;
  std::vector<int> linkB_;
};

}  // namespace nevpt2
