// The body of nevpt2.rdm_host_dm -- see the interface for the convention and
// the formulas. Everything here is a dot product over the determinant axis:
// applyAllE spreads the CI vector over the n^2 single-excitation operators
// with the forward link tables, and dm12 / dm12Dot contract the result.
module nevpt2.rdm_host_dm;

import std;
import nevpt2.link_tables;

namespace nevpt2 {

// Not exported: the determinant-axis dot product every contraction below is.
double hostDot(const double* x, const double* y, const int64_t n) {
  double s = 0.0;
  for (int64_t i = 0; i < n; ++i) s += x[i] * y[i];
  return s;
}

// Not exported: the flat (p, q) pair index of (q, p) -- dm2 reads the Gram
// matrix with its first pair transposed.
int64_t transposePair(const int64_t x, const int64_t n) { return (x % n) * n + x / n; }

HostDm12Builder::HostDm12Builder(const int64_t norb, const int64_t nelecA, const int64_t nelecB,
                                 const int64_t na, const int64_t nb)
    : norb_(norb),
      nelecA_(nelecA),
      nelecB_(nelecB),
      na_(na),
      nb_(nb),
      nlinkA_(link_tables::nlink(norb, nelecA)),
      nlinkB_(link_tables::nlink(norb, nelecB)),
      linkA_(link_tables::gen_linkstr_index(norb, nelecA)),
      linkB_(link_tables::gen_linkstr_index(norb, nelecB)) {}

Result<HostDm12Builder> HostDm12Builder::create(const int64_t norb, const int64_t nelecA,
                                                const int64_t nelecB) {
  // 32 orbitals: C(32,16) string addresses still fit the `int`
  // nevpt2.link_tables stores a target address in.
  constexpr int64_t kMaxNorb = 32;
  // norb^2 * ndet doubles, the applyAllE working set. An overflow guard on
  // the index arithmetic, not a memory policy: 2^31 doubles is 16 GiB, far
  // past what any case in this tree asks for.
  constexpr int64_t kMaxElements = int64_t{1} << 31;

  if (norb < 1 || norb > kMaxNorb)
    return err_config(
        std::format("host dm1/dm2: norb = {} is outside [1, {}]", norb, kMaxNorb));
  if (nelecA < 0 || nelecA > norb || nelecB < 0 || nelecB > norb)
    return err_config(
        std::format("host dm1/dm2: {} alpha / {} beta electrons in {} orbitals -- each spin's "
                    "count must lie in [0, {}]",
                    nelecA, nelecB, norb, norb));

  const int64_t na = link_tables::num_strings(norb, nelecA);
  const int64_t nb = link_tables::num_strings(norb, nelecB);
  // Each step divides before it multiplies, so nothing here can overflow.
  const int64_t npair = norb * norb;
  if (na > kMaxElements / npair || nb > kMaxElements / (npair * na))
    return err_config(std::format(
        "host dm1/dm2: CAS({},{}) needs {:.3g} doubles of working set (norb^2 * ndet), past the "
        "{} this host builder allows",
        nelecA + nelecB, norb, static_cast<double>(npair) * static_cast<double>(na) *
                                   static_cast<double>(nb),
        kMaxElements));

  return HostDm12Builder(norb, nelecA, nelecB, na, nb);
}

void HostDm12Builder::checkShape(const Tensor& ci, const std::string_view what,
                                 const std::source_location loc) const {
  check(ci.rank() == 2 && ci.dim(0) == na_ && ci.dim(1) == nb_,
        std::format("host dm1/dm2: {} must be the ({}, {}) CI vector of CAS({},{})", what, na_,
                    nb_, nelecA_ + nelecB_, norb_),
        loc);
}

std::vector<double> HostDm12Builder::applyAllE(const Tensor& ci) const {
  checkShape(ci, "ci");
  const int64_t n = norb_;
  const int64_t nd = ndet();
  std::vector<double> v(narrowTo<std::size_t>(n * n * nd), 0.0);
  double* const vp = v.data();
  const double* const cp = ci.data().data();

  // A row [cre, des, target, sign] of string `addr` says
  // E_{cre,des} |addr> = sign |target>, for that spin alone; the other spin's
  // string index rides along.
  const int* const la = linkA_.data();
  for (int64_t a = 0; a < na_; ++a)
    for (int64_t l = 0; l < nlinkA_; ++l) {
      const int* const row = la + (a * nlinkA_ + l) * 4;
      double* const out = vp + (row[0] * n + row[1]) * nd + row[2] * nb_;
      const double* const in = cp + a * nb_;
      const double sign = row[3];
      for (int64_t b = 0; b < nb_; ++b) out[b] += sign * in[b];
    }

  const int* const lb = linkB_.data();
  for (int64_t b = 0; b < nb_; ++b)
    for (int64_t l = 0; l < nlinkB_; ++l) {
      const int* const row = lb + (b * nlinkB_ + l) * 4;
      double* const out = vp + (row[0] * n + row[1]) * nd + row[2];
      const double* const in = cp + b;
      const double sign = row[3];
      for (int64_t a = 0; a < na_; ++a) out[a * nb_] += sign * in[a * nb_];
    }

  return v;
}

HostDm12 HostDm12Builder::dm12(const Tensor& ci) const {
  const int64_t n = norb_;
  const int64_t nd = ndet();
  const int64_t npair = n * n;
  const std::vector<double> v = applyAllE(ci);  // checks ci's shape
  HostDm12 out{Tensor({n, n}), Tensor({n, n, n, n})};
  const double* const vp = v.data();
  const double* const cp = ci.data().data();

  for (int64_t x = 0; x < npair; ++x) out.dm1.flatRef(x) = hostDot(cp, vp + x * nd, nd);

  // dm2[p,q,r,s] = <v_qp|v_rs> is the Gram matrix of the n^2 vectors, read
  // with its first pair index transposed. Gram is symmetric, so one dot
  // product per unordered pair fills two slots -- the same slot when x == y,
  // and every slot exactly once over the triangle.
  for (int64_t x = 0; x < npair; ++x)
    for (int64_t y = x; y < npair; ++y) {
      const double g = hostDot(vp + x * nd, vp + y * nd, nd);
      out.dm2.flatRef(transposePair(x, n) * npair + y) = g;
      out.dm2.flatRef(transposePair(y, n) * npair + x) = g;
    }
  return out;
}

HostDm12 HostDm12Builder::dm12Dot(const Tensor& ci, const Tensor& dir) const {
  checkShape(dir, "dir");
  const int64_t n = norb_;
  const int64_t nd = ndet();
  const int64_t npair = n * n;
  const std::vector<double> vc = applyAllE(ci);  // checks ci's shape
  const std::vector<double> vu = applyAllE(dir);
  HostDm12 out{Tensor({n, n}), Tensor({n, n, n, n})};
  const double* const vcp = vc.data();
  const double* const vup = vu.data();
  const double* const cp = ci.data().data();
  const double* const up = dir.data().data();

  // dm1_dot[p,q] = <u|E_pq|c> + <c|E_pq|u>.
  for (int64_t x = 0; x < npair; ++x)
    out.dm1.flatRef(x) = hostDot(up, vcp + x * nd, nd) + hostDot(cp, vup + x * nd, nd);

  // dm2_dot[p,q,r,s] = <v^u_qp|v^c_rs> + <v^c_qp|v^u_rs>. Not a Gram matrix,
  // but still symmetric under swapping the pairs (each slot's value is the
  // same sum read the other way round), so the same triangle serves.
  for (int64_t x = 0; x < npair; ++x)
    for (int64_t y = x; y < npair; ++y) {
      const double m = hostDot(vup + x * nd, vcp + y * nd, nd) +
                       hostDot(vcp + x * nd, vup + y * nd, nd);
      out.dm2.flatRef(transposePair(x, n) * npair + y) = m;
      out.dm2.flatRef(transposePair(y, n) * npair + x) = m;
    }
  return out;
}

}  // namespace nevpt2
