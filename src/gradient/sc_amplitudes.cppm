// nevpt2.sc_amplitudes -- the strongly-contracted amplitude T, the Lagrangian
// multipliers P / Q / R the SC-NEVPT2 gradient's source terms are assembled
// from, and the subscript rewriting that turns one of nevpt2.energy's class
// einsums into the derivative of that term. Pure host arithmetic and string
// handling -- no device, no GPU -- so the unit tier checks it card-free
// (test/gradient/); nevpt2.gradient re-exports it.
//
// Reference: Park, J. Chem. Theory Comput. 2019 (arXiv:1907.10180),
// "Analytical Gradient Theory for SC- and PC-NEVPT2"; the equation numbers
// below are that paper's. docs/references.md lists it.
//
// THE AMPLITUDES ARE ALREADY IMPLICIT IN normToEnergy. Per perturber k a
// strongly-contracted class carries three numbers, and nevpt2.energy's slab
// walk hands all three to nevpt2.energy_finish's normToEnergy (PySCF's
// _norm_to_energy):
//
//   N_k      = <Psi_k|H|Psi^(0)> = <Psi_k|Psi_k>, the perturber norm (Eq. 11)
//   H_k      = the active-space part of <Psi_k|H^(0) - E^(0)|Psi_k>
//   Delta_k  = its orbital-energy part (the `diff` argument: a sum of
//              generalized-Fock eigenvalues, e.g. eps_r + eps_s for Srs)
//
// with E_k = -N_k / (Delta_k + H_k / N_k). That denominator's reciprocal IS
// the amplitude of Park Eq. 12,
//
//   T_k = -1 / (Delta_k + H_k / N_k),      E_k = T_k N_k        (Eqs. 12, 15)
//
// so nothing new is computed here -- the amplitude is read out of the same
// division normToEnergy already does, including its vanishing-norm guard
// (|N_k| <= NUMERICAL_ZERO: no amplitude, no energy, the norm still counted).
//
// THE MULTIPLIERS ARE ITS DERIVATIVES. The integrals reach E only through
// N_k, H_k and Delta_k, so the gradient needs just three numbers per
// perturber -- the partial derivatives of E = sum_k T_k N_k. Eliminating H_k
// with H_k / N_k = -1 / T_k - Delta_k gives Park Eqs. 36-38:
//
//   P_k = dE / dN_k     = 2 T_k + Delta_k T_k^2                   (Eq. 36)
//   Q_k = dE / dH_k     = T_k^2                                   (Eq. 37)
//   R_k = dE / dDelta_k = N_k T_k^2                               (Eq. 38)
//
// and the identity every assembled pseudodensity is checked against,
//
//   P_k N_k + Q_k H_k = T_k N_k = E_k                             (Eq. 40)
//
// (P_k N_k + Q_k H_k = 2 T_k N_k + T_k^2 (Delta_k N_k + H_k) and
// Delta_k N_k + H_k = -N_k / T_k). R_k is what Eq. 39's orbital-energy
// pseudodensity d^Fock is built from, and nevpt2.gradient's full source term
// contracts it against the orbital energies' response.
//
// AND THE PSEUDOCANONICAL MULTIPLIERS ARE HERE TOO (Park Eqs. 56-57, the
// bottom of this file): one division per core and per virtual pair of the
// assembled source term, which is pure host arithmetic on a matrix and so
// belongs beside the other multipliers rather than in the device half.
export module nevpt2.sc_amplitudes;

import std;
// Re-exported: NUMERICAL_ZERO is the guard normToEnergy applies, and the
// identity above is checked against normToEnergy's own answer. It re-exports
// nevpt2.tensor in turn -- Eqs. 56-57 read a Tensor and hand two back.
export import nevpt2.energy_finish;
// Re-exported: check() is the abort tier for a malformed call, and Eqs. 56-57
// hand back a Result -- a degeneracy they cannot divide through is a
// NUMERICAL REFUSAL, not our bug.
export import nevpt2.error_handling;

export namespace nevpt2 {

// One class's per-perturber amplitudes and multipliers, in the order its
// (norm, h, delta) vectors were given. A perturber whose norm vanishes has
// t = p = q = r = 0 and contributes nothing to `energy`.
struct ScMultipliers {
  std::vector<double> t;  // Eq. 12
  std::vector<double> p;  // Eq. 36
  std::vector<double> q;  // Eq. 37
  std::vector<double> r;  // Eq. 38
  double norm = 0.0;      // sum_k N_k, normToEnergy's first return value
  double energy = 0.0;    // Eq. 15: sum_k T_k N_k, its second
};

// The amplitudes and multipliers for one class, from exactly the three
// vectors normToEnergy takes (same lengths, same order). `norm` and `energy`
// reproduce normToEnergy's pair.
ScMultipliers scMultipliers(const std::vector<double>& norm, const std::vector<double>& h,
                            const std::vector<double>& delta,
                            std::source_location loc = std::source_location::current()) {
  const std::size_t n = norm.size();
  if (h.size() != n || delta.size() != n)
    check(false,
          std::format("scMultipliers: norm/h/delta lengths differ ({}, {}, {})", n, h.size(),
                      delta.size()),
          loc);
  ScMultipliers m;
  m.t.assign(n, 0.0);
  m.p.assign(n, 0.0);
  m.q.assign(n, 0.0);
  m.r.assign(n, 0.0);
  for (std::size_t k = 0; k < n; ++k) {
    m.norm += norm[k];
    // normToEnergy's guard, verbatim: a vanishing norm is an empty perturber
    // space, so it has no amplitude and no energy (and a zero multiplier, so
    // no pseudodensity contribution either).
    if (!(std::fabs(norm[k]) > NUMERICAL_ZERO)) continue;
    const double t = -1.0 / (delta[k] + h[k] / norm[k]);
    m.t[k] = t;
    m.p[k] = 2.0 * t + delta[k] * t * t;
    m.q[k] = t * t;
    m.r[k] = norm[k] * t * t;
    m.energy += t * norm[k];
  }
  return m;
}

// sum_k a_k b_k over two equal-length host vectors -- how a multiplier vector
// meets the per-perturber norm or H it multiplies.
double dotHost(const std::vector<double>& a, const std::vector<double>& b,
               std::source_location loc = std::source_location::current()) {
  if (a.size() != b.size())
    check(false, std::format("dotHost: lengths differ ({}, {})", a.size(), b.size()), loc);
  double s = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) s += a[k] * b[k];
  return s;
}

// The subscripts of the derivative of an einsum term with respect to its
// operand `slot`.
//
// A class's norm or H term is `coef * einsum(subs, ops)` reduced to the
// class's perturber labels. Its derivative with respect to one operand is the
// SAME contraction with that operand dropped and the per-perturber multiplier
// (P for a norm term, Q for an H one) put in its place, written out over the
// dropped operand's own labels: the multiplier is appended as the LAST
// operand and carries the term's original output labels, which become the
// contraction's new output. So
//
//   derivativeSubscripts("ipqr,pqrabc,iabc->i", 0) == "pqrabc,iabc,i->ipqr"
//   derivativeSubscripts("ipqr,pqrabc,iabc->i", 1) == "ipqr,iabc,i->pqrabc"
//   derivativeSubscripts("rpji,raji,pa->rji",   1) == "rpji,pa,rji->raji"
//
// -- the last one being the case that makes this worth doing mechanically:
// the dropped operand reads the core pair in the opposite order from the
// output, and the rewritten string carries that transposition for free.
//
// The dropped operand must not repeat a label (an einsum cannot write a
// diagonal), and `slot` must index an operand: both are our bug at the call
// site, so both abort through check() naming `loc`.
std::string derivativeSubscripts(const std::string& subs, int slot,
                                 std::source_location loc = std::source_location::current()) {
  const std::size_t arrow = subs.find("->");
  if (arrow == std::string::npos)
    check(false, std::format("derivativeSubscripts: no '->' in \"{}\"", subs), loc);
  const std::string rhs = subs.substr(arrow + 2);
  std::vector<std::string> groups;
  std::string cur;
  for (const char c : subs.substr(0, arrow)) {
    if (c == ',') {
      groups.push_back(cur);
      cur.clear();
    } else if (!std::isspace(static_cast<unsigned char>(c))) {
      cur.push_back(c);
    }
  }
  groups.push_back(cur);
  if (slot < 0 || slot >= std::ssize(groups))
    check(false,
          std::format("derivativeSubscripts: slot {} is not an operand of \"{}\"", slot, subs),
          loc);
  const std::string& dropped = groups[static_cast<std::size_t>(slot)];
  for (std::size_t a = 0; a + 1 < dropped.size(); ++a)
    if (dropped.find(dropped[a], a + 1) != std::string::npos)
      check(false,
            std::format("derivativeSubscripts: operand {} of \"{}\" repeats label '{}' -- its "
                        "derivative is a diagonal, which an einsum output cannot spell",
                        slot, subs, dropped[a]),
            loc);
  std::string out;
  for (std::size_t g = 0; g < groups.size(); ++g) {
    if (g == static_cast<std::size_t>(slot)) continue;
    out += groups[g];
    out += ',';
  }
  out += rhs;  // the multiplier, over the term's original output labels
  out += "->";
  out += dropped;
  return out;
}

// --- The pseudocanonical multipliers (Park Eqs. 56-57) ----------------------
//
// The orbital convention `canonicalize(..., cas_natorb=True)` establishes is
// that the generalized Fock is DIAGONAL inside the core space and inside the
// virtual space -- constraint (O2) of docs/gradient-theory.md, Sec. 1.2. The
// energy functional reads f only on that diagonal (`Delta_t` is a sum of
// eps), so f's off-diagonal response there is unconstrained by the energy and
// has to be pinned by the convention instead: one multiplier per pair, and
// because f IS diagonal on both spaces the stationarity condition collapses
// to a division (Sec. 2.3):
//
//   z^c_ij = (1/2) (Y_ij - Y_ji) / (eps_i - eps_j)      i != j in core
//   z^c_rs = (1/2) (Y_rs - Y_sr) / (eps_r - eps_s)      r != s in virtual
//
// -- Park's Eqs. 56-57 verbatim, in the golden's own gauge, and the one place
// in Sec. 2.3 where his form transcribes unchanged (the SELECTION boundaries,
// which read the RHF Fock instead, are small Sylvester equations and are not
// these).
//
// THE TWO BLOCKS OF Y THESE READ ARE EXACT. A rotation inside core, or inside
// virtual, moves no orbital energy (f is diagonal there, and the promoted
// densities have no column outside their own space) and no active integral
// (so the f3 digests do not move either), so Y_ij and Y_rs are the integral
// half alone and carry none of the full source term's held-fixed inputs
// (nevpt2.gradient, "The full source term Y_xy").
//
// EVERY DIVISION IS SINGULAR WHERE ITS DIVISOR VANISHES, and a degenerate
// core or virtual pair is not hypothetical: an abelian subgroup splits a
// degenerate set across irreps, so the two partners have the SAME energy, and
// the golden generator only refuses a degeneracy WITHIN one irrep
// (docs/reference-data.md, "The degeneracy refusal"). The energy is invariant
// under the group, so a rotation mixing two different irreps does not move
// it: for such a pair the numerator vanishes too, the multiplier is a 0/0
// that the constraint simply does not determine, and zero is the right
// representative. So a gap at or below `gapFloor` is accepted only when its
// numerator is at or below `zeroTol`; anything else is a state whose
// multipliers would be noise, and it comes back as a NUMERICAL REFUSAL rather
// than a number (the stance `_check_degeneracy` takes on the energy side).
struct ScPseudocanonical {
  Tensor zCore;              // (n_core, n_core), antisymmetric, zero diagonal
  Tensor zVirt;              // (n_virt, n_virt)
  double maxZ = 0.0;         // the largest |z^c| over both blocks
  double smallestGap = 0.0;  // the smallest |eps_p - eps_q| actually divided by
  int64_t divided = 0;       // pairs with a usable gap
  int64_t degenerate = 0;    // pairs set to zero because both sides vanish
};

// `y` is the full (n_mo, n_mo) source term (only its core-core and
// virtual-virtual blocks are read). The antisymmetrization is written out
// because Park's is -- for a Y that is already antisymmetric, as
// `scFullRotationMatrix` hands it back, `(Y_pq - Y_qp) / 2` is just `Y_pq`.
// `eCore` / `eVirt` are the golden's orbital
// energies, and the spaces are laid out core, then `nact` active, then
// virtual -- the sidecar's own order. A shape that does not fit that is our
// bug at the call site, so it aborts; an undetermined multiplier is the
// state's and comes back as an Error.
Result<ScPseudocanonical> scPseudocanonicalMultipliers(
    const Tensor& y, const Tensor& eCore, const Tensor& eVirt, int64_t nact, double gapFloor,
    double zeroTol, std::source_location loc = std::source_location::current()) {
  const int64_t ncore = eCore.size(), nvirt = eVirt.size();
  const int64_t nmo = ncore + nact + nvirt;
  if (y.rank() != 2 || y.dim(0) != nmo || y.dim(1) != nmo)
    check(false,
          std::format("scPseudocanonicalMultipliers: Y is not a {} x {} matrix for a {} / {} / "
                      "{} partition",
                      nmo, nmo, ncore, nact, nvirt),
          loc);
  if (gapFloor < 0.0 || zeroTol < 0.0)
    check(false, "scPseudocanonicalMultipliers: the tolerances must not be negative", loc);

  ScPseudocanonical out;
  out.zCore = Tensor({ncore, ncore});
  out.zVirt = Tensor({nvirt, nvirt});
  // The two blocks differ only in where they sit in Y and which energies they
  // read, so one loop does both.
  struct Block {
    const char* name;
    int64_t offset;
    int64_t n;
    const Tensor* eps;
    Tensor* z;
  };
  const std::array<Block, 2> blocks{
      Block{"core", 0, ncore, &eCore, &out.zCore},
      Block{"virtual", ncore + nact, nvirt, &eVirt, &out.zVirt}};
  for (const Block& b : blocks) {
    for (int64_t p = 0; p < b.n; ++p)
      for (int64_t q = p + 1; q < b.n; ++q) {
        const double num = 0.5 * (y(b.offset + p, b.offset + q) - y(b.offset + q, b.offset + p));
        const double gap = (*b.eps)(p) - (*b.eps)(q);
        if (std::fabs(gap) <= gapFloor) {
          if (std::fabs(num) > zeroTol)
            return err_numerical(std::format(
                "the pseudocanonical multiplier of {} pair ({}, {}) is not determined: the "
                "orbital energies differ by {:.3e} (at or below {:.0e}) while (Y_pq - Y_qp) / 2 "
                "is {:.3e} (above {:.0e}) -- Park Eqs. 56-57 divide by that gap "
                "(docs/gradient-theory.md, Sec. 2.3)",
                b.name, p, q, gap, gapFloor, num, zeroTol));
          ++out.degenerate;
          continue;  // 0/0: the constraint does not determine it, and zero is a
                     // representative of the gauge it leaves free
        }
        const double z = num / gap;
        (*b.z)(p, q) = z;
        (*b.z)(q, p) = -z;
        out.maxZ = std::max(out.maxZ, std::fabs(z));
        out.smallestGap =
            out.divided == 0 ? std::fabs(gap) : std::min(out.smallestGap, std::fabs(gap));
        ++out.divided;
      }
  }
  return out;
}

}  // namespace nevpt2
