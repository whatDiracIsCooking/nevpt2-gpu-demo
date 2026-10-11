// nevpt2.kappa_rotation -- the ORBITAL-ROTATION PARAMETERISATION, host side:
// an antisymmetric kappa, its exponential, and the transformation an
// orthogonal U induces on the active indices of one stored array. Pure host
// arithmetic -- no device, no GPU -- so the unit tier checks it card-free
// (test/gradient/), exactly as nevpt2.sc_amplitudes is checked.
//
// Reference: Park, J. Chem. Theory Comput. 2019 (arXiv:1907.10180); the
// orbital source term Y_xy = dE/dkappa_xy is his Eq. 26, and
// docs/gradient-theory.md, Sec. 1.3 and 2.1, is this tree's own derivation of
// which blocks of it have which consumer.
//
// THE CONVENTION, ONCE. An orbital rotation is parameterised by an
// antisymmetric kappa through the orthogonal matrix U = exp(kappa), with the
// new orbitals
//
//   |a'> = sum_b |b> U_ba,                                            (U-col)
//
// so a stored array carrying an active index transforms along each of its
// active axes by contracting that axis with U's FIRST index:
//
//   X'[... a ...] = sum_b U_ba X[... b ...]                   (rotateAxes)
//
// and, to first order in t along U = exp(t kappa),
//
//   Xdot[... a ...] = sum_b kappa_ba X[... b ...]        (the kappa action)
//
// summed over the array's active axes. Every direction in this tree is read
// that way: `rotateAxes` is the finite transformation, and the kappa action is
// its derivative at t = 0. The action itself is taken ON THE DEVICE, where
// the slabs live (nevpt2.gradient's :rotation partition); the pair is pinned
// by the finite difference apps/sc_pseudodensity/ runs -- the energy along
// `rotateAxes(..., expKappa(t kappa))` against the tangent built from the
// device action (docs/testing.md, "The active-rotation source term").
//
// WHY A HOST HELPER AT ALL, when the pseudodensity assembly is device-side:
// the finite-difference side of that check wants the FINITE rotation of every
// stored integral block, and those arrive from the golden file as host
// Tensors -- rotating them here and uploading is one pass over arrays that
// are at most n_ext * n_act^3 doubles. It also makes the two properties the
// whole construction rests on checkable with no card: U is orthogonal, and
// a rotation by kappa = 0 is the IDENTITY on the data, bit for bit.
export module nevpt2.kappa_rotation;

import std;
// Re-exported: every entry point here takes and returns a host Tensor.
export import nevpt2.tensor;
import nevpt2.error_handling;  // check(): the abort tier for a malformed call

export namespace nevpt2 {

// A seeded antisymmetric n x n direction, scaled to unit Frobenius norm --
// the reproducible `kappa` the checks pair the source term against. The
// strict upper triangle is filled from `seed`'s stream and mirrored with a
// minus sign, so the diagonal is exactly zero and kappa + kappa^T is exactly
// zero (not zero to rounding).
Tensor antisymmetricKappa(int64_t n, std::uint64_t seed,
                          std::source_location loc = std::source_location::current()) {
  if (n < 0) check(false, std::format("antisymmetricKappa: negative extent {}", n), loc);
  Tensor k({n, n});
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  double norm2 = 0.0;
  for (int64_t p = 0; p < n; ++p)
    for (int64_t q = p + 1; q < n; ++q) {
      const double v = dist(gen);
      k(p, q) = v;
      k(q, p) = -v;
      norm2 += 2.0 * v * v;
    }
  if (norm2 > 0.0) {
    const double inv = 1.0 / std::sqrt(norm2);
    for (int64_t i = 0; i < k.size(); ++i) k.flatRef(i) *= inv;
  }
  return k;
}

// max |kappa + kappa^T|: zero for a direction this module will accept.
double antisymmetryResidual(const Tensor& kappa,
                            std::source_location loc = std::source_location::current()) {
  if (kappa.rank() != 2 || kappa.dim(0) != kappa.dim(1))
    check(false, "antisymmetryResidual: kappa must be a square matrix", loc);
  double m = 0.0;
  for (int64_t p = 0; p < kappa.dim(0); ++p)
    for (int64_t q = 0; q < kappa.dim(0); ++q)
      m = std::max(m, std::fabs(kappa(p, q) + kappa(q, p)));
  return m;
}

// max |U^T U - 1|: how far `u` is from orthogonal.
double orthogonalityResidual(const Tensor& u,
                             std::source_location loc = std::source_location::current()) {
  if (u.rank() != 2 || u.dim(0) != u.dim(1))
    check(false, "orthogonalityResidual: u must be a square matrix", loc);
  const int64_t n = u.dim(0);
  double m = 0.0;
  for (int64_t p = 0; p < n; ++p)
    for (int64_t q = 0; q < n; ++q) {
      double s = 0.0;
      for (int64_t i = 0; i < n; ++i) s += u(i, p) * u(i, q);
      m = std::max(m, std::fabs(s - (p == q ? 1.0 : 0.0)));
    }
  return m;
}

// U = exp(kappa), for an antisymmetric kappa: orthogonal, and EXACTLY the
// identity when kappa is exactly zero.
//
// Scaling and squaring around a Taylor series: kappa is halved until its
// largest row sum is below 1/2, where the series converges fast and every
// term is formed from the previous one, then the result is squared back. For
// kappa = 0 the scaling count is zero and every Taylor term past the first is
// an exactly-zero matrix, so the sum is the identity to the bit -- which is
// what makes "a rotation by zero changes no block" an exact statement rather
// than a tolerance.
//
// A kappa that is not square, or not antisymmetric to 1e-14, is our bug at
// the call site (every direction here is constructed antisymmetric):
// check() aborts, naming `loc`.
Tensor expKappa(const Tensor& kappa, std::source_location loc = std::source_location::current()) {
  if (kappa.rank() != 2 || kappa.dim(0) != kappa.dim(1))
    check(false, "expKappa: kappa must be a square matrix", loc);
  if (antisymmetryResidual(kappa) > 1e-14)
    check(false,
          std::format("expKappa: kappa is not antisymmetric (max |k + k^T| = {:.3e})",
                      antisymmetryResidual(kappa)),
          loc);
  const int64_t n = kappa.dim(0);

  // The scaling: s halvings bring the largest row sum under 1/2.
  double rowMax = 0.0;
  for (int64_t p = 0; p < n; ++p) {
    double row = 0.0;
    for (int64_t q = 0; q < n; ++q) row += std::fabs(kappa(p, q));
    rowMax = std::max(rowMax, row);
  }
  int squarings = 0;
  double scaled = rowMax;
  while (scaled > 0.5) {
    scaled *= 0.5;
    ++squarings;
  }
  const double shrink = std::ldexp(1.0, -squarings);  // exact: a power of two
  Tensor a({n, n});
  for (int64_t i = 0; i < a.size(); ++i) a.flatRef(i) = kappa.flat(i) * shrink;

  // exp(a) = sum_k a^k / k!, term by term: `term` is a^k / k!, and the series
  // is cut once the term can no longer move the sum (20 terms is already past
  // double precision for ||a|| <= 1/2; the bound is a guard, not the stop).
  Tensor out = Tensor::eye(n);
  Tensor term = Tensor::eye(n);
  for (int k = 1; k <= 24; ++k) {
    Tensor next({n, n});
    for (int64_t p = 0; p < n; ++p)
      for (int64_t q = 0; q < n; ++q) {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) s += term(p, i) * a(i, q);
        next(p, q) = s / static_cast<double>(k);
      }
    double mag = 0.0;
    for (int64_t i = 0; i < next.size(); ++i) mag = std::max(mag, std::fabs(next.flat(i)));
    if (mag == 0.0) break;  // kappa == 0: nothing past the identity, exactly
    for (int64_t i = 0; i < out.size(); ++i) out.flatRef(i) += next.flat(i);
    term = std::move(next);
    if (mag < 1e-20) break;
  }

  // The squarings, undoing the scaling.
  for (int s = 0; s < squarings; ++s) {
    Tensor sq({n, n});
    for (int64_t p = 0; p < n; ++p)
      for (int64_t q = 0; q < n; ++q) {
        double v = 0.0;
        for (int64_t i = 0; i < n; ++i) v += out(p, i) * out(i, q);
        sq(p, q) = v;
      }
    out = std::move(sq);
  }
  return out;
}

// `x` with each axis in `axes` transformed by `u`:
//
//   out[... a ...] = sum_b u(b, a) x[... b ...]        over every axis listed
//
// (the (U-col) convention above). `axes` may be empty, in which case this is
// a copy -- an array with no active index does not move under an active
// rotation.
//
// EXACTLY A COPY WHEN u IS THE IDENTITY, bit for bit: a zero coefficient
// contributes nothing (it is skipped, not multiplied and added), and a single
// surviving coefficient is the output, so no addition touches the value at
// all. That is what makes `rotateAxes(x, axes, expKappa(zero))` the identity
// on the data.
//
// A duplicate, out-of-range or wrongly-sized axis, or a `u` whose extent does
// not match the axes it transforms, is our bug at the call site: the axis
// lists here are written per block from the layouts nevpt2.energy documents.
Tensor rotateAxes(const Tensor& x, const std::vector<int>& axes, const Tensor& u,
                  std::source_location loc = std::source_location::current()) {
  if (u.rank() != 2 || u.dim(0) != u.dim(1))
    check(false, "rotateAxes: u must be a square matrix", loc);
  const int64_t n = u.dim(0);
  for (std::size_t k = 0; k < axes.size(); ++k) {
    if (axes[k] < 0 || axes[k] >= x.rank())
      check(false, std::format("rotateAxes: axis {} is not an axis of a rank-{} tensor",
                               axes[k], x.rank()),
            loc);
    if (x.dim(axes[k]) != n)
      check(false,
            std::format("rotateAxes: axis {} has extent {}, u is {} x {}", axes[k],
                        x.dim(axes[k]), n, n),
            loc);
    for (std::size_t j = 0; j < k; ++j)
      if (axes[j] == axes[k])
        check(false, std::format("rotateAxes: axis {} is listed twice", axes[k]), loc);
  }

  Tensor out = x;
  // One axis at a time, each pass reading the previous one's output: the
  // transformation on several axes is the composition of the one-axis ones.
  for (const int axis : axes) {
    const int64_t stride = out.stride(axis);
    const int64_t outer = out.size() / (n * stride);  // elements before the axis
    Tensor next(out.dims());
    for (int64_t o = 0; o < outer; ++o)
      for (int64_t inner = 0; inner < stride; ++inner) {
        const int64_t base = o * n * stride + inner;
        for (int64_t a = 0; a < n; ++a) {
          double acc = 0.0;
          bool first = true;
          for (int64_t b = 0; b < n; ++b) {
            const double c = u(b, a);
            if (c == 0.0) continue;
            const double v = c * out.flat(base + b * stride);
            if (first) {
              acc = v;
              first = false;
            } else {
              acc += v;
            }
          }
          next.flatRef(base + a * stride) = acc;
        }
      }
    out = std::move(next);
  }
  return out;
}

}  // namespace nevpt2
