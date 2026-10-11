// nevpt2.gradient:rotation -- the ACTIVE-ACTIVE ORBITAL ROTATION on the
// device: one antisymmetric direction's action on the arrays the class
// energies read, the tangent of each Dyall-Hamiltonian block along it, and
// the pairing of both against the pseudodensities nevpt2.gradient:terms
// assembles.
// An internal partition: nothing here is exported, so it is reachable from
// the units that `import :rotation;` and from no importer of nevpt2.gradient.
// (Used by gradient_terms.cppm, gradient_sc_classes.cppm and gradient.cpp.)
//
// WHAT THE DIRECTION DOES TO WHAT. Rotating the active orbitals along an
// antisymmetric kappa (nevpt2.kappa_rotation's convention, which this is the
// device twin of) moves every array that carries an active index:
//
//   Xdot[... a ...] = sum_b kappa_ba X[... b ...],  summed over X's active
//                                                   axes   (kappaActionDevice)
//
// -- the active-space integrals h1e and h2e, and every external integral
// block's active slots (`activeAxesOf`). What it does NOT move is listed in
// gradient.cppm, "The active-rotation source term", and is the scope of this
// whole piece of work: the RDM-side inputs (dm1/dm2/dm3 and the f3ac/f3ca
// digests) and the orbital energies are held FIXED, so this is the integral
// half of the chain rule, dE/dkappa at fixed RDMs and fixed eps.
//
// SO THE DYALL BLOCKS MOVE TOO, AND NO NEW CHAIN RULE IS NEEDED FOR THEM.
// Every make_* of nevpt2.energy is LINEAR in the integrals it reads (each of
// its terms carries exactly one integral factor), so the tangent of a block
// along the rotation is that same builder called with the DOTTED integrals,
// and with the integral-free operands zeroed:
//
//   Gdot = make_G(h1e_dot, h2e_dot, <the same RDMs>, f3ac = 0, f3ca = 0)
//
// -- f3ac/f3ca enter a16/a22 as plain addends (no integral factor), so they
// are constant under this functional and the zeros isolate the h1e/h2e part.
// A purely RDM-built metric (dm1, dm2, dm3, the hole-side dm*_h/hdm*, rm2)
// has no integral in it at all, so its tangent is exactly zero and it is
// absent from a class's `ScGDot` list.
//
// THE TWO DIRECTIONS, AND WHY BOTH ARE HERE. With the pseudodensities
// D = dE/dX (Eq. 41, per external block) and M_G = dE/dG (Eqs. 42-43, per
// active block) in hand, the ADJOINT pairing is one dot product per object,
//
//   <kappa, Y_act> = sum_blocks <D, Xdot> + sum_G <M_G, Gdot>,
//
// while the forward TANGENT reruns the same class terms under the product
// rule (`evalTermsTangent` in :terms) and finishes with the multipliers the
// walk already has, P.Ndot + Q.Hdot (Park Eqs. 36-37). The two are the same
// derivative taken in opposite modes -- the first through the DERIVATIVE of
// each term with respect to one operand, the second through the term itself
// with one operand replaced -- so holding them equal is what checks the
// pseudodensity assembly. Neither is a finite difference; the finite
// difference is the energy's own, in apps/sc_pseudodensity/, and is what
// pins the two conventions above to nevpt2.kappa_rotation's exponential map.
module nevpt2.gradient:rotation;

import std;
import nevpt2.gradient;

namespace nevpt2 {

// The axes of each external two-electron block that carry an ACTIVE index,
// in the block's physicist layout (nevpt2.energy's ExtBlock table, which is
// also the order each class's einsum subscripts spell):
//
//   Sr   (v,a,a,a) 1,2,3   Si   (a,a,c,a) 0,1,3   Sijrs (c,v,c,v) none
//   Sijr (v,a,c,c) 1       Srsi (v,v,c,a) 3       SrsiT (v,v,c,a) 3
//   Srs  (v,v,a,a) 2,3     Sij  (a,a,c,c) 0,1     Sir1 (v,a,c,a) 1,3
//   Sir2 (v,a,a,c) 1,2
//
// A slab restricts an EXTERNAL axis (batchAxis), so every active axis is
// whole within a slab and this table is the slab's as well as the block's.
std::vector<int> activeAxesOf(ExtBlock b) {
  switch (b) {
    case ExtBlock::Sr: return {1, 2, 3};
    case ExtBlock::Si: return {0, 1, 3};
    case ExtBlock::Sijrs: return {};
    case ExtBlock::Sijr: return {1};
    case ExtBlock::Srsi: return {3};
    case ExtBlock::SrsiT: return {3};
    case ExtBlock::Srs: return {2, 3};
    case ExtBlock::Sij: return {0, 1};
    case ExtBlock::Sir1: return {1, 3};
    case ExtBlock::Sir2: return {1, 2};
  }
  return {};
}

// `kappa`'s action on `x` over `axes`, as a fresh tensor from res's pool:
// sum over the listed axes of sum_b kappa_ba x[... b ...]. One einsum per
// axis, the first allocating and the rest accumulating, all on res.stream().
//
// `axes` must not be empty (a caller with no active axis skips the call
// entirely -- there is no tangent to form), and `x`'s rank is at most 6, the
// width of the label alphabet below. Both are our bug at the call site.
DeviceTensor kappaActionDevice(const DeviceTensor& x, const std::vector<int>& axes,
                               const DeviceTensor& kappa, const DeviceResources& res,
                               std::source_location loc = std::source_location::current()) {
  static constexpr std::string_view kLabels = "abcdef";
  if (axes.empty() || x.rank() < 1 || x.rank() > std::ssize(kLabels))
    check(false, "kappaActionDevice: needs at least one axis and a rank in [1, 6]", loc);
  const std::string base(kLabels.substr(0, static_cast<std::size_t>(x.rank())));
  DeviceTensor out;
  for (std::size_t k = 0; k < axes.size(); ++k) {
    if (axes[k] < 0 || axes[k] >= x.rank())
      check(false, std::format("kappaActionDevice: axis {} is not an axis of a rank-{} tensor",
                               axes[k], x.rank()),
            loc);
    const std::size_t axis = static_cast<std::size_t>(axes[k]);
    std::string in = base;
    in[axis] = 'z';  // the contracted label, outside the alphabet above
    const std::string subs = in + ",z" + base[axis] + "->" + base;
    if (k == 0) {
      out = deviceEinsumNew(subs, {&x, &kappa}, 1.0, res);
    } else {
      deviceEinsumAccum(subs, {&x, &kappa}, out, 1.0, res.stream());
    }
  }
  return out;
}

// One direction, device-resident: the antisymmetric kappa itself and its
// action on the two active-space integral blocks (which every Dyall block is
// built from, and which no slab walk re-derives). `tangent` asks for the
// forward tangent along it as well as the adjoint pairing -- that is what
// costs a second set of Dyall blocks, so it is per direction.
struct ScRotationDir {
  DeviceTensor kappa;
  DeviceTensor h1eDot;
  DeviceTensor h2eDot;
  bool tangent = false;
};

struct ScRotation {
  std::vector<ScRotationDir> dirs;
  std::size_t size() const { return dirs.size(); }
  bool empty() const { return dirs.empty(); }
};

// The requests uploaded and differentiated once, before any class runs. A
// kappa of the wrong shape, or one that is not antisymmetric, is the
// CALLER's bad input, not our bug -- it reaches here as an Error.
Result<ScRotation> makeRotation(const std::vector<ScRotationRequest>& reqs,
                                const DeviceTensor& h1e, const DeviceTensor& h2e,
                                const DeviceResources& res) {
  const int64_t norb = h1e.dims[0];
  ScRotation rot;
  rot.dirs.reserve(reqs.size());
  for (std::size_t d = 0; d < reqs.size(); ++d) {
    const Tensor& k = reqs[d].kappa;
    if (k.rank() != 2 || k.dim(0) != norb || k.dim(1) != norb)
      return err_config(
          std::format("direction {}: kappa is not a {} x {} matrix", d, norb, norb));
    double anti = 0.0;
    for (int64_t p = 0; p < norb; ++p)
      for (int64_t q = 0; q < norb; ++q) anti = std::max(anti, std::fabs(k(p, q) + k(q, p)));
    if (anti > 1e-14)
      return err_config(std::format(
          "direction {}: kappa is not antisymmetric (max |k + k^T| = {:.3e})", d, anti));
    ScRotationDir dir;
    dir.kappa = uploadTensor(k, res);
    dir.h1eDot = kappaActionDevice(h1e, {0, 1}, dir.kappa, res);
    dir.h2eDot = kappaActionDevice(h2e, {0, 1, 2, 3}, dir.kappa, res);
    dir.tangent = reqs[d].tangent;
    rot.dirs.push_back(std::move(dir));
  }
  return rot;
}

// One active-space block and its tangent along one direction: the pair
// <M_G, Gdot> is taken over, and `g` is the key the walk's accumulator for
// that block is found by (:terms' activeAccumOf).
struct ScGDot {
  const DeviceTensor* g = nullptr;
  DeviceTensor dot;
};

// The tangent of `g` in `gDots`, or nullptr -- how the product rule finds the
// dot of an operand that is an active block. A purely RDM-built metric is
// absent and so has no tangent, which is exactly right: it does not move.
// (The two routines that BUILD these lists per direction, and pair them
// against the walk's M_G accumulators, are in :terms, where ScActiveAccum
// and dotTensors live.)
const DeviceTensor* gDotOf(const std::vector<ScGDot>& gDots, const DeviceTensor* g) {
  for (const ScGDot& d : gDots)
    if (d.g == g) return &d.dot;
  return nullptr;
}

}  // namespace nevpt2
