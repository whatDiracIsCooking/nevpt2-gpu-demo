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
//
// AND THE SAME MACHINERY CARRIES THE FULL kappa SPACE. A direction built with
// a gradient sidecar (`makeGeneralRotation`) adds, to each of the quantities
// above, the part of the chain rule whose promoted index leaves the array's
// own space -- one einsum per slot out of the sidecar's one-general-index
// blocks (:general) -- plus the orbital energies' response, which the active
// block had to hold fixed. The two halves partition every slot's sum with no
// overlap, so an active-restricted kappa leaves the active answer bit for
// bit what it was (gradient.cppm, "The full source term Y_xy").
module nevpt2.gradient:rotation;

import std;
import nevpt2.gradient;
// The promotion table, the general-range action it generates, and the
// orbital-energy response: the half of the rotation that leaves the active
// space. `activeAxesOf` lives there too, keyed by the same table.
import :general;

namespace nevpt2 {

// A slab restricts an EXTERNAL axis (batchAxis), so every ACTIVE axis of a
// block is whole within a slab and :general's table -- read for its active
// slots alone, `activeAxesOf(RotBlock)` -- is the slab's as well as the
// block's. That one table is what keeps the intra-active and general halves
// of the action from disagreeing about which axis is which.

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
//
// With a gradient sidecar the direction carries three more operands: kappa's
// COLUMN BLOCKS, one per orbital space, which is what a general promotion
// contracts over (`columns`). The active one has its active ROWS zeroed,
// because the x-in-active part of an active slot's sum is exactly what
// `kappa` above already does -- so the two halves partition the sum. It also
// carries the orbital energies' response, on the host, because that is where
// each class assembles the `delta` vector it is the tangent of.
struct ScRotationDir {
  DeviceTensor kappa;  // (n_act, n_act): the active-active block
  DeviceTensor h1eDot;
  DeviceTensor h2eDot;
  bool tangent = false;

  bool general = false;
  DeviceTensor kCore;  // kappa[:, core]                       (n_mo, n_core)
  DeviceTensor kAct;   // kappa[:, act], its active ROWS zero  (n_mo, n_act)
  DeviceTensor kVirt;  // kappa[:, virt]                       (n_mo, n_virt)
  Tensor epsCoreDot;   // deps_i  (n_core)
  Tensor epsVirtDot;   // deps_r  (n_virt)
  // kappa[:, act] WITH its active rows (generalEpsResponse's column block, kept
  // for the orbital energies' mixed response, generalEpsResponseTangent, #184).
  DeviceTensor kActFull;

  // THE DIGESTS' SHARE, MERGED PER TUPLE (#184). Null by default, and then
  // every class's Dyall-block tangent along this direction is its make_* at
  // the dotted integrals with the f3 addends ZEROED -- the frozen-digest
  // functional every first-order entry point differentiates, the digests'
  // own term kept apart as #88's separate walk. Set (attachDigestTangents),
  // it is the digests' tangent along this direction -- the RDM build at
  // h2eDot, linear in h2e -- and the f3-reading blocks (Sr's a16, Si's a22)
  // take it in place of the zeros, so each tuple's Hdot_t carries the
  // digests' share BEFORE anything nonlinear meets it: what the mixed block's
  // 3x3 sandwich needs (docs/hessian-theory.md, Sec. 5.5's C1) and K3 will
  // (#158). Borrowed: the owner outlives every walk the direction enters.
  const DeviceTensor* f3acDot = nullptr;
  const DeviceTensor* f3caDot = nullptr;

  // The column block a slot over `s` contracts with. Only ever called on a
  // general direction (the active-only path never reaches :general).
  const DeviceTensor* columns(RotSpace s) const {
    switch (s) {
      case RotSpace::Core: return &kCore;
      case RotSpace::Act: return &kAct;
      case RotSpace::Virt: return &kVirt;
    }
    return nullptr;
  }
};

struct ScRotation {
  std::vector<ScRotationDir> dirs;
  // The sidecar, or null for the active-active block alone: what says whether
  // a slab walk does the general work at all.
  const GeneralIntegralsDevice* gen = nullptr;
  std::size_t size() const { return dirs.size(); }
  bool empty() const { return dirs.empty(); }
  bool general() const { return gen != nullptr; }
};

// `kappa + kappa^T`'s largest element, as a value-tier refusal would report
// it: a direction is the caller's input, so a malformed one is an Error, not
// our bug.
double kappaAntisymmetry(const Tensor& k) {
  double anti = 0.0;
  for (int64_t p = 0; p < k.dim(0); ++p)
    for (int64_t q = 0; q < k.dim(1); ++q) anti = std::max(anti, std::fabs(k(p, q) + k(q, p)));
  return anti;
}

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
    const double anti = kappaAntisymmetry(k);
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

// `k`'s rows [0, n_mo) and columns [lo, hi), as a host tensor -- and with the
// active rows zeroed when `dropActiveRows`, which is what makes the general
// half of an active slot's sum the complement of the intra-active one.
Tensor kappaColumnBlock(const Tensor& k, int64_t lo, int64_t hi, int64_t actLo, int64_t actHi,
                        bool dropActiveRows) {
  const int64_t nmo = k.dim(0);
  Tensor out({nmo, hi - lo});
  for (int64_t x = 0; x < nmo; ++x) {
    if (dropActiveRows && x >= actLo && x < actHi) continue;
    for (int64_t p = lo; p < hi; ++p) out(x, p - lo) = k(x, p);
  }
  return out;
}

// The same, over the full kappa space: every direction's column blocks, the
// active integrals' complete tangent (intra-active plus general), and the
// orbital energies' response. `gen` is held by the returned ScRotation, so it
// must outlive it -- as the caller's sidecar does.
//
// A kappa that is not an antisymmetric (n_mo, n_mo), or a sidecar whose
// active space is not the integrals', is the CALLER's input: an Error.
Result<ScRotation> makeGeneralRotation(const std::vector<ScRotationRequest>& reqs,
                                       const DeviceTensor& h1e, const DeviceTensor& h2e,
                                       const GeneralIntegralsDevice& gen,
                                       const DeviceTensor& dm1, const DeviceResources& res) {
  const int64_t nmo = gen.nmo();
  const int64_t actLo = gen.ncore, actHi = gen.ncore + gen.nact;
  if (h1e.dims[0] != gen.nact)
    return err_config(std::format(
        "the gradient sidecar's active space is {} orbitals, the integrals' is {}", gen.nact,
        h1e.dims[0]));
  ScRotation rot;
  rot.gen = &gen;
  rot.dirs.reserve(reqs.size());
  for (std::size_t d = 0; d < reqs.size(); ++d) {
    const Tensor& k = reqs[d].kappa;
    if (k.rank() != 2 || k.dim(0) != nmo || k.dim(1) != nmo)
      return err_config(
          std::format("direction {}: kappa is not a {} x {} matrix", d, nmo, nmo));
    const double anti = kappaAntisymmetry(k);
    if (anti > 1e-14)
      return err_config(std::format(
          "direction {}: kappa is not antisymmetric (max |k + k^T| = {:.3e})", d, anti));

    ScRotationDir dir;
    dir.general = true;
    dir.tangent = reqs[d].tangent;
    // The active-active block alone, exactly as the active-only path uploads
    // it: the intra-active half of every active slot's sum reads this.
    Tensor kAA({gen.nact, gen.nact});
    for (int64_t p = 0; p < gen.nact; ++p)
      for (int64_t q = 0; q < gen.nact; ++q) kAA(p, q) = k(actLo + p, actLo + q);
    dir.kappa = uploadTensor(kAA, res);
    dir.kCore = uploadTensor(kappaColumnBlock(k, 0, gen.ncore, actLo, actHi, false), res);
    dir.kAct = uploadTensor(kappaColumnBlock(k, actLo, actHi, actLo, actHi, true), res);
    dir.kVirt = uploadTensor(kappaColumnBlock(k, actHi, nmo, actLo, actHi, false), res);
    const auto columns = [&dir](RotSpace s) { return dir.columns(s); };

    // h1e and h2e: the intra-active action on the stored block, plus the
    // general part (h1e is h^eff[a,a], so its part carries the core
    // dressing's response too).
    dir.h1eDot = kappaActionDevice(h1e, {0, 1}, dir.kappa, res);
    dir.h2eDot = kappaActionDevice(h2e, {0, 1, 2, 3}, dir.kappa, res);
    addTensorInto(
        generalBlockAction(RotBlock::H1e, gen, columns, h1e.dims, res), dir.h1eDot,
        res.stream());
    addTensorInto(
        generalBlockAction(RotBlock::H2e, gen, columns, h2e.dims, res), dir.h2eDot,
        res.stream());

    // The orbital energies, read back to the host: the active column block
    // used here is the FULL one (no intra-active half to complement), so it
    // is rebuilt rather than reusing dir.kAct.
    dir.kActFull = uploadTensor(kappaColumnBlock(k, actLo, actHi, actLo, actHi, false), res);
    const ScEpsDot eps =
        generalEpsResponse(gen, dm1, dir.kCore, dir.kActFull, dir.kVirt, res);
    dir.epsCoreDot = eps.core;
    dir.epsVirtDot = eps.virt;
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

// THE PER-TUPLE MERGE OF THE DIGESTS' SHARE (#184, for #158's K3 as well):
// the digests' tangent along every TANGENT direction of `rot` -- the RDM build
// run at that direction's own h2eDot (linear in h2e at a fixed CI vector:
// #88's construction), so it is exactly the h2e_dot the Dyall blocks read,
// the sidecar's general part included -- attached to the direction
// (ScRotationDir::f3acDot / f3caDot), after which every class's Dyall-block
// tangent along it, and so every tuple's Hdot_t, carries the digests' share.
// The returned builds OWN the attached tensors: the caller holds them for as
// long as `rot` is walked. An adjoint-only direction gets none (its tangent
// blocks are never built up front). Exact digests only, as every tangent
// build: `opt` naming --cublas / --ozaki / --fused-digest is our bug (cli
// refuses the flags that reach here).
std::vector<RdmBuildResult> attachDigestTangents(ScRotation& rot, const RdmBuildOptions& opt,
                                                 const Tensor& ci, int64_t nelecA,
                                                 int64_t nelecB, const DeviceResources& res) {
  check(!opt.cublas && !opt.ozaki && !opt.fusedDigest,
        "attachDigestTangents runs the emitted or the native-fp64 BLAS digest");
  std::vector<RdmBuildResult> owned;
  owned.reserve(rot.size());
  for (ScRotationDir& d : rot.dirs) {
    if (!d.tangent) {
      owned.emplace_back();
      continue;
    }
    const int64_t norb = d.h2eDot.dims[0];
    // downloadTensor synchronizes the stream, so the rotation's einsums that
    // wrote h2eDot are complete; the build takes its integrals on the host.
    const Tensor h2eDot = downloadTensor(d.h2eDot, res.stream());
    owned.push_back(buildRdmsDevice(opt, ci, norb, nelecA, nelecB, h2eDot, res));
    d.f3acDot = &owned.back().f3ac;
    d.f3caDot = &owned.back().f3ca;
  }
  return owned;
}

}  // namespace nevpt2
