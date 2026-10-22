// nevpt2.gradient:rdm_adjoint -- the ONE CHAIN RULE that nevpt2.gradient's
// Eqs. 42-43 pseudodensities are short of: from the metric and
// Dyall-Hamiltonian BLOCKS the class einsums contract (nevpt2.energy's
// make_a16, make_a22, make_hdm2, ...) down to the five RDMs those blocks are
// built from. Its output is Park Eq. 48's coefficient tensors -- dE/d(dm1),
// dE/d(dm2), dE/d(dm3), dE/d(f3ac), dE/d(f3ca) -- which
// nevpt2.gradient's scCiSourceDevice then feeds to the sigma-vector kernels.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :rdm_adjoint;` and from no importer of nevpt2.gradient.
// (Used by gradient_sc_classes.cppm and gradient.cpp.)
//
// WHY THERE IS A SWEEP AT ALL. A class energy reads the RDMs only through its
// blocks, and every block is LINEAR in them (that is the same fact
// nevpt2.energy:sc_tangent's forward tangent rests on). So, with M_G the
// accumulator nevpt2.gradient:terms' walk already fills for block G
// (M_G = dE/dG, Eqs. 42-43),
//
//   dE/d(rho) = sum over blocks G of (dG/d(rho))^T M_G,
//
// and dG/d(rho)^T is just G's own einsum chain with the output and the RDM
// operand swapping roles -- which is exactly what
// nevpt2.sc_amplitudes' derivativeSubscripts writes. So NO DERIVATIVE IS
// HAND-TRANSCRIBED here either: each block below is its make_* chain written
// out term by term (same subscript strings, same coefficients, same
// operands, so the two read side by side, as
// nevpt2.gradient:sc_classes does for the class terms), and `adjointTerm`
// rewrites each one.
//
// THE SWEEP IS REVERSE-TOPOLOGICAL, and the three places that matters say so:
// Sij's a9 reads hdm3 which reads hdm2 (a9, hdm3, hdm2), Srs's a7 reads rm3
// which reads rm2 (a7, rm3, rm2), and make_a16 / make_a22 have internal fdm2
// intermediates of their own, handled inside their own functions. A block
// that is ALSO a class metric (hdm2 for Sij, rm2 for Srs, dm1 for Srsi) has
// its accumulator seeded from the walk's before the sweep adds to it.
//
// WHAT IS NOT DIFFERENTIATED. Integrals (h1e, h2e, the dressed one-electron
// matrix make_a17/make_a19 build) and the Kronecker delta: they are constant
// as the CI vector moves, and the RDM-free terms that read nothing else --
// make_hdm1's and make_hdm2's `2 delta` / delta-delta pieces, make_a25's
// `2 h1e^T`, make_dm*_h's delta terms' non-RDM operand -- drop out of the
// derivative exactly as they do from the tangent (energy_sc_tangent.cppm's
// make_hdm1_dot / make_hdm2_dot / make_a25_dot).
//
// Every einsum here is issued on res.stream() and every tensor it allocates
// is drawn from res's pool, freed when it goes out of scope -- the five
// adjoints when the caller's sweep is done, a block's own intermediate
// accumulator when its function returns.
module nevpt2.gradient:rdm_adjoint;

import std;
import nevpt2.gradient;
import :general;  // addTensorInto, the one way this tree adds two device tensors
import :terms;    // ScActiveAccum / activeAccumOf

namespace nevpt2 {

// The five coefficient tensors of Park Eq. 48, one per RDM the SC energy
// reads, each shaped like the RDM it is conjugate to and accumulated over all
// eight classes. `f3ac` / `f3ca` are conjugate to the integral-contracted
// 4-RDM digests, not to the 4-RDM: the energy never reads anything else of
// it, so neither does its derivative.
struct ScRdmAdjoint {
  DeviceTensor dm1, dm2, dm3, f3ac, f3ca;

  static ScRdmAdjoint zeros(const int64_t n, const DeviceResources& dr) {
    return ScRdmAdjoint{DeviceTensor::zeros({n, n}, dr),
                        DeviceTensor::zeros({n, n, n, n}, dr),
                        DeviceTensor::zeros({n, n, n, n, n, n}, dr),
                        DeviceTensor::zeros({n, n, n, n, n, n}, dr),
                        DeviceTensor::zeros({n, n, n, n, n, n}, dr)};
  }
};

// The sweep's TANGENT along a CI direction u (#156, docs/hessian-theory.md
// Sec. 4.2): every block sweep below is LINEAR in its accumulator, with
// coefficients (integrals, the Kronecker delta) that do not move with the CI
// vector, so the tangent of dE/d(rho) is the SAME sweep run on the M_G
// tangents. A class handed one of these builds its active blocks' tangents
// from `dot` (the RDMs' tangents along u: nevpt2.energy's make_* at the
// tangent RDMs, and its three tangent-only builders for the blocks with an
// RDM-free term), accumulates Mdot_G beside M_G over its walk
// (nevpt2.gradient:terms' ScCiWalk), and sweeps Mdot_G into `adjDot` exactly
// as it sweeps M_G into the plain ScRdmAdjoint -- one function per class,
// called on each. `dot` is borrowed for the call; `adjDot` is the caller's,
// accumulated over all eight classes.
//
// `epsCoreU` / `epsVirtU` (#214): the orbital energies' tangent along u,
// e'[u] = <dm1'[u], W>, for the FULL mode (their response to the CI vector
// carried at second order); null holds them fixed. Borrowed for the call.
struct ScCiDirection {
  const RdmTensors& dot;
  ScRdmAdjoint& adjDot;
  const Tensor* epsCoreU = nullptr;
  const Tensor* epsVirtU = nullptr;
};

// The walk a class carrying `d` starts from: no tangent block yet (each class
// builds its own), and the direction's orbital-energy tangent, if any.
ScCiWalk ciWalkOf(const ScCiDirection& d) {
  ScCiWalk w;
  w.epsCoreU = d.epsCoreU;
  w.epsVirtU = d.epsVirtU;
  return w;
}

// THE MIXED DIRECTION (#184, docs/hessian-theory.md Sec. 5.5): an orbital
// direction w -- rot.dirs[0], its tangent blocks built and the digests' share
// merged -- and, forward, a CI direction u. What every class reads to build
// its own ScMixedWalk (:terms), and where its reverse sweeps go.
//
// THE REVERSE SWEEP ALONG w. Every block sweep is AFFINE-linear: an
// integral-free part L(M) (a raw RDM's accumulator added in, the hole RDMs'
// delta terms, a16's / a22's f3 transposes) plus a part B(M; h1e, h2e)
// bilinear in the accumulator and the active integrals. Along w both M and
// the integrals move, so
//
//   d/dw sweep(M; h) = sweep(Mdot; h) + B(M; h'[w])
//                    = sweep(Mdot; h) + sweep(M; h'[w]) - sweep(M; 0),
//
// the last term being L(M) exactly (B vanishes at zero integrals). The first
// two go into `adjDot`, the third into `adjFree`, and the caller subtracts --
// so no adjoint function needs an integrals-only flag, and the sweep each
// class writes once runs four times with nothing transcribed.
struct ScMixedDirection {
  bool forward = false;
  bool reverse = false;
  bool epsKappa = false;  // the orbital energies respond to kappa
  bool epsCi = false;     // ... and to the CI vector
  // forward: u's tangent RDMs (the tangent build at (c, u) and h2e), the
  // digests' mixed tangent f3^{wu} (the same build at h2e'[w]), and the
  // orbital energies' tangents e'[u] and eps^{wu} (zero vectors when off).
  const RdmTensors* dotU = nullptr;
  const DeviceTensor* f3acWU = nullptr;
  const DeviceTensor* f3caWU = nullptr;
  Tensor epsCoreU, epsVirtU, epsCoreWU, epsVirtWU;
  // zero (n, n) and (n, n, n, n): the integrals the digest-only blocks are
  // built at, and the sweep's integral-free run.
  const DeviceTensor* zeroH1e = nullptr;
  const DeviceTensor* zeroH2e = nullptr;
  // reverse
  ScRdmAdjoint* adjDot = nullptr;
  ScRdmAdjoint* adjFree = nullptr;
  // THE ORBITAL PAIR (#158): the second direction is not a CI one but a
  // second ORBITAL direction v = rot.dirs[1], beside u = rot.dirs[0] -- the
  // orbital block's sandwich K3-K5 (:terms' ScMixedWalk says what each mode
  // forms). Both directions carry their tangent blocks with the digests'
  // share merged, and the orbital energies respond to kappa (`epsKappa`).
  // `forward` / `reverse` stay false: nothing CI-side runs. The reverse
  // pairing reads the guarded term's multipliers the forward finish forms
  // (#190), so `pairReverse` needs `pairForward`. `pairScaleU` / `V` are
  // |u|^2, |v|^2, the guarded term's population floor's scale.
  bool pairForward = false;
  bool pairReverse = false;
  double pairScaleU = 1.0;
  double pairScaleV = 1.0;
};

// A class's walk state for `d`, its flags and borrowed vectors set; the class
// fills in its own blocks.
ScMixedWalk mixedWalkOf(const ScMixedDirection& d) {
  ScMixedWalk w;
  w.forward = d.forward;
  w.reverse = d.reverse;
  w.epsKappa = d.epsKappa;
  w.epsCi = d.epsCi;
  w.epsCoreU = &d.epsCoreU;
  w.epsVirtU = &d.epsVirtU;
  w.epsCoreWU = &d.epsCoreWU;
  w.epsVirtWU = &d.epsVirtWU;
  w.pairForward = d.pairForward;
  w.pairReverse = d.pairReverse;
  w.pairScaleU = d.pairScaleU;
  w.pairScaleV = d.pairScaleV;
  return w;
}

// The three reverse sweeps above, for one class: `sweep(acts, h1e, h2e, out)`
// is the class's own sweep written once, `acts` its M_G and `walk.dActive`
// their tangents along w.
template <class SweepFn>
void mixedReverseSweeps(SweepFn&& sweep, const std::vector<ScActiveAccum>& acts,
                        const ScMixedWalk& walk, const DeviceTensor& h1e,
                        const DeviceTensor& h2e, const DeviceTensor& h1eDot,
                        const DeviceTensor& h2eDot, const ScMixedDirection& d) {
  sweep(walk.dActive, h1e, h2e, *d.adjDot);
  sweep(acts, h1eDot, h2eDot, *d.adjDot);
  sweep(acts, *d.zeroH1e, *d.zeroH2e, *d.adjFree);
}

// The derivative of ONE einsum term with respect to its operand `slot`,
// weighted by `mult` (the adjoint of the term's OUTPUT) and accumulated into
// `dst` (the adjoint of that operand). `subs` and `coef` are the forward
// term's own, copied verbatim from the make_* chain; `others` are its
// operands EXCEPT `slot`, in order -- which is the operand order
// derivativeSubscripts writes, with `mult` appended last. The dropped
// operand's value is never read, which is why a block's intermediate (fdm2,
// dm2b) needs no value here, only a shape.
void adjointTerm(const std::string& subs, const int slot,
                 std::vector<const DeviceTensor*> others, const DeviceTensor& mult,
                 const DeviceTensor& dst, const double coef, const wwrStream_t stream) {
  others.push_back(&mult);
  deviceEinsumAccum(derivativeSubscripts(subs, slot), others, dst, coef, stream);
}

// The adjoint of `deviceTransposeAccum(x, axes, out, coef, s)`: a transpose is
// a one-operand, no-contraction einsum, so this builds that einsum's string
// from the SAME `axes` the forward chain spells (einsum/einsum_planner.cppm's
// transposeSubscripts, which is what deviceTransposeAccum itself calls) and
// hands it to adjointTerm -- rather than inverting the permutation by hand.
void adjointTranspose(const std::vector<int>& axes, const DeviceTensor& mult,
                      const DeviceTensor& dst, const double coef, const wwrStream_t stream) {
  static constexpr std::string_view kLabels = "ABCDEFGH";
  const std::size_t rank = axes.size();
  std::string in(kLabels.substr(0, rank));
  std::string out(rank, '?');
  for (std::size_t i = 0; i < rank; ++i)
    out[i] = kLabels[static_cast<std::size_t>(axes[i])];
  adjointTerm(in + "->" + out, 0, {}, mult, dst, coef, stream);
}

// The one-electron matrix make_a17 and make_a19 build internally before they
// contract anything: h1e minus the active-space exchange trace. It reads no
// RDM, so it is CONSTANT here -- rebuilt as a value, never differentiated.
DeviceTensor adjointDressedH1e(const DeviceTensor& h1e, const DeviceTensor& h2e,
                               const DeviceResources& dr) {
  DeviceTensor dressed = deviceEinsumNew("mn->mn", {&h1e}, 1.0, dr);
  deviceEinsumAccum("mjjn->mn", {&h2e}, dressed, -1.0, dr.stream());
  return dressed;
}

// --- the diagonal scatters of make_a16 / make_a22 -----------------------------
//
// Those three are not einsums: device_einsum.cu's energy_diag_a* kernels write
// an n^4 intermediate onto a DIAGONAL SLICE of an n^6 block
// (a16[:, i, :, :, :, i] += fdm2 and the two a22 ones). The adjoint of a
// diagonal scatter is a diagonal GATHER, and that one IS an einsum -- with an
// explicit delta rather than a repeated input label, so nothing has to read a
// diagonal:
//
//   m_fdm2[p,r,a,b] += sum_t m_a16[p,t,r,a,b,t]
//                    = sum_{t,u} m_a16[p,t,r,a,b,u] delta[t,u].
//
// One function each, named for the kernel it undoes, with that kernel's own
// sign and factor.
void adjointDiagA16(const DeviceTensor& mA16, const DeviceTensor& delta,
                    const DeviceTensor& mFdm2, const wwrStream_t stream) {
  deviceEinsumAccum("ptrabu,tu->prab", {&mA16, &delta}, mFdm2, 1.0, stream);
}

void adjointDiagA22a(const DeviceTensor& mA22, const DeviceTensor& delta,
                     const DeviceTensor& mFdm2a, const wwrStream_t stream) {
  deviceEinsumAccum("ptkubc,tu->pkbc", {&mA22, &delta}, mFdm2a, -1.0, stream);
}

void adjointDiagA22b(const DeviceTensor& mA22, const DeviceTensor& delta,
                     const DeviceTensor& mFdm2b, const wwrStream_t stream) {
  deviceEinsumAccum("ptkauc,tu->pkac", {&mA22, &delta}, mFdm2b, 2.0, stream);
}

// --- Sr's blocks --------------------------------------------------------------

// make_a16, term by term: ten einsums against dm3, three transposes of the
// f3 digests (the only place the 4-RDM enters the SC energy at all, hence the
// only source of the f3 adjoints), and the fdm2 intermediate it scatters onto
// its own diagonal.
void adjointA16(const DeviceTensor& m, const DeviceTensor& h1e, const DeviceTensor& h2e,
                const DeviceTensor& delta, ScRdmAdjoint& out, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const DeviceTensor& d3 = out.dm3;
  adjointTerm("ib,rpqiac->pqrabc", 1, {&h1e}, m, d3, -1.0, s);
  adjointTerm("ia,rpqbic->pqrabc", 1, {&h1e}, m, d3, 1.0, s);
  adjointTerm("ci,rpqbai->pqrabc", 1, {&h1e}, m, d3, -1.0, s);
  adjointTranspose({1, 4, 0, 2, 5, 3}, m, out.f3ca, -1.0, s);
  adjointTerm("kbia,rpqcki->pqrabc", 1, {&h2e}, m, d3, -1.0, s);
  adjointTerm("kbaj,rpqjkc->pqrabc", 1, {&h2e}, m, d3, -1.0, s);
  adjointTerm("cbij,rpqjai->pqrabc", 1, {&h2e}, m, d3, 1.0, s);

  // fdm2 = einsum("kbij,rpajki->prab", h2e, dm3), scattered onto a16's
  // diagonal: gather the diagonal back, then push through fdm2's own term.
  {
    const int64_t n = m.dims[0];
    const DeviceTensor mFdm2 = DeviceTensor::zeros({n, n, n, n}, dr);
    adjointDiagA16(m, delta, mFdm2, s);
    adjointTerm("kbij,rpajki->prab", 1, {&h2e}, mFdm2, d3, 1.0, s);
  }

  adjointTranspose({1, 2, 0, 4, 3, 5}, m, out.f3ac, 1.0, s);
  adjointTranspose({1, 2, 0, 4, 3, 5}, m, out.f3ca, -1.0, s);
  adjointTerm("jbij,rpqiac->pqrabc", 1, {&h2e}, m, d3, 1.0, s);
  adjointTerm("cjka,rpqbjk->pqrabc", 1, {&h2e}, m, d3, -1.0, s);
  adjointTerm("jcij,rpqbai->pqrabc", 1, {&h2e}, m, d3, 1.0, s);
}

// make_a17's two terms. Its one-electron operand is the dressed matrix above,
// which carries no RDM.
void adjointA17(const DeviceTensor& m, const DeviceTensor& h1e, const DeviceTensor& h2e,
                ScRdmAdjoint& out, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const DeviceTensor dressed = adjointDressedH1e(h1e, h2e, dr);
  adjointTerm("pi,cabi->abcp", 1, {&dressed}, m, out.dm2, -1.0, s);
  adjointTerm("kpij,cabjki->abcp", 1, {&h2e}, m, out.dm3, -1.0, s);
}

// make_a19's two terms, with the same dressed matrix.
void adjointA19(const DeviceTensor& m, const DeviceTensor& h1e, const DeviceTensor& h2e,
                ScRdmAdjoint& out, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const DeviceTensor dressed = adjointDressedH1e(h1e, h2e, dr);
  adjointTerm("pi,ai->ap", 1, {&dressed}, m, out.dm1, -1.0, s);
  adjointTerm("kpij,ajki->ap", 1, {&h2e}, m, out.dm2, -1.0, s);
}

// --- Si's blocks --------------------------------------------------------------

// make_a22: twelve einsums, three f3 transposes, and TWO fdm2 intermediates
// scattered onto its diagonal -- fdm2a, one term of its own, and fdm2b, six.
void adjointA22(const DeviceTensor& m, const DeviceTensor& h1e, const DeviceTensor& h2e,
                const DeviceTensor& delta, ScRdmAdjoint& out, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = m.dims[0];
  const DeviceTensor& d2 = out.dm2;
  const DeviceTensor& d3 = out.dm3;
  adjointTerm("pb,kipjac->ijkabc", 1, {&h1e}, m, d3, -1.0, s);
  adjointTerm("pa,kibjpc->ijkabc", 1, {&h1e}, m, d3, -1.0, s);
  adjointTerm("cp,kibjap->ijkabc", 1, {&h1e}, m, d3, 1.0, s);
  adjointTerm("cqra,kibjqr->ijkabc", 1, {&h2e}, m, d3, 1.0, s);
  adjointTerm("qcpq,kibjap->ijkabc", 1, {&h2e}, m, d3, -1.0, s);
  adjointTranspose({1, 5, 0, 2, 4, 3}, m, out.f3ac, -1.0, s);

  {
    const DeviceTensor mFdm2a = DeviceTensor::zeros({n, n, n, n}, dr);
    adjointDiagA22a(m, delta, mFdm2a, s);
    adjointTerm("pqrb,kiqcpr->ikbc", 1, {&h2e}, mFdm2a, d3, 1.0, s);
  }

  adjointTerm("pqab,kiqjpc->ijkabc", 1, {&h2e}, m, d3, -1.0, s);
  adjointTerm("pcrb,kiajpr->ijkabc", 1, {&h2e}, m, d3, 1.0, s);
  adjointTerm("cqrb,kiqjar->ijkabc", 1, {&h2e}, m, d3, 1.0, s);
  adjointTranspose({1, 3, 0, 4, 2, 5}, m, out.f3ac, -1.0, s);
  adjointTranspose({1, 3, 0, 4, 2, 5}, m, out.f3ca, 1.0, s);
  adjointTerm("jb,kiac->ijkabc", 1, {&h1e}, m, d2, 2.0, s);
  adjointTerm("pjrb,kiprac->ijkabc", 1, {&h2e}, m, d3, 2.0, s);

  {
    const DeviceTensor mFdm2b = DeviceTensor::zeros({n, n, n, n}, dr);
    adjointDiagA22b(m, delta, mFdm2b, s);
    adjointTerm("pa,kipc->ikac", 1, {&h1e}, mFdm2b, d2, 1.0, s);
    adjointTerm("cp,kiap->ikac", 1, {&h1e}, mFdm2b, d2, -1.0, s);
    adjointTerm("cqra,kiqr->ikac", 1, {&h2e}, mFdm2b, d2, -1.0, s);
    adjointTerm("qcpq,kiap->ikac", 1, {&h2e}, mFdm2b, d2, 1.0, s);
    adjointTerm("pqra,kiqcpr->ikac", 1, {&h2e}, mFdm2b, d3, 1.0, s);
    adjointTerm("rcpq,kiaqrp->ikac", 1, {&h2e}, mFdm2b, d3, -1.0, s);
  }
}

// make_a23's four terms.
void adjointA23(const DeviceTensor& m, const DeviceTensor& h1e, const DeviceTensor& h2e,
                ScRdmAdjoint& out, const wwrStream_t s) {
  adjointTerm("ip,caib->abcp", 1, {&h1e}, m, out.dm2, -1.0, s);
  adjointTerm("pijk,cajbik->abcp", 1, {&h2e}, m, out.dm3, -1.0, s);
  adjointTerm("bp,ca->abcp", 1, {&h1e}, m, out.dm1, 2.0, s);
  adjointTerm("pibk,caik->abcp", 1, {&h2e}, m, out.dm2, 2.0, s);
}

// make_a25's three RDM terms; its fourth (`ap->pa` against h1e alone) reads no
// RDM, so it has no derivative -- the same term make_a25_dot drops.
void adjointA25(const DeviceTensor& m, const DeviceTensor& h1e, const DeviceTensor& h2e,
                ScRdmAdjoint& out, const wwrStream_t s) {
  adjointTerm("pi,ai->ap", 1, {&h1e}, m, out.dm1, -1.0, s);
  adjointTerm("pijk,jaik->ap", 1, {&h2e}, m, out.dm2, -1.0, s);
  adjointTerm("piaj,ij->ap", 1, {&h2e}, m, out.dm1, 2.0, s);
}

// make_dm3_h = 2 dm2 (x) delta - dm3 transposed.
void adjointDm3H(const DeviceTensor& m, const DeviceTensor& delta, ScRdmAdjoint& out,
                 const wwrStream_t s) {
  adjointTerm("abef,cd->abcdef", 0, {&delta}, m, out.dm2, 2.0, s);
  adjointTranspose({0, 1, 3, 2, 4, 5}, m, out.dm3, -1.0, s);
}

// make_dm2_h = 2 dm1 (x) delta - dm2 transposed.
void adjointDm2H(const DeviceTensor& m, const DeviceTensor& delta, ScRdmAdjoint& out,
                 const wwrStream_t s) {
  adjointTerm("ab,cd->abcd", 0, {&delta}, m, out.dm1, 2.0, s);
  adjointTranspose({0, 1, 3, 2}, m, out.dm2, -1.0, s);
}

// dm1_h = 2 delta - dm1^T (energy_Si builds it inline; make_hdm1 is the same
// expression). The constant drops, so only the transpose is left -- which is
// also make_hdm1_dot, the tangent's.
void adjointHoleDm1(const DeviceTensor& m, ScRdmAdjoint& out, const wwrStream_t s) {
  adjointTranspose({1, 0}, m, out.dm1, -1.0, s);
}

// --- Sijr's and Srsi's blocks -------------------------------------------------

// make_a3's three terms. Its hdm1 operand is a block of its own, so that
// adjoint goes into `mHdm1` and the caller sweeps hdm1 afterwards.
void adjointA3(const DeviceTensor& m, const DeviceTensor& h1e, const DeviceTensor& h2e,
               const DeviceTensor& delta, const DeviceTensor& mHdm1, ScRdmAdjoint& out,
               const wwrStream_t s) {
  adjointTerm("ia,ip->pa", 1, {&h1e}, m, mHdm1, 1.0, s);
  adjointTerm("ijka,pj,ik->pa", 2, {&h2e, &delta}, m, out.dm1, 2.0, s);
  adjointTerm("ijka,jpik->pa", 1, {&h2e}, m, out.dm2, -1.0, s);
}

// make_k27's three terms.
void adjointK27(const DeviceTensor& m, const DeviceTensor& h1e, const DeviceTensor& h2e,
                ScRdmAdjoint& out, const wwrStream_t s) {
  adjointTerm("ai,pi->pa", 1, {&h1e}, m, out.dm1, -1.0, s);
  adjointTerm("iajk,pkij->pa", 1, {&h2e}, m, out.dm2, -1.0, s);
  adjointTerm("iaji,pj->pa", 1, {&h2e}, m, out.dm1, 1.0, s);
}

// --- Srs's blocks (rm2, rm3, a7) ----------------------------------------------

// make_a7's five terms. rm2 and rm3 are blocks of their own: their adjoints
// accumulate, and the caller sweeps rm3 and then rm2 (rm3 feeds rm2).
void adjointA7(const DeviceTensor& m, const DeviceTensor& h1e, const DeviceTensor& h2e,
               const DeviceTensor& mRm2, const DeviceTensor& mRm3, const wwrStream_t s) {
  adjointTerm("bi,pqia->pqab", 1, {&h1e}, m, mRm2, -1.0, s);
  adjointTerm("ai,pqbi->pqab", 1, {&h1e}, m, mRm2, -1.0, s);
  adjointTerm("kbij,pqkija->pqab", 1, {&h2e}, m, mRm3, -1.0, s);
  adjointTerm("kaij,pqkibj->pqab", 1, {&h2e}, m, mRm3, -1.0, s);
  adjointTerm("baij,pqij->pqab", 1, {&h2e}, m, mRm2, -1.0, s);
}

// make_rm3's four terms; the last two read rm2, so this must run BEFORE
// adjointRm2.
void adjointRm3(const DeviceTensor& m, const DeviceTensor& delta, const DeviceTensor& mRm2,
                ScRdmAdjoint& out, const wwrStream_t s) {
  adjointTerm("injmkl->ijklmn", 0, {}, m, out.dm3, 1.0, s);
  adjointTerm("jn,imkl->ijklmn", 1, {&delta}, m, out.dm2, -1.0, s);
  adjointTerm("km,ijln->ijklmn", 1, {&delta}, m, mRm2, -1.0, s);
  adjointTerm("kn,ijml->ijklmn", 1, {&delta}, m, mRm2, -1.0, s);
}

// make_a7's rm2, its two terms.
void adjointRm2(const DeviceTensor& m, const DeviceTensor& delta, ScRdmAdjoint& out,
                const wwrStream_t s) {
  adjointTerm("iljk->ijkl", 0, {}, m, out.dm2, 1.0, s);
  adjointTerm("ik,jl->ijkl", 0, {&delta}, m, out.dm1, -1.0, s);
}

// --- Sij's blocks (hdm2, hdm3, a9) --------------------------------------------

// make_a9's nine terms. hdm2 and hdm3 are blocks of their own; hdm3 feeds
// hdm2, so the caller sweeps a9, then hdm3, then hdm2.
void adjointA9(const DeviceTensor& m, const DeviceTensor& h1e, const DeviceTensor& h2e,
               const DeviceTensor& mHdm2, const DeviceTensor& mHdm3, const wwrStream_t s) {
  adjointTerm("ib,pqai->pqab", 1, {&h1e}, m, mHdm2, 1.0, s);
  adjointTerm("ijib,pqaj->pqab", 1, {&h2e}, m, mHdm2, 2.0, s);
  adjointTerm("ijjb,pqai->pqab", 1, {&h2e}, m, mHdm2, -1.0, s);
  adjointTerm("ijkb,pkqaij->pqab", 1, {&h2e}, m, mHdm3, -1.0, s);
  adjointTerm("ia,pqib->pqab", 1, {&h1e}, m, mHdm2, 1.0, s);
  adjointTerm("ijja,pqib->pqab", 1, {&h2e}, m, mHdm2, -1.0, s);
  adjointTerm("ijba,pqji->pqab", 1, {&h2e}, m, mHdm2, -1.0, s);
  adjointTerm("ijia,pqjb->pqab", 1, {&h2e}, m, mHdm2, 2.0, s);
  adjointTerm("ijka,pqkjbi->pqab", 1, {&h2e}, m, mHdm3, -1.0, s);
}

// make_hdm3's nine terms: three read hdm2, five read dm1/dm2, one reads dm3.
void adjointHdm3(const DeviceTensor& m, const DeviceTensor& delta, const DeviceTensor& mHdm2,
                 ScRdmAdjoint& out, const wwrStream_t s) {
  adjointTerm("pb,qrac->pqrabc", 1, {&delta}, m, mHdm2, -1.0, s);
  adjointTerm("br,pqac->pqrabc", 1, {&delta}, m, mHdm2, -1.0, s);
  adjointTerm("bq,prac->pqrabc", 1, {&delta}, m, mHdm2, 2.0, s);
  adjointTerm("ap,bqcr->pqrabc", 1, {&delta}, m, out.dm2, 2.0, s);
  adjointTerm("ap,cr,bq->pqrabc", 2, {&delta, &delta}, m, out.dm1, -4.0, s);
  adjointTerm("cr,bqap->pqrabc", 1, {&delta}, m, out.dm2, 2.0, s);
  adjointTerm("bqapcr->pqrabc", 0, {}, m, out.dm3, -1.0, s);
  adjointTerm("ar,pc,bq->pqrabc", 2, {&delta, &delta}, m, out.dm1, 2.0, s);
  adjointTerm("ar,bqcp->pqrabc", 1, {&delta}, m, out.dm2, -1.0, s);
}

// make_hdm2, through its own dm2b intermediate. Its last two terms are
// delta against delta -- RDM-free, so absent from the derivative, exactly as
// make_hdm2_dot leaves them out.
void adjointHdm2(const DeviceTensor& m, const DeviceTensor& delta, ScRdmAdjoint& out,
                 const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = m.dims[0];
  // dm2b reaches hdm2 through one term, einsum("klij->ijkl", dm2b).
  const DeviceTensor mDm2b = DeviceTensor::zeros({n, n, n, n}, dr);
  adjointTerm("klij->ijkl", 0, {}, m, mDm2b, 1.0, s);
  adjointTerm("ikjl->ijkl", 0, {}, mDm2b, out.dm2, 1.0, s);
  adjointTerm("jk,il->ijkl", 1, {&delta}, mDm2b, out.dm1, -1.0, s);

  adjointTerm("il,kj->ijkl", 1, {&delta}, m, out.dm1, 1.0, s);
  adjointTerm("jk,li->ijkl", 1, {&delta}, m, out.dm1, 1.0, s);
  adjointTerm("ik,lj->ijkl", 1, {&delta}, m, out.dm1, -2.0, s);
  adjointTerm("jl,ki->ijkl", 1, {&delta}, m, out.dm1, -2.0, s);
}

// --- Sir's blocks -------------------------------------------------------------

// make_a12's six terms.
void adjointA12(const DeviceTensor& m, const DeviceTensor& h1e, const DeviceTensor& h2e,
                ScRdmAdjoint& out, const wwrStream_t s) {
  adjointTerm("ia,qpib->pqab", 1, {&h1e}, m, out.dm2, 1.0, s);
  adjointTerm("bi,qpai->pqab", 1, {&h1e}, m, out.dm2, -1.0, s);
  adjointTerm("ijka,qpjbik->pqab", 1, {&h2e}, m, out.dm3, 1.0, s);
  adjointTerm("kbij,qpajki->pqab", 1, {&h2e}, m, out.dm3, -1.0, s);
  adjointTerm("bjka,qpjk->pqab", 1, {&h2e}, m, out.dm2, -1.0, s);
  adjointTerm("jbij,qpai->pqab", 1, {&h2e}, m, out.dm2, 1.0, s);
}

// make_a13's twelve terms -- the longest chain of any block here.
void adjointA13(const DeviceTensor& m, const DeviceTensor& h1e, const DeviceTensor& h2e,
                const DeviceTensor& delta, ScRdmAdjoint& out, const wwrStream_t s) {
  adjointTerm("ia,qbip->pqab", 1, {&h1e}, m, out.dm2, -1.0, s);
  adjointTerm("pa,qb->pqab", 1, {&h1e}, m, out.dm1, 2.0, s);
  adjointTerm("bi,qiap->pqab", 1, {&h1e}, m, out.dm2, 1.0, s);
  adjointTerm("pa,bi,qi->pqab", 2, {&delta, &h1e}, m, out.dm1, -2.0, s);
  adjointTerm("ijka,qbjpik->pqab", 1, {&h2e}, m, out.dm3, -1.0, s);
  adjointTerm("kbij,qjapki->pqab", 1, {&h2e}, m, out.dm3, 1.0, s);
  adjointTerm("blma,qmlp->pqab", 1, {&h2e}, m, out.dm2, 1.0, s);
  adjointTerm("kpma,qbkm->pqab", 1, {&h2e}, m, out.dm2, 2.0, s);
  adjointTerm("bpma,qm->pqab", 1, {&h2e}, m, out.dm1, -2.0, s);
  adjointTerm("lbkl,qkap->pqab", 1, {&h2e}, m, out.dm2, -1.0, s);
  adjointTerm("ap,mbkl,qlmk->pqab", 2, {&delta, &h2e}, m, out.dm2, -2.0, s);
  adjointTerm("ap,lbkl,qk->pqab", 2, {&delta, &h2e}, m, out.dm1, 2.0, s);
}

// --- what a class calls ---------------------------------------------------------

// A RAW RDM that a class contracts directly (Sr's norm reads dm1/dm2/dm3,
// Sir's dm1/dm2, Srsi's dm1): its accumulator IS the Eq. 48 coefficient
// contribution, so it only has to be added in.
void adjointAddRaw(const std::vector<ScActiveAccum>& dActive, const DeviceTensor* block,
                   const DeviceTensor& dst, const wwrStream_t stream) {
  if (const DeviceTensor* m = activeAccumOf(dActive, block); m != nullptr)
    addTensorInto(*m, dst, stream);
}

}  // namespace nevpt2
