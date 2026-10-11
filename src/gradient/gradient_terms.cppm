// nevpt2.gradient:terms -- the generic machinery every SC class's
// pseudodensity assembly is written in: one bilinear term of a class's norm
// or Dyall-Hamiltonian expectation value (ScTerm), evaluating a list of them
// into the per-tuple vectors the multipliers need, differentiating the same
// list with respect to each of its operands, and contracting a pseudodensity
// back against the operand it is conjugate to.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :terms;` and from no importer of nevpt2.gradient.
// (Used by gradient_sc_classes.cppm and gradient.cpp.)
//
// Every einsum here is issued on res.stream() and every tensor it allocates
// is drawn from res's pool, freed when it goes out of scope -- the slab's
// pseudodensities at the end of the slab's iteration, a class's M / E when
// the class returns.
module nevpt2.gradient:terms;

import std;
import nevpt2.gradient;

namespace nevpt2 {

// An ScTerm operand's role: one of the class's external integral blocks (the
// index into the class's block list) or the active-space block the term
// contracts them against.
inline constexpr int kScActive = -1;

struct ScOperand {
  const DeviceTensor* t = nullptr;
  int block = kScActive;
};

// One term of a class's norm or H: `coef * einsum(subs, ops...)` accumulated
// into the class's per-tuple vector, whose indices are `subs`' output labels.
// Copied verbatim from the matching deviceEinsum* call in nevpt2.energy's
// energy_<class> -- same subscripts, same coefficient, same operands.
struct ScTerm {
  std::string subs;
  std::vector<ScOperand> ops;
  double coef = 1.0;
};

bool termHasActive(const ScTerm& term) {
  for (const ScOperand& o : term.ops)
    if (o.block == kScActive) return true;
  return false;
}

// Which terms of a list a pass covers. A class's per-tuple norm (or H) is the
// sum of both passes; only the WithActive half has an Eqs. 42-43 counterpart,
// so the two are accumulated separately and added.
enum class ScTermFilter { WithActive, WithoutActive };

// Accumulates `sum coef * einsum(term)` over the matching terms into `out`
// (shaped as the class's per-tuple vector).
void evalTerms(const std::vector<ScTerm>& terms, const DeviceTensor& out, ScTermFilter filter,
               wwrStream_t stream) {
  for (const ScTerm& term : terms) {
    if ((filter == ScTermFilter::WithActive) != termHasActive(term)) continue;
    std::vector<const DeviceTensor*> ops;
    ops.reserve(term.ops.size());
    for (const ScOperand& o : term.ops) ops.push_back(o.t);
    deviceEinsumAccum(term.subs, ops, out, term.coef, stream);
  }
}

// out += x, same shape: a no-permutation transpose, the one way this tree
// adds two device tensors.
void addInto(const DeviceTensor& x, const DeviceTensor& out, wwrStream_t stream) {
  std::vector<int> axes(static_cast<std::size_t>(x.rank()));
  for (std::size_t a = 0; a < axes.size(); ++a) axes[a] = static_cast<int>(a);
  deviceTransposeAccum(x, axes, out, 1.0, stream);
}

// <x, d>: the elementwise product of two equal-shaped tensors, summed. Issued
// as an einsum reduced to x's LEADING axis -- not to a scalar -- and finished
// on the host, so the kernel never writes a rank-0 output.
double dotTensors(const DeviceTensor& x, const DeviceTensor& d, const DeviceResources& res,
                  std::source_location loc = std::source_location::current()) {
  if (x.dims != d.dims || x.rank() < 1)
    check(false, "dotTensors: operands must have the same shape, of rank >= 1", loc);
  static constexpr std::string_view kLabels = "abcdef";
  const std::string lab(kLabels.substr(0, static_cast<std::size_t>(x.rank())));
  const DeviceTensor reduced =
      deviceEinsumNew(lab + "," + lab + "->" + lab.substr(0, 1), {&x, &d}, 1.0, res);
  const Tensor host = downloadTensor(reduced, res.stream());
  double sum = 0.0;
  for (const double v : host.data()) sum += v;
  return sum;
}

// One active-space pseudodensity under assembly (Park Eqs. 42-43's M / E),
// keyed by the metric or Dyall-Hamiltonian block it is conjugate to.
struct ScActiveAccum {
  const DeviceTensor* g = nullptr;
  DeviceTensor d;
};

// The accumulator for `g`, created on first use (shaped like `g` -- its
// conjugate). A class contracts each of its blocks in several terms, and a
// block that appeared on the norm side and on the H side would share one
// accumulator, which is what the Eqs. 42-43 identity sums over anyway.
DeviceTensor& activeAccumFor(std::vector<ScActiveAccum>& accums, const DeviceTensor* g,
                             const DeviceResources& res) {
  for (ScActiveAccum& a : accums)
    if (a.g == g) return a.d;
  accums.push_back(ScActiveAccum{g, DeviceTensor::zeros(g->dims, res)});
  return accums.back().d;
}

// Accumulates the multiplier-weighted derivative of every term with respect
// to EVERY one of its operands: an external one into dBlocks[its block
// index] (Eq. 41's D), an active one into its ScActiveAccum (Eqs. 42-43's M
// or E). `mult` is the class's per-tuple multiplier -- P for a norm term
// list, Q for an H one -- shaped as the terms' output labels.
//
// The derivative subscripts are REWRITTEN from the term's own
// (nevpt2.sc_amplitudes' derivativeSubscripts), never hand-written: each is
// the same contraction with one operand replaced by `mult` and the term's
// output labels and the dropped operand's swapping roles.
void accumDerivatives(const std::vector<ScTerm>& terms, const DeviceTensor& mult,
                      std::vector<DeviceTensor>& dBlocks,
                      std::vector<ScActiveAccum>& dActive, const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  for (const ScTerm& term : terms) {
    for (std::size_t k = 0; k < term.ops.size(); ++k) {
      const std::string subs = derivativeSubscripts(term.subs, static_cast<int>(k));
      std::vector<const DeviceTensor*> ops;
      ops.reserve(term.ops.size());
      for (std::size_t j = 0; j < term.ops.size(); ++j)
        if (j != k) ops.push_back(term.ops[j].t);
      ops.push_back(&mult);
      if (term.ops[k].block == kScActive) {
        const DeviceTensor& out = activeAccumFor(dActive, term.ops[k].t, res);
        deviceEinsumAccum(subs, ops, out, term.coef, stream);
      } else {
        deviceEinsumAccum(subs, ops, dBlocks[static_cast<std::size_t>(term.ops[k].block)],
                          term.coef, stream);
      }
    }
  }
}

// One slab's per-tuple multipliers: P and Q over the tuple vector's own
// shape, and the slab's (norm, energy). For a class whose tuples ARE its
// perturbers this is scMultipliers applied elementwise (plainMultipliers
// below); Sijr and Srsi first symmetrize and pack a triangle, then scatter
// the triangle's multipliers back over the full tuple shape.
struct ScSlabMultipliers {
  Tensor p;
  Tensor q;
  double norm = 0.0;
  double energy = 0.0;
};

// `v[b0:b1)` of a rank-1 host tensor (an orbital-energy vector).
std::vector<double> sliceHost(const Tensor& v, int64_t b0, int64_t b1) {
  return std::vector<double>(v.data().begin() + b0, v.data().begin() + b1);
}

// The tuple-is-perturber case: `delta` in the tuple vector's flat order.
ScSlabMultipliers plainMultipliers(const Tensor& norm, const Tensor& h,
                                   const std::vector<double>& delta) {
  const ScMultipliers m = scMultipliers(norm.data(), h.data(), delta);
  return ScSlabMultipliers{Tensor::fromFlat(norm.dims(), m.p),
                           Tensor::fromFlat(norm.dims(), m.q), m.norm, m.energy};
}

// The per-slab core every class but Sijrs and Srsi runs:
//
//   1. evaluate the class's norm and H term lists into per-tuple vectors
//      (splitting off the terms with no active block, which Eqs. 42-43 cannot
//      see);
//   2. turn them into the per-tuple multipliers P and Q (`multiplierFn`);
//   3. assemble this slab's pseudodensities -- Eq. 41's D for each external
//      block, Eqs. 42-43's M / E for each active one;
//   4. contract the D family back against its own slab, accumulating Eq. 40's
//      sum into `acc`.
//
// `blocks` lists the class's external integral blocks, in the order ScTerm's
// `block` indices name them; a D accumulator is allocated per block, shaped
// like it, and freed when this returns -- D is a per-slab object.
//
// `dActive` is the CALLER's: the M / E of Eqs. 42-43 have no external index
// left, so they accumulate over the whole class's walk, and the caller
// contracts them back (and frees them) once the walk is done --
// activeContraction() below.
template <class MultiplierFn>
void assembleSlab(const std::vector<ScTerm>& normTerms, const std::vector<ScTerm>& hTerms,
                  const std::vector<int64_t>& tupleDims,
                  const std::vector<const DeviceTensor*>& blocks,
                  std::vector<ScActiveAccum>& dActive, MultiplierFn&& multiplierFn,
                  ScClassGradient& acc, const DeviceResources& res) {
  const wwrStream_t stream = res.stream();

  // The with-active halves kept apart, so actReference can exclude the terms
  // Eqs. 42-43 have no counterpart for, then added in for the multipliers.
  DeviceTensor normActive = DeviceTensor::zeros(tupleDims, res);
  DeviceTensor hActive = DeviceTensor::zeros(tupleDims, res);
  DeviceTensor norm = DeviceTensor::zeros(tupleDims, res);
  DeviceTensor h = DeviceTensor::zeros(tupleDims, res);
  evalTerms(normTerms, normActive, ScTermFilter::WithActive, stream);
  evalTerms(hTerms, hActive, ScTermFilter::WithActive, stream);
  evalTerms(normTerms, norm, ScTermFilter::WithoutActive, stream);
  evalTerms(hTerms, h, ScTermFilter::WithoutActive, stream);
  addInto(normActive, norm, stream);
  addInto(hActive, h, stream);

  // No sync before the downloads: each is enqueued on the same stream as the
  // einsums that wrote its source, behind them, and downloadTensor
  // synchronizes that stream before the host reads (as nevpt2.energy's
  // per-slab finish does).
  const Tensor hostNorm = downloadTensor(norm, stream);
  const Tensor hostH = downloadTensor(h, stream);
  const Tensor hostNormActive = downloadTensor(normActive, stream);
  const Tensor hostHActive = downloadTensor(hActive, stream);

  const ScSlabMultipliers m = multiplierFn(hostNorm, hostH);
  acc.norm += m.norm;
  acc.energy += m.energy;
  acc.actReference +=
      dotHost(m.p.data(), hostNormActive.data()) + dotHost(m.q.data(), hostHActive.data());

  const DeviceTensor mp = uploadTensor(m.p, res);
  const DeviceTensor mq = uploadTensor(m.q, res);
  std::vector<DeviceTensor> dBlocks;
  dBlocks.reserve(blocks.size());
  for (const DeviceTensor* b : blocks) dBlocks.push_back(DeviceTensor::zeros(b->dims, res));
  accumDerivatives(normTerms, mp, dBlocks, dActive, res);
  accumDerivatives(hTerms, mq, dBlocks, dActive, res);

  double ext = 0.0;
  for (std::size_t i = 0; i < blocks.size(); ++i) ext += dotTensors(*blocks[i], dBlocks[i], res);
  acc.extContraction += 0.5 * ext;
}

// Eqs. 42-43 contracted back, once a class's walk is over: sum_G <G, M_G>
// over the active-space pseudodensities it accumulated.
double activeContraction(const std::vector<ScActiveAccum>& dActive, const DeviceResources& res) {
  double sum = 0.0;
  for (const ScActiveAccum& a : dActive) sum += dotTensors(*a.g, a.d, res);
  return sum;
}

// The accumulator `g` got, or nullptr -- how a class reaches one of its own M
// / E after the walk (Srs, which differentiates its Dyall block in turn).
const DeviceTensor* activeAccumOf(const std::vector<ScActiveAccum>& dActive,
                                  const DeviceTensor* g) {
  for (const ScActiveAccum& a : dActive)
    if (a.g == g) return &a.d;
  return nullptr;
}

}  // namespace nevpt2
