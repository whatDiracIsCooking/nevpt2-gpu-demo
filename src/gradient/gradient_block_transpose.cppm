// nevpt2.gradient:block_transpose -- the Dyall blocks' TRANSPOSES in the
// integrals (#240): every make_* that reads the active integrals, run
// backwards from its accumulator M_G = dE/dG onto dE/d(h1e) and dE/d(h2e).
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :block_transpose;` and from no importer of
// nevpt2.gradient. (Used by gradient_sc_classes.cppm, gradient_pc_classes.cppm
// and gradient.cpp.)
//
// WHY THEY EXIST. The orbital source term's Dyall-block half is
// sum_G <M_G, Gdot(kappa)>, and Gdot(kappa) is the block's own builder at the
// direction's dotted integrals with its integral-free operands zeroed
// (docs/gradients.md, Sec. 6: "a block's tangent is that same builder called
// with the dotted integrals"). A builder is LINEAR in the integrals it reads,
// so that pairing is <G^T(M_G), (h1e_dot, h2e_dot)> -- one transpose per
// block, run ONCE, after which the whole Y matrix is a pull-back of the
// integrals' adjoint through the kappa action (gradient.cppm, "Y in one
// reverse pass"), not one slab walk per sixteen basis directions.
//
// HOW EACH IS WRITTEN. Exactly as :rdm_adjoint writes the RDM side: each
// block's make_* chain term by term (same subscript strings, same
// coefficients, same operands, so the three read side by side), with the
// INTEGRAL operand dropped instead of the RDM one -- derivativeSubscripts'
// rewriting, through `integralTerm` below. Three things differ from the RDM
// side, and each is said where it happens:
//
//   * an integral operand may REPEAT a label (make_a16's `jbij`, make_k27's
//     `iaji`, ...): it reads a diagonal, and an einsum cannot write one, so
//     integralTerm takes the derivative onto the label-collapsed operand and
//     scatters it back onto the diagonal with a Kronecker delta;
//   * the f3 digests enter make_a16 / make_a22 as plain addends with no
//     integral factor -- they are absent here, as from the rotation's tangent
//     (their own term is the digests' transpose, hbar, which the caller adds);
//   * make_a17 / make_a19's dressed one-electron matrix h1e - sum_j (mjjn) is
//     an INTEGRAL here (the RDM side rebuilds it as a constant), so its
//     adjoint is collected and pushed back through the dressing.
//
// WHAT CHECKS THEM, exactly: duality. A builder is linear in (h1e, h2e) at
// fixed RDMs, so <G^T(M), Idot> == <M, G(Idot)> for every seeded M and Idot,
// to round-off -- scBlockTransposeDuality / pcBlockTransposeDuality in
// gradient.cpp, per class, which --dump-source-terms gates; and the one-pass
// Y against the probed one, element by element (--probed-y).
//
// Every einsum on res.stream(), every tensor from res's pool and freed when
// its function returns.
module nevpt2.gradient:block_transpose;

import std;
import nevpt2.gradient;
import :general;      // addTensorInto
import :rdm_adjoint;  // adjointTerm, adjointDiagA16 / A22a / A22b
import :terms;        // ScActiveAccum / activeAccumOf

namespace nevpt2 {

// The derivative of ONE einsum term with respect to its INTEGRAL operand
// `slot`, weighted by `mult` (the adjoint of the term's output), accumulated
// into `dst` (shaped like that operand): :rdm_adjoint's adjointTerm, plus the
// diagonal case. `others` are the term's operands EXCEPT `slot`, in order.
//
// A dropped operand that repeats one label (`jbij`) is read on a diagonal.
// The term is then the same term over the label-collapsed operand (`jbi`,
// R[j,b,i] = X[j,b,i,j]), whose derivative IS an einsum; it lands in a scratch
// tensor and is scattered back onto the diagonal against a delta, `jbi,jZ->
// jbiZ` -- two einsums of at most the forward term's own labels, so within the
// contraction kernel's limits wherever the forward was. More than one
// repeated label, or one repeated three times, appears in no make_* and is
// our bug.
void integralTerm(const std::string& subs, const int slot,
                  std::vector<const DeviceTensor*> others, const DeviceTensor& mult,
                  const DeviceTensor& dst, const double coef, const DeviceResources& dr,
                  std::source_location loc = std::source_location::current()) {
  const std::size_t arrow = subs.find("->");
  if (arrow == std::string::npos)
    check(false, std::format("integralTerm: no '->' in \"{}\"", subs), loc);
  std::vector<std::string> groups{""};
  for (const char c : subs.substr(0, arrow)) {
    if (c == ',')
      groups.emplace_back();
    else
      groups.back().push_back(c);
  }
  if (slot < 0 || slot >= std::ssize(groups))
    check(false, std::format("integralTerm: slot {} is not an operand of \"{}\"", slot, subs),
          loc);
  const std::string dropped = groups[static_cast<std::size_t>(slot)];
  std::size_t second = std::string::npos;
  for (std::size_t a = 0; a < dropped.size(); ++a) {
    const std::size_t again = dropped.find(dropped[a], a + 1);
    if (again == std::string::npos) continue;
    if (second != std::string::npos || dropped.find(dropped[a], again + 1) != std::string::npos)
      check(false,
            std::format("integralTerm: operand {} of \"{}\" repeats more than one label once",
                        slot, subs),
            loc);
    second = again;
  }
  others.push_back(&mult);
  if (second == std::string::npos) {
    deviceEinsumAccum(derivativeSubscripts(subs, slot, loc), others, dst, coef, dr.stream());
    return;
  }
  // The label-collapsed operand, and the term over it. When the repeated
  // label is a pure TRACE -- read by no other operand and absent from the
  // output, as in every make_* diagonal (`jbij,rpqiac->pqrabc`) -- the term
  // does not depend on it beyond the sum, so the derivative is constant along
  // it: it is taken over the operand with the label dropped altogether, and
  // the delta below both restores it and ties it to its partner.
  const char tied = dropped[second];
  bool traced = subs.substr(arrow + 2).find(tied) == std::string::npos;
  for (std::size_t g = 0; g < groups.size(); ++g)
    if (g != static_cast<std::size_t>(slot) && groups[g].find(tied) != std::string::npos)
      traced = false;
  std::string reduced = dropped;
  reduced.erase(second, 1);
  if (traced) reduced.erase(reduced.find(tied), 1);
  for (const char c : reduced) {
    bool elsewhere = subs.substr(arrow + 2).find(c) != std::string::npos;
    for (std::size_t g = 0; g < groups.size(); ++g)
      if (g != static_cast<std::size_t>(slot) && groups[g].find(c) != std::string::npos)
        elsewhere = true;
    if (!elsewhere)
      check(false,
            std::format("integralTerm: operand {} of \"{}\" sums label '{}' on its own", slot,
                        subs, c),
            loc);
  }
  groups[static_cast<std::size_t>(slot)] = reduced;
  std::string collapsed;
  for (std::size_t g = 0; g < groups.size(); ++g) collapsed += (g ? "," : "") + groups[g];
  collapsed += subs.substr(arrow);
  std::vector<int64_t> dims;
  for (const char c : reduced) dims.push_back(dst.dims[dropped.find(c)]);
  const DeviceTensor g = DeviceTensor::zeros(dims, dr);
  deviceEinsumAccum(derivativeSubscripts(collapsed, slot, loc), others, g, 1.0, dr.stream());
  // Back onto the diagonal: the repeated label's second slot gets a fresh one
  // (`Z`, outside every make_*'s lower-case alphabet) tied to it by delta.
  std::string spread = dropped;
  spread[second] = 'Z';
  const DeviceTensor delta = deviceEye(dst.dims[second], dr);
  deviceEinsumAccum(reduced + "," + std::string{tied, 'Z'} + "->" + spread, {&g, &delta}, dst,
                    coef, dr.stream());
}

// The two integral adjoints a transpose accumulates into: dE/d(h1e) (n, n)
// and dE/d(h2e) (n, n, n, n), the active integrals' own shapes.
struct ScIntegralBars {
  const DeviceTensor& h1e;
  const DeviceTensor& h2e;
};

// --- Sr's blocks ----------------------------------------------------------------

// make_a16's ten integral terms and its fdm2 intermediate (scattered onto
// a16's diagonal: gathered back, as :rdm_adjoint's adjointA16 does). Its
// three f3 transposes carry no integral and are absent.
void transposeA16(const DeviceTensor& m, const DeviceTensor& dm3, const DeviceTensor& delta,
                  const ScIntegralBars& out, const DeviceResources& dr) {
  integralTerm("ib,rpqiac->pqrabc", 0, {&dm3}, m, out.h1e, -1.0, dr);
  integralTerm("ia,rpqbic->pqrabc", 0, {&dm3}, m, out.h1e, 1.0, dr);
  integralTerm("ci,rpqbai->pqrabc", 0, {&dm3}, m, out.h1e, -1.0, dr);
  integralTerm("kbia,rpqcki->pqrabc", 0, {&dm3}, m, out.h2e, -1.0, dr);
  integralTerm("kbaj,rpqjkc->pqrabc", 0, {&dm3}, m, out.h2e, -1.0, dr);
  integralTerm("cbij,rpqjai->pqrabc", 0, {&dm3}, m, out.h2e, 1.0, dr);
  {
    const int64_t n = m.dims[0];
    const DeviceTensor mFdm2 = DeviceTensor::zeros({n, n, n, n}, dr);
    adjointDiagA16(m, delta, mFdm2, dr.stream());
    integralTerm("kbij,rpajki->prab", 0, {&dm3}, mFdm2, out.h2e, 1.0, dr);
  }
  integralTerm("jbij,rpqiac->pqrabc", 0, {&dm3}, m, out.h2e, 1.0, dr);
  integralTerm("cjka,rpqbjk->pqrabc", 0, {&dm3}, m, out.h2e, -1.0, dr);
  integralTerm("jcij,rpqbai->pqrabc", 0, {&dm3}, m, out.h2e, 1.0, dr);
}

// The dressed one-electron matrix make_a17 / make_a19 build first,
// h1e - einsum("mjjn->mn", h2e): its adjoint `mDressed` pushed back onto both
// integrals.
void transposeDressedH1e(const DeviceTensor& mDressed, const ScIntegralBars& out,
                         const DeviceResources& dr) {
  addTensorInto(mDressed, out.h1e, dr.stream());
  integralTerm("mjjn->mn", 0, {}, mDressed, out.h2e, -1.0, dr);
}

// make_a17's two terms, the first through the dressed matrix.
void transposeA17(const DeviceTensor& m, const DeviceTensor& dm2, const DeviceTensor& dm3,
                  const ScIntegralBars& out, const DeviceResources& dr) {
  const int64_t n = dm2.dims[0];
  const DeviceTensor mDressed = DeviceTensor::zeros({n, n}, dr);
  integralTerm("pi,cabi->abcp", 0, {&dm2}, m, mDressed, -1.0, dr);
  integralTerm("kpij,cabjki->abcp", 0, {&dm3}, m, out.h2e, -1.0, dr);
  transposeDressedH1e(mDressed, out, dr);
}

// make_a19's two terms, likewise.
void transposeA19(const DeviceTensor& m, const DeviceTensor& dm1, const DeviceTensor& dm2,
                  const ScIntegralBars& out, const DeviceResources& dr) {
  const int64_t n = dm1.dims[0];
  const DeviceTensor mDressed = DeviceTensor::zeros({n, n}, dr);
  integralTerm("pi,ai->ap", 0, {&dm1}, m, mDressed, -1.0, dr);
  integralTerm("kpij,ajki->ap", 0, {&dm2}, m, out.h2e, -1.0, dr);
  transposeDressedH1e(mDressed, out, dr);
}

// --- Si's blocks ----------------------------------------------------------------

// make_a22's twelve integral terms and its two fdm2 intermediates (fdm2a, one
// term; fdm2b, six), each gathered back off a22's diagonal first. Its three
// f3 transposes are absent, as a16's.
void transposeA22(const DeviceTensor& m, const DeviceTensor& dm2, const DeviceTensor& dm3,
                  const DeviceTensor& delta, const ScIntegralBars& out,
                  const DeviceResources& dr) {
  const int64_t n = m.dims[0];
  integralTerm("pb,kipjac->ijkabc", 0, {&dm3}, m, out.h1e, -1.0, dr);
  integralTerm("pa,kibjpc->ijkabc", 0, {&dm3}, m, out.h1e, -1.0, dr);
  integralTerm("cp,kibjap->ijkabc", 0, {&dm3}, m, out.h1e, 1.0, dr);
  integralTerm("cqra,kibjqr->ijkabc", 0, {&dm3}, m, out.h2e, 1.0, dr);
  integralTerm("qcpq,kibjap->ijkabc", 0, {&dm3}, m, out.h2e, -1.0, dr);
  {
    const DeviceTensor mFdm2a = DeviceTensor::zeros({n, n, n, n}, dr);
    adjointDiagA22a(m, delta, mFdm2a, dr.stream());
    integralTerm("pqrb,kiqcpr->ikbc", 0, {&dm3}, mFdm2a, out.h2e, 1.0, dr);
  }
  integralTerm("pqab,kiqjpc->ijkabc", 0, {&dm3}, m, out.h2e, -1.0, dr);
  integralTerm("pcrb,kiajpr->ijkabc", 0, {&dm3}, m, out.h2e, 1.0, dr);
  integralTerm("cqrb,kiqjar->ijkabc", 0, {&dm3}, m, out.h2e, 1.0, dr);
  integralTerm("jb,kiac->ijkabc", 0, {&dm2}, m, out.h1e, 2.0, dr);
  integralTerm("pjrb,kiprac->ijkabc", 0, {&dm3}, m, out.h2e, 2.0, dr);
  {
    const DeviceTensor mFdm2b = DeviceTensor::zeros({n, n, n, n}, dr);
    adjointDiagA22b(m, delta, mFdm2b, dr.stream());
    integralTerm("pa,kipc->ikac", 0, {&dm2}, mFdm2b, out.h1e, 1.0, dr);
    integralTerm("cp,kiap->ikac", 0, {&dm2}, mFdm2b, out.h1e, -1.0, dr);
    integralTerm("cqra,kiqr->ikac", 0, {&dm2}, mFdm2b, out.h2e, -1.0, dr);
    integralTerm("qcpq,kiap->ikac", 0, {&dm2}, mFdm2b, out.h2e, 1.0, dr);
    integralTerm("pqra,kiqcpr->ikac", 0, {&dm3}, mFdm2b, out.h2e, 1.0, dr);
    integralTerm("rcpq,kiaqrp->ikac", 0, {&dm3}, mFdm2b, out.h2e, -1.0, dr);
  }
}

// make_a23's four terms.
void transposeA23(const DeviceTensor& m, const DeviceTensor& dm1, const DeviceTensor& dm2,
                  const DeviceTensor& dm3, const ScIntegralBars& out, const DeviceResources& dr) {
  integralTerm("ip,caib->abcp", 0, {&dm2}, m, out.h1e, -1.0, dr);
  integralTerm("pijk,cajbik->abcp", 0, {&dm3}, m, out.h2e, -1.0, dr);
  integralTerm("bp,ca->abcp", 0, {&dm1}, m, out.h1e, 2.0, dr);
  integralTerm("pibk,caik->abcp", 0, {&dm2}, m, out.h2e, 2.0, dr);
}

// make_a25's four terms -- its RDM-free `2 h1e^T` included: it IS an integral
// term (the rotation's tangent carries it, make_a25 at the dotted integrals).
void transposeA25(const DeviceTensor& m, const DeviceTensor& dm1, const DeviceTensor& dm2,
                  const ScIntegralBars& out, const DeviceResources& dr) {
  integralTerm("pi,ai->ap", 0, {&dm1}, m, out.h1e, -1.0, dr);
  integralTerm("pijk,jaik->ap", 0, {&dm2}, m, out.h2e, -1.0, dr);
  integralTerm("ap->pa", 0, {}, m, out.h1e, 2.0, dr);
  integralTerm("piaj,ij->ap", 0, {&dm1}, m, out.h2e, 2.0, dr);
}

// --- Sijr's, Srsi's, Srs's, Sij's and Sir's blocks -------------------------------

// make_a3's three terms; hdm1 is an operand like any RDM here.
void transposeA3(const DeviceTensor& m, const DeviceTensor& dm1, const DeviceTensor& dm2,
                 const DeviceTensor& hdm1, const DeviceTensor& delta, const ScIntegralBars& out,
                 const DeviceResources& dr) {
  integralTerm("ia,ip->pa", 0, {&hdm1}, m, out.h1e, 1.0, dr);
  integralTerm("ijka,pj,ik->pa", 0, {&delta, &dm1}, m, out.h2e, 2.0, dr);
  integralTerm("ijka,jpik->pa", 0, {&dm2}, m, out.h2e, -1.0, dr);
}

// make_k27's three terms.
void transposeK27(const DeviceTensor& m, const DeviceTensor& dm1, const DeviceTensor& dm2,
                  const ScIntegralBars& out, const DeviceResources& dr) {
  integralTerm("ai,pi->pa", 0, {&dm1}, m, out.h1e, -1.0, dr);
  integralTerm("iajk,pkij->pa", 0, {&dm2}, m, out.h2e, -1.0, dr);
  integralTerm("iaji,pj->pa", 0, {&dm1}, m, out.h2e, 1.0, dr);
}

// make_a7's five terms, against its hole RDMs rm2 / rm3 (make_a7's own first
// return value and make_rm3).
void transposeA7(const DeviceTensor& m, const DeviceTensor& rm2, const DeviceTensor& rm3,
                 const ScIntegralBars& out, const DeviceResources& dr) {
  integralTerm("bi,pqia->pqab", 0, {&rm2}, m, out.h1e, -1.0, dr);
  integralTerm("ai,pqbi->pqab", 0, {&rm2}, m, out.h1e, -1.0, dr);
  integralTerm("kbij,pqkija->pqab", 0, {&rm3}, m, out.h2e, -1.0, dr);
  integralTerm("kaij,pqkibj->pqab", 0, {&rm3}, m, out.h2e, -1.0, dr);
  integralTerm("baij,pqij->pqab", 0, {&rm2}, m, out.h2e, -1.0, dr);
}

// make_a9's nine terms, against the hole RDMs hdm2 / hdm3.
void transposeA9(const DeviceTensor& m, const DeviceTensor& hdm2, const DeviceTensor& hdm3,
                 const ScIntegralBars& out, const DeviceResources& dr) {
  integralTerm("ib,pqai->pqab", 0, {&hdm2}, m, out.h1e, 1.0, dr);
  integralTerm("ijib,pqaj->pqab", 0, {&hdm2}, m, out.h2e, 2.0, dr);
  integralTerm("ijjb,pqai->pqab", 0, {&hdm2}, m, out.h2e, -1.0, dr);
  integralTerm("ijkb,pkqaij->pqab", 0, {&hdm3}, m, out.h2e, -1.0, dr);
  integralTerm("ia,pqib->pqab", 0, {&hdm2}, m, out.h1e, 1.0, dr);
  integralTerm("ijja,pqib->pqab", 0, {&hdm2}, m, out.h2e, -1.0, dr);
  integralTerm("ijba,pqji->pqab", 0, {&hdm2}, m, out.h2e, -1.0, dr);
  integralTerm("ijia,pqjb->pqab", 0, {&hdm2}, m, out.h2e, 2.0, dr);
  integralTerm("ijka,pqkjbi->pqab", 0, {&hdm3}, m, out.h2e, -1.0, dr);
}

// make_a12's six terms.
void transposeA12(const DeviceTensor& m, const DeviceTensor& dm2, const DeviceTensor& dm3,
                  const ScIntegralBars& out, const DeviceResources& dr) {
  integralTerm("ia,qpib->pqab", 0, {&dm2}, m, out.h1e, 1.0, dr);
  integralTerm("bi,qpai->pqab", 0, {&dm2}, m, out.h1e, -1.0, dr);
  integralTerm("ijka,qpjbik->pqab", 0, {&dm3}, m, out.h2e, 1.0, dr);
  integralTerm("kbij,qpajki->pqab", 0, {&dm3}, m, out.h2e, -1.0, dr);
  integralTerm("bjka,qpjk->pqab", 0, {&dm2}, m, out.h2e, -1.0, dr);
  integralTerm("jbij,qpai->pqab", 0, {&dm2}, m, out.h2e, 1.0, dr);
}

// make_a13's twelve terms; two read h1e / h2e as their MIDDLE operand.
void transposeA13(const DeviceTensor& m, const DeviceTensor& dm1, const DeviceTensor& dm2,
                  const DeviceTensor& dm3, const DeviceTensor& delta, const ScIntegralBars& out,
                  const DeviceResources& dr) {
  integralTerm("ia,qbip->pqab", 0, {&dm2}, m, out.h1e, -1.0, dr);
  integralTerm("pa,qb->pqab", 0, {&dm1}, m, out.h1e, 2.0, dr);
  integralTerm("bi,qiap->pqab", 0, {&dm2}, m, out.h1e, 1.0, dr);
  integralTerm("pa,bi,qi->pqab", 1, {&delta, &dm1}, m, out.h1e, -2.0, dr);
  integralTerm("ijka,qbjpik->pqab", 0, {&dm3}, m, out.h2e, -1.0, dr);
  integralTerm("kbij,qjapki->pqab", 0, {&dm3}, m, out.h2e, 1.0, dr);
  integralTerm("blma,qmlp->pqab", 0, {&dm2}, m, out.h2e, 1.0, dr);
  integralTerm("kpma,qbkm->pqab", 0, {&dm2}, m, out.h2e, 2.0, dr);
  integralTerm("bpma,qm->pqab", 0, {&dm1}, m, out.h2e, -2.0, dr);
  integralTerm("lbkl,qkap->pqab", 0, {&dm2}, m, out.h2e, -1.0, dr);
  integralTerm("ap,mbkl,qlmk->pqab", 1, {&delta, &dm2}, m, out.h2e, -2.0, dr);
  integralTerm("ap,lbkl,qk->pqab", 1, {&delta, &dm1}, m, out.h2e, 2.0, dr);
}

// --- every block, by name ---------------------------------------------------------

// The blocks that read an integral, one per make_*: what a class's transpose
// sweep and the duality check iterate over.
enum class DyallBlock { A16, A17, A19, A22, A23, A25, A3, K27, A7, A9, A12, A13 };

std::string_view dyallBlockName(DyallBlock b) {
  switch (b) {
    case DyallBlock::A16: return "a16";
    case DyallBlock::A17: return "a17";
    case DyallBlock::A19: return "a19";
    case DyallBlock::A22: return "a22";
    case DyallBlock::A23: return "a23";
    case DyallBlock::A25: return "a25";
    case DyallBlock::A3: return "a3";
    case DyallBlock::K27: return "k27";
    case DyallBlock::A7: return "a7";
    case DyallBlock::A9: return "a9";
    case DyallBlock::A12: return "a12";
    case DyallBlock::A13: return "a13";
  }
  return "?";
}

// Everything a builder reads beside the integrals: the RDMs and the hole-side
// metrics built from them (each class builds the ones it needs; a pointer a
// block does not read may stay null).
struct DyallOperands {
  const DeviceTensor* dm1 = nullptr;
  const DeviceTensor* dm2 = nullptr;
  const DeviceTensor* dm3 = nullptr;
  const DeviceTensor* hdm1 = nullptr;
  const DeviceTensor* hdm2 = nullptr;
  const DeviceTensor* hdm3 = nullptr;
  const DeviceTensor* rm2 = nullptr;
  const DeviceTensor* rm3 = nullptr;
  const DeviceTensor* delta = nullptr;  // (n, n) identity
};

// Block `b`'s transpose at accumulator `m`, into `out`. A null operand the
// block reads is our bug.
void dyallBlockTranspose(DyallBlock b, const DeviceTensor& m, const DyallOperands& o,
                         const ScIntegralBars& out, const DeviceResources& dr) {
  const auto need = [b](const DeviceTensor* p) -> const DeviceTensor& {
    check(p != nullptr,
          std::format("dyallBlockTranspose: {} reads an operand it was not handed",
                      dyallBlockName(b)));
    return *p;
  };
  switch (b) {
    case DyallBlock::A16: transposeA16(m, need(o.dm3), need(o.delta), out, dr); return;
    case DyallBlock::A17: transposeA17(m, need(o.dm2), need(o.dm3), out, dr); return;
    case DyallBlock::A19: transposeA19(m, need(o.dm1), need(o.dm2), out, dr); return;
    case DyallBlock::A22:
      transposeA22(m, need(o.dm2), need(o.dm3), need(o.delta), out, dr);
      return;
    case DyallBlock::A23: transposeA23(m, need(o.dm1), need(o.dm2), need(o.dm3), out, dr); return;
    case DyallBlock::A25: transposeA25(m, need(o.dm1), need(o.dm2), out, dr); return;
    case DyallBlock::A3:
      transposeA3(m, need(o.dm1), need(o.dm2), need(o.hdm1), need(o.delta), out, dr);
      return;
    case DyallBlock::K27: transposeK27(m, need(o.dm1), need(o.dm2), out, dr); return;
    case DyallBlock::A7: transposeA7(m, need(o.rm2), need(o.rm3), out, dr); return;
    case DyallBlock::A9: transposeA9(m, need(o.hdm2), need(o.hdm3), out, dr); return;
    case DyallBlock::A12: transposeA12(m, need(o.dm2), need(o.dm3), out, dr); return;
    case DyallBlock::A13:
      transposeA13(m, need(o.dm1), need(o.dm2), need(o.dm3), need(o.delta), out, dr);
      return;
  }
}

// Block `b`'s FORWARD builder at integrals (h1e, h2e), integral-free addends
// zeroed: make_* itself, as a rotation's tangent calls it -- what the duality
// check holds each transpose to.
DeviceTensor dyallBlockForward(DyallBlock b, const DeviceTensor& h1e, const DeviceTensor& h2e,
                               const DyallOperands& o, const DeviceResources& dr) {
  const int64_t n = h1e.dims[0];
  const auto zero6 = [&] { return DeviceTensor::zeros({n, n, n, n, n, n}, dr); };
  switch (b) {
    case DyallBlock::A16: {
      const DeviceTensor z1 = zero6(), z2 = zero6();
      return make_a16(h1e, h2e, *o.dm3, z1, z2, n, dr);
    }
    case DyallBlock::A17: return make_a17(h1e, h2e, *o.dm2, *o.dm3, dr);
    case DyallBlock::A19: return make_a19(h1e, h2e, *o.dm1, *o.dm2, dr);
    case DyallBlock::A22: {
      const DeviceTensor z1 = zero6(), z2 = zero6();
      return make_a22(h1e, h2e, *o.dm2, *o.dm3, z1, z2, n, dr);
    }
    case DyallBlock::A23: return make_a23(h1e, h2e, *o.dm1, *o.dm2, *o.dm3, dr);
    case DyallBlock::A25: return make_a25(h1e, h2e, *o.dm1, *o.dm2, dr);
    case DyallBlock::A3: return make_a3(h1e, h2e, *o.dm1, *o.dm2, *o.hdm1, dr);
    case DyallBlock::K27: return make_k27(h1e, h2e, *o.dm1, *o.dm2, dr);
    case DyallBlock::A7: return make_a7(h1e, h2e, *o.dm1, *o.dm2, *o.dm3, dr).second;
    case DyallBlock::A9: return make_a9(h1e, h2e, *o.hdm1, *o.hdm2, *o.hdm3, dr);
    case DyallBlock::A12: return make_a12(h1e, h2e, *o.dm1, *o.dm2, *o.dm3, dr);
    case DyallBlock::A13: return make_a13(h1e, h2e, *o.dm1, *o.dm2, *o.dm3, dr);
  }
  return DeviceTensor();
}

// Each SC class's integral-reading Dyall blocks, CLASSES order (Sijrs has
// none): the gdot lists of gradient_sc_classes.cppm, by name.
std::vector<DyallBlock> scClassDyallBlocks(int cls) {
  switch (cls) {
    case 0: return {DyallBlock::A16, DyallBlock::A17, DyallBlock::A19};
    case 1: return {DyallBlock::A22, DyallBlock::A23, DyallBlock::A25};
    case 3: return {DyallBlock::A3};
    case 4: return {DyallBlock::K27};
    case 5: return {DyallBlock::A7};
    case 6: return {DyallBlock::A9};
    case 7: return {DyallBlock::A12, DyallBlock::A13};
    default: return {};
  }
}

// A class's sweep onto the integrals: each (block tensor, name) pair whose
// accumulator the walk filled, transposed into `out`. `blocks` names the
// class's own block tensors, which key its accumulators.
void scClassIntegralSweep(const std::vector<ScActiveAccum>& acts,
                          const std::vector<std::pair<const DeviceTensor*, DyallBlock>>& blocks,
                          const DyallOperands& o, const ScIntegralBars& out,
                          const DeviceResources& dr) {
  for (const auto& [g, b] : blocks)
    if (const DeviceTensor* m = activeAccumOf(acts, g); m != nullptr)
      dyallBlockTranspose(b, *m, o, out, dr);
}

// PC's K, transposed: class `cls`'s K (pcClassMatrices' K, its natural shape)
// is one of SC's blocks reshaped -- or, for Sir, the (x, p, q) (+) (y, a, b)
// pattern of a12 and a13 -- so `gK` = dE/dK is read the way pcWalk's sweep
// reads it for the RDM side, and pushed through the same block transposes.
void pcKIntegralTranspose(int cls, const DeviceTensor& gK, const DyallOperands& o,
                          const ScIntegralBars& out, const DeviceResources& dr) {
  const int64_t n = o.dm1->dims[0];
  const wwrStream_t s = dr.stream();
  switch (cls) {
    case 0:
      dyallBlockTranspose(DyallBlock::A16, reshapeView(gK, {n, n, n, n, n, n}), o, out, dr);
      return;
    case 1:
      dyallBlockTranspose(DyallBlock::A22, reshapeView(gK, {n, n, n, n, n, n}), o, out, dr);
      return;
    case 3: dyallBlockTranspose(DyallBlock::A3, reshapeView(gK, {n, n}), o, out, dr); return;
    case 4: dyallBlockTranspose(DyallBlock::K27, reshapeView(gK, {n, n}), o, out, dr); return;
    case 5:
      dyallBlockTranspose(DyallBlock::A7, reshapeView(gK, {n, n, n, n}), o, out, dr);
      return;
    case 6:
      dyallBlockTranspose(DyallBlock::A9, reshapeView(gK, {n, n, n, n}), o, out, dr);
      return;
    case 7: {
      // K = offDiag (x) a12 + block22 (x) a13 (pcMatricesSir), as pcWalk's
      // Sir sweep splits it.
      const DeviceTensor offDiag =
          uploadTensor(Tensor::fromFlat({2, 2}, {2.0, -1.0, -1.0, 0.0}), dr);
      const DeviceTensor block22 =
          uploadTensor(Tensor::fromFlat({2, 2}, {0.0, 0.0, 0.0, 1.0}), dr);
      const DeviceTensor gK6 = reshapeView(gK, {2, n, n, 2, n, n});
      const DeviceTensor mA12 = DeviceTensor::zeros({n, n, n, n}, dr);
      const DeviceTensor mA13 = DeviceTensor::zeros({n, n, n, n}, dr);
      adjointTerm("xy,pqab->xpqyab", 1, {&offDiag}, gK6, mA12, 1.0, s);
      adjointTerm("xy,pqab->xpqyab", 1, {&block22}, gK6, mA13, 1.0, s);
      dyallBlockTranspose(DyallBlock::A12, mA12, o, out, dr);
      dyallBlockTranspose(DyallBlock::A13, mA13, o, out, dr);
      return;
    }
    default:
      check(false, std::format("pcKIntegralTranspose: class {} has no PC K (Sijrs is SC's)", cls));
  }
}

// The hole-side operands every block above may read, built from the RDMs --
// what the duality check and PC's walk hand dyallBlockTranspose. Owning; the
// RDMs are borrowed and must outlive it. `operands()` points into it, so it is
// taken where the set lives, never before a move.
struct DyallOperandSet {
  const DeviceTensor* dm1 = nullptr;
  const DeviceTensor* dm2 = nullptr;
  const DeviceTensor* dm3 = nullptr;
  DeviceTensor delta, hdm1, hdm2, hdm3, rm2, rm3;

  DyallOperands operands() const {
    return DyallOperands{dm1, dm2, dm3, &hdm1, &hdm2, &hdm3, &rm2, &rm3, &delta};
  }
};

DyallOperandSet dyallOperandSet(const DeviceTensor& dm1, const DeviceTensor& dm2,
                                const DeviceTensor& dm3, const DeviceResources& dr) {
  const int64_t n = dm1.dims[0];
  DyallOperandSet s;
  s.dm1 = &dm1;
  s.dm2 = &dm2;
  s.dm3 = &dm3;
  s.delta = deviceEye(n, dr);
  s.hdm1 = make_hdm1(dm1, dr);
  s.hdm2 = make_hdm2(dm1, dm2, dr);
  s.hdm3 = make_hdm3(dm1, dm2, dm3, s.hdm1, s.hdm2, dr);
  // make_a7's own rm2 (its first return value; it reads no integral), and rm3.
  const DeviceTensor z1 = DeviceTensor::zeros({n, n}, dr);
  const DeviceTensor z2 = DeviceTensor::zeros({n, n, n, n}, dr);
  s.rm2 = make_a7(z1, z2, dm1, dm2, dm3, dr).first;
  s.rm3 = make_rm3(dm2, dm3, s.rm2, dr);
  return s;
}

}  // namespace nevpt2
