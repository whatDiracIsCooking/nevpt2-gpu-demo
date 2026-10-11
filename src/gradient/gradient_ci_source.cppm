// nevpt2.gradient:ci_source -- Park Eq. 48's assembly: the five RDM
// coefficient tensors nevpt2.gradient:rdm_adjoint accumulates, turned into
//
//   y_I = dE/dc_I = <I| (O_D + O_D^dagger) |Psi>
//
// by nevpt2.sigma_vector's kernels. An internal partition: nothing here is
// exported, so it is reachable from the units that `import :ci_source;` and
// from no importer of nevpt2.gradient. (Used by gradient.cpp.)
//
// THE TWO SIDES. Every RDM the energy reads is a BILINEAR form in the CI
// vector, rho = <c|A|c>, so d rho / d c_I = ((A + A^T) c)[I] -- the operator
// and its adjoint, which is what Park's O_D + O_D^dagger is. For the 3-body
// sigma the adjoint half is the SAME kernel with the coefficient tensor's
// index order reversed (the operator string read backwards), so `symmetrized`
// below hands it one tensor and calls it once; for the 4-body one it is
// nevpt2.sigma_vector's own sigmaVector4DaggerDevice, which says why it has
// to be a second entry point.
//
// AND dm1 AND dm2 NEED NO SIGMA OF THEIR OWN. The determinant basis has a
// fixed particle number, so sum_t E_tt acts as the scalar N_elec on any
// vector in it, and therefore -- IDENTICALLY IN c, not just at the state --
//
//   sum_t dm3[p,q,r,s,t,t] = N_elec dm2[p,q,r,s],
//   sum_r dm2[p,q,r,r]     = N_elec dm1[p,q].
//
// Differentiating those gives the same relations between the three RDMs'
// derivatives, so a dm2 coefficient is a dm3 coefficient on the (t == u)
// diagonal over N_elec, and a dm1 coefficient is one on (r == s, t == u) over
// N_elec^2 (`folded` below). One 3-body sigma then carries all three, and
// nevpt2.sigma_vector needs no 1- or 2-body entry point.
//
// COST. Five sigma calls per y: one 3-body, and a forward and a dagger for
// each of the two f3 orders. Everything else here is host arithmetic over
// n^6, which is nothing beside them.
module nevpt2.gradient:ci_source;

import std;
import nevpt2.gradient;

namespace nevpt2 {

// The host half of the assembly -- ciSourceFolded (the fold derived above),
// ciSourceSymmetrized (d3 plus its reversed operator string) and
// ciSourceAddInto -- is pure host Tensor arithmetic and lives in
// nevpt2.gradient_plan, where the unit tier can reach it card-free.

// y_I over the whole determinant space, from the five coefficient tensors and
// the state they are differentiated at. `h2e` is the PHYSICIST-ordered active
// two-electron block the golden carries; the f3 digests are built against its
// chemists'-ordered transpose, so that is what the 4-body sigmas are handed
// (the same `transpose(h2e, {0, 2, 1, 3})` nevpt2.rdm_build derives).
//
// Every call below allocates from res's pool and launches on res's stream,
// and each returns through downloadTensor, which synchronizes that stream
// itself.
Tensor ciSourceVector(const SigmaVectorOptions& opt, const Tensor& ci, const int64_t norb,
                      const int64_t nelecA, const int64_t nelecB, const Tensor& g1,
                      const Tensor& g2, const Tensor& g3, const Tensor& gac, const Tensor& gca,
                      const Tensor& h2e, const DeviceResources& res) {
  const Tensor eriF3 = transpose(h2e, {0, 2, 1, 3});
  const Tensor d3 = ciSourceSymmetrized(ciSourceFolded(g1, g2, g3, nelecA + nelecB));
  Tensor y = sigmaVector3Device(opt, ci, norb, nelecA, nelecB, d3, res);
  // The two f3 orders, each both ways round. 0 = ca, 1 = ac, the RDM build's
  // own numbering.
  ciSourceAddInto(sigmaVector4Device(opt, 0, ci, norb, nelecA, nelecB, gca, eriF3, res), y);
  ciSourceAddInto(sigmaVector4DaggerDevice(opt, 0, ci, norb, nelecA, nelecB, gca, eriF3, res),
                  y);
  ciSourceAddInto(sigmaVector4Device(opt, 1, ci, norb, nelecA, nelecB, gac, eriF3, res), y);
  ciSourceAddInto(sigmaVector4DaggerDevice(opt, 1, ci, norb, nelecA, nelecB, gac, eriF3, res),
                  y);
  return y;
}

}  // namespace nevpt2
