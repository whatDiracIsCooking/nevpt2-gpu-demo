// nevpt2.sigma_vector -- the sigma vectors of the active-space 3-body and
// (integral-contracted) 4-body operators: a coefficient tensor over active
// indices contracted with the excited wavefunctions it multiplies, giving ONE
// AMPLITUDE PER DETERMINANT.
//
//   sigma[K] = sum_{pqrstu} d3[p,q,r,s,t,u] (E_pq E_rs E_tu |ci>)[K]
//
// WHY THIS EXISTS. An analytic CI derivative of an energy built from dm3 and
// the 4-RDM needs d(dm3)/dc and d(dm4)/dc -- objects of n^6 * ndet and
// n^8 * ndet. Park (JCTC 2019, arXiv:1907.10180) reports 658 GB for the
// 4-RDM derivative at CAS(8,8) because he STORES it. Contracted against its
// coefficient tensor on the fly it never exists: what comes out is a vector of
// length ndet, which is what these two functions build. It is the same trick
// f3ac/f3ca already play for the energy, where the 4-RDM is never built
// either -- here the contraction is against a coefficient tensor instead of
// the integrals.
//
// A PLAIN TENSOR, deliberately: the coefficient tensor is a Tensor of the
// right shape and nothing more -- no gradient-specific type -- so these
// functions are testable in isolation (test/sigma_vector/, against an
// explicit host construction) and independent of what later computes the
// coefficient tensors.
//
// THE TWO CONVENTIONS, matched to nevpt2.rdm_build's own tensors so that
// contracting a sigma back against `ci` reproduces them:
//
//   sum_K ci[K] sigmaVector3Device(d3)[K] = sum_{pqrstu} d3 . dm3
//   sum_K ci[K] sigmaVector4Device(d6, order)[K] = sum_{n^6} d6 . f3{ca,ac}
//
// with dm3[p,q,r,s,t,u] = <ci|E_pq E_rs E_tu|ci> (PySCF's make_dm123 order,
// the golden files') and f3 the integral-contracted 4-RDM digest
// docs/implementation.md, "The digests", defines:
//
//   f3ca[w,v,u,t,s,a] = sum_{xqp} eri[a,x,q,p] <ci|E_wv E_ut E_sx E_qp|ci>
//   f3ac[w,v,u,t,a,s] = sum_{xqp} eri[a,x,q,p] <ci|E_wv E_ut E_xs E_qp|ci>
//
// (the two orders differ in the middle operator and in which of the last two
// axes holds the free index `s`), where `eri` is the CHEMISTS'-ordered active
// block -- the eriF3 = transpose(h2e, {0,2,1,3}) buildRdmsDevice derives from
// the golden file's physicist-ordered h2e, and what the `eriF3` argument
// below takes. Both identities are checked in the unit tier against the
// device build's own dm3/f3ca/f3ac.
//
// COST, and why the 4-body one takes two kernel passes. COUNTED, not
// measured: nothing here has been timed at scale yet. A b-body sigma is b
// nested reverse walks over the single-excitation link tables, so the 3-body
// one costs (nla + nlb)^3 terms per determinant and needs no temporary at
// all. Done the same way the 4-body one would cost (nla + nlb)^4 * n -- 1e8
// terms per determinant at CAS(10,10), which is not a computation anyone runs
// -- so it factors through an intermediate instead: for each (a, s) pair, a
// 2-body sigma against `ci` (the eri-contracted inner half), then a 2-body
// sigma of each of those against d6's rows. That is n^2 (nla + nlb)^2 terms
// per determinant per pass, a few hundred times fewer at CAS(10,10), at the
// cost of one (n^2, ndet) temporary.
//
// Every allocation, upload, launch and copy is issued on res.stream(), the
// caller's one non-blocking stream (docs/performance.md, "One stream"); the
// kernels are src/rdm/sigma_vector.cu, linked in and called through
// sigma_vector_bridge.h. The result is one double per determinant, so it
// comes back on the host.
export module nevpt2.sigma_vector;

import std;
// Re-exported: these take host Tensors and the caller's DeviceResources, and
// hand back a host Tensor (device_tensor re-exports nevpt2.tensor and
// nevpt2.device_resources).
export import nevpt2.device_tensor;

export namespace nevpt2 {

struct SigmaVectorOptions {
  // Determinant-axis tile count, the same plan nevpt2.rdm_plan hands the RDM
  // build (tileWidthForCount / planTiles). Here it is a LAUNCH-GRANULARITY
  // knob, not a memory lever: no buffer below is sized by the tile width (the
  // output is one double per determinant and the 4-body intermediate spans
  // the whole determinant axis, since stage 2 gathers from anywhere in it),
  // so 1 is a fine default and splitting only bounds how long a single
  // launch runs. Contrast docs/performance.md, "Memory and `--tiles`".
  int64_t nTiles = 1;
};

// sigma[K] = sum_{pqrstu} d3[p,q,r,s,t,u] (E_pq E_rs E_tu |ci>)[K], as a
// Tensor of `ci`'s shape (nstringsA, nstringsB).
//
// `ci` is the CI vector of CAS(nelecA + nelecB, norb) and `d3` an
// (norb,)*6 tensor in the dm3 layout (see above). A wrong shape is a broken
// invariant, not bad input -- the caller derives both from the very
// (norb, nelec) it passes -- so it aborts through check(), as does an active
// space too large for the kernels' 32-bit determinant indexing (narrowTo).
Tensor sigmaVector3Device(const SigmaVectorOptions& opt, const Tensor& ci, int64_t norb,
                          int64_t nelecA, int64_t nelecB, const Tensor& d3,
                          const DeviceResources& res);

// The 4-body contracted sigma: the one-sided derivative direction of the f3
// digest above, contracted with `d6`,
//
//   sigma[K] = sum_{wvuts,a} d6[...] sum_{xqp} eri[a,x,q,p]
//                  (E_wv E_ut E_sx E_qp |ci>)[K]        (order 0, ca)
//
// and with the middle operator E_xs and d6's last two axes read (a, s) for
// order 1 (ac). `d6` is an (norb,)*6 tensor in that order's f3 layout and
// `eriF3` the (norb,)*4 chemists'-ordered active block; `order` is the RDM
// build's own (0 = ca, 1 = ac). Shapes and `order` are checked through
// check(), like the 3-body one.
Tensor sigmaVector4Device(const SigmaVectorOptions& opt, int order, const Tensor& ci, int64_t norb,
                          int64_t nelecA, int64_t nelecB, const Tensor& d6, const Tensor& eriF3,
                          const DeviceResources& res);

}  // namespace nevpt2
