// nevpt2.energy:pc_classes -- seven of the eight partially-contracted class
// energies (Sijrs is SC's, from nevpt2.energy:shared).
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :pc_classes;` and from no importer of nevpt2.energy.
// (Used by energy_pc.cpp.)
module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.energy:pc_classes;

import std;
import nevpt2.energy;
import :shared;
import :pc_solve;

namespace nevpt2 {

// Sr: basis (p, q, r) (d = n^3), S[pqr, abc] = dm3[r,p,q,b,a,c], K = a16,
// S_1[pqr, a] = dm2[r,p,q,a] against h1e_v_Sr, Delta = e_v[r], factor 1
// (docs/pc-nevpt2.md, "Per class"). The n^3 x n^3 matrices are 24 MB each at n = 12,
// built in a scope so S/K/a16 are freed before the slab walk.
Result<PcClassResult> pc_Sr(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                            const Tensor& ev, const DeviceTensor& dm2, const DeviceTensor& dm3,
                            const DeviceTensor& f3ac, const DeviceTensor& f3ca,
                            const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = dm2.dims[0], d = n * n * n;
  ClassSolve cs;
  {
    DeviceTensor a16 = make_a16(ai.h1e, ai.h2e, dm3, f3ac, f3ca, n, dr);
    DeviceTensor S = deviceEinsumNew("rpqbac->pqrabc", {&dm3}, 1.0, dr);
    DeviceTensor S1 = deviceEinsumNew("rpqa->pqra", {&dm2}, 1.0, dr);
    const DeviceTensor S1m = reshapeView(S1, {d, n});
    cs = NEVPT2_TRY(solveClass(reshapeView(S, {d, d}), reshapeView(a16, {d, d}), "Sr", dr, &S1m));
  }
  if (cs.result.status != PcStatus::Done) return cs.result;
  const int64_t m = std::ssize(cs.lambda);
  const DeviceTensor T = reshapeView(cs.T, {n, n, n, m});

  double e = 0.0, minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Sr), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sr, b0, b1, dr);  // h[r_batch, p, q, r]
    DeviceTensor h1 = sliceAxis(ai.h1e_v_Sr, 0, b0, b1);     // (nr, a)
    DeviceTensor Y = deviceEinsumNew("ipqr,pqrk->ik", {&slab, &T}, 1.0, dr);
    deviceEinsumAccum("ia,ak->ik", {&h1, &cs.t1}, Y, 1.0, s);
    const Tensor y = downloadTensor(Y, s);  // (nr, m)
    std::vector<double> delta(ev.data().begin() + b0, ev.data().begin() + b1);
    finishSingle(y, cs.lambda, delta, e, minDen);
  });
  cs.result.energy = e;
  checkDenominator(cs.result, minDen);
  return cs.result;
}

// Si: basis (p, q, r) (d = n^3), S[pqr, abc] = dm3_h[r,p,q,b,a,c], K = a22,
// S_1[pqr, a] = dm2_h[r,p,q,a] against h1e_v_Si, Delta = -e_c[i], factor 1.
// dm3_h / dm2_h are energy_Si's own (make_dm3_h / make_dm2_h).
Result<PcClassResult> pc_Si(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                            const Tensor& ec, const DeviceTensor& dm1, const DeviceTensor& dm2,
                            const DeviceTensor& dm3, const DeviceTensor& f3ac,
                            const DeviceTensor& f3ca, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = dm2.dims[0], d = n * n * n;
  ClassSolve cs;
  {
    DeviceTensor a22 = make_a22(ai.h1e, ai.h2e, dm2, dm3, f3ac, f3ca, n, dr);
    DeviceTensor S;
    {
      DeviceTensor dm3_h = make_dm3_h(dm2, dm3, dr);
      S = deviceEinsumNew("rpqbac->pqrabc", {&dm3_h}, 1.0, dr);
    }
    DeviceTensor S1;
    {
      DeviceTensor dm2_h = make_dm2_h(dm1, dm2, dr);
      S1 = deviceEinsumNew("rpqa->pqra", {&dm2_h}, 1.0, dr);
    }
    const DeviceTensor S1m = reshapeView(S1, {d, n});
    cs = NEVPT2_TRY(solveClass(reshapeView(S, {d, d}), reshapeView(a22, {d, d}), "Si", dr, &S1m));
  }
  if (cs.result.status != PcStatus::Done) return cs.result;
  const int64_t m = std::ssize(cs.lambda);
  const DeviceTensor T = reshapeView(cs.T, {n, n, n, m});

  double e = 0.0, minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Si), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Si, b0, b1, dr);  // h[q, p, i_batch, r]
    DeviceTensor h1 = sliceAxis(ai.h1e_v_Si, 1, b0, b1);     // (a, ni)
    DeviceTensor Y = deviceEinsumNew("qpir,pqrk->ik", {&slab, &T}, 1.0, dr);
    deviceEinsumAccum("ai,ak->ik", {&h1, &cs.t1}, Y, 1.0, s);
    const Tensor y = downloadTensor(Y, s);  // (ni, m)
    std::vector<double> delta(ec.data().begin() + b0, ec.data().begin() + b1);
    for (double& x : delta) x = -x;
    finishSingle(y, cs.lambda, delta, e, minDen);
  });
  cs.result.energy = e;
  checkDenominator(cs.result, minDen);
  return cs.result;
}

// Sijr: basis p (d = n), S = hdm1, K = a3, Delta = e_v[r] - e_c[j] - e_c[i].
// The two spin couplings are SC's "2 direct - exchange" coupling matrix, summed
// over ORDERED (j, i): E = -sum (2 Y[r,j,i,k]^2 - Y[r,j,i,k] Y[r,i,j,k]) / den
// (docs/pc-nevpt2.md, "Sijr, Srsi: two spin couplings, no explicit +- split").
Result<PcClassResult> pc_Sijr(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                              const Tensor& ec, const Tensor& ev, const DeviceTensor& dm1,
                              const DeviceTensor& dm2, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor hdm1 = make_hdm1(dm1, dr);
  DeviceTensor a3 = make_a3(ai.h1e, ai.h2e, dm1, dm2, hdm1, dr);
  ClassSolve cs = NEVPT2_TRY(solveClass(hdm1, a3, "Sijr", dr));
  if (cs.result.status != PcStatus::Done) return cs.result;
  const int64_t m = std::ssize(cs.lambda);
  const int64_t nc = std::ssize(ec);

  double e = 0.0, minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Sijr), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sijr, b0, b1, dr);  // (nr, a, c, c)
    DeviceTensor Y = deviceEinsumNew("rpji,pk->rjik", {&slab, &cs.T}, 1.0, dr);
    Tensor y = downloadTensor(Y, s);  // (nr, c, c, m)
    const int64_t nr = b1 - b0;
    for (int64_t r = 0; r < nr; ++r)
      for (int64_t j = 0; j < nc; ++j)
        for (int64_t i = 0; i < nc; ++i) {
          const double delta = ev.flat(b0 + r) - ec.flat(j) - ec.flat(i);
          for (int64_t k = 0; k < m; ++k) {
            const double den = cs.lambda[k] + delta;
            minDen = std::min(minDen, den);
            const double yd = y(r, j, i, k), yx = y(r, i, j, k);
            e -= (2.0 * yd * yd - yd * yx) / den;
          }
        }
  });
  cs.result.energy = e;
  checkDenominator(cs.result, minDen);
  return cs.result;
}

// Srsi: basis p (d = n), S = dm1, K = k27, Delta = e_v[r] + e_v[s] - e_c[i].
// E = -sum (2 Y[r,s,i,k]^2 - Y[r,s,i,k] YT[r,s,i,k]) / den over ordered (r, s),
// YT from the SrsiT slab (h[s, r_batch, i, p]) -- so, unlike SC, no full
// (v, v, c) buffer: each r-batch has both in hand.
Result<PcClassResult> pc_Srsi(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                              const Tensor& ec, const Tensor& ev, const DeviceTensor& dm1,
                              const DeviceTensor& dm2, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  DeviceTensor k27 = make_k27(ai.h1e, ai.h2e, dm1, dm2, dr);
  ClassSolve cs = NEVPT2_TRY(solveClass(dm1, k27, "Srsi", dr));
  if (cs.result.status != PcStatus::Done) return cs.result;
  const int64_t m = std::ssize(cs.lambda);
  const int64_t nc = std::ssize(ec), nv = std::ssize(ev);

  double e = 0.0, minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Srsi), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Srsi, b0, b1, dr);    // h[r_batch, s, i, p]
    DeviceTensor slabT = src.slab(ExtBlock::SrsiT, b0, b1, dr);  // h[s, r_batch, i, p]
    DeviceTensor Y = deviceEinsumNew("rsip,pk->rsik", {&slab, &cs.T}, 1.0, dr);
    DeviceTensor YT = deviceEinsumNew("srip,pk->rsik", {&slabT, &cs.T}, 1.0, dr);
    Tensor y = downloadTensor(Y, s);  // (nr, v, c, m)
    Tensor yt = downloadTensor(YT, s);
    const int64_t nr = b1 - b0;
    for (int64_t r = 0; r < nr; ++r)
      for (int64_t ss = 0; ss < nv; ++ss)
        for (int64_t i = 0; i < nc; ++i) {
          const double delta = ev.flat(b0 + r) + ev.flat(ss) - ec.flat(i);
          for (int64_t k = 0; k < m; ++k) {
            const double den = cs.lambda[k] + delta;
            minDen = std::min(minDen, den);
            const double yd = y(r, ss, i, k), yx = yt(r, ss, i, k);
            e -= (2.0 * yd * yd - yd * yx) / den;
          }
        }
  });
  cs.result.energy = e;
  checkDenominator(cs.result, minDen);
  return cs.result;
}

// Srs: basis (p, q) (d = n^2), S[pq, ab] = rm2[p,q,b,a], K = a7,
// Delta = e_v[r] + e_v[s], factor 1/2 on every ORDERED (r, s), exactly as SC
// walks the slab -- no +- split (docs/pc-nevpt2.md, "Srs, Sij: no +- split either").
Result<PcClassResult> pc_Srs(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                             const Tensor& ev, const DeviceTensor& dm1, const DeviceTensor& dm2,
                             const DeviceTensor& dm3, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = dm1.dims[0], nv = std::ssize(ev);
  ClassSolve cs;
  {
    auto [rm2, a7] = make_a7(ai.h1e, ai.h2e, dm1, dm2, dm3, dr);
    DeviceTensor S = deviceEinsumNew("pqba->pqab", {&rm2}, 1.0, dr);
    cs = NEVPT2_TRY(solveClass(reshapeView(S, {n * n, n * n}), reshapeView(a7, {n * n, n * n}),
                               "Srs", dr));
  }
  if (cs.result.status != PcStatus::Done) return cs.result;
  const int64_t m = std::ssize(cs.lambda);
  const DeviceTensor T = reshapeView(cs.T, {n, n, m});

  double e = 0.0, minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Srs), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Srs, b0, b1, dr);  // h[r_batch, s, q, p]
    DeviceTensor Y = deviceEinsumNew("rsqp,pqk->rsk", {&slab, &T}, 1.0, dr);
    Tensor y = downloadTensor(Y, s);  // (nr, v, m)
    const int64_t nr = b1 - b0;
    for (int64_t r = 0; r < nr; ++r)
      for (int64_t ss = 0; ss < nv; ++ss) {
        const double delta = ev.flat(b0 + r) + ev.flat(ss);
        for (int64_t k = 0; k < m; ++k) {
          const double den = cs.lambda[k] + delta;
          minDen = std::min(minDen, den);
          const double yk = y(r, ss, k);
          e -= 0.5 * yk * yk / den;
        }
      }
  });
  cs.result.energy = e;
  checkDenominator(cs.result, minDen);
  return cs.result;
}

// Sij: basis (p, q) (d = n^2), S = hdm2, K = a9, Delta = -(e_c[i] + e_c[j]),
// factor 1/2 on every ORDERED (i, j), as Srs.
Result<PcClassResult> pc_Sij(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                             const Tensor& ec, const DeviceTensor& dm1, const DeviceTensor& dm2,
                             const DeviceTensor& dm3, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = dm1.dims[0], nc = std::ssize(ec);
  ClassSolve cs;
  {
    DeviceTensor hdm1 = make_hdm1(dm1, dr);
    DeviceTensor hdm2 = make_hdm2(dm1, dm2, dr);
    DeviceTensor hdm3 = make_hdm3(dm1, dm2, dm3, hdm1, hdm2, dr);
    DeviceTensor a9 = make_a9(ai.h1e, ai.h2e, hdm1, hdm2, hdm3, dr);
    cs = NEVPT2_TRY(solveClass(reshapeView(hdm2, {n * n, n * n}),
                               reshapeView(a9, {n * n, n * n}), "Sij", dr));
  }
  if (cs.result.status != PcStatus::Done) return cs.result;
  const int64_t m = std::ssize(cs.lambda);
  const DeviceTensor T = reshapeView(cs.T, {n, n, m});

  double e = 0.0, minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Sij), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab = src.slab(ExtBlock::Sij, b0, b1, dr);  // h[q, p, i_batch, j]
    DeviceTensor Y = deviceEinsumNew("qpij,pqk->ijk", {&slab, &T}, 1.0, dr);
    Tensor y = downloadTensor(Y, s);  // (ni, c, m)
    const int64_t ni = b1 - b0;
    for (int64_t i = 0; i < ni; ++i)
      for (int64_t j = 0; j < nc; ++j) {
        const double delta = -(ec.flat(b0 + i) + ec.flat(j));
        for (int64_t k = 0; k < m; ++k) {
          const double den = cs.lambda[k] + delta;
          minDen = std::min(minDen, den);
          const double yk = y(i, j, k);
          e -= 0.5 * yk * yk / den;
        }
      }
  });
  cs.result.energy = e;
  checkDenominator(cs.result, minDen);
  return cs.result;
}

// Sir: basis (p, q) (+) (p, q) (d = 2 n^2), the two spin couplings as two
// blocks (docs/pc-nevpt2.md, "Sir"), (p, q) row and (a, b) column in each:
//   S11 = 2 dm2[q,p,a,b], S12 = S21 = -dm2[q,p,a,b],
//   S22 = 2 delta_pa dm1[q,b] - dm2[q,b,a,p] + delta_ab dm1[q,p];
//   K11 = 2 a12, K12 = K21 = -a12, K22 = a13 (symmetrised by solveClass);
//   S_1 = [2 dm1[q,p] ; -dm1[q,p]] against h1e_v_Sir.
// Held as (x, p, q, y, a, b) with x, y the block: each term is an outer product
// with a 2x2 block pattern. Delta = e_v[r] - e_c[i], factor 1.
Result<PcClassResult> pc_Sir(const ActiveIntegralsDevice& ai, IntegralSource& src, int64_t batch,
                             const Tensor& ec, const Tensor& ev, const DeviceTensor& dm1,
                             const DeviceTensor& dm2, const DeviceTensor& dm3,
                             const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  const int64_t n = dm1.dims[0], nc = std::ssize(ec);
  const int64_t d = 2 * n * n;
  ClassSolve cs;
  {
    DeviceTensor offDiag = uploadTensor(Tensor::fromFlat({2, 2}, {2.0, -1.0, -1.0, 0.0}), dr);
    DeviceTensor block22 = uploadTensor(Tensor::fromFlat({2, 2}, {0.0, 0.0, 0.0, 1.0}), dr);
    DeviceTensor oneBody = uploadTensor(Tensor::fromFlat({2}, {2.0, -1.0}), dr);
    DeviceTensor delta = deviceEye(n, dr);

    DeviceTensor S22 = deviceEinsumNew("pa,qb->pqab", {&delta, &dm1}, 2.0, dr);
    deviceEinsumAccum("qbap->pqab", {&dm2}, S22, -1.0, s);
    deviceEinsumAccum("ab,qp->pqab", {&delta, &dm1}, S22, 1.0, s);
    DeviceTensor S = deviceEinsumNew("xy,qpab->xpqyab", {&offDiag, &dm2}, 1.0, dr);
    deviceEinsumAccum("xy,pqab->xpqyab", {&block22, &S22}, S, 1.0, s);

    DeviceTensor a12 = make_a12(ai.h1e, ai.h2e, dm1, dm2, dm3, dr);
    DeviceTensor a13 = make_a13(ai.h1e, ai.h2e, dm1, dm2, dm3, dr);
    DeviceTensor K = deviceEinsumNew("xy,pqab->xpqyab", {&offDiag, &a12}, 1.0, dr);
    deviceEinsumAccum("xy,pqab->xpqyab", {&block22, &a13}, K, 1.0, s);

    DeviceTensor S1 = deviceEinsumNew("x,qp->xpq", {&oneBody, &dm1}, 1.0, dr);
    const DeviceTensor S1col = reshapeView(S1, {d, 1});
    cs = NEVPT2_TRY(solveClass(reshapeView(S, {d, d}), reshapeView(K, {d, d}), "Sir", dr, &S1col));
  }
  if (cs.result.status != PcStatus::Done) return cs.result;
  const int64_t m = std::ssize(cs.lambda);
  const DeviceTensor T1 = reshapeView(sliceAxis(cs.T, 0, 0, n * n), {n, n, m});
  const DeviceTensor T2 = reshapeView(sliceAxis(cs.T, 0, n * n, d), {n, n, m});
  const DeviceTensor t1 = reshapeView(cs.t1, {m});

  double e = 0.0, minDen = std::numeric_limits<double>::infinity();
  forBatches(src.extent(ExtBlock::Sir1), batch, [&](int64_t b0, int64_t b1) {
    DeviceTensor slab1 = src.slab(ExtBlock::Sir1, b0, b1, dr);  // h[r_batch, p, i, q]
    DeviceTensor slab2 = src.slab(ExtBlock::Sir2, b0, b1, dr);  // h[r_batch, p, q, i]
    DeviceTensor h1 = sliceAxis(ai.h1e_v_Sir, 0, b0, b1);       // (nr, c)
    DeviceTensor Y = deviceEinsumNew("rpiq,pqk->rik", {&slab1, &T1}, 1.0, dr);
    deviceEinsumAccum("rpqi,pqk->rik", {&slab2, &T2}, Y, 1.0, s);
    deviceEinsumAccum("ri,k->rik", {&h1, &t1}, Y, 1.0, s);
    Tensor y = downloadTensor(Y, s);  // (nr, c, m)
    const int64_t nr = b1 - b0;
    for (int64_t r = 0; r < nr; ++r)
      for (int64_t i = 0; i < nc; ++i) {
        const double delta = ev.flat(b0 + r) - ec.flat(i);
        for (int64_t k = 0; k < m; ++k) {
          const double den = cs.lambda[k] + delta;
          minDen = std::min(minDen, den);
          const double yk = y(r, i, k);
          e -= yk * yk / den;
        }
      }
  });
  cs.result.energy = e;
  checkDenominator(cs.result, minDen);
  return cs.result;
}

}  // namespace nevpt2
