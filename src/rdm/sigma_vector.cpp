// The body of nevpt2.sigma_vector -- see the interface for what a sigma
// vector is, the two conventions, and why the 4-body one factors through an
// intermediate. Everything here is host-side glue: the reverse link tables,
// the coefficient repacking each kernel reads, the tile plan, and the
// launches.
module;

// The vendor headers sigma_vector_bridge.h pulls in, textually in the global
// module fragment; the bridge itself is included after the imports (see
// rdm/rdm_accumulate_bridge.h).
#include <runtime.h>

module nevpt2.sigma_vector;

import std;

import nevpt2.common;          // narrowTo, int64_t
import nevpt2.error_handling;  // check: a wrong shape here is our bug
import nevpt2.link_tables;     // the single-excitation tables and their reverse
import nevpt2.rdm_plan;        // tileWidthForCount, planTiles
import nevpt2.wwr;             // gpuCheck, and the wwr* runtime names bare

// The linked-in sigma kernels' launchers (sigma_vector.cu): extern "C++"
// attaches them to the global module, so they keep external linkage and bind
// to the .cu's definitions (see rdm/rdm_accumulate_bridge.h).
extern "C++" {
#include "rdm/sigma_vector_bridge.h"
}

namespace nevpt2 {

// Not exported: the host-to-device staging these functions need for the
// link tables, which are int, not double -- uploadTensor's job for everything
// else. On the caller's one stream, like every other call here; `loc` is the
// caller's line, so an out-of-memory abort names the upload that asked.
template <class T>
DeviceBuffer<T> uploadSigmaInput(const std::vector<T>& h, const DeviceResources& res,
                                 std::source_location loc = std::source_location::current()) {
  DeviceBuffer<T> d(h.size(), res.shared_from_this(), loc);
  if (!h.empty()) {
    gpuCheck(wwrMemcpyAsync(d.data(), h.data(), h.size() * sizeof(T), wwrMemcpyHostToDevice,
                            res.stream()));
  }
  return d;
}

// Not exported: one active space's sizes, with every value the launchers take
// narrowed to its int exactly once, here, and checked (narrowTo aborts, the
// abort tier), plus the tile plan nevpt2.rdm_plan hands the RDM build.
struct SigmaShape {
  int64_t norb, na, nb, ndet, nla, nlb, n2, n4, n6, nK;
  int norbK, nbK, nlaK, nlbK, ndetK;
  std::vector<std::pair<int, int>> tiles;
};

SigmaShape sigmaShape(const int64_t norb, const int64_t nelecA, const int64_t nelecB,
                      const int64_t nTiles) {
  check(nTiles >= 1, "SigmaVectorOptions::nTiles must be at least 1");
  check(norb >= 1, "sigma vector: norb must be at least 1");
  SigmaShape sh{};
  sh.norb = norb;
  sh.na = link_tables::num_strings(norb, nelecA);
  sh.nb = link_tables::num_strings(norb, nelecB);
  check(sh.na > 0 && sh.nb > 0,
        std::format("sigma vector: CAS({},{}) has no determinants", nelecA + nelecB, norb));
  sh.ndet = sh.na * sh.nb;
  sh.nla = link_tables::nlink(norb, nelecA);
  sh.nlb = link_tables::nlink(norb, nelecB);
  sh.n2 = norb * norb;
  sh.n4 = sh.n2 * sh.n2;
  sh.n6 = sh.n4 * sh.n2;
  sh.nK = tileWidthForCount(sh.ndet, nTiles);
  sh.norbK = narrowTo<int>(norb);
  sh.nbK = narrowTo<int>(sh.nb);
  sh.nlaK = narrowTo<int>(sh.nla);
  sh.nlbK = narrowTo<int>(sh.nlb);
  sh.ndetK = narrowTo<int>(sh.ndet, "a determinant space past the kernels' 32-bit indexing");
  sh.tiles = planTiles(sh.ndet, sh.nK);
  // The 4-body stage 1 enumerates (n^2 batches, tile width) in ONE int index,
  // as kernels.cu's consume enumerates (n^2, ndet). Checked rather than
  // narrowed, since nothing here needs the value; more tiles shrink it.
  check(sh.n2 * sh.nK <= std::numeric_limits<int>::max(),
        std::format("sigma vector: n^2 * tile width = {} is past int -- use more tiles",
                    sh.n2 * sh.nK));
  return sh;
}

// Not exported: `ci`'s shape and a coefficient tensor's, through the abort
// tier -- the caller derives both from the (norb, nelec) it passes in.
void checkSigmaShapes(const Tensor& ci, const SigmaShape& sh, const Tensor& coef,
                      const std::string_view what,
                      const std::source_location loc = std::source_location::current()) {
  check(ci.rank() == 2 && ci.dim(0) == sh.na && ci.dim(1) == sh.nb,
        std::format("sigma vector: ci must be the ({}, {}) CI vector", sh.na, sh.nb), loc);
  check(coef.rank() == 6 && coef.size() == sh.n6,
        std::format("sigma vector: {} must be an (norb,)*6 = ({},)*6 tensor", what, sh.norb), loc);
}

// Not exported: the reverse single-excitation tables of both spins -- the only
// tables a sigma walk reads. link_tables builds the forward ones and inverts
// them per target, exactly as the RDM build's produce does for its R.
std::pair<std::vector<int>, std::vector<int>> reverseTables(const SigmaShape& sh,
                                                            const int64_t nelecA,
                                                            const int64_t nelecB) {
  const std::vector<int> fa = link_tables::gen_linkstr_index(sh.norb, nelecA);
  const std::vector<int> fb = link_tables::gen_linkstr_index(sh.norb, nelecB);
  return {link_tables::reverse_link(fa, sh.na, sh.nla),
          link_tables::reverse_link(fb, sh.nb, sh.nlb)};
}

// Not exported: stage 1's coefficients, c1[(a, s), P, Q, R, S], which carry
// the eri contraction of the 4-body digest's inner half. The middle operator
// is E_sx (ca) or E_xs (ac), so (P, Q) is (s, x) or (x, s); the innermost is
// E_qp, so (R, S) = (q, p):
//
//   c1[(a, s), P, Q, R, S] = eri[a, Q or P, R, S] where the other of P/Q is s
//
// and zero everywhere else. n^6 doubles -- d3's own size -- and mostly zero,
// which costs nothing beyond the storage: the kernel's walk enumerates index
// pairs and reads the coefficient at each, rather than looping coefficients.
std::vector<double> stageOneCoefficients(const int order, const Tensor& eri,
                                         const SigmaShape& sh) {
  const int64_t n = sh.norb;
  std::vector<double> c(narrowTo<std::size_t>(sh.n6), 0.0);
  double* const out = c.data();
  for (int64_t a = 0; a < n; ++a)
    for (int64_t s = 0; s < n; ++s) {
      const int64_t base = (a * n + s) * sh.n4;
      for (int64_t x = 0; x < n; ++x) {
        const int64_t mid = order == 0 ? s * n + x : x * n + s;
        for (int64_t q = 0; q < n; ++q)
          for (int64_t p = 0; p < n; ++p)
            out[base + mid * sh.n2 + q * n + p] = eri(a, x, q, p);
      }
    }
  return c;
}

// Not exported: stage 2's coefficients, c2[(a, s), w, v, u, t] = the d6 row
// (w,v,u,t) at that order's placement of the last two axes -- (s, a) for ca,
// (a, s) for ac. A pure repacking of d6: same n^6 values, reindexed so the
// batch (a, s) stage 1 produced leads.
std::vector<double> stageTwoCoefficients(const int order, const Tensor& d6,
                                         const SigmaShape& sh) {
  const int64_t n = sh.norb;
  std::vector<double> c(narrowTo<std::size_t>(sh.n6), 0.0);
  double* const out = c.data();
  for (int64_t a = 0; a < n; ++a)
    for (int64_t s = 0; s < n; ++s) {
      const int64_t base = (a * n + s) * sh.n4;
      for (int64_t w = 0; w < n; ++w)
        for (int64_t v = 0; v < n; ++v)
          for (int64_t u = 0; u < n; ++u)
            for (int64_t t = 0; t < n; ++t)
              out[base + (w * n + v) * sh.n2 + u * n + t] =
                  order == 0 ? d6(w, v, u, t, s, a) : d6(w, v, u, t, a, s);
    }
  return c;
}

Tensor sigmaVector3Device(const SigmaVectorOptions& opt, const Tensor& ci, const int64_t norb,
                          const int64_t nelecA, const int64_t nelecB, const Tensor& d3,
                          const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  const SigmaShape sh = sigmaShape(norb, nelecA, nelecB, opt.nTiles);
  checkSigmaShapes(ci, sh, d3, "d3");
  const auto [ra, rb] = reverseTables(sh, nelecA, nelecB);

  // Every buffer below is freed on `stream` when it goes out of scope, behind
  // the launches that read it -- no manual free, no sync before one.
  DeviceBuffer<int> dRa = uploadSigmaInput(ra, res);
  DeviceBuffer<int> dRb = uploadSigmaInput(rb, res);
  DeviceTensor dCi = uploadTensor(ci, res);
  DeviceTensor dD3 = uploadTensor(d3, res);
  // Zero-filled by its allocation, which it does not need: every determinant
  // is written by exactly one thread of exactly one tile.
  DeviceTensor dSigma = DeviceTensor::zeros({sh.na, sh.nb}, res);

  for (const auto [k0, width] : sh.tiles) {
    gpuCheck(device::sigmaVector3(stream, dCi.data(), dRa.data(), dRb.data(), dD3.data(),
                                  dSigma.data(), sh.norbK, sh.nbK, sh.nlaK, sh.nlbK, k0, width));
  }
  // downloadTensor synchronizes the stream itself, so it needs no sync of its
  // own in front of it and the frees above may queue behind it.
  return downloadTensor(dSigma, stream);
}

Tensor sigmaVector4Device(const SigmaVectorOptions& opt, const int order, const Tensor& ci,
                          const int64_t norb, const int64_t nelecA, const int64_t nelecB,
                          const Tensor& d6, const Tensor& eriF3, const DeviceResources& res) {
  const wwrStream_t stream = res.stream();
  check(order == 0 || order == 1, "sigma vector: `order` is the f3 order, 0 = ca or 1 = ac");
  const SigmaShape sh = sigmaShape(norb, nelecA, nelecB, opt.nTiles);
  checkSigmaShapes(ci, sh, d6, "d6");
  check(eriF3.rank() == 4 && eriF3.size() == sh.n4,
        std::format("sigma vector: eriF3 must be an (norb,)*4 = ({},)*4 tensor", norb));
  const auto [ra, rb] = reverseTables(sh, nelecA, nelecB);

  DeviceBuffer<int> dRa = uploadSigmaInput(ra, res);
  DeviceBuffer<int> dRb = uploadSigmaInput(rb, res);
  DeviceTensor dCi = uploadTensor(ci, res);
  // The two n^6 coefficient blocks, repacked on the host so each kernel reads
  // one contiguous (n^2, n^4) array. Pageable host memory, which the runtime
  // stages before the upload returns, so the temporaries may go right away.
  DeviceBuffer<double> dC1 = uploadSigmaInput(stageOneCoefficients(order, eriF3, sh), res);
  DeviceBuffer<double> dC2 = uploadSigmaInput(stageTwoCoefficients(order, d6, sh), res);
  // The one temporary: stage 1's n^2 intermediate vectors, each spanning the
  // WHOLE determinant axis (stage 2 gathers from anywhere in it, so it cannot
  // be a per-tile block). Overwritten element by element, like dSigma.
  DeviceTensor dMid = DeviceTensor::zeros({sh.n2, sh.ndet}, res);
  DeviceTensor dSigma = DeviceTensor::zeros({sh.na, sh.nb}, res);

  for (const auto [k0, width] : sh.tiles) {
    gpuCheck(device::sigmaVector2Batched(stream, dCi.data(), dRa.data(), dRb.data(), dC1.data(),
                                         dMid.data(), sh.norbK, sh.nbK, sh.nlaK, sh.nlbK, sh.ndetK,
                                         k0, width));
  }
  // Stage 2 reads dMid at determinants anywhere in the space, so it must not
  // start before stage 1 has covered ALL of them. The one stream is what
  // guarantees that: kernels queued on it execute in issue order, so no event
  // and no synchronization is needed between the two sweeps.
  for (const auto [k0, width] : sh.tiles) {
    gpuCheck(device::sigmaVector2Reduce(stream, dMid.data(), dRa.data(), dRb.data(), dC2.data(),
                                        dSigma.data(), sh.norbK, sh.nbK, sh.nlaK, sh.nlbK,
                                        sh.ndetK, k0, width));
  }
  return downloadTensor(dSigma, stream);
}

}  // namespace nevpt2
