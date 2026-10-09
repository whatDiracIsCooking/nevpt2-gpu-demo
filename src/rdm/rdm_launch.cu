// The launchers for the RDM-build kernels: kernels.cu is #included below and
// this file, not kernels.cu, is what the nevpt2.rdm.kernels.device library
// compiles (see CMakeLists.txt).
//
// Why an #include rather than a second source: wwr_add_gpu_device_library
// builds with separable compilation OFF -- a __global__ defined in another TU
// cannot be launched from this one -- and keeping the launchers here leaves
// kernels.cu holding kernels only.
//
// LINKED IN, not loaded as a code object, like rdm_accumulate.cu: rdm_build.cpp
// calls these through rdm_kernels_bridge.h. Every launch keeps the geometry
// it had as a code object, so the build is unchanged by the move:
//  * the index-per-thread kernels: a hand-written <<<grid, 128>>> (the block
//    size rdm_build.cpp's grid1d always used; parallel_for is also 128, but
//    hands the functor its own index type, which these kernels do not take);
//  * the dm3 digest: <<<dim3(gx, gy), dim3(16, 16)>>>, the old gemmGrid.
//    The 16x16 tile (TILE = TK = 16, __shared__ tiles, __syncthreads) is baked
//    into the kernel, so the block shape is not a free choice.
// Both grids come from common/device_index.h's gridFor, which
// replaced this file's grid1d and gemmGrid with the same arithmetic.
// The f3 digest is not here: it reads L2 directly and lives in f3_digest.cu.
// Every launch is on the stream the caller passes -- rdm_build.cpp passes the
// demo's one stream, DeviceResources' non-blocking one (docs/performance.md,
// "One stream").
#include "rdm/rdm_kernels_bridge.h"

#include "common/device_index.h"  // gridFor, int64_t
#include "rdm/kernels.cu"         // the kernels: same TU, see above

namespace nevpt2::device {

namespace {

constexpr int kBlock = 128;
constexpr int kGemmTile = 16;

int64_t sq(const int64_t norb) { return norb * norb; }

}  // namespace

wwrError_t rdmProduce(const wwrStream_t stream, const double* ci, const int* flinkA,
                      const int* flinkB, const int* rlinkA, const int* rlinkB, double* r,
                      double* l2, const int norb, const int na, const int nb, const int nla,
                      const int nlb, const int k0, const int width) {
  produce_generic<<<gridFor(width, kBlock), kBlock, 0, stream>>>(
      ci, flinkA, flinkB, rlinkA, rlinkB, r, l2, norb, na, nb, nla, nlb, k0, width);
  return wwrGetLastError();
}

wwrError_t rdmDigestDm3(const wwrStream_t stream, const double* r, const double* l2, double* dm3,
                        const int norb, const int width) {
  const int64_t n2 = sq(norb);
  // The (n2*n2 x n2) output: x over the n2 columns, y over the n2*n2 rows,
  // one 16x16 tile per block.
  const dim3 block(kGemmTile, kGemmTile);
  digest_dm3_generic<<<gridFor(n2, n2 * n2, block), block, 0, stream>>>(r, l2, dm3, norb, width);
  return wwrGetLastError();
}

wwrError_t rdmConsume(const wwrStream_t stream, const int order, const double* eri,
                      const double* l2, double* w, const int norb, const int width) {
  const unsigned grid = gridFor(sq(norb) * width, kBlock);
  if (order == 0)
    consume_ca_generic<<<grid, kBlock, 0, stream>>>(eri, l2, w, norb, width);
  else
    consume_ac_generic<<<grid, kBlock, 0, stream>>>(eri, l2, w, norb, width);
  return wwrGetLastError();
}

wwrError_t rdmFdm2(const wwrStream_t stream, const int order, const double* eri, const double* dm3,
                   double* fdm2, const int norb) {
  const unsigned grid = gridFor(sq(norb) * sq(norb), kBlock);
  if (order == 0)
    fdm2_ca_generic<<<grid, kBlock, 0, stream>>>(eri, dm3, fdm2, norb);
  else
    fdm2_ac_generic<<<grid, kBlock, 0, stream>>>(eri, dm3, fdm2, norb);
  return wwrGetLastError();
}

wwrError_t rdmWedge(const wwrStream_t stream, const int order, const double* fdm2, double* f3,
                    const int norb) {
  const unsigned grid = gridFor(sq(norb), kBlock);
  if (order == 0)
    wedge_ca_generic<<<grid, kBlock, 0, stream>>>(fdm2, f3, norb);
  else
    wedge_ac_generic<<<grid, kBlock, 0, stream>>>(fdm2, f3, norb);
  return wwrGetLastError();
}

}  // namespace nevpt2::device
