// A generic, numpy-einsum-alike GPU contraction kernel -- the demo's sole
// einsum, built so the NumPy reference's ~140 einsum calls port as literal
// string transcriptions (einsum/einsum.cpp parses the same subscript
// strings) rather than re-derived index arithmetic per call.
//
// One thread per (output element, contracted-index combination) pair rather
// than one thread per output element with an internal reduction loop: a
// handful of calls in this port have a *small* output (as low as size ~2)
// contracted over a *large* product of summed labels (up to 10^6) -- e.g.
// energy_Sr's `"ipqr,pqrabc,iabc->i"` has output size nvirt (~16) but sums
// over 6 active-space labels. One-thread-per-output there would leave all
// but ~16 of the GPU's cores idle. Splitting the contracted product across
// threads too keeps occupancy high regardless of which factor (output size
// or contracted size) dominates for a given call, at the cost of an
// atomicAdd per term instead of a per-thread serial accumulate -- reassociates
// the sum (same non-associativity the RDM digest kernels' serial Sum_K
// already has vs. NumPy's einsum), which is why this port compares against
// the golden reference to a tolerance (1e-7), not bit-for-bit.
//
// LINKED IN, not loaded as a code object: a STATIC device library
// (add_device_library over wwr_add_gpu_device_library, see CMakeLists.txt) whose launchers einsum.cpp
// and energy.cpp call through device_einsum_bridge.h, like
// rdm/rdm_accumulate.cu. Every launcher here is a hand-written
// <<<grid, 256, 0, stream>>> rather than WarpWraps' parallel_for (128 threads
// per block): 256 is the block size these kernels ran at as a code object, so
// the atomicAdd order -- and so the low bits of every class energy -- is
// unchanged by the move, and energy_sijrs NEEDS exactly kSijrsBlock.
#include "einsum/device_einsum_bridge.h"

#include <algorithm>  // std::clamp

#include "common/block_reduce.cuh"   // block_reduce, AddOp (energy_sijrs)
#include "common/device_index.h"     // idx4, idx6, gridFor, globalThreadIndex, int64_t

namespace nevpt2::device {

namespace {

// The block size of every launch in this file (see above); the grids come
// from common/device_index.h's gridFor.
constexpr unsigned kBlock = 256;

__global__ void einsum_generic(
    const double* __restrict__ opA, const double* __restrict__ opB,
    const double* __restrict__ opC, double* __restrict__ out, EinsumPlan plan,
    double scale) {
  const int64_t total = plan.outSize * plan.contractedSize;
  const int64_t tid = globalThreadIndex();
  if (tid >= total) return;

  const int64_t outIdx = tid / plan.contractedSize;
  const int64_t cIdx = tid % plan.contractedSize;

  int64_t assign[nevpt2::kEinsumMaxLabels];
  int64_t rem = outIdx;
  for (int i = plan.outRank - 1; i >= 0; --i) {
    assign[i] = rem % plan.outDims[i];
    rem /= plan.outDims[i];
  }
  rem = cIdx;
  for (int i = plan.nContractedLabels - 1; i >= 0; --i) {
    assign[plan.outRank + i] = rem % plan.contractedDims[i];
    rem /= plan.contractedDims[i];
  }

  const double* operands[kEinsumMaxOperands] = {opA, opB, opC};
  double prod = 1.0;
  for (int o = 0; o < plan.nOperands; ++o) {
    int64_t off = 0;
    for (int a = 0; a < plan.opRank[o]; ++a) {
      off += assign[plan.opAxisLabel[o][a]] * plan.opAxisStride[o][a];
    }
    prod *= operands[o][off];
  }
  atomicAdd(&out[outIdx], scale * prod);
}

// The three manual "diagonal slice" accumulations the NumPy reference does with plain
// NumPy indexing instead of an einsum (each kernel's comment below gives the exact Python -> index correspondence
// it mirrors).
// `norb` is the common active-space extent every axis here has.

// a16[:, i, :, :, :, i] += fdm2  for i in range(norb); a16 is (norb,)^6
// (p,q=i,r,a,b,c=i), fdm2 is (norb,)^4 (p,r,a,b).
__global__ void energy_diag_a16(
    const double* __restrict__ fdm2, double* __restrict__ a16, int norb) {
  const int64_t n = norb;
  const int64_t total = n * n * n * n * n;  // p,r,a,b,i
  const int64_t tid = globalThreadIndex();
  if (tid >= total) return;
  int64_t rem = tid;
  const int i = (int)(rem % n); rem /= n;
  const int b = (int)(rem % n); rem /= n;
  const int a = (int)(rem % n); rem /= n;
  const int r = (int)(rem % n); rem /= n;
  const int p = (int)rem;
  const int64_t fdm2Off = idx4(p, r, a, b, n);
  const int64_t a16Off = idx6(p, i, r, a, b, i, n);
  atomicAdd(&a16[a16Off], fdm2[fdm2Off]);
}

// a22[:, i, :, i, :, :] -= fdm2  for i in range(norb); a22 is (norb,)^6
// (I,j=i,k,a=i,b,c), fdm2 is (norb,)^4 (I,k,b,c).
__global__ void energy_diag_a22a(
    const double* __restrict__ fdm2, double* __restrict__ a22, int norb) {
  const int64_t n = norb;
  const int64_t total = n * n * n * n * n;  // I,k,b,c,i
  const int64_t tid = globalThreadIndex();
  if (tid >= total) return;
  int64_t rem = tid;
  const int i = (int)(rem % n); rem /= n;
  const int c = (int)(rem % n); rem /= n;
  const int b = (int)(rem % n); rem /= n;
  const int k = (int)(rem % n); rem /= n;
  const int I = (int)rem;
  const int64_t fdm2Off = idx4(I, k, b, c, n);
  const int64_t a22Off = idx6(I, i, k, i, b, c, n);
  atomicAdd(&a22[a22Off], -fdm2[fdm2Off]);
}

// a22[:, i, :, :, i, :] += fdm2 * 2  for i in range(norb); a22 is (norb,)^6
// (I,j=i,k,a,b=i,c), fdm2 is (norb,)^4 (I,k,a,c).
__global__ void energy_diag_a22b(
    const double* __restrict__ fdm2, double* __restrict__ a22, int norb) {
  const int64_t n = norb;
  const int64_t total = n * n * n * n * n;  // I,k,a,c,i
  const int64_t tid = globalThreadIndex();
  if (tid >= total) return;
  int64_t rem = tid;
  const int i = (int)(rem % n); rem /= n;
  const int c = (int)(rem % n); rem /= n;
  const int a = (int)(rem % n); rem /= n;
  const int k = (int)(rem % n); rem /= n;
  const int I = (int)rem;
  const int64_t fdm2Off = idx4(I, k, a, c, n);
  const int64_t a22Off = idx6(I, i, k, a, i, c, n);
  atomicAdd(&a22[a22Off], 2.0 * fdm2[fdm2Off]);
}

// SC-NEVPT2 Sijrs (the MP2-like doubly-external class) over one slab of core
// orbitals -- the device port of energy.cpp's former host loop, so the class
// no longer needs the whole (ncore*nvirt)^2 cvcv on the host. `g` is the slab
// g[il, a, j, b] = (i a | j b) for i = i0 + il, contiguous row-major
// (ni, nvirt, ncore, nvirt); every element needs only g[il, b, j, a] besides
// itself, which is in the same slab. acc[0] += norm, acc[1] += energy:
//   theta = 2 g_iajb - g_ibja,  norm = sum g_iajb theta,
//   e = sum g_iajb theta / (e_j - e_a + e_i - e_b).
// A grid-stride loop, a block_reduce per block (common/block_reduce.cuh: warp
// shuffles, then across warps), then one atomicAdd per block and quantity --
// so the sum reassociates (vs. the old serial host loop) and is compared to
// the golden to 1e-7 like everything else. blockDim.x MUST be kSijrsBlock (a
// whole number of warps; energySijrs below launches with exactly that).
constexpr unsigned kSijrsBlock = 256;
static_assert(kSijrsBlock == kBlock);
constexpr unsigned kSijrsWarps = kSijrsBlock / kWarpSize;
static_assert(kSijrsWarps * kWarpSize == kSijrsBlock);

__global__ void energy_sijrs(
    const double* __restrict__ g, int ni, int i0, int ncore, int nvirt,
    const double* __restrict__ eCore, const double* __restrict__ eVirt,
    double* __restrict__ acc) {
  // One partials buffer per quantity, so the two folds need no barrier
  // between them (block_reduce: no thread may still use the buffer on entry).
  __shared__ double sNorm[kSijrsWarps];
  __shared__ double sEner[kSijrsWarps];
  const int64_t total = int64_t{ni} * nvirt * ncore * nvirt;
  double norm = 0.0, ener = 0.0;
  for (int64_t t = globalThreadIndex(); t < total; t += gridStride()) {
    int64_t rem = t;
    const int b = (int)(rem % nvirt); rem /= nvirt;
    const int j = (int)(rem % ncore); rem /= ncore;
    const int a = (int)(rem % nvirt); rem /= nvirt;
    const int il = (int)rem;
    const double giJab = g[t];
    const double giJba = g[idx4(il, b, j, a, nvirt, ncore, nvirt)];
    const double theta = 2.0 * giJab - giJba;
    const double djba = (eCore[j] - eVirt[a]) + (eCore[i0 + il] - eVirt[b]);
    norm += giJab * theta;
    ener += giJab / djba * theta;
  }
  const cg::thread_block block = cg::this_thread_block();
  norm = block_reduce(block, sNorm, norm, AddOp{});
  ener = block_reduce(block, sEner, ener, AddOp{});
  if (threadIdx.x == 0) {
    atomicAdd(&acc[0], norm);
    atomicAdd(&acc[1], ener);
  }
}

}  // namespace

wwrError_t einsumGeneric(const wwrStream_t stream, const double* opA, const double* opB,
                         const double* opC, double* out, const EinsumPlan plan,
                         const double scale) {
  const int64_t total = plan.outSize * plan.contractedSize;
  if (total < 1) return wwrSuccess;
  einsum_generic<<<gridFor(total, kBlock), kBlock, 0, stream>>>(opA, opB, opC, out, plan,
                                                         scale);
  return wwrGetLastError();
}

namespace {

// The three diag-slice kernels each cover (p,r,a,b,i)-style n^5 index tuples.
int64_t diagTotal(const int64_t n) {
  return n * n * n * n * n;
}

}  // namespace

wwrError_t energyDiagA16(const wwrStream_t stream, const double* fdm2, double* a16,
                         const int norb) {
  const int64_t total = diagTotal(norb);
  if (total < 1) return wwrSuccess;
  energy_diag_a16<<<gridFor(total, kBlock), kBlock, 0, stream>>>(fdm2, a16, norb);
  return wwrGetLastError();
}

wwrError_t energyDiagA22a(const wwrStream_t stream, const double* fdm2, double* a22,
                          const int norb) {
  const int64_t total = diagTotal(norb);
  if (total < 1) return wwrSuccess;
  energy_diag_a22a<<<gridFor(total, kBlock), kBlock, 0, stream>>>(fdm2, a22, norb);
  return wwrGetLastError();
}

wwrError_t energyDiagA22b(const wwrStream_t stream, const double* fdm2, double* a22,
                          const int norb) {
  const int64_t total = diagTotal(norb);
  if (total < 1) return wwrSuccess;
  energy_diag_a22b<<<gridFor(total, kBlock), kBlock, 0, stream>>>(fdm2, a22, norb);
  return wwrGetLastError();
}

wwrError_t energySijrs(const wwrStream_t stream, const double* g, const int ni, const int i0,
                       const int ncore, const int nvirt, const double* eCore, const double* eVirt,
                       double* acc) {
  // A grid-stride loop, so the grid is clamped rather than sized to cover:
  // at least one block, at most 1024.
  const int64_t total = int64_t{ni} * nvirt * ncore * nvirt;
  const unsigned grid = std::clamp(gridFor(total, kBlock), 1u, 1024u);
  energy_sijrs<<<grid, kSijrsBlock, 0, stream>>>(g, ni, i0, ncore, nvirt, eCore, eVirt, acc);
  return wwrGetLastError();
}

}  // namespace nevpt2::device
