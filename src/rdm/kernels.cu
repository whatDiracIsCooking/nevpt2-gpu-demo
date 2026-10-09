// The 8 CAS-independent generic RDM-build kernels. This is the whole point of
// the generic kernel set: no per-active-space text substitution happens here
// -- this file is compiled exactly once (see CMakeLists.txt) and serves ANY
// (norb, nelec) at runtime, with norb/na/nb/nlink as kernel arguments.
//
// It is never compiled on its own: rdm_launch.cu #includes it and holds the
// launchers (see that file for why they share a TU). These kernels use no
// warp shuffle and no atomics, so WarpWraps' runtime.h -- which picks the
// vendor header off the compiler's own device macro -- is the whole prelude,
// and this one file builds for CUDA and HIP alike.
#include <runtime.h>  // WarpWraps: <cuda_runtime.h> or <hip/hip_runtime.h>

// idx2/idx4/idx6: every n^4 / n^6 offset widens to nevpt2::int64_t once, there,
// rather than through a (size_t) cast at the head of each chain.
// The kernel-local ints (thread index, link-table entries, n2, n4) stay int.
#include "common/device_index.h"

using nevpt2::int64_t;
using nevpt2::idx2;
using nevpt2::idx4;
using nevpt2::idx6;

// --- K1 generic produce (any active space, runtime norb/na/nb/nlink) ---
__global__ void produce_generic(
    const double* __restrict__ ci,
    const int* __restrict__ flink_a, const int* __restrict__ flink_b,
    const int* __restrict__ rlink_a, const int* __restrict__ rlink_b,
    double* __restrict__ R, double* __restrict__ L2,
    int norb, int na, int nb, int nla, int nlb, int k0, int width) {
  const int n2 = norb * norb;
  const int n4 = n2 * n2;
  const int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= width) return;
  const int K = k0 + kl;   // global determinant index
  const int A = K / nb;   // alpha string index
  const int B = K % nb;   // beta  string index

  // R[t,u,kl] = (E_tu|0>)[K]: gather sources that excite into K
  for (int tu = 0; tu < n2; ++tu) R[idx2(tu, kl, width)] = 0.0;
  // alpha part: sources A' with an excitation A'->A, beta spectator
  for (int rowra = 0; rowra < nla; ++rowra) {
    const int bra = ((A)*nla + rowra)*4;
    const int cra = rlink_a[bra + 0]; const int dra = rlink_a[bra + 1];
    const int gra = rlink_a[bra + 2]; const double sra = (double)rlink_a[bra + 3];
    R[idx2(cra*norb + dra, kl, width)] += sra * ci[idx2(gra, B, nb)];
  }
  // beta part: sources B' with an excitation B'->B, alpha spectator
  for (int rowrb = 0; rowrb < nlb; ++rowrb) {
    const int brb = ((B)*nlb + rowrb)*4;
    const int crb = rlink_b[brb + 0]; const int drb = rlink_b[brb + 1];
    const int grb = rlink_b[brb + 2]; const double srb = (double)rlink_b[brb + 3];
    R[idx2(crb*norb + drb, kl, width)] += srb * ci[idx2(A, grb, nb)];
  }

  // L2[p,q,r,s,kl] = (E_pq E_rs adjoint |0>)[K]: two forward walks.
  for (int pqrs = 0; pqrs < n4; ++pqrs) L2[idx2(pqrs, kl, width)] = 0.0;
  for (int rowla = 0; rowla < nla; ++rowla) {
    const int bla = ((A)*nla + rowla)*4;
    const int cla = flink_a[bla + 0]; const int dla = flink_a[bla + 1];
    const int gla = flink_a[bla + 2]; const double sla = (double)flink_a[bla + 3];
    { const int rr = cla, ss = dla; const double sg2 = sla; const int At = gla;
   for (int rowia = 0; rowia < nla; ++rowia) {
     const int bia = ((At)*nla + rowia)*4;
     const int cia = flink_a[bia + 0]; const int dia = flink_a[bia + 1];
     const int gia = flink_a[bia + 2]; const double sia = (double)flink_a[bia + 3];
      L2[idx2((cia*norb + dia)*n2 + rr*norb + ss, kl, width)] += sg2 * sia * ci[idx2(gia, B, nb)];
   }
   for (int rowib = 0; rowib < nlb; ++rowib) {
     const int bib = ((B)*nlb + rowib)*4;
     const int cib = flink_b[bib + 0]; const int dib = flink_b[bib + 1];
     const int gib = flink_b[bib + 2]; const double sib = (double)flink_b[bib + 3];
      L2[idx2((cib*norb + dib)*n2 + rr*norb + ss, kl, width)] += sg2 * sib * ci[idx2(At, gib, nb)];
   }
    }
  }
  for (int rowlb = 0; rowlb < nlb; ++rowlb) {
    const int blb = ((B)*nlb + rowlb)*4;
    const int clb = flink_b[blb + 0]; const int dlb = flink_b[blb + 1];
    const int glb = flink_b[blb + 2]; const double slb = (double)flink_b[blb + 3];
    { const int rr = clb, ss = dlb; const double sg2 = slb; const int Bt = glb;
   for (int rowja = 0; rowja < nla; ++rowja) {
     const int bja = ((A)*nla + rowja)*4;
     const int cja = flink_a[bja + 0]; const int dja = flink_a[bja + 1];
     const int gja = flink_a[bja + 2]; const double sja = (double)flink_a[bja + 3];
      L2[idx2((cja*norb + dja)*n2 + rr*norb + ss, kl, width)] += sg2 * sja * ci[idx2(gja, Bt, nb)];
   }
   for (int rowjb = 0; rowjb < nlb; ++rowjb) {
     const int bjb = ((Bt)*nlb + rowjb)*4;
     const int cjb = flink_b[bjb + 0]; const int djb = flink_b[bjb + 1];
     const int gjb = flink_b[bjb + 2]; const double sjb = (double)flink_b[bjb + 3];
      L2[idx2((cjb*norb + djb)*n2 + rr*norb + ss, kl, width)] += sg2 * sjb * ci[idx2(A, gjb, nb)];
   }
    }
  }
}

// --- K3 generic dm3 digest GEMM (any active space) ---
__global__ void digest_dm3_generic(
    const double* __restrict__ R, const double* __restrict__ L2,
    double* __restrict__ dm3, int norb, int width) {
  const int n2 = norb * norb;
  const int M = n2 * n2;    // e = pqrs (A = L2 rows)
  const int N = n2;         // tu       (B = R  rows)
  const int Kdim = width;
  const int TILE = 16, TK = 16;
  const int nthreads = blockDim.x * blockDim.y;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * TILE + threadIdx.y;  // M index (e)
  const int col = blockIdx.x * TILE + threadIdx.x;  // N index (tu)
  __shared__ double As[TILE][TK];  // L2 tile: rows m, cols K
  __shared__ double Bs[TILE][TK];  // R  tile: rows n, cols K
  double acc = 0.0;
  for (int k0 = 0; k0 < Kdim; k0 += TK) {
    for (int i = tid; i < TILE * TK; i += nthreads) {
      const int r = i / TK, c = i % TK;
      const int gm = blockIdx.y * TILE + r;
      const int gn = blockIdx.x * TILE + r;
      const int gk = k0 + c;
      As[r][c] = (gm < M && gk < Kdim)
        ? L2[idx2(gm, gk, Kdim)] : 0.0;
      Bs[r][c] = (gn < N && gk < Kdim)
        ? R[idx2(gn, gk, Kdim)] : 0.0;
    }
    __syncthreads();
    const int kk = (Kdim - k0 < TK) ? (Kdim - k0) : TK;
    if (row < M && col < N)
      for (int c = 0; c < kk; ++c)
        acc += As[threadIdx.y][c] * Bs[threadIdx.x][c];
    __syncthreads();
  }
  if (row < M && col < N)
    dm3[idx2(row, col, N)] = acc;
}

// --- K2 generic ERI consume (ca, any active space) ---
__global__ void consume_ca_generic(
    const double* __restrict__ eri,
    const double* __restrict__ L2, double* __restrict__ W,
    int norb, int width) {
  const int n2 = norb * norb;
  const int ndet = width;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // idx enumerates (af, K); af = a*norb + free (n2 of them)
  if (idx >= n2 * ndet) return;
  const int af = idx / ndet;
  const int K  = idx % ndet;
  const int a  = af / norb;
  const int fr = af % norb;   // the free index (s for ca, r for ac)
  double w = 0.0;
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int r = 0; r < norb; ++r) {
        const double e = eri[idx4(a, r, q, p, norb)];
        const int64_t l = idx2(idx4(p, q, r, fr, norb), K, ndet);
        w += e * L2[l];
      }
  W[idx2(af, K, ndet)] = w;
}

// --- generic fdm2 correction (ca, any active space) ---
__global__ void fdm2_ca_generic(
    const double* __restrict__ eri,
    const double* __restrict__ dm3, double* __restrict__ fdm2, int norb) {
  const int n4 = norb * norb * norb * norb;
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= n4) return;   // o = (i,j,k,l)
  const int i = (o / (norb * norb * norb)) % norb;
  const int j = (o / (norb * norb)) % norb;
  const int k = (o / norb) % norb;
  const int l = o % norb;
  double acc = 0.0;
  // X[a,D,E,F]=sum_{C,B,A} eri[a,C,B,A] dm3[A,B,C,D,E,F]; fdm2[i,j,k,l] = X[l,k,j,i]
  for (int C = 0; C < norb; ++C)
    for (int B = 0; B < norb; ++B)
      for (int A = 0; A < norb; ++A) {
        const double e = eri[idx4(l, C, B, A, norb)];
        const int64_t d = idx6(A, B, C, k, j, i, norb);
        acc += e * dm3[d];
      }
  fdm2[o] = acc;
}

// --- generic K4 wedge+fdm2 reconstruction (ca, any active space) ---
__global__ void wedge_ca_generic(
    const double* __restrict__ fdm2, double* __restrict__ f3, int norb) {
  const int ij = blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= norb * norb) return;
  const int i = ij / norb;
  const int j = ij % norb;
  if (j >= i) return;   // only the strict lower wedge j < i writes
  // out[j,:,i] = out[i,:,j].transpose(1,0,2,3) over the 4 free axes (1,3,4,5)
  for (int b = 0; b < norb; ++b)
    for (int c = 0; c < norb; ++c)
      for (int d = 0; d < norb; ++d)
        for (int e = 0; e < norb; ++e) {
          const int64_t tgt = idx6(j, b, i, c, d, e, norb);
          const int64_t src = idx6(i, c, j, b, d, e, norb);
          f3[tgt] = f3[src];
        }
  // out[j,i,i,:] += fdm2[j,:]  (axes 1=i,2=i fixed; free axes 3,4,5)
  for (int c = 0; c < norb; ++c)
    for (int d = 0; d < norb; ++d)
      for (int e = 0; e < norb; ++e) {
        const int64_t tgt = idx6(j, i, i, c, d, e, norb);
        const int64_t f = idx4(j, c, d, e, norb);
        f3[tgt] += fdm2[f];
      }
  // out[j,:,i,j] -= fdm2[i,:]  (axes 0=j,2=i,3=j fixed; free 1,4,5)
  for (int b = 0; b < norb; ++b)
    for (int d = 0; d < norb; ++d)
      for (int e = 0; e < norb; ++e) {
        const int64_t tgt = idx6(j, b, i, j, d, e, norb);
        const int64_t f = idx4(i, b, d, e, norb);
        f3[tgt] -= fdm2[f];
      }
}

// --- K2 generic ERI consume (ac, any active space) ---
__global__ void consume_ac_generic(
    const double* __restrict__ eri,
    const double* __restrict__ L2, double* __restrict__ W,
    int norb, int width) {
  const int n2 = norb * norb;
  const int ndet = width;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // idx enumerates (af, K); af = a*norb + free (n2 of them)
  if (idx >= n2 * ndet) return;
  const int af = idx / ndet;
  const int K  = idx % ndet;
  const int a  = af / norb;
  const int fr = af % norb;   // the free index (s for ca, r for ac)
  double w = 0.0;
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int s = 0; s < norb; ++s) {
        const double e = eri[idx4(a, s, q, p, norb)];
        const int64_t l = idx2(idx4(p, q, fr, s, norb), K, ndet);
        w += e * L2[l];
      }
  W[idx2(af, K, ndet)] = w;
}

// --- generic fdm2 correction (ac, any active space) ---
__global__ void fdm2_ac_generic(
    const double* __restrict__ eri,
    const double* __restrict__ dm3, double* __restrict__ fdm2, int norb) {
  const int n4 = norb * norb * norb * norb;
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= n4) return;   // o = (i,j,k,l)
  const int i = (o / (norb * norb * norb)) % norb;
  const int j = (o / (norb * norb)) % norb;
  const int k = (o / norb) % norb;
  const int l = o % norb;
  double acc = 0.0;
  // Y[a,A,B,D]=sum_{C,E,F} eri[a,C,E,F] dm3[A,B,C,D,E,F]; fdm2[i,j,k,l] = Y[k,i,j,l]
  for (int C = 0; C < norb; ++C)
    for (int E = 0; E < norb; ++E)
      for (int F = 0; F < norb; ++F) {
        const double e = eri[idx4(k, C, E, F, norb)];
        const int64_t d = idx6(i, j, C, l, E, F, norb);
        acc += e * dm3[d];
      }
  fdm2[o] = acc;
}

// --- generic K4 wedge+fdm2 reconstruction (ac, any active space) ---
__global__ void wedge_ac_generic(
    const double* __restrict__ fdm2, double* __restrict__ f3, int norb) {
  const int ij = blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= norb * norb) return;
  const int i = ij / norb;
  const int j = ij % norb;
  if (j >= i) return;   // only the strict lower wedge j < i writes
  // out[j,:,i] = out[i,:,j].transpose(1,0,2,3) over the 4 free axes (1,3,4,5)
  for (int b = 0; b < norb; ++b)
    for (int c = 0; c < norb; ++c)
      for (int d = 0; d < norb; ++d)
        for (int e = 0; e < norb; ++e) {
          const int64_t tgt = idx6(j, b, i, c, d, e, norb);
          const int64_t src = idx6(i, c, j, b, d, e, norb);
          f3[tgt] = f3[src];
        }
  // out[j,i,i,:] += fdm2[j,:]  (axes 1=i,2=i fixed; free axes 3,4,5)
  for (int c = 0; c < norb; ++c)
    for (int d = 0; d < norb; ++d)
      for (int e = 0; e < norb; ++e) {
        const int64_t tgt = idx6(j, i, i, c, d, e, norb);
        const int64_t f = idx4(j, c, d, e, norb);
        f3[tgt] += fdm2[f];
      }
  // out[j,:,i,j] -= fdm2[i,:]  (axes 0=j,2=i,3=j fixed; free 1,4,5)
  for (int b = 0; b < norb; ++b)
    for (int d = 0; d < norb; ++d)
      for (int e = 0; e < norb; ++e) {
        const int64_t tgt = idx6(j, b, i, j, d, e, norb);
        const int64_t f = idx4(i, b, d, e, norb);
        f3[tgt] -= fdm2[f];
      }
}
