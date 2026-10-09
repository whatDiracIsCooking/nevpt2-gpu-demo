// device_einsum_bridge.h -- the declarations shared between the host module
// units that launch the generic device einsum (einsum.cpp) and the Sijrs slab
// reduction (energy/energy.cpp), and device_einsum.cu (device-compiled,
// linked in as the nevpt2.einsum.device static library). Same shape as
// rdm/rdm_accumulate_bridge.h; see that header for why a host unit includes
// it after its imports, inside extern "C++".
//
// Plain types only across this boundary, plus EinsumPlan BY VALUE: nvcc
// compiles the .cu's host half with its own host compiler, which need not
// share libc++'s ABI with the module units. EinsumPlan is a POD of ints and
// int64_ts (einsum_plan.h static_asserts it), so both sides agree on its
// layout.
//
// Every function launches on `stream` (the caller's -- DeviceResources' one stream) and
// returns the LAUNCH status, wwrSuccess on success. All launch 256-thread
// blocks -- see device_einsum.cu's header.
#pragma once

#include "einsum/einsum_plan.h"
#include "wwr/wwr.h"  // wwrStream_t (+ wwrError_t in a device pass)

namespace nevpt2::device {

// out[o] += scale * prod over operands of op[...] for every (output element,
// contracted-index combination) pair `plan` describes. opB/opC may be null
// when plan.nOperands says they are unused.
wwrError_t einsumGeneric(wwrStream_t stream, const double* opA, const double* opB,
                         const double* opC, double* out, EinsumPlan plan, double scale);

// The three "diagonal slice" accumulations from the NumPy reference's make_a16 /
// make_a22 (device_einsum.cu has the index correspondence of each).
wwrError_t energyDiagA16(wwrStream_t stream, const double* fdm2, double* a16, int norb);
wwrError_t energyDiagA22a(wwrStream_t stream, const double* fdm2, double* a22, int norb);
wwrError_t energyDiagA22b(wwrStream_t stream, const double* fdm2, double* a22, int norb);

// acc[0] += norm, acc[1] += energy of the Sijrs class over the core-orbital
// slab g[il, a, j, b] = (i a | j b), i = i0 + il, contiguous
// (ni, nvirt, ncore, nvirt).
wwrError_t energySijrs(wwrStream_t stream, const double* g, int ni, int i0, int ncore, int nvirt,
                       const double* eCore, const double* eVirt, double* acc);

}  // namespace nevpt2::device
