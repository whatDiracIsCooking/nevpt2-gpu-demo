// nevpt2.einsum:launch -- the profiled kernel launches: the generic contraction
// and the bracket the three diag-slice launchers share.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :launch;` and from no importer of nevpt2.einsum.
// (Used by einsum.cpp.)
module;

// What einsum/device_einsum_bridge.h pulls in, textually in the GMF; the
// bridge itself is included after the imports.
#include <runtime.h>
#include "einsum/einsum_plan.h"

module nevpt2.einsum:launch;

import std;
import nevpt2.einsum;

import nevpt2.profile;
import wwr.runtime_api;

// The kernel launchers it calls: extern "C++" attaches them to the global
// module, so they bind to the .cu's definitions (see
// rdm/rdm_accumulate_bridge.h).
extern "C++" {
#include "einsum/device_einsum_bridge.h"
}

namespace nevpt2 {

void launchGeneric(const EinsumPlan& plan, const std::vector<const DeviceTensor*>& operands,
                   const DeviceTensor& out, double scale, wwrStream_t stream,
                   const std::string& label) {
  const double* opPtrs[kEinsumMaxOperands] = {nullptr, nullptr, nullptr};
  for (std::size_t i = 0; i < operands.size(); ++i) {
    opPtrs[i] = operands[i]->data();
  }

  profile::time(
      label, stream,
      [&] {
        gpuCheck(device::einsumGeneric(stream, opPtrs[0], opPtrs[1], opPtrs[2], out.data(),
                                          plan, scale));
      },
      plan.outSize, plan.contractedSize);
}

// The profiling bracket shared by the three diag-slice launchers.
template <typename Launch>
void launchDiag(const DeviceTensor& fdm2, const DeviceTensor& acc, int64_t norb, wwrStream_t stream,
                const std::string& label, Launch launch) {
  const int64_t total = norb * norb * norb * norb * norb;
  // The diag-slice launchers take int: narrowed once, checked.
  const int norbK = narrowTo<int>(norb);
  profile::time(
      label, stream,
      [&] {
        gpuCheck(launch(stream, fdm2.data(), acc.data(), norbK));
      },
      total, 1);
}

}  // namespace nevpt2
