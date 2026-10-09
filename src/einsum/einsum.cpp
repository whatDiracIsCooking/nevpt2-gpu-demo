module;

// The vendor headers device_einsum_bridge.h pulls in (runtime.h) and the one
// POD its launchers take (einsum_plan.h), textually in the GMF; the bridge
// itself is included in the purview below, after the imports, so it can name
// wwrError_t (see rdm/rdm_accumulate_bridge.h).
#include <runtime.h>
#include "einsum/einsum_plan.h"

module nevpt2.einsum;

import std;
import :planner;
import :launch;

import nevpt2.profile;
import wwr.runtime_api;

// The kernel launchers (device_einsum.cu, linked in): extern "C++" attaches
// them to the global module, so they keep external linkage and bind to the
// .cu's definitions (see rdm/rdm_accumulate_bridge.h).
extern "C++" {
#include "einsum/device_einsum_bridge.h"
}

namespace nevpt2 {

DeviceTensor deviceEinsumNew(const std::string& subscripts,
                              std::vector<const DeviceTensor*> operands, double scale,
                              const DeviceResources& res, std::source_location loc) {
  std::vector<int64_t> outDims;
  EinsumPlan plan = buildPlan(subscripts, operands, &outDims, loc);
  DeviceTensor out = DeviceTensor::zeros(outDims, res);  // zeroed
  launchGeneric(plan, operands, out, scale, res.stream(), subscripts);
  return out;
}

void deviceEinsumAccum(const std::string& subscripts, std::vector<const DeviceTensor*> operands,
                       const DeviceTensor& out, double scale, wwrStream_t stream,
                       std::source_location loc) {
  std::vector<int64_t> outDims;
  EinsumPlan plan = buildPlan(subscripts, operands, &outDims, loc);
  if (outDims != out.dims)
    check(false, std::format("deviceEinsumAccum: output shape mismatch for {}", subscripts), loc);
  launchGeneric(plan, operands, out, scale, stream, subscripts);
}

void deviceTransposeAccum(const DeviceTensor& x, const std::vector<int>& axes,
                          const DeviceTensor& out, double scale, wwrStream_t stream,
                          std::source_location loc) {
  std::string subs = transposeSubscripts(x.rank(), axes);
  deviceEinsumAccum(subs, {&x}, out, scale, stream, loc);
}

DeviceTensor deviceTransposeNew(const DeviceTensor& x,
                                 const std::vector<int>& axes, double scale,
                                 const DeviceResources& res, std::source_location loc) {
  std::string subs = transposeSubscripts(x.rank(), axes);
  return deviceEinsumNew(subs, {&x}, scale, res, loc);
}

void launchDiagA16(const DeviceTensor& fdm2, const DeviceTensor& a16, int64_t norb,
                   wwrStream_t stream) {
  launchDiag(fdm2, a16, norb, stream, "diagA16", device::energyDiagA16);
}
void launchDiagA22a(const DeviceTensor& fdm2, const DeviceTensor& a22, int64_t norb,
                    wwrStream_t stream) {
  launchDiag(fdm2, a22, norb, stream, "diagA22a", device::energyDiagA22a);
}
void launchDiagA22b(const DeviceTensor& fdm2, const DeviceTensor& a22, int64_t norb,
                    wwrStream_t stream) {
  launchDiag(fdm2, a22, norb, stream, "diagA22b", device::energyDiagA22b);
}

}  // namespace nevpt2
