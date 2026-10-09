// nevpt2.device_resources:device_select -- selectDevice, which makes DeviceResources'
// device current before its member initializers run.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :device_select;` and from no importer of nevpt2.device_resources.
// (Used by device_resources.cpp.)
module nevpt2.device_resources:device_select;

import std;
import nevpt2.device_resources;

namespace nevpt2 {

// Selects the device and returns its index, so DeviceResources' member
// initializers (device_resources.cpp) run with it current. create() has already selected it once (and returned an
// Error if that failed), so a failure here is not the environment's: abort.
int selectDevice(int idx) {
  gpuCheck(wwrSetDevice(idx));
  return idx;
}

}  // namespace nevpt2
