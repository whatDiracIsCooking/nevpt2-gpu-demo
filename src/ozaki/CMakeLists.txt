# The --ozaki digest: the three RDM-build digest GEMMs emulated on int8
# tensor cores by the Ozaki scheme. Built on BOTH backends: the WMMA calls go
# through WarpWraps' <wmma.h> (wwr::wwrwmma = nvcuda::wmma / rocwmma), so
# nothing here names a vendor. rdm/rdm_build.cpp calls it through
# ozaki_digest_bridge.h, as it does the other linked-in kernels.
#
# wwr::device: runtime.h and wmma.h by -I, and on HIP the ROCm include root
# that rocWMMA's header-only <rocwmma/rocwmma.hpp> lives under (WarpWraps
# needs no find_package for it). nevpt2::common::device: the src/ root, for
# "ozaki/ozaki_digest_bridge.h" and the "wwr/wwr.h" it includes.
add_device_library(
  NAME nevpt2.ozaki.device
  SOURCES ozaki_digest.cu
  LINK_PRIVATE wwr::device nevpt2::common::device
)
target_include_directories(nevpt2.ozaki.device PRIVATE ${NEVPT2_SRC_ROOT})
