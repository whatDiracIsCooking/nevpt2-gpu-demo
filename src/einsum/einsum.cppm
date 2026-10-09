// nevpt2.einsum -- host-side planner + launcher for the generic device einsum kernel
// (device_einsum.cu). Parses numpy-einsum subscript strings, so
// energy/energy.cpp's call sites are a mechanical, checkable transcription of
// the NumPy reference they were ported from, running on the GPU.
//
// WHY NOT wwr.tensor (cuTENSOR / hipTensor)? Probed, and a negative result on
// the HIP side: no f64 on gfx12, and linking libhiptensor aborts the process at
// load. Keep this home-brewed kernel until that changes; the measurements are
// in docs/performance.md, "What did not help".
export module nevpt2.einsum;

import std;
// Re-exported: every operand is a DeviceTensor; a call that allocates takes
// the DeviceResources (nevpt2.device_resources, through nevpt2.device_tensor) whose pool
// and stream it uses, every other call the wwrStream_t it is issued on.
export import nevpt2.wwr;
export import nevpt2.device_tensor;

export namespace nevpt2 {

// Allocates a fresh output tensor (shape inferred from `subscripts`' output
// labels) from res's pool and writes `scale * einsum(subscripts, operands...)`
// into it on res.stream(). Mirrors a host `Tensor x = einsum(...) * scale;`.
DeviceTensor deviceEinsumNew(const std::string& subscripts,
                              std::vector<const DeviceTensor*> operands,
                              double scale, const DeviceResources& res,
                              std::source_location loc = std::source_location::current());

// Accumulates `scale * einsum(subscripts, operands...)` into an existing
// `out` (already correctly shaped). Mirrors `x += einsum(...) * scale;` /
// `x -= einsum(...);` (scale=-1.0).
void deviceEinsumAccum(const std::string& subscripts, std::vector<const DeviceTensor*> operands,
                       const DeviceTensor& out, double scale, wwrStream_t stream,
                       std::source_location loc = std::source_location::current());

// Accumulates `scale * transpose(x, axes)` into an existing `out`. Same
// mechanism as deviceEinsumAccum (transpose is a 1-operand, no-contraction
// einsum whose output label order is the `axes` permutation of the input's)
// -- see transposeSubscripts in einsum/einsum.cpp for the derivation.
void deviceTransposeAccum(const DeviceTensor& x, const std::vector<int>& axes,
                          const DeviceTensor& out, double scale, wwrStream_t stream,
                          std::source_location loc = std::source_location::current());

DeviceTensor deviceTransposeNew(const DeviceTensor& x,
                                 const std::vector<int>& axes, double scale,
                                 const DeviceResources& res,
                                 std::source_location loc = std::source_location::current());

// The three manual "diagonal slice" accumulations from the NumPy reference's make_a16
// / make_a22 -- see device_einsum.cu's kernel comments for the exact index
// correspondence each one is.
void launchDiagA16(const DeviceTensor& fdm2, const DeviceTensor& a16, int64_t norb,
                   wwrStream_t stream);
void launchDiagA22a(const DeviceTensor& fdm2, const DeviceTensor& a22, int64_t norb,
                    wwrStream_t stream);
void launchDiagA22b(const DeviceTensor& fdm2, const DeviceTensor& a22, int64_t norb,
                    wwrStream_t stream);

// Every subscript string is a literal at its call site, so one that is
// malformed or does not fit its operands is our bug: deviceEinsumNew/Accum
// and deviceTransposeNew/Accum abort through nevpt2.error_handling's check()
// (the abort tier), naming `loc` -- the caller's file:line.

// --profile: every launch above goes through nevpt2.profile's time(), on the
// stream it is issued on, labelled by its subscript string (or
// "diagA16"/"diagA22a"/"diagA22b") in whatever profile::Section is open.
// Off, that is one bool check per launch.

}  // namespace nevpt2
