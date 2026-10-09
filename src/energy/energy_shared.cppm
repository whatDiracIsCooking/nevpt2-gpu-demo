// nevpt2.energy:shared -- what the SC (energy.cpp) and PC (energy_pc.cpp)
// classes share: the RDM-only intermediates PC's
// metrics and Hamiltonians are built from, and the Sijrs class, which PC reuses
// whole. An internal partition: nothing here is exported, so these have module
// linkage -- reachable from every unit that does `import :shared;` and from no
// importer of nevpt2.energy. energy.cpp defines the functions. (The slab walk,
// forBatches, is nevpt2.energy_finish's.) It imports the primary interface for the types its
// declarations name (DeviceTensor, ActiveIntegralsDevice, IntegralSource); the
// primary interface does not import it back (cmake/README.md,
// "add_cxx_module_library").
module nevpt2.energy:shared;

import std;
import nevpt2.energy;

namespace nevpt2 {

// forBatches, the slab walk every class runs, is nevpt2.energy_finish's
// (re-exported by nevpt2.energy).

// PC-NEVPT2 solves with the metric S and Hamiltonian K that the SC einsums
// contract to a scalar (docs/pc-nevpt2.md, "PC-NEVPT2 on the device: the design"), so
// it builds them from these same intermediates rather than a copy. All on
// res.stream(). These three are (n, n), PySCF's make_hdm1 / make_a3 / make_k27.
DeviceTensor make_hdm1(const DeviceTensor& dm1, const DeviceResources& dr);
DeviceTensor make_a3(const DeviceTensor& h1e, const DeviceTensor& h2e, const DeviceTensor& dm1,
                     const DeviceTensor& dm2, const DeviceTensor& hdm1,
                     const DeviceResources& dr);
DeviceTensor make_k27(const DeviceTensor& h1e, const DeviceTensor& h2e,
                      const DeviceTensor& dm1, const DeviceTensor& dm2,
                      const DeviceResources& dr);
// For PC Srs, Sij, Sir: (n, n, n, n) unless noted, PySCF's make_hdm2,
// make_hdm3 (n^6), make_a7 ({rm2, a7}), make_a9, make_a12, make_a13.
DeviceTensor make_hdm2(const DeviceTensor& dm1, const DeviceTensor& dm2,
                       const DeviceResources& dr);
DeviceTensor make_hdm3(const DeviceTensor& dm1, const DeviceTensor& dm2,
                       const DeviceTensor& dm3, const DeviceTensor& hdm1,
                       const DeviceTensor& hdm2, const DeviceResources& dr);
std::pair<DeviceTensor, DeviceTensor> make_a7(const DeviceTensor& h1e, const DeviceTensor& h2e,
                                              const DeviceTensor& dm1, const DeviceTensor& dm2,
                                              const DeviceTensor& dm3,
                                              const DeviceResources& dr);
DeviceTensor make_a9(const DeviceTensor& h1e, const DeviceTensor& h2e,
                     const DeviceTensor& hdm1, const DeviceTensor& hdm2,
                     const DeviceTensor& hdm3, const DeviceResources& dr);
DeviceTensor make_a12(const DeviceTensor& h1e, const DeviceTensor& h2e,
                      const DeviceTensor& dm1, const DeviceTensor& dm2,
                      const DeviceTensor& dm3, const DeviceResources& dr);
DeviceTensor make_a13(const DeviceTensor& h1e, const DeviceTensor& h2e,
                      const DeviceTensor& dm1, const DeviceTensor& dm2,
                      const DeviceTensor& dm3, const DeviceResources& dr);
// For PC Sr, Si: PySCF's make_a16 / make_a22 ((n)^6, `pqrabc` /
// `ijkabc`; they read the f3ac/f3ca digests) and the hole-side metrics
// energy_Si builds, dm3_h (n^6) = 2 dm2[a,b,e,f] delta[c,d] -
// dm3[a,b,d,c,e,f] and dm2_h (n^4) = 2 dm1[a,b] delta[c,d] - dm2[a,b,d,c].
DeviceTensor make_a16(const DeviceTensor& h1e, const DeviceTensor& h2e, const DeviceTensor& dm3,
                      const DeviceTensor& f3ac, const DeviceTensor& f3ca, int64_t norb,
                      const DeviceResources& dr);
DeviceTensor make_a22(const DeviceTensor& h1e, const DeviceTensor& h2e, const DeviceTensor& dm2,
                      const DeviceTensor& dm3, const DeviceTensor& f3ac,
                      const DeviceTensor& f3ca, int64_t norb, const DeviceResources& dr);
DeviceTensor make_dm3_h(const DeviceTensor& dm2, const DeviceTensor& dm3,
                        const DeviceResources& dr);
DeviceTensor make_dm2_h(const DeviceTensor& dm1, const DeviceTensor& dm2,
                        const DeviceResources& dr);
// The Sijrs class, {norm, energy}: no active index, so PC's Sijrs is exactly
// this (block2 agrees to 1e-19).
std::pair<double, double> energy_Sijrs(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                       int64_t batch, const DeviceResources& dr);

}  // namespace nevpt2
