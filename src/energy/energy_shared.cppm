// nevpt2.energy:shared -- what the SC (energy.cpp) and PC (energy_pc.cpp)
// classes share and no importer of nevpt2.energy sees: the Sijrs class, which
// PC reuses whole. An internal partition: nothing here is exported, so this
// has module linkage -- reachable from every unit that does `import :shared;`
// and from no importer of nevpt2.energy. energy.cpp defines it. (The slab
// walk, forBatches, is nevpt2.energy_finish's.) It imports the primary
// interface for the types its declaration names (ActiveIntegralsDevice,
// IntegralSource); the primary interface does not import it back
// (cmake/README.md, "add_cxx_module_library").
//
// The RDM-only intermediates both methods build their metrics and Dyall
// Hamiltonians from (make_hdm1, make_a3, make_k27, make_hdm2, make_hdm3,
// make_a7, make_a9, make_a12, make_a13, make_a16, make_a22, make_dm3_h,
// make_dm2_h) used to be declared here too. They are EXPORTED now, from
// energy.cppm ("the class metrics and Dyall Hamiltonians"): the SC gradient's
// pseudodensity assembly (nevpt2.gradient) differentiates the very einsums
// the class energies evaluate, so it needs the same intermediates, and it is
// a separate module. energy.cpp still defines them, unchanged.
module nevpt2.energy:shared;

import std;
import nevpt2.energy;

namespace nevpt2 {

// forBatches, the slab walk every class runs, is nevpt2.energy_finish's
// (re-exported by nevpt2.energy).

// The Sijrs class, {norm, energy}: no active index, so PC's Sijrs IS exactly
// this (block2 agrees to 1e-19).
std::pair<double, double> energy_Sijrs(const ActiveIntegralsDevice& ai, IntegralSource& src,
                                       int64_t batch, const DeviceResources& dr);

}  // namespace nevpt2
