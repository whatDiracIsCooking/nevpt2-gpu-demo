// nevpt2.energy:sc_finish -- the strongly-contracted finish: a slab's norm/H
// read back and turned into -sum N/(Delta+H/N) (PySCF's _norm_to_energy),
// and the orbital-energy slice it is taken against.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :sc_finish;` and from no importer of nevpt2.energy.
// (Used by energy_sc_classes.cppm and energy.cpp's Sijrs.)
module nevpt2.energy:sc_finish;

import std;
import nevpt2.energy;

namespace nevpt2 {

// pyscf.mrpt.nevpt2._norm_to_energy verbatim: the strongly-contracted class
// energy is -sum_k N_k / (Delta_k + H_k/N_k) over perturbers k with
// non-vanishing norm; the returned norm is the plain sum of N_k.
std::pair<double, double> normToEnergy(const std::vector<double>& norm,
                                        const std::vector<double>& h,
                                        const std::vector<double>& diff) {
  double normT = 0.0, enerT = 0.0;
  for (std::size_t k = 0; k < norm.size(); ++k) {
    normT += norm[k];
    if (std::fabs(norm[k]) > NUMERICAL_ZERO) {
      enerT -= norm[k] / (diff[k] + h[k] / norm[k]);
    }
  }
  return {normT, enerT};
}

// --- slab plumbing ------------------------------------------------------------

// forBatches, the slab walk, is in energy_shared.cppm (nevpt2.energy:shared;
// energy_pc.cpp walks the same slabs).

// host[b0:b1) of a rank-1 host tensor (an orbital-energy vector).
std::vector<double> sliceHost(const Tensor& v, int64_t b0, int64_t b1) {
  return std::vector<double>(v.data().begin() + b0, v.data().begin() + b1);
}

// Reads a class's per-slab norm/H back. No synchronization before the
// downloads, in any class (nevpt2.energy:sc_classes, and Sijrs in energy.cpp): each download is enqueued on the same stream
// as the kernels that wrote its source, behind them, and downloadTensor
// synchronizes that stream before the host reads the result. (No extra sync
// guards against the stream-less-download race in docs/performance.md, "One
// stream": nothing is stream-less, and stream-lint.sh --strict keeps it that
// way.) norm/H are freed by their owners, at the end of the slab's iteration.
std::pair<double, double> finish(const DeviceTensor& norm, const DeviceTensor& h,
                                 const std::vector<double>& diff, const DeviceResources& dr) {
  const wwrStream_t s = dr.stream();
  return normToEnergy(downloadTensor(norm, s).data(), downloadTensor(h, s).data(), diff);
}

}  // namespace nevpt2
