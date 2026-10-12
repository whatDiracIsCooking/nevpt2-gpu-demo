// nevpt2.gradsrc -- the `.gradsrc` file: the GPU's own SC-NEVPT2 gradient
// source terms, Y_xy = dE/dkappa_xy and y_I = dE/dc_I (Park Eqs. 26-27), as a
// NumPy `.npz` that reference_data/nevpt2_grad.py --source dump reads with a
// bare numpy.load (its load_source_terms; docs/reference-data.md, "The
// analytic gradient").
//
// WHY IT LIVES IN apps/ AND NOT src/. nevpt2.gradient is a library that
// computes; deciding to write what it computed to a file is the binary's
// business (#73). So this is the one module library under apps/: declared by
// apps/sc_pseudodensity/CMakeLists.txt beside the binary that writes it
// (nevpt2_sc_pseudodensity --dump-source-terms), and a library at all only so
// the unit tier (test/gradsrc/) can round-trip it without a card.
//
// THE FORMAT is NumPy's own, written by hand so no compression library or
// Python enters the build: a zip archive of STORED (uncompressed) `<name>.npy`
// members, each an NPY v1.0 file -- magic, header dict, C-order little-endian
// data. Every member is either '<f8' or a 0-d '<i8'. `readGradSrc` reads back
// exactly that and refuses anything else (a compressed member, a Fortran-order
// array, another dtype) as an IO Error; it is the round trip's other half, not
// a general .npz reader -- numpy.load is the consumer that matters.
//
// The members, and what nevpt2_grad.py does with each:
//
//   kappa_source  (n_mo, n_mo)   Y over the WHOLE kappa space, antisymmetric:
//                                Y_pq = (1/2) dE/dt along kappa = E_pq - E_qp
//                                (nevpt2.gradient's scFullRotationMatrix
//                                convention), so Y - Y^T is the derivative
//                                along that rotation -- the only combination
//                                the reader takes
//   ci_source     ci's shape     y_I = dE/dc_I, unprojected; the reader takes
//                                P y itself
//   ci            ci's shape     the CI vector both were taken at: the golden's,
//                                in the golden's (natural-orbital) active
//                                basis. The reader carries both terms into its
//                                own rotated basis and fixes the state's sign
//                                against this
//   ncore, nact, nvirt  0-d i8   the partition, checked against the state
//   e_corr        0-d f8         the SC total the terms differentiate
module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

export module nevpt2.gradsrc;

import std;
export import nevpt2.error_handling;
export import nevpt2.tensor;

export namespace nevpt2 {

// One `.gradsrc`'s contents. The three tensors are host copies; the shapes are
// the caller's (kappaSource square, ciSource and ci the same shape).
struct GradSrc {
  Tensor kappaSource;
  Tensor ciSource;
  Tensor ci;
  int64_t ncore = 0;
  int64_t nact = 0;
  int64_t nvirt = 0;
  double eCorr = 0.0;
};

// Writes `src` to `path`. A shape that does not fit -- kappaSource not
// (n_mo, n_mo) for the partition, ciSource and ci not the same shape -- is the
// caller's bug (check()); a file that cannot be opened or written is an IO
// Error.
Status writeGradSrc(const std::string& path, const GradSrc& src);

// Reads a `.gradsrc` as writeGradSrc writes one. A missing or unreadable file,
// a corrupt archive (bad signature, CRC mismatch, truncation), a member it does
// not write, or a missing member is an IO Error naming the file.
Result<GradSrc> readGradSrc(const std::string& path);

}  // namespace nevpt2
