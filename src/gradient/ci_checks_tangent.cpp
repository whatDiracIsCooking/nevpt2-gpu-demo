// nevpt2.ci_checks, --sc-tangent: the forward tangent of the SC-NEVPT2 total
// with respect to the CI vector, checked against the class norms' exact affine
// increment, Euler's identity and a central finite difference. Both demos run
// it -- nevpt2_demo over FullBlockSource, nevpt2_df_demo over its
// DfIntegralSource slabs (#162); ci_checks.cppm has the interface. And
// --sc-hessian (#153, nevpt2_demo): the same three kinds of check one order
// up, for the second directional derivative -- at the end of this file.
//
// WHAT IS BEING DIFFERENTIATED. Along a direction u in the space of CI
// amplitudes, the state c + t u gives RDMs
//
//   dm1/dm2  (nevpt2.rdm_host_dm, host)   dm3/f3ac/f3ca  (nevpt2.rdm_build)
//
// each a bilinear form in the CI vector -- NOT normalized, by the same
// convention nevpt2.rdm_host_dm states: these are the plain forms, which is
// also what the reverse-mode (adjoint) assembly will differentiate, since the
// normalization constraint is carried by the Lagrangian and not by the energy
// expression. The RDM tangents along u are exact and need no step size
// (nevpt2.rdm_host_dm's dm12Dot, nevpt2.rdm_build's buildRdmTangentsDevice),
// and nevpt2.energy's energiesTangentDevice turns them into Edot(u).
//
// CHECK 1a, EXACT, EVERY CLASS: THE NORM IS AFFINE IN THE RDMs. A class norm
// is the external integrals contracted twice against a metric, and every
// metric is affine in the RDMs, so for ANY displacement X of the RDMs
//
//   norm_k(dm + X) - norm_k(dm) == normDot_k(X)       exactly,
//
// with no step size and no truncation error -- the increment of an affine map
// IS its linear part. The left side is two runs of the unmodified
// energiesDevice (whose per-class norms it already reports) and the right one
// energiesTangentDevice run; X is the RDM tangent along the direction below.
// That pins down every metric tangent builder, every norm chain run with
// tangent RDMs, and the RDM-free terms the tangent must DROP (Sir's
// one-electron norm term, and Sijrs, which reads no RDM at all and so has
// normDot = energyDot = 0 identically).
//
// CHECK 1b, EXACT, THREE CLASSES: EULER. Sr, Srsi and Srs are the classes
// whose metric AND Dyall-Hamiltonian blocks are purely LINEAR in the RDMs
// (dm1/dm2/dm3 and a16/a17/a19, dm1 and k27, rm2 and a7), so their N and H --
// and hence their energy -- are homogeneous of degree 2 in the CI vector.
// Taking the direction to be the state itself must therefore give
//
//   Edot_k(c) = 2 E_k      and      Ndot_k(c) = 2 N_k
//
// for those three, again to round-off. The other five are AFFINE, not
// homogeneous, and no such identity holds for them: Si's a25 carries a
// `2 h1e^T` term and its dm1_h a `2 delta`, Sijr's hdm1 and a3 the same
// `2 delta`, Sij's hdm2/hdm3/a9 their delta-delta terms, Sir's norm its
// one-electron term, and Sijrs reads no RDM. They are printed for the record
// and checked by 1a and by the difference below. energiesTangentDevice's own
// base energies against energiesDevice's come along for free.
//
// CHECK 2, THE ONE-TIME FINITE DIFFERENCE. The total is a RATIO of quadratics
// in the CI vector (the finish divides), so a central difference along the
// straight line c +- h u has a genuine O(h^2) truncation error -- unlike the
// RDMs themselves, where a straight-line difference is exact (see
// test/rdm_host_dm/). Each difference rebuilds the RDMs from scratch at c +- h
// u and reruns energiesDevice, so nothing of the tangent path is involved in
// it. The table below sweeps h and prints the error against Edot(u) and the
// ratio across each halving: 4 while truncation dominates, then a plateau and
// a rise once round-off in E / 2h takes over.
//
// WHICH FUNCTIONAL. The golden's e_core / e_virt, held fixed: the energy is
// rerun at displaced RDMs only, and the tangent runs without the orbital
// energies' tangent (epsDot = null). That is the functional this check was
// built to license. The orbital energies' own response (#87), which
// --gradient-source adds to both of its modes given the sidecar, is checked
// elsewhere -- card-free in test/gradient_plan/ and test/energy_finish/, and
// against PySCF's exact derivative (docs/gradients.md, Sec. 5.5).
//
// THIS IS A DEVELOPMENT CROSS-CHECK, run once to license the EXACT tests
// downstream: the finite difference validates the tangent, and the tangent
// (not a step size) then validates the adjoint, at this tree's usual 1e-7.
// docs/testing.md, "The SC tangent's finite-difference cross-check", records
// the measured plateau and the h where round-off takes over.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry

module nevpt2.ci_checks;

import std;
import nevpt2.rdm_host_dm;

namespace nevpt2 {

// The step sweep: h = 2^-k, halving, from 2^-6 (a 1.6% displacement, coarse
// enough that truncation dominates) down to 2^-19 (past the round-off floor
// of differencing two independently rebuilt energies at every CAS size this
// runs on). Powers of two so that h is exact and the ratio across a halving
// is exactly the Richardson factor.
constexpr int kFirstStepExp = 6;
constexpr int kLastStepExp = 19;
constexpr std::size_t kStepCount = kLastStepExp - kFirstStepExp + 1;

// The exact checks' tolerance -- the affine increment, Euler's identity and
// the two paths' base energies are algebraic, so this is numerical noise, not
// a discretization error (the RDM build reassociates its sums, so nothing is
// bit-identical). The same 1e-10 the SC pseudodensity identities use, four
// decades above what is measured (docs/testing.md, "The SC tangent's
// finite-difference cross-check").
constexpr double kTangentExactTol = 1e-10;
// The finite difference's best relative error over the sweep, |err| / |Edot|.
// A central difference cannot do better than its own round-off floor, which
// is why the PERMANENT tests downstream difference nothing; the measured
// plateau is 6e-9 relative at CAS(4,4) and 2e-8 at CAS(8,8), so this leaves
// a factor of ~50 (docs/testing.md, "The SC tangent's finite-difference
// cross-check").
constexpr double kTangentFdTol = 1e-6;
// The Richardson window is nevpt2.gradient_plan's kFourfoldLow / kFourfoldHigh,
// the one every finite-difference check of the gradient path shares.
// How many CONSECUTIVE halvings must show that ratio somewhere in the sweep.
// Not the whole table: differencing the energy at a small displacement has an
// intrinsic hazard that can spoil an individual step. A perturber that is
// EMPTY at the base point (N_k = H_k = 0 -- N2's symmetry gives plenty) has
// both of them growing as h^2 when the state is displaced, so its energy
// -N^2/(Delta N + H) is O(h^2) -- even in h, and invisible to a central
// difference -- but N and H are then tiny differences of O(1) contractions,
// i.e. round-off, and at a step where Delta N + H happens to cancel to
// nothing the term is numerically enormous. That is a property of
// differencing the energy, not of the tangent (which skips those perturbers
// exactly as normToEnergy does), and it shows up as a single wild row.
constexpr int kRichardsonRun = 4;

// x scaled so <x|x> = 1.
nevpt2::Tensor tangentNormalized(nevpt2::Tensor x) {
  double norm2 = 0.0;
  for (std::int64_t i = 0; i < x.size(); ++i) norm2 += x.flat(i) * x.flat(i);
  const double inv = 1.0 / std::sqrt(norm2);
  for (std::int64_t i = 0; i < x.size(); ++i) x.flatRef(i) *= inv;
  return x;
}

// A seeded random direction in ci's space, projected against ci and
// normalized -- test/rdm_host_dm/'s `direction`, the same construction. The
// projection is not needed for the difference (the straight line is
// differenced as it stands); it keeps ||c + h u|| = 1 + O(h^2), so the step
// stays a rotation of the state to first order.
nevpt2::Tensor tangentDirection(const nevpt2::Tensor& ci, std::uint64_t seed) {
  nevpt2::Tensor u(ci.dims());
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::int64_t i = 0; i < u.size(); ++i) u.flatRef(i) = dist(gen);
  double overlap = 0.0;
  for (std::int64_t i = 0; i < u.size(); ++i) overlap += ci.flat(i) * u.flat(i);
  for (std::int64_t i = 0; i < u.size(); ++i) u.flatRef(i) -= overlap * ci.flat(i);
  return tangentNormalized(std::move(u));
}

// ci + h u, same shape.
nevpt2::Tensor tangentStep(const nevpt2::Tensor& ci, double h, const nevpt2::Tensor& u) {
  nevpt2::Tensor out = ci;
  for (std::int64_t i = 0; i < out.size(); ++i) out.flatRef(i) += h * u.flat(i);
  return out;
}

// The SC total at one CI vector, the whole way from the vector: dm1/dm2
// rebuilt on the host, dm3/f3ac/f3ca built on the device, then the eight
// class energies. No tangent code is involved, which is what makes it a
// reference for one. Every device tensor it allocates is freed (on res's
// stream) when it returns.
double scTotalAt(const nevpt2::HostDm12Builder& builder, const nevpt2::Tensor& ci,
                 const nevpt2::RdmBuildOptions& rdmOpt, const CiCheckIntegrals& ints,
                 const nevpt2::DeviceResources& res) {
  const nevpt2::HostDm12 dm = builder.dm12(ci);
  const nevpt2::DeviceTensor dDm1 = nevpt2::uploadTensor(dm.dm1, res);
  const nevpt2::DeviceTensor dDm2 = nevpt2::uploadTensor(dm.dm2, res);
  const nevpt2::RdmBuildResult rdm = nevpt2::buildRdmsDevice(
      rdmOpt, ci, builder.norb(), builder.nelecA(), builder.nelecB(), ints.h2e, res);
  return nevpt2::energiesDevice(ints.active, ints.source, ints.batch, dDm1, dDm2, rdm.dm3,
                                rdm.f3ac, rdm.f3ca, res)
      .total;
}

// a + b, same shape, as a fresh device tensor: a no-permutation transpose
// into a new tensor and another accumulated into it, which is how this tree
// adds two device tensors (nevpt2.gradient's addInto does the same).
nevpt2::DeviceTensor tangentSum(const nevpt2::DeviceTensor& a, const nevpt2::DeviceTensor& b,
                                const nevpt2::DeviceResources& res) {
  std::vector<int> axes(static_cast<std::size_t>(a.rank()));
  for (std::size_t i = 0; i < axes.size(); ++i) axes[i] = static_cast<int>(i);
  nevpt2::DeviceTensor out = nevpt2::deviceTransposeNew(a, axes, 1.0, res);
  nevpt2::deviceTransposeAccum(b, axes, out, 1.0, res.stream());
  return out;
}

// max |a - b| over two same-shaped host tensors.
double tangentMaxDiff(const nevpt2::Tensor& a, const nevpt2::Tensor& b) {
  if (a.dims() != b.dims()) return std::numeric_limits<double>::infinity();
  double m = 0.0;
  for (std::int64_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a.flat(i) - b.flat(i)));
  return m;
}

// Runs both checks for one golden state and `seed`'s direction, prints the
// per-class Euler table and the finite-difference sweep, and returns whether
// every criterion above holds. An active space the host RDM builder refuses
// is a configuration Error (the caller reports it once); everything else here
// is a number.
nevpt2::Result<bool> checkScTangent(const nevpt2::GoldenFile& g,
                                    const nevpt2::RdmBuildOptions& rdmOpt,
                                    const CiCheckIntegrals& ints,
                                    const nevpt2::RdmBuildResult& baseRdm, std::uint64_t seed,
                                    const nevpt2::DeviceResources& res) {
  nevpt2::Result<nevpt2::HostDm12Builder> builder =
      nevpt2::HostDm12Builder::create(g.ncas, g.nelecA, g.nelecB);
  if (!builder) return std::unexpected(builder.error());
  const nevpt2::Tensor& ci = g.get("ci");
  // The active block the RDM builds contract: the golden's own, or the
  // density-fitted demo's rebuild from B_aa -- whichever built baseRdm.
  const nevpt2::Tensor& h2e = ints.h2e;
  const std::int64_t norb = g.ncas;

  std::printf("\n--sc-tangent: the SC total's tangent with respect to the CI vector\n");

  // The host rebuild IS the dm1/dm2 every energy below reads, at the base
  // point and at every displaced one, so the whole check is internally
  // consistent; this line is how far it sits from the golden's own arrays
  // (PySCF's, which the unit tier holds it to 1e-12 against).
  const nevpt2::HostDm12 base = builder->dm12(ci);
  std::printf("  host dm1/dm2 rebuild vs the golden: max|delta| = %.2e / %.2e\n",
              tangentMaxDiff(base.dm1, g.get("dm1")), tangentMaxDiff(base.dm2, g.get("dm2")));

  const nevpt2::DeviceTensor dDm1 = nevpt2::uploadTensor(base.dm1, res);
  const nevpt2::DeviceTensor dDm2 = nevpt2::uploadTensor(base.dm2, res);
  // dm3/f3ac/f3ca at the base point are the build the caller already ran on this
  // very CI vector with these very options; only the displaced points below
  // need builds of their own.
  const nevpt2::RdmBuildResult& rdm = baseRdm;

  bool ok = true;

  // --- check 1b: Euler, with the direction equal to the state -----------------
  {
    const nevpt2::HostDm12 dot = builder->dm12Dot(ci, ci);
    const nevpt2::DeviceTensor dDm1Dot = nevpt2::uploadTensor(dot.dm1, res);
    const nevpt2::DeviceTensor dDm2Dot = nevpt2::uploadTensor(dot.dm2, res);
    const nevpt2::RdmBuildResult rdmDot =
        nevpt2::buildRdmTangentsDevice(rdmOpt, ci, ci, norb, g.nelecA, g.nelecB, h2e, res);
    const nevpt2::EnergyTangentResult t = nevpt2::energiesTangentDevice(
        ints.active, ints.source, ints.batch, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, dDm1Dot,
        dDm2Dot, rdmDot.dm3, rdmDot.f3ac, rdmDot.f3ca, /*epsDot=*/nullptr, res);
    const nevpt2::EnergyResult e = nevpt2::energiesDevice(
        ints.active, ints.source, ints.batch, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, res);

    // The tangent path recomputes the class energies on its way to the
    // derivative; they must be energiesDevice's own.
    const double dBase = std::fabs(t.total - e.total);
    ok = ok && dBase < kTangentExactTol;
    std::printf(
        "\n  base total: tangent path %.12f vs energiesDevice %.12f  |delta| = %.2e\n",
        t.total, e.total, dBase);

    std::printf("\n  Euler (direction = the state): Edot_k = 2 E_k and Ndot_k = 2 N_k, for the\n"
                "  three classes whose metric and Dyall blocks are purely linear in the RDMs\n");
    std::printf("  %-7s %6s %16s %16s %10s %16s %16s %10s\n", "class", "linear", "Edot",
                "2 E_k", "|delta|", "Ndot", "2 N_k", "|delta|");
    for (int c = 0; c < 8; ++c) {
      // Sr, Srsi, Srs (indices 0, 4, 5 of CLASSES): see "CHECK 1b" above for
      // why the other five carry an RDM-free term and so are not
      // homogeneous. Sijrs (index 2) reads no RDM at all, so its tangent is
      // exactly zero -- checked for every direction, here and below.
      const bool linear = c == 0 || c == 4 || c == 5;
      const bool sijrs = c == 2;
      const double wantE = 2.0 * t.energies[c], wantN = 2.0 * t.norms[c];
      const double dE = std::fabs(t.energyDots[c] - wantE);
      const double dN = std::fabs(t.normDots[c] - wantN);
      if (linear) ok = ok && dE < kTangentExactTol && dN < kTangentExactTol;
      if (sijrs)
        ok = ok && t.energyDots[c] == 0.0 && t.normDots[c] == 0.0;
      std::printf("  %-7s %6s %16.10f %16.10f %10.2e %16.8f %16.8f %10.2e\n",
                  nevpt2::CLASSES[c].c_str(), linear ? "yes" : (sijrs ? "none" : "affine"),
                  t.energyDots[c], wantE, dE, t.normDots[c], wantN, dN);
    }
    std::printf("  Euler identity (the three linear classes, and Sijrs exactly zero): %s\n",
                ok ? "ok" : "FAILED");
  }

  // --- check 1a / 2: one direction, the affine increment and the difference ---
  const nevpt2::Tensor u = tangentDirection(ci, seed);
  const nevpt2::HostDm12 dirDot = builder->dm12Dot(ci, u);
  const nevpt2::DeviceTensor dDm1Dot = nevpt2::uploadTensor(dirDot.dm1, res);
  const nevpt2::DeviceTensor dDm2Dot = nevpt2::uploadTensor(dirDot.dm2, res);
  const nevpt2::RdmBuildResult rdmDot =
      nevpt2::buildRdmTangentsDevice(rdmOpt, ci, u, norb, g.nelecA, g.nelecB, h2e, res);
  const nevpt2::EnergyTangentResult t = nevpt2::energiesTangentDevice(
      ints.active, ints.source, ints.batch, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, dDm1Dot,
      dDm2Dot, rdmDot.dm3, rdmDot.f3ac, rdmDot.f3ca, /*epsDot=*/nullptr, res);

  std::printf("\n  Edot(u) = %.12f  (seed 0x%llx, |u| = 1, <c|u> = 0)\n", t.totalDot,
              static_cast<unsigned long long>(seed));
  std::printf("  per class:");
  for (int c = 0; c < 8; ++c) std::printf(" %s=%.6f", nevpt2::CLASSES[c].c_str(), t.energyDots[c]);
  std::printf("\n");

  // --- check 1a: the class norms are affine in the RDMs -----------------------
  {
    // The displaced RDMs dm + X, X being the tangent RDMs along u. Only the
    // NORMS of the run below are read: its energies are the energy at a set
    // of RDMs that is not a state's, and mean nothing.
    const nevpt2::DeviceTensor dm1Plus = tangentSum(dDm1, dDm1Dot, res);
    const nevpt2::DeviceTensor dm2Plus = tangentSum(dDm2, dDm2Dot, res);
    const nevpt2::DeviceTensor dm3Plus = tangentSum(rdm.dm3, rdmDot.dm3, res);
    const nevpt2::DeviceTensor f3acPlus = tangentSum(rdm.f3ac, rdmDot.f3ac, res);
    const nevpt2::DeviceTensor f3caPlus = tangentSum(rdm.f3ca, rdmDot.f3ca, res);
    const nevpt2::EnergyResult shifted = nevpt2::energiesDevice(
        ints.active, ints.source, ints.batch, dm1Plus, dm2Plus, dm3Plus, f3acPlus, f3caPlus,
        res);

    std::printf("\n  the class norms are affine in the RDMs: norm_k(dm + X) - norm_k(dm) = "
                "Ndot_k(X), exactly\n");
    std::printf("  %-7s %16s %16s %10s\n", "class", "increment", "Ndot", "|delta|");
    for (int c = 0; c < 8; ++c) {
      const double increment = shifted.norms[c] - t.norms[c];
      const double d = std::fabs(increment - t.normDots[c]);
      ok = ok && d < kTangentExactTol;
      std::printf("  %-7s %16.10f %16.10f %10.2e\n", nevpt2::CLASSES[c].c_str(), increment,
                  t.normDots[c], d);
    }
  }

  std::printf("\n  central differences of energiesDevice along c + h u:\n");
  std::printf("  %5s %14s %20s %12s %12s %8s\n", "k", "h = 2^-k", "(E(+h)-E(-h))/2h", "|err|",
              "rel err", "ratio");
  std::array<double, kStepCount> step{};
  std::array<double, kStepCount> err{};
  for (std::size_t i = 0; i < kStepCount; ++i) {
    const int k = kFirstStepExp + static_cast<int>(i);
    const double h = std::ldexp(1.0, -k);
    step[i] = h;
    const double plus = scTotalAt(*builder, tangentStep(ci, h, u), rdmOpt, ints, res);
    const double minus = scTotalAt(*builder, tangentStep(ci, -h, u), rdmOpt, ints, res);
    const double d = (plus - minus) / (2.0 * h);
    err[i] = std::fabs(d - t.totalDot);
    const double rel = err[i] / std::max(1e-30, std::fabs(t.totalDot));
    if (i == 0) {
      std::printf("  %5d %14.3e %20.12f %12.3e %12.3e %8s\n", k, h, d, err[i], rel, "-");
    } else {
      std::printf("  %5d %14.3e %20.12f %12.3e %12.3e %8.2f\n", k, h, d, err[i], rel,
                  err[i] > 0.0 ? err[i - 1] / err[i] : 0.0);
    }
  }

  // The plateau: the smallest error of the sweep, and the step that reached
  // it -- below that h, round-off in E / 2h grows faster than h^2 falls.
  std::size_t bestAt = 0;
  for (std::size_t i = 1; i < kStepCount; ++i)
    if (err[i] < err[bestAt]) bestAt = i;
  const double bestRel = err[bestAt] / std::max(1e-30, std::fabs(t.totalDot));
  ok = ok && bestRel < kTangentFdTol;
  std::printf("\n  plateau: |err| = %.3e (relative %.3e) at h = %.3e (2^-%d); round-off takes "
              "over below it\n",
              err[bestAt], bestRel, step[bestAt], kFirstStepExp + static_cast<int>(bestAt));

  // Second order: the longest run of consecutive halvings that cut the error
  // fourfold (see kRichardsonRun for why a run and not the whole table).
  const nevpt2::FourfoldRun run = nevpt2::longestFourfoldRun(err);
  const bool richardson = run.length >= kRichardsonRun;
  ok = ok && richardson;
  std::printf("  Richardson: %d consecutive halvings cut the error fourfold (ratio in [%.1f, "
              "%.1f]), ending at h = %.3e -- %s, %d needed\n",
              run.length, nevpt2::kFourfoldLow, nevpt2::kFourfoldHigh, step[run.end],
              richardson ? "ok" : "NOT ENOUGH", kRichardsonRun);

  std::printf("\n%s: the SC tangent matches Euler and the exact norm increment to %.0e, and "
              "the finite difference to %.0e relative\n",
              ok ? "PASS" : "FAIL", kTangentExactTol, kTangentFdTol);
  return ok;
}

// =============================================================================
// --sc-hessian (#153): the SECOND directional derivative, Eddot(u, v).
//
// WHAT IS BEING DIFFERENTIATED: the same functional as --sc-tangent's above
// (plain bilinear RDMs, the golden's e_core / e_virt held fixed), twice.
// nevpt2.energy's energiesHessianDevice reads four RDM sets -- the plain one,
// the tangents along u and along v, and the second-order set at (u, v), all
// three tangent sets from ONE three-argument build
// (nevpt2.rdm_build's buildRdmHessianTangentsDevice; dm1/dm2's from the host
// rebuild's dm12Dot, whose (u, v) form is the second derivative for the same
// reason, docs/hessian-theory.md Sec. 3).
//
// CHECK H1, EXACT, THREE CLASSES: EULER AT SECOND ORDER. A class homogeneous
// of degree 2 in the CI vector has a gradient homogeneous of degree 1, so its
// Hessian maps the state to the gradient, H c = grad E, and
//
//   Eddot_k(c, u) == Edot_k(u)
//
// for Sr, Srsi and Srs (CHECK 1b's three) -- the new 3x3 and the second-order
// chains held against the ALREADY-GATED Edot(u) of energiesTangentDevice,
// along a direction no seeded pair reaches (Sec. 4.3). The reference is that
// function on its own tangent build, not the Hessian walk's u-tangent. Like
// 1b, an identity at FIXED orbital energies only: the generalized Fock is
// affine in dm1, so a responding Delta breaks the homogeneity.
//
// CHECK H2, EXACT, EVERY CLASS: THE NORM'S SECOND-ORDER INCREMENT. The norms
// are affine in the RDMs (CHECK 1a), so at the second-order RDM set X'' the
// increment norm_k(dm + X'') - norm_k(dm) IS the second-order chain's
// N''_k -- the only exact hold on the five affine classes' second-order
// blocks, which have no homogeneity to exploit (Sec. 4.3). Sijrs reads no RDM,
// so its every second derivative is exactly zero.
//
// CHECK H3, THE ONE FINITE DIFFERENCE: Edot(u) itself -- energiesTangentDevice
// on a fresh plain build and a fresh tangent build at every displaced state
// c +- h v -- differenced along v against Eddot(u, v). The milestone's single
// new difference (docs/hessian-theory.md, Sec. 9): it licenses Eddot as
// Edot's derivative at all, after which the exact gates downstream (#156's
// duality) hold the vector form to it. Same ladder, plateau and Richardson
// rules as CHECK 2, for the same reasons (Edot is again a ratio of
// polynomials in the CI vector, with the same empty-perturber hazard).
//
// Also free, and checked: the Hessian walk's own base total and Edot(u) /
// Edot(v) against energiesDevice and energiesTangentDevice.
// docs/testing.md, "The SC Hessian's finite-difference cross-check", records
// the measured plateau.

// Edot(u) at the CI vector `ci`, the whole way from the vector: dm1/dm2 and
// their tangent rebuilt on the host, dm3/f3ac/f3ca and their tangent built on
// the device, then energiesTangentDevice. No Hessian code is involved, which
// is what makes it a reference for one. Every device tensor it allocates is
// freed (on res's stream) when it returns.
double scTangentAt(const nevpt2::HostDm12Builder& builder, const nevpt2::Tensor& ci,
                   const nevpt2::Tensor& u, const nevpt2::RdmBuildOptions& rdmOpt,
                   const CiCheckIntegrals& ints, const nevpt2::DeviceResources& res) {
  const nevpt2::HostDm12 dm = builder.dm12(ci);
  const nevpt2::HostDm12 dot = builder.dm12Dot(ci, u);
  const nevpt2::DeviceTensor dDm1 = nevpt2::uploadTensor(dm.dm1, res);
  const nevpt2::DeviceTensor dDm2 = nevpt2::uploadTensor(dm.dm2, res);
  const nevpt2::DeviceTensor dDm1Dot = nevpt2::uploadTensor(dot.dm1, res);
  const nevpt2::DeviceTensor dDm2Dot = nevpt2::uploadTensor(dot.dm2, res);
  const nevpt2::RdmBuildResult rdm = nevpt2::buildRdmsDevice(
      rdmOpt, ci, builder.norb(), builder.nelecA(), builder.nelecB(), ints.h2e, res);
  const nevpt2::RdmBuildResult rdmDot = nevpt2::buildRdmTangentsDevice(
      rdmOpt, ci, u, builder.norb(), builder.nelecA(), builder.nelecB(), ints.h2e, res);
  return nevpt2::energiesTangentDevice(ints.active, ints.source, ints.batch, dDm1, dDm2,
                                       rdm.dm3, rdm.f3ac, rdm.f3ca, dDm1Dot, dDm2Dot,
                                       rdmDot.dm3, rdmDot.f3ac, rdmDot.f3ca,
                                       /*epsDot=*/nullptr, res)
      .totalDot;
}

nevpt2::Result<bool> checkScHessian(const nevpt2::GoldenFile& g,
                                    const nevpt2::RdmBuildOptions& rdmOpt,
                                    const CiCheckIntegrals& ints,
                                    const nevpt2::RdmBuildResult& baseRdm, std::uint64_t seedU,
                                    std::uint64_t seedV, const nevpt2::DeviceResources& res) {
  nevpt2::Result<nevpt2::HostDm12Builder> builder =
      nevpt2::HostDm12Builder::create(g.ncas, g.nelecA, g.nelecB);
  if (!builder) return std::unexpected(builder.error());
  const nevpt2::Tensor& ci = g.get("ci");
  const nevpt2::Tensor& h2e = ints.h2e;
  const std::int64_t norb = g.ncas;

  std::printf("\n--sc-hessian: the SC total's second derivative with respect to the CI vector\n");
  // Which functional, said where the Euler gate is stated (docs/hessian-theory.md,
  // Sec. 9.1): the identity holds only with the orbital energies fixed.
  std::printf("  functional: the orbital energies held fixed (the golden's e_core / e_virt; "
              "Euler holds only there)\n");

  const nevpt2::HostDm12 base = builder->dm12(ci);
  std::printf("  host dm1/dm2 rebuild vs the golden: max|delta| = %.2e / %.2e\n",
              tangentMaxDiff(base.dm1, g.get("dm1")), tangentMaxDiff(base.dm2, g.get("dm2")));
  const nevpt2::DeviceTensor dDm1 = nevpt2::uploadTensor(base.dm1, res);
  const nevpt2::DeviceTensor dDm2 = nevpt2::uploadTensor(base.dm2, res);
  const nevpt2::RdmBuildResult& rdm = baseRdm;
  const nevpt2::RdmTensors baseSet{dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca};

  const nevpt2::Tensor u = tangentDirection(ci, seedU);
  const nevpt2::Tensor v = tangentDirection(ci, seedV);
  bool ok = true;
  double eddot = 0.0;  // Eddot(u, v), the finite difference's target

  // --- Eddot(u, v), and check H2 at the second-order RDMs ---------------------
  // Scoped, so every RDM set of the three-argument build is freed before the
  // difference below allocates its own builds.
  {
    const nevpt2::HostDm12 dotU = builder->dm12Dot(ci, u);
    const nevpt2::HostDm12 dotV = builder->dm12Dot(ci, v);
    const nevpt2::HostDm12 ddot = builder->dm12Dot(u, v);
    const nevpt2::DeviceTensor dDm1U = nevpt2::uploadTensor(dotU.dm1, res);
    const nevpt2::DeviceTensor dDm2U = nevpt2::uploadTensor(dotU.dm2, res);
    const nevpt2::DeviceTensor dDm1V = nevpt2::uploadTensor(dotV.dm1, res);
    const nevpt2::DeviceTensor dDm2V = nevpt2::uploadTensor(dotV.dm2, res);
    const nevpt2::DeviceTensor dDm1UV = nevpt2::uploadTensor(ddot.dm1, res);
    const nevpt2::DeviceTensor dDm2UV = nevpt2::uploadTensor(ddot.dm2, res);
    const nevpt2::RdmHessianTangents t3 =
        nevpt2::buildRdmHessianTangentsDevice(rdmOpt, ci, u, v, norb, g.nelecA, g.nelecB, h2e,
                                              res);
    const nevpt2::RdmTensors setU{dDm1U, dDm2U, t3.cu.dm3, t3.cu.f3ac, t3.cu.f3ca};
    const nevpt2::RdmTensors setV{dDm1V, dDm2V, t3.cv.dm3, t3.cv.f3ac, t3.cv.f3ca};
    const nevpt2::RdmTensors setUV{dDm1UV, dDm2UV, t3.uv.dm3, t3.uv.f3ac, t3.uv.f3ca};
    const nevpt2::EnergyHessianResult hr = nevpt2::energiesHessianDevice(
        ints.active, ints.source, ints.batch, baseSet, setU, setV, setUV, res);
    eddot = hr.totalDDot;

    // Free: the walk's base total and its two first-order tangents are
    // energiesDevice's and energiesTangentDevice's own.
    const nevpt2::EnergyResult e = nevpt2::energiesDevice(
        ints.active, ints.source, ints.batch, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, res);
    const nevpt2::EnergyTangentResult tu = nevpt2::energiesTangentDevice(
        ints.active, ints.source, ints.batch, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, dDm1U,
        dDm2U, t3.cu.dm3, t3.cu.f3ac, t3.cu.f3ca, /*epsDot=*/nullptr, res);
    const nevpt2::EnergyTangentResult tv = nevpt2::energiesTangentDevice(
        ints.active, ints.source, ints.batch, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, dDm1V,
        dDm2V, t3.cv.dm3, t3.cv.f3ac, t3.cv.f3ca, /*epsDot=*/nullptr, res);
    const double dBase = std::fabs(hr.total - e.total);
    const double dU = std::fabs(hr.totalDotU - tu.totalDot);
    const double dV = std::fabs(hr.totalDotV - tv.totalDot);
    ok = ok && dBase < kTangentExactTol && dU < kTangentExactTol && dV < kTangentExactTol;
    std::printf("\n  the Hessian walk's own value and first derivatives against energiesDevice "
                "/ energiesTangentDevice:\n"
                "    E %.12f (|delta| %.2e)   Edot(u) %.12f (|delta| %.2e)   Edot(v) %.12f "
                "(|delta| %.2e)\n",
                hr.total, dBase, hr.totalDotU, dU, hr.totalDotV, dV);

    std::printf("\n  Eddot(u, v) = %.12f  (seeds 0x%llx / 0x%llx, |u| = |v| = 1, <c|u> = "
                "<c|v> = 0)\n",
                hr.totalDDot, static_cast<unsigned long long>(seedU),
                static_cast<unsigned long long>(seedV));

    // Check H2: the displaced RDMs dm + X'', X'' the second-order set. Only
    // the NORMS of the run below are read.
    const nevpt2::DeviceTensor dm1Plus = tangentSum(dDm1, dDm1UV, res);
    const nevpt2::DeviceTensor dm2Plus = tangentSum(dDm2, dDm2UV, res);
    const nevpt2::DeviceTensor dm3Plus = tangentSum(rdm.dm3, t3.uv.dm3, res);
    const nevpt2::DeviceTensor f3acPlus = tangentSum(rdm.f3ac, t3.uv.f3ac, res);
    const nevpt2::DeviceTensor f3caPlus = tangentSum(rdm.f3ca, t3.uv.f3ca, res);
    const nevpt2::EnergyResult shifted = nevpt2::energiesDevice(
        ints.active, ints.source, ints.batch, dm1Plus, dm2Plus, dm3Plus, f3acPlus, f3caPlus,
        res);
    std::printf("\n  per class, and the norms' second-order increment: norm_k(dm + X'') - "
                "norm_k(dm) = N''_k, exactly\n");
    std::printf("  %-7s %16s %16s %16s %16s %16s %10s\n", "class", "Edot(u)", "Edot(v)",
                "Eddot(u,v)", "increment", "N''", "|delta|");
    for (int c = 0; c < 8; ++c) {
      const double increment = shifted.norms[c] - hr.norms[c];
      const double d = std::fabs(increment - hr.normDDots[c]);
      ok = ok && d < kTangentExactTol;
      // Sijrs (index 2) reads no RDM: every second derivative exactly zero.
      if (c == 2) ok = ok && hr.energyDDots[c] == 0.0 && hr.normDDots[c] == 0.0;
      std::printf("  %-7s %16.10f %16.10f %16.10f %16.10f %16.10f %10.2e\n",
                  nevpt2::CLASSES[c].c_str(), hr.energyDotsU[c], hr.energyDotsV[c],
                  hr.energyDDots[c], increment, hr.normDDots[c], d);
    }
    std::printf("  norm increment (every class) and Sijrs exactly zero: %s\n",
                ok ? "ok" : "FAILED");
  }

  // --- check H1: Euler, Eddot(c, u) == Edot(u) ---------------------------------
  {
    const nevpt2::HostDm12 dotC = builder->dm12Dot(ci, ci);
    const nevpt2::HostDm12 dotU = builder->dm12Dot(ci, u);
    const nevpt2::DeviceTensor dDm1C = nevpt2::uploadTensor(dotC.dm1, res);
    const nevpt2::DeviceTensor dDm2C = nevpt2::uploadTensor(dotC.dm2, res);
    const nevpt2::DeviceTensor dDm1U = nevpt2::uploadTensor(dotU.dm1, res);
    const nevpt2::DeviceTensor dDm2U = nevpt2::uploadTensor(dotU.dm2, res);
    // (c, c), (c, u) and -- the second-order set at (c, u) -- (c, u) again:
    // the three-argument build at (ci, ci, u).
    const nevpt2::RdmHessianTangents t3 =
        nevpt2::buildRdmHessianTangentsDevice(rdmOpt, ci, ci, u, norb, g.nelecA, g.nelecB,
                                              h2e, res);
    const nevpt2::RdmTensors setC{dDm1C, dDm2C, t3.cu.dm3, t3.cu.f3ac, t3.cu.f3ca};
    const nevpt2::RdmTensors setU{dDm1U, dDm2U, t3.cv.dm3, t3.cv.f3ac, t3.cv.f3ca};
    const nevpt2::RdmTensors setCU{dDm1U, dDm2U, t3.uv.dm3, t3.uv.f3ac, t3.uv.f3ca};
    const nevpt2::EnergyHessianResult hr = nevpt2::energiesHessianDevice(
        ints.active, ints.source, ints.batch, baseSet, setC, setU, setCU, res);
    // The reference: the already-gated tangent, on its own tangent build.
    const nevpt2::RdmBuildResult rdmDot =
        nevpt2::buildRdmTangentsDevice(rdmOpt, ci, u, norb, g.nelecA, g.nelecB, h2e, res);
    const nevpt2::EnergyTangentResult t = nevpt2::energiesTangentDevice(
        ints.active, ints.source, ints.batch, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, dDm1U,
        dDm2U, rdmDot.dm3, rdmDot.f3ac, rdmDot.f3ca, /*epsDot=*/nullptr, res);

    std::printf("\n  Euler at second order: Eddot_k(c, u) = Edot_k(u), for the three classes "
                "whose\n  metric and Dyall blocks are purely linear in the RDMs\n");
    std::printf("  %-7s %6s %16s %16s %10s\n", "class", "linear", "Eddot(c,u)", "Edot(u)",
                "|delta|");
    bool euler = true;
    for (int c = 0; c < 8; ++c) {
      // Sr, Srsi, Srs (0, 4, 5): CHECK 1b's three; Sijrs (2) exactly zero.
      const bool linear = c == 0 || c == 4 || c == 5;
      const bool sijrs = c == 2;
      const double d = std::fabs(hr.energyDDots[c] - t.energyDots[c]);
      if (linear) euler = euler && d < kTangentExactTol;
      if (sijrs) euler = euler && hr.energyDDots[c] == 0.0 && t.energyDots[c] == 0.0;
      std::printf("  %-7s %6s %16.10f %16.10f %10.2e\n", nevpt2::CLASSES[c].c_str(),
                  linear ? "yes" : (sijrs ? "none" : "affine"), hr.energyDDots[c],
                  t.energyDots[c], d);
    }
    ok = ok && euler;
    std::printf("  Euler identity (the three linear classes, and Sijrs exactly zero): %s\n",
                euler ? "ok" : "FAILED");
  }

  // --- check H3: the one finite difference, of Edot(u) along v -----------------
  std::printf("\n  central differences of Edot(u) along c + h v (energiesTangentDevice on "
              "fresh builds):\n");
  std::printf("  %5s %14s %20s %12s %12s %8s\n", "k", "h = 2^-k", "(Edot(+h)-Edot(-h))/2h",
              "|err|", "rel err", "ratio");
  std::array<double, kStepCount> step{};
  std::array<double, kStepCount> err{};
  for (std::size_t i = 0; i < kStepCount; ++i) {
    const int k = kFirstStepExp + static_cast<int>(i);
    const double h = std::ldexp(1.0, -k);
    step[i] = h;
    const double plus = scTangentAt(*builder, tangentStep(ci, h, v), u, rdmOpt, ints, res);
    const double minus = scTangentAt(*builder, tangentStep(ci, -h, v), u, rdmOpt, ints, res);
    const double d = (plus - minus) / (2.0 * h);
    err[i] = std::fabs(d - eddot);
    const double rel = err[i] / std::max(1e-30, std::fabs(eddot));
    if (i == 0) {
      std::printf("  %5d %14.3e %20.12f %12.3e %12.3e %8s\n", k, h, d, err[i], rel, "-");
    } else {
      std::printf("  %5d %14.3e %20.12f %12.3e %12.3e %8.2f\n", k, h, d, err[i], rel,
                  err[i] > 0.0 ? err[i - 1] / err[i] : 0.0);
    }
  }
  std::size_t bestAt = 0;
  for (std::size_t i = 1; i < kStepCount; ++i)
    if (err[i] < err[bestAt]) bestAt = i;
  const double bestRel = err[bestAt] / std::max(1e-30, std::fabs(eddot));
  ok = ok && bestRel < kTangentFdTol;
  std::printf("\n  plateau: |err| = %.3e (relative %.3e) at h = %.3e (2^-%d); round-off takes "
              "over below it\n",
              err[bestAt], bestRel, step[bestAt], kFirstStepExp + static_cast<int>(bestAt));
  const nevpt2::FourfoldRun run = nevpt2::longestFourfoldRun(err);
  const bool richardson = run.length >= kRichardsonRun;
  ok = ok && richardson;
  std::printf("  Richardson: %d consecutive halvings cut the error fourfold (ratio in [%.1f, "
              "%.1f]), ending at h = %.3e -- %s, %d needed\n",
              run.length, nevpt2::kFourfoldLow, nevpt2::kFourfoldHigh, step[run.end],
              richardson ? "ok" : "NOT ENOUGH", kRichardsonRun);

  std::printf("\n%s: the SC Hessian matches Euler and the exact norm increment to %.0e, and "
              "the finite difference of Edot to %.0e relative (the orbital energies held "
              "fixed)\n",
              ok ? "PASS" : "FAIL", kTangentExactTol, kTangentFdTol);
  return ok;
}

// =============================================================================
// --sc-hessian-vector (#156): the VECTOR H_cc u, reverse-over-forward.
//
// WHAT IS BEING CHECKED: nevpt2.gradient's scCiHessianVectorDevice, the walk
// behind y differentiated along u and finished by two sigma calls,
// H_cc u = sigma(gdot[u], c) + sigma(g, u) (docs/hessian-theory.md,
// Sec. 4.2), on the same functional as --sc-hessian's: plain bilinear RDMs,
// the golden's e_core / e_virt held fixed.
//
// CHECK V1, EXACT: THE DUALITY. For any v,
//
//   <v, H_cc u> == Eddot(u, v),
//
// the right side energiesHessianDevice's (the forward-over-forward scalar
// --sc-hessian gates) on its own three-argument RDM build. The two share no
// code past the RDM builders: one is the reverse walk's tangent swept to the
// RDM coefficients and contracted by the sigma kernels, the other the class
// einsums at second-order RDMs and the 3x3 sandwich. Several seeded v, and the
// state itself (the one direction the projected seeds never reach). Like
// --gradient-source's <u, y> == Edot(u), an identity between two
// computations -- no step size -- so a 1e-7 gate.
//
// CHECK V2, EXACT, WHERE ndet ALLOWS: COLUMN BY COLUMN. H_cc e_I for every
// determinant I (so H_cc materialised, one Hessian-vector product per
// column), each column held to the duality against Eddot(e_I, v): a
// disagreement names the determinant, which a dot product cannot
// (Sec. 4.4: affordable exactly where --gradient-source's Jacobian is, so the
// same kJacobianMaxDet bound). The materialised H_cc's symmetry comes free --
// nothing in reverse-over-forward makes it symmetric but the algebra.
//
// CHECK V3, THE ONE FINITE DIFFERENCE: of the GPU's own EXACT y along u,
// (y(c + h u) - y(c - h u)) / 2h against H_cc u, y from scCiSourceDevice on
// fresh builds at each displaced state -- WITH THE ORBITAL-ENERGY MODE MATCHED
// ON BOTH LEGS: y carries the orbital energies' response only when handed a
// sidecar, and differencing that y against this H_cc u would difference two
// different functionals; both legs here are handed none. Same ladder and
// Richardson rules as CHECK 2 / H3, on the difference's 2-norm, with a gate
// of its own (kHessianVectorFdTol says why).
//
// Also free, and checked exactly: the walk's own Edot(u) against
// energiesTangentDevice (which pins the walk's tangent blocks to the forward
// tangent's), and <u, y> == Edot(u) for the y the same walk produces.
// docs/testing.md, "The SC Hessian-vector product's checks", records the
// measured residuals and plateau.

// The duality's gate: an identity between two computations, the tree's usual
// 1e-7 (the bound --gradient-source's <u, y> == Edot(u) carries), decades
// above what is measured (docs/testing.md, "The SC Hessian-vector product's
// checks").
constexpr double kHessianDualityTol = 1e-7;
// Up to how many determinants the column-by-column check runs: the bound
// --gradient-source's Jacobian uses (CAS(4,4) is 36, CAS(8,8) 4900), and for
// the same reason -- one Hessian-vector product and one Eddot per column.
constexpr std::int64_t kHessianColumnMaxDet = 256;
// The materialised H_cc's symmetry, RELATIVE to its largest entry. The
// matrix is the Hessian of the UNNORMALIZED functional, so it is not small
// along the state: the hole metrics' RDM-free terms (make_hdm1's 2 delta and
// its kin) make a nearly-filled hole perturber's norm move at FIRST order
// along c, and through the finish's -2/d that puts entries several decades
// above the rest on the determinants the state occupies. Symmetry is a
// round-off identity between two different columns' walks, so its residual
// scales with those entries; an absolute bound would measure the matrix's
// scale, not the algebra. About forty times the worst measured
// (docs/testing.md, "The SC Hessian-vector product's checks").
constexpr double kHessianSymmetryRelTol = 1e-9;
// The finite difference of y's gate: the best relative error of the sweep,
// ||(y+ - y-)/2h - H_cc u|| / ||H_cc u||, over the same ladder CHECK 2 runs.
// Its own bound, not kTangentFdTol, because a VECTOR's difference has a
// higher floor than a scalar's: every component of y is differenced, the
// ones on the state's own determinants included, where the curvature
// kHessianSymmetryRelTol describes makes both the truncation constant and
// y's own round-off decades larger than along the direction. The plateau
// sits where the two cross, a step or two wide, and at CAS(4,4) that is
// within a factor of ten of 1e-6 on its neighbouring rows; this leaves about
// twenty over the worst measured best (docs/testing.md, "The SC Hessian-vector
// product's checks"). The ladder is CHECK 2's: round-off takes over above
// its shortest step, so a longer one buys nothing (measured, same section).
constexpr double kHessianVectorFdTol = 1e-5;
// The duality's further directions v, fixed so a run is reproducible.
constexpr std::array<std::uint64_t, 3> kHessianDualSeeds = {0x5eed'0156u, 0x5eed'1156u,
                                                            0x5eed'2156u};

// <x, y> over two same-shaped vectors.
double hessianDot(const nevpt2::Tensor& x, const nevpt2::Tensor& y) {
  double s = 0.0;
  for (std::int64_t i = 0; i < x.size(); ++i) s += x.flat(i) * y.flat(i);
  return s;
}

// ||x||_2.
double hessianNorm(const nevpt2::Tensor& x) { return std::sqrt(hessianDot(x, x)); }

// H_cc u at the base state, the whole way from the direction: dm1/dm2's
// tangent along u on the host, dm3/f3ac/f3ca's on the device, then
// scCiHessianVectorDevice. Every device tensor it allocates is freed (on
// res's stream) when it returns.
nevpt2::ScCiHessianResult scHessianVectorAt(const nevpt2::HostDm12Builder& builder,
                                            const nevpt2::Tensor& ci, const nevpt2::Tensor& u,
                                            const nevpt2::RdmBuildOptions& rdmOpt,
                                            const CiCheckIntegrals& ints,
                                            const nevpt2::RdmTensors& base,
                                            const nevpt2::DeviceResources& res) {
  const nevpt2::HostDm12 dot = builder.dm12Dot(ci, u);
  const nevpt2::DeviceTensor dDm1Dot = nevpt2::uploadTensor(dot.dm1, res);
  const nevpt2::DeviceTensor dDm2Dot = nevpt2::uploadTensor(dot.dm2, res);
  const nevpt2::RdmBuildResult rdmDot = nevpt2::buildRdmTangentsDevice(
      rdmOpt, ci, u, builder.norb(), builder.nelecA(), builder.nelecB(), ints.h2e, res);
  const nevpt2::RdmTensors dotU{dDm1Dot, dDm2Dot, rdmDot.dm3, rdmDot.f3ac, rdmDot.f3ca};
  nevpt2::SigmaVectorOptions sigmaOpt;
  sigmaOpt.nTiles = rdmOpt.nTiles;
  return nevpt2::scCiHessianVectorDevice(ints.active, ints.source, ints.batch, ci, u,
                                         builder.nelecA(), builder.nelecB(), base, dotU,
                                         sigmaOpt, res);
}

// Eddot(u, v) at the base state, forward over forward: the three-argument
// RDM build at (ci, u, v), dm1/dm2's three tangent sets on the host, then
// energiesHessianDevice. No reverse-mode code is involved, which is what
// makes it a reference for the vector. Freed on return, as above.
double scEddotAt(const nevpt2::HostDm12Builder& builder, const nevpt2::Tensor& ci,
                 const nevpt2::Tensor& u, const nevpt2::Tensor& v,
                 const nevpt2::RdmBuildOptions& rdmOpt, const CiCheckIntegrals& ints,
                 const nevpt2::RdmTensors& base, const nevpt2::DeviceResources& res) {
  const nevpt2::HostDm12 dotU = builder.dm12Dot(ci, u);
  const nevpt2::HostDm12 dotV = builder.dm12Dot(ci, v);
  const nevpt2::HostDm12 ddot = builder.dm12Dot(u, v);
  const nevpt2::DeviceTensor dDm1U = nevpt2::uploadTensor(dotU.dm1, res);
  const nevpt2::DeviceTensor dDm2U = nevpt2::uploadTensor(dotU.dm2, res);
  const nevpt2::DeviceTensor dDm1V = nevpt2::uploadTensor(dotV.dm1, res);
  const nevpt2::DeviceTensor dDm2V = nevpt2::uploadTensor(dotV.dm2, res);
  const nevpt2::DeviceTensor dDm1UV = nevpt2::uploadTensor(ddot.dm1, res);
  const nevpt2::DeviceTensor dDm2UV = nevpt2::uploadTensor(ddot.dm2, res);
  const nevpt2::RdmHessianTangents t3 = nevpt2::buildRdmHessianTangentsDevice(
      rdmOpt, ci, u, v, builder.norb(), builder.nelecA(), builder.nelecB(), ints.h2e, res);
  const nevpt2::RdmTensors setU{dDm1U, dDm2U, t3.cu.dm3, t3.cu.f3ac, t3.cu.f3ca};
  const nevpt2::RdmTensors setV{dDm1V, dDm2V, t3.cv.dm3, t3.cv.f3ac, t3.cv.f3ca};
  const nevpt2::RdmTensors setUV{dDm1UV, dDm2UV, t3.uv.dm3, t3.uv.f3ac, t3.uv.f3ca};
  return nevpt2::energiesHessianDevice(ints.active, ints.source, ints.batch, base, setU, setV,
                                       setUV, res)
      .totalDDot;
}

// y at the CI vector `ci`, the whole way from the vector, with the orbital
// energies held FIXED (no OrbitalEnergyResponse): dm1/dm2 rebuilt on the
// host, dm3/f3ac/f3ca built on the device, then scCiSourceDevice. The
// finite difference's two legs, in the one mode H_cc u is computed in.
nevpt2::Tensor scSourceAt(const nevpt2::HostDm12Builder& builder, const nevpt2::Tensor& ci,
                          const nevpt2::RdmBuildOptions& rdmOpt, const CiCheckIntegrals& ints,
                          const nevpt2::DeviceResources& res) {
  const nevpt2::HostDm12 dm = builder.dm12(ci);
  const nevpt2::DeviceTensor dDm1 = nevpt2::uploadTensor(dm.dm1, res);
  const nevpt2::DeviceTensor dDm2 = nevpt2::uploadTensor(dm.dm2, res);
  const nevpt2::RdmBuildResult rdm = nevpt2::buildRdmsDevice(
      rdmOpt, ci, builder.norb(), builder.nelecA(), builder.nelecB(), ints.h2e, res);
  nevpt2::SigmaVectorOptions sigmaOpt;
  sigmaOpt.nTiles = rdmOpt.nTiles;
  return nevpt2::scCiSourceDevice(ints.active, ints.source, ints.batch, ci, builder.nelecA(),
                                  builder.nelecB(), dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca,
                                  sigmaOpt, /*eps=*/nullptr, res)
      .y;
}

nevpt2::Result<bool> checkScHessianVector(const nevpt2::GoldenFile& g,
                                          const nevpt2::RdmBuildOptions& rdmOpt,
                                          const CiCheckIntegrals& ints,
                                          const nevpt2::RdmBuildResult& baseRdm,
                                          std::uint64_t seedU, std::uint64_t seedV,
                                          const nevpt2::DeviceResources& res) {
  nevpt2::Result<nevpt2::HostDm12Builder> builder =
      nevpt2::HostDm12Builder::create(g.ncas, g.nelecA, g.nelecB);
  if (!builder) return std::unexpected(builder.error());
  const nevpt2::Tensor& ci = g.get("ci");

  std::printf("\n--sc-hessian-vector: H_cc u, the SC total's CI Hessian-vector product "
              "(reverse over forward)\n");
  // Which functional, said where the gates are stated (docs/hessian-theory.md,
  // Sec. 9.1): both sides of every check below hold the orbital energies fixed.
  std::printf("  functional: the orbital energies held fixed (the golden's e_core / e_virt), "
              "on every side of every check\n");

  const nevpt2::HostDm12 base = builder->dm12(ci);
  const nevpt2::DeviceTensor dDm1 = nevpt2::uploadTensor(base.dm1, res);
  const nevpt2::DeviceTensor dDm2 = nevpt2::uploadTensor(base.dm2, res);
  const nevpt2::RdmBuildResult& rdm = baseRdm;
  const nevpt2::RdmTensors baseSet{dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca};

  const nevpt2::Tensor u = tangentDirection(ci, seedU);
  bool ok = true;

  // --- H_cc u, and the two free checks on its walk ---------------------------
  const auto t0 = std::chrono::steady_clock::now();
  const nevpt2::ScCiHessianResult hu =
      scHessianVectorAt(*builder, ci, u, rdmOpt, ints, baseSet, res);
  const double huSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  H_cc u built in %.2fs (ndet=%lld, seed 0x%llx); |H_cc u| = %.10f  "
              "(|sigma(gdot, c)| = %.10f, |sigma(g, u)| = %.10f)\n",
              huSeconds, static_cast<long long>(hu.hu.size()),
              static_cast<unsigned long long>(seedU), hessianNorm(hu.hu),
              hessianNorm(hu.fromWalkTangent), hessianNorm(hu.fromDirection));
  {
    const nevpt2::HostDm12 dotU = builder->dm12Dot(ci, u);
    const nevpt2::DeviceTensor dDm1U = nevpt2::uploadTensor(dotU.dm1, res);
    const nevpt2::DeviceTensor dDm2U = nevpt2::uploadTensor(dotU.dm2, res);
    const nevpt2::RdmBuildResult rdmDot = nevpt2::buildRdmTangentsDevice(
        rdmOpt, ci, u, g.ncas, g.nelecA, g.nelecB, ints.h2e, res);
    const nevpt2::EnergyTangentResult t = nevpt2::energiesTangentDevice(
        ints.active, ints.source, ints.batch, dDm1, dDm2, rdm.dm3, rdm.f3ac, rdm.f3ca, dDm1U,
        dDm2U, rdmDot.dm3, rdmDot.f3ac, rdmDot.f3ca, /*epsDot=*/nullptr, res);
    std::printf("\n  the walk's own Edot(u) against energiesTangentDevice's, per class:\n");
    std::printf("  %-7s %18s %18s %10s\n", "class", "walk", "forward", "|delta|");
    bool walk = true;
    for (int c = 0; c < 8; ++c) {
      const double d = std::fabs(hu.edotU[c] - t.energyDots[c]);
      walk = walk && d < kTangentExactTol;
      std::printf("  %-7s %18.12f %18.12f %10.2e\n", nevpt2::CLASSES[c].c_str(), hu.edotU[c],
                  t.energyDots[c], d);
    }
    const double dTotal = std::fabs(hu.edotUTotal - t.totalDot);
    const double uy = hessianDot(u, hu.y);
    const double dUy = std::fabs(uy - t.totalDot);
    walk = walk && dTotal < kTangentExactTol && dUy < kHessianDualityTol;
    ok = ok && walk;
    std::printf("  total   %18.12f %18.12f %10.2e\n", hu.edotUTotal, t.totalDot, dTotal);
    std::printf("  <u, y> of the same walk's y: %.12f  |delta| against Edot(u) = %.2e\n", uy,
                dUy);
    std::printf("  the walk's tangent blocks and its y: %s\n", walk ? "ok" : "FAILED");
  }

  // --- check V1: the duality ---------------------------------------------------
  std::printf("\n  <v, H_cc u> against Eddot(u, v) (energiesHessianDevice), for several "
              "seeded v and the state\n");
  std::printf("  %-18s %20s %20s %11s %11s\n", "v", "<v, H_cc u>", "Eddot(u, v)", "|delta|",
              "relative");
  const auto dual = [&](const std::string& what, const nevpt2::Tensor& v) {
    const double lhs = hessianDot(v, hu.hu);
    const double rhs = scEddotAt(*builder, ci, u, v, rdmOpt, ints, baseSet, res);
    const double d = std::fabs(lhs - rhs);
    ok = ok && d < kHessianDualityTol;
    std::printf("  %-18s %20.12f %20.12f %11.2e %11.2e\n", what.c_str(), lhs, rhs, d,
                d / std::max(1e-30, std::fabs(rhs)));
  };
  dual(std::format("seed 0x{:x}", seedV), tangentDirection(ci, seedV));
  for (const std::uint64_t seed : kHessianDualSeeds)
    dual(std::format("seed 0x{:x}", seed), tangentDirection(ci, seed));
  dual("the state itself", ci);

  // --- check V2: column by column, where ndet allows ---------------------------
  const std::int64_t ndet = ci.size();
  if (ndet <= kHessianColumnMaxDet) {
    const nevpt2::Tensor v = tangentDirection(ci, seedV);
    std::printf("\n  H_cc materialised, column by column: <v, H_cc e_I> against "
                "Eddot(e_I, v) for all %lld determinants\n",
                static_cast<long long>(ndet));
    std::vector<nevpt2::Tensor> cols;
    cols.reserve(static_cast<std::size_t>(ndet));
    double cWorst = 0.0;
    std::int64_t cAt = 0;
    for (std::int64_t I = 0; I < ndet; ++I) {
      nevpt2::Tensor e(ci.dims());
      e.flatRef(I) = 1.0;
      cols.push_back(scHessianVectorAt(*builder, ci, e, rdmOpt, ints, baseSet, res).hu);
      const double lhs = hessianDot(v, cols.back());
      const double rhs = scEddotAt(*builder, ci, e, v, rdmOpt, ints, baseSet, res);
      const double d = std::fabs(lhs - rhs);
      if (d > cWorst) {
        cWorst = d;
        cAt = I;
      }
    }
    // The materialised matrix's symmetry, against its own largest entry (see
    // kHessianSymmetryRelTol for why relative).
    double sWorst = 0.0, hMax = 0.0;
    std::int64_t sI = 0, sJ = 0;
    for (std::int64_t I = 0; I < ndet; ++I)
      for (std::int64_t J = 0; J < ndet; ++J) {
        const double hIJ = cols[static_cast<std::size_t>(J)].flat(I);
        const double hJI = cols[static_cast<std::size_t>(I)].flat(J);
        if (std::fabs(hIJ - hJI) > sWorst) {
          sWorst = std::fabs(hIJ - hJI);
          sI = I;
          sJ = J;
        }
        hMax = std::max(hMax, std::fabs(hIJ));
      }
    const double sRel = sWorst / std::max(1e-30, hMax);
    const bool colsOk = cWorst < kHessianDualityTol;
    const bool symOk = sRel < kHessianSymmetryRelTol;
    ok = ok && colsOk && symOk;
    std::printf("  worst column |delta| = %.2e at determinant %lld (< %.0e): %s\n", cWorst,
                static_cast<long long>(cAt), kHessianDualityTol, colsOk ? "ok" : "FAILED");
    std::printf("  symmetry: max|H_IJ - H_JI| = %.2e at (%lld, %lld), max|H_IJ| = %.6e, "
                "relative %.2e (< %.0e): %s\n",
                sWorst, static_cast<long long>(sI), static_cast<long long>(sJ), hMax, sRel,
                kHessianSymmetryRelTol, symOk ? "ok" : "FAILED");
  } else {
    std::printf("\n  column-by-column check skipped: ndet = %lld is above %lld (the action "
                "and its duality only, docs/hessian-theory.md Sec. 4.4)\n",
                static_cast<long long>(ndet), static_cast<long long>(kHessianColumnMaxDet));
  }

  // --- check V3: the one finite difference, of y along u ------------------------
  std::printf("\n  central differences of y along c + h u (scCiSourceDevice on fresh builds, "
              "no orbital-energy response on either leg):\n");
  std::printf("  %5s %14s %16s %12s %12s %8s\n", "k", "h = 2^-k", "|(y+ - y-)/2h|",
              "|err|", "rel err", "ratio");
  const double huNorm = hessianNorm(hu.hu);
  std::array<double, kStepCount> step{};
  std::array<double, kStepCount> err{};
  for (std::size_t i = 0; i < kStepCount; ++i) {
    const int k = kFirstStepExp + static_cast<int>(i);
    const double h = std::ldexp(1.0, -k);
    step[i] = h;
    const nevpt2::Tensor plus = scSourceAt(*builder, tangentStep(ci, h, u), rdmOpt, ints, res);
    const nevpt2::Tensor minus =
        scSourceAt(*builder, tangentStep(ci, -h, u), rdmOpt, ints, res);
    nevpt2::Tensor d(ci.dims());
    double e2 = 0.0;
    for (std::int64_t j = 0; j < d.size(); ++j) {
      d.flatRef(j) = (plus.flat(j) - minus.flat(j)) / (2.0 * h);
      const double r = d.flat(j) - hu.hu.flat(j);
      e2 += r * r;
    }
    err[i] = std::sqrt(e2);
    const double rel = err[i] / std::max(1e-30, huNorm);
    if (i == 0) {
      std::printf("  %5d %14.3e %16.10f %12.3e %12.3e %8s\n", k, h, hessianNorm(d), err[i],
                  rel, "-");
    } else {
      std::printf("  %5d %14.3e %16.10f %12.3e %12.3e %8.2f\n", k, h, hessianNorm(d), err[i],
                  rel, err[i] > 0.0 ? err[i - 1] / err[i] : 0.0);
    }
  }
  std::size_t bestAt = 0;
  for (std::size_t i = 1; i < kStepCount; ++i)
    if (err[i] < err[bestAt]) bestAt = i;
  const double bestRel = err[bestAt] / std::max(1e-30, huNorm);
  ok = ok && bestRel < kHessianVectorFdTol;
  std::printf("\n  plateau: |err| = %.3e (relative %.3e) at h = %.3e (2^-%d); round-off takes "
              "over below it\n",
              err[bestAt], bestRel, step[bestAt], kFirstStepExp + static_cast<int>(bestAt));
  const nevpt2::FourfoldRun run = nevpt2::longestFourfoldRun(err);
  const bool richardson = run.length >= kRichardsonRun;
  ok = ok && richardson;
  std::printf("  Richardson: %d consecutive halvings cut the error fourfold (ratio in [%.1f, "
              "%.1f]), ending at h = %.3e -- %s, %d needed\n",
              run.length, nevpt2::kFourfoldLow, nevpt2::kFourfoldHigh, step[run.end],
              richardson ? "ok" : "NOT ENOUGH", kRichardsonRun);

  std::printf("\n%s: the SC Hessian-vector product matches Eddot(u, v) by duality to %.0e, and "
              "the finite difference of y to %.0e relative (the orbital energies held fixed "
              "on both legs)\n",
              ok ? "PASS" : "FAIL", kHessianDualityTol, kHessianVectorFdTol);
  return ok;
}

}  // namespace nevpt2
