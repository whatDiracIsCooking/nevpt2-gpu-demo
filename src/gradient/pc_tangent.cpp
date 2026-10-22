// The PC tangent check's driver and report -- see pc_tangent.cppm for what is
// checked (Euler's identity, exact on the five classes it applies to, then a
// Richardson-verified central finite difference) and why a finite difference
// is the only independent handle on the derivative of a solve.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.pc_tangent;

import std;
// pcCiHessianVectorDevice and pcCiSourceThreeTermDevice, the Hessian-vector
// check's subject (#155): this unit's business, so not re-exported.
import nevpt2.gradient;

namespace nevpt2 {

// The direction the check differentiates along:
//
//   u = sum_{pq} w_pq E_pq |ci>,  projected against |ci> and normalized,
//
// with w_pq seeded-random over the orbital pairs the state's own dm1 says are
// symmetry-allowed. Seeded, so the whole check is reproducible -- a ctest
// entry needs that. Projected against |ci> so that the displacement changes
// the state rather than (to first order) its length: the radial direction is
// the trivial one, and the one the normalized path below removes anyway.
//
// WHY NOT A RANDOM VECTOR OF AMPLITUDES. The PC metric's dropped modes are
// STRUCTURAL (docs/pc-nevpt2.md, "The singular-metric convention"), and part
// of that structure is the state's own symmetry: |ci> is a spin eigenstate in
// one spatial irrep, and strings that annihilate such a state do not
// annihilate a general vector of amplitudes. A random direction therefore
// RAISES the rank of S -- measured: Sr and Si at CAS(4,4) drop 44 of 64 modes
// at the state and only 20 at c + h u, the 24 new modes entering with
// eigenvalues ~h^2 and Dyall eigenvalues that make lambda_k + Delta_t
// negative, so the PC stage refuses the displaced state outright at EVERY h
// (docs/pc-nevpt2.md, "The tangent with respect to the CI vector"). That is
// the pseudo-inverse's rank condition failing, exactly as the derivation
// says it must, and it is a property of the direction, not of the step size.
// The spin-free generators E_pq commute with S^2, so E_pq|ci> is still a spin
// eigenstate; keeping only the pairs where dm1[p,q] is nonzero -- for a
// symmetry-adapted state, exactly the pairs whose orbitals share an irrep --
// keeps the spatial irrep too. With that direction every h^2 mode is gone and
// the displaced states' spectra come back with the state's own dropped counts
// (one h^4 mode in Sir is left, which is why the step is small as well -- see
// kStartPow in apps/pc_tangent/main.cppm). It is also the only kind of
// direction phase 2 needs, since a CI response stays inside the spin and
// symmetry sector it was solved in.
Tensor pcTangentDirection(const HostDm12Builder& builder, const Tensor& ci, const Tensor& dm1,
                          std::uint64_t seed) {
  const int64_t norb = builder.norb(), ndet = builder.ndet();
  // v[p, q, K] = (E_pq |ci>)[K].
  const std::vector<double> v = builder.applyAllE(ci);
  double dm1Max = 0.0;
  for (int64_t i = 0; i < dm1.size(); ++i) dm1Max = std::max(dm1Max, std::fabs(dm1.flat(i)));

  Tensor u(ci.dims());
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  int64_t pairs = 0;
  for (int64_t p = 0; p < norb; ++p)
    for (int64_t q = 0; q < norb; ++q) {
      // dm1[p,q] = <ci|E_pq|ci> vanishes identically when p and q sit in
      // different irreps, so this is the state's own symmetry mask.
      if (std::fabs(dm1(p, q)) <= 1e-12 * dm1Max) continue;
      ++pairs;
      const double w = dist(gen);
      const int64_t off = (p * norb + q) * ndet;
      for (int64_t k = 0; k < ndet; ++k) u.flatRef(k) += w * v[narrowTo<std::size_t>(off + k)];
    }
  double overlap = 0.0, ciNorm2 = 0.0;
  for (int64_t i = 0; i < u.size(); ++i) {
    overlap += ci.flat(i) * u.flat(i);
    ciNorm2 += ci.flat(i) * ci.flat(i);
  }
  for (int64_t i = 0; i < u.size(); ++i) u.flatRef(i) -= (overlap / ciNorm2) * ci.flat(i);
  double norm2 = 0.0;
  for (int64_t i = 0; i < u.size(); ++i) norm2 += u.flat(i) * u.flat(i);
  // A direction of length zero would make every difference quotient 0/0; it
  // cannot happen for a state with a nonzero dm1, so it is an invariant.
  check(norm2 > 0.0, "the PC tangent's CI direction came out zero");
  const double inv = 1.0 / std::sqrt(norm2);
  for (int64_t i = 0; i < u.size(); ++i) u.flatRef(i) *= inv;
  std::printf("direction: sum over %lld symmetry-allowed E_pq of w_pq E_pq|ci>, "
              "projected against ci and normalized\n",
              static_cast<long long>(pairs));
  return u;
}

// The displaced state (ci + h u) / ||ci + h u||, same shape.
//
// NORMALIZED, and it must be. The hole-side metrics are built with the
// Kronecker-delta terms of PySCF's make_hdm1 / make_hdm2 / make_dm3_h --
// 2 delta - dm1, and so on -- which are the hole RDMs only for a state of
// unit norm: at norm^2 = 1 + h^2 the delta terms and the RDM terms are on
// different scales, and the difference shows up as metric eigenvalues of
// order h^2 OF THE WRONG SIGN. Measured, at h = 2^-4 on the unnormalized
// line: Sij's metric grew 7 eigenvalues at -4.1e-3 relative (= -h^2) and the
// class was refused for its gap (docs/pc-nevpt2.md, "The tangent with
// respect to the CI vector"). On the unit sphere they are gone.
//
// This costs nothing in what is being checked: with <c|u> = 0 and ||c|| = 1,
// d/dh of 1/||c + h u|| vanishes at h = 0, so the normalized path and the
// straight line have the SAME tangent there -- the one pcTangentsDevice
// computes. And the PC energy is not a polynomial in the CI vector (there is
// a solve), so the central difference along this path is a genuine
// second-order difference, unlike the RDMs' own (test/rdm_host_dm/, where a
// straight line has no truncation error at all and the Richardson table needs
// a curved path).
Tensor pcTangentStep(const Tensor& ci, double h, const Tensor& u) {
  Tensor out = ci;
  for (int64_t i = 0; i < out.size(); ++i) out.flatRef(i) += h * u.flat(i);
  double norm2 = 0.0;
  for (int64_t i = 0; i < out.size(); ++i) norm2 += out.flat(i) * out.flat(i);
  const double inv = 1.0 / std::sqrt(norm2);
  for (int64_t i = 0; i < out.size(); ++i) out.flatRef(i) *= inv;
  return out;
}

// a * x as a fresh tensor from the pool: the Euler check's 2 * RDM, which is
// the RDM tangent along the state itself (every RDM here is quadratic in the
// CI vector -- the identity --rdm-tangent checks on the device build).
DeviceTensor pcTangentScaled(const DeviceTensor& x, double a, const DeviceResources& res) {
  std::vector<int> axes(x.dims.size());
  std::iota(axes.begin(), axes.end(), 0);
  return deviceTransposeNew(x, axes, a, res);
}

// One state's eight PC class energies, from its own host dm1/dm2 and its own
// device RDM build -- everything scoped, so only one state's buffers are
// alive at a time. A class the PC stage refuses at this displaced state is a
// Numerical Error naming the step: the difference quotient cannot be formed
// from an energy that was not computed, and quietly dropping the class would
// make the comparison meaningless.
//
// `minGap` is the conditioning guard's record of that state: the smallest S
// gap over the classes with a spectrum, as pcEnergiesDevice's own gap check
// measured it. The report gates on it being there (PcFdStep::displacedGap).
struct PcDisplacedEnergies {
  std::array<double, 8> energy{};
  double minGap = 0.0;
};

Result<PcDisplacedEnergies> pcTangentEnergies(const RdmBuildOptions& opt,
                                                const NevptIntegralsDevice& ints,
                                                const HostDm12Builder& builder, const Tensor& c,
                                                const Tensor& h2e, std::string_view what,
                                                const DeviceResources& res) {
  const HostDm12 host = builder.dm12(c);
  const DeviceTensor dm1 = uploadTensor(host.dm1, res);
  const DeviceTensor dm2 = uploadTensor(host.dm2, res);
  const RdmBuildResult rdm =
      buildRdmsDevice(opt, c, builder.norb(), builder.nelecA(), builder.nelecB(), h2e, res);
  const PcEnergyResult e =
      NEVPT2_TRY(pcEnergiesDevice(ints, dm1, dm2, rdm.dm3, rdm.f3ac, rdm.f3ca, res));
  PcDisplacedEnergies out;
  out.minGap = std::numeric_limits<double>::infinity();
  bool anySpectrum = false;
  for (int i = 0; i < 8; ++i) {
    if (e.classes[i].status != PcStatus::Done)
      return err_numerical(std::format(
          "the PC class {} is refused at the displaced state {}: {}; no finite difference can "
          "be formed there",
          CLASSES[i], what, e.classes[i].why.empty() ? "not computed" : e.classes[i].why));
    out.energy[i] = e.classes[i].energy;
    if (e.classes[i].hasSpectrum) {
      anySpectrum = true;
      out.minGap = std::min(out.minGap, e.classes[i].spectrum.gap);
    }
  }
  // No class with a spectrum means no gap was measured: that is "not
  // recorded", which the report fails on, not an infinitely safe state.
  if (!anySpectrum) out.minGap = 0.0;
  return out;
}

// The inputs both finite-difference checks (the tangent's and the Hessian's)
// take, validated once: the step ladder, the active space and the CI vector.
// `what` names the check in an Error. Returns the host RDM builder every
// state of the check goes through.
Result<HostDm12Builder> pcCheckInputs(const Tensor& ci, int64_t norb, int64_t nelecA,
                                      int64_t nelecB, std::span<const double> steps,
                                      std::string_view what) {
  // The step list is the caller's input: two steps at least, each positive
  // and half its predecessor, because the Richardson ratio and the
  // extrapolation below are written for successive halvings.
  if (std::ssize(steps) < 2)
    return err_config(std::format("the {} needs at least two finite-difference steps", what));
  for (std::size_t i = 0; i < steps.size(); ++i) {
    if (!(steps[i] > 0.0))
      return err_config(std::format("a finite-difference step must be positive, not {:g}",
                                    steps[i]));
    if (i > 0 && std::fabs(steps[i] * 2.0 - steps[i - 1]) > 1e-12 * steps[i - 1])
      return err_config(std::format(
          "the finite-difference steps must halve: {:g} does not follow {:g}", steps[i],
          steps[i - 1]));
  }
  HostDm12Builder builder = NEVPT2_TRY(HostDm12Builder::create(norb, nelecA, nelecB));
  if (ci.size() != builder.ndet())
    return err_io(std::format("the CI vector holds {} amplitudes, but CAS({},{}) has {} "
                              "determinants",
                              ci.size(), nelecA + nelecB, norb, builder.ndet()));

  // The whole check walks the unit sphere (`pcTangentStep`), so the state it
  // starts from has to be on it. Every committed golden's `ci` is -- its
  // Tr dm1 is the electron count exactly -- so a vector that is not
  // normalized is a file this was not written for, which is the caller's
  // input and comes back as an Error.
  double ciNorm2 = 0.0;
  for (int64_t i = 0; i < ci.size(); ++i) ciNorm2 += ci.flat(i) * ci.flat(i);
  if (std::fabs(ciNorm2 - 1.0) > 1e-12)
    return err_io(
        std::format("the {} needs a normalized CI vector; <c|c> = {:.12f}", what, ciNorm2));
  return builder;
}

Result<PcTangentCheck> pcTangentFdCheck(const RdmBuildOptions& rdmOpt,
                                       const NevptIntegralsDevice& ints, const Tensor& ci,
                                       int64_t norb, int64_t nelecA, int64_t nelecB,
                                       const Tensor& h2e, std::span<const double> steps,
                                       std::uint64_t seed, const DeviceResources& res) {
  const auto t0 = std::chrono::steady_clock::now();
  const HostDm12Builder builder =
      NEVPT2_TRY(pcCheckInputs(ci, norb, nelecA, nelecB, steps, "PC tangent check"));

  PcTangentCheck out;
  // dm1 and dm2 come from the HOST rebuild, not from the golden file's own
  // arrays, so the state and every displaced state go through one function
  // (the displaced ones have no file to read them from). The two agree to
  // 1e-12 on every committed golden (test/rdm_host_dm/).
  const HostDm12 host = builder.dm12(ci);
  const Tensor u = pcTangentDirection(builder, ci, host.dm1, seed);
  {
    // The state itself: its RDMs, their tangents along u, the PC energies,
    // the analytic tangent and the Euler identity. Scoped, so none of these
    // buffers is still held while the displaced states are built below.
    const HostDm12 hostDot = builder.dm12Dot(ci, u);
    const DeviceTensor dm1 = uploadTensor(host.dm1, res);
    const DeviceTensor dm2 = uploadTensor(host.dm2, res);
    const DeviceTensor dm1Dot = uploadTensor(hostDot.dm1, res);
    const DeviceTensor dm2Dot = uploadTensor(hostDot.dm2, res);
    const RdmBuildResult rdm =
        buildRdmsDevice(rdmOpt, ci, norb, nelecA, nelecB, h2e, res);
    const RdmBuildResult rdmDot =
        buildRdmTangentsDevice(rdmOpt, ci, u, norb, nelecA, nelecB, h2e, res);

    std::printf("\ncomputing the eight PC-NEVPT2 class energies on the GPU...\n");
    const PcEnergyResult ref =
        NEVPT2_TRY(pcEnergiesDevice(ints, dm1, dm2, rdm.dm3, rdm.f3ac, rdm.f3ca, res));
    for (int i = 0; i < 8; ++i) {
      out.energyRef[i] = ref.classes[i].energy;
      out.totalRef += ref.classes[i].energy;
    }

    std::printf("differentiating the PC-NEVPT2 solve along the CI direction...\n");
    const PcTangentResult tang = NEVPT2_TRY(
        pcTangentsDevice(ints, dm1, dm2, rdm.dm3, rdm.f3ac, rdm.f3ca, dm1Dot, dm2Dot, rdmDot.dm3,
                         rdmDot.f3ac, rdmDot.f3ca, /*epsDot=*/nullptr, res));
    out.classes = tang.classes;
    out.allDone = true;
    for (const PcClassTangent& c : out.classes) {
      out.allDone = out.allDone && c.result.status == PcStatus::Done;
      out.total += c.result.energy;
      out.edotTotal += c.edot;
    }
    // A refused class is the answer, not an error: the report prints it as
    // REFUSED and fails the gate. There is nothing to difference against, so
    // the displaced states are not built at all.
    if (!out.allDone) return out;

    // Euler: the tangent along the state itself, which is twice the energy
    // for every class whose matrices are linear in the RDMs
    // (kPcHomogeneousInCi). The RDM tangents along the state are twice the
    // RDMs, so no second RDM build is needed.
    const DeviceTensor dm1Two = pcTangentScaled(dm1, 2.0, res);
    const DeviceTensor dm2Two = pcTangentScaled(dm2, 2.0, res);
    const DeviceTensor dm3Two = pcTangentScaled(rdm.dm3, 2.0, res);
    const DeviceTensor f3acTwo = pcTangentScaled(rdm.f3ac, 2.0, res);
    const DeviceTensor f3caTwo = pcTangentScaled(rdm.f3ca, 2.0, res);
    const PcTangentResult euler = NEVPT2_TRY(
        pcTangentsDevice(ints, dm1, dm2, rdm.dm3, rdm.f3ac, rdm.f3ca, dm1Two, dm2Two, dm3Two,
                         f3acTwo, f3caTwo, /*epsDot=*/nullptr, res));
    for (int i = 0; i < 8; ++i) out.eulerEdot[i] = euler.classes[i].edot;
  }

  double prevErr = 0.0;
  for (const double h : steps) {
    PcFdStep st;
    st.h = h;
    std::printf("\nfinite difference at h=%.6g: two more states, each a full RDM build\n", h);
    const PcDisplacedEnergies plus = NEVPT2_TRY(
        pcTangentEnergies(rdmOpt, ints, builder, pcTangentStep(ci, h, u), h2e,
                          std::format("c + {:g} u", h), res));
    const PcDisplacedEnergies minus = NEVPT2_TRY(
        pcTangentEnergies(rdmOpt, ints, builder, pcTangentStep(ci, -h, u), h2e,
                          std::format("c - {:g} u", h), res));
    for (int i = 0; i < 8; ++i) {
      st.fdClass[i] = (plus.energy[i] - minus.energy[i]) / (2.0 * h);
      st.fdTotal += st.fdClass[i];
    }
    st.displacedGap = {plus.minGap, minus.minGap};
    st.err = std::fabs(st.fdTotal - out.edotTotal);
    st.ratio = (prevErr > 0.0 && st.err > 0.0) ? prevErr / st.err : 0.0;
    prevErr = st.err;
    out.steps.push_back(st);
  }

  // Richardson over the two finest steps: the O(h^2) term cancels, leaving
  // O(h^4), so this is the number to hold against the analytic tangent.
  const PcFdStep& fine = out.steps.back();
  const PcFdStep& coarse = out.steps[out.steps.size() - 2];
  out.richardson = (4.0 * fine.fdTotal - coarse.fdTotal) / 3.0;
  out.richardsonErr = std::fabs(out.richardson - out.edotTotal);
  out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

// The step ladder both reports print and gate on: the table of differences
// of `differenced` along c + h u, each held against the analytic `against`,
// and the verdict -- true only when every ratio is within `ratioTol` of 4,
// they make a run of at least kPcMinFourfoldRun, and the conditioning guard
// ran and passed at every displaced state.
//
// The finite difference is gated on a RUN of fourfold halvings, never on a
// step: with fewer than kPcMinFourfoldRun of them -- no steps, or one -- the
// check has measured no order of convergence and fails.
bool printPcLadder(const std::vector<PcFdStep>& steps, double ratioTol, const char* differenced,
                   const char* against) {
  std::printf(
      "\ncentral differences of %s along c + h u (second order: the error falls "
      "4x per halving)\n",
      differenced);
  const std::string errHeader = std::format("|fd - {}|", against);
  std::printf("%12s %18s %12s %8s %22s\n", "h", "fd (Ha)", errHeader.c_str(), "ratio",
              "guard gap at c +- h u");
  bool ok = true;
  std::vector<double> errs;
  bool guarded = true;
  for (const PcFdStep& st : steps) {
    errs.push_back(st.err);
    // The conditioning guard must have run at both displaced states and
    // passed: a displaced state is exactly where the metric's rank can move.
    const bool stepGuarded = st.displacedGap[0] >= PC_MIN_GAP && st.displacedGap[1] >= PC_MIN_GAP;
    guarded = guarded && stepGuarded;
    const std::string gaps =
        std::format("{:.1e} {:.1e}{}", st.displacedGap[0], st.displacedGap[1],
                    stepGuarded ? "" : " NO");
    if (st.ratio > 0.0) {
      std::printf("%12.6g %18.10f %12.2e %8.3f %22s\n", st.h, st.fdTotal, st.err, st.ratio,
                  gaps.c_str());
      ok = ok && std::fabs(st.ratio - 4.0) < ratioTol;
    } else {
      std::printf("%12.6g %18.10f %12.2e %8s %22s\n", st.h, st.fdTotal, st.err, "-",
                  gaps.c_str());
    }
  }
  const FourfoldRun run = longestFourfoldRun(errs, 4.0 - ratioTol, 4.0 + ratioTol);
  const bool runOk = run.length >= kPcMinFourfoldRun;
  std::printf("fourfold run: %d consecutive halvings (%d needed) -- %s; the conditioning guard "
              "ran and passed (gap >= %.0e) at every displaced state -- %s\n",
              run.length, kPcMinFourfoldRun, runOk ? "ok" : "NOT ENOUGH", PC_MIN_GAP,
              guarded ? "ok" : "NO");
  return ok && runOk && guarded;
}

bool printPcTangentReport(const PcTangentCheck& c, double atol, double idtol, double ratioTol) {
  std::printf(
      "\nPC-NEVPT2 tangent with respect to the CI vector (the integrals held fixed)\n"
      "  E            the class energy from the tangent path's own solve\n"
      "  |d E|        against pcEnergiesDevice's answer for the same state\n"
      "  Edot         the analytic derivative along the direction u\n"
      "  fd(h_min)    its central difference at the finest step\n"
      "  Euler        |Edot(c, c) - 2 E|, exact (0) only where the class's matrices are\n"
      "               LINEAR in the RDMs -- not for Sijr / Sij (hole-side delta terms, so\n"
      "               affine) or Sijrs (no RDM at all); those print their value in\n"
      "               brackets and are not gated (pc_tangent.cppm, kPcHomogeneousInCi)\n");
  std::printf("%-6s %18s %10s %18s %18s %10s %12s\n", "class", "E (Ha)", "|d E|", "Edot (Ha)",
              "fd(h_min) (Ha)", "|d Edot|", "Euler");
  bool ok = true;
  int computed = 0;
  double maxEnergyDelta = 0.0, maxEulerDelta = 0.0;
  const bool haveFd = !c.steps.empty();
  for (int i = 0; i < 8; ++i) {
    const PcClassTangent& k = c.classes[i];
    const char* name = CLASSES[i].c_str();
    if (k.result.status != PcStatus::Done) {
      ok = false;
      std::printf("%-6s %18s %10s %18s %18s %10s %12s  (%s)\n", name,
                  k.result.status == PcStatus::Refused ? "REFUSED" : "not yet", "-", "-", "-",
                  "-", "-", k.result.why.empty() ? "not computed" : k.result.why.c_str());
      continue;
    }
    ++computed;
    const double dE = std::fabs(k.result.energy - c.energyRef[i]);
    maxEnergyDelta = std::max(maxEnergyDelta, dE);
    ok = ok && dE < idtol;
    // Euler, where the class is homogeneous: exact, no step size
    // (pc_tangent.cppm).
    const double eulerDelta = std::fabs(c.eulerEdot[i] - 2.0 * k.result.energy);
    std::string euler = std::format("[{:.2e}]", eulerDelta);
    if (kPcHomogeneousInCi[i]) {
      maxEulerDelta = std::max(maxEulerDelta, eulerDelta);
      ok = ok && eulerDelta < idtol * std::max(1.0, std::fabs(2.0 * k.result.energy));
      euler = std::format("{:.2e}", eulerDelta);
    }
    const std::string fd =
        haveFd ? std::format("{:18.10f}", c.steps.back().fdClass[i]) : std::format("{:>18}", "-");
    const std::string dEdot =
        haveFd ? std::format("{:10.2e}", std::fabs(c.steps.back().fdClass[i] - k.edot))
               : std::format("{:>10}", "-");
    std::printf("%-6s %18.10f %10.2e %18.10f %s %s %12s\n", name, k.result.energy, dE, k.edot,
                fd.c_str(), dEdot.c_str(), euler.c_str());
  }
  ok = ok && computed == 8;
  std::printf("\nE_corr (PC-NEVPT2)       = %.10f Ha  (pcEnergiesDevice: %.10f, |delta|=%.2e)\n",
              c.total, c.totalRef, std::fabs(c.total - c.totalRef));
  std::printf("Edot (analytic, along u) = %.10f Ha\n", c.edotTotal);
  std::printf("Euler: worst |Edot(c, c) - 2 E| over the five homogeneous classes = %.2e\n",
              maxEulerDelta);

  ok = printPcLadder(c.steps, ratioTol, "the PC total", "Edot") && ok;
  if (haveFd) {
    std::printf("Richardson (4 fd(h_min) - fd(2 h_min)) / 3 = %.10f Ha   |delta vs Edot|=%.2e\n",
                c.richardson, c.richardsonErr);
    ok = ok && c.richardsonErr < atol;
    std::printf("tangent check: %d of 8 classes computed, max class |d E| vs pcEnergiesDevice "
                "%.2e, %.1fs\n",
                computed, maxEnergyDelta, c.seconds);
  }

  // A wording of its own, so no other entry's PASS line can satisfy this
  // entry's regex (or the other way round).
  std::printf("\n%s: the PC-NEVPT2 tangent matches the central finite difference (Richardson "
              "to %.0e, every ratio 4 +- %.1f, a run of %d)\n",
              ok ? "PASS" : "FAIL", atol, ratioTol, kPcMinFourfoldRun);
  return ok;
}

// --- the second derivative, Eddot(u, v) (#152) ---------------------------------
//
// What is checked, and why the difference is of the THREE-term Edot rather
// than of E or of the device's two-term tangent, is at PcHessianCheck in
// pc_tangent.cppm.

// A random vector of amplitudes, projected against `ci` and normalized: the
// negative control's direction. NOT spin-free and not masked to the state's
// symmetry, which is the point -- along it S's dropped modes lift, and the
// Hessian must refuse rather than answer (docs/hessian-theory.md, Sec. 6.2).
Tensor pcHessianControlDirection(const Tensor& ci, std::uint64_t seed) {
  Tensor w(ci.dims());
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int64_t i = 0; i < w.size(); ++i) w.flatRef(i) = dist(gen);
  double overlap = 0.0, norm2 = 0.0;
  for (int64_t i = 0; i < w.size(); ++i) overlap += ci.flat(i) * w.flat(i);
  for (int64_t i = 0; i < w.size(); ++i) w.flatRef(i) -= overlap * ci.flat(i);
  for (int64_t i = 0; i < w.size(); ++i) norm2 += w.flat(i) * w.flat(i);
  check(norm2 > 0.0, "the PC Hessian's control direction came out zero");
  for (int64_t i = 0; i < w.size(); ++i) w.flatRef(i) /= std::sqrt(norm2);
  return w;
}

// One RDM set on the device and the host, owning: dm1/dm2 uploaded from a
// host build (nevpt2.rdm_host_dm) and dm3/f3ac/f3ca from a device build.
// `hostDm1` is dm1's host copy: what the FULL mode's orbital-energy
// derivatives read (#214).
struct PcOwnedRdms {
  DeviceTensor dm1, dm2;
  RdmBuildResult rdm;
  Tensor hostDm1;
  PcRdms view() const { return {&dm1, &dm2, &rdm.dm3, &rdm.f3ac, &rdm.f3ca}; }
};

PcOwnedRdms pcOwned(const HostDm12& host, RdmBuildResult rdm, const DeviceResources& res) {
  return {uploadTensor(host.dm1, res), uploadTensor(host.dm2, res), std::move(rdm), host.dm1};
}

// --- the FULL mode's orbital energies (#214) -----------------------------------
//
// With a gradient sidecar the orbital energies respond to the CI vector,
// e_p(c) = h^eff_pp + <dm1(c), W_p> (docs/gradients.md, Sec. 5.5), affine in
// dm1: a derivative reads e' = <dm1', W> and e'' = <dm1'', W> off the same
// 1-RDM sets the RDM side reads, and a displaced state's orbital energies are
// the base state's plus <dm1(c') - dm1(c), W>, exactly. `w` null holds them
// fixed everywhere below (every helper then returns the frozen answer).

// e'[u], e'[v], e''[u, v] from three sets' host dm1, or nothing.
std::optional<OrbitalEnergyHessianDots> pcEpsDots(const OrbitalEnergyResponse* w,
                                                  const PcOwnedRdms& u, const PcOwnedRdms& v,
                                                  const PcOwnedRdms& uv) {
  if (w == nullptr) return std::nullopt;
  return OrbitalEnergyHessianDots{orbitalEnergyDot(*w, u.hostDm1), orbitalEnergyDot(*w, v.hostDm1),
                                  orbitalEnergyDot(*w, uv.hostDm1)};
}

// A displaced state's integrals: `ints`' own, viewed, with the orbital
// energies at the state whose host dm1 is `dm1At` (`dm1Base` the base
// state's, whose orbital energies `ints` carries). Views of the caller's
// integrals and of the owned eCore / eVirt, so it must outlive every use.
struct PcDisplacedInts {
  DeviceTensor eCore, eVirt;
  NevptIntegralsDevice ints;
};

PcDisplacedInts pcDisplacedInts(const NevptIntegralsDevice& ints, const OrbitalEnergyResponse* w,
                                const Tensor& dm1Base, const Tensor& dm1At,
                                const DeviceResources& res) {
  PcDisplacedInts out;
  if (w != nullptr) {
    Tensor ec = downloadTensor(ints.e_core, res.stream());
    Tensor ev = downloadTensor(ints.e_virt, res.stream());
    Tensor shift = dm1At;
    for (int64_t i = 0; i < shift.size(); ++i) shift.flatRef(i) -= dm1Base.flat(i);
    const OrbitalEnergyDot d = orbitalEnergyDot(*w, shift);
    for (int64_t i = 0; i < ec.size(); ++i) ec.flatRef(i) += d.core.flat(i);
    for (int64_t r = 0; r < ev.size(); ++r) ev.flatRef(r) += d.virt.flat(r);
    out.eCore = uploadTensor(ec, res);
    out.eVirt = uploadTensor(ev, res);
  }
  const DeviceTensor& eCore = w != nullptr ? out.eCore : ints.e_core;
  const DeviceTensor& eVirt = w != nullptr ? out.eVirt : ints.e_virt;
  out.ints = NevptIntegralsDevice{ints.h1e.view(),        ints.h2e.view(),
                                  eCore.view(),            eVirt.view(),
                                  ints.h2e_v_Sr.view(),    ints.h1e_v_Sr.view(),
                                  ints.h2e_v_Si.view(),    ints.h1e_v_Si.view(),
                                  ints.cvcv.view(),        ints.h2e_v_Sijr.view(),
                                  ints.h2e_v_Srsi.view(),  ints.h2e_v_Srs.view(),
                                  ints.h2e_v_Sij.view(),   ints.h2e_v1_Sir.view(),
                                  ints.h2e_v2_Sir.view(),  ints.h1e_v_Sir.view()};
  return out;
}

// The tangent build at the vector pair (a, b) -- B(a, b) + B(b, a), the
// RDMs' derivative at a along b, or their second derivative when a is a
// direction too (#151).
PcOwnedRdms pcTangentSet(const RdmBuildOptions& opt, const HostDm12Builder& builder,
                         const Tensor& a, const Tensor& b, const Tensor& h2e,
                         const DeviceResources& res) {
  return pcOwned(builder.dm12Dot(a, b),
                 buildRdmTangentsDevice(opt, a, b, builder.norb(), builder.nelecA(),
                                        builder.nelecB(), h2e, res),
                 res);
}

// One displaced state's three-term Edot(v) per class (pcHessianDevice with no
// second-order set: exact at ANY CI vector), from its own host dm1/dm2 and
// their tangent along v and its own plain and tangent device RDM builds --
// scoped, so only one state's buffers are alive at a time. `minGap` as
// PcDisplacedEnergies'. A class refused at this state is a Numerical Error
// naming the step: no difference quotient can be formed from it.
struct PcDisplacedTangents {
  std::array<double, 8> edot{};
  double minGap = 0.0;
};

Result<PcDisplacedTangents> pcHessianTangentAt(const RdmBuildOptions& opt,
                                               const NevptIntegralsDevice& ints,
                                               const HostDm12Builder& builder, const Tensor& c,
                                               const Tensor& v, const Tensor& h2e,
                                               std::string_view what,
                                               const OrbitalEnergyResponse* w,
                                               const Tensor& dm1Base,
                                               const DeviceResources& res) {
  const int64_t norb = builder.norb(), na = builder.nelecA(), nb = builder.nelecB();
  const PcOwnedRdms base =
      pcOwned(builder.dm12(c), buildRdmsDevice(opt, c, norb, na, nb, h2e, res), res);
  const PcOwnedRdms dot = pcTangentSet(opt, builder, c, v, h2e, res);
  // FULL mode (#214): this state's own orbital energies, and their tangent
  // along v (the second-order slot is unread: no second-order set).
  const PcDisplacedInts at = pcDisplacedInts(ints, w, dm1Base, base.hostDm1, res);
  const std::optional<OrbitalEnergyHessianDots> eps = pcEpsDots(w, dot, dot, dot);
  PcHessianSets sets{base.view(), dot.view(), dot.view()};
  sets.eps = eps ? &*eps : nullptr;
  const PcHessianResult t = NEVPT2_TRY(pcHessianDevice(at.ints, sets, res));
  PcDisplacedTangents out;
  out.minGap = std::numeric_limits<double>::infinity();
  bool anySpectrum = false;
  for (int i = 0; i < 8; ++i) {
    const PcClassResult& k = t.classes[i].result;
    if (k.status != PcStatus::Done)
      return err_numerical(std::format(
          "the PC class {} is refused at the displaced state {}: {}; no finite difference of "
          "its tangent can be formed there",
          CLASSES[i], what, k.why.empty() ? "not computed" : k.why));
    out.edot[i] = t.classes[i].edotV;
    if (k.hasSpectrum) {
      anySpectrum = true;
      out.minGap = std::min(out.minGap, k.spectrum.gap);
    }
  }
  if (!anySpectrum) out.minGap = 0.0;  // "not recorded", which the report fails on
  return out;
}

Result<PcHessianCheck> pcHessianFdCheck(const RdmBuildOptions& rdmOpt,
                                        const NevptIntegralsDevice& ints, const Tensor& ci,
                                        int64_t norb, int64_t nelecA, int64_t nelecB,
                                        const Tensor& h2e, std::span<const double> steps,
                                        std::uint64_t seedU, std::uint64_t seedV,
                                        std::uint64_t seedControl,
                                        const OrbitalEnergyResponse* eps,
                                        const DeviceResources& res) {
  const auto t0 = std::chrono::steady_clock::now();
  const HostDm12Builder builder =
      NEVPT2_TRY(pcCheckInputs(ci, norb, nelecA, nelecB, steps, "PC Hessian check"));

  PcHessianCheck out;
  out.orbitalEnergiesMove = eps != nullptr;
  const HostDm12 host = builder.dm12(ci);
  // Both directions are the tangent check's kind: spin-free number-operator
  // directions over the state's symmetry mask, orthogonal to it and of unit
  // length -- the directions along which S's dropped modes turn but do not
  // lift (docs/pc-nevpt2.md, "What the cut drops, and how it moves"). v has
  // to be one too, not only u: its tangent is taken at every displaced
  // state.
  const Tensor u = pcTangentDirection(builder, ci, host.dm1, seedU);
  const Tensor v = pcTangentDirection(builder, ci, host.dm1, seedV);
  {
    // The state itself: its RDMs; the sets at (c, u), (c, v) and (u, v) from
    // the three-argument build's one tile sweep; the lift test's (u, u) and
    // (v, v); the references; the Hessian; Euler; and the negative control.
    // Scoped, so none of it is still held while the displaced states run.
    const PcOwnedRdms base = pcOwned(
        host, buildRdmsDevice(rdmOpt, ci, norb, nelecA, nelecB, h2e, res), res);
    RdmHessianTangents three =
        buildRdmHessianTangentsDevice(rdmOpt, ci, u, v, norb, nelecA, nelecB, h2e, res);
    const PcOwnedRdms dotU = pcOwned(builder.dm12Dot(ci, u), std::move(three.cu), res);
    const PcOwnedRdms dotV = pcOwned(builder.dm12Dot(ci, v), std::move(three.cv), res);
    const PcOwnedRdms ddot = pcOwned(builder.dm12Dot(u, v), std::move(three.uv), res);
    const PcOwnedRdms ddotUU = pcTangentSet(rdmOpt, builder, u, u, h2e, res);
    const PcOwnedRdms ddotVV = pcTangentSet(rdmOpt, builder, v, v, h2e, res);
    const PcRdms b = base.view();

    std::printf("\ncomputing the eight PC-NEVPT2 class energies on the GPU...\n");
    const PcEnergyResult ref = NEVPT2_TRY(pcEnergiesDevice(
        ints, base.dm1, base.dm2, base.rdm.dm3, base.rdm.f3ac, base.rdm.f3ca, res));
    std::printf("differentiating the PC-NEVPT2 solve along u and along v (pcTangentsDevice)...\n");
    // FULL mode (#214): the orbital energies' derivatives along (u, v).
    const std::optional<OrbitalEnergyHessianDots> epsUV = pcEpsDots(eps, dotU, dotV, ddot);
    const auto tangentAlong = [&](const PcOwnedRdms& d, const OrbitalEnergyDot* epsDot) {
      return pcTangentsDevice(ints, base.dm1, base.dm2, base.rdm.dm3, base.rdm.f3ac,
                              base.rdm.f3ca, d.dm1, d.dm2, d.rdm.dm3, d.rdm.f3ac, d.rdm.f3ca,
                              epsDot, res);
    };
    const PcTangentResult tanU = NEVPT2_TRY(tangentAlong(dotU, epsUV ? &epsUV->u : nullptr));
    const PcTangentResult tanV = NEVPT2_TRY(tangentAlong(dotV, epsUV ? &epsUV->v : nullptr));
    for (int i = 0; i < 8; ++i) {
      out.energyRef[i] = ref.classes[i].energy;
      out.edotURef[i] = tanU.classes[i].edot;
      out.edotVRef[i] = tanV.classes[i].edot;
    }

    std::printf("differentiating it twice, along (u, v)...\n");
    const PcRdms vUV = ddot.view(), vUU = ddotUU.view(), vVV = ddotVV.view();
    const PcHessianResult hess = NEVPT2_TRY(pcHessianDevice(
        ints,
        PcHessianSets{b, dotU.view(), dotV.view(), &vUV, &vUU, &vVV, epsUV ? &*epsUV : nullptr},
        res));
    out.classes = hess.classes;
    out.allDone = true;
    for (const PcClassHessian& k : out.classes) {
      out.allDone = out.allDone && k.result.status == PcStatus::Done;
      out.eddotTotal += k.eddot;
    }
    // A refused class is the answer, not an error: the report prints it as
    // REFUSED and fails the gate, and no displaced state is built.
    if (!out.allDone) return out;

    // Euler at second order: Eddot(c, u). The first-order set along the
    // state itself is twice the RDMs (they are quadratic in the CI vector),
    // which is also the (c, c) set; the second-order set at (c, u) is the
    // first-order set along u -- so no RDM build at all. An identity at fixed
    // orbital energies only (docs/hessian-theory.md, Sec. 4.3): in FULL mode
    // it is not run, and the report says it was skipped.
    const PcRdms vU = dotU.view();
    if (eps == nullptr) {
      PcOwnedRdms two;
      two.dm1 = pcTangentScaled(base.dm1, 2.0, res);
      two.dm2 = pcTangentScaled(base.dm2, 2.0, res);
      two.rdm.dm3 = pcTangentScaled(base.rdm.dm3, 2.0, res);
      two.rdm.f3ac = pcTangentScaled(base.rdm.f3ac, 2.0, res);
      two.rdm.f3ca = pcTangentScaled(base.rdm.f3ca, 2.0, res);
      const PcRdms vC = two.view();
      const PcHessianResult euler =
          NEVPT2_TRY(pcHessianDevice(ints, PcHessianSets{b, vC, vU, &vU, &vC, &vUU}, res));
      for (int i = 0; i < 8; ++i) out.eulerEddot[i] = euler.classes[i].eddot;
    }

    // The negative control: (u, w) with w a random vector of amplitudes, in
    // the run's orbital-energy mode.
    std::printf("the negative control: the Hessian along u and a symmetry-breaking w...\n");
    const Tensor w = pcHessianControlDirection(ci, seedControl);
    const PcOwnedRdms dotW = pcTangentSet(rdmOpt, builder, ci, w, h2e, res);
    const PcOwnedRdms ddotUW = pcTangentSet(rdmOpt, builder, u, w, h2e, res);
    const PcOwnedRdms ddotWW = pcTangentSet(rdmOpt, builder, w, w, h2e, res);
    const PcRdms vW = dotW.view(), vUW = ddotUW.view(), vWW = ddotWW.view();
    const std::optional<OrbitalEnergyHessianDots> epsUW = pcEpsDots(eps, dotU, dotW, ddotUW);
    const PcHessianResult control = NEVPT2_TRY(pcHessianDevice(
        ints, PcHessianSets{b, vU, vW, &vUW, &vUU, &vWW, epsUW ? &*epsUW : nullptr}, res));
    out.control = control.classes;
  }

  double prevErr = 0.0;
  for (const double h : steps) {
    PcFdStep st;
    st.h = h;
    std::printf("\nfinite difference of Edot(v) at h=%.6g: two more states, each a plain and a "
                "tangent RDM build\n",
                h);
    const PcDisplacedTangents plus = NEVPT2_TRY(
        pcHessianTangentAt(rdmOpt, ints, builder, pcTangentStep(ci, h, u), v, h2e,
                           std::format("c + {:g} u", h), eps, host.dm1, res));
    const PcDisplacedTangents minus = NEVPT2_TRY(
        pcHessianTangentAt(rdmOpt, ints, builder, pcTangentStep(ci, -h, u), v, h2e,
                           std::format("c - {:g} u", h), eps, host.dm1, res));
    for (int i = 0; i < 8; ++i) {
      st.fdClass[i] = (plus.edot[i] - minus.edot[i]) / (2.0 * h);
      st.fdTotal += st.fdClass[i];
    }
    st.displacedGap = {plus.minGap, minus.minGap};
    st.err = std::fabs(st.fdTotal - out.eddotTotal);
    st.ratio = (prevErr > 0.0 && st.err > 0.0) ? prevErr / st.err : 0.0;
    prevErr = st.err;
    out.steps.push_back(st);
  }

  const PcFdStep& fine = out.steps.back();
  const PcFdStep& coarse = out.steps[out.steps.size() - 2];
  out.richardson = (4.0 * fine.fdTotal - coarse.fdTotal) / 3.0;
  out.richardsonErr = std::fabs(out.richardson - out.eddotTotal);
  out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

bool printPcHessianReport(const PcHessianCheck& c, double atol, double idtol, double edotTol,
                          double ratioTol) {
  const bool full = c.orbitalEnergiesMove;
  std::printf("\nPC-NEVPT2 second derivative with respect to the CI vector, Eddot(u, v) (the "
              "integrals %s)\n",
              full ? "held fixed, the orbital energies responding to c: FULL mode, Euler "
                     "SKIPPED, not an identity there"
                   : "and the orbital energies held fixed");
  std::printf(
      "  |d E|        the Hessian walk's class energy against pcEnergiesDevice's\n"
      "  |d Edot|     its three-term Edot(u) / Edot(v), worse of the two, against\n"
      "               pcTangentsDevice's two-term one (equal at the eigenstate, to its\n"
      "               convergence)\n"
      "  Eddot        the analytic second derivative along u and v, rotation terms included\n"
      "  fd(h_min)    the central difference of the three-term Edot(v) along u, finest step\n"
      "  Euler        |Eddot(c, u) - Edot(u)|, exact (0) only where the class is homogeneous\n"
      "               of degree 2 (pc_tangent.cppm, kPcHomogeneousInCi); Sijr / Sij / Sijrs\n"
      "               print their value in brackets and are not gated\n"
      "  kept s       the smallest kept eigenvalue of S over s_max: S^+ in the rotation terms\n"
      "               scales as its inverse (the gap, PC_MIN_GAP's, does not bound it)\n"
      "  lift         the dropped modes' first/second-order lift along (u, v) over s_max\n"
      "               (refused above PC_LIFT_TOL)\n");
  std::printf("%-6s %18s %9s %9s %18s %18s %9s %10s %9s %9s %9s\n", "class", "E (Ha)", "|d E|",
              "|d Edot|", "Eddot (Ha)", "fd(h_min) (Ha)", "|d Eddot|", "Euler", "kept s", "gap",
              "lift");
  bool ok = true;
  int computed = 0;
  double maxEnergyDelta = 0.0, maxEdotDelta = 0.0, maxEulerDelta = 0.0;
  double minKept = std::numeric_limits<double>::infinity();
  double minGap = std::numeric_limits<double>::infinity();
  std::string_view minKeptClass = "-";
  const bool haveFd = !c.steps.empty();
  for (int i = 0; i < 8; ++i) {
    const PcClassHessian& k = c.classes[i];
    const char* name = CLASSES[i].c_str();
    if (k.result.status != PcStatus::Done) {
      ok = false;
      std::printf("%-6s %18s %9s %9s %18s %18s %9s %10s %9s %9s %9.1e  (%s)\n", name,
                  k.result.status == PcStatus::Refused ? "REFUSED" : "not yet", "-", "-", "-",
                  "-", "-", "-", "-", "-", k.lift,
                  k.result.why.empty() ? "not computed" : k.result.why.c_str());
      continue;
    }
    ++computed;
    const double dE = std::fabs(k.result.energy - c.energyRef[i]);
    maxEnergyDelta = std::max(maxEnergyDelta, dE);
    ok = ok && dE < idtol;
    const double dU = std::fabs(k.edotU - c.edotURef[i]);
    const double dV = std::fabs(k.edotV - c.edotVRef[i]);
    const double dEdot = std::max(dU, dV);
    maxEdotDelta = std::max(maxEdotDelta, dEdot);
    ok = ok && dU < edotTol * std::max(1.0, std::fabs(c.edotURef[i])) &&
         dV < edotTol * std::max(1.0, std::fabs(c.edotVRef[i]));
    // Euler at second order, where the class is homogeneous: exact, no step.
    // Skipped in FULL mode (not computed, not gated).
    const double eulerDelta = std::fabs(c.eulerEddot[i] - c.edotURef[i]);
    std::string euler = full ? std::string("skipped") : std::format("[{:.2e}]", eulerDelta);
    if (!full && kPcHomogeneousInCi[i]) {
      maxEulerDelta = std::max(maxEulerDelta, eulerDelta);
      ok = ok && eulerDelta < idtol * std::max(1.0, std::fabs(c.edotURef[i]));
      euler = std::format("{:.2e}", eulerDelta);
    }
    std::string kept = "-", gap = "-";
    if (k.result.hasSpectrum) {
      if (k.result.spectrum.smallestKept < minKept) {
        minKept = k.result.spectrum.smallestKept;
        minKeptClass = CLASSES[i];
      }
      minGap = std::min(minGap, k.result.spectrum.gap);
      kept = std::format("{:.1e}", k.result.spectrum.smallestKept);
      gap = std::format("{:.1e}", k.result.spectrum.gap);
    }
    const std::string fd =
        haveFd ? std::format("{:18.10f}", c.steps.back().fdClass[i]) : std::format("{:>18}", "-");
    const std::string dEddot =
        haveFd ? std::format("{:9.2e}", std::fabs(c.steps.back().fdClass[i] - k.eddot))
               : std::format("{:>9}", "-");
    std::printf("%-6s %18.10f %9.2e %9.2e %18.10f %s %s %10s %9s %9s %9.1e\n", name,
                k.result.energy, dE, dEdot, k.eddot, fd.c_str(), dEddot.c_str(), euler.c_str(),
                kept.c_str(), gap.c_str(), k.lift);
  }
  ok = ok && computed == 8;
  std::printf("\nEddot (analytic, along u and v) = %.10f Ha\n", c.eddotTotal);
  if (full)
    std::printf("Euler: SKIPPED (FULL mode: the orbital energies respond to c, so no class is "
                "homogeneous of degree 2)\n");
  else
    std::printf(
        "Euler: worst |Eddot(c, u) - Edot(u)| over the five homogeneous classes = %.2e\n",
        maxEulerDelta);

  // The negative control: along a direction that breaks the state's symmetry
  // the dropped modes lift, and the Hessian must REFUSE (never a zero, never a
  // number). It passes when at least one class is refused for its lift and
  // every refusal there is a lift refusal.
  int liftRefused = 0, otherRefused = 0;
  std::printf("\nnegative control, (u, w) with w a random vector of amplitudes (the dropped modes "
              "must lift and be refused):\n");
  for (int i = 0; i < 8; ++i) {
    const PcClassHessian& k = c.control[i];
    const bool refused = k.result.status == PcStatus::Refused;
    const bool lifted = refused && k.lift > PC_LIFT_TOL;
    liftRefused += lifted ? 1 : 0;
    otherRefused += (refused && !lifted) || k.result.status == PcStatus::NotYet ? 1 : 0;
    std::printf("  %-6s lift %9.1e  %s\n", CLASSES[i].c_str(), k.lift,
                refused ? "REFUSED" : (k.result.status == PcStatus::Done ? "computed" : "not yet"));
  }
  const bool controlOk = liftRefused > 0 && otherRefused == 0;
  ok = ok && controlOk;
  std::printf("negative control: %d class(es) refused for their lift, %d otherwise -- %s\n",
              liftRefused, otherRefused, controlOk ? "ok" : "NO");

  ok = printPcLadder(c.steps, ratioTol, "the three-term Edot(v)", "Eddot") && ok;
  if (haveFd) {
    std::printf(
        "Richardson (4 fd(h_min) - fd(2 h_min)) / 3 = %.10f Ha   |delta vs Eddot|=%.2e\n",
        c.richardson, c.richardsonErr);
    ok = ok && c.richardsonErr < atol;
    std::printf("Hessian check: %d of 8 classes computed, max class |d E| vs pcEnergiesDevice "
                "%.2e, max |d Edot| vs pcTangentsDevice %.2e, %.1fs\n",
                computed, maxEnergyDelta, maxEdotDelta, c.seconds);
  }

  // A wording of its own, so no other entry's PASS line can satisfy this
  // entry's regex (or the other way round); it names the orbital-energy mode,
  // which is what licenses the Euler half (docs/hessian-theory.md, Sec. 4.3
  // and 9.1), and the conditioning the rotation terms carry beside the gap
  // (Sec. 6.2).
  if (full)
    std::printf("\n%s: the PC-NEVPT2 Hessian Eddot(u, v) matches the central difference of Edot "
                "(orbital-energy mode: FULL, the orbital energies responding to c, Euler skipped "
                "there; smallest kept s/s_max %.1e (%s), smallest gap %.1e against PC_MIN_GAP "
                "%.0e; Richardson to %.0e, every ratio 4 +- %.1f, a run of %d)\n",
                ok ? "PASS" : "FAIL", minKept, std::string(minKeptClass).c_str(), minGap,
                PC_MIN_GAP, atol, ratioTol, kPcMinFourfoldRun);
  else
    std::printf("\n%s: the PC-NEVPT2 Hessian Eddot(u, v) matches Euler and the central "
                "difference of Edot (orbital energies held fixed; smallest kept s/s_max %.1e "
                "(%s), smallest gap %.1e against PC_MIN_GAP %.0e; Richardson to %.0e, every "
                "ratio 4 +- %.1f, a run of %d)\n",
                ok ? "PASS" : "FAIL", minKept, std::string(minKeptClass).c_str(), minGap,
                PC_MIN_GAP, atol, ratioTol, kPcMinFourfoldRun);
  return ok;
}

// --- the Hessian-vector product, H_cc u (#155) -----------------------------------
//
// What is checked, and why the difference is of the THREE-term y, is at
// PcHessianVectorCheck in pc_tangent.cppm.

// The borrowed view nevpt2.gradient's entry points take.
RdmTensors rdmTensorsOf(const PcOwnedRdms& o) {
  return RdmTensors{o.dm1, o.dm2, o.rdm.dm3, o.rdm.f3ac, o.rdm.f3ca};
}

// a * every RDM of `o`: along the state itself the first-order set is 2 dm,
// and so is the second-order set at (c, c) -- every RDM is quadratic in c.
PcOwnedRdms pcScaledSet(const PcOwnedRdms& o, double a, const DeviceResources& res) {
  PcOwnedRdms out;
  out.dm1 = pcTangentScaled(o.dm1, a, res);
  out.dm2 = pcTangentScaled(o.dm2, a, res);
  out.rdm.dm3 = pcTangentScaled(o.rdm.dm3, a, res);
  out.rdm.f3ac = pcTangentScaled(o.rdm.f3ac, a, res);
  out.rdm.f3ca = pcTangentScaled(o.rdm.f3ca, a, res);
  out.hostDm1 = o.hostDm1;
  for (int64_t i = 0; i < out.hostDm1.size(); ++i) out.hostDm1.flatRef(i) *= a;
  return out;
}

double pcDot(const Tensor& x, const Tensor& y) {
  double s = 0.0;
  for (int64_t i = 0; i < x.size(); ++i) s += x.flat(i) * y.flat(i);
  return s;
}

double pcNorm(const Tensor& x) { return std::sqrt(pcDot(x, x)); }

// ||a - b||_2.
double pcDistance(const Tensor& a, const Tensor& b) {
  double s = 0.0;
  for (int64_t i = 0; i < a.size(); ++i) s += (a.flat(i) - b.flat(i)) * (a.flat(i) - b.flat(i));
  return std::sqrt(s);
}

// One displaced state's per-class three-term y (pcCiSourceThreeTermDevice),
// from its own host dm1/dm2 and device RDM build -- scoped, so one state's
// buffers at a time. `minGap` as PcDisplacedEnergies'. A class refused there
// is a Numerical Error naming the step.
struct PcDisplacedSource {
  std::array<Tensor, 8> yClass{};
  double minGap = 0.0;
};

Result<PcDisplacedSource> pcThreeTermSourceAt(const RdmBuildOptions& opt,
                                              const NevptIntegralsDevice& ints,
                                              const HostDm12Builder& builder, const Tensor& c,
                                              const Tensor& h2e, std::string_view what,
                                              const SigmaVectorOptions& sigma,
                                              const OrbitalEnergyResponse* w,
                                              const Tensor& dm1Base,
                                              const DeviceResources& res) {
  const int64_t norb = builder.norb(), na = builder.nelecA(), nb = builder.nelecB();
  const PcOwnedRdms base =
      pcOwned(builder.dm12(c), buildRdmsDevice(opt, c, norb, na, nb, h2e, res), res);
  // FULL mode (#214): the gated y, at this state's own orbital energies.
  const PcDisplacedInts at = pcDisplacedInts(ints, w, dm1Base, base.hostDm1, res);
  PcCiHessianResult r = NEVPT2_TRY(pcCiSourceThreeTermDevice(
      at.ints, c, na, nb, rdmTensorsOf(base), /*perClass=*/true, sigma, res, w));
  if (!r.ok)
    return err_numerical(std::format(
        "the PC CI source term is refused at the displaced state {}: {}; no finite difference "
        "of it can be formed there",
        what, r.walk.refusals()));
  PcDisplacedSource out;
  out.yClass = std::move(r.yClass);
  out.minGap = std::numeric_limits<double>::infinity();
  bool anySpectrum = false;
  for (const PcClassGradient& k : r.walk.classes)
    if (k.result.hasSpectrum) {
      anySpectrum = true;
      out.minGap = std::min(out.minGap, k.result.spectrum.gap);
    }
  if (!anySpectrum) out.minGap = 0.0;  // "not recorded", which the report fails on
  return out;
}

Result<PcHessianVectorCheck> pcHessianVectorCheck(const RdmBuildOptions& rdmOpt,
                                                  const NevptIntegralsDevice& ints,
                                                  const Tensor& ci, int64_t norb, int64_t nelecA,
                                                  int64_t nelecB, const Tensor& h2e,
                                                  std::span<const double> steps,
                                                  std::uint64_t seedU,
                                                  std::span<const std::uint64_t> seedsV,
                                                  std::uint64_t seedControl,
                                                  const OrbitalEnergyResponse* eps,
                                                  const DeviceResources& res) {
  const auto t0 = std::chrono::steady_clock::now();
  const HostDm12Builder builder =
      NEVPT2_TRY(pcCheckInputs(ci, norb, nelecA, nelecB, steps, "PC Hessian-vector check"));
  if (seedsV.empty())
    return err_config("the PC Hessian-vector check needs at least one direction v");
  SigmaVectorOptions sigma;
  sigma.nTiles = rdmOpt.nTiles;

  PcHessianVectorCheck out;
  out.orbitalEnergiesMove = eps != nullptr;
  const HostDm12 host = builder.dm12(ci);
  // u the Hessian check's own (the tangent's kind: S's dropped modes turn
  // along it but do not lift); every v the same kind.
  const Tensor u = pcTangentDirection(builder, ci, host.dm1, seedU);
  std::array<Tensor, 8> huClass{};
  {
    // The state: its RDMs, the sets along u, H_cc u; then every check that
    // needs the state's buffers. Scoped, so none is held through the steps.
    const PcOwnedRdms base = pcOwned(
        host, buildRdmsDevice(rdmOpt, ci, norb, nelecA, nelecB, h2e, res), res);
    const PcOwnedRdms dotU = pcTangentSet(rdmOpt, builder, ci, u, h2e, res);
    const PcOwnedRdms ddotUU = pcTangentSet(rdmOpt, builder, u, u, h2e, res);
    std::printf("\nH_cc u: the PC adjoint's walk differentiated along u, two sigma calls per "
                "class...\n");
    const PcCiHessianResult hu = NEVPT2_TRY(pcCiHessianVectorDevice(
        ints, ci, u, nelecA, nelecB, rdmTensorsOf(base), rdmTensorsOf(dotU),
        rdmTensorsOf(ddotUU), /*perClass=*/true, sigma, res, eps));
    for (std::size_t k = 0; k < 8; ++k) {
      out.result[k] = hu.walk.classes[k].result;
      out.lift[k] = hu.walk.classes[k].lift;
      out.walkEdot[k] = hu.walk.classes[k].ciEdot;
    }
    out.allDone = hu.ok;
    // A refused class is the answer, not an error: the report prints it and
    // fails the gate, and nothing is differenced.
    if (!hu.ok) return out;
    huClass = hu.huClass;
    out.uy = pcDot(u, hu.y);

    // 4. Euler: H_cc c, the first-order set along c and the (c, c) set both
    // 2 dm. An identity at fixed orbital energies only (docs/hessian-theory.md,
    // Sec. 4.3): in FULL mode it is not run, and the report says it was
    // skipped.
    const PcOwnedRdms two = pcScaledSet(base, 2.0, res);
    if (eps == nullptr) {
      std::printf("H_cc c, for Euler...\n");
      const PcCiHessianResult hc = NEVPT2_TRY(pcCiHessianVectorDevice(
          ints, ci, ci, nelecA, nelecB, rdmTensorsOf(base), rdmTensorsOf(two),
          rdmTensorsOf(two), /*perClass=*/true, sigma, res));
      for (std::size_t k = 0; k < 8; ++k) {
        if (!hc.ok) {
          out.euler[k] = std::numeric_limits<double>::infinity();  // refused along c: fails
          continue;
        }
        const double yn = pcNorm(hu.yClass[k]);
        const double d = pcDistance(hc.huClass[k], hu.yClass[k]);
        out.euler[k] = yn > 0.0 ? d / yn : d;
      }
    }

    // 2. The duality, against pcHessianDevice; 3. the symmetry, on the first v.
    // Both sides in the run's orbital-energy mode.
    const auto dualRow = [&](std::string what, const Tensor& v, const PcOwnedRdms& dotV,
                             const PcOwnedRdms& ddotUV, const PcOwnedRdms& ddotVV) -> Status {
      std::printf("Eddot(u, v) for v = %s (pcHessianDevice)...\n", what.c_str());
      const PcRdms vUU = ddotUU.view(), vUV = ddotUV.view(), vVV = ddotVV.view();
      const std::optional<OrbitalEnergyHessianDots> epsUV = pcEpsDots(eps, dotU, dotV, ddotUV);
      const PcHessianResult hess = NEVPT2_TRY(pcHessianDevice(
          ints,
          PcHessianSets{base.view(), dotU.view(), dotV.view(), &vUV, &vUU, &vVV,
                        epsUV ? &*epsUV : nullptr},
          res));
      PcHessianVectorCheck::Dual row;
      row.what = std::move(what);
      for (std::size_t k = 0; k < 8; ++k) {
        row.lhs[k] = pcDot(v, huClass[k]);
        // A class pcHessianDevice refuses for this pair has no Eddot: NaN,
        // which no tolerance accepts.
        row.rhs[k] = hess.classes[k].result.status == PcStatus::Done
                         ? hess.classes[k].eddot
                         : std::numeric_limits<double>::quiet_NaN();
        if (out.duals.empty()) out.edotRef[k] = hess.classes[k].edotU;
      }
      out.duals.push_back(std::move(row));
      return {};
    };
    for (std::size_t i = 0; i < seedsV.size(); ++i) {
      const Tensor v = pcTangentDirection(builder, ci, host.dm1, seedsV[i]);
      RdmHessianTangents three =
          buildRdmHessianTangentsDevice(rdmOpt, ci, u, v, norb, nelecA, nelecB, h2e, res);
      const PcOwnedRdms dotV = pcOwned(builder.dm12Dot(ci, v), std::move(three.cv), res);
      const PcOwnedRdms ddotUV = pcOwned(builder.dm12Dot(u, v), std::move(three.uv), res);
      const PcOwnedRdms ddotVV = pcTangentSet(rdmOpt, builder, v, v, h2e, res);
      NEVPT2_TRY(dualRow(std::format("seed 0x{:x}", seedsV[i]), v, dotV, ddotUV, ddotVV));
      if (i != 0) continue;
      std::printf("H_cc v, for the symmetry...\n");
      const PcCiHessianResult hv = NEVPT2_TRY(pcCiHessianVectorDevice(
          ints, ci, v, nelecA, nelecB, rdmTensorsOf(base), rdmTensorsOf(dotV),
          rdmTensorsOf(ddotVV), /*perClass=*/false, sigma, res, eps));
      out.vHu = pcDot(v, hu.hu);
      out.uHv = hv.ok ? pcDot(u, hv.hu) : std::numeric_limits<double>::quiet_NaN();
    }
    // v = c: the first-order set along c is 2 dm, the (u, c) set is the
    // tangent along u and the (c, c) set 2 dm.
    NEVPT2_TRY(dualRow("the state itself", ci, two, dotU, two));

    // 5. The negative control, in the run's orbital-energy mode.
    std::printf("the negative control: H_cc w along a symmetry-breaking w...\n");
    const Tensor w = pcHessianControlDirection(ci, seedControl);
    const PcOwnedRdms dotW = pcTangentSet(rdmOpt, builder, ci, w, h2e, res);
    const PcOwnedRdms ddotWW = pcTangentSet(rdmOpt, builder, w, w, h2e, res);
    const PcCiHessianResult hw = NEVPT2_TRY(pcCiHessianVectorDevice(
        ints, ci, w, nelecA, nelecB, rdmTensorsOf(base), rdmTensorsOf(dotW), rdmTensorsOf(ddotWW),
        /*perClass=*/false, sigma, res, eps));
    for (std::size_t k = 0; k < 8; ++k) {
      out.control[k] = hw.walk.classes[k].result;
      out.controlLift[k] = hw.walk.classes[k].lift;
    }
  }

  // 6. The finite difference of the three-term y, per class.
  Tensor huTotal(ci.dims());
  for (std::size_t k = 0; k < 8; ++k) {
    out.huNorm[k] = pcNorm(huClass[k]);
    ciSourceAddInto(huClass[k], huTotal);
  }
  out.huNorm[8] = pcNorm(huTotal);
  std::array<std::array<Tensor, 9>, 2> lastFd{};  // the two finest steps' quotients
  for (const double h : steps) {
    std::printf("\nfinite difference of y at h=%.6g: two more states, each an RDM build and a "
                "three-term y\n",
                h);
    const PcDisplacedSource plus = NEVPT2_TRY(
        pcThreeTermSourceAt(rdmOpt, ints, builder, pcTangentStep(ci, h, u), h2e,
                            std::format("c + {:g} u", h), sigma, eps, host.dm1, res));
    const PcDisplacedSource minus = NEVPT2_TRY(
        pcThreeTermSourceAt(rdmOpt, ints, builder, pcTangentStep(ci, -h, u), h2e,
                            std::format("c - {:g} u", h), sigma, eps, host.dm1, res));
    PcHessianVectorCheck::Step st;
    st.h = h;
    st.displacedGap = {plus.minGap, minus.minGap};
    std::array<Tensor, 9> fd;
    fd[8] = Tensor(ci.dims());
    for (std::size_t k = 0; k < 8; ++k) {
      fd[k] = Tensor(ci.dims());
      for (int64_t i = 0; i < ci.size(); ++i)
        fd[k].flatRef(i) = (plus.yClass[k].flat(i) - minus.yClass[k].flat(i)) / (2.0 * h);
      st.err[k] = pcDistance(fd[k], huClass[k]);
      ciSourceAddInto(fd[k], fd[8]);
    }
    st.err[8] = pcDistance(fd[8], huTotal);
    out.steps.push_back(st);
    lastFd[0] = std::move(lastFd[1]);
    lastFd[1] = std::move(fd);
  }
  const Tensor v0 = pcTangentDirection(builder, ci, host.dm1, seedsV[0]);
  for (std::size_t k = 0; k < 9; ++k) {
    Tensor rich(ci.dims());
    for (int64_t i = 0; i < ci.size(); ++i)
      rich.flatRef(i) = (4.0 * lastFd[1][k].flat(i) - lastFd[0][k].flat(i)) / 3.0;
    out.richardsonErr[k] = pcDistance(rich, k < 8 ? huClass[k] : huTotal);
    if (k == 8) {
      out.richardsonV = pcDot(v0, rich);
      out.huV = pcDot(v0, huTotal);
    }
  }
  out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

bool printPcHessianVectorReport(const PcHessianVectorCheck& c, double idtol, double dualTol,
                                double fdTol, double ratioTol) {
  const auto within = [](double a, double b, double tol) {
    return std::fabs(a - b) < tol * std::max(1.0, std::fabs(b));  // false on NaN
  };
  const bool full = c.orbitalEnergiesMove;
  std::printf("\nPC-NEVPT2 Hessian-vector product H_cc u with respect to the CI vector (the "
              "integrals %s)\n",
              full ? "held fixed, the orbital energies responding to c: FULL mode, Euler "
                     "SKIPPED, not an identity there"
                   : "and the orbital energies held fixed");
  std::printf(
      "  walk Edot    the walk's own three-term Edot(u) from its tangent rows, against\n"
      "               pcHessianDevice's (|d Edot|)\n"
      "  |H_cc u|     the class's share of the vector\n"
      "  Euler        ||H_cc c - y|| / ||y||, exact (0) only where the class is homogeneous of\n"
      "               degree 2; Sijr / Sij print theirs in brackets, Sijrs has y = 0\n"
      "  kept s       the smallest kept eigenvalue of S over s_max: S^+ in the rotation terms\n"
      "               scales as its inverse (the gap, PC_MIN_GAP's, does not bound it)\n"
      "  gap          smallest kept / largest dropped, the ratio PC_MIN_GAP guards\n"
      "  lift         the dropped modes' lift along (u, u) over s_max (refused above "
      "PC_LIFT_TOL)\n");
  std::printf("%-6s %18s %9s %14s %10s %9s %9s %9s\n", "class", "walk Edot (Ha)", "|d Edot|",
              "|H_cc u|", "Euler", "kept s", "gap", "lift");
  bool ok = true;
  int computed = 0;
  double minKept = std::numeric_limits<double>::infinity();
  double minGap = std::numeric_limits<double>::infinity();
  std::string_view minKeptClass = "-";
  for (int i = 0; i < 8; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    const PcClassResult& r = c.result[k];
    const char* name = CLASSES[k].c_str();
    if (r.status != PcStatus::Done) {
      ok = false;
      std::printf("%-6s %18s %9s %14s %10s %9s %9s %9.1e  (%s)\n", name,
                  r.status == PcStatus::Refused ? "REFUSED" : "not yet", "-", "-", "-", "-",
                  "-", c.lift[k], r.why.empty() ? "not computed" : r.why.c_str());
      continue;
    }
    ++computed;
    if (!c.allDone) continue;
    const double dEdot = std::fabs(c.walkEdot[k] - c.edotRef[k]);
    ok = ok && within(c.walkEdot[k], c.edotRef[k], idtol);
    std::string euler = full ? std::string("skipped") : std::format("[{:.2e}]", c.euler[k]);
    if (!full && kPcHomogeneousInCi[k]) {
      ok = ok && c.euler[k] < idtol;
      euler = std::format("{:.2e}", c.euler[k]);
    }
    std::string kept = "-", gap = "-";
    if (r.hasSpectrum) {
      kept = std::format("{:.1e}", r.spectrum.smallestKept);
      gap = std::format("{:.1e}", r.spectrum.gap);
      if (r.spectrum.smallestKept < minKept) {
        minKept = r.spectrum.smallestKept;
        minKeptClass = CLASSES[k];
      }
      minGap = std::min(minGap, r.spectrum.gap);
    }
    std::printf("%-6s %18.10f %9.2e %14.10f %10s %9s %9s %9.1e\n", name, c.walkEdot[k], dEdot,
                c.huNorm[k], euler.c_str(), kept.c_str(), gap.c_str(), c.lift[k]);
  }
  ok = ok && computed == 8 && c.allDone;
  if (!c.allDone) {
    std::printf("\nFAIL: the PC-NEVPT2 Hessian-vector product H_cc u was refused (above)\n");
    return false;
  }
  double edotTotal = 0.0;
  for (const double e : c.edotRef) edotTotal += e;
  const bool uyOk = within(c.uy, edotTotal, idtol);
  ok = ok && uyOk;
  std::printf("\n<u, y> of the walk's own three-term y = %.12f   Edot(u) = %.12f   |delta| = "
              "%.2e -- %s\n",
              c.uy, edotTotal, std::fabs(c.uy - edotTotal), uyOk ? "ok" : "NO");

  // The duality, per class and summed.
  std::printf("\n<v, H_cc u> against Eddot(u, v) (pcHessianDevice), per class; relative "
              "|delta| (floor 1)\n");
  std::printf("%-18s", "v");
  for (int i = 0; i < 8; ++i) std::printf(" %9s", CLASSES[static_cast<std::size_t>(i)].c_str());
  std::printf(" %20s %20s %9s\n", "<v, H_cc u>", "Eddot(u, v)", "total");
  double worstDual = 0.0;
  for (const PcHessianVectorCheck::Dual& row : c.duals) {
    std::printf("%-18s", row.what.c_str());
    double lhs = 0.0, rhs = 0.0;
    for (std::size_t k = 0; k < 8; ++k) {
      const double rel = std::fabs(row.lhs[k] - row.rhs[k]) / std::max(1.0, std::fabs(row.rhs[k]));
      ok = ok && within(row.lhs[k], row.rhs[k], dualTol);
      worstDual = std::max(worstDual, std::isnan(rel) ? std::numeric_limits<double>::infinity()
                                                      : rel);
      std::printf(" %9.1e", rel);
      lhs += row.lhs[k];
      rhs += row.rhs[k];
    }
    const double rel = std::fabs(lhs - rhs) / std::max(1.0, std::fabs(rhs));
    ok = ok && within(lhs, rhs, dualTol);
    worstDual = std::max(worstDual, std::isnan(rel) ? std::numeric_limits<double>::infinity() : rel);
    std::printf(" %20.12f %20.12f %9.1e\n", lhs, rhs, rel);
  }
  const bool symOk = within(c.vHu, c.uHv, dualTol);
  ok = ok && symOk;
  std::printf("symmetry: <v, H_cc u> = %.12f, <u, H_cc v> = %.12f, |delta| = %.2e -- %s\n", c.vHu,
              c.uHv, std::fabs(c.vHu - c.uHv), symOk ? "ok" : "NO");

  // The negative control, as the Hessian check's.
  int liftRefused = 0, otherRefused = 0;
  std::printf("\nnegative control, H_cc w with w a random vector of amplitudes (the dropped "
              "modes must lift and be refused):\n");
  for (std::size_t k = 0; k < 8; ++k) {
    const bool refused = c.control[k].status == PcStatus::Refused;
    const bool lifted = refused && c.controlLift[k] > PC_LIFT_TOL;
    liftRefused += lifted ? 1 : 0;
    otherRefused += (refused && !lifted) || c.control[k].status == PcStatus::NotYet ? 1 : 0;
    std::printf("  %-6s lift %9.1e  %s\n", CLASSES[k].c_str(), c.controlLift[k],
                refused ? "REFUSED"
                        : (c.control[k].status == PcStatus::Done ? "computed" : "not yet"));
  }
  const bool controlOk = liftRefused > 0 && otherRefused == 0;
  ok = ok && controlOk;
  std::printf("negative control: %d class(es) refused for their lift, %d otherwise -- %s\n",
              liftRefused, otherRefused, controlOk ? "ok" : "NO");

  // The finite difference, per class and summed: relative error per step,
  // the ratio, and the gate on every class with a nonzero H_cc u.
  std::printf("\ncentral differences of the three-term y along c + h u, per class: ||fd - H_cc "
              "u|| / ||H_cc u|| (ratio to the previous step)\n");
  std::printf("%12s", "h");
  for (int i = 0; i < 8; ++i) std::printf(" %16s", CLASSES[static_cast<std::size_t>(i)].c_str());
  std::printf(" %16s %16s\n", "total", "guard gap");
  bool guarded = true;
  for (std::size_t s = 0; s < c.steps.size(); ++s) {
    const PcHessianVectorCheck::Step& st = c.steps[s];
    std::printf("%12.6g", st.h);
    for (std::size_t k = 0; k < 9; ++k) {
      const double rel = c.huNorm[k] > 0.0 ? st.err[k] / c.huNorm[k] : st.err[k];
      if (s == 0)
        std::printf(" %9.2e %6s", rel, "-");
      else
        std::printf(" %9.2e %6.3f", rel,
                    st.err[k] > 0.0 ? c.steps[s - 1].err[k] / st.err[k] : 0.0);
    }
    const bool stepGuarded =
        st.displacedGap[0] >= PC_MIN_GAP && st.displacedGap[1] >= PC_MIN_GAP;
    guarded = guarded && stepGuarded;
    std::printf(" %.1e %.1e%s\n", st.displacedGap[0], st.displacedGap[1], stepGuarded ? "" : " NO");
  }
  ok = ok && guarded;
  std::printf("Richardson (4 fd(h_min) - fd(2 h_min)) / 3, relative error:");
  bool fdOk = true;
  for (std::size_t k = 0; k < 9; ++k) {
    const double rel =
        c.huNorm[k] > 0.0 ? c.richardsonErr[k] / c.huNorm[k] : c.richardsonErr[k];
    std::printf(" %s %.1e", k < 8 ? CLASSES[k].c_str() : "total", rel);
    // A class whose H_cc u is exactly zero (Sijrs; a class with no tuple)
    // has an exactly zero y everywhere: nothing to converge, nothing gated.
    if (c.huNorm[k] == 0.0) {
      fdOk = fdOk && c.richardsonErr[k] == 0.0;
      continue;
    }
    std::vector<double> errs;
    bool ratios = true;
    for (std::size_t s = 0; s < c.steps.size(); ++s) {
      errs.push_back(c.steps[s].err[k]);
      if (s > 0)
        ratios = ratios && c.steps[s].err[k] > 0.0 &&
                 std::fabs(c.steps[s - 1].err[k] / c.steps[s].err[k] - 4.0) < ratioTol;
    }
    const FourfoldRun run = longestFourfoldRun(errs, 4.0 - ratioTol, 4.0 + ratioTol);
    fdOk = fdOk && ratios && run.length >= kPcMinFourfoldRun && rel < fdTol;
  }
  std::printf("\n");
  if (!c.duals.empty())
    std::printf("projected on v = %s: <v, Richardson> = %.10f, <v, H_cc u> = %.10f, |delta| = "
                "%.2e (--pc-hessian's difference of Edot(v), read through y)\n",
                c.duals.front().what.c_str(), c.richardsonV, c.huV,
                std::fabs(c.richardsonV - c.huV));
  ok = ok && fdOk;
  std::printf("finite difference: every class with a nonzero H_cc u and the total fourfold at "
              "every halving (4 +- %.1f, a run of %d), Richardson within %.0e relative -- %s; "
              "the conditioning guard ran and passed at every displaced state -- %s\n",
              ratioTol, kPcMinFourfoldRun, fdTol, fdOk ? "ok" : "NO", guarded ? "ok" : "NO");
  std::printf("Hessian-vector check: %d of 8 classes computed, worst duality %.1e, %.1fs\n",
              computed, worstDual, c.seconds);

  std::printf("\n%s: the PC-NEVPT2 Hessian-vector product H_cc u matches Eddot(u, v) by duality "
              "to %.0e and the finite difference of the three-term y per class (%s; smallest "
              "kept s/s_max %.1e (%s), smallest gap %.1e against PC_MIN_GAP %.0e; Richardson to "
              "%.0e, every ratio 4 +- %.1f, a run of %d)\n",
              ok ? "PASS" : "FAIL", dualTol,
              full ? "orbital-energy mode: FULL, the orbital energies responding to c, Euler "
                     "skipped there"
                   : "orbital energies held fixed",
              minKept, std::string(minKeptClass).c_str(), minGap, PC_MIN_GAP, fdTol, ratioTol,
              kPcMinFourfoldRun);
  return ok;
}

}  // namespace nevpt2
