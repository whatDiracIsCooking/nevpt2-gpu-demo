// The PC tangent check's driver and report -- see pc_tangent.cppm for what is
// checked (Euler's identity, exact on the five classes it applies to, then a
// Richardson-verified central finite difference) and why a finite difference
// is the only independent handle on the derivative of a solve.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.pc_tangent;

import std;

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
Result<std::array<double, 8>> pcTangentEnergies(const RdmBuildOptions& opt,
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
  std::array<double, 8> out{};
  for (int i = 0; i < 8; ++i) {
    if (e.classes[i].status != PcStatus::Done)
      return err_numerical(std::format(
          "the PC class {} is refused at the displaced state {}: {}; no finite difference can "
          "be formed there",
          CLASSES[i], what, e.classes[i].why.empty() ? "not computed" : e.classes[i].why));
    out[i] = e.classes[i].energy;
  }
  return out;
}

Result<PcTangentCheck> pcTangentFdCheck(const RdmBuildOptions& rdmOpt,
                                       const NevptIntegralsDevice& ints, const Tensor& ci,
                                       int64_t norb, int64_t nelecA, int64_t nelecB,
                                       const Tensor& h2e, std::span<const double> steps,
                                       std::uint64_t seed, const DeviceResources& res) {
  const auto t0 = std::chrono::steady_clock::now();
  // The step list is the caller's input: two steps at least, each positive
  // and half its predecessor, because the Richardson ratio and the
  // extrapolation below are written for successive halvings.
  if (std::ssize(steps) < 2)
    return err_config("the PC tangent check needs at least two finite-difference steps");
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
    return err_io(std::format("the PC tangent check needs a normalized CI vector; <c|c> = {:.12f}",
                              ciNorm2));

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
                         rdmDot.f3ac, rdmDot.f3ca, res));
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
                         f3acTwo, f3caTwo, res));
    for (int i = 0; i < 8; ++i) out.eulerEdot[i] = euler.classes[i].edot;
  }

  double prevErr = 0.0;
  for (const double h : steps) {
    PcFdStep st;
    st.h = h;
    std::printf("\nfinite difference at h=%.6g: two more states, each a full RDM build\n", h);
    const std::array<double, 8> plus = NEVPT2_TRY(
        pcTangentEnergies(rdmOpt, ints, builder, pcTangentStep(ci, h, u), h2e,
                          std::format("c + {:g} u", h), res));
    const std::array<double, 8> minus = NEVPT2_TRY(
        pcTangentEnergies(rdmOpt, ints, builder, pcTangentStep(ci, -h, u), h2e,
                          std::format("c - {:g} u", h), res));
    for (int i = 0; i < 8; ++i) {
      st.fdClass[i] = (plus[i] - minus[i]) / (2.0 * h);
      st.fdTotal += st.fdClass[i];
    }
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

  if (haveFd) {
    std::printf(
        "\ncentral differences of the PC total along c + h u (second order: the error falls "
        "4x per halving)\n");
    std::printf("%12s %18s %12s %8s\n", "h", "fd (Ha)", "|fd - Edot|", "ratio");
    for (const PcFdStep& st : c.steps) {
      if (st.ratio > 0.0) {
        std::printf("%12.6g %18.10f %12.2e %8.3f\n", st.h, st.fdTotal, st.err, st.ratio);
        ok = ok && std::fabs(st.ratio - 4.0) < ratioTol;
      } else {
        std::printf("%12.6g %18.10f %12.2e %8s\n", st.h, st.fdTotal, st.err, "-");
      }
    }
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
              "to %.0e, every ratio 4 +- %.1f)\n",
              ok ? "PASS" : "FAIL", atol, ratioTol);
  return ok;
}

}  // namespace nevpt2
