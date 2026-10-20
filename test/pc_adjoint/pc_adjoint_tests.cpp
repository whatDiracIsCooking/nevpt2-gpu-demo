// Suites for PC-NEVPT2's reverse mode (#74; nevpt2.gradient's :pc_classes
// partition and its entry points in gradient.cpp) -- the one property of the
// adjoint that no duality and no oracle can show, because both of those need
// a number to compare: WHERE THE PC SOLVE REFUSES A CLASS, THE ADJOINT
// REFUSES IT TOO, and says so, rather than handing back a derivative with that
// class silently summed in as zero.
//
//   PcAdjointRefusalTests  a state whose PC metric is NOT gapped -- the
//                          golden's CI vector displaced along a seeded random
//                          direction, which leaves the state's symmetry
//                          sector, so the metric gains rank and the PC stage
//                          refuses (docs/pc-nevpt2.md, "What the check can and
//                          cannot displace"): pcCiSourceDevice reports exactly
//                          pcEnergiesDevice's refusals, class by class, with
//                          their reasons, and forms no y; the forward tangent
//                          refuses the same classes; the digests' adjoint and
//                          the orbital source term's matrix are Numerical
//                          Errors naming the class. And the control: the
//                          golden's own state is accepted everywhere and its
//                          adjoint's walk is block2's PC total. And for PC's
//                          H_cc u (#155): pcCiHessianVectorDevice and the
//                          three-term y refuse the displaced state's classes
//                          and form no vector; along a symmetry-breaking
//                          direction H_cc w refuses exactly the classes
//                          pcHessianDevice refuses for their lift
//                          (PC_LIFT_TOL); and off the eigenstate the
//                          three-term y is the three-term Edot's gradient,
//                          which the two-term y is not.
//
// The numerical claims -- the dualities to round-off, block2's oracle, the
// nuclear gradient -- are the golden tier's: the `nevpt2_gradient_source_pc_*`,
// `nevpt2_sc_pseudodensity_dump_pc_*` and `nevpt2_sc_pseudodensity_oracle_pc_*`
// entries in the root CMakeLists.txt, and docs/testing.md's PC sections.
//
// REQUIRES_GPU: every case runs the RDM build and the PC solves on the
// process's one DeviceResources (nevpt2.test.shared_resources).
//
// TU shape: gtest's header FIRST, then `import std;` and the modules.
#include <gtest/gtest.h>

import std;
import nevpt2.golden;
import nevpt2.gradient;
import nevpt2.rdm_host_dm;
import nevpt2.test.shared_resources;

#ifndef NEVPT2_TEST_GOLDEN_CAS44
#error "NEVPT2_TEST_GOLDEN_CAS44 must be defined by test/pc_adjoint/CMakeLists.txt"
#endif
#ifndef NEVPT2_TEST_GRADGOLD_CAS44
#error "NEVPT2_TEST_GRADGOLD_CAS44 must be defined by test/pc_adjoint/CMakeLists.txt"
#endif

namespace nevpt2::test::pc_adjoint {

constexpr std::string_view kGolden = NEVPT2_TEST_GOLDEN_CAS44;
constexpr std::string_view kGradGold = NEVPT2_TEST_GRADGOLD_CAS44;
// The displacement off the state: large enough that the symmetry-breaking
// modes it lifts clear PC_TAU (docs/pc-nevpt2.md measures them at ~h^2
// relative), so the gap or the denominators fail -- a REFUSED state, not a
// borderline one.
constexpr double kDisplacement = 0.25;
constexpr std::uint64_t kSeed = 0x5eed'0074u;

const DeviceResources* resourcesOrFail() {
  const auto& res = sharedResources();
  if (!res.has_value()) {
    ADD_FAILURE() << "DeviceResources::create failed: " << res.error().message;
    return nullptr;
  }
  return res->get();
}

// One state's device inputs: the golden's integrals, and the RDMs of `ci`
// (host dm1/dm2, device dm3/f3ac/f3ca), as the demos build them.
struct PcState {
  GoldenFile g;
  Tensor ci;
  NevptIntegralsDevice ints;
  DeviceTensor dm1, dm2;
  RdmBuildResult rdm;
};

std::unique_ptr<PcState> loadState(double displacement, const DeviceResources& res);

std::unique_ptr<PcState> loadState(bool displaced, const DeviceResources& res) {
  return loadState(displaced ? kDisplacement : 0.0, res);
}

std::unique_ptr<PcState> loadState(double displacement, const DeviceResources& res) {
  const bool displaced = displacement != 0.0;
  Result<GoldenFile> g = loadGolden(std::string(kGolden));
  if (!g) {
    ADD_FAILURE() << "loadGolden: " << g.error().message;
    return nullptr;
  }
  auto st = std::make_unique<PcState>();
  st->g = *std::move(g);
  st->ci = st->g.get("ci");
  if (displaced) {
    std::mt19937_64 gen(kSeed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    double norm2 = 0.0;
    for (int64_t i = 0; i < st->ci.size(); ++i) {
      st->ci.flatRef(i) += displacement * u(gen);
      norm2 += st->ci.flat(i) * st->ci.flat(i);
    }
    for (int64_t i = 0; i < st->ci.size(); ++i) st->ci.flatRef(i) /= std::sqrt(norm2);
  }
  Result<HostDm12Builder> b = HostDm12Builder::create(st->g.ncas, st->g.nelecA, st->g.nelecB);
  if (!b) {
    ADD_FAILURE() << "HostDm12Builder::create: " << b.error().message;
    return nullptr;
  }
  const HostDm12 dm = b->dm12(st->ci);
  st->dm1 = uploadTensor(dm.dm1, res);
  st->dm2 = uploadTensor(dm.dm2, res);
  const auto up = [&](const char* name) { return uploadTensor(st->g.get(name), res); };
  NevptIntegralsDevice& d = st->ints;
  d.h1e = up("h1e");
  d.h2e = up("h2e");
  d.e_core = up("e_core");
  d.e_virt = up("e_virt");
  d.h2e_v_Sr = up("h2e_v_Sr");
  d.h1e_v_Sr = up("h1e_v_Sr");
  d.h2e_v_Si = up("h2e_v_Si");
  d.h1e_v_Si = up("h1e_v_Si");
  d.cvcv = up("cvcv");
  d.h2e_v_Sijr = up("h2e_v_Sijr");
  d.h2e_v_Srsi = up("h2e_v_Srsi");
  d.h2e_v_Srs = up("h2e_v_Srs");
  d.h2e_v_Sij = up("h2e_v_Sij");
  d.h2e_v1_Sir = up("h2e_v1_Sir");
  d.h2e_v2_Sir = up("h2e_v2_Sir");
  d.h1e_v_Sir = up("h1e_v_Sir");
  st->rdm = buildRdmsDevice(RdmBuildOptions{}, st->ci, st->g.ncas, st->g.nelecA, st->g.nelecB,
                            st->g.get("h2e"), res);
  return st;
}

GeneralIntegralsDevice loadGeneral(const DeviceResources& res) {
  Result<GradGoldFile> gg = loadGradGold(std::string(kGradGold));
  if (!gg) {
    ADD_FAILURE() << "loadGradGold: " << gg.error().message;
    return {};
  }
  const auto up = [&](const char* name) { return uploadTensor(gg->get(name), res); };
  GeneralIntegralsDevice d;
  d.ncore = gg->ncore;
  d.nact = gg->nact;
  d.nvirt = gg->nvirt;
  d.h = up("h");
  d.heff = up("heff");
  d.f = up("f");
  d.g_xa_aa = up("g_xa_aa");
  d.g_xv_aa = up("g_xv_aa");
  d.g_xc_aa = up("g_xc_aa");
  d.g_xa_va = up("g_xa_va");
  d.g_xv_va = up("g_xv_va");
  d.g_xc_va = up("g_xc_va");
  d.g_xa_ca = up("g_xa_ca");
  d.g_xv_ca = up("g_xv_ca");
  d.g_xc_ca = up("g_xc_ca");
  d.g_xa_cv = up("g_xa_cv");
  d.g_xv_cv = up("g_xv_cv");
  d.g_xc_cv = up("g_xc_cv");
  d.g_xa_cc = up("g_xa_cc");
  d.g_xv_cc = up("g_xv_cc");
  d.g_xc_cc_j = up("g_xc_cc_j");
  d.g_xc_cc_k = up("g_xc_cc_k");
  d.g_xa_vv = up("g_xa_vv");
  d.g_xc_vv = up("g_xc_vv");
  return d;
}

Result<PcCiSourceResult> ciSource(const PcState& st, const DeviceResources& res) {
  return pcCiSourceDevice(st.ints, st.ci, st.g.nelecA, st.g.nelecB, st.dm1, st.dm2, st.rdm.dm3,
                          st.rdm.f3ac, st.rdm.f3ca, SigmaVectorOptions{}, /*eps=*/nullptr, res);
}

// The names of the classes `r` refuses, in CLASSES order.
std::vector<std::string> refusedOf(const PcEnergyResult& r) {
  std::vector<std::string> out;
  for (std::size_t c = 0; c < 8; ++c)
    if (r.classes[c].status == PcStatus::Refused) out.push_back(CLASSES[c]);
  return out;
}

TEST(PcAdjointRefusalTests, TheCiSourceTermRefusesExactlyWhatTheEnergyRefuses) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const std::unique_ptr<PcState> st = loadState(/*displaced=*/true, *res);
  ASSERT_NE(st, nullptr);
  const Result<PcEnergyResult> energy = pcEnergiesDevice(st->ints, st->dm1, st->dm2, st->rdm.dm3,
                                                         st->rdm.f3ac, st->rdm.f3ca, *res);
  ASSERT_TRUE(energy.has_value()) << energy.error().message;
  const std::vector<std::string> refused = refusedOf(*energy);
  // The premise: this state IS refused somewhere. If a kernel change ever
  // made it accepted, the test would otherwise pass vacuously.
  ASSERT_FALSE(refused.empty()) << "the displaced state was accepted by every PC class";

  const Result<PcCiSourceResult> src = ciSource(*st, *res);
  ASSERT_TRUE(src.has_value()) << src.error().message;
  EXPECT_FALSE(src->ok);
  EXPECT_FALSE(src->walk.allDone());
  // No y at all: never one with the refused classes summed in as zero.
  EXPECT_EQ(src->y.size(), 0);
  for (std::size_t c = 0; c < 8; ++c) {
    SCOPED_TRACE(CLASSES[c]);
    const PcClassResult& got = src->walk.classes[c].result;
    EXPECT_EQ(got.status, energy->classes[c].status);
    if (got.status == PcStatus::Refused) EXPECT_FALSE(got.why.empty());
  }
  const std::string why = src->walk.refusals();
  std::printf("the displaced state's refusals, as the adjoint reports them: %s\n", why.c_str());
  for (const std::string& name : refused) EXPECT_NE(why.find(name), std::string::npos) << why;
}

TEST(PcAdjointRefusalTests, APcMinGapRefusalIsTheAdjointsToo) {
  // The gap itself (#94's pcRankCut), not a denominator: a displacement so
  // small that the symmetry-breaking modes it lifts (~h^2 relative) sit
  // astride PC_TAU, so one is kept within PC_MIN_GAP of a dropped one. The
  // first of a fixed, seeded ladder of steps that does it is used -- which
  // one is the state's business; that one does is asserted.
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  std::unique_ptr<PcState> st;
  std::vector<std::string> gapRefused;
  double used = 0.0;
  for (const double h : {1e-6, 7e-7, 5e-7, 3e-7, 2e-7, 1.5e-7, 1e-7}) {
    st = loadState(h, *res);
    ASSERT_NE(st, nullptr);
    const Result<PcEnergyResult> energy = pcEnergiesDevice(
        st->ints, st->dm1, st->dm2, st->rdm.dm3, st->rdm.f3ac, st->rdm.f3ca, *res);
    ASSERT_TRUE(energy.has_value()) << energy.error().message;
    gapRefused.clear();
    for (std::size_t c = 0; c < 8; ++c)
      if (energy->classes[c].status == PcStatus::Refused &&
          energy->classes[c].why.find("PC_MIN_GAP") != std::string::npos)
        gapRefused.push_back(CLASSES[c]);
    if (!gapRefused.empty()) {
      used = h;
      break;
    }
  }
  ASSERT_FALSE(gapRefused.empty()) << "no step of the ladder left a class astride the gap";
  const Result<PcCiSourceResult> src = ciSource(*st, *res);
  ASSERT_TRUE(src.has_value()) << src.error().message;
  EXPECT_FALSE(src->ok);
  EXPECT_EQ(src->y.size(), 0);
  const std::string why = src->walk.refusals();
  std::printf("displaced by %.1e: the adjoint's refusals: %s\n", used, why.c_str());
  for (const std::string& name : gapRefused) {
    const std::size_t at = why.find(name);
    ASSERT_NE(at, std::string::npos) << why;
    EXPECT_NE(why.find("PC_MIN_GAP", at), std::string::npos) << why;
  }
}

TEST(PcAdjointRefusalTests, TheForwardTangentRefusesTheSameClasses) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const std::unique_ptr<PcState> st = loadState(/*displaced=*/true, *res);
  ASSERT_NE(st, nullptr);
  const Result<PcEnergyResult> energy = pcEnergiesDevice(st->ints, st->dm1, st->dm2, st->rdm.dm3,
                                                         st->rdm.f3ac, st->rdm.f3ca, *res);
  ASSERT_TRUE(energy.has_value()) << energy.error().message;
  const DeviceTensor z1 = DeviceTensor::zeros(st->dm1.dims, *res);
  const DeviceTensor z2 = DeviceTensor::zeros(st->dm2.dims, *res);
  const DeviceTensor z3 = DeviceTensor::zeros(st->rdm.dm3.dims, *res);
  const Result<PcTangentResult> t =
      pcTangentsDevice(st->ints, st->dm1, st->dm2, st->rdm.dm3, st->rdm.f3ac, st->rdm.f3ca, z1, z2,
                       z3, z3, z3, /*epsDot=*/nullptr, *res);
  ASSERT_TRUE(t.has_value()) << t.error().message;
  const Result<PcCiSourceResult> src = ciSource(*st, *res);
  ASSERT_TRUE(src.has_value()) << src.error().message;
  for (std::size_t c = 0; c < 8; ++c) {
    SCOPED_TRACE(CLASSES[c]);
    EXPECT_EQ(t->classes[c].result.status, src->walk.classes[c].result.status);
  }
}

TEST(PcAdjointRefusalTests, TheDigestsAdjointAndTheOrbitalMatrixAreNumericalErrors) {
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const std::unique_ptr<PcState> st = loadState(/*displaced=*/true, *res);
  ASSERT_NE(st, nullptr);
  const Result<PcEnergyResult> energy = pcEnergiesDevice(st->ints, st->dm1, st->dm2, st->rdm.dm3,
                                                         st->rdm.f3ac, st->rdm.f3ca, *res);
  ASSERT_TRUE(energy.has_value()) << energy.error().message;
  const std::vector<std::string> refused = refusedOf(*energy);
  ASSERT_FALSE(refused.empty());

  const Result<DeviceTensor> hbar =
      pcDigestAdjointDevice(RdmBuildOptions{}, st->ci, st->g.nelecA, st->g.nelecB, st->ints,
                            st->dm1, st->dm2, st->rdm.dm3, st->rdm.f3ac, st->rdm.f3ca, *res);
  ASSERT_FALSE(hbar.has_value());
  EXPECT_EQ(hbar.error().kind, ErrorKind::Numerical);
  EXPECT_NE(hbar.error().message.find(refused.front()), std::string::npos)
      << hbar.error().message;

  const GeneralIntegralsDevice gen = loadGeneral(*res);
  const Result<Tensor> y =
      pcFullRotationMatrix(st->ints, gen, st->dm1, st->dm2, st->rdm.dm3, st->rdm.f3ac,
                           st->rdm.f3ca, /*digestAdjoint=*/nullptr, {{0, 1}, {5, 9}}, 16, *res);
  ASSERT_FALSE(y.has_value());
  EXPECT_EQ(y.error().kind, ErrorKind::Numerical);
  EXPECT_NE(y.error().message.find(refused.front()), std::string::npos) << y.error().message;
}

TEST(PcAdjointRefusalTests, TheGoldenStateIsAcceptedAndItsWalkIsBlock2s) {
  // The control: the same code on the state itself refuses nothing, so the
  // refusals above are the state's and not the adjoint's.
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const std::unique_ptr<PcState> st = loadState(/*displaced=*/false, *res);
  ASSERT_NE(st, nullptr);
  const Result<PcCiSourceResult> src = ciSource(*st, *res);
  ASSERT_TRUE(src.has_value()) << src.error().message;
  EXPECT_TRUE(src->ok) << src->walk.refusals();
  EXPECT_EQ(src->y.dims(), st->ci.dims());
  EXPECT_NEAR(src->walk.total, st->g.get("e_pc_total").flat(0), 1e-7);
}

// --- the Hessian-vector product's refusals and its three-term y (#155) -------

// One RDM set, owning: host dm1/dm2 and device dm3/f3ac/f3ca.
struct OwnedSet {
  DeviceTensor dm1, dm2;
  RdmBuildResult rdm;
  RdmTensors view() const { return {dm1, dm2, rdm.dm3, rdm.f3ac, rdm.f3ca}; }
  PcRdms pc() const { return {&dm1, &dm2, &rdm.dm3, &rdm.f3ac, &rdm.f3ca}; }
};

// The tangent build at (a, b): the RDMs' derivative at a along b, or their
// second derivative when a is a direction too.
OwnedSet tangentSet(const PcState& st, const Tensor& a, const Tensor& b,
                    const DeviceResources& res) {
  Result<HostDm12Builder> builder =
      HostDm12Builder::create(st.g.ncas, st.g.nelecA, st.g.nelecB);
  EXPECT_TRUE(builder.has_value());
  const HostDm12 d = builder->dm12Dot(a, b);
  return {uploadTensor(d.dm1, res), uploadTensor(d.dm2, res),
          buildRdmTangentsDevice(RdmBuildOptions{}, a, b, st.g.ncas, st.g.nelecA, st.g.nelecB,
                                 st.g.get("h2e"), res)};
}

// A seeded spin-free direction over the state's symmetry mask, projected
// against the state and normalized: the kind along which the PC metric's
// dropped modes turn but do not lift (nevpt2.pc_tangent's own construction).
Tensor maskedDirection(const PcState& st, std::uint64_t seed) {
  Result<HostDm12Builder> builder =
      HostDm12Builder::create(st.g.ncas, st.g.nelecA, st.g.nelecB);
  EXPECT_TRUE(builder.has_value());
  const std::vector<double> e = builder->applyAllE(st.ci);
  const Tensor dm1 = builder->dm12(st.ci).dm1;
  const int64_t n = st.g.ncas, ndet = st.ci.size();
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Tensor out(st.ci.dims());
  for (int64_t p = 0; p < n; ++p)
    for (int64_t q = 0; q < n; ++q) {
      if (std::fabs(dm1(p, q)) <= 1e-12) continue;
      const double w = u(gen);
      for (int64_t k = 0; k < ndet; ++k)
        out.flatRef(k) += w * e[static_cast<std::size_t>((p * n + q) * ndet + k)];
    }
  double ov = 0.0, norm2 = 0.0;
  for (int64_t k = 0; k < ndet; ++k) ov += out.flat(k) * st.ci.flat(k);
  for (int64_t k = 0; k < ndet; ++k) out.flatRef(k) -= ov * st.ci.flat(k);
  for (int64_t k = 0; k < ndet; ++k) norm2 += out.flat(k) * out.flat(k);
  for (int64_t k = 0; k < ndet; ++k) out.flatRef(k) /= std::sqrt(norm2);
  return out;
}

double dotOf(const Tensor& a, const Tensor& b) {
  double s = 0.0;
  for (int64_t i = 0; i < a.size(); ++i) s += a.flat(i) * b.flat(i);
  return s;
}

TEST(PcAdjointRefusalTests, TheHessianVectorRefusesExactlyWhatTheEnergyRefuses) {
  // The displaced state again: H_cc u and the three-term y refuse its
  // classes, name them, and form no vector -- whatever the direction.
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const std::unique_ptr<PcState> st = loadState(/*displaced=*/true, *res);
  ASSERT_NE(st, nullptr);
  const Result<PcEnergyResult> energy = pcEnergiesDevice(st->ints, st->dm1, st->dm2, st->rdm.dm3,
                                                         st->rdm.f3ac, st->rdm.f3ca, *res);
  ASSERT_TRUE(energy.has_value()) << energy.error().message;
  ASSERT_FALSE(refusedOf(*energy).empty());
  const RdmTensors base{st->dm1, st->dm2, st->rdm.dm3, st->rdm.f3ac, st->rdm.f3ca};
  const DeviceTensor z1 = DeviceTensor::zeros(st->dm1.dims, *res);
  const DeviceTensor z2 = DeviceTensor::zeros(st->dm2.dims, *res);
  const DeviceTensor z3 = DeviceTensor::zeros(st->rdm.dm3.dims, *res);
  const RdmTensors zero{z1, z2, z3, z3, z3};
  const Tensor u(st->ci.dims());
  const Result<PcCiHessianResult> hu =
      pcCiHessianVectorDevice(st->ints, st->ci, u, st->g.nelecA, st->g.nelecB, base, zero, zero,
                              /*perClass=*/true, SigmaVectorOptions{}, *res);
  ASSERT_TRUE(hu.has_value()) << hu.error().message;
  const Result<PcCiHessianResult> y3 = pcCiSourceThreeTermDevice(
      st->ints, st->ci, st->g.nelecA, st->g.nelecB, base, false, SigmaVectorOptions{}, *res);
  ASSERT_TRUE(y3.has_value()) << y3.error().message;
  for (const PcCiHessianResult* r : {&*hu, &*y3}) {
    EXPECT_FALSE(r->ok);
    EXPECT_EQ(r->hu.size(), 0);
    EXPECT_EQ(r->y.size(), 0);
    for (std::size_t c = 0; c < 8; ++c) {
      SCOPED_TRACE(CLASSES[c]);
      EXPECT_EQ(r->walk.classes[c].result.status, energy->classes[c].status);
      EXPECT_EQ(r->huClass[c].size(), 0);
    }
  }
}

TEST(PcAdjointRefusalTests, ALiftAlongTheDirectionIsRefusedAsPcHessianDeviceRefusesIt) {
  // At the golden state, along a random amplitude vector (which breaks the
  // state's symmetry): the classes whose dropped modes lift are refused by
  // H_cc w for PC_LIFT_TOL, exactly those pcHessianDevice refuses along
  // (w, w), and no vector is formed.
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  const std::unique_ptr<PcState> st = loadState(/*displaced=*/false, *res);
  ASSERT_NE(st, nullptr);
  Tensor w(st->ci.dims());
  {
    std::mt19937_64 gen(kSeed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int64_t i = 0; i < w.size(); ++i) w.flatRef(i) = u(gen);
  }
  const OwnedSet dotW = tangentSet(*st, st->ci, w, *res);
  const OwnedSet ddotWW = tangentSet(*st, w, w, *res);
  const RdmTensors base{st->dm1, st->dm2, st->rdm.dm3, st->rdm.f3ac, st->rdm.f3ca};
  const Result<PcCiHessianResult> hw =
      pcCiHessianVectorDevice(st->ints, st->ci, w, st->g.nelecA, st->g.nelecB, base, dotW.view(),
                              ddotWW.view(), /*perClass=*/false, SigmaVectorOptions{}, *res);
  ASSERT_TRUE(hw.has_value()) << hw.error().message;
  const PcRdms b{&st->dm1, &st->dm2, &st->rdm.dm3, &st->rdm.f3ac, &st->rdm.f3ca};
  const PcRdms vW = dotW.pc(), vWW = ddotWW.pc();
  const Result<PcHessianResult> hess =
      pcHessianDevice(st->ints, PcHessianSets{b, vW, vW, &vWW, &vWW, &vWW}, *res);
  ASSERT_TRUE(hess.has_value()) << hess.error().message;
  EXPECT_FALSE(hw->ok);
  EXPECT_EQ(hw->hu.size(), 0);
  int lifted = 0;
  for (std::size_t c = 0; c < 8; ++c) {
    SCOPED_TRACE(CLASSES[c]);
    const PcClassResult& got = hw->walk.classes[c].result;
    EXPECT_EQ(got.status, hess->classes[c].result.status);
    if (got.status != PcStatus::Refused) continue;
    ++lifted;
    EXPECT_NE(got.why.find("PC_LIFT_TOL"), std::string::npos) << got.why;
    EXPECT_GT(hw->walk.classes[c].lift, PC_LIFT_TOL);
  }
  EXPECT_GT(lifted, 0);
}

TEST(PcAdjointRefusalTests, TheThreeTermYIsTheGradientOffTheEigenstate) {
  // Off the eigenstate -- displaced along a masked direction, so every class
  // is still accepted -- <v, y> of the three-term y is pcHessianDevice's
  // three-term Edot(v), exact at any CI vector, and pcCiSourceDevice's
  // two-term y is pcTangentsDevice's two-term Edot(v); the two differ by the
  // -2 rho^T S_v P A x term, which is not small here (docs/hessian-theory.md,
  // Sec. 6.2). At the eigenstate it vanishes, so a finite difference of the
  // two-term y would difference the wrong function.
  const DeviceResources* res = resourcesOrFail();
  ASSERT_NE(res, nullptr);
  std::unique_ptr<PcState> st = loadState(/*displaced=*/false, *res);
  ASSERT_NE(st, nullptr);
  const Tensor u = maskedDirection(*st, 0x5eed'0155u);
  // 2^-11: well inside the step window nevpt2_pc_tangent's ladders use, so
  // Sir's h^4 mode stays under PC_TAU.
  const double h = std::ldexp(1.0, -11);
  for (int64_t i = 0; i < st->ci.size(); ++i) st->ci.flatRef(i) += h * u.flat(i);
  double norm2 = 0.0;
  for (int64_t i = 0; i < st->ci.size(); ++i) norm2 += st->ci.flat(i) * st->ci.flat(i);
  for (int64_t i = 0; i < st->ci.size(); ++i) st->ci.flatRef(i) /= std::sqrt(norm2);
  {
    Result<HostDm12Builder> builder =
        HostDm12Builder::create(st->g.ncas, st->g.nelecA, st->g.nelecB);
    ASSERT_TRUE(builder.has_value());
    const HostDm12 dm = builder->dm12(st->ci);
    st->dm1 = uploadTensor(dm.dm1, *res);
    st->dm2 = uploadTensor(dm.dm2, *res);
    st->rdm = buildRdmsDevice(RdmBuildOptions{}, st->ci, st->g.ncas, st->g.nelecA,
                              st->g.nelecB, st->g.get("h2e"), *res);
  }
  const Tensor v = maskedDirection(*st, 0x5eed'1155u);
  const OwnedSet dotV = tangentSet(*st, st->ci, v, *res);
  const RdmTensors base{st->dm1, st->dm2, st->rdm.dm3, st->rdm.f3ac, st->rdm.f3ca};
  const Result<PcCiHessianResult> y3 = pcCiSourceThreeTermDevice(
      st->ints, st->ci, st->g.nelecA, st->g.nelecB, base, false, SigmaVectorOptions{}, *res);
  ASSERT_TRUE(y3.has_value()) << y3.error().message;
  ASSERT_TRUE(y3->ok) << y3->walk.refusals();
  const Result<PcCiSourceResult> y2 = ciSource(*st, *res);
  ASSERT_TRUE(y2.has_value()) << y2.error().message;
  ASSERT_TRUE(y2->ok) << y2->walk.refusals();
  const PcRdms b{&st->dm1, &st->dm2, &st->rdm.dm3, &st->rdm.f3ac, &st->rdm.f3ca};
  const PcRdms vV = dotV.pc();
  const Result<PcHessianResult> three = pcHessianDevice(st->ints, PcHessianSets{b, vV, vV}, *res);
  ASSERT_TRUE(three.has_value()) << three.error().message;
  const Result<PcTangentResult> twoTerm = pcTangentsDevice(
      st->ints, st->dm1, st->dm2, st->rdm.dm3, st->rdm.f3ac, st->rdm.f3ca, dotV.dm1, dotV.dm2,
      dotV.rdm.dm3, dotV.rdm.f3ac, dotV.rdm.f3ca, /*epsDot=*/nullptr, *res);
  ASSERT_TRUE(twoTerm.has_value()) << twoTerm.error().message;
  double e3 = 0.0, e2 = 0.0;
  for (std::size_t c = 0; c < 8; ++c) {
    ASSERT_EQ(three->classes[c].result.status, PcStatus::Done) << CLASSES[c];
    e3 += three->classes[c].edotV;
    e2 += twoTerm->classes[c].edot;
  }
  const double vy3 = dotOf(v, y3->y), vy2 = dotOf(v, y2->y);
  std::printf("off the eigenstate (h = 2^-11): <v, y3> = %.15f, three-term Edot = %.15f; "
              "<v, y2> = %.15f, two-term Edot = %.15f\n",
              vy3, e3, vy2, e2);
  EXPECT_NEAR(vy3, e3, 1e-10 * std::max(1.0, std::fabs(e3)));
  EXPECT_NEAR(vy2, e2, 1e-10 * std::max(1.0, std::fabs(e2)));
  EXPECT_GT(std::fabs(e3 - e2), 1e3 * std::fabs(vy3 - e3));
}

}  // namespace nevpt2::test::pc_adjoint
