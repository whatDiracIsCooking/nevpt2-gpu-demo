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
//                          adjoint's walk is block2's PC total.
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

}  // namespace nevpt2::test::pc_adjoint
