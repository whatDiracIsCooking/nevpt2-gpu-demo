// nevpt2.source_compare -- the block-by-block comparison of a GPU's source
// terms, Y_xy and y_I, with a committed source-term oracle (`*.srcgold`, SC
// #92 or PC #95), and THE known-gap list it gates on. Pure host arithmetic.
//
// WHY A LIBRARY. --source-oracle (:source_oracle) is the only caller, for
// both schemes -- SC's terms, and with --pc PC's (#74) -- so the comparison,
// the known-gap list and both tolerances live here, apart from the device code that produces what is compared -- and the
// unit tier (test/source_compare/) reads every committed oracle through it
// with no card: an oracle compared with itself passes, and a copy with one
// block perturbed fails, naming the block. It sits under apps/ beside
// nevpt2.gradsrc for gradsrc's reason: what the binary decides about what
// nevpt2.gradient computed is the binary's business.
//
// WHAT IS COMPARED (reference_data/sc_source_oracle.py's docstring has the
// arrays):
//
//   A = Y - Y^T, carried into the oracle's basis (A' = Wf^T A Wf, Wf = W =
//         act_rot on the active block and 1 elsewhere: nevpt2_grad's
//         load_source_terms transform) -- oracleBasisA -- against a_corr on
//         every non-gauge pair, grouped into the eight blocks of the four
//         spaces core / act_O / act_V / virt;
//   P y = y - <c, y> c -- projectOffState -- against py on the determinants
//         of the state's irrep (ci_mask), in the golden's own basis.
//
// THE TOLERANCES ARE NOT 1e-7: the oracle's side is a central difference,
// and its truncation at the committed steps is what the agreement can be read
// to. Each is measured from its own step ladders, and each has ONE home,
// read at configure time (apps/sc_pseudodensity/CMakeLists.txt): SC's
// docs/testing.md, "The SC source terms against PySCF's, on a card"; PC's
// docs/testing.md, "The PC source terms' oracle".
//
// THE KNOWN GAPS ARE PINNED, NOT HIDDEN. kKnownGaps is the ONE list of
// blocks the device is known to get wrong. A block on it must still DISAGREE
// by more than the tolerance and every other block must AGREE within it, so
// a comparison fails if a correct block breaks AND if a fix lands without the
// list being updated.
module;

#ifndef NEVPT2_SOURCE_ORACLE_TOL
#error "NEVPT2_SOURCE_ORACLE_TOL must be defined by apps/sc_pseudodensity/CMakeLists.txt"
#endif
#ifndef NEVPT2_PC_SOURCE_ORACLE_TOL
#error "NEVPT2_PC_SOURCE_ORACLE_TOL must be defined by apps/sc_pseudodensity/CMakeLists.txt"
#endif

export module nevpt2.source_compare;

import std;
// Re-exported: every entry point takes a SrcGoldFile, and the classification
// hands back a Result.
export import nevpt2.golden;

export namespace nevpt2 {

// Per element, absolute (Eh): what every block not on kKnownGaps must agree
// to, and what every block on it must still miss by. One per scheme, each
// measured from its own ladders (the header comment says where).
inline constexpr double kSourceOracleTol = NEVPT2_SOURCE_ORACLE_TOL;
inline constexpr double kPcSourceOracleTol = NEVPT2_PC_SOURCE_ORACLE_TOL;

[[nodiscard]] constexpr double sourceOracleTol(SourceScheme scheme) {
  return scheme == SourceScheme::Pc ? kPcSourceOracleTol : kSourceOracleTol;
}

// The four spaces, in nevpt2_grad.py's SPACE_NAMES order.
inline constexpr std::array<std::string_view, 4> kSourceSpaceNames{"core", "act_O", "act_V",
                                                                   "virt"};
// The CI half's row name, as the known-gap list spells it.
inline constexpr std::string_view kSourceCiBlock = "P y";

// One entry of THE known-gap list: a block of one committed oracle (its file
// name's stem, `_pc` included) the device is known to get wrong today.
struct KnownGap {
  std::string_view oracle;
  std::string_view block;
};

// THE known-gap list. EMPTY, and kept as an explicit "nothing is known
// wrong": every block of every classified oracle is held to agree. The SC
// oracles' list emptied with #89 (until then it held every Y block an active
// two-electron integral moves under -- N2 CAS(4,4): act_O-virt, act_V-virt;
// LiF: act_O-act_V, act_O-virt, act_V-virt -- for the f3ac/f3ca digests Y held
// fixed; docs/testing.md, "The GPU as the source-term provider" keeps that
// record), and `P y` left it with #87. The PC oracle (#95) started empty, and
// the GPU's PC terms (#74) met every block of it on their first run.
inline constexpr std::array<KnownGap, 0> kKnownGaps{};

// The oracles that have a classification at all -- one with no known gap is
// named here too, so an oracle nobody has classified is refused rather than
// read as "everything must agree".
inline constexpr std::array<std::string_view, 3> kClassifiedOracles{
    "n2_ccpvdz_cas44", "lif_def2svp_cas46", "n2_ccpvdz_cas44_pc"};

// The oracle file's name without the directory and the `.srcgold` suffix --
// what kKnownGaps and kClassifiedOracles key on.
[[nodiscard]] std::string sourceOracleStem(const std::string& path);

// The known gaps of the oracle named `stem`, or a configuration Error if it
// is not in kClassifiedOracles: classify its blocks before gating on
// it.
[[nodiscard]] Result<std::vector<std::string_view>> knownSourceGaps(std::string_view stem);

// A = y - y^T for a [nmo, nmo] `y` in the golden's basis, carried into the
// oracle's: A' = Wf^T A Wf. A `y` of another shape is our bug (check).
[[nodiscard]] Tensor oracleBasisA(const SrcGoldFile& o, const Tensor& y);

// P y = y - <c, y> c, in the golden's basis; `cy` receives <c, y>. Sizes that
// differ are our bug (check).
[[nodiscard]] Tensor projectOffState(const Tensor& ci, const Tensor& y, double& cy);

// One row of the verdict: a block of two spaces, or `P y`.
struct SourceBlockRow {
  std::string name;
  std::int64_t n = 0;  // pairs (determinants, for P y) compared
  double maxRef = 0.0;
  double maxGot = 0.0;
  double maxDelta = 0.0;
  std::int64_t atP = -1, atQ = -1;  // where (for P y: the determinant, in atP)
  bool knownGap = false;
  bool ok = true;
  std::string verdict;
};

struct SourceComparison {
  std::vector<SourceBlockRow> rows;  // the eight blocks in nevpt2_grad's order, then P y
  std::vector<std::string> agreed;     // non-empty blocks held to agree, and that did
  std::vector<std::string> stillGaps;  // non-empty blocks on the list, still off
  std::vector<std::string> failures;   // every block out of place, list typos included
  double tol = 0.0;
  bool ok = false;
};

// `a` (A' in the oracle's basis, [nmo, nmo]: oracleBasisA's) and `py` (P y in
// the golden's basis, [ndet]: projectOffState's) against `o`, block by block,
// each block held to where `gaps` puts it at `tol`. A list entry that names no
// block fails too: it would pin nothing. Shapes that are not the oracle's are
// our bug (check).
[[nodiscard]] SourceComparison compareSourceTerms(const SrcGoldFile& o, const Tensor& a,
                                                  const Tensor& py,
                                                  std::span<const std::string_view> gaps,
                                                  double tol);

}  // namespace nevpt2
