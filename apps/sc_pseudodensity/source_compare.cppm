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
// A DIRECTIONAL oracle (#105, `*_dir.srcgold`) holds seeded directions per
// block and the derivative along each instead of every element; its rows are
// the same nine, compared by projection (compareSourceDirections, below).
//
// THE TOLERANCES ARE NOT 1e-7: the oracle's side is a central difference,
// and its truncation at the committed steps is what the agreement can be read
// to. Each is measured from its own step ladders, and each has ONE home,
// read at configure time (apps/sc_pseudodensity/CMakeLists.txt): SC's
// docs/testing.md, "The SC source terms against PySCF's, on a card"; PC's
// docs/testing.md, "The PC source terms' oracle"; the directional flavour's
// two, docs/testing.md, "The directional source-term oracle".
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
#ifndef NEVPT2_SOURCE_DIRECTIONAL_TOL
#error "NEVPT2_SOURCE_DIRECTIONAL_TOL must be defined by apps/sc_pseudodensity/CMakeLists.txt"
#endif
#ifndef NEVPT2_PC_SOURCE_DIRECTIONAL_TOL
#error "NEVPT2_PC_SOURCE_DIRECTIONAL_TOL must be defined by apps/sc_pseudodensity/CMakeLists.txt"
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
// the GPU's PC terms (#74) met every block of it on their first run. The
// directional oracles (#105) started empty too: acrolein's SC and PC terms,
// which no full oracle fits, met every block of theirs on their first run,
// and N2's and LiF's directional twins agree with their full oracles on
// which blocks pass (docs/testing.md, "The directional source-term oracle").
// The four N2 CAS(4,4) oracles on CASSCF orbitals (#117,
// n2_ccpvdz_cas44_casscf[_pc][_dir]) started empty too: the GPU's SC and PC
// terms met every block their partition has (act_V's three are empty there
// by construction) on the first run, before they were classified.
inline constexpr std::array<KnownGap, 0> kKnownGaps{};

// The oracles that have a classification at all -- one with no known gap is
// named here too, so an oracle nobody has classified is refused rather than
// read as "everything must agree".
inline constexpr std::array<std::string_view, 12> kClassifiedOracles{
    "n2_ccpvdz_cas44",        "lif_def2svp_cas46",          "n2_ccpvdz_cas44_pc",
    "n2_ccpvdz_cas44_dir",    "lif_def2svp_cas46_dir",      "n2_ccpvdz_cas44_pc_dir",
    "acrolein_def2svp_cas65_dir", "acrolein_def2svp_cas65_pc_dir",
    "n2_ccpvdz_cas44_casscf",     "n2_ccpvdz_cas44_casscf_pc",
    "n2_ccpvdz_cas44_casscf_dir", "n2_ccpvdz_cas44_casscf_pc_dir"};

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
  // Read off a directional oracle (#105): a row's n counts directions, and
  // atP is the direction where its worst difference sits.
  bool directional = false;
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

// --- the directional flavour (#105) ------------------------------------------
//
// The same verdict off a DIRECTIONAL oracle: the GPU's A' (oracleBasisA's) and
// P y (projectOffState's) are projected onto the stored directions --
// <kappa, A'> = sum_{p<q} kappa_pq A'_pq, <u, P y> -- and each block's row is
// the worst |GPU - oracle| over its directions. A and P y enter the gradient
// linearly, so a wrong or missing term anywhere in a block moves a seeded
// random direction's projection with probability 1. The known-gap semantics
// are compareSourceTerms', and the tolerance is the directional one
// (sourceDirectionalOracleTol). A block this partition has pairs in, probed
// by fewer than kMinBlockDirections directions, fails: an oracle that skips a
// block would otherwise pass it unexamined.

// Per projection, absolute (Eh): the directional oracle's tolerance, one per
// scheme, measured from its ladders (docs/testing.md, "The directional
// source-term oracle" -- read at configure time like the two above).
inline constexpr double kSourceDirectionalTol = NEVPT2_SOURCE_DIRECTIONAL_TOL;
inline constexpr double kPcSourceDirectionalTol = NEVPT2_PC_SOURCE_DIRECTIONAL_TOL;

[[nodiscard]] constexpr double sourceDirectionalTol(SourceScheme scheme) {
  return scheme == SourceScheme::Pc ? kPcSourceDirectionalTol : kSourceDirectionalTol;
}

// The tolerance an oracle of this kind is read at.
[[nodiscard]] constexpr double sourceOracleTol(const SrcGoldFile& o) {
  return o.flavour == SourceFlavour::Directional ? sourceDirectionalTol(o.scheme)
                                                 : sourceOracleTol(o.scheme);
}

// #105: "at least 2 directions per block".
inline constexpr std::int64_t kMinBlockDirections = 2;

// An IO Error unless every stored kappa direction lies inside the block its
// kappa_block names -- no nonzero element outside it, and none on a gauge
// pair -- so a row reports the block it claims to. A full oracle passes
// trivially.
[[nodiscard]] Status checkDirectionBlocks(const SrcGoldFile& o);

// `a` and `py` (as compareSourceTerms takes them) against a directional
// oracle `o`. Shapes that are not the oracle's are our bug (check).
[[nodiscard]] SourceComparison compareSourceDirections(const SrcGoldFile& o, const Tensor& a,
                                                       const Tensor& py,
                                                       std::span<const std::string_view> gaps,
                                                       double tol);

// compareSourceTerms or compareSourceDirections, by the oracle's flavour.
[[nodiscard]] SourceComparison compareSourceOracle(const SrcGoldFile& o, const Tensor& a,
                                                   const Tensor& py,
                                                   std::span<const std::string_view> gaps,
                                                   double tol);

}  // namespace nevpt2
