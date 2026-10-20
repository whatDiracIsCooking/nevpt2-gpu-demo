// nevpt2.golden -- readers for the two reference files `generate_golden.py`
// writes (see that script's `write_golden` / `write_gradgold` docstrings for
// the exact binary layouts: a magic, a scalar header, then a sequence of named
// N-d f64 arrays).
// Pure host C++, no GPU -- these only need to run once at program start.
//
//   loadGolden     `*.nevpt2gold`: the CASSCF/CASCI input and the golden
//                  SC/PC-NEVPT2 answer every demo checks itself against.
//   loadGradGold   `*.gradgold`: the analytic-gradient SIDECAR beside a
//                  golden, carrying the one-general-index MO-integral
//                  inventory a gradient reads plus the full-MO-range Fock
//                  matrices and mo_coeff -- or, density-fitted (#164), the
//                  three-index blocks a `*_df` golden lacks.
//   loadSrcGold    `*.srcgold`: PySCF's exact SC source terms for a golden's
//                  state, written by reference_data/sc_source_oracle.py (#92)
//                  -- or, in its PC flavour, block2's PC ones (#95); and
//                  each in a directional flavour (#105): seeded directions
//                  per block and the oracle's derivative along each.
//
// They are two files and not one on purpose: the golden format is extensible
// (magic + array count), but adding arrays would change every committed
// golden's bytes, and a regenerated golden that differs from the committed one
// is this project's signal that something changed
// (docs/reference-data.md, "Regenerating the golden file"). A sidecar keeps
// that intact and keeps the big goldens from growing.
//
// Both files are the user's input, so every way either can be missing or
// malformed is a value-tier Error (nevpt2.error_handling, ErrorKind::IO)
// handed back to the demo's main() to report once -- never an abort.
export module nevpt2.golden;

import std;
// Re-exported: GoldenFile's arrays are Tensors, and loadGolden/require hand
// back a Result/Status the caller reports.
export import nevpt2.tensor;
export import nevpt2.error_handling;

export namespace nevpt2 {

// Which reference produced the orbitals a golden's state sits in -- the
// provenance the scalar header records (docs/reference-data.md, "The file
// format"). Every committed golden's state is a CASCI diagonalisation; what
// this says is which orbitals it ran on -- RHF's, or a converged CASSCF's
// committed in an MO sidecar (docs/reference-data.md, "What 'CASSCF input'
// means here, and why the golden state is CASCI"). A pre-kind "NEVPT2G1" file
// carries no field because, before that sidecar path existed, RHF was the only
// thing it could be.
//
// The two are different references, not a clean and a noisy version of one:
// optimising the orbitals moves E_corr by tens of millihartree. The whole
// gradient derivation is written for the first of them
// (docs/gradient-theory.md), which is why `requireLagrangianReference`
// exists; the source terms it starts from are not, which is why
// `requireSourceTermReference` accepts both.
enum class ReferenceKind {
  Casci = 0,   // a CASCI diagonalisation on converged RHF orbitals
  Casscf = 1,  // orbitals from a converged CASSCF
};

// "CASCI" / "CASSCF" -- the kind as a demo's header line prints it.
[[nodiscard]] std::string_view referenceKindName(ReferenceKind kind);

struct GoldenFile {
  // Stored as int32 (ndet as int64) in the file; held as int64_t like every
  // host size.
  int64_t ncas = 0;
  int64_t nelecA = 0;
  int64_t nelecB = 0;
  int64_t ncore = 0;
  int64_t ndet = 0;
  double eCasci = 0.0;
  double eNevpt2Total = 0.0;
  // The header's reference_kind field, which only a "NEVPT2G2" file carries:
  // the older layout predates it and is a CASCI by definition, so this
  // default is also what a G1 file loads as.
  ReferenceKind reference = ReferenceKind::Casci;
  // The byte offset the array table began at, i.e. how long this file's
  // scalar header turned out to be: read off the cursor where the header
  // parse stopped, never computed from a layout, so it differs between the
  // two magics without anything here knowing by how much. Reported for the
  // layout-contract tests (docs/reference-data.md, "The file format");
  // nothing downstream needs it.
  int64_t arraysAt = 0;
  std::unordered_map<std::string, Tensor> arrays;

  // An IO Error naming the first of `names` the file does not carry. Each
  // demo calls it right after loading, on every array it will get() (the
  // density-fitted demo again for --check-blocks' arrays; --pc's
  // pc_class_energies/e_pc_total are checked separately, in main): a missing
  // one is a file-format mismatch (the user's input), reported before any GPU
  // work.
  [[nodiscard]] Status require(std::initializer_list<std::string_view> names) const;

  // An IO Error unless the header's ndet is `expected`, the determinant count
  // the demo derives from ncas/nelecA/nelecB (it cannot be derived here: the
  // string counts live in link_tables). loadGolden has already checked that
  // ci, when present, holds ndet elements, so passing this ties the CI
  // vector's length to the active space the kernels index it with.
  [[nodiscard]] Status checkNdet(int64_t expected) const;

  // The two reference-kind gates, with `what` naming the flag asking
  // (`--sc-tangent`, say). Host-only and cheap, so a caller runs them before
  // any GPU work. A refusal is Unsupported, NOT an IO Error: a CASSCF-orbital
  // golden is a perfectly well-formed file, and a valid request this code
  // does not handle is the Unsupported tier (docs/architecture.md, "Error
  // handling").
  //
  // requireSourceTermReference: the SOURCE-TERM gate, which accepts either
  // kind. Y_xy = dE/dkappa_xy and y_I = dE/dc_I are derivatives of the
  // NEVPT2 energy at FIXED orbitals and CI vector (Park, JCTC 2019, Eqs.
  // 26-28, 36-38, 41-48): nothing in them knows which reference produced the
  // orbitals, so --sc-tangent, --gradient-source, --dump-source-terms and
  // --source-oracle sit behind this one. It exists so each of those call
  // sites says which claim it rests on, and so a future kind the source
  // terms cannot take has one place to be refused.
  [[nodiscard]] Status requireSourceTermReference(std::string_view what) const;

  // requireLagrangianReference: the LAGRANGIAN gate, which refuses anything
  // but a CASCI. The CI Z-vector -> Sylvester -> CPHF chain downstream of the
  // source terms is the CASCI branch, derived for a CASCI diagonalisation on
  // converged RHF orbitals (docs/gradient-theory.md, Sec. 1.2 and Secs.
  // 2.2-2.6); the CASSCF branch (Sec. 2.7) is not implemented. So whatever
  // consumes a full nuclear gradient -- numerical or analytic -- sits behind
  // this one.
  [[nodiscard]] Status requireLagrangianReference(std::string_view what) const;

  // The array `name`. Aborts (check) if it is absent: the demo should have
  // require()d it, so reaching here without it is our bug, not the file's.
  const Tensor& get(const std::string& name) const;
};

// The parsed file, or an IO Error if it cannot be opened, ends early, has the
// wrong magic ("NEVPT2G2", or the pre-kind "NEVPT2G1"), carries a
// reference_kind this reader does not know, or carries a malformed array
// header (a negative or oversized rank, a negative extent, or an element
// count that overflows or runs past the end of the file), has bytes after its
// last array, or carries a ci whose length is not the header's ndet.
[[nodiscard]] Result<GoldenFile> loadGolden(const std::string& path);

// ---------------------------------------------------------------------------
// The gradient sidecar (`*.gradgold`).
// ---------------------------------------------------------------------------

// A golden's gradient sidecar: the orbital-space partition its array shapes
// are written in terms of, and the arrays themselves.
//
// The scalar header is a partition and not a CI state -- the state is the
// golden's job. Every integral array has `nmo` as its leading extent (the one
// index of each that runs over the full MO range), and the rest of its extents
// are `ncore`, `nact` and `nvirt` in the order its name spells:
// `g_xa_cv`, say, is `(x t | i r)` with shape [nmo, nact, ncore, nvirt].
// docs/gradient-theory.md, §3.3 is the inventory, §3.5 the matrices, and
// `generate_golden.py`'s `_gradient_arrays` writes them.
struct GradGoldFile {
  // Stored as int32 in the file; held as int64_t like every host size.
  int64_t nao = 0;       // AO basis functions: mo_coeff's leading extent
  int64_t nmo = 0;       // = ncore + nact + nvirt, checked on load
  int64_t ncore = 0;
  int64_t nact = 0;
  int64_t nvirt = 0;
  int64_t nelecRhf = 0;  // the RHF electron count: which side of F's active
                         // spectrum is occupied (gradient-theory.md, §2.6)
  // The auxiliary-basis size, which only the DENSITY-FITTED flavour carries
  // ("NEVPT2F1", #164; 0 for a conventional "NEVPT2D1" sidecar). Such a
  // sidecar sits beside a `*_df` golden and holds, instead of §3.3's
  // four-index inventory, the two three-index blocks that golden lacks --
  // `B_cc` [naux, ncore, ncore] and `B_vv` [naux, nvirt, nvirt]
  // (gradient-theory.md, §3.8) -- plus the same full-range matrices of the
  // fitted calculation and the four tied orbital-energy arrays `g_xa_cc`,
  // `g_xc_ca`, `g_xa_vv`, `g_xv_va` under their conventional names and shapes.
  int64_t naux = 0;
  std::unordered_map<std::string, Tensor> arrays;

  // Whether this is the density-fitted flavour -- off the magic, never off
  // the array names.
  [[nodiscard]] bool densityFitted() const { return naux > 0; }

  // An IO Error naming the first of `names` the sidecar does not carry --
  // GoldenFile::require's contract, on the sidecar's arrays.
  [[nodiscard]] Status require(std::initializer_list<std::string_view> names) const;

  // The arrays the general-range inventory is built from -- what the orbital
  // source term, H_kc and H_kk read: docs/gradient-theory.md, §3.3's
  // four-index inventory and the three full-range matrices for a
  // conventional sidecar; §3.8's B_cc / B_vv, the matrices and the four tied
  // orbital-energy arrays for a density-fitted one.
  [[nodiscard]] std::span<const std::string_view> generalInventoryNames() const;
  // require(generalInventoryNames()): an IO Error naming the first one this
  // sidecar lacks. Every reader of the inventory calls it before uploading
  // any of it, so a sidecar missing one array is refused whole -- never a
  // partial H_kk, which every identity pairing the two modes would miss
  // alike (docs/hessian-theory.md, Sec. 9.1).
  [[nodiscard]] Status requireGeneralInventory() const;

  // An IO Error unless this sidecar's orbital-space partition is the one
  // `golden` was written with: the same active space and the same core. A
  // sidecar is a SEPARATE file beside a golden and is found by name, so
  // pairing it with the wrong one is a plain user mistake -- and one whose
  // arrays would otherwise be read with the wrong shapes much deeper in.
  // (The sidecar header carries no determinant count, and the golden carries
  // no n_mo, so these two are everything the two headers share.) And the two
  // must be the same calculation's: a density-fitted sidecar only beside a
  // density-fitted golden (one carrying `B_aa`) whose auxiliary basis is its
  // `naux`, a conventional one only beside a conventional golden -- fitted
  // integrals differentiated against exact ones, or the reverse, would be a
  // gradient of neither energy.
  [[nodiscard]] Status checkAgainst(const GoldenFile& golden) const;

  // The array `name`. Aborts (check) if it is absent: the caller should have
  // require()d it, so reaching here without it is our bug, not the file's.
  const Tensor& get(const std::string& name) const;
};

// The parsed sidecar of either flavour ("NEVPT2D1" conventional, "NEVPT2F1"
// density-fitted), or an IO Error if it cannot be opened, ends early, has
// another magic, carries a space partition that does not add up to nmo or a
// non-even RHF electron count that cannot fill its core (or, density-fitted,
// an naux < 1), carries a malformed array header (as loadGolden refuses one),
// has bytes after its last array, or carries an `mo_coeff` whose shape is not
// [nao, nmo] -- or a `B_cc` / `B_vv` not [naux, ncore, ncore] /
// [naux, nvirt, nvirt].
[[nodiscard]] Result<GradGoldFile> loadGradGold(const std::string& path);

// ---------------------------------------------------------------------------
// The source-term oracle (`*.srcgold`), SC and PC.
// ---------------------------------------------------------------------------

// Which correlation energy an oracle's arrays differentiate -- off its magic,
// the one thing the two flavours' files differ in
// (reference_data/golden_format.py's SRCGOLD_SCHEMES is the same two).
enum class SourceScheme {
  Sc,  // "NEVPT2S1": PySCF's strongly-contracted energy (#92)
  Pc,  // "NEVPT2P1": block2's partially-contracted energy (#95)
};

// "SC" / "PC" -- the scheme as a report prints it.
[[nodiscard]] std::string_view sourceSchemeName(SourceScheme scheme);

// What an oracle's arrays ARE, off its magic too (golden_format.py's
// SRCGOLD_KINDS): every element of A and P y (#92, #95), or seeded
// directions per block and the derivative along each (#105).
enum class SourceFlavour {
  Full,         // "NEVPT2S1" / "NEVPT2P1"
  Directional,  // "NEVPT2SD" / "NEVPT2PD"
};

// "full" / "directional".
[[nodiscard]] std::string_view sourceFlavourName(SourceFlavour flavour);

// The exact SC-NEVPT2 (or, `Pc`, PC-NEVPT2) source terms for one golden's
// state, written by reference_data/sc_source_oracle.py (#92; `--pc`, #95):
// what nevpt2_sc_pseudodensity --source-oracle holds the GPU's Y_xy and y_I
// to (docs/testing.md, "The SC source terms against PySCF's, on a card"), and
// what #74's PC ones will be held to (docs/testing.md, "The PC source terms'
// oracle"). One layout for both flavours. The script's docstring is the
// reference for what each array means and which basis it is in:
//
//   a_corr   [nmo, nmo]    A_pq = dE_corr/dkappa_pq - (p <-> q) on every
//                          non-gauge pair, in the basis whose active block is
//                          the golden's rotated by act_rot (act_O = its first
//                          nActOcc active orbitals); zero on the gauge pairs
//   act_rot  [nact, nact]  that rotation, W: Y' = W^T Y W on the active block
//   py       [ndet]        P y in the GOLDEN's determinant basis
//   ci_mask  [ndet]        1.0 on the determinants py is defined on, else 0.0
//   ci       [ndet]        the golden's CI vector, as a fingerprint
//
// The DIRECTIONAL flavour (#105) keeps act_rot and ci and replaces the rest:
//
//   kappa_dirs   [nk, nmo, nmo]  seeded unit antisymmetric kappas, each over
//                                one block's pairs, in a_corr's basis
//   kappa_block  [nk]            which block (0..7, source_compare's row order)
//   kappa_fd     [nk, L]         the central differences along each, at
//                                h = kappaStep * 2^-i: the value's ladder
//   kappa_val    [nk]            <kappa, A> = sum_{p<q} kappa_pq A_pq, the
//                                Richardson value of the first two rungs
//   ci_dirs      [nc, ndet]      seeded unit CI directions, orthogonal to ci,
//                                in the GOLDEN's determinant basis
//   ci_fd        [nc, Lc]        their ladders, from ciStep
//   ci_val       [nc]            <u, P y>, likewise
struct SrcGoldFile {
  SourceScheme scheme = SourceScheme::Sc;
  SourceFlavour flavour = SourceFlavour::Full;
  // Stored as int32 (ndet as int64) in the file; held as int64_t.
  int64_t nmo = 0;
  int64_t ncore = 0;
  int64_t nact = 0;
  int64_t nvirt = 0;
  int64_t nActOcc = 0;  // how many of the active orbitals are act_O
  int64_t ndet = 0;
  // The steps PySCF's quotients were taken at; in the directional flavour,
  // the first (coarsest) rung of each ladder.
  double kappaStep = 0.0;
  double ciStep = 0.0;
  std::unordered_map<std::string, Tensor> arrays;

  // An IO Error unless this oracle was written for `golden`'s state: the
  // same core and active space, the same determinant count, and the golden's
  // own CI vector, bit for bit (the generator copies it). Found by name
  // beside a golden, so a mismatch is a plain user mistake.
  [[nodiscard]] Status checkAgainstGolden(const GoldenFile& golden) const;

  // checkAgainstGolden, and `sidecar`'s partition: the same n_mo.
  [[nodiscard]] Status checkAgainst(const GoldenFile& golden, const GradGoldFile& sidecar) const;

  // The array `name`; loadSrcGold has required every array of the file's
  // flavour, so asking for a missing one is our bug (check).
  const Tensor& get(const std::string& name) const;
};

// The parsed oracle, or an IO Error if it cannot be opened, ends early, has
// the wrong magic (none of "NEVPT2S1", "NEVPT2P1", "NEVPT2SD", "NEVPT2PD"),
// carries a partition that does not add up or an act_O count outside the
// active space, a malformed array header, bytes after its last array, or does
// not carry its flavour's arrays above at their shapes. A directional one is
// also refused if a direction is not exactly antisymmetric, a block index is
// not one of the eight, a ladder has fewer than two rungs, or a CI direction
// is not orthogonal to ci (1e-10) -- the value would not mean what it says.
[[nodiscard]] Result<SrcGoldFile> loadSrcGold(const std::string& path);

// The case an oracle at `path` belongs to: its file name without the
// directory and the `.srcgold` suffix, less the `_dir` a directional
// oracle's name ends in and then the `_pc` a PC one's does --
// golden/<case>[_pc][_dir].srcgold all sit beside the same <case>.nevpt2gold
// and .gradgold (golden_format.srcgold_case is the Python side). A name
// lacking its kind's suffix is an IO Error: it would claim another oracle's
// place.
[[nodiscard]] Result<std::string> srcGoldCase(const std::string& path, SourceScheme scheme,
                                              SourceFlavour flavour = SourceFlavour::Full);

}  // namespace nevpt2
