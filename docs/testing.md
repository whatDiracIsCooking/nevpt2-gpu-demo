# Testing

How this tree is checked: the two test tiers, why there is no CI, the local
gate, the checks on the oracle itself (its pytest hand cases and the golden
regeneration gate), the sanitizer tier — what each tool catches, what it
cannot, and how the tier proves its own tools are on — and host code
coverage. Every figure names its card.

## Two tiers

- **The golden tier** is the numerical one. Each demo checks itself against
  a committed golden file (every class energy and the total to 1e-7) and
  prints `PASS:`; the ctest entry matches that line. The gradient work is
  checked the same way: `nevpt2_demo --sc-tangent` for the SC tangent ("The
  SC tangent's finite-difference cross-check", below), `--gradient-source`
  for the CI source term ("The CI source term's forward-reverse duality") and
  two non-demo
  binaries, `nevpt2_sc_pseudodensity` for the SC pseudodensities' exact
  identities ("The SC pseudodensity identities") and its `--kappa-act` for
  the orbital source term ("The active-rotation source term"), and
  `nevpt2_pc_tangent` for
  the PC tangent ("The PC tangent against a finite difference"). What each of
  them computes, and why the permanent ones are exact identities rather than
  finite differences, is [`gradients.md`](gradients.md).
  Every golden entry is labelled `gpu`, because there is no CPU path.
  `ctest --preset fast` runs every entry not labelled `slow`.
- **The unit tier** (`test/`, GoogleTest) checks host logic one component at
  a time — `test/<x>/` tests `nevpt2.<x>` — and is labelled `unit`
  (`ctest --preset unit`). A host-only suite has no `gpu` label and runs with
  no card visible; a suite that needs a card is `REQUIRES_GPU`. It proves
  logic (error kinds, propagation, index helpers), **never an energy**. Each
  binary also registers a `<target>.SuiteListIsComplete` drift guard
  (`cmake/README.md`, "Tests"). It is added from the root `CMakeLists.txt`
  only, so the standalone `cmake -S src` build has no gtest.

## No CI, and the local gate

There is no `.github/workflows/`, on purpose. A hosted runner has no GPU, so
it could prove only that the tree compiles and that the host helpers behave,
while every claim this project makes is numerical. A green badge that cannot
fail on a wrong answer invites the belief that something was checked.

**If you are sending a change, please don't add a workflow.** Run the gate
below on whatever card you have, and say in the pull request what you ran,
on which GPU and backend, and what it printed. A change that touches a
kernel or an energy needs a `PASS:` line from a real card. A change that
only compiles for the other backend says so.

So the gate is local, and **saying what you ran is the whole protocol**:

```bash
devtools/cpp-tier.sh                # CUDA, on a box with a card (runs the five gates below first)
devtools/cross-backend-check.sh     # does the OTHER backend still compile (the two lints too)
devtools/stream-lint.sh --strict    # every GPU call on the one stream; both scripts above run it
devtools/throw-lint.sh              # no `throw` in src/, apps/ or test/; likewise
uv run python reference_data/golden_layout_check.py   # "The golden layout contract", below; cpp-tier.sh runs it
uv run ruff check                   # the Python lint (pyproject.toml's [tool.ruff]); likewise
uv run pytest                       # "The oracle's own tests", below; likewise
devtools/golden-repro.sh            # "The golden regeneration gate", below; MANUAL, minutes of CPU
```

`cpp-tier.sh` runs the five as named phases (`stream-lint`, `throw-lint`,
`golden-layout`, `ruff`, `pytest`) before it configures, and under
`--no-test` too: none needs a card, and a `--no-test` run is often the only
run a no-card change gets. A failing one fails the tier and is named in its
summary line (`FAIL(rc=1) failed=golden-layout`). The three Python phases
run through `uv run --frozen --no-sync` after `uv sync --frozen --check`
(`in_locked_env`, `devtools/lib.sh`), so a missing or stale environment
(`/opt/venv` in every image, `.venv` on a host) **fails** the phase with
`uv sync --frozen` as the fix, rather than skipping it: #84's stray reader
reached `main` because the gate that would have caught it was one nobody
ran. An image built before pytest entered the lock is such a stale
environment — `uv sync --frozen` in the container, or a rebuild, clears it.
`cross-backend-check.sh` runs only the two lints: they read the one C++ tree
it compiles for the other backend, while the Python phases are
backend-independent and already ride every `cpp-tier.sh` run.
`golden-repro.sh` is not a phase at all ("The golden regeneration gate",
below, says why).

`cross-backend-check.sh` proves the other backend *compiles*, not that it is
right: a numerical claim needs that backend's card. The `compile-cuda` /
`compile-hip` presets are compile-and-link-only configurations (testing
off).

## The golden layout contract

Every reader of the three binary formats in `golden/` must read **both**
golden header layouts — `NEVPT2G1` (the committed CASCI goldens) and
`NEVPT2G2` (the `*_casscf*` ones) — because one header change once broke three
readers, the last caught by luck. The readers, the one parser per language
behind them, and how a header change is made are in
[`reference-data.md`](reference-data.md), "The file format". The contract is
checked in two halves:

- **C++**: `test/golden`'s `GoldenLayoutContractTests` (unit tier,
  host-only, no card) loads every `golden/*.nevpt2gold` and
  `golden/*.gradgold` present at configure time with `loadGolden` /
  `loadGradGold`, requires both golden layouts and both sidecar flavours
  (`NEVPT2D1`, density-fitted `NEVPT2F1`) to be among them, and requires every
  file of one layout to have its array table at the same derived offset
  (`GoldenFile::arraysAt`) and the two layouts' to differ.
- **Python**: `uv run python reference_data/golden_layout_check.py` — every
  committed golden, gradient sidecar and MO sidecar through every reader entry
  point in `reference_data/` and back through its writer, byte for byte; a
  golden from the current writer read back; and no parser outside
  `golden_format.py`. Since #138 it also checks the CASSCF sweep's hand-off,
  which is never committed. A `states.json` from
  `nevpt2_numgrad._write_manifest` must read back through
  `read_states_manifest`, and a transient golden with NaN PC placeholders
  (`pc_on_states.py`'s) must pass every golden reader. A committed golden
  holding any NaN or infinity is refused, so such a transient file cannot
  land in `golden/`. Ends `PASS:` or `FAIL:` with an exit status.

The Python half registers **no ctest entry**, for the same reasons the
gradient comparison does not ("Why there is no ctest entry for it", below):
it imports the offline scripts it checks, so an entry would put the PySCF
environment in the test path, and it touches no card or binary while the
golden tier is all `gpu`. Unlike that comparison it is cheap, so
`devtools/cpp-tier.sh` runs it as its `golden-layout` phase ("No CI, and the
local gate", above) on every preset, `--no-test` included. Its scan covers
`src/` and `apps/` too: #73's `.gradsrc` reader went in when no gate ran it,
and turned it red on `main`. A C++ file that reads a format other than the three golden layouts is
named, with its reason, in the check's `CPP_NON_GOLDEN_READERS`; it may open a
file but still fails the scan on a seek or a golden magic.

Measured (no card; the 15 committed goldens — 12 `G1`, 3 `G2` — 3 gradient
sidecars and 3 MO sidecars): 1.1 s wall, `PASS:`. A scratch field added to
the `G2` header table fails it on all three `G2` goldens (their array tables
misread: a non-ASCII array name) and on the "both layouts present" check; a
field added to `write_golden` alone fails the three `G2` byte round trips and
both freshly written goldens; a hand-rolled `f.seek(52)` added to a script
fails the stray-parser scan.

## The oracle's own tests

Every claim here is checked against `golden/`, and `golden/` is produced and
guarded by `reference_data/` — so the oracle's own logic gets tests:
`reference_data/tests/`, run by `uv run pytest` (pytest is in the `dev`
group; `pyproject.toml`'s `[tool.pytest.ini_options]` points it there). This
reverses the earlier "no Python test suite" decision, for one reason: the
format readers and the refusal guards are the one Python code whose silent
regression would corrupt every later check. A guard loosened by accident
still writes a golden, and every entry downstream then passes against it.

What they are, all on synthetic inputs, no PySCF solve, no card, no network:

- **`test_golden_format.py`** — every magic (`NEVPT2G2`, the `G1` golden,
  `NEVPT2D1` and the density-fitted `NEVPT2F1` with its trailing `naux`,
  #164, `NEVPT2M1`, `NEVPT2S1`/`NEVPT2P1` and the directional
  `NEVPT2SD`/`NEVPT2PD`) written by its writer
  (`write_golden`, `write_gradgold`, `write_mo_sidecar`, `write_srcgold`, or
  `pack_header` + `write_arrays` for `G1`, which no writer produces any more)
  and read back field for field and array for array, then rewritten byte for
  byte; `G1` and `G2` differing only by the magic and `reference_kind`
  (what `golden-repro.sh`'s header exception relies on). Refusals: the file
  cut at **every** byte, trailing bytes, every reader handed every other
  format, a garbage magic, rank `MAX_RANK` accepted and `MAX_RANK + 1`
  refused, negative extents and counts, a name length past the end, a
  non-ASCII name, an unknown `reference_kind`, `ci` against `ndet`, `want`
  skipping an array **and still validating it**, and `generate_golden`'s
  `--mo-from` reader's refusals. One quirk found and pinned rather than
  changed: the writer cannot write a rank-0 array (`np.ascontiguousarray`
  makes it shape `(1,)`, which is why `e_pc_total` is a one-element vector);
  the reader accepts one.
- **`test_refusal_guards.py`** — `_check_degeneracy` on a real N2/STO-3G
  molecule (`gto.M` builds the symmetry orbitals; nothing is solved) with an
  orthonormal symmetry-pure orbital set, so PySCF's own `label_orb_symm`
  labels the irreps: same-irrep core and virtual pairs at 0.99 and 1.01 of
  1e-4 Eh, a cross-irrep exact degeneracy and an active one accepted.
  `_check_pc_conditioning` at 0.99 / 1.01 of 1e3 and naming the worst block,
  and `_pc_cut_gap` on diagonal stacks whose singular values are exact.
  `_check_gradient_invariants`: (O1) on each of its three off-blocks at
  0.99 / 1.01 of 1e-5, (O2) on an off-diagonal and both diagonals at 0.9 /
  1.1 of 1e-10, (O3) with an eigenvalue at the HOMO, just inside the gap,
  and act_O one too many or too few.
- **`test_source_oracle.py`** (#117) — `sc_source_oracle.py`'s CASSCF
  reference. A `*_casscf` case needs `--mo-from` and any other case refuses
  it. A mismatched `.mogold` is refused with every mismatch listed: LiF's
  sidecar for the N2 case gives nine fields, and N2's DF sidecar gives the
  one `auxbasis` line. Each committed `n2_ccpvdz_cas44_casscf*.srcgold` must
  have one active range, an identity `act_rot`, and no direction in an act_V
  block. The one exception to "no solve": each refusal case runs N2's RHF
  first, because the fingerprint is checked after the RHF and before the
  CASCI.
- **`test_pc_on_states.py`** (#138) — the plumbing from a CASSCF sweep's
  kept states to the GPU's PC energies. `golden_format.read_states_manifest`
  splits kept states from tripped or refused ones. It refuses a doubled
  label, a missing `.mogold`, energy or `--atom` spec on a kept state, a
  missing header field, bad JSON and a non-`casscf` manifest. It reads back
  what `nevpt2_numgrad._write_manifest` writes. `pc_on_states.parse_pc`
  reads `nevpt2_demo --pc`'s table and full-precision line, including a
  device-refused class, and refuses a missing class. `pc_guard` feeds the
  table's gaps to `_check_pc_conditioning` at 0.99 / 1.01 of 1e3. The
  transient golden's NaN PC placeholders round-trip through
  `read_golden_pc_total`, and a `--work-dir` inside `golden/` is refused.
- **`test_numgrad_df.py`** (#167) — `nevpt2_numgrad.py --df`. On the fitted
  route, a named out-of-plane component of acrolein (`0z`, Cs → C1) is
  refused with the group it breaks to, before any SCF: a stand-in for
  `energy` fails the test if one runs. `--components auto` keeps the eight
  out-of-plane components as symmetry zeros. Also refused: a golden of the
  other flavour, a golden fitted in another auxiliary basis (`cc-pvdz-ri`'s
  `naux` against the golden's 140), and a sidecar of the other flavour.
  Writing into the committed CAS(8,8) `_df` sidecar, which holds no
  `grad_num`, is refused before the sweep, and so are the `--df --pc`,
  `--df --reference casscf` and bare `--auxbasis` flags. The committed
  CAS(4,4) `_df` sidecar carries its four `grad_num*` arrays after the
  twelve, read through `golden_format` only.
- **`test_sc_empty_perturber_probe.py`** — `sc_empty_perturber_probe.py`'s
  algebra, on a synthetic tuple family (random positive semi-definite metric,
  random Dyall block, first-order vectors). The four-point mixed difference
  of `F` equals the polarisation at steps 1, 1e-2 and 1e-4. With `K = eta S`,
  `rho` is zero, K6 equals the polarisation and K4 at the ray-limit
  multipliers, and the parallelogram defect vanishes. With `H = 0` (Sijrs),
  K6 is `-N^{uv}/Delta`. Otherwise `rho` and the defect are non-zero and
  `rho` survives rescaling either direction. Parallel and one-sided pairs
  give `rho = 0` whatever `K` is. `dotted` is checked as `rotate`'s
  derivative. One case reads `golden/n2_ccpvdz_cas44`: the transcription
  reproduces its class energies, and every guarded tuple is empty by its
  integrals ([`hessian-theory.md`](hessian-theory.md), §5.6; "The
  empty-perturber term, offline" below).

**The thresholds are written into the tests as numbers**, not read from the
module: a test comparing against `DEGENERACY_TOL` would move with it.
Moving a threshold therefore means editing the test in the same change.
What the hand cases cannot reach without a solve: whether the generator
hands the guards the right inputs (`_gradient_arrays`, `_block2_icmr`'s gap
recording), `_check_casscf_stationary` (it evaluates a CASSCF orbital
gradient), and `_block2_pc`'s block2-SC-equals-PySCF-SC gate (it runs both).
Those stay with "The golden regeneration gate" below.

It is the `pytest` phase of `cpp-tier.sh` ("No CI, and the local gate",
above), after `ruff`. Measured: 332 tests, 31 s, no card (RTX 3080 box's
CPU, 2026-10-20; the ten empty-perturber cases take 1 s of it). Negative controls, run once in scratch and reverted: setting
`DEGENERACY_TOL` to 1e-5 fails 4 tests (the pinned value and the three
0.99e-4 refusals); leaving the constant alone and loosening the comparison
inside `_check_degeneracy` to `DEGENERACY_TOL / 10` fails the same three
refusals. With `UV_PROJECT_ENVIRONMENT` pointed at a venv built from the
lock before pytest was added — what a container from an older image has —
and again at a directory that does not exist, `cpp-tier.sh --no-test` fails
all three Python phases at `uv sync --frozen --check` ("the locked Python
environment is missing or stale … run `uv sync --frozen`"), summary
`FAIL(rc=1) failed=golden-layout,ruff,pytest`; pytest is never run, so
nothing skips.

## The golden regeneration gate

`devtools/golden-repro.sh` re-runs the promise CLAUDE.md makes — the
generators are bit-reproducible, so a regenerated file that differs from the
committed one means something changed — for every committed N2 CAS(4,4) /
CAS(8,8) reference. Twenty files from sixteen generator runs, each with the
generator's own settings (`--out` into a scratch directory is the only flag
added):

| files | run |
|---|---|
| `n2_ccpvdz_cas44.nevpt2gold`, `.gradgold`; `n2_ccpvdz_cas88.nevpt2gold`, `.gradgold` | `generate_golden.py --gradgold` |
| `n2_ccpvdz_cas44_df.nevpt2gold`, `.gradgold`; `n2_ccpvdz_cas88_df.nevpt2gold`, `.gradgold` | `generate_golden.py --df --gradgold` (the density-fitted `NEVPT2F1` sidecar, #164); for CAS(4,4), then `nevpt2_numgrad.py --df` on the sidecar it just wrote, in the same run, so its `grad_num*` arrays are regenerated too (#167) |
| `n2_ccpvdz_cas44_casscf[_df].mogold` | `generate_golden.py --casscf --mo-out` (a fresh optimisation) |
| `n2_ccpvdz_cas44_casscf[_df].nevpt2gold` | `generate_golden.py --mo-from` the **committed** sidecar, so a drifted sidecar fails as itself and not as its golden too |
| `n2_ccpvdz_cas44.srcgold`, `n2_ccpvdz_cas44_pc.srcgold` | `sc_source_oracle.py [--pc]`, reading the committed golden |
| `n2_ccpvdz_cas44_dir.srcgold`, `n2_ccpvdz_cas44_pc_dir.srcgold` | `sc_source_oracle.py --directional [--pc]` (#105) |
| `n2_ccpvdz_cas44_casscf[_pc][_dir].srcgold` | `sc_source_oracle.py --case n2_ccpvdz_cas44_casscf --mo-from` the **committed** sidecar `[--pc] [--directional]` (#117) |

The comparison is `cmp` on the whole file — the one place in this tree a
byte comparison is right. One exception, the documented one
([`reference-data.md`](reference-data.md), "Regenerating the golden file"):
a committed `NEVPT2G1` CASCI golden regenerates as `NEVPT2G2`, so for those
four the state fields must be byte-identical, the new `reference_kind` must
be `casci`, and everything from the array count on byte-identical, each
file's header length read with `golden_format.py arrays-at`. It prints one
`PASS:`/`FAIL:` line per file (naming it) and exits 0 only if all
twenty reproduce. `--committed DIR` compares against a copy instead of `golden/`
(the generator inputs stay `golden/`'s), `--only REGEX` narrows the set,
`--out DIR` keeps the regenerated files and each run's log.

**The eight N2 `.srcgold` files are in it** because the 30-minute budget the
gradient milestone sets for any oracle regeneration holds with them: the SC
ones are seconds, the PC directional one about a minute and the PC full one
about two and a half minutes, below. Acrolein's and LiF's directional
oracles are outside its N2 scope; their byte-reproducibility is
[`reference-data.md`](reference-data.md), "The directional flavour". Each generator
is single-threaded by its own import (`generate_golden` forces it, and
`sc_source_oracle` imports `generate_golden` for exactly that), so `-j N`
runs N of them at once without changing a byte; the default is `-j 1`.

Measured (RTX 3080 box's CPU, 24 cores; no card used), all twelve `PASS:`
before #105 added the two directional files:

| run | wall |
|---|---:|
| default, `-j 1` (one generator at a time, each single-threaded) | 330 s (5 min 30 s) |
| `-j 10` | 144 s |
| `-j 12`, fourteen files, #105 (the two-card box's CPU, 24 cores, both cards running other jobs) | 152 s, all fourteen `PASS:` |
| `-j 2`, eighteen files, #117 (RTX 3080 box's CPU, measured with a concurrent sweep on the CPU) | 307 s, all eighteen `PASS:` |
| `-j 6`, twenty files, #164 (RTX 3080 box's CPU, a sibling build sharing it) | 179 s, all twenty `PASS:` |
| `-j 6`, twenty files, `n2_ccpvdz_cas44_df.gradgold` with its `grad_num`, #167 (same box, a sibling agent sharing it) | 178 s, all twenty `PASS:`; the `cas44_df` run, generator plus `nevpt2_numgrad.py --df`, 33 s |

Per generator run (the `-j 1` run): `--casscf` sidecars 2–3 s,
`n2_ccpvdz_cas44.srcgold` 9 s, the CAS(4,4) goldens 26–27 s, the CAS(8,8)
ones 32–33 s, `n2_ccpvdz_cas44_pc.srcgold` 142 s — so `-j 10`'s wall is the
PC oracle's alone. In the `-j 12` run the directional pair took 6 s (SC) and
56 s (PC), and the full PC oracle 150 s, again the whole wall. In the `-j 2`
run the four CASSCF-orbital oracles took 9 s (SC), 145 s (PC), 3 s (SC
directional) and 44 s (PC directional). **Nothing is left out** of the N2 CAS(4,4)/CAS(8,8) set
for cost: the whole set was 330 s at `-j 1` against the 30-minute budget. The larger
committed references are outside its scope, not left out for time; theirs,
where measured, are in [`reference-data.md`](reference-data.md),
"Regenerating the golden file", "The SC source-term oracle" and "The files
in `golden/` and `reference_data/`".

**A manual gate, not a `cpp-tier.sh` phase.** Even at `-j 10` it is minutes
of CPU, two orders of magnitude over the other five gates together, and it
can only fail when the generators, the lock or `golden/` change — so it is
run like the sanitizer tier and `cross-backend-check.sh`: by whoever changes
`reference_data/`, `pyproject.toml`/`uv.lock` or a small golden, and named
in the PR. Negative controls, `--committed` at a scratch copy of `golden/`
with one byte flipped: byte 300000 (array section) of
`n2_ccpvdz_cas44_df.nevpt2gold`, run with `--only` over the two CAS(4,4)
CASCI goldens, prints `PASS:` for the untouched `n2_ccpvdz_cas44` and
`FAIL: n2_ccpvdz_cas44_df.nevpt2gold -- G1->G2: the arrays differ (… differ:
char 299953 …)` — the offset past the 48-byte `G1` header — exit 1; byte
1000 of `n2_ccpvdz_cas44.srcgold`, the whole-file `cmp` path, prints
`FAIL: n2_ccpvdz_cas44.srcgold -- differs: … char 1001`, exit 1.

## The CASSCF-orbital entries

Seven golden entries read a reference whose orbitals came from a converged
`mcscf.CASSCF` rather than from RHF ([`reference-data.md`](reference-data.md),
"The MO sidecar: committed CASSCF orbitals"). They are ordinary golden checks
— the unmodified binaries, the unmodified flags, every class and the total to
1e-7 — on a file whose header says `reference_kind = CASSCF`, which both demos
print in their first line:

| entry | binary | flags | labels |
|---|---|---|---|
| `nevpt2_cas44_casscf` | `nevpt2_demo` | `--tiles 3` | `gpu;small` |
| `nevpt2_cas44_casscf_pc` | `nevpt2_demo` | `--pc --tiles 3` | `gpu;small` |
| `nevpt2_df_cas44_casscf` | `nevpt2_df_demo` | `--tiles 3 --check-blocks` | `gpu;small` |
| `nevpt2_df_cas44_casscf_pc` | `nevpt2_df_demo` | `--pc --tiles 3` | `gpu;small` |
| `nevpt2_cas1010_casscf` | `nevpt2_demo` | `--tiles 3` | `gpu` |
| `nevpt2_cas1010_casscf_pc` | `nevpt2_demo` | `--pc --tiles 3` | `gpu` |
| `nevpt2_lif_cas46_casscf` | `nevpt2_demo` | `--tiles 3` | `gpu` |

The CAS(4,4) four are `small`, so the sanitizer tier runs a `NEVPT2G2` file
too; CAS(10,10) is the production size at the shipped `--tiles 3`. The three
N2 goldens carry PC fields. LiF's (#123's golden, registered by #125 beside
its CASCI twin `nevpt2_lif_cas46`) is SC only, like that twin, so it has no
`_pc` entry.

**Two gates, and which paths each one admits.** `nevpt2.golden` has two
reference-kind gates. The **source-term** gate
(`GoldenFile::requireSourceTermReference`) accepts a CASSCF golden:
`Y_xy = dE/dkappa_xy` and `y_I = dE/dc_I` are derivatives of the energy at
fixed orbitals and CI vector, and carry no reference-kind dependence (Park
Eqs. 26–28, 36–38, 41–48). `nevpt2_demo --sc-tangent` / `--gradient-source`
and `nevpt2_sc_pseudodensity --dump-source-terms` / `--source-oracle` sit
behind it, and so does every Hessian flag (#157, below). The **Lagrangian** gate (`requireLagrangianReference`) refuses one
as `err_unsupported`, because the CI Z-vector → Sylvester → CPHF chain is
derived for CASCI on RHF orbitals ([`gradient-theory.md`](gradient-theory.md),
§1.2 and §2.2–§2.6), and the CASSCF branch (§2.7) is not implemented yet.
`reference_data/nevpt2_numgrad.py`'s default `--reference casci` route refuses
one as well; its `--reference casscf` route re-optimises every displaced state
([`reference-data.md`](reference-data.md), "The numerical gradient of a
CASSCF reference").
The split is checked in the unit tier by `test/golden`'s
`GoldenReferenceKindGateTests` and `GoldenCommittedCasscfTests`. The second
of those reads `golden/n2_ccpvdz_cas44_casscf.nevpt2gold` itself, so the split
is held against a committed reference as well as a spliced one. Host-only,
no card.

### The Hessian on CASSCF orbitals

Level 1 of the Hessian differentiates the energy functional, which does not
know how its orbitals were made, so all seven Hessian flags sit behind the
source-term gate ([`hessian-theory.md`](hessian-theory.md), §9.1). #157
registers each on `n2_ccpvdz_cas44_casscf`, the same binary and flags as its
CASCI-orbital twin, all `gpu;small`:

| entry | binary | flag | the mode its regex holds |
|---|---|---|---|
| `nevpt2_cas44_casscf_sc_hessian` | `nevpt2_demo` | `--sc-hessian` | orbital energies held fixed |
| `nevpt2_cas44_casscf_sc_hessian_vector` | `nevpt2_demo` | `--sc-hessian-vector` | held fixed on both legs |
| `nevpt2_sc_pseudodensity_cas44_casscf_sc_hessian_kc` | `nevpt2_sc_pseudodensity` | `--sc-hessian-kc` | FULL (Euler skipped, and said so) and EULER (Euler gated), off the committed `.gradgold` |
| `nevpt2_sc_pseudodensity_cas44_casscf_sc_hessian_kk_int` | `nevpt2_sc_pseudodensity` | `--sc-hessian-kk-int` | — (the twin's regex) |
| `nevpt2_sc_pseudodensity_cas44_casscf_sc_hessian_kk` | `nevpt2_sc_pseudodensity` | `--sc-hessian-kk` | — (the twin's regex, 2 pairs refused for `rho_t`) |
| `nevpt2_pc_hessian_cas44_casscf` | `nevpt2_pc_tangent` | `--pc-hessian` | held fixed, with the smallest kept `s / s_max` and gap printed |
| `nevpt2_pc_hessian_vector_cas44_casscf` | `nevpt2_pc_tangent` | `--pc-hessian-vector` | likewise |

Measured on the RTX 3080 (CUDA, `--tiles 3`, the default digest), #157: all
seven pass, each in under 3 s. Both PC entries print the smallest kept
`s / s_max` as 2.7e-05 (Sir) and the smallest gap as 1.2e+09 against
`PC_MIN_GAP`'s 1e+03 — the conditioning a PC Hessian PASS line now prints
rather than refuses (#194 owns the conditioning itself). `--sc-hessian-kk`
refuses the same two pairs for `rho_t` as on the CASCI-orbital twin.

### The source terms on CASSCF orbitals: the wave-0 canary

The CASSCF-gradient milestone rests on one claim: the source terms are
agnostic to the reference kind. Two `gpu;small` entries test it directly on
the committed CASSCF golden, with no oracle, no `.gradgold` sidecar and no new
reference data. Each is registered beside its CASCI twin and gated by that
twin's own PASS line and tolerances:

| entry | binary, flags | twin |
|---|---|---|
| `nevpt2_cas44_casscf_sc_tangent` | `nevpt2_demo --tiles 3 --sc-tangent` | `nevpt2_cas44_sc_tangent` ("The SC tangent's finite-difference cross-check") |
| `nevpt2_sc_pseudodensity_kappa_cas44_casscf` | `nevpt2_sc_pseudodensity --tiles 3 --kappa-act` | `nevpt2_sc_pseudodensity_kappa_cas44` ("The active-rotation source term") |

Both checks hold the golden's `e_core` / `e_virt` fixed on both sides of the
comparison, and both say so. Each mode differentiates the same functional,
so the duality still bites.

**Why `--kappa-act` and not `--kappa-full`.** Issue #118 named
`--kappa-full`. That check reads the `<case>.gradgold` sidecar beside the
golden. When #118 landed no CASSCF golden had a sidecar, so on
`n2_ccpvdz_cas44_casscf` it stopped with an IO error (`could not open gradient
sidecar`) before computing anything. The issue's description of the check —
no sidecar, the orbital energies held fixed, the `<kappa, Y>` pairing against
the forward `Edot(kappa)` — is `--kappa-act`, which is registered instead.
`--kappa-full` on CASSCF orbitals, and with it the recorded self-variation
envelope ("The kernels' same-run self-variation"), waited for the CASSCF
sidecar in #120. That sidecar is committed now, and since #121
`nevpt2_sc_pseudodensity_kappa_full_n2_ccpvdz_cas44_casscf` runs it beside
the CASCI twin ("The source terms on CASSCF orbitals, measured", below).

Measured on an RTX 3080 (CUDA, Release, `--tiles 3`), one run each, both PASS:

| `nevpt2_cas44_casscf_sc_tangent` | |
|---|---|
| base total, tangent path vs `energiesDevice` | 5.6e-17 |
| norm increment, worst of the eight classes | 5.1e-16 |
| Euler, worst of Sr / Srsi / Srs (energy and norm) | 6.4e-16 |
| `Edot(u)`, seed `0x5eed0032` | -1.3251e-3 |
| finite difference: fourfold halvings / plateau `|err|` (relative) | 8 / 6.3e-12 (4.8e-9) at `h = 2^-17` |

| `nevpt2_sc_pseudodensity_kappa_cas44_casscf` | |
|---|---|
| `<kappa, Y_act>` vs `Edot(kappa)`, worst `|delta|` of 4 directions | 2.3e-30 |
| `max |Y_act|` (the matrix, 6 basis directions) | 9.1e-17 |
| matrix pairing vs direct pairing | 6.1e-30 |
| finite difference: fourfold halvings / plateau `|err|` | 7 / 1.7e-11 at `h = 2^-15` |

**What the `--kappa-act` canary does not show.** At CAS(4,4) on CASSCF
orbitals, `Y_act` is zero to round-off (9.1e-17), just as the CASCI twin's
pairing is a symmetry zero ("The kernels' same-run self-variation"). So the
duality here holds between two numbers that are both zero to round-off. It
shows that the gate admits the file and that the machinery runs on CASSCF
orbitals. It is not a strong test of a non-zero active-rotation source term.
`--sc-tangent`'s `Edot(u)` is non-zero and passes every check, so that entry
carries the canary's weight.

Nothing in `src/` branches on the reference kind anywhere else: the RDM build,
the digests, the energy stage and the PC solve see a CI vector and integrals,
which is exactly the claim these entries make.

### The source terms on CASSCF orbitals, measured

#121 runs every source-term path on `n2_ccpvdz_cas44_casscf`, with its
`.gradgold` sidecar (#120), so the orbital energies move with the active
density as on the CASCI twin. The independent check is #117's four
`.srcgold` oracles, compared block by block. Nine entries are registered,
each in its CASCI twin's loop and with that twin's labels, PASS line and
tolerance. Nothing about them is CASSCF-specific:

| entry | binary, flags | labels |
|---|---|---|
| `nevpt2_gradient_source_cas44_casscf` | `nevpt2_demo --tiles 3 --gradient-source --gradgold …` | `gpu;small` |
| `nevpt2_gradient_source_pc_cas44_casscf` | `nevpt2_demo --tiles 3 --pc --gradient-source --gradgold …` | `gpu;small` |
| `nevpt2_sc_pseudodensity_kappa_full_n2_ccpvdz_cas44_casscf` | `nevpt2_sc_pseudodensity --tiles 3 --kappa-full` | `gpu` |
| `nevpt2_sc_pseudodensity_dump_n2_ccpvdz_cas44_casscf` | `… --tiles 3 --dump-source-terms build/gradsrc/….gradsrc` | `gpu` |
| `nevpt2_sc_pseudodensity_dump_pc_n2_ccpvdz_cas44_casscf` | `… --tiles 3 --pc --dump-source-terms …` | `gpu` |
| `nevpt2_sc_pseudodensity_oracle_n2_ccpvdz_cas44_casscf` | `… --tiles 3 --source-oracle golden/n2_ccpvdz_cas44_casscf.srcgold` | `gpu` |
| `nevpt2_sc_pseudodensity_oracle_pc_n2_ccpvdz_cas44_casscf` | `… --tiles 3 --pc --source-oracle …_casscf_pc.srcgold` | `gpu` |
| `nevpt2_sc_pseudodensity_oracle_dir_n2_ccpvdz_cas44_casscf` | `… --tiles 3 --source-oracle …_casscf_dir.srcgold` | `gpu` |
| `nevpt2_sc_pseudodensity_oracle_dir_pc_n2_ccpvdz_cas44_casscf` | `… --tiles 3 --pc --source-oracle …_casscf_pc_dir.srcgold` | `gpu` |

`--sc-tangent` on this golden is #118's `nevpt2_cas44_casscf_sc_tangent`
(above). It was re-run in the same pass and still passes. Acrolein's CASSCF
directional pair (#135) has its own table: "On CASSCF orbitals: acrolein",
under "The directional source-term oracle".

Measured on an RTX 3080 (CUDA, preset `default`, Release, `--tiles 3`, the
default digest), one run each. **All nine pass, and no block disagrees.**
Against the oracles, max \|GPU − oracle\| per block:

| block | SC full (tol 1e-6) | SC directional (tol 1e-8) | PC directional (tol 2e-8) |
|---|---:|---:|---:|
| core–core | 7.6e-11 | 1.0e-11 | 1.0e-11 |
| core–act_O | 0 | 0 | 0 |
| core–virt | 2.0e-10 | 6.4e-11 | 1.3e-11 |
| act_O–virt | 3.3e-10 | 2.1e-11 | 1.3e-12 |
| virt–virt | 2.2e-11 | 2.5e-11 | 1.1e-12 |
| `P y` | 6.0e-10 (12 det.) | 3.3e-11 | 4.5e-13 |

The PC **full** oracle (`…oracle_pc_n2_ccpvdz_cas44_casscf`) and the PC dump
(`…dump_pc_n2_ccpvdz_cas44_casscf`) have their numbers in one home,
[`pc-nevpt2.md`](pc-nevpt2.md), "On a CASSCF reference", under "The source
terms by reverse mode" (#160).

Every block also sits under 1e-7, including on the SC full oracle, whose gate
is 1e-6 because of PySCF's central difference ("The SC source terms against
PySCF's, on a card"). The worst is SC `P y` at 6.0e-10. The CASCI twin's
worst SC block, core–virt at 2.0e-7, is PySCF's truncation on that one pair,
and it does not appear here.

**Three blocks are empty, and one is zero on both sides.** The four CASSCF
active orbitals are all act_O in the oracle's partition, so core–act_V,
act_O–act_V and act_V–virt have no pairs. The comparator requires them to be
empty on a CASSCF oracle (`test/source_compare/`; see "The PC source terms'
oracle"). core–act_O is exactly 0.0 on both sides of all four files. On N2
that block vanishes by symmetry, as on the CASCI twin, so it **checks nothing
here**. That leaves five blocks with content: core–core, core–virt,
act_O–virt, virt–virt and `P y`.

The device's own checks, on the same runs:

| entry | check | value |
|---|---|---|
| `…gradient_source_cas44_casscf` | worst \|<u, y> − Edot(u)\|: 4 seeded directions, the state, and the full 36-column Jacobian | 1.3e-13 |
| `…gradient_source_pc_cas44_casscf` | the same for PC | 1.5e-15 |
| both | `e_core` / `e_virt` rebuilt from the sidecar against the golden's (relative) | 1.6e-14 |
| `…kappa_full_…_casscf` | integral-half duality, worst of 4 full directions | 3.6e-17 |
| | full duality with the digests' transpose, worst of 4 | 6.9e-16 |
| | the sidecar's promotion table (relative) | 4.6e-14 |
| | same-run self-variation, against the 5.0e-13 envelope | 2.0e-28 |
| | core-internal / virtual-internal finite difference: halvings, plateau \|err\| | 8, 7.5e-13 / 6, 3.3e-12 |
| | Park Eqs. 56–57 multipliers: divided / degenerate, left zero | 175 / 6 |
| `…dump_…_casscf` | the 6 written directions' exact Edot against Σ κ·Y off the matrix | 4.4e-15 |
| | `.gradsrc` read back | bit for bit |

Each wall time is within 0.03 s of its CASCI twin's in the same run, 0.46 s
to 1.56 s. So the CASSCF orbitals cost nothing extra.

The two `small` entries are the `--gradient-source` pair. They run in the
sanitizer tier like their twins, but they have **not** been measured under
any sanitizer preset ("Wall times" below has no row for them).

**The `_df` pair is not included.** `nevpt2_df_demo` has taken
`--gradient-source` and `--sc-tangent` since #162, at frozen orbital
energies ("The CI source term's forward-reverse duality"), but not on the
CASSCF golden, and it takes neither `--dump-source-terms` nor
`--source-oracle` (the `nevpt2.cli:binary_flags` table refuses them). The
two small `_df` cases do carry a density-fitted `.gradgold` since #164
(reference-data.md, "The density-fitted gradient sidecar"), which only
`nevpt2_demo`'s `nevpt2_gradient_source_cas44_df` reads so far; no `_df`
`.srcgold` exists for either reference kind.

## The SC pseudodensity identities

The golden tier's third kind of entry, beside the SC and PC energy checks:
`nevpt2_sc_pseudodensity_cas44`, `_cas88` and `_cas1010`, all `gpu`, run
`nevpt2_sc_pseudodensity` (`apps/sc_pseudodensity/`) and match
`PASS: SC pseudodensity identities hold`. What they assert is **exact**, not a
finite difference.

`src/gradient/` assembles the SC-NEVPT2 gradient's pseudodensities (Park's
Eqs. 41-47 — [`references.md`](references.md), "Analytical gradients"). A
class's norm and Dyall-Hamiltonian expectation value are quadratic forms in
that class's external integrals and linear in the active-space blocks they are
contracted against, so contracting the assembled pseudodensities back against
their own operands must reproduce the class energy — Park Eq. 40 — as an
algebraic identity. Each entry therefore checks, for all eight classes:

| check | against | tolerance |
|---|---|---|
| `(1/2) sum_blocks <x, D>` (Eq. 41's `D`) | the class energy from the amplitudes, `sum_t T_t N_t` | 1e-10 |
| `sum_G <G, M_G>` (Eqs. 42-43's `M` / `E`) | the same, less the one Sir term with no active block | 1e-10 |
| Srs' active integrals and hole RDMs contracted back (Eqs. 44-47) | `<K, E>` | 1e-10 |
| the amplitude energy, per class | `nevpt2.energy`'s own answer for the same state | 1e-10 |
| the amplitude energy, per class | the golden PySCF per-class energy | 1e-7 |

On an RTX 3080 (CUDA, Release, `--tiles 3`) **every** residual at CAS(4,4),
CAS(8,8) and CAS(10,10) is at or below 1.1e-14 — the largest are Sr's, the
class with the n^6 contractions — so the 1e-10 tolerance is four decades of
headroom, and a single mis-transcribed index does not fit inside it. The
assembly itself takes 0.00s / 0.10s / 0.45s at the three sizes, on top of the
RDM build and the class energies the entry also runs.

CAS(4,4) and CAS(8,8) are labelled `small` as well, so the sanitizer tier
covers the new device work; CAS(10,10) is not, for the reason the small cases
exist (below). The host half — the amplitude and multiplier formulas, and the
subscript rewriting that differentiates a class term — is `test/gradient/`,
card-free.

## The SC tangent's finite-difference cross-check

`nevpt2_cas44_sc_tangent` and `nevpt2_cas88_sc_tangent` (`gpu`; the CAS(4,4)
one `small`) run `nevpt2_demo --golden <case> --tiles 3 --sc-tangent` and
match `PASS: the SC tangent matches Euler`. They are **the SC energy
stage's own finite difference, and a one-time one** — the golden tier has
three in all, and they are the three conventions no identity can pin
(`docs/gradients.md`, §2): this one, the `--kappa-act` entries' third number
(the device kappa action against the exponential map, below) and the PC
tangent's (a derivative of a solve, which has no identity at all). One order
up, `--sc-hessian` adds the Hessian's own, a difference of `Edot`
itself ("The SC Hessian's finite-difference cross-check", below), and
`--sc-hessian-vector` one of `y` against the vector `H_cc u` ("The SC
Hessian-vector product's checks", below), `--sc-hessian-kc` four of `y`
along `kappa` against the mixed block's `H_kc^T w` ("The SC mixed Hessian
block's checks", below), `--sc-hessian-kk-int` second differences of the
fixed-pseudodensity pairing along `exp(s u + t v)` against the orbital
block's integral curvature ("The orbital block's integral curvature",
below), and `--sc-hessian-kk` Richardson-extrapolated second
differences of the SC total itself along the same family against the
whole orbital block ("The orbital block's sandwich", below). It
validates `energiesTangentDevice` (`nevpt2.energy`'s `:sc_tangent`
partition) so that
every later check of the reverse-mode assembly can be an exact dot-product
identity against the tangent at this tree's usual 1e-7, with no step size
anywhere. `src/gradient/ci_checks_tangent.cpp` has the derivation; the
entry checks three things.

**The two exact checks.** A class norm is affine in the RDMs, so displacing
them by any `X` must change it by exactly the tangent at `X`
(`norm_k(dm + X) - norm_k(dm) == normDot_k(X)`), which is two runs of the
unmodified `energiesDevice` against one of `energiesTangentDevice`. And for
the three classes whose metric *and* Dyall-Hamiltonian blocks are purely
linear in the RDMs — Sr, Srsi, Srs — the class energy is homogeneous of
degree 2 in the CI vector, so the tangent along the state itself is `2 E_k`
(Euler). The other five classes are affine, not homogeneous (`hdm1`'s
`2 delta`, `hdm2`'s delta-delta terms, `a25`'s `2 h1e^T`, Sir's
one-electron norm term), and Sijrs reads no RDM at all, so its tangent is
exactly zero; no Euler identity holds for those, and they are checked by the
norm increment and by the difference below. On an RTX 3080 (CUDA, Release):

| exact check | CAS(4,4) | CAS(8,8) |
|---|---|---|
| `energiesTangentDevice`'s base total vs `energiesDevice`' | 5.8e-16 | 2.3e-15 |
| the norm increment, worst of the eight classes | 7.5e-16 | 1.5e-14 |
| Euler, worst of Sr / Srsi / Srs (energy and norm) | 4.2e-16 | 1.5e-15 |

**The finite difference.** The total is a *ratio* of quadratics in the CI
vector (the finish divides), so a central difference of `energiesDevice`
along the straight line `c ± h u` — rebuilding dm1/dm2 on the host and
dm3/f3ac/f3ca on the device at every displaced vector — has a genuine
`O(h^2)` error. Measured on an RTX 3080, with `h = 2^-k` for k = 6…19 and one
seeded direction (`|u| = 1`, `<c|u> = 0`):

| case | `Edot(u)` | fourfold halvings | plateau `|err|` | plateau relative | crossover `h` |
|---|---|---|---|---|---|
| CAS(4,4) | 2.8187e-3 | 6 (k = 10…16) | 0.9–1.6e-11 | 3.0–5.6e-9 | 2^-17…2^-18 ≈ 1e-5 |
| CAS(8,8) | -3.9074e-4 | 7 (k = 6…13) | 8.6e-12 | 2.2e-8 | 2^-15 ≈ 3.1e-5 |

The ratio is 4.00 to two decimals through the truncation regime; below the
crossover `h` the error rises again as round-off in `E / 2h`, and the exact
step that reaches the plateau moves between runs (the RDM build's
`atomicAdd`s reassociate, so `E` itself carries ~1e-14). The gate is
therefore the *shape*: at least four consecutive halvings with a ratio in
[3.2, 4.8], and a best relative error under 1e-6 — ~50x above what is
measured.

**One row of the sweep can go wild, and that is a property of differencing
the energy.** A perturber that is empty at the base point (`N_k = H_k = 0` —
N2's symmetry gives plenty) has both growing as `h^2` when the state is
displaced, so its energy `-N^2/(Delta N + H)` is `O(h^2)`, even in `h`, and
invisible to a central difference. But `N` and `H` are then tiny differences
of O(1) contractions, i.e. round-off, and at a step where `Delta N + H`
happens to cancel to nothing the term is numerically enormous: at CAS(4,4),
k = 9 gives `-0.0961` where the derivative is `0.0028`. The tangent is
unaffected (it skips those perturbers exactly as `normToEnergy` does), which
is why the Richardson criterion asks for a run of good halvings rather than
every one of them.

The host half — the scalar finish's derivative, which is the only thing the
tangent differentiates by hand — is `test/energy_finish/`, card-free
(`NormToEnergyDerivativeTests`, `NormToEnergyTangentTests`).

## The SC Hessian's finite-difference cross-check

`nevpt2_cas44_sc_hessian` and `nevpt2_cas88_sc_hessian` (`gpu`; the CAS(4,4)
one `small`) run `nevpt2_demo --golden <case> --tiles 3 --sc-hessian` and
match `PASS: the SC Hessian matches Euler` (#153). They check
`energiesHessianDevice` (`nevpt2.energy`'s `:sc_hessian` partition), the SC
total's second directional derivative `Eddot(u, v)` at the golden's **fixed**
orbital energies, over the RDM sets of `buildRdmHessianTangentsDevice`
([`hessian-theory.md`](hessian-theory.md), §4.1, §4.4).
`src/gradient/ci_checks_tangent.cpp` ("--sc-hessian") has the derivation.
`u` is `--sc-tangent`'s direction (seed `0x5eed0032`) and `v` a second one
(`0x5eed0153`), each `|.| = 1` and orthogonal to the state. The PASS line says
which functional ran (the orbital energies held fixed), as §9.1 asks.

**The exact checks.** Euler at second order — a class homogeneous of degree 2
has `H c = grad E`, so `Eddot_k(c, u) == Edot_k(u)` for Sr, Srsi and Srs —
holds the new 3×3 and the second-order chains against the *already-gated*
`energiesTangentDevice`, run on its own tangent build. The five affine
classes have no such identity (Sijr, Sij, Sir and, at CAS(8,8), Si miss it
by 2e-4–3e-2, printed for the record; CAS(4,4)'s Si is zero by symmetry);
their second-order blocks are held by
the norms' increment at the second-order RDM set, `norm_k(dm + X'') -
norm_k(dm) == N''_k`, exact because the norms are affine in the RDMs. Sijrs
reads no RDM, and its every second derivative is exactly zero. The Hessian
walk's own value and first derivatives come along against `energiesDevice`
and `energiesTangentDevice`. The vector form `H_cc u` is held to this
`Eddot` by duality ("The SC Hessian-vector product's checks", below).
RTX 3080 (CUDA, `default` preset):

| exact check | CAS(4,4) | CAS(8,8) |
|---|---|---|
| the walk's base total vs `energiesDevice`' | 1.1e-15 | 1.5e-15 |
| the walk's `Edot(u)` / `Edot(v)` vs `energiesTangentDevice`'s | 4.6e-15 / 1.2e-14 | 1.5e-17 / 2.6e-16 |
| the second-order norm increment, worst of the eight classes | 1.8e-15 | 2.4e-15 |
| Euler, worst of Sr / Srsi / Srs | 4.2e-15 | 3.1e-17 |

All under the same 1e-10 as `--sc-tangent`'s exact checks.

**The finite difference: Edot, not E.** The milestone's one new difference
([`hessian-theory.md`](hessian-theory.md), §9): `energiesTangentDevice`'s
`Edot(u)` at `c ± h v`, each point a fresh host dm1/dm2 rebuild and a fresh
plain and tangent RDM build, differenced against `Eddot(u, v)`. `Edot` is a
ratio of polynomials in the CI vector, so the error is `O(h^2)`, with the
same `h = 2^-k` ladder (k = 6…19) and the same gate as the tangent's: at
least four consecutive halvings with a ratio in [3.2, 4.8], and a best
relative error under 1e-6. Measured on the RTX 3080:

| case | `Eddot(u, v)` | fourfold halvings | plateau `|err|` | plateau relative | at `h` |
|---|---|---|---|---|---|
| CAS(4,4) | 2.7746e-2 | 7 (k = 12…18) | 3.2e-10 | 1.2e-8 | 2^-18 |
| CAS(8,8) | 3.9897e-3 | 10 (k = 7…16) | 4.2e-11 | 1.1e-8 | 2^-19 |

At CAS(8,8) the plateau is reached at k = 17 already (6.7e-11) and is flat to
k = 19. The CAS(4,4) sweep shows the empty-perturber row the tangent's
does, at the same step: k = 9 gives -1.36e4 where the derivative is 0.0277 —
the same cancelling `Delta N + H` of a perturber empty at the base point
("The SC tangent's finite-difference cross-check", above), now inside `Edot`'s
finish; the Richardson run is taken from k = 12 on. The gate leaves a factor
of ~100 over the measured plateau.

The host half — Eddot's one new expression, the 3×3 between two first-order
tangents — is `test/energy_finish/`'s `NormToEnergyHessianSandwichTests`
(card-free), and the three-argument build's three sets are each the tangent
build at their pair to 1e-12 in `test/rdm/rdm_tangent_tests.cpp`'s
`RdmHessianTangentsTests`.

## The SC Hessian-vector product's checks

`nevpt2_cas44_sc_hessian_vector` and `nevpt2_cas88_sc_hessian_vector`
(`gpu`; the CAS(4,4) one `small`) run `nevpt2_demo --golden <case> --tiles 3
--sc-hessian-vector` and match `PASS: the SC Hessian-vector product matches`
(#156). They check `scCiHessianVectorDevice` (`nevpt2.gradient`), the vector
`H_cc u` by reverse over forward — the pseudodensity walk differentiated
along `u`, with the multipliers' tangents as its one new ingredient, then
`sigma(gdot[u], c) + sigma(g, u)` ([`hessian-theory.md`](hessian-theory.md),
§4.2) — at the golden's **fixed** orbital energies, on every side of every
check. `u` is `--sc-hessian`'s (seed `0x5eed0032`).
`src/gradient/ci_checks_tangent.cpp` ("--sc-hessian-vector") has the
derivation. RTX 3080 (CUDA, `default` preset), the worst of three runs at
each size (the RDM build's `atomicAdd`s reassociate, so runs differ):

| exact check | CAS(4,4) | CAS(8,8) | gate |
|---|---|---|---|
| the walk's own `Edot(u)` vs `energiesTangentDevice`'s, worst class or total | 7.7e-15 | 1.0e-16 | 1e-10 |
| `<u, y>` of the same walk's `y` vs `Edot(u)` | 7.7e-15 | 1.6e-16 | 1e-7 |
| the duality `<v, H_cc u> == Eddot(u, v)`, worst of four seeded `v` | 2.9e-13 | 4.9e-16 | 1e-7 |
| the same, `v` = the state | 7.4e-12 | 1.5e-15 | 1e-7 |
| column by column, `<v, H_cc e_I>` vs `Eddot(e_I, v)`, worst of 36 | 4.3e-12 | not run (ndet 4900) | 1e-7 |
| `H_cc`'s symmetry, `max|H_IJ - H_JI|` relative to `max|H_IJ|` | 2.4e-11 | not run | 1e-9 |

`Eddot` is `energiesHessianDevice` on its own three-argument build, so the
duality's two sides share nothing past the RDM builders. The first row pins
the walk's tangent blocks — the CI direction's — to the forward tangent's.

**`H_cc` is large along the state, and that is the functional's.** At
CAS(4,4) `max|H_IJ|` is 1.82e4, on the diagonal of the dominant determinant
(`c_0` = 0.974), against entries of 0.36–0.57 on the determinants the state
does not occupy. The functional is the *unnormalized* one, and the hole
metrics' RDM-free terms (`hdm1`'s `2 delta` and its kin) make a nearly-filled
hole perturber's norm move at first order along `c`; the finish's `-2/d`
does the rest. Symmetry's absolute residual (2.3–4.4e-7) scales with those
entries, which is why that one gate is relative; the gate leaves about forty
over the worst measured residual.

**The finite difference: `y`, the GPU's own exact one.** `scCiSourceDevice`'s
`y` at `c ± h u`, each point a fresh host dm1/dm2 rebuild and a fresh RDM
build, **with no orbital-energy response on either leg**, differenced against
`H_cc u`, on the difference's 2-norm. Same ladder (k = 6…19) and Richardson
rule as the tangent's; the gate on the best relative error is 1e-5:

| case | `|H_cc u|` | fourfold halvings | plateau `|err|` | plateau relative | at `h` |
|---|---|---|---|---|---|
| CAS(4,4) | 0.86090 | 6 (k = 12…17) | 1.4–4.3e-7 | 1.7–5.0e-7 | 2^-18 or 2^-19 |
| CAS(8,8) | 0.69375 | 10 (k = 7…16) | 4.1–4.5e-9 | 5.9–6.4e-9 | 2^-18 |

A vector difference has a higher floor than `Edot`'s scalar one. Every
component of `y` is differenced, the state's own determinants included, and
the curvature above makes both the truncation constant and `y`'s round-off
decades larger there. At CAS(4,4) the two cross within one step: in one run
the rows either side of a 1.7e-7 plateau are 2.5e-6 (k = 17, still
truncation) and 1.2e-6 (k = 19, already round-off), and in another the best
row is 5.0e-7. The 1e-6 the scalar differences use would pass only on a
lucky row, so the vector difference has its own gate, twenty over the worst
measured best. **A longer ladder does not help:** run down to
k = 26, both cases' errors rise again below 2^-18, to 8.4e-5 (CAS(4,4)) and
1.5e-5 (CAS(8,8)) relative at 2^-26. The CAS(4,4) sweep shows the
empty-perturber row at k = 9 again (relative error 6.6e8, "The SC tangent's
finite-difference cross-check", above), and the Richardson run starts after
it.

The host half — the multipliers' tangents per slab, including the two
symmetrized classes' packing — is `test/gradient_plan/`'s
`SlabMultiplierTests` (card-free), over `nevpt2.energy_finish`'s
`normToEnergyMultiplierTangent`, which `test/energy_finish/` already pins.

## The SC mixed Hessian block's checks

`nevpt2_sc_pseudodensity_cas44_sc_hessian_kc` and
`nevpt2_sc_pseudodensity_cas88_sc_hessian_kc` (`gpu`; the CAS(4,4) one
`small`) run `nevpt2_sc_pseudodensity --golden <case> --tiles 3
--sc-hessian-kc` and match `PASS: the SC mixed Hessian block H_kc's two
orientations agree` (#184). They check `nevpt2.gradient`'s
`scMixedHessianDevice` — the scalar `Eddot_kc(w, u)`, forward over forward,
in Sec. 5.5's five terms — and `scMixedHessianVectorDevice` — the vector
`H_kc^T w`, `y` differentiated along `w` — against each other and against
what neither shares ([`hessian-theory.md`](hessian-theory.md), §5.5). Both
read the committed sidecar. `apps/sc_pseudodensity/hessian_kc_check.cppm`
has the derivation; every check prints its orbital-energy mode: FULL (they
respond to `kappa` and to `c`), EULER (to `kappa` alone) or FROZEN (to
neither, the one mode a case with no sidecar has). `w` is a seeded fully
mixed antisymmetric `kappa` (seed `0x5eed0184`), `u` three seeded CI
directions and the state. RTX 3080 (CUDA, `default` preset), the worst of
three runs at each size, 2026-10-20:

| exact check | mode | CAS(4,4) | CAS(8,8) | gate |
|---|---|---|---|---|
| `h^eff + f^gamma[gamma]` against the sidecar's `f`, the whole range (relative) | — | 1.1e-16 | 4.2e-16 | 1e-10 |
| the walk's own `Edot(w)`, digests merged per tuple, against the frozen-digest tangent + `eps` + #88's term, worst class | FULL | 1.9e-14 | 8.0e-15 | 1e-7 |
| the duality `<H_kc^T w, u> == Eddot_kc(w, u)`, worst of three seeded `u` | FULL | 1.1e-13 | 2.1e-16 | 1e-7 |
| the same, `u` = the state | FULL | 2.3e-11 | 1.5e-14 | 1e-7 |
| column by column, `(H_kc^T w)_I` vs `Eddot_kc(w, e_I)`, worst of 36 | FULL | 3.0e-11 | not run (ndet 4900) | 1e-7 |
| Euler, `Eddot_kc(w, c)` vs `2 <w, Y>`, worst of Sr / Srsi / Srs | EULER | 4.5e-14 | 1.6e-14 | 1e-7 |
| the duality in that mode, a seeded `u` and the state | EULER | 5.9e-12 | 2.1e-14 | 1e-7 |
| along a core-core `w`: `|C3|` (`max|h1e'[w]|`) | FULL | 1.1e-19 (9.9e-17) | 1.6e-19 (3.1e-16) | 1e-13 |
| along a virtual-virtual `w`: `|C3|` | FULL | 0 | 0 | 1e-13 |
| along both: the duality | FULL | 2.0e-15 | 4.7e-17 | 1e-7 |
| no sidecar, an active-active `w`: the duality, three seeded `u` and the state | FROZEN | 1.2e-14 | 3.4e-14 | 1e-7 |
| the same: Euler, worst of Sr / Srsi / Srs | FROZEN | 7.0e-28 | 4.0e-15 | 1e-7 |
| the same: the walk's `Edot(w)` against `scActiveRotationDevice`'s tangent + the digests' term, worst class | FROZEN | 2.5e-28 | 1.4e-15 | 1e-7 |

Exact in every run, at both sizes: along both intra-space directions
`max|h2e'[w]|`, C4 and C1's digest share are `0.0`; in the Euler mode Sijrs'
`Eddot_kc(w, c)` is `0.0` (its only route to `c` is the orbital energies,
held fixed in `c` there); and with no sidecar both refusals fire — a `w`
leaving the active space and a responding mode, each Unsupported.

**C3 along a core-core `w` is round-off, not `0.0`, and that is `h1e'`'s.**
`h2e'[w]` is an exactly-zero tensor there, so C4 and C1's digest share
vanish identically, as §5.5 says. `h1e'[w]` is `h^eff`'s active block's
tangent, and its core dressing's two promotions of the tied core pair cancel
against an antisymmetric `w` in exact arithmetic but only to round-off in the
einsum's accumulation order (`max|h1e'[w]|` above), so C3 is that round-off
carried through the blocks. Along a virtual-virtual `w`, which no dressing
reads, `h1e'[w]` and C3 are `0.0`. The C3 gate leaves six decades over the
worst measured and sits decades below any real term.

**The Euler rows at CAS(4,4) with no sidecar are zero by symmetry.** The four
active orbitals of N2's CAS(4,4) are four different irreps, so every
active-active rotation's source term is exactly zero (as "The
active-rotation source term" records for `--kappa-act`), and both sides of
that Euler row are at `1e-17`. The duality along the same `w` is not
trivial — the seeded `u` span every irrep — and CAS(8,8) carries the
non-trivial Euler rows.

**C1's digest share alone is not small.** Along the full-mode `w` and the
first `u`, Sr's is 3.3e-1 at CAS(4,4) against a C1 of -1.1e-3 for the class
(1.9e-3 against 1.3e-4 at CAS(8,8)): the digests' and the integrals' shares
of `Hdot[w]` cancel per tuple, as the first-order digest term and the
frozen-digest tangent do (`--kappa-full`'s table). The per-tuple merge is
therefore not an optimisation but the only order in which the 3x3 meets the
right number; the merge row above is what pins it.

**The finite differences: `y`, the GPU's own exact one, dotted with `u`.**
`scCiSourceDevice`'s `y` at `exp(s w)`, both legs in the same mode as the
vector it is held to, differenced against `<H_kc^T w, u>`: along a core-core
and a virtual-virtual `w` (the stored blocks' inactive axes, the orbital
energies and `W` rotated on the host — `W` out of the sidecar's untied
`(cc|aa)` / `(ca|ca)` and `(vv|aa)` / `(va|va)` rows, held at `s = 0` to the
tied rows' `W`), an active-active one in the FULL mode (the active axes
rotated, the digests rebuilt, `W` rotated on its active pair and the orbital
energies re-formed from it) and the same active-active one in the FROZEN
mode. The same ladder (k = 6…18), run rule and gate (1e-6 of
`max(|<H_kc^T w, u>|, |<u, y>|)`) as `--kappa-full`'s sweeps:

| direction, mode | CAS(4,4) best relative | fourfold halvings | CAS(8,8) best relative | fourfold halvings |
|---|---|---|---|---|
| core-core, FULL | 1.0e-10–5.2e-9 | 6–8 | 1.0e-9–2.8e-9 | 7–9 |
| virtual-virtual, FULL | 6.9e-11–6.0e-10 | 5–6 | 4.6e-12–1.7e-9 | 4–6 |
| active-active, FULL | 2.8e-9–1.4e-8 | 7–9 | 2.1e-9–2.6e-9 | 5 |
| active-active, FROZEN | 4.8e-9–2.8e-8 | 7–8 | 6.3e-10–2.0e-9 | 8 |

These are the only checks that see C5, the orbital energies' mixed response
(the duality's two sides share its generator, and Euler runs without it);
the full-range `f` row above is the one check its Fock line's arrays get
beside them. A run takes about 3 s at CAS(4,4) and 77 s at CAS(8,8), the
four ladders' 108 `y` builds being most of the latter.

The host half — the five terms' per-perturber finish, plain and through the
two symmetrized classes' packing, and the multipliers' tangents with a Delta
tangent — is `test/gradient_plan/`'s `MixedTermsTests` and
`SlabMultiplierTests` (card-free), each against a central difference of the
class energy or of the multipliers themselves; the flag's refusals are
`test/cli/`'s `CliScPseudodensityFlagTests`.

## The orbital block's integral curvature

`nevpt2_sc_pseudodensity_cas44_sc_hessian_kk_int`,
`nevpt2_sc_pseudodensity_cas88_sc_hessian_kk_int` and
`nevpt2_sc_pseudodensity_cas44_df_sc_hessian_kk_int` (`gpu`; the two
CAS(4,4) ones `small`) run `nevpt2_sc_pseudodensity --golden <case> --tiles
3 --sc-hessian-kk-int` and match `PASS: the SC orbital Hessian's integral
curvature <Ehat', I''> = K1 + K2 agrees` — the fitted one with its route
named (#154). They check `nevpt2.gradient`'s `scOrbitalCurvatureDevice`: the
first of the orbital Hessian block's two terms, `<Ehat', I''[u, v]>`, as K1
(the first-order pairing at `W = (uv + vu)/2`, the traced pairs restored)
plus K2 (the cross-slot term) ([`hessian-theory.md`](hessian-theory.md),
§5.1 and §5.4; `apps/sc_pseudodensity/hessian_kk_check.cppm` has the
checks). Every term is reported in both modes with the digests' term in
each — reverse through #89's `hbar`, forward through the RDM build at
`h2e''`. The directions are seeded (base `0x5eed0154`): an active–active
`u` and `v`, two fully mixed kappas, and two each inside the core and the
virtual space. RTX 3080 (CUDA, `default` preset), the worst of three runs
at each case, 2026-10-20:

| exact check | CAS(4,4) | CAS(8,8) | CAS(4,4) DF | gate |
|---|---|---|---|---|
| the sidecar against itself, worst family (relative) | 3.5e-14 | 2.8e-13 | 2.1e-14 | 1e-10 |
| K1's matrix path against `--kappa-full`'s first-order pairing at a full kappa, adjoint / tangent's integral half | 6.5e-14 / 6.5e-14 | 1.8e-14 / 1.8e-14 | 5.4e-14 / 5.4e-14 | 1e-10 |
| the restored traced terms alone at that kappa / at an active–active one | 1.9e-19 / 9.5e-19 | 1.6e-18 / 1.0e-17 | 2.2e-20 / 1.7e-19 | 1e-12 |
| the duality, adjoint against tangent, worst term of every pair | 3.7e-13 | 4.6e-14 | 5.1e-13 | 1e-7 |
| the swap, `(full, active)` against `(active, full)` | 1.6e-14 | 9.4e-16 | 1.6e-14 | 1e-7 |
| the traced pairs' K1 + K2 along the active x active pair (K1's share) | 3.5e-18 (8.1e-3) | 4.0e-18 (−2.5e-4) | 1.7e-18 (8.1e-3) | 1e-12 |
| core x core and virtual x virtual, conventional: K1 alone, its duality | 4.3e-16 | 2.4e-16 | — | 1e-7 |
| core x core, fitted: the active blocks' K1 + K2 | — | — | 2.7e-16 | 1e-12 |
| K1 sidecar against K1 three-index, and K2 composition against K2 three-index, worst pair | — | — | 4.2e-13 | 1e-7 |

The pairs are `active x full` and its swap, `active x active`, `active x
core` and `active x virtual` at every case; the fitted case adds `full x
full`, `core x virtual`, `core x core` and `virtual x virtual`, whose K2 is
the three-index jet's alone. Exact in every run: along the intra-space pairs
the digests' term is `0.0` — reverse on every case, and forward and reverse
for both terms on the fitted one, with `max|h2e''|` `0.0` there — and K2 is
refused as Unsupported on the conventional cases; along `active x core` and
`active x virtual`, K1 is `0.0` (`W` vanishes for two directions in
different spaces). The refusals fire on every run: a conventional `full x
full` pair (and on the fitted case the same pair asked for the sidecar
route), and the three-index route on a conventional case.

**The traced pairs' zero is structural, and that is what it pins.** Both
restored terms depend on their matrix only through `M + M^T`, so at `W` and
at `v u` they are the same contraction of `2W` — the sum is zero for any
correct restoration and nonzero for a wrong one, while each share is the
size of K1's other pieces (the parenthesised column).

**The ladders: the pairing itself, differenced.** `L(U) = <Ehat'_0, I(U)>`,
the pseudodensities, `R_t` and `hbar` fixed at the state, has exactly this
term as its mixed second derivative along `exp(s u + t v)`. Per step `h`
the four corners' integral sets — every stored block's axes rotated on the
host, the orbital energies the rotated generalized Fock's diagonal with
`gamma` rotated on its active pair — are combined before the pairing
(`scDottedPairingDevice`) and divided by `4h^2`. The ladder runs `h = 2^-k`,
`k = 4…16` (four halvings earlier than the first-order sweeps', a second
difference's round-off growing as `1/h^2`), and is gated on the best error
relative to `max(|term|, |<u, Y>|, |<v, Y>|)` under 1e-6 and a run of three
fourfold halvings:

| pair | CAS(4,4) best relative (halvings) | CAS(8,8) | CAS(4,4) DF |
|---|---|---|---|
| active x active | 1.7e-8 (7) | 7.6e-8 (6) | 2.0e-8 (8) |
| active x core | 6.8e-9 (7) | 6.5e-9 (6) | 9.2e-9 (7) |
| active x virtual | 2.9e-9 (7) | 6.8e-8 (7) | 2.6e-9 (7) |
| core x virtual | — | — | 6.4e-9 (7) |
| core x core | — | — | 6.3e-9 (8) |
| virtual x virtual | — | — | 3.0e-8 (6) |
| along `exp(s u) exp(t v)`, against the term + (1/2) `<[u, v], Y>` | 1.2e-8 (8), active x active | 3.4e-8 (6), active x active | 4.7e-9 (6), virtual x virtual |

The last row is the parameterisation's own offset (§5.1): it is 5.0e-4 at
CAS(8,8) and 4.8e-4 on the fitted case's `virtual x virtual` pair, a million
times the ladder's error — and 2.4e-17 at conventional CAS(4,4), where the
only same-space pair is `active x active` and N2's four π orbitals are four
irreps, so `<[u, v], Y>` vanishes by symmetry (as `Y_aa` does: `<u, Y>` and
`<v, Y>` are 1e-16 there) and the row checks the term again. A mixed pair
cannot be displaced from stored blocks; the composition behind it is the
same code the block-diagonal pairs run, and on the fitted case the three-index
jet, which shares nothing with it past `B`, agrees with it at 1e-13 on
`active x full`.

A run takes about 1 s at CAS(4,4), 10 s at CAS(8,8) (the forward digests —
one plain RDM build per term per pair, printed as they run — are most of
it) and 2 s on the fitted CAS(4,4).

## The orbital block's sandwich

`nevpt2_sc_pseudodensity_cas44_sc_hessian_kk`,
`nevpt2_sc_pseudodensity_cas88_sc_hessian_kk` and
`nevpt2_sc_pseudodensity_cas44_df_sc_hessian_kk` (`gpu`; the two CAS(4,4)
ones `small`) run `nevpt2_sc_pseudodensity --golden <case> --tiles 3
--sc-hessian-kk`. They match `PASS: the SC orbital Hessian's sandwich I'^T
Ehat'' I' = K3 + K4 + K5 + K6 agrees with itself by duality ... 2 pairs
refused, with no Hessian along them (rho_t, Sec. 5.6)`, and the fitted one
ends `(density-fitted)` (#158, #190). They check `nevpt2.gradient`'s `scOrbitalSandwichDevice`, the second
of the orbital Hessian block's two terms, and `scOrbitalHessianDevice`, the
whole `Eddot_kk(u, v) = <Ehat', I''> + I'^T Ehat'' I'`
([`hessian-theory.md`](hessian-theory.md), §5.1 and §5.6;
`apps/sc_pseudodensity/hessian_kk_sandwich_check.cppm` has the checks). The
directions are seeded (base `0x5eed0158`): two active–active, two fully
mixed, two each inside the core and the virtual space, and `2 virtual +
core` (#190). The pairs are `active x full` and its swap, `full x full`,
`active x active`, `active x core`, `active x virtual`, `core x virtual`,
`core x core`, `virtual x virtual` and `virtual x (2 virtual + core)`. RTX
3080 (CUDA, `default` preset), the worst of thirteen runs at each CAS(4,4)
flavour (twelve at `--tiles 3`, one at `--tiles 1`) and of two at CAS(8,8)
(one at each), 2026-10-20:

| exact check | CAS(4,4) | CAS(8,8) | CAS(4,4) DF | gate |
|---|---|---|---|---|
| the merge: the walk's own `Edot(u)`, `Edot(v)` (the digests' share in every tuple's `Hdot`) against `--kappa-full`'s `<k, Y>` with `hbar` | 5.2e-14 | 1.2e-14 | 3.6e-14 | 1e-7 |
| the duality, forward K3 + K4 + K5 + K6 against the reverse pairing `<Ddot[v], X'[u]> + <Mdot_G[v], G'[u]> + Rdot[v] . Ddot[u]`, worst pair | 3.9e-16 | 7.1e-16 | 4.4e-16 | 1e-7 |
| K6 against its own reverse half (the ray-limit multipliers' pseudodensity tangent paired with `u`'s dotted slabs), worst pair | 6.3e-17 | 2.6e-18 | 5.6e-17 | 1e-7 |
| the swap, `(full, active)` against `(active, full)`: the sandwich / `Eddot_kk` | 1.3e-14 / 1.5e-14 | 4.8e-16 / 9.1e-16 | 1.4e-14 / 2.4e-14 | 1e-7 |
| K3 and K4 from the exchanged finish | 2.0e-20 | 3.3e-19 | 2.5e-20 | 1e-7 |
| the digests alone: K3's digest x digest share against `energiesHessianDevice` at `f3'[u]`, `f3'[v]`, and with the curvature's reverse digests' term against its forward one at `h2e''`, worst | 2.1e-13 | 2.7e-14 | 2.7e-13 | 1e-7 |
| the intra-space directions' K5 half (`h1e'`'s round-off) | 6.1e-19 | 6.3e-19 | 2.1e-18 | 1e-13 |

Exact in every run: along a core–core or virtual–virtual direction `h2e'`
and the digests' tangent `f3'` are `0.0`, and so are the digests' shares of
K3 and of K5 with that direction's Dyall block. K3's digest x digest share
is 8.8e-2 on CAS(8,8)'s `active x full`, so that check is not a check of
zero there; at CAS(4,4) it is 1e-16 or less, because an active direction
that mixes N2's four π irreps leaves every tuple's first-order triple zero
(the reason `<u, Y_aa>` vanishes there).

**The guarded tuples: K6, and the pairs with no Hessian (§5.6, #190).** The
tuples the vanishing-norm guard skips and the integrals leave empty add K6
where both directions populate them. Sijrs takes its own `P = -1/Delta` on
every guarded tuple. An H class adds K6 only where its tables are
proportional, and is refused otherwise. K6 along the pairs that carry one, the
same in every run:

| pair | CAS(4,4) | CAS(8,8) | CAS(4,4) DF |
|---|---|---|---|
| `core x core` (Sijrs alone) | 1.1756706222e-2 | −1.2488140693e-3 | 1.1732463451e-2 |
| `core x virtual` (Sijrs alone) | 3.2414869997e-4 | 2.3240140444e-5 | 3.2417719591e-4 |
| `virtual x (2 virtual + core)`, all / the H classes' | −4.1957e-2 / −2.8694e-2 | −6.0450e-3 / −4.0215e-3 | −4.1979e-2 / −2.8729e-2 |

The CAS(4,4) `core x core` value is #158's measured ladder gap, 1.1756707e-2
("The empty-perturber term, offline"). The pair's sandwich is now
3.45705779e-2, against that run's ladder 3.45705786e-2. In
`virtual x (2 virtual + core)` the core part populates no H class's empty
tuple, so every H class with empty tuples is populated by both directions with
proportional tables: Sr, Sijr, Srsi, Srs and Sir at CAS(4,4) (Si's empty
tuples are not populated by `virtual`), Sijr and Srsi at CAS(8,8).

`full x full` and `virtual x virtual` are refused as Numerical on all three
cases, every H class with populated empty tuples named. No other pair is
refused. No guarded tuple was empty by the state on any case. The separations
the thresholds sit in, worst over the runs above:

| quantity | computed pairs | refused pairs | threshold |
|---|---|---|---|
| `rho_t`, H classes | ≤1.4e-13 (CAS(4,4)), ≤1.1e-13 (DF), ≤3.3e-16 (CAS(8,8)) | a (class, pair)'s worst ≥4.5e-2 (CAS(4,4), both flavours), ≥2.9e-2 (CAS(8,8)) | `kRhoTol` 1e-7 |
| `N^{ww}/\|w\|^2` on an empty H-class tuple, unpopulated / populated | ≤1.9e-32 / ≥2.5e-7 (CAS(4,4) ≥2.5e-7, CAS(8,8) ≥3.7e-6) | — | `kPopulatedTol` 1e-14 |

The host's are ≤2.3e-12 / ≥2.1e-2 for `rho_t` ("The empty-perturber term,
offline"). `Eddot_kk` along `full x full` is refused as Unsupported on the
conventional cases (#154's K2 refusal, before any device work) and as
Numerical on the fitted one.

**The ladders: the SC total, differenced.** `E(U)` at fixed CI along `U =
exp(s u + t v)`: every stored block's axes rotated on the host, the orbital
energies the rotated generalized Fock's diagonal, and the digests rebuilt at
each corner's `h2e` (one plain RDM build and one `energiesDevice` per
corner). Its second difference converges to `Eddot_kk`, and `E(U) - L(U)`,
with `L` the fixed-pseudodensity pairing of "The orbital block's integral
curvature", to the sandwich alone. The steps are `h = 2^-k`, `k = 2…16`. The
gate is the Richardson extrapolation `(4 D(h/2) - D(h)) / 3`, whose best
error relative to `max(|Eddot_kk|, |<u, Y>|, |<v, Y>|)` must be under 1e-6.

**The ladders' shape rule (#190), and why it differs from #158's.** The
extrapolation is verified by the raw error's halvings, and that check is now
floor-aware:

- Each rung's round-off floor is `eps max_c |E_c| / h^2`, where the `E_c` are
  the four corner SC totals that rung differences. A rung whose error is
  within 10x its floor is left out of the ratio test.
- Above the floor, two consecutive halvings must each cut the error by at
  least 3.2. There is no upper bound.
- The steps start at `2^-2` rather than `2^-4`, for every pair.

The lower bound is what catches a ladder converging to the wrong number: it
sits flat, at ratio 1.00 per halving, as #158's first run did along
`core x core`. #158's rule (three halvings inside (3.2, 4.8), from `2^-4`)
failed three ladders that K6 closes, each converging under the gate:

- `core x core` at CAS(4,4) (both flavours). It halves by 5.9, 8.7 and 10.0 on
  its way to within 1.6e-9 of the target. That is a large h^4 term along the
  pair, not a wrong target.
- `virtual x (2 virtual + core)` at CAS(8,8). Its truncation error is 8e-6
  already at `2^-4` and meets the floor two halvings later; starting at `2^-2`
  shows them.
- `core x virtual` at CAS(4,4). The ratio that decided its third halving sat
  near the floor, and came out anywhere from 3.07 to 4.41 run to run (failing
  3 of 21 runs under the rule in between, which dropped the upper bound and
  started at `2^-2`).

Under the floor-aware rule, `core x virtual` passed 13 runs of 13 on each
CAS(4,4) flavour (twelve at `--tiles 3`, one at `--tiles 1`). Every other
ladder passed in the same runs and in CAS(8,8)'s two. Its two halvings are
the coarse ones, `2^-2` to `2^-4` (4.34, 6.21), whose errors (1e-5 to 1e-6)
are far above any rung's noise. Worst extrapolated relative error over those
runs (fewest halvings above the floor), `E` / `E - L`:

| pair | CAS(4,4) | CAS(8,8) | CAS(4,4) DF |
|---|---|---|---|
| active x active | 1.3e-8 / 1.1e-8 (9) | 1.2e-8 / 1.1e-8 (7) | 2.1e-8 / 2.1e-8 (9) |
| active x core | 2.1e-7 / 2.1e-7 (4) | 5.8e-8 / 5.8e-8 (8) | 1.8e-7 / 1.8e-7 (3) |
| active x virtual | 1.3e-7 / 1.3e-7 (6) | 8.5e-8 / 8.5e-8 (6) | 1.4e-7 / 1.4e-7 (6) |
| core x virtual | — / 1.0e-7 (2) | — / 6.1e-8 (7) | 2.3e-7 / 2.3e-7 (2) |
| core x core | — / 7.2e-8 (8) | — / 2.1e-8 (9) | 1.9e-8 / 1.9e-8 (8) |
| virtual x (2 virtual + core) | — / 2.8e-9 (4) | — / 3.3e-8 (3) | 2.0e-8 / 2.0e-8 (4) |

A dash is `Eddot_kk` refused on a conventional case (K2). The refused pairs
run no ladder. What a run costs is in [`performance.md`](performance.md),
"Tile floors" (`--sc-hessian-kk`).

The pair finish's assembly, card-free, is `test/gradient_plan/`'s
`PairTermsTests` (K3 + K4 + K5 against a mixed central difference of the
class energy, plain and through Sijr's packing), `GuardedCensusTests`
(§5.6's kinds of guarded tuple by hand, K6 against the tuple's own energy
along the plane, `rho_t`'s scale invariance, Sijrs's exemption, Sijr's
packing scattered back) and `SijrsFinishTests` (Sijrs's unguarded finish);
the flag's refusals are `test/cli/`'s `CliScPseudodensityFlagTests`.

## The empty-perturber term, offline

[`hessian-theory.md`](hessian-theory.md), §5.6 has the theory: the orbital
Hessian's sixth term (K6) on the tuples the guard skips, and the pairs along
which no Hessian exists. Two kinds of evidence back it, and neither is a ctest
entry.

**#158's run, on the card.** The WIP branch `issue-158-ehat-sandwich`
(218fe4b, with its `SW_DIAG` diagnostics) ran `nevpt2_sc_pseudodensity
--golden golden/n2_ccpvdz_cas44.nevpt2gold --tiles 3 --sc-hessian-kk` on
2026-10-20. The run was CUDA, `default` preset; the log does not name the
card. Its (E − L) ladder along `exp(s u + t v)` against K3 + K4 + K5:

| pair | ladder (E − L) | K3 + K4 + K5 | gap | Sijrs's K4 at `P = -1/Delta` on its guarded tuples |
|---|---|---|---|---|
| core x core | 3.45705786e-2 | 2.28138717e-2 | 1.1756707e-2 | 1.1756706e-2 (all tuples 1.2594490e-2, less the guarded run's 8.3778370e-4) |
| core x virtual | 3.189e-3 | 2.865e-3 | 3.24e-4 | not printed |
| virtual x virtual | 6.2687e-3 | 2.9145e-3 | 3.354e-3 | — (the H classes contribute; refused below) |

From `2^-7` to `2^-13` each of the three ladders changed by less than 0.02%,
at a ratio of 1.00 per halving. That is the step-independent polarisation
§5.6 predicts, not a failure to converge. The run's own
proportionality diagnostic (`|H/N(uu, vv) − H/N(uv)|`, in Eh) printed 0
outside `full x full` and `virtual x virtual`. On those two pairs it printed
0.17–11 for Sr, Si, Sijr, Srs and Sir, and 2.7–4.6 for Sijr at CAS(8,8).
Scaling `v` by 2, −1 and 0.5 scaled every ladder by the same factor, to six
figures at `h = 2^-10`, as homogeneity requires. So that check could not see
the defect.

**The host model.** `reference_data/sc_empty_perturber_probe.py` transcribes
PySCF's eight SC class formulas onto a golden's stored blocks. Its
`norm_h` reproduces each golden's `class_energies` to ≤1.2e-15. It rotates
the external blocks and orbital energies along intra-space directions, the
transformation the device's ladder applies, and evaluates §5.6's algebra on
the empty tuples from the class formulas at the first-order dotted blocks,
by polarisation, with no step. It holds the result against a
Richardson-extrapolated four-point ladder (`h = 2^-5…2^-7`) of the empty
tuples' energies under real rotations. The directions are seed 190: two
random core-core rotations, two random virtual-virtual ones, and two
proportional controls. The first control is `virt+sym`, `virt` plus a random
rotation in the generators that populate no empty tuple, found from the
integrals. The second is `2virt`. The box's CPU ran it, 4 threads, no card,
on 2026-10-20:

```bash
uv run python reference_data/sc_empty_perturber_probe.py golden/<case>.nevpt2gold
```

The census. "Empty" means every integral the tuple reads is ≤1e-12;
"small" means a guarded tuple with a larger one:

| golden | guarded tuples, by class (empty / small) | largest \|x_t\| on an empty tuple | smallest on a small one |
|---|---|---|---|
| `n2_ccpvdz_cas44` (and `_df`) | Sr 11, Si 5, Sijrs 7912, Sijr 165, Srsi 510, Srs 176, Sir 40 (all empty) | 2.7e-15 (`_df` 8.5e-16) | — |
| `n2_ccpvdz_cas88` (and `_df`) | Sijrs 2308, Sijr 24, Srsi 120 (all empty) | 2.1e-15 (`_df` 2.9e-16) | — |
| `lif_def2svp_cas46` | Sr 9, Si 4, Sijrs 1760, Sijr 90, Srsi 220, Srs 72, Sir 16 (all empty) | 1.2e-16 | — |
| `acrolein_def2svp_cas65` | Sijrs 198144 / 640, Sijr 0 / 41 | 0.0 | 2.6e-9 (Sijrs), 5.8e-8 (Sijr) |

No guarded tuple on any of the six is empty by the state. Acrolein's 41
small Sijr tuples have `N_t` from 4.5e-16 to 9.9e-15, against an unguarded
minimum of 1.06e-14: a continuum the guard cuts through.

The pairs on `n2_ccpvdz_cas44` (the `_df` case agrees to within 1%, with the
same populations and refusals):

| pair | class | tuples populated: u / v / both | max `rho` | K6, bilinear | polarisation | ladder | parallelogram defect |
|---|---|---|---|---|---|---|---|
| core x core2 | Sijrs | 1112 / 1112 / 1112 | 0 | 3.926843e-4 | 3.926843e-4 | 3.926843e-4 | −6.0e-19 |
| core x virt | Sijrs | 1112 / 7860 / 1112 | 0 | 1.196800e-4 | 1.196800e-4 | 1.196800e-4 | −2.5e-19 |
| core x virt | the five H classes `virt` populates | 0 / all / 0 | 0 | 0 | 0 | ≤5.6e-16 | 0 |
| virt x virt2 | Sijrs | 7860 / 7860 / 7860 | 0 | −1.093395e-4 | −1.093395e-4 | −1.093395e-4 | −3.0e-20 |
| virt x virt2 | Sr | 11 / 11 / 11 | 3.9e-2 | refused | 1.740044e-4 | 1.740044e-4 | 3.0e-7 |
| virt x virt2 | Sijr | 165 / 165 / 165 | 1.2e-1 | refused | 1.971512e-4 | 1.971512e-4 | −2.1e-8 |
| virt x virt2 | Srsi | 510 / 510 / 510 | 7.2e-2 | refused | 9.348625e-4 | 9.348625e-4 | −5.7e-7 |
| virt x virt2 | Srs | 176 / 176 / 176 | 8.0e-2 | refused | 2.987551e-4 | 2.987551e-4 | −7.1e-8 |
| virt x virt2 | Sir | 40 / 40 / 40 | 1.5e-1 | refused | −9.638299e-5 | −9.638299e-5 | −5.7e-6 |
| virt x virt+sym | the six classes `virt` populates (all but Si) | all / all / all | 0 | = polarisation | e.g. Sir −3.875557e-3 | = polarisation | ≤2.9e-20 |
| virt x 2virt | the same six | all / all / all | ≤2.3e-15 | = polarisation | e.g. Sir −7.751114e-3 | = polarisation | ≤1.9e-18 |

CAS(8,8) shows the same pattern. Only Sijrs is populated by core × core
and by core × virtual. On virtual × virtual, Sijr's `rho` is 5.4e-2 and
Srsi's 2.2e-2. LiF's core orbitals populate nothing (they share one irrep). On
virtual × virtual six of LiF's seven classes with empty tuples are populated (all but Si),
with H-class `rho` 2.1e-2 – 2.2e-1. Acrolein's empty tuples are Sijrs's
alone, so every acrolein pair is bilinear.

The separations, worst over the six goldens:

| quantity | noise (the side that must pass) | real (the side that must be caught) |
|---|---|---|
| an empty H-class tuple's first-order population `n[w, w]`, unit directions | ≤5.0e-33 (a symmetric direction's) | ≥1.0e-10 (LiF; ≥4.5e-8 on N2) |
| `rho`, H classes, tuples both directions populate | ≤2.3e-12 (the two proportional controls) | ≥2.1e-2 (random virtual pairs) |
| \|ladder − polarisation\| | ≤3.7e-12 | — |
| the largest integral of a guarded tuple | ≤2.7e-15 (empty: symmetry noise) | ≥2.6e-9 (small: acrolein) |

So the host's `RHO_TOL = 1e-7` sits more than four decades from either
side. The device's own separations, which #190's thresholds are taken from,
are in "The orbital block's sandwich".

**First order.** The guard drops this much from `Edot` along the probe's
core and virtual directions. Sijrs's `-Ndot/Delta`, summed over its guarded
tuples, is ≤1.4e-17 on every N2 and LiF case and 1.3e-12 / 2.5e-13 on
acrolein, where its 640 small tuples carry the whole of it. The H classes'
largest `|Ndot|` on a guarded tuple is ≤5.8e-19 on N2 and LiF and 1.7e-11
on acrolein (its small Sijr tuples). Each of these is far below every
first-order gate.

**What this meant for #190.** On N2 CAS(4,4), conventional and DF, the
core × core and core × virtual pairs are Sijrs-only, and K6 closes them. Every
random virtual × virtual pair, and `full x full`, is refused under §5.6's
test, because no Hessian exists along them. The proportional H-class path is
exercised by a pair like `virt x virt+sym`; the device runs `virtual x (2
virtual + core)`. The device's results are in "The orbital block's sandwich".

The pytest hand cases are `reference_data/tests/test_sc_empty_perturber_probe.py`
("The oracle's own tests").

## The active-rotation source term

`nevpt2_sc_pseudodensity_kappa_cas44`, `_cas88` and `_cas1010` (`gpu`;
CAS(4,4) also `small`) run `nevpt2_sc_pseudodensity --golden <case> --tiles 3
--kappa-act` and match `PASS: the active-rotation source term's adjoint and
tangent agree`. What they check is the **active-active block of Park Eq. 26's
orbital source term** `Y_xy = dE/dkappa_xy`, taken two ways out of the same
slab walk and held against itself:

| | |
|---|---|
| **A**, the adjoint | `<kappa, Y_act> = sum_blocks <D, Xdot> + sum_G <M_G, Gdot>` — the pseudodensities the entries above check, paired with the direction's action on the very arrays they are conjugate to |
| **B**, the tangent | `Edot(kappa)`: the same class terms rerun under the product rule, finished with Eqs. 36-37's multipliers |
| **C**, one finite difference | a central difference of `energiesDevice` along `exp(h kappa)` — every stored integral block rotated on the host, the RDMs and digests untouched |

`A == B` is the gate, at this tree's usual 1e-7, and it is **exact
arithmetic, not a step size**. `C` is there because A and B *share* two
conventions — the device kappa action, and "a Dyall block's tangent is its
own `make_*` called with the dotted integrals" — and a wrong tangent of an
operand cancels between the two modes; the finite difference is what pins both
to `nevpt2.kappa_rotation`'s exponential map. Same role, and the same
plateau-and-Richardson analysis, as `--sc-tangent`'s above — with a
three-halving run rather than four (`kRichardsonRun` in
`apps/sc_pseudodensity/kappa_check.cppm`), the same `[3.2, 4.8]` window.

**What is differentiated, and the two pieces that are missing.** The
functional is the SC total as a function of the MO **integrals** — every array
carrying an active index rotated — with `dm1`/`dm2`/`dm3`, the `f3ac`/`f3ca`
digests and the orbital energies all held fixed. The RDMs being fixed is the
derivative's own definition (they are functions of the CI vector). The other
two are this functional's limits: the digests carry an `h2e` dependence whose
reverse mode needs the 4-RDM contraction's transpose, and the orbital energies
respond through the active density inside the generalized Fock matrix, which
needs the `(xa|cc)` and `(xa|vv)` blocks the golden does not carry
([`gradient-theory.md`](gradient-theory.md), §1.3). `gradient.cppm`, "The
active-rotation source term", says it where the code is. **The second of the
two is lifted by the next section's entries**, which read those blocks out of
the gradient sidecar; this one's functional is unchanged, so that its
comparison against them stays exact.

Measured on an RTX 3080 (CUDA, Release, `--tiles 3`), four seeded
antisymmetric directions of unit Frobenius norm per case:

| case | `<kappa, Y_act>`, direction 0 | `max \|Y_act\|` | A vs B, worst of four | matrix vs pairing | FD plateau `\|err\|` (relative) | fourfold halvings |
|---|---|---|---|---|---|---|
| CAS(4,4) | 2.0e-16 — zero by symmetry, below | 4.0e-16 | 1.5e-31 | 2.6e-28 | 1.8e-12 (7.7e-12) | 7 |
| CAS(8,8) | 8.69193610221149e-03 | 1.5e-1 | 2.9e-15 | 4.4e-15 | 6.3e-11 (2.9e-10) | 8 |
| CAS(10,10) | 2.96982227227210e-02 | 1.3e-1 | 2.8e-15 | 3.0e-15 | 3.2e-11 (1.7e-10) | 7 |

The adjoint and the tangent agree at **machine precision** — 1e-15 against a
1e-7 gate — and the finite difference lands 1e-10 to 1e-11 away from both,
with the ratio 4.00 to two decimals through the truncation regime. The whole
entry takes <1 s / 2 s / 30 s at the three sizes (of which the matrix is
0.01 s / 1.6 s / 15 s and the 28-point sweep most of the rest), and the pool
high-water is 0.7 MB / 60.8 MB at the first two sizes; at CAS(10,10) it is the
RDM build's, which the entry runs first and which dominates it — the
CAS(10,10) `--tiles 3` row of docs/performance.md, "The memory pool".

**CAS(4,4) asserts a nontrivial zero.** N2's four active orbitals there are
its π system, and in D2h they lie in four *distinct* irreps, so every
active-active rotation mixes two of them: the energy is invariant under the
group, a scalar has no component along a non-symmetric generator, and the
whole source term vanishes identically — which is what both halves of A, B
and the matrix report (`max |Y_act| = 4.0e-16`), and what C confirms from the
unmodified energy stage (its differences fall as `h^2` towards zero and reach
1.8e-12). CAS(8,8) and CAS(10,10) have same-irrep pairs in the active space
and a source term of order 1e-1, so they are where the identity has content.

Because of that zero, the finite difference's error is judged against
`max(|Edot|, |E_corr|)` and not against `|Edot|` — which is also the right
scale on its own terms: differencing `E(+h) - E(-h)` carries round-off of
order `eps |E| / h`, so the floor is set by the energy's magnitude, not the
derivative's.

**Y_act as a matrix.** `scActiveRotationMatrix` probes the same linear
functional with the `n_act (n_act - 1) / 2` basis directions `E_pq - E_qp`
(in chunks of 16, one slab walk per chunk) and the entry checks its pairing
with each seeded kappa back against A. That is the entry point the
core/virtual extension compares against, and probing is what it costs:
the gauge invariance of the class energies under a *covariant* rotation of
integrals and RDMs together makes the directly-contractible part (`<D, X>`
and `<M_G, G>` with one active index left open) exactly symmetric, hence
invisible to an antisymmetric kappa, so all of `Y_act`'s content sits in the
chain rule through the Dyall blocks — a map *into* the integrals, which
reverse mode does not run backwards for free. Above 64 basis directions the
matrix is skipped (CAS(10,10)'s 45 are in, CAS(12,12)'s 66 are not); the
per-direction pairing is unaffected and costs nothing extra.

**Only CAS(4,4) is `small`,** unlike the pseudodensity-identity entries above.
The device work the sanitizer tier is there to watch — the kappa action, the
per-direction pairings, the product rule — is the same code at every size,
and the CAS(8,8) entry costs **103 s under memcheck and 34 s under
initcheck** against CAS(4,4)'s 1.4 s and 1.2 s (the 28-point sweep and the 28
basis probes are each a full pass over everything) — about half the memcheck
preset's whole runtime for no path CAS(4,4) does not already cover. It was
run clean under both before being dropped from `small`; "Wall times" below is
the standard the tier is kept to.

The host half — `exp(kappa)` orthogonal to 1e-14, and a rotation by
`kappa = 0` leaving every block **bit**-identical (compared as bit patterns,
with a `-0.0` and a denormal in the data) — is `test/gradient/`, card-free:
`AntisymmetricKappaTests`, `ExpKappaTests`, `RotateAxesTests`,
`RotateByZeroTests` and `KappaRotationDeathTest`.

## The full orbital source term

`nevpt2_sc_pseudodensity_kappa_full_n2_ccpvdz_cas44`, `_n2_ccpvdz_cas88` and
`_lif_def2svp_cas46` (`gpu`) run `nevpt2_sc_pseudodensity --golden <case>
--tiles 3 --kappa-full` and match `PASS: the full orbital source term's
adjoint and tangent agree`. They are the section above over the **whole**
`kappa` space — the core-core, virtual-virtual and mixed blocks of Park
Eq. 26's `Y_xy` as well as the active-active one — reading the **gradient
sidecar** beside each golden ("The gradient sidecar" in
[`reference-data.md`](reference-data.md)) for the one-general-index integral
blocks and the full-MO-range `h`, `h^eff` and generalized Fock that a rotation
leaving the active space needs. Only a case with a sidecar can run it, which
is why the three are these three.

**What moves now.** The functional is
the SC total as a function of every stored MO-integral array *and* of the
orbital-energy vectors `e_core` / `e_virt`, with `dm1`/`dm2`/`dm3` fixed (the
derivative's own definition). The orbital energies were one of the two gaps
the active block left: they are the generalized Fock's diagonals, and `f` plus
the inventory's restricted rows is exactly what their response needs
([`gradient-theory.md`](gradient-theory.md), §3.1's second structural fact).
What the integral halves still hold fixed is the `f3ac`/`f3ca` digests' own `h2e` dependence —
present on the blocks `h2e` moves under (active-active and the two mixed
blocks with an active index) and **absent by construction from the core-core
and virtual-virtual ones**, where no `h2e` index and no orbital energy moves
at all. So the two blocks Park Eqs. 56-57 read are an exact derivative.
`gradient.cppm`, "The full source term Y_xy", says it where the code is, and
records that the forward route to the digests (they are linear in `h2e`, so
`buildRdmsDevice` at the dotted `h2e` *is* their tangent) costs one digest
rebuild per direction, and that reverse mode needs that contraction's
transpose. #88 built the forward route and #89 the transpose: the same
entries now gate a sixth check on the digests' term and, beside the integral
halves' duality, the **full** one — `<kappa, Y>` with the transpose's term
against the exact `Edot` — "The digests' orbital response, forward mode" and
"… reverse mode" below have their numbers; the table here is the five checks
the duality half was measured with.

**Five checks, and the gate is their conjunction.** Measured on an RTX 3080
(CUDA, Release, `--tiles 3`):

| | CAS(4,4) | CAS(8,8) | LiF CAS(4,6) |
|---|---|---|---|
| **1** the promotion table against the arrays it reads, worst of four families | 3.5e-14 | 2.8e-13 | 6.1e-15 |
| **2** `<kappa, Y>`'s integral half vs the tangent's, worst of four seeded FULL directions | **3.5e-17** | **5.1e-16** | **6.9e-16** |
| `<kappa, Y>`, direction 0 | 2.538664648720e-03 | 1.508419492916e-02 | 4.643210528103e-02 |
| of which the orbital energies' response | −6.687e-04 | −7.060e-04 | 1.781e-03 |
| **3** the active block vs `--kappa-act`'s own entry point (vs what that entry point reproduces from itself) | 5.5e-28 (1.3e-28) | 2.2e-15 (5.3e-16) | 5.2e-15 (3.8e-15) |
| **4a** core-internal finite difference: plateau, fourfold halvings | 4.4e-12, 8 | 1.3e-12, 8 | 3.0e-12, 6 |
| **4b** virtual-internal: plateau, fourfold halvings | 1.8e-12, 6 | 1.2e-11, 5 | 1.0e-11, 6 |
| the same two directions' `eps` term, zero by (O2) | ≤3.1e-17 | ≤2.5e-17 | ≤2.7e-17 |
| the probed matrix's pairing vs the direct one, inactive-restricted | 2.6e-18 | 2.1e-17 | 6.5e-19 |
| **5** Eqs. 56-57: pairs divided / left zero as degenerate | 175 / 6 | 133 / 6 | 81 / 3 |
| max \|z^c\|, at the smallest gap divided by | 5.91e-03, 3.310e-03 Eh | 3.52e-03, 3.305e-03 Eh | 5.31e-03, 3.566e-04 Eh |
| `Y`'s two inactive blocks: basis directions probed, wall | 181, 0.45 s | 139, 7.74 s | 84, 0.63 s |
| the whole entry, wall (`ctest`, Release) | 1.11 s | 14.04 s | 1.75 s |

One run of each (the three entries, in that order, on one card). The last
digits of a pairing and the exact height of a finite-difference plateau are
the kernels' `atomicAdd` order and the round-off floor respectively, so they
move run to run — these are the values one run gave; the orders of magnitude,
the halving counts and the ratios do not move.

Check 2 is the gate, at this tree's 1e-7 against a measured 1e-16 — **nine
decades**, and exact arithmetic rather than a step size.

**The new blocks are where the content is, and CAS(4,4) shows it cleanly.**
Its active-active block is a symmetry zero of order 1e-16 (four active
orbitals in four distinct D2h irreps — "The active-rotation source term"
above), while a full direction pairs to 2.5e-03 and the inactive blocks of
`Y` reach 7.6e-03: at that case the source term is *entirely* core, virtual
and mixed. CAS(8,8) and LiF have both halves of order 1e-2.

The rest of the tier is what makes check 2 mean something:

- **Check 1 is what pins the promotion table**, which neither mode of check 2
  can see: both read the same operand tangents, so a wrong one cancels between
  them (the same reason the active block needs its finite difference). Every
  promotion rule's source array, restricted to the promoted slot's own space,
  must be the stored block under exactly the permutation that rule uses — four
  identities per two-electron block, over all of them, which pins the array
  choice and the index order; `h^eff` must come back out of `h` and the
  dressing rules' own Coulomb and exchange sources; and the generalized Fock's
  core and virtual diagonals must come back out of `h^eff` and the
  orbital-energy rules' own tied arrays, *and* be the golden's `e_core` /
  `e_virt`, which is (O2) ([`gradient-theory.md`](gradient-theory.md), §3.6)
  seen from this side and is what says the sidecar belongs to this golden.
  The residuals are two PySCF transforms of the same integrals compared, so
  1e-13 is reassociation.
- **Check 3 is "bit for bit" in the only form this tree has.** The
  construction is exact: the general part of every slot's sum is an
  *exactly*-zero tensor for an active-restricted kappa, and adding zero to a
  double changes no bit. But the two numbers come out of two separate runs of
  kernels that accumulate with `atomicAdd`, whose order the hardware picks,
  and that is measurably not reproducible: running `--kappa-act`'s own entry
  point **twice on the same input** moves its answer by 1.3e-28 at CAS(4,4),
  5.3e-16 at CAS(8,8) and 3.8e-15 at LiF, and by a different amount on the
  next run. So the two paths are held to the LARGER of that same-run spread
  and a relative 1e-11 of the pairing itself — the spread alone is a coin flip
  between two draws of one noise, while anything a wrong construction could do
  is many decades above either. Measured, the two paths differ by 5.5e-28,
  2.2e-15 and 5.2e-15: the same size as the kernels' own spread, which is the
  statement. (CAS(4,4)'s pairing is a symmetry zero of order 1e-16, which is
  why both numbers there are 1e-28; CAS(8,8)'s is 2.1e-2 and LiF's −4.0e-2,
  where 1e-15 is a relative 7e-14.) The spread itself is held to a recorded
  envelope, measured over 90 runs — "The kernels' same-run self-variation"
  below.
- **Check 4 is the only finite difference the new blocks can have, and it
  covers them.** A rotation *inside* core, or inside virtual, keeps every
  index in its own space, so the displaced integrals are the stored blocks
  with those axes transformed on the host — and the displaced orbital
  energies are the rotated generalized Fock's diagonal, which the sidecar
  carries. So the unmodified energy stage can be rerun at `exp(h kappa)` and
  differenced, with the same plateau-and-Richardson reading as the two
  sections above (ratio 4.00 to two decimals through the truncation regime,
  for five to eight consecutive halvings, down to a plateau of 1e-12 to
  1e-11). A **mixed** rotation cannot be
  displaced this way — its indices leave their spaces, which would need
  integrals with two general indices — and does not need to be: each
  promotion is ONE einsum over the whole general index, so the rows these two
  exercise are the same code the mixed rows run, and what is left for the
  mixed rows alone is which rows of `kappa` are non-zero. The two directions
  also assert a derived exactness: an intra-space rotation moves no orbital
  energy — `f` is diagonal there by (O2) and the promoted densities have no
  column outside their own space — so their `eps` term must vanish, and it
  does to 3e-17. It is held to 1e-9 rather than to machine precision because
  it inherits (O2)'s own residual, which is the canonicalisation's 1e-13 and
  not exact ([`reference-data.md`](reference-data.md), "The gradient
  sidecar").
- **Check 5 exercises Park Eqs. 56-57 on a case with distinct core and
  virtual orbital energies.** LiF is the well-separated one: all **six** of
  its core pairs are divided through, its four core energies being distinct
  with a smallest gap of 0.849 Eh, where N2's core carries a 3.3e-3 Eh 1s pair
  (the smallest gap either N2 case divides by). On the virtual side LiF
  divides 75 of its 78 pairs, the tightest at 3.57e-4 Eh — a π/σ splitting,
  above the golden generator's 1e-4 refusal and where its largest multiplier
  comes from. The three it does not, and the six each N2 case does not, are
  **exactly degenerate** pairs, and they are the other half of the same test:
  an abelian subgroup puts the two partners of a degenerate set in different
  irreps, so the energy does not move along that rotation at all, the
  numerator vanishes with the gap, and the multiplier is a 0/0 the constraint
  leaves free — left zero rather than divided through. A degenerate pair whose
  numerator did **not** vanish would be a numerical refusal, not a large
  number. The host arithmetic itself, including that refusal, is
  `test/gradient/`'s `ScPseudocanonicalTests`, card-free.

**The core-internal finite difference is what caught the one real bug in
this,** and it is worth recording what it was: `h^eff`'s core dressing has its
two core labels tied, so differentiating promotes each in turn, and while the
Coulomb pair's two promotions collapse (`(pq|xj)` is symmetric in `x` and `j`),
the exchange pair's do **not** — `(px|jq)` and `(pj|xq)` are different
integrals. Writing the exchange as twice either one leaves a term that
survives even a core-internal rotation, where the true dressing response is
zero: the difference quotient sat 6.7e-04 away from the analytic derivative
and flatlined instead of falling as `h^2`. Nothing else in the tier saw it —
not the duality (both modes share the tangent), not the restriction identities
(a derivative-only term has no stored block to be the restriction of).

**So it is pinned, card-free.** The dressing's three terms are
`nevpt2.gradient_plan`'s `heffDressingPlan`, which the device path launches
term for term, and `test/gradient_plan`'s `HeffDressingTests` (no `gpu`
label) evaluates that same plan on the host over the committed CAS(4,4)
sidecar under a core-internal rotation. Per one-electron block, the
dressing's response must vanish — the Coulomb term on its own, the exchange
pair between its two members — while each exchange member alone must clear a
1e-6 floor and the two must differ by as much. Measured: the residuals are at
most 3.0e-15 (bound 1e-13), and each exchange member alone is 1.0e-3 on
`h1e`, 3.5e-2 on `h1e_v_Sr` and 0.33 on `h1e_v_Sir`. `h1e_v_Si`'s exchange
integrals are `(core act | core core)`, an exact D2h zero at that case (an
all-σ core against π actives), so that block is held to its symmetry zero
and the other three carry the non-collapse. A plan that writes the pair as
twice either member leaves `2 |exchange|` behind, which is what the
6.7e-04 quotient was.

**Not `small`, but run under every tool anyway.** The two 13-point sweeps and
the ~180 basis probes are each a full pass over everything, which is what made
the CAS(8,8) `--kappa-act` entry too slow for the sanitizer tier, and
CAS(4,4)'s `--kappa-act` entry already covers this code's device work there.
Since the sanitizer test presets filter on `small`, the three entries were run
**by name** under `asan`, `ubsan`, memcheck (with the leak check) and
initcheck instead — `ctest --test-dir build-<preset> -R
nevpt2_sc_pseudodensity_kappa_full` — and are clean under all four:

| seconds per entry | asan | ubsan | memcheck | initcheck |
|---|---:|---:|---:|---:|
| CAS(4,4) | 3 | 3 | 10.3 | 7 |
| CAS(8,8) | 15 | 15 | 349.7 | 114 |
| LiF CAS(4,6) | 3.0 | 2.7 | 31.2 | 12.2 |

The CAS(8,8) column is the whole argument for leaving them out of `small`:
350 s under memcheck alone is more than that preset's entire current runtime
(237 s for all 148 of its entries). Worth noting what the four runs caught:
the first version of the Fock-diagonal self-check took a view of a
*temporary* device tensor, so the pool could hand the memory to the very
einsum reading it. The self-check's own number gave it away (2.4e-01 where a
reassociation residual belongs), but it is exactly the class of bug memcheck
exists for, and the reason these entries are worth the four extra runs.

### The kernels' same-run self-variation

Check 3 above is held to the larger of a relative 1e-11 and what
`--kappa-act`'s entry point (`scActiveRotationDevice`) moves by when it is
simply run twice on the same input in the same process. That spread is
measured by the run it gates, so on its own it would widen silently if a
kernel change made the `atomicAdd` nondeterminism worse. So the spread is
also held to a **recorded envelope**: each `--kappa-full` entry fails if
either rerun gap — the pairing `<kappa, Y_act>` or the tangent
`Edot(kappa)`, direction 0, absolute, in Eh — exceeds it, and prints the
comparison as `same-run self-variation: ... against the recorded envelope`.

Measured on an RTX 3080 (CUDA, Release, `nevpt2_sc_pseudodensity --golden
<case> --tiles 3 --kappa-full`), **30 runs per case, 90 in all**, one after
another on one card, every one of them a PASS:

| | CAS(4,4) | CAS(8,8) | LiF CAS(4,6) |
|---|---|---|---|
| rerun gap, pairing: min / median / max | 3.5e-30 / 5.2e-28 / 1.4e-27 | 1.4e-17 / 1.3e-15 / 5.6e-15 | 1.6e-16 / 6.2e-15 / 2.2e-14 |
| rerun gap, tangent: min / median / max | 4.8e-30 / 5.2e-28 / 1.4e-27 | 8.0e-17 / 1.3e-15 / 5.7e-15 | 1.4e-17 / 6.6e-15 / 2.2e-14 |
| check 3's `full - act`, pairing: max | 1.1e-27 | 5.1e-15 | 1.8e-14 |
| runs in which a rerun gap was exactly zero | 0 of 30 | 0 of 30 | 0 of 30 |

The one-run figures in the table above (1.3e-28, 5.3e-16, 3.8e-15) are draws
from these distributions, not their edges.

| | |
|---|---|
| **the recorded envelope** (absolute, Eh; pairing and tangent alike; all three cases) | **5e-13** |

**How it was chosen.** The largest gap in all 90 runs is LiF's 2.2e-14; its
max-to-median ratio is 3.6, CAS(8,8)'s 4.2 and CAS(4,4)'s 2.7, so the tail is
light. 5e-13 is **22x** above that largest gap and about 80x above LiF's
median — enough that a draw from the distribution measured here does not
fail the entry — and it is the size of check 3's own relative floor at LiF
(1e-11 of a 4.0e-2 pairing), so the measured spread can no longer push
check 3's tolerance much past the floor it already has. What it is there to
catch is a change of kind, not of luck: a reduction that drops to single
precision, or a race, lands decades above it. CAS(4,4)'s symmetry-zero
pairing sits fourteen decades under the envelope, so it is CAS(8,8) and LiF
that give the gate its teeth.

**Where the number lives.** This table row and nowhere else:
`apps/sc_pseudodensity/CMakeLists.txt` reads it at configure time (exactly
one `| **the recorded envelope** ... | **<x>e-<n>** |` row, or the configure
fails) and hands it to `kappa_full_check.cppm` as a compile definition, so
editing the row re-runs the configure and changes the gate. With the row
set to 1e-18 in a scratch build — below every CAS(8,8) and LiF gap above —
the CAS(8,8) and LiF entries fail on that line (`FAILED -- the kernels'
nondeterminism has grown`) and CAS(4,4)'s, whose gaps are 1e-27, still
passes: the gate fires, and on the cases that give it its teeth.

**#89's transpose kernels do not touch it.** They are one thread per output
element with no atomic (`src/rdm/f3_transpose.cu`), so they add no run-to-run
variation of their own, and the digests' term they feed is kept out of the
integral halves check 3 compares. The envelope was therefore neither
re-measured nor widened; every `--kappa-full` run since, on both cards, stayed
inside it ("The digests' orbital response, reverse mode").

## The PC tangent against a finite difference

The PC side's own entries: `nevpt2_pc_tangent_cas44`, `_cas88` and `_cas1012`,
all `gpu`, run `nevpt2_pc_tangent` (`apps/pc_tangent/`) and match
`PASS: the PC-NEVPT2 tangent matches the central finite difference`.

A finite difference again, as in the two SC sections above, and here with no
exact check to lean on at all: `pcTangentsDevice` differentiates a *solve*
(`E_t = -c b_t^T (K + Delta_t S)^+ b_t` with respect to the CI vector). The SC
tangent has its norms' affine increment and the active-rotation term has
`A == B`; a solve's derivative has no algebraic identity short of
differentiating the eigendecomposition, which is exactly what that code
avoids. So each entry displaces the state and re-solves:

| check | against | tolerance |
|---|---|---|
| each class energy, from the tangent path's own `S` and `K` | `pcEnergiesDevice`'s answer for the same state | 1e-9 |
| `Edot(c, c)`, five classes (`Sr`, `Si`, `Srsi`, `Srs`, `Sir`) | `2 E`, exact: those classes are homogeneous of degree 2 in the CI vector | 1e-9, scaled by `max(1, \|2E\|)` — so absolute at these class energies |
| the central difference's error at each `h` | 4× the next one's — second order | 4 ± 0.5 |
| those ratios as a ladder | a **run** of at least two fourfold halvings (`kPcMinFourfoldRun`), so three steps at least: no steps, or one halving, FAILS | — |
| every displaced state `c ± h u` | the PC gap check's own record of that state (the smallest `S` gap over the classes), at least `PC_MIN_GAP`; a state with no record FAILS | 1e3 |
| the Richardson extrapolation of the two finest steps | the analytic `Edot` | 1e-8 |
| a class refused by the gap or denominator check | nothing: the entry FAILS, and no derivative is reported for it | — |

**Five Euler classes here, against the SC tangent's three** (`Sr`, `Srsi`,
`Srs`, above) — the two are consistent, and the difference is what PC does
*not* contract. SC's `Si` reads `a25`, whose `2 h1e^T` is RDM-free, and SC's
`Sir` norm has a purely one-electron term; both make those class energies
affine in the RDMs rather than homogeneous. PC uses neither: its one-body
part enters only through `b_t = S x_t + S_1 h1_t`, and `S_1` is linear in the
RDMs, so PC's `Si` and `Sir` are homogeneous and Euler's identity holds for
them. `Sijr` and `Sij` (the hole-side `delta` terms) and `Sijrs` (no RDM at
all) are affine or degree 0 in both.

Each displaced state costs a host `dm1`/`dm2` rebuild plus a full device RDM
build and PC stage — the same price the SC tangent's sweep pays, at 8
displaced states rather than 28 but up to 12 orbitals, which makes
`nevpt2_pc_tangent_cas1012` the slowest entry in the tree; it is `slow` and
runs three steps instead of four. Its wall time, the measured ratios and
residuals, the card and flags they were taken with, the step at which
round-off takes over (an `--fd-start-pow` sweep past the entries' own steps)
and the three findings that
constrain the direction and the step — a random CI direction raises the
metric's rank and gets the displaced state refused at *any* `h`; the path has
to be the unit sphere; one `Sir` mode forces `h <= 2^-10` — all live in
[`pc-nevpt2.md`](pc-nevpt2.md), "The tangent with respect to the CI vector",
with the rest of PC's numbers, and are not copied here.

CAS(4,4) and CAS(8,8) are `small`, so the sanitizer tier covers the tangent's
einsums, GEMMs and second solve. The host halves are card-free: the
per-tuple arithmetic is `test/energy_finish/` (`pcCoefficients`,
`pcQuadForm`, `tangentSingle`) and the gate itself `test/pc_tangent/`, which
locks in that a refused class fails rather than reading as a zero
derivative, and (`PcTangentLadderTests`) that the difference is gated on a
run and that a displaced state with no conditioning record fails — the two
rows added for "Residual-independent checks of the gradient path" below.

## The PC Hessian against a finite difference of the three-term tangent

`nevpt2_pc_hessian_cas44` (`gpu;small`) runs `nevpt2_pc_tangent --golden
golden/n2_ccpvdz_cas44.nevpt2gold --tiles 3 --pc-hessian` and matches
`PASS: the PC-NEVPT2 Hessian Eddot(u, v) matches Euler` (#152). It checks
`pcHessianDevice`, the PC energy's second directional derivative at the
golden's **fixed** orbital energies (the PASS line says so, as
[`hessian-theory.md`](hessian-theory.md), §9.1 asks), with §6.2's rotation
terms: CAS(4,4)'s Sr null space is a rank bound that turns with the state, the
case those terms exist for.

| check | against | tolerance |
|---|---|---|
| each class energy from the Hessian walk's own `S` and `K` | `pcEnergiesDevice` | 1e-9 |
| its three-term `Edot(u)`, `Edot(v)` | `pcTangentsDevice`'s two-term one: equal at the eigenstate, to the state's convergence | 1e-8, relative, floor 1 |
| Euler `Eddot(c, u)`, five classes | `Edot(u)`, exact for a degree-2 class | 1e-9, relative, floor 1 |
| the negative control `(u, w)`, `w` a random amplitude vector | at least one class refused for its lift, none for any other reason | — |
| the central difference of the **three-term** `Edot(v)` along `c(h)` | a run of at least two fourfold halvings, 4 ± 0.5, every displaced state's gap recorded and ≥ `PC_MIN_GAP` | — |
| its Richardson extrapolation | `Eddot(u, v)` | 1e-8 |
| a class refused by the gap, the denominator or the lift test along `(u, v)` | nothing: the entry FAILS, and no second derivative is reported for it | — |

The difference is of the three-term `Edot`, not of `pcTangentsDevice`'s.
Off the eigenstate the two-term one lacks a term, and a difference of it
converges to something else; #152's first ladder plateaued for exactly that
reason (`hessian-theory.md`, §6.2). The host halves are card-free.
`test/energy_finish/`'s `PcQuadSecondTests` hold the per-tuple formula
against an independent rank-bound Gram metric whose null space turns,
re-solved by a full eigendecomposition at every point and differenced. The
same suite shows §6.1's terms alone missing there. `test/pc_tangent/`'s
`PcHessianReportTests` lock in that a refused class, a control that answers,
or a broken ladder fails. The measured numbers, and why there is no CAS(8,8)
entry, are in [`pc-nevpt2.md`](pc-nevpt2.md), "The second derivative,
Eddot(u, v)".

## The PC Hessian-vector product's checks

`nevpt2_pc_hessian_vector_cas44` (`gpu;small`) runs `nevpt2_pc_tangent
--golden golden/n2_ccpvdz_cas44.nevpt2gold --tiles 3 --pc-hessian-vector` and
matches `PASS: the PC-NEVPT2 Hessian-vector product H_cc u matches` (#155).
It checks `nevpt2.gradient`'s `pcCiHessianVectorDevice`, PC's `H_cc u` by
reverse over forward ([`hessian-theory.md`](hessian-theory.md), §4.2 and
§6.2), per class, at the golden's **fixed** orbital energies (the PASS line
says so). `u` is `--pc-hessian`'s, and so is the first `v`.

| check | against | tolerance |
|---|---|---|
| the walk's own three-term `Edot(u)` per class, and `<u, y>` of its own `y` | `pcHessianDevice`'s three-term `Edot(u)` | 1e-9, relative, floor 1 |
| `<v, H_cc u>` per class and summed, three seeded `v` and the state | `pcHessianDevice`'s `Eddot(u, v)` (the entry above's), exact | 1e-9, relative, floor 1 |
| `<v, H_cc u>` | `<u, H_cc v>`, a second reverse walk | 1e-9, relative, floor 1 |
| Euler `H_cc c`, five classes | that class's `y`, exact for a degree-2 class | 1e-9 relative to `‖y‖` |
| the negative control `H_cc w`, `w` a random amplitude vector | at least one class refused for its lift, none for any other reason | — |
| the central difference of each class's **three-term** `y` along `c(h)` | for every class with a nonzero `H_cc u` and for the total: every ratio 4 ± 0.5 and a run of at least two, every displaced state's gap recorded and ≥ `PC_MIN_GAP` | — |
| its Richardson extrapolation, per class | that class's `H_cc u` | 1e-8, relative to its norm |
| a class with `H_cc u = 0` exactly (Sijrs; Si on this golden) | its difference must be exactly 0 too | 0 |
| a class refused along `u` (gap, denominator, or lift: `|P S_u P|`, `L_uu`) | nothing: the entry FAILS and no vector is formed | — |

The one finite difference is of the GPU's own exact `y`, never a second
difference of the energy ("The PC CI source term's step is a first-order
trap", below). It is the three-term `y` (`pcCiSourceThreeTermDevice`), because
the two-term one is the gradient only at the eigenstate and the difference
visits displaced states. It is the only check here that sees the rotation
terms: the duality would still pass if both `H_cc u` and `pcHessianDevice`
dropped them. So the gate is per class, and a plateau in any class fails,
Sr's and Sir's in particular. The measured residuals, and what happens with
either piece removed, are in [`pc-nevpt2.md`](pc-nevpt2.md), "The
Hessian-vector product, H_cc u". The same section says why there is no
CAS(8,8) entry.

The host halves need no card. `test/energy_finish/`'s `PcAdjointSecondTests`
hold the per-tuple bars against `pcQuadSecond` by host-level duality, both
pair layouts included. They also check that the third term's block completes
the two-term adjoint to the three-term `Edot`. `test/pc_tangent/`'s
`PcHessianVectorReportTests` lock in that a refusal, a duality, symmetry or
Euler miss, a control that answers, or a plateau in any one class fails. The
card halves are three of `test/pc_adjoint/`'s `PcAdjointRefusalTests`. The
first: on the refused displaced state, `H_cc u` and the three-term `y`
refuse what the energy refuses and form no vector. The second: along a
symmetry-breaking direction, `H_cc w` refuses exactly the classes
`pcHessianDevice` refuses along `(w, w)`. The third: off the eigenstate the
three-term `y` is the gradient of the three-term `Edot` and the two-term `y`
is not.

## The density-fitted orbital source term

#165 runs "The full orbital source term" above on the **density-fitted**
pair, and #168 completes it with the digests' own term:
`nevpt2_sc_pseudodensity_kappa_full_cas44_df` and `_cas88_df` (`gpu`) run
`nevpt2_sc_pseudodensity --golden golden/n2_ccpvdz_<case>_df.nevpt2gold
--tiles 3 --kappa-full` beside the committed NEVPT2F1 sidecar, and match
`PASS: the full orbital source term's adjoint and tangent agree to <tol> on
every direction (density-fitted: both Y terms`. The inventory is contracted
on the device out of the whole `B[L,p,q]` instead of read
([`gradients.md`](gradients.md), §6.2), and the checks are all six of the
section above, digests' term included. Check 1 gains one row, the built
inventory against the four tied arrays the sidecar carries ready-made; check
2 keeps #165's frozen-digest duality beside the full one; and the digests'
transpose carried onto the three-index `B` (`dfDigestSourceDevice`) is held
to the four-index pairing on the four directions, as one more exact zero on
the two intra-space ones, and to the forward term along check 6's active
direction.

Beside them, on `n2_ccpvdz_cas44_df`: the identities
(`nevpt2_sc_pseudodensity_cas44_df`, `small`) and `--kappa-act`
(`nevpt2_sc_pseudodensity_kappa_cas44_df`, `small`), which read a fitted
golden's four-index blocks exactly as any other; the dump,
`nevpt2_sc_pseudodensity_dump_cas44_df`, matching the SC dump's PASS line
with `(density-fitted: both Y terms` (its checks are the integral-direct
dump's four plus a sixth, each direction's `edot_digest` against the
three-index route); and two refusals whose regex is the *unsupported*
error: `_dump_pc_cas44_df_refused` (`--pc --dump-source-terms`) and
`_oracle_cas44_df_refused` (`--source-oracle`: no fitted oracle exists).
#165's `_dump_cas44_df_refused` is gone with the refusal it held.

Measured on an RTX 3080 (CUDA, Release, `--tiles 3`), one run each. #165's
rows first:

| | CAS(4,4) DF | CAS(8,8) DF |
|---|---|---|
| **1** the promotion table against the arrays it reads, worst of four families | 2.1e-14 | 5.2e-14 |
| the built inventory against the sidecar's four carried tied arrays | 3.2e-16 | 7.0e-16 |
| **2** `<kappa, Y>`'s integral half vs the tangent's, worst of four seeded FULL directions | **4.2e-17** | **4.5e-16** |
| the frozen-digest duality, `<kappa, Y>` vs `Edot` with the orbital energies moving | **4.2e-17** | **4.5e-16** |
| `<kappa, Y>`, direction 0 | 2.569305008166e-03 | 1.508999826491e-02 |
| of which the orbital energies' response | −6.726e-04 | −7.111e-04 |
| **3** the active block vs `--kappa-act`'s own entry point (vs what that entry point reproduces from itself) | 4.3e-30 (1.9e-28) | 5.6e-17 (5.2e-16) |
| **4a** core-internal finite difference: plateau, fourfold halvings | 2.9e-12, 8 | 6.3e-13, 7 |
| **4b** virtual-internal: plateau, fourfold halvings | 3.5e-13, 6 | 8.6e-12, 5 |
| the same two directions' `eps` term, zero by (O2) | ≤7.9e-17 | ≤1.6e-17 |
| the probed matrix's pairing vs the direct one, inactive-restricted | 3.0e-18 | 2.8e-18 |
| **5** Eqs. 56-57: pairs divided / left zero as degenerate | 175 / 6 | 133 / 6 |
| max \|z^c\|, at the smallest gap divided by | 5.92e-03, 3.309e-03 Eh | 3.52e-03, 3.305e-03 Eh |

and #168's, the digests' own term (2026-10-19, the `default` preset, 582 MiB
of the card held by other processes; `--kappa-full` and the dump each one
run):

| | CAS(4,4) DF | CAS(8,8) DF |
|---|---|---|
| **2** the FULL duality, `<kappa, Y>` with the transpose's term vs the exact `Edot`, worst of four seeded FULL directions | **4.60e-14** | **5.54e-15** |
| the digests' term, direction 0 (forward; reverse agrees to the full-duality gap) | −5.90084611341534e-03 | −1.87414214822353e-02 |
| the frozen-digest duality beside it, unchanged | 3.47e-17 | 5.13e-16 |
| the transpose onto the three-index `B`, `Σ κ·D`, vs `<hbar, h2e_dot>`, worst of the four | **6.94e-18** | **5.90e-17** |
| **4** the two intra-space directions' digests' term — forward, reverse, onto `B` — and `max\|h2e_dot\|` | exactly 0.0, both | exactly 0.0, both |
| **6** the digests' term along check 3's active direction | −1.469e-20 | −2.06462996961607e-02 |
| its central difference, `f3ac`/`f3ca` alone rebuilt from the rotated fitted `h2e`: plateau (relative), fourfold halvings | 2.3e-13 (9.7e-13), 7 | 3.4e-11 (1.6e-10), 9 |
| reverse vs forward along it / onto `B` vs forward | 1.4e-29 / 4.7e-17 | 3.1e-16 / 2.3e-16 |
| dump: `Σ κ·Y` off the matrix vs `<kappa, Y>` direct, one seeded direction | 2.05e-14 | 1.39e-14 |
| dump: the six directions' exact `Edot` (frozen + digest) vs `Σ κ·Y` off the written matrix | **9.14e-14** | **1.42e-14** |
| dump: the six directions' `edot_digest` vs `Σ κ·D` (check 6) | 7.24e-14 | 1.52e-14 |
| dump: `y`'s walk vs `energiesDevice`; `e_core`/`e_virt` rebuilt from the sidecar (relative) | 1.94e-16; 7.56e-15 | 1.39e-16; 1.83e-14 |
| dump: the file read back | bit for bit | bit for bit |

The gate is again seven-plus decades inside 1e-7, the integral-direct
twins' orders. N2 CAS(4,4)'s digests' term along check 3's active direction
is round-off on the fitted state as on the exact one (−1.3e-17 there, the
same run): that direction does not move the digests at CAS(4,4), so check 6's
finite difference is substantive at CAS(8,8), where the term is −2.06e-2 and
its difference converges onto it. The CAS(8,8) DF dump is not a ctest entry
(the integral-direct CAS(8,8) dump is; the fitted one was run by hand for
the table, 24.0 s). N2's exactly degenerate virtual pairs are left zero on
the fitted state as on the exact one (six each), and nothing was refused.
What the entries cost is [`performance.md`](performance.md), "The
density-fitted orbital source term".

**Sanitizers** (CUDA, RTX 3080, each preset's Debug build; #168's run,
2026-10-19). The two `small` entries ran in the tier, clean under `asan`,
`ubsan`, memcheck (with the leak check) and initcheck, every canary green:
`nevpt2_sc_pseudodensity_cas44_df` 0.48 / 0.37 / 0.61 / 0.56 s and
`nevpt2_sc_pseudodensity_kappa_cas44_df` 0.87 / 0.68 / 1.42 / 1.18 s in that
order. The two `--kappa-full` entries, the dump and the two refusals are not
in the tier (the integral-direct `--kappa-full` entries' reason) and were
run by name, clean under the same four (memcheck: 0 errors, 0 bytes leaked):
`_kappa_full_cas44_df` 4.24 / 3.55 / 11.83 / 8.39 s, `_kappa_full_cas88_df`
19.78 / 19.40 / 455.67 / 154.78 s, `_dump_cas44_df` 4.19 / 3.20 / 17.70 /
11.81 s, `_dump_pc_cas44_df_refused` 0.51 / 0.38 / 0.72 / 0.68 s,
`_oracle_cas44_df_refused` 0.47 / 0.35 / 0.57 / 0.56 s. (#165's run of its
entries, before the digests' term: `_kappa_full_cas88_df` 344.99 s under
memcheck, 115.50 s under initcheck.) Not run under racecheck or synccheck,
and nothing here on HIP.

## The CI source term's forward-reverse duality

`nevpt2_gradient_source_cas44`, `_cas88` and `_cas1010` (all `gpu`; the first
two `small`) run `nevpt2_demo --golden <case> --tiles 3 --gradient-source` and
match `PASS: the CI source term and the forward tangent agree … , <functional>`.
They **difference nothing**: this is the exact check the one-time finite
difference above was run to license.

**Which functional (#87).** CAS(4,4) and CAS(8,8) also pass
`--gradgold golden/n2_ccpvdz_<case>.gradgold`, so both modes carry the
orbital energies' response to the active density
([`gradients.md`](gradients.md), §5.5) and the regex requires `the orbital
energies moving with the active density`; CAS(10,10) has no committed sidecar
and requires `the orbital energies held fixed`. The duality is exact on either
functional — and therefore says nothing about whether that response is right,
since both modes read the same one: "The orbital energies' CI response" below
and "The GPU as the source-term provider" are what check it.

**Density-fitted (#162).** `nevpt2_df_gradient_source_cas44` and `_cas88`
(both `gpu;small`) run the same check in `nevpt2_df_demo --tiles 3
--gradient-source` over its `DfIntegralSource` (the default `--batch 8`, so
a ragged last slab), and `nevpt2_df_cas44_sc_tangent` (`gpu;small`) and
`nevpt2_df_cas88_sc_tangent` (`gpu`) the SC tangent's. They are the same
`nevpt2.ci_checks` bodies with a different `IntegralSource` and the
DF-rebuilt active `h2e`, so the same tolerances. The orbital energies are
held fixed (that demo reads no gradient sidecar), and the regex requires
`the orbital energies held fixed (frozen-orbital-energy mode`.

`nevpt2.gradient`'s `scCiSourceDevice` assembles the CI source term
`y_I = dE/dc_I` — Park Eq. 48's `<I|(O_D + O_D^dagger)|Psi>` — in reverse
mode: the pseudodensity walk of the section above leaves one accumulator
`M_G = dE/dG` per metric / Dyall-Hamiltonian block, a sweep through each
block's own einsum chain turns those into the five RDM coefficient tensors
(`dE/d(dm1)`, `dE/d(dm2)`, `dE/d(dm3)`, `dE/d(f3ac)`, `dE/d(f3ca)`), and
`nevpt2.sigma_vector`'s kernels contract those with the RDMs' own CI
derivatives. `energiesTangentDevice` computes the same derivative in
**forward** mode. Two independent computations of one number, so for every
direction `u`

```
  <u, y> == Edot(u)
```

with no step size in it — the only difference is that the two paths
reassociate their sums differently. The entry checks that for four seeded
directions (`|u| = 1`, `<c|u> = 0`), for the state itself (Euler, the one
direction the projected random ones never reach), and — where `ndet` allows —
for every column of the Jacobian.

Measured on an RTX 3080 (CUDA, Release, `--tiles 3`), in the `default`
preset's ctest run of #87:

| case | functional | n_det | norm of `y` | worst seeded `<u,y>` residual | the state itself | worst overall | relative |
|---|---|---:|---:|---:|---:|---:|---:|
| CAS(4,4) | orbital energies moving | 36 | 0.6581 | 6.4e-15 | 1.1e-12 | 2.2e-12 | 3.4e-12 |
| CAS(8,8) | orbital energies moving | 4900 | 1.3760 | 3.8e-16 | 6.7e-14 | 6.7e-14 | 4.9e-14 |
| CAS(10,10) | orbital energies fixed | 63504 | 0.4023 | 5.1e-17 | 2.0e-14 | 2.0e-14 | 5.4e-14 |

**The density-fitted CAS(4,4), `nevpt2_gradient_source_cas44_df` (#164).**
The same binary and flags on `golden/n2_ccpvdz_cas44_df.nevpt2gold`, with
the density-fitted sidecar `golden/n2_ccpvdz_cas44_df.gradgold` as
`--gradgold`. It is the end-to-end read of that sidecar's `eps`-response arrays
([`reference-data.md`](reference-data.md), "The density-fitted gradient
sidecar"). `nevpt2_demo` runs a `*_df` golden off the fitted four-index
arrays it also carries, so this is the fitted energy's `y`. It is not
`nevpt2_df_demo`'s slab path, which takes no `--gradient-source` until #162.
The loader first rebuilds the fitted golden's `e_core`/`e_virt` from the
sidecar to 7.56e-15 relative. Measured in the `default` preset's ctest run of
#164 on the RTX 3080, it comes out at `n_det` 36 and `|y|` 0.6580. The worst
seeded residual is 3.1e-15, the state itself 1.1e-12, and the worst Jacobian
column 1.7e-12 at `I = 0`, which is also the worst overall (relative
2.6e-12). Two earlier hand runs of the same binary gave 2.8e-12 and 3.6e-12
worst, the same `atomicAdd` wander as the CAS(4,4) row above.

`relative` is the worst overall residual over `|Edot(c)|`, the state
direction's own dot product. The seeded directions' residuals are smaller in
absolute terms simply because their dot products are (1e-3 against the state
direction's O(1)); the relative column is the one to read, and it is round-off
at every size. The CAS(4,4) figure is the largest of the three and **moves
between runs** — 1.4e-12, 1.5e-12 and 2.7e-12 on three runs of the same
binary before #87, 4.0e-13 and 2.2e-12 on two after it — and the CAS(10,10)
worst likewise wanders between 8e-15 and 2e-14: the RDM build's `atomicAdd`
reassociation reaching the answer through two different paths, not a step
size. The 1e-7 gate is therefore five decades of headroom.

*Superseded, on record:* before #87 the two sidecar cases ran the
fixed-orbital-energy functional too, and measured `|y|` = 0.5803 / 0.9448 with
worst overall residuals 1.5e-12 / 8.7e-15 (relative 2.8e-12 / 9.2e-15). The
norm of `y` moved by the term the omission left out; the residual did not,
because both modes left it out alike — which is the whole point of the next
two sections.

**The full Jacobian, at CAS(4,4).** `ndet = 36`, so every column is
affordable: `y_I` against `Edot(e_I)` for all 36 determinants, one forward
tangent each. Worst column `|y_I - Edot(e_I)| = 2.2e-12` at `I = 0`, where
`max |y_I| = 0.6192` (the complete functional; 1.5e-12 and 0.5426 before
#87). This is not a stricter dot product — it is a
*localising* one: a single `<u, y>` detects a wrong sweep but says nothing
about which half of it is wrong, while a column-by-column comparison names the
determinant. The check is skipped above `ndet = 256` (CAS(8,8) is 4900), where
one tangent per determinant stops being seconds.

The walk `y` rides on is the pseudodensity walk unchanged, which the entry
also confirms the cheap way: the class energies it computes on the way are
still `nevpt2.energy`'s own, to 2.5e-16 / 1.4e-15 / 4.9e-15 at the three
sizes. Building `y` itself takes 0.01s / 0.52s at CAS(4,4) / CAS(8,8), and at
CAS(10,10) what docs/performance.md, "`--tiles` on the CI source term
(`--gradient-source`)" records at `--tiles 3` — five sigma calls (one
3-body, and a forward and a dagger for each of the two `f3` orders) on top of
the walk; the orbital energies' scatter and fold are host arithmetic over
`(n_core + n_virt) n^2` and add nothing visible — and ctest clocks the three
entries at 0.50s / 1.99s / 24.6s, including the five tangent builds and, at
CAS(4,4), the Jacobian's 36.

The sigma kernels' own halves are pinned card-side in `test/sigma_vector/`:
`SigmaVector3GammaTests` and `SigmaVector4GammaTests` check each sigma, and the
forward-plus-dagger sum that an RDM derivative actually is, against an exact
central difference of the host contraction (exact because that contraction is
quadratic in the CI vector), with negative controls — forward + forward
instead of forward + dagger, and a `d6` with its last two axes swapped — that
both miss by more than three decades.

### The dm1/dm2 fold, against an unfolded reference

The duality cannot see one step: `ciSourceFolded` folds the `dm1` and `dm2`
coefficients onto `dm3`'s diagonals, and both sides of `<u, y> == Edot(u)`
read the same folded tensor. `test/gradient_plan/`'s `CiSourceFoldTests`
(host-only, no `gpu` label) check it against `dm1`/`dm2`/`dm3` contracted
**separately**, rebuilt on the host by forward application of `E_pq` from the
committed CAS(4,4) and CAS(8,8) CI vectors, as transition forms
`<a|E..|b>` — the state's own, and the two halves of the derivative along a
seeded random `u`. Everything is asserted to 1e-12 (elementwise absolute, or
for an `n^6` contraction relative to `sum |g| |dm|`); measured worst, on the
build box's host:

| check | CAS(4,4) | CAS(8,8) |
|---|---:|---:|
| host `dm1` / `dm2` against the golden's | 1.4e-17 / 4.4e-16 | 4.2e-15 / 1.2e-14 |
| `sum_t dm3[..,t,t,..] = N dm2`, worst over state and both transition forms | 1.8e-15 | 7.8e-14 |
| `sum_r dm2[..,r,r] = N dm1`, likewise | 8.9e-16 | 2.3e-14 |
| fold against unfolded, at the state / along `d/dc` | 8.6e-17 / 1.6e-16 | 9.8e-17 / 1.4e-18 |

`CiSourceFoldNegativeControlTests` feed the same comparison a deliberately
wrong fold and assert it **fails** by at least 1e-6: dropping `dm1` (it misses
by 9.5e-5 to 2.9e-3), folding at `N - 1` electrons, the trace of PySCF's
reordered `dm2` (1.3e-5 to 1.6e-2), and `dm2` on the crossed diagonal
`d3[p,q,r,t,s,t]` (1.1e-4 to 3.0e-2) — at least seven decades above the 1e-12 tolerance.
`CiSourceVectorTests` (`gpu`, RTX 3080) run `ciSourceVector` itself at
CAS(4,4): its `y` agrees elementwise with the unfolded host derivative to
1.1e-15 relative (bound 1e-10, `test/sigma_vector/`'s), and with the sum of
its per-tensor parts to 1.4e-16.

### The orbital energies' CI response

#87 added the orbital energies' dependence on the active density to both
modes ([`gradients.md`](gradients.md), §5.5) — and the duality above cannot
check it, because both modes read the same `W_p = (pp|ut) - (1/2)(pu|tp)` and
the same class `Delta` constructions. Three things do.

**Card-free, `test/gradient_plan/`** (host-only, no `gpu` label):

- `OrbitalEnergyResponseTests` rebuild the golden's `e_core` / `e_virt` as
  `h^eff_pp + <dm1, W_p>` from the committed N2 CAS(4,4), N2 CAS(8,8) and LiF
  sidecars, to 1e-11 absolute; measured (the same arithmetic in NumPy) 1.8e-13
  / 5.8e-13 / 4.6e-14. The **negative controls** feed the same rebuild the
  picks a transcription would most plausibly get wrong, and assert each misses
  by more than 1e-6: the exchange at full weight misses by 0.20 / 0.32 / 0.32
  Eh, Coulomb and exchange swapped by 4.0 / 8.0 / 6.7, `x` read without its
  `n_core` offset by 2.7 / 5.4 / 4.5. A hand sidecar with every element
  distinct pins the picks elementwise, and forward and reverse are checked to
  be transposes of one map (to 1e-14 relative).
- `DeltaAdjointTests` hold `deltaAdjointInto` — each class's `Delta`
  transposed onto `dE/de` — to the identity `<R, Delta(e)> = <dE/de, e>` for
  all eight classes, exactly (integer inputs), and to a hand count on Sijrs.
- `test/energy_finish/`'s `NormToEnergyDeltaTests` hold the forward side's
  `R = N T^2` to a fourfold Richardson table in `Delta` and to the finish's
  degree −1 homogeneity, `Q H + R Delta = -E`.

**On a card**, every reader of the sidecar first rebuilds the orbital energies
from it and refuses one that does not belong to the golden (relative 1e-10):
`--gradient-source` and the dump measure 8.5e-15 / 2.5e-14 (N2 CAS(4,4) /
CAS(8,8)), and the dump 2.0e-15 on LiF and 4.7e-14 on acrolein.

**End to end, against PySCF** — "The GPU as the source-term provider" below:
the dump's `P y` against PySCF's exact derivative, which is the check the
omission failed.

**The negative control, run once (#87; scratch patch, not committed).** The
new term dropped from **both** modes — `scCiSourceDevice` skipping the `dm1`
fold, `normToEnergyDeltaTangent` returning zero — and everything else as
shipped, on the RTX 3080:

| | N2 CAS(4,4) | N2 CAS(8,8) | LiF CAS(4,6) |
|---|---|---|---|
| `--gradient-source --gradgold` duality, worst `\|<u,y> - Edot(u)\|` | 2.3e-12 — **PASS** | 2.6e-14 — **PASS** | — |
| dump's `P y` against PySCF's exact (2^-17), max over determinants | **3.2e-4** | — | **3.3e-3** |

The duality passes and the independent provider fails by exactly what #73
measured for the omission — a shared omission cancels between the two modes,
and only a provider outside the pair sees it.

## The analytic gradient against the numerical one

The end-to-end check of the gradient work: the **assembled analytic nuclear
gradient** against a **finite-difference** one, SC and PC. Both sides are
offline Python — `reference_data/nevpt2_grad.py` and
`reference_data/nevpt2_numgrad.py` ([`reference-data.md`](reference-data.md),
"The analytic gradient" and "The numerical gradient") — so this needs no card
and is not a ctest entry; "Why there is no ctest entry for it" at the end of
this section says why not. Everything below was run single-threaded on the
CUDA devcontainer's CPU (the RTX 3080 box; `OMP_NUM_THREADS=1`, forced by
`generate_golden`'s import), with PySCF 2.14.0 and block2 0.5.4.

### The tolerance, and why it is not 1e-7

**Every other comparison in this tree is against 1e-7 because it has an exact
oracle: a committed number.** This one does not. The oracle is
`grad_num`, a difference quotient of re-converged RHF+CASCI+NEVPT2 energies,
and the analytic side is itself the difference quotient of one scalar along a
frozen orthonormal path. Two finite differences, each with an error bar, and
the agreement cannot be tighter than they are.

Both bars are already measured where each belongs: `grad_num`'s own
Richardson residual in [`reference-data.md`](reference-data.md), "The
numerical gradient" — it is what fixes the 0.0025 bohr step there — and the
analytic side's frozen-path bars in the same file's "The analytic gradient".
What that file does not cover — the third case, and the two PC runs — is
here: N2 CAS(4,4) is **1.1e-9** (`grad_num`) against **5.1e-8** SC /
**1.6e-6** PC (skeleton) and 1.6e-11 / 4.0e-9 (multipliers); acrolein PC is
**1.7e-9** against **4.9e-7** and 1.1e-8. (The SC bars are at the SC steps
#77 converged, "The step constants, measured" below; at the steps before it
N2 SC's were 8.3e-6 / 5.9e-9. PC's steps did not change.)

The analytic side's skeleton bar is the loose one everywhere, and it is an
**upper** bound rather than an estimate: it is `|R(h/2) - g(h/2)|`, the size
of the `h^2` correction Richardson then *removes*, so it says how wrong the
raw central difference was, not how wrong the stored extrapolant is. N2 SC's
8.3e-6 against a measured agreement of 2.5e-7, before #77, was that gap —
and, as it turned out, not only that: the 2.5e-7 itself was two unconverged
quotients partly cancelling ("N2 CAS(4,4) SC fails, and that is the
finding").

So the gate is **1e-6 per component** (`nevpt2_grad.py`'s `GRAD_NUM_TOL`,
overridable with `--tol`): one decade above the worst agreement measured
below, and still tighter than the published figure it is answerable to —
Park's Table 3 ([`references.md`](references.md)) reports PC-NEVPT2 analytic
gradients within **5.0e-6** of two-point numerical ones. Tightening it to
1e-7 would be gating on the frozen path's step size, not on the gradient.
**This paragraph is the only statement of that rationale**; the code constant,
`nevpt2_grad.py`'s docstring and
[`reference-data.md`](reference-data.md) cite this section by name instead of
restating it.

### What is run, and what it comes out at

Two commands per case — the numerical gradient, then the analytic one gated
against it. The SC pair for LiF and acrolein reads `grad_num` straight out of
the `*.gradgold` sidecar; the PC pair, and N2 (whose committed sidecar
deliberately carries no `grad_num`, and whose in-place write is refused for
that reason), pass it through an `--npz`:

```bash
uv run python reference_data/nevpt2_numgrad.py --case n2_ccpvdz_cas44 \
    --pc --npz /tmp/n2_cas44.numgrad.npz          # SC and PC, one set of states
uv run python reference_data/nevpt2_grad.py --case n2_ccpvdz_cas44 \
    --grad-num /tmp/n2_cas44.numgrad.npz          # SC
uv run python reference_data/nevpt2_grad.py --case n2_ccpvdz_cas44 \
    --pc --grad-num /tmp/n2_cas44.numgrad.npz     # PC
```

Each analytic run prints one `PASS:` or `FAIL:` line and exits 1 on a
failure, the way a demo's does. **This is the single home of the verdict**;
[`reference-data.md`](reference-data.md)'s two gradient sections keep each
run's own diagnostics (residuals, condition numbers, bars, the
zero-by-symmetry and translational-invariance checks for LiF and acrolein) and
cite this table for the comparison. All five `PASS`, the SC columns at the
SC steps #77 converged ("The step constants, measured"); the PC columns'
steps did not change, and N2 PC re-run for #77 reproduces its figure:

| | LiF CAS(4,6) SC | acrolein CAS(6,5) SC | acrolein PC | N2 CAS(4,4) SC | N2 CAS(4,4) PC |
|---|---|---|---|---|---|
| components compared | 6 | 16 (in-plane) | 16 | 6 | 6 |
| **max \|g − grad_num\|** | **8.3e-9** | **6.5e-8** | **8.3e-8** | **4.4e-8** | **1.5e-8** |
| largest component (Eh/bohr) | −0.0072358611 (Li z) | +0.0266017158 (C2 y) | +0.0265842789 (C2 y) | +0.0448800448 (N z) | +0.0452520933 (N z) |
| superseded: at the SC steps before #77 (1e-4, 1e-4, 0.0025 bohr) | 5.3e-9 | 7.6e-8 | — | 2.5e-7 | — |

The superseded row is kept because N2's figure in it was **wrong for a
reason**: at those steps its `NUC_STEP` and `CI_STEP` quotients were each
~1e-6 from converged, and 2.5e-7 was the two errors partly cancelling ("N2
CAS(4,4) SC fails, and that is the finding"). LiF's and acrolein's moved by
less than either case's distance from `grad_num`; their `grad_num` is
unchanged (acrolein's from a fresh `nevpt2_numgrad.py --pc --npz`, which
reproduces the 7.6e-8 at the old steps exactly).

**That table is the `grad_num` line's verdict; each run now prints a second
one.** The residual-independent checks (below) pass on all four cases they
were run on — LiF, acrolein SC, N2 SC and N2 PC; N2 SC failed them until #77
changed its steps.

N2's own free checks, the two this file owns: `|g|` on the two
zero-by-symmetry axes is 8.0e-14 SC / 9.7e-13 PC, and the gradient summed
over the two atoms is ≤1.8e-9 per column SC (the round-off of a 16-fold
smaller nuclear step; 6.2e-11 at the old one) and ≤6.1e-11 PC.

**The PC runs' multipliers are as healthy as the SC ones**, which is worth
saying because only the source terms changed: at N2 CAS(4,4) the CI Z-vector
residual `|P(H−E)Pz + Py|` is 1.5e-15 and the orbital one `|Mλ + B|` is
8.7e-16 direct / 1.7e-12 GMRES at condition number 7.6e3, `z·c` after
projection is 1.7e-18, and the CI/(O2) fixed point closes in 5 iterations;
at acrolein, the CI residual is 1.2e-11, the orbital one 5.9e-15 direct /
8.5e-12 GMRES at condition number 1.2e4, `z·c` is 6.1e-19 and the fixed point
again closes in 6. (The SC runs' are
[`reference-data.md`](reference-data.md), "The analytic gradient"'s.)

**The out-of-plane acrolein components are the free check**, and all eight
come out at ≤5.2e-11 in the PC run (the SC figure and acrolein's column sums
are [`reference-data.md`](reference-data.md), "The analytic gradient"'s). The numerical side
cannot probe them — displacing out of plane breaks the σ_h, which
`nevpt2_numgrad.py` refuses rather than differences — while the analytic side
differences every component, because the frozen path re-solves nothing and so
has no selection to break. They come out as the zeros symmetry says they are,
and the same holds for LiF's and N2's two non-axial axes.

**PC costs what the `kappa` sweep costs**, and that is the one wall worth
quoting: one block2 PC solve per non-gauge rotation against PySCF's per-class
sum for SC, so the analytic run takes **150 s against 9.3 s** at N2 CAS(4,4)
(`n_mo` = 28, 376 rotations) and **22277 s** at acrolein (`n_mo` = 76,
2846 rotations), against the SC wall of the same sweep in
[`reference-data.md`](reference-data.md), "The analytic gradient", which keeps
the two SC walls. Nothing is wrong with that — it is
the price of a self-contained source term, and what the GPU's own dump was
meant to replace — and since #89 (SC) and #74 (PC) it does: "The GPU as the
source-term provider", below, and [`pc-nevpt2.md`](pc-nevpt2.md), "The source
terms by reverse mode", where a PC gradient from the GPU's dump costs an
E_CASCI-only sweep instead of this one.
The numerical side is far cheaper and does both schemes in one pass: 494 s for
acrolein's 96 displaced states and 35.9 s for N2's 36, SC and PC together.

### The PC CI source term's step is a first-order trap

**The one place this pipeline can silently differentiate the wrong function**,
and the reason `--pc` carries its own step constant (`CI_STEP_PC`, 1e-7,
against `CI_STEP`'s 1e-4 for SC when this was measured; SC's is `2^-17` since
#77, for its own truncation reasons, and `CI_STEP_PC` did not move).

PC-NEVPT2 solves one linear system per external tuple in a **linearly
dependent** internally contracted basis: part of each block's spectrum is
null, and `numpy.linalg.lstsq(rcond=None)` drops it. Those modes are null
**structurally** — dependencies among the class's active-space operator
strings, for any state of the reference's symmetry
([`gradients.md`](gradients.md), §7). But `y_I = dE_corr/dc_I` is wanted *per
determinant*, so the quotient displaces the CI vector along a bare
determinant axis, and a bare determinant is spatially allowed (the sweep masks
to the state's own irrep) and still **not a spin eigenfunction**. The
displaced vector therefore has components the reference's symmetry forbids,
the forbidden modes stop being null, and once a lifted mode clears the cut it
is **kept and divided by** — a term of order `h/h` in the quotient. That is a
*first-order* error: no step ladder converges it away, every residual in the
Z-vector solve still reads round-off, and the gradient is simply wrong. It is
§7's condition on the PC tangent's *direction*, met here in the offline source
term; §7's own cure (build the direction from spin-free generators) is not
available to a per-determinant derivative, so the cure here is the step, plus
the guard below.

Measured on the committed N2 CAS(4,4) case — `nevpt2_grad.py --case
n2_ccpvdz_cas44 --pc --grad-num … --ci-step h`, against one fixed
`grad_num_pc`:

| `--ci-step` h | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 3e-7 | **1e-7** | 1e-8 |
|---|---|---|---|---|---|---|---|
| max \|g − grad_num\| | 2.5e-4 | 2.5e-4 | refused | refused | refused | **1.5e-8** | 1.6e-8 |
| the run | FAIL | FAIL | guard, gap 3.6e2 | guard, gap 1.2 | guard, gap 1.9 | PASS | PASS |

**Read the plateau.** 2.481e-4 at 1e-3 and 2.488e-4 at 1e-4 — a factor of ten
in the step and three digits of the error unchanged, which is what a
first-order error looks like and what no amount of ladder-walking would have
fixed. Three decades below the SC step the same comparison lands at 1.5e-8.

The three middle rungs are the guard: `_check_pc_conditioning`, the function
that decides whether a golden may carry PC fields at all
([`reference-data.md`](reference-data.md), "PC fields"), is run at **every
displaced CI vector**, and there it sees the lifted mode sitting astride the
cut (largest dropped 3.8e-16 against smallest kept 1.4e-13 at 1e-5, and
1.2e-14 against 1.5e-14 at 1e-6) and refuses, naming the step. **The guard is
necessary and not sufficient:** at 1e-4 and above the lifted modes clear the
cut outright, the gap looks healthy, and only the comparison shows the error.
So `CI_STEP_PC` is a measured constant and not "whatever the guard accepts" —
1e-7 is where the quotient has converged, and 1e-8 buys nothing (1.6e-8).
(Since #70 the guard also holds every displaced point's rank to the reference
state's, and that does see the 1e-4 plateau — "Residual-independent checks of
the gradient path", error 5.)

**SC's ladder is the ordinary one**, which is the contrast: 5.3e-5 at 1e-3
(a FAIL, on plain truncation), 2.5e-7 at 1e-4, 7.7e-7 at 1e-5, and no
refusal at any rung. A rational function of the CI vector behaves that way;
the PC plateau above does not, and that difference is the whole reason for a
second constant. (This SC ladder was read, at the time, as a minimum at 1e-4
with round-off rising below it. It was not: it was taken at the pre-#77
nuclear and orbital steps, whose own ~1e-6 errors the CI quotient's partly
cancelled at 1e-4. With those converged the SC CI ladder falls as `h²`,
ratio 4.00 per halving, from `2^-12` to `2^-18` — "The step constants,
measured".)

The orbital (`kappa`) sweep is **not** gated, and does not need to be: the
rank that can move is the metric's, the metric is built from the active-space
RDMs, and the RDMs are a function of the CI vector alone — an orbital rotation
at fixed CI cannot move it. Gating it anyway would cost an SVD per PC block
per rotation, 2846 rotations deep at acrolein, for a hazard that is not there.

### The deliberately symmetric case, and the SC finding that did not appear

Park's result for SC-NEVPT2 is that its analytic gradients **disagree** with
numerical ones when inactive orbitals are degenerate, and that this is the
correct behaviour of a functional that is not invariant to rotations inside
the inactive space, not a bug. **The premise is not in doubt and is not new
here** — this tree already measured the non-invariance, for the energy and
before any gradient work: `Srs` is summed over *pairs* of virtuals and moves
by up to 3.0e-6 Eh as a degenerate pair is rotated, which is
[`reference-data.md`](reference-data.md), "Point-group symmetry", and the
whole reason every state is built in its detected group.

N2/cc-pVDZ CAS(4,4) is this tree's deliberately symmetric case, chosen to
provoke the gradient consequence: D2h, and **six exactly degenerate inactive
pairs** — the orbital-energy gaps the run prints are 1.6e-15, 2.2e-16, 0.0,
6.7e-16, 2.2e-15 and 0.0 Eh.

**It did not appear.** SC agrees to 4.4e-8 (2.5e-7 at the steps before #77)
and PC to 1.5e-8, both inside the gate. The reason is visible in the same line of output: every one of those six
pairs has an (O2) source term of **exactly 0.0e+00**, so its undetermined
pseudocanonical multiplier is a genuine `0/0` for which zero is the right
representative, and the gradient does not depend on the choice
([`gradient-theory.md`](gradient-theory.md), §2.3). Two guard rails put it
there, and both are properties of this tree rather than of the physics:

- **every state is built in its detected point group**
  ([`reference-data.md`](reference-data.md), "Point-group symmetry":
  `_build_mol`, never a hard-coded group, and the linear groups mapped to an
  abelian subgroup, Dooh → D2h). An abelian subgroup splits each degenerate
  set across two of its irreps, so N2's degenerate partners land in
  *different* irreps: θ is **pinned by symmetry** rather than by the
  diagonaliser's arbitrary choice — which is the point that section makes
  about the energy, carried into the gradient. The energy's derivative along a
  rotation that mixes two irreps then vanishes identically, and that is what
  the 0.0e+00 above is.
- **the remaining case is refused, not computed.** Two inactive orbitals
  degenerate *within* one irrep is where symmetry cannot pin θ — an abelian
  subgroup does not split every degenerate set (Td's E becomes A + A in D2) —
  and `generate_golden._check_degeneracy` refuses such a state outright when
  the golden is generated, so no committed case can hold one
  ([`reference-data.md`](reference-data.md), "The degeneracy refusal", which
  has the threshold and the CH4 case it rejects).

So what is recorded here is a **negative result**: the SC-degeneracy
disagreement is not reproducible anywhere in this tree. Not because the
non-invariance is absent — it is measured, above — but because a gradient
taken at a **fixed** convention never sees it: symmetry picks θ and holds it
at the reference geometry and at every displaced one, so the energy is a
smooth function of the coordinate and its analytic and numerical derivatives
agree. The disagreement needs θ to be chosen by noise, which is exactly what
the two guard rails above prevent, and no committed
gradient case has an inactive degeneracy that a symmetry-preserving
displacement could lift: LiF and N2 are diatomics — linear under *every*
displacement, so their π degeneracies can never be broken — and acrolein is
`Cs`, every irrep of which is one-dimensional, so it carries no
symmetry-required degeneracy to break.

**What N2's 2.5e-7 did and did not establish.** It was the largest SC
disagreement of the three cases, and it was *not* evidence of a degeneracy
effect. This section first read its size as the frozen-path quotient's own
floor — the skeleton bar above it was 8.3e-6, and the CI-step ladder seemed
to have its minimum there (5.3e-5 at 1e-3, 2.5e-7 at 1e-4, 7.7e-7 at 1e-5).
#70's ladders showed that reading was wrong: the 2.5e-7 was two ~1e-6 step
errors partly cancelling, and at the steps #77 converged the same case lands
at **4.4e-8**, next to PC's 1.5e-8 on the *same* state, the same six
degenerate pairs and the same constraint set ("N2 CAS(4,4) SC fails, and
that is the finding"). Either way, what the case does establish is that the
machinery handles *exact, symmetry-pinned* inactive degeneracy correctly, six
undetermined (O2) multipliers included — it has to recognise each as a `0/0`
and leave it at zero rather than divide through.

**It is not the test benzene is**, and that should be said rather than left
implied. Park's disagreement needs θ to be **unpinned** — degenerate partners
that the working abelian subgroup does not separate, or no symmetry at all —
so that the diagonaliser chooses θ, and chooses differently at each displaced
geometry. N2's partners are separated by construction, so its degeneracy is
exact but benign. N2 is the most severe symmetric case this tree can build; it
is **not** a stand-in for the unpinned-θ regime, and nothing measured here is
evidence about that regime either way.

Benzene, Park's documented SC failure case, is not in this tree and was
**not** added for this: it would need its own golden, its own sidecar and its
own PC conditioning, for one negative data point.
This is in the tradition of [`performance.md`](performance.md), "What did not
help" — the measurement was made, the expected effect was absent, and the
absence is the result.

### LiF has no PC gradient, and that is the answer

`--pc` on LiF stops before any gradient code runs:

```
$ uv run python reference_data/nevpt2_numgrad.py --case lif_def2svp_cas46 --pc
PC block iabcpqg*: lstsq's cut has only a 1.7e+00 gap (largest dropped
singular value 3.5e-14, smallest kept 4.9e-14, relative; need 1000): the
spectrum is a continuum through the cut, so PC-NEVPT2 is not well-defined on
this state ... Pass --no-pc, or pick an active space that keeps
near-doubly-occupied orbitals in the core.
$ echo $?
1
```

That is the same `_check_pc_conditioning` refusal the LiF golden's own
generation gets, and the reason that golden carries SC fields only
([`reference-data.md`](reference-data.md), "LiF and acrolein, the
gradient-validation pair"): four of its six active orbitals are essentially
empty, which flattens the PC metric. **There is no PC reference for LiF, so
there can be no PC gradient check on it** — the second PC molecule is N2
CAS(4,4) instead, and weakening the guard to manufacture one would make the
"reference" an implementation's choice of pseudo-inverse cut rather than a
number.

### SC on a CASSCF reference: LiF, and the N2 degeneracy probe

**The SC-NEVPT2 nuclear gradient on a CASSCF reference agrees with its
numerical gradient componentwise on both cases, from PySCF's source terms and
from the GPU's: LiF CAS(4,6) to 2.9e-8 Eh/bohr and N2 CAS(4,4) to 9.4e-9.
The gate is 1e-6: four PASS.** This is #125. The analytic side is
`nevpt2_grad.py --reference casscf` (#122's dense CP-CASSCF solve). The
numerical side is `nevpt2_numgrad.py --reference casscf` (#124: every
displaced state re-optimised from a cold start, then finished). Both run on
the committed `golden/<case>_casscf.mogold` orbitals.

**What was asked of N2, and which outcome it is.** N2 CAS(4,4) is the
deliberate degeneracy probe: D2h, with six exactly degenerate inactive
pairs. The pseudocanonical multipliers divide by `ε_i − ε_j`, so the issue
named three possible outcomes: (1) the degeneracy refusal fires, (2) the
numerator vanishes too and the `0/0` resolves, giving agreement, or (3)
neither, and the gradient disagrees. **The result is outcome 2.** The run
prints all six pairs, from both providers:

```
(O2): 6 degenerate pair(s) left at zero -- (11,12) gap 7.8e-16 source 0.0e+00;
(14,15) gap 7.8e-16 source 0.0e+00; (18,19) gap 4.4e-16 source 0.0e+00;
(20,21) gap 2.2e-16 source 0.0e+00; (22,23) gap 3.1e-15 source 0.0e+00;
(25,26) gap 0.0e+00 source 0.0e+00
```

The source term is exactly 0.0 on every pair, from PySCF's quotient and from
the GPU's dump alike. So `degenerate_constraints` leaves each multiplier at
zero and does not refuse. 175 pairs are divided out (`|M_oo zc + Y|`
1.5e-16), and the gradient agrees to 9.4e-9. That is the CASCI twin's
mechanism ("The deliberately symmetric case, and the SC finding that did not
appear" above), now on a CASSCF reference. D2h puts each degenerate pair's
partners in different irreps, so the energy's derivative along the rotation
that mixes them vanishes identically.

**What this says about Park, and what it does not.** The run-up to this
issue expected SC to disagree at about benzene's level (~1e-3) once a
displacement broke the degeneracy. That expectation was inferred from Park's
Table 3, not measured. It **did not appear**, and the reason is structural.
A diatomic stays linear under every displacement, so its π degeneracies are
never broken: symmetry pins the mixing angle at the reference and at all 36
displaced states. The numerical sweep shows it. Every displaced state
converged in the reference's irreps, with the continuity guard well inside
its thresholds (`1 − σ_min` ≤ 1.1e-5, `1 − |⟨c₀|c⟩|` ≤ 1.7e-7), and cold and
warm starts reached the same point. So this **neither confirms nor
contradicts** Park's benzene and porphine finding. That finding needs the
angle chosen by noise, and no case here can produce that, for the reasons
given in the CASCI subsection. Benzene was not added for this.

**Both cases' coupled Hessians, measured on the same runs.** These are the
mechanism behind each outcome:

| | LiF CAS(4,6) | N2 CAS(4,4) |
|---|---|---|
| coupled dimension (orbital + CI) | 154 + 116 = 270 | 191 + 11 = 202 |
| smallest eigenvalues of the symmetric part | **−7.3215e-3 (×2)**, +2.8218e-6 (×2), +2.8196e-3 (×2) | +5.5700e-2 (×2), +6.7099e-2 (×2), +1.0069e-1 |
| negative eigenvalues / condition number | 2 / 4.08e7 | 0 / 1.31e3 |
| right-hand side on the nearest-singular modes, `\|b·v\|/\|b\|` | ≤ 6.7e-15 on all six listed (3.1e-2 on the next, +6.4212e-3) | ≤ 6.6e-16 |
| (O2): divided / degenerate, left at zero (gap; source) | 81 / 3: (14,15) 7.8e-9, (18,19) 5.3e-15, (20,21) 1.1e-8; source 0.0 on all three | 175 / 6 (above) |
| `\|J^T L + b\|` direct / through `h_op` | 1.0e-16 / 4.7e-16 | 3.5e-17 / 1.3e-15 |

N2 is a minimum: every eigenvalue is positive, and the degenerate pairs live
in (O2), not in the coupled system. LiF's saddle (#122) is the case where
the issue's outcome 1 could have fired. It does not, and it does not need
to. Its two negative and two near-null modes are pure orbital rotations,
and the right-hand side on them is ≤ 6.7e-15 of `|b|`. Since #123 they are
known to break the symmetry ([`reference-data.md`](reference-data.md), "LiF
and acrolein on CASSCF orbitals"), so they cost the solve nothing. LiF also
has three (O2) pairs below the gap floor. Two of them, at 7.8e-9 and 1.1e-8,
are degenerate only to the CASSCF's convergence. Their source term is
exactly 0.0 too, so they are left at zero for the same reason.

**Per component** (Eh/bohr; the off-axis components are differenced by both
sides, because a diatomic stays linear):

| comp | LiF numerical | residual | LiF analytic, PySCF / GPU | \|Δ\| PySCF / GPU | N2 numerical | residual | N2 analytic, PySCF / GPU | \|Δ\| PySCF / GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 x | +3.9e-9 | 3.6e-9 | ~0 / ~0 | 3.9e-9 / 3.9e-9 | ~0 | 2.3e-11 | ~0 / ~0 | 2.8e-11 / 2.8e-11 |
| 0 z | −0.0088266040 | 2.7e-9 | −0.0088265748 / −0.0088265749 | **2.9e-8** / **2.9e-8** | +0.0502435407 | 7.9e-10 | +0.0502435501 / +0.0502435500 | **9.4e-9** / **9.3e-9** |
| 1 x | −1.52e-8 | 1.6e-8 | ~0 / ~0 | 1.5e-8 / 1.5e-8 | ~0 | 3.3e-11 | ~0 / ~0 | 7.6e-12 / 7.6e-12 |
| 1 z | +0.0088266032 | 1.5e-9 | +0.0088265749 / +0.0088265749 | 2.8e-8 / 2.8e-8 | −0.0502435400 | 1.2e-9 | −0.0502435486 / −0.0502435485 | 8.6e-9 / 8.5e-9 |
| y, both atoms | 0 | 0 | ≤ 3.4e-18 | — | 0 | 0 | ≤ 9.4e-14 | — |

"~0" means below 1e-17 (analytic) or 1e-11 (numerical). LiF's worst
disagreement, 2.9e-8, is at its numerical side's own floor. #124 measured
that floor from congruent pairs as 1.9e-8 Eh/bohr ([`reference-data.md`](reference-data.md),
"The sweeps"). N2's floor is 7.5e-10 from the same measurement, with a
largest residual of 1.2e-9. N2's 9.4e-9 sits beside its analytic side's
skeleton bar, 4.9e-9 (an upper bound, "The tolerance, and why it is not
1e-7" above). Both are inside Park's own SC figure for LiF, an RMS of
2.4e-6 ([`references.md`](references.md)). That is an anchor, not the gate.

**GPU-sourced.** `nevpt2_sc_pseudodensity --dump-source-terms` writes each
state's `.gradsrc` on the card. `nevpt2_grad.py --source both` reads it, and
the CASSCF branch takes it since this change ("The GPU as the source-term
provider" below has the mechanism). The dump is in the golden's basis. On
this branch the state is held in that basis too, with no §2.6 rotation, so
the dump is read unrotated. A PC dump on a CASSCF reference is taken the
same way since #163 ("PC on a CASSCF reference: acrolein" below, "GPU-sourced").
The two providers' terms and gradients:

| | LiF CAS(4,6) | N2 CAS(4,4) |
|---|---|---|
| dump vs PySCF, max \|ΔA\| (worst block) | 4.1e-9 (core–virt) | 3.3e-10 (act_O–virt) |
| dump vs PySCF, max \|Δ P y\| | 4.6e-9 (117 det.) | 6.0e-10 (12 det.) |
| max \|GPU-sourced − PySCF-sourced\| gradient | 3.5e-11 | 1.1e-10 |
| the GPU's `Y` alone / `y` alone | 4.1e-10 / 4.5e-10 | 1.2e-10 / 2.3e-10 |
| GPU-sourced vs `grad_num` | 2.9e-8: **PASS** | 9.3e-9: **PASS** |
| residual-independent checks | 14 of 14 | 14 of 14 |

The dumps were taken on the RTX 3080 (CUDA, preset `default`, `--tiles 3`).
The same two dumps from the RX 9060 XT (gfx1200, preset `hip`) gave every
figure in this table to the digits shown. N2's per-block figures also match
the ones the `.srcgold` oracle entry recorded on the same state ("The source
terms on CASSCF orbitals, measured" above).
LiF's CASSCF dump had no entry before this. It is now
`nevpt2_sc_pseudodensity_dump_lif_def2svp_cas46_casscf`, beside its CASCI
twin, with `nevpt2_lif_cas46_casscf` as its energy entry ("The CASSCF-orbital
entries" above).

**Walls and budgets** (Ryzen 9 5900X, single-threaded processes; PySCF
2.14.0; the GPU steps on the RTX 3080, preset `default`, `--tiles 3`):

| run | LiF CAS(4,6) | N2 CAS(4,4) |
|---|---:|---:|
| `nevpt2_numgrad.py --reference casscf`, `--workers` 12 / 10 | 20.4 s (#124's; budget 600 s) | 4.9 s (budget 600 s, declared for #125 before the run) |
| `nevpt2_grad.py --reference casscf`, PySCF-sourced | 59.3 s (budget 300 s) | 13.5 s (budget 120 s) |
| the same, `--source both` | 75.2 s | 18.2 s |
| the dump's ctest entry | 2.56 s | 1.34 s |

LiF's numerical gradient is #124's own `.npz`, reused, not re-swept. Its
reference re-optimised bit-identically to the committed `.mogold`, which is
the orbitals this run differentiates. The four budgets are bounds, not
gates (`GATE_WALL_BUDGET`, `CASSCF_WALL_BUDGET`).

Re-run with:

```bash
uv run python reference_data/nevpt2_numgrad.py --case n2_ccpvdz_cas44 \
    --reference casscf --workers 10 --states-dir /tmp/n2_casscf_states \
    --npz /tmp/n2_casscf.numgrad.npz
uv run python reference_data/nevpt2_grad.py --case n2_ccpvdz_cas44 \
    --reference casscf --grad-num /tmp/n2_casscf.numgrad.npz      # PySCF-sourced
ctest --preset default -R 'nevpt2_sc_pseudodensity_dump_(n2_ccpvdz_cas44|lif_def2svp_cas46)_casscf$'
uv run python reference_data/nevpt2_grad.py --case n2_ccpvdz_cas44 \
    --reference casscf --source both \
    --source-path build/gradsrc/n2_ccpvdz_cas44_casscf.gradsrc \
    --grad-num /tmp/n2_casscf.numgrad.npz                         # GPU-sourced
# LiF: the same with --case lif_def2svp_cas46, and its numerical gradient from
# "The sweeps" in reference-data.md (nevpt2_numgrad.py ... --npz /tmp/lif_casscf.numgrad.npz)
```

### PC on a CASSCF reference: acrolein

**The PC-NEVPT2 nuclear gradient of acrolein CAS(6,5) on a CASSCF reference
agrees with its numerical gradient to 6.0e-8 Eh/bohr in the worst of the 16
in-plane components (RMS 2.3e-8), and the 8 out-of-plane ones come out at
≤1.4e-11. The gate is 1e-6: PASS.** This is #126, the milestone's headline
result. The external anchor is Park's Table 3 ([`references.md`](references.md)):
an acrolein PC analytic-vs-numerical RMS of 3.36e-6 on a CASSCF reference,
from a two-point difference at 0.001 bohr. That is an anchor, not the gate.

**What is compared.** The analytic side is `nevpt2_grad.py --reference
casscf --pc`: block2's PC source terms `Y` and `y` (the CASCI branch's
`--pc`, unchanged) fed into #122's dense CP-CASSCF `(Z, z)` solve. The
numerical side is **this GPU's PC energy** on #124's 121 re-optimised
CASSCF states, `pc_on_states.py`'s keyed `.npz` (#138; every state guarded,
smallest gap 2.9e3), differenced by `nevpt2_numgrad.py --pc-states` with the
SC sweep's own h, h/2, h/4 stencil and Richardson pair
([`reference-data.md`](reference-data.md), "Differencing them:
`--pc-states`"). So the PC energies are ours and the derivative machinery is
offline Python, and the comparison asks whether the analytic gradient is the
derivative of the energy we compute. Our energy's own run-to-run spread is
far below anything a difference could see ([`pc-nevpt2.md`](pc-nevpt2.md),
"The PC total's run-to-run spread"). The screen ran first and is green:
#135's directional PC tier on acrolein's CASSCF orbitals ("On CASSCF
orbitals: acrolein" below). **No `grad_num_pc` is committed.** The sweep is a
manual validation run, like every row of this section.

Per component (Eh/bohr; the analytic gradient differences every component
along the frozen path, so the out-of-plane ones are its free check):

| comp | PC numerical | its Richardson residual | PC analytic | \|analytic − numerical\| |
|---|---:|---:|---:|---:|
| O x | −0.0154493570 | 1.6e-9 | −0.0154493186 | 3.8e-8 |
| O y | −0.0132698003 | 1.2e-8 | −0.0132698113 | 1.1e-8 |
| C1 x | +0.0056159200 | 1.3e-9 | +0.0056159298 | 9.8e-9 |
| C1 y | −0.0112110632 | 2.5e-8 | −0.0112110497 | 1.4e-8 |
| C2 x | +0.0233458277 | 1.4e-8 | +0.0233457994 | 2.8e-8 |
| **C2 y** | +0.0272757763 | 6.9e-8 | +0.0272757165 | **6.0e-8** |
| C3 x | −0.0116514180 | 4.3e-8 | −0.0116514477 | 3.0e-8 |
| C3 y | +0.0042104084 | 3.5e-9 | +0.0042104106 | 2.2e-9 |
| H4 x | +0.0035114108 | 1.5e-8 | +0.0035114019 | 8.9e-9 |
| H4 y | +0.0049788578 | 1.3e-8 | +0.0049788440 | 1.4e-8 |
| H5 x | −0.0009517822 | 3.1e-8 | −0.0009517975 | 1.5e-8 |
| H5 y | −0.0067247704 | 3.3e-8 | −0.0067247938 | 2.3e-8 |
| H6 x | −0.0048485788 | 1.4e-8 | −0.0048485708 | 8.0e-9 |
| H6 y | −0.0004986382 | 7.9e-9 | −0.0004986376 | 6.4e-10 |
| H7 x | +0.0004280029 | 2.7e-8 | +0.0004280032 | 3.0e-10 |
| H7 y | −0.0047606771 | 3.2e-9 | −0.0047606790 | 1.9e-9 |
| all 8 z | 0 by symmetry, not differenced | — | ≤1.4e-11 | — |

Atoms are indexed in the case's geometry order (0 O, 1–3 C, 4–7 H). The
worst component is also the one with the largest numerical bar, 6.9e-8, so
the agreement is at the oracle's own resolution. The PC gradient differs
from the SC one on the same states by up to 5.7e-4 (C1 x), four decades
above the agreement, so this is not SC agreeing with itself. The screen's
four seeded directions, differenced from the same file, agree with
`<g, d>` to ≤6.8e-8 (`--pc-states --analytic`, PASS at 1e-6). The analytic
side's free checks: the gradient sums over atoms to ≤3.8e-10 per column,
and its frozen-path bars are 4.2e-7 (skeleton, the usual upper bound) and
1.7e-8 (multipliers). The numerical one sums to 2.5e-8 (x) and 9.3e-8 (y).

**The coupled solve.** Dimension 1114 (1063 orbital variables plus the 51
in-sector CI directions orthogonal to `c`), condition number 2.7e4, no negative eigenvalue, `|J^T L + b|`
2.8e-16 direct and 3.3e-13 through `h_op`, `z·c` −1.9e-17. The three
Hessian-vector ladders hold `h_op` to ≤6.9e-9 (9 fourfold halvings each).

**The wiring checks, all exact.** Differencing the same file's
`E_CASCI + E_corr(SC)` reproduces #124's SC `grad_num` **bit for bit**, its
residuals too, and the screen's SC directional derivatives likewise. The
device's own SC energies on those states give the same SC gradient to
3.8e-12. The PC reference state was rebuilt to 5.7e-14 of the committed
golden's `e_pc_total`. The sweep's finished reference sits 1.23e-8 off it,
the finish's shift (#138).

**block2 at displaced geometries.** At the reference geometry the GPU's PC
energy is gated against block2 to 1e-7. The spot-checks close the case of a
geometry-dependent error by running block2's PC on three displaced states
of the sweep, through `generate_golden.py --mo-from <label>.mogold` (with
PC) into scratch:

| state | displacement | GPU − block2, `E_PC` (Eh) |
|---|---|---:|
| `2y+0` | C2 y +0.0025 bohr (the largest component) | −2.0e-15 |
| `0x-0` | O x −0.0025 bohr | −1.0e-15 |
| `d2+0` | screen direction d2, +0.0025 bohr | −3.2e-15 |

The same states' `E_CASCI` and SC `E_corr` came out identical to the sweep's
(difference 0). Each spot-check golden took about 31 s of CPU.

**Two of the run's fourteen residual-independent checks fail, and that is a
known property of acrolein's PC functional, not of this solve.** `y`
linearity and CI stationarity, both on seed 70, show two fourfold halvings
where the rule wants three, so `nevpt2_grad.py` exits 1 on them. Their
Richardson values are 4.9e-10 from the reference, against the 1e-6
allowance. The piece they share, `E_corr(PC)` along a CI direction, hits its
round-off floor at `2^-15`. Read as `|change| × h`, that
is ~1e-14 Eh of noise per block2 evaluation, the 100–400 ulp that #104
measured on acrolein's CASCI twin, whose `y`-linearity ladders failed the
same way ("PC's CI-direction ladders, measured" below). The ladder cannot
start coarser, because the PC guard sets its top at `2^-11`. Nothing was
loosened: the rule, `CI_LADDER_PC` and `CI_STEP_PC` are unchanged. The other
twelve hold: seed 2026's two CI ladders (three halvings each), both orbital
stationarity ladders over the 2840 rotations (11 halvings), both nuclear
ladders, both projections, the three Hessian-vector ladders and the PC
guard ledger (168 of 168 displaced CI vectors guarded, smallest gap 2.4e5,
rank unchanged). Acrolein's `P y` is held exactly where these ladders
cannot reach, by #135's directional tier.

**`--reference casci --pc` did not move.** N2 CAS(4,4) CASCI PC, run before
any change and again on the final code: every array of the `--out` file is
bit-identical, the gradient included (1.5e-8 from `grad_num`, PASS, 11 of
11). The SC CASSCF branch and its `--casci-only` smoke test (`origin/main`'s
`reference_data/` against this branch's, N2 CAS(4,4)) are bit-identical too.
The smoke test is 2.0e-8 from `pyscf.grad.casscf`.

**N2 CAS(4,4) CASSCF PC, the cost probe.** Run first, to declare the budget
(no PC numerical gradient exists for it, so the residual-independent checks
are its gate): 14 of 14 hold, the three Hessian-vector ladders within
1.2e-8, and the gradient's off-axis components ≤1.5e-11. Wall 182.0 s
serially, of which the `kappa` sweep is 111 s.

**Walls and workers** (the RTX 3080 box's CPU: 12 physical cores,
single-threaded processes; PySCF 2.14.0, block2 0.5.4):

| run | workers | wall |
|---|---|---:|
| N2 CAS(4,4) CASSCF PC analytic | 1 | 182.0 s (budget 600 s) |
| the same, `--workers 4` | 4 in the sweep | 99.7 s (sweep 111 s → 29 s), bit-identical arrays |
| acrolein CASSCF PC analytic | 10 in the sweep | **4288 s** (budget 36000 s): `y` 389 s, the 2840-pair sweep 2332 s, the checks 723 s |
| `--pc-states` differencing | 1 | 1.3 s |
| block2 spot-check goldens | 1 | ~31 s each |

The `CASSCF_WALL_BUDGET_PC` figures were declared from N2's wall before
acrolein ran: 600 s and 36000 s. The 36000 s covers a serial acrolein run,
which on the CASCI branch takes 22277 s ("What is run, and what it comes out
at" above). The acrolein run shared the CPU part of the time with
`golden-repro.sh -j 1` during the sweep and with the build and ctest tiers
afterwards, so its wall is an upper bound. `--workers N` forks the `kappa`
sweep and nothing else. Its arrays matched the serial run's exactly on N2.

Re-run with:

```bash
uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65 \
    --reference casscf --pc-states /tmp/acrolein_pc_states.npz \
    --npz /tmp/acrolein_casscf_pc.numgrad.npz
uv run python reference_data/nevpt2_grad.py --case acrolein_def2svp_cas65 \
    --reference casscf --pc --workers 10 \
    --grad-num /tmp/acrolein_casscf_pc.numgrad.npz --out /tmp/acrolein_casscf_pc.grad.npz
uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65 \
    --reference casscf --pc-states /tmp/acrolein_pc_states.npz \
    --analytic /tmp/acrolein_casscf_pc.grad.npz --npz /tmp/acrolein_casscf_pc.numgrad.npz
uv run python reference_data/generate_golden.py --atom "<2y+0's --atom spec>" \
    --basis def2-svp --ncas 5 --nelecas 6 --cas-irreps "A'=1,A\"=4" \
    --mo-from /tmp/acrolein_states/2y+0.mogold --out /tmp/spot/2y+0.nevpt2gold
```

with `/tmp/acrolein_states` and `/tmp/acrolein_pc_states.npz` from the two
commands in [`reference-data.md`](reference-data.md), "The numerical
gradient of a CASSCF reference" and "The GPU's PC energies on the kept
states". The `--atom` spec is the state's own, in that sweep's `states.json`.

#### GPU-sourced

**From the GPU's PC source terms, the same gradient agrees with the same
numerical one to 6.1e-8 Eh/bohr in the worst in-plane component (RMS
2.3e-8), at the 1e-6 gate: PASS. It differs from the block2-sourced
gradient on the same sweep by at most 3.8e-9.** This is #163. The analytic
side is `nevpt2_grad.py --reference casscf --pc --source both`. It reads the
GPU's PC dump (`nevpt2_sc_pseudodensity --pc --dump-source-terms` on
`golden/acrolein_def2svp_cas65_casscf.nevpt2gold`). It then solves block2's
`Y` and `y` through the same `kappa` sweep, Jacobian and coupled solve, so
the two gradients can differ only through the source terms. The PC dump on
a CASSCF reference was held to block2's committed oracle block by block
first, by #160 on N2 CAS(4,4) ([`pc-nevpt2.md`](pc-nevpt2.md), "On a
CASSCF reference"), and acrolein's directional PC tier is green ("On CASSCF
orbitals: acrolein" below). The numerical side is the section's own
procedure, re-run from scratch for this change. The sweep and the PC
energies were regenerated on the RTX 3080 (`pc_on_states.py` with the CUDA
`nevpt2_demo`), and every `grad_num_pc` component, residual and
translational sum came out as in the table above, to the digits shown.

**The gate is the section's, unchanged:** 1e-6 per component against
`grad_num_pc`, and the fourteen residual-independent checks. **The verdict
is the same partial one #126 recorded:** the gradient PASSes, 12 of 14
checks hold, and `nevpt2_grad.py` exits 1 on the same two ladders.

| | GPU-sourced | block2-sourced |
|---|---:|---:|
| max \|g − `grad_num_pc`\| (16 in-plane) | 6.09e-8 (C2 y) | 5.99e-8 (C2 y) |
| RMS over the 16 | 2.29e-8 | 2.27e-8 |
| out-of-plane, max \|g\| | 1.4e-11 | 1.4e-11 |
| sum over atoms, max per column | 3.8e-10 | 3.8e-10 |
| residual-independent checks | 12 of 14 | 12 of 14 |

Between the two providers, on the one sweep: max \|GPU-sourced −
block2-sourced\| is 3.8e-9 over all 24 components (RMS 1.4e-9 in-plane).
The GPU's `Y` alone, solved with block2's `y`, moves it by 3.4e-9, and the
GPU's `y` alone by 4.4e-10. The source terms themselves differ by at most
3.3e-8 in `A` (core–virt) and 4.4e-8 in `P y`. That is the size of
block2's side, a central difference ([`pc-nevpt2.md`](pc-nevpt2.md),
"block2's exact source terms, on a card"). The block2-sourced gradient of this
`--source both` run is **bit-identical** to a separate `--source pyscf` run
on the same tree. That second run reproduces #126's record: 5.99e-8, PASS,
12 of 14.

**The two failing ladders fail on the same piece, bit for bit, from either
provider.** Both differentiate block2's `E_corr(PC)` along seed 70's CI
direction, which does not depend on where `Y` and `y` came from. The
`y`-linearity ladder (that piece alone) is identical in the two runs: its
central differences, its two halvings and its round-off floor at `2^-15`.
The mechanism is the paragraph "Two of the run's fourteen
residual-independent checks fail" above. CI stationarity's whole-Lagrangian
ladder moves with the multipliers, but it falls back to pieces in both
runs. Its `E_corr` piece is that same ladder, so its verdict is identical.
GPU sourcing therefore can neither cause nor cure either failure. What a
provider changes is the reference each ladder is compared with, and the
multipliers:

| check | GPU-sourced \|Richardson − ref\| | block2-sourced |
|---|---:|---:|
| `y` linearity, seed 70 (2 halvings, **FAIL** both) | 3.1e-10 | 4.9e-10 |
| CI stationarity, seed 70 (by pieces, `E_corr` 2 halvings, **FAIL** both) | `E_corr` piece as block2's | as GPU's |
| `y` linearity, seed 2026 | 4.6e-10 | 1.1e-8 |
| CI stationarity, seed 2026 | 4.5e-10 | 1.1e-8 |
| orbital stationarity, seeds 70 / 2026 | 1.3e-8 / 9.9e-9 | 2.7e-8 / 4.3e-9 |
| nuclear ladder, seeds 70 / 2026 | 8.7e-10 / 2.3e-9 | 1.7e-9 / 2.6e-10 |
| Hessian-vector, three seeds | ≤ 6.9e-9 (identical) | ≤ 6.9e-9 |
| PC guard | 168 of 168, gap 2.4e5 | 168 of 168, gap 2.4e5 |

On the CI-direction ladders the GPU's adjoint `y` sits closer to block2's
own finite-difference ladder than block2's `y` does: 4.6e-10 against
1.1e-8 on seed 2026. block2's `y` is itself a central difference, with that
truncation. Nothing was loosened: the rule, `CI_LADDER_PC`, `CI_STEP_PC`
and the 1e-6 are unchanged.

**N2 CAS(4,4) CASSCF PC, `--source dump`** (no numerical gradient, so the
checks are its gate, as in the cost probe above): 14 of 14 hold, and the
gradient is 3.4e-9 from a block2-sourced run of the same state.

**Walls** (the RTX 3080 box, `--workers 12` in the `kappa` sweep):

| run | wall |
|---|---:|
| `nevpt2_sc_pseudodensity --pc --dump-source-terms`, acrolein CASSCF, `--tiles 3` | 178 s |
| `nevpt2_numgrad.py --reference casscf`, the 96-state componentwise sweep | 891 s |
| `pc_on_states.py --workers 10`, 97 states (GPU 69.6 s) | 79.7 s |
| `nevpt2_grad.py --reference casscf --pc --source both` | 6236 s (budget 36000 s) |
| the same, `--source pyscf` (block2) alone | 3906 s |
| N2 CAS(4,4), `--source dump`, `--workers 2` | 67.2 s |

For part of the time the two acrolein gradient runs shared the CPU with the
golden regeneration gate, pytest, and the build and ctest tiers. Their
walls are therefore upper bounds.

```bash
./build/src/nevpt2_sc_pseudodensity \
    --golden golden/acrolein_def2svp_cas65_casscf.nevpt2gold --tiles 3 --pc \
    --dump-source-terms /tmp/acrolein_casscf_pc.gradsrc
uv run python reference_data/nevpt2_grad.py --case acrolein_def2svp_cas65 \
    --reference casscf --pc --source both --workers 12 \
    --source-path /tmp/acrolein_casscf_pc.gradsrc \
    --grad-num /tmp/acrolein_casscf_pc.numgrad.npz
```

The dump needs `golden/acrolein_def2svp_cas65_casscf.gradgold`, which is
gitignored. Regenerate it with the `--mo-from … --gradgold` command in
[`reference-data.md`](reference-data.md), "Regenerating the golden file".
It came out at its recorded sha256, and the golden beside it
byte-identical.

### The density-fitted gradient against the numerical one

#166 is the density-fitted twin of this section: `nevpt2_grad.py --df`
differentiates golden/`n2_ccpvdz_cas44_df`'s fitted SC-NEVPT2 energy (fitted
RHF, `DFCASCI`, `dfnevpt2._ERIS`, every displaced geometry fitting its own
integrals), gated against #167's `grad_num` in the `NEVPT2F1` sidecar beside
it. Both providers, on one sweep:

- **GPU-sourced**: `nevpt2_sc_pseudodensity --dump-source-terms` on the
  `_df` golden (#168), whose `Y` carries both terms, the frozen-digest one
  and the digests' transpose, carried into the gradient's own active basis
  by `load_source_terms` exactly as an integral-direct dump is.
- **PySCF-sourced**: central differences of **PySCF's own DF-NEVPT2
  energy**, the same `dfnevpt2._ERIS` the `_df` golden is built from. PySCF
  has no NEVPT2 gradient of either flavour, so "PySCF-sourced" means
  quotients of its energy here as it does in the conventional runs. It is
  not a conventional calculation: the fitted and exact `E_corr` differ by
  1.2e-5 on this state, and `test_grad_df.py` holds the state to the `_df`
  golden and away from the conventional one.

**The tolerance is this section's 1e-6, unchanged.** Both sides are finite
differences, which is the case "The tolerance, and why it is not 1e-7"
covers, and the fitted case's bars sit where the conventional N2's do:
`grad_num`'s Richardson residual is **9.8e-10** (its stencil h, h/2, h/4 at
0.0025 bohr, the step read off the plateau in
[`reference-data.md`](reference-data.md), "The numerical gradient";
the fitted sidecar's own figure is "The density-fitted numerical
gradient"), and the analytic side's frozen-path bars are **5.1e-8**
(skeleton) and **2.2e-11** / 2.3e-11 (multipliers, the PySCF-sourced / both
runs), against conventional N2's 1.1e-9 / 5.1e-8 / 1.6e-11. The worst
agreement below is 4.3e-8, so the gate keeps the decade of margin that
section sets it at. Nothing about fitting moves either bar's order, so no
new number was set.

Run on the RTX 3080 box (the dump on the card, CUDA `default` preset,
Release, `--tiles 3`; everything else single-threaded on its CPU, PySCF
2.14.0):

```bash
uv run python reference_data/nevpt2_grad.py --case n2_ccpvdz_cas44 --df
./build/src/nevpt2_sc_pseudodensity --golden golden/n2_ccpvdz_cas44_df.nevpt2gold \
    --tiles 3 --dump-source-terms /tmp/n2_cas44_df.gradsrc
uv run python reference_data/nevpt2_grad.py --case n2_ccpvdz_cas44 --df \
    --source both --source-path /tmp/n2_cas44_df.gradsrc --localise
uv run python reference_data/nevpt2_grad.py --case n2_ccpvdz_cas44 --df \
    --source dump --source-path /tmp/n2_cas44_df.gradsrc --edot-check
```

| | N2 CAS(4,4) `_df` (cc-pVDZ-JKFIT, `naux` 140) |
|---|---|
| state vs the `_df` golden, `E_CASCI` / `E_corr` | 0.0 / 5.0e-16 |
| components compared | 6 (linear under every displacement) |
| **GPU-sourced, max \|g − grad_num\|** | **1.38e-8** — `PASS` |
| **PySCF-sourced, max \|g − grad_num\|** | **4.32e-8** — `PASS` |
| max \|GPU-sourced − PySCF-sourced\| | 2.94e-8; the GPU's `Y` alone moves it 2.63e-8, its `y` alone 3.05e-9 |
| the axial component, GPU / PySCF / `grad_num` (atom 0 z, Eh/bohr) | +0.0448030161 / +0.0448029867 / +0.0448030289 |
| the source terms, dump vs PySCF's exact ones: worst `A` block, worst `P y` | 1.98e-7 (core–virtual), 8.9e-9 |
| ... vs PySCF with the digests frozen in `A` and `e_core`/`e_virt` in `y` (`--localise`) | 2.9e-1 (act_O–virtual), 3.2e-4 |
| `--edot-check`: the dump's `Edot(kappa)`, frozen-digest + digests' terms, vs PySCF's exact directional derivative of the fitted energy | 18 of 18 ladders within 1e-6, `PASS` |
| zero-by-symmetry axes, \|g\|; sum over atoms (z) | ≤8.7e-15; 1.8e-9 |
| Z-vector residuals, CI / orbital direct / orbital GMRES; condition number | 8.7e-16 / 1.2e-15 / 1.7e-12 PySCF-sourced, 5.5e-15 / 1.2e-15 / 1.7e-12 GPU-sourced; 7.6e3 |
| **residual-independent checks** | **10 of 11 — `FAIL`**, both runs: the nuclear ladder, seed 70 (below) |
| wall, PySCF-sourced / both (`--localise` included) / `--edot-check` | 14.2 s / 23.3 s / 5.0 s |

**Both `Y` terms are load-bearing, and the `--localise` row is the
evidence.** The dump matches PySCF's exact fitted `A` to 2e-7 and misses the
frozen-digest one by 0.29, so it is the digests' transpose, not the
frozen-digest term alone, that the gradient reads. A scratch negative
control (not committed) solved the same sweep with PySCF's frozen-digest `A`
and the dump's `y`: **9.8e-2** from `grad_num`, the axial component +0.1425
against +0.0448. A dump without #168's term is a wrong gradient, not a
looser one.

**Flavour.** A `.gradsrc` names no flavour, but the CI vector it was taken
at does: the conventional N2 CAS(4,4) dump handed to `--df` and the fitted
one handed to the conventional run are both refused as a different state,
8.5e-5 from the golden's CI vector against `DUMP_CI_TOL`'s 1e-8. A dump with
the fitted CI vector and the conventional `E_corr` is refused by the energy
check behind it (`test_grad_df.py`).

**The one failing check is the nuclear ladder's run length, at N2's known
edge.** "What the committed cases come out at", below, records that N2 SC's
nuclear ladder closes its run of three only at its last rung, `2^-16`, with
round-off below it, and fails on run length alone under small changes to
the steps. The fitted state's frozen path has the same shape (seed 70's
ratios 2.86, 3.20, 3.60 at `2^-13`–`2^-15`, against conventional 2.87,
3.21, 3.70), and the round-off rung is where the two part: **5.105**
PySCF-sourced and **4.800** GPU-sourced at `2^-16`, against the window
(3.2, 4.8) and conventional's 3.302. Seed 2026, the same direction
reversed, passes in both runs with a run of three ending at `2^-16`. The
seed-70 ladder's Richardson value is 7.8e-9 (6.8e-9 GPU-sourced) from
`<d, g>`, well inside its 1e-6, so the quotient it checks has converged;
the check fails because the rule demands a run the ladder cannot lengthen
at this case's round-off. Neither the window nor the run length was
changed. `nevpt2_grad.py --df` exits 1 on this case for that reason, with
both `grad_num` lines reading `PASS`.

### Why there is no ctest entry for it

Three reasons, each sufficient on its own:

- **it would put PySCF and block2 in the test path.** `reference_data/` is
  offline by design — "offline Python generates the reference, the C++ GPU
  demo reproduces it" — and nothing in the build or the test tier runs any of
  it. A ctest entry that shells out to `nevpt2_grad.py` inverts that and makes
  the Python environment part of the product.
- **it needs no card, and the golden tier is 100% `gpu`.** This comparison is
  between two offline gradients; no `src/` or `apps/` binary participates.
  The GPU-sourced variant ("The GPU as the source-term provider", below) has
  one binary step — the dump — and that step is a ctest entry of its own; the
  comparison after it is offline like this one.
- **the walls are minutes to hours.** Acrolein's PC pair is 6.3 h end to
  end. Even labelled `slow` that is not an entry anyone would run, and
  `ctest --preset fast` would become unusable if it were not.

So this is a **manual gate**, like the sanitizer tier and the cross-backend
check: the commands are above, they print `PASS:`/`FAIL:` and set an exit
status, and the protocol is the same one "No CI, and the local gate" states —
run them and say what they printed.

## The GPU as the source-term provider

Every gradient in the section above was built from **PySCF's** source terms.
#73 closes the open link: `nevpt2_sc_pseudodensity --dump-source-terms PATH`
writes the GPU's own `Y_xy` and `y_I` to a `.gradsrc`, and
`nevpt2_grad.py --source both` builds the nuclear gradient from them, solves
it a second time from PySCF's on the **same** `kappa` sweep (so the two can
differ only through the source terms), and prints both beside `grad_num`.
This section is SC's; PC's GPU-sourced gradient (#74, `--pc` on both
programs) is [`pc-nevpt2.md`](pc-nevpt2.md), "The source terms by reverse
mode". The gate is the same 1e-6 per
component, for the reason "The tolerance, and why it is not 1e-7" gives.
Everything here is on CASCI references. The same GPU-sourced comparison on a
CASSCF reference (#125, LiF and N2 CAS(4,4), SC) is "SC on a CASSCF
reference: LiF, and the N2 degeneracy probe" above.

**Since #89 the GPU-sourced gradient passes** — within 1e-6 of `grad_num` on
LiF, acrolein and N2 CAS(4,4), and componentwise the PySCF-sourced one's to
well inside that, without `--localise` — because `Y` now carries the digests'
response through their transpose ([`gradients.md`](gradients.md), §6). The
numbers are "The digests' orbital response, reverse mode" below. **What
follows in this section is the record it supersedes**, kept as it was
measured: why the gradient failed before, and which omission cost what.

**Until #89 the GPU-sourced gradient failed, on every case, by four to six
decades — and the failure was entirely the one dependence the device's
functional was documented to hold fixed**: the `f3ac`/`f3ca` digests in `Y`. #73 measured it beside a second one, the
orbital energies held fixed in `y`; #87 put their response into both modes
(§5.5), and **the GPU's `y` is now PySCF's exact one** to PySCF's own
finite-difference truncation, on all four cases. With the digests put back
into PySCF's functional too, the two providers agree on `Y` to that
truncation as well. No other disagreement is left to find.

#73's dumps were taken on the RX 9060 XT (gfx1200; preset `hip`, `--tiles 3`,
the default digest), #87's on the RTX 3080 (preset `default`, `--tiles 3`,
the default digest — acrolein with its sidecar regenerated out of tree, whose
arrays are the committed golden's exactly); everything after them is
single-threaded Python, PySCF 2.14.0 (#73: the RX box's Ryzen 9 5900X):

```bash
build-hip/src/nevpt2_sc_pseudodensity --golden golden/lif_def2svp_cas46.nevpt2gold \
    --tiles 3 --dump-source-terms /tmp/lif.gradsrc
uv run python reference_data/nevpt2_grad.py --case lif_def2svp_cas46 \
    --source both --source-path /tmp/lif.gradsrc            # gated on grad_num
uv run python reference_data/nevpt2_grad.py --case lif_def2svp_cas46 \
    --source both --source-path /tmp/lif.gradsrc --sources-only --localise
# (#73's localisation; since #87 the y comparison that matters is WITHOUT
#  --localise, whose y half freezes the orbital energies the dump now moves)
# acrolein's sidecar is gitignored: regenerate it (generate_golden.py --gradgold
# --out elsewhere; the arrays match the committed golden's exactly) and pass it
# with --gradgold; its grad_num comes from nevpt2_numgrad.py --npz, --grad-num
```

### The verdict, and which term costs what

#87's runs — **superseded by #89**, whose dumps carry the digests' term and
pass; kept as the record of what that term was worth — `--source both` with
no `--localise` (N2 CAS(4,4)'s `grad_num` from `nevpt2_numgrad.py --npz`, as
acrolein's):

| | LiF CAS(4,6) | acrolein CAS(6,5) | N2 CAS(4,4) |
|---|---|---|---|
| components compared | 6 | 16 (in-plane) | 6 |
| **GPU-sourced**: max \|g − grad_num\| | **7.55e-3 — FAIL** | **6.34e-1 — FAIL** | **9.78e-2 — FAIL** |
| PySCF-sourced, same sweep: max \|g − grad_num\| | 8.3e-9 — PASS | 6.5e-8 — PASS | 4.4e-8 — PASS |
| max \|GPU-sourced − PySCF-sourced\| | 7.55e-3 | 6.34e-1 | 9.78e-2 |
| … the GPU's `Y` alone (with PySCF's `y`) | 7.55e-3 | 6.34e-1 | 9.78e-2 |
| … **the GPU's `y` alone** (with PySCF's `Y`) | **1.3e-11** | **5.9e-11** | **3.1e-9** |
| … the GPU's `y` alone, #73 (orbital energies held fixed) | 7.9e-5 | 8.4e-4 | 1.6e-4 |

The PySCF-sourced row reproduces "What is run, and what it comes out at"
exactly (8.3e-9, 6.5e-8, 4.4e-8), so the shared sweep and the second
solve change nothing. **The GPU's `y` alone is now inside the 1e-6 gate on
all three, by two and a half to five decades** — #73 measured it two decades
over (the last row, kept as the record it was): the gradient-level cost of
the orbital-energy omission was the whole of it. What is left is the digests
in `Y`, which are the GPU-sourced gradient's entire error to the digit. The
largest components show what that number *is*: LiF's Li z comes out at
−0.0147840167 against `grad_num`'s −0.0072358529, N2's at +0.1426859347
against +0.0448800876, and acrolein's worst, C2 x, at +0.6534118644 against
+0.0194046166 — wrong by factors of two to thirty, not by a tolerance (#73's
dumps, with the second omission in, gave −0.0147047919, +0.1425259048 and
+0.6525725264). Both attribution rows are first-order exact (the gradient is
linear in the source terms).

### The source terms, componentwise

`--source both` compares the dump with PySCF's `A_pq = Y_pq − Y_qp` over
every non-gauge rotation, block by block, and `P y` determinant by
determinant on the state's irrep. `--localise` repeats the comparison against
PySCF's functional **with the device's two omissions put back** — the
digests frozen at the state's along every rotation, the orbital energies
frozen at the golden's along every CI direction. Max \|dump − PySCF\| per
block, against the exact derivatives and then against the localised ones —
#73's dumps, with both omissions in (the `Y` blocks are unchanged by #87;
its `P y` is the next table):

| block | N2 CAS(4,4) | N2 CAS(8,8) | LiF CAS(4,6) | acrolein CAS(6,5) |
|---|---|---|---|---|
| core–core | 5.3e-11 / 5.3e-11 | 1.8e-11 / 1.8e-11 | 1.2e-10 / 1.2e-10 | 4.8e-10 / 4.8e-10 |
| core–virt | 2.0e-7 / 2.0e-7 | 3.5e-9 / 3.5e-9 | 6.4e-9 / 6.4e-9 | 1.7e-8 / 1.7e-8 |
| core–act | 0 (by symmetry) | **2.4e-1** / 1.2e-9 | 0 (by symmetry) | **4.0e-1** / 8.2e-4 |
| act_O–act_V | 1.8e-12 / 1.3e-11 | **3.1e-1** / 4.1e-9 | **1.4e-1** / 2.4e-9 | **2.2e-1** / 5.8e-5 |
| act_O–virt | **2.9e-1** / 4.1e-5 | **6.9e-1** / 3.5e-7 | **5.5e-1** / 1.8e-7 | **5.6e-1** / 2.2e-3 |
| act_V–virt | **1.1e-2** / 6.3e-10 | **3.8e-2** / 4.8e-10 | **5.9e-2** / 2.2e-10 | **2.8e-2** / 5.6e-7 |
| virt–virt | 4.7e-11 / 4.7e-11 | 2.6e-10 / 2.6e-10 | 2.5e-10 / 2.5e-10 | 1.8e-10 / 1.8e-10 |
| `P y`, #73 | **3.2e-4** / 9.0e-9 | **5.4e-3** / 7.2e-10 | **3.3e-3** / 5.8e-10 | **3.7e-3** / 9.5e-10 |

**`P y` since #87, against PySCF's EXACT derivative** — no `--localise`, whose
`y` half now freezes something the dump moves — max over the state's irrep:

| | N2 CAS(4,4) | N2 CAS(8,8) | LiF CAS(4,6) | acrolein CAS(6,5) |
|---|---|---|---|---|
| determinants compared | 12 | 4900 | 225 | 52 |
| max \|dump `P y` − PySCF's\| at `CI_STEP` = 2^-17 | **9.0e-9** | **7.2e-10** | **5.8e-10** | **9.6e-10** |
| … #73, orbital energies held fixed | 3.2e-4 | 5.4e-3 | 3.3e-3 | 3.7e-3 |

What is left is PySCF's own truncation, and the ladder says so: the dump
against PySCF's quotient at the worst three determinants of each case as
PySCF's step shrinks (`reference_data/nevpt2_grad.py`'s own state, reader and
projected quotient, restricted to those determinants; max over the three):

| PySCF's step h | 2^-13 | 2^-15 | 2^-17 | 2^-19 | 2^-21 |
|---|---|---|---|---|---|
| N2 CAS(4,4) | 2.3e-6 | 1.4e-7 | 9.0e-9 | 5.7e-10 | 8.2e-10 |
| acrolein CAS(6,5) | 2.4e-7 | 1.5e-8 | 9.6e-10 | 2.8e-10 | 1.9e-9 |
| LiF CAS(4,6) | 8.2e-11 | 1.5e-10 | 5.8e-10 | 2.5e-10 | 4.8e-9 |
| N2 CAS(8,8) (its worst determinant) | 4.2e-11 | 2.0e-11 | 7.2e-10 | 1.5e-10 | 3.5e-9 |

Sixteen-fold per two halvings — `h²` — down to the quotient's round-off
floor, which `E/2h` reaches at 2^-19 and climbs off at 2^-21: the leftover at
the committed step is PySCF's truncation, converging onto the dump. LiF's and
N2 CAS(8,8)'s components are small enough that every rung is already at that
floor. N2 CAS(8,8)'s 2^-17 pass differences all 4900 determinants and took
6481 s (`--sources-only`, kappa sweep included), so its ladder is run at that
pass's worst determinant alone rather than the worst three.

Read the block table three ways:

- **The conventions are right.** Core–core and virt–virt are blocks where
  neither omission reaches (no active two-electron index, no orbital energy
  moves — [`gradients.md`](gradients.md), §6), and there the dump is PySCF's
  to ≤5e-10 on every case. So the dump's sign, its `Y_pq = ½ dE/dt` normalisation and the
  reader's carry from the golden's natural-orbital basis into §2.6's are all
  correct; an error in any of them would show here first.
- **Every block an active index touches is wrong by the digests, and only
  there.** Active–active and active–virtual rotations move the active
  two-electron integrals the digests are built from, and so do core–active
  ones — zero by symmetry on N2 CAS(4,4) and LiF, but not on N2 CAS(8,8) or
  acrolein, where they are as wrong as the rest.
- **What is left after localising is PySCF's truncation, not the device.**
  Every leftover checked — the worst of each case, acrolein's 2.2e-3 and
  8.2e-4 included — converges **onto the dump** as PySCF's step shrinks — the frozen-digest
  functional has large higher derivatives, so `KAPPA_STEP = 2^-17`, converged
  for the exact functional ("The step constants, measured"), is not converged
  for this one:

| PySCF's step h | 2^-13 | 2^-15 | 2^-17 | 2^-19 | 2^-21 |
|---|---|---|---|---|---|
| N2 CAS(4,4) act_O–virt (6, 12), digests frozen: dump − PySCF | 1.1e-2 | 6.6e-4 | 4.1e-5 | 2.6e-6 | 1.6e-7 |
| N2 CAS(4,4) core–virt (1, 9): dump − PySCF | 1.7e-6 | 1.3e-6 | 2.0e-7 | 1.4e-8 | −3.4e-10 |
| LiF act_O–virt (4, 15), digests frozen: dump − PySCF | −4.5e-5 | −2.8e-6 | −1.8e-7 | −1.3e-8 | 9.9e-10 |
| acrolein act_O–virt (13, 43), digests frozen: dump − PySCF | −8.9e-1 | 5.1e-2 | 2.2e-3 | 1.3e-4 | 8.4e-6 |
| acrolein core–act_O (9, 13), digests frozen: dump − PySCF | −5.6e-1 | −1.5e-2 | −8.2e-4 | −5.1e-5 | −3.2e-6 |

  Sixteen-fold per two halvings, `h²`, once past the coarse rungs where the
  frozen-digest quotient is still wild (acrolein's flips sign at `2^-13`);
  acrolein act_V–virt and act_O–act_V do the same. The core–virt pair is
  one N2 does not differentiate cleanly at the committed step even for the
  *exact* functional (no digest moves along it); it is 2e-7 there, a decade
  inside the gate, and N2's PySCF-sourced gradient lands at 4.4e-8 regardless.

**The residual-independent checks see it too, without `grad_num`.** #73's
GPU-sourced runs failed six of their eleven ("Residual-independent checks of
the gradient path"): both `y`-linearity ladders (the dump's `y` against
directional quotients of PySCF's exact functional — #40's fourth error, the
orbital energies held fixed, is exactly what that check was built to catch),
and both CI- and both orbital-stationarity ladders of the Lagrangian. #87's
fail **two**, on all three cases: both orbital-stationarity ladders, which
read `Y`. Both `y`-linearity ladders now hold (worst of the two, 1.4e-10 /
5.4e-11 / 2.0e-11 from their Richardson limits, N2 CAS(4,4) / LiF / acrolein), and so do
both CI-stationarity ones. The nuclear ladders and the projection of `z` pass,
as they should: the solve and the contraction are right, only their input is
not.

**What closed it.** `Y`'s digest term is the 4-RDM contraction's transpose
([`gradients.md`](gradients.md), §6), in tree since #89; `y`'s
orbital-energy term, the other half of this section when #73 wrote it, is
#87's `dm1` coefficient ([`gradients.md`](gradients.md), §5.5). The dump and
`--source both` were the seam both were checked through, and the `Y` blocks
above that were the digests' — `--localise`'s columns — are now matched by
the dump against the **exact** derivatives, to PySCF's truncation: "The
digests' orbital response, reverse mode" below.

### What it costs, and what is a ctest entry

The dump's `Y` is one basis direction per MO pair, 16 to a slab walk
(`scFullRotationMatrix`); `y` is one reverse sweep. On the RX 9060 XT: `Y`
in 1.9 s (378 pairs) at N2 CAS(4,4), 27.6 s at CAS(8,8), 4.8 s (253 pairs) at
LiF and 385 s (2850 pairs) at acrolein; `y` in 0.02 s to 1.2 s. The PySCF
side's walls are "What is run"'s plus one `y` sweep and two more solves; at
LiF the whole `--source both` run is 82 s, at acrolein 1321 s. N2 CAS(8,8)'s
`--sources-only --localise` took 13129 s: its degenerate π pairs leave the
active orbitals not symmetry-pure, so every one of its 4900 determinants is
differenced, twice.

The dump is a binary run, so it is a `gpu` ctest entry:
`nevpt2_sc_pseudodensity_dump_<case>` for N2 CAS(4,4), CAS(8,8) and LiF,
writing into the build directory and matching `PASS: the GPU's source terms
Y_xy and y_I were assembled, written to … and read back unchanged` — the
matrix against the direct pairing on a seeded full direction (reassociation,
1e-9 relative), `y`'s walk against `energiesDevice` (1e-7), and the file read
back bit for bit. Not `small` (the probe count is `--kappa-full`'s reason);
not acrolein, whose sidecar is gitignored. `test/gradsrc/` holds the writer
card-free: the round trip, the bytes through an independent zip/NPY parser,
and the refusals. Everything after the dump — `--source both` — is offline
Python and stays out of ctest for the three reasons "Why there is no ctest
entry for it" gives; its componentwise half, with PySCF's side committed, is
the next section's entry.

## The digests' orbital response, forward mode

#88 adds the term both of `Y`'s modes leave out — the `f3ac`/`f3ca` digests'
own `h2e` dependence ([`gradients.md`](gradients.md), §6) — in **forward**
mode: `scDigestRotationTangentDevice` reruns the plain RDM build at the
direction's dotted `h2e` and feeds the digests it returns through
`energiesTangentDevice`. It is reported as its own term beside the
frozen-digest tangent; their sum is the exact `Edot(kappa)` at fixed CI.
The duality cannot see it (it is the term the two modes share by leaving it
out), so it has three checks of its own:

- **the independent provider, offline**: `--dump-source-terms` writes six
  seeded directions into the `.gradsrc` — `--kappa-full`'s four full ones,
  one purely active–virtual, one purely core–active — with `edot_frozen` and
  `edot_digest` along each, and `nevpt2_grad.py --edot-check` holds them
  against PySCF's own SC functional along `C exp(t u)`, `u` the direction at
  unit norm: **exact** (digests rebuilt at every point) against `frozen +
  digest`, **frozen** (digests held at the state's — `--localise`'s
  functional) against `frozen`, and the two differenced against `digest`.
  Each is a `step_ladder` over `h = 2^-3 … 2^-21`: a run of at least three
  fourfold halvings, then the Richardson extrapolant within the gradient
  path's 1e-6 per unit `kappa` ("Residual-independent checks of the gradient
  path");
- **on the card**, `--kappa-full`'s check 6: the energy stage rerun with
  every integral, RDM and orbital energy the state's own and *only* the
  digests rebuilt from `h2e` rotated on the host by `exp(h kappa)`, along
  check 3's active direction, differenced and Richardson-read as check 4 is;
- **an exact zero**: a core- or virtual-internal rotation moves no active
  integral, so `h2e_dot` — and every digest built from it — is exactly 0.0,
  and so is the term (`--kappa-full`'s check 4 directions).

Run on the RTX 3080 (CUDA, preset `default`, Release, `--tiles 3`, the
default emitted digest), Python single-threaded with PySCF 2.14.0; acrolein
with its sidecar regenerated out of tree (`generate_golden.py --gradgold
--out` elsewhere, passed with `--gradgold`) as in "The GPU as the
source-term provider":

```bash
./build/src/nevpt2_sc_pseudodensity --golden golden/<case>.nevpt2gold --tiles 3 \
    --dump-source-terms /tmp/<case>.gradsrc       # + --gradgold PATH for acrolein
uv run python reference_data/nevpt2_grad.py --case <case> \
    --source dump --source-path /tmp/<case>.gradsrc --edot-check
```

### The provider gate

**PASS on all four cases, 18 of 18 ladders each**, worst `|Richardson −
GPU|` per unit `kappa` over the six directions:

| | exact | frozen | digest | `--edot-check` wall |
|---|---:|---:|---:|---:|
| N2 CAS(4,4) | 1.2e-10 | 5.2e-10 | 3.8e-10 | 6 s |
| N2 CAS(8,8) | 2.3e-11 | 2.3e-10 | 1.9e-10 | 285 s |
| LiF CAS(4,6) | 7.3e-10 | 1.6e-10 | 1.1e-10 | 27 s |
| acrolein CAS(6,5) | 4.7e-07 ¹ | 3.0e-08 | 1.6e-09 | 85 s |

¹ the core–active direction's window maximum, below; on the other five
acrolein directions the exact ladder's worst is 2.5e-10.

**The term is not small**: along the four full directions it is
1.7e-3 to 1.6e-2 in magnitude at N2 CAS(4,4), −4.6e-2 at LiF's direction 0 against a
frozen-digest tangent of +4.6e-2 — the exact derivative there is 2.2e-4,
two decades below either piece — and 5.8e-2 to 1.3e-1 in magnitude along
the active–virtual directions. That is the size of #73's gradient failure seen
one direction at a time ("The GPU as the source-term provider").

**The leftover is PySCF's truncation, walked down.** LiF's active–virtual
direction, `|PySCF quotient − GPU|` per rung:

| h | exact | ratio | digest term | ratio |
|---|---:|---:|---:|---:|
| 2^-6 | 1.48e-05 | 3.69 | 1.74e-03 | 4.27 |
| 2^-8 | 9.49e-07 | 3.98 | 1.07e-04 | 4.02 |
| 2^-10 | 5.94e-08 | 4.00 | 6.66e-06 | 4.00 |
| 2^-12 | 3.70e-09 | 4.01 | 4.16e-07 | 4.00 |
| 2^-14 | 2.78e-10 | 3.23 | 2.60e-08 | 4.00 |
| 2^-16 | 6.2e-11 | — | 1.63e-09 | 4.00 |
| 2^-18 | 1.4e-10 | — | 6.8e-12 | — |

— ratio 4.00 per halving for seven (exact) and eleven (digest) halvings, onto the GPU's number,
until PySCF's own evaluation noise (~1e-14 Eh, divided by `2h`) takes over.

**Two cases the plain ladder cannot read, and what reads them:**

- **a direction that breaks the symmetry in every component.** N2 CAS(4,4)'s
  and LiF's core–active blocks are σ core against π actives, so every
  rotation there is symmetry-forbidden: `E(t)` is even in `t` to the last
  bit, every quotient is exactly 0.0, and the GPU's three numbers are 0.0 as
  well. There is no truncation to ladder; zero is the exact answer, and
  `--edot-check` holds the GPU to it at the same 1e-6 (the oracle's
  "zero by symmetry on both sides", "The SC source terms against PySCF's, on
  a card"). The core–active block is exercised where symmetry allows it:
  N2 CAS(8,8) (term −3.1e-2, 13 fourfold halvings) and acrolein.
- **a near-pole of the frozen-digest functional.** Along acrolein's
  core–active direction the frozen and digest pieces are +0.158 and −0.157
  and their quotients converge as `h^2` only from `2^-14` down (ratio
  6.46, 4.29, 4.06, 4.02, 4.00, 4.00, 4.00 to `2^-21`, onto the GPU's numbers
  to 3.0e-8 and 1.0e-9): a perturber empty at the state, whose `Delta N + H`
  the frozen digests leave nearly cancelling. The ladder therefore runs to
  `2^-21` rather than `KAPPA_LADDER`'s `2^-17`, which left those two a run of
  two. The **exact** functional carries no such pole — the two pieces carry
  it with opposite signs — so its truncation is below PySCF's noise on every
  rung where it would be asymptotic and its ladder shows no fourfold run.
  Since the exact quotient *is* the frozen one plus the digest one rung by
  rung, it is then held instead to a **window**: at least four consecutive
  rungs every one within 1e-6 of the GPU's sum — 17 here (`2^-5 … 2^-21`,
  worst 4.7e-7, best 8.0e-12 at `2^-14`), while its two pieces each pass
  their own ladder. A window and not a sweep minimum, so a crossing cannot
  pass for agreement.

### The on-card checks

`--kappa-full`'s new rows (one run each, the ctest entries' command line):

| | CAS(4,4) | CAS(8,8) | LiF CAS(4,6) |
|---|---|---|---|
| check 6: the digests' term along check 3's active direction | −1.3e-17 (symmetry zero ²) | −2.0656630000e-02 | 2.9009533152e-02 |
| check 6: plateau, fourfold halvings | 3.6e-12, 6 | 3.4e-12, 8 | 8.1e-11, 9 |
| check 4 directions: max \|h2e_dot\|, the term | 0.0, 0.0 | 0.0, 0.0 | 0.0, 0.0 |
| the whole entry, wall (`ctest`, Release) | 1.05 s | 17.40 s | 2.10 s |

The three entries' walls were 1.11 s, 14.04 s and 1.75 s before #88; the
dump entries' are now 1.32 s, 24.45 s and 2.54 s.

² CAS(4,4)'s four active orbitals sit in four distinct D2h irreps, so no
active rotation moves the energy at first order ("The active-rotation source
term"); its check 6 confirms the zero through the same machinery, and the
two other cases carry the content.

### The negative control

The digests' term zeroed in a scratch patch (`out[d].total = 0.0` in
`scDigestRotationTangentDevice`, never committed), rebuilt, dumped and
checked, N2 CAS(4,4) and LiF:

- `--edot-check` **FAILS**, 8 of 18 ladders each: every frozen ladder still
  passes (to 1.6e-10 or better) and every exact and digest ladder on a
  direction with content fails by exactly the missing term, 1.7e-3 … 5.8e-2
  (CAS(4,4)) and 2.9e-3 … 8.0e-2 (LiF); the symmetry-zero core–active
  direction passes, as it must;
- the **duality still passes** on both — "adjoint vs tangent (integral
  halves), every direction within 1e-07: ok" — because it never contained
  the term: that is the omission this whole section exists to catch;
- `--kappa-full` itself FAILS on LiF through check 6 (a 0.0 term against a
  finite difference that does not converge to it: no fourfold run, error
  2.9e-2), and passes on CAS(4,4), where the active direction's term is a
  symmetry zero and so is the patched one — the reason the offline provider,
  not check 6, is the gate.

### What is a ctest entry

Everything on the card already is: the dump's entries
(`nevpt2_sc_pseudodensity_dump_<case>`) now also write the six directions and
check each frozen-digest tangent against `<kappa, Y>` off the matrix they
write (`≤ 1e-7`; measured 3.6e-14, 1.5e-14, 1.8e-14 — and 3.4e-13 at
acrolein), and the `--kappa-full` entries run checks 4 and 6 above inside
their existing PASS line. `--edot-check` is offline Python and stays a
manual gate for the three reasons "Why there is no ctest entry for it"
gives. No tile floor changes: the digest tangent is the plain build
([`performance.md`](performance.md), "Tile floors").

**Under the sanitizers.** None of these six entries is `small`, so — as for
"The full orbital source term" — they were run by name (`ctest --test-dir
build-<preset> -R 'nevpt2_sc_pseudodensity_(kappa_full|dump)_'`, with the
`asan` test preset's `ASAN_OPTIONS` / `LSAN_OPTIONS`) after each preset's own
tier passed, and are clean under all four tools (seconds per entry):

| | ubsan | memcheck | initcheck |
|---|---:|---:|---:|
| `kappa_full` CAS(4,4) / CAS(8,8) / LiF | 3.4 / 19.1 / 3.1 | 11.5 / 447.1 / 39.8 | 8.1 / 152.6 / 16.4 |
| `dump` CAS(4,4) / CAS(8,8) / LiF | 3.1 / 24.9 / 3.3 | 17.4 / 691.5 / 58.9 | 11.7 / 253.7 / 23.7 |

`asan` passed all six too (dump entries 4.1 / 25.8 / 3.9 s), no ASan or LSan
report.

## The digests' orbital response, reverse mode

#89 puts the term "The digests' orbital response, forward mode" computes
into `Y` itself: `scDigestAdjointDevice` runs the slab walk's RDM sweep for
`dE/d(f3ac)` / `dE/d(f3ca)` and hands them to `buildDigestAdjointDevice`, the
f3 path's **transpose** with respect to `h2e` — tiled, `L2` re-produced per
tile, the 4-RDM never formed ([`gradients.md`](gradients.md), §6). Its result
`hbar = dE/d(h2e)` through the digests is paired with every direction's
`h2e_dot` and added to `<kappa, Y>` as a term of its own. Four things hold it:

- **the full duality, on the card**: `<kappa, Y>` with the transpose's term
  against the exact `Edot` — the frozen-digest tangent plus #88's rebuilt
  digests' term — on `--kappa-full`'s four seeded full directions, gated at
  1e-7 beside the integral halves' duality; the dump gates the same across
  the file it writes (its six directions' exact `Edot` against
  `sum kappa.Y` off the written matrix); and along check 6's active
  direction the reverse term against the forward one;
- **the transpose itself, in the unit tier**: `RdmDigestAdjointTests`
  (`test/rdm/rdm_tangent_tests.cpp`, `gpu`) — `<hbar, h2e_dot>` against
  `<G, f3(h2e_dot)>` from the plain build, per order and both together, at
  CAS(4,4) and CAS(8,8) (1e-12 relative to the products' magnitudes); every
  one of CAS(4,4)'s 256 `hbar` elements against the build at a basis
  integral; and one answer at 1, 2, 3 and 5 tiles and every forward digest;
- **the independent provider, on the card**: `--source-oracle` against
  PySCF's exact `Y`, with `kKnownGaps` empty ("The SC source terms against
  PySCF's, on a card");
- **the milestone's claim, offline**: a nuclear gradient built from a fresh
  dump against `grad_num`, and componentwise against the PySCF-sourced one,
  **without** `--localise`.

### The full duality

Max `|<kappa, Y> − Edot|` over the directions, both terms of each side in
(Release, `--tiles 3`, the default digest of each backend):

| | N2 CAS(4,4) | N2 CAS(8,8) | LiF CAS(4,6) | acrolein CAS(6,5) |
|---|---:|---:|---:|---:|
| `--kappa-full`, 4 full directions, RTX 3080 | 3.2e-14 | 7.5e-15 | 1.8e-14 | — |
| `--kappa-full`, 4 full directions, RX 9060 XT | 5.3e-14 | 7.9e-15 | 2.4e-14 | — |
| dump check 4, 6 directions, RTX 3080 | 1.3e-14 | — | 2.1e-14 | 4.6e-13 |
| check 6's direction, reverse vs forward term, RTX 3080 | 7.6e-30 ³ | 9.6e-16 | 5.1e-15 | — |
| the same, RX 9060 XT | 1.7e-29 ³ | 1.1e-15 | 1.8e-15 | — |
| the intra-space directions' reverse term | 0.0 | 0.0 | 0.0 | — |

³ CAS(4,4)'s check-6 direction is the symmetry zero of "The on-card checks"
above: both terms are ~1e-17. Acrolein has no `--kappa-full` run: its
sidecar is gitignored, so it was dumped (with its sidecar regenerated out of
tree) and not checked by the entry. Every `--kappa-full` run above also kept
the recorded self-variation envelope ("The kernels' same-run
self-variation"), which the transpose's term is not in.

### The GPU-sourced nuclear gradient

Fresh dumps (RTX 3080, preset `default`, Release, `--tiles 3`, the default
digest; acrolein's sidecar regenerated out of tree with `generate_golden.py
--gradgold --out`, its `grad_num` by `nevpt2_numgrad.py`, N2's by
`nevpt2_numgrad.py --no-write --npz`), then `nevpt2_grad.py --case <case>
--source both --source-path <dump>` with no `--localise`, single-threaded
Python, PySCF 2.14.0:

| | LiF CAS(4,6) | acrolein CAS(6,5) | N2 CAS(4,4) |
|---|---|---|---|
| components compared | 6 | 16 (in-plane) | 6 |
| **GPU-sourced**: max \|g − grad_num\| | **8.5e-9 — PASS** | **6.6e-8 — PASS** | **1.4e-8 — PASS** |
| PySCF-sourced, same sweep: max \|g − grad_num\| | 8.3e-9 — PASS | 6.5e-8 — PASS | 4.4e-8 — PASS |
| max \|GPU-sourced − PySCF-sourced\| | 1.5e-10 | 4.3e-9 | 2.9e-8 |
| … the GPU's `Y` alone (with PySCF's `y`) | 1.4e-10 | 4.3e-9 | 2.6e-8 |
| … the GPU's `y` alone (with PySCF's `Y`) | 1.2e-11 | 7.9e-11 | 3.1e-9 |
| the dump vs PySCF's exact `A`, worst block | 6.4e-9 (core–virt) | 1.7e-8 (core–virt) | 2.0e-7 (core–virt) |
| residual-independent checks | 11 of 11 | 11 of 11 | 11 of 11 |
| `--source both` wall | 90 s | 1290 s | 24 s |
| #87's GPU-sourced, for the record (digests held fixed) | 7.55e-3 — FAIL | 6.34e-1 — FAIL | 9.78e-2 — FAIL |

The `Y` attribution row fell four to eight decades, to the PySCF-sourced
gradient's own distance from `grad_num` or below; what is left of
GPU-against-PySCF is PySCF's finite-difference truncation, whose largest
block — N2's core–virt (1, 9) — is the pair "The source terms,
componentwise" already records as not differenced cleanly at the committed
step. The blocks that were the digests' are now matched to the exact
derivative: acrolein's core–act_O 6.5e-9 (was 4.0e-1), act_O–act_V 1.4e-10
(was 2.2e-1), act_O–virt 3.7e-9 (was 5.6e-1), act_V–virt 1.0e-10 (was
2.8e-2); LiF's and N2's are the oracle table's. The orbital-stationarity
ladders, the two that failed in #87's runs, hold on all three.

### The negative control

The transpose's term dropped from `<kappa, Y>` in a scratch patch
(`p.adjoint = p.integral + p.eps;` in `scFullRotationDevice`, never
committed), rebuilt (RTX 3080, `default`), N2 CAS(4,4) and LiF:

- `--kappa-full` **FAILS** on both, on the full duality line — the four
  directions miss by exactly the digests' term, 1.7e-3 … 1.6e-2 (N2 CAS(4,4))
  and 2.9e-3 … 4.6e-2 (LiF) — while the integral halves' duality still passes:
  "adjoint vs tangent (integral halves), every direction within 1e-07: ok";
- `--source-oracle` **FAILS** on both, on exactly the blocks `kKnownGaps`
  used to hold: "blocks out of place: act_O-virt, act_V-virt" (N2 CAS(4,4))
  and "act_O-act_V, act_O-virt, act_V-virt" (LiF);
- the dump **FAILS** its check 4 on both, and the provider gate — the
  nuclear gradient from that dump, `--source both` — **FAILS**: max \|g − grad_num\| = 7.55e-3 (LiF) and 9.78e-2 (N2 CAS(4,4)) — #87's GPU-sourced figures to the digit — with both orbital-stationarity ladders failing (9 of 11), while the PySCF-sourced gradient on the same sweep passes as before (8.3e-9, 4.4e-8).

### What is a ctest entry, and the sanitizers

Every on-card check above is inside an existing entry's PASS line:
`nevpt2_sc_pseudodensity_kappa_full_<case>` (the full duality, check 6's
reverse term, the intra-space zeros), `_dump_<case>` (check 4, now the full
duality) and `_oracle_<case>` (no known gap). The transpose's unit suite is
`rdm_tangent_tests`' `RdmDigestAdjointTests`, `gpu`, 0.70 s. The gradient
against `grad_num` stays the manual offline gate it was ("Why there is no
ctest entry for it"). Walls, RTX 3080 / RX 9060 XT (`ctest`, Release):

| entry | CAS(4,4) | CAS(8,8) | LiF |
|---|---|---|---|
| `kappa_full` | 1.11 s / 2.21 s | 17.30 s / 48.11 s | 2.16 s / 18.29 s |
| `dump` | 1.39 s / 2.42 s | 23.93 s / 34.82 s | 2.56 s / 7.93 s |
| `oracle` | 1.34 s / 2.31 s | — | 2.32 s / 5.60 s |

**Under the sanitizers**, the `unit` tier carries `RdmDigestAdjointTests`
into every preset (CUDA: `asan`, `ubsan`, `compute-sanitizer` memcheck and
initcheck; HIP: `hip-asan`, `hip-ubsan`), each preset's own tier passing —
`asan` 195/195, `ubsan` 195/195, memcheck 186/186, initcheck 183/183, `hip-asan` 193/193, `hip-ubsan` 193/193, with `RdmDigestAdjointTests` at 1.21 / 1.04 / 14.18 / 7.83 / 15.84 / 14.62 s. The eight non-`small` entries above were also run by name in each
CUDA preset's build, as #88's were (`ctest --test-dir build-<preset> -R
'nevpt2_sc_pseudodensity_(kappa_full|dump|oracle)_'`, with the presets'
`ASAN_OPTIONS` / `LSAN_OPTIONS` / `UBSAN_OPTIONS`): all eight pass under all four tools, no ASan, LSan, UBSan or compute-sanitizer report. Seconds per entry, `kappa_full` CAS(4,4) / CAS(8,8) / LiF, `dump` the same, `oracle` CAS(4,4) / LiF: `asan` 18.0 / 20.4 / 3.6, 4.2 / 26.0 / 4.0, 4.0 / 3.6; `ubsan` 3.5 / 19.6 / 3.3, 3.2 / 25.2 / 3.4, 3.0 / 3.1; memcheck 12.3 / 475.1 / 42.3, 17.7 / 734.0 / 60.5, 17.2 / 55.8; initcheck 9.1 / 156.7 / 17.0, 11.9 / 257.1 / 24.5, 11.4 / 22.6. The HIP presets ran their own tiers only (the unit suite above in each), not these entries by name.

## The SC source terms against PySCF's, on a card

`nevpt2_sc_pseudodensity --source-oracle golden/<case>.srcgold` (#92) is the
componentwise half of the section above made a `gpu` ctest entry:
`nevpt2_sc_pseudodensity_oracle_n2_ccpvdz_cas44` and `…_lif_def2svp_cas46`.
It assembles the dump's own `Y_xy` and `y_I` (the same code as
`--dump-source-terms`), and compares them **block by block** with PySCF's
exact source terms, computed once by `reference_data/sc_source_oracle.py` and
committed ([`reference-data.md`](reference-data.md), "The SC source-term
oracle"): `A = Y − Y^T` carried into the oracle's §2.6 basis, on every
non-gauge pair, grouped into the eight blocks of core / act_O / act_V / virt,
and `P y` on the state's irrep in the golden's own basis.

It exists because **every other on-card check of these two terms compares the
device with itself** — forward against reverse, at round-off — and a term
both modes leave out cancels there. #73 found two that way, and only through
the offline `--source both`; this is that comparison with PySCF's side
committed, so a regression in a correct block, or a fix that lands, shows up
on the next `fast` run.

**The known gaps are pinned, not hidden — and there are none.**
`kKnownGaps` in `apps/sc_pseudodensity/source_compare.cppm` is the **one** list
of blocks the device is known to get wrong; a block on it must still disagree
by **more** than the tolerance and every other block must agree **within**
it. So the entry fails when a correct block breaks, and fails when a fix
lands without the list being updated — the way #64's envelope pins the
exchange non-collapse. **The list is empty since #89**, so every block of
both oracles is held to agree. What it held until then, kept as the record:

| oracle | known gaps until #89 | `P y` |
|---|---|---|
| N2 CAS(4,4) | act_O–virt, act_V–virt — the blocks `Y`, holding the `f3ac`/`f3ca` digests fixed, got wrong | off since #87 |
| LiF CAS(4,6) | act_O–act_V, act_O–virt, act_V–virt — the same | off since #87 |

N2's act_O–act_V and both cases' core–act blocks are zero by symmetry on
**both** sides (exactly 0.0 here), so they were held to agree throughout; on
N2 CAS(8,8) and acrolein core–act was a gap too, but neither has an oracle
(their `P y` quotients cost hours, and acrolein's sidecar is gitignored). An
oracle with no entry in `kClassifiedOracles` is refused as a configuration
error, not read as "everything must agree", and a list entry naming no block
fails the run.

### The tolerance, measured from the step ladders

The tolerance cannot be 1e-7: PySCF's side is a central difference, and the
agreement can be read only to its truncation. It is read off a ladder —
`sc_source_oracle.py --kappa-step h --ci-step h --out …` at five steps,
each scratch oracle compared with the same GPU terms by the same binary
(RTX 3080, preset `default`, Release, `--tiles 3`, the default digest).
Max \|GPU − PySCF\| per block:

| block | 2^-13 | 2^-15 | **2^-17 (committed)** | 2^-19 | 2^-21 |
|---|---:|---:|---:|---:|---:|
| N2 core–virt | 5.4e-6 | 1.3e-6 | **2.0e-7** | 1.4e-8 | 3.2e-9 |
| N2 `P y` | 2.3e-6 | 1.4e-7 | **9.0e-9** | 6.4e-10 | 1.9e-9 |
| N2 core–core / virt–virt (worst) | 1.9e-9 | 1.2e-10 | **5.3e-11** | 2.6e-10 | 1.6e-9 |
| N2 act_O–act_V | 1.6e-12 | 5.9e-12 | **1.8e-12** | 5.8e-11 | 1.7e-10 |
| LiF core–virt | 3.4e-7 | 3.8e-8 | **6.4e-9** | 1.8e-9 | 6.1e-9 |
| LiF `P y` | 5.4e-9 | 4.2e-10 | **5.7e-10** | 2.2e-9 | 8.5e-9 |
| LiF core–core / virt–virt (worst) | 6.2e-10 | 9.7e-11 | **2.5e-10** | 1.7e-9 | 7.1e-9 |
| smallest known gap (N2 act_V–virt) | 1.08e-2 | 1.08e-2 | **1.08e-2** | 1.08e-2 | 1.08e-2 |

The correct blocks converge **onto the GPU** as PySCF's step shrinks, down to
the quotient's round-off floor, which they reach at 2^-19 and climb off at
2^-21: N2's `P y` sixteen-fold per two halvings (`h²`) from the first rung,
N2's core–virt more slowly at the coarse rungs (4x, then 6.5x, then 14x — the
same pair, (1, 9), whose ladder "The source terms, componentwise" records)
and LiF's already near the floor. The gaps do not move in the third digit at
any step, because they are not truncation. The worst
correct block at the committed step is N2's core–virt, at 2.0e-7: the pair
"The source terms, componentwise" above already names as the one N2 does not
difference cleanly at that step. So the gate is

| | value |
|---|---|
| **the source-oracle tolerance** | **1e-6** |

per element, absolute (Eh): five times the worst correct block at the
committed step, and four decades below the smallest known gap. The row is
this tolerance's one home: `apps/sc_pseudodensity/CMakeLists.txt` reads it at
configure time into `NEVPT2_SOURCE_ORACLE_TOL` (on `nevpt2.source_compare`), as it reads the
self-variation envelope, and the entries' comment cites this section. The
2^-17 column is the committed files exactly: the ladder's own 2^-17 oracles
are byte-identical to them.

### What the entries measure

At the committed step, max \|GPU − PySCF\| per block — since #89, on the RTX
3080 (preset `default`) and the RX 9060 XT (preset `hip`), Release, `--tiles
3`, the default digest, which agree to the four digits shown; the former
gaps' #92 values (the #73 table's, to the digit) beside them:

| block | N2 CAS(4,4) | LiF CAS(4,6) |
|---|---:|---:|
| core–core | 5.3e-11 | 1.2e-10 |
| core–act_O / core–act_V | 0 / 0 | 0 / 0 |
| core–virt | 2.0e-7 | 6.4e-9 |
| act_O–act_V | 1.8e-12 | 2.4e-10 (was **1.4e-1**, gap) |
| act_O–virt | 9.2e-10 (was **2.9e-1**, gap) | 6.5e-10 (was **5.5e-1**, gap) |
| act_V–virt | 5.8e-11 (was **1.1e-2**, gap) | 3.4e-10 (was **5.9e-2**, gap) |
| virt–virt | 4.7e-11 | 2.5e-10 |
| `P y` | 9.0e-9 (12 det.) | 5.7e-10 (225 det.) |

The former gaps now sit inside the blocks that always agreed: the worst block
of either oracle is still N2's core–virt, PySCF's own truncation ("The
tolerance, measured from the step ladders"), so the tolerance needs no
change. ctest clocks the two entries at 1.34 s and 2.32 s on the RTX 3080, 2.31 s and 5.60 s on the RX 9060 XT (the `Y` probe is 378
and 253 basis directions). Not `small`, for the dump's reason: under the
sanitizer tier the probe count would dominate.

**Negative controls, run once (#92; scratch patches, not committed),** on the
RTX 3080, both entries each time:

| patch | N2 CAS(4,4) | LiF CAS(4,6) |
|---|---|---|
| the GPU's core–virt sign flipped (a correct block broken) | **FAIL**: core–virt 1.1e-1, "a block that agreed no longer does" | **FAIL**: core–virt 1.0e-1 |
| act_O–virt "fixed" by substituting PySCF's value, list unchanged | **FAIL**: act_O–virt 0, "now AGREES — take it off kKnownGaps" | **FAIL**: the same |
| the same substitution, act_O–virt taken off `kKnownGaps` | PASS, 8 agree / 1 gap | PASS, 7 agree / 2 gaps |

So the entry fails in both directions the issue asks for, and goes green
only when the list moves with the fix. #89 emptied it in the same change
that put the transpose into `Y`. Since #95 the comparison, the list and the
tolerance live in `nevpt2.source_compare` (`apps/sc_pseudodensity/
source_compare.cppm`), and `test/source_compare/` holds it to both oracles
here, and the PC one below, with no card.

## The PC source terms' oracle

`golden/n2_ccpvdz_cas44_pc.srcgold` (#95) is the section above for
**PC-NEVPT2**, committed ahead of #74's adjoint so that its GPU PC `Y_xy` and
`y_I` land against an independent gate that already exists, rather than
against their own forward/reverse duality — which is exactly what cannot see
a term both modes leave out (#73). It is #92's file in its PC flavour (magic
`NEVPT2P1`, the same header and arrays), written by `sc_source_oracle.py
--pc` out of `nevpt2_grad.py --pc`'s own machinery: block2's PC energy on the
rebuilt state (checked against the golden's `e_pc_total`), its
`kappa_sweep` and its `pyscf_ci_source`, every displaced CI vector through
the PC guard ([`reference-data.md`](reference-data.md), "The SC source-term
oracle"). N2 CAS(4,4) only: LiF has no PC answer ("LiF has no PC gradient,
and that is the answer"), and acrolein stays the manual gate.

**Since #74 the GPU's PC terms are held to it on a card**:
`nevpt2_sc_pseudodensity_oracle_pc_n2_ccpvdz_cas44` runs `--pc
--source-oracle` against this file, every block held to agree within the
tolerance below, and every block does on its first run — what it measures,
on both cards, is [`pc-nevpt2.md`](pc-nevpt2.md), "The source terms by
reverse mode", PC's one home. Beside it the comparison itself is gated:
`nevpt2.source_compare`
(`apps/sc_pseudodensity/source_compare.cppm`) holds the comparator, the
known-gap list and both tolerances, and the unit suite
`test/source_compare/` runs every committed oracle, SC and PC, through it
with no card — the oracle against itself passes with all nine rows compared
(on a CASSCF-orbital oracle, #117, the six its partition has: act_V's three
rows must be empty there, and must not be empty anywhere else),
and a copy with **one** block moved by ten tolerances fails naming that block
and nothing else, for every block of every oracle (`P y` included), while half
a tolerance still passes. The PC oracle is classified with an **empty**
known-gap list (`kClassifiedOracles`), so the PC entry is held to every
block; `nevpt2_sc_pseudodensity --source-oracle` refuses an oracle of the
other scheme than the run's (`--pc` or not) as a configuration error.

### The tolerance, measured from the PC step ladders

As for SC, the oracle's side is a central difference. Unlike SC there is no
GPU side to converge onto yet, so each ladder is read against its own rungs:
`sc_source_oracle.py --pc --kappa-step h` (CI step held at `CI_STEP_PC`) and
`--ci-step h` (kappa held), each written with `--out` to scratch. Offline
Python, no card: single-threaded, about 150 s a file on the RX 9060 XT box's
CPU. Max \|A(h) − A(2^-19)\| per block:

| block | 2^-11 | 2^-13 | 1e-4 (`KAPPA_STEP_PC`) | 2^-15 | **2^-17 (committed)** | 2^-21 |
|---|---:|---:|---:|---:|---:|---:|
| core–virt | 3.5e-7 | 2.2e-8 | 1.5e-8 | 1.5e-9 | **1.0e-9** | 3.2e-9 |
| act_O–virt | 8.5e-8 | 5.4e-9 | 3.6e-9 | 3.8e-10 | **1.0e-10** | 2.6e-10 |
| core–core | 3.1e-8 | 2.0e-9 | 1.3e-9 | 2.8e-10 | **2.4e-10** | 1.6e-9 |
| act_V–virt | 8.8e-9 | 5.5e-10 | 3.7e-10 | 4.1e-11 | **2.2e-11** | 9.5e-11 |
| virt–virt | 8.5e-9 | 5.5e-10 | 3.8e-10 | 1.7e-10 | **1.1e-10** | 6.7e-10 |

The blocks fall sixteen-fold per two halvings (`h²`) down to `2^-15`, sit at
the quotient's ~1e-9 floor from `2^-17` (2^-17 against 2^-15 is 1.2e-9 in
core–virt) and climb off it at `2^-21`. core–act_O and core–act_V are exactly
0.0 at every rung, and act_O–act_V is zero by symmetry, at most 2.6e-10 of
round-off. So the oracle is taken at **`2^-17`, #77's converged `KAPPA_STEP`,
and not at `KAPPA_STEP_PC`**: 1e-4 is right for the PC *gradient*, which does
not move with it at that gate's scale ("The step constants, measured"), but
read per element it is still 1.5e-8 from converged.

`P y`'s ladder is the PC CI trap's ("The PC CI source term's step is a
first-order trap"): above `CI_STEP_PC` the run is refused (3e-7: the guard at
a displaced CI vector, gap 1.9), and from it down the quotient is already at
round-off, which grows as `1/h`. Max \|P y(h) − P y(h')\| over the 12
determinants of the state's irrep:

| | 1e-7 (**`CI_STEP_PC`, committed**) | 3e-8 | 1e-8 | 3e-9 |
|---|---:|---:|---:|---:|
| 1e-7 | 0 | 1.4e-9 | 4.3e-9 | 1.9e-8 |
| 3e-8 | | 0 | 4.6e-9 | 1.9e-8 |
| 1e-8 | | | 0 | 1.8e-8 |

So the committed file is good to ~1e-9 in every `A` block and a few 1e-9 in
`P y`: nothing in the oracle's own error forces the agreement looser than the
project's 1e-7, and the gate is that:

| | value |
|---|---|
| **the PC source-oracle tolerance** | **1e-7** |

per element, absolute (Eh): twenty times the oracle's worst ladder spread at
the committed steps. Its one home is this row: `apps/sc_pseudodensity/CMakeLists.txt`
reads it at configure time into `NEVPT2_PC_SOURCE_ORACLE_TOL`, beside SC's.
It was set before any GPU PC term existed, so #74's entry measured the
device against it rather than the other way round; a difference above it is
a finding about the device, and widening the row is a ratchet a human
moves. The `2^-17` rung of the kappa
ladder is the committed file byte for byte.

## The directional source-term oracle

The two sections above finite-difference **every** element of `A = Y − Y^T`.
That is seconds on N2 CAS(4,4) and LiF, but acrolein CAS(6,5) has 2,846
rotation pairs: its full PC oracle took 21,136 s to generate in #74's
scratch run, and `nevpt2_grad.py --pc --source both` 24,616 s. So acrolein,
the one low-symmetry case, had no committed oracle. #105 commits a
**directional** one. `A` and `P y` enter the gradient linearly, so a wrong
or missing term in a block moves the projection on a seeded random direction
in that block with probability 1, and the comparison stays block by block —
the resolution `kKnownGaps` keys on. The format, the generator
(`sc_source_oracle.py --directional [--pc]`) and the five committed files
are [`reference-data.md`](reference-data.md), "The directional flavour":
two seeded unit `kappa` directions per non-empty block, three seeded CI
directions (in the state's irrep, orthogonal to it, spin-adapted), each
value the Richardson extrapolant `(4 fd(h/2) − fd(h)) / 3` of the first two
rungs of a four-rung ladder, the ladder stored beside it.

**The budget: each file generates in at most 30 minutes, single-threaded.**
Measured (one at a time, the two-card box's CPU): acrolein PC 582 s, acrolein
SC 42 s, N2 PC 51 s, LiF 11 s, N2 SC 4 s (reference-data.md has the table).
The PC files cost more because every evaluation is a block2 PC solve, and
every displaced CI vector also goes through the PC guard.

On a card, `nevpt2_sc_pseudodensity [--pc] --source-oracle
golden/<case>[_pc]_dir.srcgold` assembles the same `Y_xy` and `y_I` as the
full flavour, carries `A` into the oracle's basis and projects it, and `P y`,
onto the stored directions (`compareSourceDirections` in
`nevpt2.source_compare`). Each block's row is the worst \|GPU − oracle\|
over its directions. The verdicts are the full flavour's (`kKnownGaps` is
empty for every directional file), with two more refusals: a direction with support
outside the block it is labelled with is an IO error
(`checkDirectionBlocks`), and a block that has pairs but fewer than
`kMinBlockDirections` = 2 directions **fails**, so an oracle cannot skip a
block silently. Nine entries:
`nevpt2_sc_pseudodensity_oracle_dir_{n2_ccpvdz_cas44,n2_ccpvdz_cas44_casscf,lif_def2svp_cas46,acrolein_def2svp_cas65,acrolein_def2svp_cas65_casscf}`
and `…_oracle_dir_pc_{n2_ccpvdz_cas44,n2_ccpvdz_cas44_casscf,acrolein_def2svp_cas65,acrolein_def2svp_cas65_casscf}`.
The four acrolein entries are `gpu;slow`, and each is registered only where
its gitignored gradient sidecar (CASCI or CASSCF) has been regenerated (a
configure WARNING otherwise). The acrolein `_casscf` pair is "On CASSCF
orbitals: acrolein" below. The N2 `_casscf` pair is "The source terms on
CASSCF orbitals, measured" (#121), the home of those two entries' numbers.

### The steps and the tolerance, measured from long ladders

The steps are not the full flavour's. A seeded direction is not a bare
determinant, so the PC CI quotient is not held at `CI_STEP_PC` (1e-7, "The
PC CI source term's step is a first-order trap"). Each step was chosen from
**long ladders** of the same generator (`--kappa-rungs` / `--ci-rungs`, up
to 13 rungs from `2^-7` or `2^-8`), each compared with a GPU
`--dump-source-terms` projection (RX 9060 XT, preset `hip`). Worst \|Richardson
− GPU\| per block, by the Richardson pair's first rung:

| case, block | 2^-9 | 2^-11 | 2^-12 | 2^-13 | 2^-14 | 2^-15 | 2^-17 |
|---|---:|---:|---:|---:|---:|---:|---:|
| N2 SC, core–virt | 8.8e-7 | 3.8e-7 | 1.0e-7 | 1.1e-8 | 8.8e-10 | **1.4e-10** | 1.0e-10 |
| N2 SC, `P y` | 6.7e-5 | 2.3e-7 | 1.4e-8 | 9.0e-10 | 7.6e-11 | **2.0e-11** | 2.1e-10 |
| LiF SC, core–virt | 7.7e-8 | 9.9e-9 | 5.4e-9 | 1.6e-9 | 3.1e-10 | **5.2e-11** | 7.3e-10 |
| acrolein SC, act_O–virt | 1.5e-8 | 1.4e-10 | 4.4e-12 | 4.0e-11 | 1.8e-11 | **1.0e-10** | 5.3e-10 |
| acrolein SC, core–act_O | 4.4e-10 | 1.3e-10 | 6.1e-10 | 3.6e-10 | 2.0e-9 | **2.1e-9** | 8.3e-9 |
| acrolein PC, act_O–act_V | 4.2e-5 | 2.1e-7 | 1.6e-8 | **3.8e-9** | 2.6e-9 | 2.3e-9 | — |
| acrolein PC, `P y` | — | 2.8e-10 | **4.2e-10** | 1.2e-9 | 3.0e-9 | 4.7e-9 | 1.7e-8 |
| N2 PC, `P y` | — | — | **3.0e-13** | 6.2e-13 | 1.1e-12 | 4.1e-12 | 7.0e-12 |

(Bold: the committed first rung.) Two regimes meet. Coarse steps carry
truncation, worst at near-poles: N2's core–virt and `P y`, and acrolein PC's
act_O–act_V, which needs `2^-13` to fall below 1e-8. Fine steps carry
round-off divided by `h`, which grows from about `2^-15` on. The committed
steps are where the worst block of each scheme has left the first regime:
**SC: kappa and CI at `2^-15`; PC: kappa at `2^-13`, CI at `2^-12`.** PC's
CI step is one rung inside #104's `CI_LADDER_PC` (`2^-11`), because at
`2^-11` the PC guard refuses one of N2's seeded CI directions. The worst
block of every committed file, against the same dumps:

| file | worst block | \|GPU − oracle\| |
|---|---|---:|
| `acrolein_def2svp_cas65_pc_dir` | act_O–act_V | 3.8e-9 |
| `acrolein_def2svp_cas65_dir` | core–act_O | 2.1e-9 |
| `acrolein_def2svp_cas65_dir` | core–virt | 1.5e-9 |
| `lif_def2svp_cas46_dir` | act_O–act_V | 3.9e-10 |
| `n2_ccpvdz_cas44_dir` | core–virt | 1.4e-10 |
| `n2_ccpvdz_cas44_pc_dir` | core–virt | 1.5e-11 |

So the gates are

| | value |
|---|---|
| **the directional source-oracle tolerance** | **1e-8** |
| **the PC directional source-oracle tolerance** | **2e-8** |

per direction, absolute (Eh): about five times the worst committed value of
each scheme, read off the dumps' ladders above. These rows are the
tolerances' one home: `apps/sc_pseudodensity/CMakeLists.txt` reads them at
configure time into `NEVPT2_SOURCE_DIRECTIONAL_TOL` /
`NEVPT2_PC_SOURCE_DIRECTIONAL_TOL`. They are tighter than the full flavour's
(1e-6 SC, 1e-7 PC) because the Richardson pair removes the `h²` truncation
that the full flavour's single central difference keeps. Widening a row is a
ratchet a human moves.

### What the entries measure

Release, `--tiles 3`, the default digest of each backend, on the RTX 3080
(preset `default`) and the RX 9060 XT (preset `hip`). Max \|GPU − oracle\|
per block, over its directions, for the four SC/N2-PC files. The two cards
agree to the digits shown, except where a row gives both:

| block | N2 SC | N2 PC | LiF SC | acrolein SC |
|---|---:|---:|---:|---:|
| core–core | 1.3e-11 | 5.6e-12 | 5.9e-11 | 5.1e-11 |
| core–act_O | 0 | 0 | 0 | 2.1e-9 |
| core–act_V | 0 | 0 | 0 | 6.1e-20 |
| core–virt | 1.4e-10 | 1.5e-11 | 5.2e-11 | 1.5e-9 |
| act_O–act_V | 2.6e-11 | 2.4e-12 | 3.9e-10 | 5.9e-11 / 5.8e-11 |
| act_O–virt | 3.9e-11 | 3.6e-12 | 1.6e-10 | 1.0e-10 |
| act_V–virt | 3.1e-11 | 2.5e-12 | 3.6e-10 | 6.9e-11 |
| virt–virt | 4.2e-11 | 1.0e-11 | 1.3e-10 | 4.0e-11 |
| `P y` (3 directions) | 2.0e-11 | 3.1e-13 / 3.0e-13 | 1.3e-10 | 4.3e-11 |

Acrolein PC's table, where the cards do differ (by up to 3e-9 in
act_O–act_V), is PC's one home: [`pc-nevpt2.md`](pc-nevpt2.md), "block2's
exact source terms, on a card". Every block of every file passes, with at
least a factor of five to spare (acrolein PC's act_O–act_V on the RX 9060
XT is the closest), and the known-gap list is empty. N2's and LiF's
core–act blocks are zero on both sides, by symmetry, as in the full
flavour. The values match the dump ladders' committed column above to
within a unit in the second digit: the entry and the offline analysis
compute the same projection, from different runs.

ctest walls (the `cpp-tier.sh` run, RTX 3080 / RX 9060 XT): the three
N2/LiF directional entries 1.37 / 2.25 s, 1.54 / 2.37 s and 2.33 / 5.54 s,
the same as their full twins, since both are dominated by the dump's `Y`;
**acrolein SC 175.8 / 388.5 s, acrolein PC 177.7 / 310.3 s**, in which
`Y`'s 2,850 basis directions are 175–385 s. That is why the two acrolein
entries are `slow`, and why they are not `small`: the sanitizer tier would
run the probe at Debug speed (below).

### On CASSCF orbitals: acrolein

`golden/acrolein_def2svp_cas65_casscf[_pc]_dir.srcgold` (#135;
[`reference-data.md`](reference-data.md), "On CASSCF orbitals") are the same
two oracles for acrolein's CASCI-on-CASSCF state. act_V is empty there, so
three rows are `empty` by construction and six blocks are compared. The
protocol was #117's: the GPU comparison ran on the card first, with the two
stems added to `kClassifiedOracles` and empty known-gap lists, and the
classification was kept because both passed on that first run.

Release, `--tiles 3`, the default digest, on the RTX 3080 (preset
`default`), the binary run directly with the ctest entries' flags. Max
\|GPU − oracle\| per block, over its directions (tolerances 1e-8 SC, 2e-8
PC, above):

| block | acrolein CASSCF SC | acrolein CASSCF PC |
|---|---:|---:|
| core–core | 3.0e-10 | 3.1e-10 |
| core–act_O | 3.4e-9 | 7.9e-10 |
| core–act_V | empty | empty |
| core–virt | 3.5e-9 | 4.9e-10 |
| act_O–act_V | empty | empty |
| act_O–virt | 1.4e-10 | 4.3e-11 |
| act_V–virt | empty | empty |
| virt–virt | 2.0e-10 | 1.7e-10 |
| `P y` (3 directions) | 5.3e-11 | 5.3e-10 |

Every non-empty block passes. The closest is SC's core–virt, 3.5e-9 against
1e-8, a factor of 2.9 to spare: less than the factor of five the CASCI files
were set with, and still inside the tolerance, which is not moved. PC is at
least a factor of 25 inside its 2e-8. Wall, the binary run directly: SC
180.6 s, PC 183.8 s, of which `Y`'s 2,850 basis directions are about 180 s,
the same as the CASCI twins. ctest walls in the `cpp-tier.sh --preset
default` run (RTX 3080): SC 180.5 s, PC 182.2 s, both passing.

### The RX 9060 XT segfault, characterised

The first ctest run of the acrolein PC entry, on the RX 9060 XT (`ctest -R
oracle`, eight entries in sequence), ended **`SegFault` at 310.88 s**, its
full normal run length. Its output stopped at exactly the first 4,096-byte
stdio flush (stdout is a pipe under ctest), which comes after `Y` and `y`
were assembled, so the crash came late: in the rest of the printing, at the
verdict line, or at process teardown. It has not recurred:

| | runs | crashes |
|---|---:|---:|
| RX 9060 XT, the entry alone under ctest (a loop, each run under the card's lock) | 10 | 0 |
| RX 9060 XT, inside full ctest sequences (`cpp-tier.sh --preset hip` once, `-R oracle` twice) | 3 | 0 |
| RX 9060 XT, the binary run directly, line-buffered | 3 | 0 |
| RTX 3080, the entry alone under ctest | 10 | 0 |
| RTX 3080, inside `cpp-tier.sh --preset default` | 1 | 0 |

So the RX 9060 XT crashed once in 17 runs (the first, then 16 clean), and the
RTX 3080 never in 11. The sanitizers find nothing in our code:

- **`asan`** (RTX 3080, Debug, host ASan + LSan), both acrolein entries by
  name: pass, no report (405.7 s SC, 572.4 s PC); **`ubsan`** on the acrolein
  PC entry by name: pass, no report (476.9 s);
- every preset's own tier passes, the new unit suites in it: `asan` 207/207,
  `ubsan` 207/207, `compute-sanitizer` memcheck 198/198 and initcheck
  195/195 (RTX 3080), `hip-asan` 205/205 and `hip-ubsan` 205/205 (RX 9060
  XT);
- the three small directional entries (N2 SC, N2 PC, LiF), which run the
  same comparator, printing and teardown, by name in each of the six builds:
  all pass, no report. Seconds, N2 SC / N2 PC / LiF: `asan` 4.1 / 5.0 / 3.7,
  `ubsan` 3.1 / 3.7 / 3.1, memcheck 16.9 / 19.3 / 53.2, initcheck 11.3 /
  13.1 / 22.1, `hip-asan` 13.4 / 14.4 / 38.5, `hip-ubsan` 12.0 / 12.3 / 37.0;
- not completed inside the 30-minute cap, so no verdict: compute-sanitizer
  memcheck on the acrolein PC entry (stopped at 30 min, no error printed
  before it), and `hip-asan` on the acrolein SC entry (stopped at 34.5 min,
  its stdout still buffered, no sanitizer report). With Debug device code,
  `Y`'s 2,850 directions take too long under either.

No core file was obtainable: the container's `core_pattern` pipes to apport,
and the container has no debugger. So every sanitizer is clean, the RTX 3080
never crashed, and the RX 9060 XT crashed once. The entry is kept as it is.
The crash is recorded here as an open, unexplained intermittent on the RX
9060 XT, to be followed up on its own; the ROCm runtime or process teardown
is suspected, not shown.

### Negative controls, and the cross-check against the full flavour

Two scratch patches to `apps/sc_pseudodensity/source_dump.cppm` (never
committed), each built once and run against **every** source oracle, full
and directional (RTX 3080, preset `default`, Release, `--tiles 3`):

- **digest**: the digests' term dropped from the GPU's `Y` (`&hbar` →
  `nullptr` in the two `*FullRotationMatrix` calls, #89's control);
- **eps**: the orbital energies' response dropped from `y` (`&eps` →
  `nullptr` in the two `*CiSourceDevice` calls, #87's term).

Every run FAILS, naming these blocks (max \|GPU − oracle\| in brackets) and
no other:

| oracle | digest dropped | eps dropped |
|---|---|---|
| N2 SC, full | act_O–virt (2.9e-1), act_V–virt (1.1e-2) | `P y` (3.2e-4) |
| N2 SC, directional | act_O–virt (1.4e-1), act_V–virt (1.0e-3) | `P y` (5.6e-4) |
| N2 PC, full | act_O–virt (1.5e-2), act_V–virt (3.8e-3) | `P y` (3.2e-4) |
| N2 PC, directional | act_O–virt (6.4e-3), act_V–virt (6.4e-4) | `P y` (6.1e-4) |
| LiF SC, full | act_O–act_V (1.4e-1), act_O–virt (5.5e-1), act_V–virt (5.9e-2) | `P y` (3.2e-3) |
| LiF SC, directional | act_O–act_V (5.5e-2), act_O–virt (2.9e-1), act_V–virt (2.5e-2) | `P y` (4.8e-3) |
| acrolein SC, directional | core–act_O (3.3e-2), act_O–act_V (1.6e-1), act_O–virt (3.7e-2), act_V–virt (1.2e-3) | `P y` (1.6e-3) |
| acrolein PC, directional | core–act_O (1.4e-3), act_O–act_V (1.4e-2), act_O–virt (2.3e-3), act_V–virt (7.9e-4) | `P y` (9.7e-4) |

**The cross-check:** on N2 CAS(4,4) (SC and PC) and LiF the two flavours
name **the same failing blocks** under both patches, and the same passing
ones; unpatched, both flavours pass every block (below). The full-flavour
rows reproduce the earlier records: N2's and LiF's digest rows are #92's
known gaps to the digit ("What the entries measure" above), and N2 PC's is
[`pc-nevpt2.md`](pc-nevpt2.md)'s negative control 2. On acrolein every
block with an active index fails under the digest patch except core–act_V,
whose reference is ~1e-11 on both schemes; core–act_O, zero by symmetry on
N2 and LiF, is not zero on acrolein's lower symmetry, and it is the block the
directional oracle adds to what the full ones could check.

### Acrolein's `P y`: the #103/#104 finding, resolved

#103 recorded acrolein's PC `P y` 6.5e-7 from PySCF's at `CI_STEP_PC`, growing
as `1/h` at smaller steps ([`pc-nevpt2.md`](pc-nevpt2.md), "The nuclear
gradient from the GPU's terms"). Along spin-adapted directions at `2^-12`
the GPU's `P y` agrees with block2's on both cards more than forty times
inside the PC directional tolerance (the figures are PC's, in
[`pc-nevpt2.md`](pc-nevpt2.md), "block2's exact source terms, on a card").
The 6.5e-7 was the oracle's round-off: ~100–400 ulp of `E_corr(PC)`
per evaluation on this state ("PC's CI-direction ladders, measured") divided
by a 1e-7 step. No PC known gap is needed. #74's four acrolein ladder
run-length failures (`y` linearity and CI stationarity, seeds 70 and 2026)
difference the same noisy functional and were not re-run; they stay on
record in [`pc-nevpt2.md`](pc-nevpt2.md), an offline diagnostic.

### The hours-long runs are optional diagnostics

Acrolein's full-`A` oracle sweep (`sc_source_oracle.py --case
acrolein_def2svp_cas65 [--pc]` without `--directional`; 21,136 s for PC) and
its `nevpt2_grad.py [--pc] --source both` gradient gate (1,290 s SC, 24,616 s
PC) are **optional diagnostics**, not gates. Their recorded results (#73,
#89, #74: "The GPU-sourced nuclear gradient" above and
[`pc-nevpt2.md`](pc-nevpt2.md), "The nuclear gradient from the GPU's terms")
stay on record. Acrolein's regression gate is the two directional entries.

## Residual-independent checks of the gradient path

**A converged residual is not evidence that a gradient is right**, and the
gradient work proved it five times: every solve reported a residual at
round-off while the gradient was wrong by a margin a user would have shipped
(#40's four, [`reference-data.md`](reference-data.md), "The analytic
gradient"; #41's one, "The PC CI source term's step is a first-order trap"
above). Since #70 every solve and every difference quotient of the gradient
path also has a check that **reads no residual**, and a failing one fails the
run.

**In `reference_data/nevpt2_grad.py`** (offline, so this is the same manual
gate as the section above, and needs no card): eleven checks after the
multipliers and the AO contraction, printed as their own `PASS:`/`FAIL:` line
beside the `grad_num` one, exit 1 on a failure, `--show-ladders` for every
table.

| check | what it holds | the solve or quotient it covers |
|---|---|---|
| projection of `y` | `\|c·y\| <= 1e-6`: `y` IS the projected gradient | the `y_I` quotients |
| projection of `z` | `\|c·z\| <= 1e-6`: `z` IS projected | the CI Z-vector solve |
| `y` linearity, two seeds | `<u, y>` against a step ladder of `E_corr` along `u`, through `Functional.evaluate` — not through `corr_at_ci`, the route `y` was built on | the `y_I` quotients, `CI_STEP` / `CI_STEP_PC` |
| CI stationarity, two seeds | the Lagrangian's CI part, laddered along `u` at the solved multipliers, has zero slope (its three pieces laddered one by one and summed when the whole shows no run: "PC's CI-direction ladders, measured") | the CI Z-vector solve and the CI/(O2) coupling |
| orbital stationarity, two seeds | the Lagrangian at fixed `c`, laddered along `C exp(t kappa)` over every non-gauge rotation — the set derived from the RHF occupied projector, **not** the solve's own `kappa_pairs` | the orbital Z-vector solve, `KAPPA_STEP`'s `A`, `M` and `Q` |
| nuclear ladder, two seeds | `Phi` laddered along a totally symmetric displacement against `<d, g>` | `NUC_STEP`'s quotients and their Richardson pass |
| PC guard at every displaced CI vector | a ledger: every PC energy at a CI vector other than the state's own ran the conditioning guard (counted in a different place from the evaluation), and no guarded point **kept a mode the reference state drops** (a block's smallest kept singular value under `1e-2` of the reference's) | every displaced PC quantity; orbital and nuclear ladder rungs are guarded too, as a sample |

**The ladder rule** (`step_ladder`, the same rule as the C++ checks'
`nevpt2::longestFourfoldRun`, which `test/gradient_plan/`'s
`FourfoldRunTests` pin card-free): central differences at `h = 2^-k`, the
change between neighbouring rungs must fall fourfold — ratio in (3.2, 4.8) —
for at least **3 consecutive halvings**; the **finest** such run is used, and
the Richardson extrapolant at its finest pair must sit within **1e-6** (Eh
per unit direction — `GRAD_NUM_TOL`'s value) of what is being checked. The
run is read off the ladder alone, so it says only that the quotient has
converged; comparing a Richardson value rather than the sweep's smallest
`|fd − ref|` is what stops a wrong reference passing on the rung where `c h²`
happens to cancel its error. The directions are seeded, orthogonal to the
state, spin-adapted and totally symmetric (`ci_check_direction`,
`kappa_check_direction`, `nuclear_check_direction` say why each has to be).

### Each of the five errors, reintroduced

One at a time, in a scratch copy of the script (none of it committed),
single-threaded on the CUDA devcontainer's CPU (the RTX 3080 box), PySCF
2.14.0, block2 0.5.4. Errors 1–4 on LiF CAS(4,6) SC, whose unmodified run
passes all eleven; error 5 on N2 CAS(4,4) PC, likewise. Every residual stayed
at round-off and every one of them was caught:

| error | caught by (value; the gate is 1e-6) | residuals at the time | gradient vs `grad_num` |
|---|---|---|---|
| 1. act_O–act_V taken for a gauge (#40) | orbital stationarity: 2.4e-5 and 2.2e-4 | CI 1.4e-12, orbital 2.2e-15 | 4.3e-4 |
| 2. `z` solved projected but not projected (#40) | projection of `z`: 0.335 | CI 1.4e-12, orbital 1.0e-15 | 3.9e-3 |
| 3. `y` as a componentwise quotient (#40) | projection of `y`: 2.04; `y` linearity 8.7e-5 / 1.5e-4; CI stationarity the same | CI 1.4e-12, orbital 1.0e-15 | 4.1e-6 |
| 4. `y` without the orbital energies' CI derivative (#40) | `y` linearity 6.1e-3 / 4.8e-3; CI stationarity the same | CI 1.3e-12, orbital 1.1e-15 | 7.9e-5 |
| 5. the PC CI quotient at SC's step, 1e-4 (#41) | `y` linearity 1.8e-3 / 1.6e-3; CI stationarity the same; projection of `y` 4.6e-4; the guard's rank check — 44 guarded points kept a mode the reference drops, worst `iabcpqg*` at 5.4e-12 against the reference's 9.8e-4 | CI 1.5e-15, orbital 8.7e-16 | 2.5e-4 |

Error 3 is the one the gradient itself barely shows on LiF (4.1e-6), and
error 5 is the one #41 had to find by comparison: its gap guard passed (the
lifted modes clear the cut outright at 1e-4), and the new rank check is what
sees it at the guard. A sixth scratch patch — the PC guard skipped at
displaced CI vectors, everything else intact — leaves the gradient right
(1.5e-8) and fails only the ledger: `guarded 0 of 88 displaced CI vectors`.

### What the committed cases come out at

Same machine and versions. Worst `|Richardson − ref|` over each pair of
seeds, against the 1e-6 gate. The SC columns are at the SC steps #77
converged ("The step constants, measured", below); the last column is N2 SC
at the steps before it, kept because it is the finding the next section
records:

| | LiF CAS(4,6) SC | acrolein CAS(6,5) SC | N2 CAS(4,4) PC | N2 CAS(4,4) SC | N2 SC, before #77 |
|---|---|---|---|---|---|
| projection of `y` / of `z` | 3.8e-11 / 8.3e-20 | 3.5e-11 / 4.7e-18 | 1.0e-10 / 5.4e-19 | 6.3e-9 / 1.2e-18 | **1.1e-6** / 5.4e-19 |
| `y` linearity | 3.7e-10 | 6.9e-11 | 2.2e-10 | 7.1e-9 | **1.2e-6** (8.0e-7) |
| CI stationarity | 3.5e-10 | 7.8e-11 | 2.3e-10 | 7.2e-9 | **1.2e-6** (8.0e-7) |
| orbital stationarity | 1.5e-7 | 3.9e-7 | 9.6e-9 | 8.2e-7 | 5.8e-7 |
| nuclear ladder | 1.2e-9 | 2.2e-9 | 1.0e-9 | 9.2e-9 | **2.5e-6** |
| PC guard | — | — | 88 of 88 guarded, 116 rungs sampled, rank unchanged, smallest gap 3.9e9 | — | — |
| verdict | PASS, 11 of 11 | PASS, 11 of 11 | PASS, 11 of 11 | PASS, 11 of 11 | **FAIL, 6 of 11** |
| the checks' own wall | 14 s | 50 s | 34 s | 3 s | 3 s |

At the pre-#77 steps LiF and acrolein SC also passed 11 of 11 (worst
orbital stationarity 7.1e-8 and 2.2e-7; every other row at or below 1.0e-8);
N2 PC's steps did not change, and its re-run reproduces every figure in its
column. Acrolein SC is gated against a fresh `nevpt2_numgrad.py --pc --npz`
(its sidecar is gitignored). Acrolein PC's checks have **never been run**:
its steps did not change either, so it was not re-run for #77, and it was not
re-run for #70 — it is the 6.3 h pair above. #74 ran them for the first
time, with the GPU's PC source terms: 7 of 11 hold, the four CI-side ladders
failing on run length — an open finding ([`pc-nevpt2.md`](pc-nevpt2.md), "The
source terms by reverse mode"). N2 PC's one such ladder with the GPU's terms
was diagnosed and resolved by #104 ("PC's CI-direction ladders, measured",
below); acrolein's is #105's.

**Two of these pass with little margin, and both are the ladders' own
limits on N2, not the steps':**

- **orbital stationarity on N2 SC, 8.2e-7.** N2's orbital ladder is not
  cleanly fourfold below `2^-11`, at the old steps or the new ones (5.8e-7
  before #77), so its Richardson value carries a few 1e-7 of its own.
- **the nuclear ladder on N2 SC, a run of exactly 3** (ratios 3.21, 3.70,
  3.30 to `2^-16`, seed 70) — the slow convergence of N2's frozen path the
  next section describes. Below `2^-16` the ladder is at round-off (changes
  of 1.9e-9, 3.7e-9, 1.3e-8 at `2^-17`, `2^-18`, `2^-19`, measured in a
  scratch copy with the ladder extended), so it cannot be lengthened by
  extending it. Its Richardson value is 9.2e-9 from the gradient, but the
  run length is at the rule's edge: in the step scans below, seed 70's last
  ratio drops to 3.19 at two neighbouring CI steps (`2^-15`, `2^-20`) and
  one nuclear step (`2^-16`), and the check fails there on run length alone.
  The density-fitted twin falls off that edge at the committed steps:
  "The density-fitted gradient against the numerical one", above.

### N2 CAS(4,4) SC fails, and that is the finding

**Resolved by #77** — the SC steps changed, PC's did not; the record below
is kept as it was found.

N2 SC is the run "What is run" above recorded at **2.5e-7** from `grad_num`.
Its ladders said that number was not converged on the steps it was taken at:

- **`NUC_STEP` (0.0025 bohr).** Along the symmetric stretch the frozen path's
  `Phi` converges only slowly — ratios 2.9, 3.2, 3.7, 4.2 down to `2^-16`
  bohr, where LiF's sits at 4.0 ± 0.1 from `2^-5` — and its converged slope
  is **2.5e-6** from the gradient's Richardson pair at 0.0025 / 0.00125. The
  SC energy has a near-singularity on N2's frozen path, within ~1e-4 bohr; the
  CASCI energy alone on the same path (one atom along `z`) is clean, ratio
  4.00, and the
  re-converged energy `grad_num` differences is too — `nevpt2_numgrad.py` at
  a 20× smaller step moves `grad_num` by 1.7e-8.
- **`CI_STEP` (1e-4).** `y` is **1.2e-6** off its own ladder (seed 70; 8.0e-7
  for seed 2026), and `|c·y|` is 1.1e-6 where LiF's is 4.4e-11: the SC energy
  along a CI direction has a near-pole at `h ~ 2^-7`, and 1e-4 is not far
  enough inside it.

Rerunning the same case with the steps the ladders converge at (scratch
copies again, not committed) made the point: `NUC_STEP = 2^-13` alone moves
the gradient to **1.5e-6** from `grad_num` (a FAIL), `CI_STEP = 2^-17` alone
to 7.8e-7, and both plus `KAPPA_STEP = 2^-17` to **4.4e-8** — with all
eleven checks passing (the orbital ladder closest to its gate, at 8.2e-7: on
N2 it is not cleanly fourfold below `2^-11` either). So the shipped 2.5e-7 was
two step errors of ~1e-6 partly cancelling, and the converged answer agrees
with `grad_num` better than the shipped one does. #70 did not change the
steps — that moves every recorded gradient in this section — and until #77
`nevpt2_grad.py --case n2_ccpvdz_cas44` exited 1 on these checks while its
`grad_num` line read `PASS`: the residual-independent checks doing what they
are for.

**What resolved it** (#77): the SC steps are now those three — `CI_STEP =
KAPPA_STEP = 2^-17`, `NUC_STEP = 2^-13` bohr — each read off a ladder of the
gradient itself in "The step constants, measured", below. The case passes
all eleven checks and lands **4.4e-8** from `grad_num`. Re-measured with
the committed constants (no step flags), the CI-side checks move by more
than a factor of 100 — projection of `y` 1.1e-6 → 6.3e-9, `y` linearity and
CI stationarity 1.2e-6 → 7.1e-9 / 7.2e-9 — and the nuclear ladder 2.5e-6 →
9.2e-9. The `KAPPA_STEP` change was not in #70's diagnosis
and is load-bearing: with `CI_STEP` and `NUC_STEP` converged but `KAPPA_STEP`
left at 1e-4 the gradient is 9.8e-7 from `grad_num`, inside the gate by 2%.
`nevpt2_numgrad.py`'s own step (and so every `grad_num`, the committed
sidecars' bytes included) is unchanged: #70 measured it moving `grad_num` by
1.7e-8 at a 20× smaller step, and nothing here contradicts that.

### The step constants, measured

**Every step constant is read off a ladder of the gradient itself**: the
whole `nevpt2_grad.py` run repeated with one step walked through `h = 2^-k`
and the other two held at their chosen values, recording the gradient's
change between neighbouring rungs (the ratio column is the previous change
over this one: 4 is `h²` truncation, below 1 is round-off) and its distance
from `grad_num`. Run with `--ci-step`, `--kappa-step` and `--step`, so any
rung can be repeated; same machine and versions as above.

**N2 CAS(4,4) SC**, the case that needed it (others held at `CI_STEP =
KAPPA_STEP = 2^-17`, `NUC_STEP = 2^-13`):

| k | `CI_STEP = 2^-k`: change, ratio | max \|g − grad_num\| | `KAPPA_STEP = 2^-k`: change, ratio | max \|g − grad_num\| |
|---|---|---|---|---|
| 9 | — | 1.2e-3 | — | 2.3e-6 |
| 10 | 1.3e-3 | 5.1e-5 | 7.3e-6 | 5.1e-6 |
| 11 | 3.8e-5 | 1.3e-5 | 4.6e-6, 1.6 | 4.8e-7 |
| 12 | 9.5e-6, 4.05 | 3.2e-6 | 1.9e-6, 2.4 | 1.4e-6 |
| 13 | 2.4e-6, 4.01 | 8.3e-7 | 2.9e-7, 6.7 | 1.2e-6 |
| 14 | 5.9e-7, 4.00 | 2.4e-7 | 5.2e-7, 0.55 | 6.3e-7 |
| 15 | 1.5e-7, 4.00 | 9.0e-8 | 3.3e-7, 1.6 | 3.0e-7 |
| 16 | 3.7e-8, 4.00 | 5.3e-8 | 1.9e-7, 1.8 | 1.1e-7 |
| **17** | 9.3e-9, 3.99 | **4.4e-8** | 7.1e-8, 2.7 | **4.4e-8** |
| 18 | 2.3e-9, 4.01 | 4.2e-8 | 2.0e-8, 3.5 | 2.4e-8 |
| 19 | 6.0e-10, 3.84 | 4.1e-8 | 1.6e-9 | 2.5e-8 |
| 20 | 7.3e-11 | 4.1e-8 | 1.4e-8, 0.12 | 1.2e-8 |
| 21 | 4.4e-10, 0.17 | 4.0e-8 | 3.1e-8, 0.44 | 2.0e-8 |

| k | `NUC_STEP = 2^-k` bohr: change, ratio | max \|g − grad_num\| |
|---|---|---|
| 7 | — | 3.2e-5 |
| 8 | 3.2e-5 | 5.4e-7 |
| 9 | 2.3e-6, 13.9 | 1.8e-6 |
| 10 | 9.7e-7, 2.4 | 7.8e-7 |
| 11 | 6.2e-7, 1.6 | 1.6e-7 |
| 12 | 1.8e-7, 3.6 | 1.6e-8 |
| **13** | 2.9e-8, 6.1 | **4.4e-8** |
| 14 | 4.0e-9, 7.1 | 4.7e-8 |
| 15 | 1.1e-9, 3.5 | 4.6e-8 |
| 16 | 6.8e-9, 0.17 | 5.2e-8 |
| 17 | 2.0e-8, 0.34 | 6.2e-8 |

What each says:

- **`CI_STEP = 2^-17`.** A clean `h²` ladder — ratio 4.00 from `2^-12` to
  `2^-18` — that has converged to 9e-9 at `2^-17`, one rung above where
  round-off takes over. The old 1e-4 (between `2^-13` and `2^-14`) sat
  ~1e-6 up it. The near-pole at `h ~ 2^-7` is visible as the 1.2e-3 at
  `2^-9`.
- **`KAPPA_STEP = 2^-17`.** **Not** an `h²` ladder anywhere on N2: the
  changes fall erratically and far slower than fourfold (ratios 1.6, 2.4,
  6.7, 0.55, 1.6, 1.8, 2.7, 3.5) from 7.3e-6 down to 2.0e-8 at `2^-17 →
  2^-18`, and are round-off below that. `2^-17` is the last step at which
  the change is still falling; past it the gradient moves by up to 3e-8 from
  rung to rung, all of it noise, 30× inside the gate. The old 1e-4 was ~1e-6 off. Why the
  orbital quotient converges so much more slowly than `h²` on N2 is not
  established here — this case's six exactly degenerate inactive pairs
  ("The deliberately symmetric case") are the obvious suspect — and the
  orbital-stationarity ladder sees the same thing (above).
- **`NUC_STEP = 2^-13` bohr.** Erratic above `2^-11` (the near-singularity
  within ~1e-4 bohr), settled from `2^-12` down: 4.0e-9 to the next rung at
  `2^-13`, 1.1e-9 at `2^-14 → 2^-15`, round-off below. Any of `2^-13` to
  `2^-15` gives 4.4e-8 to 4.7e-8; `2^-13` is the largest of them, the one
  least exposed to round-off on a bigger molecule. The old 0.0025 bohr is
  about `2^-8.6`.

**LiF and acrolein SC at the new steps**, the same runs with no step flags
against the old steps' results: the gradient moves by at most 1.4e-8 (LiF)
and 1.2e-7 (acrolein) per component, each lands about as close to
`grad_num` as before (8.3e-9 against 5.3e-9; 6.5e-8 against 7.6e-8), all eleven
checks pass on both (the table above), and both skeleton bars collapse —
LiF 2.2e-7 → 5.4e-10, acrolein 5.6e-7 → 1.1e-9 — because a converged
quotient's Richardson correction is small. Neither case had a convergence
problem at the old steps; the new ones are not worse for them.

**PC keeps its steps** (`CI_STEP_PC` 1e-7, `KAPPA_STEP_PC` 1e-4,
`NUC_STEP_PC` 0.0025 bohr), and that is a measured choice, not an omission.
On N2 CAS(4,4) PC, against one `grad_num_pc`:

| `KAPPA_STEP_PC` / `NUC_STEP_PC` | 1e-4 / 0.0025 (kept) | `2^-17` / 0.0025 | 1e-4 / `2^-13` | `2^-15` / `2^-13` | `2^-17` / `2^-13` | `2^-19` / `2^-13` | `2^-17` / `2^-11` | `2^-17` / `2^-15` |
|---|---|---|---|---|---|---|---|---|
| max \|g − grad_num_pc\| | **1.5e-8** | 1.6e-8 | 1.6e-8 | 1.8e-8 | 1.7e-8 | 2.4e-8 | 1.7e-8 | 1.6e-8 |
| checks | **11 of 11** | 10 of 11 | 11 of 11 | 11 of 11 | 10 of 11 | 11 of 11 | 10 of 11 | 10 of 11 |

Every PC gradient sits within 1e-8 of every other, at `grad_num_pc`'s own
level — PC's frozen path and orbital quotients carry no step error the gate
can see, so there is nothing for a change to gain. And at `KAPPA_STEP_PC =
2^-17` specifically, the SC choice, one PC check fails: CI stationarity,
seed 2026, whose slope is 8e-11 — round-off, as stationarity should be —
and whose ladder hits that floor at `2^-15` (a change of 5.0e-12, ratio
5.2) one halving earlier than at the kept steps (6.4e-12, ratio 4.07), so
its fourfold run is 2 long rather than 3. That is the ladder rule reading
round-off, not a wrong gradient, but it is a check the kept steps pass and
the SC ones do not. So the PC steps stay where every PC figure in this file
was measured, and **acrolein PC's 6.3 h pair is not re-run**: none of its
steps changed.

### A ladder can converge, cleanly, to the wrong branch

The PC CI ladders walk a **one-electron** spin-adapted direction (`sum w_pq
E_pq |c>` over the pairs the state's `dm1` couples — the C++ PC tangent's
`pcTangentDirection`) and start at `2^-11`, and neither is a free choice.
Measured on N2 CAS(4,4) PC:

- along a **two-electron** spin-adapted, totally symmetric direction, one of
  `Sir`'s null modes (`irabpq1`) lifts as `h²` — smallest kept singular value
  4.9e-8 at `2^-10`, 1.2e-8 at `2^-11` — and stays above lstsq's cut over the
  whole ladder. The ladder then converges **cleanly, ratio 4.00 for ten
  halvings**, to a value **3.04e-6** from `<u, y>`, with the gap guard
  passing at every rung (gap 8e3 to 1e9): #41's trap, met by a ladder. The
  rank check is what catches it; the gap check cannot.
- along the one-electron direction the mode lifts as `h⁴` — kept at
  1.4e-9 at `2^-5`, astride the cut at `2^-10` (the gap guard refuses one
  seed there, gap 8.1e2) and dropped from `2^-11` down. There the ladder
  agrees with `<u, y>` to **2.2e-10**.

### PC's CI-direction ladders, measured

#103 left N2 CAS(4,4) PC at 10 of 11: CI stationarity, seed 2026, two
fourfold halvings where the rule wants three (Richardson 8.3e-13 from zero).
#104 measured why, on N2 CAS(4,4) PC, from both providers: GPU-sourced
(`--source both`, PR #103's RTX 3080 dump) and PySCF-sourced (`--source
pyscf`). A scratch copy of the checks extended both ladders to `h = 2^-8 …
2^-26` and laddered the CI Lagrangian's three pieces one by one. Single-threaded
Python, PySCF 2.14.0, block2 0.5.4, no card.

**The ladder's start is fixed by the PC metric, not chosen.** Above `2^-11`
the guard refuses a rung: `irabpq1`'s gap is 4.7e2 at `2^-9` (seed 70) and
8.1e2 at `2^-10` (seed 2026). Its rank check flags `2^-8 … 2^-10`. So
`CI_LADDER_PC` cannot start any coarser.

**The ladder's end is double-precision round-off, at machine level on N2.**
Read off the round-off rungs as `|change| × h`, each evaluation's noise is:

| quantity laddered along the CI direction | noise per evaluation (Eh) |
|---|---:|
| `E_corr(PC)` (`y` linearity) | 1e-17 – 4e-17 (`E_corr` is 0.235 Eh: about one ulp) |
| `<z\|H\|c> − E_CASCI <z\|c>`, `ecore` inside `H\|c>` | 1e-16 – 2e-16 |
| the same, active-space `H\|c>` only | ~1e-17 |
| `lambda · G` | ~1e-18 |

So on N2 nothing is block2-specific. The worst term was the check's own: the
inactive constant `ecore` (~1e2 Eh) entered `<z|H|c>` and `E_CASCI <z|c>`
identically. It cancels exactly, so it added nothing but `|z| ecore eps` of
round-off, ten times the rest of the Lagrangian's.

**Why the stationarity ladder is the short one.** The Lagrangian's slope is
zero because its pieces cancel, and most of their truncation cancels with
them. The round-off does not cancel. Changes at `2^-12`, seed 2026,
PySCF-sourced:

| `E_corr` | `<z\|H − E\|c>` | `lambda · G` | the Lagrangian |
|---:|---:|---:|---:|
| 1.65e-8 | 1.69e-8 | 4.9e-11 | **4.08e-10** |

The sum's truncation is 40 times smaller than its pieces', so it reaches the
floor about two halvings sooner. The fourfold runs:

| | GPU-sourced | PySCF-sourced |
|---|---|---|
| `y` linearity, seed 70 / 2026 (one ladder; only the reference differs) | 4 / 5 | 4 / 5 |
| CI stationarity, seed 70 / 2026, `ecore` inside (PR #103's run) | 4 / **2** | pass / 3 |
| the same, re-run for #104 | 4 / 3 (ratio 4.48 at `2^-15`) | 3 / **2** (ratio 2.93 at `2^-15`) |
| CI stationarity, active-only `H\|c>` (committed) | 4 / 4 (3 in a second run) | 4 / 3 (both runs) |
| … its pieces alone, seed 2026: `E_corr` / `lambda · G` / `z` term | 5 / 4 / 6 | 5 / 4 / 6 |

Which provider lands on the rule's edge changes between runs. In PR #103
the GPU-sourced run failed; in the re-run the PySCF-sourced one did, with
the same rule and the same inputs. That is the signature of a ladder whose
round-off rung decides the count. Every Richardson value stayed at most
8.1e-11 from zero, against the 1e-6 allowance.

**The resolution, in `check_ci_stationarity`.** `FOURFOLD_RUN`, the 1e-6
allowance, `CI_STEP_PC` and `CI_LADDER_PC` are unchanged. The check changed
in two ways:

1. **The `z` term is formed from the active-space `H|c>`** (`hvec_act`),
   without `ecore`. This is the same number in exact arithmetic, with a
   floor ten times lower.
2. **When the whole Lagrangian shows no run of three, its pieces are
   laddered one by one.** Each piece is held to its own run of three and
   its own Richardson value, and those values must sum to zero within 1e-6.
   The slope is linear in the pieces, so this is the same statement. Each
   piece shows 4–6 halvings on N2, where the sum shows 3. This fallback
   exists for the edge case above, and SC never reaches it.

With both changes, N2 CAS(4,4) PC passes 11 of 11 from both providers, and
each provider's gradient is unchanged (1.86e-8 and 1.51e-8 from `grad_num`).
Read by pieces on both seeds, the sums are 2.6e-12 / 1.9e-12 (GPU) and
2.2e-10 / 7.5e-11 (PySCF).

**SC is unchanged.** LiF CAS(4,6), acrolein CAS(6,5) and N2 CAS(4,4) SC
each pass 11 of 11, on the whole Lagrangian (CI stationarity 9 / 9
halvings on LiF, 10 / 9 on acrolein, 8 / 9 on N2), so the fallback is never
reached. LiF's column of "What the committed cases come out at" reproduces
(CI stationarity 3.5e-10 / 2.2e-10). The worst CI stationarity moves from
7.8e-11 to 7.0e-11 on acrolein, and from 7.2e-9 to 7.1e-9 on N2 SC.

**Negative controls** (scratch, `--source dump`): the dump's `y` scaled on
the 4 in-sector determinants whose alpha string is in irrep 1, which is the
dominant determinant's.

| `y` scaled by | `\|dy\|` | `y` linearity | CI stationarity, whole | … by pieces (forced) | gradient vs `grad_num` |
|---|---:|---|---|---|---|
| 1.01 | 1.4e-3 | **FAIL** 4.7e-4 / 4.1e-4 | **FAIL** 4.7e-4 / 4.1e-4 | **FAIL** 4.7e-4 / 4.1e-4 | **FAIL** 1.1e-4 |
| 1.0001 | 1.4e-5 | **FAIL** 4.7e-6 / 4.1e-6 | **FAIL** 4.7e-6 / 4.1e-6 | **FAIL** 4.7e-6 / 4.1e-6 | **FAIL** 1.08e-6 |

The CI-direction checks catch a `y` error at the size where the gradient
itself only just fails, and the by-pieces reading catches it equally.

**N2 CAS(8,8) PC did not fit #104's 30-minute budget per run.** Its PC
`grad_num` took 233 s. But `--source pyscf` was still inside PySCF's `y_I`
quotients when it was stopped at 30 minutes. The one CAS(8,8) dump at hand
(PR #103's) is refused at load: its `E_corr` is 2.7e-10 from the rebuilt
state's, against `REF_STATE_TOL`. So its CI ladders are unmeasured.

**Acrolein CAS(6,5) PC is not re-run here** (its `--source both` is 6.8 h;
#105 re-checks it through the directional oracle). Its four failures are not
all of N2's kind. Its two failing `y`-linearity ladders difference
`E_corr(PC)` alone, so neither change above touches them. Their round-off
rungs in PR #103's table (seed 70: changes 3.4e-10 … 2.8e-9 at `2^-14 …
2^-16`) put each evaluation's noise near 1e-14 to 4e-14 on a 0.547 Eh
energy. That is roughly 100–400 ulp, against N2's one. So acrolein's PC
reference functional is itself noisy along CI directions, which supports
#103's hypothesis there. It also meant a PC regime for acrolein's CI
directions was #105's to measure, and #105 measured it: along spin-adapted
directions the PC guard accepts `2^-12`, and there the GPU's `P y` agrees
with block2's two decades inside the 1e-7 gate ("The directional
source-term oracle", "Acrolein's `P y`: the #103/#104 finding, resolved").

### The C++ side

The same rule, shared: `nevpt2::longestFourfoldRun` (`nevpt2.gradient_plan`)
is now what the SC tangent (`--sc-tangent`, a run of 4), the active-rotation
and full orbital source terms (`--kappa-act`, `--kappa-full`, runs of 3) and
the PC tangent gate on. The PC tangent's gate (above) gained the two rows the
issue asked for: a **run** of at least two fourfold halvings — so a check of
one halving, or of none, no longer passes on its exact identities alone, and
`--fd-steps` is 3..8 — and the gap check's record at **every** displaced
state. Both are host logic, pinned card-free in `test/pc_tangent/`
(`PcTangentLadderTests`) and `test/gradient_plan/` (`FourfoldRunTests`);
the three `nevpt2_pc_tangent_*` entries pass under the new gate unchanged.

## Sanitizers

Some bugs show up only some of the time — a call left on the legacy stream
racing the one stream, or a stream-ordered free running before the last
kernel that reads the buffer. Repeated runs are weak evidence against those:
a race that fails 10% of the time survives 20 runs ~12% of the time. So the
tree has a sanitizer tier: five presets, small golden cases to run them on,
and canaries that prove each tool is actually on.

### Running them

```bash
# CUDA (RTX 3080)
cmake --preset asan  && cmake --build --preset asan  -j && ctest --preset asan
cmake --preset ubsan && cmake --build --preset ubsan -j && ctest --preset ubsan
cmake --preset compute-sanitizer && cmake --build --preset compute-sanitizer -j
ctest --preset compute-sanitizer                                  # memcheck (the default tool)
cmake --preset compute-sanitizer -DNEVPT2_COMPUTE_SANITIZER_TOOL=initcheck && ctest --preset compute-sanitizer
# (racecheck / synccheck the same way; switching tool needs no rebuild)

# HIP (RX 9060 XT): host only
cmake --preset hip-asan  && cmake --build --preset hip-asan  -j && ctest --preset hip-asan
cmake --preset hip-ubsan && cmake --build --preset hip-ubsan -j && ctest --preset hip-ubsan

# or through the tier script, which picks the test preset of the same name:
devtools/cpp-tier.sh --preset asan
```

Every sanitizer option defaults OFF, so `default`, `hip` and the other
presets are unaffected. Each sanitizer preset is a Debug build in its own
`build-<preset>/` directory, and its test preset runs only the entries
labelled `small`, the whole unit tier (`unit`) and the canaries that build
registered. The `compute-sanitizer` test preset drops the `death` suites:
`EXPECT_DEATH` forks, and compute-sanitizer does not follow the child. Those
suites still run under the four host presets.

**What a change touching streams, allocation or frees must be clean under:**
`asan`, `ubsan`, `hip-asan`, and `compute-sanitizer` with memcheck and with
initcheck, every canary green.

**memcheck's leak check is on by default**
(`NEVPT2_COMPUTE_SANITIZER_LEAK_CHECK`, ON; inert unless compute-sanitizer
runs memcheck). Both demos leak 0 device allocations, so a buffer that
outlives its context is a failing test: fix the leak, don't turn the check
off.

**LSan suppressions** go in `cmake/nevpt2_lsan.supp` and must name a vendor
library, never one of our frames. Today it holds one family: the ROCm
runtime leaks ~105 host allocations (~6 KB) of its own at exit under
`hip-asan`, with stacks entirely inside `libhsa-runtime64.so` /
`libamdhip64.so`. The CUDA runtime reports none.

### What each tool does, and what it does NOT catch

| tool | preset(s) | catches | does NOT catch |
|---|---|---|---|
| **AddressSanitizer** (+ LeakSanitizer) | `asan`, `hip-asan` | host heap/stack/global out-of-bounds, host use-after-free, host leaks at exit | anything in device memory or device code. Device code is uninstrumented on both backends: nvcc has no device ASan, and on HIP device ASan needs an `xnack+` target ID, which gfx1200 does not have (below) |
| **UBSan** (`-fno-sanitize-recover`) | `ubsan`, `hip-ubsan` | host signed overflow, misaligned/null access, bad shifts, invalid enum/bool loads, ...; the first finding aborts | device code (clang does not support `-fsanitize=undefined` for `amdgcn`, and nvcc has none) |
| **compute-sanitizer memcheck** | `compute-sanitizer` | device out-of-bounds and misaligned accesses, accesses to freed device memory (including a **stream-ordered** `wwrFreeAsync` that precedes the read in stream order), CUDA API errors, and with the leak check, device allocations never freed | uninitialized reads (initcheck's job); host memory; a **cross-stream race** (below) |
| **initcheck** | `compute-sanitizer` + `-DNEVPT2_COMPUTE_SANITIZER_TOOL=initcheck` | device global-memory reads of bytes nothing wrote | races, out-of-bounds |
| **racecheck** | ... `=racecheck` | shared-memory hazards *between threads of one block* (RAW/WAR/WAW on `__shared__`) in every kernel of ours (cuBLAS's are excluded, below) | anything in global memory, and anything *between kernels or streams* |
| **synccheck** | ... `=synccheck` | invalid barrier use *inside a kernel*: warps of one block at different `__syncthreads()`, bad `__syncwarp` masks | stream or event ordering between kernels |
| *(none, on HIP)* | | | ROCm has no compute-sanitizer counterpart; HIP gets host ASan/UBSan only |

**Nothing here detects a cross-stream race directly.** racecheck and
synccheck look at shared memory and barriers *inside one kernel*; no tool
checks the ordering *between* streams. memcheck catches a race's
*consequence* only when it is an access to memory already freed at the
moment of the access. Measured on the RTX 3080 with a scratch program: a long
kernel on the legacy stream reading a buffer while `cudaFreeAsync` frees it
on a non-blocking stream, with no ordering between the two, ran clean under
memcheck **3 out of 3 times**. The deterministic form of that bug — a free
that comes *before* the read in stream order — is reported every time (the
`free_before_read` canary), and still reported when the buffer is a
pool-backed `DeviceBuffer` whose freed block the demos' pool keeps mapped
(the `pool_free_before_read` canary). So the tier is strong evidence against "freed
before its last reader on the same stream", and **no evidence** against "a
call left on the legacy stream". That is `devtools/stream-lint.sh --strict`'s
job (see [`performance.md`](performance.md), "One stream").

**Device ASan on gfx1200 is not possible.** `rocminfo` reports `XNACK
enabled: NO`. ROCm 7.2.4's clang rejects the target ID outright ("invalid
target ID 'gfx1200:xnack+'"), and on plain `gfx1200` it ignores
`-fsanitize=address` for the device pass with a warning. So `hip-asan`
instruments host code only, and `add_device_library` adds
`-fno-gpu-sanitize` under the sanitizer options to make that explicit.

**racecheck excludes cuBLAS** (`--kernel-name-exclude kns=cublas`). Unscoped,
`nevpt2_cas88_cublas` failed with 32 hazard sites (~10^7 individual hazards),
every one between a read and a write inside cuBLASLt's
`cublasLt_fused_imma_dgemm_kernel_sm80`, the `--cublas` emulated DGEMM.
Every other entry, including `--blas-digest`'s native cuBLAS DGEMM and every
kernel of ours, was clean. A shared-memory hazard is internal to its kernel,
so no argument of ours can cause one; whether it is a cuBLAS bug or a pattern
racecheck cannot model, it is NVIDIA's, and the `--cublas` results still
match the golden to 1e-7. memcheck and initcheck stay unscoped, because a bad
argument of ours *does* fault inside a vendor kernel. synccheck runs
unscoped too, cuBLAS included, and is clean.

**The compute-sanitizer build leaves out `--cublas`'s declined case.**
`CublasEmulDeclineTests` (`test/cublas/`) asserts that `probe_emul_bits`
returns -1 on a shape cuBLAS declines. cuBLAS declines because its own
workspace `cudaMallocAsync` fails with `out of memory` (see
[`performance.md`](performance.md), "The cuBLAS fixed-point-emulation digest
(`--cublas`)"). memcheck reports each such failed call as an API error, 4 for
that case's two GEMMs, so the entry cannot be clean under it. Nothing we do
causes that failure, and the check is not loosened. The suite is compiled out
of the `compute-sanitizer` build and runs in every other build on the CUDA
card. The engaged cases (`CublasEmulDigestTests`, `CublasEmulProbeTests`) run
under memcheck and initcheck, and both report them clean.

**What the tier has caught.** initcheck found `--cublas`'s engaged-bits probe
reading uninitialized device memory (6,528 errors at CAS(4,4), inside
cuBLAS's `max_scale_pack_ker`): it ran on freshly allocated operands. The
probe now runs on the first tile's operands right after that tile's produce
wrote them (the first tile is always full width, so the probe's `K` is
exactly what was written); results and timings did not change. memcheck's
leak check found 63 (`nevpt2_demo`) and 54 (`nevpt2_df_demo`) device
allocations never freed; every allocation is now a self-freeing
`DeviceBuffer`, and both demos leak 0 at CAS(4,4) and CAS(8,8).

### The small cases

compute-sanitizer multiplies run time (racecheck worst of all), so the tier
runs on small goldens: CAS(4,4) and CAS(8,8) on N2/cc-pVDZ, conventional and
DF, from `generate_golden.py` with its usual single-threaded, 1e-12,
point-group-symmetric settings. All four pass its degeneracy guard,
regenerate byte-identically, and carry PC fields.

```bash
uv run python reference_data/generate_golden.py --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44
uv run python reference_data/generate_golden.py --ncas 4 --nelecas 4 --df --name n2_ccpvdz_cas44_df
uv run python reference_data/generate_golden.py --ncas 8 --nelecas 8 --name n2_ccpvdz_cas88
uv run python reference_data/generate_golden.py --ncas 8 --nelecas 8 --df --name n2_ccpvdz_cas88_df
```

The two conventional cases also carry a gradient sidecar, which the committed
commands add with `--gradgold`
([`reference-data.md`](reference-data.md), "The gradient sidecar"); the flag
writes a separate `*.gradgold` file and leaves these goldens' bytes alone.

| case | ncore | n_det | E_corr (conventional / DF) |
|---|---:|---:|---|
| CAS(4,4) | 5 | 36 | −0.2349210164 / −0.2349326445 |
| CAS(8,8) | 3 | 4900 | −0.2170319290 / −0.2170716671 |

**Nothing in the code assumes a size.** The RDM kernels take `norb`, the
string counts and the link counts at run time, and the tile plan is a
ceil-division of `n_det`: CAS(4,4)'s 36 determinants at `--tiles 3` make
three 12-wide tiles, narrower than the digests' 16-wide shared-memory tile.
`--cublas` engages (`bits=53`) at both sizes.

**CAS(4,4) does not exercise every class; CAS(8,8) does.** At CAS(4,4) the
active space is the four π orbitals and the five core orbitals are all σ, so
`Si` is zero by symmetry (norm 0, energy 0, in PySCF's reference too). Its
kernels still run, but a check against an exact zero says little. CAS(8,8)
gives all eight classes non-zero norms (`Si` 0.0159). So CAS(4,4) is the
fastest smoke case, and every path variant runs on CAS(8,8):

| ctest entry (label `small`) | what it adds |
|---|---|
| `nevpt2_cas44`, `nevpt2_cas88` | the default digest and BLAS consume, `--tiles 3` (the tile loop and the on-device accumulate run 3×) |
| `nevpt2_cas88_consume_emitted` | `kernels.cu`'s consume kernels |
| `nevpt2_cas88_profile`, `nevpt2_df_cas88_profile` | the event-bracketed launches (`--profile`), DF slab DGEMMs included |
| `nevpt2_cas88_blas_digest` / `nevpt2_cas88_digest_emitted` | the other backend's default digest (CUDA / HIP) |
| `nevpt2_cas88_cublas` | the emulated digest, its probe and the f3 scatter (CUDA only) |
| `nevpt2_cas88_ozaki`, `nevpt2_df_cas88_pc_ozaki` | the int8 Ozaki digest, SC and PC |
| `nevpt2_df_cas44`, `nevpt2_df_cas88` | density fitting with `--check-blocks`; the default `--batch 8` leaves a ragged last slab on both (19 virtuals = 8 + 8 + 3, 17 = 8 + 8 + 1) |
| `nevpt2_cas44_pc`, `nevpt2_cas88_pc`, `nevpt2_df_cas44_pc`, `nevpt2_df_cas88_pc` | `--pc`: the solver handle's eigensolves, the `d x d` BLAS GEMMs and the PC slab einsums, on both demos |
| `nevpt2_cas88_rdm_tangent` | `--rdm-tangent`: the tangent RDM build beside the plain one, checked against twice it (the derivative itself is checked against a finite difference in the unit tier, `test/rdm/rdm_tangent_tests.cpp`) |
| `nevpt2_cas44_sc_tangent` | `--sc-tangent`: the SC class energies' tangent with respect to the RDMs, against the norms' exact affine increment, Euler's identity and a finite difference ("The SC tangent's finite-difference cross-check", below). `nevpt2_cas88_sc_tangent` runs the same at CAS(8,8) and is **not** `small` |
| `nevpt2_cas44_sc_hessian` | `--sc-hessian`: the three-argument tangent RDM build (three `L2`/`R` blocks per tile) and the SC energy's second derivative over four RDM sets, against Euler at second order, the norms' second-order increment and a finite difference of `Edot` ("The SC Hessian's finite-difference cross-check", below). `nevpt2_cas88_sc_hessian` runs the same at CAS(8,8) and is **not** `small` |
| `nevpt2_cas44_sc_hessian_vector` | `--sc-hessian-vector`: the gradient walk carrying a CI direction (a second set of block tangents and accumulators per class, swept beside the plain ones), three sigma calls, and `H_cc` materialised over 36 Hessian-vector products against as many three-argument builds ("The SC Hessian-vector product's checks", above). `nevpt2_cas88_sc_hessian_vector` runs the duality and the difference at CAS(8,8) and is **not** `small` |
| `nevpt2_sc_pseudodensity_cas44`, `nevpt2_sc_pseudodensity_cas88` | the SC gradient's pseudodensity assembly and its identities ("The SC pseudodensity identities", above) |
| `nevpt2_sc_pseudodensity_cas44_sc_hessian_kc` | `--sc-hessian-kc`: the gradient walk carrying a mixed direction (an orbital `w` with its digests' tangent merged, a CI `u`'s tangent blocks and the mixed blocks; or the walk's tangent along `w` with three sweeps more per class), the tangent RDM build at `h2e'[w]`, the sigma calls at `h2e'[w]`, and 36 columns ("The SC mixed Hessian block's checks", above). `nevpt2_sc_pseudodensity_cas88_sc_hessian_kc` runs the same at CAS(8,8) less the columns and is **not** `small` |
| `nevpt2_sc_pseudodensity_cas44_sc_hessian_kk_int`, `nevpt2_sc_pseudodensity_cas44_df_sc_hessian_kk_int` | `--sc-hessian-kk-int`: the gradient walk over dotted input sets (whole-block second-order inputs, four per pair), their generators — the promotion rule at a symmetric matrix with the traced pairs restored, the composition, and on the fitted case the three-index jets over the whole `B` — the digests' transpose and a plain RDM build per term per pair, and the ladders' pairings ("The orbital block's integral curvature", above). `nevpt2_sc_pseudodensity_cas88_sc_hessian_kk_int` runs the same at CAS(8,8) and is **not** `small` |
| `nevpt2_sc_pseudodensity_cas44_sc_hessian_kk`, `nevpt2_sc_pseudodensity_cas44_df_sc_hessian_kk` | `--sc-hessian-kk`: the gradient walk carrying two orbital directions with their digests' tangents merged (two tangent block sets per class, the cross tables, and the pseudodensities' tangent along one of them), one plain RDM build per direction, the curvature's walks, `energiesHessianDevice` at the digests' tangents, and the ladders' corner energies, each with its RDM build ("The orbital block's sandwich", above). `nevpt2_sc_pseudodensity_cas88_sc_hessian_kk` runs the same at CAS(8,8) and is **not** `small` |
| `nevpt2_pc_tangent_cas44`, `nevpt2_pc_tangent_cas88` | the PC tangent: a second solve per class, the tangent slab einsums and the `(m, m)` blocks' GEMMs, plus ten RDM builds and PC stages per entry ("The PC tangent against a finite difference", above) |
| `nevpt2_pc_hessian_cas44` | the PC Hessian: the dropped eigenvectors kept from the solve, their `(d, nd)` projections and slab einsums, the three-argument RDM build and five more tangent builds ("The PC Hessian against a finite difference of the three-term tangent", above) |
| `nevpt2_pc_hessian_vector_cas44` | the PC Hessian-vector product: the PC adjoint walk carrying a CI direction (the dropped eigenvectors, the tangent slab einsums against `T_u`, the `(d, nd)` bar accumulators and their sandwiches, a second sweep per class), per-class sigma calls, four Hessian-vector products, the three-term `y` at eight displaced states, and four `pcHessianDevice` runs for the duality ("The PC Hessian-vector product's checks", above) |
| `nevpt2_cas44_casscf`, `nevpt2_cas44_casscf_pc`, `nevpt2_df_cas44_casscf`, `nevpt2_df_cas44_casscf_pc` | the same four paths on **CASSCF orbitals** — a `NEVPT2G2` golden whose `reference_kind` is CASSCF ("The CASSCF-orbital entries", above). Same CAS(4,4) shape, so the same cost |
| `nevpt2_cas44_casscf_sc_tangent`, `nevpt2_sc_pseudodensity_kappa_cas44_casscf` | the two source-term canaries on **CASSCF orbitals** (`--sc-tangent`, `--kappa-act`; "The source terms on CASSCF orbitals: the wave-0 canary", above). Their CASCI twins' cost |
| `nevpt2_gradient_source_cas44_casscf`, `nevpt2_gradient_source_pc_cas44_casscf` | the SC and PC CI source terms' duality on **CASSCF orbitals**, with the CASSCF sidecar ("The source terms on CASSCF orbitals, measured", above). Their CASCI twins' cost |
| `nevpt2_gradient_source_cas44_df` | the SC CI source term's duality on the **density-fitted** CAS(4,4) golden with its density-fitted sidecar (#164; "The CI source term's forward-reverse duality", above). Its CASCI twin's shape; not yet run under any sanitizer preset |
| `nevpt2_df_gradient_source_cas44`, `nevpt2_df_gradient_source_cas88`, `nevpt2_df_cas44_sc_tangent` | the CI source term's duality and the SC tangent over **density-fitted** slabs (#162): the gradient walk and the tangent on owning, freed-per-slab DF slabs ("The CI source term's forward-reverse duality", above). Not yet measured under any sanitizer preset |

They are ordinary golden checks too, so `ctest --preset fast` and
`ctest --preset hip` run them on every backend.

### Canaries

A canary is a deliberately buggy run (`nevpt2_sanitizer_canary <mode>`, its
own small binary in `apps/sanitizer_canary/`, so neither demo carries a
planted bug). It passes only when the run fails **and** prints the tool's
report (`cmake/nevpt2_check_sanitizer_canary.cmake`; not `WILL_FAIL`, which
would credit a crash for the wrong reason). Each is registered only in the
build whose sanitizer must catch it, so **a green canary proves that
sanitizer is on in this build**. A canary that goes red means its sanitizer
is off or not failing runs: fix that, never delete the canary.

| canary | sanitizer | the bug | expected report |
|---|---|---|---|
| `nevpt2_canary_asan_heap_overflow` | ASan (both backends) | reads one past a `new double[64]` | `AddressSanitizer: heap-buffer-overflow` |
| `nevpt2_canary_ubsan_signed_overflow` | UBSan (both backends) | `INT_MAX + 1` | `runtime error: signed integer overflow` |
| `nevpt2_canary_memcheck_device_oob` | memcheck | a kernel reads one past its buffer | `Invalid __global__ read of size 8` |
| `nevpt2_canary_memcheck_free_before_read` | memcheck | `wwrFreeAsync(buf, S)`, *then* a kernel on S reads `buf` | `Invalid __global__ read of size 8` |
| `nevpt2_canary_memcheck_pool_free_before_read` | memcheck | the same, through a pool-backed `DeviceBuffer` at the demos' release threshold (the freed block stays in the pool) | `Invalid __global__ read of size 8` |
| `nevpt2_canary_memcheck_leak` (only with the leak check, the default) | memcheck `--leak-check full` | a 512-byte `wwrMallocAsync` never freed | `Leaked 512 bytes at` |
| `nevpt2_canary_initcheck_uninit_read` | initcheck | a kernel reads a fresh `wwrMallocAsync` buffer | `Uninitialized __global__ memory read of size 8` |
| `nevpt2_canary_racecheck_shared_race` | racecheck | thread 0 writes a `__shared__` slot that all 64 threads read, no barrier | `Race reported between` |
| `nevpt2_canary_synccheck_divergent_barrier` | synccheck | the block's two warps wait at two different `__syncthreads()` | `Barrier error detected` |

**Every canary goes red with its sanitizer off.** Run by hand through the
same verdict script with no launcher — against a plain Release build for the
host canaries, against `build-compute-sanitizer/`'s binary for the device
ones — each mode ran to completion, printed `nothing caught it`, exited 0,
and was failed ("Canary exited 0: the sanitizer did not fail the run"). The
ASan canary is red under `ubsan` and the UBSan canary under `asan`; on a
plain `hip` build both host canaries are red. The leak canary is red without
the leak flag (plain memcheck: `0 errors`). Two forms that look like the bug
and are **not** reported, both on the RTX 3080: a `__syncthreads()` that half
a block skips (synccheck is silent), and every thread writing a `__shared__`
slot and then reading its own value back (racecheck is silent, because each
read is forwarded from the thread's own write).

**The device canaries also prove the demo entries are wrapped.**
`CMAKE_TEST_LAUNCHER` is read when an executable target is *created*, not at
`add_test`. Set beside the root `CMakeLists.txt`'s `add_test` calls, it left
every demo entry unwrapped — memcheck "passed" in 0.4s per case — while the
canaries, handed the launcher explicitly, still passed. So the launcher is
set in `src/` before the targets exist (`cmake/nevpt2_sanitizers.cmake`), and
the canaries read it back off their own target's `TEST_LAUNCHER` property,
the same property ctest wraps the demos with. A launcher that fails to attach
turns them red.

**A finding can't hide behind `PASS:`.** `PASS_REGULAR_EXPRESSION` makes
ctest ignore the exit code. compute-sanitizer reports an error and lets the
demo carry on to print `PASS:`, and LSan reports at exit, after `PASS:`, so
either would pass on the regex alone. Every demo entry also carries a
`FAIL_REGULAR_EXPRESSION` on the tools' own report lines
(`ERROR SUMMARY: [1-9]`, `RACECHECK SUMMARY: [1-9]`, `ERROR: ...Sanitizer`,
`runtime error:`). Every unit-tier suite entry carries the same expression
(`cmake/add_gtest_suite_tests.cmake`). gtest's exit code already fails a
suite whose assertion fails, but a sanitizer report that leaves the exit code
alone would not.

### The unit tier under the sanitizers

Every unit-tier suite runs in each sanitizer preset. A `REQUIRES_GPU` suite
runs under compute-sanitizer and a host-only one runs bare
(`test/CMakeLists.txt`). Last run 2026-10-09/10: CUDA on the RTX 3080, HIP on
the RX 9060 XT, each preset's own Debug build, memcheck with the leak check.
**No suite needed a fix and no LSan suppression was added.**

| preset | entries run | unit entries | result |
|---|---:|---:|---|
| `asan` | 106 | 90 | all pass, canary green |
| `ubsan` | 106 | 90 | all pass, canary green |
| `compute-sanitizer`, memcheck + leak check | 103 | 84 (the 5 `death` suites dropped) | all pass, 4 `sanitizer_canary` entries green; all 36 GPU unit entries report `LEAK SUMMARY: 0 bytes leaked in 0 allocations`, `ERROR SUMMARY: 0 errors` |
| `compute-sanitizer`, initcheck | 100 | 84 | all pass, canary green |
| `hip-asan` | 104 | 89 | all pass, canary green |
| `hip-ubsan` | 104 | 89 | all pass, canary green |

The two `compute-sanitizer` rows were re-run last, after `cublas_emul_tests`
joined the tier. Their build has 89 unit entries: 70 suites (5 of them
`death`) plus 19 `<target>.SuiteListIsComplete` guards. The 84 that ran are
those less the 5 `death` suites. `CublasEmulDeclineTests` is compiled out of
that build (above). In both rows the unit tier added 25.6 s of `sec*proc`
(ctest's label summary). The two `hip-*` rows were re-run on the same tree,
with every unit binary in. Their 89 unit entries are 70 suites (6 of them
`death`, which run there, `CublasEmulStubDeathTest` among them) plus 19
guards: the HIP build has the `cublas_emul` stub suites where CUDA has the
digest ones. There the unit tier took 134–154 s of `sec*proc`. The `asan` and
`ubsan` rows were re-run on that tree too. Their 90 unit entries are 71 suites
(5 of them `death`) plus 19 guards: the compute-sanitizer build's 70 plus
`CublasEmulDeclineTests`, which is compiled in everywhere but there. In those
two rows the unit tier took 20–25 s of `sec*proc`.

### Wall times

Seconds per ctest entry: CUDA on the RTX 3080, HIP on the RX 9060 XT, each
sanitizer preset's own Debug build, racecheck with cuBLAS excluded,
memcheck with the leak check. "plain" is the same entry unsanitized
(`ctest --preset fast` / `hip`, Release).

| entry | asan | ubsan | memcheck | initcheck | racecheck | synccheck | hip-asan | hip-ubsan | plain CUDA | plain HIP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nevpt2_cas44` | 0.54 | 0.34 | 0.65 | 0.46 | 1.71 | 0.55 | 1.60 | 0.35 | 0.43 | 0.70 |
| `nevpt2_cas88` | 0.53 | 0.43 | 3.41 | 1.51 | 98.81 | 1.45 | 3.43 | 2.54 | 0.42 | 1.64 |
| `nevpt2_cas88_consume_emitted` | 0.46 | 0.38 | 3.55 | 1.65 | 98.89 | 1.30 | 3.51 | 2.72 | 0.36 | 1.65 |
| `nevpt2_cas88_profile` | 0.51 | 0.40 | 3.31 | 1.40 | 102.16 | 1.46 | 3.42 | 2.54 | 0.40 | 1.57 |
| `nevpt2_cas88_blas_digest` | 0.50 | 0.40 | 2.26 | 1.16 | 23.50 | 0.70 | — | — | 0.39 | — |
| `nevpt2_cas88_cublas` | 0.54 | 0.42 | 2.61 | 1.41 | 14.05 | 0.85 | — | — | 0.40 | — |
| `nevpt2_cas88_digest_emitted` | — | — | — | — | — | — | 5.82 | 5.02 | — | 1.64 |
| `nevpt2_df_cas44` | 0.42 | 0.30 | 0.46 | 0.46 | 2.92 | 0.45 | 2.28 | 1.37 | 0.30 | 1.41 |
| `nevpt2_df_cas88` | 0.52 | 0.42 | 3.31 | 1.46 | 104.19 | 1.44 | 3.42 | 2.50 | 0.41 | 1.64 |
| `nevpt2_cas44_pc` | 0.51 | 0.34 | 0.72 | 0.67 | — | — | 1.37 | 0.38 | — | — |
| `nevpt2_cas88_pc` | 0.67 | 0.50 | 4.22 | 2.17 | — | — | 2.91 | 1.60 | — | — |
| `nevpt2_df_cas44_pc` | 0.51 | 0.35 | 0.72 | 0.67 | — | — | 1.42 | 0.41 | — | — |
| `nevpt2_df_cas88_pc` | 0.67 | 0.51 | 4.17 | 2.17 | — | — | 2.94 | 1.64 | — | — |
| `nevpt2_cas44_casscf` | 0.46 | — | 0.56 | — | — | — | — | — | 0.33 | 0.33 |
| `nevpt2_cas44_casscf_pc` | 0.51 | — | 0.73 | — | — | — | — | — | 0.34 | 0.37 |
| `nevpt2_df_cas44_casscf` | 0.48 | — | 0.57 | — | — | — | — | — | 0.33 | 0.34 |
| `nevpt2_df_cas44_casscf_pc` | 0.53 | — | 0.72 | — | — | — | — | — | 0.35 | 0.37 |
| `nevpt2_sc_pseudodensity_kappa_cas44` | 0.88 | 0.67 | 1.42 | 1.17 | — | — | — | — | 0.38 | — |
| `nevpt2_pc_tangent_cas44` | 0.72 | 0.51 | 1.72 | 1.32 | — | — | — | — | 0.44 | — |
| `nevpt2_pc_tangent_cas88` | 3.07 | 3.05 | 45.22 | 19.25 | — | — | — | — | 2.27 | — |
| `nevpt2_gradient_source_cas44` | 0.95 | 0.69 | 3.47 | 2.72 | — | — | — | — | 0.46 | — |
| `nevpt2_gradient_source_cas88` | 2.52 | 2.51 | 59.25 | 37.07 | — | — | — | — | 1.91 | — |
| `nevpt2_sc_pseudodensity_cas44_sc_hessian_kk_int` | 9.92 | — | 9.08 | 8.24 | — | — | — | — | 1.03 | — |
| `nevpt2_sc_pseudodensity_cas44_df_sc_hessian_kk_int` | 19.10 | — | 22.25 | 17.52 | — | — | — | — | 1.97 | — |
| `nevpt2_sc_pseudodensity_cas44_sc_hessian_kk` | 9.83 | — | 16.09 | 13.15 | — | — | — | — | 1.49 | — |
| `nevpt2_sc_pseudodensity_cas44_df_sc_hessian_kk` | 9.99 | — | 17.45 | 13.70 | — | — | — | — | 1.52 | — |
| each canary | 0.36 | 0.35 | 0.75–0.86 | 0.49 | 0.53 | 0.77 | 0.37 | 0.36 | — | — |

The two `*_sc_hessian_kk` rows (#158, 2026-10-20) were run clean under
`asan`, memcheck (with the leak check) and initcheck on CUDA, each inside its
preset's whole run, and not under ubsan, racecheck, synccheck or on HIP;
their `—` columns mean "not measured", not "clean".

The two `*_sc_hessian_kk_int` rows (#154, 2026-10-20) were run clean under
`asan`, memcheck (with the leak check) and initcheck on CUDA, each inside its
preset's whole run, and not under ubsan, racecheck, synccheck or on HIP;
their `—` columns mean "not measured", not "clean".

The four `*_casscf*` rows ("The CASSCF-orbital entries", above) have been run
clean under `asan` and under memcheck (with the leak check) on CUDA and plain
on both backends, and **not** under ubsan, initcheck, racecheck, synccheck,
`hip-asan` or `hip-ubsan`; their four columns of `—` mean "not measured", not
"clean". They were also measured with another job sharing the host and the
card, so they are upper bounds — they are the CAS(4,4) paths beside them at
the same cost, which is the point of the row rather than the figure.

The `nevpt2_cas88_profile` row was measured with the RDM-build-only profiler
that `--profile` has since absorbed. `nevpt2_df_cas88_profile` and the two
`--ozaki` small entries have not been timed under the tier; the `--ozaki`
pair has been run clean under memcheck (with the leak check), racecheck,
initcheck, synccheck, `asan` and `ubsan`, but not yet `hip-asan` /
`hip-ubsan`. racecheck and synccheck have not been run on the `--pc` entries,
nor on `nevpt2_sc_pseudodensity_kappa_cas44`, the two `nevpt2_pc_tangent_*`
entries, the two `nevpt2_gradient_source_*` ones or the three
`nevpt2_sc_pseudodensity_kappa_full_*` ones. All seven have been run clean
under `asan`, `ubsan`, memcheck (with the leak check) and initcheck on CUDA —
the `kappa_full` three by name, since they are not in the tier at all ("The
full orbital source term" above says why, and what they cost under each tool);
none of the seven has been run on HIP, and their `—` columns above mean "not
measured", not "clean".

Everything except racecheck and `nevpt2_gradient_source_cas88` finishes in
well under a minute, so CAS(8,8) is affordable under every tool. racecheck
costs ~100s per CAS(8,8) entry on the emitted digest's 16×16 shared-memory
tiles (the BLAS paths are cheaper because cuBLAS is excluded), ~7 minutes for
the whole preset. memcheck takes ~8× the plain run at CAS(8,8), and initcheck
~3.5× — with two exceptions, both of which earn their time by what they
check. `nevpt2_pc_tangent_cas88` is 45s under memcheck because it runs ten
RDM builds and ten PC stages. `nevpt2_gradient_source_cas88` is 59s under
memcheck and 37s under initcheck (~31× and ~19× its plain run): it is the
only `small` entry that runs the SIGMA kernels, whose reverse-table walks are
many short launches over the whole determinant axis (five of them per `y`,
plus one tangent RDM build per direction), and that per-launch cost is what
the sanitizers multiply. 59s makes it the second-slowest entry in the tier
after racecheck's; it stays `small` because the duality it checks is this
milestone's headline claim. On HIP the sanitizer
builds are Debug, so their device code is `-O0` too, which is why
`nevpt2_cas88_digest_emitted` is ~3.5× the plain run there while host ASan on
CUDA costs almost nothing.

### What the tier leaves out, and why

- **A kernel-name *include* filter for racecheck/synccheck**
  (`--kernel-name kns=<namespace>`, plus a build check that every kernel
  mangles that namespace). racecheck *excludes* cuBLAS instead, and synccheck
  runs unscoped. An include filter would need every kernel in one namespace,
  and ours are not (`kernels.cu`'s sit in the global namespace). The exclude
  keeps every kernel of ours in scope with no naming rule to enforce.
- **A `no_sanitizer` label and a CI sanitizer preset.** No entry needs
  exempting, and there is no CI.
- **The large cases.** CAS(10,10) and up are too slow under
  compute-sanitizer. The salicylaldimine goldens and the two
  gradient-validation cases (LiF, acrolein) stay out too, so the tier stays on
  N2.

## Coverage

clang source-based coverage of **host** code: which lines of `src/` and the
apps under `apps/` (all but the sanitizer canary) the `fast` test set
executes. It is a map of what the
tests never reach, not a measure of correctness — a line counted as covered
was run, not checked. The golden tier stays the only numerical claim.

```bash
devtools/coverage.sh                          # CUDA: configure, build, ctest, report
devtools/coverage.sh --preset hip-coverage    # the same on the AMD card
devtools/coverage.sh --html build-coverage/html --lcov build-coverage/coverage.lcov
```

The `coverage` / `hip-coverage` presets are **Release** builds with
`NEVPT2_ENABLE_COVERAGE` (cmake/README.md, "`nevpt2_coverage.cmake`"):
the configuration `fast` tests. Source-based counters go in before the
optimizer, so the counts are exact at `-O3`. A Debug build was tried first
and dropped: on HIP it compiles device code `-O0` too, the CAS(10,10)
`--digest-emitted` entries took ~104 s each on the RX 9060 XT, and
`nevpt2_cas1010_ozaki` had not finished after more than ten minutes (5.9 s under Release).
Each preset's test preset runs what `fast` runs, each process writing
`build-<preset>/profraw/<pid>.profraw`. The script merges those with
`llvm-profdata`, finds every binary carrying a coverage map, and reports
through `llvm-cov`, dropping `COVERAGE_IGNORE_REGEX` (devtools/config.sh:
`deps/`, `_deps/`, `test/`, `apps/sanitizer_canary/`, system headers).

ctest runs **serially**. Several golden entries at once on one card abort
(six of them did under `-j4` on the RTX 3080, every one passing when run
alone), and an aborted entry's host paths would then read as never executed.
A failing test still produces a report, but the script exits 1.

### What it cannot see

- **No `.cu`, on either backend** — kernels and their launchers alike. There
  is no llvm-cov for device code, and instrumenting only the host half of a
  `.cu` would differ by backend (nvcc's host compiler under CUDA, a `-x hip`
  CXX unit whose device pass would take the flags under HIP). A kernel's
  correctness is the golden tier's to prove.
- **Anything on the way to an abort.** A process that ends in `std::abort()`
  writes no profile: the `check`/`gpuCheck`/`narrowTo` tier, and every
  `EXPECT_DEATH` child. Those paths read as unexecuted even though the
  `death` suites exercise them. This is most of what `src/common/common.cppm`
  misses.
- **The `slow` entries.** CAS(12,12) and the salicylaldimine cases are not in
  `fast`, so a path only they reach (large tile counts, for one) is missed.

### Measured

`devtools/coverage.sh -j 24` and `devtools/coverage.sh --preset hip-coverage
-j 24`, Release, clang 20.1.8, on the tree of #109 (one card at a time, each
under its card's lock), totals over `src/` + the four apps llvm-cov finds
instrumented (`integral_direct`, `density_fit`, `sc_pseudodensity`,
`pc_tangent`) (covered / total). A covered line was run, not checked:

| card, preset | entries | regions | functions | lines | branches | ctest phase |
|---|---:|---:|---:|---:|---:|---:|
| RTX 3080, `coverage` | 247/247 passed | 5946/6549 (90.79%) | 732/752 (97.34%) | 12205/12913 (94.52%) | 3588/4523 (79.33%) | 211.2 s |
| RX 9060 XT, `hip-coverage` | 241/241 passed | 5924/6544 (90.53%) | 728/752 (96.81%) | 12165/12898 (94.32%) | 3569/4521 (78.94%) | 586.3 s |

Each run merged one profile per entry from 35 instrumented binaries. The
whole script (configure, build, ctest, report) took about 253 s on the 3080
and 621 s on the 9060 XT. Both reports warn that 4 functions have mismatched
data (llvm-cov's `warning: 4 functions have mismatched data`); the totals are
llvm-cov's as printed. Against the previous rows (76/76 and 71/71 entries, 13
binaries, a tree without `src/gradient/`, `apps/sc_pseudodensity/` or
`apps/pc_tangent/`: CUDA 89.20% / 97.51% / 93.27% / 78.17%, HIP 88.03% /
95.85% / 92.46% / 76.40%), every percentage rose except **CUDA's function
coverage, which fell from 97.51% to 97.34%** (6 of 241 functions missed then,
20 of 752 now). On
that previous tree a Debug build of the CUDA preset gave the identical CUDA
totals, as it should; that was not repeated here. The HIP run misses 17 regions more
than the CUDA one: 10 in `src/rdm/rdm_build.cpp`, 4 in `src/cublas/` (on HIP
the refusing stub, which nothing calls: `--cublas` is rejected at flag
parsing) and 3 in `src/cli/cli.cpp`.

The lowest file is `src/common/common.cppm` (`narrowTo`'s failure path, which
aborts); after it come the apps' `main.cppm` (the error-reporting branches of
argument and golden-file handling, which no ctest entry provokes),
`apps/sc_pseudodensity/source_oracle.cppm`, `src/einsum/einsum_planner.cppm`
and `src/rdm/rdm_build.cpp`.

