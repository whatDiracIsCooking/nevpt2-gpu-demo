# Testing

How this tree is checked: the two test tiers, why there is no CI, the local
gate, the sanitizer tier — what each tool catches, what it cannot, and
how the tier proves its own tools are on — and host code coverage. Every
figure names its card.

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
devtools/cpp-tier.sh                # CUDA, on a box with a card (runs the four gates below first)
devtools/cross-backend-check.sh     # does the OTHER backend still compile (the two lints too)
devtools/stream-lint.sh --strict    # every GPU call on the one stream; both scripts above run it
devtools/throw-lint.sh              # no `throw` in src/, apps/ or test/; likewise
uv run python reference_data/golden_layout_check.py   # "The golden layout contract", below; cpp-tier.sh runs it
uv run ruff check                   # the Python lint (pyproject.toml's [tool.ruff]); likewise
```

`cpp-tier.sh` runs the four as named phases (`stream-lint`, `throw-lint`,
`golden-layout`, `ruff`) before it configures, and under `--no-test` too:
none needs a card, and a `--no-test` run is often the only run a no-card
change gets. A failing one fails the tier and is named in its summary line
(`FAIL(rc=1) failed=golden-layout`). The two Python phases run through
`uv run --frozen --no-sync` after `uv sync --frozen --check`, so a missing or
stale environment (`/opt/venv` in every image, `.venv` on a host) **fails**
the phase with `uv sync --frozen` as the fix, rather than skipping it: #84's
stray reader reached `main` because the gate that would have caught it was
one nobody ran. `cross-backend-check.sh` runs only the two lints: they read
the one C++ tree it compiles for the other backend, while the Python phases
are backend-independent and already ride every `cpp-tier.sh` run.

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
  `loadGradGold`, requires both layouts to be among them, and requires every
  file of one layout to have its array table at the same derived offset
  (`GoldenFile::arraysAt`) and the two layouts' to differ.
- **Python**: `uv run python reference_data/golden_layout_check.py` — every
  committed golden, gradient sidecar and MO sidecar through every reader entry
  point in `reference_data/` and back through its writer, byte for byte; a
  golden from the current writer read back; and no parser outside
  `golden_format.py`. Ends `PASS:` or `FAIL:` with an exit status.

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

## The CASSCF-orbital entries

Six golden entries read a reference whose orbitals came from a converged
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

The CAS(4,4) four are `small`, so the sanitizer tier runs a `NEVPT2G2` file
too; CAS(10,10) is the production size at the shipped `--tiles 3`. All three
goldens carry PC fields.

**What there is deliberately no entry for: the gradient paths.** Every
derivative in this tree is derived for the CASCI-on-RHF Lagrangian, so
`nevpt2_demo --sc-tangent` and `--gradient-source` **refuse** a CASSCF golden
as an `err_unsupported` before any card is touched, and so does
`reference_data/nevpt2_numgrad.py`. That refusal is checked in the unit tier
instead, by `test/golden`'s `GoldenCommittedCasscfTests` — which reads
`golden/n2_ccpvdz_cas44_casscf.nevpt2gold` itself, so the gate is held against
a committed reference and not only against a file a test spliced together.
Host-only, no card.

Nothing in `src/` branches on the reference kind anywhere else: the RDM build,
the digests, the energy stage and the PC solve see a CI vector and integrals,
which is exactly the claim these entries make.

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
tangent's (a derivative of a solve, which has no identity at all). It
validates `energiesTangentDevice` (`nevpt2.energy`'s `:sc_tangent`
partition) so that
every later check of the reverse-mode assembly can be an exact dot-product
identity against the tangent at this tree's usual 1e-7, with no step size
anywhere. `apps/integral_direct/tangent_check.cppm` has the derivation; the
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
high-water is 0.7 MB / 60.8 MB / 1679 MB — the last being the CAS(10,10) RDM
build's, which the entry runs first and which dominates it.

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
sizes. Building `y` itself takes 0.01s / 0.52s / 5.87s — five sigma calls (one
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
with no card — the oracle against itself passes with all nine rows compared,
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
| CI stationarity, two seeds | the Lagrangian's CI part, laddered along `u` at the solved multipliers, has zero slope | the CI Z-vector solve and the CI/(O2) coupling |
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
time, with the GPU's PC source terms: 7 of 11 hold, and the four CI-side
ladders end on run length at block2's round-off rather than on a value
([`pc-nevpt2.md`](pc-nevpt2.md), "The source terms by reverse mode", which
also records N2 PC's one such ladder with the GPU's terms).

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
| `nevpt2_sc_pseudodensity_cas44`, `nevpt2_sc_pseudodensity_cas88` | the SC gradient's pseudodensity assembly and its identities ("The SC pseudodensity identities", above) |
| `nevpt2_pc_tangent_cas44`, `nevpt2_pc_tangent_cas88` | the PC tangent: a second solve per class, the tangent slab einsums and the `(m, m)` blocks' GEMMs, plus ten RDM builds and PC stages per entry ("The PC tangent against a finite difference", above) |
| `nevpt2_cas44_casscf`, `nevpt2_cas44_casscf_pc`, `nevpt2_df_cas44_casscf`, `nevpt2_df_cas44_casscf_pc` | the same four paths on **CASSCF orbitals** — a `NEVPT2G2` golden whose `reference_kind` is CASSCF ("The CASSCF-orbital entries", above). Same CAS(4,4) shape, so the same cost |

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
| each canary | 0.36 | 0.35 | 0.75–0.86 | 0.49 | 0.53 | 0.77 | 0.37 | 0.36 | — | — |

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
two demos under `apps/` the `fast` test set executes. It is a map of what the
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

`devtools/coverage.sh`, Release, clang 20.1.8, totals over `src/` + the two
demos (covered / total):

| card, preset | entries | regions | functions | lines | branches |
|---|---:|---:|---:|---:|---:|
| RTX 3080, `coverage` | 76/76 passed | 1338/1500 (89.20%) | 235/241 (97.51%) | 2982/3197 (93.27%) | 770/985 (78.17%) |
| RX 9060 XT, `hip-coverage` | 71/71 passed | 1316/1495 (88.03%) | 231/241 (95.85%) | 2942/3182 (92.46%) | 751/983 (76.40%) |

Each run merged one profile per entry from 13 instrumented binaries; the
ctest phase took 44.9 s on the 3080 and 143.5 s on the 9060 XT. A Debug build
of the CUDA preset gave the identical CUDA totals, as it should. The HIP
totals are lower mostly because `src/cublas/` there is the refusing stub
(`--cublas` is rejected at flag parsing on HIP), which nothing calls.

The lowest files are the two demos' `main.cppm` (the error-reporting branches
of argument and golden-file handling, which no ctest entry provokes) and
`src/common/common.cppm` (`narrowTo`'s failure path, which aborts).

