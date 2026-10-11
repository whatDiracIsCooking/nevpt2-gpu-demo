# Reference data

The golden files are the oracle: every number this code prints is checked
against one of them, to 1e-7. This page says what is in them, how they are
made, why the generator is set up the way it is, and what each committed file
is for. The generator is `reference_data/generate_golden.py`, an offline
Python tool. Nothing in a build or a test runs it: its output is committed,
and no Python or PySCF runs at run time.

## What "CASSCF input" means here, and why the golden state is CASCI

What the demos read back is what any CASSCF/CASCI package hands a downstream
NEVPT2 driver, however the orbitals were obtained: a CI vector, the cheap 1-
and 2-RDMs, and a set of MO-integral blocks. The generator canonicalises the
state exactly as `pyscf.mrpt.NEVPT.kernel` does, and stores PySCF's own
per-class SC-NEVPT2 answer alongside.

The orbitals the committed goldens use are **converged RHF** orbitals, and the
state is a plain `mcscf.CASCI` diagonalisation on them: a linear-algebra step
with no orbital optimiser anywhere in the chain, which reproduces bit for bit
given the two settings in "Reproducibility" below. What rules a fresh CASSCF
out of the committed files is **not** reproducibility — it reproduces too, and
the next subsection is the measurement — but that its orbitals are a
*different reference*, worth tens of millihartree in `E_corr`, and that
optimising them costs a run.

It never writes `dm3`, `f3ac` or `f3ca`. Producing those from the CI vector is
the expensive step the GPU build exists for, and storing them would let the
demo skip the part it is supposed to demonstrate.

### A fresh CASSCF reproduces, measured

**This section used to assert that a full CASSCF run is "not reproducible" and
that a NEVPT2 energy built on one "wanders by tens of millihartree between
identical invocations". Nothing ever measured that, and the measurement
contradicts it: the run-to-run spread is exactly zero, in E_CASSCF, in
E_corr(SC) and in the orbital-gradient norm alike.** The old figure is the
size of a *different* effect — what changing the reference from CASCI to
CASSCF costs, the last table below — not of any run-to-run wander. The claim
is recorded here refuted rather than quietly dropped.

`reference_data/casscf_repro_probe.py` is the measurement, and re-runs it. Per
run it does, from scratch: RHF, a symmetry-adapted `mcscf.CASSCF`, the orbital
gradient at the converged orbitals (`mc.get_grad`, i.e. the 2-norm of
`mc1step.gen_g_hop(...)[0]`, which is what `conv_tol_grad` gates), a **fresh**
deterministic `mcscf.CASCI` on those orbitals, and then the generator's own
downstream unchanged — `NEVPT.canonicalize(cas_natorb=True)`,
`_check_degeneracy`, `_pyscf_dms`, `_pyscf_per_class` — so `E_corr(SC)` is the
number a golden built on these orbitals would carry.

Conditions: PySCF 2.14.0, NumPy 2.5.3, Python 3.12.3 (the CUDA devcontainer's
`/opt/venv`), **single-threaded** (forced by importing `generate_golden`), FCI
`conv_tol` 1e-12, N2/cc-pVDZ at 1.1 Å in the auto-detected D2h, `mcscf.CASSCF`
with `conv_tol` 1e-12 / `conv_tol_grad` 1e-8 and `max_cycle_macro` 200.
**Offline CPU only: no card and no build are involved**, on a 24-core host
used one thread at a time.

Five back-to-back runs per case:

| case | E_CASSCF | ΔE_CASSCF | E_corr(SC) | ΔE_corr | ‖g_orb‖ | \|E_CASCI − E_CASSCF\| | macro |
|---|---|---|---|---|---|---|---|
| CAS(4,4) | −109.046139074737 | **0** | −0.207071732083 | **0** | 5.9e-9 | 2.0e-13 | 5 |
| CAS(10,10) | −109.129689184292 | **0** | −0.126182539677 | **0** | 2.5e-8 | 8.5e-14 | 200 (ceiling) |

Every run of both cases agreed **bit for bit** — E_CASSCF, all eight class
energies, the gradient norm, the CI vector — and so did three separate
*processes* at CAS(4,4), compared at full `repr` precision. No run landed on a
different stationary point: same active irreps, natural occupations identical,
active- and core-subspace overlap singular values with run 0 of
1.000000000000, `1 − |⟨ci|ci₀⟩| ≤ 2.3e-15`. `_check_degeneracy` passed on the
CASSCF orbitals of both cases, and those orbitals came out symmetry-pure to
1.7e-14 (so a `--mo-from` sidecar could re-derive `orbsym` from them). PySCF's
`mc1step` is a deterministic algorithm on deterministic input; single-threaded,
that is all it takes. The one thing it does *not* do at CAS(10,10) is converge
to `conv_tol_grad` 1e-8 — it hits the 200-macro ceiling at ‖g_orb‖ 2.5e-8 —
while `conv_tol_grad` 1e-7 **converges** in 16 macro iterations to the same
energy, class energies and E_corr, bit for bit.

**Different starting points, same solution.** Jitter and a different
stationary point are different failure modes, and only the second would have
justified the old claim, so the probe also starts the optimiser somewhere
else: `--perturb EPS` rotates the RHF start by a random **same-irrep** (so
symmetry-preserving) rotation of EPS rad, a different seed per run.

| case | perturbation | runs | ΔE_CASSCF | ΔE_corr(SC) | ‖g_orb‖ reached | stationary point |
|---|---|---|---|---|---|---|
| CAS(4,4) | 0.1 rad | 5 | 7.8e-13 | 6.3e-9 | 8.7e-8 … 7.4e-7 | unchanged |
| CAS(4,4) | 0.3 rad | 5 | 4.5e-13 | 1.6e-8 | 2.6e-7 … 9.6e-7 | unchanged |
| CAS(10,10) | 0.1 rad | 5 | 4.1e-11 | 8.9e-8 | 5.2e-8 … 7.8e-7 | unchanged |

"Unchanged" is the full fingerprint again: no active-orbital permutation, the
same irrep labels, active-subspace overlap singular values ≥ 0.999999997 with
run 0, natural occupations within 4.7e-7 and `1 − |⟨ci|ci₀⟩| ≤ 1.0e-11`. What
differs between these runs is only how far the optimiser got before the macro
ceiling, and E_corr tracks that residual gradient linearly (next subsection).
That is the invariance argument holding rather than failing: with
`cas_natorb` canonicalisation the active space is pinned, so a
differently-rotated *route* to the same stationary point cannot move E_corr by
more than the leftover gradient allows — ~1e-8, not ~1e-2.

This measurement is about **one fixed geometry**. The separate statement in
[`gradient-theory.md`](gradient-theory.md), §1.1 and §4.4 — that a *displaced*
CASSCF would not give a finite difference a stable reference to be a gradient
of — is a different claim about a different calculation, and nothing here
measures it.

### The `--casscf` / `--mo-from` gate thresholds

How much a residual orbital gradient costs E_corr, measured at CAS(10,10) by
loosening `conv_tol_grad` alone (ΔE_corr against the converged
−0.126182539677):

| `conv_tol_grad` | ‖g_orb‖ reached | macro | E_corr(SC) | ΔE_corr |
|---|---|---|---|---|
| 1e-8 | 2.521e-8 | 200 (ceiling) | −0.126182539677 | — |
| 1e-7 | 2.521e-8 | 16 | −0.126182539677 | 0 |
| 1e-6 | 3.372e-7 | 15 | −0.126182503816 | 3.6e-8 |
| 1e-5 | 8.596e-7 | 17 | −0.126182428117 | 1.1e-7 |
| 3.16e-4 (PySCF's default) | 2.759e-5 | 15 | −0.126178941415 | 3.6e-6 |

so dE_corr/d‖g_orb‖ ≈ 0.11–0.13 Eh per unit norm at CAS(10,10), and 0.057 at
CAS(4,4) (the same test there: ‖g_orb‖ 1.126e-5 costs 6.4e-7). And the scale
of an input a gate must *refuse* — the RHF orbitals, which is what the
committed goldens are built on:

| orbitals | ‖g_orb‖ | E_CASCI | E_CASCI − E_CASSCF |
|---|---|---|---|
| CAS(4,4), RHF | 6.98e-2 | −109.017971290758 | +2.82e-2 |
| CAS(10,10), RHF | 7.54e-2 | −109.048231643881 | +8.15e-2 |

(both E_CASCI values are the committed goldens' own `e_casci`, which is what
those files hold: CASCI on RHF orbitals.)

**The two thresholds**, for the stationarity gate a committed-orbital sidecar
needs:

- **‖g_orb‖ ≤ 1e-6 — the proposal, confirmed**, as the refusal. It is the
  2-norm of `mc.get_grad(mo, (casdm1, casdm2), eris)` at the sidecar's
  orbitals, with the RDMs of a fresh CASCI on them. A properly converged run
  lands 40× under it (2.5e-8 at CAS(10,10), 5.9e-9 at CAS(4,4)); RHF orbitals
  are five decades over; a PySCF-default-converged CASSCF (2.8e-5) is over by
  a decade, which it should be, since its E_corr is 3.6e-6 from converged —
  well outside the 1e-7 the demos check to. The caveat the slope above forces:
  1e-6 is a **stationarity** check, not an E_corr guarantee, because orbitals
  accepted right *at* the boundary could sit ~1.2e-7 from the converged E_corr
  at CAS(10,10). So pair it with `--casscf` optimising at `conv_tol` 1e-12 and
  `conv_tol_grad` **1e-7** — which converges, unlike 1e-8, and reaches 2.5e-8
  — and every sidecar this tree writes is 40× inside the gate with an E_corr
  bit-identical to the converged one. Tightening the refusal itself to 1e-7
  buys the E_corr guarantee but leaves only 4× of headroom over what
  `mc1step` reaches at CAS(10,10), so it may refuse a larger case's
  legitimate sidecar; 1e-6 plus the tight `--casscf` setting is the
  recommendation.
- **\|E_CASCI − E_CASSCF\| ≤ 1e-9 — the proposal, confirmed** with four
  decades of headroom: measured 2.0e-13 (CAS(4,4)) and 8.5e-14 (CAS(10,10))
  between a run's E_CASSCF and a fresh `mcscf.CASCI` on its orbitals at
  `FCI_CONV_TOL`. It is a fingerprint check in energy form — a wrong basis,
  geometry or active space shows up here by decades (RHF orbitals: 2.8e-2 and
  8.1e-2) — and not a stationarity check, which is the other threshold's job.

**Where "tens of millihartree" actually comes from.** The CASCI-vs-CASSCF
*reference* difference, which is real, is the whole reason to want
CASSCF-orbital goldens:

| case | golden E_corr (CASCI on RHF) | E_corr on CASSCF orbitals | difference |
|---|---|---|---|
| CAS(4,4) | −0.234921016374 | −0.207071732083 | 2.78e-2 |
| CAS(10,10) | −0.194651972215 | −0.126182539677 | 6.85e-2 |

A golden built on CASSCF orbitals is a *different* number, not a noisier one.

Re-measure all of the above with:

```bash
uv run python reference_data/casscf_repro_probe.py --ncas 4 --nelecas 4 --runs 5
uv run python reference_data/casscf_repro_probe.py --runs 5               # CAS(10,10)
uv run python reference_data/casscf_repro_probe.py --runs 5 --perturb 0.1
uv run python reference_data/casscf_repro_probe.py --runs 1 --conv-tol-grad 1e-5
```

## Regenerating the golden file

Not needed to build or run anything: the committed files are the reference.
To regenerate one, or to build a different active space:

```bash
uv run python reference_data/generate_golden.py   # default: N2/cc-pVDZ CAS(10,10)

uv run python reference_data/generate_golden.py \
    --atom "N 0 0 0; N 0 0 1.1" --basis sto-3g --ncas 4 --nelecas 4 \
    --name n2_sto3g_cas44 --out /tmp/n2_sto3g_cas44.nevpt2gold
./src/build/nevpt2_demo --golden /tmp/n2_sto3g_cas44.nevpt2gold --tiles 1

# the density-fitted goldens (nevpt2_df_demo's oracle)
uv run python reference_data/generate_golden.py --df                     # n2_ccpvdz_cas1010_df
uv run python reference_data/generate_golden.py --df --ncas 12 --nelecas 12 --no-pc --name n2_ccpvdz_cas1212_df
uv run python reference_data/generate_golden.py --df --basis cc-pvtz --name n2_ccpvtz_cas1010_df

# the small cases, with their gradient sidecars (see "The gradient sidecar")
uv run python reference_data/generate_golden.py --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44 --gradgold
uv run python reference_data/generate_golden.py --ncas 8 --nelecas 8 --name n2_ccpvdz_cas88 --gradgold

# the 12-orbital files: CAS(10,12) is the PC case, CAS(12,12) is refused one
uv run python reference_data/generate_golden.py --ncas 12 --nelecas 10 --name n2_ccpvdz_cas1012
uv run python reference_data/generate_golden.py --ncas 12 --nelecas 12 --no-pc --name n2_ccpvdz_cas1212

# the gradient-validation pair (see "LiF and acrolein" below), both with sidecars
uv run python reference_data/generate_golden.py \
    --atom reference_data/geometries/lif.xyz --basis def2-svp \
    --ncas 6 --nelecas 4 --cas-irreps "B1=3,B2=3" --no-pc \
    --name lif_def2svp_cas46 --gradgold
uv run python reference_data/generate_golden.py \
    --atom reference_data/geometries/acrolein.xyz --basis def2-svp \
    --ncas 5 --nelecas 6 --cas-irreps "A'=1,A\"=4" \
    --name acrolein_def2svp_cas65 --gradgold

# the numerical gradient in those two sidecars (see "The numerical gradient")
uv run python reference_data/nevpt2_numgrad.py --case lif_def2svp_cas46
uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65

# the ANALYTIC gradient of the same two states, which writes no committed
# file at all (see "The analytic gradient")
uv run python reference_data/nevpt2_grad.py --case lif_def2svp_cas46
uv run python reference_data/nevpt2_grad.py --case acrolein_def2svp_cas65
```

**Regeneration is a check.** Every committed golden regenerates
byte-identical, CAS(12,12) included (generated twice in separate processes,
~8 min each, single-threaded). A regenerated file that differs from the
committed one means something changed.

### Reproducibility

Two settings that PySCF does not default to make this work, and neither may
be dropped to make generation faster.

- **Single-threading.** `generate_golden.py` sets `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 before numpy is imported,
  overriding any inherited value on purpose.
- **A 1e-12 Davidson tolerance** (`FCI_CONV_TOL`), asserted to have converged.

With PySCF's defaults (OpenMP over every core, Davidson `conv_tol` 1e-8),
neither the conventional nor the DF path reproduced on a 24-core machine.
Back-to-back builds at CAS(10,10), worst pair of eight:

| setting | ΔE_CASCI | ΔE_corr |
|---|---|---|
| DF, defaults | 7.4e-10 | 8.7e-8 |
| conventional, defaults | 7.3e-10 | 7.0e-8 |
| DF, `OMP_NUM_THREADS=1` | 0 | 0 (bit-identical) |
| DF, `fcisolver.conv_tol` 1e-12, threaded | 2.6e-13 | 2.4e-9 |
| DF, both | 0 | 0 (bit-identical) |

Threaded, the OpenMP/BLAS reductions differ at ~1e-14, which is enough for a
Davidson stopping at 1e-8 to stop somewhere else. E_corr inherits the
CI-vector error linearly, where E_CASCI inherits it only quadratically, which
is why a 1e-10 energy wobble becomes a ~1e-7 E_corr one. Tightening the SCF's
`conv_tol` does nothing; the SCF was never the problem. The loose default was
also an accuracy problem, not only a reproducibility one: at 1e-8,
CAS(10,10) E_corr lands up to 1.06e-7 from the converged value.

**Why not tighter than 1e-12.** PySCF's FCI solver sets `lindep` = 1e-12, and
the Davidson drops any correction vector with `|r|² < lindep`, so the
residual floors at ~1e-6 whatever `conv_tol`, `conv_tol_residual` or
`max_space` say (this is why 1e-14 runs out of cycles). Lowering `lindep`
(1e-16 to 1e-22) does let it converge to a measured `‖Hc − Ec‖` of 1e-8 to
1e-10. With point-group symmetry on (next section), the committed 1e-12
agrees with such a residual-1e-10 run on every case:

| golden | E_corr at 1e-12 | E_corr at residual ~1e-10 | Δ |
|---|---|---|---|
| `n2_ccpvdz_cas1010` | −0.1946519722 | −0.1946519729 | 7e-10 |
| `n2_ccpvdz_cas1010_df` | −0.1947020047 | −0.1947020054 | 7e-10 |
| `n2_ccpvtz_cas1010_df` | −0.2933410933 | −0.2933410919 | 1.4e-9 |
| `n2_ccpvdz_cas1212` | −0.1642545886 | −0.1642545884 | 2e-10 |
| `n2_ccpvdz_cas1212_df` | −0.1642975446 | −0.1642975445 | 1e-10 |

So past 1e-12 the `lindep` floor buys at most 1.4e-9, and 1e-12 stays.

### Point-group symmetry

**The molecule is built with point-group symmetry, and without it the
CAS(12,12) energy is not well-defined.** N2's canonical virtuals come in
exactly degenerate π/δ pairs (five at CAS(12,12), one at CAS(10,10)), and
the rotation inside such a pair is arbitrary. The `Srs` class,
`−N/(Δ + h/N)` summed over *pairs* of virtuals, is not invariant to that
rotation; `Sr`, summed over single virtuals, is. Rotating only within the
degenerate virtual pairs by θ, on one fixed CI vector:

| case | max over θ of ΔSr | max over θ of ΔSrs |
|---|---|---|
| CAS(10,10) (1 pair) | 6e-11 | 6.4e-8 |
| CAS(12,12) (5 pairs) | 1e-15 | 3.0e-6 |

Without symmetry, numerical noise picks θ. A CI vector that differs at 1e-9
lands on a different θ and so on a different `Srs`: fully converged
CAS(12,12) CI vectors scattered by up to ~2e-5 in E_corr, all of it in `Srs`
(the other seven classes agreed to ≤1e-11).

With symmetry, each degenerate partner sits in its own irrep, so symmetry
fixes the rotation instead of noise. `_build_mol` auto-detects the point
group and maps only the two linear groups to an abelian subgroup
(`LINEAR_SUBGROUP`: Dooh → D2h, Coov → C2v); PySCF itself drops any other
non-abelian group to an abelian one (Td → D2). For N2 that is D2h. `Dooh`
itself is not usable: the CAS(10,10) active space holds half of a π pair,
which PySCF's cylindrical-symmetry FCI refuses. **Do not hard-code a group**:
a fixed `symmetry="D2h"` fails on every molecule without D2h symmetry (H2O
raises `PointGroupSymmetryError` asking for C2v, CO the same for Coov).

The result is the **D2h-adapted** SC-NEVPT2 energy, a fixed convention, not
an orientation-free one. Symmetry picks one orientation of each degenerate
pair and holds it; it does not make `Srs` independent of that choice, which
strongly-contracted NEVPT2 as PySCF implements it (and as `src/` reproduces
it) is not. Another equally valid orientation would give an E_corr different
by up to the 3e-6 the rotation test shows. For a golden, what matters is that
the convention is fixed and reproducible to the byte.

### The degeneracy refusal

An abelian subgroup does not split every degenerate set: Td's E becomes
A + A in D2, so both partners share an irrep and noise picks their rotation
again. So `_check_degeneracy` refuses to write a golden in which two
canonical core or virtual orbitals of the same irrep are closer than
`DEGENERACY_TOL` = 1e-4 Eh. The threshold comes from three measurements on
N2:

- Fock noise is ~1.4e-9, from how far symmetry-degenerate partners split.
- `Srs` sensitivity is ~2.3e-5 Eh/rad, from the rotation test above.
- So a 1e-4 gap costs ~3e-10 Eh.

The smallest real same-irrep gap in the committed goldens is 2.5e-2
(cc-pVTZ). H2O sto-3g CAS(4,4) and CO cc-pVDZ CAS(6,6), conventional and DF,
generate. CH4 cc-pVDZ CAS(4,4) is refused: its active space cuts a t2 set and
leaves the E pair split by only 1.5e-6, which would cost ~2e-8 Eh.

### PC fields

Every committed golden **except the two CAS(12,12) files** also carries
`pc_class_energies` (eight, in `CLASSES` order) and `e_pc_total` (one
element): the partially-contracted NEVPT2 answer from block2's
`WickICNEVPT2`, run by `generate_golden.py` (`_block2_pc`) on the same state,
orbitals and CI vector. The header's `e_nevpt2_total` is still the SC total.
**These numbers are block2's, and nothing else checks them**: PySCF has no
PC-NEVPT2. What the generator does check before writing them:

- **block2's SC equals PySCF's SC** that the file already holds, per class,
  to 1e-7 (`BLOCK2_SC_TOL`), so block2 is seeing this state, these
  integrals and these RDMs. Measured: ≤9.8e-10 on every file.
- **PC Sijrs equals SC Sijrs** to 1e-7 (no active index, so one function).
- **The PC answer is well-defined** (`_check_pc_conditioning`). block2 solves
  each external tuple by `lstsq(rcond=None)`, a pseudo-inverse cut at
  `eps · d · s_max`. If that cut falls in a gap of the singular-value
  spectrum, any cut in the gap gives the same energy. If it falls in a
  continuum, the golden would record one implementation's choice. The guard
  is the per-tuple gap ratio (smallest kept over largest dropped singular
  value), which must be at least `PC_MIN_GAP` = 1e3 in every block. The
  tightest block per file:

  | golden | tightest block | largest dropped | smallest kept | gap |
  |---|---|---|---|---|
  | `n2_ccpvdz_cas44` / `_df` | `irabpq1` (Sir) | 7.3e-17 / 5.7e-17 | 1.7e-5 | 3.2e11 / 3.8e11 |
  | `n2_ccpvdz_cas88` / `_df` | `rabcpqg*` (Sr) | 6.5e-16 / 1.0e-15 | 1.7e-10 | 3.6e5 / 2.7e5 |
  | `n2_ccpvdz_cas1010` / `_df` | `iabcpqg*` (Si) | 1.6e-16 / 1.1e-16 | 1.8e-10 / 1.7e-10 | 1.1e6 / 1.7e6 |
  | `n2_ccpvtz_cas1010_df` | `iabcpqg*` (Si) | 9.5e-16 | 2.5e-11 | 2.6e4 |
  | `n2_ccpvdz_cas1012` | `rabcpqg*` (Sr) | 2.8e-15 | 8.1e-10 | 3.9e5 |
  | `n2_ccpvdz_cas1212` (**refused**) | `rabcpqg*` (Sr) | 3.8e-13 | 4.5e-13 | **1.3** |

  These are relative singular values of `H_D`, maxed or minned over the
  block's tuples. The gap is the minimum of the per-tuple ratio, so it need
  not equal the quotient of the two columns. CAS(12,12)'s Si block is refused
  too, at 7.5. Every generator run prints this per block, and the device PC
  solve picks its own cut inside the gap.

  **Why a gap ratio, and not a raised-cut test.** Re-solving each block with
  the cut raised to a fixed 1e-10 and requiring the energy to stay within
  1e-9 wrongly refuses `n2_ccpvtz_cas1010_df` (Sr moves 6.4e-8). The effect
  is real but is not ambiguity: cc-pVTZ's real modes start at 2.5e-11
  relative, so a fixed 1e-10 cut drops physics. The gap ratio has no fixed
  scale.

**N2 CAS(12,12) is refused**: Sr's metric is a continuum through the cut,
and PC Sr moves ~2e-7 between defensible cuts (the cause is a 1s orbital in
the active space; see [`pc-nevpt2.md`](pc-nevpt2.md)). So both CAS(12,12)
files are generated with `--no-pc`, and the 12-orbital PC case is
`n2_ccpvdz_cas1012`, CAS(10,12) with both N 1s orbitals in the core. `--pc`
refuses a golden without PC fields before the RDM build starts.

Adding the PC fields leaves every other array and the header bit-identical.
Their cost is block2's dense `n^8` 4-RDM (`fci.rdm.make_dm1234`), built once
and shared by its SC and PC kernels: 2–2.5 min single-threaded at
CAS(10,10), 49 min at CAS(10,12) (63 min for the whole file; measured with
other generator runs sharing the machine, so an upper bound).
`--rdm4-threads N` threads that one build and gives identical bits; the
committed files used 1.

### The file format

A flat, trivially parsed sequence: a magic number, a scalar header, then
named N-d `f64` arrays. It is not `.npz`, because parsing a zip container
and `.npy` headers in C++ would need a dependency or a hand-written zip
reader for no benefit. `write_golden`'s docstring in `generate_golden.py`
has the exact layout, and `src/golden/golden.cppm`/`.cpp` is the reader.

### The gradient sidecar

An analytic NEVPT2 gradient reads MO integrals the energy never touches —
blocks with one index over the **full** MO range
([`gradient-theory.md`](gradient-theory.md), §3). Those live in a **separate**
file beside the golden, `<case>.gradgold`, written by
`generate_golden.py --gradgold` and read by `loadGradGold`
(`src/golden/gradgold.cpp`).

**Why a sidecar and not more arrays in the golden.** The golden format is
extensible in principle — a magic plus an array count — but adding arrays
would change every committed golden's bytes, and "a regenerated file that
differs from the committed one means something changed" is the invariant this
whole directory rests on (above). A sidecar keeps that intact and keeps the
big goldens from growing. Adding `--gradgold` to the two small cases'
generator commands left both `.nevpt2gold` files byte-identical, which is the
check that it did.

**What it carries**, and nothing else: the seventeen one-general-index
MO-integral blocks of [`gradient-theory.md`](gradient-theory.md), §3.3 — as
eighteen arrays, because `(xc|cc)` is stored on its two independent restricted
slices — plus §3.5's full-MO-range matrices (`h`, `heff`, `f`, `f_h`,
`fock_rhf`) and `mo_coeff`: 24 arrays. Three blocks are stored only on the
slices the gradient reaches (`(xc|cc)` with two core labels tied, `(xa|vv)`
and `(xc|vv)` with the two virtual labels coinciding), and the eighteenth
block `(xv|vv)` is **absent** — §3.4. Its absence is a constant factor, not a
change of order, and the two committed cases show exactly how modest that is:

| | CAS(4,4) | CAS(8,8) |
|---|---|---|
| ncore / nact / nvirt (nmo = 28) | 5 / 4 / 19 | 3 / 8 / 17 |
| `*.nevpt2gold` | 0.24 MB | 0.48 MB |
| `*.gradgold` | 1.52 MB | 1.86 MB |
| its two largest rows, `(xv\|cv)` + `(xv\|va)` | 0.73 MB | 0.71 MB |
| the omitted `(xv\|vv)`, had it been stored | 1.54 MB | 1.10 MB |

At CAS(4,4) the one omitted block alone would have outweighed the whole file;
at CAS(8,8), with a smaller virtual space, it would have added 59%. Neither
is a change of asymptotics — which is why the gradient validation set is small
molecules (§3.4, and "The active-space-selection hazard" in the same
document).

**The invariants are asserted before a file is written.** `generate_golden.py`
refuses to write a sidecar whose state is not the one the derivation describes
(§3.6), the way `_check_degeneracy` refuses a noise-dependent SC energy:

- **(O1)** the RHF Fock's core–active, core–virtual and active–virtual blocks
  are zero. Measured on both committed cases, the largest element of any of
  the three is **1.0e-7** (CAS(4,4): core–act exactly 0 — those pairs are in
  different D2h irreps — core–virt 9.4e-8, act–virt 1.0e-7; CAS(8,8): 7.3e-8
  / 5.6e-8 / 1.0e-7). That is the RHF convergence, not machine precision:
  `mf.get_fock()` is built from the converged density while `mf.mo_coeff`
  diagonalises the previous iteration's Fock. It cannot be tightened without
  raising `mf.conv_tol`, which would change the golden state, so the refusal
  threshold (`O1_FOCK_TOL`) is 1e-5 — two decades of headroom, and still four
  decades below what a wrongly selected active space would show.
- **(O2)** the generalized Fock is diagonal inside core and inside virtual,
  and those diagonals reproduce the golden's `e_core` / `e_virt`: **1.8e-13**
  at CAS(4,4), **5.9e-13** at CAS(8,8), against a 1e-10 threshold.
- **(O3)** `F`'s active block splits at the RHF gap: 2 of 4 active orbitals
  below the HOMO at CAS(4,4), 4 of 8 at CAS(8,8), none inside the gap. Both
  are `nelec_rhf/2 − ncore = 7 − ncore`, as §2.6 requires.

**What reads it.** `nevpt2_sc_pseudodensity --kappa-full` — the full orbital
source term `Y_xy`, whose core, virtual and mixed blocks are exactly what the
general-range blocks are for ([`testing.md`](testing.md), "The full orbital
source term"). It is the only consumer, and the three `gpu` entries that run
it are the three committed cases with a sidecar: N2 CAS(4,4), N2 CAS(8,8) and
LiF CAS(4,6). Its first check is the sidecar against itself and the golden —
every one of §3.3's arrays restricted to a slice the golden already holds, and
`f`'s core/virtual diagonal against `e_core`/`e_virt`, which is (O2) seen from
the C++ side.

**Committed, and still to come.** The two small cases are committed:

```bash
uv run python reference_data/generate_golden.py --ncas 4 --nelecas 4 \
    --name n2_ccpvdz_cas44 --gradgold
uv run python reference_data/generate_golden.py --ncas 8 --nelecas 8 \
    --name n2_ccpvdz_cas88 --gradgold
```

**The two gradient-validation molecules carry sidecars too** (see "LiF and
acrolein, the gradient-validation pair" below for their generator commands and
their states). They are the measured illustration of the paragraph above — the
inventory makes a small case cheap and does not make a bigger one cheap:

| | LiF CAS(4,6) | acrolein CAS(6,5) |
|---|---|---|
| ncore / nact / nvirt (nmo) | 4 / 6 / 13 (23) | 12 / 5 / 59 (76) |
| `*.nevpt2gold` | 0.20 MB | 7.1 MB |
| `*.gradgold` | 0.92 MB | 56.94 MB |
| (O1) core–act / core–virt / act–virt | 0.0 / 2.54e-7 / 3.73e-8 | 2.36e-7 / 1.73e-7 / 2.73e-7 |
| (O2) | 4.77e-14 | 9.00e-13 |
| (O3) active orbitals below the RHF HOMO | 2 of 6 | 3 of 5 |

LiF's sidecar is committed; **acrolein's is gitignored**
(`golden/acrolein_def2svp_cas65.gradgold`), the same precedent the big goldens
set: 57 MB is past GitHub's 50 MB warning and past the 10 MB
`check-added-large-files` limit the pre-commit hook enforces. The 7.1 MB golden
beside it **is** committed, so only the gradient half needs regenerating (~31 s,
byte-identical). Both goldens stayed byte-identical with and without
`--gradgold`, as at CAS(4,4)/CAS(8,8).

`--gradgold` is conventional-integral only and is refused with `--df`: the
derivation names the conventional Fock, and a fitted sidecar's (O1)/(O2) would
hold against a different one.

### The numerical gradient

`reference_data/nevpt2_numgrad.py` adds four more arrays to a sidecar: the
**finite-difference** nuclear gradient an analytic NEVPT2 gradient is checked
against. It is offline tooling like `block2_pc_probe.py` — no build and no
test runs it — it rebuilds every displaced state with
`generate_golden._build_state` itself (so each one carries the same
single-threading, 1e-12 Davidson tolerance, auto-detected point group and
irrep-selected active space), and it writes no `*.nevpt2gold` byte. Only the
two gradient-validation cases carry it; the N2 CAS(4,4)/CAS(8,8) sidecars stay
the 24 arrays `--gradgold` writes.

| array | shape | what |
|---|---|---|
| `grad_num` | (natm, 3) | d(`E_CASCI` + `E_corr` SC)/dR, Eh/bohr, Richardson-extrapolated |
| `grad_num_residual` | (natm, 3) | \|R(h/2) − R(h)\|, an error bar on the entry beside it |
| `grad_num_probed` | (natm, 3) | 1.0 where the component was differenced; 0.0 where displacing would change the detected point group, i.e. **zero by symmetry** ([`gradient-theory.md`](gradient-theory.md), §4.3) rather than measured |
| `grad_num_step` | (1,) | the coarse step h in bohr; the stencil is h, h/2, h/4 |

**What makes it possible is the CASCI reference** ("What 'CASSCF input' means
here" above): RHF → CASCI is deterministic at every displaced geometry, so two
displaced energies are two values of one function. A CASSCF reference would
land on a differently-rotated solution at each displacement and the difference
quotient would be meaningless.

**The refusals.** The active space is selected by irrep and the group is
detected per geometry, so a displacement that lowers the symmetry silently
swaps in a different set of orbitals. The script refuses, rather than
differencing it:

```
$ uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65 --components all
atom 0 (O) z, step +0.0025 bohr: the detected point group changes from Cs (used: Cs)
at the reference geometry to C1 (used: C1) under this displacement, ... Refused
rather than differenced (docs/gradient-theory.md, §4.3 ...)
$ echo $?
1
```

The check runs over every step of the stencil before the first displaced SCF,
so a refusal costs nothing. `--components auto` (the default) instead *keeps* those
components out of the finite-difference set and records them as the exact
zeros symmetry makes them — `grad_num_probed` is what says which is which, and
an analytic gradient that does not return zero there is wrong without any
finite difference being needed. A second refusal covers a selection that
reordered across a space boundary at an unchanged group (core/active irrep
labels differing from the reference geometry's); a reorder *within* one irrep
is invisible to both and shows up instead as a residual that will not settle.
Before any of that, the rebuilt reference state is checked against the golden
it claims to differentiate: `E_CASCI` and `E_corr` are **bit-identical**
(0.0) on both cases, and a difference past 1e-10 is refused as a wiring error.

**The step is measured, not guessed.** Per component the stencil is six
displaced energies — `g(h) = [E(+h) − E(−h)]/2h` at h, h/2 and h/4 — and two
Richardson extrapolations `R(h) = [4 g(h/2) − g(h)]/3`, of which the finer is
stored and their difference is the residual. Bit-reproducible is not smooth:
each energy is exact but carries its own convergence tolerance, and a
difference quotient divides that by the step. Max residual over acrolein's
`0x`, `1y`, `4y`, `7y` (its worst components), against h:

| h (bohr) | 0.04 | 0.02 | 0.01 | 0.005 | 0.0025 | 0.00125 | 0.000625 |
|---|---|---|---|---|---|---|---|
| max residual | 2.8e-6 | 1.9e-6 | 3.1e-7 | 6.0e-8 | 4.5e-9 | 1.3e-9 | 4.5e-9 |

Truncation falls off until 1/2h amplifies the energies' own convergence floor
through, and at 0.000625 the residual turns back up. The extrapolants at
0.0025 and 0.00125 agree to 1.4e-9, so **0.0025 bohr** is the shipped step
(`STEP_BOHR`): one halving above the minimum, where the stored value has
converged and the residual is not yet reading the floor.

**What the two cases come out at**, both at 0.0025 bohr:

| | LiF CAS(4,6) | acrolein CAS(6,5) |
|---|---|---|
| components probed / zero by symmetry | 6 / 0 | 16 / 8 |
| displaced CASCI+NEVPT2 runs | 36 | 96 |
| wall, single-threaded | 4.8 s | 61.8 s |
| the nontrivial component | −0.0072358529 (Li z), +0.0072358528 (F z) | +0.0266016617 (C2 y), largest of the 16 |
| max Richardson residual | 1.1e-10 | 4.5e-9 |
| free checks: \|g\| where zero by symmetry | ≤1.1e-10 (the four non-axial) | exact 0, not probed |
| free check: sum over atoms per column | ≤6.3e-11 | ≤4.6e-9 |

A diatomic is linear however it is displaced, so **every** LiF component
survives the point-group check and the four non-axial ones are differenced
rather than skipped — which is why they appear above as a measured ≤1.1e-10
instead of an exact zero. Acrolein's eight out-of-plane components are the
refused ones, kept out of the set by `--components auto`. The column sums are
translational invariance, which the generator reports and nothing enforces.

**Regenerating a sidecar is two commands**, `generate_golden.py --gradgold`
and then this script; re-running the second one reproduces the file byte for
byte (it replaces its own arrays rather than appending). The committed LiF
sidecar is sha256 `ae8de2cb…` after both, and acrolein's gitignored one
`dbafd4da…`; the 24-array intermediates are still `093a7fbe…` / `8001e0c5…`
("LiF and acrolein, the gradient-validation pair" below), and both file sizes
are unchanged to the 0.92 MB / 56.94 MB above.

### The analytic gradient

`reference_data/nevpt2_grad.py` is the other half: the **Z-vector solve and
the AO contraction**, over a `pyscf.grad.lagrange.Gradients` subclass. Like
`nevpt2_numgrad.py` it is offline tooling — no build, no test and no binary in
`src/` or `apps/` runs it, and it writes no committed file — and that is the
design, not an accident: this repo's shape is "offline Python generates the
reference, the C++ GPU demo reproduces it", so a PySCF solve inside the
runtime would invert it. The derivation it implements is
[`gradient-theory.md`](gradient-theory.md), §1–§2; what the C++ side does and
does not carry is [`gradients.md`](gradients.md), §1.

There is no `pyscf/grad/nevpt2.py` — that absence is the gap, and the four
hooks the base class leaves to a child are exactly the four pieces:

| hook | here |
|---|---|
| `get_wfn_response` | `y_I = dE_corr/dc_I` and `A_pq = Y_pq − Y_qp`, plus the CI relaxation and the (O2)–CI coupling |
| `get_Aop_Adiag` | `P (H − E_CASCI) P` from the CASCI reference, block-diagonal against the constraint Jacobian — the decoupling of §2.2 |
| `get_ham_response` | `d(E_CASCI + E_corr)/dx` at frozen wavefunction parameters |
| `get_LdotJnuc` | the multipliers against `dF/dx`, `df/dx` and `d<z\|H\|c>/dx` |

**It is a Lagrangian gradient with differenced skeleton derivatives**, and the
distinction from `grad_num` is the point. With the multipliers in hand the
whole gradient is the nuclear derivative of one scalar along a **frozen
orthonormal path** `C(x) = S(x)^{−1/2} S(x₀)^{1/2} C(x₀)` at fixed CI vector
— four evaluations per Cartesian component, against `nevpt2_numgrad.py`'s six
**re-converged** RHF+CASCI+NEVPT2 runs. Nothing re-solves, so every component
is differenced, including the ones that are zero by symmetry (which
`nevpt2_numgrad.py` must skip, §4.3): they come out as the zeros they are and
are reported as the free check. Replacing the quotients with PySCF's
derivative integrals needs the NEVPT2 **relaxed densities** — `Y_xy` fixes
only the antisymmetric part of the generalized Fock, and the AO contraction
needs the densities themselves — which `src/gradient/` builds and frees
unread ([`gradients.md`](gradients.md), §5.2).

**What the two cases come out at**, source terms from PySCF, 0.0025 bohr:

| | LiF CAS(4,6) | acrolein CAS(6,5) |
|---|---|---|
| `kappa` directions = constraints | 246 | 2846 |
| CI Z-vector residual `\|P(H−E)Pz + Py\|` | 1.404e-12 | 2.006e-12 |
| orbital residual `\|M λ + B\|`, direct / GMRES | 1.0e-15 / 1.5e-12 | 5.9e-15 / 8.7e-12 |
| the orbital system's condition number | 8.2e4 | 1.2e4 |
| CI/(O2) fixed point | 6 iterations, last step 1.9e-15 | 6 iterations, last step 3.7e-15 |
| `z·c` after projection | 4.0e-19 | −5.7e-19 |
| gauge identity (§1.3, no multiplier) | 2.0e-6 | 2.7e-4 |
| Richardson bar, skeleton / multipliers | 2.2e-7 / 1.8e-9 | 5.6e-7 / 1.2e-8 |
| `\|g\|` on the zero-by-symmetry axes | 1.7e-15 | 3.0e-11 |
| sum over atoms per column | ≤5.6e-11 | ≤1.0e-8 |
| **max `\|g − grad_num\|`** | **5.3e-9** (6 components) | **7.6e-8** (16 in-plane) |
| wall, single-threaded | 56 s | 1029 s |

The cost is the `kappa` sweep: one central difference of the SC-NEVPT2 energy
per non-gauge rotation, `O(n_mo²)` of them — 937 s of acrolein's 1029 s.
That is what makes the source terms self-contained and is the one thing a GPU
dump would replace.

The last row is **not** a gate here, and its tolerance is issue #41's to
state: `grad_num` is itself a finite difference (its own error bar is 1.1e-10
on LiF and 4.5e-9 on acrolein, "The numerical gradient" above), and the
analytic side's own bar is the Richardson row. Acrolein's `grad_num` is read
from `nevpt2_numgrad.py --no-write`, since its sidecar is gitignored.

**How it is validated**, since the analytic-versus-numerical comparison and
its tolerance are issue #41's and not a gate here: `--casci-only` drops
`E_corr` from the source terms and from the scalar, leaving a plain CASCI
whose analytic gradient PySCF already has. That one comparison exercises the
frozen path, the full constraint Jacobian, the selection-boundary multipliers,
the CPHF block and the differentiation at once. Against the **numerical
CASCI** gradient (`nevpt2_numgrad.py`'s own `grad_casci`, which it prints but
does not store) it lands at **2.0e-8** on LiF and **5.3e-8** on acrolein.

**`pyscf.grad.casci` is not an oracle for either committed case**, and the run
says so rather than reporting a disagreement as a failure. It divides its
redundant core–act_O and act_V–virtual blocks by orbital-energy denominators
assuming act_O is the *first* `n_elec/2 − n_core` active columns;
`--cas-irreps` orders the active block by irrep, so on LiF PySCF's own answer
lands **4.7e-4** from the numerical CASCI gradient while this pipeline lands
2.0e-8. `--energy-ordered` drops the irrep selection and restores the
configuration PySCF assumes; the two analytic gradients then agree to
**1.7e-5** on acrolein. (Calling it at all needs
`mo_coeff=mc.mo_coeff, ci=mc.ci`: its default is `mc._scf.mo_coeff`, the
*unsorted* RHF orbitals, which do not describe one state together with
`mc.ci`.)

**Four things an implementation has to get right, none of which shows up in a
residual.** All four were found by the checks above, and all four leave the
Z-vector residuals at round-off while moving the gradient by 1e-2 to 1e-5:

1. **The act_O–act_V rotation is not a gauge.** It leaves the energy alone —
   it is an active–active rotation, so §1.3 applies — but it **moves the RHF
   occupied space**, so `F_{act_O,act_V} = 0` is a genuine Brillouin
   condition with a multiplier of its own. act_O and act_V must therefore be
   index ranges, which is why the state is rotated into the basis that
   diagonalises `F`'s active block (§2.6) rather than carrying act_O as a
   subspace of the natural-orbital basis.
2. **The CI multiplier must be projected against the state**, not merely
   solved in a projected operator. `|c>` is an exact null vector of
   `P (H − E) P`, so a multiple of it moves no residual, and a Jacobi
   preconditioner does not commute with the projector — the solve comes back
   with `z` almost *parallel* to the state, and the `− (dE_CASCI/dx)(z·c)`
   term the Lagrangian drops is then not zero.
3. **`y_I` must be the projected gradient, taken along directions orthogonal
   to the state.** The SC energy off the unit sphere is not the energy of any
   state (the hole-side metrics' constants are the hole RDMs only at unit
   norm, [`gradients.md`](gradients.md), §7) and it has poles there, so a
   quotient along a bare determinant axis walks off the sphere and returns a
   number whose own linearity fails — by 4e-2 at the dominant determinant on
   acrolein, against 1e-7 for every pair of determinants not containing it.
   It converges over three decades while doing so.
4. **`y_I` must carry the orbital energies' own CI derivative.**
   `e_core`/`e_virt` are the generalized Fock's diagonal and so are functions
   of the active density; the device's `y_I` holds them fixed.

## Density fitting

`nevpt2_df_demo` computes SC-NEVPT2 from three-index `(L|pq)` tensors
instead of four-index MO blocks.

**Its oracle is PySCF's own DF-NEVPT2, not the conventional answer.**
`generate_golden.py --df` density-fits the *whole* calculation (DF-RHF,
`DFCASCI` on the same `with_df`, and `dfnevpt2._ERIS` called directly; its
docstring says why not through `NEVPT.kernel`) with one auxiliary basis,
`<basis>-jkfit` by default (`--auxbasis` overrides it). Different integrals
give a different CASCI state, so a `*_df` golden is a different reference.
Two checks that it is PySCF's DF-NEVPT2 and nothing of ours:

- On one DF-CASCI state, the generator's per-class sum and PySCF's own
  `mrpt.NEVPT(mc).kernel()` (which logs "Using density fitting integrals")
  agree **exactly** (difference 0.0).
- The generator refuses to write a file whose `B_*` blocks do not rebuild
  PySCF's four-index DF blocks to 1e-10.

The DF and conventional answers differ by the fitting error: at
CAS(10,10)/cc-pVDZ, E_corr is −0.1947020047 (DF) against −0.1946519722
(conventional), 5.0e-5 Eh apart. So a DF result is only ever checked against
a `*_df` file.

**Only four three-index blocks are needed**: `B_aa`, `B_ca`, `B_va` and
`B_cv`. No class has two virtuals in one charge distribution, so there is no
`(L|vv)`. The active `h2e` is rebuilt exactly from `B_aa` and handed to the
*same* `nevpt2.rdm_build` (the f3 digests touch only that block), and every
class energy is the *same* einsum transcription (`nevpt2.energy`). What
differs is where each class's external block comes from:

- each class walks its external index (virtual for Sr/Sijr/Srsi/Srs/Sir,
  core for Si/Sijrs/Sij) in `--batch`-sized slabs and asks an
  `IntegralSource` for one slab at a time;
- `DfIntegralSource` builds a slab as one `wwrblasDgemm` over the auxiliary
  index plus one permutation into the physicist layout (`src/df_integrals/`;
  the table of which `B` feeds which block is in `df_integrals.cpp`), uses it
  and frees it, so no full external block ever exists on the device;
- Srsi pairs `(r,s)` with `(s,r)`, so it also asks for its block batched
  along the *second* virtual (`SrsiT`) and keeps the small `(v,v,c)` output
  whole;
- Sijrs (MP2-like, no RDM) is a device reduction over slabs, in **both**
  demos.

**The DF blocks are checked against the four-index arrays too.**
`--check-blocks` builds every block whole and compares it against the
four-index arrays a `*_df` file also carries: all ≤ 3.5e-16 on both cards.
Because those arrays are there, the unmodified `nevpt2_demo` also runs on a
DF file (`nevpt2_cas1010_df_blocks`), a check of the golden file that
involves no DF code at all. Every DF ctest entry agrees with its golden to
≤ 3e-14 Eh.

**`--batch` is a memory lever, and at the committed sizes it buys nothing
measurable.** RTX 3080, energy stage (slab builds included), `--tiles 3`:

| `--batch` | cc-pVDZ CAS(10,10): largest slab / time | cc-pVTZ CAS(10,10): largest slab / time |
|---|---|---|
| 0 (whole blocks) | 0.195 MB / 0.77s | 1.758 MB / 1.20s |
| 8 (the default) | 0.098 MB / 0.87s | 0.293 MB / 1.30s |
| 3 | 0.037 MB / 0.94s | 0.110 MB / 1.39s |
| 1 | 0.012 MB / 1.11s | 0.037 MB / 2.22s |

The integral-direct path holds all external blocks at once: 0.44 MB
(cc-pVDZ), 2.7 MB (cc-pVTZ). Batching cuts the peak as designed and the
energies do not move, but on a 10 GB card nothing here needs it. Whether DF
wins anything, memory or time, needs a system with a large core × virtual
space (N2 has ncore=2, or 1 at CAS(12,12)), and that has not been measured.
The default `--batch 8` was picked to put several slabs through every
virtual-indexed block of the committed cases, not tuned. The cc-pVTZ golden
(nvirt=48) exists to exercise slab boundaries, including a ragged last slab
at `--batch 5`, not memory pressure.

## The small cases

`n2_ccpvdz_cas44` and `n2_ccpvdz_cas88` and their `_df` twins are CAS(4,4)
and CAS(8,8) on the same molecule and basis as the production files, made
with the same generator settings (0.2–0.8 MB each). They exist for the
sanitizer tier, because compute-sanitizer is too slow for the production
sizes (see [`testing.md`](testing.md)), and they are ordinary `small`-labelled
entries in `ctest --preset fast` too. CAS(8,8) is the one with all eight
classes non-zero. They are also the two cases that carry a gradient sidecar
(see "The gradient sidecar" above), for the same reason: small enough that the
general-range integral inventory is cheap.

## Salicylaldimine CAS(8,8)

The one golden pair that is not N2: salicylaldimine (2-methanimidoylphenol,
C7H7NO), CAS(8,8) in 6-31G**, and its density-fitted twin with
6-31G**-RIFIT. Same generator and settings (single-threaded, 1e-12 Davidson,
point-group symmetry), so everything above applies.

**Inputs, committed under `reference_data/`:**

- `geometries/salicylaldimine.xyz`: PubChem CID 260739's 3D conformer
  (MMFF94), with z set to exactly 0 (max |z| was 3.0e-4 Å) so that PySCF
  detects **Cs**. It is not optimised at any level of theory, and it is the
  conformer PubChem ships: the OH and the imine NH point *away* from each
  other, so it is **not** the intramolecularly H-bonded O–H···N enol. Nothing
  here checks geometry-dependent physics, only that the GPU reproduces
  PySCF and block2 on this state.
- `basis/6-31gss-rifit.nw`: 6-31G**-RIFIT from the Basis Set Exchange
  (NWChem format, H/C/N/O; PySCF ships no RIFIT for the Pople sets and reads
  the file by path through `--auxbasis`), 539 auxiliary functions here. The
  `CARTESIAN` line in it is ignored: PySCF builds both bases spherical (161
  AOs).

**The active space is picked by irrep** (`--cas-irreps`, `--core-irreps`).
The default, the 8 RHF orbitals around the Fermi level, is 6 π plus the N
lone pair and a σ*. The π space is 4 π + 4 π*, all A″, and needs the core
spelled out as well: PySCF's own core guess (the 28 lowest orbitals) contains
RHF orbital 27 (A″) and not 29 (A′, occupied), so with `{'A"': 8}` alone the
active space comes out as 3 occupied π + 5 virtual, and the occupied σ
orbital 29 drops out of both core and active space. With
`--core-irreps "A'=27,A\"=1"` the active orbitals are RHF 27/28/30/31
(occupied) and 32/33/35/42 (virtual), and the lowest π stays in the core.

```bash
uv run python reference_data/generate_golden.py \
    --atom reference_data/geometries/salicylaldimine.xyz --basis '6-31g**' \
    --ncas 8 --nelecas 8 --cas-irreps 'A"=8' --core-irreps "A'=27,A\"=1" \
    --name salicylaldimine_631gss_cas88
#   ... and with --df --auxbasis reference_data/basis/6-31gss-rifit.nw
#   --name salicylaldimine_631gss_cas88_df for the DF twin
```

| | conventional | DF (6-31G**-RIFIT) |
|---|---|---|
| ncore / nact / nvirt | 28 / 8 / 125 | 28 / 8 / 125 |
| n_det | 4900 | 4900 |
| E_CASCI | -398.5308286866 | -398.5575060086 |
| E_corr SC (PySCF) | -1.2004269587 | -1.1970345741 |
| E_corr PC (block2) | -1.2016254186 | -1.1982267679 |
| block2 SC − PySCF SC, max class | 1.9e-10 | 1.9e-10 |
| smallest PC gap (block, ≥ 1e3 needed) | 1.1e7 (`iabcpqg*`) | 9.5e6 (`iabcpqg*`) |
| file size raw / xz -9 | 145 MB / 67 MB | 166 MB / 70 MB |
| generation, single-threaded | 155 s | 149 s |

**The DF E_CASCI is 27 mEh below the conventional one.** That is fitting
error, not a bug. RIFIT is an RI-MP2 basis, and the generator fits *every*
integral with the one `--auxbasis`, the RHF's Coulomb and exchange included,
which a JK basis is built for and RIFIT is not. The DF golden is still a
valid oracle, being PySCF's DF-NEVPT2 on exactly those fitted integrals,
which is what `nevpt2_df_demo` reproduces. But its absolute energies are not
a DF approximation of the conventional ones to the usual ~1e-5.

**Both files are gitignored.** `cvcv` alone is (28·125)² doubles = 98 MB, so
both are past GitHub's 100 MB limit raw and past its 50 MB warning even under
xz. Regeneration from a clean run is **byte-identical** (sha256 `1055cd8b…`,
`8b6a1b8d…`), so the commands above are the reference. Their four ctest
entries are registered only when the files exist.

**What they check**, `--tiles 3`, both cards (RX 9060 XT with its default
BLAS digest, RTX 3080 with its default emitted digest), all PASS:

| entry | E_corr \|delta\| HIP / CUDA | wall HIP / CUDA |
|---|---|---|
| `nevpt2_salicylaldimine_cas88` | 7.6e-15 / 3.3e-15 | 35.3 s / 1.2 s |
| `nevpt2_salicylaldimine_cas88_pc` | 1.4e-11 / 1.4e-11 vs block2 (8/8 classes, 0 refused) | 2.1 s / 0.8 s |
| `nevpt2_df_salicylaldimine_cas88` (`--check-blocks`, every block ≤ 8.3e-16 on both) | 5.1e-15 / 3.3e-15 | 38.3 s / 1.3 s |
| `nevpt2_df_salicylaldimine_cas88_pc` | 1.3e-11 / 1.3e-11 vs block2 (8/8, 0 refused) | 2.1 s / 0.8 s |

**The SC energy stage is ~75× slower on HIP than on CUDA here, and only
here.** The integral-direct SC stage takes 34–37 s on the RX 9060 XT against
0.46 s on the RTX 3080, while N2 CAS(8,8) takes 0.22 s on the same AMD card.
PC's stage is not affected (1.6 s HIP, 0.24 s CUDA). `--profile` puts the
HIP time in the same einsums that lead on CUDA, those with a small output
and a large contraction, each 70–100× slower:

| einsum | outSize | contracted | HIP ms | CUDA ms |
|---|---|---|---|---|
| `ipqr,rpqbac,iabc->i` | 125 | 262144 | 6115.6 | 64.2 |
| `rsqp,rsba,pqab->rs` | 15625 | 4096 | 5685.2 | 53.9 |
| `rsqp,rsba,pqba->rs` | 15625 | 4096 | 5643.7 | 54.8 |
| `ipqr,pqrabc,iabc->i` | 125 | 262144 | 4510.4 | 66.2 |
| eight `...->ir` (Sir) einsums | 3500 | 4096 | ~1460–1550 each | — |

The likely cause, **not verified**: `device_einsum` runs one thread per
(output, contracted index) pair and folds them with `atomicAdd`, so these
shapes pile up to 262,144 fp64 atomics onto each of 125 addresses, and fp64
atomic contention on gfx1200 costs far more than on the RTX 3080. N2 never
shows it, because its nvirt (16–48) keeps those outputs and contractions
small. The two SC entries are labelled `slow` because of the HIP time.

## LiF and acrolein, the gradient-validation pair

The two molecules the analytic-gradient work is validated on, chosen so the
comparison has an **external** reference: Park reports both, with the same
active-space sizes, in the paper the derivation follows
([`references.md`](references.md), "Analytical gradients"; his Table 2 lists
acrolein (6e,5o) among the optimised systems, his Table 3 LiF (4e,6o) among the
analytic-vs-numerical gradient tests). So a later disagreement is this tree's
bug, not open physics. Same generator and settings as everything above
(single-threaded, 1e-12 Davidson, point-group symmetry auto-detected),
`_check_degeneracy` passing on both, and both carry a gradient sidecar (see
"The gradient sidecar" above).

**Inputs, committed under `reference_data/geometries/`:**

- `lif.xyz`: Li and F on the z axis at the experimental equilibrium bond
  length, r_e = 1.5639 Å (Huber & Herzberg). Linear, so the detected group is
  **Coov**, mapped to **C2v** by `LINEAR_SUBGROUP` — the same path N2's Dooh →
  D2h takes, and the reason the π partners land in different irreps (B1, B2)
  where symmetry, not noise, pins the rotation between them.
- `acrolein.xyz`: prop-2-enal, s-trans, PubChem CID 7847's 3D conformer
  (MMFF94) with z set to exactly 0 (max |z| was 4.0e-4 Å) so that PySCF detects
  **Cs**, exactly as `salicylaldimine.xyz` does. Not optimised at any level of
  theory. Planar Cs is what the gradient work needs: an in-plane displacement
  cannot break the σ_h, so the `A'`/`A"` labels the active space is written in
  survive it ([`gradient-theory.md`](gradient-theory.md), §4.3).

**Both active spaces are picked by irrep** (`--cas-irreps`), not by orbital
energy, for the same §4.2 reason: an energy-ordered selection can reorder under
a displacement. PySCF's own core guess is correct for both (the `ncore` lowest
orbitals are exactly the ones wanted), so neither needs `--core-irreps`.

- **LiF CAS(4,6)** is the **pure π space**, `B1=3,B2=3`: the occupied F 2p π
  pair plus two π* pairs, with the σ bond left in the core. The
  energy-ordered CAS(4,6) (`σ*`, `σ*` in place of the second π* pair) is
  **not usable**: two of its active natural occupations come out at 1.7e-7 and
  6.7e-9, below PySCF's `FRAC_OCC_THRESHOLD` (1e-6), so `canonicalize` rotates
  those active orbitals together with the virtuals — PySCF warns "Active
  orbitals [2 3] ... are canonicalized with external orbitals", and its own
  comment there says the zeroth-order Hamiltonian is then strongly affected.
  The consequence is measurable: on that state block2's SC `Si` came out
  **1.4e-7** from PySCF's, past `BLOCK2_SC_TOL`, and the generator refused the
  file. The π space's smallest active occupation is 6.8e-6, above the
  threshold, and block2's SC then agrees with PySCF's to **1.9e-15**.
- **Acrolein CAS(6,5)** is `A'=1,A"=4`: the oxygen in-plane lone pair (A′)
  plus the two occupied π and the two lowest π* (A″). The selected orbitals are
  RHF 12, 13, 14, 15 and 18 — note the 18: the second π* is **not** the orbital
  an energy-ordered CAS(6,5) would take (RHF 16 is an A′ σ*), which is the
  whole reason this case is specified by irrep.

| | LiF | acrolein |
|---|---|---|
| geometry / basis | `lif.xyz`, def2-SVP | `acrolein.xyz`, def2-SVP |
| detected group → used | Coov → C2v | Cs → Cs |
| active space | CAS(4e,6o), `B1=3,B2=3` | CAS(6e,5o), `A'=1,A"=4` |
| nao = nmo / ncore / nvirt | 23 / 4 / 13 | 76 / 12 / 59 |
| n_det | 225 | 100 |
| E_CASCI | -106.8522555004 | -190.6559118485 |
| E_corr SC (PySCF) | -0.1968408342 | -0.5466581859 |
| E_corr PC (block2) | **refused** (below) | -0.5473408112 |
| block2 SC − PySCF SC, max class | 1.9e-15 | 4.4e-15 |
| smallest PC gap (block, ≥ 1e3 needed) | **1.7** (`iabcpqg*`) | 7.4e5 (`ijabpq+`) |
| file size, golden / sidecar | 0.20 MB / 0.92 MB | 7.1 MB / 56.94 MB |
| generation, single-threaded | 1.6 s | 31.0 s |

**LiF's PC answer is refused, and that is a finding, not a workaround.**
`_check_pc_conditioning` rejects three of its thirteen PC blocks —
`iabcpqg*` (Si) at a gap of 1.7, `rsabpq-` at 5.3, `rsabpq+` at 49, against the
1e3 the guard needs — for the same reason N2 CAS(12,12) is refused: the lstsq
cut falls in a continuum of singular values (`iabcpqg*`: largest dropped
3.5e-14, smallest kept 4.9e-14, relative), so the PC energy would record one
implementation's choice of cut. Four of LiF's six active orbitals are
essentially empty (occupations 3.0e-3, 3.0e-3, 6.8e-6, 6.8e-6), which is what
flattens the PC metric; it is a property of a 4-electron space on an ionic
closed-shell diatomic rather than of this basis or bond length — the
energy-ordered CAS(4,6)'s smallest active occupation stayed between 9e-10 and
4e-8 in all four bases probed (def2-SVP, cc-pVDZ, 6-31G*, cc-pVTZ) at each of
r = 1.5639, 1.7 and 2.0 Å. **The π space's margin is basis-dependent, and
def2-SVP is on the right side of it**: its smallest occupation is 6.8e-6,
against 6.9e-7 in cc-pVDZ, which would land back under
`FRAC_OCC_THRESHOLD` and bring the mixed canonicalisation back. So the LiF golden is
generated with `--no-pc` and carries SC fields only, and there is deliberately
no `nevpt2_lif_cas46_pc` ctest entry. Acrolein carries both, and its PC gaps
are four or more decades clear of the guard.

**What they check**, `--tiles 3` (not a measured floor: one tile fits at these
active-space sizes, as for the N2 CAS(8,8) entries), RTX 3080, default emitted
digest, all PASS:

| entry | E_corr \|delta\| | max class \|delta\| | wall |
|---|---|---|---|
| `nevpt2_lif_cas46` | 1.6e-14 vs PySCF | 1.5e-14 | 0.04 s |
| `nevpt2_acrolein_cas65` | 1.8e-15 vs PySCF | 9.8e-16 | 0.04 s |
| `nevpt2_acrolein_cas65_pc` | 4.6e-14 vs block2 (8/8 classes, 0 refused) | 4.8e-14 | 0.08 s |

Device pool high-water for the whole run is 2.2 MB (LiF), 7.8 MB (acrolein SC)
and 10.3 MB (acrolein PC). The HIP side of these three entries has not been
measured, so none of them is labelled `slow`.

All four files regenerate **byte-identical** from the commands above (goldens
sha256 `ef97ad2c…` / `cb8e3228…`, sidecars `093a7fbe…` / `8001e0c5…`), and the
goldens are byte-identical with `--gradgold` dropped. Those two sidecar hashes
are the 24-array files `--gradgold` writes; both sidecars then get the
numerical gradient added by a second command, which is what the committed LiF
file holds (see "The numerical gradient" above for the hashes after it, the
step, and the residuals — the measured in-plane gradient of each molecule is
there and nowhere else).

## CAS(14,14)

`n2_ccpvdz_cas1414` (n_det = 11.8M) is the next head-to-head point (see
[`performance.md`](performance.md), "Head-to-head vs PySCF"), and it is
**not committed** (`.gitignore`'s `golden/*cas1414*`): regenerate it with
`generate_golden.py --ncas 14 --nelecas 14 --no-pc --name n2_ccpvdz_cas1414`
(`--no-pc`: block2's dense `n^8` 4-RDM is ~12 GB at 14 orbitals, and like
CAS(12,12) the active space holds a 1s orbital; neither the command nor
the PC question has been run at this size). The raw
file is ~95 MB. Its CI vector is dense, high-entropy f64 (no exact zeros;
99% of the coefficients are below 1e-6 but not zero), so gzip is useless
(1.03×) and xz reaches only 71 MB (1.34×), still past GitHub's 50 MB warning.
A committed golden must stay lossless, being the element-exact reference, so
a lossy sparse or thresholded store, which *would* shrink it to a few MB, is
not used.

## The files in `golden/` and `reference_data/`

| Path | Role |
|---|---|
| `reference_data/generate_golden.py` | the **only** file that defines the golden state (`block2_pc_probe.py` imports its `_build_state`). Runs a deterministic `mcscf.CASCI` (see "What "CASSCF input" means here" above), canonicalises it exactly as `pyscf.mrpt.NEVPT.kernel` does, and writes the CI vector, `dm1`/`dm2`, every MO-integral block, and PySCF's own per-class NEVPT2 answer into one flat binary file. `--df` does the same density-fitted and adds the `B_*` blocks. Unless `--no-pc`, it also runs block2's PC-NEVPT2 on the same state and writes `pc_class_energies` / `e_pc_total` (see "PC fields" above) |
| `reference_data/block2_pc_probe.py` | rebuilds a golden's state with `generate_golden._build_state` (imported, not copied), checks it is bit-identical to the committed CI vector, and runs block2's `WickSCNEVPT2` and `WickICNEVPT2` (PC) on it: per-class energies, timings, peak RSS, and the PC linear-solve / metric conditioning (`--metric`). Writes no golden. See [`pc-nevpt2.md`](pc-nevpt2.md) |
| `reference_data/nevpt2_numgrad.py` | the **numerical** nuclear gradient: displaced RHF → CASCI → SC-NEVPT2 runs (states built by `generate_golden._build_state`, imported) differenced and Richardson-extrapolated into a sidecar's `grad_num` / `grad_num_residual` / `grad_num_probed` / `grad_num_step`. Refuses a displacement whose detected point group — or whose core/active irrep selection — differs from the reference geometry's, rather than produce a number from a different active space. Writes no golden and no new sidecar; only the two gradient-validation cases carry its arrays (see "The numerical gradient" above) |
| `reference_data/nevpt2_grad.py` | the **analytic** nuclear gradient: a `pyscf.grad.lagrange.Gradients` subclass that solves the CI and orbital Z-vector equations of [`gradient-theory.md`](gradient-theory.md), §2 and contracts them into a gradient, differentiating one scalar along a frozen orthonormal path. Source terms from PySCF by default, or from a `.gradsrc` dump (`--source dump`) when the GPU side grows one. Writes **nothing** unless `--out` is given, and never a committed file; `--casci-only` is its own validation (see "The analytic gradient" above) |
| `reference_data/casscf_repro_probe.py` | the **CASSCF reproducibility** measurement: ≥5 back-to-back RHF → symmetry-adapted `mcscf.CASSCF` → orbital-gradient → fresh `mcscf.CASCI` → SC-NEVPT2 runs under `generate_golden`'s own settings (imported, so single-threaded with the same Davidson tolerance), reporting the spread of E_CASSCF / E_corr / ‖g_orb‖ and whether any run changed stationary point. `--perturb EPS` starts the optimiser from a random same-irrep rotation instead. Writes no golden; its output is what "A fresh CASSCF reproduces, measured" above quotes, and where the `--mo-from` gate thresholds come from |
| `reference_data/pyscf_contract_time.py` | the CPU side of the head-to-head: reads a golden file and times PySCF's `NEVPTcontract` (`make_dm123` + `_contract4pdm` ×2) on the *same frozen CI vector* the demo reads, with no CASCI re-solve (see [`performance.md`](performance.md), "Head-to-head vs PySCF") |
| `golden/n2_ccpvdz_cas1010.nevpt2gold` | the **default case**: CAS(10,10) on N2/cc-pVDZ |
| `golden/n2_ccpvdz_cas1212.nevpt2gold` | CAS(12,12) on the same molecule, the scaling case (see [`performance.md`](performance.md), "Scaling with active-space size"). `slow`-labelled; run with `--tiles 40`. Carries **no** PC fields: its PC metric is ungapped (see "PC fields" above) |
| `golden/n2_ccpvdz_cas1012.nevpt2gold` | the 12-orbital **PC** case: CAS(10,12) on the same molecule (both N 1s in the core, n_det = 627k), with SC and PC fields. Exists because CAS(12,12) cannot be a PC golden. Its SC fields pass `nevpt2_demo --tiles 40` (CUDA, \|ΔE_corr\| 4e-15); its ctest entry is the PC one, `nevpt2_cas1012_pc` (`--pc --tiles 40`, `slow`) |
| `golden/n2_ccpvdz_cas1010_df.nevpt2gold`, `golden/n2_ccpvdz_cas1212_df.nevpt2gold`, `golden/n2_ccpvtz_cas1010_df.nevpt2gold` | the **density-fitted** goldens (`--df`, aux basis `<basis>-jkfit`), the oracle for `nevpt2_df_demo`: PySCF's DF-NEVPT2 as the answer, plus the four three-index `B_*` blocks. A different reference from the conventional files (see "Density fitting" above). The cc-pVTZ one (nvirt = 48) exists for slab boundaries, not memory pressure |
| `golden/n2_ccpvdz_cas44.nevpt2gold`, `golden/n2_ccpvdz_cas88.nevpt2gold` and their `_df` twins | the **small** cases (see "The small cases" above) |
| `golden/n2_ccpvdz_cas44.gradgold`, `golden/n2_ccpvdz_cas88.gradgold` | the **gradient sidecars** for those two cases: the general-range MO-integral blocks an analytic gradient reads, plus the full-range Fock matrices and `mo_coeff` (see "The gradient sidecar" above). Separate files, so the goldens beside them are untouched. `test/golden/gradgold_tests.cpp` round-trips the reader and checks the CAS(4,4) one against the golden beside it |
| `reference_data/geometries/salicylaldimine.xyz`, `reference_data/basis/6-31gss-rifit.nw`, `golden/salicylaldimine_631gss_cas88[_df].nevpt2gold` | the non-N2 case (see "Salicylaldimine CAS(8,8)" above). The two goldens are **gitignored** and regenerate byte-identically in ~2.5 min each; their four ctest entries register only once they exist |
| `reference_data/geometries/lif.xyz`, `golden/lif_def2svp_cas46.nevpt2gold`, `golden/lif_def2svp_cas46.gradgold` | the **LiF smoke test** of the gradient-validation pair (see "LiF and acrolein, the gradient-validation pair" above): linear, Coov → C2v, CAS(4,6) as the pure π space by irrep, def2-SVP. **SC fields only** — its PC metric is refused as ungapped, so no `--pc` entry exists for it. Both files are committed (0.20 / 0.92 MB), and the sidecar carries the **numerical gradient** on top of the 24 arrays `--gradgold` writes, so it takes two commands to regenerate (see "The numerical gradient" above) |
| `reference_data/geometries/acrolein.xyz`, `golden/acrolein_def2svp_cas65.nevpt2gold` | the **acrolein** half of the pair: planar Cs, CAS(6,5) = the oxygen lone pair + 2 π + 2 π*, def2-SVP, with SC **and** PC fields. 7.1 MB, committed |
| `golden/acrolein_def2svp_cas65.gradgold` | **not committed** (gitignored, 57 MB — the one file of the pair that is): acrolein's gradient sidecar. Regenerate it with the `--gradgold` command in "Regenerating the golden file" followed by the `nevpt2_numgrad.py` one for its numerical gradient; the golden beside it is unaffected |
| `golden/n2_ccpvdz_cas1414.nevpt2gold` | **not committed**; see "CAS(14,14)" above |
