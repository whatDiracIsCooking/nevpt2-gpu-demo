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

**Every committed golden's state is a plain `mcscf.CASCI` diagonalisation** —
a linear-algebra step with no orbital optimiser anywhere in the chain, which
reproduces bit for bit given the two settings in "Reproducibility" below. What
differs between the committed files is *which orbitals* that CASCI runs on,
and the header's `reference_kind` says which ("The file format" below):

| `reference_kind` | orbitals | committed files |
|---|---|---|
| `casci` | converged RHF | every golden without `_casscf` in its name |
| `casscf` | a converged `mcscf.CASSCF`, committed in an MO sidecar | the `*_casscf*` files ("The MO sidecar: committed CASSCF orbitals" below) |

**The CASSCF-orbital references are built in two steps, and the sidecar is the
design.** `--casscf --mo-out FILE` runs the optimiser **once** and commits its
converged `mo_coeff` to a `*.mogold` file; `--mo-from FILE` then runs the same
deterministic CASCI every other golden is built with, on those committed
orbitals. So a golden's state depends on a *file*, never on an optimiser in
the regeneration chain. Three things that buys:

- **Provenance.** The sidecar carries the fingerprint of the calculation it
  belongs to — geometry in the symmetry frame, basis, AO/MO counts, point
  group, `ncore`/`ncas`/`nelecas`, the irrep selection, `E_CASSCF` — and
  `--mo-from` rebuilds all of it from its own flags and refuses on any
  difference, reporting every mismatch at once.
- **Independence from the optimiser.** The committed orbitals pin the
  reference across PySCF versions, convergence settings and starting points.
  Regenerating the golden re-runs only the deterministic part, so the property
  this directory rests on survives: a regenerated file that differs from the
  committed one means something changed.
- **Cost.** The optimisation happens once, not on every regeneration.

**None of that is a reproducibility argument.** A fresh `mcscf.CASSCF` *does*
reproduce under these settings — the next subsection is that measurement, and
it refutes an earlier claim in this file that it does not. What optimising the
orbitals changes is the **reference**: E_corr moves by tens of millihartree
(the last table in "The `--casscf` / `--mo-from` gate thresholds"), a
different number rather than a noisier one, which is why both kinds are worth
committing.

**What a CASSCF-orbital golden is not an input for: the GPU's Lagrangian.**
The gradient derivation has a branch per reference
([`gradient-theory.md`](gradient-theory.md), §5): on RHF orbitals the
orbital constraint is RHF Brillouin plus the active-space selection, solved
by CPHF; on CASSCF orbitals it is Park's Eq. 23 as written, CASSCF
stationarity, solved as one coupled CP-CASSCF system (§1.4, §2.7). The
second branch is built **offline only**, in
`reference_data/nevpt2_grad.py --reference casscf` (SC-NEVPT2 with PySCF's
own source terms, or with `--pc` PC-NEVPT2 with block2's; "The CASSCF
branch" below). What the GPU computes is the part the two branches share:
the source terms and the energy's parameter derivatives (tangents and
Hessian blocks), taken at fixed orbitals and CI vector, which carry no
reference-kind dependence. Those paths sit behind the source-term gate and
**take** a `casscf` golden (`nevpt2_demo --sc-tangent` / `--gradient-source`,
`nevpt2_sc_pseudodensity --dump-source-terms` / `--source-oracle`, and the
Hessian flags; "The file format" below); only the Lagrangian gate refuses
one, and no on-card path calls it yet
([`gradient-theory.md`](gradient-theory.md), §1.1).
`reference_data/nevpt2_numgrad.py`'s default `--reference casci` route
refuses a `casscf` golden too; its `--reference casscf` route re-optimises
every displaced state instead ("The numerical gradient of a CASSCF
reference" below). One more exception
is data, not a derivative: `--gradgold` is
permitted with `--mo-from` (refused with `--casscf`), and writes the
one-general-index integral arrays on the committed CASSCF orbitals, the
input the second branch will read, at the price of a weaker provenance
contract ("The CASSCF-reference gradient sidecar" below).

Whichever orbitals it is handed, the generator never writes `dm3`, `f3ac` or
`f3ca`. Producing those from the CI vector is
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

This measurement is about **one fixed geometry**. The precondition
[`gradient-theory.md`](gradient-theory.md), §4.4 names for a numerical
gradient on CASSCF orbitals — that each *displaced* optimisation lands on the
same stationary branch as the reference — is a different claim about a
different calculation, measured instead by `nevpt2_numgrad.py --reference
casscf`'s continuity guard and cold-vs-warm probe ("The numerical gradient of
a CASSCF reference" below).

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
committed **CASCI** goldens are built on:

| orbitals | ‖g_orb‖ | E_CASCI | E_CASCI − E_CASSCF |
|---|---|---|---|
| CAS(4,4), RHF | 6.98e-2 | −109.017971290758 | +2.82e-2 |
| CAS(10,10), RHF | 7.54e-2 | −109.048231643881 | +8.15e-2 |

(both E_CASCI values are the committed CASCI goldens' own `e_casci`, which is
what those files hold: CASCI on RHF orbitals.)

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

**Both are implemented as written**, in `generate_golden.py`:
`CASSCF_GRAD_TOL` (1e-6) and `CASSCF_E_TOL` (1e-9), checked by
`_check_casscf_stationary`, with `CASSCF_CONV_TOL` 1e-12 /
`CASSCF_CONV_TOL_GRAD` 1e-7 / `CASSCF_MAX_CYCLE_MACRO` 200 on `--casscf`'s own
optimisation. One number moves in the move from probe to gate, and it is worth
knowing which: the gate evaluates the gradient with the RDMs of the
**generator's own** CASCI — the 1e-12-Davidson state the golden records —
rather than the optimiser's warm-started CI vector, and that CI vector's
residual floors at ~1e-6 ("Reproducibility" below). So the same converged
orbitals measure about a decade higher through the gate than through the
probe:

| case | ‖g_orb‖, optimiser's CI (the probe) | ‖g_orb‖, the generator's CASCI (the gate) | \|E_CASCI − E_CASSCF\| |
|---|---|---|---|
| CAS(4,4) | 5.860e-9 | 9.153e-8 | 1.705e-13 |
| CAS(10,10) | 2.521e-8 | 7.513e-8 | 8.527e-14 |
| CAS(4,4), DF | 5.786e-9 | 9.037e-8 | 1.847e-13 |

Still 11–13× inside the refusal, and the goldens those runs wrote are the
**committed** CASSCF-orbital files ("The MO sidecar: committed CASSCF
orbitals" below), carrying E_corr −0.207071732083 and −0.126182539677 — the
probe's numbers to every digit it prints. That is the second
argument against tightening the refusal to 1e-7: that would leave the real
gate well under 2× of headroom, not 4×.

**On re-optimised displaced states the gate is unchanged, and it bit.**
`nevpt2_numgrad.py --reference casscf` builds every displaced state through
`_build_state`'s `--mo-from` path, so this gate refuses there exactly as it
does here. With `generate_golden`'s optimiser alone, acrolein's displaced
states measured 1.1e-7 … 1.2e-6 on it while the optimiser's own norm was
≤ 1.1e-8, and two of 24 were refused (1.016e-6 and 1.201e-6). The gate's
number is not dominated by the fresh CASCI; `mc1step`'s orbitals are
stationary only for its own floored CI vector. The numerical gradient
therefore finishes each displaced state to ~1e-9 on this gate, and the
threshold stays untouched ("Why the displaced states are finished" below).
The same mechanism puts the committed CASSCF goldens' `E_corr` 7.2e-9 (LiF)
and 1.3e-8 (acrolein) from a fully converged one, inside the demos' 1e-7.

**Where "tens of millihartree" actually comes from.** The CASCI-vs-CASSCF
*reference* difference, which is real, is the whole reason to want
CASSCF-orbital goldens — and both columns are now committed files, the twin
pairs the demos are checked against:

| case | E_corr, the CASCI golden (RHF orbitals) | E_corr, the CASSCF golden | difference |
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

### The CASSCF polish

**Under the settings above, `mc1step` does not converge LiF CAS(4,6) or
acrolein CAS(6,5)** (#122 found it; "LiF and acrolein on CASSCF orbitals"
below is the case pair). Both run the full 200 macro iterations and stop
just above `conv_tol_grad` — ‖g_orb‖ 1.2e-7 (LiF) and 2.2e-7 (acrolein), the
optimiser's own norm — so `--casscf` refused both. The log of a stalled
macro iteration (LiF, `verbose` 5) says what is happening: `|u−1|` is
exactly 0, `dE` exactly 0, and `max_stepsize_scheduler` shrinks the step
cap by 0.3 every iteration (to ~1e-17 by macro 39), because a step that
does not lower the energy by `conv_tol` (1e-12) counts as an energy
increase. The augmented-Hessian solver no longer produces a step worth
taking. Our reading of PySCF's `soscf.ciah.davidson_cc` fits this: it
accepts an AH solution once its residual is under
`sqrt(ah_conv_tol)` = 1e-6 at PySCF's default `ah_conv_tol` (1e-12), and
that is not a small residual once the gradient is ~1e-7 itself.

**What was tried, and what converged.** Each probe was run from the
generator's own RHF + `--cas-irreps` start, or from the stalled point (its
orbitals and CI vector), single-threaded, offline CPU only:

| change (nothing else touched) | start | LiF | acrolein |
|---|---|---|---|
| none (the committed settings) | RHF | stalls, 200 macro | stalls, 200 macro |
| `ah_lindep` 1e-14 → 1e-20 | RHF, and stalled point | stalls | stalls |
| FCI `lindep` 1e-14 → 1e-16 | RHF | stalls | stalls |
| FCI `conv_tol_residual` 1e-9 + `lindep` 1e-20 | stalled point | stalls | stalls |
| `mcscf.newton` (second-order, CI coupled) | RHF | stalls (CI gradient 2.9e-7) | stalls at ‖g‖ 3.4e-4, E 6.8e-6 **above** the `mc1step` point |
| `mcscf.newton` + FCI `lindep` 1e-16 | RHF | converged, 10 macro | stalls, as the row above |
| `mcscf.newton` + FCI `lindep` 1e-16 | stalled point | stalls | stalls |
| **`ah_conv_tol` 1e-12 → 1e-16** | RHF | converged, 11 macro | converged, 157 macro |
| **`ah_conv_tol` 1e-12 → 1e-16** | **stalled point** | **converged, 2 macro** | **converged, 2 macro** |

`ah_conv_tol` is the one knob that matters (PySCF's own docstring names it
and `ah_lindep` as the accuracy controls); 1e-15, 1e-16 and 1e-18 give the
same E_CASSCF to 12 decimals and the same ‖g_orb‖ to 4 digits, from either
start, so 1e-16 sits inside a plateau.

**Why a fallback and not the setting.** With `ah_conv_tol` 1e-16 applied
from the start, a fresh `--casscf` reproduces the committed
`n2_ccpvdz_cas44_casscf.mogold` and `n2_ccpvdz_cas44_casscf_df.mogold` byte
for byte but **not** `n2_ccpvdz_cas1010_casscf.mogold`: CAS(10,10) then
lands at ‖g_orb‖ 2.635e-9 instead of 2.521e-8, a different, equally
converged point inside the same tolerance. A regenerated file that differs
is this directory's signal that something changed, so the setting stays
PySCF's and the tight AH solve is a **polish**: `optimise_mo` runs it only
when the first optimisation has not converged, as a second `mc1step` run
warm-started from the first one's orbitals and CI vector, every other
setting unchanged (`CASSCF_POLISH_AH_CONV_TOL` in `generate_golden.py`).
Every committed N2 sidecar converges first time, so none of them runs it,
and the line `--casscf` prints says `+ polish macro=N` when one did. If the
polish does not converge either, `--casscf` refuses as before. No guard is
touched by it: `conv_tol_grad`, `CASSCF_GRAD_TOL` and the `--mo-from` gate,
`_check_degeneracy` and `_check_pc_conditioning` all run unchanged on its
output.

**The polish moves the stalled point by almost nothing**, which is the
check that it finishes the same optimisation rather than finding another
one: against the stalled point, E_CASSCF moves by +1.4e-14 (LiF) and
−1.3e-11 (acrolein), and the active-subspace overlap singular values are
≥ 0.999999999999 (LiF) and ≥ 0.9999999975 (acrolein).

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
# ... and their _df twins, with the density-fitted sidecar (see "The
# density-fitted gradient sidecar")
uv run python reference_data/generate_golden.py --df --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44_df --gradgold
uv run python reference_data/generate_golden.py --df --ncas 8 --nelecas 8 --name n2_ccpvdz_cas88_df --gradgold
# ... then the density-fitted numerical gradient into the CAS(4,4) one (see
# "The density-fitted numerical gradient"); CAS(8,8)'s carries none
uv run python reference_data/nevpt2_numgrad.py --df --case n2_ccpvdz_cas44
# ... and the density-fitted ANALYTIC gradient gated against it, which writes
# nothing (docs/testing.md, "The density-fitted gradient against the numerical one")
uv run python reference_data/nevpt2_grad.py --df --case n2_ccpvdz_cas44

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
# file at all (see "The analytic gradient") and gates itself against grad_num
uv run python reference_data/nevpt2_grad.py --case lif_def2svp_cas46
uv run python reference_data/nevpt2_grad.py --case acrolein_def2svp_cas65

# the PC pair, and the symmetric N2 case -- neither writes a sidecar, so the
# numerical half goes to an .npz (docs/testing.md, "The analytic gradient
# against the numerical one" has all of these and what they print)
uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65 \
    --pc --npz /tmp/acrolein.numgrad.npz
uv run python reference_data/nevpt2_grad.py --case acrolein_def2svp_cas65 \
    --pc --grad-num /tmp/acrolein.numgrad.npz
uv run python reference_data/nevpt2_numgrad.py --case n2_ccpvdz_cas44 \
    --pc --npz /tmp/n2_cas44.numgrad.npz
uv run python reference_data/nevpt2_grad.py --case n2_ccpvdz_cas44 \
    --pc --grad-num /tmp/n2_cas44.numgrad.npz

# PySCF's exact SC source terms for two of those states, the oracle the
# nevpt2_sc_pseudodensity_oracle_* entries hold the GPU's to (see "The SC
# source-term oracle")
uv run python reference_data/sc_source_oracle.py --case n2_ccpvdz_cas44
uv run python reference_data/sc_source_oracle.py --case lif_def2svp_cas46
# ... and block2's PC ones for N2 CAS(4,4), golden/n2_ccpvdz_cas44_pc.srcgold
# (see "The PC flavour")
uv run python reference_data/sc_source_oracle.py --case n2_ccpvdz_cas44 --pc
# ... and the directional flavour, golden/<case>[_pc]_dir.srcgold (see "The
# directional flavour"): acrolein SC and PC, and N2 / LiF beside their full ones
uv run python reference_data/sc_source_oracle.py --case acrolein_def2svp_cas65 --directional
uv run python reference_data/sc_source_oracle.py --case acrolein_def2svp_cas65 --directional --pc
uv run python reference_data/sc_source_oracle.py --case n2_ccpvdz_cas44 --directional
uv run python reference_data/sc_source_oracle.py --case n2_ccpvdz_cas44 --directional --pc
uv run python reference_data/sc_source_oracle.py --case lif_def2svp_cas46 --directional
# ... and all four kinds on N2 CAS(4,4)'s CONVERGED CASSCF orbitals,
# golden/n2_ccpvdz_cas44_casscf[_pc][_dir].srcgold (see "On CASSCF orbitals"
# under "The SC source-term oracle"); the sidecar is the committed one
uv run python reference_data/sc_source_oracle.py --case n2_ccpvdz_cas44_casscf \
    --mo-from golden/n2_ccpvdz_cas44_casscf.mogold            # also with --pc,
                                                              # --directional [--pc]
# ... and the directional two on acrolein CAS(6,5)'s CASSCF orbitals,
# golden/acrolein_def2svp_cas65_casscf[_pc]_dir.srcgold (#135)
uv run python reference_data/sc_source_oracle.py --case acrolein_def2svp_cas65_casscf \
    --mo-from golden/acrolein_def2svp_cas65_casscf.mogold --directional    # also with --pc

# the three goldens on CONVERGED CASSCF orbitals, each in two steps (see "The
# MO sidecar"). The FIRST command of each pair rewrites the committed sidecar
# and is only needed if the sidecar itself is being rebuilt; with the sidecar
# in hand, the second command alone regenerates the golden byte for byte. The
# same case flags go to both: they are the sidecar's fingerprint, and a
# mismatch is refused
uv run python reference_data/generate_golden.py --ncas 4 --nelecas 4 \
    --name n2_ccpvdz_cas44_casscf --casscf \
    --mo-out golden/n2_ccpvdz_cas44_casscf.mogold
uv run python reference_data/generate_golden.py --ncas 4 --nelecas 4 \
    --name n2_ccpvdz_cas44_casscf \
    --mo-from golden/n2_ccpvdz_cas44_casscf.mogold
# ... and the same with --gradgold, which also writes its gradient sidecar,
# golden/n2_ccpvdz_cas44_casscf.gradgold (see "The CASSCF-reference gradient
# sidecar"; the golden comes out byte-identical either way)

uv run python reference_data/generate_golden.py --ncas 10 --nelecas 10 \
    --name n2_ccpvdz_cas1010_casscf --casscf \
    --mo-out golden/n2_ccpvdz_cas1010_casscf.mogold
uv run python reference_data/generate_golden.py --ncas 10 --nelecas 10 \
    --name n2_ccpvdz_cas1010_casscf \
    --mo-from golden/n2_ccpvdz_cas1010_casscf.mogold

uv run python reference_data/generate_golden.py --df --ncas 4 --nelecas 4 \
    --name n2_ccpvdz_cas44_casscf_df --casscf \
    --mo-out golden/n2_ccpvdz_cas44_casscf_df.mogold
uv run python reference_data/generate_golden.py --df --ncas 4 --nelecas 4 \
    --name n2_ccpvdz_cas44_casscf_df \
    --mo-from golden/n2_ccpvdz_cas44_casscf_df.mogold

# the gradient-validation pair on CONVERGED CASSCF orbitals (see "LiF and
# acrolein on CASSCF orbitals"), same two steps; both --casscf runs go through
# the polish ("The CASSCF polish"), and the --mo-from step writes the gradient
# sidecar too. Acrolein's .gradgold is gitignored (57 MB): this is how to get it
uv run python reference_data/generate_golden.py \
    --atom reference_data/geometries/lif.xyz --basis def2-svp \
    --ncas 6 --nelecas 4 --cas-irreps "B1=3,B2=3" --no-pc \
    --name lif_def2svp_cas46_casscf --casscf \
    --mo-out golden/lif_def2svp_cas46_casscf.mogold
uv run python reference_data/generate_golden.py \
    --atom reference_data/geometries/lif.xyz --basis def2-svp \
    --ncas 6 --nelecas 4 --cas-irreps "B1=3,B2=3" --no-pc \
    --name lif_def2svp_cas46_casscf \
    --mo-from golden/lif_def2svp_cas46_casscf.mogold --gradgold

uv run python reference_data/generate_golden.py \
    --atom reference_data/geometries/acrolein.xyz --basis def2-svp \
    --ncas 5 --nelecas 6 --cas-irreps "A'=1,A\"=4" \
    --name acrolein_def2svp_cas65_casscf --casscf \
    --mo-out golden/acrolein_def2svp_cas65_casscf.mogold
uv run python reference_data/generate_golden.py \
    --atom reference_data/geometries/acrolein.xyz --basis def2-svp \
    --ncas 5 --nelecas 6 --cas-irreps "A'=1,A\"=4" \
    --name acrolein_def2svp_cas65_casscf \
    --mo-from golden/acrolein_def2svp_cas65_casscf.mogold --gradgold

# the C1 case (see "HSOH, the C1 case"): CASSCF orbitals only, the same two
# steps, then the re-optimised states' numerical gradient written into the
# sidecar the second step wrote (the one CASSCF sidecar committed with one).
# --atom is given as this relative path: the .mogold keeps it as typed
uv run python reference_data/generate_golden.py \
    --atom reference_data/geometries/hsoh.xyz --basis def2-svp \
    --ncas 6 --nelecas 6 --name hsoh_def2svp_cas66_casscf --casscf \
    --mo-out golden/hsoh_def2svp_cas66_casscf.mogold
uv run python reference_data/generate_golden.py \
    --atom reference_data/geometries/hsoh.xyz --basis def2-svp \
    --ncas 6 --nelecas 6 --name hsoh_def2svp_cas66_casscf \
    --mo-from golden/hsoh_def2svp_cas66_casscf.mogold --gradgold
uv run python reference_data/nevpt2_numgrad.py --case hsoh_def2svp_cas66 \
    --reference casscf --workers 12
```

**Regeneration is a check.** Every committed golden regenerates
byte-identical, CAS(12,12) included (generated twice in separate processes,
~8 min each, single-threaded). A regenerated file that differs from the
committed one means something changed.

**For the small N2 cases the check is a script**, `devtools/golden-repro.sh`:
it reruns the commands above for every committed N2 CAS(4,4) / CAS(8,8)
reference — the four goldens, the two gradient sidecars, the two CASSCF
`.mogold` sidecars (a fresh `--casscf` each), the two `*_casscf[_df]`
goldens (`--mo-from` the committed sidecar) and the eight N2 `.srcgold`
oracles (full and directional, SC and PC, on RHF and on CASSCF orbitals) —
and the C1 case's three HSOH files ("HSOH, the C1 case" below) into a scratch directory, compares each with the committed file byte for byte
(the `G1` CASCI goldens by the header exception below, nothing looser), and
prints `PASS:` / `FAIL:` per file. It makes no claim over
`n2_ccpvdz_cas44_casscf.gradgold`, whose contract is a tolerance ("The
CASSCF-reference gradient sidecar" below). What it covers, why the `.srcgold` files are
in it, and its wall time are in [`testing.md`](testing.md), "The golden
regeneration gate".

**The CASSCF-orbital files regenerate byte-identically from their
committed sidecars**, with no exception to make for the header: they were
written after the `reference_kind` field existed, so they are `NEVPT2G2`
already and `cmp` compares them whole. Measured, each regenerated from
`golden/<case>.mogold` into a scratch `--out` and compared with the committed
file (CUDA devcontainer, PySCF 2.14.0, single-threaded):

| golden | size | regenerated | sha256 (both) |
|---|---:|---|---|
| `n2_ccpvdz_cas44_casscf.nevpt2gold` | 238 653 B | identical | `79f564e8…` |
| `n2_ccpvdz_cas44_casscf_df.nevpt2gold` | 470 637 B | identical | `8d36d59c…` |
| `n2_ccpvdz_cas1010_casscf.nevpt2gold` | 1 130 061 B | identical | `f8f00352…` |
| `lif_def2svp_cas46_casscf.nevpt2gold` | 202 138 B | identical | `28367a4b…` |
| `acrolein_def2svp_cas65_casscf.nevpt2gold` | 7 121 421 B | identical | `007c5a7e…` |
| `hsoh_def2svp_cas66_casscf.nevpt2gold` | 1 455 261 B | identical | `5c12261f…` |

(LiF and acrolein measured on the RTX 3080 box's CPU at #123, same PySCF, each
with and without `--gradgold`; "LiF and acrolein on CASSCF orbitals" has
their sidecars' hashes. HSOH by `golden-repro.sh` at #213, on the two-card
box's CPU; "HSOH, the C1 case" has its sidecars' hashes.)

**One exception, and it applies to the pre-kind files only.** The committed
**CASCI** goldens are written in the `NEVPT2G1` layout; the generator now
writes `NEVPT2G2`, which differs by the magic's last byte and the four-byte
`reference_kind` field at the end of the scalar header ("The file format"
below). So a regenerated CASCI golden is 4 bytes longer than its committed
twin, and the comparison skips the two headers, each file's header length
read off that file rather than written out:

```bash
at() { uv run python reference_data/golden_format.py arrays-at "$1"; }
cmp -i "$(at golden/n2_ccpvdz_cas44.nevpt2gold):$(at /tmp/n2_ccpvdz_cas44.nevpt2gold)" \
    golden/n2_ccpvdz_cas44.nevpt2gold /tmp/n2_ccpvdz_cas44.nevpt2gold
```

Everything after the header — the array count and every array — must still be
byte-identical, and that is what the check means. Nothing in the tree needs
the committed files rewritten: `loadGolden` and every reader in
`reference_data/` accept both magics ("The file format" below lists them).

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
  | `n2_ccpvdz_cas44_casscf` / `_df` | `ijabpq-` (Sij) | 0 / 0 | 1.2e-4 | 9.0e10 / 9.1e10 |
  | `n2_ccpvdz_cas1010_casscf` | `rabcpqg*` (Sr) | 8.8e-16 | 7.1e-10 | 1.1e6 |
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

**One reader per language, and the list of readers.** A header change is
made here, so this is where the readers are written down. One header change
(`G1` → `G2`, a single 4-byte field) once broke three separate readers, found
at three different times, the last only because a rebase happened to precede
a merge — so there is now exactly one parse per language, and every reader
derives where the array table begins from where its header read stopped,
never from a length it knows:

| format | parser (the only one) | reached through |
|---|---|---|
| `.nevpt2gold` | C++: `loadGolden` (`src/golden/golden.cpp`, over the `nevpt2.golden:reader` cursor); `GoldenFile::arraysAt` is the derived offset | every demo's `main` (`apps/integral_direct`, `apps/density_fit`, `apps/pc_tangent`, `apps/sc_pseudodensity`) |
| `.nevpt2gold` | Python: `golden_format.read_golden` / `read_golden_header` (`reference_data/golden_format.py`); `arrays_at` is the derived offset | `nevpt2_numgrad.py` (`read_golden_header`, `read_golden_pc_total`), `nevpt2_grad.py` (the same two), `block2_pc_probe.read_golden`, `pyscf_contract_time.read_golden` |
| `.gradgold` (`NEVPT2D1`; density-fitted `NEVPT2F1`, #164) | C++: `loadGradGold` (`src/golden/gradgold.cpp`); `GradGoldFile::densityFitted()` / `naux` are the flavour | `apps/sc_pseudodensity` (`--kappa-full`, `--dump-source-terms`, `--source-oracle`), `apps/integral_direct` (`--gradient-source --gradgold`) |
| `.gradgold` (both flavours) | Python: `golden_format.read_gradgold` (`naux` in the header only for `NEVPT2F1`) | `nevpt2_numgrad.py` (rewrites the sidecar with `grad_num`), `nevpt2_grad.py` (reads `grad_num`) |
| `.mogold` | Python: `golden_format.read_mo_sidecar` | `generate_golden.read_mo_sidecar` (`--mo-from`, which adds its own checks); no C++ reader — it is generator input |
| `.srcgold` (`NEVPT2S1` SC, `NEVPT2P1` PC; directional `NEVPT2SD` / `NEVPT2PD`, #105) | C++: `loadSrcGold` (`src/golden/srcgold.cpp`); `SrcGoldFile::scheme` and `::flavour` are the kind, `srcGoldCase` the golden a `<case>[_pc][_dir].srcgold` belongs to | `apps/sc_pseudodensity` (`--source-oracle`; a PC one with `--pc`, #74; a directional one through `compareSourceDirections`, #105); `test/source_compare/` (all four kinds, through `nevpt2.source_compare`) |
| `.srcgold` (all four) | Python: `golden_format.read_srcgold` (`header["scheme"]`, `header["flavour"]`), `srcgold_name`, `srcgold_case` | `sc_source_oracle.py` (reads back what it wrote, `--pc` and `--directional` too); `golden_layout_check.py` |

`.gradsrc` (`nevpt2_grad.py --source dump`) is a NumPy `.npz`, not this
format, and is read by `np.load` (`nevpt2_grad.load_source_terms`, and
`load_edot_directions` for its optional `edot_kappa` / `edot_frozen` /
`edot_digest` set, #88 — `--edot-check`). It is
the one file here the C++ side **writes**: `nevpt2_sc_pseudodensity
--dump-source-terms` through `nevpt2.gradsrc` (`apps/sc_pseudodensity/`),
whose `readGradSrc` exists only for that binary's round trip and
`test/gradsrc/`. It is never committed. The other writers are in `generate_golden.py`
(`write_golden`, `write_gradgold`, `write_mo_sidecar`) and `golden_format.py`
(`write_srcgold`), and pack their scalar headers from the same tables
`golden_format.py` reads them by (`GOLDEN_HEADERS`, `GRADGOLD_HEADERS`,
`MO_SIDECAR_HEADERS`, `SRCGOLD_HEADERS`, keyed by magic).

**Changing a header**, then, is: a new magic and a new entry in
`golden_format.GOLDEN_HEADERS` (never an edit to an existing entry — that
misreads every committed file of that layout), the matching reads in
`loadGolden`, and both halves of the layout contract green:

```bash
uv run python reference_data/golden_layout_check.py   # ends PASS:
ctest --preset unit -R GoldenLayoutContractTests        # loadGolden / loadGradGold
```

`golden_layout_check.py` reads every committed `.nevpt2gold` — both layouts
must be present, and are: `G1` for the CASCI goldens, `G2` for the
`*_casscf*` ones — and every `.gradgold`, `.mogold` and `.srcgold`, re-packs each from its
own magic's header table and requires the original bytes back, runs each
through every reader entry point in the table above and requires one answer,
reads back a golden from the writer as it is now, and refuses any parser
outside `golden_format.py` (any byte unpacking, seeking or format magic in
`reference_data/`; any file reading or magic in `src/` or `apps/` outside
`src/golden/`). `test/golden`'s `GoldenLayoutContractTests` loads every
`.nevpt2gold`, `.gradgold` and `.srcgold` in `golden/` with the C++ readers. Why the
Python half is a manual gate and not a ctest entry is in
[`testing.md`](testing.md), "The golden layout contract".

**`reference_kind`: which reference built the orbitals.** The scalar header
ends with an `int32` saying what the state is — `0` CASCI, `1` CASSCF
(`generate_golden.REFERENCE_KINDS`, `ReferenceKind` in
`src/golden/golden.cppm`). Provenance belongs in the file rather than in a
file *name*, because a consumer has to be able to refuse what it does not
support. The C++ side has two gates on it. The **source-term** gate
(`GoldenFile::requireSourceTermReference`) accepts either kind: `Y_xy` and
`y_I` are derivatives of the energy at fixed orbitals and CI vector (Park
Eqs. 26–28, 36–38, 41–48), so `nevpt2_demo`'s `--sc-tangent` and
`--gradient-source` and `nevpt2_sc_pseudodensity`'s `--dump-source-terms`
and `--source-oracle` run on a `casscf` golden, as do the SC and PC Hessian
flags (#157, #203). The **Lagrangian** gate
(`GoldenFile::requireLagrangianReference`, an `err_unsupported`) refuses one,
because the Z-vector → Sylvester → CPHF chain is derived for CASCI on RHF
orbitals — see [`gradient-theory.md`](gradient-theory.md), §1.2 and §2.2–§2.6;
the CASSCF branch, §2.7, exists offline only, in `nevpt2_grad.py`; no
on-card path calls this gate yet — and so does
`reference_data/nevpt2_numgrad.py`'s `--reference casci` route
(`_check_reference_kind`). Both
demos print the kind in their header line.

**Two magics, because the field is newer than most of the files.** `NEVPT2G2`
is the layout with the field, which the five `*_casscf*` goldens are written
in; `NEVPT2G1` is the layout from before it, which every committed **CASCI**
golden is written in and which means CASCI — the only thing it could mean,
since until the CASSCF-orbital path a golden had always been a CASCI on RHF
orbitals. The reader accepts both and loads a `G1` file as CASCI, so adding
the field rewrote no committed file, and committing the first `G2` ones did
not either. An unknown kind code is an IO Error, like a bad
magic: a file this reader cannot interpret, as distinct from a reference it
can read and will not differentiate.

### The MO sidecar: committed CASSCF orbitals

`--casscf --mo-out FILE` runs a symmetry-adapted `mcscf.CASSCF` **once** and
writes its converged `mo_coeff` to `FILE` — a `*.mogold` file, the third
format in this directory. `--mo-from FILE` then builds a golden by running the
*same deterministic `mcscf.CASCI`* every other golden is built with, on those
orbitals instead of RHF's. It writes `reference_kind = casscf` ("The file
format" above) and nothing else about the generator changes: the
canonicalisation, `_check_degeneracy`, the RDMs, PySCF's per-class answer and
block2's PC all run exactly as they do on the RHF path.

**The committed pairs** (regeneration commands in "Regenerating the golden
file" above; the N2 goldens' ctest entries in [`testing.md`](testing.md),
"The CASSCF-orbital entries"; the LiF and acrolein pair, #123, is described
in "LiF and acrolein on CASSCF orbitals" below — LiF's golden has its energy
entry, `nevpt2_lif_cas46_casscf`, and a source-term dump (#125), and
acrolein's is read by the directional source-term oracles (#135), registered
only where its gitignored gradient sidecar has been regenerated):

| sidecar | golden | E_CASSCF | ‖g_orb‖ at `--casscf` | purity defect | macro |
|---|---|---|---|---|---:|
| `n2_ccpvdz_cas44_casscf.mogold` | `n2_ccpvdz_cas44_casscf.nevpt2gold` | −109.0461390747 | 5.860e-9 | 1.3e-14 | 5 |
| `n2_ccpvdz_cas1010_casscf.mogold` | `n2_ccpvdz_cas1010_casscf.nevpt2gold` | −109.1296891843 | 2.521e-8 | 1.8e-14 | 16 |
| `n2_ccpvdz_cas44_casscf_df.mogold` | `n2_ccpvdz_cas44_casscf_df.nevpt2gold` | −109.0458757487 | 5.786e-9 | 5.8e-15 | 5 |
| `lif_def2svp_cas46_casscf.mogold` | `lif_def2svp_cas46_casscf.nevpt2gold` | −106.9035612554 | 3.691e-9 | 1.4e-15 | 200 + 2 (polish) |
| `acrolein_def2svp_cas65_casscf.mogold` | `acrolein_def2svp_cas65_casscf.nevpt2gold` | −190.6794323473 | 5.774e-9 | 1.0e-14 | 200 + 2 (polish) |

and what the goldens they wrote carry (the `--mo-from` gate's own two numbers,
then the state):

| golden | ‖g_orb‖ (the gate) | \|E_CASCI − E_CASSCF\| | E_corr (SC) | E_corr (PC, block2) |
|---|---|---|---|---|
| `n2_ccpvdz_cas44_casscf` | 9.153e-8 | 1.705e-13 | −0.2070717321 | −0.2072316444 |
| `n2_ccpvdz_cas1010_casscf` | 7.513e-8 | 8.527e-14 | −0.1261825397 | −0.1298993211 |
| `n2_ccpvdz_cas44_casscf_df` | 9.037e-8 | 1.847e-13 | −0.2070810097 | −0.2072408933 |
| `lif_def2svp_cas46_casscf` | 5.701e-7 | 2.984e-13 | −0.1473025529 | **refused** (`--no-pc`) |
| `acrolein_def2svp_cas65_casscf` | 1.721e-7 | 3.126e-13 | −0.5221559894 | −0.5229006181 |

The DF pair is fitted throughout — the CASSCF optimisation included
(`--df` is passed to both commands, and the sidecar records the auxiliary
basis as part of its fingerprint), so it is PySCF's DF answer on DF-optimised
orbitals, which is the only self-consistent thing for `nevpt2_df_demo` to be
checked against. The two conventional SC E_corr values are the ones
`reference_data/casscf_repro_probe.py` printed for the same orbitals, to every
digit it prints ("A fresh CASSCF reproduces, measured" above).

**Why commit the orbitals instead of optimising them in line.** Not because a
CASSCF run would wander — it does not, and "A fresh CASSCF reproduces,
measured" above is that measurement. Because the golden's state should depend
on a *file*, not on an optimiser: the committed orbitals pin the reference
across PySCF versions and settings, the expensive optimisation happens once,
and the second step keeps the property the whole directory rests on — a
regenerated golden that differs from the committed one means something
changed. Running `--mo-from` twice on one sidecar gives byte-identical
`.nevpt2gold` files, and each committed CASSCF-orbital golden regenerates
byte-identically from its committed sidecar — the hashes are in "Regenerating
the golden file" above.

**The sidecar is a fingerprint plus coefficients.** Same flat idiom as
`.gradgold` — own magic (`NEVPT2M1`), a scalar header, the same named-array
section — plus one section the other two formats do not have: UTF-8
key/value text, because a fingerprint has to name a basis and a point group.
`write_mo_sidecar`'s docstring in `generate_golden.py` has the exact layout.
It carries `mo_coeff` (already in core/active/virtual column order, so
`--mo-from` hands it to `mcscf.CASCI` unsorted), the geometry as coordinates
in the symmetry frame, the AO and MO counts, `ncore`/`ncas`/`nelecas`, the
detected point group, the basis and auxiliary-basis names, the irrep
selection, and `e_casscf`. `--mo-from` rebuilds all of that from the molecule
and flags in front of it and **refuses on any difference**, reporting every
mismatch at once. Nothing in `src/` reads a `.mogold`: it is an input to the
generator, not a reference for the demos, so it has no C++ reader.

Two fields are recorded and *not* compared: the `--atom` string as typed
(which may be a path to an `.xyz` file whose contents can change under it —
hence comparing coordinates) and the PySCF/NumPy versions (which would turn
every harmless bump into a refusal). A dependency change that actually moves
this state does not need a version compare to be caught: it lands in
`|E_CASCI − E_CASSCF|`, by decades.

**Symmetry is re-derived, and impure orbitals are refused.** `orbsym` comes
from `symm.label_orb_symm` on the committed coefficients, and
`_check_mo_symmetry` first measures how much of each orbital's norm lies
outside its own irrep, refusing above `MO_PURITY_TOL` (1e-9). That refusal
exists because PySCF's own answer to an impure `mo_coeff` is one of two silent
ones: above `label_orb_symm`'s 1e-7 it **replaces** the orbitals with
`symm.symmetrize_space`'s projection of them (a different state than the
sidecar's), and between 1e-9 and 1e-7 it labels them anyway behind a
`logger.warn` that `verbose=0` never prints (and the symmetry-adapted FCI
solver then imposes a symmetry the orbitals do not have). The orbitals
`--casscf` itself wrote measure 1.3e-14 (CAS(4,4)), 1.8e-14 (CAS(10,10)) and
5.8e-15 (the DF CAS(4,4)) through this check — five decades inside the
refusal; those are the committed sidecars' own figures (the table above). With
`--cas-irreps`/`--core-irreps`, the irrep composition of the loaded core and
active windows must also be the one the flags ask for — `--mo-from` does not
re-sort the orbitals, so that is what checks they are ordered as the flags
describe.

**The stationarity gate.** `_check_casscf_stationary` runs alongside
`_check_degeneracy` and takes the same stance — refuse rather than write a
golden whose answer is not the one it claims:

- the CASSCF orbital-gradient 2-norm at the sidecar's orbitals, above
  `CASSCF_GRAD_TOL` (1e-6);
- `|E_CASCI − E_CASSCF(sidecar)|`, above `CASSCF_E_TOL` (1e-9).

Both thresholds, what they measure and the headroom they have are in "The
`--casscf` / `--mo-from` gate thresholds" above. `--casscf` applies the first
of them to its own result too, so `--mo-out` never writes a file `--mo-from`
would turn around and refuse, and it refuses an optimisation that is still
unconverged after its one polish ("The CASSCF polish" above) outright.

**What the three deliberately broken inputs produce** (CAS(4,4), the sidecar
above):

| input | refused by | measured |
|---|---|---|
| RHF orbitals in the sidecar, everything else unchanged | the gradient gate | ‖g_orb‖ 6.982e-2 |
| `--atom "N 0 0 0; N 0 0 1.1001"` | the fingerprint | geometry 1.89e-4 Bohr off |
| `--basis cc-pvtz` | the fingerprint | `nao`/`nmo` 28 → 60, basis name |

`--gradgold` is permitted with `--mo-from` and refused with `--casscf` (which
writes no golden for a sidecar to sit beside); what that costs is "The
CASSCF-reference gradient sidecar" below.

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
source term"), was the first consumer; its four `gpu` entries are N2
CAS(4,4), its CASSCF-orbital twin, N2 CAS(8,8) and LiF CAS(4,6). It is no
longer the only one: `nevpt2_demo --gradient-source [--pc] --gradgold`, and
`nevpt2_sc_pseudodensity`'s `--dump-source-terms`, `--source-oracle`,
`--sc-hessian-kc`, `--sc-hessian-kk-int` and `--sc-hessian-kk` (each also
with `--pc`) read the same sidecar, and the density-fitted flavour has its
own readers ("The density-fitted gradient sidecar" below). Its first check is
the sidecar against itself and the golden —
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

#### The density-fitted gradient sidecar

`--df --gradgold` writes the **density-fitted** flavour (#164), beside a
`*_df` golden: the same `.gradgold` suffix, its own magic `NEVPT2F1`, and the
`NEVPT2D1` header followed by one more `int32`, `naux`
(`golden_format.GRADGOLD_HEADERS`; `loadGradGold` reads both flavours and
reports this one as `GradGoldFile::densityFitted()`). A new magic rather than
more arrays under `NEVPT2D1`, because its integrals belong to a different
calculation, the fitted one. The magic is what says so, and no consumer has
to guess it from which arrays are present. `GradGoldFile::checkAgainst`
refuses a fitted sidecar beside a conventional golden, a conventional one
beside a fitted golden, and an `naux` that is not the golden's `B_aa` leading
extent. [`gradient-theory.md`](gradient-theory.md), §3.8 is the inventory it
carries out:

- **`B_cc` and `B_vv`**, `[naux, ncore, ncore]` and `[naux, nvirt, nvirt]`:
  the two blocks of `(L|pq)` the golden lacks. With the golden's `B_aa`,
  `B_ca`, `B_va` and `B_cv` they make the whole tensor, and the whole tensor
  is what `Y` reads over its full range. They are stored square, like the
  golden's four.
- **§3.5's six matrices** (`mo_coeff`, `h`, `heff`, `f`, `f_h`, `fock_rhf`)
  of the *fitted* calculation, built by the same code as the conventional
  ones (`_gradient_matrices`) out of `dfnevpt2._ERIS` and the density-fitted
  RHF.
- **The four tied orbital-energy arrays**, `g_xa_cc`, `g_xc_ca`, `g_xa_vv`,
  `g_xv_va`, with the conventional sidecar's names and shapes, each `Σ_L B B`.
  They are what `--gradient-source`'s `eps`-response mode reads, with `heff`
  (`OrbitalEnergyResponse`, `loadOrbitalEnergyResponse`). Carrying them under
  the conventional layout means the one C++ reader of that response reads a
  fitted sidecar unchanged. They are redundant with the six `B` blocks.

No other conventional `(xP|QR)` array is carried: the density-fitted gradient
never forms one (§3.8). The generator asserts (O1)–(O3) against the fitted
Fock, the one §3.8 says a fitted gradient's invariants hold against. It also
requires the four tied arrays and `heff` to rebuild the golden's
`e_core`/`e_virt` (`DF_EPS_REBUILD_TOL`, 1e-10 relative) before it writes the
file. Both small cases are committed:

```bash
uv run python reference_data/generate_golden.py --df --ncas 4 --nelecas 4 \
    --name n2_ccpvdz_cas44_df --gradgold
uv run python reference_data/generate_golden.py --df --ncas 8 --nelecas 8 \
    --name n2_ccpvdz_cas88_df --gradgold
```

Measured on those two (cc-pVDZ-JKFIT, `naux` = 140, nmo = 28):

| | CAS(4,4) `_df` | CAS(8,8) `_df` |
|---|---|---|
| ncore / nact / nvirt | 5 / 4 / 19 | 3 / 8 / 17 |
| `*_df.nevpt2gold` | 0.47 MB | 0.79 MB |
| `*_df.gradgold` (12 arrays; CAS(4,4)'s 16 with `grad_num*`, #167) | 0.86 MB | 0.95 MB |
| `B_cc` / `B_vv` | 0.028 / 0.404 MB | 0.010 / 0.324 MB |
| the four tied arrays, `g_xv_va` of them | 0.38 MB, 0.32 MB | 0.58 MB, 0.52 MB |
| the six matrices | 0.04 MB | 0.04 MB |
| (O1) core–act / core–virt / act–virt | 0.0 / 9.28e-8 / 1.00e-7 | 7.32e-8 / 5.57e-8 / 1.00e-7 |
| (O2) | 2.10e-13 | 4.22e-13 |
| (O3) active orbitals below the RHF HOMO | 2 of 4 | 4 of 8 |
| `e_core`/`e_virt` from `heff` + the tied arrays, relative | 7.56e-15 | 1.83e-14 |

Both `_df` goldens stay byte-identical with and without `--gradgold`, and
`devtools/golden-repro.sh` regenerates both sidecars byte for byte, the
CAS(4,4) one with the four `grad_num*` arrays `nevpt2_numgrad.py --df`
appends to it ("The density-fitted numerical gradient" below). The
largest array is `g_xv_va`, the full `(x r | s t)`, and it is the price of
reusing the conventional reader. The response reads only its `r = s` tie,
which `B_vv`'s diagonal and `B_va` already give. A DF-specific builder of
`OrbitalEnergyResponse` would drop it, but at these sizes it costs less than
`B_vv`. `test/golden`'s `GradGoldDfCas44Tests` checks the CAS(4,4) one against
its golden card-free: every element of the four tied arrays against `Σ_L B B`
over the six blocks (which pins `B_cc` and `B_vv` in full, off-diagonal
included, to the golden's fitted tensor), `f` against `e_core`/`e_virt`,
`heff` against the golden's one-electron fields, and the orbital energies
rebuilt from the tied arrays. **What reads it**, so far, is the `eps`-response
mode of `nevpt2_demo --gradient-source --gradgold`, which runs a `*_df`
golden off its four-index arrays: the `gpu;small` entry
`nevpt2_gradient_source_cas44_df` ([`testing.md`](testing.md), "The CI source
term's forward-reverse duality"), and the density-fitted `Y` (#165):
`nevpt2_sc_pseudodensity --kappa-full` on either `*_df` golden reads its
`B_cc`, `B_vv` and three full-range matrices to contract the general-range
inventory, and holds the four tied arrays to what it built
([`gradients.md`](gradients.md), §6.2; `nevpt2_sc_pseudodensity_kappa_full_cas44_df`
and `_cas88_df`). The CAS(4,4) one is also read by the fitted dump,
`nevpt2_sc_pseudodensity_dump_cas44_df` (#168), and by the density-fitted
orbital and mixed Hessian entries: `--sc-hessian-kk-int` (#154) and
`--sc-hessian-kk` (#158), and `--pc` with `--sc-hessian-kk-int`,
`--sc-hessian-kc` and `--sc-hessian-kk` (#200–#202); the CAS(8,8) one by
`--pc --sc-hessian-kk-int` as well. `nevpt2_df_demo`'s own
`--gradient-source` (#162) does not read it: it holds the orbital energies
fixed.

### The CASSCF-reference gradient sidecar

`golden/n2_ccpvdz_cas44_casscf.gradgold` is the gradient sidecar beside the
CASSCF-orbital CAS(4,4) golden: the same 24 arrays `--gradgold` writes for a
CASCI case — the one-general-index integrals of
[`gradient-theory.md`](gradient-theory.md), §3.3, and §3.5's `mo_coeff`, `h`,
`heff`, `f`, `f_h` and `fock_rhf`, with the RHF electron count in the header —
evaluated on the **committed** CASSCF orbitals. `fock_rhf` and the RHF
electron count are carried so the layout is exactly the CASCI sidecars'; the
CASSCF branch reads neither ([`gradient-theory.md`](gradient-theory.md),
§3.5). Without it the GPU source-term
path on a CASSCF golden has to hold the orbital energies fixed; with it, the
complete functional is in reach. The measurement is #121's, not this file's:
[`testing.md`](testing.md), "The source terms on CASSCF orbitals, measured".
It is written by the `--mo-from` step with `--gradgold` added:

```bash
uv run python reference_data/generate_golden.py --ncas 4 --nelecas 4 \
    --name n2_ccpvdz_cas44_casscf \
    --mo-from golden/n2_ccpvdz_cas44_casscf.mogold --gradgold
```

No optimiser runs in that step, so it is as deterministic as every other
`--mo-from` run: run twice into scratch, both sidecars were byte-identical
(1 515 263 B, sha256 `91313fa7…`), and the golden it rewrites beside them was
byte-identical to the committed `n2_ccpvdz_cas44_casscf.nevpt2gold`
(`79f564e8…`, the table in "Regenerating the golden file" above) — adding
`--gradgold` changed no golden byte, as at the CASCI cases.

**The provenance contract, and what it costs.** The `.mogold` design's
argument (top of this file, "The MO sidecar" above) is that a golden's state
depends on a *file*, never on an optimiser in the regeneration chain. That
holds for this sidecar *given* the committed `.mogold`, and no further: the
orbitals it is evaluated on came out of `mcscf.CASSCF`, so regenerating it from
scratch puts an orbital optimiser back in the chain. **This sidecar is
therefore reproducible from scratch to a tolerance, not byte for byte** — the
tolerance being how far an optimiser run's converged orbitals may sit from the
committed ones, which is what the `--mo-from` gate bounds (‖g_orb‖ ≤ 1e-6,
\|E_CASCI − E_CASSCF\| ≤ 1e-9; "The `--casscf` / `--mo-from` gate
thresholds" above) and what "A fresh CASSCF reproduces, measured" shows a
different optimiser route actually does (E_corr within 1.6e-8 at CAS(4,4)). A
fresh `--casscf --mo-out` on the box this file was written on did reproduce the
committed `.mogold` byte for byte, as `golden-repro.sh` checks — so the chain
was in fact exact there — but that is an observation about one PySCF build,
not the contract. Accordingly `devtools/golden-repro.sh` keeps its
byte-for-byte contract over the CASCI set (and the `.mogold` / `*_casscf*`
golden pair it already regenerates) and **makes no claim over this sidecar**;
it does not regenerate it.

**The alternative, considered and rejected:** one `.mogold` per displaced
geometry, which would make every CASSCF-reference array byte-reproducible
from committed files. It buys exactness the gradient tier does not need —
every comparison downstream is a tolerance (1e-7) — at the price of dozens of
sidecars per case (a central difference is two per Cartesian coordinate, more
with Richardson), each one a committed optimiser output to keep in step.

**What it is checked against.** (O2) is asserted exactly as for a CASCI
sidecar — the generalized Fock is pseudocanonical in core and virtual and its
diagonals are the golden's `e_core`/`e_virt`: **3.86e-13**, against 1e-10.
(O1) and (O3) are **not** asserted: they are the CASCI branch's
([`gradient-theory.md`](gradient-theory.md), §3.6), the statement that core,
active and virtual are each spanned by RHF canonical orbitals (§1.2), and
converged CASSCF orbitals are by construction not that. (O1) is still measured
and printed, as a record of how far from RHF-canonical they are: core–act 0
(different D2h irreps), core–virt 9.13e-3, act–virt 2.80e-1. The CASSCF
branch's counterpart, (S1), the residual orbital gradient, is the `--mo-from`
gate's, which has already run on the state.

**What stays refused.** A sidecar's `grad_num` arrays are a *numerical*
gradient, and differencing needs a re-converged reference at every
displacement. `nevpt2_numgrad.py --reference casscf` does exactly that
("The numerical gradient of a CASSCF reference" below), but writes its
gradient to an `--npz` only, so this file carries the 24 `--gradgold` arrays
and no `grad_num`, and stays byte-identical. (The C1 case's sidecar is the
one exception, by its case-table switch: "HSOH, the C1 case" below.) Permitting
`--gradgold` to write integral arrays is not permission to difference
committed orbitals at a displaced geometry. Beyond the layout contract
(`golden_layout_check.py` and `golden_tests`'
`EveryCommittedGradientSidecarLoadsAgainstItsGolden`, which glob every
`*.gradgold`), the ctest tier reads it wherever its CASCI twin's sidecar is
read: `nevpt2_demo --gradient-source [--pc]`, `nevpt2_sc_pseudodensity`'s
`--kappa-full`, `--dump-source-terms [--pc]` and `--source-oracle [--pc]`
against the full and directional oracles (#121, #160), and the orbital and
mixed Hessian entries on this golden, SC (#157) and `--pc` (#203).

**LiF and acrolein have one each since #123**, written the same way by
their `--mo-from … --gradgold` step and under the same contract: LiF's is
committed, acrolein's (57 MB) gitignored. LiF's is read on a card by
`nevpt2_sc_pseudodensity_dump_lif_def2svp_cas46_casscf` (#125); acrolein's,
where regenerated, by its directional source-term oracles (#135). Their (O2)
residuals, (O1)
records and hashes are in "LiF and acrolein on CASSCF orbitals" below.

### The numerical gradient

`reference_data/nevpt2_numgrad.py` adds four more arrays to a sidecar: the
**finite-difference** nuclear gradient an analytic NEVPT2 gradient is checked
against. It is offline tooling like `block2_pc_probe.py` — no build and no
test runs it — it rebuilds every displaced state with
`generate_golden._build_state` itself (so each one carries the same
single-threading, 1e-12 Davidson tolerance, auto-detected point group and
irrep-selected active space), and it writes no `*.nevpt2gold` byte. Only the
two gradient-validation cases and the density-fitted N2 CAS(4,4) sidecar
carry it ("The density-fitted numerical gradient" below); the conventional
N2 CAS(4,4)/CAS(8,8) sidecars stay the 24 arrays `--gradgold` writes, and the
density-fitted CAS(8,8) one the 12 `--df --gradgold` writes.

| array | shape | what |
|---|---|---|
| `grad_num` | (natm, 3) | d(`E_CASCI` + `E_corr` SC)/dR, Eh/bohr, Richardson-extrapolated |
| `grad_num_residual` | (natm, 3) | \|R(h/2) − R(h)\|, an error bar on the entry beside it |
| `grad_num_probed` | (natm, 3) | 1.0 where the component was differenced; 0.0 where displacing would change the detected point group, i.e. **zero by symmetry** ([`gradient-theory.md`](gradient-theory.md), §4.3) rather than measured |
| `grad_num_step` | (1,) | the coarse step h in bohr; the stencil is h, h/2, h/4 |

**`--pc` differences the partially-contracted energy on the same displaced
states**, adding `grad_num_pc` and `grad_num_pc_residual` — block2's
`WickICNEVPT2` on each one, gated per state by the golden path's own
`_check_pc_conditioning`, so a displaced state whose PC metric has gone
ungapped is refused rather than differenced. It writes **no sidecar**: the
sidecar's contract is the 24 arrays `--gradgold` writes plus the four SC
`grad_num*` ones, and the committed files hold exactly that. `--npz PATH`
writes the gradient arrays on their own instead, which is also how a case
whose committed sidecar deliberately holds no `grad_num` (N2 CAS(4,4)) reaches
`nevpt2_grad.py --grad-num`. One PC evaluation costs far more than the whole
SC sweep around it, so `--pc` is opt-in and reports both gradients from one
set of displaced states rather than being a second run. **LiF has no PC
answer at all** and `--pc` on it stops at the reference state, quoting the
same refusal the golden's generation does ("LiF and acrolein, the
gradient-validation pair" below).

A third case can be asked for by name, `n2_ccpvdz_cas44` — the committed
small case, whose sidecar deliberately holds no `grad_num` and whose
**in-place write is refused** for that reason (`sidecar_grad_num` is False in
`CASES`): writing one there would move a committed golden's bytes, so `--npz`
or `--out` is the route. It is the
**deliberately symmetric** case of the gradient comparison
([`testing.md`](testing.md), "The analytic gradient against the numerical
one"): D2h, six exactly degenerate inactive pairs, and linear under every
displacement, so all six components are differenced and none is refused.

**What makes it possible is the CASCI reference** ("What 'CASSCF input' means
here" above): RHF → CASCI is deterministic at every displaced geometry, so two
displaced energies are two values of one function. Committed CASSCF orbitals
carried to a displaced geometry are not the CASSCF solution there, so this
route refuses a `casscf` golden; `--reference casscf` is the other route,
re-optimising every displaced state ("The numerical gradient of a CASSCF
reference" below).

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

#### The density-fitted numerical gradient

`nevpt2_numgrad.py --df` (#167) differences the **density-fitted** energy,
the oracle the density-fitted analytic gradient (#166) is held to. Every
displaced state is the one `generate_golden.py --df` builds (fitted RHF,
`DFCASCI` sharing its `with_df`, `dfnevpt2._ERIS` called directly), the
reference is checked against `golden/<case>_df.nevpt2gold`, and the four
`grad_num*` arrays go into the `NEVPT2F1` sidecar beside it through
`golden_format.read_gradgold` and `generate_golden.write_gradgold`, which
keep the magic and `naux` they read. No second parser is involved. A nuclear
displacement moves the fitting metric `(P|Q)` as well as `B`, and each
displaced geometry builds its own fitting object, so that term is in the
difference with no derivation needed.

The rules are the conventional route's, unchanged: the same stencil and
step, and the same refusals. A displacement that changes the detected point
group, or the core/active irrep labels, is refused and never averaged over.
`reference_data/tests/test_numgrad_df.py` asserts that refusal on the fitted
route (acrolein, `0z`: Cs → C1), before any SCF. Three more refusals are
specific to `--df`, all before any SCF:

- a golden of the other flavour (`B_aa` present or absent);
- a golden fitted in another auxiliary basis (its `B_aa` leading extent
  against `--auxbasis`'s `naux` on the molecule);
- a sidecar of the other flavour (`naux` in its header, equal to the run's).

`--df` is SC and `--reference casci` only. The in-place write is gated per
flavour (`sidecar_grad_num_df` in `CASES`), so the CAS(8,8) `_df` sidecar
refuses one.

Committed for `n2_ccpvdz_cas44_df`, at 0.0025 bohr (stencil 0.0025 / 0.00125
/ 0.000625), single-threaded, on the RTX 3080 box's CPU (no card used) with a
sibling build sharing it:

| | N2 CAS(4,4) `_df` |
|---|---|
| auxiliary basis, `naux` | cc-pVDZ-JKFIT, 140 |
| components probed / zero by symmetry | 6 / 0 (linear under every displacement, like the conventional N2) |
| displaced DF-CASCI+NEVPT2 runs | 36 |
| wall, reference + sweep | 2.9 s (3.8 s for the whole process) |
| reference vs the `_df` golden, `E_CASCI` / `E_corr` | 0.0 / 0.0 |
| `grad_num` z, atom 0 / atom 1 | +0.0448030289 / −0.0448030281 |
| max Richardson residual | 9.8e-10 |
| free check: \|g\| transverse | ≤3.3e-11 |
| free check: sum over atoms per column | ≤7.4e-10 (z) |

A second run reproduces the sidecar byte for byte: sha256 `0e74783d…`,
from the `--df --gradgold` intermediate's `286f3813…`, 855 986 B against
855 693 B. The twelve arrays before it are untouched. `devtools/golden-repro.sh`
runs this script on the sidecar its own `--df --gradgold` run has just
written, so the whole committed file is regenerated, the numerical gradient
included (docs/testing.md, "The golden regeneration gate").

### The numerical gradient of a CASSCF reference

`nevpt2_numgrad.py --reference casscf` (#124; the acrolein sweep below is
#133's) is the oracle for
`nevpt2_grad.py --reference casscf` ("The CASSCF branch" below). It
differences `golden/<case>_casscf.nevpt2gold`'s state, and it **re-optimises
every displaced state**. Committed orbitals carried to a displaced geometry
are not the CASSCF solution there, so differencing them would differentiate
the wrong functional. Each displaced state takes four steps:

1. **`generate_golden.casscf_optimise`**, the body of `--casscf` (RHF →
   symmetry-adapted `mc1step` → the fallback polish of "The CASSCF polish"),
   from a **cold** start: the RHF orbitals sorted by `--cas-irreps`.
2. **The finish** (`finish_casscf`; "Why the displaced states are finished"
   below).
3. **The golden's own `--mo-from` CASCI and SC-NEVPT2** on the finished
   orbitals, with every refusal of `_build_state` in force, the `--mo-from`
   stationarity gate (`CASSCF_GRAD_TOL`) included.
4. **The continuity guard** before anything is differenced (below).

**The reference ties to the committed files with no tolerance relaxed.**
Re-optimised by step 1, the reference geometry's orbitals are compared
**bit for bit** with the committed `.mogold`. Those orbitals must then
reproduce the golden's `E_CASCI` and `E_corr` within `REF_STATE_TOL`. Both
checks measure exactly 0 on LiF and acrolein. The finished reference, around
which the differences are taken, is reported against the committed files and
refused if its `E_CASSCF` moves by more than `CASSCF_E_TOL`:

| | LiF CAS(4,6) | acrolein CAS(6,5) |
|---|---|---|
| step-1 orbitals vs the committed `.mogold` | bit-identical | bit-identical |
| committed orbitals vs golden, `E_CASCI` / `E_corr` | 0 / 0 | 0 / 0 |
| finish's shift: `E_CASSCF` / `E_corr` | −5.8e-13 / +7.2e-9 | +1.2e-12 / +1.3e-8 |
| finished `‖g_orb‖` / `--mo-from` gate | 7.5e-11 / 7.0e-11 | 8.1e-10 / 7.9e-10 |

The `E_corr` shift is the committed golden's own convergence error, the
consequence of the next subsection's mechanism. It is 1e-8, an order inside
the 1e-7 the demos check to.

The cases, flags and stencil match the CASCI route: the same `CASES`, the
same h = 0.0025 bohr with h/2 and h/4, the same `--components auto` and
point-group refusal, the same selection check. The route is **SC only**:
`--pc` is refused here. The PC gradient of a CASSCF reference differences the
GPU's PC energies on this sweep's own kept states instead, with
`--pc-states` (#126; "The GPU's PC energies on the kept states" below).
It writes **`--npz` only**: `<case>_casscf.gradgold` stays the 24
`--gradgold` arrays and its bytes do not move. The exception is a case whose
sidecar is committed with a CASSCF `grad_num` (`sidecar_grad_num_casscf`;
only `hsoh_def2svp_cas66`, #213, "HSOH, the C1 case" below): there a
componentwise sweep writes its four `grad_num*` arrays in place, and
`nevpt2_grad.py --reference casscf` reads them by default. `nevpt2_grad.py --reference
casscf --grad-num PATH` reads that `.npz`. Since this change `--grad-num` is
no longer refused on that branch, and it refuses an `.npz` whose `reference`
is not the run's.

#### Why the displaced states are finished

Differencing step 1's states directly does not work, and the failure is
measured, not assumed. On acrolein, displaced states built by
`generate_golden`'s path alone (polish only) gave these results. Ladder:
components 0x, 1y, 4y, 7y, 24 states per step. Screen: the four seeded
directions.

| | value |
|---|---|
| optimiser's own `‖g_orb‖` (its CI vector), every state | ≤ 1.1e-8 |
| `--mo-from` gate's `‖g_orb‖` (the fresh 1e-12-Davidson CASCI's RDMs) | 1.1e-7 … 1.2e-6 |
| states the gate refused (`> CASSCF_GRAD_TOL` = 1e-6) | 2 of 24 at 0.0025 bohr (1y+0 at 1.016e-6, 1y-2 at 1.201e-6); 1 of 24 at 0.005 (the same geometry as 1y+0, the same 1.016e-6) |
| directional screen: Richardson residual / \|D − ⟨g, d⟩\| | up to 6.2e-5 / 5.0e-5 |

`mc1step` converges the orbitals against a CI vector whose residual bottoms
out near 1e-6, so the orbitals are stationary for *that* vector. With an
accurate CI vector the same orbitals measure up to 1e-6, which is what the
gate sees. `E_corr` is first order in the orbitals, so it carries
1e-8-sized noise. Re-converging six such states with an accurate FCI solver
moved their `E_corr` by up to 3.3e-8 and their `E_CASSCF` by ≤ 1.2e-12.

**The finish is two stages after step 1.** Stage 1 is `mc1step` again, with
generate_golden's polish settings and `ah_lindep` lowered to 1e-20. Below
`‖g_orb‖` = 1e-7 the augmented-Hessian subspace `(1, g)` has an overlap
eigenvalue `‖g‖²` under PySCF's `ah_lindep` (1e-14), and the step it returns
is exactly zero. So a first run that converged just under `conv_tol_grad`
stays frozen there; LiF's Li z+0.0025 converged at 9.43e-8, acrolein's 0x+2
at 7.5e-8. Its stalled-and-polished neighbours sit at ~4e-9. Stage 1 brings
every state to the same ~4e-9. Stage 2 is `mcscf.newton`, orbitals and CI
coupled, with the FCI `lindep` at 1e-16 and `conv_tol` at 1e-14,
`conv_tol_grad` 1e-9, and at most 100 macro iterations. Newton often stalls
just above its threshold, between 1e-10 and 1.5e-9. That is why the macro
ceiling is part of the definition and why the guard reads the norm reached,
not the flag. The measured need for both stages:

| acrolein | step 1 only | + newton | + both stages |
|---|---|---|---|
| direction d2: residual / \|D − ⟨g, d⟩\| | 6.2e-5 / 5.0e-5 | 8.1e-8 / 6.1e-8 | 1.1e-7 / 8.3e-8 |
| component 0x: residual / \|g − grad_num\| | — | 1.23e-6 / 1.20e-6 | 1.3e-9 / 3.6e-8 |

The newton-only 0x failure traced to one state, 0x+2, the state that
converged first time at 7.5e-8. Started from stage 1 instead, it moved by
−1.13e-9 Eh, onto the smooth curve: 0x's \|g − grad_num\| fell to 9.4e-9.
Re-finishing a finished state moves its `E_corr` by ≤ 1.1e-12, so the
finished states are stable. A tighter CASCI on acrolein's 100 determinants
changes nothing, because PySCF diagonalises a space that small exactly.

**The finish never replaces step 1.** It runs after it, and the committed
`.mogold` is what step 1 reproduces bit for bit. Nothing in
`generate_golden.py` changed its numbers: regenerating every committed
`*_casscf.mogold` with `--casscf --mo-out` into scratch gives files
byte-identical (`cmp`) to the committed ones (N2 CAS(4,4), CAS(10,10), DF
CAS(4,4); LiF; acrolein).

The `--mo-from` gate stays **unchanged and refusing** on every displaced
state. Finished, the states measure ≤ 1.4e-9 (LiF 4.2e-10) on it, against 1e-6.

#### The continuity guard's thresholds

A displaced state is refused, not differenced, when any of three signals
trips. The refusal names each state and each tripped signal:

- **`energy`**: \|E_CASSCF(s) − E_CASSCF(0) − s ⟨g_CASSCF, d⟩\|, the
  displaced energy against its first-order prediction from
  `pyscf.grad.casscf` at the reference. The cold-start run-to-run band at a
  fixed geometry is exactly 0 ("A fresh CASSCF reproduces, measured"), so
  what is left is the second-order remainder.
- **`gradient`**: `generate_golden`'s optimiser did not set `converged`
  (`CASSCF_CONV_TOL_GRAD` = 1e-7), or the finished state's own `‖g_orb‖` is
  over that same 1e-7.
- **`overlap`**: 1 − σ_min of the active-subspace overlap with the
  reference's active orbitals (carried into the displaced AO basis and
  Löwdin-orthonormalised), or 1 − \|⟨c₀\|c(s)⟩\| with c(s) rotated into the
  reference's active basis. This catches an active orbital swapped out, or a
  root flip. An active–active rotation, the gauge, costs nothing; that is a
  hand case in `reference_data/tests/test_numgrad_casscf.py`.

The thresholds are absolute and calibrated on the shipped stencil. Each sits
at least ten times above the largest value LiF's and acrolein's sweeps
measured, and below what an oversized displacement produces
(`casscf_repro_probe.py --case lif_def2svp_cas46 --displace 0z`):

| signal | threshold | max measured, LiF / acrolein sweeps | LiF Li z, `--displace` ladder |
|---|---|---|---|
| `energy` | 3e-5 | 5.6e-7 / 2.8e-6 | 5.6e-7 at 0.0025; **trips from 0.02** (3.5e-5) |
| `gradient` (finished `‖g_orb‖`) | 1e-7 | 4.5e-10 / 1.4e-9 | ≤ 9.1e-10 at every step to 0.32 |
| `overlap`: 1 − σ_min | 1e-4 | 1.4e-6 / 5.0e-6 | 1.7e-7 at 0.0025; **trips from 0.08** (1.7e-4) |
| `overlap`: 1 − \|⟨c₀\|c⟩\| | 1e-5 | 2.1e-8 / 5.9e-7 | 2.1e-8 at 0.0025; **trips from 0.08** (2.1e-5) |

So an oversized stencil is refused, and the refusal names its signal. `nevpt2_numgrad.py --case lif_def2svp_cas46 --reference casscf --step 0.04 --components 0z --no-write` exits 1 with `the continuity guard refuses 4 displaced state(s) of 6`. Each refused state is named with `tripped energy` and its numbers: `0z+0` at 1.48e-4, `0z+1` (0.02 bohr) at 3.62e-5, against 3e-5. The 0.01-bohr rung passes. The thresholds are for the shipped step, which is the point: a stencil coarse enough to leave the measured envelope is not differenced.

**Cold start defines the oracle; warm start is the detector.** After the
sweep, the first two coarse-step geometries are re-optimised from the
reference's converged orbitals and CI vector, carried into the displaced AO
basis and projected irrep by irrep, through the same path. Each is compared
with its cold state. A different `E_CASSCF` (> `CASSCF_E_TOL`), active
space or CI vector would mark the geometry ambiguous and refuse the sweep.
Measured, they are the same stationary point everywhere:

| geometry | ΔE_CASSCF | 1 − σ_min | 1 − \|⟨c_cold\|c_warm⟩\| |
|---|---|---|---|
| LiF Li x ±0.0025 | −1.1e-13, −1.3e-13 | 1.7e-15, 1.3e-15 | ≤ 1.1e-15 |
| acrolein d0 ±0.0025 (screen) | +1.2e-12, −6.3e-13 | 3.3e-16, 1.0e-15 | ≤ 1.9e-15 |
| acrolein O x ±0.0025 (gate) | −1.1e-12, −1.6e-12 | ≤ 6.7e-16 | ≤ 6.7e-16 |

**This answers the precondition [`gradient-theory.md`](gradient-theory.md)
§4.4 left unmeasured, for these two cases.** No displaced state tripped a
signal, and cold and warm agree to round-off. That includes LiF's off-axis
components, where the saddle's symmetry-breaking modes sit ("LiF and
acrolein on CASSCF orbitals" below). Every LiF state stayed in C2v, which the
symmetry-adapted optimiser cannot leave.

#### The sweeps

All runs are CPU only, single-threaded per process, `--workers 12` on the
gfx1200 box's Ryzen 9 5900X (12 physical cores, 24 SMT threads). No card is
involved. Each state is an independent cold start, collected by label, so
the result does not depend on `--workers`. On LiF, `--workers 1` and
`--workers 12` give bit-identical arrays, every one of them (`cmp` on the
`.npz` contents) at 120.5 s vs 20.4 s.

| | LiF CAS(4,6) | acrolein screen | acrolein CAS(6,5) componentwise |
|---|---|---|---|
| what | 6 components (0 zero by symmetry) | 4 seeded directions (`SCREEN_SEED` 124) | 16 probed, 8 zero by symmetry |
| displaced states (+ 2 warm) | 36 | 24 | 96 |
| wall (whole run), declared budget | 20.4 s, 600 s | 424.6 s, 1800 s | 950.8 s, 3600 s |
| max Richardson residual | 1.6e-8 | 1.1e-7 | 7.5e-8 |
| against `nevpt2_grad.py --reference casscf` | max \|g − grad_num\| 2.9e-8 | max \|D − ⟨g, d⟩\| 8.3e-8: **PASS** (1e-6) | max \|g − grad_num\| 8.3e-8 (RMS 3.1e-8) |
| free check: CASSCF-only gradient vs `pyscf.grad.casscf` | 1.7e-10 | 1.9e-9 | 3.0e-9 |
| translational sums | ≤ 1.1e-8 | — | 3.2e-8 (x), 9.7e-8 (y) |

**The residual band is not the CASCI twin's, and this says why.** The issue
asked for a residual "in the band its CASCI twin reports". That is 1.1e-10
on LiF and 4.5e-9 on acrolein ("The numerical gradient" above). The CASSCF
route measures 1.6e-8 and 7.5e-8. A CASCI state is one deterministic
diagonalisation per geometry. A CASSCF state is the end of an iterative
optimisation whose stopping point is itself noise. LiF's congruent pairs
(mirror images ±x/±y, and Li +z against F −z) are the same internal geometry
and should agree exactly. Finished, they agree to 2.4e-11 Eh in
`E_CASCI + E_corr`, which is 1.9e-8 Eh/bohr in the finest central
difference: the LiF floor, measured directly. With step 1's states alone
the same pairs differed by 2.9e-10 Eh (2.3e-7 Eh/bohr; residual 2.4e-7),
with the optimiser at its ~4e-9 plateau. The gate stays the analytic
gradient against this one at 1e-6, and the bars above are far inside it.
Park's own SC analytic-vs-numerical RMS on these two systems is 2.4e-6 (LiF)
and 3.4e-6 (acrolein) at a 0.001 bohr step ([`references.md`](references.md),
"Analytical gradients"; his Table 3). This is an external anchor, not the
gate.

**One sweep, three consumers.** `--states-dir DIR` keeps every displaced
state's finished orbitals as `<label>.mogold`, beside a `states.json`
manifest. Labels are `<atom><axis><sign><rung>` and `d<n><sign><rung>`; the
manifest records each state's `--atom` spec, displacement, energies, signals
and the finish settings. The screen and the componentwise gate share one
directory, merged by label. `pc_on_states.py`'s per-state `--no-pc`
goldens for the GPU ("The GPU's PC energies on the kept states" below) and
#126's block2 spot-checks consume those sidecars
(`generate_golden.py --atom SPEC … --mo-from DIR/<label>.mogold`) rather
than optimising their own. The one reader of the manifest is
`golden_format.read_states_manifest`. Nothing in it is committed. A state the guard
refuses has no `.mogold`; its manifest entry carries `tripped` (the signal
names) or `refused` (the refusal text) instead, and the run exits 1 after
naming every one. In the sweeps above none was refused.

Re-run all of the above with:

```bash
uv run python reference_data/nevpt2_grad.py --case acrolein_def2svp_cas65 \
    --reference casscf --out /tmp/acrolein.grad.npz     # the analytic side
uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65 \
    --reference casscf --directional 4 --analytic /tmp/acrolein.grad.npz \
    --workers 12 --states-dir /tmp/acrolein_states --npz /tmp/screen.npz
uv run python reference_data/nevpt2_numgrad.py --case acrolein_def2svp_cas65 \
    --reference casscf --workers 12 --states-dir /tmp/acrolein_states \
    --analytic /tmp/acrolein.grad.npz --npz /tmp/acrolein_casscf.numgrad.npz
uv run python reference_data/nevpt2_numgrad.py --case lif_def2svp_cas46 \
    --reference casscf --workers 12 --npz /tmp/lif_casscf.numgrad.npz
uv run python reference_data/casscf_repro_probe.py --case lif_def2svp_cas46 \
    --displace 0z --runs 3 --workers 12 \
    --steps 0.000625,0.00125,0.0025,0.005,0.01,0.02,0.04,0.08,0.16,0.32
```

#### The GPU's PC energies on the kept states

`reference_data/pc_on_states.py` (#138) turns a `--states-dir` into the
GPU's PC-NEVPT2 energy at every kept state, written to one keyed `.npz`.
`nevpt2_numgrad.py --pc-states` differences that file (#126, below). This
script differences nothing and gives no verdict. Per state it runs three steps:

1. **A transient golden** on CPU, in `--workers` processes. This is
   `generate_golden.generate(..., pc=False, mo_from=<label>.mogold)`, the
   in-process `--no-pc --mo-from`, with every `_build_state` refusal in force.
   The `.mogold` already holds the two-stage finish, so nothing is
   re-optimised, and block2 never runs. `nevpt2_demo --pc` refuses a golden
   without block2's `pc_class_energies` / `e_pc_total`, so the transient
   file carries them as **NaN**. A NaN matches no energy, so the demo's own
   PC comparison prints `FAIL` by construction. The script ignores that exit
   status and reads the parsed lines. The files go to `--work-dir`, never
   to `golden/` (refused), and are deleted afterwards.
2. **The wiring check**: the transient golden's `E_CASCI` and `E_corr` must
   equal the manifest's for the same state within `REF_STATE_TOL`.
3. **The GPU**, serially, under `--gpu-lock`. `nevpt2_demo --tiles 3` must
   `PASS` the device's SC against that golden's PySCF SC. `nevpt2_demo --pc
   --tiles 3` gives the energy from #134's `E_corr full precision
   (PC-NEVPT2)` line, plus each class's `d`, `dropped`, `max dropped`,
   `min kept` and `gap`.

**The PC conditioning guard per state** is
`generate_golden._check_pc_conditioning`, at the same `PC_MIN_GAP` (1e3). It
is handed the device's per-class gaps, because block2 does not run here. Those
gaps come from the eigenvalues of each class's metric `S`, cut at
`PC_TAU · s_max`, not from block2's per-tuple lstsq singular values. A class
the device refuses fails the guard with the device's reason. The gaps are
recorded as printed, to two significant figures. A refused state stays in
the `.npz` with a NaN `e_pc` and its reason. Its `dropped` and `gap` rows
show a metric mode moving between geometries. The reference state is held
to the committed golden's `e_pc_total` within the demos' 1e-7.

Measured on acrolein CAS(6,5), with #124's kept states (the screen's 24 and
the gate's 96, plus `ref`). Run on the gfx1200 box with `--workers 8` and
the HIP build's `nevpt2_demo`:

| | value |
|---|---|
| states processed / refused by the guard / not built | 121 / 0 / 0 |
| wiring: max \|transient − sweep\| `E_CASCI` / `E_corr` | 0 / 1.1e-16 |
| max \|GPU SC − PySCF SC\| on the transient goldens | 3.2e-15 |
| reference geometry: GPU `E_PC` − committed `e_pc_total` | +1.23e-8 |
| smallest guard gap (every state: `Sij`, nothing dropped) | 2.9e3 |
| `dropped` per class, every state | Sr 50, Si 85, Sir 2, the rest 0 |
| wall: transient goldens / GPU (242 demo runs) / total | 14.6 s / 123.5 s / 138.1 s |

The reference offset is the finish's own +1.3e-8 shift in `E_corr` ("The
numerical gradient of a CASSCF reference" above), carried into PC. No
class's `dropped` count changes between states, so no metric mode moves in
or out across the stencil. The smallest gap, `Sij` at 2.9e3, sits under
three times `PC_MIN_GAP` at every geometry. Nothing is dropped there, so
that gap is its smallest kept eigenvalue over `PC_TAU`.

```bash
uv run python reference_data/pc_on_states.py \
    --states-dir /tmp/acrolein_states --demo build-hip/src/nevpt2_demo \
    --workers 8 --work-dir /tmp/pc_states --gpu-lock /tmp/nevpt2-gpu-hip.lock \
    --npz /tmp/acrolein_pc_states.npz
```

#### Differencing them: `--pc-states`

`nevpt2_numgrad.py --reference casscf --pc-states NPZ --npz PATH` (#126) runs
no sweep. It reads `pc_on_states.py`'s file and differences
`E_CASCI + E_PC` per coordinate with this route's own stencil and Richardson
pair (`stencil`, `_difference`), into `grad_num_pc` /
`grad_num_pc_residual` / `grad_num_probed` in the `.npz` that
`nevpt2_grad.py --reference casscf --pc --grad-num PATH` reads. A coordinate
needs all six of its states built, guarded and with a finite `e_pc`, or the
run is refused and names the states: a refused state is a hole in the
stencil, never a number to difference around. The screen's directions in the
same file are differenced too, and with `--analytic` (a `--pc` gradient)
they are gated at 1e-6 like the SC screen. Beside PC, the same states'
`E_CASCI + E_corr(SC)` is differenced from the same file (PySCF's on the
transient goldens, and the device's), so the SC gradient of the sweep that
made the states comes back out of this file alone: the wiring check of the
whole hand-off. The verdict, the wiring result and the block2 spot-checks
are in [`testing.md`](testing.md), "PC on a CASSCF reference: acrolein".

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

**SC or PC is one flag, because the Lagrangian is a statement about the
reference and not about the contraction.** `--pc` swaps `E_corr` for block2's
partially-contracted energy (`_block2_icmr`) and nothing else: (O1), (O2),
(C1), the constraint Jacobian, the frozen path and the solve order are
identical, and the two source terms are difference quotients of that one
scalar either way. What `--pc` does need is the generalized Fock's diagonal
rebuilt at every point (block2 runs `canonicalized = True` and uses the
`mo_energy` it is handed), the dense 4-RDM rebuilt whenever the CI vector
moves, a PC answer to exist on the state at all, and its **own** CI step —
1e-7, chosen for a reason that is not truncation
([`testing.md`](testing.md), "The analytic gradient against the numerical
one"). Its orbital and nuclear steps are its own constants too since #77,
held where every PC figure was measured while SC's moved
([`testing.md`](testing.md), "The step constants, measured"). The cost is one block2 PC solve per non-gauge rotation, so it is the
`kappa` sweep that pays.

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

**What the two cases come out at**, source terms from PySCF, at the SC
steps (`CI_STEP = KAPPA_STEP = 2^-17`, `NUC_STEP = 2^-13` bohr since #77 —
[`testing.md`](testing.md), "The step constants, measured", which has why
and what each moved):

| | LiF CAS(4,6) | acrolein CAS(6,5) |
|---|---|---|
| `kappa` directions = constraints | 246 | 2846 |
| CI Z-vector residual `\|P(H−E)Pz + Py\|` | 1.404e-12 | 2.006e-12 |
| orbital residual `\|M λ + B\|`, direct / GMRES | 7.9e-16 / 1.5e-12 | 6.4e-15 / 8.7e-12 |
| the orbital system's condition number | 8.2e4 | 1.2e4 |
| CI/(O2) fixed point | 6 iterations, last step 1.1e-16 | 6 iterations, last step 3.4e-15 |
| `z·c` after projection | 8.3e-20 | 4.7e-18 |
| gauge identity (§1.3, no multiplier) | 2.0e-6 | 2.7e-4 |
| Richardson bar, skeleton / multipliers | 5.4e-10 / 4.7e-12 | 1.1e-9 / 2.8e-10 |
| `\|g\|` on the zero-by-symmetry axes | 6.9e-14 | 2.1e-12 |
| sum over atoms per column | ≤1.9e-11 | ≤4.7e-9 |
| wall, single-threaded | 56 s | 1029 s |

The two walls were measured at the steps before #77; the step does not
change the number of evaluations (688 and 5882), so they were not re-run on
an idle machine. At those steps the rows that moved read: orbital residual
direct 1.0e-15 / 5.9e-15, fixed point's last step 1.9e-15 / 3.7e-15, `z·c`
4.0e-19 / −5.7e-19, Richardson bars 2.2e-7 / 1.8e-9 and 5.6e-7 / 1.2e-8,
zero-by-symmetry 1.7e-15 / 3.0e-11, column sums ≤5.6e-11 / ≤1.0e-8.

The cost is the `kappa` sweep: one central difference of the SC-NEVPT2 energy
per non-gauge rotation, `O(n_mo²)` of them — 937 s of acrolein's 1029 s.
That is what makes the source terms self-contained, and it is what the GPU's
own dump (`nevpt2_sc_pseudodensity --dump-source-terms`, read with `--source
dump`) replaces. Since #87 (the orbital energies' response in `y`) and #89
(the digests' response in `Y`) a gradient built from the dump passes the same
gate against `grad_num` — [`testing.md`](testing.md), "The GPU as the
source-term provider", which also keeps the record of how it failed before.

**The comparison against `grad_num` is the gate, and it lives in
[`testing.md`](testing.md), "The analytic gradient against the numerical
one"** — the verdict per case, SC and PC, the walls of the PC runs, the third
(deliberately symmetric) case, and the one statement of why its tolerance
cannot be this repo's 1e-7: `grad_num` is itself a finite difference (its
error bar is in "The numerical gradient" above) and the analytic side's own
bar is the Richardson row in this table, so the tolerance is set by those two
rather than by machine precision. Acrolein's `grad_num` needs its gitignored
sidecar regenerated first, or an `--npz` from `nevpt2_numgrad.py`.

**How it is validated** independently of that comparison: `--casci-only` drops
`E_corr` from the source terms and from the scalar, leaving a plain CASCI
whose analytic gradient PySCF already has. That one comparison exercises the
frozen path, the full constraint Jacobian, the selection-boundary multipliers,
the CPHF block and the differentiation at once. Against the **numerical
CASCI** gradient (`nevpt2_numgrad.py`'s own `grad_casci`, which it prints and
also writes to an `--npz`, where `--casci-only --grad-num` gates against it)
it lands at **1.8e-8** on LiF, **1.7e-7** on acrolein and **1.3e-8** on N2
CAS(4,4), at the SC steps since #77. At the steps before it, the same runs
against the same `grad_casci` give 2.0e-8, 1.5e-7 and 1.5e-8.

**This section first recorded acrolein at 5.3e-8 (#60), and that figure was
never what the code produced** (#80). Neither side has moved since:

- *The numerical side reproduces itself bit for bit.* Three acrolein
  `nevpt2_numgrad.py --no-write --npz` runs (two concurrent, one after) wrote
  byte-identical files, every array equal; #60's own `nevpt2_numgrad.py`
  (checked out at that merge) prints every per-component line identically,
  `grad_casci` included; and a fresh LiF run equals the committed
  `lif_def2svp_cas46.gradgold`'s four `grad_num*` arrays exactly. `uv.lock`
  and `pyproject.toml` are byte-identical between #60 and now (one commit
  has ever touched `uv.lock`, before #60), and `generate_golden` forced
  single-threading then as now.
- *No displaced point is on another root.* At all 96 displaced acrolein
  points every CASCI converged, its energy is the lowest of four roots of the
  same active Hamiltonian diagonalised without symmetry to 1.7e-13, the gap
  to the second root is never below 0.151 Eh, and `grad_casci`'s own
  Richardson residual (computed, but not kept, by `nevpt2_numgrad.py`) is at
  most 1.6e-9.
- *The analytic side at #60 gives today's answer.* #60's `nevpt2_grad.py
  --casci-only` returns a gradient **identical** (max difference 0.0) to
  today's at the old steps (`--step 0.0025 --kappa-step 1e-4 --ci-step
  1e-4`), and it lands **1.511e-7** from `grad_casci`, at atom 2 x. The same
  check on LiF at #60 gives 2.044e-8, which is the 2.0e-8 the section
  recorded alongside the 5.3e-8.

So 5.3e-8 does not come from the command the section implies at the code
of its time; the PR that recorded it lists that command but not its output.
It is a recording error, not drift, and no committed golden or sidecar is
touched by it. (One of the sixteen component differences, atom 7 y, is
5.34e-8, but nothing on record says that is how the figure arose.) What
remains unexplained is the size of the 1.5e-7–1.7e-7 gap itself: at the
current steps it is about 25 times the analytic side's skeleton Richardson
bar (6.6e-9) and 100 times `grad_casci`'s (1.6e-9), and neither bar accounts
for it; it is well inside the 1e-6 gate.

**`pyscf.grad.casci` is not an oracle for either committed case** of the
pair, and the run says so rather than reporting a disagreement as a failure.
It divides its redundant core–act_O and act_V–virtual blocks by
orbital-energy denominators assuming act_O is the *first*
`n_elec/2 − n_core` active columns;
`--cas-irreps` orders the active block by irrep, so on LiF PySCF's own answer
lands **4.7e-4** from the numerical CASCI gradient while this pipeline lands
2.0e-8. `--energy-ordered` drops the irrep selection and restores the
configuration PySCF assumes; the two analytic gradients then agree to
**1.7e-5** on acrolein. N2 CAS(4,4) needs no such flag — its active space is
energy-ordered to begin with, so it is the one committed case PySCF's own
analytic gradient oracles directly, and there the two agree to **9.1e-6**
while this pipeline sits 1.3e-8 from the numerical CASCI gradient: that gap is
the size of PySCF's redundant-block denominators, not of an error here.
(Calling it at all needs
`mo_coeff=mc.mo_coeff, ci=mc.ci`: its default is `mc._scf.mo_coeff`, the
*unsorted* RHF orbitals, which do not describe one state together with
`mc.ci`.)

**Four things an implementation has to get right, none of which shows up in a
residual.** All four were found by the checks above, and all four leave the
Z-vector residuals at round-off while moving the gradient by 1e-2 to 1e-5.
Each now has a gated check of its own that does fail on it — reintroduced one
at a time, with the numbers, in [`testing.md`](testing.md),
"Residual-independent checks of the gradient path":

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
   of the active density. The device's `y_I` held them fixed until #87 put
   their response in ([`gradients.md`](gradients.md), §5.5).

#### The CASSCF branch (`--reference casscf`, #122)

`--reference casscf --mo-from FILE` differentiates SC-NEVPT2 on a CASCI run
on converged CASSCF orbitals ([`gradient-theory.md`](gradient-theory.md),
§1.4 and §2.7), with PySCF's own source terms or, with `--source dump` /
`both` (#125), the GPU's dumped ones; `--mo-from` defaults to
`golden/<case>_casscf.mogold` where one is committed. Of the four hooks,
`get_Aop_Adiag` is the one that changes: the operator is
`pyscf.mcscf.newton_casscf.gen_g_hop`'s orbital+CI Hessian `h_op`, restricted
to the non-redundant rotations plus the in-sector CI directions orthogonal to
`c`, **built column by column and factorised** (LU), its symmetric part
diagonalised. `z^c` is solved first from (O2)'s own rows, as on the CASCI
branch; there is no (O1), no selection multiplier and no §2.6 rotation.
`--cp-solve cg` runs `lagrange.Gradients.solve_lagrange`'s conjugate
gradient through the same operator instead (on N2 CAS(4,4) it converges to
`|J L + b|` = 2.9e-12 and moves the gradient by 4.3e-13 from the dense
answer). The constraints in `Phi` are `gen_g_hop`'s `g_all` at each
displaced geometry. Its numerical gradient is `nevpt2_numgrad.py
--reference casscf --npz PATH`, read with `--grad-num PATH` ("The numerical
gradient of a CASSCF reference" below; the SC verdict on it, LiF and N2,
PySCF- and GPU-sourced, is [`testing.md`](testing.md), "SC on a CASSCF
reference: LiF, and the N2 degeneracy probe");
without one, the gate is the residual-independent checks, with two of the
branch's own: **the Hessian-vector ladder** — `h_op`
against central differences of `g_all` along three seeded random directions,
no NEVPT2 content — and, with `--casci-only`, **the smoke test** —
multipliers exactly zero, gradient against `pyscf.grad.casscf`.

**`--pc` on this branch** (#126) is the CASCI branch's `--pc` fed into the
same coupled solve: `Y`, `y` and `Phi`'s `E_corr` become block2's PC energy
at the PC steps, with the PC guard at every displaced CI vector, while
`h_op`, (S1), (C1), (O2) and the basis do not change. Its numerical gradient
is the GPU's PC energy on #124's re-optimised states ("Differencing them:
`--pc-states`" above), and without one the residual-independent checks are
the gate, as for SC. Its own wall-clock budget (`CASSCF_WALL_BUDGET_PC`), the
verdict and the walls are in [`testing.md`](testing.md), "PC on a CASSCF
reference: acrolein". `--workers N` runs the `kappa` sweep, the largest
share of a PC run's cost, in `N` forked processes, with the result
bit-identical to the serial one (measured there). With `--source dump` /
`both` (#163) the GPU's PC dump (`nevpt2_sc_pseudodensity --pc
--dump-source-terms` on the `*_casscf` golden) replaces block2's `Y` and
`y`, read unrotated as the SC one is; `both` solves block2's on the same
sweep beside it ([`testing.md`](testing.md), "PC on a CASSCF reference:
acrolein", "GPU-sourced").

**LiF and acrolein ran on scratch orbitals, not committed ones.** Their
CASSCF references are #123's to commit; these were written by
`generate_golden.py --casscf --mo-out` into a scratch directory with one
refusal lifted — the optimiser's own `converged` flag. Under the generator's
settings both optimisations stall just above `conv_tol_grad` = 1e-7 for all
200 macro iterations (LiF at `|g_orb|` 1.2e-7, acrolein at 2.2e-7, the
optimiser's norm) and never set it, so `--casscf` as committed refuses both;
every other refusal ran, and `--mo-from`'s gate passed both (5.8e-7 and
2.2e-7 against its 1e-6). N2 CAS(4,4) is the committed
`golden/n2_ccpvdz_cas44_casscf.mogold`, and its rebuilt state matches that
golden's `E_CASCI` exactly and `E_corr` to 2.8e-17.

**Since #123 both have committed references** (`golden/<case>_casscf.mogold`,
found by name; "LiF and acrolein on CASSCF orbitals" below), finished by
"The CASSCF polish" from the very point those scratch runs stalled at. The
table below stays the scratch measurement; re-run on the committed orbitals
(RTX 3080 box's CPU, single-threaded), SC, the rebuilt state matches each
committed golden's `E_CASCI` and `E_corr` exactly (0.0), 14 of 14
residual-independent checks hold on both, and the spectra keep their shape:
LiF −7.3215e-3 (×2), +2.8218e-6 (×2), +2.8196e-3, cond 4.08e7, 2 negative;
acrolein +3.6171e-3, +3.9915e-3, +4.5201e-3, cond 2.66e4, 0 negative. Wall
61.9 s and 1055.7 s, inside the 300 s / 3600 s budgets.

Measured on the gfx1200 box's CPU, single-threaded per run with up to six
runs side by side:

| | N2 CAS(4,4) | LiF CAS(4,6) | acrolein CAS(6,5) |
|---|---|---|---|
| orbitals | committed | scratch | scratch |
| coupled dimension: orbital + CI (in-sector dets − 1) | 191 + 11 = 202 | 154 + 116 = 270 | 1063 + 51 = 1114 |
| dense build (one `h_op` per column) | 0.3 s | 0.3 s | 30.4 s |
| `max\|J − Jᵀ\| / max\|J\|` | 2.2e-15 | 1.2e-16 | 6.0e-15 |
| smallest eigenvalues of the symmetric part | 5.57e-2 (×2), 6.71e-2 (×2) | **−7.32e-3 (×2)**, 2.82e-6 (×2), 2.82e-3 | 3.62e-3, 3.99e-3, 4.52e-3 |
| condition number | 1.3e3 | 4.1e7 | 2.7e4 |
| `\|Jᵀ L + b\|` direct / through `h_op`, against `\|b\|` | 3.5e-17 / 1.3e-15, 0.165 | 1.2e-16 / 1.5e-15, 0.168 | 3.3e-16 / 2.6e-13, 0.284 |
| max `\|Z\|`, `\|z\|` | 3.0e-2, 3.8e-2 | 0.63, 1.8e-2 | 1.26, 0.117 |
| Hessian-vector ladders, max `\|Richardson − J v\|` | 4.8e-9 – 1.2e-8 | 2.2e-8 – 3.9e-8 | 9.5e-9 – 1.2e-8 |
| gauge identity, whole Lagrangian | 3.7e-9 | 2.7e-7 | 9.3e-9 |
| Richardson bar, skeleton / multipliers | 4.9e-9 / 2.2e-11 | 5.6e-10 / 5.8e-12 | 2.1e-9 / 3.0e-10 |
| residual-independent checks | 14 of 14 | 14 of 14 | 14 of 14 |
| smoke test (`--casci-only`): multipliers / `\|g − pyscf.grad.casscf\|` | exactly 0 / 1.96e-8 | exactly 0 / 5.04e-8 | exactly 0 / 8.77e-9 |
| wall, SC / `--casci-only` | 14.4 s / 6.8 s | 58 s / 5.7 s | 1069 s / 783 s |
| declared budget (`CASSCF_WALL_BUDGET`) | 120 s | 300 s | 3600 s |

**LiF's stationary point is a saddle, and the spectrum says where.** Its
coupled Hessian has a doubly degenerate negative pair and a doubly
degenerate near-null pair (2.8e-6), all four pure orbital rotations along
which the right-hand side vanishes to round-off (`|b·v|/|b|` ≤ 1.4e-15): the
symmetry-adapted optimiser cannot leave the point group, and these are
rotations it cannot take. They do not enter the multipliers, and every
check holds; #125 probed them and they stay out of the solve on the
committed orbitals too ([`testing.md`](testing.md), "SC on a CASSCF
reference: LiF, and the N2 degeneracy probe"). Acrolein's
and N2's spectra are positive, with the right-hand side on acrolein's
nearest-singular modes at up to 2.5e-2 of `|b|`.

The budget is a bound, not a gate: a run over it says so and does not fail.
The acrolein wall is the `kappa` sweep, as on the CASCI branch (same
evaluation count to within the gauge set); the dense build is 3% of it.

**`--reference casci` is unchanged by this branch, measured.** The script
before and after, on LiF and N2 (SC and `--casci-only`) and acrolein (SC):
every array `--out` writes is bit-identical, the gradient included, the two
new string fields (`reference`, `mo_from`) aside.

### The SC source-term oracle

`golden/n2_ccpvdz_cas44.srcgold` and `golden/lif_def2svp_cas46.srcgold` hold
**PySCF's exact SC-NEVPT2 source terms** for the two goldens' states — the
independent side of the `nevpt2_sc_pseudodensity_oracle_*` entries
([`testing.md`](testing.md), "The SC source terms against PySCF's, on a
card"). `reference_data/sc_source_oracle.py` writes them, and computes
nothing new: `nevpt2_grad.py`'s own state (`build_state`), its central
differences at its converged SC steps (`KAPPA_STEP` = `CI_STEP` = 2^-17, #77),
its `kappa_sweep` and its `pyscf_ci_source` — exactly what `--source both`
compares a dump with. The script's docstring is the array-by-array
description; in short:

| array | shape | what |
|---|---|---|
| `a_corr` | `[n_mo, n_mo]` | `A_pq = dE_corr/dkappa_pq − (p ↔ q)` on every non-gauge pair, at fixed CI vector, in `nevpt2_grad`'s §2.6 basis (act_O = the first `n_act_occ` active orbitals); zero on the act_O–act_O / act_V–act_V gauges |
| `act_rot` | `[n_act, n_act]` | `W`, the active-block rotation that basis is; the reader carries the GPU's `Y` across as `W^T Y W` |
| `py` | `[n_det]` | `P y` on the state's irrep, carried **back** into the golden's own determinant basis by the orthogonal `T(W^T)`, the golden's CI sign applied |
| `ci_mask` | `[n_det]` | 1.0 on the determinants `py` is defined on |
| `ci` | `[n_det]` | the golden's CI vector, a fingerprint `loadSrcGold`'s reader holds bit for bit |

The scalar header (`NEVPT2S1`, `golden_format.SRCGOLD_HEADERS`) is the
partition, `n_act_occ`, `n_det` and the two steps. `A` stays in the §2.6
basis because the gauge entries are not differenced, so it cannot be carried
back; `P y` can, exactly, because the irrep sector maps onto a sector of the
golden's basis — the script checks that (and that `T(W^T) T(W) = 1`) and
refuses rather than write a mixed one. N2 CAS(4,4) keeps 12 of its 36
determinants, LiF all 225 (its active orbitals are not symmetry-pure in the
§2.6 basis, "The source terms, componentwise" in [`testing.md`](testing.md)).

**Reproducible byte for byte**, like every generator here — single-threaded
through `generate_golden`'s import, and nothing iterative but the 1e-12
Davidson. Measured: both files regenerated with `--out` to a scratch
directory compare identical with `cmp` (and once more at the committed steps
inside the tolerance ladder of [`testing.md`](testing.md)). 7440 and 10096
bytes; 10 s and 52 s wall, single-threaded on the RTX 3080 box's CPU.

```bash
uv run python reference_data/sc_source_oracle.py --case n2_ccpvdz_cas44 --out /tmp/n2.srcgold
cmp /tmp/n2.srcgold golden/n2_ccpvdz_cas44.srcgold    # silent: identical
```

The full flavour covers only these two cases: N2 CAS(8,8)'s `P y` alone is
6481 s of quotients, and acrolein's full sweep takes hours. Acrolein has a
directional oracle instead ("The directional flavour" below). `--kappa-step` /
`--ci-step` exist only so the tolerance ladder can be rerun; the committed
files are at the defaults.

#### The PC flavour

`golden/n2_ccpvdz_cas44_pc.srcgold` (#95) is the same file for **block2's
PC-NEVPT2** energy: `sc_source_oracle.py --pc`, the same generator and
writer, the same header and five arrays, under the magic `NEVPT2P1`
(`golden_format.SRCGOLD_MAGIC_PC`; `SourceScheme::Pc` in `loadSrcGold`). One
layout with the magic as the flavour — not a second format — and the name
`<case>_pc.srcgold` keeps it beside the same `<case>.nevpt2gold` /
`.gradgold` as the SC oracle (`golden_format.srcgold_case`, `srcGoldCase`).
Everything comes out of `nevpt2_grad.py --pc`'s machinery: `build_state(...,
pc=True)` (its PC energy checked against the golden's `e_pc_total`; measured
1.0e-15), `Functional(pc=True)`, `kappa_sweep` and `pyscf_ci_source`. Every
displaced CI vector goes through the PC guard, and the script refuses to write
if one was unguarded or changed a PC block's rank. The steps are `KAPPA_STEP`
(`2^-17`, not `KAPPA_STEP_PC`) and `CI_STEP_PC` (1e-7); why, and the ladders,
are in [`testing.md`](testing.md), "The PC source terms' oracle". N2 CAS(4,4)
only: LiF's golden has no PC fields, and acrolein is the manual gate's
(`nevpt2_grad.py --pc --source both`; [`pc-nevpt2.md`](pc-nevpt2.md), "The
source terms by reverse mode").

**Byte-reproducible**, measured: the committed file, regenerated with the
command above, and the kappa ladder's `2^-17` rung, written separately with
`--out`, compare identical with `cmp`. 7440 bytes; about 150 s single-threaded
on the RX 9060 XT box's CPU (the 376-pair kappa sweep, two block2 PC solves a
pair, is nearly all of it).

```bash
uv run python reference_data/sc_source_oracle.py --case n2_ccpvdz_cas44 --pc --out /tmp/n2_pc.srcgold
cmp /tmp/n2_pc.srcgold golden/n2_ccpvdz_cas44_pc.srcgold    # silent: identical
```

#### The directional flavour

The full flavour costs two energies per rotation pair. Acrolein CAS(6,5) has
2,846 of them, and its full PC oracle took 21,136 s in #74's scratch run, so
acrolein, the one low-symmetry case, had no committed oracle. #105 adds a
**directional flavour** of the same format: `sc_source_oracle.py
--directional [--pc]`, the same generator, writer and scalar header, under
the magics `NEVPT2SD` / `NEVPT2PD` (`golden_format.SRCGOLD_KINDS`;
`SourceFlavour::Directional` in `loadSrcGold`), named
`<case>[_pc]_dir.srcgold` (`golden_format.srcgold_name` / `srcgold_case`,
`srcGoldCase`). `A` and `P y` enter the gradient linearly, so a wrong term in
a block moves a seeded random direction's projection with probability 1, and
the comparison stays block by block. Per non-empty block, two seeded unit
`kappa` directions over that block's pairs; three seeded CI directions
(`nevpt2_grad.ci_direction`: orthogonal to the state, in its irrep, a spin
eigenfunction). Each value is the Richardson extrapolant of the first two
rungs of a four-rung ladder, and the ladder is stored beside it:

| array | shape | what |
|---|---|---|
| `kappa_dirs` | `[n_k, n_mo, n_mo]` | the directions, antisymmetric, unit norm over `p < q`, each inside one block, in the §2.6 basis `a_corr` is in |
| `kappa_block` | `[n_k]` | which block (0..7, the comparator's row order) |
| `kappa_fd` | `[n_k, rungs]` | central differences of `E_corr(C exp(h kappa))` at fixed CI vector, `h = kappa_step · 2^-i` |
| `kappa_val` | `[n_k]` | `<kappa, A> = Σ_{p<q} kappa_pq A_pq`, `(4 fd(h/2) − fd(h)) / 3` |
| `ci_dirs` | `[n_c, n_det]` | the CI directions, in the golden's determinant basis (carried by `T(W^T)`, the CI sign applied) |
| `ci_fd`, `ci_val` | `[n_c, rungs]`, `[n_c]` | the same for `E_corr` along the unit sphere: `<u, P y>` |
| `act_rot`, `ci` | | as in the full flavour |

The header's `kappa_step` / `ci_step` are each ladder's first rung. The
loader refuses a direction that is not exactly antisymmetric, a block index
outside 0..7, a ladder of fewer than two rungs and a CI direction not
orthogonal to `ci`. The comparator refuses a direction that is not inside
its labelled block (`checkDirectionBlocks`), and fails a block that fewer
than two directions probe. The steps and the tolerance were measured from
long ladders, in [`testing.md`](testing.md), "The directional source-term
oracle".

Five files are committed: acrolein SC and PC, plus N2 CAS(4,4) SC and PC
and LiF SC beside their full oracles as the cross-check. Single-threaded wall
time per file, run one at a time on the two-card box's CPU (24 cores; #105):

| file | bytes | wall |
|---|---:|---:|
| `acrolein_def2svp_cas65_pc_dir.srcgold` | 743929 | 582 s |
| `acrolein_def2svp_cas65_dir.srcgold` | 743929 | 42 s |
| `n2_ccpvdz_cas44_pc_dir.srcgold` | 102833 | 51 s |
| `lif_def2svp_cas46_dir.srcgold` | 76401 | 11 s |
| `n2_ccpvdz_cas44_dir.srcgold` | 102833 | 4 s |

Every file is inside the 30-minute budget. **Byte-reproducible**, measured:
each file was generated twice, once with all five running in parallel and
once one at a time, and both copies compare identical with `cmp`. After the
container was rebuilt, LiF and both acrolein files were generated a third
time and compared identical again (12 s, 44 s and 616 s, with other jobs on
the CPU). The N2 pair is in `devtools/golden-repro.sh`, which reproduced it
too ([`testing.md`](testing.md), "The golden regeneration gate").

```bash
uv run python reference_data/sc_source_oracle.py --case acrolein_def2svp_cas65 --directional --pc --out /tmp/a.srcgold
cmp /tmp/a.srcgold golden/acrolein_def2svp_cas65_pc_dir.srcgold    # silent: identical
```

The acrolein oracles are read with `golden/acrolein_def2svp_cas65.gradgold`.
That sidecar is gitignored ("The gradient sidecar"), so their ctest entries
exist only where it has been regenerated. The generator itself does not need
the sidecar.

#### On CASSCF orbitals

`golden/n2_ccpvdz_cas44_casscf[_pc][_dir].srcgold` (#117) are all four kinds
above for the state of `golden/n2_ccpvdz_cas44_casscf.nevpt2gold`: a CASCI on
the converged CASSCF orbitals of the committed
`golden/n2_ccpvdz_cas44_casscf.mogold`. `sc_source_oracle.py --case
n2_ccpvdz_cas44_casscf --mo-from FILE` builds it through
`nevpt2_grad.build_state(reference="casscf")`, which is
`generate_golden._build_state`'s own `--mo-from` path, so the sidecar passes
the same refusals the golden did: the fingerprint ("The MO sidecar: committed
CASSCF orbitals"; every mismatch listed in one refusal), symmetry purity and
the stationarity gate ("The `--casscf` / `--mo-from` gate thresholds"). The
golden must say `reference_kind = casscf`. A `*_casscf` case without
`--mo-from` is refused, and so is `--mo-from` on any other case. **No new
format**: the magics, header and arrays are the ones above. Only the
orbitals and the partition differ. With no RHF to split it
([`gradient-theory.md`](gradient-theory.md), section 1.4), the active space
is one range (`n_act_occ = n_act`), act_V is empty and every active-active
rotation is a gauge, so no directional block touches act_V. The state is
never rotated, so `act_rot` is the identity. `--pc` attaches block2's PC
energy through `nevpt2_grad.attach_pc_energy`, the same conditioning guard
(on a state it builds without `pc`; the gradient driver's CASSCF branch
calls the same function for its own `--pc` since #126). Measured:
the PC energy is 2.1e-15 from the golden's `e_pc_total`, and every displaced
CI vector was guarded with no rank change. The steps are each flavour's own.

Byte-reproducible, measured: each file was generated twice with `--out` and
the two copies compare identical with `cmp`. A third copy, written to
`golden/` by the default path, is also identical. All three copies were run
one at a time while a concurrent PySCF sweep used the CPU, so the wall times
are an upper bound:

| file | bytes | wall |
|---|---:|---:|
| `n2_ccpvdz_cas44_casscf.srcgold` | 7440 | 21 s |
| `n2_ccpvdz_cas44_casscf_pc.srcgold` | 7440 | 165–212 s |
| `n2_ccpvdz_cas44_casscf_dir.srcgold` | 64913 | 3 s |
| `n2_ccpvdz_cas44_casscf_pc_dir.srcgold` | 64913 | 44 s |

`devtools/golden-repro.sh` regenerates all four from the committed sidecar
([`testing.md`](testing.md), "The golden regeneration gate"). A mismatched
sidecar is refused before the CASCI runs. LiF's `.mogold` handed to the N2
case lists nine mismatches (`nao`, `nmo`, `ncore`, `ncas`, `symbols`,
`basis`, `groupname`, `cas_irreps` and the geometry). N2's own DF sidecar
lists one (`auxbasis`). `reference_data/tests/test_source_oracle.py` pins
both. The four are in `kClassifiedOracles` with **empty** known-gap lists.
They were classified only after `nevpt2_sc_pseudodensity --golden
golden/n2_ccpvdz_cas44_casscf.nevpt2gold --tiles 3 [--pc] --source-oracle`
had held the GPU's terms to every block of each of them at the existing
tolerances, on the first run. The comparator's host suites
(`test/source_compare/`) expect act_V's three rows empty on these oracles
and on no other. Since #121 four `gpu` ctest entries run the GPU comparison
against them. The measurement is in [`testing.md`](testing.md), "The source
terms on CASSCF orbitals, measured".

```bash
uv run python reference_data/sc_source_oracle.py --case n2_ccpvdz_cas44_casscf \
    --mo-from golden/n2_ccpvdz_cas44_casscf.mogold --directional --pc --out /tmp/c.srcgold
cmp /tmp/c.srcgold golden/n2_ccpvdz_cas44_casscf_pc_dir.srcgold    # silent: identical
```

**Acrolein CAS(6,5) on CASSCF orbitals** (#135) has the directional pair
only, `golden/acrolein_def2svp_cas65_casscf[_pc]_dir.srcgold`, for the state
of `golden/acrolein_def2svp_cas65_casscf.nevpt2gold` and built from its
committed `.mogold`. The full flavour is out of reach for the reason the
CASCI twin's is. The same partition holds: act_V is empty, so each file
probes six blocks (12 kappa directions) and three CI directions, against
the CASCI twin's nine. The PC run attached block2's PC energy 5.7e-14 from
the golden's `e_pc_total` and guarded all 24 displaced CI vectors with no
rank change (smallest gap 2.4e+05).

Byte-reproducible, measured: each file was generated twice with `--out` and
the two copies compare identical with `cmp`, and the SC file a third time,
identical again. The two PC copies ran at the same time as each other, the
first two SC copies and a `-j 8` build, so those walls are an upper bound;
the third SC copy ran beside the two PC copies only:

| file | bytes | wall |
|---|---:|---:|
| `acrolein_def2svp_cas65_casscf_pc_dir.srcgold` | 466393 | 425 s (both copies) |
| `acrolein_def2svp_cas65_casscf_dir.srcgold` | 466393 | 29 s, 28 s, 26 s |

Both are in `kClassifiedOracles` with **empty** known-gap lists, classified
by the same protocol as N2's: only after `nevpt2_sc_pseudodensity --golden
golden/acrolein_def2svp_cas65_casscf.nevpt2gold --tiles 3 [--pc]
--source-oracle` had held the GPU's terms to every non-empty block of each,
on the first run ([`testing.md`](testing.md), "The directional source-term
oracle" has the numbers). They are read with the gitignored
`golden/acrolein_def2svp_cas65_casscf.gradgold` (regenerated with the
commands under "Regenerating the golden file"; sha256 in "LiF and acrolein on
CASSCF orbitals"), so their two ctest entries exist only where it has been
regenerated. `devtools/golden-repro.sh` stays N2-only.

```bash
uv run python reference_data/sc_source_oracle.py --case acrolein_def2svp_cas65_casscf \
    --mo-from golden/acrolein_def2svp_cas65_casscf.mogold --directional --pc --out /tmp/a.srcgold
cmp /tmp/a.srcgold golden/acrolein_def2svp_cas65_casscf_pc_dir.srcgold    # silent: identical
```

### The CASSCF skeleton nuclear derivatives

`reference_data/casscf_hessian.py` (#215) is item 1 of
[`hessian-theory.md`](hessian-theory.md), §8 for the CASSCF reference: the
nuclear derivatives with the wavefunction parameters held fixed **in the
connected sense**. The CI vector is fixed, and the orbitals ride
`nevpt2_grad.py`'s frozen orthonormal path,
`C(R) = S(R)^{-1/2} S(R0)^{1/2} C0` (its `frozen_coeff`, imported, not
re-derived). Along that path the energy is one scalar, `E_nuc + Σ h D +
½ Σ (pq|rs) Γ`, with the CASSCF state's full-space 1- and 2-RDM. The module
provides:

- **the perturbed Hamiltonians**, per Cartesian coordinate:
  `Skeleton.perturbed_h1(k)` (the full `n_mo × n_mo` MO `dh/dR_k`) and
  `Skeleton.perturbed_eri(k, ranges)` (`d(pq|rs)/dR_k` over any four MO
  slices). Each is the skeleton term (`hcore_generator`, `-int2e_ip1` on each
  index position) plus the connection term `t^k = C0^T S0 dC/dR_k`, solved
  elementwise in `S0`'s eigenbasis from `dS/dR_k`. `Skeleton.gradient()` is
  their contraction with the densities;
- **`L_RR`** (`Skeleton.hessian()`), as six named terms:
  - `nuc`: `hess_nuc`;
  - `h_skel`: `pyscf.hessian.rhf`'s `hcore_generator`;
  - `eri_skel`: `int2e_ipip1` / `int2e_ipvip1` / `int2e_ip1ip2`, contracted
    with the AO 2-RDM at weights 2 / 2 / 4;
  - `conn2`: the connection's second derivative against the generalized
    Fock;
  - `conn_x_skel`: the first-order connection against the skeleton
    derivative's generalized Fock;
  - `conn_x_conn`: the connection with itself through the
    `(n_mo·n_occ)²` matrix the module's docstring writes out.

None of it assumes stationarity, so it holds for any orthonormal `C0` and
CI vector. Its hand cases run on a state that is not stationary
([`testing.md`](testing.md), "The oracle's own tests",
`test_casscf_hessian.py`).

**How it is checked.** `main` rebuilds the case's committed CASSCF state
(`golden/<case>_casscf.mogold` through `generate_golden._build_state`'s
`--mo-from` path, every gate included) and checks five things:

- the rebuilt `E(R0)` against the state's own energy;
- the gradient against a ladder of `E` along every Cartesian coordinate,
  `h = 2^-3 … 2^-13` bohr;
- `L_RR` against energy ladders, `h = 2^-1 … 2^-9` bohr: second differences
  on the diagonal, and four-point mixed differences on three seeded
  whole-molecule direction pairs `d1ᵀ L d2` (plus every coordinate pair on
  the diatomics, up to 9 coordinates);
- **every element** of `L_RR`, column by column, against a ladder of the
  analytic gradient along that column's coordinate, `h = 2^-3 … 2^-13` bohr.
  The gradient is re-evaluated on the path at each rung, re-anchored at
  `C(R)`; the path composes, so this is the path energy's own slope. A
  column's run rule reads the max-norm change of the whole quotient vector,
  and its gate is the worst element's `|Richardson − analytic|`;
- the translational sum rules `Σ_A dE/dR_A = 0` and `Σ_B L_{R_A R_B} = 0`
  per direction pair, and `L`'s symmetry, all against 1e-9.

Every displaced energy is rebuilt from scratch at the displaced geometry, on
a symmetry-off molecule in the same frame. Each ladder is judged by
`nevpt2_grad.ladder_check`, the gradient's own rule: a run of three fourfold
halvings, with the Richardson value within `LADDER_TOL` = 1e-6 of the
analytic one. One addition: a component that is **zero by symmetry** has no
truncation error to walk down, so it passes only if the analytic value is
under 1e-9 and every rung is under 1e-6. The gradient is also compared with
`pyscf.grad.casscf`'s. That gradient assumes stationarity, so the comparison
is informational: the two differ by the orbital gradient's size.

**Measured** (2026-10-21; CPU only, `--workers 12` on the 12-core
reference box, single-threaded BLAS per process; every ladder passes on all
three committed CASSCF states):

| | N2 CAS(4,4) | LiF CAS(4,6) | acrolein CAS(6,5) |
|---|---|---|---|
| coordinates | 6 | 6 | 24 |
| `\|E(R0) − E_CASSCF\|` | 8.5e-14 | 1.9e-13 | 1.0e-12 |
| `max \|Σ_A dE/dR_A\|` | 7.1e-15 | 8.9e-16 | 2.1e-14 |
| `max \|Σ_B L_AB\|` | 3.1e-12 | 1.4e-11 | 1.4e-11 |
| `max \|L − Lᵀ\|` | 8.9e-15 | 1.8e-14 | 1.3e-13 |
| gradient vs `E` ladders: count, worst | 6, 5.2e-10 | 6, 1.3e-9 | 24, 4.3e-9 |
| `L_RR` vs `E` ladders: count, worst | 30, 2.5e-7 | 30, 1.1e-7 | 33, 4.2e-7 |
| `L_RR` vs gradient columns: count, worst element | 6, 2.3e-10 | 6, 3.5e-11 | 24, 2.9e-10 |
| ladders zero by symmetry (of the above) | 16 | 16 | 8 |
| `max \|dE/dR − pyscf.grad.casscf\|` (informational) | 1.9e-8 | 6.1e-8 | 1.2e-8 |
| analytic gradient / `L_RR`, one process | 0.2 s / 0.8 s | 0.1 s / 0.3 s | 5.5 s / 26 s |
| whole run, wall | 8 s | 5 s | 6.6 min |

Units are Eh/bohr for the gradient and Eh/bohr² for `L_RR`. The gate is
1e-6 for both, and the sum rules' and symmetry's is 1e-9. On acrolein the
energies take 35 s and the 528 path gradients of the column ladders 5.4
min. The `pyscf.grad.casscf` difference sits at the size of the committed
states' orbital gradients (`|g_orb|` 9.2e-8, 5.7e-7, 1.7e-7), as it should:
PySCF's gradient assumes a stationary point.

**What did not work as a gate on acrolein.** Two element-by-element
judgements fail the run rule without any element being wrong. A four-point
`E` ladder on every one of acrolein's 276 coordinate pairs passed 321 of
324 ladders. The three that failed, `L[e16,e18]`, `L[e16,e21]` and
`L[e19,e22]` (|L| 9.7e-4, 7.7e-4, 2.2e-2), showed only two consecutive
fourfold halvings before round-off. Their Richardson values agree with the
analytic elements to 5.4e-10, 1.1e-9 and 1.1e-8. Judging the gradient
columns element by element passed 575 of 576. The exception was `L[0,22]`
(|L| 4.9e-4), whose quotient has converged to 2.3e-13 by `2^-8` with
halving ratios 7.7, 5.2, 4.3, 4.1. That is an `h⁴` tail, then round-off, so
the first ratio falls outside the 3.2–4.8 window. Both are small elements,
with truncation errors too small to walk down three halvings before
round-off. So the script judges whole columns and gives the four-point `E`
ladders only seeded pairs above 9 coordinates
(`casscf_hessian.PAIR_LADDER_MAX_COORDS`).

```bash
uv run python reference_data/casscf_hessian.py --case n2_ccpvdz_cas44 --workers 12
uv run python reference_data/casscf_hessian.py --case lif_def2svp_cas46 --workers 12
uv run python reference_data/casscf_hessian.py --case acrolein_def2svp_cas65 --workers 12
```

The script writes nothing unless `--out` is given (the gradient, `L_RR` and
its terms, `.npz`). It is the skeleton half of the analytic CASSCF Hessian:
the parameter responses that complete
[`hessian-theory.md`](hessian-theory.md), §8's assembly are #219's.

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
general-range integral inventory is cheap. Their `_df` twins carry the
density-fitted one ("The density-fitted gradient sidecar").

CAS(4,4) is the small case that also exists on **CASSCF orbitals**,
conventional and DF (`n2_ccpvdz_cas44_casscf[_df]`, see "The MO sidecar:
committed CASSCF orbitals" above), so the sanitizer tier reads a
`reference_kind = casscf` file as well. CAS(8,8) has no CASSCF twin: at the
same molecule and basis it would exercise nothing the CAS(4,4) pair and the
CAS(10,10) one do not.

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

### LiF and acrolein on CASSCF orbitals

The CASSCF references the gradient validation needs (#123): the same two
molecules, geometries, bases and `--cas-irreps` selections as above, with
the orbitals optimised once by `--casscf --mo-out` and the golden plus its
gradient sidecar built on them by `--mo-from … --gradgold` (commands in
"Regenerating the golden file" above). Neither needs the other's
reason to exist: LiF is the cheap SC case with a componentwise sweep, and
acrolein the only one with a PC answer.

**Both optimisations stall under the generator's settings and are finished
by the polish** ("The CASSCF polish" above), 200 macro iterations plus 2.
Every guard then passes, none of them changed:

| | LiF CAS(4,6) | acrolein CAS(6,5) |
|---|---|---|
| `--casscf`: `converged` | **yes**, after the polish | **yes**, after the polish |
| `--casscf`: ‖g_orb‖ (optimiser's CI), against `CASSCF_GRAD_TOL` 1e-6 | 3.691e-9 | 5.774e-9 |
| `--casscf`: purity defect, against 1e-9 | 1.4e-15 | 1.0e-14 |
| active irreps | B1, B2 ×3 | A′ + A″ ×4 |
| E_CASSCF | −106.9035612554 | −190.6794323473 |
| `--mo-from` gate: ‖g_orb‖ (the generator's CASCI), against 1e-6 | 5.701e-7 | 1.721e-7 |
| `--mo-from` gate: \|E_CASCI − E_CASSCF\|, against 1e-9 | 2.984e-13 | 3.126e-13 |
| `_check_degeneracy` | passes | passes |
| block2 SC − PySCF SC, max class | (not run: `--no-pc`) | 6.1e-16 |
| `_check_pc_conditioning`, smallest gap (≥ 1e3 needed) | **refused**: 8.8e2 (`iabcpqg*`) | passes: 2.4e5 (`ijabpq+`) |
| E_corr SC (PySCF) | −0.1473025529 | −0.5221559894 |
| E_corr PC (block2) | refused, so `--no-pc` | −0.5229006181 |
| (O2), against 1e-10 / (O1) measured, not asserted: core–act, core–virt, act–virt | 2.13e-14 / 0, 5.41e-3, 6.26e-1 | 1.55e-12 / 3.55e-1, 7.14e-3, 2.04e-1 |
| `--casscf` wall, single-threaded | 3.7 s | 94.9 s |
| `--mo-from --gradgold` wall | 2.0 s | 38.6 s |

The `--mo-from` gate's gradient is the one to read against the refusal: it
uses the 1e-12-Davidson CASCI the golden records, whose CI residual floors
near 1e-6 ("Reproducibility" above), so it sits 30–150× above the
optimiser's own (the same effect as at N2, "The `--casscf` / `--mo-from`
gate thresholds" above); LiF's 5.7e-7 leaves 1.75× of headroom, acrolein's
5.8×.

**LiF's PC answer is refused on CASSCF orbitals too.** Generated without
`--no-pc` into scratch, `_check_pc_conditioning` stops at `iabcpqg*` (Si):
largest dropped singular value 3.5e-14, smallest kept 7.1e-14, a gap of
8.8e2 against the 1e3 it needs — the same block, and the same continuum, that
refuses it on RHF orbitals. So the LiF CASSCF golden is `--no-pc`, like its
CASCI twin, and carries SC fields only.

**LiF's point is a saddle, and that is not a refusal.** On the committed
orbitals, `nevpt2_grad.py --case lif_def2svp_cas46 --reference casscf` finds
the same spectrum #122 found on scratch ones ("The CASSCF branch" above): two
negative eigenvalues (−7.3215e-3, ×2) and a near-null pair (2.8218e-6, ×2),
all pure orbital rotations with the right-hand side on them at ≤ 6.7e-15
of `|b|`. No guard looks at the Hessian, and none refuses: the
symmetry-adapted optimiser converges within its point group, and these are
rotations it cannot take. **That they break the symmetry is measured, not
assumed**: the dense orbital-orbital Hessian (`mc1step.gen_g_hop`'s `h_op`,
column by column, at the committed orbitals and the generator's CASCI
RDMs) over all 154 non-redundant rotations has the same structure — two
negative modes at −3.6608e-3 and two near-null at 1.4109e-6, half the
coupled values above in `gen_g_hop`'s normalisation — and **every one of
those four modes has zero weight on same-irrep rotations** (in C2v they
rotate A1/A2 orbitals into the B1/B2 π space, e.g. 18 A1 ↔ 8 B1, 19 A2 ↔
9 B2). Restricted to the 44 symmetry-allowed rotations, the space the
optimiser works in, the same Hessian is positive definite, smallest
eigenvalue +3.44e-3. So the committed point is a minimum within C2v and a
saddle only along symmetry-breaking σ/π mixings; the gradient carries no
component along them (‖g_orb‖ is 5.701e-7 over both sets). It is also not
what stalled the optimiser: the polish converges inside the
positive-definite symmetric space ("The CASSCF polish" above). Leaving C2v
to follow those modes would mean building the molecule without symmetry,
which "Point-group symmetry" above rules out for reproducibility.

**What is committed.** LiF: all three files (`.mogold` 4 597 B, golden
202 138 B, sidecar 916 239 B). Acrolein: the `.mogold` (46 732 B) and the
golden (7 121 421 B, like its CASCI twin); its sidecar (56 940 831 B) is
**gitignored**, so its regeneration command and fingerprint are the
deliverable. Each file regenerated byte-identically in a second process
(each golden also with `--gradgold` dropped):

| file | sha256 |
|---|---|
| `lif_def2svp_cas46_casscf.mogold` | `fc35f5199f4f307ff62bc7348e0272581f00358b07d636736085284dbc4dd47f` |
| `lif_def2svp_cas46_casscf.nevpt2gold` | `28367a4bb63c7842e907899fcc46f255595fa874a2d7ecd4f584bd28f2ee80b5` |
| `lif_def2svp_cas46_casscf.gradgold` | `8796d740035e9751afb2fa53a86bd42e9fb3da489292ce0e6615355536d292de` |
| `acrolein_def2svp_cas65_casscf.mogold` | `08f999cea4e6371eb7189ac80be8eadf1bde2bbb47c0d0f37e1111221e39f433` |
| `acrolein_def2svp_cas65_casscf.nevpt2gold` | `007c5a7e9685a895051c22084232de77a31c9ec4ac0d248578bdc3482aaecd45` |
| `acrolein_def2svp_cas65_casscf.gradgold` (gitignored) | `300341d45a30145e174b788f23f5db4b6f0fb5bd8c683ece8d0f0d37fdd87c8a` |

Measured on the RTX 3080 box's CPU (PySCF 2.14.0, NumPy 2.5.3, Python
3.12.3), offline: no card and no build are involved. The two gradient
sidecars carry the provenance contract of "The CASSCF-reference gradient
sidecar" above: byte-reproducible from the committed `.mogold`, and from
scratch only to the `--mo-from` gate's tolerance, so `golden-repro.sh` makes
no claim over them (it regenerates only N2). Since #125 LiF's golden has two
ctest entries, `nevpt2_lif_cas46_casscf` (the energy) and
`nevpt2_sc_pseudodensity_dump_lif_def2svp_cas46_casscf` (its GPU source
terms, which the SC gradient verdict reads). Acrolein's has #135's two
directional-oracle entries, registered only where its gitignored sidecar
has been regenerated.

## HSOH, the C1 case

The nuclear-Hessian milestone needs a molecule with **no point-group
symmetry at its own minimum** (#213). Then every Cartesian displacement
keeps the group, so the auto-detected group and the selected active space
survive all 3N of them, and the full 3N×3N Hessian is reachable without
changing how the active space is selected. Nothing is empty or dropped for
symmetry's sake either, so the §9.1 refusals of
[`hessian-theory.md`](hessian-theory.md) do not fire.

The case is **HSOH (sulfenic acid) CAS(6,6)/def2-SVP on CASSCF orbitals**:
`reference_data/geometries/hsoh.xyz`, `--ncas 6 --nelecas 6`, the active
space the generator's default picks (the RHF orbitals around the Fermi
level, there being no irreps to pick by), and the commands in "Regenerating
the golden file" above. The generator logs `point group: C1 (detected)`
at both steps. There is no CASCI twin.

**Why CAS(6,6), not the CAS(4,4) the issue asked about.** The test was the
one `nevpt2_numgrad.py --reference casscf` applies: a cold-start
`casscf_optimise` at the reference and at each ±0.0025 bohr single-coordinate
displacement must stay on one CASSCF branch. The table counts the displaced
states whose E_CASSCF jumps by more than 1e-4 Eh. All rows use def2-SVP and
the generator's own optimiser path.

| candidate | geometry | branch jumps (of 24 / 36) | other findings |
|---|---|---:|---|
| HSOH CAS(4,4) | its own CASSCF minimum | 3 (−22.2 mEh) | the full componentwise sweep's continuity guard refused 5 of 72 states, the same lower branch |
| HSOH CAS(2,2) | the CAS(4,4) minimum | 9 (−12.8 mEh) | |
| HSOH CAS(4,3) | its own minimum | 0 | block2's PC Si class came out −0.000000000000: an empty class in all but name, so it was rejected |
| HSOH CAS(6,5) | its own minimum | 0 | 5 of 25 optimisations did not converge, even with the polish |
| **HSOH CAS(6,6)** | its own minimum | **0** | 25 of 25 converged, ‖g_orb‖ ≤ 1.0e-8 |
| FCH₂OH (gauche) CAS(4,4) | a model geometry, not optimised | 35 (−12.8 mEh) | |
| FCH₂OH CAS(2,2) / CAS(4,3) | the same | 19 / 12 (12 not converged) | |
| FCH₂OH CAS(6,6) | the same | 0 | not taken further: HSOH, the first candidate, works |

HSOH's CAS(6,6) is the three σ bonds (S–O, S–H, O–H) with their σ*. Its
natural occupations are 1.9816, 1.9756, 1.9676, 0.0331, 0.0246, 0.0175.
CAS(4,4)'s jumps all land on one lower stationary point, 22.2 mEh down,
which a cold start reaches from some displaced geometries and not from the
reference.

**The geometry is a CASSCF minimum, on the generator's own path.** It was
found by BFGS on Cartesian coordinates. Each evaluation was a cold
`generate_golden.casscf_optimise` (the `--casscf` body, polish included) in
the auto-detected frame, with `pyscf.grad.casscf` rotated back into the
input frame. The search started from a model structure (r(SO) 1.66 Å,
r(SH) 1.35 Å, r(OH) 0.965 Å, ∠HSO 98°, ∠SOH 107.5°, dihedral 90°) and
stopped after 16 iterations at max |∂E_CASSCF/∂x| 6.7e-7 Eh/bohr. The
coordinates are written to 10 decimals, Å. At that point the CP-CASSCF
Hessian `nevpt2_grad.py` builds (orbital + CI, dimension 875) has **no
negative eigenvalue**: the smallest is +6.98e-3 and the condition number
5.5e4. This is the case where C1 removes LiF's caveat ("LiF and acrolein on
CASSCF orbitals" above): every rotation is symmetry-allowed, and the point
is a minimum along all of them.

The case's properties, measured from the committed files:

| | HSOH CAS(6,6) |
|---|---|
| point group (detected / used) | C1 / C1 |
| `--casscf`: `converged` | **yes**, after the polish (200 macro iterations + 2) |
| `--casscf`: ‖g_orb‖ (optimiser's CI), against 1e-6 | 9.093e-9 |
| `--casscf`: purity defect, against 1e-9 | 2.9e-15 |
| E_CASSCF | −473.4209458161 |
| `--mo-from` gate: ‖g_orb‖ (the generator's CASCI), against 1e-6 | 4.653e-7 (2.1× headroom) |
| `--mo-from` gate: \|E_CASCI − E_CASSCF\|, against 1e-9 | 6.821e-13 |
| n_core / n_act / n_virt, n_det | 10 / 6 / 26, 400 |
| `_check_degeneracy`: smallest same-irrep canonical gap, core / virtual (≥ 1e-4 needed) | 2.33e-3 / 2.98e-3 Eh |
| block2 SC − PySCF SC, max class | 2.1e-15 |
| `_check_pc_conditioning`, smallest gap (≥ 1e3 needed) | passes: 5.9e5 (`rabcpqg*`); `iabcpqg*` 2.4e6, every other block ≥ 7.8e9 |
| E_corr SC (PySCF) | −0.2250783777 |
| E_corr PC (block2) | −0.2262152688 |
| (O2), against 1e-10 / (O1) measured, not asserted: core–act, core–virt, act–virt | 2.15e-13 / 2.52e-1, 1.18e-3, 3.97e-1 |
| `--casscf` wall, single-threaded | 14.9 s |
| `--mo-from --gradgold` wall | 28.1 s |

The smallest core gap is the S 2p set, which C1 splits; it sits 23× above
the refusal.

**Its sidecar carries `grad_num`, the one CASSCF sidecar that does.**
`nevpt2_numgrad.py --reference casscf` writes it in place for this case only
(`sidecar_grad_num_casscf` in its `CASES`); for N2 and LiF the route still
writes `--npz` only, and refuses the in-place write. `nevpt2_grad.py
--reference casscf` reads it by default, as a CASCI case's sidecar is read.
The sweep: all 12 components, 72 displaced states, every one inside the
continuity guard (largest `e_jump` 1.6e-6, `1 − σ_min` 2.9e-6, `1 −
|⟨c₀|c⟩|` 5.5e-7, ‖g_orb‖ 1.0e-9). The reference re-optimised bit-identically
to the committed `.mogold`, and the cold-versus-warm probe found the same
stationary point. The largest Richardson residual is 3.8e-7 Eh/bohr (S x);
the other eleven are ≤ 1.1e-7. The free check, the CASSCF-only numerical
gradient against `pyscf.grad.casscf`, agrees to 3.2e-9. Translational sums
are ≤ 1.8e-8. Wall: 153 s at `--workers 12` (budget 600 s). The analytic
gradient against it is [`testing.md`](testing.md), "SC on a CASSCF
reference: the C1 case".

**What is committed, and its reproducibility.** All three files: the
`.mogold` (14 519 B), the golden (1 455 261 B, `NEVPT2G2`, SC and PC) and
the sidecar (8 593 652 B: the 24 `--gradgold` arrays and the four `grad_num*`
ones). Unlike every other CASSCF-orbital file, `devtools/golden-repro.sh`
regenerates all three byte for byte. The `.mogold` comes from a fresh
`--casscf`. The golden and the sidecar come from `--mo-from` the committed
`.mogold`, with the sweep run on the sidecar just written. The sweep's
reference must re-optimise to the committed orbitals bit for bit or it
refuses, and every displaced state is a cold start, so the `grad_num` is as
deterministic as the golden.

| file | sha256 |
|---|---|
| `hsoh_def2svp_cas66_casscf.mogold` | `5956ea72c428bad1c99f977fa99813ecc118e6e450551def20edc54697e0a7b3` |
| `hsoh_def2svp_cas66_casscf.nevpt2gold` | `5c12261ffc5842c4655f0df20c536bd8d82f31cce52140222f5dac2462b457e4` |
| `hsoh_def2svp_cas66_casscf.gradgold` | `4123f945de4a4f1b4be6be85a384cde7f1ffed8caf04b16a78bc7fcf4485e831` |

Measured on the two-card box's CPU (Ryzen 9 5900X; PySCF 2.14.0, NumPy
2.5.3, Python 3.12.3). Its three ctest entries are in
[`testing.md`](testing.md), "The CASSCF-orbital entries".

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
| `reference_data/generate_golden.py` | the **only** file that defines the golden state (`block2_pc_probe.py` imports its `_build_state`). Runs a deterministic `mcscf.CASCI` (see "What "CASSCF input" means here" above), canonicalises it exactly as `pyscf.mrpt.NEVPT.kernel` does, and writes the CI vector, `dm1`/`dm2`, every MO-integral block, and PySCF's own per-class NEVPT2 answer into one flat binary file. `--df` does the same density-fitted and adds the `B_*` blocks. Unless `--no-pc`, it also runs block2's PC-NEVPT2 on the same state and writes `pc_class_energies` / `e_pc_total` (see "PC fields" above). `--casscf --mo-out` / `--mo-from` is the two-step CASSCF-orbital path, and the only thing here that runs an orbital optimiser (see "The MO sidecar" above), polishing once with a tighter AH solve if the first run does not converge ("The CASSCF polish" above) |
| `reference_data/block2_pc_probe.py` | rebuilds a golden's state with `generate_golden._build_state` (imported, not copied), checks it is bit-identical to the committed CI vector, and runs block2's `WickSCNEVPT2` and `WickICNEVPT2` (PC) on it: per-class energies, timings, peak RSS, and the PC linear-solve / metric conditioning (`--metric`). Writes no golden. See [`pc-nevpt2.md`](pc-nevpt2.md) |
| `reference_data/nevpt2_numgrad.py` | the **numerical** nuclear gradient: displaced RHF → CASCI → SC-NEVPT2 runs (states built by `generate_golden._build_state`, imported) differenced and Richardson-extrapolated into a sidecar's `grad_num` / `grad_num_residual` / `grad_num_probed` / `grad_num_step`. Refuses a displacement whose detected point group — or whose core/active irrep selection — differs from the reference geometry's, rather than produce a number from a different active space, and refuses a golden whose `reference_kind` is not the `--reference` asked for. `--reference casscf` re-optimises every displaced state through `generate_golden.casscf_optimise` plus a finish, guarded by the continuity check, with a directional screen (`--directional N --analytic PATH`), `--workers N` processes and the states kept for reuse (`--states-dir`) ("The numerical gradient of a CASSCF reference" above); with `--pc-states NPZ` it runs no sweep and differences `pc_on_states.py`'s GPU PC energies on a finished one into `grad_num_pc` (#126, "Differencing them: `--pc-states`" above). `--pc` differences block2's partially-contracted energy on the same displaced states (and writes no sidecar); `--npz PATH` writes the gradient arrays alone, which is how a PC one, or any case whose committed sidecar holds no `grad_num`, reaches `nevpt2_grad.py --grad-num`. `--df` differences the density-fitted energy into the `NEVPT2F1` sidecar beside a `_df` golden, refusing a golden or sidecar of the other flavour or auxiliary basis (#167, "The density-fitted numerical gradient" above). Writes no golden and no new sidecar; only the two gradient-validation cases, `n2_ccpvdz_cas44_df` and, on the `--reference casscf` route, `hsoh_def2svp_cas66_casscf` (#213) carry its arrays in their sidecars (see "The numerical gradient" and "HSOH, the C1 case" above) |
| `reference_data/nevpt2_grad.py` | the **analytic** nuclear gradient: a `pyscf.grad.lagrange.Gradients` subclass that solves the CI and orbital Z-vector equations of [`gradient-theory.md`](gradient-theory.md), §2 and contracts them into a gradient, differentiating one scalar along a frozen orthonormal path. SC by default, block2's PC with `--pc` (the same Lagrangian: only `E_corr` changes). Source terms from PySCF by default, or from the GPU's `.gradsrc` dump (`--source dump`, written by `nevpt2_sc_pseudodensity --dump-source-terms`; with `--pc` a PC dump, `--pc --dump-source-terms`, #74 — a dump of the other scheme is refused), or both side by side (`--source both`: the two compared componentwise and solved on one sweep; [`testing.md`](testing.md), "The GPU as the source-term provider"). `--edot-check` instead holds the dump's forward `Edot(kappa)` along its seeded directions — frozen-digest tangent plus the digests' term — against PySCF's exact directional derivative, step-laddered, and stops ([`testing.md`](testing.md), "The digests' orbital response, forward mode"). Ends by gating itself against `grad_num` and printing `PASS:`/`FAIL:` ([`testing.md`](testing.md), "The analytic gradient against the numerical one"). Writes **nothing** unless `--out` is given, and never a committed file; `--casci-only` is its own validation (see "The analytic gradient" above). `--reference casscf` is the CP-CASSCF branch, SC or `--pc` (#122, #126; "The CASSCF branch" above), from either provider's source terms (`--source dump` / `both`: SC #125, PC #163), and `--workers N` forks its `kappa` sweep, bit-identically. `--df` (#166) differentiates a `_df` golden's density-fitted SC energy, every evaluation on fitted integrals, against its NEVPT2F1 sidecar's `grad_num` ([`testing.md`](testing.md), "The density-fitted gradient against the numerical one") |
| `reference_data/pc_on_states.py` | the **GPU's PC energies on a CASSCF sweep's kept states** (#138): per state of a `nevpt2_numgrad.py --reference casscf --states-dir`, a transient `--no-pc --mo-from` golden (NaN PC placeholders, built in `--workers` processes, deleted afterwards, never in `golden/`), the wiring check against the manifest, `nevpt2_demo` SC (must PASS) and `--pc` (the full-precision line and the per-class spectrum), and `_check_pc_conditioning` on the device's gaps. Writes a keyed `.npz` for `nevpt2_numgrad.py --pc-states` (#126) and nothing else; needs a built demo and a card ("The GPU's PC energies on the kept states" above) |
| `reference_data/pc_null_mode_probe.py` | **what PC's singular-metric cut drops and how it moves with the CI vector**: rebuilds every PC class's `S`/`K`/`S_1` from a golden the device's way (reproducing its block2 `pc_class_energies`), classifies the dropped modes (null for every singlet / every vector of the abelian irrep / the state's symmetry family), measures the kept–dropped coupling and the second-order lift per direction kind, and checks the PC second derivative with and without the null-space rotation terms against a difference of the exact first derivative. CPU only, writes nothing ([`pc-nevpt2.md`](pc-nevpt2.md), "What the cut drops, and how it moves") |
| `reference_data/pc_orbital_hessian_probe.py` | **PC's orbital Hessian blocks before code** (#199): on `pc_null_mode_probe.py`'s class matrices, rotates a golden's active integrals and stored blocks at the fixed CI vector. It checks three things. `H_kk` along two active–active directions, with no rotation term, against an energy ladder and a Baker–Campbell–Hausdorff gauge. `H_kc` with and without its one rotation term, against a difference of the exact orbital derivative and the active-rotation gauge `Eddot_cc(what c, u) + Edot(what u)`. Along intra-space pairs, the PC class energies' mixed derivative against four-point ladders, beside SC's `rho` on the same (class, pair). Reuses `sc_empty_perturber_probe.py`'s rotations. CPU only, writes nothing ([`pc-nevpt2.md`](pc-nevpt2.md), "The orbital blocks, offline"; [`hessian-theory.md`](hessian-theory.md), §6.4–§6.8) |
| `reference_data/sc_empty_perturber_probe.py` | **what SC's guarded (empty) perturbers do to the orbital Hessian**: transcribes PySCF's eight SC class formulas onto a golden's stored blocks (reproducing its `class_energies`), classifies the guarded tuples as empty by their integrals or merely small, and along seeded intra-space direction pairs evaluates the sixth term, the proportionality defect `rho`, the polarisation a four-point ladder converges to (checked against one, under real rotations), the parallelogram defect, and what the guard drops at first order. Reuses `pc_null_mode_probe.State` for the RDMs. CPU only, writes nothing ([`testing.md`](testing.md), "The empty-perturber term, offline"; [`hessian-theory.md`](hessian-theory.md), §5.6) |
| `reference_data/casscf_hessian.py` | the **CASSCF reference's skeleton nuclear derivatives** (#215): the perturbed Hamiltonians `dh/dR`, `d(pq\|rs)/dR` in the MO basis along `nevpt2_grad.py`'s frozen orthonormal path (`frozen_coeff`, imported), and the second partial `L_RR` from `pyscf.hessian.rhf`'s integral drivers contracted with the CASSCF densities, plus the connection's second-order terms. Checked against ladders of the path energy and the translational sum rules on a case's committed `_casscf.mogold` state. CPU only; writes nothing unless `--out` (see "The CASSCF skeleton nuclear derivatives" above) |
| `reference_data/casscf_repro_probe.py` | the **CASSCF reproducibility** measurement: ≥5 back-to-back RHF → symmetry-adapted `mcscf.CASSCF` → orbital-gradient → fresh `mcscf.CASCI` → SC-NEVPT2 runs under `generate_golden`'s own settings (imported, so single-threaded with the same Davidson tolerance), reporting the spread of E_CASSCF / E_corr / ‖g_orb‖ and whether any run changed stationary point. `--perturb EPS` starts the optimiser from a random same-irrep rotation instead. `--case NAME --displace COMP --steps …` runs `nevpt2_numgrad.py --reference casscf`'s own state path along a ladder of displacements and prints the continuity guard's three signals per step (where its thresholds come from). Writes no golden; its output is what "A fresh CASSCF reproduces, measured" above quotes, and where the `--mo-from` gate thresholds come from |
| `reference_data/sc_source_oracle.py` | **PySCF's exact SC source terms** for N2 CAS(4,4) and LiF, written as `golden/<case>.srcgold` out of `nevpt2_grad.py`'s own machinery at its converged steps; the oracle `nevpt2_sc_pseudodensity --source-oracle` gates the GPU's against (see "The SC source-term oracle" above). With `--pc`, **block2's PC ones** for N2 CAS(4,4), `golden/n2_ccpvdz_cas44_pc.srcgold` ("The PC flavour"). With `--directional`, the **directional flavour** for acrolein, N2 and LiF, `golden/<case>[_pc]_dir.srcgold` ("The directional flavour"). With `--case n2_ccpvdz_cas44_casscf --mo-from FILE`, all four kinds on **converged CASSCF orbitals** ("On CASSCF orbitals"), and with `--case acrolein_def2svp_cas65_casscf --mo-from FILE --directional`, acrolein's directional pair on them (#135), with the sidecar fingerprint-checked as `generate_golden.py --mo-from` checks it. Byte-reproducible |
| `reference_data/golden_format.py` | the **one Python parser** of the three formats in `golden/` (and of the `.srcgold` oracle, plus its writer `write_srcgold`): the header tables (keyed by magic) the writers pack from and the readers read by, `read_golden` / `read_golden_header` / `read_gradgold` / `read_mo_sidecar` / `read_srcgold`, and `arrays-at FILE` for the regeneration check's `cmp -i`. `struct` and NumPy only, and no thread settings, so the timing script can use it (see "The file format" above) |
| `reference_data/golden_layout_check.py` | the Python half of the **layout contract**: every committed file of both golden layouts, and every sidecar, through every reader and back through its writer byte for byte, plus a scan for any parser outside `golden_format.py`. Ends `PASS:`/`FAIL:`; a manual gate, no ctest entry ([`testing.md`](testing.md), "The golden layout contract") |
| `reference_data/tests/` | the **oracle's own tests** (pytest, `uv run pytest`): every format's writer → reader round trip and every reader refusal in `golden_format.py`, and `_check_degeneracy` / `_check_pc_conditioning` / `_check_gradient_invariants` on synthetic inputs just either side of each threshold, plus `nevpt2_numgrad.py`'s own refusals (`test_numgrad_casscf.py`, and `test_numgrad_df.py`'s symmetry-changing displacement refused on the `--df` route). No solve, no card; a `cpp-tier.sh` phase ([`testing.md`](testing.md), "The oracle's own tests") |
| `devtools/golden-repro.sh` | the **regeneration gate**: the committed N2 CAS(4,4) / CAS(8,8) references and the C1 case's three HSOH files regenerated into scratch and compared byte for byte (see "Regenerating the golden file" above; [`testing.md`](testing.md), "The golden regeneration gate") |
| `reference_data/pyscf_contract_time.py` | the CPU side of the head-to-head: reads a golden file and times PySCF's `NEVPTcontract` (`make_dm123` + `_contract4pdm` ×2) on the *same frozen CI vector* the demo reads, with no CASCI re-solve (see [`performance.md`](performance.md), "Head-to-head vs PySCF") |
| `golden/n2_ccpvdz_cas1010.nevpt2gold` | the **default case**: CAS(10,10) on N2/cc-pVDZ |
| `golden/n2_ccpvdz_cas1212.nevpt2gold` | CAS(12,12) on the same molecule, the scaling case (see [`performance.md`](performance.md), "Scaling with active-space size"). `slow`-labelled; run with `--tiles 40`. Carries **no** PC fields: its PC metric is ungapped (see "PC fields" above) |
| `golden/n2_ccpvdz_cas1012.nevpt2gold` | the 12-orbital **PC** case: CAS(10,12) on the same molecule (both N 1s in the core, n_det = 627k), with SC and PC fields. Exists because CAS(12,12) cannot be a PC golden. Its SC fields pass `nevpt2_demo --tiles 40` (CUDA, \|ΔE_corr\| 4e-15); its ctest entry is the PC one, `nevpt2_cas1012_pc` (`--pc --tiles 40`, `slow`) |
| `golden/n2_ccpvdz_cas1010_df.nevpt2gold`, `golden/n2_ccpvdz_cas1212_df.nevpt2gold`, `golden/n2_ccpvtz_cas1010_df.nevpt2gold` | the **density-fitted** goldens (`--df`, aux basis `<basis>-jkfit`), the oracle for `nevpt2_df_demo`: PySCF's DF-NEVPT2 as the answer, plus the four three-index `B_*` blocks. A different reference from the conventional files (see "Density fitting" above). The cc-pVTZ one (nvirt = 48) exists for slab boundaries, not memory pressure |
| `golden/n2_ccpvdz_cas44.nevpt2gold`, `golden/n2_ccpvdz_cas88.nevpt2gold` and their `_df` twins | the **small** cases (see "The small cases" above) |
| `golden/n2_ccpvdz_cas44_casscf.mogold`, `golden/n2_ccpvdz_cas44_casscf.nevpt2gold` | the **CASSCF-orbital** CAS(4,4) pair: the committed converged `mo_coeff` (6.6 KB) and the golden a deterministic CASCI on it writes (`reference_kind = casscf`, `NEVPT2G2`, 0.24 MB, SC **and** PC). The `small` CASSCF entries run on it, and `test/golden`'s `GoldenCommittedCasscfTests` checks the two reference-kind gates split on a committed file — the source-term gate passing it, the Lagrangian gate refusing it (see "The MO sidecar: committed CASSCF orbitals" above) |
| `golden/n2_ccpvdz_cas1010_casscf.mogold`, `golden/n2_ccpvdz_cas1010_casscf.nevpt2gold` | the same at the **production size**, CAS(10,10) (1.13 MB, SC and PC; `--tiles 3`). Its sidecar's optimisation takes 16 macro iterations and its golden ~2 min to rebuild, nearly all of it block2's dense 4-RDM |
| `golden/n2_ccpvdz_cas44_casscf_df.mogold`, `golden/n2_ccpvdz_cas44_casscf_df.nevpt2gold` | the **density-fitted** CASSCF-orbital case, `nevpt2_df_demo`'s CASSCF reference (0.47 MB, SC and PC): fitted throughout, the orbital optimisation included, so the aux basis is part of the sidecar's fingerprint |
| `golden/n2_ccpvdz_cas44.gradgold`, `golden/n2_ccpvdz_cas88.gradgold` | the **gradient sidecars** for those two cases: the general-range MO-integral blocks an analytic gradient reads, plus the full-range Fock matrices and `mo_coeff` (see "The gradient sidecar" above). Separate files, so the goldens beside them are untouched. `test/golden/gradgold_tests.cpp` round-trips the reader and checks the CAS(4,4) one against the golden beside it. Neither holds a `grad_num`: they stay the 24 arrays `--gradgold` writes, and CAS(4,4)'s numerical gradient goes to an `--npz` instead — `nevpt2_numgrad.py` refuses the in-place write for that case rather than move these bytes ([`testing.md`](testing.md), "The analytic gradient against the numerical one") |
| `golden/n2_ccpvdz_cas44_df.gradgold`, `golden/n2_ccpvdz_cas88_df.gradgold` | the **density-fitted gradient sidecars** (#164, `NEVPT2F1`; 0.86 / 0.95 MB) beside the two small `_df` goldens: `B_cc` and `B_vv` (the blocks of `(L\|pq)` the golden lacks), the fitted calculation's six full-range matrices, and the four tied orbital-energy arrays (see "The density-fitted gradient sidecar" above). The CAS(4,4) one also carries the density-fitted **numerical gradient**, `nevpt2_numgrad.py --df`'s four `grad_num*` arrays (#167, "The density-fitted numerical gradient" above), so it takes two commands to regenerate. `devtools/golden-repro.sh` regenerates both byte for byte, that one with its `grad_num`; `test/golden/gradgold_tests.cpp`'s `GradGoldDfCas44Tests` holds the CAS(4,4) one to its golden, and `nevpt2_gradient_source_cas44_df` reads it on a card; `nevpt2_sc_pseudodensity --kappa-full` reads both (#165); `nevpt2_grad.py --df` gates against the CAS(4,4) one's `grad_num` (#166); `nevpt2_df_demo` reads neither |
| `golden/n2_ccpvdz_cas44_casscf.gradgold` | the **CASSCF-reference gradient sidecar** (1.52 MB): the same 24 arrays on the committed CASSCF orbitals of `n2_ccpvdz_cas44_casscf.mogold`, written by its `--mo-from` step with `--gradgold`. Reproducible from the committed `.mogold` byte for byte, from scratch only to a tolerance, so `golden-repro.sh` makes no claim over it; (O2) asserted, (O1)/(O3) not (see "The CASSCF-reference gradient sidecar" above) |
| `golden/n2_ccpvdz_cas44.srcgold`, `golden/lif_def2svp_cas46.srcgold` | the **SC source-term oracles** (7.4 / 10.1 KB): PySCF's exact `A` and `P y` for those two states (see "The SC source-term oracle" above). Read by `loadSrcGold` and nothing else at run time; `test/golden`'s `GoldenLayoutContractTests` and `SrcGoldLoadErrorTests` hold the reader to them |
| `golden/*_dir.srcgold` (five on RHF orbitals; the CASSCF-orbital four are the `n2_ccpvdz_cas44_casscf*` and `acrolein_def2svp_cas65_casscf*` rows below) | the **directional source-term oracles** (#105; `NEVPT2SD` / `NEVPT2PD`): seeded directions per block and the oracle's converged derivative along each, for acrolein CAS(6,5) SC and PC, N2 CAS(4,4) SC and PC and LiF SC ("The directional flavour" above has their sizes). `nevpt2_sc_pseudodensity_oracle_dir_*` hold the GPU's terms to them; `test/source_compare/` and `test/golden` hold the comparator and the reader |
| `golden/n2_ccpvdz_cas44_pc.srcgold` | the **PC source-term oracle** (7.4 KB, `NEVPT2P1`): block2's `A` and `P y` for N2 CAS(4,4)'s PC energy ("The PC flavour" above). `nevpt2_sc_pseudodensity_oracle_pc_n2_ccpvdz_cas44` holds the GPU's PC source terms to it (#74); `test/source_compare/` holds the comparator to it, and `test/golden` the reader |
| `golden/n2_ccpvdz_cas44_casscf.srcgold`, `_pc.srcgold`, `_dir.srcgold`, `_pc_dir.srcgold` | the **source-term oracles on CASSCF orbitals** (#117): all four kinds (SC/PC, full/directional; 7.4 KB each full, 65 KB each directional) for `n2_ccpvdz_cas44_casscf.nevpt2gold`'s state, built with `sc_source_oracle.py --mo-from` the committed `.mogold` ("On CASSCF orbitals" above). `devtools/golden-repro.sh` regenerates them byte for byte. `test/source_compare/` and `test/golden` hold the comparator and the reader to them, and four `gpu` ctest entries hold the device's terms to them (#121; [`testing.md`](testing.md), "The source terms on CASSCF orbitals, measured") |
| `golden/acrolein_def2svp_cas65_casscf_dir.srcgold`, `_pc_dir.srcgold` | acrolein CAS(6,5)'s **directional source-term oracles on CASSCF orbitals** (#135; 466 KB each) for `acrolein_def2svp_cas65_casscf.nevpt2gold`'s state, built with `sc_source_oracle.py --mo-from` the committed `.mogold` ("On CASSCF orbitals" above). `nevpt2_sc_pseudodensity_oracle_dir[_pc]_acrolein_def2svp_cas65_casscf` hold the GPU's terms to them where the gitignored sidecar has been regenerated; `test/source_compare/` and `test/golden` hold the comparator and the reader |
| `reference_data/geometries/n2.xyz` | the N2 geometry as an `.xyz`, byte-for-byte `generate_golden.py`'s own default `--atom "N 0 0 0; N 0 0 1.1"` — so the two gradient scripts, which take a geometry file, can be pointed at the committed `n2_ccpvdz_cas44` state. That it *is* the same state is checked, not assumed: both scripts refuse unless the rebuilt `E_CASCI`, `E_corr` and `e_pc_total` match the golden's (measured 0.0, bit-identical). N2 is the **deliberately symmetric** case of the gradient comparison (see [`testing.md`](testing.md), "The analytic gradient against the numerical one") |
| `reference_data/geometries/salicylaldimine.xyz`, `reference_data/basis/6-31gss-rifit.nw`, `golden/salicylaldimine_631gss_cas88[_df].nevpt2gold` | the non-N2 case (see "Salicylaldimine CAS(8,8)" above). The two goldens are **gitignored** and regenerate byte-identically in ~2.5 min each; their four ctest entries register only once they exist |
| `reference_data/geometries/lif.xyz`, `golden/lif_def2svp_cas46.nevpt2gold`, `golden/lif_def2svp_cas46.gradgold` | the **LiF smoke test** of the gradient-validation pair (see "LiF and acrolein, the gradient-validation pair" above): linear, Coov → C2v, CAS(4,6) as the pure π space by irrep, def2-SVP. **SC fields only** — its PC metric is refused as ungapped, so no `--pc` entry exists for it. Both files are committed (0.20 / 0.92 MB), and the sidecar carries the **numerical gradient** on top of the 24 arrays `--gradgold` writes, so it takes two commands to regenerate (see "The numerical gradient" above) |
| `reference_data/geometries/acrolein.xyz`, `golden/acrolein_def2svp_cas65.nevpt2gold` | the **acrolein** half of the pair: planar Cs, CAS(6,5) = the oxygen lone pair + 2 π + 2 π*, def2-SVP, with SC **and** PC fields. 7.1 MB, committed |
| `golden/acrolein_def2svp_cas65.gradgold` | **not committed** (gitignored, 57 MB — the one file of the CASCI pair that is): acrolein's gradient sidecar. Regenerate it with the `--gradgold` command in "Regenerating the golden file" followed by the `nevpt2_numgrad.py` one for its numerical gradient; the golden beside it is unaffected |
| `golden/lif_def2svp_cas46_casscf.mogold`, `golden/lif_def2svp_cas46_casscf.nevpt2gold`, `golden/lif_def2svp_cas46_casscf.gradgold` | the **LiF CASSCF reference** (#123): the converged CASSCF orbitals (4.6 KB, through "The CASSCF polish"), the golden on them (0.20 MB, `reference_kind = casscf`, **SC only**: its PC metric is refused here too) and its 24-array gradient sidecar (0.92 MB, no `grad_num`: `nevpt2_numgrad.py --reference casscf` writes its numerical gradient to an `--npz` only). All three committed; `nevpt2_grad.py --reference casscf` finds the `.mogold` by name (see "LiF and acrolein on CASSCF orbitals" above) |
| `golden/acrolein_def2svp_cas65_casscf.mogold`, `golden/acrolein_def2svp_cas65_casscf.nevpt2gold` | the **acrolein CASSCF reference** (#123): the converged CASSCF orbitals (47 KB, through the polish) and the golden on them (7.1 MB, SC **and** PC), both committed |
| `golden/acrolein_def2svp_cas65_casscf.gradgold` | **not committed** (gitignored, 57 MB): the acrolein CASSCF reference's gradient sidecar. Regenerate it with the `--mo-from … --gradgold` command in "Regenerating the golden file"; its sha256 is in "LiF and acrolein on CASSCF orbitals" above |
| `reference_data/geometries/hsoh.xyz`, `golden/hsoh_def2svp_cas66_casscf.mogold`, `golden/hsoh_def2svp_cas66_casscf.nevpt2gold`, `golden/hsoh_def2svp_cas66_casscf.gradgold` | the **C1 case** (#213): HSOH at its own CASSCF(6,6)/def2-SVP minimum, no point-group symmetry, so every displacement keeps the group. CASSCF orbitals only: the `.mogold` (15 KB), the golden (1.46 MB, SC **and** PC) and its sidecar (8.6 MB), the one CASSCF sidecar that carries `nevpt2_numgrad.py --reference casscf`'s `grad_num`. All committed; `golden-repro.sh` regenerates all three byte for byte; three ctest entries (SC, PC, `--gradient-source`) (see "HSOH, the C1 case" above) |
| `golden/n2_ccpvdz_cas1414.nevpt2gold` | **not committed**; see "CAS(14,14)" above |
