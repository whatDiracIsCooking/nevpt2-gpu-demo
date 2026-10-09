# Reference data

The golden files are the oracle: every number this code prints is checked
against one of them, to 1e-7. This page says what is in them, how they are
made, why the generator is set up the way it is, and what each committed file
is for. The generator is `reference_data/generate_golden.py`, an offline
Python tool. Nothing in a build or a test runs it: its output is committed,
and no Python or PySCF runs at run time.

## What "CASSCF input" means here, and why the golden state is CASCI

A full CASSCF run is **not reproducible**. Its orbital optimiser converges to
differently-rotated solutions of the same energy from run to run, so a NEVPT2
energy built on a fresh CASSCF wanders by tens of millihartree between
identical invocations, which is useless as a golden value. A **CASCI**
diagonalisation on converged RHF orbitals is a plain linear-algebra step that
reproduces bit for bit, given the two settings in "Reproducibility" below. So
that is what `generate_golden.py` runs.

What the demos read back is what any CASSCF/CASCI package hands a downstream
NEVPT2 driver, however the orbitals were obtained: a CI vector, the cheap 1-
and 2-RDMs, and a set of MO-integral blocks. The generator canonicalises the
state exactly as `pyscf.mrpt.NEVPT.kernel` does, and stores PySCF's own
per-class SC-NEVPT2 answer alongside.

It never writes `dm3`, `f3ac` or `f3ca`. Producing those from the CI vector is
the expensive step the GPU build exists for, and storing them would let the
demo skip the part it is supposed to demonstrate.

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

# the 12-orbital files: CAS(10,12) is the PC case, CAS(12,12) is refused one
uv run python reference_data/generate_golden.py --ncas 12 --nelecas 10 --name n2_ccpvdz_cas1012
uv run python reference_data/generate_golden.py --ncas 12 --nelecas 12 --no-pc --name n2_ccpvdz_cas1212
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
classes non-zero.

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
| `reference_data/pyscf_contract_time.py` | the CPU side of the head-to-head: reads a golden file and times PySCF's `NEVPTcontract` (`make_dm123` + `_contract4pdm` ×2) on the *same frozen CI vector* the demo reads, with no CASCI re-solve (see [`performance.md`](performance.md), "Head-to-head vs PySCF") |
| `golden/n2_ccpvdz_cas1010.nevpt2gold` | the **default case**: CAS(10,10) on N2/cc-pVDZ |
| `golden/n2_ccpvdz_cas1212.nevpt2gold` | CAS(12,12) on the same molecule, the scaling case (see [`performance.md`](performance.md), "Scaling with active-space size"). `slow`-labelled; run with `--tiles 40`. Carries **no** PC fields: its PC metric is ungapped (see "PC fields" above) |
| `golden/n2_ccpvdz_cas1012.nevpt2gold` | the 12-orbital **PC** case: CAS(10,12) on the same molecule (both N 1s in the core, n_det = 627k), with SC and PC fields. Exists because CAS(12,12) cannot be a PC golden. Its SC fields pass `nevpt2_demo --tiles 40` (CUDA, \|ΔE_corr\| 4e-15); its ctest entry is the PC one, `nevpt2_cas1012_pc` (`--pc --tiles 40`, `slow`) |
| `golden/n2_ccpvdz_cas1010_df.nevpt2gold`, `golden/n2_ccpvdz_cas1212_df.nevpt2gold`, `golden/n2_ccpvtz_cas1010_df.nevpt2gold` | the **density-fitted** goldens (`--df`, aux basis `<basis>-jkfit`), the oracle for `nevpt2_df_demo`: PySCF's DF-NEVPT2 as the answer, plus the four three-index `B_*` blocks. A different reference from the conventional files (see "Density fitting" above). The cc-pVTZ one (nvirt = 48) exists for slab boundaries, not memory pressure |
| `golden/n2_ccpvdz_cas44.nevpt2gold`, `golden/n2_ccpvdz_cas88.nevpt2gold` and their `_df` twins | the **small** cases (see "The small cases" above) |
| `reference_data/geometries/salicylaldimine.xyz`, `reference_data/basis/6-31gss-rifit.nw`, `golden/salicylaldimine_631gss_cas88[_df].nevpt2gold` | the non-N2 case (see "Salicylaldimine CAS(8,8)" above). The two goldens are **gitignored** and regenerate byte-identically in ~2.5 min each; their four ctest entries register only once they exist |
| `golden/n2_ccpvdz_cas1414.nevpt2gold` | **not committed**; see "CAS(14,14)" above |
