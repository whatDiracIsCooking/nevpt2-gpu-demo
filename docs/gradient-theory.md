# Gradient theory: the NEVPT2 Lagrangian on two references

The analytic-gradient derivation for the two references a golden can carry
([`reference-data.md`](reference-data.md), "What 'CASSCF input' means here, and
why the golden state is CASCI"). Every golden's state is a CASCI
diagonalisation; what differs is the orbitals it runs on, and that is what
decides the Lagrangian:

| branch | orbitals | orbital constraint | orbital Z-vector solve |
|---|---|---|---|
| **CASCI** (`reference_kind = casci`) | converged RHF, then an active-space **selection** | (O1): the RHF Fock block-diagonal over core/active/virtual (§1.2) | CPHF, decoupled from the CI equation (§2.2–§2.6) |
| **CASSCF** (`reference_kind = casscf`) | a converged `mcscf.CASSCF` | CASSCF stationarity, `A − A† = 0` — Park's Eq. 23 as written (§1.4) | CP-CASSCF, one system coupled to the CI equation (§2.7) |

Park's theory (Ref. [1], below) is written for the second. The first is not
the paper's, and its constraint set, Z-vector equations and solve order are
derived here.

**Most of the derivation does not depend on the reference**, and is written
once:

| part | shared, or per reference |
|---|---|
| §0 — notation, the energy as `E(T, C, c)`, the `P`/`Q`/`R` chain rule | shared |
| §1.1 — Park's Eq. 23, and which of its lines depends on the reference | shared framing |
| §1.2 — (O1), the selection, the RHF partition | **CASCI** |
| §1.3 — the active–active block is a gauge | shared |
| §1.4 — Eq. 23 restored | **CASSCF** |
| §2.1 — the source terms `Y` and `y` | shared |
| §2.2–§2.6 — decoupled CI equation, selection multipliers, CPHF, solve order, `act_O`/`act_V` recovery | **CASCI** (§2.3's `z^c` half is shared) |
| §2.7 — the coupled CP-CASSCF system and its solve order | **CASSCF** |
| §3 — the integral inventory Eq. 28 reads | shared (§3.5–§3.6 mark their CASCI-only rows) |
| §3.8 — the same inventory in three-index, density-fitted form | shared |
| §4.1–§4.2 — the active-space-selection hazard | **CASCI** |
| §4.3 — the in-plane-`Cs` displacement rule | shared |
| §4.4 — what is not that hazard, Park's SC non-invariance included | shared |
| §4.5 — first-order related work, and why the Hessian's two refusals do not reach the gradient | shared |
| §5 — what this settles | one column per reference |

**Both branches' Z-vector solves are offline Python only**
([`gradients.md`](gradients.md), §1): the CASCI branch's, and the CASSCF
branch's coupled system (`nevpt2_grad.py --reference casscf`). Every
derivative path on the GPU refuses a `casscf` golden (§1.1).

This document is **derivation only**. It describes no code and names no file
that does not yet exist; what gets built, and how it is checked, belongs in the
implementation and testing documents — [`gradients.md`](gradients.md) is the
code side of this one, and says which parts of §1–§4 are implemented and
which are not. It quotes **no measured numbers**: the
array and file sizes the inventory in §3 implies are measurements of a
particular case and belong in [`reference-data.md`](reference-data.md), the
finite-difference tolerances in [`testing.md`](testing.md), the costs in
[`performance.md`](performance.md). What appears here instead are *size
formulas* in `n_mo`, `n_core`, `n_act`, `n_virt` and, density-fitted,
`naux`, which are algebra.

It settles three things the rest of the gradient work rests on:

1. **§1–§2** — the explicit constraints and Z-vector equations for each
   reference. On RHF orbitals the CI and orbital Z-vector equations
   **decouple** (the coupling that remains runs one way, through the
   pseudocanonical multipliers), and the orbital solve is ordinary **CPHF**.
   On CASSCF orbitals they do not: one **CP-CASSCF** system, with no
   selection to differentiate across.
2. **§3** — the inventory of MO-integral arrays Park's Eq. 28 actually reads,
   the same for both. Seventeen arrays, each with exactly one index over the
   full MO range. The one array of the eighteen that is *not* needed is the
   expensive one. Density-fitted (§3.8), `κ` never reaches the fitting
   metric, and `Y` reads the whole three-index `(L|pq)`: the `--df` goldens
   carry four of its six space blocks, and lack `(L|cc)` and `(L|vv)`.
3. **§4** — the CASCI branch's active-space-selection hazard, and the settled
   rule for numerical validation, which both branches share: **in-plane
   displacements of a planar `Cs` molecule only**.

## 0. Notation

Park's index convention, the same one Angeli's spinless formulation uses:
`i, j, k` label
**core** (inactive, doubly occupied) orbitals, `a, b, c, d, e` label **active**
orbitals, `r, s` label **virtual** orbitals, and `x, y, z, w` label a **general**
MO — any of the three. Counts are `n_core`, `n_act`, `n_virt` and
`n_mo = n_core + n_act + n_virt`. `n_occ = n_core + n_act` is the union of the
two spaces the zeroth-order density lives on, and it is the range that keeps
recurring below.

On the CASCI branch two *different* partitions of the MO space are in play,
and keeping them apart is most of the work in §1.2 (the CASSCF branch has only
the first; §1.4):

| partition | spaces | set by |
|---|---|---|
| the NEVPT2 partition | core / active / virtual | the active-space **selection** |
| the RHF partition | occupied `O` / virtual `V` | the RHF **solution** |

They are not nested the same way. `core` is a subset of `O` and `virtual` is a
subset of `V`, but the active space straddles: write `act_O = act ∩ O` and
`act_V = act ∩ V`. For a π active space picked by irrep, both halves are
non-empty by construction (`reference-data.md`, "Salicylaldimine CAS(8,8)":
occupied π plus virtual π*).

Orbital rotations are parameterised the usual way: `C(κ) = C exp(κ)` with `κ`
real antisymmetric, so `δC_p = Σ_x C_x κ_xp` and

```
  δ h_pq     = Σ_x [ κ_xp h_xq + κ_xq h_px ]
  δ (pq|rs)  = Σ_x [ κ_xp (xq|rs) + κ_xq (px|rs)
                   + κ_xr (pq|xs) + κ_xs (pq|rx) ]
```

in chemists' notation. `F` is the **RHF Fock matrix** (built from the RHF
density, i.e. from `O` alone; the CASCI branch's constraint, and nowhere in
the CASSCF branch), `f` is the **generalized Fock** (built from the
zeroth-order density `d⁽⁰⁾`: 2 on the core diagonal, `γ⁽¹⁾` on the active
block, 0 on the virtual), and `ε_p = f_pp`. The golden's `e_core` / `e_virt`
arrays are `f`'s core and virtual diagonals.

### The energy, as a function of three things

Park Eq. 21: `E = E(T, C, c)` — amplitudes, orbital coefficients, CI
coefficients. The amplitude dependence drops out of the nuclear gradient
because the Hylleraas functional is stationary in `T` (Eq. 22's second term
vanishes), which is as true for a CASCI reference as for a CASSCF one. What
remains is the `C` and `c` dependence, and that is what a Lagrangian has to
handle.

For the strongly-contracted energy the `C` and `c` dependence enters through
three scalars per external tuple `t`, which are exactly the three this tree
already computes ([`implementation.md`](implementation.md), §3):

```
  N_t     the perturber norm            x_t^T S x_t
  H_t     the Dyall matrix element      x_t^T K x_t
  Delta_t the tuple's orbital-energy sum
  E_class = - sum_t N_t / (Delta_t + H_t/N_t)
```

Park's SC Lagrangian (his Eq. 31) introduces one multiplier per scalar —
`P_l` for `N`, `Q_l` for `H`, `R_l` for `Delta` — so that the active-space work
is done once and shared by every external tuple. Those multipliers are nothing
but the three partial derivatives of the finish above. With

```
  T_t = -1 / (Delta_t + H_t/N_t)        the amplitude, so that E_t = T_t N_t
```

differentiating `E_t = -N_t/(Delta_t + H_t/N_t)` gives

```
  P_t = dE/dN_t     = 2 T_t + Delta_t T_t^2     ( = -(Delta_t + 2 H_t/N_t) T_t^2 )
  Q_t = dE/dH_t     = T_t^2
  R_t = dE/dDelta_t = N_t T_t^2
```

which are Park's Eqs. 36, 37 and 38. Nothing in that step is specific to the
reference: it is the chain rule through the scalar finish, and it is the only
place in the whole derivation where real calculus (rather than bookkeeping)
happens on the energy side. Everything downstream — the pseudodensities of
Eqs. 41–47, the `Y_xy` of Eq. 28, the CI derivative of Eq. 48 — is `P`, `Q` and
`R` contracted back against integrals and densities.

Two consequences worth stating early:

- **`R_t` is the multiplier of an orbital-energy sum, so it produces a
  pseudodensity that is diagonal in the core and virtual spaces** (Park
  Eq. 39: `d^Fock_rr + d^Fock_ss = R_rs`). The energy functional reads `f`
  only on its diagonal there. That is precisely why the pseudocanonical
  condition has to appear as an explicit constraint: `f`'s *off-diagonal*
  response is unconstrained by the energy and must be pinned by the orbital
  convention instead. §2 picks this up.
- **`Delta_t` is the only route by which orbital energies enter.** Together
  with the core dressing of `h^eff`, it is what pulls integrals the energy
  never reads in any form into the gradient's inventory. §3 picks this up.

## 1. The constraint set

### 1.1 What Park's Eq. 23 assumes

Park's Lagrangian is

```
  L = E + (1/2) tr[ Z^dag (A - A^dag) ]          CASSCF orbital stationarity
        + (1/2) tr[ X (C^dag S C - 1) ]          orthonormality
        + sum_M z_M^dag (H - E) c_M              CI eigenvector
        - (1/2) sum_M x_M (c_M^dag c_M - 1)      CI normalisation
        + sum_{i<j} z^c_ij f_ij
        + sum_{r<s} z^c_rs f_rs                  pseudocanonicality
```

(his Eq. 23; `A - A^dag = 0` is the CASSCF orbital-gradient condition).

**The first line is the only one that depends on the reference.** The other
five — orthonormality, the CI eigenvector and its normalisation, and the two
pseudocanonical blocks — hold for both branches and carry the same
multipliers in both. The first line is a statement about how the orbitals were
made:

- **On CASSCF orbitals it holds**, and Eq. 23 is the Lagrangian as written
  (§1.4).
- **On RHF orbitals it does not.** A CASCI state is not stationary with
  respect to orbital rotation: `A - A^dag` is not zero at those orbitals, and
  there is no sense in which it could be, since no orbital optimiser was run.
  What makes the core/active/virtual partition hold still is RHF stationarity
  plus a *selection*, and the first line has to be rewritten as that (§1.2).

Which branch a golden is on is recorded, not inferred: its header carries the
reference that built its orbitals ([`reference-data.md`](reference-data.md),
"The file format"). Neither branch is a reproducibility choice — a fresh
`mcscf.CASSCF` reproduces at a fixed geometry under the generator's settings
([`reference-data.md`](reference-data.md), "A fresh CASSCF reproduces,
measured"). What the two differ in is the constraint set, and so the
Z-vector equations.

**In the tree only the CASCI branch is implemented, and the restriction is
enforced rather than assumed.** What the tree computes is source terms and
the energy's parameter Hessian, derivatives at fixed orbitals and CI vector
that carry no reference-kind dependence, so those paths sit behind
`nevpt2.golden`'s source-term gate and take a `casscf` golden. The Lagrangian
gate, `requireLagrangianReference`, refuses one as an unsupported-request
error and cites this section; no path in `src/` has a Lagrangian to put
behind it yet. `reference_data/nevpt2_numgrad.py`'s CASCI route refuses one
before it computes an energy (its `--reference casscf` route re-optimises
every displaced state instead). The gradient sidecars written for one
(`--mo-from --gradgold`, [`reference-data.md`](reference-data.md), "The
CASSCF-reference gradient sidecar") are read by the source-term path and the
orbital Hessian flags, never by a Lagrangian. The offline analytic gradient
(`reference_data/nevpt2_grad.py`) builds the CASCI branch on RHF orbitals by
default and, with `--reference casscf`, the CASSCF one — §1.4 and §2.7's
coupled CP-CASSCF system (§5); its numerical gradient re-optimises every
displaced state, under a guard for the precondition §4.4 names.

### 1.2 The CASCI branch: what is true on RHF orbitals

A `casci` golden's orbitals come out of a fixed four-step chain: converged RHF, then
an active-space **selection** among the RHF canonical orbitals (by energy
order, or by irrep with `--cas-irreps` / `--core-irreps`), then the CASCI
diagonalisation, then `canonicalize(..., cas_natorb=True)`, which rotates
*within* core and *within* virtual to diagonalise the generalized Fock `f` and
*within* the active space to natural orbitals, carrying the CI vector along.

Each of those three rotations is internal to one space, so **core, active and
virtual are each spanned by a fixed set of RHF canonical orbitals** even after
canonicalisation. Since `F` is diagonal in the RHF canonical basis, that single
observation gives the whole orbital constraint set in one line:

> **(O1) `F` is block-diagonal over the core/active/virtual partition.**
> `F_ai = 0`, `F_ri = 0`, `F_ra = 0`.

Equivalently: core and virtual are invariant subspaces of the RHF Fock
operator. This is gauge-covariant — it survives any rotation internal to a
space, the active natural-orbital rotation included — which the more familiar
spelling `F_{V,O} = 0` is not, because an active rotation mixes `act_O` into
`act_V` and so moves `O` itself.

(O1) contains two physically different statements, and separating them is what
§4 is about:

| block of (O1) | what makes it true | kind of statement |
|---|---|---|
| `F` between `O` and `V`: `core`–`act_V`, `core`–`virtual`, `act_O`–`act_V`, `act_O`–`virtual` | RHF stationarity (Brillouin) | a **convergence condition** |
| `F` within `O`: `core`–`act_O`. `F` within `V`: `act_V`–`virtual` | the selection — core and virtual are unions of RHF eigenvectors | a **selection**, not a stationarity condition |

RHF is invariant to rotations inside `O` and inside `V`, so no stationarity
condition constrains those two blocks. They hold because of how the active
space was picked. That is the hazard of §4, and it has no counterpart in
Park's Eq. 23, where CASSCF stationarity covers core–active and active–virtual
alike.

The remaining constraints carry over from Eq. 23 unchanged, and are the
CASSCF branch's too (§1.4):

> **(O2) Pseudocanonicality.** `f_ij = 0` for `i ≠ j` in core, `f_rs = 0` for
> `r ≠ s` in virtual. This is what `canonicalize` establishes and what the
> energy functional assumes when it reads only `e_core` and `e_virt`
> (§0's note on `R_t`). Park's multipliers `z^c` and his Eqs. 56–57 apply
> verbatim.
>
> **(C1) CI eigenvector.** `sum_J (H_IJ - E_CASCI δ_IJ) c_J = 0`, with
> multiplier `z_I`, plus `c^dag c = 1` with multiplier `x`. One state, so one
> `z` and one `x`. (`E_CASCI` is the eigenvalue of the active Hamiltonian on
> whichever orbitals the branch has; on CASSCF orbitals it is `E_CASSCF`.)
>
> **(N1) Orthonormality.** `C^dag S C = 1`, multiplier `X`. Automatically
> satisfied by the `exp(κ)` parameterisation; its multiplier is the
> energy-weighted density the AO contraction needs at the end.

So the CASCI branch's Lagrangian is

```
  L = E + sum_{x in act+virt, i in core} Z_xi F_xi
        + sum_{r in virt, a in act}      Z_ra F_ra
        + (1/2) tr[ X (C^dag S C - 1) ]
        + z^dag (H - E_CASCI) c - (1/2) x (c^dag c - 1)
        + sum_{i<j} z^c_ij f_ij + sum_{r<s} z^c_rs f_rs
```

— the same length as Eq. 23 but structurally simpler, because the orbital
constraint is a *linear* condition on a Fock matrix that **does not depend on
the CI vector**, where `A - A^dag` does. §2.2 is what that buys.

### 1.3 The active–active block is a gauge, not a constraint

No multiplier appears for a rotation among active orbitals, and none is needed.
The active space is treated by a **full** CI, so rotating the active orbitals
and transforming the CI vector accordingly leaves the CASCI state, and hence
every NEVPT2 class energy, unchanged. The active–active block of `κ` is
therefore redundant with the CI parameters: if `∂L/∂c_I = 0` holds, then
`∂L/∂κ_{aa'} = 0` follows by the chain rule along the invariance direction.
This is the same redundancy CASSCF has, arrived at without CASSCF — so it
holds on both branches, and §1.4 adds nothing to it.

Two things follow that are easy to get wrong:

- The natural-orbital convention `cas_natorb=True` picks a representative of
  that gauge. It is **not** a constraint, it needs no multiplier, and a
  reordering of the natural orbitals between two displaced geometries does
  **not** move the energy. Degenerate active occupation numbers are not a
  hazard here (contrast §4, where the core/active boundary is).
- The active–active block of the source term `Y` is nevertheless *non-zero*,
  and it is the natural first thing to validate, because the invariance makes
  it a closed identity: `Y` restricted to an active rotation is exactly
  cancelled by the CI source term contracted with the induced CI rotation. No
  Z-vector solve is involved. What it does **not** need is new densities — but
  it does need two integral arrays the energy never reads, because `κ_{aa'}`
  also moves the active density inside `h^eff` and the orbital energies. They
  are `(xa|cc)` and `(xa|vv)` in §3.3's table, restricted to `x` active: the
  `(a'a|jj)` and `(a'a|rr)` pairings, which no class energy touches. Assuming
  "the active block needs only what the golden already carries" is the easiest
  mistake to make here.

### 1.4 The CASSCF branch: Eq. 23 restored

A `casscf` golden's orbitals come out of a converged `mcscf.CASSCF`, committed
once to an MO sidecar ([`reference-data.md`](reference-data.md), "The MO
sidecar: committed CASSCF orbitals"); the same deterministic CASCI and the
same `canonicalize(..., cas_natorb=True)` then run on them. Canonicalisation
rotates within core, within active and within virtual, and the CASSCF energy
is invariant to all three, so its orbital gradient transforms covariantly and
**stays zero**: the canonicalised orbitals are as stationary as the converged
ones.

So the first line of Eq. 23 holds, and the Lagrangian is Park's, verbatim:

```
  L = E + (1/2) tr[ Z^dag (A - A^dag) ]          (S1) CASSCF orbital stationarity
        + (1/2) tr[ X (C^dag S C - 1) ]          (N1)
        + sum_M z_M^dag (H - E) c_M              (C1)
        - (1/2) sum_M x_M (c_M^dag c_M - 1)      (C1)
        + sum_{i<j} z^c_ij f_ij
        + sum_{r<s} z^c_rs f_rs                  (O2)
```

with one state, so one `z` and one `x`. `A` is the CASSCF orbital gradient,
up to the factor-of-two convention:

```
  A_xp = sum_q h_xq d_pq + sum_{qrs} (xq|rs) D_pqrs      p in core + active
  A_xr = 0                                               r in virtual
```

with `d` and `D` the reference's one- and two-particle densities over core
and active (closed-shell on the core). Unlike `F`, both are built from the
CI vector.

**(S1) carries multipliers only on its non-redundant blocks** — core–active,
core–virtual and active–virtual — `n_core·n_act + n_core·n_virt +
n_act·n_virt` of them. The other three blocks are redundant for the CASSCF
energy and need none from (S1):

- **core–core and virtual–virtual** are not redundant for the *NEVPT2* energy,
  which is not invariant there; (O2) pins them, with the same `z^c` as the
  CASCI branch.
- **active–active** is §1.3's gauge, unchanged.

**What the branch does not have** is as important as what it has:

- **No (O1) and no RHF partition.** `F` appears nowhere in the constraint
  set. Core–active and active–virtual are covered by stationarity rather than
  by a selection, so there are no selection-boundary multipliers (§2.3's
  Sylvester half), no `act_O`/`act_V` split, and §2.6's recovery of it is
  unnecessary rather than merely inapplicable — there is no RHF partition for
  it to recover.
- **No selection hazard.** Stationarity is a smooth condition: where the
  CASSCF orbital Hessian is non-singular, the stationary orbitals are a
  differentiable function of the geometry, and the kink §4.1 describes cannot
  occur. What takes its place is §2.7's conditioning and §4.4's
  solution-branch precondition.

**Stationarity is assumed exact.** Eq. 23 assumes `A - A^dag = 0`; a
converged optimiser stops at a residual orbital gradient, and the gradient
this Lagrangian delivers is off by a term of first order in that residual.
The CASCI branch makes the same assumption of its own optimiser — (O1)'s
`O`–`V` blocks are RHF Brillouin — so this is not a new kind of error, only a
different optimiser's residual. What the generator accepts from the CASSCF
one, and what a residual costs the energy, are measured in
[`reference-data.md`](reference-data.md), "The `--casscf` / `--mo-from` gate
thresholds".

## 2. The Z-vector equations

### 2.1 Source terms

Park Eqs. 26–27:

```
  Y_xy = dE/dkappa_xy        the orbital source term, a full n_mo x n_mo matrix
  y_I  = dE/dc_I             the CI source term
```

Both are derivatives of the energy at **fixed** constraints — pure
differentiation, no solve. The whole point of the Lagrangian is that these are
the only derivatives of `E` anyone has to take; the geometry dependence is
carried by the multipliers and by derivative integrals.

Both are the same on the two branches: they differentiate the NEVPT2 energy,
which reads the orbitals and the CI vector and does not know how either was
made. Only what consumes them differs.

`Y` is needed over the **full** `n_mo × n_mo` range. Every block has a
consumer: core–core and virtual–virtual feed (O2)'s multipliers on both
branches, active–active feeds §1.3's consistency check on both, and the three
mixed blocks feed the orbital Z-vector equation — on the CASCI branch
core–active and active–virtual through (O1)'s selection multipliers and
core–virtual through CPHF's right-hand side, on the CASSCF branch all three
through CP-CASSCF's right-hand side. There is no block to skip.

### 2.2 The CASCI branch: the CI Z-vector equation decouples, and is solved before the orbital one

Differentiate `L` with respect to `c_I`:

```
  dL/dc_I = y_I + sum_J (H_IJ - E_CASCI δ_IJ) z_J - x c_I
            + sum_{xy} Z_xy dF_xy/dc_I
            + sum_{i<j} z^c_ij df_ij/dc_I + sum_{r<s} z^c_rs df_rs/dc_I
          = 0
```

**The `Z` term is identically zero.** `F` is built from the RHF density, which
is determined by the RHF occupied space alone; it has no CI-coefficient
dependence whatsoever. (In Park's Eq. 23 the corresponding term is
`Z^dag ∂(A - A^dag)/∂c`, and it does *not* vanish: the CASSCF orbital gradient
is built from the CASSCF 1- and 2-RDMs, hence from `c`. That coupling is what
makes CP-CASSCF one big linear system; §2.7.)

**The `z^c` term is not zero.** (O2) constrains the *generalized* Fock, which
is built from the active density and so depends on `c`:

```
  df_pq/dc_I = sum_tu (dgamma_tu/dc_I) [ (pq|tu) - (1/2)(pt|uq) ]
  dgamma_tu/dc_I = [ (E_tu + E_ut) c ]_I
```

But it couples in one direction only. The `z^c` are divisions of `Y` alone
(§2.3: no other constraint responds to a core–core or virtual–virtual
rotation), so they are known before this equation is written, and their term
is a right-hand side, not a coupling:

```
  y^c_I = sum_tu [ (E_tu + E_ut) c ]_I W_tu
  W_tu  = sum_{p<q, same inactive space} z^c_pq [ (pq|tu) - (1/2)(pt|uq) ]
```

a one-body sigma vector of one multiplier-weighted `n_act × n_act` matrix. Its
integrals are `(cc|aa)`, `(ca|ac)`, `(vv|aa)` and `(va|av)` slices of §3.3's
inventory.

So the CI equation is one projected solve:

```
  P (H - E_CASCI) P z = - P (y + y^c),        P = 1 - |c><c|
  x = c^dag (y + y^c)
```

a single linear solve in the determinant space, with the reference state
projected out (it is the null direction of `H - E_CASCI`, and `x` absorbs the
component along it). `E_CASCI` is the CASCI eigenvalue, so this is an
ordinary CASCI Z-vector equation and nothing about NEVPT2 enters it except
through `y` and `y^c`.

**This is the main structural simplification of a CASCI reference**, and it
sets the solve order: the system is block-triangular, `z^c`, then `z`, then
the orbital multipliers, and `z` is known before the orbital equation is
touched.

### 2.3 The pseudocanonical and selection multipliers: no iteration

The pseudocanonical half of this section — `z^c`, Eqs. 56–57, and the
degeneracy refusal guarding their divisions — is **shared**: the CASSCF
branch uses it verbatim (§2.7), since on both branches no constraint but (O2)
responds to a core–core or virtual–virtual rotation. The selection half — the
Sylvester equations — is the **CASCI branch** only.

Now differentiate with respect to `κ` on the blocks that **do not change the
RHF density**: core–core, virtual–virtual, `core`–`act_O`, and
`act_V`–`virtual`. A rotation inside `O`, or inside `V`, leaves the AO-basis
RHF density — and so the AO-basis `F` — untouched, so its only effect is on
the MO representation. For any Fock-like matrix `M` and any `κ`,

```
  delta M_pq = sum_x [ kappa_xp M_xq + kappa_xq M_px ]
```

and if `M` happens to be **diagonal** on both spaces the block connects, this
collapses to `δM_pq = κ_qp (M_qq - M_pp)`: one multiplier, one orbital-energy
difference, a division.

**For (O2) that is exactly the situation**, because `canonicalize` diagonalised
`f` within core and within virtual. So Park's Eqs. 56–57 apply verbatim, in
the golden's own gauge:

```
  z^c_ij = (1/2) (Y_ij - Y_ji) / (eps_i - eps_j)      i, j in core,    i != j
  z^c_rs = (1/2) (Y_rs - Y_sr) / (eps_r - eps_s)      r, s in virtual, r != s
```

**For (O1)'s selection boundaries it is not**, and this is the one place where
transcribing Park's form would be wrong. The matrix there is `F`, not `f`, and
`F` is *not* diagonal within core (canonicalisation diagonalised `f` there
instead) nor within the active space (natural orbitals). What the stationarity
condition gives is therefore a small **Sylvester equation** per block:

```
  F^core Z - Z F^actO = S_ip        i in core,    p in act_O
  F^virt Z - Z F^actV = S_rp        r in virtual, p in act_V
```

where `S` is the block's source: `Y_ip - Y_pi` (or `Y_rp - Y_pr`), plus the
CI-relaxation and `z^c` terms §2.4's right-hand side carries, restricted to
this block. Unlike a core–core rotation, a core–active or active–virtual one
moves the active orbitals and the core density, so it moves the active
Hamiltonian and `f`; both of those multipliers are known by now (§2.2), so
the terms are numbers, not unknowns.

This is still not an iteration. Diagonalise `F` inside each of the four
spaces — `n_core`, `n_act`, `n_virt` sized symmetric eigenproblems, cheap, and
the active one is the diagonalisation §2.6 needs anyway — and in that basis
every Sylvester equation is again a plain division by `ε^RHF_p - ε^RHF_i`.
Equivalently: rotate `Y` into the **RHF canonical gauge**, where `F` is fully
diagonal, divide, and rotate the multipliers back. Do *not* try to impose (O1)
and (O2) in the same diagonal basis — no single gauge diagonalises both `F` and
`f`, which is the whole reason there are two constraints rather than one.

All of these are computed **before** the Z-vector solve, exactly as Park's
"Finalizing Gradient Evaluation" prescribes, and folded into its right-hand
side: they multiply constraints that *do* respond to the density-changing
`O`–`V` rotation, so they contribute there even though their own equations
were not iterative.

Every one of these divisions is singular where its divisor vanishes. For (O2)
that is the core or virtual degeneracy the golden generator already refuses
([`reference-data.md`](reference-data.md), "The degeneracy refusal") — and the
gradient needs that guard *more* than the energy does, since the energy's
sensitivity to a same-irrep rotation is bounded while the multiplier goes as
`1/Δε`. For the selection boundaries it is §4's hazard.

### 2.4 The CPHF equation (CASCI branch)

What is left is the density-changing part: `κ` between `O` and `V`, which for
this partition means `core`–`act_V`, `core`–`virtual` and `act_O`–`virtual`.
Note what is *not* in that list: `act_O`–`act_V` is an `O`–`V` rotation too,
but it is internal to the active space, so it is §1.3's gauge rather than a
constraint. The system below is therefore the ordinary CPHF one with the
`act_O × act_V` rows **and** columns struck out — symmetrically, since those
equations are the redundant ones and `Z` has no component there.

On what is left, constraint (O1) is the RHF Brillouin condition, its derivative
is the RHF orbital Hessian, and the equation is the ordinary **CPHF** equation:

```
  sum_{sj} A_{ri,sj} Z_sj = - [ Y_ri - Y_ir
                                + (CI relaxation: z^dag (dH/dkappa_ri) c + h.c.)
                                + (the z^c and selection multipliers of 2.3) ]
```

where, in this one block, `i` and `j` stand for any `O` orbital and `r` and
`s` for any `V` orbital, and `A_{ri,sj} = ∂F_ri/∂κ_sj` is the closed-shell RHF
orbital Hessian — the same
operator an MP2 or a CASCI gradient needs, with no multireference content of
its own. The CI-relaxation term is the orbital derivative of the CI constraint
contracted with the `z` already in hand; written out, it is a CASCI-style
orbital gradient built from the symmetrised **transition** density between the
reference and `|z>`, `d^z_pq = <z|E_pq|Psi> + <Psi|E_pq|z>`, plus its
two-particle partner. It needs no integral block beyond §3's inventory.

### 2.5 Solve order (CASCI branch)

```
  1.  Y_xy, y_I, z^c                 source terms, differentiation only; and
                                     Eqs. 56-57, divisions of Y (2.3).
  2.  P (H - E) P z = -P (y + y^c)   CI Z-vector. Decoupled from Z (2.2).
  3.  Z_ip, Z_rp                     two small Sylvester equations; no
                                     iteration (2.3).
  4.  A Z = -rhs                     CPHF on the O-V block, less its
                                     active-internal part (2.4). The only
                                     iterative step in the whole procedure.
  5.  relaxed densities, X           then contract against derivative integrals
                                     in the AO basis.
```

Compared with the CASSCF branch (§2.7) this is one coupled system fewer:
steps 2 and 4 are separate solves here, where CP-CASSCF merges them into one.
Everything specific to NEVPT2 is in step 1.

### 2.6 Recovering the RHF partition from a canonicalized state (CASCI branch)

This section exists only because the CASCI branch has an RHF partition. The
CASSCF branch has none, so there is nothing to recover (§1.4, §2.7).

Steps 3 and 4 need to know which active orbitals came from `O` and which from
`V`, and the natural-orbital rotation has mixed them (§0). It is recoverable
exactly, with one small diagonalisation: because the active space is a union of
RHF canonical orbitals (§1.2), `F` restricted to the active block has the
*selected RHF orbital energies* as its eigenvalues and the pre-canonicalisation
active orbitals as its eigenvectors. Diagonalising the `n_act × n_act` matrix
`F_ab` therefore undoes the natural-orbital gauge and splits the active space
into `act_O` and `act_V` by which side of the RHF gap each eigenvalue falls.
It is also the basis §2.3's active-side Sylvester equations want, so the one
diagonalisation serves both.

For that to be available at all, the reference data has to carry `F` in the
final MO basis. §3.5 lists it among what the sidecar must hold, and §3.6 turns
its block structure into a cheap self-check.

### 2.7 The CASSCF branch: one coupled CP-CASSCF system

Differentiate §1.4's Lagrangian with respect to `c_I`:

```
  dL/dc_I = y_I + y^c_I + sum_J (H_IJ - E δ_IJ) z_J - x c_I
            + (1/2) sum_{xy} Z_xy d(A - A^dag)_xy/dc_I
          = 0
```

`y^c` is §2.2's, unchanged. **The `Z` term is what §2.2 could drop and this
branch cannot:** `A` is built from the reference's 1- and 2-RDMs, so
`∂(A − A†)/∂c ≠ 0`, and the CI equation reads the orbital multipliers. The
orbital equation, on (S1)'s three non-redundant blocks, reads the CI ones in
turn — `z^dag (dH/dkappa) c + h.c.`, which on the CASCI branch was a known
term on CPHF's right-hand side, is now a product with an unknown. So §2.2's
decoupling fails, and **steps 2 and 4 of §2.5 merge into one system**:

```
  [ G_kk  G_kc ] [ Z ]       [ Y_xy - Y_yx + (z^c terms) ]
  [ G_ck  G_cc ] [ z ]  =  - [ P (y + y^c)               ]

  G_kk Z = (1/2) d(A - A^dag)/dkappa . Z     orbital-orbital
  G_kc z = z^dag (dH/dkappa) c + h.c.        orbital-CI
  G_ck Z = (1/2) Z . d(A - A^dag)/dc         CI-orbital
  G_cc   = P (H - E) P                       CI-CI
```

with `kappa` running over core–active, core–virtual and active–virtual only
and the CI space projected against `c`: dimension `n_core·n_act +
n_core·n_virt + n_act·n_virt` plus the determinant count less one. The two
off-diagonal blocks are each other's transpose up to normalisation, since
both are mixed second derivatives of the CASSCF energy; the whole matrix is
the CASSCF electronic Hessian at the converged point — the operator a
second-order CASSCF optimiser applies — solved once, against an NEVPT2
right-hand side. Like CPHF's, it is applied from the AO integrals and
`mo_coeff`; §3's inventory sizes `Y`, on both branches, not either Hessian.

**What it costs.** One linear system over the union of the two spaces, where
the CASCI branch has two smaller ones solved in turn, and an operator with
multireference content of its own (`A` reads `D`). Its conditioning is the
CASSCF Hessian's: a near-singular Hessian — a nearly redundant rotation, or a
second root close to the reference — makes the multipliers large and
inaccurate. That is this branch's analogue of §4.2's small selection gap, and
it is the same condition (§1.4) under which the stationary orbitals stop
being a smooth function of the geometry.

**What it buys.** No selection-boundary multipliers: §2.3's Sylvester
equations and §2.6's recovery are gone, the latter unnecessary rather than
merely inapplicable. No RHF Fock and no RHF electron count anywhere (§3.5).
And none of §4.1–§4.2's hazard class, because core–active and active–virtual
are covered by stationarity rather than by a selection.

**What carries over untouched.** The source terms `Y` and `y` (§2.1); `z^c`,
Eqs. 56–57, and the degeneracy refusal guarding their divisions (§2.3); `y^c`
(§2.2); the active–active gauge (§1.3); the whole §3 inventory.

Solve order (CASSCF branch):

```
  1.    Y_xy, y_I, z^c               as §2.5's step 1, unchanged.
  2+4.  G [Z; z] = -rhs              the coupled CP-CASSCF system above. The
                                     only linear system in the procedure.
  3.    (nothing)                    no selection multipliers.
  5.    relaxed densities, X         Z's contribution goes through A's
                                     response, not F's; then the same AO
                                     contraction.
```

## 3. The integral-block inventory for Eq. 28

This is the part that sizes the gradient sidecar. A dump of all MO integrals is
`n_mo^4` doubles, which is tolerable for a small case and not for a large one;
the committed cases' actual sizes are in
[`reference-data.md`](reference-data.md). The question is which arrays Park's
Eq. 28 reads, and the answer is a short, mechanical list.

**All of §3 is shared by the two branches.** Eq. 28 is `Y`, and `Y` is a
derivative of the energy functional, which does not know how its orbitals
were made (§2.1). The only rows that depend on the reference are two of
§3.5's and two of §3.6's, which exist for the CASCI branch's RHF partition and
are marked there.

### 3.1 The promotion rule

Eq. 28 writes the orbital source term as a sum of terms of the form
`(integral) x (pseudodensity)`:

```
  Y_xy = [ h d^(0) + h^eff d^eff + h^eff' d^eff' + h d + f^h d^h + f d^Fock
         + g(d^eff + d^eff' + d^eff'') d^(0)_core + g(d^Fock) d^(0)
         + g(d^h) d^c_act + gtilde((1/2) d^eff'' + d) d^act
         + sum_kl D_kl K_lk ]_xy
```

with `g` and `gtilde` the Coulomb-exchange and exchange-only contractions of
Eq. 29–30. Every term there is a **matrix product** closing on a pseudodensity,
and every pseudodensity has restricted support. The structural consequence is
the one thing to take from Eq. 28's shape:

> **Exactly one index of every integral in Eq. 28 runs over the full MO range.**

That is not an accident of how Park grouped the terms; it is forced by the
chain rule. From §0, `∂(pq|rs)/∂κ_xy` is non-zero only when `y` occupies one of
the four slots, and the promoted index `x` takes its place. So:

> **Promotion rule.** For every MO-integral array the energy reads, replace one
> index at a time by a general index `x`, keeping the charge-distribution
> pairing. The union of the results, over all four slots and all arrays, is
> exactly what `Y` needs.

Applying it mechanically is both the derivation and the inventory. Nothing has
to be read off Eq. 28 term by term.

One wrinkle, and it is where all the interesting sizing lives: the energy also
reads **finished** quantities that are not raw integrals — the orbital-energy
vectors `e_core` and `e_virt`, and the effective one-electron matrices.
Substituting their definitions puts them inside the rule's reach:

```
  h^eff_pq = h_pq + sum_{j in core} [ 2 (pq|jj) - (pj|jq) ]
  f_pq     = h^eff_pq + sum_{tu in act} gamma_tu [ (pq|tu) - (1/2)(pt|uq) ]
  eps_p    = f_pp
```

(`f^h` and the `h^eff` variants differ only in which density is contracted,
not in shape.) Two structural facts govern what that adds, and both *shrink*
what the rule would otherwise suggest:

- **The core density is diagonal**, so every integral the core dressing
  contributes already has its two core labels **tied** — `(pq|jj)` and
  `(pj|jq)`, never `(pq|jk)`. Promoting one slot leaves the tie, or the shared
  index, in place.
- **`Delta_t` reads `f` only on its diagonal** (`eps_i`, `eps_r`), so arrays
  arising from `∂f_pp/∂κ` carry the two `p` labels tied wherever the promoted
  index did not take one of them.

A tied pair costs a factor `n_core` or `n_virt` instead of its square, so
these rows are an order of magnitude smaller than their unrestricted shape.
Over-allocating them is the easiest way to get the sidecar wrong, so §3.3
marks every one.

### 3.2 What the energy reads

The nine MO-integral arrays, as `generate_golden.py` slices them and as
`src/energy/` consumes them. Space signatures are written as unordered
multisets, `c`/`a`/`v` for core/active/virtual:

| golden field | chemists' form | signature | class |
|---|---|---|---|
| `h2e` | `(ab\|cd)` | `aaaa` | the active Hamiltonian, every class |
| `h2e_v_Sr` | `(rb\|tu)` | `aaav` | `Sr` |
| `h2e_v_Si` | `(ai\|tu)` | `aaac` | `Si` |
| `cvcv` | `(ir\|js)` | `ccvv` | `Sijrs` |
| `h2e_v_Sijr` | `(jt\|ir)` | `ccav` | `Sijr` |
| `h2e_v_Srsi` | `(st\|ir)` | `cavv` | `Srsi` |
| `h2e_v_Srs` | `(rt\|su)` | `aavv` | `Srs` |
| `h2e_v_Sij` | `(it\|ju)` | `ccaa` | `Sij` |
| `h2e_v1_Sir`, `h2e_v2_Sir` | `(ri\|tu)`, `(rt\|iu)` | `caav` | `Sir` |

plus the one-electron and effective one-electron slices (`h1e`, `h1e_v_Sr`,
`h1e_v_Si`, `h1e_v_Sir`) and the two orbital-energy vectors.

Of the fifteen four-index space signatures, six are absent from the table:
`cccc`, `ccca`, `cccv`, `cvvv`, `avvv`, `vvvv`. The first three are not truly
absent — they are hidden inside `h^eff` and `f`, and §3.1's first structural
fact says they occur only with their core labels tied. The last three are
absent outright: **no integral this energy reads carries more than two virtual
labels.** That is the same structural fact the density-fitted path already
turns to account — no class puts two virtuals in one charge distribution, so
there is no `(L|vv)` block
([`reference-data.md`](reference-data.md), "Density fitting"). §3.4 is what it
is worth to the gradient, and §3.8 why the density-fitted gradient does not
keep the missing `(L|vv)`.

### 3.3 What the gradient needs

Promotion cannot add a virtual label, so the bound on virtuals survives. Write
`(xP|QR)` for the array in which the general index `x` shares a charge
distribution with a label from space `P` — `(xv|ca)` is `(x r | i a)`, and it is
a *different* array from `(xa|cv)` and `(xc|va)`, which hold the same four space
labels in a different pairing. There are eighteen such arrays. **Seventeen are
needed**, and this is the whole inventory:

| array | shape | restriction | differentiates |
|---|---|---|---|
| `(xa\|aa)` | `n_mo·n_act^3` | — | `aaaa`, `aaav`, `aaac` |
| `(xv\|aa)` | `n_mo·n_virt·n_act^2` | — | `aaav`, `caav`, `eps_r` |
| `(xa\|va)` | `n_mo·n_act^2·n_virt` | — | `aaav`, `aavv`, `caav`, `eps_*` |
| `(xc\|aa)` | `n_mo·n_core·n_act^2` | — | `aaac`, `caav`, `h^eff`, `eps_i` |
| `(xa\|ca)` | `n_mo·n_act^2·n_core` | — | `aaac`, `ccaa`, `caav`, `eps_*` |
| `(xv\|va)` | `n_mo·n_virt^2·n_act` | — | `aavv`, `cavv` |
| `(xc\|ca)` | `n_mo·n_core^2·n_act` | — | `ccaa`, `ccav`, `h^eff` |
| `(xa\|cv)` | `n_mo·n_act·n_core·n_virt` | — | `caav`, `ccav`, `cavv` |
| `(xv\|ca)` | `n_mo·n_virt·n_core·n_act` | — | `caav`, `ccav` |
| `(xc\|va)` | `n_mo·n_core·n_virt·n_act` | — | `caav`, `cavv`, `h^eff` |
| `(xc\|cv)` | `n_mo·n_core^2·n_virt` | — | `ccav`, `ccvv`, `h^eff` |
| `(xv\|cv)` | `n_mo·n_virt^2·n_core` | — | `cavv`, `ccvv` |
| `(xa\|cc)` | `n_mo·n_act·n_core^2` | — | `h^eff` (exchange), `eps_i` |
| `(xv\|cc)` | `n_mo·n_virt·n_core^2` | — | `h^eff` (exchange), `eps_r` |
| `(xc\|cc)` | `n_mo·n_core^2` | two of the three core labels **coincide** | `h^eff`, `eps_i` |
| `(xa\|vv)` | `n_mo·n_act·n_virt` | the two virtual labels **coincide** | `eps_r` |
| `(xc\|vv)` | `n_mo·n_core·n_virt` | the two virtual labels **coincide** | `eps_r` |

The "differentiates" column names which energy-side quantity each array is the
promotion of: a signature from §3.2's table, one of the effective one-electron
matrices, or an orbital energy. The last five rows are the ones the *energy*
never reads in any form — they exist only because the gradient has to
differentiate the core dressing of `h^eff` and the orbital energies, which the
energy consumes as finished numbers. They are also where §3.1's two structural
facts pay off:

- `(xc|cc)` would be `n_mo·n_core^3` unrestricted. It is not needed that way.
  Every route to it runs through `(pq|jj)` or `(pj|jq)` with two core labels
  already tied, so only slices with two of the three coinciding are ever
  touched: `n_mo·n_core^2`, a factor `n_core` cheaper.
- `(xa|vv)` and `(xc|vv)` come only from `∂eps_r/∂κ` in the form `(rr|x·)`, so
  the two virtual labels coincide: `n_mo·n_virt` rather than `n_mo·n_virt^2`,
  a factor `n_virt` cheaper. These are the two rows most worth getting right,
  because `n_virt` is the large count.

The class attribution matters in exactly one place. Most rows are reached by
several classes, and the `h^eff` / `eps` routes reach any class with the
corresponding external index, so "which class needs this" is rarely
restrictive. The exceptions are the three rows that carry two labels from the
same inactive space in *different* charge distributions, which only a class
with two external indices of that kind can produce:

- `(xv|cv)` in full comes from `Srsi` and `Sijrs` alone;
- `(xc|cv)` in full from `Sijr` and `Sijrs` alone;
- `(xv|va)` in full from `Srs` and `Srsi` alone.

"In full" is the load-bearing qualifier: `h^eff`'s and `eps_r`'s exchange
terms, `(rj|jr)` and friends, also want these three arrays, but only on slices
where the repeated index is shared between the two charge distributions —
`(xj|jr)`, not `(xi|jr)`. Those slices are a factor `n_core` or `n_virt`
cheaper and are reached by any class with a matching external index. The full
arrays are the expensive ones, and they are the restricted ones.
`(xv|cv)` and `(xv|va)` are the two largest rows in the inventory, and
`(xc|cv)` is among the next. §3.4 is what that buys.

### 3.4 The one array that is not needed, and what that buys

The eighteenth array is

```
  (xv|vv)        n_mo · n_virt^3
```

and it never appears: no integral the energy reads carries three virtual
labels (§3.2), and promotion cannot create one. It is also, by a wide margin,
the **largest** of the eighteen — of order `n_mo^4` once `n_virt` approaches
`n_mo`, which it does for any realistic basis. Its absence is the entire
difference between the inventory and a full dump.

The honest accounting:

- The inventory's own total is dominated by the two `n_mo · n_virt^2 · ...`
  rows, `(xv|cv)` and `(xv|va)`, with `(xc|cv)` and `(xv|cc)` behind them.
  Summed, that is `O(n_mo · n_virt^2 · n_occ)`.
- A full dump is `n_mo^4`, i.e. `O(n_mo · n_virt^3)`.
- The ratio is `O(n_occ / n_virt)`: **a constant factor, not a change of
  order.**

Two things follow, and both are decisions the rest of the gradient work should
take as settled:

1. **The inventory makes the small cases cheap; it does not make a large case
   cheap.** For a case with a hundred-plus AOs and a small active space the
   sidecar is still far larger than the golden it sits beside. Naming the
   blocks is what lets a reader allocate exactly, not what makes the big case
   affordable. The lever that actually works is choosing small molecules for
   gradient validation — which is independently why the validation set is LiF
   and acrolein rather than the largest committed case.
2. **The heaviest rows have an escape hatch, and it is a known trade.**
   `(xv|cv)` and `(xc|cv)` are wanted *in full* by `Sijrs`, `Sijr` and `Srsi`
   alone (§3.3), and
   `Sijrs` has no active index at all — its orbital-gradient contribution is an
   ordinary MP2-like term that a production implementation contracts in the
   **AO** basis and never stores as an MO array. Storing them is a *validation*
   choice: it keeps all of `Y` on one side of the seam, so a duality check can
   cover the full rotation space in one test. If the sidecar ever has to
   shrink, those rows are where to cut, at the cost of moving part of `Y` out
   of the checked path.

### 3.5 What the sidecar must carry besides integrals

Everything else Eq. 28 and §2 read is `n_mo × n_mo` or smaller, so it is free
by comparison, but it has to be *there*:

| quantity | shape | why |
|---|---|---|
| `mo_coeff` | `n_ao × n_mo` | the AO contraction at step 5 |
| `h` | `n_mo × n_mo` | `h d^(0)` and `h d`; the one-electron promotion |
| `f` (generalized Fock) | `n_mo × n_mo` | `f d^Fock`; its core/virtual diagonal is the golden's `e_core`/`e_virt` |
| `f^h` | `n_mo × n_mo` | `f^h d^h` — the Fock with the active space fully occupied |
| `h^eff`, `h^eff'`, `h^eff''` | `n_mo × n_mo` each | `h^eff d^eff` and friends; the golden carries only the restricted slices (`h1e`, `h1e_v_*`) |
| `F` (RHF Fock) | `n_mo × n_mo` | **CASCI branch only**: §2.4's CPHF operator, and §2.6's recovery of `act_O`/`act_V` |
| the RHF electron count | scalar | **CASCI branch only**: fixes which side of `F`'s active spectrum is occupied |

The CASSCF branch reads neither — it has no RHF partition (§1.4) — and adds
nothing to the list: `A` and its response are built from `mo_coeff`, the AO
integrals and the reference densities the energy path already has (§2.7).

The golden's existing one-electron fields are **restrictions** of these: the
sidecar's job on the one-electron side is to carry the same matrices over the
full MO range. The definitions of the three `h^eff` variants are Angeli's; see
[`references.md`](references.md).

### 3.6 Invariants a reader can check

Three properties of the above are exact by construction and cost nothing to
assert, which makes them the right acceptance checks for whatever writes the
sidecar:

- `F`'s core–active, core–virtual and active–virtual blocks are zero — that is
  (O1), and it is the single statement that the active-space selection is a set
  of RHF canonical orbitals.
- `f`'s core–core and virtual–virtual off-diagonals are zero — that is (O2),
  and its diagonals reproduce the golden's `e_core` and `e_virt`.
- Diagonalising `F`'s active block splits the active space into exactly
  `n_elec_rhf/2 - n_core` eigenvalues on the occupied side of the RHF gap and
  the rest above it, with no eigenvalue inside the gap. That is the recovery of
  `act_O` and `act_V` (§2.6) working; if it does not come out that way, the
  active space was not a set of RHF canonical orbitals.

The first and third are the **CASCI branch's**: on CASSCF orbitals `F` is
not block-diagonal and has no reason to be, and there is no RHF gap to split
across. The second is shared. The CASSCF branch's counterpart of the first is
(S1) itself — the residual orbital gradient — which the generator already
evaluates before it accepts CASSCF orbitals
([`reference-data.md`](reference-data.md), "The `--casscf` / `--mo-from` gate
thresholds").

A violation of the first two means the state is not the one this derivation
describes, and every multiplier downstream is meaningless. They are cheap, so
they should be checked rather than assumed — the same stance the PC gap check
takes ([`implementation.md`](implementation.md), §4.3).

### 3.7 Density orders

For completeness, since it bounds the other half of the cost: `Y` needs
`γ⁽¹⁾`, `γ⁽²⁾`, `γ⁽³⁾` and the **integral-contracted** 4-RDM digests this tree
already builds for the energy (`implementation.md`, §2.4) — Park's `K` matrix
(his Eq. 35) and his `D` (Eqs. 46–47) read nothing deeper. `y_I` needs the CI
derivatives of all of those (his Eqs. 48–49), which is the step he identifies
as the scaling bottleneck of the whole method, at `O(n_act^9)`. So the gradient
does **not** raise the density order the energy already pays for; it adds a
derivative of it.

### 3.8 The density-fitted inventory

The density-fitted energy ([`reference-data.md`](reference-data.md), "Density
fitting") reads three-index blocks instead of §3.2's four-index arrays, in the
symmetric-metric convention:

```
  (pq|rs) = sum_L B[L,p,q] B[L,r,s]
  B[L,p,q] = sum_M W_LM (M|pq),        W^T W = V^{-1},   V_LM = (L|M)
```

with the metric already absorbed into `B` (any factor `W` of `V^{-1}`;
`generate_golden.py`'s `_df_blocks` and `src/df_integrals/`'s module header).
`B` is symmetric in its two MO indices. The question this section answers is
the DF counterpart of §3.3's: **which blocks of `B` does `Y`, over its full
`n_mo × n_mo` range (§2.1), read** — against the four a `--df` golden
carries: `B_aa` `(L|aa)`, `B_ca` `(L|ca)`, `B_va` `(L|va)` and `B_cv`
`(L|cv)`, of extents `naux × n_act × n_act`, `naux × n_core × n_act`,
`naux × n_virt × n_act` and `naux × n_core × n_virt`. Every committed
`*_df.nevpt2gold` carries exactly those four, and no other `B_*` array.

**κ never reaches the fitting metric.** An orbital rotation `C(κ) = C exp(κ)`
transforms the MO indices of the three-centre integrals `(M|pq) = Σ_{μν}
(M|μν) C_μp C_νq` and nothing else: `(M|μν)` and `V_LM = (L|M)` are integrals
over AO and auxiliary functions alone, which sit at the nuclei and do not see
`C`. So `W` is a constant under `κ`, the auxiliary index is inert, and

```
  δ B[L,p,q] = sum_x [ κ_xp B[L,x,q] + κ_xq B[L,p,x] ]
```

— §0's one-electron rule, applied to each auxiliary slice. Substituted into
`(pq|rs) = Σ_L B B`, the product rule gives back §0's four-term rule for
`δ(pq|rs)` exactly. So §3.1's promotion rule carries over unchanged, and in
three-index form it is simpler: **one `B` factor is one charge distribution**,
so "keeping the pairing" is automatic — promote one MO index of one `B`
factor, `B[L,y,q] → B[L,x,q]`. The metric's derivative appears only under a
*nuclear* displacement, where `(M|μν)` and `V` both move with the atoms; that
is the derivative-integral contraction on the AO side, not part of `Y`, and
this inventory does not size it.

**What the energy reads.** The DF energy is a function of `h`, `B` and the CI
vector, and `B` enters it in two ways. The class blocks are products of the
four carried blocks (the table in `src/df_integrals/df_integrals.cpp`'s
`slab()`; the active `h2e` is `B_aa · B_aa`). The **finished** quantities —
the `h^eff` slices and `e_core`/`e_virt` — are contractions of `B` too, and
substituting their definitions (§3.1) shows two more blocks, which the golden
does not need because it carries those quantities as numbers:

- `B_cc`, **in full**: `h^eff`'s exchange dressing `Σ_j (pj|jq)` on its
  core-indexed slices (`h1e_v_Si`, `h1e_v_Sir`) is `Σ_{L,j} B[L,p,j]
  B[L,j,i]`, with `j ≠ i`; `ε_i` reads `(ij|ji)` the same way. Its Coulomb
  dressing `(pq|jj)` reads only the diagonal `B[L,j,j]`.
- `B_vv`, **on its diagonal only**: `ε_r` reads `(rr|jj)` and `(rr|tu)`,
  i.e. `B[L,r,r]`. Its exchange terms `(rj|jr)` and `(rt|ur)` are `B_vc ·
  B_cv` and `B_va · B_av`, carried. Nothing reads `B_vv` off its diagonal:
  that is §3.2's "no class puts two virtuals in one charge distribution",
  and it is why the energy needs no `(L|vv)`.

**What `Y` reads.** Write `Γ[L,q,y] = ∂E/∂B[L,q,y] + ∂E/∂B[L,y,q]`, the
three-index adjoint of the energy. Then the two-electron part of `Y` is

```
  Y_xy = sum_{L,q} B[L,x,q] Γ[L,q,y]          (plus the one-electron h terms, §3.5)
```

Two factors, two questions. Building `Γ` reads what the energy reads — the
four carried blocks, `B_cc` in full, the diagonal of `B_vv`. The promoted
factor `B[L,x,q]` is read for **every** `x` and for every `q` in `Γ`'s
support, and that support covers all three spaces (`Γ[L,v,a]` is non-zero
from `Sr` alone, `Γ[L,c,a]` from `Si`, `Γ[L,a,a]` from every class). So row
`x` of `Y` reads `B[L,x,·]` over the whole MO range:

| rows of `Y` | promoted factor | carried? |
|---|---|---|
| `x` core | `B[L,i,q]` = `B_cc`, `B_ca`, `B_cv` | **no**: `B_cc` |
| `x` active | `B[L,t,q]` = `B_ac`, `B_aa`, `B_av` | yes (`B_ac`, `B_av` are `B_ca`, `B_va` transposed) |
| `x` virtual | `B[L,r,q]` = `B_vc`, `B_va`, `B_vv` | **no**: `B_vv`, in full |

**`Y` over its full range reads the whole `(L|pq)` tensor.** Row by row
against §3.3, each array is `(xP|QR) = Σ_L B[L,x,p] B[L,q,r]`: a strip
`B_xP` (`x` general, `p` in `P`) times the block `B_QR`, and a restricted row
reads the diagonal of its tied block:

| array | three-index form | rebuildable from the carried four? |
|---|---|---|
| `(xa\|aa)` | `B_xa · B_aa` | **yes** |
| `(xv\|aa)` | `B_xv · B_aa` | no: `B_vv` (rows `x` virtual) |
| `(xa\|va)` | `B_xa · B_va` | **yes** |
| `(xc\|aa)` | `B_xc · B_aa` | no: `B_cc` (rows `x` core) |
| `(xa\|ca)` | `B_xa · B_ca` | **yes** |
| `(xv\|va)` | `B_xv · B_va` | no: `B_vv` (rows `x` virtual) |
| `(xc\|ca)` | `B_xc · B_ca` | no: `B_cc` (rows `x` core) |
| `(xa\|cv)` | `B_xa · B_cv` | **yes** |
| `(xv\|ca)` | `B_xv · B_ca` | no: `B_vv` (rows `x` virtual) |
| `(xc\|va)` | `B_xc · B_va` | no: `B_cc` (rows `x` core) |
| `(xc\|cv)` | `B_xc · B_cv` | no: `B_cc` (rows `x` core) |
| `(xv\|cv)` | `B_xv · B_cv` | no: `B_vv` (rows `x` virtual) |
| `(xa\|cc)` | `B_xa · B_cc` | no: `B_cc` in full, every row |
| `(xv\|cc)` | `B_xv · B_cc` | no: `B_cc` in full, every row; `B_vv` (rows `x` virtual) |
| `(xc\|cc)`, tied | `(xi\|jj)` = `B_xc · diag B_cc`; `(xj\|ji)` = `B_xc · B_cc` | no: `B_cc` in full, both factors |
| `(xa\|vv)`, tied | `(xt\|rr)` = `B_xa · diag B_vv` | no: `diag B_vv`, every row |
| `(xc\|vv)`, tied | `(xi\|rr)` = `B_xc · diag B_vv` | no: `B_cc` (rows `x` core); `diag B_vv` |

Where §3.3 has `B_xc`, `B_xa`, `B_xv`, the strip is three blocks: `B_xa` is
`B_ca`, `B_aa`, `B_va`, all carried; `B_xc` is `B_cc`, `B_ac`, `B_vc`; `B_xv`
is `B_cv`, `B_av`, `B_vv`. Four of the seventeen rows — the four `(xa|QR)`
with `QR` in a carried block — rebuild from the golden as it is. The other
thirteen need one of two blocks, and nothing else is missing.

**The missing blocks**, in `B`'s six-block decomposition by space pair, are
exactly two:

| block | needed | stored square | packed (`B[L,p,q] = B[L,q,p]`) | its diagonal alone |
|---|---|---|---|---|
| `B_cc` `(L\|cc)` | in full: `Y`'s core rows; `Γ` (the `h^eff` and `ε_i` exchange) | `naux · n_core^2` | `naux · n_core (n_core + 1) / 2` | `naux · n_core` |
| `B_vv` `(L\|vv)` | in full: `Y`'s virtual rows; its diagonal also in `Γ` (`ε_r`) | `naux · n_virt^2` | `naux · n_virt (n_virt + 1) / 2` | `naux · n_virt` |

against the carried four's `naux · (n_act^2 + n_core n_act + n_virt n_act +
n_core n_virt)`. With both added, the six blocks are the complete tensor:
`naux · (n_core^2 + n_act^2 + n_virt^2 + n_core n_act + n_act n_virt +
n_core n_virt)` stored square, `naux · n_mo (n_mo + 1) / 2` fully packed.
The two missing blocks are carried, stored square, by the density-fitted
gradient sidecar beside a `--df` golden (#164;
[`reference-data.md`](reference-data.md), "The density-fitted gradient
sidecar"), so the goldens themselves stay as they are.

**`B_vv` is the energy's DF economy, and the gradient does not keep it.**
The energy needs no `(L|vv)` because no charge distribution it reads holds
two virtuals off the diagonal. Promotion puts a general index beside a
virtual one, so `Y`'s virtual row — `x = s` against a `q = r` in `Γ`'s
support — reads `B[L,s,r]` for every pair. That is not only `Y`'s
virtual–virtual block: `Y_sb` (virtual–active) reads `Σ B[L,s,r]
Γ[L,r,b]` through `Sr`'s `Γ[L,r,b]`, and `Y_si` (virtual–core) likewise
through every class that reads `B_cv`. So `B_vv` is wanted by every block in the
virtual row, and the two mixed ones are non-redundant on both branches
(§2.1). What *does* survive is §3.4's saving in its strongest form: the
gradient never forms a four-index array at all (the derivation's statement:
the in-tree code, #165, does form §3.3's arrays on the device from `B` —
[`gradients.md`](gradients.md), §6.2, says why), so `(xv|vv)` — the one §3.4
names — is never built, and the largest three-index block is `naux ·
n_virt^2` where the conventional inventory's largest rows are `n_mo ·
n_virt^2 · n_core` and `n_mo · n_virt^2 · n_act` (§3.3). Since `naux` is
`O(n_mo)` for a fitting basis, the DF inventory is `O(n_mo^3)` against the
conventional one's `O(n_mo · n_virt^2 · n_occ)`: smaller by a factor of
order `n_occ`. The bytes for a given case are a measurement and belong in
[`reference-data.md`](reference-data.md).

**§1.3's trap, in three-index form.** `Y`'s active–active block needed two
arrays the energy never reads, `(a'a|jj)` and `(a'a|rr)` — `(xa|cc)` and
`(xa|vv)` restricted to `x` active, both on tied pairs. Their counterparts are

```
  (a'a|jj) = sum_L B[L,a',a] B[L,j,j]        B_aa against diag B_cc
  (a'a|rr) = sum_L B[L,a',a] B[L,r,r]        B_aa against diag B_vv
```

`B_aa` is carried; the two diagonals, `naux · n_core` and `naux · n_virt`,
are not. So the active–active block wants only the **diagonals** of the two
missing blocks — and the same holds of the orbital energies' response to the
active density, which `y_I` and the active block both read: of its four tied
arrays (`gradient_plan.cppm`'s `OrbitalEnergyResponse`), `(ii|ut)` and
`(rr|ut)` are `B_aa` against the two diagonals, and `(iu|ti)` and `(ru|tr)`
are `B_ac · B_ca` and `B_av · B_va`, carried. The trap is the same one: the
active block is not "what the golden already carries". The full active
*row* asks for more — `Y_tj` (active–core) reads `(ut|ji)` with `j ≠ i`,
`(xa|cc)` off its tie, from the exchange dressing of `h1e_v_Si`, so `B_cc` in
full — and the full range more again, both blocks in full.

**The rest carries over.** The digests' own orbital response (§3.7's
`f3ac`/`f3ca`) reads `h2e = B_aa · B_aa` alone, whose promotion is `B_xa ·
B_aa`: carried, so that term needs no new block. §2.2's `W_tu` reads `B_cc`
and `B_vv` off their diagonals (`(pq|tu)` with `p < q`), a consumer of the
same two blocks outside `Y`. §3.5's `n_mo × n_mo` list is unchanged in shape;
its matrices must be the density-fitted calculation's own (every integral
inside `h^eff`, `f` and `F` is `Σ_L B B`), and §3.6's invariants hold against
those.

## 4. The active-space-selection hazard, and the settled rule

§4.1–§4.2 are the **CASCI branch's** hazard. It does not exist on the CASSCF
branch, where core–active and active–virtual are held by stationarity rather
than by a selection (§1.4); that branch's counterparts are §2.7's
conditioning and the precondition in §4.4. §4.3's rule and §4.4 are
**shared**.

### 4.1 A selection is not a stationarity condition

§1.2's table is the whole hazard in one row: the `core`–`act_O` and
`act_V`–`virtual` blocks of (O1) hold because the active space was *picked*,
not because anything converged. A selection is a **discrete** function of the
nuclear geometry. It is locally constant, and where it changes it is not
differentiable at all.

Within a region where the selection is stable, there is no problem: core and
virtual are invariant subspaces of `F`, (O1) is a perfectly smooth constraint,
and §2.3's divisions deliver its multipliers. At a point where the selection
changes, the energy as a function of geometry has a kink. The analytic gradient
is then the gradient of one branch, the numerical gradient straddles the kink,
and the two disagree for a reason that is neither a bug in the derivation nor a
bug in the code.

### 4.2 Three ways it bites

- **Energy-ordered selection can reorder.** With the active space taken as the
  `n_act` RHF orbitals around the Fermi level, a displacement that swaps two
  orbital energies across the core/active or active/virtual boundary swaps
  which orbital is correlated. The CASCI state changes discontinuously.
- **Irrep-based selection needs the point group to survive.** `--cas-irreps`
  picks the active space by irrep ([`reference-data.md`](reference-data.md),
  "Salicylaldimine CAS(8,8)"), and the group is **auto-detected** per geometry
  ([`reference-data.md`](reference-data.md), "Point-group symmetry"). A
  displacement that lowers the symmetry destroys the labels the selection is
  written in, and the "same" active space at the displaced geometry is a
  different set of orbitals — or the selection fails outright.
- **A near-degenerate boundary is ill-conditioned even when it is stable.**
  §2.3's selection multipliers divide by `eps^RHF_p - eps^RHF_i` across the
  boundary. A small gap gives a large multiplier and an inaccurate one. This is
  the same quantity the golden generator already guards inside the core and
  virtual spaces ([`reference-data.md`](reference-data.md), "The degeneracy
  refusal"); the gradient extends the concern to the *boundaries* of the active
  space, where the generator does not look, and with a worse power of the gap.

### 4.3 The rule

> **Numerical validation of the gradient uses in-plane displacements of a
> planar `Cs` molecule only.**

Two independent justifications, and it needs both:

**It keeps the selection valid.** An in-plane displacement of a planar molecule
preserves the molecular plane, hence the `σ_h` reflection, hence `Cs`, hence the
`A'`/`A"` labels the active space is specified in. Every displaced geometry is
detected as the same point group with the same irreps, so `--cas-irreps` picks
the same orbitals, and the comparison is between two energies of the *same*
method. An out-of-plane displacement breaks the plane; the detected group drops
to `C1`, `A'`/`A"` cease to exist, and there is nothing left for the selection
to mean. A numerical-gradient generator should therefore **refuse** a
displacement whose detected point group differs from the reference geometry's,
rather than quietly produce a number from a different active space.

**Nothing is lost.** At a planar geometry every out-of-plane gradient component
is **zero by symmetry**. The gradient is a vector field invariant under `σ_h`;
reflection maps each atom to itself and negates the out-of-plane component, so
that component must equal its own negative. The components an out-of-plane
displacement would probe are exactly the ones that are analytically zero. They
are not a gap in the validation — they are a **free check**: an analytic
gradient that does not return zero there is wrong, and no finite difference is
needed to find out.

So the rule costs nothing and buys a well-defined comparison. It is also why a
planar `Cs` molecule is the right shape of test case in the first place: enough
independent in-plane components to be a real test, and a symmetry that an
in-plane displacement cannot break.

**Both justifications hold on the CASSCF branch too.** The second is a
property of the gradient, not of the reference. The first loses its selection
but keeps its point: a symmetry-adapted CASSCF is specified in the same irrep
labels, and an in-plane displacement keeps them meaningful, so each displaced
optimisation is asked for the same active space.

### 4.4 What is *not* a hazard, and what is a different one

Three things that look like this hazard and are not, listed because conflating
them leads to chasing the wrong bug. All three apply to both branches:

- **The active natural-orbital gauge.** Degenerate active occupation numbers
  make the natural orbitals ambiguous, and the ambiguity is harmless: the
  energy is invariant to an active–active rotation (§1.3), so a natural-orbital
  reordering between displaced geometries does not move `E`. Do not add a
  guard for it.
- **Run-to-run reproducibility, and the CASSCF branch's precondition.**
  Whether a single geometry's state is well defined run to run is a
  *precondition* for a numerical gradient, not the selection question. For
  the CASCI branch it holds by construction. For the CASSCF branch it is
  measured at one fixed geometry, perturbed starting points included
  ([`reference-data.md`](reference-data.md), "A fresh CASSCF reproduces,
  measured"). What a finite difference additionally needs is that each
  *displaced* optimisation lands on the same stationary branch as the
  reference — the branch §1.4's smoothness argument guarantees exists, not
  one the optimiser is guaranteed to find. A displaced optimisation that
  lands on a different stationary point gives a number from a different
  reference. `nevpt2_numgrad.py --reference casscf` guards every displaced
  state against it and probes for it by a warm start; what it measured is
  [`reference-data.md`](reference-data.md), "The numerical gradient of a
  CASSCF reference".
  It is the CASSCF branch's own hazard, and it is a different one from §4.1's.
- **Park's SC non-invariance finding — reference-independent.** The strongly-contracted energy is not
  invariant to rotations *among* the inactive orbitals — core and virtual — so
  where those are degenerate the SC gradient and the SC numerical gradient
  legitimately disagree. Park reports exactly that, and reports that it blocks
  SC geometry optimisation on symmetric cases while PC stays well behaved.
  **This tree has already measured the energy-side half of it**: `Srs`, summed
  over *pairs* of virtuals, moves under a rotation within a degenerate virtual
  pair where `Sr` does not, which is why the goldens are built with
  point-group symmetry ([`reference-data.md`](reference-data.md), "Point-group
  symmetry"). It is a property of the SC functional, living inside core and
  virtual, and so of neither reference: it reaches the gradient through the
  `z^c` divisions, which are the same on both branches (§2.3), so a CASSCF
  reference does not remove it. It is also independent of
  §4.1–§4.3, which are about the *boundaries* of the active space and apply
  to SC and PC alike. A symmetric
  case is therefore a useful deliberate test of the former and a thing to
  avoid for the latter; when the two are measured against each other, the
  result belongs in [`testing.md`](testing.md) with its card and flags.

### 4.5 Related work at first order, and what the Hessian's two refusals leave alone

[`hessian-theory.md`](hessian-theory.md) finds two places where an energy
second derivative does not exist. One is SC's symmetry-empty perturbers,
populated along a symmetry-breaking pair (§5.6 there). The other is PC's cut
null modes, lifted along a symmetry-breaking direction (§6.2 there). The
first-order literature closest to them is below. Full entries are in
[`references.md`](references.md), "Related work: the two 'no Hessian here'
findings".

- **Park 2019 (Ref. [1])**: SC non-invariance at degenerate inactive
  orbitals (§4.4). It is a first-order effect, but it is not the empty
  perturbers. The SC functional moves under a rotation inside core or
  virtual, so its gradient and its numerical gradient disagree where those
  orbitals are degenerate. The empty-perturber finding leaves the gradient
  exact and concerns only the second derivative.
- **Nishimoto 2019 (PC-NEVPT2 first derivatives)**: linearly dependent
  functions are removed by a threshold, and the gradient's accuracy "is not
  affected because all postenergy processes … use only linearly independent
  vectors". The energy's continuity across geometries "has not been
  investigated carefully". The first statement agrees with what this tree
  uses: at an exact eigenstate the two-term PC tangent
  ([`gradients.md`](gradients.md), §4.3) is the derivative of the energy
  the code computes.
- **Nishimoto 2023 (IPEA-shifted CASPT2 first derivatives)**: a term that
  couples the linearly dependent and independent spaces. It has the
  structure of the third term `-2 rho^T Sdot P A x` in
  [`hessian-theory.md`](hessian-theory.md), §6.2. In CASPT2 the IPEA shift
  makes the coupling. In PC-NEVPT2 it appears only off the eigenstate, at
  the displaced CI vectors a finite difference visits, and it vanishes at
  the state the gradient is taken at.
- **Park 2021 (XMCQDPT2 gradients)**: a spurious acrolein frequency traced
  to the overlap threshold. It shows a threshold that is harmless to a
  gradient appearing in a Hessian built from that gradient.

**First order is unaffected by both findings**, as each section of
[`hessian-theory.md`](hessian-theory.md) shows:

- An empty SC tuple's energy is `O(h^2)` along every direction, so its
  first derivative is the guard's zero.
- A Gram metric's dropped eigenvalues do not move at first order.
- The PC third term vanishes at an exact eigenstate.

Nothing in §1–§4 changes, and the Z-vector equations, source terms and
validation rule stand as written.

## 5. What this settles

| question | CASCI on RHF orbitals | CASSCF orbitals |
|---|---|---|
| Does Park's Eq. 23 apply? | Not its first line. CASCI is not orbital-stationary; the constraint is (O1), `F` block-diagonal over core/active/virtual (§1.2). | **Verbatim**, the `(1/2) tr[Z^dag (A - A^dag)]` line included (§1.4). |
| CP-CASSCF or CPHF? | **CPHF** (§2.4). The RHF Fock has no CI dependence, so the CI equation does not read `Z`; the system is block-triangular — `z^c`, then `z`, then the orbital multipliers (§2.2, §2.5). | **CP-CASSCF** (§2.7). `∂(A − A†)/∂c ≠ 0`, so the CI and orbital equations are one coupled system over `(Z, z)`: §2.5's steps 2 and 4 merged. |
| Which multipliers need a linear solve? | One: `Z` on the `O`–`V` block, the only iterative step. The pseudocanonical multipliers are Park's Eqs. 56–57 verbatim; the selection-boundary ones are small Sylvester equations, not plain divisions, because `F` is not diagonal in the golden's gauge (§2.3). | One: the coupled `(Z, z)` system, on core–active, core–virtual and active–virtual. The pseudocanonical multipliers are the same divisions; there are no selection-boundary ones (§2.7). |
| Does the CI equation see the orbital constraints? | Only through `z^c` — one way, as a right-hand side `y^c` (§2.2). | Through `z^c` as on the left, **and** through `Z` (§2.7). |
| Does the active–active block need a multiplier? | No. It is a gauge, redundant with the CI parameters (§1.3). | No — the same gauge (§1.3). |
| Is the RHF partition (`act_O`/`act_V`) needed? | Yes, recovered by one `n_act × n_act` diagonalisation (§2.6). | No. There is no RHF partition; §2.6 is unnecessary (§1.4). |
| Which MO-integral arrays does Eq. 28 read? | Seventeen one-general-index arrays, §3.3's table, three of them only on restricted slices. | The same seventeen. |
| Which does it not read? | `(xv\|vv)`, and that omission is the whole saving over an `n_mo^4` dump (§3.4). | The same. |
| Density-fitted, which `B` blocks does `Y` read? | All six: the whole `(L\|pq)`. The `--df` goldens carry `B_aa`, `B_ca`, `B_va`, `B_cv`; `B_cc` (`naux·n_core^2`) and `B_vv` (`naux·n_virt^2`) are missing, and the active–active block needs only their diagonals. `κ` never reaches the metric (§3.8). | The same. |
| What else must the reference data carry? | §3.5: `mo_coeff`, `h`, `f`, `f^h`, the three `h^eff` variants and the RHF Fock, each `n_mo × n_mo`, plus the RHF electron count. | §3.5 less the RHF Fock and the RHF electron count. |
| What can make the derivative ill-defined? | The selection: a reordering or a lost point group puts a kink in the energy; a small gap across an active-space boundary makes the selection multipliers large (§4.1–§4.2). | Not a selection. A near-singular CASSCF Hessian (§2.7), and a displaced optimisation landing on a different stationary branch (§4.4). |
| How is numerical validation set up? | In-plane displacements of a planar `Cs` molecule, refusing any displacement whose detected point group changes; out-of-plane components checked against zero instead (§4.3). | The same rule (§4.3). |
| Does Park's SC non-invariance apply? | Yes, through `z^c` (§4.4). | Yes, identically: it is a property of the SC functional, not of the reference (§4.4). |
| Is it implemented? | The source terms in `src/`; the Z-vector solve offline only ([`gradients.md`](gradients.md), §1). | The coupled solve offline only (`nevpt2_grad.py --reference casscf`, [`gradients.md`](gradients.md), §1); every derivative path in the tree refuses a `casscf` golden (§1.1). |

## References

1. J. W. Park, *Analytical gradient theory for strongly contracted (SC-) and
   partially contracted (PC-) N-electron valence state perturbation theory
   (NEVPT2)*, J. Chem. Theory Comput. **15**, 5417 (2019).
   [doi:10.1021/acs.jctc.9b00762](https://doi.org/10.1021/acs.jctc.9b00762),
   [arXiv:1907.10180](https://arxiv.org/abs/1907.10180). **Equation numbers
   throughout this document are this paper's**, and it is listed in
   [`references.md`](references.md) with the rest of the literature.

The NEVPT2 energy expressions the derivative is taken of, and the definitions
of the `h^eff` variants Eq. 28 reads, are Angeli, Cimiraglia and Malrieu's
spinless formulation, also in [`references.md`](references.md).
