# Gradients: what the code computes

What the gradient path in this tree actually computes, how the two modes
check each other, and where it stops. The **derivation** — the Lagrangian on
each of the two references a golden can carry (CASCI on RHF orbitals, and on
CASSCF orbitals), their constraints, Z-vector equations and solve orders, the
integral inventory they share — is [`gradient-theory.md`](gradient-theory.md);
this document describes the **code**, and cites that one rather than
restating it. All of the code is on the CASCI branch.

Like [`architecture.md`](architecture.md), it quotes **no measured numbers**.
Every tolerance, residual, plateau and timing lives in
[`testing.md`](testing.md), [`performance.md`](performance.md),
[`pc-nevpt2.md`](pc-nevpt2.md) or [`reference-data.md`](reference-data.md),
cited below by section name, so a figure has exactly one home. What appears
here instead are *size formulas* in `n` (the active-space orbital count) and
`ndet`, which are algebra.

## 1. What is here, and what is not

**This is the source-term half of an analytic gradient, and nothing
downstream of it.** Every piece below computes a derivative of the NEVPT2
energy with respect to a *parameter of the wavefunction* — the CI vector, or
an orbital rotation — at one fixed geometry. Nothing here differentiates with
respect to a nuclear coordinate.

| | state |
|---|---|
| the SC energy's derivative w.r.t. the CI vector, `y_I = dE/dc_I` (Park Eq. 48) | **in tree**, both modes, checked against each other exactly (§5) — and, given the `*.gradgold` sidecar, of the **complete** functional, the orbital energies' response to the active density included (§5.5) |
| the SC energy's pseudodensities (Park Eqs. 41–47) | **in tree**, checked by their own algebraic identities (§5.2) |
| the PC energy's derivative w.r.t. the CI vector | **in tree, both modes**: forward (§4.3), checked against a finite difference, and reverse — `y_I = dE_pc/dc_I` by the PC solve's adjoint (§5.6, #74) — checked against it exactly, with the orbital energies' response in both, and against block2's committed derivative |
| the PC orbital source term `Y_xy = dE_pc/dkappa_xy` | **in tree, both modes** (§6.1, #74): over the whole `kappa` space, the digests' transpose and the orbital energies' response in it, the adjoint against the PC forward tangent exactly, and against block2's committed `A` block by block |
| the orbital source term `Y_xy = dE/dkappa_xy` (Park Eq. 26) | **in tree over the whole `kappa` space** — active–active, core–core, virtual–virtual and mixed — both modes, checked against each other exactly (§6), and **exact**: with the digests' term below in it, the full `<kappa, Y>` against the full forward `Edot(kappa)` holds to round-off too |
| the orbital source term on a **density-fitted** case, `Y_xy` of the fitted energy | **in tree, both terms, both modes** over the whole `kappa` space (§6.2): `nevpt2_sc_pseudodensity --kappa-full` on a `*_df` golden contracts the inventory out of the whole `B[L,p,q]` (#165) and gates the frozen-digest duality, the promotion table, the active block, the two intra-space finite differences and Eqs. 56–57, and (#168) the digests' own term — the full duality, its own finite difference forward and reverse, the exact zeros, and the transpose carried onto the three-index `B` against the four-index pairing. `--dump-source-terms` writes the fitted `Y` and `y` with both terms; `--source-oracle` stays refused (no fitted `.srcgold` exists), as does `--pc` there. **A density-fitted SC nuclear gradient is built from them** (#166): `reference_data/nevpt2_grad.py --df --source dump` / `both` on N2 CAS(4,4) `_df`, both `Y` terms carried into the gradient's own active basis, against the fitted `grad_num` of the NEVPT2F1 sidecar (#167) at the 1e-6 gate for two finite differences, beside PySCF's DF-NEVPT2-sourced gradient on the same sweep — both within the gate, with the residual-independent checks one short of passing on N2's known nuclear-ladder edge ([`testing.md`](testing.md), "The density-fitted gradient against the numerical one") |
| the digests' own orbital response, the `f3ac`/`f3ca` digests' `h2e` dependence | **in tree, both modes** (§6): forward, along any direction as a term of its own beside the frozen-digest tangent (#88) — checked against PySCF's exact directional derivative offline and a finite difference on the card; reverse, `Y` carrying it through the digests' **transpose**, a tiled kernel that never forms the 4-RDM (#89) — checked against the forward term exactly, and against PySCF's exact `Y` block by block on the card |
| the RDMs' own derivatives w.r.t. the CI vector | **in tree**, host `dm1`/`dm2` and device `dm3`/`f3ac`/`f3ca` (§4.1) |
| the pseudocanonical multipliers `z^c` (Park Eqs. 56–57) | **in tree**, divided out of `Y`'s two inactive blocks, refusing a degeneracy the division cannot resolve (§6) |
| the GPU's `Y_xy` and `y_I` **written out**, for a nuclear gradient to be built from | **in tree**, SC and (with `--pc`, #74) PC: `nevpt2_sc_pseudodensity --dump-source-terms PATH` writes both to a `.gradsrc` (§8), which `reference_data/nevpt2_grad.py --source dump` / `both` reads. A gradient built from it **is the SC-NEVPT2 nuclear gradient**: within the 1e-6 gate of `grad_num` on LiF, acrolein and N2 CAS(4,4), and componentwise PySCF-sourced's to well inside it ([`testing.md`](testing.md), "The GPU as the source-term provider", which keeps the #73 failure — `Y` holding the digests fixed — on record as superseded). **On a CASSCF reference too**: SC (#125), `--reference casscf --source dump` / `both` on LiF CAS(4,6) and N2 CAS(4,4) ([`testing.md`](testing.md), "SC on a CASSCF reference: LiF, and the N2 degeneracy probe"), and PC (#163), `--reference casscf --pc --source dump` / `both` on acrolein CAS(6,5) against the GPU-differenced `grad_num_pc`, beside block2-sourced on the same sweep ([`testing.md`](testing.md), "PC on a CASSCF reference: acrolein", "GPU-sourced"). **And density-fitted** (#166), SC on the CASCI reference: `nevpt2_grad.py --df` (the density-fitted row above) |
| the CASCI branch's Z-vector solve: the CI Z-vector equation, the selection-boundary multipliers, the CPHF solve | **not in tree** — **offline Python only**, `reference_data/nevpt2_grad.py`, a `pyscf.grad.lagrange.Gradients` subclass ([`reference-data.md`](reference-data.md), "The analytic gradient"). Deliberately offline: this repo's shape is "offline Python generates the reference, the C++ GPU demo reproduces it", and a PySCF solve inside the runtime would invert that. Derived in [`gradient-theory.md`](gradient-theory.md), §2.2–§2.6; nothing in `src/` implements them |
| the CASSCF branch: Park's Eq. 23 as written, and its coupled CP-CASSCF `(Z, z)` solve | **not in tree** — **offline Python only**, `reference_data/nevpt2_grad.py --reference casscf --mo-from FILE` (#122): `z^c`, then the coupled system built densely from `pyscf.mcscf.newton_casscf.gen_g_hop`'s orbital+CI Hessian and factorised, SC with PySCF's own source terms or the GPU's dumped ones (`--source dump` / `both`, #125), or, with `--pc`, PC with block2's (#126) or the GPU's dumped PC ones (#163) ([`reference-data.md`](reference-data.md), "The analytic gradient"). **What is validated, per reference:** SC on LiF CAS(4,6) and N2 CAS(4,4) componentwise, PySCF- and GPU-sourced, N2 being the inactive-degeneracy probe ([`testing.md`](testing.md), "SC on a CASSCF reference: LiF, and the N2 degeneracy probe"), and SC and PC on acrolein CAS(6,5), PC both block2- and GPU-sourced ([`testing.md`](testing.md), "PC on a CASSCF reference: acrolein"). The CASCI-on-RHF reference's cases, the GPU's PC dump among them, are [`testing.md`](testing.md), "What is run, and what it comes out at" and "The GPU as the source-term provider". Derived in [`gradient-theory.md`](gradient-theory.md), §1.4 and §2.7. Its numerical gradients are offline too: `nevpt2_numgrad.py --reference casscf` re-optimises every displaced state (SC), and `--pc-states` differences the GPU's PC energies on those states ([`testing.md`](testing.md), "PC on a CASSCF reference: acrolein"). The in-tree Lagrangian path still refuses a `casscf`-kind golden as `err_unsupported`, and the one sidecar `--mo-from --gradgold` writes for one, `golden/n2_ccpvdz_cas44_casscf.gradgold`, is data no derivative path reads yet ([`reference-data.md`](reference-data.md), "The CASSCF-reference gradient sidecar") |
| derivative MO integrals — the one- and two-electron skeleton derivatives — and the nuclear-repulsion derivative | **not in tree**, and not in the reference data either |
| a nuclear gradient, a geometry optimisation, or any derivative w.r.t. a coordinate | **not in tree** |
| a *numerical* nuclear gradient | **offline Python only** — `reference_data/nevpt2_numgrad.py` writes it into the `*.gradgold` sidecar ([`reference-data.md`](reference-data.md), "The numerical gradient"). No binary in `src/` or `apps/` reads it; it is what `nevpt2_grad.py`'s analytic answer is gated against, SC and PC, at a tolerance that cannot be this repo's 1e-7 because both sides are finite differences ([`testing.md`](testing.md), "The analytic gradient against the numerical one") |

**Nothing is held fixed any more by the orbital source term `Y`.** The last
dependence it held, the `f3ac`/`f3ca` digests' own `h2e` dependence, is in
both modes: forward as a term beside the frozen-digest tangent (#88), reverse
through the digests' transpose (#89), and the two meet in a full duality as
exact as the integral halves' (§6). The orbital energies, which were held
fixed with it, move in `Y` too. A nuclear gradient built from the GPU's own
`Y` and `y` is therefore the NEVPT2 gradient, and is measured to be
([`testing.md`](testing.md), "The GPU as the source-term provider").

**The CI source term holds nothing fixed, given the sidecar.** `e_core` /
`e_virt` are the generalized Fock's diagonal and so move with the active
density; `y_I` and its forward tangent both carry that response when they are
handed the sidecar's integrals (§5.5), which the dump always is and
`--gradient-source` is wherever a sidecar is committed. Without one —
CAS(10,10) — both modes hold the orbital energies at the golden's and say so.
The omission they used to share showed in no check this tree's binaries run —
each mode pair differentiates the same functional — and only in a nuclear
gradient: the `.gradsrc` dump (§8) is what measured it, and what now measures
it gone, in [`testing.md`](testing.md), "The GPU as the source-term
provider".

## 2. Forward and reverse, and why the permanent tests are exact

Every derivative here is computed **twice, in two modes**, and the agreement
between them is the test. That is the whole validation strategy, and it is
why the golden tier's gradient entries contain almost no step sizes.

- **Forward mode** carries a *direction*. Given a direction `u` in determinant
  space, the tangent machinery propagates it through the same pipeline the
  energy runs — the RDM build, then the class energies — and returns one
  scalar, `Edot(u) = d/dt E(c + t u)`.
- **Reverse mode** carries a *functional*. The pseudodensity walk accumulates
  `dE/d(each array the energy reads)`, sweeps those down to the five RDMs,
  and contracts them against the RDMs' own CI derivatives to give the whole
  vector `y_I = dE/dc_I` in one pass.

**The duality.** Both compute the same linear map, so for *any* direction

```
  <u, y>  ==  Edot(u)
```

with **no step size in it**. The two sides share no code past the slab walk:
forward mode substitutes tangent blocks into the class einsums and
differentiates only the scalar finish; reverse mode rewrites every einsum's
subscripts, sweeps the blocks backwards and ends in a sigma-vector kernel.
Two independent computations of one number, agreeing to round-off — the only
difference between them is the order their sums are reassociated in. That is
the gate `--gradient-source` enforces ([`testing.md`](testing.md), "The CI
source term's forward-reverse duality"), and it is *exact arithmetic*, not a
tolerance on a limit.

A single dot product detects a wrong sweep but does not localise it, so where
the determinant space is small enough the same entry compares `y_I` against
`Edot(e_I)` **column by column** — the one check that names the determinant a
mis-transcribed index landed on.

**Euler's identity is the second exact check, and it probes a direction the
seeded ones cannot.** Where a class's metric and Dyall-Hamiltonian blocks are
*linear* in the RDMs, the class energy is homogeneous of degree 2 in the CI
vector, so the tangent along the state itself is exactly twice the energy.
That holds for three of the eight SC classes and five of the eight PC ones —
which classes, and why the two sets differ, is in §4.2 and §4.3. The RDMs'
own tangents obey it too (a form quadratic in `c` has tangent `2 x` along
`c`), which is what `--rdm-tangent` checks against the plain build.

**Where a finite difference is still unavoidable: three places in the golden
tier, and no more.** An identity can only check a path against *itself*; a
convention shared by both modes cancels and stays invisible. So each shared
convention is pinned once, by differencing the unmodified energy stage:

| the finite difference | what it pins that no identity can |
|---|---|
| `nevpt2_demo --sc-tangent` | that `energiesTangentDevice` is the derivative of `energiesDevice` at all. Run once, to license every later dot-product identity against the tangent |
| `nevpt2_sc_pseudodensity --kappa-act`'s third number | the device kappa action against `nevpt2.kappa_rotation`'s exponential map. The adjoint and the tangent *share* that convention, so they cannot check it |
| `nevpt2_pc_tangent` | the PC tangent, which differentiates a **solve** and has no algebraic identity short of differentiating the eigendecomposition — exactly what that code avoids (§4.3) |
| `nevpt2_demo --sc-hessian` | one order up (#153): that `energiesHessianDevice`'s `Eddot(u, v)` is the derivative of `Edot(u)` at all — a difference of `energiesTangentDevice` itself ([`hessian-theory.md`](hessian-theory.md), §9) |
| `nevpt2_demo --sc-hessian-vector` | the vector `H_cc u` against a difference of the GPU's own `y`, the orbital-energy mode matched on both legs (#156). Not a convention the duality leaves open — `Eddot`'s difference already licenses what the duality holds `H_cc u` to — but the vector against the reverse walk directly ([`hessian-theory.md`](hessian-theory.md), §9) |

Each is a Richardson table, judged on its *shape* — a run of halvings whose
error falls fourfold — rather than on any single step, because the RDM
build's `atomicAdd`s reassociate and the energy itself carries round-off that
the difference quotient divides by `2h`. The measured plateaus, the ratios
and **the step at which round-off takes over** are in
[`testing.md`](testing.md), "The SC tangent's finite-difference cross-check"
and "The active-rotation source term", and — for PC, where the rest of PC's
numbers live — in [`pc-nevpt2.md`](pc-nevpt2.md), "The tangent with respect
to the CI vector" and, for the gate values and wall times of the PC entries
themselves, [`testing.md`](testing.md), "The PC tangent against a finite
difference". The unit tier differences more than those three — each sigma
vector against a host construction, and the tangent RDM build against the
plain device build, both card-side — but they are component checks, not the
gradient path end to end.

**One case where a straight-line difference is not second order at all.** The
RDMs are *exactly* quadratic in the CI vector, so a central difference of
`dm1`/`dm2` along the straight line `c ± h u` has **zero truncation error** —
it is the tangent, to round-off, at every `h`. A Richardson table of it
measures nothing. The unit tier therefore differences them along the
**normalized** path `(c + h u) / ||c + h u||`, which is curved and restores
the `O(h^2)` term the table needs (`test/rdm_host_dm/`). The energy totals
are different: they divide, so their difference quotients are genuinely
second order.

## 3. The pieces

Dependency order; `src/gradient/`'s components are in
[`architecture.md`](architecture.md), §4 with the rest of the graph.

| module | computes | checked by |
|---|---|---|
| `nevpt2.rdm_host_dm` (`src/rdm/`) | `dm1`/`dm2` from a CI vector, and their tangents. Host only — no kernel, no BLAS | `test/rdm_host_dm/`: each committed golden's own arrays rebuilt, and the tangents against a curved-path difference |
| `nevpt2.rdm_build`'s `:tangent` partition | `buildRdmTangentsDevice`: the tangent of the `dm3`/`f3ac`/`f3ca` build; and `buildRdmHessianTangentsDevice`, its three RDM sets at `(c, u)`, `(c, v)`, `(u, v)` from one sweep (#153) | `--rdm-tangent` against twice the plain build; `test/rdm/rdm_tangent_tests.cpp` against a finite difference, and the three-argument sets against the tangent at each pair |
| `nevpt2.sigma_vector` (`src/rdm/`) | the 3-body and integral-contracted 4-body sigma vectors, forward and dagger | `test/sigma_vector/`, against an exact host contraction, with negative controls |
| `nevpt2.energy`'s `:sc_tangent` partition | `energiesTangentDevice`: the eight SC class energies' tangent in the RDMs | `--sc-tangent`: two exact checks and one finite difference |
| `nevpt2.energy`'s `:sc_hessian` partition | `energiesHessianDevice`: their second directional derivative `Eddot(u, v)`, frozen orbital energies (#153) | `--sc-hessian`: two exact checks and one finite difference of `Edot` |
| `nevpt2.energy`'s `:pc_tangent` partition | `pcTangentsDevice`: the PC classes' tangent — the *solve* differentiated | `nevpt2_pc_tangent`: Euler, and a Richardson-verified difference |
| `nevpt2.sc_amplitudes` (`src/gradient/`) | the amplitude `T`, the multipliers `P`/`Q`/`R`, the pseudocanonical multipliers `z^c` (Park Eqs. 56–57) and `derivativeSubscripts`. Host only | `test/gradient/`, card-free |
| `nevpt2.kappa_rotation` (`src/gradient/`) | the orbital-rotation parameterisation: antisymmetric `kappa`, `exp(kappa)`, an array's active axes transformed. Host only | `test/gradient/`, card-free |
| `nevpt2.gradient_plan` (`src/gradient/`) | `nevpt2.gradient`'s pure host arithmetic: the promotion table, the classes' `Delta` vectors and slab multipliers (and, #156, their tangents along a CI direction), `termHasActive`, the CI source term's host fold; and `longestFourfoldRun`, the step-ladder rule every finite-difference check of the gradient path gates on ([`testing.md`](testing.md), "Residual-independent checks of the gradient path"). Host only | `test/gradient_plan/`, card-free |
| `nevpt2.gradient` (`src/gradient/`) | the SC pseudodensities, the orbital source term `Y_xy` over the whole `kappa` space, and `y_I = dE/dc_I`; and `dfGeneralIntegralsDevice` (`gradient_df.cpp`, #165), the density-fitted case's general-range inventory contracted out of the whole `B` tensor, and `dfDigestSourceDevice` (#168), the digests' transpose carried onto the three-index `B` (§6.2); and `scCiHessianVectorDevice` (#156), the CI Hessian-vector product `H_cc u` by reverse over forward ([`hessian-theory.md`](hessian-theory.md), §4.2) | `nevpt2_sc_pseudodensity` (identities, `--kappa-act` and `--kappa-full`), `nevpt2_demo --gradient-source` (duality), `nevpt2_demo --sc-hessian-vector` (duality against `Eddot`) |
| `nevpt2.gradient`'s `:pc_classes` partition | PC's reverse mode (#74): each PC class's solve differentiated through its own `S`, `K` and `S_1` (`pcClassMatrices`, `pcSolveClass`, exported from `nevpt2.energy` for this), giving `y_I`, `Y_xy`, the digests' adjoint and the PC rotation tangent (`pcCiSourceDevice`, `pcFullRotationDevice` / `Matrix`, `pcDigestAdjointDevice`, `pcDigestRotationTangentDevice`). Its per-tuple arithmetic is `nevpt2.energy_finish`'s `pcAdjoint*` / `pcRotationTangent*` | `nevpt2_demo --gradient-source --pc` (CI duality), `nevpt2_sc_pseudodensity --pc --dump-source-terms` (rotation dualities) and `--pc --source-oracle` (block2's `A`, `P y`); `test/energy_finish/` card-free; `test/pc_adjoint/` the refusals |
| `nevpt2.pc_tangent` (`src/gradient/`) | the PC tangent's finite-difference check: the analytic tangent against re-solves at displaced states | `nevpt2_pc_tangent`; the gate itself in `test/pc_tangent/` |
| `nevpt2.golden`'s `loadGradGold` | the `*.gradgold` sidecar reader — the one-general-index MO-integral arrays and full-MO-range Fock matrices the core, virtual and mixed blocks of `Y` read — and `checkAgainst`, which says a sidecar and a golden describe one state (and, since #164, one calculation: a density-fitted `NEVPT2F1` sidecar, carrying `B_cc` / `B_vv` and the four tied orbital-energy arrays, pairs only with a `*_df` golden of its `naux`) | `test/golden/gradgold_tests.cpp`, host-only; `--kappa-full` is the consumer, and `--gradient-source --gradgold` reads either flavour's orbital-energy response |

`nevpt2.gradient` and `nevpt2.pc_tangent` are **leaves**: nothing in `src/`
imports either. `nevpt2.gradient` is linked by `nevpt2_sc_pseudodensity` and,
for `--gradient-source` and `--sc-hessian-vector`, by `nevpt2_demo`; `nevpt2_pc_tangent` is the
only app linking `nevpt2.pc_tangent`, and `nevpt2_sc_pseudodensity` the only
one linking `nevpt2.kappa_rotation` (its `--kappa-act` and `--kappa-full`
finite differences rotate stored blocks on the host).

## 4. Forward mode: the tangent chain

### 4.1 The RDMs' tangents

`dm1` and `dm2` are golden-file *inputs* — what PySCF returned — so nothing
in the tree computed them from the CI vector beside them, and no derivative
with respect to that vector had anything to be checked against.
`nevpt2.rdm_host_dm` closes that: it rebuilds both on the host from
`nevpt2.link_tables`' single-excitation tables, which gives a free
consistency check on every committed golden *and* the bilinear forms whose
tangents the gradient needs. With `|v_pq> = E_pq|c>`,

```
  dm1[p,q]      = <c|v_pq>                dm2[p,q,r,s]      = <v_qp|v_rs>
  dm1_dot[p,q]  = <u|v^c_pq> + <c|v^u_pq> dm2_dot[p,q,r,s]  = <v^u_qp|v^c_rs>
                                                             + <v^c_qp|v^u_rs>
```

— the two cross terms of a bilinear form, which is the shape of everything in
this section. The CI vector is deliberately **not** normalized there: these
are the plain bilinear forms, so a caller that wants the normalized state's
derivative supplies a direction already projected against `c`. It
materializes `n^2 ndet` doubles, so it is a small- and medium-CAS tool, not a
production path.

`buildRdmTangentsDevice` does the same for the device build's three outputs,
and **adds no kernel**: per tile it runs `produce` twice (once from the CI
vector, once from the direction) and each digest twice over the swapped
operands, then runs the `fdm2`/wedge step once on the tangent accumulators —
that step is linear in what it reads, so it maps tangents to tangents
unchanged. Both cross terms of a bilinear form need both directions'
`produce` output at once, so there is **no cheaper arrangement to look for**:
the per-tile working set is `2 n^4 + 3 n^2` doubles per determinant against
the plain build's `n^4 + 2 n^2`, which is why the plain build's tile floors do
not carry over ([`performance.md`](performance.md), "Tile floors", has the
tangent build's own, and "The tangent RDM build (`--rdm-tangent`)" what it
costs).

Only the two **exact** digests have a tangent path: `--cublas`, `--ozaki` and
`--fused-digest` are refused with `--rdm-tangent` at flag parsing, so a
tangent is never formed from an approximate digest.

### 4.2 The SC class energies' tangent

`energiesTangentDevice` needs **no new contraction logic**, and that is by
construction. At fixed integrals each class's per-perturber norm and
Dyall-Hamiltonian expectation value is a quadratic form in that class's
external integrals and **linear in the one active-space block per term** it
contracts them against. So the tangent is the same einsum chain with the
block replaced by its own tangent — and each tangent block is the same
`make_*` builder called with the tangent RDMs, because every builder is
linear in the RDMs too. The partition reuses the class energies' own per-slab
chains, `Delta` vectors and host packings; it transcribes no subscript
string.

What it does write is **three tangent-only builders, all for RDM-free
terms** — terms that are constant as the RDMs move and so absent from the
derivative: the hole-side metrics' Kronecker-delta terms, and the one-electron
term in `make_a25`. Two more cases are flags rather than builders: Sir's
purely one-electron norm term is switched off in the tangent, and Sijrs
contracts no RDM at all, so its tangent is **exactly** zero.

**The one actual derivative is the scalar finish's.** The integrals reach `E`
only through the three per-perturber scalars `N`, `H` and `Delta`, so
`Edot = sum_k (P_k Ndot_k + Q_k Hdot_k)` with Park Eqs. 36–37's multipliers
(§5.1). That host arithmetic is the only hand-differentiated expression on
the SC forward path, and `test/energy_finish/` checks it card-free.

**Euler holds for three classes here.** The three whose metric *and* Dyall
blocks are purely linear in the RDMs — Sr, Srsi, Srs — are homogeneous of
degree 2 in the CI vector. The other five are **affine**, not homogeneous —
four of them for the RDM-free terms listed above, and Sijrs because it reads
no RDM at all — so no Euler identity holds for them;
they are checked by the norm increment instead (a class norm is affine in the
RDMs, so displacing them by any `X` must change it by exactly the tangent at
`X` — two runs of the unmodified energy stage against one of the tangent).

### 4.3 PC: differentiating the solve, not the eigendecomposition

PC does not contract its perturbers down to two scalars; it solves in the
whole span of the class's internal functions, `E_t = -c b_t^T A_t^+ b_t` with
`A_t = K + Delta_t S` and `b_t = S x_t`
([`pc-nevpt2.md`](pc-nevpt2.md), "PC-NEVPT2 on the device: the design"). Only
`S`, `K` and `S_1` carry the CI vector, so with `x = A_t^+ b_t` — the solve
the energy already ran —

```
  Edot_t = -c (2 x^T bdot_t - x^T Adot_t x)
```

and **no eigenvector or eigenvalue derivative appears**. There is therefore
no degenerate-eigenvalue case to handle, and nothing here reruns the
eigensolve. Why that is legitimate is §7.

Two consequences shape the code. First, a matrix tangent is formed as one
*subtraction* of two builds rather than one build from the tangent RDMs,
because the hole-side metrics' RDM-free constants do not scale with the CI
vector. Second, **a class the gap or denominator check refuses is refused in
the tangent too** — through the same status the energy stage uses, never
reported as a zero derivative. A refusal at a *displaced* state, where the
metric is a different matrix, is a numerical `Error` naming the step: the
difference quotient would otherwise be formed from an energy that does not
exist.

**The orbital energies move here too, when asked** (#74): handed
`epsDot`, every class adds `sum_t R_t Deltadot_t`, `R_t = dE_t/dDelta_t`
read off the same solve — `c |g_t|^2` for a single coupling, `2 |g_u|^2 -
g_u . g_v` for Sijr/Srsi's pair — and Sijrs its SC term, exactly as §5.5's SC
tangent does. `nevpt2_pc_tangent`'s finite difference runs without it: it
differences the energy at the golden's fixed orbital energies.

**Euler holds for five classes here, against the SC tangent's three, and the
two are consistent.** The extra pair is Si and Sir. What makes those affine on
the SC side is machinery PC does not use — SC's Si reads an intermediate whose
one-electron term carries no RDM, and SC's Sir norm has a purely one-electron
term — whereas PC's one-body part enters only through `b_t = S x_t + S_1 h1_t`,
and `S_1` is linear in the RDMs. Both tangents agree on the other six.

## 5. Reverse mode: pseudodensities, and the CI source term

### 5.1 The amplitudes and multipliers already exist

A strongly-contracted class carries three numbers per perturber — the norm
`N_k`, the active-space part `H_k` of the Dyall expectation value, and the
orbital-energy sum `Delta_k` — and the energy's scalar finish divides them.
That division's reciprocal **is** Park Eq. 12's amplitude, so
`nevpt2.sc_amplitudes` computes nothing new: it reads `T_k` out of the finish
the energy stage already runs, including its vanishing-norm guard. Its three
derivatives are Park Eqs. 36–38 (`P_k = dE/dN_k`, `Q_k = dE/dH_k`,
`R_k = dE/dDelta_k`), and the identity `P_k N_k + Q_k H_k = E_k` (Eq. 40) is
what every assembled pseudodensity is checked against. `R_k` is the
multiplier of an orbital-energy sum; nothing consumes it yet
([`gradient-theory.md`](gradient-theory.md), §2.3 says what will).

### 5.2 One routine differentiates every class

A pseudodensity is "everything that multiplies this integral". Since `N_t`
and `H_t` are quadratic in the external integrals and linear in the active
block, the derivative with respect to an integral is a sum over the class's
terms and over each term's two integral slots, of the multiplier times *that
term with the slot dropped*. Dropping an external slot gives Park Eq. 41's
two-particle pseudodensity; dropping the active operand instead gives
Eqs. 42–43's intermediates, from which Eqs. 44–47's active-space
pseudodensities are built.

So each class is written here as the **list of its norm and `H` terms** — the
same subscript strings, coefficients and operands as the matching einsum calls
in `nevpt2.energy`, so the two read side by side — and one generic routine
both evaluates that list (giving the per-tuple `N_t` and `H_t` the multipliers
need) and differentiates it, by rewriting each term's subscripts per operand.
**No derivative is hand-transcribed.** Four classes need more than a term
list, and each says so where it happens: Sijrs has no active space at all
(its pseudodensity is written out by hand), Sijr and Srsi symmetrize pairs of
tuples (their multipliers come from a triangle and are scattered back), and
Sir reads one integral block on its diagonal, which an einsum output cannot
spell.

**And the result is checked exactly.** Contracting the assembled
pseudodensities back against their own operands must reproduce the class
energy — Park Eq. 40 — because the forms are homogeneous of degree 2 in the
integrals and degree 1 in the active blocks. These are algebraic identities
in exact arithmetic: a mis-transcribed index does not satisfy them. They are
`nevpt2_sc_pseudodensity`'s gate ([`testing.md`](testing.md), "The SC
pseudodensity identities").

The pseudodensities are **contracted back and freed as the walk reaches each
slab's and each class's end**. Nothing consumes them yet, and holding all
eight classes' at once would raise the peak for no reader; the place the
Z-vector source terms will plug in is where those contractions are taken.

### 5.3 One chain rule further in: from blocks to RDMs

The walk above leaves one accumulator `M_G = dE/dG` per metric and
Dyall-Hamiltonian block. Every block is *linear* in the RDMs — the same fact
forward mode rests on — so

```
  dE/d(rho) = sum over blocks G of (dG/d(rho))^T M_G
```

and `(dG/d(rho))^T` is just `G`'s own einsum chain with the output and the RDM
operand swapping roles, which is what the same subscript rewriting writes. No
derivative is hand-transcribed here either. The sweep is
**reverse-topological**, because blocks read blocks: a class's three-body
hole intermediate reads the two-body one, and Srs' active intermediate reads
its three-body RDM-side partner. Out comes Park Eq. 48's five coefficient
tensors: `dE/d(dm1)`, `dE/d(dm2)`, `dE/d(dm3)`, `dE/d(f3ac)`, `dE/d(f3ca)`.

### 5.4 From RDMs to determinants: the sigma vectors

The last step would need `d(dm3)/dc` and `d(dm4)/dc` — objects of
`n^6 ndet` and `n^8 ndet`. Park stores the four-body one and reports its size
as the blocker. **Contracted against its coefficient tensor on the fly it
never exists**: what comes out is one amplitude per determinant, a vector of
length `ndet`. That is the same trick `f3ac`/`f3ca` already play for the
energy, where the 4-RDM is never built either — here the contraction is
against a coefficient tensor instead of against the integrals.

`nevpt2.sigma_vector` is deliberately written against **plain tensors** — no
gradient-specific type — so it is testable in isolation and independent of
whatever computes the coefficient tensors. Two entry points:

- the **3-body** sigma, three nested reverse walks over the link tables,
  which needs no temporary at all;
- the **4-body contracted** sigma, which done the same way would cost
  `(nla + nlb)^4 n` terms per determinant — not a computation anyone runs — so
  it factors through an intermediate instead: a 2-body sigma against the CI
  vector for each `(a, s)` pair, then a 2-body sigma of each of those against
  the coefficient tensor's rows. **Two kernel passes through one
  `(n^2, ndet)` intermediate**, at `n^2 (nla + nlb)^2` terms per determinant
  per pass instead of `(nla + nlb)^4 n` in one.

Its 4-body dagger is a separate entry point; the 3-body dagger is the same
kernel with the coefficient tensor's index order reversed.

**And `dm1` and `dm2` need no sigma of their own.** The determinant basis has
a fixed particle number, so the trace of a one-body generator acts as the
scalar electron count on any vector in it — *identically in `c`*, not just at
the state. Differentiating that relation makes a `dm2` coefficient a `dm3`
coefficient on one diagonal over `N_elec`, and a `dm1` coefficient one on two
diagonals over `N_elec^2`. One 3-body sigma therefore carries all three, and
there is no 1- or 2-body entry point.

### 5.5 `y_I`, and the gate

`scCiSourceDevice` is those three steps in one pass over the class energies'
own slab walk: the pseudodensity walk, the sweep down to the five coefficient
tensors, and five sigma calls (one 3-body; a forward and a dagger for each of
the two `f3` orders). Its result is `y_I = dE/dc_I` in the CI vector's own
shape, plus — free, since it rode on exactly that walk — the Eq. 40–47
numbers the identity entries report.

It adds **no tile-sized buffer**: its accumulators are `n^2 + n^4 + 3 n^6`
doubles, independent of `--tiles`, and the 4-body sigma's intermediate spans
the whole determinant axis by construction, since its second pass gathers
from anywhere in it ([`performance.md`](performance.md), "`--tiles` on the CI
source term (`--gradient-source`)").

The gate is §2's duality, plus the Jacobian column by column where `ndet`
allows. What `y` is *for* is the right-hand side of the decoupled CI Z-vector
equation ([`gradient-theory.md`](gradient-theory.md), §2.2) — the one
derivative of `E` the Lagrangian needs in determinant space.

**The orbital energies move with the CI vector, and both modes carry it.**
`Delta_t` reads `e_core` / `e_virt`, which are the generalized Fock's
diagonal; `f` is built from the active 1-RDM, so

```
  e_p = h^eff_pp + sum_ut gamma_ut W_p[u, t],   W_p[u, t] = (pp|ut) - (1/2)(pu|tp)
```

for `p` core or virtual — affine in `dm1`, with `W` read off four of the
sidecar's tied arrays (`orbitalEnergyResponse`, `nevpt2.gradient_plan`).
Handed `W` (scCiSourceDevice's `eps`):

- **reverse**: every class scatters its `R_t` (Park Eq. 38) onto the orbital
  energies its `Delta_t` reads — `deltaAdjointInto` probes the class's own
  `delta` construction with unit vectors, so no transpose is transcribed —
  and the summed `dE/de` becomes one more `dm1` coefficient,
  `sum_p (dE/de_p) W_p`, folded in before the sigma like the rest;
- **forward**: `energiesTangentDevice`'s `epsDot` is `e_dot = <dm1_dot, W>`,
  and every class adds `sum_t R_t Deltadot_t`, `Deltadot` being its own
  `Delta` construction read at `e_dot` — Sijrs included, whose `Delta` is
  four orbital energies and whose tangent is otherwise zero.

The duality stays exact, now on the complete functional. **It cannot check
the new term**, though, and that is the lesson of the omission it replaces:
both modes read the same `W`, so a wrong `W` — or the term dropped from both
— cancels between them exactly as the omission did. What checks it is
outside the pair: the card-free `test/gradient_plan/` suites rebuild the
golden's `e_core` / `e_virt` from `W`, which pins every index, and
`normToEnergyDeltaDerivative`'s `R` is held to a difference quotient in
`test/energy_finish/`; end to end, the dump's `P y` is held against PySCF's
exact derivative, which an omission shared by both modes fails
([`testing.md`](testing.md), "The orbital energies' CI response" and "The GPU
as the source-term provider"). `--sc-tangent`'s finite difference does not
license this term: it differences the energy at the golden's fixed orbital
energies, the functional it was built for, and runs the tangent without
`epsDot`.

A case with no sidecar cannot form `W` — the golden does not carry `(pp|ut)`
for `p` outside the active space — so there both modes hold the orbital
energies fixed, and `--gradient-source` prints which functional it
differentiated.

### 5.6 PC: the solve's adjoint (#74)

PC's reverse mode is the tangent of §4.3 read the other way. Every PC class
energy is a sum over its tuples of quadratic forms in the eigenbasis rows
`y_t = U^T b_t` of the solve the energy ran, so for **any** perturbation of
`S`, `K`, `S_1` and the integrals

```
  dE = sum_t ybar_t . (U^T db_t)  +  < U mgg U^T, dA_t >,     ybar_t = dE/dy_t
```

with `mgg` the `(m, m)` weight of the tuples' forms — `dA^+ = -A^+ dA A^+`
again, so again no eigenvector or eigenvalue derivative. Everything the
source terms need follows: `dE/dT` and `dE/dt_1` (the slab einsum's adjoint
in its basis operand), `dE/dK = U sym(mgg) U^T`, `dE/dS = sym(U (dE/dT)^T)
+ U sym(sum_t Delta_t mgg_t) U^T`, `dE/dS_1 = U (dE/dt_1)^T`, and
`R_t = dE/dDelta_t`. The per-tuple half is host arithmetic
(`nevpt2.energy_finish`'s `pcAdjointSingle` / `pcAdjointPair`, unit-tested
against an energy that inverts the full matrix), and Sijr's and Srsi's
exchange rows are folded the way their two layouts store them.

**No new sweep.** PC's `K` is a16 / a22 / a3 / k27 / a7 / a9 / a12 + a13 —
SC's own Dyall blocks — and its `S` and `S_1` are permutations of the RDMs
or of SC's hole-side metrics, so `dE/dK`, `dE/dS`, `dE/dS_1` go down to the
five Eq. 48 coefficient tensors through §5.3's per-block adjoints unchanged,
Sir's block patterns term by term with the same `adjointTerm`. From there
`y_I` is §5.4's sigma assembly, and §5.5's orbital-energy response folds in
the same way, `R_t` scattered by `deltaAdjointInto` through each PC class's
own `Delta` (PC walks Sijr, Srsi and Sir in its own tuple order,
`nevpt2.gradient_plan`'s `pcSijrDelta` / `pcSrsiDelta` / `pcSirDelta`).
Sijrs is SC's class, here as in the energy.

**Off the state's symmetry sector too.** `y_I` is read on determinant axes,
which leave the sector, and there `S`'s kept range rotates at first order.
That costs nothing: `S`'s dropped modes are linear dependencies among the
operator strings, so `K` annihilates them as well, and the rotation's term
`x^T K N (...)` vanishes. The lifted modes are second order — the reason a
finite difference needs `CI_STEP_PC`'s small step (§7), not a term of the
derivative.

**Refused where the energy refuses.** A class the gap or denominator check
refuses carries that `PcClassResult` out of every entry point; `y` is then
not formed at all, and the digests' adjoint and the matrix of `Y` are
Numerical Errors naming the class. Never a sum with a zero in it
(`test/pc_adjoint/`).

The gate is §2's duality against `pcTangentsDevice` with `epsDot` — four
seeded directions, the state, and the Jacobian column by column at CAS(4,4)
(`nevpt2_demo --gradient-source --pc`). It cannot see a term both modes
leave out, so block2's exact `P y` stands beside it (§6.1).

## 6. The orbital source term

Park Eq. 26's `Y_xy = dE/dkappa_xy` comes out of the same slab walk — over
the **whole** `kappa` space, and in **both directions**, which is the point:

| | |
|---|---|
| the **adjoint** | `<kappa, Y>`: the pseudodensities of §5.2 paired with the direction's action on the very arrays they are conjugate to, plus the orbital energies' response against the multiplier of their sum |
| the **tangent** | `Edot(kappa)`: the same class terms rerun under the product rule, finished with the Eqs. 36–37 multipliers |

One derivative taken two ways, so holding them equal checks the assembly —
exact arithmetic, no step size. The central differences of the unmodified
energy stage along `exp(h kappa)` beside them — one for an active rotation,
one for a rotation inside the core space and one inside the virtual space —
are there because the two directions **share** their operand tangents (the
device kappa action, "a Dyall block's tangent is its own builder called with
the dotted integrals", and the promotion rule below), and a wrong tangent
cancels between them (§2).

**What moves, exactly.** A rotation at fixed CI coefficients moves every
stored array that carries an index it touches. For the **active–active** block
that is the active-space integrals and every external block's active slots;
for the core, virtual and mixed blocks it is every other slot of every block
as well, plus the one-electron blocks' core dressing (they are effective
one-electron matrices, and the core density is fixed in the MO indices, so
differentiating promotes its two tied labels too) and **the orbital
energies**, which are the generalized Fock's diagonals. All of that is here.
Held fixed are

- the **RDMs** — correct, and the derivative's own definition: they are
  functions of the CI vector, which this derivative holds fixed;
- the **`f3ac`/`f3ca` digests**, in the integral halves the like-for-like
  duality compares — *not* a property of the exact derivative. The digests
  are the active two-electron integrals contracted with the 4-RDM, so the
  exact `dE/dkappa` carries a term through them. It is computed beside each
  integral half rather than inside it, in both modes: forward by rebuilding
  the digests, reverse through that contraction's **transpose** — the 4-RDM
  against a six-index conjugate density (both below) — and `Y` as the
  callers read it includes it;
- and nothing else. The **orbital energies** were held fixed too while only
  the active block existed, because their response needs integral blocks the
  golden does not carry ([`gradient-theory.md`](gradient-theory.md), §1.3,
  which is explicit that "the active block needs only what the golden already
  carries" is the easy mistake here). The `*.gradgold` sidecar does carry
  them, and the full-space entry point reads them, so `deps` is now part of
  both modes.

So the integral halves are exact except through the digests, and the digests'
term completes them — while the core–core and virtual–virtual blocks are
exact without it, since neither the digests nor the orbital energies move
there at all: no active two-electron index and no orbital energy responds to
a rotation that stays inside one inactive space. That is what makes Eqs.
56–57 below a division of an exact numerator.

**The digests' term, forward mode (#88).** A digest's `kappa`-tangent is the
digest builder run on the `kappa`-dotted active integrals — the pattern #27
set for the CI tangent ("a Dyall block's tangent is its own builder called
with the dotted integrals"). At a fixed CI vector the whole device RDM build
is **linear** in `h2e` — consume, digest, the `fdm2` correction and the wedge
each carry exactly one integral factor — so `buildRdmsDevice` at
`h2e_dot(kappa)` *is* `(f3ac_dot, f3ca_dot)`, with `h2e_dot` the same tensor
the frozen-digest tangent's Dyall blocks read (the intra-active action plus
the sidecar's general part; its `dm3` output reads no integral and is
dropped). The energy's response to that displacement at fixed integrals is
`energiesTangentDevice` with every other RDM tangent zero — the digests enter
`make_a16`/`make_a22` as plain addends, so only the classes' `H` side moves
and no orbital energy does. So `scDigestRotationTangentDevice` adds **no
kernel**: one plain RDM build and one tangent slab walk per direction, at the
plain build's working set and so its tile floors
([`performance.md`](performance.md), "Tile floors").

It is reported as **its own term**, beside the frozen-digest tangent and not
folded into it, so the duality keeps comparing like with like and stays
exact; the exact `Edot(kappa)` is their sum. Neither mode can check it — it
is the one term they share by both leaving it out — so it is held to things
outside the pair: on the card, a finite difference of the energy with only
the digests rebuilt along an active direction (`--kappa-full`'s check 6),
and an exact zero along every intra-space rotation, which moves no active
integral; offline, PySCF's exact directional derivative minus its
frozen-digest one, along directions `--dump-source-terms` writes into the
`.gradsrc` with both terms (`nevpt2_grad.py --edot-check`;
[`testing.md`](testing.md), "The digests' orbital response, forward mode").
That is the oracle the reverse half below is checked against. Like every
tangent path here it is formed from an exact digest only: `--kappa-full` and
`--dump-source-terms` refuse `--cublas`, `--ozaki` and `--fused-digest` (§8).

**The digests' term, reverse mode (#89): the transpose.** `Y` is probed basis
direction by basis direction, so carrying the term into it needs it as a
*pairing*, `<hbar, h2e_dot(kappa)>`, with `hbar = dE/d(h2e)` through the
digests computed once. Two steps, each already its own checked piece. The
slab walk's RDM sweep (§5.3, the one the CI source term runs) leaves
`G_o = dE/d(f3_o)` for the two orders. Then the build's f3 path, linear in
`h2e` at a fixed CI vector, is run **backwards**:

```
  forward, per order o              its transpose, run in reverse
  per tile: W  = eri . L2           once:     Gt, fdm2bar = wedge^T(G_o)
            f3_pre += W . L2                  hbar += fdm2^T(fdm2bar, dm3)
  once:     fdm2 = eri . dm3        per tile: Wbar = Gt^T . L2        (one DGEMM)
            f3_o = wedge(f3_pre, fdm2)        hbar += consume^T(Wbar, L2)
```

`L2` — the doubly-excited wavefunctions the forward build contracts the
4-RDM out of — is re-produced tile by tile by the plain build's own
`produce` kernel, so the 4-RDM against its six-index conjugate density is
taken without the 4-RDM ever existing, at the plain build's per-tile working
set ([`performance.md`](performance.md), "Tile floors" and "The digests'
transpose"). `nevpt2.rdm_build`'s `buildDigestAdjointDevice` (its `:adjoint`
partition) drives four new kernels in `src/rdm/f3_transpose.cu` — the
wedge's adjoint onto the pre-wedge digest and onto `fdm2`, `fdm2`'s onto the
integrals, `consume`'s onto the integrals — one thread per output element,
no atomic, on both backends; `nevpt2.gradient`'s `scDigestAdjointDevice` is
the walk plus that call. `scFullRotationDevice` / `scFullRotationMatrix` take
the result as `digestAdjoint` and add the pairing to `<kappa, Y>` as a term
of its own (`digest`), kept out of the integral halves.

So there are **two dualities**, both exact arithmetic: the integral halves'
(what it always was), and the **full** one — `<kappa, Y>` with the
transpose's term against the frozen-digest tangent plus the rebuilt digests'
term — which `--kappa-full` gates beside it at the same 1e-7 and the dump
gates across the file it writes. The forward term is the oracle; PySCF's
exact `Y`, committed block by block (`--source-oracle`), is the independent
provider; and a nuclear gradient built from the result is held to `grad_num`
([`testing.md`](testing.md), "The digests' orbital response, reverse mode").
The transpose reads only the tile count of the build options: it has one
route, whichever digest built the RDMs it is handed.

**The core, virtual and mixed blocks are one promotion rule, not a table of
derivatives.** Every stored two-electron block is a slice of the full MO
integral tensor in one layout, so the chain rule on it is: for each of the
four slots, replace that index by a general one and contract it with `kappa`.
Each right-hand side is one of the sidecar's one-general-index arrays, picked
by the promoted slot's partner and the other charge distribution — so the
whole half is four contractions per block, generated from the block's space
signature, with no derivative transcribed per class
([`gradient-theory.md`](gradient-theory.md), §3.1 and §3.3). Two places need
more than the rule: the effective one-electron blocks' core dressing (where
the Coulomb pair's two promotions collapse and the exchange pair's do **not**
— they are different integrals, and writing the exchange as twice either one
is wrong in a way that survives even a rotation inside the core space), and
the orbital energies, whose response reads the generalized Fock and the
inventory's restricted rows.

**What checks it, since neither mode can.** Both modes read the same operand
tangents, so a wrong promotion cancels between them, exactly as a wrong
operand tangent does for the active block (§2). Three things stand in for the
finite difference there:

- every promotion rule's source array, restricted to the promoted slot's own
  space, must be the stored block under the permutation that rule uses — one
  identity per block per slot, which pins the array choice and the index
  order; and the effective one-electron matrix and the generalized Fock's
  diagonal must come back out of the dressing and orbital-energy rules' own
  sources;
- a rotation **inside** the core space, or inside the virtual space, keeps
  every index in its own space, so the displaced integrals are the stored
  blocks with those axes transformed on the host and the displaced orbital
  energies are the rotated Fock's diagonal — which means the unmodified
  energy stage can be rerun and differenced, exactly as for the active block;
- a **mixed** rotation cannot be displaced that way (its indices leave their
  spaces, which would need integrals with two general indices) and does not
  need to be: each promotion is one contraction over the whole general index,
  so the rows the two intra-space differences exercise are the same code the
  mixed rows run.

**Park Eqs. 56–57.** The orbital convention diagonalises the generalized Fock
inside the core space and inside the virtual space, so the energy reads it
there only on its diagonal and its off-diagonal response has to be pinned by
that convention instead: one multiplier per pair, and because the matrix is
diagonal on both spaces the stationarity condition collapses to a division by
an orbital-energy difference ([`gradient-theory.md`](gradient-theory.md),
§2.3). Those two blocks of `Y` are what it divides. Every such division is
singular where its divisor vanishes, and a degenerate pair is not
hypothetical: an abelian subgroup splits a degenerate set across irreps, so
both partners keep the same energy. The energy does not move along a rotation
mixing two irreps, so for such a pair the numerator vanishes too, the
multiplier is a `0/0` the constraint does not determine, and zero is the right
representative — while a degenerate pair whose numerator does *not* vanish is
a **numerical refusal**, not a number.

**The chain rule through the Dyall blocks needs no new term list.** Every
block builder is linear in the integrals it reads, so a block's tangent is
that same builder called with the dotted integrals and its integral-free
operands zeroed.

**Why `Y` as a matrix has to be probed.** The class energies are gauge
invariant under a *covariant* rotation of integrals and RDMs together
([`gradient-theory.md`](gradient-theory.md), §1.3), which makes the
directly-contractible part — the pseudodensities paired with one active index
left open — exactly **symmetric**, hence invisible to an antisymmetric
`kappa`. All of `Y_act`'s content therefore sits in the chain rule through the
Dyall blocks, which is a map *into* the integrals, not out of them, and
reverse mode does not run it backwards for free. So a matrix is assembled by
probing the same linear functional with basis directions, in chunks, one slab
walk per chunk; a caller who wants a *pairing* pays nothing extra. The
active-block check skips the matrix above a fixed number of basis directions,
and the full-space one probes only the index pairs its reader needs — the two
inactive blocks, for Eqs. 56–57 — both policies of the check, not caps in the
assembly.
Park Eqs. 44–47's active-space pseudodensities are not on this path
and are not completed into Eq. 35's RDM-conjugate form, which has no consumer
until the full `Y` needs the RDM side too.

### 6.1 PC's orbital source term (#74)

Along a rotation at fixed CI only the integrals move — PC's `S` and `S_1`
are RDM-only — so §5.6's identity gives the pairing

```
  <kappa, Y_pc> = sum_slabs <dE/dslab, slabdot>  +  <dE/dK, Kdot>
                + sum_t R_t Deltadot_t           +  <hbar_pc, h2e_dot>
```

with every dotted operand SC's: the slab and one-electron tangents are
:terms' `slabDots` (the promotion rule, §6), `Kdot` is the class's own `K`
builder at the direction's dotted integrals with the digests zeroed
(`pcClassMatrices` — "a Dyall block's tangent is its own builder"), the
orbital energies' response is §6's, and `hbar_pc` is the digests' transpose
(#89) fed PC's own `dE/d(f3ac)`, `dE/d(f3ca)` from Sr's a16 and Si's a22.
The forward tangent along the same direction is computed from the same rows
by separate per-tuple code (`pcRotationTangent*`: `ydot` from the dotted slab
through `T`, `mK = U^T sym(Kdot) U`), plus the digests' forward term — the RDM
build at `h2e_dot` through `pcTangentsDevice`. So PC has §6's two dualities:
the integral halves', and the full one with both digests' terms; the dump
gates both (§8). Neither can see an omission they share; block2's exact `A`,
committed (`--pc --source-oracle`), and a nuclear gradient against `grad_num`
can, and are held ([`pc-nevpt2.md`](pc-nevpt2.md), "The source terms by
reverse mode"). Park Eqs. 56–57 stay offline, as for SC.

### 6.2 The density-fitted orbital source term (#165, #168), and which binary runs it

**Both of `Y`'s terms over the full `kappa` range are in tree on a
density-fitted case: the frozen-digest term (#165) and the digests' own
(#168).** A `*_df` golden is a
different calculation — every integral is `Σ_L B B` — so its `Y` needs the
general-range inventory *of the fitted integrals*.
[`gradient-theory.md`](gradient-theory.md), §3.8, settles that this needs no
new derivation: the metric is absorbed into `B` and never sees `kappa`, so
§3.1's promotion rule acts on one `B` factor at a time, and every
one-general-index array is `(x p | q r) = Σ_L B[L,x,p] B[L,q,r]` — the
promoted strip `B[:, x, P]` against the block `B_QR`, with the three tied rows
reading a diagonal of `B_cc` or `B_vv`. What it needs is the whole of `B`: the
golden's four blocks and the two the NEVPT2F1 sidecar adds (#164).

So `nevpt2.gradient`'s `dfGeneralIntegralsDevice` assembles `B[L,p,q]` over
the MO range and **contracts the same `GeneralIntegralsDevice` the
conventional sidecar fills**, one contraction over `L` per array, on the
device. Everything downstream — the promotion table, the dressing and
orbital-energy rules, both modes, Eqs. 56–57 — runs on it unchanged, and the
energy stage reads the fitted golden's own four-index blocks (a `--df` golden
carries them beside `B`, exactly as `nevpt2_demo` reads them,
[`reference-data.md`](reference-data.md), "Density fitting"). That is a
deliberate economy of *code*, not of memory: the arrays are formed at the
conventional inventory's sizes, which the validation cases afford (their cost
is [`performance.md`](performance.md), "The density-fitted orbital source
term"). §3.8's statement that the DF gradient need never form a four-index
array is about what the derivation *reads*; the slab-wise contraction that
would honour it on the device — a strip of `B` dotted by `kappa`, contracted
against `B` per block — is not in tree. The file stays the three-index one.

**What is checked** is `--kappa-full`'s whole list (§6, §8). #165 landed it
on the frozen-digest half: the integral halves' duality and, with the
orbital energies' response added to both, the frozen-digest duality, at
1e-7; the promotion table
against the fitted golden's stored blocks and the sidecar's `h^eff` / `f` /
`e_core` / `e_virt`, plus one row of its own — the built inventory against
the four tied orbital-energy arrays the sidecar carries ready-made; the active
block against `--kappa-act`'s own entry point; the two intra-space finite
differences, which displace the fitted golden's blocks; and Eqs. 56–57 with
the degeneracy refusal. #168 adds the digests' own term, and with it every
check §6 lists for the integral-direct path: the **full** duality (the
transpose's term in `<kappa, Y>` against the rebuilt digests' term in
`Edot`) on the four seeded directions, the digests' term against its own
finite difference along an active direction (the golden's fitted `h2e`
rotated on all four axes, which is both `B_aa` factors rotated) in forward
mode and the transpose held to it, and an exact zero along the two
intra-space directions. The frozen-digest duality stays beside the full one.
The PASS line says `(density-fitted: both Y terms`. Numbers:
[`testing.md`](testing.md), "The density-fitted orbital source term".

**The digests' own term on the fitted path (#168): the same calls, and a
three-index transpose beside them.** §6's two-term shape carries over with
no new derivation (#161; [`gradient-theory.md`](gradient-theory.md), §3.8):
the fitted golden's `h2e` *is* `Σ_L B_aa B_aa`, so the forward term —
`buildRdmsDevice` at `h2e_dot(kappa)` — and the transpose —
`scDigestAdjointDevice`, leaving `hbar = dE/d(h2e)` — are the
integral-direct calls run on the fitted `h2e`, with `h2e_dot`'s general part
reading the built inventory's `g_xa_aa`. That is what `Y` carries
(`scFullRotationDevice` / `scFullRotationMatrix` with `digestAdjoint`), on
both the check and the dump.

The issue asked for a transpose that reaches the three-index `B` rather than
a four-index block. `hbar` is `n_act^4`, the transpose's own output, so the
four-index object the fitted path does not need is `g_xa_aa`
(`n_mo · n_act^3`), which `h2e_dot`'s general part reads. In `nevpt2.energy`'s
layout `h2e[p,q,r,s] = Σ_L B[L,p,r] B[L,q,s]`, and §3.1's rule on each `B`
factor gives `Bdot[L,t,u] = Σ_x (κ_xt B[L,x,u] + κ_xu B[L,t,x])`, so

```
  <hbar, h2e_dot(kappa)> = <Bbar, Bdot> = Σ_{x, t∈act} κ_xt D[x, t]
  Bbar[L,t,u] = Σ_{v,w} (hbar[t,v,u,w] + hbar[v,t,w,u]) B_aa[L,v,w]
  D[x, t]     = Σ_{L,u} B[L,x,u] (Bbar[L,t,u] + Bbar[L,u,t])
```

— the digests' term of `Y` as an `(n_mo, n_act)` matrix, out of `hbar` and
the strip `B[L, x, a]`, i.e. the golden's `B_ca`, `B_aa`, `B_va` and nothing
of the sidecar's (`nevpt2.gradient`'s `dfDigestSourceDevice`; the 4-RDM is
still never formed, since `hbar` comes from the tiled transpose).

**What is in tree, and why it is not the only route.** The three-index route
is computed and **gated against** the four-index pairing — on `--kappa-full`'s
four directions, along check 6's active direction, as one more exact zero on
the two intra-space directions, and on every direction the dump writes — at
the duality's 1e-7, the same functional in a different association. It does
not *replace* the four-index pairing in the `Y` that is written: that `Y` is
probed basis direction by basis direction through the integral halves, which
already read the whole inventory built from `B` (#165's economy of code, not
of memory, above), so `g_xa_aa` is formed either way and routing one of
`Y`'s terms around it would save nothing at these sizes. Where it matters —
a fitted gradient that never forms the inventory — `D` is the digests' term
ready-made; the inventory-free integral halves it would sit beside are not in
tree (the slab-wise contraction above). Cost: [`performance.md`](performance.md),
"The density-fitted orbital source term".

**The binary decision: `nevpt2_sc_pseudodensity`, with no `--df` flag.** A
fitted case is known by its sidecar's NEVPT2F1 magic, and `checkAgainst`
already refuses a fitted sidecar beside an exact golden and the reverse. The
reasons, in order:

- **Everything else this binary runs already reads a fitted golden.** The
  identities and `--kappa-act` read the four-index blocks a `--df` golden
  carries; only the general-range inventory differs, and it is one function
  away from the conventional one. A second binary would duplicate
  `:kappa_full_check`'s six checks for one changed line.
- **The orbital source term reads whole blocks.** The general action and the
  two finite differences are written for whole blocks (§6, "Integral-direct
  only" in `gradient.cppm`), so `nevpt2_df_demo`'s slab walk — the reason the
  CI checks moved there (#162) — buys this path nothing.
- **A flag would restate the file.** `--df` beside a NEVPT2F1 sidecar says
  nothing the magic does not, and beside a NEVPT2D1 one it would be a
  contradiction to refuse; `densityFitted()` is read off the magic and never
  off the array names, and that is the one switch.

`--dump-source-terms` on a fitted sidecar writes the exact `Y` with both
terms (#168), `y` with the orbital energies' response read from the same
sidecar, and `edot_kappa` / `edot_frozen` / `edot_digest` along its seeded
directions, read back bit for bit — the file #166's nuclear gradient is built
from. Until #168 it was refused, since half a dump is refused rather than
written. Two combinations stay refused as *unsupported*:

- **`--source-oracle`**: there is nothing to compare against.
  `reference_data/sc_source_oracle.py` writes PySCF's (or block2's) exact
  source terms of the *conventional* calculation only, so no fitted
  `.srcgold` exists. `nevpt2_grad.py` has a fitted path since #166
  (`--df`), but nothing writes its source terms out as a `.srcgold`. The
  fitted `Y`'s independent provider is the nuclear gradient built from the
  dump against the fitted `grad_num` (#166, #167), and `nevpt2_grad.py --df
  --source both`'s componentwise comparison with PySCF's DF-NEVPT2 quotients
  beside it ([`testing.md`](testing.md), "The density-fitted gradient
  against the numerical one").
- **`--pc --dump-source-terms`**: PC's rotation tangent over the fitted
  inventory has no gate of its own in tree (`--kappa-full` is SC's), and a
  dump is never written ungated.

`README.md`, "Which
binary takes which flag", and `test/cli/` follow the decision: no binary
parses `--df`, and `nevpt2_df_demo` refuses `--kappa-act`, `--kappa-full` and
`--dump-source-terms`, naming `nevpt2_sc_pseudodensity`.

## 7. Where the path is differentiable

Three conditions, each enforced in code rather than assumed.

**`PC_MIN_GAP`: the pseudo-inverse's rank condition.** `A^+` in §4.3 is a
*pseudo*-inverse, and a pseudo-inverse is differentiable only where the rank
is **locally constant** — otherwise `dA^+` does not exist and the tangent
expression is meaningless. Local rank constancy is exactly what the
singular-metric convention already enforces: the modes the cut drops are
**exact null modes of the state's metric** — not of every CI vector: some are
null for every singlet (spin), some are fixed in number for every singlet while
*which* combinations are null turns with the state (a rank bound), some are
null only across states sharing the molecule's full symmetry
([`pc-nevpt2.md`](pc-nevpt2.md), "What the cut drops, and how it moves") — and
`PC_MIN_GAP` demands decades of separation between them and the smallest kept
mode, so no kept mode can sink under the cut under a small change of the state
([`pc-nevpt2.md`](pc-nevpt2.md), "The singular-metric convention", has the
constant and the measured spectra). The metric is a Gram matrix, so its
dropped eigenvalues never move at first order, and the rank condition the
*first* derivative needs holds along any direction. The solve is the S-cut
restricted inverse, whose derivative has extra terms; at first order they drop
because `b_t` lies in the kept range and because, **at an eigenstate**, `K`
annihilates `S`'s dropped space. Off an eigenstate that second fact fails, so
the two-term tangent is the derivative only at the state itself — which is all
the gradient needs, and not enough for a difference of it
([`hessian-theory.md`](hessian-theory.md), §6.2). So the
tangent is valid **precisely on the states the energy stage already accepts,
and only there**: a refused class is refused in the tangent, never given a
zero derivative (§4.3), and `test/pc_tangent/` locks that in. The adjoint
(§5.6, #74) rests on the same condition and refuses the same classes, by
the gap and by the denominators alike (`test/pc_adjoint/`).

**The CI direction must stay inside the state's symmetry sector, and the path
must be the unit sphere.** A *random* CI direction raises the metric's rank at
every step size — the displaced state has modes the state's own symmetry
forbids — and the PC stage then refuses it at any `h`. That is the rank
condition failing as the derivation says it must, and it is a property of the
direction, so the check builds its direction from the spin-free generators
that commute with `S^2`. Separately, the hole-side metrics' RDM-free constants
are the hole RDMs only *at unit norm*, so a straight-line displacement puts
them on the wrong scale and shows up as metric eigenvalues of the wrong sign;
the normalized path removes that, and with the direction orthogonal to the
state both paths have the same tangent at `h = 0`, so nothing is lost. With
both in place one `Sir` mode still grows, fast enough in `h` that it falls
under the cut once the step is small enough — so the step has an upper bound
as well. The measurements behind those three — direction, path, step bound —
are in [`pc-nevpt2.md`](pc-nevpt2.md), "What the check can and cannot
displace". None of them is a limitation of the analytic tangent: all three
are constraints on how a *finite difference* of it may be taken.

The **offline** PC gradient meets the same upper bound on the step, in the
same place and for the same reason: `reference_data/nevpt2_grad.py --pc`
takes `y_I = dE_corr/dc_I` by differencing the PC energy along determinant
directions, so a step large enough to lift a dropped mode past the cut puts a
*first-order* error into the source term — one that no step ladder converges
away and that no residual in the Z-vector solve can see. Its step is
therefore its own constant rather than the SC one's, and the rank condition
above is re-checked at every displaced CI vector
([`testing.md`](testing.md), "The analytic gradient against the numerical
one", has the ladder, the gap collapse and what it cost before it was
caught).

**The active-space selection is not a stationarity condition.** A selection is
a discrete function of geometry: locally constant, and at the points where it
changes not differentiable at all. That is a hazard for the *numerical*
gradient rather than for anything in §4–§6, and the settled rule is
[`gradient-theory.md`](gradient-theory.md), §4.3: **in-plane displacements of
a planar `Cs` molecule only**, because an in-plane displacement preserves the
molecular plane and hence the irrep labels the active space is specified in,
while every out-of-plane component is zero by symmetry anyway. The rule is
enforced in code where the displacements are taken:
`reference_data/nevpt2_numgrad.py` **refuses** a displacement whose detected
point group differs from the reference geometry's — and, at an unchanged
group, one whose core or active irrep labels differ — marking that component
as zero-by-symmetry rather than differencing it
([`reference-data.md`](reference-data.md), "The numerical gradient"). The two
gradient-validation molecules — LiF and acrolein — were committed for exactly
this ([`reference-data.md`](reference-data.md), "LiF and acrolein, the
gradient-validation pair").

## 8. How it is driven

Nothing above runs in a plain energy run. Each piece is a flag or a binary of
its own, and each prints one `PASS:` line a ctest entry matches
([`testing.md`](testing.md), "Two tiers").

| invocation | what runs |
|---|---|
| `nevpt2_demo --rdm-tangent`, `nevpt2_df_demo --rdm-tangent` | the tangent RDM build beside the plain one, checked against twice it |
| `nevpt2_demo --sc-tangent`, `nevpt2_df_demo --sc-tangent` | the SC total's tangent: the norms' affine increment, Euler, and the one-time finite difference |
| `nevpt2_demo --sc-hessian` | the SC total's second directional derivative `Eddot(u, v)` at fixed orbital energies (#153): Euler at second order, the norms' second-order increment, and one finite difference of `Edot` |
| `nevpt2_demo --sc-hessian-vector` | the vector `H_cc u` by reverse over forward at fixed orbital energies (#156): the duality `<v, H_cc u> == Eddot(u, v)`, `H_cc` column by column and its symmetry where `ndet` allows, and one finite difference of `y` |
| `nevpt2_demo --gradient-source --pc` | PC's `y_I` (§5.6) against `pcTangentsDevice`, the same three checks and the walk's PC total against block2's (#74) |
| `nevpt2_demo --gradient-source` | `y_I = dE/dc_I` in reverse mode, gated on the duality against the forward tangent, plus the Jacobian where `ndet` allows. With `--gradgold PATH` both modes carry the orbital energies' response (§5.5) and the sidecar is first checked against the golden (its `h^eff` and `W` must rebuild `e_core` / `e_virt`); without it they hold the orbital energies fixed. The PASS line names which |
| `nevpt2_df_demo --gradient-source` | the same duality and Jacobian over the density-fitted slabs (#162), always at **frozen** orbital energies: the demo reads no sidecar, and the PASS line says `frozen-orbital-energy mode` |
| `nevpt2_sc_pseudodensity` | the Eqs. 40–47 identities for all eight classes |
| `nevpt2_sc_pseudodensity --kappa-act` | the active–active orbital source term: adjoint against tangent, with a finite difference beside them |
| `nevpt2_sc_pseudodensity --kappa-full` | the same over the whole `kappa` space, out of the `*.gradgold` sidecar (`--gradgold PATH` overrides the derived path): the duality, the sidecar's promotion table against the arrays it reads, the active block against `--kappa-act`'s own entry point, two intra-space finite differences, and Eqs. 56–57; and `Edot(kappa)` in its two terms — the frozen-digest tangent and the digests' own (§6) — with the **full** duality against `<kappa, Y>` carrying the digests' transpose (#89), the digests' term held to its own finite difference along an active direction (forward and reverse) and to an exact zero along the two intra-space ones. On a `*_df` golden beside its density-fitted sidecar (#165, §6.2), the inventory is contracted out of the whole `B` and checked against the sidecar's carried tied arrays too; every check above runs, the digests' term included (#168), plus the frozen-digest duality kept beside the full one and the transpose carried onto the three-index `B` held to the four-index pairing; the PASS line names the fitted case |
| `nevpt2_sc_pseudodensity --dump-source-terms PATH` | not a check of the derivatives but their **export**: `Y_xy` over every MO pair (`scFullRotationMatrix`, out of the sidecar as `--kappa-full` reads it) and `y_I` (`scCiSourceDevice`, with the orbital energies' response out of the same sidecar), written to `PATH` as a `.gradsrc` — a NumPy `.npz` holding `kappa_source`, `ci_source`, the `ci` they were taken at, the partition and `e_corr`, all in the golden's own basis, and (#88) `edot_kappa` / `edot_frozen` / `edot_digest`: seeded directions — `--kappa-full`'s four full ones, one purely active–virtual, one purely core–active — and `Edot` along each in its two terms — then read back. `Y` carries the digests' transpose (#89). Its gate is that the matrix is the pairing, that each direction's exact `Edot` (frozen + digest) is `<kappa, Y>` off the matrix written, that `y`'s walk ran on this state and that the file reads back bit for bit. `reference_data/nevpt2_grad.py --source dump` builds the SC-NEVPT2 nuclear gradient from it, carrying both terms into its own active basis; `--source both` puts that gradient beside PySCF's and `grad_num`; `--edot-check` holds the directions' `Edot` against PySCF's exact directional derivative. On a `*_df` golden (#168, §6.2) the same file from the fitted inventory, with a sixth check: each direction's `edot_digest` against the transpose carried onto `B` |
| `nevpt2_sc_pseudodensity --source-oracle PATH` | the dump's two terms checked against an **independent** provider: PySCF's exact `A = Y − Y^T` and `P y`, committed as `golden/<case>.srcgold` for N2 CAS(4,4) and LiF (`reference_data/sc_source_oracle.py`), compared block by block. Every block must meet the measured tolerance: `source_compare.cppm`'s `kKnownGaps`, which pinned the §6 digest omission's blocks until #89, is empty (docs/testing.md, "The SC source terms against PySCF's, on a card"). Refused on a `*_df` golden: there is no fitted oracle (§6.2) |
| `nevpt2_sc_pseudodensity --pc --dump-source-terms PATH` / `--pc --source-oracle PATH` | the same two jobs for PC-NEVPT2's `Y` and `y` (§5.6, §6.1, #74): the dump marked `pc`, its four checks plus a fifth — the PC integral halves' duality on the six directions — and the oracle against block2's `golden/n2_ccpvdz_cas44_pc.srcgold`. Both require the golden's PC fields and first hold the PC energy to them |
| `nevpt2_pc_tangent` | the PC tangent: Euler, and a Richardson-verified central difference. `--fd-steps` and `--fd-start-pow` set the sweep |

`--sc-tangent` and `--gradient-source` run on both demos (#162): `y_I` and
`Edot(u)` differentiate the CI vector at **fixed** integrals, so nothing in
them differentiates an integral and the density-fitted demo needs no second
copy of the blocks -- only the slab entry points every class already walks.
(The ORBITAL source term is the one that needs that copy; §3.8 of
[`gradient-theory.md`](gradient-theory.md) is its inventory, and it is
`nevpt2_sc_pseudodensity`'s on a density-fitted case too, §6.2.) They are two different
checks of the same derivative and so refuse each other; `--gradient-source`
has a `--pc` path (#74) on the integral-direct demo only, `--sc-tangent`
none.

**A flag a binary does not act on is refused, never ignored.** The gradient
binaries used to parse `--rdm-tangent` and do nothing with it; now one table
in `nevpt2.cli` says which binary implements which flag, and anything else is
an *unsupported* error naming the binaries that do. The matrix, and why each
refusal is there, is in [`README.md`](../README.md), "Which binary takes which
flag"; `test/cli/` holds it, host-only.

**Everything that builds a tangent runs the emitted or the native-fp64 BLAS
digest.** `--cublas`, `--ozaki` and `--fused-digest` are refused at flag
parsing by `--rdm-tangent`, `--sc-tangent` and `--gradient-source` (as a
combination), by `nevpt2_sc_pseudodensity`'s `--kappa-full` and
`--dump-source-terms` (as a combination: both build the digests' tangent
along a rotation, §6) and by `nevpt2_pc_tangent` (as flags it does not
implement), because a tangent built from an approximate digest is not the
derivative of anything the energy stage computed. `nevpt2_sc_pseudodensity`'s
other modes — the identities, `--kappa-act` and `--source-oracle` — are the
exception, and consistently so: they build no tangent RDMs — they read the
RDMs the build produced, whichever digest produced them — so they take any
digest the demos take.

**The reference data.** The gradient checks read the ordinary golden file; the
`*.gradgold` **sidecar** beside it carries the MO-integral arrays the
*orbital* gradient will need and the energy never reads — kept in a separate
file so the goldens do not grow for every reader that does not want them.
What it holds, which cases carry one, and how it is generated are in
[`reference-data.md`](reference-data.md), "The gradient sidecar" and "The
numerical gradient"; the inventory it implements is
[`gradient-theory.md`](gradient-theory.md), §3.3 and §3.5. `--kappa-full` and
`--dump-source-terms` read all of it — the core, virtual and mixed blocks of
`Y` are exactly what those arrays are for — and `--gradient-source` four of
its tied arrays and `h^eff`, for the orbital energies' response (§5.5); only
the cases that carry one can run those.

## 9. Where the numbers are

| you want | read |
|---|---|
| the identities' residuals, the finite-difference plateaus and the step at which round-off takes over, per case, with card and flags | [`testing.md`](testing.md), "The SC pseudodensity identities", "The SC tangent's finite-difference cross-check", "The active-rotation source term", "The full orbital source term", "The CI source term's forward-reverse duality" |
| how far the kernels' `atomicAdd` order moves a result run to run, and the envelope that spread is held to | [`testing.md`](testing.md), "The kernels' same-run self-variation" |
| which ctest entries check the PC tangent, their gate values and what they cost | [`testing.md`](testing.md), "The PC tangent against a finite difference" |
| the PC tangent's ratios, residuals, step constraints and crossover | [`pc-nevpt2.md`](pc-nevpt2.md), "The tangent with respect to the CI vector" |
| what the tangent build costs, and the tile floors it needs | [`performance.md`](performance.md), "The tangent RDM build (`--rdm-tangent`)", "Tile floors", "`--tiles` on the CI source term (`--gradient-source`)" |
| the sidecar's arrays and sizes, and the numerical gradient | [`reference-data.md`](reference-data.md), "The gradient sidecar", "The numerical gradient", "LiF and acrolein, the gradient-validation pair" |
| the orbital energies' CI response: how well `W` rebuilds the golden's orbital energies, and what a wrong pick or a dropped term misses by | [`testing.md`](testing.md), "The orbital energies' CI response" |
| a nuclear gradient built from the GPU's own source terms, against PySCF-sourced and `grad_num`, and which omission costs what | [`testing.md`](testing.md), "The GPU as the source-term provider" |
| the digests' orbital response in forward mode: the GPU's `Edot(kappa)` against PySCF's exact one, the ladders, the on-card checks and the negative control | [`testing.md`](testing.md), "The digests' orbital response, forward mode" |
| the digests' orbital response in reverse mode: the full duality, the source oracle with no known gap, the GPU-sourced nuclear gradient and the negative control | [`testing.md`](testing.md), "The digests' orbital response, reverse mode" |
| the density-fitted orbital source term's checks (both terms) and dump, and what it costs | [`testing.md`](testing.md), "The density-fitted orbital source term"; [`performance.md`](performance.md), "The density-fitted orbital source term" |
| the density-fitted nuclear gradient from the GPU's dump and from PySCF's DF-NEVPT2, against the fitted `grad_num` | [`testing.md`](testing.md), "The density-fitted gradient against the numerical one" |
| what the digests' transpose costs, and its tile floors | [`performance.md`](performance.md), "The digests' transpose", "Tile floors" |
| PC's source terms by reverse mode: the dualities, block2's oracle, the PC nuclear gradient from the GPU's terms, the refusals and the negative controls | [`pc-nevpt2.md`](pc-nevpt2.md), "The source terms by reverse mode" |
| the derivation this implements (its CASCI branch), and the CASSCF branch only the offline `nevpt2_grad.py --reference casscf` implements | [`gradient-theory.md`](gradient-theory.md) |
