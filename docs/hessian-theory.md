# Hessian theory: the NEVPT2 energy's second derivatives

What an analytic *second* derivative of this energy would require. The
first-derivative derivation — the Lagrangian on each of the two references, the
constraint sets, the Z-vector equations and solve orders, the integral
inventory — is [`gradient-theory.md`](gradient-theory.md), and this document
builds on it rather than restating it: §0's notation, §0's `P`/`Q`/`R` chain
rule, §1.3's active–active gauge, §2.1's source terms and §3's inventory are
assumed throughout and cited by section number.

Like [`gradient-theory.md`](gradient-theory.md), it is **derivation, not code**,
and it carries **no measured numbers**. Level 1 (§2–§6) has since been built:
what is in the tree, what is refused and what is deliberately not built is
[`gradients.md`](gradients.md), §1.1, and the "as built" notes below
(§9.1's second table among them) say where the code sits. Levels 2 and 3
(§7–§8) are not in the tree. Size formulas in `n_mo`, `n_core`, `n_act`,
`n_virt` and `ndet` are algebra and stay here; what any of them works out to in bytes or seconds belongs in
[`performance.md`](performance.md) or [`reference-data.md`](reference-data.md).

**The result, stated once up front.** The NEVPT2 energy is *exactly quadratic*
in the CI vector and multilinear in a fixed, small number of MO-integral
factors, composed with one scalar rational finish per external tuple. Every
second derivative of it therefore decomposes into

- the **existing** first-order machinery evaluated at new argument
  combinations — the tangent RDM builder at two arbitrary directions, the
  class einsums at second-order blocks, the digest path at second-order
  dotted integrals — and
- **one new hand-differentiated expression**: the 3×3 Hessian of the scalar
  finish in `(N, H, Delta)`, host arithmetic, with two exact identities of its
  own (§2.3).

That is the same structural argument that made the *first* tangent need "no
new contraction logic" ([`gradients.md`](gradients.md), §4.2), one order up,
and it is the reason a second derivative is a much smaller undertaking here
than the first one was. What is *not* small is anything downstream of the
parameter Hessian — the response solves and the derivative integrals a nuclear
Hessian needs (§7, §8).

## 0. Notation, and what is added to §0's

[`gradient-theory.md`](gradient-theory.md), §0's index convention and orbital
parameterisation hold unchanged. Three additions:

- `theta` denotes the whole parameter vector the energy depends on,
  `theta = (kappa, c)` — orbital rotations and CI coefficients.

  **The amplitudes `T` are not among them, but not for the reason the gradient
  gives.** [`gradient-theory.md`](gradient-theory.md), §0 drops them because the
  Hylleraas functional is stationary in `T`, so `dE/dT = 0` kills the term. That
  is a *first*-order argument and it does not extend: with `E` stationary in `T`,

  ```
    d^2 E / dR^2  =  E_RR + 2 E_RT . (dT/dR) + (dT/dR) . E_TT . (dT/dR)
  ```

  and only the `E_T d^2T/dR^2` term dies. **A second derivative needs the
  amplitude response `dT/dR`** — the same way an SCF gradient needs no orbital
  response but an SCF Hessian needs CPHF. It is not an extra unknown here only
  because of how each variant's amplitudes are obtained: SC's are **closed
  form** (Park Eq. 12), so eliminating them is exact and §2 differentiates the
  eliminated energy `E = -N^2/d` — which carries `dT` automatically; PC's come
  from a solve, and §6 differentiates that solve explicitly, reusing its
  factorisation. So `T` never appears as a parameter *as implemented*, and §1.1
  is where that matters.
- `u`, `v` are **directions** in parameter space. A second derivative is always
  written as a bilinear form `Eddot(u, v)`, never as a materialised matrix:
  §5.3 is why.
- The PC per-class prefactor that [`gradients.md`](gradients.md), §4.3 writes
  as `c` is written `nu` here, so that `c` means the CI vector throughout this
  document and only that.

The three scalars per external tuple `t` are §0's:

```
  N_t     the perturber norm            x_t^T S x_t
  H_t     the Dyall matrix element      x_t^T K x_t
  Delta_t the tuple's orbital-energy sum
  T_t     = -1 / (Delta_t + H_t/N_t)    the amplitude, so E_t = T_t N_t
```

and it is convenient to name the one combination that keeps recurring,

```
  d_t = N_t Delta_t + H_t        so T_t = -N_t / d_t  and  E_t = -N_t^2 / d_t
```

## 1. Three things called "the second derivative", and only one of them is small

The first-derivative document distinguishes the *source terms* `Y`, `y` from
everything downstream of them, and [`gradients.md`](gradients.md), §1 is a
table of what is in the tree on each side of that line. The same line matters
more at second order, because the three levels differ in cost by orders of
magnitude:

| level | object | what it needs |
|---|---|---|
| **1. the energy functional's parameter Hessian** | `d^2 E / dtheta dtheta` — the action of it on a direction, `(H_kk u_k + H_kc u_c, H_ck u_k + H_cc u_c)` | §2–§6. Existing machinery at new arguments, plus one 3×3. **No** solve, **no** derivative integrals, **no** constraint multipliers |
| **2. the Lagrangian's parameter Hessian** | `d^2 L / dtheta dtheta`, i.e. level 1 **plus** `Z . d^2 g / dtheta dtheta` over the constraints `g` | level 1, plus second derivatives of the *constraints*. On the CASSCF branch that is a **third** derivative of the CASSCF energy contracted with `Z` (§7) |
| **3. the nuclear Hessian** | `d^2 E / dR_A dR_B` | level 2, plus second-derivative integrals, plus the first-order response of every parameter to **every** coordinate: `3N` solves of the **reference's** Hessian, with level 1's operator consumed as `3N` Hessian-vector products against those responses (§8) |

Level 1 is the analogue of what the gradient path delivered: `Y` and `y` were
the energy's first derivatives in the parameters, with the solve left offline
on purpose ([`gradients.md`](gradients.md), §1). Level 1 is their second-order
partner, and it is what a second-order orbital optimiser, a linear-response
property or an excited-state gradient would consume. Levels 2 and 3 are the
offline half, and §8 argues that level 3's *oracle* — not its implementation —
is what makes it a different kind of project.

Only level 1 is derived in full below. §7 and §8 say precisely what the other
two add, so that the boundary is a stated decision rather than an omission.

### 1.1 Is an analytic nuclear Hessian possible at all — both variants, both references?

**Yes, for all four combinations.** It is unpublished, not impossible, and the
distinction matters: the reasons it has not been done are cost and the
availability of a good-enough alternative (§8), not an obstruction in the
theory. Three things have to hold, and it is worth separating them because only
the first is interesting.

**(a) The energy must be twice differentiable in `R`. Every condition for that
but two is one the *gradient* already needs.** The exceptions are the PC
metric's rank (the fourth row) and SC's empty perturbers (the last): a
dropped mode that a displacement lifts, and an empty perturber that a
displacement populates, are both harmless at first order and not, in
general, twice differentiable (§6.2, §5.6).

| condition | which branch | why second order adds nothing |
|---|---|---|
| no intruder: `Delta_t + H_t/N_t` must not pass through zero | both | NEVPT2's Dyall zeroth order is what makes this rare rather than endemic; where it fails, the *energy* is singular, so there is no gradient either |
| the CASSCF Hessian must be non-singular at the solution | CASSCF | the implicit function theorem gives `theta(R)` as smooth as the stationarity conditions, which are `C^infinity`. **One derivative and two need the same condition**; this is `gradient-theory.md`, §2.7's conditioning statement, unchanged |
| the selection must be locally constant, and the pseudocanonical divisions non-degenerate | CASCI | `gradient-theory.md`, §4.1–§4.3 and §2.3's refusal. A selection change is a kink: the *gradient* is already wrong there |
| the PC metric's rank must be locally constant | PC only | **second order only, along a symmetry-breaking displacement.** The cut is a threshold cut, and what it drops are exact null modes of *this state's* metric — spin, rank-bound and spatial-symmetry modes — not modes null for every CI vector (§6.2). A Gram metric's dropped eigenvalues never move at first order (`P S_u P = 0`), so the gradient's condition is unchanged. At second order the rank stays constant along every direction that keeps the state's spin and full symmetry, a totally symmetric displacement's response included, and along every orbital direction whatever its symmetry, because `S` is built from the RDMs alone (§6.5); along one that leaves the symmetry — a non-totally-symmetric displacement's CI response — a class whose dropped modes include spatial-symmetry ones lifts at `h^2` (`L_uv != 0`), and the energy is not twice differentiable along that pair |
| no root crossing | both | `gradient-theory.md`, §4.4 |
| no empty perturber populated by two displacements with different Dyall energies | SC only | **second order only.** At a geometry with point-group symmetry, a tuple whose integrals vanish by symmetry has `N_t = 0`; a non-totally-symmetric displacement populates it at first order, and its energy is then homogeneous of degree 2 but a quadratic form only when the contracted function's energy `H_t/N_t` is the same along every combination of the displacements (§5.6). The gradient is `C^1` there and exact; the Hessian is not a bilinear form |

The PC row went the other way only in part. Along totally symmetric
displacements PC's restricted inverse is twice differentiable for the same
reason it is differentiable in the CI vector, but in the non-totally-symmetric
blocks of a nuclear Hessian a class with spatial-symmetry dropped modes has no
second derivative, and §9.1 refuses that (class, pair). The last row went the
other way for SC alone: PC's energy is a quadratic form in `b_t` at fixed
`A_t`, with no Rayleigh quotient, so an empty tuple costs it nothing.

**(b) Every derivative object must be computable in closed form.** All of them
are: second-derivative AO integrals (standard, and in PySCF); the parameter
Hessian (§2–§6); the constraints' second derivatives (§7); the `3N` reference
response solves — and the reference-side machinery here is **established, not
hypothetical**: analytic CASSCF second derivatives and the coupled-perturbed
CASSCF equations are published and implemented, including the treatment of the
redundant variables the active–active gauge (§1.3 of
[`gradient-theory.md`](gradient-theory.md)) puts in the response equations
([`references.md`](references.md), "Derivative theory").

**(c) The amplitude response must be available**, which §0 explains is needed
at second order even though it is not at first. It is: SC's amplitudes are
closed form, so the response is differentiation rather than a solve; PC's come
from a solve whose factorisation is reused (§6.1). Neither variant acquires a
new linear system for `T`.

So the four combinations differ in *work*, not in possibility:

| | CASCI on RHF orbitals | CASSCF orbitals |
|---|---|---|
| **SC** | possible. Carries §2.3's selection multipliers and §4's hazard class, so the smooth branch's Hessian is not the PES Hessian near a selection change | **the cleanest of the four.** Park's Lagrangian as written, standard CP-CASSCF response, closed-form amplitudes |
| **PC** | possible; both columns' and both rows' caveats at once | possible, and the extra algebra over SC is *smaller* at second order than at first (§6.3) — the cost is the per-tuple re-solves, not new derivation |

One thing that is **not** a blocker, despite looking like a candidate: the
eigendecomposition PC runs per class is never differentiated (§6.1), so no
degenerate-eigenvalue case arises. A vanishing perturber norm was listed here
as a second one, and that was wrong for SC at second order. `E_t` does tend
to zero with `N_t`, but a perturber vanishes with `N_t` and `H_t` together,
so `d_t` vanishes too. The finish's second partials then grow as `1/N_t`
(§2.1), and the energy of an empty tuple populated along two directions is
twice differentiable only in the case §5.6 names.

The one qualification worth stating plainly: on the **CASCI-on-RHF** branch the
active-space selection makes the potential energy surface only *piecewise*
smooth, so an analytic Hessian there is the Hessian of one smooth branch rather
than of the physical surface. That is a property of the reference choice, not of
NEVPT2 or of second derivatives — the same is true of the gradient — but it is
the reason the CASSCF column is the one to build toward if nuclear second
derivatives are ever the goal.

## 2. The scalar finish's Hessian: the only new calculus

§0 of [`gradient-theory.md`](gradient-theory.md) observes that the chain rule
through the scalar finish is "the only place in the whole derivation where real
calculus (rather than bookkeeping) happens on the energy side", and that
everything downstream is `P`, `Q`, `R` contracted back against integrals and
densities. That observation is what makes this section short: at second order
the *same* sentence holds, with the 3×3 Hessian of the finish in place of its
gradient.

### 2.1 The six second partials

Differentiating `E_t = -N_t^2 / d_t`, `d_t = N_t Delta_t + H_t`, twice, and
expressing the result in the amplitude `T_t` (dropping the `t` subscript):

```
  E_NN = (2 T / N) (1 + Delta T)^2
  E_NH = (2 T^2 / N) (1 + Delta T)
  E_ND = T^2 (3 + 2 Delta T)
  E_HH = 2 T^3 / N
  E_HD = 2 T^3
  E_DD = 2 N T^3
```

(a subscript `D` is `Delta`). These are the second-order partners of Park's
Eqs. 36–38, and like them they are reference-independent: they differentiate
the finish, which does not know how its orbitals or its CI vector were made.

Two structural relations fall out, each a free card-free check on the
arithmetic:

```
  E_ND = N E_NH + Q                    Q = T^2 = E_H, Park Eq. 37
  E_DD = N E_HD = N^2 E_HH
```

**Prefer the `d` form below for implementation.** Three of the forms above
carry an explicit `1/N`, and `N` is a perturber norm the energy stage already
guards for vanishing (`normToEnergy`'s `NUMERICAL_ZERO`). The singularities are
**removable** — `T ~ N` as `N -> 0`, so `E_HH = 2T^3/N` is really `-2N^2/d^3`
and tends to zero — and the two forms agree to round-off at every `N > 0`,
small ones included; the failure is at `N = 0` **exactly**, where the `T` form
evaluates `0/0` and returns a NaN while the `d` form, at `H != 0`, returns
zero. The guard exists for `N = 0`, so that case will be hit, though with
`H = 0` as well ("What "removable" does not cover", below). In `d` the same six partials
are `1/N`-free:

```
  E_NN = -2/d + 4 N Delta/d^2 - 2 N^2 Delta^2/d^3
  E_NH = 2 N/d^2 - 2 N^2 Delta/d^3
  E_ND = 3 N^2/d^2 - 2 N^3 Delta/d^3
  E_HH = -2 N^2/d^3
  E_HD = -2 N^3/d^3
  E_DD = -2 N^4/d^3
```

Note that **none of Park's first derivatives has this problem** — `P`, `Q`, `R`
are `1/N`-free as written — so it is new at second order, and it is the second
reason (after §2.2) to carry `(N, d)` rather than `(N, H, Delta)` and `T`.

**What "removable" does not cover.** The limit above is at fixed `H != 0`. A
real perturber does not vanish that way: `N_t` and `H_t` are both quadratic
forms in the same function, so they vanish together, `H_t ~ eta N_t` with
`eta` that function's Dyall energy, and `d_t = N_t (Delta_t + eta) -> 0`.
Along that ray `T -> -1/(Delta + eta)` stays finite and the second partials
grow as `1/N`, `E_NN = -2 eta^2 / (N (Delta + eta)^3)` and likewise
`E_NH`, `E_HH`. Their contraction with `Ndot`, `Hdot` (each of order
`sqrt(N)`) stays finite and depends on the direction. At `N = 0` exactly the
`d` form is `0/0` too. So the guard's zero is the right value of the finish's
partials there, and the tuple's second-order content is §5.6's sixth term.

### 2.2 The `(H, Delta)` block is rank one, so there are four new numbers, not six

`E` reads `H` and `Delta` **only** through `d = N Delta + H`. Hence
`E_H = E_d`, `E_Delta = N E_d` — which is exactly why Park's `R_t = N_t T_t^2`
is `N_t` times his `Q_t = T_t^2`, a relation the tree already carries — and at
second order the whole `(H, Delta)` 2×2 is the outer product of `(1, N)` with
itself:

```
  [ E_HH  E_HD ]  =  E_dd  (1, N) (x) (1, N),        E_dd = 2 T^3 / N
  [ E_HD  E_DD ]
```

So the finish's Hessian has **four** independent entries — `E_NN`, `E_NH`,
`E_ND`, `E_dd` — not six, and only one of them, `E_dd`, is genuinely new
information beyond what §2.1's two relations and `E_NH` already give. A
formulation that carries `(N, d)` rather than `(N, H, Delta)` makes this
automatic and is the one to write.

### 2.3 Eq. 40's second-order partner: two exact zeros

`E` is homogeneous of **degree 1** in `(N, H)` at fixed `Delta`: scaling
`N -> lambda N`, `H -> lambda H` scales `d -> lambda d` and
`E = -N^2/d -> lambda E`. Euler's identity for that is

```
  N E_N + H E_H = E        i.e.  P_t N_t + Q_t H_t = E_t
```

which **is** Park's Eq. 40 — the identity every assembled pseudodensity in the
tree is already checked against. Differentiating Euler once more, a
degree-1-homogeneous function's Hessian annihilates the ray:

```
  N E_NN + H E_NH = 0
  N E_NH + H E_HH = 0
```

Both reduce to the amplitude's defining identity `T (Delta + H/N) = -1`:
substituting §2.1's forms, each bracket collapses to
`1 + Delta T + H T / N = 1 + T(Delta + H/N) = 0`. So the second-order partner
of Eq. 40 costs nothing to derive and nothing to check, and it is the right
first test of the new arithmetic — two exact zeros, no tolerance on a limit, in
the slot `test/energy_finish/` already occupies card-free.

### 2.4 The same 3×3 differentiates the multipliers

`P`, `Q`, `R` are the finish's gradient, so their own tangents along a
direction are the 3×3 applied to the first-order scalar tangents:

```
  Pdot = E_NN Ndot + E_NH Hdot + E_ND Ddot
  Qdot = E_NH Ndot + E_HH Hdot + E_HD Ddot
  Rdot = E_ND Ndot + E_HD Hdot + E_DD Ddot
```

This matters for §4.2: reverse mode's whole pseudodensity walk is `P`, `Q` and
`R` contracted against integrals and densities, so differentiating the walk
needs the multipliers' tangents and nothing else new. **The one new expression
serves both modes.**

## 3. The RDMs are exactly quadratic, so their Hessian already exists

With `|v_pq> = E_pq|c>`, [`gradients.md`](gradients.md), §4.1 writes the RDMs
as bilinear forms, `dm1[p,q] = <c|v_pq>`, `dm2[p,q,r,s] = <v_qp|v_rs>`, and the
same shape holds for `dm3` and for both digests. Every one of them is therefore
a **homogeneous quadratic** function of the CI vector, from which three things
follow immediately:

```
  d^2(dm)/dc^2 [u, v]  ==  buildRdmTangentsDevice(opt, u, v, ...)
  d^3(dm)/dc^3         ==  0
  buildRdmTangentsDevice(opt, c, c, ...)  ==  2 * buildRdmsDevice(opt, c, ...)
```

The first is the point: the existing tangent builder takes **two arbitrary
vectors**, and the cross-term structure it computes is precisely the second
derivative's bilinear form. Nothing has to be written, and no kernel has to be
added, for the RDM half of the CI-block Hessian.

The third line is `--rdm-tangent`'s existing Euler gate
([`gradients.md`](gradients.md), §8). It is the `u = v = c` case of the first,
which means **the Hessian's RDM half is already under test in its degenerate
case** — and that the general case's own check is the one
`test/rdm/rdm_tangent_tests.cpp` already runs for a general direction, at a new
argument pair. That check is the file's `RdmSecondDerivativeTests` (#151): the
builder at two independent random directions `(u, v)` against a central
difference of the tangent build along `v` at `(c, u)`, symmetric under
`u <-> v` and the same about a second, unrelated base state.

The second line — the third derivative vanishing — is what bounds the whole
problem: there is no tower of higher-order builders behind this. The same
suite asserts it, as a mixed central difference of the tangent build along two
further directions.

## 4. The CI block, `H_cc`

### 4.1 Two terms, and what each is made of

`E = sum_t f(N_t, H_t, Delta_t)` with `f` §2's finish. `N_t` and `H_t` are
quadratic in `c` (each contracts one active-space block, and every block
builder is linear in the RDMs — [`gradients.md`](gradients.md), §4.2); `Delta_t`
is **affine** in `c`, through the generalized Fock's diagonal and hence the
active density (§4.3 below, and [`gradients.md`](gradients.md), §5.5). So the
chain rule at second order has exactly two terms:

```
  Eddot(u, v) = sum_t [ P_t N''_t[u,v] + Q_t H''_t[u,v] + R_t Delta''_t[u,v] ]
              + sum_t ( Ndot_t[u], Hdot_t[u], Ddot_t[u] )
                       . Hess f_t .
                       ( Ndot_t[v], Hdot_t[v], Ddot_t[v] )
```

- **The first sum is Park's own multipliers against second-order blocks.** The
  second-order blocks are the *same* `make_*` builders at §3's second-order
  RDMs, because each builder is linear in the RDMs. The RDM-free terms have
  `N'' = H'' = 0` — a term constant in the RDMs is absent from the first
  derivative and from the second alike — so the tangent-only builders
  [`gradients.md`](gradients.md), §4.2 describes need no second-order partners.
- **The second sum is §2's 3×3 sandwiched between two copies of the
  *first-order* per-perturber tangents.** Those scalars already exist: they are
  what `energiesTangentDevice` forms internally to assemble
  `Edot = sum_t (P_t Ndot_t + Q_t Hdot_t + R_t Ddot_t)`.

So `Eddot(u, v)` is: the tangent RDM build at `(u, v)`, at `(c, u)` and at
`(c, v)`; the existing block builders and class einsums on all of them; and
§2's 3×3 in the scalar finish. **No new contraction logic, no new kernel, one
new host expression.**

The guard's skipped tuples add nothing here, on every golden §5.6's census
covers. A tuple whose external integrals vanish (`x_t = 0`) has
`N_t = H_t = 0` at every CI vector, so no CI direction can populate it. The
census finds every guarded tuple of those cases to be either one of these or
a real perturber below the guard, which the guard holds at zero near the
state. A guarded tuple with `x_t != 0` and an exactly zero perturber, empty
because of the state, would be populated by a CI direction that breaks the
state's symmetry, and would need §5.6's term in `H_cc` too. None of the
census's goldens has one.

One API consequence worth stating, because it is the only place the second
derivative is not simply "the first-order path called again": the second sum
needs direction `u`'s and direction `v`'s per-perturber tangents *at the same
time*, so the slab walk must carry **three** tangent sets — `u`, `v` and the
second-order one — rather than being run twice. The per-slab working set grows
accordingly, which means the tangent build's tile floors do not carry over and
must be re-measured ([`performance.md`](performance.md), "Tile floors", is
where that number lives).

In the tree (#153) this is `nevpt2.energy`'s `energiesHessianDevice` (the
`:sc_hessian` partition: the tangent's class walks carrying the four RDM
sets, through block builders shared with `:sc_tangent`), fed by
`nevpt2.rdm_build`'s `buildRdmHessianTangentsDevice` (one sweep over
`(c, u, v)`, §4.4) and finished by `nevpt2.energy_finish`'s
`normToEnergyHessianSandwich`, at fixed orbital energies only. Its gate is
`nevpt2_demo --sc-hessian`: §4.3's Euler, the norms' exact second-order
increment and §9's one finite difference of `Edot`
([`testing.md`](testing.md), "The SC Hessian's finite-difference
cross-check").

### 4.2 `H_cc u` as a vector: reverse-over-forward

The scalar `Eddot(u, v)` is enough to *gate* a Hessian but not to use one. The
full vector `H_cc u` comes from differentiating reverse mode's walk along `u`,
and the same linearity argument applies at every step:

- The walk accumulates `g = (g1, g2, g3, gac, gca) = dE/d(each RDM)`, each
  entry a sum of `P`, `Q`, `R` against integrals and densities
  ([`gradients.md`](gradients.md), §5.2–§5.3). Its tangent `gdot` is the same
  einsums at tangent inputs, with the multipliers' tangents from §2.4 — which
  is the only new ingredient.
- The sigma finish contracts `g` against the RDMs' own CI derivative. That map
  is bilinear in `c` (§3), so differentiating it along `u` gives **two** sigma
  calls instead of one:

```
  H_cc u  =  sigma(gdot[u], c)  +  sigma(g, u)
```

Both are the existing `ciSourceVector` entry point at different arguments.

**The gate is exact.** For any `v`,

```
  <v, H_cc u>  ==  Eddot(u, v)
```

with no step size in it — the direct analogue of the first-order duality
`<u, y> == Edot(u)` that [`gradients.md`](gradients.md), §2 makes the whole
validation strategy, and it has the same property: the two sides share no code
past the slab walk. Where `ndet` allows, the same comparison column by column
localises a mis-transcribed index, exactly as `--gradient-source` does today.

In the tree (#156) this is `nevpt2.gradient`'s `scCiHessianVectorDevice`:
one walk carries the plain accumulators and their tangents along `u` (each
class's active blocks rebuilt at the RDMs' tangents with `nevpt2.energy`'s
own builders, the per-slab `Pdot`/`Qdot` from `normToEnergyMultiplierTangent`,
and the class's RDM sweep run on both), at fixed orbital energies only. Its
gate is `nevpt2_demo --sc-hessian-vector`: the duality against
`energiesHessianDevice`, the columns and `H_cc`'s symmetry at CAS(4,4), and
one finite difference of the GPU's own `y` with the orbital-energy mode
matched on both legs ([`testing.md`](testing.md), "The SC Hessian-vector
product's checks").

### 4.3 Euler at second order — and why it is a frozen-orbital-energy identity

For a class homogeneous of degree 2 in `c`, `H c = y`, so

```
  Eddot(c, u)  ==  Edot(u)
```

for every `u`. This is the strongest free check available, because it holds the
**new** 3×3 against the **already-gated** `Edot(u)` along a direction no seeded
pair reaches — the same role Euler plays for the first tangent
([`gradients.md`](gradients.md), §2). It is available for the three SC classes
and five PC classes §4.2 and §4.3 of that document identify, and for no others:
the affine classes have no homogeneity to exploit at either order.

**But it is an identity only at fixed orbital energies.** The generalized Fock
is built from the zeroth-order density, which carries the constant 2 on the
core diagonal beside `gamma^(1)` on the active block
([`gradient-theory.md`](gradient-theory.md), §0). So `f` is *affine* in `dm1`,
`Delta_t` is affine in `c`, and the moment the orbital energies are allowed to
respond — the sidecar-fed mode of [`gradients.md`](gradients.md), §5.5 — even
the three otherwise-degree-2 classes stop being homogeneous and the identity
acquires correction terms.

This is not a new restriction: the first-order Euler checks already run at the
golden's fixed orbital energies, because `--sc-tangent` takes no sidecar. It
does mean the second-order check inherits the same mode split, and that **the
Euler check and the orbital-energy response must not be switched on
together** — stating which of the two a given invocation is in is part of the
check, as `--gradient-source`'s PASS line already does for the first-order
case.

### 4.4 What it costs, in `n` and `ndet`

[`gradients.md`](gradients.md), §4.1 gives the per-tile working set in doubles
per determinant: `n^4 + 2n^2` for the plain build against `2n^4 + 3n^2` for the
tangent, the `n^4` term being `produce`'s output and the `n^2` terms the
accumulators.

A naive reading of §4.1 would multiply that by four, since `Eddot(u,v)` wants
four RDM sets — `(c,c)`, `(c,u)`, `(c,v)` and `(u,v)`. **It does not, and this
is the one piece of cost algebra worth having before any code.** `produce` is
run per *vector*, not per RDM set, and the four sets involve only **three**
distinct vectors. So the `n^4` term is

```
  3 n^4 + O(n^2)      Hessian      (produce at c, u, v)
  2 n^4 + 3 n^2       tangent      (produce at c and the direction)
    n^4 + 2 n^2       plain build
```

— about **1.5× the tangent build**, not 4×, with the `n^2` term growing with
the number of simultaneous accumulator sets. Whether that fits is a
measurement and belongs in [`performance.md`](performance.md), "Tile floors";
the point here is that the scaling in `n` does not change, so no case that can
run `--rdm-tangent` is ruled out by an order of magnitude.

**`H_cc`'s own size.** §5.3 argues the orbital blocks exist only as an action.
`H_cc` is `ndet x ndet`, so the same is true of it above small `ndet` — and the
threshold is one the tree already knows: a materialised `H_cc` is affordable
exactly where `--gradient-source`'s column-by-column Jacobian check is
affordable today ([`gradients.md`](gradients.md), §2), and the column-by-column
Hessian check inherits that bound for the same reason. Above it, `H_cc u` and
nothing more.

## 5. The orbital blocks, `H_kc` and `H_kk`

### 5.1 Two sources of curvature in `kappa`, and only one of them is the energy's

Write the energy as a function of the MO integrals, `E(kappa) = Ehat(I(kappa))`
with `I` standing for the whole integral set and
`C(kappa) = C exp(kappa)` ([`gradient-theory.md`](gradient-theory.md), §0).
Then the chain rule separates the two origins cleanly, and the separation is
the design:

```
  d^2 E / dkappa^2 [u,v]  =  <Ehat' , I''[u,v]>  +  I'[u]^T Ehat'' I'[v]
```

- **The first term is the integrals' own curvature, and it reuses the
  machinery `Y` is built from — not the matrix `Y`.** `Ehat'` is the energy's
  *first* derivative with respect to the integrals — the pseudodensity set
  that the existing orbital source term is already built from and already
  validated as, the digests' own response (#89) included. `I''` is a purely
  one-electron-and-two-electron *transformation* question: for
  `I(kappa) = U^dag I U` with `U = exp(kappa)`, `I''` collects both `exp`'s
  `kappa^2/2` term and the terms where two of an integral's indices are
  rotated once each. No new energy-side machinery — but the two halves of
  `I''` read very different data, and on a conventional case one of them
  reaches arrays the gradient sidecar does not carry (the enumeration below,
  and §5.4).
- **The second term is the energy's own curvature in the integrals**, from the
  fact that each `N_t`, `H_t` contracts several integral factors, so the two
  derivatives can land on different ones. This is the genuinely new term, and
  it is a finite sum over which factor each derivative hits.

Every piece of either term is an existing contraction with one or two
integral factors replaced by dotted ones — exactly the shape #88's forward
digest tangent already has, "that build at the dotted integrals". So the
orbital Hessian's terms are enumerated rather than invented, and the
enumeration below is the cost estimate.

**The parameterisation's own second-order term.** With `U = exp(kappa)` and
`kappa = s u + t v`, the mixed derivative `d^2 U / ds dt` at zero is

```
  W = (u v + v u) / 2          symmetric, since u and v are antisymmetric
```

so the second-order change of anything that depends on the orbitals has two
parts, and the split between them is what decides §5.4:

```
  z''[u,v] = z'[W] + z^x[u,v]

  z'[W]      every slot rotated once by W: §3.1's promotion rule, ONE
             general index, evaluated at a symmetric argument
  z^x[u,v]   two DIFFERENT slots rotated once each, one by u and one by v,
             plus u <-> v: TWO general indices
```

for every `kappa`-dependent input `z` the energy reads — the external integral
blocks `X`, the active `h1e` / `h2e`, the digests `f3(h2e)`, and the orbital
energies `eps` (which at fixed MO densities are affine in the integrals, their
tied core and active density labels being rotated slots like any other —
`src/gradient/gradient_general.cppm`'s header writes their first-order rule).

Two facts about the parameterisation follow, and both matter to whoever
checks this term. It exists **because `Y != 0`**: NEVPT2 is not stationary in
`kappa`, so `z'[W]` does not drop the way the same term drops from a CASSCF
Hessian at convergence. And it is **parameterisation-dependent**: a ladder
that displaces along `exp(s u) exp(t v)` instead of `exp(s u + t v)` has the
mixed derivative `Eddot(u,v) + (1/2) Edot([u,v])` (Baker–Campbell–Hausdorff;
`[u,v]` is antisymmetric, so the extra term is the gradient's own pairing), and
is checking a different bilinear form.

**The five terms.** Write each tuple's scalars as the class energies build
them — `N_t` and `H_t` quadratic forms in the tuple's external integrals `x_t`
(`src/gradient/gradient.cppm`'s header), `H_t`'s active-space operand a Dyall
block `K` that is **linear** in `(h1e, h2e, f3)` (every `make_*` is; §5.2 has
`f3` linear in `h2e`), and `Delta_t` linear in `eps`. The chain rule through
`E = sum_t f(N_t, H_t, Delta_t)` at fixed `c` then has exactly five terms
**over the tuples the energy stage keeps**: `N_t` reads no active integral,
`K` has no second derivative in the integrals, and `Delta_t` none in `eps`.
The chain rule assumes `f` is twice differentiable at each tuple's
`(N_t, H_t)`, and at the tuples the guard skips (`N_t = H_t = 0`) it is not.
Those tuples add a **sixth term**, K6, and on some direction pairs make
`Eddot_kk` not a bilinear form at all. §5.6 derives both. An earlier version
of this paragraph said "there is no sixth", and #158's ladders showed it was
wrong.

| # | term | what it is | existing contraction it reuses | arrays it reads |
|---|---|---|---|---|
| K1 | **same-slot curvature** (§5.1's first term, half one) | `<Ehat', z'[W]>`: the full first-order pairing at `W` | `scFullRotationDevice`'s pairing, either mode — `adjoint` = `ext + act + eps` plus the digests' `<hbar, h2e'[W]>` (#89), or `tangent` plus #88's digest term — at `W` instead of an antisymmetric `kappa` | **exactly `Y`'s**: the sidecar's eighteen integral arrays (§3.3's seventeen rows, `(xc\|cc)` on its two slices), `h`, `heff`, `f`, plus the golden's stored blocks for the intra-active part |
| K2 | **cross-slot curvature** (§5.1's first term, half two) | `<Ehat', z^x[u,v]>` | the same pairings — `<D, X>` per external block, `<M_G, G>` per active block, `<hbar, h2e>` for the digests, `R_t` against `Delta_t` — at a new input | **two-general-index arrays**: §5.4's table. **Not in the sidecar** |
| K3 | **the finish's curvature** (§5.1's second term) | `sum_t (Ndot, Hdot, Ddot)_t[u] . Hess f_t . (Ndot, Hdot, Ddot)_t[v]` | the forward tangent's per-tuple `Ndot_t` / `Hdot_t` (`assembleSlab`'s `normDot` / `hDot`, before they meet `P` and `Q`) and `Ddot_t` (the class's `deltaOf` at `deps`), at `u` and at `v`; §2's 3×3 between them | the sidecar only: first-order dotted inputs `z'[u]`, `z'[v]` |
| K4 | **both external slots, one per direction** (second term) | `sum_t [ P_t N_t^{uv} + Q_t H_t^{uv} ]`: every norm and H term with its two `x` slots filled by `X'[u]` and `X'[v]`, symmetrised, `K` undotted | the class term lists under the product rule (`:terms`' `evalTermsTangent`), with the two slots' dotted operands drawn from different directions | the sidecar only |
| K5 | **one external slot and the Dyall block** (second term) | `sum_t Q_t [ H terms with one x slot at X'[u] and K at K'[v] ] + (u <-> v)` | the same evaluator; `K'[v]` is the forward tangent's own block tangent — `make_*` at `h1e'[v]`, `h2e'[v]` (`rotationTangentBlocks`) — plus `f3'[v]`, `buildRdmsDevice` at `h2e'[v]` (#88) | the sidecar only (`h2e'` reads `g_xa_aa`, `h1e'` reads `heff`) |

So `Eddot_kk(u, v)` is **five terms, every one an existing contraction at new
arguments, plus §5.6's K6 on the guarded tuples**, and the new code is two
*input generators* — the promotion at a symmetric argument (K1) and the
two-slot promotion (K2) — plus §2's 3×3. K6 is K4's own cross tables at
different multipliers. Per direction pair `(u, v)` that is:

- **one slab walk holding both directions' first-order dotted inputs**, for
  K3–K5 — the same "several tangent sets at once" shape §4.1 found for
  `H_cc`, with the same consequence for its working set;
- **one first-order pairing at `W`** (K1) and **one at `z^x[u,v]`** (K2) —
  which can share a walk, since both are linear in their input and pair the
  same pseudodensities;
- **the digests' share**: K3 and K5 need `f3'[u]` and `f3'[v]`, one #88
  rebuild per *direction*, reusable across every pair that direction enters;
  K1's and K2's is `<hbar, h2e'[W] + h2e^x[u,v]>` with #89's `hbar`, which is
  direction-independent and built **once per state**. So reverse mode is the
  cheaper of the two for the first term's digest part.

Three things K1 needs that the first-order path does not do, each read off the
code that would be reused:

- **The entry points refuse a symmetric argument.** `makeRotation` and
  `makeGeneralRotation` (`src/gradient/gradient_rotation.cppm`) return
  `err_config` for a `kappa` that is not antisymmetric, and `W` is symmetric by
  construction. The promotion rules themselves are the product rule and hold
  for any matrix; the refusal is the only thing in the way of most of them.
- **Three places drop a traced pair's promotions because they cancel against
  an antisymmetric `kappa`**: `h1e_v_Sr`'s subtracted trace `sum_b (rb|bt)`
  (`accumGeneralOneElectron`'s step 3), the hoisted `V2tr` trace of `Sir`
  (`slabDots`, `rotBlockInfo`) and, in the `Sir` class, the statement that
  `v2tr` does not move. At `W` those parts do **not** vanish and have to be
  restored. What *is* invariant is a traced pair's dependence on the rotated
  projector `U P U^T`, whose second order is `(W P + P W) - (u P v + v P u)`
  — K1's traced-pair terms plus K2's on the same pair — and that vanishes when
  both `u` and `v` are active–active. A free exact zero for the restored terms,
  and the reason neither half may be dropped alone.
- **The matrix `Y` cannot stand in for the pairing.** `scFullRotationMatrix`
  returns `Y`'s antisymmetric part; `<Ehat', z'[W]>` pairs the symmetric `W`
  with the *symmetric* part of the unsymmetrised integral-side matrix — the
  part that function's own comment calls invisible to an antisymmetric
  `kappa`. K1 is the pairing run at `W`, never a contraction with a stored `Y`.

And one thing K3 needs: its sandwich is **nonlinear per tuple**, so the
per-tuple `Hdot_t[u]` must carry the digests' share *before* the 3×3 meets it.
Today #88's term is a separate walk reduced to one scalar per class; K3 needs
the two merged per tuple.

### 5.2 The digests are linear in `h2e`, so there is no second-order digest kernel

The one place a genuinely new kernel might have been expected turns out not to
need one. `src/rdm/rdm_build.cppm` records that at a fixed CI vector
`f3ac`/`f3ca` are **linear** in `h2e` — "consume, digest, fdm2 and the wedge
each carry one integral factor" — which is the property #89's transpose rests
on. Linearity means

```
  d^2 f3 / dh2e^2  ==  0
```

so every second derivative through the digests is either

- `f3(d^2 h2e / dkappa^2)` — the **plain** digest path run once at §5.1's
  second-order dotted integrals, or
- the tangent build at `(c, u)` with dotted integrals —
  `buildRdmTangentsDevice` handed `h2e'` in place of `h2e`, #88's pattern
  one order up — which is `H_kc`'s digest term in its forward orientation.

The digests contribute no new tensor machinery to the orbital Hessian at all.
A reverse side that carried a *CI* tangent of the digests back onto `h2e`
would need #89's transpose applied to *tangent* digests rather than a
second-order transpose — again because of linearity — and §5.5 is why
`H_kc` is not computed in that orientation: in the one it is computed in,
the digests' orbital response goes through the 4-body sigma's own `h2e`
operand and needs no transpose at all.

### 5.3 Sizes: action only, and the asymmetry that follows

`Y` is an `n_mo x n_mo` matrix, and
[`gradient-theory.md`](gradient-theory.md), §2.1 argues every block of it has a
consumer. `H_kk` is `n_mo^2 x n_mo^2`, and `H_kc` is `n_mo^2 x ndet`. Neither
can be materialised at the sizes this tree runs: the orbital block alone is
`n_mo^4` doubles, which at the committed cases' `n_mo` is *exactly* the size of
the full MO-integral dump that
[`gradient-theory.md`](gradient-theory.md), §3 opens by declining — the reason
the gradient sidecar carries a block list instead. Whatever makes a full
integral dump intolerable for a case makes a materialised `H_kk` intolerable
for the same case, at the same `n_mo`.

So the orbital blocks exist **only as an action on a direction**, and the
deliverable is a matrix-free operator, not a matrix. Two consequences:

- the exactness budget changes shape. `Y` could be checked block by block
  against an independent provider (`--source-oracle` against PySCF's exact
  `A = Y - Y^T`). An operator cannot be; it can only be checked on directions.
  That is why §4.2's duality and §9's symmetry check carry more weight at
  second order than they do at first.
- the Hessian-vector **ladder** becomes the validation pattern of choice for
  the parts no identity reaches. There is precedent in the tree already:
  `reference_data/tests/test_cp_casscf.py` validates
  `pyscf.mcscf.newton_casscf.gen_g_hop`'s `h_op` by Richardson-verified
  Hessian-vector ladders ([`reference-data.md`](reference-data.md), "The
  analytic gradient", has the measured spreads). The same construction applies
  to level 1's operator.

### 5.4 Does the orbital Hessian need anything the sidecar does not carry?

[`gradient-theory.md`](gradient-theory.md), §3 exists because this question had
a non-obvious answer for `Y`: the active–active block alone needed two arrays
no class energy reads (`(xa|cc)` and `(xa|vv)`, §1.3), and "assuming the active
block needs only what the golden already carries" is called out there as the
easiest mistake to make. The second derivative has the same trap, and it sits
where the expectation was that the answer would be cheap.

**The verdict.**

| block / term | needs anything the `.gradgold` sidecar lacks? |
|---|---|
| `H_kc` | **no.** One `kappa` derivative (§5.5's C1–C5): the first-order dotted inputs `z'[w]` (sidecar), the RDM tangent at `(c, u)` (§3), the tangent build at dotted `h2e` (§5.2), and the orbital energies' mixed response — whose `gamma` line reads the tied rows `deps` already does (`g_xa_cc`, `g_xc_ca`, `g_xa_vv`, `g_xv_va`) and whose Fock line, the generalized Fock's off-diagonal response to `gamma`, reads four untied rows the sidecar also carries (`g_xc_aa`, `g_xa_ca`, `g_xv_aa`, `g_xa_va`) |
| `H_kk`, K1 and K3–K6 | **no.** K1 reads exactly `Y`'s arrays at a symmetric argument; K3–K5 read only first-order dotted inputs (§5.1's table), and so does K6, which is K4's cross tables on the guarded tuples (§5.6) |
| `H_kk`, K2 | **yes**, on a conventional case: **twelve two-general-index arrays, whose union is the whole MO-integral tensor** (below). On a density-fitted case, **no**: the golden and its `NEVPT2F1` sidecar already carry all of `B[L,p,q]` |

So the expectation recorded here before the enumeration — that §5.1's first
term "plausibly needs nothing new" because `I''` is "built from those same
arrays" — holds for its same-slot half (K1) and is **wrong** for its cross-slot
half (K2). §5.1's *second* term, the one that looked like the risk, needs
nothing new at all.

**§3.1's rule, run at second order.** For every array the energy reads —
§3.2's two-electron table and, through their definitions, the `h^eff` slices
and the orbital energies — promote **two** slots, each to its own general
index, keeping the charge-distribution pairing. Two slots in the same
distribution give `(xy|QR)`, the other distribution's two labels kept; two in
different distributions give `(xP|yQ)`. The union over all arrays and all six
slot pairs is:

| array | shape | reached from |
|---|---|---|
| `(xy\|aa)` | `n_mo^2·n_act^2` | every block with an `aa` distribution: `h2e`, `Sr`, `Si`, `Sir`'s `(ri\|tu)`; `h^eff`'s Coulomb dressing on `h1e` |
| `(xy\|va)` | `n_mo^2·n_virt·n_act` | `Sr`, `Srs`, `Srsi`, `Sir`'s `(rt\|iu)`; `eps_r`'s active exchange |
| `(xy\|ca)` | `n_mo^2·n_core·n_act` | `Si`, `Sij`, `Sijr`, `Sir`'s `(rt\|iu)`; `eps_i`'s active exchange |
| `(xy\|cv)` | `n_mo^2·n_core·n_virt` | `Sijrs`, `Sijr`, `Srsi`, `Sir`'s `(ri\|tu)`; `eps_r`'s core exchange |
| `(xy\|cc)` | `n_mo^2·n_core^2` | **no class block** — `h^eff`'s core exchange `(pj\|jq)` on its core-indexed slices (`h1e_v_Si`, `h1e_v_Sir`) and `eps_i`'s `(ij\|ji)`, each with `p` and `j` promoted. In full: `j` and the free core label are different indices |
| `(xy\|vv)`, tied | `n_mo^2·n_virt` | **no class block** — `eps_r`'s Coulomb `(rr\|jj)` and `(rr\|tu)` with the density pair promoted; the two virtual labels coincide |
| `(xa\|ya)` | `n_mo^2·n_act^2` | `h2e`, `Sr`, `Si`, `Srs`, `Sij`, `Sir`'s `(rt\|iu)` |
| `(xv\|ya)` | `n_mo^2·n_virt·n_act` | `Sr`, `Srs`, `Srsi`, `Sir` (both blocks), `Sijr`'s mirror `(xa\|yv)` |
| `(xc\|ya)` | `n_mo^2·n_core·n_act` | `Si`, `Sij`, `Sijr`, `Srsi`, `Sir` (both blocks) |
| `(xc\|yc)` | `n_mo^2·n_core^2` | `Sijrs`, `Sijr`, `Sij`; `h^eff`'s and `eps_i`'s core dressing |
| `(xc\|yv)` | `n_mo^2·n_core·n_virt` | `Sijrs`, `Sijr`, `Srsi`, `Sir`'s `(rt\|iu)` |
| `(xv\|yv)` | `n_mo^2·n_virt^2` | `Sijrs`, `Srs`, `Srsi`; `eps_r`'s exchange, on its tied slice |

(`(xa|yv)` and `(xv|ya)` are one array with `x` and `y` exchanged, as are
`(xc|ya)` / `(xa|yc)` and `(xc|yv)` / `(xv|yc)`; the `h^eff` and orbital-energy
routes also reach the ten class-block rows, so only the two they alone reach
are attributed to them above.)

**The first-order bound does not survive, and the eighteenth array comes
back.** §3.4's omission rested on "no integral the energy reads carries three
virtual labels, and promotion cannot create one" — a statement about the
*fixed* labels, which one promotion leaves at most two of. Two promotions turn
two labels general, and the general indices range over the virtuals: `Sijrs`'s
`(ir|js)` along two core–virtual rotations, or `Srs`'s `(rt|su)` along two
active–virtual ones, reads `(xv|yv)` at `x`, `y` virtual — integrals with
**four** virtual labels. `(xv|vv)`, the array §3.4 is built on never needing,
is `(xv|yv)`'s `y`-virtual slice. And the six `(xP|yQ)` rows alone are already
**every** MO integral: any `(pq|rs)` is `(xP|yQ)` with `x = p`, `P` the space of
`q`, `y = r`, `Q` the space of `s`. So no block list smaller than the full
tensor serves K2 for a general direction pair — §5.3's observation that a
materialised `H_kk` is the size of the full integral dump is, for K2's *input*,
literally true.

**What follows, per case kind:**

- **Conventional (`NEVPT2D1`) — the sidecar does not suffice, and the missing
  data is the full `n_mo^4` MO-integral tensor** that
  [`gradient-theory.md`](gradient-theory.md), §3 opens by declining. Nothing
  writes it today. Its natural writer is `generate_golden.py`'s
  `_gradient_arrays`, the `--gradgold` path, as one more named array (a full
  `ao2mo` transform; `ppaa` is already one of its groups) — **opt-in**, so the
  committed `.gradgold` files do not grow, and whether it is an array in a `D1`
  file or a sidecar of its own is the implementing issue's decision; either
  must pass `reference_data/golden_layout_check.py`. The alternative that stores
  nothing is AO-direct — each cross-slot integral is a four-index transformation
  with `C u` and `C v` in place of two coefficient columns — and needs AO
  integrals, which no file in `golden/` carries and the runtime does not
  compute. Which committed cases a full dump is affordable for is a size
  question for [`reference-data.md`](reference-data.md), not this document.
- **Conventional, restricted directions — the sidecar suffices when one of `u`,
  `v` is active–active.** Then one general index is active, and every row above
  collapses to a §3.3 array with its ties intact (`(xy|vv)` tied to `g_xa_vv`,
  `(xy|cc)` to the full `g_xa_cc`, `(xv|yv)` to `g_xv_va`, and so on down the
  table). That is the second-order counterpart of §1.3's active block, reachable
  today, and it is where the traced-pair zero of §5.1 lives.
- **Density-fitted (`NEVPT2F1`) — the sidecar suffices.** The golden's `B_aa`,
  `B_ca`, `B_va`, `B_cv` and the sidecar's `B_cc`, `B_vv` (in full) are the
  whole `B[L,p,q]` ([`gradient-theory.md`](gradient-theory.md), §3.8). In
  three-index form `z^x` for a two-electron block is a product of two
  *first-order* dotted factors across the two distributions — #168's `Bdot`,
  one per direction — plus one *second-order* dotted factor within a
  distribution, `sum_xy (u_xp v_yq + v_xp u_yq) B[L,x,y]`, which reads `B`
  over the full range. All carried. The twelve arrays need never be formed —
  forming them would be the `n_mo^4` cost again — which makes K2 on a fitted
  case a three-index contraction, the slab-wise kind
  [`gradients.md`](gradients.md), §6.2 records as not in tree.

A Hessian path must still **refuse** what it cannot compute rather than return
a partial second derivative (§9.1) — and the refusal now has a definite
trigger: K2 on a conventional case without the full tensor, for any direction
pair with neither member active–active. An omitted K2 is invisible to the
forward/reverse duality (both modes would omit it) and to the symmetry check;
only a Hessian-vector ladder sees it.

In the tree (#154) K1 and K2 are `nevpt2.gradient`'s
`scOrbitalCurvatureDevice`, the I'' half of `Eddot_kk(u, v)` and nothing
else: each term is a **dotted input set** — every stored block, `h1e`, `h2e`
and the orbital energies already differentiated, plus `v2tr`'s — paired over
the existing walk in both modes (`:rotation`'s `ScDotted`; a direction
carrying one bypasses the promotion rule in `:terms`' `slabDots`). The
generators are the `:curvature` partition's:

- **K1** is the first-order rule at `W` (`dottedAtMatrix`): the promotion
  machinery with its antisymmetry refusal lifted, plus §5.1's restorations —
  `h1e_v_Sr`'s traced `b` promoted over its x-in-active rows too, and `v2tr`'s
  dot taken as the trace of Sir2's *whole* dot rather than of its general
  half. Both restorations depend on `M` only through `M + M^T`, which is why
  an antisymmetric kappa may drop them and why §5.1's traced-pair sum is
  zero for two active–active directions.
- **K2 with one direction `o` active–active** (`composedCross`) needs no
  two-general-index array. Applying `o`'s first-order rule to the other
  direction `i`'s first-order inputs gives every pair of slots once, the
  same slot twice included, which is the first-order rule at the product
  `i o`: `A_o A_i z = z^x[o, i] + z'[i o]`. An active–active `o` reads `i`'s
  dotted inputs only at active values of active slots, which they carry; on
  hidden slots it leaves a traced active pair invariant (the projector
  commutes with `o`) and the core and virtual labels untouched, and turns the
  orbital energies' density labels as `[o, gamma]`. So
  `z^x = A_o^vis(z'[i]) - z'[i o]` and
  `eps^x = <[o, gamma], W'[i]> - eps'[i o]`, `W'` being §5.5's C5 generator.
- **K2 on a density-fitted case** (`dfDottedCross`), for any pair: every input
  is a sum over `L` of products of two `B(U)_L = U^T B_L U` factors (the
  traced and tied pairs among them) plus `U^T h U`, so with
  `U = 1 + s u + t v` the cross-slot term is each product's `st` coefficient,
  `F_st G + F G_st + F_s G_t + F_t G_s`, from `B_s = u^T B + B u` and
  `B_st = u^T B v + v^T B u` — three-index throughout. The same evaluator at
  first order (`dfDottedFirst`) is an independent K1 on that case.
- A conventional pair with neither member active–active is refused as
  Unsupported before any device work; lifting that is #180.

The gate is `nevpt2_sc_pseudodensity --sc-hessian-kk-int`
([`testing.md`](testing.md), "The orbital block's integral curvature"): each
term's duality, the traced pairs' and the intra-space zeros, and the ladder
this section asks for — of the pairing itself, the pseudodensities frozen, so
that it sees I'' and nothing of K3–K5: `L(U) = <Ehat'_0, I(U)>` has exactly
this term as its mixed second derivative along `exp(s u + t v)`, displaced
from stored blocks along every block-diagonal pair.

### 5.5 The mixed block `H_kc`: three orientations, two of them built

Write `w` for an orbital direction — an ordinary antisymmetric
`(n_mo, n_mo)` `kappa`, no relation to §5.1's symmetric `W`, which does not
arise here — and `u` for a CI direction. The mixed block is the bilinear form

```
  Eddot_kc(w, u) = d^2/ds dt  E( exp(s w), c + t u )  at s = t = 0
                 = <w, H_kc u>  =  <H_kc^T w, u>
```

**There is no parameterisation term, and so none of K1's hazards.** `c` enters
linearly and `kappa` only once, so `exp`'s second-order term `W` never appears.
`w` is an antisymmetric `kappa` like every first-order direction:
`makeRotation` / `makeGeneralRotation` accept it, the three traced-pair
promotions that cancel against an antisymmetric `kappa` cancel here too (they
are integral-side and do not depend on `c`), and the antisymmetric part of
`Y` — the only part the stored matrix holds — is the only part an
antisymmetric `w` pairs with. Everything in §5.1's list of K1's hazards is
specific to `H_kk`.

**What depends on what.** The external blocks `x_t`, the active `h1e` / `h2e`
and the `h^eff` slices depend on `kappa` alone. The RDMs `dm1`–`dm3` depend on
`c` alone. Two inputs depend on both: the digests `f3`, quadratic in `c` and
linear in `h2e` (§5.2), and the orbital energies, affine in the active density
`gamma` and moving with `kappa` through the integrals and the generalized Fock
([`gradients.md`](gradients.md), §5.5; `src/gradient/gradient_general.cppm`'s
header writes `deps`). The chain rule through `E = sum_t f(N_t, H_t, Delta_t)`
then has exactly five mixed terms. There is no sixth: `x` does not move with
`c`; `N_t`'s metric reads no integral; the Dyall block is bilinear in RDMs and
integrals, up to RDM-free terms with no `c`-derivative; and `Delta_t` is linear
in the orbital energies. Unlike `H_kk`'s, this "no sixth" also holds on the
guarded tuples of every golden §5.6's census covers. Each is either empty by
its integrals or a real perturber below the guard. On an empty one the
perturber is `s psi_t[w](c + t u) + O(s^2)`, with `psi_t[w]` linear in the CI
vector, so `E_t = s^2 F_t(psi_t[w](c + t u)) + O(s^3)` (§5.6's `F_t`, degree
2) and its mixed derivative at `s = t = 0` vanishes identically. A tuple empty because of the state would
break this, as it would §4.1's.

| # | term | what it is | existing contraction it reuses | arrays it reads |
|---|---|---|---|---|
| C1 | **the finish's curvature** | `sum_t (Ndot, Hdot, Ddot)_t[w] . Hess f_t . (Ndot, Hdot, Ddot)_t[u]` | the orbital triple is `scFullRotationDevice`'s tangent mode per tuple (`assembleSlab`'s `normDot` / `hDot`, `deltaOf` at `deps`), **with #88's digest share merged in per tuple** — K3's requirement, §5.1; the CI triple is `energiesTangentDevice`'s per tuple, with `epsDot`; §2's 3×3 between them, as `normToEnergyHessianSandwich` already does for two CI triples | first-order dotted inputs only |
| C2 | **one external slot along `w`, the block along `u`** | `sum_t [ P_t N_t + Q_t H_t ]` with one `x` slot at `x'[w]` and the metric or Dyall block at its CI tangent, symmetrised over the two slots | `evalTermsTangent` over the class term lists, with the block operand the CI tangent block (`make_*` at the tangent RDMs, as the energy's tangent builds it) instead of the plain one | the sidecar's slab tangents |
| C3 | **the Dyall block, both legs** | `sum_t Q_t x^T K^{wu} x`, `K^{wu}` every `make_*` at the tangent RDMs **and** the dotted `h1e'[w]`, `h2e'[w]`, digests zeroed | `rotationTangentBlocks`' builders, handed the tangent RDMs in place of the plain ones — every `make_*` is bilinear in the two, and the RDM-free terms drop exactly as in the CI tangent | `h1e'` reads `heff`, `h2e'` reads `g_xa_aa` |
| C4 | **the digests, both legs** | `sum_t Q_t x^T K(f3^{wu}) x`, `f3^{wu} = d^2 f3 / dc dh2e [u, h2e'[w]]` | §5.2: `buildRdmTangentsDevice` at `(c, u)` with `h2e'[w]` in place of `h2e` (its `dm*` outputs read no integral and are the CI tangent's again), fed to `energiesTangentDevice` with every other tangent zero — #88's construction one order up | `g_xa_aa` |
| C5 | **the orbital energies, both legs** | `sum_t R_t Delta(eps^{wu})_t` with `eps^{wu}_p = 2 sum_x w_xp f^gamma_xp[gamma'] + 2 sum_{x, a, b in act} w_xa gamma'_ab [(pp\|xb) - (1/2)(px\|bp)]`, `gamma' = dm1'[u]`, `f^gamma[g]_xp = sum_ab g_ab [(xp\|ab) - (1/2)(xa\|bp)]` (`p` core or virtual) | the second line is `generalEpsResponse`'s `gamma` line at `dm1'[u]`. **The first is new**: the generalized Fock's *off-diagonal* response to `gamma`, which the first-order path never forms because it reads the stored `f`. The `h^eff` line and the core-density line have no `c`-derivative | `g_xa_cc`, `g_xc_ca`, `g_xa_vv`, `g_xv_va` (the `gamma` line); `g_xc_aa`, `g_xa_ca`, `g_xv_aa`, `g_xa_va` (the Fock line) |

C1 is K3 and C2 is K5 with one direction moved from the integrals to the
RDMs — C1 needs the same per-tuple merge of the digests' orbital share that K3
does — and C3–C4 sit where `H_kk` has K1–K2, a block at a second-order input,
except that here the input is mixed and reads first-order data only. C5 is the one input generator `H_kc` adds; it reads
only sidecar arrays, and it carries its own exact identity —
`h^eff + f^gamma[gamma]` at the state's `gamma` is the sidecar's `f` over the
**whole** MO range, the off-diagonal extension of `scGeneralIntegralCheck`'s
`fock`, which rebuilds the diagonal only.

**Three orientations.** The same bilinear form can leave the computation in
three shapes, and §10's question was which to build:

| | orientation | how | the digests' share | RDM build | what is new |
|---|---|---|---|---|---|
| **F** | the scalar `Eddot_kc(w, u)` | forward over forward: one slab walk carrying the plain blocks, the CI tangent set at `u` and the orbital dotted inputs at `w`, finished by C1's sandwich | C4: the **tangent build at dotted integrals** | the tangent build at `(c, u)`, at `h2e` and at `h2e'[w]` (or one sweep carrying both integral tensors); plus #88's plain build at `h2e'[w]` for C1's orbital triple | the walk holding a CI tangent set beside an orbital one — four block sets per class, the count `energiesHessianDevice` already holds; C5's Fock line |
| **R1** | `H_kc^T w`, a determinant vector | reverse in `c`, forward in `kappa`: the derivative along `w` of `y = ciSourceVector(g; h2e)` | `ciSourceVector(0, 0, 0, gac, gca; h2e'[w])` — the 4-body sigmas at dotted `h2e`, **no transpose** | none beyond the plain build's: #88's plain build at `h2e'[w]` for the multiplier tangents | the pseudodensity walk's tangent along an *orbital* direction — §4.2's `gdot` with the operand tangents `scFullRotationDevice` forms instead of RDM tangents, the multiplier tangents §2.4 fed the orbital triple, and the RDM sweep at dotted integrals (it is linear in them); C5's `W'[w]` in the `eps` fold |
| **R2** | `H_kc u`, an orbital vector | forward in `c`, reverse in `kappa`: the derivative along `u` of `Y`, **probed** basis direction by basis direction as `Y` is (the gauge argument of [`gradients.md`](gradients.md), §6 holds at every `c`, so it holds for `Ydot` too) | `<hbardot[u], h2e'[w]>`, where `hbardot[u]` needs **#89's transpose applied to tangent digests** — the transpose at the pair `(c, u)` | the transpose's tangent: no new kernel (the four `f3_transpose.cu` kernels are linear in each operand), but a new driver at the tangent build's working set, `produce` at `c` and at `u` per tile | that driver, plus §4.2's `gdot[u]` walk paired against `z'[w]` in chunks |

R1 is the cheap one, and the reason is §5.2's linearity seen from the other
side. `y` reads `h2e` in exactly two places — the 4-body sigmas' operand and
the RDM sweep's block chains — and is linear in it in both, so differentiating
`y` along `w` is two `ciSourceVector` calls (one at `gdot[w]`, one with only
`gac`, `gca` at `h2e'[w]`) and never asks for a transpose. Nor does it run a
tangent RDM build: the CI vector is fixed along `w`, so R1 runs at the
**plain** build's tile floors ([`performance.md`](performance.md), "Tile
floors"). F needs the tangent build, at the tangent build's floors. Neither
needs §4.4's three-vector working set: `H_kc` involves two CI vectors, `c` and
`u`, not three, so the unmeasured floors of §10's first bullet do not gate it.

**The verdict: F and R1, with the duality between them as the gate.**

```
  <H_kc^T w, u>  ==  Eddot_kc(w, u)        to round-off, no step size
```

the mixed block's analogue of `<u, y> == Edot(u)` and §4.2's
`<v, H_cc u> == Eddot(u, v)`, and column by column where `ndet` allows (the
`--gradient-source` bound, §4.4). **R2 is not built.** Every consumer this
document describes contracts `H_kc` between an orbital vector and a CI vector
— level 3's `3N` responses (§8) consume `Eddot` on pairs of response vectors,
so `H_ck` applied to each response's orbital half is R1 — and for any such
contraction `<w, H_kc u> = <H_kc^T w, u>` makes R1 sufficient. Only a consumer
that wants the orbital half of the action as a vector in its own right — a
second-order orbital optimiser's residual, which nothing in the tree is —
needs R2, and if one arrives the gate is the same duality, against R1, with
the transpose's tangent as the new piece.

**What the duality sees, and what it cannot.** F and R1 share the operand
tangents `z'[w]` (the promotion rule, `slabDots`, `rotationTangentBlocks`,
`generalEpsResponse`), §2's 3×3 and the multiplier tangents, `h2e'[w]` and C5's
Fock-line generator. They share no code past those: the tangent RDM build
against the sigma vectors, forward-over-forward against the differentiated
walk, and — for the digests — the tangent build at `h2e'[w]` against the
4-body sigma at `h2e'[w]`. So the duality pins the assembly and nothing it is
fed. What it is fed is already pinned: `z'[w]` by every first-order check `Y`
passes, PySCF's exact `A` among them; the 3×3 card-free (§2.3); C5's generator
by the full-range `f` identity above. And **a term both routes omit is
invisible to it** — C4 or C5 left out of both cancels exactly, the way the §6
digest gap survived until #89. Three things outside the pair see an omission:

- **Euler in `c`.** For a class homogeneous of degree 2 in `c` (§4.3's three
  SC classes; PC's five), `E(kappa, s c) = s^2 E(kappa, c)` at orbital
  energies held fixed *in `c`*, so its `kappa`-derivative is homogeneous too
  and, per class,

  ```
    Eddot_kc(w, c)  ==  2 <w, Y>            any w, mixed rotations included
  ```

  against the **already-gated** first-order `Y` (`scFullRotationDevice`'s
  pairing with the digests' share, or the forward tangent plus #88's term,
  per class). It reaches C1–C4 — C4's `u = c` case is
  `buildRdmTangentsDevice(c, c) == 2 buildRdmsDevice(c)` at `h2e'[w]`, the
  `--rdm-tangent` identity at dotted integrals — and holds for F and R1 alike
  (`<H_kc^T w, c>`). Its mode is §4.3's: the orbital energies may respond to
  `kappa` (`deps` at the state's `gamma`) but not to `c`, so C5 and the CI
  triple's `Ddot` are off, and the PASS line says so.
- **The intra-space exact zeros.** Along a core–core or virtual–virtual `w`
  no active integral moves, so C3, C4 and the digests' share of C1 vanish
  **identically** — the negative control `--kappa-full` already runs for the
  first-order digest term. C1's integral part, C2 and C5 do *not* vanish
  there: the external blocks move, and so does the Fock line, since `f`'s
  core and virtual blocks are diagonal only at the state, not along `u`.
- **One finite difference: of the gated `y`, along `kappa`.** A central
  difference of `scCiSourceDevice` (orbital-energy response on, digests
  rebuilt at the displaced `h2e`) along `exp(s w)`, dotted with `u`, against
  `<H_kc^T w, u>` — along the directions `--kappa-full` already displaces: a
  core–core and a virtual–virtual rotation, whose displaced blocks, orbital
  energies and `W` are host transforms of stored arrays, and an
  active–active one through the device `kappa` action (as `--kappa-act`
  displaces the energy), with the orbital energies and `W` rotated on their
  active indices. It is the only check that sees C5 (Euler
  runs with it off) and the only one that sees the digests' mixed term on the
  affine classes. A mixed `w` cannot be displaced and need not be, for
  `--kappa-full`'s reason: each promotion is one contraction over the whole
  general index. A difference of `Y` along `c` is the weaker choice: in the
  complete functional it needs `f` at the displaced `gamma`, which is C5's own
  generator — the difference would share the term it is meant to check.

Like every second derivative here, both legs must run in the same
orbital-energy mode, and the mode is part of the check: Euler with the
response in `c` off, the difference and the duality in either, named.

**PC.** The same five terms and the same three orientations, through §6.1's
`-nu b^T A^+ b`: along `w`, `S` and `S_1` do not move (they are RDM-only;
[`gradients.md`](gradients.md), §6.1), so `b'_w = S x'_w + S_1 h1'_w` and
`A'_w = K'_w + Delta'_w S`; along `u`, `S`, `S_1`, `K` and `Delta` move; the
mixed `b^{wu} = S'_u x'_w + S_1'_u h1'_w` and
`A^{wu} = K^{wu} + Delta^{wu} S + Delta'_w S'_u`. The orbital leg adds no
null-space question of its own: the cut is on `S`'s eigenvalues
([`pc-nevpt2.md`](pc-nevpt2.md), "The singular-metric convention") and `S`
does not move with `kappa`, so the kept span is fixed along `w` and
`G'_w = -G A'_w G` exactly. §6.2's derivative along the CI direction is
then used once per `(w, u)`, and it leaves **one** rotation term,
`-2 rho^T S_u P A_w x`. That term is not zero, because a Dyall block at
dotted integrals couples `S`'s dropped space from the ket side (§6.5). F is
`pcTangentsDevice`'s walk beside `pcFullRotationDevice`'s orbital tangents,
plus that term. R1 is the derivative along `w` of PC's **three-term** `y`,
the walk `pcCiHessianVectorDevice` builds fed orbital operands. It is not
the derivative of `pcCiSourceDevice`'s two-term `y`, which lacks the term.
Euler covers PC's five homogeneous classes. §6.4–§6.8 give the whole
derivation and its checks.

**Data.** §5.4's row stands: nothing the sidecar lacks, the Fock line's four
rows included. A case with no sidecar can form `H_kc` only along active–active
`w` with the orbital energies held fixed in both legs — the functional
`--kappa-act` differentiates at first order — and must refuse any other `w`
rather than return the integral half alone (§9.1's discipline). The
density-fitted case needs no new data either (first-order `Bdot` per
direction), but `scFullRotationDevice` is integral-direct, and so is this.

In the tree (#184) F is `nevpt2.gradient`'s `scMixedHessianDevice` and R1
its `scMixedHessianVectorDevice`, both over the same slab walk with a
**mixed direction** (`:rdm_adjoint`'s `ScMixedDirection`, `:terms`'
`ScMixedWalk`): `w` is the walk's one rotation direction, and each class
builds `u`'s tangent blocks, C3's (its integral-carrying `make_*` at `w`'s
dotted integrals and `u`'s tangent RDMs, tangent-only builders where a block
has an RDM-free term) and, for `a16` / `a22`, the digest-only builds C1's
digest share and C4 read. F finishes each slab's eleven tables per perturber
(`nevpt2.gradient_plan`'s `mixedTermsPacked`, through each symmetrized
class's own packing) into the five terms, reported apart. R1 accumulates
`Mdot_G` along `w` — the multipliers' tangents against the plain
derivative, and the multipliers against the external slots' tangents — and
runs each class's own sweep on it; the sweep is affine in the integrals it is
handed, so its tangent along `w` is the sweep at `h'[w]` less the sweep at
zero integrals, and no adjoint function is touched. **The per-tuple merge of
the digests' share is `ScRotationDir::f3acDot` / `f3caDot`**
(`attachDigestTangents`): a tangent direction handed the RDM build at its
own `h2eDot` builds its `a16` / `a22` tangents with them in place of the
zeros, so every tuple's `Hdot_t` carries the share before anything nonlinear
meets it — C1's requirement here and K3's in §5.1. C5's generator is
`:general`'s `generalEpsResponseTangent`, the tangent `W'[w]` of the
orbital energies' response `W`, symmetrized over its active pair so one
tensor serves F (`orbitalEnergyDot(W', gamma'[u])`) and R1
(`orbitalEnergyDm1Coefficient(W', dE/de)`), out of `fockGammaColumns`,
whose full-range identity is `scGeneralIntegralCheck`'s `fockFull`. With no
sidecar both entry points form an active–active `w` with the orbital
energies frozen and refuse every other `w`, and a responding mode, as
Unsupported; the digests still move there, since `h2e'[w]` along an
active–active `w` needs nothing the golden lacks. The gate is
`nevpt2_sc_pseudodensity --sc-hessian-kc`, the binary that owns the orbital
directions and the sidecar ([`testing.md`](testing.md), "The SC mixed
Hessian block's checks").

### 5.6 The empty perturbers: a sixth term, and pairs with no Hessian

The energy stage skips every tuple with `|N_t| <= NUMERICAL_ZERO`
(`normToEnergy`'s guard), and every derivative in the tree skips it too. At
first order that is exact. At second order it is not, and it is where the
orbital block's chain rule (§5.1) stops being complete.

**Three kinds of guarded tuple.** They behave differently, so the derivation
names them:

| kind | what it is | where it comes from |
|---|---|---|
| **empty by the integrals** | `x_t = 0`: every external integral the tuple reads vanishes, so its perturber `psi_t = sum_mu x_t[mu] O_mu \|c>` is zero at **every** CI vector | point-group symmetry: no active label `mu` has the irrep the tuple's external orbitals need |
| **empty by the state** | `x_t != 0` but `psi_t = 0` exactly (`S x_t = 0`) | the state's occupation, a rank bound, or a symmetry higher than the abelian group the CI vector is masked by (§6.2's three kinds of dropped mode, seen from SC) |
| **small** | a real perturber with `0 < N_t <= NUMERICAL_ZERO` | tiny integrals: acrolein's deep cores |

Only the first two are empty. A small tuple's energy is the guard's zero on a
neighbourhood of the state, so every derivative of the energy the code
computes is zero there. A ladder whose step leaves that neighbourhood sees
the tuple switch on: a jump of order `NUMERICAL_ZERO`, and the tuple's own
curvature, which is as small as its integrals. A census of six committed
goldens ([`testing.md`](testing.md), "The empty-perturber term, offline")
finds the first kind and the third, and none of the second.

**When an empty tuple is populated at first order.** Along `kappa = s u + t v`
the integrals move as `x_t = s x'_t[u] + t x'_t[v] + O(2)`, so the perturber
is `s psi_t[u] + t psi_t[v] + O(2)`, with `psi_t[w] = sum_mu x'_t[w][mu]
O_mu |c>` (the active operators move too along an active rotation, but they
multiply `x_t = 0`). The tuple is populated along `w` exactly when
`N_t^{ww} = 2 <psi_t[w]|psi_t[w]>` is non-zero; `N_t^{uv}` is K4's cross
table (§5.1), both slots filled, one by each direction, and symmetrised.
For a tuple empty by symmetry, a direction's totally symmetric component
maps every tuple's symmetry sector to itself and cannot populate it. Only a
non-totally-symmetric component can, and only one that carries a populated
tuple's integrals into the empty tuple's sector. So which blocks populate
which classes is a property of the orbitals' irreps. On the committed N2
cases (measured, [`testing.md`](testing.md)):

- **core × core** populates Sijrs alone. This is consistent with N2's core
  orbitals all being σ: mixing them breaks inversion, which some of Sijrs's
  empty tuples rest on. It does not break the parity in π orbitals that the
  H classes' empty tuples rest on.
- **virtual × virtual** populates every class whose empty tuples carry a
  virtual index.
- **active × active** at CAS(4,4) (four π actives) populates none.
- a fully mixed direction populates every class with empty tuples (#158's
  device run).
- **a CI direction** populates none: `x_t = 0` does not depend on `c`.

A tuple empty by the state is different. Any direction that moves `x_t` off
`S`'s null space populates it, a symmetric orbital direction included, and
so does a CI direction that breaks the state's symmetry. That is the one way
`H_cc` and `H_kc` would acquire this term (§4.1, §5.5).

**The tuple's energy along the plane.** `N_t` and `H_t` are quadratic forms
in `x_t` at fixed RDMs, so on an empty tuple

```
  N_t(s, t) = Q_N(s, t) + O(3),   Q_N = (s^2 N^{uu} + 2 s t N^{uv} + t^2 N^{vv}) / 2
  H_t(s, t) = Q_H(s, t) + O(3),   likewise from H^{uu}, H^{uv}, H^{vv}
  E_t(s, t) = F_t(s, t) + O(3),   F_t = -Q_N^2 / (Q_N Delta_t + Q_H)
```

with `Delta_t` at the state (its change and `x''` enter at third order).
`F_t` is homogeneous of degree 2 in `(s, t)`. Everything turns on whether it
is a quadratic form.

**When it is: the sixth term.** If the H tables are proportional to the N
tables on the tuple, `(H^{uu}, H^{uv}, H^{vv}) = eta_t (N^{uu}, N^{uv},
N^{vv})`, then `F_t = -Q_N / (Delta_t + eta_t)` and

```
  K6(u, v) = sum_{t guarded, empty} -N_t^{uv} / (Delta_t + eta_t)
           = sum_{t guarded, empty} [ Pbar_t N_t^{uv} + Qbar_t H_t^{uv} ]
  Pbar = 2 Tbar + Delta Tbar^2,   Qbar = Tbar^2,   Tbar = -1 / (Delta + eta)
```

which is **K4 on the guarded tuples, at Park's multipliers' limit along the
ray** instead of the guard's zeros. The second form is the one to compute.
`-(N^{uv})^2 / (N^{uv} Delta + H^{uv})` is the same number where it is
defined, but it is `0/0` when the two perturbers are orthogonal
(`N^{uv} = 0`).

- **Sijrs has `H = 0`**, so `eta = 0` and `Pbar = -1/Delta` on every pair:
  the class's own multiplier. Sijrs's energy is linear in `N_t`
  (`energy_sijrs` applies no guard at all), so the only error is that its
  derivative path applies the guard (`scMultipliers` zeroes `P` at
  `|N_t| <= NUMERICAL_ZERO`). Sijrs should take `P = -1/Delta` on every
  tuple at every order. #158's run shows this is the whole of the core ×
  core gap ([`testing.md`](testing.md)).
- **One direction alone** is always a quadratic form along its line:
  `Eddot(u, u)` gains `-N^{uu} / (Delta + H^{uu}/N^{uu})` on every tuple `u`
  populates.
- **A tuple only one direction populates** contributes nothing to the mixed
  term (`N^{uv} = H^{uv} = 0`, `F_t` depends on one variable).
- **Parallel perturbers**, `psi_t[v] = alpha psi_t[u]`, make the tables rank
  one and proportional automatically. A pair `v = alpha u + w` with `w`
  totally symmetric is the non-trivial example.

**Is proportionality automatic, SC having one function per tuple?** No. An
empty tuple has no function at the state: the displacement defines it. Its
perturbers along `u` and along `v` are two different functions in the
tuple's uncontracted space `span{O_mu |c>}`. The condition is that the Dyall
operator's compression onto their span be `eta` times the metric's. That is
automatic only when the span is one-dimensional, so the two perturbers are
parallel, or when `K` vanishes on it (Sijrs). In general it fails. Every
H class's empty tuples that two random virtual directions both populate fail
it on the N2 and LiF cases, by defects many decades above the round-off
the proportional controls sit at ([`testing.md`](testing.md)).

**When it is not: no Hessian along the pair.** A degree-2 homogeneous `F` is
twice differentiable at the origin only if it is a quadratic form. If
`-Q_N^2 / (Q_N Delta + Q_H) = q` for a quadratic `q`, then
`Q_N^2 = q (Q_N Delta + Q_H)`. Where both directions populate the tuple and
their perturbers are not parallel, `Q_N` is positive definite in two
variables and so irreducible over the reals. It therefore divides `q` or
`Q_N Delta + Q_H`, and either way `Q_H` is a multiple of `Q_N`. So outside
the proportional case:

- **the directional second derivative exists along every line**,
  `D2(w) = 2 F_t(w)` plus the other five terms. The energy is `C^1` and its
  gradient exact; the gradient is homogeneous of degree 1 and not
  differentiable at the state, so a difference of the gradient's `A`
  component along `B` returns `dF/dA` at the unit vector `e_B`, which is not
  symmetric in `(A, B)`;
- **the four-point ladder along `exp(s u + t v)` converges, to the wrong
  thing.** Its leading term is independent of the step,
  `(F(u + v) - F(u - v)) / 2 = (D2(u + v) - D2(u - v)) / 4`. That is the
  polarisation of the directional second derivatives, and it is not
  bilinear: the parallelogram defect
  `D2(u + v) + D2(u - v) - 2 D2(u) - 2 D2(v)` is not zero. This is why
  #158's ladders sat flat from one halving to the next, a fixed distance
  from K1–K5, instead of failing to converge;
- **rescaling one direction cannot expose it.** `F` is homogeneous, so
  `v -> lambda v` scales the stencil by exactly `lambda` whether or not `F` is
  a quadratic form. Only a sum of directions can.

So along such a pair there is no `H_kk` operator to approximate. Any number
returned would be the polarisation of a function that is not quadratic, and
`H_kk u` as a vector, which needs linearity in `v`, would not exist. **The
(class, pair) is refused.** The vector `H_kk u` exists only if every plane
through `u` is proportional, which on these cases means `u` populates no
H class's empty tuple: a totally symmetric `u`, or one breaking only
symmetries the H classes' empty tuples do not rest on.

**The refusal test.** It mirrors PC's `L_uv` (§6.2). It is computed per
(class, pair) on the empty guarded tuples that both directions populate
(`N^{uu}`, `N^{vv}` above a population floor), from K4's cross tables at
`(u, u)`, `(v, v)` and `(u, v)`, which the walk already forms. Normalise to
unit populations,

```
  c = N^{uv} / sqrt(N^{uu} N^{vv}),   k = H^{uv} / sqrt(N^{uu} N^{vv}),
  eta_u = H^{uu} / N^{uu},   eta_v = H^{vv} / N^{vv},   eta = (eta_u + eta_v) / 2,

  rho_t = max(|eta_u - eta_v|, 2 |k - c eta|) / |Delta_t + eta|
```

`rho_t` is zero exactly when the tables are proportional, and it is
invariant under rescaling either direction. It measures the relative size of
the tuple's departure from a quadratic form, since
`F_t - (-Q_N/(Delta + eta)) = Q_N (R - eta) / ((Delta + R)(Delta + eta))`
with `R = Q_H/Q_N` confined to the pencil's range. Refuse the (class, pair)
as `err_numerical` when the largest `rho_t` exceeds a tolerance. The message
should name the class, the number of tuples both directions populate and the
worst `rho_t`. Otherwise add K6 at `eta`. Three details:

- **Exempt Sijrs**, whose `H` is identically zero.
- **Score empty tuples only.** A small tuple's cross tables are real, and
  scoring it would refuse a pair over a perturber the guard holds at zero. To
  tell the two kinds apart, read each tuple's largest integral against the
  integrals' own symmetry noise ([`testing.md`](testing.md) has the
  measured gap).
- **Take the thresholds from the measured separation**, as `PC_LIFT_TOL`
  was. The population floor sits between the noise of a symmetric
  direction's tables and the smallest real population; `rho`'s tolerance sits
  between the proportional controls' round-off and the smallest real defect.
  The host's figures are in [`testing.md`](testing.md); the device's
  reassociation must be measured on the card, not assumed. A metric in
  Hartree, such as `|H^{uu}/N^{uu} - H^{uv}/N^{uv}|`, is the wrong test: it
  is undefined when `N^{uv}` vanishes, which is exactly what two directions of
  different irreps produce.

A guarded tuple with `x_t != 0` whose `N_t` is round-off against
`|x_t|^2 |S|` is empty by the state. None of the census's goldens has one.
Until that case is derived and measured, the Hessian entry points should
refuse it rather than score it.

**The nuclear Hessian, and what it means physically.** At a geometry with
point-group symmetry, a totally symmetric mode's response — orbitals, CI
vector and derivative integrals — is totally symmetric and never populates a
symmetry-empty tuple, so that block of the Hessian is unaffected. A
non-totally-symmetric mode populates them:

- its own diagonal second derivative exists, so a mode alone in its irrep
  has a well-defined curvature;
- **two modes of one irrep** generally give a non-proportional pair, so that
  irrep's block of the harmonic analysis is not defined;
- **two modes of different irreps** populate different symmetry sectors of
  each tuple. Their perturbers are orthogonal, so `N^{AB} = H^{AB} = 0` and
  the mixed value is zero either way, which is the block-diagonal Hessian
  symmetry predicts. But `eta_A != eta_B` in general, so
  `D2(A + B) != D2(A) + D2(B)` and the energy surface is not locally
  quadratic there either.

This is a property of SC-NEVPT2's strong contraction at symmetric
geometries, not of this tree. The contracted function's energy `H_t/N_t` of
an empty tuple is set by the direction of the displacement. Off symmetry
nothing is empty by symmetry and the energy is `C^2`, but `F_t`'s Hessian is homogeneous
of degree 0. So the Hessian near a symmetric geometry changes by a finite
amount over a neighbourhood that shrinks onto it. What the literature we
could read says around it is below ("Related work"). Of the committed cases, N2 and LiF are
diatomic: their one mode is totally symmetric and unaffected. Acrolein (Cs)
has non-totally-symmetric modes, but the census finds its empty tuples only
in Sijrs (`H = 0`), so its K6 is exact. **PC is unaffected:** its `E_t` is a
quadratic form in `b_t` at fixed `A_t`, and §6.1's `2 bdot_u^T G bdot_v` is
evaluated on every tuple, so an empty one needs no sixth term and no
refusal.

**First order is unaffected.** An empty tuple's energy is `O(h^2)` along
every direction, so its first derivative is zero, which is what the guard's
zero multipliers give. `Y`, `y` and every `Edot` are exact. A central
difference of the energy is even in the step and unaffected too, and so is
[`gradients.md`](gradients.md) as a whole. The Sijrs guard above is the one
first-order mismatch with the energy. It is zero on empty tuples, whose
`Ndot` vanishes, and measured negligible on small ones
([`testing.md`](testing.md)).

**Related work.** From a literature search on 2026-10-20, which was not
exhaustive. Full entries are in [`references.md`](references.md), "Related
work: the two 'no Hessian here' findings".

- **What is known around it.** SC-NEVPT2 is not invariant to rotations
  within the inactive and virtual spaces (Angeli, Pastore and Cimiraglia
  2007; Guo *et al.* 2016). Park 2019 reports SC gradients failing at
  degenerate inactive orbitals (benzene, porphine) "due to the lack of the
  invariance". That is a first-order effect with a different mechanism: the
  functional moves under a rotation the gradient assumes free. Here the
  gradient is exact, and what fails is the existence of a bilinear second
  derivative. Internally contracted MRCI and ACPF also delete zero-norm
  functions, and CASPT2 programs remove zero-norm components by threshold
  (the OpenMolcas manual's `THRESHOLDS`). In a variational contraction the
  function's coefficient absorbs its scale. Our reading, which no paper we
  found states, is that this pathology belongs to strong contraction: the
  contracted function is `V Psi_0` itself, with no free coefficient, so its
  energy `H_t/N_t` is set by the direction of the displacement.
- **The mathematics is elementary.** A degree-2 homogeneous rational
  function is twice differentiable at the origin only if it is a quadratic
  form. The application to SC's empty classes is what we did not find.
- **What we did not find stated:**
  - that a class empty by symmetry is populated at first order along a
    symmetry-breaking direction;
  - that the SC energy is then not `C^2` at a symmetric geometry along
    non-proportional pairs;
  - the proportionality condition, or K6;
  - that Sijrs (`H = 0`) is always exact.
  We found no analytic SC- or PC-NEVPT2 Hessian at all (§8).
- **An observation about PySCF.** PySCF's `_norm_to_energy`
  (`pyscf/mrpt/nevpt2.py`, the version `uv.lock` pins) skips a tuple only
  when `|norm| <= NUMERICAL_ZERO`, and evaluates `-norm / (diff + h/norm)` on
  every other tuple. Its Sijrs path applies no cutoff. A finite-difference
  step of size `h` populates an empty tuple at order `h^2`, far above that
  cutoff for any practical step. So a finite-difference SC Hessian built
  from PySCF energies at a symmetric
  geometry includes the populated tuple at every displaced point. Along a
  non-proportional pair it returns the step-independent polarisation above,
  with no sign that the pair has no Hessian. This describes the cutoff's
  effect, not a PySCF bug: PySCF's energy and its first derivative are
  right.
- **Status.** Not found in the literature we could read (date 2026-10-20).
  Not read:
  - the full text of the Angeli SC originals (2001, both papers; 2002;
    2007);
  - the cited-by lists of Angeli 2001 and Park 2019;
  - Molpro's, ORCA's and BAGEL's empty-class thresholds;
  - the older contracted perturbation theories that use one function per
    class (Malrieu's; Meyer's CEPA).

## 6. PC at second order: the eigen-derivative still does not appear

### 6.1 Differentiating the solve twice

With §0's renaming, PC solves in the whole span
([`pc-nevpt2.md`](pc-nevpt2.md), "One idea: PC is SC without the final
contraction"):

```
  E_t = -nu b_t^T A_t^+ b_t,      A_t = K + Delta_t S,    b_t = S x_t + S_1 h1_t
```

and [`gradients.md`](gradients.md), §4.3's first derivative, with
`x = A_t^+ b_t` the solve the energy already ran, is
`Edot_t = -nu (2 x^T bdot_t - x^T Adot_t x)` — no eigenvalue or eigenvector
derivative anywhere. Differentiating again, in two directions, and writing
`G = A^+`:

```
  Eddot_t[u,v] = -nu [ 2 b''^T x + 2 bdot_u^T G bdot_v
                       + 2 bdot_u^T G'_v b + 2 bdot_v^T G'_u b
                       + b^T G''_{uv} b ]
```

Every term is assembled from objects that already exist, plus `A^+` applied to
more right-hand sides:

| term | what it needs |
|---|---|
| `b''` | §3's second-order RDMs through the same `S`, `S_1` builders |
| `A''` | likewise, and the orbital-energy second-order term through `Delta''` |
| `bdot_u^T G bdot_v` | **one extra solve per direction**, against the factorisation the energy stage already computed |
| `G'`, `G''` sandwiched on `b` | `x`, `Adot`, `A''` and the extra solves above |

No eigensolve is rerun and no eigenvector derivative appears, at either order.

### 6.2 The null-space terms do not drop — the cut's null space turns with the state

§6.1 differentiates `A^+` as if the energy stage's solve were the
Moore–Penrose inverse of `A` and as if its null space never moved. Neither
holds, and the second derivative needs the terms that correct both. What the
cut drops, mode by mode, and every figure behind this section are measured in
[`pc-nevpt2.md`](pc-nevpt2.md), "What the cut drops, and how it moves".

**What the solve is.** The energy stage cuts `S`, not `A`: with `Q` the kept
eigenvectors of `S` (`s_k > PC_TAU s_max`, a plain threshold —
`pcRankCut` knows nothing about which modes are structural), it computes

```
  M = Q (Q^T A Q)^-1 Q^T,      E_t = -nu b_t^T M b_t
```

which is `A^+` only where `A` annihilates `S`'s dropped space. At an
eigenstate it does — `K`'s columns are `<Phi_mu|[H_D, O_w]|c>`, and
`O_w|c> = 0` with `H_D|c> = E|c>` make them vanish — but at any other CI
vector, including every displaced state a finite difference visits, `K` couples
the two spaces. Golub and Pereyra's formula
([`references.md`](references.md), "Numerical linear algebra") is then the
wrong object: it needs `A`'s own rank locally constant, and `A`'s is not.
`P K P` stays zero (the bra `O_w|c'>` vanishes), but `P K Q` does not, so off
the eigenstate `A` is a saddle-point matrix on `S`'s dropped space and its
eigenvalues there rise as `h^2` — along the very directions where `S` keeps its
rank. (block2's per-tuple `lstsq` on `H_D` is `A^+` with a cut, so its PC
energy off an eigenstate is a different function from the device's.) `M` is
differentiable wherever `S`'s rank is locally constant, and its derivative
carries `S`'s pseudo-inverse, not `A^+ A^+`. With `P` the dropped-space projector and `S^+` on the kept
space,

```
  dM = -M dA M + P dS S^+ M + M S^+ dS P - M S^+ dS P A M - M A P dS S^+ M
```

**First order.** `b_t = S x_t + S_1 h1_t` lies in `S`'s range at every CI
vector (`S_1` is a Gram column of the same functions), so

```
  Edot_t = -nu (2 x^T bdot_t - x^T Adot_t x - 2 rho^T Sdot P A x),     rho = S^+ x
```

The third term vanishes exactly where `P A = 0`, at the eigenstate, which is
why [`gradients.md`](gradients.md), §4.3's two-term tangent is right *there*
and why its first-order check passes. Off the eigenstate it is the derivative
of nothing: a central difference of the device's two-term `Edot` along `u`
converges to the Hessian **plus** the derivative of the missing term,
`2 rho^T S_v P (A_u x - S_u S^+ b)`, not to the Hessian. A finite-difference
gate on the second derivative must difference the three-term `Edot` (or the
energy twice).

**Second order, at the eigenstate.** With `g = bdot - Adot x` and
`sigma = S^+ b`,

```
  F_uv = 2 b_uv^T x - x^T A_uv x + 2 g_u^T M g_v                         (6.1's terms)
       + 2 rho^T S_u P g_v - 2 rho^T S_v P A_u x + 2 rho^T S_v P S_u sigma   (the rotation)
  Eddot_t[u,v] = -nu F_uv
```

symmetric in `(u, v)`. Every rotation term carries the kept–dropped block of
`S`'s tangent, `P S_u` or `P S_v`, so they vanish where the dropped space does
not turn — and it does turn, for rank-bound modes along any singlet direction
and for spatial-symmetry modes along any direction that breaks that symmetry,
`nevpt2_pc_tangent`'s included. They need no eigen-derivative and no new solve —
`S^+ = U U^T` and `P = I - T U^T` in the solve's own `U` and `T = S U`
([`pc-nevpt2.md`](pc-nevpt2.md), "The solve: once per class, not once per
tuple") — so §6.1's conclusion that the eigendecomposition is never
differentiated survives; its "no null projector" does not.

**Why the null space moves.** "Null for every CI vector" is false: on the
whole fixed-`M_s` determinant space no class drops anything. The dropped
modes are exact null modes of *this state's* metric, of three kinds:

- **spin**: one combination per active orbital in Sr and Si, and one in Sir,
  that two orderings of a repeated active index produce identically on an
  `S = 0` state. Null for every singlet; these do not move.
- **a rank bound**: Sr's functions live in the `(N-1)`-electron doublet
  active space and Si's in the `(N+1)`-electron one. When `n^3` exceeds that
  space's dimension, every singlet has the same nullity but *which*
  combinations are null depends on `c`. The rank is constant along any singlet
  direction, to every order, and the null space turns — this is CAS(4,4)'s
  Sr, the case #152's ladder found.
- **the state's own spatial symmetry**: modes null across the states that
  share the Hamiltonian's full point group (for a linear molecule, more than
  the abelian subgroup the CI vector is masked by), but not for a generic
  vector of the abelian irrep.

**Is the rank constant to second order?** For a Gram metric the dropped
eigenvalues never move at first order (`P S_u P = 0` identically). Their
second-order coefficient along `(u, v)` is

```
  L_uv = P (S_uv - S_u S^+ S_v - S_v S^+ S_u) P
```

built from objects the Hessian already holds (§3's second-order metric, the two
tangents, `S^+`). It vanishes — the rank is constant through second order and
the repaired `F_uv` is the second derivative of the energy the code computes —
along every direction that keeps the state's full symmetry and spin: the
spin-free number-operator directions `nevpt2_pc_tangent` displaces along, the
states of the symmetry family, and on a rank-bound class any singlet direction.
It is **non-zero**, for a class whose dropped modes include spatial-symmetry
ones, along a direction that leaves the state's symmetry: a generic vector of
the abelian irrep where the molecule's group is larger, or any vector of
another irrep (the CI response to a non-totally-symmetric displacement).
Spin and rank-bound modes do not lift along any singlet direction. Where `L`
is non-zero the rank changes at the order the Hessian probes: the energy
agrees with its fixed-rank continuation only on a neighbourhood whose size is
set by `PC_TAU` (in exact arithmetic the span gains a function), and its
second derivative is the derivative of the threshold convention:
**refuse** the class for that direction pair, with `L_uv` (and `L_uu`, `L_vv`)
as the test — measured noise and measured lifts are decades apart.

**Sir's `h^4` mode, in these terms.** Along `nevpt2_pc_tangent`'s direction
(number operators of the state's own orbitals), the displaced functions stay in
the old span at first order, so `L_uu = 0` and the mode Sir's spatial-symmetry
null space carries rises only at fourth order. Through second order the rank is
constant and the repaired formula is exact; the mode is the reason the
difference's step has an upper bound, not a reason to refuse. Along a generic
direction of the abelian irrep the same mode rises at `h^2` and is refused by
the rule above.

**What `PC_MIN_GAP` still guarantees, and what it does not.** It still is the
licence for the rank condition — no kept mode can sink under the cut — and a
class the gap or denominator check refuses is refused here too. It does not
bound the second derivative's conditioning: the rotation terms carry `S^+`
once or twice, so the smallest **kept** eigenvalue of `S`, not the gap, sets
how large they can be, and with it how large the Hessian is, how small a step
its own finite difference needs, and how far a CI-convergence-sized error in
`c` moves it. The measured sizes are in
[`pc-nevpt2.md`](pc-nevpt2.md), "What the cut drops, and how it moves"; a
second-derivative gate should print the smallest kept `s` beside the gap.
How sensitive the inversion is to noise in `c` is known in general (see
"Related work" below). That the CI residual limits an analytic second
derivative through it is not something we found stated.

One qualification is unchanged: the conclusions depend on the cut sitting
below the first real mode. A cut raised to drop physics
([`pc-nevpt2.md`](pc-nevpt2.md) measures what that costs the energy) would put
genuinely non-null modes into `P` and make the energy depend on the cut at
first order already.

**The vector, `H_cc u`.** §4.2 carries over unchanged: `H_cc u = sigma(gdot[u],
c) + sigma(g, u)`, with `g` PC's adjoint. `F_uv` above is bilinear in the two
first-order sets `(S_u, K_u, S_1u)` and `(S_v, K_v, S_1v)`, apart from its
linear part `2 b_uv^T x - x^T A_uv x`, which is `g` paired with the
second-order set (the second sigma call). So `gdot[u]` is the gradient of that
bilinear part with respect to the `v` set, at fixed `u` — §6.2's formula read
in reverse in `v`, rotation terms included, through `y_v = U^T b_v`,
`r_v = Vn b_v`, `alpha_v` and the dropped blocks `Vn S_v U`, `Vn K_v U` —
and then the class's own RDM sweep, which is linear in its accumulators. No
eigenvector derivative appears here either. The `g` beside it is the
**three-term** one: its first-order part carries `-2 rho^T S_v P A x`, so
`y = sigma(g, c)` is the energy's gradient at any CI vector. That is the
function a finite difference of `y` at displaced states differences. The two
forms agree at an exact eigenstate.

In the tree (#155) this is `nevpt2.gradient`'s `pcCiHessianVectorDevice`, with
`pcCiSourceThreeTermDevice` for the three-term `y` alone. It reuses
`pcHessianDevice`'s own per-class blocks (`nevpt2.energy`'s
`pcClassDirection`), applies the same refusals (gap, denominator, and the
lift along `(u, u)`), and holds the orbital energies fixed. Its gate is
`nevpt2_pc_tangent --pc-hessian-vector` ([`testing.md`](testing.md), "The PC
Hessian-vector product's checks"). That gate runs the duality against
`pcHessianDevice`, and a per-class difference of the three-term `y`. The
difference is the only check that sees the rotation terms: the duality would
still pass if both sides omitted them.

**Related work.** From the same search as §5.6's (2026-10-20), which was
also not exhaustive. Full entries are in [`references.md`](references.md),
"Related work: the two 'no Hessian here' findings". None of this section's
five claims is stated in our form. Each has known parts around it. Nobody
has needed the claims about lifted modes and the projector terms: the
literature has no analytic CASPT2 or NEVPT2 Hessian, and every second
derivative we found is a finite difference of an analytic gradient (§8).

- **The dropped modes are exactly null (spin, rank bound, point group).**
  - *Known.* Exact linear dependence among contracted functions is
    recognised. The OpenMolcas CASPT2 manual's `THRESHOLDS` keyword has
    separate thresholds "for removal of zero-norm components" and "for
    removal of near linear dependencies". Nishimoto 2023 (§II B 3) calls
    the effect "noticeable even when only exact linear dependency exists".
  - *Not found.* That every mode PC's cut drops is exactly null at the
    reference, or the split into spin, rank-bound and spatial-symmetry
    modes.
- **Symmetry-breaking directions lift null modes at `h^2`, so there is no
  Hessian.**
  - *Known: a change of rank makes the energy discontinuous.*
    - Li, Zhao and Evangelista 2025 find EOM-ic-MRUCCSD curves
      "discontinuous … due to the change in the number of linearly
      independent operators". They expect this to affect any theory that
      applies excitation operators to an internally contracted state.
    - Nishimoto 2023 (after Eq. 63) notes that his treatment "does not solve
      the problem arising when the number of linearly independent vectors
      varies in different geometries".
    - Nishimoto 2019 says the energy's continuity "has not been investigated
      carefully".
    - Park 2021 traces a spurious XMCQDPT2 frequency for acrolein to the
      overlap threshold.
    - Joubert-Doriol 2022 finds the same discontinuity outside electronic
      structure, for moving bases whose dimension changes.
  - *Known: the mathematics is textbook.* Stewart 1969: the pseudo-inverse
    is continuous only at constant rank. Golub–Pereyra 1973: its
    derivative needs locally constant rank. Rellich–Kato perturbation
    theory (Kato): a zero eigenvalue of an analytic positive semidefinite
    family rises at second order.
  - *Not found:*
    - spatial-symmetry-protected null modes that become non-null at a
      distorted geometry, in any multireference method;
    - the `h^2` law in this setting;
    - the energy failing to be `C^2` at the symmetric point under a fixed
      relative cut;
    - the split between directions that keep the modes null and directions
      that lift them.
- **`L_uv` as the test.** It is the textbook second-order degenerate
  (Schur-complement) correction to a zero eigenvalue (Kato). We found no
  use of it, in quantum chemistry, to decide whether a Hessian exists.
- **The projector terms do not drop, and the off-eigenstate third term.**
  - *Known.* Golub–Pereyra give the constant-rank first derivative.
    Nishimoto 2023 (Eqs. 55–63, IPEA-shifted CASPT2) adds a term coupling
    the dependent and independent spaces, with the structure of our
    `-2 rho^T Sdot_v P A x`. There the IPEA shift couples into the
    dependent space; here the coupling comes from being off the eigenstate.
  - *Consistent with it.* Nishimoto 2019 states that a PC-NEVPT2
    gradient's accuracy "is not affected because all postenergy processes
    … use only linearly independent vectors". That holds at an exact
    eigenstate, where the third term vanishes, as above.
  - *Not found.* Any citation of Golub–Pereyra for an internally
    contracted metric in quantum chemistry, or any second-order projector
    term.
- **The smallest kept `s` amplifies the CI residual.**
  - *Known in general.* Safari, Anderson and Li Manni find that noise in
    stochastic RDMs "propagates non-linearly" through the inversion of the
    CASPT2 metric. Approximate (cumulant-truncated) RDMs are known to give
    false intruders.
  - *Not found.* Residual amplification as a limit on an analytic second
    derivative.
- **Status.** Not found in the literature we could read (date 2026-10-20),
  for each "not found" above. Not read:
  - Evangelista–Gauss 2011; Hanauer–Köhn 2011 and 2012;
  - Celani–Werner 2003; Györffy 2013; Werner–Knowles 1988;
  - Nishimoto 2023's supporting Tables S1–S2;
  - the RMS-CASPT2 gradient (Song, Martínez, Shiozaki);
  - any analytic CASPT2 or NEVPT2 Hessian preprint from 2025–26.

### 6.3 Why PC gains relative to SC here

PC's first derivative is the harder of the two — it differentiates a solve and
has no algebraic identity short of differentiating the eigendecomposition,
which is why `nevpt2_pc_tangent` is one of the tree's three unavoidable finite
differences ([`gradients.md`](gradients.md), §2). At second order the gap
narrows: SC's second derivative needs a new scalar Hessian (§2), while PC's
needs only extra applications of a factorisation it already has, because the
`A^+` sandwich is *already* the general expression. PC's Euler identity covers
five classes against SC's three (§4.3), too. The expensive thing on the PC side
is not the algebra but the extra solves, one pair per direction per tuple.

### 6.4 PC's orbital blocks: §6.1's solve with orbital-dotted inputs

SC's orbital blocks are §5.1–§5.6. This section and §6.5–§6.8 are PC's,
written so that the blocks' implementing issues (#200, #201, #202) need not
re-derive anything. The notation is §6.1's with `G = M`, §6.2's S-cut
restricted inverse, together with the solve's eigenbasis
([`pc-nevpt2.md`](pc-nevpt2.md), "The solve: once per class, not once per
tuple"). `U` is `(d, m)` with `U^T S U = I` and `U^T sym(K) U = Lambda`,
`T = S U`, and `t_1 = S_1^T U`. Every tuple's `A_t = K + Delta_t S` is then
diagonal on the kept span, `D_t = Lambda + Delta_t`, with rows
`yhat_t = U^T b_t` and `xhat_t = D_t^-1 yhat_t` (`x = U xhat`). All of this
is `PcClassSolve`'s own data.

**What moves along `kappa`, and what does not.** The tree differentiates
`E(kappa, c)`: the integrals rotated by `C exp(kappa)` with the CI
coefficients held fixed (`kappaActionDevice` and §3.1's promotion rule of
[`gradient-theory.md`](gradient-theory.md)). The RDMs therefore do not move
along any orbital direction, and **neither do `S` and `S_1`**, which are
built from the RDMs alone. That holds for every block, active–active
included. `Q`, `P`, `S^+`, `U`, `T` and `t_1` are fixed along `kappa`. What
moves is

- the external rows `x_t` (the stored blocks) and the one-electron slices
  `h1_t` (`h1e_v_*`): §5.1's `z'[w]`;
- `K`, through its builder's integrals. Every Dyall block is linear in
  `(h1e, h2e, f3ac, f3ca)`, its RDM-free terms included, and `f3` is linear
  in `h2e` (§5.2). So `K'[w]` is the builder at
  `(h1e'[w], h2e'[w], f3(h2e'[w]))`. That is `pcClassMatrices` at the dotted
  integrals, with the digests' tangent merged in (`attachDigestTangents`'
  `f3acDot` / `f3caDot`, Sr's `a16` and Si's `a22` only). The merge must
  happen **before** the similarity transform below, for the same reason K3
  needs it per tuple in §5.1: the solve is nonlinear in `A`;
- `Delta_t`, through the orbital energies (`deps`), and only when they
  respond.

**First order** (`Y_pc`, [`gradients.md`](gradients.md), §6.1, restated in
this notation). Along `w`,

```
  b_w = S x'_w + S_1 h1'_w,     A_w = K'_w + Delta'_w S
  yhat_w = T^T x'_w + t_1^T h1'_w            the DOTTED slab through the PLAIN T
  Ahat_w = U^T sym(K'_w) U + Delta'_w I      U^T S U = I, so Delta' is diagonal
```

and, because `Q` does not move, `M_w = -M A_w M` **exactly, at any CI
vector**. The first derivative is the two-term `-nu (2 x^T b_w - x^T A_w x)`
everywhere, with no third term (contrast §6.2's `-2 rho^T S_v P A x`, which
exists because a CI direction turns `S`).

**Second order along two orbital directions, `H_kk`.** Along
`exp(s w + t v)` every input has §5.1's `z'' = z'[W] + z^x[w, v]`, and `S`,
`S_1` have none. So `b_wv = S x''_wv + S_1 h1''_wv`, `A_wv = K'' +
Delta''_wv S`, with `K''` the builder at `(h1e'', h2e'', f3(h2e''))`, and

```
  M_wv = M A_w M A_v M + M A_v M A_w M - M A_wv M

  Eddot_t[w, v] = -nu [ 2 b_wv^T x - x^T A_wv x  +  2 g_w^T M g_v ],     g = b' - A' x
```

which is §6.1's formula with no projector, rotation or lift term (§6.5 says
why). The two-coupling classes (Sijr, Srsi, `E_t = -nu b1^T M b2`) carry the
bilinear version, `b1_wv^T x2 + x1^T b2_wv - x1^T A_wv x2 + g1_w^T M g2_v +
g1_v^T M g2_w`. `hessianPair`'s polarisation of the single form gives it, as
it does for `H_cc`. The formula splits exactly where SC's does (§5.1):

- **`2 b_wv^T x - x^T A_wv x` is the first-order pairing at second-order
  inputs.** That is `<Ehat'_pc, I''[w, v]> = K1 + K2`, #200's term (§6.7).
- **`2 g_w^T M g_v` is the whole of `I'^T Ehat'' I'`**, #202's sandwich. In
  the eigenbasis it is
  `sum_t -2 nu gammahat_w,t^T D_t^-1 gammahat_v,t`, with
  `gammahat_w,t = yhat_w,t - Ahat_w,t xhat_t`. That is `pcQuadSecond`'s
  `2 cross` term with both legs orbital, the rotation blocks empty. PC has
  no counterpart of SC's K3/K4/K5 split, of §2's 3×3, or of K6. Its finish
  is a quadratic form in `b` against `A^-1`, not a rational function of
  `(N, H, Delta)`, so the per-tuple curvature is one bilinear form in the
  two first-order sets. #158's walk structure carries over: one slab walk
  holding both directions' dotted slabs and `K'` blocks. Its per-tuple
  finish does not. It is the two-leg `cross` above, with no guarded-tuple
  census (§6.6).

**The extra solves.** In the eigenbasis there are none, in the factorisation
sense. §6.1's "one extra solve per direction" becomes, per class and orbital
direction:

- one slab walk through the plain `T` (`yhat_w`, the energy's own GEMM at the
  dotted slab);
- `K'_w`'s build (`pcClassMatrices` at the dotted integrals) and, for Sr and
  Si, the digests' tangent, which is one plain RDM build at `h2e'[w]` (#88);
- one similarity transform `U^T sym(K'_w) U`, the cost of
  `PcClassDirection::mK`;
- for `H_kc` only, the dropped block `H_w = Vn sym(K'_w) U` (§6.5).

Per tuple, the cost is one `m x m` product `Ahat_w xhat_t` and diagonal
divides by `D_t`. Each direction's set is reusable across every pair it
enters, the same economy §5.1 states for SC's digests.

**The orbital energies' two modes.**

- **Frozen:** `Delta' = Delta'' = 0`. `Ahat_w` is `U^T sym(K'_w) U` alone.
- **Responding to `kappa`** (`deps` out of the sidecar, the mode SC's
  `H_kk` always runs in): `Ahat_w` gains `Delta'_w,t I` per tuple, and
  `A_wv` gains `Delta''_wv S`. The latter is part of K1+K2's `eps` term
  (`ScCurvatureTerm::eps`, at `eps'' = eps'[W] + eps^x`), so the sandwich
  sees only the first-order `Delta'`.

`pcQuadSecond` takes one `Delta` per tuple and no `Delta` tangents, because
`pcHessianDevice` is frozen-`eps`. A responding mode needs the per-tuple
`Delta'_t` of each leg added to that leg's `apply` (an identity block,
`nS = I` here, scaled per tuple). That is the one per-tuple change.

**In the tree (#202).** The sandwich is `nevpt2.gradient`'s
`pcOrbitalSandwichDevice`, over one per-class walk (`:pc_classes`'
`pcClassSandwich`) holding both directions' dotted slabs and `K'` blocks,
each `K'` with its digests' tangent merged before the similarity transform.
Forward, each tuple runs `pcQuadSecond` with the two orbital legs
`(yhat_u, mK_u, Delta'_u)` and `(yhat_v, mK_v, Delta'_v)`, `nS = G = r = 0`
and the mixed set zero: its `2 cross` term alone. Both legs' `Delta'` go
through `PcDeltaTangents`, so a frozen mode is `Delta' = 0` on both.
Reverse, `v` is `pcAdjointSecond*`'s direction (`mK_v`, `nS_v = 0`, no
dropped block), and its bars are taken to the class's own objects and
paired with `u`'s first-order inputs: `(dE/dslab)'[v]` with `u`'s dotted
slabs, `(dE/dK)'[v] = U sym(mggDot) U^T` with `u`'s whole `K'[u]`, and
`R'_t[v]` (`pcAdjointSecondRow`'s `rDot`) with `Delta'_t[u]`. Sijrs runs
SC's own `gradientSijrs` on the pair. `pcOrbitalHessianDevice` assembles
`Eddot_kk = <Ehat'_pc, I''> + the sandwich` in one orbital-energy mode,
dropping the curvature's `eps` term in the frozen one, and keeps the
curvature's K2 refusal. The gate is `nevpt2_sc_pseudodensity --pc
--sc-hessian-kk` ([`testing.md`](testing.md), "PC's orbital block's
sandwich").

**The mixed inputs, `H_kc`** (`w` orbital, `u` a CI direction; §6.5 has the
formula):

```
  b_wu = S'_u x'_w + S'_1u h1'_w               yhat_wu = Tdot_u^T x'_w + t1dot_u^T h1'_w
  A_wu = K_wu + Delta_wu S + Delta'_w S'_u     (Delta'_u S'_w = 0: S'_w = 0)
```

`K_wu` is the builder at `u`'s tangent RDMs and `w`'s dotted integrals
(§5.5's C3), with the digests at both legs (C4). Those are
`buildRdmTangentsDevice` at `(c, u)` handed `h2e'[w]`. `Delta_wu` is C5's
`eps^{wu}` (`generalEpsResponseTangent`), reused verbatim because the
orbital energies are the same for SC and PC. `Tdot_u`, `t1dot_u`, `nS_u`,
`mK_u` and `G_u` are `pcClassDirection`'s at the CI leg.

### 6.5 The cut's null space does not move along `kappa`; `H_kc` carries one rotation term

**`S` does not move along any orbital direction, so `H_kk` has no projector,
rotation or lift term.** This is in the functional the tree differentiates
(§6.4). The milestone plan expected an active–active `kappa` to rotate `S`
orthogonally at constant rank, and so to need §6.2's rotation terms. That
is the same fact seen through the active–active gauge
([`gradient-theory.md`](gradient-theory.md), §1.3), and it is not what the
tree computes. With `what` the determinant-space generator of the rotation,
`sum_pq w_pq a+_p a_q` on both spins,

```
  E(exp(s w), c)  =  E(1, exp(s what) c)          w active-active
```

On the right-hand side the RDMs turn, so `S -> R S R^T` with `R` the
class's tensor power of `exp(s w)`, at constant rank. §6.2's rotation terms
are non-zero there, because they belong to the CI direction `what c`. On the
left-hand side, which is what `kappaActionDevice` and the promotion rule
compute, nothing in `S` moves. Both sides are the same function. #200–#202
implement the left. So, for every (class, orbital pair), active–active
included:

- `P S_w = 0`, and `L_ww = L_wv = 0` identically. There is no lift test and
  no lift refusal.
- `M(kappa) = Q (Q^T A(kappa) Q)^-1 Q^T` with `Q` fixed. It is as smooth as
  `A` wherever the denominator check passes, and §6.4's `H_kk` is the
  derivative of the energy the code computes.

**`H_kc`: `S` moves along `u` only, and one rotation term survives.**
Differentiate `M_w = -M A_w M` along `u`, with `M_u` §6.2's `dM`:

```
  Eddot_t[w, u] / (-nu) = 2 b_wu^T x - x^T A_wu x + 2 g_w^T (M b_u + M_u b)
```

This holds exactly, at any CI vector. At the eigenstate (`P A = 0`), and
with `P b = P b_w = 0` (`b` and `b_w` lie in `S`'s range: `S` is fixed
along `w`, and `S_1` is a Gram column), it is

```
  Eddot_t[w, u] = -nu [ 2 b_wu^T x - x^T A_wu x + 2 g_w^T M g_u
                        - 2 rho^T S_u P A_w x ]                       rho = S^+ x
```

This is §6.2's `F_uv` with one leg orbital. Of its three rotation terms, two
carry `P S_w` and vanish. The third, `-2 rho^T S_v P A_u x` with `v` the CI
leg, is the one above.

**Why the surviving term is not zero.** `P A_w x = P K'_w x`, because
`P S = 0`. A Dyall block is `K_{mu nu} = <Phi_mu|[H_D, O_nu]|c>`
([`pc-nevpt2.md`](pc-nevpt2.md), "Symmetrisation"). It annihilates a dropped
mode `n` (`O_n|c> = 0`) from the bra side at any integrals, but from the ket
side only when `H_D|c>` is proportional to `|c>`. At dotted integrals,

```
  (K'_w n)_mu = -<Phi_mu| O_n |r_w>,     r_w = (H_D'[w] - <c|H_D'[w]|c>) |c>
```

`r_w` is the state's residual under the dotted Dyall Hamiltonian, and it is
not zero. So `sym(K'_w)` couples the kept and dropped spaces from one side.
§6.2's three kinds of dropped mode give:

- **spin modes:** zero. `O_n` annihilates every singlet, and `H_D'[w]` is
  spin-free, so `r_w` is a singlet;
- **rank-bound modes** (null for this `c` only): non-zero in general;
- **spatial-symmetry modes:** zero along a `w` that keeps the state's full
  symmetry, because `r_w` shares it. Non-zero in general along a `w` that
  breaks it.

The term also needs `P S_u != 0`, a CI direction that turns the dropped
space (§6.2). It is present along **every** `w` that moves `K`, not along
active–active `w` alone: every block except core–core and virtual–virtual.
Those two move no active integral and leave `h^eff`'s active block
invariant, so `K'_w = 0` (§6.8's intra-space zeros).

**In the solve's own objects.** With `G_u = Vn sym(S_u) U` and
`H_w = Vn sym(K'_w) U` (`PcClassDirection`'s `G` at the CI leg and `H` at
the orbital one), and `C = U^T U` (`PcDroppedBase::C`, so
`S^+ = U U^T`), the term is

```
  -2 rho^T S_u P A_w x  =  -2 xhat^T C G_u^T H_w xhat
```

That is exactly `pcRotationBlocks`' `A1 = C G_u^T H_v` with `v` the orbital
leg. `B1`, `B2`, `A2` and `Ju r_v` all vanish there, because `G_w = 0` and
`r_w = Vn b_w = 0`. So #201's forward scalar can run `pcQuadSecond` with `u`
the CI leg and `v` the orbital leg:

| leg `v` field | value |
|---|---|
| `yv` | `yhat_w` |
| `mKv` | `U^T sym(K'_w) U` |
| `nSv`, `Gv`, `rv` | zero |
| `Hv` | `H_w` |
| `yuv` | `Tdot_u^T x'_w + t1dot_u^T h1'_w` |
| `mKuv` | `U^T sym(K_wu) U` |
| `nSuv` | zero |

A responding mode adds §6.4's per-tuple `Delta` tangents in both legs, and
`Delta_wu I + Delta'_w nS_u` in `mKuv`. For the two-coupling classes the term is
`nu [ rho1^T S_u P A_w x2 + rho2^T S_u P A_w x1 ]`, as `hessianPair`
polarises it.

**The reverse orientation, `H_kc^T w`, differentiates the three-term `y`.**
§5.5's R1 for PC is the derivative along `w` of §6.2's three-term `y`
(`pcCiHessianVectorDevice`'s walk; `pcCiSourceThreeTermDevice`). It is
**not** the derivative of `pcCiSourceDevice`'s two-term one. The rotation
term is exactly the `w`-derivative of the third term,
`-2 rho^T S_v P A x`, at the eigenstate: its `rho_w` and `x_w` factors
multiply `P A = 0`. Its adjoint in `S_v` adds, per tuple,

```
  dE/dS  +=  2 nu sym( rho (P K'_w x)^T )  =  2 nu sym( U C xhat xhat^T H_w^T Vn )
```

to the walk's tangent, built from the walk's own `C`, `xhat` and `Vn` and the
orbital leg's `H_w`. A `y` without the third term, differentiated along `w`,
omits it. So would a duality against a forward scalar that also omits it.
Euler cannot see it either (`u = c` gives `P S_c = 2 P S = 0`, §6.8).

**The lift, along `(w, u)`.** The dropped eigenvalues' second-order
coefficient has `L_ww = L_wu = 0` identically (`S_w = S_wu = 0`), so the only
test is the CI leg's own: first-order `P S_u P` and `L_uu` against
`PC_LIFT_TOL`, exactly `pcCiHessianVectorDevice`'s. `H_kc` inherits the CI
side's lift refusal through its `c` direction, and nothing more. The
determinant vector `H_kc^T w` has no CI direction to test, so the consumer
that pairs it with a `u` applies `u`'s test. That is how
`pcCiHessianVectorDevice` leaves `v`'s to `pcHessianDevice`. #201's forward
scalar tests its `u`.

**Which (class, block) pairs carry which terms.** Which modes each class
drops on which golden is measured in [`pc-nevpt2.md`](pc-nevpt2.md), "What
the cut drops, and how it moves". The terms follow from the kinds:

| class | `H_kk` | `H_kc` |
|---|---|---|
| Sr, Si | §6.4's terms; no rotation, no lift. The digests' share in `K'` | §6.4's terms, plus the rotation term on rank-bound and symmetry modes (spin modes give zero), plus the CI leg's lift test. The digests' share in `K'_w` and `K_wu` |
| Sir | the same, no digests | the same, no digests. Its spatial-symmetry mode lifts along a CI direction that breaks that symmetry, and the (class, `u`) is then refused |
| Sijr, Srsi, Srs, Sij | §6.4's terms | §6.4's terms. The rotation term and the lift test are carried generically and vanish where `nd = 0` |
| Sijrs | SC's Sijrs, unchanged: K1+K2 through its SC pseudodensity (`gradientSijrs`), and §5.6's sandwich with `P = -1/Delta` on every tuple (`sijrsMultipliers`). Exact, never refused | SC's Sijrs, unchanged. It reads no RDM, so in a mode with the orbital energies fixed in `c` its share is zero. In the full mode it is §5.5's C5 through its `Delta` |

In the tree (#201) the forward scalar is `nevpt2.gradient`'s
`pcMixedHessianDevice` and the reverse vector its
`pcMixedHessianVectorDevice`, both over one per-class walk
(`:pc_classes`' `pcClassMixed`). Forward, each class runs
`pcQuadSecond` per tuple exactly as the table above sets its legs: the CI
leg's blocks from `pcClassDirection` at `u`'s tangent RDMs, the orbital
leg's `mK_w` and `H_w` from the class's K builder at `w`'s dotted integrals
with the digests' tangent along `w` merged in (`attachDigestTangents`,
before the similarity transform), and `mK_wu` from the builder at `u`'s
tangent RDMs and `w`'s dotted integrals with `f3^{wu}` in the digest slots,
less the same builder at the plain RDMs and zero digests. A responding
mode passes `Delta'_w` per tuple through `pcQuadSecond`'s
`PcDeltaTangents`. The rotation term is reported apart, as the difference
from the same finish with no dropped space. Reverse, `w` is
`pcAdjointSecond*`'s direction (`mK_w`, `H_w`, `y_w`, `G_w = 0`), so its
bars land on the CI leg's objects: `gbarDot` is the rotation term's
adjoint above. The mixed set's part is the plain `ybar` against the dotted
slabs, onto `T_v`, and the plain `dE/dK` swept at `w`'s dotted integrals
less the same sweep at zero integrals; the 4-body sigmas at `h2e'[w]` take
the plain digest coefficients. Both orientations refuse a responding-in-`c`
mode as `err_unsupported`, and the forward one refuses a `u` that lifts
as `err_numerical`, naming the class. The gate is
`nevpt2_sc_pseudodensity --pc --sc-hessian-kc`
([`testing.md`](testing.md), "PC's mixed Hessian block's checks").

### 6.6 PC has no empty-perturber term and no `rho_t` refusal

**§5.6's pathology does not carry over.** PC's finish evaluates every tuple.
`E_t = -nu sum_k yhat_k^2 / (lambda_k + Delta_t)` divides by no norm, and
nothing in the PC classes applies `NUMERICAL_ZERO`. Take an integral-empty
tuple (`x_t = 0`, `h1_t = 0`, so `b_t = 0` at every CI vector). Along
`kappa = s w + t v`, `S` is fixed and `M` depends smoothly on `(s, t)` through
`K` and `Delta` only, wherever the denominator check passes. So

```
  E_t(s, t) = -nu (s b_w + t b_v)^T M (s b_w + t b_v) + O(3)
```

is a **quadratic form** in `(s, t)`. Its mixed derivative,
`-2 nu b_w^T M b_v`, is §6.4's sandwich at `x = 0` (`g = b'`). There is no
Rayleigh quotient: an empty tuple's "contracted function" is the whole span,
so the direction of the displacement does not fix an energy, as it does in
§5.6.

- **No K6.** The general formula already contains the term, on every tuple.
- **No `rho_t` refusal**, and no population floor to measure.
- **`H_kc` and `H_cc` are untouched by integral-empty tuples.** `x_t = 0`
  for every `c`, so `b_t = s b_w(c + t u)` and `E_t = O(s^2)`: its mixed
  derivative vanishes, as in §5.5.
- **A tuple empty by the state** (`x_t != 0`, `b_t = S x_t = 0`, so `x_t` lies
  in `P`'s range) is covered too, unlike SC's refused row. Along `kappa`,
  `b_w = S x'_w` and the argument above applies. Along a CI `u`,
  `b_u = S_u x_t = S_u P x_t`, which lies in `Q`'s range wherever
  `P S_u P = 0`. Then `E_t = -nu t^2 b_u^T M b_u + O(t^3)`, again a
  quadratic form, and §6.2's formula needs no special case wherever the lift
  test passes.
- **Sijrs is SC's class.** It is exact through `sijrsMultipliers`, and
  `rho_t` exempts it (`H = 0`).

So §9.1's four guarded-tuple rows are SC's alone. PC's seven solved classes
add no row. [`pc-nevpt2.md`](pc-nevpt2.md), "The orbital blocks, offline"
has the numbers: along virtual–virtual pairs that populate empty tuples in
five PC classes, on which SC's `rho_t` is far above its tolerance, the PC
class energies' four-point ladders land on the closed-form bilinear values.

### 6.7 PC's integral adjoint `Ehat'_pc`, and the curvature term

**`Ehat'_pc` already exists.** It is the set #74's reverse mode forms for
`Y_pc` ([`gradients.md`](gradients.md), §6.1), one pairing per line of

```
  <kappa, Y_pc> = sum_slabs <dE/dslab, slabdot> + <dE/dK, Kdot>
                + sum_t R_t Deltadot_t          + <hbar_pc, h2e_dot>
```

Per class, those pairings are:

- `dE/dslab` (`T ybar` per tuple), paired with every stored block's dot;
- `dE/dK = U sym(mgg) U^T`, taken through the class's own Dyall-block
  adjoints onto `(h1e, h2e)` (`rotAct`);
- `R_t = dE/dDelta_t`, scattered onto `dE/de` (`rotEps`);
- `hbar_pc` (`pcDigestAdjointDevice`): Sr's and Si's `dE/d(f3ac)`,
  `dE/d(f3ca)`, transposed onto `h2e`. It is direction-independent and built
  once per state.

Sijrs contributes SC's pseudodensity (`gradientSijrs`). `dE/dS` and
`dE/dS_1` are not part of it: they have no integral dependence.

**K1 and K2 for PC.** `<Ehat'_pc, I''[u, v]> = K1 + K2` is that pairing at
the dotted sets #154's generators produce. The generators are `ScDotted`:
`dottedAtMatrix` for K1, and `composedCross` or `dfDottedCross` for K2. They
produce **inputs**, not pairings, so they serve PC unchanged. What #200 adds
is a PC pairing that accepts a direction carrying a `ScDotted`, with four
pieces:

- slabs: the bypass `:terms`' `slabDots` already has for SC;
- `Kdot`: `pcClassMatrices` at the set's `h1e` / `h2e` fields;
- `Delta`: from its orbital-energy field;
- digests: `hbar_pc` against its `h2e` field.

For the duality, the forward tangent is `pcRotationTangent*` at the same
inputs, plus `pcDigestRotationTangentDevice` at `h2e''`. Two remarks are
specific to PC:

- **PC reads no `v2tr`.** `dottedAtMatrix`'s `v2tr` restoration is SC's
  alone. PC's Sir pairs `dE/dslab` against `h2e_v2_Sir`'s own dotted block,
  which the generator forms in full. The `h1e_v_Sr` restoration applies
  unchanged: PC reads `h1e_v_Sr` as Sr's one-body slab. §5.1's traced-pair
  zero for two active–active directions therefore holds for PC with
  `h1e_v_Sr`'s share alone.
- **`K` at a symmetric `W`.** `pcClassMatrices` is linear in its integrals
  and accepts any dotted pair. `mK = U^T sym(K'') U` as at first order.

**K2 hits the same conventional-case refusal (#180).** PC reads the same
inputs SC does: the twelve stored blocks, `h1e`, `h2e` and the orbital
energies. §5.4's union is over arrays, not classes. So K2's two-general-index
data is the same, and so is the rule:

- **a pair with an active–active member:** `composedCross`;
- **a density-fitted case:** `dfDottedCross`, any pair;
- **any other conventional pair:** `err_unsupported` before any device work.

#180 lifts the refusal for both variants at once.

**One precondition on the density-fitted case.** Before #200, PC's
first-order orbital source term on a fitted golden had no gate in the tree.
`--kappa-full` is SC's, and `--pc --dump-source-terms` is refused there
([`gradients.md`](gradients.md), §6.2). K1 at an antisymmetric matrix is that
first-order pairing. So #200's (and #201's, #202's) density-fitted entry
must first gate PC's first-order dualities on the fitted inventory, the
integral half and the full one on seeded directions, as `--kappa-full` does
for SC. Its K1/K2 on that case mean nothing until then. `--pc
--sc-hessian-kk-int` runs that gate as its first check, on every case.

### 6.8 Identities for PC's orbital blocks

**`H_kc`.**

- **Duality**, `<H_kc^T w, u> == Eddot_kc(w, u)`, round-off exact: the
  three-term walk's tangent against the forward scalar (§6.5).
- **Euler in `c`**, for PC's five homogeneous classes (Sr, Si, Srsi, Srs,
  Sir; [`gradients.md`](gradients.md), §4.3). Their `S`, `K` and `S_1` are
  linear in the RDMs, so `E(kappa, s c) = s^2 E(kappa, c)` with the orbital
  energies fixed in `c` (the cut is relative, hence invariant). Per class,

  ```
    Eddot_kc(w, c)  ==  2 <w, Y_pc>          any w
  ```

  against the already-gated `Y_pc` (`pcFullRotationDevice`'s adjoint with
  `hbar_pc`). It reaches §6.4's mixed terms, the digests' both-legs term
  included, but not the rotation term (`P S_c = 0`) and not C5 (off in
  that mode). Sijr and Sij are affine (`hdm1`, `hdm2`), and Sijrs has
  degree 0.
- **The active-rotation gauge** (`w` active–active, any `u`, no step size).
  From `E(exp(s w), c + t u) = E(1, exp(s what)(c + t u))`,

  ```
    Eddot_kc(w, u)  ==  Eddot_cc(what c, u)  +  <y, what u>
  ```

  with `Eddot_cc` `pcHessianDevice`'s and `y` the gated one. At first order
  it is §1.3's `<w, Y_pc> == <y, what c>`. **It is the only step-free check
  that sees the rotation term.** The left side carries it as §6.5's single
  term. The right side carries it as `Eddot_cc`'s own rotation terms along
  `what c`, which turns `S` orthogonally. `what c` never lifts: `P S_{what c}
  P = 0` and `L = 0` identically for an orthogonal similarity, so only `u`'s
  test applies. Both sides run with the orbital energies frozen, or both
  responding; never mixed. `what` is a non-symmetric one-body operator.
  (PySCF's `direct_spin1.contract_1e` reads only `f`'s lower triangle;
  `direct_nosym`'s does not.)
- **The intra-space exact zeros.** Along a core–core or virtual–virtual `w`,
  `K'_w = 0`. So `H_w`, the rotation term, `K_wu` (C3/C4's analogue) and the
  digests' share vanish identically. `yhat_w`, `Delta'_w` and `b_wu` do not.
- **One finite difference of the gated three-term `y` along `kappa`** (#201's
  acceptance). It sees the rotation term and C5.

**`H_kk`.**

- **K1 at an antisymmetric matrix is the first-order pairing** `<M, Y_pc>`,
  as `scMatrixRotationDevice`'s is SC's.
- **The traced-pair zero** for two active–active directions (§6.7: Sr's
  share alone).
- **Forward against reverse** on each term, and the digests' sandwich share
  against `pcHessianDevice` run at the digests' tangents alone (`dotU =
  f3'[w]`, `dotV = f3'[v]`, every other RDM tangent and the second-order set
  zero). With `S_u = 0` in both legs its rotation terms and lift tests are
  empty. This is #202's "the digests' share against the plain digest path".
- **The gauge tying `H_kk` to `H_kc`** (`w` active–active, any `v`, no step
  size). `C exp(t v) exp(s w)` at CI `c` is `C exp(t v)` at CI
  `exp(s what) c`. With Baker–Campbell–Hausdorff
  (`exp(t v) exp(s w) = exp(t v + s w + (ts/2)[v, w] + ...)`), that gives

  ```
    Eddot_kk(v, w) + (1/2) <[v, w], Y_pc>  ==  Eddot_kc(v, what c)
  ```

  in a matched orbital-energy mode. It holds the assembled `Eddot_kk`
  against `H_kc`, whose rotation term the right-hand side carries and the
  left does not. It is #202's identity against #201 when both have landed.
  It holds for SC as well, with SC's blocks.
- **Ladders** of the total PC energy along `exp(s u + t v)` (#202), with
  §5.6's floor-aware rule unchanged. PC has no non-bilinear pair to refuse
  (§6.6), so every ladder must converge to `Eddot_kk`.

The offline evidence for §6.5 and §6.6 is
`reference_data/pc_orbital_hessian_probe.py`. Its figures are in
[`pc-nevpt2.md`](pc-nevpt2.md), "The orbital blocks, offline". It covers
`H_kk` along active–active pairs against an energy ladder, the mixed block
with and without the rotation term against a difference of the exact
orbital derivative, both gauges, and the empty tuples.

## 7. Level 2: what the Lagrangian's second derivative adds

Level 1 is the Hessian of the NEVPT2 *energy* in the parameters. A nuclear
Hessian also reads the Hessian of the **Lagrangian**, because the constraints
carry geometry dependence of their own. Writing the constraints of
[`gradient-theory.md`](gradient-theory.md), §1 as `g(theta; R) = 0` with
multipliers `Z`, the Lagrangian is `L = E + Z . g` and

```
  d^2 L / dtheta dtheta  =  d^2 E / dtheta dtheta  +  Z . d^2 g / dtheta dtheta
```

The multipliers themselves are **not** differentiated. That is not a
convenience, it is the **2n+2 rule**: in a fully variational Lagrangian the
state and orbital response parameters obey Wigner's 2n+1 rule while the
Lagrange multipliers obey the stronger 2n+2 one
([`references.md`](references.md), "Derivative theory"), so a *second*
derivative needs the parameters to first order and the multipliers only to
**zeroth** — the gradient's own `Z`, used as computed. §8 is where that economy
is cashed in. What does **not** survive is the constraint term:

- **CASCI branch.** Four of
  [`gradient-theory.md`](gradient-theory.md), §1.2's constraints carry
  multipliers, and they do **not** all live in the orbital block. Only (O1)
  does: `F` is the RHF Fock, and §1.2 is explicit that it "does not depend on
  the CI vector", so `Z . d^2 F / dtheta^2` is `kappa`-`kappa` only. Note what
  that object is, though — `dF/dkappa` is the RHF orbital Hessian, i.e. CPHF's
  own operator ([`gradient-theory.md`](gradient-theory.md), §2.4), so
  `d^2 F / dkappa^2` is the **derivative of** that operator: a third derivative
  of the RHF energy, not the operator itself.

  The other three spread across all three blocks, and assuming otherwise is the
  easy mistake here:

  | constraint | multiplier | blocks it contributes to | why |
  |---|---|---|---|
  | (O1) `F` block-diagonal | `Z` | `kappa`-`kappa` | `F` is the RHF Fock: CI-independent |
  | (O2) pseudocanonicality `f_ij`, `f_rs` | `z^c` | **all three** | `f` is the **generalized** Fock, built from `d^(0)` and so linear in the active 1-RDM |
  | (C1) CI eigenvector + norm | `z`, `x` | `kappa`-`c` and `c`-`c` | `H` carries `kappa` through the integrals; `z^dag (H - E_CASCI) c` with `E_CASCI = c^dag H c` is cubic in `c`, and the norm term contributes `-x delta_IJ` |
  | (N1) orthonormality | `X` | `kappa`-`kappa` | the `exp(kappa)` parameterisation's own second order |

  This branch is still the mild one, but for a different reason than "no CI
  coupling": none of these needs a third derivative of a *multireference*
  quantity. (O1)'s is a third derivative of the RHF energy — closed form, AO
  based; (O2)'s and (C1)'s read the generalized Fock and the active-space
  Hamiltonian, both **linear** in the RDMs, so their second derivatives in `c`
  are §3's second-order RDMs again.
- **CASSCF branch.** The constraint is `A - A^dagger = 0`
  ([`gradient-theory.md`](gradient-theory.md), §1.4), and `A` is built from the
  reference's 1- and 2-RDMs. So `d^2 g / dtheta^2` contracted with `Z` is a
  **third** derivative of the CASSCF energy — the derivative of the CP-CASSCF
  operator §2.7 solves, not that operator itself. It couples `(kappa, kappa)`,
  `(kappa, c)` and `(c, c)` alike.

That third-derivative term is the honest reason level 2 is a step up from level
1 and not a rewrite of it, and it is entirely a *reference*-side object: it
involves the CASSCF energy, not the NEVPT2 one, so none of §2–§6's machinery
touches it. On the CASCI branch it is reachable from the same CPHF operator the
offline gradient already builds; on the CASSCF branch there is no PySCF entry
point for it, which is the same gap §8 runs into.

## 8. Level 3: the nuclear Hessian, and why its oracle is the hard part

With `L` stationary in **every** variable at the solution — the multipliers
included, which is what "fully variational" means — write `Theta` for that
whole set. Then the envelope theorem gives the gradient as
`dE/dR_A = L_{R_A}`, and differentiating once more:

```
  d^2 E / dR_A dR_B  =  L_{R_A R_B}  +  sum_Theta L_{R_A Theta} Theta^(1)_B
  where   L_{Theta Theta} Theta^(1)_B  =  - L_{Theta R_B}
```

(symmetric in `A` and `B` despite appearances, since the second term equals
`-Theta^(1)_A . L_{Theta Theta} . Theta^(1)_B`).

**Which operator is `L_{Theta Theta}` — this is the part most easily got
wrong.** It is *not* level 1's Hessian, and it is not level 2's either. With
the multipliers in `Theta` it is a saddle-point (KKT) matrix whose
multiplier–multiplier block is zero. Unpacking it via the 2n+1/2n+2 rules (§7)
gives the form worth implementing:

- the **wavefunction parameters** are needed to first order only, and their
  response solves against the **reference's** electronic Hessian — CPHF on the
  CASCI branch, the coupled CP-CASSCF operator on the CASSCF one
  ([`gradient-theory.md`](gradient-theory.md), §2.4 and §2.7). That operator is
  one PySCF already provides and `reference_data/nevpt2_grad.py` already builds
  from `pyscf.mcscf.newton_casscf.gen_g_hop`;
- the **multipliers** are needed to zeroth order only — the gradient's own `Z`,
  undifferentiated — contracted against the constraints' *second* derivatives,
  which is the §7 term;
- **level 1's operator is consumed, not solved**: it enters as the bilinear
  form `theta^(1)_A . E_{theta theta} . theta^(1)_B`, i.e. as `3N`
  Hessian-vector products against the responses. That is exactly the
  action-on-a-direction deliverable §5.3 argues for, which is the one piece of
  good news in this section.

Three things are needed that the gradient did not need:

1. **Second-derivative integrals** — `d^2 h / dR_A dR_B` and
   `d^2 (pq|rs) / dR_A dR_B`, plus the nuclear-repulsion second derivative.
   Neither is in the tree nor in the reference data
   ([`gradients.md`](gradients.md), §1 lists even the *first* derivatives as
   absent from both).
2. **The constraints' second derivatives**, §7 — including the CASSCF branch's
   third-derivative term, contracted with the gradient's own `Z`.
3. **`3N` response solves** of the reference's Hessian. This is the structural
   difference, and it is worth being blunt about: the Handy–Schaefer
   interchange buys a *single* Z-vector solve for a *first* derivative, and
   buys nothing on the parameter-response side at second order. The 2n+1 rule
   caps the order needed at one, not the *count*: each of the `3N` coordinates
   needs its own `theta^(1)_B`, so [`gradient-theory.md`](gradient-theory.md),
   §2.5/§2.7's single solve becomes `3N` of them. What 2n+2 does buy is that
   no second-order parameter response `theta^(2)_AB` is ever needed — there
   would be `3N(3N+1)/2` of those — because the zeroth-order multiplier absorbs
   them all. That is the same economy the gradient uses, applied one order up,
   and it is the difference between a feasible calculation and an infeasible
   one.

None of that is conceptually hard. The **oracle** is, and that is the argument
for not starting here:

- **This is what the field does, and it is worth knowing before choosing
  otherwise.** No analytic second derivative of any internally contracted
  MRPT2 has been published. Both 2019 NEVPT2 first-derivative papers — Park's,
  which [`gradient-theory.md`](gradient-theory.md) derives from, and
  Nishimoto's independent PC-NEVPT2 one — obtain second derivatives
  *seminumerically*, by a stencil over their own analytic gradient, and
  Nishimoto's frequencies are computed that way
  ([`references.md`](references.md), "Analytic gradients"). A tree that wanted
  NEVPT2 frequencies would be doing the state of the art by differencing the
  gradient it already has offline.
- There is no `pyscf.hessian.lagrange` to subclass the way
  `reference_data/nevpt2_grad.py` subclasses `pyscf.grad.lagrange.Gradients`
  ([`reference-data.md`](reference-data.md), "The analytic gradient"), and
  PySCF has no analytic CASSCF Hessian either. So the reference for a nuclear
  NEVPT2 Hessian would have to be a **finite difference of
  `nevpt2_grad.py`'s own analytic gradient** — an oracle that is itself the
  thing under test one order down, and the same construction the published
  implementations use as their *answer*.
- That stencil is `6N` displaced runs, each a full CASCI (or, on the CASSCF
  branch, a re-optimisation) plus a gradient, and each must pass the
  in-plane-`Cs` displacement rule
  ([`gradient-theory.md`](gradient-theory.md), §4.3) — a rule
  `reference_data/nevpt2_numgrad.py` already enforces by *refusing*
  displacements that change the detected point group or the core/active irrep
  labels.
- The selection hazard gets qualitatively worse, not just quantitatively.
  §4.1–§4.2 of [`gradient-theory.md`](gradient-theory.md) describe a selection
  as a discrete function of geometry: locally constant, non-differentiable where
  it changes. A kink in the energy is a *finite* error in the gradient and an
  **unbounded** one in the Hessian, and no residual in any solve can see it.
  The mixed `A != B` components are the exposed ones, because a two-coordinate
  stencil visits points no single-coordinate displacement does.
- PC adds the step-size bound of [`gradients.md`](gradients.md), §7 on top: a
  step large enough to lift a dropped mode past the cut puts a first-order
  error into the differenced quantity, and that bound now has to hold
  simultaneously with a *second*-difference step small enough to resolve
  curvature.

A reference you spend longer trusting than computing is not a reference. Levels
1 and 2 have exact identities available (§9); level 3 has a ladder against a
derivative of a derivative, and that asymmetry — not the integral work — is
what makes it a different project.

## 9. The exactness budget

The first-derivative tier's strength is that it is gated by identities, with
a finite difference only where a convention both modes share would otherwise
go unseen ([`gradients.md`](gradients.md), §2). Level 1 inherits that, and gains one
identity the first order does not have:

| check | what it pins | cost |
|---|---|---|
| **Eq. 40's second-order partner**, §2.3 | the new 3×3 alone, card-free, two exact zeros | free |
| **the finish's structural relations**, §2.1–§2.2 | `E_ND = N E_NH + Q`, `E_DD = N^2 E_HH`, and the `(H, Delta)` rank-one form | free |
| **`--rdm-tangent`'s Euler gate**, §3 | the second-order RDM builder in its `u = v = c` case — *already in the tree* | already run |
| **symmetry** `Eddot(u,v) == Eddot(v,u)` | little. The swapped-operand structure of the digest path makes it close to a tautology — it catches a bookkeeping slip, not a wrong derivative. Do not lean on it | free |
| **Euler** `Eddot(c,u) == Edot(u)`, §4.3 | the new 3×3 against the *already-gated* `Edot(u)`, along a direction no seeded pair reaches. Three SC / five PC classes, **frozen orbital energies only** | free |
| **duality** `<v, H_cc u> == Eddot(u,v)`, §4.2 | the full-vector `H_cc u` against the scalar form, to round-off, no step size. The second order's analogue of `<u,y> == Edot(u)` | one scalar pass per direction |
| **one finite difference of `Edot`** | that `Eddot` is the derivative of `Edot` at all — the shared-convention problem no identity can reach, pinned once exactly as `--sc-tangent` pins `Edot` | one ladder, once |
| **Hessian-vector ladders**, §5.3 | the orbital blocks, which cannot be checked block by block against an independent provider because they are an operator, not a matrix | per direction |

**One new finite difference for the whole of level 1.** That is the argument for
doing it: the second-derivative tier can be as nearly step-size-free as the
first, which is not a property a second derivative usually has.

*As built*, #156's acceptance adds a second ladder beside it: a difference of
the GPU's own `y` along `u` against `H_cc u`, the orbital-energy mode matched
on both legs. It pins no convention the duality leaves open (`Eddot`'s
difference already licenses the scalar the duality holds `H_cc u` to); it
holds the vector against the reverse walk's own `y` directly, independent of
`Eddot`. #184's adds the mixed block's, §5.5's one difference of the gated
`y` along `kappa`, run along each of the directions `--kappa-full` and
`--kappa-act` displace: unlike `H_cc u`'s it is not redundant, being the one
check that sees C5. PC's `H_cc u` (#155) has the same kind of ladder, of the
three-term `y`, per class, and for PC it is not redundant either: the duality
against `pcHessianDevice` cannot see §6.2's rotation terms, because both sides
share the blocks they are built from. PC's scalar `Eddot` (#152) carries the
twin of SC's one difference of `Edot`. The orbital blocks carry the
Hessian-vector ladders the table plans for: #154's second difference of the
fixed-pseudodensity pairing along `exp(s u + t v)`, the one check that sees an
omitted K2, and #158's of the SC total itself, which found §5.6's term; PC's
three (#200–#202) are the same ladders on PC's pairing, its three-term `y`
along `kappa` and its total. As built, then, each of level 1's ten modes
(five SC, five PC) runs a ladder of its own; [`gradients.md`](gradients.md),
§2 tabulates them with what each pins.

### 9.1 What a second derivative must refuse, and what it must merely say

This tree's discipline is that an unsupported or numerically invalid request is
**refused**, never approximated and never silently given a zero
([`architecture.md`](architecture.md), "Error handling"). The derivation above
fixes what that means at second order, and the list is worth having before any
code because two of its entries are not obvious.

| situation | what must happen | why |
|---|---|---|
| `--cublas`, `--ozaki` or `--fused-digest` with any Hessian | **refuse**, `err_unsupported` at flag parsing, before any other check on the digest flags (the tangent flags refuse the same digests as an invalid combination, [`gradients.md`](gradients.md), §8) | a second derivative built from an approximate digest is not the derivative of anything the energy stage computed — the same argument as for a tangent, with the same force |
| a Hessian flag on a binary that does not implement it | **refuse**, `err_unsupported`, naming the binary that does | a flag parsed and then ignored answers a question nobody asked, under the name of the one they did |
| a PC class the gap or denominator check refuses | **refuse** that class, `err_numerical` naming it, through the same status the energy, tangent and adjoint use | `PC_MIN_GAP` is §6.2's licence for the rank: no kept mode can sink under the threshold cut, which drops only this state's own null modes. A refused class has no valid second derivative, and a zero would be a wrong answer rather than a missing one |
| a PC (class, direction pair) along which the dropped modes lift: the Schur lift `L_uv = P (S_uv − S_u S⁺ S_v − S_v S⁺ S_u) P`, or `L_uu`, or `L_vv` (or, to first order, `P S_u P`), above round-off | **refuse** that (class, pair), `err_numerical` | along a direction that breaks the state's spatial symmetry the cut's dropped modes genuinely lift at `h²`, so the PC energy is not twice differentiable there (§6.2). A number would be the derivative of an energy whose rank changes under it |
| a PC class whose kept `S` is ill-conditioned — a small smallest kept `s / s_max` — with its gap above `PC_MIN_GAP` | **not** a refusal: **print** it | `PC_MIN_GAP` guards the rank, not the `S⁺` conditioning §6.2's rotation terms carry; the derivative is still the energy's. Saying it on the PASS line is what keeps it visible |
| a PC `H_kc` (class, CI direction `u`) along which `S`'s dropped modes lift: `P S_u P` or `L_uu` above `PC_LIFT_TOL` | **refuse** that (class, pair), `err_numerical`, through the class's status | `L_ww = L_wu = 0` identically, so the CI leg's test is the whole test (§6.5). A (class, `u`) with no second derivative in `u` has none in `(w, u)` either. The vector `H_kc^T w` has no `u` to test, so its consumer applies `u`'s test |
| a PC `H_kk` (class, orbital pair), active–active included | **not** a refusal, and no lift test runs | `S` does not move along `kappa` (§6.5), so the rank cannot change. Such a test would measure nothing |
| a PC tuple that is empty, by the integrals or by the state | **not** a refusal, and no sixth term | PC's energy is a quadratic form in `b_t`, and the general formula already holds the empty tuple's term (§6.6). SC's four guarded-tuple rows below do not apply to PC's seven solved classes. PC's Sijrs is SC's and takes SC's rows |
| a conventional case whose sidecar lacks the full MO-integral tensor K2 reads (§5.4), for a direction pair with neither member active–active | **refuse** `H_kk`, SC's and PC's alike (§6.7) | a partial `H_kk` is not a Hessian, and the omission would be invisible in every identity that pairs the two modes (both would omit it) — exactly how the §6 digest gap survived until #89 |
| a sidecar missing any array of the inventory it is read for | **refuse** the whole sidecar | the same argument: a partial inventory is a partial `H_kk` (or `H_kc`) that no pairing of the two modes would notice |
| a guarded tuple (`\|N_t\| <= NUMERICAL_ZERO`) that is **small**, a real perturber under the guard | **not** a refusal: contribute zero | the guard holds its energy at zero on a neighbourhood of the state, so zero is the derivative of the energy the code computes (§5.6) |
| a guarded tuple **empty by the integrals**, populated by at most one direction of the pair, or by both with proportional tables (`rho_t` at round-off) | **not** a refusal: contribute §5.6's K6, which is K4 at the ray-limit multipliers | `E_t` is a quadratic form along the pair. A zero here was the earlier version of this row, and #158's ladders measured it as wrong. Sijrs always falls in this row and simply takes `P = -1/Delta` (§5.6) |
| a guarded tuple **empty by the integrals** that both directions populate with non-proportional tables (`rho_t` above tolerance) | **refuse** that (class, pair), `err_numerical`, naming the class, the tuple count and the worst `rho_t` | the energy is not twice differentiable along the pair: the directional second derivatives exist but their polarisation is not bilinear (§5.6). A number here, even the one a ladder converges to, is not a Hessian element |
| a guarded tuple **empty by the state** (`x_t != 0`, `N_t` round-off against `\|x_t\|^2 \|S\|`) | **refuse**, any block | a CI direction can populate it too, and §5.6 derives the integral-empty case only. None of the census's goldens has one |
| the Euler gate with the orbital-energy response on | **not** a refusal: skip the gate and **say so** | §4.3 — the identity does not hold in that mode, but the *computation* is perfectly valid and is the one a nuclear gradient would consume. Refusing it would be refusing the useful mode. The PASS line names which mode ran, as `--gradient-source`'s already does |
| a `casscf`-kind golden | **not** a refusal, unlike the in-tree Lagrangian path | level 1 is **reference-agnostic**: it differentiates the energy functional, which does not know how its orbitals were made ([`gradient-theory.md`](gradient-theory.md), §2.1 makes this argument for `Y` and `y`, and §2 of this document inherits it). Only levels 2 and 3 branch on the reference |

The last two rows are the non-obvious ones, and they point opposite ways: one
mode must stay available with a weaker check, and one reference kind must
**not** inherit a refusal that the first-derivative path has for reasons that
do not apply here.

**PC's orbital blocks** (§6.4–§6.8) take three rows of their own, above. The
other rows apply to them unchanged:

- the approximate digests, the wrong binary, a gap- or denominator-refused
  class, a missing sidecar array and a `casscf`-kind golden;
- the Euler row, with `H_kc`'s two modes as SC's;
- the conditioning row. `H_kc`'s rotation term carries `S^+` once (through
  `C = U^T U`), so the smallest kept `s` sets its size as it sets the CI
  block's. `H_kk` carries no `S^+` at all, only the denominators `D_t`.

Two things are rules for what is built, not refusals:

- `H_kc` is never formed without its rotation term;
- `H_kc^T w` differentiates the three-term `y`, never the two-term one.

Leaving out either is invisible to the duality (§6.5).

**As built** (#157 tabulates it, #203 extends it to PC's orbital blocks; the
flags are #152–#158, #184, #190 and #200–#202). Seven Hessian flags exist:
`nevpt2_demo --sc-hessian` / `--sc-hessian-vector` (`H_cc`),
`nevpt2_sc_pseudodensity --sc-hessian-kc` / `--sc-hessian-kk-int` /
`--sc-hessian-kk` (`H_kc`, `H_kk`'s two terms and the whole block) and
`nevpt2_pc_tangent --pc-hessian` / `--pc-hessian-vector` (PC's `H_cc`). The
three orbital-block flags also take `--pc`, which makes them PC's `H_kc`,
curvature and `H_kk` (#200–#202) under every row below unchanged; `--pc` is
not a Hessian flag of its own and adds no binary. Each row above, as the tree
has it:

| row | as built | held by |
|---|---|---|
| approximate digest with any Hessian | `nevpt2.cli`'s `checkHessianDigests`, run in `parseArgs` **before** `finalize()`, so the kind is `err_unsupported` whatever else the digest flags would have drawn (`--fused-digest` with no GEMM digest, `--cublas` on HIP). `nevpt2_pc_tangent` implements none of the three, so it refuses the digest flag itself — `err_unsupported` too | `test/cli`'s `CliHessianRefusalTests`, every flag × every digest, both orders, and each orbital-block flag with `--pc` in every order |
| a Hessian flag on the wrong binary | the binary/flag table (`nevpt2.cli:binary_flags`; README.md, "Which binary takes which flag"): `err_unsupported`, "only `<binary>`", with `--pc` beside it or not | `CliHessianRefusalTests`, `CliMatrixTests` |
| a PC class the gap or denominator check refuses | the class comes back `PcStatus::Refused` with its reason, through the status the energy, tangent and adjoint use; the run FAILs | the PC Hessian entries' own tables |
| the PC Schur lift above round-off | `pcHessianDevice` (`P S_u P`, `P S_v P`, `L_uv`, `L_uu`, `L_vv`) and `pcCiHessianVectorDevice` (`P S_u P` and `L_uu`), relative to `s_max` against `PC_LIFT_TOL`, `err_numerical` through the class's status (#152, #155); and `pcMixedHessianDevice` (the CI leg's `P S_u P` and `L_uu`, the same test, #201), which returns the refusal as an `err_numerical` naming each refused class. Each check's negative control — a symmetry-breaking direction — must be refused for its lift, or the run FAILs. The separation the threshold sits in is [`pc-nevpt2.md`](pc-nevpt2.md), "What the cut drops, and how it moves" | `nevpt2_pc_hessian_cas44`, `nevpt2_pc_hessian_vector_cas44` and their `_casscf` twins; `nevpt2_sc_pseudodensity_cas44_pc_hessian_kc` and its fitted twin |
| PC conditioning | printed, never refused: each PC Hessian table has a `kept s` and a `gap` column, and each PASS line names the smallest kept `s / s_max`, its class, and the smallest gap against `PC_MIN_GAP`. What that conditioning costs is #194's | the PC Hessian entries' regexes ([`testing.md`](testing.md), "The Hessian on CASSCF orbitals") |
| K2 with neither direction active–active, conventional | `scOrbitalCurvatureDevice` refuses the pair as `err_unsupported`, never K1 alone (#154; a fitted case takes the three-index route instead; widening it is #180). `Eddot_kk` keeps the refusal, SC's and PC's (`pcOrbitalHessianDevice`, #202) alike | `--sc-hessian-kk-int`'s refusal checks; `--pc --sc-hessian-kk`'s check 7 |
| a sidecar missing an array | `GradGoldFile::requireGeneralInventory` (`nevpt2.golden`), which every reader of the general-range inventory — `--kappa-full`, `--source-oracle`, all three orbital Hessian flags — calls before uploading any of it: `err_io` naming the array, the whole sidecar refused | `test/golden`'s `GradGoldCas44Tests` / `GradGoldDfCas44Tests`: the committed sidecars with each array erased in turn |
| the four guarded-tuple rows | `scOrbitalSandwichDevice` (`nevpt2.gradient_plan`'s `emptyTermPacked`, #190). A small tuple contributes zero and is never scored. An integral-empty tuple both directions populate adds K6 at the ray-limit multipliers. Sijrs takes its own `P = -1/Delta` on every guarded tuple, through an unguarded finish at every order (`sijrsMultipliers`), so its first-order terms are exact too. A (class, pair) whose worst `rho_t` exceeds `kRhoTol` is refused as `err_numerical`, naming the class, the tuple count and the worst `rho_t`, and so is any tuple empty by the state | `--sc-hessian-kk`'s entries, whose PASS line counts the refused pairs; the thresholds and separations are [`testing.md`](testing.md), "The orbital block's sandwich" |
| Euler with the orbital-energy response on | the one block with a responding mode is `H_kc`: `--sc-hessian-kc` runs its FULL mode (the orbital energies responding to `kappa` and to `c`), prints that Euler is **skipped** there and why, and gates Euler in its EULER mode (held fixed in `c`); its PASS line names both. The `H_cc` flags run at fixed orbital energies only and say so on their PASS lines; `--gradgold` beside them is an invalid configuration, because nothing would read it — `H_cc`'s responding mode is not built. With `--pc`, `--sc-hessian-kc`'s scalar and vector run with the orbital energies responding to `kappa` (EULER) or frozen (FROZEN), and the responding-in-`c` mode is refused, `err_unsupported`, so no PC check skips Euler; `--sc-hessian-kk-int` runs EULER, `--sc-hessian-kk` both. Each PASS line names its mode | `nevpt2_sc_pseudodensity_cas44_casscf_sc_hessian_kc`'s regex, and the three `_cas44_casscf_pc_*` entries' |
| a `casscf`-kind golden | every Hessian flag, with `--pc` or without, passes `requireSourceTermReference`, which accepts both kinds, before any card is touched; `requireLagrangianReference` still has no call site in `src/` | one entry per flag on `n2_ccpvdz_cas44_casscf`, and one per orbital-block flag with `--pc` ([`testing.md`](testing.md), "The Hessian on CASSCF orbitals") |

## 10. What this settles, and what it does not

**Settled.**

- The energy's parameter Hessian needs **one** new hand-differentiated
  expression (§2), and that expression has two exact identities of its own
  (§2.3) and four independent entries rather than six (§2.2).
- The RDMs' second derivative **is** the existing tangent builder at two
  directions, and the third derivative is zero (§3). There is no tower behind
  this.
- `H_cc` is two terms: Park's multipliers against second-order blocks, and the
  new 3×3 between two first-order tangent triples (§4.1). Its vector form is
  two sigma calls (§4.2), gated exactly against the scalar form.
- The digests need **no second-order kernel**, because `f3` is linear in `h2e`
  (§5.2). The orbital Hessian's digest terms are the existing #88 and #89 paths
  at new arguments.
- PC's second derivative needs **no eigen-derivative, but it does need the
  null-space terms** (§6.2): the solve is the S-cut restricted inverse, not
  `A^+`, and the dropped space is exact for the state but turns with it (spin
  modes stay put; rank-bound and spatial-symmetry modes rotate), so `F_uv`
  carries rotation terms in `S^+ = U U^T` and `P = I - T U^T`, objects the
  solve already holds. A direction pair whose second-order lift `L_uv` is
  non-zero — one that leaves the state's symmetry — changes the rank at the
  order the Hessian probes and is refused for that class. A finite difference
  of the device's two-term `Edot` does not converge to the Hessian off the
  eigenstate; the three-term one does.
- `H_kk` is **five terms**, every one an existing contraction at new
  arguments (§5.1's table): the first-order pairing at the parameterisation's
  symmetric `W = (uv + vu)/2` (K1), the same pairings at a two-slot promotion
  (K2), and three terms that read only first-order dotted inputs — §2's 3×3
  between the forward tangent's per-tuple scalars (K3), the class terms with
  their two external slots dotted along different directions (K4), and one
  external slot against the Dyall block's tangent (K5). The new code is two
  input generators and the 3×3. **Plus a sixth term on the tuples the
  guard skips** (§5.6).
- **The empty perturbers (§5.6).** A tuple whose integrals vanish by
  symmetry is populated at first order by the non-totally-symmetric part of
  an orbital direction, never by a CI direction. Its energy along a pair is
  then homogeneous of degree 2. It is a quadratic form, adding
  K6 = `sum -N^{uv}/(Delta + eta)` (K4 at the ray-limit multipliers), exactly
  when its H tables are proportional to its N tables on the pair: always for
  Sijrs (`H = 0`, `P = -1/Delta`), along one direction, and for parallel
  perturbers. In general it is not, and the energy is then not twice
  differentiable along the pair. The ladder converges to the polarisation of
  the directional second derivatives, which is not bilinear, and the
  (class, pair) is refused on the scale-free defect `rho_t`. First order is
  untouched. `H_cc` and `H_kc` are untouched on every golden the census
  covers. A nuclear Hessian at a symmetric geometry inherits it in its
  non-totally-symmetric blocks, wherever an H class has empty tuples; PC
  does not inherit it at all.
- **The sidecar's sufficiency, settled (§5.4).** `H_kc` needs nothing new;
  `H_kk`'s K1 and K3–K5 need nothing new; K2 needs twelve two-general-index
  arrays whose union is the **whole MO-integral tensor** — four-virtual
  integrals included, so §3.4's omitted `(xv|vv)` comes back — on a
  conventional case, unless one direction is active–active, and **nothing new**
  on a density-fitted one, whose golden and sidecar already carry all of `B`.
- **`H_kc` is five terms and is built in two orientations (§5.5).** C1–C5:
  §2's 3×3 between the orbital and the CI tangent triples, one external slot
  against the block's CI tangent, the Dyall block and the digests each at
  both legs, and the orbital energies' mixed response — whose Fock line is
  the one new input generator. Built forward as the scalar
  `Eddot_kc(w, u)` (the tangent build at dotted integrals) and in reverse as
  the determinant vector `H_kc^T w` (`y` differentiated along `w`: two
  `ciSourceVector` calls, no transpose, plain-build tile floors), gated by
  `<H_kc^T w, u> == Eddot_kc(w, u)`. The orbital-vector orientation — the
  one that would need #89's transpose applied to tangent digests — is not
  built: every consumer contracts `H_kc` between two vectors, for which the
  duality makes `H_kc^T w` sufficient. What the duality cannot see, a term
  both omit, is held by Euler in `c` against the already-gated `Y`
  (`Eddot_kc(w, c) == 2 <w, Y>`, homogeneous classes), the intra-space exact
  zeros of the active-integral terms, and one finite difference of the gated
  `y` along `kappa`.
- **PC's orbital blocks (§6.4–§6.8), settled before code (#199).**
  - **`S` and `S_1` do not move along any `kappa`, active–active included.**
    The tree rotates the integrals at fixed CI. The orthogonal turn of `S`
    belongs to the gauge-equivalent CI direction `what c`. So `H_kk` is
    §6.1's formula on a fixed kept span: no projector, rotation or lift
    term. It is `<Ehat'_pc, I''>` (K1+K2, at the dotted sets #154's
    generators make) plus one bilinear sandwich `2 g_w^T M g_v`, with no
    3×3, no K3–K5 split and no K6.
  - **`H_kc` carries one rotation term**, `-2 rho^T S_u P A_w x`, along
    every `w` that moves `K` (all blocks but core–core and
    virtual–virtual), on rank-bound and symmetry modes. A Dyall block at
    dotted integrals couples the dropped space from the ket side. `H_kc`'s
    reverse orientation must differentiate the three-term `y`. Its only
    lift test is the CI leg's (`P S_u P`, `L_uu`).
  - **PC has no empty-perturber term and no `rho_t` refusal**: its energy is
    a quadratic form in `b_t`.
  - **`Ehat'_pc`** is #74's set. K2's conventional refusal is SC's.
  - **Two step-free gauges** hold `H_kc` against `H_cc` and `H_kk` against
    `H_kc` along active–active directions.
  - **One precondition:** on a density-fitted case, PC's first-order
    orbital term must be gated before K1/K2 run there.
- The exactness budget costs **one** new finite difference (§9).
- A second derivative needs the wavefunction parameters only to **first** order
  and the multipliers only to **zeroth** — the 2n+1/2n+2 rules, §7. So no
  second-order parameter response is ever formed, and level 1's operator is
  *consumed* by the `3N` responses rather than solved against (§8). The solves
  are of the **reference's** Hessian, which PySCF provides and the offline
  gradient already builds.
- A nuclear Hessian's hard part is its oracle, not its integrals (§8) — and the
  published NEVPT2 derivative implementations agree, taking their second
  derivatives seminumerically from their own analytic gradients.

**Answered by building it.** This list used to open the "not settled" one;
the Hessian milestone (#149–#158, #184, #190) and the PC orbital Hessian one
(#199–#203) answered each, and level 1 is in the tree, SC's and PC's every
block ([`gradients.md`](gradients.md), §1.1).

- **The tile floors (§4.1, §4.4).** The three-argument build holds one `L2`
  and one `R` block per vector, `3 n^4 + 4 n^2` doubles per determinant
  against the tangent build's `2 n^4 + 3 n^2`, so its floors sit at or above
  the tangent's, as §4.4 predicts; every other Hessian flag's floor is a build the
  tree already had (#153, #156, #184, #154, #158). Which build sets each
  flag's floor, and the measured floors, are
  [`performance.md`](performance.md), "Tile floors" and "Which build sets a
  Hessian flag's floor". At the integral-direct CAS(12,12) that floor sits
  above the tile count the plain build is recommended at.
- **`H_kc`'s route (§5.5).** Settled before code (#183) and built as settled
  (#184): the forward scalar and the reverse determinant vector, gated by
  duality, Euler in `c` and one finite difference of `y` along `kappa`. The
  orbital-vector orientation is not built, on purpose.
- **PC's `H_kc` (§6.4, §6.5).** Built as derived (#201): the same two
  orientations, with §6.5's one rotation term in the scalar and its adjoint
  in the derivative of the three-term `y`, gated by duality, Euler in `c`,
  §6.8's active-rotation gauge and per-class differences of the three-term
  `y`. The orbital energies are held fixed in `c` (PC's CI side holds them
  there), and a responding-in-`c` mode is refused.
- **PC's `H_kk` (§6.4, §6.6, §6.8).** Built as derived (#202): the sandwich
  is one bilinear form per tuple, with no rotation, projector or lift term
  and no guarded-tuple census, forward against reverse in both orbital-energy
  modes. The ladders of the PC total converge to the assembled `Eddot_kk` on
  every block-diagonal pair, virtual x virtual included, which SC refuses,
  and §6.8's gauge against #201's `H_kc` holds step-free. CAS(8,8) stays a
  diagnostic, for #194's reason.
- **PC's curvature `<Ehat'_pc, I''>` (§6.7) and its precondition.** Built as
  derived (#200): #154's generators against #74's PC adjoint, under SC's K2
  refusal, with PC's first-order orbital source term gated first on every
  case, density-fitted included, which meets §6.7's precondition.
- **PC's orbital blocks under §9.1 (#203).** `--pc` with each orbital-block
  flag takes every row of §9.1 that SC's does: the approximate digests and
  the wrong binary refused as `err_unsupported`, and a `casscf`-kind golden
  accepted, each orbital-block flag running on CASSCF orbitals with its
  PASS line naming its orbital-energy mode.
- **PC's second-order licence (§6.2).** The cut is a threshold cut whose
  dropped modes are this state's own null modes, so the null-space terms do
  not drop: the rotation terms are carried, and a (class, pair) along which
  the dropped modes lift is refused on `L_uv`, `L_uu`, `L_vv` and `P S_u P`
  against `PC_LIFT_TOL` (#152, #155). `PC_MIN_GAP` stays the rank's licence
  and is not the conditioning's: the smallest kept `s / s_max` is printed on
  every PC Hessian PASS line, never refused (§9.1).
- **The empty perturbers (§5.6).** Built as derived (#190): K6 on the
  integral-empty tuples whose tables are proportional on the pair, and an
  `err_numerical` refusal of every (class, pair) whose worst `rho_t` exceeds
  the device's threshold, and of any tuple empty by the state.
- **K2's data on a conventional case (§5.4), decided as a refusal.** #154
  builds K2 where the data already covers it — a direction pair with an
  active–active member, or a density-fitted case by the three-index jets —
  and refuses every other conventional pair as `err_unsupported`, never K1
  alone. Which data source lifts that refusal is #180's question.

**Not settled, and these are the questions worth answering before more code.**

- Where a conventional case's K2 gets its data (#180): an opt-in full
  MO-integral array written by `generate_golden.py`'s `--gradgold` path —
  affordable only where a full dump is, which is a size question for
  [`reference-data.md`](reference-data.md) — or an AO-direct second-order
  transformation, which needs AO integrals nothing in the tree carries.
- PC's second derivative on a poorly conditioned metric (#194). §6.2's
  second-order formula assumes `P A = 0`, which holds only to the golden's CI
  residual, and `S^+` amplifies that residual by the inverse of the smallest
  kept `s`. Where that is small the second-order checks converge to a value
  that stands off the analytic one, so the PC Hessian's gated entries are the
  CAS(4,4) ones ([`pc-nevpt2.md`](pc-nevpt2.md), "The second derivative,
  Eddot(u, v)").
- Level 2's CASSCF-branch third-derivative term (§7): whether anything in PySCF
  reaches it, or whether it would have to be built from `gen_g_hop`'s operator
  by differentiation.
- Whether the Euler check's frozen-orbital-energy restriction (§4.3) can be
  lifted by carrying the affine part explicitly, which would recover an
  identity in the sidecar-fed mode rather than leaving that mode checked only
  by duality.
- The device's thresholds for §5.6's refusal are measured on the N2 cases
  #190 runs ([`testing.md`](testing.md), "The orbital block's sandwich"),
  the way `PC_LIFT_TOL` was. A case with small H-class tuples, or a smaller
  real population than N2's, has not been run on the card.
- A tuple empty by the state (§5.6): its term in `H_cc` and `H_kc` is not
  derived here, and no committed golden in the census exercises it. Until a
  case has one, it is refused.

## 11. The order the pieces can be built in

Not an implementation plan. Stages 1–6 have since been built, each behind
its gate, and what each became is [`gradients.md`](gradients.md), §1.1.
What belongs here is the **dependency order the derivation imposes**, and
the gate each piece can be held to, because those are consequences of
§2–§9 rather than choices.

Two properties make the ordering unusually forgiving. Each stage's gate is
available *before* the next stage exists, so nothing has to be built on trust;
and the stages share almost no new machinery, so a later one being expensive
does not make an earlier one worse.

| # | piece | gate available at that point | new machinery |
|---|---|---|---|
| 1 | §2's 3×3 in the `d` form, host only | §2.3's two exact zeros, §2.1–§2.2's three structural relations. Card-free | one host expression. Nothing else in the document can be checked until this is right, and everything downstream reuses it (§2.4) |
| 2 | `Eddot(u, v)`, SC, scalar (§4.1) | §4.3's Euler, frozen `eps` — against the *already-gated* `Edot(u)`. Plus the one finite difference of `Edot` (§9) | the three-argument slab walk (§4.4) |
| 3 | `H_cc u`, SC, vector (§4.2) | the duality `<v, H_cc u> == Eddot(u,v)` against stage 2, round-off exact; column by column where `ndet` allows (§4.4) | `gdot` through the pseudodensity walk, two sigma calls |
| 4 | PC's `Eddot` and `H_cc u` (§6) | PC's Euler covers **five** classes against SC's three, plus the same duality; the refusals of §9.1 carried | extra solves against the existing factorisation. No new derivation (§6.3) |
| 5 | `H_kc`: the scalar `Eddot_kc(w, u)` and the vector `H_kc^T w` (§5.5) | the duality `<H_kc^T w, u> == Eddot_kc(w, u)`, round-off exact, column by column where `ndet` allows; Euler in `c`, `Eddot_kc(w, c) == 2 <w, Y>` against the *already-gated* `Y`, homogeneous classes, orbital energies fixed in `c`; the intra-space exact zeros of C3–C4; one finite difference of the gated `y` along `kappa`; the full-range `f` identity for C5's generator | the walk carrying a CI tangent set beside the orbital dotted inputs (F); stage 3's `gdot` walk fed orbital operand tangents, with the RDM sweep at dotted integrals (R1); C5's Fock line; the per-tuple merge of the digests' orbital share that stage 6's K3 reuses. No data the sidecar lacks, no transpose, no three-vector working set |
| 6 | `H_kk` as an action (§5.1–§5.6) | K1 reuses the pairing `Y` is built from, already validated block by block against PySCF, at a symmetric argument; the traced-pair zero for active–active pairs (§5.1); Hessian-vector ladders along `exp(s u + t v)` (§5.3), which on a pair §5.6 refuses converge to a polarisation, not to a Hessian element, and must not be held to one | the symmetric-argument and two-slot promotions (§5.1's table), and K2's data source (§5.4) — the full tensor on a conventional case; K6 and the `rho_t` refusal on the guarded tuples (§5.6). The expensive stage |
| 7 | levels 2 and 3, offline (§7–§8) | a stencil over `nevpt2_grad.py`'s analytic gradient — the oracle §8 argues against | `3N` reference response solves, second-derivative integrals, the constraints' second derivatives. A different kind of project |

Stage 1 is a prerequisite for all six others and is the smallest piece in the
document. Stages 2–4 are the ones §9's "one new finite difference" claim covers
in full. Stage 5 needs stages 2 and 3 (the sandwich, and the walk's tangent)
and nothing from stage 6, and goes first of the two orbital stages because it
is the cheap one and lands a piece stage 6 reuses; its PC half rides on stage
4. Stage 6's enumeration is done (§5.1, §5.4), and what it found is that the
orbital block's cost is set by its data, not its algebra: K2 reads the whole
integral tensor on a conventional case.
Stage 7 is where the character of the work changes, and §8 is the argument for
not starting there.

## References

[`references.md`](references.md) holds the literature. Park's gradient paper
(J. Chem. Theory Comput. **15**, 5417 (2019)) is the source of every equation
number cited above — Eqs. 36–38 for the multipliers, Eq. 40 for the identity
§2.3 extends, Eqs. 26–27 for the source terms this document differentiates a
second time.

**There is no published analytic second derivative of any internally
contracted MRPT2**, NEVPT2 included, so unlike
[`gradient-theory.md`](gradient-theory.md) — which derives from a paper and
marks where it departs — this document has no external counterpart to be
checked against as a whole. It does not follow that none of it has external
support, and the distinction matters for how much each part should be trusted:

| part | external support |
|---|---|
| §2, the scalar finish's 3×3 and its identities | **none needed.** Elementary calculus on a rational function of three scalars, and verified against a central difference and its own two exact zeros |
| §3–§5, the structural results | **none available, and none needed.** These are claims about *this tree's* multilinearity — the RDMs' exact quadraticity, the block builders' linearity, `f3`'s linearity in `h2e` — each checkable by inspection of the code they describe. §3–§4's chain was additionally verified on a model carrying the same algebra (quadratic densities, RDM-linear blocks, the rational finish): §4.1's two terms against a central difference, §4.2's duality and the Hessian's symmetry to round-off, and §4.3's Euler identity holding in the homogeneous frozen-`eps` case and **failing** in both of the cases §4.2–§4.3 say it must — an RDM-free term, and a responding `eps` |
| §5.6, the empty perturbers | **none found** in the literature we could read (§5.6, "Related work", which lists what is known around it and what was not read). Elementary: a degree-2 homogeneous rational function and when it is a quadratic form. Checked on a host model running PySCF's own class formulas on the committed goldens (`reference_data/sc_empty_perturber_probe.py`): the sixth term against a Richardson ladder, the parallelogram defect, the census, and `rho_t`'s separation ([`testing.md`](testing.md), "The empty-perturber term, offline"), plus synthetic hand cases. It was found the hard way: §5.1 said "no sixth term" until #158's ladders disagreed |
| §6.4–§6.8, PC's orbital blocks | **none found**, and none needed beyond §6.2's. The S-cut inverse along a direction that leaves `S` fixed is elementary, and the surviving rotation term is §6.2's formula with one leg orbital. Checked on a host model (`reference_data/pc_orbital_hessian_probe.py`, [`pc-nevpt2.md`](pc-nevpt2.md), "The orbital blocks, offline"): `H_kk` against energy ladders, `H_kc` with and without the term against a difference of the exact orbital derivative, both gauges, and the empty tuples' ladders |
| §6, the pseudo-inverse differentiated twice | **Golub–Pereyra**, and it was needed: the formula first written here carried a single `A^+` in the range-complement terms where the correct one carries `A^+ A^+` ([`references.md`](references.md), "Numerical linear algebra"). The null modes that lift, `L_uv` and the projector terms: textbook mathematics (Stewart, Kato) with partial precedents in quantum chemistry, none stated in this form (§6.2, "Related work") |
| §7–§8, what order of response a second derivative needs | **Helgaker–Jørgensen's 2n+1/2n+2 rules**, and these were needed too: the operator the `3N` responses solve against is the *reference's* Hessian, not level 1's or level 2's ([`references.md`](references.md), "Derivative theory") |
| §8's oracle argument | **confirmed by practice.** Both 2019 NEVPT2 first-derivative papers take their second derivatives seminumerically from their analytic gradients |

So §2–§5 stand on the identity-first validation of §9, which is what that
section is for; §6–§8 stand on established derivative theory, and were wrong in
two specific places until they were checked against it. Neither is a reason for
confidence in the parts that have not yet been implemented and measured.
