# Hessian theory: the NEVPT2 energy's second derivatives

What an analytic *second* derivative of this energy would require. The
first-derivative derivation — the Lagrangian on each of the two references, the
constraint sets, the Z-vector equations and solve orders, the integral
inventory — is [`gradient-theory.md`](gradient-theory.md), and this document
builds on it rather than restating it: §0's notation, §0's `P`/`Q`/`R` chain
rule, §1.3's active–active gauge, §2.1's source terms and §3's inventory are
assumed throughout and cited by section number.

Like [`gradient-theory.md`](gradient-theory.md), it is **derivation, not code**,
and it carries **no measured numbers**. Nothing described here is in the tree.
Size formulas in `n_mo`, `n_core`, `n_act`, `n_virt` and `ndet` are algebra and
stay here; what any of them works out to in bytes or seconds belongs in
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
is one the *gradient* already needs** — the second derivative adds no new
differentiability requirement, which is the central point of this subsection:

| condition | which branch | why second order adds nothing |
|---|---|---|
| no intruder: `Delta_t + H_t/N_t` must not pass through zero | both | NEVPT2's Dyall zeroth order is what makes this rare rather than endemic; where it fails, the *energy* is singular, so there is no gradient either |
| the CASSCF Hessian must be non-singular at the solution | CASSCF | the implicit function theorem gives `theta(R)` as smooth as the stationarity conditions, which are `C^infinity`. **One derivative and two need the same condition**; this is `gradient-theory.md`, §2.7's conditioning statement, unchanged |
| the selection must be locally constant, and the pseudocanonical divisions non-degenerate | CASCI | `gradient-theory.md`, §4.1–§4.3 and §2.3's refusal. A selection change is a kink: the *gradient* is already wrong there |
| the PC metric's rank must be locally constant | PC only | the dropped modes are linear dependencies among **active-space operator strings** (§6.2), which depend on the active-space size and the spin/particle sector and **not on geometry**. So the structural rank is constant along *any* displacement, and PC is no worse than SC here |
| no root crossing | both | `gradient-theory.md`, §4.4 |

The PC row is the one that could have gone the other way, and it does not: PC's
pseudo-inverse is differentiable along nuclear displacements for the same
structural reason it is differentiable in the CI vector.

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

Two things that are **not** blockers, despite looking like candidates. The
eigendecomposition PC runs per class is never differentiated (§6.1), so no
degenerate-eigenvalue case arises. And a vanishing perturber norm is benign:
`E_t = -N_t^2/d_t` tends to zero with `N_t`, and all six second partials stay
finite in the `d` form (§2.1).

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
evaluates `0/0` and returns a NaN while the `d` form returns zero. That is
precisely the case the guard exists for, so it is the case that will be hit. In
`d` the same six partials are `1/N`-free:

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
`E = sum_t f(N_t, H_t, Delta_t)` at fixed `c` then has exactly five terms, and
there is no sixth: `N_t` reads no active integral, `K` has no second
derivative in the integrals, and `Delta_t` none in `eps`.

| # | term | what it is | existing contraction it reuses | arrays it reads |
|---|---|---|---|---|
| K1 | **same-slot curvature** (§5.1's first term, half one) | `<Ehat', z'[W]>`: the full first-order pairing at `W` | `scFullRotationDevice`'s pairing, either mode — `adjoint` = `ext + act + eps` plus the digests' `<hbar, h2e'[W]>` (#89), or `tangent` plus #88's digest term — at `W` instead of an antisymmetric `kappa` | **exactly `Y`'s**: the sidecar's eighteen integral arrays (§3.3's seventeen rows, `(xc\|cc)` on its two slices), `h`, `heff`, `f`, plus the golden's stored blocks for the intra-active part |
| K2 | **cross-slot curvature** (§5.1's first term, half two) | `<Ehat', z^x[u,v]>` | the same pairings — `<D, X>` per external block, `<M_G, G>` per active block, `<hbar, h2e>` for the digests, `R_t` against `Delta_t` — at a new input | **two-general-index arrays**: §5.4's table. **Not in the sidecar** |
| K3 | **the finish's curvature** (§5.1's second term) | `sum_t (Ndot, Hdot, Ddot)_t[u] . Hess f_t . (Ndot, Hdot, Ddot)_t[v]` | the forward tangent's per-tuple `Ndot_t` / `Hdot_t` (`assembleSlab`'s `normDot` / `hDot`, before they meet `P` and `Q`) and `Ddot_t` (the class's `deltaOf` at `deps`), at `u` and at `v`; §2's 3×3 between them | the sidecar only: first-order dotted inputs `z'[u]`, `z'[v]` |
| K4 | **both external slots, one per direction** (second term) | `sum_t [ P_t N_t^{uv} + Q_t H_t^{uv} ]`: every norm and H term with its two `x` slots filled by `X'[u]` and `X'[v]`, symmetrised, `K` undotted | the class term lists under the product rule (`:terms`' `evalTermsTangent`), with the two slots' dotted operands drawn from different directions | the sidecar only |
| K5 | **one external slot and the Dyall block** (second term) | `sum_t Q_t [ H terms with one x slot at X'[u] and K at K'[v] ] + (u <-> v)` | the same evaluator; `K'[v]` is the forward tangent's own block tangent — `make_*` at `h1e'[v]`, `h2e'[v]` (`rotationTangentBlocks`) — plus `f3'[v]`, `buildRdmsDevice` at `h2e'[v]` (#88) | the sidecar only (`h2e'` reads `g_xa_aa`, `h1e'` reads `heff`) |

So `Eddot_kk(u, v)` is **five terms, every one an existing contraction at new
arguments**, and the new code is two *input generators* — the promotion at a
symmetric argument (K1) and the two-slot promotion (K2) — plus §2's 3×3. Per
direction pair `(u, v)` that is:

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
| `H_kk`, K1 and K3–K5 | **no.** K1 reads exactly `Y`'s arrays at a symmetric argument; K3–K5 read only first-order dotted inputs (§5.1's table) |
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
in the orbital energies.

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
`G'_w = -G A'_w G` exactly. Whatever §6.2 settles for the pseudo-inverse's
derivative along a CI direction is used once per `(w, u)`, and nothing about it
moves along `w`. F is `pcTangentsDevice`'s walk beside
`pcFullRotationDevice`'s orbital tangents; R1 is PC's adjoint (`pcCiSourceDevice`)
differentiated along `w` — the walk tangent PC's `H_cc u` builds, fed orbital
operands; Euler covers PC's five homogeneous classes.

**Data.** §5.4's row stands: nothing the sidecar lacks, the Fock line's four
rows included. A case with no sidecar can form `H_kc` only along active–active
`w` with the orbital energies held fixed in both legs — the functional
`--kappa-act` differentiates at first order — and must refuse any other `w`
rather than return the integral half alone (§9.1's discipline). The
density-fitted case needs no new data either (first-order `Bdot` per
direction), but `scFullRotationDevice` is integral-direct, and so is this.

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

### 6.2 The null-space terms drop — because the null space is structural, not small

The derivative of a Moore–Penrose pseudo-inverse at locally constant rank is
Golub and Pereyra's (Ref. in [`references.md`](references.md), "Numerical
linear algebra"), which for symmetric `A` reads

```
  dA^+ = -A^+ dA A^+ + A^+ A^+ dA P_N + P_N dA A^+ A^+
```

**Note the `A^+ A^+` in the two range-complement terms**: the easy mistake is
to write a single `A^+` there by analogy with the first term, and it is wrong —
a numerical check against a difference quotient on a rank-deficient matrix
whose null space *moves* misses by an O(1) relative amount, not by a small one.
It happens not to matter for the conclusion below, because both terms carry a
`P_N` factor either way, but the formula is the formula.

At first order the last two terms drop simply because `b_t` lies in the kept
range, so `P_N b = 0` ([`gradients.md`](gradients.md), §7). **That argument
alone does not survive to second order**: terms carrying
`P_N dA A^+ x` and `bdot^T P_N` appear, and `bdot` has no reason to lie in the
range.

They drop anyway, for a stronger reason than the first-order one, and it is
worth spelling out because it is the whole licence for PC's second derivative.
The modes the cut drops are **structural** — "linear dependencies among the
class's active-space operator strings, null for *every* CI vector, not
small-for-this-state" ([`gradients.md`](gradients.md), §7). A dropped mode `w`
is therefore a vector with `sum_mu w_mu |Phi_mu> = 0` *as a function*, not
merely a small eigenvalue of this state's metric. Hence, identically in `c`,

```
  S(c) w = 0   and   K(c) w = 0   for every c
  =>  dS . w = dK . w = 0  and  d^2 S . w = d^2 K . w = 0
  =>  P_N dA = dA P_N = 0  and  P_N A'' = A'' P_N = 0
```

so every null-space term vanishes at **every** order, not just the first —
`dA^+` collapses to its first term, `-A^+ dA A^+`, and stays collapsed under
further differentiation. (This is also why the `A^+ A^+` correction above
changes nothing here: it only ever multiplies a `P_N dA`.) The
condition that licenses this is the one the energy stage already enforces — the
singular-metric convention's structural cut, with `PC_MIN_GAP` guaranteeing
local rank constancy ([`pc-nevpt2.md`](pc-nevpt2.md), "The singular-metric
convention") — so the second derivative is valid precisely on the states the
energy stage accepts, and a class the gap or denominator check refuses must be
refused in the Hessian too, exactly as in the tangent and the adjoint.

Two honest qualifications:

- The identity is exact in the idealisation and approximate in arithmetic: the
  dropped eigenvalues are not exactly zero, only structurally so, and the
  measured magnitudes are the table in
  [`pc-nevpt2.md`](pc-nevpt2.md), "The singular-metric convention". Nothing
  amplifies them — `A^+` is identically zero on the dropped subspace, so no
  small denominator meets a small numerator — but the claim is "bounded by the
  measured null-mode size", not "zero".
- The conclusion depends on the cut being the *structural* one. A cut raised to
  drop physics ([`pc-nevpt2.md`](pc-nevpt2.md) measures what that costs the
  energy) would make `P_N dA` genuinely non-zero and put the dropped terms
  back. The second derivative is one more reason the cut must stay where it is.

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
exactly three finite differences in the whole of it
([`gradients.md`](gradients.md), §2). Level 1 inherits that, and gains one
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
`Eddot`.

### 9.1 What a second derivative must refuse, and what it must merely say

This tree's discipline is that an unsupported or numerically invalid request is
**refused**, never approximated and never silently given a zero
([`architecture.md`](architecture.md), "Error handling"). The derivation above
fixes what that means at second order, and the list is worth having before any
code because two of its entries are not obvious.

| situation | what must happen | why |
|---|---|---|
| `--cublas`, `--ozaki` or `--fused-digest` with any Hessian | **refuse**, `err_unsupported` at flag parsing, as the tangent flags already are ([`gradients.md`](gradients.md), §8) | a second derivative built from an approximate digest is not the derivative of anything the energy stage computed — the same argument as for a tangent, with the same force |
| a PC class the gap or denominator check refuses | **refuse** that class, `err_numerical` naming it, through the same status the energy, tangent and adjoint use | §6.2's licence is the structural cut. A refused class has no valid second derivative, and a zero would be a wrong answer rather than a missing one |
| a conventional case whose sidecar lacks the full MO-integral tensor K2 reads (§5.4), for a direction pair with neither member active–active | **refuse** `H_kk` | a partial `H_kk` is not a Hessian, and the omission would be invisible in every identity that pairs the two modes (both would omit it) — exactly how the §6 digest gap survived until #89 |
| `N_t = 0` exactly | **not** a refusal: contribute zero | `E_t = -N_t^2/d_t -> 0` and §2.1's `d` forms all stay finite. A NaN here would be an implementation bug, so this is a `check`-able invariant, not an error path |
| the Euler gate with the orbital-energy response on | **not** a refusal: skip the gate and **say so** | §4.3 — the identity does not hold in that mode, but the *computation* is perfectly valid and is the one a nuclear gradient would consume. Refusing it would be refusing the useful mode. The PASS line names which mode ran, as `--gradient-source`'s already does |
| a `casscf`-kind golden | **not** a refusal, unlike the in-tree Lagrangian path | level 1 is **reference-agnostic**: it differentiates the energy functional, which does not know how its orbitals were made ([`gradient-theory.md`](gradient-theory.md), §2.1 makes this argument for `Y` and `y`, and §2 of this document inherits it). Only levels 2 and 3 branch on the reference |

The last two rows are the non-obvious ones, and they point opposite ways: one
mode must stay available with a weaker check, and one reference kind must
**not** inherit a refusal that the first-derivative path has for reasons that
do not apply here.

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
- PC's second derivative needs **no eigen-derivative and no null projector**:
  the null-space terms that the first-order "`b` is in the range" argument
  would have left behind drop for a stronger reason — the cut is structural, so
  `S`, `K` and all their CI derivatives annihilate the dropped subspace
  identically (§6.2).
- `H_kk` is **five terms**, every one an existing contraction at new
  arguments (§5.1's table): the first-order pairing at the parameterisation's
  symmetric `W = (uv + vu)/2` (K1), the same pairings at a two-slot promotion
  (K2), and three terms that read only first-order dotted inputs — §2's 3×3
  between the forward tangent's per-tuple scalars (K3), the class terms with
  their two external slots dotted along different directions (K4), and one
  external slot against the Dyall block's tangent (K5). The new code is two
  input generators and the 3×3.
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

**Not settled, and these are the questions worth answering before any code.**

- The three-tangent-set slab walk's working set, and therefore its tile floors
  (§4.1). This is a measurement, and it decides which committed cases can run a
  Hessian at all.
- Where a conventional case's K2 gets its data (§5.4): an opt-in full
  MO-integral array written by `generate_golden.py`'s `--gradgold` path —
  affordable only where a full dump is, which is a size question for
  [`reference-data.md`](reference-data.md) — or an AO-direct second-order
  transformation, which needs AO integrals nothing in the tree carries; or
  K2 on the density-fitted pair alone, in three-index form.
- Level 2's CASSCF-branch third-derivative term (§7): whether anything in PySCF
  reaches it, or whether it would have to be built from `gen_g_hop`'s operator
  by differentiation.
- Whether the Euler check's frozen-orbital-energy restriction (§4.3) can be
  lifted by carrying the affine part explicitly, which would recover an
  identity in the sidecar-fed mode rather than leaving that mode checked only
  by duality.

## 11. The order the pieces can be built in

Not an implementation plan — that is a separate discussion, and this document
describes no code. What belongs here is the **dependency order the derivation
imposes**, and the gate each piece can be held to, because those are
consequences of §2–§9 rather than choices.

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
| 6 | `H_kk` as an action (§5.1–§5.4) | K1 reuses the pairing `Y` is built from, already validated block by block against PySCF, at a symmetric argument; the traced-pair zero for active–active pairs (§5.1); Hessian-vector ladders along `exp(s u + t v)` (§5.3) | the symmetric-argument and two-slot promotions (§5.1's table), and K2's data source (§5.4) — the full tensor on a conventional case. The expensive stage |
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
| §6, the pseudo-inverse differentiated twice | **Golub–Pereyra**, and it was needed: the formula first written here carried a single `A^+` in the range-complement terms where the correct one carries `A^+ A^+` ([`references.md`](references.md), "Numerical linear algebra") |
| §7–§8, what order of response a second derivative needs | **Helgaker–Jørgensen's 2n+1/2n+2 rules**, and these were needed too: the operator the `3N` responses solve against is the *reference's* Hessian, not level 1's or level 2's ([`references.md`](references.md), "Derivative theory") |
| §8's oracle argument | **confirmed by practice.** Both 2019 NEVPT2 first-derivative papers take their second derivatives seminumerically from their analytic gradients |

So §2–§5 stand on the identity-first validation of §9, which is what that
section is for; §6–§8 stand on established derivative theory, and were wrong in
two specific places until they were checked against it. Neither is a reason for
confidence in the parts that have not yet been implemented and measured.
