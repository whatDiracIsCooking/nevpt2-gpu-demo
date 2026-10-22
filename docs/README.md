# docs/

Current-state documentation: what the code **is**, what it costs, and how it
is checked.

| file | answers |
|---|---|
| [`architecture.md`](architecture.md) | How is the tree put together? The `wwr*` layering that lets one `src/` build for CUDA or HIP, the host-modules / device-C++20 boundary and the `*_bridge.h` seam, the component graph, the one-stream / one-pool / RAII discipline, the one host integer type, the two-tier error model (abort on our bug, a returned `Result` otherwise), and the build layer. |
| [`implementation.md`](implementation.md) | How does a number get computed? CI vector → link tables → produce `R`/`L2` → digests → fdm2/wedge → `dm3`/`f3ac`/`f3ca` → the slab walk and 146 einsums → `normToEnergy` → `PASS:`, then the PC fork off the same `S` and `K`. |
| [`energy.tex`](energy.tex) | What exactly does the density-fitted energy evaluate? LaTeX (`pdflatex energy.tex`, twice; the rendered [`energy.pdf`](energy.pdf) is committed beside it and must be rebuilt when the `.tex` changes): the four paths SC/PC × CASCI/CASSCF, why CASCI vs. CASSCF is settled offline in the golden and SC vs. PC at run time, the slab-by-slab DGEMM over `B`, the `dm3`/`f3ca`/`f3ac` digests, every SC class's norm and `H` and every PC class's solve, written as equations with pseudocode per path, and an appendix of the RDM-only intermediates rendered from the einsum chains themselves. |
| [`performance.md`](performance.md) | What does it cost, and which knobs matter? Where the time goes, scaling with active-space size, the head-to-head against PySCF, the four digests, memory and `--tiles`, the one-stream rule, and what was measured not to help. |
| [`pc-nevpt2.md`](pc-nevpt2.md) | How is PC-NEVPT2 done on the device, and how well does it agree? The block2 reference, the once-per-class eigensolve and the singular-metric convention, each class, accuracy and cost. |
| [`reference-data.md`](reference-data.md) | Where do the golden files come from, and why can they be trusted? Why CASCI, the reproducibility and symmetry settings, the PC fields, the file format and its sidecars (MO, gradient, source-term oracle), the offline numerical and analytic gradients, density fitting, every committed and gitignored case. |
| [`testing.md`](testing.md) | How is it checked? The golden and unit tiers, why there is no CI, the local gate (with the golden layout contract, the oracle's own tests and the regeneration gate), the gradient path's checks and the source-term oracles, the analytic gradient against the numerical one (SC and PC — and the one tolerance in this tree that is not 1e-7, with the reason), the sanitizer tier with what each tool can and cannot see, and host code coverage. |
| [`gradients.md`](gradients.md) | What does the gradient path compute, and what does it deliberately not? The source terms that are in the tree — the CI source term `y_I`, the SC pseudodensities, the PC tangent, the active–active orbital block — against the Z-vector solve, the derivative integrals and the nuclear gradient that are not; the forward/reverse duality that makes the permanent tests exact dot products rather than finite differences, and the three differentiability conditions the code enforces. Code, not derivation. |
| [`gradient-theory.md`](gradient-theory.md) | What would an analytic gradient of this energy require? The reference-independent half written once — notation and the `P`/`Q`/`R` chain rule, the active–active gauge, the source terms, the inventory of MO-integral arrays the orbital source term reads and the one it does not, its density-fitted counterpart (the three-index blocks it reads, the two the `--df` goldens lack, and why `κ` never reaches the fitting metric), the in-plane-`Cs` validation rule — and a branch per reference for the constraints, Z-vector equations and solve order: CASCI on RHF orbitals (CPHF, decoupled from the CI equation, with its active-space-selection hazard) and CASSCF orbitals (Park's Eq. 23 restored, one coupled CP-CASSCF system). The first-order related work, and why the Hessian's two refusals leave the gradient exact. Derivation, not code. |
| [`hessian-theory.md`](hessian-theory.md) | What would an analytic *second* derivative require? The three things called a second derivative and which of them is small — the energy's parameter Hessian, the Lagrangian's, the nuclear one — then the energy's in full: the scalar finish's 3×3 and Eq. 40's second-order partner, the RDMs' exact quadratic structure, the CI block in both modes, the orbital blocks as an action rather than a matrix — five terms, one of which reads the whole MO-integral tensor on a conventional case, plus a sixth on the perturbers the guard skips, and the symmetry-breaking pairs along which SC has no Hessian at all — and PC's solve differentiated twice, with the null-space terms that do not drop and the symmetry-breaking directions along which PC has no Hessian either. Each of the two "no Hessian here" findings comes with its related work and what the literature we could read does not state. Then what it must refuse, and the dependency order the derivation imposes. Derivation, not code; nothing in it is in the tree. |
| [`references.md`](references.md) | What is this built on? The literature cited, including the related work that the two Hessian refusals are placed against, and the software with where each version is pinned. |

## Where numbers live

**`architecture.md`, `implementation.md`, `energy.tex`, `gradients.md`,
`gradient-theory.md`, `hessian-theory.md` and `references.md` carry no
measured numbers.** Every timing, sweep, accuracy figure and negative result
lives in `performance.md`, `pc-nevpt2.md`, `reference-data.md` or
`testing.md`, with the card and flags it was measured with — *including the
things that turned out not to work* (the device-resident RDM build that was
no faster; several streams measuring the same as one; tiling as a memory
lever rather than a speed lever).

Each measurement lives in exactly one place. A number restated elsewhere would
drift from its source and become a second, quieter source of truth. So the
other documents, and the code's comments, cite a section by file and name —
`docs/testing.md, "Sanitizers"` — rather than repeat the figure.

If you are about to add a figure to any of those seven — `architecture.md`,
`implementation.md`, `energy.tex`, `gradients.md`, `gradient-theory.md`,
`hessian-theory.md`, `references.md` — it belongs in one of the four measured
documents instead. A *size formula* in `n_mo`, `n_core`, `n_act`, `n_virt` (and `naux`) is algebra
and stays with the derivation; the bytes it works out to for a given case are
a measurement and live in `reference-data.md`.

## Related

| file | role |
|---|---|
| `../README.md` | the summary — what it is, status and limitations, the quickstart, requirements, running the demos, "One tree, two backends" |
| `cmake/README.md` | the CMake macro layer |
| `docker/README.md` | the container images and how they chain |
