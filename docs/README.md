# docs/

Current-state documentation: what the code **is**, what it costs, and how it
is checked.

| file | answers |
|---|---|
| [`architecture.md`](architecture.md) | How is the tree put together? The `wwr*` layering that lets one `src/` build for CUDA or HIP, the host-modules / device-C++20 boundary and the `*_bridge.h` seam, the component graph, the one-stream / one-pool / RAII discipline, the one host integer type, the two-tier error model (abort on our bug, a returned `Result` otherwise), and the build layer. |
| [`implementation.md`](implementation.md) | How does a number get computed? CI vector → link tables → produce `R`/`L2` → digests → fdm2/wedge → `dm3`/`f3ac`/`f3ca` → the slab walk and 146 einsums → `normToEnergy` → `PASS:`, then the PC fork off the same `S` and `K`. |
| [`performance.md`](performance.md) | What does it cost, and which knobs matter? Where the time goes, scaling with active-space size, the head-to-head against PySCF, the four digests, memory and `--tiles`, the one-stream rule, and what was measured not to help. |
| [`pc-nevpt2.md`](pc-nevpt2.md) | How is PC-NEVPT2 done on the device, and how well does it agree? The block2 reference, the once-per-class eigensolve and the singular-metric convention, each class, accuracy and cost. |
| [`reference-data.md`](reference-data.md) | Where do the golden files come from, and why can they be trusted? Why CASCI, the reproducibility and symmetry settings, the PC fields, density fitting, every committed and gitignored case. |
| [`testing.md`](testing.md) | How is it checked? The golden and unit tiers, why there is no CI, the local gate, the sanitizer tier with what each tool can and cannot see, and host code coverage. |
| [`references.md`](references.md) | What is this built on? The literature cited, and the software with where each version is pinned. |

## Where numbers live

**`architecture.md`, `implementation.md` and `references.md` carry no
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

If you are about to add a figure to `architecture.md` or
`implementation.md`, it belongs in one of the four measured documents instead.

## Related

| file | role |
|---|---|
| `../README.md` | the summary — what it is, status and limitations, the quickstart, requirements, running the demos, "One tree, two backends" |
| `cmake/README.md` | the CMake macro layer |
| `docker/README.md` | the container images and how they chain |
