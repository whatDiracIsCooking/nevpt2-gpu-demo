# nevpt2-gpu-demo

**Second-order n-electron valence state perturbation theory (NEVPT2) on the
GPU, end to end.** Given a CASCI/CASSCF wavefunction (a CI vector plus MO
integrals), it builds the active-space 3-RDM and the two integral-contracted
4-RDM intermediates on the device, then assembles the eight perturber-class
energies on the device as well — for both the strongly-contracted
(**SC-NEVPT2**) and, with `--pc`, the partially-contracted (**PC-NEVPT2**)
variant. The 1- and 2-RDMs are read from the input.

- **One source tree, two vendors.** The same `src/` builds for NVIDIA (CUDA)
  or AMD (ROCm/HIP), through the backend-neutral
  [WarpWraps](https://github.com/whatDiracIsCooking/WarpWraps) runtime layer.
  Both have been run on real cards.
- **Checked, not just fast.** Every run compares each class energy and the
  total correlation energy against a committed reference — PySCF for SC,
  block2 for PC (PySCF has no PC-NEVPT2) — to 1e-7, and prints `PASS:` only
  if all of them match.
- **Integral-direct or density-fitted.** A second binary builds the external
  integrals on the device from three-index density-fitting tensors and checks
  against PySCF's DF-NEVPT2.

How it compares with PySCF's CPU contraction on the same CI vector, and how
that changes with active-space size, is measured in
[docs/performance.md](docs/performance.md), "Head-to-head vs PySCF". This
README carries no measured numbers; `docs/` holds all of them, including the ideas that didn't pan out.

## Status and limitations

This is a research code, not a quantum-chemistry package. Read this before
you try to run it on your own system.

- **The input is a `.nevpt2gold` file, not an interface to your CASSCF
  code.** Each file holds a CI vector, `dm1`/`dm2`, the MO-integral blocks and
  the reference answer. `reference_data/generate_golden.py` writes one from a
  PySCF CASCI (`--atom`, `--basis`, `--ncas`, `--nelecas`, `--df`, and
  `--cas-irreps`/`--core-irreps` to choose the active orbitals by symmetry),
  so a new molecule means generating a new file. That needs PySCF, and block2
  for the PC fields.
- **What has been checked.** The committed cases are N₂: cc-pVDZ CAS(4,4),
  (8,8), (10,10), (10,12) and (12,12), plus DF variants that include a
  cc-pVTZ CAS(10,10). The first non-N₂ case, salicylaldimine CAS(8,8)/6-31G**,
  is too large to commit and has to be regenerated. Two more non-N₂ cases are
  committed for the gradient work: LiF CAS(4,6) and acrolein CAS(6,5), both
  def2-SVP (LiF with no PC reference — see
  [`docs/reference-data.md`](docs/reference-data.md)). Three of the N₂ cases —
  CAS(4,4), CAS(10,10) and a DF CAS(4,4) — are committed a second time on
  converged CASSCF orbitals (next bullet). Anything else is untested.
- **The CI vector is read in, and so are `dm1`/`dm2`.** The GPU builds the
  3-RDM and the 4-RDM intermediates. A real CASSCF code produces the 1- and
  2-RDMs cheaply, and building them here as dense string-algebra operators
  would not fit in memory past tiny active spaces.
- **One active space and one CI vector per run.** There is no persistent
  device session for a geometry scan or a trajectory, so each run pays its
  whole setup again.
- **Every golden's state is a CASCI diagonalisation, never a fresh CASSCF in
  the regeneration chain** — that is what keeps it reproducible to the last
  bit. Which *orbitals* it runs on is recorded in the file: most committed
  goldens are a CASCI on RHF orbitals, and three are a CASCI on **converged
  CASSCF orbitals** committed beside them in an MO sidecar (N₂/cc-pVDZ
  CAS(4,4), CAS(10,10), and a DF CAS(4,4)), so the energy path has
  CASSCF-orbital references as well. The optimiser runs once, offline, into
  that sidecar; regenerating the golden from it is deterministic
  (docs/reference-data.md, "What "CASSCF input" means here, and why the golden state is CASCI").
- **The gradient work stays CASCI-only.** Its derivation is the CASCI-on-RHF
  Lagrangian, so every derivative path refuses a CASSCF-orbital golden rather
  than differentiate a reference it does not cover. One gradient sidecar is
  written for one, as data no derivative path reads yet
  (docs/reference-data.md, "The CASSCF-reference gradient sidecar").
- **On the GPU, the gradient work is source terms, not a nuclear gradient.**
  The binaries differentiate the NEVPT2 energy with respect to the CI vector
  and with respect to an orbital rotation (the active–active block, or the
  whole orbital source term out of a `*.gradgold` sidecar), at one fixed
  geometry. The Z-vector solve, the derivative integrals and any derivative
  with respect to a nuclear coordinate are not implemented in `src/` or
  `apps/`. A nuclear gradient exists only in offline Python:
  `reference_data/nevpt2_grad.py` solves the Z-vector equations and builds
  an analytic one (from PySCF's source terms, or from the GPU's own, written
  out by `nevpt2_sc_pseudodensity --dump-source-terms`), and
  `reference_data/nevpt2_numgrad.py` writes a *numerical* one into a
  sidecar's `grad_num`, which nothing in `src/` or `apps/` reads
  ([`docs/gradients.md`](docs/gradients.md)).
- **It needs a GPU.** There is no CPU path.

## Quickstart

```bash
git clone --recurse-submodules https://github.com/whatDiracIsCooking/nevpt2-gpu-demo.git
cd nevpt2-gpu-demo
cmake --preset default && cmake --build --preset default -j   # CUDA; --preset hip for ROCm
ctest --preset fast        # every check not labelled `slow`; needs a card
ctest --preset unit        # the GoogleTest unit tier; its host-only suites need no card
```

If you cloned without `--recurse-submodules`, run
`git submodule update --init --recursive`: `deps/WarpWraps` is required, and
configure stops without it. Configuring with testing on (every preset but
`compile-cuda`/`compile-hip`) fetches GoogleTest `v1.17.0`, so the first configure needs
the network.

There is also a standalone build with no ctest and no network:

```bash
cmake -S src -B src/build && cmake --build src/build -j   # add -DNEVPT2_GPU_BACKEND=HIP for ROCm
./src/build/nevpt2_demo --golden golden/n2_ccpvdz_cas1010.nevpt2gold --tiles 3
```

A run prints each class's norm and energy beside the reference's, the total
correlation energy, the time each GPU stage took, and finally
`PASS: per-class norms/energies and the total match the golden PySCF reference to 1e-07`
(with `--pc`, a `PASS: PC-NEVPT2 ...` line too). A run that doesn't match
exits non-zero. What each stage costs is in
[docs/performance.md](docs/performance.md), "Where the time goes".

## Requirements

### Hardware

- **One GPU, and nothing runs without it.** Every reference check
  (`ctest --preset fast`, either demo) needs a real card. Only the host-only
  GoogleTest suites run card-free (`ctest --preset unit -LE gpu`; some unit
  suites are labelled `gpu` too), and they check logic, never an energy.
- **The two cards it has been run on**, and the ones the presets target:

  | backend | card | arch | preset |
  |---|---|---|---|
  | CUDA | NVIDIA RTX 3080 (10 GB) | `sm_86` | `default` |
  | HIP / ROCm | AMD RX 9060 XT (16 GB) | `gfx1200` | `hip` |

  Another card is a configure flag (`-DCUDA_ARCH=sm_XX` or
  `-DHIP_ARCH=gfxNNNN`). It is also unmeasured: no reference has been checked
  on it.
- **Device memory decides `--tiles`.** The tile counts below were measured on
  the two cards above (docs/performance.md, "Memory and `--tiles`"). A card
  with less memory may
  need more tiles. At CAS(10,10), an under-provisioned split has
  **corrupted device memory** (`an illegal memory access`) instead of
  failing cleanly, so re-measure rather than lower the count.

### Toolchain (bare host)

The devcontainer below has all of this installed. Outside it you need:

- **clang + libc++** (the image uses clang 20), with `clang-scan-deps` and
  `libc++.modules.json`. They are needed for C++23 named modules and
  `import std`; gcc is not supported.
- **CMake 4.2.x, exactly.** The `import std` gate UUID is pinned for each
  release (`cmake/nevpt2_toolchain.cmake`), and any other version stops at
  configure. You also need **Ninja**.
- **The CUDA backend:** CUDA toolkit **≥ 13.0** (configure refuses an older
  one; `--cublas` needs 13.0's fixed-point emulation). The presets expect
  it at `/usr/local/cuda`; otherwise pass `-DCUDAToolkit_ROOT=...`.
- **The HIP backend:** ROCm (the image installs 7.2.4). The binaries link
  only the HIP runtime, hipBLAS and hipSOLVER (`--pc`'s eigensolves), but
  WarpWraps' configure requires a wider set of ROCm libraries regardless:
  hipBLASLt, hipSPARSE, hipFFT, hipRAND, rocThrust/rocPRIM, hipRTC and
  amd_smi. A ROCm install with its math libraries has all of them. On CUDA,
  the matching requirement is CCCL ≥ 3.0, which ships with the 13.0 toolkit.
- **Network on the first configure** of any preset that has testing on. It
  fetches GoogleTest. The standalone `cmake -S src` build fetches nothing.
- **`uv`**, for the offline Python tools only (`uv sync`, then
  `generate_golden.py` and the rest). Python never runs in a build or a
  ctest entry; the local gate (`devtools/cpp-tier.sh`) runs its lint, the
  golden layout check and the oracle's own tests through it.

### Devcontainer (optional)

The containers are how the maintainer builds and tests, and you don't need
them. There are three variants under `.devcontainer/`, driven by
`devtools/devcontainer.sh` (`--cuda`, the default, `--hip`, or
`--combined`). They have been run on a Linux host. The host needs:

- **Docker with BuildKit, and Node.js.** The script runs the devcontainer CLI
  through `npx`, so you don't install it yourself.
- **Images built locally.** Nothing is pulled prebuilt. Run
  `docker/build.sh cuda`, `docker/build.sh hip` or
  `docker/build.sh combined` (the combined image is about 40 GB, with both
  SDKs). `docker/README.md` covers the image chain.
- **NVIDIA (`cuda`, `combined`):** an NVIDIA driver new enough for
  CUDA 13 (R580 or later) and the **NVIDIA Container Toolkit**. The
  container gets the card through `--gpus all`.
- **AMD (`hip`, `combined`):** the **amdgpu** kernel driver, so that
  `/dev/kfd` and `/dev/dri` exist on the host. Both are passed in as
  devices, together with the host's `render` and `video` group ids, which
  `devtools/devcontainer.sh` resolves from `ROCM_GROUPS` in
  `devtools/config.sh`; a wrong gid shows up as `hipErrorNoDevice`.
  `combined` needs **both** vendors' cards, and on a host without
  `/dev/kfd` it doesn't start at all.
- **Claude Code is not installed unless you ask.** The maintainer works with
  it in the container (`INSTALL_CLAUDE_CODE=1 docker/build.sh cuda`); a plain
  build leaves it and Node out. The devcontainers still bind-mount the host's
  `~/.claude` (created empty if missing) and read `GH_TOKEN` for that tooling.
  Neither is needed to build or test.

```bash
docker/build.sh cuda && devtools/devcontainer.sh up && devtools/devcontainer.sh shell
devtools/devcontainer.sh --hip up        # the ROCm variant; pass the flag on every call, `down` included
```

## Running the demos directly

`ctest` is the easy way to run them, and each registered entry already
carries the right flags. Calling a binary yourself does the same work:
read a reference file, build the RDMs, compute the class energies, check them
against the reference to 1e-7 and print `PASS:`.

**Where the binaries are:** `build/src/` after `cmake --preset default`,
`build-hip/src/` after `cmake --preset hip`, and `src/build/` after the
standalone `cmake -S src -B src/build`.

| binary | reads | what it is |
|---|---|---|
| `nevpt2_demo` | any reference file, normally a conventional one | integral-direct NEVPT2, with every integral block read from the file. A `_df` file also carries those blocks, so it passes there too (`nevpt2_cas1010_df_blocks` checks that) |
| `nevpt2_df_demo` | a density-fitted file (`golden/*_df.nevpt2gold`) | the external integrals built on the device from three-index `B` tensors. It checks against PySCF's **DF**-NEVPT2, which is a different reference |

```bash
B=build/src    # or build-hip/src, or src/build

$B/nevpt2_demo    --golden golden/n2_ccpvdz_cas1010.nevpt2gold    --tiles 3          # the default case
$B/nevpt2_demo    --golden golden/n2_ccpvdz_cas1010.nevpt2gold    --tiles 3 --pc     # + PC-NEVPT2, checked against block2
$B/nevpt2_demo    --golden golden/n2_ccpvdz_cas1212.nevpt2gold    --tiles 40         # the large case (slow)
$B/nevpt2_demo    --golden golden/n2_ccpvdz_cas1012.nevpt2gold    --tiles 40 --pc    # the 12-orbital PC case (slow)
$B/nevpt2_df_demo --golden golden/n2_ccpvdz_cas1010_df.nevpt2gold --tiles 3 --check-blocks
$B/nevpt2_df_demo --golden golden/n2_ccpvtz_cas1010_df.nevpt2gold --tiles 3 --batch 5
```

**`--tiles`, per case** (a memory lever, not a speed one; these are the
counts the ctest entries use, and the measured floors behind them are in
docs/performance.md, "Tile floors"):

| reference file | `--tiles` |
|---|---|
| CAS(4,4), CAS(8,8), CAS(10,10), conventional or `_df`, cc-pVDZ or cc-pVTZ; LiF CAS(4,6) and acrolein CAS(6,5) | `3` |
| CAS(12,12), CAS(10,12) | `40` |
| CAS(12,12) with `--cublas` | `80` or `150` (at fewer tiles emulation can decline; see below) |

The gitignored reference files (CAS(14,14) and the salicylaldimine CAS(8,8)
pair) have to be regenerated with `reference_data/generate_golden.py` before
either demo can read them.

**Flags.** Both binaries take the shared flags below (`src/cli/`), plus
`--golden <file>`, which is required; the rows marked *demo only* are one
binary's:

| flag | does |
|---|---|
| `--tiles N` | split the RDM build into N tiles (default 3) |
| `--pc` | also compute PC-NEVPT2 and check it against block2's answer. Needs a file with PC fields (not the CAS(12,12) ones) |
| `--profile` | print per-phase timing tables (RDM build, energy, and DF integrals in the DF demo) |
| `--cublas` | **CUDA only**: the fp64-emulated cuBLAS digest. Read the `engaged bits=` line. `DECLINED -> ran native fp64!` means it did not engage, and the timing is meaningless |
| `--mantissa-bits N` | the maximum mantissa bits for `--cublas`'s emulation, 1..53 (default 53); refused without `--cublas`, so CUDA only |
| `--blas-digest` / `--digest-emitted` | choose the digest. The default differs by backend: emitted on CUDA, BLAS on HIP |
| `--ozaki` | the int8 tensor-core (Ozaki-scheme) digest, on **both** backends. Accurate to fp64 rounding, slower than the default digest on both cards; not usable with `--cublas` |
| `--ozaki-pairs P` | keep only digit pairs with p + q <= P (0..14; default 14, all 64 pairs). P <= 5 makes cc-pVTZ's PC metric refuse; refused without `--ozaki` |
| `--ozaki-check` | also run each `--ozaki` GEMM natively and print the largest difference (a diagnostic; synchronizes per GEMM); refused without `--ozaki` |
| `--fused-digest` | run a GEMM digest's three GEMMs as one, over the stacked `[R; W_ca; W_ac]`. Needs `--blas-digest`, `--cublas` or `--ozaki`; off by default. Pays most with `--cublas` |
| `--consume-emitted` | use the emitted kernel for the consume step instead of the GEMM |
| `--rdm-tangent` | also run the tangent RDM build — the derivative of dm3/f3ac/f3ca with respect to the CI vector — and check it against twice the plain build. It holds about twice the per-tile intermediates, so it needs more tiles than the plain build; not usable with `--cublas`, `--ozaki` or `--fused-digest` |
| `--pool-threshold BYTES\|max` | the memory pool's release threshold (default `max`) |
| `--sc-tangent` | **both demos**: instead of the golden comparison, differentiate the SC total with respect to the CI vector and check that tangent against the class norms' exact affine increment, Euler's identity and a central finite difference of the energy itself. Same digest restriction as `--rdm-tangent`, no `--pc` path, and no `--profile` (it prints no profile tables) |
| `--gradient-source` | **both demos**: instead of the golden comparison, assemble the SC-NEVPT2 CI source term `y_I = dE/dc_I` (Park Eq. 48) in reverse mode and check it against the forward tangent by duality — `<u, y> = Edot(u)`, exactly, for several seeded directions and for the state itself, plus the full Jacobian column by column where the determinant space is small enough. With `--pc` (integral-direct demo only), the same for PC-NEVPT2's `y_I`: the PC solve's adjoint against its tangent (#74). On the DF demo the orbital energies are held fixed and the PASS line says so. Same digest restriction as `--rdm-tangent`, and no `--profile`; not with `--sc-tangent`, nor (DF demo) with `--check-blocks` |
| `--sc-hessian` | **integral-direct demo only** (#153): instead of the golden comparison, take the SC total's *second* directional derivative `Eddot(u, v)` with respect to the CI vector, at the golden's fixed orbital energies, and check it against Euler's identity at second order (`Eddot(c, u) = Edot(u)` on the three homogeneous classes), the class norms' exact second-order increment and one central finite difference of `Edot`. Its three-argument tangent RDM build needs more tiles than `--rdm-tangent`'s. Same digest restriction as `--rdm-tangent`, no `--pc` path, no `--profile`, and not with `--sc-tangent` or `--gradient-source` |
| `--batch N` | **DF demo only**: slab length along each integral block's batch index (default 8; `0` = one slab per block; >= 0). A memory lever |
| `--check-blocks` | **DF demo only**: also compare every DF-built integral block against the file's four-index arrays |

An argument no binary takes, or a flag missing its value, is an *invalid
configuration* error carrying the binary's usage line.

**Which binary takes which flag.** One table in `src/cli/` (the
`:binary_flags` partition) says which of the four binaries implements each
flag, and a binary given a flag it does not implement refuses it as
*unsupported*, naming the binaries that do — it never parses a flag and then
ignores it. `test/cli/` holds this matrix against the table:

| flag | `nevpt2_demo` | `nevpt2_df_demo` | `nevpt2_sc_pseudodensity` | `nevpt2_pc_tangent` |
|---|:-:|:-:|:-:|:-:|
| `--golden`, `--tiles`, `--pool-threshold` | ✓ | ✓ | ✓ | ✓ |
| `--blas-digest`, `--digest-emitted`, `--consume-emitted` | ✓ | ✓ | ✓ | ✓ |
| `--cublas`, `--mantissa-bits`, `--ozaki`, `--ozaki-pairs`, `--ozaki-check`, `--fused-digest` | ✓ | ✓ | ✓ | refused |
| `--pc` | ✓ | ✓ | ✓ (`--dump-source-terms`, `--source-oracle`) | refused |
| `--profile` | ✓ | ✓ | refused | refused |
| `--rdm-tangent` | ✓ | ✓ | refused | refused |
| `--sc-tangent`, `--gradient-source` | ✓ | ✓ | refused | refused |
| `--sc-hessian`, `--sc-hessian-vector` | ✓ | refused | refused | refused |
| `--batch`, `--check-blocks` | refused | ✓ | refused | refused |
| `--kappa-act`, `--kappa-full`, `--sc-hessian-kc` | refused | refused | ✓ | refused |
| `--gradgold` | ✓ | refused | ✓ | refused |
| `--dump-source-terms`, `--source-oracle` | refused | refused | ✓ | refused |
| `--fd-steps`, `--fd-start-pow` | refused | refused | refused | ✓ |
| `--pc-hessian` | refused | refused | refused | ✓ |

There is no `--df` row, on purpose (#165): a density-fitted case is a
`*_df` golden, and the orbital-source-term flags (`--kappa-act`,
`--kappa-full`) take one in `nevpt2_sc_pseudodensity` with no flag of its
own — `--kappa-full` knows it by its sidecar's density-fitted magic and runs
both of `Y`'s terms there, the digests' own included (#168), and
`--dump-source-terms` writes the fitted `Y` and `y`. No binary parses `--df`,
and `nevpt2_df_demo` keeps refusing the orbital-source-term flags. Two
combinations are refused at run time on a fitted sidecar: `--source-oracle`
(no fitted oracle exists) and `--pc --dump-source-terms` (no PC gate over the
fitted inventory) ([`docs/gradients.md`](docs/gradients.md), §6.2, has the
decisions and their reasons).

Why the refusals: `nevpt2_pc_tangent` is a tangent path throughout, and the
tangent RDM build is the emitted or the native-fp64 BLAS digest only; it is
also PC throughout, so `--pc` would change nothing. `nevpt2_sc_pseudodensity`
takes `--pc` for its two source-term jobs alone — they write and compare
PC-NEVPT2's source terms with it (#74) — and refuses it with the identities,
`--kappa-act` and `--kappa-full`, which are the SC energy's. Neither gradient binary prints a profile table or
runs the `--rdm-tangent` check. `--dump-source-terms` and `--source-oracle`
are `nevpt2_sc_pseudodensity`'s alone because `Y_xy` needs the gradient
sidecar's full-MO-range arrays; `nevpt2_demo` computes `y_I` too, but half a
dump is refused rather than written. `nevpt2_demo` reads the sidecar for one
thing, `--gradient-source`'s orbital-energy response. Beyond the matrix, a combination one binary
cannot run is an *invalid configuration* error (`--sc-tangent` with `--ozaki`,
`--kappa-full` or `--dump-source-terms` with `--cublas`, `--ozaki` or
`--fused-digest` — both build the digests' tangent, which an approximate
digest has no path for,
`--kappa-act` with `--kappa-full`, `--dump-source-terms` with either,
`--source-oracle` with any of the three, `--sc-hessian-kc` with any of the
four or with `--pc`, `--cublas`, `--ozaki` or `--fused-digest` (#184),
`--gradgold` without `--kappa-full`, `--sc-hessian-kc`, `--dump-source-terms`
or `--source-oracle`, or in
`nevpt2_demo` without `--gradient-source`, `--check-blocks` with
`--sc-tangent` or `--gradient-source`, `--pc` in
`nevpt2_sc_pseudodensity` without `--dump-source-terms` or `--source-oracle`,
`--sc-tangent`, `--sc-hessian` or `--sc-hessian-vector` with `--pc`,
`--sc-hessian` with `--sc-tangent` or `--gradient-source`,
`--sc-hessian-vector` with any of those three), and
`--sc-tangent` / `--gradient-source` / `--sc-hessian` / `--sc-hessian-vector`
with `--profile` is
refused as *unsupported*, since none prints the tables, as is `nevpt2_df_demo
--gradient-source --pc` (the PC source term has no density-fitted slab path).

### The gradient checks

Two further binaries, plus three of the flags above, check the
**gradient** work: derivatives of the NEVPT2 energy with respect to the CI
vector and to an orbital rotation, each computed two independent ways and held
against the other. They read the same reference files, and each prints its own
`PASS:` line. What they compute, and what the gradient path deliberately does
*not* do, is [`docs/gradients.md`](docs/gradients.md).

| invocation | extra flags | what it checks |
|---|---|---|
| `nevpt2_demo --rdm-tangent` (also `nevpt2_df_demo`) | — | the tangent RDM build beside the plain one, against twice it |
| `nevpt2_demo --sc-tangent` (also `nevpt2_df_demo`) | — | the SC total's derivative with respect to the CI vector |
| `nevpt2_demo --sc-hessian` | — | the SC total's second directional derivative `Eddot(u, v)`, with the orbital energies held fixed (#153) |
| `nevpt2_demo --sc-hessian-vector` | — | the vector `H_cc u` (the CI block of the SC total's Hessian, by reverse over forward) against `Eddot` by duality, with the orbital energies held fixed (#156) |
| `nevpt2_demo --gradient-source` | `[--gradgold PATH]`, `[--pc]` | the CI source term `y_I = dE/dc_I`, against that tangent by duality; with the `*.gradgold` sidecar, both carry the orbital energies' response to the active density, without it both hold them fixed. With `--pc`, PC-NEVPT2's `y_I` (the PC solve's adjoint) against the PC tangent (#74) |
| `nevpt2_df_demo --gradient-source` | — | the SC CI source term against the tangent by the same duality, over the density-fitted slabs, with the orbital energies held fixed (#162) |
| `nevpt2_sc_pseudodensity` | `--kappa-act`, or `--kappa-full [--gradgold PATH]` | the SC pseudodensity identities; with `--kappa-act`, the active–active orbital source term instead; with `--kappa-full`, the whole orbital source term out of the `*.gradgold` sidecar — on a `*_df` golden, out of the whole three-index `B`, with both of its terms (#165, #168) |
| `nevpt2_sc_pseudodensity --sc-hessian-kc` | `[--gradgold PATH]` | the mixed orbital-CI block `H_kc` of the SC total's Hessian, as the scalar `Eddot_kc(w, u)` and the vector `H_kc^T w` against each other by duality, with Euler in `c` and a finite difference of `y` along `kappa`; out of the `*.gradgold` sidecar when there is one, the orbital energies responding, and with none along an active–active `w` alone, the orbital energies frozen (#184) |
| `nevpt2_sc_pseudodensity --source-oracle PATH` | `[--gradgold PATH]` | the GPU's own `Y_xy` and `y_I` against PySCF's exact ones, committed as a `golden/*.srcgold`, block by block: every block must stay within the measured tolerance (the known-gap list is empty since #89) ([`docs/testing.md`](docs/testing.md), "The SC source terms against PySCF's, on a card"). With `--pc`, PC-NEVPT2's against block2's, `golden/n2_ccpvdz_cas44_pc.srcgold` (#74; docs/testing.md, "The PC source terms' oracle"); the oracle's scheme must be the run's. A directional oracle (`golden/*_dir.srcgold`, #105) is compared along its seeded directions per block instead (docs/testing.md, "The directional source-term oracle") |
| `nevpt2_sc_pseudodensity --dump-source-terms PATH` | `[--gradgold PATH]`, `[--pc]` | writes the GPU's own `Y_xy` and `y_I` to a `.gradsrc` (a NumPy `.npz`) for `reference_data/nevpt2_grad.py --source dump` / `both`, and reads it back; what a nuclear gradient built from it comes out at is [`docs/testing.md`](docs/testing.md), "The GPU as the source-term provider". With `--pc`, PC-NEVPT2's, marked as such, with the PC adjoint held to the PC forward tangent along seeded rotations (#74; [`docs/pc-nevpt2.md`](docs/pc-nevpt2.md), "The source terms by reverse mode") |
| `nevpt2_pc_tangent` | `--fd-steps N` (3..8, default 4), `--fd-start-pow N` (1..30, default 12: the first step is `2^-N`) | PC's derivative with respect to the CI vector, against a Richardson-verified finite difference |
| `nevpt2_pc_tangent --pc-hessian` | the same two | PC's *second* directional derivative `Eddot(u, v)`, with the orbital energies held fixed and the rotation terms of S's turning null space ([`docs/hessian-theory.md`](docs/hessian-theory.md), §6.2), against Euler at second order, a symmetry-breaking negative control that must be refused, and a Richardson-verified difference of the three-term `Edot` (#152) |

Everything that builds a *tangent* — the first four rows and
`nevpt2_pc_tangent` — runs the emitted or the native-fp64 BLAS digest;
`--cublas`, `--ozaki` and `--fused-digest` are refused there, since a tangent
of an approximate digest is not the derivative of what the energy stage
computed. `nevpt2_sc_pseudodensity` builds no tangent RDMs and takes any
digest, except with `--kappa-full`, `--dump-source-terms` and
`--sc-hessian-kc`, which build the digests' tangent (and, the last, the
tangent RDM build) and refuse them too. `--sc-tangent` and `--gradient-source` run on both demos and refuse
each other; `--gradient-source` has a `--pc` path (#74) on the integral-direct
demo only, `--sc-tangent` none (the PC tangent's own check is
`nevpt2_pc_tangent`).

## One tree, two backends

`src/` builds for CUDA or HIP — one backend per build, picked by
`NEVPT2_GPU_BACKEND` (CUDA by default; `cmake --preset hip` or
`-DNEVPT2_GPU_BACKEND=HIP` for ROCm).

- **Host code** is C++23 named modules (clang + libc++, `import std;`). It
  reaches the GPU only through WarpWraps' backend-neutral names (`wwrMalloc`,
  `wwrStream_t`, `wwrblasDgemm`, …), which bind to `cuda*` or `hip*` at
  configure time.
- **Kernels** are C++20 `.cu` files that `#include <runtime.h>` from
  WarpWraps, compiled by `nvcc` or by clang as `-x hip`. Each is a static
  device library linked into the binary; nothing is loaded at run time.
- **`--cublas` is CUDA-only.** It uses CUDA 13.0's fp64 fixed-point
  emulation (`CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`), which has no ROCm
  equivalent. A HIP binary rejects the flag.
- **The default digest differs by backend:** the emitted kernel on CUDA and
  native-fp64 BLAS on HIP, each the faster one on its card
  (docs/performance.md, "Native-fp64 BLAS digest").

How the layering works is [`docs/architecture.md`](docs/architecture.md),
§1–§3.

## Repository layout

| path | what |
|---|---|
| `src/` | the library, one directory per component, both backends (per-file inventory: [`docs/architecture.md`](docs/architecture.md), §4) |
| `apps/` | the binaries: `integral_direct/` → `nevpt2_demo`, `density_fit/` → `nevpt2_df_demo`, the two gradient checks `sc_pseudodensity/` and `pc_tangent/`, and `sanitizer_canary/` |
| `test/` | the GoogleTest unit tier: host logic, never an energy |
| `golden/` | the committed reference files, which are the oracle, and their MO, gradient and source-term-oracle sidecars ([`docs/reference-data.md`](docs/reference-data.md)) |
| `reference_data/` | the offline Python that generates them (PySCF, block2) and checks their format, the offline gradients (analytic and numerical), and the CPU side of the head-to-head ([`docs/reference-data.md`](docs/reference-data.md), "The files in `golden/` and `reference_data/`") |
| `cmake/` | the build's macro layer and the sanitizer tier ([`cmake/README.md`](cmake/README.md)) |
| `deps/WarpWraps` | git submodule: the backend-neutral GPU layer |
| `docs/` | how it is built, how a number gets computed, what it costs, and how it is checked ([`docs/README.md`](docs/README.md) is the index) |
| `devtools/`, `docker/`, `.devcontainer/` | the maintainer's containers and check scripts |

## Testing

There are two tiers. The **reference tier** is each demo checking itself
against a committed reference file and printing `PASS:`; every one of those
entries needs a card. The **unit tier** (`ctest --preset unit`) checks host
logic with GoogleTest. There is **no CI, deliberately**: a hosted runner has
no GPU, so it could prove the tree compiles but not that any energy is right.
Please don't add a `.github/workflows/`; say in your pull request what you
ran, on which card. The local gate, and what to run before claiming a
result, is in [`docs/testing.md`](docs/testing.md), "No CI, and the local
gate". The sanitizer presets
(ASan, UBSan, compute-sanitizer) and their limits are in
docs/testing.md, "Sanitizers"; host code coverage (`devtools/coverage.sh`)
and what it cannot see are in docs/testing.md, "Coverage".

## Documentation

| you want | read |
|---|---|
| how a number gets computed, stage by stage | [`docs/implementation.md`](docs/implementation.md) |
| how the tree is layered, built, and split across two vendors | [`docs/architecture.md`](docs/architecture.md) |
| what the gradient path computes, and what it deliberately does not | [`docs/gradients.md`](docs/gradients.md) |
| the gradient derivation for this reference: the Lagrangian, the Z-vector equations, the integral inventory | [`docs/gradient-theory.md`](docs/gradient-theory.md) |
| what it costs, which knobs matter, and what didn't help | [`docs/performance.md`](docs/performance.md) |
| PC-NEVPT2: the design, accuracy against block2, cost | [`docs/pc-nevpt2.md`](docs/pc-nevpt2.md) |
| where the reference files come from and why they can be trusted | [`docs/reference-data.md`](docs/reference-data.md) |
| the test tiers, the sanitizers, coverage, and what each can't see | [`docs/testing.md`](docs/testing.md) |
| the method papers and the pinned software versions | [`docs/references.md`](docs/references.md) |

`performance.md`, `pc-nevpt2.md`, `reference-data.md` and `testing.md` are
the only places with numbers in them. The other docs cite their sections by
name and quote no figures of their own, so a measurement never
has a second copy that can go stale.

## References

The method is NEVPT2 as introduced by Angeli, Cimiraglia, Evangelisti,
Leininger and Malrieu (J. Chem. Phys. **114**, 10252 (2001)), with the
strongly- and partially-contracted variants of Angeli, Cimiraglia and Malrieu
(J. Chem. Phys. **117**, 9138 (2002)). The full list, and the reference
software, is [`docs/references.md`](docs/references.md). To cite this code,
use [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" reads it).

## License

MIT — see [`LICENSE`](LICENSE). The `deps/WarpWraps` submodule is MIT too.
