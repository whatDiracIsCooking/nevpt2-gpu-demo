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
  is too large to commit and has to be regenerated. Anything else is
  untested.
- **The CI vector is read in, and so are `dm1`/`dm2`.** The GPU builds the
  3-RDM and the 4-RDM intermediates. A real CASSCF code produces the 1- and
  2-RDMs cheaply, and building them here as dense string-algebra operators
  would not fit in memory past tiny active spaces.
- **One active space and one CI vector per run.** There is no persistent
  device session for a geometry scan or a trajectory, so each run pays its
  whole setup again.
- **The reference is CASCI, not a fresh CASSCF,** to keep it reproducible to
  the last bit (docs/reference-data.md, "What "CASSCF input" means here, and why the golden state is CASCI").
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
  `generate_golden.py` and the rest). Python never runs in a build or a test.

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
| CAS(4,4), CAS(8,8), CAS(10,10), conventional or `_df`, cc-pVDZ or cc-pVTZ | `3` |
| CAS(12,12), CAS(10,12) | `40` |
| CAS(12,12) with `--cublas` | `80` or `150` (at fewer tiles emulation can decline; see below) |

The gitignored reference files (CAS(14,14) and the salicylaldimine CAS(8,8)
pair) have to be regenerated with `reference_data/generate_golden.py` before
either demo can read them.

**Flags.** Both binaries take all of these, plus `--golden <file>`, which is
required:

| flag | does |
|---|---|
| `--tiles N` | split the RDM build into N tiles (default 3) |
| `--pc` | also compute PC-NEVPT2 and check it against block2's answer. Needs a file with PC fields (not the CAS(12,12) ones) |
| `--profile` | print per-phase timing tables (RDM build, energy, and DF integrals in the DF demo) |
| `--cublas` | **CUDA only**: the fp64-emulated cuBLAS digest. Read the `engaged bits=` line. `DECLINED -> ran native fp64!` means it did not engage, and the timing is meaningless |
| `--mantissa-bits N` | the maximum mantissa bits for `--cublas`'s emulation, 1..53 (default 53); refused without `--cublas`, so CUDA only |
| `--blas-digest` / `--digest-emitted` | choose the digest. The default differs by backend: emitted on CUDA, BLAS on HIP |
| `--ozaki` | the int8 tensor-core (Ozaki-scheme) digest, on **both** backends. Accurate to fp64 rounding, slower than the default digest on both cards; not usable with `--cublas` |
| `--ozaki-pairs P` | keep only digit pairs with p + q <= P (0..14; default 14, all 64 pairs). P <= 5 makes cc-pVTZ's PC metric refuse |
| `--ozaki-check` | also run each `--ozaki` GEMM natively and print the largest difference (a diagnostic; synchronizes per GEMM) |
| `--consume-emitted` | use the emitted kernel for the consume step instead of the GEMM |
| `--pool-threshold BYTES\|max` | the memory pool's release threshold (default `max`) |
| `--batch N` | **DF demo only**: slab length along each integral block's batch index (default 8; `0` = one slab per block). A memory lever |
| `--check-blocks` | **DF demo only**: also compare every DF-built integral block against the file's four-index arrays |

An unrecognized argument prints the binary's usage line.

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
| `apps/` | the binaries: `integral_direct/` → `nevpt2_demo`, `density_fit/` → `nevpt2_df_demo`, `sanitizer_canary/` |
| `test/` | the GoogleTest unit tier: host logic, never an energy |
| `golden/` | the committed reference files, which are the oracle ([`docs/reference-data.md`](docs/reference-data.md)) |
| `reference_data/` | the offline Python that generates them (PySCF, block2), and the CPU side of the head-to-head |
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
docs/testing.md, "Sanitizers".

## Documentation

| you want | read |
|---|---|
| how a number gets computed, stage by stage | [`docs/implementation.md`](docs/implementation.md) |
| how the tree is layered, built, and split across two vendors | [`docs/architecture.md`](docs/architecture.md) |
| what it costs, which knobs matter, and what didn't help | [`docs/performance.md`](docs/performance.md) |
| PC-NEVPT2: the design, accuracy against block2, cost | [`docs/pc-nevpt2.md`](docs/pc-nevpt2.md) |
| where the reference files come from and why they can be trusted | [`docs/reference-data.md`](docs/reference-data.md) |
| the test tiers, the sanitizers, and what each can't see | [`docs/testing.md`](docs/testing.md) |
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
