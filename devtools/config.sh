#!/usr/bin/env bash
# Per-project settings for everything in devtools/. EDIT THIS FILE, not the
# scripts -- the scripts are meant to survive being copied into the next repo
# unchanged, and this is the one place that knows what the project is called
# and how its tests run.
#
# Sourced (never executed) by devcontainer.sh, worktree.sh, doctor.sh,
# cpp-tier.sh and cross-backend-check.sh. Every value uses ${VAR:-default} so
# an environment variable still wins for a one-off:
#
#   BUILD_JOBS=4 devtools/cpp-tier.sh
#
# The one thing this file CANNOT reach is the .devcontainer/<variant>/devcontainer.json
# files -- JSON cannot source shell. The volume names, the image tag and the
# .git bind path are spelled out there too; PROJECT_NAME below must match the
# `<name>-pytest-tmp-` style prefixes used there, or `worktree.sh rm` and
# `worktree.sh gc` will not recognise this project's volumes as its own.
# doctor.sh checks that the two agree and warns when they have drifted.
#
# The C++ side has a second config file with the same job: CMakePresets.json
# holds the build settings (backend, build type, GPU architecture, build
# directory). What lives HERE is only which preset the scripts should drive --
# see CMAKE_PRESET / CTEST_PRESET below.
#
# What this project does NOT have, so the scripts carry no knob for it:
# coverage (COVERAGE_*), an install tier, and a pytest pre-push gate (there
# is no Python test suite here -- the .py files are offline tools). If you
# find a script referencing one, it is drift.

# --- identity -------------------------------------------------------------

# Prefix for this project's docker volumes and per-worktree build images, and
# the name of the devcontainer. Must match devcontainer.json (see above).
# It ends up in docker resource names, so keep it lowercase.
PROJECT_NAME=${PROJECT_NAME:-nevpt2}

# Which devcontainer.json devcontainer.sh drives. Relative paths resolve
# against the repo root, so this works from any cwd and from any worktree.
# There is one per GPU file of docker/ that this project actually uses:
#
#   .devcontainer/cuda/devcontainer.json  THE development container: clang-20,
#                                         CMake 4.2, CUDA 13. Needs an NVIDIA
#                                         GPU and nvidia-container-toolkit.
#   .devcontainer/hip/devcontainer.json   the same toolchain with ROCm and no
#                                         CUDA at all. Needs an AMD card and
#                                         the amdgpu kernel driver. It pins
#                                         CMAKE_PRESET/CTEST_PRESET to `hip` in
#                                         containerEnv, so cpp-tier.sh needs no
#                                         flag in there.
#   .devcontainer/combined/devcontainer.json
#                                         BOTH SDKs and BOTH cards, ~40GB. The
#                                         only variant where an NVIDIA and an
#                                         AMD GPU are reachable at once: the
#                                         cuda one passes `--gpus all` and no
#                                         AMD device nodes, the hip one passes
#                                         /dev/kfd + /dev/dri and no `--gpus`,
#                                         and that one carries both sets. A
#                                         build still targets exactly ONE
#                                         backend -- what this buys is that
#                                         switching costs an env var rather
#                                         than a container, which is worth
#                                         more now that src/ is ONE tree
#                                         built either way: `CMAKE_PRESET=hip
#                                         CTEST_PRESET=hip devtools/cpp-tier.sh`
#                                         in there runs the ROCm tier against
#                                         the AMD card in the same shell that
#                                         just ran the CUDA one, off the same
#                                         sources. Needs `docker/build.sh
#                                         combined`, and both vendors' cards:
#                                         docker refuses --device=/dev/kfd on a
#                                         host without one, so it will not
#                                         start on an NVIDIA-only box.
#
# This is only the FALLBACK. For one call use the variant flag --
# `devtools/devcontainer.sh --hip up` -- and for a whole shell session, export
# the variable. Precedence is flag, then variable, then this.
DEVCONTAINER_CONFIG=${DEVCONTAINER_CONFIG:-.devcontainer/cuda/devcontainer.json}

# --- C++ build (CMake) ------------------------------------------------------

# Which CMakePresets.json presets devtools/cpp-tier.sh drives. The presets
# themselves -- backend, build type, GPU arch, build directory -- live in
# CMakePresets.json; only the CHOICE lives here.
#
#   default      CUDA (src/), sm_86, Release, into build/
#   workstation  the SAME configuration into build-workstation/, so a host
#                build and a container build can coexist without reconfiguring
#                each other (a CMake cache records absolute paths, and the two
#                mount the workspace at different ones)
#   debug        CUDA, -O0 -g, into build-debug/
#   hip          the ROCm backend (src/), gfx1200, into build-hip/
#   compile-cuda / compile-hip
#                compile-and-link only, NEVPT2_BUILD_TESTING=OFF: checks the
#                tree builds, on a box with or without a card. (There is no CI;
#                docs/testing.md, "No CI, and the local gate".)
#                cross-backend-check.sh drives compile-hip.
#
# TEST PRESETS, and the honest reading of a green run:
#
#   default   every registered test -- CAS(10,10) ~2s AND CAS(12,12) ~47s
#   fast      everything not labelled `slow`
#   unit      the GoogleTest unit tier alone (test/, label `unit`)
#   hip       the ROCm runtime tier
#
# EVERY GOLDEN ENTRY IS LABELLED `gpu`. Only the GoogleTest unit tier's
# host-only suites are not, so `ctest -LE gpu` selects just them:
# a green run of it proves host logic, never an energy. A numerical claim
# needs a real card, always.
#
# Override for one run rather than editing:
#   CMAKE_PRESET=hip CTEST_PRESET=hip devtools/cpp-tier.sh
#
# Set CTEST_PRESET empty to configure and build without running ctest.
CMAKE_PRESET=${CMAKE_PRESET:-default}
CTEST_PRESET=${CTEST_PRESET:-fast}

# --- cross-backend check --------------------------------------------------

# devtools/cross-backend-check.sh compiles the tree for the backend this build
# is NOT targeting. Here that is more than a nicety: src/ is ONE tree built
# for both backends through WarpWraps' wwr* names, so a CUDA-only spelling (a
# raw cuda* call, an nvcc-only construct in a kernel) slipped in outside
# src/cublas/cublas_emul.cpp (CUDA-only by design) breaks only the HIP
# build, and nothing else catches it.
#
# A build targets exactly one backend and CMAKE_PRESET above picks it, so these
# describe the OTHER one. The preset is the one knob: the backend (the name the
# script prints), the toolchain probe and the image all follow from it, so
# `CROSS_CHECK_PRESET=compile-cuda` checks -- and says it checks -- CUDA. Every
# configure preset with "hip" in its name inherits `hip` and every other one
# `default` (CUDA) in CMakePresets.json, hence the match. Each can still be set
# on its own. Flip the preset default if the default build ever becomes HIP.
CROSS_CHECK_PRESET=${CROSS_CHECK_PRESET:-compile-hip}
case $CROSS_CHECK_PRESET in
  *hip*) _cross_backend=HIP  _cross_tool=hipcc ;;
  *)     _cross_backend=CUDA _cross_tool=nvcc ;;
esac
CROSS_CHECK_BACKEND=${CROSS_CHECK_BACKEND:-$_cross_backend}
CROSS_CHECK_TOOL=${CROSS_CHECK_TOOL:-$_cross_tool}
unset _cross_backend _cross_tool

# The image carrying that toolchain, used when this machine lacks the tool.
# `docker/build.sh hip` (or `cuda`) builds and tags it -- that script derives the
# tag from PROJECT_NAME exactly as the line below does, so the two agree by
# construction.
CROSS_CHECK_IMAGE=${CROSS_CHECK_IMAGE:-${PROJECT_NAME}:${CROSS_CHECK_BACKEND,,}}

# Its own build directory, separate from every preset's, because two presets
# sharing one directory silently reconfigure it back and forth -- a full
# rebuild each way. One per backend for the same reason (and a cache that
# switches compiler does not reconfigure cleanly). Matches the `build-*/` line
# in .gitignore.
CROSS_CHECK_BUILD_DIR=${CROSS_CHECK_BUILD_DIR:-build-cross-check-${CROSS_CHECK_BACKEND,,}}

# What --device-only builds, one target per line: the nevpt2_device_libraries
# umbrella, which every add_device_library() target joins (cmake/
# add_device_library.cmake) -- the half nvcc-vs-clang strictness bites.
# An umbrella rather than a hand-kept list of the *.device targets, because the
# hand-kept list drifted the first time a kernel library was added (f3_digest
# was missing from it). Empty makes --device-only refuse rather than
# build nothing and report PASS.
CROSS_CHECK_DEVICE_TARGETS=${CROSS_CHECK_DEVICE_TARGETS-nevpt2_device_libraries}

# --- the ROCm runtime tier -------------------------------------------------
#
# cross-backend-check.sh above proves the other backend COMPILES. On a machine
# with a real AMD card, `cpp-tier.sh --rocm` is the stronger statement: it runs
# that backend's ctest against the hardware, by re-execing cpp-tier.sh inside
# ROCM_IMAGE with the device nodes and group ids below.

ROCM_IMAGE=${ROCM_IMAGE:-${PROJECT_NAME}:hip}
ROCM_PRESET=${ROCM_PRESET:-hip}

# The device nodes ROCm needs, one per line. /dev/kfd is the compute driver
# interface and /dev/dri carries the render nodes; both are required, and
# neither is passed into a container by default the way --gpus all handles
# NVIDIA.
ROCM_DEVICES=${ROCM_DEVICES:-"/dev/kfd
/dev/dri"}

# The HOST groups owning those nodes, one per line. Both container paths
# resolve these to NUMERIC gids with getent before passing them to
# --group-add, and that matters more than it looks: `--group-add render`
# resolves the name INSIDE the container, where it has a different id or does
# not exist, so the run then fails at the first device call with
# "hipErrorNoDevice (no ROCm-capable device is detected)" -- which reads
# exactly like a box with no GPU, and has been misread as one.
#
# On this box: render=109, video=44. Those numbers are also written into
# .devcontainer/hip/devcontainer.json AND .devcontainer/combined/devcontainer.json
# as ${localEnv:...} fallbacks, since JSON cannot loop; adding a name here
# reaches cpp-tier.sh automatically but still needs the reference written into
# BOTH of those json files by hand.
ROCM_GROUPS=${ROCM_GROUPS:-"render
video"}

# Parallel compile jobs for `cmake --build`. Empty lets Ninja pick (cores + 2).
# This tree is small -- a dozen translation units plus four nvcc device-library
# compiles -- so the default is fine here; a larger module-heavy tree would
# want it lower than the core count (see the devbox skill).
BUILD_JOBS=${BUILD_JOBS:-}

# --- tests ----------------------------------------------------------------

# THERE IS NO PYTHON TEST SUITE. generate_golden.py and pyscf_contract_time.py
# are offline tools, not tests, and nothing collects them. TEST_CMD is left
# pointing at pytest so a script that insists on one does something
# recognisable rather than failing obscurely, but the real suite is ctest --
# see CTEST_PRESET above.
TEST_CMD=${TEST_CMD:-pytest}
JOBS=${JOBS:-8}
FAST_TEST_ARGS=${FAST_TEST_ARGS:-}
PREPUSH_PATHS=${PREPUSH_PATHS:-}

# Host CPU bounds for a devcontainer, applied by devcontainer.sh on `up` and
# `rebuild`. Both empty = unbounded, and nothing is applied.
#
#   CPUSET  pin the container to specific host cores, e.g. 0-11
#   CPUS    cap it at n cores' worth of CPU quota, e.g. 8
#
# These matter MORE here than in most projects: docs/performance.md is full
# of timing measurements, and several of its numbers carry an explicit
# "the box was contended" caveat because they were taken without bounds. A
# head-to-head against PySCF on 24 CPU threads is worthless unmeasured against
# a pinned container.
CPUSET=${CPUSET:-}
CPUS=${CPUS:-}

# Where cpp-tier.sh writes its run logs. Relative paths are resolved against
# the repo root. Gitignored. `.gitignore` and `.dockerignore` both name this
# path, so keep them in step if you rename it.
#
# NB docker/compose.yaml writes its own logs to .log/ instead.
SLOW_TIER_REPORTS=${SLOW_TIER_REPORTS:-.slow-tier-reports}

# --- environment ----------------------------------------------------------

# Where to look for the project's virtualenv, in order. `.venv` is what
# `uv sync` makes on the host, `/opt/venv` is what the Dockerfile bakes in.
VENV_PATHS=${VENV_PATHS:-.venv /opt/venv}

# Optional tools doctor.sh reports on, one `name:what is lost without it` per
# line. A missing one is a WARN, never a FAIL.
#
# Most of this list is absent on the HOST and present in the container, which
# is the point: run doctor.sh on both sides and the warnings tell you which
# half of the workflow you are on.
#
# clang++ and clang-scan-deps are back on this list: the host tree is C++23
# named modules, and cmake/nevpt2_toolchain.cmake refuses to configure without
# both (plus libc++'s libc++.modules.json).
DOCTOR_OPTIONAL_TOOLS=${DOCTOR_OPTIONAL_TOOLS:-"docker:devcontainer, compose services and worktree containers
gh:the PR flow (also needs GH_TOKEN)
cmake:configuring and building the C++ tree at all
clang++:the host compiler -- C++23 named modules need clang + libc++
clang-scan-deps:module dependency scanning; configure refuses without it
ninja:the generator every preset uses
nvcc:compiling the device libraries and the whole CUDA backend (src/)
hipcc:the HIP backend (src/, NEVPT2_GPU_BACKEND=HIP); present only in the \`hip\`/\`combined\` image
sccache:warm rebuilds
clang-format:the C++ formatting pass -- run by hand, it is not a git hook
nsys:Nsight Systems profiling from inside the container
compute-sanitizer:the GPU memcheck/racecheck run
uv:the Python tier (the two offline PySCF tools)
ruff:the lint pass pre-commit runs"}

# Which GPU vendors doctor.sh probes for a LIVE DEVICE, space-separated.
# Both, because this box has both and a build targets exactly one -- and
# because every test in this project needs a real card.
DOCTOR_GPU_VENDORS=${DOCTOR_GPU_VENDORS:-"nvidia amd"}

# Paths that must exist for some slice of the suite to run, one
# `path:what skips without it` per line. A RELATIVE path is resolved against
# the repo root.
#
# These are the GOLDEN FILES -- this project's oracle. Without them the ctest
# entries are not registered at all (CMakeLists.txt warns and skips rather
# than failing), so their absence turns into a silent "0 tests ran" instead of
# a failure, which is exactly the thing worth warning about up front.
#
# The CAS(14,14) golden is deliberately NOT listed: it is gitignored (71 MB)
# and absent in every fresh clone by design.
DOCTOR_REQUIRED_PATHS=${DOCTOR_REQUIRED_PATHS:-"golden/n2_ccpvdz_cas1010.nevpt2gold:the default CAS(10,10) golden -- the nevpt2_cas1010 ctest entry is not registered without it
golden/n2_ccpvdz_cas1212.nevpt2gold:the CAS(12,12) scaling golden -- the nevpt2_cas1212 ctest entry is not registered without it
deps/WarpWraps/CMakeLists.txt:the WarpWraps submodule checkout (not yet used by the build)"}

# --- the agent in the images -----------------------------------------------

# Where the Claude Code version pin lives, one path per line, relative to the
# repo root: every file carrying an `ARG CLAUDE_CODE_VERSION=` line. These are
# the GPU Dockerfiles, each of which runs docker/install-claude-code.sh as its
# last layer -- NOT Dockerfile.base, because a bump there would rebuild the
# CUDA toolkit and the ~19GB ROCm install downstream of it.
#
# devtools/claude-version.sh reads, compares and rewrites exactly these, and
# doctor.sh warns when they disagree or have fallen behind the npm registry.
CLAUDE_PIN_FILES=${CLAUDE_PIN_FILES:-"docker/Dockerfile.cuda
docker/Dockerfile.hip
docker/Dockerfile.combined"}

# Whether docker/build.sh installs Claude Code into the GPU images at all.
# OFF by default: nothing in the build or the tests needs it, and someone
# building the container to reproduce a number should not get an agent
# installed. The maintainer's workflow (.claude/, the pin above) wants it on:
# export INSTALL_CLAUDE_CODE=1 before docker/build.sh. Changing it rebuilds
# only the last layers of each GPU image, as a pin bump does.
INSTALL_CLAUDE_CODE=${INSTALL_CLAUDE_CODE:-0}

CLAUDE_NPM_PACKAGE=${CLAUDE_NPM_PACKAGE:-@anthropic-ai/claude-code}
CLAUDE_REGISTRY_TIMEOUT=${CLAUDE_REGISTRY_TIMEOUT:-8}
