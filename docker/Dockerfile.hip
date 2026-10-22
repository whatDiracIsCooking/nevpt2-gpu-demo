# syntax=docker/dockerfile:1.7
# ============================================================================
# hip -- base + ROCm, with no CUDA toolkit at all
# ============================================================================
# A HIP build needs no CUDA, which is the point of this being a sibling of
# Dockerfile.cuda rather than a layer on it: ROCm alone is ~30GB, and neither
# backend should pay for the other.
#
#   docker/build.sh hip       # builds base first, then this; tags :hip
#
# By hand, and the parent must already be tagged (see Dockerfile.base, "WHY THE
# FOUR FILES CHAIN BY TAG"):
#
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.base -t nevpt2:base .
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.hip  -t nevpt2:hip .
#
# Both from the REPO ROOT: the trailing `.` is the context, and this file
# bind-mounts docker/install-rocm.sh from it.
#
# This is the image devtools/cross-backend-check.sh and `devtools/cpp-tier.sh
# --rocm` reach for (CROSS_CHECK_IMAGE / ROCM_IMAGE in devtools/config.sh), so
# it is worth having even on a machine with no AMD card -- the cross-backend
# check only compiles.
ARG PROJECT_NAME=nevpt2
ARG PARENT_IMAGE=${PROJECT_NAME}:base
FROM ${PARENT_IMAGE}

ARG ROCM_VERSION=7.2.4

# Drop the ~12.1GB of ROCm that only a RUNNING kernel needs -- the Tensile and
# rocFFT kernel objects, composable-kernel's static archives, and rocalution.
# rccl and hiptensor are NOT dropped anymore: wwr.hip.rccl and wwr.hip.hiptensor
# link them, so pruning either would fail find_package at configure. See
# install-rocm.sh for the measured list and why nothing on it is a link-time
# dependency. OFF here: this is the dev image and it runs kernels.
# Set it to 1 for a compile-only image (docker/README.md, "The -ci variants,
# and the registry").
ARG ROCM_PRUNE=0
USER root

# Re-stamped rather than merely inherited from the parent, so that the
# PROJECT_NAME the devcontainer passes to THIS build is consumed where it is
# declared. See Dockerfile.base for what the label is for.
ARG PROJECT_NAME
LABEL devcontainer.project="${PROJECT_NAME}"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=bind,source=docker/install-rocm.sh,target=/tmp/install-rocm.sh \
    ROCM_VERSION="${ROCM_VERSION}" ROCM_PRUNE="${ROCM_PRUNE}" \
    bash /tmp/install-rocm.sh

# The LLVM target for the AMD card this image is built for. Exported as
# GPU_TARGETS / AMDGPU_TARGETS so a HIP CMake build has a default to read.
# gfx1200 is RDNA4 (Radeon RX 9060 XT); override for another card:
#   --build-arg GPU_TARGETS=gfx1100
ARG GPU_TARGETS=gfx1200
ENV GPU_TARGETS=${GPU_TARGETS} \
    AMDGPU_TARGETS=${GPU_TARGETS} \
    ROCM_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:${PATH}

# OpenMolcas, OPT-IN and PINNED, for ONE offline script:
# reference_data/molcas_casscf_hessian.py (#223), the cross-check of the
# analytic CASSCF nuclear Hessian against OpenMolcas's McKinley. Nothing in the
# build, ctest or the golden tier needs it. docker/install-openmolcas.sh
# carries the rest: why from source (no pip, conda-forge or Ubuntu package),
# why LINALG=Internal, and how the driver's one Python dependency is pinned.
# Placed BEFORE the two agents' layers because it is the slow one (a Fortran
# build) and the rarely-bumped one: an agent bump then re-runs only the cheap
# layers after it.
#
# The pin is TWO lines in each GPU Dockerfile: the release tag, and the commit
# that tag must resolve to (the script refuses a moved tag). Bump both, in all
# three files; doctor.sh warns when the copies disagree. ENV too, so doctor.sh
# can compare a running container's install ($prefix/VERSION) with what its
# image pinned.
ARG OPENMOLCAS_VERSION=v26.06
ARG OPENMOLCAS_COMMIT=8355057f32d65706a35996b5ab07cac2962bb728
ENV OPENMOLCAS_VERSION=${OPENMOLCAS_VERSION} \
    OPENMOLCAS_COMMIT=${OPENMOLCAS_COMMIT}
# OPT-IN AT THIS LAYER, as the two agents below are: INSTALL_OPENMOLCAS=0, this
# ARG's default, skips it (and the gfortran it brings). Unlike theirs, the
# value docker/build.sh forwards from devtools/config.sh is 0 as well; turn
# it on for one build with `INSTALL_OPENMOLCAS=1 docker/build.sh <target>`.
ARG INSTALL_OPENMOLCAS=0
ENV INSTALL_OPENMOLCAS=${INSTALL_OPENMOLCAS}
RUN --mount=type=bind,source=docker/install-openmolcas.sh,target=/tmp/install-openmolcas.sh \
    bash /tmp/install-openmolcas.sh

# Claude Code, OPT-IN and PINNED -- the last root layer in this file on purpose: a version
# bump re-runs only this and the two lines after it, never the SDK above it.
# docker/install-claude-code.sh carries the rest (why not Dockerfile.base, and
# which drift this ended). It is the fourth version to bump in more than one
# place: the same ARG is declared by Dockerfile.cuda, .hip and .combined, so
# bump it with `devtools/claude-version.sh --apply`, which rewrites all three
# at once -- doctor.sh warns when they have drifted apart.
#
# ENV as well as ARG so the INTENDED version is readable from inside a running
# container (`printenv CLAUDE_CODE_VERSION`), not merely inferable from the
# binary: that is what lets doctor.sh tell "pinned and installed" apart from
# "someone npm-installed over the top".
ARG CLAUDE_CODE_VERSION=2.1.306
ENV CLAUDE_CODE_VERSION=${CLAUDE_CODE_VERSION}
# OPT-IN AT THIS LAYER: INSTALL_CLAUDE_CODE=0, this ARG's default, skips the
# install -- Node with it, since nothing else in the image needs Node -- so a
# plain `docker build` of this file does not install an agent you did not ask
# for. docker/build.sh is the other caller, and it ALWAYS forwards the value
# from devtools/config.sh, where this project sets it to 1: a build driven by
# build.sh carries the agent, a build driven by hand does not. Turn it off for
# one build with `INSTALL_CLAUDE_CODE=0 docker/build.sh cuda`. ENV too, so
# doctor.sh can tell "opted out" from "missing".
ARG INSTALL_CLAUDE_CODE=0
ENV INSTALL_CLAUDE_CODE=${INSTALL_CLAUDE_CODE}
RUN --mount=type=bind,source=docker/install-claude-code.sh,target=/tmp/install-claude-code.sh \
    bash /tmp/install-claude-code.sh

# The Antigravity CLI (`agy`), OPT-IN and PINNED -- the same leaf-layer
# treatment as Claude Code above, and for the same reason: a bump re-runs this
# layer and nothing before it. docker/install-agy.sh carries the rest -- why not
# upstream's install.sh, and why the pin is a RELEASE (a version plus an opaque
# build id) rather than a version: the manifest endpoint serves only the
# CURRENT release, so a version on its own is not addressable the way
# `claude-code@2.1.292` is, and an older one would be unfetchable.
#
# TWO lines to bump, not one: AGY_SHA512 belongs to the exact tarball
# AGY_RELEASE names, and install-agy.sh verifies it rather than recording it.
# This is the fifth version declared in more than one place -- the same two
# ARGs are in Dockerfile.cuda, .hip and .combined -- so bump them with
# `devtools/agy-version.sh --apply`, which rewrites all three at once;
# doctor.sh warns when they have drifted apart.
#
# AGY_SHA512 is deliberately NOT an ENV: it is the tarball's hash, consumed by
# the RUN below and of no use to anything in a running container, so it stays
# out of the image's environment. AGY_RELEASE is an ENV for the same reason
# CLAUDE_CODE_VERSION is -- it is what lets doctor.sh tell the image's own
# pinned copy from something installed over the top of it, which `agy
# --version` cannot: that prints 1.4.4 and never the build id.
#
# AGY_CLI_DISABLE_AUTO_UPDATE is the binary's own opt-out of the background
# self-update it otherwise runs during normal use -- the counterpart of
# DISABLE_AUTOUPDATER in .devcontainer/*/devcontainer.json. The install is
# root-owned and the container runs as uid 1000, so the attempt fails
# regardless; what this stops is it being attempted behind the image's back.
ARG AGY_RELEASE=1.4.4-4783783166214144
ARG AGY_SHA512=7ce149d350344c1046c7b59b6659bcc5d659f351bf3a07c6c085887d93427ceb89d927752e4ccd0844b53ffb91d33d6a49f72835ceb276c7cffdd13cfb8023b8
ENV AGY_RELEASE=${AGY_RELEASE} \
    AGY_CLI_DISABLE_AUTO_UPDATE=1
# OPT-IN AT THIS LAYER, exactly as INSTALL_CLAUDE_CODE above: INSTALL_AGY=0,
# this ARG's default, skips the install -- 213MiB of binary that nothing in
# the build or the tests needs -- so a plain `docker build` of this file does
# not install an agent you did not ask for. docker/build.sh ALWAYS forwards
# the value from devtools/config.sh, where this project sets it to 1; turn it
# off for one build with `INSTALL_AGY=0 docker/build.sh cuda`. ENV too, so
# doctor.sh can tell "opted out" from "missing".
ARG INSTALL_AGY=0
ENV INSTALL_AGY=${INSTALL_AGY}
RUN --mount=type=bind,source=docker/install-agy.sh,target=/tmp/install-agy.sh \
    AGY_SHA512="${AGY_SHA512}" bash /tmp/install-agy.sh

USER ubuntu
RUN hipconfig --version
