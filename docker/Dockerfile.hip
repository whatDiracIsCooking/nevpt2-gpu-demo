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
ARG CLAUDE_CODE_VERSION=2.1.292
ENV CLAUDE_CODE_VERSION=${CLAUDE_CODE_VERSION}
# OPT-IN: INSTALL_CLAUDE_CODE=0, the default, skips the install -- Node with
# it, since nothing else in the image needs Node -- so building the container
# does not install an agent you did not ask for. docker/build.sh forwards it
# from devtools/config.sh; `INSTALL_CLAUDE_CODE=1 docker/build.sh cuda` turns
# it on. ENV too, so doctor.sh can tell "opted out" from "missing".
ARG INSTALL_CLAUDE_CODE=0
ENV INSTALL_CLAUDE_CODE=${INSTALL_CLAUDE_CODE}
RUN --mount=type=bind,source=docker/install-claude-code.sh,target=/tmp/install-claude-code.sh \
    bash /tmp/install-claude-code.sh

USER ubuntu
RUN hipconfig --version
