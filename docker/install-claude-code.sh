#!/usr/bin/env bash
# Install Node and Claude Code AT AN EXPLICIT VERSION.
#
# Called by docker/Dockerfile.{cuda,hip,combined} as their last root layer --
# the second script, after install-rocm.sh, that more than one Dockerfile
# drives. Deliberately NOT in Dockerfile.base, and that is
# the whole point: base is the parent of every GPU image, so a bump there would
# invalidate the CUDA toolkit and the ~19GB ROCm install downstream of it. Pinned at the leaves, a bump re-runs this script and the
# two trivial layers after it, and nothing else.
#
# WHY THIS EXISTS AT ALL -- the drift it ends
# This replaces ghcr.io/anthropics/devcontainer-features/claude-code, which the
# three devcontainer.json files used to declare. That feature's install.sh runs
# a bare, UNPINNED `npm install -g @anthropic-ai/claude-code`, so the version
# baked into an image is "whatever the registry had when that layer was first
# built". Three things then freeze it there:
#
#   1. the feature's content never changes, so its layer is a permanent cache
#      hit -- a rebuild re-uses it and installs nothing;
#   2. `devtools/devcontainer.sh rebuild` passes --remove-existing-container,
#      which recreates the CONTAINER from the same IMAGE, bumps nothing, and
#      reports success;
#   3. the updater is pinned off -- correctly, since the npm prefix is /usr
#      and root-owned, so an update started in here can only half-happen --
#      which means claude cannot self-correct either. It is
#      DISABLE_AUTOUPDATER=1 in each .devcontainer/*/devcontainer.json.
#
# devcontainer-lock.json does not help: it pins the FEATURE's digest, not the
# version of the package the feature installs. Measured on the reference box
# before this landed: four images off one lockfile, three at claude 2.1.197 and
# one at 2.1.289 -- the version was a function of build date, and nothing
# reported it. The pin is a reviewable line in git instead (ARG
# CLAUDE_CODE_VERSION in each GPU Dockerfile), bumped by
# devtools/claude-version.sh and reported by devtools/doctor.sh.
set -euo pipefail

# OPT-IN. The images build and test this project without Claude Code, so an
# outside user building the container should not get it by default. The
# maintainer's workflow sets INSTALL_CLAUDE_CODE=1 (devtools/config.sh, or the
# environment of docker/build.sh).
if [ "${INSTALL_CLAUDE_CODE:-0}" != 1 ]; then
  echo "=== Claude Code not installed (INSTALL_CLAUDE_CODE=${INSTALL_CLAUDE_CODE:-0}; set 1 to opt in) ==="
  exit 0
fi

: "${CLAUDE_CODE_VERSION:?set by the ARG of the same name in the Dockerfile}"

# Node is here only because Claude Code is an npm package -- nothing else in
# this tree needs it (the `npx` in devtools/devcontainer.sh runs on the HOST).
# The feature installed Node 18, which is EOL; 22 is the current LTS.
NODE_MAJOR=${NODE_MAJOR:-22}

install -d -m 0755 /etc/apt/keyrings
# --batch --yes is load-bearing, not boilerplate: Dockerfile.combined runs this
# script ON TOP OF the `cuda` image, which already ran it, so the keyring is
# already there. A bare `gpg --dearmor -o` on an existing file asks whether to
# overwrite, opens /dev/tty to ask, and dies with `cannot open '/dev/tty'` in a
# build that has none. The cuda and hip images never see it -- they come off
# base, where the file is absent -- so the breakage lands on `combined` alone,
# and only once this layer is no longer cached.
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
  | gpg --batch --yes --dearmor -o /etc/apt/keyrings/nodesource.gpg
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] \
https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
  > /etc/apt/sources.list.d/nodesource.list
apt-get update
apt-get install -y --no-install-recommends nodejs
rm -rf /var/lib/apt/lists/*

# The nodesource deb sets prefix=/usr, so claude lands at /usr/bin/claude --
# exactly where the feature used to put it, which is what keeps the
# root-owned-prefix reasoning above, and doctor.sh's pin check, true of this
# image.
prefix=$(npm prefix -g)
if [ "$prefix" != /usr ]; then
  echo "install-claude-code: npm global prefix is '$prefix', expected /usr" >&2
  echo "  doctor.sh's pin check assumes /usr/bin/claude" >&2
  exit 1
fi

npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"

# The guard that matters. npm turns a missing or yanked version into an error
# by itself, but a TYPO'd dist-tag resolves to something else and installs it
# quietly -- and a silently-wrong Claude in every container is the exact
# failure this script exists to end. Fail the build instead.
installed=$(claude --version | cut -d' ' -f1)
if [ "$installed" != "$CLAUDE_CODE_VERSION" ]; then
  echo "install-claude-code: asked for ${CLAUDE_CODE_VERSION}, got ${installed}" >&2
  exit 1
fi

echo "=== Claude Code ${CLAUDE_CODE_VERSION} at $(command -v claude), node $(node --version) ==="
