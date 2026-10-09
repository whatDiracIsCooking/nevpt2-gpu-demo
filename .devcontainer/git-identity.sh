#!/usr/bin/env bash
# Apply the host's git identity INSIDE the container.
#
# Runs from `postCreateCommand` in both .devcontainer/*/devcontainer.json, as
# the container user, with the workspace as its cwd. It is not a host tool and
# there is nothing to run on the host -- devtools/devcontainer.sh is what
# resolves the two values this reads, and exports them for the
# `${localEnv:...}` lookups in those files.
#
# WHY THIS EXISTS AT ALL. Three separate things have to be true before a
# `git commit` inside the container can work, and until this script only two of
# them were:
#
#   the repo is visible      the NEVPT2_GIT_DIR bind in devcontainer.json
#   a push authenticates     gh + /etc/gitconfig, from docker/Dockerfile.base
#   commits have an author   THIS -- and nothing supplied it, so the first
#                            commit died with "Author identity unknown" after
#                            the work was already staged
#
# MEASURED in nevpt2:cuda, which is why the fallbacks do not cover it: the
# image's `ubuntu` user has the GECOS field "Ubuntu", so git auto-detects a
# NAME and then fails on the email --
#
#   fatal: unable to auto-detect email address (got 'ubuntu@49011452b048.(none)')
#
# That is also why both values are passed rather than just the email: with only
# an email supplied, git commits happily and attributes the work to "Ubuntu".
#
# WHY --global, AND NEVER --local. The container shares the HOST's git dir
# through the NEVPT2_GIT_DIR bind, so `git config --local` in here writes your
# host repo's .git/config -- the same reason docker/Dockerfile.base puts its URL
# rewrite in /etc/gitconfig instead of in `origin`'s URL. --global is
# $HOME/.gitconfig, which is the container's own filesystem: it survives a stop
# and a start, a rebuild discards it, and postCreateCommand puts it back.
set -u

name=${NEVPT2_GIT_USER_NAME:-}
email=${NEVPT2_GIT_USER_EMAIL:-}

# An unset host value arrives here as an EMPTY STRING, not as an unset variable
# -- the same trap GH_TOKEN has (devtools/doctor.sh warns about that one). So
# this writes nothing at all rather than writing an empty ident, and says what
# it skipped. MEASURED, and the reason the values are carried in under these
# NEVPT2_* names instead of as GIT_AUTHOR_NAME/GIT_AUTHOR_EMAIL directly:
#
#   no identity anywhere   "Author identity unknown"  -- recoverable in here
#                          with one `git config --global` call
#   GIT_AUTHOR_NAME=""     "fatal: empty ident name (for <>) not allowed", and
#                          it beats a GOOD user.name in config, so setting one
#                          by hand could not fix it either
if [ -z "$name" ] || [ -z "$email" ]; then
  echo "git-identity: no host identity was passed in -- leaving git unset." >&2
  echo "git-identity: commits in here will fail with 'Author identity unknown'." >&2
  echo "git-identity: fix on the HOST, then devtools/devcontainer.sh rebuild:" >&2
  echo "git-identity:   git config --global user.name 'Your Name'" >&2
  echo "git-identity:   git config --global user.email 'you@example.com'" >&2
  exit 0
fi

git config --global user.name "$name"
git config --global user.email "$email"
echo "git-identity: $name <$email> -> $HOME/.gitconfig"
