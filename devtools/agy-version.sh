#!/usr/bin/env bash
# Report -- and bump -- the Antigravity CLI release the GPU images are pinned to.
#
#   devtools/agy-version.sh                    pinned vs upstream vs installed
#   devtools/agy-version.sh --apply            rewrite the pin to upstream's current release
#   devtools/agy-version.sh --apply 1.4.4-4783783166214144
#                                              rewrite it to an explicit release
#   devtools/agy-version.sh --apply --rebuild  ... and rebuild the images
#                                              that already exist locally
#   devtools/agy-version.sh --porcelain        one `key value` per line
#
# The report, and --rebuild after it rebuilds, also name every running
# container of this project still on a pre-bump image -- a container keeps the
# image it was created from, so rebuilding the image is not enough. It says how
# to recreate each one and never does it itself.
#
# This is devtools/claude-version.sh's twin, and the differences are all
# upstream's:
#
#   TWO lines per file, not one. The pin is `ARG AGY_RELEASE=` AND
#   `ARG AGY_SHA512=` in each file named by AGY_PIN_FILES in
#   devtools/config.sh, because a release id does not say which bytes it names
#   and docker/install-agy.sh VERIFIES the hash rather than recording it. A
#   release bumped without its hash fails the build, loudly, which is the
#   intended behaviour -- but it is still a bump that reached one line of two,
#   so this rewrites both or neither.
#
#   A RELEASE, not a version. `1.4.4-4783783166214144` -- the version plus an
#   opaque build id, which together are the storage URL's path segment.
#   Upstream's manifest endpoint serves only the CURRENT release, so a version
#   on its own is not addressable the way `claude-code@2.1.292` is.
#
#   ROLLING BACK NEEDS A RELEASE ID YOU ALREADY HAVE -- a previous value of the
#   pin, which is in git. Upstream publishes no index of older releases, and
#   the manifest carries a hash for the current one only, so `--apply
#   <release>` for anything else DOWNLOADS that tarball (~60MB) to hash it.
#
# Exit status: 0 when every file agrees with every other (and, for --apply,
# when the rewrite landed), 1 otherwise. Being BEHIND upstream is not a
# failure -- a pin is a deliberate choice of release, and this script's job is
# to make the gap visible, not to decide it.
#:
# Reading the manifest needs the network. It is one unauthenticated GET of
# $AGY_MANIFEST_BASE/$AGY_MANIFEST_PLATFORM.json with a short timeout; when it
# fails, the upstream column reads `-` and everything else still works, which
# is what keeps doctor.sh usable on a box with no outbound route.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

MANIFEST_BASE=${AGY_MANIFEST_BASE:-https://antigravity-cli-auto-updater-974169037036.us-central1.run.app/manifests}
MANIFEST_PLATFORM=${AGY_MANIFEST_PLATFORM:-linux_amd64}
TIMEOUT=${AGY_MANIFEST_TIMEOUT:-8}

# The tarball URL's platform segment, which is NOT the manifest's spelling of
# the same platform (linux-x64 vs linux_amd64; both upstream's). Kept here so
# `--apply <an older release>` builds the same URL docker/install-agy.sh does.
URL_BASE=${AGY_URL_BASE:-https://storage.googleapis.com/antigravity-public/antigravity-cli}
URL_PLATFORM=${AGY_PLATFORM:-linux-x64}

tarball_url() { echo "${URL_BASE}/$1/${URL_PLATFORM}/cli_${URL_PLATFORM//-/_}.tar.gz"; }

# --- the pin ---------------------------------------------------------------

# Every file that carries the ARGs, as an array. A path that does not exist is
# a hard error rather than a skip: a renamed Dockerfile that silently drops out
# of this list is the drift, not a tidying-up.
pin_files() {
  config_lines "${AGY_PIN_FILES:-}"
  PIN_FILES=("${CONFIG_LINES[@]}")
  local f
  for f in "${PIN_FILES[@]}"; do
    [ -f "$f" ] || { echo "agy-version: no such file: $f (AGY_PIN_FILES)" >&2; exit 1; }
  done
  [ "${#PIN_FILES[@]}" -gt 0 ] || {
    echo "agy-version: AGY_PIN_FILES is empty -- nothing to report on" >&2
    exit 1
  }
}

release_in() { sed -n 's/^ARG AGY_RELEASE=//p' "$1" | head -1; }
sha_in()     { sed -n 's/^ARG AGY_SHA512=//p'  "$1" | head -1; }

# The pinned release and hash, or nothing when the files disagree. DISAGREE is
# left holding a `file=release` list for EVERY file either way -- the whole
# list, not just up to the first clash, because a partial bump that reached one
# file of three is the case this exists to name, and naming two of them would
# send you back to grep for the third.
#
# The HASH is part of the comparison, not a second check after it: two files
# naming the same release with different hashes is a worse state than two
# naming different releases, since one of them will fail its build with what
# reads like a corrupt download.
pinned_release() {
  local f r s first="" firsts=""
  DISAGREE=()
  PINNED=""
  PINNED_SHA=""
  for f in "${PIN_FILES[@]}"; do
    r=$(release_in "$f"); s=$(sha_in "$f")
    DISAGREE+=("$f=${r:-NONE}")
    [ -n "$first" ]  || first=$r
    [ -n "$firsts" ] || firsts=$s
    [ "$r" = "$first" ]  || first=DISAGREE
    [ "$s" = "$firsts" ] || first=DISAGREE
  done
  [ "$first" = DISAGREE ] && return 1
  PINNED=$first
  PINNED_SHA=$firsts
  [ -n "$PINNED" ] && [ -n "$PINNED_SHA" ]
}

# --- upstream ---------------------------------------------------------------

# One GET, parsed once, into three globals: the version, the release id (the
# URL's path segment -- the thing this project pins) and the tarball's sha512.
# Empty on any failure, which every caller treats as "unreachable" rather than
# as an answer.
UP_VERSION="" UP_RELEASE="" UP_SHA=""
read_manifest() {
  local json
  json=$(curl -fsSL --max-time "$TIMEOUT" "${MANIFEST_BASE}/${MANIFEST_PLATFORM}.json" 2>/dev/null) || return 0
  # The release id is a path segment of the url, not a field of its own:
  #   .../antigravity-cli/<release>/<platform>/cli_<platform>.tar.gz
  eval "$(printf '%s' "$json" | python3 -c '
import json, sys, shlex
try:
    m = json.load(sys.stdin)
    url = m.get("url", "")
    parts = url.split("/")
    rel = parts[-3] if len(parts) >= 3 else ""
    print("UP_VERSION=" + shlex.quote(str(m.get("version", ""))))
    print("UP_RELEASE=" + shlex.quote(rel))
    print("UP_SHA="     + shlex.quote(str(m.get("sha512", ""))))
except Exception:
    pass
' 2>/dev/null)"
}

# The sha512 of a release that is NOT the current one. The manifest cannot
# answer for it, so the only honest way to fill the second pin line is to fetch
# the bytes and hash them -- ~60MB, which is why this is never on the report
# path and only runs for an explicit `--apply <release>`.
sha_of_release() {
  local rel=$1 url tmp sum
  url=$(tarball_url "$rel")
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' RETURN
  echo "agy-version: hashing $url" >&2
  curl -fsSL --max-time 300 -o "$tmp/agy.tar.gz" "$url" || {
    echo "agy-version: could not download $url" >&2
    echo "  a release is addressed by version AND build id -- check the id you gave" >&2
    return 1
  }
  sum=$(sha512sum "$tmp/agy.tar.gz" | cut -d' ' -f1)
  echo "$sum"
}

# What `agy` on THIS side actually is. Host and container both answer, and the
# answer means different things: in a container it should equal the pin's
# VERSION half, on the host it is just whatever the user installed and is
# reported for contrast rather than checked.
#
# Only the version half: `agy --version` prints 1.4.4 and never the build id,
# which is exactly why the Dockerfiles also carry AGY_RELEASE as an ENV.
installed_version() {
  command -v agy >/dev/null 2>&1 || return 0
  agy --version 2>/dev/null | head -1
}

release_ok() { [[ $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z]+)?$ ]]; }
sha512_ok()  { [[ $1 =~ ^[0-9a-f]{128}$ ]]; }

# --- rewrite ----------------------------------------------------------------

# Both lines, in every file, or none. A file missing either ARG stops the run
# BEFORE anything is written, so a half-applied bump is not a state this can
# leave behind.
apply_pin() {
  local want=$1 wantsha=$2 f before
  release_ok "$want"   || { echo "agy-version: '$want' is not a release id" >&2; exit 1; }
  sha512_ok "$wantsha" || { echo "agy-version: '$wantsha' is not a sha512" >&2; exit 1; }
  for f in "${PIN_FILES[@]}"; do
    [ -n "$(release_in "$f")" ] || { echo "agy-version: $f has no ARG AGY_RELEASE line" >&2; exit 1; }
    [ -n "$(sha_in "$f")" ]     || { echo "agy-version: $f has no ARG AGY_SHA512 line" >&2; exit 1; }
  done
  for f in "${PIN_FILES[@]}"; do
    before=$(release_in "$f")
    sed -i -e "s/^ARG AGY_RELEASE=.*/ARG AGY_RELEASE=${want}/" \
           -e "s/^ARG AGY_SHA512=.*/ARG AGY_SHA512=${wantsha}/" "$f"
    if [ "$(release_in "$f")" != "$want" ] || [ "$(sha_in "$f")" != "$wantsha" ]; then
      echo "agy-version: rewrite of $f did not take" >&2; exit 1
    fi
    printf '  %s: %s -> %s\n' "$f" "$before" "$want"
  done
}

# Rebuild only what is already built. Nothing here invents a ~40GB `combined`
# build on a box that has never had one; the images it skips are named, so the
# choice stays the caller's.
rebuild_built_images() {
  local f target
  for f in "${PIN_FILES[@]}"; do
    target=${f##*Dockerfile.}
    if docker image inspect "${PROJECT_NAME}:${target}" >/dev/null 2>&1; then
      echo "agy-version: + docker/build.sh $target"
      docker/build.sh "$target"
    else
      echo "agy-version: ${PROJECT_NAME}:${target} not built here -- skipping"
      echo "             build it with: docker/build.sh $target"
    fi
  done
}

# --- the running containers -------------------------------------------------

# Which running containers of this project predate the image they came from.
# The reasoning is devtools/claude-version.sh's, which has it at length: a
# container keeps the image it was CREATED from, so a rebuilt image leaves it
# behind, and image ids cannot be compared directly because
# `updateRemoteUserUID` derives a `vsc-*-uid` image from ours. What holds is
# that a derived image carries every layer of its parent, so a container is
# current exactly when the current `<PROJECT_NAME>:<variant>`'s TOP layer is
# among its own image's layers.
#
# REPORT ONLY: it never recreates a container, since someone may be working in
# it. Silent when docker is absent or its daemon is down -- which is the inside
# of every container, where doctor.sh has this covered.
image_layers() {
  docker image inspect -f '{{range .RootFS.Layers}}{{println .}}{{end}}' "$1" 2>/dev/null \
    | sed '/^$/d'
}

report_containers() {
  command -v docker >/dev/null 2>&1 || return 0
  local ids
  ids=$(docker ps -q --filter "label=devcontainer.project=$PROJECT_NAME" 2>/dev/null) || return 0
  if [ -z "$ids" ]; then
    echo "containers none running"
    return 0
  fi

  local cid name cfg folder variant top rel n=0 stale=() hints=() unchecked=()
  for cid in $ids; do
    n=$((n + 1))
    name=$(docker inspect -f '{{.Name}}' "$cid" 2>/dev/null || echo "$cid")
    name=${name#/}
    cfg=$(docker inspect -f '{{index .Config.Labels "devcontainer.config_file"}}' "$cid" 2>/dev/null || true)
    folder=$(docker inspect -f '{{index .Config.Labels "devcontainer.local_folder"}}' "$cid" 2>/dev/null || true)
    variant=
    [ -z "$cfg" ] || variant=$(basename "$(dirname "$cfg")")
    if [ -z "$variant" ]; then
      unchecked+=("$name (no devcontainer.config_file label -- not started by devcontainer.sh?)")
      continue
    fi
    top=$(image_layers "${PROJECT_NAME}:$variant" | tail -1 || true)
    if [ -z "$top" ]; then
      unchecked+=("$name ($variant: no ${PROJECT_NAME}:$variant image here to compare against)")
      continue
    fi
    # Not `grep -q`: it exits at the first match, and under pipefail the
    # SIGPIPE that leaves the writer would read as no match at all.
    image_layers "$(docker inspect -f '{{.Image}}' "$cid" 2>/dev/null)" \
      | grep -xF "$top" >/dev/null && continue
    rel=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$cid" 2>/dev/null \
      | sed -n 's/^AGY_RELEASE=//p' | head -1 || true)
    stale+=("$name ($variant, agy ${rel:-?}, created from an image that is no longer ${PROJECT_NAME}:$variant)")
    hints+=("${folder:-<its checkout>}/devtools/devcontainer.sh --$variant rebuild")
  done

  if [ ${#stale[@]} -eq 0 ]; then
    echo "containers $n running, none on a pre-bump image"
  else
    echo "containers $n running, ${#stale[@]} on a pre-bump image:"
    local i
    for i in "${!stale[@]}"; do
      echo   "           ${stale[$i]}"
      echo   "           recreate it: ${hints[$i]}"
    done
  fi
  [ ${#unchecked[@]} -eq 0 ] || printf '           not checked: %s\n' "${unchecked[@]}"
}

# --- arguments --------------------------------------------------------------

mode=report
want=""
rebuild=
while [ $# -gt 0 ]; do
  case $1 in
    --apply)     mode=apply ;;
    --rebuild)   mode=apply; rebuild=1 ;;
    --porcelain) mode=porcelain ;;
    -h|--help)   usage; exit 0 ;;
    -*)          echo "agy-version: unknown option $1" >&2; usage >&2; exit 2 ;;
    *)           want=$1 ;;
  esac
  shift
done

pin_files

case $mode in
  porcelain)
    pinned_release || true
    read_manifest
    printf 'pinned %s\n'    "${PINNED:--}"
    printf 'sha %s\n'       "${PINNED_SHA:--}"
    printf 'files %s\n'     "${DISAGREE[*]}"
    printf 'upstream %s\n'  "${UP_RELEASE:--}"
    printf 'installed %s\n' "$(installed_version || true)"
    [ -n "$PINNED" ]
    ;;

  report)
    read_manifest
    here=$(installed_version)
    if pinned_release; then
      printf 'pinned     %s   (%s)\n' "$PINNED" "${PIN_FILES[*]}"
      printf 'sha512     %s\n' "$PINNED_SHA"
    else
      printf 'pinned     DISAGREE: %s\n' "${DISAGREE[*]}"
      echo   '           (a release AND its sha512 must match across all of them)'
      echo   '           fix with: devtools/agy-version.sh --apply <release>'
    fi
    printf 'upstream   %s   (%s/%s.json)\n' "${UP_RELEASE:-unreachable}" "$MANIFEST_BASE" "$MANIFEST_PLATFORM"
    printf 'installed  %s   (%s)\n' "${here:-none}" "$(command -v agy || echo 'not on PATH here')"
    echo
    if [ -n "$PINNED" ] && [ -n "$UP_RELEASE" ] && [ "$PINNED" != "$UP_RELEASE" ]; then
      echo "the pin is behind upstream: $PINNED -> $UP_RELEASE"
      echo "  bump it:  devtools/agy-version.sh --apply"
      echo "  then:     docker/build.sh cuda   (and hip / combined as you use them)"
    elif [ -n "$PINNED" ] && [ "$PINNED" = "$UP_RELEASE" ]; then
      echo "the pin is current"
      # The hash is pinned separately, so "current release, wrong bytes" is a
      # reachable state -- a hand-edited AGY_RELEASE, or a bump that stopped
      # halfway. Say so here rather than at `docker build`, which is where it
      # would otherwise surface as a checksum failure.
      if [ -n "$UP_SHA" ] && [ "$PINNED_SHA" != "$UP_SHA" ]; then
        echo "BUT the pinned sha512 is not upstream's for that release:"
        echo "  pinned   $PINNED_SHA"
        echo "  upstream $UP_SHA"
        echo "  every build of a GPU image will fail its checksum until this is fixed:"
        echo "    devtools/agy-version.sh --apply"
      fi
    fi
    report_containers
    [ -n "$PINNED" ]
    ;;

  apply)
    # The format check BEFORE the network, not inside apply_pin after it: a
    # typo'd id would otherwise spend a 404 (or, worse, a 60MB download) before
    # anything looked at its shape, and report it as an unreachable release
    # rather than as a malformed one.
    if [ -n "$want" ] && ! release_ok "$want"; then
      echo "agy-version: '$want' is not a release id" >&2
      echo "  it is a version and a build id, as in 1.4.4-4783783166214144" >&2
      exit 1
    fi
    read_manifest
    wantsha=""
    if [ -z "$want" ]; then
      [ -n "$UP_RELEASE" ] || { echo "agy-version: manifest unreachable and no release given" >&2; exit 1; }
      want=$UP_RELEASE
      wantsha=$UP_SHA
      echo "agy-version: upstream's current release is $want"
    elif [ "$want" = "$UP_RELEASE" ]; then
      # The manifest already carries this one's hash: no 60MB download.
      wantsha=$UP_SHA
    else
      # Anything else -- a rollback, or a release named while the manifest is
      # unreachable -- has to be hashed from the bytes.
      wantsha=$(sha_of_release "$want") || exit 1
    fi
    apply_pin "$want" "$wantsha"
    if [ -n "$rebuild" ]; then
      rebuild_built_images
      report_containers
    else
      echo "agy-version: rebuild to pick it up -- docker/build.sh cuda (and hip / combined)"
      echo "             or re-run with --rebuild to do the ones already built here"
    fi
    ;;
esac
