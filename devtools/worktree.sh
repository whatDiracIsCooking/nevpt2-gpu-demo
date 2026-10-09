#!/usr/bin/env bash
# Create and remove git worktrees for this repo, following the layout the rest
# of the tooling already assumes.
#
#   devtools/worktree.sh add <name> [start-point] [--up]  (--up: up + shell in)
#   devtools/worktree.sh add <name> --hip --up            (any variant flag)
#   devtools/worktree.sh rm  <name> [--force]
#   devtools/worktree.sh sync                    fast-forward local main to origin
#   devtools/worktree.sh gc [--dry-run] [--yes]  remove orphaned docker state
#   devtools/worktree.sh list
#
# The layout: every worktree is a sibling directory under the repo root, named
# after its branch --  <root>/main, <root>/<name>, ...  --  and this repo is
# already laid out that way. `add foo` therefore makes <root>/foo on branch
# `foo`; `rm foo` removes it again.
#: --- end of `usage` output; what follows is for a reader of the file --------
#
# A worktree container works with no devcontainer.json edit: that file bind
# mounts the *common* git dir (main/.git) at its literal host path, and a
# worktree's `.git` file points into main/.git/worktrees/<name>. So the only
# machine-specific assumption is that `main` stays where it is -- if you move
# the main checkout you must also fix the .git mount path in devcontainer.json.
#
# A variant flag (--cuda / --hip / --combined, or whatever
# .devcontainer/ holds) is passed straight through to devcontainer.sh, so
# `add <name> --hip --up` gives that worktree a ROCm container. It affects
# --up only: the worktree itself is backend-neutral, and the flag is not
# remembered -- later calls in that checkout need it again, or an exported
# DEVCONTAINER_CONFIG. A worktree per backend is the tidy way to run two at
# once, since the volume names carry the devcontainer id (below) and so are not
# shared between them, unlike two variants of one worktree.
#
# Removal is the part that is easy to get wrong, and the reason both `rm` and
# `gc` ask DOCKER which volumes a container holds instead of rebuilding the
# names here. The names are
# `<PROJECT_NAME>-{sccache,build,build-hip,uvcache}-${devcontainerId}`, and
# that last field is the load-bearing one: `devcontainerId` is a hash the
# devcontainer CLI derives from the workspace folder, NOT the worktree
# basename, so there is no way back from a volume name to the worktree that
# owns it. This file used to assume the basename was in there and build
# `<PROJECT_NAME>-<suffix>-<name>` by hand; nothing matched, which meant `rm`
# silently left every volume behind and `gc` called every volume on the box an
# orphan -- the live primary checkout's included. `git worktree remove` knows
# nothing about any of it either, so `rm` tears the container down and takes
# the volumes it was holding with it.
#
# `rm` is deliberately variant-BLIND where `devcontainer.sh down` is not: it
# matches containers on the workspace folder alone, so every variant of that
# worktree goes, along with every vsc-<name>-<hash> image the variants built.
# Tearing a worktree down should not depend on remembering which container you
# last had up in it.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

WORKSPACE=$REPO_ROOT
ROOT=$(dirname "$WORKSPACE")

# The named volumes one container holds, this project's only. This is the one
# place either command learns a volume name, which is what keeps `rm` and `gc`
# from drifting apart -- and it cannot drift from devcontainer.json either,
# since it reads what docker actually mounted rather than a list of suffixes
# kept in step by hand. The $PROJECT_NAME filter is the safety gate: a
# devcontainer also mounts things this project did not create.
container_volumes() {
  docker inspect --format \
    '{{range .Mounts}}{{if eq .Type "volume"}}{{println .Name}}{{end}}{{end}}' \
    "$1" 2>/dev/null | grep -E "^$PROJECT_NAME-" || true
}

# Every devcontainer on the box as `<id>\t<host folder>`. EVERY repo's, not
# just ours -- `devcontainer.local_folder` is set by the CLI generally, so each
# caller has to scope the result itself before deleting anything.
devcontainer_folders() {
  docker ps -a --filter "label=devcontainer.local_folder" --format '{{.ID}}' \
    | while read -r cid; do
        [ -n "$cid" ] || continue
        cf=$(docker inspect --format \
          '{{ index .Config.Labels "devcontainer.local_folder" }}' \
          "$cid" 2>/dev/null)
        printf '%s\t%s\n' "$cid" "$cf"
      done
}

# How this project's build images are recognised. The devcontainer CLI tags an
# image `vsc-<worktree-basename>-<64hex>`, which says nothing about WHICH repo
# it came from -- two checkouts on one box both produce a `vsc-main-...`. So
# image removal is gated on a label every docker/Dockerfile.* stamps
# (LABEL devcontainer.project="${PROJECT_NAME}", passed as a build arg from
# devcontainer.json) rather than on the tag alone. An image without the label
# is never removed: it belongs to another project, or predates the label, and
# in both cases guessing is how you delete someone else's base image.
IMAGE_LABEL="devcontainer.project=$PROJECT_NAME"

# Our images, as tab-separated `<repo>\t<tag>\t<id>` rows -- one per TAG, which
# is what `docker images` prints. Empty when docker is absent.
project_images() {
  docker images --filter "label=$IMAGE_LABEL" \
    --format '{{.Repository}}'$'\t''{{.Tag}}'$'\t''{{.ID}}' 2>/dev/null || true
}

# The worktree basename a devcontainer image name belongs to, or failure when
# the name is not one. The CLI tags `vsc-<basename>-<64hex>` -- and appends
# `-uid` when devcontainer.json sets `updateRemoteUserUID`, which BOTH of this
# repo's configs do, so in practice every image here carries that suffix. Both
# matchers used to require the hash at the END of the name, so every real image
# failed the test and a multi-GB image leaked per worktree -- the one thing the
# teardown comment below calls the largest leak. The hash is still matched in
# full, so `feat` cannot cross-match `feat-batching`.
image_repo_worktree() {
  local rest=${1#vsc-}
  [ "$rest" != "$1" ] || return 1
  rest=${rest%-uid}
  local h=${rest##*-} b=${rest%-*}
  [[ $h =~ ^[0-9a-f]{64}$ ]] || return 1
  [ -n "$b" ] || return 1
  printf '%s\n' "$b"
}

# All git commands run against the checkout this script lives in, never against
# a hardcoded `main`. `git worktree` operates on the whole worktree set from any
# member, so this is equivalent -- and it sidesteps machines where `main` is
# owned by another user (git's safe.directory guard would otherwise reject it).
GIT=(git -C "$WORKSPACE")

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

# Every container/volume the given worktree directory owns. `docker` may not be
# present (the worktree tooling is useful without ever building a container), so
# callers tolerate a failure here.
teardown_container() {
  local path=$1 name ids id
  name=$(basename "$path")
  if ! command -v docker >/dev/null 2>&1; then
    echo "  docker not found -- skipping container and volume cleanup"
    return 0
  fi
  ids=$(docker ps -aq \
    --filter "label=devcontainer.local_folder=$path" 2>/dev/null || true)
  # ORDER MATTERS: ask every container what it holds BEFORE removing any of
  # them. A volume name cannot be rebuilt from the worktree name (see the
  # devcontainerId note at the top of this file), so once the containers are
  # gone the only thing that knew is gone too -- and the volumes leak with no
  # `rm <name>` that will ever match them again. Both variants of a worktree
  # are collected: `rm` is variant-blind on purpose, and cuda and hip have
  # different devcontainer ids and so disjoint volume sets.
  local vols=""
  if [ -n "$ids" ]; then
    vols=$(while read -r id; do
             [ -n "$id" ] || continue
             container_volumes "$id"
           done <<<"$ids" | sort -u)
    # Loop: an exited container from a crashed run carries the same label, and
    # a quoted "$ids" holding two newline-separated ids reaches docker as one
    # bogus name -- leaving both containers, and their volumes, behind.
    while read -r id; do
      [ -n "$id" ] || continue
      echo "  removing container $id"
      docker rm -f "$id" >/dev/null || true
    done <<<"$ids"
  else
    echo "  no container for $path"
  fi
  # Now that nothing references them. A volume whose container was already
  # removed by hand before this ran is NOT reachable here -- `gc` catches that
  # case, where an unreferenced <PROJECT_NAME>-* volume is an orphan by
  # definition.
  if [ -n "$vols" ]; then
    while read -r vol; do
      [ -n "$vol" ] || continue
      echo "  removing volume $vol"
      docker volume rm "$vol" >/dev/null || true
    done <<<"$vols"
  fi
  # The build also leaves a per-worktree image, tagged vsc-<basename>-<64hex>
  # (the `naming to .../vsc-...` line in `add`'s build log). That hash is of
  # the worktree PATH, not of the build, so a rebuild reuses the name rather
  # than adding another -- and every GPU variant that has been up here left a
  # `:<variant>` tag on its own image (devcontainer.sh's tag_variant_image),
  # which is what keeps the variants other than the last one attributable to
  # this worktree at all. Neither the container removal above nor `git
  # worktree remove` touches them, so drop them here -- on a multi-GB base
  # image this is the largest thing that would otherwise leak. An image an
  # earlier rebuild left with NO tag is `docker image prune`'s job rather than
  # ours, since `<none>` says nothing about which worktree it came from.
  #
  # REMOVAL IS BY TAG, NOT BY IMAGE ID, and that is not a style choice.
  # `updateRemoteUserUID` is a no-op on a host that is already uid 1000, so the
  # CLI's derived image is byte-identical between worktrees and docker gives
  # them ONE id under two names -- measured here: a throwaway worktree's
  # `vsc-gctest-<hash>-uid` and the live `vsc-docker-perms-<hash>-uid` shared
  # id cd402c286148. `docker rmi -f <id>` drops every tag on an id, so by-id
  # removal would have taken the live worktree's image with it. By tag, docker
  # untags and deletes the image only when the last tag goes.
  local imgs
  imgs=$(project_images | while IFS=$'\t' read -r rep tag _id; do
           [ -n "$rep" ] || continue
           [ "$tag" = "<none>" ] && continue
           b=$(image_repo_worktree "$rep") || continue
           [ "$b" = "$name" ] && printf '%s:%s\n' "$rep" "$tag"
         done | sort -u)
  if [ -n "$imgs" ]; then
    while read -r img; do
      [ -n "$img" ] || continue
      echo "  removing image $img"
      docker rmi -f "$img" >/dev/null || true
    done <<<"$imgs"
  fi
}

cmd_add() {
  local name="" start="" up=0 variant=""
  for arg in "$@"; do
    case "$arg" in
      --up) up=1 ;;
      # A variant flag is not interpreted here -- it is handed to
      # devcontainer.sh below, which owns the flag-to-config mapping. Accepting
      # any --<word> that is not one of ours keeps the two in step: adding a
      # .devcontainer/<name>/ needs no edit in this file either. The cost is
      # that a typo is only caught by devcontainer.sh, and only under --up.
      --*)  if [ -n "$variant" ]; then
              echo "add: two variant flags: $variant and $arg" >&2; exit 2
            fi
            variant=$arg ;;
      -*)   echo "unknown flag: $arg" >&2; exit 2 ;;
      *)    if [ -z "$name" ]; then name=$arg; else start=$arg; fi ;;
    esac
  done
  [ -n "$name" ] || { echo "add: need a worktree name" >&2; exit 2; }
  if [ "$name" = "main" ]; then
    echo "add: refusing to touch 'main'" >&2; exit 2
  fi

  local path="$ROOT/$name"
  if [ -e "$path" ]; then
    echo "add: $path already exists" >&2; exit 2
  fi
  # Check before calling git: `git worktree add` creates the branch first and
  # only then the directory, so if the root is not writable it leaves a dangling
  # branch behind and dies on a cryptic "could not create leading directories".
  # If the root is owned by another user, fix it once with
  #   sudo chown $(id -u) "$ROOT"
  # Numeric uid, not name: the host user and the container user are typically
  # both uid 1000 under different names (yours vs `ubuntu`), so a `chown
  # ubuntu` fails on the host while the uid always resolves.
  if [ ! -w "$ROOT" ]; then
    echo "add: $ROOT is not writable by $(id -un) (uid $(id -u))" >&2
    echo "     fix once with: sudo chown $(id -u) $ROOT" >&2
    exit 1
  fi

  # Offline heads-up before we branch. `add` forks the new worktree from this
  # checkout's HEAD (below), so if HEAD is behind its upstream -- typically
  # `main` behind origin/main after a merged PR -- the new branch starts life
  # on stale code. We do NOT fetch here (that is `sync`'s job); this reflects
  # the last fetch. Only nag when no explicit start-point was given, since a
  # named start-point is a deliberate choice.
  if [ -z "$start" ]; then
    # NB: not `up` -- that name holds the --up flag above, and shadowing it here
    # left it set to the upstream ref, so `[ "$up" -eq 1 ]` below died with
    # "integer expression expected" and --up never fired.
    local upstream behind
    upstream=$("${GIT[@]}" rev-parse --abbrev-ref --symbolic-full-name \
         '@{upstream}' 2>/dev/null || true)
    if [ -n "$upstream" ]; then
      behind=$("${GIT[@]}" rev-list --count "HEAD..$upstream" 2>/dev/null || echo 0)
      if [ "${behind:-0}" -gt 0 ]; then
        echo "note: HEAD is $behind commit(s) behind $upstream (as of last fetch);"
        echo "      the new worktree will fork from stale code. Refresh with:"
        echo "        devtools/worktree.sh sync"
      fi
    fi
  fi

  # Attach an existing branch of this name if there is one, otherwise create it.
  # `git worktree add -b` fails if the branch exists, so branch first.
  if "${GIT[@]}" show-ref --verify --quiet "refs/heads/$name"; then
    echo "branch '$name' exists -- attaching it"
    "${GIT[@]}" worktree add "$path" "$name"
  else
    "${GIT[@]}" worktree add "$path" -b "$name" ${start:+"$start"}
  fi

  echo
  echo "worktree ready at $path (branch $name)"
  # ${variant:+"$variant"} rather than a bare "$variant": an empty variant must
  # expand to NO argument, and a quoted empty string would reach devcontainer.sh
  # as an argument that is not a command and abort with "unknown command: ".
  if [ "$up" -eq 1 ]; then
    echo "bringing its container up${variant:+ ($variant)}..."
    "$path/devtools/devcontainer.sh" ${variant:+"$variant"} up
    echo "opening a shell inside..."
    # exec: this is the last step, and `shell` is interactive -- hand the tty
    # straight to it rather than returning through this script.
    exec "$path/devtools/devcontainer.sh" ${variant:+"$variant"} shell
  else
    local v=${variant:+"$variant "}
    echo "next: cd $path && devtools/devcontainer.sh ${v}up && devtools/devcontainer.sh ${v}shell"
    echo "  (or: devtools/worktree.sh add $name ${v}--up  to do all three at once)"
  fi
}

cmd_rm() {
  local name="" force=0
  for arg in "$@"; do
    case "$arg" in
      --force|-f) force=1 ;;
      -*)         echo "unknown flag: $arg" >&2; exit 2 ;;
      *)          name=$arg ;;
    esac
  done
  [ -n "$name" ] || { echo "rm: need a worktree name" >&2; exit 2; }
  if [ "$name" = "main" ]; then
    echo "rm: refusing to remove 'main'" >&2; exit 2
  fi

  local path="$ROOT/$name"
  if [ "$path" = "$WORKSPACE" ]; then
    echo "rm: refusing to remove the worktree you are calling from" >&2
    echo "    run this from another checkout (e.g. main)" >&2
    exit 2
  fi
  if [ ! -d "$path" ]; then
    echo "rm: no worktree at $path" >&2; exit 2
  fi

  echo "tearing down container and volumes for $name"
  teardown_container "$path"

  echo "removing worktree $path"
  if [ "$force" -eq 1 ]; then
    "${GIT[@]}" worktree remove --force "$path"
  else
    # Refuses if the worktree has uncommitted changes -- rerun with --force.
    "${GIT[@]}" worktree remove "$path"
  fi
  "${GIT[@]}" worktree prune

  # Delete the branch too, but never silently lose commits: -d refuses an
  # unmerged branch, -D (under --force) does not.
  if "${GIT[@]}" show-ref --verify --quiet "refs/heads/$name"; then
    if [ "$force" -eq 1 ]; then
      "${GIT[@]}" branch -D "$name"
    elif "${GIT[@]}" branch -d "$name" 2>/dev/null; then
      :
    else
      echo "branch '$name' has unmerged commits -- left in place"
      echo "  delete with: git -C '$WORKSPACE' branch -D $name"
    fi
  fi
}

# Fast-forward the local `main` checkout to origin/main. Nothing else in the
# tooling does this: after a PR merges, origin/main advances but local main
# stays put, and `add` (which forks from HEAD) then starts new worktrees on
# stale code. --ff-only is deliberate -- if local main has diverged, refuse
# rather than manufacture a merge commit you would have to look at, not
# automate.
cmd_sync() {
  # Find the checkout that actually holds `main` rather than assuming the
  # <root>/main sibling: this is the one command that is useful from ANY
  # checkout (the pr flow calls it right after a merge), and the assumption
  # broke everywhere else -- from .claude/worktrees/<name> it resolved to
  # .claude/worktrees/main, and a plain single-checkout clone has main at the
  # repo root, so both failed with "no main checkout at ...". `git worktree
  # list` knows where main is in every layout; fall back to the sibling path
  # only to keep the old error message when nothing has main checked out.
  local MAIN
  MAIN=$("${GIT[@]}" worktree list --porcelain | awk '
    /^worktree /       { p = substr($0, 10) }
    $0 == "branch refs/heads/main" { print p; exit }')
  [ -n "$MAIN" ] || MAIN="$ROOT/main"
  [ -e "$MAIN/.git" ] || {
    echo "sync: no worktree has 'main' checked out (looked for $MAIN)" >&2
    exit 1; }
  local MG=(git -C "$MAIN")
  if ! "${MG[@]}" remote get-url origin >/dev/null 2>&1; then
    echo "sync: main checkout has no 'origin' remote" >&2; exit 1
  fi
  echo "fetching origin..."
  "${MG[@]}" fetch --prune origin
  # git refuses to update a checked-out branch's ref via fetch refspec, and
  # main IS checked out in the main worktree, so we fast-forward in place
  # rather than with `fetch origin main:main`. That needs main to be the branch
  # actually checked out there; if it is not, ff-ing would advance the wrong
  # branch, so bail with the behind-count instead.
  local cur
  cur=$("${MG[@]}" symbolic-ref --short -q HEAD || true)
  if [ "$cur" != "main" ]; then
    local behind
    behind=$("${MG[@]}" rev-list --count main..origin/main 2>/dev/null || echo '?')
    echo "sync: main checkout is on '$cur', not 'main' -- not fast-forwarding"
    echo "      local main is $behind behind origin/main; switch to it there to ff"
    exit 1
  fi
  if ! "${MG[@]}" merge --ff-only origin/main; then
    echo "sync: local main has diverged from origin/main -- resolve by hand" >&2
    exit 1
  fi
}

# Reconcile docker state against the worktrees that actually exist. Removing a
# worktree with plain `git worktree remove` (bypassing `rm` above), or an
# agent's own worktrees under .claude/worktrees/, leave the container, the
# named volumes and the per-worktree build image behind with no `rm <name>`
# that will ever match them. gc finds those and (with confirmation) deletes
# them. Scoped to THIS repo's resources by label and name prefix -- it never
# touches images or volumes it did not create, so it is safe beside other
# docker workloads on the same host.
cmd_gc() {
  local dry=0 yes=0
  for arg in "$@"; do
    case "$arg" in
      --dry-run|-n) dry=1 ;;
      --yes|-y)     yes=1 ;;
      -*) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
  done
  if ! command -v docker >/dev/null 2>&1; then
    echo "gc: docker not found -- nothing to reconcile"
    echo "orphans: 0"
    return 0
  fi

  # Clear git's own stale bookkeeping first, but only on a real run: a worktree
  # whose directory was deleted by hand lingers as `prunable` until this runs,
  # and it keeps that basename "live" for the volume/image checks below. Skip
  # it under --dry-run so a report (doctor calls one) never mutates anything;
  # locked worktrees are left alone either way -- prune refuses them by design.
  [ "$dry" -eq 0 ] && "${GIT[@]}" worktree prune

  # Paths as well as basenames: the image check below can only match on the
  # basename the CLI puts in a tag, but the volume check compares whole folders
  # (that is what the container labels carry), and two checkouts of this repo on
  # one box share basenames while never sharing a path.
  local live_paths live_names
  live_paths=$("${GIT[@]}" worktree list --porcelain \
    | sed -n 's/^worktree //p' | sort -u)
  live_names=$(while read -r p; do
                 [ -n "$p" ] && basename "$p"
               done <<<"$live_paths" | sort -u)

  local -a orph_c=() orph_v=() orph_i=()

  # Containers: devcontainer labels each with the host folder it was built for.
  # Folder gone -> worktree gone -> the container is a leak. Scoped to folders
  # this checkout could actually own -- a direct child of our worktree root, or
  # one of the nested .claude/worktrees/<name> -- because that label is set by
  # every devcontainer on the box, including other repos', and a container
  # whose folder is gone is still not OURS to delete.
  while IFS=$'\t' read -r id folder; do
    [ -n "$id" ] || continue
    [ -n "$folder" ] || continue
    [ -d "$folder" ] && continue
    if [ "$(dirname "$folder")" = "$ROOT" ] \
       || [[ $folder == "$WORKSPACE/.claude/worktrees/"* ]]; then
      orph_c+=("$id|$folder")
    fi
  done < <(devcontainer_folders)

  # Volumes: `<PROJECT_NAME>-<suffix>-${devcontainerId}`, and that id is a hash
  # of the workspace folder rather than the worktree basename -- so there is
  # nothing in the name to compare a live worktree against. (The old test here
  # stripped a suffix and compared the REST to a live basename, which matched
  # nothing and reported every volume on the box as an orphan, the live primary
  # checkout's included.) Ask docker what is referenced instead, and classify by
  # the referring container's folder:
  #
  #   mounted by a container in a live worktree  -> keep, it is in use
  #   mounted only by containers elsewhere       -> ours only if one of those
  #                                                 folders is one THIS checkout
  #                                                 could own; another clone of
  #                                                 this repo is not ours
  #   mounted by nothing at all                  -> orphan by definition, which
  #                                                 is what a hand-removed
  #                                                 container leaves behind
  #
  # Orphaned CONTAINERS are removed before orphaned volumes (see below), so a
  # volume held only by one of those is free by the time its turn comes.
  local kept_vols="" ours_vols="" seen_vols=""
  while IFS=$'\t' read -r id folder; do
    [ -n "$id" ] || continue
    [ -n "$folder" ] || continue
    local cvols
    cvols=$(container_volumes "$id")
    [ -n "$cvols" ] || continue
    seen_vols+=$'\n'"$cvols"
    if grep -qxF "$folder" <<<"$live_paths"; then
      kept_vols+=$'\n'"$cvols"
    elif [ "$(dirname "$folder")" = "$ROOT" ] \
         || [[ $folder == "$WORKSPACE/.claude/worktrees/"* ]]; then
      ours_vols+=$'\n'"$cvols"
    fi
  done < <(devcontainer_folders)

  while read -r vol; do
    [ -n "$vol" ] || continue
    if grep -qxF "$vol" <<<"$kept_vols"; then continue; fi
    if grep -qxF "$vol" <<<"$seen_vols"; then
      if ! grep -qxF "$vol" <<<"$ours_vols"; then continue; fi
    fi
    orph_v+=("$vol")
  done < <(docker volume ls --format '{{.Name}}' \
           | grep -E "^$PROJECT_NAME-" || true)

  # Images: the build tags each worktree's image vsc-<basename>-<64hex>, plus
  # the `-uid` suffix both configs here provoke -- image_repo_worktree above
  # owns that shape and the exact-hash match that stops `feat` cross-matching
  # `feat-batching`. Only images carrying this project's label are considered
  # (see IMAGE_LABEL) -- the tag alone cannot tell two repos' `vsc-main-...`
  # apart. Orphan if the basename has no live worktree. Rebuild leftovers
  # for a LIVE worktree are deliberately not handled here -- they stay tagged
  # with a live name; `rm` clears a worktree's whole set on teardown, and
  # `docker image prune` clears cross-worktree churn.
  #
  # One entry per TAG, matching teardown_container: `docker images` prints a
  # row per tag, and every variant that has been up on a worktree leaves a
  # second tag on that worktree's image (devcontainer.sh's tag_variant_image).
  # Removal is by tag rather than by id for the shared-id reason spelled out in
  # teardown_container, so each tag genuinely is its own unit of work here --
  # `sort -u` is still wanted, but only to drop exact duplicate rows.
  while IFS=$'\t' read -r rep tag _id; do
    [ -n "$rep" ] || continue
    [ "$tag" = "<none>" ] && continue
    local b
    b=$(image_repo_worktree "$rep") || continue
    grep -qxF "$b" <<<"$live_names" || orph_i+=("$rep:$tag")
  done < <(project_images | sort -u)

  local total=$(( ${#orph_c[@]} + ${#orph_v[@]} + ${#orph_i[@]} ))

  if [ "${#orph_c[@]}" -gt 0 ]; then
    echo "orphaned containers (folder gone):"
    for c in "${orph_c[@]}"; do echo "  ${c%%|*}  (${c#*|})"; done
  fi
  if [ "${#orph_v[@]}" -gt 0 ]; then
    echo "orphaned volumes:"
    for v in "${orph_v[@]}"; do echo "  $v"; done
  fi
  if [ "${#orph_i[@]}" -gt 0 ]; then
    echo "orphaned images:"
    for i in "${orph_i[@]}"; do echo "  $i"; done
  fi

  # Stable machine-readable line -- doctor.sh greps for this prefix.
  echo "orphans: $total (${#orph_c[@]} containers," \
       "${#orph_v[@]} volumes, ${#orph_i[@]} images)"
  [ "$total" -gt 0 ] || { echo "nothing to reconcile"; return 0; }

  if [ "$dry" -eq 1 ]; then
    echo "(dry run -- nothing removed; re-run without --dry-run to delete)"
    return 0
  fi
  if [ "$yes" -ne 1 ]; then
    local ans
    printf "remove these %d resource(s)? [y/N] " "$total"
    read -r ans || ans=""
    case "$ans" in y|Y|yes|YES) ;; *) echo "aborted"; return 0 ;; esac
  fi
  for c in "${orph_c[@]}"; do
    echo "  removing container ${c%%|*}"
    docker rm -f "${c%%|*}" >/dev/null || true
  done
  for v in "${orph_v[@]}"; do
    echo "  removing volume $v"
    docker volume rm "$v" >/dev/null || true
  done
  for i in "${orph_i[@]}"; do
    echo "  removing image $i"
    docker rmi -f "$i" >/dev/null || true
  done
  echo "done."
}

case "${1:-}" in
  add)  shift; cmd_add  "$@" ;;
  rm)   shift; cmd_rm   "$@" ;;
  sync) shift; cmd_sync "$@" ;;
  gc)   shift; cmd_gc   "$@" ;;
  list) "${GIT[@]}" worktree list ;;
  ""|-h|--help|help) usage ;;
  *) echo "unknown command: $1" >&2; echo >&2; usage >&2; exit 2 ;;
esac
