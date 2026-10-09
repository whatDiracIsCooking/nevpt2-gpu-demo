---
name: devbox
description: >-
  Drive this repo's containers and its container-backed sibling worktrees —
  devtools/devcontainer.sh (up/rebuild/shell/test/down) and
  devtools/worktree.sh (add/rm/sync/gc/list). Use when the user wants to bring
  a container up, get a shell or run the suite inside it, rebuild after editing
  a Dockerfile or devcontainer.json, create or tear down a full container-backed
  worktree, or clean up leaked docker state. NOT for the lightweight
  .claude/worktrees/ checkouts — those are the `worktree` skill.
---

# Containers and container-backed worktrees

Two scripts, one model: **a container is keyed on its workspace folder path**,
so every checkout gets its own container, its own named volumes and its own
build image. That is what makes several worktrees usable at once, and it is
also why removal needs `worktree.sh rm` rather than `git worktree remove`.

Both scripts resolve their workspace from their own location, so they act on
*the checkout they live in*, from any cwd. `devtools/devcontainer.sh up` run
from worktree `foo` brings up `foo`'s container, never main's.

## The container

```bash
devtools/devcontainer.sh up        # reuse if it exists, else create
devtools/devcontainer.sh rebuild   # recreate from scratch
devtools/devcontainer.sh shell     # bash inside
devtools/devcontainer.sh test      # the Python tier inside (see the `test` skill)
devtools/devcontainer.sh down      # stop and remove the container
```

- **Use `rebuild`, never `up`, after editing `devcontainer.json` or a
  `Dockerfile`.** `up` reuses the running container, applies none of the change,
  and reports success with the same container id — the single most expensive
  mistake here, because everything downstream looks like a code bug. The
  workspace is a bind mount and the caches are named volumes, so both survive a
  rebuild; anything written to the container's own filesystem does not.
- `down` removes the container but **keeps the volumes** — the sccache store,
  the CMake build tree (`build`, or `build-hip` under `--hip`) and the uv
  download cache. That is deliberate: `up` again is cheap and nothing is lost.
  Reclaiming the volumes is `worktree.sh rm`'s job.
- `shell -c "<cmd>"` runs one command inside — how you inspect a container
  without an interactive session, and how the C++ tier is run:
  `devtools/devcontainer.sh shell -c devtools/cpp-tier.sh`.
- **`shell` and `test` need the container already running**, and say so:
  `devcontainer exec` neither creates one nor starts a stopped one, so after a
  reboot or a `docker stop` they refuse with *"the `<variant>` container …
  exists, but is stopped"* and print the `up` that fixes it. `up` on a stopped
  container is a `docker start` — no rebuild, seconds. Reach for `rebuild`
  only if the config or a `Dockerfile` has changed since.

### What the container takes from your host, and when

Three values cross the boundary at **create time** — `up` on a container that
does not exist yet, or `rebuild`. An `export` afterwards reaches a *running*
container never, which is the one thing to remember here:

| Host value | How it gets in | Without it |
|---|---|---|
| `GH_TOKEN` | `containerEnv` reads `${localEnv:GH_TOKEN}`; `docker/Dockerfile.base` installs `gh` and makes it git's push credential in `/etc/gitconfig` | `gh` is unauthenticated and `git push` fails — `doctor.sh` warns |
| your git identity | `devcontainer.sh` resolves `git var GIT_AUTHOR_IDENT` and exports `NEVPT2_GIT_USER_NAME`/`_EMAIL`; `.devcontainer/git-identity.sh` writes them to the container's `~/.gitconfig` from `postCreateCommand` | `git commit` fails with "Author identity unknown" *after* staging |
| `~/.claude` | bound in as a directory, so the CLI's OAuth session is shared | `claude` in there asks you to log in every run |

So the whole PR flow — commit, push, `gh pr create` — runs **inside** the
container, and that is what makes `claude` in there able to ship its own work.
All three are non-fatal on their own: a container missing any of them still
builds and tests, it just cannot commit, push, or stay logged in. `doctor.sh`
inside the container reports all three.

The identity resolution is deliberately git's own (`git var`), not
`git config --get user.name`: git falls back to `/etc/passwd`'s GECOS field, so
a host with only `user.email` set commits fine and `git config --get` would
report it as broken. The container cannot do that fallback for itself — its
`ubuntu` user's GECOS is `Ubuntu`, which resolves a *name* and no email.

### Three variants, one toolchain

`DEVCONTAINER_CONFIG` in `devtools/config.sh` picks which `devcontainer.json`
every subcommand drives. A relative path resolves against the repo root, so it
works from any cwd and any worktree. Each drives its own Dockerfile under
`docker/`, and all three share `docker/Dockerfile.base` as the parent:

| Config | Dockerfile | Toolchain |
|---|---|---|
| `.devcontainer/cuda/` | `Dockerfile.cuda` | clang-20 + libc++, CMake 4.2, Ninja, CUDA 13, sccache. Needs an NVIDIA GPU + the container toolkit. **The default.** |
| `.devcontainer/hip/` | `Dockerfile.hip` | the same toolchain with ROCm and no CUDA. Needs an AMD card. |
| `.devcontainer/combined/` | `Dockerfile.combined` | both SDKs, ~40GB. Needs an NVIDIA card **and** an AMD one — the only variant that reaches both. |

**Which cards a variant can actually see** is a separate thing from which SDK
it ships, and it is set by `runArgs`, not by the image:

| Config | NVIDIA | AMD |
|---|---|---|
| `cuda` | `--gpus all` | — |
| `hip` | — | `--device /dev/kfd --device /dev/dri` + numeric host `render`/`video` gids |
| `combined` | `--gpus all` | the same three AMD args |

The two mechanisms are unrelated, which is why neither implies the other.
`--gpus all` puts **nothing** in `HostConfig.Devices` — it records a
`DeviceRequest` that the nvidia-container-toolkit's OCI hook honours at
container *start*. The AMD side has no hook at all: the two device nodes are
passed straight through and keep their **host** ownership, so the
`--group-add` values must be numeric **host** gids.
`devtools/devcontainer.sh` resolves those from `ROCM_GROUPS` in
`devtools/config.sh` and exports `NEVPT2_RENDER_GID` / `NEVPT2_VIDEO_GID`;
the literals baked into the two json files are only the fallback for a path
that does not run it (a bare `devcontainer up`, VS Code "Reopen in
Container"). A group *name* there would resolve inside the container, where
`render` is a different gid, and the first device call would fail with
`hipErrorNoDevice` — indistinguishable from a box with no AMD card.

A build still targets exactly one backend; what `combined` buys is that
switching costs an environment variable instead of a container:

```bash
devtools/devcontainer.sh --combined shell          # both cards reachable
# inside, after the CUDA tier:
CMAKE_PRESET=hip CTEST_PRESET=hip devtools/cpp-tier.sh
```

Each preset lands in its own build volume (`build/` and `build-hip/`), so the
two do not reconfigure each other.

All three inherit the **CPU reference LAPACK** from `Dockerfile.base`
(no LAPACK: this project's oracle is the committed golden files, not a CPU
reference BLAS). If a container predates a Dockerfile change — `rebuild`, do
not `apt-get install` inside it, or the next rebuild loses the fix and nothing
says so.

Neither variant has anything prebuilt for the C++ side -- but there is also
nothing to fetch at build time (no GoogleTest; the `deps/WarpWraps` submodule
is checked out with the repo and not yet built against), so **the first build
in a fresh container is long and the shared `sccache` store is what makes the
second one short.** That store is a bind mount, not a named volume, and it is
the PRIMARY checkout's `.sccache` for every worktree —
`devtools/devcontainer.sh up`/`rebuild` resolves it and creates it. It
survives a `rebuild`, since nothing in the container's own filesystem holds it.

What it shares is the **store**, not the **hits**: the devcontainer mounts each
worktree at its own host path, and a restored BMI keeps the absolute path it was
built under, so main and `.claude/worktrees/<name>` do not hit each other's
entries. `docker/compose.yaml` mounts every service at `/workspace` instead, so
hits DO cross worktrees there. Neither path sets `SCCACHE_BASEDIRS` — see
`CMakeLists.txt`'s "Compiler cache" section for why that would break the build
rather than widen the cache.

Each config carries an **`initializeCommand` that builds its parent image**
(`docker/build.sh base`, or `cuda` for the combined variant) before the
devcontainer's own `docker build` runs — the files chain by tag, and a
devcontainer build cannot produce a parent on its own. It runs on every `up` and
`rebuild`, so a `Dockerfile.base` edit propagates with no extra step.

**The CUDA one is the default**, because it is where the project actually
builds. Override for one call with a variant flag, or for a whole session with
the variable:

```bash
devtools/devcontainer.sh --hip shell
export DEVCONTAINER_CONFIG=.devcontainer/hip/devcontainer.json
```

`doctor.sh` prints which config is active; a path that does not exist is a FAIL,
and every container command refuses until it is fixed.

### Bumping the Claude Code in the image

The agent is **pinned** (`ARG CLAUDE_CODE_VERSION` in each GPU `Dockerfile`,
installed by `docker/install-claude-code.sh` as the last layer), not installed
by a devcontainer feature. So it moves only when you move it:

```bash
devtools/claude-version.sh                 # pinned / registry / installed here,
                                           # + running containers on a pre-bump image
devtools/claude-version.sh --apply         # rewrite all three to the registry's latest
devtools/claude-version.sh --apply 2.1.300 # or to one you name
devtools/claude-version.sh --rebuild       # ...and rebuild the images already built here
```

Then `devtools/devcontainer.sh rebuild` for whichever variant you work in — the
pin is in the image, so a `rebuild` without a re-`docker build` would recreate
the container from the same image and change nothing, and the converse holds
too: a rebuilt image does not reach a container that is already running.
`claude-version.sh` (bare, or after `--rebuild`) names every running container
still on a pre-bump image, with the `rebuild` that recreates it — it never
recreates one itself. `up` and `rebuild` print
a one-line warning when the pin is behind the registry or the three files
disagree, and `doctor.sh` says the same thing on demand.

**Do not `npm install -g @anthropic-ai/claude-code` inside a container.** It is
lost on the next rebuild, and `doctor.sh` reports it as *"claude on PATH is A,
this image pinned B"* — the same class of mistake as `apt-get install`ing the
LAPACK headers in there.

### Bounding a container's CPU

`CPUSET` (host cores) and `CPUS` (a core-count quota) come from
`devtools/config.sh`, or from the environment for one call, and are applied by
`up` and `rebuild` with `docker update`:

```bash
CPUSET=0-11 devtools/devcontainer.sh up      # pin to host cores 0-11
CPUS=8      devtools/devcontainer.sh up      # cap at 8 cores' worth
CPUSET=0-11 devtools/devcontainer.sh up      # re-pin one already running
```

This is **not** `JOBS`, and not `BUILD_JOBS`. Those are two runners' worker
counts — pytest's and `cmake --build`'s — and neither reaches anything else, so
a compile started from a `shell` by hand still takes the whole box.
`CPUSET`/`CPUS` bound everything in the container.

`BUILD_JOBS` is less critical here than it would be on a larger tree, but still worth knowing: a C++23 module build is memory-hungry per
job (the scanner plus a BMI cache), so the right value is usually *lower* than
the core count and lower than what you would give pytest. Empty lets Ninja pick
cores + 2, which is what OOM-kills a 16-core box on a module-heavy tree — and an
OOM-killed compiler reads as a mysterious `ninja: build stopped`, not as a
memory problem.

Reach for it when two worktree containers are up at once: pinned to disjoint
cores (`CPUSET=0-11` in one, `CPUSET=12-23` in the other) they physically
cannot contend, which is the only way a timing measurement in one of them means
anything. Limits belong to the **container**, and a container is per workspace
folder — so pin per worktree, not per shell; two `shell`s from the same
checkout share one container and one set of limits.

Three things to know:

- **On the CUDA container this costs a restart, deliberately.** `docker update`
  regenerates the device cgroup from `HostConfig.Devices`, which for `--gpus
  all` is **empty** — the GPU ask lives in `DeviceRequests`, honoured by the
  NVIDIA runtime's OCI hook only at container *start*. So a bare `docker update`
  silently revokes GPU access (`nvidia-smi: Failed to initialize NVML`, and on
  this repo a configure-time failure, since `CMAKE_CUDA_ARCHITECTURES=native`
  queries the device). `devcontainer.sh` restarts to re-run the hook; the limits
  persist across it because they live in the container config. The HIP variant
  is untouched — `--device=/dev/kfd` populates `HostConfig.Devices`, which
  `docker update` preserves — so the restart is gated on `DeviceRequests`.
- `docker update` changes only what it is passed, so a second call with just
  `CPUSET` leaves an earlier `CPUS` quota in place, and running `up` with both
  unset **clears nothing** — it only stops applying. `rebuild` is how you get
  an unbounded container back; a fresh one starts with no limits.
- If nothing is running, the limits are skipped with a `cpu-limit: no running
  container …` line on stderr rather than an error — `up` had already done its
  work.

### Killing a container test run on the host does not kill it inside

`npx` dies; the pytest master and its xdist workers keep running at 100% CPU,
and nothing says so — the next run just comes out slower and you measure *that*.
After interrupting:

```bash
devtools/devcontainer.sh shell -c "ps -eo pid,pcpu,cmd --sort=-pcpu | head"
```

`pkill -f pytest` reaches only the master (a worker's command line is a bare
`python3 -u -c import sys;...`), so kill the workers **by PID**.

## Container-backed worktrees

Sibling directories under the repo root, named after their branch —
`<root>/main`, `<root>/<name>` — each with a container, three named volumes and
a build image.

```bash
devtools/worktree.sh list
devtools/worktree.sh add <name> [start-point] [--up]
devtools/worktree.sh rm  <name> [--force]
devtools/worktree.sh sync
devtools/worktree.sh gc [--dry-run] [--yes]
```

**`add`** creates or attaches branch `<name>` at `<root>/<name>`; `--up` also
brings the container up and drops you into a shell. It forks from the calling
checkout's **local HEAD**, and warns (offline, from the last fetch) when that
HEAD is behind its upstream — heed it, or the new branch starts on stale code.
It refuses the name `main`.

If it fails with *"$ROOT is not writable"*: the repo root, not the checkout,
needs to be writable. Fix once with the **numeric** uid —

```bash
sudo chown $(id -u) <root>
```

— because the host user and the container's `ubuntu` are typically both uid
1000 under different names, so `chown ubuntu` fails on the host.

**`rm`** is the one that matters. `git worktree remove` knows nothing about the
container, its `<PROJECT_NAME>-{sccache,build,build-hip,uvcache}-<devcontainerId>`
volumes or the `vsc-<name>-<64hex>-uid` image; `rm` tears down all of it, then
removes the worktree and deletes the branch — with `-d`, so an unmerged branch
is left in place and reported rather than lost. It refuses to remove the
checkout you are calling it from: run it from another one (typically main).

**`sync`** fast-forwards `main` to `origin/main`. Nothing else does this: after
a PR merges `origin/main` moves and local `main` does not, and `add` forks from
local HEAD. It finds whichever checkout has `main` on it (so it works from any
worktree, including a `.claude/worktrees/` one) and is `--ff-only` — a diverged
`main` is refused rather than turned into a merge commit.

**`gc`** reconciles docker against the worktrees that actually exist, for
everything an earlier plain `git worktree remove` leaked. Always show
`--dry-run` first; it prints a `orphans: N (…)` line and deletes nothing.
It is scoped to this repo — volumes by `PROJECT_NAME` prefix, images by the
`devcontainer.project` label, containers by folders under this repo's root or
its `.claude/worktrees/` — so it is safe beside other docker workloads. An
image with no label is never touched.

`doctor.sh` runs `gc --dry-run` for its "docker hygiene" line, which is how
leaked state surfaces without anyone going looking.

## The other front end: docker compose

`docker/compose.yaml` runs **the same `docker/Dockerfile.cuda` image** as one-shot
batch jobs — configure, build, test, exit — teeing everything to `.log/`. It is
not layered on the devcontainer and neither is deprecated:

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)     # required; compose.yaml uses :? not a default
alias dc='docker compose -f docker/compose.yaml run --rm'
dc build ; dc test ; dc asan ; dc compute-sanitizer
```

Use compose when you want to read a result and throw the container away — CI,
nightlies, a sanitizer sweep, a clean reproducible run. Use the devcontainer
when you want to *stay inside*: iterating, agents, debugging. Compose services
are global (not per worktree) and honour no `CPUSET`/`CPUS`, so two concurrent
compose runs will contend. `docker/README.md` has the full variable list.

The image is built by `docker/build.sh`, which walks the tag chain
(`Dockerfile.base`, then `Dockerfile.cuda`) and tags the result both
`<project>:cuda` and `<project>:latest` — the second being what `NEVPT2_IMAGE`
defaults to:

```bash
docker/build.sh cuda
```

Run it from anywhere; the build context is always the repo root, because the
Dockerfiles read `pyproject.toml`, `uv.lock` and the install scripts relative to
it. Do **not** hand-roll `docker build -f docker/Dockerfile.cuda` — with no
parent tagged it tries to *pull* `<project>:base` and fails with `pull access
denied`, which reads like a registry problem.

## When NOT to use this skill

For a quick isolated checkout, a spike, or agent scratch space, use the
lightweight `.claude/worktrees/<name>` layout and the **`worktree`** skill — no
container, no volumes, no image, and cheap to throw away. Reach for the
container-backed ones only when the work needs the image: the full suite, a
toolchain that exists only inside, or a GPU.
