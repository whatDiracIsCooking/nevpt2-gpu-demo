---
name: doctor
description: >-
  Diagnose a degraded environment: run devtools/doctor.sh and turn each
  finding into the fix, plus the failures it cannot see (orphaned pytest
  workers, a container that ignored your config edit). Use when something
  behaves oddly — commits fail with "No module named pre_commit", tests skip or
  vanish, git says "not a repository" inside a container, a worktree command
  cannot find its docker state — or when the user just asks what is wrong here.
---

# Reading a degraded environment

```bash
devtools/doctor.sh
```

It exits non-zero **only** on a FAIL (the core suite cannot run). A WARN means
some slice of the suite will skip, which is often the correct state of a
machine. So the output is not pass/fail: it is a list of what this environment
cannot do, and the job is to say which of those matter for the task at hand.

Run it **first** when anything behaves oddly, and again after any fix.

## Run it on both sides, and compare

This repo has a host half and a container half, and doctor is the thing that
tells you which one you are on:

```bash
devtools/doctor.sh                                    # the host
devtools/devcontainer.sh shell -c devtools/doctor.sh  # the CUDA container
```

On a bare host, **around a dozen warnings is the expected, healthy state**.
What is genuinely container-only is the clang/GPU toolchain -- `cmake`,
`ninja`, `clang++`, `clang-scan-deps`, `clang-format`, `nvcc`,
`compute-sanitizer`, `nsys`, `sccache` -- plus `hipconfig`, which is present
only
in the images built from `docker/Dockerfile.hip` and
`docker/Dockerfile.combined`, not the `cuda` one, and
the golden reference files, which are this project's oracle, and which
`docker/Dockerfile.base` installs and a bare host has no reason to.

Note what is NOT on that list any more: `cmake-format` and `cmake-lint` come
from the `cmakelang[yaml]` dev dependency, and doctor resolves tools through
`VENV_PATHS` (`find_tool` in `devtools/lib.sh`) rather than `PATH`. On the host
they report `[ ok ]` with a `not on PATH; using .../.venv/bin/...` note, which
is normal rather than a finding. That is not a broken machine; it is the
laptop side of a two-sided workflow, good for editing, linting and the Python
suite. The same list appearing *inside* the container is a real problem.

Never report the host list on its own as "the environment is degraded." Say
which side you ran on, and what the other side would cover.

## Finding → what it means → fix

**`[FAIL] not inside a git repository`** — nothing else can be checked. Usually
means a container without the `.git` bind mount: a worktree's `.git` is a file
pointing outside the workspace folder, so `.devcontainer/<variant>/devcontainer.json`
mounts the main checkout's `.git` common dir inside. `devtools/devcontainer.sh`
injects that host path as `NEVPT2_GIT_DIR` on `up`/`rebuild`, so the usual cause
is a container brought up another way (a direct `devcontainer up`, or VS Code
"Reopen in Container" with the fallback path unedited). Bring it up with
`devtools/devcontainer.sh rebuild`; `up` will not apply the mount change.

**`[warn] <file> names volumes that do not start with '<name>-'`** — a
half-applied rename. `PROJECT_NAME` is spelled out in `devtools/config.sh` *and*
in **every** `devcontainer.json` file (JSON cannot source shell), and while they
disagree `worktree.sh rm` and `gc` do not recognise this project's volumes, so
every worktree you tear down leaks its whole set. Each file is checked
separately, so the warning names which one drifted. The fix is to make the odd
one out agree with `PROJECT_NAME` in `devtools/config.sh`.

**`[warn] cannot resolve this repo's git dir for the container .git mount`** —
`doctor` ran somewhere `git rev-parse --path-format=absolute --git-common-dir`
returns nothing, so the path `devcontainer.sh` would inject as `NEVPT2_GIT_DIR`
is empty and the container's `.git` mount would fail. Run doctor from inside the
nevpt2-gpu-demo checkout.

**`[FAIL] DEVCONTAINER_CONFIG does not exist`** — `devtools/config.sh` points at
a `devcontainer.json` that is not there, and *every* `devcontainer.sh` command
refuses until it is fixed. The shipped values are
`.devcontainer/cuda/devcontainer.json` (the default), `.devcontainer/hip/devcontainer.json`
and `.devcontainer/combined/devcontainer.json`.

**`[warn] worktree root not writable` / `primary checkout owned by uid N`**
— `worktree.sh add` cannot create sibling checkouts, and git ops in main fail.
Fix once with the numeric uid: `sudo chown -R $(id -u) <root>`. Not `chown
ubuntu` — the host user and the container's `ubuntu` are usually both uid 1000
under different names, so the name form fails on the host.

**`[warn] N broken/prunable linked worktree(s)`** — a worktree directory was
deleted out from under git. `git worktree prune` clears the bookkeeping; then
check `devtools/worktree.sh gc --dry-run`, because a hand-deleted worktree
usually left its container and volumes behind too.

**`[warn] no venv found` / `pytest not found` / `uv not found`** — `uv sync`.
`uv` is the only assumed host tool; the venv is built from the tracked
`uv.lock`, never with `uv pip install`.

**`[warn] uv.lock does not name the project '<name>' -- it is stale`** — the
lock and `pyproject.toml` disagree, usually after a dependency edit or a
half-finished rename. `uv sync --frozen` is what both Dockerfiles run, so
this surfaces as a failed **image build** a long way from the cause. Fix with
`uv lock` and commit the result; do not hand-edit the lock, and do not "work
around it" by dropping `--frozen`, which is the line that keeps the container
and the host on the same versions.

**`[warn] <tool> not found -- lost: <what>`** — an optional tool from
`DOCTOR_OPTIONAL_TOOLS` in `devtools/config.sh`. Install it *or* accept that its
slice of the suite will skip — but say which, because that is exactly the
coverage a green run will not have. This list is the project's to maintain: when
a test starts depending on a tool, add it here.

**`[warn] gh cannot see <owner>/<repo>`** — the token authenticates but was
never granted THIS repository. A fine-grained PAT names its repositories one by
one, and `gh auth status` reports a clean login either way, so without this
check the first symptom is `gh pr create` failing with "Could not resolve to a
Repository" *after* the branch is already pushed. Fix it under Repository
access on the token, not by re-authenticating. Blocks the `pr` and `milestone`
skills entirely.

**`[warn] gh present but GH_TOKEN empty`** — `gh` is installed but
unauthenticated, so the PR flow fails at push time rather than at the start.
Export a fine-grained PAT (Contents + Pull requests: read/write) on the **host**
and `rebuild`: the container takes `GH_TOKEN` from the host env at create time,
so an `export` after the container is up does not reach it.

**`[warn] no git identity`** — `git commit` will fail with "Author identity
unknown", and it fails *after* the change is staged, so it is the last thing in
the PR flow to break. Fix it on the **host** (`git config --global user.name` /
`user.email`) and then `rebuild`: `devtools/devcontainer.sh` resolves your
identity with `git var GIT_AUTHOR_IDENT` at `up`/`rebuild` time and passes it in
as `NEVPT2_GIT_USER_NAME`/`_EMAIL`, which
`.devcontainer/git-identity.sh` writes into the container's own `~/.gitconfig`
from `postCreateCommand`. A `git config` run **inside** the container works
until the next `rebuild` discards it, so it is not the fix.

Two things make this specific to the container rather than to you:
`git var` is used instead of `git config --get user.name` because git falls
back to the GECOS field of `/etc/passwd` — a host with only `user.email` set
commits perfectly well, and `git config --get` would call it broken. The
image's `ubuntu` user has the GECOS field `Ubuntu`, so git in there resolves a
*name* and then fails on the email
(`unable to auto-detect email address (got 'ubuntu@<id>.(none)')`) — which is
why both halves are carried in, and why passing only the email would silently
attribute your commits to "Ubuntu".

**`[warn] <path> absent`** — a `DOCTOR_REQUIRED_PATHS` entry. There are two,
and they fail very differently:

- **`golden/n2_ccpvdz_cas1010.nevpt2gold`** and
  **`golden/n2_ccpvdz_cas1212.nevpt2gold`** — this project's ORACLE. They are
  committed, so a fresh clone has them; if one is missing, the root
  `CMakeLists.txt` emits a WARNING and simply does not register that ctest
  entry. That is the failure worth knowing about: it turns "the data is
  gone" into `0 tests ran` and a green exit, not a red one. Always read
  ctest's test COUNT, not just its colour.
- **`golden/*cas1414*`** — deliberately absent. 71 MB xz, gitignored; the
  CAS(14,14) case is regenerated on demand with `generate_golden.py`, never
  committed. Its absence is correct and doctor does not check for it.


6. **A dependency that is present but stale.** A golden file existing satisfies
   `DOCTOR_REQUIRED_PATHS`; it says nothing about which commit is checked out.
   `git submodule status` (a leading `+` means the checkout differs from the
   gitlink this branch records) is the check doctor does not make.

## Reporting

Say what is degraded and what it costs, not just the counts: "no failures, 21
warnings — all the C++ toolchain, because this is the host and it lives in the
CUDA image; the Python suite runs here, the C++ tier does not." A bare "doctor
passes" throws away the entire point of the command, and a bare "21 warnings"
reads as alarming when it is the expected state.
