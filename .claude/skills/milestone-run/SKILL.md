---
name: milestone-run
description: >-
  Complete a GitHub milestone end to end by orchestrating subagents: read the
  milestone's issue DAG, fan out one subagent per ready issue (each in its own
  .claude/worktrees/ worktree, building and testing on a GPU card), merge
  their PRs one at a time, and launch newly unblocked issues until the
  milestone is done. Use when the user invokes `/milestone-run <milestone
  URL|number> [max=N] [both] [dry-run]` or asks to "complete / execute / ship
  this milestone with subagents". Invoking it is the user's merge grant for
  this milestone's PRs. Requires gh + GH_TOKEN. Not for creating a milestone —
  that is `milestone`.
---

# `/milestone-run <milestone> [max=N] [both] [dry-run]` — milestone DAG → merged PRs

`milestone` turns a plan into a DAG of PR-sized issues; this skill **executes**
one. You are the **orchestrator**: you schedule, create worktrees, review,
re-test, merge and tear down. Subagents implement one issue each, ship a PR
whose body says what ran on which card, and **hand it back without merging**.

**There is no CI here, on purpose** (CLAUDE.md, "There is no CI, on purpose").
`gh pr checks` prints "no checks reported" and that is correct. Nothing on
GitHub can tell you a PR is green: the only evidence is what an agent — or
you — ran on a card and wrote down. So the merge gate in step 5 is the PR
body's *What was run* list plus your own re-run on the rebased tree, never a
check status.

**Invoking `/milestone-run` is the merge grant** — for PRs that close an issue
in *this* milestone, opened by *this* run's subagents, once the gates in step 5
pass. It does not cover any other PR, `--admin`, or merging a PR whose run
failed or was never run. If the auto-mode classifier still blocks a merge, stop
merging and report the ready PR links; do not work around the block.

## Arguments

- **Milestone** — a URL (`…/milestone/<N>`), `#N`, or a bare number. The
  number is all you need.
- **`max=N`** — the most subagents in flight at once. Default **3**. Every
  agent builds its own tree (`build/`, `build-hip/`, … are per worktree) and
  runs ctest on a shared card, so the box, not GitHub, is the limit: past ~4,
  warn the user that builds will thrash (module builds are memory-hungry per
  job and OOM rather than fail cleanly) and GPU runs will queue, then honour
  the number they gave.
- **`both`** — require every PR to pass ctest on **both** cards (CUDA and HIP).
  Without it, **one card suffices**: each agent runs ctest on the card it is
  assigned and `devtools/cross-backend-check.sh` proves the other backend
  *compiles* — the same split CLAUDE.md asks of any PR. Ask for `both` when the
  milestone touches backend-sensitive code — device kernels, the BLAS digest
  (whose default differs by backend), `DeviceResources`/streams, the
  WarpWraps pin — because a compile proves nothing numerical.
- **`dry-run`** — do steps 1–2, print the schedule, and stop. No worktree, no
  agent, no `gh` write.

## 1. Read the milestone and rebuild the DAG

```bash
gh api repos/{owner}/{repo}/milestones/<N> --jq '{title,state,open_issues,closed_issues}'
gh issue list --milestone "<title>" --state all --limit 200 \
  --json number,title,state,body
```

`--milestone` wants the **title**, not the number. For each issue, take its
blockers from the native dependency API, falling back to the `## Depends on`
lines that `milestone` writes into every body:

```bash
gh api repos/{owner}/{repo}/issues/<n>/dependencies/blocked_by --jq '.[].number'
```

Then classify every issue:

- **done** — closed.
- **ready** — open, every blocker closed, and no open PR already references it
  (`gh pr list --search "<n> in:body" --state open --json number,headRefName`).
- **resumable** — open, with an open PR whose head branch is `issue-<n>-*`
  *and* a local `.claude/worktrees/issue-<n>-*` checkout. An earlier run of
  this skill died or was compacted mid-flight: adopt it. A PR whose body
  records a passing run goes straight to step 5. Otherwise give a fresh agent
  the existing worktree and PR ("finish PR #M") instead of a new branch. An
  open PR that fails either test is someone else's: leave it alone and report
  it.
- **blocked** — some blocker still open. A blocker *outside* the milestone that
  is open is a hard stop for that branch: report it, don't work around it.

A cycle, or an issue with no `Acceptance` section, means the milestone is
malformed: stop and say so rather than guess an order.

## 2. Show the schedule, then start

Print the wave table (issue, title, blockers, state), the ready set, `max`, and
whether `both` is on. On `dry-run`, stop here. Otherwise start without waiting
for a reply — the invocation was the go-ahead. Stop and ask instead only when
step 1 found something odd (cycle, external open blocker, an issue that looks
bigger than one PR, an issue whose acceptance would mean changing a golden
file or the 1e-7 tolerance).

## 3. Pre-flight, once, on main

```bash
git fetch origin main --quiet && devtools/worktree.sh sync
nvidia-smi -L                                     # CUDA card alive?
rocminfo | grep -m1 gfx                           # HIP card alive?
devtools/cpp-tier.sh --no-test                    # warm sccache: CUDA
devtools/cpp-tier.sh --preset hip --no-test       # warm sccache: HIP
```

Both cards are reachable only from the `combined` devcontainer; in any other
container one of the two probes fails, and that is the "one dead" case below.
Several cold WarpWraps builds at once will OOM this box, so a warm sccache is
what makes `max` agents affordable. Warm only the backends you will use (each
agent's `cross-backend-check.sh` also builds the `ci-*` preset of the other
backend, so under the default both caches get used). The live cards decide
the assignment pool: both alive → alternate; one dead → every agent gets the
live one (and say so); both dead, or one dead under `both` → no agent can
pass its gate, so don't launch. Report it and stop.

## 4. Dispatch loop

While any issue is ready or in flight:

1. **Launch** `min(max − in_flight, |ready|)` ready issues. For each, **you**
   create the worktree, one at a time, so parallel agents never race on
   `git fetch` or the ref lock:

   ```bash
   git fetch origin main --quiet
   git worktree add .claude/worktrees/<wt> -b <wt> origin/main
   git branch --unset-upstream <wt>   # else it tracks origin/main, and a bare push targets main
   git -C .claude/worktrees/<wt> submodule update --init --recursive
   ```

   `<wt>` is `issue-<n>-<slug>`, the slug a few lowercase, hyphenated words of
   the title (`issue-81-tile-autosize`). Don't use the Agent tool's
   `isolation: "worktree"`: it lives elsewhere and tears itself down, which
   removes the checkout you need for the re-test and rebase in step 5. Also
   hand each agent:

   - **a card** (unless `both`): alternate CUDA / HIP across launches so each
     card carries about half the test load. The agent falls back to the other
     card if its own is down.
   - **an insertion anchor**, if the issue adds a component. `src/CMakeLists.txt`
     lists components as `add_subdirectory(...)` lines in one block, so N
     agents all appending after the last one conflict on every merge. Give each
     in-flight agent a *different*, non-adjacent existing
     `add_subdirectory(<X>)` line to insert after; sequential 3-way merges of
     non-adjacent hunks stay clean. The root `CMakeLists.txt` registers ctest
     entries in grouped blocks (per demo, then `small`, then canaries); give a
     new test entry the same treatment — a different existing `add_test` block
     to follow.
   - **a job count** `J = $(nproc) / max` (24 cores, `max=3` → `-j 8`).
     `cpp-tier.sh` otherwise lets every agent take the whole box.

   Then spawn a background `general-purpose` subagent with the prompt in
   "Subagent prompt" below. Record `issue → {wt, agent id, card, anchor}`.

2. **Wait for notifications**; never poll on a short timer. When an agent
   reports:
   - **"PR #M ready"** → step 5. Merges are serial: if one is in progress,
     queue this one.
   - **stuck** (ambiguous issue, a ctest entry failed, passing would need a
     tolerance loosened, a golden regenerated or a tile count lowered, scope
     bigger than one PR) → don't merge. Mark the issue **stuck**. Its
     dependents stay blocked. Keep its worktree. Carry on with every other
     branch of the DAG.

3. **Watch for stragglers.** If an agent runs long, look at its worktree
   yourself (`git -C <wt> log origin/main..`, `gh pr list --head <wt>`). An
   agent "waiting for checks" is waiting for something that will never come —
   there are no checks; nudge it with `SendMessage`. Use `TaskStop` on one
   that has already handed back.

**Status line.** After every launch, merge, or stuck report, print one line —
`merged #81 → PR #90 (3/9 done) · launched #85 [hip] · in flight: #82 #84 ·
stuck: none` — so the user can follow a long run without reading the
transcript.

## 5. Review, re-test, merge — one PR at a time

Merge **sequentially**, never in parallel and never with `--auto`. With no
required checks, `--auto` merges the instant the PR is mergeable — i.e. on
nothing. Before each merge:

```bash
gh pr view <M> --json mergeable,body,files,baseRefOid
git -C <wt> diff --stat origin/main...HEAD
```

- `mergeable` is `MERGEABLE`;
- the body has `Closes #<n>` and, under *What was run*, the preset, card and
  ctest pass count (e.g. `--preset default`, RTX 3080, 22/22) — at least one
  card, or **both** under `both` — plus `cross-backend-check.sh` PASS. A
  compile-only run is not a test. A ticked box with no numbers next to it is
  not evidence;
- no ratchet moved: nothing under `golden/` changed unless the issue's scope
  says so, `kAtol` (1e-7, `apps/*/main.cppm`) is untouched, no tile count in
  a ctest entry was lowered, no sanitizer canary or `nevpt2_lsan.supp` entry
  naming one of our frames was added or removed, and `stream-lint.sh` was not
  weakened. Any of these is a human's call — mark the issue stuck instead;
- if the diff mentions `--cublas`, the body quotes the `engaged bits=` line.

**Re-test on the tree you are about to merge.** The agent tested on the `main`
its worktree forked from, and every merge since has moved `main`. With no CI,
nothing else will ever run the combination. So if `origin/main` has moved past
the PR's base, rebase and re-run before merging:

```bash
git -C <wt> fetch origin main --quiet
git -C <wt> rebase origin/main
# from <wt>, in the background (Bash run_in_background) so the loop keeps going:
flock /tmp/nevpt2-gpu-<card>.lock devtools/cpp-tier.sh <preset> -j <J> -- -LE slow
devtools/cross-backend-check.sh -j <J>
git -C <wt> push --force-with-lease
```

`<preset>`/`<card>` are the agent's (both pairs under `both`). `-LE slow`
drops the CAS(12,12) entries: the agent's full run already covered them, and
this re-run is for what the rebase could have broken.
Append the re-run's preset, card and pass count to the PR body's *What was
run* (`gh pr edit <M> --body-file …`), marked as the post-rebase re-run. If
the PR's base already *is* `origin/main` — the first merge of a wave — the
agent's run is the run and needs no repeat.

**Conflicts** during that rebase, two cases:

- **Only `add_subdirectory` / `add_test` blocks** in a `CMakeLists.txt` →
  resolve it yourself, keeping both sides; the re-run above is the check.
  `git -C <wt> add <file>` and `git -C <wt> rebase --continue`.
- **Anything else** (a `.cppm`, a `.cu`, `main.cppm`, a golden) →
  `git -C <wt> rebase --abort` and send it back to the agent that wrote the PR
  (`SendMessage` resumes it with its context): rebase, resolve, re-run its
  suites, push, re-report ready.

A re-run that fails after a clean rebase is a semantic conflict between two
PRs that each passed alone: do not merge, send it back to the agent with the
failing ctest output, and say so in the status line.

Then `gh pr merge <M> --squash` — never `--admin`, never `--delete-branch`
(the `pr` skill says why; the remote branch is auto-deleted on merge).

**After each merge:**

- tear the worktree down. It holds the submodule, so plain `git worktree
  remove` refuses; per the `worktree` skill, check it is clean first, then
  `--force`:

  ```bash
  git -C .claude/worktrees/<wt> status --porcelain   # must print NOTHING
  git worktree remove --force .claude/worktrees/<wt>
  git branch -D <wt>                                 # squash-merged: -d refuses
  ```

  Any output from `status` means stop and report it, not `--force` past it;
- confirm the issue closed (`gh issue view <n> --json state`);
- print the status line, recompute **ready**, and go back to step 4.1.

## 6. Finish

When nothing is ready or in flight:

- `devtools/worktree.sh sync`, so local `main` has every merge;
- if every issue is closed, close the milestone
  (`gh api -X PATCH repos/{owner}/{repo}/milestones/<N> -f state=closed`);
- report one line per issue: `#n → PR #M merged [cuda|hip|both, n/n]`, or
  **stuck** with its reason, or **blocked** naming its blocker. Spend words
  only on what deviated: blocked merges, rebases, post-rebase failures, stuck
  issues, card fallbacks, decisions made for the user.

Leave the worktrees of stuck issues standing, and say where they are.

## Subagent prompt

Fill in `<…>`. Keep it self-contained — the agent has none of your context.
`<preset>` is `--preset default` for CUDA and `--preset hip` for HIP; `<card>`
is `cuda` or `hip`. Under `both`, give both command pairs and drop the
fallback line.

```
Resolve GitHub issue #<n> (<title>) in this repo. Read it first:
`gh issue view <n>` — its Scope, Acceptance and Files / area sections are the
spec. Your worktree already exists at <abs path to .claude/worktrees/<wt>>,
on branch <wt>, forked from origin/main with deps/WarpWraps initialised —
EnterWorktree {"path": …} into it and do not create another. Read
.claude/CLAUDE.md before you write code: it holds this repo's rules (one
non-blocking stream, DeviceBuffer only, wwr* names only, no CI, tile counts).

Implement the issue, then ship it with .claude/skills/pr/SKILL.md steps 1-4,
with these overrides:
- Do NOT merge and do NOT tear down the worktree. Your job ends at an open PR
  whose body records a passing run. Then reply exactly: "PR #<M> ready" + one
  line naming the card, preset and ctest pass count.
- There is no CI. `gh pr checks` saying "no checks reported" is correct; do
  not wait for anything.
- Test on ONE card, your assigned <card>; cross-backend-check.sh covers the
  other backend's compile. Siblings share the card, so build unlocked and take
  the card's lock only to test:
    devtools/cpp-tier.sh <preset> -j <J> --no-test
    flock /tmp/nevpt2-gpu-<card>.lock devtools/cpp-tier.sh <preset> -j <J>
    devtools/cross-backend-check.sh -j <J>
  cpp-tier runs devtools/stream-lint.sh --strict first; a finding is a
  failure. If the change touches device allocation, streams or kernels, also
  run the sanitizer presets CLAUDE.md lists for your card ("The sanitizer
  tier") under the same lock. If your card is down (nvidia-smi -L / rocminfo),
  use the other one and say so in the PR. An out-of-memory error on the card
  means a sibling's run overlapped yours: re-run under the lock before calling
  it a failure.
- If a ctest entry fails, or passing would mean loosening the 1e-7 tolerance,
  regenerating a golden, lowering a --tiles count, or deleting a canary, stop
  and reply "STUCK: <reason>" — no PR needed.
- Commit and push with `git -C <abs worktree path> …` (protect-main
  false-positive). Push with `git -C … push -u origin <wt>` — the branch has
  no upstream on purpose. Never --no-verify, never CLAUDE_ALLOW_MAIN_EDITS.
- PR body via `--body-file` in your scratchpad (there is no PR template), with
  `Closes #<n>`, and under "What was run" the exact commands, preset, card,
  ctest pass count (n/n) and cross-backend-check result. Only list what you
  ran and read the output of.
- If you add a component, add its add_subdirectory line in src/CMakeLists.txt
  right after `add_subdirectory(<anchor>)`, NOT after the last one — siblings
  are adding components in parallel. A new ctest entry in the root
  CMakeLists.txt goes after <test anchor>. If asked to rebase, resolve a
  CMakeLists.txt conflict by keeping BOTH sides.
- If the issue is ambiguous in a way the code can't answer, reply
  "STUCK: <question>" rather than guessing.
```
