---
name: pr
description: >-
  Ship the current work: commit the relevant changes, push the branch, open a
  GitHub PR with gh, and merge it. Use when the user asks to "PR this", "ship
  it", "commit/push/merge", or otherwise wants the working-tree changes landed
  on main end to end. Requires gh + GH_TOKEN.
---

# Commit → push → PR → merge

Drive the current changes all the way to `main` in one pass. Each step has a
gotcha; follow them in order.

## 1. Commit the relevant changes

- **Never commit on `main`.** If the session is on `main`, create a branch
  first (`git switch -c <topic>`). If the session is in a `.claude/worktrees/`
  worktree, the branch already exists — use it.
- Stage **relevant** changes, not everything: read `git status` and `git diff`,
  stage what belongs to this task, and leave unrelated untracked files behind.
  No blanket `git add -A` without looking.
- The pre-commit hook lints every touched file — a finding blocks the commit;
  fix it, don't `--no-verify`.
- If `pyproject.toml` changed, run `uv lock` and commit the lock with it. The
  lock is the only place exact versions live, and the image installs it with
  `--frozen`, so a stale lock breaks the container rather than the host.

## 2. Check the repo's own invariants

Before pushing, look at what the diff touches and whether this repo guards it:

- a **ratchet** (a test pinning a measured number or a floor) may only move in
  the improving direction, in its own commit, saying what moved. A loosened
  threshold produces a green run — that is exactly why it needs a human.
- the **reference chain** (whatever independently checks correctness here) is
  not something to adjust to make a test pass.

A repo with neither has nothing to do in this step.

## 3. Push

```bash
git push -u origin <branch>
```

- A push touching any `.py` auto-runs the fast test gate
  (REMOVED in this repo -- there is no Python test suite and no pre-push gate,
so a push implies nothing; run `devtools/cpp-tier.sh` on a box with a card
yourself). Historically a successful push *implied* that
  gate passed. Do not bypass with `--no-verify`.
- If the worktree's git dir has gone missing (`fatal: not a git repository:
  …/worktrees/<name>`), push from the main checkout instead:
  `git -C <repo-root>/main push origin <branch>`.

## 4. Create the PR

```bash
gh pr create --title "<concise title>" --body-file <path>
```

**There is NO `.github/pull_request_template.md` in this repo** (there is no
`.github/` at all). So write the body yourself, into a scratch file, and pass
it with `--body-file` — and make it carry the thing a template would have
forced: **an explicit list of which tiers you actually ran**, ticked only where
you ran the command and read the output.

That list is not ceremony here, it is the ONLY record that anything was
checked: there is no CI (see CLAUDE.md, "There is no CI, on purpose"), every
test needs a real GPU, and nothing on GitHub can verify a single numerical
claim this project makes. A PR body that does not say what ran is a PR nobody
can review.

The harness's PR-body footer rules apply.

## 5. Merge

```bash
gh pr merge <number> --squash
# the remote branch is auto-deleted on merge (see below) -- nothing to run here
```

- **There is no CI to wait for.** `gh pr checks` will say "no checks reported"
  — that is the expected, correct output, not a misconfiguration. Nothing runs
  server-side, so the merge decision rests entirely on what you ran locally and
  wrote into the PR body. Run `devtools/cpp-tier.sh` on a box with a card, and
  `devtools/cross-backend-check.sh` for the other backend, BEFORE opening the
  PR — not after.
- **Never pass `--admin`.** It merges past required status checks, and it is the
  one move in this whole flow that can defeat the branch ruleset. Red checks
  mean fix them or hand the PR back — never override. A ruleset with an empty
  bypass list makes the server refuse it anyway, but do not lean on that: the
  bypass list is a setting someone can widen, and this rule is the intent.
- **Never pass `--delete-branch`**: it attempts a local `git checkout main`,
  which fails when `main` is held by another worktree. The remote branch is
  reclaimed automatically instead (next bullet).
- **Remote branch cleanup is automatic.** GitHub's "Automatically delete head
  branches" is ON for this repo (`gh api repos/{owner}/{repo} --jq
  .delete_branch_on_merge` → `true`). It is a per-repo setting, so if a merged
  ref is still there, re-check the setting rather than assuming. Confirm with
  `git ls-remote --heads origin <branch>` (empty output = gone). Do not delete
  a surviving remote branch unasked — the auto-mode classifier treats
  `git push origin --delete` as destructive, so that is the user's call. The
  *local* branch is still yours to delete in step 6.
- If GitHub reports the PR as not yet mergeable, wait a moment and re-check
  with `gh pr view <number> --json mergeable,mergeStateStatus` before retrying
  — do not force.

## 6. After the merge

`origin/main` has moved but local `main` has not — and `worktree.sh add` forks
from local HEAD, so a stale `main` silently seeds stale branches:

```bash
devtools/worktree.sh sync
```

It finds whichever checkout has `main` on it and fast-forwards it there, so it
runs from any worktree; it refuses (`--ff-only`) rather than merging if that
`main` has diverged.

If the work happened in a `.claude/worktrees/` worktree, whether to tear it
down depends on how this skill was invoked:

- **Bare `/pr`** — leave the worktree standing. The work is merged, but the
  checkout stays for follow-up commits; do not remove it unless asked.
- **`/pr full`** — reclaim it now, per the `worktree` skill's "`/pr full` —
  ship, then tear the worktree down" section: exit with
  `ExitWorktree {"action": "keep"}` if the session is inside it, then
  `git worktree remove` + `git branch -d <name>` (`-D` after a squash merge).
  A worktree that built anything has `deps/WarpWraps` initialized, and plain
  `remove` then refuses ("working trees containing submodules cannot be moved
  or removed"). Check that `git -C <wt> status --porcelain` is empty, then use
  `--force`. The worktree skill's "Clean up" section has the details.

Otherwise — a plain topic branch on an ordinary checkout — the local branch can
go with `git branch -d <branch>` (or `-D` after a squash merge, since the SHAs
differ; the merge already confirmed the patches landed).

## When NOT to merge

Stop after step 4 and hand the PR to the user instead of merging when:

- CI is red, or the push gate was bypassed with `--no-verify`;
- the diff moves a ratchet floor or touches the correctness references in ways
  only a human should sign off on;
- the user asked for a PR but not a merge.
