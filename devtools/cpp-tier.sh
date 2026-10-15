#!/usr/bin/env bash
# Configure, build and ctest the C++ tree -- the ONLY tier that proves src/
# still works, on whichever backend the preset picks. There is no install
# tier; every golden ctest entry is labelled `gpu`, so this needs a real card,
# and a green `-LE gpu` run proves only the host-only GoogleTest suites
# (test/) -- logic, never an energy. It runs five gates first, none of which
# needs a card: devtools/stream-lint.sh --strict, devtools/throw-lint.sh (no
# `throw` in src/ or apps/), reference_data/golden_layout_check.py (the golden
# layout contract, and its scan for stray file readers in src/ and apps/),
# `ruff check` and `pytest reference_data/tests` (the oracle's own hand
# cases), the last three through `uv run` against the locked environment
# (they FAIL, never skip, when that environment is missing or stale).
# devtools/golden-repro.sh, the regeneration gate, is NOT a phase: it costs
# minutes of CPU (docs/testing.md, "The golden regeneration gate").
#
#   devtools/cpp-tier.sh [--preset NAME] [--fresh] [--clean] [--no-test]
#                        [--tidy] [--rocm] [-j N] [--report DIR]
#                        [-- <ctest args>]
#
# It is NOT on the push gate: a C++23
# named-module build with CUDA separable compilation costs minutes even warm,
# and a gate that costs minutes is one people learn to bypass with --no-verify.
# Run it yourself, on a box with a card, before opening a PR. There is no CI
# to run it for you, on purpose (docs/testing.md, "No CI, and the local gate"): a
# hosted runner has no GPU, and every ctest entry here needs one.
#
# THE TOOLCHAIN IS NOT ON YOUR HOST. clang-20 with libc++'s module manifest,
# CMake 4.2 and nvcc live in docker/Dockerfile.cuda, so the normal way to call
# this is from inside that container (which DEVCONTAINER_CONFIG in config.sh
# already selects):
#
#   devtools/devcontainer.sh shell -c devtools/cpp-tier.sh
#
# On a bare host with its own clang and CUDA, use --preset workstation. The two
# presets are the same configuration and differ only in binaryDir -- nothing is
# expected prebuilt anywhere (the one fetch, GoogleTest, lands in each build
# dir's own _deps/), so the point is simply that a host build and a container build get separate
# build directories instead of reconfiguring each other's.
# `devtools/doctor.sh` lists which of the two situations you are in.
#
# Flags:
#   --preset NAME   configure/build preset (default: CMAKE_PRESET from
#                   devtools/config.sh). The test preset follows it when one of
#                   the same name exists, and is skipped with a note when not
#                   -- see `ctest --list-presets` (the ci-* presets have
#                   none). This is how the sanitizer tier runs:
#                   --preset asan | ubsan | compute-sanitizer on the CUDA card,
#                   hip-asan | hip-ubsan on the AMD one, each configuring,
#                   building and testing its own build-<preset>/ (the small
#                   CAS(4,4)/CAS(8,8) entries, the unit tier -- less its
#                   `death` suites under compute-sanitizer -- and that
#                   sanitizer's canaries).
#                   The compute-sanitizer tool is a cache variable, so pick a
#                   non-default one by configuring first:
#                     cmake --preset compute-sanitizer \
#                       -DNEVPT2_COMPUTE_SANITIZER_TOOL=initcheck
#                     devtools/cpp-tier.sh --preset compute-sanitizer
#   --fresh         wipe the CMake cache first (`--fresh`). Needed after
#                   changing a toolchain variable; a plain reconfigure keeps
#                   the old value and the change appears to do nothing.
#   --clean         rebuild every object (`--clean-first`), keeping the cache.
#   --no-test       configure and build only. The five gates above still
#                   run: they need no card and cost seconds, and a
#                   --no-test run is often the only run a no-card change
#                   gets.
#   --tidy          after a successful build, run clang-tidy over the .cpp
#                   implementation units and apps/*/main.cppm using the
#                   build's compile_commands.json. Reports findings; does NOT
#                   fail the run (the comment above the clang-tidy phase in
#                   this script says why). Off by default so the normal
#                   tier stays a build-and-test.
#   --rocm          run the HIP preset against a REAL AMD GPU, from the HOST.
#                   Re-execs this script inside ROCM_IMAGE with /dev/kfd,
#                   /dev/dri and the NUMERIC host render/video gids passed
#                   through (config.sh's ROCM_* settings say why numeric).
#                   Implies --preset hip unless --preset says otherwise.
#                   Every other flag is forwarded. Exits 2 rather than 0 when
#                   it cannot run, so a skip is never read as a pass.
#                   cross-backend-check.sh proves the other backend COMPILES;
#                   this proves it RUNS, and needs a card to say so.
#   -j N            parallel compile jobs (default: BUILD_JOBS from config.sh,
#                   or Ninja's own choice when that is empty). Module builds
#                   are memory-hungry per job -- a too-high N OOMs rather than
#                   failing cleanly.
#   --report DIR    timestamped log + one-line summary (default:
#                   SLOW_TIER_REPORTS from config.sh).
#   -- <args>       everything after `--` is passed to ctest verbatim
#                   (e.g. `-- -R Constants` to scope the run). ctest has
#                   one entry per golden case, so -R matches names like
#                   nevpt2_cas1010 / nevpt2_cas1212 / nevpt2_cas1010_cublas.
#
# Exit status is the first failing phase's, so a scheduler can alert on it,
# and the summary line names every phase that failed (`failed=...`).
#: -- help stops here --
#
# Deliberately NOT what docker/compose.yaml's `build`/`test` services do, even
# though the commands overlap: compose owns the batch path (it brings its own
# container, tees to .log/, for an unattended run), this owns the
# already-inside-a-container path. Changing the preset list means touching
# CMakePresets.json, which both read -- neither hardcodes one.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

preset=$CMAKE_PRESET
test_preset=$CTEST_PRESET
fresh=0
clean=0
run_tests=1
tidy=0
rocm=0
jobs=$BUILD_JOBS
report_dir=$SLOW_TIER_REPORTS
declare -a passthrough=()
preset_explicit=0

while [ $# -gt 0 ]; do
  case "$1" in
    --preset)  preset=$2; preset_explicit=1; shift 2;;
    --fresh)   fresh=1; shift;;
    --clean)   clean=1; shift;;
    --no-test) run_tests=0; shift;;
    --tidy)    tidy=1; shift;;
    --rocm)    rocm=1; shift;;
    -j|--jobs) jobs=$2; shift 2;;
    --report)  report_dir=$2; shift 2;;
    --)        shift; passthrough=("$@"); break;;
    -h|--help) usage_from_header "${BASH_SOURCE[0]}"; exit 0;;
    *)         echo "cpp-tier: unknown flag: $1" >&2; exit 2;;
  esac
done

# --preset switches the test preset with it, or the run silently tests the
# wrong build directory: ctest --preset default reads build/, while
# --preset asan built into build-asan/.
[ "$preset_explicit" -eq 1 ] && test_preset=$preset

# --rocm: re-exec inside the ROCm image with the GPU attached, then stop.
#
# This is the one path where cpp-tier.sh launches a container rather than
# assuming it is already in one, and it is here rather than in a script of its
# own because everything after this point -- preset handling, the test-preset
# fallback, logging, --tidy -- is what we want to run, unchanged, on the other
# side. The container calls this same script without --rocm.
if [ "$rocm" = 1 ]; then
  [ "$preset_explicit" -eq 1 ] || { preset=$ROCM_PRESET; test_preset=$ROCM_PRESET; }

  command -v docker >/dev/null 2>&1 || {
    echo "cpp-tier --rocm: cannot run -- no docker on this machine." >&2
    exit 2
  }

  docker image inspect "$ROCM_IMAGE" >/dev/null 2>&1 || {
    cat >&2 <<EOF
cpp-tier --rocm: cannot run -- docker image '$ROCM_IMAGE' is not built.

  Build it once (build.sh walks the chain -- docker/Dockerfile.hip needs
  docker/Dockerfile.base built and tagged first):

    docker/build.sh hip

  That tags <PROJECT_NAME>:hip, which is what ROCM_IMAGE defaults to. Or set
  ROCM_IMAGE to an image that has the ROCm toolchain.
EOF
    exit 2
  }

  # The device nodes, and the gids that make them openable. A missing /dev/kfd
  # means no amdgpu -- worth naming, because the alternative is a build that
  # succeeds and then reports no device, which reads like a code problem.
  declare -a dev_args=()
  while IFS= read -r dev; do
    [ -n "$dev" ] || continue
    [ -e "$dev" ] || {
      echo "cpp-tier --rocm: cannot run -- $dev does not exist." >&2
      echo "  No AMD compute device here. Is amdgpu loaded? (lsmod | grep amdgpu)" >&2
      exit 2
    }
    dev_args+=(--device="$dev")
  done <<<"$ROCM_DEVICES"

  # NUMERIC, resolved on the HOST. See config.sh's ROCM_GROUPS: passing the
  # NAME lets the container resolve it against its own /etc/group, and the run
  # then fails exactly like a machine with no GPU.
  declare -a group_args=()
  while IFS= read -r grp; do
    [ -n "$grp" ] || continue
    # Numeric HOST gid, or empty; host_gid (lib.sh) owns the getent exit-2 trap.
    gid=$(host_gid "$grp")
    [ -n "$gid" ] || {
      echo "cpp-tier --rocm: cannot run -- no '$grp' group on this host." >&2
      echo "  getent group $grp found nothing; ROCM_GROUPS in config.sh names it." >&2
      exit 2
    }
    group_args+=(--group-add "$gid")
  done <<<"$ROCM_GROUPS"

  # Mounted at its own host path, not /workspace, for two reasons: a CMake
  # cache records absolute directories, so this shares build-hip/ with a
  # devcontainer build instead of fighting it; and `git rev-parse` keeps
  # working, which is what puts the revision in the log name. A worktree's
  # .git points into the main checkout, outside this tree, so that directory
  # is mounted too when it is not already under us.
  declare -a mount_args=(-v "$REPO_ROOT:$REPO_ROOT")
  git_common=$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null || true)
  case "$git_common" in
    "")     ;;
    /*)     ;;
    *)      git_common="$REPO_ROOT/$git_common";;
  esac
  if [ -n "$git_common" ] && [ -d "$git_common" ]; then
    git_common=$(cd "$git_common" && pwd)
    case "$git_common" in
      "$REPO_ROOT"/*) ;;
      *) mount_args+=(-v "$git_common:$git_common");;
    esac
  fi

  # Rebuilt from parsed state rather than replayed from "$@", so --rocm is
  # dropped and the resolved preset travels explicitly.
  declare -a inner=(devtools/cpp-tier.sh --preset "$preset" --report "$report_dir")
  [ "$fresh" = 1 ]     && inner+=(--fresh)
  [ "$clean" = 1 ]     && inner+=(--clean)
  [ "$run_tests" = 0 ] && inner+=(--no-test)
  [ "$tidy" = 1 ]      && inner+=(--tidy)
  [ -n "$jobs" ]       && inner+=(-j "$jobs")
  [ ${#passthrough[@]} -gt 0 ] && inner+=(-- "${passthrough[@]}")

  echo "cpp-tier --rocm: $ROCM_IMAGE, preset=$preset, devices=$(echo $ROCM_DEVICES)"
  exec docker run --rm \
    "${dev_args[@]}" "${group_args[@]}" \
    -u "$(id -u):$(id -g)" -e HOME=/tmp \
    "${mount_args[@]}" -w "$REPO_ROOT" \
    "$ROCM_IMAGE" "${inner[@]}"
fi

for tool in cmake ninja; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "cpp-tier: $tool not found -- this needs the CUDA container:" >&2
    echo "  devtools/devcontainer.sh shell -c devtools/cpp-tier.sh" >&2
    echo "(or --preset workstation, on a host with its own clang and CUDA)" >&2
    exit 1
  }
done

case "$report_dir" in /*) ;; *) report_dir="$REPO_ROOT/$report_dir";; esac
mkdir -p "$report_dir"
stamp=$(date +%Y%m%d-%H%M%S)
rev=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
log="$report_dir/cpp-$stamp-$rev-$preset.log"

echo "cpp-tier: $REPO_ROOT @ $rev  preset=$preset -> $log"

# Where this preset puts compile_commands.json. Asked of CMake rather than
# guessed: `default` uses build/ but the others use build-<preset>/, and a
# hardcoded build/ would have --tidy silently lint the previous preset's
# database.
build_dir=$(cmake --preset "$preset" -N 2>/dev/null |
  sed -n 's/^.*Binary directory: *//p' | head -1)
[ -n "$build_dir" ] || build_dir="$REPO_ROOT/build"

# `jq -e` is not assumed to be present, so the "does this test preset exist"
# question is answered by asking ctest, which is. --show-only lists the tests a
# preset would run and exits non-zero when the preset is unknown.
has_test_preset() {
  [ -n "$test_preset" ] || return 1
  ctest --preset "$test_preset" --show-only >/dev/null 2>&1
}

declare -a failed_phases=()
run_phase() {
  local name=$1; shift
  echo "--- $name: $* ---" | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  [ "$status" -eq 0 ] || failed_phases+=("$name")
  return "$status"
}

# The three Python phases below run through in_locked_env (devtools/lib.sh):
# in the LOCKED environment, and FAILING -- never skipping -- when it is
# missing or stale.

rc=0
set +e

# Every GPU call on the one non-blocking stream (devtools/stream-lint.sh). A
# gate: --strict, and its exit status is folded into rc, so a
# finding fails the tier (the build and tests still run, to report them too).
run_phase stream-lint "$REPO_ROOT/devtools/stream-lint.sh" --strict || rc=$?

# No `throw` in src/ or apps/ (devtools/throw-lint.sh): a failure is check()
# (abort tier) or a returned Error (value tier), never an exception. A gate,
# folded into rc the same way.
run_phase throw-lint "$REPO_ROOT/devtools/throw-lint.sh" || rc=$?

# The golden layout contract (docs/testing.md, "The golden layout contract"):
# every committed golden through every Python reader, and the scan for a stray
# file reader in src/ or apps/ -- the check #84 turned red because no gate ran
# it. Then ruff over the Python tree (pyproject.toml's [tool.ruff]). Both are
# gates, folded into rc like the lints, and both run under --no-test too.
run_phase golden-layout in_locked_env python \
  "$REPO_ROOT/reference_data/golden_layout_check.py" || rc=$?
run_phase ruff in_locked_env ruff check "$REPO_ROOT" || rc=$?

# The oracle's own hand cases (docs/testing.md, "The oracle's own tests"): the
# golden formats' readers and writers and the refusal guards that keep a
# noise-dependent state out of golden/, on synthetic inputs -- no PySCF solve,
# no card, no network. A gate like the two above, and like them it FAILS when
# the environment lacks pytest (an image built before the lock added it).
run_phase pytest in_locked_env pytest -q "$REPO_ROOT/reference_data/tests" || rc=$?

configure=(cmake --preset "$preset")
[ "$fresh" -eq 1 ] && configure+=(--fresh)
[ "$rc" -eq 0 ] && { run_phase configure "${configure[@]}" || rc=$?; }

if [ "$rc" -eq 0 ]; then
  build=(cmake --build --preset "$preset")
  [ -n "$jobs" ] && build+=(-j "$jobs")
  [ "$clean" -eq 1 ] && build+=(--clean-first)
  run_phase build "${build[@]}" || rc=$?
fi

if [ "$rc" -eq 0 ] && [ "$run_tests" -eq 1 ]; then
  if has_test_preset; then
    run_phase ctest ctest --preset "$test_preset" "${passthrough[@]}" || rc=$?
  else
    echo "cpp-tier: no test preset '${test_preset:-<empty>}' -- built only" \
      | tee -a "$log"
    # Ask CMakePresets.json rather than naming the set here: a hardcoded list
    # goes stale the first time a preset is added, and this message is the one
    # place someone looks to find out which presets can actually run ctest.
    # `ctest --list-presets` prints a header and then two-space-indented,
    # quoted names.
    avail=$(ctest --list-presets 2>/dev/null |
      sed -n 's/^  "\([^"]*\)".*/\1/p' | paste -sd, - | sed 's/,/, /g')
    echo "          (test presets in CMakePresets.json: ${avail:-none found})" \
      | tee -a "$log"
  fi
fi

# clang-tidy is deliberately advisory -- run on request (--tidy), never folded
# into rc. It lints the src/*.cpp implementation units and the apps/*/main.cppm
# binaries (module units nothing imports, so effectively implementation units).
# There is NO .clang-tidy anywhere in the tree, so it runs clang-tidy's default
# checks (clang-diagnostic-*, clang-analyzer-*) with the default header filter,
# which reports only the files named on the command line. clang-tidy exits 0 on
# warnings, so a run with findings still passes this phase -- read the log.
# The tree is not clean under those defaults today: the static analyzer
# reports a few findings nobody has triaged, and some may be false positives.
# Two reasons it stays advisory: those findings, and clang-tidy's C++23-module
# support, which is still incomplete and can emit the odd false diagnostic on
# a module construct -- something a build gate should not fail on. Making it a
# gate means adding a .clang-tidy and triaging the findings first.
if [ "$rc" -eq 0 ] && [ "$tidy" -eq 1 ]; then
  if ! command -v clang-tidy >/dev/null 2>&1; then
    echo "cpp-tier: clang-tidy not found -- skipping --tidy" | tee -a "$log"
  else
    mapfile -t tidy_files < <(git -C "$REPO_ROOT" ls-files 'src/*.cpp' 'apps/*/main.cppm')
    if [ "${#tidy_files[@]}" -eq 0 ]; then
      echo "cpp-tier: no src/*.cpp or apps/*/main.cppm to tidy" | tee -a "$log"
    else
      # Advisory: the phase's own status is reported but never folded into rc.
      run_phase clang-tidy clang-tidy -p "$build_dir" --quiet "${tidy_files[@]}" \
        || echo "cpp-tier: clang-tidy reported findings (advisory)" | tee -a "$log"
    fi
  fi
fi

set -e

verdict=$([ "$rc" -eq 0 ] && echo PASS || echo "FAIL(rc=$rc)")
if [ "${#failed_phases[@]}" -gt 0 ]; then
  verdict="$verdict failed=$(IFS=,; echo "${failed_phases[*]}")"
fi
summary="cpp-tier $verdict  rev=$rev  preset=$preset  stamp=$stamp"
echo "$summary" | tee -a "$log"
echo "$summary" >> "$report_dir/summary.log"
exit "$rc"
