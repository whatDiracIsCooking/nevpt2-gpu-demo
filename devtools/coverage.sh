#!/usr/bin/env bash
# Host code coverage: configure, build and test a coverage preset, then merge
# its profiles and report what src/ and apps/ executed. clang source-based
# coverage (cmake/nevpt2_coverage.cmake): host CXX units only, every `.cu`
# left uninstrumented on both backends, so a kernel line is never in the
# report and a kernel's correctness stays the golden tier's to prove.
#
#   devtools/coverage.sh [--preset NAME] [--no-build | --report-only]
#                        [--html DIR] [--lcov FILE] [-j N] [-- <ctest args>]
#
# Needs a card: the test preset is what `fast` runs (golden entries + the
# unit tier). ctest runs SERIALLY, on purpose -- several golden entries on
# one card at once abort for lack of device memory, and a run with aborted
# entries reports their host paths as never executed. A test failure still
# produces the report (from the processes that did write a profile) but
# makes the script exit 1, so a partial number is never mistaken for a clean
# one.
#
# Flags:
#   --preset NAME   configure/build/test preset (default: COVERAGE_PRESET from
#                   devtools/config.sh, `coverage`; `hip-coverage` on the AMD
#                   card).
#   --no-build      skip configure and build; re-test the existing build dir.
#   --report-only   skip build AND ctest; re-report the last run's merged
#                   profile (e.g. after editing COVERAGE_IGNORE_REGEX).
#   --html DIR      also write llvm-cov's line-by-line HTML into DIR.
#   --lcov FILE     also export the merged profile as an lcov tracefile.
#   -j N            parallel compile jobs (default: BUILD_JOBS from config.sh).
#   -- <args>       passed to ctest verbatim (e.g. `-- -L unit` for the unit
#                   tier alone). Profiles from a narrowed run cover only it.
#
# The summary table is what docs/testing.md, "Coverage" quotes; re-measure
# with this script, not by hand, so the number keeps one definition.
#: -- help stops here --
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

preset=$COVERAGE_PRESET
build=1
run_tests=1
html_dir=
lcov_file=
jobs=$BUILD_JOBS
declare -a passthrough=()

while [ $# -gt 0 ]; do
  case "$1" in
    --preset)   preset=$2; shift 2;;
    --no-build) build=0; shift;;
    --report-only) build=0; run_tests=0; shift;;
    --html)     html_dir=$2; shift 2;;
    --lcov)     lcov_file=$2; shift 2;;
    -j|--jobs)  jobs=$2; shift 2;;
    --)         shift; passthrough=("$@"); break;;
    -h|--help)  usage_from_header "${BASH_SOURCE[0]}"; exit 0;;
    *)          echo "coverage: unknown flag: $1" >&2; exit 2;;
  esac
done

# llvm-cov and llvm-profdata must be the SAME LLVM release as the clang that
# instrumented the build: the raw profile format changes between releases.
# Debian/Ubuntu ship them versioned (llvm-cov-20) or under /usr/lib/llvm-<N>/.
clang_major=$(clang++ -dumpversion | cut -d. -f1)
find_llvm() {
  local t
  for t in "/usr/lib/llvm-$clang_major/bin/$1" "$1-$clang_major" "$1"; do
    command -v "$t" >/dev/null 2>&1 && { command -v "$t"; return 0; }
  done
  echo "coverage: no $1 for LLVM $clang_major (the clang that builds the tree)." >&2
  echo "  Install llvm-$clang_major, or put $1-$clang_major on PATH." >&2
  exit 2
}
llvm_cov=$(find_llvm llvm-cov)
llvm_profdata=$(find_llvm llvm-profdata)
llvm_readelf=$(find_llvm llvm-readelf)

# Every non-default preset builds into build-<preset>/, and the coverage test
# presets point LLVM_PROFILE_FILE at its profraw/ (CMakePresets.json). A
# preset that broke that convention fails the CMakeCache check below.
build_dir="$REPO_ROOT/build-$preset"
profile_dir="$build_dir/profraw"

if [ "$build" -eq 1 ]; then
  cmake --preset "$preset"
  declare -a jarg=()
  [ -n "$jobs" ] && jarg=(-j "$jobs")
  cmake --build --preset "$preset" "${jarg[@]}"
fi
[ -f "$build_dir/CMakeCache.txt" ] || {
  echo "coverage: $build_dir is not configured; drop --no-build." >&2
  exit 2
}
grep -q '^NEVPT2_ENABLE_COVERAGE:BOOL=ON' "$build_dir/CMakeCache.txt" || {
  echo "coverage: $build_dir was not configured with NEVPT2_ENABLE_COVERAGE=ON." >&2
  exit 2
}

merged="$build_dir/coverage.profdata"
test_rc=0
if [ "$run_tests" -eq 1 ]; then
  # Stale profiles from an earlier run (or an older build of the same binary)
  # would be merged in, or rejected as mismatched; start empty.
  rm -rf "$profile_dir" "$merged"
  mkdir -p "$profile_dir"

  ctest --preset "$preset" "${passthrough[@]}" || test_rc=$?

  shopt -s nullglob
  profiles=("$profile_dir"/*.profraw)
  shopt -u nullglob
  [ "${#profiles[@]}" -gt 0 ] || {
    echo "coverage: no profiles in $profile_dir -- did any test run?" >&2
    exit 1
  }
  "$llvm_profdata" merge -sparse "${profiles[@]}" -o "$merged"
  echo
  echo "coverage: merged ${#profiles[@]} profiles"
else
  [ -f "$merged" ] || { echo "coverage: no $merged yet; run without --report-only." >&2; exit 2; }
fi

# Every instrumented binary in the build, found by its coverage-mapping
# section rather than listed by name, so a new demo or gtest binary is
# counted by being built.
declare -a objects=()
while IFS= read -r -d '' f; do
  # grep >/dev/null, not grep -q: -q exits at the first match, readelf then
  # dies of SIGPIPE, and under pipefail the binary is silently skipped.
  "$llvm_readelf" -S "$f" 2>/dev/null | grep __llvm_covmap >/dev/null && objects+=("$f")
done < <(find "$build_dir" -path '*/CMakeFiles' -prune -o -path '*/_deps' -prune -o \
              -type f -perm -u+x -print0)
[ "${#objects[@]}" -gt 0 ] || { echo "coverage: no instrumented binaries in $build_dir" >&2; exit 1; }
declare -a obj_args=("${objects[0]}")
for f in "${objects[@]:1}"; do obj_args+=(-object "$f"); done

declare -a cov_args=(-instr-profile "$merged" -ignore-filename-regex "$COVERAGE_IGNORE_REGEX")

echo "coverage: ${#objects[@]} instrumented binaries, preset $preset"
"$llvm_cov" report "${cov_args[@]}" "${obj_args[@]}"

if [ -n "$html_dir" ]; then
  "$llvm_cov" show "${cov_args[@]}" -format=html -show-line-counts-or-regions \
    -output-dir "$html_dir" "${obj_args[@]}"
  echo "coverage: HTML in $html_dir/index.html"
fi
if [ -n "$lcov_file" ]; then
  "$llvm_cov" export "${cov_args[@]}" -format=lcov "${obj_args[@]}" >"$lcov_file"
  echo "coverage: lcov tracefile $lcov_file"
fi

if [ "$test_rc" -ne 0 ]; then
  echo "coverage: ctest FAILED (exit $test_rc) -- the numbers above are partial." >&2
  exit 1
fi
