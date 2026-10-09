#!/usr/bin/env bash
# Fail on any `throw` under src/, apps/ and test/ -- the two-tier error model's gate.
#
#   devtools/throw-lint.sh [FILE ...]
#
# WHY THIS EXISTS. This tree reports a failure in exactly two ways
# (docs/architecture.md, "Error handling"):
#   abort    our bug -- a broken invariant (`check`, nevpt2.error_handling) or
#            a failed GPU runtime/BLAS/solver call (`gpuCheck`, nevpt2.wwr):
#            print the caller's file:line and std::abort().
#   value    not our bug -- bad input, bad configuration, an unsupported
#            request, a numerical refusal: return a Result<T> / Status holding
#            an Error, propagate it with NEVPT2_TRY, report() it once at main.
# An exception is a third, unmanaged path: nothing catches one, so it ends in
# std::terminate with no file:line of ours. Nothing in the tree throws, and
# this lint keeps a new one from creeping in.
#
# WHY A LINT AND NOT -fno-exceptions. The `import std` and WarpWraps BMIs must
# be built with the same flags as their importers, so exceptions cannot be
# switched off for our units alone; and the standard library can still throw
# (std::bad_alloc), which, uncaught, terminates -- the abort tier already.
# What we can rule out is OUR code throwing, and that is a text check.
#
# WHAT IT FLAGS: the keyword `throw` in any tracked C++/CUDA file under src/
# and apps/, with comments and string/char literals blanked first (the same
# scrub as devtools/stream-lint.sh). `rethrow_exception` and friends are other
# words and are not matched; nothing in the tree calls them either.
#
# WHAT IT DOES NOT DO. It reads text: a raw string literal R"(...)" that
# contains the word is not blanked (and would be reported -- a false positive,
# never a miss), and it cannot see a throw inside a library we call.
#
# It always fails on a finding -- there is no report-only mode, because the
# report is empty. cpp-tier.sh and cross-backend-check.sh run
# it beside devtools/stream-lint.sh --strict.
#
# Flags:
#   FILE ...   lint these files instead of every tracked C++/CUDA file under
#              src/, apps/ and test/.
#
# Exit status: 0 nothing found, 1 something found, 2 the lint could not run.
#: -- help stops here --
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

declare -a files=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage_from_header "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "throw-lint: unknown flag: $1" >&2; exit 2 ;;
    *) files+=("$1") ;;
  esac
  shift
done

command -v perl >/dev/null 2>&1 || {
  echo "throw-lint: cannot run -- no perl on this machine" >&2
  exit 2
}

if [ "${#files[@]}" -eq 0 ]; then
  mapfile -t files < <(git -C "$REPO_ROOT" ls-files \
    'src/*.cpp' 'src/*.cppm' 'src/*.cu' 'src/*.cuh' 'src/*.h' 'src/*.hpp' \
    'apps/*.cpp' 'apps/*.cppm' 'apps/*.cu' 'apps/*.cuh' 'apps/*.h' 'apps/*.hpp' \
    'test/*.cpp' 'test/*.cppm' 'test/*.cu' 'test/*.cuh' 'test/*.h' 'test/*.hpp')
fi
if [ "${#files[@]}" -eq 0 ]; then
  echo "throw-lint: no files to lint" >&2
  exit 2
fi

set +e
perl -e '
use strict;
use warnings;

# Blank comments and string/char literals, keeping every newline (so offsets
# still map to lines) and the quotes themselves. Same as stream-lint.sh.
sub scrub {
  my ($s) = @_;
  $s =~ s{(//[^\n]*)|(/\*.*?\*/)|("(?:\\.|[^"\\\n])*")|(\x27(?:\\.|[^\x27\\\n])*\x27)}{
    my $m = $&;
    defined $3 || defined $4 ? substr($m, 0, 1) . (" " x (length($m) - 2)) . substr($m, -1)
                             : ($m =~ s/[^\n]/ /gr)
  }gse;
  return $s;
}

my @findings;
for my $f (sort @ARGV) {
  open my $fh, "<", $f or die "throw-lint: cannot read $f\n";
  local $/; my $raw = <$fh>; close $fh;
  my $clean = scrub($raw);
  my @lines = split /\n/, $raw, -1;
  while ($clean =~ /\bthrow\b/g) {
    my $line = 1 + (substr($clean, 0, $-[0]) =~ tr/\n//);
    my $text = $lines[$line - 1] // "";
    $text =~ s/^\s+|\s+$//g;
    push @findings, sprintf("%s:%d: [throw] use check() for our bug, return an Error for bad input\n    %s\n",
                            $f, $line, $text);
  }
}
print for @findings;
printf "throw-lint: %d finding(s) in %d file(s)\n", scalar(@findings), scalar(@ARGV);
exit(@findings ? 1 : 0);
' "${files[@]}"
found=$?
set -e

case "$found" in
  0) echo "throw-lint: PASS (no throw in src/, apps/ or test/)"; exit 0 ;;
  1) echo "throw-lint: FAIL"; exit 1 ;;
  *) echo "throw-lint: the lint itself failed (exit $found)" >&2; exit 2 ;;
esac
