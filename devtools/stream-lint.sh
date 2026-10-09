#!/usr/bin/env bash
# Report every GPU call under src/, apps/ and test/ that is not issued on an explicit stream.
#
#   devtools/stream-lint.sh [--strict] [FILE ...]
#
# WHY THIS EXISTS. All GPU work in both demos runs on ONE non-blocking stream,
# DeviceResources', threaded through every call, with allocation
# stream-ordered on it too. The legacy default stream synchronizes with
# nothing created non-blocking, in either direction (docs/performance.md, "One
# stream"), so any call still on the legacy stream -- a
# stream-less copy, a literal nullptr where the stream goes, a device-wide
# sync, a synchronous free -- is a SILENT race with the work on that stream.
# Both stream bugs that cost real debugging time here were exactly that, and
# no sanitizer sees one (docs/testing.md, "Sanitizers": compute-sanitizer memcheck
# ran a cross-stream premature free clean 3/3). This lint finds them by
# reading the source, before a wrong number does. It is a GATE:
# cpp-tier.sh and cross-backend-check.sh run it with --strict.
#
# WHAT IT FLAGS, under src/, apps/ and test/ (comments and string literals are ignored):
#   alloc    a synchronous wwrMalloc / wwrFree (or a raw cuda*/hip* one).
#            Allocate a nevpt2::DeviceBuffer
#            (wwrMallocFromPoolAsync on the stream) and let it free itself
#            (wwrFreeAsync on the stream, at the end of its scope).
#   copy     a stream-less wwrMemcpy / wwrMemset (use the *Async form on the
#            stream).
#   sync     wwrDeviceSynchronize, or wwrStreamSynchronize on a literal
#            nullptr/0.
#   literal  a literal nullptr / 0 / NULL passed where a stream goes: to a
#            stream parameter of any function declared under src/, apps/ or test/ (found by
#            reading the declarations), or of the runtime/BLAS calls that take
#            one (wwrEventRecord, wwrMemcpyAsync, wwrblasSetStream,
#            parallel_for, ...).
#   default  a wwrStream_t PARAMETER defaulted to nullptr/0 -- a hidden
#            literal at every call that omits it. (A variable or member
#            initialized to nullptr is not flagged: an out-parameter for
#            wwrStreamCreateWithFlags is spelled that way, as in
#            apps/sanitizer_canary. So a null stream spelled that way has
#            to be caught in review, not here.)
#   launch   a <<<...>>> kernel launch without a stream argument, or with a
#            literal one.
#
# WHAT IT DOES NOT DO. It reads text, not types: it cannot see a stream that
# arrives as nullptr through a variable, and it only knows the stream
# position of functions whose declaration names wwrStream_t. It is a net for
# the common spellings, not a proof.
#
# Flags:
#   --strict   exit 1 when anything is reported. Also STREAM_LINT_STRICT=1.
#              Off by default for an ad-hoc run (a report you read); the two
#              local gates, cpp-tier.sh and cross-backend-check.sh, pass it
#              (when the report went empty).
#   FILE ...   lint these files instead of every tracked C++/CUDA file under
#              src/, apps/ and test/.
#
# Exit status: 0 report-only (or --strict and nothing found), 1 --strict and
# something found, 2 the lint could not run.
#: -- help stops here --
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

strict=${STREAM_LINT_STRICT:-0}
declare -a files=()
while [ $# -gt 0 ]; do
  case "$1" in
    --strict) strict=1 ;;
    -h|--help) usage_from_header "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "stream-lint: unknown flag: $1" >&2; exit 2 ;;
    *) files+=("$1") ;;
  esac
  shift
done

command -v perl >/dev/null 2>&1 || {
  echo "stream-lint: cannot run -- no perl on this machine" >&2
  exit 2
}

if [ "${#files[@]}" -eq 0 ]; then
  mapfile -t files < <(git -C "$REPO_ROOT" ls-files \
    'src/*.cpp' 'src/*.cppm' 'src/*.cu' 'src/*.cuh' 'src/*.h' 'src/*.hpp' \
    'apps/*.cpp' 'apps/*.cppm' 'apps/*.cu' 'apps/*.cuh' 'apps/*.h' 'apps/*.hpp' \
    'test/*.cpp' 'test/*.cppm' 'test/*.cu' 'test/*.cuh' 'test/*.h' 'test/*.hpp')
fi
if [ "${#files[@]}" -eq 0 ]; then
  echo "stream-lint: no files to lint" >&2
  exit 2
fi

set +e
perl -e '
use strict;
use warnings;

# Stream position (0-based argument index) of the runtime/BLAS calls that take
# one. Functions declared under src/, apps/ and test/ are added from their declarations below.
my %streamArg = (
  wwrStreamSynchronize => [0], wwrStreamWaitEvent => [0],
  wwrEventRecord => [1], wwrMemcpyAsync => [4], wwrMemsetAsync => [3],
  wwrMallocAsync => [2], wwrFreeAsync => [1], wwrblasSetStream => [1],
  cublasSetStream => [1], parallel_for => [0],
);

my $paren;
$paren = qr{\((?:[^()]++|(??{$paren}))*\)};
my $literal = qr{^\s*(?:nullptr|0|NULL|0x0)\s*$};

# Blank comments and string/char literals, keeping every newline (so offsets
# still map to lines) and the quotes themselves.
sub scrub {
  my ($s) = @_;
  $s =~ s{(//[^\n]*)|(/\*.*?\*/)|("(?:\\.|[^"\\\n])*")|(\x27(?:\\.|[^\x27\\\n])*\x27)}{
    my $m = $&;
    defined $3 || defined $4 ? substr($m, 0, 1) . (" " x (length($m) - 2)) . substr($m, -1)
                             : ($m =~ s/[^\n]/ /gr)
  }gse;
  return $s;
}

# Split an argument/parameter list (without its outer parens) at top-level
# commas. $angles: also treat <> as brackets (declarations: std::array<int, 4>).
sub splitArgs {
  my ($s, $angles) = @_;
  my @out; my $depth = 0; my $cur = "";
  for my $c (split //, $s) {
    if ($c =~ /[({\[]/ || ($angles && $c eq "<")) { $depth++ }
    elsif ($c =~ /[)}\]]/ || ($angles && $c eq ">")) { $depth-- }
    if ($c eq "," && $depth == 0) { push @out, $cur; $cur = ""; next }
    $cur .= $c;
  }
  push @out, $cur if $cur =~ /\S/ || @out;
  return @out;
}

my (%raw, %clean);
for my $f (@ARGV) {
  open my $fh, "<", $f or die "stream-lint: cannot read $f\n";
  local $/; $raw{$f} = <$fh>; close $fh;
  $clean{$f} = scrub($raw{$f});
}

# Pass 1: every declaration under src/, apps/ and test/ with a wwrStream_t parameter.
for my $f (keys %clean) {
  my $s = $clean{$f};
  while ($s =~ /\b(\w+)\s*($paren)/g) {
    my ($name, $list) = ($1, $2);
    next unless $list =~ /\bwwrStream_t\b/;
    my @params = splitArgs(substr($list, 1, -1), 1);
    for my $i (0 .. $#params) {
      push @{ $streamArg{$name} }, $i
        if $params[$i] =~ /\bwwrStream_t\b/ && !grep { $_ == $i } @{ $streamArg{$name} // [] };
    }
  }
}

my @findings;
sub report {
  my ($f, $pos, $cat, $what) = @_;
  my $line = 1 + (substr($clean{$f}, 0, $pos) =~ tr/\n//);
  my $text = (split /\n/, $raw{$f}, -1)[$line - 1] // "";
  $text =~ s/^\s+|\s+$//g;
  push @findings, [$f, $line, $cat, $what, $text];
}

for my $f (sort keys %clean) {
  my $s = $clean{$f};
  while ($s =~ /\b((?:wwr|cuda|hip)(?:Malloc|Free))\s*\(/g) {
    report($f, $-[0], "alloc", "synchronous $1 (use a nevpt2::DeviceBuffer)");
  }
  while ($s =~ /\b((?:wwr|cuda|hip)(?:Memcpy|Memset))\s*\(/g) {
    report($f, $-[0], "copy", "stream-less $1 (use ${1}Async on the stream)");
  }
  while ($s =~ /\b((?:wwr|cuda|hip)DeviceSynchronize)\s*\(/g) {
    report($f, $-[0], "sync", "$1 (synchronize the stream instead)");
  }
  while ($s =~ /\bwwrStream_t\s+\w+\s*=\s*(?:nullptr|0|NULL)\s*(?=[,)])/g) {
    report($f, $-[0], "default", "wwrStream_t parameter defaulted to a literal");
  }
  while ($s =~ /<<<(.*?)>>>/gs) {
    my $at = $-[0];  # before splitArgs, whose own matches reset @-
    my @cfg = splitArgs($1, 0);
    if (@cfg < 4) {
      report($f, $at, "launch", "kernel launch with no stream (runs on the legacy stream)");
    } elsif ($cfg[3] =~ $literal) {
      report($f, $at, "launch", "kernel launch on a literal stream");
    }
  }
  for my $name (sort keys %streamArg) {
    while ($s =~ /\b\Q$name\E\s*($paren)/g) {
      my $at = $-[0];
      my @args = splitArgs(substr($1, 1, -1), 0);
      for my $i (@{ $streamArg{$name} }) {
        next unless $i <= $#args && $args[$i] =~ $literal;
        my $cat = $name eq "wwrStreamSynchronize" ? "sync" : "literal";
        report($f, $at, $cat, "literal stream passed to $name (argument " . ($i + 1) . ")");
      }
    }
  }
}

my %count;
for my $x (sort { $a->[0] cmp $b->[0] || $a->[1] <=> $b->[1] } @findings) {
  my ($f, $line, $cat, $what, $text) = @$x;
  $count{$cat}++;
  printf "%s:%d: [%s] %s\n    %s\n", $f, $line, $cat, $what, $text;
}
my @cats = qw(alloc copy sync literal default launch);
printf "stream-lint: %d finding(s) in %d file(s): %s\n", scalar(@findings), scalar(keys %clean),
  join(" ", map { "$_=" . ($count{$_} // 0) } @cats);
exit(@findings ? 1 : 0);
' "${files[@]}"
found=$?
set -e

case "$found" in
  0) echo "stream-lint: PASS (nothing off the stream)"; exit 0 ;;
  1)
    if [ "$strict" = 1 ]; then
      echo "stream-lint: FAIL (--strict)"
      exit 1
    fi
    echo "stream-lint: report-only -- not failing (--strict or STREAM_LINT_STRICT=1 to fail)"
    exit 0
    ;;
  *) echo "stream-lint: the lint itself failed (exit $found)" >&2; exit 2 ;;
esac
