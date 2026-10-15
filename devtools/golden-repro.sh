#!/usr/bin/env bash
# Regenerate the committed N2 CAS(4,4) / CAS(8,8) references with the
# generators' own settings, into a scratch directory, and compare each with
# the committed file BYTE FOR BYTE. The regeneration gate of docs/testing.md,
# "The golden regeneration gate".
#
#   devtools/golden-repro.sh [-j N] [--only REGEX] [--committed DIR]
#                            [--out DIR]
#
# CLAUDE.md promises the generators are bit-reproducible -- single-threaded,
# a 1e-12 Davidson, point-group symmetry -- and that "a regenerated file that
# differs from the committed one means something changed". This is the one
# place that re-runs the promise, and the one place a byte comparison is right:
# everywhere else compares against a tolerance.
#
# What it regenerates (fourteen files, twelve generator runs):
#   generate_golden.py     n2_ccpvdz_cas44 / cas88 .nevpt2gold + .gradgold,
#                          their _df twins, the two .mogold sidecars (a fresh
#                          --casscf each) and the two *_casscf[_df] goldens
#                          (--mo-from the COMMITTED sidecar, as
#                          docs/reference-data.md's commands do, so a sidecar
#                          that drifted fails as itself, not as its golden too)
#   sc_source_oracle.py    n2_ccpvdz_cas44.srcgold and n2_ccpvdz_cas44_pc.srcgold,
#                          and their directional twins (#105) *_dir.srcgold /
#                          *_pc_dir.srcgold (all four read
#                          golden/n2_ccpvdz_cas44.nevpt2gold, the committed
#                          one, as their input)
#
# The comparison is `cmp` on the whole file, with ONE documented exception
# (docs/reference-data.md, "Regenerating the golden file"): a committed CASCI
# golden written as NEVPT2G1 regenerates as NEVPT2G2, four bytes longer by the
# reference_kind field. For those it requires the state fields byte-identical,
# the new reference_kind to be 0 (casci), and everything from the array count
# on byte-identical -- each file's header length read off that file
# (golden_format.py arrays-at), never written out here.
#
# Each generator is single-threaded by its own import (generate_golden.py
# forces OMP/OPENBLAS/MKL_NUM_THREADS=1), so running several at once changes no
# byte -- only the wall time. The default is one at a time, the configuration
# the gradient milestone's 30-minute oracle budget is stated for; -j N is an
# opt-in speed-up.
#
# Flags:
#   -j N            generator runs at once (default 1).
#   --only REGEX    only the files whose name matches (grep -E), and only the
#                   runs that produce them.
#   --committed DIR compare against DIR instead of golden/ (a scratch copy,
#                   for the negative control). Generator INPUTS -- the
#                   --mo-from sidecars and the .srcgold's golden -- are always
#                   the repo's golden/.
#   --out DIR       regenerate into DIR and keep it (default: a mktemp
#                   directory, removed on exit). Each run's log is DIR/<run>.log.
#
# Prints one PASS:/FAIL: line per file and a summary with the wall time; exits
# 0 only when every file reproduced. It runs the Python in the LOCKED
# environment (in_locked_env, devtools/lib.sh): a missing or stale one fails.
# A MANUAL gate like the sanitizer tier, not a cpp-tier.sh phase: it costs
# minutes of CPU (docs/testing.md, "The golden regeneration gate", has the
# measurement).
#: -- help stops here --
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

jobs=1
only=""
committed="$REPO_ROOT/golden"
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    -j|--jobs)   jobs=$2; shift 2;;
    --only)      only=$2; shift 2;;
    --committed) committed=$2; shift 2;;
    --out)       out=$2; shift 2;;
    -h|--help)   usage_from_header "${BASH_SOURCE[0]}"; exit 0;;
    *)           echo "golden-repro: unknown flag: $1" >&2; exit 2;;
  esac
done
case "$committed" in /*) ;; *) committed="$PWD/$committed";; esac
[ -d "$committed" ] || { echo "golden-repro: no directory $committed" >&2; exit 2; }

if [ -n "$out" ]; then
  case "$out" in /*) ;; *) out="$PWD/$out";; esac
  mkdir -p "$out"
  scratch=$out
else
  scratch=$(mktemp -d "${TMPDIR:-/tmp}/golden-repro.XXXXXX")
  trap 'rm -rf "$scratch"' EXIT
fi

# The environment is checked ONCE, up front, so a stale one fails before ten
# generators are launched; every run below then goes through it anyway.
in_locked_env python -c "" || exit 1

gen="$REPO_ROOT/reference_data/generate_golden.py"
src="$REPO_ROOT/reference_data/sc_source_oracle.py"
G="$REPO_ROOT/golden"
S=$scratch

# One generator run per line: <run name> | <files it writes> | <command>.
# The flags are docs/reference-data.md, "Regenerating the golden file"'s, plus
# --out into the scratch directory.
declare -a run_names=() run_files=() run_cmds=()
add_run() { run_names+=("$1"); run_files+=("$2"); run_cmds+=("$3"); }
add_run cas44 "n2_ccpvdz_cas44.nevpt2gold n2_ccpvdz_cas44.gradgold" \
  "python $gen --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44 --gradgold --out $S/n2_ccpvdz_cas44.nevpt2gold"
add_run cas44_df "n2_ccpvdz_cas44_df.nevpt2gold" \
  "python $gen --df --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44_df --out $S/n2_ccpvdz_cas44_df.nevpt2gold"
add_run cas88 "n2_ccpvdz_cas88.nevpt2gold n2_ccpvdz_cas88.gradgold" \
  "python $gen --ncas 8 --nelecas 8 --name n2_ccpvdz_cas88 --gradgold --out $S/n2_ccpvdz_cas88.nevpt2gold"
add_run cas88_df "n2_ccpvdz_cas88_df.nevpt2gold" \
  "python $gen --df --ncas 8 --nelecas 8 --name n2_ccpvdz_cas88_df --out $S/n2_ccpvdz_cas88_df.nevpt2gold"
add_run cas44_casscf_mo "n2_ccpvdz_cas44_casscf.mogold" \
  "python $gen --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44_casscf --casscf --mo-out $S/n2_ccpvdz_cas44_casscf.mogold"
add_run cas44_casscf "n2_ccpvdz_cas44_casscf.nevpt2gold" \
  "python $gen --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44_casscf --mo-from $G/n2_ccpvdz_cas44_casscf.mogold --out $S/n2_ccpvdz_cas44_casscf.nevpt2gold"
add_run cas44_casscf_df_mo "n2_ccpvdz_cas44_casscf_df.mogold" \
  "python $gen --df --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44_casscf_df --casscf --mo-out $S/n2_ccpvdz_cas44_casscf_df.mogold"
add_run cas44_casscf_df "n2_ccpvdz_cas44_casscf_df.nevpt2gold" \
  "python $gen --df --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44_casscf_df --mo-from $G/n2_ccpvdz_cas44_casscf_df.mogold --out $S/n2_ccpvdz_cas44_casscf_df.nevpt2gold"
add_run cas44_src "n2_ccpvdz_cas44.srcgold" \
  "python $src --case n2_ccpvdz_cas44 --out $S/n2_ccpvdz_cas44.srcgold"
add_run cas44_src_pc "n2_ccpvdz_cas44_pc.srcgold" \
  "python $src --case n2_ccpvdz_cas44 --pc --out $S/n2_ccpvdz_cas44_pc.srcgold"
add_run cas44_src_dir "n2_ccpvdz_cas44_dir.srcgold" \
  "python $src --case n2_ccpvdz_cas44 --directional --out $S/n2_ccpvdz_cas44_dir.srcgold"
add_run cas44_src_pc_dir "n2_ccpvdz_cas44_pc_dir.srcgold" \
  "python $src --case n2_ccpvdz_cas44 --directional --pc --out $S/n2_ccpvdz_cas44_pc_dir.srcgold"

# --only: keep a run when any of its files matches.
declare -a sel=()
for i in "${!run_names[@]}"; do
  if [ -z "$only" ] || tr ' ' '\n' <<<"${run_files[$i]}" | grep -Eq -- "$only"; then
    sel+=("$i")
  fi
done
[ "${#sel[@]}" -gt 0 ] || { echo "golden-repro: --only '$only' matches nothing" >&2; exit 2; }
case "$jobs" in
  ''|*[!0-9]*|0) echo "golden-repro: -j needs a positive integer, not '$jobs'" >&2; exit 2;;
esac

echo "golden-repro: ${#sel[@]} generator run(s), -j $jobs, into $scratch"
echo "golden-repro: comparing against $committed"
t0=$(date +%s)

# Each run records its exit status and wall time beside its log.
launch() {
  local i=$1 name=${run_names[$1]} t
  t=$(date +%s)
  # shellcheck disable=SC2086  # the command is a word list built above
  ( in_locked_env ${run_cmds[$i]} >"$S/$name.log" 2>&1; echo $? >"$S/$name.rc" )
  echo $(( $(date +%s) - t )) >"$S/$name.secs"
  echo "golden-repro: [$name] done in $(cat "$S/$name.secs") s (exit $(cat "$S/$name.rc"))"
}
for i in "${sel[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$jobs" ]; do wait -n || true; done
  launch "$i" &
done
wait

# The byte comparison. Echoes the reason on failure.
arrays_at() { in_locked_env python "$REPO_ROOT/reference_data/golden_format.py" arrays-at "$1"; }
compare() {
  local c=$1 n=$2 mc mn at_c at_n kind
  [ -f "$c" ] || { echo "no committed file"; return 1; }
  [ -f "$n" ] || { echo "the generator wrote no file"; return 1; }
  cmp -s "$c" "$n" && { echo "identical, $(stat -c%s "$c") B"; return 0; }
  mc=$(head -c 8 "$c"); mn=$(head -c 8 "$n")
  if [ "$mc" = NEVPT2G1 ] && [ "$mn" = NEVPT2G2 ]; then
    at_c=$(arrays_at "$c") || { echo "committed header unreadable"; return 1; }
    at_n=$(arrays_at "$n") || { echo "regenerated header unreadable"; return 1; }
    [ "$at_n" -eq $((at_c + 4)) ] ||
      { echo "G1->G2 header lengths $at_c -> $at_n, expected +4"; return 1; }
    cmp -s -i 8:8 -n $((at_c - 8)) "$c" "$n" ||
      { echo "G1->G2: the state fields differ ($(cmp -i 8:8 -n $((at_c - 8)) "$c" "$n" 2>&1))"; return 1; }
    kind=$(od -An -tx1 -j "$at_c" -N 4 "$n" | tr -d ' \n')
    [ "$kind" = 00000000 ] ||
      { echo "G1->G2: regenerated reference_kind is $kind, not casci"; return 1; }
    cmp -s -i "$at_c:$at_n" "$c" "$n" ||
      { echo "G1->G2: the arrays differ ($(cmp -i "$at_c:$at_n" "$c" "$n" 2>&1))"; return 1; }
    echo "identical after the G1->G2 header (state fields and every array byte-identical; reference_kind casci), $(stat -c%s "$c") B"
    return 0
  fi
  echo "differs: $mc vs $mn, $(stat -c%s "$c") vs $(stat -c%s "$n") B; $(cmp "$c" "$n" 2>&1 | head -1)"
  return 1
}

npass=0
nfail=0
for i in "${sel[@]}"; do
  name=${run_names[$i]}
  rc=$(cat "$S/$name.rc" 2>/dev/null || echo "?")
  for f in ${run_files[$i]}; do
    if [ -n "$only" ] && ! grep -Eq -- "$only" <<<"$f"; then continue; fi
    if [ "$rc" != 0 ]; then
      echo "FAIL: $f -- generator run '$name' exited $rc; last lines of its log:"
      tail -5 "$S/$name.log" | sed 's/^/    /'
      nfail=$((nfail + 1))
    elif why=$(compare "$committed/$f" "$S/$f"); then
      echo "PASS: $f -- $why (run '$name', $(cat "$S/$name.secs") s)"
      npass=$((npass + 1))
    else
      echo "FAIL: $f -- $why"
      nfail=$((nfail + 1))
    fi
  done
done

wall=$(( $(date +%s) - t0 ))
if [ "$nfail" -eq 0 ]; then
  echo "golden-repro: PASS $npass/$npass reproduced byte for byte, ${wall} s wall (-j $jobs)"
  exit 0
fi
echo "golden-repro: FAIL $nfail of $((npass + nfail)) did not reproduce, ${wall} s wall (-j $jobs)"
exit 1
