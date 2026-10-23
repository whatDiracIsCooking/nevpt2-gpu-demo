#!/usr/bin/env bash
# Regenerate the committed N2 CAS(4,4) / CAS(8,8) references, and the C1
# case's three HSOH CAS(6,6) files (#213), and the two CASSCF response
# sidecars (#219), with the generators' own settings, into a scratch directory, and compare each with
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
# What it regenerates (thirty files, twenty-two generator runs):
#   generate_golden.py     n2_ccpvdz_cas44 / cas88 .nevpt2gold + .gradgold,
#                          and cas44's full MO-integral tensor sidecar
#                          n2_ccpvdz_cas44.erigold (#180, --erigold, in the
#                          same run: it changes no byte of the other two),
#                          their _df twins (each with its density-fitted
#                          NEVPT2F1 .gradgold, #164), the two .mogold sidecars (a fresh
#                          --casscf each) and the two *_casscf[_df] goldens
#                          (--mo-from the COMMITTED sidecar, as
#                          docs/reference-data.md's commands do, so a sidecar
#                          that drifted fails as itself, not as its golden too),
#                          the _casscf one with its full tensor
#                          n2_ccpvdz_cas44_casscf.erigold (#222) and the
#                          _casscf_df one with its NEVPT2F1 .gradgold (#220)
#   sc_source_oracle.py    n2_ccpvdz_cas44.srcgold and n2_ccpvdz_cas44_pc.srcgold,
#                          and their directional twins (#105) *_dir.srcgold /
#                          *_pc_dir.srcgold (all four read
#                          golden/n2_ccpvdz_cas44.nevpt2gold, the committed
#                          one, as their input), and the same four on CASSCF
#                          orbitals (#117), n2_ccpvdz_cas44_casscf[_pc][_dir]
#                          .srcgold (--mo-from the COMMITTED sidecar, against
#                          the committed n2_ccpvdz_cas44_casscf.nevpt2gold)
#   nevpt2_numgrad.py      the four grad_num* arrays in n2_ccpvdz_cas44_df.gradgold
#                          (#167): run inside the cas44_df run, right after its
#                          generator and on the sidecar that generator just
#                          wrote, as docs/reference-data.md's two commands do,
#                          so the file compared is the committed one in full
#                          (the 12 --df --gradgold arrays AND the numerical
#                          gradient), not the 12 alone. The reference state is
#                          checked against the COMMITTED _df golden. LiF's and
#                          acrolein's conventional grad_num sidecars are not
#                          covered here.
#   the C1 case (#213)     hsoh_def2svp_cas66_casscf.mogold (a fresh --casscf),
#                          its golden and .gradgold (--mo-from the COMMITTED
#                          .mogold, --gradgold), then nevpt2_numgrad.py
#                          --reference casscf's four grad_num* arrays into
#                          that sidecar: every displaced state re-optimised
#                          cold, deterministic like the reference, so the
#                          sidecar is compared whole
#                          (docs/reference-data.md, "HSOH, the C1 case").
#   casscf_nuclear_hessian.py  n2_ccpvdz_cas44_casscf.respgold and
#                          hsoh_def2svp_cas66_casscf.respgold (#219), the
#                          CASSCF response sidecars, --mo-from the COMMITTED
#                          .mogold and held to the committed golden's frame
#                          (docs/reference-data.md, "The analytic CASSCF
#                          nuclear Hessian").
#   integral_perturbation.py  n2_ccpvdz_cas44_casscf[_df].pertgold (#220),
#                          the integral perturbation sidecars, --mo-from the
#                          COMMITTED .mogold and held to the committed
#                          golden's ci and gradient sidecar's mo_coeff
#                          (docs/reference-data.md, "The integral
#                          perturbation sidecar").
#
# What it deliberately does NOT regenerate: golden/n2_ccpvdz_cas44_casscf.gradgold,
# the CASSCF-reference gradient sidecar (#120). An orbital optimiser sits
# upstream of it (the .mogold), so its contract is reproducibility to a
# tolerance, not byte for byte, and this gate makes no claim over it
# (docs/reference-data.md, "The CASSCF-reference gradient sidecar"). The
# cas44_casscf run below writes one (since #222 it passes --gradgold
# --erigold, the full tensor needing the sidecar's run) but does not compare
# it; the golden it writes is byte-identical either way.
# (The HSOH sidecar above is the exception by construction: it is built from
# the committed .mogold, as every --mo-from file here is, so the byte claim
# is the same one its golden's is -- and so is the density-fitted
# n2_ccpvdz_cas44_casscf_df.gradgold, #220's, for the same reason.)
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
num="$REPO_ROOT/reference_data/nevpt2_numgrad.py"
G="$REPO_ROOT/golden"
S=$scratch

# One generator run per line: <run name> | <files it writes> | <command>.
# The flags are docs/reference-data.md, "Regenerating the golden file"'s, plus
# --out into the scratch directory.
# An optional fourth field is a second command run after the first succeeds,
# in the same run (same log, one exit status): a step that rewrites a file
# the first one wrote.
declare -a run_names=() run_files=() run_cmds=() run_posts=()
add_run() { run_names+=("$1"); run_files+=("$2"); run_cmds+=("$3"); run_posts+=("${4:-}"); }
add_run cas44 "n2_ccpvdz_cas44.nevpt2gold n2_ccpvdz_cas44.gradgold n2_ccpvdz_cas44.erigold" \
  "python $gen --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44 --gradgold --erigold --out $S/n2_ccpvdz_cas44.nevpt2gold"
add_run cas44_df "n2_ccpvdz_cas44_df.nevpt2gold n2_ccpvdz_cas44_df.gradgold" \
  "python $gen --df --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44_df --gradgold --out $S/n2_ccpvdz_cas44_df.nevpt2gold" \
  "python $num --df --case n2_ccpvdz_cas44 --sidecar $S/n2_ccpvdz_cas44_df.gradgold"
add_run cas88 "n2_ccpvdz_cas88.nevpt2gold n2_ccpvdz_cas88.gradgold" \
  "python $gen --ncas 8 --nelecas 8 --name n2_ccpvdz_cas88 --gradgold --out $S/n2_ccpvdz_cas88.nevpt2gold"
add_run cas88_df "n2_ccpvdz_cas88_df.nevpt2gold n2_ccpvdz_cas88_df.gradgold" \
  "python $gen --df --ncas 8 --nelecas 8 --name n2_ccpvdz_cas88_df --gradgold --out $S/n2_ccpvdz_cas88_df.nevpt2gold"
add_run cas44_casscf_mo "n2_ccpvdz_cas44_casscf.mogold" \
  "python $gen --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44_casscf --casscf --mo-out $S/n2_ccpvdz_cas44_casscf.mogold"
# With --gradgold --erigold since #222: the full tensor is compared, the
# gradient sidecar the run also writes is not (header, "What it deliberately
# does NOT regenerate").
add_run cas44_casscf "n2_ccpvdz_cas44_casscf.nevpt2gold n2_ccpvdz_cas44_casscf.erigold" \
  "python $gen --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44_casscf --mo-from $G/n2_ccpvdz_cas44_casscf.mogold --gradgold --erigold --out $S/n2_ccpvdz_cas44_casscf.nevpt2gold"
add_run cas44_casscf_df_mo "n2_ccpvdz_cas44_casscf_df.mogold" \
  "python $gen --df --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44_casscf_df --casscf --mo-out $S/n2_ccpvdz_cas44_casscf_df.mogold"
add_run cas44_casscf_df "n2_ccpvdz_cas44_casscf_df.nevpt2gold n2_ccpvdz_cas44_casscf_df.gradgold" \
  "python $gen --df --ncas 4 --nelecas 4 --name n2_ccpvdz_cas44_casscf_df --mo-from $G/n2_ccpvdz_cas44_casscf_df.mogold --gradgold --out $S/n2_ccpvdz_cas44_casscf_df.nevpt2gold"
# The C1 case (#213): HSOH CAS(6,6), CASSCF orbitals only. The .mogold is a
# fresh --casscf; the golden and its gradient sidecar come from --mo-from the
# COMMITTED .mogold, and the sidecar's grad_num from a --reference casscf
# sweep run right after it, on the sidecar it just wrote (as cas44_df's
# post-step does). That sweep re-optimises 72 displaced states, so it runs
# --workers 12 (bit-identical to one worker); the reference it re-optimises
# must reproduce the COMMITTED .mogold bit for bit, or it refuses. --atom is
# the relative path the committed .mogold records (this script runs at the
# repo root): the sidecar keeps the spec as typed, so another spelling of the
# same path would move its bytes.
add_run hsoh_casscf_mo "hsoh_def2svp_cas66_casscf.mogold" \
  "python $gen --atom reference_data/geometries/hsoh.xyz --basis def2-svp --ncas 6 --nelecas 6 --name hsoh_def2svp_cas66_casscf --casscf --mo-out $S/hsoh_def2svp_cas66_casscf.mogold"
add_run hsoh_casscf "hsoh_def2svp_cas66_casscf.nevpt2gold hsoh_def2svp_cas66_casscf.gradgold" \
  "python $gen --atom reference_data/geometries/hsoh.xyz --basis def2-svp --ncas 6 --nelecas 6 --name hsoh_def2svp_cas66_casscf --mo-from $G/hsoh_def2svp_cas66_casscf.mogold --gradgold --out $S/hsoh_def2svp_cas66_casscf.nevpt2gold" \
  "python $num --case hsoh_def2svp_cas66 --reference casscf --workers 12 --states-dir $S/hsoh_states --sidecar $S/hsoh_def2svp_cas66_casscf.gradgold"
# The CASSCF response sidecars (#219): the 3N responses and the analytic
# CASSCF Hessian of each committed _casscf state, from --mo-from the COMMITTED
# .mogold, held (and refused otherwise) to the committed golden's ci and
# gradient sidecar's mo_coeff bit for bit. --no-ladders: the sidecar is the
# solve alone; the re-optimised ladders that check it write nothing.
hes="$REPO_ROOT/reference_data/casscf_nuclear_hessian.py"
add_run cas44_casscf_resp "n2_ccpvdz_cas44_casscf.respgold" \
  "python $hes --case n2_ccpvdz_cas44 --no-ladders --write $S/n2_ccpvdz_cas44_casscf.respgold"
add_run hsoh_casscf_resp "hsoh_def2svp_cas66_casscf.respgold" \
  "python $hes --case hsoh_def2svp_cas66 --no-ladders --write $S/hsoh_def2svp_cas66_casscf.respgold"
# The integral perturbation sidecars (#220): every coordinate's perturbed
# Hamiltonian of the committed _casscf[_df] state, in its gradient sidecar's
# inventory, from --mo-from the COMMITTED .mogold and held (and refused
# otherwise) to the committed golden's ci and gradient sidecar's mo_coeff bit
# for bit. --no-ladders: the sidecar is the analytic derivative alone; the
# ladders that check it write nothing.
ipt="$REPO_ROOT/reference_data/integral_perturbation.py"
add_run cas44_casscf_pert "n2_ccpvdz_cas44_casscf.pertgold" \
  "python $ipt --case n2_ccpvdz_cas44 --no-ladders --write $S/n2_ccpvdz_cas44_casscf.pertgold"
add_run cas44_casscf_df_pert "n2_ccpvdz_cas44_casscf_df.pertgold" \
  "python $ipt --case n2_ccpvdz_cas44 --df --no-ladders --write $S/n2_ccpvdz_cas44_casscf_df.pertgold"
add_run cas44_src "n2_ccpvdz_cas44.srcgold" \
  "python $src --case n2_ccpvdz_cas44 --out $S/n2_ccpvdz_cas44.srcgold"
add_run cas44_src_pc "n2_ccpvdz_cas44_pc.srcgold" \
  "python $src --case n2_ccpvdz_cas44 --pc --out $S/n2_ccpvdz_cas44_pc.srcgold"
add_run cas44_src_dir "n2_ccpvdz_cas44_dir.srcgold" \
  "python $src --case n2_ccpvdz_cas44 --directional --out $S/n2_ccpvdz_cas44_dir.srcgold"
add_run cas44_src_pc_dir "n2_ccpvdz_cas44_pc_dir.srcgold" \
  "python $src --case n2_ccpvdz_cas44 --directional --pc --out $S/n2_ccpvdz_cas44_pc_dir.srcgold"
add_run cas44_casscf_src "n2_ccpvdz_cas44_casscf.srcgold" \
  "python $src --case n2_ccpvdz_cas44_casscf --mo-from $G/n2_ccpvdz_cas44_casscf.mogold --out $S/n2_ccpvdz_cas44_casscf.srcgold"
add_run cas44_casscf_src_pc "n2_ccpvdz_cas44_casscf_pc.srcgold" \
  "python $src --case n2_ccpvdz_cas44_casscf --mo-from $G/n2_ccpvdz_cas44_casscf.mogold --pc --out $S/n2_ccpvdz_cas44_casscf_pc.srcgold"
add_run cas44_casscf_src_dir "n2_ccpvdz_cas44_casscf_dir.srcgold" \
  "python $src --case n2_ccpvdz_cas44_casscf --mo-from $G/n2_ccpvdz_cas44_casscf.mogold --directional --out $S/n2_ccpvdz_cas44_casscf_dir.srcgold"
add_run cas44_casscf_src_pc_dir "n2_ccpvdz_cas44_casscf_pc_dir.srcgold" \
  "python $src --case n2_ccpvdz_cas44_casscf --mo-from $G/n2_ccpvdz_cas44_casscf.mogold --directional --pc --out $S/n2_ccpvdz_cas44_casscf_pc_dir.srcgold"

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
  # shellcheck disable=SC2086
  ( { in_locked_env ${run_cmds[$i]} &&
      { [ -z "${run_posts[$i]}" ] || in_locked_env ${run_posts[$i]}; }
    } >"$S/$name.log" 2>&1; echo $? >"$S/$name.rc" )
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
