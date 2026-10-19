#!/usr/bin/env bash
# EXPLORATORY: stub test of run_resultF.sh's scheduler (not results; trains nothing; touches nothing outside
# <resultF>/smoke/sched_test/). It copies run_resultF.sh with three swaps, each checked to apply exactly once:
#   rfpy (driver/rf.py)  -> marker files in state/ (done_<job>), untouched always true
#   run_job              -> stub_job: a short sleep, then done_<job>, or exit 7 for a job named in FAILJOBS
#                           (f1train_<q> marks A2 trained first, as result F's script trains A2 before B2 and C2,
#                           unless NO_A2=<q>)
#   gates.py             -> passes
# and runs it with a fake sandbox (IW=<test>/fake_iw) through these scenarios:
#   A  an optional job fails; data seeds 1-4 and the reports already exist (as data_seeds1to4.sh left them), and a
#      process runs experiments.shift_iw.run outside the sandbox: DONE, naming the optional failure; no data or report
#      job started; evalF complete
#   B  f1train fails after A2 (B2 or C2 failed): evalF runs --allow-missing (partial); not DONE
#   C  resume of B with nothing failing: only the unfinished jobs run, evalF complete, DONE
#   D  f1train fails before A2: evalF blocked; not DONE
#   E  a process runs experiments.shift_iw.run with its working directory in the sandbox, then one runs
#      data_seeds1to4.sh: run and dry both refuse (exit 75)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RF=$(cd "$HERE/.." && pwd)
T=$HERE/sched_test
rm -rf "$T"
mkdir -p "$T"
python3 - "$RF/run_resultF.sh" "$T/run_test.sh" <<'EOF'
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
swaps = [
    ('rfpy() { "$PY" -I -B "$RF/driver/rf.py" "$@"; }   # the standard-library helper (driver/rf.py)\n',
     '''rfpy() {  # STUB: driver/rf.py replaced by marker files in state/
  case $1 in
    done) [ -f "$STATE/done_$2" ] ;;
    untouched) echo "untouched: True (stub)"; return 0 ;;
    *) echo "stub rf.py $*"; return 0 ;;
  esac
}
stub_job() {  # STUB job: JOB CPU MODE
  local j=$1 q
  sleep 0.$((RANDOM % 3 + 1))
  echo "[$(date +%T)] $j on cpu $2${3:+ ($3)}" >> "$LOGS/order.log"
  case $j in f1train_*) q=${j#f1train_}; [ "${NO_A2:-}" = "$q" ] || touch "$STATE/done_a2ok_$q" ;; esac
  case " ${FAILJOBS:-} " in *" $j "*) return 7 ;; esac
  if [ "$j" = evalF ]; then
    if [ "$3" = partial ]; then touch "$STATE/done_evalF-partial"; echo "**Overall: STUB (partial)**" > "$RF/results/COMPARE.md"
    else touch "$STATE/done_evalF"; echo "**Overall: STUB (complete)**" > "$RF/results/COMPARE.md"; fi
  else touch "$STATE/done_$j"; fi
  return 0
}
'''),
    ('    run_job "$j" "$cpu" "$mode" &\n', '    stub_job "$j" "$cpu" "$mode" &\n'),
    ('  py_job "${c[0]}" "$LOGS/gates.log" "$RF/driver/gates.py" "$@"\n', '  echo "stub gates" >> "$LOGS/gates.log"\n'),
]
for a, b in swaps:
    assert s.count(a) == 1, f"swap does not apply exactly once: {a!r}"
    s = s.replace(a, b)
open(dst, "w").write(s)
EOF
FAKE=$T/fake_iw
mkdir -p "$FAKE/repo/runs"
fails=0
check() { if eval "$2"; then echo "  ok   $1"; else echo "  FAIL $1"; fails=$((fails + 1)); fi; }
newdir() { rm -rf "$T/$1"; mkdir -p "$T/$1/state"; cp "$T/run_test.sh" "$T/$1/run_test.sh"; }
predata() {  # what data_seeds1to4.sh leaves: data seeds 1-4 and the report of each quality, finished
  local q s
  for q in medium-expert expert; do for s in 1 2 3 4; do touch "$1/state/done_data_${q}_s$s"; done; touch "$1/state/done_report_$q"; done
}
run() { local d=$1; shift; (cd "$d" && env IW="$FAKE" "$@" bash run_test.sh "${MODE:-run}" > out.txt 2>&1) && echo 0 || echo $?; }

echo "A: an optional job fails; data and reports already there; a stranger outside the sandbox"
newdir A; predata "$T/A"
(cd /tmp && exec bash -c 'sleep 60; true' -m experiments.shift_iw.run) & STRANGER=$!
rc=$(run "$T/A" FAILJOBS="esttrain_expert")
kill $STRANGER 2>/dev/null || true
check "exit 0" '[ "$rc" = 0 ]'
check "DONE written" '[ -f "$T/A/DONE" ]'
check "DONE names the optional failure" 'grep -q "esttrain_expert(failed)" "$T/A/DONE" && grep -q "esteval_expert(blocked)" "$T/A/DONE"'
check "no data or report job started" '! grep -Eq " (data_|report_)" "$T/A/logs/order.log"'
check "10 data and report jobs skipped as finished" '[ "$(grep -Ec "skip \(finished\): (data_|report_)" "$T/A/logs/run.log")" = 10 ]'
check "evalF complete" 'grep -q "evalF on cpu [0-9]* (full)" "$T/A/logs/order.log" && [ -f "$T/A/state/done_evalF" ]'
check "evalF started last" '[ "$(tail -n 1 "$T/A/logs/order.log" | grep -c evalF)" = 1 ]'
check "job times recorded" '[ "$(wc -l < "$T/A/state/job_times.tsv")" = "$(wc -l < "$T/A/logs/order.log")" ]'

echo "B: f1train fails after A2"
newdir B; predata "$T/B"
rc=$(run "$T/B" FAILJOBS="f1train_medium-expert")
check "exit 1" '[ "$rc" = 1 ]'
check "no DONE" '[ ! -f "$T/B/DONE" ]'
check "f1eval blocked" 'grep -q "BLOCKED (an input job failed): f1eval_medium-expert" "$T/B/logs/run.log"'
check "evalF ran --allow-missing" 'grep -q "evalF on cpu [0-9]* (partial)" "$T/B/logs/order.log" && [ -f "$T/B/state/done_evalF-partial" ]'
check "not DONE names f1train" 'grep -q "not DONE: failed or blocked core jobs: .*f1train_medium-expert(failed)" "$T/B/logs/run.log"'

echo "C: resume of B"
n_before=$(wc -l < "$T/B/logs/order.log")
rc=$(run "$T/B" FAILJOBS="")
check "exit 0" '[ "$rc" = 0 ]'
check "DONE written" '[ -f "$T/B/DONE" ]'
check "only f1train, f1eval (medium-expert) and evalF ran" '[ "$(tail -n +$((n_before + 1)) "$T/B/logs/order.log" | sed "s/^\[[^]]*\] //; s/ on cpu.*//" | sort | tr "\n" " ")" = "evalF f1eval_medium-expert f1train_medium-expert " ]'
check "evalF complete" 'tail -n 1 "$T/B/logs/order.log" | grep -q "evalF on cpu [0-9]* (full)"'

echo "D: f1train fails before A2"
newdir D; predata "$T/D"
rc=$(run "$T/D" FAILJOBS="f1train_expert" NO_A2=expert)
check "exit 1" '[ "$rc" = 1 ]'
check "evalF blocked, never started" 'grep -q "BLOCKED (an input job failed): evalF" "$T/D/logs/run.log" && ! grep -q evalF "$T/D/logs/order.log"'
check "no DONE" '[ ! -f "$T/D/DONE" ]'

echo "E: another process works in the sandbox"
newdir E
(cd "$FAKE/repo" && exec bash -c 'sleep 60; true' -m experiments.shift_iw.run) & BUSY=$!
sleep 1
rc=$(run "$T/E")
check "run refused (exit 75)" '[ "$rc" = 75 ] && grep -q "REFUSED: another process works in the sandbox" "$T/E/out.txt"'
rc=$(MODE=dry run "$T/E")
check "dry refused (exit 75)" '[ "$rc" = 75 ]'
check "nothing started" '[ ! -f "$T/E/logs/order.log" ]'
kill $BUSY 2>/dev/null || true
(cd /tmp && exec bash -c 'sleep 60; true' "$RF/data_seeds1to4.sh") & BUSY=$!
sleep 1
rc=$(run "$T/E")
check "run refused while data_seeds1to4.sh runs (exit 75)" '[ "$rc" = 75 ] && grep -q "data_seeds1to4.sh" "$T/E/out.txt"'
kill $BUSY 2>/dev/null || true

echo "stub scheduler test: $([ $fails = 0 ] && echo PASSED || echo "FAILED ($fails)") ($(date '+%F %T'))"
[ $fails = 0 ]
