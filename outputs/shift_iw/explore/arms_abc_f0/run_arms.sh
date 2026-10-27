#!/usr/bin/env bash
# EXPLORATORY: arms A (credible-target pull), B (certified throttle ceiling), C (guarded release) and result F's no-band
# control F0, on the weighting test (spec version 5) in the cloud sandbox. Seen seeds 0-4 only; nothing here is
# confirmatory (that needs fresh seeds 15-19 and its own pre-registration). Pass marks and kill rules:
# routes2/arms/PASSMARKS_arms_ABC_and_F_control.md, frozen with a sha256sum sidecar before this code was written; every
# command checks it first and refuses on any difference.
#
#   bash run_arms.sh start      the seed-0 screen: gates and unit checks, wait for the result F rerun, data seeds 1-4
#                               and reports for random, medium-replay and medium (D1 checked against the committed
#                               reports), room checks and bands for their seed 0, then A, B, C and F0 on seed 0 (only
#                               the quantities the kill rules read, plus F0's seed-0 score), then SCREEN_seed0.md. Stops
#                               there.
#   bash run_arms.sh continue   seeds 1-4 for every arm the screen did not stop (refuses unless SCREEN_seed0.json is
#                               complete and verified), then every kill rule across the five seeds (KILLS.md), then
#                               scoring: returns on the three banks and the four situations, every mark (RESULTS.md)
#   bash run_arms.sh status     what is finished, running, failed, blocked or pending (reads only)
#   bash run_arms.sh stop       stop a background run and its jobs (a stopped job restarts from scratch on resume)
#   bash run_arms.sh dry        read-only checks of imports, gates, paths and refusals; trains nothing
#   bash run_arms.sh test       the unit checks: synthetic (test_arms.py) and real seed-0 ones (checks.py)
#   bash run_arms.sh smoke      every job script end to end on real seed-0 inputs with a few steps, 50 starts and 3,000
#                               tries, into state/smoke only (SMOKE TEST, deleted afterwards; not results)
#   setsid nohup bash run_arms.sh start > logs/nohup_start.out 2>&1 < /dev/null &      (background)
#
# start and continue are resumable: finished work is never redone (each job's output is checked by code/rt.py done).
# The screen and the kill checks count as done only when complete and built on the current version of every record
# they read; scoring starts only when every evaluation is done and counts as done only for the current kill checks. A
# job whose input failed for good (a TD3+BC host that went non-finite, an invalid band) is settled: the arm jobs record
# the stop, so the kill checks can complete and only the marks that need that seed are not met.
# They wait (WAIT=1, the default; WAIT=0 refuses instead) while the result F rerun holds its lock
# (resultF/state/run.lock, read through /proc/locks, never opened) or while any process running
# experiments.shift_iw.run, command_le_train_f1.py or run_resultF.sh has its working directory in the sandbox
# (run_resultF.sh: anywhere). They never touch resultF/state/ or resultF/logs/.
#
# Environment (defaults in brackets): CPUS ["1 2 3"; CPU 0 is left to other sessions], SRC [/home/user/bca, read only],
# IW [<scratchpad>/iw_rerun], PY [/tmp/claude-0/bcavenv/bin/python3], WAIT [1].
# Every job: cd <IW>/repo; PYTHONPATH=<IW>/repo:<IW>/explore PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
# OPENBLAS_CORETYPE=Haswell NPY_DISABLE_CPU_FEATURES=<the AVX-512 list> XLA_FLAGS=--xla_cpu_max_isa=AVX2 nice -n 10
# taskset -c <one CPU> python -B ... (run_seed0.sh's and run_resultF.sh's settings: one CPU and one thread per job).
# Writes only: routes2/arms/{state,logs,SCREEN*,KILLS*,RESULTS.md,results.json*} and new files under <IW>/repo/runs/
# (the package's own stage records with their .sha256 record checksums, result F's unchanged script's A2/B2/C2 of
# seeds 1-4 under channel/train_f1/, this run's arm records under shift_iw_v5_explore/arms_abc_f0/, and one
# EXPLORATORY_arms_abc_f0.txt label in each runs/ root listing every file this run created; the list is refreshed after
# every job and again when the runner finishes, is stopped or exits).
set -euo pipefail
ARMS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE=$ARMS/code
SRC=${SRC:-/home/user/bca}
IW=${IW:-$(cd "$ARMS/../../iw_rerun" && pwd)}
RF=$(cd "$ARMS/../.." && pwd)/resultF
PY=${PY:-/tmp/claude-0/bcavenv/bin/python3}
CPUS=${CPUS:-"1 2 3"}
WAIT=${WAIT:-1}
NPY_OFF="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
ARITH_ENV=(OPENBLAS_CORETYPE=Haswell "NPY_DISABLE_CPU_FEATURES=$NPY_OFF" XLA_FLAGS=--xla_cpu_max_isa=AVX2)
CHAN=$SRC/outputs/shift_iw/explore/channel
STATE=$ARMS/state
LOGS=$ARMS/logs
export ARMS_SRC="$SRC" ARMS_IW="$IW"
mkdir -p "$STATE" "$LOGS"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/run.log"; }
rt() { "$PY" -I -B "$CODE/rt.py" "$@"; }

py_job() {  # py_job CPU LOG python-arguments... (foreground; pinned to one CPU; no bytecode written anywhere)
  local cpu=$1 logf=$2
  shift 2
  (cd "$IW/repo" && exec env "${ARITH_ENV[@]}" PYTHONPATH="$IW/repo:$IW/explore" PYTHONDONTWRITEBYTECODE=1 \
     JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 ARMS_SRC="$SRC" ARMS_IW="$IW" TMPDIR="$STATE/tmp" \
     nice -n 10 taskset -c "$cpu" "$PY" -B "$@") >> "$logf" 2>&1
}

# ------------------------------------------------------------------------------------------------ the sandbox is busy?

lock_held() {  # is a lock held on this file? read-only: /proc/locks by inode (the file is never opened)
  local ino
  [ -e "$1" ] || return 1
  ino=$(stat -c %i "$1") || return 1
  grep -qE "^[0-9]+: +(FLOCK|POSIX|OFDLCK) +[A-Z]+ +[A-Z]+ +[0-9]+ +[0-9a-f]+:[0-9a-f]+:$ino " /proc/locks
}

sandbox_busy() {  # prints and succeeds if the result F rerun or another process works in the sandbox
  local p a m r cwd found="" iw_real
  iw_real=$(cd "$IW" 2>/dev/null && pwd -P) || iw_real=$IW
  if lock_held "$RF/state/run.lock"; then found="$found the result F rerun holds $RF/state/run.lock;"; fi
  for p in /proc/[0-9]*; do
    p=${p#/proc/}
    [ "$p" = "$$" ] && continue
    m=0; r=0
    while IFS= read -r -d '' a; do
      case $a in
        experiments.shift_iw.run|*command_le_train_f1.py|*command_le_train.py|*command_le_train_est.py) m=1 ;;
        run_resultF.sh|*/run_resultF.sh) r=1 ;;
      esac
    done 2>/dev/null < "/proc/$p/cmdline" || continue
    if [ $m = 1 ]; then
      cwd=$(readlink "/proc/$p/cwd" 2>/dev/null) || continue
      case "$cwd/" in "$IW"/*|"$iw_real"/*) found="$found pid $p ($(tr '\0' ' ' < "/proc/$p/cmdline" | cut -c1-80) in $cwd);" ;; esac
    fi
    [ $r = 1 ] && found="$found pid $p (run_resultF.sh);"
  done
  [ -n "$found" ] && { echo "$found"; return 0; }
  return 1
}

wait_for_sandbox() {
  local busy
  while busy=$(sandbox_busy); do
    if [ "$WAIT" != 1 ]; then say "REFUSED: another process works in the sandbox:$busy"; exit 75; fi
    say "waiting: another process works in the sandbox:$busy (checking again in 60 s)"
    sleep 60
  done
}

# ------------------------------------------------------------------------------------------------ jobs

run_job() {  # run_job JOB CPU: one job, then its done check; exit 0 only if its output is finished and current
  local j=$1 cpu=$2 kind=${1%%_*} rest q k arm rc=0 logf=$LOGS/$1.log
  rest=${j#*_}; q=${rest%_s*}; k=${rest##*_s}
  echo "[$(date '+%F %T')] start on cpu $cpu (pid $BASHPID)" >> "$logf"
  if rt done "$j" 2>> "$logf"; then   # settled meanwhile (e.g. its host failed for good): nothing to run
    echo "[$(date '+%F %T')] settled without running (code/rt.py done)" >> "$logf"
    return 0
  fi
  case $kind in
    data) py_job "$cpu" "$logf" -m experiments.shift_iw.run data --quality "$q" --seeds "$k" || rc=$? ;;
    report) py_job "$cpu" "$logf" -m experiments.shift_iw.run report --quality "$q" || rc=$? ;;
    d1) rt d1 "$q" >> "$logf" 2>&1 || rc=$? ;;
    host) py_job "$cpu" "$logf" -m experiments.shift_iw.run host --quality "$q" --seeds "$k" --kinds gap || rc=$? ;;
    room) py_job "$cpu" "$logf" -m experiments.shift_iw.run room --quality "$q" --seeds "$k" || rc=$? ;;
    band) py_job "$cpu" "$logf" -m experiments.shift_iw.run band --quality "$q" --seeds "$k" --kinds gap || rc=$? ;;
    A|B|C) py_job "$cpu" "$logf" "$CODE/arm_$(echo "$kind" | tr 'A-Z' 'a-z').py" --quality "$q" --seed "$k" || rc=$? ;;
    F0) py_job "$cpu" "$logf" "$CODE/arm_f0.py" --quality "$q" --seed "$k" || rc=$? ;;
    A2) rt quarantine "$j" >> "$logf" 2>&1 || rc=$?
        [ $rc -eq 0 ] && { py_job "$cpu" "$logf" "$CHAN/command_le_train_f1.py" train "$q" "$k" || rc=$?; } ;;
    A2eval) py_job "$cpu" "$logf" "$CHAN/command_le_train_f1.py" evaluate "$q" "$k" || rc=$? ;;
    screen) py_job "$cpu" "$logf" "$CODE/screen.py" || rc=$? ;;
    kills) py_job "$cpu" "$logf" "$CODE/kills.py" || rc=$? ;;
    eval) arm=${rest%%_*}; rest=${rest#*_}; q=${rest%_s*}; k=${rest##*_s}
          py_job "$cpu" "$logf" "$CODE/score.py" eval --arm "$arm" --quality "$q" --seed "$k" || rc=$? ;;
    marks) py_job "$cpu" "$logf" "$CODE/score.py" marks || rc=$? ;;
    *) echo "unknown job $j" >> "$logf"; return 64 ;;
  esac
  if [ $rc -eq 0 ] && rt done "$j" 2>> "$logf"; then
    echo "[$(date '+%F %T')] finished" >> "$logf"
    return 0
  fi
  [ $rc -eq 0 ] && rc=70
  echo "[$(date '+%F %T')] FAILED (exit $rc)" >> "$logf"
  return $rc
}

# ---- scheduler (begin)
declare -A ST=() DEPS=() CPU_OF=() NAME_OF=()
FREE=()
JOBS=()

# The screen and the kill checks run once their inputs settle (even after an input job failed), so their reports say
# what is missing; they count as done only when complete and current (rt.py done), so a resumed run redoes them after
# the failed inputs finish. Scoring (marks) is not settling: it waits until every evaluation is done.
is_settling() { case $1 in screen|kills) return 0 ;; *) return 1 ;; esac; }

deps_state() {  # ok | wait | blocked
  local d any_bad=0
  for d in ${DEPS[$1]//,/ }; do
    case ${ST[$d]:-absent} in
      done) ;;
      failed|blocked|absent) any_bad=1 ;;
      *) echo wait; return ;;
    esac
  done
  if [ $any_bad = 1 ] && ! is_settling "$1"; then echo blocked; return; fi
  echo ok
}

start_ready() {
  local j cpu pid st
  for j in "${JOBS[@]}"; do
    [ ${#FREE[@]} -gt 0 ] || return 0
    [ "${ST[$j]}" = pending ] || continue
    st=$(deps_state "$j")
    case $st in
      ok) ;;
      blocked) ST[$j]=blocked; say "BLOCKED (an input job failed): $j"; continue ;;
      *) continue ;;
    esac
    cpu=${FREE[0]}
    FREE=("${FREE[@]:1}")
    run_job "$j" "$cpu" &
    pid=$!
    CPU_OF[$pid]=$cpu
    NAME_OF[$pid]=$j
    ST[$j]=running
    echo "$pid $cpu $(date +%s)" > "$STATE/$j.pid"
    say "started: $j on cpu $cpu (pid $pid)"
  done
}

reap() {
  local pid rc=0 j t0=""
  wait -n -p pid || rc=$?
  j=${NAME_OF[$pid]}
  if [ $rc -eq 0 ]; then ST[$j]=done; rm -f "$STATE/$j.failed"; say "done: $j"
  else ST[$j]=failed; echo "$rc" > "$STATE/$j.failed"; say "FAILED (exit $rc): $j (log $LOGS/$j.log)"; fi
  read -r _ _ t0 2>/dev/null < "$STATE/$j.pid" || true
  printf '%s\t%s\t%s\t%s\t%s\n' "$j" "${CPU_OF[$pid]}" "${t0:-0}" "$(date +%s)" "$rc" >> "$STATE/job_times.tsv"
  rm -f "$STATE/$j.pid"
  FREE+=("${CPU_OF[$pid]}")
  unset "CPU_OF[$pid]" "NAME_OF[$pid]"
  refresh_label
}

load_jobs() {  # load_jobs start|continue
  local line j d
  JOBS=(); ST=(); DEPS=()
  while IFS=$'\t' read -r j d; do
    [ -n "$j" ] || continue
    JOBS+=("$j"); DEPS[$j]=$d
    if rt done "$j" 2>/dev/null; then ST[$j]=done; else ST[$j]=pending; fi
  done < <(rt jobs "$1")
  [ ${#JOBS[@]} -gt 0 ]
}

schedule() {
  read -r -a FREE <<< "$CPUS"
  local j
  for j in "${JOBS[@]}"; do [ "${ST[$j]}" = done ] && say "skip (finished): $j"; done
  while :; do
    start_ready
    [ ${#CPU_OF[@]} -gt 0 ] || break
    reap
  done
}
# ---- scheduler (end)

refresh_label() {  # the EXPLORATORY label's file list, after every job (only once this run has a baseline)
  [ -f "$STATE/baseline_sandbox.json" ] || return 0
  rt label >> "$LOGS/label.log" 2>&1 || say "the label's file list could not be refreshed (logs/label.log)"
}

FINISHED=0
finish_up() {  # once per run: the untouched check and the label (also when the runner is stopped or exits early)
  [ "$FINISHED" = 1 ] && return 0
  FINISHED=1
  rt untouched >> "$LOGS/run.log" 2>&1 || say "the untouched check FAILED (state/untouched.json)"
  say "$(tail -n 1 "$LOGS/run.log")"
  rt label >> "$LOGS/run.log" 2>&1 || say "the label's file list could not be written (logs/run.log)"
}
on_signal() { trap - TERM INT; say "the runner was stopped (signal); writing the untouched check and the label"; finish_up || true; exit "$1"; }

unit_checks() {  # the synthetic unit checks and the real seed-0 ones (section 8 step 1)
  local c ok=0
  read -r -a c <<< "$CPUS"
  mkdir -p "$STATE/tmp"
  : > "$LOGS/test.log"
  if (cd "$IW/repo" && exec env "${ARITH_ENV[@]}" PYTHONPATH="$IW/repo:$IW/explore:$CODE" PYTHONDONTWRITEBYTECODE=1 \
        JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 ARMS_SRC="$SRC" ARMS_IW="$IW" TMPDIR="$STATE/tmp" \
        nice -n 10 taskset -c "${c[0]}" "$PY" -B -m unittest -v test_arms) >> "$LOGS/test.log" 2>&1; then
    say "unit checks (synthetic, test_arms.py): passed ($(grep -E '^Ran ' "$LOGS/test.log" | tail -1))"
  else say "unit checks (synthetic) FAILED (logs/test.log)"; ok=1; fi
  if py_job "${c[0]}" "$LOGS/checks_real.log" "$CODE/checks.py"; then
    say "unit checks (real seed-0 inputs, checks.py): passed (state/checks_real.json)"
  else say "unit checks (real seed-0 inputs) FAILED (logs/checks_real.log)"; ok=1; fi
  return $ok
}

# ------------------------------------------------------------------------------------------------ smoke

smoke() {  # every job script end to end on real seed-0 inputs with a few steps, 50 starts per bank and 3,000 tries,
           # into state/smoke only (ARMS_SMOKE); labelled SMOKE TEST; its numbers are not results and are deleted
  local c sm=$STATE/smoke ok=1 step
  read -r -a c <<< "$CPUS"
  rm -rf "$sm"; mkdir -p "$sm"
  : > "$LOGS/smoke.log"
  date '+%F %T' > "$STATE/smoke.marker"; sleep 1
  for step in "arm_a.py --quality medium-expert --seed 0" "arm_b.py --quality medium-expert --seed 0" \
              "arm_c.py --quality random --seed 0" "arm_f0.py --quality medium-expert --seed 0" "screen.py" \
              "kills.py --smoke-stub" "score.py eval --arm A --quality medium-expert --seed 0" \
              "score.py eval --arm B --quality medium-expert --seed 0" "score.py eval --arm C --quality random --seed 0" \
              "score.py eval --arm F --quality medium-expert --seed 0" "score.py marks"; do
    # shellcheck disable=SC2086
    if (export ARMS_SMOKE="$sm" ARMS_SMOKE_SETTINGS=1; py_job "${c[0]}" "$LOGS/smoke.log" "$CODE"/$step); then
      say "SMOKE: $step: ran"
    else say "SMOKE: $step: FAILED (logs/smoke.log)"; ok=0; fi
  done
  local n
  n=$(find "$sm" -name 'record.json' | wc -l)
  say "SMOKE: $n records written under $sm (smoke-test settings; not results)"
  if [ -n "$(find "$IW" -newer "$STATE/smoke.marker" -type f -not -path '*/__pycache__/*' 2>/dev/null | head -1)" ]; then
    say "SMOKE: a file in the sandbox changed during the smoke test"; ok=0; fi
  rm -rf "$sm"
  say "SMOKE: removed $sm (its numbers are not results)"
  [ $ok -eq 1 ] && { say "SMOKE: every job script ran end to end"; return 0; }
  return 1
}

# ------------------------------------------------------------------------------------------------ dry, status, stop

dry() {
  local c ok=1 q marker probe_ok
  read -r -a c <<< "$CPUS"
  marker=$STATE/dry.marker
  date '+%F %T' > "$marker"
  sleep 1
  say "DRY: read-only checks (nothing is trained; nothing is written outside $ARMS); cpus $CPUS"
  bash -n "${BASH_SOURCE[0]}" && say "DRY: run_arms.sh parses"
  rt passmarks | while read -r l; do say "DRY: $l"; done
  rt passmarks > /dev/null || ok=0
  for c1 in "${c[@]}"; do taskset -c "$c1" true || { say "DRY: cannot pin to CPU $c1"; ok=0; }; done
  if busy=$(sandbox_busy); then say "DRY: the sandbox is busy now:$busy (start would wait)"; else
    say "DRY: the sandbox is free (no result F lock, no other process working there)"; fi
  say "DRY: code SHA-256: $(cd "$CODE" && sha256sum ./*.py | awk '{printf "%s %s; ", $2, substr($1, 1, 12)}')"
  if py_job "${c[0]}" "$LOGS/gates_dry.log" "$CODE/gates.py" --dry; then say "DRY: gates passed (state/gates_dry.json)"
  else say "DRY: gates FAILED (logs/gates_dry.log; state/gates_dry.json)"; ok=0; fi
  say "DRY: the screen's jobs: $(rt jobs start | awk -F'\t' '{printf "%s[%s] ", $1, $2}')"
  if rt jobs continue > /dev/null 2>> "$LOGS/dry.log"; then say "DRY: 'continue' did NOT refuse before the screen"; ok=0
  else say "DRY: 'continue' refuses as expected before the screen: $(tail -n 1 "$LOGS/dry.log")"; fi
  # each job script on real inputs with --dry (loads and gates; trains nothing; writes nothing)
  for q in random medium-replay medium medium-expert expert; do
    for a in b c; do
      if py_job "${c[0]}" "$LOGS/dry.log" "$CODE/arm_$a.py" --quality "$q" --seed 0 --dry; then
        say "DRY: arm_$a.py $q seed 0: $(tail -n 1 "$LOGS/dry.log" | cut -c1-160)"
      else say "DRY: arm_$a.py $q seed 0 --dry FAILED (logs/dry.log)"; ok=0; fi
    done
  done
  for q in medium-expert expert; do
    for s in a f0; do
      if py_job "${c[0]}" "$LOGS/dry.log" "$CODE/arm_$s.py" --quality "$q" --seed 0 --dry; then
        say "DRY: arm_$s.py $q seed 0: $(tail -n 1 "$LOGS/dry.log" | cut -c1-160)"
      else say "DRY: arm_$s.py $q seed 0 --dry FAILED (logs/dry.log)"; ok=0; fi
    done
  done
  # the real commands that must refuse now (each run only while its refusing input is missing)
  probe() {  # probe NAME GUARD-FILE EXPECTED-TEXT python-arguments...
    local name=$1 guard=$2 want=$3 r=0 pf
    shift 3
    if [ -e "$guard" ]; then say "DRY: probe '$name' skipped ($guard exists, so it would run for real)"; return 0; fi
    pf=$LOGS/dry_probe_$(printf %s "$name" | tr -c 'A-Za-z0-9_.-' '_').log
    : > "$pf"
    py_job "${c[0]}" "$pf" "$@" || r=$?
    if [ $r -ne 0 ] && grep -q -- "$want" "$pf"; then say "DRY: '$name' refused as expected (exit $r): $want"
    else say "DRY: '$name' did NOT refuse as expected (exit $r; $pf)"; ok=0; fi
  }
  local V5=$IW/repo/runs/shift_iw_v5
  probe "arm A on random seed 0 (no band yet)" "$V5/random/band/gap/seed0/record.json" "has not finished" \
    "$CODE/arm_a.py" --quality random --seed 0
  probe "F0 on random" "/nonexistent" "F0 runs on" "$CODE/arm_f0.py" --quality random --seed 0
  probe "F0 on medium-expert seed 1 (A2 of seed 1 not trained)" \
    "$IW/repo/runs/shift_iw_v5_explore/channel/train_f1/medium-expert/A2_cmd_strong/seed1/record.json" \
    "F0 runs after A2" "$CODE/arm_f0.py" --quality medium-expert --seed 1
  probe "arm B on medium seed 1 (no host yet)" "$V5/medium/host/gap/seed1/record.json" "has not finished" \
    "$CODE/arm_b.py" --quality medium --seed 1
  probe "kills before the screen" "$ARMS/SCREEN_seed0.json" "refused" "$CODE/kills.py"
  probe "scoring before the kill checks" "$ARMS/KILLS.json" "refused" "$CODE/score.py" marks
  probe "evaluation before the kill checks" "$ARMS/KILLS.json" "refused" "$CODE/score.py" eval --arm A \
    --quality expert --seed 0
  probe "band fit for random seed 0 before its room check" "$V5/random/room/seed0/record.json" \
    "the room check of seed 0" -m experiments.shift_iw.run band --quality random --seeds 0 --kinds gap
  # nothing written outside this folder
  local newer
  newer=$( { find "$IW" -newer "$marker" -type f -not -path '*/__pycache__/*' 2>/dev/null;
             find "$SRC/outputs/shift_iw" "$SRC/experiments" -newer "$marker" 2>/dev/null; } | head -20)
  if [ -n "$newer" ]; then say "DRY: files changed in the sandbox or the repository during the dry run: $newer"; ok=0
  else say "DRY: nothing in the sandbox or the repository changed during the dry run"; fi
  [ $ok -eq 1 ] && { say "DRY: all checks passed"; return 0; }
  say "DRY: some check FAILED"; return 1
}

ours() { kill -0 "$1" 2>/dev/null && tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -q 'run_arms\.sh'; }

status() {
  local phase j state pid cpu t0 n_done=0 n_all=0
  echo "arms A, B, C and F0 (EXPLORATORY) at $ARMS (sandbox $IW)"
  rt passmarks || true
  if lock_held "$STATE/run.lock"; then echo "runner: RUNNING ($(cat "$STATE/phase" 2>/dev/null))"; else echo "runner: not running"; fi
  if busy=$(sandbox_busy); then echo "working in the sandbox now:$busy"; fi
  for phase in start continue; do
    rows=$(rt jobs "$phase" 2>/dev/null) || { echo "$phase: $(rt jobs "$phase" 2>&1 >/dev/null | tail -1)"; continue; }
    echo "== $phase"
    while IFS=$'\t' read -r j _; do
      n_all=$((n_all + 1))
      if rt done "$j" 2>/dev/null; then state=finished; n_done=$((n_done + 1))
      elif [ -f "$STATE/$j.pid" ] && read -r pid cpu t0 < "$STATE/$j.pid" && ours "$pid"; then
        state="running on cpu $cpu for $(( ($(date +%s) - t0) / 60 )) min; last: $(tail -n 1 "$LOGS/$j.log" 2>/dev/null | cut -c1-90)"
      elif [ -f "$STATE/$j.failed" ]; then state="FAILED (exit $(cat "$STATE/$j.failed"); log logs/$j.log)"
      else state=pending; fi
      printf '  %-34s %s\n' "$j" "$state"
    done <<< "$rows"
  done
  echo "$n_done of $n_all jobs finished"
  for f in SCREEN_seed0.md KILLS.md RESULTS.md; do [ -f "$ARMS/$f" ] && echo "written: $ARMS/$f"; done
  return 0
}

kill_tree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done; kill -TERM "$1" 2>/dev/null || true; }

stop_run() {
  local pid f c
  if [ -f "$STATE/runner.pid" ] && read -r pid < "$STATE/runner.pid" && ours "$pid"; then
    kill -STOP "$pid"
    for c in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$c"; done
    kill -TERM "$pid" || true; kill -CONT "$pid" || true
    echo "stopped the runner (pid $pid) and its jobs"
  else
    echo "no runner is running"
  fi
  for f in "$STATE"/*.pid; do
    [ -e "$f" ] || continue
    case $f in */runner.pid) continue ;; esac
    if read -r pid _ < "$f" && ours "$pid"; then kill_tree "$pid"; echo "stopped $(basename "$f" .pid)"; fi
    rm -f "$f"
  done
  rm -f "$STATE"/*.failed
  return 0
}

# ------------------------------------------------------------------------------------------------ main

CMD=${1:-}
case $CMD in
  status) status; exit 0 ;;
  stop) stop_run; exit 0 ;;
  dry|test|smoke|start|continue) ;;
  *) echo "usage: bash run_arms.sh start|continue|status|stop|dry|test|smoke" >&2; exit 64 ;;
esac

rt passmarks > /dev/null || { say "REFUSED: the frozen pass marks differ from their sidecar or from the frozen digest"; exit 3; }
exec 9> "$STATE/run.lock"
if ! flock -n 9; then echo "another run_arms.sh is running (lock $STATE/run.lock)" >&2; exit 75; fi
for c in $CPUS; do taskset -c "$c" true 2>/dev/null || { echo "refused: cannot pin to CPU $c" >&2; exit 64; }; done

if [ "$CMD" = dry ]; then dry; exit $?; fi
if [ "$CMD" = test ]; then unit_checks; exit $?; fi
if [ "$CMD" = smoke ]; then smoke; exit $?; fi

echo "$$" > "$STATE/runner.pid"
echo "$CMD" > "$STATE/phase"
say "EXPLORATORY arms A, B, C and F0: $CMD; cpus $CPUS; sandbox $IW"
read -r -a C0 <<< "$CPUS"
if [ "$CMD" = start ]; then
  if ! py_job "${C0[0]}" "$LOGS/gates.log" "$CODE/gates.py"; then say "REFUSED: the gates did not pass (logs/gates.log)"; exit 3; fi
  say "gates passed (state/gates.json)"
  if ! unit_checks; then say "REFUSED: the unit checks did not pass"; exit 3; fi
  wait_for_sandbox
  rt baseline >> "$LOGS/run.log" 2>&1
  trap 'on_signal 143' TERM; trap 'on_signal 130' INT; trap 'finish_up || true' EXIT
  load_jobs start || { say "REFUSED: no job list"; exit 3; }
  schedule
  finish_up
  if rt done screen 2>/dev/null; then
    say "SCREEN written: $ARMS/SCREEN_seed0.md; $(rt screen-allows 2>&1 | tail -1)"
    say "seeds 1-4 start only with: bash run_arms.sh continue"
    exit 0
  fi
  say "the screen is not finished (see status); rerun start to resume"
  exit 1
fi
# continue
if ! rt screen-allows > /dev/null 2>> "$LOGS/run.log"; then
  say "REFUSED: $(tail -n 1 "$LOGS/run.log")"; exit 4
fi
if ! py_job "${C0[0]}" "$LOGS/gates.log" "$CODE/gates.py"; then say "REFUSED: the gates did not pass (logs/gates.log)"; exit 3; fi
wait_for_sandbox
rt baseline >> "$LOGS/run.log" 2>&1
trap 'on_signal 143' TERM; trap 'on_signal 130' INT; trap 'finish_up || true' EXIT
load_jobs continue || { say "REFUSED: no job list"; exit 3; }
schedule
finish_up
if rt done marks 2>/dev/null; then say "RESULTS written: $ARMS/RESULTS.md"; exit 0; fi
say "not finished (see status); rerun continue to resume"
exit 1
