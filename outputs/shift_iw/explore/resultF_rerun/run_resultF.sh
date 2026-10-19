#!/usr/bin/env bash
# EXPLORATORY: outside the version-5 pre-registration.
#
# Result F's actor for the live panel: a seed-0 rerun in the cloud sandbox (iw_rerun) of the weighting test's result F,
# "Does the band, weighted for where it is read, raise return?", with result F's own code unchanged, for medium-expert
# and expert: "Fixed fallback, exact weights" (A2_cmd_strong; its script also trains B2 and C2), "No extra caution"
# (le_none; room-lower-edge also trains the perfect lower edge), and optionally version 5's unweighted lower edge and
# "Estimated weights" (A3, C3). Every row is trained with the desktop's settings and code, from the sandbox's own seed-0
# host (the live panel's TD3+BC learner): its cloned actor and its frozen critic. Then every trained actor is scored on
# both mixes' 1,000 fixed starts by the scripts' own evaluate modes, and eval_resultF.py adds the episodes and the main
# test's four situations. See PLAN.md.
#
#   bash run_resultF.sh            run (or resume) everything; finished work is never redone
#   bash run_resultF.sh dry        read-only checks of everything that can be checked without training
#   bash run_resultF.sh status     what is finished, running, failed, blocked or pending (reads only)
#   bash run_resultF.sh stop       stop a background run and its jobs (a stopped job restarts from scratch on resume)
#   bash run_resultF.sh verify     the sandbox's earlier files and the repository untouched (state/untouched.json)
#   setsid nohup bash run_resultF.sh > logs/nohup.out 2>&1 < /dev/null &     (background; survives the terminal)
#
# run and dry refuse to start while another process works in the sandbox: any process running
# `python -m experiments.shift_iw.run` whose working directory is inside the sandbox, or data_seeds1to4.sh (the main
# session's separate script that builds data seeds 1-4 and the report of each quality). The done checks accept that
# script's data and reports (finished records under this specification; the report on seeds 0-4, newer than every data
# record), so the run skips them and never rebuilds them.
#
# Environment (defaults in brackets):
#   CPUS        CPUs to use, one job per CPU, each job pinned to its CPU ["1 2 3": other sessions share this 4-CPU
#               machine, so CPU 0 is left to them]
#   QUALITIES   ["medium-expert expert"] (result F covered only these)
#   V5LE        1: also version 5's unweighted lower edge (explore_arms.py --lower-edge unweighted) [1]
#   EST         1: also the estimated-weights arms (command_le_train_est.py) [1]
#   ARITH       desktop: the processor settings the sandbox's run used (OpenBLAS Haswell, numpy's AVX-512 off, XLA
#               capped at AVX2); native: this machine's own [desktop]
#   SRC         the repository, read only [/home/user/bca]
#   IW          the sandbox [<scratchpad>/iw_rerun]
#   PY          the Python [/tmp/claude-0/bcavenv/bin/python3]
#
# DONE (the file <this folder>/DONE) needs every core job finished: data, report, d1, room, band, room-lower-edge and
# its evaluation, result F's training, evaluation and checks, and evalF in its complete form (`eval_resultF.py run`).
# The optional jobs (est*, v5le*) are left out of the DONE condition: if one fails, DONE still comes and names it. If
# B2, C2 or a lower-edge job fails but A2's record and actor are good, evalF runs `eval_resultF.py run --allow-missing`
# (A2 scored, No extra caution scored when it exists, the missing rows listed); that run is not DONE.
#
# Writes only: <this folder>/{state,logs,results,smoke,DONE} and new files under <IW>/repo/runs/ (the package's own
# records and their sha256sum record checksums, written by the package's stages, as the owner allowed; result F's
# scripts' records; one EXPLORATORY_resultF_rerun.txt label in each runs/ root, rewritten at the end of every run with
# the list of every file the run created and of data_seeds1to4.sh's outputs; an interrupted arm of this run is moved to
# state/quarantine/ before its script restarts, see driver/rf.py quarantine). Never writes into the repository, never
# touches a file the sandbox already has, never creates, edits or deletes an approval, spec, pass-mark or declined
# file. Result F's scripts run in place from the repository's channel/ folder (their frozen pass marks sit beside
# them), with python -B (no bytecode is written anywhere).
#
# Every job: cd <IW>/repo; PYTHONPATH=<IW>/repo:<IW>/explore PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
# OMP_NUM_THREADS=1 [ARITH settings] nice -n 10 taskset -c <one CPU> python -B ... One CPU and one thread for every
# job, so result F's arms (A2, B2, C2, A3, C3) share their arithmetic with "No extra caution" and the room check in this
# sandbox. This is version 5's way (its stages ran pinned, one thread), but NOT the desktop's setting for the channel
# arms: the desktop trained result F's scripts unpinned with OMP_NUM_THREADS=2
# (outputs/shift_iw/explore/curves/orchestrate.py lines 3-4). XLA's CPU arithmetic depends on how many CPUs and threads
# a process may use, so the comparison of the cloned actor's hash with the desktop's is information only.
# Logs: logs/<job>.log, logs/run.log.
set -euo pipefail
RF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${SRC:-/home/user/bca}
IW=${IW:-/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/iw_rerun}
PY=${PY:-/tmp/claude-0/bcavenv/bin/python3}
CPUS=${CPUS:-"1 2 3"}
QUALITIES=${QUALITIES:-"medium-expert expert"}
V5LE=${V5LE:-1}
EST=${EST:-1}
ARITH=${ARITH:-desktop}
NPY_OFF="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
case $ARITH in
  desktop) ARITH_ENV=(OPENBLAS_CORETYPE=Haswell "NPY_DISABLE_CPU_FEATURES=$NPY_OFF" XLA_FLAGS=--xla_cpu_max_isa=AVX2) ;;
  native) ARITH_ENV=() ;;
  *) echo "ARITH is desktop or native, not $ARITH" >&2; exit 64 ;;
esac
export RF_SRC="$SRC" RF_IW="$IW" RF_QUALITIES="$QUALITIES" RF_ARITH="$ARITH"   # read by driver/*.py, eval_resultF.py
REPO=$IW/repo
XPL=$IW/explore
V5=$REPO/runs/shift_iw_v5
EX=$REPO/runs/shift_iw_v5_explore
CHAN=$SRC/outputs/shift_iw/explore/channel
STATE=$RF/state
LOGS=$RF/logs
SEED=0
mkdir -p "$STATE" "$LOGS" "$RF/results"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/run.log"; }
rfpy() { "$PY" -I -B "$RF/driver/rf.py" "$@"; }   # the standard-library helper (driver/rf.py)

# ------------------------------------------------------------------------------------------------ jobs

all_jobs() {  # every job, highest priority first (a free CPU takes the first ready one)
  local q s
  for q in $QUALITIES; do echo "room_$q"; done                        # the longest, on the critical path
  for q in $QUALITIES; do for s in 1 2 3 4; do echo "data_${q}_s$s"; done; done   # for the band's report gate only
  for q in $QUALITIES; do echo "report_$q"; done
  for q in $QUALITIES; do echo "band_$q"; done
  for q in $QUALITIES; do echo "f1train_$q"; done                     # result F: A2, then B2, then C2
  for q in $QUALITIES; do echo "roomle_$q"; done                      # No extra caution, Perfect lower edge
  if [ "$EST" = 1 ]; then for q in $QUALITIES; do echo "esttrain_$q"; done; fi
  for q in $QUALITIES; do echo "f1eval_$q"; echo "roomleeval_$q"; done
  if [ "$V5LE" = 1 ]; then for q in $QUALITIES; do echo "v5le_$q"; done; fi
  for q in $QUALITIES; do echo "f1checks_$q"; done
  if [ "$EST" = 1 ]; then for q in $QUALITIES; do echo "estchecks_$q"; echo "esteval_$q"; done; fi
  if [ "$V5LE" = 1 ]; then for q in $QUALITIES; do echo "v5leeval_$q"; done; fi
  for q in $QUALITIES; do echo "d1_$q"; done
  echo evalF                                                          # last: waits for every other job to settle
}

is_optional() { case ${1%%_*} in esttrain|esteval|estchecks|v5le|v5leeval) return 0 ;; *) return 1 ;; esac; }

quality_of() { local r=${1#*_}; echo "${r%_s*}"; }

deps_of() {  # the jobs a job needs finished
  local j=$1 k=${1%%_*} q
  q=$(quality_of "$j")
  case $k in
    room|data) ;;
    report|d1) echo "data_${q}_s1 data_${q}_s2 data_${q}_s3 data_${q}_s4" ;;
    band) echo "room_$q report_$q" ;;          # run.stage_band: the room record (proxy check) and the D1 report
    roomle) echo "room_$q" ;;
    f1train|esttrain|v5le) echo "band_$q" ;;   # command_le_train.World / explore_arms: data, host, band, approval
    f1eval) echo "f1train_$q" ;;
    esteval) echo "esttrain_$q" ;;
    v5leeval) echo "v5le_$q" ;;
    roomleeval) echo "roomle_$q" ;;
    f1checks|estchecks) echo "band_$q roomle_$q" ;;   # the checks read the no-caution actor (le_none)
    evalF) for q in $QUALITIES; do echo "f1eval_$q roomleeval_$q"; done ;;   # the complete form; see evalf_state
  esac
}

is_done() { rfpy done "$1" 2>> "$LOGS/run.log"; }

sandbox_busy() {  # prints and succeeds if another process works in the sandbox (see the header)
  local p a m d cwd found="" iw_real
  iw_real=$(cd "$IW" 2>/dev/null && pwd -P) || iw_real=$IW
  for p in /proc/[0-9]*; do
    p=${p#/proc/}
    [ "$p" = "$$" ] && continue
    m=0; d=0
    while IFS= read -r -d '' a; do
      case $a in experiments.shift_iw.run) m=1 ;; data_seeds1to4.sh|*/data_seeds1to4.sh) d=1 ;; esac
    done 2>/dev/null < "/proc/$p/cmdline" || continue
    if [ $m = 1 ]; then
      cwd=$(readlink "/proc/$p/cwd" 2>/dev/null) || continue
      case "$cwd/" in "$IW"/*|"$iw_real"/*) found="$found pid $p (experiments.shift_iw.run in $cwd);" ;; esac
    fi
    [ $d = 1 ] && found="$found pid $p (data_seeds1to4.sh);"
  done
  [ -n "$found" ] && { echo "$found"; return 0; }
  return 1
}

py_job() {  # py_job CPU LOG python-arguments... (foreground; pinned to one CPU; never writes bytecode)
  local cpu=$1 logf=$2
  shift 2
  (cd "$REPO" && exec env "${ARITH_ENV[@]}" PYTHONPATH="$REPO:$XPL" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
     OMP_NUM_THREADS=1 RF_SRC="$SRC" RF_IW="$IW" RF_QUALITIES="$QUALITIES" RF_ARITH="$ARITH" \
     nice -n 10 taskset -c "$cpu" "$PY" -B "$@") >> "$logf" 2>&1
}

run_job() {  # run_job JOB CPU [MODE]: one job, then its done check; exit 0 only if its output is finished and current
  local j=$1 cpu=$2 mode=${3:-} k=${1%%_*} q s rc=0 logf=$LOGS/$1.log donejob=$1
  q=$(quality_of "$j")
  echo "[$(date '+%F %T')] start on cpu $cpu (pid $BASHPID)${mode:+ ($mode)}" >> "$logf"
  case $k in
    data) s=${j##*_s}; py_job "$cpu" "$logf" -m experiments.shift_iw.run data --quality "$q" --seeds "$s" || rc=$? ;;
    report) py_job "$cpu" "$logf" -m experiments.shift_iw.run report --quality "$q" || rc=$? ;;
    d1) rfpy d1 "$q" >> "$logf" 2>&1 || rc=$? ;;
    room) py_job "$cpu" "$logf" -m experiments.shift_iw.run room --quality "$q" --seeds $SEED || rc=$? ;;
    band) py_job "$cpu" "$logf" -m experiments.shift_iw.run band --quality "$q" --seeds $SEED --kinds gap || rc=$? ;;
    roomle) py_job "$cpu" "$logf" -m experiments.shift_iw.run room-lower-edge --quality "$q" --seeds $SEED || rc=$? ;;
    roomleeval) py_job "$cpu" "$logf" -m experiments.shift_iw.run evaluate --quality "$q" --seeds $SEED \
                  --groups room_lower_edge || rc=$? ;;
    f1train) rfpy quarantine "$j" >> "$logf" 2>&1 || rc=$?
             [ $rc -eq 0 ] && { py_job "$cpu" "$logf" "$CHAN/command_le_train_f1.py" train "$q" $SEED || rc=$?; } ;;
    f1eval) py_job "$cpu" "$logf" "$CHAN/command_le_train_f1.py" evaluate "$q" $SEED || rc=$? ;;
    f1checks) py_job "$cpu" "$logf" "$CHAN/command_le_train_f1.py" checks "$q" $SEED || rc=$? ;;
    esttrain) rfpy quarantine "$j" >> "$logf" 2>&1 || rc=$?
              [ $rc -eq 0 ] && { py_job "$cpu" "$logf" "$CHAN/command_le_train_est.py" train "$q" $SEED || rc=$?; } ;;
    esteval) py_job "$cpu" "$logf" "$CHAN/command_le_train_est.py" evaluate "$q" $SEED || rc=$? ;;
    estchecks) py_job "$cpu" "$logf" "$CHAN/command_le_train_est.py" checks "$q" $SEED || rc=$? ;;
    v5le) py_job "$cpu" "$logf" "$XPL/explore_arms.py" --quality "$q" --seed $SEED --lower-edge unweighted || rc=$? ;;
    # --groups bca_lower_edge only: the sandbox's existing 'bca' evaluation (the live panel's) is never redone
    v5leeval) py_job "$cpu" "$logf" "$XPL/explore_eval.py" --quality "$q" --seed $SEED --groups bca_lower_edge \
                || rc=$? ;;
    evalF) if [ "$mode" = partial ]; then   # A2 good, but a core row is missing: score A2 and what exists
             donejob=evalF-partial
             py_job "$cpu" "$logf" "$RF/eval_resultF.py" run --allow-missing || rc=$?
           else
             py_job "$cpu" "$logf" "$RF/eval_resultF.py" run || rc=$?
           fi ;;
    *) echo "unknown job $j" >> "$logf"; return 64 ;;
  esac
  if [ $rc -eq 0 ] && is_done "$donejob"; then
    echo "[$(date '+%F %T')] finished" >> "$logf"
    return 0
  fi
  [ $rc -eq 0 ] && rc=70
  echo "[$(date '+%F %T')] FAILED (exit $rc)" >> "$logf"
  return $rc
}

# ------------------------------------------------------------------------------------------------ the scheduler

declare -A ST=()          # job -> pending | running | done | failed | blocked
declare -A CPU_OF=() NAME_OF=()
FREE=()
EVALF_MODE=""

evalf_state() {  # the last job: wait for every other job to settle; then full | partial | blocked
  local j q full=1
  for j in "${!ST[@]}"; do
    [ "$j" = evalF ] && continue
    case ${ST[$j]} in pending|running) echo wait; return ;; esac
  done
  for j in $(deps_of evalF); do [ "${ST[$j]:-absent}" = done ] || full=0; done
  if [ $full = 1 ]; then echo ok-full; return; fi
  for q in $QUALITIES; do is_done "a2ok_$q" || { echo blocked; return; }; done   # A2's record and actor good
  echo ok-partial
}

deps_state() {  # ok | ok-full | ok-partial | wait | blocked
  local d
  if [ "$1" = evalF ]; then evalf_state; return; fi
  for d in $(deps_of "$1"); do
    case ${ST[$d]:-absent} in
      done) ;;
      failed|blocked) echo blocked; return ;;
      absent) echo blocked; return ;;
      *) echo wait; return ;;
    esac
  done
  echo ok
}

start_ready() {
  local j cpu pid st mode
  for j in "${JOBS[@]}"; do
    [ ${#FREE[@]} -gt 0 ] || return 0
    [ "${ST[$j]}" = pending ] || continue
    st=$(deps_state "$j")
    case $st in
      ok|ok-*) mode=${st#ok}; mode=${mode#-} ;;
      blocked) ST[$j]=blocked; say "BLOCKED (an input job failed): $j"; continue ;;
      *) continue ;;
    esac
    [ "$j" = evalF ] && EVALF_MODE=$mode
    cpu=${FREE[0]}
    FREE=("${FREE[@]:1}")
    run_job "$j" "$cpu" "$mode" &
    pid=$!
    CPU_OF[$pid]=$cpu
    NAME_OF[$pid]=$j
    ST[$j]=running
    echo "$pid $cpu $(date +%s)" > "$STATE/$j.pid"
    say "started: $j on cpu $cpu (pid $pid)${mode:+ ($mode)}"
  done
}

reap() {
  local pid rc=0 j cpu t0=""
  wait -n -p pid || rc=$?
  j=${NAME_OF[$pid]}
  if [ $rc -eq 0 ]; then
    ST[$j]=done; rm -f "$STATE/$j.failed"; say "done: $j"
  else
    ST[$j]=failed; echo "$rc" > "$STATE/$j.failed"; say "FAILED (exit $rc): $j (log $LOGS/$j.log)"
  fi
  read -r _ cpu t0 2>/dev/null < "$STATE/$j.pid" || true
  printf '%s\t%s\t%s\t%s\t%s\n' "$j" "${CPU_OF[$pid]}" "${t0:-0}" "$(date +%s)" "$rc" >> "$STATE/job_times.tsv"
  rm -f "$STATE/$j.pid"
  FREE+=("${CPU_OF[$pid]}")
  unset "CPU_OF[$pid]" "NAME_OF[$pid]"
}

# ------------------------------------------------------------------------------------------------ gates, dry, status

gates() {  # read-only gate check in the jobs' own environment (driver/gates.py)
  local c
  read -r -a c <<< "$CPUS"
  py_job "${c[0]}" "$LOGS/gates.log" "$RF/driver/gates.py" "$@"
}

dry() {  # everything that can be checked without training; writes nothing outside this folder
  local q c ok=1 j
  read -r -a c <<< "$CPUS"
  say "DRY: read-only checks (nothing is trained; nothing is written outside $RF); cpus $CPUS"
  for c1 in "${c[@]}"; do taskset -c "$c1" true || { say "DRY: cannot pin to CPU $c1"; ok=0; }; done
  bash -n "${BASH_SOURCE[0]}" && say "DRY: run_resultF.sh parses"
  say "DRY: file SHA-256: $(cd "$RF" && sha256sum run_resultF.sh eval_resultF.py driver/*.py data_seeds1to4.sh \
        | awk '{printf "%s %s; ", $2, substr($1, 1, 12)}')"
  export RF_BASELINE_DIR=$STATE/dry_baseline   # the dry mode's own baseline; the real run takes its own
  rm -rf "$RF_BASELINE_DIR"
  rfpy baseline >> "$LOGS/dry.log" 2>&1
  if gates --dry; then say "DRY: gates passed (state/gates_dry.json)"; else say "DRY: gates FAILED (logs/gates.log)"; ok=0; fi
  # the band's report gate (run.require_design), on the build_report the package would make and on the real report
  # (driver/report_dry.py; nothing written)
  if py_job "${c[0]}" "$LOGS/dry.log" "$RF/driver/report_dry.py"; then
    say "DRY: the band's report gate accepts the report on seeds 0-4 (D1 pass; state/report_dry.json)"
  else say "DRY: report gate dry check FAILED (logs/dry.log; state/report_dry.json)"; ok=0; fi
  for q in $QUALITIES; do
    # what the run will skip: data seeds 1-4 and the report (data_seeds1to4.sh's), by the run's own done checks
    local have="" miss=""
    for j in "data_${q}_s1" "data_${q}_s2" "data_${q}_s3" "data_${q}_s4" "report_$q"; do
      if is_done "$j"; then have="$have $j"; else miss="$miss $j"; fi
    done
    say "DRY: $q: finished, the run will skip:${have:- none}; the run will build:${miss:- none}"
    if [ -z "$miss" ]; then
      rfpy d1 "$q" --dry >> "$LOGS/dry.log" 2>&1 && say "DRY: $(tail -n 1 "$LOGS/dry.log")" || ok=0
    fi
    py_job "${c[0]}" "$LOGS/dry.log" -m experiments.shift_iw.run status --quality "$q" --seeds 0 1 2 3 4 || ok=0
    for a in "data --quality $q --seeds 1 --help" "report --quality $q --help"; do
      # shellcheck disable=SC2086
      py_job "${c[0]}" "$LOGS/dry.log" -m experiments.shift_iw.run $a > /dev/null || ok=0
    done
    # the real commands, each run only while the input that makes it refuse is missing (so it cannot train or write);
    # each probe's output goes to its own file, and only that file is judged
    probe() {  # probe NAME GUARD-FILE EXPECTED-TEXT python-arguments...
      local name=$1 guard=$2 want=$3 r=0 pf
      shift 3
      if [ -e "$guard" ]; then say "DRY: $name skipped ($guard exists, so the command would run for real)"; return 0; fi
      pf=$LOGS/dry_probe_$(printf %s "$name" | tr -c 'A-Za-z0-9_.-' '_').log
      : > "$pf"
      py_job "${c[0]}" "$pf" "$@" || r=$?
      if [ $r -ne 0 ] && grep -q -- "$want" "$pf"; then say "DRY: $name refused as expected (exit $r): $want"
      else say "DRY: $name did NOT refuse as expected (exit $r; $pf)"; ok=0; fi
    }
    probe "band $q" "$V5/$q/room/seed0/record.json" "the room check of seed 0" \
      -m experiments.shift_iw.run band --quality "$q" --seeds 0 --kinds gap
    probe "room-lower-edge $q" "$V5/$q/room/seed0/record.json" "the room check of seed 0" \
      -m experiments.shift_iw.run room-lower-edge --quality "$q" --seeds 0
    probe "evaluate room_lower_edge $q" "$V5/$q/room_lower_edge/seed0/record.json" "lower edge's room check" \
      -m experiments.shift_iw.run evaluate --quality "$q" --seeds 0 --groups room_lower_edge
    probe "f1 checks $q" "$V5/$q/band/gap/seed0/record.json" "gap band of seed 0 has not finished" \
      "$CHAN/command_le_train_f1.py" checks "$q" 0
    probe "est checks $q" "$V5/$q/band/gap/seed0/record.json" "gap band of seed 0 has not finished" \
      "$CHAN/command_le_train_est.py" checks "$q" 0
    probe "explore_arms --smoke --lower-edge unweighted $q" "$V5/$q/band/gap/seed0/record.json" \
      "gap band of seed 0 has not finished" "$XPL/explore_arms.py" --quality "$q" --seed 0 --lower-edge unweighted \
      --smoke --out "$RF/smoke"
  done
  if py_job "${c[0]}" "$LOGS/eval_check.log" "$RF/eval_resultF.py" check; then
    say "DRY: eval_resultF.py check passed (state/eval_check.json)"
  else say "DRY: eval_resultF.py check FAILED (logs/eval_check.log)"; ok=0; fi
  # eval_resultF.py's run mode end to end with the sandbox's TD3+BC host actor standing in for A2 and No extra caution
  # (a smoke test of the scoring, episodes, situations and report code by the --allow-missing path, and of the
  # refusal without it; writes only smoke/eval/, labelled SMOKE TEST)
  if py_job "${c[0]}" "$LOGS/eval_smoke.log" "$RF/driver/smoke_eval.py"; then
    say "DRY: eval_resultF.py run mode ran end to end on stand-in actors (smoke/eval/; not results)"
  else say "DRY: eval_resultF.py smoke run FAILED (logs/eval_smoke.log)"; ok=0; fi
  rfpy untouched >> "$LOGS/dry.log" 2>&1 || ok=0
  say "DRY: $(tail -n 1 "$LOGS/dry.log")"
  say "DRY: jobs: $(for j in $(all_jobs); do printf '%s[%s]%s ' "$j" "$(deps_of "$j" | tr ' \n' ',,' | sed 's/,$//')" \
        "$(is_optional "$j" && echo '(optional)')"; done)"
  [ $ok -eq 1 ] && { say "DRY: all checks passed"; return 0; }
  say "DRY: some check FAILED"; return 1
}

ours() {  # alive and running this script (a runner or a job's subshell), not a stranger that reused a PID
  kill -0 "$1" 2>/dev/null && tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -q 'run_resultF\.sh'
}

status() {
  local j state pid cpu t0 n_done=0 n_all=0 busy
  echo "result F seed-0 sandbox rerun at $RF (sandbox $IW)"
  [ -f "$RF/DONE" ] && echo "DONE: $(head -1 "$RF/DONE")"
  if flock -n "$STATE/run.lock" true 2>/dev/null; then echo "runner: not running"; else echo "runner: RUNNING"; fi
  if busy=$(sandbox_busy); then echo "working in the sandbox now:$busy"; fi
  [ -f "$STATE/gates.json" ] && echo "gates: $(grep -o '"all_passed": [a-z]*' "$STATE/gates.json")"
  for j in $(all_jobs); do
    n_all=$((n_all + 1))
    if is_done "$j"; then state=finished; n_done=$((n_done + 1))
    elif [ -f "$STATE/$j.pid" ] && read -r pid cpu t0 < "$STATE/$j.pid" && ours "$pid"; then
      state="running on cpu $cpu for $(( ($(date +%s) - t0) / 60 )) min; last: $(tail -n 1 "$LOGS/$j.log" 2>/dev/null | cut -c1-100)"
    elif [ -f "$STATE/$j.failed" ]; then state="FAILED (exit $(cat "$STATE/$j.failed"); log logs/$j.log)"
    elif [ "$j" = evalF ] && is_done evalF-partial; then state="partial results only (run --allow-missing)"
    else state=pending; fi
    printf '  %-28s %s%s\n' "$j" "$state" "$(is_optional "$j" && echo ' (optional)')"
  done
  echo "$n_done of $n_all jobs finished"
  [ -f "$RF/results/COMPARE.md" ] && grep -m1 '^\*\*Overall' "$RF/results/COMPARE.md"
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

case ${1:-run} in
  status) status; exit 0 ;;
  stop) stop_run; exit 0 ;;
  verify) rfpy untouched; exit $? ;;
  dry|run) ;;
  *) echo "usage: bash run_resultF.sh [run|dry|status|stop|verify]" >&2; exit 64 ;;
esac

exec 9> "$STATE/run.lock"
if ! flock -n 9; then echo "another run_resultF.sh is running (lock $STATE/run.lock)" >&2; exit 75; fi
if BUSY=$(sandbox_busy); then
  say "REFUSED: another process works in the sandbox ($IW):$BUSY wait until it has finished (data_seeds1to4.sh: logs/data_*.log, logs/report_*.log)"
  exit 75
fi
for c in $CPUS; do taskset -c "$c" true 2>/dev/null || { echo "refused: cannot pin to CPU $c" >&2; exit 64; }; done
read -r -a FREE <<< "$CPUS"

if [ "${1:-run}" = dry ]; then dry; exit $?; fi

rm -f "$RF/DONE"
echo "$$" > "$STATE/runner.pid"
say "EXPLORATORY result F seed-0 sandbox rerun: qualities $QUALITIES; V5LE=$V5LE EST=$EST; cpus $CPUS; ARITH=$ARITH (${ARITH_ENV[*]:-native}); sandbox $IW"
rfpy baseline >> "$LOGS/run.log" 2>&1
rfpy provenance >> "$LOGS/run.log" 2>&1     # data_seeds1to4.sh's outputs, by path (state/data_seeds1to4.json)
if ! gates; then say "REFUSED: the gates did not pass (logs/gates.log; state/gates.json)"; exit 3; fi
say "gates passed (state/gates.json)"
rfpy label >> "$LOGS/run.log" 2>&1   # EXPLORATORY_resultF_rerun.txt in both runs/ roots (the file list comes at the end)

mapfile -t JOBS < <(all_jobs)
for j in "${JOBS[@]}"; do
  if is_done "$j"; then ST[$j]=done; say "skip (finished): $j"; else ST[$j]=pending; fi
done
while :; do
  start_ready
  [ ${#CPU_OF[@]} -gt 0 ] || break
  reap
done

CORE_BAD=$(for j in "${JOBS[@]}"; do is_optional "$j" && continue; case ${ST[$j]} in failed|blocked) echo "$j(${ST[$j]})" ;; esac; done)
OPT_BAD=$(for j in "${JOBS[@]}"; do is_optional "$j" || continue; case ${ST[$j]} in failed|blocked) echo "$j(${ST[$j]})" ;; esac; done)
UNTOUCHED=0
rfpy untouched >> "$LOGS/run.log" 2>&1 || UNTOUCHED=$?
say "$(tail -n 1 "$LOGS/run.log")"
rfpy label --final >> "$LOGS/run.log" 2>&1 || say "the label's file list could not be written (logs/run.log)"
[ -n "$OPT_BAD" ] && say "optional rows that failed or were blocked (not part of DONE): $(echo $OPT_BAD)"
if [ -n "$CORE_BAD" ] || [ $UNTOUCHED -ne 0 ] || ! is_done evalF; then
  [ "$EVALF_MODE" = partial ] && say "evalF scored A2 and what exists (run --allow-missing): results/COMPARE.md lists the missing rows"
  say "not DONE: failed or blocked core jobs: $(echo $CORE_BAD); untouched check exit $UNTOUCHED$([ $UNTOUCHED -ne 0 ] && echo ' (state/untouched.json: which paths changed, when, and the jobs running then)'). Rerun to resume."
  exit 1
fi
{
  echo "DONE $(date '+%F %T'): every core job finished (EXPLORATORY)"
  grep -m1 '^\*\*Overall' "$RF/results/COMPARE.md" 2>/dev/null || true
  [ -n "$OPT_BAD" ] && echo "optional rows that failed or were blocked (not part of DONE): $(echo $OPT_BAD)"
  echo "results: $RF/results (scores_seed0.json, COMPARE.md, episodes_seed0.json, situations_seed0.json); actors under $EX/channel and $V5/<quality>/room_lower_edge; every file created: $V5/EXPLORATORY_resultF_rerun.txt"
} > "$RF/DONE"
say "$(head -1 "$RF/DONE")"
exit 0
