#!/bin/sh
# Stage 2, version 7: run every remaining Stage 2 phase on this laptop (V7_SPEC.md C2, C5, C6, C7, C8; README step 12).
#   sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/run_v7.sh [--lanes N]
#       N phases at once, each on its own processor: default 12, at most 16. Run it inside tmux. After any restart
#       (Windows, WSL, a closed window), open Ubuntu as root again and run the same command: it continues.
#       One Ctrl-C stops safely: no new phase starts and the running ones finish (this can take over an hour).
#   sh .../run_v7.sh --retry <phase>     after 'UNFINISHED: <phase>' (one more attempt, run alone; type it yourself)
#   sh .../run_v7.sh --lanes 12 --reviewed <phase>   after 'REVIEW: stand-in error', once the cloud has looked
#   sh .../run_v7.sh --plan-only         the plan and the state of every phase; runs nothing
#   sh .../run_v7.sh status              what is running, finished and waiting
#   sh .../run_v7.sh results             gather the results now (for example while a phase waits for --retry)
# Every start: the v6 code is put back in /tmp, the short self-test runs (restore, versions, constants, the gate on
# every lane processor, the gate's logs, one processor per lane, git, the probe records), then the runner. It refuses
# until the handoff, your approval and its import are complete (README steps 9-11). When every phase is done it gathers
# the results and prints a line starting 'Finished'.
set -u
PKG=$(cd "$(dirname "$0")" && pwd -P)
. "$PKG/lib/common_v7.sh"
require_root
require_repo
LIBPY="$PKG/lib/runner_v7.py"
case "${1:-}" in
    status)
        exec $CLEAN_ENV "$(system_python)" -I "$LIBPY" status;;
    results)
        place_code || exit 1
        $CLEAN_ENV "$PY" -I "$PKG/lib/make_results_v7.py" results --partial
        exit $?;;
esac
require_no_identity
LANES=12
RETRY=""
ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --lanes) LANES=${2:-}; shift 2;;
        --retry) RETRY=${2:-}; shift 2;;
        --reviewed) ARGS="$ARGS --reviewed ${2:-}"; shift 2;;
        --plan-only) ARGS="$ARGS --plan-only"; shift;;
        *) echo "refused: unknown argument '$1' (see the first lines of this file)"; exit 1;;
    esac
done
case "$LANES" in
    ''|*[!0-9]*) echo "refused: --lanes takes a number from 1 to 16"; exit 1;;
esac
if [ "$LANES" -lt 1 ] || [ "$LANES" -gt 16 ]; then echo "refused: --lanes takes a number from 1 to 16"; exit 1; fi
mkdir -p "$R7"
if ! place_code; then
    echo "refused: restore: the v6 code could not be put back in /tmp (run restore_v7.sh); nothing was started"
    exit 1
fi
if [ ! -x "$PY" ]; then echo "refused: $VENV/bin/python does not exist (README steps 5-6)"; exit 1; fi
if [ -n "$RETRY" ]; then
    # --retry must be typed at a terminal: standard input is passed on, never replaced.
    $CLEAN_ENV "$PY" -I "$LIBPY" retry "$RETRY" --argv "run_v7.sh --retry $RETRY"
    CODE=$?
    [ "$CODE" = 10 ] && exit 0
    [ "$CODE" = 0 ] || exit "$CODE"
fi
case " $ARGS " in
    *" --plan-only "*)
        # shellcheck disable=SC2086
        exec $CLEAN_ENV "$PY" -I "$LIBPY" run --lanes "$LANES" $ARGS;;
esac
echo "== Stage 2 v7 run, $(date -u +%FT%TZ), commit $(git -C "$REPO" rev-parse --short HEAD), lanes $LANES"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$R7/selftest"
$CLEAN_ENV "$PY" -I "$PKG/lib/selftest_v7.py" short 2>&1 | tee "$R7/selftest/report_short_$STAMP.txt" | \
    grep -E '^(FAIL|READY|NOT READY)'
if ! grep -q '^READY$' "$R7/selftest/report_short_$STAMP.txt"; then
    echo "refused: the short self-test is not READY (lines above; full report $R7/selftest/report_short_$STAMP.txt)."
    echo "Nothing was started. If it says 'waiting for handoff/approval', do README steps 9-11 first."
    exit 1
fi
# Ctrl-C reaches the runner (same terminal); this shell waits for it and then reports.
trap ':' INT TERM
trap '' HUP
# shellcheck disable=SC2086
$CLEAN_ENV "$PY" -I "$LIBPY" run --lanes "$LANES" $ARGS
CODE=$?
case "$CODE" in
    0)
        # C7's R1, information only (it routes nothing; a failure here is reported and the results are still made)
        $CLEAN_ENV "$PY" -I "$PKG/lib/selftest_v7.py" r1 || echo "WARN R1 (information only) did not finish; see above"
        if $CLEAN_ENV "$PY" -I "$PKG/lib/make_results_v7.py" results; then
            echo "Finished. Commit and push the results folder (README step 13), then push the run files (step 14)."
        else
            echo "Every phase ended, but gathering the results failed (lines above); paste them into the cloud session."
            exit 1
        fi;;
    3) echo "Stopped before the end (nothing failed). Run the same command again to continue.";;
    4) echo "WAITING FOR RETRY (lines above): run_v7.sh --retry <phase> continues it.";;
    5) echo "WAITING FOR REVIEW (lines above): restart with --reviewed <phase> once the cloud has looked.";;
    *) echo "Stopped: a check or a phase failed (lines above). Nothing was deleted. Paste the output into the cloud session.";;
esac
exit "$CODE"
