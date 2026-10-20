#!/bin/sh
# Near-80 (version 9, main setting): the desktop entry. DRAFT, not frozen. POSIX sh. Run it as root (HOME=/root)
# inside tmux, from anywhere:
#   sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh run [--lanes N]
#       the official run (default 10 lanes, 1-16); after any restart, run the same command: it continues.
#       One Ctrl-C stops safely (no new phase starts; the running ones finish). A second Ctrl-C stops them now
#       (they rerun on the next start).
#   sh .../run_near80.sh plumbing [--lanes N]   the unit tests, a 40-step smoke of every phase kind on seed 0, the
#       fresh-seed code path and the reproduction check (scratch only; required once before 'run'; a failure other
#       than a reproduction difference may be run again)
#   sh .../run_near80.sh status                 what is running, done, waiting, stopped and skipped (reads only)
#   sh .../run_near80.sh selftest               the short self-test alone (prints READY or NOT READY)
#   sh .../run_near80.sh plan                   the phase plan and each phase's state; runs nothing
#   sh .../run_near80.sh retry PHASE_ID         one more attempt of an UNFINISHED phase (type it yourself)
#   sh .../run_near80.sh budget-continue STAGE  lifts a budget pause by one forecast of that stage (type it yourself)
#   sh .../run_near80.sh report                 the halt report, when the run stopped early
#   sh .../run_near80.sh results                copies every record (no model file) into near80/results-<utc>/
#       (also done at the end of a finished run)
#   sh .../run_near80.sh stub --fixture DIR [--lanes N] [--seeds a,b] [--refuse a]   stub mode on a test fixture
#       outside the repository (no training: every learner is replaced by a stand-in that takes no gradient step)
# Every command except status, plan and stub first checks: root with HOME=/root, the repository, git's identity
# (it must not be the approver's or the cloud's); run, plumbing, selftest, retry and report then the version 7
# placement of the version 6 code and the short self-test; then the runner. Nothing trains before the owner's approval
# is imported (every phase checks it again). Version 8's checks on version 7's run are gone (it finished).
# Exit codes: 0 finished; 3 stopped safely before the end; 4 waiting for a retry; 5 paused by the budget guard;
# 97 refused; other: a phase or a check failed (nothing deleted).
set -u
REPO=/home/user/bca
N80=$REPO/outputs/shift/beyond-the-data/near80
CODE=$N80/code
V7_LIB=$REPO/outputs/shift/beyond-the-data/stage2_v7/laptop/lib
V7_PLACE=$V7_LIB/selftest_v7.py
VENV=/root/bcavenv
LINK=/tmp/claude-0/bcavenv
PY=$LINK/bin/python
APPROVER_EMAIL=35092758+dbayha7@users.noreply.github.com
CLEAN_ENV="env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root LANG=C.UTF-8 LC_ALL=C.UTF-8"
RUN_ENV="$CLEAN_ENV PYTHONPATH=$REPO PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/nonexistent/near80_no_pycache JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 NEAR80_HOST=desktop"
HERE=$(cd "$(dirname "$0")" && pwd -P)

refuse() { echo "refused: $*"; exit 97; }

require_root() {
    if [ "$(id -u)" != 0 ] || [ "${HOME:-}" != /root ]; then
        refuse "run this as root with HOME=/root (the desktop entry; version 9 has no other)"
    fi
}

require_repo() {
    if [ "$HERE" != "$N80" ]; then
        refuse "this script must sit at $N80 (it is at $HERE); the frozen code's paths are fixed"
    fi
    if ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
        refuse "$REPO is not a git clone of the repository"
    fi
}

require_no_identity() {
    # version 7's rule: the phases refuse to run under the approver's identity or the cloud's.
    EMAIL=$(git -C "$REPO" config --get user.email || true)
    case "$EMAIL" in
        *"$APPROVER_EMAIL"*|*noreply@anthropic.com*)
            echo "refused: git's user.email here is $EMAIL. Remove it (git config --global --unset user.email) and"
            echo "commit later with git -c user.name=... -c user.email=... commit."
            exit 97;;
    esac
}

link_venv() {
    # /tmp/claude-0/bcavenv -> /root/bcavenv (version 7's link; /tmp may be emptied at every start)
    mkdir -p /tmp/claude-0 && chmod 700 /tmp/claude-0
    if [ -L "$LINK" ]; then
        [ "$(readlink "$LINK")" = "$VENV" ] || ln -sfn "$VENV" "$LINK"
    elif [ -e "$LINK" ]; then
        if [ "$(readlink -f "$LINK")" != "$(readlink -f "$VENV")" ]; then
            echo "FAIL $LINK exists and is not a link to $VENV; remove it and run again"
            return 1
        fi
    else
        ln -s "$VENV" "$LINK"
    fi
    return 0
}

system_python() {
    for p in /usr/bin/python3 "$PY"; do
        if [ -x "$p" ]; then echo "$p"; return 0; fi
    done
    refuse "no Python found"
}

place_code() {
    # version 7's placement of the version 6 code (it never overwrites a file that differs)
    link_venv || return 1
    $CLEAN_ENV "$(system_python)" -I "$V7_PLACE" place
}

RUNNER="$CODE/near80_runner.py"
CMD=${1:-}
[ $# -gt 0 ] && shift
case "$CMD" in
    status|plan)
        # read only, standard library only: any Python 3.8 or later
        exec $RUN_ENV "$(system_python)" "$RUNNER" "$CMD";;
    stub)
        exec $RUN_ENV NEAR80_STUB_PYTHONPATH="${NEAR80_STUB_PYTHONPATH:-}" "$PY" "$RUNNER" stub "$@";;
    budget-continue|results)
        require_root
        require_repo
        require_no_identity
        link_venv || refuse "the Python environment link could not be made"
        [ -x "$PY" ] || refuse "$PY does not exist"
        exec $RUN_ENV "$PY" "$RUNNER" "$CMD" "$@";;
    run|plumbing|selftest|retry|report) ;;
    *) echo "usage: sh $N80/run_near80.sh run|plumbing|status|selftest|plan|retry PHASE_ID|budget-continue STAGE|"
       echo "       report|results|stub --fixture DIR"
       exit 1;;
esac
require_root
require_repo
require_no_identity
if ! place_code; then
    refuse "the version 6 code could not be placed (version 7's placement); nothing was started"
fi
if [ ! -x "$PY" ]; then refuse "$PY does not exist"; fi
LANES=10
ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --lanes) LANES=${2:-}; shift 2;;
        *) ARGS="$ARGS $1"; shift;;
    esac
done
case "$LANES" in
    ''|*[!0-9]*) refuse "--lanes takes a number from 1 to 16";;
esac
if [ "$LANES" -lt 1 ] || [ "$LANES" -gt 16 ]; then refuse "--lanes takes a number from 1 to 16"; fi
FOR=$CMD
[ "$CMD" = report ] && FOR=selftest
$RUN_ENV "$PY" "$RUNNER" selftest --for "$FOR" --lanes "$LANES"
if [ $? -ne 0 ]; then
    echo "refused: the short self-test is not READY (lines above); nothing was started"
    exit 97
fi
[ "$CMD" = selftest ] && exit 0
case "$CMD" in
    run|plumbing)
        # Ctrl-C reaches the runner (same terminal); this shell waits for it and then reports.
        trap ':' INT TERM
        trap '' HUP
        $RUN_ENV "$PY" "$RUNNER" "$CMD" --lanes "$LANES"
        CODE_=$?;;
    retry)
        # shellcheck disable=SC2086
        $RUN_ENV "$PY" "$RUNNER" retry $ARGS
        CODE_=$?;;
    report)
        $RUN_ENV "$PY" "$RUNNER" report
        CODE_=$?;;
esac
case "$CODE_" in
    0)
        if [ "$CMD" = run ]; then
            if $RUN_ENV "$PY" "$RUNNER" results; then
                echo "Finished: every phase is done, stopped or skipped on record. Commit and push the results folder"
                echo "with the commands above (README, 'After the run'), then push the run files:"
                echo "sh $N80/push_runfiles_near80.sh"
            else
                echo "Every phase ended, but gathering the results failed (lines above); paste them into the session."
                exit 1
            fi
        fi;;
    3) echo "Stopped before the end (nothing failed). Type the same command again inside tmux to continue.";;
    4) echo "WAITING FOR RETRY (lines above): type sh $N80/run_near80.sh retry <phase>, then run again inside tmux.";;
    5) echo "PAUSED: the budget guard (lines above). To continue, type: sh $N80/run_near80.sh budget-continue <stage>"
       echo "then start the run again inside tmux: sh $N80/run_near80.sh run --lanes $LANES";;
    97) echo "Refused (lines above). Nothing was started.";;
    *) echo "Stopped: a check or a phase failed (lines above). Nothing was deleted.";;
esac
exit "$CODE_"
