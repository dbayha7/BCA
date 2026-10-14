#!/bin/sh
# Stage 2, version 7: the laptop self-test (V7_SPEC.md C11). It runs no BCA code and starts no phase.
#   sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/selftest_v7.sh            the full self-test
#       (README step 8, once): every check, and the probes P1-P3, which also make the seed 2-4 reference copies.
#       About 25-40 minutes. Paste EVERYTHING it prints into the cloud session.
#   sh .../selftest_v7.sh --short     the short self-test (after every restore; run_v7.sh runs it before every run)
#   sh .../selftest_v7.sh --remake <probe>    only when the cloud asks (a damaged reference file, C7)
# The last lines are READY, READY (waiting for handoff/approval) or NOT READY (<what failed>), then the probe results
# and a block to paste. A copy is saved as /home/user/bca/runs/shift_explore/stage2_v7/selftest/report_*.txt.
set -u
PKG=$(cd "$(dirname "$0")" && pwd -P)
. "$PKG/lib/common_v7.sh"
require_root
require_repo
FORM=full
EXTRA=""
case "${1:-}" in
    --short) FORM=short;;
    --remake)
        if [ -z "${2:-}" ]; then echo "usage: selftest_v7.sh --remake <probe, e.g. p1_seed2_stall>"; exit 1; fi
        FORM=remake; EXTRA="--probe $2";;
    "") ;;
    *) echo "usage: selftest_v7.sh [--short | --remake <probe>]"; exit 1;;
esac
mkdir -p "$R7/selftest"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPORT="$R7/selftest/report_${FORM}_$STAMP.txt"
STATUS="$R7/selftest/.status_$STAMP"
echo 1 > "$STATUS"
{
    echo "== Stage 2 v7 self-test ($FORM) $STAMP, commit $(git -C "$REPO" rev-parse --short HEAD)"
    if ! place_code; then
        echo "NOT READY (restore: the v6 code could not be placed; run restore_v7.sh)"
    elif [ ! -x "$PY" ]; then
        echo "NOT READY (versions: $VENV/bin/python does not exist; README steps 5-6)"
    else
        # shellcheck disable=SC2086
        $CLEAN_ENV "$PY" -I "$PKG/lib/selftest_v7.py" "$FORM" $EXTRA
        echo $? > "$STATUS"
    fi
} 2>&1 | tee "$REPORT"
CODE=$(cat "$STATUS")
rm -f "$STATUS"
echo "(saved to $REPORT)"
exit "$CODE"
