#!/bin/sh
# Stage 2 off-cloud self-test. Run it first, inside WSL as root, and paste everything it prints back to the cloud.
#   usage:  sh /home/user/bca/outputs/shift/beyond-the-data/stage2_gpu/selftest.sh [--cpu N]
# It restores and verifies the payload (restore.sh), then checks, in this order: the gate's own check of its
# pre-registration on this processor (the anchors; nothing else can run if it fails), the versions, ThrusterChain's
# check_env, the approval and every pin through the pinned code (read only), git, the one-processor rule, the late
# files, and (information only) the graphics card. It runs no Stage 2 phase. The report is also saved under
# /home/user/bca/runs/shift_explore/stage2_offcloud/.
set -u
PKG=$(cd "$(dirname "$0")" && pwd -P)
. "$PKG/lib/common.sh"
require_root
require_repo
CPU=0
if [ "${1:-}" = "--cpu" ] && [ -n "${2:-}" ]; then CPU=$2; fi
mkdir -p "$OFFCLOUD"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPORT="$OFFCLOUD/selftest_$STAMP.txt"
STATUS="$OFFCLOUD/.selftest_status"
echo 1 > "$STATUS"
{
    echo "== Stage 2 off-cloud self-test $STAMP, commit $(git -C "$REPO" rev-parse --short HEAD), package $(sha256sum "$PKG/lib/selftest_checks.py" | cut -c1-12)"
    do_restore
    if [ -x "$LINK/bin/python" ]; then
        $CLEAN_ENV "$LINK/bin/python" -I "$PKG/lib/selftest_checks.py" --cpu "$CPU" --json "$OFFCLOUD/selftest_$STAMP.json"
        echo $? > "$STATUS"
    else
        echo "FAIL environment: $VENV/bin/python does not exist (README, step 6)"
        echo "RESULT: NOT READY (1 failed: environment)"
    fi
} 2>&1 | tee "$REPORT"
echo "(saved to $REPORT)"
exit "$(cat "$STATUS")"
