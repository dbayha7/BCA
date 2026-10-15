#!/bin/sh
# Run the confirmatory phases of Stage 2 (fresh seeds 5-9) on this machine, then bundle the results to push.
#   usage (inside WSL, as root, preferably inside tmux):
#     sh /home/user/bca/outputs/shift/beyond-the-data/stage2_gpu/run_confirmatory.sh --lanes 1|2 [--cpus A,B]
#     sh .../run_confirmatory.sh --lanes 1 --plan-only        # show the plan and what is finished; run nothing
# --lanes is the number of phases at once. The pre-registration allows at most 2 processes at once in all, the cloud
# included: use the number the cloud gives you.
# Every call: restore and verify the payload (restore.sh); refuse without the cloud's two late files; refuse when git's
# identity here is the approver's, the clone is shallow, or a tracked file is changed; check the gate on every
# processor a lane uses; then run the phases in the pre-registered order through the pinned launch.sh (each phase
# once; a rerun continues where the last one stopped). When the pooled look exists, the results are copied to
# outputs/shift/beyond-the-data/stage2_gpu/results/ for you to commit and push.
# To stop safely: press Ctrl-C once and wait; the phases already running finish first.
set -u
PKG=$(cd "$(dirname "$0")" && pwd -P)
. "$PKG/lib/common.sh"
require_root
require_repo
echo "== Stage 2 confirmatory run, $(date -u +%FT%TZ), commit $(git -C "$REPO" rev-parse --short HEAD)"
if ! do_restore --require-late; then
    echo "refused: the payload did not verify, or the cloud's late files are not here yet (see the lines above)."
    echo "Nothing was started. If the late files are missing: wait for the cloud's message, then git pull and run again."
    exit 1
fi
if [ ! -x "$LINK/bin/python" ]; then
    echo "refused: $VENV/bin/python does not exist (README, step 6)"
    exit 1
fi
EMAIL=$(git -C "$REPO" config --get user.email || true)
case "$EMAIL" in
    *35092758+dbayha7@users.noreply.github.com*|*noreply@anthropic.com*)
        echo "refused: git's user.email here is $EMAIL. The Stage 2 code refuses to run under the identity that made"
        echo "the approval commit. Remove it (git config --global --unset user.email) and commit the results later with"
        echo "git -c user.name=... -c user.email=... commit (README, step 11)."
        exit 1;;
esac
$CLEAN_ENV BCA_OFFCLOUD_TEST="${BCA_OFFCLOUD_TEST:-}" BCA_OFFCLOUD_TEST_PREFLIGHT="${BCA_OFFCLOUD_TEST_PREFLIGHT:-}" \
    "$LINK/bin/python" -I "$PKG/lib/runner.py" "$@"
CODE=$?
case " $* " in *" --plan-only "*) exit $CODE;; esac
if [ "$CODE" = 0 ]; then
    $CLEAN_ENV "$LINK/bin/python" -I "$PKG/lib/make_results.py" || exit 1
    echo "Finished. Commit and push the results folder (README, step 11)."
elif [ "$CODE" = 3 ]; then
    echo "Stopped before the end (nothing failed). Run the same command again to continue."
else
    echo "Stopped: a check or a phase failed (lines above). Nothing was deleted. Paste the output to the cloud."
fi
exit $CODE
