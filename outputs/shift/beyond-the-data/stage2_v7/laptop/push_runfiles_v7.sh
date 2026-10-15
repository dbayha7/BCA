#!/bin/sh
# Stage 2, version 7: push the laptop's new run files to a new orphan branch (the user's decision Q6; README step 14).
#   usage (inside WSL, as root, typed at the terminal, after step 13):
#     sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/push_runfiles_v7.sh
# What it pushes: every file under runs/shift_explore/stage2_v7/ (the ledger, the probe records, the self-test reports,
# the reference copies in refs/, the logs, the moved files of interrupted phases; lock files left out) and every file
# the laptop's phases wrote under runs/shift_explore/stage2/ (model files included) that is not in the cloud's final
# snapshot, with a list of their fingerprints (RUNFILES_SHA256SUMS). It refuses anything else, refuses when a file the
# cloud made has changed, and refuses a file under runs/shift_explore/stage2/ that no v7 phase's ledger names. It uses
# a temporary git index (your checkout, its index and the main branch are not touched), commits as you
# (David Bayha <35092758+dbayha7@users.noreply.github.com>) in two parts, and pushes branch stage2-v7-run-files,
# retrying up to five times. Run it again if the network failed: it rebuilds the same commits.
set -u
PKG=$(cd "$(dirname "$0")" && pwd -P)
. "$PKG/lib/common_v7.sh"
require_root
require_repo
TEST=""
if [ "${1:-}" = "--v7-test" ] && [ -n "${2:-}" ]; then
    TEST="--v7-test $2"
fi
if [ -z "$TEST" ] && [ ! -t 0 ]; then
    echo "refused: run this yourself at a terminal (standard input is not one)"
    exit 1
fi
PYX="$PY"
[ -x "$PYX" ] || PYX=$(system_python)
PLAN=$(mktemp -d /tmp/v7-runfiles-XXXXXX)
trap 'rm -rf "$PLAN"' EXIT
# shellcheck disable=SC2086
$CLEAN_ENV "$PYX" -I "$PKG/lib/make_results_v7.py" $TEST runfiles-plan --dir "$PLAN" || exit 1
# shellcheck disable=SC1091
. "$PLAN/env.sh"
cd "$REPO_DIR" || exit 1
export GIT_INDEX_FILE="$PLAN/index"
export GIT_AUTHOR_NAME="$AUTHOR_NAME" GIT_AUTHOR_EMAIL="$AUTHOR_EMAIL" GIT_AUTHOR_DATE="$COMMIT_DATE"
export GIT_COMMITTER_NAME="$AUTHOR_NAME" GIT_COMMITTER_EMAIL="$AUTHOR_EMAIL" GIT_COMMITTER_DATE="$COMMIT_DATE"
git read-tree --empty || exit 1
git update-index --add -z --stdin < "$PLAN/r7.list" || exit 1
T1=$(git write-tree) || exit 1
C1=$(git commit-tree "$T1" -m "Stage 2 v7 run files, part 1 of 2: the laptop's records (runs/shift_explore/stage2_v7: ledger, probe records, reference copies, logs; $R7_COUNT files)") || exit 1
git update-index --add -z --stdin < "$PLAN/out.list" || exit 1
H=$(git hash-object -w "$PLAN/RUNFILES_SHA256SUMS") || exit 1
git update-index --add --cacheinfo "100644,$H,RUNFILES_SHA256SUMS" || exit 1
T2=$(git write-tree) || exit 1
C2=$(git commit-tree "$T2" -p "$C1" -m "Stage 2 v7 run files, part 2 of 2: the laptop's new outputs under runs/shift_explore/stage2 ($OUT_COUNT files, model files included) and the SHA-256 of every file

The cloud's final snapshot is $SNAPSHOT on claude/stage2-run-files; nothing it holds is repeated here.") || exit 1
unset GIT_INDEX_FILE
# shellcheck disable=SC2086
$CLEAN_ENV "$PYX" -I "$PKG/lib/make_results_v7.py" $TEST runfiles-check --tree "$T2" --dir "$PLAN" || exit 1
REMOTE=$(git ls-remote origin "refs/heads/$ORPHAN_BRANCH" | cut -f1)
if [ -n "$REMOTE" ]; then
    if [ "$REMOTE" = "$C2" ]; then
        echo "already pushed: $ORPHAN_BRANCH is at $C2"
        exit 0
    fi
    echo "refused: origin already has a branch $ORPHAN_BRANCH at another commit ($REMOTE); nothing was pushed."
    echo "Paste this into the cloud session."
    exit 1
fi
# Retries in POSIX sh (dash has no '**'): the wait doubles from 5 seconds (1 second in a test fixture).
n=0
DELAY=5
[ -n "$TEST" ] && DELAY=1
until git push origin "$C2:refs/heads/$ORPHAN_BRANCH"; do
    n=$((n + 1))
    if [ "$n" -ge 5 ]; then
        echo "the push failed five times (network or sign-in?); run this again later (it rebuilds the same commits)"
        exit 1
    fi
    echo "the push failed (try $n of 5); trying again in $DELAY seconds"
    sleep "$DELAY"
    DELAY=$((DELAY * 2))
done
echo "pushed $ORPHAN_BRANCH at $C2 ($R7_COUNT records under stage2_v7, $OUT_COUNT new outputs). Tell the cloud."
