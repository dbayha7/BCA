#!/bin/sh
# Near-80 (version 9): push the desktop's run files to their own branch (version 7's push_runfiles_v7.sh, restated).
#   usage (inside Ubuntu, as root, typed at the terminal, after the run printed "Finished" and the results were pushed):
#     sh /home/user/bca/outputs/shift/beyond-the-data/near80_v9/push_runfiles_near80_v9.sh
# What it pushes: every file under runs/shift_explore/near80_v9/ (the records, the ledger, the logs, the moved files of
# interrupted phases, and the model files: critics, cloned and trained actors, version 6's bands), with a list of
# their fingerprints (RUNFILES_SHA256SUMS). Left out: lock files, temporary files, and the training files (*.hdf5),
# whose fingerprints the data records hold and which the code regenerates from the seed's log. It uses a temporary git
# index (your checkout, its index and its branch are not touched), commits as you (David Bayha
# <35092758+dbayha7@users.noreply.github.com>) as one commit with no parent, and pushes branch
# claude/near80-v9-run-files, retrying up to five times. Run it again if the network failed: it rebuilds the same commit.
set -u
REPO=/home/user/bca
N80=$REPO/outputs/shift/beyond-the-data/near80_v9
PY=/tmp/claude-0/bcavenv/bin/python
CLEAN_ENV="env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root LANG=C.UTF-8 LC_ALL=C.UTF-8"
RUN_ENV="$CLEAN_ENV PYTHONPATH=$REPO PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/nonexistent/near80_no_pycache JAX_PLATFORMS=cpu OMP_NUM_THREADS=1"
if [ "$(id -u)" != 0 ] || [ "${HOME:-}" != /root ]; then
    echo "refused: run this as root with HOME=/root"
    exit 97
fi
if [ ! -t 0 ]; then
    echo "refused: run this yourself at a terminal (standard input is not one)"
    exit 97
fi
if [ ! -x "$PY" ]; then
    echo "refused: $PY does not exist; run sh $N80/run_near80_v9.sh selftest first (it makes the link)"
    exit 97
fi
PLAN=$(mktemp -d /tmp/near80v9-runfiles-XXXXXX)
trap 'rm -rf "$PLAN"' EXIT
$RUN_ENV "$PY" "$N80/code/near80_runner.py" runfiles-plan --dir "$PLAN" || exit 1
# shellcheck disable=SC1091
. "$PLAN/env.sh"
cd "$REPO_DIR" || exit 1
export GIT_INDEX_FILE="$PLAN/index"
export GIT_AUTHOR_NAME="$AUTHOR_NAME" GIT_AUTHOR_EMAIL="$AUTHOR_EMAIL" GIT_AUTHOR_DATE="$COMMIT_DATE"
export GIT_COMMITTER_NAME="$AUTHOR_NAME" GIT_COMMITTER_EMAIL="$AUTHOR_EMAIL" GIT_COMMITTER_DATE="$COMMIT_DATE"
git read-tree --empty || exit 1
git update-index --add -z --stdin < "$PLAN/files.list" || exit 1
H=$(git hash-object -w "$PLAN/RUNFILES_SHA256SUMS") || exit 1
git update-index --add --cacheinfo "100644,$H,RUNFILES_SHA256SUMS" || exit 1
T=$(git write-tree) || exit 1
C=$(git commit-tree "$T" -m "Near-80 (version 9) run files: runs/shift_explore/near80_v9 ($COUNT files, model files included; training files left out) and the SHA-256 of every file") || exit 1
unset GIT_INDEX_FILE
REMOTE=$(git ls-remote origin "refs/heads/$RUNFILES_BRANCH" | cut -f1)
if [ -n "$REMOTE" ]; then
    if [ "$REMOTE" = "$C" ]; then
        echo "already pushed: $RUNFILES_BRANCH is at $C"
        exit 0
    fi
    echo "refused: origin already has a branch $RUNFILES_BRANCH at another commit ($REMOTE); nothing was pushed."
    echo "Paste this into the session."
    exit 1
fi
n=0
DELAY=5
until git push origin "$C:refs/heads/$RUNFILES_BRANCH"; do
    n=$((n + 1))
    if [ "$n" -ge 5 ]; then
        echo "the push failed five times (network or sign-in?); run this again later (it rebuilds the same commit)"
        exit 1
    fi
    echo "the push failed (try $n of 5); trying again in $DELAY seconds"
    sleep "$DELAY"
    DELAY=$((DELAY * 2))
done
echo "pushed $RUNFILES_BRANCH at $C ($COUNT files). Tell the session."
