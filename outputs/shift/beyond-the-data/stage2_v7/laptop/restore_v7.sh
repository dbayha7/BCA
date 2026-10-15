#!/bin/sh
# Stage 2, version 7: put the run files and the v6 code where the pinned code expects them (V7_SPEC.md C4, C6).
#   usage (inside WSL, as root):  sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/restore_v7.sh
# It fetches ONLY the run-files branch claude/stage2-run-files (about 0.8 GB the first time; git may ask for your
# GitHub user name and token, README step 4, and shows its progress), checks that the pinned
# snapshot commit is on it and that its list of fingerprints (RUNFILES_SHA256SUMS) is the pinned one, and extracts
# runs/ with 'git archive <commit> runs | tar -x --skip-old-files': no file is ever overwritten, the cloud's logs and
# the files of phases the cloud did not finish are left out. Then it checks every extracted file against the list,
# places the v6 code and fallback.py in /tmp (WSL empties /tmp at every start) and links /tmp/claude-0/bcavenv to
# /root/bcavenv. Before the handoff the snapshot is 901c877; after it, the one HANDOFF.json names (README step 9).
# It never touches the main branch's index or tracked files (runs/ is ignored by git). Safe to run any number of times.
set -u
PKG=$(cd "$(dirname "$0")" && pwd -P)
. "$PKG/lib/common_v7.sh"
require_root
require_repo
mkdir -p "$R7"
link_venv || exit 1
echo "== Stage 2 v7 restore, $(date -u +%FT%TZ), commit $(git -C "$REPO" rev-parse --short HEAD)"
# The download runs here, in the shell, so that git can ask for your GitHub sign-in (user name and token) and shows
# its progress; the Python part then works on the downloaded branch only (--no-fetch).
FETCH=1
for a in "$@"; do
    case "$a" in --no-fetch|--v7-test) FETCH=0;; esac
done
if [ "$FETCH" = 1 ]; then
    echo "Downloading the run files (branch $RUNFILES_BRANCH; about 0.8 GB the first time; git shows its progress)"
    n=0
    until git -C "$REPO" fetch --no-tags origin "+refs/heads/$RUNFILES_BRANCH:refs/remotes/origin/$RUNFILES_BRANCH"; do
        n=$((n + 1))
        if [ "$n" -ge 5 ]; then
            echo "NOT READY (restore: the run files could not be downloaded five times: network or GitHub sign-in;"
            echo "see the git lines above). Nothing was changed. Run this again; if it fails the same way, paste the"
            echo "output into the cloud session."
            exit 1
        fi
        echo "the download failed (try $n of 5); trying again in 30 seconds"
        sleep 30
    done
fi
$CLEAN_ENV "$(system_python)" -I "$PKG/lib/selftest_v7.py" "$@" --no-fetch restore
CODE=$?
if [ "$CODE" != 0 ]; then
    echo "NOT READY (restore): nothing was overwritten. Run this again; if it fails the same way, paste the output"
    echo "into the cloud session (README, 'If something stops')."
fi
exit "$CODE"
