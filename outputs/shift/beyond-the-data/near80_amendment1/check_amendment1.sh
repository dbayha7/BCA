#!/bin/sh
# Near-80 amendment 1: the desktop check before approval (reads only; trains nothing; PREREGISTRATION.md section 19.7).
# Type, as root inside Ubuntu (wsl -d Ubuntu-24.04 -u root), after git pull:
#   sh /home/user/bca/outputs/shift/beyond-the-data/near80_amendment1/check_amendment1.sh
# About 20-30 minutes. Its last line is "amendment 1 desktop check: PASSED" or "... FAILED"; a line starting with
# "refused:" means nothing was checked. It writes only near80_amendment1/desktop_check_<utc>.json (+ .sha256), which
# you commit with your approval. --no-tests skips the four unit test files (cloud tests only: an official check
# without them cannot pass, and approve_local_near80.py accepts only a result with all four run and passed).
set -u
REPO=/home/user/bca
HERE=$REPO/outputs/shift/beyond-the-data/near80_amendment1
if [ "$(id -u)" != 0 ] || [ "${HOME:-}" != /root ]; then echo "refused: run this as root with HOME=/root"; exit 97; fi
# version 7's link /tmp/claude-0/bcavenv -> /root/bcavenv (as run_near80.sh makes it; /tmp may be emptied at a start)
mkdir -p /tmp/claude-0 && chmod 700 /tmp/claude-0
[ -e /tmp/claude-0/bcavenv ] || ln -s /root/bcavenv /tmp/claude-0/bcavenv
PY=/tmp/claude-0/bcavenv/bin/python
[ -x "$PY" ] || { echo "refused: $PY does not exist"; exit 97; }
exec env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONPATH=$REPO PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/nonexistent/near80_no_pycache JAX_PLATFORMS=cpu \
    OMP_NUM_THREADS=1 nice -n 5 "$PY" "$HERE/check_amendment1.py" "$@"
