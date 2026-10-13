#!/bin/sh
# Put the Stage 2 payload and the committed Stage 2 files where the pinned code expects them, and verify every hash
# (the payload manifest, then the pre-registration's own pins). Never overwrites a file that differs; refuses instead.
# Safe to run any number of times. selftest.sh and run_confirmatory.sh run it first, every time, because WSL empties
# /tmp whenever it restarts.
#   usage (inside WSL, as root):  sh /home/user/bca/outputs/shift/beyond-the-data/stage2_gpu/restore.sh
set -eu
PKG=$(cd "$(dirname "$0")" && pwd -P)
. "$PKG/lib/common.sh"
require_root
require_repo
do_restore "$@"
