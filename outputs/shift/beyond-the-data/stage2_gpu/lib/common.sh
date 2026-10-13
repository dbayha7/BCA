# Shared by restore.sh, selftest.sh and run_confirmatory.sh (sourced, not run). POSIX sh.
# The pinned Stage 2 code fixes these paths; none of them may change.
REPO=/home/user/bca
EXPECTED_PKG=$REPO/outputs/shift/beyond-the-data/stage2_gpu
VENV=${BCA_VENV:-/root/bcavenv}
LINK=/tmp/claude-0/bcavenv
OFFCLOUD=$REPO/runs/shift_explore/stage2_offcloud
BRANCH=claude/main-bca-connection-x4o46s
# A clean environment for every Python process this package starts: nothing from the caller's shell (for example a
# JAX or XLA setting) can reach the pinned code. launch.sh then adds exactly what the gate records.
CLEAN_ENV="env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root LANG=C.UTF-8 LC_ALL=C.UTF-8"

require_root() {
    if [ "$(id -u)" != 0 ] || [ "${HOME:-}" != /root ]; then
        echo "refused: run this as root inside WSL with HOME=/root (open it with: wsl -d Ubuntu-22.04 -u root)"
        exit 1
    fi
}

require_repo() {
    if [ "$PKG" != "$EXPECTED_PKG" ]; then
        echo "refused: this package must sit at $EXPECTED_PKG (it is at $PKG); the pinned code's paths are fixed"
        exit 1
    fi
    if ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
        echo "refused: $REPO is not a git clone of the repository"
        exit 1
    fi
}

link_venv() {
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
    if [ ! -x "$LINK/bin/python" ]; then
        echo "WARN $VENV/bin/python does not exist yet (README, step 6)"
    fi
    return 0
}

system_python() {
    for p in /usr/bin/python3 "$LINK/bin/python"; do
        if [ -x "$p" ]; then echo "$p"; return 0; fi
    done
    echo "refused: no Python found (apt-get install -y python3)" >&2
    exit 1
}

do_restore() {
    # Every call: /tmp is emptied when WSL restarts, and runs/ must hold exactly the pinned files.
    mkdir -p "$OFFCLOUD"
    rm -f "$OFFCLOUD/restore_last.json"
    link_venv || return 1
    $CLEAN_ENV "$(system_python)" -I "$PKG/lib/verify_payload.py" --place --json "$OFFCLOUD/restore_last.json" "$@"
}
