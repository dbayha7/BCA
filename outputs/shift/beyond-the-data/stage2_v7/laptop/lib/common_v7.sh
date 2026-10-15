# Stage 2, version 7: shared by restore_v7.sh, selftest_v7.sh, run_v7.sh and push_runfiles_v7.sh (sourced, not run).
# POSIX sh. The pinned code fixes every path below; none of them may change (V7_SPEC.md C4).
REPO=/home/user/bca
V7=$REPO/outputs/shift/beyond-the-data/stage2_v7
EXPECTED_PKG=$V7/laptop
R7=$REPO/runs/shift_explore/stage2_v7
VENV=/root/bcavenv
LINK=/tmp/claude-0/bcavenv
PY=$LINK/bin/python
BRANCH=claude/main-bca-connection-x4o46s
RUNFILES_BRANCH=claude/stage2-run-files
APPROVER_EMAIL=35092758+dbayha7@users.noreply.github.com
# Every Python process these scripts start gets a clean environment: nothing from the caller's shell (a JAX, XLA or
# OpenBLAS setting, for example) reaches it. launch_v7.sh then gives each phase exactly its eight variables.
CLEAN_ENV="env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root LANG=C.UTF-8 LC_ALL=C.UTF-8"

require_root() {
    if [ "$(id -u)" != 0 ] || [ "${HOME:-}" != /root ]; then
        echo "refused: run this as root inside WSL with HOME=/root (open Ubuntu with: wsl -d Ubuntu-24.04 -u root)"
        exit 1
    fi
}

require_repo() {
    if [ "$PKG" != "$EXPECTED_PKG" ]; then
        echo "refused: this package must sit at $EXPECTED_PKG (it is at $PKG); the pinned code's paths are fixed"
        exit 1
    fi
    if ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
        echo "refused: $REPO is not a git clone of the repository (README, step 4)"
        exit 1
    fi
}

require_no_identity() {
    # The v6 and v7 code refuse to run under the identity that made the approval commit, or the cloud's.
    EMAIL=$(git -C "$REPO" config --get user.email || true)
    case "$EMAIL" in
        *"$APPROVER_EMAIL"*|*noreply@anthropic.com*)
            echo "refused: git's user.email here is $EMAIL. Remove it (git config --global --unset user.email) and"
            echo "commit later with git -c user.name=... -c user.email=... commit (README, steps 10 and 13)."
            exit 1;;
    esac
}

link_venv() {
    # /tmp/claude-0/bcavenv -> /root/bcavenv (WSL empties /tmp at every start, so this runs every time).
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
        echo "WARN $VENV/bin/python does not exist yet (README, step 5)"
    fi
    return 0
}

system_python() {
    # Any Python 3.8 or later for the standard-library helpers (restore, placement); the study's own Python otherwise.
    for p in /usr/bin/python3 "$LINK/bin/python"; do
        if [ -x "$p" ]; then echo "$p"; return 0; fi
    done
    echo "refused: no Python found (apt-get install -y python3)" >&2
    exit 1
}

place_code() {
    # Puts the v6 code and fallback.py back at the scratchpad paths launch.sh fixes (V7_SPEC.md C4), never
    # overwriting a file that differs, and links the Python environment. Quiet unless something is wrong.
    link_venv || return 1
    $CLEAN_ENV "$(system_python)" -I "$PKG/lib/selftest_v7.py" place "$@"
}
