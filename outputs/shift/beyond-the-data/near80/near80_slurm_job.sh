#!/bin/sh
# Near-80 (version 9, main setting): the Slurm job script run_near80_cluster.sh submits. DRAFT, not frozen. POSIX sh.
#   near80_slurm_job.sh plumbing LANES          the plumbing (one job)
#   near80_slurm_job.sh run LANES SEED          one fresh seed's phases (both worlds, up to the actor training;
#                                               the runner ends once only lock-dependent phases remain)
#   near80_slurm_job.sh group LANES             the group's lock, every evaluation and the look
#   near80_slurm_job.sh stub LANES FIXTURE [SEEDS]   stub mode on a test fixture outside the clone (trains nothing)
# It never runs as root, refuses the approver's and the cloud's git identities, passes the job's memory to the
# runner (NEAR80_MEM_TOTAL_KB from SLURM_MEM_PER_NODE) and runs the same frozen runner with the same checks as the
# desktop entry. One Slurm SIGTERM (a cancel) is a safe stop: the running phases finish, nothing new starts.
#
# The pinned dependency path. The gate's pre-registration (runs/shift/preregistration.json, pinned by hash) records
# the shift dependencies at the literal path /root/.cache/bca-shift/py, and experiments.shift.acceptance.check_software
# refuses any other real path (a symlink included), so every real phase (data_main onwards) needs that path to exist
# on the compute node. A job cannot write /root, but an unprivileged user namespace can mount over it: the script
# re-executes itself inside `unshare --map-root-user --mount` (a tmpfs on /root, BCA_SHIFT_DEPS bound at the pinned
# path), then drops into a nested user namespace that maps the job's real uid back (`unshare --map-user`), so that
# id -u is the real user again and the "never as root" rule below still holds; the real /root is never touched.
# NEAR80_NS=1 marks the inner run. Stub mode never reads the gate's pre-registration (stand-ins replace it), so the
# wrapper is harmless there and necessary for plumbing, run and group. Tested on the Orion nodes (util-linux 2.37.4,
# user namespaces enabled) on 20 October 2026: the gate's check passed inside, the stub run finished all 19 phases.
set -u
if [ "${NEAR80_NS:-}" != 1 ] && [ -n "${BCA_SHIFT_DEPS:-}" ] && [ -d "$BCA_SHIFT_DEPS" ]; then
    PINNED=/root/.cache/bca-shift/py
    if [ "$(realpath -m "$PINNED" 2>/dev/null)" != "$(realpath "$BCA_SHIFT_DEPS")" ]; then
        NEAR80_NS=1; NEAR80_NS_DEPS=$(realpath "$BCA_SHIFT_DEPS"); NEAR80_NS_UID=$(id -u); NEAR80_NS_GID=$(id -g)
        export NEAR80_NS NEAR80_NS_DEPS NEAR80_NS_UID NEAR80_NS_GID
        exec unshare --map-root-user --mount -- sh -c '
            mount -t tmpfs -o size=64m,mode=0755 none /root && mkdir -p /root/.cache/bca-shift/py \
              && mount --bind "$NEAR80_NS_DEPS" /root/.cache/bca-shift/py \
              || { echo "refused: could not provide the pinned dependency path /root/.cache/bca-shift/py"; exit 97; }
            BCA_SHIFT_DEPS=/root/.cache/bca-shift/py; export BCA_SHIFT_DEPS
            exec unshare --map-user="$NEAR80_NS_UID" --map-group="$NEAR80_NS_GID" -- sh "$0" "$@"' "$0" "$@"
    fi
fi
APPROVER_EMAIL=35092758+dbayha7@users.noreply.github.com
MODE=${1:-}
LANES=${2:-10}
SEED=${3:-}
FIXTURE=""
STUB_SEEDS=""
if [ "$MODE" = stub ]; then FIXTURE=${3:-}; STUB_SEEDS=${4:-}; SEED=""; fi
[ "${NEAR80_HOST:-}" = cluster ] || { echo "refused: NEAR80_HOST is not cluster"; exit 97; }
[ -n "${NEAR80_ROOT:-}" ] && [ -n "${NEAR80_PYTHON:-}" ] || { echo "refused: NEAR80_ROOT or NEAR80_PYTHON unset"; exit 97; }
case "$NEAR80_ROOT" in /scratch/dbayha/*) ;; *) echo "refused: NEAR80_ROOT must lie under /scratch/dbayha"; exit 97;; esac
[ "$(id -u)" != 0 ] || { echo "refused: never as root"; exit 97; }
# the stand-in HOME (run_near80_cluster.sh exports NEAR80_HOME; nothing is written under /users)
if [ -n "${NEAR80_HOME:-}" ]; then
    case "$NEAR80_HOME" in /scratch/dbayha/*) ;; *) echo "refused: NEAR80_HOME must lie under /scratch/dbayha"; exit 97;; esac
    mkdir -p "$NEAR80_HOME" || { echo "refused: cannot create $NEAR80_HOME"; exit 97; }
    HOME=$NEAR80_HOME; export HOME
fi
EMAIL=$(git -C "$NEAR80_ROOT" config --get user.email || true)
case "$EMAIL" in
    *"$APPROVER_EMAIL"*|*noreply@anthropic.com*) echo "refused: git's user.email here is $EMAIL"; exit 97;;
esac
N80=$NEAR80_ROOT/outputs/shift/beyond-the-data/near80
RUNNER=$N80/code/near80_runner.py
MEM_KB=""
if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then MEM_KB=$((SLURM_MEM_PER_NODE * 1024)); fi
if [ -z "$MEM_KB" ] && [ -n "${SLURM_MEM_PER_CPU:-}" ] && [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    MEM_KB=$((SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK * 1024))
fi
export PYTHONPATH=$NEAR80_ROOT PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/nonexistent/near80_no_pycache
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 NEAR80_HOST=cluster NEAR80_ROOT NEAR80_PYTHON
[ -n "$MEM_KB" ] && export NEAR80_MEM_TOTAL_KB=$MEM_KB
echo "near-80 v9 job: $MODE lanes $LANES seed ${SEED:-} fixture ${FIXTURE:-} on $(hostname), $(date -u +%FT%TZ), memory ${MEM_KB:-?} kB, uid $(id -u), namespace ${NEAR80_NS:-0}, deps ${BCA_SHIFT_DEPS:-unset}, HOME $HOME"
trap ':' TERM INT
case "$MODE" in
    plumbing)
        "$NEAR80_PYTHON" "$RUNNER" selftest --for plumbing --lanes "$LANES" || exit 97
        "$NEAR80_PYTHON" "$RUNNER" plumbing --lanes "$LANES"; CODE_=$?;;
    run)
        [ -n "$SEED" ] || { echo "refused: run needs a seed"; exit 97; }
        "$NEAR80_PYTHON" "$RUNNER" selftest --for run --lanes "$LANES" || exit 97
        "$NEAR80_PYTHON" "$RUNNER" run --lanes "$LANES" --seeds "$SEED" --tag "s$SEED"; CODE_=$?;;
    group)
        "$NEAR80_PYTHON" "$RUNNER" selftest --for run --lanes "$LANES" || exit 97
        "$NEAR80_PYTHON" "$RUNNER" run --lanes "$LANES" --group-only --tag group; CODE_=$?;;
    stub)
        [ -n "$FIXTURE" ] || { echo "refused: stub needs a fixture folder (outside the clone)"; exit 97; }
        if [ -n "$STUB_SEEDS" ]; then
            "$NEAR80_PYTHON" "$RUNNER" stub --fixture "$FIXTURE" --lanes "$LANES" --seeds "$STUB_SEEDS"; CODE_=$?
        else
            "$NEAR80_PYTHON" "$RUNNER" stub --fixture "$FIXTURE" --lanes "$LANES"; CODE_=$?
        fi;;
    *) echo "usage: near80_slurm_job.sh plumbing|run|group|stub LANES [SEED | FIXTURE [SEEDS]]"; exit 1;;
esac
echo "near-80 v9 job ends: exit $CODE_, $(date -u +%FT%TZ)"
exit "$CODE_"
