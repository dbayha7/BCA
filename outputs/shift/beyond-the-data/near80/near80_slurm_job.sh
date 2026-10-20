#!/bin/sh
# Near-80 (version 9, main setting): the Slurm job script run_near80_cluster.sh submits. DRAFT, not frozen. POSIX sh.
#   near80_slurm_job.sh plumbing LANES          the plumbing (one job)
#   near80_slurm_job.sh run LANES SEED          one fresh seed's phases (both worlds)
#   near80_slurm_job.sh group LANES             the group's lock and look
# It never runs as root, refuses the approver's and the cloud's git identities, passes the job's memory to the
# runner (NEAR80_MEM_TOTAL_KB from SLURM_MEM_PER_NODE) and runs the same frozen runner with the same checks as the
# desktop entry. One Slurm SIGTERM (a cancel) is a safe stop: the running phases finish, nothing new starts.
set -u
APPROVER_EMAIL=35092758+dbayha7@users.noreply.github.com
MODE=${1:-}
LANES=${2:-10}
SEED=${3:-}
[ "${NEAR80_HOST:-}" = cluster ] || { echo "refused: NEAR80_HOST is not cluster"; exit 97; }
[ -n "${NEAR80_ROOT:-}" ] && [ -n "${NEAR80_PYTHON:-}" ] || { echo "refused: NEAR80_ROOT or NEAR80_PYTHON unset"; exit 97; }
case "$NEAR80_ROOT" in /scratch/dbayha/*) ;; *) echo "refused: NEAR80_ROOT must lie under /scratch/dbayha"; exit 97;; esac
[ "$(id -u)" != 0 ] || { echo "refused: never as root"; exit 97; }
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
echo "near-80 v9 job: $MODE lanes $LANES seed ${SEED:-} on $(hostname), $(date -u +%FT%TZ), memory ${MEM_KB:-?} kB"
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
    *) echo "usage: near80_slurm_job.sh plumbing|run|group LANES [SEED]"; exit 1;;
esac
echo "near-80 v9 job ends: exit $CODE_, $(date -u +%FT%TZ)"
exit "$CODE_"
