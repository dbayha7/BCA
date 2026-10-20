#!/bin/sh
# Near-80 (version 9, main setting): the cluster entry for hpc.charlotte.edu (Slurm). DRAFT, not frozen. POSIX sh.
# Run it on a login node, never as root, from a clone under /scratch/dbayha:
#   export NEAR80_HOST=cluster NEAR80_ROOT=/scratch/dbayha/bca NEAR80_PYTHON=/scratch/dbayha/bcavenv/bin/python
#   sh $NEAR80_ROOT/outputs/shift/beyond-the-data/near80/run_near80_cluster.sh selftest
#       the short self-test on the login node (READY or NOT READY); runs nothing
#   sh .../run_near80_cluster.sh plumbing [--lanes N]    one job: the unit tests, the smoke, the fresh-seed path and
#       the reproduction check (scratch, under $HOME/near80_v9_plumbing/)
#   sh .../run_near80_cluster.sh run [--lanes N]         one job per fresh seed (its 17 phases, both worlds; each
#       job's runner starts only its seed's phases), then one job for the group's lock and look, which waits for the
#       seed jobs (sbatch --dependency). Every job runs the same frozen code with the same approval import and
#       manifest checks as the desktop.
#   sh .../run_near80_cluster.sh status                  the runners' status files (reads only)
#   sh .../run_near80_cluster.sh plan | report | results | retry PHASE_ID | budget-continue STAGE
#       as on the desktop, on the login node (retry and budget-continue are typed; the next 'run' picks them up)
# Options: --account A (default staruser), --partition P (default Orion), --lanes N (processors per job, default 10),
#          --mem G (GB per job, default 24), --time T (default 30-00:00:00, the 30-day limit), --seeds a,b (only
#          these seeds' jobs).
#
# WHAT run_near80.sh ASSUMED THAT THIS ENTRY CHANGES (the desktop entry keeps working as it is):
#   1. root with HOME=/root inside WSL         -> never root; the job's own HOME; the clone's path from NEAR80_ROOT
#      (an absolute path under /scratch/dbayha, never /home/user/bca); the phases' environment records that HOME.
#   2. the venv at /root/bcavenv linked to /tmp/claude-0/bcavenv, numpy 2.2.6, JAX on the processor
#                                             -> NEAR80_PYTHON: a venv built from the package's pins
#      (outputs/shift/beyond-the-data/stage2_gpu/requirements-lock.txt on module python/3.12.3 or a 3.10), with the
#      shift dependencies in their own folder (BCA_SHIFT_DEPS; experiments/shift/install_deps.sh); the self-test still
#      checks numpy 2.2.6 and JAX on the processor.
#   3. version 7's placement of the version 6 code under /tmp (selftest_v7.py place)
#                                             -> not done; the pinned libraries' hard-coded /home/user/bca paths are
#      rebased onto NEAR80_ROOT in the process (policy_phase.rebase_pinned_libraries) after their bytes are verified.
#   4. tmux and one runner per machine          -> sbatch jobs; one runner per job, several jobs sharing one output
#      root (runs/shift_explore/near80_v9/ under the clone, on /scratch); runner locks and status files carry the
#      job's tag; every phase holds its own lock.
#   5. the memory rule reading /proc/meminfo    -> the job's Slurm memory (NEAR80_MEM_TOTAL_KB from SLURM_MEM_PER_NODE)
#      stands for MemTotal; free memory is that minus the running phases' reservations.
#   6. processors from the machine's topology   -> within the job's allocation (sched_getaffinity), one phase per
#      processor with taskset, as before.
#   7. the gate's data records and step 2's saved files under runs/ (restored by version 7's restore on the desktop)
#                                             -> must be present under NEAR80_ROOT/runs/ too (the run-files branches
#      and the stage2_gpu payload hold them); the self-test checks the files the plumbing reads.
#   8. the results and run files pushed from the desktop as the owner
#                                             -> the same scripts on the login node with the owner's GitHub sign-in
#      (results: near80/results-<utc>/ on the main branch; run files: the branch claude/near80-v9-run-files).
# The identity rule is the desktop's: git's user.email in the clone must be neither the approver's nor the cloud's.
set -u
ACCOUNT=staruser
PARTITION=Orion
LANES=10
MEM_GB=24
TIME=30-00:00:00
SEEDS=""
APPROVER_EMAIL=35092758+dbayha7@users.noreply.github.com
HERE=$(cd "$(dirname "$0")" && pwd -P)

refuse() { echo "refused: $*"; exit 97; }

require_env() {
    [ "${NEAR80_HOST:-}" = cluster ] || refuse "set NEAR80_HOST=cluster (this is the cluster entry)"
    [ -n "${NEAR80_ROOT:-}" ] || refuse "set NEAR80_ROOT to the clone under /scratch/dbayha"
    [ -n "${NEAR80_PYTHON:-}" ] || refuse "set NEAR80_PYTHON to the venv's python"
    case "$NEAR80_ROOT" in
        /scratch/dbayha/*) ;;
        *) refuse "NEAR80_ROOT must lie under /scratch/dbayha (got $NEAR80_ROOT); never /home/user/bca";;
    esac
    if [ "$(id -u)" = 0 ]; then refuse "never run the cluster entry as root"; fi
    N80=$NEAR80_ROOT/outputs/shift/beyond-the-data/near80
    if [ "$HERE" != "$N80" ]; then
        refuse "this script must sit at $N80 (it is at $HERE): NEAR80_ROOT must name the clone this script is in"
    fi
    git -C "$NEAR80_ROOT" rev-parse --git-dir >/dev/null 2>&1 || refuse "$NEAR80_ROOT is not a git clone"
    [ -x "$NEAR80_PYTHON" ] || refuse "$NEAR80_PYTHON is not executable"
    EMAIL=$(git -C "$NEAR80_ROOT" config --get user.email || true)
    case "$EMAIL" in
        *"$APPROVER_EMAIL"*|*noreply@anthropic.com*)
            echo "refused: git's user.email here is $EMAIL. Remove it (git config --global --unset user.email); the"
            echo "results and run files are committed with git -c user.name=... -c user.email=... by their scripts."
            exit 97;;
    esac
    command -v sbatch >/dev/null 2>&1 || SBATCH_MISSING=1
}

RUN_ENV="env PYTHONPATH=$NEAR80_ROOT PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/nonexistent/near80_no_pycache JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 NEAR80_HOST=cluster NEAR80_ROOT=$NEAR80_ROOT NEAR80_PYTHON=$NEAR80_PYTHON"
CMD=${1:-}
[ $# -gt 0 ] && shift
ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --account) ACCOUNT=${2:-}; shift 2;;
        --partition) PARTITION=${2:-}; shift 2;;
        --lanes) LANES=${2:-}; shift 2;;
        --mem) MEM_GB=${2:-}; shift 2;;
        --time) TIME=${2:-}; shift 2;;
        --seeds) SEEDS=${2:-}; shift 2;;
        *) ARGS="$ARGS $1"; shift;;
    esac
done
case "$LANES" in ''|*[!0-9]*) refuse "--lanes takes a number from 1 to 16";; esac
if [ "$LANES" -lt 1 ] || [ "$LANES" -gt 16 ]; then refuse "--lanes takes a number from 1 to 16"; fi
case "$CMD" in
    selftest|plumbing|run|status|plan|report|results|retry|budget-continue) ;;
    *) echo "usage: sh .../run_near80_cluster.sh selftest|plumbing|run|status|plan|report|results|retry PHASE_ID|budget-continue STAGE"
       exit 1;;
esac
require_env
N80=$NEAR80_ROOT/outputs/shift/beyond-the-data/near80
RUNNER=$N80/code/near80_runner.py
JOB=$N80/near80_slurm_job.sh
LOGS=$NEAR80_ROOT/runs/shift_explore/near80_v9/slurm
case "$CMD" in
    status|plan|report)
        exec $RUN_ENV "$NEAR80_PYTHON" "$RUNNER" "$CMD";;
    results|retry|budget-continue)
        # shellcheck disable=SC2086
        exec $RUN_ENV "$NEAR80_PYTHON" "$RUNNER" "$CMD" $ARGS;;
    selftest)
        exec $RUN_ENV "$NEAR80_PYTHON" "$RUNNER" selftest --for run --lanes "$LANES";;
esac
# plumbing and run: the self-test first, then sbatch
$RUN_ENV "$NEAR80_PYTHON" "$RUNNER" selftest --for "$CMD" --lanes "$LANES" || {
    echo "refused: the short self-test is not READY (lines above); nothing was submitted"; exit 97; }
[ "${SBATCH_MISSING:-}" = 1 ] && refuse "sbatch is not on PATH (run this on a login node of hpc.charlotte.edu)"
mkdir -p "$LOGS"
SB="sbatch --account=$ACCOUNT --partition=$PARTITION --nodes=1 --ntasks=1 --cpus-per-task=$LANES --mem=${MEM_GB}G --time=$TIME --export=ALL,NEAR80_HOST=cluster,NEAR80_ROOT=$NEAR80_ROOT,NEAR80_PYTHON=$NEAR80_PYTHON,BCA_SHIFT_DEPS=${BCA_SHIFT_DEPS:-}"
if [ "$CMD" = plumbing ]; then
    # shellcheck disable=SC2086
    $SB --job-name=near80v9-plumbing --output="$LOGS/plumbing-%j.out" "$JOB" plumbing "$LANES" || exit 1
    echo "submitted the plumbing job; watch $LOGS/plumbing-<job id>.out; then: sh $N80/run_near80_cluster.sh run"
    exit 0
fi
if [ -z "$SEEDS" ]; then
    SEEDS=$($RUN_ENV "$NEAR80_PYTHON" -c "import sys; sys.path.insert(0, '$N80/code'); import near80_constants as K; print(','.join(str(s) for s in K.FRESH))") || exit 1
fi
DEPS=""
OLDIFS=$IFS; IFS=,
for s in $SEEDS; do
    IFS=$OLDIFS
    # shellcheck disable=SC2086
    JID=$($SB --parsable --job-name="near80v9-s$s" --output="$LOGS/seed$s-%j.out" "$JOB" run "$LANES" "$s") || exit 1
    echo "submitted seed $s as job $JID"
    DEPS="$DEPS${DEPS:+:}$JID"
    IFS=,
done
IFS=$OLDIFS
# shellcheck disable=SC2086
JID=$($SB --parsable --job-name=near80v9-look --dependency="afterany:$DEPS" --output="$LOGS/look-%j.out" "$JOB" group "$LANES") || exit 1
echo "submitted the lock-and-look job $JID (after jobs $DEPS). Watch $LOGS/; then on the login node:"
echo "  sh $N80/run_near80_cluster.sh status"
echo "  sh $N80/run_near80_cluster.sh results      (commit and push with the commands it prints)"
echo "  sh $N80/push_runfiles_near80.sh"
exit 0
