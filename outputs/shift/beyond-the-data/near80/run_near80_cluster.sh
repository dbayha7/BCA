#!/bin/sh
# Near-80 (version 9, main setting): the cluster entry for hpc.charlotte.edu (Slurm). DRAFT, not frozen. POSIX sh.
# Run it on a login node, never as root, from a git clone under /scratch/dbayha:
#   export NEAR80_HOST=cluster NEAR80_ROOT=/scratch/dbayha/bca-near80/repo NEAR80_PYTHON=/scratch/dbayha/bca-near80/env/bin/python
#   export BCA_SHIFT_DEPS=/scratch/dbayha/bca-near80/shiftdeps
#   sh $NEAR80_ROOT/outputs/shift/beyond-the-data/near80/run_near80_cluster.sh selftest
#       the short self-test on the login node (READY or NOT READY); runs nothing
#   sh .../run_near80_cluster.sh plumbing                one job (4 processors, 16 GB, 2 h): the unit tests, the
#       smoke, the fresh-seed path and the reproduction check (scratch, under $HOME/near80_v9_plumbing/)
#   sh .../run_near80_cluster.sh run                     one job per fresh seed (8 processors, 24 GB, 3 h: its
#       phases of both worlds up to the actor training; each job's runner starts only its seed's phases and ends
#       once only lock-dependent phases remain), then one group job (16 processors, 24 GB, 1 h) for the lock,
#       every evaluation and the look, which waits for the seed jobs (sbatch --dependency=afterany). Every job runs
#       the same frozen code with the same approval import and manifest checks as the desktop.
#   sh .../run_near80_cluster.sh status                  the runners' status files (reads only)
#   sh .../run_near80_cluster.sh plan | report | results | retry PHASE_ID | budget-continue STAGE
#       as on the desktop, on the login node (retry and budget-continue are typed; the next 'run' picks them up)
#   sh .../run_near80_cluster.sh stub --fixture DIR [--seeds a,b] [--lanes N] [--job-name NAME] [--time T]
#       one job in stub mode on a test fixture outside the clone (stand-ins, nothing trained, no approval needed):
#       the plumbing-style test of this entry on the cluster. Default --time 01:00:00. A test uses --seeds 0 and a
#       job name near80-smoke-*, never a fresh seed.
# Options: --account A (default staruser), --partition P (default Orion), --constraint C (default caslake: one
#          processor type for the whole run; '' for none), --lanes N (processors per job; default per command as
#          above), --mem G (GB per job; default per command), --time T (default per command), --seeds a,b (only
#          these seeds' jobs), --fixture DIR and --job-name NAME (the stub command's fixture; the job names' prefix,
#          default near80v9).
# HOME: every command and every job runs with HOME=$NEAR80_HOME (default: the folder 'home' beside the clone, under
# /scratch/dbayha), so that nothing is written under /users and the phases record that HOME; git's identity is read
# under it (the rule is the desktop's: user.email must be neither the approver's nor the cloud's).
# The pinned dependency path: the gate's pre-registration fixes the shift dependencies at /root/.cache/bca-shift/py;
# near80_slurm_job.sh provides it inside an unprivileged user namespace (its header says how). The login node needs
# only BCA_SHIFT_DEPS for the self-test.
# sbatch: the owner's wrapper /users/dbayha/bin/sbatch (first on PATH; it adds --exclude=str-gpu11) is used as found.
#
# WHAT run_near80.sh ASSUMED THAT THIS ENTRY CHANGES (the desktop entry keeps working as it is, as the fallback):
#   1. root with HOME=/root inside WSL         -> never root; a stand-in HOME under /scratch; the clone's path from
#      NEAR80_ROOT (an absolute path under /scratch/dbayha, never /home/user/bca).
#   2. the venv at /root/bcavenv linked to /tmp/claude-0/bcavenv, numpy 2.2.6, JAX on the processor
#                                             -> NEAR80_PYTHON: a venv built from the package's pins
#      (outputs/shift/beyond-the-data/stage2_gpu/requirements-lock.txt with uv 0.11.32 and Python 3.10.20), with the
#      shift dependencies in their own folder (BCA_SHIFT_DEPS; experiments/shift/install_deps.sh); the self-test
#      still checks numpy 2.2.6 and JAX on the processor.
#   3. version 7's placement of the version 6 code under /tmp (selftest_v7.py place)
#                                             -> not done; the pinned libraries' hard-coded /home/user/bca paths are
#      rebased onto NEAR80_ROOT in the process (policy_phase.rebase_pinned_libraries) after their bytes are verified.
#   4. tmux and one runner per machine          -> sbatch jobs; one runner per job, several jobs sharing one output
#      root (runs/shift_explore/near80_v9/ under the clone, on /scratch); runner locks and status files carry the
#      job's tag; every phase holds its own lock; the seed jobs leave the lock-dependent phases to the group job.
#   5. the memory rule reading /proc/meminfo    -> the job's Slurm memory (NEAR80_MEM_TOTAL_KB from SLURM_MEM_PER_NODE)
#      stands for MemTotal; free memory is that minus the running phases' reservations.
#   6. processors from the machine's topology   -> within the job's allocation (sched_getaffinity; Slurm hands out
#      non-contiguous ids), one phase per processor with taskset, as before.
#   7. the gate's data records and step 2's saved files under runs/ (restored by version 7's restore on the desktop)
#                                             -> must be present under NEAR80_ROOT/runs/ too: runs/shift,
#      runs/shift_explore/room, runs/shift_explore/restructure/step2 and runs/shift_explore/stage2 (version 6's run
#      folder, read by version 6's policy code), copied read-only from the desktop clone; the self-test checks them.
#   8. the results and run files pushed from the desktop as the owner
#                                             -> the same scripts on the login node in the owner's own SSH session
#      (results: near80/results-<utc>/ on the main branch; run files: the branch claude/near80-v9-run-files); or the
#      folders come back to the desktop by tar and are pushed from there.
set -u
ACCOUNT=staruser
PARTITION=Orion
CONSTRAINT=caslake
LANES=""
MEM_GB=""
TIME=""
SEEDS=""
FIXTURE=""
JOB_NAME=near80v9
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
    NEAR80_HOME=${NEAR80_HOME:-$(dirname "$NEAR80_ROOT")/home}
    case "$NEAR80_HOME" in /scratch/dbayha/*) ;; *) refuse "NEAR80_HOME must lie under /scratch/dbayha (got $NEAR80_HOME)";; esac
    mkdir -p "$NEAR80_HOME" || refuse "cannot create $NEAR80_HOME"
    HOME=$NEAR80_HOME; export HOME NEAR80_HOME
    EMAIL=$(git -C "$NEAR80_ROOT" config --get user.email || true)
    case "$EMAIL" in
        *"$APPROVER_EMAIL"*|*noreply@anthropic.com*)
            echo "refused: git's user.email here is $EMAIL. Remove it (git config --global --unset user.email); the"
            echo "results and run files are committed with git -c user.name=... -c user.email=... by their scripts."
            exit 97;;
    esac
    command -v sbatch >/dev/null 2>&1 || SBATCH_MISSING=1
}

CMD=${1:-}
[ $# -gt 0 ] && shift
ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --account) ACCOUNT=${2:-}; shift 2;;
        --partition) PARTITION=${2:-}; shift 2;;
        --constraint) CONSTRAINT=${2:-}; shift 2;;
        --lanes) LANES=${2:-}; shift 2;;
        --mem) MEM_GB=${2:-}; shift 2;;
        --time) TIME=${2:-}; shift 2;;
        --seeds) SEEDS=${2:-}; shift 2;;
        --fixture) FIXTURE=${2:-}; shift 2;;
        --job-name) JOB_NAME=${2:-}; shift 2;;
        *) ARGS="$ARGS $1"; shift;;
    esac
done
case "$CMD" in
    selftest|plumbing|run|status|plan|report|results|retry|budget-continue|stub) ;;
    *) echo "usage: sh .../run_near80_cluster.sh selftest|plumbing|run|status|plan|report|results|retry PHASE_ID|budget-continue STAGE|stub --fixture DIR"
       exit 1;;
esac
# the per-command job sizes (near80_constants.CLUSTER; the smoke of 20 October 2026): lanes, GB, time
case "$CMD" in
    plumbing) D_LANES=4; D_MEM=16; D_TIME=02:00:00;;
    run)      D_LANES=8; D_MEM=24; D_TIME=03:00:00;;
    stub)     D_LANES=4; D_MEM=16; D_TIME=01:00:00;;
    *)        D_LANES=8; D_MEM=24; D_TIME=03:00:00;;
esac
G_LANES=16; G_MEM=24; G_TIME=01:00:00
[ -n "$LANES" ] || LANES=$D_LANES
[ -n "$MEM_GB" ] || MEM_GB=$D_MEM
[ -n "$TIME" ] || TIME=$D_TIME
case "$LANES" in ''|*[!0-9]*) refuse "--lanes takes a number from 1 to 16";; esac
if [ "$LANES" -lt 1 ] || [ "$LANES" -gt 16 ]; then refuse "--lanes takes a number from 1 to 16"; fi
case "$MEM_GB" in ''|*[!0-9]*) refuse "--mem takes a whole number of GB";; esac
require_env
N80=$NEAR80_ROOT/outputs/shift/beyond-the-data/near80
RUNNER=$N80/code/near80_runner.py
JOB=$N80/near80_slurm_job.sh
LOGS=$NEAR80_ROOT/runs/shift_explore/near80_v9/slurm
RUN_ENV="env PYTHONPATH=$NEAR80_ROOT PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/nonexistent/near80_no_pycache JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 NEAR80_HOST=cluster NEAR80_ROOT=$NEAR80_ROOT NEAR80_PYTHON=$NEAR80_PYTHON HOME=$HOME"
CONSTRAINT_OPT=""
[ -n "$CONSTRAINT" ] && CONSTRAINT_OPT="--constraint=$CONSTRAINT"
EXPORTS="--export=ALL,HOME=$HOME,NEAR80_HOME=$HOME,NEAR80_HOST=cluster,NEAR80_ROOT=$NEAR80_ROOT,NEAR80_PYTHON=$NEAR80_PYTHON,BCA_SHIFT_DEPS=${BCA_SHIFT_DEPS:-},PYTHONDONTWRITEBYTECODE=1"
sb() {
    # sbatch with the common options: $1 lanes, $2 GB, $3 time, then the job's own options and the job script
    _l=$1; _m=$2; _t=$3; shift 3
    # shellcheck disable=SC2086
    sbatch --account="$ACCOUNT" --partition="$PARTITION" $CONSTRAINT_OPT --nodes=1 --ntasks=1 --cpus-per-task="$_l" \
        --mem="${_m}G" --time="$_t" "$EXPORTS" "$@"
}
case "$CMD" in
    status|plan|report)
        exec $RUN_ENV "$NEAR80_PYTHON" "$RUNNER" "$CMD";;
    results|retry|budget-continue)
        # shellcheck disable=SC2086
        exec $RUN_ENV "$NEAR80_PYTHON" "$RUNNER" "$CMD" $ARGS;;
    selftest)
        exec $RUN_ENV "$NEAR80_PYTHON" "$RUNNER" selftest --for run --lanes "$LANES";;
esac
if [ "$CMD" = stub ]; then
    # stub mode on a fixture outside the clone; no self-test gate (the stub's fixture carries its own approval and
    # the runner refuses a fixture inside the clone); one job; the log lands beside the fixture.
    [ -n "$FIXTURE" ] || refuse "stub needs --fixture DIR (a folder outside the clone)"
    FIXTURE=$(realpath -m "$FIXTURE")
    case "$FIXTURE" in "$NEAR80_ROOT"/*|"$NEAR80_ROOT") refuse "the stub fixture must lie outside the clone";; esac
    [ "${SBATCH_MISSING:-}" = 1 ] && refuse "sbatch is not on PATH (run this on a login node of hpc.charlotte.edu)"
    mkdir -p "$(dirname "$FIXTURE")"
    JID=$(sb "$LANES" "$MEM_GB" "$TIME" --parsable --job-name="$JOB_NAME-stub" --output="$FIXTURE-slurm-%j.out" \
        "$JOB" stub "$LANES" "$FIXTURE" "$SEEDS") || exit 1
    echo "submitted the stub job $JID (fixture $FIXTURE, seeds ${SEEDS:-all}); watch $FIXTURE-slurm-$JID.out"
    exit 0
fi
# plumbing and run: the self-test first, then sbatch
$RUN_ENV "$NEAR80_PYTHON" "$RUNNER" selftest --for "$CMD" --lanes "$LANES" || {
    echo "refused: the short self-test is not READY (lines above); nothing was submitted"; exit 97; }
[ "${SBATCH_MISSING:-}" = 1 ] && refuse "sbatch is not on PATH (run this on a login node of hpc.charlotte.edu)"
mkdir -p "$LOGS"
if [ "$CMD" = plumbing ]; then
    sb "$LANES" "$MEM_GB" "$TIME" --job-name="$JOB_NAME-plumbing" --output="$LOGS/plumbing-%j.out" \
        "$JOB" plumbing "$LANES" || exit 1
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
    JID=$(sb "$LANES" "$MEM_GB" "$TIME" --parsable --job-name="$JOB_NAME-s$s" --output="$LOGS/seed$s-%j.out" \
        "$JOB" run "$LANES" "$s") || exit 1
    echo "submitted seed $s as job $JID"
    DEPS="$DEPS${DEPS:+:}$JID"
    IFS=,
done
IFS=$OLDIFS
JID=$(sb "$G_LANES" "$G_MEM" "$G_TIME" --parsable --job-name="$JOB_NAME-look" --dependency="afterany:$DEPS" \
    --output="$LOGS/look-%j.out" "$JOB" group "$G_LANES") || exit 1
echo "submitted the lock, evaluation and look job $JID (after jobs $DEPS). Watch $LOGS/; then on the login node:"
echo "  sh $N80/run_near80_cluster.sh status"
echo "  sh $N80/run_near80_cluster.sh results      (then, in your own SSH session, commit and push with the commands it prints)"
echo "  sh $N80/push_runfiles_near80.sh            (in your own SSH session: it needs a terminal and your GitHub sign-in)"
exit 0
