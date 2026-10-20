"""Near-80 (version 9, main setting): every fixed constant shared by the approval chain, the runner and the looks.
DRAFT, not frozen, awaiting the owner. Standard library only, so that the approval code can import it without the
gate or JAX.

Version 9 supersedes version 8 in this folder: the main test's setting only (the densest-15% removal ball; no near-80
band setting, no random-rows setting), on fresh seeds no study of this generator has used, with BCA's two groups
pooled over every held-out recorded row outside the removal ball (the refusal fix), an outcome-free count of the
full-throttle group before any training, the refusal rule (BCA refusing on a seed fails BCA's own arms on that
seed, never the seed: in C1 they count as the no-caution actor; C2 and C3 leave the seed out), four new trained arms
(the mean-difference switch with the plain 90% gap, with twice it, turning on the hand-set range limit, and BCA's
switched ramp trained alone) and three claims (C1: version 8's claim 4 marks unchanged, the hand-set range limit added
to the world-blind rivals; C2: BCA's switched ramp beats each plain twin by 5 points; C3, the mirror: a twin beats it).

Changing any value here changes the code manifest and so the fingerprint the owner approves. DECISION.md (20 October
2026, after three reviews) fixed every value below; the one item still open is OWNER_ANSWER (the owner's own words
accepting the choices of section 16), and freeze.py refuses to freeze while it is None.

Host profiles (section 12 of PREREGISTRATION.md): the desktop (version 8's paths, root inside WSL) runs version 9. A
cluster profile (hpc.charlotte.edu, Slurm, no root, a clone under /scratch/dbayha) is kept for the paths' sake; its
entry scripts were drafted, never run, and are not part of version 9 (V9_BUILD_NOTES.md says where they are). The host
is chosen by NEAR80_HOST; the paths that depend on it are left out of the frozen constants block (HOST_DEPENDENT),
everything else is frozen.
"""
import importlib.util
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

# ------------------------------------------------------------------------------------------------ placeholders
# DECISION.md (20 October 2026) fixed the seed count (ten, 40-49), C2's margin (5.0 points, ceil(0.8 n) positive seeds,
# all three twins), C3 (the mirror: -5.0 points, at most 1 positive seed) and MIN_CERTIFIED (8). Nothing is open here
# but OWNER_ANSWER below; freeze.py refuses while this dict is not empty, while any mark value is None, while
# OWNER_ANSWER is None, and while the documents carry "TODO(DECISION" or "**DRAFT".
PLACEHOLDERS = {}

# ------------------------------------------------------------------------------------------------ host profiles
HOST_PROFILES = {
    "desktop": dict(root="/home/user/bca", python="/tmp/claude-0/bcavenv/bin/python", home="/root",
                    requires_root=True, entry="run_near80.sh (tmux, one runner, lanes 1-16)",
                    memory="WSL: MemTotal minus 8 GiB (version 7's rule)"),
    "cluster": dict(root="NEAR80_ROOT: a clone under /scratch/dbayha (never /home/user/bca)",
                    python="NEAR80_PYTHON: a venv built from the package's pins, or module python/3.12.3",
                    home="the job's own HOME (never root)", requires_root=False,
                    entry="not part of version 9: the sbatch entry (account staruser, partition Orion, 48-CPU nodes, "
                          "30-day limit; one job per seed or per phase group) was drafted and never run; its two "
                          "scripts are kept outside this folder (V9_BUILD_NOTES.md)",
                    memory="the job's Slurm memory (NEAR80_MEM_TOTAL_KB) minus 8 GiB"),
    "test": dict(root="NEAR80_ROOT (a clone used by the unit tests; never an official run)", python="sys.executable",
                 home="the user's HOME", requires_root=False, entry="the unit tests and stub runs only",
                 memory="the fixture's own meminfo"),
}
HOST = os.environ.get("NEAR80_HOST", "desktop")
if HOST not in HOST_PROFILES:
    raise SystemExit(f"refused: NEAR80_HOST={HOST!r} is not one of {tuple(HOST_PROFILES)}")
CLUSTER_SCRATCH = "/scratch/dbayha"
DESKTOP_ROOT = "/home/user/bca"
DESKTOP_PYTHON = "/tmp/claude-0/bcavenv/bin/python"


def _host_root():
    if HOST == "desktop":
        return Path(DESKTOP_ROOT)
    root = os.environ.get("NEAR80_ROOT")
    if not root:
        raise SystemExit(f"refused: NEAR80_HOST={HOST} needs NEAR80_ROOT (the repository clone)")
    p = Path(root)
    if HOST == "cluster":
        if os.path.realpath(str(p)) == os.path.realpath(DESKTOP_ROOT) or not p.is_absolute():
            raise SystemExit(f"refused: on the cluster NEAR80_ROOT must be an absolute path other than {DESKTOP_ROOT}")
    return p


def _host_python():
    if HOST == "desktop":
        return DESKTOP_PYTHON
    if HOST == "cluster":
        py = os.environ.get("NEAR80_PYTHON")
        if not py:
            raise SystemExit("refused: NEAR80_HOST=cluster needs NEAR80_PYTHON (the venv's python)")
        return py
    return sys.executable


def _host_home():
    if HOST == "desktop":
        return "/root"
    return os.environ.get("HOME") or os.environ.get("USERPROFILE") or str(Path.home())


# ------------------------------------------------------------------------------------------------ paths
ROOT = _host_root()
N80_REL = "outputs/shift/beyond-the-data/near80"
N80 = ROOT / N80_REL
CODE = N80 / "code"
OUT = ROOT / "runs" / "shift_explore" / "near80_v9"          # official outputs (write-once); version 8's stay apart
PRE_OUT = OUT / "pre"                                       # the frozen code's outcome-free counts before approval
BTD = ROOT / "outputs" / "shift" / "beyond-the-data"
S1_DIR = BTD / "restructure" / "step1"
S2_DIR = BTD / "restructure" / "step2"
S1_RESULTS = S1_DIR / "results" / "results.json"
S2_RESULTS = BTD / "restructure" / "step2_results" / "results.json"
S2_OUT = ROOT / "runs" / "shift_explore" / "restructure" / "step2"     # step 2's saved actors, read-only
V6_CODE = BTD / "stage2" / "code"
V7_DIR = BTD / "stage2_v7"
V7_PLACE = V7_DIR / "laptop" / "lib" / "selftest_v7.py"              # version 7's placement of the version 6 code
PYTHON = _host_python()
HOME = _host_home()
V8_OUT = ROOT / "runs" / "shift_explore" / "near80"                   # version 8's outputs (never written)

# ------------------------------------------------------------------------------------------------ identity
LABEL = ("Version 9, near-80 main: the main test's setting on fresh seeds, with BCA's two groups pooled over every "
         "held-out row outside the removal ball")
VERSION = "version 9, near-80 main"
PREVIOUS_VERSION = "version 8, near-80"
BRANCH = "claude/main-bca-connection-x4o46s"
APPROVER_EMAIL = "35092758+dbayha7@users.noreply.github.com"
APPROVER_NAME = "David Bayha"
AGENT_EMAILS = ("noreply@anthropic.com",)
# The owner's own words accepting the choices of PREREGISTRATION.md section 16 (DECISION.md section 16, defaults
# 1-11), with the time and the place they were said: dict(text=..., utc=..., where=..., choices=[...]). The one item
# still open: no agent fills it; the freeze refuses while it is None.
OWNER_ANSWER = None

# ------------------------------------------------------------------------------------------------ the approval files
PREREG_KIND = "near80-preregistration"
MANIFEST_KIND = "near80-code-manifest"
LOCAL_KIND = "near80-local-approval"
IMPORT_KIND = "near80-import-record"
LOCAL_MODE = "local-terminal"
STATUS_FINAL = "final, awaiting approval"
STATUS_LINE_FINAL = "FINAL, awaiting the owner's approval. Nothing has been trained."
STATUS_DRAFT = "draft"
STATUS_LINE_DRAFT = "DRAFT, not frozen, awaiting the owner. Nothing has been trained."
PREFIX_LEN = 16
PREREG_NAME = "prereg.json"
MANIFEST_NAME = "code_manifest.json"
APPROVAL_NAME = "approval_near80_{prefix}.json"
IMPORT_NAME = "import_near80_{prefix}.json"
HISTORY_DIR = "prereg_history"
DRAFT_NAME = "prereg_draft.json"
DRAFT_HISTORY_NAME = "prereg_draft_{prefix}.json"
SUPERSEDED_NAME = "prereg_superseded_{prefix}.json"
MANIFEST_HISTORY_DIR = "code_manifest_history"
PRE_APPROVAL_DIR = "pre_approval"
COUNTS_FROZEN_NAME = "pooled_facts_seeds0_4_spent10_14.json"   # the pre-approval facts (nothing trained)
TEST_MARKER = "NEAR80_TEST_FIXTURE"
TEST_MARKER_TEXT = "near-80 test fixture: synthetic files only\n"

# Version 8's files that stay in this folder as history (its approval chain, its amendment, its frozen checks and its
# results). The approval script and the freeze allow exactly these beside version 9's own files; nothing reads them.
V8_PREFIX, V8_AMENDMENT_PREFIX = "51e677ab76c8d2f7", "52fe05b92900290b"
V8_HISTORY_FILES = ("approval_near80_51e677ab76c8d2f7.json", "import_near80_51e677ab76c8d2f7.json", "amendment_1.json",
                    "approval_near80_amendment1_52fe05b92900290b.json",
                    "import_near80_amendment1_52fe05b92900290b.json")
V8_HISTORY_DIRS = ("results-20261020T115246Z",)
V8_FROZEN_NAME = "no_training_checks_seeds0_4.json"
V9_NOTES = "V9_BUILD_NOTES.md"

# Files the code manifest lists, relative to N80 (code/ is listed by name; freeze refuses any other file there).
TOP_FILES = ("PREREGISTRATION.md", "README.md", "approve_local_near80.py", "run_near80.sh", "push_runfiles_near80.sh")
CODE_FILES = ("near80_constants.py", "near80lib.py", "facts_near80.py", "test_near80.py", "approval_check.py",
              "import_approval.py", "near80_prereg.py", "freeze.py", "test_approval.py", "near80_runner.py",
              "phases.py", "phase_main.py", "settingdata.py", "learner_phases.py", "policy_phase.py",
              "evaluate_phase.py", "ledger.py", "stubs.py", "test_runner.py", "pooled.py", "claims_text.py",
              "test_pooled.py", "test_v9.py")
TEST_FILES = ("test_near80.py", "test_approval.py", "test_runner.py", "test_pooled.py", "test_v9.py")
# Repository code the near-80 code imports or executes, pinned file by file in the manifest (repository_files).
PIN_DIRS = ("experiments/shift", "calibration", "algorithms", "runtime",
            "outputs/shift/beyond-the-data/restructure/step1", "outputs/shift/beyond-the-data/restructure/step2",
            "outputs/shift/beyond-the-data/stage2/code", "runs/shift_explore/room/code",
            "outputs/shift/beyond-the-data/stage2_v7/code", "outputs/shift/beyond-the-data/stage2_v7/laptop/lib")
PIN_SUFFIXES = (".py", ".sh", ".json", ".md")
PIN_FILES = ("outputs/shift/beyond-the-data/restructure/step1/results/results.json",
             "outputs/shift/beyond-the-data/restructure/step2_results/results.json",
             "outputs/shift/beyond-the-data/stage2/prereg.json")

# ------------------------------------------------------------------------------------------------ seeds and worlds
# The seed registry of this generator (DATA_SEED 20261012; every seed any study has used or named). Version 9's fresh
# seeds come from the block 40-59, which no registry or record names (checked: this repository, the near-80 branches,
# and outputs/shift/beyond-the-data/placement_test/seed_registry.json on claude/placement-test-design).
SEED_REGISTRY = {
    "0-4": "seen by everything (the gate, version 6 stage 2, steps 1 and 2, the routes, near-80 version 8)",
    "5-9": "version 6/7's confirmatory seeds",
    "10-14": "near-80 version 8's confirmatory seeds (run 19 October 2026; claim 4 failed on 10, 12 and 13)",
    "15-19": "named as the fallback block for step 2's confirmation (version 8, choice 6)",
    "20-24": "the placement test's optional margin block (seed_registry.json, claude/placement-test-design)",
    "25-29": "the placement test's planned confirmatory seeds (seed_registry.json)",
    "30-34": "the placement test's extension block (seed_registry.json)",
    "100-119": "the gate's confirmation seeds (logs generated, never trained on)",
    "900-909": "stage 0's design seeds",
}
USED_SEEDS = frozenset(list(range(0, 35)) + list(range(100, 120)) + list(range(900, 910)))
SMOKE_SEEDS = (0,)                         # the plumbing's seed (a gate seed): scratch and stub runs only
SPENT_SEEDS = (10, 11, 12, 13, 14)         # version 8's fresh seeds: read by the pre-approval facts only (section 8.1)
FRESH_SEED_BLOCK = (40, 59)                # the block version 9 draws its fresh seeds from, in order
# DECISION.md 2.4: ten fresh seeds, 40-49; seeds 50-59 are reserved for a version 10 and refused by this one.
FRESH = tuple(range(40, 50))
RESERVED_SEEDS = tuple(range(50, 60))
GROUPS = ("smoke", "fresh")
WORLDS = ("stall", "no_stall")
WORLD_TAG = {"stall": 1, "no_stall": 2}    # step 1's tags; never end a tag list in 0
SITUATIONS = ("recorded", "strong_current", "big_waves", "big_kicks")   # thruster_chain.SITUATIONS order
SCORED = "strong_current"
KEEPS_SITUATIONS = ("recorded", "big_waves", "big_kicks")               # section 9.2's order
SITUATION_PLAIN = {"recorded": "recorded conditions", "strong_current": "strong current",
                   "big_waves": "big waves", "big_kicks": "big kicks"}
WORLD_PLAIN = {"stall": "stalling world", "no_stall": "no-stall world"}

# ------------------------------------------------------------------------------------------------ settings and data
SETTINGS = ("main",)
SETTING_REMOVAL = {"main": "gap"}
HOLD = 0.70                                # held to 70% at test (the unseen-throttle readout, carried over)
CAP = 0.80                                 # capped at 80% at test (the overshoot readout)
KNEE32 = 0.800000011920929                 # float(numpy.float32(0.8)) == thruster_chain.KNEE
VARIANT = "base"                           # the gate's variant; fresh seeds use its recorder, sizes and stall slope

# ------------------------------------------------------------------------------------------------ learners
ALPHA = 5.0
ACTOR_STEPS, CHUNK = 20_000, 4_000
CRITIC_UPDATES = 100_000
SMOKE_STEPS, SMOKE_CHUNK, SMOKE_UPDATES, SMOKE_STARTS = 40, 20, 40, 20
CONTROLLER = "stall_aware"
GAP_RULE = dict(upper_quantile=0.99, min_gap=0.05, min_tail=9)
CERT = dict(alpha=0.1, credibility=0.95, draws=1000, statistic_quantile=0.05, margin=0.10)
CERT_TAG_HEAD = (20261015, 9700)           # certificate rng tags: [20261015, 9700, seed, WORLD_TAG[world]] (+[1] / [2])
CERT_MIN_ROWS = 29                         # reported: 0.9^29 < 0.05, the smallest group size whose draws can certify
ARM_RULE = dict(fixed_height=1.0, conformal_margin=0.10, rank_p=0.05)   # step 2's arms rule, unchanged
MEANDIFF_MARGIN = 0.10                     # the mean-difference switch: on when mean(full throttle) - mean(rest) > 0.10
W0_TOL = 1e-5
# The pooled groups (the refusal fix): every held-out recorded row outside the removal ball, that is the whole
# calibration split and the whole evaluation split minus the ball's rows, split by step 1's gap rule (t > t_lo is the
# full-throttle group), together with the training-split survivors step 1 already scored. Every switch arm reads them.
POOLED_SPLITS = ("calibration", "evaluation")

# ------------------------------------------------------------------------------------------------ arms (logical ids)
PRIMARY = "obca_k1"
W0 = "w0_check"
STEP2_RAMP_ARMS = ("obca_k1", "obca_k2", "obca_forced", "fixed_gated", "conformal_k1", "meangap_k1")
# C2's and C3's twins (DECISION.md 2.5): the mean-difference switch with the plain 90%-quantile gap, with twice that
# gap, and turning on the hand-set range limit. Each makes BCA's call by the mean difference, so BCA ahead of or
# behind one is a reading about heights.
TWIN_ARMS = ("meandiff_q90x1", "meandiff_q90x2", "meandiff_limit")
NOISE_ARM = "obca_k1_alone"                                        # BCA's switched ramp trained alone: the noise
                                                                   # floor's copy and C2's and C3's BCA side
BCA_DEPENDENT = ("obca_k1", "obca_k2", "obca_forced", "obca_k1_alone")   # refused when BCA's height is not finite
NEW_ARMS = ("meandiff_k1", "meandiff_q90x1", "meandiff_q90x2", "meandiff_limit", "obca_k1_alone")
LONE_CALLS = ("meandiff_k1", "meandiff_q90x1", "meandiff_q90x2", "obca_k1_alone")   # ramp programs of one, in order;
                                                                                    # then meandiff_limit (a hinge)
MATCHED = ("even_pull", "push_less", "random_1", "random_2")
WORLD_BLIND = ("standin", "even_pull", "random_1", "random_2", "obca_forced")   # step 2's order (ties: first)
HAND_SET_LIMIT = "range_hinge"
LEAD_RIVALS = WORLD_BLIND + (HAND_SET_LIMIT,)      # C1's net-lead rivals (DECISION.md 2.5; ties go to the first)
REPORTED_ONLY = ("obca_k2", "fixed_gated", "meangap_k1", "conformal_k1", "meandiff_k1", "obca_k1_alone")
ARMS = {
    "main": ("none", "clone", "full_none", W0) + STEP2_RAMP_ARMS + NEW_ARMS + ("perfect_band", "range_hinge")
            + MATCHED + ("standin",),
}
STALL_ONLY_ARMS = ("full_none",)
ARM_PLAIN = {"obca_k1": "BCA's switched ramp", "obca_forced": "the always-on ramp", "standin": "the matched range limit",
             "even_pull": "the matched even pull", "random_1": "the first matched random surface",
             "random_2": "the second matched random surface", "range_hinge": "the hand-set range limit",
             "none": "the controller trained without caution", "full_none": "the full-log controller",
             "push_less": "the matched push-less penalty", "perfect_band": "the perfect band",
             "clone": "the cloned actor", "w0_check": "the plumbing check",
             "obca_k2": "the ramp at twice BCA's height", "fixed_gated": "BCA's switch with height 1.0",
             "conformal_k1": "the split-conformal switch", "meangap_k1": "the rank-test switch",
             "meandiff_k1": "the mean-difference switch with the mean gap as its height",
             "meandiff_q90x1": "the mean-difference switch with the plain 90% gap",
             "meandiff_q90x2": "the mean-difference switch with twice the plain 90% gap",
             "meandiff_limit": "the mean-difference switch with the hand-set range limit",
             "obca_k1_alone": "BCA's switched ramp trained alone (the noise-floor copy)"}

# ------------------------------------------------------------------------------------------------ marks (section 9)
MARKS = dict(
    harm_min=30.0,                  # validity: full-log actor minus no caution, stalling world, strong current
    past80_min=0.25,                # validity: no caution's share of strong-current steps past 80%
    noise_floor_max=5.0,            # |obca_k1 - obca_k1_alone| pooled, stalling world, strong current (drops C2)
    net_lead_margin=10.0,           # C1 (step 2's three go marks, version 8's claim 4)
    keeps_floor=-3.0,
    even_pull_margin=15.0,
)
# C2 (DECISION.md 2.6): per twin, d = NET(obca_k1_alone) - NET(twin) over the certified seeds; "BCA ahead" when the
# mean of d is at least net_margin and d is positive in at least ceil(positive_share * n) seeds (8 of 10, 8 of 9, 7 of
# 8; positive_seeds_min is the ten-seed value); C2 holds when BCA is ahead of all three twins.
MARKS_C2 = dict(net_margin=5.0, positive_seeds_min=8, positive_share=0.8, multiplicity="all_three")
# C3 (the mirror, one claim per twin): "the twin ahead" when the mean of d is at most net_margin and d is positive in
# at most positive_seeds_max seeds. Otherwise the reading is a tie ("not shown to beat T by 5 points").
MARKS_C3 = dict(net_margin=-5.0, positive_seeds_max=1)
MIN_CERTIFIED = 8                          # C2 and C3 are read only when at least this many seeds certified
MULTIPLICITY_RULES = ("all_three", "each_separately")
DEFAULTS = dict(setting="main", seeds="ten fresh seeds, 40-49", pooled_groups=True,
                refusal_rule="per arm: no caution in C1; the seed left out of C2 and C3",
                noise_floor="obca_k1 trained alone", hosts="TD3+BC actor-only", lanes=10, name="version 9, near-80 main",
                machine="the desktop")

# ------------------------------------------------------------------------------------------------ the runner
PHASE_KINDS = ("data_main", "critic", "critic_full", "critic_second", "switch", "clone", "reference_actors", "policy",
               "lock", "evaluate", "look")
PHASE_PLAIN = {"data_main": "the main setting's data, the gate's checks and the pooled group counts",
               "critic": "the critic", "critic_full": "the full-log critic", "critic_second": "BCA's second critic",
               "switch": "BCA's switch and height on the pooled groups", "clone": "the cloned actor",
               "reference_actors": "the no-caution and full-log actors", "policy": "the actor training",
               "lock": "the lock", "evaluate": "the evaluation", "look": "the look"}
DEFAULT_LANES, MAX_LANES = 10, 16
GIB = 1024 ** 3
UNMEASURED_GIB, HEADROOM_GIB, MEASURED_FACTOR = 7, 8, 1.25
POLICY_KINDS = ("policy",)
POLICY_CAP_UNMEASURED = 2
EXIT_REFUSED, EXIT_STOP = 97, 2
EXIT_PAUSED = 5
PYCACHE_PREFIX = "/nonexistent/near80_no_pycache"
PHASE_ENV = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "HOME": HOME,
             "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONPYCACHEPREFIX": PYCACHE_PREFIX, "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1"}
if HOST == "cluster":
    PHASE_ENV["PATH"] = os.environ.get("PATH", PHASE_ENV["PATH"])        # the module's python and its tools
# Version 8's ledger (results-20261020T115246Z): 1.52 CPU-hours per main-setting seed (seeds 11 and 14: 1.53 and
# 1.51, both worlds, every phase); the five programs run alone (four lone ramp calls of about 63 s in the stalling
# world and 40 s in the no-stall world, and the hinge) add about 0.14; the pooled scores seconds. About 1.65 expected;
# 2.2 is the forecast with margin (the guard pauses at twice it; the cap is 2.5 times it: 55 CPU-hours for ten seeds).
FORECAST_CPU_HOURS_PER_SEED = 2.2
FORECAST_CPU_HOURS = {"plumbing": 2.5, "fresh": FORECAST_CPU_HOURS_PER_SEED * len(FRESH)}
BUDGET_STAGES = ("fresh",)
PAUSE_FACTOR = 2.0
CAP_CPU_HOURS = round(2.5 * FORECAST_CPU_HOURS_PER_SEED * len(FRESH), 1)
REPRO_PEAK_KINDS = ("critic", "critic_full", "critic_second", "switch", "clone", "reference_actors", "policy")
SMOKE_FLOOR_GIB = 2.0
RUNFILES_BRANCH = "claude/near80-v9-run-files"

# Output layout under an output root (OUT, or a scratch or fixture folder). Every JSON has a '<name>.sha256' sidecar.
LAYOUT = dict(
    data_main="data/main/{world}/seed{seed}/data.json",
    train_file="data/{setting}/{world}/seed{seed}/train.hdf5",
    critic="learners/{setting}/{world}/seed{seed}/critic.json",
    critic_full="learners/full/stall/seed{seed}/critic.json",
    critic_second="learners/{setting}/{world}/seed{seed}/critic_second.json",
    switch="learners/{setting}/{world}/seed{seed}/switch.json",
    clone="learners/{setting}/{world}/seed{seed}/clone.json",
    reference_actors="learners/{setting}/{world}/seed{seed}/reference_actors.json",
    policy_lock="policy/{setting}/{world}/seed{seed}/lock.json",
    policy_stop="policy/{setting}/{world}/seed{seed}/stop.json",
    stage_lock="stage_lock/{group}.json",
    values="eval/{setting}/{world}/seed{seed}/values.json",
    look="looks/{group}.json",
    look_md="looks/{group}.md",
    halt_report="looks/halt_report.json",
    phase_stop="stops/{phase_id}.json",
    plumbing="plumbing/plumbing.json",
    reproduction_differs="plumbing/reproduction_differs.json",
    plumbing_failed="plumbing/failed_{utc}.json",
    budget_continue="budget/{stage}/{n}.json",
    ledger_dir="ledger/{phase_id}",
    phase_lock="locks/{phase_id}.lock",
    log="logs/{phase_id}.log",
    interrupted="interrupted/{phase_id}/{utc}",
    status="status.json",
    runner_lock="runner.lock",
    runner_log="runner.log",
    retry="retry/{phase_id}/{n}.json",
)

# Names whose values depend on the host (NEAR80_HOST): left out of the frozen constants block.
HOST_DEPENDENT = ("HOST", "ROOT", "N80", "CODE", "OUT", "PRE_OUT", "BTD", "S1_DIR", "S2_DIR", "S1_RESULTS", "S2_RESULTS",
                  "S2_OUT", "V6_CODE", "V7_DIR", "V7_PLACE", "PYTHON", "HOME", "V8_OUT", "PHASE_ENV")


def path(root, name, **kw):
    """The path of layout entry `name` under the output root `root`."""
    return Path(root) / LAYOUT[name].format(**kw)


def phase_id(kind, setting="x", world="x", seed=None, group=None):
    """'<kind>.<setting>.<world>.<s{seed} | group | x>', e.g. 'policy.main.stall.s40', 'lock.x.x.fresh'."""
    tail = f"s{int(seed)}" if seed is not None else (group or "x")
    return f"{kind}.{setting}.{world}.{tail}"


def seed_holder(seed):
    """Who holds a used seed (SEED_REGISTRY), or None when no study names it."""
    s = int(seed)
    for span, who in SEED_REGISTRY.items():
        a, b = (int(x) for x in span.split("-"))
        if a <= s <= b:
            return who
    return None


def group_of(seed):
    """'smoke' for the plumbing's seed 0, 'fresh' for a version 9 seed; refuses every other seed, naming its holder."""
    s = int(seed)
    if s in FRESH:
        return "fresh"
    if s in SMOKE_SEEDS:
        return "smoke"
    who = seed_holder(s)
    if who is not None:
        raise ValueError(f"refused: seed {s} is not one of version 9's seeds: it belongs to {who}")
    raise ValueError(f"refused: seed {s} is not one of version 9's seeds (the plumbing's {SMOKE_SEEDS}; the fresh "
                     f"seeds {FRESH})")


def placeholders_present():
    """The open items: PLACEHOLDERS' names, any MARKS_C2 or MARKS_C3 value still None, MIN_CERTIFIED None, and
    OWNER_ANSWER None (the owner's own words). Empty when everything is fixed (then freeze.py may freeze)."""
    out = list(PLACEHOLDERS)
    for k, v in MARKS_C2.items():
        if v is None and f"c2_{k}" not in out:
            out.append(f"c2_{k}")
    for k, v in MARKS_C3.items():
        if v is None and f"c3_{k}" not in out:
            out.append(f"c3_{k}")
    if MIN_CERTIFIED is None:
        out.append("min_certified")
    if OWNER_ANSWER is None and "owner_answer" not in out:
        out.append("owner_answer")
    if MARKS_C2.get("multiplicity") not in (None,) + MULTIPLICITY_RULES:
        out.append("c2_multiplicity (not one of the known rules)")
    if not (FRESH_SEED_BLOCK[0] <= min(FRESH) and max(FRESH) <= FRESH_SEED_BLOCK[1]):
        out.append("seeds (outside the block)")
    if set(FRESH) & set(RESERVED_SEEDS):
        out.append("seeds (a reserved seed)")
    return out


def c2_positive_needed(n_certified):
    """ceil(positive_share * n): the seeds C2's sign part requires (8 of 10, 8 of 9, 7 of 8)."""
    import math
    return int(math.ceil(float(MARKS_C2["positive_share"]) * int(n_certified) - 1e-9))


def load_local(name):
    """N80/code/<name>.py loaded by explicit path as module 'near80_<name>' (avoids step 1's and step 2's pooled.py
    and freeze.py, which step 2's machinery puts first on sys.path)."""
    key = f"near80_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, str(CODE / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def as_dict():
    """Every constant above as plain JSON-able values (copied into prereg.json's 'constants' block), without the
    host-dependent paths (HOST_DEPENDENT)."""
    out = {}
    for k, v in sorted(globals().items()):
        if k.isupper() and k not in HOST_DEPENDENT:
            if isinstance(v, Path):
                v = str(v)
            elif isinstance(v, frozenset):
                v = sorted(v)
            out[k] = v
    return out
