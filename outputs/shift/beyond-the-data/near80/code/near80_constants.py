"""Near-80 (version 8): every fixed constant shared by the approval chain, the runner and the looks. FINAL, awaiting
the owner's approval. Standard library only, so that the approval code can import it without the gate or JAX.

Changing any value here changes the code manifest and so the fingerprint the owner approves."""
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True

# ------------------------------------------------------------------------------------------------ paths
ROOT = Path("/home/user/bca")
N80_REL = "outputs/shift/beyond-the-data/near80"
N80 = ROOT / N80_REL
CODE = N80 / "code"
OUT = ROOT / "runs" / "shift_explore" / "near80"            # official outputs (write-once)
PRE_OUT = OUT / "pre"                                       # the frozen-code no-training checks before approval
BTD = ROOT / "outputs" / "shift" / "beyond-the-data"
S1_DIR = BTD / "restructure" / "step1"
S2_DIR = BTD / "restructure" / "step2"
S1_RESULTS = S1_DIR / "results" / "results.json"
S2_RESULTS = BTD / "restructure" / "step2_results" / "results.json"
S2_OUT = ROOT / "runs" / "shift_explore" / "restructure" / "step2"     # step 2's saved actors (desktop), read-only
V6_CODE = BTD / "stage2" / "code"
V7_DIR = BTD / "stage2_v7"
V7_R7 = ROOT / "runs" / "shift_explore" / "stage2_v7"                 # version 7's runner folder (its locks)
V7_PLACE = V7_DIR / "laptop" / "lib" / "selftest_v7.py"              # version 7's placement of the version 6 code
PYTHON = "/tmp/claude-0/bcavenv/bin/python"
SCRATCH = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad")

# ------------------------------------------------------------------------------------------------ identity
LABEL = ("Version 8, near-80: the main test's setting with every recorded throttle from 70% to 90% also hidden")
VERSION = "version 8, near-80"
BRANCH = "claude/main-bca-connection-x4o46s"
APPROVER_EMAIL = "35092758+dbayha7@users.noreply.github.com"
APPROVER_NAME = "David Bayha"
AGENT_EMAILS = ("noreply@anthropic.com",)
OWNER_ANSWER = dict(text="accept the defaults", utc="2026-10-15T19:24:00Z",
                    where="a comment on the ThrusterChain page", choices="every default of section 16")

# ------------------------------------------------------------------------------------------------ the approval files
PREREG_KIND = "near80-preregistration"
MANIFEST_KIND = "near80-code-manifest"
LOCAL_KIND = "near80-local-approval"
IMPORT_KIND = "near80-import-record"
LOCAL_MODE = "local-terminal"
STATUS_FINAL = "final, awaiting approval"
STATUS_LINE_FINAL = "FINAL, awaiting the owner's approval. Nothing has been trained."
PREFIX_LEN = 16
PREREG_NAME = "prereg.json"
MANIFEST_NAME = "code_manifest.json"
APPROVAL_NAME = "approval_near80_{prefix}.json"
IMPORT_NAME = "import_near80_{prefix}.json"
HISTORY_DIR = "prereg_history"
DRAFT_HISTORY_NAME = "prereg_draft_{prefix}.json"
MANIFEST_HISTORY_DIR = "code_manifest_history"
PRE_APPROVAL_DIR = "pre_approval"
NO_TRAINING_FROZEN_NAME = "no_training_checks_seeds0_4.json"
TEST_MARKER = "NEAR80_TEST_FIXTURE"
TEST_MARKER_TEXT = "near-80 test fixture: synthetic files only\n"

# Files the code manifest lists, relative to N80 (code/ is listed by name; freeze refuses any other file there).
TOP_FILES = ("PREREGISTRATION.md", "README.md", "approve_local_near80.py", "run_near80.sh", "push_runfiles_near80.sh")
CODE_FILES = ("near80_constants.py", "near80lib.py", "facts_near80.py", "test_near80.py", "approval_check.py",
              "import_approval.py", "near80_prereg.py", "freeze.py", "test_approval.py", "near80_runner.py",
              "phases.py", "phase_main.py", "settingdata.py", "learner_phases.py", "policy_phase.py",
              "evaluate_phase.py", "v6bca.py", "ledger.py", "stubs.py", "test_runner.py", "pooled.py",
              "claims_text.py", "test_pooled.py")
TEST_FILES = ("test_near80.py", "test_approval.py", "test_runner.py", "test_pooled.py")
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
EXPLORATORY = (0, 1, 2, 3, 4)
CONFIRMATORY = (10, 11, 12, 13, 14)
RESERVED = (5, 6, 7, 8, 9)                 # version 6/7's; refused always, in every mode
GROUPS = ("exploratory", "confirmatory")
WORLDS = ("stall", "no_stall")
WORLD_TAG = {"stall": 1, "no_stall": 2}    # step 1's tags; never end a tag list in 0
SITUATIONS = ("recorded", "strong_current", "big_waves", "big_kicks")   # thruster_chain.SITUATIONS order
SCORED = "strong_current"
KEEPS_SITUATIONS = ("recorded", "big_waves", "big_kicks")               # section 9.2's order
SITUATION_PLAIN = {"recorded": "recorded conditions", "strong_current": "strong current",
                   "big_waves": "big waves", "big_kicks": "big kicks"}
WORLD_PLAIN = {"stall": "stalling world", "no_stall": "no-stall world"}

# ------------------------------------------------------------------------------------------------ settings and data
SETTINGS = ("main", "near80", "random_rows")
SETTING_REMOVAL = {"main": "gap", "near80": "ball_band", "random_rows": "ball_random_rows"}
BAND_LO, BAND_HI, BAND_SCOPE = 0.70, 0.90, "any"
HOLD = 0.70                                # held to 70% at test (unseen-throttle gain)
CAP = 0.80                                 # capped at 80% at test (overshoot loss); recorder.CappedController's cap
KNEE32 = 0.800000011920929                 # float(numpy.float32(0.8)) == thruster_chain.KNEE: the knee arms' start
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
ARM_RULE = dict(fixed_height=1.0, conformal_margin=0.10, rank_p=0.05)   # step 2's arms rule, unchanged
MEANDIFF_MARGIN = 0.10                     # meandiff_k1: on when mean(full-throttle) - mean(rest) > 0.10
W0_TOL = 1e-5

# ------------------------------------------------------------------------------------------------ arms (logical ids)
PRIMARY = "obca_k1"
W0 = "w0_check"
STEP2_RAMP_ARMS = ("obca_k1", "obca_k2", "obca_forced", "fixed_gated", "conformal_k1", "meangap_k1")
NEW_RAMP_ARMS = ("meandiff_k1", "knee_ramp")
MATCHED = ("even_pull", "push_less", "random_1", "random_2")
WORLD_BLIND = ("standin", "even_pull", "random_1", "random_2", "obca_forced")   # step 2's order (ties: first)
REPORTED_ONLY = ("obca_k2", "fixed_gated", "conformal_k1", "meangap_k1", "meandiff_k1")
HAND_SET_LIMIT = "range_hinge"            # the draft's "range_hinge_k1"
ARMS = {
    "near80": ("none", "clone", "full_none", W0) + STEP2_RAMP_ARMS + NEW_RAMP_ARMS
              + ("perfect_band", "range_hinge", "knee_limit") + MATCHED + ("standin", "version6_bca"),
    "random_rows": ("none", "clone", W0, "obca_k1", "range_hinge"),
    "main": ("none", "clone", "full_none", W0) + STEP2_RAMP_ARMS + ("perfect_band", "range_hinge") + MATCHED
            + ("standin",),
}
STALL_ONLY_ARMS = ("full_none",)
ARM_PLAIN = {"obca_k1": "BCA's switched ramp", "obca_forced": "the always-on ramp", "standin": "the matched range limit",
             "even_pull": "the matched even pull", "random_1": "the first matched random surface",
             "random_2": "the second matched random surface", "range_hinge": "the hand-set range limit",
             "none": "the controller trained without caution", "full_none": "the full-log controller",
             "version6_bca": "version 6's BCA", "push_less": "the matched push-less penalty",
             "perfect_band": "the perfect band", "knee_ramp": "the ramp started at the stall",
             "knee_limit": "the range limit placed at the stall", "clone": "the cloned actor",
             "w0_check": "the plumbing check", "obca_k2": "the ramp at twice BCA's height",
             "fixed_gated": "BCA's switch with height 1.0", "conformal_k1": "the split-conformal switch",
             "meangap_k1": "the rank-test switch", "meandiff_k1": "the mean-difference switch"}

# ------------------------------------------------------------------------------------------------ marks (section 9)
MARKS = dict(
    harm_min=30.0,                  # full-log actor minus no caution, near-80, stalling, strong current (validity, stop)
    past80_min=0.25,                # no caution's share of strong-current steps past 80% (validity, stop)
    rival_band_cost_min=5.0,        # hand-set range limit's band cost (stop on 0-4; claim 3 validity)
    noise_floor_max=5.0,            # |random rows - main| for range_hinge and obca_k1 on 0-4 (drops claim 3)
    net_lead_margin=10.0,           # claims 1 and 4
    keeps_floor=-3.0,               # claims 1 and 4, three other situations
    even_pull_margin=15.0,          # claims 1 and 4
    gain_min=5.0, gain_positive_seeds_min=4,              # claim 2
    overshoot_max=5.0,                                    # claim 2
    band_cost_diff_min=5.0, band_cost_positive_seeds_min=4,   # claim 3
    check10_seeds_min=4,                                  # claims 2 and 3 validity (of 5 fresh seeds)
)
DEFAULTS = dict(thrusters="any", band=(0.70, 0.90), with_ball=True, signs="both", region_floor=0.35,
                main_on_fresh_seeds=True, random_rows_control=True, version6_bca=True,
                after_poor_bca="run seeds 10-14 anyway", checkpoint=None, hosts="TD3+BC actor-only",
                when="after version 7's run on the desktop has finished", lanes=10, name="version 8, near-80",
                claim2_marks=5.0)

# ------------------------------------------------------------------------------------------------ the runner
PHASE_KINDS = ("fixed_checks", "data_main", "data_band", "critic", "critic_full", "critic_second", "switch", "clone",
               "reference_actors", "v6_band", "policy", "lock", "evaluate", "readouts", "look_exploratory",
               "look_confirmatory")
PHASE_PLAIN = {"fixed_checks": "the checks on the fixed test start sets",
               "data_main": "the main setting's data and the gate's checks",
               "data_band": "the near-80 and random-rows data and their no-training checks",
               "critic": "the critic", "critic_full": "the full-log critic", "critic_second": "BCA's second critic",
               "switch": "BCA's switch and height", "clone": "the cloned actor",
               "reference_actors": "the no-caution and full-log actors", "v6_band": "version 6's BCA band",
               "policy": "the actor training", "lock": "the lock", "evaluate": "the evaluation",
               "readouts": "the readouts", "look_exploratory": "the exploratory look",
               "look_confirmatory": "the confirmatory look"}
DEFAULT_LANES, MAX_LANES = 10, 16
GIB = 1024 ** 3
UNMEASURED_GIB, HEADROOM_GIB, MEASURED_FACTOR = 7, 8, 1.25
POLICY_KINDS = ("policy",)
POLICY_CAP_UNMEASURED = 2
EXIT_REFUSED, EXIT_STOP = 97, 2
EXIT_PAUSED = 5                            # the runner: the budget guard paused the run (budget-continue lifts it)
PHASE_ENV = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/root",
             "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1",
             "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1"}
FORECAST_CPU_HOURS = {"plumbing": 2.5, "exploratory": 16.0, "confirmatory_band": 16.0, "confirmatory_main": 8.5}
BUDGET_STAGES = ("exploratory", "confirmatory_band", "confirmatory_main")   # the stages that can pause (plumbing has
#                                                                            no phase of the official plan)
PAUSE_FACTOR = 2.0
CAP_CPU_HOURS = 50.0
# Peak memory from the plumbing (section 13): the reproduction check runs these kinds at full length in one process,
# whose peak seeds their reservations; every other kind reserves at least SMOKE_FLOOR_GIB until an official attempt of
# it has been measured.
REPRO_PEAK_KINDS = ("critic", "critic_full", "critic_second", "switch", "clone", "reference_actors", "policy")
SMOKE_FLOOR_GIB = 2.0
V7_RESULTS = V7_DIR / "results" / "BUNDLE.json"            # version 7's gathered results (its run has finished)
RUNFILES_BRANCH = "claude/near80-run-files"

# Output layout under an output root (OUT, or a scratch or fixture folder). Every JSON has a '<name>.sha256' sidecar.
LAYOUT = dict(
    fixed="fixed/fixed_checks.json",
    data_main="data/main/{world}/seed{seed}/data.json",
    data_band="data/band/{world}/seed{seed}/data.json",
    train_file="data/{setting}/{world}/seed{seed}/train.hdf5",
    critic="learners/{setting}/{world}/seed{seed}/critic.json",
    critic_full="learners/full/stall/seed{seed}/critic.json",
    critic_second="learners/{setting}/{world}/seed{seed}/critic_second.json",
    switch="learners/{setting}/{world}/seed{seed}/switch.json",
    clone="learners/{setting}/{world}/seed{seed}/clone.json",
    reference_actors="learners/{setting}/{world}/seed{seed}/reference_actors.json",
    v6_band="learners/{setting}/{world}/seed{seed}/v6_band.json",
    policy_lock="policy/{setting}/{world}/seed{seed}/lock.json",
    policy_stop="policy/{setting}/{world}/seed{seed}/stop.json",
    stage_lock="stage_lock/{group}.json",
    values="eval/{setting}/{world}/seed{seed}/values.json",
    readouts="readouts/{setting}/{world}/seed{seed}/readouts.json",
    look="looks/{group}.json",
    look_md="looks/{group}.md",
    halt_report="looks/halt_report.json",
    phase_stop="stops/{phase_id}.json",
    plumbing="plumbing/plumbing.json",                         # written only when the plumbing passed
    reproduction_differs="plumbing/reproduction_differs.json",  # the reproduction differed: nothing official, ever
    plumbing_failed="plumbing/failed_{utc}.json",               # any other plumbing failure (it may be run again)
    budget_continue="budget/{stage}/{n}.json",                  # one typed 'budget-continue STAGE' each
    version7_ended="version7_ended.json",                       # the typed acknowledgement that version 7 ended
    ledger_dir="ledger/{phase_id}",
    phase_lock="locks/{phase_id}.lock",
    log="logs/{phase_id}.log",
    interrupted="interrupted/{phase_id}/{utc}",
    status="status.json",
    runner_lock="runner.lock",
    runner_log="runner.log",
    retry="retry/{phase_id}/{n}.json",
)


def path(root, name, **kw):
    """The path of layout entry `name` under the output root `root`."""
    return Path(root) / LAYOUT[name].format(**kw)


def phase_id(kind, setting="x", world="x", seed=None, group=None):
    """'<kind>.<setting>.<world>.<s{seed} | group | x>', e.g. 'policy.near80.stall.s3', 'lock.x.x.exploratory'."""
    tail = f"s{int(seed)}" if seed is not None else (group or "x")
    return f"{kind}.{setting}.{world}.{tail}"


def group_of(seed):
    s = int(seed)
    if s in RESERVED:
        raise ValueError(f"refused: seed {s} belongs to version 6/7 (5-9)")
    if s in EXPLORATORY:
        return "exploratory"
    if s in CONFIRMATORY:
        return "confirmatory"
    raise ValueError(f"refused: seed {s} is not one of this version's seeds (0-4, 10-14)")


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
    """Every constant above as plain JSON-able values (copied into prereg.json's 'constants' block)."""
    out = {}
    for k, v in sorted(globals().items()):
        if k.isupper():
            out[k] = str(v) if isinstance(v, Path) else v
    return out
