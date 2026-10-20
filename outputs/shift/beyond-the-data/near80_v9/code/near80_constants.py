"""Near-80 (version 9, DRAFT): every fixed constant shared by the approval chain, the runner and the looks. DRAFT,
not frozen, awaiting the owner. Standard library only, so that the approval code can import it without the gate or
JAX.

Version 9 is the deciding test (PREREGISTRATION.md): the main setting only (the main test's densest-15% ball), on fresh
seeds, with BCA's two held-out groups pooled from every held-out recorded row outside the ball, an outcome-free count
of the full-throttle group before any training, a refusal rule that fails only the arms that need BCA, the plain twins
as trained arms, and a claims block (CLAIMS) whose open values are TODO markers that freeze.py refuses to freeze.

Changing any value here changes the code manifest and so the fingerprint the owner approves."""
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True

# ------------------------------------------------------------------------------------------------ the open values
# Every value DECISION.md must fill is this marker (or a string that starts with it). freeze.py refuses while any
# remains here, in prereg_draft.json, PREREGISTRATION.md or README.md (freeze.todo_problems).
TODO = "TODO(DECISION.md)"

# ------------------------------------------------------------------------------------------------ paths
ROOT = Path("/home/user/bca")
N80_REL = "outputs/shift/beyond-the-data/near80_v9"
N80 = ROOT / N80_REL
CODE = N80 / "code"
OUT = ROOT / "runs" / "shift_explore" / "near80_v9"         # official outputs (write-once)
PRE_OUT = OUT / "pre"                                       # the frozen-code no-training checks before approval
BTD = ROOT / "outputs" / "shift" / "beyond-the-data"
S1_DIR = BTD / "restructure" / "step1"
S2_DIR = BTD / "restructure" / "step2"
S1_RESULTS = S1_DIR / "results" / "results.json"
S2_RESULTS = BTD / "restructure" / "step2_results" / "results.json"
S2_OUT = ROOT / "runs" / "shift_explore" / "restructure" / "step2"     # step 2's saved actors (desktop), read-only
V6_CODE = BTD / "stage2" / "code"
V7_DIR = BTD / "stage2_v7"
V7_PLACE = V7_DIR / "laptop" / "lib" / "selftest_v7.py"              # version 7's placement of the version 6 code
V8_DIR = BTD / "near80"                                                # version 8 (spent): never imported
V8_R8 = ROOT / "runs" / "shift_explore" / "near80"                    # version 8's runner folder (its locks)
V8_RESULTS = V8_DIR / "results-20261020T115246Z" / "RESULTS.json"    # version 8's gathered results (finished)
PYTHON = "/tmp/claude-0/bcavenv/bin/python"

# ------------------------------------------------------------------------------------------------ identity
LABEL = ("Version 9, near-80 deciding test: the main setting on fresh seeds, BCA's two groups pooled from every "
         "held-out recorded row outside the ball")
VERSION = "version 9, near-80 deciding test"
BRANCH = "claude/main-bca-connection-x4o46s"
APPROVER_EMAIL = "35092758+dbayha7@users.noreply.github.com"
APPROVER_NAME = "David Bayha"
AGENT_EMAILS = ("noreply@anthropic.com",)
# The owner's answer to the choices of section 16 (not given yet: the freeze refuses while it is open).
OWNER_ANSWER = dict(text=TODO, utc=TODO, where=TODO, choices="every default of section 16 unless the answer says other")

# ------------------------------------------------------------------------------------------------ the approval files
PREREG_KIND = "near80v9-preregistration"
MANIFEST_KIND = "near80v9-code-manifest"
LOCAL_KIND = "near80v9-local-approval"
IMPORT_KIND = "near80v9-import-record"
LOCAL_MODE = "local-terminal"
STATUS_FINAL = "final, awaiting approval"
STATUS_LINE_FINAL = "FINAL, awaiting the owner's approval. Nothing has been trained."
PREFIX_LEN = 16
PREREG_NAME = "prereg.json"
MANIFEST_NAME = "code_manifest.json"
APPROVAL_NAME = "approval_near80v9_{prefix}.json"
IMPORT_NAME = "import_near80v9_{prefix}.json"
HISTORY_DIR = "prereg_history"
DRAFT_HISTORY_NAME = "prereg_draft_{prefix}.json"
MANIFEST_HISTORY_DIR = "code_manifest_history"
PRE_APPROVAL_DIR = "pre_approval"
NO_TRAINING_FROZEN_NAME = "no_training_checks_seeds0_4.json"
TEST_MARKER = "NEAR80V9_TEST_FIXTURE"
TEST_MARKER_TEXT = "near-80 version 9 test fixture: synthetic files only\n"

# Files the code manifest lists, relative to N80 (code/ is listed by name; freeze refuses any other file there).
TOP_FILES = ("PREREGISTRATION.md", "README.md", "approve_local_near80_v9.py", "run_near80_v9.sh",
             "push_runfiles_near80_v9.sh")
CODE_FILES = ("near80_constants.py", "near80lib.py", "facts_v9.py", "test_near80.py", "approval_check.py",
              "import_approval.py", "near80_prereg.py", "freeze.py", "test_approval.py", "near80_runner.py",
              "phases.py", "phase_main.py", "settingdata.py", "learner_phases.py", "policy_phase.py",
              "evaluate_phase.py", "ledger.py", "stubs.py", "test_runner.py", "pooled.py", "claims_text.py",
              "test_pooled.py", "test_v9.py")
TEST_FILES = ("test_near80.py", "test_approval.py", "test_runner.py", "test_pooled.py", "test_v9.py")
# Repository code the version-9 code imports or executes, pinned file by file in the manifest (repository_files).
PIN_DIRS = ("experiments/shift", "calibration", "algorithms", "runtime",
            "outputs/shift/beyond-the-data/restructure/step1", "outputs/shift/beyond-the-data/restructure/step2",
            "outputs/shift/beyond-the-data/stage2/code", "runs/shift_explore/room/code",
            "outputs/shift/beyond-the-data/stage2_v7/code", "outputs/shift/beyond-the-data/stage2_v7/laptop/lib")
PIN_SUFFIXES = (".py", ".sh", ".json", ".md")
PIN_FILES = ("outputs/shift/beyond-the-data/restructure/step1/results/results.json",
             "outputs/shift/beyond-the-data/restructure/step2_results/results.json",
             "outputs/shift/beyond-the-data/stage2/prereg.json")

# ------------------------------------------------------------------------------------------------ seeds and worlds
# The main study's generator (master seed 20261012). Every seed block any study of it has generated, loaded or named,
# from every seed registry and record in the repository (checked 20 October 2026: origin/claude/placement-test-design's
# placement_test/seed_registry.json, near80/prereg.json, stage2/PREREGISTRATION.md, experiments/shift/acceptance.py and
# stage0.py, and a search of every branch for seed lists and run folders of this generator). Version 9 never touches
# any of them; its fresh seeds come from FRESH_POOL, which none of them names.
SEED_REGISTRY = {
    "0-4": "seen by everything (the gate, version 6 Stage 2, step 1, step 2, routes round 1, near-80 version 8's "
           "exploratory stage); version 9 uses seed 0 for the plumbing only and seeds 0-4 for the frozen no-training "
           "facts (commands only)",
    "5-9": "version 6/7's confirmatory seeds (acceptance.py retune_gate_seeds; stage2/PREREGISTRATION.md); their "
           "retune-physics logs exist (stage2_gpu payload)",
    "10-14": "near-80 version 8's fresh seeds (near80/prereg.json): generated and trained on 16-20 October 2026; spent",
    "15-19": "named by near-80 version 8 as the fallback block for step 2's confirmation (near80/prereg.json, choice "
             "6); never generated",
    "20-24": "named by the placement design (not approved) as an optional exploratory margin block "
             "(placement_test/seed_registry.json)",
    "25-34": "reserved by the placement design (not approved): 25-29 confirmatory, 30-34 its extension "
             "(placement_test/DESIGN.md, seed_registry.json)",
    "35-49": "version 9's reservation (this file, FRESH_POOL): never generated, never named elsewhere (process.md "
             "section 2.1)",
    "100-119": "the original gate's confirmation seeds (acceptance.py); logs generated",
    "900-909": "Stage 0's design seeds (stage0.py)",
}
EXPLORATORY = (0, 1, 2, 3, 4)              # seen: the plumbing (seed 0) and the frozen no-training facts only
SPENT = (10, 11, 12, 13, 14)               # version 8's fresh seeds: refused always
RESERVED = (5, 6, 7, 8, 9)                 # version 6/7's; refused always, in every mode
NAMED_ELSEWHERE = tuple(range(15, 35)) + tuple(range(100, 120)) + tuple(range(900, 910))   # refused always
FRESH_POOL = tuple(range(35, 50))          # version 9's reservation: 15 fresh seeds that no study has named
                                           # (DECISION.md section 2.3; process.md section 2.2)
N_FRESH = TODO                             # 5, 10 or 15, from DECISION.md: the first N_FRESH of FRESH_POOL are used
N_FRESH_CHOICES = (5, 10, 15)
# The confirmatory seeds: the first N_FRESH of the pool. While N_FRESH is open the whole pool stands in (the plan and
# the tests can be built; the freeze refuses until N_FRESH is set). The rest of the pool stays reserved and refused.
CONFIRMATORY = FRESH_POOL[:N_FRESH] if isinstance(N_FRESH, int) else FRESH_POOL
GROUPS = ("exploratory", "confirmatory")
OFFICIAL_GROUPS = ("confirmatory",)        # the official plan: seeds 0-4 carry no phase of the official run
WORLDS = ("stall", "no_stall")
WORLD_TAG = {"stall": 1, "no_stall": 2}    # step 1's tags; never end a tag list in 0
SITUATIONS = ("recorded", "strong_current", "big_waves", "big_kicks")   # thruster_chain.SITUATIONS order
SCORED = "strong_current"
KEEPS_SITUATIONS = ("recorded", "big_waves", "big_kicks")               # section 9's order
SITUATION_PLAIN = {"recorded": "recorded conditions", "strong_current": "strong current",
                   "big_waves": "big waves", "big_kicks": "big kicks"}
WORLD_PLAIN = {"stall": "stalling world", "no_stall": "no-stall world"}

# ------------------------------------------------------------------------------------------------ settings and data
SETTINGS = ("main",)
SETTING_REMOVAL = {"main": "gap"}
HOLD = 0.70                                # held to 70% at test (reported: the unseen-throttle gain)
CAP = 0.80                                 # capped at 80% at test (reported: the overshoot loss)
VARIANT = "base"                           # the gate's variant; fresh seeds use its recorder, sizes and stall slope
# BCA's two held-out groups (PREREGISTRATION.md section 3): the survivors (training rows past t_lo, held out from the
# second critic) and every row of these splits whose command lies outside the ball, split at t_lo. The calibration
# bank is part of the calibration split. No scored number reads these rows.
HELDOUT_SPLITS = ("calibration", "evaluation")
CERTIFIABLE_MIN = 36                       # calibration.reference.certifiable's bar at 0.1 / 0.95 / 1,000 (reported)

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
MEANDIFF_MARGIN = 0.10                     # the plain twins: on when mean(full-throttle) - mean(rest) > 0.10
W0_TOL = 1e-5

# ------------------------------------------------------------------------------------------------ arms (logical ids)
PRIMARY = "obca_k1"
W0 = "w0_check"
STEP2_RAMP_ARMS = ("obca_k1", "obca_k2", "obca_forced", "fixed_gated", "conformal_k1", "meangap_k1")
PLAIN_TWINS = ("meandiff_k1", "meandiff_limit")    # the mean-difference switch with the ramp; with the range limit
MATCHED = ("even_pull", "push_less", "random_1", "random_2")
WORLD_BLIND = ("standin", "even_pull", "random_1", "random_2", "obca_forced")   # step 2's order (ties: first)
HAND_SET_LIMIT = "range_hinge"
# The refusal rule (section 8): when BCA's tail threshold is not finite where its switch is on, these arms fail that
# seed-world (they read BCA's height, or are matched to obca_k1's strength); every other arm trains and is scored.
BCA_HEIGHT_ARMS = ("obca_k1", "obca_k2", "obca_forced")
BCA_MATCHED_ARMS = ("standin",) + MATCHED
ARMS = {
    "main": ("none", "clone", "full_none", W0) + STEP2_RAMP_ARMS + PLAIN_TWINS + ("perfect_band", "range_hinge")
            + MATCHED + ("standin",),
}
STALL_ONLY_ARMS = ("full_none",)
ARM_PLAIN = {"obca_k1": "BCA's switched ramp", "obca_forced": "the always-on ramp", "standin": "the matched range limit",
             "even_pull": "the matched even pull", "random_1": "the first matched random surface",
             "random_2": "the second matched random surface", "range_hinge": "the hand-set range limit",
             "none": "the controller trained without caution", "full_none": "the full-log controller",
             "push_less": "the matched push-less penalty", "perfect_band": "the perfect band",
             "clone": "the cloned actor", "w0_check": "the plumbing check", "obca_k2": "the ramp at twice BCA's height",
             "fixed_gated": "BCA's switch with height 1.0", "conformal_k1": "the split-conformal switch",
             "meangap_k1": "the rank-test switch", "meandiff_k1": "the mean-difference switch with the ramp",
             "meandiff_limit": "the mean-difference switch with the hand-set range limit"}

# ------------------------------------------------------------------------------------------------ the claims block
# C1: version 8's claim 4 (step 2's three go marks in the main setting), marks unchanged.
# C2: BCA's switched ramp beats each plain twin, paired by seed. Its margins, its statistic and its multiplicity rule
#     are DECISION.md's: they stay TODO here until filled, and freeze.py refuses while any TODO remains.
# Both: a seed whose data or a needed phase failed, or where BCA refused (its height arms and the matched rivals
# failed), fails every mark that needs BCA; it is never replaced (section 9).
C2_STATISTICS = ("mean", "paired_t", "interval")  # mean: the pooled paired difference minus the margin is >= 0;
#                                                   paired_t: a one-sided paired t-test that it exceeds the margin;
#                                                   interval: the two-sided t-interval at the comparison's level
#                                                   (alpha, or alpha / comparisons under bonferroni): passes when
#                                                   the mean is at least the margin and the interval's lower end is
#                                                   above 0; 'excluded' when its upper end is below the margin,
#                                                   'unresolved' otherwise (reported; DECISION.md section 3)
C2_MULTIPLICITY = ("all", "holm", "bonferroni")   # all: C2 holds only if every comparison passes (intersection-union,
#                                                   no adjustment); holm, bonferroni: family-wise at C2's alpha
C2_MEASURES = ("net", "stall_points")             # net: step 2's net across worlds (strong current); stall_points:
#                                                   strong-current points in the stalling world
CLAIMS = dict(
    C1=dict(name="step 2 confirmed in the main setting (version 8's claim 4)",
            marks=dict(net_lead_margin=10.0, keeps_floor=-3.0, even_pull_margin=15.0),
            failed_seed="fails every mark"),
    C2=dict(name="BCA's switched ramp beats each plain twin",
            primary=PRIMARY,
            comparators=("conformal_k1", "meandiff_k1", "meandiff_limit"),
            measure="net",
            margins={"conformal_k1": TODO, "meandiff_k1": TODO, "meandiff_limit": TODO},
            statistic=TODO,
            alpha=TODO,                    # one-sided level of paired_t / the family-wise level; None for 'mean'
            min_positive_seeds=TODO,       # BCA ahead of the twin in at least this many seeds; None for no such rule
            multiplicity=TODO,
            noise_floor_max=TODO,          # None, or C2 is reported only when a paired difference's standard error
            #                                across seeds exceeds this many points
            failed_seed="fails every comparison"),
    # The decision rule (section 9.4): which outcome the test reads as BCA-specific credit shown. One of
    # DECISION_RULES; DECISION.md's.
    decision=TODO,
    # Validity (section 9.0), version 8's rule carried to the main setting: without the harm the test is built on,
    # neither claim is read either way. Pooled over the seeds that did not fail, stalling world, strong current.
    validity=dict(harm_min=30.0, past80_min=0.25,
                  harm="the full-log controller minus the controller trained without caution",
                  past80="the controller trained without caution's share of steps past 80%"),
)
DECISION_RULES = ("C2", "C1_and_C2")
MARKS = dict(CLAIMS["C1"]["marks"])
DEFAULTS = dict(seed_pool="35-49", n_fresh=N_FRESH, pooled_groups="every held-out recorded row outside the ball",
                pre_training_count=True, refusal_fails_bca_arms_only=True, plain_twins_trained=True,
                hosts="TD3+BC actor-only", when="after version 8's run has finished (it has)", lanes=10,
                name="version 9, near-80 deciding test")

# ------------------------------------------------------------------------------------------------ the runner
PHASE_KINDS = ("data_main", "critic", "critic_full", "critic_second", "switch", "clone", "reference_actors", "policy",
               "lock", "evaluate", "look_exploratory", "look_confirmatory")
PHASE_PLAIN = {"data_main": "the main setting's data, the gate's checks and the outcome-free group count",
               "critic": "the critic", "critic_full": "the full-log critic", "critic_second": "BCA's second critic",
               "switch": "BCA's switch and height (the pooled held-out groups)", "clone": "the cloned actor",
               "reference_actors": "the no-caution and full-log actors", "policy": "the actor training",
               "lock": "the lock", "evaluate": "the evaluation", "look_exploratory": "the plumbing look",
               "look_confirmatory": "the confirmatory look"}
DEFAULT_LANES, MAX_LANES = 10, 16
GIB = 1024 ** 3
UNMEASURED_GIB, HEADROOM_GIB, MEASURED_FACTOR = 7, 8, 1.25
POLICY_KINDS = ("policy",)
POLICY_CAP_UNMEASURED = 2
EXIT_REFUSED, EXIT_STOP = 97, 2
EXIT_PAUSED = 5                            # the runner: the budget guard paused the run (budget-continue lifts it)
# PYTHONPYCACHEPREFIX: Python reads and writes compiled bytecode only under this (absent) folder, so a __pycache__ next
# to a pinned source file is never loaded in place of the verified source (with PYTHONDONTWRITEBYTECODE=1, nothing is
# written there either). Stub runs, on synthetic fixtures only, leave it out (near80_runner.Cfg.phase_env).
PYCACHE_PREFIX = "/nonexistent/near80_no_pycache"
PHASE_ENV = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/root",
             "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONPYCACHEPREFIX": PYCACHE_PREFIX, "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1"}
# CPU-hours, from version 8's ledger (results-20261020T115246Z/ledger): the main setting cost 1.51-1.54 CPU-hours per
# fresh seed where both worlds' actors trained (seeds 11, 14; stalling-world policy 41-42 min, no-stall 9-10 min, three
# critics about 5.7 min each); version 9 adds two plain-twin programs per world (about 1.6 min), so 1.6 per seed.
CPU_HOURS_PER_FRESH_SEED = 1.6
FORECAST_CPU_HOURS = {"plumbing": 2.5,
                      "confirmatory_main": CPU_HOURS_PER_FRESH_SEED * len(CONFIRMATORY)}
BUDGET_STAGES = ("confirmatory_main",)     # the stage that can pause (the plumbing has no phase of the official plan)
PAUSE_FACTOR = 2.0
CAP_CPU_HOURS = 2.0 * (FORECAST_CPU_HOURS["plumbing"] + FORECAST_CPU_HOURS["confirmatory_main"])
# Peak memory from the plumbing (section 13): the reproduction check runs these kinds at full length in one process,
# whose peak seeds their reservations; every other kind reserves at least SMOKE_FLOOR_GIB until an official attempt of
# it has been measured.
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
    plumbing="plumbing/plumbing.json",                         # written only when the plumbing passed
    reproduction_differs="plumbing/reproduction_differs.json",  # the reproduction differed: nothing official, ever
    plumbing_failed="plumbing/failed_{utc}.json",               # any other plumbing failure (it may be run again)
    budget_continue="budget/{stage}/{n}.json",                  # one typed 'budget-continue STAGE' each
    version8_ended="version8_ended.json",                       # the typed acknowledgement that version 8 ended
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
    """'<kind>.<setting>.<world>.<s{seed} | group | x>', e.g. 'policy.main.stall.s40', 'lock.x.x.confirmatory'."""
    tail = f"s{int(seed)}" if seed is not None else (group or "x")
    return f"{kind}.{setting}.{world}.{tail}"


def refusal_of(seed):
    """Why a seed is not one of this version's (None when it is): every taken block is refused in every mode."""
    s = int(seed)
    if s in EXPLORATORY or s in CONFIRMATORY:
        return None
    if s in RESERVED:
        return f"refused: seed {s} belongs to version 6/7 (5-9)"
    if s in SPENT:
        return f"refused: seed {s} is one of version 8's fresh seeds (10-14), spent on 16-20 October 2026"
    if s in FRESH_POOL:
        return (f"refused: seed {s} is reserved for version 9 (35-49) but is not among its confirmatory seeds "
                f"({CONFIRMATORY[0]}-{CONFIRMATORY[-1]})")
    if s in NAMED_ELSEWHERE:
        return f"refused: seed {s} is named by another study of this generator (SEED_REGISTRY)"
    return f"refused: seed {s} is not one of this version's seeds (0-4 for the plumbing, {seed_words()} fresh)"


def seed_words():
    return f"{CONFIRMATORY[0]}-{CONFIRMATORY[-1]}"


def group_of(seed):
    why = refusal_of(seed)
    if why is not None:
        raise ValueError(why)
    return "exploratory" if int(seed) in EXPLORATORY else "confirmatory"


_open_default = object()


def todo_paths(x=_open_default, path="CLAIMS"):
    """Every place of x (default: the claims block, the seed count and the owner's answer) that still holds the TODO
    marker, as 'name/key/...' strings (freeze.py refuses while this is not empty). A value None is filled (it means
    'no such rule'), not open."""
    if x is _open_default:
        out = todo_paths(CLAIMS, "CLAIMS") + todo_paths(OWNER_ANSWER, "OWNER_ANSWER")
        return out + (["N_FRESH"] if not isinstance(N_FRESH, int) else [])
    if isinstance(x, dict):
        return [p for k, v in x.items() for p in todo_paths(v, f"{path}/{k}")]
    if isinstance(x, (list, tuple)):
        return [p for i, v in enumerate(x) for p in todo_paths(v, f"{path}/{i}")]
    return [path] if isinstance(x, str) and x.startswith(TODO) else []


def load_local(name):
    """N80/code/<name>.py loaded by explicit path as module 'near80v9_<name>' (avoids step 1's and step 2's pooled.py
    and freeze.py, which step 2's machinery puts first on sys.path)."""
    key = f"near80v9_{name}"
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
