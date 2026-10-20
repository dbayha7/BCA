"""Near-80 (version 9): every fixed constant shared by the approval chain, the runner and the looks. Not frozen yet,
awaiting the owner. Standard library only, so that the approval code can import it without the gate or JAX.

Version 9 is the deciding test (PREREGISTRATION.md; the design is decide/DECISION.md draft 2): the main setting only
(the main test's densest-15% ball), on 15 fresh seeds (35-49), with BCA's two held-out groups pooled from every
held-out recorded row outside the ball, an outcome-free count of the full-throttle group before any training and a
held-out top-up to 36 rows, a refusal rule that fails only BCA's arms, step 2's six-arm program with the textbook
split-conformal switch in conformal_k1's slot, the plain switch's dose arms, the computed plain switch with the
hand-set limit, and three claims (C1, C2a, C2b) read by one verdict table.

Changing any value here changes the code manifest and so the fingerprint the owner approves. This file is never
amended (approve_local_near80_v9.py, NEVER_AMENDED)."""
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True

# ------------------------------------------------------------------------------------------------ the open-value marker
# The marker of a value the design has not filled. Version 9 has none left (every value below is DECISION.md's);
# freeze.py still refuses while the marker appears in this file (outside this line), prereg_draft.json,
# PREREGISTRATION.md or README.md, or a claims value is None (freeze.todo_problems, freeze.claims_problems).
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
V7_PLACE = BTD / "stage2_v7" / "laptop" / "lib" / "selftest_v7.py"   # version 7's placement of the version 6 code
V8_DIR = BTD / "near80"                                                # version 8 (spent): never imported
V8_R8 = ROOT / "runs" / "shift_explore" / "near80"                    # version 8's runner folder (its locks)
V8_RESULTS = V8_DIR / "results-20261020T115246Z" / "RESULTS.json"    # version 8's gathered results (finished)
V8_PREREG = V8_DIR / "prereg.json"                                    # version 8's approved sentences ("as version 8")
PYTHON = "/tmp/claude-0/bcavenv/bin/python"

# ------------------------------------------------------------------------------------------------ identity
LABEL = ("Version 9, near-80 deciding test: the main setting on fresh seeds 35-49, BCA's two groups pooled from every "
         "held-out recorded row outside the ball and topped up to 36 rows")
VERSION = "version 9, near-80 deciding test"
BRANCH = "claude/near80-v9"
APPROVER_EMAIL = "35092758+dbayha7@users.noreply.github.com"
APPROVER_NAME = "David Bayha"
AGENT_EMAILS = ("noreply@anthropic.com",)
# The owner's answer to the choices of section 16 is the approval itself: DECISION.md section 4.4 is titled "Owner
# decisions. Approving the package accepts the defaults". No words of the owner are written here by anyone else.
OWNER_ANSWER = dict(how="approving this fingerprint at the owner's own terminal (section 11) accepts every default of "
                        "section 16 (DECISION.md section 4.4)",
                    given_at="the owner's approval (section 11)", choices="every default of section 16")

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
TAG_CHECK_NAME = "tag_check.json"                   # the top-up streams' tags, unused on every branch (freeze)
PREVIEW_NAME = "preview_v9.json"                    # the pinned cloud preview (DECISION.md section 2.8)
PREVIEW_SCRIPT = "preview/preview_v9.py"
TEST_MARKER = "NEAR80V9_TEST_FIXTURE"
TEST_MARKER_TEXT = "near-80 version 9 test fixture: synthetic files only\n"

# Files the code manifest lists, relative to N80 (code/ is listed by name; freeze refuses any other file there).
TOP_FILES = ("PREREGISTRATION.md", "README.md", "approve_local_near80_v9.py", "run_near80_v9.sh",
             "push_runfiles_near80_v9.sh", PREVIEW_SCRIPT)
CODE_FILES = ("near80_constants.py", "near80lib.py", "topup.py", "arms_v9.py", "facts_v9.py", "test_near80.py",
              "approval_check.py", "import_approval.py", "near80_prereg.py", "freeze.py", "test_approval.py",
              "near80_runner.py", "phases.py", "phase_main.py", "settingdata.py", "learner_phases.py",
              "policy_phase.py", "evaluate_phase.py", "ledger.py", "stubs.py", "test_runner.py", "pooled.py",
              "claims_text.py", "test_pooled.py", "test_v9.py", "test_pooled_v9.py")
TEST_FILES = ("test_near80.py", "test_approval.py", "test_runner.py", "test_pooled.py", "test_v9.py",
              "test_pooled_v9.py")
# Repository code the version-9 code imports or executes, pinned file by file in the manifest (repository_files).
PIN_DIRS = ("experiments/shift", "calibration", "algorithms", "runtime",
            "outputs/shift/beyond-the-data/restructure/step1", "outputs/shift/beyond-the-data/restructure/step2",
            "outputs/shift/beyond-the-data/stage2/code", "runs/shift_explore/room/code",
            "outputs/shift/beyond-the-data/stage2_v7/code", "outputs/shift/beyond-the-data/stage2_v7/laptop/lib")
PIN_SUFFIXES = (".py", ".sh", ".json", ".md")
PIN_FILES = ("outputs/shift/beyond-the-data/restructure/step1/results/results.json",
             "outputs/shift/beyond-the-data/restructure/step2_results/results.json",
             "outputs/shift/beyond-the-data/stage2/prereg.json",
             "outputs/shift/beyond-the-data/near80/prereg.json",
             "outputs/shift/beyond-the-data/near80/prereg.json.sha256")

# ------------------------------------------------------------------------------------------------ seeds and worlds
# The main study's generator (master seed 20261012). Every seed block any study of it has generated, loaded or named,
# from every seed registry and record in the repository (checked 20 October 2026: origin/claude/placement-test-design's
# placement_test/seed_registry.json, near80/prereg.json, stage2/PREREGISTRATION.md, experiments/shift/acceptance.py and
# stage0.py, and a search of every branch for seed lists and run folders of this generator). Version 9 never touches
# any of them; its fresh seeds are FRESH_POOL, which none of them names.
SEED_REGISTRY = {
    "0-4": "seen by everything (the gate, version 6 Stage 2, step 1, step 2, routes round 1, near-80 version 8's "
           "exploratory stage); version 9 uses seed 0 for the plumbing only, and seeds 0-4 for the frozen no-training "
           "facts (commands only) and the pinned preview (no actor trained)",
    "5-9": "version 6/7's confirmatory seeds (acceptance.py retune_gate_seeds; stage2/PREREGISTRATION.md); their "
           "retune-physics logs exist (stage2_gpu payload)",
    "10-14": "near-80 version 8's fresh seeds (near80/prereg.json): generated and trained on 16-20 October 2026; spent "
             "(read only by the pinned preview, which trains no actor)",
    "15-19": "named by near-80 version 8 as the fallback block for step 2's confirmation (near80/prereg.json, choice "
             "6); never generated",
    "20-24": "named by the placement design (not approved) as an optional exploratory margin block "
             "(placement_test/seed_registry.json)",
    "25-34": "reserved by the placement design (not approved): 25-29 confirmatory, 30-34 its extension "
             "(placement_test/DESIGN.md, seed_registry.json)",
    "35-49": "version 9's fresh seeds (this file, FRESH_POOL): never generated, never named elsewhere (DECISION.md "
             "section 2.3; process.md section 2.1)",
    "100-119": "the original gate's confirmation seeds (acceptance.py); logs generated",
    "900-909": "Stage 0's design seeds (stage0.py)",
}
EXPLORATORY = (0, 1, 2, 3, 4)              # seen: the plumbing (seed 0), the frozen no-training facts, the preview
SPENT = (10, 11, 12, 13, 14)               # version 8's fresh seeds: refused always (the preview's one exception)
RESERVED = (5, 6, 7, 8, 9)                 # version 6/7's; refused always, in every mode
NAMED_ELSEWHERE = tuple(range(15, 35)) + tuple(range(100, 120)) + tuple(range(900, 910))   # refused always
FRESH_POOL = tuple(range(35, 50))          # version 9's 15 fresh seeds (DECISION.md section 2.3)
N_FRESH = 15                               # DECISION.md section 4.4, default 2: 15 (35-49); the option was 10
N_FRESH_CHOICES = (10, 15)
CONFIRMATORY = FRESH_POOL[:N_FRESH]
GROUPS = ("exploratory", "confirmatory")
OFFICIAL_GROUPS = ("confirmatory",)        # the official plan: seeds 0-4 carry no phase of the official run
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
HOLD = 0.70                                # held to 70% at test (reported: the unseen-throttle gain)
CAP = 0.80                                 # capped at 80% at test (reported: the overshoot loss)
VARIANT = "base"                           # the gate's variant; fresh seeds use its recorder, sizes and stall slope
# BCA's two held-out groups (PREREGISTRATION.md section 3.1): the survivors (training rows past t_lo, held out from the
# second critic) and every row of these splits whose command lies outside the ball, plus the top-up rows outside the
# ball, split at t_lo. The calibration bank is part of the calibration split. No scored number reads these rows.
HELDOUT_SPLITS = ("calibration", "evaluation")
CERTIFIABLE_MIN = 36                       # calibration.reference.certifiable's bar at 0.1 / 0.95 / 1,000
EXACT_FLOOR = 29                           # the smallest group that can certify at all (reported)
# The held-out top-up (section 3.2): blocks of 100 new recorded episodes, block k (1-10) from
# default_rng([20261012, seed, 61, k]) (starts and noise) and default_rng([20261012, seed, 62, k]) (coin), the same
# streams in both worlds; each world adds blocks until its own count reaches CERTIFIABLE_MIN or the cap.
TOPUP = dict(master=20261012, tag_starts=61, tag_coin=62, block_episodes=100, first_block=1, cap=10,
             target=CERTIFIABLE_MIN)
TOPUP_SPLIT_CODE = 3                       # infos/split of a top-up row in the groups' view (0-2 are the log's)
ONE_PER_EPISODE_TAG_HEAD = (20261020, 9901)    # the one-row-per-episode variant: [..., seed, WORLD_TAG] + [1] / [2]
COUNT_SCRAMBLE_TAG = (20261020, 9902)      # the data-time check that the count reads no outcome (a permutation)

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
SPLIT_CONFORMAL = dict(level=0.90, rank="ceil((n + 1) * level)")        # the textbook split-conformal rank
WILKS = dict(level=0.90, confidence=0.95)  # the exact Wilks twin (reported only)
MEANDIFF_MARGIN = 0.10                     # the plain switch: on when mean(full-throttle) - mean(rest) > 0.10
W0_TOL = 1e-5

# ------------------------------------------------------------------------------------------------ arms (logical ids)
PRIMARY = "obca_k1"
W0 = "w0_check"
# Step 2's six-arm program, with the textbook split-conformal switch in conformal_k1's slot (section 5).
STEP2_RAMP_ARMS = ("obca_k1", "obca_k2", "obca_forced", "fixed_gated", "splitconf_k1", "meangap_k1")
# Step 2's own six arms: the plumbing's reproduction check trains step 2's own arm table (section 8.2).
STEP2_OWN_RAMP_ARMS = ("obca_k1", "obca_k2", "obca_forced", "fixed_gated", "conformal_k1", "meangap_k1")
SLOT_OF_SPLITCONF = 4                      # the place splitconf_k1 takes in step 2's program (conformal_k1's)
DOSE_ARMS = ("plain_h25", "plain_h50")    # the plain switch with the ramp at a fixed height; trained only where on
DOSE_HEIGHTS = {"plain_h25": 2.5, "plain_h50": 5.0}
PLAIN_LIMIT = "plain_limit"                # the plain switch with the hand-set limit: computed, never trained
MATCHED = ("even_pull", "push_less", "random_1", "random_2")
WORLD_BLIND = ("standin", "even_pull", "random_1", "random_2", "obca_forced")   # step 2's order (ties: first)
HAND_SET_LIMIT = "range_hinge"
# The refusal rule (section 3.3): obca_k1 and obca_k2 are refused in a world where BCA's switch is on and R_tail is
# +infinity; obca_forced is refused in both worlds when the stalling world's R_tail is +infinity (switch on or off);
# the matched rivals are unmatched when the stalling world's obca_k1 is refused. fixed_gated is never refused.
BCA_HEIGHT_ARMS = ("obca_k1", "obca_k2", "obca_forced")
BCA_MATCHED_ARMS = ("standin",) + MATCHED
ARMS = {
    "main": ("none", "clone", "full_none", W0) + STEP2_RAMP_ARMS + DOSE_ARMS + (PLAIN_LIMIT, "perfect_band",
                                                                               "range_hinge") + MATCHED + ("standin",),
}
TRAINED_ONLY_WHERE_ON = DOSE_ARMS          # where the plain switch is off they are the no-caution actor (computed)
COMPUTED_ARMS = (PLAIN_LIMIT,)             # always computed from two trained actors
STALL_ONLY_ARMS = ("full_none",)
ARM_PLAIN = {"obca_k1": "BCA's switched ramp", "obca_forced": "the always-on ramp", "standin": "the matched range limit",
             "even_pull": "the matched even pull", "random_1": "the first matched random surface",
             "random_2": "the second matched random surface", "range_hinge": "the hand-set range limit",
             "none": "the controller trained without caution", "full_none": "the full-log controller",
             "push_less": "the matched push-less penalty", "perfect_band": "the perfect band",
             "clone": "the cloned actor", "w0_check": "the plumbing check", "obca_k2": "the ramp at twice BCA's height",
             "fixed_gated": "BCA's switch with height 1.0", "splitconf_k1": "the split-conformal switch",
             "conformal_k1": "step 2's split-conformal switch", "meangap_k1": "the rank-test switch",
             "plain_h25": "the plain switch with the ramp at height 2.5",
             "plain_h50": "the plain switch with the ramp at height 5.0",
             "plain_limit": "the plain switch with the hand-set limit"}

# ------------------------------------------------------------------------------------------------ the claims
# Fixed by DECISION.md draft 2 (sections 2.5 and 3; PREREGISTRATION.md sections 9, 13 and 14). Not choices.
IMPUTE_CAP = {15: 2, 10: 1}[N_FRESH]       # seeds missing for reasons not BCA's that C1 imputes (more: unreadable)
C2_MIN_SEEDS = {15: 13, 10: 9}[N_FRESH]    # each contrast needs its own numbers on at least this many seeds
CLAIMS = dict(
    C1=dict(name="step 2's result on fresh seeds (L1)",
            marks=dict(net_lead_margin=10.0, keeps_floor=-3.0, even_pull_margin=15.0),
            world_blind=WORLD_BLIND, impute_cap=IMPUTE_CAP,
            bca_own=("a refusal", "BCA's switch off in the stalling world", "BCA's switched-ramp actor not finite"),
            not_bca=("a failed data check", "a plumbing stop of a phase", "a rival-side failure")),
    C2a=dict(name="BCA's height against plain split conformal's (L3a, a dose)", primary=PRIMARY,
             comparator="splitconf_k1", outcomes=("shown", "not shown")),
    C2b=dict(name="BCA against the plain switch with the hand-set limit (L2)", primary=PRIMARY,
             comparator=PLAIN_LIMIT, outcomes=("shown", "excluded", "unresolved"),
             readings=("U < 0: the plain rule did better",
                       "0 <= U < 5 and L > -5: BCA and the plain rule were within 5 points of each other",
                       "otherwise: the plain rule was within 5 points of BCA or ahead")),
    C2=dict(measure="net", bar=5.0, interval=0.95, statistic="two-sided t-interval, t(0.975, n - 1) from n",
            shown="mean >= bar and L > 0", excluded="U < bar (C2b only)", min_seeds=C2_MIN_SEEDS,
            impute="least favourable observed value: the smallest observed lead for shown, the largest for excluded"),
    dose=dict(arms=DOSE_ARMS, margin=2.0, compares="the better mean NET of the two fixed heights minus BCA's"),
    episode_share_min=0.80,
    validity=dict(harm_min=30.0, past80_min=0.25,
                  harm="the full-log controller minus the controller trained without caution",
                  past80="the controller trained without caution's share of steps past 80%"),
    decision="verdict_table",
)
DECISION_RULES = ("verdict_table",)
# The verdict table (section 14), in the order the first row that applies is chosen. R0 is the runner's (no look).
VERDICT_ROWS = ("R1", "R2", "R3_marks", "R3_refusal", "R3_switch_off", "R4", "R5", "R5u", "R6", "R7", "R7u")
MARKS = dict(CLAIMS["C1"]["marks"], harm_min=CLAIMS["validity"]["harm_min"],
             past80_min=CLAIMS["validity"]["past80_min"])
C2_BAR = CLAIMS["C2"]["bar"]
DOSE_MARGIN = CLAIMS["dose"]["margin"]
EPISODE_SHARE_MIN = CLAIMS["episode_share_min"]
DEFAULTS = dict(seed_pool="35-49", n_fresh=N_FRESH, pooled_groups="every held-out recorded row outside the ball",
                topup="to 36 rows, blocks of 100 episodes, at most 10", pre_training_count=True,
                refusal_fails_bca_arms_only=True, dose_arms=True, memory_rule="version 8's", exploratory_retrain=False,
                hosts="TD3+BC actor-only", lanes=10, branch=BRANCH, name=VERSION)

# ------------------------------------------------------------------------------------------------ stops and halts
# DECISION.md section 2.6 (PREREGISTRATION.md section 10): every event that changes what runs or what a claim reads,
# with the row of the verdict table it can feed. test_v9 scans the code: every PhaseStop / stop reason is a
# STOP_REASONS key of STOP_REASON_ROWS, every runner halt a key of HALT_KINDS, and each maps to a row here.
STOP_TABLE = {
    "plumbing_tests": dict(event="unit tests or smoke fail (plumbing)", feeds="R0",
                           section10="A plumbing failure before any official phase"),
    "reproduction": dict(event="reproduction differs (plumbing, seed 0)", feeds="R0",
                         section10="A plumbing failure before any official phase"),
    "failed_seed": dict(event="a gate data check fails on a fresh seed", feeds="R2",
                        section10="A failed seed"),
    "topup": dict(event="full-throttle group below 36", feeds="none", section10="The count below 36"),
    "topup_cap": dict(event="top-up cap reached below 36", feeds="R3", section10="The count below 36"),
    "refusal_on": dict(event="BCA refuses with its switch on", feeds="R3", section10="A refusal"),
    "stall_inf_off": dict(event="the stalling world's R_tail = +inf with BCA's switch off", feeds="R3",
                          section10="A refusal"),
    "no_stall_inf_off": dict(event="the no-stall world's R_tail = +inf with BCA's switch off", feeds="none",
                             section10="A refusal"),
    "bca_not_finite": dict(event="BCA's own switched-ramp actor not finite", feeds="R3",
                           section10="BCA's own switched-ramp actor not finite"),
    "rival_side": dict(event="a rival-side failure", feeds="R2", section10="A rival-side failure"),
    "phase_stop": dict(event="a plumbing stop of a phase", feeds="R2", section10="A plumbing stop of a phase"),
    "interrupted": dict(event="a phase attempt interrupted", feeds="none", section10="An interrupted attempt"),
    "runner_halt": dict(event="a phase attempt FAILED, or the runner halts", feeds="none",
                        section10="A failed attempt, or a halt of the runner"),
    "budget": dict(event="budget guard", feeds="none", section10="The budget guard"),
    "validity": dict(event="validity (2.5) fails", feeds="R1", section10="Validity fails"),
}
# Every recorded stop of a phase is one of STOP_REASONS (phases.py), each a plumbing stop of a phase.
STOP_REASON_ROWS = {"training_file": "phase_stop", "fingerprint": "phase_stop", "learner": "phase_stop",
                    "height0": "phase_stop", "check": "phase_stop"}
# Every halt of the runner (near80_runner.Runner.halt(kind, ...)), each "a phase attempt FAILED, or the runner halts".
HALT_KINDS = {"failed_attempt": "a phase attempt failed (any exit but done, stopped or refused)",
              "refused_attempt": "a phase refused (exit 97): a guard refused, nothing was computed",
              "admission_refused": "the runner's admission refused a phase",
              "unexplained_files": "a phase has files that no attempt of this runner wrote",
              "never_fits": "a phase can never fit in memory",
              "stuck_plan": "no phase can start and the plan is not finished"}
HALT_ROW = "runner_halt"

# ------------------------------------------------------------------------------------------------ the runner
PHASE_KINDS = ("data_main", "critic", "critic_full", "critic_second", "switch", "clone", "reference_actors", "policy",
               "lock", "evaluate", "look_confirmatory")
PHASE_PLAIN = {"data_main": "the main setting's data, the gate's checks, the outcome-free group count and the top-up",
               "critic": "the critic", "critic_full": "the full-log critic", "critic_second": "BCA's second critic",
               "switch": "BCA's switch and height (the pooled held-out groups)", "clone": "the cloned actor",
               "reference_actors": "the no-caution and full-log actors", "policy": "the actor training",
               "lock": "the lock", "evaluate": "the evaluation", "look_confirmatory": "the confirmatory look"}
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
# CPU-hours (DECISION.md section 4.1, PREREGISTRATION.md section 12): about 1.52 per fresh seed for step 2's arm set
# (version 8's ledger, seeds 11 and 14), plus about 2 minutes for the dose arms and 1 per seed-world for scoring the
# pooled groups; forecast 1.7 per seed.
CPU_HOURS_PER_FRESH_SEED = 1.7
FORECAST_CPU_HOURS = {"plumbing": 2.5,
                      "confirmatory_main": CPU_HOURS_PER_FRESH_SEED * len(CONFIRMATORY)}
BUDGET_STAGES = ("confirmatory_main",)     # the stage that can pause (the plumbing has no phase of the official plan)
PAUSE_FACTOR = 2.0
CAP_CPU_HOURS = 45.0
# Peak memory from the plumbing (section 13): the reproduction check runs these kinds at full length in one process,
# whose peak seeds their reservations; every other kind reserves at least SMOKE_FLOOR_GIB until an official attempt of
# it has been measured. The memory rule is version 8's (DECISION.md section 4.4, default 4).
REPRO_PEAK_KINDS = ("critic", "critic_full", "critic_second", "switch", "clone", "reference_actors", "policy")
SMOKE_FLOOR_GIB = 2.0
RUNFILES_BRANCH = "claude/near80-v9-run-files"
# The results copy (near80_runner.make_results): a record under data/ is copied under data_records/ (the repository's
# .gitignore ignores data/), and the copy refuses when git check-ignore matches any file it wrote.
RESULTS_RENAMES = {"data": "data_records"}

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
    """Why a seed is not one of this version's (None when it is): every taken block is refused in every mode (the
    pinned preview's one exception for the spent seeds 10-14 is preview/preview_v9.py's own, never this function's)."""
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
    marker, as 'name/key/...' strings (freeze.py refuses while this is not empty)."""
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
