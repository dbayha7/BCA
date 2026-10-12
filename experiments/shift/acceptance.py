"""The ThrusterChain shift test, stage by stage: the build checks, the outcome-free world card, the data, the
pre-registration and its hash check, the harm checks (H1-H4) with the stop rule, the gate report, the user's approval,
and (only after that approval) the BCA stage, detection, policy and the verdict.

Stages, in order (python -m experiments.shift.acceptance <stage> ...):
  build_checks  No learning. The environment's checks (stage0.build_checks: check_env on all 8 ids, replay of every
                row, step() against dynamics(), the proven bound, the truth anchors), the learners' checks (the plain
                run equals td3_bc.make_train_step bit for bit; even caution x1 equals plain; the actor-only step equals
                td3_bc_update's actor step; frozen critics stay frozen; the exact Q of a learned actor equals
                truth.py's; slopes against central differences in the no-stall comparison world; the attribution
                terms sum to the return difference; BCA's update traces identically to td3_bc_bca.update), the unit
                suites, the SHA-256 of the earlier gate's files (runs/glq and its pre-registration) and of the shared
                Python environment's package list before and after, and a timing smoke on random data (no return).
  stage0        The world card (stage0.run_stage0) on calibration seeds 900-904 (the retune: 905-909, stall slope 3.0);
                if a mark fails, only the pre-registered fallback grid is searched.
  data          The logs, gaps, random removals, gapped calibration banks, frozen row pools, noise maps and removed
                shares m(s), with SHA-256, for every seed list (gate 0-4, retune 5-9, confirmation 100-119 with the
                no-stall comparison world on 100-104, calibration 900-909). Checks: T1-T5 on every seed, no gap row
                in gapped training data or in the gapped bank, random removal takes the same count, the leak counter
                finds every gap row of the unmasked split.
  prereg        Writes runs/shift/preregistration.json and its .sha256 sidecar once: every mark, the Holm families,
                the verdict wording, the stop rule with the alpha ladder rule (the lead's amendment to the design), the
                retune, the seed rule, the budget, the world, the Stage-0 and data records, the SHA-256 of every file
                of experiments/shift and of every other repository file the gate imports, the dependency directory
                (its path and a SHA-256 over its files) and the software (Python, numpy, jax, jaxlib, flax, optax,
                scipy, gymnasium versions and the numeric environment flags JAX_ENABLE_X64, JAX_PLATFORMS, XLA_FLAGS,
                JAX_DEFAULT_MATMUL_PRECISION, OMP_NUM_THREADS). Only the real preset, and only in runs/shift. Refused
                when any learner output exists. Every later stage checks the sidecar, the body (it must equal what the
                hashed code writes), every hash, the dependency directory and the software, and refuses on any
                difference.
  harm_critic   H1 per seed: fixed-controller critics (stall-aware controller on the full, gapped, randomly thinned
                and placebo logs; rising-thrust controller on the full and gapped logs; the no-stall comparison world's
                full and gapped logs), errors against the exact truth on the frozen pools; then pooled marks (a)-(f).
  harm_return   H2 per seed and alpha: the critic swap (actor-only runs from one cloned actor; only the frozen critic
                differs), exact points, unrecorded-throttle shares and exact per-step attribution; pooled marks
                (a)-(e). The alpha ladder rule: alpha 2.5 first; alpha 5 only if H2 failed at 2.5, then alpha 10
                only if it failed at 5. The first alpha at which H1 and H2 both pass is used for H3, H4 and every
                later arm.
  room          H3 per seed at that alpha: actor-only arms on the frozen plain gapped TD3+BC critic; harm-placed
                caution against strength-matched even caution (matched blind and locked before any return), the
                amount-matched and shuffled versions; a situation is room-certified when it wins by 3 points.
  power         H4: the spread of two paired differences of co-trained runs gives the number of confirmation seeds.
  report        The gate report (stop report): every value above, the shift cards, the stop reason in plain words,
                and whether the BCA stage may be considered. Its SHA-256 is what the user approves.
  approve       The user, at a terminal, types the start of the report's SHA-256; the approval file holds it. No agent
                message counts as approval.
  bca, detect, policy, verdict  Only with the approval file: the BCA stage on fresh confirmation seeds, detection
                (D1-D3, one Holm family), policy (level 1 and level 2 Holm families, P2-P4) and the verdict.

Stop rule (pre-registered): H1 fails, or H2 fails at every alpha of the ladder -> the one pre-registered retune
(stall slope 3.0; Stage 0 on calibration seeds 905-909; the gate on fresh seeds 5-9, the same alpha ladder). If that
fails too: stop, 'this world does not produce critic-caused harm from the missing common command'; BCA is never run.
No situation room-certified -> stop (the cap-2 result is shown). N > 20 -> stop as underpowered. Every pooled mark
needs every seed; a failed seed fails the mark and is never dropped or replaced.

Outputs: runs/shift/pre/ (before the pre-registration), runs/shift/preregistration.json, and
runs/shift/<first 16 hex of the pre-registration's SHA-256>/<stage>/... after it. Every JSON is strict, with a .sha256
sidecar; per-seed stages are resumable. Nothing is ever written under runs/glq.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import experiments.shift as PKG  # noqa: E402
from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import detection as DT  # noqa: E402
from experiments.shift import learners as L  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402
from experiments.shift import stage0 as S0  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.shift import truth as TR  # noqa: E402

SCHEMA = "bca-shift-acceptance-v1"
RUNS_DEFAULT = ROOT / "runs" / "shift"
PREREG_NAME = "preregistration.json"
APPROVAL_NAME = "approval.json"
SHIFT_DIR = "experiments/shift"
NEW_FILES = tuple(f"{SHIFT_DIR}/{name}" for name in (
    "__init__.py", "install_deps.sh", "requirements-shift.txt", "thruster_chain.py", "recorder.py", "dataset.py",
    "shift.py", "truth.py", "stage0.py", "learners.py", "detection.py", "acceptance.py", "SHIFT_ENV.md",
    "test_thruster_chain.py", "test_dataset.py", "test_shift.py", "test_truth.py", "test_learners.py",
    "test_acceptance.py"))
REUSED_FILES = (
    "experiments/signal/gapped_lq.py", "experiments/signal/lq_harness.py", "experiments/signal/glq_match.py",
    "experiments/truth/gap.py", "algorithms/td3_bc.py", "algorithms/td3_bc_bca.py", "calibration/advantage.py",
    "calibration/bank.py", "calibration/dose.py", "calibration/iql_actors.py", "calibration/iql_imports.py",
    "calibration/iql_loss.py", "calibration/iql_network.py", "calibration/iql_reference.py", "calibration/iql_scale.py",
    "calibration/iql_state.py", "calibration/iql_targets.py", "calibration/network.py", "calibration/reference.py",
    "calibration/wbcp.py", "runtime/networks.py", "runtime/rewards.py", "runtime/published.py",
    "configs/td3_bc.yaml", "configs/experiment.yaml",
    # the rest of the import closure: package files and modules that the reused ones import
    "algorithms/__init__.py", "calibration/__init__.py", "runtime/__init__.py", "experiments/signal/__init__.py",
    "experiments/truth/__init__.py", "experiments/truth/labeller.py", "experiments/truth/pools.py",
    "experiments/truth/state_restore.py")
SOFTWARE_PACKAGES = ("numpy", "jax", "jaxlib", "flax", "optax", "scipy", "h5py", "gymnasium")
SOFTWARE_FLAGS = ("JAX_ENABLE_X64", "JAX_PLATFORMS", "XLA_FLAGS", "JAX_DEFAULT_MATMUL_PRECISION", "OMP_NUM_THREADS")
EARLIER_GATE_FILES = ("experiments/signal/glq_preregistration.json", "experiments/signal/glq_preregistration.json.sha256",
                      "experiments/signal/glq_acceptance.py", "experiments/signal/glq_train.py",
                      "experiments/signal/glq_match.py", "experiments/signal/gapped_lq.py")
UNIT_SUITES = ("experiments.shift.test_thruster_chain", "experiments.shift.test_dataset", "experiments.shift.test_shift",
               "experiments.shift.test_truth", "experiments.shift.test_learners", "experiments.shift.test_acceptance")
VARIANTS = ("base", "retune")
ROLES = ("gate", "confirmation", "calibration")
SITUATIONS = TC.SITUATIONS
SHIFTED = TC.SHIFTED_SITUATIONS
LABEL = TC.SITUATION_LABELS
TAG_BOOT, TAG_CARD_TIE = 51, 52


# ------------------------------------------------------------------------------------------------ settings

@dataclass(frozen=True)
class GateConfig:
    """Every size and seed list of the gate. preset 'real' is the pre-registered one; 'smoke' is a tiny version for
    tests and for checking the code in a temporary directory (its numbers mean nothing; its 40-episode logs report
    marks T1-T5 without requiring them, since such small logs miss them by chance)."""

    preset: str = "real"
    sizes: tuple = DS.SIZES
    gate_seeds: tuple = (0, 1, 2, 3, 4)
    retune_gate_seeds: tuple = (5, 6, 7, 8, 9)
    confirmation_seeds: tuple = tuple(range(100, 120))
    no_stall_confirmation_seeds: tuple = (100, 101, 102, 103, 104)
    calibration_seeds: tuple = S0.CALIBRATION_SEEDS
    retune_calibration_seeds: tuple = S0.RETUNE_SEEDS
    stall_slope: float = 1.5
    retune_stall_slope: float = S0.RETUNE_STALL_SLOPE
    alpha_ladder: tuple = (2.5, 5.0, 10.0)
    bank_size: int = TC.BANK_SIZE
    pool_size: int = SH.POOL_SIZE
    ms_draws: int = SH.MS_DRAWS
    noise_map_draws: int = 16
    noise_map_horizon: int = TR.NOISE_MAP_HORIZON
    truth_horizon: int = TR.HORIZON
    bootstrap_replicates: int = 1000
    attribution_starts: int = TC.BANK_SIZE
    attribution_starts_other: int = 200
    n_shuffled: int = 2
    card_ms_states: int = 2000
    card_knn_states: int = 5000
    seed_share: float = 0.8
    max_confirmation_seeds: int = 20
    require_world_marks: bool = True
    learn: L.LearnSettings = field(default_factory=L.LearnSettings)

    def gate(self, variant):
        return self.gate_seeds if variant == "base" else self.retune_gate_seeds

    def calibration(self, variant):
        return self.calibration_seeds if variant == "base" else self.retune_calibration_seeds

    def slope(self, variant):
        return self.stall_slope if variant == "base" else self.retune_stall_slope

    def seeds(self, variant, role, world):
        if role == "gate":
            return self.gate(variant)
        if role == "confirmation":
            return self.confirmation_seeds if world == "stall" else self.no_stall_confirmation_seeds
        if role == "calibration":
            return self.calibration(variant) if world == "stall" else ()
        raise ValueError(f"role must be one of {ROLES}")


SMOKE_LEARN = replace(L.LearnSettings(), updates=200, chunk=100, clone_steps=100, clone_arm_steps=100,
                      actor_only_steps=100, actor_only_chunk=50, oracle_refresh=50, oracle_horizon=60,
                      strength_states=200, band_fit_steps=50, bca_warmup=100, bca_refresh=50, row_chunk=1024)
PRESETS = {
    "real": GateConfig(),
    "smoke": GateConfig(preset="smoke", sizes=(40, 10, 10), gate_seeds=(0, 1), retune_gate_seeds=(5,),
                        confirmation_seeds=(100,), no_stall_confirmation_seeds=(100,), calibration_seeds=(900,),
                        retune_calibration_seeds=(905,), bank_size=40, pool_size=60, ms_draws=64, noise_map_draws=4,
                        noise_map_horizon=60, truth_horizon=200, bootstrap_replicates=20, attribution_starts=10,
                        attribution_starts_other=5,
                        card_ms_states=50, card_knn_states=100, require_world_marks=False, learn=SMOKE_LEARN),
}


def config_dict(cfg):
    d = asdict(cfg)
    d["learn"] = cfg.learn.as_dict()
    return d


def config_from_dict(d):
    d = dict(d)
    learn = L.LearnSettings(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.pop("learn").items()})
    d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
    return GateConfig(**d, learn=learn)


# ------------------------------------------------------------------------------------------------ marks and wording

PASS_MARKS = {
    "S0": dict(check_env_warnings_max=0, independent_dynamics_max=1e-12, closed_form_relative_max=1e-10,
               bellman_max=1e-9, truncation_max=1e-6, slope_max=1e-6, attribution_sum_max=1e-8,
               exact_q_learned_actor_max=1e-12, unit_test_failures_max=0, budget_restate_factor=1.5),
    "S1": dict(S0.MARKS, fallback_grid="full-throttle share target {0.12, 0.13} x wobble {0.3, 0.35} x "
                                       "normal-driving damping {0.25, 0.35}, default first; nothing else may be "
                                       "searched"),
    "S2": dict(gap_rows_in_gapped_training_max=0, gap_rows_in_gapped_bank_max=0, random_same_count=True,
               leak_counter_finds_every_gap_row=True, t1_t5_every_seed=True),
    "H1": dict(ratio_pooled_min=2.0, ratio_lower_min=1.5, ratio_every_seed_min=1.5, band=1.5,
               hidden_physics_min=1.5, pull_price_share_min=0.25, pull_seed_share=0.8,
               d4rl_far_band_reported=1.3),
    "H2": dict(situation="strong_current", harm_min=5.0, harm_lower_gt=0.0, harm_seed_share=0.8,
               specific_min=3.0, specific_lower_gt=0.0, physics_min=3.0, physics_lower_gt=0.0,
               mechanism_ratio_min=1.5, mechanism_seed_share=0.8, loss_rate_ratio_min=2.0),
    "H3": dict(margin_min=3.0, margin_lower_gt=0.0, seed_share=0.8, recorded_min=-1.0, certified_min=1),
    "H4": dict(z_alpha=2.128, z_power=0.842, effect=3.0, inflation=1.25, n_min=10, n_max=20),
    "B1": dict(top_share=0.10, d1_auc_mean_min=0.60, d1_lower_gt=0.50, d2_ratio_min=1.05, d2_log_lower_gt=0.0,
               d3_mean_gt=0.0, d3_lower_gt=0.0, holm_alpha=0.05),
    "B2": dict(level1_min=2.0, holm_alpha=0.05, p2_lower_gt=-2.0, p3_min=5.0, p3_lower_gt=0.0, p4_lower_gt=-2.0,
               harm_present_min=3.0, noise_map_within=2.0, cost_mean_max=-2.0, inconclusive_half_width_max=5.0),
}
MARK_TEXT = {
    "H1": "The critic is wrong where data are missing: with everything else equal, the critic trained on the gapped "
          "log is at least twice as wrong (median advantage error) at the removed commands as the critic trained on "
          "the full log; no more wrong far from the gap; random removal and another initialisation do not explain "
          "it; the stall makes it at least 1.5 times worse than in the no-stall comparison world; and the gapped "
          "critic prefers a pull past 80% by at least 25% of its true price more than the full critic does.",
    "H2": "Acting on that error costs return: from one cloned actor, the actor trained against the gapped critic "
          "scores at least 5 points below the one trained against the full critic in strong current, more than "
          "random removal and the no-stall comparison world explain, it pushes past 80% at least 1.5 times as often, "
          "and the exact step-by-step accounting puts the loss at its past-80% steps: the loss rate there is positive "
          "and at least twice the loss rate at its other steps (when the other steps show no loss at all, a rate of "
          "zero or a gain, any positive loss rate at the past-80% steps meets this).",
    "H3": "Room: caution placed by the exact harm, inside BCA's dose range [1, 1.5), beats the same strength of even "
          "caution, the same amount of caution and the same doses shuffled, each by at least 3 points.",
    "H4": "Power: the number of confirmation seeds N from the spread of real co-trained runs must be at most 20.",
}
MARK_LABELS = {
    "H1": dict(a="the gapped critic is at least twice as wrong at the removed commands (and 1.5 times in every seed)",
               b="it is no more wrong far from the gap (ratio within 1/1.5 to 1.5)",
               c="random removal of the same number of rows does not explain it",
               d="another initialisation does not explain it, on every pool",
               e="the stall makes it at least 1.5 times worse than in the no-stall comparison world",
               f="the gapped critic prefers a pull past 80% more than the full critic, in at least 4 of 5 seeds"),
    "H2": dict(a="the gapped critic's actor loses at least 5 points in strong current",
               b="at least 3 points more than random removal explains",
               c="at least 3 points more than in the no-stall comparison world",
               d="it pushes past 80% at least 1.5 times as often",
               e="the loss sits at its past-80% steps (a positive loss rate there, at least twice the rate of its "
                 "other steps; met by any positive rate when the other steps lose nothing)"),
}
HOLM_FAMILIES = {
    "about": "each family is tested under Holm's correction, a correction for testing several things at once",
    "detection": "D1, D2 and D3 together (one-sided, 0.05)",
    "policy_level_1": "BCA minus plain TD3+BC on each room-certified shifted situation (one-sided, 0.05)",
    "policy_level_2": "on each situation that passed level 1: BCA minus flattened, BCA minus shuffled and BCA minus "
                      "strength-matched even caution, all tests together (one-sided, 0.05)",
}
VERDICTS = {
    "V1": "BCA finds the harm, and its caution, placed where it finds it, improves returns out of distribution while "
          "still improving on the recording",
    "V1_noise_rider": "placing the same doses by target noise alone does about as well, so the gain does not show "
                      "detection of critic error",
    "V2": "BCA helps out of distribution, but not because of where it places caution",
    "V7": "BCA helps out of distribution and where it places its caution matters, but its doses do not follow the "
          "exact critic error, so the gain is not shown to come from finding the harm",
    "V7_recording_rider": "it also does not improve on the recording by the required 5 points",
    "V8": "BCA finds the harm and its caution, placed where it finds it, improves returns out of distribution, but it "
          "does not improve on the recording by the required 5 points",
    "V3": "BCA finds the harm, but its caution is too weak to change returns",
    "V4": "No detectable benefit",
    "V5": "BCA costs performance",
    "V6": "Inconclusive",
}
MARK_NAMES = dict(
    detection="the detection marks (D1: BCA's dose ranks the most over-preferred rows first; D2: it puts more caution "
              "in the full-throttle region on the gapped log than on the full log; D3: it follows the exact "
              "over-preference beyond target noise and the region)",
    level1="level 1 (BCA beats plain TD3+BC by at least 2 points on a room-certified shifted situation)",
    level2="level 2 (on that situation BCA also beats its own doses spread evenly, its doses shuffled, and even "
           "caution of the same strength)",
    P2="no loss in recorded conditions (P2)", P3="still improving on the recording by 5 points (P3)",
    P4="no harm without the shift, on the full log (P4)")
VERDICT_ORDER = (
    "The verdict is chosen by the first rule that applies, in this order. "
    "'BCA costs performance' (V5): no loss in recorded conditions (P2) or no harm without the shift (P4) fails, or "
    "BCA minus plain TD3+BC is at most -2 points with a 95% upper bound below 0 on a room-certified situation. "
    "'Inconclusive' (V6): the plain run loses less than 3 points to unrecorded throttle on the confirmation seeds, or "
    "the 95% interval of a level-1 difference is wider than 5 points either side. "
    "'BCA finds the harm, and its caution ... improves returns out of distribution while still improving on the "
    "recording' (V1): the detection marks pass, levels 1 and 2 pass on at least one situation, and P2, P3 and P4 pass. "
    "'BCA helps out of distribution and where it places its caution matters, but its doses do not follow the exact "
    "critic error' (V7): levels 1 and 2 pass on at least one situation but the detection marks fail (with the added "
    "sentence 'it also does not improve on the recording by the required 5 points' when P3 fails too). "
    "'BCA finds the harm ... but it does not improve on the recording by the required 5 points' (V8): levels 1 and 2 "
    "pass on at least one situation and the detection marks pass, but P3 fails. "
    "'BCA helps out of distribution, but not because of where it places caution' (V2): level 1 passes somewhere and "
    "level 2 fails on every situation where it passed. "
    "'BCA finds the harm, but its caution is too weak to change returns' (V3): the detection marks pass and level 1 "
    "fails everywhere. 'No detectable benefit' (V4): otherwise. The sentence on target noise is added to V1 and V8 "
    "when placing the same doses by the noise map scores within 2 points of BCA, or better.")
STOP_RULE = {
    "harm": "H1 fails, or H2 fails at every alpha of the ladder: run the one pre-registered retune (stall slope 3.0, "
            "Stage 0 on calibration seeds 905-909, the gate on fresh seeds 5-9 with the same alpha ladder). If it "
            "fails again, stop and report 'this world does not produce critic-caused harm from the missing common "
            "command'. BCA is never run.",
    "room": "No shifted situation room-certified: stop and report, with the cap-2 result, so the user can see "
            "whether BCA's dose range is the limit.",
    "power": "N > 20: stop as underpowered and report the smallest effect 20 seeds can detect.",
    "seeds": "Every pooled mark needs every seed. A failed seed fails the mark and is never dropped or replaced.",
}
ALPHA_RULE = ("The alpha ladder rule (an amendment to the design, fixed before any outcome). H2 runs first at TD3+BC "
              "alpha 2.5. If H2 fails at 2.5, H2 is rerun at "
              "alpha 5, and if it fails there, at alpha 10. H1 uses fixed-controller critics and does not depend on "
              "alpha. The first alpha at which H1 and H2 both pass is used for H3, H4 and every later arm, the same "
              "for every arm. The choice never looks at any BCA output. Only if no alpha passes does the stall-slope "
              "3.0 retune run, with the same ladder.")
STOP_MESSAGES = {
    "no_harm": "This world does not produce critic-caused harm from the missing common command.",
    "no_room": "No shifted situation is room-certified: caution placed by the exact harm, inside BCA's dose range, "
               "does not beat even caution there, so no BCA result could be informative.",
    "underpowered": "Underpowered: more than 20 confirmation seeds would be needed.",
    "ready": "The gate passed. The BCA stage may be run on the room-certified situations if the user approves.",
}
BUDGET = dict(minutes=dict(plain_host=8, fixed_controller_critic=6, bca_family_run=12, behaviour_cloning=1.5),
              before_stop_process_hours=13, retune_extra_process_hours="12 (25 worst case)",
              bca_stage_minutes_per_seed=115, bca_stage_process_hours={"N=10": 21, "N=20": 40})


# ------------------------------------------------------------------------------------------------ JSON and paths

def plain(x):
    """numpy -> Python, recursively; nonfinite floats -> None (strict JSON)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def canonical(payload):
    return json.dumps(plain(payload), indent=1, sort_keys=True, allow_nan=False)


def write_json(path, payload):
    """Strict JSON (sorted keys) and a '<name>.sha256' sidecar; returns the SHA-256 of the text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical(payload)
    path.write_text(text, encoding="utf8")
    digest = hashlib.sha256(text.encode("utf8")).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf8")
    return digest


def read_json(path, verify=True):
    """A JSON written by write_json; None when missing or (with verify) when its sidecar does not match."""
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf8")
    if verify:
        side = path.with_name(path.name + ".sha256")
        if not side.exists() or side.read_text(encoding="utf8").split()[0] != hashlib.sha256(
                text.encode("utf8")).hexdigest():
            return None
    return json.loads(text)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refuse_protected(path):
    """Refuse any output under runs/glq (the earlier gate) or inside the repository's code directories."""
    p = Path(path).resolve()
    for protected in (ROOT / "runs" / "glq", ROOT / "experiments", ROOT / "algorithms", ROOT / "calibration",
                      ROOT / "runtime", ROOT / "configs"):
        if p == protected or protected in p.parents:
            raise SystemExit(f"refused: {p} is inside {protected}; outputs go to runs/shift or a temporary directory")
    return p


class GateError(RuntimeError):
    """A stage refuses: an earlier stage is missing or failed, or the stop rule forbids it."""


class PreregError(GateError):
    """The pre-registration, its sidecar or a hashed file differs."""


def pre_dir(runs):
    return Path(runs) / "pre"


def data_dir(runs, variant, world, rep):
    return pre_dir(runs) / "data" / variant / world / f"seed{rep}"


# ------------------------------------------------------------------------------------------------ hashes

def code_hashes(files, code_root=ROOT):
    return {rel: file_sha256(Path(code_root) / rel) for rel in files}


def unhashed_files(code_root=ROOT):
    """Files in experiments/shift that the pre-registration would not hash (it must hash every one)."""
    here = Path(code_root) / SHIFT_DIR
    names = {f"{SHIFT_DIR}/{p.name}" for p in here.iterdir() if p.is_file() and not p.name.endswith((".pyc", ".pyo"))}
    return sorted(names - set(NEW_FILES)), sorted(set(NEW_FILES) - names)


def earlier_gate_state():
    """SHA-256 of the earlier gapped linear-quadratic gate's files (its Amendment A1 run) and a listing of runs/glq
    (names, sizes, times)."""
    files = {rel: file_sha256(ROOT / rel) if (ROOT / rel).exists() else None for rel in EARLIER_GATE_FILES}
    listing = []
    glq = ROOT / "runs" / "glq"
    if glq.exists():
        for p in sorted(glq.rglob("*")):
            if p.is_file():
                st = p.stat()
                listing.append(f"{p.relative_to(glq)}|{st.st_size}|{st.st_mtime_ns}")
    return dict(files=files, runs_glq_listing_sha256=hashlib.sha256("\n".join(listing).encode()).hexdigest(),
                runs_glq_files=len(listing))


def environment_state():
    pk = PKG.environment_packages()
    return dict(packages=len(pk), sha256=hashlib.sha256("\n".join(pk).encode()).hexdigest())


def software_state():
    """What can change the numbers outside the hashed files: the Python, numpy, jax, jaxlib, flax, optax, scipy, h5py
    and gymnasium versions, the numeric environment flags (JAX_ENABLE_X64, JAX_PLATFORMS, XLA_FLAGS,
    JAX_DEFAULT_MATMUL_PRECISION, OMP_NUM_THREADS; None when unset), whether jax's float64 mode is on, and jax's
    backend. The pre-registration records it; every later stage refuses when it differs."""
    import importlib.metadata as metadata

    versions = {}
    for name in SOFTWARE_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            module = sys.modules.get(name)
            versions[name] = getattr(module, "__version__", None)
    return dict(python=sys.version.split()[0], packages=versions,
                flags={k: os.environ.get(k) for k in SOFTWARE_FLAGS},
                jax_float64_enabled=bool(jax.config.jax_enable_x64), jax_backend=jax.default_backend())


def unhashed_imports(code_root=ROOT):
    """Repository files of the modules imported in this process that the pre-registration does not hash (test modules
    left out). Must be empty when a stage runs: every file the gate executes is hashed."""
    root = Path(code_root).resolve()
    hashed = set(NEW_FILES) | set(REUSED_FILES)
    out = set()
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path or name.rsplit(".", 1)[-1].startswith("test_"):
            continue
        path = Path(path).resolve()
        if root in path.parents:
            rel = path.relative_to(root).as_posix()
            if rel not in hashed:
                out.add(rel)
    return sorted(out)


def check_software(payload):
    """PreregError unless the dependency directory and the software equal what the pre-registration recorded."""
    deps = PKG.dependency_record()
    if deps != payload.get("dependency_directory"):
        raise PreregError(f"the dependency directory differs from the pre-registration: {deps} != "
                          f"{payload.get('dependency_directory')}")
    soft = software_state()
    if soft != payload.get("software"):
        diff = sorted(k for k in set(soft) | set(payload.get("software") or {})
                      if soft.get(k) != (payload.get("software") or {}).get(k))
        raise PreregError(f"the software or numeric flags differ from the pre-registration in: {', '.join(diff)} "
                          f"(now {dict((k, soft.get(k)) for k in diff)})")


# ------------------------------------------------------------------------------------------------ the world card

def stage0_path(runs, variant):
    return pre_dir(runs) / f"stage0_{variant}.json"


def stage0_spec(runs, variant):
    """The recording controller's numbers chosen by the stage0 stage of a variant (the default, or the fallback grid's
    choice); GateError when stage 0 has not run or failed."""
    rec = read_json(stage0_path(runs, variant))
    if rec is None:
        raise GateError(f"run the stage0 stage for the {variant} world first (no verified {stage0_path(runs, variant)})")
    if not rec.get("passed") or rec.get("chosen") is None:
        raise GateError(f"stage 0 of the {variant} world failed: no setting of the fallback grid passes every mark")
    return RC.RecorderSpec(**rec["chosen"])


def stage_stage0(runs, cfg, variant="base", grid=True, log=print):
    """S1: stage0.run_stage0 on the variant's calibration seeds; if a mark fails and grid, the pre-registered
    fallback grid (stage0.search_fallback) and nothing else."""
    started = time.perf_counter()
    seeds, slope = cfg.calibration(variant), cfg.slope(variant)
    record = S0.run_stage0(tuple(seeds), stall_slope=slope, log=log)
    chosen = RC.RecorderSpec(**record["recorder"]) if record["passed"] else None
    grid_records = []
    if chosen is None and grid:
        log("a Stage-0 mark failed: searching the pre-registered fallback grid only")
        chosen, grid_records = S0.search_fallback(tuple(seeds), stall_slope=slope, log=log)
    out = dict(kind="shift-stage0", about="the outcome-free world card: marks T1-T9 on calibration seeds, no learning",
               variant=variant, seeds=list(seeds), stall_slope=slope, record=record, fallback_records=grid_records,
               chosen=None if chosen is None else chosen.as_dict(), passed=chosen is not None,
               seconds=time.perf_counter() - started)
    write_json(stage0_path(runs, variant), out)
    return out


def world_marks(log, gap, spec, physics):
    """T1-T5 of one stalling-world log and its gap, by the same formulas as stage0.world_card (which generates its own
    1,000-episode log; this one takes the log it is given)."""
    train = log.split_rows("train")
    a = np.asarray(log["actions"][train], np.float64)
    x = log["infos/state"][train]
    press = log.diagnostic("infos/full_throttle_command")[train]
    recorder = RC.Recorder(spec, physics)
    region = recorder.in_region(x)
    ing = gap.contains(a)
    past = TC.past_knee(a, physics)
    rnd = SH.make_removal(log, "random", gap).removed[train]
    same = np.all(TC.dynamics(x[~ing], a[~ing], physics)[0] == TC.dynamics(x[~ing], a[~ing], TC.NO_STALL)[0], axis=1)
    card = dict(region_share=float(region.mean()), full_throttle_share=float(press.mean()),
                full_throttle_commands_removed=float(ing[press].mean()) if press.any() else 0.0,
                removed_rows_full_throttle_share=float(press[ing].mean()) if ing.any() else 0.0,
                region_rows_kept=float(np.mean(~ing[region])) if region.any() else 0.0,
                kept_past80_rows=int(past[~ing].sum()), kept_past80_share=float(past[~ing].mean()),
                full_log_past80_share=float(past.mean()),
                random_keeps_past80=float(past[~rnd].sum() / max(1, past.sum())),
                identical_rule_share=float(same.mean()), centre_thruster1=float(gap.centre[0]),
                competing_ratio=float(gap.competing_ratio))
    m = S0.MARKS
    marks = dict(T1=card["full_throttle_commands_removed"] >= m["T1_full_throttle_removed_min"]
                 and card["centre_thruster1"] <= m["T1_centre_thruster1_max"],
                 T2=card["kept_past80_share"] <= m["T2_kept_past80_share_max"]
                 and card["random_keeps_past80"] >= m["T2_random_keeps_past80_min"],
                 T3=card["competing_ratio"] >= m["T3_competing_ratio_min"],
                 T4=card["identical_rule_share"] >= m["T4_identical_rule_min"],
                 T5=card["region_rows_kept"] >= m["T5_region_kept_min"])
    return card, marks


# ------------------------------------------------------------------------------------------------ data (S2)

def noise_map(states, actions, physics, draws, horizon, seed, chunk=8192):
    """truth.noise_map (the SD of TD3's smoothed target under a perfect critic of the stall-aware controller) with
    the float64 JAX rollout truth.q_value_jax in place of the numpy one: the same numbers to rounding (tested), many
    times faster on 50,000 rows."""
    policy = RC.stall_aware(physics)
    x1, _ = TC.dynamics(np.asarray(states, np.float64), np.asarray(actions, np.float64), physics)
    base = policy(x1)
    eps = np.clip(TR.TD3_NOISE * np.random.default_rng(list(seed)).standard_normal((draws,) + base.shape),
                  -TR.TD3_CLIP, TR.TD3_CLIP)
    tq = np.stack([TR.q_value_jax(policy, x1, np.clip(base + eps[i], -1.0, 1.0), physics, horizon, chunk=chunk)
                   for i in range(draws)])
    return physics.gamma * tq.std(0)


def data_seed(runs, cfg, variant, role, world, rep, spec, force=False, log=print):
    """One seed's log, gap, removals, checks and (by role) pools, noise map and m(s); record.json + arrays.npz."""
    out = data_dir(runs, variant, world, rep)
    old = read_json(out / "record.json")
    if old is not None and not force and old.get("role") == role and old.get("config_preset") == cfg.preset:
        return old
    started = time.perf_counter()
    slope = cfg.slope(variant)
    physics = TC.world_physics(world, slope)
    lg = DS.generate(rep, world, spec, cfg.sizes, slope)
    gap, grec = SH.define_gap(lg)
    rec = dict(kind="shift-data", about="one recorded log, its removal rule and the frozen rows the harm checks use",
               variant=variant, role=role, world=world, rep=int(rep), config_preset=cfg.preset, sizes=list(cfg.sizes),
               stall_slope=slope, recorder=spec.as_dict(), physics=physics.as_dict(), log_sha256=lg.array_hashes(),
               bank_rows=int(len(lg.bank_rows)), gap_record_sha256=SH.GAP.canonical_sha256(grec))
    if gap is None:
        rec.update(defined=False, reason=grec.get("reason"), passed=False, seconds=time.perf_counter() - started)
        write_json(out / "record.json", rec)
        log(f"data {variant}/{world}/seed {rep}: the gap is not defined ({grec.get('reason')})")
        return rec
    rg, rr = SH.make_removal(lg, "gap", gap), SH.make_removal(lg, "random", gap)
    train, bank = lg.split_rows("train"), lg.bank_rows
    checks = dict(gap_rows_in_gapped_training=SH.leak_count(rg.kept_train_rows(lg), lg, gap),
                  gap_rows_in_gapped_bank=SH.leak_count(rg.bank_rows(lg), lg, gap),
                  random_same_count=bool(rr.count == rg.count and rr.bank_removed.sum() == rg.bank_removed.sum()),
                  leak_counter_finds_every_training_gap_row=bool(SH.leak_count(train, lg, gap) == rg.count),
                  leak_counter_finds_every_bank_gap_row=bool(SH.leak_count(bank, lg, gap) == int(rg.bank_removed.sum())),
                  evaluation_split_untouched=bool(not rg.removed[lg.split_rows("evaluation")].any()))
    passed = (checks["gap_rows_in_gapped_training"] == 0 and checks["gap_rows_in_gapped_bank"] == 0
              and checks["random_same_count"] and checks["leak_counter_finds_every_training_gap_row"]
              and checks["leak_counter_finds_every_bank_gap_row"] and checks["evaluation_split_untouched"])
    rec.update(defined=True, gap=gap.as_dict(), removals={k: r.summary(lg) for k, r in (("gap", rg), ("random", rr))},
               checks=checks)
    if world == "stall":
        card, marks = world_marks(lg, gap, spec, physics)
        rec.update(world_card=card, t1_t5=marks, t1_t5_required=cfg.require_world_marks)
        passed = passed and (all(marks.values()) or not cfg.require_world_marks)
    recorder = RC.Recorder(spec, physics)
    arrays = {}
    if role == "gate":
        pools = SH.make_pools(lg, gap, recorder, size=cfg.pool_size, ms_draws=cfg.ms_draws)
        for k in ("removed", "region", "far"):
            arrays[f"pool_{k}"] = pools[k]
        arrays["far_removed_share"] = pools["far_removed_share"]
        rec["pools"] = dict(sha256=pools["sha256"], rule=pools["rule"], sizes={k: int(len(pools[k])) for k in
                                                                             ("removed", "region", "far")},
                            c_hi=pools["c_hi"].tolist(), c_cap=pools["c_cap"].tolist())
        passed = passed and all(len(pools[k]) == min(cfg.pool_size, len(pools[k])) and len(pools[k]) > 0
                                for k in ("removed", "region", "far"))
    if role in ("gate", "confirmation") and world == "stall":
        x = lg["infos/state"][train]
        a = np.asarray(lg["actions"][train], np.float64)
        ms, ms_se = SH.removed_share(x, rg, recorder, draws=cfg.ms_draws, seed=(TC.DATA_SEED, rep, SH.TAG_MS, 1000))
        arrays["removed_share"], arrays["removed_share_se"] = ms, ms_se
        rec["maps"] = dict(removed_share=dict(mean=float(ms.mean()), max_se=float(ms_se.max())))
        if role == "confirmation":
            arrays["noise_map"] = noise_map(x, a, physics, cfg.noise_map_draws, cfg.noise_map_horizon,
                                            (TC.DATA_SEED, rep, 11))
            rec["maps"]["noise_map"] = dict(median=float(np.median(arrays["noise_map"])),
                                            p90=float(np.quantile(arrays["noise_map"], 0.9)))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "arrays.npz", **arrays)
    rec.update(arrays_sha256={k: DS.sha256_array(v) for k, v in arrays.items()}, passed=bool(passed),
               seconds=time.perf_counter() - started)
    write_json(out / "record.json", rec)
    log(f"data {variant}/{world}/seed {rep} ({role}): {'pass' if passed else 'FAIL'}, "
        f"{rg.count} gap rows removed, {time.perf_counter() - started:.0f}s")
    return rec


def stage_data(runs, cfg, variant="base", roles=ROLES, worlds=("stall", "no_stall"), seeds=None, force=False,
               log=print):
    """S2 for the variant (outcome-free; it may also run after the pre-registration, for example for the retune's
    confirmation seeds: every later stage regenerates each log and checks it against its record)."""
    spec = stage0_spec(runs, variant)
    out = {}
    for role in roles:
        for world in worlds:
            for rep in (cfg.seeds(variant, role, world) if seeds is None else
                        [s for s in seeds if s in cfg.seeds(variant, role, world)]):
                rec = data_seed(runs, cfg, variant, role, world, rep, spec, force, log)
                out[f"{variant}/{world}/seed{rep}"] = bool(rec.get("passed"))
    return out


@dataclass
class SeedData:
    """One seed's verified data: the regenerated log (checked against its record), the gap, the removals ('none',
    'gap', 'random'), the frozen pools and the maps (by role), and prepared learner data on demand."""

    rep: int
    world: str
    variant: str
    record: dict
    log: object
    gap: object
    removals: dict
    arrays: dict
    physics: object
    spec: object
    _prepared: dict = field(default_factory=dict)

    def norm(self):
        return L.normalization(self.log, self.removals["gap"])

    def prepared(self, kind, learn):
        if kind not in self._prepared:
            self._prepared[kind] = L.prepare(self.log, self.removals[kind], self.norm(), learn)
        return self._prepared[kind]

    def train_position(self, rows):
        """Positions of log rows among the training rows (where the per-training-row maps are stored)."""
        return np.searchsorted(self.log.split_rows("train"), np.asarray(rows, np.int64))


def load_seed(runs, cfg, variant, world, rep):
    """Regenerate one seed's log and verify it, its gap, removals and arrays against its data record."""
    rec = read_json(data_dir(runs, variant, world, rep) / "record.json")
    if rec is None:
        raise GateError(f"no verified data record for {variant}/{world}/seed {rep}: run the data stage")
    if not rec.get("passed"):
        raise GateError(f"the data checks failed on {variant}/{world}/seed {rep}")
    spec = RC.RecorderSpec(**rec["recorder"])
    slope = rec["stall_slope"]
    lg = DS.generate(rep, world, spec, tuple(rec["sizes"]), slope)
    if lg.array_hashes() != rec["log_sha256"]:
        raise PreregError(f"{variant}/{world}/seed {rep}: the regenerated log differs from its record")
    g = rec["gap"]
    gap = SH.Gap(centre=tuple(g["centre"]), rho=g["rho"], mass=g["mass"], rule=g["rule"], q=g["q"],
                 competing_ratio=math.inf if g["competing_ratio"] is None else g["competing_ratio"],
                 record_sha256=g["record_sha256"])
    removals = {k: SH.make_removal(lg, k, gap) for k in SH.REMOVALS}
    for k in ("gap", "random"):
        s = removals[k].summary(lg)
        if (s["kept_rows_sha256"] != rec["removals"][k]["kept_rows_sha256"]
                or s["bank_rows_sha256"] != rec["removals"][k]["bank_rows_sha256"]):
            raise PreregError(f"{variant}/{world}/seed {rep}: the {k} removal differs from its record")
    path = data_dir(runs, variant, world, rep) / "arrays.npz"
    arrays = {}
    if rec["arrays_sha256"]:
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
    if {k: DS.sha256_array(v) for k, v in arrays.items()} != rec["arrays_sha256"]:
        raise PreregError(f"{variant}/{world}/seed {rep}: arrays.npz differs from its record")
    return SeedData(int(rep), world, variant, rec, lg, gap, removals, arrays, lg.physics, spec)


# ------------------------------------------------------------------------------------------------ pre-registration

@dataclass
class Prereg:
    path: Path
    digest: str
    payload: dict
    config: GateConfig
    runs: Path
    code_root: Path

    @property
    def root(self):
        return self.runs / self.digest[:16]


PREREG_VOLATILE = ("created", "code_sha256", "dependency_sha256", "dependency_directory", "software")


def world_spec(runs, cfg, variant):
    try:
        spec = stage0_spec(runs, variant)
    except GateError:
        return None
    return dict(recorder=spec.as_dict(), stall_physics=TC.world_physics("stall", cfg.slope(variant)).as_dict(),
                no_stall_physics=TC.world_physics("no_stall", cfg.slope(variant)).as_dict())


def anchor_table(runs, cfg):
    """Per variant and world, the recording controller's and the top controller's mean return on every bank (0 and
    100 points)."""
    out = {}
    for variant in VARIANTS:
        try:
            spec = stage0_spec(runs, variant)
        except GateError:
            continue
        for world in TC.WORLDS:
            physics = TC.world_physics(world, cfg.slope(variant))
            out[f"{variant}/{world}"] = TR.anchors(physics, spec, n=cfg.bank_size)
    return out


def data_index(runs):
    """Every data record under runs/shift/pre/data: its sidecar SHA-256 and whether it passed."""
    base = pre_dir(runs) / "data"
    out = {}
    if base.exists():
        for p in sorted(base.rglob("record.json")):
            rec = read_json(p)
            key = str(p.parent.relative_to(base))
            out[key] = dict(record_sha256=file_sha256(p) if rec is not None else None,
                            passed=None if rec is None else bool(rec.get("passed")),
                            role=None if rec is None else rec.get("role"))
    return out


def required_data(cfg, runs):
    """The data records the pre-registration needs: the base world's gate, confirmation and calibration seeds; the
    retune's gate and calibration seeds when its Stage 0 passed."""
    need = []
    for variant in VARIANTS:
        if world_spec(runs, cfg, variant) is None:
            continue
        roles = ROLES if variant == "base" else ("gate", "calibration")
        for role in roles:
            for world in TC.WORLDS:
                need += [f"{variant}/{world}/seed{r}" for r in cfg.seeds(variant, role, world)]
    return need


def plan(cfg, prefix="python -m experiments.shift.acceptance"):
    """Every command of the gate in order (one process per seed where a stage is per seed)."""
    lines = [f"{prefix} build_checks", f"{prefix} stage0 --variant base", f"{prefix} stage0 --variant retune",
             f"{prefix} data --variant base", f"{prefix} data --variant retune --roles gate calibration",
             f"{prefix} prereg"]
    for variant in VARIANTS:
        note = "" if variant == "base" else "   # only if the stop rule calls the retune"
        s = cfg.gate(variant)
        lines += [f"{prefix} harm_critic --variant {variant} --seeds {r}{note}" for r in s]
        lines.append(f"{prefix} harm_critic --variant {variant} --pooled{note}")
        for alpha in cfg.alpha_ladder:
            extra = "" if alpha == cfg.alpha_ladder[0] else "   # only if H2 failed at the previous alpha"
            lines += [f"{prefix} harm_return --variant {variant} --alpha {alpha:g} --seeds {r}{note}{extra}" for r in s]
            lines.append(f"{prefix} harm_return --variant {variant} --alpha {alpha:g} --pooled{note}{extra}")
        lines += [f"{prefix} room --variant {variant} --seeds {r}{note}" for r in s]
        lines.append(f"{prefix} room --variant {variant} --pooled{note}")
        lines += [f"{prefix} power --variant {variant} --seeds {r}{note}" for r in s]
        lines.append(f"{prefix} power --variant {variant} --pooled{note}")
    lines += [f"{prefix} report --seeds <each gate seed>", f"{prefix} report",
              f"{prefix} approve   # the user, at a terminal; nothing below runs without it",
              f"{prefix} bca --seeds <each confirmation seed 100..99+N>", f"{prefix} detect --seeds <...>",
              f"{prefix} detect --pooled", f"{prefix} policy", f"{prefix} verdict"]
    return lines


def prereg_payload(cfg, runs, code_root=ROOT):
    """Everything the pre-registration fixes (module docstring); the volatile fields are informational or checked
    separately (code and dependency hashes)."""
    s0 = {v: read_json(stage0_path(runs, v)) for v in VARIANTS}
    bc = read_json(pre_dir(runs) / "build_checks.json")
    return dict(
        kind="bca-shift-preregistration", schema=SCHEMA, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        statement="Every mark, seed list, alpha rung, retune and setting of the ThrusterChain shift test, fixed "
                  "before any learner of this pre-registration trains. acceptance.load_prereg refuses every later "
                  "stage when this file, its sidecar or any hashed file differs, or when this body differs from what "
                  "the hashed code writes.",
        amendments=dict(alpha_ladder_rule=ALPHA_RULE),
        config=config_dict(cfg), pass_marks=PASS_MARKS, mark_text=MARK_TEXT, mark_labels=MARK_LABELS,
        holm_families=HOLM_FAMILIES,
        verdicts=VERDICTS, verdict_order=VERDICT_ORDER, stop_rule=STOP_RULE, stop_messages=STOP_MESSAGES,
        alpha_ladder=list(cfg.alpha_ladder), alpha_rule=ALPHA_RULE,
        retune=dict(stall_slope=cfg.retune_stall_slope, calibration_seeds=list(cfg.retune_calibration_seeds),
                    gate_seeds=list(cfg.retune_gate_seeds), rule=STOP_RULE["harm"]),
        seeds=dict(gate=list(cfg.gate_seeds), retune_gate=list(cfg.retune_gate_seeds),
                   confirmation=list(cfg.confirmation_seeds), no_stall_confirmation=list(cfg.no_stall_confirmation_seeds),
                   calibration=list(cfg.calibration_seeds), retune_calibration=list(cfg.retune_calibration_seeds),
                   rule="a mark 'in at least 4 of 5 seeds' needs ceil(0.8 n) of the n seeds; every pooled mark needs "
                        "every seed"),
        power_rule="N = max(10, ceil(((2.128 + 0.842) x 1.25 x s / 3)^2)), s the larger standard deviation of the "
                   "null pair and of "
                   "even caution x1.25 minus plain over the gate seeds, on every room-certified situation; N > 20 stops",
        budget=dict(BUDGET, measured=None if bc is None else bc.get("timing"),
                    restated=None if bc is None or not bc.get("timing") else bc["timing"].get("restate_budget")),
        bca=dict(L.BCA_SETTINGS, placements=list(L.PLACEMENTS)),
        world={v: world_spec(runs, cfg, v) for v in VARIANTS}, anchors=anchor_table(runs, cfg),
        stage0={v: None if s0[v] is None else dict(sha256=file_sha256(stage0_path(runs, v)), passed=s0[v]["passed"],
                                                   chosen=s0[v]["chosen"]) for v in VARIANTS},
        build_checks=None if bc is None else dict(sha256=file_sha256(pre_dir(runs) / "build_checks.json"),
                                                  passed=bc.get("passed")),
        data=data_index(runs), run_list=plan(cfg),
        code_sha256=code_hashes(NEW_FILES, code_root), dependency_sha256=code_hashes(REUSED_FILES, code_root),
        dependency_directory=PKG.dependency_record(), software=software_state())


def learner_outputs(runs):
    """Files under runs/shift other than the pre-registration stages (any learner output)."""
    runs = Path(runs)
    if not runs.exists():
        return []
    return sorted(str(p) for d in runs.iterdir() if d.is_dir() and d.name != "pre" for p in d.rglob("*") if p.is_file())


def write_prereg(runs=RUNS_DEFAULT, cfg=None, code_root=ROOT, log=print):
    """S3: write the pre-registration once. Refused when it exists, when any learner output exists, when an
    experiments/shift file would go unhashed, or when a required earlier stage is missing or failed."""
    cfg = PRESETS["real"] if cfg is None else cfg
    runs = Path(runs)
    path = runs / PREREG_NAME
    if cfg.preset == "real" and runs.resolve() != RUNS_DEFAULT.resolve():
        raise GateError(f"refused: the real preset is pre-registered only in {RUNS_DEFAULT} (one official "
                        "pre-registration; the smoke preset is for checking the code in a temporary directory)")
    if path.exists():
        raise GateError(f"{path} exists: the pre-registration is written once")
    outputs = learner_outputs(runs)
    if outputs:
        raise GateError(f"refused: learner outputs already exist ({outputs[0]} ...); nothing may be tuned after "
                        "outcomes are seen")
    extra, missing = unhashed_files(code_root)
    if extra or missing:
        raise GateError(f"refused: experiments/shift holds files outside the hashed list {extra} or misses {missing}")
    loose = unhashed_imports(code_root)
    if loose:
        raise GateError(f"refused: imported repository files outside the hashed lists: {loose}")
    if PKG.dependency_record()["unexpected_entries"]:
        raise GateError("refused: the dependency directory holds unexpected entries")
    bc = read_json(pre_dir(runs) / "build_checks.json")
    if bc is None or not bc.get("passed"):
        raise GateError("refused: the build checks have not passed (run build_checks)")
    stage0_spec(runs, "base")
    if read_json(stage0_path(runs, "retune")) is None:
        raise GateError("refused: run stage0 --variant retune first (the retune is pre-registered with the rest)")
    have = data_index(runs)
    bad = [k for k in required_data(cfg, runs) if not (have.get(k) or {}).get("passed")]
    if bad:
        raise GateError(f"refused: data records missing or failed: {', '.join(bad[:8])}{' ...' if len(bad) > 8 else ''}")
    payload = prereg_payload(cfg, runs, code_root)
    text = canonical(payload)
    runs.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf8")
    digest = hashlib.sha256(text.encode("utf8")).hexdigest()
    path.with_name(PREREG_NAME + ".sha256").write_text(f"{digest}  {PREREG_NAME}\n", encoding="utf8")
    log(f"wrote {path} (sha256 {digest}); outputs go to {runs / digest[:16]}")
    return digest


def load_prereg(runs=RUNS_DEFAULT, code_root=ROOT):
    """Verify the pre-registration (sidecar, every hashed file, the body against what the hashed code writes) and
    return it; PreregError on any difference."""
    runs = Path(runs)
    path = runs / PREREG_NAME
    if not path.exists():
        raise PreregError(f"no pre-registration at {path}: run the prereg stage first")
    text = path.read_text(encoding="utf8")
    digest = hashlib.sha256(text.encode("utf8")).hexdigest()
    side = path.with_name(PREREG_NAME + ".sha256")
    if not side.exists() or side.read_text(encoding="utf8").split()[0] != digest:
        raise PreregError(f"{path}: its SHA-256 {digest} differs from its sidecar (the file was changed)")
    payload = json.loads(text)
    extra, missing = unhashed_files(code_root)
    if extra or missing:
        raise PreregError(f"experiments/shift holds unhashed files {extra} or misses {missing}")
    if set(payload.get("code_sha256", {})) != set(NEW_FILES):
        raise PreregError("the hashed file list differs from the code's list")
    now = code_hashes(NEW_FILES, code_root)
    bad = [rel for rel, h in payload["code_sha256"].items() if now[rel] != h]
    if bad:
        raise PreregError("files changed since the pre-registration: " + ", ".join(bad))
    if set(payload.get("dependency_sha256", {})) != set(REUSED_FILES):
        raise PreregError("the hashed list of reused files differs from the code's list")
    deps = code_hashes(REUSED_FILES, code_root)
    changed = [rel for rel, h in payload["dependency_sha256"].items() if deps[rel] != h]
    if changed:
        raise PreregError("reused files changed since the pre-registration: " + ", ".join(changed))
    cfg = config_from_dict(payload["config"])
    if cfg.preset not in PRESETS or cfg != PRESETS[cfg.preset]:
        raise PreregError("the pre-registered configuration is not the code's preset")
    if runs.resolve() == RUNS_DEFAULT.resolve() and cfg.preset != "real":
        raise PreregError("runs/shift takes the real preset only")
    if cfg.preset == "real" and runs.resolve() != RUNS_DEFAULT.resolve():
        raise PreregError(f"the real preset runs only from the one official pre-registration in {RUNS_DEFAULT}")
    check_software(payload)
    fresh = json.loads(canonical(prereg_payload(cfg, runs, code_root)))
    # data records written after the pre-registration (the retune's confirmation seeds) are allowed: every later
    # stage regenerates each log and checks it against its record; the pre-registered records must be unchanged
    fresh["data"] = {k: v for k, v in fresh.get("data", {}).items() if k in payload.get("data", {})}
    edited = sorted(k for k in set(fresh) | set(payload) if k not in PREREG_VOLATILE and fresh.get(k) != payload.get(k))
    if edited:
        raise PreregError("the pre-registration differs from what the hashed code and the recorded stages give in: "
                          + ", ".join(edited))
    return Prereg(path, digest, payload, cfg, runs, Path(code_root))


def guard(prereg):
    """Re-check at the start of every stage that neither the pre-registration nor a hashed file changed."""
    if file_sha256(prereg.path) != prereg.digest:
        raise PreregError(f"{prereg.path} changed since it was loaded")
    now = code_hashes(NEW_FILES, prereg.code_root)
    bad = [rel for rel, h in prereg.payload["code_sha256"].items() if now[rel] != h]
    deps = code_hashes(REUSED_FILES, prereg.code_root)
    bad += [rel for rel, h in prereg.payload["dependency_sha256"].items() if deps[rel] != h]
    if bad:
        raise PreregError("files changed since the pre-registration: " + ", ".join(bad))
    check_software(prereg.payload)


def stage_record(prereg, stage, started, **fields):
    guard(prereg)
    return dict(kind=f"shift-{stage}", stage=stage, prereg_sha256=prereg.digest, preset=prereg.config.preset,
                machine=platform.node(), software=software_state(), seconds=time.perf_counter() - started,
                finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields)


def finished(path, prereg):
    rec = read_json(path)
    return rec if rec is not None and rec.get("prereg_sha256") == prereg.digest else None


def anchors_of(prereg, variant, world):
    an = prereg.payload["anchors"].get(f"{variant}/{world}")
    if an is None:
        raise GateError(f"no anchors for {variant}/{world} in the pre-registration")
    return an


# ------------------------------------------------------------------------------------------------ learner caches

def _variant_root(prereg, stage, variant):
    return prereg.root / stage / variant


def critic_dir(prereg, variant, world, removal, controller, placebo, rep):
    name = f"{controller}_{'full' if removal == 'none' else removal}{'_placebo' if placebo else ''}"
    return _variant_root(prereg, "critics", variant) / world / name / f"seed{rep}"


def get_critic(prereg, sd, removal, controller, placebo=False, train=True, log=print):
    """A fixed-controller critic of one seed (trained once, then read back with its SHA-256 checked). Returns
    (record, critic params or None when the run failed)."""
    learn = prereg.config.learn
    d = critic_dir(prereg, sd.variant, sd.world, removal, controller, placebo, sd.rep)
    rec = finished(d / "record.json", prereg)
    if rec is None:
        if not train:
            raise GateError(f"the critic {d} has not been trained (run harm_critic for seed {sd.rep})")
        started = time.perf_counter()
        result, state = L.fixed_critic(sd.prepared(removal, learn), controller, learn, sd.rep, placebo=placebo,
                                       out_dir=d, log=log)
        rec = stage_record(prereg, "critic", started, result=result)
        write_json(d / "record.json", rec)
        return rec, None if result["failed"] else state.critic.params
    if rec["result"]["failed"]:
        return rec, None
    return rec, L.load_tree(d / "critic.msgpack", L.critic_template(), rec["result"]["file_sha256"])


def get_clone(prereg, sd, log=print):
    """The cloned actor of one seed and world (behaviour cloning on the gapped log's kept rows, 20,000 steps)."""
    learn = prereg.config.learn
    d = _variant_root(prereg, "clone", sd.variant) / sd.world / f"seed{sd.rep}"
    rec = finished(d / "record.json", prereg)
    if rec is None:
        started = time.perf_counter()
        params, result = L.clone(sd.prepared("gap", learn), learn, sd.rep, log=log)
        result["file_sha256"] = L.save_tree(d / "actor.msgpack", params)
        rec = stage_record(prereg, "clone", started, result=result)
        write_json(d / "record.json", rec)
        return params
    return L.load_tree(d / "actor.msgpack", L.actor_template(), rec["result"]["file_sha256"])


HOST_ARMS = {"plain": dict(stream=0, table=None), "plain_stream1": dict(stream=1, table=None),
             "even_x1.25": dict(stream=0, table=1.25), "even_x2": dict(stream=0, table=2.0)}


def host_dir(prereg, variant, world, removal, alpha, arm, rep, stage="hosts"):
    return _variant_root(prereg, stage, variant) / world / ("full" if removal == "none" else removal) / \
        f"alpha{alpha:g}" / arm / f"seed{rep}"


def get_host(prereg, sd, removal, alpha, arm="plain", table=None, train=True, stage="hosts", log=print):
    """A co-trained plain TD3+BC run (or even caution, or the null pair's second stream) at alpha. Returns (record,
    final AgentTrainState or None when the run failed)."""
    learn = prereg.config.learn
    d = host_dir(prereg, sd.variant, sd.world, removal, alpha, arm, sd.rep, stage)
    rec = finished(d / "record.json", prereg)
    if rec is None:
        if not train:
            raise GateError(f"the run {d} has not been trained")
        spec = HOST_ARMS.get(arm, dict(stream=0, table=None))
        prep = sd.prepared(removal, learn)
        tab = table if table is not None else spec["table"]
        tab = None if tab is None else (np.full(prep.n, float(tab)) if np.isscalar(tab) else np.asarray(tab))
        started = time.perf_counter()
        result, state, _ = L.cotrain(prep, learn, sd.rep, alpha=alpha, table=tab, stream=spec["stream"], out_dir=d,
                                     log=log, name=f"TD3+BC {arm} ({sd.world}, {removal} log)")
        rec = stage_record(prereg, "host", started, result=result, arm=arm, alpha=alpha, removal=removal)
        write_json(d / "record.json", rec)
        return rec, None if result["failed"] else state
    if rec["result"]["failed"]:
        return rec, None
    final = rec["result"]["snapshots"][-1]
    state = L.load_tree(d / "snapshot_100.msgpack", L.agent_template(learn), final["file_sha256"])
    return rec, state


# ------------------------------------------------------------------------------------------------ H1

H1_CRITICS = (("stall", "none", "stall_aware", False), ("stall", "gap", "stall_aware", False),
              ("stall", "random", "stall_aware", False), ("stall", "none", "stall_aware", True),
              ("stall", "none", "rising_thrust", False), ("stall", "gap", "rising_thrust", False),
              ("no_stall", "none", "stall_aware", False), ("no_stall", "gap", "stall_aware", False))
POOLS = ("removed", "region", "far")


def h1_name(world, removal, controller, placebo):
    return f"{world}:{controller}:{'full' if removal == 'none' else removal}{':placebo' if placebo else ''}"


def pool_inputs(sd, controller):
    """The frozen pools' inputs exactly as a critic sees them (float32 states and commands) and the fixed controller's
    own command pi(s) there: removed and far rows at their recorded commands, full-throttle-region rows at c_hi (the
    full-throttle pattern) and c_cap (the same pattern capped at 80%).

    c_hi and c_cap are shift.make_pools' float64 commands (the pattern, and the pattern clipped to +-0.8) rounded to
    float32, as the critic sees them. A float32 command of 80% is float32(0.8) = thruster_chain.KNEE, the knee itself,
    so c_cap and the stall-aware controller's capped commands are never past 80% and both worlds give the same Q
    there."""
    lg = sd.log
    pol = RC.stall_aware(sd.physics) if controller == "stall_aware" else RC.rising_thrust(sd.physics)
    c_hi64 = np.asarray(RC.Recorder(sd.spec, sd.physics).pattern, np.float64)
    c_hi = c_hi64.astype(np.float32)
    c_cap = np.clip(c_hi64, -0.8, 0.8).astype(np.float32)
    out = {}
    for k in POOLS:
        rows = sd.arrays[f"pool_{k}"]
        x32 = np.asarray(lg["observations"][rows], np.float32)
        x = x32.astype(np.float64)
        pi = pol(x).astype(np.float32)
        acts = dict(policy=pi)
        if k == "region":
            acts.update(c_hi=np.broadcast_to(c_hi, pi.shape).copy(), c_cap=np.broadcast_to(c_cap, pi.shape).copy())
        else:
            acts["logged"] = np.asarray(lg["actions"][rows], np.float32)
        out[k] = dict(rows=rows, x=x, acts=acts, episode=lg["infos/episode"][rows].astype(np.int64))
    return out, pol


def truth_on_pools(inputs, pol, physics, horizon):
    """Q^pi at every pool input in one float64 batch (truth.q_value)."""
    keys = [(k, a) for k in POOLS for a in inputs[k]["acts"]]
    X = np.concatenate([inputs[k]["x"] for k, _ in keys])
    A = np.concatenate([inputs[k]["acts"][a].astype(np.float64) for k, a in keys])
    Q = TR.q_value(pol, X, A, physics, horizon)
    out, i = {}, 0
    for k, a in keys:
        n = len(inputs[k]["x"])
        out[(k, a)] = Q[i:i + n]
        i += n
    return out


def critic_errors(params, prepared, inputs, truth):
    """Per pool: the advantage error e_A = e(s, a) - e(s, pi(s)), e = Q1 - Q^pi (at the recorded command; at c_hi on
    the region pool) and, on the region pool, the pull past 80% o(s) = [Q1(c_hi) - Q1(c_cap)] - [Q^pi(c_hi) -
    Q^pi(c_cap)]."""
    out = {}
    for k in POOLS:
        obs = prepared.normalize(inputs[k]["x"])
        q = {a: L.critic_rows(params, obs, v)[0][:, 0] for a, v in inputs[k]["acts"].items()}
        e = {a: q[a] - truth[(k, a)] for a in q}
        a_main = "c_hi" if k == "region" else "logged"
        out[f"eA|{k}"] = e[a_main] - e["policy"]
        if k == "region":
            out["pull|region"] = TR.pull_past_80(q["c_hi"], q["c_cap"], truth[(k, "c_hi")], truth[(k, "c_cap")])
    return out


def slope_report(params, prepared, inputs, pol, physics, horizon, rows=300, h=1e-3):
    """Reported only: the critic's action-slope error at (s, pi(s)) on far rows, |grad_a Q1 - grad_a Q^pi|, and the
    same in throttle units (divided by the median exact action curvature, by second differences).

    The critic's slope is taken at the float32 command it sees; the exact slope at the controller's own float64
    command pi(s), which for the stall-aware controller is at most 0.8, strictly below the knee (KNEE, the float32
    number nearest 0.8): the slope there is the unstalled one, never the stalled side or the kink. The curvature
    uses central differences, or, where a central step would cross the knee, one-sided differences towards zero."""
    x = inputs["far"]["x"][:rows]
    pi32 = inputs["far"]["acts"]["policy"][:rows]
    pi = np.asarray(pol(x), np.float64)
    g1 = L.critic_rows(params, prepared.normalize(x), pi32, grad=True)[1]
    gt = TR.action_slope(pol, x, pi, physics, horizon)
    q0 = TR.q_value(pol, x, pi, physics, horizon)
    curv = []
    for j in range(TC.DA):
        inward = np.abs(pi[:, j]) + h > physics.knee
        step = np.where(inward, -np.sign(pi[:, j]) * h, h)
        first, second = pi.copy(), pi.copy()
        first[:, j] = pi[:, j] + step
        second[:, j] = np.where(inward, pi[:, j] + 2 * step, pi[:, j] - h)
        qa, qb = TR.q_value(pol, x, first, physics, horizon), TR.q_value(pol, x, second, physics, horizon)
        curv.append(np.where(inward, q0 - 2 * qa + qb, qa + qb - 2 * q0) / h ** 2)
    err = np.linalg.norm(g1 - gt, axis=1)
    c = float(np.median(np.abs(np.stack(curv, 1))))
    return dict(rows=int(len(x)), median_slope_error=float(np.median(err)), median_action_curvature=c,
                median_slope_error_throttle_units=float(np.median(err) / (2 * c)) if c > 0 else None)


def h1_seed(prereg, variant, rep, log=print):
    """H1 for one seed: train (or read) the eight fixed-controller critics and measure their errors on the pools."""
    guard(prereg)
    cfg = prereg.config
    out = _variant_root(prereg, "harm_critic", variant) / f"seed{rep}"
    done = finished(out / "values.json", prereg)
    if done is not None:
        return done
    started = time.perf_counter()
    seeds = {w: load_seed(prereg.runs, cfg, variant, w, rep) for w in TC.WORLDS}
    arrays, values, records = {}, {}, {}
    for world in TC.WORLDS:
        sd = seeds[world]
        for controller in L.CONTROLLERS:
            critics = [c for c in H1_CRITICS if c[0] == world and c[2] == controller]
            if not critics:
                continue
            inputs, pol = pool_inputs(sd, controller)
            truth = truth_on_pools(inputs, pol, sd.physics, cfg.truth_horizon)
            prefix = f"{world}:{controller}"
            for k in POOLS:
                arrays[f"{prefix}:episode|{k}"] = inputs[k]["episode"]
                if "removed_share" in sd.arrays:
                    arrays[f"{prefix}:removed_share|{k}"] = sd.arrays["removed_share"][sd.train_position(inputs[k]["rows"])]
            arrays[f"{prefix}:price|region"] = truth[("region", "c_cap")] - truth[("region", "c_hi")]
            for _, removal, _, placebo in critics:
                name = h1_name(world, removal, controller, placebo)
                rec, params = get_critic(prereg, sd, removal, controller, placebo, log=log)
                records[name] = dict(failed=rec["result"]["failed"], critic_sha256=rec["result"].get("critic_sha256"),
                                     final_critic_loss=(rec["result"]["chunks"] or [{}])[-1].get("critic_loss"))
                if params is None:
                    continue
                errs = critic_errors(params, sd.prepared("gap", cfg.learn), inputs, truth)
                for key, v in errs.items():
                    arrays[f"{name}|{key}"] = v
                values[name] = {key: DT.median_abs(v) if key.startswith("eA") else float(np.median(v))
                                for key, v in errs.items()}
                if removal == "none" and not placebo and world == "stall" and controller == "stall_aware":
                    values[name]["slope"] = slope_report(params, sd.prepared("gap", cfg.learn), inputs, pol,
                                                         sd.physics, cfg.truth_horizon)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "rows.npz", **arrays)
    rec = stage_record(prereg, "harm_critic_seed", started, variant=variant, rep=int(rep), critics=records,
                       medians=values, rows_sha256={k: DS.sha256_array(v) for k, v in arrays.items()},
                       about="H1 for one seed: fixed-controller critics' errors against the exact truth on frozen rows")
    write_json(out / "values.json", rec)
    return rec


def _load_rows(prereg, stage, variant, seeds, extra=""):
    rows = {}
    for s in seeds:
        d = _variant_root(prereg, stage, variant) / extra / f"seed{s}" if extra else \
            _variant_root(prereg, stage, variant) / f"seed{s}"
        rec = finished(d / "values.json", prereg)
        if rec is None:
            raise GateError(f"{stage} has not finished seed {s} ({d})")
        with np.load(d / "rows.npz") as z:
            arr = {k: z[k] for k in z.files}
        if {k: DS.sha256_array(v) for k, v in arr.items()} != rec["rows_sha256"]:
            raise PreregError(f"{d / 'rows.npz'} differs from its record")
        rows[s] = arr
    return rows


def _in_band(r, band):
    return r is not None and 1.0 / band <= r <= band


def h1_pooled(prereg, variant, log=print):
    """H1's pass rule (a)-(f) over the variant's gate seeds."""
    guard(prereg)
    cfg = prereg.config
    m = PASS_MARKS["H1"]
    started = time.perf_counter()
    seeds = list(cfg.gate(variant))
    rows = _load_rows(prereg, "harm_critic", variant, seeds)
    boot = cfg.bootstrap_replicates

    def ratio(num, den, pool, prefix="stall:stall_aware", tag=0):
        a = {s: rows[s].get(f"{num}|eA|{pool}") for s in seeds}
        b = {s: rows[s].get(f"{den}|eA|{pool}") for s in seeds}
        if any(v is None for v in a.values()) or any(v is None for v in b.values()):
            return dict(per_seed={str(s): None for s in seeds}, pooled=None, interval95=[None, None],
                        missing="a critic of some seed failed")
        blocks = {s: rows[s][f"{prefix}:episode|{pool}"] for s in seeds}
        return DT.pooled_ratio(a, b, blocks, boot, [TC.DATA_SEED, TAG_BOOT, tag])

    full, gapc = "stall:stall_aware:full", "stall:stall_aware:gap"
    a = ratio(gapc, full, "removed", tag=1)
    b = ratio(gapc, full, "far", tag=2)
    c = ratio("stall:stall_aware:random", full, "removed", tag=3)
    d = {k: ratio("stall:stall_aware:full:placebo", full, k, tag=4 + i) for i, k in enumerate(POOLS)}
    tw = ratio("no_stall:stall_aware:gap", "no_stall:stall_aware:full", "removed", "no_stall:stall_aware", tag=8)
    e_val = DT.safe_ratio(a["pooled"], tw["pooled"])
    pull = []
    for s in seeds:
        og, of = rows[s].get(f"{gapc}|pull|region"), rows[s].get(f"{full}|pull|region")
        price = float(np.median(rows[s]["stall:stall_aware:price|region"]))
        if og is None or of is None:
            pull.append(dict(seed=s, passed=False, missing=True))
            continue
        mg, mf = float(np.median(og)), float(np.median(of))
        pull.append(dict(seed=s, gapped_median=mg, full_median=mf, true_price_median=price,
                         excess_over_full=mg - mf, passed=bool(mg > 0 and mg - mf >= m["pull_price_share_min"] * price)))
    per_a = list(a["per_seed"].values())
    marks = dict(
        a=bool(a["pooled"] is not None and a["pooled"] >= m["ratio_pooled_min"] and a["interval95"][0] is not None
               and a["interval95"][0] >= m["ratio_lower_min"]
               and all(v is not None and v >= m["ratio_every_seed_min"] for v in per_a)),
        b=bool(_in_band(b["pooled"], m["band"]) and all(_in_band(v, m["band"]) for v in b["interval95"])),
        c=bool(_in_band(c["pooled"], m["band"])),
        d=bool(all(_in_band(v["pooled"], m["band"]) for v in d.values())),
        e=bool(e_val is not None and e_val >= m["hidden_physics_min"]),
        f=bool(sum(p["passed"] for p in pull) >= DT.need(m["pull_seed_share"], len(seeds))))
    linear = dict(removed=ratio("stall:rising_thrust:gap", "stall:rising_thrust:full", "removed",
                                "stall:rising_thrust", 9),
                  far=ratio("stall:rising_thrust:gap", "stall:rising_thrust:full", "far", "stall:rising_thrust", 10))
    g1_table = dict(removed=a["pooled"], far=b["pooled"], region=ratio(gapc, full, "region", tag=11)["pooled"],
                    far_within_d4rl_band_1_3=_in_band(b["pooled"], m["d4rl_far_band_reported"]),
                    note="the D4RL protocol's G1 numbers on this world, reported as a check of the protocol")
    maps, spear = [], {}
    for s in seeds:
        gv, fv = rows[s].get(f"{gapc}|eA|removed"), rows[s].get(f"{full}|eA|removed")
        if gv is None or fv is None:
            continue
        ms_all, lr_all = [], []
        for k in POOLS:
            g, f = np.abs(rows[s][f"{gapc}|eA|{k}"]), np.abs(rows[s][f"{full}|eA|{k}"])
            ms_all.append(rows[s][f"stall:stall_aware:removed_share|{k}"])
            lr_all.append(np.log((g + 1e-9) / (f + 1e-9)))
        ms_all, lr_all = np.concatenate(ms_all), np.concatenate(lr_all)
        spear[str(s)] = DT.spearman(ms_all, lr_all)
        maps.append(dict(seed=s, groups=DT.state_map(ms_all, dict(log_error_ratio=lr_all),
                                                     np.random.default_rng([TC.DATA_SEED, s, TAG_CARD_TIE]).permutation(
                                                         len(ms_all)))))
    complete = all(f"{gapc}|eA|removed" in rows[s] and f"{full}|eA|removed" in rows[s] for s in seeds)
    passed = bool(complete and all(marks.values()))
    rec = stage_record(prereg, "harm_critic", started, variant=variant, seeds=seeds, passed=passed, marks=marks,
                       gap_error_ratio=a, far_ratio=b, random_ratio=c, placebo_ratio=d, no_stall_gap_ratio=tw,
                       hidden_physics_ratio=e_val, pull_past_80=pull,
                       reported=dict(rising_thrust_controller=linear, d4rl_g1_table=g1_table,
                                     per_state_map=maps, spearman_removed_share_log_error_ratio=spear),
                       rule=MARK_TEXT["H1"],
                       about="H1, the critic is wrong where data are missing (pooled over the gate seeds)")
    write_json(_variant_root(prereg, "harm_critic", variant) / "pooled.json", rec)
    log(f"H1 ({variant}): {'pass' if passed else 'FAIL'}\n" + "\n".join(
        f"  {'passed' if v else 'FAILED'}: {MARK_LABELS['H1'][k]}" for k, v in marks.items()))
    return rec


# ------------------------------------------------------------------------------------------------ the stop rule

def harm_state(prereg, variant):
    """Where the harm checks of a variant stand: H1's pooled result, H2's per alpha, the chosen alpha (the first alpha
    of the ladder at which H1 and H2 both pass), whether the harm checks failed, and what runs next."""
    cfg = prereg.config
    h1 = finished(_variant_root(prereg, "harm_critic", variant) / "pooled.json", prereg)
    h2 = {}
    for a in cfg.alpha_ladder:
        r = finished(_variant_root(prereg, "harm_return", variant) / f"alpha{a:g}" / "pooled.json", prereg)
        h2[a] = None if r is None else bool(r["passed"])
    out = dict(variant=variant, h1=None if h1 is None else bool(h1["passed"]), h2={f"{a:g}": v for a, v in h2.items()},
               alpha=None, failed=False, next=None)
    if h1 is not None and not h1["passed"]:
        out.update(failed=True, reason="H1 failed")
        return out
    for a in cfg.alpha_ladder:
        if h2[a] is None:
            if a == cfg.alpha_ladder[0]:
                out["next"] = f"H2 at alpha {a:g}" + ("" if h1 is not None else " (and H1)")
                out["next_alpha"] = a
            else:
                out["next"] = f"H2 at alpha {a:g}" if h1 is not None else "H1 pooled"
                out["next_alpha"] = a if h1 is not None else None
            return out
        if h2[a]:
            if h1 is None:
                out["next"] = "H1 pooled"
                return out
            out["alpha"] = a
            return out
    out.update(failed=True, reason="H2 failed at every alpha of the ladder")
    return out


def variant_allowed(prereg, variant):
    if variant == "base":
        return True
    if prereg.payload["world"].get("retune") is None:
        raise GateError("the retune world has no passing Stage 0: the retune is not available")
    if not harm_state(prereg, "base")["failed"]:
        raise GateError("the retune runs only when the base world's harm checks failed (H1, or H2 at every alpha)")
    return True


def active_variant(prereg):
    return "retune" if harm_state(prereg, "base")["failed"] else "base"


def resolve_variant(prereg, variant):
    v = active_variant(prereg) if variant in (None, "auto") else variant
    variant_allowed(prereg, v)
    return v


def require_alpha_allowed(prereg, variant, alpha):
    """The alpha ladder rule: alpha 2.5 first; a higher rung only when H1 passed and H2 failed at the previous rung."""
    ladder = list(prereg.config.alpha_ladder)
    if alpha not in ladder:
        raise GateError(f"alpha must be one of the pre-registered ladder {ladder}")
    k = ladder.index(alpha)
    if k == 0:
        return
    st = harm_state(prereg, variant)
    if st["h1"] is not True:
        raise GateError(f"alpha {alpha:g} runs only after H1 passed (H1: {st['h1']})")
    prev = st["h2"][f"{ladder[k - 1]:g}"]
    if prev is not False:
        raise GateError(f"alpha {alpha:g} runs only when H2 failed at alpha {ladder[k - 1]:g} (H2 there: {prev})")


def chosen_alpha(prereg, variant):
    st = harm_state(prereg, variant)
    if st["alpha"] is None:
        raise GateError(f"no alpha chosen yet for the {variant} world: {st.get('reason') or st['next']}")
    return st["alpha"]


# ------------------------------------------------------------------------------------------------ H2

def _evaluate(params, prepared, anchors, cfg, failed=False):
    if params is None or failed:
        return None
    return L.evaluate_actor(params, prepared, anchors, n=cfg.bank_size)


def _pts(ev, name, s):
    v = ev.get(name)
    return None if v is None else v[s]["points"]


def _diff(a, b):
    return None if a is None or b is None else a - b


def host_errors(sd, state, prepared, horizon):
    """Reported only (H1's last reported item): a co-trained TD3+BC run's critic against its own actor on the frozen
    removed and far pools: the median |advantage error| at the recorded commands, with the actor's exact Q."""
    out = {}
    for k in ("removed", "far"):
        rows = sd.arrays[f"pool_{k}"]
        x = np.asarray(sd.log["observations"][rows], np.float32).astype(np.float64)
        obs = prepared.normalize(x)
        a = np.asarray(sd.log["actions"][rows], np.float32)
        pi = L.actor_rows(state.actor.params, obs)
        q_a = L.critic_rows(state.critic.params, obs, a)[0][:, 0]
        q_pi = L.critic_rows(state.critic.params, obs, pi)[0][:, 0]
        t_a = L.exact_q(state.actor.params, prepared, x, a.astype(np.float64), horizon=horizon)
        t_pi = L.exact_q(state.actor.params, prepared, x, pi.astype(np.float64), horizon=horizon)
        out[k] = DT.median_abs(TR.advantage_error(q_a, q_pi, t_a, t_pi))
    return out


def h2_seed(prereg, variant, alpha, rep, log=print):
    """H2 for one seed at one alpha: the critic swap, its points, shares and exact attribution."""
    guard(prereg)
    cfg, learn = prereg.config, prereg.config.learn
    out = _variant_root(prereg, "harm_return", variant) / f"alpha{alpha:g}" / f"seed{rep}"
    done = finished(out / "values.json", prereg)
    if done is not None:
        return done
    variant_allowed(prereg, variant)
    require_alpha_allowed(prereg, variant, alpha)
    started = time.perf_counter()
    st, tw = (load_seed(prereg.runs, cfg, variant, w, rep) for w in ("stall", "no_stall"))
    pg, pt = st.prepared("gap", learn), tw.prepared("gap", learn)
    crit = {(w, r): get_critic(prereg, sd, r, "stall_aware", train=False)[1]
            for w, sd in (("stall", st), ("no_stall", tw)) for r in (("gap", "none", "random") if w == "stall"
                                                                    else ("gap", "none"))}
    clone_s, clone_t = get_clone(prereg, st, log), get_clone(prereg, tw, log)
    _, host_g = get_host(prereg, st, "gap", alpha, "plain", log=log)
    _, host_f = get_host(prereg, st, "none", alpha, "plain", log=log)
    actors, failed = {"clone": clone_s, "no_stall_clone": clone_t}, {"clone": False, "no_stall_clone": False}

    def swap(prep, start, named):
        ok = [(n, c) for n, c in named if c is not None]
        for n, c in named:
            failed[n] = c is None
        if not ok:
            return
        ao = L.ActorOnly(prep, start, learn, rep, alpha=alpha)
        stacked, info = ao.run([L.Arm(n, critic=i) for i, (n, _) in enumerate(ok)], [c for _, c in ok],
                               weighted=False, log=log)
        for i, (n, _) in enumerate(ok):
            actors[n] = ao.arm_params(stacked, i)
            failed[n] = bool(info["failed"][i])

    swap(pg, clone_s, [("gapped", crit[("stall", "gap")]), ("full", crit[("stall", "none")]),
                       ("random", crit[("stall", "random")])])
    swap(pt, clone_t, [("no_stall_gapped", crit[("no_stall", "gap")]), ("no_stall_full", crit[("no_stall", "none")])])
    swap(pg, clone_s, [("host_swap_gapped", None if host_g is None else host_g.critic.params),
                       ("host_swap_full", None if host_f is None else host_f.critic.params)])
    actors["td3bc_gapped"], failed["td3bc_gapped"] = (None, True) if host_g is None else (host_g.actor.params, False)
    out.mkdir(parents=True, exist_ok=True)
    actors_sha = L.save_tree(out / "actors.msgpack", {k: v for k, v in actors.items() if v is not None})
    an_s, an_t = anchors_of(prereg, variant, "stall"), anchors_of(prereg, variant, "no_stall")
    ev = {n: _evaluate(p, pt if n.startswith("no_stall") else pg, an_t if n.startswith("no_stall") else an_s, cfg, failed[n])
          for n, p in actors.items()}
    attribution = {}
    if not (failed["gapped"] or failed["full"]):
        for s in SITUATIONS:
            n_attr = cfg.attribution_starts if s == PASS_MARKS["H2"]["situation"] else cfg.attribution_starts_other
            x0 = TC.bank_starts(s, cfg.bank_size)[:n_attr]
            attr = TR.attribution(L.mlp_policy(actors["gapped"], pg), L.mlp_policy(actors["full"], pg), x0, st.physics)
            attribution[s] = dict(L.loss_rate_ratio(attr, st.physics), starts=int(len(x0)))
    values = {}
    for s in SITUATIONS:
        p = {n: _pts(ev, n, s) for n in actors}
        values[s] = dict(points=p, harm=_diff(p["full"], p["gapped"]), random_harm=_diff(p["full"], p["random"]),
                         specific=_diff(p["random"], p["gapped"]), no_stall_harm=_diff(p["no_stall_full"], p["no_stall_gapped"]),
                         harm_minus_no_stall_harm=_diff(_diff(p["full"], p["gapped"]),
                                                    _diff(p["no_stall_full"], p["no_stall_gapped"])),
                         host_swap_harm=_diff(p["host_swap_full"], p["host_swap_gapped"]),
                         td3bc_minus_cloning=_diff(p["td3bc_gapped"], p["clone"]),
                         unrecorded_throttle_share={n: None if ev[n] is None else ev[n][s]["unrecorded_throttle_share"]
                                                    for n in actors},
                         loss_from_unrecorded_throttle={n: None if ev[n] is None else
                                                        ev[n][s]["loss_from_unrecorded_throttle"] for n in actors},
                         attribution=attribution.get(s))
    hosts = {k: None if h is None else host_errors(st, h, pg, cfg.truth_horizon)
             for k, h in (("gapped", host_g), ("full", host_f))}
    if hosts["gapped"] and hosts["full"]:
        hosts["ratio_removed"] = DT.safe_ratio(hosts["gapped"]["removed"], hosts["full"]["removed"])
        hosts["ratio_far"] = DT.safe_ratio(hosts["gapped"]["far"], hosts["full"]["far"])
    rec = stage_record(prereg, "harm_return_seed", started, variant=variant, alpha=alpha, rep=int(rep),
                       failed=failed, actors_sha256=actors_sha, evaluations=ev, situations=values,
                       host_critic_errors=hosts,
                       actor_names=sorted(k for k, v in actors.items() if v is not None),
                       about="H2 for one seed: actors trained against frozen critics that differ only in their data")
    write_json(out / "values.json", rec)
    return rec


def _complete(values, n):
    return len([v for v in values if v is not None]) == n


def _mark_mean(values, n, minimum, lower_gt, share=None, need_share=None):
    s = DT.summary(values)
    ok = _complete(values, n) and s["mean"] is not None and s["mean"] >= minimum and s["lower"] is not None \
        and s["lower"] > lower_gt
    if share is not None:
        ok = ok and s["positive"] >= DT.need(need_share, n)
    return dict(s, passed=bool(ok))


def loss_rate_passes(past, other, ratio_min):
    """H2(e), as pre-registered (MARK_TEXT['H2']): the loss rate at the past-80% steps is positive and at least
    ratio_min times the loss rate at the other steps; when the other steps lose nothing (a rate of zero or a gain),
    any positive rate at the past-80% steps meets it (the ratio is then infinite or the loss sits entirely there)."""
    if past is None or other is None or not past > 0:
        return False
    return bool(other <= 0 or past / other >= ratio_min)


def h2_pooled(prereg, variant, alpha, log=print):
    cfg = prereg.config
    m = PASS_MARKS["H2"]
    started = time.perf_counter()
    seeds = list(cfg.gate(variant))
    recs = []
    for s in seeds:
        r = finished(_variant_root(prereg, "harm_return", variant) / f"alpha{alpha:g}" / f"seed{s}" / "values.json",
                     prereg)
        if r is None:
            raise GateError(f"harm_return at alpha {alpha:g} has not finished seed {s}")
        recs.append(r)
    n = len(seeds)
    by = {}
    for sit in SITUATIONS:
        v = [r["situations"][sit] for r in recs]
        sg = [x["unrecorded_throttle_share"]["gapped"] for x in v]
        sf = [x["unrecorded_throttle_share"]["full"] for x in v]
        mg, mf = DT.mean(sg), DT.mean(sf)
        ratio = None if mg is None or mf is None else (math.inf if mf == 0 and mg > 0 else DT.safe_ratio(mg, mf))
        attr = [x["attribution"] for x in v]
        if all(a is not None for a in attr):
            sp, so = sum(a["sum_loss_past_80"] for a in attr), sum(a["sum_loss_other"] for a in attr)
            np_, no = sum(a["steps_past_80"] for a in attr), sum(a["steps_other"] for a in attr)
            lp, lo = (sp / np_ if np_ else None), (so / no if no else None)
            lr = dict(loss_rate_past_80=lp, loss_rate_other=lo, ratio=DT.safe_ratio(lp, lo) if lo and lo > 0 else None,
                      passed=loss_rate_passes(lp, lo, m["loss_rate_ratio_min"]), rule=MARK_LABELS["H2"]["e"])
        else:
            lr = dict(passed=False, missing=True)
        by[sit] = dict(
            harm=_mark_mean([x["harm"] for x in v], n, m["harm_min"], m["harm_lower_gt"], True, m["harm_seed_share"]),
            specific=_mark_mean([x["specific"] for x in v], n, m["specific_min"], m["specific_lower_gt"]),
            hidden_physics=_mark_mean([x["harm_minus_no_stall_harm"] for x in v], n, m["physics_min"],
                                      m["physics_lower_gt"]),
            mechanism=dict(share_gapped=sg, share_full=sf, ratio_of_means=ratio,
                           seeds_higher=int(sum(1 for a, b in zip(sg, sf) if a is not None and b is not None and a > b)),
                           passed=bool(_complete(sg, n) and _complete(sf, n) and ratio is not None
                                       and ratio >= m["mechanism_ratio_min"]
                                       and sum(1 for a, b in zip(sg, sf) if a is not None and b is not None and a > b)
                                       >= DT.need(m["mechanism_seed_share"], n))),
            loss_rate=lr,
            reported=dict(host_swap_harm=DT.summary([x["host_swap_harm"] for x in v]),
                          td3bc_minus_cloning=DT.summary([x["td3bc_minus_cloning"] for x in v]),
                          no_stall_harm=DT.summary([x["no_stall_harm"] for x in v]),
                          random_harm=DT.summary([x["random_harm"] for x in v])))
    host_ratio = dict(removed=DT.geometric_mean([(r.get("host_critic_errors") or {}).get("ratio_removed")
                                                 for r in recs]),
                      far=DT.geometric_mean([(r.get("host_critic_errors") or {}).get("ratio_far") for r in recs]),
                      note="reported only: the co-trained TD3+BC runs' critics against their own actors (gapped over "
                           "full log), median absolute advantage error")
    main = by[m["situation"]]
    marks = dict(a=main["harm"]["passed"], b=main["specific"]["passed"], c=main["hidden_physics"]["passed"],
                 d=main["mechanism"]["passed"], e=main["loss_rate"]["passed"])
    passed = bool(all(marks.values()))
    rec = stage_record(prereg, "harm_return", started, variant=variant, alpha=alpha, seeds=seeds, passed=passed,
                       marks=marks, situations=by, host_critic_error_ratio=host_ratio, rule=MARK_TEXT["H2"],
                       about="H2, acting on the critic error costs return (pooled; marks on strong current, the other "
                             "situations reported)")
    write_json(_variant_root(prereg, "harm_return", variant) / f"alpha{alpha:g}" / "pooled.json", rec)
    log(f"H2 ({variant}, alpha {alpha:g}): {'pass' if passed else 'FAIL'}\n" + "\n".join(
        f"  {'passed' if v else 'FAILED'}: {MARK_LABELS['H2'][k]}" for k, v in marks.items()))
    return rec


# ------------------------------------------------------------------------------------------------ H3

def verify_lock(path, prereg):
    """The room check's (or the BCA stage's) blind record: it must exist, match its sidecar and belong to this
    pre-registration before any return is computed."""
    guard(prereg)
    rec = finished(path, prereg)
    if rec is None:
        raise GateError(f"no verified lock at {path}: no return may be computed before the match file is locked")
    return rec


def room_match(runner, critic, none_params, eval_obs, ladder, target, failed_arms, learn, log=print):
    """The room check's blind strength match (learners.match_even), or, when the harm-placed or no-caution arm failed
    or harm-placed caution has no finite positive strength, no match at all: then the strength-matched arm is
    missing, this seed's H3 differences against it are None, and every pooled mark that needs the seed fails (a failed
    seed fails the mark and is never dropped) instead of the stage crashing and the gate staying pending."""
    if failed_arms.get("harm_placed") or failed_arms.get("none") or not (math.isfinite(target) and target > 0):
        return dict(multiplier=None, knob=None, ladder_multiplier=None, skipped=True,
                    reason="the harm-placed or no-caution arm failed or has no finite positive strength, so no even "
                           "caution can be matched to it"), {}
    return L.match_even(runner, [critic], none_params, eval_obs, ladder, target, learn, log=log)


def h3_seed(prereg, variant, rep, log=print):
    """H3 for one seed at the chosen alpha: train every arm and match even caution blind, lock it, then score."""
    guard(prereg)
    cfg, learn = prereg.config, prereg.config.learn
    variant_allowed(prereg, variant)
    alpha = chosen_alpha(prereg, variant)
    out = _variant_root(prereg, "room", variant) / f"alpha{alpha:g}" / f"seed{rep}"
    done = finished(out / "values.json", prereg)
    if done is not None:
        return done
    started = time.perf_counter()
    st = load_seed(prereg.runs, cfg, variant, "stall", rep)
    pg = st.prepared("gap", learn)
    lock = out / "room_blind.json"
    if finished(lock, prereg) is None:
        _, host = get_host(prereg, st, "gap", alpha, "plain", log=log)
        clone = get_clone(prereg, st, log)
        arms = L.room_arms(cfg.n_shuffled)
        params, S, match, info = {}, {}, None, dict(history=[], rows_sha256=None, failed=[])
        if host is not None:
            critic = host.critic.params
            ao = L.ActorOnly(pg, clone, learn, rep, alpha=alpha)
            stacked, info = ao.run(arms, [critic], log=log)
            params = {a.name: ao.arm_params(stacked, i) for i, a in enumerate(arms)}
            S = {k: L.strength(p, params["none"], pg.eval_obs) for k, p in params.items()}
            ladder = [(a.value, S[a.name]) for a in arms if a.kind == "even" and a.value > 1]
            failed_arms = dict(zip([a.name for a in arms], info["failed"]))
            match, evaluated = room_match(ao, critic, params["none"], pg.eval_obs, ladder, S["harm_placed"],
                                          failed_arms, learn, log=log)
            mult = match["multiplier"]
            if mult is not None:
                params["strength_matched_even"] = params["none"] if match["knob"] == 0 else (
                    params[f"even_x{match['ladder_multiplier']:g}"] if match["ladder_multiplier"] is not None
                    else evaluated[mult])
                S["strength_matched_even"] = L.strength(params["strength_matched_even"], params["none"], pg.eval_obs)
        out.mkdir(parents=True, exist_ok=True)
        sha = L.save_tree(out / "actors.msgpack", params) if params else None
        blind = stage_record(prereg, "room_blind", started, variant=variant, alpha=alpha, rep=int(rep),
                             host_failed=host is None, names=sorted(params), actors_sha256=sha, strengths=S,
                             match=match, dose_history=info["history"], arms_failed=dict(zip([a.name for a in arms],
                                                                                               info["failed"])),
                             rows_sha256=info["rows_sha256"],
                             about="the room check's lock: every arm trained and even caution matched in strength "
                                   "to harm-placed caution, written before any return is computed")
        write_json(lock, blind)
    blind = verify_lock(lock, prereg)
    params = {}
    if blind["names"]:
        params = L.load_tree(out / "actors.msgpack", {k: L.actor_template() for k in blind["names"]},
                             blind["actors_sha256"])
    an = anchors_of(prereg, variant, "stall")
    failed = blind["arms_failed"]
    ev = {k: _evaluate(p, pg, an, cfg, failed.get(k, False)) for k, p in params.items()}
    values = {}
    even_names = ["none"] + [f"even_x{m:g}" for m in L.ROOM_EVEN_LADDER]
    for s in SITUATIONS:
        p = {k: _pts(ev, k, s) for k in params}
        hp, cap2 = p.get("harm_placed"), p.get("cap2")
        shuf = [p.get(f"shuffled_{k}") for k in range(cfg.n_shuffled)]
        shuf_mean = None if any(v is None for v in shuf) or not shuf else float(np.mean(shuf))
        evens = {k: p.get(k) for k in even_names}
        best = None if any(v is None for v in evens.values()) else max(evens, key=evens.get)
        values[s] = dict(points=p, harm_placed_minus_matched_even=_diff(hp, p.get("strength_matched_even")),
                         harm_placed_minus_amount_matched=_diff(hp, p.get("amount_matched")),
                         harm_placed_minus_shuffled=_diff(hp, shuf_mean), harm_placed_minus_none=_diff(hp, p.get("none")),
                         best_even=best, best_even_points=None if best is None else evens[best],
                         harm_placed_minus_best_even=_diff(hp, None if best is None else evens[best]),
                         reversed_minus_harm_placed=_diff(p.get("reversed"), hp),
                         cap2_minus_matched_even=_diff(cap2, p.get("strength_matched_even")),
                         cap2_minus_best_even=_diff(cap2, None if best is None else evens[best]),
                         cap2_minus_none=_diff(cap2, p.get("none")))
    rec = stage_record(prereg, "room_seed", started, variant=variant, alpha=alpha, rep=int(rep),
                       lock_sha256=file_sha256(lock), evaluations=ev, situations=values,
                       about="H3 for one seed: caution placed by the exact harm against even caution, scored after "
                             "the lock")
    write_json(out / "values.json", rec)
    return rec


def h3_pooled(prereg, variant, log=print):
    cfg = prereg.config
    m = PASS_MARKS["H3"]
    alpha = chosen_alpha(prereg, variant)
    started = time.perf_counter()
    seeds = list(cfg.gate(variant))
    recs = []
    for s in seeds:
        r = finished(_variant_root(prereg, "room", variant) / f"alpha{alpha:g}" / f"seed{s}" / "values.json", prereg)
        if r is None:
            raise GateError(f"room has not finished seed {s}")
        recs.append(r)
    n = len(seeds)
    rec_none = [r["situations"]["recorded"]["harm_placed_minus_none"] for r in recs]
    recorded_ok = bool(_complete(rec_none, n) and DT.mean(rec_none) >= m["recorded_min"])
    by, certified = {}, []
    for s in SITUATIONS:
        v = [r["situations"][s] for r in recs]
        mk = {k: _mark_mean([x[k] for x in v], n, m["margin_min"], m["margin_lower_gt"], True, m["seed_share"])
              for k in ("harm_placed_minus_matched_even", "harm_placed_minus_amount_matched",
                        "harm_placed_minus_shuffled")}
        ok = bool(s in SHIFTED and all(x["passed"] for x in mk.values()) and recorded_ok)
        by[s] = dict(marks=mk, certified=ok,
                     shown=dict(harm_placed_minus_best_even=DT.summary([x["harm_placed_minus_best_even"] for x in v]),
                                best_even=[x["best_even"] for x in v],
                                cap2_minus_matched_even=DT.summary([x["cap2_minus_matched_even"] for x in v]),
                                cap2_minus_best_even=DT.summary([x["cap2_minus_best_even"] for x in v]),
                                reversed_minus_harm_placed=DT.summary([x["reversed_minus_harm_placed"] for x in v]),
                                harm_placed_minus_none=DT.summary([x["harm_placed_minus_none"] for x in v])))
        if ok:
            certified.append(s)
    passed = len(certified) >= m["certified_min"]
    rec = stage_record(prereg, "room", started, variant=variant, alpha=alpha, seeds=seeds, passed=bool(passed),
                       certified=certified, recorded_conditions=dict(values=rec_none, mean=DT.mean(rec_none),
                                                                     passed=recorded_ok),
                       situations=by, rule=MARK_TEXT["H3"],
                       about="H3, room: which shifted situations caution placed by the exact harm could help")
    write_json(_variant_root(prereg, "room", variant) / f"alpha{alpha:g}" / "pooled.json", rec)
    log(f"H3 ({variant}, alpha {alpha:g}): certified {certified or 'none'}")
    return rec


# ------------------------------------------------------------------------------------------------ H4

def room_state(prereg, variant):
    try:
        alpha = chosen_alpha(prereg, variant)
    except GateError:
        return None
    return finished(_variant_root(prereg, "room", variant) / f"alpha{alpha:g}" / "pooled.json", prereg)


def h4_seed(prereg, variant, rep, log=print):
    cfg = prereg.config
    variant_allowed(prereg, variant)
    alpha = chosen_alpha(prereg, variant)
    room = room_state(prereg, variant)
    if room is None or not room["passed"]:
        raise GateError("power runs only after the room check certified at least one situation")
    out = _variant_root(prereg, "power", variant) / f"alpha{alpha:g}" / f"seed{rep}"
    done = finished(out / "values.json", prereg)
    if done is not None:
        return done
    started = time.perf_counter()
    st = load_seed(prereg.runs, cfg, variant, "stall", rep)
    pg = st.prepared("gap", cfg.learn)
    an = anchors_of(prereg, variant, "stall")
    ev = {}
    for arm in ("plain", "plain_stream1", "even_x1.25"):
        _, state = get_host(prereg, st, "gap", alpha, arm, log=log)
        ev[arm] = None if state is None else L.evaluate_actor(state.actor.params, pg, an, n=cfg.bank_size)
    values = {s: dict(null_pair=_diff(_pts(ev, "plain_stream1", s), _pts(ev, "plain", s)),
                      even_minus_plain=_diff(_pts(ev, "even_x1.25", s), _pts(ev, "plain", s))) for s in SITUATIONS}
    rec = stage_record(prereg, "power_seed", started, variant=variant, alpha=alpha, rep=int(rep), evaluations=ev,
                       situations=values, about="H4 for one seed: two paired differences of co-trained TD3+BC runs")
    write_json(out / "values.json", rec)
    return rec


def required_seeds(s, marks=PASS_MARKS["H4"]):
    return max(marks["n_min"], int(math.ceil(((marks["z_alpha"] + marks["z_power"]) * marks["inflation"] * s
                                              / marks["effect"]) ** 2)))


def h4_pooled(prereg, variant, log=print):
    cfg = prereg.config
    m = PASS_MARKS["H4"]
    alpha = chosen_alpha(prereg, variant)
    room = room_state(prereg, variant)
    if room is None or not room["passed"]:
        raise GateError("power runs only after the room check certified at least one situation")
    started = time.perf_counter()
    seeds = list(cfg.gate(variant))
    recs = []
    for s in seeds:
        r = finished(_variant_root(prereg, "power", variant) / f"alpha{alpha:g}" / f"seed{s}" / "values.json", prereg)
        if r is None:
            raise GateError(f"power has not finished seed {s}")
        recs.append(r)
    sds, complete = {}, True
    for s in SITUATIONS:
        for k in ("null_pair", "even_minus_plain"):
            v = [r["situations"][s][k] for r in recs]
            complete &= s not in room["certified"] or _complete(v, len(seeds))
            sds[f"{s}:{k}"] = DT.summary(v)["sd"]
    used = [sds[f"{s}:{k}"] for s in room["certified"] for k in ("null_pair", "even_minus_plain")]
    s_max = None if any(v is None for v in used) or not used else max(used)
    N = None if s_max is None else required_seeds(s_max)
    passed = bool(complete and N is not None and N <= m["n_max"])
    smallest = None if s_max is None else (m["z_alpha"] + m["z_power"]) * m["inflation"] * s_max / math.sqrt(m["n_max"])
    rec = stage_record(prereg, "power", started, variant=variant, alpha=alpha, seeds=seeds, passed=passed,
                       certified=room["certified"], sd=sds, s=s_max, N=N, smallest_effect_at_20_seeds=smallest,
                       rule=MARK_TEXT["H4"], about="H4, power: how many confirmation seeds the BCA stage needs")
    write_json(_variant_root(prereg, "power", variant) / f"alpha{alpha:g}" / "pooled.json", rec)
    log(f"H4 ({variant}, alpha {alpha:g}): spread of the paired differences (largest standard deviation) {s_max}, "
        f"confirmation seeds needed {N}: {'pass' if passed else 'FAIL'}")
    return rec


# ------------------------------------------------------------------------------------------------ the gate report

def gate_outcome(prereg):
    """The stop rule applied to what has run: 'pending' (with the next stage), 'stop' (with the reason) or 'ready'."""
    guard(prereg)
    base = harm_state(prereg, "base")
    harm = dict(base=base)
    variant, hs = "base", base
    if base["failed"]:
        if prereg.payload["world"].get("retune") is None:
            return dict(state="stop", reason="no_harm", variant="base", harm=harm,
                        detail="the retune world has no passing Stage 0, so the retune cannot run")
        ret = harm_state(prereg, "retune")
        harm["retune"] = ret
        if ret["failed"]:
            return dict(state="stop", reason="no_harm", variant="retune", harm=harm)
        variant, hs = "retune", ret
    if hs["alpha"] is None:
        return dict(state="pending", next=f"{variant} world: {hs['next']}", variant=variant, harm=harm)
    room = room_state(prereg, variant)
    if room is None:
        return dict(state="pending", next=f"{variant} world: room (H3)", variant=variant, alpha=hs["alpha"], harm=harm)
    if not room["passed"]:
        return dict(state="stop", reason="no_room", variant=variant, alpha=hs["alpha"], harm=harm)
    power = finished(_variant_root(prereg, "power", variant) / f"alpha{hs['alpha']:g}" / "pooled.json", prereg)
    if power is None:
        return dict(state="pending", next=f"{variant} world: power (H4)", variant=variant, alpha=hs["alpha"],
                    harm=harm, certified=room["certified"])
    if not power["passed"]:
        return dict(state="stop", reason="underpowered", variant=variant, alpha=hs["alpha"], harm=harm,
                    certified=room["certified"], N=power["N"], smallest_effect_at_20_seeds=power["smallest_effect_at_20_seeds"])
    return dict(state="ready", reason="ready", variant=variant, alpha=hs["alpha"], harm=harm,
                certified=room["certified"], N=power["N"])


def _last_alpha_run(prereg, variant):
    alphas = [a for a in prereg.config.alpha_ladder
              if (_variant_root(prereg, "harm_return", variant) / f"alpha{a:g}").exists()]
    return alphas[-1] if alphas else None


def report_cards(prereg, rep, log=print):
    """The shift cards of one gate seed (stalling world, gapped log): reference controllers and every learned actor
    the harm checks produced, on every situation."""
    guard(prereg)
    cfg = prereg.config
    outcome = gate_outcome(prereg)
    variant = outcome["variant"]
    if rep not in cfg.gate(variant):
        raise GateError(f"seed {rep} is not a gate seed of the {variant} world")
    out = prereg.root / "report" / variant / f"cards_seed{rep}.json"
    done = finished(out, prereg)
    if done is not None:
        return done
    started = time.perf_counter()
    st = load_seed(prereg.runs, cfg, variant, "stall", rep)
    pg = st.prepared("gap", cfg.learn)
    recorder = RC.Recorder(st.spec, st.physics)
    ref = SH.card_reference(st.log, st.removals["gap"], recorder)
    policies = {"recording controller": TR.recorder_on_bank(st.physics, st.spec, TC.BANK_SIZE),
                "stall-aware controller": RC.stall_aware(st.physics),
                "rising-thrust controller": RC.rising_thrust(st.physics)}
    alpha = outcome.get("alpha") or _last_alpha_run(prereg, variant)
    if alpha is not None:
        d = _variant_root(prereg, "harm_return", variant) / f"alpha{alpha:g}" / f"seed{rep}"
        rec = finished(d / "values.json", prereg)
        if rec is not None:
            acts = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in rec["actor_names"]},
                               rec["actors_sha256"])
            for k, label in (("clone", "cloned actor"), ("gapped", "actor of the gapped critic (H2)"),
                             ("full", "actor of the full-log critic (H2)"),
                             ("td3bc_gapped", "plain TD3+BC on the gapped log")):
                if k in acts and not rec["failed"].get(k):
                    policies[label] = L.mlp_policy(acts[k], pg, label)
        d = _variant_root(prereg, "room", variant) / f"alpha{alpha:g}" / f"seed{rep}"
        blind = finished(d / "room_blind.json", prereg)
        if blind is not None and blind["names"]:
            acts = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in blind["names"]},
                               blind["actors_sha256"])
            for k, label in (("none", "room check, no extra caution"), ("harm_placed", "room check, harm-placed")):
                if k in acts:
                    policies[label] = L.mlp_policy(acts[k], pg, label)
    an = anchors_of(prereg, variant, "stall")
    cards = {}
    for name, pol in policies.items():
        cards[name] = {s: SH.shift_card(pol, s, st.physics, ref, an, ms_states=cfg.card_ms_states,
                                        knn_states=cfg.card_knn_states) for s in SITUATIONS}
        log(f"shift cards, seed {rep}: {name}")
    rec = stage_record(prereg, "cards", started, variant=variant, rep=int(rep), alpha=alpha, cards=cards,
                       about="the per-state shift card of every policy and situation on this seed's gapped log")
    write_json(out, rec)
    return rec


def _fmt(x, d=2):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{d}f}"


def report_markdown(rep):
    """The gate report in plain words."""
    o = rep["outcome"]
    lines = ["# ThrusterChain shift test: gate report", "",
             f"Pre-registration: `{rep['prereg_sha256']}`", "",
             "## Outcome", "", rep["message"], ""]
    if o.get("detail"):
        lines += [o["detail"], ""]
    lines += [f"- World: {'stall slope 3.0 (the pre-registered retune)' if o['variant'] == 'retune' else 'stall slope 1.5'}",
              f"- TD3+BC alpha chosen by the alpha ladder: {_fmt(o.get('alpha'), 1)}",
              f"- Room-certified situations: {', '.join(LABEL[s] for s in o.get('certified', [])) or 'none'}",
              f"- Confirmation seeds needed (N): {_fmt(o.get('N'))}",
              f"- The BCA stage may be considered: {_fmt(rep['bca_allowed'])}", ""]
    for variant, h in rep["harm"].items():
        lines += [f"## Harm checks, {variant} world", ""]
        h1 = h.get("H1")
        if h1:
            lines += ["### H1: is the critic wrong where data are missing?", "", h1["rule"], "",
                      "| Mark | Passed |", "|---|---|"]
            lines += [f"| {MARK_LABELS['H1'][k]} | {_fmt(v)} |" for k, v in h1["marks"].items()]
            lines += ["", f"Gap error ratio (gapped / full critic at the removed commands): "
                          f"{_fmt(h1['gap_error_ratio']['pooled'])}, 95% interval "
                          f"{_fmt(h1['gap_error_ratio']['interval95'][0])} to "
                          f"{_fmt(h1['gap_error_ratio']['interval95'][1])}.", ""]
        for alpha, h2 in sorted(h.get("H2", {}).items()):
            sc = h2["situations"]["strong_current"]
            lines += [f"### H2 at alpha {alpha}: does acting on the error cost return?", "",
                      f"Harm in strong current (full-log critic's actor minus gapped critic's actor): "
                      f"{_fmt(sc['harm']['mean'])} points (lower bound {_fmt(sc['harm']['lower'])}). "
                      f"Passed: {_fmt(h2['passed'])}.", "", "| Mark | Passed |", "|---|---|"]
            lines += [f"| {MARK_LABELS['H2'][k]} | {_fmt(v)} |" for k, v in h2["marks"].items()] + [""]
        h3 = h.get("H3")
        if h3:
            lines += ["### H3: is there room for caution placed by the exact harm?", "",
                      "| Situation | vs strength-matched even | vs amount-matched | vs shuffled | best even, minus | "
                      "cap-2 vs matched even | certified |", "|---|---|---|---|---|---|---|"]
            for s in [x for x in SITUATIONS if x in h3["situations"]]:
                v = h3["situations"][s]
                mk, sh = v["marks"], v["shown"]
                lines.append(f"| {LABEL[s]} | {_fmt(mk['harm_placed_minus_matched_even']['mean'])} | "
                             f"{_fmt(mk['harm_placed_minus_amount_matched']['mean'])} | "
                             f"{_fmt(mk['harm_placed_minus_shuffled']['mean'])} | "
                             f"{_fmt(sh['harm_placed_minus_best_even']['mean'])} | "
                             f"{_fmt(sh['cap2_minus_matched_even']['mean'])} | {_fmt(v['certified'])} |")
            lines.append("")
        h4 = h.get("H4")
        if h4:
            lines += ["### H4: how many seeds?", "",
                      f"Spread of the paired differences across the gate seeds (the larger standard deviation of the "
                      f"null pair and of even caution x1.25 minus plain, in points): {_fmt(h4['s'])}. Confirmation "
                      f"seeds needed for a 3-point effect: {_fmt(h4['N'])}; at most 20 allowed. Passed: "
                      f"{_fmt(h4['passed'])}.", ""]
    lines += ["## How to read this", "",
              "- Points: 0 is the recording controller, 100 the stall-aware controller (scores above 100 are possible).",
              "- Each mark is fixed in the pre-registration; a failed seed fails a pooled mark.",
              "- Nothing BCA-related has run. The BCA stage runs only if the user approves this report "
              "(`python -m experiments.shift.acceptance approve`, typed at a terminal).", ""]
    return "\n".join(lines)


def stage_report(prereg, log=print):
    """The gate report (the stop report). Refused while the gate is still running."""
    guard(prereg)
    cfg = prereg.config
    outcome = gate_outcome(prereg)
    if outcome["state"] == "pending":
        raise GateError(f"the gate has not reached a stop or its end; next: {outcome['next']}")
    started = time.perf_counter()
    harm = {}
    variants = ["base"] + (["retune"] if outcome["variant"] == "retune" else [])
    for v in variants:
        h = dict(H1=finished(_variant_root(prereg, "harm_critic", v) / "pooled.json", prereg), H2={})
        for a in cfg.alpha_ladder:
            r = finished(_variant_root(prereg, "harm_return", v) / f"alpha{a:g}" / "pooled.json", prereg)
            if r is not None:
                h["H2"][f"{a:g}"] = r
        h["H3"] = room_state(prereg, v)
        st = harm_state(prereg, v)
        if st["alpha"] is not None:
            h["H4"] = finished(_variant_root(prereg, "power", v) / f"alpha{st['alpha']:g}" / "pooled.json", prereg)
        harm[v] = h
    variant = outcome["variant"]
    cards, missing = {}, []
    for s in cfg.gate(variant):
        c = finished(prereg.root / "report" / variant / f"cards_seed{s}.json", prereg)
        if c is None:
            missing.append(s)
        else:
            cards[str(s)] = dict(sha256=file_sha256(prereg.root / "report" / variant / f"cards_seed{s}.json"),
                                 cards=c["cards"])
    if missing:
        raise GateError(f"compute the shift cards first: report --seeds {' '.join(map(str, missing))}")
    density = {}
    for v in VARIANTS:
        s0 = read_json(stage0_path(prereg.runs, v)) or {}
        for r in ((s0.get("record") or {}).get("per_seed") or []):
            note = r.get("density_check_note")
            if note:
                density[f"{v}/seed{r['rep']}"] = dict(passes=note["passes"], r_over_rho=note["r_over_rho"])
    h1 = harm.get(variant, {}).get("H1") or {}
    d4rl_notes = dict(
        density_check=density, g1_table=(h1.get("reported") or {}).get("d4rl_g1_table"),
        notes=["The D4RL density check's count marks are geometrically hard to reach when the 10-neighbour radius in "
               "(state, command) space is several gap radii; check this on Hopper's data before GPU time.",
               "The share of steps inside the removed ball misreports exposure: the shift card's unrecorded-throttle "
               "share and loss from unrecorded throttle are reported beside it.",
               "The target-noise confound sits at the action limits: the noise map ranks the affected rows well, so "
               "noise-map controls are needed on Hopper too.",
               "Where the densest-15% ball lands is a knife edge; check the landing on every seed."])
    n = outcome.get("N")
    rep = stage_record(prereg, "report", started, outcome=outcome, message=STOP_MESSAGES[outcome["reason"]],
                       bca_allowed=outcome["state"] == "ready", harm=harm, cards=cards, d4rl_notes=d4rl_notes,
                       stage0={v: prereg.payload["stage0"][v] for v in VARIANTS},
                       build_checks=prereg.payload["build_checks"], data=prereg.payload["data"],
                       bca_stage_budget=None if n is None else dict(
                           seeds=n, minutes_per_seed=BUDGET["bca_stage_minutes_per_seed"],
                           process_hours=n * BUDGET["bca_stage_minutes_per_seed"] / 60),
                       decision="Approve the BCA stage only if at least one shifted situation is room-certified and "
                                "N <= 20. Approval is typed by the user at a terminal; no agent message counts.",
                       about="the gate report: every harm-check value, the shift cards and the stop rule's outcome")
    d = prereg.root / "report"
    digest = write_json(d / "report.json", rep)
    (d / "report.md").write_text(report_markdown(json.loads(canonical(rep))), encoding="utf8")
    log(f"gate report: {rep['message']} (report.json sha256 {digest})")
    return rep


def approval_path(prereg):
    return prereg.root / APPROVAL_NAME


def stage_approve(prereg, input_fn=input, stdin=None, log=print):
    """The user's approval of the BCA stage: only at a terminal, by typing the first 16 characters of the gate
    report's SHA-256. Writes approval.json (with a sidecar) holding that SHA-256."""
    guard(prereg)
    stdin = sys.stdin if stdin is None else stdin
    path = prereg.root / "report" / "report.json"
    rep = finished(path, prereg)
    if rep is None:
        raise GateError("no verified gate report: run the report stage first")
    if not rep.get("bca_allowed"):
        raise GateError(f"the gate report does not allow the BCA stage: {rep.get('message')}")
    if approval_path(prereg).exists():
        raise GateError(f"{approval_path(prereg)} exists already")
    if not (hasattr(stdin, "isatty") and stdin.isatty()):
        raise GateError("refused: approval must be typed by the user at a terminal (standard input is not a terminal); "
                        "no agent message counts as approval")
    digest = file_sha256(path)
    log(f"Gate report: {path}\nSHA-256: {digest}\n{rep['message']}")
    typed = input_fn("To approve the BCA stage, type the first 16 characters of the report's SHA-256: ").strip()
    if typed != digest[:16]:
        raise GateError("not approved: the typed characters do not match")
    body = dict(kind="bca-shift-approval", report_path=str(path.resolve()), report_sha256=digest,
                prereg_sha256=prereg.digest, approved_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                statement="Approved by the user at a terminal after reading the gate report with this SHA-256.")
    write_json(approval_path(prereg), body)
    L.verify_approval(approval_path(prereg))
    log(f"approval written: {approval_path(prereg)}")
    return body


def load_approval(prereg):
    """The verified approval of this pre-registration (ApprovalError otherwise)."""
    approval = L.verify_approval(approval_path(prereg))
    if approval.prereg_sha256 != prereg.digest:
        raise L.ApprovalError(f"{L.BCA_REFUSAL} (the approval belongs to another pre-registration)")
    if Path(approval.report_path).resolve() != (prereg.root / "report" / "report.json").resolve():
        raise L.ApprovalError(f"{L.BCA_REFUSAL} (the approval names another report)")
    return approval


# ------------------------------------------------------------------------------------------------ the BCA stage

BCA_ARMS = ("plain", "bca", "flattened", "shuffled", "noise_map", "even_x1.25", "even_x2", "matched_even", "cloning",
            "full_plain", "full_bca", "no_stall_plain", "no_stall_bca", "band")
BCA_FAMILY = ("bca", "flattened", "shuffled", "noise_map", "full_bca", "no_stall_bca")   # arms that run BCA's code
BCA_OWN = ("bca", "full_bca", "no_stall_bca")   # BCA's own runs: a failure scores 0 points and is never rerun
PLACEMENT_CONTROLS = ("flattened", "shuffled", "noise_map")   # comparison arms that use BCA's doses blindly
# arms that must share data, initial parameters and batch-row stream (checked by hash in bca_scores)
SHARED_DATA_GROUPS = dict(
    gapped=("plain", "bca", "flattened", "shuffled", "noise_map", "even_x1.25", "even_x2", "matched_even", "cloning"),
    full=("full_plain", "full_bca"), no_stall=("no_stall_plain", "no_stall_bca"))
PLACEMENT_OF = dict(bca="own", flattened="flattened", shuffled="shuffled", noise_map="noise_map", full_bca="own",
                    no_stall_bca="own")
ARM_TEXT = dict(plain="plain TD3+BC", bca="TD3+BC with BCA", flattened="BCA's doses spread evenly in each batch",
                shuffled="BCA's doses shuffled within each batch", noise_map="BCA's doses placed by the noise map",
                even_x1_25="even caution x1.25", even_x2="even caution x2", matched_even="even caution matched in "
                "strength to BCA", cloning="behaviour cloning only", full_plain="plain TD3+BC on the full log",
                full_bca="BCA on the full log", no_stall_plain="plain TD3+BC, no-stall comparison world",
                no_stall_bca="BCA, no-stall comparison world", band="BCA's band on the frozen plain critic")


def bca_context(prereg, approval):
    """The approved BCA stage's world, alpha, certified situations and seeds; refuses without approval or when the
    gate did not end ready."""
    guard(prereg)
    L.require_approval(approval)
    if approval.prereg_sha256 != prereg.digest or Path(approval.report_path).resolve() != (
            prereg.root / "report" / "report.json").resolve():
        raise L.ApprovalError(f"{L.BCA_REFUSAL} (the approval belongs to another pre-registration or report)")
    outcome = gate_outcome(prereg)
    if outcome["state"] != "ready":
        raise GateError(f"the gate did not end ready for the BCA stage ({outcome['state']}: {outcome.get('reason')})")
    seeds = list(prereg.config.confirmation_seeds[:outcome["N"]])
    return outcome["variant"], outcome["alpha"], outcome["certified"], seeds


def _bca_dir(prereg, variant, alpha, rep):
    return prereg.root / "bca" / variant / f"alpha{alpha:g}" / f"seed{rep}"


def _arm_state_template(prereg, arm, rep, alpha):
    learn = prereg.config.learn
    if arm in BCA_FAMILY:
        return L.bca_initial_state(L.make_args(learn, rep, alpha), L.bca_config())[1]
    return L.agent_template(learn)


def arm_actor(prereg, variant, alpha, rep, arm, snapshot=100):
    """The actor params of a finished arm at a snapshot (100 or 50), or None when the arm failed."""
    d = _bca_dir(prereg, variant, alpha, rep) / arm
    rec = finished(d / "record.json", prereg)
    if rec is None:
        raise GateError(f"the BCA-stage arm {arm} of seed {rep} has not finished")
    if rec["failed"]:
        return None
    if arm == "cloning":
        return L.load_tree(d / "actor.msgpack", L.actor_template(), rec["result"]["file_sha256"])
    snap = next(s for s in rec["result"]["snapshots"] if int(round(100 * s["update"] / rec["result"]["updates"])) == snapshot)
    state = L.load_tree(d / f"snapshot_{snapshot:03d}.msgpack", _arm_state_template(prereg, arm, rep, alpha),
                        snap["file_sha256"])
    return state.native.actor.params if arm in BCA_FAMILY else state.actor.params


def arm_state(prereg, variant, alpha, rep, arm):
    d = _bca_dir(prereg, variant, alpha, rep) / arm
    rec = finished(d / "record.json", prereg)
    if rec is None or rec["failed"]:
        return None
    snap = rec["result"]["snapshots"][-1]
    return L.load_tree(d / "snapshot_100.msgpack", _arm_state_template(prereg, arm, rep, alpha), snap["file_sha256"])


def run_bca_arm(prereg, approval, variant, alpha, rep, arm, data, log=print):
    """One arm of the BCA stage (trained once; a failed comparison arm, the placement controls included, is rerun once
    with identical seeds; BCA's own runs are never rerun)."""
    L.require_approval(approval)
    learn = prereg.config.learn
    d = _bca_dir(prereg, variant, alpha, rep) / arm
    rec = finished(d / "record.json", prereg)
    if rec is not None:
        return rec
    st, tw = data["stall"], data.get("no_stall")
    pg, pf = st.prepared("gap", learn), st.prepared("none", learn)
    started = time.perf_counter()
    attempts, result = [], None
    for attempt in range(1 if arm in BCA_OWN else 2):
        if arm in ("plain", "even_x1.25", "even_x2", "full_plain", "no_stall_plain", "matched_even"):
            prep = pf if arm == "full_plain" else (tw.prepared("gap", learn) if arm == "no_stall_plain" else pg)
            table = None
            if arm in ("even_x1.25", "even_x2"):
                table = np.full(prep.n, 1.25 if arm == "even_x1.25" else 2.0)
            if arm == "matched_even":
                lock = verify_lock(_bca_dir(prereg, variant, alpha, rep) / "match_lock.json", prereg)
                if lock["multiplier"] is None:
                    raise GateError("the strength match found no multiplier (BCA's strength is outside the ladder)")
                table = np.full(prep.n, float(lock["multiplier"]))
            result, _, _ = L.cotrain(prep, learn, rep, alpha=alpha, table=table, out_dir=d, log=log,
                                     name=ARM_TEXT.get(arm.replace(".", "_"), arm))
        elif arm in BCA_FAMILY:
            prep = pf if arm == "full_bca" else (tw.prepared("gap", learn) if arm == "no_stall_bca" else pg)
            nm = st.arrays["noise_map"][st.train_position(pg.rows)] if arm == "noise_map" else None
            result, _, _ = L.bca_cotrain(prep, learn, rep, approval, placement=PLACEMENT_OF[arm], noise_map=nm,
                                         alpha=alpha, out_dir=d, log=log)
        elif arm == "cloning":
            params, result = L.clone(pg, learn, rep, steps=learn.clone_arm_steps, log=log)
            result["file_sha256"] = L.save_tree(d / "actor.msgpack", params)
        elif arm == "band":
            host = arm_state(prereg, variant, alpha, rep, "plain")
            if host is None:
                result = dict(failed=True, reason="the plain run failed")
            else:
                doses, result = L.bca_band(host, pg, learn, rep, approval, log=log)
                d.mkdir(parents=True, exist_ok=True)
                np.savez(d / "doses.npz", doses=doses)
                result.update(failed=not bool(np.all(np.isfinite(doses))), doses_sha256=DS.sha256_array(doses))
        else:
            raise ValueError(f"unknown arm {arm!r}")
        attempts.append(bool(result["failed"]))
        if not result["failed"]:
            break
    rec = stage_record(prereg, "bca_arm", started, variant=variant, alpha=alpha, rep=int(rep), arm=arm,
                       result=result, failed=bool(result["failed"]), attempts=attempts,
                       about=f"BCA stage, seed {rep}: {ARM_TEXT.get(arm.replace('.', '_'), arm)}")
    write_json(d / "record.json", rec)
    return rec


def bca_match(prereg, approval, variant, alpha, rep, data, log=print):
    """Blind strength matching of even caution to BCA (co-trained): S of BCA, x1.25 and x2 against plain on the
    evaluation states, one interpolation (glq_match.match_parallel, one round) whose run is the matched arm; the lock
    is written before any return of this seed is computed."""
    L.require_approval(approval)
    learn = prereg.config.learn
    lock = _bca_dir(prereg, variant, alpha, rep) / "match_lock.json"
    rec = finished(lock, prereg)
    if rec is not None:
        return rec
    started = time.perf_counter()
    pg = data["stall"].prepared("gap", learn)
    acts = {a: arm_actor(prereg, variant, alpha, rep, a) for a in ("plain", "bca", "even_x1.25", "even_x2")}
    if any(v is None for v in acts.values()):
        rec = stage_record(prereg, "bca_match", started, multiplier=None, failed=True,
                           reason="a run the match needs failed", rep=int(rep))
        write_json(lock, rec)
        return rec
    S = {a: L.strength(p, acts["plain"], pg.eval_obs) for a, p in acts.items()}
    if not (math.isfinite(S["bca"]) and S["bca"] > 0):
        rec = stage_record(prereg, "bca_match", started, strengths=S, multiplier=None, failed=True, rep=int(rep),
                           reason="BCA's run has no finite positive strength, so no even caution can be matched to it")
        write_json(lock, rec)
        return rec
    ran = {}

    def evaluate(proposals):
        out = []
        for _, knob in proposals:
            m = 1.0 + float(knob)
            tmp = _bca_dir(prereg, variant, alpha, rep) / "matched_even"
            table = np.full(pg.n, m)
            result, state, _ = L.cotrain(pg, learn, rep, alpha=alpha, table=table, out_dir=tmp, log=log,
                                         name=ARM_TEXT["matched_even"])
            ran[m] = result
            out.append(float("nan") if result["failed"] else L.strength(state.actor.params, acts["plain"], pg.eval_obs))
        return out

    states, rounds = L.M.match_parallel({"even": ([0.0, 0.25, 1.0], [0.0, S["even_x1.25"], S["even_x2"]])},
                                        {"bca": S["bca"]}, evaluate, learn.match_tol, 1, log=log)
    m = L.M.public(states[("even", "bca")])
    mult = None if m["knob"] is None else 1.0 + m["knob"]
    source = None if mult is None else ("plain" if m["knob"] == 0 else ("even_x1.25" if mult == 1.25 else
                                                                         ("even_x2" if mult == 2.0 else "matched_even")))
    rec = stage_record(prereg, "bca_match", started, rep=int(rep), strengths=S, match=m, multiplier=mult,
                       matched_arm=source, rounds=rounds, failed=mult is None,
                       glq_match_sha256=L.M.source_sha256(),
                       about="even caution matched in strength to BCA, chosen from strengths alone and locked before "
                             "any return of this seed is computed")
    write_json(lock, rec)
    if source == "matched_even" and mult in ran:
        d = _bca_dir(prereg, variant, alpha, rep) / "matched_even"
        write_json(d / "record.json", stage_record(prereg, "bca_arm", started, variant=variant, alpha=alpha,
                                                   rep=int(rep), arm="matched_even", result=ran[mult],
                                                   failed=bool(ran[mult]["failed"]), attempts=[ran[mult]["failed"]]))
    return rec


def stage_bca(prereg, approval, rep, arms=None, log=print):
    """Every arm of one confirmation seed, the blind match, then the scores (after the lock)."""
    variant, alpha, certified, seeds = bca_context(prereg, approval)
    if rep not in seeds:
        raise GateError(f"seed {rep} is not one of the first N = {len(seeds)} confirmation seeds {seeds}")
    cfg = prereg.config
    data = {"stall": load_seed(prereg.runs, cfg, variant, "stall", rep)}
    no_stall_seeds = list(cfg.no_stall_confirmation_seeds[:5])
    if rep in no_stall_seeds:
        data["no_stall"] = load_seed(prereg.runs, cfg, variant, "no_stall", rep)
    order = [a for a in BCA_ARMS if arms is None or a in arms]
    for arm in order:
        if arm in ("no_stall_plain", "no_stall_bca") and "no_stall" not in data:
            continue
        if arm == "matched_even":
            lock = bca_match(prereg, approval, variant, alpha, rep, data, log)
            src = lock.get("matched_arm")
            if src is None or src == "matched_even":
                if src is None:
                    d = _bca_dir(prereg, variant, alpha, rep) / "matched_even"
                    if finished(d / "record.json", prereg) is None:
                        write_json(d / "record.json", stage_record(
                            prereg, "bca_arm", time.perf_counter(), variant=variant, alpha=alpha, rep=int(rep),
                            arm="matched_even", result=dict(failed=True, reason="no multiplier matched"), failed=True,
                            attempts=[True]))
                else:
                    run_bca_arm(prereg, approval, variant, alpha, rep, arm, data, log)
            continue
        run_bca_arm(prereg, approval, variant, alpha, rep, arm, data, log)
    if arms is None:
        return bca_scores(prereg, approval, rep, log)
    return None


def shared_inputs(record_of):
    """What the arms of one confirmation seed must share, by hash (record_of(arm) gives an arm's record or None):
    one initial-parameter hash over every co-trained arm; within each SHARED_DATA_GROUPS group one data hash
    (prepared.describe()['sha256']; the gapped, full and no-stall logs differ by design), and one batch-row stream
    hash over the group's co-trained arms (cloning draws its own rows). Returns the hashes and a list of problems."""
    inits, data, streams, problems = {}, {}, {}, []
    for group, arms in SHARED_DATA_GROUPS.items():
        for arm in arms:
            r = record_of(arm)
            res = r.get("result") if isinstance(r, dict) else None
            if not isinstance(res, dict):
                continue
            if res.get("initial_sha256"):
                inits[arm] = res["initial_sha256"]
            if isinstance(res.get("data"), dict) and res["data"].get("sha256"):
                data.setdefault(group, {})[arm] = res["data"]["sha256"]
            if res.get("batch_stream_sha256"):
                streams.setdefault(group, {})[arm] = res["batch_stream_sha256"]
    if len(set(inits.values())) > 1:
        problems.append(f"initial parameters differ: {inits}")
    for kind, table in (("data", data), ("batch-row stream", streams)):
        for group, by_arm in table.items():
            if len(set(by_arm.values())) > 1:
                problems.append(f"{kind} differs within the {group} arms: {by_arm}")
    return dict(initial_sha256=inits, data_sha256=data, batch_stream_sha256=streams, problems=problems)


def bca_scores(prereg, approval, rep, log=print):
    """Exact points of every arm of one seed, computed only after its strength match is locked."""
    variant, alpha, certified, seeds = bca_context(prereg, approval)
    cfg = prereg.config
    d = _bca_dir(prereg, variant, alpha, rep)
    out = d / "scores.json"
    done = finished(out, prereg)
    if done is not None:
        return done
    lock = verify_lock(d / "match_lock.json", prereg)
    started = time.perf_counter()
    st = load_seed(prereg.runs, cfg, variant, "stall", rep)
    pg = st.prepared("gap", cfg.learn)
    tw = load_seed(prereg.runs, cfg, variant, "no_stall", rep) if rep in cfg.no_stall_confirmation_seeds[:5] else None
    an_s = anchors_of(prereg, variant, "stall")
    scores, failed = {}, {}
    for arm in BCA_ARMS:
        if arm == "band" or (arm.startswith("no_stall") and tw is None):
            continue
        a = "matched_even"
        src = arm if arm != a else (lock.get("matched_arm") or a)
        rec = finished(d / src / "record.json", prereg)
        if rec is None:
            raise GateError(f"seed {rep}: the arm {src} has not finished (the scorer refuses while a run is missing)")
        failed[arm] = bool(rec["failed"])
        scores[arm] = {}
        for snap in (100, 50) if src != "cloning" else (100,):
            p = None if rec["failed"] else arm_actor(prereg, variant, alpha, rep, src, snap)
            prep = tw.prepared("gap", cfg.learn) if arm.startswith("no_stall") else pg
            an = anchors_of(prereg, variant, "no_stall") if arm.startswith("no_stall") else an_s
            scores[arm][str(snap)] = None if p is None else L.evaluate_actor(p, prep, an, n=cfg.bank_size)
    shared = shared_inputs(lambda arm: finished(d / arm / "record.json", prereg))
    if shared["problems"]:
        raise GateError(f"seed {rep}: the BCA-stage arms do not share what they must: {'; '.join(shared['problems'])}")
    inits = shared["initial_sha256"]
    rec = stage_record(prereg, "bca_scores", started, variant=variant, alpha=alpha, rep=int(rep), lock_sha256=
                       file_sha256(d / "match_lock.json"), scores=scores, failed=failed,
                       initial_parameters_sha256=inits, same_initial_parameters=True,
                       data_sha256=shared["data_sha256"], batch_stream_sha256=shared["batch_stream_sha256"],
                       about="exact points of every BCA-stage arm of one seed (final and 50% snapshots)")
    write_json(out, rec)
    return rec


# ------------------------------------------------------------------------------------------------ detection (B1)

def _signals(state, prepared, args, prepared_rows_states, chunk=8192):
    """Comparator signals at the kept training rows: critic disagreement |Q1 - Q2|, critic size |Q1| and the raw TD
    residual |target - Q1| (TD3's smoothed target with a fixed key) of the run's final critic, and data thinness (the
    distance to the 10th nearest other training row in standardised (state, command) space)."""
    from scipy.spatial import cKDTree

    import algorithms.td3_bc_bca as P

    q = L.critic_rows(state.native.critic.params, prepared.train.obs, prepared.actions)[0]
    models = (L.ACTOR, L.CRITIC)
    tgt = []
    target = jax.jit(lambda nat, b, k: P.native_target(args, models, nat, b, k))
    key = jax.random.PRNGKey(TAG_BOOT)
    for i in range(0, prepared.n, chunk):
        b = jax.tree_util.tree_map(lambda x: x[i:i + chunk], prepared.train)
        tgt.append(np.asarray(target(state.native, b, jax.random.fold_in(key, i)), np.float64))
    tgt = np.concatenate(tgt)
    xa = np.concatenate([prepared_rows_states, prepared.actions.astype(np.float64)], 1)
    z = (xa - xa.mean(0)) / xa.std(0)
    thin = cKDTree(z).query(z, k=11)[0][:, -1]
    return dict(critic_disagreement=np.abs(q[:, 0] - q[:, 1]), critic_size=np.abs(q[:, 0]),
                td_residual=np.abs(tgt - q[:, 0]), data_thinness=thin)


def detect_seed(prereg, approval, rep, log=print):
    """B1 for one seed: BCA's doses at its final snapshot against the exact over-preference of its own critic and
    actor, the comparators, the band on the frozen plain critic, the concentration ratios and the state maps."""
    variant, alpha, certified, seeds = bca_context(prereg, approval)
    cfg, learn = prereg.config, prereg.config.learn
    out = prereg.root / "detect" / variant / f"alpha{alpha:g}" / f"seed{rep}"
    done = finished(out / "values.json", prereg)
    if done is not None:
        return done
    started = time.perf_counter()
    st = load_seed(prereg.runs, cfg, variant, "stall", rep)
    pg, pf = st.prepared("gap", learn), st.prepared("none", learn)
    pos = st.train_position(pg.rows)
    nm, ms = st.arrays["noise_map"][pos], st.arrays["removed_share"][pos]
    tie = np.random.default_rng([TC.DATA_SEED, rep, TAG_CARD_TIE]).permutation(pg.n)
    state = arm_state(prereg, variant, alpha, rep, "bca")
    values = dict(failed=state is None)
    arrays = {}
    if state is not None:
        doses = L.bca_doses(state, pg, approval)
        op = L.over_preference_rows(state.native.actor.params, state.native.critic.params, pg,
                                    horizon=cfg.truth_horizon, x64=True)
        D = op["D"]
        values.update(DT.detection_seed(doses, D, nm, pg.region, tie, PASS_MARKS["B1"]["top_share"]))
        args = L.make_args(learn, rep, alpha)
        values["comparators"] = DT.comparator_scores(doses, D, dict(noise_map=nm, **_signals(state, pg, args, pg.states)),
                                                     tie, PASS_MARKS["B1"]["top_share"])
        values["state_map"] = DT.state_map(ms, dict(over_preference=D, dose=doses, noise_map=nm), tie)
        arrays.update(doses=doses, over_preference=D)
        full = arm_state(prereg, variant, alpha, rep, "full_bca")
        if full is not None:
            c_full = DT.concentration(L.bca_doses(full, pf, approval), pf.region)
            values["concentration_full_log"] = c_full
            values["concentration_ratio"] = DT.safe_ratio(values["concentration"], c_full)
        if rep in cfg.no_stall_confirmation_seeds[:5]:
            other = arm_state(prereg, variant, alpha, rep, "no_stall_bca")
            if other is not None:
                tw = load_seed(prereg.runs, cfg, variant, "no_stall", rep)
                ptw = tw.prepared("gap", learn)
                c_no_stall = DT.concentration(L.bca_doses(other, ptw, approval), ptw.region)
                values["concentration_no_stall"] = c_no_stall
                values["stall_over_no_stall_concentration_reported"] = DT.safe_ratio(values["concentration"], c_no_stall)
    band_rec = finished(_bca_dir(prereg, variant, alpha, rep) / "band" / "record.json", prereg)
    host = arm_state(prereg, variant, alpha, rep, "plain")
    if band_rec is not None and not band_rec["failed"] and host is not None:
        with np.load(_bca_dir(prereg, variant, alpha, rep) / "band" / "doses.npz") as z:
            band = z["doses"]
        Dh = L.over_preference_rows(host.actor.params, host.critic.params, pg, horizon=cfg.truth_horizon,
                                    x64=True)["D"]
        values["band_on_frozen_plain_critic"] = DT.detection_seed(band, Dh, nm, pg.region, tie,
                                                                   PASS_MARKS["B1"]["top_share"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "rows.npz", **arrays)
    rec = stage_record(prereg, "detect_seed", started, variant=variant, alpha=alpha, rep=int(rep), values=values,
                       rows_sha256={k: DS.sha256_array(v) for k, v in arrays.items()},
                       about="detection for one seed: does BCA's dose rank the rows by the exact harm?")
    write_json(out / "values.json", rec)
    return rec


def detect_pooled(prereg, approval, log=print):
    variant, alpha, certified, seeds = bca_context(prereg, approval)
    started = time.perf_counter()
    per = []
    for s in seeds:
        r = finished(prereg.root / "detect" / variant / f"alpha{alpha:g}" / f"seed{s}" / "values.json", prereg)
        if r is None:
            raise GateError(f"detect has not finished seed {s}")
        per.append(r["values"])
    fam = DT.detection_family(per, PASS_MARKS["B1"])
    comps = sorted({k for v in per for k in v.get("comparators", {})})
    reported = dict(
        placement_efficiency=DT.summary([v.get("placement_efficiency") for v in per]),
        dose_interquartile_range=DT.summary([v.get("dose_interquartile_range") for v in per]),
        comparators={c: dict(d1_auc=DT.mean([v.get("comparators", {}).get(c, {}).get("d1_auc") for v in per]),
                             placement_efficiency=DT.mean([v.get("comparators", {}).get(c, {}).get("placement_efficiency")
                                                           for v in per])) for c in comps},
        band_on_frozen_plain_critic=dict(d1_auc=DT.mean([(v.get("band_on_frozen_plain_critic") or {}).get("d1_auc")
                                                         for v in per])),
        stall_over_no_stall_concentration=DT.mean([v.get("stall_over_no_stall_concentration_reported") for v in per]),
        note="the stall / no-stall concentration ratio is reported, never gated: kept rows obey identical rules in "
             "both worlds, so a method that learns only from the gapped log must give about 1")
    rec = stage_record(prereg, "detect", started, variant=variant, alpha=alpha, seeds=seeds, family=fam,
                       passed=fam["passed"], reported=reported, holm_family=HOLM_FAMILIES["detection"],
                       about="detection D1-D3 pooled over the confirmation seeds (one family under Holm's correction for "
                             "testing several things at once)")
    write_json(prereg.root / "detect" / variant / f"alpha{alpha:g}" / "pooled.json", rec)
    return rec


# ------------------------------------------------------------------------------------------------ policy and verdict

def _points_table(prereg, variant, alpha, seeds):
    """{seed: {arm: {situation: points}}} with the failed-run rules: a failed run of BCA itself (bca, full_bca,
    no_stall_bca) scores 0 points (the recording controller's level) everywhere; a failed comparison arm (after its one
    rerun), the placement controls (flattened, shuffled, noise_map) included, scores None and every difference with
    it is set to 0. So a failed run always counts against BCA, never for it."""
    out = {}
    for s in seeds:
        r = finished(_bca_dir(prereg, variant, alpha, s) / "scores.json", prereg)
        if r is None:
            raise GateError(f"the BCA stage has not scored seed {s} (the scorer refuses while a run is missing)")
        row = {}
        for arm, by_snap in r["scores"].items():
            ev = by_snap.get("100")
            if ev is None:
                row[arm] = {sit: (0.0 if arm in BCA_OWN else None) for sit in SITUATIONS}
                row[arm]["_loss"] = {sit: None for sit in SITUATIONS}
            else:
                row[arm] = {sit: ev[sit]["points"] for sit in SITUATIONS}
                row[arm]["_loss"] = {sit: ev[sit]["loss_from_unrecorded_throttle"] for sit in SITUATIONS}
        out[s] = row
    return out


def _differences(table, a, b, sit):
    out = []
    for s, row in table.items():
        x, y = row.get(a, {}).get(sit), row.get(b, {}).get(sit)
        out.append(0.0 if x is None or y is None else x - y)
    return out


def decide(detection_passed, level1, level2, p2, p3, p4, harm_present, half_width_ok, cost, noise_rider):
    """The verdict code and sentence from the marks (VERDICT_ORDER), mechanically."""
    if cost or not p2 or not p4:
        return "V5", VERDICTS["V5"]
    if not harm_present or not half_width_ok:
        return "V6", VERDICTS["V6"]
    l1 = [s for s, ok in level1.items() if ok]
    placed = any(level2.get(s) for s in l1)
    rider = f"; {VERDICTS['V1_noise_rider']}" if noise_rider else ""
    if detection_passed and placed and p2 and p3 and p4:
        return "V1", VERDICTS["V1"] + rider
    if placed and not detection_passed:
        return "V7", VERDICTS["V7"] + ("" if p3 else f"; {VERDICTS['V7_recording_rider']}")
    if placed and not p3:
        return "V8", VERDICTS["V8"] + rider
    if l1 and not placed:
        return "V2", VERDICTS["V2"]
    if detection_passed and not l1:
        return "V3", VERDICTS["V3"]
    return "V4", VERDICTS["V4"]


def stage_policy(prereg, approval, log=print):
    """B2: level 1 and level 2 (Holm), P2-P4, harm present, and every reported difference."""
    variant, alpha, certified, seeds = bca_context(prereg, approval)
    m = PASS_MARKS["B2"]
    started = time.perf_counter()
    table = _points_table(prereg, variant, alpha, seeds)
    level1_d = {s: _differences(table, "bca", "plain", s) for s in certified}
    adj1 = DT.holm({s: DT.p_greater(v) for s, v in level1_d.items()})
    level1 = {s: bool(adj1[s] < m["holm_alpha"] and DT.mean(v) >= m["level1_min"]) for s, v in level1_d.items()}
    tests = {f"{s}:{b}": _differences(table, "bca", b, s) for s in certified if level1[s]
             for b in ("flattened", "shuffled", "matched_even")}
    adj2 = DT.holm({k: DT.p_greater(v) for k, v in tests.items()})
    level2 = {s: bool(level1[s] and all(adj2[f"{s}:{b}"] < m["holm_alpha"] and DT.mean(tests[f"{s}:{b}"]) > 0
                                        for b in ("flattened", "shuffled", "matched_even"))) for s in certified}
    p2v = _differences(table, "bca", "plain", "recorded")
    p2 = bool((DT.t_lower(p2v) if DT.t_lower(p2v) is not None else -math.inf) > m["p2_lower_gt"])
    p3v = _differences(table, "bca", "cloning", "recorded")
    p3 = bool(DT.mean(p3v) >= m["p3_min"] and (DT.t_lower(p3v) or -math.inf) > m["p3_lower_gt"])
    p4v = {s: _differences(table, "full_bca", "full_plain", s) for s in ["recorded"] + list(certified)}
    p4 = bool(all((DT.t_lower(v) if DT.t_lower(v) is not None else -math.inf) > m["p4_lower_gt"] for v in p4v.values()))
    harm = {s: DT.mean([row["plain"]["_loss"][s] or 0.0 for row in table.values()]) for s in certified}
    harm_present = bool(all(v is not None and v >= m["harm_present_min"] for v in harm.values()))
    hw = {s: DT.half_width(v) for s, v in level1_d.items()}
    half_ok = bool(all(v is not None and v <= m["inconclusive_half_width_max"] for v in hw.values()))
    cost = bool(any(DT.mean(v) <= m["cost_mean_max"] and (DT.t_upper(v) if DT.t_upper(v) is not None else math.inf) < 0
                    for v in level1_d.values()))
    noise = {s: DT.mean(_differences(table, "bca", "noise_map", s)) for s in certified}
    rider_sits = [s for s in certified if level2[s]] or [s for s in certified if level1[s]]
    noise_rider = bool(any(noise[s] is not None and noise[s] <= m["noise_map_within"] for s in rider_sits))
    best_even = {s: DT.mean([row["bca"][s] - max(v for v in (row.get(e, {}).get(s) for e in
                                                             ("even_x1.25", "even_x2", "matched_even")) if v is not None)
                             if any(row.get(e, {}).get(s) is not None for e in ("even_x1.25", "even_x2", "matched_even"))
                             else 0.0 for row in table.values()]) for s in SITUATIONS}
    price = [row["no_stall_bca"]["recorded"] - row["no_stall_plain"]["recorded"] for row in table.values()
            if row.get("no_stall_bca", {}).get("recorded") is not None and row.get("no_stall_plain", {}).get("recorded")
            is not None]
    det = finished(prereg.root / "detect" / variant / f"alpha{alpha:g}" / "pooled.json", prereg)
    if det is None:
        raise GateError("run detect --pooled first")
    code, text = decide(det["passed"], level1, level2, p2, p3, p4, harm_present, half_ok, cost, noise_rider)
    rec = stage_record(prereg, "policy", started, variant=variant, alpha=alpha, seeds=seeds, certified=certified,
                       level1=dict(differences=level1_d, p_holm=adj1, passed=level1, family=HOLM_FAMILIES["policy_level_1"]),
                       level2=dict(differences=tests, p_holm=adj2, passed=level2, family=HOLM_FAMILIES["policy_level_2"]),
                       P2=dict(values=p2v, lower=DT.t_lower(p2v), passed=p2),
                       P3=dict(values=p3v, mean=DT.mean(p3v), lower=DT.t_lower(p3v), passed=p3),
                       P4=dict(values=p4v, passed=p4), harm_present=dict(loss_points=harm, passed=harm_present),
                       half_widths=hw, cost=cost, noise_map_rider=noise_rider,
                       reported=dict(bca_minus_noise_map=noise, bca_minus_best_even=best_even,
                                     no_stall_price_recorded=DT.summary(price),
                                     points={str(s): row for s, row in table.items()}),
                       verdict=dict(code=code, text=text, detection_passed=det["passed"]),
                       about="policy (B2): does BCA's caution help out of distribution, beyond blind caution, while "
                             "still improving on the recording?")
    write_json(prereg.root / "policy" / variant / f"alpha{alpha:g}" / "pooled.json", rec)
    return rec


def stage_verdict(prereg, approval, log=print):
    """The verdict table: every mark with its value, in plain words, and the fixed verdict sentence."""
    variant, alpha, certified, seeds = bca_context(prereg, approval)
    started = time.perf_counter()
    det = finished(prereg.root / "detect" / variant / f"alpha{alpha:g}" / "pooled.json", prereg)
    pol = finished(prereg.root / "policy" / variant / f"alpha{alpha:g}" / "pooled.json", prereg)
    if det is None or pol is None:
        raise GateError("run detect --pooled and policy first")
    rows = [("D1", "BCA's dose ranks the rows with the largest exact over-preference first (how often such a row "
                   "gets a larger dose than another row; 0.5 is chance)",
             det["family"]["D1"]["mean"], det["family"]["D1"]["passed"]),
            ("D2", "BCA puts more caution in the full-throttle region on the gapped log than on the full log",
             det["family"]["D2"]["ratio_seed_mean"], det["family"]["D2"]["passed"]),
            ("D3", "BCA's dose follows the exact over-preference beyond target noise and the region",
             det["family"]["D3"]["mean"], det["family"]["D3"]["passed"])]
    for s in certified:
        rows.append((f"level 1, {LABEL[s]}", "BCA minus plain TD3+BC (points)", DT.mean(pol["level1"]["differences"][s]),
                     pol["level1"]["passed"][s]))
        rows.append((f"level 2, {LABEL[s]}", "BCA beats the same doses spread evenly, shuffled, and even caution of "
                                             "the same strength", None, pol["level2"]["passed"][s]))
    rows += [("P2", "no loss in recorded conditions (lower bound of BCA minus plain)", pol["P2"]["lower"],
              pol["P2"]["passed"]),
             ("P3", "still improves on the recording: BCA minus behaviour cloning (points)", pol["P3"]["mean"],
              pol["P3"]["passed"]),
             ("P4", "no harm without the shift (full log)", None, pol["P4"]["passed"]),
             ("harm present", "the plain run still loses at least 3 points to unrecorded throttle on these seeds",
              None, pol["harm_present"]["passed"])]
    md = ["# ThrusterChain shift test: verdict", "", f"**{pol['verdict']['code']}: {pol['verdict']['text']}.**", "",
          "| Mark | What it asks | Measured | Passed |", "|---|---|---|---|"]
    md += [f"| {a} | {b} | {_fmt(c)} | {_fmt(d)} |" for a, b, c, d in rows]
    md += ["", f"Seeds: {', '.join(map(str, seeds))}. Every arm, seed and situation is listed in policy/pooled.json.",
           "", VERDICT_ORDER, ""]
    rec = stage_record(prereg, "verdict", started, variant=variant, alpha=alpha, seeds=seeds,
                       verdict=pol["verdict"], table=[dict(mark=a, asks=b, measured=c, passed=d) for a, b, c, d in rows],
                       about="the verdict, applied mechanically from the pre-registered marks and wording")
    write_json(prereg.root / "verdict.json", rec)
    (prereg.root / "verdict.md").write_text("\n".join(md), encoding="utf8")
    log(f"verdict: {pol['verdict']['code']}: {pol['verdict']['text']}")
    return rec


# ------------------------------------------------------------------------------------------------ build checks (S0)

def _trees_equal(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def _max_diff(a, b):
    return max(float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def learner_checks(updates=1000, log=print):
    """The learners' build checks on a small log (seed 900, 40 + 10 + 10 episodes); no return is computed."""
    import algorithms.td3_bc as BASE
    import algorithms.td3_bc_bca as P

    learn = replace(SMOKE_LEARN, updates=updates, chunk=updates // 2)
    lg = DS.generate(900, "stall", sizes=(40, 10, 10))
    gap, _ = SH.define_gap(lg)
    rg = SH.make_removal(lg, "gap", gap)
    pg = L.prepare(lg, rg, L.normalization(lg, rg), learn)
    out = {}
    args = L.make_args(learn, 0)
    _, state, _ = L.cotrain(pg, learn, 0)
    rng, s0, (actor, critic) = BASE.initialize(args, TC.DX, TC.DA, 1.0)
    step = BASE.make_train_step(args, actor.apply, critic.apply, pg.train, 1.0)
    direct = jax.jit(lambda r: jax.lax.scan(step, r, None, length=updates))((rng, s0, jnp.int32(0)))[0][1]
    out["plain_equals_make_train_step_bitwise"] = _trees_equal(state, direct)
    _, even1, _ = L.cotrain(pg, learn, 0, table=np.ones(pg.n))
    out["even_x1_minus_plain_max"] = _max_diff(even1, state)
    batch = jax.tree_util.tree_map(lambda x: x[:256], pg.train)
    new = jax.jit(lambda s, b: BASE.td3_bc_update(args, actor.apply, critic.apply, s, b, 2, jax.random.PRNGKey(3),
                                                  1.0)[0])(s0, batch)
    mine = jax.jit(lambda a, c, b: L.actor_step(args, a, c, b)[0])(s0.actor, new.critic.params, batch)
    out["actor_step_equals_td3_bc_update_bitwise"] = _trees_equal(mine.params, new.actor.params)
    rec, fstate = L.fixed_critic(pg, "stall_aware", learn, 0, updates=200)
    fixed = L.FixedController("stall_aware", pg.physics, pg.obs_mean, pg.obs_std)
    out["fixed_controller_stays_fixed"] = _trees_equal(fstate.actor.params, fixed.params())
    clone, _ = L.clone(pg, learn, 0, steps=100, chunk=100)
    critic_before = jax.tree_util.tree_map(lambda x: np.array(x), fstate.critic.params)
    ao = L.ActorOnly(pg, clone, learn, 0, steps=100, chunk=50)
    ao.run([L.Arm("a"), L.Arm("b", "even", 2.0)], [fstate.critic.params])
    out["frozen_critic_unchanged"] = _trees_equal(critic_before, fstate.critic.params)
    pol = L.mlp_policy(clone, pg)
    x, a = pg.states[:50], pg.actions[:50].astype(np.float64)
    out["exact_q_learned_actor_max"] = float(np.max(np.abs(L.exact_q(clone, pg, x, a, chunk=64)
                                                           - TR.q_value_jax(pol, x, a))))
    attr = TR.attribution(pol, RC.stall_aware(), TC.bank_starts("strong_current", 20))
    out["attribution_sum_max"] = attr["max_sum_error"]
    K = RC.textbook_gain(TC.NO_STALL)
    lin = RC.LinearController(K, None, "textbook gain, unclipped")
    xs = TC.draw_starts(30, np.random.default_rng(5), dict(current=(0.0, 0.3), wave=(0.0, 0.2), kick=0.2))
    a0 = lin(xs)
    g = TR.action_slope(lin, xs, a0, TC.NO_STALL, 400)
    h = 1e-4
    fd = np.stack([(TR.q_value(lin, xs, a0 + h * np.eye(TC.DA)[j], TC.NO_STALL, 400)
                    - TR.q_value(lin, xs, a0 - h * np.eye(TC.DA)[j], TC.NO_STALL, 400)) / (2 * h)
                   for j in range(TC.DA)], 1)
    out["slope_vs_central_differences_max"] = float(np.max(np.abs(g - fd)))
    cfg_b = L.bca_config()
    rng_b, bst, models = L.bca_initial_state(args, cfg_b)
    j1 = jax.make_jaxpr(lambda st, b, r, it: P.update(args, cfg_b, models, st, b, it, r, 1.0))(bst, batch, rng_b,
                                                                                               jnp.int32(2))
    j2 = jax.make_jaxpr(lambda st, b, r, it: L.bca_update(args, cfg_b, models, st, b, it, r, 1.0))(bst, batch, rng_b,
                                                                                                   jnp.int32(2))
    out["bca_own_placement_traces_identically"] = str(j1) == str(j2)
    v = jnp.asarray(np.random.default_rng(1).uniform(1, 1.5, 256), jnp.float32)
    sig = jnp.asarray(np.random.default_rng(2).standard_normal(256), jnp.float32)
    out["placements_keep_the_doses"] = bool(
        np.allclose(np.sort(np.asarray(L.place(v, "shuffled", jax.random.PRNGKey(0)))), np.sort(np.asarray(v)))
        and np.allclose(np.sort(np.asarray(L.place(v, "noise_map", None, sig))), np.sort(np.asarray(v)))
        and abs(float(np.mean(L.place(v, "flattened", None))) - float(np.mean(v))) < 1e-6)
    m = PASS_MARKS["S0"]
    out["passed"] = bool(out["plain_equals_make_train_step_bitwise"] and out["even_x1_minus_plain_max"] < 1e-5
                         and out["actor_step_equals_td3_bc_update_bitwise"] and out["fixed_controller_stays_fixed"]
                         and out["frozen_critic_unchanged"]
                         and out["exact_q_learned_actor_max"] <= m["exact_q_learned_actor_max"]
                         and out["attribution_sum_max"] <= m["attribution_sum_max"]
                         and out["slope_vs_central_differences_max"] <= m["slope_max"]
                         and out["bca_own_placement_traces_identically"] and out["placements_keep_the_doses"])
    return out


def run_unit_suites(suites=UNIT_SUITES, log=print):
    """Run the unit suites in a subprocess; count tests, failures and errors."""
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "unittest", *suites], cwd=ROOT, env=env, capture_output=True, text=True)
    tail = p.stderr.strip().splitlines()[-6:]
    ran = next((int(line.split()[1]) for line in p.stderr.splitlines() if line.startswith("Ran ")), None)
    failures = sum(int(part.split("=")[1].rstrip(",)")) for line in p.stderr.splitlines() if line.startswith("FAILED")
                   for part in line.split("(")[1].split() if "=" in part)
    return dict(suites=list(suites), ran=ran, failures=failures if p.returncode else 0, returncode=p.returncode,
                seconds=time.perf_counter() - t0, tail=tail)


def timing_smoke(updates=2000):
    """Plain TD3+BC and fixed-controller critic updates on random data of the real size (42,500 rows): minutes per
    100,000 updates against the budget. No return is computed."""
    from runtime.networks import Transition

    rng = np.random.default_rng(0)
    n = 42_500
    train = Transition(jnp.asarray(rng.standard_normal((n, TC.DX)), jnp.float32),
                       jnp.asarray(rng.uniform(-1, 1, (n, TC.DA)), jnp.float32),
                       jnp.asarray(-rng.uniform(0, 2, n), jnp.float32),
                       jnp.asarray(rng.standard_normal((n, TC.DX)), jnp.float32), jnp.zeros(n, jnp.float32))
    learn = replace(L.LearnSettings(), updates=updates, chunk=updates)
    fake = L.Prepared(rep=0, world="stall", removal="gap", physics=TC.STALL, spec=RC.DEFAULT_SPEC, rows=np.arange(n),
                      states=np.zeros((n, TC.DX)), actions=np.zeros((n, TC.DA), np.float32), region=np.zeros(n, bool),
                      train=train, bank_rows=np.arange(10), bank=train, obs_mean=np.zeros(TC.DX, np.float32),
                      obs_std=np.ones(TC.DX, np.float32), eval_rows=np.arange(10), eval_obs=train.obs[:10])
    out = {}
    for name, fn in (("plain_host", lambda: L.cotrain(fake, learn, 0)),
                     ("fixed_controller_critic", lambda: L.fixed_critic(fake, "stall_aware", learn, 0))):
        fn()
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        out[name] = dict(seconds=dt, minutes_per_100k=dt * 100_000 / updates / 60)
    budget = BUDGET["minutes"]
    out["restate_budget"] = bool(out["plain_host"]["minutes_per_100k"] > PASS_MARKS["S0"]["budget_restate_factor"]
                                 * budget["plain_host"] or out["fixed_controller_critic"]["minutes_per_100k"]
                                 > PASS_MARKS["S0"]["budget_restate_factor"] * budget["fixed_controller_critic"])
    return out


def stage_build_checks(runs, cfg, unittests=True, adversarial_steps=10_000, log=print):
    """S0 (module docstring). Writes runs/shift/pre/build_checks.json."""
    started = time.perf_counter()
    before = dict(earlier_gate=earlier_gate_state(), environment=environment_state())
    log("build checks: environment (check_env, replay, bound, truth anchors)")
    env = S0.build_checks(replay_seed=cfg.calibration_seeds[0], adversarial_steps=adversarial_steps)
    log("build checks: learners")
    learners = learner_checks(log=log)
    units = None
    if unittests:
        log("build checks: unit suites")
        units = run_unit_suites(log=log)
    log("build checks: timing smoke on random data")
    timing = timing_smoke()
    after = dict(earlier_gate=earlier_gate_state(), environment=environment_state())
    unchanged = before == after
    passed = bool(env["passed"] and learners["passed"] and units is not None and units["failures"] == 0
                  and units["returncode"] == 0 and unchanged)
    out = dict(kind="shift-build-checks", about="S0: the environment, the truth and the learners are exact before "
               "any number is trusted; nothing disturbed the earlier gate or the shared Python environment",
               environment_checks=env, learner_checks=learners, unit_suites=units, timing=timing,
               earlier_gate_and_environment=dict(before=before, after=after, unchanged=unchanged),
               passed=passed, seconds=time.perf_counter() - started, preset=cfg.preset)
    write_json(pre_dir(runs) / "build_checks.json", out)
    log(f"build checks: {'pass' if passed else 'FAIL'}")
    return out


# ------------------------------------------------------------------------------------------------ CLI

def build_parser():
    p = argparse.ArgumentParser(description="ThrusterChain shift test: the pre-registered stages (module docstring).")
    p.add_argument("stage", choices=("plan", "build_checks", "stage0", "data", "prereg", "harm_critic", "harm_return",
                                     "room", "power", "report", "approve", "bca", "detect", "policy", "verdict"))
    p.add_argument("--runs", default=str(RUNS_DEFAULT), help="output root (default runs/shift)")
    p.add_argument("--preset", choices=sorted(PRESETS), default="real",
                   help="sizes before the pre-registration (real; smoke only for checking the code in a temp dir)")
    p.add_argument("--variant", default="auto", help="base, retune or auto (the stop rule decides)")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--pooled", action="store_true", help="the pooled marks instead of per-seed work")
    p.add_argument("--alpha", default="auto", help="a rung of the alpha ladder, or auto (the next rung due)")
    p.add_argument("--roles", nargs="+", choices=ROLES, default=list(ROLES))
    p.add_argument("--worlds", nargs="+", choices=sorted(TC.WORLDS), default=["stall", "no_stall"])
    p.add_argument("--arms", nargs="+", choices=BCA_ARMS)
    p.add_argument("--force", action="store_true", help="data stage only: redo existing seed records")
    p.add_argument("--no-unittests", action="store_true", help="build checks without the unit suites (then they "
                                                                "cannot pass)")
    p.add_argument("--no-grid", action="store_true", help="stage0 without the fallback grid search")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    runs = refuse_protected(a.runs)
    try:
        if a.stage in ("plan", "build_checks", "stage0", "data", "prereg"):
            cfg = PRESETS[a.preset]
            if runs.resolve() == RUNS_DEFAULT.resolve() and cfg.preset != "real":
                raise GateError("runs/shift takes the real preset only; use --runs <temporary directory>")
            variant = "base" if a.variant == "auto" else a.variant
            if a.stage == "plan":
                print("\n".join(plan(cfg)))
            elif a.stage == "build_checks":
                stage_build_checks(runs, cfg, unittests=not a.no_unittests)
            elif a.stage == "stage0":
                stage_stage0(runs, cfg, variant, grid=not a.no_grid)
            elif a.stage == "data":
                print(json.dumps(stage_data(runs, cfg, variant, tuple(a.roles), tuple(a.worlds), a.seeds, a.force),
                                 indent=1))
            else:
                write_prereg(runs, cfg)
            return 0
        prereg = load_prereg(runs)
        guard(prereg)
        loose = unhashed_imports(prereg.code_root)
        if loose:
            raise PreregError(f"imported repository files outside the hashed lists: {loose}")
        cfg = prereg.config
        if a.stage in ("harm_critic", "harm_return", "room", "power"):
            variant = resolve_variant(prereg, a.variant)
            seeds = a.seeds or list(cfg.gate(variant))
            if a.stage == "harm_critic":
                if a.pooled:
                    h1_pooled(prereg, variant)
                else:
                    for s in seeds:
                        h1_seed(prereg, variant, s)
            elif a.stage == "harm_return":
                if a.alpha == "auto":
                    st = harm_state(prereg, variant)
                    if st.get("next_alpha") is None:
                        raise GateError("no H2 run is due (" + (st.get("reason") or st.get("next")
                                                                 or f"alpha chosen: {st['alpha']:g}") + ")")
                    alpha = float(st["next_alpha"])
                else:
                    alpha = float(a.alpha)
                if a.pooled:
                    h2_pooled(prereg, variant, alpha)
                else:
                    for s in seeds:
                        h2_seed(prereg, variant, alpha, s)
            elif a.stage == "room":
                if a.pooled:
                    h3_pooled(prereg, variant)
                else:
                    for s in seeds:
                        h3_seed(prereg, variant, s)
            else:
                if a.pooled:
                    h4_pooled(prereg, variant)
                else:
                    for s in seeds:
                        h4_seed(prereg, variant, s)
        elif a.stage == "report":
            if a.seeds:
                for s in a.seeds:
                    report_cards(prereg, s)
            else:
                stage_report(prereg)
        elif a.stage == "approve":
            stage_approve(prereg)
        else:
            approval = load_approval(prereg)
            if a.stage == "bca":
                _, _, _, seeds = bca_context(prereg, approval)
                for s in a.seeds or seeds:
                    stage_bca(prereg, approval, s, a.arms)
            elif a.stage == "detect":
                if a.pooled:
                    detect_pooled(prereg, approval)
                else:
                    _, _, _, seeds = bca_context(prereg, approval)
                    for s in a.seeds or seeds:
                        detect_seed(prereg, approval, s)
            elif a.stage == "policy":
                stage_policy(prereg, approval)
            else:
                stage_verdict(prereg, approval)
    except (GateError, L.ApprovalError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
