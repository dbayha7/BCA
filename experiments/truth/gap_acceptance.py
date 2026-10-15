"""The action-gap protocol of the D4RL-RT track: pre-registration, host integrity, G1-pre to G4b, the verdict table.

The question: is a learned critic wrong mainly where the logged data are thin, and does caution placed there beat
even caution? The track removes the densest cluster of logged actions (about 15% of rows) from training and brings
it back at evaluation (gap.py defines and freezes it). Hosts trained on the gapped data (gap_train.py) are compared
with full-data hosts of the same seed. Rollout truth (labeller.py) measures the critic's true error. No BCA arm
and no BCA band fit exist here; gap_train refuses --method bca with the --allow-bca-band wording.

Order of work per task, in a world directory W (README: 'Action gap'):
  gap define / check / freeze           (gap.py; data only)       W/gap/gap.json, g0.json, gap.json.sha256
  preregister --gap-sha256              before any host run       W/gap_prereg.json (+ .sha256): plan, sizes,
                                                                  seeds, every pass mark, the gap sha256
  train 3 gapped (gap_train.py --gap-prereg-sha256; it refuses without the frozen gap pre-registration and writes
  its sha256 into resolved.json) and 3 full-data (train.py) hosts per task, seeds 202609171/172/173
  stage0 --gap-prereg-sha256 --host     no training               W/stage0/summary.json (option C's restore and
                                                                  one-step gate on the gapped copy, with its
                                                                  misaligned negative control), W/h0.json (H0:
                                                                  splits, schedule, events, each gapped run's
                                                                  gap pre-registration, and each seed pair's
                                                                  resolved row and training code equal except
                                                                  data; rerunnable after a failure)
  freeze                                refuses unless H0 passes  W/prereg.json (+ .sha256): hosts, checkpoint
                                        and the gate passes under sha256s, gate tolerances, the state source,
                                        the frozen tolerances     and the gap and gap-prereg sha256s
  pools, labels, score, screen, extract, evaluate, extraction-score, verdict, run   (all take --prereg-sha256)
Every stage after `freeze` refuses unless W/prereg.json hashes to the given sha256, its sidecar and embedded body
hash agree, W/gap/gap.json and W/gap_prereg.json still hash to the recorded values, and no analysis source file
changed (--allow-code-change marks every output). The stage functions re-check this themselves (verify_stage).

Hosts are keyed 'gap-<seed>' and 'full-<seed>'; a seed's two hosts are a pair (same init and evaluation bank; the
minibatches differ). Errors are in raw reward units: e = Q1 - Q^pi, e_A(s, a) = e(s, a) - e(s, pi_h(s)), each host
under its OWN policy pi_h.

Pools per host (stage pools):
  G1   the four frozen G0 pools (raw rows, the same for all six hosts): G (gap rows of native training), G_held (gap
       rows of native withheld), K_match, K_far. Slots (logged action, pi_h(s)). Rows at the velocity clip are
       dropped and counted when states are rebuilt from the observation.
  V    every visited state of 100 episodes of pi_h on option C's common reset bank (702609171 + i), with pi_h(s).
  V3   up to 1,500 in-gap and 1,500 out-of-gap visited states (seeded); slots (pi_h(s), a_out) with
       a_out = clip(c* + (pi - c*) 1.1 rho / ||pi - c*||, +-1) (a fixed axis when pi = c*).
  T, Pi (gapped hosts, G4 plans) option C's pool T (50,000 of the host's own kept training rows, slots (data, pi))
       and pool Pi (3,000 states, 3 + 2d slots; its central differences give the true policy gradient).
Labels (stage labels; resumable shards): G1 deterministic at both slots (all hosts); G1_rough (8 jitter rollouts at
1e-4 data SD; the data SD is over the NATIVE rows for all six hosts, so jitter never differs by arm) and G1_gap
(16 noisy-target rollouts) on 150 G rows x both slots (gapped hosts); V3 deterministic at pi (all hosts) and a_out
(gapped hosts, in-gap states); V3_rough (8 jitter rollouts on 150 in-gap states x both slots, gapped hosts) for
sigma_J; T and Pi (gapped hosts, G4). G1_smooth (8 jitter rollouts on every G1 point, all hosts) only when G1-pre
requires the smoothed truth.

Scores (pass marks from the pre-registration, never from code constants):
  G1-pre  the Stage 0 gate passes under the frozen tolerances (and its negative control fails); per gapped host the
          truth-noise share on G, mean jitter variance / Var(e) over G's (logged, pi) points. Above 0.05 on any
          gapped host, the jitter mean (G1_smooth) replaces the single rollouts in G1. The definition gap (noisy-target
          mean - Q^pi) is reported, never counted.
  G1      per pair p: R_X,p = median |e_A| on pool X under gap_S / the same under full_S, X in G, G_held, K_match,
          K_far; pooled = geometric mean over pairs; 95% intervals by an episode-block bootstrap (ORIGINAL D4RL
          episodes). Cross-section under gapped critics: median |e_A|(G) / median |e_A|(K_match). Seed-swap placebo:
          the G ratio between full critics of different seeds (and between gapped critics, reported). Pass: (a)
          pooled R_G >= 2.0, lower >= 1.5, every R_G,p >= 1.5; (b) pooled R_Kfar in [1/1.3, 1.3] with its interval
          in [1/1.5, 1.5]; (c) pooled cross-section >= 1.5; (d) pooled R_G_held >= 1.5, lower >= 1.2; (e) every
          full-full placebo ratio in [1/1.5, 1.5] and R_G >= 1.33 x the largest placebo max(r, 1/r). Reported: the raw
          |e| ratio, the share of e_A > 0 on G, core (d <= 0.5 rho) vs rim, ratios by tercile of the gapped host's
          steps to termination from (s, pi(s)).
  G2      S_gap = share of visited steps with ||pi(s) - c*|| <= rho (S_near at 1.25 rho), per host. Pass: pooled
          S_gap over gapped hosts (steps summed) >= 0.05 with 95% lower bound >= 0.02 (bootstrap over hosts, then
          episodes); at least host_count_min gapped hosts >= 0.03. Reported: gapped / full ratios, the behaviour's
          training mass, cross-policy rates (pi_gap on the full host's states and the reverse), the d / rho
          distribution, whether a host has < 300 in-gap states (G3 then pools hosts and says so).
  G3      (a) Delta J_p = mean over the common bank's episodes of J(gap_S) - J(full_S) (paired by reset seed);
          pooled = mean over pairs; t-interval over pairs at 97.5% (with one pair, the episode bootstrap); pass when
          pooled <= -3 and the upper bound < 0; the MDE and the train.py final-evaluation cross-check are reported,
          and a second look (pairs 174/175, the user's OK) is flagged when the point is <= -3 but the interval holds 0.
          (b) o_rel = (Q1 - Q^pi)(s, pi(s)) / the host's mean |Q^pi| over its labelled states: median in-gap > 0 and
          median in-gap - median out-of-gap >= 0.05 with 95% lower bound > 0 (blocks: host episodes). (c) at in-gap
          states A_pred = Q1(pi) - Q1(a_out), A_true = Q^pi(pi) - Q^pi(a_out); misled when A_pred > 0 and A_true <
          -3 sigma_J (sigma_J: the host's median single-rollout jitter SD on V3_rough); share >= 0.20. G3 passes on
          (a) and ((b) or (c)); its mechanism part is (b) or (c).
  G4a     per gapped critic, option C's Level-1 screen (extraction.Level1Screen) on pool T with the true gradient
          from pool Pi. Placements OR_abs (rank |e_A|) and OR_GAP (rank by ascending ||pi_host(s_i) - c*||, seeded
          ties), 16 SHUF, matched to rho* = 2 rho_host. A placement passes when, on every gapped critic, its raw
          first-order true gain is > 0, >= P0(2)'s + 0.25 |P0(2)'s|, and above the 95th percentile of the SHUF arms.
  G4b     option C's extraction on each frozen gapped critic (20k actor steps, fresh Adam, units = critics x seeds),
          arms NONE, P0 ladder, OR_abs, OR_signed, OR_GAP, 4 SHUF, ANTI_abs, ANTI_signed, ANTI_GAP matched to
          S(P0(2)), 100 common-reset episodes per policy. Per placement at 97.5% hierarchical intervals: J(P*) -
          J(P0(2)) >= 3 with lower > 0; best P (ladder and match) - best of NONE and the P0 ladder >= 2; J(P*) - mean
          J(SHUF*) >= 2. CNG = [share of sum (v_i - 1)+ on rows with ||pi_host(s_i) - c*|| <= 1.5 rho] / [those rows'
          share of T]. Verdict: 'room near the gap' if OR_GAP passes or OR_abs passes with CNG >= 1.5; 'room, not at
          the gap' if OR_abs passes with CNG < 1.5 and OR_GAP fails; 'no room' if both fail.
  R7      (a) J(OR_signed*) - J(ANTI_signed*) >= 3 with z >= 3; (b) at most 1 of 6 SHUF pairs with |z| > 2; (c) MDE
          <= 2 for both placements (else add extraction seeds up to 8; still failing: inconclusive); (d) every match
          within 2% and Level-1 cos(Delta g_P, Delta g_P0) < 0.98 for both placements on every critic; (e) blind.
Stop rule (pre-registered; protocol.stop_rule 'enforced'): G1, G2 and G3 need G1-pre (no G result before the
integrity checks pass; H0 is required by the freeze); G4a runs only if G1 passes all of
(a)-(e), G2 passes and G3's mechanism part passes; G4b only if G4a passes for at least one placement. The
synthetic_smoke profile sets 'exercise_all_stages' (refused on real data) so tests reach every stage; every
output then says so.

Synthetic data and random-init hosts: every output says SYNTHETIC and is a test fixture, NOT A FINDING.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.truth import acceptance as A  # noqa: E402
from experiments.truth import gap as GAP  # noqa: E402
from experiments.truth import labeller as LB  # noqa: E402
from experiments.truth import pools as PL  # noqa: E402
from experiments.truth import state_restore as SR  # noqa: E402
from experiments.truth.state_restore import require  # noqa: E402

GAP_PREREG_SCHEMA = "bca-truth-gap-prereg-v1"
PREREG_SCHEMA = "bca-truth-gap-final-v1"
STAGE_SCHEMA = "bca-truth-gap-stage-v1"
H0_SCHEMA = "bca-truth-gap-h0-v1"
WORLD = "D4RL-RT action gap"
SYNTHETIC_NOTE = A.SYNTHETIC_NOTE
ARMS = ("gap", "full")
CODE_FILES = tuple(A.CODE_FILES) + ("experiments/truth/gap.py", "experiments/truth/gap_acceptance.py",
                                    "experiments/truth/gap_train.py", "train.py", "runtime/config.py")
PLANS = {"full": ["G1_pre", "G1", "G2", "G3", "G4a", "G4b", "R7"], "no_g4": ["G1_pre", "G1", "G2", "G3"]}
PLACEMENTS = ("OR_abs", "OR_GAP")

PASS_MARKS = {  # the approved action-gap design
    "H0": dict(normalization_flag_sd=0.05),
    "G1_pre": dict(truth_noise_share_max=0.05),
    "G1": dict(ratio_min=2.0, ratio_lower95_min=1.5, pair_ratio_min=1.5, kfar_band=1.3, kfar_ci_band=1.5,
               cross_section_min=1.5, held_ratio_min=1.5, held_lower95_min=1.2, placebo_band=1.5,
               placebo_margin=1.33),
    "G2": dict(share_min=0.05, share_lower95_min=0.02, host_share_min=0.03, host_count_min=2,
               in_gap_states_min=300),
    "G3": dict(return_cost_min=3.0, ci_level=0.975, overestimate_median_min=0.0, overestimate_margin_min=0.05,
               misled_share_min=0.20, misled_jitter_sds=3.0),
    "G4a": dict(gain_margin_min=0.25, shuf_percentile=95.0),
    "G4b": dict(matched_margin_min=3.0, best_margin_min=2.0, shuf_margin_min=2.0, ci_level=0.975, cng_min=1.5,
                cng_radius=1.5),
    "R7": dict(A.PASS_MARKS["R7"]),
}

DESIGN = dict(
    host_step=1_000_000, gamma=0.99, horizon=1000, slots=256, shard_size=4096, seed=20261012,
    seeds=[202609171, 202609172, 202609173], stop_rule="enforced", bootstrap=dict(replicates=1000),
    g1_pre=dict(states=150, jitter_repeats=8, jitter_scale=1e-4, gap_repeats=16, noise_sd=0.2, noise_clip=0.5),
    g2=dict(episodes=100, reset_seed_base=702609171),
    g3=dict(in_gap=1500, out_gap=1500, rough_states=150, rough_repeats=8, out_scale=1.1, power=0.8),
    g4=dict(pool_T=dict(rows=50000), pool_Pi=dict(episodes=100, every=25, states=3000, reset_seed_base=702609171),
            actions=dict(delta=0.1, critic_step=0.1), thinness=dict(k=10), r4a=dict(level=2.0, shuf=16, s=0.83,
                                                                                     clip=2.5),
            extraction=json.loads(json.dumps(A.DESIGN["extraction"]))),
    r7=dict(power=0.8, alpha=0.05))
SMOKE = json.loads(json.dumps(DESIGN))  # tiny sizes for the synthetic end-to-end test only (refuses real data)
SMOKE.update(host_step=0, slots=32, shard_size=256, seeds=[202609171, 202609172], stop_rule="exercise_all_stages",
             bootstrap=dict(replicates=50),
             g1_pre=dict(states=4, jitter_repeats=3, jitter_scale=1e-4, gap_repeats=3, noise_sd=0.2, noise_clip=0.5),
             g2=dict(episodes=4, reset_seed_base=702609171),
             g3=dict(in_gap=12, out_gap=12, rough_states=4, rough_repeats=3, out_scale=1.1, power=0.8))
SMOKE["g4"].update(pool_T=dict(rows=96), pool_Pi=dict(episodes=6, every=5, states=24, reset_seed_base=702609171),
                   r4a=dict(level=2.0, shuf=6, s=0.83, clip=2.5),
                   extraction=json.loads(json.dumps(A.SMOKE["extraction"])))
PROFILES = {"design": DESIGN, "synthetic_smoke": SMOKE}
# Canonical sha256 of the approved pass marks and protocols (2026-10-12). The pre-registration compares against these
# literals, not against the live constants, so an edit of PASS_MARKS or DESIGN made before `preregister` shows up as
# pass_marks_equal_design / protocol_equal_design = False (POST HOC in the verdict) instead of passing silently.
PINNED_SHA256 = dict(pass_marks="6fcc7cbbc6f55b3f1d062071827b4816e586928f6233639bb381f32660850193",
                     protocol=dict(design="7b0007c4e4d7ac714daa07aa7236b7d324338d23677eb7c9fae8ac7650bb976c",
                                   synthetic_smoke="f76cc25626499f25fc034b0a69335e81018c54bafe5d9665d0237bc1ebc77d2d"))
# The training code a seed pair must share (source.json hashes; H0 gates on them). Other source differences are
# reported.
PAIR_CODE = ("train.py", "runtime/td3_bc.py", "runtime/common.py", "runtime/networks.py", "runtime/config.py",
             "runtime/validation.py", "runtime/environment.py", "runtime/rewards.py", "runtime/published.py",
             "algorithms/td3_bc.py", "algorithms/td3_bc_bca.py", "configs/td3_bc.yaml", "configs/experiment.yaml")


# ---------------------------------------------------------------- small utilities

def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


canonical_sha256 = A.canonical_sha256
read_json = A.read_json
jsonable = A.jsonable


def write_once(path, value):
    return A.write_once(path, value)


def code_hashes():
    return {name: SR.file_sha256(ROOT / name) for name in CODE_FILES}


def host_key(arm, seed):
    require(arm in ARMS, f"arm in {ARMS}")
    return f"{arm}-{int(seed)}"


def parse_key(key):
    arm, _, seed = str(key).partition("-")
    require(arm in ARMS and seed.isdigit(), f"host key is gap-<seed> or full-<seed>, got {key!r}")
    return arm, int(seed)


def host_dir(world, key):
    return Path(world) / f"host_{key}"


def planned_keys(protocol):
    return [host_key(arm, s) for s in protocol["seeds"] for arm in ARMS]


def finite_or_none(x):
    return A.finite_or_none(x)


def median_abs(x):
    x = np.abs(np.asarray(x, np.float64))
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def safe_ratio(a, b):
    if a is None or b is None or not b > 0 or not np.isfinite(a) or not np.isfinite(b):
        return None
    return float(a / b)


def geometric_mean(values):
    v = [x for x in values]
    if not v or any(x is None or not x > 0 for x in v):
        return None
    return float(np.exp(np.mean(np.log(v))))


def magnitude(r):
    return None if r is None or not r > 0 else max(r, 1.0 / r)


def interval(values, level):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], np.float64)
    if not len(v):
        return None, None
    tail = 100 * (1 - level) / 2
    return float(np.percentile(v, tail)), float(np.percentile(v, 100 - tail))


def hierarchical_interval(values_by_group, replicates, seed, level=0.95):
    """Percentile interval of the mean by resampling groups, then items within each resampled group."""
    groups = [np.asarray(v, np.float64) for v in values_by_group if len(v)]
    if not groups:
        return None, None
    g = np.random.default_rng(seed)
    out = np.empty(int(replicates))
    for b in range(len(out)):
        picks = g.integers(0, len(groups), len(groups))
        out[b] = np.mean(np.concatenate([groups[i][g.integers(0, len(groups[i]), len(groups[i]))] for i in picks]))
    return interval(out, level)


# ---------------------------------------------------------------- pre-registrations and the hash gate

class PreregError(ValueError):
    pass


def manifest(world, gap_sha256):
    return GAP.read_manifest(Path(world) / "gap" / "gap.json", gap_sha256, require_frozen=True)


def _write_document(world, name, body):
    document = dict(body=body, body_sha256=canonical_sha256(body))
    text = LB.strict_json(jsonable(document)).encode()
    world = Path(world)
    world.mkdir(parents=True, exist_ok=True)
    (world / name).write_bytes(text)
    digest = sha256_hex(text)
    (world / f"{name}.sha256").write_text(f"{digest}  {name}\n", encoding="utf8")
    return digest


def _read_document(path, expected_sha256):
    path = Path(path)
    if not path.is_file():
        raise PreregError(f"{path} is missing")
    data = path.read_bytes()
    digest = sha256_hex(data)
    if not expected_sha256 or digest != expected_sha256:
        raise PreregError(f"hash mismatch: {path} has sha256 {digest}, the stage was given {expected_sha256!r}; "
                          "refusing to run")
    side = path.with_name(path.name + ".sha256")
    if not side.is_file() or side.read_text(encoding="utf8").split()[0] != digest:
        raise PreregError(f"{side} is missing or does not match {path}; refusing to run")
    document = json.loads(data)
    if canonical_sha256(document["body"]) != document["body_sha256"]:
        raise PreregError(f"the body of {path} does not match its embedded hash; refusing to run")
    return document["body"], digest


def preregister(world, task, gap_sha256, *, profile="design", plan="full", pass_marks=None):
    """W/gap_prereg.json: plan, sizes, seeds and every pass mark, frozen BEFORE the first host run (train.py prints
    returns while it trains). Embeds the gap sha256; refuses unless the gap is frozen. The order is enforced where the
    hosts are made: gap_train refuses without this file's sha256 and writes it into each gapped run's resolved.json
    before training, and H0 checks it. Marks and protocol are compared with the pinned approved digests."""
    world = Path(world)
    task = SR.task_spec(task)
    require(profile in PROFILES and plan in PLANS, f"profile in {sorted(PROFILES)}, plan in {sorted(PLANS)}")
    m = manifest(world, gap_sha256)
    require(m.data["task"] == task.name, "the gap is for another task")
    require(m.data["defined"], "the gap manifest defines no gap")
    frozen = read_json(world / "gap" / "freeze.json")
    require(frozen["gap_sha256"] == m.sha256, "freeze.json belongs to another gap.json")
    synthetic = bool(m.data["synthetic"])
    require(profile != "synthetic_smoke" or synthetic, "the synthetic_smoke profile refuses real data")
    require(not frozen["synthetic_g0_override"] or synthetic, "a G0 override exists only for synthetic data")
    for name in ("gap_prereg.json", "prereg.json"):
        require(not (world / name).exists(), f"{world / name} exists; preregister once per world")
    require(not list(world.glob("host_*")), "per-host analysis outputs (W/host_*) exist; the pre-registration must "
            "precede them")
    protocol = json.loads(json.dumps(PROFILES[profile]))
    require(protocol["stop_rule"] == "enforced" or synthetic, "the stop rule is enforced on real data")
    marks = json.loads(json.dumps(PASS_MARKS if pass_marks is None else pass_marks))
    marks_pinned = canonical_sha256(marks) == PINNED_SHA256["pass_marks"]
    protocol_pinned = canonical_sha256(protocol) == PINNED_SHA256["protocol"][profile]
    body = dict(
        schema=GAP_PREREG_SCHEMA, world=WORLD, task=task.name, dataset=task.dataset, profile=profile,
        plan=PLANS[plan], plan_name=plan, synthetic=synthetic, gap_sha256=m.sha256,
        gap_definition_sha256=m.data["definition_sha256"], gap=dict(m.data["gap"]),
        gapped_file=dict(file=m.data["gapped"]["file"], sha256=m.data["gapped"]["sha256"]),
        original_file=dict(file=m.data["original"]["file"], sha256=m.data["original"]["sha256"]),
        g0=dict(sha256=SR.file_sha256(world / "gap" / "g0.json"), passed=frozen["g0_passed"],
                synthetic_override=frozen["synthetic_g0_override"]),
        hosts_planned=planned_keys(protocol), protocol=protocol, pass_marks=marks,
        pass_marks_equal_design=marks_pinned, protocol_equal_design=protocol_pinned,
        code_sha256=code_hashes(), frozen_utc=utc(),
        note=(SYNTHETIC_NOTE if synthetic else "Frozen before the first host run: every G mark, pool size and seed "
              "is pre-registered; anything else is POST HOC."))
    return _write_document(world, "gap_prereg.json", body)


def load_gap_prereg(world, gap_prereg_sha256, gap_sha256):
    body, digest = _read_document(Path(world) / "gap_prereg.json", gap_prereg_sha256)
    if body["gap_sha256"] != gap_sha256:
        raise PreregError("the gap pre-registration was frozen for another gap; refusing")
    manifest(world, gap_sha256)
    return body, digest


def load_prereg(world, expected_sha256, allow_code_change=False):
    """(body, sha256, code_changed) after the hash gate of the final pre-registration."""
    body, digest = _read_document(Path(world) / "prereg.json", expected_sha256)
    try:
        manifest(world, body["gap_sha256"])
        _read_document(Path(world) / "gap_prereg.json", body["gap_prereg_sha256"])
    except ValueError as error:
        raise PreregError(f"the gap or the gap pre-registration changed since the freeze: {error}") from error
    changed = sorted(k for k, v in code_hashes().items() if body["code_sha256"].get(k) != v)
    if changed and not allow_code_change:
        raise PreregError(f"analysis source changed since the freeze: {changed}; refusing to run "
                          "(--allow-code-change runs anyway and marks every output)")
    return body, digest, bool(changed)


def verify_stage(world, body, prereg_sha, code_changed=False):
    frozen, _, changed = load_prereg(world, prereg_sha, allow_code_change=True)
    if canonical_sha256(body) != canonical_sha256(frozen):
        raise PreregError("the stage was given a body that differs from the frozen pre-registration; refusing to run")
    if changed and not code_changed:
        raise PreregError("analysis source changed since the freeze and the caller did not acknowledge it "
                          "(--allow-code-change); refusing to run")
    return True


def stage_header(body, prereg_sha, stage, key=None, code_changed=False):
    head = dict(schema=STAGE_SCHEMA, world=WORLD, stage=stage, task=body["task"], dataset=body["dataset"], host=key,
                profile=body["profile"], plan=body["plan"], prereg_sha256=prereg_sha, gap_sha256=body["gap_sha256"],
                synthetic=bool(body["synthetic"]), code_changed_since_prereg=bool(code_changed),
                stop_rule=body["protocol"]["stop_rule"], utc=utc())
    head["note"] = SYNTHETIC_NOTE if body["synthetic"] else "pre-registered stage of the action-gap protocol"
    if body["protocol"]["stop_rule"] != "enforced":
        head["note"] += " The stop rule is NOT applied (synthetic_smoke exercises every stage)."
    return head


# ---------------------------------------------------------------- hosts

def _data_dir_for(m, arm, data_dir=None):
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    return Path(m.data["gapped" if arm == "gap" else "original"]["path"]).parent


def restore(world, m, key, run_dir, *, data_dir=None, step=None, expected_sha256=None):
    """pools.restore_host for a planned host key, with the gap checks of its arm."""
    arm, seed = parse_key(key)
    host = PL.restore_host(run_dir, data_dir=_data_dir_for(m, arm, data_dir), step=step,
                           expected_sha256=expected_sha256,
                           gap_manifest=Path(world) / "gap" / "gap.json" if arm == "gap" else None)
    require(host.seed == seed, f"{run_dir} is seed {host.seed}, not {key}")
    require(host.task.name == m.data["task"], f"{run_dir} is another task")
    if arm == "gap":
        require(host.identity["action_gap"] is not None and host.identity["action_gap"]["manifest_sha256"] == m.sha256,
                f"{run_dir} is not a gapped host of this gap")
        require(host.identity["data"]["sha256"] == m.data["gapped"]["sha256"], "the host's data is not the gapped copy")
    else:
        require(host.identity["action_gap"] is None, f"{run_dir} is a gapped run, not a full-data control")
        require(host.identity["data"]["sha256"] == m.data["original"]["sha256"], "the host's data is not the original")
    return host


def host_for(world, body, key, data_dir=None, host_dirs=None):
    record = body["hosts"][key]
    m = manifest(world, body["gap_sha256"])
    run_dir = (host_dirs or {}).get(key, record["run_dir"])
    host = restore(world, m, key, run_dir, data_dir=data_dir, step=record["step"],
                   expected_sha256=record["checkpoint_sha256"])
    pr, args = body["protocol"], host.args
    require(float(args.discount) == float(pr["gamma"]),
            f"host discount {args.discount} differs from the pre-registered label gamma {pr['gamma']}")
    require((float(args.policy_noise), float(args.noise_clip)) == (float(pr["g1_pre"]["noise_sd"]),
                                                                   float(pr["g1_pre"]["noise_clip"])),
            "host target-smoothing noise differs from the pre-registered noisy-target mode")
    return host, m


def _events(run_dir):
    import gzip

    path = Path(run_dir) / "events.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pair_row_signature(row):
    """A resolved row without what may differ between the two hosts of a seed pair: the run id, the data pin and the
    gap record (and a synthetic fixture's note). Everything else (hyperparameters, seed, evaluation banks, schedule,
    reservation) must be equal."""
    import copy

    row = copy.deepcopy(row)
    for k in ("run_id", "cache", "action_gap", "synthetic_note"):
        row.pop(k, None)
    row.get("protocol", {}).pop("run_id", None)
    return row


def _differing(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _differing(a.get(k), b.get(k), f"{prefix}{k}.") if k in a and k in b else [f"{prefix}{k}"]
        return out
    return [] if a == b else [prefix.rstrip(".")]


def pair_checks(world, seed, run_dirs, gp_frozen_utc):
    """H0's seed-pair identity: the two runs of a seed share the resolved row except data and run id (gated), the
    training code PAIR_CODE in source.json (gated, real runs only) and report the JAX platform and other source
    differences (flagged)."""
    rows = {arm: json.loads((Path(run_dirs[arm]) / "resolved.json").read_text(encoding="utf8")) for arm in ARMS}
    sig = {arm: pair_row_signature(rows[arm]) for arm in ARMS}
    same = canonical_sha256(jsonable(sig["gap"])) == canonical_sha256(jsonable(sig["full"]))
    out = dict(rows_identical_except_data=same, row_differences=_differing(sig["gap"], sig["full"]))
    synthetic = any((Path(run_dirs[arm]) / "SYNTHETIC_HOST.txt").is_file() for arm in ARMS)
    sources = {arm: Path(run_dirs[arm]) / "source.json" for arm in ARMS}
    if synthetic:
        out["code_identical"] = "skipped: SYNTHETIC random-init host"
    elif not all(p.is_file() for p in sources.values()):
        out["code_identical"] = False
        out["code_error"] = "source.json is missing (train.py writes it)"
    else:
        src = {arm: read_json(p) for arm, p in sources.items()}
        code = [k for k in PAIR_CODE if src["gap"].get(k) != src["full"].get(k)]
        other = sorted(k for k in set(src["gap"]) | set(src["full"])
                       if k not in PAIR_CODE and k != "runtime" and src["gap"].get(k) != src["full"].get(k))
        env = {arm: (src[arm].get("runtime") or {}).get("execution_environment", {}) for arm in ARMS}
        platforms = {arm: env[arm].get("JAX_PLATFORMS") for arm in ARMS}
        out.update(code_identical=not code, code_differences=code, other_source_differences=other,
                   jax_platforms=platforms, device_flag=platforms["gap"] != platforms["full"])
    # Informational: when the full-data control's resolved.json was written relative to the gap pre-registration
    # (file time; the gapped run's order is enforced through its recorded gap pre-registration sha256).
    mtime = datetime.fromtimestamp((Path(run_dirs["full"]) / "resolved.json").stat().st_mtime, timezone.utc)
    out["full_resolved_written_utc"] = mtime.isoformat()
    out["full_written_before_gap_prereg"] = bool(mtime < datetime.fromisoformat(gp_frozen_utc))
    gates = [out["rows_identical_except_data"]] + ([out["code_identical"]] if isinstance(out["code_identical"], bool)
                                                   else [])
    out["passed"] = bool(all(gates))
    return out


def host_integrity(world, gp, m, hosts, *, gp_sha=None, data_dir=None):
    """H0: every host restores exactly, trained the intended rows with the repo's code, on schedule; every gapped
    host was trained under this gap pre-registration; the two hosts of each seed differ only in their data."""
    from runtime.config import typed
    from runtime.validation import verify_events
    from experiments.wbcp import freeze_scores as FS

    pr = gp["protocol"]
    native = m.native_split_rows()
    gap_rows = m.array("gap_rows")
    expect = {"full": native, "gap": {k: v[~np.isin(v, gap_rows)] for k, v in native.items()}}
    records, norms = {}, {}
    for key in planned_keys(pr):
        arm, seed = parse_key(key)
        run_dir = Path(hosts[key]).expanduser().resolve()
        checks, details = {}, {}
        try:
            step, ckpt, recorded = FS.locate_checkpoint(run_dir, pr["host_step"])
            host = restore(world, m, key, run_dir, data_dir=data_dir, step=step)
            checks["restore_and_standardized_obs_gate"] = True
        except Exception as error:  # recorded, never raised: H0 lists every failure
            records[key] = dict(run_dir=str(run_dir), passed=False,
                                checks=dict(restore_and_standardized_obs_gate=False),
                                error=f"{type(error).__name__}: {error}")
            continue
        synthetic_host = bool(host.identity["synthetic_host"])
        digest = host.identity["checkpoint"]["sha256"]
        checks["checkpoint_sha256_equals_result_json"] = recorded is not None and recorded == digest
        result = read_json(run_dir / "result.json")
        row = json.loads((run_dir / "resolved.json").read_text(encoding="utf8"))
        if synthetic_host:
            require(gp["synthetic"], "a synthetic host in a real world")
            checks["updates_and_checkpoint_schedule"] = "skipped: SYNTHETIC random-init host"
            checks["verify_events"] = "skipped: SYNTHETIC random-init host"
        else:
            expected = row["expected_counts"]
            checks["updates_and_checkpoint_schedule"] = bool(
                result.get("completed") is True and result.get("steps_completed") == pr["host_step"]
                and expected["host_updates"] == pr["host_step"]
                and [c["step"] for c in result.get("checkpoints", [])] == expected["checkpoint_steps"])
            try:
                events = _events(run_dir)
                ok = SR.file_sha256(run_dir / "events.jsonl.gz") == result.get("events_sha256")
                _, _, _, protocol = typed(PL.adapt_archived_row(row)[0])
                verify_events(protocol, events, result["evaluations"], host="td3")
                checks["verify_events"] = bool(ok)
            except Exception as error:
                checks["verify_events"] = False
                details["verify_events_error"] = f"{type(error).__name__}: {error}"
        split = {name: host.rows[ids] for name, ids in (("training", host.training_ids),
                                                        ("withheld", host.withheld_ids), ("bank", host.bank_ids))}
        for name in ("training", "withheld", "bank"):
            checks[f"{name}_rows_equal_native{'_minus_gap' if arm == 'gap' else ''}"] = bool(
                np.array_equal(split[name], expect[arm][name]))
        leaks = GAP.leak_counts(m, split)
        details["gap_rows_inside"] = leaks
        # The reservation record the run itself trained with (its preparation.json), not the restore's rebuild.
        own = read_json(run_dir / "preparation.json")["metadata"]["reservation"]
        rebuilt = canonical_sha256(PL._json_ready(host.prepared.metadata["reservation"]))
        checks["reservation_record_equals_rebuild"] = canonical_sha256(PL._json_ready(own)) == rebuilt
        if arm == "gap":
            checks["no_gap_row_in_training_withheld_bank"] = all(v == 0 for v in leaks.values())
            checks["projected_reservation_of_this_gap"] = own.get("gap_manifest_sha256") == m.sha256
            recorded = (row.get("action_gap") or {}).get("gap_prereg_sha256")
            details["gap_prereg_sha256_recorded"] = recorded
            checks["trained_under_this_gap_prereg"] = gp_sha is not None and recorded == gp_sha
        else:
            checks["reservation_record_equals_manifest"] = (canonical_sha256(PL._json_ready(own))
                                                            == m.data["native"]["record_sha256"])
        norms[key] = (host.obs_mean.astype(np.float64), host.obs_std.astype(np.float64))
        bools = [v for v in checks.values() if isinstance(v, bool)]
        records[key] = dict(run_dir=str(run_dir), checkpoint=dict(path=str(ckpt), step=step, sha256=digest),
                            identity=host.identity, checks=checks, details=details, synthetic_host=synthetic_host,
                            passed=bool(all(bools)))
    normalization, pairs = {}, {}
    for seed in pr["seeds"]:
        g, f = host_key("gap", seed), host_key("full", seed)
        if g in norms and f in norms:
            (mg, sg), (mf, sf) = norms[g], norms[f]
            shift = float(np.max(np.abs(mg - mf) / sf))
            scale = float(np.max(np.abs(np.log(sg / sf))))
            normalization[str(seed)] = dict(max_mean_shift_in_sd=shift, max_abs_log_sd_ratio=scale,
                                            flagged=bool(max(shift, scale) > gp["pass_marks"]["H0"]
                                                         ["normalization_flag_sd"]))
        try:
            pairs[str(seed)] = pair_checks(world, seed, {"gap": Path(hosts[g]).expanduser().resolve(),
                                                         "full": Path(hosts[f]).expanduser().resolve()},
                                           gp["frozen_utc"])
        except Exception as error:  # recorded, never raised
            pairs[str(seed)] = dict(passed=False, error=f"{type(error).__name__}: {error}")
    passed = (all(r["passed"] for r in records.values()) and len(records) == len(planned_keys(pr))
              and len(pairs) == len(pr["seeds"]) and all(p["passed"] for p in pairs.values()))
    return dict(schema=H0_SCHEMA, world=WORLD, task=gp["task"], gap_sha256=gp["gap_sha256"], synthetic=gp["synthetic"],
                hosts=records, pairs=pairs, normalization=normalization, passed=bool(passed), utc=utc(),
                note=SYNTHETIC_NOTE if gp["synthetic"] else "H0 host integrity (no outcome is read)",
                rule="every check exact; each seed pair's resolved rows equal except data and run id, and (real runs) "
                     "the training code PAIR_CODE equal; normalization differences, a JAX platform difference and a "
                     "full control written before the gap pre-registration are reported and flagged, never a stop")


def stage0(world, gap_sha256, gap_prereg_sha256, hosts, *, data_dir=None, rows=1000, seed=0):
    """Option C's Stage 0 restore and one-step gate on the gapped copy (W/stage0) and H0 on the planned hosts
    (W/h0.json). Needs the finished host runs; reads no outcome. After a failed H0 (a code or data fault: stop, fix,
    re-verify) a rerun keeps the Stage 0 gate of the same gapped file, moves the failed h0.json to
    h0.failed-<n>.json (kept for the record) and checks again. A passed h0.json is never replaced."""
    world = Path(world)
    gp, gp_sha = load_gap_prereg(world, gap_prereg_sha256, gap_sha256)
    m = manifest(world, gap_sha256)
    require(sorted(hosts) == sorted(gp["hosts_planned"]), f"--host must name exactly {gp['hosts_planned']}")
    require(not (world / "prereg.json").exists(), "the final pre-registration exists; H0 is not rerun after it")
    gapped = m.gapped_path(data_dir)
    summary_path = world / "stage0" / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        require(summary["file"]["sha256"] == m.data["gapped"]["sha256"], "W/stage0 holds a gate of another file")
    else:
        summary = A.stage0(world, gp["task"], file=gapped, rows=rows, seed=seed)
    h0 = host_integrity(world, gp, m, hosts, gp_sha=gp_sha, data_dir=data_dir)
    h0["gap_prereg_sha256"] = gp_sha
    path = world / "h0.json"
    if path.exists():
        require(not read_json(path)["passed"], f"{path} passed already; H0 is written once")
        n = len(list(world.glob("h0.failed-*.json")))
        path.rename(world / f"h0.failed-{n}.json")
    write_once(path, h0)
    return summary, h0


def freeze(world, *, gap_sha256, gap_prereg_sha256, hosts, tolerances=None, state_source="auto"):
    """W/prereg.json: the gap pre-registration plus the hosts, the gate tolerances and the state source. Refuses
    unless H0 passed, the Stage 0 pilot gate passes and its negative control fails under the tolerances."""
    from experiments.wbcp import freeze_scores as FS

    world = Path(world)
    gp, gp_sha = load_gap_prereg(world, gap_prereg_sha256, gap_sha256)
    m = manifest(world, gap_sha256)
    require(not (world / "prereg.json").exists(), f"{world / 'prereg.json'} exists; freeze once per world")
    require(not list(world.glob("host_*")) and not (world / "g4").exists(),
            "labelled or trained outputs exist; the freeze must precede them")
    summary_path = world / "stage0" / "summary.json"
    require(summary_path.is_file(), "run stage0 first")
    summary = read_json(summary_path)
    require(summary["dataset"] == gp["dataset"] and summary["file"]["sha256"] == m.data["gapped"]["sha256"],
            "Stage 0 ran on another file")
    require(state_source in ("auto", "rebuilt"), "state_source is 'auto' or 'rebuilt'")
    tol = dict(SR.DEFAULT_TOLERANCES, **(tolerances or {}))
    pilot = summary["gate_rebuilt"] if state_source == "rebuilt" and summary.get("gate_rebuilt") else summary["gate"]
    require(A.gate_passes(pilot, tol), f"the Stage 0 pilot gate ({pilot['source']} states) does not pass under {tol}")
    require(not A.gate_passes(summary["gate_negative"], tol),
            f"the Stage 0 negative control passes under {tol}; the tolerances are too loose")
    h0_path = world / "h0.json"
    require(h0_path.is_file(), "run stage0 first (H0)")
    h0 = read_json(h0_path)
    require(h0["gap_prereg_sha256"] == gp_sha, "H0 ran under another gap pre-registration")
    require(h0["passed"], "H0 failed: a code or data fault; stop, fix and re-verify (no G result is computed)")
    require(sorted(hosts) == sorted(gp["hosts_planned"]), f"--host must name exactly {gp['hosts_planned']}")
    records = {}
    for key, run_dir in sorted(hosts.items()):
        run_dir = Path(run_dir).expanduser().resolve()
        step, ckpt, recorded = FS.locate_checkpoint(run_dir, gp["protocol"]["host_step"])
        digest = SR.file_sha256(ckpt)
        require(digest == h0["hosts"][key]["checkpoint"]["sha256"], f"{ckpt} differs from the one H0 verified")
        arm, seed = parse_key(key)
        records[key] = dict(arm=arm, seed=seed, run_dir=str(run_dir), checkpoint=ckpt.name, step=step,
                            checkpoint_sha256=digest)
    body = dict(
        schema=PREREG_SCHEMA, world=WORLD, task=gp["task"], dataset=gp["dataset"], profile=gp["profile"],
        plan=gp["plan"], plan_name=gp["plan_name"], synthetic=gp["synthetic"], gap_sha256=gap_sha256,
        gap_prereg_sha256=gp_sha, gap=gp["gap"], hosts=records, state_source=state_source,
        stage0=dict(summary_sha256=SR.file_sha256(summary_path), gate_pilot=dict(
            source=pilot["source"], restore_miss=pilot["restore_miss"], next_obs_miss=pilot["next_obs_miss"],
            reward_miss=pilot["reward_miss"], terminal_agreement=pilot["terminal_agreement"],
            rows_checked=pilot["rows_checked"]), negative_control_failed_under_frozen_tolerances=True),
        h0_sha256=SR.file_sha256(h0_path), gate_tolerances=tol, protocol=gp["protocol"], pass_marks=gp["pass_marks"],
        pass_marks_equal_design=gp["pass_marks_equal_design"],
        protocol_equal_design=gp.get("protocol_equal_design", False), g0=gp["g0"], code_sha256=code_hashes(),
        frozen_utc=utc(),
        note=(SYNTHETIC_NOTE if gp["synthetic"] else "Frozen after H0 and before any truth label; the marks are the "
              "gap pre-registration's."))
    return _write_document(world, "prereg.json", body)


# ---------------------------------------------------------------- pools

def build_g1_pool(host, policy, probe, m, state_source):
    task = host.task
    raws, codes = [], []
    for code, name in enumerate(("G", "G_held", "K_match", "K_far")):
        r = m.array(f"pool_{name}").astype(np.int64)
        raws.append(r)
        codes.append(np.full(len(r), code, np.int64))
    raw_rows, code = np.concatenate(raws), np.concatenate(codes)
    states = SR.logged_states(task, host.raw, raw_rows, state_source)
    dropped = {name: int((~states.kept[code == c]).sum()) for c, name in enumerate(("G", "G_held", "K_match", "K_far"))}
    raw_rows, code = raw_rows[states.kept], code[states.kept]
    states = SR.logged_states(task, host.raw, raw_rows, state_source)
    obs = SR.observation(task, states.qpos, states.qvel)
    logged = np.asarray(host.raw["actions"])[raw_rows].astype(np.float32)
    mismatched = int(np.any(obs.astype(np.float32) != np.asarray(host.raw["observations"])[raw_rows], axis=1).sum())
    pi = policy.actions(obs)
    acts = np.stack([logged, pi], axis=1)
    n = len(raw_rows)
    q = probe.q(np.repeat(obs, 2, axis=0), acts.reshape(2 * n, -1)).reshape(n, 2, 2)
    arrays = dict(qpos=states.qpos, qvel=states.qvel, obs=obs, raw_row=raw_rows, code=code,
                  episode=m.original_episodes(raw_rows), actions=acts, q=q, d_logged=m.d_over_rho(logged),
                  d_pi=m.d_over_rho(pi), in_gap_logged=m.in_gap(logged), in_gap_pi=m.in_gap(pi),
                  in_host_training=np.isin(raw_rows, host.rows[host.training_ids]))
    meta = dict(kind="G1", sub_pools=["G", "G_held", "K_match", "K_far"], slots=["data", "pi"],
                counts={name: int((code == c).sum()) for c, name in enumerate(("G", "G_held", "K_match", "K_far"))},
                dropped_at_velocity_clip=dropped, state_source=states.source, obs_mismatches_vs_log=mismatched,
                episodes="original D4RL episode ids")
    return arrays, meta


def build_visit_pool(host, physics, policy, m, pr):
    g2 = pr["g2"]
    seeds = [g2["reset_seed_base"] + i for i in range(g2["episodes"])]
    q0, v0 = PL.reset_bank(physics, seeds)
    ep, rec = PL.policy_episodes(physics, policy, q0, v0, horizon=pr["horizon"], every=1)
    obs = SR.observation(host.task, rec["qpos"], rec["qvel"])
    pi = policy.actions(obs)
    d = m.d_over_rho(pi)
    scores = PL.normalized_scores(host.task.dataset, ep["returns"])
    arrays = dict(qpos=rec["qpos"], qvel=rec["qvel"], obs=obs, episode=rec["episode"], t=rec["t"], pi=pi, d_pi=d,
                  in_gap=m.in_gap(pi), near=d <= m.data["gap"]["near"], episode_returns=ep["returns"],
                  episode_scores=scores, episode_lengths=ep["lengths"], episode_terminated=ep["terminated"],
                  episode_failed=ep["failed"], reset_seeds=np.asarray(seeds, np.int64))
    meta = dict(kind="V", states=int(len(obs)), episodes=int(len(seeds)), reset_seed_base=int(g2["reset_seed_base"]),
                horizon=int(pr["horizon"]), host_score_mean=float(scores.mean()),
                simulator_errors=int(ep["failed"].sum()), in_gap_states=int(arrays["in_gap"].sum()))
    return arrays, meta


def out_of_gap_actions(pi, centre, rho, scale):
    """a_out = clip(c* + (pi - c*) scale rho / ||pi - c*||, +-1); pi = c* uses the first axis."""
    pi = np.asarray(pi, np.float64)
    v = pi - centre[None]
    norm = np.linalg.norm(v, axis=1)
    unit = np.zeros_like(v)
    unit[:, 0] = 1.0
    unit = np.where(norm[:, None] > 0, v / np.where(norm > 0, norm, 1.0)[:, None], unit)
    return np.clip(centre[None] + scale * rho * unit, -1.0, 1.0).astype(np.float32)


def build_g3_pool(host, probe, m, V, pr):
    g3, seed = pr["g3"], pr["seed"]
    inside, outside = np.flatnonzero(V["in_gap"]), np.flatnonzero(~V["in_gap"])
    pick_in = inside[PL._seeded_choice(len(inside), g3["in_gap"], seed, 31)] if len(inside) else inside
    pick_out = outside[PL._seeded_choice(len(outside), g3["out_gap"], seed, 32)] if len(outside) else outside
    idx = np.concatenate([pick_in, pick_out]).astype(np.int64)
    in_gap = np.r_[np.ones(len(pick_in), bool), np.zeros(len(pick_out), bool)]
    pi = V["pi"][idx].astype(np.float32)
    a_out = out_of_gap_actions(pi, m.centre, m.rho, g3["out_scale"])
    acts = np.stack([pi, np.where(in_gap[:, None], a_out, pi)], axis=1).astype(np.float32)
    n = len(idx)
    obs = V["obs"][idx]
    q = (probe.q(np.repeat(obs, 2, axis=0), acts.reshape(2 * n, -1)).reshape(n, 2, 2) if n
         else np.zeros((0, 2, 2), np.float32))
    arrays = dict(qpos=V["qpos"][idx], qvel=V["qvel"][idx], obs=obs, visit_index=idx, episode=V["episode"][idx],
                  t=V["t"][idx], in_gap=in_gap, actions=acts, q=q, d_pi=V["d_pi"][idx],
                  d_out=m.d_over_rho(acts[:, 1]) if n else np.zeros(0))
    meta = dict(kind="V3", slots=["pi", "a_out"], states=int(n), in_gap=int(len(pick_in)), out_gap=int(len(pick_out)),
                available_in_gap=int(len(inside)), available_out_gap=int(len(outside)),
                a_out_rule=f"clip(c* + (pi - c*) {g3['out_scale']} rho / ||pi - c*||, +-1)",
                a_out_inside_gap=int(np.sum(m.in_gap(acts[in_gap, 1]))) if in_gap.any() else 0)
    return arrays, meta


def stage_pools(world, body, prereg_sha, key, *, data_dir=None, host_dirs=None, code_changed=False):
    verify_stage(world, body, prereg_sha, code_changed)
    out = host_dir(world, key) / "pools"
    require(not (out / "summary.json").exists(), f"{out} is complete")
    pr = body["protocol"]
    arm, _ = parse_key(key)
    host, m = host_for(world, body, key, data_dir, host_dirs)
    policy = host.policy(pr["slots"])
    invariance = A.require_invariant(policy, host)
    probe = PL.CriticProbe(host)
    head = stage_header(body, prereg_sha, "pools", key, code_changed)
    common = dict(head, host=host.identity, position_invariance=invariance, gap=body["gap"])
    source = body["state_source"]
    arrays, meta = build_g1_pool(host, policy, probe, m, source)
    PL.save_pool(out, "G1", arrays, dict(common, **meta))
    physics = SR.Physics(host.task.dataset)
    try:
        V, meta = build_visit_pool(host, physics, policy, m, pr)
        PL.save_pool(out, "V", V, dict(common, **meta))
        arrays, meta = build_g3_pool(host, probe, m, V, pr)
        PL.save_pool(out, "V3", arrays, dict(common, **meta))
        if arm == "gap" and any(r in body["plan"] for r in ("G4a", "G4b")):
            g4 = pr["g4"]
            arrays, meta = PL.build_pool_T(host, policy, probe, rows=g4["pool_T"]["rows"], seed=pr["seed"],
                                           state_source=source)
            PL.save_pool(out, "T", arrays, dict(common, **meta))
            thin = PL.ThinnessIndex(host, k=g4["thinness"]["k"])
            w = host.withheld_ids
            radius, _ = thin.fit_radius(PL.features(host, host.converted["observations"][w],
                                                    host.converted["actions"][w]), host.episodes[w])
            p = g4["pool_Pi"]
            arrays, meta = PL.build_pool_Pi(host, physics, policy, probe, thin, PL.StateIndex(host),
                                            episodes=p["episodes"], every=p["every"], states=p["states"],
                                            reset_seed_base=p["reset_seed_base"], seed=pr["seed"],
                                            horizon=pr["horizon"], delta=g4["actions"]["delta"],
                                            critic_step=g4["actions"]["critic_step"])
            PL.save_pool(out, "Pi", arrays, dict(common, radius=radius, **meta))
    finally:
        physics.close()
    # One jitter scale for all six hosts: the data SD over the NATIVE converted rows (the raw state arrays are the
    # same in the original and the gapped copy), so jitter truth never differs between the arms of a pair.
    native_rows = m.array("native_raw_current").astype(np.int64)
    sd_q, sd_v = SR.state_sd(SR.logged_states(host.task, host.raw, native_rows, source))
    np.savez(out / "jitter_sd.npz", qpos=sd_q, qvel=sd_v)
    return write_once(out / "summary.json", dict(common, pools=sorted(p.stem for p in out.glob("*.json")
                                                                      if p.stem != "summary")))


# ---------------------------------------------------------------- labels

def label_jobs(body, key, smooth=False):
    """{job: (pool, mode, repeats, selection)} for one host. selection is a tuple read by job_pairs."""
    pr = body["protocol"]
    arm, _ = parse_key(key)
    p1, p3 = pr["g1_pre"], pr["g3"]
    jobs = {"G1": ("G1", "deterministic", 1, ("all", (0, 1)))}
    if arm == "gap":
        jobs["G1_rough"] = ("G1", "jitter", p1["jitter_repeats"], ("code0", p1["states"], 1, (0, 1)))
        jobs["G1_gap"] = ("G1", "noisy_target", p1["gap_repeats"], ("code0", p1["states"], 1, (0, 1)))
    jobs["V3"] = ("V3", "deterministic", 1, ("g3", arm))
    if arm == "gap":
        jobs["V3_rough"] = ("V3", "jitter", p3["rough_repeats"], ("in_gap", p3["rough_states"], 2, (0, 1)))
        if any(r in body["plan"] for r in ("G4a", "G4b")):
            jobs["T"] = ("T", "deterministic", 1, ("all", (0, 1)))
            jobs["Pi"] = ("Pi", "deterministic", 1, ("all", None))
    if smooth:
        jobs["G1_smooth"] = ("G1", "jitter", p1["jitter_repeats"], ("all", (0, 1)))
    return jobs


def job_pairs(arrays, selection, seed):
    """(state, slot) arrays of a job, sorted by item id."""
    n, S = arrays["actions"].shape[:2]
    kind = selection[0]
    if kind == "all":
        slots = np.arange(S) if selection[1] is None else np.asarray(selection[1], np.int64)
        st, sl = np.repeat(np.arange(n), len(slots)), np.tile(slots, n)
    elif kind == "code0":
        cand = np.flatnonzero(arrays["code"] == 0)
        pick = cand[PL._seeded_choice(len(cand), selection[1], seed, 100 + selection[2])] if len(cand) else cand
        slots = np.asarray(selection[3], np.int64)
        st, sl = np.repeat(pick, len(slots)), np.tile(slots, len(pick))
    elif kind == "in_gap":
        cand = np.flatnonzero(arrays["in_gap"])
        pick = cand[PL._seeded_choice(len(cand), selection[1], seed, 100 + selection[2])] if len(cand) else cand
        slots = np.asarray(selection[3], np.int64)
        st, sl = np.repeat(pick, len(slots)), np.tile(slots, len(pick))
    else:
        require(kind == "g3", f"unknown selection {kind}")
        st = np.arange(n)
        sl = np.zeros(n, np.int64)
        if selection[1] == "gap":
            inside = np.flatnonzero(arrays["in_gap"])
            st, sl = np.r_[st, inside], np.r_[sl, np.ones(len(inside), np.int64)]
    st, sl = np.asarray(st, np.int64), np.asarray(sl, np.int64)
    order = np.argsort(st * S + sl, kind="stable")
    return st[order], sl[order]


def job_items(arrays, st, sl):
    S = arrays["actions"].shape[1]
    return dict(qpos=arrays["qpos"][st], qvel=arrays["qvel"][st], action=arrays["actions"][st, sl].astype(np.float32),
                item_id=(st * S + sl).astype(np.int64))


def smoothing_required(world):
    path = Path(world) / "scores" / "G1_pre.json"
    return path.is_file() and bool(read_json(path)["values"]["smoothed_truth_required"])


def _pool(world, key, name):
    return PL.load_pool(host_dir(world, key) / "pools", name)


def stage_labels(world, body, prereg_sha, key, *, jobs=None, worker=0, workers=1, data_dir=None, host_dirs=None,
                 log=print, code_changed=False):
    verify_stage(world, body, prereg_sha, code_changed)
    pr = body["protocol"]
    plan_jobs = label_jobs(body, key, smooth=smoothing_required(world))
    jobs = list(plan_jobs) if jobs is None else list(jobs)
    require(all(j in plan_jobs for j in jobs), f"jobs for {key}: {sorted(plan_jobs)}")
    host, _ = host_for(world, body, key, data_dir, host_dirs)
    policy = host.policy(pr["slots"])
    sd = np.load(host_dir(world, key) / "pools" / "jitter_sd.npz")
    physics = SR.Physics(host.task.dataset)
    records = {}
    try:
        for job in jobs:
            pool, mode, repeats, selection = plan_jobs[job]
            arrays, _ = _pool(world, key, pool)
            st, sl = job_pairs(arrays, selection, pr["seed"])
            if not len(st):
                records[job] = dict(shards=0, env_steps=0, empty=True)
                continue
            p1 = pr["g1_pre"]
            spec = LB.LabelSpec(host.task.dataset, mode=mode, repeats=int(repeats), gamma=float(pr["gamma"]),
                                horizon=int(pr["horizon"]), seed=int(pr["seed"]),
                                jitter_scale=float(p1["jitter_scale"]), noise_sd=float(p1["noise_sd"]),
                                noise_clip=float(p1["noise_clip"]), shard_size=int(pr["shard_size"]))
            turn = (worker + list(plan_jobs).index(job)) % workers
            recs = LB.run_labels(host_dir(world, key) / "labels" / job, physics, policy, spec,
                                 job_items(arrays, st, sl),
                                 jitter_sd=(sd["qpos"], sd["qvel"]) if mode == "jitter" else None, worker=turn,
                                 workers=workers, log=log)
            records[job] = dict(shards=len(recs), env_steps=int(sum(r["env_steps"] for r in recs)))
    finally:
        physics.close()
    marker = dict(stage_header(body, prereg_sha, "labels", key, code_changed), jobs=records, worker=worker,
                  workers=workers)
    LB._write_atomic(host_dir(world, key) / "labels" / f"progress_worker{worker}.json",
                     LB.strict_json(jsonable(marker)).encode())
    return records


def load_job(world, body, key, job, smooth=None):
    """(q, length, status) of shape (n_states, n_slots, K), NaN where not labelled, or None for an empty job.
    The items are rebuilt from the pool and must hash to the label manifest's inputs (labels made for another pool
    are refused)."""
    smooth = smoothing_required(world) if smooth is None else smooth
    pool, _, _, selection = label_jobs(body, key, smooth=smooth or job == "G1_smooth")[job]
    arrays, _ = _pool(world, key, pool)
    st, sl = job_pairs(arrays, selection, body["protocol"]["seed"])
    n, S = arrays["actions"].shape[:2]
    if not len(st):
        return None
    labels, man = LB.load_labels(host_dir(world, key) / "labels" / job)
    items = job_items(arrays, st, sl)
    require(all(SR.array_sha256(items[k]) == man["inputs"][k] for k in ("qpos", "qvel", "action", "item_id")),
            f"labels of {key}/{job} were made for other pool states or actions; refusing to score them")
    K = man["spec"]["repeats"]
    q = np.full((n, S, K), np.nan)
    length = np.zeros((n, S, K), np.int64)
    status = np.full((n, S, K), -1, np.int64)
    a, b = np.divmod(labels["item_id"], S)
    q[a, b], length[a, b], status[a, b] = labels["q"], labels["length"], labels["status"]
    return q, length, status


def labels_complete(world, body, key):
    for job in label_jobs(body, key, smooth=smoothing_required(world)):
        try:
            load_job(world, body, key, job)
        except (ValueError, FileNotFoundError, KeyError):
            return False
    return True


# ---------------------------------------------------------------- G1-pre and G1

def g1_pre_values(q1, qpi, rough, gap):
    """Truth-noise share on G for one gapped host: mean jitter variance over the rough pairs / Var(e) over G's
    (logged, pi) points; and the definition gap (noisy-target mean - Q^pi) on the same pairs."""
    e = (q1 - qpi)[np.isfinite(qpi)]
    have = np.isfinite(rough).all(axis=-1)
    jv = rough[have].var(axis=-1, ddof=1) if rough.shape[-1] >= 2 else np.zeros(0)
    var_e = float(np.var(e, ddof=1)) if len(e) >= 2 else None
    mean_jv = float(jv.mean()) if len(jv) else None
    gh = np.isfinite(gap).all(axis=-1) & np.isfinite(qpi)
    dgap = gap[gh].mean(axis=-1) - qpi[gh]
    return dict(pairs=int(len(jv)), mean_jitter_variance=mean_jv, var_e_G=var_e,
                truth_noise_share=mean_jv / var_e if mean_jv is not None and var_e not in (None, 0.0) else None,
                jitter_sd_median=float(np.median(np.sqrt(jv))) if len(jv) else None,
                definition_gap=dict(pairs=int(len(dgap)), mean=float(dgap.mean()) if len(dgap) else None,
                                    mean_abs=float(np.abs(dgap).mean()) if len(dgap) else None,
                                    note="noisy-target mean minus Q^pi; reported separately, never critic error"))


def score_g1_pre(world, body, prereg_sha, code_changed=False):
    marks = body["pass_marks"]["G1_pre"]
    summary = read_json(Path(world) / "stage0" / "summary.json")
    require(SR.file_sha256(Path(world) / "stage0" / "summary.json") == body["stage0"]["summary_sha256"],
            "stage0/summary.json changed since the freeze")
    tol = body["gate_tolerances"]
    pilot = (summary["gate_rebuilt"] if body["state_source"] == "rebuilt" and summary.get("gate_rebuilt")
             else summary["gate"])
    gate = bool(A.gate_passes(pilot, tol) and not A.gate_passes(summary["gate_negative"], tol))
    per = {}
    for seed in body["protocol"]["seeds"]:
        key = host_key("gap", seed)
        arrays, _ = _pool(world, key, "G1")
        det = load_job(world, body, key, "G1", smooth=False)[0][..., 0]
        rough = load_job(world, body, key, "G1_rough", smooth=False)
        gapj = load_job(world, body, key, "G1_gap", smooth=False)
        g = arrays["code"] == 0
        q1 = arrays["q"][..., 0].astype(np.float64)
        rough_v = rough[0][g] if rough is not None else np.zeros((0, 2, 1))
        gap_v = gapj[0][g] if gapj is not None else np.zeros((0, 2, 1))
        per[str(seed)] = g1_pre_values(q1[g], det[g], rough_v, gap_v)
    shares = [v["truth_noise_share"] for v in per.values()]
    noise_ok = all(s is not None and s <= marks["truth_noise_share_max"] for s in shares)
    values = dict(gate=dict(source=pilot["source"], passes_under_frozen_tolerances=gate, tolerances=tol),
                  per_gapped_host=per, smoothed_truth_required=not noise_ok,
                  smoothed_rule="truth-noise share above the mark on any gapped host: the jitter mean (G1_smooth) "
                                "replaces the single rollouts in G1")
    passes = dict(gate=gate, truth_measured=all(s is not None for s in shares))
    return values, passes


def g1_tables(world, body, smooth):
    """Per host: e_A (logged minus pi slot), e at the logged action, the pi rollout length; rows finite under every
    host are kept (the same rows for every ratio)."""
    keys = planned_keys(body["protocol"])
    ref = None
    out = {}
    for key in keys:
        arrays, _ = _pool(world, key, "G1")
        if ref is None:
            ref = arrays
        require(np.array_equal(arrays["raw_row"], ref["raw_row"]), "G1 pools hold different rows across hosts")
        if smooth:
            q = load_job(world, body, key, "G1_smooth", smooth=True)[0]
            qpi = np.where(np.isfinite(q).all(axis=-1), np.nanmean(q, axis=-1), np.nan)
        else:
            qpi = load_job(world, body, key, "G1", smooth=False)[0][..., 0]
        length = load_job(world, body, key, "G1", smooth=False)[1][..., 0]
        q1 = arrays["q"][..., 0].astype(np.float64)
        e = q1 - qpi
        out[key] = dict(ea=e[:, 0] - e[:, 1], e_logged=e[:, 0], length_pi=length[:, 1].astype(np.float64))
    ok = np.all([np.isfinite(v["ea"]) for v in out.values()], axis=0)
    common = dict(code=ref["code"][ok], block=ref["episode"][ok], d_logged=ref["d_logged"][ok],
                  dropped_nonfinite=int((~ok).sum()))
    return {k: {n: v[ok] for n, v in t.items()} for k, t in out.items()}, common


def g1_score(tables, common, seeds, marks, replicates, seed_boot):
    """G1 on per-host arrays (pure NumPy; known-answer tested). tables[key] has ea, e_logged, length_pi; common has
    code (0 G, 1 G_held, 2 K_match, 3 K_far), block (original episode) and d_logged."""
    code, block = common["code"], common["block"]
    names = {0: "G", 1: "G_held", 2: "K_match", 3: "K_far"}
    gk = lambda s: host_key("gap", s)
    fk = lambda s: host_key("full", s)

    def pooled(mask, num, den):
        return geometric_mean([safe_ratio(median_abs(tables[num(s)]["ea"][mask]),
                                          median_abs(tables[den(s)]["ea"][mask])) for s in seeds])

    def boot(mask, num, den, tag):
        idx = np.flatnonzero(mask)
        if not len(idx):
            return np.zeros(0)
        # block_bootstrap hands the statistic positions into idx (with repeats): idx[i] are resampled rows.
        return A.block_bootstrap(block[idx], lambda i: pooled(idx[i], num, den), replicates, [seed_boot, tag])

    ratios, pooled_r, ci = {}, {}, {}
    for c, name in names.items():
        mask = code == c
        ratios[name] = {str(s): safe_ratio(median_abs(tables[gk(s)]["ea"][mask]), median_abs(tables[fk(s)]["ea"][mask]))
                        for s in seeds}
        pooled_r[name] = pooled(mask, gk, fk)
        ci[name] = list(interval(boot(mask, gk, fk, 30 + c), 0.95))
    G, KM = code == 0, code == 2
    cross = {str(s): safe_ratio(median_abs(tables[gk(s)]["ea"][G]), median_abs(tables[gk(s)]["ea"][KM])) for s in seeds}
    cross_pooled = geometric_mean(list(cross.values()))
    placebo_full, placebo_gap = {}, {}
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            placebo_full[f"{a}/{b}"] = safe_ratio(median_abs(tables[fk(a)]["ea"][G]),
                                                  median_abs(tables[fk(b)]["ea"][G]))
            placebo_gap[f"{a}/{b}"] = safe_ratio(median_abs(tables[gk(a)]["ea"][G]),
                                                 median_abs(tables[gk(b)]["ea"][G]))
    largest = max([magnitude(r) for r in placebo_full.values() if r is not None], default=None)
    raw_ratio = geometric_mean([safe_ratio(median_abs(tables[gk(s)]["e_logged"][G]),
                                           median_abs(tables[fk(s)]["e_logged"][G])) for s in seeds])
    signed = [float(np.mean(tables[gk(s)]["ea"][G] > 0)) if G.any() else None for s in seeds]
    strata = {name: pooled(G & m, gk, fk) for name, m in (("core", common["d_logged"] <= 0.5),
                                                          ("rim", common["d_logged"] > 0.5))}
    terciles = {}
    for s in seeds:
        length = tables[gk(s)]["length_pi"][G]
        if len(length) >= 3:
            cut = np.quantile(length, [1 / 3, 2 / 3])
            bins = np.digitize(length, cut)
            terciles[str(s)] = [safe_ratio(median_abs(tables[gk(s)]["ea"][G][bins == t]),
                                           median_abs(tables[fk(s)]["ea"][G][bins == t])) for t in range(3)]
    values = dict(rows={name: int((code == c).sum()) for c, name in names.items()},
                  rows_dropped_nonfinite=common.get("dropped_nonfinite", 0), ratios=ratios, pooled=pooled_r,
                  intervals95=ci, cross_section=cross, cross_section_pooled=cross_pooled,
                  placebo_full=placebo_full, placebo_gap=placebo_gap, largest_placebo_magnitude=largest,
                  raw_abs_e_ratio_G=raw_ratio, share_ea_positive_G=signed, core_rim=strata,
                  termination_terciles=terciles, pairing="gap_S vs full_S on the same frozen rows",
                  bootstrap=f"episode blocks (original D4RL episodes), {replicates} replicates")
    rg, lo_g = pooled_r["G"], ci["G"][0]
    rk, (k_lo, k_hi) = pooled_r["K_far"], ci["K_far"]
    rh, lo_h = pooled_r["G_held"], ci["G_held"][0]
    m = marks
    passes = dict(
        a=rg is not None and rg >= m["ratio_min"] and lo_g is not None and lo_g >= m["ratio_lower95_min"]
        and all(r is not None and r >= m["pair_ratio_min"] for r in ratios["G"].values()),
        b=rk is not None and 1 / m["kfar_band"] <= rk <= m["kfar_band"] and k_lo is not None
        and 1 / m["kfar_ci_band"] <= k_lo and k_hi <= m["kfar_ci_band"],
        c=cross_pooled is not None and cross_pooled >= m["cross_section_min"],
        d=rh is not None and rh >= m["held_ratio_min"] and lo_h is not None and lo_h >= m["held_lower95_min"],
        e=bool(placebo_full) and all(r is not None and 1 / m["placebo_band"] <= r <= m["placebo_band"]
                                     for r in placebo_full.values())
        and rg is not None and largest is not None and rg >= m["placebo_margin"] * largest)
    return values, passes


# ---------------------------------------------------------------- G2 and G3

def g2_score(visits, seeds, marks, replicates, seed_boot, behaviour_mass, cross=None):
    """visits[key] = dict(in_gap (n,), near (n,), episode (n,), d_pi (n,)). Pure NumPy."""
    per = {}
    for key, v in visits.items():
        per[key] = dict(steps=int(len(v["in_gap"])), S_gap=float(np.mean(v["in_gap"])) if len(v["in_gap"]) else None,
                        S_near=float(np.mean(v["near"])) if len(v["near"]) else None,
                        in_gap_states=int(np.sum(v["in_gap"])),
                        d_over_rho=GAP._summary(v["d_pi"]))
    gaps = [host_key("gap", s) for s in seeds]
    total = sum(per[k]["steps"] for k in gaps)
    pooled = float(sum(np.sum(visits[k]["in_gap"]) for k in gaps) / total) if total else None
    g = np.random.default_rng([seed_boot, 41])
    episodes = {k: [visits[k]["in_gap"][visits[k]["episode"] == e] for e in np.unique(visits[k]["episode"])]
                for k in gaps}
    reps = []
    for _ in range(int(replicates)):
        hosts = [gaps[i] for i in g.integers(0, len(gaps), len(gaps))]
        num = den = 0
        for k in hosts:
            eps = episodes[k]
            if not eps:
                continue
            for i in g.integers(0, len(eps), len(eps)):
                num += eps[i].sum()
                den += len(eps[i])
        if den:
            reps.append(num / den)
    lo, hi = interval(reps, 0.95)
    host_ok = sum(1 for k in gaps if per[k]["S_gap"] is not None and per[k]["S_gap"] >= marks["host_share_min"])
    ratio = {str(s): safe_ratio(per[host_key("gap", s)]["S_gap"], per[host_key("full", s)]["S_gap"]) for s in seeds}
    few = [k for k in gaps if per[k]["in_gap_states"] < marks["in_gap_states_min"]]
    values = dict(per_host=per, pooled_S_gap_gapped=pooled, pooled_interval95=[lo, hi], hosts_above=host_ok,
                  ratio_gapped_over_full=ratio, behaviour_training_mass=behaviour_mass, cross_policy=cross,
                  hosts_below_in_gap_states_min=few, g3_pools_hosts=bool(few),
                  rule="steps summed over gapped hosts; bootstrap over hosts, then episodes")
    passes = dict(share=pooled is not None and pooled >= marks["share_min"] and lo is not None
                  and lo >= marks["share_lower95_min"],
                  hosts=host_ok >= marks["host_count_min"])
    return values, passes


def g3_score(returns, v3, sigma, seeds, marks, protocol, replicates, seed_boot, final_eval=None):
    """returns[key] = per-episode normalized scores on the common bank (aligned by reset seed); v3[key] =
    dict(in_gap, q1 (n, 2), qpi (n, 2), episode); sigma[seed] = the gapped host's median jitter SD. Pure NumPy."""
    from scipy import stats as st

    level = marks["ci_level"]
    dj, per_pair = [], {}
    g = np.random.default_rng([seed_boot, 51])
    for s in seeds:
        a, b = np.asarray(returns[host_key("gap", s)]), np.asarray(returns[host_key("full", s)])
        require(a.shape == b.shape, "paired banks differ in length")
        d = a - b
        boots = [d[g.integers(0, len(d), len(d))].mean() for _ in range(int(replicates))]
        per_pair[str(s)] = dict(delta_J=float(d.mean()), episode_interval=list(interval(boots, level)),
                                J_gap=float(a.mean()), J_full=float(b.mean()))
        dj.append(float(d.mean()))
    n = len(dj)
    pooled = float(np.mean(dj)) if n else None
    if n >= 2:
        sd = float(np.std(dj, ddof=1))
        tq = float(st.t.ppf(1 - (1 - level) / 2, n - 1))
        lo, hi = pooled - tq * sd / math.sqrt(n), pooled + tq * sd / math.sqrt(n)
        mde = (tq + float(st.norm.ppf(protocol["g3"]["power"]))) * sd / math.sqrt(n)
        how = f"t-interval over {n} pairs at {level}"
    else:
        lo, hi = per_pair[str(seeds[0])]["episode_interval"] if n else (None, None)
        mde, sd, how = None, None, "one pair: episode bootstrap"
    a_pass = pooled is not None and pooled <= -marks["return_cost_min"] and hi is not None and hi < 0
    second_look = bool(pooled is not None and pooled <= -marks["return_cost_min"] and not a_pass)
    o_in, o_out, blocks_in, blocks_out, full_in = [], [], [], [], {}
    misled, inside_states = [], 0
    for s in seeds:
        for arm in ARMS:
            key = host_key(arm, s)
            t = v3[key]
            ok = np.isfinite(t["qpi"][:, 0])
            scale = float(np.mean(np.abs(t["qpi"][ok, 0]))) if ok.any() else None
            if not scale:
                continue
            o = (t["q1"][:, 0] - t["qpi"][:, 0]) / scale
            if arm == "full":
                sel = ok & t["in_gap"]
                full_in[str(s)] = float(np.median(o[sel])) if sel.any() else None
                continue
            o_in.append(o[ok & t["in_gap"]])
            o_out.append(o[ok & ~t["in_gap"]])
            blocks_in.append(np.asarray([f"{s}:{e}" for e in t["episode"][ok & t["in_gap"]]]))
            blocks_out.append(np.asarray([f"{s}:{e}" for e in t["episode"][ok & ~t["in_gap"]]]))
            both = t["in_gap"] & np.isfinite(t["qpi"]).all(axis=1)
            inside_states += int(both.sum())
            if both.any() and sigma.get(str(s)) is not None:
                pred = t["q1"][both, 0] - t["q1"][both, 1]
                true = t["qpi"][both, 0] - t["qpi"][both, 1]
                misled.append((pred > 0) & (true < -marks["misled_jitter_sds"] * sigma[str(s)]))
    oi = np.concatenate(o_in) if o_in else np.zeros(0)
    oo = np.concatenate(o_out) if o_out else np.zeros(0)
    med_in = float(np.median(oi)) if len(oi) else None
    med_out = float(np.median(oo)) if len(oo) else None
    diff = med_in - med_out if med_in is not None and med_out is not None else None
    lo_b = None
    if len(oi) and len(oo):
        bi, bo = np.concatenate(blocks_in), np.concatenate(blocks_out)
        allv, allb = np.r_[oi, oo], np.r_[bi, bo]
        side = np.r_[np.ones(len(oi), bool), np.zeros(len(oo), bool)]

        def stat(idx):
            si = side[idx]
            if not si.any() or si.all():
                return None
            return float(np.median(allv[idx][si]) - np.median(allv[idx][~si]))

        lo_b = interval(A.block_bootstrap(allb, stat, replicates, [seed_boot, 52]), 0.95)[0]
    mis = np.concatenate(misled) if misled else np.zeros(0, bool)
    share = float(mis.mean()) if len(mis) else None
    values = dict(
        return_cost=dict(per_pair=per_pair, pooled_delta_J=pooled, interval=[lo, hi], level=level, how=how,
                         between_pair_sd=sd, mde=mde, second_look_eligible=second_look,
                         train_py_final_evaluation=final_eval),
        overestimate=dict(median_o_rel_in_gap=med_in, median_o_rel_out_of_gap=med_out, difference=diff,
                          difference_lower95=lo_b, in_gap_states=int(len(oi)), out_of_gap_states=int(len(oo)),
                          full_hosts_median_o_rel_in_gap=full_in),
        misled=dict(share=share, states=int(len(mis)), sigma_J=sigma,
                    rule="A_pred = Q1(pi) - Q1(a_out) > 0 and A_true = Q^pi(pi) - Q^pi(a_out) < -3 sigma_J"))
    passes = dict(
        a_return_cost=bool(a_pass),
        b_overestimate=bool(med_in is not None and med_in > marks["overestimate_median_min"] and diff is not None
                            and diff >= marks["overestimate_margin_min"] and lo_b is not None and lo_b > 0),
        c_misled=bool(share is not None and share >= marks["misled_share_min"]))
    values["mechanism"] = bool(passes["b_overestimate"] or passes["c_misled"])
    return values, passes


def final_evaluation(run_dir):
    """train.py's final 20-episode evaluation mean (normalized), when the run has one (a cross-check only)."""
    path = Path(run_dir) / "result.json"
    if not path.is_file():
        return None
    final = [e for e in read_json(path).get("evaluations", []) if e.get("kind") == "final"]
    if not final:
        return None
    scores = [ep["normalized_score"] for ep in final[-1].get("episodes", []) if "normalized_score" in ep]
    return float(np.mean(scores)) if scores else None


def stage_score(world, body, prereg_sha, *, requirements=None, data_dir=None, host_dirs=None, code_changed=False):
    """G1-pre, G1, G2 and G3 (each written once; existing ones are read). requirements limits the set (run_all
    scores G1-pre alone first, because its smoothing decision can add labels before G1)."""
    verify_stage(world, body, prereg_sha, code_changed)
    wanted = lambda req: req in body["plan"] and (requirements is None or req in requirements)
    pr, marks = body["protocol"], body["pass_marks"]
    out = Path(world) / "scores"
    seeds = pr["seeds"]
    reps, master = pr["bootstrap"]["replicates"], pr["seed"]
    results = {}

    def record(req, values, passes, extra=None):
        r = dict(stage_header(body, prereg_sha, "score", None, code_changed), requirement=req, values=values,
                 passes=passes, passed=bool(all(passes.values())), pass_marks=marks[req], **(extra or {}))
        write_once(out / f"{req}.json", r)
        return r

    if wanted("G1_pre"):
        if (out / "G1_pre.json").exists():
            results["G1_pre"] = read_json(out / "G1_pre.json")
        else:
            results["G1_pre"] = record("G1_pre", *score_g1_pre(world, body, prereg_sha, code_changed))
    m = manifest(world, body["gap_sha256"])
    pre = results.get("G1_pre") or (read_json(out / "G1_pre.json") if (out / "G1_pre.json").exists() else None)
    if pre is not None:
        require(pre["prereg_sha256"] == prereg_sha, "scores/G1_pre.json was written under another pre-registration")

    def integrity_passed():  # stop rule 3: no G result is computed until H0 and G1-pre pass
        return "G1_pre" not in body["plan"] or (pre is not None and pre["passed"])

    if wanted("G1") and not (out / "G1.json").exists():
        require(integrity_passed(), "G1-pre has not passed: an integrity failure stops G1")
        smooth = bool(pre["values"]["smoothed_truth_required"])
        if smooth:
            require(all(labels_complete(world, body, k) for k in planned_keys(pr)),
                    "the smoothed truth is required: run labels (job G1_smooth) on every host first")
        tables, common = g1_tables(world, body, smooth)
        values, passes = g1_score(tables, common, seeds, marks["G1"], reps, master)
        values["truth"] = "jitter mean (G1_smooth)" if smooth else "single deterministic rollout"
        results["G1"] = record("G1", values, passes)
    if wanted("G2") and not (out / "G2.json").exists():
        require(integrity_passed(), "G1-pre has not passed: an integrity failure stops G2 (stop rule 3)")
        visits = {k: {n: _pool(world, k, "V")[0][n] for n in ("in_gap", "near", "episode", "d_pi")}
                  for k in planned_keys(pr)}
        cross = {}
        for s in seeds:
            gkey, fkey = host_key("gap", s), host_key("full", s)
            hg, _ = host_for(world, body, gkey, data_dir, host_dirs)
            hf, _ = host_for(world, body, fkey, data_dir, host_dirs)
            obs_g, obs_f = _pool(world, gkey, "V")[0]["obs"], _pool(world, fkey, "V")[0]["obs"]
            pg, pf = hg.policy(pr["slots"]), hf.policy(pr["slots"])
            share = lambda policy, obs: float(np.mean(m.in_gap(policy.actions(obs)))) if len(obs) else None
            cross[str(s)] = dict(pi_gap_on_full_states=share(pg, obs_f), pi_full_on_gap_states=share(pf, obs_g))
        values, passes = g2_score(visits, seeds, marks["G2"], reps, master, m.data["gap"]["masses"]["training"], cross)
        results["G2"] = record("G2", values, passes)
    if wanted("G3") and not (out / "G3.json").exists():
        require(integrity_passed(), "G1-pre has not passed: an integrity failure stops G3 (stop rule 3)")
        V = {k: _pool(world, k, "V")[0] for k in planned_keys(pr)}
        for s in seeds:  # paired by seed AND by reset state: both hosts of a pair ran the same reset bank, in order
            require(np.array_equal(V[host_key("gap", s)]["reset_seeds"], V[host_key("full", s)]["reset_seeds"]),
                    f"the V pools of seed {s} ran different reset banks; G3's return pairing is broken")
        returns = {k: V[k]["episode_scores"] for k in planned_keys(pr)}
        v3, sigma = {}, {}
        for k in planned_keys(pr):
            arrays, _ = _pool(world, k, "V3")
            det = load_job(world, body, k, "V3")
            qpi = det[0][..., 0] if det is not None else np.full(arrays["actions"].shape[:2], np.nan)
            v3[k] = dict(in_gap=arrays["in_gap"], q1=arrays["q"][..., 0].astype(np.float64), qpi=qpi,
                         episode=arrays["episode"])
            arm, s = parse_key(k)
            if arm == "gap":
                rough = load_job(world, body, k, "V3_rough")
                if rough is not None:
                    v = rough[0][np.isfinite(rough[0]).all(axis=-1)]
                    sigma[str(s)] = float(np.median(v.std(axis=-1, ddof=1))) if len(v) else None
                else:
                    sigma[str(s)] = None
        final = {k: final_evaluation(body["hosts"][k]["run_dir"]) for k in planned_keys(pr)}
        values, passes = g3_score(returns, v3, sigma, seeds, marks["G3"], pr, reps, master, final)
        results["G3"] = record("G3", values, passes)
    for req in ("G1", "G2", "G3"):
        if req not in results and (out / f"{req}.json").exists():
            results[req] = read_json(out / f"{req}.json")
    return results


# ---------------------------------------------------------------- the stop rule

def g4_gate(world, body):
    """(allowed, reason) for G4a: G1 passes all of (a)-(e), G2 passes, G3's mechanism part passes."""
    if body["protocol"]["stop_rule"] == "exercise_all_stages":
        require(body["synthetic"], "the stop rule is enforced on real data")
        return True, "SYNTHETIC smoke: the stop rule is not applied (exercise_all_stages)"
    out = Path(world) / "scores"
    if not all((out / f"{r}.json").exists() for r in ("G1", "G2", "G3")):
        return False, "G1, G2 and G3 must be scored first"
    g1, g2, g3 = (read_json(out / f"{r}.json") for r in ("G1", "G2", "G3"))
    if not g1["passes"]["e"]:
        return False, "G1(e) failed (seed placebo): G1 is inconclusive and G4 is not run"
    if not g1["passed"]:
        return False, "G1 did not pass all of (a)-(e)"
    if not g2["passed"]:
        return False, "G2 did not pass"
    if not g3["values"]["mechanism"]:
        return False, "G3's mechanism part ((b) or (c)) did not pass"
    return True, "G1, G2 and G3's mechanism passed"


def g4a_placements(world, body):
    """{placement: passes on every gapped critic} from the G4a files, or None when one is missing."""
    out = {}
    for p in PLACEMENTS:
        vals = []
        for s in body["protocol"]["seeds"]:
            f = host_dir(world, host_key("gap", s)) / "scores" / "G4a.json"
            if not f.is_file():
                return None
            vals.append(read_json(f)["placement_passes"][p])
        out[p] = bool(all(vals))
    return out


def g4b_gate(world, body):
    if body["protocol"]["stop_rule"] == "exercise_all_stages":
        return True, "SYNTHETIC smoke: the stop rule is not applied (exercise_all_stages)"
    placements = g4a_placements(world, body)
    if placements is None:
        return False, "G4a is incomplete"
    if not any(placements.values()):
        return False, "G4a passed for no placement"
    return True, f"G4a passed for {[p for p, ok in placements.items() if ok]}"


# ---------------------------------------------------------------- G4a (Level-1 screen) and G4b (extraction)

def oracle_arms(body, poolT, qpiT, m, n_shuf):
    """Weights on pool-T rows with truth: OR_abs / OR_signed (option C's oracles), OR_GAP (rank by ascending
    ||pi_host(s_i) - c*||, seeded ties), their ANTI arms and SHUF_k (OR_abs's multiset permuted)."""
    from experiments.signal.d4rl_level1 import assignments
    from experiments.truth import extraction as EX

    pr = body["protocol"]
    r4a = pr["g4"]["r4a"]
    q1 = poolT["q"][..., 0].astype(np.float64)
    e = q1 - qpiT
    ok = np.isfinite(e).all(axis=1)
    require(ok.sum() >= 2, "pool T truth is missing for almost every row")
    e_a = e[ok, 0] - e[ok, 1]
    s, clip = r4a["s"], r4a["clip"]
    sig_abs, shuf, anti_abs, _ = EX.oracle_weights(e_a, "abs", pr["seed"], s=s, clip=clip, n_shuf=n_shuf)
    sig_signed, _, anti_signed, _ = EX.oracle_weights(e_a, "signed", pr["seed"], s=s, clip=clip)
    d = m.d_over_rho(poolT["actions"][ok, 1])
    tie = np.random.default_rng([pr["seed"], 3]).permutation(len(d))
    gap_assigned, _ = assignments(-d, tie, [], [], s, clip)
    arms = {"OR_abs": sig_abs, "OR_signed": sig_signed, "OR_GAP": gap_assigned["SIG"], "ANTI_abs": anti_abs,
            "ANTI_signed": anti_signed, "ANTI_GAP": gap_assigned["ANTI"]}
    arms.update({f"SHUF_{k}": w for k, w in enumerate(shuf)})
    return arms, d, ok


def cng(weights, d, radius):
    """Caution-near-gap concentration: [share of sum (v - 1)+ on rows with d <= radius] / [their share of rows]
    for v = 1 + beta M (beta > 0 cancels)."""
    w = np.maximum(np.asarray(weights, np.float64), 0.0)
    near = np.asarray(d) <= radius
    if not near.any() or w.sum() <= 0:
        return None
    return float((w[near].sum() / w.sum()) / near.mean())


def truth_gradient_Pi(world, body, key):
    arrays, _ = _pool(world, key, "Pi")
    n, S, _ = arrays["actions"].shape
    q = load_job(world, body, key, "Pi")[0][..., 0]
    dd = (S - 3) // 2
    grad = (q[:, 2:2 + dd] - q[:, 2 + dd:2 + 2 * dd]) / arrays["cd_step"]
    ok = np.isfinite(grad).all(axis=1)
    return arrays["obs"][ok], grad[ok], int((~ok).sum())


def stage_screen(world, body, prereg_sha, seed, *, data_dir=None, host_dirs=None, code_changed=False):
    """G4a on one gapped critic."""
    verify_stage(world, body, prereg_sha, code_changed)
    from experiments.truth import extraction as EX

    allowed, reason = g4_gate(world, body)
    require(allowed, f"stop rule: {reason}; G4a is not run")
    key = host_key("gap", seed)
    pr, marks = body["protocol"], body["pass_marks"]["G4a"]
    host, m = host_for(world, body, key, data_dir, host_dirs)
    poolT, _ = _pool(world, key, "T")
    qpiT = load_job(world, body, key, "T")[0][..., 0]
    arms, d, keep = oracle_arms(body, poolT, qpiT, m, pr["g4"]["r4a"]["shuf"])
    obs_Pi, grad_Pi, dropped = truth_gradient_Pi(world, body, key)
    screen = EX.Level1Screen(host, poolT["obs"][keep], poolT["actions"][keep, 0], obs_Pi, grad_Pi)
    level = pr["g4"]["r4a"]["level"]
    names = list(arms)
    b = screen.b_of(np.stack([arms[k] for k in names]))
    p0 = screen.gains((level - 1.0) * screen.b0)
    records = {}
    for k, name in enumerate(names):
        beta = screen.match_beta(b[k], level)
        records[name] = dict(beta=beta, gains=None if beta is None else screen.gains(beta * b[k]),
                             cos_with_p0=screen.cosine(b[k], screen.b0))
    shuf = [records[k]["gains"] for k in names if k.startswith("SHUF_") and records[k]["gains"] is not None]
    pct = {kind: float(np.percentile([g[kind] for g in shuf], marks["shuf_percentile"])) if shuf else None
           for kind in ("raw", "adam")}

    def passes_for(name, kind="raw"):
        g = records[name]["gains"]
        if g is None or pct[kind] is None:
            return dict(margin=False, above_shuf=False)
        return dict(margin=bool(g[kind] > 0 and g[kind] - p0[kind] >= marks["gain_margin_min"] * abs(p0[kind])),
                    above_shuf=bool(g[kind] > pct[kind]))

    placement = {p: passes_for(p) for p in PLACEMENTS}
    values = dict(level=level, rho_host=screen.rho_host, lambda_Q=screen.lam, pool_T_rows=int(screen.N),
                  pool_T_rows_without_truth=int((~keep).sum()), pool_Pi_states=int(len(obs_Pi)),
                  pool_Pi_dropped_nonfinite=dropped, p0_gains=p0, arms=records, shuf_percentile=pct,
                  preconditioner=screen.preconditioner, primary="raw first-order gain",
                  adam_reported={p: passes_for(p, "adam") for p in PLACEMENTS},
                  near_gap_rows_share=float(np.mean(d <= body["pass_marks"]["G4b"]["cng_radius"])),
                  stop_rule=reason)
    record = dict(stage_header(body, prereg_sha, "screen", key, code_changed), requirement="G4a", values=values,
                  passes={p: v for p, v in placement.items()},
                  placement_passes={p: bool(all(v.values())) for p, v in placement.items()},
                  passed=bool(any(all(v.values()) for v in placement.values())), pass_marks=marks)
    write_once(host_dir(world, key) / "scores" / "G4a.json", record)
    return record


def g4_dir(world, seed):
    return Path(world) / "g4" / f"critic_{int(seed)}"


def stage_extract(world, body, prereg_sha, seed, *, data_dir=None, host_dirs=None, log=print, code_changed=False):
    """G4b's actor extraction on one frozen gapped critic (TRAINS ACTORS: oracle and control arms only)."""
    verify_stage(world, body, prereg_sha, code_changed)
    from flax import serialization

    from experiments.truth import extraction as EX

    allowed, reason = g4b_gate(world, body)
    require(allowed, f"stop rule: {reason}; G4b is not run")
    pr = body["protocol"]
    ex = pr["g4"]["extraction"]
    key = host_key("gap", seed)
    out = g4_dir(world, seed)
    require(not (out / "matched.json").exists(), f"{out / 'matched.json'} exists; extraction runs once")
    screen = read_json(host_dir(world, key) / "scores" / "G4a.json")
    require(screen["prereg_sha256"] == prereg_sha, "G4a ran under another pre-registration")
    host, m = host_for(world, body, key, data_dir, host_dirs)
    poolT, _ = _pool(world, key, "T")
    qpiT = load_job(world, body, key, "T")[0][..., 0]
    arms, _, keep_T = oracle_arms(body, poolT, qpiT, m, ex["shuf"])
    n = int(keep_T.sum())
    stats = EX.PolicyStats(host, A.withheld_states(host, ex["s_states"], pr["seed"]))
    extractor = EX.Extractor(host, poolT["converted_id"][keep_T], batch=ex["batch"], block=ex["block"])
    ones = np.ones(n, np.float32)
    level = ex["level"]
    units = {}
    out.mkdir(parents=True, exist_ok=True)
    with A.forbid_simulator():
        for unit in ex["seeds"]:
            udir = out / f"unit_{unit}"
            udir.mkdir(exist_ok=False)
            policies = {}

            def train(name, v, kind, knob):
                params, diag = extractor.train(v, int(seed) * 1000 + int(unit), ex["actor_steps"])
                payload = serialization.to_bytes(params)
                (udir / A._policy_file(name)).write_bytes(payload)
                pi, q = stats(params)
                policies[name] = dict(kind=kind, knob=knob, file=A._policy_file(name),
                                      params_sha256=sha256_hex(payload), q1_mean_withheld=float(q.mean()),
                                      training=diag)
                return pi

            pi_none = train("NONE", ones, "NONE", 1.0)
            policies["NONE"]["S"] = 0.0
            for mult in ex["p0_ladder"]:
                pi = train(f"P0({mult})", ones * np.float32(mult), "P0", float(mult))
                policies[f"P0({mult})"]["S"] = EX.strength(pi, pi_none)
            target = policies[f"P0({level})"]["S"]
            matched = {}
            for arm, w in arms.items():
                centre = screen["values"]["arms"].get(arm, {}).get("beta")
                ladder = [centre * f for f in ex["ladder_factors"]] if centre else list(ex["fallback_ladder"])
                svals = []
                for j, beta in enumerate(ladder):
                    name = f"{arm}@L{j}"
                    pi = train(name, (1.0 + beta * w).astype(np.float32), arm, float(beta))
                    policies[name]["S"] = EX.strength(pi, pi_none)
                    svals.append(policies[name]["S"])
                beta_star, rule = EX.match_ladder(ladder, svals, target)
                rec = dict(ladder=ladder, ladder_S=svals, ladder_source="G4a beta x factors" if centre else
                           "fallback ladder", rule=rule, beta=beta_star)
                if beta_star is not None:
                    pi = train(f"{arm}*", (1.0 + beta_star * w).astype(np.float32), arm, float(beta_star))
                    s_star = EX.strength(pi, pi_none)
                    policies[f"{arm}*"]["S"] = s_star
                    rec.update(S=s_star, fidelity=abs(s_star / target - 1.0) if target > 0 else None)
                matched[arm] = rec
                log(f"gapped critic {seed} unit {unit} {arm}: beta* {beta_star}")
            units[str(unit)] = dict(policies=policies, S_target=target, matched=matched,
                                    target_definition=f"S(P0({level})) against NONE")
    record = dict(stage_header(body, prereg_sha, "extract", key, code_changed), units=units,
                  simulator_forbidden_during_matching=True, pool_T_rows=n,
                  pool_T_rows_without_truth=int((~keep_T).sum()), host=host.identity, arms=sorted(arms),
                  s_states=int(len(stats.s)), stop_rule=reason)
    path = write_once(out / "matched.json", record)
    (out / "matched.json.sha256").write_text(f"{SR.file_sha256(path)}  matched.json\n", encoding="utf8")
    return record


def stage_evaluate(world, body, prereg_sha, seed, *, data_dir=None, host_dirs=None, log=print, code_changed=False):
    """Verify the matching lock, then score every needed policy on the common extraction reset bank."""
    verify_stage(world, body, prereg_sha, code_changed)
    from flax import serialization

    from experiments.truth import extraction as EX

    pr = body["protocol"]
    ex = pr["g4"]["extraction"]
    key = host_key("gap", seed)
    out = g4_dir(world, seed)
    lock = (out / "matched.json.sha256").read_text(encoding="utf8").split()[0]
    require(SR.file_sha256(out / "matched.json") == lock, "matched.json changed after its lock; refusing to evaluate")
    matched = read_json(out / "matched.json")
    require(matched["prereg_sha256"] == prereg_sha, "matched.json was written under another pre-registration")
    host, _ = host_for(world, body, key, data_dir, host_dirs)
    policy = host.policy(ex["eval_slots"])
    invariance = A.require_invariant(policy, host)
    physics = SR.Physics(host.task.dataset)
    seeds = [ex["eval_reset_seed_base"] + i for i in range(ex["eval_episodes"])]
    results = {}
    try:
        q0, v0 = PL.reset_bank(physics, seeds)
        for unit, u in matched["units"].items():
            names = ["NONE"] + [f"P0({x})" for x in ex["p0_ladder"]]
            names += [k for k in u["policies"] if any(k.startswith(f"{p}@") for p in PLACEMENTS)]
            names += [f"{arm}*" for arm, r in u["matched"].items() if r.get("beta") is not None]
            scores = {}
            for name in names:
                p = u["policies"][name]
                payload = (out / f"unit_{unit}" / p["file"]).read_bytes()
                require(sha256_hex(payload) == p["params_sha256"], f"{p['file']} differs from matched.json")
                params = serialization.from_bytes(host.native.actor.params, payload)
                sc, ep, _ = EX.evaluate(physics, policy, params, q0, v0, host.task.dataset, horizon=pr["horizon"])
                scores[name] = dict(mean=float(sc.mean()), episodes=sc.tolist(),
                                    mean_length=float(ep["lengths"].mean()), simulator_errors=int(ep["failed"].sum()))
            results[unit] = scores
            log(f"gapped critic {seed} unit {unit}: evaluated {len(names)} policies")
    finally:
        physics.close()
    record = dict(stage_header(body, prereg_sha, "evaluate", key, code_changed), matched_lock_verified=True,
                  position_invariance=invariance, matched_sha256=lock, reset_seeds=[seeds[0], seeds[-1]],
                  episodes=len(seeds), units=results)
    return write_once(out / "evaluation.json", record)


def g4b_score(J, matched_all, cos, cngs, seeds, ex, marks, m7, r7, replicates, boot_seed):
    """G4b and R7 on evaluation means (pure). J[(seed, unit)] = {policy: mean return}."""
    from scipy import stats as st

    units = sorted(J)
    level = ex["level"]
    get = lambda u, name: J[u].get(name)
    by_seed = lambda f: [[f(u) for u in units if u[0] == s and f(u) is not None] for s in seeds]
    mean_of = lambda name: (float(np.mean([get(u, name) for u in units]))
                            if units and all(get(u, name) is not None for u in units) else None)
    p0_names = ["NONE"] + [f"P0({x})" for x in ex["p0_ladder"]]
    p0_means = {k: mean_of(k) for k in p0_names}
    p0_best = max([v for v in p0_means.values() if v is not None], default=None)
    shuf_names = [f"SHUF_{k}*" for k in range(ex["shuf"])]
    per, mdes = {}, {}
    for p in PLACEMENTS:
        d_match = lambda u, p=p: (get(u, f"{p}*") - get(u, f"P0({level})")) if get(u, f"{p}*") is not None else None
        dm = np.array([d_match(u) for u in units if d_match(u) is not None])
        lo, hi = hierarchical_interval(by_seed(d_match), replicates, [boot_seed, 81 + PLACEMENTS.index(p)],
                                       marks["ci_level"])
        names = sorted({k for u in units for k in J[u] if k.startswith(f"{p}@")} | {f"{p}*"})
        best = max([v for v in (mean_of(k) for k in names) if v is not None], default=None)

        def d_shuf(u, p=p):
            vals = [get(u, k) for k in shuf_names]
            if get(u, f"{p}*") is None or any(v is None for v in vals):
                return None
            return get(u, f"{p}*") - float(np.mean(vals))

        ds = [d_shuf(u) for u in units if d_shuf(u) is not None]
        sd = float(dm.std(ddof=1)) if len(dm) >= 2 else None
        mdes[p] = ((st.norm.ppf(1 - r7["alpha"] / 2) + st.norm.ppf(r7["power"])) * sd / math.sqrt(len(dm))
                   if sd is not None else None)
        values = dict(matched_differences=dm.tolist(), matched_margin=float(dm.mean()) if len(dm) else None,
                      matched_interval=[lo, hi], best_mean=best, best_margin=(best - p0_best)
                      if best is not None and p0_best is not None else None,
                      shuf_margin=float(np.mean(ds)) if ds else None, cng=cngs.get(p), mde=mdes[p])
        passes = dict(matched=values["matched_margin"] is not None and values["matched_margin"]
                      >= marks["matched_margin_min"] and lo is not None and lo > 0,
                      best=values["best_margin"] is not None and values["best_margin"] >= marks["best_margin_min"],
                      shuf=values["shuf_margin"] is not None and values["shuf_margin"] >= marks["shuf_margin_min"])
        per[p] = dict(values=values, passes=passes, passed=bool(all(passes.values())))
    abs_ok, gap_ok = per["OR_abs"]["passed"], per["OR_GAP"]["passed"]
    c_abs = cngs.get("OR_abs")
    if gap_ok or (abs_ok and c_abs is not None and c_abs >= marks["cng_min"]):
        room = "room near the gap"
    elif abs_ok and not gap_ok:
        room = "room, not at the gap"
    else:
        room = "no room"
    known = [get(u, "OR_signed*") - get(u, "ANTI_signed*") for u in units
             if get(u, "OR_signed*") is not None and get(u, "ANTI_signed*") is not None]
    z_known = A.paired_z(known)
    pairs = []
    for i in range(ex["shuf"]):
        for j in range(i + 1, ex["shuf"]):
            d = [get(u, f"SHUF_{i}*") - get(u, f"SHUF_{j}*") for u in units
                 if get(u, f"SHUF_{i}*") is not None and get(u, f"SHUF_{j}*") is not None]
            z = A.paired_z(d)
            pairs.append(dict(pair=[i, j], z=finite_or_none(z), z_infinite=z is not None and math.isinf(z),
                              fires=z is not None and abs(z) > m7["null_z"], n=len(d)))
    fired = sum(1 for p in pairs if p["fires"])
    fid = [r.get("fidelity") for mu in matched_all.values() for r in mu["matched"].values()]
    unmatched = sum(1 for f in fid if f is None)
    fid_ok = [f for f in fid if f is not None]
    v7 = dict(known_answer=float(np.mean(known)) if known else None, known_answer_z=finite_or_none(z_known),
              known_answer_n=len(known), null_pairs=pairs, null_pairs_fired=fired, mde=mdes,
              max_match_error=max(fid_ok) if fid_ok else None, unmatched_arms=unmatched, level1_cosines=cos,
              blind=True)
    p7 = dict(a=v7["known_answer"] is not None and v7["known_answer"] >= m7["known_answer_min"]
              and z_known is not None and z_known >= m7["known_answer_z_min"],
              b=len(pairs) > 0 and fired <= m7["null_pairs_max"],
              c=all(v is not None and v <= m7["mde_max"] for v in mdes.values()),
              d=unmatched == 0 and bool(fid_ok) and max(fid_ok) <= m7["match_tolerance"]
              and all(c is not None and c < m7["cosine_max"] for cs in cos.values() for c in cs.values()),
              e=True)
    if not p7["c"]:
        room += " (inconclusive: R7(c) MDE above the mark; add extraction seeds up to 8 units per critic)"
    v4 = dict(units=len(units), critics=seeds, placements=per, p0_best_mean=p0_best,
              p0_best_rule="max over NONE (m = 1) and the P0 ladder of the mean J over units", verdict=room,
              ci_level=marks["ci_level"])
    p4 = {p: per[p]["passed"] for p in PLACEMENTS}
    return (v4, p4), (v7, p7)


def stage_g4_score(world, body, prereg_sha, *, code_changed=False):
    verify_stage(world, body, prereg_sha, code_changed)
    pr, marks = body["protocol"], body["pass_marks"]
    ex = pr["g4"]["extraction"]
    seeds = pr["seeds"]
    m = manifest(world, body["gap_sha256"])
    J, matched_all, cos, cng_by = {}, {}, {}, {p: [] for p in PLACEMENTS}
    for seed in seeds:
        key = host_key("gap", seed)
        out = g4_dir(world, seed)
        evaluation, matched = read_json(out / "evaluation.json"), read_json(out / "matched.json")
        require(evaluation["prereg_sha256"] == prereg_sha == matched["prereg_sha256"], "mixed pre-registrations")
        require(evaluation["matched_lock_verified"] and matched["simulator_forbidden_during_matching"], "R7e broken")
        screen = read_json(host_dir(world, key) / "scores" / "G4a.json")
        cos[str(seed)] = {p: screen["values"]["arms"][p]["cos_with_p0"] for p in PLACEMENTS}
        poolT, _ = _pool(world, key, "T")
        qpiT = load_job(world, body, key, "T")[0][..., 0]
        arms, d, _ = oracle_arms(body, poolT, qpiT, m, ex["shuf"])
        for p in PLACEMENTS:
            cng_by[p].append(cng(arms[p], d, marks["G4b"]["cng_radius"]))
        for unit, scores in evaluation["units"].items():
            J[(seed, unit)] = {k: v["mean"] for k, v in scores.items()}
            matched_all[(seed, unit)] = matched["units"][unit]
    cngs = {p: (float(np.mean(v)) if all(x is not None for x in v) else None) for p, v in cng_by.items()}
    (v4, p4), (v7, p7) = g4b_score(J, matched_all, cos, cngs, seeds, ex, marks["G4b"], marks["R7"], pr["r7"],
                                   pr["bootstrap"]["replicates"], pr["seed"])
    v4["cng_per_critic"] = cng_by
    out_dir = Path(world) / "scores"
    written = {}
    for req, values, passes, passed in (("G4b", v4, p4, any(p4.values())), ("R7", v7, p7, all(p7.values()))):
        if req not in body["plan"]:
            continue
        record = dict(stage_header(body, prereg_sha, "g4-score", None, code_changed), requirement=req, values=values,
                      passes=passes, passed=bool(passed), pass_marks=marks[req])
        write_once(out_dir / f"{req}.json", record)
        written[req] = record
    return written


# ---------------------------------------------------------------- verdict

KEYS = {"G1_pre": ("smoothed_truth_required",), "G1": ("pooled", "cross_section_pooled", "largest_placebo_magnitude"),
        "G2": ("pooled_S_gap_gapped", "pooled_interval95", "hosts_above"),
        "G3": ("mechanism",), "G4b": ("verdict",), "R7": ("known_answer", "null_pairs_fired", "mde")}


def verdict(world, body, prereg_sha, *, code_changed=False):
    verify_stage(world, body, prereg_sha, code_changed)
    world = Path(world)
    rows = []
    require(SR.file_sha256(world / "gap" / "g0.json") == body["g0"]["sha256"],
            "gap/g0.json differs from the one the pre-registration recorded; refusing")
    require(SR.file_sha256(world / "h0.json") == body["h0_sha256"],
            "h0.json differs from the one the final pre-registration recorded; refusing")
    g0 = read_json(world / "gap" / "g0.json")
    for part in ("g0a", "g0b"):
        r = g0.get(part)
        status = "missing" if r is None else ("pass" if r["passed"] else "fail")
        if status == "fail" and body["g0"]["synthetic_override"]:
            status = "fail (SYNTHETIC override)"
        rows.append(dict(requirement=part.upper(), status=status, passes=None if r is None else r["passes"]))
    h0 = read_json(world / "h0.json")
    rows.append(dict(requirement="H0", status="pass" if h0["passed"] else "fail",
                     normalization=h0["normalization"],
                     flags={s: dict(device=p.get("device_flag"), full_before_gap_prereg=p.get(
                         "full_written_before_gap_prereg")) for s, p in h0.get("pairs", {}).items()}))
    mixed, flags = [], []
    for req in body["plan"]:
        if req == "G4a":
            per = {}
            for s in body["protocol"]["seeds"]:
                f = host_dir(world, host_key("gap", s)) / "scores" / "G4a.json"
                if f.is_file():
                    r = read_json(f)
                    mixed += [str(f)] if r["prereg_sha256"] != prereg_sha else []
                    flags.append(r["code_changed_since_prereg"])
                    per[str(s)] = r["placement_passes"]
            allowed, reason = g4_gate(world, body)
            if len(per) == len(body["protocol"]["seeds"]):
                placements = {p: all(v[p] for v in per.values()) for p in PLACEMENTS}
                status = "pass" if any(placements.values()) else "fail"
                rows.append(dict(requirement=req, status=status, placements=placements, per_critic=per))
            else:
                rows.append(dict(requirement=req, status="missing" if allowed else "not run (stop rule)",
                                 reason=reason))
            continue
        f = world / "scores" / f"{req}.json"
        if f.is_file():
            r = read_json(f)
            mixed += [str(f)] if r["prereg_sha256"] != prereg_sha else []
            flags.append(r["code_changed_since_prereg"])
            rows.append(dict(requirement=req, status="pass" if r["passed"] else "fail", passes=r["passes"],
                             values={k: r["values"].get(k) for k in KEYS.get(req, ())}))
        else:
            allowed, reason = True, None
            if req in ("G4b", "R7"):
                allowed, reason = g4_gate(world, body)
                if allowed and g4a_placements(world, body) is not None:
                    allowed, reason = g4b_gate(world, body)
            rows.append(dict(requirement=req, status="missing" if allowed else "not run (stop rule)", reason=reason))
    require(not mixed, f"outputs from another pre-registration: {mixed}")
    statuses = {r["requirement"]: r["status"] for r in rows}
    g4b = next((r for r in rows if r["requirement"] == "G4b" and r["status"] in ("pass", "fail")), None)
    if g4b is not None:
        meaning = f"G4b: {g4b['values']['verdict']}"
    elif statuses.get("G1") in ("pass", "fail"):
        meaning = ("G1-G3 scored; G4 " + ("pending" if g4_gate(world, body)[0] else
                                          f"not run under the stop rule ({g4_gate(world, body)[1]})"))
    else:
        meaning = "incomplete: see rows"
    if body["synthetic"]:
        meaning = "SYNTHETIC TEST RUN, NOT A FINDING: " + meaning
    post_hoc = dict(code_changed=bool(code_changed or any(flags)),
                    pass_marks_equal_design=body["pass_marks_equal_design"],
                    protocol_equal_design=body.get("protocol_equal_design", False),
                    g0_synthetic_override=body["g0"]["synthetic_override"],
                    stop_rule=body["protocol"]["stop_rule"])
    post_hoc["reasons"] = [text for flag, text in (
        (post_hoc["code_changed"], "analysis code changed after the freeze (--allow-code-change)"),
        (not post_hoc["pass_marks_equal_design"], "pass marks differ from the approved design's pinned digest"),
        (not post_hoc["protocol_equal_design"], "protocol (sizes, seeds, plan settings) differs from the approved "
                                                "design's pinned digest"),
        (post_hoc["g0_synthetic_override"], "G0 failed and was overridden (synthetic only)"),
        (post_hoc["stop_rule"] != "enforced", "the stop rule was not applied")) if flag]
    record = dict(stage_header(body, prereg_sha, "verdict", None, code_changed or any(flags)), rows=rows,
                  statuses=statuses, meaning=meaning, post_hoc_flags=post_hoc)
    path = world / "verdict.json"
    if path.exists():
        path.unlink()
    write_once(path, record)
    (world / "verdict.md").write_text(verdict_markdown(record), encoding="utf8")
    return record


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def verdict_markdown(record):
    lines = []
    if record["synthetic"]:
        lines += ["**SYNTHETIC TEST RUN. NOT A FINDING.** " + SYNTHETIC_NOTE, ""]
    lines += [f"# Action-gap verdict: {record['dataset']} (profile {record['profile']}, stop rule "
              f"{record['stop_rule']})", "", f"Pre-registration sha256 `{record['prereg_sha256']}`, gap sha256 "
              f"`{record['gap_sha256']}`. {record['meaning']}.", ""]
    reasons = record.get("post_hoc_flags", {}).get("reasons") or []
    if reasons:
        lines += ["**POST HOC:** " + "; ".join(reasons) + ".", ""]
    lines += ["| Stage | Status | Key values |", "|---|---|---|"]
    for row in record["rows"]:
        if "values" in row:
            text = ", ".join(f"{k} {_fmt(v)}" for k, v in row["values"].items())
        elif "placements" in row:
            text = "placements " + _fmt(row["placements"])
        elif "reason" in row:
            text = row["reason"]
        elif "flags" in row:
            text = "pair flags " + _fmt(row["flags"]) + "; normalization " + _fmt(
                {s: v["flagged"] for s, v in row["normalization"].items()})
        elif "passes" in row and row["passes"]:
            text = ", ".join(f"{k} {v}" for k, v in row["passes"].items())
        else:
            text = ""
        lines.append(f"| {row['requirement']} | {row['status']} | {text} |")
    return "\n".join(lines) + "\n"


def combine(g4b_paths):
    """Cross-task G4b claim per placement (design X, option C's rule): the placement passes on at least one task and
    the other task's matched margin (point estimate) is not negative. Takes the two tasks' W/scores/G4b.json."""
    records = [read_json(p) for p in g4b_paths]
    require(len(records) == 2 and len({r["dataset"] for r in records}) == 2, "two G4b records of two tasks")
    out = {}
    for p in PLACEMENTS:
        passed = {r["dataset"]: bool(r["values"]["placements"][p]["passed"]) for r in records}
        margins = {r["dataset"]: r["values"]["placements"][p]["values"]["matched_margin"] for r in records}
        claim = any(passed.values()) and all(v is not None and v >= 0 for v in margins.values())
        out[p] = dict(passed_by_task=passed, matched_margins=margins, cross_task_claim=bool(claim))
    synthetic = any(r["synthetic"] for r in records)
    return dict(rule="a placement passes on at least one task and the other task's matched margin is not negative",
                placements=out, synthetic=synthetic, note=SYNTHETIC_NOTE if synthetic else "pre-registered cross-task "
                "rule")


# ---------------------------------------------------------------- run everything planned

def run_all(world, prereg_sha, *, data_dir=None, host_dirs=None, allow_code_change=False, log=print):
    body, sha, changed = load_prereg(world, prereg_sha, allow_code_change)
    kw = dict(data_dir=data_dir, host_dirs=host_dirs, code_changed=changed)
    keys = planned_keys(body["protocol"])
    for key in keys:
        if not (host_dir(world, key) / "pools" / "summary.json").exists():
            log(f"pools: {key}")
            stage_pools(world, body, sha, key, **kw)
        if not labels_complete(world, body, key):
            log(f"labels: {key}")
            stage_labels(world, body, sha, key, log=log, **kw)
    if "G1_pre" in body["plan"] and not (Path(world) / "scores" / "G1_pre.json").exists():
        stage_score(world, body, sha, requirements=["G1_pre"], **kw)  # its smoothing decision can add labels
    if smoothing_required(world):
        for key in keys:
            if not labels_complete(world, body, key):
                log(f"labels (smoothed truth): {key}")
                stage_labels(world, body, sha, key, log=log, **kw)
    stage_score(world, body, sha, **kw)
    if "G4a" in body["plan"]:
        allowed, reason = g4_gate(world, body)
        log(f"G4a: {reason}")
        if allowed:
            for seed in body["protocol"]["seeds"]:
                if not (host_dir(world, host_key("gap", seed)) / "scores" / "G4a.json").exists():
                    stage_screen(world, body, sha, seed, **kw)
            allowed, reason = g4b_gate(world, body)
            log(f"G4b: {reason}")
            if allowed and "G4b" in body["plan"]:
                for seed in body["protocol"]["seeds"]:
                    out = g4_dir(world, seed)
                    if not (out / "matched.json").exists():
                        stage_extract(world, body, sha, seed, log=log, **kw)
                    if not (out / "evaluation.json").exists():
                        stage_evaluate(world, body, sha, seed, log=log, **kw)
                if not all((Path(world) / "scores" / f"{r}.json").exists() for r in ("G4b", "R7") if r in body["plan"]):
                    stage_g4_score(world, body, sha, code_changed=changed)
    return verdict(world, body, sha, code_changed=changed)


# ---------------------------------------------------------------- command line

def _hosts(values):
    out = {}
    for v in values or []:
        key, _, path = v.partition("=")
        parse_key(key)
        require(path, f"--host takes KEY=RUN_DIR (KEY gap-<seed> or full-<seed>), got {v!r}")
        out[key] = path
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Action-gap protocol (see the module docstring).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preregister", help="freeze plan, sizes, seeds and marks before any host run")
    p.add_argument("--world", required=True, type=Path)
    p.add_argument("--task", required=True)
    p.add_argument("--gap-sha256", required=True)
    p.add_argument("--profile", default="design", choices=sorted(PROFILES))
    p.add_argument("--plan", default="full", choices=sorted(PLANS))
    for name in ("stage0", "freeze"):
        p = sub.add_parser(name)
        p.add_argument("--world", required=True, type=Path)
        p.add_argument("--gap-sha256", required=True)
        p.add_argument("--gap-prereg-sha256", required=True)
        p.add_argument("--host", action="append", required=True, help="KEY=RUN_DIR, KEY gap-<seed> or full-<seed>")
        p.add_argument("--data-dir", type=Path)
        if name == "stage0":
            p.add_argument("--rows", type=int, default=1000)
            p.add_argument("--seed", type=int, default=0)
        else:
            p.add_argument("--state-source", choices=("auto", "rebuilt"), default="auto")
            for tol, default in SR.DEFAULT_TOLERANCES.items():
                p.add_argument("--" + tol.replace("_", "-"), type=float, default=default)
    stages = ("pools", "labels", "score", "screen", "extract", "evaluate", "g4-score", "verdict", "run")
    for name in stages:
        p = sub.add_parser(name)
        p.add_argument("--world", required=True, type=Path)
        p.add_argument("--prereg-sha256", required=True)
        p.add_argument("--data-dir", type=Path)
        p.add_argument("--host-dir", action="append", help="KEY=RUN_DIR override of a frozen run directory")
        p.add_argument("--allow-code-change", action="store_true")
        if name in ("pools", "labels"):
            p.add_argument("--host", required=True, help="gap-<seed> or full-<seed>")
        if name in ("screen", "extract", "evaluate"):
            p.add_argument("--critic", type=int, required=True, help="seed of a gapped host")
        if name == "labels":
            p.add_argument("--jobs", nargs="*")
            p.add_argument("--worker", type=int, default=0)
            p.add_argument("--workers", type=int, default=1)
    cb = sub.add_parser("combine", help="cross-task G4b rule over two tasks' W/scores/G4b.json")
    cb.add_argument("g4b", nargs=2, type=Path)
    args = parser.parse_args(argv)
    if args.command == "combine":
        print(LB.strict_json(jsonable(combine(args.g4b))))
        return
    if args.command == "preregister":
        print(preregister(args.world, args.task, args.gap_sha256, profile=args.profile, plan=args.plan))
        return
    if args.command == "stage0":
        _, h0 = stage0(args.world, args.gap_sha256, args.gap_prereg_sha256, _hosts(args.host), data_dir=args.data_dir,
                       rows=args.rows, seed=args.seed)
        print(LB.strict_json(jsonable(dict(h0_passed=h0["passed"], hosts={k: dict(passed=v["passed"],
                                                                                checks=v["checks"])
                                                                           for k, v in h0["hosts"].items()}))))
        return
    if args.command == "freeze":
        tol = {k: getattr(args, k) for k in SR.DEFAULT_TOLERANCES}
        print(freeze(args.world, gap_sha256=args.gap_sha256, gap_prereg_sha256=args.gap_prereg_sha256,
                     hosts=_hosts(args.host), tolerances=tol, state_source=args.state_source))
        return
    host_dirs = _hosts(args.host_dir)
    if args.command == "run":
        r = run_all(args.world, args.prereg_sha256, data_dir=args.data_dir, host_dirs=host_dirs,
                    allow_code_change=args.allow_code_change)
        print(verdict_markdown(r))
        return
    body, sha, changed = load_prereg(args.world, args.prereg_sha256, args.allow_code_change)
    kw = dict(data_dir=args.data_dir, host_dirs=host_dirs, code_changed=changed)
    if args.command == "pools":
        stage_pools(args.world, body, sha, args.host, **kw)
    elif args.command == "labels":
        stage_labels(args.world, body, sha, args.host, jobs=args.jobs, worker=args.worker, workers=args.workers, **kw)
    elif args.command == "score":
        stage_score(args.world, body, sha, **kw)
    elif args.command == "screen":
        stage_screen(args.world, body, sha, args.critic, **kw)
    elif args.command == "extract":
        stage_extract(args.world, body, sha, args.critic, **kw)
    elif args.command == "evaluate":
        stage_evaluate(args.world, body, sha, args.critic, **kw)
    elif args.command == "g4-score":
        stage_g4_score(args.world, body, sha, code_changed=changed)
    elif args.command == "verdict":
        print(verdict_markdown(verdict(args.world, body, sha, code_changed=changed)))


if __name__ == "__main__":
    main()
