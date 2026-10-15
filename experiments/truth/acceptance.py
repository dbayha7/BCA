"""Acceptance gate of the D4RL-RT world: Stage 0, the pre-registration and its SHA-256 gate, truth labels,
R1-R7 with the design's pass marks, the external-validity graft, and the verdict table.

Last piece of the truth instrument (experiments/truth/README.md has the commands and costs). The question the
world must be able to answer: does putting caution (a larger TD3+BC behaviour-cloning multiplier) where the critic
is actually wrong improve return, compared with spreading the same caution evenly? Seven requirements, each with a
numeric pass mark fixed in advance (PASS_MARKS, from design_d4rl_rt.json), decide whether this world can judge
that. No BCA arm is ever run here. The one BCA-band step (R5-signal) is implemented but refuses to run unless
--allow-bca-band is given; it is off by default and was never run in this build.

Order of work, per task (hopper = hopper-medium-v2, walker2d = walker2d-medium-replay-v2), in a world directory W:
  stage0   no training, no pre-registration needed. HDF5 key listing (state_restore.describe_file); the restore /
           one-step gate on a seeded 1,000-row pilot plus its misaligned-by-one negative control, and again from
           states rebuilt from the observation when the file has infos/*; the observation-rebuild check (infos
           state vs state rebuilt from the observation, every row); kNN thinness of the bank rows (pools
           .ThinnessIndex on the host's split); the eps-noise share of the TD target on the bank rows (needs the
           host checkpoint); the dataset's mean episode normalized return; the D-thin(0.1) row counts. Stage 0
           numbers are descriptive and come before the pre-registration; the pilot's misses inform the gate
           tolerances that freeze hashes.
  freeze   writes W/prereg.json = {"body": ..., "body_sha256": ...} and W/prereg.json.sha256. The body fixes the
           plan (which requirements are scored), the hosts (run directories and checkpoint sha256s), the protocol
           (pool sizes, seeds, slots, label modes, bootstrap, extraction arms and ladders), the pass marks, the gate
           tolerances with the Stage 0 pilot they rest on, and the sha256 of every analysis source file. It refuses
           when W already holds any labelled or trained output, so it always precedes them. It also refuses when,
           under the tolerances being frozen, the Stage 0 pilot gate fails or its misaligned-by-one negative
           control passes. --state-source rebuilt (hashed as protocol.state_source) restores every pool state
           from the observation even when the file has infos/*, for files whose infos states fail the gate.
  every later stage takes --prereg-sha256 HEX and refuses unless sha256(W/prereg.json) equals it, equals the
           sidecar, and the embedded body hash is right. It also refuses when an analysis source file changed
           since the freeze, unless --allow-code-change (then every output says so and the verdict marks it).
           The stage functions themselves re-check this (verify_stage), so a direct Python call with another
           body or hash refuses too.
           Pass marks and sizes are read from the pre-registration, never from code constants.
  pools    per critic: restore the host (pools.restore_host), build pools L, Pi and T, Q1/Q2 at every point,
           thinness c and d_k, the noiseless residual b and TD-target draws on L, and R6's inputs.
  labels   per critic: labeller.run_labels jobs (resumable shards; --worker/--workers split them):
             L, Pi   deterministic Q^pi at every slot of every state      T   deterministic at (data, pi)
             rough_L, rough_Pi  jitter mode, `repeats` rollouts at (pi, data) of `states_L` / `states_Pi` states
             gap_L, gap_Pi      noisy-target mode at the same pairs (the definition gap, reported separately)
  score    per critic, NumPy only: R1, R2, R3a, R5-world, R6.
  screen   per critic: R4a, the Level-1 first-order screen at the host actor (extraction.Level1Screen).
  extract  per critic, TRAINS ACTORS (R3b/R4b/R7): every arm of every unit (critic, extraction seed) on the frozen
           critic, strength matching to S(P0(2)) with the simulator forbidden (R7e), params and matched.json
           written with a sha256 lock before any return is computed.
  evaluate per critic: checks the lock, then the normalized return of every policy on one common reset bank;
           R3b's on-policy truth labels of the P0(0.25) arm.
  extraction-score   R3b, R4b, R7 over all critics (hierarchical bootstrap: critics, then seeds).
  verdict  the table of every planned requirement, its value, mark and pass/fail/missing, and the world verdict.
  run      every planned stage in order, skipping finished ones. combine: the cross-task R4b rule.
The graft (plan 'graft') is the judge's external-validity check: pools L and Pi and their labels on the one
existing Hopper checkpoint (seed 202609171), scored for R1, R2, R5-world and R6. It trains nothing.

Definitions used by the scores (all errors in raw reward units; e = Q1 - Q^pi; e_A(s, a) = e(s, a) - e(s, pi(s))):
  thin c <= thin_max (2), dense c >= dense_min (10); c and d_k as pools.ThinnessIndex.
  R1  (a) share of pool-Pi points (s, pi(s)) thin >= 0.10; (b) share of the bank rows (s, a_logged) thin >= 0.03;
      (c) share of pool-Pi points dense >= 0.40.
  R2  points: every non-pi slot of pools L and Pi. (a) Spearman(d_k, |e_A|) >= 0.30 and its 95% episode-block
      bootstrap lower bound >= 0.15 (blocks: the logged D4RL episode, or the host episode for Pi); (b) median |e_A|
      thin >= 3 x dense; (c) thin points' share of sum e_A^2 >= 2 x their share of points; (d) partial Spearman of
      (d_k, |e_A|) given log(1 + steps to termination of the pi(s) rollout) >= 0.20 (ranks residualized on the
      control's ranks, then Pearson). The definition gap (noisy-target mean - Q^pi) is reported, never counted.
  R3a pool Pi: predicted gain Q1(s, a_cg) - Q1(s, pi(s)), true gain Q^pi(s, a_cg) - Q^pi(s, pi(s)); misled when
      predicted > 0 and true < -3 sigma_J, sigma_J = the median jitter SD of one rollout over the roughness pairs.
      Pi states are on-policy samples, so uniform weights are visitation weights. Pass: misled share >= 0.20;
      thin misled rate >= 2 x dense (thin/dense at (s, pi(s))); over thin states mean true gain < 0 while mean
      predicted gain > 0.
  R3b units (critic, seed), P0(m), m in {0.25, 0.5, 1, 2, 4}: (i) mean over units of (Qbar(0.25) - Qbar(1)) /
      |Qbar(1)| >= 0.05, Qbar = mean Q1(s, pi_m(s)) over the withheld S states; (ii) J(best m) - J(0.25) >= 5
      normalized points with the 95% CI above 0 (best m by the mean over units); (iii) on 500 on-policy states per
      unit of the P0(0.25) arm, labelled with that arm's own policy, sum over thin states of (Q1 - Q^pi)+ >= 0.5 x
      the sum over all.
  R4a raw first-order true gain of OR-L-abs (beta matched to rho* = 2 rho_host) minus P0(2)'s >= 0.25 |P0's| with
      OR-L-abs's gain > 0, and OR-L-abs's gain above the 95th percentile of 16 SHUF arms (each matched alone). The
      Adam-preconditioned gains are reported beside them.
  R4b J(OR-L-abs*) - J(P0(2)) >= 3.0 with 95% CI above 0 (* = matched to S(P0(2))); max over OR-L-abs's ladder
      and match of mean J minus max over NONE (m = 1) and the P0 ladder of mean J >= 2.0; J(OR-L-abs*) -
      mean_k J(SHUF_k*) >= 2.0.
      The design's cross-task rule (pass on one task, the other not negative) is `combine`.
  R5-world pool L. (a) mean within-row variance of 32 TD-target draws / variance over rows of (t_0 - Q1) <= 0.25;
      (b) Spearman(|b|, |e|) at the logged actions >= 0.30, bootstrap lower bound >= 0.20; (c) mean jitter variance
      of L's roughness pairs / Var(e) over L's (pi, data) points <= 0.05. If the mean jitter variance over all pairs
      exceeds 5% of Var(e_A) over R2's points, the record says the smoothed truth is needed.
  R6  (a) host mean normalized return on Pi's episodes - the dataset's mean episode normalized return >= 10;
      (b) 5-fold logistic regression (standardized obs and their squares, L2) separating Pi states from as many
      withheld states, pooled out-of-fold AUC >= 0.75; (c) median |pi(s) - a| / median |a - a_nn| over withheld rows
      >= 1.5; (d) share of pool-Pi points with c >= 3 >= 0.50.
  R7  inside R4b: (a) J(OR-L-signed*) - J(ANTI-signed*) >= 3 with z >= 3; (b) at most 1 of the 6 SHUF pairs with
      |z| > 2; (c) minimum detectable effect (z_0.975 + z_0.8) SD / sqrt(units) of R4b's matched difference <= 2.0;
      (d) every matched arm within 2% of S* and Level-1 cos(Delta g_OR, Delta g_P0) < 0.98 on every critic;
      (e) matching ran with simulator construction forbidden and its lock verified before evaluation.
A per-critic requirement passes when it passes on every planned critic.

Synthetic data: every output on a synthetic file (synthetic.py, pools.write_synthetic_run) says SYNTHETIC and is a
test fixture, never a finding. The synthetic_smoke profile refuses real data.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.truth import labeller as LB  # noqa: E402
from experiments.truth import pools as PL  # noqa: E402
from experiments.truth import state_restore as SR  # noqa: E402
from experiments.truth.state_restore import DEFAULT_DATA, require, task_spec  # noqa: E402

PREREG_SCHEMA = "bca-truth-prereg-v1"
STAGE_SCHEMA = "bca-truth-stage-v1"
SYNTHETIC_NOTE = ("SYNTHETIC: computed on a synthetic D4RL-format file and/or a randomly initialised host; a test "
                  "fixture, NOT A FINDING.")
CODE_FILES = ("experiments/truth/state_restore.py", "experiments/truth/labeller.py", "experiments/truth/pools.py",
              "experiments/truth/extraction.py", "experiments/truth/acceptance.py", "algorithms/td3_bc.py",
              "algorithms/td3_bc_bca.py", "runtime/td3_bc.py", "experiments/signal/d4rl_level1.py",
              # host restore, scores, Adam/Transition and the simulator source pins also shape every number
              "experiments/wbcp/freeze_scores.py", "runtime/common.py", "runtime/networks.py",
              "experiments/ood/simulator.py")

PASS_MARKS = {  # design_d4rl_rt.json requirements R1-R7
    "R1": dict(pi_thin_share_min=0.10, withheld_thin_share_min=0.03, pi_dense_share_min=0.40),
    "R2": dict(spearman_min=0.30, spearman_lower95_min=0.15, thin_dense_median_ratio_min=3.0,
               thin_sq_share_ratio_min=2.0, partial_spearman_min=0.20),
    "R3a": dict(misled_share_min=0.20, thin_dense_misled_ratio_min=2.0, misled_jitter_sds=3.0),
    "R3b": dict(q_rise_min=0.05, return_drop_min=5.0, thin_overestimate_share_min=0.50),
    "R4a": dict(gain_margin_min=0.25, shuf_percentile=95.0),
    "R4b": dict(matched_margin_min=3.0, best_margin_min=2.0, shuf_margin_min=2.0),
    "R5_world": dict(noise_share_max=0.25, spearman_min=0.30, spearman_lower95_min=0.20, truth_noise_share_max=0.05,
                     smoothed_truth_share=0.05),
    "R5_signal": dict(spearman_min=0.30, partial_spearman_min=0.20),
    "R6": dict(return_gain_min=10.0, auc_min=0.75, action_ratio_min=1.5, supported_share_min=0.50,
               supported_c_min=3),
    "R7": dict(known_answer_min=3.0, known_answer_z_min=3.0, null_pairs_max=1, null_z=2.0, mde_max=2.0,
               match_tolerance=0.02, cosine_max=0.98),
    "thinness": dict(thin_max=2, dense_min=10),
}
PLANS = {
    "full": ["R1", "R2", "R3a", "R3b", "R4a", "R4b", "R5_world", "R6", "R7"],
    "graft": ["R1", "R2", "R5_world", "R6"],
}
PER_CRITIC = ("R1", "R2", "R3a", "R4a", "R5_world", "R6")
EXTRACTION = ("R3b", "R4b", "R7")

DESIGN = dict(  # design_d4rl_rt.json sizes; per critic unless stated
    layout="D-nat", host_step=1_000_000, gamma=0.99, horizon=1000, slots=256, shard_size=4096,
    pool_L=dict(training_rows=2000), pool_Pi=dict(episodes=100, every=25, states=3000, reset_seed_base=702609171),
    pool_T=dict(rows=50000), actions=dict(delta=0.1, critic_step=0.1),
    roughness=dict(states_L=150, states_Pi=150, repeats=8, jitter_scale=1e-4),
    gap=dict(repeats=16, noise_sd=0.2, noise_clip=0.5), thinness=dict(k=10), bootstrap=dict(replicates=1000),
    eps_noise=dict(draws=32), classifier=dict(folds=5, l2=1e-2), r6=dict(action_rows=5000),
    r4a=dict(level=2.0, shuf=16, s=0.83, clip=2.5),
    extraction=dict(actor_steps=20000, block=1000, batch=256, seeds=[1, 2, 3, 4], level=2.0,
                    p0_ladder=[0.25, 0.5, 0.77, 1.3, 2.0, 4.0], ladder_factors=[0.25, 0.5, 1.0, 2.0, 4.0],
                    fallback_ladder=[0.1, 0.3, 1.0, 3.0, 10.0], shuf=4, s_states=5000, eval_episodes=100,
                    eval_reset_seed_base=802609171, eval_slots=128),
    r3b=dict(m=[0.25, 0.5, 1.0, 2.0, 4.0], onpolicy_states=500, onpolicy_every=10),
    r7=dict(power=0.8, alpha=0.05), seed=20261011, designed_critics=3)

SMOKE = json.loads(json.dumps(DESIGN))  # tiny sizes for the synthetic end-to-end test only
SMOKE.update(host_step=0, slots=32, shard_size=256, pool_L=dict(training_rows=24),
             pool_Pi=dict(episodes=6, every=5, states=24, reset_seed_base=702609171), pool_T=dict(rows=96),
             roughness=dict(states_L=4, states_Pi=4, repeats=3, jitter_scale=1e-4),
             gap=dict(repeats=3, noise_sd=0.2, noise_clip=0.5), bootstrap=dict(replicates=50),
             eps_noise=dict(draws=4), classifier=dict(folds=3, l2=1e-2), r6=dict(action_rows=64),
             r4a=dict(level=2.0, shuf=6, s=0.83, clip=2.5),
             extraction=dict(actor_steps=20, block=10, batch=32, seeds=[1, 2], level=2.0,
                             p0_ladder=[0.25, 0.5, 2.0, 4.0], ladder_factors=[0.5, 1.0, 2.0],
                             fallback_ladder=[0.3, 1.0, 3.0], shuf=4, s_states=64, eval_episodes=4,
                             eval_reset_seed_base=802609171, eval_slots=32),
             r3b=dict(m=[0.25, 0.5, 1.0, 2.0, 4.0], onpolicy_states=12, onpolicy_every=3), designed_critics=3)
PROFILES = {"design": DESIGN, "synthetic_smoke": SMOKE}


# ---------------------------------------------------------------- small utilities

def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value):
    return sha256_hex(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def jsonable(value):
    return PL._json_ready(value)


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"{path} exists; stage outputs are written once (delete it to recompute)")
    LB._write_atomic(path, LB.strict_json(jsonable(value)).encode())
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def code_hashes():
    return {name: SR.file_sha256(ROOT / name) for name in CODE_FILES}


def critic_dir(world, seed):
    return Path(world) / f"critic_{int(seed)}"


def finite_or_none(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


# ---------------------------------------------------------------- statistics (NumPy / SciPy)

def spearman(x, y):
    from scipy import stats

    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(stats.spearmanr(x, y)[0])


def partial_spearman(x, y, z):
    """Spearman of x and y given z: ranks of x and y residualized (with intercept) on the ranks of z, then Pearson."""
    from scipy import stats

    x, y, z = (np.asarray(v, np.float64) for v in (x, y, z))
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    basis = np.stack([np.ones_like(rz), rz], axis=1)
    res = [r - basis @ np.linalg.lstsq(basis, r, rcond=None)[0] for r in (rx, ry)]
    if np.std(res[0]) == 0 or np.std(res[1]) == 0:
        return None
    return float(np.corrcoef(res[0], res[1])[0, 1])


def block_bootstrap(blocks, statistic, replicates, seed):
    """Percentile values of statistic(indices) over block-resampled index sets (blocks drawn with replacement)."""
    _, inverse = np.unique(np.asarray(blocks), return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1, len(order)]
    members = [order[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    g = np.random.default_rng(seed)
    values = []
    for _ in range(int(replicates)):
        pick = g.integers(0, len(members), len(members))
        v = statistic(np.concatenate([members[i] for i in pick]))
        if v is not None:
            values.append(v)
    return np.asarray(values, np.float64)


def lower95(values):
    return float(np.percentile(values, 2.5)) if len(values) else None


def auc(pos, neg):
    from scipy import stats

    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if not len(pos) or not len(neg):
        return None
    ranks = stats.rankdata(np.r_[pos, neg])
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logistic_fit(Z, y, l2):
    """L2-penalized logistic regression (intercept unpenalized) by L-BFGS; returns weights [b, w]."""
    from scipy.optimize import minimize

    X = np.c_[np.ones(len(Z)), Z]
    sgn = 2.0 * y - 1.0

    def f(w):
        m = sgn * (X @ w)
        loss = np.mean(np.logaddexp(0.0, -m)) + 0.5 * l2 * np.sum(w[1:] ** 2)
        p = 1.0 / (1.0 + np.exp(np.clip(m, -500, 500)))
        grad = -(X * (sgn * p)[:, None]).mean(axis=0)
        grad[1:] += l2 * w[1:]
        return loss, grad

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def classifier_auc(features_pos, features_neg, folds, l2, seed):
    """Pooled out-of-fold AUC of a logistic regression on [x, x^2] (per-fold z-scoring), class-balanced folds."""
    X = np.r_[features_pos, features_neg].astype(np.float64)
    X = np.c_[X, X ** 2]
    y = np.r_[np.ones(len(features_pos)), np.zeros(len(features_neg))]
    g = np.random.default_rng([seed, 11])
    fold = np.empty(len(y), np.int64)
    for cls in (0, 1):
        idx = g.permutation(np.flatnonzero(y == cls))
        fold[idx] = np.arange(len(idx)) % folds
    score = np.zeros(len(y))
    for f in range(folds):
        train, test = fold != f, fold == f
        mu, sd = X[train].mean(axis=0), X[train].std(axis=0)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        w = logistic_fit(Z[train], y[train], l2)
        score[test] = w[0] + Z[test] @ w[1:]
    return auc(score[y == 1], score[y == 0])


def hierarchical_ci(values_by_critic, replicates, seed):
    """95% percentile interval of the mean by resampling critics, then seeds within each resampled critic."""
    groups = [np.asarray(v, np.float64) for v in values_by_critic if len(v)]
    if not groups:
        return None, None
    g = np.random.default_rng(seed)
    out = np.empty(int(replicates))
    for b in range(len(out)):
        picks = g.integers(0, len(groups), len(groups))
        out[b] = np.mean(np.concatenate([groups[i][g.integers(0, len(groups[i]), len(groups[i]))] for i in picks]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_z(d):
    d = np.asarray(d, np.float64)
    if len(d) < 2:
        return None
    se = d.std(ddof=1) / math.sqrt(len(d))
    if se == 0:
        return math.copysign(math.inf, d.mean()) if d.mean() != 0 else 0.0
    return float(d.mean() / se)


# ---------------------------------------------------------------- pre-registration

def stage_header(body, prereg_sha, stage, critic=None, code_changed=False):
    head = dict(schema=STAGE_SCHEMA, world="D4RL-RT", stage=stage, task=body["task"], dataset=body["dataset"],
                critic=critic, profile=body["profile"], plan=body["plan"], prereg_sha256=prereg_sha,
                synthetic=bool(body["synthetic"]), code_changed_since_prereg=bool(code_changed), utc=utc())
    head["note"] = SYNTHETIC_NOTE if body["synthetic"] else "pre-registered stage of the D4RL-RT acceptance gate"
    return head


def gate_passes(record, tol):
    """state_restore.one_step_gate's pass rule applied to a stored gate record under tolerances tol."""
    return bool(record["restore_miss"]["max"] is not None and record["restore_miss"]["max"] <= tol["tol_restore"]
                and record["next_obs_miss"]["p99"] is not None
                and record["next_obs_miss"]["p99"] <= tol["tol_next_obs"]
                and record["reward_miss"]["p99"] is not None and record["reward_miss"]["p99"] <= tol["tol_reward"]
                and record["terminal_agreement"] >= tol["min_terminal_agreement"])


def freeze(world, task, *, hosts, stage0_dir, profile="design", plan="full", tolerances=None, data_dir=None,
           pass_marks=None, state_source="auto"):
    """Write W/prereg.json and its sha256 sidecar; refuses when labelled or trained outputs already exist, when the
    Stage 0 pilot gate fails, or when its misaligned negative control passes, under the tolerances being frozen.
    state_source 'rebuilt' restores every pool state from the observation (x = 0) even when the file has infos/*,
    for the case where Stage 0 shows the infos states fail the gate but the rebuilt ones pass."""
    world = Path(world)
    task = task_spec(task)
    require(profile in PROFILES and plan in PLANS, f"profile in {sorted(PROFILES)}, plan in {sorted(PLANS)}")
    existing = [p.name for p in world.glob("critic_*")] + [p.name for p in world.glob("extraction")]
    require(not existing, f"{world} already holds {existing}; the pre-registration must precede every label")
    require(not (world / "prereg.json").exists(), f"{world / 'prereg.json'} exists; freeze once per world")
    summary_path = Path(stage0_dir) / "summary.json"
    require(summary_path.is_file(), f"{summary_path} is missing; run stage0 first")
    summary = read_json(summary_path)
    require(summary["dataset"] == task.dataset, "Stage 0 ran on another task")
    synthetic = bool(summary["synthetic"])
    require(profile != "synthetic_smoke" or synthetic, "the synthetic_smoke profile refuses real data")
    protocol = json.loads(json.dumps(PROFILES[profile]))
    protocol["state_source"] = state_source
    tol = dict(SR.DEFAULT_TOLERANCES, **(tolerances or {}))
    require(state_source in ("auto", "rebuilt"), "state_source is 'auto' (infos/* when present) or 'rebuilt'")
    if state_source == "rebuilt" and summary.get("gate_rebuilt") is not None:
        pilot = summary["gate_rebuilt"]  # the file has infos/*, but the states are rebuilt from the observation
    else:
        pilot = summary["gate"]  # 'auto', or a file without infos/* (its gate already restored rebuilt states)
    require(gate_passes(pilot, tol),
            f"the Stage 0 pilot gate ({pilot['source']} states) does not pass under tolerances {tol}; labels would "
            "not be trusted")
    # The misaligned-by-one negative control must FAIL under the tolerances being frozen: tolerances loose enough
    # to pass a restore of the wrong row cannot certify that restores are right.
    require(not gate_passes(summary["gate_negative"], tol),
            f"the Stage 0 negative control (states misaligned by one row) passes under tolerances {tol}; they are "
            "too loose to certify the restore")
    require(hosts, "at least one host (--host SEED=RUN_DIR)")
    require(plan != "graft" or len(hosts) == 1, "the graft labels one existing checkpoint")
    host_records = {}
    for seed, run_dir in sorted(hosts.items()):
        run_dir = Path(run_dir).expanduser().resolve()
        from experiments.wbcp import freeze_scores as FS

        step, ckpt, recorded = FS.locate_checkpoint(run_dir, protocol["host_step"])
        digest = SR.file_sha256(ckpt)
        require(recorded is None or recorded == digest, f"{ckpt} differs from its result.json record")
        host_records[str(int(seed))] = dict(run_dir=str(run_dir), checkpoint=ckpt.name, step=step,
                                            checkpoint_sha256=digest)
    for m in protocol["r3b"]["m"]:
        require(m == 1.0 or m in protocol["extraction"]["p0_ladder"], "R3b's m values must be on the P0 ladder")
    require(protocol["extraction"]["level"] in protocol["extraction"]["p0_ladder"], "the matching level must be on "
            "the P0 ladder")
    marks = json.loads(json.dumps(PASS_MARKS if pass_marks is None else pass_marks))
    body = dict(
        schema=PREREG_SCHEMA, world="D4RL-RT", task=task.name, dataset=task.dataset, profile=profile, plan=PLANS[plan],
        plan_name=plan, synthetic=synthetic, hosts=host_records, data=summary["file"],
        data_dir=str(Path(data_dir).expanduser().resolve()) if data_dir else str(Path(summary["file"]["path"]).parent),
        stage0=dict(summary_sha256=SR.file_sha256(summary_path), gate_pilot=dict(
            source=pilot["source"], restore_miss=pilot["restore_miss"], next_obs_miss=pilot["next_obs_miss"],
            reward_miss=pilot["reward_miss"], terminal_agreement=pilot["terminal_agreement"],
            rows_checked=pilot["rows_checked"]),
            negative_control_failed=not summary["gate_negative"]["passed"],
            negative_control_failed_under_frozen_tolerances=True),
        gate_tolerances=tol, protocol=protocol, pass_marks=marks,
        pass_marks_equal_design=canonical_sha256(marks) == canonical_sha256(PASS_MARKS),
        code_sha256=code_hashes(), frozen_utc=utc(),
        note=(SYNTHETIC_NOTE if synthetic else "Frozen before any truth label; every later number of the planned "
              "stages is pre-registered, anything else is POST HOC."))
    document = dict(body=body, body_sha256=canonical_sha256(body))
    text = LB.strict_json(document).encode()
    world.mkdir(parents=True, exist_ok=True)
    (world / "prereg.json").write_bytes(text)
    digest = sha256_hex(text)
    (world / "prereg.json.sha256").write_text(f"{digest}  prereg.json\n", encoding="utf8")
    return digest


class PreregError(ValueError):
    pass


def load_prereg(world, expected_sha256, allow_code_change=False):
    """(body, sha256, code_changed) after the hash gate; raises PreregError when anything does not match."""
    path = Path(world) / "prereg.json"
    if not path.is_file():
        raise PreregError(f"{path} is missing: freeze the pre-registration first")
    data = path.read_bytes()
    digest = sha256_hex(data)
    if not expected_sha256 or digest != expected_sha256:
        raise PreregError(f"pre-registration hash mismatch: {path} has sha256 {digest}, the stage was given "
                          f"{expected_sha256!r}; refusing to run")
    sidecar = path.with_name("prereg.json.sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf8").split()[0] != digest:
        raise PreregError(f"{sidecar} is missing or does not match {path}; refusing to run")
    document = json.loads(data)
    body = document["body"]
    if canonical_sha256(body) != document["body_sha256"]:
        raise PreregError("the pre-registration body does not match its embedded hash; refusing to run")
    changed = sorted(k for k, v in code_hashes().items() if body["code_sha256"].get(k) != v)
    if changed and not allow_code_change:
        raise PreregError(f"analysis source changed since the freeze: {changed}; refusing to run "
                          "(--allow-code-change runs anyway and marks every output)")
    return body, digest, bool(changed)


def verify_stage(world, body, prereg_sha, code_changed=False):
    """The hash gate inside every stage function, not only on the command line: W/prereg.json must hash to
    prereg_sha, its sidecar and embedded body hash must agree, and `body` must be exactly the frozen body. A
    changed analysis source is refused unless the caller already acknowledged it (code_changed=True, which
    load_prereg returns only under --allow-code-change and which marks every output). Raises PreregError."""
    expected = load_prereg(world, prereg_sha, allow_code_change=True)
    frozen, _, changed = expected
    if canonical_sha256(body) != canonical_sha256(frozen):
        raise PreregError("the stage was given a body that differs from the frozen pre-registration; refusing to run")
    if changed and not code_changed:
        raise PreregError("analysis source changed since the freeze and the caller did not acknowledge it "
                          "(--allow-code-change); refusing to run")
    return True


def host_for(body, seed, data_dir=None, host_dirs=None):
    record = body["hosts"][str(int(seed))]
    run_dir = (host_dirs or {}).get(int(seed), record["run_dir"])
    data_dir = data_dir or body["data_dir"]
    host = PL.restore_host(run_dir, data_dir=data_dir, step=record["step"], expected_sha256=record["checkpoint_sha256"])
    require(host.identity["data"]["sha256"] == body["data"]["sha256"], "the host's data file differs from Stage 0's")
    # The labels' discount and the definition-gap noise must be the critic's own, or e = Q1 - Q^pi mixes definitions.
    pr, args = body["protocol"], host.args
    require(float(args.discount) == float(pr["gamma"]),
            f"host discount {args.discount} differs from the pre-registered label gamma {pr['gamma']}")
    require((float(args.policy_noise), float(args.noise_clip)) == (float(pr["gap"]["noise_sd"]),
                                                                   float(pr["gap"]["noise_clip"])),
            "host target-smoothing noise differs from the pre-registered noisy-target mode")
    return host


# ---------------------------------------------------------------- Stage 0 (before the pre-registration)

def stage0(world, task, *, data_dir=DEFAULT_DATA, file=None, run_dir=None, rows=1000, seed=0, eps_draws=32,
           tolerances=None, host_seed=202609171):
    """Stage 0: no training. Writes W/stage0/*.json and summary.json."""
    task = task_spec(task)
    out = Path(world) / "stage0"
    if file is not None:
        path = Path(file).expanduser().resolve()
        identity = dict(path=str(path), sha256=SR.file_sha256(path), pinned=False, size_bytes=path.stat().st_size)
    else:
        path, identity = SR.cached_file(task, data_dir)
    synthetic = SR._synthetic_mark(path) is not None
    head = dict(schema=STAGE_SCHEMA, world="D4RL-RT", stage="stage0", task=task.name, dataset=task.dataset,
                synthetic=synthetic, file=identity, utc=utc(),
                note=SYNTHETIC_NOTE if synthetic else "Stage 0, descriptive, before the pre-registration")
    keys = SR.describe_file(task, path)
    physics = SR.Physics(task.dataset)
    try:
        gates = {}
        for name, kwargs in (("gate", {}), ("gate_negative", dict(offset=1))):
            g = SR.run_gate(task, path, rows=rows, seed=seed, tolerances=tolerances, physics=physics, **kwargs)
            arrays = g.pop("_arrays")
            gates[name] = g
            out.mkdir(parents=True, exist_ok=True)
            with (out / f"{name}.npz").open("xb") as handle:
                np.savez(handle, **arrays)
        if keys["has_infos_qpos"] and keys["has_infos_qvel"]:
            g = SR.run_gate(task, path, rows=rows, seed=seed, tolerances=tolerances, physics=physics, source="rebuilt")
            g.pop("_arrays")
            gates["gate_rebuilt"] = g
    finally:
        physics.close()
    raw = PL.read_raw(path)
    if keys["has_infos_qpos"] and keys["has_infos_qvel"]:
        _, rows_conv, _ = SR.converted_rows({k: raw[k] for k in PL.RAW_KEYS})
        inf = SR.logged_states(task, raw, rows_conv, "infos")
        reb = SR.logged_states(task, raw, rows_conv, "rebuilt")
        k = reb.kept
        dq = np.abs(inf.qpos[k, 1:] - reb.qpos[k, 1:])
        dv = np.abs(inf.qvel[k] - reb.qvel[k])
        rebuild = dict(rows=int(len(k)), kept=int(k.sum()), dropped_at_velocity_clip=int((~k).sum()),
                       max_abs_qpos_except_x=float(dq.max()) if k.any() else None,
                       max_abs_qvel=float(dv.max()) if k.any() else None,
                       rows_identical_except_x=int(((dq == 0).all(axis=1) & (dv == 0).all(axis=1)).sum()))
    else:
        rebuild = dict(rows=None, note="the file has no infos/qpos and infos/qvel; states are rebuilt from the "
                       "observation (x = 0), rows at the velocity clip dropped", rows_at_velocity_clip=
                       keys["rows_at_velocity_clip"])
    split = dict(available=False)
    noise = dict(available=False, reason="needs the host checkpoint (--run-dir)")
    dataset_scores = None
    dthin = PL.d_thin_mask(raw, 0.1, seed)[1]
    if run_dir is not None:
        host = PL.restore_host(run_dir, data_dir=path.parent)
        require(host.identity["data"]["sha256"] == identity["sha256"], "the run's cache differs from the Stage 0 file")
        thin = PL.ThinnessIndex(host)
        xw = PL.features(host, host.converted["observations"][host.withheld_ids],
                         host.converted["actions"][host.withheld_ids])
        radius, _ = thin.fit_radius(xw, host.episodes[host.withheld_ids])
        b = host.bank_ids
        c_bank = thin.counts(PL.features(host, host.converted["observations"][b], host.converted["actions"][b]),
                             host.episodes[b])
        marks = PASS_MARKS["thinness"]
        split = dict(available=True, host=host.identity, radius=radius, bank_rows=int(len(b)),
                     bank_thin_share=float(np.mean(c_bank <= marks["thin_max"])),
                     bank_dense_share=float(np.mean(c_bank >= marks["dense_min"])),
                     bank_c_median=float(np.median(c_bank)))
        probe = PL.CriticProbe(host)
        conv = host.converted
        t = probe.target_draws(conv["rewards"][b], conv["next_observations"][b], conv["terminals"][b], eps_draws, seed)
        q1 = probe.q(conv["observations"][b], conv["actions"][b])[:, 0].astype(np.float64)
        within = t.var(axis=1, ddof=1).mean()
        noise = dict(available=True, rows="bank", draws=int(eps_draws), mean_within_row_variance=float(within),
                     residual_variance=float(np.var(t[:, 0] - q1, ddof=1)),
                     noise_share=float(within / np.var(t[:, 0] - q1, ddof=1)))
        _, scores = PL.dataset_episode_scores(host)
        dataset_scores = dict(episodes=int(len(scores)), mean=float(scores.mean()), median=float(np.median(scores)))
    summary = dict(head, keys=keys, gate=gates["gate"], gate_negative=gates["gate_negative"],
                   gate_rebuilt=gates.get("gate_rebuilt"), rebuild=rebuild, split=split, eps_noise=noise,
                   dataset_episode_scores=dataset_scores, d_thin_0_1=dthin, host_seed=host_seed)
    write_once(out / "summary.json", summary)
    return summary


# ---------------------------------------------------------------- pools and labels

def stage_pools(world, body, prereg_sha, seed, *, data_dir=None, host_dirs=None, code_changed=False):
    verify_stage(world, body, prereg_sha, code_changed)
    out = critic_dir(world, seed) / "pools"
    require(not (out / "summary.json").exists(), f"{out} is complete")
    pr = body["protocol"]
    host = host_for(body, seed, data_dir, host_dirs)
    policy = host.policy(pr["slots"])
    invariance = require_invariant(policy, host)
    probe = PL.CriticProbe(host)
    thin = PL.ThinnessIndex(host, k=pr["thinness"]["k"])
    w = host.withheld_ids
    radius, dk_w = thin.fit_radius(PL.features(host, host.converted["observations"][w], host.converted["actions"][w]),
                                   host.episodes[w])
    state_index = PL.StateIndex(host)
    master = pr["seed"]
    head = stage_header(body, prereg_sha, "pools", seed, code_changed)
    common = dict(head, host=host.identity, radius=radius, k=pr["thinness"]["k"], position_invariance=invariance,
                  withheld_rows_for_radius=int(len(w)), normalization=dict(mean=host.obs_mean, std=host.obs_std))
    arrays, meta = PL.build_pool_L(host, policy, probe, thin, training_rows=pr["pool_L"]["training_rows"], seed=master,
                                   delta=pr["actions"]["delta"], critic_step=pr["actions"]["critic_step"],
                                   draws=pr["eps_noise"]["draws"], target_seed=master,
                                   state_source=pr.get("state_source", "auto"))
    PL.save_pool(out, "L", arrays, dict(common, **meta))
    physics = SR.Physics(host.task.dataset)
    try:
        p = pr["pool_Pi"]
        arrays, meta = PL.build_pool_Pi(host, physics, policy, probe, thin, state_index, episodes=p["episodes"],
                                        every=p["every"], states=p["states"], reset_seed_base=p["reset_seed_base"],
                                        seed=master, horizon=pr["horizon"], delta=pr["actions"]["delta"],
                                        critic_step=pr["actions"]["critic_step"])
        PL.save_pool(out, "Pi", arrays, dict(common, **meta))
    finally:
        physics.close()
    if any(r in body["plan"] for r in ("R4a", "R4b", "R3b")):
        arrays, meta = PL.build_pool_T(host, policy, probe, rows=pr["pool_T"]["rows"], seed=master,
                                       state_source=pr.get("state_source", "auto"))
        PL.save_pool(out, "T", arrays, dict(common, **meta))
    r6 = PL.r6_inputs(host, policy, state_index, rows=pr["r6"]["action_rows"], seed=master)
    _, scores = PL.dataset_episode_scores(host)
    r6.update(obs_mean=host.obs_mean, obs_std=host.obs_std)
    PL.save_pool(out, "R6", r6, dict(common, kind="R6", dataset_episode_score_mean=float(scores.mean()),
                                     dataset_episodes=int(len(scores))))
    sd_q, sd_v = SR.state_sd(SR.logged_states(host.task, host.raw, host.rows, pr.get("state_source", "auto")))
    np.savez(out / "jitter_sd.npz", qpos=sd_q, qvel=sd_v)
    return write_once(out / "summary.json", dict(common, pools=[p.stem for p in sorted(out.glob("*.json"))
                                                                if p.stem != "summary"]))


def require_invariant(policy, host):
    """The fixed-shape actor must give a row the same action in any slot (labeller.FrozenPolicy.position_invariance)
    on logged observations, or lockstep rollouts would depend on the slot schedule."""
    probe = host.converted["observations"][host.training_ids[:policy.batch]].astype(np.float64)
    record = policy.position_invariance(probe)
    require(record["passed"], "the jitted actor is not position-invariant at this batch shape: " + json.dumps(record))
    return record


def roughness_states(n, k, seed, tag):
    return PL._seeded_choice(n, k, seed, 100 + tag)


def label_jobs(body):
    """{job: (pool, mode, repeats, state subset rule, slots)} for the plan."""
    pr = body["protocol"]
    jobs = {"L": ("L", "deterministic", 1, None, None), "Pi": ("Pi", "deterministic", 1, None, None)}
    if any(r in body["plan"] for r in ("R4a", "R4b", "R3b")):
        jobs["T"] = ("T", "deterministic", 1, None, None)
    rough, gap = pr["roughness"], pr["gap"]
    for pool, tag in (("L", 1), ("Pi", 2)):
        jobs[f"rough_{pool}"] = (pool, "jitter", rough["repeats"], (rough[f"states_{pool}"], tag), [0, 1])
        jobs[f"gap_{pool}"] = (pool, "noisy_target", gap["repeats"], (rough[f"states_{pool}"], tag), [0, 1])
    return jobs


def stage_labels(world, body, prereg_sha, seed, *, jobs=None, worker=0, workers=1, data_dir=None, host_dirs=None,
                 log=print, code_changed=False):
    verify_stage(world, body, prereg_sha, code_changed)
    pr = body["protocol"]
    host = host_for(body, seed, data_dir, host_dirs)
    policy = host.policy(pr["slots"])
    cdir = critic_dir(world, seed)
    sd = np.load(cdir / "pools" / "jitter_sd.npz")
    jitter_sd = (sd["qpos"], sd["qvel"])
    plan_jobs = label_jobs(body)
    jobs = list(plan_jobs) if jobs is None else list(jobs)
    physics = SR.Physics(host.task.dataset)
    records = {}
    try:
        for job in jobs:
            pool, mode, repeats, subset, slots = plan_jobs[job]
            arrays, _ = PL.load_pool(cdir / "pools", pool)
            states = None if subset is None else roughness_states(len(arrays["qpos"]), subset[0], pr["seed"], subset[1])
            items = PL.pool_items(arrays, slots=slots, states=states)
            spec = LB.LabelSpec(host.task.dataset, mode=mode, repeats=int(repeats), gamma=float(pr["gamma"]),
                                horizon=int(pr["horizon"]), seed=int(pr["seed"]),
                                jitter_scale=float(pr["roughness"]["jitter_scale"]),
                                noise_sd=float(pr["gap"]["noise_sd"]), noise_clip=float(pr["gap"]["noise_clip"]),
                                shard_size=int(pr["shard_size"]))
            out = cdir / "labels" / job
            # Rotate the shard-to-worker map per job (a bijection on workers, so every shard still has exactly one
            # worker): otherwise worker 0 takes shard 0 of every job, including every one-shard roughness and gap
            # job. Labels do not depend on which worker or slot computes them (test_labeller).
            turn = (worker + list(plan_jobs).index(job)) % workers
            recs = LB.run_labels(out, physics, policy, spec, items, jitter_sd=jitter_sd if mode == "jitter" else None,
                                 worker=turn, workers=workers, log=log)
            records[job] = dict(shards=len(recs), env_steps=int(sum(r["env_steps"] for r in recs)))
    finally:
        physics.close()
    marker = dict(stage_header(body, prereg_sha, "labels", seed, code_changed), jobs=records, worker=worker,
                  workers=workers)
    LB._write_atomic(cdir / "labels" / f"progress_worker{worker}.json", LB.strict_json(jsonable(marker)).encode())
    return records


def labels_complete(world, body, seed):
    cdir = critic_dir(world, seed) / "labels"
    for job in label_jobs(body):
        try:
            LB.load_labels(cdir / job)
        except (ValueError, FileNotFoundError, KeyError):
            return False
    return True


def load_job(world, seed, job, n_states, n_slots, pool=None):
    """Labels of a job as (q, length, status) arrays of shape (n_states, n_slots, K), NaN where not labelled.

    With the job's pool arrays, the labelled items are rebuilt from the pool (state = item_id // n_slots, slot =
    item_id % n_slots) and their qpos, qvel and action must hash to the label manifest's inputs, so labels made for
    another pool file (e.g. pools rebuilt after labelling) are refused instead of being read against new states."""
    arrays, manifest = LB.load_labels(critic_dir(world, seed) / "labels" / job)
    K = manifest["spec"]["repeats"]
    if pool is not None:
        require(pool["actions"].shape[:2] == (n_states, n_slots), f"pool shape differs from job {job}")
        st, sl = np.divmod(arrays["item_id"], n_slots)
        rebuilt = dict(qpos=pool["qpos"][st], qvel=pool["qvel"][st], action=pool["actions"][st, sl].astype(np.float32))
        require(all(SR.array_sha256(rebuilt[k]) == manifest["inputs"][k] for k in rebuilt),
                f"labels of job {job} were made for other pool states or actions; refusing to score them")
    q = np.full((n_states, n_slots, K), np.nan)
    length = np.zeros((n_states, n_slots, K), np.int64)
    status = np.full((n_states, n_slots, K), -1, np.int64)
    st, sl = np.divmod(arrays["item_id"], n_slots)
    q[st, sl] = arrays["q"]
    length[st, sl] = arrays["length"]
    status[st, sl] = arrays["status"]
    return q, length, status


# ---------------------------------------------------------------- per-critic scores (NumPy only)

def _pool(world, seed, name):
    return PL.load_pool(critic_dir(world, seed) / "pools", name)


def truth_tables(world, seed):
    """Pools L and Pi with their deterministic truth and the roughness / gap pairs."""
    out = {}
    for name in ("L", "Pi"):
        arrays, meta = _pool(world, seed, name)
        n, S = arrays["actions"].shape[:2]
        q, length, status = load_job(world, seed, name, n, S, arrays)
        rough = load_job(world, seed, f"rough_{name}", n, S, arrays)[0]
        gap = load_job(world, seed, f"gap_{name}", n, S, arrays)[0]
        out[name] = dict(arrays=arrays, meta=meta, qpi=q[..., 0], length=length[..., 0], status=status[..., 0],
                         rough=rough, gap=gap, q1=arrays["q"][..., 0].astype(np.float64))
    return out


def _pairs(table, key):
    """Roughness/gap pairs (state, slot) present in a job, with their K values."""
    v = table[key]
    present = np.isfinite(v).all(axis=2)
    st, sl = np.nonzero(present)
    return st, sl, v[st, sl]


def score_r1(tables, marks, thin):
    L, Pi = tables["L"], tables["Pi"]
    c_pi = Pi["arrays"]["c"][:, 0]
    c_bank = L["arrays"]["c"][L["arrays"]["in_bank"], 1]
    values = dict(pi_thin_share=float(np.mean(c_pi <= thin["thin_max"])),
                  withheld_thin_share=float(np.mean(c_bank <= thin["thin_max"])) if len(c_bank) else None,
                  pi_dense_share=float(np.mean(c_pi >= thin["dense_min"])), pi_points=int(len(c_pi)),
                  bank_rows=int(len(c_bank)), pi_c_median=float(np.median(c_pi)))
    passes = dict(a=values["pi_thin_share"] >= marks["pi_thin_share_min"],
                  b=values["withheld_thin_share"] is not None
                  and values["withheld_thin_share"] >= marks["withheld_thin_share_min"],
                  c=values["pi_dense_share"] >= marks["pi_dense_share_min"])
    return values, passes


def r2_points(tables):
    """Every non-pi slot of L and Pi: |e_A|, d_k, c, block, log(1 + steps to termination of pi(s)), gap."""
    cols = {k: [] for k in ("abs_ea", "ea", "dk", "c", "block", "term", "gap", "pool")}
    dropped = 0
    for code, name in enumerate(("L", "Pi")):
        t = tables[name]
        e = t["q1"] - t["qpi"]
        ea = e - e[:, [0]]
        n, S = ea.shape
        ok = np.isfinite(ea) & (t["status"] != 2)
        ok[:, 0] = False
        dropped += int((~np.isfinite(ea[:, 1:])).sum())
        st, sl = np.nonzero(ok)
        cols["ea"].append(ea[st, sl])
        cols["abs_ea"].append(np.abs(ea[st, sl]))
        cols["dk"].append(t["arrays"]["dk"][st, sl])
        cols["c"].append(t["arrays"]["c"][st, sl])
        cols["block"].append(t["arrays"]["episode"][st])
        cols["term"].append(np.log1p(t["length"][st, 0].astype(np.float64)))
        g = np.full_like(ea, np.nan)
        have = np.isfinite(t["gap"]).all(axis=2)
        g[have] = t["gap"][have].mean(axis=1) - t["qpi"][have]
        cols["gap"].append(g[st, sl])
        cols["pool"].append(np.full(len(st), code))
    return {k: np.concatenate(v) for k, v in cols.items()}, dropped


def score_r2(tables, marks, thin, replicates, seed):
    pts, dropped = r2_points(tables)
    x, y = pts["dk"], pts["abs_ea"]
    rho = spearman(x, y)
    boot = block_bootstrap(pts["block"], lambda i: spearman(x[i], y[i]), replicates, [seed, 21])
    is_thin, is_dense = pts["c"] <= thin["thin_max"], pts["c"] >= thin["dense_min"]
    med_thin = float(np.median(y[is_thin])) if is_thin.any() else None
    med_dense = float(np.median(y[is_dense])) if is_dense.any() else None
    ratio = med_thin / med_dense if med_thin is not None and med_dense not in (None, 0.0) else None
    sq = pts["ea"] ** 2
    thin_sq_share = float(sq[is_thin].sum() / sq.sum()) if sq.sum() > 0 else None
    thin_point_share = float(is_thin.mean())
    sq_ratio = thin_sq_share / thin_point_share if thin_sq_share is not None and thin_point_share > 0 else None
    partial = partial_spearman(x, y, pts["term"])
    gap = pts["gap"][np.isfinite(pts["gap"])]
    values = dict(points=int(len(y)), points_dropped_nonfinite=dropped, spearman=rho, spearman_lower95=lower95(boot),
                  bootstrap_replicates=int(len(boot)), thin_points=int(is_thin.sum()), dense_points=int(is_dense.sum()),
                  median_abs_ea_thin=med_thin, median_abs_ea_dense=med_dense, thin_dense_median_ratio=ratio,
                  thin_sq_share=thin_sq_share, thin_point_share=thin_point_share, thin_sq_share_ratio=sq_ratio,
                  partial_spearman_given_termination=partial,
                  definition_gap=dict(pairs=int(len(gap)), mean=float(gap.mean()) if len(gap) else None,
                                      mean_abs=float(np.abs(gap).mean()) if len(gap) else None,
                                      note="noisy-target mean minus Q^pi; reported separately, never critic error"),
                  per_pool={name: dict(points=int((pts["pool"] == k).sum()),
                                       spearman=spearman(x[pts["pool"] == k], y[pts["pool"] == k]))
                            for k, name in enumerate(("L", "Pi"))})
    passes = dict(a=rho is not None and rho >= marks["spearman_min"] and values["spearman_lower95"] is not None
                  and values["spearman_lower95"] >= marks["spearman_lower95_min"],
                  b=ratio is not None and ratio >= marks["thin_dense_median_ratio_min"],
                  c=sq_ratio is not None and sq_ratio >= marks["thin_sq_share_ratio_min"],
                  d=partial is not None and partial >= marks["partial_spearman_min"])
    return values, passes


def jitter_sds(tables):
    sds, variances = [], []
    for name in ("L", "Pi"):
        _, _, v = _pairs(tables[name], "rough")
        if v.shape[-1] >= 2:
            sds.append(v.std(axis=1, ddof=1))
            variances.append(v.var(axis=1, ddof=1))
    sds = np.concatenate(sds) if sds else np.zeros(0)
    return sds, (np.concatenate(variances) if variances else np.zeros(0))


def score_r3a(tables, marks, thin):
    Pi = tables["Pi"]
    S = Pi["qpi"].shape[1]
    cg = S - 1
    pred = Pi["q1"][:, cg] - Pi["q1"][:, 0]
    true = Pi["qpi"][:, cg] - Pi["qpi"][:, 0]
    ok = np.isfinite(true)
    sds, _ = jitter_sds(tables)
    sigma = float(np.median(sds)) if len(sds) else None
    require(sigma is not None, "R3a needs the roughness labels")
    misled = ok & (pred > 0) & (true < -marks["misled_jitter_sds"] * sigma)
    c = Pi["arrays"]["c"][:, 0]
    is_thin, is_dense = ok & (c <= thin["thin_max"]), ok & (c >= thin["dense_min"])
    rate = lambda m: float(misled[m].mean()) if m.any() else None
    r_thin, r_dense = rate(is_thin), rate(is_dense)
    values = dict(states=int(ok.sum()), jitter_sd_median=sigma, threshold=-marks["misled_jitter_sds"] * sigma,
                  misled_share=float(misled[ok].mean()) if ok.any() else None, misled_rate_thin=r_thin,
                  misled_rate_dense=r_dense, thin_states=int(is_thin.sum()), dense_states=int(is_dense.sum()),
                  thin_mean_true_gain=float(true[is_thin].mean()) if is_thin.any() else None,
                  thin_mean_predicted_gain=float(pred[is_thin].mean()) if is_thin.any() else None,
                  weights="uniform over pool-Pi states (on-policy samples, so visitation weights)")
    ratio_ok = (r_thin is not None and r_dense is not None
                and (r_thin >= marks["thin_dense_misled_ratio_min"] * r_dense if r_dense > 0 else r_thin > 0))
    passes = dict(share=values["misled_share"] is not None and values["misled_share"] >= marks["misled_share_min"],
                  thin_vs_dense=bool(ratio_ok),
                  thin_direction=values["thin_mean_true_gain"] is not None and values["thin_mean_true_gain"] < 0
                  and values["thin_mean_predicted_gain"] > 0)
    return values, passes


def score_r5_world(tables, r2_values, marks, replicates, seed):
    L = tables["L"]
    a = L["arrays"]
    t = a["targets"].astype(np.float64)
    q1_data = L["q1"][:, 1]
    within = float(t.var(axis=1, ddof=1).mean())
    resid_var = float(np.var(t[:, 0] - q1_data, ddof=1))
    e = L["q1"] - L["qpi"]
    e_data = e[:, 1]
    ok = np.isfinite(e_data)
    b = a["residual_b"]
    rho = spearman(np.abs(b[ok]), np.abs(e_data[ok]))
    blocks = a["episode"][ok]
    bb, ee = np.abs(b[ok]), np.abs(e_data[ok])
    boot = block_bootstrap(blocks, lambda i: spearman(bb[i], ee[i]), replicates, [seed, 51])
    _, _, rough_L = _pairs(L, "rough")
    jitter_var_L = float(rough_L.var(axis=1, ddof=1).mean()) if rough_L.shape[-1] >= 2 and len(rough_L) else None
    e_pts = e[:, :2][np.isfinite(e[:, :2])]
    var_e = float(np.var(e_pts, ddof=1))
    _, all_var = jitter_sds(tables)
    pts, _ = r2_points(tables)
    var_ea = float(np.var(pts["ea"], ddof=1))
    smoothed = bool(len(all_var) and all_var.mean() > marks["smoothed_truth_share"] * var_ea)
    ea = e - e[:, [0]]
    rho_ea = spearman(np.abs(b[ok]), np.abs(ea[ok, 1]))
    values = dict(noise_share=within / resid_var if resid_var > 0 else None, mean_within_row_variance=within,
                  residual_variance=resid_var, target_draws=int(t.shape[1]), spearman_abs_b_abs_e=rho,
                  spearman_lower95=lower95(boot), spearman_abs_b_abs_ea=rho_ea, rows=int(ok.sum()),
                  truth_noise_share=jitter_var_L / var_e if jitter_var_L is not None and var_e > 0 else None,
                  jitter_variance_L=jitter_var_L, var_e_L=var_e,
                  smoothed_truth_required=smoothed,
                  smoothed_rule="mean jitter variance over all roughness pairs > share x Var(e_A) over R2's points")
    passes = dict(a=values["noise_share"] is not None and values["noise_share"] <= marks["noise_share_max"],
                  b=rho is not None and rho >= marks["spearman_min"] and values["spearman_lower95"] is not None
                  and values["spearman_lower95"] >= marks["spearman_lower95_min"],
                  c=values["truth_noise_share"] is not None
                  and values["truth_noise_share"] <= marks["truth_noise_share_max"])
    return values, passes


def score_r6(world, seed, tables, marks, classifier, master):
    Pi = tables["Pi"]
    r6, meta = _pool(world, seed, "R6")
    host_score = float(Pi["arrays"]["episode_scores"].mean())
    gain = host_score - meta["dataset_episode_score_mean"]
    obs_pi = Pi["arrays"]["obs"]
    w_obs = r6["withheld_obs"]
    take = np.random.default_rng([master, 61]).permutation(len(w_obs))[:len(obs_pi)]
    std = lambda o: ((np.asarray(o, np.float32) - r6["obs_mean"]) / r6["obs_std"]).astype(np.float64)
    auc_value = classifier_auc(std(obs_pi), std(w_obs[take]), classifier["folds"], classifier["l2"], master)
    med_ann = float(np.median(r6["a_minus_ann"]))
    ratio = float(np.median(r6["pi_minus_a"])) / med_ann if med_ann > 0 else None
    c_pi = Pi["arrays"]["c"][:, 0]
    supported = float(np.mean(c_pi >= marks["supported_c_min"]))
    values = dict(host_mean_score=host_score, dataset_mean_episode_score=meta["dataset_episode_score_mean"],
                  return_gain=gain, classifier_auc=auc_value, classifier_states=[int(len(obs_pi)), int(len(take))],
                  action_ratio=ratio, median_pi_minus_a=float(np.median(r6["pi_minus_a"])), median_a_minus_ann=med_ann,
                  withheld_rows=int(len(w_obs)), supported_share=supported)
    passes = dict(a=gain >= marks["return_gain_min"], b=auc_value is not None and auc_value >= marks["auc_min"],
                  c=ratio is not None and ratio >= marks["action_ratio_min"],
                  d=supported >= marks["supported_share_min"])
    return values, passes


def stage_score(world, body, prereg_sha, seed, *, code_changed=False):
    verify_stage(world, body, prereg_sha, code_changed)
    pr, marks = body["protocol"], body["pass_marks"]
    thin = marks["thinness"]
    tables = truth_tables(world, seed)
    out = critic_dir(world, seed) / "scores"
    results = {}
    reps, master = pr["bootstrap"]["replicates"], pr["seed"]
    r2v = None
    for req in ("R1", "R2", "R3a", "R5_world", "R6"):
        if req not in body["plan"]:
            continue
        if (out / f"{req}.json").exists():
            results[req] = read_json(out / f"{req}.json")
            r2v = results[req]["values"] if req == "R2" else r2v
            continue
        if req == "R1":
            values, passes = score_r1(tables, marks["R1"], thin)
        elif req == "R2":
            values, passes = score_r2(tables, marks["R2"], thin, reps, master)
            r2v = values
        elif req == "R3a":
            values, passes = score_r3a(tables, marks["R3a"], thin)
        elif req == "R5_world":
            values, passes = score_r5_world(tables, r2v, marks["R5_world"], reps, master)
        else:
            values, passes = score_r6(world, seed, tables, marks["R6"], pr["classifier"], master)
        record = dict(stage_header(body, prereg_sha, "score", seed, code_changed), requirement=req, values=values,
                      passes=passes, passed=bool(all(passes.values())), pass_marks=marks[req])
        write_once(out / f"{req}.json", record)
        results[req] = record
    return results


# ---------------------------------------------------------------- R4a (Level-1 screen, no training)

def oracle_arms(body, poolT, qpiT, n_shuf):
    """Weights of the oracle arms on pool T rows from the truth (data and pi slots)."""
    from experiments.truth import extraction as EX

    pr = body["protocol"]
    q1 = poolT["q"][..., 0].astype(np.float64)
    e = q1 - qpiT
    # A row whose rollout hit a simulator error (status 2, q NaN) has no truth: it leaves the oracle pool, and the
    # screen and the extraction train on the same kept rows (counted in their records).
    ok = np.isfinite(e).all(axis=1)
    require(ok.sum() >= 2, "pool T truth is missing for almost every row")
    e_a = e[ok, 0] - e[ok, 1]
    s, clip = pr["r4a"]["s"], pr["r4a"]["clip"]
    sig_abs, shuf, anti_abs, m_abs = EX.oracle_weights(e_a, "abs", pr["seed"], s=s, clip=clip, n_shuf=n_shuf)
    sig_signed, _, anti_signed, _ = EX.oracle_weights(e_a, "signed", pr["seed"], s=s, clip=clip)
    arms = {"OR_abs": sig_abs, "OR_signed": sig_signed, "ANTI_abs": anti_abs, "ANTI_signed": anti_signed}
    arms.update({f"SHUF_{k}": w for k, w in enumerate(shuf)})
    return arms, e_a, ok


def truth_gradient_Pi(world, seed):
    arrays, _ = _pool(world, seed, "Pi")
    n, S, d = arrays["actions"].shape
    q = load_job(world, seed, "Pi", n, S, arrays)[0][..., 0]
    dd = (S - 3) // 2
    grad = (q[:, 2:2 + dd] - q[:, 2 + dd:2 + 2 * dd]) / arrays["cd_step"]
    ok = np.isfinite(grad).all(axis=1)
    return arrays["obs"][ok], grad[ok], int((~ok).sum())


def stage_screen(world, body, prereg_sha, seed, *, data_dir=None, host_dirs=None, code_changed=False):
    verify_stage(world, body, prereg_sha, code_changed)
    from experiments.truth import extraction as EX

    pr, marks = body["protocol"], body["pass_marks"]["R4a"]
    host = host_for(body, seed, data_dir, host_dirs)
    poolT, _ = _pool(world, seed, "T")
    n, S, _ = poolT["actions"].shape
    qpiT = load_job(world, seed, "T", n, S, poolT)[0][..., 0]
    arms, _, keep_T = oracle_arms(body, poolT, qpiT, pr["r4a"]["shuf"])
    obs_Pi, grad_Pi, dropped = truth_gradient_Pi(world, seed)
    screen = EX.Level1Screen(host, poolT["obs"][keep_T], poolT["actions"][keep_T, 0], obs_Pi, grad_Pi)
    level = pr["r4a"]["level"]
    names = list(arms)
    b = screen.b_of(np.stack([arms[k] for k in names]))
    p0 = screen.gains((level - 1.0) * screen.b0)
    records = {}
    for k, name in enumerate(names):
        beta = screen.match_beta(b[k], level)
        dg = beta * b[k] if beta is not None else None
        records[name] = dict(beta=beta, gains=None if dg is None else screen.gains(dg),
                             cos_with_p0=screen.cosine(b[k], screen.b0))
    shuf = [records[k]["gains"] for k in names if k.startswith("SHUF_") and records[k]["gains"] is not None]
    pct = {kind: float(np.percentile([g[kind] for g in shuf], marks["shuf_percentile"])) if shuf else None
           for kind in ("raw", "adam")}
    orabs = records["OR_abs"]["gains"]

    def margin_pass(kind):
        if orabs is None:
            return False
        return orabs[kind] > 0 and orabs[kind] - p0[kind] >= marks["gain_margin_min"] * abs(p0[kind])

    values = dict(level=level, rho_host=screen.rho_host, lambda_Q=screen.lam, pool_T_rows=int(screen.N),
                  pool_T_rows_without_truth=int((~keep_T).sum()),
                  pool_Pi_states=int(len(obs_Pi)), pool_Pi_dropped_nonfinite=dropped, p0_gains=p0, arms=records,
                  shuf_percentile=pct, preconditioner=screen.preconditioner,
                  adam_reported={"margin": margin_pass("adam"),
                                 "above_shuf": orabs is not None and pct["adam"] is not None
                                 and orabs["adam"] > pct["adam"]},
                  primary="raw first-order gain")
    passes = dict(margin=margin_pass("raw"),
                  above_shuf=orabs is not None and pct["raw"] is not None and orabs["raw"] > pct["raw"])
    record = dict(stage_header(body, prereg_sha, "screen", seed, code_changed), requirement="R4a", values=values,
                  passes=passes, passed=bool(all(passes.values())), pass_marks=marks)
    write_once(critic_dir(world, seed) / "scores" / "R4a.json", record)
    return record


# ---------------------------------------------------------------- R5-signal (BCA band; OFF by default)

class BandFitDisabled(RuntimeError):
    pass


def stage_r5_signal(world, body, prereg_sha, seed, *, allow_bca_band=False, data_dir=None, host_dirs=None,
                    updates=None, code_changed=False):
    """BCA's frozen q1 width W = lambda eta u against the truth (a diagnostic stop rule before any BCA training).
    Fits a BCA scale network: refuses unless allow_bca_band. Never run in this build."""
    if not allow_bca_band:
        raise BandFitDisabled("R5-signal fits a BCA band; it is off unless --allow-bca-band is given (and approved)")
    verify_stage(world, body, prereg_sha, code_changed)
    import jax

    from calibration.wbcp import calibrate
    from experiments.signal.frozen_signals import fit_scales
    from runtime.config import resolve, typed
    from algorithms import td3_bc_bca as P

    pr, marks = body["protocol"], body["pass_marks"]["R5_signal"]
    host = host_for(body, seed, data_dir, host_dirs)
    row = resolve(PL.TD3_CONFIG, "bca", int(seed), Path(world).resolve(), host.task.name)
    _, args, spec, _ = typed(row)
    config = spec.config()
    rng, template, models = P.initialize(args, config, host.task.obs_dim, host.task.action_dim, host.max_action)
    prepared = host.prepared
    updates = int(updates or 100_000)
    scales, log, fingerprints = fit_scales(args, config, models, host.native, template, prepared.training, ("q1",),
                                           updates, 1000, rng, host.max_action)
    calibrator, unit = scales["q1"]
    heldout = prepared.heldout
    target = P.native_target(args, models, host.native, heldout, jax.random.PRNGKey(pr["seed"]), host.max_action)
    q1 = models[1].apply(host.native.critic.params, heldout.obs, heldout.action)[..., 0]
    eta = models[2].apply(calibrator.params, heldout.obs, heldout.action)
    scores = np.abs(np.asarray(target - q1)) / (np.maximum(np.asarray(eta), 1e-6) * float(unit))
    cal = calibrate(scores, np.random.default_rng([pr["seed"], 71]), alpha=config.posterior.alpha,
                    beta=config.posterior.credibility, draws=config.posterior.draws)
    tables = truth_tables(world, seed)
    L = tables["L"]
    obs_n = host.standardized(L["arrays"]["obs"])
    def width(a):
        eta = np.asarray(models[2].apply(calibrator.params, obs_n, a))
        return cal.threshold * np.maximum(eta, 1e-6) * float(unit)

    w_data = width(L["arrays"]["actions"][:, 1])
    e = L["q1"] - L["qpi"]
    ea = e - e[:, [0]]
    eps_sd = L["arrays"]["targets"].std(axis=1, ddof=1)
    values = dict(spearman_w_abs_e=spearman(w_data, np.abs(e[:, 1])),
                  spearman_w_abs_ea=spearman(w_data, np.abs(ea[:, 1])),
                  partial_spearman_given_eps_sd=partial_spearman(w_data, np.abs(e[:, 1]), eps_sd),
                  knn_baseline_spearman=spearman(L["arrays"]["dk"][:, 1], np.abs(e[:, 1])), threshold=cal.threshold,
                  scale_fit_updates=updates, fingerprints=fingerprints)
    passes = dict(spearman=values["spearman_w_abs_e"] is not None
                  and values["spearman_w_abs_e"] >= marks["spearman_min"],
                  partial=values["partial_spearman_given_eps_sd"] is not None
                  and values["partial_spearman_given_eps_sd"] >= marks["partial_spearman_min"],
                  baseline=values["spearman_w_abs_e"] is not None and values["knn_baseline_spearman"] is not None
                  and values["spearman_w_abs_e"] >= values["knn_baseline_spearman"])
    record = dict(stage_header(body, prereg_sha, "r5_signal", seed, code_changed), requirement="R5_signal",
                  values=values, passes=passes, passed=bool(all(passes.values())), diagnostic_only=True)
    write_once(critic_dir(world, seed) / "scores" / "R5_signal.json", record)
    return record


# ---------------------------------------------------------------- extraction (TRAINS ACTORS; prereg-gated)

@contextlib.contextmanager
def forbid_simulator():
    """R7(e): strengths are matched with simulator construction forbidden, so no return can inform a knob."""
    import gym

    def refuse(*_, **__):
        raise RuntimeError("simulator construction is forbidden while strengths are matched (R7e)")

    with mock.patch.object(gym, "make", refuse), mock.patch.object(SR.Physics, "__init__", refuse):
        yield


def _policy_file(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name) + ".msgpack"


def withheld_states(host, rows, seed):
    w = host.withheld_ids
    ids = w[PL._seeded_choice(len(w), rows, seed, 8)]
    return host.converted["observations"][ids].astype(np.float64)


def stage_extract(world, body, prereg_sha, seed, *, data_dir=None, host_dirs=None, log=print, code_changed=False):
    """Train every arm of every unit of this critic on the frozen critic; match strengths blind; lock the record."""
    verify_stage(world, body, prereg_sha, code_changed)
    from flax import serialization

    from experiments.truth import extraction as EX

    require(any(r in body["plan"] for r in EXTRACTION), "the plan has no extraction stage")
    pr = body["protocol"]
    ex = pr["extraction"]
    out = Path(world) / "extraction" / f"critic_{int(seed)}"
    require(not (out / "matched.json").exists(), f"{out / 'matched.json'} exists; extraction runs once")
    screen = read_json(critic_dir(world, seed) / "scores" / "R4a.json")
    require(screen["prereg_sha256"] == prereg_sha, "R4a ran under another pre-registration")
    host = host_for(body, seed, data_dir, host_dirs)
    poolT, _ = _pool(world, seed, "T")
    n, S, _ = poolT["actions"].shape
    qpiT = load_job(world, seed, "T", n, S, poolT)[0][..., 0]
    arms, _, keep_T = oracle_arms(body, poolT, qpiT, ex["shuf"])
    n = int(keep_T.sum())
    stats = EX.PolicyStats(host, withheld_states(host, ex["s_states"], pr["seed"]))
    extractor = EX.Extractor(host, poolT["converted_id"][keep_T], batch=ex["batch"], block=ex["block"])
    ones = np.ones(n, np.float32)
    level = ex["level"]
    units = {}
    out.mkdir(parents=True, exist_ok=True)
    with forbid_simulator():
        for unit in ex["seeds"]:
            udir = out / f"unit_{unit}"
            udir.mkdir(exist_ok=False)
            policies = {}

            def train(name, v, kind, knob):
                params, diag = extractor.train(v, int(seed) * 1000 + int(unit), ex["actor_steps"])
                payload = serialization.to_bytes(params)
                (udir / _policy_file(name)).write_bytes(payload)
                pi, q = stats(params)
                policies[name] = dict(kind=kind, knob=knob, file=_policy_file(name), params_sha256=sha256_hex(payload),
                                      q1_mean_withheld=float(q.mean()), training=diag)
                return pi

            pi_none = train("NONE", ones, "NONE", 1.0)
            policies["NONE"]["S"] = 0.0
            for m in ex["p0_ladder"]:
                pi = train(f"P0({m})", ones * np.float32(m), "P0", float(m))
                policies[f"P0({m})"]["S"] = EX.strength(pi, pi_none)
            target = policies[f"P0({level})"]["S"]
            matched = {}
            for arm, w in arms.items():
                center = screen["values"]["arms"].get(arm, {}).get("beta")
                ladder = ([center * f for f in ex["ladder_factors"]] if center else list(ex["fallback_ladder"]))
                svals = []
                for j, beta in enumerate(ladder):
                    name = f"{arm}@L{j}"
                    pi = train(name, (1.0 + beta * w).astype(np.float32), arm, float(beta))
                    policies[name]["S"] = EX.strength(pi, pi_none)
                    svals.append(policies[name]["S"])
                beta_star, rule = EX.match_ladder(ladder, svals, target)
                record = dict(ladder=ladder, ladder_S=svals, ladder_source="R4a beta x factors" if center else
                              "fallback ladder", rule=rule, beta=beta_star)
                if beta_star is not None:
                    pi = train(f"{arm}*", (1.0 + beta_star * w).astype(np.float32), arm, float(beta_star))
                    s_star = EX.strength(pi, pi_none)
                    policies[f"{arm}*"]["S"] = s_star
                    record.update(S=s_star, fidelity=abs(s_star / target - 1.0) if target > 0 else None)
                matched[arm] = record
                log(f"critic {seed} unit {unit} {arm}: beta* {beta_star}")
            units[str(unit)] = dict(policies=policies, S_target=target, matched=matched,
                                    target_definition=f"S(P0({level})) against NONE")
    record = dict(stage_header(body, prereg_sha, "extract", seed, code_changed), units=units,
                  simulator_forbidden_during_matching=True, pool_T_rows=int(n),
                  pool_T_rows_without_truth=int((~keep_T).sum()), host=host.identity,
                  arms=sorted(arms), s_states=int(len(stats.s)))
    path = write_once(out / "matched.json", record)
    (out / "matched.json.sha256").write_text(f"{SR.file_sha256(path)}  matched.json\n", encoding="utf8")
    return record


def stage_evaluate(world, body, prereg_sha, seed, *, data_dir=None, host_dirs=None, log=print, code_changed=False):
    """Verify the matching lock, then score every needed policy on the common reset bank; R3b on-policy labels."""
    verify_stage(world, body, prereg_sha, code_changed)
    from flax import serialization

    from experiments.truth import extraction as EX

    pr = body["protocol"]
    ex = pr["extraction"]
    out = Path(world) / "extraction" / f"critic_{int(seed)}"
    lock = (out / "matched.json.sha256").read_text(encoding="utf8").split()[0]
    require(SR.file_sha256(out / "matched.json") == lock, "matched.json changed after its lock; refusing to evaluate")
    matched = read_json(out / "matched.json")
    require(matched["prereg_sha256"] == prereg_sha, "matched.json was written under another pre-registration")
    host = host_for(body, seed, data_dir, host_dirs)
    policy = host.policy(ex["eval_slots"])
    invariance = require_invariant(policy, host)
    physics = SR.Physics(host.task.dataset)
    seeds = [ex["eval_reset_seed_base"] + i for i in range(ex["eval_episodes"])]
    results, onpolicy = {}, {}
    try:
        q0, v0 = PL.reset_bank(physics, seeds)
        for unit, u in matched["units"].items():
            names = ["NONE"] + [f"P0({m})" for m in ex["p0_ladder"]]
            names += [k for k in u["policies"] if k.startswith("OR_abs@")]
            names += [f"{arm}*" for arm, r in u["matched"].items() if r.get("beta") is not None]
            scores = {}
            for name in names:
                p = u["policies"][name]
                payload = (out / f"unit_{unit}" / p["file"]).read_bytes()
                require(sha256_hex(payload) == p["params_sha256"], f"{p['file']} differs from matched.json")
                params = serialization.from_bytes(host.native.actor.params, payload)
                every = pr["r3b"]["onpolicy_every"] if (name == "P0(0.25)" and "R3b" in body["plan"]) else None
                sc, ep, rec = EX.evaluate(physics, policy, params, q0, v0, host.task.dataset, horizon=pr["horizon"],
                                          every=every)
                scores[name] = dict(mean=float(sc.mean()), episodes=sc.tolist(),
                                    mean_length=float(ep["lengths"].mean()), simulator_errors=int(ep["failed"].sum()))
                if every:
                    onpolicy[unit] = (params, rec)
            results[unit] = scores
            log(f"critic {seed} unit {unit}: evaluated {len(names)} policies")
        r3b = None
        if "R3b" in body["plan"]:
            r3b = onpolicy_truth(world, body, host, physics, policy, onpolicy, out, seed)
    finally:
        physics.close()
    record = dict(stage_header(body, prereg_sha, "evaluate", seed, code_changed), matched_lock_verified=True,
                  position_invariance=invariance,
                  matched_sha256=lock, reset_seeds=[seeds[0], seeds[-1]], episodes=len(seeds), units=results,
                  r3b_onpolicy=r3b)
    return write_once(out / "evaluation.json", record)


def onpolicy_truth(world, body, host, physics, policy, onpolicy, out, seed):
    """R3b (iii): label on-policy states of each unit's P0(0.25) arm with that arm's own policy (deterministic and
    one jitter repeat) and record (Q1 - Q^pi) and thinness at (s, pi_arm(s))."""
    from experiments.truth import extraction as EX

    pr = body["protocol"]
    pools_meta = read_json(critic_dir(world, seed) / "pools" / "L.json")
    thin = PL.ThinnessIndex(host, k=pr["thinness"]["k"])
    thin.radius = pools_meta["radius"]
    probe = PL.CriticProbe(host)
    sd = np.load(critic_dir(world, seed) / "pools" / "jitter_sd.npz")
    summary = {}
    for unit, (params, rec) in onpolicy.items():
        arm_policy = EX.with_params(host.policy(pr["slots"]), params)
        pick = PL._seeded_choice(len(rec["t"]), pr["r3b"]["onpolicy_states"], pr["seed"], 9)
        qpos, qvel = rec["qpos"][pick], rec["qvel"][pick]
        obs = SR.observation(host.task, qpos, qvel)
        pi = arm_policy.actions(obs)
        q1 = probe.q(obs, pi)[:, 0].astype(np.float64)
        c = thin.counts(PL.features(host, obs, pi), np.full(len(obs), -1, np.int64))
        items = dict(qpos=qpos, qvel=qvel, action=pi, item_id=np.arange(len(obs), dtype=np.int64))
        labels = {}
        for mode in ("deterministic", "jitter"):
            spec = LB.LabelSpec(host.task.dataset, mode=mode, repeats=1, gamma=float(pr["gamma"]),
                                horizon=int(pr["horizon"]), seed=int(pr["seed"]),
                                jitter_scale=float(pr["roughness"]["jitter_scale"]), shard_size=int(pr["shard_size"]))
            LB.run_labels(out / f"onpolicy_{unit}_{mode}", physics, arm_policy, spec, items,
                          jitter_sd=(sd["qpos"], sd["qvel"]) if mode == "jitter" else None)
            labels[mode] = LB.load_labels(out / f"onpolicy_{unit}_{mode}")[0]["q"][:, 0]
        over = q1 - labels["deterministic"]
        np.savez(out / f"onpolicy_{unit}.npz", q1=q1, qpi=labels["deterministic"], qpi_jitter=labels["jitter"], c=c,
                 overestimate=over)
        summary[unit] = dict(states=int(len(obs)), mean_overestimate=float(np.nanmean(over)),
                             mean_abs_jitter_difference=float(np.nanmean(np.abs(labels["jitter"]
                                                                                  - labels["deterministic"]))))
    return summary


def stage_extraction_score(world, body, prereg_sha, *, code_changed=False):
    """R3b, R4b and R7 over all critics' units."""
    verify_stage(world, body, prereg_sha, code_changed)
    pr, marks = body["protocol"], body["pass_marks"]
    ex = pr["extraction"]
    thin_max = marks["thinness"]["thin_max"]
    level = ex["level"]
    critics = sorted(int(s) for s in body["hosts"])
    J, matched_all, cos = {}, {}, []
    over_all, c_all = [], []
    for seed in critics:
        out = Path(world) / "extraction" / f"critic_{seed}"
        evaluation = read_json(out / "evaluation.json")
        matched = read_json(out / "matched.json")
        require(evaluation["prereg_sha256"] == prereg_sha == matched["prereg_sha256"], "mixed pre-registrations")
        require(evaluation["matched_lock_verified"] and matched["simulator_forbidden_during_matching"], "R7e broken")
        screen = read_json(critic_dir(world, seed) / "scores" / "R4a.json")
        cos.append(screen["values"]["arms"]["OR_abs"]["cos_with_p0"])
        for unit, scores in evaluation["units"].items():
            J[(seed, unit)] = {k: v["mean"] for k, v in scores.items()}
            matched_all[(seed, unit)] = matched["units"][unit]
            f = out / f"onpolicy_{unit}.npz"
            if f.is_file():
                d = np.load(f)
                over_all.append(d["overestimate"])
                c_all.append(d["c"])
    units = sorted(J)
    by_critic = lambda f: [[f(u) for u in units if u[0] == s and f(u) is not None] for s in critics]
    reps = pr["bootstrap"]["replicates"]
    out_dir = Path(world) / "extraction"
    results = {}

    def get(u, name):
        return J[u].get(name)

    if "R4b" in body["plan"] or "R7" in body["plan"]:
        d_match = lambda u: (get(u, "OR_abs*") - get(u, f"P0({level})")) if get(u, "OR_abs*") is not None else None
        dm = np.array([d_match(u) for u in units if d_match(u) is not None])
        lo, hi = hierarchical_ci(by_critic(d_match), reps, [pr["seed"], 81])
        mean_of = lambda name: (float(np.mean([get(u, name) for u in units])) if all(get(u, name) is not None
                                                                                     for u in units) else None)
        or_names = sorted({k for u in units for k in J[u] if k.startswith("OR_abs@")} | {"OR_abs*"})
        or_best = max([m for m in (mean_of(k) for k in or_names) if m is not None], default=None)
        # Even caution at its best strength includes m = 1 (NONE, v = 1). Without it, a world where NONE beats every
        # P0(m != 1) arm and every OR-L-abs arm could still pass the best-strength margin.
        p0_names = ["NONE"] + [f"P0({m})" for m in ex["p0_ladder"]]
        p0_means = {k: mean_of(k) for k in p0_names}
        p0_best = max([m for m in p0_means.values() if m is not None], default=None)
        p0_best_arm = max((k for k in p0_names if p0_means[k] is not None), key=lambda k: p0_means[k], default=None)
        shuf_names = [f"SHUF_{k}*" for k in range(ex["shuf"])]

        def d_shuf(u):
            vals = [get(u, k) for k in shuf_names]
            if get(u, "OR_abs*") is None or any(v is None for v in vals):
                return None
            return get(u, "OR_abs*") - float(np.mean(vals))

        ds = [d_shuf(u) for u in units if d_shuf(u) is not None]
        v4 = dict(units=len(units), critics=critics, matched_differences=dm.tolist(),
                  matched_margin=float(dm.mean()) if len(dm) else None, matched_ci95=[lo, hi],
                  or_abs_best_mean=or_best, p0_best_mean=p0_best, p0_best_arm=p0_best_arm,
                  p0_best_rule="max over NONE (m = 1) and the P0 ladder of the mean J over units",
                  best_margin=(or_best - p0_best) if or_best is not None and p0_best is not None else None,
                  shuf_margin=float(np.mean(ds)) if ds else None,
                  designed_critics=pr["designed_critics"],
                  fewer_critics_than_designed=len(critics) < pr["designed_critics"])
        m4 = marks["R4b"]
        p4 = dict(matched=v4["matched_margin"] is not None and v4["matched_margin"] >= m4["matched_margin_min"]
                  and lo is not None and lo > 0,
                  best=v4["best_margin"] is not None and v4["best_margin"] >= m4["best_margin_min"],
                  shuf=v4["shuf_margin"] is not None and v4["shuf_margin"] >= m4["shuf_margin_min"])
        if "R4b" in body["plan"]:
            results["R4b"] = (v4, p4)
        if "R7" in body["plan"]:
            m7 = marks["R7"]
            known = [get(u, "OR_signed*") - get(u, "ANTI_signed*") for u in units
                     if get(u, "OR_signed*") is not None and get(u, "ANTI_signed*") is not None]
            z_known = paired_z(known)
            pairs = []
            for i in range(ex["shuf"]):
                for j in range(i + 1, ex["shuf"]):
                    d = [get(u, f"SHUF_{i}*") - get(u, f"SHUF_{j}*") for u in units
                         if get(u, f"SHUF_{i}*") is not None and get(u, f"SHUF_{j}*") is not None]
                    z = paired_z(d)  # +-inf when every unit gives the same nonzero difference
                    pairs.append(dict(pair=[i, j], z=finite_or_none(z), z_infinite=z is not None and math.isinf(z),
                                      fires=z is not None and abs(z) > m7["null_z"], n=len(d)))
            fired = sum(1 for p in pairs if p["fires"])
            from scipy import stats as st

            sd = float(dm.std(ddof=1)) if len(dm) >= 2 else None
            mde = ((st.norm.ppf(1 - pr["r7"]["alpha"] / 2) + st.norm.ppf(pr["r7"]["power"])) * sd / math.sqrt(len(dm))
                   if sd is not None else None)
            fid = [r.get("fidelity") for mu in matched_all.values() for r in mu["matched"].values()]
            unmatched = sum(1 for f in fid if f is None)
            fid_ok = [f for f in fid if f is not None]
            v7 = dict(known_answer=float(np.mean(known)) if known else None, known_answer_z=finite_or_none(z_known),
                      known_answer_n=len(known), null_pairs=pairs, null_pairs_fired=fired, mde=mde, paired_sd=sd,
                      max_match_error=max(fid_ok) if fid_ok else None, unmatched_arms=unmatched,
                      level1_cosines=cos, blind=True)
            p7 = dict(a=v7["known_answer"] is not None and v7["known_answer"] >= m7["known_answer_min"]
                      and z_known is not None and z_known >= m7["known_answer_z_min"],
                      b=len(pairs) > 0 and fired <= m7["null_pairs_max"],
                      c=mde is not None and mde <= m7["mde_max"],
                      d=unmatched == 0 and bool(fid_ok) and max(fid_ok) <= m7["match_tolerance"]
                      and all(c is not None and c < m7["cosine_max"] for c in cos),
                      e=True)
            results["R7"] = (v7, p7)
    if "R3b" in body["plan"]:
        m3 = marks["R3b"]
        rise = lambda u: ((matched_all[u]["policies"]["P0(0.25)"]["q1_mean_withheld"]
                           - matched_all[u]["policies"]["NONE"]["q1_mean_withheld"])
                          / abs(matched_all[u]["policies"]["NONE"]["q1_mean_withheld"]))
        rises = [rise(u) for u in units]
        name_of = lambda m: "NONE" if m == 1.0 else f"P0({m})"
        means = {m: float(np.mean([get(u, name_of(m)) for u in units])) for m in pr["r3b"]["m"]}
        best = max(means, key=means.get)
        drop = lambda u: get(u, name_of(best)) - get(u, "P0(0.25)")
        drops = [drop(u) for u in units]
        lo, hi = hierarchical_ci(by_critic(drop), reps, [pr["seed"], 83])
        if over_all:
            over, c = np.concatenate(over_all), np.concatenate(c_all)
            ok = np.isfinite(over)
            pos = np.maximum(over[ok], 0.0)
            share = float(pos[c[ok] <= thin_max].sum() / pos.sum()) if pos.sum() > 0 else None
        else:
            share = None
        v3 = dict(q_rise_mean=float(np.mean(rises)), q_rises=rises,
                  mean_return_by_m={str(k): v for k, v in means.items()},
                  best_m=best, return_drop_mean=float(np.mean(drops)), return_drop_ci95=[lo, hi],
                  thin_overestimate_share=share, onpolicy_states=int(sum(len(o) for o in over_all)))
        p3 = dict(q_rise=v3["q_rise_mean"] >= m3["q_rise_min"],
                  return_drop=v3["return_drop_mean"] >= m3["return_drop_min"] and lo is not None and lo > 0,
                  thin_overestimate=share is not None and share >= m3["thin_overestimate_share_min"])
        results["R3b"] = (v3, p3)
    written = {}
    for req, (values, passes) in results.items():
        record = dict(stage_header(body, prereg_sha, "extraction-score", None, code_changed), requirement=req,
                      values=values, passes=passes, passed=bool(all(passes.values())), pass_marks=marks[req])
        write_once(out_dir / f"{req}.json", record)
        written[req] = record
    return written


# ---------------------------------------------------------------- verdict

def verdict(world, body, prereg_sha, *, code_changed=False):
    verify_stage(world, body, prereg_sha, code_changed)
    world = Path(world)
    critics = sorted(int(s) for s in body["hosts"])
    rows, mixed, code_flags = [], [], []
    for req in body["plan"]:
        if req in PER_CRITIC:
            per = {}
            for seed in critics:
                f = critic_dir(world, seed) / "scores" / f"{req}.json"
                if f.is_file():
                    r = read_json(f)
                    if r["prereg_sha256"] != prereg_sha:
                        mixed.append(str(f))
                    code_flags.append(r.get("code_changed_since_prereg", False))
                    per[seed] = dict(passed=r["passed"], passes=r["passes"], values=r["values"])
                else:
                    per[seed] = None
            status = ("missing" if any(v is None for v in per.values())
                      else "pass" if all(v["passed"] for v in per.values()) else "fail")
            rows.append(dict(requirement=req, status=status, per_critic={str(k): v for k, v in per.items()},
                             pass_marks=body["pass_marks"][req]))
        else:
            f = world / "extraction" / f"{req}.json"
            if f.is_file():
                r = read_json(f)
                if r["prereg_sha256"] != prereg_sha:
                    mixed.append(str(f))
                code_flags.append(r.get("code_changed_since_prereg", False))
                rows.append(dict(requirement=req, status="pass" if r["passed"] else "fail", passes=r["passes"],
                                 values=r["values"], pass_marks=body["pass_marks"][req]))
            else:
                rows.append(dict(requirement=req, status="missing", pass_marks=body["pass_marks"][req]))
    require(not mixed, f"outputs from another pre-registration: {mixed}")
    statuses = {r["requirement"]: r["status"] for r in rows}
    complete = all(s != "missing" for s in statuses.values())
    passed = complete and all(s == "pass" for s in statuses.values())
    if body["plan_name"] == "graft":
        meaning = ("External-validity graft: real critic errors sit where data is thin" if passed else
                   "External-validity graft: the planned checks did not all pass (see rows)")
    else:
        meaning = ("The world passes every planned requirement and can judge BCA's placement" if passed else
                   "The world does not (yet) pass; per the design, failures of R1, R2 or R4 move to D-thin(0.1), "
                   "then D-mix(10); if all fail, the world is rejected for the BC-multiplier hook")
    if body["synthetic"]:  # the meaning line is often read alone (the run command prints it)
        meaning = "SYNTHETIC TEST RUN, NOT A FINDING: " + meaning
    record = dict(stage_header(body, prereg_sha, "verdict", None, code_changed or any(code_flags)), rows=rows,
                  statuses=statuses, complete=complete, passed=passed, meaning=meaning,
                  post_hoc_flags=dict(code_changed=bool(code_changed or any(code_flags)),
                                      pass_marks_equal_design=body["pass_marks_equal_design"]))
    path = world / "verdict.json"
    if path.exists():
        path.unlink()
    write_once(path, record)
    (world / "verdict.md").write_text(verdict_markdown(record), encoding="utf8")
    return record


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def verdict_markdown(record):
    lines = []
    if record["synthetic"]:
        lines += ["**SYNTHETIC TEST RUN. NOT A FINDING.** " + SYNTHETIC_NOTE, ""]
    plan = ", ".join(record["plan"])
    lines += [f"# D4RL-RT verdict: {record['dataset']} (profile {record['profile']}, plan {plan})", "",
              f"Pre-registration sha256 `{record['prereg_sha256']}`. {record['meaning']}.", "",
              "| Requirement | Status | Key values |", "|---|---|---|"]
    keys = {"R1": ("pi_thin_share", "withheld_thin_share", "pi_dense_share"),
            "R2": ("spearman", "spearman_lower95", "thin_dense_median_ratio", "thin_sq_share_ratio",
                   "partial_spearman_given_termination"),
            "R3a": ("misled_share", "misled_rate_thin", "misled_rate_dense"),
            "R3b": ("q_rise_mean", "return_drop_mean", "thin_overestimate_share"),
            "R4a": ("rho_host",), "R4b": ("matched_margin", "best_margin", "shuf_margin"),
            "R5_world": ("noise_share", "spearman_abs_b_abs_e", "truth_noise_share"),
            "R6": ("return_gain", "classifier_auc", "action_ratio", "supported_share"),
            "R7": ("known_answer", "known_answer_z", "null_pairs_fired", "mde", "max_match_error")}
    for row in record["rows"]:
        req = row["requirement"]
        if "per_critic" in row:
            parts = []
            for seed, r in row["per_critic"].items():
                if r is None:
                    parts.append(f"{seed}: missing")
                else:
                    vals = ", ".join(f"{k} {_fmt(r['values'].get(k))}" for k in keys.get(req, ()) if k in r["values"])
                    parts.append(f"{seed}: {vals}")
            text = "; ".join(parts)
        else:
            vals = row.get("values", {})
            text = ", ".join(f"{k} {_fmt(vals.get(k))}" for k in keys.get(req, ()) if k in vals)
        lines.append(f"| {req} | {row['status']} | {text} |")
    return "\n".join(lines) + "\n"


def combine(verdict_paths):
    """The design's cross-task R4b rule: pass on at least one task with the other not negative."""
    records = [read_json(p) for p in verdict_paths]
    tasks = {}
    for r in records:
        row = next((x for x in r["rows"] if x["requirement"] == "R4b"), None)
        tasks[r["dataset"]] = None if row is None or row["status"] == "missing" else row
    require(all(v is not None for v in tasks.values()) and len(tasks) == 2, "R4b must be complete on both tasks")
    passed = {k: v["status"] == "pass" for k, v in tasks.items()}
    margins = {k: v["values"]["matched_margin"] for k, v in tasks.items()}
    ok = any(passed.values()) and all(m is not None and m >= 0 for m in margins.values())
    synthetic = any(r["synthetic"] for r in records)
    return dict(rule="R4b passes on at least one task and the other task's matched margin is not negative",
                passed_by_task=passed, matched_margins=margins, gate_passed=ok, synthetic=synthetic,
                note=SYNTHETIC_NOTE if synthetic else "pre-registered cross-task gate")


# ---------------------------------------------------------------- run everything planned

def run_all(world, prereg_sha, *, data_dir=None, host_dirs=None, allow_code_change=False, log=print):
    body, sha, changed = load_prereg(world, prereg_sha, allow_code_change)
    kw = dict(data_dir=data_dir, host_dirs=host_dirs, code_changed=changed)
    for seed in sorted(int(s) for s in body["hosts"]):
        cdir = critic_dir(world, seed)
        if not (cdir / "pools" / "summary.json").exists():
            log(f"pools: critic {seed}")
            stage_pools(world, body, sha, seed, **kw)
        if not labels_complete(world, body, seed):
            log(f"labels: critic {seed}")
            stage_labels(world, body, sha, seed, log=log, **kw)
        if not all((cdir / "scores" / f"{r}.json").exists() for r in body["plan"] if r in PER_CRITIC and r != "R4a"):
            log(f"score: critic {seed}")
            stage_score(world, body, sha, seed, code_changed=changed)
        if "R4a" in body["plan"] and not (cdir / "scores" / "R4a.json").exists():
            log(f"screen: critic {seed}")
            stage_screen(world, body, sha, seed, **kw)
    if any(r in body["plan"] for r in EXTRACTION):
        for seed in sorted(int(s) for s in body["hosts"]):
            out = Path(world) / "extraction" / f"critic_{seed}"
            if not (out / "matched.json").exists():
                log(f"extract: critic {seed}")
                stage_extract(world, body, sha, seed, log=log, **kw)
            if not (out / "evaluation.json").exists():
                log(f"evaluate: critic {seed}")
                stage_evaluate(world, body, sha, seed, log=log, **kw)
        if not all((Path(world) / "extraction" / f"{r}.json").exists() for r in EXTRACTION if r in body["plan"]):
            stage_extraction_score(world, body, sha, code_changed=changed)
    return verdict(world, body, sha, code_changed=changed)


# ---------------------------------------------------------------- command line

def _hosts(values):
    out = {}
    for v in values or []:
        seed, _, path = v.partition("=")
        require(seed.isdigit() and path, f"--host takes SEED=RUN_DIR, got {v!r}")
        out[int(seed)] = path
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="D4RL-RT acceptance gate (see the module docstring).")
    sub = parser.add_subparsers(dest="command", required=True)

    s0 = sub.add_parser("stage0", help="Stage 0: no training, no pre-registration")
    s0.add_argument("--world", required=True, type=Path)
    s0.add_argument("--task", required=True)
    where = s0.add_mutually_exclusive_group()
    where.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    where.add_argument("--file", type=Path)
    s0.add_argument("--run-dir", type=Path, help="host run directory (split, eps-noise share)")
    s0.add_argument("--rows", type=int, default=1000)
    s0.add_argument("--seed", type=int, default=0)

    fz = sub.add_parser("freeze", help="write the pre-registration and print its sha256")
    fz.add_argument("--world", required=True, type=Path)
    fz.add_argument("--task", required=True)
    fz.add_argument("--host", action="append", required=True, help="SEED=RUN_DIR (repeat for each critic)")
    fz.add_argument("--profile", default="design", choices=sorted(PROFILES))
    fz.add_argument("--plan", default="full", choices=sorted(PLANS))
    fz.add_argument("--data-dir", type=Path)
    fz.add_argument("--state-source", choices=("auto", "rebuilt"), default="auto",
                    help="restore pool states from infos/* when present (auto) or rebuild them from observations")
    for tol, default in SR.DEFAULT_TOLERANCES.items():
        fz.add_argument("--" + tol.replace("_", "-"), type=float, default=default)

    stages = ("pools", "labels", "score", "screen", "r5-signal", "extract", "evaluate", "extraction-score", "verdict",
              "run")
    for name in stages:
        p = sub.add_parser(name)
        p.add_argument("--world", required=True, type=Path)
        p.add_argument("--prereg-sha256", required=True)
        p.add_argument("--data-dir", type=Path)
        p.add_argument("--host", action="append", help="SEED=RUN_DIR override of the frozen run directory")
        p.add_argument("--allow-code-change", action="store_true")
        if name not in ("extraction-score", "verdict", "run"):
            p.add_argument("--critic", type=int, required=True)
        if name == "labels":
            p.add_argument("--jobs", nargs="*")
            p.add_argument("--worker", type=int, default=0)
            p.add_argument("--workers", type=int, default=1)
        if name == "r5-signal":
            p.add_argument("--allow-bca-band", action="store_true",
                           help="fit BCA's band (OFF by default; needs the user's approval)")
            p.add_argument("--updates", type=int, default=100_000)
    cb = sub.add_parser("combine", help="cross-task R4b rule over two verdict.json files")
    cb.add_argument("verdicts", nargs=2, type=Path)
    args = parser.parse_args(argv)

    if args.command == "stage0":
        s = stage0(args.world, args.task, data_dir=args.data_dir, file=args.file, run_dir=args.run_dir, rows=args.rows,
                   seed=args.seed)
        print(LB.strict_json(jsonable({k: s[k] for k in ("dataset", "synthetic", "rebuild", "eps_noise")}
                                      | {"gate_passed": s["gate"]["passed"],
                                         "negative_control_passed": s["gate_negative"]["passed"]})))
        return
    if args.command == "freeze":
        tol = {k: getattr(args, k) for k in SR.DEFAULT_TOLERANCES}
        digest = freeze(args.world, args.task, hosts=_hosts(args.host), stage0_dir=args.world / "stage0",
                        profile=args.profile, plan=args.plan, tolerances=tol, data_dir=args.data_dir,
                        state_source=args.state_source)
        print(digest)
        return
    if args.command == "combine":
        print(LB.strict_json(jsonable(combine(args.verdicts))))
        return
    host_dirs = _hosts(args.host)
    if args.command == "run":
        r = run_all(args.world, args.prereg_sha256, data_dir=args.data_dir, host_dirs=host_dirs,
                    allow_code_change=args.allow_code_change)
        print(LB.strict_json(jsonable({k: r[k] for k in ("statuses", "passed", "meaning", "synthetic")})))
        return
    body, sha, changed = load_prereg(args.world, args.prereg_sha256, args.allow_code_change)
    kw = dict(data_dir=args.data_dir, host_dirs=host_dirs, code_changed=changed)
    if args.command == "pools":
        stage_pools(args.world, body, sha, args.critic, **kw)
    elif args.command == "labels":
        stage_labels(args.world, body, sha, args.critic, jobs=args.jobs, worker=args.worker, workers=args.workers, **kw)
    elif args.command == "score":
        stage_score(args.world, body, sha, args.critic, code_changed=changed)
    elif args.command == "screen":
        stage_screen(args.world, body, sha, args.critic, **kw)
    elif args.command == "r5-signal":
        stage_r5_signal(args.world, body, sha, args.critic, allow_bca_band=args.allow_bca_band, updates=args.updates,
                        **kw)
    elif args.command == "extract":
        stage_extract(args.world, body, sha, args.critic, **kw)
    elif args.command == "evaluate":
        stage_evaluate(args.world, body, sha, args.critic, **kw)
    elif args.command == "extraction-score":
        stage_extraction_score(args.world, body, sha, code_changed=changed)
    elif args.command == "verdict":
        r = verdict(args.world, body, sha, code_changed=changed)
        print(verdict_markdown(r))


if __name__ == "__main__":
    main()
