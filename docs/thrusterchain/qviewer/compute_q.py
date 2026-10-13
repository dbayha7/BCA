"""Data for the "critic in action" viewer of the ThrusterChain page.

For each dataset quality (random, medium-replay, medium, medium-expert, expert), each TD3+BC host critic of seed 0
("gap": trained with the densest-15% removal, which takes out almost every 80% hold of thruster 1; "full": trained on
all the data) and two situations (the training mix of currents and the strong test mix):
  - the host's own actor drives 100 steps from one fixed start (the real episode), by truth.rollout with the float64
    copy of the actor (truth.MLPPolicy), exactly as the package's evaluate_actor scores it;
  - at every visited state, thruster 1's command is swept over 41 values from 0 to -1 (0% to 100% upstream, rounded to
    float32 as env.step rounds a command) with thrusters 2-4 at the actor's command, and at each value we record
      the critic's Q(s, a) = min(Q1, Q2) (float32 network, normalised float32 observation, float32 command), and
      the exact true value Q^pi(s, a) of the actor's own policy: one step with that command, then the actor drives;
      horizon 600, discount 0.98 (truth.q_value_jax, float64);
  - the same two numbers at the actor's own command.
Start choice (fixed before any outcome of these starts was looked at): the first start of the shared start banks whose
current lies inside the interquartile range (25th to 75th percentile) of its own mix's current law. The banks share
one uniform draw per start, so this is the same start index in both mixes (same floats and waves, only c differs).

Context (summary numbers only): the same 80%-versus-100% comparison on the first ten starts of each bank.

Writes /root/bca-shift/runs/qviewer/q_data.json and q_checks.json, then copies both to the Windows qviewer folder.
Reads runs/shift_iw_v5 through the package's own loaders (run.load_data, run.load_host_state, data.load_start_banks),
which verify every file against its recorded SHA-256. Writes nothing anywhere else.
"""

import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import experiments.shift  # noqa: F401  (gymnasium path and version checks; before gymnasium)
from experiments.shift import learners as LR
from experiments.shift import recorder as RC
from experiments.shift import thruster_chain as TC
from experiments.shift import truth as TR
from experiments.shift_iw import data as D
from experiments.shift_iw import run as RUN
from experiments.shift_iw import world as W

ROOT = Path("/root/bca-shift/runs/shift_iw_v5")
OUT = Path("/root/bca-shift/runs/qviewer")
WIN = Path("/mnt/c/Users/David/dev/bca-shift-work/qviewer")
SEED = 0
H = 600                      # horizon of every exact value
H_CHECK = 1000               # truth.HORIZON, for the truncation check
STEPS = 100
KINDS = ("gap", "full")
SITUATIONS = (("training", "none"), ("strong", "strong"))   # (name on the page, start-bank law)
QUALITIES = W.QUALITIES
N_SWEEP = 41
SWEEP = (-(np.arange(N_SWEEP) / (N_SWEEP - 1.0))).astype(np.float32).astype(np.float64) + 0.0   # 0 ... -1, float32
I80 = int(np.flatnonzero(SWEEP == -TC.KNEE)[0])   # the 80% point (float32(-0.8), the knee: it does not stall)
I100 = N_SWEEP - 1
CONTEXT_STARTS = 10
TOL_REL = 1e-6
STATE_FIELDS = ["position of float 1", "speed of float 1", "position of float 2", "speed of float 2",
                "position of float 3", "speed of float 3", "position of float 4", "speed of float 4",
                "position of float 5", "speed of float 5", "current c", "wave w1", "wave w2"]
assert I80 == 32 and SWEEP[0] == 0.0 and SWEEP[-1] == -1.0


def sig4(x):
    """4 significant digits, as Python floats (JSON numbers); nested lists for arrays."""
    a = np.asarray(x, np.float64)
    if a.ndim == 0:
        v = float(a)
        if not math.isfinite(v):
            raise ValueError("non-finite number in the output")
        return float(f"{v:.4g}") + 0.0
    return [sig4(v) for v in a]


def ints(x):
    return [int(v) for v in np.asarray(x).ravel()]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def critic_q(critic_params, states, actions, mean, std):
    """The critic's two heads and q = min(Q1, Q2) at float64 states and commands, fed as the critic sees them: the
    float32 observation normalised in float32 (learners.Prepared.normalize) and the float32 command."""
    obs = (np.asarray(states, np.float32) - mean) / std
    heads = LR.critic_rows(critic_params, obs, np.asarray(actions, np.float32))[0]
    return heads.min(axis=1), heads


def sweep(pol, critic_params, mean, std, states, cmds, a1):
    """Critic q and exact Q^pi at thruster-1 commands a1 (n, k) per state, thrusters 2-4 at cmds (n, 4)."""
    n, k = a1.shape
    A = np.repeat(np.asarray(cmds, np.float64)[:, None, :], k, axis=1)
    A[:, :, 0] = a1
    S = np.repeat(np.asarray(states, np.float64)[:, None, :], k, axis=1).reshape(-1, TC.DX)
    A = A.reshape(-1, TC.DA)
    q, _ = critic_q(critic_params, S, A, mean, std)
    qpi = TR.q_value_jax(pol, S, A, W.PHYSICS, horizon=H)
    return q.reshape(n, k), qpi.reshape(n, k)


def choose_start(banks):
    """The first start whose current lies inside the interquartile range of its own mix's law, in both mixes."""
    rows = []
    for j in range(len(banks["none"])):
        ok = True
        for _, law in SITUATIONS:
            c = banks[law][j, TC.CURRENT]
            lo, hi = W.inverse_cdf(np.array([0.25, 0.75]), law)
            ok &= bool(lo <= c <= hi)
        rows.append(ok)
        if ok:
            return j, rows
    raise RuntimeError("no typical start")


def tail_check(pol, states_all, rewards, qpi_actor):
    """Check 1: Q^pi(s_t, pi(s_t)) = sum_{k<100-t} g^k r_{t+k} + g^(100-t) x (the actor's own simulated tail from s_100,
    H - (100 - t) steps). Returns the largest relative and absolute gaps."""
    g = W.GAMMA
    tail = TR.rollout(pol, states_all[STEPS][None], W.PHYSICS, length=H - 1, record=True).rewards[0]
    pw = g ** np.arange(H)
    rel, ab = 0.0, 0.0
    for t in range(STEPS):
        m = STEPS - t
        total = float(np.sum(pw[:m] * rewards[t:])) + g ** m * float(np.sum(pw[:H - m] * tail[:H - m]))
        d = abs(total - qpi_actor[t])
        ab = max(ab, d)
        rel = max(rel, d / abs(qpi_actor[t]))
    return rel, ab


def episode_summary(cmds, push, q_sw, qpi_sw, q_act, qpi_act):
    n = len(cmds)
    rows = np.arange(n)
    crit_rise = q_sw[:, I100] - q_sw[:, I80]
    true_chg = qpi_sw[:, I100] - qpi_sw[:, I80]
    keeps_rising = np.all(np.diff(q_sw[:, I80:], axis=1) > 0, axis=1)
    keeps_falling = np.all(np.diff(qpi_sw[:, I80:], axis=1) < 0, axis=1)
    rise_fall = (crit_rise > 0) & (true_chg < 0)
    a_hat = q_sw.argmax(1)
    a_best = qpi_sw.argmax(1)
    cost = qpi_sw[rows, a_best] - qpi_sw[rows, a_hat]
    pull = crit_rise - true_chg
    t1_past = cmds[:, 0] < -TC.KNEE
    mean_or_none = lambda v, m: float(v[m].mean()) if m.any() else None
    return {
        "actor_thruster1_past80_steps": int(t1_past.sum()),
        "actor_any_thruster_past80_steps": int(np.any(np.abs(cmds) > TC.KNEE, axis=1).sum()),
        "a_hat_past80_states": int((a_hat > I80).sum()),
        "a_hat_at_100_states": int((a_hat == I100).sum()),
        "a_hat_at_80_states": int((a_hat == I80).sum()),
        "best_true_past80_states": int((a_best > I80).sum()),
        "best_true_at_80_states": int((a_best == I80).sum()),
        "critic_higher_at_100_than_80_states": int((crit_rise > 0).sum()),
        "critic_keeps_rising_80_to_100_states": int(keeps_rising.sum()),
        "truth_lower_at_100_than_80_states": int((true_chg < 0).sum()),
        "truth_keeps_falling_80_to_100_states": int(keeps_falling.sum()),
        "critic_rises_while_truth_falls_states": int(rise_fall.sum()),
        "critic_rises_while_truth_falls_steps": ints(np.flatnonzero(rise_fall)),
        "critic_change_80_to_100_mean": float(crit_rise.mean()),
        "truth_change_80_to_100_mean": float(true_chg.mean()),
        "critic_change_80_to_100_where_truth_falls_mean": mean_or_none(crit_rise, rise_fall),
        "truth_change_80_to_100_where_critic_rises_mean": mean_or_none(true_chg, rise_fall),
        "over_preference_100_vs_80_mean": float(pull.mean()),
        "over_preference_100_vs_80_max": float(pull.max()),
        "true_cost_of_a_hat_mean": float(cost.mean()),
        "true_cost_of_a_hat_max": float(cost.max()),
        "critic_minus_truth_at_actor_command_mean": float(np.mean(q_act - qpi_act)),
        "push_mean_where_critic_rises_while_truth_falls": mean_or_none(push, rise_fall),
        "push_mean_elsewhere": mean_or_none(push, ~rise_fall),
    }


SUMMARY_NOTES = {
    "actor_thruster1_past80_steps": "steps (of 100) at which the actor commanded thruster 1 past 80% upstream",
    "actor_any_thruster_past80_steps": "steps at which the actor commanded any thruster past 80%",
    "a_hat_past80_states": "states at which the swept command the critic rates highest (a-hat) is past 80%",
    "a_hat_at_100_states": "states at which a-hat is 100%",
    "a_hat_at_80_states": "states at which a-hat is exactly 80%",
    "best_true_past80_states": "states at which the swept command with the highest true value Q^pi is past 80%",
    "best_true_at_80_states": "states at which the swept command with the highest Q^pi is exactly 80%",
    "critic_higher_at_100_than_80_states": "states at which the critic's Q at 100% is above its Q at 80%",
    "critic_keeps_rising_80_to_100_states": "states at which the critic's Q rises at every sweep step from 80% to 100%",
    "truth_lower_at_100_than_80_states": "states at which Q^pi at 100% is below Q^pi at 80%",
    "truth_keeps_falling_80_to_100_states": "states at which Q^pi falls at every sweep step from 80% to 100%",
    "critic_rises_while_truth_falls_states": "states at which the critic's Q is higher at 100% than at 80% while Q^pi "
                                             "is lower",
    "critic_rises_while_truth_falls_steps": "the steps t (0-99) of those states",
    "critic_change_80_to_100_mean": "mean over states of Q(100%) - Q(80%)",
    "truth_change_80_to_100_mean": "mean over states of Q^pi(100%) - Q^pi(80%)",
    "critic_change_80_to_100_where_truth_falls_mean": "mean of Q(100%) - Q(80%) over the states where the critic rises "
                                                      "while the truth falls",
    "truth_change_80_to_100_where_critic_rises_mean": "mean of Q^pi(100%) - Q^pi(80%) over the same states",
    "over_preference_100_vs_80_mean": "mean over states of [Q(100%) - Q(80%)] - [Q^pi(100%) - Q^pi(80%)]: how much "
                                      "more the critic prefers 100% over 80% than the truth does",
    "over_preference_100_vs_80_max": "its largest value over the states",
    "true_cost_of_a_hat_mean": "mean over states of Q^pi(best swept command) - Q^pi(a-hat): the true value lost by "
                               "taking the critic's top-rated thruster-1 command instead of the truly best one",
    "true_cost_of_a_hat_max": "its largest value",
    "critic_minus_truth_at_actor_command_mean": "mean over states of Q - Q^pi at the actor's own command (the critic's "
                                                "level error)",
    "push_mean_where_critic_rises_while_truth_falls": "mean water push c + w1 at those states",
    "push_mean_elsewhere": "mean water push c + w1 at the other states",
}


def context_block(pol, critic_params, mean, std, x0s):
    """The 80%-versus-100% comparison on the first CONTEXT_STARTS starts of a bank (summary numbers only)."""
    run = TR.rollout(pol, x0s, W.PHYSICS, length=STEPS, record=True)
    states = run.states[:, :STEPS].reshape(-1, TC.DX)
    cmds = run.commands.reshape(-1, TC.DA)
    a1 = np.repeat(SWEEP[[I80, I100]][None], len(states), axis=0)
    q, qpi = sweep(pol, critic_params, mean, std, states, cmds, a1)
    crit_rise, true_chg = q[:, 1] - q[:, 0], qpi[:, 1] - qpi[:, 0]
    return dict(starts=int(len(x0s)), states=int(len(states)), mean_return=float(run.J.mean()),
                returns=[float(v) for v in run.J],
                actor_thruster1_past80_share=float(np.mean(cmds[:, 0] < -TC.KNEE)),
                critic_higher_at_100_than_80_share=float(np.mean(crit_rise > 0)),
                truth_lower_at_100_than_80_share=float(np.mean(true_chg < 0)),
                critic_rises_while_truth_falls_share=float(np.mean((crit_rise > 0) & (true_chg < 0))),
                critic_change_80_to_100_mean=float(crit_rise.mean()),
                truth_change_80_to_100_mean=float(true_chg.mean()),
                over_preference_100_vs_80_mean=float(np.mean(crit_rise - true_chg)))


def round_summary(s):
    out = {}
    for k, v in s.items():
        if isinstance(v, list):
            out[k] = v
        elif isinstance(v, float):
            out[k] = sig4(v)
        else:
            out[k] = v
    return out


def main():
    t_start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    checks = dict(schema="bca-shift-qviewer-checks-v1", tolerance_relative=TOL_REL, hosts=[], episodes=[],
                  anchors=[], start=None)
    log = lambda m: print(time.strftime("%H:%M:%S ") + m, flush=True)

    ctx0 = RUN.Context(root=ROOT, quality="expert", log=lambda m: None)
    spec = RUN.spec_digest(ctx0)
    banks, brec = D.load_start_banks(ctx0.banks_root)        # verifies the HDF5 manifest and the JSON sidecar
    j, _ = choose_start(banks)
    start_info = {}
    for name, law in SITUATIONS:
        x0 = banks[law][j]
        c = float(x0[TC.CURRENT])
        start_info[name] = dict(bank=law, start_index=int(j), current=c, current_quantile=float(W.cdf(c, law)),
                                interquartile_range=[float(v) for v in W.inverse_cdf(np.array([0.25, 0.75]), law)],
                                wave_amplitude=float(math.hypot(x0[TC.WAVE1], x0[TC.WAVE2])),
                                start_state=[float(v) for v in x0],
                                anchor_random=float(brec["anchors"][law]["anchor"]["random"]),
                                anchor_top=float(brec["anchors"][law]["anchor"]["top"]))
    first = banks["none"][0, TC.CURRENT]
    checks["start"] = dict(rule="the first start whose current lies inside the interquartile range of its own mix, in "
                                "both mixes (the banks share one uniform draw per start, so it is one index)",
                           chosen_index=int(j),
                           start0_quantile=float(W.cdf(first, "none")),
                           start0_currents={law: float(banks[law][0, TC.CURRENT]) for _, law in SITUATIONS})
    log(f"start index {j}: " + ", ".join(f"{k} c={v['current']:.3f} (quantile {v['current_quantile']:.3f})"
                                         for k, v in start_info.items()))

    # anchors recomputed from their definition (bank means) and the reference returns from the chosen start
    for name, law in SITUATIONS:
        x0s = banks[law]
        sa = TR.rollout(RC.stall_aware(W.PHYSICS), x0s, W.PHYSICS).J
        rq = TR.rollout(D.quality_on_bank("random", len(x0s)), x0s, W.PHYSICS).J
        a = brec["anchors"][law]["anchor"]
        checks["anchors"].append(dict(situation=name, bank=law, stall_aware_mean=float(sa.mean()),
                                      recorded_top=float(a["top"]), random_mean=float(rq.mean()),
                                      recorded_random=float(a["random"]),
                                      ok=bool(sa.mean() == a["top"] and rq.mean() == a["random"])))
        start_info[name]["stall_aware_return_from_start"] = float(sa[j])
        start_info[name]["random_commands_return_from_start"] = float(rq[j])

    episodes, summaries, context = [], [], []
    timing = {}
    for quality in QUALITIES:
        tq = time.perf_counter()
        ctx = RUN.Context(root=ROOT, quality=quality, log=lambda m: None)
        data = RUN.load_data(ctx, SEED, spec)           # verifies every data file against world.json and manifests
        mean, std = D.normalization(data)              # the learners' rule (gapped kept rows), both hosts
        for kind in KINDS:
            rec = RUN.require_host(ctx, kind, SEED, spec)
            state = RUN.load_host_state(ctx, kind, SEED, spec)   # checks the snapshot file's SHA-256
            snap = rec["result"]["snapshots"][-1]
            actor_sha = LR.tree_sha256(state.actor.params)
            critic_sha = LR.tree_sha256(state.critic.params)
            norm_ok = bool(np.array_equal(np.asarray(rec["result"]["data"]["obs_mean"], np.float32), mean)
                           and np.array_equal(np.asarray(rec["result"]["data"]["obs_std"], np.float32), std))
            pol = TR.MLPPolicy(state.actor.params, mean, std, name=f"{quality} {kind} actor (float64 copy)")
            host_check = dict(quality=quality, critic=kind, seed=SEED, record=str(RUN.host_dir(ctx, kind, SEED)),
                              snapshot_update=int(snap["update"]), snapshot_file_sha256=snap["file_sha256"],
                              snapshot_file_ok=True, critic_sha256=critic_sha,
                              critic_sha256_ok=critic_sha == snap["critic_sha256"],
                              actor_sha256_ok=actor_sha == snap["actor_sha256"], normalisation_ok=norm_ok,
                              host_failed=bool(rec["result"]["failed"]))
            for sit, law in SITUATIONS:
                te = time.perf_counter()
                x0 = banks[law][j]
                run = TR.rollout(pol, x0[None], W.PHYSICS, length=STEPS, record=True)
                S_all = run.states[0]
                states, cmds, rewards = S_all[:STEPS], run.commands[0], run.rewards[0]
                push = states[:, TC.CURRENT] + states[:, TC.WAVE1]
                # float32 network actor vs its float64 copy (the copy is what drives)
                obs_n = (np.asarray(states, np.float32) - mean) / std
                a32 = LR.actor_rows(state.actor.params, obs_n).astype(np.float64)
                # the actor's own command: separately computed values
                q_act, _ = critic_q(state.critic.params, states, cmds, mean, std)
                qpi_act = TR.q_value_jax(pol, states, cmds, W.PHYSICS, horizon=H)
                # the sweep
                q_sw, qpi_sw = sweep(pol, state.critic.params, mean, std, states, cmds,
                                     np.repeat(SWEEP[None], STEPS, axis=0))
                # check 3: the sweep's code path at the actor's own thruster-1 command
                q_own, qpi_own = sweep(pol, state.critic.params, mean, std, states, cmds, cmds[:, :1])
                # check 1: discounted sum of the episode's own rewards plus the simulated tail
                rel1, abs1 = tail_check(pol, S_all, rewards, qpi_act)
                # truncation: horizon 600 vs truth.HORIZON (1,000)
                qpi_long = TR.q_value_jax(pol, states, cmds, W.PHYSICS, horizon=H_CHECK)
                pts = float(W.points(run.J[0], brec["anchors"][law]["anchor"]))
                summ = episode_summary(cmds, push, q_sw, qpi_sw, q_act, qpi_act)
                episodes.append([quality, kind, sit, sig4(run.J[0]), sig4(run.Jd[0]), sig4(pts), sig4(states),
                                 sig4(cmds), sig4(rewards), sig4(push), sig4(q_act), sig4(qpi_act), sig4(q_sw),
                                 sig4(qpi_sw), ints(q_sw.argmax(1)), ints(qpi_sw.argmax(1))])
                summaries.append([quality, kind, sit, round_summary(dict(summ, return_100_steps=float(run.J[0]),
                                                                         points=pts))])
                ep_check = dict(
                    quality=quality, critic=kind, situation=sit,
                    tail_sum_max_relative=rel1, tail_sum_max_absolute=abs1, tail_sum_ok=bool(rel1 <= TOL_REL),
                    sweep_reproduces_critic_max_abs=float(np.max(np.abs(q_own[:, 0] - q_act))),
                    sweep_reproduces_truth_max_abs=float(np.max(np.abs(qpi_own[:, 0] - qpi_act))),
                    sweep_reproduces_truth_max_relative=float(np.max(np.abs(qpi_own[:, 0] - qpi_act) / np.abs(qpi_act))),
                    horizon_600_vs_1000_max_relative=float(np.max(np.abs(qpi_long - qpi_act) / np.abs(qpi_act))),
                    float32_actor_vs_float64_copy_max_abs=float(np.max(np.abs(a32 - cmds))),
                    all_finite=bool(np.all(np.isfinite(q_sw)) and np.all(np.isfinite(qpi_sw))))
                ep_check["sweep_reproduces_ok"] = bool(ep_check["sweep_reproduces_critic_max_abs"] == 0.0
                                                       and ep_check["sweep_reproduces_truth_max_relative"] <= 1e-12)
                checks["episodes"].append(ep_check)
                timing[f"{quality}/{kind}/{sit}"] = round(time.perf_counter() - te, 1)
                log(f"{quality} {kind} {sit}: J {run.J[0]:.1f} ({pts:.1f} points); a-hat past 80% at "
                    f"{summ['a_hat_past80_states']} states, critic up 80->100 at "
                    f"{summ['critic_higher_at_100_than_80_states']}, truth down at "
                    f"{summ['truth_lower_at_100_than_80_states']}; tail check {rel1:.1e}; "
                    f"{timing[f'{quality}/{kind}/{sit}']}s")
                # context: the first ten starts of the bank
                cb = context_block(pol, state.critic.params, mean, std, banks[law][:CONTEXT_STARTS])
                cb["points_mean"] = float(W.points(cb["mean_return"], brec["anchors"][law]["anchor"]))
                context.append([quality, kind, sit, round_summary(cb)])
            checks["hosts"].append(host_check)
        timing[quality] = round(time.perf_counter() - tq, 1)

    passed = dict(
        critic_hashes=all(h["critic_sha256_ok"] and h["actor_sha256_ok"] and h["snapshot_file_ok"]
                          for h in checks["hosts"]),
        normalisation=all(h["normalisation_ok"] for h in checks["hosts"]),
        tail_sums=all(e["tail_sum_ok"] for e in checks["episodes"]),
        sweep_reproduces=all(e["sweep_reproduces_ok"] for e in checks["episodes"]),
        anchors=all(a["ok"] for a in checks["anchors"]),
        finite=all(e["all_finite"] for e in checks["episodes"]))
    checks["passed"] = passed
    checks["all_passed"] = all(passed.values())
    checks["worst"] = dict(
        tail_sum_max_relative=max(e["tail_sum_max_relative"] for e in checks["episodes"]),
        sweep_reproduces_critic_max_abs=max(e["sweep_reproduces_critic_max_abs"] for e in checks["episodes"]),
        sweep_reproduces_truth_max_relative=max(e["sweep_reproduces_truth_max_relative"] for e in checks["episodes"]),
        horizon_600_vs_1000_max_relative=max(e["horizon_600_vs_1000_max_relative"] for e in checks["episodes"]),
        float32_actor_vs_float64_copy_max_abs=max(e["float32_actor_vs_float64_copy_max_abs"]
                                                  for e in checks["episodes"]))
    checks["checks_explained"] = {
        "tail_sum": "Q^pi at the actor's command at state t equals the discounted sum (0.98) of the episode's own "
                    "rewards from t to step 99 plus 0.98^(100 - t) times the actor's own simulated tail from the "
                    "episode's last state (600 - (100 - t) steps); tolerance 1e-6 relative",
        "critic_hashes": "the loaded snapshot file's SHA-256 (checked by the package's loader), the critic's and the "
                         "actor's parameter hashes (learners.tree_sha256) equal record.json's final snapshot",
        "sweep_reproduces": "the sweep's code path, called at the actor's own thruster-1 command, gives the critic's Q "
                            "bit for bit and Q^pi within 1e-12 relative of the separately computed values at the "
                            "actor's command",
        "anchors": "the bank means of the stall-aware controller and of random commands, recomputed, equal "
                   "start_banks.json's anchors",
        "normalisation": "the observation mean and SD recomputed by data.normalization equal record.json's",
        "horizon_600_vs_1000": "reported only: Q^pi with horizon 600 against truth.HORIZON = 1,000",
        "float32_actor_vs_float64_copy": "reported only: the float32 network actor against its float64 copy that drives"}

    out = {
        "schema": "bca-shift-qviewer-v1",
        "note": [
            "Critic in action: ThrusterChain, version 5 of the importance-weighting experiment, TD3+BC (twin delayed "
            "deep deterministic policy gradient plus behaviour cloning) hosts of data seed 0.",
            "episodes: one array per (quality, critic, situation), fields in episode_fields order. Per-state arrays have "
            "100 rows, t = 0...99 (the state before the t-th command).",
            "state: the 13 numbers of the state, fields in state_fields order (a float's position is its distance "
            "from rest). cmd: the actor's 4 throttles (negative = upstream). reward: r = -(|s|^2 + 0.3 |a|^2). "
            "push: the water's push c + w1.",
            "q_actor / qpi_actor: the critic's Q(s, a) = min(Q1, Q2) and the exact true value Q^pi(s, a) at the actor's "
            "own command. q_sweep / qpi_sweep: the same at each of the 41 thruster-1 commands in sweep_a1 (0 to -1, "
            "i.e. 0% to 100% upstream, float32 values), thrusters 2-4 held at the actor's command.",
            "Q^pi: one step with the command, then the actor drives; horizon 600, discount 0.98, float64 simulation of "
            "the deterministic physics (truth.q_value_jax).",
            "a_hat: index into sweep_a1 of the command the critic rates highest (a-hat). a_best_true: index of the "
            "swept command with the highest Q^pi. i80 is the index of 80% (the knee; it does not stall); indices "
            "above it are past 80% (stall).",
            "return: the undiscounted 100-step return; discounted_return: with discount 0.98; points: 100 (J - J_random)"
            " / (J_stall_aware - J_random) with the start bank's mean anchors (situation_info).",
            "gap critic: trained with the densest-15% removal (almost every 80% hold of thruster 1 removed); full "
            "critic: trained on all the data.",
            "summaries: full-precision statistics per episode (rounded to 4 significant digits), notes in "
            "summary_notes. context: the 80%-versus-100% comparison on the first ten starts of each bank.",
            "All numbers rounded to 4 significant digits."],
        "episode_fields": ["quality", "critic", "situation", "return", "discounted_return", "points", "state", "cmd",
                           "reward", "push", "q_actor", "qpi_actor", "q_sweep", "qpi_sweep", "a_hat", "a_best_true"],
        "state_fields": STATE_FIELDS,
        "sweep_a1": sig4(SWEEP),
        "i80": I80,
        "qualities": list(QUALITIES),
        "critics": list(KINDS),
        "situations": [s for s, _ in SITUATIONS],
        "situation_info": {k: {kk: (sig4(vv) if isinstance(vv, float) else
                                    (sig4(vv) if isinstance(vv, list) and vv and isinstance(vv[0], float) else vv))
                               for kk, vv in v.items()} for k, v in start_info.items()},
        "meta": dict(seed=SEED, steps=STEPS, horizon=H, gamma=W.GAMMA, critic="min(Q1, Q2)",
                     start_rule=checks["start"]["rule"], start_index=int(j), spec_sha256=spec,
                     script_sha256=sha256_file(__file__), checks_all_passed=checks["all_passed"],
                     seconds=round(time.perf_counter() - t_start, 1)),
        "episodes": episodes,
        "summary_notes": SUMMARY_NOTES,
        "summaries": summaries,
        "context_fields_note": "per (quality, critic, situation): the first ten starts of the bank, every visited "
                               "state; shares are of states (1,000 per row)",
        "context": context,
    }
    checks["timing_seconds"] = timing
    checks["total_seconds"] = round(time.perf_counter() - t_start, 1)
    out["meta"]["seconds"] = checks["total_seconds"]
    text = json.dumps(out, separators=(",", ":"), allow_nan=False)
    (OUT / "q_data.json").write_text(text)
    (OUT / "q_checks.json").write_text(json.dumps(RUN.plain(checks), indent=1, allow_nan=False))
    for name in ("q_data.json", "q_checks.json"):
        shutil.copyfile(OUT / name, WIN / name)
    log(f"wrote q_data.json ({len(text) / 1e6:.2f} MB) and q_checks.json; all checks passed: {checks['all_passed']}; "
        f"{checks['total_seconds']}s")
    return 0 if checks["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
