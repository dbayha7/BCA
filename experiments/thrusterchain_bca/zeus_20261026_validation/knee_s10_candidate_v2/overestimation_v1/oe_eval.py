"""The `oe_evaluate` job (OVERESTIMATION_RULE.md, "Reads, per unit"): one 20k policy (native or BCA) of one seed.
Rollouts from the 128 common starts, own-action probes at 48 visited states, logged and withheld probes at 192
dataset rows, each against the policy's own exact unclipped continuation. Imported by oe_worker.py inside the
worker process (GPU); the pure helpers (`dataset_probe_rows`, `w_probes`) are tested on the CPU."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
for _p in (str(CAND.parent / "knee_edge_ood_v3"), str(CAND / "native_screen"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import stage1_eval as E                                     # noqa: E402
import qpi                                                  # noqa: E402
from core import require                                    # noqa: E402
from knee_region import KNEE, SLOTS, past_knee_a0           # noqa: E402
import oe_seeds as OS                                       # noqa: E402

WORLD, DATA_ARM, CHECKPOINT = "stall3", "cutoff", "020000"
W_SLOTS = ("s05", "s10", "s20")                             # withheld a0 magnitudes 0.85, 0.9, 1.0 (|float32 slot value|; the table is W-)
PROBE_NAMES = ("logged",) + W_SLOTS
N_ROWS = 192
IDENTITY_TOL = 1e-6                                         # amendment A1: I1 relative tolerance (was 1e-9)
ACTION_TOL = 1e-6                                           # amendment A1: I2 max |pi(s) - executed| (was bit equality)


def dataset_probe_rows(arrays, seed, n_rows=N_ROWS):
    """192 CUTOFF-retained fitting rows (split code 0, |a0| <= KNEE), without replacement, by the package seed.
    Returns exact float64 states, float32 observations, times, logged float32 actions and the (episode, t) rows."""
    split = np.asarray(arrays["split_codes"])
    actions = np.asarray(arrays["actions"], np.float32)
    episodes, horizon = actions.shape[:2]
    fitting = np.flatnonzero(split == 0)
    retained = ~past_knee_a0(actions.reshape(-1, 4))[0].reshape(episodes, horizon)
    candidates = [(int(e), int(t)) for e in fitting for t in range(horizon) if retained[e, t]]
    require(len(candidates) >= n_rows, "Too few retained fitting rows")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed))))
    picks = np.sort(rng.choice(len(candidates), n_rows, replace=False))
    rows = [candidates[k] for k in picks]
    states = np.stack([np.asarray(arrays["states"][e, t], np.float64) for e, t in rows])
    obs = np.stack([np.asarray(arrays["observations"][e, t], np.float32) for e, t in rows])
    times = np.asarray([t for _, t in rows], np.int64)
    logged = np.stack([actions[e, t] for e, t in rows])
    expected_obs = np.concatenate([states, ((E.HORIZON - times) / E.HORIZON)[:, None]], axis=1).astype(np.float32)
    require(np.array_equal(obs, expected_obs), "Dataset observation is not (state, remaining time fraction)")
    require(bool(np.all(np.abs(logged[:, 0].astype(np.float64)) <= KNEE)), "A probe row's logged a0 is withheld")
    return {"states": states, "obs": obs, "times": times, "logged": logged, "rows": rows}


def w_probes(logged):
    """(n, 4, 4) float32: slot 0 the logged command; slots 1..3 the logged command with a0 replaced by
    sign(a0) * {0.85, 0.9, 1.0} (a zero a0 takes the positive side); a1..a3 logged."""
    logged = np.asarray(logged, np.float32)
    n = len(logged)
    out = np.repeat(logged[:, None, :], 1 + len(W_SLOTS), axis=1).copy()
    sign = np.where(logged[:, 0] < 0, -1.0, 1.0).astype(np.float32)
    for k, name in enumerate(W_SLOTS, start=1):
        out[:, k, 0] = sign * np.float32(abs(SLOTS[name]))
    require(bool(np.all(np.abs(out[:, 1:, 0].astype(np.float64)) > KNEE)), "A W probe is not past the knee")
    require(bool(np.all(np.abs(out.astype(np.float64)) <= 1.0)), "Probe out of range")
    return out


def probe_values(policy, states, obs, times, probes, world, budget):
    """Q^pi (own unclipped continuation), Q1, Q2 and U at every (row, slot)."""
    n, k = probes.shape[:2]
    budget.charge(times, k)
    values, calls, receipt = qpi.raw_values_from_states(states, times, probes, policy.act, world, clip_all=False)
    q1, q2, _ = E.critic(policy, obs, probes)
    u = np.asarray(policy.width(np.repeat(np.asarray(obs, np.float32), k, axis=0), probes.reshape(-1, 4)), np.float64).reshape(n, k)
    require(np.isfinite(values).all() and np.isfinite(u).all(), "Non-finite probe value")
    return {"qpi": values, "q1": q1, "q2": q2, "u": u, "forward_calls": int(calls), "receipt": receipt}


def measure(policy, arrays, bank, seeds, budget, deadline):
    """Every read of one policy (OVERESTIMATION_RULE.md "Reads, per unit"), given the dataset arrays, the 128 start
    states and the package seeds. Pure simulator and policy calls; tested on the CPU with a fake policy."""
    require(bank.shape == (128, 13), "The E2 bank is the formal G2 v2's 128 starts")
    roll = E.rollouts(bank, policy.act, WORLD, "pi", budget)
    rates = E.step_rates(roll["executed"], roll["obs"])
    deadline()
    states, obs, times, rows = E.visited_states(roll, seeds["visited_states"])
    own = policy.act(obs)
    executed_there = np.stack([roll["executed"][e, t] for e, t in rows])
    i2_max_abs = float(np.max(np.abs(own.astype(np.float64) - executed_there.astype(np.float64))))
    i2_ok = bool(i2_max_abs <= ACTION_TOL)
    own_vals = probe_values(policy, states, obs, times, own[:, None, :], WORLD, budget)
    t0 = times == 0
    starts = np.asarray([e for e, t in rows if t == 0])
    i1_rel = float(np.max(np.abs(own_vals["qpi"][t0, 0] - roll["returns"][starts]) / np.maximum(1.0, np.abs(roll["returns"][starts]))))
    deadline()
    ds = dataset_probe_rows(arrays, seeds["dataset_rows"])
    probes = w_probes(ds["logged"])
    d_vals = probe_values(policy, ds["states"], ds["obs"], ds["times"], probes, WORLD, budget)
    return {"returns": roll["returns"], "J": float(np.mean(roll["returns"])), "rates": rates, "exposure_w": rates["w_any_depth"],
            "rollout_receipt": roll["receipt"],
            "own": {"rows": rows, "times": times, "actions": own, "executed": executed_there, "qpi": own_vals["qpi"][:, 0], "q1": own_vals["q1"][:, 0],
                    "q2": own_vals["q2"][:, 0], "u": own_vals["u"][:, 0], "forward_calls": own_vals["forward_calls"],
                    "receipt": own_vals["receipt"]},
            "instrument": {"I1_max_rel": i1_rel, "I1_tol": IDENTITY_TOL, "I1_ok": bool(i1_rel <= IDENTITY_TOL),
                           "I2_max_abs": i2_max_abs, "I2_tol": ACTION_TOL, "I2_ok": i2_ok},
            "data": {"rows": ds["rows"], "times": ds["times"], "slots": list(PROBE_NAMES), "logged": ds["logged"], "probes": probes,
                     "qpi": d_vals["qpi"], "q1": d_vals["q1"], "q2": d_vals["q2"], "u": d_vals["u"],
                     "forward_calls": d_vals["forward_calls"], "receipt": d_vals["receipt"]},
            "budget_used": int(budget.used)}


def oe_evaluate(job, out, deadline):
    import screen_worker as SW                              # the worker process has imported it already
    W1 = SW.W1
    require(job.get("arm") in ("native", "bca") and job.get("world", WORLD) == WORLD, "oe_evaluate: arm native|bca, world stall3")
    runtime, evaluate_worker, _, _ = SW.rerun1()
    budget = W1.Budget(job["transitions"], job["timeout_seconds"])
    seeds = OS.load()["numpy"]
    dataset, plan = W1.load_world(job, WORLD)
    dirs = {"warmup": job["runs"]["warmup"], "branch": job["runs"]["branch"]}
    restore = W1.policies if job["arm"] == "native" else SW.policies_bca
    _, pols, assertions, adapter = restore(job, runtime, evaluate_worker, dataset, plan, DATA_ARM, dirs, (CHECKPOINT,))
    policy = pols[CHECKPOINT]
    deadline()
    reads = measure(policy, dataset.arrays, E.bank_states(W1.E2_BANK), seeds, budget, deadline)
    policy.assert_unchanged()
    return {"kind": "oe_evaluate", "package": "knee_s10_candidate_v2/overestimation_v1", "arm": job["arm"], "kappa": SW.KAPPA,
            "seed_index": job["seed_index"], "checkpoint": CHECKPOINT, "world": WORLD, "assertions": assertions,
            "dataset_sha256": dataset.digest, "seeds": dict(seeds), **reads}
