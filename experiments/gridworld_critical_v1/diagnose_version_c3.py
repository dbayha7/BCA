"""An instrumented run of Version C3 (PROTOCOL.md, Amendment 9). A measurement: no new method.

    python -B diagnose_version_c3.py --label diag_version_c3 --baselines published_validation_2 --seeds 6401 6402 6403

Version C3 is trained exactly as run_version_c3_published.py trains it. After each refresh the
extra read-outs of Amendment 9 are taken on the held-out rows and written to drift.json. The
read-outs use no random numbers of the run and change nothing in it.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import run_version_c3_published as C3  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import critical_env as E  # noqa: E402
import learner as L  # noqa: E402
import version_c as VC  # noqa: E402

METHOD = "version_c3"
RC, C2 = C3.RC, C3.C2
RECORDS = []


def make_probe():
    obs_fn, obs_dim = L.OBSERVATIONS["one_hot"]
    cal = L.calibrator(obs_dim)

    def probe(state, bank, key):
        reference = state["reference"]
        obs, next_obs = obs_fn(bank["s"]), obs_fn(bank["s2"])
        noise = jax.random.normal(key, bank["a"].shape)               # the noise the refresh's own target used
        noisy = L.smoothed_action(state["actor_t"], next_obs, noise)
        clean = L.actor_apply(state["actor_t"], next_obs)

        def both(action, where):
            return jnp.minimum(L.critic_apply(state["q1_t"], where, action), L.critic_apply(state["q2_t"], where, action))

        compared = VC.comparison_commands(jax.random.fold_in(key, 7), bank["a"])
        return {"flagged_noisy": VC.cannot_certify(reference, next_obs, noisy),
                "flagged_clean": VC.cannot_certify(reference, next_obs, clean),
                "charge_noisy": VC.charge(cal, reference, next_obs, noisy, capped=True),
                "charge_clean": VC.charge(cal, reference, next_obs, clean, capped=True),
                "q_noisy": both(noisy, next_obs), "q_clean": both(clean, next_obs),
                "q_recorded": jnp.minimum(L.critic_apply(state["q1"], obs, bank["a"]),
                                          L.critic_apply(state["q2"], obs, bank["a"])),
                "flagged_compared": VC.cannot_certify(reference, obs, compared),
                "q_compared": L.critic_apply(state["q1"], obs, compared),
                "commands": L.actor_apply(state["actor"], obs_fn(jnp.arange(100, dtype=jnp.int32)))}

    return jax.jit(jax.vmap(probe, in_axes=(0, 0, 0)))


PROBE = None


def refresh(states, banks, keys, runs, step, heldout):
    """Version C2's refresh (the cap held), then the read-outs of Amendment 9 on the same rows."""
    global PROBE
    states, events = C2.refresh(states, banks, keys, runs, step, heldout)
    if PROBE is None:
        PROBE = make_probe()
    refresh_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1_000_000 + step))(keys)
    out = {name: np.asarray(value) for name, value in PROBE(states, banks, refresh_keys).items()}
    task = E.published()
    s, s2 = np.asarray(banks["s"]), np.asarray(banks["s2"])
    recorded = np.asarray(banks["a"]).argmax(axis=2)
    alive = np.asarray(banks["done"]) == 0
    for i, run in enumerate(runs):
        value = task.value(out["commands"][i].astype(np.float64))        # exact value of the current policy
        true_q = task.command_values(value)
        live = alive[i]
        RECORDS.append({
            "recipe": run["recipe"], "data_seed": run["data_seed"], "train_seed": run["train_seed"], "step": step,
            "cap": float(np.asarray(states["reference"]["width_cap"])[i]), "unit": float(np.asarray(states["unit"])[i]),
            "next_commands_flagged_with_noise": float(out["flagged_noisy"][i][live].mean()),
            "next_commands_flagged_without_noise": float(out["flagged_clean"][i][live].mean()),
            "charge_at_next_command_with_noise": float(out["charge_noisy"][i][live].mean()),
            "charge_at_next_command_without_noise": float(out["charge_clean"][i][live].mean()),
            "score_with_noise_minus_without": float((out["q_noisy"][i] - out["q_clean"][i])[live].mean()),
            "comparison_commands_flagged": float(out["flagged_compared"][i].mean()),
            "score_at_flagged_comparison_commands": (float(out["q_compared"][i][out["flagged_compared"][i]].mean())
                                                     if out["flagged_compared"][i].any() else None),
            "error_at_recorded_commands": float((out["q_recorded"][i] - true_q[s[i], recorded[i]]).mean()),
            "error_at_next_command_without_noise": float((out["q_clean"][i] - value[s2[i]])[live].mean()),
            "error_at_next_command_with_noise": float((out["q_noisy"][i] - value[s2[i]])[live].mean()),
            "true_value_at_recorded_commands": float(true_q[s[i], recorded[i]].mean())})
    return states, events


if __name__ == "__main__":
    VC.target_c = C3.plain_target
    RC.refresh = refresh
    RC.METHOD = METHOD
    RC.SOURCES = RC.SOURCES + ("run_version_c2_published.py", "run_version_c3_published.py", "diagnose_version_c3.py")
    RC.main()
    label = sys.argv[sys.argv.index("--label") + 1]
    (HERE / "results" / label / "drift.json").write_text(json.dumps(RECORDS, indent=1))
    print("read-outs written", flush=True)
