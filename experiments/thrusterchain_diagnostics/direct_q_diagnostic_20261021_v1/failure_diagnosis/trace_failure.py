"""One authorized304-transition replay trace; never evaluates the main query bank."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
V1 = HERE.parent
sys.path.insert(0, str(V1))
import diagnostic as D


def run():
    output = HERE / "trace_run01"
    output.mkdir(exist_ok=False)
    frozen_files = [V1/name for name in ("diagnostic.py", "run_diagnostic.py", "protocol.json", "freeze.json",
        "prepared/preparation_manifest.json", "results/status.json", "results/execution_receipt.json")]
    preserved = {p.relative_to(V1).as_posix(): D.digest(p) for p in frozen_files}
    D.write_json(output/"execution_receipt.json", dict(started_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=D.digest(__file__), preserved_v1_sha256=preserved,
        authorized_additional_transitions=304, planned_main_returns=0,
        operation="same first four batched forced-Q paths and four scalar forced-Q paths; callbacks/dynamics logged without extra calls"))
    engine = D.load_engine()
    import numpy as np
    before = D.parameter_hashes(engine)
    C, E = engine.C, engine.eval_module
    original_c, original_e = C.dynamics, E.dynamics
    states = np.zeros((4, 13)); states[:, 10] = [.3, 1.1, .3, 1.1]; states[:, 11] = .2
    regimes, starts = np.array([0, 0, 1, 1]), np.array([0, 50, 99, 99])
    actions = np.array([[.3, -.4, .8, -.2], [1, -1, .5, -.5], [0, 0, 0, 0], [1, -.8, .2, -.4]], np.float32)
    plant, policy = {"batch": [], "scalar": []}, {"batch": [], "scalar": []}
    counts = {"batch": 0, "scalar": 0}
    expected = [(t, g, np.flatnonzero((starts <= t) & (regimes == g)))
                for t in range(100) for g in (0, 1) if np.any((starts <= t) & (regimes == g))]
    cursor, scalar_fixture, scalar_time = 0, None, None

    def batch_policy(obs):
        obs = np.asarray(obs)
        times = np.rint(100*(1-obs[:, -1])).astype(int)
        D.require(np.all(times == times[0]), "unexpected mixed-time callback")
        t = int(times[0]); ids = np.flatnonzero(starts < t)
        D.require(len(ids) == len(obs), "unexpected callback row mapping")
        out = engine.result.policy(obs)
        for i, o, a in zip(ids, obs, out):
            policy["batch"].append(dict(fixture=int(i), time=t, observation=o.copy(), action=np.asarray(a).copy(), batch_size=len(obs)))
        return out

    def batch_dynamics(state, action, regime):
        nonlocal cursor
        t, g, ids = expected[cursor]; cursor += 1
        D.require(g == regime and len(ids) == len(state), "unexpected batched plant call mapping")
        nxt, reward = original_e(state, action, regime)
        counts["batch"] += len(ids)
        for i, s, a, n, r in zip(ids, state, action, nxt, reward):
            plant["batch"].append(dict(fixture=int(i), time=t, state=np.asarray(s).copy(), action=np.asarray(a).copy(), next_state=np.asarray(n).copy(), reward=float(r)))
        return nxt, reward

    def scalar_policy(obs):
        out = engine.result.policy(obs[None, :])[0]
        t = int(np.rint(100*(1-obs[-1])))
        policy["scalar"].append(dict(fixture=int(scalar_fixture), time=t, observation=obs.copy(), action=np.asarray(out).copy(), batch_size=1))
        return out

    def scalar_dynamics(state, action, regime):
        nonlocal scalar_time
        nxt, reward = original_c(state, action, regime)
        counts["scalar"] += 1
        plant["scalar"].append(dict(fixture=int(scalar_fixture), time=int(scalar_time), state=np.asarray(state).copy(), action=np.asarray(action).copy(), next_state=np.asarray(nxt).copy(), reward=float(reward)))
        scalar_time += 1
        return nxt, reward

    result, caught = {}, None
    try:
        E.dynamics = batch_dynamics
        batched = E.batch_policy_q(batch_policy, states, regimes, starts, actions, objective=C.Objective(), float32_deployment=True)
        E.dynamics = original_e
        C.dynamics = scalar_dynamics
        scalar = []
        for i in range(4):
            scalar_fixture, scalar_time = i, int(starts[i])
            scalar.append(C.policy_q(states[i], int(regimes[i]), actions[i], scalar_policy, C.Objective(), t=int(starts[i]), float32_deployment=True))
        scalar = np.asarray(scalar)
        D.require(counts == {"batch": 152, "scalar": 152}, "trace did not use exactly304 plant transitions")
        arrays = dict(fixture_initial_states=states, fixture_regimes=regimes, fixture_start_times=starts, fixture_forced_actions=actions,
                      batch_returns=batched, scalar_returns=scalar)
        for path in ("batch", "scalar"):
            for name, records in (("plant", plant[path]), ("policy", policy[path])):
                for field in records[0]:
                    arrays[path+"_"+name+"_"+field] = np.asarray([r[field] for r in records])
        with (output/"complete_trace.npz").open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        pb = {(v["fixture"], v["time"]): v for v in plant["batch"]}
        ps = {(v["fixture"], v["time"]): v for v in plant["scalar"]}
        ab = {(v["fixture"], v["time"]): v for v in policy["batch"]}
        ass = {(v["fixture"], v["time"]): v for v in policy["scalar"]}
        comparisons = []
        for key in sorted(pb):
            b, s = pb[key], ps[key]
            row = dict(fixture=key[0], time=key[1], forced_action=key not in ab,
                pre_state_exactly_equal=bool(np.array_equal(b["state"], s["state"])),
                pre_state_max_abs_difference=float(np.max(np.abs(b["state"]-s["state"]))),
                command_exactly_equal=bool(np.array_equal(b["action"], s["action"])),
                command_max_abs_difference=float(np.max(np.abs(b["action"]-s["action"]))),
                post_state_max_abs_difference=float(np.max(np.abs(b["next_state"]-s["next_state"]))),
                reward_difference=float(b["reward"]-s["reward"]),
                observation_exactly_equal=None, observation_max_abs_difference=None,
                actor_output_exactly_equal=None, actor_output_max_abs_difference=None,
                batch_path_actor_batch_size=None, scalar_path_actor_batch_size=None)
            if key in ab:
                ba, sa = ab[key], ass[key]
                row.update(observation_exactly_equal=bool(np.array_equal(ba["observation"], sa["observation"])),
                    observation_max_abs_difference=float(np.max(np.abs(ba["observation"].astype(float)-sa["observation"].astype(float)))),
                    actor_output_exactly_equal=bool(np.array_equal(ba["action"], sa["action"])),
                    actor_output_max_abs_difference=float(np.max(np.abs(ba["action"].astype(float)-sa["action"].astype(float)))),
                    batch_path_actor_batch_size=ba["batch_size"], scalar_path_actor_batch_size=sa["batch_size"])
            comparisons.append(row)
        per_fixture = []
        for i in range(4):
            rows = [r for r in comparisons if r["fixture"] == i]
            first = lambda predicate: next((r for r in rows if predicate(r)), None)
            per_fixture.append(dict(fixture=i, start_time=int(starts[i]), batch_return=float(batched[i]), scalar_return=float(scalar[i]),
                return_difference=float(batched[i]-scalar[i]),
                first_pre_state_difference=first(lambda r: not r["pre_state_exactly_equal"]),
                first_command_difference=first(lambda r: not r["command_exactly_equal"]),
                first_identical_observation_different_actor_output=first(lambda r: r.get("observation_exactly_equal") and not r["actor_output_exactly_equal"]),
                max_command_difference=max(r["command_max_abs_difference"] for r in rows),
                max_pre_state_difference=max(r["pre_state_max_abs_difference"] for r in rows)))
        result = dict(status="complete", fixtures=per_fixture, additional_transition_counts=counts,
            additional_transitions=sum(counts.values()), main_returns_evaluated=0,
            original_failed_run=dict(status_sha256=preserved["results/status.json"], recorded_fixture_transitions=0,
                actual_fixture_transitions_before_assertion=304, main_returns_evaluated=0,
                stale_counter_reason="v1 status receives fixture count only after validate_truth returns; assertion interrupted its return"),
            identical_float32_observations_with_different_actor_outputs=sum(bool(r.get("observation_exactly_equal")) and not r.get("actor_output_exactly_equal", True) for r in comparisons),
            comparison_rows=len(comparisons), policy_callback_rows={k: len(v) for k, v in policy.items()},
            interpretation="Inspect first identical-observation/different-action row to distinguish batch-dependent actor arithmetic from later closed-loop feedback; no policy or tolerance changed")
        D.write_csv(output/"aligned_trace.csv", comparisons)
    except Exception as error:
        caught = error
        result = dict(status="failed", error=repr(error), traceback=traceback.format_exc(), additional_transition_counts=counts,
            additional_transitions=sum(counts.values()), main_returns_evaluated=0)
    finally:
        C.dynamics, E.dynamics = original_c, original_e
        after = D.parameter_hashes(engine)
        current = {p.relative_to(V1).as_posix(): D.digest(p) for p in frozen_files}
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(), parameters_before=before, parameters_after=after,
            frozen_parameters_unchanged=before == after, preserved_v1_sha256=preserved,
            preserved_v1_files_unchanged=current == preserved, trace_script_sha256=D.digest(__file__),
            source_closure=engine.metadata["source_closure"], source_sha256=engine.metadata["source_sha256"],
            artifacts_sha256={p.name: D.digest(p) for p in sorted(output.iterdir())})
        D.write_json(output/"failure_anatomy.json", result)
    if caught is not None: raise caught
    D.require(before == after and current == preserved, "unexpected frozen artifact or parameter mutation")
    print(json.dumps({key: result[key] for key in ("status", "fixtures", "additional_transitions", "identical_float32_observations_with_different_actor_outputs", "frozen_parameters_unchanged", "preserved_v1_files_unchanged")}, indent=2))


if __name__ == "__main__":
    run()
