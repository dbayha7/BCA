"""Authorized Path A child process. Nothing scientific executes on import."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch as L


class PhysicalCounter:
    CAPS = {"returns": 153600, "own_qpi": 57600, "common_reference": 21600, "prefix_parity": 4800}

    def __init__(self, plant, deadline, progress):
        self.plant, self.deadline, self.progress = plant, deadline, progress
        self.attempted = dict.fromkeys(self.CAPS, 0)
        self.completed = dict.fromkeys(self.CAPS, 0)
        self.phase = None

    def begin(self, phase):
        L.require(phase in self.CAPS, "Unknown evaluation phase")
        self.deadline()
        self.phase = phase
        self.progress(self.receipt())

    def __call__(self, state, action):
        self.deadline()
        L.require(self.phase in self.CAPS, "Uncounted simulator phase")
        L.require(state.shape == (13,) and action.shape == (4,), "One complete state/action call required")
        L.require(self.attempted[self.phase] < self.CAPS[self.phase] and sum(self.attempted.values()) < 237600,
                  "Evaluation physical cap reached")
        self.attempted[self.phase] += 1
        value = self.plant(state, action)
        self.completed[self.phase] += 1
        if sum(self.completed.values()) % 1000 == 0:
            self.progress(self.receipt())
        return value

    def receipt(self):
        return {"attempted": dict(self.attempted), "completed": dict(self.completed),
                "execution_counts_known": self.attempted == self.completed,
                "critic_updates": 0, "actor_updates": 0, "auxiliary_updates": 0, "calibrations": 0}


def callbacks(runtime, adapter, state, identity):
    import numpy as np
    jax = adapter.context.native.jax
    agent, reference = state.microstep.agent_state, state.aux.posterior
    actor = jax.jit(lambda obs: agent.actor.apply_fn(agent.actor.params, obs).sample(seed=None))
    critic = jax.jit(lambda obs, act: agent.dual_q.apply_fn(agent.dual_q.params, obs, act)[..., 0])
    def raw_width(obs, act):
        value, usable, valid = adapter.context.pipeline.live_width(adapter.context.models, reference, obs, act)
        return value, usable, valid
    width = jax.jit(raw_width)
    before = runtime.tree_digest(state)
    def widths(obs, act):
        value, usable, valid = width(obs, act)
        L.require(np.all(np.asarray(usable)) and np.all(np.asarray(valid)), "Invalid frozen uncertainty readout")
        return np.asarray(value)
    def unchanged():
        L.require(before == runtime.tree_digest(state), "Evaluation changed a trained checkpoint")
    return {"actor": lambda obs: np.asarray(actor(obs), np.float32),
            "q1": lambda obs, act: np.asarray(critic(obs, act)), "width": widths,
            "identity": identity, "assert_unchanged": unchanged}


def evaluate_job(protocol, output, deadline, binding, training, services, review):
    import numpy as np
    evaluator = L.load("_path_a_evaluate", HERE / "evaluate.py")
    core = L.PAYLOAD / "transfer_bundle/core_v4"
    sys.path.insert(0, str(core))
    from thrusterchain import environment as canonical
    L.require(Path(canonical.__file__).resolve() == core / "thrusterchain/environment.py", "Wrong canonical plant")
    physics = canonical.Physics(stall=True, stall_slope=3.)
    source_result = L.read(L.V2 / "setup_attempt_1/RESULT.json")
    L.require(physics.as_dict() == source_result["physics"] and canonical.provenance() == source_result["runtime"]["canonical"],
              "Saved task and evaluation physics differ")
    def progress(receipt):
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, sort_keys=True) + "\n")
    counter = PhysicalCounter(lambda x, a: canonical.dynamics(x, a, physics), deadline, progress)
    geometry = L.read(L.V2 / "setup_attempt_1/geometry.json")
    task = binding.D.T.Task(counter, geometry)
    A, B = canonical.matrices(physics)
    controller = binding.D.T.tracking_controller(A, B, protocol["geometry_recipe"])
    policies = {}
    try:
        for condition in protocol["conditions"]:
            for seed in range(3):
                for arm in ("native", "bca"):
                    deadline()
                    branch = output.parent / f"{condition}_s{seed}_{arm}"
                    warmup = output.parent / f"{condition}_s{seed}_warmup"
                    L.accepted_job(warmup, 0)
                    L.accepted_job(branch, 0)
                    adapter, state, restored_native, restored_runtime = training.restore_final(
                        protocol, condition, seed, arm, branch, warmup, services=services, training_review=review)
                    L.require(restored_native is services.native and restored_runtime is services.runtime,
                              "Restoration changed numerical services")
                    identity = {"checkpoint_sha256": L.sha(branch / "full_state.msgpack"),
                        "normalization_sha256": binding.I.digest(services.runtime.host(adapter.context.normalization)),
                        "dataset_sha256": binding.DATASET_SHA256, "condition": condition, "seed_index": seed,
                        "arm": arm, "cumulative_critic_updates": 20000, "training_kappa": 3 if arm == "bca" else 0,
                        "checkpoint_receipt_sha256": L.sha(branch / "checkpoint.json")}
                    policies[(seed, condition, arm)] = callbacks(services.runtime, adapter, state, identity)
        report = evaluator.evaluate_study(task, controller, geometry, protocol, policies, counter,
                                         lambda name, value: L.save(output / name, value))
        L.require(report["status"] == "completed_frozen_evaluation" and
                  report["completed_policy_evaluations"] == 12 and report["completed_paired_cells"] == 6 and
                  report["completed_return_episodes"] == 1536, "Final evaluation incomplete")
        for policy in policies.values():
            policy["assert_unchanged"]()
        report["physical_counts"] = counter.receipt()
        report["runtime"] = services.runtime.runtime_receipt(services.native)
        L.require(counter.receipt()["execution_counts_known"], "Unknown physical work")
        L.save(output / "report.json", report)
        return report
    except BaseException as exc:
        L.save(output / "evaluation_failure.json", {"error": type(exc).__name__ + ": " + str(exc),
               "counts": counter.receipt(), "traceback": traceback.format_exc(), "no_automatic_retry": True})
        raise


def run(job_id):
    L.require(os.name == "posix", "Manual existing WSL runtime required")
    started = time.monotonic()
    protocol = L.check(approval=True)
    expected = next((j for j in L.jobs() if j["id"] == job_id), None)
    L.require(expected is not None, "Unknown frozen job")
    output = HERE / "run_1" / job_id
    job = L.read(output / "JOB.json")
    L.require(all(job[k] == v for k, v in expected.items()) and job["manifest_sha256"] == L.sha(HERE / "MANIFEST.json"),
              "Job not bound to the current one-shot reservation")
    L.require(L.read(HERE / "run_1/ATTEMPT.json")["manifest_sha256"] == job["manifest_sha256"], "Wrong attempt")
    end = started + min(float(job["remaining_seconds"]), protocol["internal_process_seconds"])
    def deadline():
        if time.monotonic() >= end:
            raise TimeoutError("Global remaining execution time exhausted")
    receipt = {"status": "failed", "identity_unchanged": False, "error": None, "no_automatic_retry": True}
    try:
        sys.path.insert(0, str(L.REPO))
        binding = L.load("_path_a_binding", HERE / "binding.py")
        training = L.load("_path_a_train", HERE / "train.py")
        services = training.load_runtime(binding)
        review = {"training_authorized": True, "protocol_sha256": binding.I.digest(protocol),
                  "dataset_sha256": binding.DATASET_SHA256, "geometry_sha256": binding.GEOMETRY_SHA256,
                  "approval_sha256": L.sha(HERE / "LAUNCH_APPROVAL.json")}
        if job["arm"] == "evaluation":
            report = evaluate_job(protocol, output, deadline, binding, training, services, review)
        else:
            warmup = None if job["arm"] == "warmup" else output.parent / f"{job['condition']}_s{job['seed_index']}_warmup"
            if warmup is not None:
                L.accepted_job(warmup, 0)
            report = training.run_job(protocol, job["condition"], job["seed_index"], job["arm"], output,
                                      warmup_dir=warmup, deadline=deadline, services=services, training_review=review)
            L.require(report["status"] == "completed_training_job" and report["execution_counts_known"],
                      "Training job did not complete with known work")
            expected_counts = {"critic": 10000, "actor": 5000, "auxiliary": 10000, "calibrations": 1}
            L.require(report["counts"]["attempted"] == expected_counts and report["counts"]["completed"] == expected_counts,
                      "Per-job physical update budget differs")
        L.check(approval=True)
        receipt.update(status="completed", identity_unchanged=True, report_sha256=L.sha(output / "report.json"))
    except BaseException as exc:
        receipt.update(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
    receipt["elapsed_seconds"] = time.monotonic() - started
    L.save(output / "WORKER_EXIT.json", receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.job)["status"] == "completed" else 1)
