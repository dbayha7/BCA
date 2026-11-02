"""Stage 2 GPU jobs (approval A5), one per process, dispatched by stage2.py under its caps. User-run in the recorded
WSL GPU environment; agents never run this file and never open its checkpoints.

Every adapter is built by bindings.create (the 4.4 assertions) through the Stage 1 worker's helpers, and every fork
is the authenticated Stage 1 10k warmup (exact restore, matching receipt checked):
- precheck: BCA's width U from each stall fork's 10k shadow auxiliary at the 64 `d12_e2_anchors` (48 gated), at the
  near-edge and past-knee slots with a1..a3 = 0 (plus the actor's own a1..a3, reported), Q1 at the same probes, and
  the magnitude control U(a1 = +-0.69999999) against U(a1 = 0) at a0 = 0. No branch is trained before it.
- branch: runtime.branch(fork, view, seed, "bca" | "distance" | "native", the warmup's matched coefficient), where
  the view's seeds[seed_index] holds the fork's own Stage 2 branch seeds (seeds.stage2_protocol_seeds), so the three
  branches of a fork share seeds and minibatches; 10k updates (calibration at 15k), audits at 15k and 20k. A native
  replicate's phase-0 minibatch digest must differ from its fork's Stage 1 native branch.
- evaluate: for one seed, every Stage 2 branch and the fork's untrained actor on the fresh Stage 2 banks; KC_s10 of
  Q1 - 3U (BCA) and of Q1 (replicate) at the 64 `stage2_e2_anchors` in stall CUTOFF; the Stage 1 native branches
  re-evaluated on the Stage 2 banks.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import integrity
integrity.verify(sys.argv, __file__)          # every file this job loads, before any package import (PROTOCOL_V2.md change 6)

import numpy as np

import bindings as B
import probes as P
import seeds as S
import stage1_eval as E
import stage1_worker as W1
import step03
from core import require, sha_bytes, sha_file
from knee_region import GATED_PAST_KNEE, SLOTS

FORKS = {"stall_cutoff": ("cutoff", "stall3", ("bca", "distance", "native")),
         "stall_random": ("random", "stall3", ("bca", "native")),
         "nostall_cutoff": ("cutoff", "nostall", ("bca", "native"))}
DETECT_SLOTS = ("edge_knee", "edge_775", "edge_750", "edge_725", "edge_700") + tuple(GATED_PAST_KNEE)
D12, E2_BANK, E1_BANK, E2_ANCHORS = "d12_e2_anchors", "stage2_e2_bank", "stage2_e1_bank", "stage2_e2_anchors"


def stage2_view(seed_index, data_arm, world):
    return {"seeds": S.stage2_protocol_seeds(S.load(), seed_index, data_arm, world)}


def fork_job(job, seed_index, data_arm, world):
    """The Stage 1 helpers read job['arm'] as the data arm and job['seed_index']; research provenance only."""
    require(job.get("provenance", "research") == "research", "Stage 2 runs on research data only")
    return {**job, "seed_index": seed_index, "arm": data_arm, "world": world, "provenance": "research"}


def width_and_q(policy, obs, probes_array):
    n, slots = probes_array.shape[:2]
    rows = np.repeat(np.asarray(obs, np.float32), slots, axis=0)
    u = np.asarray(policy.width(rows, probes_array.reshape(-1, 4)), np.float64).reshape(n, slots)
    q1 = E.critic(policy, obs, probes_array)[0]
    return u, q1


# ------------------------------------------------------------------------------------------------ jobs

def precheck(job, out, deadline):
    runtime, evaluate_worker, _, _ = step03.rerun1()
    anchors = [E.build_anchor(D12, k) for k in range(S.BANKS[D12]["count"])]
    obs = np.stack([a["obs"] for a in anchors])
    gated = np.asarray([a["t"] in (0, 25, 50) for a in anchors])
    result = {"kind": "precheck", "slots": list(DETECT_SLOTS), "gated": gated, "seeds": {}}
    dataset, plan = W1.load_world(job, "stall3")
    for i in (0, 1, 2):
        result["seeds"][i] = {}
        for data_arm in ("cutoff", "random"):
            deadline()
            sub = fork_job(job, i, data_arm, "stall3")
            adapter, state, warm, assertions, _, _, _ = W1.restore_warm(runtime, sub, dataset, plan,
                                                                         job["stage1_runs"][f"warmup_s{i}_{data_arm}_stall3"])
            policy = evaluate_worker.FrozenPolicy(runtime, adapter, state)
            probes = P.build(len(anchors), DETECT_SLOTS)
            u, q1 = width_and_q(policy, obs, probes)
            own = policy.act(obs)
            E.classify(own)
            own_u, _ = width_and_q(policy, obs, P.build(len(anchors), DETECT_SLOTS, a123=own[:, 1:], kind="actor_own"))
            magnitude = np.zeros((len(anchors), 3, 4), np.float32)
            magnitude[..., 0] = np.float32(SLOTS["zero"])
            magnitude[:, :, 1] = np.asarray([-SLOTS["edge_700"], SLOTS["edge_700"], 0.0], np.float32)
            mag_u, _ = width_and_q(policy, obs, magnitude)
            policy.assert_unchanged()
            result["seeds"][i][data_arm] = {"assertions": assertions, "U": u, "Q1": q1, "own_U": own_u,
                                            "own_a123_past_knee": E.a123_past(own),
                                            "magnitude_U": {"a1_plus_0.7": mag_u[:, 0], "a1_minus_0.7": mag_u[:, 1], "a1_zero": mag_u[:, 2]},
                                            "warmup_report_sha256": sha_file(Path(job["stage1_runs"][f"warmup_s{i}_{data_arm}_stall3"]) / "report.json")}
    return result


def branch(job, out, deadline):
    runtime, _, gradient_audit, _ = step03.rerun1()
    i, data_arm, world, arm = job["seed_index"], job["data_arm"], job["world"], job["branch_arm"]
    require(arm in ("bca", "distance", "native"), "Unknown Stage 2 branch arm")
    sub = fork_job(job, i, data_arm, world)
    dataset, plan = W1.load_world(sub, world)
    adapter, state, warm, assertions, expected, probe, _ = W1.restore_warm(runtime, sub, dataset, plan, job["warmup_dir"])
    anchor = state.microstep.agent_state.actor.params
    view = stage2_view(i, data_arm, world)
    adapter, state = runtime.branch(adapter, state, view, i, arm, warm["matching"]["distance_coefficient"])
    branch_assertions = B.assert_adapter(adapter, expected)
    audits = {}

    def audit(current, step):
        result = gradient_audit.audit(adapter, current, probe, anchor_params=anchor if arm == "distance" else None)
        artifact = W1.write_json(out / f"gradient_audit_{step:06d}.json", runtime.host(result))
        audits[str(step)] = {**artifact, "summary": W1.audit_summary(result)}

    def on_step(current, metrics):
        if int(metrics["counts"]["critic"]) == 15_000:
            audit(current, 15_000)

    state, metrics, minibatches = W1.train(runtime, adapter, state, W1.BRANCH_UPDATES, out, deadline, job, on_step)
    final = runtime.validate(adapter, state)
    require(final["counts"]["critic"] == 20_000 and final["counts"]["refresh_accepted"] == 2, "Branch counts differ")
    audit(state, 20_000)
    stage1 = json.loads((Path(job["stage1_branch_dir"]) / "report.json").read_text(encoding="utf-8"))
    distinct = minibatches["phase0_minibatch_sha256"] != stage1["minibatches"]["phase0_minibatch_sha256"]
    if arm == "native":                       # the replicate must not be a copy of the Stage 1 branch (protocol)
        require(distinct, "Native replicate repeats its fork's Stage 1 native minibatch stream")
    checkpoint = W1.save_state(runtime, out, "020000", state)
    return {"kind": "branch", "fork": {"seed_index": i, "data_arm": data_arm, "world": world}, "branch_arm": arm,
            "kappa": adapter.kappa, "stage2_seeds": view["seeds"][i], "binding": expected, "assertions": assertions,
            "branch_assertions": branch_assertions, "distance_coefficient": warm["matching"]["distance_coefficient"],
            "counts": runtime.host(final["counts"]), "checks": runtime.host(final["checks"]),
            "initial_parameter_hashes": warm["initial_parameter_hashes"], "normalization": warm["normalization"],
            "gradient_audits": audits, "checkpoints": {"020000": checkpoint}, "minibatches": minibatches,
            "stage1_native_minibatch_sha256": stage1["minibatches"]["phase0_minibatch_sha256"],
            "minibatches_differ_from_stage1_native": distinct, "last_metrics": runtime.host(metrics)}


def evaluate(job, out, deadline):
    runtime, evaluate_worker, _, _ = step03.rerun1()
    budget = W1.Budget(job["transitions"], job["timeout_seconds"])
    i = job["seed_index"]
    banks = {"e2": E.bank_states(E2_BANK), "e1": E.bank_states(E1_BANK)}
    anchors = [E.build_anchor(E2_ANCHORS, k) for k in range(S.BANKS[E2_ANCHORS]["count"])]
    result = {"kind": "evaluate", "seed_index": i, "cells": {}}
    for cell, (data_arm, world, arms) in FORKS.items():
        deadline()
        sub = fork_job(job, i, data_arm, world)
        dataset, plan = W1.load_world(sub, world)
        adapter, template, assertions, expected, _ = W1.bind(runtime, sub, dataset, plan, data_arm, i)
        warm_dir = Path(job["stage1_runs"][f"warmup_s{i}_{data_arm}_{world}"])
        warm = json.loads((warm_dir / "report.json").read_text(encoding="utf-8"))
        require(adapter.context.initial_parameter_hashes == warm["initial_parameter_hashes"], "Evaluation template differs")
        untrained = evaluate_worker.FrozenPolicy(runtime, adapter, template)
        state10 = runtime.restore_exact(adapter, template, W1.read_checkpoint(warm_dir, warm, "010000"))
        entry = {"assertions": assertions, "world": world,
                 "J_E1_untrained": float(E.rollouts(banks["e1"], untrained.act, world, "pi", budget)["returns"].mean()),
                 "branches": {}}
        view = stage2_view(i, data_arm, world)
        for arm in arms:
            deadline()
            directory = Path(job["runs"][f"branch_s{i}_{cell}_{arm}"])
            report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            aa, tt = runtime.branch(adapter, state10, view, i, arm, warm["matching"]["distance_coefficient"])
            B.assert_adapter(aa, expected)
            state = runtime.restore_exact(aa, tt, W1.read_checkpoint(directory, report, "020000"))
            require(runtime.tree_digest(state) == report["checkpoints"]["020000"]["state_digest"], "Restored state differs")
            policy = evaluate_worker.FrozenPolicy(runtime, aa, state)
            summary, _ = W1.free_summary(policy, banks, world, budget)
            branch_entry = {"free": summary}
            if cell == "stall_cutoff" and arm in ("bca", "native"):
                rows = [{"reset_seed": a["reset_seed"], "current": tuple(a["current"]), "prefix_actions": a["prefix_actions"]}
                        for a in anchors]
                obs = np.stack([a["obs"] for a in anchors])
                gated = P.build(len(anchors), E.GATED_SLOTS)
                budget.charge(np.asarray([a["t"] for a in anchors]), gated.shape[1])
                qpi_clip, _, _ = E.qpi.raw_values(rows, gated, policy.act, world, clip_all=True)
                u, q1 = width_and_q(policy, obs, gated)
                branch_entry["anchors"] = {"gated_slots": list(E.GATED_SLOTS), "qpi_clip": qpi_clip, "q1": q1, "U": u,
                                           "valid": np.all(np.isfinite(qpi_clip), axis=1),
                                           "times": [a["t"] for a in anchors]}
            policy.assert_unchanged()
            entry["branches"][arm] = branch_entry
        stage1_pol = W1.policies(sub, runtime, evaluate_worker, dataset, plan, data_arm,
                                 {"warmup": str(warm_dir), "branch": job["stage1_runs"][f"branch_s{i}_{data_arm}_{world}"]},
                                 ("020000",))[1]["020000"]
        entry["stage1_native_on_stage2_bank"], _ = W1.free_summary(stage1_pol, banks, world, budget)
        result["cells"][cell] = entry
    result["transitions"] = budget.used
    return result


# ------------------------------------------------------------------------------------------------ entry

KINDS = {"preflight": W1.preflight, "precheck": precheck, "branch": branch, "evaluate": evaluate}


def run_job(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    out = Path(job_path).resolve().parent
    started = time.monotonic()
    report = {"job": job["id"], "kind": job["kind"], "started_utc": datetime.now(timezone.utc).isoformat(),
              "producer_sha256": W1.producer(KINDS.get(job["kind"]))}
    try:
        require(job["kind"] in KINDS, "Unknown job kind")
        step03.set_worker_environment()
        import units
        limit = int(job.get("transitions") or 0) or None       # precheck and evaluate: the stage allowance; branch: count only
        with units.SimulatorCounter(limit) as counter:
            report.update(KINDS[job["kind"]](job, out, W1.Deadline(job["timeout_seconds"])))
        report["simulator_steps_exact"] = counter.steps
        if limit is not None:                                   # the precheck's 2,400 anchor-rebuild steps count too
            report["transitions_prospective"] = report.get("transitions")
            report["transitions"] = counter.steps
        report.update(status="completed", unit_label="COMPLETE")
    except BaseException as exc:
        report.update(status="failed", unit_label="INCOMPLETE", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            report["interrupted"] = type(exc).__name__
    report["elapsed_seconds"] = time.monotonic() - started
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    step03._write_new(out / "report.json", W1.sanitize(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True)
    result = run_job(parser.parse_args().job)
    sys.exit(0 if result.get("status") == "completed" else 1)
