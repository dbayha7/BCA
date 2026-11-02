"""Stage 1 GPU jobs (approval A4), one per process, dispatched by stage1.py under its caps. User-run in the recorded
WSL GPU environment; agents never run this file and never open its checkpoints.

Every job binds knee-edge data through bindings.create (rerun1 `_initialize_data`, unchanged, with the 4.4 assertions
on every stream) and trains or evaluates with rerun1's own runtime functions:
- warmup: 10k native updates with the shadow auxiliary fit and the 10k calibration (runtime.step), the distance-arm
  gradient match (runtime.match), the 10k gradient audit; saves the complete 10k state.
- branch: a fresh template, exact restore of the authenticated 10k state, runtime.branch(..., "native", recorded
  coefficient) with Stage 1's seeds, 10k more updates (calibration at 15k), audits at 15k and 20k; saves 15k and 20k.
- bc_only: a BC-only actor on stall CUTOFF with its own adapter (assertions, CUTOFF n RANDOM normalization), trained
  with Unifloral `bc.py`'s own train step (selected source nodes, hash-pinned).
- r0: the 10k adequacy and reach checks. evaluate: every Stage 1 measurement for one seed index.
Every warmup and branch job records the sha256 of its phase-0 minibatch index arrays, in order. Every report records
the fingerprint of the code its job kind runs (units.fingerprint), so a sealed unit is reused only under unchanged code.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import ast
import hashlib
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
import seeds as S
import stage1_eval as E
import step03
import step04
from core import PAYLOAD, canonical_json, require, sha_bytes, sha_file

WARMUP_UPDATES = BRANCH_UPDATES = 10_000
BC_UPDATES = 10_000                       # settled convention: the actor-update count of a native 20k run
BC_CHUNK = 1_000
PROGRESS_EVERY = 500
AUDIT_STEPS = {"warmup": (10_000,), "branch": (15_000, 20_000)}
BC_SOURCE = PAYLOAD / "unifloral_td3bc_validation_v1/upstream/algorithms/bc.py"
BC_PROVENANCE = PAYLOAD / "unifloral_td3bc_validation_v1/provenance/unifloral.json"
BC_NODES = ("Args", "AgentTrainState", "Transition", "DeterministicTanhActor", "create_train_state", "make_train_step")
E2_BANK, E1_BANK = "stage1_e2_bank", "stage1_e1_bank"
E2_ANCHORS, E1_ANCHORS = "stage1_e2_anchors", "stage1_e1_anchors"
INTERNAL_MARGIN_SECONDS = 20.0


# ------------------------------------------------------------------------------------------------ shared helpers

class Deadline:
    def __init__(self, seconds):
        self.end = time.monotonic() + max(0.0, float(seconds) - INTERNAL_MARGIN_SECONDS)

    def __call__(self):
        require(time.monotonic() < self.end, "Job deadline reached")


class Budget(step04.StepBudget):
    pass


def write_json(path, value):
    step03._write_new(path, value)
    return {"path": Path(path).name, "sha256": sha_file(path)}


def write_bytes(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return {"path": Path(path).name, "sha256": sha_bytes(raw)}


def progress(out, record):
    step03._progress(Path(out), record)


def smoke_options(job):
    """Smoke-only reductions (stage1_smoke.py): a subset of seeds, arms or runs and fewer anchors. Research jobs refuse
    them, and refuse every test override."""
    smoke = job.get("smoke") or {}
    if job.get("provenance", "research") == "research":
        require(not smoke and job.get("test_seeds") is None and job.get("training_seeds") is None
                and job.get("visited_seed") is None, "Research jobs take no smoke reductions or test overrides")
    else:
        rows = job.get("training_seeds")
        require(isinstance(rows, list) and len(rows) == 3 and job.get("test_seeds") is not None
                and job.get("visited_seed") is not None, "Smoke jobs need test data, training and visited seeds")
        require(job.get("kind") not in ("r0", "evaluate") or smoke.get("stream_seed") is not None,
                "Smoke evaluation uses test-only streams (smoke.stream_seed), never the sealed ones")
    return smoke


def protocol_view(job):
    """The protocol fields rerun1's _initialize_data and branch read: Stage 1 training seeds by seed index."""
    if job.get("provenance", "research") == "research":
        require(job.get("training_seeds") is None, "Research jobs use the sealed training seeds")
        rows = S.load()["training"]
    else:
        rows = job["training_seeds"]
        require(isinstance(rows, list) and len(rows) == 3, "Smoke jobs carry three test training-seed rows")
    return {"seeds": [dict(r) for r in rows]}


def load_world(job, world):
    """The sealed A3 dataset of a world, its re-derived Stage 1 plan, and equality with the sealed A3 data plan."""
    import dataset_knee as D
    collection = json.loads((Path(job["collection_dir"]) / "collection.json").read_text(encoding="utf-8"))
    spec = collection["datasets"][world]
    dataset = D.load(Path(job["collection_dir"]) / spec["file"])
    require(dataset.digest == spec["sha256"], f"{world} dataset differs from the sealed A3 collection")
    plan = B.plan_streams(dataset, world, test_seeds=job.get("test_seeds"))
    for arm in B.ARMS:
        fresh = json.loads(json.dumps(plan["plans"][arm]["receipt"], default=step03._plain))
        require(canonical_json(fresh) == canonical_json(collection["data_plan"][world]["receipts"][arm]),
                f"{world} {arm}: re-derived data plan differs from the sealed A3 plan")
    return dataset, plan


def row_ids(dataset, plan, arm, stream):
    source = plan["plans"][arm]["sources"][stream]
    return np.asarray(dataset.arrays["trajectory_ids"], np.int64)[source[:, 0]] * 100 + source[:, 1].astype(np.int64)


def attach_audit_ids(adapter, dataset, plan, arm):
    """rerun1's gradient audit reads data.source_row_ids and the probe-id digest from the bound receipt."""
    train, probe = row_ids(dataset, plan, arm, "train"), row_ids(dataset, plan, arm, "probe")
    require(len(np.unique(train)) == len(train) and len(np.unique(probe)) == len(probe) == 768, "Row ids not unique")
    require(bool(np.all(np.isin(probe, train))), "Probe rows outside retained replay")
    data = adapter.context.data
    data.source_row_ids = train
    data.receipt["gradient_probe_source_row_ids_sha256"] = hashlib.sha256(probe.astype("<i8").tobytes()).hexdigest()
    return probe


def bind(runtime, job, dataset, plan, arm, seed_index):
    """bindings.create with the driver-side expectation, the recorded device and the audit ids."""
    expected = B.expectation(plan, arm)
    adapter, state, assertions = B.create(protocol_view(job), seed_index, arm, plan, expected)
    kinds = adapter.context.runtime["device_kinds"]
    require(job.get("provenance", "research") != "research" or kinds == [step03.K.RECORDED_DEVICE_KIND],
            f"Device kind {kinds} differs from the recorded one")
    probe = attach_audit_ids(adapter, dataset, plan, arm)
    return adapter, state, assertions, expected, probe


def audit_summary(result):
    """Lambda, the reach radius 2 lambda |dQ1/da0| and gradient signs at high-load probe rows."""
    rows = result["per_row"]
    high = np.asarray(rows["high_load"], bool)
    q = np.asarray(rows["action_loss_gradients"]["q"], np.float64)          # -lambda dQ1/da per row
    bc = np.asarray(rows["action_loss_gradients"]["bc"], np.float64)
    _, side, _ = E.classify(np.asarray(rows["actor_actions"], np.float32))
    reach = 2.0 * np.abs(q[:, 0])
    past = high & (side == -1)
    return {"lambda_by_batch": [b["lambda_from_unpenalized_Q1"] for b in result["batches"]],
            "high_load_rows": int(high.sum()),
            "median_reach_radius_high_load": float(np.median(reach[high])) if high.any() else None,
            "share_dQ1_da0_negative_high_load": float(np.mean(q[high, 0] > 0)) if high.any() else None,
            "actor_past_knee_minus_high_load": int(past.sum()),
            "share_bc_pulls_back_when_past_knee": float(np.mean(bc[past, 0] < 0)) if past.any() else None}


def train(runtime, adapter, state, updates, out, deadline, job, on_step=None):
    """runtime.step `updates` times; the phase-0 minibatch indices are hashed in order (train_worker's sampler)."""
    digest, batches, metrics = hashlib.sha256(), 0, None
    for index in range(updates):
        deadline()
        if int(np.asarray(state.microstep.phase)) == 0:
            digest.update(np.asarray(adapter.sample_indices(state.microstep.rng)).astype("<i4").tobytes())
            batches += 1
        state, metrics = runtime.step(adapter, state)
        if on_step is not None:
            on_step(state, metrics)
        if (index + 1) % PROGRESS_EVERY == 0:
            progress(out, {"status": "running", "job": job["id"], "stage": "training", "step": index + 1,
                           "counts": runtime.host(metrics["counts"])})
    return state, metrics, {"phase0_minibatch_sha256": digest.hexdigest(), "phase0_batches": batches}


def save_state(runtime, out, name, state):
    from flax import serialization
    return {"state": write_bytes(out / f"state_{name}.msgpack", serialization.to_bytes(state)),
            "actor": write_bytes(out / f"actor_{name}.msgpack", serialization.to_bytes(state.microstep.agent_state.actor.params)),
            "state_digest": runtime.tree_digest(state), "actor_digest": runtime.tree_digest(state.microstep.agent_state.actor.params)}


def read_checkpoint(directory, report, name):
    entry = report["checkpoints"][name]["state"]
    raw = (Path(directory) / entry["path"]).read_bytes()
    require(sha_bytes(raw) == entry["sha256"], f"Checkpoint {name} bytes differ from its report")
    return raw


def run_audit(gradient_audit, runtime, adapter, state, probe, out, step):
    result = gradient_audit.audit(adapter, state, probe)
    artifact = write_json(out / f"gradient_audit_{step:06d}.json", runtime.host(result))
    return {**artifact, "summary": audit_summary(result)}


# ------------------------------------------------------------------------------------------------ training jobs

def warmup(job, out, deadline):
    smoke_options(job)
    runtime, _, gradient_audit, _ = step03.rerun1()
    dataset, plan = load_world(job, job["world"])
    adapter, state, assertions, expected, probe = bind(runtime, job, dataset, plan, job["arm"], job["seed_index"])
    context = adapter.context
    untrained_digest = runtime.tree_digest(state.microstep.agent_state.actor.params)
    require(untrained_digest == context.initial_parameter_hashes["actor"], "Fresh actor digest differs")
    state, metrics, minibatches = train(runtime, adapter, state, WARMUP_UPDATES, out, deadline, job)
    final = runtime.validate(adapter, state)
    require(final["counts"]["critic"] == 10_000 and final["counts"]["refresh_accepted"] == 1, "Warmup counts differ")
    before = runtime.tree_digest(state)
    matching, _, _ = runtime.match(adapter, state)
    require(before == runtime.tree_digest(state), "Matching mutated the warmup state")
    audit = run_audit(gradient_audit, runtime, adapter, state, probe, out, 10_000)
    checkpoint = save_state(runtime, out, "010000", state)
    return {"kind": "warmup", "binding": expected, "assertions": assertions, "counts": runtime.host(final["counts"]),
            "device_kinds": list(context.runtime["device_kinds"]),
            "checks": runtime.host(final["checks"]), "initial_parameter_hashes": context.initial_parameter_hashes,
            "normalization": context.normalization, "untrained_actor_digest": untrained_digest,
            "matching": runtime.host(matching), "gradient_audits": {"10000": audit}, "checkpoints": {"010000": checkpoint},
            "minibatches": minibatches, "last_metrics": runtime.host(metrics)}


def restore_warm(runtime, job, dataset, plan, warm_dir):
    """A fresh template, then the exact authenticated 10k state; returns (adapter, state, warm report, assertions)."""
    warm = json.loads((Path(warm_dir) / "report.json").read_text(encoding="utf-8"))
    require(warm["status"] == "completed" and warm["kind"] == "warmup", "Warmup is not completed")
    adapter, template, assertions, expected, probe = bind(runtime, job, dataset, plan, job["arm"], job["seed_index"])
    require(adapter.context.initial_parameter_hashes == warm["initial_parameter_hashes"], "Fresh template differs from warmup")
    require(adapter.context.normalization == warm["normalization"], "Normalization differs from warmup")
    state = runtime.restore_exact(adapter, template, read_checkpoint(warm_dir, warm, "010000"))
    m = warm["matching"]
    require(runtime.tree_digest(state.aux.posterior) == m["reference_digest"]
            and runtime.tree_digest(state.microstep.agent_state.actor.params) == m["anchor_actor_digest"],
            "Restored reference or anchor differs from the matching receipt")
    return adapter, state, warm, assertions, expected, probe, template


def branch(job, out, deadline):
    smoke_options(job)
    runtime, _, gradient_audit, _ = step03.rerun1()
    dataset, plan = load_world(job, job["world"])
    adapter, state, warm, assertions, expected, probe, _ = restore_warm(runtime, job, dataset, plan, job["warmup_dir"])
    adapter, state = runtime.branch(adapter, state, protocol_view(job), job["seed_index"], "native",
                                    warm["matching"]["distance_coefficient"])
    branch_assertions = B.assert_adapter(adapter, expected)
    audits, checkpoints = {}, {}

    def on_step(current, metrics):
        if int(metrics["counts"]["critic"]) == 15_000:
            audits["15000"] = run_audit(gradient_audit, runtime, adapter, current, probe, out, 15_000)
            checkpoints["015000"] = save_state(runtime, out, "015000", current)

    state, metrics, minibatches = train(runtime, adapter, state, BRANCH_UPDATES, out, deadline, job, on_step)
    final = runtime.validate(adapter, state)
    require(final["counts"]["critic"] == 20_000 and final["counts"]["refresh_accepted"] == 2, "Branch counts differ")
    require(set(audits) == {"15000"}, "The 15k audit and checkpoint were not taken")
    audits["20000"] = run_audit(gradient_audit, runtime, adapter, state, probe, out, 20_000)
    checkpoints["020000"] = save_state(runtime, out, "020000", state)
    return {"kind": "branch", "binding": expected, "assertions": assertions, "branch_assertions": branch_assertions,
            "warmup_report_sha256": sha_file(Path(job["warmup_dir"]) / "report.json"),
            "distance_coefficient": warm["matching"]["distance_coefficient"],
            "counts": runtime.host(final["counts"]), "checks": runtime.host(final["checks"]),
            "initial_parameter_hashes": warm["initial_parameter_hashes"], "normalization": warm["normalization"],
            "gradient_audits": audits, "checkpoints": checkpoints, "minibatches": minibatches,
            "last_metrics": runtime.host(metrics)}


def bc_module(native):
    """Unifloral bc.py's own definitions (hash-pinned source; CLI, Gym, D4RL and logging nodes are not compiled)."""
    source = BC_SOURCE.read_bytes()
    pinned = json.loads(BC_PROVENANCE.read_text(encoding="utf-8"))["source_sha256"]["algorithms/bc.py"]
    require(hashlib.sha256(source).hexdigest() == pinned, "bc.py differs from the pinned source")
    tree = ast.parse(source.decode("utf-8"))
    keep = []
    for node in tree.body:
        names = ([node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else
                 [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else [])
        if any(n in BC_NODES for n in names):
            keep.append(node)
    require(sorted(n.name if hasattr(n, "name") else n.targets[0].id for n in keep) == sorted(BC_NODES), "bc.py nodes missing")
    import types
    from collections import namedtuple
    from dataclasses import dataclass
    import distrax
    import flax.linen as nn
    import optax
    from flax.training.train_state import TrainState
    module = types.ModuleType("unifloral_pinned_bc_slice")
    module.__dict__.update(namedtuple=namedtuple, dataclass=dataclass, jax=native.jax, jnp=native.jnp, nn=nn, optax=optax,
                           distrax=distrax, TrainState=TrainState)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(BC_SOURCE), "exec"), module.__dict__)
    module._receipt = {"source": str(BC_SOURCE), "sha256": hashlib.sha256(source).hexdigest(), "nodes": list(BC_NODES)}
    return module


def bc_actor(native, bc, context, seed):
    """bc.py main's initialization (lines after 'Initialize agent'), with the CUTOFF n RANDOM normalization."""
    jax, jnp = native.jax, native.jnp
    args = bc.Args(seed=int(seed), dataset="thrusterchain-finite-horizon-v1", num_updates=BC_UPDATES)
    rng = jax.random.PRNGKey(int(seed))
    actor_net = bc.DeterministicTanhActor(4, context.obs_mean, context.obs_std)
    rng, rng_actor = jax.random.split(rng)
    agent = bc.AgentTrainState(actor=bc.create_train_state(args, rng_actor, actor_net, [jnp.zeros(14)]))
    return args, rng, agent, actor_net


def bc_only(job, out, deadline):
    smoke_options(job)
    runtime, _, _, _ = step03.rerun1()
    dataset, plan = load_world(job, "stall3")
    adapter, _, assertions, expected, _ = bind(runtime, job, dataset, plan, "cutoff", job["seed_index"])
    context = adapter.context
    native = context.native
    jax = native.jax
    bc = bc_module(native)
    seed = protocol_view(job)["seeds"][job["seed_index"]]["bc_only_seed"]
    args, rng, agent, actor_net = bc_actor(native, bc, context, seed)
    step_fn = bc.make_train_step(args, actor_net.apply, context.train_transition)
    run = jax.jit(lambda carry: jax.lax.scan(step_fn, carry, None, BC_CHUNK))
    carry, losses = (rng, agent), []
    for chunk in range(BC_UPDATES // BC_CHUNK):
        deadline()
        carry, loss = run(carry)
        losses.append(float(np.asarray(loss["bc_loss"])[-1]))
        progress(out, {"status": "running", "job": job["id"], "stage": "bc_training", "step": (chunk + 1) * BC_CHUNK})
    agent = carry[1]
    require(int(np.asarray(agent.actor.step)) == BC_UPDATES, "BC update count differs")
    from flax import serialization
    actor = write_bytes(out / "bc_actor.msgpack", serialization.to_bytes(agent.actor.params))
    return {"kind": "bc_only", "binding": expected, "assertions": assertions, "bc_source": bc._receipt, "seed": seed,
            "updates": BC_UPDATES, "bc_loss_by_chunk": losses, "normalization": context.normalization,
            "actor": actor, "actor_digest": runtime.tree_digest(agent.actor.params)}


# ------------------------------------------------------------------------------------------------ evaluation jobs

class ActOnly:
    """A deployed-command policy for actors without a critic (the BC-only actor)."""

    def __init__(self, jax, apply_fn, params):
        self._act = jax.jit(lambda obs: apply_fn(params, obs).sample(seed=None))

    def act(self, observations):
        value = np.asarray(self._act(np.asarray(observations, np.float32)))
        require(value.shape == (len(observations), 4) and value.dtype == np.float32 and np.isfinite(value).all()
                and (np.abs(value) <= 1).all(), "Invalid deployed BC action")
        return value


def policies(job, runtime, evaluate_worker, dataset, plan, arm, run_dirs, checkpoints):
    """(untrained policy, {checkpoint: policy}) for one run, rebuilt from authenticated checkpoints."""
    adapter, template, assertions, expected, _ = bind(runtime, job, dataset, plan, arm, job["seed_index"])
    warm = json.loads((Path(run_dirs["warmup"]) / "report.json").read_text(encoding="utf-8"))
    require(adapter.context.initial_parameter_hashes == warm["initial_parameter_hashes"], "Evaluation template differs")
    untrained = evaluate_worker.FrozenPolicy(runtime, adapter, template)
    require(runtime.tree_digest(template.microstep.agent_state.actor.params) == warm["untrained_actor_digest"],
            "Untrained actor differs from the warmup's initialization")
    out = {}
    state10 = runtime.restore_exact(adapter, template, read_checkpoint(run_dirs["warmup"], warm, "010000"))
    if "010000" in checkpoints:
        out["010000"] = evaluate_worker.FrozenPolicy(runtime, adapter, state10)
    if any(c in checkpoints for c in ("015000", "020000")):
        report = json.loads((Path(run_dirs["branch"]) / "report.json").read_text(encoding="utf-8"))
        aa, tt = runtime.branch(adapter, state10, protocol_view(job), job["seed_index"], "native", report["distance_coefficient"])
        B.assert_adapter(aa, expected)
        for name in ("015000", "020000"):
            if name in checkpoints:
                state = runtime.restore_exact(aa, tt, read_checkpoint(run_dirs["branch"], report, name))
                require(runtime.tree_digest(state) == report["checkpoints"][name]["state_digest"], "Restored state digest differs")
                out[name] = evaluate_worker.FrozenPolicy(runtime, aa, state)
    return untrained, out, assertions, adapter


def free_summary(policy, banks, world, budget, *, keep_e2_pi=False):
    """J for pi, pi_clip and pi_clipall on the E2 and E1 banks in one world, rates on E2 pi."""
    result, kept = {}, None
    for bank, states in banks.items():
        for variant in E.VARIANTS:
            roll = E.rollouts(states, policy.act, world, variant, budget)
            result[f"{bank}/{variant}"] = {"J": float(roll["returns"].mean()), "returns": roll["returns"],
                                           "receipt": roll["receipt"]}
            if bank == "e2" and variant == "pi":
                result["e2_rates"] = E.step_rates(roll["executed"], roll["obs"])
                kept = roll if keep_e2_pi else None
    for bank in banks:
        result[f"{bank}/delta_clip"] = result[f"{bank}/clip"]["J"] - result[f"{bank}/pi"]["J"]
        result[f"{bank}/delta_clipall"] = result[f"{bank}/clipall"]["J"] - result[f"{bank}/pi"]["J"]
    return result, kept


def anchors(smoke_seed=None):
    e2 = [E.build_anchor(E2_ANCHORS, i, smoke_seed) for i in range(S.BANKS[E2_ANCHORS]["count"])]
    e1 = [E.build_anchor(E1_ANCHORS, i, smoke_seed) for i in range(S.BANKS[E1_ANCHORS]["count"])]
    return e2, e1


def r0(job, out, deadline):
    """R0 at the 10k fork, every seed: adequacy for stall CUTOFF and RANDOM on 32 fresh E1 starts, then the stall
    CUTOFF actor's R_deep- rate on 32 fresh E2 starts."""
    runtime, evaluate_worker, _, _ = step03.rerun1()
    budget = Budget(job["transitions"], job["timeout_seconds"])
    dataset, plan = load_world(job, "stall3")
    smoke = smoke_options(job)
    e1, e2 = E.bank_states("r0_e1_starts", smoke.get("stream_seed")), E.bank_states("r0_e2_starts", smoke.get("stream_seed"))
    seeds = {}
    for seed_index in smoke.get("seeds", (0, 1, 2)):
        sub = dict(job, seed_index=seed_index)
        rows = {}
        for arm in smoke.get("arms", B.ARMS):
            deadline()
            untrained, pol, _, _ = policies(sub, runtime, evaluate_worker, dataset, plan, arm,
                                            {"warmup": job["runs"][f"warmup_s{seed_index}_{arm}_stall3"]}, ("010000",))
            j0 = float(E.rollouts(e1, untrained.act, "stall3", "pi", budget)["returns"].mean())
            j10 = float(E.rollouts(e1, pol["010000"].act, "stall3", "pi", budget)["returns"].mean())
            rows[arm] = {"J_E1_untrained": j0, "J_E1_10k": j10, "gain": j10 - j0}
            if arm == "cutoff":
                roll = E.rollouts(e2, pol["010000"].act, "stall3", "pi", budget)
                rows[arm]["reach"] = E.step_rates(roll["executed"], roll["obs"])
        seeds[seed_index] = rows
    adequacy = all(row["gain"] >= 50.0 for rows in seeds.values() for row in rows.values())
    fail_early = adequacy and all(seeds[s]["cutoff"]["reach"]["r_deep_minus"] < 0.05 for s in seeds)
    return {"kind": "r0", "seeds": seeds, "adequacy_ok": bool(adequacy), "fail_early": bool(fail_early),
            "transitions": budget.used}


def evaluate(job, out, deadline):
    """Every Stage 1 measurement for one seed index (four native runs at 15k and 20k, plus the BC-only actor)."""
    runtime, evaluate_worker, _, _ = step03.rerun1()
    budget = Budget(job["transitions"], job["timeout_seconds"])
    i = job["seed_index"]
    smoke = smoke_options(job)
    stream = smoke.get("stream_seed")
    banks = {"e2": E.bank_states(E2_BANK, stream), "e1": E.bank_states(E1_BANK, stream)}
    e2_anchors, e1_anchors = anchors(stream)
    if smoke.get("anchors"):                           # index lists, so the smoke covers every anchor time
        e2_anchors = [e2_anchors[k] for k in smoke["anchors"]["e2"]]
        e1_anchors = [e1_anchors[k] for k in smoke["anchors"]["e1"]]
    wanted = [tuple(r) for r in smoke.get("runs", [(a, w) for w in ("stall3", "nostall") for a in B.ARMS])]
    visited_seed = S.load()["numpy"]["stage1_visited_sample"] if job.get("visited_seed") is None else job["visited_seed"]
    result = {"kind": "evaluate", "seed_index": i, "runs": {}, "smoke": smoke or None}
    normalizations = {}
    for world in ("stall3", "nostall"):
        if not any(w == world for _, w in wanted):
            continue
        dataset, plan = load_world(job, world)
        for arm in B.ARMS:
            if (arm, world) not in wanted:
                continue
            name = f"s{i}_{arm}_{world}"
            dirs = {"warmup": job["runs"][f"warmup_{name}"], "branch": job["runs"][f"branch_{name}"]}
            untrained, pols, assertions, adapter = policies(job, runtime, evaluate_worker, dataset, plan, arm, dirs,
                                                            ("015000", "020000"))
            stall_trained = world == "stall3"
            run = {"assertions": assertions, "J_E1_untrained": float(E.rollouts(banks["e1"], untrained.act, world, "pi",
                                                                                    budget)["returns"].mean())}
            for ckpt, policy in pols.items():
                deadline()
                own, roll = free_summary(policy, banks, world, budget, keep_e2_pi=(ckpt == "020000"))
                entry = {"own_world": own}
                if stall_trained:
                    entry["nostall_world"], _ = free_summary(policy, banks, "nostall", budget)
                    entry["A_e2"] = entry["nostall_world"]["e2/pi"]["J"] - own["e2/pi"]["J"]
                if ckpt == "020000":
                    deadline()
                    entry["anchors_e2"] = E.anchor_values(policy, e2_anchors, world, budget, stall_trained=stall_trained)
                    entry["anchors_e1"] = E.anchor_values(policy, e1_anchors, world, budget, stall_trained=stall_trained)
                    if stall_trained:                          # V1 inside the unit, so a failure leaves it unsealed
                        for key in ("anchors_e2", "anchors_e1"):
                            ok, worst = E.stall_equals_nostall(entry[key])
                            require(ok, f"{name} {key}: stall = no-stall on the S9 slots fails ({worst:.3g} > 1e-9)")
                        entry["stall_equals_nostall_s9"] = True
                    states, obs, times, rows = E.visited_states(roll, visited_seed)
                    vp = E.P.build(len(states), E.VISITED_SLOTS)
                    budget.charge(times, vp.shape[1])
                    vq, _, _ = E.qpi.raw_values_from_states(states, times, vp, policy.act, world, clip_all=True)
                    v1, v2, vmin = E.critic(policy, obs, vp)
                    entry["visited"] = {"rows": rows, "slots": list(E.VISITED_SLOTS), "qpi_clip": vq, "q1": v1, "q2": v2,
                                        "qmin": vmin}
                    entry["visited_vs_training_smd"] = E.standardized_shift(roll["obs"], adapter.context.normalization["mean"],
                                                                            adapter.context.normalization["std"])
                    entry["target_smoothing_w_rate"] = target_rate(adapter, policy.state)
                    policy.assert_unchanged()
                run[ckpt] = entry
            result["runs"][name] = run
            normalizations[world] = adapter.context.normalization
    if len(normalizations) == 2:
        result["training_input_shift_smd"] = E.world_shift(normalizations["stall3"], normalizations["nostall"])
    if job.get("runs", {}).get(f"bc_only_s{i}"):
        result["bc_only"] = bc_summary(job, runtime, banks, budget)
    result["transitions"] = budget.used
    return result


def target_rate(adapter, state):
    """Mean probability that a target-smoothed action on replay next_obs falls in W (exact over the clipped noise)."""
    context = adapter.context
    jax = context.native.jax
    target = state.microstep.agent_state.actor_target
    act = jax.jit(lambda obs: target.apply_fn(target.params, obs).sample(seed=None))
    next_obs = np.asarray(context.train_transition.next_obs, np.float32)
    mu = np.concatenate([np.asarray(act(next_obs[k:k + 8192]))[:, 0] for k in range(0, len(next_obs), 8192)])
    return {"rows": int(len(mu)), "rate": E.target_smoothing_rate(mu, float(context.args.policy_noise),
                                                                  float(context.args.noise_clip))}


def bc_summary(job, runtime, banks, budget):
    from flax import serialization
    directory = Path(job["runs"][f"bc_only_s{job['seed_index']}"])
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    raw = (directory / report["actor"]["path"]).read_bytes()
    require(sha_bytes(raw) == report["actor"]["sha256"], "BC actor bytes differ from its report")
    dataset, plan = load_world(job, "stall3")
    adapter, _, _, _, _ = bind(runtime, job, dataset, plan, "cutoff", job["seed_index"])
    native = adapter.context.native
    bc = bc_module(native)
    _, _, agent, actor_net = bc_actor(native, bc, adapter.context, report["seed"])
    params = serialization.from_bytes(agent.actor.params, raw)
    require(runtime.tree_digest(params) == report["actor_digest"], "BC actor digest differs")
    policy = ActOnly(native.jax, actor_net.apply, params)
    roll = E.rollouts(banks["e2"], policy.act, "stall3", "pi", budget)
    return {"J_e2": float(roll["returns"].mean()), "e2_rates": E.step_rates(roll["executed"], roll["obs"])}


# ------------------------------------------------------------------------------------------------ entry

def preflight(job, out, deadline):
    """The recorded stack before any unit: rerun1's runtime receipt, the device kind and the worker environment."""
    runtime, _, _, _ = step03.rerun1()
    holder, ledger = {}, {"reproduction_values": 0}
    status, receipt = step03.K.preflight(step03.checked_receipt(runtime, holder), None, ledger)
    require(status == "PASS", f"Preflight failed: {receipt}")
    return {"kind": "preflight", "receipt": receipt}


KINDS = {"preflight": preflight, "warmup": warmup, "branch": branch, "bc_only": bc_only, "r0": r0, "evaluate": evaluate}


def run_job(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    out = Path(job_path).resolve().parent
    started = time.monotonic()
    report = {"job": job["id"], "kind": job["kind"], "started_utc": datetime.now(timezone.utc).isoformat(),
              "producer_sha256": producer(KINDS.get(job["kind"]))}
    try:
        require(job["kind"] in KINDS, "Unknown job kind")
        step03.set_worker_environment()
        deadline = Deadline(job["timeout_seconds"])
        import units
        limit = int(job.get("transitions") or 0) or None       # evaluation kinds: the stage allowance; training: count only
        with units.SimulatorCounter(limit) as counter:
            report.update(KINDS[job["kind"]](job, out, deadline))
        report["simulator_steps_exact"] = counter.steps
        if limit is not None:                                   # charged to the stage cap: the exact count, not a formula
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
    step03._write_new(out / "report.json", sanitize(report))
    return report


def producer(function):
    """The code fingerprint of a job kind's function (units.fingerprint), or None for an unknown kind."""
    if function is None:
        return None
    import inspect
    import units
    return units.fingerprint(Path(inspect.getsourcefile(function)).stem, function.__name__)


def sanitize(value):
    """JSON-safe: arrays to lists, non-finite floats to strings (the report must never carry NaN)."""
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else ("nan" if np.isnan(v) else ("positive_infinity" if v > 0 else "negative_infinity"))
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True)
    result = run_job(parser.parse_args().job)
    sys.exit(0 if result.get("status") == "completed" else 1)
