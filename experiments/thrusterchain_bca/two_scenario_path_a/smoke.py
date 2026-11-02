"""Disposable short-schedule numerical check; never a production checkpoint.

Importing this file performs no model initialization, fitting, or simulation.
Only a separately approved smoke launcher may call run_learning_smoke.
"""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
EST = timezone(timedelta(hours=-5))
KEYS = ("critic", "actor", "auxiliary", "calibrations")
CAPS = dict(zip(KEYS, (14, 8, 14, 3)))


def _train_module():
    name = "_two_scenario_path_a_smoke_train"
    path = HERE / "train.py"
    if name in sys.modules:
        value = sys.modules[name]
        if Path(value.__file__).resolve() != path.resolve():
            raise ValueError("Smoke train module path collision")
        return value
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class Ledger:
    """Reserve exact kernel work before dispatch, including duplicate restore work."""
    def __init__(self):
        self.attempted = dict.fromkeys(KEYS, 0)
        self.completed = dict.fromkeys(KEYS, 0)
        self.in_step = None

    def begin(self):
        require(self.in_step is None, "A previous smoke microstep is unresolved")
        self.in_step = {"attempted": dict.fromkeys(KEYS, 0), "completed": dict.fromkeys(KEYS, 0)}

    def dispatch(self, event):
        require(self.in_step is not None, "Smoke dispatch without a declared step")
        for key in KEYS:
            source = "refresh" if key == "calibrations" else key
            a, c = int(event["attempted"][source]), int(event["completed"][source])
            old_a, old_c = self.in_step["attempted"][key], self.in_step["completed"][key]
            require(0 <= old_c <= c <= a and old_a <= a, "Nonmonotone smoke accounting")
            self.attempted[key] += a - old_a
            self.completed[key] += c - old_c
            self.in_step["attempted"][key], self.in_step["completed"][key] = a, c
            require(self.attempted[key] <= CAPS[key], "Smoke physical cap exceeded: " + key)

    def finish(self):
        require(self.in_step is not None and self.in_step["attempted"] == self.in_step["completed"],
                "Smoke kernel completion is unresolved")
        self.in_step = None

    def receipt(self):
        return {"attempted": dict(self.attempted), "completed": dict(self.completed),
                "execution_counts_known": self.attempted == self.completed}


def short_warmup(runtime, native, adapter, state):
    """Keep genuine zero counters; only shorten the disposable schedule and budget."""
    c = runtime.validate(adapter, state)["counts"]
    require(c["critic"] == c["actor"] == c["auxiliary"] == c["refresh_accepted"] == 0,
            "Smoke requires fresh zero-update initialization")
    jnp = native.jnp
    state = state._replace(refresh_at=jnp.array([4], jnp.int32), critic_budget=jnp.int32(4))
    actor = state.microstep.agent_state.actor
    adapter = runtime.make_adapter(adapter.context, state, "native", 0., actor.params, [4], 4)
    runtime.validate(adapter, state)
    return adapter, state


def short_fork(runtime, native, adapter, warm, protocol, arm):
    """Production RNG-fold equations at a truthful 4-step boundary, not R.branch()."""
    require(arm in ("native", "bca"), "Only the reviewed native/BCA arms are checked")
    counts = runtime.validate(adapter, warm)["counts"]
    require(counts["critic"] == 4 and counts["actor"] == 2 and counts["auxiliary"] == 4 and
            counts["refresh_accepted"] == 1 and counts["phase"] == 0 and counts["reference_ready"],
            "A genuinely calibrated four-step warmup is required")
    jax, jnp = native.jax, native.jnp
    seed = protocol["seeds"][0]
    rng = jax.random.fold_in(warm.microstep.rng, seed["branch_native_seed"])
    aux_rng = jax.random.fold_in(warm.aux_rng, seed["branch_aux_seed"])
    state = warm._replace(microstep=warm.microstep._replace(rng=rng, reference=warm.aux.posterior),
                          aux_rng=aux_rng, refresh_at=jnp.array([4, 6], jnp.int32), critic_budget=jnp.int32(8))
    result = runtime.make_adapter(adapter.context, state, arm, 0. if arm == "native" else 3.,
                                  warm.microstep.agent_state.actor.params, [4, 6], 8)
    runtime.validate(result, state)
    return result, state


def run_learning_smoke(protocol, services, training_review, deadline, output, persist):
    """Return disposable native/BCA policy states and a complete or failed receipt.

    Genuine original runtime kernels consume the exact retained Bottleneck fit
    and calibration rows. No artificial posterior or evaluator labels are used.
    The caller separately charges physical evaluator checks and global time.
    """
    require(callable(deadline) and callable(persist), "Bounded launcher callbacks are required")
    require(isinstance(training_review, dict) and training_review.get("scope") == "disposable_path_a_smoke",
            "A separate explicit smoke-scoped review is required")
    output = Path(output)
    require(output.is_dir() and not any(output.iterdir()), "A new empty smoke learning directory is required")
    T, R = _train_module(), services.runtime
    ledger, policies, started = Ledger(), {}, time.monotonic()
    receipt = {"schema": "two_scenario.path_a.learning_smoke.v1", "status": "failed_learning_smoke",
        "production_checkpoint": False, "condition": "bottleneck", "seed_index": 0,
        "started_est": datetime.now(EST).isoformat(), "error": None,
        "physical_caps": CAPS, "simulator_transitions": 0, "automatic_retry": False,
        "calibration": "Genuine original calibration using the fixed 128 retained calibration rows; invalid reference is a failure",
        "coverage": "Numerical native/BCA paths on retained slab data only; not a learning/efficacy or Honeypot update validation"}
    _save(output / "START.json", {**receipt, "protocol_sha256": services.binding.I.digest(protocol),
        "missing_final_receipt": "Persisted dispatches are lower bounds; stated physical caps are upper bounds. Never treat missing counts as zero."})
    def notify(phase, event=None):
        value = {"phase": phase, "counts": ledger.receipt(), "observed_est": datetime.now(EST).isoformat()}
        if event is not None:
            value["dispatch"] = event
        persist(value)
    def checkpoint(name, adapter, state):
        deadline()
        raw = services.serialization.to_bytes(state)
        path = output / (name + ".msgpack")
        with path.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        diagnostic = {}
        restored = R.restore_exact(adapter, state, raw, diagnostic=diagnostic)
        require(R.tree_digest(restored) == R.tree_digest(state), "Decoded smoke state changed")
        entry = {"file": path.name, "sha256": T.digest(raw), "state_digest": R.tree_digest(state),
                 "cumulative_counts": R.host(R.validate(adapter, state)["counts"]),
                 "exact_restore": diagnostic, "production_checkpoint": False}
        receipt.setdefault("checkpoints", {})[name] = entry
        _save(output / (name + ".json"), entry)
        return restored
    def step(adapter, state, phase):
        deadline()
        ledger.begin()
        notify(phase)
        def dispatch(event):
            ledger.dispatch(event)
            notify(phase, event)
        result, _ = R.step(adapter, state, progress=dispatch)
        ledger.finish()
        notify(phase + ":committed")
        return result
    try:
        deadline()
        prepared, adapter, state = T._prepare(protocol, "bottleneck", 0, services, training_review)
        receipt["prepared_metadata"] = prepared.metadata
        receipt["runtime"] = R.host(adapter.context.runtime)
        receipt["normalization"] = R.host(adapter.context.normalization)
        adapter, state = short_warmup(R, services.native, adapter, state)
        state = checkpoint("initial_disposable", adapter, state)
        for j in range(4):
            state = step(adapter, state, "warmup_step_%d" % (j + 1))
        warm = checkpoint("warmup_4", adapter, state)
        require(ledger.completed == dict(zip(KEYS, (4, 2, 4, 1))), "Warmup physical counts differ")
        before_warm = R.tree_digest(warm)
        native_adapter, native_start = short_fork(R, services.native, adapter, warm, protocol, "native")
        bca_adapter, bca_start = short_fork(R, services.native, adapter, warm, protocol, "bca")
        require(R.tree_digest(warm) == before_warm and R.tree_digest(native_start) == R.tree_digest(bca_start),
                "Short paired fork changed warmup or differs between arms")
        require(R.config(native_adapter.context.args) == R.config(bca_adapter.context.args), "BC or native settings differ")
        receipt["paired_fork_state_digest"] = R.tree_digest(native_start)
        for arm, branch_adapter, start_state in (("native", native_adapter, native_start), ("bca", bca_adapter, bca_start)):
            state = checkpoint(arm + "_start_4", branch_adapter, start_state)
            for cumulative in (5, 6, 7):
                state = step(branch_adapter, state, arm + "_step_%d" % cumulative)
            require(R.validate(branch_adapter, state)["counts"]["phase"] == 1,
                    "Restore comparison must preserve the odd-phase held batch before an actor update")
            restored_odd = checkpoint(arm + "_held_batch_7", branch_adapter, state)
            first = step(branch_adapter, state, arm + "_actor_original_8")
            repeated = step(branch_adapter, restored_odd, arm + "_actor_restored_8")
            checkpoint(arm + "_actor_original_8", branch_adapter, first)
            checkpoint(arm + "_actor_restored_8", branch_adapter, repeated)
            require(R.tree_digest(first) == R.tree_digest(repeated), "Exact restored actor next step differs for " + arm)
            state = first
            state = checkpoint(arm + "_final_8", branch_adapter, state)
            counts = R.validate(branch_adapter, state)["counts"]
            require(counts["critic"] == 8 and counts["actor"] == 4 and counts["auxiliary"] == 8 and counts["refresh_accepted"] == 2,
                    "Final disposable counters differ")
            policies[arm] = {"adapter": branch_adapter, "state": state}
        deadline()
        require(ledger.attempted == ledger.completed == CAPS, "Smoke total physical counts differ")
        receipt.update(status="passed_learning_smoke", restored_actor_next_step_exact={"native": True, "bca": True},
                       restored_boundary={"critic_before": 7, "phase_before": 1, "critic_after": 8,
                                          "includes": "held batch, actor/critic/auxiliary updates and targets; no calibration duplicated"})
    except BaseException as exc:
        policies = {}
        receipt.update(status="failed_learning_smoke", error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
    receipt.update(counts=ledger.receipt(), execution_counts_known=ledger.attempted == ledger.completed,
                   elapsed_seconds=time.monotonic() - started, finished_est=datetime.now(EST).isoformat())
    # Fail closed on write failure; parent retains the pre-dispatch progress file.
    _save(output / "RESULT.json", receipt)
    persist({"phase": "learning_smoke_terminal", "counts": ledger.receipt(), "status": receipt["status"]})
    return policies, receipt
