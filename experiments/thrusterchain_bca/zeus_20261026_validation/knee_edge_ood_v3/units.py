"""Unit sealing, code fingerprints and 5.0 reproduction for the Stage 1 and Stage 2 supervisors (A4, A5).

- Fingerprint: the sha256 of exactly the code a job kind runs inside the worker modules. Starting from the job
  function and the worker's `run_job` (which builds, stamps and serializes every report), every module-level function
  or class reached by name (directly or through a module alias such as `W1.` or `E.`) is collected. Each is hashed as
  its syntax tree (decorators included, comments and layout excluded), together with the top-level statements
  (imports and constants) of every worker module the job's process imports. Code outside the worker modules is
  pinned by the sealed manifests. So a fix confined to the body of `evaluate` leaves the fingerprints of sealed
  warmups and branches unchanged, as 5.0 permits; a change to a worker module's imports or constants changes every
  kind of that process.
- The sequence of package manifests over the attempts may change at most once (reverting to an earlier package is a
  second change), so the one code-only fix of 5.0 cannot be stretched by alternating packages.
- A sealed unit counts only if the supervisor judged it completed (SUPERVISOR_RESULT.json), not on the worker's own
  report, and only if it has no NOT_REPRODUCED.json marker.
- A sealed unit whose fingerprint changed must re-run and reproduce its sealed report bit for bit (5.0). A re-run
  that does not reproduce, or that does not complete, leaves a NOT_REPRODUCED.json marker pointing at the original
  report (with the keys that differed): it is never sealed afterwards, and its next run is compared with the original
  again, so a second failure under unchanged code ends the step INCONCLUSIVE through attempts.finish.
- Time and transitions are charged once: a reused unit's originally charged supervisor time, a re-run's own cost.
"""
import ast
import hashlib
import json
import symtable
from pathlib import Path

from core import HERE, require, sha_file

WORKER_MODULES = ("stage1_worker", "stage1_eval", "stage2_worker")
VOLATILE = {"started_utc", "finished_utc", "elapsed_seconds", "producer_sha256", "warmup_report_sha256"}


class UnitFailed(Exception):
    pass


def _globals_used(table):
    """Every global name any scope of a function or class reads (nested scopes, comprehensions and lambdas included),
    resolved by the compiler's own symbol tables, so a local of the same name elsewhere cannot hide it."""
    names = {s.get_name() for s in table.get_symbols() if s.is_global() or (s.is_referenced() and not s.is_local()
                                                                          and not s.is_free() and not s.is_parameter())}
    for child in table.get_children():
        names |= _globals_used(child)
    return names


def _parse(name):
    source = (HERE / f"{name}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    tables = {t.get_name(): t for t in symtable.symtable(source, f"{name}.py", "exec").get_children()}
    defs, top, aliases = {}, [], {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defs[node.name] = node
        else:
            top.append(ast.dump(node))
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in WORKER_MODULES:
                        aliases[a.asname or a.name] = (a.name, None)
            elif isinstance(node, ast.ImportFrom) and node.module in WORKER_MODULES:
                for a in node.names:
                    aliases[a.asname or a.name] = (node.module, a.name)
    return {"source": source, "defs": defs, "top": top, "aliases": aliases,
            "globals": {n: _globals_used(tables[n]) for n in defs}}


def _imported(parsed, module):
    """The worker modules a process running `module` imports (closure over worker-module imports)."""
    out, stack = set(), [module]
    while stack:
        m = stack.pop()
        if m not in out:
            out.add(m)
            stack.extend(target for target, _ in parsed[m]["aliases"].values())
    return out


def fingerprint(module, function):
    """sha256 over the syntax trees reachable from module.function and module.run_job inside the worker modules, plus
    the top-level statements of every worker module the process imports."""
    require(module in WORKER_MODULES, f"Not a worker module: {module}")
    parsed = {m: _parse(m) for m in WORKER_MODULES}
    require(function in parsed[module]["defs"], f"{module}.{function} not found")
    seen, stack = set(), [(module, function)] + ([(module, "run_job")] if "run_job" in parsed[module]["defs"] else [])
    while stack:
        mod, name = stack.pop()
        if (mod, name) in seen:
            continue
        seen.add((mod, name))
        p = parsed[mod]
        for ref in p["globals"][name]:                # global names, scope-exact (symtable)
            if ref in p["defs"]:
                stack.append((mod, ref))
            elif ref in p["aliases"] and p["aliases"][ref][1] is not None:
                target, attr = p["aliases"][ref]
                if attr in parsed[target]["defs"]:
                    stack.append((target, attr))
        for node in ast.walk(p["defs"][name]):         # module-alias attributes such as W1.train or E.rollouts
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in p["aliases"]:
                target, attr = p["aliases"][node.value.id]
                if attr is None and node.attr in parsed[target]["defs"]:
                    stack.append((target, node.attr))
    h = hashlib.sha256()
    for mod, name in sorted(seen):
        h.update(f"{mod}.{name}\n".encode() + ast.dump(parsed[mod]["defs"][name]).encode())
    for mod in sorted(_imported(parsed, module) | {m for m, _ in seen}):
        h.update(f"{mod} top-level\n".encode() + "\n".join(parsed[mod]["top"]).encode())
    return h.hexdigest()


def manifest_sequence_ok(plan):
    """At most one change in the sequence of package manifests over all attempts, the current one included."""
    sequence = [h.get("manifest_sha256") for h in plan["history"]] + [plan["manifest_sha256"]]
    require(all(isinstance(m, str) and m for m in sequence), "An attempt report lacks its package manifest; resolve it by hand")
    changes = sum(a != b for a, b in zip(sequence, sequence[1:]))
    return changes <= 1, sequence


def differences(result, ref):
    """Top-level keys whose reproduction values differ, with the largest absolute numeric difference where both are
    numeric trees of the same shape (the 5.0 report of a discrepancy)."""
    a = json.loads(reproduction_key(result))
    b = json.loads(reproduction_key(ref))
    out = {}
    for key in sorted(set(a) | set(b)):
        if json.dumps(a.get(key), sort_keys=True) != json.dumps(b.get(key), sort_keys=True):
            largest = _max_abs(a.get(key), b.get(key))
            out[key] = {"max_abs_difference": largest,
                        "note": "differs in type, sign of zero or structure" if largest in (None, 0.0) else None}
    return out


def _max_abs(x, y):
    if isinstance(x, bool) or isinstance(y, bool):
        return None
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(float(x) - float(y))
    if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        parts = [_max_abs(p, q) for p, q in zip(x, y)]
        return None if any(p is None for p in parts) else max(parts, default=0.0)
    if isinstance(x, dict) and isinstance(y, dict) and set(x) == set(y):
        parts = [_max_abs(x[k], y[k]) for k in x]
        return None if any(p is None for p in parts) else max(parts, default=0.0)
    return None


def read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def reproduction_key(report):
    """A unit report without its timing, code fingerprint and links to other reports' bytes."""
    return json.dumps({k: v for k, v in (report or {}).items() if k not in VOLATILE}, sort_keys=True)


def classify(run_dir, plan):
    """(sealed {unit: attempt}, must_reproduce {unit: reference unit dir}, dropped [units]) after A.check."""
    sealed, reference, dropped = {}, {}, []
    for unit, attempt in plan["sealed_units"].items():
        unit_dir = Path(run_dir) / attempt / unit
        marker = read(unit_dir / "NOT_REPRODUCED.json")
        if marker:
            reference[unit] = marker["reference_dir"]
        elif (read(unit_dir / "SUPERVISOR_RESULT.json") or {}).get("completed"):
            sealed[unit] = attempt
        else:
            dropped.append(unit)
    return sealed, reference, dropped


def same_inputs(unit_dir, job_now):
    """The sealed unit's job.json agrees with the job the current supervisor builds, on every field of job_now
    (identity, seeds, provenance and data source; caps and directory links excluded by the caller)."""
    sealed_job = read(Path(unit_dir) / "job.json") or {}
    return all(json.dumps(sealed_job.get(k), sort_keys=True) == json.dumps(v, sort_keys=True) for k, v in job_now.items())


def take(unit, *, run_dir, directory, sealed, reference, fingerprint_now, caps, cap_key, run, job_now=None):
    """Reuse, reproduce or run one unit. Returns (unit dir, report, mode) with mode in reused/reproduced/ran; raises
    UnitFailed. `run()` dispatches the job and charges its own cost; it returns (report or None, outcome). A sealed
    unit is reused only if both its code fingerprint and its job inputs (job_now) are unchanged."""
    ref, ref_dir = None, None
    if unit in sealed:
        unit_dir = Path(run_dir) / sealed[unit] / unit
        report = read(unit_dir / "report.json")
        require(report is not None, f"Sealed unit {unit} has no readable report")
        if report.get("producer_sha256") == fingerprint_now and (job_now is None or same_inputs(unit_dir, job_now)):
            charged = read(unit_dir / "SUPERVISOR_RESULT.json") or {}
            caps[cap_key] += float(charged.get("elapsed_seconds", report.get("elapsed_seconds", 0.0)))
            caps["evaluation_transitions"] += int(report.get("transitions", 0) or 0)
            return unit_dir, report, "reused"
        ref, ref_dir = report, unit_dir
    elif unit in reference:
        ref_dir = Path(reference[unit])
        ref = read(ref_dir / "report.json")
        require(ref is not None, f"Reference report of {unit} is unreadable")
    try:
        result, outcome = run()
    except BaseException:
        if ref is not None:                            # an interrupted re-run keeps the reference too
            _mark(directory, unit, ref_dir, "the re-run was interrupted", {})
        raise
    if result is None:
        if ref is not None:
            _mark(directory, unit, ref_dir, "the re-run did not complete", {})
        raise UnitFailed(f"job {unit} did not complete: {outcome}")
    if ref is not None and reproduction_key(result) != reproduction_key(ref):
        diff = differences(result, ref)
        _mark(directory, unit, ref_dir, "the re-run does not reproduce", diff)
        raise UnitFailed(f"unit {unit} does not reproduce its sealed report (5.0); differing keys: {diff}")
    return Path(directory) / unit, result, ("reproduced" if ref is not None else "ran")


def _mark(directory, unit, ref_dir, reason, diff):
    """The marker keeps the pointer to the original report, so the unit is compared with it again next time."""
    unit_dir = Path(directory) / unit
    if unit_dir.is_dir():
        with (unit_dir / "NOT_REPRODUCED.json").open("x", encoding="utf-8") as stream:
            json.dump({"reference_dir": str(ref_dir), "reason": reason, "differing_keys": diff,
                       "rule": "5.0: units made by changed code must reproduce bit for bit"}, stream, indent=2)


# ------------------------------------------------------------------------------------------ v2: integrity, exact counting

INTEGRITY_SUFFIXES = (".py", ".json", ".md")
# PROTOCOL_V3.md R4: files outside the package that a job reaches by path. Relative to the validation folder:
# rerun1 design.py (runtime.match loads it for the distance coefficient) and the supervisor every dispatch runs under.
EXTERNAL_DEPENDENCIES = ("central_quartile_training_v1_evaluation_rerun1/design.py", "matched_policy_pilot_v1/launch.py")
# Relative to core.PAYLOAD: the Unifloral payload modules rerun1 runtime.py loads by path (native_loader also verifies
# the upstream Unifloral sources against its own constants, so recording it pins them), and every source of the BCA
# pipeline's uncertainty package and of the canonical core's `thrusterchain` package (the environment wrapper,
# canonical/__init__.py with its SOURCE_HASHES, datasets.py). Review finding R4 (2026-10-31): v3's first closure
# missed all of these.
PAYLOAD_DEPENDENCIES = ("unifloral_td3bc_validation_v1/native_loader.py", "unifloral_td3bc_validation_v1/candidate.py",
                        "unifloral_td3bc_validation_v1/provenance/unifloral.json", "unifloral_checkpoint_v1/training.py",
                        "unifloral_bca_pipeline_validation_v1/bca_bridge.py",
                        "unifloral_bca_pipeline_validation_v1/uncertainty_provenance.json")
PAYLOAD_TREES = ("unifloral_bca_pipeline_validation_v1/uncertainty_source", "transfer_bundle/core_v4/thrusterchain")
SEALED_MANIFESTS = ("MANIFEST.json", "A2_MANIFEST.json", "A3_MANIFEST.json", "A4_MANIFEST.json", "A5_MANIFEST.json",
                    "EXPLORATORY_MANIFEST.json")


def _tree(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix in (".py", ".json")
                  and "__pycache__" not in p.parts)


def dependency_closure():
    """Every file a job may load, as absolute paths: the package's sources, manifests, seeds, protocol and approval
    files; rerun1's reused sources (step03.RERUN1_SOURCES, already hash-pinned); EXTERNAL_DEPENDENCIES; the payload
    files and trees above. Every package .py file must be listed by a sealed manifest, so the record is the sealed
    set and not whatever happens to be in the folder; `integrity.changed_files` also reports any importable file added
    to the package folder after the record (review finding R4). Still outside: the Unifloral upstream tree (verified by
    native_loader's constants, which the record pins) and the Python and JAX installation itself."""
    import json
    import step03
    from core import PAYLOAD, VALIDATION
    files = [p for p in sorted(HERE.iterdir()) if p.is_file() and p.suffix in INTEGRITY_SUFFIXES]
    listed = set()
    for name in SEALED_MANIFESTS:
        if (HERE / name).is_file():
            listed |= set(json.loads((HERE / name).read_text(encoding="utf-8")).get("files", {}))
    unlisted = [p.name for p in files if p.suffix == ".py" and p.name not in listed]
    require(not unlisted, f"dependency closure: package sources no sealed manifest lists: {unlisted}")
    files += [step03.RERUN1 / name for name in sorted(step03.RERUN1_SOURCES)]
    files += [VALIDATION / rel for rel in EXTERNAL_DEPENDENCIES]
    files += [PAYLOAD / rel for rel in PAYLOAD_DEPENDENCIES]
    for tree in PAYLOAD_TREES:
        found = _tree(PAYLOAD / tree)
        require(found, f"dependency closure: empty payload tree {tree}")
        files += found
    for path in files:
        require(path.is_file(), f"dependency closure: missing {path}")
    return files


def integrity_record():
    """What a job is allowed to load (PROTOCOL_V2.md change 6; closure widened by PROTOCOL_V3.md R4): the sha256 of
    every file in `dependency_closure()`. Written into every job.json by the supervisor; `integrity.verify` re-hashes
    the same paths in the worker before any package import, and `integrity.changed_files` again after the work."""
    return {"package_dir": str(HERE), "files": {str(p): sha_file(p) for p in dependency_closure()},
            "external_dependencies": list(EXTERNAL_DEPENDENCIES), "payload_dependencies": list(PAYLOAD_DEPENDENCIES),
            "payload_trees": list(PAYLOAD_TREES)}


def _unit_spend(unit_dir):
    """One unit directory's physical spend: seconds (the supervisor's measurement, else the worker's, with the source
    named), exact simulator steps, completion, and explicit unknowns with their upper bounds."""
    result, report, job = read(unit_dir / "SUPERVISOR_RESULT.json") or {}, read(unit_dir / "report.json") or {}, read(unit_dir / "job.json") or {}
    if "elapsed_seconds" in result:
        seconds, source = result["elapsed_seconds"], "supervisor"
    elif "elapsed_seconds" in report:
        seconds, source = report["elapsed_seconds"], "worker"
    else:
        seconds, source = None, None
    steps = report.get("simulator_steps_exact")
    entry = {"kind": job.get("kind"), "seconds": None if seconds is None else float(seconds), "seconds_source": source,
             "simulator_steps": None if steps is None else int(steps),
             "completed": bool(result["completed"]) if "completed" in result else report.get("status") == "completed"}
    if seconds is None:
        entry["upper_bound_seconds"] = float(job.get("timeout_seconds") or result.get("cap_seconds") or 0.0)
    if steps is None:                              # evaluation kinds carry their allowance; training kinds count only
        entry["upper_bound_simulator_steps"] = int(job["transitions"]) if job.get("transitions") else None
    marker = read(unit_dir / "NOT_REPRODUCED.json")
    if marker:
        entry.update(not_reproduced=True, reference_dir=marker.get("reference_dir"))
    return entry


def attempt_spend(attempt_dir):
    """Physical spend of one attempt directory, every unit counted once; unknown seconds and unknown steps are listed
    with upper bounds (None when a training unit lost its count: unbounded), never added as zero."""
    out = {"units": {}, "known_seconds": 0.0, "known_simulator_steps": 0, "unknown_seconds_units": [],
           "unknown_seconds_upper_bound": 0.0, "unknown_step_units": [], "unknown_steps_upper_bound": 0,
           "unbounded_step_units": [], "not_reproduced_units": []}
    attempt_dir = Path(attempt_dir)
    if not attempt_dir.is_dir():
        return out
    for unit_dir in sorted(p for p in attempt_dir.iterdir() if p.is_dir()):
        e = _unit_spend(unit_dir)
        out["units"][unit_dir.name] = e
        if e["seconds"] is None:
            out["unknown_seconds_units"].append(unit_dir.name)
            out["unknown_seconds_upper_bound"] += e["upper_bound_seconds"]
        else:
            out["known_seconds"] += e["seconds"]
        if e["simulator_steps"] is None:
            out["unknown_step_units"].append(unit_dir.name)
            if e["upper_bound_simulator_steps"] is None:
                out["unbounded_step_units"].append(unit_dir.name)
            else:
                out["unknown_steps_upper_bound"] += e["upper_bound_simulator_steps"]
        else:
            out["known_simulator_steps"] += e["simulator_steps"]
        if e.get("not_reproduced"):
            out["not_reproduced_units"].append(unit_dir.name)
    return out


def ledger(run_dir, current_attempt=None):
    """PROTOCOL_V3.md R6: physical spend of every PRIOR attempt of a stage (the current one, named, is excluded; its
    physical spend is `physical(...)` at the terminal write). The attempt's `caps` remain the logical ledger, which
    charges a reused unit at its original cost: caps and this ledger therefore overlap on reused units by design, and
    cumulative physical spend is ledger + physical, never ledger + caps."""
    out = {"attempts": {}, "prior_known_seconds": 0.0, "prior_known_simulator_steps": 0, "prior_unknown_units": [],
           "prior_unknown_seconds_upper_bound": 0.0, "prior_unknown_step_units": [], "prior_unknown_steps_upper_bound": 0,
           "prior_unbounded_step_units": [], "prior_nonreproducing_units": [], "excluded_current_attempt": current_attempt}
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return out
    for attempt in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("attempt_")):
        if attempt.name == current_attempt:
            continue
        s = attempt_spend(attempt)
        out["attempts"][attempt.name] = s["units"]
        out["prior_known_seconds"] += s["known_seconds"]
        out["prior_known_simulator_steps"] += s["known_simulator_steps"]
        out["prior_unknown_units"] += [f"{attempt.name}/{u}" for u in s["unknown_seconds_units"]]
        out["prior_unknown_seconds_upper_bound"] += s["unknown_seconds_upper_bound"]
        out["prior_unknown_step_units"] += [f"{attempt.name}/{u}" for u in s["unknown_step_units"]]
        out["prior_unknown_steps_upper_bound"] += s["unknown_steps_upper_bound"]
        out["prior_unbounded_step_units"] += [f"{attempt.name}/{u}" for u in s["unbounded_step_units"]]
        out["prior_nonreproducing_units"] += [f"{attempt.name}/{u}" for u in s["not_reproduced_units"]]
    return out


def physical(directory, prior):
    """The current attempt's physical spend and the cumulative total with the prior attempts (`ledger`)."""
    now = attempt_spend(directory)
    prior = prior or {}
    return {"current_attempt": {k: v for k, v in now.items() if k != "units"},
            "cumulative_known_seconds": prior.get("prior_known_seconds", 0.0) + now["known_seconds"],
            "cumulative_known_simulator_steps": prior.get("prior_known_simulator_steps", 0) + now["known_simulator_steps"],
            "cumulative_unknown_seconds_upper_bound": prior.get("prior_unknown_seconds_upper_bound", 0.0) + now["unknown_seconds_upper_bound"],
            "cumulative_unknown_steps_upper_bound": prior.get("prior_unknown_steps_upper_bound", 0) + now["unknown_steps_upper_bound"],
            "cumulative_unbounded_step_units": len(prior.get("prior_unbounded_step_units", [])) + len(now["unbounded_step_units"]),
            "note": "caps is the logical ledger (a reused unit is charged at its original cost); this block is work "
                    "actually run, in this attempt and in all attempts of the stage"}


class Terminal:
    """PROTOCOL_V3.md R5: the single owner of an attempt's terminal write. The lock is held through the write (review
    finding R5): the first caller writes, a caller that arrives during the write waits and then returns None, and
    `done` is set only after the write returned, so a write that raises leaves the owner open for the next caller.
    The owner writes a copy of the report with the counter's exact and attempted simulator steps and the elapsed time
    snapshotted into it, so no terminal record lacks its accounting and a timer that fires late cannot overwrite it."""

    def __init__(self, write, counter=None):
        import threading
        import time
        self._write, self._counter, self._lock, self.done = write, counter, threading.Lock(), False
        self._started, self._clock = time.monotonic(), time.monotonic

    def snapshot(self, report):
        if self._counter is not None:
            report["transitions"] = self._counter.steps                 # the exact count is what the cap governs
            report["simulator_steps_exact"] = self._counter.steps
            report["simulator_steps_attempted"] = self._counter.attempted
        report["elapsed_seconds"] = self._clock() - self._started
        return report

    def finish(self, directory, report, **fields):
        """Write once. Returns the written record, or None when another caller has already written."""
        with self._lock:
            if self.done:
                return None
            record = self.snapshot({**report, **fields})
            written = self._write(directory, record)
            self.done = True
            return record if written is None else written

    def end(self, directory, report, **fields):
        """finish() for the main thread: the written record, or the report when another caller wrote first."""
        written = self.finish(directory, report, **fields)
        return report if written is None else written


class SimulatorCounter:
    """Counts every simulator transition of this process, on both paths the package uses: `env.step` on environments
    made through `worlds.environment` (collection, anchor prefixes, the environment evaluator), and the canonical
    `dynamics()` function on the module `worlds.load_environment()` returns (the raw float64 loops, the lineage replay,
    the thrust check), counted per transition row over every leading dimension (R3). The environment's own `step`
    reaches `dynamics` through the canonical module's global, not this alias, so nothing is counted twice. With a
    limit it fails closed: the call that would cross the limit raises CapExhausted before it runs (PROTOCOL_V2.md
    change 6). `attempted` counts reservations, `steps` completed calls (R3)."""

    def __init__(self, limit=None):
        self.steps, self.attempted, self.limit, self._originals, self._dims = 0, 0, limit, None, None

    def _take(self, n):
        """Reserve n steps before the call: the cap is checked on attempted work, and `steps` (completed) advances in
        `_complete` only after the call returns, so an exception inside the simulator leaves attempted > steps."""
        n = int(n)
        if self.limit is not None and self.attempted + n > self.limit:
            from step03 import CapExhausted
            raise CapExhausted(f"exact simulator-step cap {self.limit} would be exceeded ({self.attempted} + {n})")
        self.attempted += n
        return n

    def _complete(self, n):
        self.steps += n

    def rows(self, states, actions):
        """The transitions one dynamics() call simulates: the product of the broadcast leading dimensions of states
        (..., DX) and actions (..., DA); a 1-D state is one transition; mismatched leading shapes are rejected by
        numpy's broadcast rule. (PROTOCOL_V3.md R3: v2 counted states.shape[0], so a (1280, 100, 13) validation replay
        counted 1,280 instead of 128,000.)"""
        import numpy as np
        x, a = np.asarray(states), np.asarray(actions)
        dx, da = self._dims
        require(x.ndim >= 1 and x.shape[-1] == dx, f"dynamics states must end in the state dimension {dx}: {x.shape}")
        require(a.ndim >= 1 and a.shape[-1] == da, f"dynamics actions must end in the command dimension {da}: {a.shape}")
        lead = np.broadcast_shapes(x.shape[:-1], a.shape[:-1])
        return int(np.prod(lead, dtype=np.int64)) if lead else 1

    def __enter__(self):
        import numpy as np
        import worlds
        module = worlds.load_environment()
        counter = self
        counter._originals = (worlds.environment, module, module.dynamics)
        counter._dims = (int(module.DX), int(module.DA))             # states (..., 13), commands (..., 4)
        original_environment, original_dynamics = worlds.environment, module.dynamics

        def environment(tag, start="recorded"):
            env = original_environment(tag, start)
            inner = env.step

            def step(action):
                n = counter._take(1)
                result = inner(action)
                counter._complete(n)
                return result
            env.step = step
            return env

        def dynamics(states, actions, physics):
            n = counter._take(counter.rows(states, actions))
            result = original_dynamics(states, actions, physics)
            counter._complete(n)
            return result
        worlds.environment, module.dynamics = environment, dynamics
        return self

    def __exit__(self, *exc):
        import worlds
        original_environment, module, original_dynamics = self._originals
        worlds.environment, module.dynamics = original_environment, original_dynamics
        return False
