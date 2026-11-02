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


def integrity_record():
    """What a job is allowed to load: the sha256 of every package source, manifest, seed, protocol and approval file,
    and of rerun1's reused sources. Written into every job.json by the supervisor; `integrity.verify` re-hashes the
    same paths in the worker before any package import (PROTOCOL_V2.md change 6)."""
    import step03
    files = {str(p): sha_file(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.suffix in INTEGRITY_SUFFIXES}
    files.update({str(step03.RERUN1 / name): sha_file(step03.RERUN1 / name) for name in sorted(step03.RERUN1_SOURCES)})
    return {"package_dir": str(HERE), "files": files}


class SimulatorCounter:
    """Counts every simulator transition of this process, on both paths the package uses: `env.step` on environments
    made through `worlds.environment` (collection, anchor prefixes, the environment evaluator), and the canonical
    `dynamics()` function on the module `worlds.load_environment()` returns (the raw float64 loops, the lineage replay,
    the thrust check), counted per state row. The environment's own `step` reaches `dynamics` through the canonical
    module's global, not this alias, so nothing is counted twice. With a limit it fails closed: the call that would
    cross the limit raises CapExhausted before it runs (PROTOCOL_V2.md change 6)."""

    def __init__(self, limit=None):
        self.steps, self.limit, self._originals = 0, limit, None

    def _take(self, n):
        if self.limit is not None and self.steps + n > self.limit:
            from step03 import CapExhausted
            raise CapExhausted(f"exact simulator-step cap {self.limit} would be exceeded ({self.steps} + {n})")
        self.steps += n

    def __enter__(self):
        import numpy as np
        import worlds
        module = worlds.load_environment()
        counter = self
        counter._originals = (worlds.environment, module, module.dynamics)
        original_environment, original_dynamics = worlds.environment, module.dynamics

        def environment(tag, start="recorded"):
            env = original_environment(tag, start)
            inner = env.step

            def step(action):
                counter._take(1)
                return inner(action)
            env.step = step
            return env

        def dynamics(states, actions, physics):
            x = np.asarray(states)
            counter._take(1 if x.ndim <= 1 else int(x.shape[0]))
            return original_dynamics(states, actions, physics)
        worlds.environment, module.dynamics = environment, dynamics
        return self

    def __exit__(self, *exc):
        import worlds
        original_environment, module, original_dynamics = self._originals
        worlds.environment, module.dynamics = original_environment, original_dynamics
        return False
