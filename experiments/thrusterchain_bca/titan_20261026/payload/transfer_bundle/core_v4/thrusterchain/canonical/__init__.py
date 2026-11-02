"""Load unchanged canonical definitions through a small, explicit dependency bridge.

The two upstream Python files are archival source, not importable packages.
Their hashes are checked before selected original AST nodes are executed.
No upstream repository path, package registration, or dependency gate is imported.
"""
import ast
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE_HASHES = {
    "thrusterchain.py": "f22f2e214ad9231bee4ce60ef55e64409534c2654c4e184406553ba8ed67dde1",
    "gapped_lq.py": "f8ec7cc6c7d9d013d3e509cd3aba0a6dedf1f371480b4e1977f51e2b5dd923aa",
}
MATRIX_SHA256 = "289afb581b4fdbc97fd68566c0b42ff6244b20686e9e49ba177e7e6bd57454a1"


def provenance():
    """Authenticate the local source bytes and matrix artifact, without old paths."""
    record = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    for filename, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"canonical source hash mismatch: {filename}")
    actual = hashlib.sha256((ROOT / "plant.json").read_bytes()).hexdigest()
    if actual != record["plant_sha256"]:
        raise RuntimeError("frozen canonical plant hash mismatch")
    return record


def _assignment_names(node):
    if isinstance(node, ast.Assign):
        return {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
    return set()


def _execute(path, module, *, definitions, assignments):
    parsed = ast.parse(path.read_bytes(), filename=str(path))
    selected = []
    found = set()
    for node in parsed.body:
        names = _assignment_names(node)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in definitions:
            selected.append(node)
            found.add(node.name)
        elif names and names <= assignments:
            selected.append(node)
            found.update(names)
    missing = (definitions | assignments) - found
    if missing:
        raise RuntimeError(f"canonical dependency selection missing: {sorted(missing)}")
    tree = ast.Module(body=selected, type_ignores=[])
    exec(compile(tree, str(path), "exec"), module.__dict__)


@lru_cache(maxsize=1)
def load_canonical():
    """Return the original canonical functions and Env class, isolated from BCA."""
    provenance()
    import gymnasium as gym
    from gymnasium import spaces

    plant = types.ModuleType(__name__ + "._plant")
    plant.__dict__.update(np=np, math=math, dataclass=dataclass)
    sys.modules[plant.__name__] = plant
    _execute(ROOT / "gapped_lq.py", plant,
             definitions={"WorldSpec", "base_input"},
             assignments={"DS", "DA", "NC", "PRIMARY_RUNG", "STAGE0_SPEC"})
    frozen = json.loads((ROOT / "plant.json").read_text(encoding="utf-8"))
    a10 = np.asarray(frozen["A10"], dtype=np.float64)
    a10.setflags(write=False)

    def frozen_transition_matrix(spec, attempts=1000):
        if spec != plant.STAGE0_SPEC:
            raise ValueError("only the authenticated canonical Stage0 plant is supported")
        return a10, frozen["accepted_coupling_attempt"]

    plant.transition_matrix = frozen_transition_matrix
    module = types.ModuleType(__name__ + "._source")
    module.__dict__.update(np=np, math=math, asdict=asdict, dataclass=dataclass,
                           lru_cache=lru_cache, gym=gym, spaces=spaces, GLQ=plant)
    sys.modules[module.__name__] = module
    definitions = {
        "Physics", "world_physics", "push_pattern", "_matrices", "matrices", "_columns",
        "thrust", "dynamics", "thrust_jax", "dynamics_jax", "past_knee", "stalled",
        "push", "wave_amplitude", "float_radius", "start_law", "draw_starts",
        "in_start_set", "bank_starts", "outside_recorded_conditions", "observation_bound",
        "render_line", "ThrusterChainEnv",
    }
    assignments = {
        "DS", "DE", "DA", "DX", "CURRENT", "WAVE1", "WAVE2", "POSITIONS",
        "FLOAT_BOUND", "START_FLOAT_RADIUS", "START_CURRENT", "START_WAVE_AMPLITUDE",
        "WAVE_ROUNDING", "KNEE", "OBS_LOW", "OBS_HIGH", "DATA_SEED", "BANK_SEED",
        "BANK_SIZE", "START_LAWS", "SITUATIONS", "SITUATION_LABELS", "SHIFTED_SITUATIONS",
        "RECORDED_CURRENT_MAX", "RECORDED_WAVE_MAX", "STALL", "NO_STALL", "WORLDS", "WORLD_LABELS",
    }
    _execute(ROOT / "thrusterchain.py", module, definitions=definitions, assignments=assignments)
    a, b = module.matrices()
    digest = hashlib.sha256(a.astype("<f8").tobytes() + b.astype("<f8").tobytes()).hexdigest()
    if digest != MATRIX_SHA256:
        raise RuntimeError("canonical full matrix digest mismatch")
    return module
