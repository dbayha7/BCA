"""Scoped development amendment: fixed-width connector and one-action hazard.

This module reuses authenticated v1 task/data implementations. It never imports
the physical plant or a learner. Geometry is arithmetic on saved trace rows;
collection is an explicit, separately authorized operation using an injected
task. The existing v1/v2 files and their module globals are never modified.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import types

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V1 = HERE.parent / "two_scenario_v1"
PARENT_MANIFEST_SHA256 = "a48e465c133093ada9a79955161d4b10fce8b393d05c0a5eb088afb3109bf7cd"
CONNECTOR_RADIUS = 0.01


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_original_sources():
    manifest = V1 / "MANIFEST.json"
    if _sha(manifest) != PARENT_MANIFEST_SHA256:
        raise ValueError("Original v1 manifest changed")
    pins = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    result = {}
    for name in ("task.py", "data.py"):
        path = V1 / name
        relative = path.relative_to(REPO).as_posix()
        actual = _sha(path)
        if pins.get(relative) != actual:
            raise ValueError("Original v1 source differs from its seal: " + relative)
        result[relative] = actual
    return result


ORIGINAL_SOURCE_HASHES = verify_original_sources()
T = importlib.import_module("experiments.thrusterchain_bca.two_scenario_v1.task")
D = importlib.import_module("experiments.thrusterchain_bca.two_scenario_v1.data")
for module, filename in ((T, "task.py"), (D, "data.py")):
    if Path(module.__file__).resolve() != (V1 / filename).resolve():
        raise ValueError("Original module resolved outside its authenticated source")
verify_original_sources()


def _distance_to_segments(point, starts, ends):
    delta, offset = ends - starts, starts - point
    den = np.sum(delta * delta, axis=1)
    f = np.clip(np.divide(-np.sum(offset * delta, axis=1), den,
                          out=np.zeros_like(den), where=den > 0), 0.0, 1.0)
    return np.linalg.norm(offset + f[:, None] * delta, axis=1)


def geometry_from_saved(measurements, trace, recipe):
    """Return geometry and named checks without running or changing the plant.

    The caller must persist this complete result before enforcing the checks.
    No alternative radius, action, reference time or hazard is searched.
    """
    if len(trace) != 105:
        raise ValueError("Expected the complete 100-route plus five-excursion trace")
    b, c, goal_time = int(recipe["b_time"]), int(recipe["c_time"]), int(recipe["goal_time"])
    if (b, c, goal_time) != (25, 30, 75):
        raise ValueError("This amendment preserves the original B/C/D reference times")
    route = np.asarray([trace[0]["state"]] + [row["next_state"] for row in trace[:100]], np.float64)
    sources = np.asarray([row["state"] for row in trace[:100]], np.float64)
    old_route = np.asarray(measurements["geometry"]["nominal_route"], np.float64)
    hazard_state = np.asarray(trace[100]["next_state"], np.float64)
    hazard_source = np.asarray(trace[100]["state"], np.float64)
    old_excursion = np.asarray(measurements["geometry"]["excursion"], np.float64)
    if route.shape != (101, 13) or sources.shape != (100, 13) or old_route.shape != route.shape or hazard_state.shape != (13,) or hazard_source.shape != (13,) or old_excursion.shape != (6, 13):
        raise ValueError("Malformed saved full-state arrays")
    actions = np.asarray([row["action"] for row in trace], np.float64)
    if actions.shape != (105, 4):
        raise ValueError("Malformed saved action array")
    complete_excursion = np.asarray([trace[100]["state"]] + [row["next_state"] for row in trace[100:]], np.float64)
    excursion_sources = np.asarray([row["state"] for row in trace[100:]], np.float64)
    finite = bool(np.isfinite(route).all() and np.isfinite(hazard_state).all() and
                  np.isfinite(hazard_source).all() and np.isfinite(actions).all() and
                  np.isfinite(complete_excursion).all())
    checks = {
        "captured_states_and_actions_finite": finite,
        "nominal_source_next_continuity": bool(np.array_equal(sources, route[:-1])),
        "nominal_matches_saved_measurements": bool(np.array_equal(route, old_route)),
        "excursion_source_next_continuity": bool(np.array_equal(excursion_sources, complete_excursion[:-1])),
        "excursion_matches_saved_measurements": bool(np.array_equal(complete_excursion, old_excursion)),
        "hazard_source_is_exact_B": bool(np.array_equal(hazard_source, route[b])),
        "first_opposing_action_unchanged": bool(np.array_equal(actions[100], -actions[b])),
        "trace_phases_and_times_unchanged": all(row["phase"] == "nominal" and row["time"] == i for i, row in enumerate(trace[:100])) and
                                           all(row["phase"] == "excursion" and row["time"] == 25 + i for i, row in enumerate(trace[100:])),
        "initial_state_unchanged": bool(np.array_equal(route[0], np.asarray(recipe["initial_state"], np.float64))),
        "fixed_connector_radius": True,
        "danger_separation_finite": None,
        "danger_radius_finite": None,
        "danger_radius_minimum": None,
        "B_outside_connector": None,
        "C_outside_connector": None,
        "safe_first_edge_avoids_danger": None,
        "bad_first_edge_enters_danger": None,
        "goal_danger_disjoint": None,
        "danger_connector_disjoint": None,
        "goal_not_reached_before_C": None,
    }
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        center = hazard_state[:10]
        separation = float(np.min(_distance_to_segments(center, route[:-1, :10], route[1:, :10])))
        # Same radius recipe/order as v1; nonfinite evidence is checked separately.
        radius = min(float(recipe["danger_radius_max"]), float(recipe["danger_radius_fraction"]) * separation)
        delta = route[c, :10] - route[b, :10]
        trim = float(recipe["connector_trim_fraction"])
        connector_start, connector_end = route[b, :10] + trim * delta, route[c, :10] - trim * delta
        connector_length = float(np.linalg.norm(delta))
        hazard_connector_distance = float(_distance_to_segments(center, connector_start[None], connector_end[None])[0])
        hazard_goal_distance = float(np.linalg.norm(center - route[goal_time, :10]))
    geometry = {
        "goal": {"center": route[goal_time, :10].tolist(), "radius": float(recipe["goal_radius"])},
        "danger": {"center": center.tolist(), "radius": radius},
        "connector": {"kind": "capsule", "start": connector_start.tolist(), "end": connector_end.tolist(), "radius": CONNECTOR_RADIUS},
        "A": route[0].tolist(), "B": route[b].tolist(), "C": route[c].tolist(), "D": route[goal_time].tolist(),
        "b_time": b, "c_time": c, "danger_time": b,
        "nominal_route": route.tolist(), "excursion": [route[b].tolist(), hazard_state.tolist()],
        "danger_route_separation": separation,
        "edge_semantics": "closed straight segment in ten physical coordinates",
        "recipe": copy.deepcopy(recipe),
        "amendment": {"name": "fixed_width_one_action_hazard_v3", "connector_radius": CONNECTOR_RADIUS,
                      "hazard_source": "saved trace[100].next_state", "development_trace_reused": True,
                      "controller_plant_B_C_D_unchanged": True},
    }
    checks["danger_separation_finite"] = bool(np.isfinite(separation))
    checks["danger_radius_finite"] = bool(np.isfinite(radius))
    checks["danger_radius_minimum"] = bool(np.isfinite(radius) and radius >= float(recipe["minimum_region_radius"]))
    if finite and np.isfinite(connector_length) and connector_length > 0 and np.isfinite(radius) and radius > 0:
        checks["B_outside_connector"] = not bool(T.contains(route[b], geometry["connector"]))
        checks["C_outside_connector"] = not bool(T.contains(route[c], geometry["connector"]))
        checks["safe_first_edge_avoids_danger"] = not bool(T.edge_intersects(route[b], route[b + 1], geometry["danger"]))
        checks["bad_first_edge_enters_danger"] = bool(T.edge_intersects(hazard_source, hazard_state, geometry["danger"]))
        checks["goal_danger_disjoint"] = bool(hazard_goal_distance > radius + float(recipe["goal_radius"]))
        checks["danger_connector_disjoint"] = bool(hazard_connector_distance > radius + CONNECTOR_RADIUS)
        checks["goal_not_reached_before_C"] = not bool(np.any(T.contains(route[:c + 1], geometry["goal"])))
    return {"geometry": geometry,
            "values": {"danger_route_separation": separation, "danger_radius": radius,
                       "connector_length": connector_length, "connector_radius": CONNECTOR_RADIUS,
                       "B_connector_endpoint_clearance": float(np.linalg.norm(route[b, :10] - connector_start)),
                       "C_connector_endpoint_clearance": float(np.linalg.norm(route[c, :10] - connector_end)),
                       "danger_goal_center_distance": hazard_goal_distance,
                       "danger_connector_centerline_distance": hazard_connector_distance},
            "checks": checks, "failed_criteria": [key for key, value in checks.items() if value is False],
            "not_evaluable": [key for key, value in checks.items() if value is None],
            "note": "Development-informed geometry only; no collection, policy or BCA efficacy is established."}


def recorder_action(controller, state, time, family, rng, geometry, config):
    """Only family 2's action at B is reversed; family 3 is the safe nominal arm."""
    if family not in (0, 1, 2, 3):
        raise ValueError("Unknown recorder family")
    action = np.asarray(controller(state, time), np.float64)
    sigma = 0.0 if family == 3 else float(config["wide_action_noise"] if family == 1 else config["action_noise"])
    # Preserve original draw shape/order, including zero-scale normal draws.
    noise = rng.normal(0.0, sigma, 4)
    if family == 2 and time == int(geometry["danger_time"]):
        action = -action
    return T.deployed_action(action + noise)


def _adapted_collector():
    verify_original_sources()
    original = D.collect
    copied_globals = dict(original.__globals__)
    copied_globals["recorder_action"] = recorder_action
    adapted = types.FunctionType(original.__code__, copied_globals, "collect", original.__defaults__, original.__closure__)
    adapted.__kwdefaults__ = copy.copy(original.__kwdefaults__)
    adapted.__annotations__ = dict(original.__annotations__)
    adapted.__module__ = __name__
    adapted.__doc__ = "Original sealed collector and schema with only the isolated v3 recorder_action hook replaced."
    return adapted


collect = _adapted_collector()
