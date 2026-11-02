"""Extract the ThrusterChain core v4 environment contract into one compact JSON for the site.

Reads only JSON contracts from the frozen transfer bundle (no simulator, no model). Validates the
documented dynamics formula against every saved fixture transition before exporting it, so the
worked examples on the page use a formula that reproduces the canonical NumPy plant to float64
tolerance. Writes docs/thrusterchain/site/data/environment.json.

Run from the repository root:
    python -I docs/thrusterchain/site/build/extract_environment.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CORE = ROOT / "experiments/thrusterchain_bca/titan_20261026/payload/transfer_bundle/core_v4"
OUT = ROOT / "docs/thrusterchain/site/data/environment.json"

KNEE = 0.800000011920929  # float32(0.8), environment_contract.json
TOL = 1e-9


def f32(x: float) -> float:
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]


def thrust(a: list[float], stall: bool, slope: float) -> list[float]:
    out = []
    for v in a:
        if not stall:
            out.append(v)
            continue
        clipped = max(-KNEE, min(KNEE, v))
        excess = max(abs(v) - KNEE, 0.0)
        sign = 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
        out.append(clipped - slope * sign * excess)
    return out


def step(A, B, x, a, stall, slope):
    """x_next = A x + B thrust(a) with the exogenous rows handled by A itself (current constant, wave rotation)."""
    u = thrust(a, stall, slope)
    nxt = []
    for i in range(13):
        s = 0.0
        for j in range(13):
            s += A[i][j] * x[j]
        for k in range(4):
            s += B[i][k] * u[k]
        nxt.append(s)
    return nxt, u


def reward(x, a):
    return -(sum(v * v for v in x[:10]) + 0.3 * sum(v * v for v in a))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    contract = json.loads((CORE / "contracts/environment_contract.json").read_text("utf8"))
    fixtures = json.loads((CORE / "contracts/dynamics_fixtures.json").read_text("utf8"))
    bank = json.loads((CORE / "contracts/trajectory_bank.json").read_text("utf8"))
    registry = json.loads((CORE / "contracts/scenario_registry.json").read_text("utf8"))
    A, B = fixtures["A"], fixtures["B"]
    if len(A) != 13 or any(len(r) != 13 for r in A) or len(B) != 13 or any(len(r) != 4 for r in B):
        raise SystemExit("unexpected matrix shapes")

    # 1. Validate the formula against every fixture case.
    checked, worst = 0, 0.0
    case_keys = sorted(fixtures["cases"][0].keys()) if fixtures["cases"] else []
    for case in fixtures["cases"]:
        stall = case["world"] == "stall"
        slope = float(fixtures["physics"].get("stall_slope", 1.5))
        x = case["state"]
        # `command` is the float32-quantized, clipped command the plant executed; `action_requested` is the raw request.
        a = [max(-1.0, min(1.0, f32(v))) for v in case["command"]]
        got_next, u = step(A, B, x, a, stall, slope)
        d = max(abs(g - e) for g, e in zip(got_next, case["next_state"]))
        worst = max(worst, d)
        if d > TOL:
            raise SystemExit(f"fixture {case['id']} next_state mismatch {d}")
        d = abs(reward(x, a) - case["reward"])
        worst = max(worst, d)
        if d > TOL:
            raise SystemExit(f"fixture {case['id']} reward mismatch {d}")
        d = max(abs(g - e) for g, e in zip(u, case["thrust"]))
        worst = max(worst, d)
        if d > TOL:
            raise SystemExit(f"fixture {case['id']} thrust mismatch {d}")
        checked += 1

    # 2. Validate the playback trajectories step by step with the same formula and attach rewards/returns.
    trajectories = []
    for tr in bank["trajectories"]:
        stall = tr["world"] == "stall"
        states = [s["state"] for s in tr["steps"]]
        actions = tr["action_sequence"]
        if len(states) != len(actions) + 1:
            raise SystemExit("trajectory length mismatch")
        rewards, pushes, thrusts, worst_tr = [], [], [], 0.0
        for t, a_raw in enumerate(actions):
            a = [max(-1.0, min(1.0, f32(v))) for v in a_raw]
            nxt, u = step(A, B, states[t], a, stall, 1.5)
            d = max(abs(g - e) for g, e in zip(nxt, states[t + 1]))
            worst_tr = max(worst_tr, d)
            if d > 1e-7:  # exported sequences were produced by the canonical plant; allow float accumulation
                raise SystemExit(f"trajectory {tr['name']} step {t} mismatch {d}")
            rewards.append(reward(states[t], a))
            thrusts.append(u)
            x = states[t]
            pushes.append(x[10] + x[11])  # current + wave1, the water's push on every float (0.3 * v * (c + w1))
        gamma = 0.98
        disc = sum((gamma ** t) * r for t, r in enumerate(rewards))
        trajectories.append({
            "name": tr["name"], "world": tr["world"], "horizon": tr["horizon"],
            "sha256": tr.get("trajectory_sha256"),
            "states": [[round(v, 10) for v in s] for s in states],
            "actions": [[round(f32(v), 10) for v in a] for a in actions],
            "thrust": [[round(v, 10) for v in u] for u in thrusts],
            "rewards": [round(r, 10) for r in rewards],
            "discounted_return": disc,
            "undiscounted_return": sum(rewards),
            "max_replay_error": worst_tr,
        })

    # 3. Export.
    out = {
        "schema": "thrusterchain-site-environment-v1",
        "sources": {
            "environment_contract": sha(CORE / "contracts/environment_contract.json"),
            "dynamics_fixtures": sha(CORE / "contracts/dynamics_fixtures.json"),
            "trajectory_bank": sha(CORE / "contracts/trajectory_bank.json"),
            "scenario_registry": sha(CORE / "contracts/scenario_registry.json"),
            "core_relative": "experiments/thrusterchain_bca/titan_20261026/payload/transfer_bundle/core_v4",
        },
        "validation": {
            "fixture_cases_checked": checked, "fixture_case_keys": case_keys,
            "max_abs_error": worst, "tolerance": TOL,
            "formula": "x_next = A x + B thrust(a); r = -(sum s^2 + 0.3 sum a^2) on the pre-transition state and float32-clipped command",
        },
        "config": contract["default_config"],
        "state_order": contract["state"]["order"],
        "observation_order": contract["observation"]["order"],
        "observation_bounds": {"low": contract["observation"]["bounds_low"], "high": contract["observation"]["bounds_high"]},
        "dynamics": contract["dynamics"],
        "reset": contract["reset"],
        "step": contract["step"],
        "A": A, "B": B,
        "physics": fixtures["physics"],
        "fixture_cases": fixtures["cases"],
        "scenario": {k: registry[k] for k in ("regular_worlds", "regular_offline_collection_family", "regular_evaluation_families", "role_of_existing_families", "regular_action_coverage", "scientific_status")},
        "trajectories": trajectories,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False), "utf8", newline="\n")
    print(json.dumps({"written": str(OUT.relative_to(ROOT)), "bytes": OUT.stat().st_size,
                      "fixture_cases_checked": checked, "max_abs_error": worst,
                      "trajectories": [(t["name"], t["world"], round(t["discounted_return"], 4), t["max_replay_error"]) for t in trajectories]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
