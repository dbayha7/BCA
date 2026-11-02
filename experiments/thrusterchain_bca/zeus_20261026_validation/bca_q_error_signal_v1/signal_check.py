"""Q-error signal check v1 (rule: RULE.md, written first). Forward passes only on the knee-edge exploratory checkpoints:
Q1's own one-step error, delta1 = Q1(s,a) - r - gamma (1 - done) Q1(s', pi(s')), on every training transition, binned by
the logged a0 at high load. Nothing is fit or trained. User-run in the recorded WSL GPU environment; agents never run
this file and never open its checkpoints.

  /home/david/bca-operator-audit-20261026/bin/python -B -u signal_check.py
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "knee_edge_ood_v3"
for _p in (str(V3), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                            # noqa: E402

KNEE = float(np.float32(0.8))
SEEDS, ARMS, WORLDS = (0, 1, 2), ("cutoff", "random"), ("stall3", "nostall")
THRESHOLD = 0.20
BATCH = 8192


def bins(a0, high):
    """Masks on the logged a0: edge [-0.8, -0.7] and onset [-0.9, -0.8) and deep (< -0.9) at high load; past-knee
    at low load."""
    a0 = np.asarray(a0, np.float64)
    high = np.asarray(high, bool)
    past = a0 < -KNEE
    return {"edge_high": high & (a0 >= -KNEE) & (a0 <= -0.7),
            "onset_high": high & past & (a0 >= -0.9),
            "deep_high": high & (a0 < -0.9),
            "past_low": ~high & past}


def bin_stats(delta, masks):
    out = {}
    for name, m in masks.items():
        d = np.asarray(delta, np.float64)[m]
        out[name] = {"rows": int(m.sum()), "mean": float(d.mean()) if d.size else None,
                     "median": float(np.median(d)) if d.size else None, "sd": float(d.std()) if d.size else None}
    e, o = out["edge_high"]["mean"], out["onset_high"]["mean"]
    out["I_w"] = (o - e) if e is not None and o is not None else None
    return out


def decide(stats):
    """stats: {f"s{i}_{arm}_{world}": {"020000": bin_stats}}. The RANDOM-arm verdict of RULE.md."""
    d = {}
    for i in SEEDS:
        a, b = stats[f"s{i}_random_stall3"]["020000"]["I_w"], stats[f"s{i}_random_nostall"]["020000"]["I_w"]
        d[i] = (a - b) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) else None
    if any(v is None for v in d.values()):
        return "INVALID", d
    passed = sum(v >= THRESHOLD for v in d.values())
    return ("SIGNAL" if passed >= 2 else "NO SIGNAL"), d


def delta1(policy, transition, gamma):
    obs, act = np.asarray(transition.obs, np.float32), np.asarray(transition.action, np.float32)
    rew, nxt = np.asarray(transition.reward, np.float64), np.asarray(transition.next_obs, np.float32)
    done = np.asarray(transition.done, np.float64)
    out = np.empty(len(obs))
    for k in range(0, len(obs), BATCH):
        sl = slice(k, k + BATCH)
        q_sa = np.asarray(policy.q(obs[sl], act[sl]), np.float64)[:, 0]
        q_next = np.asarray(policy.q(nxt[sl], policy.act(nxt[sl])), np.float64)[:, 0]
        out[sl] = q_sa - rew[sl] - gamma * (1.0 - done[sl]) * q_next
    return out


def main():
    import step03
    step03.set_worker_environment()
    import stage1_eval as E
    import stage1_worker as W1
    import probes as P
    from core import VALIDATION, require, sha_file
    from knee_region import load_band, load_g
    exp = VALIDATION / "knee_edge_ood_v2" / "stage1_exploratory_run" / "attempt_1"
    collection = VALIDATION / "knee_edge_ood_v2" / "step04_run" / "attempt_1"
    runtime, evaluate_worker, _, _ = step03.rerun1()
    anchors = [E.build_anchor(W1.E2_ANCHORS, k) for k in range(48)]
    anchor_obs = np.stack([a["obs"] for a in anchors])
    gated = P.build(48, E.GATED_SLOTS)
    stats, validity = {}, {}
    for i in SEEDS:
        evaluation = json.loads((exp / f"evaluate_s{i}" / "report.json").read_text(encoding="utf-8"))
        for world in WORLDS:
            for arm in ARMS:
                name = f"s{i}_{arm}_{world}"
                job = {"provenance": "research", "collection_dir": str(collection), "seed_index": i, "test_seeds": None,
                       "training_seeds": None, "visited_seed": None}
                dataset, plan = W1.load_world(job, world)
                dirs = {"warmup": str(exp / f"warmup_{name}"), "branch": str(exp / f"branch_{name}")}
                _, pols, _, adapter = W1.policies(job, runtime, evaluate_worker, dataset, plan, arm, dirs, ("010000", "020000"))
                transition = adapter.context.train_transition
                gamma = float(adapter.context.args.gamma)
                obs = np.asarray(transition.obs, np.float32)
                masks = bins(np.asarray(transition.action, np.float32)[:, 0], load_band(load_g(obs)) >= 1)
                stats[name] = {"gamma": gamma}
                for ckpt, policy in pols.items():
                    d = delta1(policy, transition, gamma)
                    require(bool(np.all(np.isfinite(d))), f"{name} {ckpt}: non-finite delta1")
                    stats[name][ckpt] = bin_stats(d, masks)
                    policy.assert_unchanged()
                q_now = E.critic(pols["020000"], anchor_obs, gated)[0]
                q_saved = np.asarray(evaluation["runs"][name]["020000"]["anchors_e2"]["q1"], np.float64)[:48]
                rel = float(np.max(np.abs(q_now - q_saved) / np.maximum(1.0, np.abs(q_saved))))
                validity[name] = {"anchor_q1_max_rel_diff": rel, "ok": rel <= 1e-5}
                print(json.dumps({"run": name, "I_w_20k": stats[name]["020000"]["I_w"],
                                  "onset_rows": stats[name]["020000"]["onset_high"]["rows"], "anchor_q1_rel": rel}), flush=True)
    verdict, d = decide(stats)
    if not all(v["ok"] for v in validity.values()):
        verdict = "INVALID"
    out = {"schema": "bca_q_error_signal_v1.result.v1", "written_utc": datetime.now(timezone.utc).isoformat(),
           "rule_sha256": hashlib.sha256((HERE / "RULE.md").read_bytes()).hexdigest(),
           "code_sha256": hashlib.sha256((HERE / "signal_check.py").read_bytes()).hexdigest(),
           "verdict": verdict, "D_random_stall_minus_nostall_20k": d, "threshold": THRESHOLD,
           "validity": validity, "stats": stats}
    path = HERE / f"SIGNAL_RESULT_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(W1.sanitize(out), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "D_by_seed": d, "written": path.name}, indent=2))


if __name__ == "__main__":
    main()
