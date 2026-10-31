"""Step 0.4: collection and exact probes (approval A3; user-run in the recorded WSL GPU environment).

Requires a PASS or flagged PASS from step 0.3 (A2); after a flagged PASS the approval must acknowledge the written
prediction "Stage 1 V0(b) or V0(d) may end INCONCLUSIVE". Cap: 3,000,000 exact simulator steps or 1 hour for the step.

Units (5.0 reruns repeat only an unsealed unit):
- "collection": knee_edge_v1 and its no-stall twin (2 x 1,280 episodes, sealed entropy, research provenance), each
  validated by the schema (bit-for-bit lineage and replay) and saved once; reconciliation 4.7 with L1.4 and the twin
  check; the data-level thrust check (4.5b); the Stage 1 data plan (bindings.plan_streams for both worlds); the step
  0.4 anchors (rebuilt by recorder-prefix replay in stall3 from their sealed draw rules) and L1.5. Sealed in
  collection.json.
- "probes": exact Q^pi at every anchor under the three frozen full-data native 20k actors, clipped, in stall3,
  nostall and stall15_identity; the 4.5 cross-check (environment evaluator on the same layout, compared at -KNEE,
  s10 and s20); the literal identities; stall = no-stall on the supported slots; L2 per continuation seed on the 48
  gated E2 anchors; the reported Stage 1 reachability forecast. Sealed in probes.json.

Run in WSL (instructions and the approval command: A3_README.md):
  /home/david/bca-operator-audit-20261026/bin/python -B -u <this file> --execute
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys
import threading

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np

import attempts as A
import bindings as B
import dataset_knee as D
import probes as P
import qpi
import reconcile as C
import recorder
import seeds as S
import step03
import step04_math as SM4
import thrust_check as T
import units as UN
import worlds
from core import canonical_json, require, sha_file
from knee_region import GATED_PAST_KNEE, S9, SLOTS

RUN = HERE / "step04_run"
MANIFEST = HERE / "A3_MANIFEST.json"
APPROVAL = HERE / "A3_APPROVAL.json"
A3_REQUIRED_FILES = {"step04.py", "step04_math.py", "attempts.py", "units.py", "test_step04.py", "A3_README.md",
                     "PROTOCOL_AMENDMENT_1.md", "PROTOCOL_V2.md"}
APPROVAL_WORDS = ("I approve A3 (step 0.4, collection and exact probes) for knee_edge_ood_v2, "
                  "including protocol amendment 1 (L1.5 calibrated within group) and PROTOCOL_V2.md")
FLAG_ACK = ("I acknowledge that knee_edge_ood_v1 step 0.3 ended INCONCLUSIVE on G0.5 and the written prediction: "
            "Stage 1 V0(b) or V0(d) is likely to end INCONCLUSIVE")
# v2 (PROTOCOL_V2.md, change 1): step 0.3 is not rerun. Its one sealed attempt, in the v1 package, is read and pinned.
V1_A2_REPORT = HERE.parent / "knee_edge_ood_v1" / "step03_run" / "attempt_1" / "report.json"
V1_A2_REPORT_SHA256 = "6c15575541f86ba024949655b987d27bd5f9288d242a428de0df92fedc504215"
G05_STOP_REASON = "under 25% of steep ordinary probes have F <= 0.5 in seed "
STEP_TRANSITIONS, STEP_SECONDS = 3_000_000, 60 * 60     # exact simulator steps (PROTOCOL_V2.md change 6); dry run 2,587,760
EPISODES = 1280
E2_ANCHORS = ("step04_e2_anchors_t0", "step04_e2_anchors_t25", "step04_e2_anchors_t50", "step04_e2_anchors_t75")
E1_ANCHORS = "step04_e1_anchors"
START_BANK = "stage1_e2_bank"
CROSS_SLOTS = ("edge_knee", "s10", "s20")
COLLECTION_PRODUCERS = ("step04.py", "step04_math.py")
A2_MANIFEST_SHA256 = "655e3aa9ee218dff280136aa707ec1c32977a4511d3a1812bcd18e1bbd47d57b"     # as committed in 84202ebd
UPSTREAM_PINS = {"MANIFEST.json": step03.A1_MANIFEST_SHA256, "A2_MANIFEST.json": A2_MANIFEST_SHA256}


def layouts():
    """Forced slots per world: stall3 and nostall S9 + W + T3 images (the same 19-slot array, so the S9 rows of the
    stall = no-stall check come from identical actor batches; nostall's image values are reported only);
    stall15_identity -KNEE + W + T15 images."""
    w = list(GATED_PAST_KNEE)
    return {"stall3": {"named": list(S9) + w, "images": w, "image_world": "stall3"},
            "nostall": {"named": list(S9) + w, "images": w, "image_world": "stall3"},
            "stall15_identity": {"named": ["edge_knee"] + w, "images": w, "image_world": "stall15_identity"}}


def layout_probes(n, world):
    spec = layouts()[world]
    parts = [P.build(n, tuple(spec["named"]))]
    if spec["images"]:
        parts.append(P.build(n, tuple(spec["images"]), kind="identity_image", world=spec["image_world"]))
    return np.concatenate(parts, axis=1), spec["named"] + [f"T({x})" for x in spec["images"]]


# ------------------------------------------------------------------------------------------------ anchors and starts

def build_anchor(name, item):
    """An anchor from its sealed draw rule: reset, then the recorder's commands for t steps in stall3."""
    spec = S.bank_item(name, item)
    t = spec["t"]
    env = worlds.environment("stall3")
    try:
        obs, _ = env.reset(seed=spec["reset_seed"], options={"start": {"current": spec["current"]}})
        uniforms = S.prefix_uniforms(name, item, t)
        prefix = np.zeros((t, 4), np.float32)
        for step in range(t):
            prefix[step] = recorder.requested_command(obs, uniforms[step])[0].astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(prefix[step])
            require(not (terminated or truncated), "Prefix reached the horizon")
        return {"bank": name, "item": item, "t": t, "reset_seed": spec["reset_seed"], "current": spec["current"],
                "prefix_actions": prefix, "state": env.state, "obs": np.asarray(obs, np.float32)}
    finally:
        env.close()


def anchor_sets():
    e2 = [build_anchor(n, i) for n in E2_ANCHORS for i in range(S.BANKS[n]["count"])]
    e1 = [build_anchor(E1_ANCHORS, i) for i in range(S.BANKS[E1_ANCHORS]["count"])]
    return e2, e1


def bank_starts(name):
    out = []
    for i in range(S.BANKS[name]["count"]):
        spec = S.bank_item(name, i)
        env = worlds.environment("stall3")
        try:
            obs, _ = env.reset(seed=spec["reset_seed"], options={"start": {"current": spec["current"]}})
            out.append(np.asarray(obs, np.float32))
        finally:
            env.close()
    return np.stack(out)


# ------------------------------------------------------------------------------------------------ units

def run_collection(directory, budget, *, entropy, episodes, provenance, test_seeds=None):
    budget.charge_raw(2 * 2 * episodes * 100)                    # collection plus the replay check, both worlds
    data = {}
    for world in ("stall3", "nostall"):
        data[world] = D.generate(episodes=episodes, entropy=entropy, world=world, provenance=provenance)
        D.save(data[world], directory / f"knee_edge_v1_{world}.npz")
    recon = C.reconcile(data, test_seeds=test_seeds)
    s = data["stall3"].arrays
    thrust_ok, thrust = T.check(np.asarray(s["states"]), np.asarray(s["actions"]), "stall3")
    plans = {w: B.plan_streams(data[w], w, test_seeds=test_seeds) for w in ("stall3", "nostall")}
    plan_record = {w: {"receipts": {arm: plans[w]["plans"][arm]["receipt"] for arm in B.ARMS},
                       "heldout_ids": plans[w]["heldout_ids"], "dataset_sha256": data[w].digest} for w in plans}
    e2, e1 = anchor_sets()
    starts = bank_starts(START_BANK)
    gated = [a for a in e2 if a["t"] in (0, 25, 50)]
    l15 = {}
    for world in ("stall3", "nostall"):
        arrays = data[world].arrays
        rows = C.M.fitting_rows(arrays)
        ids = np.asarray(arrays["trajectory_ids"])[rows["episodes"]].repeat(100) * 100 + rows["times"]
        deep = np.repeat(np.asarray(arrays["reset_component"])[rows["episodes"]] == 1, 100)
        l15[world] = SM4.l15(rows["obs"], rows["episode_ids"], rows["times"], deep, starts,
                             np.stack([a["obs"] for a in gated]), np.asarray([a["t"] for a in gated]), train_row_ids=ids)
    plumbing_ok = recon["label"] != "INCOMPLETE" and thrust_ok
    record = {"unit_label": "INCOMPLETE" if not plumbing_ok else "COMPLETE", "datasets": {w: {"sha256": d.digest,
              "file": f"knee_edge_v1_{w}.npz", "episodes": episodes, "entropy": entropy} for w, d in data.items()},
              "reconcile": recon, "thrust_check": {"ok": thrust_ok, **thrust}, "l15": l15,
              "l15_ok": all(v["pass"] for v in l15.values()), "plumbing_ok": plumbing_ok,
              "l14_ok": all(recon["worlds"][w]["expected_checks"][k] for w in recon["worlds"]
                            for k in ("l14_random_pooled_high", "l14_random_mid", "l14_random_deep", "l14_cutoff_zero")),
              "expected_ok": all(all(recon["worlds"][w]["expected_checks"].values()) for w in recon["worlds"]),
              "data_plan": plan_record,
              "anchors": {"e2": [{k: a[k] for k in ("bank", "item", "t", "reset_seed", "current", "prefix_actions", "state")} for a in e2],
                          "e1": [{k: a[k] for k in ("bank", "item", "t", "reset_seed", "current", "prefix_actions", "state")} for a in e1]},
              "start_bank": START_BANK, "transitions_charged": budget.used,
              "producer_sha256": {f: sha_file(HERE / f) for f in COLLECTION_PRODUCERS}}
    step03._write_new(directory / "collection.json", record)
    return record


def _anchor_dicts(rows):
    return [{"reset_seed": int(a["reset_seed"]), "current": tuple(a["current"]),
             "prefix_actions": np.asarray(a["prefix_actions"], np.float32).reshape(-1, 4)} for a in rows]


def evaluate_seed(policy, anchors, budget):
    """All worlds' values for one continuation, with the cross-check, identities and stall = no-stall."""
    times = np.asarray([len(a["prefix_actions"]) for a in anchors])
    out, cross = {}, {}
    for world in ("stall3", "nostall", "stall15_identity"):
        probes_array, names = layout_probes(len(anchors), world)
        budget.charge(times, 2 * probes_array.shape[1])            # raw evaluator plus the environment evaluator
        raw, raw_calls, raw_receipt = qpi.raw_values(anchors, probes_array, policy.act, world, clip_all=True)
        env, env_calls, env_receipt = qpi.env_values(anchors, probes_array, policy.act, world, clip_all=True)
        cols = [names.index(s) for s in CROSS_SLOTS if s in names]
        ok, worst = qpi.cross_check(raw[:, cols], env[:, cols])
        cross[world] = {"ok": bool(ok and raw_calls == env_calls), "largest_scaled_deviation": worst,
                        "slots": [names[c] for c in cols]}
        out[world] = {"values": raw, "names": names, "raw_receipt": raw_receipt, "env_receipt": env_receipt}
    identity = {}
    for world in ("stall3", "stall15_identity"):
        v, names = out[world]["values"], out[world]["names"]
        a0 = np.asarray([[SLOTS[n] for n in GATED_PAST_KNEE]], np.float64)
        q_a = v[:, [names.index(n) for n in GATED_PAST_KNEE]]
        q_img = v[:, [names.index(f"T({n})") for n in GATED_PAST_KNEE]]
        residual, receipt = worlds.identity_residual(q_a, q_img, a0, world)
        scaled = np.abs(residual) / np.maximum(1.0, np.abs(q_a))
        identity[world] = {"ok": bool(np.all(scaled <= 1e-6)), "max_scaled_residual": float(scaled.max()), **receipt}
    s3, ns = out["stall3"], out["nostall"]
    s9 = [s3["names"].index(n) for n in S9]
    equal_ok, equal_worst = qpi.cross_check(s3["values"][:, s9], ns["values"][:, [ns["names"].index(n) for n in S9]])
    policy.assert_unchanged()
    return out, cross, identity, {"ok": bool(equal_ok), "largest_scaled_deviation": equal_worst}


def l2_for(values, valid):
    s3, ns = values["stall3"], values["nostall"]
    col = lambda v, n: v["values"][:, v["names"].index(n)]
    res = SM4.neighbor_residuals(np.stack([col(s3, n) for n in S9], axis=1),
                                 {n: col(s3, n) for n in GATED_PAST_KNEE}, {n: col(ns, n) for n in ("s05", "s10", "s20")})
    items, pv = SM4.l2_rule(res, col(s3, "edge_knee"), col(s3, "s10"), valid)
    return items, SM4.reach_forecast(res, valid, pv), res


def run_probes(directory, budget, collection, runtime, evaluate_worker, exports, input_hashes):
    e2 = _anchor_dicts(collection["anchors"]["e2"])
    e1 = _anchor_dicts(collection["anchors"]["e1"])
    gated = np.asarray([a["t"] in (0, 25, 50) for a in collection["anchors"]["e2"]])
    per_seed, ok_flags = {}, {"identity": True, "equal": True, "cross": True}
    for s in range(3):
        budget.check_time()
        policy, _, _ = step03.build_policy(f"seed{s}_full", json.loads((step03.RERUN1 / "protocol.json").read_text()),
                                           runtime, evaluate_worker, exports)
        values, cross, identity, equal = evaluate_seed(policy, e2 + e1, budget)
        n2 = len(e2)
        e2_vals = {w: dict(v, values=v["values"][:n2]) for w, v in values.items()}
        e1_vals = {w: dict(v, values=v["values"][n2:]) for w, v in values.items()}
        valid2 = np.all(np.isfinite(np.concatenate([v["values"] for v in e2_vals.values()], axis=1)), axis=1)
        items, forecast, res = l2_for({w: dict(v, values=v["values"][gated]) for w, v in e2_vals.items()}, valid2[gated])
        e1_items = l2_for(e1_vals, np.all(np.isfinite(np.concatenate([v["values"] for v in e1_vals.values()], axis=1)), axis=1))[0]
        ok_flags["identity"] &= all(x["ok"] for x in identity.values())
        ok_flags["equal"] &= equal["ok"]
        ok_flags["cross"] &= all(x["ok"] for x in cross.values())
        per_seed[s] = {"l2_gated_e2": items, "reach_forecast": forecast, "l2_e1_reported": e1_items, "cross_check": cross,
                       "identity": identity, "stall_equals_nostall_s9": equal,
                       "values": {w: {"names": v["names"], "values": v["values"]} for w, v in values.items()},
                       "rho_hat_gated": {k: {n: x for n, x in d.items()} for k, d in res.items()}}
        step03._write_new(directory / f"probes_seed{s}.json", per_seed[s])
        del policy
    record = {"unit_label": "INCOMPLETE" if not all(ok_flags.values()) else "COMPLETE", "checks": ok_flags,
              "l2": {s: v["l2_gated_e2"] for s, v in per_seed.items()},
              "reach_forecast": {s: v["reach_forecast"] for s, v in per_seed.items()}, "transitions": budget.used}
    for path, digest in input_hashes.items():                    # inputs re-hashed before the unit is sealed
        require(sha_file(path) == digest, f"Sealed input changed during the attempt: {path}")
    step03._write_new(directory / "probes.json", record)
    return record


# ------------------------------------------------------------------------------------------------ the attempt

class StepBudget(step03.Budget):
    def charge_raw(self, transitions):
        self.check_time()
        if self.used + int(transitions) > self.limit:
            raise step03.CapExhausted(f"transition cap {self.limit} would be exceeded")
        self.used += int(transitions)


def reproduction_key(record):
    """Everything a collection record asserts, except accounting and the producing-code hashes."""
    kept = {k: v for k, v in record.items() if k not in ("transitions_charged", "producer_sha256")}
    return canonical_json(json.loads(json.dumps(kept, default=step03._plain)))


def a2_outcome(approval, report_path=None, report_sha256=None):
    """v2 rule (PROTOCOL_V2.md, change 1): v1's single step 0.3 attempt is the step 0.3 record. It may be PASS, or
    INCONCLUSIVE for the G0.5 share reason ONLY, with the known-answer reproduction and the past-knee recount passing.
    Its reach forecasts are recorded, not used for admission (v1's label stopped before them; Stage 1's R0 tests reach
    on the real critics). Every other outcome still stops. The approval always carries FLAG_ACK. v2 never runs
    step 0.3 itself."""
    require(not (HERE / "step03_run").exists() and not (HERE / "A2_APPROVAL.json").exists(),
            "v2 does not run step 0.3: its step03.py copy is a hash anchor only")
    path = V1_A2_REPORT if report_path is None else Path(report_path)
    require(path.is_file(), "v1's step 0.3 (A2) report is missing")
    require(sha_file(path) == (V1_A2_REPORT_SHA256 if report_sha256 is None else report_sha256),
            "v1's step 0.3 report differs from the pinned file")
    last = json.loads(path.read_text(encoding="utf-8"))
    require(last.get("approval", {}).get("a2_manifest_sha256") == A2_MANIFEST_SHA256,
            "The step 0.3 report was not produced under the committed A2 package")
    label, reasons = str(last.get("label", "")), list(last.get("reasons") or [])
    if label == "INCONCLUSIVE":
        require(reasons and all(r.startswith(G05_STOP_REASON) for r in reasons),
                f"Step 0.3 ended INCONCLUSIVE for a reason v2 does not carry: {reasons}")
    else:
        require(label.startswith("PASS"), f"Step 0.3 ended {label}; A3 cannot be requested")
    require(last.get("known_answer_ok") is True and last.get("pooled_past_knee_ok") is True,
            "Step 0.3's known-answer reproduction or past-knee recount did not pass")
    require(FLAG_ACK in approval.get("approval", ""), "The v2 A3 approval must carry the written acknowledgement")
    return {"label": label, "reasons": reasons, "v1_flagged": bool(last.get("flagged")), "v2_prediction_required": True,
            "carried_by": "PROTOCOL_V2.md change 1",
            "reach_recorded_not_gated": {"probe_forecast": last.get("probe_forecast"),
                                         "visited_forecast": last.get("visited_forecast")},
            "report": str(path.relative_to(HERE.parent)), "report_sha256": sha_file(path)}


def execute(*, entropy=None, episodes=EPISODES, provenance="research", test_seeds=None):
    plan = A.check(approval_path=APPROVAL, manifest_path=MANIFEST, run_dir=RUN, words=APPROVAL_WORDS,
                   units=["collection", "probes"], sealed_files={"collection": "collection.json", "probes": "probes.json"},
                   required_files=A3_REQUIRED_FILES, here=HERE)
    upstream = A.verify_upstream(HERE, UPSTREAM_PINS)
    a2 = a2_outcome(plan["approval"])
    entropy = S.load()["numpy"]["dataset_entropy"] if entropy is None else entropy
    directory = RUN / f"attempt_{plan['attempt']}"
    directory.mkdir(parents=True, exist_ok=False)
    step03._write_new(directory / "ATTEMPT.json", {**plan, "a2": a2, "upstream": upstream})
    report = {"step": "0.4", "approval": plan, "a2": a2, "units": plan["units"], "failed_unit": "preflight"}
    counter = UN.SimulatorCounter(STEP_TRANSITIONS).__enter__()       # every simulator step, whichever module takes it
    watchdog = threading.Timer(STEP_SECONDS, _watchdog, args=(directory, report))
    watchdog.daemon = True
    watchdog.start()                                                   # the hour is a hard cap (PROTOCOL_V2.md change 6)
    try:
        step03.set_worker_environment()
        runtime, evaluate_worker, gradient_audit, rerun_inputs = step03.rerun1()
        ledger = {"reproduction_values": 0}
        status, receipt = step03.K.preflight(step03.checked_receipt(runtime, {}), None, ledger)
        report["preflight"] = {"status": status, "receipt": receipt}
        if status != "PASS":
            report.update(label="INCOMPLETE", reasons=["preflight: " + str(receipt)])
            return _finish(directory, report)
        budget = StepBudget(STEP_TRANSITIONS, STEP_SECONDS)
        if "collection" in plan["units"]:
            report["failed_unit"] = "collection"
            collection = run_collection(directory, budget, entropy=entropy, episodes=episodes, provenance=provenance,
                                        test_seeds=test_seeds)
        else:
            collection = json.loads((RUN / plan["sealed_units"]["collection"] / "collection.json").read_text(encoding="utf-8"))
            changed = [f for f, d in collection["producer_sha256"].items() if sha_file(HERE / f) != d]
            if changed:                                  # 5.0: changed producing code must reproduce the unit bit for bit
                report["failed_unit"] = "collection"
                fresh = run_collection(directory, budget, entropy=entropy, episodes=episodes, provenance=provenance,
                                       test_seeds=test_seeds)
                require(reproduction_key(fresh) == reproduction_key(collection),
                        f"Collection does not reproduce under the changed code ({changed})")
                report["collection_reproduced"] = {"changed_files": changed, "sealed": plan["sealed_units"]["collection"]}
                collection = fresh
        report["collection_unit"] = collection["unit_label"]
        probes_record = None
        if collection["unit_label"] != "INCOMPLETE":
            report["failed_unit"] = "probes"
            if "probes" in plan["units"]:
                protocol = json.loads((step03.RERUN1 / "protocol.json").read_text(encoding="utf-8"))
                evaluate_worker.validate_protocol(protocol)
                exports, input_hashes = evaluate_worker.assemble_exports(protocol, rerun_inputs.continuation_root() / "run",
                                                                         step03.sha_file(step03.RERUN1 / "protocol.json"))
                probes_record = run_probes(directory, budget, collection, runtime, evaluate_worker, exports, input_hashes)
            else:
                probes_record = json.loads((RUN / plan["sealed_units"]["probes"] / "probes.json").read_text(encoding="utf-8"))
        checks = (probes_record or {}).get("checks", {"identity": True, "equal": True, "cross": True})
        l2 = {int(k): v for k, v in (probes_record or {}).get("l2", {}).items()}
        label, reasons = SM4.label(plumbing_ok=collection["plumbing_ok"], identity_ok=checks["identity"], equal_ok=checks["equal"],
                                   cross_check_ok=checks["cross"], item5_by_seed={s: v["item5"] for s, v in l2.items()},
                                   l14_ok=collection["l14_ok"], l15_ok=collection["l15_ok"], expected_ok=collection["expected_ok"],
                                   items_by_seed=l2)
        if label == "INCOMPLETE":
            report["failed_unit"] = "collection" if not collection["plumbing_ok"] else "probes"
        else:
            report["failed_unit"] = None
        report.update(label=label, reasons=reasons, l2=l2, reach_forecast=(probes_record or {}).get("reach_forecast"),
                      transitions_prospective=budget.used,
                      standing_prediction_for_A4="Stage 1 V0(d) may end INCONCLUSIVE" if label == "PASS" else None)
    except step03.CapExhausted as exc:
        report.update(label="INCOMPLETE", reasons=[f"{report['failed_unit']} cap: {exc}"])
    except BaseException as exc:
        report.update(label="INCOMPLETE", reasons=[f"{type(exc).__name__}: {exc}"])
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            report["interrupted"] = type(exc).__name__
            _finish(directory, report)
            raise
    finally:
        watchdog.cancel()
        counter.__exit__(None, None, None)
        report["transitions"] = counter.steps                          # the exact count is what the cap governs
    return _finish(directory, report)


def _watchdog(directory, report, exit=os._exit):
    """The hard time cap: at STEP_SECONDS the attempt is written INCOMPLETE and the process exits, whatever it is
    doing. `Budget.check_time` only runs between charges, so on its own it could overrun (PROTOCOL_V2.md change 6)."""
    report.update(label="INCOMPLETE", reasons=[f"time cap {STEP_SECONDS} s exceeded: the watchdog ended the attempt"],
                  watchdog=True)
    try:
        _finish(directory, report)
    except BaseException:
        pass
    exit(3)


def _finish(directory, report):
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    A.finish(report)
    step03._write_new(directory / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    result = execute()
    print(json.dumps({"label": result.get("label"), "reasons": result.get("reasons")}, indent=2, default=step03._plain))
