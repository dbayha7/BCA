"""Development G2 screen of candidate knee_s10_v2 (rule: G2_DEV_RULE_v2.md, written first). No training, no
checkpoint, no GPU, no sealed file touched. Development data: never confirmatory.

  PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/g2_dev_v2.py

The measurements, D4 bounds and label code are v1's sealed formal driver, imported unchanged (its preflight and
execute are never called). The one v2 thing is the prefix rule (prefix_rule.py). Writes g2_development_result_v2.json.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "knee_s10_candidate_v1"
FORMAL = V1 / "formal_g2"
for _p in (str(HERE), str(FORMAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import g2_formal as G                                       # noqa: E402  v1's sealed driver (replaces `recorder` first)
import numpy as np                                          # noqa: E402

import dataset_knee as D                                    # noqa: E402
import fixtures as F                                        # noqa: E402
import g2_seeds as GS                                       # noqa: E402
import pi_ref                                               # noqa: E402
import prefix_rule as PR                                    # noqa: E402
import qpi                                                  # noqa: E402
import stage1_eval as E                                     # noqa: E402
import step04_math as SM4                                   # noqa: E402
from core import require                                    # noqa: E402

ENTROPY = F.TEST_ENTROPY + 21              # 424263: v1's development collection entropy (outcomes known)
STREAM = 707_070_707                       # v1's development anchor and start stream
EPISODES = 1280
ANCHOR_BANK, START_BANK, ANCHORS = "stage1_e2_anchors", "stage1_e2_bank", 64
RULE, OUT = HERE / "G2_DEV_RULE_v2.md", HERE / "g2_development_result_v2.json"
V1_DEV = V1 / "g2_development_result_r3.json"
CODE = {"prefix_rule.py": HERE / "prefix_rule.py", "g2_dev_v2.py": HERE / "g2_dev_v2.py",
        "../knee_s10_candidate_v1/pi_ref.py": V1 / "pi_ref.py",
        "../knee_s10_candidate_v1/recorder_wide.py": V1 / "recorder_wide.py",
        "../knee_s10_candidate_v1/formal_g2/g2_formal.py": FORMAL / "g2_formal.py",
        "../knee_s10_candidate_v1/formal_g2/d4_bounds.json": FORMAL / "d4_bounds.json"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def start_rows():
    rows = []
    for i in range(128):
        spec = E.stream_item(START_BANK, i, STREAM)
        rows.append({"reset_seed": spec["reset_seed"], "current": tuple(spec["current"]), "prefix_actions": np.zeros((0, 4), np.float32)})
    return rows


def d3_by_time(data, anchors):
    """Reported only: D3's anchor share within each gated time group, per world (the same l15 call on the subset)."""
    out = {}
    for world, ds in data.items():
        arrays = ds.arrays
        rows = G.M.fitting_rows(arrays)
        deep_ep = np.repeat(np.asarray(arrays["reset_component"])[rows["episodes"]] == 1, 100)
        out[world] = {}
        for t in GS.GATED_T:
            sub = [a for a in anchors if a["t"] == t]
            l15 = SM4.l15(rows["obs"], rows["episode_ids"], rows["times"], deep_ep, np.stack([a["obs"] for a in sub[:1]]),
                          np.stack([a["obs"] for a in sub]), np.asarray([a["t"] for a in sub]))
            out[world][str(t)] = {"anchor_share": l15["anchor_share"], "n": len(sub)}
    return out


def pairing(d, h, by_time):
    """Reported only: v2 against v1's development r3 result on the same data, reset seeds and prefix uniforms."""
    if not V1_DEV.is_file():
        return {"available": False}
    v1 = json.loads(V1_DEV.read_text(encoding="utf-8"))
    return {"available": True, "v1_result": V1_DEV.name, "v1_sha256": sha(V1_DEV),
            "D3_anchor_share": {w: {"v1": v1["D"][w]["D3"]["anchor_share"], "v2": d[w]["D3"]["anchor_share"]} for w in d},
            "D3_start_share": {w: {"v1": v1["D"][w]["D3"]["start_share"], "v2": d[w]["D3"]["start_share"]} for w in d},
            "data_side_same_verdicts": {w: all(v1["D"][w][k]["ok"] == d[w][k]["ok"] for k in ("D1", "D2", "B1")) for w in d},
            "H2_median_rho_s10": {"v1": v1["H"]["H2"]["median_exact_rho_s10"], "v2": h["H2"]["median_exact_rho_s10"]},
            "median_exact_rho": {"v1": v1["H"]["reported"]["median_exact_rho"], "v2": h["reported"]["median_exact_rho"]},
            "projection_split": {"v1": v1["H"]["reported"]["rho_s10_split_by_a0_projection_at_first_continuation_state"],
                                 "v2": h["reported"]["rho_s10_split_by_a0_projection_at_first_continuation_state"]},
            "v2_D3_anchor_share_by_time": by_time}


def main():
    import stage1_smoke
    sealed = stage1_smoke.sealed_values()
    formal = GS.load()["numpy"]
    require(STREAM not in sealed and ENTROPY not in sealed and ENTROPY not in formal.values() and STREAM not in formal.values(),
            "Development stream or entropy collides with a sealed or formal value")
    require(ENTROPY in GS.DEVELOPMENT_VALUES and STREAM in GS.DEVELOPMENT_VALUES, "Development values must be the audited ones")
    require(pi_ref.gains_sha256() == pi_ref.GAINS_SHA256, "pi_ref gains differ from the pin")
    bounds = json.loads((FORMAL / "d4_bounds.json").read_text(encoding="utf-8"))
    seeds = {"numpy": {"random_control": F.TEST_SEEDS["random_control"]}}
    budget = G.Budget(G.STEP_CAP, G.SECONDS_CAP)
    started = datetime.now(timezone.utc).isoformat()
    out = {"schema": "knee_s10_candidate_v2.g2_development.v1", "candidate": "knee_s10_v2", "sigma": G.SIGMA,
           "mode": "exact (pi_ref v2)",
           "status": "DEVELOPMENT screen under G2_DEV_RULE_v2.md on v1's development entropy and stream (outcomes of "
                     "v1's screens known); not a formal G2; never confirmatory",
           "rule": RULE.name, "rule_sha256": sha(RULE), "code_sha256": {k: sha(p) for k, p in CODE.items()},
           "prefix_rule": PR.RULE, "pi_ref_gains_sha256": pi_ref.gains_sha256(), "entropy": ENTROPY, "stream": STREAM,
           "episodes": EPISODES, "started_utc": started, "headline": "INCOMPLETE", "all_labels": ["INCOMPLETE"],
           "inconclusive_cause": None}
    try:
        data = {}
        for world in G.WORLDS:
            budget.charge(f"collection_{world}", EPISODES * 100)
            data[world] = D.generate(episodes=EPISODES, entropy=ENTROPY, world=world, provenance="smoke")
        out["collection"] = {w: {"sha256": ds.digest, "episodes": EPISODES, "entropy": ENTROPY, "provenance": "smoke",
                                 "recorder_sha256": ds.metadata["recorder_sha256"]} for w, ds in data.items()}
        budget.charge("anchor_prefixes_both_worlds", 2 * sum(E.stream_item(ANCHOR_BANK, i, STREAM)["t"] for i in range(ANCHORS)))
        anchors = [PR.build_anchor(ANCHOR_BANK, i, STREAM) for i in range(ANCHORS)]
        gated = [a for a in anchors if a["t"] in GS.GATED_T]
        require(len(gated) == 48, f"Expected 48 gated anchors, got {len(gated)}")
        out["P1"] = {"anchors_built": len(anchors), "gated": len(gated),
                     "all_both_worlds_identical": all(a["both_worlds_identical"] for a in anchors),
                     "prefix_commands_past_knee": int(sum(int(np.sum(G.past_knee_a0(a["prefix_actions"])[0])) for a in anchors if a["t"] > 0)),
                     "ok": True}
        s_rows = start_rows()
        starts = G.start_obs(s_rows)
        out["D"] = G.d_checks(data, starts, gated, bounds, seeds)
        out["H"] = G.h_checks(gated, budget)
        budget.charge("returns", 2 * len(s_rows) * 100)
        starts_x = np.stack([qpi.raw_anchor_state(r) for r in s_rows])
        out["H"]["reported"]["mean_return_e2_starts_stall3"] = {
            "pi_ref_v2": G.mean_return(pi_ref.act, starts_x),
            "zero": G.mean_return(lambda o: np.zeros((len(o), 4), np.float32), starts_x)}
        by_time = d3_by_time(data, gated)
        out["reported_pairing_with_v1_development"] = pairing(out["D"], out["H"], by_time)
        headline, all_labels, cause = G.labels(out["D"], out["H"])
        out.update(headline=headline, all_labels=all_labels, inconclusive_cause=cause)
    except Exception:                                        # noqa: BLE001  always leave a record
        out["error"] = traceback.format_exc()
        out.update(headline="INCOMPLETE", all_labels=["INCOMPLETE (" + out["error"].strip().splitlines()[-1] + ")"])
    out["accounting"] = {"simulator_steps_exact": budget.used, "by_unit": budget.ledger, "step_cap": G.STEP_CAP,
                         "seconds": time.monotonic() - budget.start, "seconds_cap": G.SECONDS_CAP}
    out["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out["ledger_entry"] = ("Candidate knee_s10_v2 (v1 with the in-support prefix rule), sigma s10, mode exact (pi_ref v2); "
                           f"development G2 v2: {out['headline']}")
    OUT.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    summary = G.summary({**out, "attempt": "development"})
    summary["P1"] = out.get("P1")
    summary["D3"] = {w: out["D"][w]["D3"] for w in out["D"]} if "D" in out else None
    summary["pairing_with_v1_development"] = out.get("reported_pairing_with_v1_development")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
