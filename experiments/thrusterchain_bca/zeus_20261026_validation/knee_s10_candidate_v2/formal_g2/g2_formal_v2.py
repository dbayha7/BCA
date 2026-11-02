"""The formal G2 of candidate knee_s10_v2 (rule: G2_FORMAL_RULE_v2.md, written first). User-run, after the user's own
approval (G2_FORMAL_APPROVAL.json) naming the sealed manifest. Exact computation only: no training, no checkpoint.

  PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/formal_g2/g2_formal_v2.py --execute

Measurements, the D4 bounds check and the label code are v1's sealed formal driver
(../../knee_s10_candidate_v1/formal_g2/g2_formal.py), imported unchanged and hashed in the manifest; its preflight and
execute are never called. This file owns what differs in v2: the seeds (g2_seeds_v2), the anchors built with the v2
prefix rule in both worlds (P1), the preflight, the report and the ledger entry.

Writes formal_g2_run/attempt_<k>/report.json (always, INCOMPLETE with the traceback if anything raises), the two
datasets (*.npz, git-ignored) and collection.json with their digests.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CAND = HERE.parent
V1 = CAND.parent / "knee_s10_candidate_v1"
V1F = V1 / "formal_g2"
for _p in (str(V1F), str(CAND), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import g2_formal as G                                       # noqa: E402  v1's sealed driver; replaces `recorder` first
import numpy as np                                          # noqa: E402

import dataset_knee as D                                    # noqa: E402
import g2_seeds_v2 as GS2                                   # noqa: E402
import pi_ref                                               # noqa: E402
import prefix_rule as PR                                    # noqa: E402
import qpi                                                  # noqa: E402
import worlds                                               # noqa: E402
from core import require                                    # noqa: E402
from knee_region import past_knee_a0                        # noqa: E402

RULE, MANIFEST, APPROVAL = HERE / "G2_FORMAL_RULE_v2.md", HERE / "G2_FORMAL_MANIFEST.json", HERE / "G2_FORMAL_APPROVAL.json"
BOUNDS, RUN = HERE / "d4_bounds.json", HERE / "formal_g2_run"
BOUNDS_SHA = "060460daae462b8202dd36d279be0f95479f6e77ac9668892246457ca7f8f082"      # v1's sealed bounds, byte for byte
MEASUREMENTS = V1F / "g2_formal.py"
WORDS = "I approve the formal G2 for candidate knee_s10_v2 under OOD definition v4 r3, scoring all 96 gated anchors"
ANCHOR_BANK, START_BANK = "g2_e2_anchors", "g2_e2_bank"
WORLDS = G.WORLDS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_files():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {name: digest for section in ("files", "upstream") for name, digest in manifest[section].items()}


def verify_manifest():
    changed = [name for name, digest in manifest_files().items() if sha(HERE / name) != digest]
    require(not changed, "Sealed file changed: " + ", ".join(changed))
    return sha(MANIFEST)


def attempts():
    if not RUN.is_dir():
        return []
    return sorted((p for p in RUN.iterdir() if p.is_dir() and p.name.startswith("attempt_")), key=lambda p: int(p.name[8:]))


def check_approval(approval, manifest_sha, previous_reports):
    """The attempt rules of G2_FORMAL_RULE_v2.md on an approval dict and the earlier attempts' reports (pure)."""
    require(WORDS in approval.get("approval", ""), "Approval wording missing")
    require(approval.get("package_manifest_sha256") == manifest_sha, "Approval names a different package")
    attempt = approval.get("attempt")
    require(type(attempt) is int and attempt == len(previous_reports) + 1, f"Approval must name attempt {len(previous_reports) + 1}")
    require(attempt <= 2, "There is no attempt 3")
    if previous_reports:
        last = previous_reports[-1]
        retry = last.get("headline") == "INCOMPLETE" or (last.get("headline") == "INCONCLUSIVE" and last.get("inconclusive_cause") == "H1")
        require(retry, f"No re-attempt after {last.get('headline')} (G2_FORMAL_RULE_v2.md, Attempts)")
        require(len({r.get("manifest_sha256") for r in previous_reports} | {manifest_sha}) <= 2, "At most one code-only fix")
    return attempt


def preflight():
    require(APPROVAL.is_file(), "G2_FORMAL_APPROVAL.json is missing: the user's own approval is required")
    manifest_sha = verify_manifest()
    approval = json.loads(APPROVAL.read_text(encoding="utf-8"))
    previous = []
    for p in attempts():
        require((p / "report.json").is_file(), f"{p.name} has no report")
        previous.append(json.loads((p / "report.json").read_text(encoding="utf-8")))
    attempt = check_approval(approval, manifest_sha, previous)
    seeds = GS2.load()
    bounds = json.loads(BOUNDS.read_text(encoding="utf-8"))
    require(bounds["schema"] == "knee_s10_candidate_v1.formal_g2.d4_bounds.v1" and set(bounds["gated"]) == {"dens_mid", "dens_deep"},
            "d4_bounds.json is not the sealed D4 bounds")
    require(sha(BOUNDS) == BOUNDS_SHA == sha(V1F / "d4_bounds.json"), "d4_bounds.json is not v1's sealed bounds byte for byte")
    import stage1_smoke
    sealed = stage1_smoke.sealed_values()
    v1_formal = set(G.GS.load()["numpy"].values())
    require(not (set(seeds["numpy"].values()) & (sealed | set(GS2.DEVELOPMENT_VALUES) | v1_formal)),
            "Formal seed equals a sealed, development or v1 formal value")
    require(pi_ref.gains_sha256() == pi_ref.GAINS_SHA256, "pi_ref gains differ from the pin")
    return {"approval": approval, "attempt": attempt, "manifest_sha256": manifest_sha, "seeds": seeds, "bounds": bounds,
            "rule_sha256": sha(RULE), "seeds_sha256": sha(GS2.SEEDS_PATH), "bounds_sha256": sha(BOUNDS),
            "measurements_sha256": sha(MEASUREMENTS), "prefix_rule_sha256": sha(CAND / "prefix_rule.py")}


# ----------------------------------------------------------------------------------------------- the anchors (P1)

def run_prefix_both_worlds(reset_seed, current, uniforms):
    """The v2 prefix from one reset, run in stall3 and nostall; refuses unless the two agree bit for bit and every
    command is inside the retained support and not past the knee (P1). Returns the stall3 run."""
    t = int(len(uniforms))
    runs = {}
    for world in WORLDS:
        env = worlds.environment(world)
        try:
            obs, _ = env.reset(seed=int(reset_seed), options={"start": {"current": tuple(current)}})
            prefix, trace = np.zeros((t, 4), np.float32), [np.asarray(obs, np.float32).copy()]
            for step in range(t):
                prefix[step] = PR.executed_command(obs, uniforms[step])
                obs, _, terminated, truncated, _ = env.step(prefix[step])
                require(not (terminated or truncated), "Prefix reached the horizon")
                trace.append(np.asarray(obs, np.float32).copy())
            runs[world] = {"prefix": prefix, "trace": np.stack(trace), "state": np.asarray(env.state, np.float64).copy()}
        finally:
            env.close()
    a, b = runs["stall3"], runs["nostall"]
    require(np.array_equal(a["prefix"], b["prefix"]) and np.array_equal(a["trace"], b["trace"])
            and np.array_equal(a["state"], b["state"]), "The prefix differs between the worlds (P1)")
    PR.check_prefix(a["trace"][:-1], a["prefix"])
    return a


def build_anchor(item):
    """An anchor from its sealed draw rule: reset, then the v2 prefix for t steps, identical in both worlds."""
    spec = GS2.bank_item(ANCHOR_BANK, item)
    t = spec["t"]
    run = run_prefix_both_worlds(spec["reset_seed"], spec["current"], GS2.prefix_uniforms(ANCHOR_BANK, item, t))
    return {"bank": ANCHOR_BANK, "item": item, "t": t, "reset_seed": spec["reset_seed"], "current": spec["current"],
            "prefix_actions": run["prefix"], "obs": run["trace"][-1], "state": run["state"], "both_worlds_identical": True}


def start_rows():
    rows = []
    for i in range(GS2.BANKS[START_BANK]["count"]):
        spec = GS2.bank_item(START_BANK, i)
        rows.append({"reset_seed": spec["reset_seed"], "current": tuple(spec["current"]), "prefix_actions": np.zeros((0, 4), np.float32)})
    return rows


# ------------------------------------------------------------------------------------------------------ the driver

def execute():
    pre = preflight()
    attempt_dir = RUN / f"attempt_{pre['attempt']}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    budget = G.Budget(G.STEP_CAP, G.SECONDS_CAP)
    report = {"schema": "knee_s10_candidate_v2.formal_g2.report.v1", "candidate": "knee_s10_v2", "sigma": G.SIGMA,
              "mode": "exact (pi_ref v2)", "rule": RULE.name, "rule_sha256": pre["rule_sha256"],
              "manifest_sha256": pre["manifest_sha256"], "seeds_sha256": pre["seeds_sha256"], "bounds_sha256": pre["bounds_sha256"],
              "measurements": {"driver": "../../knee_s10_candidate_v1/formal_g2/g2_formal.py", "sha256": pre["measurements_sha256"]},
              "prefix_rule": {"file": "../prefix_rule.py", "sha256": pre["prefix_rule_sha256"], **PR.RULE},
              "pi_ref_gains_sha256": pi_ref.gains_sha256(), "approval": pre["approval"], "attempt": pre["attempt"],
              "started_utc": started, "headline": "INCOMPLETE", "all_labels": ["INCOMPLETE"], "inconclusive_cause": None}
    try:
        seeds, bounds = pre["seeds"], pre["bounds"]
        entropy = seeds["numpy"]["dataset_entropy"]
        D._sealed_dataset_entropy = lambda: entropy          # research provenance is tied to THIS candidate's sealed entropy
        data, collection = {}, {}
        for world in WORLDS:
            budget.charge(f"collection_{world}", seeds["episodes"] * 100)
            ds = D.generate(episodes=seeds["episodes"], entropy=entropy, world=world, provenance="research")
            path = D.save(ds, attempt_dir / f"knee_s10_v2_{world}.npz")
            collection[world] = {"path": path.name, "sha256": ds.digest, "episodes": seeds["episodes"], "entropy": entropy,
                                 "recorder_sha256": ds.metadata["recorder_sha256"], "behavior": ds.metadata["behavior"]["name"]}
            data[world] = ds
        (attempt_dir / "collection.json").write_text(json.dumps(collection, indent=2) + "\n", encoding="utf-8")
        report["collection"] = collection
        count = GS2.BANKS[ANCHOR_BANK]["count"]
        budget.charge("anchor_prefixes_both_worlds", 2 * sum(GS2.bank_item(ANCHOR_BANK, i)["t"] for i in range(count)))
        all_anchors = [build_anchor(i) for i in range(count)]                   # P1 holds for every anchor, gated or not
        items = GS2.gated_items(ANCHOR_BANK)
        require(len(items) == 96, f"Expected 96 gated anchors, got {len(items)}")
        anchors = [all_anchors[i] for i in items]
        report["P1"] = {"anchors_built": len(all_anchors), "gated": len(anchors),
                        "all_both_worlds_identical": all(a["both_worlds_identical"] for a in all_anchors),
                        "prefix_commands_past_knee": int(sum(int(np.sum(past_knee_a0(a["prefix_actions"])[0])) for a in all_anchors if a["t"] > 0)),
                        "ok": True}
        s_rows = start_rows()
        starts = G.start_obs(s_rows)
        report["D"] = G.d_checks(data, starts, anchors, bounds, seeds)
        report["H"] = G.h_checks(anchors, budget)
        budget.charge("returns", 2 * len(s_rows) * 100)
        starts_x = np.stack([qpi.raw_anchor_state(r) for r in s_rows])
        report["H"]["reported"]["mean_return_e2_starts_stall3"] = {
            "pi_ref_v2": G.mean_return(pi_ref.act, starts_x), "zero": G.mean_return(lambda o: np.zeros((len(o), 4), np.float32), starts_x)}
        headline, all_labels, cause = G.labels(report["D"], report["H"])
        report.update(headline=headline, all_labels=all_labels, inconclusive_cause=cause)
    except Exception:                                        # noqa: BLE001  always leave a report
        report["error"] = traceback.format_exc()
        report.update(headline="INCOMPLETE", all_labels=["INCOMPLETE (" + report["error"].strip().splitlines()[-1] + ")"])
    try:
        report["integrity_after"] = {"manifest_sha256": verify_manifest(), "ok": True}
    except Exception as exc:                                 # noqa: BLE001
        report["integrity_after"] = {"ok": False, "error": str(exc)}
        report.update(headline="INCOMPLETE", all_labels=report["all_labels"] + ["INCOMPLETE (integrity after the run)"])
    report["accounting"] = {"simulator_steps_exact": budget.used, "by_unit": budget.ledger, "step_cap": G.STEP_CAP,
                            "seconds": time.monotonic() - budget.start, "seconds_cap": G.SECONDS_CAP}
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["ledger_entry"] = ("Candidate knee_s10_v2 (v1 with the in-support anchor prefix), sigma s10, mode exact (pi_ref v2); "
                              f"development G2 v2 PASS; formal G2 attempt {pre['attempt']}: {report['headline']}")
    (attempt_dir / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    return report


def summary(report):
    out = G.summary(report)
    out["P1"] = report.get("P1")
    if "D" in report:
        out["D3"] = {w: report["D"][w]["D3"] for w in report["D"]}
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(summary(execute()), indent=1, default=float))
