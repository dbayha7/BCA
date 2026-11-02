"""The formal G2 of candidate knee_s10_v1 (rule: G2_FORMAL_RULE.md, written first). User-run, after the user's own
approval (G2_FORMAL_APPROVAL.json) naming the sealed manifest. Exact computation only: no training, no checkpoint.

  PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B knee_s10_candidate_v1/formal_g2/g2_formal.py --execute

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
V3 = CAND.parent / "knee_edge_ood_v3"
for _p in (str(V3), str(CAND), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if "recorder" in sys.modules:
    raise SystemExit("A recorder module was imported before the widened recorder could replace it")
import recorder_wide                                        # noqa: E402  must replace `recorder` before any v3 import
sys.modules["recorder"] = recorder_wide

import numpy as np                                          # noqa: E402

import dataset_knee as D                                    # noqa: E402
import g2_seeds as GS                                       # noqa: E402
import masks as M                                           # noqa: E402
import pi_ref                                               # noqa: E402
import probes as P                                          # noqa: E402
import qpi                                                  # noqa: E402
import region_independent as RI                             # noqa: E402
import stage1_eval as E                                     # noqa: E402
import stage1_math as SM                                    # noqa: E402
import step04_math as SM4                                   # noqa: E402
import worlds                                               # noqa: E402
from core import require                                    # noqa: E402
from knee_region import GATED_PAST_KNEE, KNEE, S9, SLOTS as SLOT_VALUES, load_band, load_g, past_knee_a0  # noqa: E402

for _m in (D, E):
    require(_m.recorder is recorder_wide, f"{_m.__name__} does not use the widened recorder")

RULE, MANIFEST, APPROVAL = HERE / "G2_FORMAL_RULE.md", HERE / "G2_FORMAL_MANIFEST.json", HERE / "G2_FORMAL_APPROVAL.json"
BOUNDS, RUN = HERE / "d4_bounds.json", HERE / "formal_g2_run"
WORDS = "I approve the formal G2 for candidate knee_s10_v1 under OOD definition v4 r3, scoring all 96 gated anchors"
WORLDS = ("stall3", "nostall")
SIGMA, C_SIGMA, TOL_SIGMA = "s10", 0.3, 0.15
R_REQ = (0.5 + C_SIGMA * TOL_SIGMA) / (1 - C_SIGMA)
D4_WINDOW, B1_BAR, H1_TOL = (-0.95, -0.85), 0.99, 1e-9
D3_BARS = {"start_share": 0.95, "anchor_share": 0.90}
STEP_CAP, SECONDS_CAP = 1_000_000, 1_800
ANCHOR_BANK, START_BANK = "g2_e2_anchors", "g2_e2_bank"
SLOTS = list(E.GATED_SLOTS)
S10 = float(SLOT_VALUES["s10"])
ORDER = ("INCOMPLETE", "INCONCLUSIVE", "NOT DATA-OOD", "NOT TESTABLE", "PASS")


class CapExhausted(RuntimeError):
    pass


class Budget:
    """Exact simulator-step and wall-clock caps; a charge that would cross a cap raises before the steps run."""

    def __init__(self, steps, seconds):
        self.limit, self.seconds, self.used, self.start, self.ledger = steps, seconds, 0, time.monotonic(), {}

    def charge(self, unit, steps):
        if time.monotonic() - self.start > self.seconds:
            raise CapExhausted(f"time cap {self.seconds} s exceeded before {unit}")
        if self.used + int(steps) > self.limit:
            raise CapExhausted(f"step cap {self.limit} would be exceeded by {unit} ({self.used} + {steps})")
        self.used += int(steps)
        self.ledger[unit] = self.ledger.get(unit, 0) + int(steps)


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
    """The attempt rules of G2_FORMAL_RULE.md on an approval dict and the earlier attempts' reports (pure)."""
    require(WORDS in approval.get("approval", ""), "Approval wording missing")
    require(approval.get("package_manifest_sha256") == manifest_sha, "Approval names a different package")
    attempt = approval.get("attempt")
    require(type(attempt) is int and attempt == len(previous_reports) + 1, f"Approval must name attempt {len(previous_reports) + 1}")
    require(attempt <= 2, "There is no attempt 3")
    if previous_reports:
        last = previous_reports[-1]
        retry = last.get("headline") == "INCOMPLETE" or (last.get("headline") == "INCONCLUSIVE" and last.get("inconclusive_cause") == "H1")
        require(retry, f"No re-attempt after {last.get('headline')} (G2_FORMAL_RULE.md, Attempts)")
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
    seeds = GS.load()
    bounds = json.loads(BOUNDS.read_text(encoding="utf-8"))
    require(bounds["schema"] == "knee_s10_candidate_v1.formal_g2.d4_bounds.v1" and set(bounds["gated"]) == {"dens_mid", "dens_deep"},
            "d4_bounds.json is not the sealed D4 bounds")
    import stage1_smoke
    sealed = stage1_smoke.sealed_values()
    require(not (set(seeds["numpy"].values()) & (sealed | set(GS.DEVELOPMENT_VALUES))), "Formal seed equals a sealed or development value")
    require(pi_ref.gains_sha256() == pi_ref.GAINS_SHA256, "pi_ref gains differ from the pin")
    return {"approval": approval, "attempt": attempt, "manifest_sha256": manifest_sha, "seeds": seeds, "bounds": bounds,
            "rule_sha256": sha(RULE), "seeds_sha256": sha(GS.SEEDS_PATH), "bounds_sha256": sha(BOUNDS)}


# ----------------------------------------------------------------------------------------------- the measurements

def build_anchor(item):
    """An anchor from its sealed draw rule: reset, then the widened recorder's commands for t steps in stall3."""
    spec = GS.bank_item(ANCHOR_BANK, item)
    t = spec["t"]
    env = worlds.environment("stall3")
    try:
        obs, _ = env.reset(seed=spec["reset_seed"], options={"start": {"current": spec["current"]}})
        uniforms = GS.prefix_uniforms(ANCHOR_BANK, item, t)
        prefix = np.zeros((t, 4), np.float32)
        for step in range(t):
            prefix[step] = recorder_wide.requested_command(obs, uniforms[step])[0].astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(prefix[step])
            require(not (terminated or truncated), "Prefix reached the horizon")
        return {"bank": ANCHOR_BANK, "item": item, "t": t, "reset_seed": spec["reset_seed"], "current": spec["current"],
                "prefix_actions": prefix, "obs": np.asarray(obs, np.float32)}
    finally:
        env.close()


def start_rows():
    rows = []
    for i in range(GS.BANKS[START_BANK]["count"]):
        spec = GS.bank_item(START_BANK, i)
        rows.append({"reset_seed": spec["reset_seed"], "current": tuple(spec["current"]), "prefix_actions": np.zeros((0, 4), np.float32)})
    return rows


def start_obs(rows):
    out = []
    for r in rows:
        env = worlds.environment("stall3")
        try:
            obs, _ = env.reset(seed=r["reset_seed"], options={"start": {"current": r["current"]}})
        finally:
            env.close()
        out.append(np.asarray(obs, np.float32))
    return np.stack(out)


def in_cutoff_support(obs, a):
    """Per row: is command a inside CUTOFF's retained data for the load band of obs (B1)?"""
    high = load_band(load_g(obs)) >= 1
    a0, side = a[:, 0].astype(np.float64), np.abs(a[:, 1:].astype(np.float64))
    hi = np.where(high, float(np.float32(-0.6)), KNEE)
    return (a0 >= -KNEE) & (a0 <= hi) & np.all(side <= 0.7 + 1e-7, axis=1)


def d4_check(rows_bands, a0, bounds):
    lo, hi = D4_WINDOW
    out = {}
    for band, name in ((1, "mid"), (2, "deep"), (None, "hi")):
        in_band = rows_bands >= 1 if band is None else rows_bands == band
        count = int(np.sum(in_band & (a0 >= lo) & (a0 <= hi)))
        density = count / (int(in_band.sum()) * (hi - lo)) if in_band.any() else 0.0
        b = bounds["bounds"][f"dens_{name}"]
        out[name] = {"band_rows": int(in_band.sum()), "rows_in_window": count, "density": float(density),
                     "lower": b["lower"], "upper": b["upper"], "below": bool(density < b["lower"]), "above": bool(density > b["upper"]),
                     "gated": name != "hi"}
    return out


def d_checks(data, starts, anchors, bounds, seeds):
    out = {}
    for world, ds in data.items():
        arrays = ds.arrays
        require(ds.metadata["recorder_sha256"] == sha(CAND / "recorder_wide.py")
                and ds.metadata["behavior"]["name"] == "knee_s10_wide_window_v1", "Dataset was not made by the widened recorder")
        rows = M.fitting_rows(arrays)
        a0 = rows["actions"][:, 0].astype(np.float64)
        past = past_knee_a0(rows["actions"])[0]
        independent = RI.withheld_region(rows["actions"])[0]
        arm, _ = M.arm_masks(rows["obs"], rows["actions"], rows["times"], rows["episode_ids"], seed=seeds["numpy"]["random_control"])
        cutoff = arm["cutoff"]
        high = rows["bands"] >= 1
        deep_ep = np.repeat(np.asarray(arrays["reset_component"])[rows["episodes"]] == 1, 100)
        a_obs, a_t = np.stack([a["obs"] for a in anchors]), np.asarray([a["t"] for a in anchors])
        l15 = SM4.l15(rows["obs"], rows["episode_ids"], rows["times"], deep_ep, starts, a_obs, a_t)
        d4 = d4_check(rows["bands"], a0, bounds)
        nxt = np.asarray(arrays["observations"])[rows["episodes"], 1:100].reshape(-1, 14)
        b1 = float(np.mean(in_cutoff_support(nxt, pi_ref.act(nxt))))
        retained_high = a0[cutoff & high]
        out[world] = {
            "D1": {"cutoff_W_rows": int(np.sum(cutoff & past)), "predicates_agree": bool(np.array_equal(past, independent)),
                   "cutoff_is_complement_of_W": bool(np.array_equal(cutoff, ~past)),
                   "ok": bool(np.sum(cutoff & past) == 0 and np.array_equal(past, independent) and np.array_equal(cutoff, ~past))},
            "D2": {"cutoff_high_load_a0_at_or_past_s10": int(np.sum(retained_high <= S10)),
                   "min_distance_to_s10": float(np.min(np.abs(retained_high - S10))),
                   "ok": bool(np.sum(retained_high <= S10) == 0 and np.min(np.abs(retained_high - S10)) >= 0.0999)},
            "D3": {"start_share": l15["start_share"], "anchor_share": l15["anchor_share"], "bars": D3_BARS,
                   "ok": bool(l15["start_share"] >= D3_BARS["start_share"] and l15["anchor_share"] >= D3_BARS["anchor_share"])},
            "D4": {**d4, "below_lower": bool(any(v["below"] for v in d4.values() if v["gated"])),
                   "above_upper": bool(any(v["above"] for v in d4.values() if v["gated"]))},
            "B1": {"share_inside_cutoff_support": b1, "next_states": int(len(nxt)), "bar": B1_BAR, "ok": bool(b1 >= B1_BAR)},
            "reported": {"fitting_rows": int(len(a0)), "high_load_rows": int(high.sum()), "share_high": float(high.mean()),
                         "share_deep": float(np.mean(rows["bands"] == 2)),
                         "high_load_depth_ge_0p1_rows": int(np.sum(high & (a0 <= S10))),
                         "twin_W_rows": int(past.sum()), "cutoff_rows": int(cutoff.sum()),
                         "l15_general_threshold": {k: v for k, v in l15.items() if k not in ("start_share", "anchor_share", "pass")},
                         "dataset_sha256": ds.digest},
        }
    return out


def h_checks(anchors, budget):
    rows = [{"reset_seed": a["reset_seed"], "current": tuple(a["current"]), "prefix_actions": a["prefix_actions"]} for a in anchors]
    probes = P.build(len(rows), SLOTS)
    steps = int(sum(100 - a["t"] for a in anchors)) * len(SLOTS)
    budget.charge("continuations_stall3", steps)
    q_st, calls, receipt = qpi.raw_values(rows, probes, pi_ref.act, "stall3", clip_all=True)
    budget.charge("continuations_nostall", steps)
    q_ns, _, receipt_ns = qpi.raw_values(rows, probes, pi_ref.act, "nostall", clip_all=True)
    ix = {n: SLOTS.index(n) for n in SLOTS}
    rel = max(float(np.max(np.abs(q_st[:, ix[n]] - q_ns[:, ix[n]]) / np.maximum(1.0, np.abs(q_st[:, ix[n]])))) for n in S9)
    rho = {n: q_ns[:, ix[n]] - q_st[:, ix[n]] for n in ("s05", "s10", "s20", "p10")}
    med = lambda x: float(np.median(x)) if np.size(x) else None          # noqa: E731
    budget.charge("projection_readout", len(rows))
    env = worlds.load_environment()
    ph = worlds.physics("stall3")
    x0 = np.stack([qpi.raw_anchor_state(r) for r in rows])
    t0 = np.asarray([len(r["prefix_actions"]) for r in rows])
    x1, _ = env.dynamics(x0, np.tile(np.float64(S10), (len(rows), 1)) * np.array([1, 0, 0, 0]), ph)
    obs1 = np.concatenate([x1, ((100 - (t0 + 1)) / 100)[:, None]], axis=1).astype(np.float32)
    raw1 = pi_ref.raw(obs1)
    a0_projected = raw1[:, 0] < -KNEE
    res = SM4.neighbor_residuals(q_st[:, [ix[n] for n in S9]], {n: q_st[:, ix[n]] for n in GATED_PAST_KNEE},
                                 {n: q_ns[:, ix[n]] for n in ("s05", "s10", "s20")})
    preds = SM4.predictors(q_st[:, [ix[n] for n in S9]])
    pv = {k: {s: f(float(SLOT_VALUES[s])) for s in ("s10", "p10")} for k, f in preds.items()}
    min_form = np.abs(np.minimum.reduce([pv[k]["s10"] for k in pv]) - q_ns[:, ix["s10"]]) <= TOL_SIGMA
    h2 = med(rho[SIGMA])
    finite = bool(np.all(np.isfinite(q_st)) and np.all(np.isfinite(q_ns)))
    return {
        "H1": {"max_rel_stall_minus_nostall_on_S9": rel, "tol": H1_TOL, "all_finite": finite, "ok": bool(finite and rel <= H1_TOL)},
        "H2": {"median_exact_rho_s10": h2, "n_anchors": len(rows), "scoring_set": "all gated anchors", "R_req": R_REQ,
               "ok": bool(finite and h2 is not None and h2 >= R_REQ)},
        "reported": {
            "median_exact_rho": {n: med(rho[n]) for n in rho},
            "quantiles_exact_rho_s10": {str(q): float(np.quantile(rho["s10"], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "rho_s10_by_time": {str(t): med(rho["s10"][[a["t"] == t for a in anchors]]) for t in GS.GATED_T},
            "rho_s10_split_by_a0_projection_at_first_continuation_state": {
                "projected_share": float(np.mean(a0_projected)), "median_projected": med(rho["s10"][a0_projected]),
                "median_not_projected": med(rho["s10"][~a0_projected]),
                "any_side_projected_share": float(np.mean(np.any(np.abs(raw1[:, 1:]) > 0.7, axis=1)))},
            "predictors": {"item5_max_form_share": float(np.mean(SM.item5_ok(res["item5_error"]))),
                           "item5_min_form_share_s10": float(np.mean(min_form)),
                           "median_rho_hat": {n: med(res["rho_hat"][n]) for n in ("s05", "s10", "s20")}},
            "item6_sign_control": {
                "max_P1_P3_share_negative": float(np.mean(np.maximum(pv["P1"]["p10"], pv["P3"]["p10"]) - q_st[:, ix["p10"]] < 0)),
                "P2_share_negative_outside_fit_range": float(np.mean(pv["P2"]["p10"] - q_st[:, ix["p10"]] < 0)),
                "exact_share_negative": float(np.mean(rho["p10"] < 0))},
            "median_abs_Q_at_edge_knee": med(np.abs(q_st[:, ix["edge_knee"]])),
            "forward_calls": int(calls), "receipts": [receipt, receipt_ns]},
    }


def mean_return(policy, starts_x):
    env = worlds.load_environment()
    ph = worlds.physics("stall3")
    x, total = starts_x.copy(), np.zeros(len(starts_x))
    for t in range(100):
        obs = np.concatenate([x, np.full((len(x), 1), (100 - t) / 100)], axis=1).astype(np.float32)
        x, r = env.dynamics(x, policy(obs).astype(np.float64), ph)
        total += 0.98 ** t * r
    return float(total.mean())


def labels(d, h):
    """The rule's labels from the D and H check dicts (pure). Returns (headline, all labels, inconclusive cause)."""
    out, cause = [], None
    if not h["H1"]["ok"]:
        out.append("INCONCLUSIVE (H1: the stall = no-stall identity fails; an instrument fault)")
        cause = "H1"
    above = [w for w in d if d[w]["D4"]["above_upper"]]
    if above:
        out.append("INCONCLUSIVE (D4 above its upper bound in " + ", ".join(above) + ": the realized data are not the designed condition)")
        cause = cause or "D4"
    bad_d = [f"{w} {k}" for w in d for k in ("D1", "D2", "D3") if not d[w][k]["ok"]]
    if bad_d:
        out.append("NOT DATA-OOD (" + ", ".join(bad_d) + ")")
    bad_t = [f"{w} D4" for w in d if d[w]["D4"]["below_lower"]] + [f"{w} B1" for w in d if not d[w]["B1"]["ok"]] \
        + ([] if h["H2"]["ok"] else ["H2"])
    if bad_t:
        out.append("NOT TESTABLE (" + ", ".join(bad_t) + ")")
    all_labels = out or ["PASS"]
    headline = min(all_labels, key=lambda s: ORDER.index(s.split(" (")[0]))
    return headline.split(" (")[0], all_labels, cause


# ------------------------------------------------------------------------------------------------------ the driver

def execute():
    pre = preflight()
    attempt_dir = RUN / f"attempt_{pre['attempt']}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    budget = Budget(STEP_CAP, SECONDS_CAP)
    report = {"schema": "knee_s10_candidate_v1.formal_g2.report.v1", "candidate": "knee_s10_v1", "sigma": SIGMA,
              "mode": "exact (pi_ref v2)", "rule": RULE.name, "rule_sha256": pre["rule_sha256"],
              "manifest_sha256": pre["manifest_sha256"], "seeds_sha256": pre["seeds_sha256"], "bounds_sha256": pre["bounds_sha256"],
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
            path = D.save(ds, attempt_dir / f"knee_s10_v1_{world}.npz")
            collection[world] = {"path": path.name, "sha256": ds.digest, "episodes": seeds["episodes"], "entropy": entropy,
                                 "recorder_sha256": ds.metadata["recorder_sha256"], "behavior": ds.metadata["behavior"]["name"]}
            data[world] = ds
        (attempt_dir / "collection.json").write_text(json.dumps(collection, indent=2) + "\n", encoding="utf-8")
        report["collection"] = collection
        items = GS.gated_items(ANCHOR_BANK)
        require(len(items) == 96, f"Expected 96 gated anchors, got {len(items)}")
        budget.charge("anchor_prefixes", sum(GS.bank_item(ANCHOR_BANK, i)["t"] for i in items))
        anchors = [build_anchor(i) for i in items]
        s_rows = start_rows()
        starts = start_obs(s_rows)
        report["D"] = d_checks(data, starts, anchors, bounds, seeds)
        report["H"] = h_checks(anchors, budget)
        budget.charge("returns", 2 * len(s_rows) * 100)
        starts_x = np.stack([qpi.raw_anchor_state(r) for r in s_rows])
        report["H"]["reported"]["mean_return_e2_starts_stall3"] = {
            "pi_ref_v2": mean_return(pi_ref.act, starts_x), "zero": mean_return(lambda o: np.zeros((len(o), 4), np.float32), starts_x)}
        headline, all_labels, cause = labels(report["D"], report["H"])
        report.update(headline=headline, all_labels=all_labels, inconclusive_cause=cause)
    except Exception:                                        # noqa: BLE001  always leave a report
        report["error"] = traceback.format_exc()
        report.update(headline="INCOMPLETE", all_labels=["INCOMPLETE (" + report["error"].strip().splitlines()[-1] + ")"])
    try:
        report["integrity_after"] = {"manifest_sha256": verify_manifest(), "ok": True}
    except Exception as exc:                                 # noqa: BLE001
        report["integrity_after"] = {"ok": False, "error": str(exc)}
        report.update(headline="INCOMPLETE", all_labels=report["all_labels"] + ["INCOMPLETE (integrity after the run)"])
    report["accounting"] = {"simulator_steps_exact": budget.used, "by_unit": budget.ledger, "step_cap": STEP_CAP,
                            "seconds": time.monotonic() - budget.start, "seconds_cap": SECONDS_CAP}
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["ledger_entry"] = ("Candidate knee_s10_v1, sigma s10, mode exact (pi_ref v2); development G2 v1 INCONCLUSIVE, "
                              f"development G2 r3 PASS; formal G2 attempt {pre['attempt']}: {report['headline']}")
    (attempt_dir / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    return report


def summary(report):
    keys = ("headline", "all_labels", "inconclusive_cause", "attempt", "accounting")
    out = {k: report.get(k) for k in keys}
    if "D" in report:
        out["D"] = {w: {**{k: report["D"][w][k]["ok"] for k in ("D1", "D2", "D3", "B1")},
                        "D4": {"below": report["D"][w]["D4"]["below_lower"], "above": report["D"][w]["D4"]["above_upper"]}}
                    for w in report["D"]}
        out["D4_density"] = {w: {n: report["D"][w]["D4"][n]["density"] for n in ("mid", "deep", "hi")} for w in report["D"]}
    if "H" in report:
        out["H1"], out["H2"] = report["H"]["H1"], report["H"]["H2"]
        out["reported"] = {k: report["H"]["reported"].get(k) for k in (
            "median_exact_rho", "quantiles_exact_rho_s10", "rho_s10_by_time",
            "rho_s10_split_by_a0_projection_at_first_continuation_state", "mean_return_e2_starts_stall3")}
    if "error" in report:
        out["error"] = report["error"]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(summary(execute()), indent=1, default=float))
