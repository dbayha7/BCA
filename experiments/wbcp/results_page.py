"""Build one tab-organized HTML page with every semi-synthetic benchmark result and its plots.

Reads the result artifacts under runs/ (Changes 1-11 of DEPENDENCE.md), embeds them as JSON,
and renders charts with Plotly (loaded from cdnjs). Rerun after new experiments:

python experiments/wbcp/results_page.py --output <new or existing .html path>
"""

import argparse
import datetime
import glob
import hashlib
import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = os.path.join(ROOT, "runs")
HOP = "wbcp_dependence/hopper-u100000"
SPA = "wbcp_dependence/hopper-u100000-spacing"
RES = "wbcp_dependence/hopper-u100000-reservation"
OTH = "wbcp_dependence/other-datasets"
WEI = "wbcp_dependence/hopper-u100000-weighting"

# (file pattern, change, label); order is the page's order
EXPERIMENTS = [
    ("wbcp_bench/blocks.json", "1", "Whole-episode banks, every tilt"),
    ("wbcp_bench/main.json", "5", "Shift baseline, independent rows"),
    ("wbcp_bench/rawdisc.json", "5", "Misspecified weight model"),
    (f"{HOP}/e0_iid.json", "4", "Independent rows, no shift"),
    (f"{HOP}/e1_blocks.json", "3", "Whole-episode banks by size"),
    (f"{HOP}/e2_k*.json", "4", "Random K rows per episode"),
    (f"{HOP}/e3_k*.json", "5", "Random K under shift"),
    (f"{SPA}/e6_strat_k*.json", "6", "Spaced K rows per episode"),
    (f"{SPA}/e7_*.json", "7", "K = 10 under shift, spaced vs random"),
    (f"{SPA}/e8_*.json", "8", "Spaced K = 5 under shift"),
    (f"{RES}/r[0-9]_*.json", "9", "BCA's own sampler"),
    (f"{OTH}/w_*.json", "10", "walker2d-medium-replay"),
    (f"{OTH}/p_*.json", "10", "pen-cloned"),
    (f"{OTH}/d_*.json", "10", "Weight-fit follow-up"),
    (f"{WEI}/a_*.json", "11", "Bank-size sweep under strong shift"),
    (f"{WEI}/b_*.json", "11", "Shuffled-weight control"),
]


def load(rel):
    with open(os.path.join(RUNS, rel), encoding="utf-8") as handle:
        return json.load(handle)


def dataset_of(frozen):
    for key, name in (("walker2d", "walker2d-medium-replay"), ("pen-cloned", "pen-cloned"), ("hopper", "hopper-medium")):
        if key in frozen:
            return name
    return frozen


def design_of(s):
    if s.get("blocks"):
        text = "Whole episodes"
    elif s.get("per_episode"):
        spacing = s.get("spacing") or "random"
        text = f"K={s['per_episode']} " + {"random": "random", "stratified": "spaced", "reservation": "BCA sampler"}[spacing]
    else:
        text = "Independent rows"
    if s.get("shuffle_tilt") is not None:
        text += ", weights shuffled"
    if s.get("weight_fit", 1000) != 1000:
        text += f", weight fit {s['weight_fit']}"
    if s.get("discriminator") == "raw":
        text += ", misspecified weights"
    return text


def arm(a):
    keys = dict(f="failures", T="trials", lo=None, hi=None, r="risk", rc="risk_certified", q95="risk_q95",
                q99="risk_q99", x="excess_given_fail", th="threshold", ab="abstain")
    out = {k: a.get(v) for k, v in keys.items() if v}
    out["lo"], out["hi"] = a["ci"]
    return out


def experiments():
    exps, blocks = [], []
    for pattern, change, label in EXPERIMENTS:
        for path in sorted(glob.glob(os.path.join(RUNS, pattern)), key=lambda p: (len(p), p)):
            rel = os.path.relpath(path, RUNS).replace("\\", "/")
            data = load(rel)
            if "settings" not in data or "blocks" not in data:
                continue
            s = data["settings"]
            eid = rel.replace("wbcp_dependence/", "").replace(".json", "")
            exps.append(dict(id=eid, change=change, label=label, file=rel, dataset=dataset_of(s["frozen"]),
                             design=design_of(s), trials=s["trials"], seed=s["seed"]))
            for b in data["blocks"]:
                blocks.append(dict(e=eid, n=b["n"], t=b["tilt"], y=b["gamma"], s=b["score"], L=b["lambda_star"],
                                   ne=b["calibration_n_eff"]["oracle"], nh=b["calibration_n_eff"]["estimated"],
                                   cs=b["calibration_size"]["mean"], wb=b["oracle_wbar"],
                                   A={k: arm(v) for k, v in b["arms"].items()}))
    return exps, blocks


def predictions(rel):
    data = load(rel + "/predictions.json") if not rel.endswith(".json") else load(rel)
    scores = {}
    for name, entry in data["scores"].items():
        scores[name] = dict(lambda_star=entry["lambda_star"], rho=entry["icc"], rho_score=entry["icc_score"],
                            lags={str(k): v for k, v in entry["lag_correlation"].items()},
                            designs=[dict(design=d["design"], n=d["n"], size=d["mean_bank_size"], Dicc=d["design_effect_icc"],
                                          Dmc=d["design_effect_mc"], picc=d["predicted_failure_icc"],
                                          pmc=d["predicted_failure_mc"], eps=d["episodes_reserved"],
                                          share=d["reserved_share_of_dataset"]) for d in entry["designs"]])
    return dict(created=data["created_utc"], pool=data["pool"], dataset=data["dataset"], scores=scores)


def sha(rel):
    with open(os.path.join(RUNS, rel), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def first_line(rel):
    with open(os.path.join(RUNS, rel), encoding="utf-8") as handle:
        return handle.readline().strip().lstrip("# ")


def build():
    exps, blocks = experiments()
    pools = {}
    for key, rel in (("hopper-medium", "wbcp_frozen/hopper-medium-v2-s202609171-u100000/frozen.json"),
                     ("walker2d-medium-replay", "wbcp_frozen/walker2d-s202609171-u100000/frozen.json"),
                     ("pen-cloned", "wbcp_frozen/pen-cloned-s202609171-u100000/frozen.json")):
        meta = load(rel)
        pools[key] = dict(rows=meta["rows"], updates=meta.get("updates"))
    pred = {
        "hopper": predictions(f"{HOP}/predictions.json"),
        "hopper-spacing": predictions(f"{SPA}/predictions.json"),
        "hopper-k5": predictions(f"{RES}/predictions_k5"),
        "hopper-k18": predictions(f"{RES}/predictions_k18"),
        "walker2d-n1024": predictions(f"{OTH}/predictions_walker2d_n1024"),
        "walker2d-n8192": predictions(f"{OTH}/predictions_walker2d_n8192"),
        "pen-n1024": predictions(f"{OTH}/predictions_pen_n1024"),
        "pen-n8192": predictions(f"{OTH}/predictions_pen_n8192"),
    }
    mech = {key: {k: v for k, v in load(f"{WEI}/{name}.json").items() if k != "settings"}
            for key, name in (("aligned", "c_density1"), ("shuffled", "c_shuffled_density1"), ("policy", "c_policy1"))}
    expectations = []
    for rel, label, change in ((f"{HOP}/expectations_shift.md", "Shift re-check under thinned banks", "5"),
                               (f"{SPA}/expectations_shift.md", "Spaced banks under shift", "6-7"),
                               (f"{SPA}/expectations_strat5.md", "Spaced K = 5 under shift", "8"),
                               (f"{RES}/expectations.md", "BCA's thinned reservation", "9"),
                               (f"{OTH}/expectations.md", "Two more datasets (stages A, B, D)", "10"),
                               (f"{WEI}/expectations.md", "Why WBCP exceeds 5% under strong shift", "11")):
        expectations.append(dict(file=rel, label=label, change=change, sha=sha(rel), header=first_line(rel)))
    for name in ("predictions.json",):
        expectations.insert(0, dict(file=f"{HOP}/{name}", label="Design-effect predictions", change="2-4",
                                    sha=sha(f"{HOP}/{name}"), header="created " + load(f"{HOP}/{name}")["created_utc"]))
    return dict(
        generated=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        exps=exps, blocks=blocks, pools=pools, pred=pred, mech=mech,
        slack=load(f"{WEI}/certificate_slack.json")["rows"],
        dominant=load(f"{HOP}/posthoc_dominant_episode.json"),
        tilted_hopper=load(f"{HOP}/posthoc_tilted_design_effect.json"),
        tilted_rho={"walker2d-medium-replay": load(f"{OTH}/posthoc_tilted_rho_walker2d.json")["rows"],
                    "pen-cloned": load(f"{OTH}/posthoc_tilted_rho_pen-cloned.json")["rows"]},
        configs=load(f"{RES}/rows_per_episode.json")["hosts"],
        expectations=expectations,
    )


def clean(value):
    """JSON has no infinity or NaN: store them as null (the page reads null as not finite)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    data = clean(build())
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    with open(os.path.join(os.path.dirname(__file__), "results_page_template.html"), encoding="utf-8") as handle:
        template = handle.read()
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(template.replace("__DATA__", payload))
    print(f"wrote {args.output}: {len(data['blocks'])} result blocks from {len(data['exps'])} files")


if __name__ == "__main__":
    main()
