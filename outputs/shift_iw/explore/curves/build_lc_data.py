"""Pack the learning curves (runs/shift_iw_v5_explore/curves/<quality>/seed<k>/*.json) into one compact JSON block for
the ThrusterChain page. Read only. Missing runs are null, so the page can be built before every job has finished.

    python build_lc_data.py OUT.json        (run in WSL, where the curves live)
"""
import json
import sys
from pathlib import Path

ROOT = Path("/root/bca-shift/runs/shift_iw_v5_explore/curves")
QUALITIES = ("random", "medium-replay", "medium", "medium-expert", "expert")
SEEDS = (0, 1, 2, 3, 4)
LAWS = ("none", "mild", "strong")
ACTOR = {"oracle": ("le_none", "le_oracle", "le_oracle_x1.5"),
         "band": ("v5_unweighted", "v5_exact_strong", "v5_exact_mild"),
         "cmd": ("cmd_strong", "cmd_train", "untargeted"),
         "f1": ("A2_cmd_strong", "B2_cmd_train", "C2_untargeted"),
         "est": ("A3_estimated", "C3_untargeted")}
RENAME = {("cmd", "cmd_train"): "cmd_train", ("cmd", "untargeted"): "cmd_untargeted",
          ("est", "C3_untargeted"): "C3_untargeted"}
COTRAIN = ("host_gap", "host_full", "capped_unweighted", "capped_exact_strong")


def load(q, s, name):
    p = ROOT / q / f"seed{s}" / f"{name}.json"
    return json.loads(p.read_text())["curves"] if p.exists() else None


def pack(entries, steps):
    """entries: per seed, the curve entry or None -> {pts: {law: [per seed list or null]}, ret: ..., match: [...]}"""
    if all(e is None for e in entries):
        return None
    out = {"pts": {}, "ret": {}, "match": [None if e is None else e.get("final_matches_saved") for e in entries]}
    for law in LAWS:
        out["pts"][law] = [None if e is None else [round(v, 2) for v in e["points"][law]] for e in entries]
        out["ret"][law] = [None if e is None else [round(v, 1) for v in e["J"][law]] for e in entries]
    for e in entries:
        if e is not None and e["steps"] != steps:
            raise SystemExit(f"unexpected steps {e['steps'][:3]}... (want {steps[:3]}...)")
    return out


def main(out):
    actor_steps = list(range(0, 20001, 1000))
    co_steps = list(range(5000, 100001, 5000))
    data = {"qualities": QUALITIES, "laws": LAWS, "seeds": SEEDS, "actor": {"steps": actor_steps, "arms": {}},
            "cotrain": {"steps": co_steps, "arms": {}}, "missing": []}
    for fam, arms in ACTOR.items():
        for arm in arms:
            key = RENAME.get((fam, arm), arm)
            per_q = {}
            for q in QUALITIES:
                ents = []
                for s in SEEDS:
                    c = load(q, s, fam)
                    ents.append(None if c is None or arm not in c else c[arm])
                    if c is None and not (fam == "cmd" and q not in ("medium-expert", "expert")):
                        data["missing"].append(f"{q}/seed{s}/{fam}")
                per_q[q] = pack(ents, actor_steps)
            data["actor"]["arms"][key] = per_q
    for run in COTRAIN:
        per_q = {}
        for q in QUALITIES:
            ents = []
            for s in SEEDS:
                c = load(q, s, run)
                ents.append(None if c is None else c[run])
                if c is None:
                    data["missing"].append(f"{q}/seed{s}/{run}")
            per_q[q] = pack(ents, co_steps)
        data["cotrain"]["arms"][run] = per_q
    data["missing"] = sorted(set(data["missing"]))
    text = json.dumps(data, separators=(",", ":"))
    Path(out).write_text(text, encoding="utf-8")
    print(f"{len(text)} characters; {len(data['missing'])} missing runs")


if __name__ == "__main__":
    main(sys.argv[1])
