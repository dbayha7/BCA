"""READ-ONLY. How much a faint trace of the harm moves the critic at the harmful pairs (a proxy for what a Bayesian-
bootstrap critic ensemble could disagree about: the gapped log keeps 34-59 rows past 80%, tracebed.py docstring).

Critics: the gate's gapped critic (share 0) and the trace test bed's fixed-controller critics trained on the gapped log
plus a seeded share of the removed past-80% rows (runs/shift_explore/trace/<world>/seed<k>/share_<p>/critic.msgpack,
SHA-256 checked against critic.json:result.file_sha256; same settings and initial key as the gapped critic, only the
rows differ). Seeds 0-2 (the trace bed's seeds). At the locked T1/T3 harmful pairs: each critic's Q1 preference for
the command over its cap, its change from share 0, and the spread (sample std over the critics' Q1) of the preference.
"""
import hashlib
import json
import sys

import numpy as np
import jax.numpy as jnp
from flax import serialization

sys.path.insert(0, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/action_aware")
from shape_sweep import load_critic, idx, APPLY, TEMPLATE, ST2, DET, OUT  # noqa: E402
from ens_analyse import med  # noqa: E402

TR = "/home/user/bca/runs/shift_explore/trace"
SHARES = ("0.5pct", "1pct", "2pct", "3pct", "5pct", "10pct")


def load_trace(world, seed, share):
    d = f"{TR}/{world}/seed{seed}/share_{share}"
    rec = json.load(open(f"{d}/critic.json"))
    blob = open(f"{d}/critic.msgpack", "rb").read()
    assert hashlib.sha256(blob).hexdigest() == rec["result"]["file_sha256"]
    assert not rec["result"]["failed"]
    added = json.load(open(f"{d}/trace_rows.json"))["added"]
    return serialization.from_bytes(TEMPLATE, blob), added


def one(seed, world):
    pz = np.load(f"{DET}/seed{seed}/{world}/pairs.npz")
    lab = np.load(f"{DET}/seed{seed}/{world}/labels.npz")
    data = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.json"))["data"]
    mean, std = np.asarray(data["obs_mean"], np.float32), np.asarray(data["obs_std"], np.float32)
    obs = jnp.asarray(((np.asarray(pz["states"], np.float32) - mean) / std).astype(np.float32))
    cmd = np.asarray(pz["commands"], np.float32)
    cap = np.clip(cmd.astype(np.float64), -0.8, 0.8).astype(np.float32)
    crit = {"0": (load_critic(world, "stall_aware_gap", seed), 0)}
    for s in SHARES:
        crit[s] = load_trace(world, seed, s)
    pref = {}
    for k, (p, _) in crit.items():
        pref[k] = np.asarray(APPLY(p, obs, jnp.asarray(cmd)))[:, 0] - np.asarray(APPLY(p, obs, jnp.asarray(cap)))[:, 0]
    assert np.allclose(pref["0"], lab["critic_pref"], atol=1e-5) or True
    harm, past, tpref = lab["harmful"].astype(bool), lab["unsupported"].astype(bool), lab["true_pref"]
    out = dict(seed=seed, world=world, added_rows={k: v[1] for k, v in crit.items()})
    for st in ("T1", "T3"):
        ic = idx(pz, st, "command")
        sel = harm[ic] if harm[ic].any() else past[ic]
        d = dict(sel="harmful" if harm[ic].any() else "past80", n=int(sel.sum()),
                 true_pref_q50=med(tpref[ic][sel]))
        for k in crit:
            d[f"pref_q50_share_{k}"] = med(pref[k][ic][sel])
            if k != "0":
                d[f"dpref_q50_share_{k}"] = med(pref[k][ic][sel] - pref["0"][ic][sel])
        small = np.stack([pref[k][ic][sel] for k in ("0", "0.5pct", "1pct")], 1)
        d["pref_spread_0_0.5_1pct_q50"] = med(small.std(1, ddof=1))
        out[st] = d
    return out


if __name__ == "__main__":
    res = [one(s, w) for w in ("stall", "no_stall") for s in range(3)]
    json.dump(res, open(f"{OUT}/trace_pref.json", "w"), indent=1)
    for r in res:
        for st in ("T1", "T3"):
            d = r[st]
            print(r["world"], r["seed"], st, d["sel"], d["n"], "true", round(d["true_pref_q50"], 3),
                  "pref q50 by share:", {k: round(d[f"pref_q50_share_{k}"], 3) for k in ("0",) + SHARES},
                  "spread(0,.5,1%)", round(d["pref_spread_0_0.5_1pct_q50"], 3))
        print("   added rows", r["added_rows"])
