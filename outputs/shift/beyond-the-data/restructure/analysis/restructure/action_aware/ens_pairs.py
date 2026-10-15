"""READ-ONLY. Critic-disagreement at the locked detection pairs, seeds 0-4, both worlds.

Forward passes of SAVED fixed-controller critics (stall-aware controller's value) at the saved pair sets. No training,
no fitting. Every critic file is SHA-256-checked against its record.json. Outputs go under this folder only.

Critics (G = runs/shift/68d9e8779e148b7b/critics/base/<world>/<name>/seed<k>/critic.msgpack):
  stall:    stall_aware_gap (the label critic), stall_aware_full, stall_aware_full_placebo (other init), stall_aware_random
  no_stall: stall_aware_gap, stall_aware_full
All use the gapped log's normalisation (acceptance.SeedData.norm), read from the Stage 2 band.json data block.

Ensembles (sample std, ddof=1, over heads):
  E2  = the gapped critic's own two heads (deployable today: same data, two inits, same target)
  EI  = full + full_placebo, 4 heads (same FULL data, two critic inits) -- stall only; init spread WHERE DATA EXIST
  EL  = gap + random + full + full_placebo (stall) / gap + full (no_stall): LEAKY reference -- members saw the
        full-throttle commands the gapped log lacks; not deployable, shows what data would add
"""
import hashlib
import json
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization

sys.path.insert(0, "/home/user/bca")
import algorithms.td3_bc as BASE  # noqa: E402

G = "/home/user/bca/runs/shift/68d9e8779e148b7b/critics/base"
ST2 = "/home/user/bca/runs/shift_explore/stage2/detect"
DET = "/home/user/bca/runs/shift_explore/detect"
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/action_aware"
DX, DA = 13, 4
CRITIC = BASE.DualCritic(DX, DA)
TEMPLATE = CRITIC.init(jax.random.PRNGKey(0), jnp.zeros(DX), jnp.zeros(DA))
APPLY = jax.jit(CRITIC.apply)
NAMES = {"stall": ("stall_aware_gap", "stall_aware_full", "stall_aware_full_placebo", "stall_aware_random"),
         "no_stall": ("stall_aware_gap", "stall_aware_full")}


def load_critic(world, name, seed):
    d = f"{G}/{world}/{name}/seed{seed}"
    rec = json.load(open(f"{d}/record.json"))
    blob = open(f"{d}/critic.msgpack", "rb").read()
    if hashlib.sha256(blob).hexdigest() != rec["result"]["file_sha256"]:
        raise SystemExit(f"{d}: SHA-256 mismatch")
    if rec["result"]["failed"]:
        raise SystemExit(f"{d}: failed critic")
    return serialization.from_bytes(TEMPLATE, blob)


def q_all(params, obs, act, chunk=8192):
    out = []
    for i in range(0, len(obs), chunk):
        out.append(np.asarray(APPLY(params, jnp.asarray(obs[i:i + chunk]), jnp.asarray(act[i:i + chunk])), np.float64))
    return np.concatenate(out)


def cap80(a):
    return np.clip(np.asarray(a, np.float32).astype(np.float64), -0.8, 0.8).astype(np.float32)


def run(seed, world):
    data = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.json"))["data"]
    mean = np.asarray(data["obs_mean"], np.float32)
    std = np.asarray(data["obs_std"], np.float32)
    pz = np.load(f"{DET}/seed{seed}/{world}/pairs.npz")
    lab = np.load(f"{DET}/seed{seed}/{world}/labels.npz")
    obs = ((np.asarray(pz["states"], np.float32) - mean) / std).astype(np.float32)
    cmd = np.asarray(pz["commands"], np.float32)
    cap = cap80(cmd)
    Q, Qc = {}, {}
    for nm in NAMES[world]:
        p = load_critic(world, nm, seed)
        Q[nm] = q_all(p, obs, cmd)        # (n, 2) at the pair's command
        Qc[nm] = q_all(p, obs, cap)       # (n, 2) at its 80% cap
    chk = dict(q1_cmd_max_abs=float(np.max(np.abs(Q["stall_aware_gap"][:, 0] - lab["q1_command"]))),
               q2_cmd_max_abs=float(np.max(np.abs(Q["stall_aware_gap"][:, 1] - lab["q2_command"]))),
               q1_cap_max_abs=float(np.max(np.abs(Qc["stall_aware_gap"][:, 0] - lab["q1_cap"]))))
    np.savez_compressed(f"{OUT}/q_seed{seed}_{world}.npz", **{f"cmd__{k}": v for k, v in Q.items()},
                        **{f"cap__{k}": v for k, v in Qc.items()})
    return chk


if __name__ == "__main__":
    res = {}
    for world in ("stall", "no_stall"):
        for seed in range(5):
            res[f"{world}/seed{seed}"] = run(seed, world)
            print(world, seed, res[f"{world}/seed{seed}"], flush=True)
    json.dump(res, open(f"{OUT}/ens_pairs_checks.json", "w"), indent=1)
