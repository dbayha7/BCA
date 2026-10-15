"""Plain TD3+BC coverage with its own band q +/- |Q1 - Q2| (EXPLORATORY; see PLAIN_TWIN_PREREG.md).

Reads (never writes) runs/shift_iw_v5: per quality, seed and host kind, the coverage stage's steps.npz (y, q, in-ball
flags and BCA's outcomes) and the host critic; recomputes Q1 and Q2 at the same rows and checks min(Q1, Q2) against
the saved q. Writes runs/shift_iw_v5_explore/plain_twin/<quality>.json.

Usage: python plain_twin_coverage.py <quality> [<quality> ...]
"""
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

from experiments.shift_iw import coverage as CV
from experiments.shift_iw import data as D
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import run as RN
from experiments.shift_iw import world as W

OUT = RN.ROOT_DEFAULT.parent / "shift_iw_v5_explore" / "plain_twin"
LABEL = ("EXPLORATORY: outside the pre-registered version-5 claims; plain TD3+BC's band is q +/- |Q1 - Q2|, the owner's "
         "choice of 2026-10-13 ('Its two critics' disagreement (Recommended)')")


def twin(host, trans, chunk=8192):
    """Q1 and Q2 of the host critic at the recorded commands, float64."""
    critic = LI.models()[1]
    f = jax.jit(lambda p, o, a: critic.apply(p, o, a))
    q1, q2 = [], []
    n = len(trans.obs)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        q = np.asarray(f(host.critic.params, trans.obs[i0:i1], trans.action[i0:i1]), np.float64)
        q1.append(q[:, 0])
        q2.append(q[:, 1])
    return np.concatenate(q1), np.concatenate(q2)


def share(mask_values):
    return float(np.mean(mask_values)) if len(mask_values) else None


def one(ctx, spec, seed, kind, data):
    prep = D.prepare(data, RN.HOST_KINDS[kind], ctx.cfg)
    host = RN.load_host_state(ctx, kind, seed, spec)
    z = np.load(RN.coverage_dir(ctx, kind, seed) / "steps.npz")
    out = {}
    for tilt in W.TILT_NAMES:
        log, rows = data.test(tilt)
        trans = LI.transitions(log, rows, prep)
        q1, q2 = twin(host, trans)
        y, q, in_ball = z[f"{tilt}__y"], z[f"{tilt}__q"], z[f"{tilt}__in_ball"].astype(bool)
        gap = np.abs(np.minimum(q1, q2) - q)
        if gap.max() > 1e-4 * max(1.0, float(np.abs(q).max())):
            raise SystemExit(f"min(Q1, Q2) does not reproduce the saved q ({ctx.quality} seed {seed} {kind} {tilt}): "
                             f"largest difference {gap.max():.3g}")
        U = np.abs(q1 - q2)
        cov = np.abs(y - q) <= U
        o_u = z[f"{tilt}__unweighted__outcome"]
        u_width = z[f"{tilt}__unweighted__width"]
        row = dict(steps=int(len(y)), in_ball_steps=int(in_ball.sum()),
                   check_min_twin_vs_saved_q=float(gap.max()))
        for part, m in (("all", np.ones(len(y), bool)), ("outside", ~in_ball), ("inside", in_ball)):
            if not m.any():
                row[part] = None
                continue
            r = dict(plain=share(cov[m]), unweighted_bca=share(o_u[m] == CV.COVERED),
                     plain_median_width=float(np.median(U[m])), bca_unweighted_median_width=float(np.median(u_width[m])),
                     median_abs_error=float(np.median(np.abs(y - q)[m])))
            if f"{tilt}__exact__outcome" in z:
                o_w = z[f"{tilt}__exact__outcome"]
                r.update(weighted_bca=share(o_w[m] == CV.COVERED), weighted_bca_refused=share(o_w[m] == CV.REFUSED))
            row[part] = r
        out[tilt] = row
    return out


def across(per_seed, path):
    vals = []
    for v in per_seed:
        for p in path:
            v = None if v is None else v.get(p)
        vals.append(v)
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return dict(CV.across_seeds(np.asarray(vals, np.float64)), per_seed=vals)


def main(qualities):
    OUT.mkdir(parents=True, exist_ok=True)
    for quality in qualities:
        started = time.time()
        ctx = RN.Context(quality=quality)
        spec = RN.spec_digest(ctx)
        res = {k: [] for k in ("gap", "full")}
        for seed in range(5):
            data = RN.load_data(ctx, seed, spec)
            for kind in res:
                res[kind].append(one(ctx, spec, seed, kind, data))
            print(time.strftime("%H:%M:%S"), quality, "seed", seed, "done", flush=True)
        summary = {}
        for kind, per_seed in res.items():
            summary[kind] = {}
            for tilt in W.TILT_NAMES:
                summary[kind][tilt] = {}
                for part in ("all", "outside", "inside"):
                    keys = set()
                    for s in per_seed:
                        if s[tilt].get(part):
                            keys |= set(s[tilt][part])
                    summary[kind][tilt][part] = {k: across([s[tilt] for s in per_seed], (part, k)) for k in sorted(keys)}
        rec = dict(label=LABEL, quality=quality, seconds=round(time.time() - started, 1),
                   spec_sha256=RN.spec_digest(ctx) if isinstance(spec, str) else None,
                   summary=summary, per_seed=res)
        (OUT / f"{quality}.json").write_text(json.dumps(rec, indent=1, default=float))
        print("wrote", OUT / f"{quality}.json", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(RN.QUALITIES))
