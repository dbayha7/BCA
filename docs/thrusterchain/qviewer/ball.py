"""Extra data for the "critic in action" viewer: the removed ball and what each critic's training data hold.

Reads runs/shift_iw_v5 through the package's own loaders (run.load_data verifies every data file against world.json
and the data record; run.require_host reads each host's record) and q_data.json (the viewer data). Writes
/root/bca-shift/runs/qviewer/qv_extra.json and copies it to the Windows qviewer folder. Writes nothing else.

  ball        the removed ball of data seed 0 (the same for every quality: the expert's densest-15% ball), as
              world.json records it: every command a with ||a - centre|| <= rho is removed from the gap host's
              training rows.
  training    per quality and host: kept training rows, and the share of them whose thruster-1 command is past 80%
              upstream (a1 < -0.8), exactly at the 80% hold (a1 == float32(-0.8)), and inside the ball.
  slices      per episode of q_data.json (same order) and state: the interval of thruster-1 throttle (upstream share,
              0 to 1) for which the swept command (a1, actor's a2..a4) lies inside the ball, or null. Computed in
              closed form from the actor's command as q_data.json stores it (4 significant digits) and checked
              against Gap.contains on the 41 swept commands.
"""

import json
import math
import shutil
from pathlib import Path

import numpy as np

import experiments.shift  # noqa: F401
from experiments.shift_iw import run as RUN
from experiments.shift_iw import world as W

ROOT = Path("/root/bca-shift/runs/shift_iw_v5")
OUT = Path("/root/bca-shift/runs/qviewer")
WIN = Path("/mnt/c/Users/David/dev/bca-shift-work/qviewer")
SEED = 0


def sig4(v):
    return float(f"{float(v):.4g}") + 0.0


def main():
    q = json.loads((OUT / "q_data.json").read_text())
    F = {f: i for i, f in enumerate(q["episode_fields"])}
    ctx0 = RUN.Context(root=ROOT, quality="expert", log=lambda m: None)
    spec = RUN.spec_digest(ctx0)
    ball = None
    training = []
    checks = {"ball_same_every_quality": True, "host_removal_matches": True, "slice_mismatch_points": 0,
              "slice_boundary_points": 0}
    for quality in W.QUALITIES:
        ctx = RUN.Context(root=ROOT, quality=quality, log=lambda m: None)
        data = RUN.load_data(ctx, SEED, spec)
        g = data.gap
        b = dict(centre=[float(v) for v in g.centre], rho=float(g.rho), mass=float(g.mass))
        if ball is None:
            ball = b
        elif b != ball:
            checks["ball_same_every_quality"] = False
        log = data.train_log
        row = dict(quality=quality)
        for kind, removal in RUN.HOST_KINDS.items():
            rec = RUN.require_host(ctx, kind, SEED, spec)
            if rec.get("removal") != removal:
                checks["host_removal_matches"] = False
            rm = data.removal(removal)
            kept = rm.kept_train_rows(log)
            a = np.asarray(log["actions"][kept], np.float32).astype(np.float64)
            inball = g.contains(a)
            row[kind] = dict(rows=int(len(kept)),
                             past80=float(np.mean(a[:, 0] < float(np.float32(-0.8)) - 1e-9)),
                             at80=float(np.mean(a[:, 0] == float(np.float32(-0.8)))),
                             in_ball=float(np.mean(inball)))
        row["world_removal"] = {k: data.world["removal"][k] for k in
                                ("hold_share_training", "holds_removed_training", "removed_that_are_holds",
                                 "kept_share", "random_share_training")}
        training.append(row)

    c = np.asarray(ball["centre"]); rho = ball["rho"]
    sweep = np.asarray(q["sweep_a1"], np.float64)
    from experiments.shift import shift as SH
    gap = SH.Gap(centre=tuple(c), rho=rho, mass=ball["mass"], rule="", q=0.15, competing_ratio=0.0, record_sha256="")
    slices = []
    actor_in_ball = []
    for e in q["episodes"]:
        cmd = np.asarray(e[F["cmd"]], np.float64)
        sl, inb = [], []
        for t in range(len(cmd)):
            d2 = float(np.sum((cmd[t, 1:] - c[1:]) ** 2))
            if d2 < rho * rho:
                half = math.sqrt(rho * rho - d2)
                lo, hi = -(c[0] + half), -(c[0] - half)           # upstream share (a1 = -u)
                sl.append([sig4(lo), sig4(hi)])
            else:
                lo = hi = None
                sl.append(None)
            pts = np.column_stack([sweep.astype(np.float32).astype(np.float64), np.repeat(cmd[t:t + 1, 1:], len(sweep), 0)])
            mem = gap.contains(pts)
            for k, u in enumerate(-sweep):
                pred = lo is not None and lo <= u <= hi
                if pred != bool(mem[k]):
                    near = lo is not None and min(abs(u - lo), abs(u - hi)) < 1e-6
                    checks["slice_boundary_points" if near else "slice_mismatch_points"] += 1
            inb.append(bool(gap.contains(cmd[t:t + 1])[0]))
        slices.append(sl)
        actor_in_ball.append([int(v) for v in inb])
    out = dict(schema="bca-shift-qviewer-extra-v1",
               note=["ball: the removed ball of data seed 0 (world.json gap; the same for every quality): commands with "
                     "||a - centre|| <= rho were removed from the gap host's training rows.",
                     "training: per quality and host ('gap' = densest-15% removal, 'full' = none), the kept training "
                     "rows and the shares of them with thruster 1 past 80% upstream (a1 < -0.8), exactly at the 80% "
                     "hold (a1 = -0.8), and inside the ball.",
                     "slices: per episode (q_data.json order) and state, the thruster-1 throttle interval [low, high] "
                     "(upstream share, 0-1) in which the swept command lies inside the ball, or null.",
                     "actor_in_ball: per episode and state, 1 if the actor's own command lies inside the ball."],
               ball=dict(centre=[sig4(v) for v in ball["centre"]], rho=sig4(ball["rho"]), mass=ball["mass"]),
               training=[{k: (({kk: (sig4(vv) if isinstance(vv, float) else vv) for kk, vv in v.items()})
                              if isinstance(v, dict) else v) for k, v in r.items()} for r in training],
               slices=slices, actor_in_ball=actor_in_ball, checks=checks)
    assert checks["ball_same_every_quality"] and checks["host_removal_matches"], checks
    assert checks["slice_mismatch_points"] == 0, checks
    p = OUT / "qv_extra.json"
    p.write_text(json.dumps(out, separators=(",", ":")))
    shutil.copy(p, WIN / "qv_extra.json")
    print(json.dumps(checks))
    for r in training:
        print(r)


if __name__ == "__main__":
    main()
