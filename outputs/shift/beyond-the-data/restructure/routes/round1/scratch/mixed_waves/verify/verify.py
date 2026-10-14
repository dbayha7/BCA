"""Adversarial verification of route 5 (mixed_waves). EXPLORATORY, seeds 0-4 only, read-only (forward passes and
rollouts of saved networks). Writes only verify.json beside this file.

1. Independent rebuild of the spliced t_lo, the primary evidence groups and the local primary arm (own-world critics,
   same rng tags as the route) -> must match results.json per seed-world.
2. Single-critic scoring: every row of both regions scored by ONE world's gate critic (+ its normalisation and
   controller), as one critic of a spliced log would; local primary arm rerun.
3. Outcome-free hand-set rivals the kill rule did not include: blanket caps c in a ladder (clip every thruster to
   [-c, c] every step; 0.80 is the true knee), and clip-always, oracle, never reproduced.
"""
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path("/home/user/bca")
STEP1 = ROOT / "outputs/shift/beyond-the-data/restructure/step1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STEP1))
import step1lib as S  # noqa: E402
import evaluate as E  # noqa: E402
import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402
import experiments.shift.truth as TR  # noqa: E402
from calibration import wbcp  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = json.loads((ROOT / "runs/shift_explore/routes/mixed_waves/results.json").read_text())
SEEDS = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
CUT = 0.25
CAPS = [0.75, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.95]
N = 200


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stalls(amp, m):
    amp = np.asarray(amp, np.float64)
    return amp > CUT if m == 1 else amp <= CUT


def silverman(x):
    x = np.asarray(x, np.float64)
    iqr = np.subtract(*np.quantile(x, [0.75, 0.25]))
    return 0.9 * min(np.std(x, ddof=1), iqr / 1.34) * len(x) ** -0.2


def roll(policy, physics, x0, lo, hi, t_lo):
    x = np.asarray(x0, np.float64).copy()
    J = np.zeros(len(x))
    for _ in range(physics.length):
        a = np.clip(policy(x), -1.0, 1.0)
        fire = np.abs(a).max(1) > t_lo
        a = np.where(fire[:, None], np.clip(a, lo, hi), a)
        x, r = TC.dynamics(x, a, physics)
        J = J + r
    return J


def local_decisions(tail_s, tail_a, rest_s, rest_a, a0s, h, seed, m, cfg):
    dec = np.zeros(len(a0s), bool)
    for i, a0 in enumerate(a0s):
        wr = np.exp(-0.5 * ((rest_a - a0) / h) ** 2)
        wt = np.exp(-0.5 * ((tail_a - a0) / h) ** 2)
        cr = wbcp.calibrate(rest_s, S.rng([20261015, 9960, seed, m, 3, 2]), weights=wr, test_mass=1.0,
                            alpha=cfg["alpha"], beta=cfg["credibility"], draws=cfg["draws"])
        ct = wbcp.calibrate(tail_s, S.rng([20261015, 9960, seed, m, 1, 1]), weights=wt, test_mass=1.0,
                            alpha=cfg["alpha"], beta=cfg["credibility"], draws=cfg["draws"])
        dec[i] = S.pairwise_quantile(ct.crossings, cr.crossings, cfg["statistic_quantile"]) > cfg["margin"]
    return dec


def main():
    c0 = time.process_time()
    spec, _ = S.load_prespec()
    cfg = spec["certificate"]
    gcfg = spec["groups"]["gap_rule"]
    p, variant = E.gate_context(spec)
    learn = p.config.learn
    gamma = float(spec["scores"]["gamma"])
    anchors = {w: A.anchors_of(p, variant, w)["strong_current"] for w in S.WORLDS}
    x0 = np.asarray(TC.bank_starts("strong_current", TC.BANK_SIZE)[:N], np.float64)
    a0s = TC.wave_amplitude(x0)
    out = {}
    for seed in SEEDS:
        W = {}
        for w in S.WORLDS:
            sd = A.load_seed(p.runs, p.config, variant, w, seed)
            lg, pg = sd.log, sd.prepared("gap", learn)
            amp = TC.wave_amplitude(np.asarray(lg["infos/state"], np.float64))
            cal = lg.split_rows("calibration")
            cal = cal[~sd.gap.contains(lg["actions"][cal])]
            bank = np.asarray(pg.bank_rows, np.int64)
            _, critic = A.get_critic(p, sd, "gap", "stall_aware", train=False)
            actor, _ = E.none_actor(w, seed)
            W[w] = dict(lg=lg, pg=pg, amp=amp, cal=cal, bank=bank, critic=critic,
                        fixed=L.FixedController("stall_aware", pg.physics, pg.obs_mean, pg.obs_std),
                        policy=L.mlp_policy(actor, pg, "none"), physics=pg.physics,
                        kept_amp=amp[pg.rows], kept_act=np.asarray(pg.actions, np.float64))
        # rollouts that do not depend on m
        never = {(a, w): roll(W[a]["policy"], W[w]["physics"], x0, 0, 0, 9.0) for a in S.WORLDS for w in S.WORLDS}
        caps = {(c, a, w): roll(W[a]["policy"], W[w]["physics"], x0, -np.full(4, c), np.full(4, c), -1.0)
                for c in CAPS for a in S.WORLDS for w in S.WORLDS}
        out[seed] = {}
        for m in (1, 2):
            R = RES["per_seed"][f"seed{seed}"][f"W{m}"]
            reg = stalls(a0s, m)

            def pts(Jd, a):
                return np.where(reg, TR.points(Jd[(a, "stall")], anchors["stall"]),
                                TR.points(Jd[(a, "no_stall")], anchors["no_stall"]))

            # spliced t_lo and range, recomputed
            kept = np.concatenate([W[w]["kept_act"][stalls(W[w]["kept_amp"], m) == (w == "stall")] for w in S.WORLDS])
            gr = S.gap_rule(S.throttle(kept.astype(np.float32)), gcfg["upper_quantile"], gcfg["min_gap"],
                            gcfg["min_tail"])
            t_lo = gr["t_lo"]
            lo, hi = np.quantile(kept, 0.005, axis=0), np.quantile(kept, 0.995, axis=0)
            e = dict(t_lo=t_lo, t_lo_matches=abs(t_lo - R["t_lo"]) < 1e-12,
                     range_matches=bool(np.allclose(lo, R["recorded_range"]["lo"], atol=1e-12)
                                        and np.allclose(hi, R["recorded_range"]["hi"], atol=1e-12)))
            # evidence rows, per world (region-matched)
            rows = {}
            for w in S.WORLDS:
                d = W[w]
                lg = d["lg"]
                tc = S.throttle(np.asarray(lg["actions"][d["cal"]]))
                tb = S.throttle(np.asarray(lg["actions"][d["bank"]]))
                inreg_c = stalls(d["amp"][d["cal"]], m) == (w == "stall")
                inreg_b = stalls(d["amp"][d["bank"]], m) == (w == "stall")
                rows[w] = dict(tail=d["cal"][inreg_c & (tc > t_lo)], rest=d["bank"][inreg_b & (tb <= t_lo)])
            e["n_tail"] = {w: int(len(rows[w]["tail"])) for w in S.WORLDS}
            e["n_tail_matches"] = (e["n_tail"]["stall"] == R["n_full_throttle"]["primary"]["stall_region"]
                                   and e["n_tail"]["no_stall"] == R["n_full_throttle"]["primary"]["help_region"])
            on_J = {(a, w): roll(W[a]["policy"], W[w]["physics"], x0, lo, hi, t_lo) for a in S.WORLDS for w in S.WORLDS}
            gain = {a: pts(on_J, a) - pts(never, a) for a in S.WORLDS}
            e["clip_always"] = float(gain["stall"].mean())
            e["clip_always_matches"] = abs(e["clip_always"] - RES["summary"]["clip_always"]["per_seed_world"][f"{seed}_{m}"]) < 1e-9
            e["oracle"] = float(np.where(reg, gain["stall"], 0).mean())
            for c in CAPS:
                cg = {a: pts({k[1:]: v for k, v in caps.items() if k[0] == c}, a) - pts(never, a) for a in S.WORLDS}
                e[f"cap_{c:g}"] = float(cg["stall"].mean())
                e[f"cap_{c:g}_nostall_actor"] = float(cg["no_stall"].mean())
            e["clip_always_nostall_actor"] = float(gain["no_stall"].mean())
            # scoring: own-world critic (as the route) and each single critic
            for scorer in ("own", "stall", "no_stall"):
                ts, ta, rs, ra = [], [], [], []
                for w in S.WORLDS:
                    cw = w if scorer == "own" else scorer
                    C = W[cw]
                    lg = W[w]["lg"]
                    for kind, S_, A_ in (("tail", ts, ta), ("rest", rs, ra)):
                        r = rows[w][kind]
                        tr = L._transition(lg, r, C["pg"].obs_mean, C["pg"].obs_std)
                        S_.append(np.asarray(E.overestimates(C["critic"], tr, C["fixed"], gamma), np.float64))
                        A_.append(W[w]["amp"][r])
                ts, ta, rs, ra = (np.concatenate(v) for v in (ts, ta, rs, ra))
                h = silverman(ta)
                if scorer == "own":
                    e["h_matches"] = abs(h - R["h"]["primary"]) < 1e-12
                    e["mean_tail_stall_region"] = float(ts[:e["n_tail"]["stall"]].mean())
                dec = local_decisions(ts, ta, rs, ra, a0s, h, seed, m, cfg)
                wl = S.certificate(ts, rs, [20261015, 9961, seed, m, 1], cfg)
                e[f"local_{scorer}"] = float(np.where(dec, gain["stall"], 0).mean())
                e[f"local_{scorer}_on_stall"] = float(dec[reg].mean())
                e[f"local_{scorer}_on_help"] = float(dec[~reg].mean())
                e[f"world_level_{scorer}"] = bool(wl["on"])
                e[f"tail_mean_{scorer}"] = dict(stall_region=float(ts[:e['n_tail']['stall']].mean()),
                                                help_region=float(ts[e['n_tail']['stall']:].mean()))
                e[f"rest_mean_{scorer}"] = float(rs.mean())
            e["local_own_matches"] = abs(e["local_own"] - RES["summary"]["local_primary"]["per_seed_world"][f"{seed}_{m}"]) < 1e-9
            out[seed][m] = e
            log(seed, m, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()})
    agg = {}
    keys = [k for k in out[SEEDS[0]][1] if isinstance(out[SEEDS[0]][1][k], float)]
    for k in keys:
        agg[k] = float(np.mean([out[s][m][k] for s in SEEDS for m in (1, 2)]))
    res = dict(seeds=SEEDS, per=out, mixed_world_means=agg, cpu_s=time.process_time() - c0)
    (HERE / f"verify_{'_'.join(map(str, SEEDS))}.json").write_text(json.dumps(S.jsonable(res), indent=1))
    log("MEANS", json.dumps({k: round(v, 2) for k, v in agg.items()}))
    log("cpu", time.process_time() - c0)


if __name__ == "__main__":
    main()
