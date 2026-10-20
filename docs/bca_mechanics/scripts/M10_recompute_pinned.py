"""M10 (harm switch certificate): recompute every near-80 certificate from its recorded inputs, in the pinned environment.

Reads only: the near-80 run folder (runs/shift_explore/near80: data records, critic_second.msgpack + .json, switch.json),
the gate's pinned records (runs/shift) and the pinned code (outputs/shift/beyond-the-data/near80/code, restructure/step1).
Writes only under /tmp/bridge/work/m10/. Nothing is trained: the second critic is loaded from its saved file and applied
to the recorded rows, exactly as learner_phases.switch does, then step1lib.certificate is run with the recorded streams.

Usage: python recompute_m10.py <setting> [seed ...]
"""
import sys, os, json, time, traceback
sys.dont_write_bytecode = True
ROOT = "/home/user/bca"
CODE = ROOT + "/outputs/shift/beyond-the-data/near80/code"
for p in (CODE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import near80_constants as K
import phases as PH
import settingdata as SD
import learner_phases as LPh
from experiments.shift import learners as L

OUT = "/tmp/bridge/work/m10"
os.makedirs(OUT + "/overestimates", exist_ok=True)
setting = sys.argv[1]
seeds = [int(s) for s in sys.argv[2:]] or ([10, 11, 12, 13, 14] if setting == "main" else [0, 1, 2, 3, 4])
S1 = LPh.step1lib()


def num(v):
    return float(v) if isinstance(v, str) else v


def group_of(seed):
    try:
        g = K.group_of(seed)
        return g if isinstance(g, str) else ("exploratory" if seed in K.EXPLORATORY else "confirmatory")
    except Exception:
        return "exploratory" if seed in K.EXPLORATORY else "confirmatory"


results = []
for seed in seeds:
    for world in ("stall", "no_stall"):
        t0 = time.time()
        tag = f"{setting}/{world}/seed{seed}"
        try:
            key = PH.PhaseKey("switch", setting, world, seed, group_of(seed))
            run = PH.Run(key, K.OUT, mode="official")
            ctx = SD.context(run, setting, seed, worlds=(world,))
            pg = ctx.prepared(world)
            gr = LPh.gap_rule_of(pg)
            d = SD.learner_dir(K.OUT, setting, world, seed)
            rec2 = SD.load_record(d / "critic_second.json")
            recorded = SD.load_record(d / "switch.json")
            surv = np.asarray(pg.rows[gr["tail"]], np.int64)
            surv_ok = S1.array_sha256(surv) == rec2.get("survivors_sha256")
            critic2 = SD.load_tree_checked(d / "critic_second.msgpack", L.critic_template(), rec2["file_sha256"])
            gamma = float(ctx.learn.gamma)
            tail, rest = LPh.switch_scores(critic2, pg, gr, gamma)
            entry = LPh.certificate_entry(tail, rest, seed, world, gr)
            cert, rcert = entry["certificate"], recorded["certificate"]
            diffs = {}
            for k in sorted(set(cert) | set(rcert)):
                a, b = cert.get(k), rcert.get(k)
                if isinstance(a, (int, float, str)) and isinstance(b, (int, float, str)) and not isinstance(a, bool):
                    try:
                        fa, fb = num(a), num(b)
                        diffs[k] = dict(recomputed=fa, recorded=fb, equal=bool(fa == fb or (abs(fa - fb) <= 1e-9 * max(1.0, abs(fb)))))
                    except Exception:
                        diffs[k] = dict(recomputed=a, recorded=b, equal=bool(a == b))
                else:
                    diffs[k] = dict(recomputed=a, recorded=b, equal=bool(a == b))
            # the throttle index of the kept training rows and of the bank, for the gap-rule description
            t_train = S1.throttle(np.asarray(pg.actions))
            t_bank = S1.throttle(np.asarray(pg.bank.action))
            hi = np.sort(t_train[t_train > np.quantile(t_train, 0.99)])
            res = dict(setting=setting, world=world, seed=seed, seconds=round(time.time() - t0, 1),
                       gap_rule=dict(defined=bool(gr["defined"]), t_lo=gr["t_lo"], t_up=gr["t_up"], gap=gr["gap"],
                                     n_tail_training=int(gr["n_tail"]), n_above_quantile=int(gr["n_above_quantile"]),
                                     q99_training=float(np.quantile(t_train, 0.99)), n_training=int(len(t_train)),
                                     n_bank=int(len(t_bank)), n_bank_past_t_lo=int((t_bank > gr["t_lo"]).sum()),
                                     max_training_throttle=float(t_train.max()), n_training_at_t_up_or_more=int((t_train >= gr["t_up"]).sum()),
                                     recorded_t_lo=recorded["t_lo"], recorded_t_up=recorded["t_up"], recorded_n_survivors=recorded["n_survivors"]),
                       survivors_sha256_equal=bool(surv_ok), critic_second_sha256=rec2["file_sha256"],
                       critic_second_sha256_equal_recorded=bool(rec2["file_sha256"] == recorded.get("critic_second_sha256")),
                       gamma=gamma, n_tail=int(len(tail)), n_rest=int(len(rest)),
                       recomputed=cert, recorded=rcert, diffs=diffs,
                       all_equal=bool(all(v["equal"] for v in diffs.values())),
                       rank_test=dict(recomputed=entry["rank_test"], recorded=recorded["rank_test"]),
                       groups=dict(recomputed=entry["groups"], recorded=recorded["groups"]))
            np.savez(f"{OUT}/overestimates/{setting}_{world}_seed{seed}.npz", tail=np.asarray(tail, np.float64),
                     rest=np.asarray(rest, np.float64), t_lo=gr["t_lo"], t_up=gr["t_up"], n_survivors=int(len(surv)),
                     seed=seed, world_tag=K.WORLD_TAG[world], bank_throttle=t_bank, survivor_throttle=t_train[gr["tail"]])
            print(f"{tag}: n_tail {len(tail)} (recorded {rcert['n_tail']}), q05 {cert['q05']} (recorded {rcert['q05']}), "
                  f"height {cert['height']} (recorded {rcert['height']}), all_equal {res['all_equal']}, {res['seconds']}s", flush=True)
        except Exception as e:
            res = dict(setting=setting, world=world, seed=seed, error=repr(e), traceback=traceback.format_exc())
            print(f"{tag}: ERROR {e!r}", flush=True)
            print(traceback.format_exc(), flush=True)
        results.append(res)
        with open(f"{OUT}/recomp_{setting}.json", "w") as f:
            json.dump(results, f, indent=1, default=str)
print("done", setting, flush=True)
