"""READ-ONLY. Offline selection among SAVED policy-test actors by BCA's lower edge (the 'certified lower bound' use).

For every seed and world, every actor the policy test scored (values.json evaluations), at the 5,000 evaluation-split
states (pg.eval_obs, recorded conditions) and at the full-throttle-region training states:
  Q  = min_h Q_h(s, pi(s)) of the frozen gapped critic (what the actor climbs, both heads)
  U~ = the saved weighted band's width at (s, pi(s));  cert = share with log w(s, pi(s)) < l* (BCA can certify)
Selection rules (no return is read to select):
  critic_only     argmax mean Q                      (no BCA)
  lower_edge      argmax mean (Q - U~)               (BCA's lower edge, k = 1)
  certified       argmax mean Q among actors with cert >= 1 - alpha = 0.9 at the region states (fallback: highest
                  cert)
Then reads the selected actor's exact returns from values.json (strong current, recorded) and the Spearman correlation of
each criterion with the returns. Caveat: the saved classifier is the one fitted with the no-caution actor, not each
candidate's own (a per-actor refit would be a fit). Usage: python select.py
"""
import json
import time

import numpy as np
from scipy.stats import spearmanr

import common as C

L = C.L


def main():
    t0 = time.perf_counter()
    res = []
    for rep in range(5):
        for world in ("stall", "no_stall"):
            sd, pg = C.seed_data(world, rep)
            band = C.Band(world, rep)
            crit = C.critic(sd)
            acts, lock, vals = C.policy_actors(world, rep)
            ev = vals["evaluations"]
            names = sorted(ev)
            obs_e = np.asarray(pg.eval_obs, np.float32)
            obs_r = np.asarray(pg.train.obs, np.float32)[np.asarray(pg.region, bool)]
            rows = {}
            for n in names:
                r = {}
                for tag, o in (("eval", obs_e), ("region", obs_r)):
                    a = L.actor_rows(acts[n], o)
                    q = C.qmin(crit, o, a)
                    u = band.width(o, a)
                    lw = band.logw(o, a)
                    r[tag] = dict(Q=float(q.mean()), U=float(u.mean()), LE=float((q - u).mean()),
                                  cert=float((lw < band.l_star).mean()),
                                  past80=float(np.any(np.abs(a.astype(np.float64)) > C.KNEE, 1).mean()))
                r["points"] = {s: ev[n][s]["points"] for s in ev[n]}
                rows[n] = r
            sel = {}
            for tag in ("eval", "region"):
                Q = {n: rows[n][tag]["Q"] for n in names}
                LE = {n: rows[n][tag]["LE"] for n in names}
                cert = {n: rows[n]["region"]["cert"] for n in names}
                ok = [n for n in names if cert[n] >= 1 - C.ALPHA]
                pick_c = max(ok, key=lambda n: Q[n]) if ok else max(names, key=lambda n: cert[n])
                for rule, pick in (("critic_only", max(names, key=lambda n: Q[n])),
                                   ("lower_edge", max(names, key=lambda n: LE[n])), ("certified", pick_c)):
                    sel[f"{rule}_{tag}"] = dict(actor=pick, sc=rows[pick]["points"]["strong_current"],
                                                rec=rows[pick]["points"]["recorded"])
                for crit_name, D in (("Q", Q), ("LE", LE)):
                    for s in ("strong_current", "recorded"):
                        sel[f"spearman_{crit_name}_{tag}_vs_{s}"] = float(spearmanr(
                            [D[n] for n in names], [rows[n]["points"][s] for n in names]).correlation)
            sel["spearman_cert_region_vs_strong_current"] = float(spearmanr(
                [rows[n]["region"]["cert"] for n in names], [rows[n]["points"]["strong_current"] for n in names]).correlation)
            res.append(dict(rep=rep, world=world, rows=rows, selection=sel))
            print(f"seed {rep} {world:8s} " + " ".join(f"{k}={v['actor']}({v['sc']:.1f})" for k, v in sel.items()
                                                       if isinstance(v, dict)) +
                  f" | rho LE_eval~SC {sel['spearman_LE_eval_vs_strong_current']:.2f} Q_eval~SC "
                  f"{sel['spearman_Q_eval_vs_strong_current']:.2f} cert~SC "
                  f"{sel['spearman_cert_region_vs_strong_current']:.2f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    C.dump(res, "select.json")


if __name__ == "__main__":
    main()
