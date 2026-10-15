"""Independent numbers review of the "critic in action" data (reviewer; CPU 1). Read only everywhere; prints to stdout
and writes /root/bca-shift/runs/qviewer/review/review_numbers.json only.

Own code paths (not compute_q.py's): a numpy float64 actor and a numpy float32 critic built from the loaded
parameters, a numpy rollout of TC.dynamics for the episode and for every exact value Q^pi (no truth.rollout, no
q_value_jax, no MLPPolicy, no learners.critic_rows), the observation normalisation read from record.json (not
data.normalization), the snapshot file hashed with hashlib, and the summaries recomputed from scratch.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

import experiments.shift  # noqa: F401
from experiments.shift import learners as LR
from experiments.shift import recorder as RC
from experiments.shift import shift as SH
from experiments.shift import thruster_chain as TC
from experiments.shift_iw import data as D
from experiments.shift_iw import run as RUN
from experiments.shift_iw import world as W

ROOT = Path("/root/bca-shift/runs/shift_iw_v5")
OUTD = Path("/root/bca-shift/runs/qviewer/review")
Q4 = json.loads(Path("/root/bca-shift/runs/qviewer/q_data.json").read_text())
Q7 = json.loads(Path("/root/bca-shift/runs/qviewer/precise/q_data.json").read_text())
QV = json.loads(Path("/mnt/c/Users/David/dev/bca-shift-work/qviewer/qv_data.json").read_text())
LAW = {"training": "none", "strong": "strong"}
COMBOS = [("expert", "gap", "strong"), ("medium", "full", "training"), ("random", "full", "strong"),
          ("medium-replay", "gap", "training")]
CTX_COMBOS = [("expert", "gap", "strong"), ("random", "full", "strong"), ("medium", "full", "training")]
STATES = [0, 23, 40, 71, 99]
THR = [0, 16, 32, 33, 36, 40]          # 0%, 40%, 80%, 82.5%, 90%, 100%
H, HL, G = 600, 1000, 0.98
SWEEP = (-(np.arange(41) / 40.0)).astype(np.float32).astype(np.float64) + 0.0
I80, I100 = 32, 40
F7 = {f: i for i, f in enumerate(Q7["episode_fields"])}
FV = {f: i for i, f in enumerate(QV["fields"])}
REPORT = {}
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


class Actor:
    def __init__(self, params, mean, std):
        p = params["params"]
        self.W = [np.asarray(p[f"Dense_{i}"]["kernel"], np.float64) for i in range(3)]
        self.b = [np.asarray(p[f"Dense_{i}"]["bias"], np.float64) for i in range(3)]
        self.mean, self.std = np.asarray(mean, np.float64), np.asarray(std, np.float64)

    def __call__(self, x):
        o = np.asarray(x, np.float64).astype(np.float32).astype(np.float64)
        h = (o - self.mean) / self.std
        h = np.maximum(h @ self.W[0] + self.b[0], 0.0)
        h = np.maximum(h @ self.W[1] + self.b[1], 0.0)
        return np.tanh(h @ self.W[2] + self.b[2])


def critic(cparams, states, actions, mean, std):
    p = cparams["params"]["VmapCritic_0"]
    obs = (np.asarray(states, np.float32) - mean) / std
    x = np.concatenate([obs, np.asarray(actions, np.float32)], -1).astype(np.float32)
    heads = []
    for h in range(2):
        z = np.maximum(x @ np.asarray(p["Dense_0"]["kernel"][h], np.float32) + np.asarray(p["Dense_0"]["bias"][h], np.float32), 0)
        z = np.maximum(z @ np.asarray(p["Dense_1"]["kernel"][h], np.float32) + np.asarray(p["Dense_1"]["bias"][h], np.float32), 0)
        heads.append((z @ np.asarray(p["Dense_2"]["kernel"][h], np.float32) + np.asarray(p["Dense_2"]["bias"][h], np.float32))[:, 0])
    heads = np.stack(heads, 1).astype(np.float64)
    return heads.min(1), heads


def episode(actor, x0, L=100):
    x = np.asarray(x0, np.float64)[None]
    S, C, R = [x[0]], [], []
    for _ in range(L):
        a = np.clip(actor(x), -1.0, 1.0)
        x2, r = TC.dynamics(x, a, W.PHYSICS)
        S.append(x2[0]), C.append(a[0]), R.append(r[0])
        x = x2
    return np.array(S), np.array(C), np.array(R)


def rollout_many(actor, X0, L=100):
    x = np.asarray(X0, np.float64)
    S, C, J = [], [], np.zeros(len(x))
    for _ in range(L):
        a = np.clip(actor(x), -1.0, 1.0)
        S.append(x), C.append(a)
        x, r = TC.dynamics(x, a, W.PHYSICS)
        J = J + r
    return np.stack(S, 1), np.stack(C, 1), J


def qpi(actor, X, A, horizon, chunk=2100):
    out = []
    for i in range(0, len(X), chunk):
        x, q = TC.dynamics(X[i:i + chunk], A[i:i + chunk], W.PHYSICS)
        disc = 1.0
        for _ in range(1, horizon):
            disc *= G
            x, r = TC.dynamics(x, actor(x), W.PHYSICS)
            q = q + disc * r
        out.append(q)
    return np.concatenate(out)


def relmax(a, b, floor=1e-12):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))


def absmax(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def find(lst, q, c, s):
    for i, e in enumerate(lst):
        if (e[0], e[1], e[2]) == (q, c, s):
            return i
    raise KeyError((q, c, s))


def summary(cmds, push, qs, ps, qa, pa):
    n = len(cmds)
    hat, best = qs.argmax(1), ps.argmax(1)
    cr, tc = qs[:, I100] - qs[:, I80], ps[:, I100] - ps[:, I80]
    rf = (cr > 0) & (tc < 0)
    cost = ps[np.arange(n), best] - ps[np.arange(n), hat]
    return dict(actor_thruster1_past80_steps=int(np.sum(cmds[:, 0] < -TC.KNEE)),
                a_hat_past80_states=int(np.sum(hat > I80)), a_hat_at_100_states=int(np.sum(hat == I100)),
                a_hat_at_80_states=int(np.sum(hat == I80)),
                best_true_past80_states=int(np.sum(best > I80)), best_true_at_80_states=int(np.sum(best == I80)),
                critic_higher_at_100_than_80_states=int(np.sum(cr > 0)),
                critic_keeps_rising_80_to_100_states=int(np.sum(np.all(np.diff(qs[:, I80:], axis=1) > 0, 1))),
                truth_lower_at_100_than_80_states=int(np.sum(tc < 0)),
                truth_keeps_falling_80_to_100_states=int(np.sum(np.all(np.diff(ps[:, I80:], axis=1) < 0, 1))),
                critic_rises_while_truth_falls_states=int(rf.sum()),
                critic_rises_while_truth_falls_steps=[int(v) for v in np.flatnonzero(rf)],
                critic_change_80_to_100_mean=float(cr.mean()), truth_change_80_to_100_mean=float(tc.mean()),
                critic_change_80_to_100_where_truth_falls_mean=float(cr[rf].mean()) if rf.any() else None,
                truth_change_80_to_100_where_critic_rises_mean=float(tc[rf].mean()) if rf.any() else None,
                over_preference_100_vs_80_mean=float((cr - tc).mean()),
                true_cost_of_a_hat_mean=float(cost.mean()), true_cost_of_a_hat_max=float(cost.max()),
                critic_minus_truth_at_actor_command_mean=float(np.mean(qa - pa)),
                push_mean_where_critic_rises_while_truth_falls=float(push[rf].mean()) if rf.any() else None,
                push_mean_elsewhere=float(push[~rf].mean()) if (~rf).any() else None), hat, best


def s4(v):
    return float(f"{v:.4g}") + 0.0


def cmp_summary(mine, theirs, label):
    bad = []
    for k, v in mine.items():
        w = theirs.get(k, "MISSING")
        if isinstance(v, float):
            if w is None or w == "MISSING" or abs(v - w) > 5e-4 * max(abs(w), 1e-9) + 1e-12:
                bad.append((k, v, w))
        elif v != w:
            bad.append((k, v, w))
    log(f"  summary vs {label}: {'all equal' if not bad else bad}")
    return bad


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    ctx0 = RUN.Context(root=ROOT, quality="expert", log=lambda m: None)
    spec = RUN.spec_digest(ctx0)
    banks, brec = D.load_start_banks(ctx0.banks_root)
    # ---------------- start choice and anchors
    st = {}
    for name, law in LAW.items():
        c0, c1 = float(banks[law][0, TC.CURRENT]), float(banks[law][1, TC.CURRENT])
        lo, hi = W.inverse_cdf(np.array([0.25, 0.75]), law)
        emp = float(np.mean(banks[law][:, TC.CURRENT] <= c1))
        st[name] = dict(c0=c0, q0=float(W.cdf(c0, law)), c1=c1, q1=float(W.cdf(c1, law)), iqr=[float(lo), float(hi)],
                        empirical_quantile_in_bank=emp, anchors=brec["anchors"][law]["anchor"], bank_size=len(banks[law]))
        sa = RC.stall_aware(W.PHYSICS)
        _, _, Jsa = rollout_many(sa, banks[law])
        rq = D.quality_on_bank("random", len(banks[law]))
        from experiments.shift import truth as TR  # only for the streamed random controller (it needs t)
        Jr = TR.rollout(rq, banks[law], W.PHYSICS).J
        st[name].update(stall_aware_from_start=float(Jsa[1]), random_from_start=float(Jr[1]),
                        stall_aware_mean=float(Jsa.mean()), random_mean=float(Jr.mean()))
        v = QV["sit"][name]
        log(f"start {name}: c0={c0:.4f} (q {st[name]['q0']:.4f}), c1={c1:.4f} (q {st[name]['q1']:.4f}, empirical "
            f"{emp:.3f}), IQR {lo:.4f}-{hi:.4f}; anchors {st[name]['anchors']}; recomputed means SA {Jsa.mean():.4f} "
            f"random {Jr.mean():.4f}; from start SA {Jsa[1]:.4f} random {Jr[1]:.4f}")
        log(f"   QV sit: {v}")
    REPORT["start"] = st

    # ---------------- per combo
    REPORT["combos"] = {}
    for (q, c, s) in COMBOS:
        law = LAW[s]
        log(f"=== {q} / {c} / {s}")
        ctx = RUN.Context(root=ROOT, quality=q, log=lambda m: None)
        hdir = ROOT / q / "host" / c / "seed0"
        rec = json.loads((hdir / "record.json").read_text())
        snap = rec["result"]["snapshots"][-1]
        fsha = hashlib.sha256((hdir / "snapshot_100.msgpack").read_bytes()).hexdigest()
        state = RUN.load_host_state(ctx, c, 0, spec)
        mean = np.asarray(rec["result"]["data"]["obs_mean"], np.float32)
        std = np.asarray(rec["result"]["data"]["obs_std"], np.float32)
        hashes = dict(file=fsha == snap["file_sha256"], actor=LR.tree_sha256(state.actor.params) == snap["actor_sha256"],
                      critic=LR.tree_sha256(state.critic.params) == snap["critic_sha256"], update=snap["update"],
                      removal=rec.get("removal"))
        log(f"  hashes {hashes}")
        actor = Actor(state.actor.params, mean, std)
        cp = state.critic.params
        # evaluated mean over the whole bank against the run's own evaluation record
        ev = json.loads((ROOT / q / "evaluate" / "seed0" / "hosts.json").read_text())["evaluations"]
        _, _, Jbank = rollout_many(actor, banks[law])
        Jrec = ev[f"plain_{c}"][law]["J"]
        log(f"  bank mean J mine {Jbank.mean():.10f} vs hosts.json {Jrec:.10f} (diff {Jbank.mean() - Jrec:.2e})")
        # the episode
        S_all, C, R = episode(actor, banks[law][1])
        S = S_all[:100]
        push = S[:, TC.CURRENT] + S[:, TC.WAVE1]
        J, Jd = float(R.sum()), float(np.sum(R * G ** np.arange(100)))
        an = brec["anchors"][law]["anchor"]
        pts = 100.0 * (J - an["random"]) / (an["top"] - an["random"])
        i7, iv = find(Q7["episodes"], q, c, s), find(QV["eps"], q, c, s)
        e7, ev_ = Q7["episodes"][i7], QV["eps"][iv]
        log(f"  J {J:.6f} Jd {Jd:.6f} points {pts:.6f} | Q7 {e7[F7['return']]} {e7[F7['discounted_return']]} "
            f"{e7[F7['points']]} | QV {ev_[FV['return']]} {ev_[FV['discounted_return']]} {ev_[FV['points']]}")
        diffs = dict(state=relmax(S, e7[F7["state"]], 1e-3), cmd=absmax(C, e7[F7["cmd"]]),
                     reward=relmax(R, e7[F7["reward"]]), push=absmax(push, e7[F7["push"]]))
        # reward really includes the speeds: compare with positions only
        pos_only = -(np.sum(S[:, 0:10:2] ** 2, 1) + 0.3 * np.sum(C ** 2, 1))
        full = -(np.sum(S[:, 0:10] ** 2, 1) + 0.3 * np.sum(C ** 2, 1))
        diffs["reward_vs_positions_only_formula_maxabs"] = absmax(R, pos_only)
        diffs["reward_vs_all10_formula_maxabs"] = absmax(R, full)
        diffs["speed_share_of_float_cost_mean"] = float(np.mean(np.sum(S[:, 1:10:2] ** 2, 1) / np.sum(S[:, 0:10] ** 2, 1)))
        # critic at the actor's command and the full sweep
        qa, heads_a = critic(cp, S, C, mean, std)
        A = np.repeat(C[:, None, :], 41, 1)
        A[:, :, 0] = SWEEP
        Xs = np.repeat(S[:, None, :], 41, 1).reshape(-1, 13)
        As = A.reshape(-1, 4)
        qs = critic(cp, Xs, As, mean, std)[0].reshape(100, 41)
        diffs["q_actor_rel"] = relmax(qa, e7[F7["q_actor"]])
        diffs["q_sweep_rel"] = relmax(qs, e7[F7["q_sweep"]])
        diffs["q_actor_abs"] = absmax(qa, e7[F7["q_actor"]])
        diffs["q_sweep_abs"] = absmax(qs, e7[F7["q_sweep"]])
        # exact values: actor's command (all 100) and the whole sweep (horizon 600)
        tq = time.time()
        pa = qpi(actor, S, C, H)
        ps = qpi(actor, Xs, As, H).reshape(100, 41)
        log(f"  exact values for 4,200 commands in {time.time() - tq:.1f}s")
        diffs["qpi_actor_rel"] = relmax(pa, e7[F7["qpi_actor"]])
        diffs["qpi_sweep_rel"] = relmax(ps, e7[F7["qpi_sweep"]])
        diffs["qpi_actor_abs"] = absmax(pa, e7[F7["qpi_actor"]])
        diffs["qpi_sweep_abs"] = absmax(ps, e7[F7["qpi_sweep"]])
        # the 4-digit file
        e4 = Q4["episodes"][find(Q4["episodes"], q, c, s)]
        diffs["q4_qpi_sweep_rel"] = relmax(ps, e4[F7["qpi_sweep"]])
        diffs["q4_q_sweep_rel"] = relmax(qs, e4[F7["q_sweep"]])
        # tail identity, own numbers: Q^pi(s_t, pi(s_t)) = sum_k<100-t g^k r_t+k + g^(100-t) * own tail
        tail_states = S_all[100][None]
        tail = []
        x = tail_states
        for _ in range(H - 1):
            x, r = TC.dynamics(x, actor(x), W.PHYSICS)
            tail.append(r[0])
        tail = np.array(tail)
        worst_tail = 0.0
        for t in range(100):
            m = 100 - t
            tot = float(np.sum(G ** np.arange(m) * R[t:])) + G ** m * float(np.sum(G ** np.arange(H - m) * tail[:H - m]))
            worst_tail = max(worst_tail, abs(tot - pa[t]) / abs(pa[t]))
        diffs["own_tail_identity_rel"] = worst_tail
        # decoded viewer values
        def dec(rows, ref):
            out = []
            for r_, v0 in zip(rows, ref):
                cs = np.cumsum(np.asarray(r_, np.float64)) / 1000.0
                out.append(v0 + cs)
            return np.array(out)
        qv_q = dec(ev_[FV["q_sweep"]], ev_[FV["q_actor"]])
        qv_p = dec(ev_[FV["qpi_sweep"]], ev_[FV["qpi_actor"]])
        diffs["viewer_q_sweep_abs"] = absmax(qv_q, qs)
        diffs["viewer_qpi_sweep_abs"] = absmax(qv_p, ps)
        diffs["viewer_q_actor_abs"] = absmax(ev_[FV["q_actor"]], qa)
        diffs["viewer_qpi_actor_abs"] = absmax(ev_[FV["qpi_actor"]], pa)
        log("  diffs " + json.dumps({k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in diffs.items()}))
        # sample table: 5 states x 6 throttles
        Xsm = np.repeat(S[STATES], len(THR), 0)
        Asm = np.repeat(C[STATES], len(THR), 0)
        Asm[:, 0] = np.tile(SWEEP[THR], len(STATES))
        psm_long = qpi(actor, Xsm, Asm, HL).reshape(len(STATES), len(THR))
        pa_long = qpi(actor, S[STATES], C[STATES], HL)
        h_rel = max(relmax(psm_long, ps[STATES][:, THR]), relmax(pa_long, pa[STATES]))
        sample = []
        for i, t in enumerate(STATES):
            for j, k in enumerate(THR):
                sample.append(dict(t=t, u=round(-SWEEP[k] * 100, 1), Q_mine=float(qs[t, k]), Q_data=e7[F7["q_sweep"]][t][k],
                                   Qpi_mine=float(ps[t, k]), Qpi_data=e7[F7["qpi_sweep"]][t][k],
                                   Q_viewer=float(qv_q[t, k]), Qpi_viewer=float(qv_p[t, k]),
                                   Qpi_1000=float(psm_long[i, j])))
        log(f"  horizon 600 vs 1000 on the sample (30 swept + 5 actor commands): max rel {h_rel:.2e}")
        for row in sample[:6] + sample[12:18]:
            log("   ", json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}))
        # summaries
        mine, hat, best = summary(C, push, qs, ps, qa, pa)
        mine["return_100_steps"], mine["points"] = J, pts
        hat_mism = int(np.sum(hat != np.asarray(e7[F7["a_hat"]])))
        best_mism = int(np.sum(best != np.asarray(e7[F7["a_best_true"]])))
        log(f"  a_hat mismatches {hat_mism}, a_best_true mismatches {best_mism}")
        if hat_mism:
            for t in np.flatnonzero(hat != np.asarray(e7[F7["a_hat"]])):
                srt = np.sort(qs[t])[::-1]
                log(f"    t={t}: mine {hat[t]} data {e7[F7['a_hat']][t]}; top two critic values {srt[0]:.7f} {srt[1]:.7f}")
        s7 = Q7["summaries"][find(Q7["summaries"], q, c, s)][3]
        s4d = Q4["summaries"][find(Q4["summaries"], q, c, s)][3]
        sv = QV["summ"][iv]
        b7 = cmp_summary(mine, s7, "precise q_data")
        b4 = cmp_summary({k: (s4(v) if isinstance(v, float) else v) for k, v in mine.items()}, s4d, "4-digit q_data")
        bv = cmp_summary({k: (s4(v) if isinstance(v, float) else v) for k, v in mine.items()}, sv, "viewer summ")
        # actor in ball and the removed-region marks, at full precision
        wj = json.loads((ROOT / q / "data" / "seed0" / "world.json").read_text())
        gap = wj["gap"]
        cen, rho = np.asarray(gap["centre"], np.float64), float(gap["rho"])
        inb = np.sqrt(np.sum((C - cen) ** 2, 1)) <= rho
        inb_mism = int(np.sum(inb.astype(int) != np.asarray(ev_[FV["actor_in_ball"]])))
        sw_in = np.sqrt(np.sum((A - cen) ** 2, 2)) <= rho      # (100, 41)
        sl = ev_[FV["slice"]]
        js_in = np.array([[bool(sl[t]) and (-SWEEP[k] * 100 >= sl[t][0] * 100 - 1e-9) and (-SWEEP[k] * 100 <= sl[t][1] * 100 + 1e-9)
                           for k in range(41)] for t in range(100)])
        log(f"  actor in ball: {int(inb.sum())} states, viewer mismatches {inb_mism}; swept marks mismatches "
            f"{int(np.sum(sw_in != js_in))}")
        REPORT["combos"][f"{q}/{c}/{s}"] = dict(hashes=hashes, J=J, Jd=Jd, points=pts, bank_J=float(Jbank.mean()),
                                                bank_J_record=Jrec, diffs=diffs, horizon_rel=h_rel, sample=sample,
                                                summary=mine, summary_mismatch=dict(precise=b7, four=b4, viewer=bv),
                                                hat_mism=hat_mism, best_mism=best_mism, inball=int(inb.sum()),
                                                inball_mism=inb_mism, swept_mark_mism=int(np.sum(sw_in != js_in)))

    # ---------------- context (first ten starts)
    REPORT["context"] = {}
    for (q, c, s) in CTX_COMBOS:
        law = LAW[s]
        ctx = RUN.Context(root=ROOT, quality=q, log=lambda m: None)
        hdir = ROOT / q / "host" / c / "seed0"
        rec = json.loads((hdir / "record.json").read_text())
        state = RUN.load_host_state(ctx, c, 0, spec)
        mean = np.asarray(rec["result"]["data"]["obs_mean"], np.float32)
        std = np.asarray(rec["result"]["data"]["obs_std"], np.float32)
        actor = Actor(state.actor.params, mean, std)
        S, C, J = rollout_many(actor, banks[law][:10])
        S, C = S.reshape(-1, 13), C.reshape(-1, 4)
        A80, A100 = C.copy(), C.copy()
        A80[:, 0], A100[:, 0] = SWEEP[I80], SWEEP[I100]
        q80, q100 = critic(state.critic.params, S, A80, mean, std)[0], critic(state.critic.params, S, A100, mean, std)[0]
        p80, p100 = qpi(actor, S, A80, H), qpi(actor, S, A100, H)
        cr, tc = q100 - q80, p100 - p80
        mine = dict(actor_thruster1_past80_share=float(np.mean(C[:, 0] < -TC.KNEE)),
                    critic_higher_at_100_than_80_share=float(np.mean(cr > 0)),
                    truth_lower_at_100_than_80_share=float(np.mean(tc < 0)),
                    critic_rises_while_truth_falls_share=float(np.mean((cr > 0) & (tc < 0))),
                    critic_change_80_to_100_mean=float(cr.mean()), truth_change_80_to_100_mean=float(tc.mean()),
                    mean_return=float(J.mean()))
        theirs = next(x for x in QV["ctx"] if (x["quality"], x["critic"], x["situation"]) == (q, c, s))
        bad = [(k, v, theirs[k]) for k, v in mine.items() if abs(s4(v) - theirs[k]) > 5e-4 * max(abs(theirs[k]), 1e-9) + 1e-12]
        log(f"context {q}/{c}/{s}: {json.dumps({k: round(v, 5) for k, v in mine.items()})}; vs viewer "
            f"{'all equal' if not bad else bad}")
        REPORT["context"][f"{q}/{c}/{s}"] = dict(mine=mine, mismatch=bad)

    # ---------------- decoded viewer values over all 20 episodes
    neg, zero_diff, hat_not_top, best_not_top, vs7 = 0, 0, 0, 0, 0.0
    worst_neg = 0.0
    examples = []
    for e, e7 in zip(QV["eps"], Q7["episodes"]):
        assert tuple(e[:3]) == tuple(e7[:3])
        for t in range(100):
            qd = e[FV["q_actor"]][t] + np.cumsum(e[FV["q_sweep"]][t]) / 1000.0
            pd = e[FV["qpi_actor"]][t] + np.cumsum(e[FV["qpi_sweep"]][t]) / 1000.0
            vs7 = max(vs7, absmax(qd, e7[F7["q_sweep"]][t]), absmax(pd, e7[F7["qpi_sweep"]][t]))
            h, b = e[FV["a_hat"]][t], e[FV["a_best_true"]][t]
            cost = pd[b] - pd[h]
            if cost < -1e-12:
                neg += 1
                worst_neg = min(worst_neg, cost)
                if len(examples) < 6:
                    examples.append((e[:3], t, h, b, round(cost, 6), e7[F7["qpi_sweep"]][t][b] - e7[F7["qpi_sweep"]][t][h]))
            elif h != b and abs(cost) < 5e-4:
                zero_diff += 1
            if qd.max() > qd[h] + 1e-12:
                hat_not_top += 1
            if pd.max() > pd[b] + 1e-12:
                best_not_top += 1
    log(f"viewer decode: worst |decoded - precise| {vs7:.2e}; 'value lost by taking a-hat' negative at {neg} states "
        f"(worst {worst_neg:.4f}), rounds to 0.000 with a-hat != true best at {zero_diff}; decoded curve higher than the "
        f"marked a-hat at {hat_not_top} states, than the marked true best at {best_not_top}")
    for ex in examples:
        log("   ", ex)
    REPORT["viewer_decode"] = dict(worst=vs7, negative_cost=neg, worst_negative=worst_neg, zero_cost_hat_ne_best=zero_diff,
                                   hat_not_top=hat_not_top, best_not_top=best_not_top)

    # ---------------- summaries in the viewer vs the 4-digit file (all 20)
    bad = []
    for sv, s4row in zip(QV["summ"], Q4["summaries"]):
        for k, v in s4row[3].items():
            if sv.get(k) != v:
                bad.append((s4row[:3], k, v, sv.get(k)))
    log(f"viewer summ vs 4-digit summaries: {'all equal' if not bad else bad[:10]}")
    (OUTD / "review_numbers.json").write_text(json.dumps(REPORT, indent=1, default=float))
    log("done")


if __name__ == "__main__":
    main()
