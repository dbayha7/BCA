"""Builds qv_data.json, the compact data the page's "critic in action" viewer embeds, from
  precise/q_data.json  (compute_q.py's computation at 7 significant digits, by compute_q_precise.py)
  qv_extra.json        (ball.py: the removed ball, what each critic's training rows hold, the ball's slice per state)
It only reformats: no number is recomputed except the encoding of the sweeps (below), which it checks on the way.

Encoding of the sweeps (exact): every value in precise/q_data.json has 7 significant digits and at most 6 decimals
(its values lie between 4.4 and 298 in size). For each state, the 41 values of the critic's Q(s, a) and of the true
value Q^pi(s, a) become integers in millionths relative to the value at the actor's own command,
r_k = round(1e6 v_k) - round(1e6 v_actor), stored as r_0, r_1 - r_0, and then the second differences
(r_k - r_(k-1)) - (r_(k-1) - r_(k-2)), which are small because the curves are smooth. The viewer adds them back up twice
and divides (round(1e6 v_actor) + r_k) by 1e6, which gives exactly the double of the 7-digit value. The check: that
decoding, done the viewer's way, equals every swept value exactly.

Run (WSL, CPUs 0-3): python3 build_qv_data.py   (paths are fixed; writes qv_data.json next to this script)
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "precise" / "q_data.json"
EXTRA = HERE / "qv_extra.json"
OUT = HERE / "qv_data.json"


def s4(v):
    return float(f"{float(v):.4g}") + 0.0


SCALE = 1_000_000


def enc_sweep(row, ref):
    ra = int(round(SCALE * ref))
    r = [int(round(SCALE * v)) - ra for v in row]
    d1 = [r[k] - r[k - 1] for k in range(1, len(r))]
    return [r[0], d1[0]] + [d1[k] - d1[k - 1] for k in range(1, len(d1))], ra


def dec_sweep(enc, ra):
    """The viewer's decoding (script.js decode), step for step."""
    out, r, d = [], 0, 0
    for k, x in enumerate(enc):
        if k == 0:
            r = x
        else:
            d = x if k == 1 else d + x
            r += d
        out.append((ra + r) / SCALE)
    return out


def main():
    d = json.loads(SRC.read_text())
    x = json.loads(EXTRA.read_text())
    ck = json.loads((HERE / "precise" / "q_checks.json").read_text())
    assert ck["all_passed"] is True
    assert d["schema"] == "bca-shift-qviewer-v1-precise", d["schema"]
    assert d["meta"]["checks_all_passed"] is True
    same = d["same_as_4_digit_file"]
    assert same["episodes"] and same["indices"] and same["values_within_4_digit_rounding"], same
    F = {f: i for i, f in enumerate(d["episode_fields"])}
    assert len(x["slices"]) == len(d["episodes"])
    u = [s4(-100.0 * v) + 0.0 for v in d["sweep_a1"]]
    assert u[0] == 0.0 and u[-1] == 100.0 and u[d["i80"]] == 80.0

    sit = {}
    for name, v in d["situation_info"].items():
        sit[name] = dict(start_index=v["start_index"], current=s4(v["current"]), quantile=s4(v["current_quantile"]),
                         iqr=[s4(t) for t in v["interquartile_range"]], wave_amplitude=s4(v["wave_amplitude"]),
                         anchor_random=s4(v["anchor_random"]), anchor_top=s4(v["anchor_top"]),
                         stall_aware_from_start=s4(v["stall_aware_return_from_start"]),
                         random_from_start=s4(v["random_commands_return_from_start"]))

    eps, n_exact = [], 0
    for e, sl, ib in zip(d["episodes"], x["slices"], x["actor_in_ball"]):
        st = e[F["state"]]
        cur = {round(s[10], 6) for s in st}
        assert len(cur) == 1 and abs(next(iter(cur)) - sit[e[F["situation"]]]["current"]) < 5e-4 * abs(next(iter(cur)))
        qa, pa = e[F["q_actor"]], e[F["qpi_actor"]]
        assert len(qa) == len(pa) == len(st) == d["meta"]["steps"]
        qs, ps = [], []
        for t in range(len(st)):
            for src, ref, dst in ((e[F["q_sweep"]][t], qa[t], qs), (e[F["qpi_sweep"]][t], pa[t], ps)):
                assert ref == float(f"{ref:.7g}") and all(v == float(f"{v:.7g}") for v in src)
                enc, ra = enc_sweep(src, ref)
                assert ra == round(ref * SCALE)            # what the viewer computes from the stored actor value
                dec = dec_sweep(enc, ra)
                assert dec == list(src), (e[:3], t, [(a, b) for a, b in zip(dec, src) if a != b][:3])
                n_exact += len(src)
                dst.append(enc)
        eps.append([e[F["quality"]], e[F["critic"]], e[F["situation"]],
                    s4(e[F["return"]]), s4(e[F["discounted_return"]]), s4(e[F["points"]]),
                    [s4(v) for v in e[F["push"]]],
                    [[s4(s[2 * j]) for j in range(5)] for s in st],
                    [[s4(v) for v in c] for c in e[F["cmd"]]],
                    [s4(v) for v in e[F["reward"]]],
                    qa, pa, qs, ps, e[F["a_hat"]], e[F["a_best_true"]], sl, ib])
    fields = ["quality", "critic", "situation", "return", "discounted_return", "points", "push", "pos", "cmd",
              "reward", "q_actor", "qpi_actor", "q_sweep", "qpi_sweep", "a_hat", "a_best_true", "slice",
              "actor_in_ball"]
    summ = [dict(s[3], quality=s[0], critic=s[1], situation=s[2]) for s in d["summaries"]]
    assert [s[:3] for s in d["summaries"]] == [e[:3] for e in d["episodes"]] == [c[:3] for c in d["context"]]
    for s in summ:
        for k, v in s.items():
            if isinstance(v, float):
                s[k] = s4(v)
    ctx = [dict({k: (s4(v) if isinstance(v, float) else v) for k, v in c[3].items() if k != "returns"},
                quality=c[0], critic=c[1], situation=c[2]) for c in d["context"]]
    train = {}
    for r in x["training"]:
        train[r["quality"]] = dict(gap=r["gap"], full=r["full"], **r["world_removal"])
    out = dict(
        schema="bca-shift-qviewer-page-v2",
        notes=[
            "The critic in action, ThrusterChain: version 5 of the importance-weighting experiment, TD3+BC hosts of "
            "data seed 0, one start (index %d) of each start bank, the host's own actor driving 100 steps." %
            d["meta"]["start_index"],
            "eps: one array per (quality, critic, situation) in the order of 'fields'. push: c + w1 per state. pos: "
            "the five floats' position numbers. cmd: the actor's four throttles (negative = upstream). reward: per "
            "state. q_actor, qpi_actor: Q(s, a) = min(Q1, Q2) and Q^pi(s, a) at the actor's own command (7 significant "
            "digits).",
            "q_sweep, qpi_sweep: per state, the 41 swept thruster-1 throttles u (upstream %, thrusters 2-4 at the "
            "actor's command) as integers in millionths relative to the value at the actor's command: r_0, r_1 - r_0, "
            "then second differences. Sum twice to get r_k; value_k = (round(1e6 actor value) + r_k) / 1e6, exactly "
            "the 7-significant-digit value.",
            "a_hat, a_best_true: indices into u of the critic's highest-rated swept throttle and of the swept throttle "
            "with the highest Q^pi (computed at full precision). slice: [low, high] upstream share (0-1) in which the "
            "swept command lies inside the removed ball, or null. actor_in_ball: 1 if the actor's command lies inside it.",
            "summ, ctx: per-episode statistics and the first-ten-starts comparison, 4 significant digits; notes in "
            "summary_notes. Points: 100 (J - J_random) / (J_stall_aware - J_random) with the start bank's mean anchors.",
        ],
        u=u, i80=d["i80"], qualities=d["qualities"], critics=d["critics"], situations=d["situations"],
        sit=sit, ball=x["ball"], train=train, fields=fields, eps=eps, summ=summ, summary_notes=d["summary_notes"],
        ctx=ctx,
        meta=dict(seed=d["meta"]["seed"], steps=d["meta"]["steps"], horizon=d["meta"]["horizon"],
                  gamma=d["meta"]["gamma"], critic=d["meta"]["critic"], start_rule=d["meta"]["start_rule"],
                  start_index=d["meta"]["start_index"], spec_sha256=d["meta"]["spec_sha256"],
                  checks_all_passed=d["meta"]["checks_all_passed"], sweep_values_decoded_exactly=n_exact,
                  horizon_600_vs_1000_max_relative=float(f"{ck['worst']['horizon_600_vs_1000_max_relative']:.2g}"),
                  tail_sum_max_relative=float(f"{ck['worst']['tail_sum_max_relative']:.2g}"),
                  hosts_checked=len(ck["hosts"])))
    text = json.dumps(out, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.name}: {len(text.encode('utf-8')) / 1e6:.3f} MB; {n_exact:,} swept values decode exactly")


if __name__ == "__main__":
    main()
