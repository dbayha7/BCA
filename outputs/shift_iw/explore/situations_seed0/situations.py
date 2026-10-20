"""The seed-0 rerun's 15 learned actors (TD3+BC, BCA, BCA IW on the gapped log; five qualities) in the main test's four
situations. EXPLORATORY: an evaluation only. It trains nothing, fits no band and reads no approval-gated step; it loads
the finished snapshots through the rerun driver's own loader (driver/jobs.learners_of) and rolls each actor's float64
copy (truth.MLPPolicy) in the stalling world (world.PHYSICS = thruster_chain.STALL), as evaluate_actor does.

Starts: thruster_chain.bank_starts(situation), the main test's 1,000 fixed starts per situation. Replay episode: start
11 of each bank, the start the live panel's fixed-controller mode uses (script const D.start).
Points: the weighting test's scale on each situation's bank (data.anchors: 0 = the random quality's mean return,
100 = the stall-aware controller's), as the learned-actor mode shows for the two mixes.

Run from <iw_rerun>/repo with PYTHONPATH=<iw_rerun>/repo:<iw_rerun>/explore:<iw_rerun>/driver.
Writes <sit4>/situations_seed0.json.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np

import experiments.shift  # noqa: F401
from experiments.shift import thruster_chain as TC
from experiments.shift import truth as TR
from experiments.shift_iw import data as D
from experiments.shift_iw import run as R
from experiments.shift_iw import world as W
import jobs as J

OUT = Path(sys.argv[1])
START = 11
SITS = ("recorded", "big_kicks", "big_waves", "strong_current")
assert set(SITS) == set(TC.SITUATIONS)
assert W.PHYSICS is TC.STALL or W.PHYSICS == TC.STALL

t0 = time.time()
banks = {s: TC.bank_starts(s) for s in SITS}
for s, b in banks.items():
    assert b.shape == (TC.BANK_SIZE, TC.DX), (s, b.shape)
anchors = D.anchors(banks)
print("anchors", {s: (round(a["anchor"]["random"], 3), round(a["anchor"]["top"], 3)) for s, a in anchors.items()},
      f"{time.time()-t0:.0f}s", flush=True)

ctx0 = J.ctx_of("expert")
spec = R.spec_digest(ctx0)
avg, eps, facts = {}, {}, {}
for q in J.QUALITIES:
    ctx = J.ctx_of(q)
    data = R.load_data(ctx, J.SEED, spec)
    prep = D.prepare(data, "gap", ctx.cfg)
    mean, std = prep.obs_mean, prep.obs_std
    for name, (actor, critic, f) in J.learners_of(ctx, spec).items():
        facts[f"{q}/{name}"] = f
        pol = TR.MLPPolicy(actor, mean, std, name=f"{q} {name} actor (float64 copy)")
        for s in SITS:
            x0 = banks[s]
            run = TR.rollout(pol, x0, W.PHYSICS)
            Jm = float(run.J.mean())
            se = float(run.J.std(ddof=1) / math.sqrt(len(x0)))
            avg[f"{q}|{name}|{s}"] = dict(J=Jm, se=se, Jd=float(run.Jd.mean()),
                                         points=float(W.points(Jm, anchors[s]["anchor"])),
                                         points_se=100.0 * se / (anchors[s]["anchor"]["top"] - anchors[s]["anchor"]["random"]))
            rec = TR.rollout(pol, x0[START][None], W.PHYSICS, length=100, record=True)
            assert abs(float(rec.J[0]) - float(run.J[START])) <= 1e-9 * max(1.0, abs(float(run.J[START])))
            eps[f"{q}|{name}|{s}"] = dict(quality=q, learner=name, situation=s, start_index=START,
                                          J=float(rec.J[0]), Jd=float(rec.Jd[0]),
                                          points=float(W.points(rec.J[0], anchors[s]["anchor"])),
                                          states=rec.states[0][:100].tolist(), cmds=rec.commands[0].tolist(),
                                          rewards=rec.rewards[0].tolist(), final_state=rec.states[0][100].tolist())
        print(q, name, " ".join(f"{s} {avg[f'{q}|{name}|{s}']['points']:.1f}" for s in SITS), f"{time.time()-t0:.0f}s", flush=True)

body = dict(kind="iw-rerun-situations-seed0",
            label="EXPLORATORY: outside the pre-registered claims; evaluation of the seed-0 rerun's actors only",
            seed=J.SEED, start_index=START, steps=100, situations=list(SITS), learners=list(J.LEARNERS),
            qualities=list(J.QUALITIES), world="stall",
            starts="thruster_chain.bank_starts(situation), 1,000 per situation (the main test's fixed starts)",
            points_rule="weighting-test scale per situation bank: 100 (J - J_random) / (J_stall_aware - J_random) (world.points, data.anchors)",
            anchors={s: dict(anchor=a["anchor"], controllers=a["controllers"]) for s, a in anchors.items()},
            bank_start11={s: banks[s][START].tolist() for s in SITS},
            learner_facts=facts, avg=avg, episodes=eps,
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seconds=round(time.time() - t0, 1))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(body))
print("written", OUT, f"{time.time()-t0:.0f}s")
