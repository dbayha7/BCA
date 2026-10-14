"""READ-ONLY. Dump each seed/world's gapped training rows and calibration-bank rows (normalised obs, commands,
rewards, next obs) to this folder, through the gate's own read path (A.load_seed regenerates the log and checks it
against its record). No training, no fitting. Usage: python dump.py <seed>"""
import json
import sys
import time
import numpy as np

sys.path.insert(0, "/home/user/bca")
import experiments.shift.acceptance as A  # noqa: E402

OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
ST2 = "/home/user/bca/runs/shift_explore/stage2/detect"

p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
A.guard(p)
variant = A.gate_outcome(p)["variant"]
rep = int(sys.argv[1])
for world in ("stall", "no_stall"):
    t = time.time()
    sd = A.load_seed(p.runs, p.config, variant, world, rep)
    pg = sd.prepared("gap", p.config.learn)
    rec = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{rep}/band.json"))
    assert np.allclose(np.asarray(rec["data"]["obs_mean"], np.float32), pg.obs_mean)
    assert np.allclose(np.asarray(rec["data"]["obs_std"], np.float32), pg.obs_std)
    np.savez(f"{OUT}/data_seed{rep}_{world}.npz",
             train_obs=np.asarray(pg.train.obs), train_act=np.asarray(pg.train.action),
             train_states=np.asarray(pg.states), region=np.asarray(pg.region),
             bank_obs=np.asarray(pg.bank.obs), bank_act=np.asarray(pg.bank.action),
             bank_rew=np.asarray(pg.bank.reward), bank_next=np.asarray(pg.bank.next_obs),
             bank_rows=np.asarray(pg.bank_rows), obs_mean=pg.obs_mean, obs_std=pg.obs_std)
    print(rep, world, pg.n, len(pg.bank_rows), f"{time.time()-t:.1f}s", flush=True)
