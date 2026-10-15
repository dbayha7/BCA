"""Engineering timing only (prints elapsed seconds, not points): one 200-start, 100-step rollout of a saved Stage 2
'none' actor in another world's physics, plus one gate-critic forward pass over the visited steps. Read-only."""
import sys, time
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/consumer")
import common as C  # noqa: E402
TC, L = C.TC, C.L
t0 = time.perf_counter()
sd, pg = C.seed_data("stall", 0)
acts, _, _ = C.policy_actors("stall", 0)
pol = L.mlp_policy(acts["none"], pg, "none")
crit = C.critic(sd)
t1 = time.perf_counter()
x = np.asarray(TC.bank_starts("strong_current", TC.BANK_SIZE)[:200], np.float64)
S, A_ = [], []
phys = TC.world_physics("no_stall")
for t in range(pg.physics.length):
    a = np.clip(pol(x), -1, 1); S.append(x); A_.append(a)
    x, r = TC.dynamics(x, a, phys)
t2 = time.perf_counter()
S = np.concatenate(S); A_ = np.concatenate(A_)
mean, std = np.asarray(pg.obs_mean, np.float32), np.asarray(pg.obs_std, np.float32)
q = C.qmin(crit, (S.astype(np.float32) - mean) / std, A_)
t3 = time.perf_counter()
print(f"load {t1-t0:.1f}s rollout {t2-t1:.1f}s forward {t3-t2:.1f}s rows {len(S)} finite {bool(np.all(np.isfinite(q)))}")
