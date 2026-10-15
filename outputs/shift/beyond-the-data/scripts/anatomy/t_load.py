import time, numpy as np
from common import *
t=time.perf_counter()
S = Seed(0, worlds=("stall","no_stall"))
print("load", S.load_seconds, S.variant, S.alpha)
pg = S.pg
print(pg.n, pg.region.mean(), pg.states.shape)
for k in H2_CRITICS:
    c = S.h2_critic(k); print(k, c is not None)
acts, rec = S.h2_actors(); print(list(acts))
racts, blind, vals = S.room_actors(); print(list(racts))
print("pattern", S.pattern)
sa = RC.stall_aware(S.physics)
x = pg.states[pg.region][:5]
print(sa(x))
print(time.perf_counter()-t)
