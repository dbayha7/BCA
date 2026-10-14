import time, sys
t=time.perf_counter()
import numpy as np
import experiments.shift.acceptance as A
import experiments.shift.learners as L
import experiments.shift.thruster_chain as TC
prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT)); A.guard(prereg)
cfg=prereg.config; outcome=A.gate_outcome(prereg); variant=outcome['variant']
sd=A.load_seed(prereg.runs, cfg, variant, 'stall', 0)
pg=sd.prepared('gap', cfg.learn)
print('load s', time.perf_counter()-t)
for a in ['states','region','n','eval_obs','train','obs_mean','obs_std','actions']:
    x=getattr(pg,a,None); print(a, type(x).__name__, getattr(x,'shape',None) if x is not None and not hasattr(x,'obs') else (x.obs.shape if x is not None else None))
print([k for k in dir(pg) if not k.startswith('_')])
print('KNEE',TC.KNEE,'DX',TC.DX,'DS',TC.DS, hasattr(TC,'push'))
