"""READ-ONLY: current range of the whole log (states only) and of the strong-current evaluation starts."""
import sys
import numpy as np
sys.path.insert(0, "/home/user/bca")
import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402
p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT)); A.guard(p); variant = A.gate_outcome(p)["variant"]
for rep in [int(s) for s in sys.argv[1:]]:
    for world in ("stall", "no_stall"):
        lg = A.load_seed(p.runs, p.config, variant, world, rep).log
        c = np.asarray(lg["infos/state"])[:, TC.CURRENT]
        print(rep, world, "log current min/max", float(c.min()), float(c.max()), flush=True)
x = TC.bank_starts("strong_current", TC.BANK_SIZE)[:200]
print("strong-current starts current min/max", float(x[:, TC.CURRENT].min()), float(x[:, TC.CURRENT].max()),
      "amplitude min/max", float(TC.wave_amplitude(x).min()), float(TC.wave_amplitude(x).max()),
      "share amplitude > 0.25", float((TC.wave_amplitude(x) > 0.25).mean()))
