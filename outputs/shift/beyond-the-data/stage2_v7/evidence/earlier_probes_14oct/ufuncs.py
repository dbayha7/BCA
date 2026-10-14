import sys, numpy as np
x = np.random.default_rng(1).standard_normal(1_000_000) * 3
out = {}
for name in ("tanh", "exp", "log", "sin", "cos", "expm1", "log1p", "arctan", "sqrt", "power"):
    f = getattr(np, name)
    y = f(np.abs(x) + 1e-3) if name in ("log", "log1p", "sqrt") else (np.power(np.abs(x), 1.7) if name == "power" else f(x))
    out[name] = y
for name in ("tanh", "exp"):
    out[name + "32"] = getattr(np, name)(x.astype(np.float32))
np.savez(sys.argv[1], **out)
