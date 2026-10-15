"""Engineering timing only: CPU seconds of one certificate look on synthetic scores (no route data)."""
import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/bca"); sys.path.insert(0, "/home/user/bca/outputs/shift/beyond-the-data/restructure/step1")
import numpy as np
import step1lib as S1
spec, _ = S1.load_prespec(); cfg = spec["certificate"]
g = np.random.default_rng(1)
for nt, nr in ((700, 300), (7000, 3000), (14000, 6000)):
    t = time.process_time(); c = S1.certificate(g.normal(size=nt), g.normal(size=nr), [1, 2, 3], cfg)
    print(nt, nr, f"{time.process_time() - t:.2f}s cpu")
