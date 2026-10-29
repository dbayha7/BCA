"""Run a notebook's code cells in order in a real IPython kernel (jupyter_client + ipykernel) and save every cell's
text outputs (stream, text/plain, text/markdown; images noted only) to a JSON file. Nothing is written outside the
run folder: the kernel's cwd, Jupyter's runtime/data/config dirs, IPython's dir and Matplotlib's config dir all point
into it. Usage: python run_nb.py NOTEBOOK OUT_JSON RUN_DIR [TIMEOUT_SECONDS_PER_CELL]
"""
import json
import os
import sys
import time

NB, OUT, RUN = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
CELL_TIMEOUT = float(sys.argv[4]) if len(sys.argv) > 4 else 900.0
for sub in ("jrt", "jdata", "jconf", "ipy", "mpl"):
    os.makedirs(os.path.join(RUN, "_env", sub), exist_ok=True)
env = dict(os.environ)
env.update({
    "PYTHONDONTWRITEBYTECODE": "1", "JAX_PLATFORMS": "cpu",
    "JUPYTER_RUNTIME_DIR": os.path.join(RUN, "_env", "jrt"),
    "JUPYTER_DATA_DIR": os.path.join(RUN, "_env", "jdata"),
    "JUPYTER_CONFIG_DIR": os.path.join(RUN, "_env", "jconf"),
    "IPYTHONDIR": os.path.join(RUN, "_env", "ipy"),
    "MPLCONFIGDIR": os.path.join(RUN, "_env", "mpl"),
    "JUPYTER_PLATFORM_DIRS": "1",
})
os.environ.update({k: env[k] for k in ("JUPYTER_RUNTIME_DIR", "JUPYTER_DATA_DIR", "JUPYTER_CONFIG_DIR", "IPYTHONDIR",
                                       "MPLCONFIGDIR", "PYTHONDONTWRITEBYTECODE", "JUPYTER_PLATFORM_DIRS")})
sys.dont_write_bytecode = True

from jupyter_client import KernelManager  # noqa: E402

nb = json.load(open(NB, encoding="utf-8"))
km = KernelManager(kernel_name="python3")
km.kernel_cmd = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
km.start_kernel(cwd=RUN, env=env)
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=120)
# the kernel itself must not write bytecode caches either
kc.execute_interactive("import sys; sys.dont_write_bytecode = True", timeout=60)

results = []
t0 = time.perf_counter()
try:
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        outs = []

        def hook(msg, outs=outs):
            t, ct = msg["header"]["msg_type"], msg["content"]
            if t == "stream":
                outs.append({"type": "stream", "name": ct["name"], "text": ct["text"]})
            elif t in ("execute_result", "display_data"):
                d = ct.get("data", {})
                o = {"type": t}
                if "text/markdown" in d:
                    o["markdown"] = d["text/markdown"]
                if "text/plain" in d and "image/png" not in d:
                    o["plain"] = d["text/plain"]
                if "image/png" in d:
                    o["image"] = True
                outs.append(o)
            elif t == "error":
                outs.append({"type": "error", "ename": ct["ename"], "evalue": ct["evalue"],
                             "tb": "\n".join(ct.get("traceback", []))[-3000:]})
        reply = kc.execute_interactive(src, timeout=CELL_TIMEOUT, output_hook=hook, store_history=False)
        status = reply["content"]["status"]
        results.append({"cell": i, "status": status, "outputs": outs})
        print(f"cell {i}: {status} ({time.perf_counter() - t0:.1f} s)", flush=True)
        if status != "ok":
            break
finally:
    kc.stop_channels()
    km.shutdown_kernel(now=True)
json.dump({"notebook": NB, "seconds": round(time.perf_counter() - t0, 1), "cells": results},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("done", OUT)
