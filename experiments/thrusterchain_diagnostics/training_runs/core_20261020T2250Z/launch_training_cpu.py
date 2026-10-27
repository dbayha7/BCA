"""Resource-bounded launcher for the authorized Zeus diagnostic core only."""
import ctypes
import json
import os
from pathlib import Path
import runpy
import sys

if os.name != "nt":
    raise SystemExit("This launcher is for the Windows Zeus runtime.")
mode, output, *extra = sys.argv[1:]
repository = Path(r"C:\Users\David\dev\BCA").resolve()
allowed = repository / "experiments" / "thrusterchain_diagnostics" / "training_runs"
destination = Path(output).resolve()
if not destination.is_relative_to(allowed):
    raise SystemExit("Output must stay in the isolated suite's training_runs directory.")
worker = int(extra[extra.index("--worker-index") + 1]) if "--worker-index" in extra else 0
if worker not in (0, 1, 2):
    raise SystemExit("Only the three frozen worker partitions are supported.")
mask = (0xF << (12 + 4 * worker)) if mode == "run" else 0xF00
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
handle = kernel.GetCurrentProcess()
kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
if not kernel.SetPriorityClass(handle, 0x40) or not kernel.SetProcessAffinityMask(handle, mask):
    raise OSError(ctypes.get_last_error(), "Failed to enforce the declared CPU resource limits")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
os.chdir(repository)
sys.path.insert(0, str(repository))
print(json.dumps(dict(event="resource_limits", pid=os.getpid(), priority="IDLE", affinity_mask=hex(mask),
                     cores=[i for i in range(24) if mask & (1 << i)], backend="cpu", mode=mode)), flush=True)
sys.argv = ["full_run", mode, "--output", str(destination), *extra]
runpy.run_module("experiments.thrusterchain_diagnostics.full_run", run_name="__main__")
