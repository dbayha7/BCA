"""Module M3, companion to M3_recompute_pinned.py: the held-out calibration rows' water current and largest throttle,
for the bands whose rows the gate's data loader can rebuild (near-80 version-6 bands and stage 2's fixed-critic bands),
so that the recorded scores can be read by current and by throttle. Nothing is trained; the rows are rebuilt by the same
loader the studies used (settingdata.gate_seed + settingdata.prepared) and checked against the band record's SHA-256.
Run inside WSL from /home/user/bca with the pinned environment (see M3_bank_rows_pinned.sh). Writes
/tmp/bridge/work/m3/M3_bank_rows.npz with, per band label, current_<label> (float32, the 11th state number) and
throttle_<label> (float32, max |a| over the 4 thrusters), plus a JSON of the SHA-256 checks.
"""
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path("/home/user/bca")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "outputs/shift/beyond-the-data/near80/code"))
import numpy as np  # noqa: E402
import settingdata as SD  # noqa: E402

OUT = Path("/tmp/bridge/work/m3")
OUT.mkdir(parents=True, exist_ok=True)
NEAR80 = ROOT / "runs/shift_explore/near80"
STAGE2 = ROOT / "runs/shift_explore/stage2/detect/uniform"
KEYS = ("rows", "train", "bank", "bank_rows", "normalization")
learn = SD.gate_prereg().config.learn
arrays, checks = {}, {}
t0 = time.perf_counter()
for study, setting, root, recname in (("near80", "near80", NEAR80, "v6_band.json"), ("stage2", "main", STAGE2, "band.json")):
    for world in ("stall", "no_stall"):
        for seed in range(5):
            sd = SD.gate_seed(seed, world)
            prep = SD.prepared(setting, sd, learn)
            if study == "near80":
                rec = json.loads((SD.learner_dir(NEAR80, "near80", world, seed) / recname).read_text())
            else:
                rec = json.loads((STAGE2 / world / "fixed" / f"seed{seed}" / recname).read_text())
            label = f"{study}_{world}_fixed_seed{seed}"
            obs = np.asarray(prep.bank.obs, np.float32)
            act = np.asarray(prep.bank.action, np.float32)
            arrays[f"current_{label}"] = obs[:, 10]
            arrays[f"throttle_{label}"] = np.abs(act).max(axis=1)
            checks[label] = dict(n=int(obs.shape[0]), bank_rows_record=int(rec["data"]["bank_rows"]),
                                 sha_equal={k: bool(prep.sha256[k] == rec["data"]["sha256"][k]) for k in KEYS})
            print(label, checks[label], f"{time.perf_counter() - t0:.0f}s", flush=True)
np.savez(OUT / "M3_bank_rows.npz", **arrays)
(OUT / "M3_bank_rows.json").write_text(json.dumps(dict(checks=checks, python=sys.version.split()[0],
                                                       numpy=np.__version__), indent=1))
print("DONE", len(checks), "bands", flush=True)
