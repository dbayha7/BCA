"""Copy what the ThrusterChain figures need out of the saved Unifloral host study. Read-only on the study.

    wsl -u root -e bash -lc 'PYTHONDONTWRITEBYTECODE=1 /opt/bca-shift-venv/bin/python -B \
        /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_env_check/extract_saved_runs.py'

The study's runs live in WSL under /root/bca-hosts/runs/shift_hosts, readable by root only. This script
plays nothing, scores nothing and trains nothing: every number it copies was computed and saved by the
study under the owner's approvals. It writes saved_runs_extract.npz and saved_runs_extract.json beside
itself, and nothing under /root.

What is copied, for each data quality (medium-expert, medium):
  the training log of dataset seed 0 (prepared.npz): every row's command, its water state, and whether the
      row was removed from training (the "ball" around the 80% hold); and the study's own removal record;
  for each of the 12 setups (TD3+BC, CQL, IQL, ReBRAC; alone, with standard BCA, with importance-weighted
      BCA) and each training seed 0 to 4, from the final actor's saved scores (scores-step100000.npz):
      the mean points and the share of steps past 80% throttle on each of the 7 banks, and the saved
      episodes of the viewer's fixed starts on the strong-current-law bank (commands, stall flags, states).
Where a run exists in both the version 3 folder and the version 4 supplement, the supplement's is used
(the study reports that every such pair reproduced).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = Path("/root/bca-hosts/runs/shift_hosts")
QUALITIES = ("medium-expert", "medium")
LEARNERS = ("td3_bc", "cql", "iql", "rebrac")
VARIANTS = ("alone", "standard", "iw")
SEEDS = range(5)
BANKS = ("none", "mild", "strong", "recorded", "strong_current", "big_waves", "big_kicks")
SHOWN_BANK = "strong"
ROOTS = (("version 4 supplement", RUNS / "supplement_v4" / "stage2"), ("version 3", RUNS / "stage2"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    arrays, meta = {}, {"source": str(RUNS), "shown_bank": SHOWN_BANK, "banks": BANKS, "qualities": {}}
    for quality in QUALITIES:
        prepared = RUNS / "data" / quality / "seed0" / "prepared.npz"
        with np.load(prepared, allow_pickle=False) as saved:
            arrays[f"{quality}/data/action"] = saved["full/action"]
            arrays[f"{quality}/data/removed"] = saved["full/in_ball"]
            arrays[f"{quality}/data/water"] = saved["full/state"][:, 10:13]
        world = json.loads((RUNS / "source" / quality / "data" / "seed0" / "world.json").read_text())
        record = {"prepared_npz_sha256": sha256(prepared), "removal": world["removal"],
                  "ball_centre": world["gap"]["centre"], "ball_radius": world["gap"]["rho"],
                  "hold_command": world["hold_command"], "runs": {}}
        for learner in LEARNERS:
            for variant in VARIANTS:
                setup = f"{learner}-{variant}"
                for seed in SEEDS:
                    found = [(tag, root / quality / f"seed{seed}" / setup / "lanes" / "scores-step100000.npz")
                             for tag, root in ROOTS]
                    found = [(tag, path) for tag, path in found if path.is_file()]
                    if not found:
                        record["runs"][f"{setup}/seed{seed}"] = None
                        continue
                    tag, path = found[0]
                    key = f"{quality}/{setup}/seed{seed}"
                    with np.load(path, allow_pickle=False) as saved:
                        arrays[f"{key}/points"] = np.array([saved[f"scored/{b}/points"].mean() for b in BANKS])
                        arrays[f"{key}/past_80_share"] = np.array([saved[f"scored/{b}/stall_share"].mean() for b in BANKS])
                        episodes = f"scored/{SHOWN_BANK}/episodes"
                        has_episodes = f"{episodes}/start" in saved.files      # the study saved them for seed 0 only
                        if has_episodes:
                            arrays[f"{key}/start"] = saved[f"{episodes}/start"]
                            arrays[f"{key}/commands"] = saved[f"{episodes}/commands"].astype(np.float32)
                            arrays[f"{key}/stalled"] = saved[f"{episodes}/stalled"]
                            arrays[f"{key}/states"] = saved[f"{episodes}/states"].astype(np.float32)
                    record["runs"][f"{setup}/seed{seed}"] = {"from": tag, "scores_npz_sha256": sha256(path),
                                                             "episodes": bool(has_episodes)}
        meta["qualities"][quality] = record
    np.savez_compressed(HERE / "saved_runs_extract.npz", **arrays)
    (HERE / "saved_runs_extract.json").write_text(json.dumps(meta, indent=1), newline="\n")
    for quality in QUALITIES:
        runs = meta["qualities"][quality]["runs"]
        missing = [name for name, value in runs.items() if value is None]
        print(f"{quality}: {sum(v is not None for v in runs.values())} of {len(runs)} runs copied; missing: {missing or 'none'}")
    print("wrote saved_runs_extract.npz and saved_runs_extract.json")


if __name__ == "__main__":
    main()
