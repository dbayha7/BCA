#!/bin/sh
# Module M7, a side check of data/M7_near80_training_dose.json: the normalization constants in each near-80 band record
# (v6_band.json data.obs_mean / obs_std) against the mean and spread of that setting's own training file. They differ
# by up to 5.6e-3 because every setting uses the gate's normalization, the states kept under the main test's removal
# (outputs/shift/beyond-the-data/near80/code/settingdata.py:7, 154), so near80 and random_rows share one set of constants.
# The dose readout used the record's constants, as the learner did. Pinned environment through the bridge:
#   pwsh -NoProfile -File <scratchpad>\wslrun.ps1 -ScriptFile docs\bca_mechanics\scripts\M7_near80_normalization_check.sh -TimeoutSec 600 -Name m7norm
# Output on record: data/M7_near80_normalization_check.out
cd /home/user/bca || exit 97
export PYTHONDONTWRITEBYTECODE=1
/tmp/claude-0/bcavenv/bin/python - <<'PY'
import json, numpy as np, h5py
root = "/home/user/bca/runs/shift_explore/near80"
for setting in ("near80", "random_rows"):
    for world in ("stall", "no_stall"):
        for seed in range(5):
            rec = json.load(open(f"{root}/learners/{setting}/{world}/seed{seed}/v6_band.json"))
            with h5py.File(f"{root}/data/{setting}/{world}/seed{seed}/train.hdf5", "r") as f:
                obs = np.asarray(f["observations"][()], np.float32)
            mean, std = obs.mean(0), obs.std(0) + np.float32(0.001)
            jm, js = np.asarray(rec["data"]["obs_mean"], np.float32), np.asarray(rec["data"]["obs_std"], np.float32)
            print(f"{setting}/{world}/seed{seed}: max|mean diff| {np.abs(mean-jm).max():.2e} max|std diff| {np.abs(std-js).max():.2e}  (record mean[10] {jm[10]:.5f} recomputed {mean[10]:.5f})")
PY
