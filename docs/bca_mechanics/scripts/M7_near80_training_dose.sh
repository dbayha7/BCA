#!/bin/sh
# Module M7 (the dose), real-data part: BCA's dose at the RECORDED TRAINING ROWS of the near-80 run (version 8),
# recomputed from recorded inputs in the project's pinned environment. Nothing is trained: the band (eta_f, u_f, R) is
# read from runs/shift_explore/near80/learners/<setting>/<world>/seed<K>/v6_band.msgpack (its SHA-256 checked against
# v6_band.json), the rows from runs/shift_explore/near80/data/<setting>/<world>/seed<K>/train.hdf5, and the dose is
# computed by calibration/dose.py's own frozen_level_dose, the function algorithms/td3_bc_bca.bc_readout calls (the
# readout point of module M7: the recorded state and the recorded command of each row).
#
# Environment: the pinned WSL environment (Ubuntu-24.04, /home/user/bca, Python 3.10.20, NumPy 2.2.6, JAX 0.6.2, CPU),
# run through the owner's bridge from Windows:
#   pwsh -NoProfile -File <scratchpad>\wslrun.ps1 -ScriptFile docs\bca_mechanics\scripts\M7_near80_training_dose.sh -TimeoutSec 1200 -Name m7dose
# or directly inside WSL:  sh docs/bca_mechanics/scripts/M7_near80_training_dose.sh
# Output: /tmp/bridge/work/m7/M7_near80_training_dose.json (a copy is kept in docs/bca_mechanics/data/) and a table.
cd /home/user/bca || exit 97
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/bca JAX_PLATFORMS=cpu
mkdir -p /tmp/bridge/work/m7
/tmp/claude-0/bcavenv/bin/python - <<'PY'
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/bca/outputs/shift/beyond-the-data/stage2/code")   # version 6's band.py (pure functions)
import band as B                      # outputs/shift/beyond-the-data/stage2/code/band.py (the band near-80 restates)
import stage2_common as C
import jax, jax.numpy as jnp, h5py
from calibration import dose as D, reference as REF

DX, DA, BLEND = 13, 4, 0.5
cal = B.calibrator(DX)
root = "/home/user/bca/runs/shift_explore/near80"
out = dict(label="M7 real-data test: BCA's dose at the recorded training rows of the near-80 run (version 8); "
                 "nothing trained; pinned environment", python=sys.version.split()[0], numpy=np.__version__,
           jax=jax.__version__, blend=BLEND, bands=[])
print("python %s numpy %s jax %s" % (sys.version.split()[0], np.__version__, jax.__version__))
print("%-28s %6s %7s %7s %8s %9s %7s %7s %7s %7s %4s %9s %9s refused valid" % (
    "band", "rows", "R", "unit", "eta mean", "dose mean", "min", "p5", "p95", "max", "inv", "max drop", "cf gap"))
for setting in ("near80", "random_rows"):
    for world in ("stall", "no_stall"):
        for seed in range(5):
            tag = "%s/%s/seed%d" % (setting, world, seed)
            rec = json.load(open("%s/learners/%s/v6_band.json" % (root, tag)))
            band = C.load_tree("%s/learners/%s/v6_band.msgpack" % (root, tag), B.band_template(DX, DA, False),
                               rec["band_file_sha256"])
            assert float(band.R) == rec["uniform"]["R"] and float(band.unit) == rec["uniform"]["unit"], tag
            with h5py.File("%s/data/%s/%s/seed%d/train.hdf5" % (root, setting, world, seed), "r") as f:
                obs = np.asarray(f["observations"][()], np.float32)
                act = np.asarray(f["actions"][()], np.float32)
            mean, std = obs.mean(0), obs.std(0) + np.float32(0.001)      # experiments/shift/learners.normalization
            jm, js = np.asarray(rec["data"]["obs_mean"], np.float32), np.asarray(rec["data"]["obs_std"], np.float32)
            norm_ok = bool(np.allclose(mean, jm, atol=2e-6) and np.allclose(std, js, atol=2e-6))
            obs_n = ((obs - jm) / js).astype(np.float32)               # the gate's normalisation (learners._transition)
            # the readout path of td3_bc_bca.bc_readout: eta_f at the recorded (s_i, a_i), then frozen_level_dose
            ref = REF.FrozenReference(band.cal_params, band.unit, band.R, band.R, band.R,
                                      jnp.asarray(float(rec["uniform"]["bank_rows"]), jnp.float32), jnp.asarray(True))

            @jax.jit
            def readout(o, a):
                eta = cal.apply(ref.cal_params, o, a)
                d = D.frozen_level_dose(ref, eta, BLEND)
                return eta, d.width, d.dose, d.support_mask, d.inputs_valid

            etas, widths, doses, support, valid = [], [], [], [], True
            for i in range(0, len(obs_n), 8192):
                e, w, d, s, v = readout(jnp.asarray(obs_n[i:i + 8192]), jnp.asarray(act[i:i + 8192]))
                etas.append(np.asarray(e, np.float64)); widths.append(np.asarray(w, np.float64))
                doses.append(np.asarray(d, np.float64)); support.append(np.asarray(s, bool)); valid &= bool(v)
            eta, U, m, sup = map(np.concatenate, (etas, widths, doses, support))
            u = float(np.float32(band.unit))
            # the weighting test's M3 statistics (experiments/shift_iw/coverage.dose_order) on these rows: falls of
            # the dose as the width rises, and the gap to the closed form 1 + 0.5 U/(U + u) in float64
            order = np.lexsort((m, U)); step = np.diff(m[order])
            closed = 1.0 + BLEND * U / (U + u)
            thr = np.abs(act).max(1)
            groups = {"throttle_at_most_70": thr <= 0.7, "throttle_70_to_90": (thr > 0.7) & (thr <= 0.9),
                      "throttle_past_90": thr > 0.9}
            hist_edges = np.linspace(1.0, 4.0, 31)
            hist = np.histogram(U / u, bins=hist_edges)[0]
            entry = dict(setting=setting, world=world, seed=seed, rows=int(len(m)), R=float(band.R), unit=u,
                         normalisation_matches_record=norm_ok, bank_rows=int(rec["uniform"]["bank_rows"]),
                         eta=dict(mean=float(eta.mean()), min=float(eta.min()), max=float(eta.max())),
                         width_over_unit=dict(mean=float((U / u).mean()), min=float((U / u).min()),
                                              max=float((U / u).max())),
                         dose=dict(mean=float(m.mean()), sd=float(m.std()), min=float(m.min()), max=float(m.max()),
                                   p5=float(np.percentile(m, 5)), p50=float(np.percentile(m, 50)),
                                   p95=float(np.percentile(m, 95))),
                         dose_by_throttle={k: dict(rows=int(v.sum()), mean=float(m[v].mean()) if v.any() else None)
                                           for k, v in groups.items()},
                         in_range=bool(np.all(m >= 1.0) and np.all(m < 1.5)),
                         inversions=int(np.sum(step < 0)), max_drop=float(max(0.0, -float(step.min()))),
                         max_closed_form_gap=float(np.abs(m - closed).max()),
                         refused_rows=int((~np.isfinite(U)).sum()), supported_all=bool(sup.all()),
                         inputs_valid=valid,
                         width_over_unit_histogram=dict(edges=hist_edges.tolist(), counts=hist.tolist()))
            out["bands"].append(entry)
            print("%-28s %6d %7.4f %7.4f %8.4f %9.4f %7.4f %7.4f %7.4f %7.4f %4d %9.2e %9.2e %7d %s" % (
                tag, len(m), float(band.R), u, eta.mean(), m.mean(), m.min(), np.percentile(m, 5),
                np.percentile(m, 95), m.max(), int(np.sum(step < 0)), max(0.0, -float(step.min())),
                np.abs(m - closed).max(), int((~np.isfinite(U)).sum()), valid))
json.dump(out, open("/tmp/bridge/work/m7/M7_near80_training_dose.json", "w"), indent=1)
print("written /tmp/bridge/work/m7/M7_near80_training_dose.json")
PY
echo "exit $?"
