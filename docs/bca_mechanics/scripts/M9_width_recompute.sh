# M9 check R1 in the pinned environment (Python 3.10.20, numpy 2.2.6, jax 0.6.2, flax 0.10.7): run from Windows with
#   pwsh -NoProfile -File <scratchpad>\wslrun.ps1 -ScriptFile docs\bca_mechanics\scripts\M9_width_recompute.sh -TimeoutSec 900 -Name m9width
# The job runs as root at /home/user/bca (the owner's clone, read only). The Python between the PY markers is
# docs/bca_mechanics/scripts/M9_width_recompute.py, copied in verbatim (the owner's clone does not hold docs/bca_mechanics).
cd /home/user/bca || exit 1
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
mkdir -p /tmp/bridge/work/m9
cat > /tmp/bridge/work/m9/M9_width_recompute.py <<'PY'
"""M9 real-data check R1: recompute BCA's band width at the actor's command from recorded inputs and compare it with
the recorded width scores of stage 2's detection test.

What it recomputes.  U(s, a) = R * u_f * max(eta_f(s, a), 1e-6) in float32 (module M9, line M9.1), read at the
24,500 detection pairs of each seed: the no-caution actor's own command, that command capped at 80% throttle, the
recorded command, the cloned actor's command and the knee command, at training states (sets T1, T2), strong-current
rollout states (T3) and recorded rows (T4).  Inputs (read only): the saved band (eta_f's parameters, R, u_f) and the
pairs; nothing is trained.  Two routes:
  (a) stage 2's own code path, band.width_rows (outputs/shift/beyond-the-data/stage2/code/band.py:141-155, 189-203),
      the function the recorded scores were written with: expected equal bit for bit;
  (b) the weighting test's expression, lower_edge.band_width (experiments/shift_iw/lower_edge.py:196-201), or a plain
      re-typing of the formula when that module cannot be imported: expected within one float32 rounding step
      (the multiplications are associated differently).

Environment.  The project's pinned environment inside WSL (Python 3.10.20, numpy 2.2.6, jax 0.6.2, flax 0.10.7),
run through the bridge job docs/bca_mechanics/scripts/M9_width_recompute.sh, which copies this file into the WSL
scratch folder and runs it against the owner's clone (read only):
    PYTHONPATH=/home/user/bca /tmp/claude-0/bcavenv/bin/python M9_width_recompute.py --repo /home/user/bca         --root /home/user/bca/runs/shift_explore
Output of 20 October 2026 (saved as docs/bca_mechanics/data/M9/width_recompute_pinned.out): route (a) equal on all 5
seeds; route (b) max |difference| 6.0e-8 to 1.2e-7, one float32 rounding step.  The Windows venv (numpy 2.5.4,
jax 0.12.0, flax 0.12.10, PYTHONPATH=M:\) gives the same two results on copies of the run files (route (b) only:
the stage 2 code and lower_edge.py import only in the pinned environment).

Recorded inputs under --root:
    detect/seed<K>/stall/pairs.npz                                   (states float64, commands float32, set, role, pair)
    stage2/detect/uniform/stall/fixed/seed<K>/band.json, band.msgpack
    stage2/detect/uniform/stall/fixed/seed<K>/scores/bca_width_uniform_fixed.npz   (the recorded widths, float64)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--repo", default=os.environ.get("BCA_REPO") or os.path.abspath(os.path.join(HERE, "..", "..", "..")))
REPO = _pre.parse_known_args()[0].repo          # the repository root (the clone whose code and run files are read)
STAGE2 = os.path.join(REPO, "outputs", "shift", "beyond-the-data", "stage2", "code")
for p in (REPO, STAGE2):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax                      # noqa: E402
import jax.numpy as jnp         # noqa: E402
from flax import serialization  # noqa: E402

from calibration.network import Calibrator  # noqa: E402

DX, DA, EPS = 13, 4, 1e-6


def load_stage2():
    try:
        import band as B            # stage 2's band.py (its import needs algorithms/, calibration/, runtime/)
        return B
    except Exception as e:          # noqa: BLE001
        print("stage 2's band.py could not be imported here:", repr(e)[:200])
        return None


def load_lower_edge():
    try:
        from experiments.shift_iw import lower_edge as LE   # refuses numpy != 2.2.6 outside the pinned environment
        return LE
    except Exception as e:          # noqa: BLE001
        print("experiments.shift_iw.lower_edge could not be imported here:", repr(e)[:200])
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO, help="repository root holding calibration/, experiments/ and the stage 2 code")
    ap.add_argument("--root", default="/home/user/bca/runs/shift_explore")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    print(f"python {sys.version.split()[0]} numpy {np.__version__} jax {jax.__version__}")
    B, LE = load_stage2(), load_lower_edge()
    cal = B.calibrator(DX) if B else Calibrator(jnp.zeros(DX), jnp.ones(DX), state_dep=True)
    tmpl = (B.band_template(DX, DA, False) if B
            else cal.init(jax.random.PRNGKey(0), jnp.zeros(DX)[None], jnp.zeros(DA)[None]))
    print(f"{'seed':4s} {'R':>9s} {'u_f':>9s} {'U_max':>7s} {'pairs':>6s} | {'(a) stage 2 code path':>22s} | "
          f"{'(b) lower_edge / plain formula':>30s}")
    ok_a, ok_b = True, True
    for seed in [int(s) for s in a.seeds.split(",")]:
        d = os.path.join(a.root, "stage2", "detect", "uniform", "stall", "fixed", f"seed{seed}")
        rec = json.load(open(os.path.join(d, "band.json")))
        blob = open(os.path.join(d, "band.msgpack"), "rb").read()
        raw = serialization.msgpack_restore(blob)
        R, unit, U_max = float(raw["R"]), float(raw["unit"]), float(raw["U_max"])
        assert R == rec["uniform"]["R"] and unit == rec["uniform"]["unit"]
        pz = np.load(os.path.join(a.root, "detect", f"seed{seed}", "stall", "pairs.npz"))
        mean = np.asarray(rec["data"]["obs_mean"], np.float32)
        std = np.asarray(rec["data"]["obs_std"], np.float32)
        obs = (pz["states"].astype(np.float32) - mean) / std                  # the learners' normalisation
        act = pz["commands"].astype(np.float32)
        recorded = np.load(os.path.join(d, "scores", "bca_width_uniform_fixed.npz"))["scores"]
        chunks = range(0, len(obs), 8192)
        if B:
            band = serialization.from_bytes(tmpl, blob)
            Ua = np.asarray(B.width_rows(cal, band, obs, act, False), np.float64)          # [M9.1] via stage 2
            cal_params = band.cal_params
        else:
            cal_params = serialization.from_state_dict(tmpl, raw["cal_params"])
            Ua = None
        if LE:
            fb = jax.jit(lambda o, x: LE.band_width(cal_params, jnp.float32(unit), o, x, jnp.float32(R)))
        else:
            fb = jax.jit(lambda o, x: jnp.float32(R) * jnp.float32(unit) * jnp.maximum(cal.apply(cal_params, o, x), EPS))
        Ub = np.concatenate([np.asarray(fb(jnp.asarray(obs[i:i + 8192]), jnp.asarray(act[i:i + 8192])), np.float64)
                             for i in chunks])
        da = (f"max diff {np.max(np.abs(Ua - recorded)):.1e}, equal {np.array_equal(Ua, recorded)}" if Ua is not None
              else "not available")
        db = f"max diff {np.max(np.abs(Ub - recorded)):.1e}, equal {np.array_equal(Ub, recorded)}"
        ok_a &= Ua is not None and bool(np.array_equal(Ua, recorded))
        ok_b &= bool(np.max(np.abs(Ub - recorded)) <= 2.0e-7)
        print(f"{seed:4d} {R:9.6f} {unit:9.6f} {U_max:7.4f} {len(recorded):6d} | {da:>22s} | {db:>30s}")
    print("RESULT (a):", "every recomputed width equals its recorded value" if ok_a else "not bit-exact / unavailable")
    print("RESULT (b):", "within one float32 rounding step (2e-7) of the record" if ok_b else "DIFFERENCES FOUND")


if __name__ == "__main__":
    main()
PY
echo "commit: $(git log --oneline -1)"
PYTHONPATH=/home/user/bca /tmp/claude-0/bcavenv/bin/python /tmp/bridge/work/m9/M9_width_recompute.py --repo /home/user/bca --root /home/user/bca/runs/shift_explore 2>&1 | grep -v "^WARNING"
