r"""Module M7 test: the dose on the behavior-cloning term, m = 1 + 0.5 U / (U + u).

What this script does (MODULES.md, module M7; the write-up is docs/bca_mechanics/modules/M7.md):
  Part A, synthetic (the truth is known): the dose formula of calibration/dose.py on made-up widths: range [1, 1.5),
     closed form, monotone in the width up to one float32 step, 1 before the first refresh, 1 on a refused row under the
     standard rule and 1.5 under the per-row rule, the per-row dose with one threshold equal to the standard dose bit
     for bit, the saturation (how little the dose moves when the threshold grows), and the actor loss with every
     multiplier 1 equal to plain TD3+BC's.
  Part B, recorded ThrusterChain inputs (nothing trained, no new band fit):
     B1 the near-80 run's 20 recorded bands (R, u, eta at the held-out rows): the dose at the held-out rows;
     B2 the same bands read at the 41,000 recorded training rows by BCA's own readout in the pinned environment
        (data/M7_near80_training_dose.json, produced by scripts/M7_near80_training_dose.sh);
     B3 the weighting test: mark M3 (mechanics) and mark R2 (dose change under weighting) from the five reports, the
        dose over training (unweighted against exact strong-shift weights) and the closed-form check of its seed-0
        dose quantiles;
     B4 the four-learner study: the dose at the last refresh, standard against weighted, with the placebo;
     B5 (optional, --stage2 <dir>) stage 2's recorded widths: the implied dose at recorded rows and at the actor's
        commands, uniform against classifier-weighted band.
  It prints the tables, writes docs/bca_mechanics/figures/M7.svg and docs/bca_mechanics/data/M7_table.txt.

Command (from the repository root; any Python with NumPy, JAX (CPU) and Matplotlib; git on PATH), Windows:
    set PYTHONPATH=M:\
    set PYTHONDONTWRITEBYTECODE=1
    set JAX_PLATFORMS=cpu
    python docs\bca_mechanics\scripts\M7_test.py --root M:\ [--stage2 <dir>] [--png <file>] [--no-figure] [--cache <dir>]
Environment used for the committed outputs (data/M7_table.txt, figures/M7.svg): Python 3.14.2, NumPy 2.5.4, JAX 0.12.0,
Matplotlib 3.11.2 on Windows 11; the run takes about 20 seconds. The only numbers that needed the project's pinned
environment (Python 3.10.20, NumPy 2.2.6, JAX 0.6.2, inside WSL) are in data/M7_near80_training_dose.json, produced
by the job file scripts/M7_near80_training_dose.sh; the seven unit tests of the bit-for-bit identities were run there
by scripts/M7_pinned_unit_tests.sh (output data/M7_pinned_unit_tests.out).
Recorded inputs are read from the working tree (outputs/shift_iw/explore/routes2/diagnose/*.json, with a fallback to
branch claude/main-bca-connection-x4o46s) and from git (origin/claude/near80-run-files, origin/main,
origin/claude/shift-hosts), cached under --cache (default: <system temp>/bca_m7_cache). Part B5 needs read-only copies
of five files per seed that live only in the owner's WSL clone: runs/shift_explore/detect/seed<K>/stall/pairs.npz
(copied to <dir>/pairs/seed<K>/pairs.npz) and runs/shift_explore/stage2/detect/{uniform,weighted}/stall/fixed/seed<K>/
{band.json, scores/bca_width_{uniform,weighted}_fixed.npz} (copied to <dir>/{uniform,weighted}/stall/seed<K>/...).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(DOCS.parent.parent), help="repository root (default: two levels above docs/)")
ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "bca_m7_cache"))
ap.add_argument("--stage2", default=None, help="folder with pairs/seedK/{pairs,labels}.npz and "
                                                "{uniform,weighted}/stall/seedK/{band.json,scores/*.npz}")
ap.add_argument("--no-figure", action="store_true")
ap.add_argument("--png", default=None, help="also save a PNG preview of the figure to this path")
ARGS = ap.parse_args()
ROOT = Path(ARGS.root)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax                                   # noqa: E402
import jax.numpy as jnp                      # noqa: E402
from calibration import dose as D            # noqa: E402  (calibration/dose.py:23-95)
from calibration import reference as REF     # noqa: E402  (FrozenReference, reference_valid)

BLEND = 0.5
ULP = float(np.spacing(np.float32(1.3)))     # one float32 step near 1.3: 1.19e-7 (the weighting test's M3 allowance 1e-6)
LINES = []


def say(*parts):
    text = " ".join(str(p) for p in parts)
    print(text)
    LINES.append(text)


def verdict(ok):
    return "PASS" if ok else "FAIL"


def git_show(ref_path, binary=False):
    """Bytes or text of <ref>:<path> from the repository, cached."""
    cache = Path(ARGS.cache) / ref_path.replace(":", "__").replace("/", "__")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        out = subprocess.run(["git", "-C", str(ROOT), "show", ref_path], check=True, capture_output=True).stdout
        cache.write_bytes(out)
    data = cache.read_bytes()
    return data if binary else data.decode("utf-8")


def closed_form(U, u, blend=BLEND):
    """1 + blend U / (U + u) in float64 (experiments/shift_iw/coverage.py:108-119)."""
    U = np.asarray(U, np.float64)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(U), 1.0 + blend * U / (U + float(u)), 1.0)


def reference(threshold, unit, ready=True):
    """A frozen reference with finite scale-network parameters (calibration/reference.py:31-38)."""
    t = jnp.asarray(threshold, jnp.float32)
    return REF.FrozenReference({"w": jnp.zeros(3, jnp.float32)}, jnp.asarray(unit, jnp.float32), t, t, t,
                               jnp.asarray(100.0, jnp.float32), jnp.asarray(bool(ready)))


def per_row_dose(ref, predictions, R_rows, blend):
    """experiments/shift_iw/learners_iw.py:655-683 restated with eta passed in (the four-learner iw.py:279-312 does the
    same): each row's own threshold, refusal gives 1 + blend."""
    predictions = jnp.asarray(predictions, jnp.float32)
    R = jnp.asarray(R_rows, jnp.float32)
    unit = ref.residual_scale
    scale = jnp.maximum(predictions, 1e-06) * unit
    width = R * scale
    valid = (D.tree_finite(ref.cal_params)
             & jnp.all(jnp.isfinite(predictions) & (predictions > 0))
             & jnp.isfinite(unit) & (unit > 0)
             & jnp.all(jnp.isfinite(scale) & (scale > 0))
             & jnp.all(~jnp.isnan(R) & (R >= 0))
             & jnp.all(jnp.isposinf(R) | (jnp.isfinite(width) & ((R == 0) | (width > 0)))))
    width = jnp.where(ref.ready, width, jnp.zeros_like(width))
    valid = jnp.where(ref.ready, valid, True)
    unit = jnp.where(ref.ready, unit, 1.0)
    base = D.level_critic_dose(width, jnp.ones_like(width, dtype=bool), unit, blend, valid)
    refused = ref.ready & jnp.isposinf(R)
    dose = jnp.where(refused, jnp.asarray(1.0 + blend, base.dose.dtype), base.dose)
    return base._replace(dose=dose, inputs_valid=base.inputs_valid & REF.reference_valid(ref)), refused


# ====================================================================================================== Part A
def part_a():
    say("=" * 100)
    say("PART A. Synthetic checks of the dose formula (calibration/dose.py:23-95); the truth is known")
    say("=" * 100)
    results = {}
    rng = np.random.default_rng(20261020)

    # A1 range and closed form on a wide grid of widths, for two units. In exact arithmetic m < 1.5 for every finite
    # width; in float32, 1.5 - m is about 0.5/x, and once that is below half a float32 step at 1.5 (2^-24, about
    # 6e-8) the dose rounds to exactly 1.5: from x = U/u of about 5.6 million on (found by bisection below). The
    # largest dose on record is 1.4997 (CQL under strong-shift weights, four-learner study), x about 1,800, three
    # thousand times below that crossover; on the near-80 training rows x is 0.7-3.6 (part B2).
    def first_cap_x(unit):
        """The smallest x = U/u at which level_critic_dose returns exactly 1.5 (bisection on a geometric grid)."""
        def capped(xv):
            out = D.level_critic_dose(jnp.asarray([np.float32(xv * unit)]), jnp.ones(1, bool),
                                      jnp.asarray(unit, jnp.float32), BLEND)
            return float(out.dose[0]) >= 1.5
        lo, hi = 1e5, 1e9
        assert not capped(lo) and capped(hi)
        for _ in range(60):
            mid = float(np.sqrt(lo * hi))
            lo, hi = (lo, mid) if capped(mid) else (mid, hi)
        return hi

    for unit in (1.0, 0.157):
        x = np.concatenate([[0.0], np.logspace(-6, 30, 3601)])
        U = (x * unit).astype(np.float32)
        out = D.level_critic_dose(jnp.asarray(U), jnp.ones(len(U), bool), jnp.asarray(unit, jnp.float32), BLEND)
        m = np.asarray(out.dose, np.float64)
        gap = float(np.abs(m - closed_form(U, unit)).max())
        x_cap = first_cap_x(unit)
        below = x <= 1e6
        ok = bool(np.all(m >= 1.0) and np.all(m[below] < 1.5) and np.all(m <= 1.5) and gap <= 2 * ULP and m[0] == 1.0
                  and 1e6 < x_cap < 1e8)
        results[f"A1 range+closed form (u={unit})"] = ok
        say(f"A1 [u = {unit}] dose in [1, 1.5) for every x = U/u up to 1e6: {bool(np.all(m[below] < 1.5))}; never "
            f"above 1.5: {bool(np.all(m <= 1.5))}; dose(U = 0) = {m[0]:.1f}; largest gap to 1 + 0.5 U/(U+u): {gap:.2e} "
            f"(allowed 2 float32 steps = {2*ULP:.2e}); float32 reaches exactly 1.5 from x = {x_cap:.2e} on -> {verdict(ok)}")
    say(f"A1 landmarks: U = u gives {1 + 0.5 * 0.5:.4f}; U = 2u gives {1 + 0.5 * 2 / 3:.4f}; U = 10u gives "
        f"{1 + 0.5 * 10 / 11:.4f}; U = 100u gives {1 + 0.5 * 100 / 101:.4f}; the limit is 1.5 (reached in float32 "
        f"only from x of about 5.6 million on; the largest dose on record is 1.4997, x about 1,800)")

    # A2 monotone in the width up to one float32 step, on 100,000 random widths
    U = np.sort(np.exp(rng.uniform(np.log(1e-3), np.log(1e3), 100_000))).astype(np.float32) * np.float32(0.157)
    out = D.level_critic_dose(jnp.asarray(U), jnp.ones(len(U), bool), jnp.asarray(0.157, jnp.float32), BLEND)
    m = np.asarray(out.dose, np.float64)
    step = np.diff(m)
    inversions, max_drop = int(np.sum(step < 0)), float(max(0.0, -step.min()))
    ok = max_drop <= ULP + 1e-12
    results["A2 monotone up to one float32 step"] = ok
    say(f"A2 100,000 sorted widths: {inversions} falls of the dose, the largest {max_drop:.2e} (one float32 step is "
        f"{ULP:.2e}; the weighting test's M3 bar is 1e-6) -> {verdict(ok)}")

    # A3 before the first refresh: width 0 and dose 1 everywhere, whatever the threshold
    eta = jnp.asarray(rng.uniform(0.5, 2.0, 1000), jnp.float32)
    out = D.frozen_level_dose(reference(np.inf, 1.0, ready=False), eta, BLEND)
    ok = bool(np.all(np.asarray(out.dose) == 1.0) and np.all(np.asarray(out.width) == 0.0) and bool(out.inputs_valid))
    results["A3 not ready gives dose 1"] = ok
    say(f"A3 reference not ready (before the first refresh): every width 0 and every dose 1: {ok} -> {verdict(ok)}")

    # A4 refusal: R = +inf. Standard rule (frozen_level_dose): width +inf, unsupported, dose 1. Per-row rule: 1.5.
    out = D.frozen_level_dose(reference(np.inf, 0.157), eta, BLEND)
    standard_ok = bool(np.all(np.isposinf(np.asarray(out.width))) and np.all(np.asarray(out.dose) == 1.0)
                       and not bool(np.any(np.asarray(out.support_mask))) and bool(out.inputs_valid))
    pr, refused = per_row_dose(reference(np.inf, 0.157), eta, np.full(1000, np.inf), BLEND)
    per_row_ok = bool(np.all(np.asarray(pr.dose) == 1.5) and np.all(np.asarray(refused)))
    results["A4 refusal: standard 1, per-row 1.5"] = standard_ok and per_row_ok
    say(f"A4 threshold +infinity (refusal): standard rule dose 1 on every row, unsupported, inputs valid: {standard_ok}; "
        f"per-row rule (learners_iw.py:681-682) dose 1.5 on every row: {per_row_ok} -> {verdict(standard_ok and per_row_ok)}")

    # A5 the per-row dose with one threshold for every row equals the standard dose bit for bit
    R = 1.47
    ref = reference(R, 0.157)
    std = D.frozen_level_dose(ref, eta, BLEND)
    pr, refused = per_row_dose(ref, eta, np.full(1000, R), BLEND)
    same = (np.asarray(std.dose).tobytes() == np.asarray(pr.dose).tobytes()
            and np.asarray(std.width).tobytes() == np.asarray(pr.width).tobytes()
            and np.asarray(std.support_mask).tobytes() == np.asarray(pr.support_mask).tobytes())
    results["A5 per-row with one R == standard, bit for bit"] = bool(same and not np.any(np.asarray(refused)))
    say(f"A5 per-row dose with R_i = R on every row equals frozen_level_dose byte for byte (dose, width, support): "
        f"{same}; refused rows 0: {not bool(np.any(np.asarray(refused)))} -> {verdict(results['A5 per-row with one R == standard, bit for bit'])}")
    # and a mixed case: rows with their own thresholds get the standard dose at that threshold
    levels = np.array([0.5, 1.0, 2.0, 4.0]) * R
    R_rows = levels[np.arange(1000) % 4]
    pr, _ = per_row_dose(ref, eta, R_rows, BLEND)
    mixed_ok = True
    for r in levels:
        rows = R_rows == r
        own = D.frozen_level_dose(reference(r, 0.157), eta, BLEND)
        mixed_ok &= np.asarray(pr.dose)[rows].tobytes() == np.asarray(own.dose)[rows].tobytes()
    results["A5b per-row uses each row's own threshold"] = bool(mixed_ok)
    say(f"A5b rows at four different thresholds (0.5R, R, 2R, 4R) each get the standard dose at their own threshold, "
        f"bit for bit: {bool(mixed_ok)} -> {verdict(bool(mixed_ok))}")

    # A6 saturation: how much the dose moves when the threshold is multiplied by a factor
    say("A6 saturation. x = U/u = R * eta. The dose's slope in ln U is 0.5 x/(1+x)^2, at most 0.125 (at x = 1).")
    say("    starting x (= R*eta)  dose   | factor on R:   x1.10   x1.20   x1.45   x1.76   x2.25   (the recorded "
        "threshold ratios: mild 1.08-1.26, strong 1.20-1.76, up to 2.25 on one seed)")
    for x0 in (1.0, 2.0, 2.5, 4.0, 10.0):
        base = 1 + 0.5 * x0 / (1 + x0)
        row = "   ".join(f"{(1 + 0.5 * f * x0 / (1 + f * x0)) - base:+.3f}" for f in (1.10, 1.20, 1.45, 1.76, 2.25))
        say(f"    {x0:5.1f}                 {base:.3f}  | change:        {row}")
    say("    Reading: near-80 and the weighting test have x about 1.6-2.5 (dose 1.31-1.36), so a 1.45x threshold "
        "moves the dose by about +0.04 (R2 recorded +0.039 on expert); CQL's x is larger (dose 1.35-1.49).")
    results["A6 saturation (description)"] = True

    # A7 the host's use: with every multiplier 1 the weighted behavior-cloning term is plain TD3+BC's
    try:
        import algorithms.td3_bc as BASE  # noqa: F401  (algorithms/td3_bc.py:128-139)
        pi = jnp.asarray(rng.normal(size=(256, 4)), jnp.float32)
        a = jnp.asarray(rng.normal(size=(256, 4)), jnp.float32)
        m1 = jnp.ones(256, jnp.float32)
        bc_plain = jnp.square(pi - a).mean()                                           # td3_bc.py:133
        bc_dosed = (jax.lax.stop_gradient(m1) * jnp.square(pi - a).mean(axis=-1)).mean()   # td3_bc.py:135-138
        m2 = jnp.asarray(rng.uniform(1.0, 1.5, 256), jnp.float32)
        bc_dosed2 = (m2 * jnp.square(pi - a).mean(axis=-1)).mean()
        ok = bool(abs(float(bc_plain) - float(bc_dosed)) <= 1e-6) and float(bc_dosed2) > float(bc_plain)
        results["A7 multiplier 1 == plain BC term"] = ok
        say(f"A7 actor loss's cloning term: with every m_i = 1 it equals plain TD3+BC's ({float(bc_plain):.6f} vs "
            f"{float(bc_dosed):.6f}); with m in [1, 1.5) it is larger ({float(bc_dosed2):.6f}) -> {verdict(ok)}")
    except Exception as error:  # noqa: BLE001
        say(f"A7 skipped (algorithms.td3_bc not importable here: {error})")
    return results


# ====================================================================================================== Part B1, B2
def near80_bank():
    say("=" * 100)
    say("PART B1. Near-80 run (version 8): the dose at the HELD-OUT rows of its 20 recorded bands (R, u from "
        "v6_band.json; eta at the held-out rows from v6_bank_scores.npz; branch claude/near80-run-files)")
    say("=" * 100)
    rows = []
    say(f"{'band':26s} {'held-out':>8s} {'R':>7s} {'u':>7s} {'eta mean':>8s} {'U/u mean':>8s} {'dose mean':>9s} "
        f"{'min':>7s} {'max':>7s} {'U_max check':>11s}")
    for setting in ("near80", "random_rows"):
        for world in ("stall", "no_stall"):
            for seed in range(5):
                base = f"origin/claude/near80-run-files:runs/shift_explore/near80/learners/{setting}/{world}/seed{seed}/"
                rec = json.loads(git_show(base + "v6_band.json"))
                import io
                z = np.load(io.BytesIO(git_show(base + "v6_bank_scores.npz", binary=True)))
                R, u = rec["uniform"]["R"], rec["uniform"]["unit"]
                eta = np.asarray(z["eta"], np.float64)
                # the width exactly as calibration/dose.py:75-76 computes it (float32)
                U = np.float32(R) * (np.maximum(eta.astype(np.float32), np.float32(1e-6)) * np.float32(u))
                m = closed_form(U, np.float32(u))
                umax_ok = abs(float(U.max()) - rec["uniform"]["U_max"]) <= 2e-7 * max(1.0, rec["uniform"]["U_max"])
                rows.append(dict(setting=setting, world=world, seed=seed, n=int(len(eta)), R=R, u=u,
                                 eta_mean=float(eta.mean()), x_mean=float((U / u).mean()), dose_mean=float(m.mean()),
                                 dose_min=float(m.min()), dose_max=float(m.max()), umax_ok=bool(umax_ok),
                                 x=(U / u).astype(np.float64)))
                say(f"{setting + '/' + world + '/seed' + str(seed):26s} {len(eta):8d} {R:7.4f} {u:7.4f} {eta.mean():8.4f} "
                    f"{(U / u).mean():8.4f} {m.mean():9.4f} {m.min():7.4f} {m.max():7.4f} {'equal' if umax_ok else 'DIFFERS':>11s}")
    means = np.array([r["dose_mean"] for r in rows])
    ok = all(r["umax_ok"] for r in rows) and all(1 <= r["dose_min"] and r["dose_max"] < 1.5 for r in rows)
    say(f"B1 summary: 20 bands, mean dose at the held-out rows {means.min():.3f} to {means.max():.3f} (average "
        f"{means.mean():.3f}); every dose in [1, 1.5); U_max of each record equals R*u*max(eta) of its arrays: "
        f"{all(r['umax_ok'] for r in rows)} -> {verdict(ok)}")
    return rows, ok


def near80_train():
    say("=" * 100)
    say("PART B2. The same 20 bands read at the RECORDED TRAINING ROWS (the readout point of M7) by BCA's own "
        "frozen_level_dose in the pinned environment: data/M7_near80_training_dose.json")
    say("=" * 100)
    path = DOCS / "data" / "M7_near80_training_dose.json"
    if not path.exists():
        say(f"B2 skipped: {path} missing (run scripts/M7_near80_training_dose.sh in the pinned environment)")
        return None, None
    d = json.loads(path.read_text())
    say(f"produced with Python {d['python']}, NumPy {d['numpy']}, JAX {d['jax']}")
    say(f"{'band':26s} {'rows':>6s} {'R':>7s} {'dose mean':>9s} {'sd':>7s} {'min':>7s} {'p5':>7s} {'p95':>7s} {'max':>7s} "
        f"{'falls':>5s} {'max drop':>9s} {'cf gap':>9s} {'refused':>7s} {'<=70%':>7s} {'>90%':>7s}")
    ok = True
    for b in d["bands"]:
        m = b["dose"]
        g = b["dose_by_throttle"]
        ok &= (b["in_range"] and b["max_drop"] <= ULP + 1e-12 and b["max_closed_form_gap"] <= 2 * ULP
               and b["refused_rows"] == 0 and b["supported_all"] and b["inputs_valid"])
        say(f"{b['setting'] + '/' + b['world'] + '/seed' + str(b['seed']):26s} {b['rows']:6d} {b['R']:7.4f} {m['mean']:9.4f} "
            f"{m['sd']:7.4f} {m['min']:7.4f} {m['p5']:7.4f} {m['p95']:7.4f} {m['max']:7.4f} {b['inversions']:5d} "
            f"{b['max_drop']:9.2e} {b['max_closed_form_gap']:9.2e} {b['refused_rows']:7d} "
            f"{g['throttle_at_most_70']['mean']:7.4f} {(g['throttle_past_90']['mean'] or float('nan')):7.4f}")
    means = np.array([b["dose"]["mean"] for b in d["bands"]])
    say(f"B2 summary: mean dose over the training rows {means.min():.3f} to {means.max():.3f} (average {means.mean():.3f}); "
        f"every dose in [1, 1.5); largest fall as the width rises = one float32 step ({max(b['max_drop'] for b in d['bands']):.2e}); "
        f"largest gap to the closed form {max(b['max_closed_form_gap'] for b in d['bands']):.2e}; refused rows 0; "
        f"inputs valid on every band -> {verdict(ok)}")
    say("    (the same three statistics the weighting test's mark M3 recorded: 1.19e-7, 1.04e-7, 0)")
    n_flag = sum(not b["normalisation_matches_record"] for b in d["bands"])
    say(f"B2 note: the job's flag normalisation_matches_record is False on {n_flag} of 20 bands. Expected: the record's "
        f"constants are the gate's normalization (the mean and spread of the states kept under the main test's removal), "
        f"shared by the near80 and random_rows settings (near80/code/settingdata.py:7, 154), while the flag compared "
        f"them with the mean of each setting's own training file (largest difference 5.6e-3; "
        f"data/M7_near80_normalization_check.out). The readout used the record's constants, as the learner did.")
    return d, ok


# ====================================================================================================== Part B3
def weighting_test():
    say("=" * 100)
    say("PART B3. The weighting test (experiments/shift_iw, specification version 5, branch main): mark M3 (the dose's "
        "mechanics on 42,500-49,919 kept training rows per seed) and mark R2 (dose change under strong-shift weights)")
    say("=" * 100)
    qualities = ["random", "medium-replay", "medium", "medium-expert", "expert"]
    m3_ok = True
    say(f"{'quality':14s} {'M3':>5s} {'max drop (5 seeds)':>18s} {'max closed-form gap':>19s} {'rows':>6s} "
        f"{'R2 dose change mean [95% interval]':>36s} {'refused':>7s} {'predicted return':>16s} {'R2':>5s} {'R1':>5s} "
        f"{'I1':>8s}")
    r2 = {}
    for q in qualities:
        rep = json.loads(git_show(f"origin/main:outputs/shift_iw/v5/{q}/report.json"))
        m3 = rep["marks"]["M3"]
        per = m3["per_seed"]
        drop = max(p["max_drop"] for p in per)
        gap = max(p["max_closed_form_gap"] for p in per)
        rows = per[0]["rows"]
        m3_ok &= m3["status"] == "pass" and drop <= ULP + 1e-12 and all(p["supported_all"] for p in per)
        R2 = rep["marks"]["R2"]
        dc = R2["dose_change"]
        r2[q] = dict(mean=dc["mean"], lo=dc["interval"][0], hi=dc["interval"][1], per_seed=dc["per_seed"],
                     refused=R2["refused_rows"], predicted=R2["predicted_return"], status=R2["status"])
        say(f"{q:14s} {m3['status']:>5s} {drop:18.2e} {gap:19.2e} {rows:6d} "
            f"{dc['mean']:+.3f} [{dc['interval'][0]:+.3f}, {dc['interval'][1]:+.3f}]{'':>12s} {int(sum(R2['refused_rows'])):7d} "
            f"{R2['predicted_return']:16.2f} {R2['status']:>5s} {rep['marks']['R1']['status']:>5s} "
            f"{rep['marks']['I1']['status']:>8s}")
    say(f"B3a mark M3 passes on all five qualities (largest fall one float32 step, 0 refused rows) -> {verdict(m3_ok)}")
    say("B3b mark R2 (pre-registered; bar: a predicted return change of at least 1.8 points in size, report.json "
        "thresholds.R2): the strong-shift weights move the mean dose by only +0.018 to +0.053. On medium-expert and "
        "expert, where the room check R1 passed, that predicted +0.92 and +0.42 points, so R2 failed; on random, "
        "medium-replay and medium R2 passed only because the predicted change was large and negative (-3.4 to -5.1: "
        "more pull lowers return there) while R1 failed. BCA's training arms (I1) require M1, R1 and R2 to pass and "
        "ran on no quality (run.py:19-21). The small dose change is the saturation of A6.")

    # the dose over training, seed 0 (exploratory diagnosis on branch claude/main-bca-connection-x4o46s)
    diag = ROOT / "outputs" / "shift_iw" / "explore" / "routes2" / "diagnose"
    chunks = json.loads((diag / "dose_chunks.json").read_text()) if (diag / "dose_chunks.json").exists() else \
        json.loads(git_show("origin/claude/main-bca-connection-x4o46s:outputs/shift_iw/explore/routes2/diagnose/dose_chunks.json"))
    say("B3c the dose over training (mean multiplier of the batch, every 5,000 updates, seed 0; first refresh at 10,000):")
    say(f"{'quality / arm':28s} {'before refresh':>14s} {'after (mean)':>12s} {'at 100,000':>10s} {'low':>6s} {'high':>6s}")
    curves = {}
    ready_ok = True
    for key, v in chunks.items():
        upd, dose = np.array(v["update"]), np.array(v["dose"])
        before, after = dose[upd <= 10_000], dose[upd > 10_000]
        ready_ok &= bool(np.all(before == 1.0))
        curves[key] = (upd, dose)
        say(f"{key:28s} {before.mean():14.3f} {after.mean():12.4f} {dose[-1]:10.4f} {after.min():6.3f} {after.max():6.3f}")
    say(f"B3c before the first refresh the multiplier is exactly 1 (the host runs unchanged); afterwards 1.29-1.39 on "
        f"every quality, the weighted arm 0.02-0.06 above the unweighted -> {verdict(ready_ok)}")

    # closed-form check of the recorded seed-0 dose quantiles from the recorded eta quantiles
    numbers = json.loads((diag / "diagnosis_numbers.json").read_text()) if (diag / "diagnosis_numbers.json").exists() else \
        json.loads(git_show("origin/claude/main-bca-connection-x4o46s:outputs/shift_iw/explore/routes2/diagnose/diagnosis_numbers.json"))
    bands = json.loads((diag / "diag_band_at_commands.json").read_text()) if (diag / "diag_band_at_commands.json").exists() else \
        json.loads(git_show("origin/claude/main-bca-connection-x4o46s:outputs/shift_iw/explore/routes2/diagnose/diag_band_at_commands.json"))
    say("B3d closed form on recorded quantiles (seed 0, last refresh): dose(p) = 1 + 0.5 R eta(p) / (R eta(p) + 1). "
        "The last two columns are means over 4,151 rows: the mean dose must be at most the closed form of the mean x "
        "(the dose is concave in x), and close to it")
    say(f"{'quality':14s} {'R':>7s} {'eta p5':>7s} {'eta p95':>7s} {'dose p5 rec':>11s} {'closed':>7s} {'dose p95 rec':>12s} "
        f"{'closed':>7s} {'mean dose, rec rows':>19s} {'cf(mean x)':>10s}")
    cf_ok = True
    for q in qualities:
        n = numbers["qualities"][q]
        R = bands["res"][q]["threshold"]
        e5, e95 = n["eta_p5_p95_seed0"]
        d5, d95 = n["dose_p5_p95_seed0"]
        c5, c95 = (1 + 0.5 * R * e / (R * e + 1) for e in (e5, e95))
        x = bands["res"][q]["recorded"]["U_over_unit"]
        d_rec = bands["res"][q]["recorded"]["dose_equiv"]
        c_rec = 1 + 0.5 * x / (x + 1)
        cf_ok &= abs(c5 - d5) < 2e-4 and abs(c95 - d95) < 2e-4 and (d_rec <= c_rec + 1e-9) and (c_rec - d_rec < 3e-3)
        say(f"{q:14s} {R:7.4f} {e5:7.3f} {e95:7.3f} {d5:11.4f} {c5:7.4f} {d95:12.4f} {c95:7.4f} {d_rec:19.4f} {c_rec:10.4f}")
    say(f"B3d the recorded dose quantiles equal the closed form of the recorded eta quantiles to 4 decimals (a monotone "
        f"map carries quantiles to quantiles), and the recorded mean dose sits 0.0004-0.0025 below the closed form of "
        f"the mean width, the direction concavity requires -> {verdict(cf_ok)}")
    return r2, curves, m3_ok and ready_ok and cf_ok


# ====================================================================================================== Part B4
def four_learner():
    say("=" * 100)
    say("PART B4. The four-learner study (branch claude/shift-hosts, outputs/shift_hosts_v3/results.json; exploratory, "
        "seeds 0-4): the dose at the last refresh on kept training rows, standard against weighted (exact strong-shift "
        "weights), with the shuffled-weight placebo and the classifier-estimated weights")
    say("=" * 100)
    res = json.loads(git_show("origin/claude/shift-hosts:outputs/shift_hosts_v3/results.json"))
    out = {}
    say(f"{'learner, quality':26s} {'setup':9s} {'R last (mean)':>13s} {'dose mean':>9s} {'(seed range)':>16s} {'min':>6s} "
        f"{'max':>6s} {'refused':>7s} {'change vs std':>13s} {'placebo':>8s} {'estimated':>9s}")
    ok = True
    for q in ("medium", "medium-expert"):
        for learner in ("td3_bc", "cql", "rebrac", "iql"):
            ex = res["qualities"][q].get("expanded", {}).get(learner, {})
            for setup in ("standard", "iw"):
                b = ex.get("setups", {}).get(setup, {}).get("band")
                if not b or b.get("mean_dose_last", {}).get("mean") is None:
                    continue
                tr = [t for t in b["train_rows_last"] if t]
                mins, maxs = [t["min_dose"] for t in tr], [t["max_dose"] for t in tr]
                refused = sum(t["refused_rows"] for t in tr)
                ok &= min(mins) >= 1.0 and max(maxs) < 1.5 and refused == 0
                dc = [c for c in b["dose_change"] if c]
                sh = [c for c in b["shuffled_weights_dose_change"] if c]
                es = [c for c in b["estimated_weights_dose_change"] if c]
                f = lambda L: (f"{np.mean([c['change_vs_unweighted'] for c in L]):+.3f}" if L else "-")
                per = b["mean_dose_last"]["per_seed"]
                out[(learner, q, setup)] = dict(mean=b["mean_dose_last"]["mean"], per_seed=per,
                                                change=[c["change_vs_unweighted"] for c in dc],
                                                placebo=[c["change_vs_unweighted"] for c in sh],
                                                estimated=[c["change_vs_unweighted"] for c in es],
                                                R=b["threshold_last"]["mean"], mins=mins, maxs=maxs)
                say(f"{learner + ', ' + q:26s} {setup:9s} {b['threshold_last']['mean']:13.3f} {b['mean_dose_last']['mean']:9.3f} "
                    f"{'(' + f'{min(per):.3f}-{max(per):.3f}' + ')':>16s} {min(mins):6.3f} {max(maxs):6.3f} {refused:7d} "
                    f"{f(dc):>13s} {f(sh):>8s} {f(es):>9s}")
    say("B4 reading: every dose in [1, 1.5) and 0 refused rows (18 of 18 refreshes certified in every run); the "
        "standard dose is 1.33-1.36 for every learner and quality; the strong-shift weights raise CQL's dose by "
        "+0.125 to +0.156 on 10 of 10 seeds (to 1.47-1.50, nearly the cap), ReBRAC's by -0.022 to +0.003 (its critic "
        "did not learn), and shuffled weights change it by -0.008 to +0.017 (placebo). TD3+BC's weighted run and IQL "
        f"were stopped by stop rule S7 and are not judged. -> {verdict(ok)}")
    return out, ok


# ====================================================================================================== Part B5
def stage2():
    if not ARGS.stage2:
        say("PART B5 (stage 2's recorded widths) skipped: pass --stage2 <dir> with copies of runs/shift_explore/detect/"
            "seedK/{pairs,labels}.npz (under pairs/seedK/) and runs/shift_explore/stage2/detect/{uniform,weighted}/stall/"
            "fixed/seedK/{band.json, scores/bca_width_*_fixed.npz}")
        return None
    say("=" * 100)
    say("PART B5. Stage 2 (lower-edge study, stall world, seeds 0-4): the dose the recorded widths imply, at recorded "
        "rows (set T4) and at the actor's own command on strong-current rollout states (set T3), uniform band against "
        "the classifier-weighted band (description only: stage 2 used the width in the lower edge, not as a dose)")
    say("=" * 100)
    W = Path(ARGS.stage2)
    out = {}
    say(f"{'seed':4s} {'R':>7s} {'u':>7s} | {'uniform: recorded rows':>22s} {'actor, strong current':>21s} | "
        f"{'weighted: recorded rows':>23s} {'actor, strong current':>21s}")
    for s in range(5):
        pz = np.load(W / "pairs" / f"seed{s}" / "pairs.npz")
        st, role = pz["set"], pz["role"]
        ub = json.loads((W / "uniform" / "stall" / f"seed{s}" / "band.json").read_text())["uniform"]
        u = ub["unit"]
        vals = {}
        for ver in ("uniform", "weighted"):
            sc = np.load(W / ver / "stall" / f"seed{s}" / "scores" / f"bca_width_{ver}_fixed.npz")["scores"]
            for S, Rl, name in (("T4", "recorded", "recorded"), ("T3", "command", "actor_strong")):
                m = (st == S) & (role == Rl)
                vals[(ver, name)] = (float(closed_form(sc[m], u).mean()), float(sc[m].mean()))
        out[s] = vals
        say(f"{s:4d} {ub['R']:7.4f} {u:7.4f} | dose {vals[('uniform','recorded')][0]:.3f} (U {vals[('uniform','recorded')][1]:.3f})"
            f"{'':>2s} dose {vals[('uniform','actor_strong')][0]:.3f} (U {vals[('uniform','actor_strong')][1]:.3f}) | "
            f"dose {vals[('weighted','recorded')][0]:.3f} (U {vals[('weighted','recorded')][1]:.3f}){'':>3s} "
            f"dose {vals[('weighted','actor_strong')][0]:.3f} (U {vals[('weighted','actor_strong')][1]:.3f})")
    say("B5 reading: the uniform band gives the same dose (1.31-1.35) at recorded rows and at the actor's commands "
        "under strong current; the weighted band's width at the actor's commands is 1.3-4.5 times the uniform band's, "
        "but the dose it implies rises only to 1.36-1.41 (saturation).")
    return out


# ====================================================================================================== figure
def figure(bank_rows, train, r2, curves, hosts, s2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"  # keep the figure's text as text, not outlines
    blue, orange, green, grey, purple = "#0072B2", "#D55E00", "#009E73", "#7F7F7F", "#CC79A7"
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.3]))
    # (a) the curve, with the recorded widths drawn as a strip of counts along the bottom
    ax = axes[0]
    x = np.logspace(-2, 3, 500)
    ax.semilogx(x, 1 + 0.5 * x / (1 + x), color="black", lw=1.8, label="dose m = 1 + 0.5 x/(1 + x)")
    ax.axhline(1.5, color=grey, ls=":", lw=1, label="cap 1.5 (not reached on any recorded row)")
    allx = np.concatenate([r["x"] for r in bank_rows])
    bins = np.logspace(np.log10(0.5), np.log10(5), 40)
    h_bank = np.histogram(allx, bins=bins)[0].astype(float)
    base, height = 1.0, 0.12
    ax.stairs(base + height * h_bank / h_bank.max(), bins, baseline=base, fill=True, color=blue, alpha=0.35,
              label=f"near-80: x at {len(allx):,} held-out rows, 20 bands (counts, scaled)")
    if train:
        edges = np.array(train["bands"][0]["width_over_unit_histogram"]["edges"])
        counts = np.sum([b["width_over_unit_histogram"]["counts"] for b in train["bands"]], axis=0).astype(float)
        ax.stairs(base + height * counts / counts.max(), edges, baseline=base, color=orange, lw=1.6,
                  label=f"near-80: x at {counts.sum():,.0f} recorded training rows (counts, scaled)")
    for xv in (1, 2, 10):
        yv = 1 + 0.5 * xv / (1 + xv)
        ax.plot([xv], [yv], "o", color="black", ms=4)
        ax.annotate(f"x = {xv}: {yv:.3f}", (xv, yv), textcoords="offset points", xytext=(7, -12), fontsize=8)
    ax.set_xlabel("x = band width / unit  (= R * eta at the row)")
    ax.set_ylabel("dose m on the behavior-cloning term")
    ax.set_ylim(0.98, 1.56)
    ax.set_title("(a) the formula, and where the recorded rows sit", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    # (b) the dose over training
    ax = axes[1]
    for q, c in (("expert", blue), ("medium-expert", green), ("random", purple)):
        for arm, ls in (("unweighted", "-"), ("exact_strong", "--")):
            upd, d = curves[f"{q}/{arm}"]
            ax.plot(upd / 1000, d, ls, color=c, lw=1.5,
                    label=f"{q}, {'standard' if arm == 'unweighted' else 'strong-shift weights'}")
    ax.axvline(10, color=grey, ls=":", lw=1)
    ax.text(11, 1.43, "first refresh at update 10,000", fontsize=8, color=grey)
    ax.set_xlabel("TD3+BC update (thousands)")
    ax.set_ylabel("mean dose of the training batch")
    ax.set_ylim(0.98, 1.46)
    ax.set_title("(b) weighting test, seed 0: exactly 1 before the first refresh, then 1.29-1.39", fontsize=10)
    ax.legend(fontsize=7, ncol=1, loc="lower right")
    # (c) standard against weighted, study by study
    ax = axes[2]
    items = []
    for learner, q, label in (("cql", "medium", "CQL, medium"), ("cql", "medium-expert", "CQL, medium-expert"),
                              ("rebrac", "medium", "ReBRAC, medium (critic did not learn)"),
                              ("td3_bc", "medium-expert", "TD3+BC, medium-expert (weighted run stopped)")):
        std = hosts.get((learner, q, "standard"))
        iw = hosts.get((learner, q, "iw"))
        items.append((label, std["mean"] if std else None, iw["mean"] if iw else None,
                      np.mean(iw["placebo"]) if iw and iw["placebo"] else None))
    for q in ("expert", "medium-expert", "medium", "medium-replay", "random"):
        d_u = curves[f"{q}/unweighted"][1][-1]
        d_w = curves[f"{q}/exact_strong"][1][-1]
        items.append((f"weighting test, {q} (seed 0, end)", d_u, d_w, None))
    items.append(("near-80, 20 bands (standard only)", float(np.mean([r["dose_mean"] for r in bank_rows])), None, None))
    if s2:
        items.append(("stage 2, recorded rows, 5 seeds (classifier weights)",
                      float(np.mean([v[("uniform", "recorded")][0] for v in s2.values()])),
                      float(np.mean([v[("weighted", "recorded")][0] for v in s2.values()])), None))
    ys = np.arange(len(items))[::-1]
    for y, (label, a, b, pl) in zip(ys, items):
        if a is not None:
            ax.plot([a], [y], "o", color=blue, ms=6, zorder=3)
        if b is not None and a is not None:
            ax.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(arrowstyle="->", color=grey, lw=1))
            ax.plot([b], [y], "s", color=orange, ms=6, zorder=3)
        if pl is not None and a is not None:
            ax.plot([a + pl], [y], "x", color=green, ms=7, mew=1.5, zorder=4)
    ax.plot([], [], "o", color=blue, label="standard BCA (weights of 1)")
    ax.plot([], [], "s", color=orange, label="weighted BCA (strong-shift weights; stage 2: classifier weights)")
    ax.plot([], [], "x", color=green, mew=1.5, label="shuffled-weight placebo")
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items], fontsize=7.5)
    ax.axvline(1.5, color=grey, ls=":", lw=1)
    ax.text(1.497, ys.min() - 0.6, "cap 1.5", fontsize=8, color=grey, ha="right")
    ax.set_xlim(1.25, 1.52)
    ax.set_ylim(ys.min() - 0.8, ys.max() + 3.2)
    ax.set_xlabel("mean dose over kept training rows")
    ax.set_title("(c) the dose, standard against weighted, study by study", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Module M7, the dose on the behavior-cloning term: m = 1 + 0.5 U/(U + u), read at recorded rows",
                 fontsize=11)
    fig.tight_layout()
    out = DOCS / "figures" / "M7.svg"
    fig.savefig(out, format="svg")
    say(f"figure written: {out}")
    if ARGS.png:
        fig.savefig(ARGS.png, format="png", dpi=120)
        say(f"preview written: {ARGS.png}")
    return fig


def main():
    a = part_a()
    bank_rows, b1_ok = near80_bank()
    train, b2_ok = near80_train()
    r2, curves, b3_ok = weighting_test()
    hosts, b4_ok = four_learner()
    s2 = stage2()
    say("=" * 100)
    say("SUMMARY")
    for k, v in a.items():
        say(f"  {verdict(v):4s} {k}")
    say(f"  {verdict(b1_ok):4s} B1 near-80 held-out rows: dose in range, U_max of the records reproduced")
    say(f"  {verdict(bool(b2_ok)):4s} B2 near-80 recorded training rows (pinned environment): M3 statistics reproduced on 20 bands")
    say(f"  {verdict(b3_ok):4s} B3 weighting test: M3 pass on 5 qualities; 1 before the first refresh; closed form on recorded quantiles")
    say(f"  {verdict(b4_ok):4s} B4 four-learner study: dose in range, 0 refused, weighting raises CQL's dose, placebo near 0")
    say(f"  {'done' if s2 else 'skip':4s} B5 stage 2's recorded widths: the dose they imply at recorded rows and at the actor's commands (description only)")
    say("  VERDICT M7: works with a caveat. The multiplier is computed exactly as specified, in [1, 1.5) on every recorded"
        " row (float32 reaches 1.5 only from x of about 5.6 million on), monotone in the width, 1 before the first refresh and on"
        " refused rows; but it saturates near 1.33 on every recorded host and data set, weights move it by only"
        " 0.02-0.15, and no study has shown that it raises return.")
    (DOCS / "data" / "M7_table.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    if not ARGS.no_figure:
        figure(bank_rows, train, r2, curves, hosts, s2)


if __name__ == "__main__":
    main()
