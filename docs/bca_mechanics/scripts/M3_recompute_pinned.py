"""Module M3 (the scores) recomputed from recorded inputs in the project's PINNED environment.

Run inside WSL (Python 3.10.20, numpy 2.2.6, jax 0.6.2 on CPU), from the owner's clone /home/user/bca, read-only:
    cd /home/user/bca && JAX_PLATFORMS=cpu /tmp/claude-0/bcavenv/bin/python -u /tmp/bridge/work/m3/M3_recompute_pinned.py
(the sh job file M3_recompute_pinned.sh does exactly this). Nothing is trained: the critics and the error-size
networks are read from the saved files, the held-out rows are rebuilt by the gate's data loader, and only forward
passes and calibration/wbcp.calibrate run.

For every recorded band it does four things and writes /tmp/bridge/work/m3/M3_pinned_recompute.json:
  (a) recompute the held-out scores through the study's own function (version 6's band.bank_scores, the one the
      run used) from the saved critic, the saved error-size network and unit, and the refresh key rebuilt from its
      recorded rule; compare with the stored score array bit for bit;
  (b) recompute the same scores by hand from the pieces y, q, eta and u with freeze_reference's arithmetic
      (scale = max(eta, 1e-6) * u in float32, then float64; |y - q| from the float32 residual), and show how a
      float64-throughout version would differ;
  (c) run calibration/reference.freeze_reference itself (M3 + M5) on the recomputed pieces with the refresh's
      posterior key; compare threshold, lambda_hat, lambda_hpd and n_eff with the record (float32 bit for bit) and
      the float64 diagnostics with the record's refresh block;
  (d) run wbcp.calibrate on the STORED score array with the same key (the record's own threshold_recomputed check,
      redone) and on the recomputed scores.
Bands: near-80 version-6 bands (setting near80, both worlds, seeds 0-4: full recompute); stage 2's uniform bands on the
fixed critic (both worlds, seeds 0-4: full recompute, the gate's gap critic) and on the host critic (stalling world,
seeds 0-4: (d) only, that critic needs check 3's host snapshot, not loaded here).
"""
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path("/home/user/bca")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "outputs/shift/beyond-the-data/near80/code"))

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import near80_constants as K  # noqa: E402
import settingdata as SD  # noqa: E402
import learner_phases as LP  # noqa: E402
import policy_phase as PP  # noqa: E402
import algorithms.td3_bc_bca as TB  # noqa: E402
from experiments.shift import learners as L  # noqa: E402
from calibration import wbcp  # noqa: E402
from calibration import reference as REF  # noqa: E402

OUT = Path("/tmp/bridge/work/m3")
OUT.mkdir(parents=True, exist_ok=True)
NEAR80 = ROOT / "runs/shift_explore/near80"
STAGE2 = ROOT / "runs/shift_explore/stage2/detect/uniform"
KEY_TARGET, KEY_POSTERIOR = 1213156420, 1347375956   # td3_bc_bca.refresh's two fold-in constants
ALPHA, BETA, DRAWS = 0.1, 0.95, 1000
PREPARED_KEYS = ("rows", "train", "bank", "bank_rows", "normalization")


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def f32eq(a, b):
    return bool(np.float32(a).tobytes() == np.float32(b).tobytes())


def words(key):
    return np.asarray(key, np.uint32).ravel()


t_all = time.perf_counter()
S, MC, facts = PP.load_step2()
B = MC.B          # version 6's band.py, verified bytes
learn = SD.gate_prereg().config.learn
config = B.bca_config()
print("versions", sys.version.split()[0], "numpy", np.__version__, "jax", jax.__version__, flush=True)
print("step-2 machinery loaded", facts, f"{time.perf_counter() - t_all:.1f}s", flush=True)

_seeds = {}


def gate_seed(seed, world):
    if (seed, world) not in _seeds:
        t = time.perf_counter()
        _seeds[(seed, world)] = SD.gate_seed(seed, world)
        print(f"log regenerated: seed {seed} {world} ({time.perf_counter() - t:.1f}s)", flush=True)
    return _seeds[(seed, world)]


def refresh_key_of(seed, n_rows):
    """band_rows_and_keys' refresh key: fold_in(PRNGKey(default_rng([20261015, seed, 41, 1]).integers(2**31 - 1)), 20000)."""
    rng = jax.random.PRNGKey(0)   # not used for the refresh key
    _, _, key = B.band_rows_and_keys(rng, n_rows, B.BAND_FIT_STEPS, learn.batch_size, seed)
    return key


def from_stored(stored_scores, refresh_key, rec_uniform, rec_refresh):
    """(d) M5 on the stored score array with the refresh's own posterior stream."""
    post = jax.random.fold_in(refresh_key, KEY_POSTERIOR)
    c = wbcp.calibrate(np.asarray(stored_scores, np.float64), np.random.default_rng(words(post)),
                       alpha=ALPHA, beta=BETA, draws=DRAWS)
    return dict(threshold_f64=c.threshold, lambda_hat_f64=c.lambda_hat, lambda_hpd_f64=c.lambda_hpd,
                n_eff=c.n_eff, sigma_post=c.sigma_post,
                equal_f64_threshold=bool(c.threshold == rec_refresh["threshold"]),
                equal_f64_lambda_hat=bool(c.lambda_hat == rec_refresh["lambda_hat"]),
                equal_f64_lambda_hpd=bool(c.lambda_hpd == rec_refresh["lambda_hpd"]),
                equal_f64_sigma_post=bool(c.sigma_post == rec_refresh["sigma_post"]),
                equal_f32_R=f32eq(c.threshold, rec_uniform["R"]),
                equal_f32_lambda_hat=f32eq(c.lambda_hat, rec_uniform["lambda_hat"]),
                equal_f32_lambda_hpd=f32eq(c.lambda_hpd, rec_uniform["lambda_hpd"]),
                equal_f32_n_eff=f32eq(c.n_eff, rec_refresh["n_eff"]))


def full_recompute(label, prep, critic, seed, cal_params, unit, stored_scores, stored_eta, rec_uniform, rec_refresh,
                   band_R):
    args = MC.L.make_args(learn, seed)
    fixed = MC.L.FixedController(K.CONTROLLER, prep.physics, prep.obs_mean, prep.obs_std)
    native = B.fixed_native(critic, fixed, learn.lr)
    rng, state, cal = B.initial_state(args, config, native, MC.L.DX, MC.L.DA)
    models = (fixed, MC.L.CRITIC, cal)
    rows, keys, refresh_key = B.band_rows_and_keys(rng, prep.n, B.BAND_FIT_STEPS, learn.batch_size, seed)
    bank = prep.bank
    # (a) the study's own path
    scores_B, eta_B = B.bank_scores(args, models, state, bank, refresh_key, cal_params, unit)
    # (b) the pieces, float32 as td3_bc_bca.refresh computes them
    y = TB.native_target(args, models, state.native, bank, jax.random.fold_in(refresh_key, KEY_TARGET), 1.0)
    q = models[1].apply(state.native.critic.params, bank.obs, bank.action).min(axis=-1)
    eta = cal.apply(cal_params, bank.obs, bank.action)
    residual = y - q                                    # float32, what refresh passes to freeze_reference
    y32, q32, eta32, r32 = (np.asarray(v, np.float32) for v in (y, q, eta, residual))
    unit32 = np.float32(np.asarray(unit, np.float32))
    scale32 = np.maximum(eta32, np.float32(1e-6)) * unit32          # positive_scale, float32
    scores_hand = np.abs(r32.astype(np.float64)) / scale32.astype(np.float64)
    scale64 = np.maximum(eta32.astype(np.float64), 1e-6) * float(unit32)
    scores_alt_scale64 = np.abs(r32.astype(np.float64)) / scale64
    scores_alt_all64 = np.abs(y32.astype(np.float64) - q32.astype(np.float64)) / scale64
    # (c) the exact code path, M3 + M5
    post = jax.random.fold_in(refresh_key, KEY_POSTERIOR)
    ref, valid, diag = REF.freeze_reference(cal_params, jnp.asarray(unit), eta, residual, post, config.posterior)
    # (d) from the stored scores and from the recomputed scores
    d_stored = from_stored(stored_scores, refresh_key, rec_uniform, rec_refresh)
    c_recomp = wbcp.calibrate(scores_B, np.random.default_rng(words(post)), alpha=ALPHA, beta=BETA, draws=DRAWS)
    stored_scores = np.asarray(stored_scores, np.float64)
    out = dict(
        label=label, n=int(scores_B.size), full_recompute=True,
        refresh_key=words(refresh_key).tolist(), posterior_words=words(post).tolist(),
        target_words=words(jax.random.fold_in(refresh_key, KEY_TARGET)).tolist(),
        a_scores_equal_stored=bool(np.array_equal(scores_B, stored_scores)),
        a_scores_max_abs_diff=float(np.max(np.abs(scores_B - stored_scores))),
        a_eta_equal_stored=bool(np.array_equal(np.asarray(eta_B, np.float64), np.asarray(stored_eta, np.float64))),
        a_sha256_recomputed=sha(scores_B), a_sha256_stored=sha(stored_scores),
        b_hand_equals_study_path=bool(np.array_equal(scores_hand, scores_B)),
        b_all_finite=bool(np.all(np.isfinite(scores_B))), b_all_scale_positive=bool(np.all(scale32 > 0)),
        b_alt_scale64_rows_differ=int(np.sum(scores_alt_scale64 != scores_B)),
        b_alt_scale64_max_rel=float(np.max(np.abs(scores_alt_scale64 - scores_B) / scores_B)),
        b_alt_all64_rows_differ=int(np.sum(scores_alt_all64 != scores_B)),
        b_alt_all64_max_rel=float(np.max(np.abs(scores_alt_all64 - scores_B) / scores_B)),
        c_freeze_reference=dict(
            valid=bool(valid), ready=bool(ref.ready), certified=bool(diag["certified"]),
            R_f32=float(ref.threshold), lambda_hat_f32=float(ref.lambda_hat), lambda_hpd_f32=float(ref.lambda_hpd),
            n_eff_f32=float(ref.n_eff), threshold_f64=float(diag["threshold"]),
            lambda_hat_f64=float(diag["lambda_hat"]), lambda_hpd_f64=float(diag["lambda_hpd"]),
            sigma_post=float(diag["sigma_post"]), n_eff=float(diag["n_eff"]), scores=int(diag["scores"])),
        d_from_stored_scores=d_stored,
        d_recomputed_scores_threshold_f64=c_recomp.threshold,
        record=dict(R=rec_uniform["R"], lambda_hat=rec_uniform["lambda_hat"], lambda_hpd=rec_uniform["lambda_hpd"],
                    unit=rec_uniform["unit"], band_R=float(band_R), refresh=dict(rec_refresh)),
    )
    fr = out["c_freeze_reference"]
    out["c_bit_for_bit"] = dict(
        R=bool(f32eq(fr["R_f32"], rec_uniform["R"]) and f32eq(fr["R_f32"], band_R)),
        lambda_hat=f32eq(fr["lambda_hat_f32"], rec_uniform["lambda_hat"]),
        lambda_hpd=f32eq(fr["lambda_hpd_f32"], rec_uniform["lambda_hpd"]),
        n_eff=f32eq(fr["n_eff_f32"], rec_refresh["n_eff"]),
        f64_threshold=bool(fr["threshold_f64"] == rec_refresh["threshold"]),
        f64_lambda_hat=bool(fr["lambda_hat_f64"] == rec_refresh["lambda_hat"]),
        f64_lambda_hpd=bool(fr["lambda_hpd_f64"] == rec_refresh["lambda_hpd"]),
        f64_sigma_post=bool(fr["sigma_post"] == rec_refresh["sigma_post"]),
        recomputed_scores_threshold_f64=bool(c_recomp.threshold == rec_refresh["threshold"]),
    )
    out["summary"] = dict(
        share_le_1=float(np.mean(scores_B <= 1.0)), share_le_R=float(np.mean(scores_B <= rec_uniform["R"])),
        share_le_lambda_hat=float(np.mean(scores_B <= rec_uniform["lambda_hat"])),
        q90=float(np.quantile(scores_B, 0.9)), median_abs_residual=float(np.median(np.abs(r32))),
        median_scale=float(np.median(scale32)), unit=float(unit32), eta_min=float(eta32.min()),
        eta_max=float(eta32.max()), score_min=float(scores_B.min()), score_max=float(scores_B.max()))
    np.savez(OUT / f"{label}.npz", y=y32, q=q32, eta=eta32, unit=unit32, residual=r32, scale=scale32,
             scores=scores_B, scores_stored=stored_scores, refresh_key=words(refresh_key), posterior_words=words(post))
    return out


results, errors = [], []


def run(label, fn):
    t = time.perf_counter()
    try:
        r = fn()
        r["seconds"] = time.perf_counter() - t
        results.append(r)
        flags = {k: v for k, v in r.items() if k.startswith("a_") and isinstance(v, bool)}
        print(label, "ok", flags, r.get("c_bit_for_bit"),
              "stored->R equal:", r.get("d_from_stored_scores", {}).get("equal_f64_threshold"),
              f"{r['seconds']:.1f}s", flush=True)
    except Exception as error:  # noqa: BLE001
        errors.append(dict(label=label, error=repr(error), trace=traceback.format_exc()))
        print(label, "ERROR", repr(error), flush=True)
        traceback.print_exc()


def near80_band(world, seed):
    def go():
        sd = gate_seed(seed, world)
        band, rec = LP.load_v6_band(NEAR80, "near80", world, seed)
        d = SD.learner_dir(NEAR80, "near80", world, seed)
        crec = SD.load_record(d / "critic.json")
        csha = crec.get("file_sha256") or crec["result"]["file_sha256"]
        critic = SD.load_tree_checked(d / "critic.msgpack", L.critic_template(), csha)
        prep = SD.prepared("near80", sd, learn)
        data_equal = {k: bool(prep.sha256[k] == rec["data"]["sha256"][k]) for k in PREPARED_KEYS}
        z = np.load(d / "v6_bank_scores.npz")
        out = full_recompute(f"near80_{world}_seed{seed}", prep, critic, seed, band.cal_params, band.unit,
                             z["scores"], z["eta"], rec["uniform"], rec["refresh"], band.R)
        out.update(study="near-80 version-6 band", world=world, critic="fixed", seed=seed,
                   prepared_rows_equal_record=data_equal, stored_sha256_in_record=rec["bank_scores_sha256"],
                   record_threshold_recomputed=rec["uniform"]["threshold_recomputed"])
        return out
    return go


def stage2_band(world, critic_kind, seed):
    def go():
        d = STAGE2 / world / critic_kind / f"seed{seed}"
        rec = json.loads((d / "band.json").read_text())
        z = np.load(d / "bank_scores.npz")
        band = L.load_tree(d / "band.msgpack", B.band_template(MC.L.DX, MC.L.DA, False), rec["band_file_sha256"])
        label = f"stage2_{world}_{critic_kind}_seed{seed}"
        if critic_kind == "fixed":
            sd = gate_seed(seed, world)
            critic = SD.gate_critic(sd, "gap")
            prep = SD.prepared("main", sd, learn)
            data_equal = {k: bool(prep.sha256[k] == rec["data"]["sha256"][k]) for k in PREPARED_KEYS}
            out = full_recompute(label, prep, critic, seed, band.cal_params, band.unit, z["scores"], z["eta"],
                                 rec["uniform"], rec["refresh"], band.R)
            out.update(prepared_rows_equal_record=data_equal)
        else:
            key = refresh_key_of(seed, int(rec["data"]["rows"]))
            sc = np.asarray(z["scores"], np.float64)
            out = dict(label=label, n=int(sc.size), full_recompute=False,
                       refresh_key=words(key).tolist(),
                       posterior_words=words(jax.random.fold_in(key, KEY_POSTERIOR)).tolist(),
                       d_from_stored_scores=from_stored(sc, key, rec["uniform"], rec["refresh"]),
                       a_sha256_stored=sha(sc),
                       record=dict(R=rec["uniform"]["R"], lambda_hat=rec["uniform"]["lambda_hat"],
                                   lambda_hpd=rec["uniform"]["lambda_hpd"], unit=rec["uniform"]["unit"],
                                   band_R=float(band.R), refresh=dict(rec["refresh"])),
                       summary=dict(share_le_1=float(np.mean(sc <= 1.0)),
                                    share_le_R=float(np.mean(sc <= rec["uniform"]["R"])),
                                    share_le_lambda_hat=float(np.mean(sc <= rec["uniform"]["lambda_hat"])),
                                    q90=float(np.quantile(sc, 0.9)), unit=float(rec["uniform"]["unit"]),
                                    score_min=float(sc.min()), score_max=float(sc.max())))
        out.update(study="stage 2 uniform band", world=world, critic=critic_kind, seed=seed,
                   stored_sha256_in_record=rec["bank_scores_sha256"],
                   record_threshold_recomputed=rec["uniform"]["threshold_recomputed"])
        return out
    return go


for world in ("stall", "no_stall"):
    for seed in range(5):
        run(f"near80_{world}_seed{seed}", near80_band(world, seed))
for world in ("stall", "no_stall"):
    for seed in range(5):
        run(f"stage2_{world}_fixed_seed{seed}", stage2_band(world, "fixed", seed))
for seed in range(5):
    run(f"stage2_stall_host_seed{seed}", stage2_band("stall", "host", seed))

report = dict(
    environment=dict(python=sys.version.split()[0], numpy=np.__version__, jax=jax.__version__,
                     platform=str(jax.devices()[0].platform), repo_commit=None, step2_facts=facts),
    settings=dict(alpha=ALPHA, beta=BETA, draws=DRAWS, key_target=KEY_TARGET, key_posterior=KEY_POSTERIOR),
    bands=results, errors=errors, seconds=time.perf_counter() - t_all)
try:
    import subprocess
    report["environment"]["repo_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                          capture_output=True, text=True).stdout.strip()
except Exception:  # noqa: BLE001
    pass
(OUT / "M3_pinned_recompute.json").write_text(json.dumps(report, indent=1, default=float))
print("DONE", len(results), "bands,", len(errors), "errors,", f"{report['seconds']:.0f}s", flush=True)
