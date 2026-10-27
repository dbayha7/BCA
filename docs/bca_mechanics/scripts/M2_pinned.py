"""M2 (the error-size network eta and the unit u) checked on the near-80 run's recorded inputs, in the project's PINNED
environment. Companion of M2_test.py, which reads this script's report (docs/bca_mechanics/data/M2_pinned_recompute.json).

What it does, for the two frozen-critic settings of the near-80 run (near80 and random_rows), both worlds, seeds 0-4,
the twenty version-6 bands on record:
  1. regenerates the seed's recorded log read-only through near-80's own data code (settingdata.gate_seed, which checks
     the log against the gate's data record), prepares the setting's removal (settingdata.prepared) and checks the
     prepared rows' hashes against the band record (v6_band.json, data.sha256);
  2. loads the saved frozen critic (critic.msgpack) and the saved band (v6_band.msgpack: the frozen error-size network's
     parameters, the unit u_f and the threshold R), each checked against its recorded SHA-256;
  3. M2's outputs: eta at every held-out calibration row (must equal the stored v6_bank_scores.npz 'eta' byte for byte)
     and at every kept TRAINING row (the rows the 20,000 fit steps drew their batches from): finite, above the 0.01 floor;
  4. M2's purpose: the residual e = y - q (module M1, the refresh's own noise key on the held-out rows; a definite fresh
     key on the training rows, where the fit itself used a different key at every step) and the shares of rows with
     |e| <= u_f * eta (the band at threshold 1, what the fit aims at: 0.9) and |e| <= R * u_f * eta (the band of module
     M6), on the training rows and on the held-out rows; the unit against SD(e) on each set; the 90% point of the score
     |e| / (u_f * eta) inside each tenth of the rows by eta (1.0 everywhere if eta were calibrated row by row) and
     inside each tenth by the water current; Spearman's rank correlation of eta with |e|;
  5. writes M2_pinned_recompute.json (one record per band) and M2_pinned_rows.npz (per-row eta, residual, current and
     largest throttle, training and held-out) into the output folder given as the first argument.

Nothing is trained and no band is fitted: the log regeneration is the simulation of the recorded controller from its
seed (the same call every near-80 phase makes before it reads data); the networks are loaded from disk; only forward
passes run.

Environment (must match the gate's pre-registration or its loader refuses): the pinned venv
/tmp/claude-0/bcavenv/bin/python (Python 3.10.20, numpy 2.2.6, jax 0.6.2 CPU, flax 0.10.7) with JAX_PLATFORMS=cpu
OMP_NUM_THREADS=1 and JAX_ENABLE_X64, XLA_FLAGS, JAX_DEFAULT_MATMUL_PRECISION unset, working directory /home/user/bca
(the owner's clone, read-only). The job file M2_pinned_job.sh runs it through the WSL bridge:
    pwsh -NoProfile -File <scratchpad>/wslrun.ps1 -ScriptFile <repo>/docs/bca_mechanics/scripts/M2_pinned_job.sh -TimeoutSec 1800
Usage inside WSL:
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python docs/bca_mechanics/scripts/M2_pinned.py OUT_DIR
        [--settings near80,random_rows] [--seeds 0,1,2,3,4] [--worlds stall,no_stall]
"""
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
ROOT = "/home/user/bca"
for p in (ROOT, ROOT + "/outputs/shift/beyond-the-data/near80/code"):
    if p not in sys.path:
        sys.path.insert(0, p)

for flag, want in (("JAX_PLATFORMS", "cpu"), ("OMP_NUM_THREADS", "1")):
    if os.environ.get(flag) != want:
        sys.exit(f"set {flag}={want} (the gate's pre-registered numeric flags)")
for flag in ("JAX_ENABLE_X64", "XLA_FLAGS", "JAX_DEFAULT_MATMUL_PRECISION"):
    if os.environ.get(flag) is not None:
        sys.exit(f"unset {flag} (the gate's pre-registered numeric flags)")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from scipy import stats  # noqa: E402

import settingdata as SD  # noqa: E402  (near-80's data code; read-only here)
import near80_constants as K  # noqa: E402

RUN = ROOT + "/runs/shift_explore/near80"            # the near-80 run files (read-only)
KEY_REFRESH_TARGET = 1213156420                       # td3_bc_bca.refresh's target-noise fold (line 249)
KEY_FIT_TARGET = 1179210836                           # td3_bc_bca.fit_scale's target-noise fold (line 167)
ETA_FLOOR = 0.01                                      # calibration/network.py:30
STAGE2_CODE = ROOT + "/outputs/shift/beyond-the-data/stage2/code"
BAND_PY_SHA256 = "89553406b6cc7bd682a50153e4b74c51f49b5eff022641c614400eb972b232bd"  # near-80 code_manifest.json:137
CURRENT_INDEX = 10                                    # the water current is state number 11 (index 10)


def deciles_by(key_values, n=10):
    order = np.argsort(np.asarray(key_values, np.float64), kind="stable")
    return np.array_split(order, n)


def q90(x):
    return float(np.quantile(np.asarray(x, np.float64), 0.9))


def set_stats(eta, e, unit, R, current, throttle):
    """The checkable statements of M2 on one set of rows."""
    eta = np.asarray(eta, np.float64)
    ab = np.abs(np.asarray(e, np.float64))
    inner = unit * np.maximum(eta, 1e-6)              # u_f * eta_f: the band at threshold 1 (positive_scale)
    score = ab / inner
    out = dict(
        rows=int(len(eta)),
        eta_finite=bool(np.all(np.isfinite(eta))), eta_above_floor=bool(np.all(eta > ETA_FLOOR)),
        eta_min=float(eta.min()), eta_median=float(np.median(eta)), eta_mean=float(eta.mean()), eta_max=float(eta.max()),
        eta_sd=float(eta.std()),
        share_within_unit_eta=float(np.mean(ab <= inner)),         # the fit's target: 0.9
        share_within_band=float(np.mean(ab <= R * inner)),         # M6's band
        score_q90=q90(score),                                       # R would be about this if the set were the bank
        sd_e=float(np.std(np.asarray(e, np.float64))), mean_e=float(np.mean(np.asarray(e, np.float64))),
        median_abs_e=float(np.median(ab)), q90_abs_e=q90(ab),
        spearman_eta_abs_e=float(stats.spearmanr(eta, ab)[0]),
        spearman_eta_current=float(stats.spearmanr(eta, current)[0]),
        spearman_abs_e_current=float(stats.spearmanr(ab, current)[0]),
    )
    # inside each tenth of the rows by eta: if eta were calibrated row by row, the 90% point of the score would be 1
    parts = deciles_by(eta)
    out["by_eta_decile"] = dict(
        eta_mean=[float(eta[p].mean()) for p in parts],
        abs_e_mean=[float(ab[p].mean()) for p in parts],
        abs_e_q90=[q90(ab[p]) for p in parts],
        score_q90=[q90(score[p]) for p in parts],
        share_within_unit_eta=[float(np.mean(ab[p] <= inner[p])) for p in parts],
        share_within_band=[float(np.mean(ab[p] <= R * inner[p])) for p in parts],
    )
    parts = deciles_by(current)
    out["by_current_decile"] = dict(
        current_mean=[float(np.asarray(current)[p].mean()) for p in parts],
        eta_mean=[float(eta[p].mean()) for p in parts],
        abs_e_mean=[float(ab[p].mean()) for p in parts],
        abs_e_q90=[q90(ab[p]) for p in parts],
        score_q90=[q90(score[p]) for p in parts],
        share_within_unit_eta=[float(np.mean(ab[p] <= inner[p])) for p in parts],
        share_within_band=[float(np.mean(ab[p] <= R * inner[p])) for p in parts],
    )
    full = np.asarray(throttle) >= 0.999                          # a full-throttle command (some thruster at +-1)
    out["full_throttle"] = dict(rows=int(full.sum()),
                                eta_mean=float(eta[full].mean()) if full.any() else None,
                                abs_e_mean=float(ab[full].mean()) if full.any() else None,
                                share_within_unit_eta=float(np.mean(ab[full] <= inner[full])) if full.any() else None,
                                share_within_band=float(np.mean(ab[full] <= R * inner[full])) if full.any() else None)
    return out


def main(argv):
    out = argv[1]
    opt = lambda name, default: argv[argv.index(name) + 1] if name in argv else default  # noqa: E731
    settings = tuple(opt("--settings", "near80,random_rows").split(","))
    seeds = tuple(int(s) for s in opt("--seeds", "0,1,2,3,4").split(","))
    worlds = tuple(opt("--worlds", "stall,no_stall").split(","))
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    learn = SD.gate_prereg().config.learn          # the gate's learner settings (gamma 0.98, lr 3e-4, batch 256, ...)
    with open(STAGE2_CODE + "/band.py", "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != BAND_PY_SHA256:
            sys.exit("stage2/code/band.py is not the version near-80 pinned")
    sys.path.insert(0, STAGE2_CODE)
    import band as B                                   # noqa: E402
    import algorithms.td3_bc_bca as P                  # noqa: E402
    from experiments.shift import learners as L        # noqa: E402

    code_sha = {rel: hashlib.sha256(open(ROOT + "/" + rel, "rb").read()).hexdigest() for rel in (
        "calibration/network.py", "calibration/reference.py", "algorithms/td3_bc_bca.py", "algorithms/td3_bc.py",
        "experiments/shift/learners.py", "outputs/shift/beyond-the-data/stage2/code/band.py")}
    print(f"settings loaded in {time.perf_counter() - t0:.0f} s; gamma {learn.gamma}, batch {learn.batch_size}, "
          f"band fit steps {learn.band_fit_steps}, noise {learn.policy_noise} clipped at {learn.noise_clip}", flush=True)
    crit_apply = jax.jit(lambda prm, o, a: L.CRITIC.apply(prm, o, a).min(axis=-1))
    records, arrays = [], {}
    for world in worlds:
        for seed in seeds:
            sd = None
            for setting in settings:
                t1 = time.perf_counter()
                d = f"{RUN}/learners/{setting}/{world}/seed{seed}"
                with open(d + "/v6_band.json") as f:
                    rec = json.load(f)
                with open(d + "/critic.json") as f:
                    crec = json.load(f)
                if rec.get("failed"):
                    records.append(dict(setting=setting, world=world, seed=seed, failed=True, reason=rec.get("reason")))
                    continue
                # 1. the rows: regenerate the log read-only (once per seed and world), prepare the setting's removal
                if sd is None:
                    sd = SD.gate_seed(seed, world)
                prep = SD.prepared(setting, sd, learn)
                hashes_equal = {k: prep.sha256[k] == rec["data"]["sha256"][k]
                                for k in ("rows", "train", "bank", "bank_rows", "normalization")}
                # 2. the saved networks, checked against their recorded SHA-256
                critic = L.load_tree(d + "/critic.msgpack", L.critic_template(), crec["file_sha256"])
                band = L.load_tree(d + "/v6_band.msgpack", B.band_template(L.DX, L.DA, False), rec["band_file_sha256"])
                with np.load(d + "/v6_bank_scores.npz") as z:
                    stored_scores, stored_eta = np.ascontiguousarray(z["scores"]), np.ascontiguousarray(z["eta"])
                args = L.make_args(learn, seed)
                config = B.bca_config()
                fixed = L.FixedController(K.CONTROLLER, prep.physics, prep.obs_mean, prep.obs_std)
                native = B.fixed_native(critic, fixed, learn.lr)       # the frozen critic as the host's state
                native_sha_equal = B._tree_sha(native) == rec["native_sha256"]
                rng, _, cal = B.initial_state(args, config, native, L.DX, L.DA)
                models = (fixed, L.CRITIC, cal)
                _, _, refresh_key = B.band_rows_and_keys(rng, prep.n, int(rec["steps"]), learn.batch_size, seed)
                unit = float(band.unit)
                R = float(rec["uniform"]["R"])
                unit_equal = bool(np.float32(unit) == np.float32(rec["uniform"]["unit"]))
                # 3. M2's outputs: eta at the held-out rows (byte for byte) and at the training rows
                eta_bank = np.asarray(cal.apply(band.cal_params, prep.bank.obs, prep.bank.action), np.float64)
                eta_equal = eta_bank.tobytes() == stored_eta.tobytes()
                eta_train = np.asarray(cal.apply(band.cal_params, prep.train.obs, prep.train.action), np.float64)
                # 4. the residual (M1) on both sets: the refresh's own key on the held-out rows; a definite fresh key
                #    (the fit's fold applied to the refresh key) on the training rows, and a second draw for stability
                y_bank = P.native_target(args, models, native, prep.bank,
                                         jax.random.fold_in(refresh_key, KEY_REFRESH_TARGET), 1.0)
                q_bank = crit_apply(critic, prep.bank.obs, prep.bank.action)
                e_bank = np.asarray(y_bank - q_bank, np.float64)
                scores = np.abs(e_bank) / np.asarray(unit * jnp.maximum(jnp.asarray(eta_bank, jnp.float32), 1e-6), np.float64)
                scores_equal = scores.tobytes() == stored_scores.tobytes()
                key_train = jax.random.fold_in(refresh_key, KEY_FIT_TARGET)
                y_train = P.native_target(args, models, native, prep.train, key_train, 1.0)
                q_train = crit_apply(critic, prep.train.obs, prep.train.action)
                e_train = np.asarray(y_train - q_train, np.float64)
                y_train2 = P.native_target(args, models, native, prep.train, jax.random.fold_in(key_train, 2), 1.0)
                e_train2 = np.asarray(y_train2 - q_train, np.float64)
                cur_bank = np.asarray(prep.bank.obs[:, CURRENT_INDEX]) * prep.obs_std[CURRENT_INDEX] + prep.obs_mean[CURRENT_INDEX]
                cur_train = np.asarray(prep.train.obs[:, CURRENT_INDEX]) * prep.obs_std[CURRENT_INDEX] + prep.obs_mean[CURRENT_INDEX]
                thr_bank = np.max(np.abs(np.asarray(prep.bank.action)), axis=1)
                thr_train = np.max(np.abs(np.asarray(prep.train.action)), axis=1)
                train_stats = set_stats(eta_train, e_train, unit, R, cur_train, thr_train)
                train_stats["share_within_unit_eta_second_draw"] = float(
                    np.mean(np.abs(e_train2) <= unit * np.maximum(eta_train, 1e-6)))
                train_stats["sd_e_second_draw"] = float(np.std(e_train2))
                bank_stats = set_stats(eta_bank, e_bank, unit, R, cur_bank, thr_bank)
                # the caveat, measured: the same band with eta(s, a) replaced by one number, its mean over the training rows
                eta_bar = float(eta_train.mean())
                constant_eta = dict(
                    eta_mean_train=eta_bar,
                    train_share_within_unit_eta=float(np.mean(np.abs(e_train) <= unit * eta_bar)),
                    bank_share_within_unit_eta=float(np.mean(np.abs(e_bank) <= unit * eta_bar)),
                    train_share_within_band=float(np.mean(np.abs(e_train) <= R * unit * eta_bar)),
                    bank_share_within_band=float(np.mean(np.abs(e_bank) <= R * unit * eta_bar)))
                row = dict(
                    setting=setting, world=world, seed=seed, failed=False,
                    n_train=int(prep.n), n_bank=int(len(prep.bank.obs)),
                    rows_hashes_equal=all(hashes_equal.values()), rows_hashes=hashes_equal,
                    native_sha_equal=bool(native_sha_equal), unit_equals_record=unit_equal,
                    accepted_share_recorded=rec["accepted_share"], steps_recorded=int(rec["steps"]),
                    eta_bank_byte_equal=bool(eta_equal), scores_byte_equal=bool(scores_equal),
                    unit=unit, R=R, lambda_hat=float(rec["uniform"]["lambda_hat"]),
                    unit_over_sd_e_train=unit / train_stats["sd_e"], unit_over_sd_e_bank=unit / bank_stats["sd_e"],
                    q90_abs_e_ratio_bank_over_train=bank_stats["q90_abs_e"] / train_stats["q90_abs_e"],
                    constant_eta=constant_eta,
                    train=train_stats, bank=bank_stats, seconds=time.perf_counter() - t1,
                )
                records.append(row)
                tag = f"{setting}_{world}_seed{seed}"
                arrays[tag + "_eta_train"] = eta_train.astype(np.float32)
                arrays[tag + "_e_train"] = e_train.astype(np.float32)
                arrays[tag + "_current_train"] = cur_train.astype(np.float32)
                arrays[tag + "_throttle_train"] = thr_train.astype(np.float32)
                arrays[tag + "_eta_bank"] = eta_bank.astype(np.float32)
                arrays[tag + "_e_bank"] = e_bank.astype(np.float32)
                arrays[tag + "_current_bank"] = cur_bank.astype(np.float32)
                arrays[tag + "_throttle_bank"] = thr_bank.astype(np.float32)
                print(f"{setting:11s} {world:9s} seed {seed}: train {prep.n} bank {row['n_bank']}; hashes "
                      f"{row['rows_hashes_equal']}, native {native_sha_equal}, eta byte-equal {eta_equal}, scores byte-equal "
                      f"{scores_equal}; eta train [{train_stats['eta_min']:.3f}, {train_stats['eta_max']:.3f}] bank "
                      f"[{bank_stats['eta_min']:.3f}, {bank_stats['eta_max']:.3f}]; unit {unit:.4f} SD(e) train "
                      f"{train_stats['sd_e']:.4f} bank {bank_stats['sd_e']:.4f}; share <= u*eta train "
                      f"{train_stats['share_within_unit_eta']:.3f} (2nd draw {train_stats['share_within_unit_eta_second_draw']:.3f}) "
                      f"bank {bank_stats['share_within_unit_eta']:.3f}; <= R*u*eta train {train_stats['share_within_band']:.3f} "
                      f"bank {bank_stats['share_within_band']:.3f}; R {R:.4f}, score q90 train {train_stats['score_q90']:.3f} "
                      f"bank {bank_stats['score_q90']:.3f}; Spearman(eta,|e|) train {train_stats['spearman_eta_abs_e']:.3f} "
                      f"bank {bank_stats['spearman_eta_abs_e']:.3f}; with eta replaced by its mean {eta_bar:.3f}: share <= u*eta "
                      f"train {constant_eta['train_share_within_unit_eta']:.3f} bank {constant_eta['bank_share_within_unit_eta']:.3f}; "
                      f"q90|e| bank/train {row['q90_abs_e_ratio_bank_over_train']:.2f}; {row['seconds']:.0f} s", flush=True)
    with open(out + "/M2_pinned_recompute.json", "w") as f:
        json.dump(dict(written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, code_sha256=code_sha,
                       environment=dict(python=sys.version.split()[0], numpy=np.__version__, jax=jax.__version__,
                                        JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS"),
                                        OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS")),
                       settings=dict(gamma=learn.gamma, policy_noise=learn.policy_noise, noise_clip=learn.noise_clip,
                                     controller=K.CONTROLLER, batch_size=learn.batch_size,
                                     band_fit_steps=learn.band_fit_steps, eta_floor=ETA_FLOOR,
                                     training_noise_key="fold_in(fold_in(refresh_key, 1179210836), [none|2])"),
                       bands=records), f, indent=1)
    np.savez(out + "/M2_pinned_rows.npz", **arrays)
    print(f"wrote {out}/M2_pinned_recompute.json and M2_pinned_rows.npz in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv)
