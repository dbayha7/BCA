"""M1 (the residual) recomputed from the near-80 run's recorded inputs, in the project's PINNED environment.

What it does, for the near-80 setting, both worlds, seeds 0-4 (the ten version-6 bands on record):
  1. regenerates the seed's recorded log read-only through near-80's own data code (settingdata.gate_seed, which
     checks the log against the gate's data record) and prepares the near-80 removal's rows; the prepared rows'
     hashes must equal the ones stored in the band record (v6_band.json, data.sha256);
  2. loads the saved frozen critic (critic.msgpack) and the saved band (v6_band.msgpack), each checked against its
     recorded SHA-256;
  3. recomputes the target y, the critic value q and the residual e = y - q at every held-out row with the refresh's
     own noise key (module M1), compares y with the host's own critic loss (algorithms/td3_bc.td3_bc_update) under
     the same key, and then carries the residual downstream: the scores (M3) must equal the stored
     v6_bank_scores.npz byte for byte, and the threshold (M5) must equal the stored R;
  4. writes M1_real_results.json (one row per band) and M1_real_rows.npz (the per-row residuals, scales, throttles
     and currents) into the output folder given as the first argument. M1_test.py reads both.

Nothing is trained: the log regeneration is the simulation of the recorded controller from its seed (the same call
every near-80 phase makes before it reads data), the networks are loaded from disk, and only forward passes run.

Environment (must match the gate's pre-registration or its loader refuses): the pinned venv
/tmp/claude-0/bcavenv/bin/python (Python 3.10.20, numpy 2.2.6, jax 0.6.2 CPU, flax 0.10.7) with
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 and JAX_ENABLE_X64, XLA_FLAGS, JAX_DEFAULT_MATMUL_PRECISION unset, working
directory /home/user/bca. The job file M1_real.sh runs it through the WSL bridge:
    pwsh -NoProfile -File <scratchpad>/wslrun.ps1 -ScriptFile <repo>/docs/bca_mechanics/scripts/M1_real.sh -TimeoutSec 3000
Usage inside WSL:
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python docs/bca_mechanics/scripts/M1_real.py OUT_DIR
        [--seeds 0,1,2,3,4] [--worlds stall,no_stall]
"""
import hashlib
import json
import os
import sys
import time
from dataclasses import replace

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

import settingdata as SD  # noqa: E402  (near-80's data code; read-only here)
import near80_constants as K  # noqa: E402

RUN = ROOT + "/runs/shift_explore/near80"           # the near-80 run files (read-only)
KEY_TARGET, KEY_POSTERIOR = 1213156420, 1347375956   # td3_bc_bca.refresh's fold_in constants (lines 249 and 268)
STAGE2_CODE = ROOT + "/outputs/shift/beyond-the-data/stage2/code"
BAND_PY_SHA256 = "89553406b6cc7bd682a50153e4b74c51f49b5eff022641c614400eb972b232bd"  # near-80 code_manifest.json:137


def main(argv):
    out = argv[1]
    seeds = tuple(int(s) for s in (argv[argv.index("--seeds") + 1] if "--seeds" in argv else "0,1,2,3,4").split(","))
    worlds = tuple((argv[argv.index("--worlds") + 1] if "--worlds" in argv else "stall,no_stall").split(","))
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    learn = SD.gate_prereg().config.learn          # the gate's learner settings (gamma 0.98, lr 3e-4, batch 256, ...)
    # version 6's band code, the same bytes near-80 executed (its code manifest pins this SHA-256)
    with open(STAGE2_CODE + "/band.py", "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != BAND_PY_SHA256:
            sys.exit("stage2/code/band.py is not the version near-80 pinned")
    sys.path.insert(0, STAGE2_CODE)
    import band as B                                   # noqa: E402
    import algorithms.td3_bc as BASE                   # noqa: E402
    import algorithms.td3_bc_bca as P                  # noqa: E402
    from calibration.reference import positive_scale   # noqa: E402
    from experiments.shift import learners as L        # noqa: E402

    print(f"settings loaded in {time.perf_counter() - t0:.0f} s; gamma {learn.gamma}, lr {learn.lr}, batch "
          f"{learn.batch_size}, noise {learn.policy_noise} clipped at {learn.noise_clip}", flush=True)
    rows, arrays = [], {}
    for world in worlds:
        for seed in seeds:
            t1 = time.perf_counter()
            d = f"{RUN}/learners/near80/{world}/seed{seed}"
            with open(d + "/v6_band.json") as f:
                rec = json.load(f)
            with open(d + "/critic.json") as f:
                crec = json.load(f)
            if rec.get("failed"):
                rows.append(dict(world=world, seed=seed, failed=True, reason=rec.get("reason")))
                continue
            # 1. the rows: regenerate the log read-only, prepare the near-80 removal, compare the hashes
            sd = SD.gate_seed(seed, world)
            prep = SD.prepared("near80", sd, learn)
            hashes_equal = {k: prep.sha256[k] == rec["data"]["sha256"][k]
                            for k in ("rows", "train", "bank", "bank_rows", "normalization")}
            bank = prep.bank
            n = int(len(bank.obs))
            # 2. the saved networks, checked against their recorded SHA-256
            critic = L.load_tree(d + "/critic.msgpack", L.critic_template(), crec["file_sha256"])
            critic_sha_equal = L.tree_sha256(critic) == crec["critic_sha256"]
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
            key_y = jax.random.fold_in(refresh_key, KEY_TARGET)
            # 3. M1: y [M1.1-M1.3], q [M1.4], e [M1.5] at every held-out row, with the refresh's own noise key
            y = P.native_target(args, models, native, bank, key_y, 1.0)
            q_heads = L.CRITIC.apply(critic, bank.obs, bank.action)          # (n, 2): Q1 and Q2 at (s, a)
            q = q_heads.min(axis=-1)
            e = np.asarray(y - q, np.float32)
            # the host's own target: td3_bc_update's critic loss under the same key, against the loss built from BCA's y
            _, metrics = BASE.td3_bc_update(args, fixed.apply, L.CRITIC.apply, native, bank, 1, key_y, 1.0)
            host_loss = float(metrics["critic_loss"])
            bca_loss = float(jnp.square(q_heads - y[:, None]).mean(axis=0).sum())
            # a fresh noise draw (same law, another key) and the noise-free target (noise 0)
            y_fresh = P.native_target(args, models, native, bank, jax.random.fold_in(refresh_key, 20261020), 1.0)
            y_quiet = P.native_target(replace(args, policy_noise=0.0), models, native, bank, key_y, 1.0)
            # downstream: M3 scores byte for byte, M5 threshold
            eta = cal.apply(band.cal_params, bank.obs, bank.action)
            scale = np.asarray(positive_scale(eta, band.unit), np.float64)
            scores = np.abs(np.asarray(y - q, np.float64)) / scale
            scores_equal = scores.tobytes() == stored_scores.tobytes()
            eta_equal = np.asarray(eta, np.float64).tobytes() == stored_eta.tobytes()
            sha_equal = hashlib.sha256(np.ascontiguousarray(scores).tobytes()).hexdigest() == rec["bank_scores_sha256"]
            max_rel = float(np.max(np.abs(scores - stored_scores) / np.maximum(stored_scores, 1e-300)))
            thr = B.uniform_threshold_check(scores, refresh_key, config, rec["uniform"]["R"])
            # statistics of the residual, in critic value units and in units of u_f
            u = float(band.unit)
            R = float(rec["uniform"]["R"])
            inner = np.asarray(band.unit * jnp.maximum(eta, 1e-6), np.float64)      # u_f * eta_f: the band at R = 1
            width = R * inner                                                          # U = R * u_f * eta_f (M6)
            abs_e = np.abs(e.astype(np.float64))
            a = np.asarray(bank.action)
            raw_obs = np.asarray(bank.obs) * prep.obs_std + prep.obs_mean
            row = dict(
                world=world, seed=seed, failed=False, n=n,
                rows_hashes_equal=all(hashes_equal.values()), rows_hashes=hashes_equal,
                critic_sha_equal=bool(critic_sha_equal), native_sha_equal=bool(native_sha_equal),
                done_all_zero=bool(np.all(np.asarray(bank.done) == 0)),
                all_finite=bool(np.all(np.isfinite(e)) and np.all(np.isfinite(np.asarray(y)))),
                host_loss=host_loss, bca_loss=bca_loss, loss_rel_diff=abs(host_loss - bca_loss) / abs(host_loss),
                scores_equal=bool(scores_equal), eta_equal=bool(eta_equal), scores_sha_equal=bool(sha_equal),
                scores_max_rel_diff=max_rel,
                R_recorded=R, R_recomputed=float(thr["recomputed"]), R_equal_float32=bool(thr["equal"]),
                unit=u, mean_e=float(e.mean()), sd_e=float(e.std()), median_abs_e=float(np.median(abs_e)),
                q90_abs_e=float(np.quantile(abs_e, 0.9)), max_abs_e=float(abs_e.max()),
                share_e_positive=float(np.mean(e > 0)),
                share_within_unit_eta=float(np.mean(abs_e <= inner)),
                share_within_band=float(np.mean(abs_e <= width)),
                sd_noise_part=float(np.std(np.asarray(y - y_quiet, np.float64))),
                sd_fresh_draw_diff=float(np.std(np.asarray(y - y_fresh, np.float64))),
                sd_e_fresh=float(np.std(np.asarray(y_fresh - q, np.float64))),
                r_min=float(np.min(np.asarray(bank.reward))), r_max=float(np.max(np.asarray(bank.reward))),
                q_min=float(np.min(np.asarray(q))), q_max=float(np.max(np.asarray(q))),
                y_min=float(np.min(np.asarray(y))), y_max=float(np.max(np.asarray(y))),
                seconds=time.perf_counter() - t1,
            )
            rows.append(row)
            tag = f"{world}_seed{seed}"
            arrays[tag + "_e"] = e
            arrays[tag + "_y"] = np.asarray(y, np.float32)
            arrays[tag + "_q"] = np.asarray(q, np.float32)
            arrays[tag + "_eta"] = np.asarray(eta, np.float32)
            arrays[tag + "_throttle"] = np.max(np.abs(a), axis=1).astype(np.float32)
            arrays[tag + "_current"] = raw_obs[:, 10].astype(np.float32)
            print(f"{world:9s} seed {seed}: n {n}, rows hashes {all(hashes_equal.values())}, critic {critic_sha_equal}, "
                  f"native {native_sha_equal}, host loss {host_loss:.6g} vs BCA's y {bca_loss:.6g}, scores byte-equal "
                  f"{scores_equal}, eta byte-equal {eta_equal}, R {R:.6f} recomputed {thr['recomputed']:.6f} "
                  f"equal {thr['equal']}, unit {u:.4f}, SD(e) {row['sd_e']:.4f}, {row['seconds']:.0f} s", flush=True)
    with open(out + "/M1_real_results.json", "w") as f:
        json.dump(dict(written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN,
                       environment=dict(python=sys.version.split()[0], numpy=np.__version__, jax=jax.__version__,
                                        JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS"),
                                        OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS")),
                       settings=dict(gamma=learn.gamma, policy_noise=learn.policy_noise, noise_clip=learn.noise_clip,
                                     controller=K.CONTROLLER),
                       bands=rows), f, indent=1)
    np.savez(out + "/M1_real_rows.npz", **arrays)
    print(f"wrote {out}/M1_real_results.json and M1_real_rows.npz in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv)
