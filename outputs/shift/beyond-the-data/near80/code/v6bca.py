"""Near-80 (version 8): version 6's BCA band fit and width, restated on the pure functions of version 6's band.py.

Version 6's band.fit_on_context fits BCA's band on a frozen critic, but it first calls version 6's approval check,
which binds to version 6's pre-registration. Near-80 reports version 6's BCA as an arm (choice 8), so fit() restates
fit_on_context's orchestration for critic_kind 'fixed' (the setting's frozen fixed-controller critic, the residual
target with the stall-aware controller's next command through band.fixed_native) WITHOUT that approval call. Uniform
weights only (version 6's primary). Every function it calls is version 6's own, executed from verified bytes by step 2's
machinery (machinery.B is band.py, machinery.P is policy.py; this module never writes `import band`):
  band.bca_config, fixed_native, initial_state, band_rows_and_keys, fit_band_arrays, band_from_reference,
  bank_scores, uniform_threshold_check, width_rows, band_template, uniform_band;
  policy.BCARunner and policy.bca_arm (version 6's actor step with its paired lower edge, k = 1).

A band that cannot be fitted (the refresh gives no ready reference, or no finite threshold) is recorded with
failed: true; it is not a stop, and the version6_bca arm is then missing (every mark that needs it fails; it is
reported only).

Training happens only inside fit() and train_arm(); stub mode replaces both (stubs.py), and nothing here runs before
the owner's approval is imported.
"""
import sys

sys.dont_write_bytecode = True

import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import near80_constants as K  # noqa: E402

V6_ARM = "version6_bca"
V6_COEF = (1.0, 0.0, 0.0)          # version 6's primary: the paired lower edge, k = 1


def _mc():
    import policy_phase as PP
    return PP.load_step2()[1]


def fit(prep, critic, seed, world, learn, steps=None, log=None):
    """Version 6's band on the setting's frozen critic, uniform weights. Returns (Band or None, record, extras with
    the bank scores) ."""
    MC = _mc()
    B, L = MC.B, MC.L
    t0 = time.perf_counter()
    args = L.make_args(learn, seed)
    config = B.bca_config()
    fixed = L.FixedController(K.CONTROLLER, prep.physics, prep.obs_mean, prep.obs_std)
    native = B.fixed_native(critic, fixed, learn.lr)
    rng, state, cal = B.initial_state(args, config, native, L.DX, L.DA)
    models = (fixed, L.CRITIC, cal)
    n_steps = int(B.BAND_FIT_STEPS if steps is None else steps)
    rows, keys, refresh_key = B.band_rows_and_keys(rng, prep.n, n_steps, learn.batch_size, seed)
    record = dict(kind="near80-v6-band", seed=int(seed), world=world, critic="fixed", weighted=False, steps=n_steps,
                  settings=B.SETTINGS, data=prep.describe())
    try:
        state, accepted, diag = B.fit_band_arrays(args, config, models, state, prep.train, prep.bank, rows, keys,
                                                  refresh_key)
    except (ValueError, FloatingPointError) as error:
        return None, dict(record, failed=True, reason=f"the band fit failed: {error}"), None
    ref = state.posterior
    record.update(accepted_share=accepted, refresh=diag, native_sha256=B._tree_sha(state.native))
    if not bool(ref.ready):
        return None, dict(record, failed=True, reason="the refresh did not produce a ready reference"), None
    try:
        band = B.band_from_reference(ref, cal, np.asarray(prep.bank.obs), np.asarray(prep.bank.action))
    except ValueError as error:
        return None, dict(record, failed=True, reason=str(error)), None
    scores, e_bank = B.bank_scores(args, models, state, prep.bank, refresh_key, ref.cal_params, ref.residual_scale)
    record["uniform"] = dict(R=float(ref.threshold), lambda_hat=float(ref.lambda_hat),
                             lambda_hpd=float(ref.lambda_hpd), unit=float(ref.residual_scale), R_c=float(band.R_c),
                             U_max=float(band.U_max), certified=bool(band.certified), bank_rows=int(len(scores)),
                             threshold_recomputed=B.uniform_threshold_check(scores, refresh_key, config,
                                                                            ref.threshold))
    record.update(failed=False, seconds=time.perf_counter() - t0)
    if log:
        log(f"version 6's band: seed {seed} {world} R={record['uniform']['R']:.4g}")
    return band, record, dict(scores=np.asarray(scores, np.float64), eta=np.asarray(e_bank, np.float64))


def band_template():
    MC = _mc()
    return MC.B.band_template(MC.L.DX, MC.L.DA, False)


def width_rows(band, obs, actions):
    """Version 6's width U(s, a) at many (normalized observation, command) pairs (band.width_rows, uniform)."""
    MC = _mc()
    return MC.B.width_rows(MC.P.CAL, band, obs, actions, False)


def train_arm(pg, clone, critic, learn, seed, band, cfg, shared, steps, chunk, log=None):
    """The version6_bca arm: version 6's BCARunner (the room's RoomRunner with BCA's kinds) with the paired lower edge
    at k = 1 on this setting's band. shared: the room's shared arrays of this program (build_shared); the band and the
    width at the recorded commands are added. Returns (trained entry of the arm, runner)."""
    MC = _mc()
    import jax.numpy as jnp
    u_rec = width_rows(band, np.asarray(pg.train.obs), pg.actions)
    shared_b = dict(shared, band=band, u_rec=jnp.asarray(np.asarray(u_rec, np.float32)))
    runner = MC.P.BCARunner(pg, clone, critic, learn, seed, cfg=cfg, shared=shared_b, steps=steps, chunk=chunk)
    out = runner.run([MC.P.bca_arm(V6_ARM, "bca_u", V6_COEF)], log)
    return out[V6_ARM], runner
