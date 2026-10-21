"""Evaluation-only native BCA residual bands on the frozen designed query bank.

No optimizer, reference refresh, or policy selection occurs here. The residual
target is the current native noisy one-step target, not long-horizon policy Q.
"""

import numpy as np

from .continuous import KNEE, dynamics
from .training_eval import _deploy_actions, _observation_batch, CONTEXT_NAMES


CALIBRATION_EVALUATION_NAMESPACE = 2026102003
CALIBRATION_EVALUATION_SEED = 102001


def _scalar(value):
    value = float(np.asarray(value))
    return value if np.isfinite(value) else "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"


def evaluate_calibration(result, query_bank, *, seed=CALIBRATION_EVALUATION_SEED):
    """Return a JSON-compatible summary and raw [query-state, candidate] arrays.

    A ready infinite threshold gives vacuous coverage and a unit BC multiplier.
    Native dose usability is reported verbatim, never as certified support.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("diagnostic seed must be a nonnegative integer")
    identity = result.record.get("identity", {})
    common = dict(schema="thrusterchain-native-residual-evaluation-v1",
                  query_bank_sha256=query_bank.metadata["arrays_sha256"],
                  target="current native one-step noisy TD3 target minus current twin-min Q",
                  coverage_claim="empirical designed-query residual coverage only; no long-horizon Q coverage guarantee",
                  support_semantics="native dose usability, not certified statistical or population action support",
                  selection="none; evaluation cannot fit, refresh, select, restart, or modify training")
    if identity.get("arm") != "bca":
        return dict(common, status="absent", reason="method has no native BCA band", coverage=None), {}
    if result.carry is None:
        return dict(common, status="refused", reason=result.record.get("reason", "no BCA policy checkpoint"),
                    coverage=None), {}
    from . import training as T
    jax, jnp, _, native, _, _, Transition, _ = T._stack()
    settings = T.TrainingSettings(**identity["settings"])
    objective = query_bank.objective
    if objective.horizon != settings.horizon or objective.gamma != settings.gamma:
        raise ValueError("query and training objectives differ")
    args = T.make_args(settings, int(identity["seed"]))
    states0, regimes0, times0 = query_bank.states, query_bank.regimes, query_bank.times
    if np.any(times0 >= objective.horizon):
        raise ValueError("native residual queries must be nonterminal")
    obs0 = _observation_batch(states0, regimes0, times0, objective, True)
    actor = _deploy_actions(result.policy(obs0), len(obs0), True)
    fixed = np.array([np.zeros(4), np.full(4, -KNEE), np.full(4, KNEE),
                      np.full(4, -1.), np.full(4, 1.)])
    actions = np.concatenate((np.broadcast_to(fixed, (len(obs0), 5, 4)), actor[:, None, :]), axis=1)
    actions = _deploy_actions(actions.reshape(-1, 4), len(obs0) * 6, True)
    if not np.isfinite(actions).all():
        return dict(common, status="failed", reason="nonfinite deployed candidate actions", coverage=None), {}
    states, regimes, times = (np.repeat(value, 6, axis=0) for value in (states0, regimes0, times0))
    observations = np.repeat(obs0, 6, axis=0)
    nxt, rewards = np.empty_like(states), np.empty(len(states))
    for regime in (0, 1):
        mask = regimes == regime
        if np.any(mask):
            nxt[mask], rewards[mask] = dynamics(states[mask], actions[mask], regime)
    next_obs = _observation_batch(nxt, regimes, times + 1, objective, True)
    batch = Transition(jnp.asarray((observations - result.mean) / result.std, jnp.float32),
                       jnp.asarray(actions, jnp.float32), jnp.asarray(rewards, jnp.float32),
                       jnp.asarray((next_obs - result.mean) / result.std, jnp.float32),
                       jnp.asarray(times + 1 == objective.horizon, jnp.float32))
    # Independent, fixed evaluation noise. Do not consume the checkpoint PRNG.
    noise_key = jax.random.fold_in(jax.random.PRNGKey(seed), CALIBRATION_EVALUATION_NAMESPACE)
    noise_key = jax.random.fold_in(noise_key, 1213156420)
    target = np.asarray(native.native_target(args, result.models, result.state.native, batch, noise_key), np.float64)
    q = np.asarray(result.models[1].apply(result.state.native.critic.params, batch.obs, batch.action).min(axis=-1), np.float64)
    dose = native.bc_readout(result.models, result.state, batch, settings.blend)
    width, multiplier, usable = (np.asarray(value) for value in (dose.width, dose.dose, dose.support_mask))
    residual = target - q
    arrays = dict(target=target, q=q, reward=np.asarray(batch.reward, np.float64), residual=residual,
                  absolute_residual=np.abs(residual), width=width, dose=multiplier, dose_usable=usable,
                  context=np.repeat(query_bank.contexts, 6), time=times,
                  done=np.asarray(batch.done, bool))
    arrays = {key: value.reshape(len(obs0), 6) for key, value in arrays.items()}
    arrays["actions"] = actions.reshape(len(obs0), 6, 4)
    if not bool(np.asarray(dose.inputs_valid)) or not np.isfinite(target).all() or not np.isfinite(q).all():
        return dict(common, status="failed", reason="native residual or band inputs invalid", coverage=None), arrays
    reference = result.state.posterior
    ready = bool(np.asarray(reference.ready))
    threshold = float(np.asarray(reference.threshold))
    covered = np.abs(residual) <= width
    arrays["covered"] = covered.reshape(len(obs0), 6)
    def summary(mask):
        finite_width = width[mask][np.isfinite(width[mask])]
        return dict(queries=int(np.sum(mask)),
                    coverage=float(np.mean(covered[mask])) if ready and np.any(mask) else None,
                    mean_finite_half_width=float(np.mean(finite_width)) if len(finite_width) else None,
                    infinite_half_width_fraction=float(np.mean(np.isposinf(width[mask]))) if np.any(mask) else None,
                    native_dose_usable_fraction=float(np.mean(usable[mask])) if np.any(mask) else None,
                    mean_bc_multiplier=float(np.mean(multiplier[mask])) if np.any(mask) else None,
                    mean_absolute_residual=float(np.mean(np.abs(residual[mask]))) if np.any(mask) else None)
    last_refresh = result.record.get("refreshes", [])
    checkpoint_step = int(np.asarray(result.carry[2]))
    last_refresh = [item for item in last_refresh if int(item["update"]) <= checkpoint_step]
    last_refresh = last_refresh[-1] if last_refresh else None
    pooled = summary(np.ones(len(target), bool))
    status = "unready" if not ready else "uncertified" if np.isposinf(threshold) else "evaluated"
    record = dict(common, status=status, checkpoint_update=checkpoint_step, query_states=len(obs0), candidates=6,
                  query_law=query_bank.metadata.get("query_law", "explicit frozen query bank"),
                  seed=seed, noise_key_folds=[CALIBRATION_EVALUATION_NAMESPACE, 1213156420],
                  posterior_ready=ready, frozen_threshold=_scalar(reference.threshold),
                  frozen_residual_scale=_scalar(reference.residual_scale),
                  last_refresh_update=None if last_refresh is None else int(last_refresh["update"]),
                  last_refresh_numerically_certified=None if last_refresh is None else last_refresh.get("posterior_certified"),
                  band_definition="current twin-min Q plus/minus frozen threshold times frozen positive scale",
                  frozen_reference="unchanged checkpoint reference; generally from update 95000 at final 100000",
                  reuse_limit="calibration bank was reused adaptively; query coverage is empirical and not a fresh-holdout guarantee",
                  coverage=pooled["coverage"], coverage_vacuous=bool(ready and np.isposinf(threshold)),
                  band_refusal_rate=float(np.mean(~(usable & ready))),
                  band_refusal_semantics="fraction without both a ready reference and native dose usability; no actions are refused",
                  nominal_residual_coverage=1.0 - settings.miscoverage,
                  pooled=pooled,
                  strata={name: summary(np.repeat(query_bank.contexts, 6) == index)
                          for index, name in enumerate(CONTEXT_NAMES)},
                  refusal_definition="unready or ready infinite threshold is not a usable calibrated intervention; native fallback multiplier is one",
                  executed_policy_refusal=False)
    return record, arrays
