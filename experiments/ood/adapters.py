"""Read-only TD3+BC/ReBRAC checkpoint queries, preserving original host equations.

The caller must bind provenance and independently verify prepared data/events.
Loading a model is not scientific run acceptance. IQL/CQL are not inferred from
these interfaces; their different state and target contracts need later adapters.
"""

import hashlib
import importlib
from pathlib import Path
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parity_gate(path, reference, actual, *, device):
    """Save actual/reference/errors before applying the unchanged absolute gate."""
    reference, actual = np.asarray(reference), np.asarray(actual)
    same_shape = reference.shape == actual.shape
    error = (np.abs(actual.astype(np.float64) - reference.astype(np.float64))
             if same_shape else np.asarray(np.nan))
    with Path(path).open("xb") as stream:
        np.savez(stream, reference=reference, actual=actual, error=error,
                 device=np.asarray(device), same_shape=np.asarray(same_shape), absolute_tolerance=np.asarray(1e-6))
    require(same_shape, "Action shapes differ; inputs archived.")
    require(np.isfinite(error).all() and (error <= 1e-6).all(), "Saved-action parity failed.")


def continuation_keys(seed, horizon):
    """One saved (time,2) key bank per state, reused across actions/continuations."""
    import jax
    require(type(seed) is int and 0 <= seed < 2**32, "Invalid base seed.")
    require(type(horizon) is int and 0 < horizon <= 1000, "Invalid key horizon.")
    base = jax.random.PRNGKey(seed)
    return np.stack([np.asarray(jax.random.fold_in(base, t)) for t in range(horizon)])


def verify_continuation_keys(keys, seed, horizon):
    expected = continuation_keys(seed, horizon)
    keys = np.asarray(keys)
    require(keys.dtype == expected.dtype and keys.shape == expected.shape
            and np.array_equal(keys, expected), "Changed continuation stream; competing actions must reuse it.")


def _schema_match(template, actual, path="checkpoint"):
    if isinstance(template, dict):
        require(isinstance(actual, dict) and set(template) == set(actual), "Missing/extra fields: " + path)
        for k in template:
            _schema_match(template[k], actual[k], path + "/" + k)
    elif template is None:
        require(actual is None, "Unexpected model component: " + path)
    else:
        a, b = np.asarray(template), np.asarray(actual)
        require(a.shape == b.shape, "Shape mismatch: " + path)
        # Flax step=0 is a Python integer at initialization, an int32 later.
        integer_scalar = a.shape == () and a.dtype.kind in "iu" and b.dtype.kind in "iu"
        require(integer_scalar or a.dtype == b.dtype, "Dtype mismatch: " + path)
        require(b.dtype.kind in "biuf", "Nonnumeric checkpoint leaf: " + path)
        # Positive infinity is meaningful only in posterior radius storage.
        valid = ~np.isnan(b) & (b >= 0) if "/radii/" in path else np.isfinite(b)
        require(valid.all(), "Invalid checkpoint values: " + path)


class CheckpointAdapter:
    def __init__(self, path, expected_sha256, host, args, config, prepared,
                 *, expected_step=1000000, fixture=False):
        from flax import serialization as S
        import jax.numpy as jnp
        from runtime.validation import checkpoint_counts
        require(host in ("td3_bc", "rebrac"), "No adapter for this host identity.")
        require(type(expected_step) is int and (expected_step == 1000000 or fixture is True),
                "Non-1M checkpoints are engineering fixtures only.")
        self.path = Path(path)
        payload = self.path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == expected_sha256, "Checkpoint hash mismatch.")
        self.host, self.args, self.config = host, args, config
        self.P = importlib.import_module("algorithms." + host + "_bca")
        require(type(args) is self.P.BASE.Args and type(config) is self.P.Config,
                "Wrong typed host arguments/configuration.")
        train = prepared.training
        self.obs_dim, self.action_dim = train.obs.shape[1], train.action.shape[1]
        self.mean, self.std = jnp.asarray(prepared.obs_mean), jnp.asarray(prepared.obs_std)
        require(self.mean.shape == self.std.shape == (self.obs_dim,) and
                np.isfinite(self.mean).all() and np.isfinite(self.std).all() and (self.std > 0).all(),
                "Invalid prepared observation normalization.")
        self.bound = prepared.max_action
        require(self.bound == 1.0, "First-tranche normalized action bound must be one.")
        rng, template, self.models = (self.P.initialize(args, config, train) if host == "rebrac"
            else self.P.initialize(args, config, self.obs_dim, self.action_dim, self.bound))
        expected = dict(state=template, training_rng=rng, step=jnp.int32(0))
        try:
            decoded = S.msgpack_restore(payload)
            _schema_match(S.to_state_dict(expected), decoded)
            self.counts = checkpoint_counts(decoded, "td3" if host == "td3_bc" else host,
                                            expected_step, args.policy_freq)
            restored = S.from_bytes(expected, payload)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Malformed checkpoint: " + str(exc)) from exc
        self.state = restored["state"]
        self._serialized = S.to_bytes(restored)
        self._restored = restored
        self._file_sha = expected_sha256
        if config.is_posterior:
            require(bool(self.P.posterior_storage_valid(self.state.posterior)), "Invalid stored posterior.")
            if host == "rebrac":
                require(bool(self.P.calibration_storage_valid(self.models, self.state)),
                        "Calibrator statistics disagree with prepared training data.")

    def assert_unchanged(self):
        from flax import serialization as S
        require(S.to_bytes(self._restored) == self._serialized, "Checkpoint query mutated model state.")
        require(hashlib.sha256(self.path.read_bytes()).hexdigest() == self._file_sha,
                "Checkpoint file changed after loading.")

    def _observations(self, observations, normalized=False):
        import jax.numpy as jnp
        x = jnp.asarray(observations)
        require(x.ndim == 2 and x.shape[1] == self.obs_dim and np.isfinite(x).all(),
                "Expected finite batched observations.")
        return x if normalized else (x - self.mean) / self.std

    def actions(self, observations):
        x = self._observations(observations)
        result = np.asarray(self.models[0].apply(self.state.native.actor.params, x))
        require(result.shape == (len(x), self.action_dim) and np.isfinite(result).all()
                and (np.abs(result) <= self.bound).all(), "Invalid actor outputs; no repair.")
        self.assert_unchanged()
        return result

    def _batch(self, batch):
        expected = self.P.BASE.C.TransitionNA if self.host == "rebrac" else self.P.BASE.C.Transition
        require(type(batch) is expected, "Exact host transition tuple required, including recorded next_action for ReBRAC.")
        n = len(batch.obs)
        require(n > 0 and batch.obs.shape == batch.next_obs.shape == (n, self.obs_dim)
                and batch.action.shape == (n, self.action_dim)
                and batch.reward.shape == batch.done.shape == (n,), "Transition shapes disagree.")
        require(all(np.isfinite(x).all() for x in batch) and np.isin(batch.done, [0, 1]).all()
                and (np.abs(batch.action) <= self.bound).all(), "Invalid transition values.")
        if self.host == "rebrac":
            require(bool(self.P.transition_valid(batch)), "Invalid recorded next actions.")

    def targets(self, batch, key):
        self._batch(batch)
        key = np.asarray(key)
        require(key.shape == (2,) and key.dtype == np.uint32, "Explicit saved uint32 target key required.")
        # Keep original reward/terminal/target-noise conventions, no invented max
        # over all actions and no replacement for ReBRAC's recorded next action.
        value = self.P.native_target(self.args, self.models, self.state.native, batch, key)
        require(np.isfinite(value).all(), "Nonfinite frozen target.")
        self.assert_unchanged()
        return np.asarray(value)

    def target_components(self, batch, key):
        """Independent arithmetic readout for auditing the original target function."""
        import jax
        import jax.numpy as jnp
        self._batch(batch)
        native = self.state.native
        actor = native.actor_target.params if self.host == "td3_bc" else native.actor.target_params
        critic = native.critic_target.params if self.host == "td3_bc" else native.critic.target_params
        proposed = self.models[0].apply(actor, batch.next_obs)
        noise = jnp.clip(jax.random.normal(key, proposed.shape) * self.args.policy_noise,
                         -self.args.noise_clip, self.args.noise_clip)
        action = jnp.clip(proposed + noise, -self.bound, self.bound)
        q = self.models[1].apply(critic, batch.next_obs, action)
        q = q.min(-1 if self.host == "td3_bc" else 0)
        if self.host == "rebrac":
            q -= self.args.critic_bc_coef * ((action - batch.next_action)**2).sum(-1)
        return action, q

    def score(self, observations, actions, *, normalized=False):
        if not self.config.is_posterior:
            return None  # Native width is N/A, not zero.
        import jax.numpy as jnp
        from calibration.dose import frozen_level_dose
        obs = self._observations(observations, normalized)
        actions = jnp.asarray(actions)
        require(actions.shape == (len(obs), self.action_dim) and np.isfinite(actions).all()
                and (np.abs(actions) <= self.bound).all(), "Invalid scoring actions.")
        post = self.state.posterior
        require(bool(post.ready), "No ready frozen posterior; do not interpret warmup zero as certainty.")
        scale = self.models[2].apply(post.cal_params, obs, actions)
        dose = frozen_level_dose(post, scale, "full", self.config.blend)
        require(bool(dose.inputs_valid), "Invalid frozen width/dose.")
        self.assert_unchanged()
        return {k: np.asarray(v) for k, v in dict(scale=scale, width=dose.width,
            dose=dose.dose, usable=dose.support_mask, radius=post.radii.radius[0],
            bayesian_radius=post.radii.bayesian_radius[0],
            conformal_radius=post.radii.conformal_radius[0], unit=post.residual_scale).items()}
