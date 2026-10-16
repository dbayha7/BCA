"""Float64 scoring adapters for the four Unifloral actors and CQL's declared scoring noise (plan section 3.9 and
choice 7 of section 4.4).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. Scoring
environment only (/opt/bca-shift-venv, numpy 2.2.6): it imports experiments/shift (truth.MLPPolicy and the
recorder's Controller), which refuses any other numpy (experiments/shift/__init__.py:101-103).

Each adapter is the learner's actor run in float64: its parameters cast to float64, the observation rounded to float32
first (the observation a deployed network sees, ThrusterChainEnv._obs; truth.MLPPolicy's round_obs), and nan_to_num on
the command, as every Unifloral evaluation does (_policy_step: td3_bc.py:140, cql.py:158, iql.py:158, rebrac.py:167).
They are built from an actor parameter file (actors_f32.read_actor_file; its layout is in actors_f32's docstring):
  TD3+BC  truth.MLPPolicy (truth.py:221-257) as is, given obs_std + 1e-3 as its spread (the network's own
          normalization, td3_bc.py:105, computed in float64 from the float32 fields), tanh output
  IQL     the same class on the mean layers Dense_0..2 (iql.py:117-126, eval=True: the mean, already through tanh);
          logstd is not used
  ReBRAC  ReBRACPolicy: raw observation, three hidden relu layers, tanh output (rebrac.py:127-139 with actor_ln and
          norm_obs off, rebrac.py:50-52)
  CQL     CQLPolicy: raw observation, three hidden relu layers, the log-spread head Dense_3 and the mean head Dense_4
          (creation order, cql.py:108-115); the command tanh(mean + std eps) with std = exp(clip(log_std, -5, 2)),
          eps the bank's declared noise at that start and step; CQLMeanPolicy: the readout tanh(mean), shown, not
          judged (cql.py's mean head has no tanh of its own, cql.py:115-119)
CQL's scoring noise (choice 7; INTERFACES.md section 4): for the bank with index b of BANKS,
  eps = np.random.default_rng([20261016, 1004, b]).standard_normal((1000, 100, 4)).astype(np.float32),
eps[j, t] being the noise of start j at step t, the same for every CQL setup and seed. Stream 1004 is unused elsewhere
(the start banks use 1000 and 1002; 1001 is retired, experiments/shift_iw/data.py:91-92).
"""

import numpy as np

from experiments.shift import recorder as RC
from experiments.shift import truth as TR
from experiments.shift_hosts import actors_f32 as A32

LEARNERS = A32.LEARNERS
# The 7 banks in index order (INTERFACES.md section 3, bca_core.BANKS): the weighting test's start banks, then the main
# study's four situations in the stalling world. The index is the CQL noise stream's last word.
BANKS = ("none", "mild", "strong", "recorded", "strong_current", "big_waves", "big_kicks")
MASTER_SEED = 20261016          # experiments/shift_iw/world.py:62
CQL_NOISE_STREAM = 1004         # bca_core.CQL_NOISE_STREAM (plan section 3.4)
BANK_SIZE, LENGTH, NUM_ACTIONS = 1000, 100, 4


def bank_index(bank):
    """The index of a bank (name or index) in BANKS."""
    if isinstance(bank, (int, np.integer)) and not isinstance(bank, bool):
        if not 0 <= int(bank) < len(BANKS):
            raise ValueError(f"bank index must be 0-{len(BANKS) - 1}")
        return int(bank)
    if bank not in BANKS:
        raise ValueError(f"unknown bank {bank!r}; one of {', '.join(BANKS)}")
    return BANKS.index(bank)


def cql_noise(bank):
    """CQL's declared scoring noise of a bank (module docstring): float32 [1000, 100, 4]. Always the whole bank's
    draw; a subset of starts takes rows of it (never a smaller draw)."""
    rng = np.random.default_rng([MASTER_SEED, CQL_NOISE_STREAM, bank_index(bank)])
    return rng.standard_normal((BANK_SIZE, LENGTH, NUM_ACTIONS)).astype(np.float32)


def _observe(x):
    """The float32 observation as float64 (truth.MLPPolicy's round_obs, truth.py:243-244)."""
    return np.asarray(x, np.float64).astype(np.float32).astype(np.float64)


def _layers(params, names):
    p = params.get("params", params)
    return ([np.asarray(p[k]["kernel"], np.float64) for k in names], [np.asarray(p[k]["bias"], np.float64)
                                                                       for k in names])


def _hidden(o, Ws, bs):
    h = o
    for W, b in zip(Ws, bs):
        h = np.maximum(h @ W + b, 0.0)
    return h


class MLPAdapter(RC.Controller):
    """TD3+BC's and IQL's adapter: truth.MLPPolicy on Dense_0..2 with obs_std + 1e-3 as its spread, then nan_to_num."""

    def __init__(self, learner, params, obs_mean, obs_std, name=None):
        p = params.get("params", params)
        mean_layers = {"params": {k: p[k] for k in ("Dense_0", "Dense_1", "Dense_2")}}
        spread = np.asarray(obs_std, np.float32).astype(np.float64) + A32.NORM_EPS
        self.learner = learner
        self.mlp = TR.MLPPolicy(mean_layers, np.asarray(obs_mean, np.float32).astype(np.float64), spread,
                                max_action=1.0, round_obs=True, name=f"{learner} actor (float64)")
        self.name = name or self.mlp.name

    def __call__(self, x):
        return np.nan_to_num(self.mlp(x))


class ReBRACPolicy(RC.Controller):
    """ReBRAC's adapter: tanh(W3 relu(W2 relu(W1 relu(W0 o + b0) + b1) + b2) + b3) on the raw float32 observation o,
    then nan_to_num (rebrac.py:127-139, actor_ln and norm_obs off)."""

    def __init__(self, params, name="rebrac actor (float64)"):
        self.Ws, self.bs = _layers(params, ("Dense_0", "Dense_1", "Dense_2", "Dense_3"))
        self.learner, self.name = "rebrac", name

    def __call__(self, x):
        h = _hidden(_observe(x), self.Ws[:3], self.bs[:3])
        return np.nan_to_num(np.tanh(h @ self.Ws[3] + self.bs[3]))


class _CQLBody:
    """CQL's actor up to its distribution, in float64: (mean, std) at the raw float32 observation (cql.py:108-120)."""

    def __init__(self, params):
        self.Ws, self.bs = _layers(params, ("Dense_0", "Dense_1", "Dense_2", "Dense_3", "Dense_4"))

    def mean_std(self, x):
        h = _hidden(_observe(x), self.Ws[:3], self.bs[:3])
        log_std = h @ self.Ws[3] + self.bs[3]
        std = np.exp(np.clip(log_std, A32.CQL_LOG_STD[0], A32.CQL_LOG_STD[1]))
        mean = h @ self.Ws[4] + self.bs[4]
        return mean, std


class CQLPolicy(RC.Controller):
    """CQL's scored adapter: tanh(mean + std eps[:, t]) with nan_to_num; eps float32 [n, 100, 4] holds the declared
    noise of the n starts being rolled, in their order (cql_noise(bank)[starts]). Time-indexed: truth.rollout calls
    act(x, t)."""

    def __init__(self, params, eps, name="cql actor (float64, declared noise)"):
        self.body = _CQLBody(params)
        eps = np.asarray(eps)
        if eps.dtype != np.float32 or eps.ndim != 3 or eps.shape[2] != NUM_ACTIONS:
            raise ValueError("eps must be float32 [n, steps, 4] (cql_noise(bank)[starts])")
        self.eps = eps
        self.learner, self.name = "cql", name

    def act(self, x, t):
        mean, std = self.body.mean_std(x)
        e = self.eps[:, int(t)].astype(np.float64)
        if len(e) != len(mean):
            raise ValueError(f"{len(mean)} states but noise for {len(e)} starts")
        return np.nan_to_num(np.tanh(mean + std * e))

    def __call__(self, x):
        raise TypeError("CQL's sampled command needs the step index; call act(x, t)")


class CQLMeanPolicy(RC.Controller):
    """CQL's readout: tanh(mean) with nan_to_num (deterministic; shown, not judged; choice 7)."""

    def __init__(self, params, name="cql actor, tanh of the mean (float64)"):
        self.body = _CQLBody(params)
        self.learner, self.name = "cql", name

    def __call__(self, x):
        mean, _ = self.body.mean_std(x)
        return np.nan_to_num(np.tanh(mean))


def adapter(actor, *, bank=None, starts=None, readout="scored"):
    """The float64 adapter of an ActorFile (actors_f32.read_actor_file). CQL's scored readout needs the bank (its
    declared noise) and, when only some starts are rolled, their indices in the bank (starts), in rollout order."""
    if readout not in A32.READOUTS[actor.learner]:
        raise ValueError(f"{actor.learner} has readouts {A32.READOUTS[actor.learner]}, not {readout!r}")
    if actor.learner in A32.NORMALIZED:
        return MLPAdapter(actor.learner, actor.params, actor.norm["obs_mean"], actor.norm["obs_std"])
    if actor.learner == "rebrac":
        return ReBRACPolicy(actor.params)
    if readout == "tanh_mean":
        return CQLMeanPolicy(actor.params)
    if bank is None:
        raise ValueError("CQL's scored command needs the bank whose declared noise it uses")
    eps = cql_noise(bank)
    if starts is not None:
        eps = eps[np.asarray(starts, np.int64)]
    return CQLPolicy(actor.params, eps)


def bank_noise(actor, bank, starts=None, readout="scored"):
    """The noise a Float32Policy needs for the same rollout as adapter(...): eps rows of the bank, or None."""
    if actor.learner != "cql" or readout != "scored":
        return None
    eps = cql_noise(bank)
    return eps if starts is None else eps[np.asarray(starts, np.int64)]
