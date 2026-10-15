"""READ-ONLY loaders for the consumer study (angle B: change how BCA's band is used).

Loads SAVED artifacts only: the Stage 2 weighted fixed-critic bands (band.msgpack via flax msgpack_restore, the
route of scratchpad/impact/exact.py), their tau.json, the uniform band.json (normalisation), the gate's gapped
fixed-controller critic (A.get_critic(train=False): read path only), the cloned actor, and the Stage 2 policy-test
actors (lock.json SHA-256 checked). No training, no band or classifier fit, no run_stage2 phase, no approval used.
Writes nothing outside this folder.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/bca")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import flax.linen as nn  # noqa: E402
from flax import serialization  # noqa: E402

from calibration.network import Calibrator  # noqa: E402
import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402
import experiments.shift.truth as TR  # noqa: E402

REPO = Path("/home/user/bca")
R2 = REPO / "runs/shift_explore/stage2"
DET = REPO / "runs/shift_explore/detect"
OUT = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/consumer")
KNEE = TC.KNEE
ALPHA = 0.1


class ClassifierNet(nn.Module):  # byte-for-byte the architecture of stage2/code/weights.py ClassifierNet
    hidden: int = 64

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs / (1.0 + 0.001), action], axis=-1)
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x).squeeze(-1)


CAL = Calibrator(jnp.zeros(TC.DX), jnp.ones(TC.DX), state_dep=True)
CLS = ClassifierNet()

_PRE = {}


def prereg():
    if "p" not in _PRE:
        p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
        A.guard(p)
        _PRE["p"] = p
        _PRE["variant"] = A.gate_outcome(p)["variant"]
    return _PRE["p"], _PRE["variant"]


def seed_data(world, rep):
    p, variant = prereg()
    sd = A.load_seed(p.runs, p.config, variant, world, rep)
    pg = sd.prepared("gap", p.config.learn)
    return sd, pg


def anchors(world):
    p, variant = prereg()
    return A.anchors_of(p, variant, world)


def critic(sd):
    p, _ = prereg()
    rec, params = A.get_critic(p, sd, "gap", "stall_aware", train=False)
    if params is None:
        raise SystemExit("critic failed in the gate")
    return params


def clone(world, rep):
    p, variant = prereg()
    d = A._variant_root(p, "clone", variant) / world / f"seed{rep}"
    rec = A.finished(d / "record.json", p)
    return L.load_tree(d / "actor.msgpack", L.actor_template(), rec["result"]["file_sha256"])


def policy_actors(world, rep):
    """Stage 2 policy-test actors (stall: R2/policy/seed<K>; no_stall: .../no_stall), every tree checked."""
    d = R2 / "policy" / f"seed{rep}"
    if world == "no_stall":
        d = d / "no_stall"
    lock = json.loads((d / "lock.json").read_text())
    acts = L.load_tree(d / lock["actors_file"], {k: L.actor_template() for k in lock["names"]},
                       lock["actors_file_sha256"])
    for n in lock["names"]:
        if L.tree_sha256(acts[n]) != lock["actor_sha256"][n]:
            raise SystemExit(f"actor {n} differs from the lock")
    vals = json.loads((d / "values.json").read_text())
    return acts, lock, vals


class Band:
    """The saved weighted fixed-critic band of one seed and world, evaluated with its frozen networks."""

    def __init__(self, world, rep, version="weighted"):
        base = R2 / "detect" / version / world / "fixed" / f"seed{rep}"
        uni = json.loads((R2 / "detect" / "uniform" / world / "fixed" / f"seed{rep}" / "band.json").read_text())
        self.rec = json.loads((base / "band.json").read_text())
        self.tau = float(json.loads((base / "tau.json").read_text())["tau"])
        b = serialization.msgpack_restore((base / "band.msgpack").read_bytes())
        self.b = b
        self.mean = np.asarray(uni["data"]["obs_mean"], np.float32)
        self.std = np.asarray(uni["data"]["obs_std"], np.float32)
        self.unit, self.U_max = float(b["unit"]), float(b["U_max"])
        self.weighted = version == "weighted"
        if self.weighted:
            self.l_star = float(b["l_star"])
            self.log_t = jnp.asarray(np.asarray(b["log_t"], np.float32))
            self.R_tab = jnp.asarray(np.asarray(b["R_tab"], np.float32))
        else:
            self.R = float(b["R"])
        cal_p, cls_p = b["cal_params"], b.get("cls_params")
        self._eta = jax.jit(lambda o, a: CAL.apply(cal_p, o, a))
        if self.weighted:
            self._lw = jax.jit(lambda o, a: CLS.apply(cls_p, o, a))
            log_t, R_tab, unit, U_max, l_star = self.log_t, self.R_tab, self.unit, self.U_max, self.l_star

            def uw(o, a):
                e = jnp.maximum(CAL.apply(cal_p, o, a), 1e-6)
                lw = CLS.apply(cls_p, o, a)
                return jnp.interp(lw, log_t, R_tab) * unit * e + U_max * jax.nn.softplus(lw - l_star)
            self._uw = jax.jit(uw)

    def norm(self, states):
        return (np.asarray(states, np.float32) - self.mean) / self.std

    def logw(self, obs_n, a):
        return np.asarray(self._lw(jnp.asarray(obs_n, jnp.float32), jnp.asarray(a, jnp.float32)), np.float64)

    def width(self, obs_n, a):
        if self.weighted:
            return np.asarray(self._uw(jnp.asarray(obs_n, jnp.float32), jnp.asarray(a, jnp.float32)), np.float64)
        e = np.maximum(np.asarray(self._eta(jnp.asarray(obs_n, jnp.float32), jnp.asarray(a, jnp.float32)),
                                  np.float64), 1e-6)
        return self.R * self.unit * e

    def eta(self, obs_n, a):
        return np.asarray(self._eta(jnp.asarray(obs_n, jnp.float32), jnp.asarray(a, jnp.float32)), np.float64)


_Q = {}


def qmin(critic_params, obs_n, a):
    if "f" not in _Q:
        _Q["f"] = jax.jit(lambda p, o, a: L.CRITIC.apply(p, o, a).min(axis=-1))
    return np.asarray(_Q["f"](critic_params, jnp.asarray(obs_n, jnp.float32), jnp.asarray(a, jnp.float32)),
                      np.float64)


def recorded_range(pg):
    a = np.asarray(pg.actions, np.float64)
    return np.quantile(a, 0.005, axis=0), np.quantile(a, 0.995, axis=0)


def state_box(pg, q=0.005):
    s = np.asarray(pg.states, np.float64)
    return np.quantile(s, q, axis=0), np.quantile(s, 1 - q, axis=0)


def dump(obj, name):
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))


def plain_wcp_limit(band, pg, rep):
    """Closed-form refusal limit of plain weighted split conformal (miscoverage 0.1) on the band's threshold half of
    the bank (band.split_bank's permutation, tag 71): log(alpha / (1 - alpha)) + log sum_j w_j, w_j = exp(logit)."""
    lw = band.logw(np.asarray(pg.bank.obs, np.float32), np.asarray(pg.bank.action, np.float32))
    perm = np.random.default_rng([20261015, int(rep), 71]).permutation(len(lw))
    a = lw[np.sort(perm[: len(lw) // 2])]
    m = a.max()
    return float(np.log(ALPHA / (1 - ALPHA)) + m + np.log(np.exp(a - m).sum()))
