"""EXPLORATORY (not a gated output, changes no pre-registered rule; no BCA code runs here).

Shared loaders for the harm-anatomy scripts. Reads gate outputs only through the gate's own verified loaders
(pattern of outputs/shift/stop-report/cards/cards_outside_gate.py). Writes only under runs/shift_explore/anatomy.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import experiments.shift.acceptance as A
import experiments.shift.learners as L
import experiments.shift.recorder as RC
import experiments.shift.thruster_chain as TC
import experiments.shift.truth as TR

OUT = Path("/home/user/bca/runs/shift_explore/anatomy")
LABEL = "EXPLORATORY: computed outside the gate on saved gate artifacts; not a gated output; no BCA code was run"

H2_CRITICS = {  # name in this study: (world, removal)
    "gapped": ("stall", "gap"),
    "full": ("stall", "none"),
    "random": ("stall", "random"),
    "no_stall_gapped": ("no_stall", "gap"),
    "no_stall_full": ("no_stall", "none"),
}


class Seed:
    """Everything of one gate seed this study reads."""

    def __init__(self, rep, worlds=("stall",)):
        t = time.perf_counter()
        self.prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
        A.guard(self.prereg)
        self.cfg = self.prereg.config
        self.learn = self.cfg.learn
        self.outcome = A.gate_outcome(self.prereg)
        self.variant = self.outcome["variant"]
        self.alpha = self.outcome.get("alpha") or A._last_alpha_run(self.prereg, self.variant)
        self.rep = int(rep)
        self.sd = {w: A.load_seed(self.prereg.runs, self.cfg, self.variant, w, rep) for w in worlds}
        self.st = self.sd["stall"]
        self.pg = self.st.prepared("gap", self.learn)
        self.physics = self.st.physics
        self.recorder = RC.Recorder(self.st.spec, self.physics)
        self.pattern = np.asarray(self.recorder.pattern, np.float64)
        self.load_seconds = time.perf_counter() - t

    def prepared(self, world):
        return self.sd[world].prepared("gap", self.learn)

    # ------------------------------------------------------------------ critics
    def h2_critic(self, name):
        world, removal = H2_CRITICS[name]
        return A.get_critic(self.prereg, self.sd[world], removal, "stall_aware", train=False)[1]

    def host(self, removal="gap"):
        return A.get_host(self.prereg, self.st, removal, self.alpha, "plain", train=False)[1]

    # ------------------------------------------------------------------ actors
    def h2_actors(self):
        d = A._variant_root(self.prereg, "harm_return", self.variant) / f"alpha{self.alpha:g}" / f"seed{self.rep}"
        rec = A.finished(d / "values.json", self.prereg)
        acts = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in rec["actor_names"]},
                           rec["actors_sha256"])
        return {k: v for k, v in acts.items() if not rec["failed"].get(k)}, rec

    def room_actors(self):
        d = A._variant_root(self.prereg, "room", self.variant) / f"alpha{self.alpha:g}" / f"seed{self.rep}"
        blind = A.finished(d / "room_blind.json", self.prereg)
        acts = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in blind["names"]},
                           blind["actors_sha256"])
        vals = A.finished(d / "values.json", self.prereg)
        return acts, blind, vals


def critic_q(params, prepared, states, actions):
    """Critic head 1 at float64 states (rounded to float32, normalised) and commands (float32)."""
    obs = prepared.normalize(states)
    return L.critic_rows(params, obs, np.asarray(actions, np.float32))[0][:, 0]


def critic_q_grad(params, prepared, states, actions):
    obs = prepared.normalize(states)
    q, g = L.critic_rows(params, obs, np.asarray(actions, np.float32), grad=True)
    return q[:, 0], g


def actor_cmd(params, prepared, states):
    return L.actor_rows(params, prepared.normalize(states)).astype(np.float64)


def true_q(policy, states, actions, physics, horizon=1000):
    """Exact Q^policy (float64, JAX) at float32-rounded inputs (what a critic sees)."""
    x = np.asarray(np.asarray(states, np.float32), np.float64)
    a = np.asarray(np.asarray(actions, np.float32), np.float64)
    return TR.q_value_jax(policy, x, a, physics, horizon=horizon, chunk=8192)


def visited_states(policy, situation, physics, n_starts=200, every=1, rng=None, take=None):
    """States (and the policy's commands) along exact rollouts from the first n_starts of a situation's bank."""
    x0 = TC.bank_starts(situation)[:n_starts]
    run = TR.rollout(policy, x0, physics, record=True)
    S = run.states[:, :-1][:, ::every].reshape(-1, TC.DX)
    C = run.commands[:, ::every].reshape(-1, TC.DA)
    if take is not None and take < len(S):
        rng = rng or np.random.default_rng(0)
        idx = np.sort(rng.choice(len(S), take, replace=False))
        S, C = S[idx], C[idx]
    return S, C


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = dict(label=LABEL, **payload)
    path = OUT / name
    path.write_text(json.dumps(jsonable(payload), indent=1, sort_keys=True))
    print(f"wrote {path}", flush=True)
    return path


def log(*a):
    print(*a, flush=True)
    sys.stdout.flush()
