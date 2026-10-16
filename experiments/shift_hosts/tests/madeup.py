"""Made-up arrays for the stage-0 tests (plan section 4.5), shaped like ThrusterChain's but not its data, registered
with the guard so BCA's code may run on them (guard.require_made_up). An experiment-local attachment (plan section
3.1) that changes none of the repository's declared methods. numpy only (plus guard's hashing), deterministic from a
seed: every stream is np.random.default_rng([20261016, 7779, seed]) (tag 7779 is unused elsewhere; the weighting
test's synthetic arrays use 7777, learners_iw.py:327).

The made-up world, loosely after experiments/shift_iw/world.py (nothing here is the simulator):
  states      13 numbers: 10 float positions and speeds, the current c at index 10 (fixed per episode; training law
              density proportional to e^(-2c) on [0.1, 1.1], the strong law e^(+2c): world.inverse_cdf, lines 108-116),
              and two wave numbers at 11-12 (amplitude 0.1-0.4)
  commands    4 numbers in [-1, 1]: a made-up controller plus noise, rounded to float32; in half the episodes 30% of
              steps are uniformly random
  episodes    100 steps; next_obs is the next state (at t = 99 the state after the last step); done is 0 everywhere
  the ball    the 10% of training commands nearest the training commands' median, cut from training and calibration
              rows (never from the strong-law test steps), so ReBRAC's roll substitutes some next commands
  rows        train: 22 episodes, about 1,980 kept rows; bank: 5 kept rows from each of 60 calibration episodes (300
              rows); strong: 10 strong-law episodes (1,000 test steps, never cut)
  next_action train: Unifloral's roll over the kept rows, np.roll(action, -1, axis=0) (rebrac.py:335; equal to
              jnp.roll bit for bit); bank: the roll over the kept calibration rows in stored order, then the bank rows
              (plan section 3.8); strong: the roll over the test log
  weights     the exact strong-law weight w(c) = p_strong(c) / p_train(c) (world.ratio, lines 84-89 and 128-135), per
              row
  per row     R_levels: thresholds at 4 levels, [0.5, 1, 2, 4][row % 4] (as test_learners_iw.py:375-377); R_refused:
              +inf on every third row (row % 3 == 0), 1 elsewhere (test_learners_iw.py:393-394); dose: a fixed dose
              in [1, 1.5) (test a3); width: a fixed positive width (test a3, IQL). All float32 over training rows.

build(seed) registers with the guard: the container (data_sha256 = hash of (train, bank), as the weighting test's
data_sha256) and the bare pytrees a restated step or refresh receives, in Unifloral's Transition field order:
(obs, action, reward, next_obs, done) of train, bank and strong (td3_bc.py:62, cql.py:63, iql.py:60) and
(obs, action, reward, next_obs, next_action, done) of each (rebrac.py:68). Pass exactly these arrays (float32, the
whole set); any other subset or dtype is not registered and is refused.
"""

import math
from collections import namedtuple
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.shift_hosts import guard

MASTER_SEED = 20261016
STREAM = 7779
OBS, ACT, LENGTH = 13, 4, 100
C_LO, C_HI = 0.1, 1.1
LAMBDA_TRAIN, LAMBDA_STRONG = -2.0, 2.0
CUT_SHARE = 0.10
BANK_PER_EPISODE = 5

Rows = namedtuple("Rows", "obs action reward next_obs done")      # Unifloral's Transition order (td3_bc.py:62)
SETS = ("train", "bank", "strong")


def normaliser(lam):
    """M(lam) = integral of exp(lam c) over [0.1, 1.1] (restated from experiments/shift_iw/world.py:84-89)."""
    lam = float(lam)
    if lam == 0.0:
        return C_HI - C_LO
    return math.exp(lam * C_LO) * math.expm1(lam * (C_HI - C_LO)) / lam


def ratio_strong(c):
    """Exact w(c) = p_strong(c) / p_train(c) = (M(-2) / M(2)) exp(4c) (restated from world.py:128-135)."""
    c = np.asarray(c, np.float64)
    return (normaliser(LAMBDA_TRAIN) / normaliser(LAMBDA_STRONG)) * np.exp((LAMBDA_STRONG - LAMBDA_TRAIN) * c)


def inverse_cdf(u, lam):
    """The current law's quantile function (restated from world.py:108-116)."""
    u = np.asarray(u, np.float64)
    if lam == 0.0:
        return C_LO + (C_HI - C_LO) * u
    c = C_LO + np.log1p(u * math.expm1(lam * (C_HI - C_LO))) / lam
    return np.clip(c, C_LO, C_HI)


def _simulate(rng, episodes, lam, coupling):
    """(states float64 [E, 101, 13], commands float32 [E, 100, 4], rewards float32 [E, 100], c float64 [E])."""
    c = inverse_cdf(rng.uniform(size=episodes), lam)
    amp = rng.uniform(0.1, 0.4, episodes)
    phase = rng.uniform(0.0, 2.0 * math.pi, episodes)
    noisy = rng.uniform(size=episodes) < 0.5
    x = 0.5 * rng.standard_normal((episodes, 10))
    states = np.zeros((episodes, LENGTH + 1, OBS))
    commands = np.zeros((episodes, LENGTH, ACT), np.float32)
    rewards = np.zeros((episodes, LENGTH), np.float32)
    for t in range(LENGTH + 1):
        states[:, t, :10] = x
        states[:, t, 10] = c
        states[:, t, 11] = amp * np.sin(phase + 0.3 * t)
        states[:, t, 12] = amp * np.cos(phase + 0.3 * t)
        if t == LENGTH:
            break
        drive = -0.6 * x[:, :4] - 0.3 * x[:, 4:8] - 0.4 * c[:, None] + 0.35 * rng.standard_normal((episodes, ACT))
        random = noisy[:, None] & (rng.uniform(size=(episodes, 1)) < 0.3)
        a = np.where(random, rng.uniform(-1.0, 1.0, (episodes, ACT)), np.clip(drive, -1.0, 1.0))
        a = a.astype(np.float32)
        commands[:, t] = a
        r = (-0.1 * (x ** 2).sum(1) - 0.1 * (a.astype(np.float64) ** 2).sum(1) - c
             + 0.05 * rng.standard_normal(episodes))
        rewards[:, t] = r.astype(np.float32)
        x = 0.9 * x + a.astype(np.float64) @ coupling + 0.05 * c[:, None] + 0.05 * rng.standard_normal((episodes, 10))
    return states, commands, rewards, c


def _flatten(states, commands, rewards, c):
    E = commands.shape[0]
    obs = states[:, :LENGTH].reshape(E * LENGTH, OBS).astype(np.float32)
    nxt = states[:, 1:].reshape(E * LENGTH, OBS).astype(np.float32)
    return dict(obs=obs, action=commands.reshape(E * LENGTH, ACT), reward=rewards.reshape(E * LENGTH),
                next_obs=nxt, done=np.zeros(E * LENGTH, np.float32),
                episode=np.repeat(np.arange(E, dtype=np.int32), LENGTH),
                t=np.tile(np.arange(LENGTH, dtype=np.int32), E), current=np.repeat(c, LENGTH))


def _true_next(episode, t, rows):
    """Whether each row's rolled next command is its own episode's command at t + 1."""
    e, s = episode[rows], t[rows]
    out = np.zeros(len(rows), bool)
    out[:-1] = (e[1:] == e[:-1]) & (s[1:] == s[:-1] + 1)
    return out


def _take(d, rows):
    return {k: v[rows] for k, v in d.items()}


@dataclass(frozen=True, eq=False)
class MadeUp:
    """One seed's made-up arrays (module docstring). train / bank / strong are Rows of float32 arrays; the rest are
    aligned per-row extras."""

    world: str
    seed: int
    train: Rows
    bank: Rows
    strong: Rows
    next_action: dict       # {set: float32 [n, 4]} ReBRAC's a' by the set's rule
    next_true: dict         # {set: bool [n]} whether a' is the episode's own next command
    episode: dict           # {set: int32 [n]}
    t: dict                 # {set: int32 [n]}
    current: dict           # {set: float64 [n]} c (index 10, before float32 rounding)
    weight: dict            # {set: float64 [n]} exact strong-law weight w(c)
    in_ball: dict           # {set: bool [n]} command inside the made-up ball (False on every kept row)
    ball_center: Any        # float32 [4]
    ball_radius: float
    R_levels: Any           # float32 [n_train]
    R_refused: Any          # float32 [n_train]
    dose: Any               # float32 [n_train], in [1, 1.5)
    width: Any              # float32 [n_train], positive

    @property
    def n(self):
        return int(len(self.train.obs))

    def rows(self, which="train"):
        """{obs, action, reward, next_obs, done} of a set, for unifloral_import.make_transition."""
        return dict(getattr(self, which)._asdict())

    def leaves(self, which="train", learner="td3_bc"):
        """The set's arrays in Unifloral's Transition field order for `learner` (ReBRAC adds next_action before
        done, rebrac.py:68)."""
        r = getattr(self, which)
        if learner == "rebrac":
            return (r.obs, r.action, r.reward, r.next_obs, self.next_action[which], r.done)
        return (r.obs, r.action, r.reward, r.next_obs, r.done)

    def arrays(self):
        """Every array under flat names ("train/obs", ..., "rows/R_levels", "ball/center"), for handoff.write."""
        out = {}
        for s in SETS:
            for f, v in getattr(self, s)._asdict().items():
                out[f"{s}/{f}"] = v
            for name in ("next_action", "next_true", "episode", "t", "current", "weight", "in_ball"):
                out[f"{s}/{name}"] = getattr(self, name)[s]
        out.update({"rows/R_levels": self.R_levels, "rows/R_refused": self.R_refused, "rows/dose": self.dose,
                    "rows/width": self.width, "ball/center": self.ball_center,
                    "ball/radius": np.asarray(self.ball_radius, np.float64)})
        return out


def build(seed=0, *, train_episodes=22, calibration_episodes=60, strong_episodes=10, register=True):
    """The made-up arrays of `seed`, registered with the guard (module docstring) unless register=False."""
    rng = np.random.default_rng([MASTER_SEED, STREAM, int(seed)])
    coupling = 0.08 * rng.standard_normal((ACT, 10))
    tr = _flatten(*_simulate(rng, train_episodes, LAMBDA_TRAIN, coupling))
    ca = _flatten(*_simulate(rng, calibration_episodes, LAMBDA_TRAIN, coupling))
    st = _flatten(*_simulate(rng, strong_episodes, LAMBDA_STRONG, coupling))
    center = np.median(tr["action"], axis=0).astype(np.float32)
    dist = lambda a: np.sqrt(((a.astype(np.float64) - center) ** 2).sum(1))
    radius = float(np.quantile(dist(tr["action"]), CUT_SHARE))
    inside = lambda a: dist(a) <= radius

    keep = np.flatnonzero(~inside(tr["action"]))
    train = _take(tr, keep)
    keep_c = np.flatnonzero(~inside(ca["action"]))
    cal = _take(ca, keep_c)
    picks = []
    for e in range(calibration_episodes):
        rows = np.flatnonzero(cal["episode"] == e)
        picks.append(np.sort(rng.choice(rows, BANK_PER_EPISODE, replace=False)))
    bank_rows = np.concatenate(picks)
    bank = _take(cal, bank_rows)
    cal_next = np.roll(cal["action"], -1, axis=0)
    strong = st

    n = len(keep)
    level = np.array([0.5, 1.0, 2.0, 4.0], np.float32)
    R_levels = level[np.arange(n) % 4]
    R_refused = np.where(np.arange(n) % 3 == 0, np.inf, 1.0).astype(np.float32)
    dose = (1.0 + 0.5 * rng.uniform(size=n)).astype(np.float32)
    width = (0.05 + 0.5 * np.abs(rng.standard_normal(n))).astype(np.float32)

    def rows_of(d):
        return Rows(d["obs"], d["action"], d["reward"], d["next_obs"], d["done"])

    m = MadeUp(
        world=guard.MADE_UP_WORLD, seed=int(seed), train=rows_of(train), bank=rows_of(bank), strong=rows_of(strong),
        next_action=dict(train=np.roll(train["action"], -1, axis=0), bank=cal_next[bank_rows],
                         strong=np.roll(strong["action"], -1, axis=0)),
        next_true=dict(train=_true_next(tr["episode"], tr["t"], keep),
                       bank=_true_next(ca["episode"], ca["t"], keep_c)[bank_rows],
                       strong=_true_next(st["episode"], st["t"], np.arange(len(st["t"])))),
        episode=dict(train=train["episode"], bank=bank["episode"], strong=strong["episode"]),
        t=dict(train=train["t"], bank=bank["t"], strong=strong["t"]),
        current=dict(train=train["current"], bank=bank["current"], strong=strong["current"]),
        weight=dict(train=ratio_strong(train["current"]), bank=ratio_strong(bank["current"]),
                    strong=ratio_strong(strong["current"])),
        in_ball=dict(train=inside(train["action"]), bank=inside(bank["action"]), strong=inside(strong["action"])),
        ball_center=center, ball_radius=radius, R_levels=R_levels, R_refused=R_refused, dose=dose, width=width)
    if register:
        register_all(m)
    return m


def register_all(m):
    """Register m with the guard: the container and every bare pytree of module docstring. Returns the digests."""
    guard.register_made_up(m)
    digests = dict(container=guard.data_sha256(m))
    for s in SETS:
        for learner in ("td3_bc", "rebrac"):
            digests[f"{s}/{learner}"] = guard.register_arrays(m.leaves(s, learner), m.world)
    return digests
