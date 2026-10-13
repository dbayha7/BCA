"""EXPLORATORY (plan section 4.3). NO BCA: no band, calibrator, threshold, dose or BCA update; every signal here is a
stand-in or an oracle. The deployment triggers: at every step of an exact rollout the actor proposes a command a at
state s, and a trigger decides whether the fallback replaces it for that step.

Every score-based trigger fires when its score at (s, a) is above its 95% level: the ceil((n + 1) * 0.95)-th smallest
score (fallback.conformal_threshold) at the held-out recorded pairs, which are the calibration split's rows (200
recorded episodes no learner saw) outside the removed ball, as in the fallback diagnostic. A further 5,000
evaluation-split rows (the fallback diagnostic's) check that each level flags about 5% of recorded pairs.

  unrec          the perfect detector of past-80% proposals: any |a_k| > 80% (thruster_chain.past_knee). It uses
                 knowledge of the gap; a reference, not a method. No level.
  onestep        the one-step oracle, re-referenced (room spec change A1): [delta*(s, r(s)) - delta*(s, a)]_+, the
                 critic's extra overrating of a over the recorded command at the same state, by its exact one-step
                 error. delta*(s, a) = r(s, a) + 0.98 min_h Q_h(s', c(s')) - min_h Q_h(s, a) is the room harness's
                 oracle64 (one float64 simulator step, the frozen fixed-controller gapped critic of the world, c = the
                 stall-aware controller, the critic's scored policy). A deployment state has no recorded command, so
                 r(s) is the recording controller's own command at s without its wobble (normal driving, never the
                 full-throttle command: the commands the gapped log keeps). The same function sets the level at the
                 held-out recorded pairs, so the level absorbs the recorder's wobble.
  density        behaviour density: 1/2 sum_k ((a_k - c_k(s)) / sigma_k)^2, c the cloned actor of the world, sigma_k
                 the maximum-likelihood spread of the training commands around it (root mean square of a_ik - c_k(s_i)
                 over the gapped training rows). Fires when the plan's penalty [score - q95]_+ is positive.
  ratio          classifier ratio: log w_g(s, a), the logit of a classifier that separates target pairs
                 (s_i, clip(pi(s_i) + a_i - c(s_i), -1, 1)) from recorded pairs (s_i, a_i) at every gapped training
                 row (pi the deployed actor, c the cloned actor; equal class sizes, so the logit is log w). Network,
                 steps and optimiser as the earlier classifier readout (principled/classifier_ratio.py): 2 hidden
                 layers of 128 (ReLU), 10,000 Adam steps (learning rate 0.001), batch 512, logistic loss.
  heads          two-head disagreement |Q1 - Q2|(s, a) of the frozen fixed-controller gapped critic of the world.
Reported:
  onestep_lit    the plan's literal one-step trigger |delta*(s, a)| (no re-referencing).
  onestep_clone  onestep re-referenced to the cloned actor's command c(s) (the data's estimate of the recorded
                 command) instead of the recording controller's.
  ratio_refuse   the classifier ratio against plan 4.2's limit l* instead of its 95% level: fires when
                 log w_g(s, a) > log(0.1 / 0.9) + log sum_j w_g(s_j, a_j) over the gapped calibration bank's rows
                 (the score-free refusal limit of plain weighted split conformal at miscoverage 0.1, computed here in
                 NumPy; no BCA code).
  sa95           the fallback diagnostic's distance check, carried over: the 10th-nearest-neighbour distance of
                 (s, a) to the training (state, command) rows in standard units above its 95% level.

Shape readout (spec v2, change C4; reported, never gating): Signals.shape_readout reads every trigger along thruster-1
throttle at the room's 2,000 full-throttle-region training states (the room's check-(e) geometry), so that a reader
can tell whether a trigger's catches come from the command or only from the unusual state.
"""

import time

import numpy as np

import common as C

H = C.load_room_harness()
FB = C.load_fallback()
L, TC, RC, TR = H.L, H.TC, H.RC, H.TR

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import flax.linen as nn  # noqa: E402
import optax  # noqa: E402

LEVEL = 95
CLF_HIDDEN, CLF_STEPS, CLF_BATCH, CLF_LR = 128, 10_000, 512, 1e-3
REFUSE_MISCOVERAGE = 0.1
TAG_CLF = 81
WORLDS = ("stall", "no_stall")
ACTOR_TAG = dict(gapped=1, td3bc_gapped=2, no_stall_gapped=3)
QUANTILES = (50, 90, 95, 99)

PRIMARY_TRIGGERS = ("unrec", "onestep", "density", "ratio", "heads")
REPORTED_TRIGGERS = ("onestep_lit", "onestep_clone", "ratio_refuse", "sa95")
TRIGGERS = PRIMARY_TRIGGERS + REPORTED_TRIGGERS
SCORED = ("onestep", "onestep_clone", "onestep_lit", "density", "ratio", "heads")   # scores with a 95% level

TRIGGER_TEXT = dict(
    unrec="perfect detector: fires when the actor proposes past 80% on any thruster (uses knowledge of the gap)",
    onestep=("one-step oracle, re-referenced: [delta*(s, r(s)) - delta*(s, a)]_+ above its 95% level on held-out "
             "recorded pairs; r(s) the recording controller's command at s without its wobble"),
    density="behaviour density: 1/2 sum_k ((a_k - c_k(s)) / sigma_k)^2 above its 95% level on held-out recorded pairs",
    ratio="classifier ratio: log w_g(s, a) above its 95% level on held-out recorded pairs",
    heads="two-head disagreement: |Q1 - Q2|(s, a) above its 95% level on held-out recorded pairs",
    onestep_lit="the plan's literal one-step trigger: |delta*(s, a)| above its 95% level on held-out recorded pairs",
    onestep_clone="one-step oracle re-referenced to the cloned actor: [delta*(s, c(s)) - delta*(s, a)]_+ above its 95% level",
    ratio_refuse="classifier ratio above plan 4.2's limit l* (score-free refusal of weighted split conformal at 0.1)",
    sa95="distance check carried over from the fallback diagnostic: (state, command) 10th-neighbour distance above its 95% level",
)


class Clf(nn.Module):
    """principled/classifier_ratio.py's classifier, restated."""

    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(CLF_HIDDEN)(x))
        h = nn.relu(nn.Dense(CLF_HIDDEN)(h))
        return nn.Dense(1)(h)[..., 0]


def fit_classifier(X0, X1, seed_tags):
    """Logistic regression network: label 0 for X0 (recorded pairs), 1 for X1 (target pairs). Returns (params,
    losses every 1,000 steps, seconds)."""
    t0 = time.perf_counter()
    X = jnp.asarray(np.concatenate([X0, X1]), jnp.float32)
    y = jnp.asarray(np.concatenate([np.zeros(len(X0)), np.ones(len(X1))]), jnp.float32)
    rng = np.random.default_rng(list(seed_tags))
    params = Clf().init(jax.random.PRNGKey(int(rng.integers(2 ** 31 - 1))), X[:2])
    opt = optax.adam(CLF_LR)
    st = opt.init(params)

    @jax.jit
    def step(params, st, idx):
        def lf(p):
            return optax.sigmoid_binary_cross_entropy(Clf().apply(p, X[idx]), y[idx]).mean()
        loss, g = jax.value_and_grad(lf)(params)
        up, st = opt.update(g, st, params)
        return optax.apply_updates(params, up), st, loss

    losses = []
    for i in range(CLF_STEPS):
        params, st, loss = step(params, st, jnp.asarray(rng.integers(len(X), size=CLF_BATCH)))
        if (i + 1) % 1000 == 0:
            losses.append(float(loss))
    return params, losses, time.perf_counter() - t0


def heldout_rows(sd):
    """The fallback diagnostic's held-out recorded pairs (calibration split outside the removed ball) and its 5,000
    evaluation-split check rows, exactly as fallback.Reference selects them (float64 states and commands)."""
    log, gap = sd.log, sd.gap
    cal = log.split_rows("calibration")
    cal = cal[~gap.contains(log["actions"][cal])]
    ev = log.split_rows("evaluation")
    ev = ev[~gap.contains(log["actions"][ev])]
    ev = ev[np.sort(np.random.default_rng([TC.DATA_SEED, sd.rep, 77]).choice(len(ev), FB.EVAL_CHECK_ROWS,
                                                                             replace=False))]
    xc, ac = np.asarray(log["infos/state"][cal], np.float64), np.asarray(log["actions"][cal], np.float64)
    xe, ae = np.asarray(log["infos/state"][ev], np.float64), np.asarray(log["actions"][ev], np.float64)
    return xc, ac, xe, ae


def _quantiles(v):
    v = np.asarray(v, np.float64)
    return {str(q): float(np.quantile(v, q / 100)) for q in QUANTILES}


class Signals:
    """Every trigger's score and level for one deployed actor in one world.

    prep: the deployment world's gapped Prepared (its training rows, normalisation, physics, recorder spec);
    actor_prep: the actor's home world's Prepared (its own input normalisation); critic: the deployment world's frozen
    fixed-controller gapped critic; clone_params: the deployment world's cloned actor; ref: fallback.Reference of the
    deployment world (recorded range, distance check)."""

    def __init__(self, rep, world, actor_name, actor_params, actor_prep, prep, sd, critic, clone_params, ref,
                 log_fn=C.log):
        t0 = time.perf_counter()
        self.rep, self.world, self.actor_name = int(rep), world, actor_name
        self.prep, self.critic, self.ref = prep, critic, ref
        self.physics = prep.physics
        self.recorder = RC.Recorder(prep.spec, prep.physics)
        self.clone = L.mlp_policy(clone_params, prep, "cloned actor")
        self.actor = L.mlp_policy(actor_params, actor_prep, actor_name)
        self.seconds = {}
        info = dict(world=world, actor=actor_name, critic="frozen fixed-controller gapped critic of the deployment "
                    "world (acceptance.get_critic(prereg, sd, 'gap', 'stall_aware', train=False))",
                    next_command="stall_aware", gamma=float(self.physics.gamma), level=LEVEL)
        xc, ac, xe, ae = heldout_rows(sd)
        if len(xc) != ref.n_cal or len(xe) != ref.n_check:
            raise SystemExit("refused: held-out rows differ from the fallback diagnostic's")
        info.update(heldout_rows=int(len(xc)), check_rows=int(len(xe)))

        # behaviour density
        a_tr = np.asarray(prep.actions, np.float64)
        c_tr = self._clone(prep.states)
        self.sigma = np.sqrt(np.mean((a_tr - c_tr) ** 2, axis=0))
        info["density_sigma"] = self.sigma.tolist()

        # classifier ratio
        pi_tr = np.clip(self.actor(prep.states), -1.0, 1.0)
        tgt = np.clip(pi_tr + a_tr - c_tr, -1.0, 1.0)
        obs_tr = prep.normalize(prep.states)
        tags = (20261013, self.rep, TAG_CLF, WORLDS.index(world), ACTOR_TAG[actor_name])
        params, losses, sec = fit_classifier(np.concatenate([obs_tr, a_tr], 1), np.concatenate([obs_tr, tgt], 1), tags)
        self.clf = params
        self._clf_fn = jax.jit(lambda X: Clf().apply(self.clf, X))
        self.seconds["classifier_fit"] = sec
        lb = sd.log
        bank = prep.bank_rows
        xb, ab = np.asarray(lb["infos/state"][bank], np.float64), np.asarray(lb["actions"][bank], np.float64)
        lw = self._logw(xb, ab)
        m = float(lw.max())
        w = np.exp(lw - m)
        self.l_star = float(np.log(REFUSE_MISCOVERAGE / (1 - REFUSE_MISCOVERAGE)) + m + np.log(w.sum()))
        info["classifier"] = dict(hidden=CLF_HIDDEN, depth=2, steps=CLF_STEPS, batch=CLF_BATCH, lr=CLF_LR,
                                  seed_tags=list(tags), losses_every_1000=losses, seconds=sec,
                                  target_past80_share=float(TC.past_knee(tgt).mean()),
                                  actor_past80_share_at_training_states=float(TC.past_knee(pi_tr).mean()),
                                  refusal_limit_l_star=self.l_star, bank_rows=int(len(bank)),
                                  bank_kish_ess=float(w.sum() ** 2 / np.sum(w ** 2)))
        log_fn(f"  classifier fitted in {sec:.0f}s (loss {losses[-1]:.4f}), l* {self.l_star:.3f}, "
               f"bank ESS {info['classifier']['bank_kish_ess']:.1f}")

        # levels at the held-out recorded pairs; rates at the check rows
        t1 = time.perf_counter()
        self.thr, self.levels = {}, {}
        for name in SCORED:
            hs, cs = self.score(name, xc, ac), self.score(name, xe, ae)
            thr = FB.conformal_threshold(hs, LEVEL)
            self.thr[name] = thr
            self.levels[name] = dict(threshold=thr, heldout_quantiles=_quantiles(hs), check_quantiles=_quantiles(cs),
                                     heldout_rate_above=float(np.mean(hs > thr)),
                                     check_rate_above=float(np.mean(cs > thr)),
                                     heldout_zero_share=float(np.mean(hs == 0.0)))
            if name == "ratio":
                self.thr["ratio_refuse"] = self.l_star
                self.levels["ratio_refuse"] = dict(threshold=self.l_star, heldout_rate_above=float(np.mean(hs > self.l_star)),
                                                   check_rate_above=float(np.mean(cs > self.l_star)))
        d_rec = H.oracle64(critic, prep, xc, ac, self.physics)
        info["delta_at_heldout_recorded"] = dict(mean=float(d_rec.mean()), quantiles=_quantiles(d_rec))
        self.levels["sa95"] = dict(threshold=float(ref.sa.thr[95]), check_rate_above=float(ref.sa.check_rate[95]))
        self.seconds["levels"] = time.perf_counter() - t1
        info["levels"] = self.levels
        t2 = time.perf_counter()
        info["shape"] = self.shape_readout()
        self.seconds["shape"] = time.perf_counter() - t2
        log_fn(f"  shape readout ({self.seconds['shape']:.0f}s): " + ", ".join(
            f"{k} falls {v['fall_share_80_to_81']:.2f} fires {v['fire_rate']['0.8']:.2f}->{v['fire_rate']['1']:.2f}"
            for k, v in info["shape"]["triggers"].items()))
        self.seconds["total"] = time.perf_counter() - t0
        info["seconds"] = self.seconds
        self.info = info

    # ---------------------------------------------------------------------------------------- shape readout

    def shape_readout(self):
        """Reported, never gating (spec v2, change C4): does a trigger respond to the command, or only to the unusual
        state? The room's check-(e) geometry, read on triggers: at the room harness's 2,000 full-throttle-region
        training states of the deployment world (H.shape_rows: default_rng([20261015, seed, 65])), the deployed
        actor's command clip(pi(s), -1, 1) with thruster 1 set along the actor's own push to each throttle of the
        room's check-(e) grid (H.SHAPE_GRID; 80% is the float32 knee, which is not past it; the other thrusters stay
        at the actor's command). Per trigger: the share of states whose score falls / rises from 80% to 81% (the
        room's (e1) reading; the room's 25% limit is shown for reference and gates nothing here), the mean score and
        the firing rate at each grid throttle, and the firing rate at the actor's own command. Scores: the trigger's
        own score (unrec: max_k |a_k|; ratio_refuse: the classifier logit; sa95: the 10th-neighbour distance, at 80%
        and 81% only, with firing from its exact decision at every throttle). The critic surface, the classifier and
        the neighbour search only: no rollout, no return, no point value."""
        rows = H.shape_rows(self.prep, self.rep)
        x = np.asarray(self.prep.states, np.float64)[rows]
        base = np.clip(self.actor(x), -1.0, 1.0)
        sign = np.where(base[:, 0] < 0, -1.0, 1.0)
        grid = tuple(float(m) for m in H.SHAPE_GRID)

        def at(m):
            aa = base.copy()
            aa[:, 0] = sign * (TC.KNEE if m == 0.80 else m)
            return aa

        cmds = {m: at(m) for m in grid}
        out = dict(reported_only=True, gates_nothing=True, rows=int(len(rows)),
                   rows_rule=(f"room harness shape_rows: default_rng([20261015, {self.rep}, {H.TAG_SHAPE}]) choice of "
                              f"the deployment world's full-throttle-region training rows"),
                   grid=list(grid), actor_thruster1_past80_share=float(np.mean(np.abs(base[:, 0]) > TC.KNEE)),
                   actor_past80_share=float(np.mean(TC.past_knee(base, self.physics))),
                   push_negative_share=float(np.mean(sign < 0)),
                   room_e1_fall_share_limit_for_reference=float(H.SHAPE_MAX_FALL_SHARE), triggers={})
        for name in TRIGGERS:
            means = None
            if name == "sa95":
                s80 = self.ref.sa.dist(np.concatenate([x, cmds[0.80]], 1))
                s81 = self.ref.sa.dist(np.concatenate([x, cmds[0.81]], 1))
                fire = {m: self.fire(name, x, cmds[m]) for m in grid}
                fire_own = self.fire(name, x, base)
                score = "10th-neighbour distance in standard units (fallback.Novelty.dist), at 80% and 81% only"
            elif name == "unrec":
                S = {m: np.max(np.abs(cmds[m]), axis=1) for m in grid}
                fire = {m: self.fire(name, x, cmds[m]) for m in grid}
                fire_own = self.fire(name, x, base)
                score = "max_k |a_k|"
            else:
                S = {m: self.score(name, x, cmds[m]) for m in grid}
                fire = {m: S[m] > self.thr[name] for m in grid}
                fire_own = self.score(name, x, base) > self.thr[name]
                score = "the trigger's score (Signals.score), firing above its level (Signals.thr)"
            if name != "sa95":
                s80, s81 = S[0.80], S[0.81]
                means = {f"{m:g}": float(np.mean(S[m])) for m in grid}
            fall, rise = float(np.mean(s81 < s80)), float(np.mean(s81 > s80))
            rate = {f"{m:g}": float(np.mean(fire[m])) for m in grid}
            out["triggers"][name] = dict(
                score=score, fall_share_80_to_81=fall, rise_share_80_to_81=rise,
                unchanged_share_80_to_81=float(1.0 - fall - rise),
                fall_share_within_room_e1_limit=bool(fall <= H.SHAPE_MAX_FALL_SHARE), mean_score=means,
                fire_rate=rate, fire_rate_full_minus_80=float(rate["1"] - rate["0.8"]),
                fire_rate_at_actor_command=float(np.mean(fire_own)))
        return out

    # ---------------------------------------------------------------------------------------- scores

    def _clone(self, x):
        return np.clip(self.clone(np.asarray(x, np.float64)), -1.0, 1.0)

    def _logw(self, x, a):
        X = np.concatenate([self.prep.normalize(x), np.asarray(a, np.float32)], 1)
        return np.asarray(self._clf_fn(jnp.asarray(X, jnp.float32)), np.float64)

    def _delta(self, x, a):
        return H.oracle64(self.critic, self.prep, x, a, self.physics, next_command="stall_aware", q_head="min")

    def _rel(self, x, a, a_ref):
        n = len(x)
        d = self._delta(np.concatenate([x, x]), np.concatenate([np.asarray(a, np.float64), a_ref]))
        return np.maximum(d[n:] - d[:n], 0.0)

    def score(self, name, x, a):
        x = np.asarray(x, np.float64)
        a = np.asarray(a, np.float64)
        if name == "onestep":
            return self._rel(x, a, np.asarray(self.recorder(x), np.float64))
        if name == "onestep_clone":
            return self._rel(x, a, self._clone(x))
        if name == "onestep_lit":
            return np.abs(self._delta(x, a))
        if name == "density":
            return 0.5 * np.sum(((a - self._clone(x)) / self.sigma) ** 2, axis=1)
        if name in ("ratio", "ratio_refuse"):
            return self._logw(x, a)
        if name == "heads":
            q = L.critic_rows(self.critic, self.prep.normalize(x), a.astype(np.float32))[0]
            return np.abs(q[:, 0] - q[:, 1])
        raise ValueError(name)

    def fire(self, name, x, a):
        if name == "unrec":
            return TC.past_knee(a, self.physics)
        if name == "sa95":
            return self.ref.sa.novel(np.concatenate([x, a], 1), self.ref.sa.thr[95])
        return self.score(name, x, a) > self.thr[name]
