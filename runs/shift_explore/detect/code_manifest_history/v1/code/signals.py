"""EXPLORATORY detection rehearsal (plan section 4.4), NO BCA: the SIGNAL script. It never reads a label.

Each signal is a score at a (state, command) pair; larger means "more suspect". Every one is fitted from the world's
gapped training log, its frozen gapped critic, its cloned actor and its gapped critic's actor only (no label, no
return, no exact value of any policy). The oracles use one exact simulator step (as the room test's main oracle) and
are references, not data-only signals. Notation: s a state (float64 of a float32 number), a a command (float32),
[x]_+ = max(x, 0), z(s, a) the (state, command) standardised by the mean and SD of the world's kept training rows.

Data-only stand-ins (plan 4.4's list; the estimators of plan 4.2, read as scores):
  distance           d10(s, a): the Euclidean distance in z-space to the 10th-nearest kept training row (exact,
                     scipy cKDTree; the gate's distance-to-data space; in-sample at training states)
  density            N(s, a) = 1/2 sum_k ((a_k - c_k(s)) / sigma_k)^2, c the world's cloned actor (float32 network at
                     the normalised float32 state), sigma_k = sqrt(mean_i (a_ik - c_k(s_i))^2) over the kept training
                     rows (maximum likelihood). The plan's [N - q95]_+ flags the same pairs at the 95% level.
  classifier_ratio   log w(s, a) = the logit of a classifier g of target pairs against recorded pairs (stage2/
                     weights.py's estimator = standins.py's fit_classifier, restated): Dense 64 - ReLU - Dense 64 -
                     ReLU - Dense 1 on (normalised observation / 1.001, command); target pairs (s_i, clip(pi(s_i) +
                     a_i - c(s_i), -1, 1)) labelled 1 against recorded pairs (s_i, a_i) labelled 0 at the same rows,
                     pi the world's gapped critic's actor, c the cloned actor; balanced logistic loss; 2,000 Adam steps
                     (runtime.networks.torch_adam, learning rate 0.001) of 512 rows from default_rng([20261015, seed,
                     82]); initialisation from PRNGKey(default_rng([20261015, seed, 84]).integers(2^31 - 1)). The
                     plan's U_max * softplus(log w - l*) is a monotone transform of log w (U_max is a BCA quantity and
                     is not read); l* = log(0.1 / 0.9) + logsumexp over the calibration bank of log w (the "cannot
                     certify" limit) is recorded and its flags are reported beside the 95% level.
  two_head           |Q1(s, a) - Q2(s, a)| of the world's frozen gapped critic.
  residual_evidence  -(1/10) sum_{j in NN10(s, a)} delta_j: minus the mean one-step error of the 10 nearest kept
                     training rows in z-space (the distance's neighbours), delta_j = r_j + 0.98 min_h Q_h(s'_j,
                     c(s'_j)) - min_h Q_h(s_j, a_j) with the logged r_j and s'_j, c the stall-aware controller as the
                     critic's target reads it (learners.FixedController), no smoothing noise (BCA's residual form, the
                     room's delta* at recorded commands, the trace test bed's definition).
Oracles (references; they read the true first simulator step):
  perfect_band          the room's main oracle, re-referenced (change A1): [delta*(s, ref) - delta*(s, a)]_+, ref the
                        pair's reference command (the recorded command at T1, T2 and T4 states; the cloned actor's
                        command at T3 states, where no recorded command exists); delta*(s, a) = r(s, a) + 0.98 min_h
                        Q_h(s', c(s')) - min_h Q_h(s, a), (s', r) one float64 step of thruster_chain.dynamics_jax in
                        the world's physics, c the stall-aware controller (room harness oracle64, q_head='min').
                        The sanity check of the measures. Zero at every T4 row by construction (a = ref there).
  perfect_band_signed   -delta*(s, a) (reported): no reference; at one state its command-minus-cap difference equals
                        the main form's whenever both main-form values are positive.
  perfect_band_literal  |delta*(s, a)| (reported): the plan's literal form, which change A1 showed falls just past
                        the knee.
Reported data-only rival (no signal; plan section 2 point 8):
  range_hinge           100 * sum_j ([a_j - hi_j]_+^2 + [lo_j - a_j]_+^2), lo_j / hi_j the 0.5% / 99.5% quantiles of
                        the kept training commands (the room harness's hinge_bounds and HINGE_SCALE).
"""

import time

import numpy as np

import common as C

STANDIN_NAMES = ("distance", "density", "classifier_ratio", "two_head", "residual_evidence")
ORACLE_NAMES = ("perfect_band", "perfect_band_signed", "perfect_band_literal")
REPORTED_NAMES = ("range_hinge",)
ALL_NAMES = STANDIN_NAMES + ORACLE_NAMES + REPORTED_NAMES
CHEAP_SIGNAL = "two_head"
K_NN = 10
CLS_STEPS, CLS_BATCH, CLS_LR, CLS_HIDDEN = 2000, 512, 1e-3, 64
TAG_CLS_ROWS, TAG_CLS_INIT = 82, 84
MISCOVERAGE = 0.1
CHUNK = 8192
SCORES_DIR = "scores"


class Signal:
    """A scoring function with its record: name, units ('value': critic value units; 'other'), needs_ref (reads the
    pair's reference command), kind ('stand_in', 'oracle', 'reported'), formula, sha256 (of everything it fitted),
    info. Call: fn(states (m, 13) float64, commands (m, 4) float32[, refs (m, 4)]) -> (m,) float64."""

    def __init__(self, name, fn, units, kind, formula, sha256, info=None, needs_ref=False):
        self.name, self._fn, self.units, self.kind = name, fn, units, kind
        self.formula, self.sha256, self.info, self.needs_ref = formula, sha256, dict(info or {}), bool(needs_ref)

    def __call__(self, states, commands, refs=None):
        if self.needs_ref:
            if refs is None:
                raise ValueError(f"{self.name} needs the pairs' reference commands")
            return np.asarray(self._fn(states, commands, refs), np.float64)
        return np.asarray(self._fn(states, commands), np.float64)


# ------------------------------------------------------------------------------------------------ fitted pieces

class KNN:
    """Exact neighbours of (state, command) among the kept training rows in z-space (the gate's distance-to-data
    space, standardised by the training rows' own mean and SD), scipy cKDTree; cached for the last query."""

    def __init__(self, states, actions):
        from scipy.spatial import cKDTree

        X = np.concatenate([np.asarray(states, np.float64), np.asarray(actions, np.float32).astype(np.float64)], 1)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.tree = cKDTree((X - self.mu) / self.sd)
        self._cache = {}

    def query(self, states, commands):
        """(distances (m, 10), row indices (m, 10)), cached by the queries' content SHA-256 (distance and residual
        evidence share one search)."""
        key = C.content_sha256({"s": np.asarray(states, np.float64), "a": np.asarray(commands, np.float32)})
        if key not in self._cache:
            X = np.concatenate([np.asarray(states, np.float64), np.asarray(commands, np.float32).astype(np.float64)], 1)
            self._cache[key] = self.tree.query((X - self.mu) / self.sd, k=K_NN)
        return self._cache[key]


def one_step_errors(critic, pg):
    """delta_j at every kept training row (signals module docstring, residual_evidence)."""
    H, A, L, RC, TC, TR = C.gate()
    import jax.numpy as jnp

    fixed = L.FixedController("stall_aware", pg.physics, pg.obs_mean, pg.obs_std)
    nxt = np.asarray(pg.train.next_obs, np.float32)
    a_next = np.asarray(fixed.apply(fixed.params(), jnp.asarray(nxt)), np.float32)
    qn = L.critic_rows(critic, nxt, a_next)[0].min(1)
    q = L.critic_rows(critic, np.asarray(pg.train.obs), pg.actions)[0].min(1)
    return np.asarray(pg.train.reward, np.float64) + pg.physics.gamma * qn - q


def _classifier_net():
    import flax.linen as nn
    import jax.numpy as jnp

    class ClassifierNet(nn.Module):
        """g's logit: Dense 64 - ReLU - Dense 64 - ReLU - Dense 1 on (obs / 1.001, command)."""

        hidden: int = CLS_HIDDEN

        @nn.compact
        def __call__(self, obs, action):
            x = jnp.concatenate([obs / (1.0 + 0.001), action], axis=-1)
            x = nn.relu(nn.Dense(self.hidden)(x))
            x = nn.relu(nn.Dense(self.hidden)(x))
            return nn.Dense(1)(x).squeeze(-1)

    return ClassifierNet()


def fit_classifier(pg, none, clone, rep):
    """The classifier ratio's fit (module docstring). Returns (apply(obs, a) -> logit, l*, record)."""
    H, A, L, RC, TC, TR = C.gate()
    import jax
    import jax.numpy as jnp

    t0, c0 = time.perf_counter(), time.process_time()
    net = _classifier_net()
    seed = int(C.rng(rep, TAG_CLS_INIT).integers(2 ** 31 - 1))
    p0 = net.init(jax.random.PRNGKey(seed), jnp.zeros(TC.DX), jnp.zeros(TC.DA))
    rows = C.rng(rep, TAG_CLS_ROWS).integers(pg.n, size=(CLS_STEPS, CLS_BATCH))
    obs = pg.train.obs
    act = jnp.asarray(pg.actions)
    c_tr = jnp.asarray(L.actor_rows(clone, np.asarray(obs)))
    tx = L.adam(CLS_LR)

    def fit(p0, actor_params, rows):
        pi = L.ACTOR.apply(actor_params, obs)
        tgt = jnp.clip(pi + act - c_tr, -1.0, 1.0)

        def loss(p, r):
            lt = net.apply(p, obs[r], tgt[r])
            lr_ = net.apply(p, obs[r], act[r])
            return jnp.mean(jax.nn.softplus(-lt)) + jnp.mean(jax.nn.softplus(lr_))

        def body(carry, r):
            p, s = carry
            val, g = jax.value_and_grad(loss)(p, r)
            u, s = tx.update(g, s, p)
            return (jax.tree_util.tree_map(lambda x, y: x + y, p, u), s), val

        (p, _), ls = jax.lax.scan(body, (p0, tx.init(p0)), rows)
        bank_logit = net.apply(p, pg.bank.obs, pg.bank.action)
        lstar = jnp.log(MISCOVERAGE / (1.0 - MISCOVERAGE)) + jax.scipy.special.logsumexp(bank_logit)
        return p, lstar, ls, bank_logit

    p, lstar, ls, bank_logit = jax.jit(fit)(p0, none, jnp.asarray(rows, jnp.int32))
    ls = np.asarray(ls, np.float64)
    apply = jax.jit(net.apply)

    def logit(obs_norm, a):
        out = []
        for i in range(0, len(obs_norm), CHUNK):
            out.append(np.asarray(apply(p, jnp.asarray(obs_norm[i:i + CHUNK], jnp.float32),
                                        jnp.asarray(a[i:i + CHUNK], jnp.float32)), np.float64))
        return np.concatenate(out) if out else np.zeros(0)

    rec = dict(init_seed=seed, steps=CLS_STEPS, batch=CLS_BATCH, lr=CLS_LR, rows_sha256=L.tree_sha256(rows),
               loss_first100=float(ls[:100].mean()), loss_last100=float(ls[-100:].mean()), finite=bool(np.all(
                   np.isfinite(ls))), l_star=float(lstar), bank_rows=int(len(pg.bank_rows)),
               bank_logit_mean=float(np.mean(bank_logit)), bank_logit_max=float(np.max(bank_logit)),
               params_sha256=L.tree_sha256(p), seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0,
               estimator="stage2/weights.py = standins.py fit_classifier (restated; same network, rows, init, steps)")
    return logit, float(lstar), rec


# ------------------------------------------------------------------------------------------------ the signals

def stand_in_signals(ctx, world, names=ALL_NAMES, log_fn=C.log):
    """{name: Signal} for one seed and world, fitted from that world's artifacts only (module docstring). names:
    the subset to build (the fits of the others are skipped)."""
    H, A, L, RC, TC, TR = C.gate()
    t0 = time.perf_counter()
    rep = int(ctx.rep)
    pg = ctx.prepared(world)
    critic, crec = ctx.critic(world)
    clone = ctx.clone(world)[0]
    none = ctx.none(world)
    critic_sha = L.tree_sha256(critic)
    base = dict(world=world, rep=rep, prepared_sha256=pg.sha256)
    out, fits = {}, {}
    need_knn = any(n in names for n in ("distance", "residual_evidence"))
    knn = KNN(pg.states, pg.actions) if need_knn else None
    knn_sha = C.content_sha256({"mu": knn.mu, "sd": knn.sd}) if knn is not None else None

    if "distance" in names:
        def distance(s, a):
            return knn.query(s, a)[0][:, K_NN - 1]
        out["distance"] = Signal("distance", distance, "other", "stand_in",
                                 "d10(s, a): distance in z-space to the 10th-nearest kept training row",
                                 C.sha256_bytes(f"distance|{knn_sha}|{pg.sha256['rows']}".encode()), base)
    if "residual_evidence" in names:
        t1 = time.perf_counter()
        delta = one_step_errors(critic, pg)
        fits["residual_evidence"] = dict(delta_mean=float(delta.mean()), delta_sd=float(delta.std()),
                                         delta_sha256=L.tree_sha256(delta), seconds=time.perf_counter() - t1)

        def residual(s, a):
            return -delta[knn.query(s, a)[1]].mean(axis=1)
        out["residual_evidence"] = Signal(
            "residual_evidence", residual, "value", "stand_in",
            "-(1/10) sum_{j in NN10(s, a)} delta_j, delta_j = r_j + 0.98 min_h Q_h(s'_j, c(s'_j)) - min_h Q_h(s_j, a_j)",
            C.sha256_bytes(f"residual|{knn_sha}|{L.tree_sha256(delta)}".encode()),
            dict(base, **fits["residual_evidence"]))
    if "density" in names:
        a_clone = L.actor_rows(clone, np.asarray(pg.train.obs)).astype(np.float64)
        sigma = np.sqrt(np.mean((pg.actions.astype(np.float64) - a_clone) ** 2, axis=0))

        def density(s, a):
            c = L.actor_rows(clone, pg.normalize(s)).astype(np.float64)
            return 0.5 * np.sum(((np.asarray(a, np.float32).astype(np.float64) - c) / sigma) ** 2, axis=1)
        out["density"] = Signal("density", density, "other", "stand_in",
                                "N(s, a) = 1/2 sum_k ((a_k - c_k(s)) / sigma_k)^2, sigma by maximum likelihood",
                                C.sha256_bytes(f"density|{L.tree_sha256(clone)}|{sigma.tobytes().hex()}".encode()),
                                dict(base, sigma=sigma.tolist(), clone_sha256=L.tree_sha256(clone)))
    if "classifier_ratio" in names:
        logit, lstar, crec_ = fit_classifier(pg, none, clone, rep)
        fits["classifier_ratio"] = crec_

        def classifier(s, a):
            return logit(pg.normalize(s), np.asarray(a, np.float32))
        out["classifier_ratio"] = Signal("classifier_ratio", classifier, "other", "stand_in",
                                         "log w(s, a), the classifier's logit (target pairs against recorded pairs)",
                                         C.sha256_bytes(f"classifier|{crec_['params_sha256']}".encode()),
                                         dict(base, **crec_, extra_level={"l_star": lstar}))
    if "two_head" in names:
        def two_head(s, a):
            q = L.critic_rows(critic, pg.normalize(s), np.asarray(a, np.float32))[0]
            return np.abs(q[:, 0] - q[:, 1])
        out["two_head"] = Signal("two_head", two_head, "value", "stand_in", "|Q1(s, a) - Q2(s, a)| of the frozen critic",
                                 C.sha256_bytes(f"two_head|{critic_sha}".encode()), dict(base, critic_sha256=critic_sha))
    physics = pg.physics

    def delta_star(s, a):
        return H.oracle64(critic, pg, np.asarray(s, np.float64), np.asarray(a, np.float32).astype(np.float64),
                          physics, next_command="stall_aware", q_head="min")
    osha = C.sha256_bytes(f"oracle|{critic_sha}|{sorted(physics.as_dict().items())}".encode())
    if "perfect_band" in names:
        def perfect_band(s, a, ref):
            return np.maximum(delta_star(s, ref) - delta_star(s, a), 0.0)
        out["perfect_band"] = Signal("perfect_band", perfect_band, "value", "oracle",
                                     "[delta*(s, ref) - delta*(s, a)]_+ (the room's main oracle, change A1)", osha,
                                     dict(base, critic_sha256=critic_sha), needs_ref=True)
    if "perfect_band_signed" in names:
        out["perfect_band_signed"] = Signal("perfect_band_signed", lambda s, a: -delta_star(s, a), "value", "oracle",
                                            "-delta*(s, a)", osha, dict(base, critic_sha256=critic_sha))
    if "perfect_band_literal" in names:
        out["perfect_band_literal"] = Signal("perfect_band_literal", lambda s, a: np.abs(delta_star(s, a)), "value",
                                             "oracle", "|delta*(s, a)| (the plan's literal form)", osha,
                                             dict(base, critic_sha256=critic_sha))
    if "range_hinge" in names:
        lo, hi = H.hinge_bounds(pg.actions)

        def hinge(s, a):
            a = np.asarray(a, np.float32).astype(np.float64)
            return H.HINGE_SCALE * np.sum(np.maximum(a - hi, 0.0) ** 2 + np.maximum(lo - a, 0.0) ** 2, axis=1)
        out["range_hinge"] = Signal("range_hinge", hinge, "other", "reported",
                                    "100 sum_j ([a_j - hi_j]_+^2 + [lo_j - a_j]_+^2), the recorded range",
                                    C.sha256_bytes(f"hinge|{lo.tobytes().hex()}|{hi.tobytes().hex()}".encode()),
                                    dict(base, lo=lo.tolist(), hi=hi.tolist()))
    for s in out.values():
        s.info.setdefault("fit_seconds_total", time.perf_counter() - t0)
    if log_fn:
        log_fn(f"signals built seed {rep} {world}: {', '.join(out)} ({time.perf_counter() - t0:.0f}s)")
    return out


# ------------------------------------------------------------------------------------------------ scoring

def score_signal(name, fn, ctx, world, out_dir=None, *, pairs_out=C.OUT, units=None, extra=None, label=C.LABEL,
                 chunk=CHUNK, log_fn=C.log):
    """Score every pair of the locked pair sets of (seed, world) with fn and write the scores once, hashed:
    <out_dir>/<name>.npz and <name>.json (the record: the signal, its fitted-parameter SHA-256, the pair sets'
    SHA-256, the scores' file and content SHA-256). Never reads a label. fn: a Signal, or any callable
    fn(states (m, 13) float64, commands (m, 4) float32) -> (m,) with optional .name, .sha256 and .needs_ref (Stage 2
    passes BCA's WidthReader through detect.score_bca_width). Returns the record."""
    import pairs as P

    rep = int(ctx.rep)
    arrays, prec = P.load(rep, world, pairs_out)
    out_dir = C.seed_dir(rep, world, pairs_out) / SCORES_DIR if out_dir is None else out_dir
    out_dir = C.guard_out(out_dir)
    if (out_dir / f"{name}.npz").exists() or (out_dir / f"{name}.json").exists():
        raise SystemExit(f"refused: scores of {name} already exist in {out_dir}; scores are written once")
    s, a, ref = arrays["states"], arrays["commands"], arrays["refs"]
    t0, c0 = time.perf_counter(), time.process_time()
    needs_ref = bool(getattr(fn, "needs_ref", False))
    parts = []
    for i in range(0, len(s), chunk):
        args = (s[i:i + chunk], a[i:i + chunk]) + ((ref[i:i + chunk],) if needs_ref else ())
        v = np.asarray(fn(*args), np.float64)
        if v.shape != (len(args[0]),):
            raise ValueError(f"{name}: scores of shape {v.shape}, expected ({len(args[0])},)")
        parts.append(v)
    scores = np.concatenate(parts)
    sec, cpu = time.perf_counter() - t0, time.process_time() - c0
    if np.any(np.isnan(scores)):
        raise ValueError(f"{name}: NaN scores")
    fsha, csha = C.save_npz(out_dir / f"{name}.npz", {"scores": scores})
    rec = dict(label=label, kind="detect-scores", signal=name, rep=rep, world=world, file=f"{name}.npz",
               file_sha256=fsha, content_sha256=csha, pairs_dir=str(C.seed_dir(rep, world, pairs_out)),
               pairs_file_sha256=prec["file_sha256"], pairs_content_sha256=prec["content_sha256"],
               signal_sha256=getattr(fn, "sha256", None), units=units or getattr(fn, "units", "value"),
               signal_kind=getattr(fn, "kind", "external"), formula=getattr(fn, "formula", None),
               needs_ref=needs_ref, signal_info=C.plain(getattr(fn, "info", {})), n=int(len(scores)),
               infinite=int(np.sum(~np.isfinite(scores))), score_seconds=sec, score_cpu_seconds=cpu, reads_labels=False,
               **(extra or {}))
    C.write_json(out_dir / f"{name}.json", rec)
    if log_fn:
        log_fn(f"scored {name} seed {rep} {world}: {len(scores)} pairs ({sec:.0f}s)")
    return rec
