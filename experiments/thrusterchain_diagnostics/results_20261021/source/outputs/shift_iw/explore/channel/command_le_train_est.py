"""The return test with estimated instead of exact weights (EXPLORATORY; pass marks frozen in
channel/PASSMARKS_estimated_weights.md before this code was written; the owner's go: "Proceed with all three").

Estimated weight of a (row, command): w_hat(observation) x odds_hat(observation, command), where
  w_hat     version 5's estimated current-law weight (classifier.py, the strong tilt's logistic regression)
  odds_hat  a classifier separating recorded commands (class 0) from the actor's command plus its spread (class 1,
            BCA's own target noise 0.2 clipped at 0.5) at the kept training rows' states, equal classes, so its odds
            estimate pi_h(a | s) / (beta(a | s) Z(s)); 2 x 128 ReLU, Adam 1e-3, 3,000 steps, 512 per class, refitted
            from scratch at every refresh.
Refusal: "no value credit" where the command is inside the removed ball or its estimated odds exceed the largest
estimated odds of any recorded command among the kept training rows; version 5's capped threshold where the test mass
is past the onset of refusal; the band penalty elsewhere. Nothing exact about the recording process or the laws is
used. Arms A3 (estimated, strong law) and C3 (untargeted, matched to A3's no-credit share).

Usage (from /root/bca-shift, PYTHONPATH=/root/bca-shift, JAX_PLATFORMS=cpu):
    python command_le_train_est.py checks|train|evaluate <quality> <seed>
"""
import hashlib
import importlib.util
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from experiments.shift import learners as LR
from experiments.shift import thruster_chain as TC
from experiments.shift_iw import data as D
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import lower_edge as LE
from experiments.shift_iw import run as RN
from experiments.shift_iw import world as W

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load("cle", "command_le_train.py")
F1 = _load("cle_f1", "command_le_train_f1.py")
CB = T.CB

OUT = T.OUT.parent / "train_est"
LABEL = ("EXPLORATORY: outside the version-5 pre-registration; pass marks frozen in "
         "channel/PASSMARKS_estimated_weights.md before this code was written")
PASSMARKS = HERE / "PASSMARKS_estimated_weights.md"
ARMS = ("A3_estimated", "C3_untargeted")
TAG_ODDS = 4245
ODDS_STEPS, ODDS_BATCH, ODDS_LR = 3000, 512, 1e-3


def passmarks_ok():
    want = (HERE / "PASSMARKS_estimated_weights.md.sha256").read_text().split()[0]
    got = hashlib.sha256(PASSMARKS.read_bytes()).hexdigest()
    if want != got:
        raise SystemExit(f"the frozen pass marks changed: {got} != {want}")
    return got


def this_code():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), HERE / "command_le_train.py", HERE / "command_le_train_f1.py",
                      HERE / "command_band_check.py")}


# ------------------------------------------------------------------------------------------------ the odds classifier

class OddsNet(nn.Module):
    @nn.compact
    def __call__(self, obs, a):
        h = jnp.concatenate([obs, a], axis=-1)
        h = nn.relu(nn.Dense(128)(h))
        h = nn.relu(nn.Dense(128)(h))
        return nn.Dense(1)(h)[..., 0]


NET = OddsNet()
_FIT = {}


def _fit_program():
    if "fit" not in _FIT:
        tx = optax.adam(ODDS_LR)

        def step(carry, batch):
            params, opt = carry
            o0, a0, o1, a1 = batch

            def loss_fn(p):
                l0 = NET.apply(p, o0, a0)
                l1 = NET.apply(p, o1, a1)
                return jnp.mean(jax.nn.softplus(l0)) + jnp.mean(jax.nn.softplus(-l1))

            loss, g = jax.value_and_grad(loss_fn)(params)
            upd, opt = tx.update(g, opt, params)
            return (optax.apply_updates(params, upd), opt), loss

        def run(params, o_all, a_rec, a_pi, r0, r1, z):
            opt = tx.init(params)
            batches = (o_all[r0], a_rec[r0], o_all[r1], jnp.clip(a_pi[r1] + z, -1.0, 1.0))
            (params, _), losses = jax.lax.scan(step, (params, opt), batches)
            return params, losses

        _FIT["fit"] = jax.jit(run)
    return _FIT["fit"]


def fit_odds(obs, a_rec, a_pi, seed, refresh):
    """The command-odds classifier at the current actor (deterministic: default_rng([20261016, seed, 4245, refresh]))."""
    rng = np.random.default_rng([W.MASTER_SEED, int(seed), TAG_ODDS, int(refresh)])
    n = len(obs)
    key = jax.random.PRNGKey(int(rng.integers(2 ** 31 - 1)))
    params = NET.init(key, jnp.zeros((1, obs.shape[1]), jnp.float32), jnp.zeros((1, TC.DA), jnp.float32))
    r0 = rng.integers(n, size=(ODDS_STEPS, ODDS_BATCH))
    r1 = rng.integers(n, size=(ODDS_STEPS, ODDS_BATCH))
    z = np.clip(CB.SPREAD * rng.standard_normal((ODDS_STEPS, ODDS_BATCH, TC.DA)), -CB.CLIP, CB.CLIP)
    params, losses = _fit_program()(params, jnp.asarray(obs, jnp.float32), jnp.asarray(a_rec, jnp.float32),
                                    jnp.asarray(a_pi, jnp.float32), jnp.asarray(r0), jnp.asarray(r1),
                                    jnp.asarray(z, jnp.float32))
    losses = np.asarray(losses)
    return params, dict(final_loss=float(losses[-200:].mean()), first_loss=float(losses[:50].mean()))


_LOGIT = {}


def logits(params, obs, a, chunk=16384):
    if "f" not in _LOGIT:
        _LOGIT["f"] = jax.jit(lambda p, o, x: NET.apply(p, o, x))
    out = [np.asarray(_LOGIT["f"](params, jnp.asarray(obs[i:i + chunk], jnp.float32),
                                  jnp.asarray(a[i:i + chunk], jnp.float32)), np.float64)
           for i in range(0, len(obs), chunk)]
    return np.concatenate(out)


# ------------------------------------------------------------------------------------------------ in-graph

def band_read_est(arm, idx, obs, a):
    """(R to use, no-credit flag, size-refused flag) at rows idx, observations obs and commands a (no gradient)."""
    a0 = jax.lax.stop_gradient(a)
    lo = NET.apply(arm["odds"], obs, a0)
    lm = arm["log_w"][idx] + lo
    in_ball = jnp.sum(jnp.square(a0 - arm["centre"]), axis=-1) <= arm["rho2"]
    absent = in_ball | (lo > arm["odds_max"])
    m = jnp.exp(jnp.clip(lm, arm["log_floor"], 80.0))
    gi = jnp.searchsorted(arm["grid"], m, side="left")
    R = arm["thresholds"][jnp.minimum(gi, T.MAXG - 1)]
    size_ref = ~absent & ((gi >= arm["n_grid"]) | ~jnp.isfinite(R))
    return jnp.where(absent, 1.0, jnp.where(size_ref, arm["cap"], R)), absent, size_ref


def penalty_est(critic_params, shared, idx, obs, a, arm):
    q1 = LR.CRITIC.apply(critic_params, obs, a)[..., 0]
    w_ref = jnp.maximum(q1 - arm["q1_rec"][idx], 0.0)
    R, absent, _ = band_read_est(arm, idx, obs, a)
    U = LE.band_width(shared["cal_params"], shared["unit"], obs, a, R)
    w_cert = jnp.maximum(U - jax.lax.stop_gradient(arm["u_logged"][idx]), 0.0)
    return jnp.where(absent, w_ref, w_cert)


_PROGRAMS = {}


def program(args, kind):
    key = (kind, LR._args_key(args))
    if key not in _PROGRAMS:
        pen = penalty_est if kind == "band" else T.penalty_untargeted
        _PROGRAMS[key] = jax.jit(partial(LE._loop, args, pen))
    return _PROGRAMS[key]


# ------------------------------------------------------------------------------------------------ refresh

class Est:
    """The estimated-weight ingredients of one World that do not depend on the actor."""

    def __init__(self, wd):
        self.wd = wd
        clf = RN.classifiers_of(wd.data)["strong"]
        log = wd.data.train_log
        self.log_w_train = clf.log_weight(np.asarray(log["observations"][wd.prep.rows], np.float64))
        self.log_w_bank = clf.log_weight(np.asarray(log["observations"][wd.prep.bank_rows], np.float64))
        self.law_classifier = clf.describe()
        self.bank_obs = np.asarray(wd.prep.bank.obs, np.float32)
        self.a_rec32 = np.asarray(wd.prep.actions, np.float32)

    def refresh(self, actor_params, refresh_idx):
        wd = self.wd
        a_pi = np.asarray(LR.actor_rows(actor_params, wd.obs), np.float32)
        odds, fit_info = fit_odds(wd.obs, self.a_rec32, a_pi, wd.seed, refresh_idx)
        lo_bank = logits(odds, self.bank_obs, np.asarray(wd.a_bank, np.float32))
        w_bank = np.exp(np.clip(self.log_w_bank + lo_bank, -700, 700))
        th = CB.Thresholds(wd.scores, wd.stream, w_bank)
        if len(th.table.grid) > T.MAXG:
            raise SystemExit(f"the table has {len(th.table.grid)} grid points, more than {T.MAXG}")
        lo_rec = logits(odds, wd.obs, self.a_rec32)
        odds_max = float(lo_rec.max())
        R_rec, _ = th.read(self.log_w_train + lo_rec)
        u_rec = LE.logged_widths(wd.band, wd.prep, LE.as_float32_thresholds(R_rec), stage=T.STAGE)
        lo_pi = logits(odds, wd.obs, a_pi)
        a64 = a_pi.astype(np.float64)
        absent = np.asarray(wd.gap.contains(a64), bool) | (lo_pi > odds_max)
        _, beyond = th.read(np.where(absent, np.inf, self.log_w_train + lo_pi))
        size = beyond & ~absent
        grid = np.full(T.MAXG, np.inf, np.float32)
        grid[:len(th.table.grid)] = th.table.grid
        thr = np.full(T.MAXG, np.inf, np.float32)
        thr[:len(th.table.grid)] = th.table.thresholds
        n = wd.prep.n
        tree = dict(k=jnp.float32(1.0), pull=jnp.ones(n, jnp.float32), odds=odds,
                    odds_max=jnp.float32(odds_max), log_w=jnp.asarray(self.log_w_train, jnp.float32),
                    centre=jnp.asarray(np.asarray(wd.gap.centre), jnp.float32),
                    rho2=jnp.float32(float(wd.gap.rho) ** 2), log_floor=jnp.float32(math.log(th.floor)),
                    grid=jnp.asarray(grid), thresholds=jnp.asarray(thr), n_grid=jnp.int32(len(th.table.grid)),
                    cap=jnp.float32(th.cap), u_logged=jnp.asarray(u_rec, jnp.float32),
                    q1_rec=jnp.asarray(wd.q1_rec))
        past = np.asarray(TC.stalled(a_pi), bool).any(1)
        info = dict(kish_share=CB.CV.kish(w_bank) / len(w_bank), cap_over_R_u=th.cap / wd.R_u,
                    grid_points=int(len(th.table.grid)), odds_max=odds_max, **fit_info,
                    no_credit_share=float(absent.mean()), capped_share=float(size.mean()),
                    past_80_share=float(past.mean()),
                    no_credit_share_past_80=float(absent[past].mean()) if past.any() else None,
                    no_credit_share_within_80=float(absent[~past].mean()),
                    capped_share_within_80=float(size[~past].mean()))
        return tree, info, dict(absent=absent, size=size, th=th, odds=odds, odds_max=odds_max, a_pi=a_pi)


# ------------------------------------------------------------------------------------------------ check (V1, V2)

def checks(quality, seed, log):
    wd = T.World(quality, seed)
    est = Est(wd)
    pi_none = RN.controllers_of(wd.ctx, "room_lower_edge", seed, wd.spec)["le_none"]
    tree, info, x = est.refresh(pi_none, 99)
    # V2: the exact rule's no-credit set (command_le_train_f1.refresh) against the estimated one
    _, _, inf_exact, _ = F1.refresh(wd, pi_none, "strong")
    a_pi = x["a_pi"]
    past = np.asarray(TC.stalled(a_pi), bool).any(1)
    absent = x["absent"]
    agree = float(np.mean(absent == inf_exact))
    sel = past & inf_exact
    v2_past = float(absent[sel].mean()) if sel.any() else None
    # in-graph decisions equal the numpy ones
    _, abs_j, size_j = jax.jit(band_read_est)(tree, jnp.arange(wd.prep.n), jnp.asarray(wd.obs), jnp.asarray(a_pi))
    graph_agree = dict(no_credit=float(np.mean(np.asarray(abs_j) == absent)),
                       capped=float(np.mean(np.asarray(size_j) == x["size"])))
    # V1: one draw per row from the test distribution; truth by one exact step (BCA's target rule)
    rng = np.random.default_rng([W.MASTER_SEED, int(seed), CB.TAG, 3])
    z = np.clip(CB.SPREAD * rng.standard_normal(a_pi.shape), -CB.CLIP, CB.CLIP)
    a_q = np.clip(a_pi.astype(np.float64) + z, -1.0, 1.0).astype(np.float32)
    lo_q = logits(x["odds"], wd.obs, a_q)
    absent_q = np.asarray(wd.gap.contains(a_q.astype(np.float64)), bool) | (lo_q > x["odds_max"])
    R_q, beyond_q = x["th"].read(np.where(absent_q, np.inf, est.log_w_train + lo_q))
    refused_q = absent_q | beyond_q
    U_q = LE.widths_at(wd.band, wd.prep, a_q, LE.as_float32_thresholds(R_q), stage=T.STAGE).astype(np.float64)
    err_q = CB.truth_error(wd.ctx, wd.host, wd.prep, wd.seed, a_q, 92)
    w_law = W.ratio(wd.train.c, CB.TILT)
    q_past = np.asarray(TC.stalled(a_q), bool).any(1)
    v1 = dict(all=CB.outcome_stats(err_q, U_q, refused_q, w_law, np.ones(wd.prep.n, bool)),
              past_80=CB.outcome_stats(err_q, U_q, refused_q, w_law, q_past),
              within_80=CB.outcome_stats(err_q, U_q, refused_q, w_law, ~q_past))
    out = dict(seed=int(seed), refresh=info, law_classifier=est.law_classifier,
               V1=v1, V2=dict(agreement=agree, estimated_no_credit_share=float(absent.mean()),
                              exact_no_credit_share=float(inf_exact.mean()),
                              exact_no_credit_past_80_also_estimated=v2_past,
                              estimated_no_credit_where_exact_certifies=float(absent[~inf_exact].mean())),
               in_graph_agreement=graph_agree)
    log(f"{quality} seed {seed}: V1 covered-or-refused {v1['all']['covered_or_refused']:.3f} (refused "
        f"{v1['all']['refused']:.3f}); V2 agreement {agree:.3f}, past-80% exact no-credit also estimated "
        f"{v2_past}; estimated no-credit {absent.mean():.3f} vs exact {inf_exact.mean():.3f}; in-graph {graph_agree}; "
        f"odds fit loss {info['first_loss']:.3f} -> {info['final_loss']:.3f}")
    d = OUT / quality
    d.mkdir(parents=True, exist_ok=True)
    (d / f"checks_seed{int(seed)}.json").write_text(json.dumps(dict(label=LABEL, **out), indent=1, default=float),
                                                    encoding="utf-8")


# ------------------------------------------------------------------------------------------------ training

def arm_dir(quality, arm, seed):
    return OUT / quality / arm / f"seed{int(seed)}"


def train_arm(wd, est, arm, log):
    cfg = wd.ctx.cfg
    rows, rows_sha = wd.rows()
    clone_params, clone_rec = wd.start()
    state = TrainState.create(apply_fn=LR.ACTOR.apply, params=clone_params, tx=LR.adam(cfg.lr))
    chunks, metrics, t0 = [], [], time.perf_counter()
    match = None
    if arm == "C3_untargeted":
        rec = json.loads((arm_dir(wd.ctx.quality, "A3_estimated", wd.seed) / "record.json").read_text())
        match = [c["no_credit_share"] for c in rec["chunks"]]
    for c in range(rows.shape[0]):
        if arm == "C3_untargeted":
            tree, picked = wd.untargeted_tree(match[c], c)
            info = dict(target_share=match[c], picked_share=picked)
            prog = program(wd.args, "untargeted")
        else:
            tree, info, _ = est.refresh(state.params, c)
            prog = program(wd.args, "band")
        state, aux = prog(wd.host.critic.params, state, wd.prep.train, wd.shared, rows[c], tree)
        aux = np.asarray(aux, np.float64)
        metrics.append(aux.mean(0).tolist())
        chunks.append(dict(chunk=c, **info, W_mean=float(aux[:, 4].mean()), q_mean=float(aux[:, 1].mean())))
        log(f"{wd.ctx.quality} seed {wd.seed} {arm}: chunk {c + 1}/{rows.shape[0]} " +
            ", ".join(f"{k} {v:.3f}" for k, v in info.items() if isinstance(v, float)) +
            f", W mean {chunks[-1]['W_mean']:.4f}")
    jax.block_until_ready(state.params)
    finite = bool(LR._finite_tree(state.params))
    d = arm_dir(wd.ctx.quality, arm, wd.seed)
    d.mkdir(parents=True, exist_ok=True)
    sha = LR.save_tree(d / "actor.msgpack", state.params)
    rec = dict(label=LABEL, exploratory=True, quality=wd.ctx.quality, seed=wd.seed, arm=arm,
               passmarks_sha256=passmarks_ok(), spec_sha256=wd.spec, code_sha256=LI.code_hashes(),
               this_code=this_code(), law_classifier=est.law_classifier,
               settings=dict(spread=CB.SPREAD, clip=CB.CLIP, grid_ratio=CB.GRID_RATIO, floor=CB.FLOOR,
                             odds=dict(steps=ODDS_STEPS, batch=ODDS_BATCH, lr=ODDS_LR, hidden=[128, 128])),
               rows_sha256=rows_sha, clone=clone_rec, chunks=chunks, metric_names=list(LE.METRIC_NAMES),
               chunk_metrics=metrics, finite=finite, actor_sha256=sha, seconds=time.perf_counter() - t0,
               finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (d / "record.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    log(f"{wd.ctx.quality} seed {wd.seed} {arm}: finished in {rec['seconds']:.0f}s (finite {finite})")


def evaluate(quality, seed, log):
    ctx = RN.Context(quality=quality)
    spec = RN.spec_digest(ctx)
    banks, sb = D.load_start_banks(ctx.banks_root)
    data = RN.load_data(ctx, seed, spec)
    prep = D.prepare(data, "gap", ctx.cfg)
    ev = {}
    for arm in ARMS:
        d = arm_dir(quality, arm, seed)
        rec = json.loads((d / "record.json").read_text())
        params = LR.load_tree(d / "actor.msgpack", LR.actor_template(), rec["actor_sha256"])
        ev[arm] = LI.evaluate_actor(params, prep, banks, sb["anchors"])
        log(f"{quality} seed {seed} {arm}: strong J {ev[arm]['strong']['J']:.2f} ({ev[arm]['strong']['points']:.2f} "
            f"points), none J {ev[arm]['none']['J']:.2f}")
    (OUT / quality / f"evaluate_seed{int(seed)}.json").write_text(
        json.dumps(dict(label=LABEL, quality=quality, seed=int(seed), evaluations=ev,
                        start_banks_json_sha256=LI.sha256_file(ctx.banks_root / "start_banks.json")),
                   indent=1, default=float), encoding="utf-8")


def main(argv):
    mode, quality, seed = argv[0], argv[1], int(argv[2])
    log = lambda m: print(time.strftime("%H:%M:%S ") + str(m), flush=True)
    passmarks_ok()
    if mode == "checks":
        checks(quality, seed, log)
    elif mode == "train":
        wd = T.World(quality, seed)
        est = Est(wd)
        for arm in ARMS:
            if (arm_dir(quality, arm, seed) / "record.json").exists():
                log(f"{quality} seed {seed} {arm}: already finished")
                continue
            train_arm(wd, est, arm, log)
    elif mode == "evaluate":
        evaluate(quality, seed, log)
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
