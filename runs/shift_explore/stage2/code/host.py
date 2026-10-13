"""Stage 2, the full host (plan section 5.4): TD3+BC co-trained with BCA's lower edge, critic and actor together,
100,000 updates, with the band on BCA's own schedule.

THE UPDATE (td3_bc_lower_edge_update): td3_bc.td3_bc_update restated term for term (algorithms/td3_bc.py is never
edited) with one change in the actor loss:

    actor loss = -lambda * mean_i[ Q1(s_i, pi(s_i)) - kW_i ] + mean_ij (pi(s_i)_j - a_ij)^2,   lambda on Q1 alone
    kW_i = k_paired [U(s_i, pi(s_i)) - U(s_i, a_i)]_+ + k_unpaired U(s_i, pi(s_i))      (BCA host)
    kW_i = c * 100 * sum_j ([pi_j - hi_j]_+^2 + [lo_j - pi_j]_+^2)                      (range-hinge host)
    kW_i = 0                                                                              (equals td3_bc_update)

The critic loss, target, smoothing noise, policy delay 2 and soft target updates are td3_bc's, unchanged; the
cloning pull stays at weight 1 on every row (no dose).

BCA'S SCHEDULE (td3_bc_bca.update and learners.bca_cotrain, restated around the new consumer): every update first
runs td3_bc_bca.fit_scale on the batch (BCA's scale fit, unchanged); the width is read from the FROZEN reference
(td3_bc_bca.refresh: first after update 10,000, then every 5,000 updates, on the calibration bank, refresh keys as
the gate's bca_cotrain); before the first refresh the width is 0 (BCA's own rule), so the run starts as plain
TD3+BC. At every refresh the extras are recomputed eagerly: R_c and U_max (uniform), or the classifier with the
current actor and the tabulated threshold (weighted). Initial parameters and the stream of batch rows and target
noise are td3_bc.initialize's for the seed, the same as plain TD3+BC's (the gate's shared-data rule).

ARMS per seed: stalling world: the BCA host (k = 1, the detection test's version), plain TD3+BC (the gate's saved
run for seeds 0-4, the fresh run for seeds 5-9), the range-hinge host matched blind to the BCA host's strength
(strength = RMS change of the final actor's command from plain TD3+BC's on the 5,000 evaluation-split states;
ladder c = 0.1, 0.3, 1, 3, 10, then glq_match). No-stall world: the BCA host, plain TD3+BC (trained here: the gate
trained none there) and the range-hinge host at the stalling world's locked c. Lock before returns, as everywhere.

Every phase function requires a verified approval.
"""

import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_common as C  # noqa: E402
import band as B  # noqa: E402
import weights as Wt  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import algorithms.td3_bc as BASE  # noqa: E402
import algorithms.td3_bc_bca as P  # noqa: E402
from calibration.reference import reference_valid  # noqa: E402

HINGE_SCALE = 100.0
HINGE_LADDER = (0.1, 0.3, 1.0, 3.0, 10.0)
HINGE_EXTEND = 10.0 ** 0.5
TAG_REFRESH = 39             # learners.TAG_REFRESH (the gate's refresh seed tag)


# --------------------------------------------------------------------------------------------- the update

def td3_bc_lower_edge_update(args, actor_apply_fn, critic_apply_fn, agent_state, batch, it, rng_noise, max_action,
                             penalty_fn=None):
    """td3_bc.td3_bc_update (unweighted path) with kW = penalty_fn(pi) per row inside the actor's objective.
    penalty_fn=None is td3_bc_update term for term."""
    noise = jax.random.normal(rng_noise, batch.action.shape) * args.policy_noise
    noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
    next_action = actor_apply_fn(agent_state.actor_target.params, batch.next_obs)
    next_action = jnp.clip(next_action + noise, -max_action, max_action)
    target_q = critic_apply_fn(agent_state.critic_target.params, batch.next_obs, next_action)
    target_q = jnp.min(target_q, axis=-1)
    target = batch.reward + (1.0 - batch.done) * args.discount * target_q
    target = jax.lax.stop_gradient(target)

    def _critic_loss_fn(params):
        q = critic_apply_fn(params, batch.obs, batch.action)
        return jnp.square(q - target[:, None]).mean(axis=0).sum()

    critic_loss, critic_grad = jax.value_and_grad(_critic_loss_fn)(agent_state.critic.params)
    agent_state = agent_state._replace(critic=agent_state.critic.apply_gradients(grads=critic_grad))

    def _actor_loss_fn(params, critic_params):
        pi = actor_apply_fn(params, batch.obs)
        q = critic_apply_fn(critic_params, batch.obs, pi)[..., 0]
        lmbda = args.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
        bc = jnp.square(pi - batch.action).mean()
        if penalty_fn is None:
            return (-lmbda * q.mean() + bc, (q.mean(), lmbda, bc, jnp.float32(0.0)))
        kw = penalty_fn(pi)
        return (-lmbda * (q - kw).mean() + bc, (q.mean(), lmbda, bc, kw.mean()))

    def _do_actor_update(state):
        grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
        (a_loss, aux), a_grad = grad_fn(state.actor.params, state.critic.params)
        state = state._replace(actor=state.actor.apply_gradients(grads=a_grad))

        def _soft(src, tgt):
            return tgt.replace(step=tgt.step + 1, params=optax.incremental_update(src.params, tgt.params, args.tau))

        state = state._replace(actor_target=_soft(state.actor, state.actor_target),
                               critic_target=_soft(state.critic, state.critic_target))
        return (state, (a_loss, *aux))

    def _skip_actor_update(state):
        zero = jnp.float32(0.0)
        return (state, (zero, zero, zero, zero, zero))

    agent_state, (actor_loss, q_mean, lmbda, bc_loss, kw_mean) = jax.lax.cond(
        it % args.policy_freq == 0, _do_actor_update, _skip_actor_update, agent_state)
    return agent_state, {"critic_loss": critic_loss, "actor_loss": actor_loss, "q_mean": q_mean, "lambda": lmbda,
                         "bc_loss": bc_loss, "kW_mean": kw_mean}


def hinge(pi, lo, hi):
    return HINGE_SCALE * (jnp.square(jnp.maximum(pi - hi, 0.0)) + jnp.square(jnp.maximum(lo - pi, 0.0))).sum(axis=-1)


class Extras(NamedTuple):
    """What the frozen reference does not hold, recomputed at every refresh (eagerly)."""

    R_c: Any
    U_max: Any
    log_t: Any
    R_tab: Any
    l_star: Any
    cls_params: Any


def initial_extras(weighted, cls_template=None):
    z = jnp.zeros(B.GRID, jnp.float32)
    return Extras(B.f32(0.0), B.f32(0.0), z, z, B.f32(1e30), cls_template if weighted else None)


def band_of(state, extras):
    ref = state.posterior
    return B.Band(ref.cal_params, ref.residual_scale, ref.threshold, extras.R_c, extras.U_max,
                  jnp.isfinite(ref.threshold), ref.ready, extras.log_t, extras.R_tab, extras.l_star, extras.cls_params)


def bca_host_update(args, config, models, state, extras, batch, it, rng, coef, weighted):
    """td3_bc_bca.update with the dose replaced by the lower edge: BCA's scale fit on the batch (unchanged), then the
    td3_bc update whose actor climbs Q1 - kW with U read from the frozen reference at the actor's command."""
    fitted, diag = P.fit_scale(args, config, models, state, batch, rng, 1.0)
    band = band_of(state, extras)
    cal = models[2]
    u_rec = B.width(cal, band, batch.obs, batch.action, weighted)

    def penalty_fn(pi):
        # gradient paths: eta_f's command input (both versions); weighted also the frozen classifier's command input,
        # in R_c(w(s, pi)) and in the refusal term (declared in the pre-registration); no parameter of eta_f or of
        # the classifier gets a gradient
        u = B.width(cal, band, batch.obs, pi, weighted)
        return coef[0] * B.paired(u, u_rec) + coef[1] * u

    native, metrics = td3_bc_lower_edge_update(args, models[0].apply, models[1].apply, state.native, batch, it, rng,
                                               1.0, penalty_fn)
    valid = P.tree_finite((batch, state.native, native, metrics)) & diag["scale_inputs_valid"] \
        & reference_valid(state.posterior) & jnp.all(jnp.isfinite(u_rec))
    metrics.update(posterior_ready=state.posterior.ready, width_recorded_mean=jnp.mean(u_rec))
    result = jax.lax.cond(valid, lambda _: fitted._replace(native=native), lambda _: state, None)
    metrics.update(diag, inputs_valid=valid)
    return result, metrics


def make_bca_step(args, config, models, dataset, coef, weighted):
    """td3_bc_bca.make_train_step's layout (rng -> kr, kb, kn), calling bca_host_update."""
    def step(carry, _):
        rng, state, extras, it = carry
        kr, kb, kn = jax.random.split(rng, 3)
        idx = jax.random.randint(kb, (args.batch_size,), 0, len(dataset.obs))
        batch = jax.tree_util.tree_map(lambda x: x[idx], dataset)
        result, metrics = bca_host_update(args, config, models, state, extras, batch, it + 1, kn, coef, weighted)
        proposed = (kr, result, extras, it + 1)
        accepted = jax.lax.cond(metrics["inputs_valid"], lambda _: proposed, lambda _: carry, None)
        return accepted, metrics
    return step


def make_hinge_step(args, dataset, lo, hi, c):
    """td3_bc.make_train_step's layout (rng -> rng, rng_batch, rng_noise) with the range hinge (c = 0: plain)."""
    def step(carry, _):
        rng, state, it = carry
        it = it + 1
        rng, rng_batch, rng_noise = jax.random.split(rng, 3)
        idx = jax.random.randint(rng_batch, (args.batch_size,), 0, dataset.obs.shape[0])
        batch = jax.tree_util.tree_map(lambda x: x[idx], dataset)
        state, loss = td3_bc_lower_edge_update(args, _actor_apply(), _critic_apply(), state, batch, it, rng_noise,
                                               1.0, lambda pi: c * hinge(pi, lo, hi))
        return (rng, state, it), loss
    return step


def _actor_apply():
    import experiments.shift.learners as L
    return L.ACTOR.apply


def _critic_apply():
    import experiments.shift.learners as L
    return L.CRITIC.apply


# --------------------------------------------------------------------------------------------- extras at a refresh

def refresh_extras(args, config, models, state, bank, key, weighted, rep, actor_apply=None, train=None,
                   cloned=None, cls_p0=None, cls_rows=None):
    """Eager, at a refresh, after td3_bc_bca.refresh has frozen the reference with `key`: R_c and U_max (uniform);
    for the weighted version the classifier fitted with the current actor, the bank's scores with the frozen
    reference, the threshold tabulated on the bank's threshold half and U_max there."""
    ref = state.posterior
    cal = models[2]
    if not bool(ref.ready):
        return None, dict(ready=False)
    try:
        ub = B.band_from_reference(ref, cal, np.asarray(bank.obs), np.asarray(bank.action))
    except ValueError as error:
        return None, dict(ready=True, error=str(error))
    if not weighted:
        ex = Extras(ub.R_c, ub.U_max, jnp.zeros(B.GRID, jnp.float32), jnp.zeros(B.GRID, jnp.float32), B.f32(1e30),
                    None)
        return ex, dict(ready=True, R=float(ref.threshold), R_c=float(ub.R_c), U_max=float(ub.U_max))
    scores, _ = B.bank_scores(args, models, state, bank, key, ref.cal_params, ref.residual_scale)
    A_idx, _ = B.split_bank(len(scores), rep)
    obs = np.asarray(train.obs)
    cls_params, cdiag = Wt.fit_for_actor(actor_apply, state.native.actor.params, obs, np.asarray(train.action),
                                         cloned, cls_rows, cls_p0)
    lw = np.asarray(Wt.logit(cls_params, jnp.asarray(bank.obs), jnp.asarray(bank.action)), np.float64)
    log_t, R_raw, R_c_tab, l_star = B.tabulate_threshold(scores[A_idx], lw[A_idx], rep, config)
    wb = B.weighted_band(ub, log_t, R_c_tab, l_star, cls_params, cal, np.asarray(bank.obs)[A_idx],
                         np.asarray(bank.action)[A_idx], lw[A_idx])
    ex = Extras(wb.R_c, wb.U_max, wb.log_t, wb.R_tab, wb.l_star, cls_params)
    return ex, dict(ready=True, R=float(ref.threshold), l_star=l_star, U_max=float(wb.U_max), classifier=cdiag)


# --------------------------------------------------------------------------------------------- one run

def bca_host_run(prepared, settings, rep, approval, *, coef=(1.0, 0.0), weighted=False, cloned=None, updates=None,
                 log_fn=C.log):
    """One co-trained BCA host (lower-edge consumer). Returns (record, final td3_bc_bca State). Refuses without
    approval. cloned: the cloned actor's commands at the training rows (weighted version only)."""
    C.recheck(approval)
    import experiments.shift.learners as L
    t0 = time.perf_counter()
    args = L.make_args(settings, rep, 5.0)
    config = B.bca_config()
    rng, state, models = L.bca_initial_state(args, config)   # construction only (td3_bc.initialize + calibrator)
    total = int(settings.updates if updates is None else updates)
    cls_p0, cls_rows = (Wt.cls_init(rep, L.DX, L.DA)[0], Wt.cls_rows(rep, prepared.n)) if weighted else (None, None)
    extras = initial_extras(weighted, cls_p0)
    step = make_bca_step(args, config, models, prepared.train, jnp.asarray(coef, jnp.float32), weighted)
    refreshes = L.refresh_steps(settings, total)
    snaps = L.snapshot_steps(settings, total)
    refresh_seed = int(L.rng_np(settings, rep, TAG_REFRESH).integers(2 ** 31 - 1))
    carry = (rng, state, extras, jnp.int32(0))
    programs, chunks, refresh_log, done = {}, [], [], 0
    record = dict(kind="stage2-bca-host", rep=int(rep), coef=list(coef), weighted=bool(weighted), updates=total,
                  args=asdict(args), bca=B.SETTINGS, data=prepared.describe(),
                  prereg_sha256=approval.prereg_sha256, approval_sha256=approval.file_sha256,
                  initial_sha256=L.tree_sha256((state.native.actor.params, state.native.critic.params)),
                  batch_stream_sha256=L.batch_stream_sha256(rng, prepared.n, args.batch_size), refresh_seed=refresh_seed,
                  failed=False)
    for target in L._events(total, settings.chunk, snaps, refreshes):
        length = target - done
        if length not in programs:
            programs[length] = jax.jit(lambda c, length=length: jax.lax.scan(step, c, None, length=length))
        carry, metrics = programs[length](carry)
        done = target
        try:
            P.require_valid(metrics)
        except FloatingPointError as error:
            record.update(failed=True, failed_at=target, failure=str(error))
            break
        chunks.append(dict(L._chunk_summary(metrics), update=target))
        if target in refreshes:
            key = jax.random.fold_in(jax.random.PRNGKey(refresh_seed), target)
            new, diag, _ = P.refresh(args, config, models, carry[1], prepared.bank, key, 1.0,
                                     heldout_ids=np.arange(len(prepared.bank.obs)))
            ex, ediag = refresh_extras(args, config, models, new, prepared.bank, key, weighted, rep,
                                       actor_apply=L.ACTOR.apply, train=prepared.train, cloned=cloned,
                                       cls_p0=cls_p0, cls_rows=cls_rows)
            if ex is None:
                ex = carry[2]
            carry = (carry[0], new, ex, carry[3])
            refresh_log.append(dict(step=target, **{k: bool(v) for k, v in diag.items()}, extras=ediag))
        if log_fn:
            log_fn(f"BCA host seed {rep}: update {target}/{total}")
    record.update(chunks=chunks, refreshes=refresh_log, final_update=done, seconds=time.perf_counter() - t0)
    return record, carry[1]


def hinge_host_run(prepared, settings, rep, c, *, updates=None, log_fn=C.log):
    """One co-trained TD3+BC with the range hinge at coefficient c (no BCA; c = 0 is plain TD3+BC). Same initial
    parameters and stream as plain TD3+BC. Returns (record, final AgentTrainState)."""
    import experiments.shift.learners as L
    H = C.load_room_harness()
    t0 = time.perf_counter()
    args = L.make_args(settings, rep, 5.0)
    total = int(settings.updates if updates is None else updates)
    rng, state, _ = BASE.initialize(args, L.DX, L.DA, 1.0)
    lo, hi = H.hinge_bounds(prepared.actions)
    step = make_hinge_step(args, prepared.train, jnp.asarray(lo, jnp.float32), jnp.asarray(hi, jnp.float32),
                           jnp.float32(c))
    runner = (rng, state, jnp.int32(0))
    programs, done, chunks, failed = {}, 0, [], False
    while done < total:
        length = int(min(settings.chunk, total - done))
        if length not in programs:
            programs[length] = jax.jit(lambda r, length=length: jax.lax.scan(step, r, None, length=length))
        runner, metrics = programs[length](runner)
        done += length
        chunks.append(dict(L._chunk_summary(metrics), update=done))
        if not L._finite_tree(metrics) or not L._finite_tree(runner[1].actor.params):
            failed = True
            break
    return dict(kind="stage2-hinge-host", rep=int(rep), c=float(c), updates=total, failed=failed, chunks=chunks,
                recorded_range=dict(lo=lo.tolist(), hi=hi.tolist()), seconds=time.perf_counter() - t0), runner[1]


# --------------------------------------------------------------------------------------------- phases

def host_dir(rep, world, paths=None):
    return (paths or C.PATHS).out / "host" / world / f"seed{int(rep)}"


def _strength(params, ref_params, eval_obs):
    import experiments.shift.learners as L
    return float(L.strength(params, ref_params, eval_obs))


def train(rep, world, approval, version, locked_c=None, log_fn=C.log, paths=None):
    """One seed and world of the full-host comparison, up to the lock (no return). Stalling world: the BCA host, plain
    TD3+BC (the gate's / the fresh run), the range-hinge ladder and blind matching to the BCA host's strength (both
    against plain TD3+BC's final actor), and a check that the hinge program with c = 0 reproduces plain TD3+BC for
    the first 2,000 updates (within 1e-5). No-stall world: the BCA host, plain TD3+BC trained here with the gate's
    learners.cotrain, the hinge host at the stalling world's locked c."""
    C.recheck(approval)
    import experiments.shift.learners as L
    from experiments.signal import glq_match as M
    from fresh import context
    t0 = time.perf_counter()
    d = host_dir(rep, world, paths)
    if (d / "lock.json").exists():
        raise SystemExit(f"refused: {d} is already locked")
    weighted = version == "weighted"
    ctx = context(rep, worlds=("stall", "no_stall") if world == "no_stall" else ("stall",))
    prep = ctx.prepared(world)
    learn = ctx.learn
    cloned = L.actor_rows(ctx.clone(world)[0], np.asarray(prep.train.obs)) if weighted else None
    rec_b, st_b = bca_host_run(prep, learn, rep, approval, coef=(1.0, 0.0), weighted=weighted, cloned=cloned,
                               log_fn=log_fn)
    actors = {"bca_host": st_b.native.actor.params}
    records = {"bca_host": rec_b}
    if world == "stall":
        plain = ctx.host("stall")[0]
        rec_p = dict(kind="plain TD3+BC", source="the gate's saved run (seeds 0-4) or the fresh run (seeds 5-9)")
    else:
        rec_p, plain, _ = L.cotrain(prep, learn, rep, alpha=5.0, log=log_fn, name=f"plain TD3+BC no-stall seed {rep}")
        rec_p = {k: v for k, v in rec_p.items() if k != "chunks"}
    actors["plain"] = plain.actor.params
    records["plain"] = rec_p
    check = None
    if world == "stall":
        r0, s0 = hinge_host_run(prep, learn, rep, 0.0, updates=2000)
        r1, s1, _ = L.cotrain(prep, learn, rep, alpha=5.0, updates=2000)
        diff = max(float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))
                   for a, b in zip(jax.tree_util.tree_leaves(s0.actor.params), jax.tree_util.tree_leaves(s1.actor.params)))
        check = dict(updates=2000, max_abs_diff=diff, passed=bool(diff <= 1e-5))
    target = _strength(actors["bca_host"], actors["plain"], prep.eval_obs) if not rec_b["failed"] else float("nan")
    hinge_arms, match = {}, None
    if world == "stall":
        S_of = {}
        for c in HINGE_LADDER:
            r, s = hinge_host_run(prep, learn, rep, c, log_fn=log_fn)
            hinge_arms[c] = (r, s.actor.params)
            S_of[c] = _strength(s.actor.params, actors["plain"], prep.eval_obs) if not r["failed"] else float("nan")
        if np.isfinite(target) and target > 0:
            for _ in range(3):
                fin = [v for v in S_of.values() if np.isfinite(v)]
                if fin and target > max(fin):
                    c = max(S_of) * HINGE_EXTEND
                    r, s = hinge_host_run(prep, learn, rep, c, log_fn=log_fn)
                    hinge_arms[c] = (r, s.actor.params)
                    S_of[c] = _strength(s.actor.params, actors["plain"], prep.eval_obs) if not r["failed"] else \
                        float("nan")

            def evaluate(proposals):
                out = []
                for _, c in proposals:
                    c = float(c)
                    if c not in hinge_arms:
                        r, s = hinge_host_run(prep, learn, rep, c, log_fn=log_fn)
                        hinge_arms[c] = (r, s.actor.params)
                        S_of[c] = _strength(s.actor.params, actors["plain"], prep.eval_obs) if not r["failed"] else \
                            float("nan")
                    out.append(S_of[c])
                return out

            ks = sorted(S_of)
            states, rounds = M.match_parallel({"hinge": ([0.0] + ks, [0.0] + [S_of[k] for k in ks])},
                                              {"bca_host": target}, evaluate, M.MATCH_TOL, M.MATCH_ROUNDS, log=log_fn)
            match = M.public(states[("hinge", "bca_host")])
            match.update(rounds=rounds, glq_match_sha256=M.source_sha256())
        for c, (r, p) in hinge_arms.items():
            actors[f"hinge@{c:.10g}"] = p
            records[f"hinge@{c:.10g}"] = {k: v for k, v in r.items() if k != "chunks"}
        matched = None if match is None or match.get("knob") is None else f"hinge@{float(match['knob']):.10g}"
    else:
        matched = None
        if locked_c is not None:
            r, s = hinge_host_run(prep, learn, rep, locked_c, log_fn=log_fn)
            actors["hinge_locked"] = s.actor.params
            records["hinge_locked"] = {k: v for k, v in r.items() if k != "chunks"}
    actors["clone"] = ctx.clone(world)[0]
    finite = {k: L._finite_tree(v) for k, v in actors.items()}
    scored = ["bca_host", "plain", "clone"] + ([matched] if matched else []) + \
        (["hinge_locked"] if "hinge_locked" in actors else [])
    names = sorted(actors)
    file_sha = C.save_tree(d / "actors.msgpack", {k: actors[k] for k in names}, paths)
    C.write_json(d / "lock.json", dict(
        label=C.LABEL_LONG, kind="stage2-host-lock", rep=int(rep), world=world, version=version, role=ctx.role,
        prereg_sha256=approval.prereg_sha256, approval_sha256=approval.file_sha256, records=records,
        check_hinge_zero_equals_plain=check, target_strength=target, hinge_match=match, matched_hinge=matched,
        locked_c=locked_c, finite=finite, scored_names=scored, names=names,
        actor_sha256={k: L.tree_sha256(actors[k]) for k in names}, actors_file="actors.msgpack",
        actors_file_sha256=file_sha, seconds=time.perf_counter() - t0,
        about="the lock: written before any return of any host arm is computed"), paths)


def evaluate(rep, world, approval, log_fn=C.log, paths=None):
    """Verify the host lock, then exact returns of its scored actors -> values.json, once."""
    C.recheck(approval)
    import experiments.shift.acceptance as A
    import experiments.shift.learners as L
    from fresh import context
    d = host_dir(rep, world, paths)
    if (d / "values.json").exists():
        raise SystemExit(f"refused: {d}/values.json exists")
    lock = C.read_json(d / "lock.json")
    if lock.get("prereg_sha256") != approval.prereg_sha256:
        raise SystemExit("refused: the host lock was written under another pre-registration")
    acts = C.load_tree(d / lock["actors_file"], {k: L.actor_template() for k in lock["names"]},
                       lock["actors_file_sha256"])
    if any(L.tree_sha256(acts[k]) != lock["actor_sha256"][k] for k in lock["names"]):
        raise SystemExit("refused: host actors differ from the lock")
    ctx = context(rep, worlds=("stall", "no_stall"))
    prep = ctx.prepared(world)
    an = ctx.anchors(world)
    ev = {n: A._evaluate(acts[n], prep, an, ctx.cfg, failed=not lock["finite"][n]) for n in lock["scored_names"]}
    C.write_json(d / "values.json", dict(label=C.LABEL_LONG, kind="stage2-host-values", rep=int(rep), world=world,
                                         prereg_sha256=approval.prereg_sha256,
                                         lock_sha256=C.file_sha256(d / "lock.json"), evaluations=ev,
                                         points={n: (None if ev[n] is None else {s: ev[n][s]["points"] for s in ev[n]})
                                                 for n in ev}), paths)
