"""Near-80 (version 9): the learner phases. Each calls the gate's own learner code, unchanged, on the main
setting's prepared rows; only what the build spec (section 5.6) lists is restated.

  critic            learners.fixed_critic(prepared, 'stall_aware', learn, seed) (the gate's call in
                    acceptance.get_critic): 100,000 updates, then frozen.
  critic_full       the same on learners.prepare(log, removals['none'], the gate's normalization) of the stalling world,
                    on every seed of the plan.
  critic_second     step 1's construction (restructure/step1/train_critics.py): step 1's gap rule on the setting's kept
                    commands, the survivors removed as well, the same data checks, learners.fixed_critic. Restated
                    because step 1's script refuses other seeds and checks its design's expected counts. A gap rule
                    that is not defined is not a stop: the record says defined: false and the switch is off.
  switch            step 1's scoring (restructure/step1/evaluate.py) on BCA's two POOLED groups (version 9, section 3):
                    one-sided overestimates, under the second critic, of the survivors and of every held-out recorded
                    row and top-up row outside the ball (the whole calibration split, the whole evaluation split and
                    the data record's top-up blocks, regenerated and checked, split at t_lo; near80lib.pooled_groups);
                    BCA's certificate (step1lib.certificate, margin 0.10, the same rng tags), the rank test, the group
                    summaries, the textbook split-conformal quantiles and the mean difference, which every switch arm
                    reads. Also recorded (reported only, no arm reads them): the groups by source and their distinct
                    episodes, step 2's uncorrected split-conformal gap, the exact Wilks twin, the one-row-per-episode
                    variant of every switch (one row per episode chosen uniformly, default_rng([20261020, 9901, seed,
                    world, 1 or 2])) and the bank-only groups of version 8. The arm table row is computed later,
                    inside the policy phase (arms_v9.arm_table).
  clone             learners.clone(prepared, learn, seed).
  reference_actors  learners.ActorOnly(prepared, clone, learn, seed, alpha=5).run(arms, critics, weighted=False), with
                    the gate's check-2 layout so that the reproduction check can be bit for bit: stalling world
                    [none: setting critic, full_none: full-log critic, pad: setting critic]; no-stall world [none,
                    pad]. The pads are discarded.
A learner that fails (non-finite) is a recorded stop (phases.PhaseStop).
"""
import sys

sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import near80_constants as K  # noqa: E402
import ledger as LG  # noqa: E402
import phases as PH  # noqa: E402
import settingdata as SD  # noqa: E402
import near80lib as N  # noqa: E402
from experiments.shift import learners as L  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402

_STEP1 = {}


def step1lib():
    if "lib" not in _STEP1:
        _STEP1["lib"] = N.load_step1lib()
    return _STEP1["lib"]


def save_tree_once(path, tree):
    """Flax msgpack of a pytree (the bytes learners.save_tree writes), created exclusively; returns its SHA-256."""
    from flax import serialization
    return LG.write_bytes_once(path, serialization.to_bytes(tree))


def _trim(result):
    return {k: v for k, v in (result or {}).items() if k not in ("chunks",)}


def _learner_dir(run):
    k = run.key
    return SD.learner_dir(run.out_root, k.setting, k.world, k.seed)


# --------------------------------------------------------------------------------------------- critics

def train_critic(prep, learn, seed, updates, log):
    result, state = L.fixed_critic(prep, K.CONTROLLER, learn, seed, updates=updates, log=log)
    if result.get("failed"):
        raise PH.PhaseStop(PH.STOP_REASONS["learner"], dict(what="the critic", result=_trim(result)))
    return result, state.critic.params


def critic(run, ctx):
    k = run.key
    pg = ctx.prepared(k.world)
    result, params = train_critic(pg, ctx.learn, k.seed, run.updates, run.log)
    sha = save_tree_once(_learner_dir(run) / "critic.msgpack", params)
    return dict(kind="near80v9-critic", setting=k.setting, world=k.world, seed=k.seed, controller=K.CONTROLLER,
                result=_trim(result), updates=result.get("final_update", result.get("updates")), file_sha256=sha,
                critic_sha256=L.tree_sha256(params), failed=False, data=pg.describe())


def critic_full(run, ctx):
    k = run.key
    sd = ctx.seed("stall")
    prep = L.prepare(sd.log, sd.removals["none"], sd.norm(), ctx.learn)
    result, params = train_critic(prep, ctx.learn, k.seed, run.updates, run.log)
    d = Path(run.out_root) / Path(K.LAYOUT["critic_full"].format(seed=k.seed)).parent
    sha = save_tree_once(d / "critic.msgpack", params)
    return dict(kind="near80v9-critic-full", setting="full", world="stall", seed=k.seed, controller=K.CONTROLLER,
                result=_trim(result), updates=result.get("final_update", result.get("updates")), file_sha256=sha,
                critic_sha256=L.tree_sha256(params), failed=False, data=prep.describe())


def gap_rule_of(pg):
    S1 = step1lib()
    return S1.gap_rule(S1.throttle(pg.actions), **K.GAP_RULE)


def second_removal(setting, sd, pg, gr):
    """The setting's removal with the survivors (the gap rule's full-throttle rows among its kept training rows)
    removed as well, the bank unchanged. Returns (shift.Removal, survivors as log rows)."""
    surv = np.asarray(pg.rows[gr["tail"]], np.int64)
    rm = SD.removal(setting, sd)
    removed = rm.removed.copy()
    if removed[surv].any():
        raise PH.PhaseStop(PH.STOP_REASONS["check"], dict(what="a survivor is already removed"))
    removed[surv] = True
    return SH.Removal(rm.kind + "_minus_survivors", removed, rm.bank_removed.copy(), sd.gap), surv


def second_checks(prep, pg, surv, sd=None):
    """Step 1's training-data checks of the second critic, and version 9's: no held-out or top-up row in its training
    rows (top-up rows are never log rows, so the log's held-out rows decide it)."""
    extra = {}
    if sd is not None:
        h = N.heldout_rows(sd.log, sd.gap, K.HELDOUT_SPLITS)
        extra["no_heldout_or_topup_row_in_training"] = bool(len(np.intersect1d(h, prep.rows)) == 0
                                                            and int(np.max(prep.rows, initial=-1)) < sd.log.n_rows)
    return dict(
        extra,
        rows=bool(prep.n == pg.n - len(surv)),
        no_survivor_in_training=bool(not np.isin(surv, prep.rows).any()),
        bank_unchanged=bool(np.array_equal(prep.bank_rows, pg.bank_rows)),
        normalisation_unchanged=bool(np.array_equal(prep.obs_mean, pg.obs_mean)
                                     and np.array_equal(prep.obs_std, pg.obs_std)),
        eval_rows_unchanged=bool(np.array_equal(prep.eval_rows, pg.eval_rows)),
        training_rows_are_kept_minus_survivors=bool(np.array_equal(prep.rows, np.setdiff1d(pg.rows, surv))))


def critic_second(run, ctx):
    k = run.key
    S1 = step1lib()
    pg = ctx.prepared(k.world)
    gr = gap_rule_of(pg)
    rec = dict(kind="near80v9-critic-second", setting=k.setting, world=k.world, seed=k.seed, controller=K.CONTROLLER,
               gap_rule=dict(defined=bool(gr["defined"]), t_lo=gr["t_lo"], t_up=gr["t_up"], n_tail=gr["n_tail"],
                             gap=gr["gap"], reason=gr.get("reason")), defined=bool(gr["defined"]))
    if not gr["defined"]:
        rec.update(file_sha256=None, failed=False, note="the gap rule is not defined: no second critic, the switch "
                                                        "is off")
        return rec
    sd = ctx.seed(k.world)
    rm2, surv = second_removal(k.setting, sd, pg, gr)
    prep = L.prepare(sd.log, rm2, sd.norm(), ctx.learn)
    checks = second_checks(prep, pg, surv, sd)
    if not all(checks.values()):
        raise PH.PhaseStop(PH.STOP_REASONS["check"], dict(what="the second critic's training-data checks",
                                                          checks=checks), checks)
    result, params = train_critic(prep, ctx.learn, k.seed, run.updates, run.log)
    sha = save_tree_once(_learner_dir(run) / "critic_second.msgpack", params)
    rec.update(survivors=surv.tolist(), survivors_sha256=S1.array_sha256(surv), n_survivors=int(len(surv)),
               training_rows_sha256=S1.array_sha256(prep.rows), checks=checks, result=_trim(result),
               file_sha256=sha, critic_sha256=L.tree_sha256(params), failed=False, data=prep.describe())
    return rec


# --------------------------------------------------------------------------------------------- the switch

def overestimates(critic, tr, fixed, gamma, rows=None, noise=None):
    """One-sided overestimates at the transitions tr (runtime Transition, normalised) [rows]: noiseless c(o') or, with
    noise (draws, n, 4), the mean over draws of min_h Q(o', clip(c(o') + noise, -1, 1))."""
    import jax.numpy as jnp
    S = step1lib()
    obs, act, rew, nxt, done = (np.asarray(x) for x in (tr.obs, tr.action, tr.reward, tr.next_obs, tr.done))
    if rows is not None:
        obs, act, rew, nxt, done = obs[rows], act[rows], rew[rows], nxt[rows], done[rows]
    q = L.critic_rows(critic, obs, act)[0].min(1)
    c = np.asarray(fixed.apply(fixed.params(), jnp.asarray(nxt, jnp.float32)), np.float32)
    if noise is None:
        qn = L.critic_rows(critic, nxt, c)[0].min(1)
    else:
        qn = np.zeros(len(obs))
        for k in range(len(noise)):
            qn += L.critic_rows(critic, nxt, np.clip(c + noise[k], -1.0, 1.0).astype(np.float32))[0].min(1)
        qn /= len(noise)
    return S.overestimate(q, rew.astype(np.float64), done.astype(np.float64), qn, gamma)


def undefined_entry(gr):
    """The arm-table entry of a world whose gap rule is not defined: the switch is off, nothing is certified."""
    nan = "nan"
    return dict(t_lo=gr.get("t_lo"), t_up=gr.get("t_up"),
                certificate=dict(on=False, height=nan, q05=None, R_tail=nan, reason="the gap rule is not defined"),
                groups=dict(n_full_throttle=0, n_rest=0, mean_full_throttle=nan, mean_rest=nan, q90_full_throttle=nan,
                            q90_rest=nan),
                rank_test=dict(U=None, p=nan),
                split_conformal=dict(on=False, gap=nan, height=0.0, q_tail=nan, q_rest=nan),
                mean_difference=dict(on=False, gap=nan))


def bank_switch_scores(critic2, pg, gr, gamma):
    """(tail, rest) of version 8 and step 1 (the survivors and the bank rows past t_lo; the bank rows at or below
    t_lo): the reproduction check ties this code to step 1's saved certificate with it, and the switch record reports
    it beside the pooled groups. No arm reads it."""
    S1 = step1lib()
    surv_idx = np.flatnonzero(gr["tail"])
    fixed = L.FixedController(K.CONTROLLER, pg.physics, pg.obs_mean, pg.obs_std)
    bank_t = S1.throttle(np.asarray(pg.bank.action))
    bank_ft = S1.full_throttle(bank_t, gr["t_lo"])
    s_tr = overestimates(critic2, pg.train, fixed, gamma, rows=surv_idx)
    s_bk = overestimates(critic2, pg.bank, fixed, gamma)
    return np.concatenate([s_tr, s_bk[bank_ft]]), s_bk[~bank_ft]


def pooled_rows(lg, gap, pg, gr):
    """(tail rows, rest rows, survivors): BCA's two pooled groups as rows of the groups' view lg (the log, or the log
    and its top-up blocks; near80lib.pooled_groups): the survivors, then every held-out and top-up row outside the
    ball past t_lo; every held-out and top-up row outside the ball at or below t_lo."""
    surv = np.asarray(pg.rows[gr["tail"]], np.int64)
    tail, rest = N.pooled_groups(lg, gap, surv, gr["t_lo"], K.HELDOUT_SPLITS)
    return tail, rest, surv


def switch_scores(critic2, lg, gap, pg, gr, gamma):
    """(tail, rest, rows): the second critic's one-sided overestimates of BCA's two pooled groups (pooled_rows), all
    held out from the second critic, scored with the same overestimates, the same stall-aware controller and the
    gate's normalization; rows = (tail rows, rest rows, survivors). The survivors are scored from the prepared training
    transitions (as step 1 and version 8 did); every other row from learners' own transition builder on the view."""
    tail_rows, rest_rows, surv = pooled_rows(lg, gap, pg, gr)
    fixed = L.FixedController(K.CONTROLLER, pg.physics, pg.obs_mean, pg.obs_std)
    surv_idx = np.flatnonzero(gr["tail"])
    s_sv = overestimates(critic2, pg.train, fixed, gamma, rows=surv_idx)
    held_tail = tail_rows[len(surv):]
    tr_t = L._transition(lg, held_tail, pg.obs_mean, pg.obs_std)
    tr_r = L._transition(lg, rest_rows, pg.obs_mean, pg.obs_std)
    s_t = overestimates(critic2, tr_t, fixed, gamma) if len(held_tail) else np.zeros(0)
    s_r = overestimates(critic2, tr_r, fixed, gamma) if len(rest_rows) else np.zeros(0)
    return np.concatenate([s_sv, s_t]), np.asarray(s_r, np.float64), (tail_rows, rest_rows, surv)


def certificate_entry(tail, rest, seed, world, gr, plain=True):
    """Step 1's certificate, rank test and groups for one seed-world, as step 1's results record them (non-finite
    numbers as the strings 'inf', '-inf', 'nan', which step2lib.num reads); with plain, version 9's plain switches on
    the same two groups: the textbook split conformal (with step 2's uncorrected gap) and the mean difference."""
    S1 = step1lib()
    tags = [*K.CERT_TAG_HEAD, int(seed), K.WORLD_TAG[world]]
    tail, rest = np.asarray(tail, np.float64), np.asarray(rest, np.float64)
    cert = S1.certificate(tail, rest, tags, K.CERT)
    rank = S1.rank_test(tail, rest)
    groups = dict(n_full_throttle=int(len(tail)), n_rest=int(len(rest)),
                  mean_full_throttle=float(tail.mean()) if len(tail) else float("nan"),
                  mean_rest=float(rest.mean()) if len(rest) else float("nan"),
                  q90_full_throttle=float(np.quantile(tail, 0.9, method="inverted_cdf")) if len(tail) else float("nan"),
                  q90_rest=float(np.quantile(rest, 0.9, method="inverted_cdf")) if len(rest) else float("nan"))
    out = dict(t_lo=gr["t_lo"], t_up=gr["t_up"], certificate=cert, groups=groups, rank_test=rank)
    if plain:
        out["split_conformal"] = N.split_conformal(tail, rest, K.ARM_RULE["conformal_margin"],
                                                   K.SPLIT_CONFORMAL["level"])
        out["mean_difference"] = N.mean_difference(tail, rest, K.MEANDIFF_MARGIN)
    return S1.jsonable(out)


def reported_entry(tail, rest, seed, world, gr):
    """A reported-only entry (the one-row-per-episode variant): every switch's call and height, no arm reads it."""
    e = certificate_entry(tail, rest, seed, world, gr)
    e["wilks"] = step1lib().jsonable(N.wilks_twin(tail, rest, K.CERT["margin"], K.WILKS["level"],
                                                  K.WILKS["confidence"]))
    return e


def num(x):
    """A recorded number (the strings 'inf', '-inf' and 'nan' of step 1's records included); nan when unreadable."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def refused(cert):
    """BCA refused: its switch is on but its tail threshold R_tail, and so its height, is not finite (section 3.3)."""
    return bool(cert.get("on") and not np.isfinite(num(cert.get("R_tail", cert.get("height")))))


def switch(run, ctx):
    k = run.key
    d = _learner_dir(run)
    rec2 = SD.load_record(d / "critic_second.json")
    pg = ctx.prepared(k.world)
    gr = gap_rule_of(pg)
    base = dict(kind="near80v9-switch", setting=k.setting, world=k.world, seed=k.seed, groups_rule="pooled")
    if bool(gr["defined"]) != bool(rec2.get("defined")):
        raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what="the gap rule differs from the second critic's"))
    if not gr["defined"]:
        entry = step1lib().jsonable(undefined_entry(gr))
        return dict(base, t_lo=gr["t_lo"], t_up=gr["t_up"], defined=False, n_survivors=0,
                    certificate=entry["certificate"], rank_test=entry["rank_test"], groups=entry["groups"],
                    entry=entry, bca_refused=False, gap_rule_undefined=True)
    surv = np.asarray(pg.rows[gr["tail"]], np.int64)
    if step1lib().array_sha256(surv) != rec2.get("survivors_sha256"):
        raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what="the survivors differ from the second critic's"))
    critic2 = SD.load_tree_checked(d / "critic_second.msgpack", L.critic_template(), rec2["file_sha256"])
    sd = ctx.seed(k.world)
    lg, drec = ctx.heldout_view(k.world)
    count = drec.get("group_count") or {}
    h = N.heldout_rows(lg, sd.gap, K.HELDOUT_SPLITS)
    rows_sha, tr_sha = SD.heldout_fingerprints(sd, lg, h)
    if rows_sha != drec.get("heldout_rows_sha256"):
        raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what="the held-out rows differ from the data record's"))
    if tr_sha != drec.get("heldout_transitions_sha256"):
        raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what="the held-out transitions differ from the data "
                                                                     "record's"))
    tail, rest, (tail_rows, rest_rows, surv_rows) = switch_scores(critic2, lg, sd.gap, pg, gr, ctx.learn.gamma)
    checks = N.group_checks(lg, sd.gap, pg.rows, surv_rows, tail_rows, rest_rows, gr["t_lo"], K.HELDOUT_SPLITS)
    checks["counts_equal_the_outcome_free_count"] = bool(len(tail) == count.get("n_tail")
                                                         and len(rest) == count.get("n_rest"))
    checks["passed"] = bool(checks["passed"] and checks["counts_equal_the_outcome_free_count"])
    if not checks["passed"]:
        raise PH.PhaseStop(PH.STOP_REASONS["check"], dict(what="BCA's two pooled groups", checks=checks), checks)
    entry = certificate_entry(tail, rest, k.seed, k.world, gr)
    wilks = step1lib().jsonable(N.wilks_twin(tail, rest, K.CERT["margin"], K.WILKS["level"], K.WILKS["confidence"]))
    wt = K.WORLD_TAG[k.world]
    one = (N.one_per_episode(lg, tail_rows, [*K.ONE_PER_EPISODE_TAG_HEAD, k.seed, wt, 1]),
           N.one_per_episode(lg, rest_rows, [*K.ONE_PER_EPISODE_TAG_HEAD, k.seed, wt, 2]))
    one_entry = reported_entry(tail[one[0]], rest[one[1]], k.seed, k.world, gr)
    bt, br = bank_switch_scores(critic2, pg, gr, ctx.learn.gamma)
    bank_entry = certificate_entry(bt, br, k.seed, k.world, gr, plain=False)
    src = dict(tail=N.group_sources(lg, tail_rows, surv_rows), rest=N.group_sources(lg, rest_rows, surv_rows))
    cert = entry["certificate"]
    return dict(base, t_lo=gr["t_lo"], t_up=gr["t_up"], defined=True, n_survivors=int(len(surv)),
                certificate=cert, rank_test=entry["rank_test"], groups=entry["groups"], entry=entry,
                split_conformal=entry["split_conformal"], mean_difference=entry["mean_difference"],
                group_sources=src, group_checks=checks,
                outcome_free_count=dict(n_tail=count.get("n_tail"), n_rest=count.get("n_rest"),
                                        certifiable=count.get("certifiable"), below_bar=count.get("below_bar"),
                                        recorded=count.get("recorded"), topup_blocks=len(lg.blocks) if hasattr(
                                            lg, "blocks") else 0),
                bca_refused=refused(cert),
                reported_only=dict(
                    wilks=dict(wilks, bca_R_tail=cert.get("R_tail"), bca_R_rest=cert.get("R_rest"),
                               R_tail_equals_wilks_upper=bool(num(cert.get("R_tail")) == num(wilks.get(
                                   "upper_tail")))),
                    one_row_per_episode=dict(rule="one row per episode, chosen uniformly with default_rng([20261020, "
                                                  "9901, seed, world, 1 or 2])", n_tail=int(len(one[0])),
                                             n_rest=int(len(one[1])), certificate=one_entry["certificate"],
                                             groups=one_entry["groups"], rank_test=one_entry["rank_test"],
                                             split_conformal=one_entry["split_conformal"],
                                             mean_difference=one_entry["mean_difference"], wilks=one_entry["wilks"]),
                    distinct_episode_share_tail=src["tail"].get("distinct_share"),
                    bank_only=dict(certificate=bank_entry["certificate"], groups=bank_entry["groups"],
                                   rank_test=bank_entry["rank_test"]),
                    note="no arm reads these: the pooled groups above are BCA's evidence"),
                heldout_rows_sha256=rows_sha, heldout_transitions_sha256=tr_sha,
                critic_second_sha256=rec2["file_sha256"], gamma=float(ctx.learn.gamma))


# --------------------------------------------------------------------------------------------- actors

def clone(run, ctx):
    k = run.key
    pg = ctx.prepared(k.world)
    if run.steps is None:
        params, result = L.clone(pg, ctx.learn, k.seed, log=run.log)
    else:
        params, result = L.clone(pg, ctx.learn, k.seed, steps=run.steps, chunk=min(5_000, int(run.steps)),
                                 log=run.log)
    if result.get("failed"):
        raise PH.PhaseStop(PH.STOP_REASONS["learner"], dict(what="the cloned actor", result=result))
    sha = save_tree_once(_learner_dir(run) / "clone.msgpack", params)
    return dict(kind="near80v9-clone", setting=k.setting, world=k.world, seed=k.seed, result=result, file_sha256=sha,
                actor_sha256=L.tree_sha256(params), failed=False)


def reference_layout(setting, world):
    """[(arm name, critic name)] of the reference actors' one vmapped program (the gate's check-2 layout)."""
    if world == "stall":
        return [("none", "setting"), ("full_none", "full"), ("pad", "setting")]
    return [("none", "setting"), ("pad", "setting")]


def saved_reference_names(setting, world):
    return ("none", "full_none") if world == "stall" else ("none",)


def reference_actors(run, ctx):
    k = run.key
    pg = ctx.prepared(k.world)
    start = ctx.clone(k.world)
    crit = {"setting": ctx.critic(k.world)}
    layout = reference_layout(k.setting, k.world)
    if any(c == "full" for _, c in layout):
        crit["full"] = ctx.full_critic()
    ao = L.ActorOnly(pg, start, ctx.learn, k.seed, alpha=K.ALPHA, steps=run.steps, chunk=run.chunk)
    arms = [L.Arm(name, critic=i) for i, (name, _) in enumerate(layout)]
    stacked, info = ao.run(arms, [crit[c] for _, c in layout], weighted=False, log=run.log)
    keep = saved_reference_names(k.setting, k.world)
    actors, failed = {}, {}
    for i, (name, _) in enumerate(layout):
        failed[name] = bool(info["failed"][i])
        if name in keep:
            actors[name] = ao.arm_params(stacked, i)
    bad = [n for n in keep if failed[n]]
    if bad:
        raise PH.PhaseStop(PH.STOP_REASONS["learner"], dict(what=f"reference actors {bad}"))
    from flax import serialization
    names = sorted(actors)
    sha = LG.write_bytes_once(_learner_dir(run) / "reference_actors.msgpack",
                              serialization.to_bytes({n: actors[n] for n in names}))
    return dict(kind="near80v9-reference-actors", setting=k.setting, world=k.world, seed=k.seed, alpha=K.ALPHA,
                layout=[dict(arm=a, critic=c) for a, c in layout], names=names,
                actor_sha256={n: L.tree_sha256(actors[n]) for n in names}, file_sha256=sha, failed=failed,
                steps=ao.steps, chunk=ao.chunk, rows_sha256=info.get("rows_sha256"),
                start_sha256=info.get("start_sha256"))


DISPATCH = dict(critic=critic, critic_full=critic_full, critic_second=critic_second, switch=switch, clone=clone,
                reference_actors=reference_actors)
