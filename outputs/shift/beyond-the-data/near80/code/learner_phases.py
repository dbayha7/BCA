"""Near-80 (version 8): the learner phases. Each calls the gate's own learner code, unchanged, on one setting's
prepared rows; only what the build spec (section 5.6) lists is restated.

  critic            learners.fixed_critic(prepared, 'stall_aware', learn, seed) (the gate's call in
                    acceptance.get_critic): 100,000 updates, then frozen.
  critic_full       the same on learners.prepare(log, removals['none'], the gate's normalization) of the stalling world,
                    seeds 10-14 (seeds 0-4 read the gate's full-log critic through its loader).
  critic_second     step 1's construction (restructure/step1/train_critics.py): step 1's gap rule on the setting's kept
                    commands, the survivors removed as well, the same data checks, learners.fixed_critic. Restated
                    because step 1's script refuses other seeds and checks its design's expected counts. A gap rule
                    that is not defined is not a stop: the record says defined: false and the switch is off.
  switch            step 1's scoring (restructure/step1/evaluate.py): one-sided overestimates of the survivors and the
                    bank under the second critic, BCA's certificate (step1lib.certificate, margin 0.10), the rank test
                    and the group summaries. The arm table row is computed later, inside the policy phase.
  clone             learners.clone(prepared, learn, seed).
  reference_actors  learners.ActorOnly(prepared, clone, learn, seed, alpha=5).run(arms, critics, weighted=False), with
                    the gate's check-2 layout so that the reproduction check can be bit for bit: stalling world
                    [none: setting critic, full_none: full-log critic, pad: setting critic]; no-stall world [none,
                    pad]. The pads are discarded. The random-rows setting saves 'none' only.
  v6_band           version 6's band (v6bca.fit); a failure is recorded, not a stop.
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
    return dict(kind="near80-critic", setting=k.setting, world=k.world, seed=k.seed, controller=K.CONTROLLER,
                result=_trim(result), updates=result.get("final_update", result.get("updates")), file_sha256=sha,
                critic_sha256=L.tree_sha256(params), failed=False, data=pg.describe())


def critic_full(run, ctx):
    k = run.key
    sd = ctx.seed("stall")
    prep = L.prepare(sd.log, sd.removals["none"], sd.norm(), ctx.learn)
    result, params = train_critic(prep, ctx.learn, k.seed, run.updates, run.log)
    d = Path(run.out_root) / Path(K.LAYOUT["critic_full"].format(seed=k.seed)).parent
    sha = save_tree_once(d / "critic.msgpack", params)
    return dict(kind="near80-critic-full", setting="full", world="stall", seed=k.seed, controller=K.CONTROLLER,
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


def second_checks(prep, pg, surv):
    """Step 1's training-data checks of the second critic."""
    return dict(
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
    rec = dict(kind="near80-critic-second", setting=k.setting, world=k.world, seed=k.seed, controller=K.CONTROLLER,
               gap_rule=dict(defined=bool(gr["defined"]), t_lo=gr["t_lo"], t_up=gr["t_up"], n_tail=gr["n_tail"],
                             gap=gr["gap"], reason=gr.get("reason")), defined=bool(gr["defined"]))
    if not gr["defined"]:
        rec.update(file_sha256=None, failed=False, note="the gap rule is not defined: no second critic, the switch "
                                                        "is off")
        return rec
    sd = ctx.seed(k.world)
    rm2, surv = second_removal(k.setting, sd, pg, gr)
    prep = L.prepare(sd.log, rm2, sd.norm(), ctx.learn)
    checks = second_checks(prep, pg, surv)
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
                certificate=dict(on=False, height=nan, q05=None, reason="the gap rule is not defined"),
                groups=dict(n_full_throttle=0, n_rest=0, mean_full_throttle=nan, mean_rest=nan, q90_full_throttle=nan,
                            q90_rest=nan),
                rank_test=dict(U=None, p=nan))


def switch_scores(critic2, pg, gr, gamma):
    """(tail, rest): the second critic's one-sided overestimates of the full-throttle group (the survivors and the
    bank rows past t_lo) and of the rest (the bank rows at or below t_lo); all held out from the second critic."""
    S1 = step1lib()
    surv_idx = np.flatnonzero(gr["tail"])
    fixed = L.FixedController(K.CONTROLLER, pg.physics, pg.obs_mean, pg.obs_std)
    bank_t = S1.throttle(np.asarray(pg.bank.action))
    bank_ft = S1.full_throttle(bank_t, gr["t_lo"])
    s_tr = overestimates(critic2, pg.train, fixed, gamma, rows=surv_idx)
    s_bk = overestimates(critic2, pg.bank, fixed, gamma)
    return np.concatenate([s_tr, s_bk[bank_ft]]), s_bk[~bank_ft]


def certificate_entry(tail, rest, seed, world, gr):
    """Step 1's certificate, rank test and groups for one seed-world, as step 1's results record them (non-finite
    numbers as the strings 'inf', '-inf', 'nan', which step2lib.num reads)."""
    S1 = step1lib()
    tags = [*K.CERT_TAG_HEAD, int(seed), K.WORLD_TAG[world]]
    cert = S1.certificate(tail, rest, tags, K.CERT)
    rank = S1.rank_test(tail, rest)
    groups = dict(n_full_throttle=int(len(tail)), n_rest=int(len(rest)),
                  mean_full_throttle=float(tail.mean()) if len(tail) else float("nan"),
                  mean_rest=float(rest.mean()) if len(rest) else float("nan"),
                  q90_full_throttle=float(np.quantile(tail, 0.9, method="inverted_cdf")) if len(tail) else float("nan"),
                  q90_rest=float(np.quantile(rest, 0.9, method="inverted_cdf")) if len(rest) else float("nan"))
    return S1.jsonable(dict(t_lo=gr["t_lo"], t_up=gr["t_up"], certificate=cert, groups=groups, rank_test=rank))


def switch(run, ctx):
    k = run.key
    d = _learner_dir(run)
    rec2 = SD.load_record(d / "critic_second.json")
    pg = ctx.prepared(k.world)
    gr = gap_rule_of(pg)
    base = dict(kind="near80-switch", setting=k.setting, world=k.world, seed=k.seed)
    if bool(gr["defined"]) != bool(rec2.get("defined")):
        raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what="the gap rule differs from the second critic's"))
    if not gr["defined"]:
        entry = step1lib().jsonable(undefined_entry(gr))
        return dict(base, t_lo=gr["t_lo"], t_up=gr["t_up"], defined=False, n_survivors=0,
                    certificate=entry["certificate"], rank_test=entry["rank_test"], groups=entry["groups"],
                    entry=entry)
    surv = np.asarray(pg.rows[gr["tail"]], np.int64)
    if step1lib().array_sha256(surv) != rec2.get("survivors_sha256"):
        raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what="the survivors differ from the second critic's"))
    critic2 = SD.load_tree_checked(d / "critic_second.msgpack", L.critic_template(), rec2["file_sha256"])
    tail, rest = switch_scores(critic2, pg, gr, ctx.learn.gamma)
    entry = certificate_entry(tail, rest, k.seed, k.world, gr)
    return dict(base, t_lo=gr["t_lo"], t_up=gr["t_up"], defined=True, n_survivors=int(len(surv)),
                certificate=entry["certificate"], rank_test=entry["rank_test"], groups=entry["groups"], entry=entry,
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
    return dict(kind="near80-clone", setting=k.setting, world=k.world, seed=k.seed, result=result, file_sha256=sha,
                actor_sha256=L.tree_sha256(params), failed=False)


def reference_layout(setting, world):
    """[(arm name, critic name)] of the reference actors' one vmapped program (the gate's check-2 layout)."""
    if world == "stall":
        if setting in ("near80", "main"):
            return [("none", "setting"), ("full_none", "full"), ("pad", "setting")]
        return [("none", "setting"), ("pad", "setting"), ("pad_2", "setting")]
    return [("none", "setting"), ("pad", "setting")]


def saved_reference_names(setting, world):
    return ("none", "full_none") if (world == "stall" and setting in ("near80", "main")) else ("none",)


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
    return dict(kind="near80-reference-actors", setting=k.setting, world=k.world, seed=k.seed, alpha=K.ALPHA,
                layout=[dict(arm=a, critic=c) for a, c in layout], names=names,
                actor_sha256={n: L.tree_sha256(actors[n]) for n in names}, file_sha256=sha, failed=failed,
                steps=ao.steps, chunk=ao.chunk, rows_sha256=info.get("rows_sha256"),
                start_sha256=info.get("start_sha256"))


def v6_band(run, ctx):
    import v6bca
    k = run.key
    pg = ctx.prepared(k.world)
    band, rec, extras = v6bca.fit(pg, ctx.critic(k.world), k.seed, k.world, ctx.learn, steps=run.steps, log=run.log)
    rec = dict(rec, kind="near80-v6-band", setting=k.setting, world=k.world, seed=k.seed)
    if band is None:
        rec.update(band_file_sha256=None)
        return rec
    d = _learner_dir(run)
    rec["band_file_sha256"] = save_tree_once(d / "v6_band.msgpack", band)
    if extras is not None:
        import io
        buf = io.BytesIO()
        np.savez(buf, scores=extras["scores"], eta=extras["eta"])
        rec["bank_scores_file_sha256"] = LG.write_bytes_once(d / "v6_bank_scores.npz", buf.getvalue())
        rec["bank_scores_sha256"] = LG.sha256_bytes(np.ascontiguousarray(extras["scores"]).tobytes())
    return rec


def load_v6_band(out_root, setting, world, seed):
    """(Band or None, record) of a v6_band phase; None when the fit failed (the arm is then missing)."""
    import v6bca
    d = SD.learner_dir(out_root, setting, world, seed)
    rec = SD.load_record(d / "v6_band.json")
    if rec.get("failed"):
        return None, rec
    band = SD.load_tree_checked(d / "v6_band.msgpack", v6bca.band_template(), rec["band_file_sha256"])
    return band, rec


DISPATCH = dict(critic=critic, critic_full=critic_full, critic_second=critic_second, switch=switch, clone=clone,
                reference_actors=reference_actors, v6_band=v6_band)
