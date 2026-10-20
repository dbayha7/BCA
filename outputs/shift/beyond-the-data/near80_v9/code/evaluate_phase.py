"""Near-80 (version 9, DRAFT): the evaluations and the reproduction check.

evaluate (one seed and world of the main setting; after the group's lock lists this lock's SHA-256): for every scored
actor of the lock, with the gate's anchors and the setting's prepared data:
  as trained     acceptance._evaluate (learners.evaluate_actor: exact 100-step returns on the four fixed test start
                 sets, points, the same actor capped at 80% by recorder.CappedController, the share of steps past 80%);
  held to 70%    the same rollouts with recorder.CappedController(learners.mlp_policy(params, prepared), cap=0.70);
  one recorded rollout per situation of the as-trained policy: the zone shares (near80lib.throttle_zones) and the
                 lower tail (mean points of the worst tenth of the starts: the worst 100 of 1,000).
  unseen_gain = points - held70_points; overshoot_loss = capped80_points - points (reported only).
The scores come from simulator rollouts on the four fixed test start sets, never from a recorded row: nothing here
reads BCA's pooled held-out groups.
reproduction (plumbing only, scratch, full length, seed 0): the removal, the critic, the cloned actor, the no-caution
  actor, the second critic and the switch computed with the as-built bank-only groups must reproduce the gate's and
  step 1's saved files bit for bit; then policy_phase.train_world itself (the function every official policy phase
  runs, with the main setting's arm set and step 2's ramp programs) must reproduce step 2's saved plumbing-check,
  switched-ramp and hand-set range-limit actors bit for bit. The pooled groups' certificate on seed 0 is recorded
  beside it (reported, never compared: step 1 never computed it).
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
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
from experiments.shift import acceptance as A  # noqa: E402
from experiments.shift import learners as L  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.shift import truth as TR  # noqa: E402

VALUE_FIELDS = ("points", "held70_points", "capped80_points", "unseen_gain", "overshoot_loss", "past80_share",
                "zone_shares", "lower_tail_points")
REPRO_PLAIN = dict(removal="the main test's removal", critic="the main test's critic",
                   clone="the main test's cloned actor", none="the main test's no-caution actor",
                   w0_check="step 2's plumbing-check actor", obca_k1="step 2's switched-ramp actor",
                   range_hinge="step 2's hand-set range-limit actor")


# --------------------------------------------------------------------------------------------- values

def tail_count(n):
    return max(1, int(n) // 10)


def actor_values(params, prep, anchors, cfg, failed=False):
    """(evaluation, {situation: {held70_points, zone_shares, lower_tail_points}}) of one actor; (None, None) when it
    failed (a non-finite network)."""
    ev = A._evaluate(params, prep, anchors, cfg, failed=failed)
    if ev is None:
        return None, None
    n = int(cfg.bank_size)
    pol = L.mlp_policy(params, prep)
    out = {}
    for s in TC.SITUATIONS:
        x0 = TC.bank_starts(s, n)
        rec = TR.rollout(pol, x0, prep.physics, record=True)
        held = TR.rollout(RC.CappedController(pol, cap=K.HOLD), x0, prep.physics)
        z = N.throttle_zones(np.asarray(rec.commands).reshape(-1, TC.DA))
        pts = np.sort(np.asarray(TR.points(rec.J, anchors[s]), np.float64))
        out[s] = dict(held70_points=float(TR.points(held.J, anchors[s]).mean()),
                      zone_shares=[float(np.mean(z == k)) for k in (0, 1, 2)],
                      lower_tail_points=float(pts[:tail_count(n)].mean()),
                      finite=bool(np.all(np.isfinite(rec.J)) and np.all(np.isfinite(held.J))))
    return ev, out


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def situations_table(ev, extra):
    """The values.json 'situations' block from per-actor evaluations."""
    sits = {}
    for s in TC.SITUATIONS:
        t = {f: {} for f in VALUE_FIELDS}
        for name in ev:
            e, x = ev[name], extra[name]
            if e is None or x is None or not x[s]["finite"]:
                for f in VALUE_FIELDS:
                    t[f][name] = None
                continue
            p, h, c = _f(e[s]["points"]), _f(x[s]["held70_points"]), _f(e[s]["capped_points"])
            t["points"][name] = p
            t["held70_points"][name] = h
            t["capped80_points"][name] = c
            t["unseen_gain"][name] = None if p is None or h is None else p - h
            t["overshoot_loss"][name] = None if p is None or c is None else c - p
            t["past80_share"][name] = _f(e[s]["unrecorded_throttle_share"])
            t["zone_shares"][name] = x[s]["zone_shares"]
            t["lower_tail_points"][name] = _f(x[s]["lower_tail_points"])
        sits[s] = t
    return sits


def full_arm_ids(setting, world, arm_ids):
    """Every logical id of near80_constants.ARMS[setting]: stall-only ids are null in the no-stall world; an unmatched
    control or a failed stand-in is null."""
    out = {}
    for logical in K.ARMS[setting]:
        name = (arm_ids or {}).get(logical)
        if world != "stall" and logical in K.STALL_ONLY_ARMS:
            name = None
        out[logical] = name
    return out


def compute_values(run, ctx, names, finite, acts, arm_ids, lock_sha, source):
    k = run.key
    world = k.world
    prep = ctx.prepared(world)
    an = ctx.anchors(world)
    t0 = time.perf_counter()
    ev, extra = {}, {}
    for name in names:
        ev[name], extra[name] = actor_values(acts[name], prep, an, ctx.cfg, failed=not finite.get(name, False))
        run.log(f"evaluated {name}")
    return dict(kind="near80v9-values", setting=k.setting, world=world, seed=k.seed, group=k.group, source=source,
                lock_sha256=lock_sha, arm_ids=full_arm_ids(k.setting, world, arm_ids),
                situations=situations_table(ev, extra), evaluations=ev, bank_size=int(ctx.cfg.bank_size),
                evaluation_seconds=time.perf_counter() - t0)


def stage_lock_record(run):
    path = Path(run.out_root) / K.LAYOUT["stage_lock"].format(group=run.key.group)
    try:
        return LG.read_json_verified(path)
    except LG.RecordError as error:
        raise PH.Refused(f"refused: {error} (no evaluation before the group's lock)") from None


def evaluate(run, ctx):
    """One seed and world of the main setting (module docstring)."""
    import policy_phase as PP
    k = run.key
    steps, _, _ = PP.sizes(run)
    stage = stage_lock_record(run)
    lock, acts, lock_sha = PP.verify_lock(run, k.setting, k.seed, k.world, steps)
    pid = PH.phase_id(PH.make_key("policy", k.setting, k.world, k.seed))
    if (stage.get("locks") or {}).get(pid, {}).get("lock_sha256") != lock_sha:
        raise PH.Refused(f"refused: the {k.group} lock does not list this lock's SHA-256 ({pid})")
    names = list(lock["scored_names"])
    return compute_values(run, ctx, names, lock["finite"], acts, lock["arm_ids"], lock_sha, "trained")


# --------------------------------------------------------------------------------------------- step 2's saved actors

def step2_paths(run, seed, world):
    """(folder of step 2's saved lock, actors and values; the committed step-2 results file): the reproduction check
    reads step 2's saved lock of seed 0."""
    return K.S2_OUT / "policy" / f"seed{seed}" / world, K.S2_RESULTS


# --------------------------------------------------------------------------------------------- reproduction

def repro_item(name, passed, detail):
    return dict(passed=bool(passed), detail=detail, plain=REPRO_PLAIN[name])


def reproduction_items(got, want):
    """The reproduction check's verdicts from what the near-80 code produced (got) and the saved references (want),
    both {item: value}: removal (kept and bank hashes), critic (file SHA-256), clone and none (actor SHA-256),
    critic_second (file SHA-256), switch (q05, height, on), w0_check, obca_k1 and range_hinge (actor SHA-256). A
    difference in the second critic or the switch is reported under obca_k1 (BCA's switched ramp reads them)."""
    items = {}
    for name in ("removal", "critic", "clone", "none", "w0_check", "range_hinge"):
        items[name] = repro_item(name, got.get(name) is not None and got.get(name) == want.get(name),
                                 dict(version9=got.get(name), saved=want.get(name)))
    parts = {p: bool(got.get(p) is not None and got.get(p) == want.get(p)) for p in ("critic_second", "switch",
                                                                                   "obca_k1")}
    items["obca_k1"] = repro_item("obca_k1", all(parts.values()),
                                  dict(parts=parts, **{p: dict(version9=got.get(p), saved=want.get(p)) for p in parts}))
    return items


def _cert_key(cert):
    return dict(q05=cert.get("q05"), height=cert.get("height"), on=cert.get("on"))


class ReproContext(SD.SettingContext):
    """The main setting of seed 0 through the version-9 code, for the reproduction check: the seed's data from the
    gate's loader (no data record is needed), the prepared rows of the main removal, and the
    critic, cloned actor and no-caution actor the check itself trained; everything else (the full-log actor, the
    anchors, the gate's configuration) as SettingContext gives it for the main setting on seed 0."""

    def __init__(self, run, built, seeds):
        super().__init__(run, "main", 0)
        self._sd.update(seeds)
        self._built = built

    def prepared(self, world="stall"):
        return self._built[world]["prep"]

    def critic(self, world="stall"):
        return self._built[world]["crit"]

    def clone(self, world="stall"):
        return self._built[world]["clone"]

    def none(self, world="stall"):
        return self._built[world]["none"]


# Stops inside the reproduction that are not a reproduction difference: a saved input whose fingerprint does not
# match, or a training file that fails its check, is a missing or damaged input (section 8.2: it can be retried once
# fixed). reproduce_policy lets them through, so the reproduction process exits without a verdict and the plumbing
# writes the retryable plumbing/failed_<utc>.json, never the final reproduction_differs.json.
REPRO_RETRYABLE_STOPS = ("fingerprint", "training_file")


def reproduce_policy(run, ctx, S, MC, table, seed=0):
    """policy_phase.train_world on the main setting of seed 0 in both worlds (the stalling world first, its payload
    then serving as the no-stall world's lock, as in the official phases). Returns ({world: actors or None},
    {world: detail}); a recorded stop inside train_world (the height-0 check, a learner, a built-in check) leaves that
    world's actors None (its items then differ). A stop for a saved input or a training file (REPRO_RETRYABLE_STOPS)
    is raised again: it is not a reproduction difference."""
    import policy_phase as PP
    got, detail, stall_payload = {}, {}, None
    for world in K.WORLDS:
        key = PH.make_key("policy", "main", world, seed)
        prun = PH.Run(key, run.out_root, mode="scratch", prereg=run.prereg, approval=run.approval, log=run.log)
        row = PP.arm_row_from("main", seed, world, {f"{w}/seed{seed}": table[w] for w in K.WORLDS}, S)
        try:
            payload, actors = PP.train_world(prun, ctx, S, MC, row, stall_lock=stall_payload)
        except PH.PhaseStop as stop:
            if stop.reason in tuple(PH.STOP_REASONS[k] for k in REPRO_RETRYABLE_STOPS):
                raise
            got[world] = None
            detail[world] = dict(stopped=stop.reason, detail=stop.detail)
            continue
        if world == "stall":
            stall_payload = payload
        got[world] = actors
        detail[world] = dict(ramp_programs=payload.get("ramp_programs"), checks=payload.get("checks"))
    return got, detail


def reproduction(run):
    """Plumbing only (scratch, full length, seed 0, both worlds; the main setting through the
    near-80 code path). Trains the critic, the cloned actor, the no-caution actor and the second critic of each world
    exactly as the official phases do, scores the switch, then runs policy_phase.train_world (the official policy
    phases' own function) on them, and compares with the gate's saved files, step 1's results and step 2's saved
    locks. Returns {world: {item: {passed, detail, plain}}}."""
    import learner_phases as LP
    import policy_phase as PP
    if run.mode != "scratch" or run.steps is not None:
        raise PH.Refused("refused: the reproduction check runs in scratch mode, at full length, only")
    S, MC, _ = PP.load_step2(run)
    p = SD.gate_prereg()
    learn = p.config.learn
    seed = 0
    s1 = json.loads(Path(K.S1_RESULTS).read_bytes().decode("utf8"))
    out, built = {}, {}
    pooled = {}
    for world in K.WORLDS:
        sd = SD.gate_seed(seed, world)
        rm = SD.removal("main", sd)
        se, sg = rm.summary(sd.log), (sd.record.get("removals") or {}).get("gap") or {}
        prep = L.prepare(sd.log, rm, sd.norm(), learn)
        res, crit = LP.train_critic(prep, learn, seed, None, run.log)
        from flax import serialization
        crit_file = LG.sha256_bytes(serialization.to_bytes(crit))
        grec, _ = A.get_critic(p, sd, "gap", K.CONTROLLER, train=False)
        clone_p, _ = L.clone(prep, learn, seed, log=run.log)
        ao = L.ActorOnly(prep, clone_p, learn, seed, alpha=K.ALPHA)
        layout = LP.reference_layout("main", world)
        crits = {"setting": crit}
        if any(c == "full" for _, c in layout):
            crits["full"] = A.get_critic(p, SD.gate_seed(seed, "stall"), "none", K.CONTROLLER, train=False)[1]
        stacked, _ = ao.run([L.Arm(n, critic=i) for i, (n, _) in enumerate(layout)], [crits[c] for _, c in layout],
                            weighted=False, log=run.log)
        none = ao.arm_params(stacked, 0)
        gr = LP.gap_rule_of(prep)
        rm2, surv = LP.second_removal("main", sd, prep, gr)
        prep2 = L.prepare(sd.log, rm2, sd.norm(), learn)
        _, crit2 = LP.train_critic(prep2, learn, seed, None, run.log)
        tail, rest = LP.bank_switch_scores(crit2, prep, gr, learn.gamma)
        entry = LP.certificate_entry(tail, rest, seed, world, gr)
        pt, pr, _ = LP.switch_scores(crit2, sd, prep, gr, learn.gamma)
        pooled[world] = LP.certificate_entry(pt, pr, seed, world, gr)
        s1e = s1["per_seed_world"][f"{world}/seed{seed}"]
        built[world] = dict(prep=prep, crit=crit, clone=clone_p, none=none, entry=entry, sd=sd)
        out[world] = dict(
            got=dict(removal=[se["kept_rows_sha256"], se["bank_rows_sha256"]], critic=crit_file,
                     clone=L.tree_sha256(clone_p), none=L.tree_sha256(none),
                     critic_second=LG.sha256_bytes(serialization.to_bytes(crit2)),
                     switch=_cert_key(entry["certificate"])),
            want=dict(removal=[sg["kept_rows_sha256"], sg["bank_rows_sha256"]],
                      critic=grec["result"]["file_sha256"],
                      clone=L.tree_sha256(SD.gate_clone(seed, world)),
                      none=L.tree_sha256(SD.gate_check2_actor(seed, "gapped" if world == "stall"
                                                              else "no_stall_gapped")),
                      critic_second=s1e["critic"]["file_sha256"], switch=_cert_key(s1e["certificate"])))
    seeds = {w: built[w]["sd"] for w in K.WORLDS}
    ctx = ReproContext(run, built, seeds)
    acts, detail = reproduce_policy(run, ctx, S, MC, {w: built[w]["entry"] for w in K.WORLDS}, seed)
    for world in K.WORLDS:
        d, _ = step2_paths(run, seed, world)
        lock = LG.read_json_verified(d / "lock.json")
        for name in (K.W0, K.PRIMARY, "range_hinge"):
            a = (acts.get(world) or {}).get(name)
            out[world]["got"][name] = None if a is None else L.tree_sha256(a)
            out[world]["want"][name] = lock["actor_sha256"].get(name)
    items = {w: reproduction_items(out[w]["got"], out[w]["want"]) for w in K.WORLDS}
    for w in K.WORLDS:
        for name in (K.W0, K.PRIMARY, "range_hinge"):
            items[w][name]["detail"]["policy"] = detail.get(w)
        items[w]["obca_k1"]["detail"]["pooled_groups_reported_only"] = dict(
            certificate=pooled[w]["certificate"], groups=pooled[w]["groups"],
            note="version 9's pooled groups on seed 0 (reported, never compared)")
    return items
