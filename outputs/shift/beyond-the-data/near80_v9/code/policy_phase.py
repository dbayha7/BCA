"""Near-80 (version 9): the actor-training phases (step 2's actor-only protocol, restated for version 9), and the lock.

Step 2's code is imported, never edited: step2lib (after step2lib.verify_manifest() passes and step 2's code
manifest has the SHA-256 prereg.json pins) and machinery (which executes version 6's code and step 1's library from
verified bytes and refuses otherwise). Reused unchanged from machinery: RampRunner, ramp_arm, gap_rule_check,
check_imports, best_standin, v6_prereg, standin_matched, standin_locked, ramp_shape, info_only, log; and through it
the room harness (H: ProgramConfig, build_shared, random_surface_params, RoomArm, RoomRunner, family_arm, the ladders,
max_param_diff, region_strength, MAIN_BAND) and version 6's policy code (P: room_scales, match_controls, strengths_of,
arms_record, standins_module, MATCHED). Restated here, because step 2's versions call machinery.context (which refuses
every seed but 0-4 and reads only the gate's saved artifacts) or write only under step 2's folder: train_stall,
train_no_stall (one function, train_world), locked_controls, reference, h0_check, strengths, save_and_lock,
verify_lock. They take a settingdata.SettingContext and write under version 9's output root.

The arm table is arms_v9.arm_table (PREREGISTRATION.md sections 3.3 and 5): step 2's rule with version 9's refusal
rule (a refusal marks arms, never raises), the textbook split-conformal switch in conformal_k1's slot, the dose arms
and the computed plain switch with the hand-set limit. step2lib.arm_table is still called, and when it does not
refuse, the restated rule must equal it on the arms they share (a built-in check).

Step 2's ramp programs (ramp_layout; sections 5 and 8.2). machinery.RampRunner compiles one vmapped program per
number of ramp arms, and a member's bits depend on that number, so BCA's switched ramp and the plumbing check run in
exactly step 2's programs: in the stalling world w0_check alone, then a batch of six ramp arms with obca_k1 first; in
the no-stall world one batch of seven, w0_check first and obca_k1 second. A refused arm's place in its batch is kept
by a height-0 pad (refused_pad_<arm>), checked equal to no caution (the height-0 check) and never saved or scored, so
every other arm trains in the program it would have had without the refusal. The two dose arms train together in a
program of their own, after step 2's programs, and only in a world where the plain switch is on; where it is off they
are the no-caution actor by definition and never enter the height-0 check.
  stalling world: w0_check alone first (version 6's check (b)); the six; perfect_band; range_hinge; the controls'
      ladders and version 6's matcher blind to obca_k1's strength; the stand-in matched; the dose arms where on. When
      obca_k1 is refused there, nothing can be matched (no target): the matched rivals are unmatched (BCA's own).
  no-stall world: w0_check and the six in one program, perfect_band, range_hinge, the dose arms where on; the controls
      and the stand-in at the stalling world's locked coefficients (absent when the stalling world has none).
Every height-0 ramp arm and every pad must reproduce the no-caution actor within W0_TOL (else a recorded stop). The
lock (every actor's SHA-256, every computed arm's source, every unmatched rival's reason) is written before any
return, points or score of any trained arm is computed.

Why a matched rival is unmatched (unmatched_reason, read by the look): 'bca_refused' (obca_k1 refused: no target),
'bca_switch_off' (BCA's switch is off where full throttle stalls: no target), 'bca_not_finite' (BCA's own actor is not
finite), 'bca_no_positive_strength' (BCA's switched ramp has no positive strength to match: no target), all BCA's own;
'rival_side' (the matcher could not match although BCA's strength is positive, a rival network that is not finite, or
an error inside the stand-in), not BCA's.
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import near80_constants as K  # noqa: E402
import arms_v9 as AV  # noqa: E402
import ledger as LG  # noqa: E402
import phases as PH  # noqa: E402
import settingdata as SD  # noqa: E402

LOCKED_FAMILIES = ("even_pull", "push_less", "random_1", "random_2")
STALL_ONLY_IDS = ("full_none",)
_STEP2 = {}


# --------------------------------------------------------------------------------------------- step 2's code

def load_step2(run=None):
    """(step2lib, machinery, facts): step2lib after verify_manifest() passes, then machinery. In official and scratch
    mode step 2's code manifest and prespec must have the SHA-256 that prereg.json pins."""
    if "S" not in _STEP2:
        if str(K.S2_DIR) not in sys.path:
            sys.path.insert(0, str(K.S2_DIR))
        import step2lib as S
        man, man_sha = S.verify_manifest()
        spec, spec_sha = S.load_prespec()
        import machinery as MC
        _STEP2.update(S=S, MC=MC, facts=dict(code_manifest_sha256=man_sha, prespec_sha256=spec_sha))
    S, MC, facts = _STEP2["S"], _STEP2["MC"], _STEP2["facts"]
    if run is not None and not run.stub:
        for pin, got in (("s2_code_manifest_sha256", facts["code_manifest_sha256"]),
                         ("s2_prespec_sha256", facts["prespec_sha256"])):
            want = run.pin(pin)
            if want is None and run.official:
                raise PH.Refused(f"refused: prereg.json pins no {pin}")
            if want is not None and want != got:
                raise PH.Refused(f"refused: step 2's {pin[3:]} is {got[:16]}..., prereg.json pins {want[:16]}...")
    return S, MC, facts


def sizes(run):
    """(steps, chunk, full_length) of the actor programs."""
    steps = K.ACTOR_STEPS if run.steps is None else int(run.steps)
    chunk = min(K.CHUNK, steps) if run.chunk is None else int(run.chunk)
    if steps <= 0 or chunk <= 0 or steps % chunk or steps > K.ACTOR_STEPS:
        raise PH.Refused(f"refused: {steps} actor steps in chunks of {chunk}")
    return steps, chunk, steps == K.ACTOR_STEPS


def policy_dir(out_root, setting, world, seed):
    return Path(out_root) / Path(K.LAYOUT["policy_lock"].format(setting=setting, world=world, seed=seed)).parent


# --------------------------------------------------------------------------------------------- the arm table

def arm_row(setting, seed, world, out_root, S=None):
    """The arm-table row of the main setting, one seed and world, from BOTH worlds' switch records (obca_forced uses
    the stalling world's height): arm_row_from."""
    S = S or load_step2()[0]
    per = {}
    for w in K.WORLDS:
        rec = SD.load_record(SD.learner_dir(out_root, setting, w, seed) / "switch.json")
        per[f"{w}/seed{int(seed)}"] = rec["entry"]
    return arm_row_from(setting, seed, world, per, S)


def restated_table(S, per, seed, rule=K.ARM_RULE):
    """arms_v9.arm_table: {world: row}. An entry that cannot be read is a recorded stop."""
    try:
        return AV.arm_table(per, seed, rule)
    except AV.TableError as error:
        raise PH.PhaseStop(PH.STOP_REASONS["check"], dict(what="the switch records cannot be read",
                                                          error=str(error))) from None


def same_as_step2(row, s2row):
    return AV.same_as_step2(row, s2row)


def arm_row_from(setting, seed, world, per, S):
    """arm_row on the switch entries {'<world>/seed<k>': entry} of both worlds: the restated table (refusals marked,
    never raised), checked against step2lib's own arm_table when that does not refuse. row['refused']: the arms
    refused in this world; row['bca_refused']: BCA's switched ramp itself was refused here."""
    if setting != "main":
        raise PH.Refused(f"refused: {setting!r} is not a setting of version 9")
    table = restated_table(S, per, seed)
    row = table[world]
    s2per = {k: {kk: v for kk, v in e.items() if kk in ("t_lo", "t_up", "certificate", "groups", "rank_test")}
             for k, e in per.items()}
    try:
        s2row = S.arm_table({"per_seed_world": s2per}, seeds=[int(seed)], rule=K.ARM_RULE)[world][int(seed)]
        refusal = None
    except (SystemExit, ValueError, TypeError, KeyError) as error:
        s2row, refusal = None, str(error)
    if s2row is not None and not same_as_step2(row, s2row):
        raise PH.PhaseStop(PH.STOP_REASONS["check"], dict(what="the restated arm table differs from step2lib.arm_table",
                                                          restated=row, step2=s2row))
    row["arm_table_refusal"] = refusal
    if refusal is not None and not any(a.get("refused") for w_row in table.values() for a in w_row["arms"].values()):
        raise PH.PhaseStop(PH.STOP_REASONS["check"], dict(what="step2lib.arm_table refused for a reason the restated "
                                                               "table does not see", error=refusal))
    row["refusal_events"] = AV.refusal_events(table)
    return row


def step2_row_from(seed, world, per):
    """Step 2's own row (arms_v9.step2_table): the plumbing's reproduction trains it."""
    try:
        return AV.step2_table(per, seed)[world]
    except AV.TableError as error:
        raise PH.PhaseStop(PH.STOP_REASONS["check"], dict(what="step 2's own arm table", error=str(error))) from None


def refused_pad(name):
    return f"refused_pad_{name}"


def ramp_names(setting, row=None):
    """The ramp arms that are trained, kept and scored (on the setting's own ramp, t_lo to t_up), when not refused:
    the six of step 2's program (the row's own, splitconf_k1 in conformal_k1's slot unless the row is step 2's own
    table) and, where the plain switch is on, the dose arms."""
    six = tuple((row or {}).get("the_six") or K.STEP2_RAMP_ARMS)
    dose = K.DOSE_ARMS if (row or {}).get("dose_on") else ()
    refused = set((row or {}).get("refused") or [])
    return tuple(n for n in six + dose if n not in refused)


def ramp_layout(setting, world, row=None):
    """The vmapped ramp programs of the main RampRunner, in call order: [(arm names), ...]. Step 2's own layout
    (module docstring): stalling world [(w0_check,), (the six,)]; no-stall world [(w0_check, the six)]; then the two
    dose arms together where the plain switch is on. With a row, a refused arm's place holds its height-0 pad."""
    row = row or {}
    refused = set(row.get("refused") or [])
    six = tuple(refused_pad(n) if n in refused else n for n in (row.get("the_six") or K.STEP2_RAMP_ARMS))
    calls = [(K.W0,), six] if world == "stall" else [(K.W0,) + six]
    if row.get("dose_on"):
        calls.append(tuple(K.DOSE_ARMS))
    return calls


def layout_arm(MC, row, name):
    """The ramp arm of a name of ramp_layout (w0_check and a refused arm's pad are height 0, k 0)."""
    if name == K.W0 or name.startswith("refused_pad_"):
        return MC.ramp_arm(name, 0.0, 0.0)
    return MC.ramp_arm(name, row["arms"][name]["h"], row["arms"][name]["k"])


# --------------------------------------------------------------------------------------------- restated helpers

def reference(MC, runner, trained, none, full):
    """The actor every height-0 arm must reproduce: the reference phase's 'none' actor at full length (version 6's
    check (b)); in a shortened scratch or stub run, the room's own W = 0 program (kind plain, pull 1) at the same step
    count."""
    if full:
        return none, "the no-caution actor of this setting (version 6's check (b))"
    if "plain_ref" not in trained:
        trained.update(runner.run([MC.H.RoomArm("plain_ref", "plain", (0.0,), 1.0)], MC.log))
    return trained["plain_ref"]["params"], "shortened run only: the room's W = 0 program at the same step count"


def h0_check(MC, trained, names, ref, ref_name):
    rows = {}
    for n in names:
        d = MC.H.max_param_diff(trained[n]["params"], ref)
        rows[n] = dict(max_abs_diff=d, passed=bool(trained[n]["finite"] and d <= MC.W0_TOL))
    return dict(arms=rows, tol=MC.W0_TOL, reference=ref_name, passed=bool(all(r["passed"] for r in rows.values())))


def strengths(MC, trained, extra, none, pg):
    s, sr = MC.P.strengths_of(trained, none, pg)
    for k, p in extra.items():
        s[k] = float(MC.L.strength(p, none, pg.eval_obs))
        sr[k] = MC.H.region_strength(p, none, pg)
    return s, sr


def locked_controls(MC, stall_lock):
    """The even pull, push less and random surfaces at the stalling world's locked knobs and coefficients (the same
    penalty in both worlds; never re-matched)."""
    H = MC.H
    arms, info = [], {}
    for f in LOCKED_FAMILIES:
        m = (stall_lock.get("matches") or {}).get(f) or {}
        if not m.get("matched") or not m.get("arm"):
            info[f] = dict(locked=False, reason="not matched in the stalling world")
            continue
        rec = stall_lock["arms"][m["arm"]]["arm"]
        name = f"{f}_locked"
        if f == "even_pull":
            nu = float(rec["knob"])
            arms.append(H.RoomArm(name, "plain", (0.0,), 1.0 + nu, family=f, knob=nu))
        elif f == "push_less":
            arms.append(H.RoomArm(name, "push_less", tuple(float(c) for c in rec["coef"]), family=f,
                                  knob=float(rec["knob"])))
        else:
            arms.append(H.RoomArm(name, "random", tuple(float(c) for c in rec["coef"]), surface=int(rec["surface"]),
                                  family=f, knob=float(rec["knob"])))
        info[f] = dict(locked=True, name=name, stall_arm=m["arm"], knob=rec["knob"], coef=rec["coef"],
                       pull=rec["pull"], surface=rec["surface"])
    return arms, info


def stop(run, payload, reason, checks=None):
    raise PH.PhaseStop(reason, dict(payload, checks=checks), checks)


# --------------------------------------------------------------------------------------------- one world

def train_world(run, ctx, S, MC, row, stall_lock=None, stall_lock_sha=None):
    """Every arm of the main setting, one seed and world (module docstring); returns the lock payload (without the
    actor hashes, which save_and_lock adds) and the actors."""
    t_start = time.perf_counter()
    H, L, P = MC.H, MC.L, MC.P
    key = run.key
    setting, world, seed = key.setting, key.world, int(key.seed)
    steps, chunk, full = sizes(run)
    refused = set(row.get("refused") or [])
    base = dict(setting=setting, world=world, seed=seed, group=key.group, mode=run.mode, steps=steps, chunk=chunk,
                full_length=full, arm_table=row, stall_lock_sha256=stall_lock_sha)
    t0 = time.perf_counter()
    pg = ctx.prepared(world)
    critic = ctx.critic(world)
    clone = ctx.clone(world)
    none = ctx.none(world)
    full_none = ctx.full_none() if world == "stall" else None
    checks = dict(gap_rule=MC.gap_rule_check(pg, row))
    if not checks["gap_rule"]["passed"]:
        stop(run, base, "step 1's gap rule gives another t_lo or t_up on this data", checks)
    seconds = dict(load=time.perf_counter() - t0)
    t1 = time.perf_counter()
    cfg = H.ProgramConfig() if world == "stall" else H.ProgramConfig(physics=pg.physics, next_command="stall_aware")
    shared, sinfo = H.build_shared(pg, critic, seed, cfg=cfg, need_harm=False, log_fn=MC.log)
    nets = {j: H.random_surface_params(seed, j) for j in (1, 2)}
    g, scale = (P.room_scales(pg, critic, shared, clone, seed, cfg, nets) if world == "stall" else (None, None))
    seconds["shared_and_scales"] = time.perf_counter() - t1
    runner = MC.RampRunner(pg, clone, critic, ctx.learn, seed, cfg=cfg, shared=shared,
                           random_nets={j: nets[j][0] for j in (1, 2)}, steps=steps, chunk=chunk,
                           t_lo=row["t_lo"], t_up=row["t_up"])
    names = tuple(n for n in ramp_names(setting) if n not in refused)
    layout = ramp_layout(setting, world, row)
    t1 = time.perf_counter()
    locked_info = {}
    if world == "stall":
        trained = runner.run([layout_arm(MC, row, n) for n in layout[0]], MC.log)   # w0_check alone first
        ref, ref_name = reference(MC, runner, trained, none, full)
        w0 = H.max_param_diff(trained[K.W0]["params"], ref)
        checks["b_w0"] = dict(max_abs_diff=w0, tol=MC.W0_TOL, reference=ref_name,
                              passed=bool(trained[K.W0]["finite"] and w0 <= MC.W0_TOL))
        if not checks["b_w0"]["passed"]:
            stop(run, base, PH.STOP_REASONS["height0"], checks)
        arms = [layout_arm(MC, row, n) for n in layout[1]]                         # step 2's six-arm program
        arms += [H.RoomArm("perfect_band", "band", H.MAIN_BAND), H.RoomArm("range_hinge", "hinge", (1.0,))]
        arms += [H.family_arm("even_pull", nu, scale) for nu in H.EVEN_LADDER]
        arms += [H.family_arm(f, k, scale) for f in ("push_less", "random_1", "random_2") for k in H.SLOPE_LADDER]
        trained.update(runner.run(arms, MC.log))
        rest = layout[2:]
    else:
        locked, locked_info = locked_controls(MC, stall_lock)
        arms = [layout_arm(MC, row, n) for n in layout[0]]                         # step 2's seven-arm program
        arms += [H.RoomArm("perfect_band", "band", H.MAIN_BAND), H.RoomArm("range_hinge", "hinge", (1.0,))] + locked
        trained = runner.run(arms, MC.log)
        ref, ref_name = reference(MC, runner, trained, none, full)
        w0 = H.max_param_diff(trained[K.W0]["params"], ref)
        checks["b_w0"] = dict(max_abs_diff=w0, tol=MC.W0_TOL, reference=ref_name,
                              passed=bool(trained[K.W0]["finite"] and w0 <= MC.W0_TOL))
        rest = layout[1:]
    for call in rest:                                                              # meandiff_k1 alone
        trained.update(runner.run([layout_arm(MC, row, n) for n in call], MC.log))
    pads = [n for n in trained if n.startswith("refused_pad_")]
    padding = None
    if pads:
        padding = dict(arms=sorted(pads), why="the places of refused arms in step 2's programs (height 0); never "
                                             "saved or scored",
                       max_abs_diff_to_w0={n: float(H.max_param_diff(trained[n]["params"], trained[K.W0]["params"]))
                                           for n in sorted(pads)})
        for n in pads:
            trained.pop(n)
    twin = row["arms"]["meandiff_limit"]
    limit = dict(switch=bool(twin["switch"]), actor=None, same_as_range_hinge=None)
    if twin["switch"]:
        t2 = time.perf_counter()
        trained.update(runner.run([H.RoomArm("meandiff_limit", "hinge", (1.0,))], MC.log))   # a hinge program alone
        limit.update(actor="meandiff_limit", seconds=time.perf_counter() - t2,
                     same_as_range_hinge=float(H.max_param_diff(trained["meandiff_limit"]["params"],
                                                                trained["range_hinge"]["params"])))
    else:
        limit.update(actor="none", why="switched off: the no-caution actor (a switched-off caution is no caution)")
    zero = [n for n in names if row["arms"][n]["h"] == 0.0]
    checks["h0_reproduces_reference"] = h0_check(MC, trained, zero, ref, ref_name)
    if not (checks["b_w0"]["passed"] and checks["h0_reproduces_reference"]["passed"]):
        stop(run, base, PH.STOP_REASONS["height0"], checks)
    seconds["arms"] = time.perf_counter() - t1
    # ---- matching (stalling world) or the stalling world's locked coefficients (no-stall world)
    t1 = time.perf_counter()
    matches, standin, s_actors, s_record = None, None, {}, {}
    target = None
    if world == "stall":
        if K.PRIMARY in refused:
            target = float("nan")
            why = "BCA refused (its height is not finite): there is no strength to match (section 8)"
        else:
            target = float(L.strength(trained[K.PRIMARY]["params"], none, pg.eval_obs)) \
                if trained[K.PRIMARY]["finite"] else float("nan")
            why = "obca_k1 failed or has no positive strength"
        if np.isfinite(target) and target > 0:
            matches = P.match_controls(runner, trained, none, pg, target, scale, P.MATCHED, MC.log)
        else:
            matches = {f: dict(record=None, arm=None, matched=False, reason=why) for f in P.MATCHED}
        best = MC.best_standin()
        standin = dict(family=best, matched=False, arm=None)
        if np.isfinite(target) and target > 0:
            try:
                SM = P.standins_module(MC.v6_prereg())
                s_trained, s_match, s_info = MC.standin_matched(SM, ctx, "stall", pg, critic, clone, none, best, target,
                                                                shared, cfg, steps, chunk, MC.log)
                name = s_match["arm"]
                ok = name is not None and s_trained[name]["finite"]
                standin.update(record=s_match["record"], matched=bool(ok), arm=f"standin:{name}" if ok else None,
                               coefficient=list(s_trained[name]["arm"]["coef"]) if ok else None,
                               knob=s_trained[name]["arm"]["knob"] if ok else None,
                               reason=None if ok else "no finite matched arm", data_side=s_info["info"],
                               scale=s_info["scale"], seconds=s_info["seconds"])
                s_actors = {f"standin:{k}": v["params"] for k, v in s_trained.items() if v["finite"]}
                s_record = {f"standin:{k}": dict(arm=v["arm"], finite=v["finite"], chunk_metrics=v["chunk_metrics"])
                            for k, v in s_trained.items()}
            except Exception as error:   # a rival that cannot run fails every measure that needs it (counts against BCA)
                standin.update(error=repr(error))
                MC.log(f"stand-in error: {error!r}")
        else:
            standin.update(reason=why)
    else:
        st = (stall_lock or {}).get("standin") or {}
        standin = dict(family=st.get("family"), coefficient=st.get("coefficient"), stall_arm=st.get("arm"), arm=None)
        if standin["coefficient"] is not None:
            try:
                SM = P.standins_module(MC.v6_prereg())
                got, s_info = MC.standin_locked(SM, ctx, "no_stall", pg, critic, clone, none, standin["family"],
                                                standin["coefficient"], steps, chunk, MC.log)
                if got["finite"]:
                    s_actors = {"standin_locked": got["params"]}
                    standin["arm"] = "standin_locked"
                s_record = {"standin_locked": dict(arm=got["arm"], finite=got["finite"],
                                                   chunk_metrics=got["chunk_metrics"])}
                standin.update(no_stall_data_side=s_info["info"], seconds=s_info["seconds"])
            except Exception as error:
                standin["error"] = repr(error)
                MC.log(f"stand-in error: {error!r}")
    seconds["matching_and_standin"] = time.perf_counter() - t1
    S_, SR = strengths(MC, trained, s_actors, none, pg)
    shape = MC.info_only("ramp shape", lambda: MC.ramp_shape(
        pg, critic, none, seed, row["t_lo"], row["t_up"], {n: (row["arms"][n]["h"], row["arms"][n]["k"])
                                                          for n in names}))
    # ---- logical ids -> actor names (a refused arm, and a rival that could not be matched, is null)
    ids = {}
    for logical in K.ARMS[setting]:
        if logical in ("none", "clone"):
            ids[logical] = logical
        elif logical == "full_none":
            ids[logical] = "full_none" if full_none is not None else None
        elif logical in K.MATCHED:
            if world == "stall":
                ids[logical] = (matches or {}).get(logical, {}).get("arm")
            else:
                ids[logical] = locked_info[logical]["name"] if (locked_info.get(logical) or {}).get("locked") else None
        elif logical == "standin":
            ids[logical] = (standin or {}).get("arm")
        elif logical == "meandiff_limit":
            ids[logical] = limit["actor"]
        elif logical in refused:
            ids[logical] = None
        else:
            ids[logical] = logical if logical in trained else None
    actors = {k: v["params"] for k, v in trained.items()}
    actors.update(s_actors)
    actors.update(none=none, clone=clone)
    if full_none is not None:
        actors["full_none"] = full_none
    finite = {k: bool(v["finite"]) for k, v in trained.items()}
    finite.update({k: True for k in s_actors}, none=True, clone=True)
    if full_none is not None:
        finite["full_none"] = True
    scored = sorted({v for v in ids.values() if v})
    refusal = dict(bca_refused=bool(row.get("bca_refused")), refused_arms=sorted(refused),
                   reasons={n: row["arms"][n].get("refused_reason") for n in sorted(refused)},
                   matched_rivals_failed=bool(world == "stall" and K.PRIMARY in refused),
                   rule=("a refused arm, and in the stalling world the matched rivals when BCA's switched ramp is "
                         "refused, fail this seed-world; every other arm is trained and scored (section 8)"))
    payload = dict(base, kind="near80v9-policy-lock", checks=checks,
                   arms=dict(P.arms_record(trained), **s_record), strengths=S_, region_strengths=SR,
                   target_strength=target, matches=matches,
                   matched_names=None if matches is None else {f: m["arm"] for f, m in matches.items()},
                   locked_controls=locked_info or None, standin=standin, arm_ids=ids, refusal=refusal,
                   plain_twins=dict(meandiff_k1=row["arms"]["meandiff_k1"], meandiff_limit=limit),
                   scored_names=scored, finite={k: finite[k] for k in scored},
                   slope_rms=g, coefficient_per_knob=scale, random_surface_seeds={j: nets[j][1] for j in (1, 2)},
                   shared=sinfo, ramp_shape=shape, chunk_seconds=runner.chunk_seconds,
                   ramp_programs=[list(c) for c in layout], ramp_padding=padding,
                   seconds=dict(seconds, by_kind=runner.seconds, total=time.perf_counter() - t_start))
    return payload, actors


# --------------------------------------------------------------------------------------------- lock

def save_and_lock(run, MC, actors, payload):
    """actors.msgpack (write-once), then the lock payload with every actor's parameter SHA-256; phase_main writes it as
    lock.json, before any return, points or score of any trained arm is computed."""
    from flax import serialization
    verified = MC.check_imports()
    k = run.key
    d = policy_dir(run.out_root, k.setting, k.world, k.seed)
    names = sorted(actors)
    file_sha = LG.write_bytes_once(d / "actors.msgpack", serialization.to_bytes({n: actors[n] for n in names}))
    return dict(payload, names=names, actor_sha256={n: MC.L.tree_sha256(actors[n]) for n in names},
                actors_file="actors.msgpack", actors_file_sha256=file_sha, verified_imports=verified,
                about=("the lock: every actor trained and every control matched in strength, written before any "
                       "return, points or score of any trained arm is computed"))


def lock_problems(lock, run, setting, seed, world, steps):
    """Why a version-9 lock cannot be used (empty: usable)."""
    problems = []
    if lock.get("kind") != "near80v9-policy-lock":
        problems.append("not a version-9 policy lock")
    if lock.get("mode") != run.mode:
        problems.append(f"written in mode {lock.get('mode')}, not {run.mode}")
    if run.official and lock.get("prereg_sha256") != run.prereg_sha256:
        problems.append("trained under another pre-registration")
    if lock.get("setting") != setting or lock.get("seed") != int(seed) or lock.get("world") != world:
        problems.append(f"lock is for {lock.get('setting')}/{lock.get('world')}/seed {lock.get('seed')}")
    if lock.get("steps") != int(steps):
        problems.append(f"not {steps} actor steps")
    if not all(bool(v.get("passed")) for v in (lock.get("checks") or {}).values() if isinstance(v, dict)):
        problems.append("a built-in check did not pass")
    if run.official and len(lock.get("cpus") or []) != 1:
        problems.append("not run on one processor")
    return problems


def verify_lock(run, setting, seed, world, steps):
    """(lock, actors, lock SHA-256): the lock's sidecar, lock_problems and every actor's SHA-256."""
    S, MC, _ = load_step2(run)
    d = policy_dir(run.out_root, setting, world, seed)
    lock = SD.load_record(d / "lock.json")
    problems = lock_problems(lock, run, setting, seed, world, steps)
    if problems:
        raise PH.Refused(f"refused: {d}/lock.json: {problems}")
    acts = SD.load_tree_checked(d / lock["actors_file"], {k: MC.L.actor_template() for k in lock["names"]},
                                lock["actors_file_sha256"])
    bad = [k for k in lock["names"] if MC.L.tree_sha256(acts[k]) != lock["actor_sha256"][k]]
    if bad:
        raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what=f"actors differ from the lock: {bad}"))
    return lock, acts, LG.file_sha256(d / "lock.json")


# --------------------------------------------------------------------------------------------- the phase

def run_policy(run, ctx):
    """The policy phase: the arm row, then train_world, then the lock payload (phase_main writes lock.json)."""
    S, MC, s2 = load_step2(run)
    k = run.key
    steps, _, _ = sizes(run)
    row = arm_row(k.setting, k.seed, k.world, run.out_root, S)
    stall_lock, stall_sha = None, None
    if k.world == "no_stall":
        stall_lock, _, stall_sha = verify_lock(run, k.setting, k.seed, "stall", steps)
    payload, actors = train_world(run, ctx, S, MC, row, stall_lock, stall_sha)
    payload["step2_code"] = s2
    return save_and_lock(run, MC, actors, payload)


def policy_stop_payload(run, stop):
    k = run.key
    return dict(kind="near80v9-policy-stop", setting=k.setting, world=k.world, seed=k.seed, group=k.group,
                mode=run.mode, reason=stop.reason, checks=stop.checks, detail=stop.detail)


def stage_lock(run):
    """The lock phase (kind near80v9-stage-lock): every policy phase of the group with its lock's SHA-256, its stop's
    SHA-256 or its skip reason. Written before any evaluation of the group; evaluate refuses unless it lists its lock's
    SHA-256."""
    group = run.key.group
    keys = PH.plan(spec=run.plan)
    locks = {}
    for k in [x for x in keys if x.kind == "policy" and x.group == group]:
        pid = PH.phase_id(k)
        st = PH.done(k, run.out_root)
        if st == "done":
            locks[pid] = dict(lock_sha256=LG.file_sha256(Path(run.out_root) / PH.record_rel(k)))
        elif st == "stopped":
            p = Path(run.out_root) / PH.stop_rel(k)
            if not p.is_file():
                p = Path(run.out_root) / PH.policy_stop_rel(k)
            locks[pid] = dict(stop_sha256=LG.file_sha256(p))
        else:
            why = PH.rule_skip(k, run.out_root, keys)
            if st != "skipped" and why is None:
                raise PH.Refused(f"refused: {pid} is not finished; the lock waits for every policy phase")
            rec = LG.ok_record(Path(run.out_root) / PH.skipped_rel(k)) or {}
            locks[pid] = dict(skipped=rec.get("reason") or why)
    return dict(kind="near80v9-stage-lock", group=group, locks=locks, plan=run.plan)
