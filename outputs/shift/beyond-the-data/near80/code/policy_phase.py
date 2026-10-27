"""Near-80 (version 8): the actor-training phases (step 2's actor-only protocol, restated for near-80), and the lock.

Step 2's code is imported, never edited: step2lib (after step2lib.verify_manifest() passes and step 2's code
manifest has the SHA-256 prereg.json pins) and machinery (which executes version 6's code and step 1's library from
verified bytes and refuses otherwise). Reused unchanged from machinery: RampRunner, ramp_arm, gap_rule_check,
check_imports, best_standin, v6_prereg, standin_matched, standin_locked, ramp_shape, info_only, log; and through it
the room harness (H: ProgramConfig, build_shared, random_surface_params, RoomArm, RoomRunner, family_arm, the ladders,
max_param_diff, region_strength, MAIN_BAND) and version 6's policy code (P: room_scales, match_controls, strengths_of,
arms_record, standins_module, MATCHED). Restated here, because step 2's versions call machinery.context (which refuses
every seed but 0-4 and reads only the gate's saved artifacts) or write only under step 2's folder: train_stall,
train_no_stall (one function, train_world), locked_controls, reference, h0_check, strengths, save_and_lock,
verify_lock. They take a settingdata.SettingContext and write under the near-80 output root.

Step 2's ramp programs in every setting (ramp_layout; PREREGISTRATION.md sections 5.3 and 8.2). machinery.RampRunner
compiles one vmapped program per number of ramp arms, and a member's bits depend on that number, so BCA's switched ramp
and the plumbing check run in exactly step 2's programs in every setting and world: in the stalling world w0_check
alone, then a batch of six ramp arms with obca_k1 first; in the no-stall world one batch of seven, w0_check first and
obca_k1 second. The band costs, the noise floor and the reproduction check then compare actors from one program.
  near-80, stalling world: w0_check alone first (version 6's check (b)); the six step-2 ramp arms in one vmapped
      program at the setting's t_lo, t_up; meandiff_k1 in a program of its own; knee_ramp in a second RampRunner
      starting at the stall (KNEE32); perfect_band; range_hinge; knee_limit (the hinge with its limits at -+KNEE32);
      the controls' ladders and version 6's matcher blind to obca_k1's strength; the stand-in matched; version6_bca
      (version 6's paired lower edge, k = 1, on this setting's version-6 band).
  near-80, no-stall world: step 2's no-stall arm set (w0_check and the six step-2 ramp arms in one program) plus
      meandiff_k1 (a program of its own), knee_ramp, knee_limit and version6_bca; the controls and the stand-in at the
      stalling world's locked coefficients.
  random rows, both worlds: w0_check, obca_k1 and range_hinge only; references none and clone; unmatched. obca_k1
      trains in step 2's six-arm batch, the other five places filled with copies of it (pad_ramp_1 .. pad_ramp_5, its
      own switch and height); the copies are checked equal to it, recorded, and never saved or scored.
  main setting, seeds 10-14: exactly step 2's train_stall / train_no_stall arm sets and programs.
Every height-0 ramp arm must reproduce the no-caution actor within W0_TOL (else a recorded stop). The lock (every
actor's SHA-256) is written before any return, points or score of any trained arm is computed.
"""
import sys

sys.dont_write_bytecode = True

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
    """The arm-table row of one setting, seed and world: step2lib.arm_table on a step-1-like record built from BOTH
    worlds' switch records (obca_forced uses the stalling world's height), then the new arms: meandiff_k1 (switch:
    mean full-throttle overestimate minus mean rest > 0.10; height that difference) and, in the near-80 setting,
    knee_ramp (obca_k1's switch and height, ramp from KNEE32 to the setting's t_up). arm_table's refusals (a switched-on
    arm with a non-finite height, a negative height) and a missing gap rule are recorded stops."""
    S = S or load_step2()[0]
    per = {}
    for w in K.WORLDS:
        rec = SD.load_record(SD.learner_dir(out_root, setting, w, seed) / "switch.json")
        per[f"{w}/seed{int(seed)}"] = rec["entry"]
    return arm_row_from(setting, seed, world, per, S)


def arm_row_from(setting, seed, world, per, S):
    """arm_row on the switch entries {'<world>/seed<k>': entry} of both worlds (the reproduction check passes the
    entries it computed)."""
    try:
        row = S.arm_table({"per_seed_world": per}, seeds=[int(seed)], rule=K.ARM_RULE)[world][int(seed)]
    except (SystemExit, ValueError, TypeError, KeyError) as error:
        if setting != "random_rows":
            raise PH.PhaseStop(PH.STOP_REASONS["check"], dict(what="the arm table cannot be built from the switches",
                                                              error=str(error))) from None
        row = primary_row(S, per[f"{world}/seed{int(seed)}"], str(error))
    gap_mean = float(row["sources"]["mean_gap"])
    on = bool(np.isfinite(gap_mean) and gap_mean > K.MEANDIFF_MARGIN)
    row["arms"]["meandiff_k1"] = dict(switch=on, h=gap_mean if on else 0.0, k=1.0)
    if setting == "near80":
        o = row["arms"][K.PRIMARY]
        row["arms"]["knee_ramp"] = dict(switch=o["switch"], h=o["h"], k=1.0, t_lo=K.KNEE32, t_up=row["t_up"])
    return row


def primary_row(S, entry, why):
    """The random-rows setting trains BCA's switched ramp only. When step2lib.arm_table refuses because of an arm the
    random-rows setting does not train (for example the always-on ramp, whose height is the stalling world's), the
    row of obca_k1 alone is read from the same entry with arm_table's own rule for obca_k1: switch = the certificate's
    'on', height = its height when on, else 0, k = 1; switched on with a height that is not finite is a stop."""
    try:
        c = entry["certificate"]
        on, h = bool(c["on"]), S.num(c["height"])
        t_lo, t_up = S.num(entry["t_lo"]), S.num(entry["t_up"])
    except (KeyError, ValueError, TypeError) as error:
        raise PH.PhaseStop(PH.STOP_REASONS["check"], dict(what="the switch cannot be read", error=str(error))) from None
    if on and not np.isfinite(h):
        raise PH.PhaseStop(PH.STOP_REASONS["check"], dict(what="BCA's switch is on but its height is not finite",
                                                          error=why))
    g = entry.get("groups") or {}
    try:
        gap = S.num(g.get("mean_full_throttle")) - S.num(g.get("mean_rest"))
    except ValueError:
        gap = float("nan")
    return dict(t_lo=t_lo, t_up=t_up, arms={K.PRIMARY: dict(switch=on, h=h if on else 0.0, k=1.0)},
                sources=dict(bca_on=on, bca_height=h, mean_gap=gap, arm_table_refusal=why))


PAD_RAMP_ARMS = tuple(f"pad_ramp_{i}" for i in range(1, len(K.STEP2_RAMP_ARMS)))


def ramp_names(setting):
    """The setting's ramp arms that are kept and scored (on the setting's own ramp, t_lo to t_up)."""
    if setting == "random_rows":
        return (K.PRIMARY,)
    if setting == "near80":
        return K.STEP2_RAMP_ARMS + ("meandiff_k1",)
    return K.STEP2_RAMP_ARMS


def ramp_layout(setting, world):
    """The vmapped ramp programs of the setting's main RampRunner, in call order: [(arm names), ...]. Step 2's own
    layout in every setting (module docstring): stalling world [(w0_check,), (the six,)]; no-stall world
    [(w0_check, the six)]. The random-rows setting's six places are obca_k1 and its five copies; the near-80
    setting's meandiff_k1 runs last, alone. (knee_ramp runs on a RampRunner of its own.)"""
    six = (K.PRIMARY,) + PAD_RAMP_ARMS if setting == "random_rows" else tuple(K.STEP2_RAMP_ARMS)
    calls = [(K.W0,), six] if world == "stall" else [(K.W0,) + six]
    if setting == "near80":
        calls.append(("meandiff_k1",))
    return calls


def layout_arm(MC, row, name):
    """The ramp arm of a name of ramp_layout (a pad copies obca_k1's height and k; w0_check is height 0, k 0)."""
    if name == K.W0:
        return MC.ramp_arm(K.W0, 0.0, 0.0)
    src = K.PRIMARY if name in PAD_RAMP_ARMS else name
    return MC.ramp_arm(name, row["arms"][src]["h"], row["arms"][src]["k"])


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


def knee_shared(shared):
    """The room's shared arrays with the hinge's limits moved to the stall: lo = -KNEE32, hi = +KNEE32 on every
    thruster (float32)."""
    import jax.numpy as jnp
    return dict(shared, lo=jnp.asarray(np.full(4, -K.KNEE32), jnp.float32),
                hi=jnp.asarray(np.full(4, K.KNEE32), jnp.float32))


def stop(run, payload, reason, checks=None):
    raise PH.PhaseStop(reason, dict(payload, checks=checks), checks)


# --------------------------------------------------------------------------------------------- one world

def train_world(run, ctx, S, MC, row, stall_lock=None, stall_lock_sha=None):
    """Every arm of one setting, seed and world (module docstring); returns the lock payload (without the actor
    hashes, which save_and_lock adds)."""
    t_start = time.perf_counter()
    H, L, P = MC.H, MC.L, MC.P
    key = run.key
    setting, world, seed = key.setting, key.world, int(key.seed)
    steps, chunk, full = sizes(run)
    matched = setting in ("near80", "main")
    near80 = setting == "near80"
    base = dict(setting=setting, world=world, seed=seed, group=key.group, mode=run.mode, steps=steps, chunk=chunk,
                full_length=full, arm_table=row, stall_lock_sha256=stall_lock_sha)
    t0 = time.perf_counter()
    pg = ctx.prepared(world)
    critic = ctx.critic(world)
    clone = ctx.clone(world)
    none = ctx.none(world)
    full_none = ctx.full_none() if (world == "stall" and matched) else None
    checks = dict(gap_rule=MC.gap_rule_check(pg, row))
    if not checks["gap_rule"]["passed"]:
        stop(run, base, "step 1's gap rule gives another t_lo or t_up on this data", checks)
    seconds = dict(load=time.perf_counter() - t0)
    t1 = time.perf_counter()
    cfg = H.ProgramConfig() if world == "stall" else H.ProgramConfig(physics=pg.physics, next_command="stall_aware")
    shared, sinfo = H.build_shared(pg, critic, seed, cfg=cfg, need_harm=False, log_fn=MC.log)
    nets = {j: H.random_surface_params(seed, j) for j in (1, 2)}
    g, scale = (P.room_scales(pg, critic, shared, clone, seed, cfg, nets) if (world == "stall" and matched)
                else (None, None))
    seconds["shared_and_scales"] = time.perf_counter() - t1
    runner = MC.RampRunner(pg, clone, critic, ctx.learn, seed, cfg=cfg, shared=shared,
                           random_nets={j: nets[j][0] for j in (1, 2)}, steps=steps, chunk=chunk,
                           t_lo=row["t_lo"], t_up=row["t_up"])
    names = ramp_names(setting)
    layout = ramp_layout(setting, world)
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
        if matched:
            arms += [H.RoomArm("perfect_band", "band", H.MAIN_BAND), H.RoomArm("range_hinge", "hinge", (1.0,))]
            arms += [H.family_arm("even_pull", nu, scale) for nu in H.EVEN_LADDER]
            arms += [H.family_arm(f, k, scale) for f in ("push_less", "random_1", "random_2") for k in H.SLOPE_LADDER]
        else:
            arms += [H.RoomArm("range_hinge", "hinge", (1.0,))]
        trained.update(runner.run(arms, MC.log))
        rest = layout[2:]
    else:
        locked, locked_info = locked_controls(MC, stall_lock) if matched else ([], {})
        arms = [layout_arm(MC, row, n) for n in layout[0]]                         # step 2's seven-arm program
        if matched:
            arms += [H.RoomArm("perfect_band", "band", H.MAIN_BAND), H.RoomArm("range_hinge", "hinge", (1.0,))] + locked
        else:
            arms += [H.RoomArm("range_hinge", "hinge", (1.0,))]
        trained = runner.run(arms, MC.log)
        ref, ref_name = reference(MC, runner, trained, none, full)
        w0 = H.max_param_diff(trained[K.W0]["params"], ref)
        checks["b_w0"] = dict(max_abs_diff=w0, tol=MC.W0_TOL, reference=ref_name,
                              passed=bool(trained[K.W0]["finite"] and w0 <= MC.W0_TOL))
        rest = layout[1:]
    for call in rest:                                                              # meandiff_k1 alone (near-80)
        trained.update(runner.run([layout_arm(MC, row, n) for n in call], MC.log))
    padding = None
    if any(n in trained for n in PAD_RAMP_ARMS):
        same = {n: float(H.max_param_diff(trained[n]["params"], trained[K.PRIMARY]["params"])) for n in PAD_RAMP_ARMS}
        padding = dict(arms=list(PAD_RAMP_ARMS), copies_of=K.PRIMARY, max_abs_diff=same,
                       equal=bool(all(v == 0.0 for v in same.values())),
                       why="the other five places of step 2's six-arm program; never saved or scored")
        for n in PAD_RAMP_ARMS:
            trained.pop(n)
    extra_seconds = {}
    if near80:
        kr = row["arms"]["knee_ramp"]
        if not float(row["t_up"]) > K.KNEE32:
            stop(run, base, "the ramp started at the stall has no length: t_up is not above 80%", checks)
        t2 = time.perf_counter()
        knee_runner = MC.RampRunner(pg, clone, critic, ctx.learn, seed, cfg=cfg, shared=shared,
                                    random_nets={j: nets[j][0] for j in (1, 2)}, steps=steps, chunk=chunk,
                                    t_lo=K.KNEE32, t_up=row["t_up"])
        trained.update(knee_runner.run([MC.ramp_arm("knee_ramp", kr["h"], kr["k"])], MC.log))
        limit_runner = H.RoomRunner(pg, clone, critic, ctx.learn, seed, cfg=cfg, shared=knee_shared(shared),
                                    random_nets={j: nets[j][0] for j in (1, 2)}, steps=steps, chunk=chunk)
        trained.update(limit_runner.run([H.RoomArm("knee_limit", "hinge", (1.0,))], MC.log))
        extra_seconds.update(knee=time.perf_counter() - t2, knee_by_kind=knee_runner.seconds,
                             limit_by_kind=limit_runner.seconds)
        t2 = time.perf_counter()
        v6 = dict(arm=None, band_failed=None)
        band, brec = _load_v6_band(run, setting, world, seed)
        v6["band_failed"] = bool(band is None)
        if band is not None:
            import v6bca
            got, _ = v6bca.train_arm(pg, clone, critic, ctx.learn, seed, band, cfg, shared, steps, chunk, MC.log)
            trained[v6bca.V6_ARM] = got
            v6["arm"] = v6bca.V6_ARM if got["finite"] else None
        else:
            v6["reason"] = brec.get("reason")
        extra_seconds["version6_bca"] = time.perf_counter() - t2
    else:
        v6 = None
    zero = [n for n in names if row["arms"][n]["h"] == 0.0]
    if near80 and row["arms"]["knee_ramp"]["h"] == 0.0:
        zero.append("knee_ramp")
    checks["h0_reproduces_reference"] = h0_check(MC, trained, zero, ref, ref_name)
    if not (checks["b_w0"]["passed"] and checks["h0_reproduces_reference"]["passed"]):
        stop(run, base, PH.STOP_REASONS["height0"], checks)
    seconds["arms"] = time.perf_counter() - t1
    # ---- matching (stalling world) or the stalling world's locked coefficients (no-stall world)
    t1 = time.perf_counter()
    matches, standin, s_actors, s_record = None, None, {}, {}
    target = None
    if matched and world == "stall":
        target = float(L.strength(trained[K.PRIMARY]["params"], none, pg.eval_obs)) if trained[K.PRIMARY]["finite"] \
            else float("nan")
        if np.isfinite(target) and target > 0:
            matches = P.match_controls(runner, trained, none, pg, target, scale, P.MATCHED, MC.log)
        else:
            matches = {f: dict(record=None, arm=None, matched=False,
                               reason="obca_k1 failed or has no positive strength") for f in P.MATCHED}
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
    elif matched:
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
    # ---- logical ids -> actor names
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
        elif logical == "version6_bca":
            ids[logical] = (v6 or {}).get("arm")
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
    payload = dict(base, kind="near80-policy-lock", checks=checks,
                   arms=dict(P.arms_record(trained), **s_record), strengths=S_, region_strengths=SR,
                   target_strength=target, matches=matches,
                   matched_names=None if matches is None else {f: m["arm"] for f, m in matches.items()},
                   locked_controls=locked_info or None, standin=standin, version6_bca=v6, arm_ids=ids,
                   scored_names=scored, finite={k: finite[k] for k in scored},
                   slope_rms=g, coefficient_per_knob=scale, random_surface_seeds={j: nets[j][1] for j in (1, 2)},
                   shared=sinfo, ramp_shape=shape, chunk_seconds=runner.chunk_seconds,
                   ramp_programs=[list(c) for c in layout], ramp_padding=padding,
                   seconds=dict(seconds, extra=extra_seconds, by_kind=runner.seconds,
                                total=time.perf_counter() - t_start))
    return payload, actors


def _load_v6_band(run, setting, world, seed):
    import learner_phases as LP
    return LP.load_v6_band(run.out_root, setting, world, seed)


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
    """Why a near-80 lock cannot be used (empty: usable)."""
    problems = []
    if lock.get("kind") != "near80-policy-lock":
        problems.append("not a near-80 policy lock")
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
    return dict(kind="near80-policy-stop", setting=k.setting, world=k.world, seed=k.seed, group=k.group,
                mode=run.mode, reason=stop.reason, checks=stop.checks, detail=stop.detail)


def step2_saved_root(run):
    if run.stub:
        import stubs
        return stubs.step2_root(run.fixture)
    return K.S2_OUT


def verify_step2_saved_locks(run, seeds=K.EXPLORATORY):
    """Step 2's saved locks of seeds 0-4 (both worlds), read-only: each lock's sidecar, step2lib.lock_problems with
    step 2's own prespec and code manifest, and every actor's SHA-256. Returns {world/seed<k>: hashes}."""
    S, MC, s2 = load_step2(run)
    root = step2_saved_root(run)
    out = {}
    for s in seeds:
        for w in K.WORLDS:
            d = Path(root) / "policy" / f"seed{s}" / w
            try:
                lock = LG.read_json_verified(d / "lock.json")
            except LG.RecordError as error:
                raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what=f"step 2's lock {d}",
                                                                        error=str(error))) from None
            problems = S.lock_problems(lock, s2["prespec_sha256"], s2["code_manifest_sha256"], s, w, True,
                                       K.ACTOR_STEPS)
            if problems:
                raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what=f"step 2's lock {d}", problems=problems))
            acts = SD.load_tree_checked(d / lock["actors_file"], {k: MC.L.actor_template() for k in lock["names"]},
                                        lock["actors_file_sha256"])
            bad = [k for k in lock["names"] if MC.L.tree_sha256(acts[k]) != lock["actor_sha256"][k]]
            if bad:
                raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what=f"step 2's actors {d}", actors=bad))
            out[f"{w}/seed{s}"] = dict(lock_sha256=LG.file_sha256(d / "lock.json"),
                                       actors_file_sha256=lock["actors_file_sha256"])
    return out


def stage_lock(run):
    """The lock phase (kind near80-stage-lock): every policy phase of the group with its lock's SHA-256, its stop's
    SHA-256 or its skip reason, and (exploratory) step 2's saved locks of seeds 0-4 verified. Written before any
    evaluation of the group; evaluate and readouts refuse unless it lists their lock's SHA-256."""
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
    rec = dict(kind="near80-stage-lock", group=group, locks=locks, plan=run.plan)
    if group == "exploratory":
        rec["step2_saved"] = verify_step2_saved_locks(run, K.EXPLORATORY)
    return rec
