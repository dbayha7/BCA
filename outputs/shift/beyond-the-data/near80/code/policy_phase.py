"""Near-80 (version 9, main setting): the actor-training phases (step 2's actor-only protocol, restated), the arm
table with the refusal rule, the two new arms, the noise-floor copy, and the lock. DRAFT, not frozen.

Step 2's code is imported, never edited: step2lib (after step2lib.verify_manifest() passes and step 2's code
manifest has the SHA-256 prereg.json pins) and machinery (which executes version 6's code and step 1's library from
verified bytes and refuses otherwise). Reused unchanged from machinery: RampRunner, ramp_arm, gap_rule_check,
check_imports, best_standin, v6_prereg, standin_matched, standin_locked, ramp_shape, info_only, log; and through it
the room harness (H) and version 6's policy code (P). Restated here, because step 2's versions call machinery.context
(which refuses every seed but 0-4) or write only under step 2's folder: train_world, locked_controls, reference,
h0_check, strengths, save_and_lock, verify_lock.

The arm table (arm_row_from). Both worlds' switch records (one certificate entry each, on the POOLED groups) give,
through step2lib.arm_table's own rules, the six step-2 ramp arms; then the new arms (PREREGISTRATION.md 5.3), every
one switched by the mean difference: on when mean(full throttle) - mean(rest) > 0.10 (MEANDIFF_MARGIN):
  meandiff_k1     h = the mean difference itself, k = 1 (version 8's reported arm; the bottom of the dose ladder)
  meandiff_q90x1  twin 1: h = the plain 90%-quantile gap of the same two groups (split conformal's statistic), k = 1
  meandiff_q90x2  twin 2: the same h, k = 2 (twice the plain gap)
  meandiff_limit  twin 3: on -> the room's hinge at the recorded range with coefficient 1.0 (the range_hinge arm's
                  own program, run alone); off -> the ramp with height 0 (checked equal to the no-caution actor)
  obca_k1_alone   BCA's switched ramp (obca_k1's switch and height) trained alone in a ramp program of one: the
                  noise floor's copy and C2's and C3's BCA side (section 9.1)
The twins never refuse: they do not read BCA's height. A plain gap that is not positive while the switch is on gives
a height of 0 (recorded in the row's 'why').
THE REFUSAL RULE (section 5.4). step2lib.arm_table refuses a switched-on arm whose height is not finite. Version 8
let that refusal stop the whole policy phase. Version 9 reads the refusal per arm: an arm whose switch is on with a
height that is not finite is REFUSED (recorded, never trained, its actor missing); every other arm trains and is
scored. The BCA-dependent arms are obca_k1, obca_k2 (BCA's height), obca_forced (the stalling world's BCA height, in
both worlds) and obca_k1_alone. A refused arm's place in step 2's six-arm program is filled by a pad (the ramp with
height 0, named pad_refused_<arm>, never saved or scored), so the other members keep step 2's batch size. With
obca_k1 refused there is no strength to match: the matched rivals and the stand-in are unmatched (null), as in
version 8 when BCA's switch stayed off. The look reads a refused arm as the no-caution actor in C1 and leaves the
seed out of C2 and C3 (pooled.py); nothing of the seed stops.

Step 2's ramp programs (ramp_layout; PREREGISTRATION.md sections 5.5 and 8.2): in the stalling world w0_check alone,
then the six step-2 ramp arms in one vmapped program with obca_k1 first; in the no-stall world one batch of seven,
w0_check first and obca_k1 second. Then, in extra calls that leave those programs' bits unchanged, the lone calls in
the fixed order K.LONE_CALLS (meandiff_k1, meandiff_q90x1, meandiff_q90x2, obca_k1_alone, each a ramp program of
one), then meandiff_limit (the hinge program, or a height-0 ramp). The reproduction check runs the same function on
seed 0 with step 1's own switch entries and must give step 2's saved w0_check, obca_k1 and range_hinge actors bit for
bit.
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
    mode step 2's code manifest and prespec must have the SHA-256 that prereg.json pins. step2lib's hard-coded
    repository root is set to this host's before machinery reads it (the cluster and the tests run elsewhere)."""
    if "S" not in _STEP2:
        if str(K.S2_DIR) not in sys.path:
            sys.path.insert(0, str(K.S2_DIR))
        import step2lib as S
        S.ROOT = Path(K.ROOT)
        man, man_sha = S.verify_manifest()
        spec, spec_sha = S.load_prespec()
        rebase_pinned_libraries(S, spec)
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


def _rebase(value, old, new):
    import dataclasses
    if isinstance(value, Path):
        try:
            rel = value.relative_to(old)
        except ValueError:
            return value
        return new / rel
    if isinstance(value, tuple):
        return tuple(_rebase(v, old, new) for v in value)
    if isinstance(value, list):
        return [_rebase(v, old, new) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            try:
                object.__setattr__(value, f.name, _rebase(getattr(value, f.name), old, new))
            except (AttributeError, TypeError):
                pass
        return value
    return value


def rebase_module(mod, old, new):
    """Every path of a module computed from its hard-coded repository root `old` (version 6's stage2_common.py and
    step 1's step1lib.py write ROOT = /home/user/bca) rebased onto this host's root `new`. The bytes executed are
    still the pinned ones (step2lib.exec_verified checked them); only where they point changes."""
    old, new = Path(old), Path(new)
    for k, v in list(vars(mod).items()):
        if k.startswith("__"):
            continue
        nv = _rebase(v, old, new)
        if nv is not v:
            setattr(mod, k, nv)
    return mod


def rebase_pinned_libraries(S, spec):
    """On a host other than the desktop (the cluster, the tests): execute version 6's stage2_common.py and step 1's
    step1lib.py from their verified bytes under the names machinery will ask for (step2lib.exec_verified returns a
    module already loaded under its name when its SHA-256 is the pinned one), with their paths rebased onto K.ROOT,
    so that machinery, policy.py and the room harness read this host's clone. On the desktop nothing is done."""
    if Path(K.ROOT) == Path(K.DESKTOP_ROOT):
        return None
    inputs = spec["inputs"]
    v6 = inputs["v6_code"]
    old, new = Path(K.DESKTOP_ROOT), Path(K.ROOT)
    C = S.exec_verified("stage2_common", new / v6["dir"] / "stage2_common.py", v6["sha256"]["stage2_common.py"])
    rebase_module(C, old, new)
    S1 = S.exec_verified("step1lib", new / inputs["step1lib"]["path"], inputs["step1lib"]["sha256"])
    rebase_module(S1, old, new)
    return C, S1


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
    """The arm-table row of one seed and world: step2lib.arm_table's rules on a step-1-like record built from BOTH
    worlds' switch records (obca_forced uses the stalling world's height), applied per arm with the refusal rule,
    then the new arms."""
    S = S or load_step2()[0]
    per = {}
    for w in K.WORLDS:
        rec = SD.load_record(SD.learner_dir(out_root, setting, w, seed) / "switch.json")
        per[f"{w}/seed{int(seed)}"] = rec["entry"]
    return arm_row_from(setting, seed, world, per, S)


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


def step2_arms(S, per, seed, world):
    """step2lib.arm_table's six arms from the entries of both worlds, computed arm by arm with arm_table's own
    rules, so that one refused arm does not hide the others. Returns (arms {name: {switch, h, k, refused, why}},
    sources, t_lo, t_up)."""
    e = per[f"{world}/seed{int(seed)}"]
    stall = per[f"stall/seed{int(seed)}"]
    rule = dict(K.ARM_RULE)
    c, g, rt = e["certificate"], e["groups"], e["rank_test"]
    on_bca, h_bca = bool(c["on"]), S.num(c["height"])
    stall_h = S.num(stall["certificate"]["height"])
    gap90 = S.num(g["q90_full_throttle"]) - S.num(g["q90_rest"])
    on_conf = bool(gap90 > float(rule["conformal_margin"]))
    p = S.num(rt["p"])
    on_rank = bool(p < float(rule["rank_p"]))
    meangap = S.num(g["mean_full_throttle"]) - S.num(g["mean_rest"])
    arms = dict(
        obca_k1=dict(switch=on_bca, h=h_bca if on_bca else 0.0, k=1.0),
        obca_k2=dict(switch=on_bca, h=h_bca if on_bca else 0.0, k=2.0),
        obca_forced=dict(switch=True, h=stall_h, k=1.0),
        fixed_gated=dict(switch=on_bca, h=float(rule["fixed_height"]) if on_bca else 0.0, k=1.0),
        conformal_k1=dict(switch=on_conf, h=gap90 if on_conf else 0.0, k=1.0),
        meangap_k1=dict(switch=on_rank, h=max(meangap, 0.0) if on_rank else 0.0, k=1.0))
    for name, a in arms.items():
        a["refused"], a["why"] = False, None
        if a["switch"] and not np.isfinite(a["h"]):
            a["refused"] = True
            a["why"] = (f"switched on but its height {a['h']} is not finite (the full-throttle group was not "
                        f"certified)")
            a["h"] = None
        elif a["h"] < 0:
            a["refused"], a["why"], a["h"] = True, f"a negative height {a['h']}", None
    sources = dict(bca_on=on_bca, bca_height=h_bca, stall_bca_height=stall_h, conformal_gap=gap90,
                   plain_gap_information_only=S.num(c.get("plain_gap", "nan")), rank_p=p, mean_gap=meangap,
                   n_full_throttle=g.get("n_full_throttle"), n_rest=g.get("n_rest"))
    return arms, sources, S.num(e["t_lo"]), S.num(e["t_up"])


def arm_row_from(setting, seed, world, per, S):
    """arm_row on the switch entries {'<world>/seed<k>': entry} of both worlds (the reproduction check passes the
    entries it computed). Every arm is recorded; a refused one carries refused: true and h None."""
    try:
        arms, sources, t_lo, t_up = step2_arms(S, per, seed, world)
    except (ValueError, TypeError, KeyError) as error:
        raise PH.PhaseStop(PH.STOP_REASONS["check"], dict(what="the arm table cannot be read from the switches",
                                                          error=str(error))) from None
    if not (np.isfinite(t_lo) and np.isfinite(t_up)):
        raise PH.PhaseStop(PH.STOP_REASONS["check"], dict(what="the gap rule's ends are not finite"))
    gap_mean = float(sources["mean_gap"])
    on = bool(np.isfinite(gap_mean) and gap_mean > K.MEANDIFF_MARGIN)
    arms["meandiff_k1"] = dict(switch=on, h=gap_mean if on else 0.0, k=1.0, refused=False, why=None)
    gap90 = float(sources["conformal_gap"])
    h1 = max(gap90, 0.0) if np.isfinite(gap90) else 0.0
    why90 = None if (not on or h1 > 0) else f"the plain 90% gap {gap90} was not positive; height 0"
    arms["meandiff_q90x1"] = dict(switch=on, h=h1 if on else 0.0, k=1.0, refused=False, why=why90)
    arms["meandiff_q90x2"] = dict(switch=on, h=h1 if on else 0.0, k=2.0, refused=False, why=why90)
    arms["meandiff_limit"] = dict(switch=on, h=1.0 if on else 0.0, k=1.0, refused=False, why=None,
                                  kind="hinge" if on else "ramp_h0")
    o = arms[K.PRIMARY]
    arms[K.NOISE_ARM] = dict(switch=o["switch"], h=o["h"], k=1.0, refused=o["refused"], why=o["why"])
    refused = sorted(n for n, a in arms.items() if a["refused"])
    if any(n not in K.BCA_DEPENDENT for n in refused):
        raise PH.PhaseStop(PH.STOP_REASONS["check"], dict(what="an arm that does not depend on BCA's height was "
                                                               "refused", refused=refused))
    return dict(t_lo=t_lo, t_up=t_up, arms=arms, sources=sources, refused=refused,
                bca_refused=bool(K.PRIMARY in refused), meandiff_on=on,
                rule="the refusal rule: a refused arm is left out; every other arm trains and is scored")


def ramp_names(setting):
    """The ramp arms of step 2's programs that are kept and scored."""
    return K.STEP2_RAMP_ARMS


def ramp_layout(setting, world):
    """The vmapped ramp programs of the main RampRunner, in call order: [(arm names), ...]. Step 2's own layout:
    stalling world [(w0_check,), (the six,)]; no-stall world [(w0_check, the six)]. Then the lone calls, each alone,
    in K.LONE_CALLS' order: (meandiff_k1,), (meandiff_q90x1,), (meandiff_q90x2,), (obca_k1_alone,). meandiff_limit
    runs last, on the room's own hinge program or as a height-0 ramp (train_world)."""
    six = tuple(K.STEP2_RAMP_ARMS)
    calls = [(K.W0,), six] if world == "stall" else [(K.W0,) + six]
    calls += [(n,) for n in K.LONE_CALLS]
    return calls


def pad_name(name):
    return f"pad_refused_{name}"


def layout_arm(MC, row, name):
    """The ramp arm of a name of ramp_layout (w0_check is height 0, k 0; a refused arm's place is a height-0 pad)."""
    if name == K.W0:
        return MC.ramp_arm(K.W0, 0.0, 0.0)
    a = row["arms"][name]
    if a.get("refused"):
        return MC.ramp_arm(pad_name(name), 0.0, 0.0)
    return MC.ramp_arm(name, a["h"], a["k"])


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
    """Every arm of the main setting for one seed and world (module docstring); returns the lock payload (without
    the actor hashes, which save_and_lock adds)."""
    t_start = time.perf_counter()
    H, L, P = MC.H, MC.L, MC.P
    key = run.key
    setting, world, seed = key.setting, key.world, int(key.seed)
    steps, chunk, full = sizes(run)
    refused = list(row.get("refused") or [])
    bca_refused = bool(row.get("bca_refused"))
    base = dict(setting=setting, world=world, seed=seed, group=key.group, mode=run.mode, steps=steps, chunk=chunk,
                full_length=full, arm_table=row, stall_lock_sha256=stall_lock_sha, refused_arms=refused,
                bca_refused=bca_refused)
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
    g, scale = P.room_scales(pg, critic, shared, clone, seed, cfg, nets) if world == "stall" else (None, None)
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
    extra_seconds = {}
    t2 = time.perf_counter()
    for call in rest:                                    # the lone calls in K.LONE_CALLS' order (a refused one skipped)
        members = [n for n in call if not row["arms"][n].get("refused")]
        if members:
            trained.update(runner.run([layout_arm(MC, row, n) for n in members], MC.log))
    lim = row["arms"]["meandiff_limit"]
    if lim.get("kind") == "hinge":                       # the hand-set range limit's own program, run alone, last
        trained.update(runner.run([H.RoomArm("meandiff_limit", "hinge", (1.0,))], MC.log))
    else:
        trained.update(runner.run([MC.ramp_arm("meandiff_limit", 0.0, 0.0)], MC.log))
    extra_seconds["new_arms"] = time.perf_counter() - t2
    padding = None
    pads = [n for n in trained if n.startswith("pad_refused_")]
    if pads:
        padding = dict(arms=pads, why="the places of refused arms in step 2's six-arm program; never saved or scored")
        for n in pads:
            trained.pop(n)
    zero = [n for n in names + tuple(K.NEW_ARMS)
            if n in trained and row["arms"][n].get("h") == 0.0 and not (n == "meandiff_limit" and
                                                                        lim.get("kind") == "hinge")]
    checks["h0_reproduces_reference"] = h0_check(MC, trained, zero, ref, ref_name)
    if not (checks["b_w0"]["passed"] and checks["h0_reproduces_reference"]["passed"]):
        stop(run, base, PH.STOP_REASONS["height0"], checks)
    seconds["arms"] = time.perf_counter() - t1
    # ---- matching (stalling world) or the stalling world's locked coefficients (no-stall world)
    t1 = time.perf_counter()
    matches, standin, s_actors, s_record = None, None, {}, {}
    target = None
    if world == "stall":
        if K.PRIMARY in trained and trained[K.PRIMARY]["finite"]:
            target = float(L.strength(trained[K.PRIMARY]["params"], none, pg.eval_obs))
        else:
            target = float("nan")
        if np.isfinite(target) and target > 0:
            matches = P.match_controls(runner, trained, none, pg, target, scale, P.MATCHED, MC.log)
        else:
            why = ("BCA refused on this seed: no strength to match" if bca_refused
                   else "obca_k1 failed or has no positive strength")
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
            standin["reason"] = "BCA refused on this seed: no strength to match" if bca_refused else \
                "obca_k1 failed or has no positive strength"
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
    shape_arms = {n: (row["arms"][n]["h"], row["arms"][n]["k"]) for n in names + tuple(K.LONE_CALLS) if n in trained}
    shape = MC.info_only("ramp shape", lambda: MC.ramp_shape(pg, critic, none, seed, row["t_lo"], row["t_up"],
                                                            shape_arms))
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
                   locked_controls=locked_info or None, standin=standin, arm_ids=ids,
                   scored_names=scored, finite={k: finite[k] for k in scored},
                   slope_rms=g, coefficient_per_knob=scale, random_surface_seeds={j: nets[j][1] for j in (1, 2)},
                   shared=sinfo, ramp_shape=shape, chunk_seconds=runner.chunk_seconds,
                   ramp_programs=[list(c) for c in layout], ramp_padding=padding,
                   meandiff_limit_kind=lim.get("kind"),
                   seconds=dict(seconds, extra=extra_seconds, by_kind=runner.seconds,
                                total=time.perf_counter() - t_start))
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
    """The policy phase: the arm row (with the refusal rule), then train_world, then the lock payload (phase_main
    writes lock.json)."""
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


def stage_lock(run):
    """The lock phase (kind near80-stage-lock): every policy phase of the group with its lock's SHA-256, its stop's
    SHA-256 or its skip reason. Written before any evaluation of the group; evaluate refuses unless it lists the
    lock's SHA-256."""
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
    return dict(kind="near80-stage-lock", group=group, locks=locks, plan=run.plan)
