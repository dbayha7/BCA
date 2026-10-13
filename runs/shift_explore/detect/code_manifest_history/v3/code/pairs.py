"""EXPLORATORY detection rehearsal (plan section 4.4), NO BCA: the locked pair sets T1-T4 of one seed and world.

Built from the seed's gate artifacts only (the world's training log, the world's gapped critic's actor = check 2's
saved 'gapped' / 'no_stall_gapped' actor, the world's saved cloned actor, exact rollouts from the strong-current
bank), before any signal or label exists. No signal and no label is computed here.

Every state is stored as the float64 of a float32 number (what the environment returns and every critic and actor
sees), so the signal script, the label script and Stage 2's BCA width all read exactly the same state. Commands are
float32. 'cap' is the gate's 80% cap: float32(clip(float64(command), -0.8, 0.8)) (acceptance's c_cap rule;
float32(0.8) is the knee itself, so a cap is never past 80%).

  T1  1,500 full-throttle-region training states (prepared.region), chosen by default_rng([20261015, seed, 101])
      among the world's region rows; roles: command (the world's gapped critic's actor's command), cap, recorded (the
      row's own command), cloned (the world's cloned actor's command), knee (the command clipped at 81%:
      float32(clip(float64(command), -0.81, 0.81)); an addition, see the spec's deviations).
  T2  1,500 other training states (outside the region), default_rng([20261015, seed, 102]); roles command, cap,
      recorded, cloned.
  T3  1,500 strong-current states visited by the world's gapped critic's actor (T3a) and 1,500 visited by the
      stall-aware controller (T3b): exact float64 rollouts (truth.rollout) in the world's physics from the first 200
      starts of the strong-current bank, 100 steps each (20,000 visited states s_t, t = 0..99, per policy), chosen by
      default_rng([20261015, seed, 103, 0]) and ([.., 103, 1]); the state is rounded to float32; roles command (the
      gapped critic's actor's command at that state) and cap. The actor is rolled out as the gate evaluates it
      (learners.mlp_policy: the same network in float64 on float32-rounded observations); the pair's command is the
      float32 network's command at the stored state.
  T4  5,000 held-out recorded rows at their recorded commands: the calibration split's rows with the gapped log's
      removal rule applied (gap.contains) and the calibration-bank rows left out, default_rng([20261015, seed, 81])
      (the stand-ins' held-out rule, sections 4.2 and 4.3; the same pool that plan section 3 names for tau).

'ref' per pair: the reference command the re-referenced (change A1) forms read at the pair's state: the recorded
command at T1, T2 and T4 states; at T3 states (no recorded command exists) the cloned actor's command at the state.
"""

import numpy as np

import common as C

PAIR_SETS = ("T1", "T2", "T3", "T4")
ROLES = ("command", "cap", "recorded", "cloned", "knee")
SET_ROLES = {"T1": ("command", "cap", "recorded", "cloned", "knee"), "T2": ("command", "cap", "recorded", "cloned"),
             "T3": ("command", "cap"), "T4": ("recorded",)}
# The plan's roles (plan 4.4's table). Only these enter the ranking and false-flag pools; the 81% knee pairs of T1
# (spec D3, an addition) enter only the knee measures (spec change C2).
PLAN_ROLES = {"T1": ("command", "cap", "recorded", "cloned"), "T2": ("command", "cap", "recorded", "cloned"),
              "T3": ("command", "cap"), "T4": ("recorded",)}
N_T1, N_T2, N_T3_EACH, N_T4 = 1500, 1500, 1500, 5000
T3_STARTS, T3_SITUATION = 200, "strong_current"
TAG_T1, TAG_T2, TAG_T3, TAG_HELD = 101, 102, 103, 81
CAP, KNEE_PLUS = 0.8, 0.81
FILE = "pairs.npz"
RECORD = "pairs.json"


def cap80(commands):
    return np.clip(np.asarray(commands, np.float32).astype(np.float64), -CAP, CAP).astype(np.float32)


def cap81(commands):
    return np.clip(np.asarray(commands, np.float32).astype(np.float64), -KNEE_PLUS, KNEE_PLUS).astype(np.float32)


def held_out_rows(sd, rep):
    """T4's rows: the calibration split, the gap removed (the gapped log's removal rule), the calibration-bank rows
    left out, 5,000 drawn by default_rng([20261015, seed, 81]) (standins.held_out_rows' rule). Returns (log rows,
    float64 states of the float32 observations, float32 commands, pool size, pool rows)."""
    lg = sd.log
    cal = lg.split_rows("calibration")
    cal = cal[~sd.gap.contains(lg["actions"][cal])]
    held = np.setdiff1d(cal, lg.bank_rows)
    pick = np.sort(C.rng(rep, TAG_HELD).choice(len(held), min(N_T4, len(held)), replace=False))
    rows = held[pick]
    states = np.asarray(lg["observations"][rows], np.float32).astype(np.float64)
    return rows, states, np.asarray(lg["actions"][rows], np.float32), int(len(held)), held


def visited(policy, physics, rep, which):
    """N_T3_EACH states visited by `policy` in strong current (module docstring), rounded to float32; with their
    start index and time step."""
    _, _, _, _, TC, TR = C.gate()
    x0 = TC.bank_starts(T3_SITUATION)[:T3_STARTS]
    run = TR.rollout(policy, x0, physics, record=True)
    S = run.states[:, :-1]                                   # (starts, 100, 13): s_0 .. s_99
    n_starts, steps = S.shape[:2]
    flat = S.reshape(-1, TC.DX)
    idx = np.sort(C.rng(rep, TAG_T3, which).choice(len(flat), N_T3_EACH, replace=False))
    states = flat[idx].astype(np.float32).astype(np.float64)
    if not np.all(np.isfinite(states)):
        raise ValueError("visited states are not finite")
    return states, idx // steps, idx % steps, dict(starts=int(n_starts), steps=int(steps), pool=int(len(flat)),
                                                   past80_share_along_rollout=float(
                                                       TC.past_knee(run.commands, physics).mean()))


def build(ctx, world, log_fn=C.log):
    """The pair sets of one seed and world (module docstring). Returns (arrays, info)."""
    H, A, L, RC, TC, TR = C.gate()
    rep = int(ctx.rep)
    pg = ctx.prepared(world)
    physics = pg.physics
    none = ctx.none(world)
    clone = ctx.clone(world)[0]
    obs = np.asarray(pg.train.obs)
    if not np.array_equal(pg.states.astype(np.float32).astype(np.float64), pg.states):
        raise ValueError("training states are not exactly float32")
    cols = {k: [] for k in ("states", "commands", "refs", "set", "role", "pair", "log_row", "train_index",
                            "t3_source", "t3_start", "t3_step")}

    def add(set_name, role, states, commands, refs, pair, log_row, train_index, src="", start=-1, step=-1):
        n = len(states)
        cols["states"].append(np.asarray(states, np.float64))
        cols["commands"].append(np.asarray(commands, np.float32))
        cols["refs"].append(np.asarray(refs, np.float32))
        cols["set"].append(np.full(n, set_name, "<U2"))
        cols["role"].append(np.full(n, role, "<U8"))
        cols["pair"].append(np.asarray(pair, np.int64))
        cols["log_row"].append(np.broadcast_to(np.asarray(log_row, np.int64), (n,)).copy())
        cols["train_index"].append(np.broadcast_to(np.asarray(train_index, np.int64), (n,)).copy())
        cols["t3_source"].append(np.full(n, src, "<U10"))
        cols["t3_start"].append(np.broadcast_to(np.asarray(start, np.int64), (n,)).copy())
        cols["t3_step"].append(np.broadcast_to(np.asarray(step, np.int64), (n,)).copy())

    info = dict(world=world, rep=rep, n_train_rows=int(pg.n), region_rows=int(pg.region.sum()))
    # T1 and T2: training states
    for set_name, mask, n, tag in (("T1", pg.region, N_T1, TAG_T1), ("T2", ~pg.region, N_T2, TAG_T2)):
        pool = np.flatnonzero(mask)
        ti = np.sort(C.rng(rep, tag).choice(pool, min(n, len(pool)), replace=False))
        s = pg.states[ti]
        cmd = L.actor_rows(none, obs[ti])
        rec = pg.actions[ti]
        cln = L.actor_rows(clone, obs[ti])
        pid = np.arange(len(ti))
        lr = pg.rows[ti]
        roles = {"command": cmd, "cap": cap80(cmd), "recorded": rec, "cloned": cln, "knee": cap81(cmd)}
        for role in SET_ROLES[set_name]:
            add(set_name, role, s, roles[role], rec, pid, lr, ti)
        info[set_name] = dict(states=int(len(ti)), pool=int(len(pool)), rule=f"default_rng([20261015, {rep}, {tag}])",
                              command_past80_share=float(TC.past_knee(cmd).mean()))
    # T3: strong-current states visited by the actor (T3a) and by the stall-aware controller (T3b)
    pols = (("actor", L.mlp_policy(none, pg, "gapped critic's actor (float64 copy)")), ("controller",
                                                                                          RC.stall_aware(physics)))
    base = 0
    for which, (src, pol) in enumerate(pols):
        s, st, tt, vinfo = visited(pol, physics, rep, which)
        o = pg.normalize(s)
        cmd = L.actor_rows(none, o)
        cln = L.actor_rows(clone, o)
        pid = base + np.arange(len(s))
        base += len(s)
        for role, a in (("command", cmd), ("cap", cap80(cmd))):
            add("T3", role, s, a, cln, pid, -1, -1, src, st, tt)
        info[f"T3_{src}"] = dict(vinfo, states=int(len(s)), rule=f"default_rng([20261015, {rep}, {TAG_T3}, {which}])",
                                 command_past80_share=float(TC.past_knee(cmd).mean()))
    # T4: held-out recorded rows
    rows, s, a, n_pool, pool_rows = held_out_rows(ctx.sd[world], rep)
    add("T4", "recorded", s, a, a, np.arange(len(rows)), rows, -1)
    info["T4"] = dict(rows=int(len(rows)), pool=n_pool, rule=f"default_rng([20261015, {rep}, {TAG_HELD}])",
                      recorded_past80_share=float(TC.past_knee(a).mean()),
                      pool_rows_sha256=C.content_sha256({"pool_rows": pool_rows}))
    arrays = {k: np.concatenate(v) for k, v in cols.items()}
    info["n_pairs"] = int(len(arrays["set"]))
    info["counts"] = {f"{s_}/{r_}": int(np.sum((arrays["set"] == s_) & (arrays["role"] == r_)))
                      for s_ in PAIR_SETS for r_ in SET_ROLES[s_]}
    info["artifacts"] = dict(none_actor_sha256=L.tree_sha256(none), clone_sha256=L.tree_sha256(clone),
                             prepared_sha256=pg.sha256)
    return arrays, info


def save(arrays, info, rep, world, out=C.OUT, extra=None, label=None):
    """Write the pair sets once (file + record, hashed, stamped with the current detect spec and code manifest).
    label: by default chosen by the folder (common.label_for: the Stage 1 label in the rehearsal's folder, Stage 2's
    in runs/shift_explore/stage2); a Stage 1 label in Stage 2's folder is refused."""
    d = C.seed_dir(rep, world, out)
    label = C.label_for(d, label)
    if (d / FILE).exists() or (d / RECORD).exists():
        raise SystemExit(f"refused: pair sets exist in {d}; they are written once")
    fsha, csha = C.save_npz(d / FILE, arrays)
    rec = dict(label=label, kind="detect-pairs", rep=int(rep), world=world, file=FILE, file_sha256=fsha,
               content_sha256=csha, info=info, **(extra or {}))
    rec.update(C.stamp())
    C.write_json(d / RECORD, rec)
    return rec


def load(rep, world, out=C.OUT):
    """The saved pair sets, verified (record sidecar, file sidecar, file and content SHA-256 the record names) and
    written under the current detect spec and code manifest (common.check_version)."""
    d = C.seed_dir(rep, world, out)
    rec = C.read_json(d / RECORD)
    if rec is None:
        raise SystemExit(f"refused: no pair sets at {d}; run the 'pairs' phase first")
    C.check_version(rec, f"the pair sets in {d}")
    arrays = C.load_npz(d / FILE, rec["file_sha256"], rec["content_sha256"])
    return arrays, rec


def view(arrays):
    """Index helpers: per set, the row index of each role by pair id (-1 where absent)."""
    out = {}
    for s in PAIR_SETS:
        m = arrays["set"] == s
        if not m.any():
            continue
        pid = arrays["pair"][m]
        n = int(pid.max()) + 1
        idx = {}
        rows = np.flatnonzero(m)
        for r in SET_ROLES[s]:
            ix = np.full(n, -1, np.int64)
            rr = rows[arrays["role"][rows] == r]
            ix[arrays["pair"][rr]] = rr
            idx[r] = ix
        out[s] = idx
    return out
