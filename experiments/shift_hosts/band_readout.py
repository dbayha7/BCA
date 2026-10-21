"""The band's readouts in the training environment: the learner's own target y, its critic value q and BCA's band width
U at rows handed over by file from the scoring side, the band's coverage at three kinds of pairs, and the viewer's
per-step arrays (plan sections 3.1, 4.6, 5.4 and 9.2; claims 4a and 4b). Training environment (/opt/bca-venv).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. It reads a
finished run (runner.py: record.json and a snapshot), rebuilds the learner exactly as the run built it
(runner.build_setup on the same prepared file) and computes, with the learner's own band_parts
(unifloral_<learner>_bca), nothing the run did not already define. No process here holds the simulator: next states
come from the scoring environment by file (evaluate.actor_transitions, evaluate.visited_transitions) or from the
prepared file.

Rows (plan section 5.4, "Coverage ... at three kinds of pairs", strong mix):
  recorded  the prepared strong-law test steps (test_strong): recorded state and command, the recorded next
            observation (the float32 of the exact next state) and reward
  actor     the same recorded states with the actor's own command, the simulator's exact next state and reward
            (evaluate.actor_transitions; CQL's sampled command uses the strong bank's declared noise at the test
            step's own episode and step, eps[episode, t], a decision recorded in the runner report)
  visited   the actor's own visited states on the first 200 strong-mix starts (evaluate.visited_transitions)
At each row: y and q by the learner's band_parts from the snapshot's state (TD3+BC and ReBRAC: y with the target's
noise, q = min of the online critics; CQL: y with its entropy term, q = min of the ten critics; IQL: y = r + 0.98 V(s'),
q = min of Polyak(target, critic)); ReBRAC's a' is the next recorded command for recorded rows (test_strong/next_action,
the roll over the test log) and the stored actor's own command at the next state for the actor and visited rows (no
recorded next command exists there); eta = the frozen reference's scale network at (s, a); U = R x u x max(eta, 1e-6)
(coverage.width's float32 arithmetic) with R the standard threshold of the run's last refresh, and, for an IW run, also
each row's own threshold R(w(c)) looked up in that refresh's table at the row's exact weight w(c), c = state[:, 10]
(the recorded rows carry it as weight_strong; runner.strong_weight restates world.ratio for the others).

Coverage (restated from experiments/shift_iw/coverage.py:100-106, 251-316): covered when q - U <= y <= q + U (ties
covered); outcome 1 covered, 0 not covered, -1 refused; a weighted (IW) arm refuses every row whose command lies inside
the seed's ball (the exact density ratio is +inf there; coverage.refusals; the ball's rule
experiments/truth/gap.py:165-172, centre and radius from the prepared file's meta), and every arm refuses a row whose
width is +inf (R = +inf); coverage = covered / all rows of a group (refused rows count against it), with an
episode-bootstrap 95% interval (2,000 draws, seed (20261016, 77)), the refused share and the coverage among certified
rows; outside the ball, inside it, and all.
Target-noise keys (restated from experiments/shift_iw/learners_iw.py:407-431, coverage_key and critic_parts):
PRNGKey(default_rng([20261016, rep, 51, 2]).integers(2**31 - 1)) for the recorded rows (tag 51 the weighting test's
TAG_Y, 2 the strong tilt's index), with one more tag 1 (actor) or 2 (visited) for the other groups, and per chunk of
8,192 rows fold_in(key, chunk index); chunks are padded by repeating their last row, as critic_parts pads them.

The viewer (plan section 9.2): for seed 0's final BCA actors, at every step of the viewer's fixed-start episodes (the
scores file of evaluate.score_actor(keep_viewer=True)), the critic's value q at the actor's command and the band width
U there, R the standard threshold or, for IW, the episode's own threshold at its test mass w(c).

Permission: guard.require_bca(permission, rows) before any band arithmetic (INTERFACES.md section 8: "a band readout
(on its rows)"); real runs need gate.require_run_approval() (called first in the command line, before anything is
read), made-up runs guard.MADE_UP on registered made-up rows.

Usage (training environment, processor):
  python -m experiments.shift_hosts.band_readout coverage --run-dir <run> [--step T] --out <dir>
      [--actor-transitions <file>] [--visited-transitions <file>] [--made-up-file <made-up prepared file>]
  python -m experiments.shift_hosts.band_readout viewer --run-dir <run> [--step T] --scores <scores.npz> --out <file>
"""

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from experiments.shift_hosts import gate, guard
from experiments.shift_hosts import handoff as H
from experiments.shift_hosts import runner as RN

sys.dont_write_bytecode = True

SCHEMA = "shift-hosts-band-readout-v1"
CHUNK = 8192                     # learners_iw.critic_parts's chunk
COVERED, MISSED, REFUSED = 1, 0, -1
BOOTSTRAP = 2000                 # coverage.BOOTSTRAP
BOOT_SEED = (20261016, 77)       # coverage.episode_bootstrap's seed
MASTER_SEED = 20261016
TAG_Y = RN.COVERAGE_KEY_TAG      # 51, learners_iw.TAG_Y
STRONG_TILT = 2                  # world.TILT_NAMES.index("strong")
GROUPS = ("recorded", "actor", "visited")
GROUP_TAG = dict(actor=1, visited=2)


# ------------------------------------------------------------------------------------------------ coverage arithmetic

def width(threshold, unit, scale):
    """U = R x (max(scale, 1e-6) x unit) in float32; +inf where R = +inf (restated from coverage.py:100-106, the
    arithmetic of calibration/dose.frozen_level_dose). threshold: one number or one per row."""
    s = np.maximum(np.asarray(scale, np.float32), np.float32(1e-6)) * np.float32(unit)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(threshold, np.float32) * s


def refusals(in_ball, arm):
    """Rows an arm refuses before any threshold (coverage.py:251-259): the weighted arm ("iw") refuses every row whose
    command lies inside the ball; the unweighted arm ("standard") refuses none."""
    in_ball = np.asarray(in_ball, bool)
    if arm == "iw":
        return in_ball.copy()
    if arm == "standard":
        return np.zeros(in_ball.shape, bool)
    raise ValueError("arm must be 'standard' or 'iw'")


def outcomes(y, q, U, refused=None):
    """Per row: 1 covered (q - U <= y <= q + U), 0 not covered, -1 refused (given refusals, or a non-finite width)
    (coverage.py:262-272)."""
    y, q, U = (np.asarray(v, np.float64) for v in (y, q, U))
    covered = (y >= q - U) & (y <= q + U)
    out = np.where(covered, COVERED, MISSED).astype(np.int8)
    refuse = ~np.isfinite(U)
    if refused is not None:
        refuse = refuse | np.asarray(refused, bool)
    out[refuse] = REFUSED
    return out


def episode_bootstrap(values, episodes, boot=BOOTSTRAP, seed=BOOT_SEED, level=0.95, counts=None):
    """Percentile interval of a mean by resampling whole episodes (coverage.py:275-287)."""
    values = np.asarray(values, np.float64)
    counts = np.ones_like(values) if counts is None else np.asarray(counts, np.float64)
    ids, inv = np.unique(np.asarray(episodes), return_inverse=True)
    sums = np.bincount(inv, values, len(ids))
    ns = np.bincount(inv, counts, len(ids))
    rng = np.random.default_rng(list(seed))
    idx = rng.integers(len(ids), size=(int(boot), len(ids)))
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = sums[idx].sum(1) / ns[idx].sum(1)
    lo, hi = np.nanquantile(reps, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def summarize(outcome, episodes, mask=None, boot=BOOTSTRAP, seed=BOOT_SEED):
    """Coverage of a group of rows (coverage.py:290-306): rows, coverage = covered / rows (refused count against it)
    with its episode-bootstrap interval, the refused share, and the coverage among certified rows."""
    o = np.asarray(outcome)
    ep = np.asarray(episodes)
    if mask is not None:
        o, ep = o[np.asarray(mask, bool)], ep[np.asarray(mask, bool)]
    n = int(len(o))
    if n == 0:
        return dict(steps=0, coverage=None, interval=None, refused_share=None, certified_coverage=None)
    cov = (o == COVERED).astype(np.float64)
    ref = o == REFUSED
    certified = int((~ref).sum())
    return dict(steps=n, episodes=int(len(np.unique(ep))), coverage=float(cov.mean()),
                interval=episode_bootstrap(cov, ep, boot, seed), refused_share=float(ref.mean()),
                certified_steps=certified, certified_coverage=float(cov[~ref].mean()) if certified else None)


def groups(outcome, episodes, in_ball, **kw):
    """summarize() outside the ball, inside it and overall (coverage.py:309-315)."""
    in_ball = np.asarray(in_ball, bool)
    return dict(outside=summarize(outcome, episodes, ~in_ball, **kw),
                inside=summarize(outcome, episodes, in_ball, **kw), all=summarize(outcome, episodes, None, **kw))


def in_ball(actions, ball):
    """Whether each command lies in the seed's ball: ||a - centre||_2 <= rho in float64 (restated from
    experiments/truth/gap.py:165-172 and shift.py:86-88: float32 commands compared at their exact float64 value).
    ball: {"centre": [4], "rho": float} (the prepared file's meta.ball.gap)."""
    a = np.atleast_2d(np.asarray(actions).astype(np.float64))
    d = np.sqrt(np.sum((a - np.asarray(ball["centre"], np.float64)[None, :]) ** 2, axis=-1))
    return d <= float(ball["rho"])


def ball_of(prepared):
    meta = prepared.meta.get("ball") or {}
    gap = meta.get("gap") or meta
    if "centre" not in gap or "rho" not in gap:
        raise ValueError(f"{prepared.path}: no ball in the prepared file's meta")
    return dict(centre=[float(v) for v in gap["centre"]], rho=float(gap["rho"]))


def coverage_key(rep, group):
    """The target-noise key of a coverage group (restated from learners_iw.coverage_key: PRNGKey of
    default_rng([20261016, rep, 51, *tags]).integers(2**31 - 1)), tags (2,) for the recorded strong-law rows and
    (2, 1) / (2, 2) for the actor and visited rows."""
    import jax

    tags = [STRONG_TILT] + ([GROUP_TAG[group]] if group in GROUP_TAG else [])
    seed = int(np.random.default_rng([MASTER_SEED, int(rep), TAG_Y, *tags]).integers(2 ** 31 - 1))
    return jax.random.PRNGKey(seed)


# ------------------------------------------------------------------------------------------------ a run, reloaded

class Loaded(NamedTuple):
    record: dict
    setup: Any              # runner.Setup, rebuilt from the same prepared file
    carry: tuple            # the snapshot's (rng, agent_state[, bca_state])
    step: int
    table: Any              # the IW table of the last refresh at or before step (iw.Table), or None
    R_standard: float       # the standard threshold of that refresh (+inf before any refresh)
    prepared: Any


def load_run(run_dir, prepared, permission, step=None):
    """A finished run's record, its setup rebuilt exactly as the run built it (runner.build_setup on the same prepared
    file, checked by SHA-256), and the snapshot at `step` (default: the last refresh for BCA runs, the last step
    otherwise) in that structure, after handoff.read's checks and a check of the snapshot's SHA-256 against the
    record."""
    run_dir = Path(run_dir)
    rec = H.read_json(run_dir / "record.json")
    if rec.get("outcome") != "complete":
        raise ValueError(f"{run_dir}: the run did not complete ({rec.get('outcome')}); no readout")
    if rec["data"]["npz_sha256"] != prepared.npz_sha256:
        raise ValueError(f"{run_dir} ran on another prepared file than {prepared.path}")
    ident = rec["identity"]
    sched = RN.Schedule(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ident["schedule"].items()})
    s = RN.build_setup(rec["learner"], rec["setup"], prepared, permission, sched)
    refreshed = [r["step"] for r in rec.get("refreshes") or []]
    if step is None:
        step = max(refreshed) if refreshed else sched.steps
    snap = (rec.get("snapshots") or {}).get(str(int(step)))
    if snap is None:
        raise ValueError(f"{run_dir} has no snapshot at outer step {step} ({sorted(rec.get('snapshots') or {})})")
    carry, man = RN.load_snapshot(snap["path"], s.carry0)
    if man["npz_sha256"] != snap["sha256"]:
        raise H.HandoffError(f"{snap['path']} differs from the run record's SHA-256")
    table, R_std = None, math.inf
    done = [r for r in rec.get("refreshes") or [] if r["step"] <= int(step)]
    if done:
        last = done[-1]
        R_std = float(np.asarray(carry[2].reference.threshold)) if len(carry) > 2 else math.inf
        if last.get("iw"):
            from experiments.shift_hosts import iw as IW

            table = IW.table_from_summary(last["iw"]["table"])
    return Loaded(rec, s, carry, int(step), table, R_std, prepared)


def _pad(tree, size):
    """Pad every leaf's first axis to `size` by repeating its last row (restated from learners_iw._pad)."""
    import jax
    import jax.numpy as jnp

    def pad(x):
        x = jnp.asarray(x)
        extra = size - x.shape[0]
        return x if extra == 0 else jnp.concatenate([x, jnp.repeat(x[-1:], extra, 0)])

    return jax.tree_util.tree_map(pad, tree)


_PROGRAMS = {}


def _programs(L):
    """The compiled band parts of a loaded run: (y, q) by the learner's own band_parts, and eta of the frozen scale
    network, at a padded chunk."""
    import jax

    s = L.setup
    key = (s.learner, s.setup, id(s.built.apply_fns[0]))
    if key not in _PROGRAMS:
        parts = jax.jit(lambda agent, rows, k: s.module.band_parts(s.args, s.built.apply_fns, agent, rows, k))
        eta = jax.jit(lambda scale_apply_params, obs, act: s.carry0[2].scale.apply_fn(scale_apply_params, obs, act))
        actor = jax.jit(lambda params, obs: s.built.apply_fns[0](params, obs).sample(seed=None))
        _PROGRAMS[key] = (parts, eta, actor)
    return _PROGRAMS[key]


def transition_of(L, rows, next_action=None):
    """Unifloral's Transition of `rows` ({obs, action, reward, next_obs, done}) for the run's learner; ReBRAC takes
    next_action (an array) for its target's a'."""
    from experiments.shift_hosts import unifloral_import as UI

    if L.setup.learner == "rebrac":
        if next_action is None:
            raise ValueError("ReBRAC's band target needs a next command per row")
        return UI.make_transition("rebrac", rows, next_action=np.asarray(next_action, np.float32))
    return UI.make_transition(L.setup.learner, rows)


def actor_next_commands(L, next_obs):
    """The stored actor's own command at each next observation (ReBRAC's a' for the actor and visited rows): the tanh
    output of its deterministic actor (rebrac.py:118-139)."""
    import jax.numpy as jnp

    _, _, actor = _programs(L)
    out = []
    for i0 in range(0, len(next_obs), CHUNK):
        part = jnp.asarray(next_obs[i0:i0 + CHUNK])
        n = part.shape[0]
        padded = _pad(part, CHUNK)
        out.append(np.asarray(actor(L.carry[1].actor.params, padded))[:n])
    return np.concatenate(out).astype(np.float32)


def band_values(L, trans, key, permission):
    """y, q (float32 [n]) by the learner's own band_parts from the snapshot's state, and eta (float32 [n]) of the frozen
    scale network at (s, a), in chunks of 8,192 rows with fold_in(key, chunk index) (module docstring). An entry point:
    guard.require_bca(permission, trans) first."""
    import jax

    guard.require_bca(permission, trans)
    parts, eta_fn, _ = _programs(L)
    n = int(trans.obs.shape[0])
    ys, qs, es = [], [], []
    bca = L.carry[2] if len(L.carry) > 2 else None
    for j, i0 in enumerate(range(0, n, CHUNK)):
        part = jax.tree_util.tree_map(lambda x: x[i0:i0 + CHUNK], trans)
        m = int(part.obs.shape[0])
        padded = _pad(part, CHUNK)
        y, q = parts(L.carry[1], padded, jax.random.fold_in(key, j))
        ys.append(np.asarray(y)[:m])
        qs.append(np.asarray(q)[:m])
        if bca is not None:
            es.append(np.asarray(eta_fn(bca.reference.cal_params, padded.obs, padded.action))[:m])
    y, q = np.concatenate(ys), np.concatenate(qs)
    eta = np.concatenate(es) if es else None
    return y, q, eta


def thresholds_at(L, masses):
    """Each row's IW threshold R(w) from the last refresh's table at its test mass (iw.lookup, then float32 as BCA
    stores a threshold, iw.table_float32); a mass beyond the table's grid is refused (+inf). Returns (R float32 [n],
    rows beyond the grid)."""
    from experiments.shift_hosts import iw as IW

    m = np.asarray(masses, np.float64)
    grid = L.table.grid
    inside = (m > 0) & np.isfinite(m) & (m <= grid[-1])
    R = np.full(m.shape, np.inf, np.float32)
    if inside.any():
        _, idx = IW.lookup(L.table, m[inside])
        R[inside] = IW.table_float32(L.table)[idx]
    return R, int(np.sum(~inside))


def readout(L, rows, *, key, permission, masses, ball_flags, episodes, next_action=None):
    """y, q, eta and the band at `rows`, and coverage per arm (module docstring). Returns (arrays, summary)."""
    trans = transition_of(L, rows, next_action)
    y, q, eta = band_values(L, trans, key, permission)
    bca = L.carry[2]
    unit = float(np.asarray(bca.reference.residual_scale))
    arrays = dict(y=y, q=q, eta=eta, abs_error=np.abs(y.astype(np.float64) - q.astype(np.float64)),
                  in_ball=np.asarray(ball_flags, bool), episode=np.asarray(episodes, np.int64),
                  mass=np.asarray(masses, np.float64))
    summary = dict(rows=int(len(y)), unit=unit, standard_threshold=L.R_standard)
    arms = dict(standard=np.full(len(y), np.float32(L.R_standard), np.float32))
    beyond = 0
    if L.table is not None:
        arms["iw"], beyond = thresholds_at(L, masses)
    summary["masses_beyond_grid"] = beyond
    for arm, R in arms.items():
        U = width(R, unit, eta)
        o = outcomes(y, q, U, refusals(ball_flags, arm))
        arrays[f"{arm}/R"], arrays[f"{arm}/U"], arrays[f"{arm}/outcome"] = R, U, o
        summary[arm] = groups(o, episodes, ball_flags)
    return arrays, summary


def coverage(L, permission, *, actor_file=None, visited_file=None, log=print):
    """The three kinds of pairs (module docstring) for a loaded BCA run. Returns ({group: arrays}, {group: summary})."""
    p = L.prepared
    rep = p.rep
    ball = ball_of(p)
    out_arrays, out_summary = {}, {}
    rec_rows = p.rows("test_strong")
    started = time.monotonic()
    a, s = readout(L, rec_rows, key=coverage_key(rep, "recorded"), permission=permission,
                   masses=p.field("test_strong", "weight_strong"), ball_flags=p.field("test_strong", "in_ball"),
                   episodes=p.field("test_strong", "episode"), next_action=p.field("test_strong", "next_action"))
    s["seconds"] = round(time.monotonic() - started, 3)
    out_arrays["recorded"], out_summary["recorded"] = a, s
    for group, path in (("actor", actor_file), ("visited", visited_file)):
        if path is None:
            continue
        started = time.monotonic()
        arr, man = H.read(path)
        meta = man.get("meta") or {}
        actor_sha = (L.record.get("snapshots") or {}).get(str(L.step), {}).get("actor", {}).get("sha256")
        if meta.get("actor_npz_sha256") != actor_sha:
            raise ValueError(f"{path} was written for another actor file than the snapshot's ({actor_sha})")
        rows = dict(obs=arr["obs"], action=arr["action32"], reward=arr["reward"], next_obs=arr["next_obs"],
                    done=np.zeros(len(arr["obs"]), np.float32))
        if group == "actor":
            if arr["state"].tobytes() != np.asarray(p.field("test_strong", "state")).tobytes():
                raise ValueError(f"{path}: its states are not the prepared strong-law test states")
            masses, episodes = p.field("test_strong", "weight_strong"), p.field("test_strong", "episode")
        else:
            masses, episodes = RN.strong_weight(arr["state"][:, 10]), arr["start"]
        nxt = actor_next_commands(L, arr["next_obs"]) if L.setup.learner == "rebrac" else None
        a, s = readout(L, rows, key=coverage_key(rep, group), permission=permission, masses=masses,
                       ball_flags=in_ball(arr["action32"], ball), episodes=episodes, next_action=nxt)
        s.update(seconds=round(time.monotonic() - started, 3), file=str(path), npz_sha256=man["npz_sha256"],
                 next_action_rule="the stored actor's command at the next state" if nxt is not None else None)
        out_arrays[group], out_summary[group] = a, s
    log(f"coverage of {L.record['run_id']} at outer step {L.step}: groups {sorted(out_summary)}")
    return out_arrays, out_summary


def viewer(L, permission, states, commands):
    """The viewer's per-step arrays of one bank (plan 9.2): q at the actor's command and U there, for k episodes of
    100 steps (states float64 [k, 101, 13], commands [k, 100, 4]); U with the standard threshold, and for IW also at
    each episode's own test mass w(c), c = states[:, 0, 10]. Returns {name: float32 [k, 100]}."""
    states = np.asarray(states, np.float64)
    commands = np.asarray(commands)
    k, T = commands.shape[0], commands.shape[1]
    obs = states[:, :T].reshape(k * T, -1).astype(np.float32)
    nxt = states[:, 1:T + 1].reshape(k * T, -1).astype(np.float32)
    act = commands.reshape(k * T, -1).astype(np.float32)
    rows = dict(obs=obs, action=act, reward=np.zeros(k * T, np.float32), next_obs=nxt,
                done=np.zeros(k * T, np.float32))
    na = actor_next_commands(L, nxt) if L.setup.learner == "rebrac" else None
    trans = transition_of(L, rows, na)
    import jax

    _, q, eta = band_values(L, trans, jax.random.PRNGKey(0), permission)
    unit = float(np.asarray(L.carry[2].reference.residual_scale))
    out = dict(q=q.reshape(k, T), U_standard=width(np.float32(L.R_standard), unit, eta).reshape(k, T))
    if L.table is not None:
        masses = np.repeat(RN.strong_weight(states[:, 0, 10]), T)
        R, _ = thresholds_at(L, masses)
        out["U_iw"] = width(R, unit, eta).reshape(k, T)
    return out


# ------------------------------------------------------------------------------------------------ command line

def _log(msg):
    print(f"[{RN.utc_now()}] {msg}", flush=True)


def _prepared_and_permission(run_dir, made_up_file, purpose):
    """The run's prepared file and the permission to read the band on it: real runs ask gate.require_run_approval()
    before reading anything; made-up runs need --made-up-file and registered made-up arrays."""
    rec_meta = H.read_json(Path(run_dir) / "record.json")
    if rec_meta["data"]["made_up"]:
        if made_up_file is None:
            raise gate.ApprovalError("a made-up run's readout names its --made-up-file")
        prepared = RN.read_prepared(made_up_file, made_up=True)
        return prepared, None, rec_meta
    approval = gate.require_run_approval(purpose)
    prepared = RN.read_prepared(rec_meta["data"]["path"], made_up=False)
    return prepared, approval, rec_meta


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m experiments.shift_hosts.band_readout")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--run-dir", required=True)
    c.add_argument("--step", type=int)
    c.add_argument("--out", required=True)
    c.add_argument("--actor-transitions")
    c.add_argument("--visited-transitions")
    c.add_argument("--made-up-file")
    v = sub.add_parser("viewer")
    v.add_argument("--run-dir", required=True)
    v.add_argument("--step", type=int)
    v.add_argument("--scores", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--made-up-file")
    a = p.parse_args(argv)
    try:
        rec0 = H.read_json(Path(a.run_dir) / "record.json")
        purpose = f"band readout ({a.command}) of {rec0.get('run_id')}"
        made_up = rec0["data"]["made_up"]
        if not made_up:
            gate.require_run_approval(purpose)                 # before anything else is read (plan 6.1)
        from experiments.shift_hosts import unifloral_import as UI

        UI.prepare(learners=(rec0["learner"],), device="cpu")
        prepared, permission, _ = _prepared_and_permission(a.run_dir, a.made_up_file, purpose)
        if made_up:
            permission = RN.made_up_permission(prepared)
        L = load_run(a.run_dir, prepared, permission, a.step)
        if L.setup.bca is None:
            raise ValueError("band readouts are for the BCA setups")
        out = Path(a.out)
        if a.command == "coverage":
            arrays, summary = coverage(L, permission, actor_file=a.actor_transitions,
                                       visited_file=a.visited_transitions, log=_log)
            flat = {f"{g}/{k}": v for g, d in arrays.items() for k, v in d.items() if v is not None}
            man = H.write(out / "coverage.npz", flat, meta=dict(schema=SCHEMA, kind="coverage",
                                                                run_id=L.record["run_id"], step=L.step))
            H.write_json(out / "coverage.json", dict(schema=SCHEMA, kind="coverage", run_id=L.record["run_id"],
                                                     learner=L.setup.learner, setup=L.setup.setup, step=L.step,
                                                     record_sha256=H.sha256_file(Path(a.run_dir) / "record.json"),
                                                     arrays_npz_sha256=man["npz_sha256"], groups=summary,
                                                     standard_threshold=L.R_standard,
                                                     iw_table=L.table is not None, device="cpu"))
        else:
            arrs, man = H.read(a.scores)
            flat = {}
            for name in arrs:
                if name.endswith("/episodes/states"):
                    stem = name[:-len("/episodes/states")]
                    got = viewer(L, permission, arrs[name], arrs[f"{stem}/episodes/commands"])
                    flat.update({f"{stem}/{k}": v for k, v in got.items()})
            H.write(out, flat, meta=dict(schema=SCHEMA, kind="viewer", run_id=L.record["run_id"], step=L.step,
                                         scores_npz_sha256=man["npz_sha256"]))
        print(json.dumps(dict(status="complete", out=str(out)), sort_keys=True), flush=True)
        return 0
    except gate.ApprovalError as error:
        print(f"REFUSED: {error}", file=sys.stderr, flush=True)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
