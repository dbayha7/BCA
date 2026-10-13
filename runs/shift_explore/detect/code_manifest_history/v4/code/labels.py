"""EXPLORATORY detection rehearsal (plan section 4.4), NO BCA: the LABEL script. It cannot see any signal.

It reads only the locked pair sets of one seed and world (pairs.npz, verified against its record and sidecars), the
world's frozen gapped critic (the critic's own preference is part of the label's definition) and exact simulation.
It imports no signal code: this file imports only common.py and pairs.py (test_detect.py checks its import list),
it refuses to run when signals.py, measures.py or detect.py is loaded in the same process (so it always runs as a
process of its own), and it never opens a score file. Its output (labels.npz + labels.json, with sidecars) is joined
with any signal's scores only by measures.py, after both are hashed.

Per pair (s, a), with cap = float32(clip(float64(a), -0.8, 0.8)) and c the stall-aware controller (the critic's
scored policy, recorder.stall_aware, in both worlds: check 2's no-stall critics are stall-aware-controller critics):
  unsupported    any thruster past 80% (thruster_chain.past_knee: |a_k| > the float32 knee)
  critic_pref    Q1(s, a) - Q1(s, cap)                  the frozen gapped critic's first head (the head the actor
                                                         climbs), at the float32-rounded normalised state
  true_pref      Q^c(s, a) - Q^c(s, cap)                exact float64 action values of the stall-aware controller in
                                                         the world's physics: one rollout of 1,000 steps (truth.
                                                         q_value_jax, x64), truncation 0.98^1000 ~ 1.7e-9 relative
  over_pref      critic_pref - true_pref                (truth.pull_past_80's quantity)
  harmful        unsupported AND over_pref >= 0.1 AND true_pref <= -0.05   (plan 4.4: the critic's preference for the
                                                         command over its cap exceeds the true preference by at least
                                                         0.1, and the cap is truly better by at least 0.05)
A pair that is not past 80% equals its cap bit for bit: its preferences are exactly 0 and it is harmless; it is not
simulated (true values stored as NaN, 'simulated' False). Every role gets labels against its own cap (a cap's cap is
itself, so every cap pair is harmless).
"""

import sys
import time

import numpy as np

import common as C
import pairs as P

HARM_OVER, HARM_TRUE = 0.1, 0.05
HORIZON = 1000
FILE = "labels.npz"
RECORD = "labels.json"
FORBIDDEN_MODULES = ("signals", "measures", "detect")


def refuse_if_signals_loaded():
    loaded = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    if loaded:
        raise SystemExit(f"refused: the label script must run without any signal code loaded ({', '.join(loaded)})")


def compute(ctx, world, arrays, log_fn=C.log):
    """The labels of the pair sets `arrays` (module docstring). Returns (label arrays, info)."""
    refuse_if_signals_loaded()
    H, A, L, RC, TC, TR = C.gate()
    t0, c0 = time.perf_counter(), time.process_time()
    pg = ctx.prepared(world)
    physics = pg.physics
    critic, crec = ctx.critic(world)
    s = np.asarray(arrays["states"], np.float64)
    a = np.asarray(arrays["commands"], np.float32)
    cap = P.cap80(a)
    unsupported = TC.past_knee(a, physics)
    same = np.all(cap == a, axis=1)
    if np.any(unsupported & same) or np.any(~unsupported & ~same):
        raise ValueError("the cap and the past-80% test disagree")
    obs = pg.normalize(s)
    q = L.critic_rows(critic, obs, a)[0]
    qc = L.critic_rows(critic, obs, cap)[0]
    t_crit = time.perf_counter() - t0
    sim = np.flatnonzero(~same)
    true_cmd = np.full(len(a), np.nan)
    true_cap = np.full(len(a), np.nan)
    pol = RC.stall_aware(physics)
    t1 = time.perf_counter()
    if len(sim):
        true_cmd[sim] = TR.q_value_jax(pol, s[sim], a[sim].astype(np.float64), physics, horizon=HORIZON, chunk=4096)
        true_cap[sim] = TR.q_value_jax(pol, s[sim], cap[sim].astype(np.float64), physics, horizon=HORIZON, chunk=4096)
    t_sim = time.perf_counter() - t1
    critic_pref = np.where(same, 0.0, q[:, 0] - qc[:, 0])
    true_pref = np.where(same, 0.0, true_cmd - true_cap)
    over_pref = critic_pref - true_pref
    harmful = unsupported & (over_pref >= HARM_OVER) & (true_pref <= -HARM_TRUE)
    if not (np.all(np.isfinite(critic_pref)) and np.all(np.isfinite(true_pref))):
        raise ValueError("labels are not finite")
    out = dict(unsupported=unsupported, harmful=harmful, critic_pref=critic_pref, true_pref=true_pref,
               over_pref=over_pref, simulated=~same, q1_command=q[:, 0], q2_command=q[:, 1], q1_cap=qc[:, 0],
               q2_cap=qc[:, 1], true_command=true_cmd, true_cap=true_cap)
    counts = {}
    for st in P.PAIR_SETS:
        for r in P.SET_ROLES[st]:
            m = (arrays["set"] == st) & (arrays["role"] == r)
            if m.any():
                counts[f"{st}/{r}"] = dict(n=int(m.sum()), unsupported=int(unsupported[m].sum()),
                                           harmful=int(harmful[m].sum()))
    for src in ("actor", "controller"):
        m = (arrays["set"] == "T3") & (arrays["role"] == "command") & (arrays["t3_source"] == src)
        counts[f"T3{'a' if src == 'actor' else 'b'}/command"] = dict(n=int(m.sum()), unsupported=int(unsupported[m].sum()),
                                                                     harmful=int(harmful[m].sum()))
    info = dict(world=world, rep=int(ctx.rep), counts=counts, simulated_pairs=int(len(sim)),
                critic_record_sha256=C.sha256_bytes(A.canonical(crec).encode("utf8")) if crec is not None else None,
                critic_params_sha256=L.tree_sha256(critic), controller="stall_aware", physics=physics.as_dict(),
                horizon=HORIZON, thresholds=dict(over_pref_at_least=HARM_OVER, cap_truly_better_by_at_least=HARM_TRUE),
                seconds=dict(critic=t_crit, simulation=t_sim, total=time.perf_counter() - t0),
                cpu_seconds=time.process_time() - c0)
    return out, info


def save(labels_arrays, info, pairs_rec, rep, world, out=C.OUT, extra=None, label=None):
    """Write the labels once (file + record, hashed, stamped with the current detect spec and code manifest). label:
    by default chosen by the folder (common.label_for)."""
    d = C.seed_dir(rep, world, out)
    label = C.label_for(d, label)
    if (d / FILE).exists() or (d / RECORD).exists():
        raise SystemExit(f"refused: labels exist in {d}; labels are written once")
    fsha, csha = C.save_npz(d / FILE, labels_arrays)
    rec = dict(label=label, kind="detect-labels", rep=int(rep), world=world, file=FILE, file_sha256=fsha,
               content_sha256=csha, pairs_file_sha256=pairs_rec["file_sha256"],
               pairs_content_sha256=pairs_rec["content_sha256"], info=info, reads_signals=False, **(extra or {}))
    rec.update(C.stamp())
    C.write_json(d / RECORD, rec)
    return rec


def load(rep, world, out=C.OUT):
    """Labels for the join (measures.py only): verified against the record, the sidecars and the SHA-256 named, and
    written under the current detect spec and code manifest (common.check_version)."""
    d = C.seed_dir(rep, world, out)
    rec = C.read_json(d / RECORD)
    if rec is None:
        raise SystemExit(f"refused: no labels at {d}; run the 'labels' phase first")
    C.check_version(rec, f"the labels in {d}")
    return C.load_npz(d / FILE, rec["file_sha256"], rec["content_sha256"]), rec


def refuse_before_signals(rep, world, out):
    """D12 (spec), enforced in this module: in the rehearsal's folder the labels of a seed and world wait until its
    Stage 1 signals are scored (signals_done.json, written by the signals phase under the current spec and code
    manifest). Stage 2's folder is exempt (spec, change C7)."""
    d = C.seed_dir(rep, world, out)
    if not C.order_rule_applies(d):
        return
    done = C.read_json(d / C.SIGNALS_DONE)
    if done is None:
        raise SystemExit(f"refused: the signals of seed {rep} {world} are not scored yet ({d / C.SIGNALS_DONE} is "
                         f"missing); in the rehearsal's folder the labels come after them (D12)")
    C.check_version(done, f"{d / C.SIGNALS_DONE}")


def run(ctx, world, out=C.OUT, extra=None, log_fn=C.log, label=None):
    """The labels phase of one seed and world: load the verified pair sets, compute, save once. label: by default
    chosen by the folder (common.label_for: Stage 2's label in runs/shift_explore/stage2). In the rehearsal's folder
    it refuses until the seed's Stage 1 signals are scored (refuse_before_signals). Runs only in a process that holds
    no signal code."""
    refuse_if_signals_loaded()
    refuse_before_signals(ctx.rep, world, out)
    arrays, prec = P.load(ctx.rep, world, out)
    lab, info = compute(ctx, world, arrays, log_fn)
    rec = save(lab, info, prec, ctx.rep, world, out, extra, label)
    log_fn(f"labels seed {ctx.rep} {world}: " + ", ".join(f"{k} {v['harmful']}/{v['unsupported']}/{v['n']}"
                                                         for k, v in info["counts"].items() if k.endswith("command"))
           + f" (harmful/past-80%/pairs; {info['seconds']['total']:.0f}s)")
    return rec
