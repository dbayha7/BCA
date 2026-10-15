"""Learning curves of the ThrusterChain weighting test (EXPLORATORY; the owner asked on 2026-10-14: "im going to need to
see learning curves as well comprehensively for all possible env setups").

Every curve comes from rerunning a run exactly as it was trained, with checkpoints added, and evaluating each
checkpoint by the package's own exact evaluation (learners_iw.evaluate_actor: the 100-step return from the 1,000 fixed
starts of each of the three start banks: no shift, mild, strong). A rerun that has a saved final actor must end at that
actor bit for bit, and the record says whether it does.

Groups (one quality and seed per call):
  actor   actor-only runs (20,000 steps from the cloned actor, against the frozen gapped critic), checkpoints every
          1,000 steps (each 4,000-step chunk split in four; refreshes stay at the chunk starts):
            oracle      the lower-edge room check's float64 program: le_none (no caution), le_oracle, le_oracle_x1.5
            band        version 5's lower edge with BCA's band: unweighted, exact_strong, exact_mild
            cmd         the first return test: cmd_strong, cmd_train, untargeted          (top two qualities only)
            f1          the rerun with the fixed fallback: A2_cmd_strong, B2_cmd_train, C2_untargeted
            est         estimated weights: A3_estimated, C3_untargeted
  cotrain co-trained runs (100,000 updates), snapshots every 5,000 updates (the runs' own chunk, so the computation is
          unchanged): host_gap, host_full (plain TD3+BC), capped_unweighted, capped_exact_strong (BCA's capped pull)

Usage (from /root/bca-shift, PYTHONPATH=/root/bca-shift, JAX_PLATFORMS=cpu):
    python curves.py actor <quality> <seed> [<family> ...]
    python curves.py cotrain <quality> <seed> <run>
Outputs: runs/shift_iw_v5_explore/curves/<quality>/seed<k>/<family or run>.json
"""
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from experiments.shift import learners as LR
from experiments.shift_iw import data as D
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import lower_edge as LE
from experiments.shift_iw import run as RN

CHANNEL = Path("/mnt/c/Users/David/dev/bca-shift-work/channel")
HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, CHANNEL / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load("cle", "command_le_train.py")
F1 = _load("cle_f1", "command_le_train_f1.py")
EST = _load("cle_est", "command_le_train_est.py")

EX = RN.ROOT_DEFAULT.parent / "shift_iw_v5_explore"
OUT = EX / "curves"
LABEL = ("EXPLORATORY learning curves: reruns of existing runs with checkpoints added (each checked against its saved "
         "final actor), and the new arms on the three lower qualities (shown only, no pass marks)")
SUB = 1_000                      # actor-only checkpoint spacing
CO_SNAPSHOTS = tuple(round(0.05 * k, 2) for k in range(1, 21))   # every 5,000 of 100,000 updates
TOP = ("medium-expert", "expert")
FAMILIES = ("oracle", "band", "cmd", "f1", "est")


def this_code():
    files = [Path(__file__).resolve()] + [CHANNEL / f for f in ("command_le_train.py", "command_le_train_f1.py",
                                                                 "command_le_train_est.py", "command_band_check.py")]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


class Evaluator:
    def __init__(self, ctx, spec, seed):
        self.banks, sb = D.load_start_banks(ctx.banks_root)
        self.anchors = sb["anchors"]
        self.prep = D.prepare(RN.load_data(ctx, seed, spec), "gap", ctx.cfg)

    def __call__(self, params):
        ev = LI.evaluate_actor(params, self.prep, self.banks, self.anchors)
        return {law: dict(J=float(v["J"]), points=float(v["points"]), J_se=float(v["J_se"])) for law, v in ev.items()}


def curve_entry(steps, evals, verified=None, note=None, extra=None):
    laws = sorted(evals[0])
    out = dict(steps=steps, J={l: [e[l]["J"] for e in evals] for l in laws},
               points={l: [e[l]["points"] for e in evals] for l in laws},
               J_se={l: [e[l]["J_se"] for e in evals] for l in laws}, final_matches_saved=verified)
    if note:
        out["note"] = note
    if extra:
        out.update(extra)
    return out


def split(rows_chunk):
    n = rows_chunk.shape[0] // SUB
    return [rows_chunk[i * SUB:(i + 1) * SUB] for i in range(n)]


# ------------------------------------------------------------------------------------------------ actor-only families

def family_oracle(wd, ev, log):
    """le_none / le_oracle / le_oracle_x1.5, side by side in the room check's float64 program (lower_edge.Runner)."""
    ctx, seed = wd.ctx, wd.seed
    rec = RN.require(RN.room_le_dir(ctx, seed) / "record.json", wd.spec, "the lower edge's room check")
    if rec["room_lower_edge"].get("stopped"):
        return {}
    saved = RN.controllers_of(ctx, "room_lower_edge", seed, wd.spec)
    shared, _ = LE.oracle_shared(wd.prep, wd.host.critic.params, LE.W.PHYSICS)
    clone, _ = wd.start()
    runner = LE.Runner(wd.prep, clone, wd.host.critic.params, ctx.cfg, seed, "oracle", shared)
    le = RN.spec_lower_edge(RN.frozen_spec(ctx, wd.spec))
    arms = [dict(name=n, k=le["rule"].k[n]) for n in LE.ROOM_ARMS]
    n = len(arms)
    states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), runner.start)
    tree = runner.arm_tree(arms)
    evals = {a["name"]: [ev(clone)] for a in arms}
    steps = [0]
    with runner.context():
        program = runner.program(n)
        for c in range(runner.rows.shape[0]):
            for r in split(runner.rows[c]):
                states, _ = program(runner.critic, states, wd.prep.train, runner.shared, r, tree)
                steps.append(steps[-1] + SUB)
                for i, a in enumerate(arms):
                    evals[a["name"]].append(ev(jax.tree_util.tree_map(lambda x, i=i: x[i], states.params)))
            log(f"oracle family: step {steps[-1]}")
    out = {}
    for i, a in enumerate(arms):
        final = jax.tree_util.tree_map(lambda x, i=i: x[i], states.params)
        same = LE.max_param_diff(final, saved[a["name"]]) == 0.0 if a["name"] in saved else None
        out[a["name"]] = curve_entry(steps, evals[a["name"]], same)
    return out


def family_band(wd, ev, log):
    """Version 5's lower edge with BCA's band (lower_edge.band_arms: each arm alone, R from run.lower_edge_arm)."""
    ctx, seed = wd.ctx, wd.seed
    R_u = wd.R_u
    clone, _ = wd.start()
    shared = LE.band_shared(wd.band, LE.SCOPE)
    runner = LE.Runner(wd.prep, clone, wd.host.critic.params, ctx.cfg, seed, "band", shared, band=wd.band,
                       stage=LE.SCOPE)
    out = {}
    for arm in RN.LE_ARMS:
        a = RN.lower_edge_arm(wd.data, wd.prep, wd.brec, arm, R_u, None)
        R, _, _ = LE.capped_thresholds(a["R"], a["pool"])
        R32 = LE.as_float32_thresholds(R)
        spec_ = dict(name=arm, k=1.0, R=R32, u_logged=LE.logged_widths(wd.band, wd.prep, R32, stage=LE.SCOPE))
        states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (1,) + jnp.shape(x)), runner.start)
        tree = runner.arm_tree([spec_])
        evals, steps = [ev(clone)], [0]
        program = runner.program(1)
        for c in range(runner.rows.shape[0]):
            for r in split(runner.rows[c]):
                states, _ = program(runner.critic, states, wd.prep.train, runner.shared, r, tree)
                steps.append(steps[-1] + SUB)
                evals.append(ev(jax.tree_util.tree_map(lambda x: x[0], states.params)))
        final = jax.tree_util.tree_map(lambda x: x[0], states.params)
        p = EX / ctx.quality / "bca_lower_edge" / arm / f"seed{seed}"
        same = None
        if (p / "record.json").exists():
            r = json.loads((p / "record.json").read_text())
            same = LE.max_param_diff(final, LR.load_tree(p / "actor.msgpack", LR.actor_template(),
                                                         r["actor_sha256"])) == 0.0
        out[f"v5_{arm}"] = curve_entry(steps, evals, same)
        log(f"band family: {arm} done (matches saved: {same})")
    return out


def _my_family(wd, ev, log, mod, arms, laws, est=None, refused_key="refused_share"):
    """The command-weighted arms (command_le_train / _f1 / _est), each rerun with checkpoints."""
    cfg = wd.ctx.cfg
    out = {}
    shares = {}
    for arm in arms:
        rows, _ = wd.rows()
        clone, _ = wd.start()
        state = TrainState.create(apply_fn=LR.ACTOR.apply, params=clone, tx=LR.adam(cfg.lr))
        evals, steps, chunk_info = [ev(clone)], [0], []
        untargeted = arm in ("untargeted", "C2_untargeted", "C3_untargeted")
        for c in range(rows.shape[0]):
            if untargeted:
                src = {"untargeted": "cmd_strong", "C2_untargeted": "A2_cmd_strong",
                       "C3_untargeted": "A3_estimated"}[arm]
                tree, picked = wd.untargeted_tree(shares[src][c], c)
                info = dict(target_share=shares[src][c], picked_share=picked)
                prog = mod.program(wd.args, "untargeted")
            else:
                if est is not None:
                    tree, info, _ = est.refresh(state.params, c)
                elif mod is T:
                    tree, info, _ = wd.refresh(state.params, laws[arm])
                else:
                    tree, info, _, _ = F1.refresh(wd, state.params, laws[arm])
                prog = mod.program(wd.args, "band")
            chunk_info.append({k: v for k, v in info.items() if isinstance(v, (int, float)) or v is None})
            for r in split(rows[c]):
                state, _ = prog(wd.host.critic.params, state, wd.prep.train, wd.shared, r, tree)
                steps.append(steps[-1] + SUB)
                evals.append(ev(state.params))
        if not untargeted:
            shares[arm] = [ci[refused_key] for ci in chunk_info]
        p = mod.arm_dir(wd.ctx.quality, arm, wd.seed)
        same = None
        if (p / "record.json").exists():
            r = json.loads((p / "record.json").read_text())
            same = LE.max_param_diff(state.params, LR.load_tree(p / "actor.msgpack", LR.actor_template(),
                                                                r["actor_sha256"])) == 0.0
        else:   # a new arm (the lower qualities): save it like the original runner would
            p.mkdir(parents=True, exist_ok=True)
            sha = LR.save_tree(p / "actor.msgpack", state.params)
            (p / "record.json").write_text(json.dumps(dict(label=LABEL + " (trained here, new)", arm=arm,
                                                           quality=wd.ctx.quality, seed=wd.seed, actor_sha256=sha,
                                                           chunks=chunk_info), indent=1, default=float),
                                           encoding="utf-8")
        out[arm] = curve_entry(steps, evals, same, extra=dict(chunks=chunk_info))
        log(f"{arm} done (matches saved: {same})")
    return out


def family_cmd(wd, ev, log):
    return _my_family(wd, ev, log, T, ("cmd_strong", "cmd_train", "untargeted"), T.LAWS)


def family_f1(wd, ev, log):
    return _my_family(wd, ev, log, F1, F1.ARMS, F1.LAWS, refused_key="no_credit_share")


def family_est(wd, ev, log):
    return _my_family(wd, ev, log, EST, EST.ARMS, {}, est=EST.Est(wd), refused_key="no_credit_share")


# ------------------------------------------------------------------------------------------------ co-trained

def cotrain(quality, seed, run, log):
    ctx = RN.Context(quality=quality)
    spec = RN.spec_digest(ctx)
    cfg = LI.settings(snapshots=CO_SNAPSHOTS)
    data = RN.load_data(ctx, seed, spec)
    tmp = OUT / quality / f"seed{seed}" / f"_tmp_{run}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    if run in ("host_gap", "host_full"):
        kind = run.split("_")[1]
        prep = D.prepare(data, RN.HOST_KINDS[kind], cfg)
        result, state, _ = LI.train_host(prep, cfg, seed, out_dir=tmp, log=log, name=f"curve rerun {run}")
        saved_dir = RN.host_dir(ctx, kind, seed)
        saved = {s["update"]: s["file_sha256"] for s in RN.require_host(ctx, kind, seed, spec)["result"]["snapshots"]}
        template = LR.agent_template(cfg)
        actor_of = lambda st: st.actor.params
    elif run in ("capped_unweighted", "capped_exact_strong"):
        arm = run[len("capped_"):]
        prep = D.prepare(data, "gap", cfg)
        approval = RN.approval_for(ctx, "bca")
        grid_ratio = RN.spec_per_step(RN.frozen_spec(ctx, spec))["grid_ratio"]
        weighting = RN.bca_weighting(data, prep, arm, grid_ratio)
        result, state, _ = LI.bca_cotrain(prep, cfg, seed, approval, weighting=weighting, out_dir=tmp, log=log,
                                          name=f"curve rerun {run}", stage="bca")
        saved_dir = EX / quality / "bca" / arm / f"seed{seed}"
        rec = json.loads((saved_dir / "record.json").read_text())
        saved = {s["update"]: s["file_sha256"] for s in rec["result"]["snapshots"]}
        template = LI.bca_template(cfg, seed)
        actor_of = lambda st: st.native.actor.params
    else:
        raise SystemExit(f"unknown co-trained run {run!r}")
    ev = Evaluator(ctx, spec, seed)
    total = result["updates"]
    steps, evals, matches = [], [], {}
    for snap in result["snapshots"]:
        u = snap["update"]
        f = tmp / f"snapshot_{int(round(100 * u / total)):03d}.msgpack"
        st = LR.load_tree(f, template, snap["file_sha256"])
        steps.append(int(u))
        evals.append(ev(actor_of(st)))
        if u in saved:
            matches[str(u)] = snap["file_sha256"] == saved[u]
    shutil.rmtree(tmp)
    entry = curve_entry(steps, evals, all(matches.values()) if matches else None,
                        extra=dict(snapshot_matches=matches, failed=bool(result.get("failed"))))
    return {run: entry}


# ------------------------------------------------------------------------------------------------ main

def write(quality, seed, name, body):
    d = OUT / quality / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(dict(label=LABEL, quality=quality, seed=int(seed), group=name,
                                                    code=this_code(), curves=body,
                                                    finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
                                               indent=1, default=float), encoding="utf-8")


def main(argv):
    mode, quality, seed = argv[0], argv[1], int(argv[2])
    log = lambda m: print(time.strftime("%H:%M:%S ") + f"{quality} seed {seed}: " + str(m), flush=True)
    if mode == "actor":
        fams = argv[3:] or list(FAMILIES)
        wd = T.World(quality, seed)
        ev = Evaluator(wd.ctx, wd.spec, seed)
        for fam in fams:
            if fam == "cmd" and quality not in TOP:
                continue
            if (OUT / quality / f"seed{seed}" / f"{fam}.json").exists():
                log(f"{fam} already finished")
                continue
            t0 = time.perf_counter()
            body = globals()[f"family_{fam}"](wd, ev, log)
            write(quality, seed, fam, body)
            log(f"{fam} finished in {time.perf_counter() - t0:.0f}s: " +
                ", ".join(f"{k} matches {v['final_matches_saved']}" for k, v in body.items()))
    elif mode == "cotrain":
        run = argv[3]
        if (OUT / quality / f"seed{seed}" / f"{run}.json").exists():
            log(f"{run} already finished")
            return
        t0 = time.perf_counter()
        body = cotrain(quality, seed, run, log)
        write(quality, seed, run, body)
        log(f"{run} finished in {time.perf_counter() - t0:.0f}s: matches {body[run]['final_matches_saved']}")
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
