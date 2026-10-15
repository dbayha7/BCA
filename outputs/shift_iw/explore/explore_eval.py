"""EXPLORATORY evaluation of the BCA arms of explore_arms.py: one quality and one seed per call.

EXPLORATORY: outside the pre-registered claims; the version-5 gates did not pass.

The package's exact evaluation, as run.stage_evaluate runs it for its 'bca' and 'bca_lower_edge' groups: exact
100-step returns (learners_iw.evaluate_actor: truth.rollout in float64) of each final actor from every fixed start of
the shared start banks of all three mixes ('none', 'mild', 'strong'), with points beside them (0 = the random quality,
100 = the stall-aware controller, the banks' anchors). The prepared data use the gapped normalisation rule
(data.prepare(data, 'gap')), as every controller here was trained with it. Actors are loaded as run.controllers_of
loads them: the capped pull's from snapshot_100.msgpack (the co-trained state's actor), the lower edge's from
actor.msgpack, each checked against its record's SHA-256.

Inputs: the explore records under --out (each must be finished, labelled EXPLORATORY and name the frozen spec), and,
read only, runs/shift_iw_v5's start banks and the seed's data.
Outputs: <out>/<quality>/evaluate/seed<k>/<bca|bca_lower_edge>.json (+ .sha256): {evaluations: {name: {law: {J, J_se,
points, points_se, finite, starts}}}} with names 'bca_<arm>' and 'le_bca_<arm>', as the official records name them.
Arms not finished yet are listed in 'missing_arms'; such a record is redone on the next call.

Run from /root/bca-shift with PYTHONPATH=/root/bca-shift:
  python explore_eval.py --quality expert --seed 0 [--groups bca bca_lower_edge] [--out DIR]
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import experiments.shift  # noqa: F401  (gymnasium path and version checks)

from experiments.shift import learners as LR
from experiments.shift_iw import data as D
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import run as R

import explore_arms as XA

GROUPS = ("bca", "bca_lower_edge")


def eval_path(out, quality, seed, group):
    return Path(out) / quality / "evaluate" / f"seed{int(seed)}" / f"{group}.json"


def explore_record(path, spec):
    """A finished explore record (sidecar checked) under this spec, labelled exploratory; else None."""
    rec = R.finished(path, spec)
    if rec is None:
        return None
    if rec.get("label") != XA.LABEL or rec.get("exploratory") is not True:
        raise R.Refused(f"refused: {path} is not labelled exploratory")
    return rec


def controllers(out, quality, seed, group, spec):
    """({name: actor params}, {name: record path}, [missing arms], whether any is a smoke run) of a group's finished
    explore arms."""
    params, paths, missing, smoke = {}, {}, [], False
    if group == "bca":
        for arm in XA.CAPPED_ARMS:
            d = XA.capped_dir(out, quality, arm, seed)
            rec = explore_record(d / "record.json", spec)
            if rec is None:
                missing.append(arm)
                continue
            smoke = smoke or bool(rec.get("smoke"))
            if rec["result"]["failed"]:
                missing.append(f"{arm} (training failed at update {rec['result'].get('failed_at')})")
                continue
            snap = rec["result"]["snapshots"][-1]
            if snap["update"] != rec["result"]["final_update"]:
                raise R.Refused(f"refused: {d} has no final snapshot")
            # run.controllers_of: the co-trained state's actor from snapshot_100.msgpack
            state = LR.load_tree(d / "snapshot_100.msgpack", LI.bca_template(LI.settings(), seed), snap["file_sha256"])
            actor = state.native.actor.params
            if LR.tree_sha256(actor) != snap["actor_sha256"]:
                raise R.Refused(f"refused: the actor in {d / 'snapshot_100.msgpack'} differs from its record")
            params[f"bca_{arm}"], paths[f"bca_{arm}"] = actor, d / "record.json"
    elif group == "bca_lower_edge":
        for arm in XA.LOWER_EDGE_ARMS:
            d = XA.lower_edge_dir(out, quality, arm, seed)
            rec = explore_record(d / "record.json", spec)
            if rec is None:
                missing.append(arm)
                continue
            smoke = smoke or bool(rec.get("smoke"))
            params[f"le_bca_{arm}"] = LR.load_tree(d / "actor.msgpack", LR.actor_template(), rec["actor_sha256"])
            paths[f"le_bca_{arm}"] = d / "record.json"
    else:
        raise R.Refused(f"refused: unknown group {group!r}")
    return params, paths, missing, smoke


def evaluate(ctx, out, seed, groups):
    spec = R.spec_digest(ctx)
    R.require(R.banks_record_path(ctx), spec, "the start banks")
    R.require(R.data_record_path(ctx, seed), spec, f"data for seed {seed}")
    todo = []
    for g in groups:
        rec = R.finished(eval_path(out, ctx.quality, seed, g), spec)
        if rec is None or rec.get("missing_arms"):
            todo.append(g)
    if not todo:
        ctx.log(f"evaluate {ctx.quality} seed {seed}: {list(groups)} already finished")
        return
    banks, sb = D.load_start_banks(ctx.banks_root)
    anchors = sb["anchors"]
    data = R.load_data(ctx, seed, spec)
    prep = D.prepare(data, "gap", ctx.cfg)   # every controller here uses the gapped normalisation rule
    for group in todo:
        started = time.perf_counter()
        params, paths, missing, smoke = controllers(out, ctx.quality, seed, group, spec)
        if not params:
            ctx.log(f"evaluate {ctx.quality} {group} seed {seed}: no finished arm yet ({missing})")
            continue
        ev = {name: LI.evaluate_actor(p, prep, banks, anchors) for name, p in params.items()}
        inputs = {str(Path(p).relative_to(out)): LI.sha256_file(p) for p in paths.values()}   # under --out
        inputs_v5 = R.links(ctx, [R.banks_record_path(ctx), R.data_record_path(ctx, seed)])   # under shift_iw_v5
        R.write_record(eval_path(out, ctx.quality, seed, group), R.stage_record(
            "explore-evaluate", spec, started, label=XA.LABEL, exploratory=True, smoke=smoke,
            smoke_label=XA.SMOKE_LABEL if smoke else None, quality=ctx.quality, seed=int(seed), group=group,
            evaluations=ev, missing_arms=missing, inputs_sha256=inputs, inputs_sha256_v5=inputs_v5,
            start_banks_json_sha256=LI.sha256_file(ctx.banks_root / "start_banks.json"),
            explore_code_sha256={Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 Path(XA.__file__).name: XA.this_code()}))
        ctx.log(f"evaluate {ctx.quality} {group} seed {seed}: finished in {time.perf_counter() - started:.0f}s; "
                + "; ".join(f"{n}: strong J {e['strong']['J']:.2f} ({e['strong']['points']:.2f} points)"
                            for n, e in ev.items()) + (f"; MISSING {missing}" if missing else ""))


def main(argv=None):
    p = argparse.ArgumentParser(description=f"{XA.LABEL}. Evaluate the BCA arms of one quality and seed.")
    p.add_argument("--quality", required=True, choices=R.QUALITIES)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--groups", nargs="*", choices=GROUPS, default=list(GROUPS))
    p.add_argument("--out", default=str(XA.EXPLORE_ROOT))
    p.add_argument("--smoke", action="store_true", help="only to allow a scratch --out (evaluation itself is exact)")
    a = p.parse_args(argv)
    try:
        out = XA.check_out(a.out, a.smoke)
        ctx = R.Context(root=XA.V5_ROOT, quality=a.quality)   # learners_iw.settings(), as run.stage_evaluate
        ctx.log(XA.LABEL)
        evaluate(ctx, out, a.seed, a.groups)
    except (R.Refused, LI.ApprovalError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
