"""EXPLORATORY BCA training arms for spec version 5: one quality and one seed per call.

EXPLORATORY: outside the pre-registered claims; the version-5 gates did not pass.

Why this exists. The version-5 report allows no BCA arm on any quality (bca_allowed and bca_lower_edge_allowed are
false everywhere), so the official stages `run.py bca` and `run.py bca-lower-edge` refuse. The owner answered
"Run the arms anyway, labelled exploratory (Recommended)" (recorded in runs/shift_iw_v5/approval.json,
exploratory_go). This runner trains the same arms the official stages would train, with the package's own functions
and the same arguments, settings, seeds and streams. The only differences from the official stages:
  1. the report's gate (M1, R1, R2 for the capped pull; M1, L1, L2 for the lower edge) is not consulted (the report's
     verdicts are copied into each record, for the reader);
  2. outputs go to runs/shift_iw_v5_explore/<quality>/... (never under runs/shift_iw_v5, which is only read);
  3. every record is labelled EXPLORATORY.

Arms (spec.json bca_training_arms and lower_edge.planned_arms):
  capped pull (co-trained TD3+BC with BCA, 100,000 updates; learners_iw.bca_cotrain, as run.stage_bca calls it)
    unweighted            weighting None
    exact_strong          per-step exact weights, strong mix (each training row's dose at its own threshold)
    exact_strong_single   single-threshold exact weights, strong mix (version 3's arm, a comparison)
  lower edge (actor-only against the frozen gapped critic, from the cloned actor; lower_edge.band_arms, as
  run.stage_bca_lower_edge calls it, all requested arms in one call, each arm in its own program)
    unweighted            R_i = R_u on every row
    exact_strong          R_i = R(w_strong(c_i)) from the gapped band's exact strong table
    exact_mild            R_i = R(w_mild(c_i)) from the exact mild table (shown only)

Inputs (read only, each checked as the official stage checks it): runs/shift_iw_v5/approval.json (verified by
learners_iw.verify_approval through run.approval_for, scopes 'bca' and 'bca_lower_edge' for this quality), iw/spec.json
(its sidecar), the seed's data, the gapped host, the gapped band and the room check's record and actors.

Outputs under --out (default /root/bca-shift/runs/shift_iw_v5_explore):
  <quality>/bca/<arm>/seed<k>/snapshot_050.msgpack, snapshot_100.msgpack, actor.msgpack (each with a .sha256 sidecar),
                                record.json (+ .sha256)
  <quality>/bca_lower_edge/<arm>/seed<k>/actor.msgpack (+ .sha256), record.json (+ .sha256)
A finished record is never redone (delete it to redo); records are written last.

--smoke: a few updates and steps (for checking the code and its inputs, not results); refused unless --out is a
scratch folder outside runs/shift_iw_v5 and runs/shift_iw_v5_explore.

Run from /root/bca-shift with PYTHONPATH=/root/bca-shift:
  python explore_arms.py --quality expert --seed 0 [--capped unweighted ...] [--lower-edge unweighted ...] [--out DIR]
  (no --capped and no --lower-edge: all six arms; --capped or --lower-edge alone with no names: none of that kind)
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import experiments.shift  # noqa: F401  (gymnasium path and version checks)

from experiments.shift import learners as LR
from experiments.shift_iw import data as D
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import lower_edge as LE
from experiments.shift_iw import run as R

LABEL = "EXPLORATORY: outside the pre-registered claims; the version-5 gates did not pass"
SMOKE_LABEL = "SMOKE TEST: a few updates only, for checking the code and its inputs; not results"
V5_ROOT = R.ROOT_DEFAULT                                   # /root/bca-shift/runs/shift_iw_v5 (read only)
EXPLORE_ROOT = D.REPO / "runs" / "shift_iw_v5_explore"
CAPPED_ARMS = R.BCA_ARMS                                   # ("unweighted", "exact_strong", "exact_strong_single")
LOWER_EDGE_ARMS = R.LE_ARMS                                # ("unweighted", "exact_strong", "exact_mild")
DIFFERENCES = [
    "the report's gate (bca_allowed / bca_lower_edge_allowed: M1, R1, R2 / M1, L1, L2) is not consulted",
    "outputs go to runs/shift_iw_v5_explore/<quality>/... instead of runs/shift_iw_v5/<quality>/...",
    "every record is labelled EXPLORATORY",
]
# smoke settings: only step counts shrink (refreshes and snapshots still happen); never used for the real run
SMOKE_SETTINGS = dict(updates=400, chunk=100, bca_warmup=100, bca_refresh=100, clone_steps=200,
                      actor_only_steps=200, actor_only_chunk=100)


def this_code():
    """SHA-256 of this file (the package's own files are in every record's code_sha256)."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def write_sidecar(path, digest):
    path = Path(path)
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n")


def check_out(out, smoke):
    """Refuse an output folder inside runs/shift_iw_v5 or the code tree; a smoke run also refuses the real output
    folder."""
    out = D.refuse_protected(out)
    # D.refuse_protected compares resolved paths with /root/bca-shift/experiments etc.; the package itself resolves
    # through the symlink to /mnt/c/.../shift_iw, so its real folder, the spec's folder and this runner's folder are
    # refused here by their resolved paths
    for root, why in [(V5_ROOT.resolve(), "the pre-registered run's folder (read only)"),
                      (Path("/mnt/c/Users/David/dev/BCA").resolve(), "the BCA repository"),
                      (Path(LI.PACKAGE).resolve(), "the experiment package (read only)"),
                      (Path(LI.SPEC_JSON).resolve().parent, "the specification's folder (read only)"),
                      (Path(__file__).resolve().parent, "the exploratory runner's code folder")]:
        if out == root or root in out.parents:
            raise R.Refused(f"refused: {out} is inside {why}")
    if smoke and (out == EXPLORE_ROOT.resolve() or EXPLORE_ROOT.resolve() in out.parents):
        raise R.Refused(f"refused: a smoke run writes to a scratch folder, not to {EXPLORE_ROOT}")
    return out


def exploratory_go(approval):
    """The owner's exploratory go from the verified approval file (the bytes learners_iw.verify_approval checked)."""
    blob = Path(approval.path).read_bytes()
    if hashlib.sha256(blob).hexdigest() != approval.file_sha256:
        raise R.Refused("refused: approval.json changed after it was verified")
    go = json.loads(blob.decode("utf8")).get("exploratory_go")
    answer = (go or {}).get("answer_verbatim", "")
    if not isinstance(answer, str) or "exploratory" not in answer.lower():
        raise R.Refused("refused: approval.json holds no exploratory go from the owner (exploratory_go."
                        "answer_verbatim)")
    return go


def approval_of(ctx, scope):
    """run.approval_for (verify_approval: sidecar, the owner's words, spec and code SHA-256, the scope naming this
    quality) plus the owner's exploratory go."""
    approval = R.approval_for(ctx, scope)
    return approval, exploratory_go(approval)


def report_verdicts(ctx, spec):
    """The pre-registered report's verdicts for this quality, copied into each record (not a gate)."""
    rep = R.finished(R.report_path(ctx), spec)
    if rep is None:
        return dict(report="missing")
    return dict(report_sha256=LI.sha256_file(R.report_path(ctx)), bca_allowed=rep.get("bca_allowed"),
                bca_lower_edge_allowed=rep.get("bca_lower_edge_allowed"),
                marks={k: R._get(rep, "marks", k, "status") for k in ("M1", "R1", "R2", "L1", "L2", "I1", "L3")})


def common_fields(ctx, spec, seed, go, cfg, smoke):
    return dict(label=LABEL, exploratory=True, smoke=bool(smoke), smoke_label=SMOKE_LABEL if smoke else None,
                quality=ctx.quality, seed=int(seed), exploratory_go=go, differences_from_official_stage=DIFFERENCES,
                pre_registered_report=report_verdicts(ctx, spec), settings=cfg.as_dict(),
                explore_code_sha256={Path(__file__).name: this_code()})


def capped_dir(out, quality, arm, seed):
    return Path(out) / quality / "bca" / arm / f"seed{int(seed)}"


def lower_edge_dir(out, quality, arm, seed):
    return Path(out) / quality / "bca_lower_edge" / arm / f"seed{int(seed)}"


# ------------------------------------------------------------------------------------------------ capped pull

def train_capped(ctx, out, seed, arms, smoke):
    """run.stage_bca's body without the report gate: the same approval check (scope 'bca' for this quality), data,
    weighting (run.bca_weighting with spec.json's grid ratio) and call (learners_iw.bca_cotrain)."""
    spec = R.spec_digest(ctx)
    approval, go = approval_of(ctx, "bca")
    body = R.frozen_spec(ctx, spec)
    grid_ratio = R.spec_per_step(body)["grid_ratio"]
    todo = [a for a in arms if R.finished(capped_dir(out, ctx.quality, a, seed) / "record.json", spec) is None]
    if not todo:
        ctx.log(f"capped pull {ctx.quality} seed {seed}: {list(arms)} already finished")
        return
    data_rec = R.require(R.data_record_path(ctx, seed), spec, f"data for seed {seed}")
    data = R.load_data(ctx, seed, spec)
    prep = D.prepare(data, "gap", ctx.cfg)
    inputs = R.links(ctx, [R.data_record_path(ctx, seed)])
    ctx.log(f"capped pull {ctx.quality} seed {seed}: inputs {inputs}; data world sha {data_rec['world_json_sha256']}; "
            f"{prep.n} kept training rows, {len(prep.bank.obs)} bank rows; grid ratio {grid_ratio}; arms {todo}")
    for arm in todo:
        started = time.perf_counter()
        d = capped_dir(out, ctx.quality, arm, seed)
        weighting = R.bca_weighting(data, prep, arm, grid_ratio)
        result, state, _ = LI.bca_cotrain(prep, ctx.cfg, seed, approval, weighting=weighting, out_dir=d, log=ctx.log,
                                          name=f"BCA {arm}", stage="bca")
        for snap in result["snapshots"]:
            if "file_sha256" in snap:
                total = result["updates"]
                write_sidecar(d / f"snapshot_{int(round(100 * snap['update'] / total)):03d}.msgpack",
                              snap["file_sha256"])
        actor_sha = LR.save_tree(d / "actor.msgpack", state.native.actor.params)
        write_sidecar(d / "actor.msgpack", actor_sha)
        R.write_record(d / "record.json", R.stage_record(
            "explore-bca", spec, started, arm=arm, use_of_band="capped pull (BCA's own dose on the pull toward the "
                                                              "recorded command), co-trained",
            result=result, actor_sha256=actor_sha, actor_tree_sha256=LR.tree_sha256(state.native.actor.params),
            inputs_sha256_v5=inputs, **common_fields(ctx, spec, seed, go, ctx.cfg, smoke)))
        ctx.log(f"capped pull {ctx.quality} seed {seed} {arm}: {'FAILED' if result['failed'] else 'finished'} in "
                f"{time.perf_counter() - started:.0f}s (uncertified refreshes {result['uncertified_refreshes']}, "
                f"refused rows max {result.get('refused_rows_max')})")


# ------------------------------------------------------------------------------------------------ lower edge

def train_lower_edge(ctx, out, seed, arms, smoke):
    """run.stage_bca_lower_edge's body without the report gate: the same approval check (scope 'bca_lower_edge' for
    this quality), inputs (data, gapped host, gapped band, room check), arm thresholds (run.lower_edge_arm) and call
    (lower_edge.band_arms with spec.json's k and check (e) rule, the room check's no-caution actor)."""
    spec = R.spec_digest(ctx)
    approval, go = approval_of(ctx, LE.SCOPE)
    body = R.frozen_spec(ctx, spec)
    le = R.spec_lower_edge(body)
    todo = [a for a in arms if R.finished(lower_edge_dir(out, ctx.quality, a, seed) / "record.json", spec) is None]
    if not todo:
        ctx.log(f"lower edge {ctx.quality} seed {seed}: {list(arms)} already finished")
        return
    R.require(R.data_record_path(ctx, seed), spec, f"data for seed {seed}")
    R.require_host(ctx, "gap", seed, spec)
    R.require_band(ctx, "gap", seed, spec)
    R.require(R.room_dir(ctx, seed) / "record.json", spec, f"the room check of seed {seed}")
    started = time.perf_counter()
    data = R.load_data(ctx, seed, spec)
    prep = D.prepare(data, "gap", ctx.cfg)
    host = R.load_host_state(ctx, "gap", seed, spec)
    brec = R.require_band(ctx, "gap", seed, spec)
    band = LI.load_band(R.band_dir(ctx, "gap", seed) / "band.msgpack", brec["band_sha256"], host, prep, ctx.cfg, seed,
                        approval, brec["band"], stage=LE.SCOPE)
    R_u = R.num(brec["thresholds"]["unweighted"]["threshold"])
    specs = [R.lower_edge_arm(data, prep, brec, a, R_u, None) for a in todo]
    _, room_params = R.load_room_actors(ctx, seed, spec)
    inputs = R.links(ctx, [R.band_dir(ctx, "gap", seed) / "record.json", R.host_dir(ctx, "gap", seed) / "record.json",
                           R.room_dir(ctx, seed) / "record.json", R.data_record_path(ctx, seed)])
    ctx.log(f"lower edge {ctx.quality} seed {seed}: inputs {inputs}; R_u {R_u:.6g}; k {le['k']}; arms {todo}")
    try:
        params, rec = LE.band_arms(prep, band, host.critic.params, specs, le["k"], ctx.cfg, seed, approval,
                                   none_params=room_params["none"], check_e_rule=le["rule"].check_e, stage=LE.SCOPE,
                                   log=ctx.log)
    except ValueError as error:
        raise R.Refused(f"refused: the lower edge's BCA arms of seed {seed}: {error}") from error
    shared = {k: v for k, v in rec.items() if k != "arms"}
    for arm in todo:
        d = lower_edge_dir(out, ctx.quality, arm, seed)
        sha = LR.save_tree(d / "actor.msgpack", params[arm])
        write_sidecar(d / "actor.msgpack", sha)
        R.write_record(d / "record.json", R.stage_record(
            "explore-bca-lower-edge", spec, started, arm=arm,
            use_of_band="lower edge (penalty k [U(s, pi(s)) - U(s, a_i)]_+ in the actor's objective), actor-only on "
                        "the frozen gapped critic",
            result=dict(shared, arm=rec["arms"][arm], arms_in_this_job=todo), actor_sha256=sha,
            inputs_sha256_v5=inputs, **common_fields(ctx, spec, seed, go, ctx.cfg, smoke)))
    ctx.log(f"lower edge {ctx.quality} seed {seed}: {todo} finished in {time.perf_counter() - started:.0f}s; "
            + "; ".join(f"{a}: refused rows {rec['arms'][a]['refused_rows']}, R mean {rec['arms'][a]['R_mean']:.4g}, "
                        f"check (e) passed {R._get(rec['arms'][a], 'check_e', 'passed')}" for a in todo))


# ------------------------------------------------------------------------------------------------ CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=f"{LABEL}. Train the BCA arms of one quality and seed.")
    p.add_argument("--quality", required=True, choices=R.QUALITIES)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--capped", nargs="*", choices=CAPPED_ARMS, default=None,
                   help="capped-pull arms (default: all three, unless only --lower-edge is given)")
    p.add_argument("--lower-edge", nargs="*", choices=LOWER_EDGE_ARMS, default=None,
                   help="lower-edge arms, trained in one call (default: all three, unless only --capped is given)")
    p.add_argument("--out", default=str(EXPLORE_ROOT))
    p.add_argument("--smoke", action="store_true", help="a few updates and steps, to a scratch --out only")
    a = p.parse_args(argv)
    if a.capped is None and a.lower_edge is None:
        capped, lower = list(CAPPED_ARMS), list(LOWER_EDGE_ARMS)
    else:
        capped, lower = list(a.capped or []), list(a.lower_edge or [])
    if a.seed not in R.SEEDS:
        print(f"refused: seed {a.seed} is not one of the specification's seeds {R.SEEDS}", file=sys.stderr)
        return 2
    cfg = LI.settings(**SMOKE_SETTINGS) if a.smoke else LI.settings()
    try:
        out = check_out(a.out, a.smoke)
        ctx = R.Context(root=V5_ROOT, quality=a.quality, cfg=cfg)
        ctx.log(f"{LABEL}{' -- ' + SMOKE_LABEL if a.smoke else ''}")
        ctx.log(f"quality {a.quality} seed {a.seed}: capped {capped}, lower edge {lower}; reading {V5_ROOT}, writing "
                f"{out}; settings {cfg.as_dict() if a.smoke else 'learners_iw.settings() (the specification)'}")
        if capped:
            train_capped(ctx, out, a.seed, capped, a.smoke)
        if lower:
            train_lower_edge(ctx, out, a.seed, lower, a.smoke)
    except (R.Refused, LI.ApprovalError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
