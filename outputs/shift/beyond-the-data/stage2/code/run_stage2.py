"""Stage 2 (the flags claim): every phase, in the pre-registered order. EXPLORATORY on gate seeds 0-4, CONFIRMATORY on
fresh seeds 5-9, under runs/shift_explore/stage2/prereg.json.

Every phase first loads prereg.json (verified against its sidecar) and refuses unless it is final (approvable: no
missing pin). Every phase that fits, reads or trains with BCA (marked BCA below) also calls stage2_common.require_approval():
approval.json must verify against the CURRENT prereg.json's SHA-256 and every pinned script must be unchanged;
best_standin.json must already exist (the rival is chosen before any BCA run); and every pinned component of another
job (room test, fallback diagnostic, stand-ins, detection script) must still match the pins prereg.json records
(prereg_payload.check_pins). Phases write once and refuse to overwrite, and a write-once phase refuses while its
inputs are incomplete (a recorded failure is an input; a file not yet written is not):
  - detect_marks needs, for every seed of the group, either a recorded failure (a band fit that failed, a fresh seed
    whose data or learners failed) or a complete set: the width scores, tau and the labels, for the stalling world's
    fixed and host critics and the no-stall world's fixed critic, and the five stand-ins' score records;
  - deploy needs the version decision (the uniform band is used only if neither version passed);
  - pooled needs the group's detection marks and every seed's deployment values or a recorded failure (the policy
    marks need every seed's values.json, stop.json or a recorded fresh-seed failure).
Gate seeds 0-4 read the detection rehearsal's own pair sets, labels and stand-in scores, which must already exist:
Stage 2 never builds or writes anything in the rehearsal's folder.

  no BCA   fresh_data --seed K            (K in 5-9) new logs of both worlds: gap, removals, checks, world marks
  no BCA   fresh_learners --seed K        (K in 5-9) critics, clones, the no-caution actors, plain TD3+BC
  no BCA   select_standin                 the best data-only stand-in by plan 4.2's rule -> best_standin.json
  no BCA   detect_pairs --seed K --world W             the locked pair sets (the rehearsal's for seeds 0-4)
  no BCA   detect_labels --seed K --world W            the labels, in a process with no signal code (rehearsal's
                                                        for seeds 0-4; computed for 5-9)
  no BCA   detect_standins --seed K --world W          the stand-ins' scores on the same pairs (rehearsal's for 0-4)
  BCA      detect_fit --seed K --world W --critic fixed|host     BCA's band (uniform), bank scores, tau
  BCA      detect_fit_weighted --seed K --world W                 the weighted version (only when declared)
  BCA      detect_score --seed K --world W --critic C --version V  BCA's width at the locked pair sets
  BCA      detect_marks --group exploratory|confirmatory --version V   marks, claim rule, version decision
  BCA      policy_train / policy_eval / policy_no_stall_train / policy_no_stall_eval --seed K
  BCA      deploy --seed K --world W     Variant B
  BCA      host_train / host_eval --seed K --world W
  BCA      pooled --group exploratory|confirmatory

Run conditions (fixed in the pre-registration): the weighted version runs only if the uniform version fails direction
or size on seeds 0-4; the policy test and the full host run only with a version that passed direction and size, and
only if the room test's pooled look (runs/shift_explore/room/pooled.json, its pinned spec) passed its room, recovery
and keeps-improvement marks; the deployment check runs in any case. Seeds 5-9 use the version and every setting
frozen on seeds 0-4.
"""

import argparse
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_common as C  # noqa: E402

GROUPS = dict(exploratory=C.EXPLORATORY_SEEDS, confirmatory=C.CONFIRMATORY_SEEDS)
# The policy test's rivals: the five stand-in forms that carry the stand-ins test's mark. Since that test's spec v2
# the user chooses them there (its approval: 'approve F1-F3' = these re-referenced forms, 'decline F1-F3' = the plan's
# literal forms density_plan, distance_plan, two_head_plan with the same range hinge and classifier ratio); its
# marks.json records the choice as mark_families, and select_standin follows it. MAIN_STANDINS is the fallback for a
# marks.json without that field.
MAIN_STANDINS = ("range_hinge", "density", "classifier_ratio", "distance", "two_head")


def group_of(rep):
    return "exploratory" if int(rep) in C.EXPLORATORY_SEEDS else "confirmatory"


def final_prereg(paths=None):
    prereg = C.load_prereg(paths)
    if not prereg.payload.get("approvable"):
        raise SystemExit(f"refused: prereg.json is a draft (missing pins {prereg.payload.get('missing_pins')})")
    return prereg


def bca_gate(paths=None, pins_fn=None):
    """The gate of every BCA phase: approval of the current pre-registration, pinned code, best_standin.json, and the
    other jobs' pinned components unchanged since prereg.json was written (pins_fn: a stand-in in the tests)."""
    prereg = final_prereg(paths)
    approval = C.require_approval(paths)
    C.read_json((paths or C.PATHS).out / "best_standin.json")
    import prereg_payload as PP
    PP.check_pins(prereg, pins_fn)
    return approval


def detect_module(prereg, which="detect"):
    """The locked detection script from its pinned copy (its whole code manifest verified first). which='labels'
    loads only its label script, for a process that must hold no signal code (the script refuses otherwise)."""
    import detect_contract as DC
    C.verify_component(C.DETECT_OUT, prereg.payload["pins"]["detect_manifest_sha256"])
    pin = prereg.payload["pins"]["detect_module"]
    if which == "labels":
        path = Path(pin["path"]).with_name("labels.py")
        return C.load_pinned_module("stage2_detect_labels", path, C.file_sha256(path))   # verified by the manifest
    mod = C.load_pinned_module("stage2_detect", pin["path"], pin["sha256"])
    DC.check_detect_module(mod)
    return mod


def pairs_root(rep, paths=None, rehearsal_root=None):
    """Where a seed's locked pair sets, labels and stand-in scores live: the rehearsal's own folder for gate seeds 0-4
    (runs/shift_explore/detect, made by the rehearsal before any BCA run; Stage 2 only reads there), Stage 2's folder
    for fresh seeds 5-9. rehearsal_root: a stand-in for the rehearsal's folder in the synthetic tests."""
    if int(rep) in C.EXPLORATORY_SEEDS:
        return Path(rehearsal_root) if rehearsal_root is not None else C.DETECT_OUT
    return (paths or C.PATHS).out / "detect_fresh"


PAIRS_RECORD = "pairs.json"        # the detection script's pair-set record (its pairs.RECORD)


def locked_pair_sets(D, ctx, world, rep, paths=None, rehearsal_root=None):
    """The locked pair sets of (seed, world) through the detection script's pair_sets. For gate seeds 0-4 they must
    already exist in the rehearsal's folder: pair_sets then only loads and verifies them (it builds and saves only
    when the record is absent, and Stage 2 must never write there), so this refuses when the record is absent. For
    fresh seeds 5-9 they are built once in Stage 2's folder."""
    root = pairs_root(rep, paths, rehearsal_root)
    if int(rep) in C.EXPLORATORY_SEEDS and not (seed_world_dir(rep, world, root) / PAIRS_RECORD).exists():
        raise SystemExit(f"refused: the rehearsal's pair sets of gate seed {rep} ({world}) do not exist under {root}; "
                         f"Stage 2 reads them and never builds them there (run the rehearsal's pairs phase first)")
    return D.pair_sets(ctx, world, out=root)


def seed_world_dir(rep, world, root):
    return Path(root) / f"seed{int(rep)}" / world


def detect_root(paths=None):
    return (paths or C.PATHS).out / "detect"


def decision(version, paths=None):
    """The detection decision of the exploratory group for the uniform version (written by detect_marks)."""
    return C.read_json(detect_root(paths) / f"marks_exploratory_{version}.json", required=False)


def chosen_version(paths=None):
    """The version the policy test, the full host and the confirmation use: 'uniform' if it passed direction and size
    on seeds 0-4, else 'weighted' if that did, else None (no BCA training arm)."""
    u = decision("uniform", paths)
    if u is None:
        raise SystemExit("refused: the exploratory detection marks (uniform) have not been computed")
    if not u["marks"]["direction_or_size_failed"]:
        return "uniform"
    w = decision("weighted", paths)
    if w is None:
        raise SystemExit("refused: the uniform version failed direction or size; run the weighted version's detection "
                         "test first")
    return None if w["marks"]["direction_or_size_failed"] else "weighted"


def room_condition():
    """Plan 4.8: a BCA training arm runs only if the room test passed its room, recovery and keeps-improvement marks.
    Read from the room's pooled look (sidecar-verified, its pinned spec)."""
    p = C.ROOM_OUT / "pooled.json"
    rec = C.read_json(p, required=False)
    if rec is None:
        return dict(passed=False, reason="the room test has no pooled look (stopped at futility or not finished)")
    if rec.get("kind") != "room-pooled" or rec.get("spec_sha256") != C.ROOM_PINNED["spec_sha256"]:
        raise SystemExit("refused: the room's pooled.json is not the pinned room test's pooled look")
    ok = bool(rec["room"]["passed"] and rec["recovery"]["passed"] and rec["keeps_improvement"]["passed"])
    return dict(passed=ok, pooled_sha256=C.file_sha256(p), room=rec["room"]["passed"],
                recovery=rec["recovery"]["passed"], keeps_improvement=rec["keeps_improvement"]["passed"])


def training_gate(paths=None):
    v = chosen_version(paths)
    if v is None:
        raise SystemExit("refused: neither version passed detection direction and size; no BCA training arm runs")
    rc = room_condition()
    if not rc["passed"]:
        raise SystemExit(f"refused: the room test did not pass its room, recovery and keeps-improvement marks ({rc})")
    return v, rc


# --------------------------------------------------------------------------------------------- no-BCA phases

def phase_fresh_data(rep, paths=None):
    prereg = final_prereg(paths)
    import fresh as F
    for world in ("stall", "no_stall"):
        F.make_fresh_data(rep, world, prereg)


def phase_fresh_learners(rep, paths=None):
    prereg = final_prereg(paths)
    import fresh as F
    F.make_fresh_learners(rep, prereg, paths=paths)


def select_best_standin(marks, families=MAIN_STANDINS):
    """Plan 4.2's rule, fixed before any BCA run: among the five stand-ins that carry the stand-ins test's mark
    (`families`: its marks.json's mark_families, as the user chose them there), those that capture the room; the best
    is the one with the largest pooled matched gain over no caution in strong current; if none captures the room, the
    largest pooled matched gain among all five; ties go to the earlier name in `families`. None when no stand-in has a
    pooled gain."""
    table = {f: dict(captures_the_room=bool(marks[f]["captures_the_room"]),
                     gain=marks[f]["gain_strong_current"]["mean"]) for f in families if f in marks}
    capture = [f for f in families if f in table and table[f]["captures_the_room"] and table[f]["gain"] is not None]
    pool = capture or [f for f in families if f in table and table[f]["gain"] is not None]
    best = None
    for f in pool:
        if best is None or table[f]["gain"] > table[best]["gain"]:
            best = f
    return best, table, bool(capture)


def phase_select_standin(paths=None):
    prereg = final_prereg(paths)
    paths = paths or C.PATHS
    out = paths.out / "best_standin.json"
    if out.exists():
        raise SystemExit(f"refused: {out} exists")
    if paths.approval.exists() and any((paths.out / d).exists() for d in ("detect", "policy", "host", "deploy")):
        raise SystemExit("refused: a BCA output exists already; the rival must be chosen before any BCA run")
    pin = prereg.payload["pins"]["standins_marks_spec_sha256"]
    rec = C.read_json(C.STANDINS_OUT / "marks.json", required=False)
    if rec is None:
        why = standins_never_run()
        if why is None:
            raise SystemExit("refused: the stand-ins test has not written marks.json yet and the room test does not "
                             "rule it out; the rival is chosen only from finished Stage 1 results")
        best, table, captured, reason = None, {}, False, why
    else:
        if rec.get("kind") != "standins-marks" or rec.get("spec_sha256") != pin:
            raise SystemExit("refused: the stand-ins' marks.json is not the pinned stand-ins test's")
        families = tuple(rec.get("mark_families") or MAIN_STANDINS)
        if len(families) != 5 or any(f not in rec["marks"] for f in families):
            raise SystemExit(f"refused: the stand-ins' marks.json does not hold marks for five mark families "
                             f"({families})")
        best, table, captured = select_best_standin(rec["marks"], families)
        reason = None
    C.write_json(out, dict(label=C.LABEL_LONG, kind="stage2-best-standin", best=best, captured_any=captured,
                           table=table, reason=reason, prereg_sha256=prereg.sha256,
                           families=None if rec is None else list(families),
                           marks_sha256=None if rec is None else C.file_sha256(C.STANDINS_OUT / "marks.json"),
                           rule=select_best_standin.__doc__, written=time.strftime("%Y-%m-%dT%H:%M:%S%z")), paths)
    C.log(f"best stand-in: {best} ({'captures the room' if captured else 'none captures the room'})")


def standins_never_run():
    """The reason the stand-ins test will never run (plan 4.2: only after the room test passes its room, recovery and
    keeps-improvement marks), or None while it still may: the room's futility look said stop, or its pooled look
    exists and did not pass."""
    fut = C.read_json(C.ROOM_OUT / "futility.json", required=False)
    if fut is not None and fut.get("spec_sha256") == C.ROOM_PINNED["spec_sha256"] and fut.get("stop") is True:
        return "the room test stopped at its futility look; the stand-ins test never runs"
    rc = room_condition()
    if "pooled_sha256" in rc and not rc["passed"]:
        return "the room test's pooled look did not pass; the stand-ins test never runs"
    return None


def phase_detect_pairs(rep, world, paths=None):
    """The locked pair sets of a seed and world (no signal, no label): for seeds 0-4 the rehearsal's, which must exist
    (loaded and verified read-only; refused when absent), built and saved in Stage 2's folder for seeds 5-9."""
    prereg = final_prereg(paths)
    import fresh as F
    D = detect_module(prereg)
    if int(rep) in C.EXPLORATORY_SEEDS and not (seed_world_dir(rep, world, pairs_root(rep, paths)) /
                                                  PAIRS_RECORD).exists():
        locked_pair_sets(D, None, world, rep, paths)      # refuses before any data is loaded
    ctx = F.context(rep, worlds=("stall", "no_stall"))
    locked_pair_sets(D, ctx, world, rep, paths)


def phase_detect_labels(rep, world, paths=None):
    """The labels, by the detection script's label code in a process that holds no signal code (its own rule): for
    seeds 0-4 the rehearsal's labels must exist and verify; for seeds 5-9 they are computed once in Stage 2's folder.
    No BCA."""
    prereg = final_prereg(paths)
    import fresh as F
    root = pairs_root(rep, paths)
    LB = detect_module(prereg, "labels")
    if (seed_world_dir(rep, world, root) / "labels.json").exists():
        LB.load(rep, world, root)
        C.log(f"labels of seed {rep} {world} exist and verify ({root})")
        return
    if int(rep) in C.EXPLORATORY_SEEDS:
        raise SystemExit("refused: the rehearsal's labels of a gate seed are missing; run the detection rehearsal's "
                         "labels phase (Stage 1) first")
    ctx = F.context(rep, worlds=("stall", "no_stall"))
    LB.run(ctx, world, out=root, extra=dict(stage2_prereg_sha256=prereg.sha256))


def phase_detect_standins(rep, world, paths=None):
    """The five detection stand-ins (detect_contract.STANDIN_NAMES) scored by the detection script on the same pairs
    (no BCA): the rehearsal's for seeds 0-4 (verified present, read-only; nothing is fitted or written), computed in
    Stage 2's folder for seeds 5-9."""
    prereg = final_prereg(paths)
    import fresh as F
    import detect_contract as DC
    D = detect_module(prereg)
    root = pairs_root(rep, paths)
    todo = []
    for name in DC.STANDIN_NAMES:
        rec = seed_world_dir(rep, world, root) / "scores" / f"{name}.json"
        if rec.exists():
            C.read_json(rec)
            continue
        if int(rep) in C.EXPLORATORY_SEEDS:
            raise SystemExit(f"refused: the rehearsal's scores of {name} for gate seed {rep} are missing")
        todo.append(name)
    if not todo:
        return
    ctx = F.context(rep, worlds=("stall", "no_stall"))
    fns = D.stand_in_signals(ctx, world, names=tuple(todo))
    for name in todo:
        D.score_signal(name, fns[name], ctx, world, None, pairs_out=root)


# --------------------------------------------------------------------------------------------- BCA phases

def band_dir(rep, world, critic, version, paths=None):
    return detect_root(paths) / version / world / critic / f"seed{int(rep)}"


def phase_detect_fit(rep, world, critic, paths=None):
    approval = bca_gate(paths)
    import band as B
    import deploy_check as DCK
    import fresh as F
    prereg = C.load_prereg(paths)
    D = detect_module(prereg)
    ctx = F.context(rep, worlds=("stall", "no_stall"))
    t4 = t4_log_rows(D, ctx, world, rep, paths)
    d = band_dir(rep, world, critic, "uniform", paths)
    band, rec = B.fit_on_context(ctx, world, critic, approval, d)
    if band is not None:
        pg = ctx.prepared(world)
        reader = B.WidthReader(band, rec, pg.obs_mean, pg.obs_std, approval, B.TC_DX())
        DCK.fit_tau(ctx, world, reader, approval, d, exclude_log_rows=t4)


def t4_log_rows(D, ctx, world, rep, paths=None):
    """The log rows of the detection script's T4 set, left out of tau's rows so that the calibration-sanity mark
    (share of T4 above tau) is read on rows tau never saw."""
    import numpy as np
    ps = locked_pair_sets(D, ctx, world, rep, paths)
    return np.asarray(ps["log_row"])[np.asarray(ps["set"]) == "T4"]


def phase_detect_fit_weighted(rep, world, critic, paths=None):
    approval = bca_gate(paths)
    group = group_of(rep)
    if group == "exploratory":
        u = decision("uniform", paths)
        if u is None or not u["marks"]["direction_or_size_failed"]:
            raise SystemExit("refused: the weighted version is declared only when the uniform version fails direction "
                             "or size on seeds 0-4")
    elif chosen_version(paths) != "weighted":
        raise SystemExit("refused: the frozen version for seeds 5-9 is not the weighted one")
    import band as B
    import deploy_check as DCK
    import fresh as F
    D = detect_module(C.load_prereg(paths))
    ctx = F.context(rep, worlds=("stall", "no_stall"))
    t4 = t4_log_rows(D, ctx, world, rep, paths)
    d = band_dir(rep, world, critic, "weighted", paths)
    band, rec = B.fit_weighted_on_context(ctx, world, critic, approval, band_dir(rep, world, critic, "uniform", paths),
                                          d)
    pg = ctx.prepared(world)
    reader = B.WidthReader(band, rec, pg.obs_mean, pg.obs_std, approval, B.TC_DX())
    DCK.fit_tau(ctx, world, reader, approval, d, exclude_log_rows=t4)


def phase_detect_score(rep, world, critic, version, paths=None):
    approval = bca_gate(paths)
    prereg = C.load_prereg(paths)
    import band as B
    import fresh as F
    D = detect_module(prereg)
    ctx = F.context(rep, worlds=("stall", "no_stall"))
    d = band_dir(rep, world, critic, version, paths)
    if (d / "scores_record.json").exists():
        raise SystemExit(f"refused: {d}/scores_record.json exists")
    band, rec = B.load_band(d, approval, B.TC_DX(), B.TC_DA())
    if band is None:
        raise SystemExit("refused: this band's fit failed")
    pg = ctx.prepared(world)
    reader = B.WidthReader(band, rec, pg.obs_mean, pg.obs_std, approval, B.TC_DX())
    out = C.guard_out(d / "scores", paths)
    srec = D.score_bca_width(reader, ctx, world, out, stage2_prereg_sha256=approval.prereg_sha256, version=version,
                             pairs_out=pairs_root(rep, paths))
    C.write_json(d / "scores_record.json", dict(label=C.LABEL_LONG, kind="stage2-width-scores", rep=rep, world=world,
                                                critic=critic, version=version, detect=srec,
                                                scores_path=str(out / f"{reader.name}.json"),
                                                band_file_sha256=reader.sha256, prereg_sha256=approval.prereg_sha256),
                 paths)


def recorded_seed_failure(rep, world, critic, version, paths=None):
    """The reason a seed counts as FAILED on record for (world, critic, version), or None: a fresh seed whose data or
    learners failed (fresh.recorded_failure), or a band fit that failed (band.json with failed=True, of this version or,
    for the weighted version, of the uniform fit it is built on). A failed seed fails every mark."""
    import fresh as F
    why = F.recorded_failure(rep, paths)
    if why is not None:
        return why
    for v in (version, "uniform") if version == "weighted" else (version,):
        b = C.read_json(band_dir(rep, world, critic, v, paths) / "band.json", required=False)
        if b is not None and b.get("failed"):
            return f"the {v} band fit of seed {rep} ({world}, {critic} critic) failed: {b.get('reason')}"
    return None


def detection_inputs(rep, world, critic, version, paths=None, rehearsal_root=None):
    """('failed', reason) | ('complete', (scores_record.json, tau.json, labels.json)) | ('incomplete', [missing])
    for one seed, world and critic of one version."""
    why = recorded_seed_failure(rep, world, critic, version, paths)
    if why is not None:
        return "failed", why
    d = band_dir(rep, world, critic, version, paths)
    lp = seed_world_dir(rep, world, pairs_root(rep, paths, rehearsal_root)) / "labels.json"
    need = (d / "scores_record.json", d / "tau.json", lp)
    missing = [str(x) for x in need if not x.exists()]
    return ("incomplete", missing) if missing else ("complete", need)


# What the detection marks read, per seed: (world, critic, version); None = the version under test. The first gates
# the marks; the host critic is reported "as written" (uniform weights, BCA as published) whatever the version.
DETECTION_READS = (("stall", "fixed", None), ("stall", "host", "uniform"), ("no_stall", "fixed", None))


def detection_completeness(group, version, paths=None, rehearsal_root=None):
    """Refuse (SystemExit) unless every seed of the group is complete or failed on record for every critic the marks
    read, and has the five stand-ins' score records (stalling world) when it is not failed. Returns
    {(rep, world, critic): (status, info)}."""
    import detect_contract as DC
    out, missing = {}, []
    for rep in GROUPS[group]:
        for world, critic, v in DETECTION_READS:
            st, info = detection_inputs(rep, world, critic, v or version, paths, rehearsal_root)
            out[(rep, world, critic)] = (st, info)
            if st == "incomplete":
                missing.append(f"seed {rep} {world} {critic}: {info}")
        if out[(rep, "stall", "fixed")][0] == "complete":
            root = pairs_root(rep, paths, rehearsal_root)
            for n in DC.STANDIN_NAMES:
                sp = seed_world_dir(rep, "stall", root) / "scores" / f"{n}.json"
                if not sp.exists():
                    missing.append(f"seed {rep}: stand-in scores {sp}")
    if missing:
        raise SystemExit("refused: the detection marks are written once and their inputs are incomplete (run "
                         "detect_pairs, detect_labels, detect_standins, detect_fit and detect_score first): "
                         + "; ".join(missing))
    return out


def phase_detect_marks(group, version, paths=None):
    """Join widths and labels (after both were hashed), compute the measures with the locked script, the marks and
    the claim rule; for the exploratory uniform version also the version decision. Refuses while any seed's inputs
    are incomplete (detection_completeness); a failed seed fails every mark."""
    approval = bca_gate(paths)
    prereg = C.load_prereg(paths)
    import detect_contract as DC
    D = detect_module(prereg)
    seeds = GROUPS[group]
    if group == "confirmatory":
        v = chosen_version(paths)
        if v is not None and version != v:
            raise SystemExit(f"refused: seeds 5-9 use the frozen version {v}")
    out = detect_root(paths) / f"marks_{group}_{version}.json"
    if out.exists():
        raise SystemExit(f"refused: {out} exists")
    status = detection_completeness(group, version, paths)

    def measures(rep, world, critic):
        st, info = status[(rep, world, critic)]
        if st != "complete":
            return None
        s_path, tau_path, lp = info
        s = C.read_json(s_path)
        tau = C.read_json(tau_path)
        return DC.check_measures(D.measures(s["scores_path"], lp, level=tau["tau"]))

    def standin_measures(rep):
        st, info = status[(rep, "stall", "fixed")]
        if st != "complete":
            return {n: None for n in DC.STANDIN_NAMES}
        lp = info[2]
        root = pairs_root(rep, paths)
        return {n: DC.check_measures(D.measures(seed_world_dir(rep, "stall", root) / "scores" / f"{n}.json", lp))
                for n in DC.STANDIN_NAMES}

    bca = [measures(rep, "stall", "fixed") for rep in seeds]
    stand = {n: [] for n in DC.STANDIN_NAMES}
    for rep in seeds:
        for n, m in standin_measures(rep).items():
            stand[n].append(m)
    marks = DC.detection_marks(bca, stand, n_seeds=len(seeds), group=group)
    no_stall = [measures(rep, "no_stall", "fixed") for rep in seeds]
    host = [measures(rep, "stall", "host") for rep in seeds]
    C.write_json(out, dict(label=C.LABEL_LONG, kind="stage2-detection-marks", group=group, version=version,
                           seeds=list(seeds), prereg_sha256=approval.prereg_sha256, marks=marks,
                           seed_status={f"seed{r}/{w}/{c}": dict(status=st, reason=None if st == "complete" else info)
                                        for (r, w, c), (st, info) in status.items()},
                           per_seed_fixed_critic=bca, standins=stand,
                           selectivity_reported=DC.selectivity(bca, no_stall),
                           host_critic_reported=host, no_stall_reported=no_stall,
                           next=("weighted version declared: run detect_fit_weighted and repeat" if
                                 (group == "exploratory" and version == "uniform" and
                                  marks.get("direction_or_size_failed")) else None)), paths)
    C.log(f"detection {group} {version}: claim rule met {marks.get('claim_rule_met')}, direction/size failed "
          f"{marks.get('direction_or_size_failed')}")


def phase_policy(rep, which, paths=None):
    approval = bca_gate(paths)
    import policy as PO
    if which == "train":
        version, rc = training_gate(paths)
        best = C.read_json((paths or C.PATHS).out / "best_standin.json")["best"]
        PO.train(rep, approval, version, rc, best, paths=paths)
    elif which == "eval":
        PO.evaluate(rep, approval, "stall", paths=paths)
    elif which == "no_stall_train":
        version, _ = training_gate(paths)
        PO.train_no_stall(rep, approval, version, paths=paths)
    elif which == "no_stall_eval":
        PO.evaluate(rep, approval, "no_stall", paths=paths)


def deploy_version(paths=None):
    """The band version the deployment check reads: the chosen version; the uniform band only if neither version
    passed direction and size. Refuses (chosen_version) while the decision is not made: the deployment values are
    written once, so they must not be read on a band that is not the chosen version."""
    return chosen_version(paths) or "uniform"


def phase_deploy(rep, world, paths=None):
    """Variant B on the no-caution actor (always), the bca_k1 actor (when the policy test locked one) and plain
    TD3+BC (stalling world, its as-written host band)."""
    approval = bca_gate(paths)
    version = deploy_version(paths)
    import band as B
    import deploy_check as DCK
    import fresh as F
    import policy as PO
    ctx = F.context(rep, worlds=("stall", "no_stall"))
    pg = ctx.prepared(world)
    actors, readers = {}, {}
    d = band_dir(rep, world, "fixed", version, paths)
    band, rec = B.load_band(d, approval, B.TC_DX(), B.TC_DA())
    if band is None:
        raise SystemExit("refused: the fixed-critic band of this seed and world failed")
    rd = B.WidthReader(band, rec, pg.obs_mean, pg.obs_std, approval, B.TC_DX())
    tau = C.read_json(d / "tau.json")["tau"]
    actors["none"], readers["none"] = ctx.none(world), (rd, tau)
    lockdir = PO.seed_dir(rep, None if world == "stall" else "no_stall", paths)
    if (lockdir / "lock.json").exists():
        lock, acts, _ = PO.verify_lock(lockdir, approval)
        actors["bca_k1"], readers["bca_k1"] = acts["bca_k1"], (rd, tau)
    if world == "stall":
        dh = band_dir(rep, "stall", "host", "uniform", paths)      # plain TD3+BC: the as-written (uniform) band
        if (dh / "band.json").exists():
            bh, rh = B.load_band(dh, approval, B.TC_DX(), B.TC_DA())
            if bh is not None:
                actors["td3bc"] = ctx.td3bc()
                readers["td3bc"] = (B.WidthReader(bh, rh, pg.obs_mean, pg.obs_std, approval, B.TC_DX()),
                                    C.read_json(dh / "tau.json")["tau"])
    DCK.run(ctx, world, actors, readers, approval, (paths or C.PATHS).out / "deploy" / world / f"seed{rep}",
            paths=paths)


def phase_host(rep, world, which, paths=None):
    approval = bca_gate(paths)
    import host as HO
    if which == "train":
        version, _ = training_gate(paths)
        locked_c = None
        if world == "no_stall":
            lock = C.read_json(HO.host_dir(rep, "stall", paths) / "lock.json")
            m = lock.get("hinge_match") or {}
            locked_c = None if m.get("knob") is None else float(m["knob"])
        HO.train(rep, world, approval, version, locked_c, paths=paths)
    else:
        HO.evaluate(rep, world, approval, paths=paths)


def claim_fields(group, detection_rule_met, policy_passed):
    """The pooled look's claim fields. Only the confirmatory group (fresh seeds 5-9) can carry the flags claim; for
    the exploratory group (seeds 0-4, used to design the test) flags_claim is None and the text cannot be mistaken
    for a claim."""
    met = bool(detection_rule_met and policy_passed)
    if group == "confirmatory":
        return dict(detection_claim=bool(detection_rule_met), flags_claim=met,
                    flags_claim_text=("BCA flags when the actor relies on commands the data do not support, and acting "
                                      "on that flag repairs the harm while keeping TD3+BC's gains" if met else
                                      "the flags claim is not supported on fresh seeds 5-9"),
                    marks_met=met)
    return dict(detection_claim=None, flags_claim=None,
                flags_claim_text=("exploratory, not a claim: the detection and policy marks were met on seeds 0-4, which "
                                  "were used to design this test" if met else
                                  "exploratory, not a claim: the detection and policy marks were not all met on seeds "
                                  "0-4"),
                marks_met=met)


def phase_pooled(group, paths=None):
    """The group's verdict: the detection marks and claim rule, the policy marks (M1-M5) and the claim sentence; the
    full host reported. Only seeds 5-9 carry the confirmatory claim. Written once, so it refuses until the group's
    detection marks exist and every seed has deployment values or a recorded failure."""
    approval = bca_gate(paths)
    import policy as PO
    import deploy_check as DCK
    paths = paths or C.PATHS
    out = paths.out / f"pooled_{group}.json"
    if out.exists():
        raise SystemExit(f"refused: {out} exists")
    seeds = GROUPS[group]
    version = chosen_version(paths)
    det = C.read_json(detect_root(paths) / f"marks_{group}_{version or 'uniform'}.json", required=False)
    if det is None:
        raise SystemExit(f"refused: the {group} detection marks ({version or 'uniform'}) have not been computed")
    dep, missing = {}, []
    for rep in seeds:
        r = C.read_json(paths.out / "deploy" / "stall" / f"seed{rep}" / "values.json", required=False)
        if r is None:
            why = recorded_seed_failure(rep, "stall", "fixed", version or "uniform", paths)
            if why is None:
                missing.append(rep)
            dep[rep] = None
        else:
            dep[rep] = DCK.summary_for_marks(r)
    if missing:
        raise SystemExit(f"refused: the deployment check of seeds {missing} (stalling world) has not been run")
    pol = None
    if version is not None and room_condition()["passed"]:
        pol = PO.policy_marks(seeds, approval, dep, paths)
    host = {}
    for rep in seeds:
        for world in ("stall", "no_stall"):
            host[f"{world}/seed{rep}"] = C.read_json(paths.out / "host" / world / f"seed{rep}" / "values.json",
                                                    required=False)
    rule_met = bool(det["marks"].get("claim_rule_met"))
    fields = claim_fields(group, rule_met, pol is not None and pol["passed"])
    C.write_json(out, dict(
        label=C.LABEL_LONG, kind="stage2-pooled", group=group, seeds=list(seeds), version=version,
        status=("CONFIRMATORY (fresh seeds 5-9, every setting frozen)" if group == "confirmatory" else
                "EXPLORATORY (seeds 0-4: post hoc with respect to the room test's futility look on seeds 0 and 1); "
                "nothing here supports a claim"),
        prereg_sha256=approval.prereg_sha256, detection=det, detection_claim_rule_met=rule_met, policy=pol,
        deployment=dep, host_reported={k: (None if v is None else v.get("points")) for k, v in host.items()},
        claim_sentence=None if pol is None else pol["claim_sentence"], **fields), paths)


# --------------------------------------------------------------------------------------------- command line

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=(
        "fresh_data", "fresh_learners", "select_standin", "detect_pairs", "detect_labels", "detect_standins",
        "detect_fit",
        "detect_fit_weighted", "detect_score", "detect_marks", "policy_train", "policy_eval", "policy_no_stall_train",
        "policy_no_stall_eval", "deploy", "host_train", "host_eval", "pooled"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--world", choices=("stall", "no_stall"), default="stall")
    ap.add_argument("--critic", choices=("fixed", "host"), default="fixed")
    ap.add_argument("--version", choices=("uniform", "weighted"), default="uniform")
    ap.add_argument("--group", choices=tuple(GROUPS))
    a = ap.parse_args(argv)
    p = a.phase
    if p in ("select_standin",):
        return phase_select_standin()
    if p in ("detect_marks", "pooled"):
        if a.group is None:
            raise SystemExit("--group is required")
        return phase_detect_marks(a.group, a.version) if p == "detect_marks" else phase_pooled(a.group)
    if a.seed is None or a.seed not in C.EXPLORATORY_SEEDS + C.CONFIRMATORY_SEEDS:
        raise SystemExit("--seed must be one of 0-9")
    if p == "fresh_data":
        return phase_fresh_data(a.seed)
    if p == "fresh_learners":
        return phase_fresh_learners(a.seed)
    if p == "detect_pairs":
        return phase_detect_pairs(a.seed, a.world)
    if p == "detect_labels":
        return phase_detect_labels(a.seed, a.world)
    if p == "detect_standins":
        return phase_detect_standins(a.seed, a.world)
    if p == "detect_fit":
        return phase_detect_fit(a.seed, a.world, a.critic)
    if p == "detect_fit_weighted":
        return phase_detect_fit_weighted(a.seed, a.world, a.critic)
    if p == "detect_score":
        return phase_detect_score(a.seed, a.world, a.critic, a.version)
    if p.startswith("policy_"):
        return phase_policy(a.seed, p[len("policy_"):])
    if p == "deploy":
        return phase_deploy(a.seed, a.world)
    if p.startswith("host_"):
        return phase_host(a.seed, a.world, p[len("host_"):])
    raise SystemExit(f"unknown phase {p}")


if __name__ == "__main__":
    main()
