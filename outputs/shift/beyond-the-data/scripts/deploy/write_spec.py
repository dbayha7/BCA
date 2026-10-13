"""EXPLORATORY: write runs/shift_explore/deploy/spec.json (+ .sha256) before any real deploy run.

The spec may be rewritten only while no job output, stop file or summary exists (no real run has happened). A rewrite
must raise the version and name the SHA-256 of the spec it replaces, and must state its changes (CHANGES, written to
changes_before_any_run); the old spec goes to spec_history/ and the outputs computed under it (smoke checks and their
logs, code manifest and copies) to superseded/v<k>/.

Usage: python write_spec.py
"""

import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import signals as S  # noqa: E402
import run_job as RJ  # noqa: E402

H = S.H
VERSION = 2
PREVIOUS_SPEC_SHA256 = "a9ff3a26f0fbbd65c5d4774f1489e8c38062adaea2b39eeff63a53c34b7fd566"   # spec v1
NO_RETURN_SEEN = ("Made before any full deploy run. No deploy point value had been seen: the v1 seed-0 smoke checks "
                  "withheld every point value. No room-test outcome file (values, futility or pooled look) was read. "
                  "Nothing was chosen on returns.")
CHANGES = [
    dict(id="C1 Stage-2 bar read in strong current only",
         finding="review of the deploy component, finding 1 (minor)",
         change=("The reported Stage-2 bar (plan 5.3's deployment-check bar: at least 10 points over random switching at "
                 "the same rate, cost at most 2 points in recorded conditions) now reads the gain over random in strong "
                 "current only (summarise.STAGE2_BAR situations = strong current). Before, it passed on a gain in strong "
                 "current or in big kicks, the situations of the channel-open bar. The channel-open bar is unchanged "
                 "(strong current or big kicks; plan 4.3). tables.md now shows both bars for every trigger."),
         reason=("Plan 5.3 reads its marks in strong current unless it says otherwise, and its deployment-check bar names "
                 "no other situation. Before the change a trigger could be reported as clearing the Stage-2 bar on a "
                 "big-kicks gain alone: in the reviewer's synthetic records (12 points over random in big kicks, 5 in "
                 "strong current, cost 1) the bar read true; it now reads false."),
         evidence="scratchpad/review_deploy/stage2_bar_check.py (reviewer; synthetic, writes nothing)",
         returns_seen=NO_RETURN_SEEN),
    dict(id="C2 the override also covers a stop at the room's futility look",
         finding="review of the deploy component, finding 2 (minor)",
         change=("When the room's pooled.json does not exist, a full job now reads the room's futility.json "
                 "(sidecar-verified, labelled EXPLORATORY, of kind room-futility, naming the pinned room spec), and only "
                 "its stop flag. Stop true: the room failed at its futility look, after which its code never writes "
                 "pooled.json; the job runs only with --after-room-failure and records that. Stop false: the pooled look "
                 "is still to come, and the job refuses. No futility.json: the job refuses. With pooled.json present "
                 "nothing changes. Deviation D7 and the launch condition say so."),
         reason=("D7 justifies the override with plan 4.8: the deployment check remains when the room fails. A futility "
                 "stop is one of the main ways the room can fail (plan 4.1, then 4.5), and the room's code writes no "
                 "pooled.json after it (run_seed.phase_pooled refuses unless futility.json says continue), so before "
                 "this change the override could not be used in that case. The override's own reason covers both ways "
                 "of failing. Whether the override is used at all remains the user's decision (D7)."),
         evidence="runs/shift_explore/room/code/run_seed.py phase_pooled (pinned copy, same SHA-256 as the room manifest)",
         returns_seen=NO_RETURN_SEEN),
    dict(id="C3 the launch condition is carried into the pooled outputs",
         finding="review of the deploy component, finding 3 (minor)",
         change=("Every values file and, now, every stop file records the launch condition its job ran under. The "
                 "summary collects them, refuses unless all 25 jobs share the same one (which room look was read, that "
                 "file's SHA-256, its pass flags or stop flag, whether the override was used), and copies it into "
                 "summary.json (launch_condition, launch_condition_text) and the first lines of tables.md, in plain "
                 "words."),
         reason=("Before, a reader of the pooled tables could not tell that the mark had been read after a failed room "
                 "test, and jobs launched under different room outcomes could have been pooled together."),
         returns_seen=NO_RETURN_SEEN),
    dict(id="C4 shape readout for every trigger (reported, never a gate)",
         finding="review of the deploy component, finding 4 (minor)",
         change=("New readout (signals.Signals.shape_readout; written to each job's signals.shape, pooled over seeds in "
                 "summary.json shape and in tables.md). At the room's 2,000 full-throttle-region training states of the "
                 "deployment world (the room harness's shape_rows: the states of the room's check (e)), the deployed "
                 "actor's command with thruster 1 set along its own push to each throttle of the room's check-(e) grid "
                 "(70% to full throttle; 80% is the float32 knee, which is not past it). For every trigger: the share of "
                 "states whose score falls, rises or stays from 80% to 81%, the mean score and the firing rate at each "
                 "throttle, and the firing rate at the actor's own command. See shape_readout."),
         reason=("The room's check (e) gates penalties, because the actor climbs a penalty during training, and a "
                 "penalty that falls past 80% rewards the first steps of over-throttle. A trigger is never climbed: it "
                 "only decides at each step whether the fallback takes over, so its shape decides nothing here and no "
                 "trigger is gated on it. Without the readout, though, a reader cannot tell whether a trigger's catches "
                 "in strong current come from the command (its firing rate rises past 80%) or only from the unusual "
                 "state (its rate is flat across throttles). The reviewer's read-only check on seed 0 (stalling world, "
                 "gapped critic's actor) found, for example, that the re-referenced one-step oracle falls from 80% to "
                 "81% on 5.4% of the states and fires on 6% at 80% and 72% at full throttle, while two-head "
                 "disagreement falls on 43.9% and fires on 2-3% at every throttle. A rerun of that check reproduced "
                 "every number."),
         evidence=("scratchpad/review_deploy/shape_check.py (reviewer; no rollout, no points); its rerun log "
                   "scratchpad/deploy_fix/verify_shape.log"),
         returns_seen=NO_RETURN_SEEN),
    dict(id="C5 manifest history and smoke logs after a spec rewrite (housekeeping)",
         finding="found while applying C1-C4",
         change=("write_manifest.py now links a manifest that a spec rewrite moved to superseded/v<k>/ into the new "
                 "manifest's history and gives the new manifest the next version number (before, it would have "
                 "restarted at version 1 with no history). write_spec.py now also moves the old spec's smoke logs "
                 "(logs/smoke_*.log) to superseded/v<k>/logs/."),
         reason="so that the record of what ran under which spec stays complete; no number is affected",
         returns_seen=NO_RETURN_SEEN),
]
CHANGE_CHECK = ("The v2 seed-0 smoke must give, for every rule of every job, the same SHA-256 of the per-start points "
                "as the v1 smoke (superseded/v1/smoke/): the changes alter no rule, level, trigger or fallback. Only "
                "hashes are compared; no point value is read.")


def any_run_output():
    found = sorted(C.OUT.glob("seed*.json")) + sorted(C.OUT.glob("stop_*.json")) + sorted(C.OUT.glob("partial/*.json"))
    found += [C.OUT / n for n in ("summary.json", "tables.md") if (C.OUT / n).exists()]
    return found


def payload():
    room = C.read_json(C.ROOM_SPEC)
    pins = C.pins()
    trig = {}
    for t in S.TRIGGERS:
        trig[t] = dict(role="primary (plan 4.3)" if t in S.PRIMARY_TRIGGERS else "reported", text=S.TRIGGER_TEXT[t],
                       fallbacks=["clip99"] if t == "sa95" else list(RJ.FALLBACKS))
    trig["unrec"]["formula"] = "fire iff max_k |a_k| > 0.800000011920929 (thruster_chain.past_knee; the knee in both worlds)"
    trig["onestep"]["formula"] = ("score = max(delta*(s, r(s)) - delta*(s, a), 0); fire iff score > q95. "
                                  "delta*(s, a) = r(s, a) + 0.98 * min_h Q_h(s', c(s')) - min_h Q_h(s, a) "
                                  "(room harness oracle64: one float64 step of thruster_chain.dynamics_jax, the frozen "
                                  "float32 critic at float32-rounded normalised states, c = recorder.stall_aware of the "
                                  "world's physics); r(s) = recorder.Recorder(spec, physics)(s): the recording "
                                  "controller's normal-driving command without wobble or the full-throttle command")
    trig["onestep_clone"]["formula"] = "as onestep with r(s) replaced by clip(c(s), -1, 1), c the world's cloned actor"
    trig["onestep_lit"]["formula"] = "score = |delta*(s, a)|; fire iff score > q95"
    trig["density"]["formula"] = ("score = 1/2 sum_k ((a_k - c_k(s)) / sigma_k)^2, c = clip(cloned actor, -1, 1), "
                                  "sigma_k = sqrt(mean_i (a_ik - c_k(s_i))^2) over the gapped training rows; fire iff "
                                  "score > q95 (the plan's penalty [score - q95]_+ is positive)")
    trig["ratio"]["formula"] = ("score = logit g(normalise(s), a); g fitted by logistic loss to label 0 at (s_i, a_i) "
                                "and label 1 at (s_i, clip(pi(s_i) + a_i - c(s_i), -1, 1)) for every gapped training "
                                "row i (equal class sizes, so the logit is log w); pi the deployed actor (its own input "
                                "normalisation), c the world's cloned actor; 2 hidden layers of 128 (ReLU), 10,000 Adam "
                                "steps, learning rate 0.001, batch 512, rows drawn by default_rng([20261013, seed, 81, "
                                "world index, actor tag]), init key from the same stream; fitted once per seed, world "
                                "and actor; fire iff score > q95")
    trig["ratio_refuse"]["formula"] = ("fire iff logit g(s, a) > l* = log(0.1 / 0.9) + log sum_j exp(logit g(s_j, a_j)) "
                                       "over the world's gapped calibration-bank rows (Prepared.bank_rows)")
    trig["heads"]["formula"] = "score = |Q1(s, a) - Q2(s, a)| (learners.critic_rows); fire iff score > q95"
    trig["sa95"]["formula"] = ("fallback.Reference.sa.novel: the 10th-nearest-neighbour distance of (s, a) to the gapped "
                               "training (state, command) rows in standard units above the fallback diagnostic's 95% "
                               "split-conformal level")
    return dict(
        label=C.LABEL, kind="deploy-spec", version=VERSION, previous_spec_sha256=PREVIOUS_SPEC_SHA256,
        changes_before_any_run=CHANGES, changes_check=CHANGE_CHECK,
        component="deploy: the deployment-check channel (plan section 4.3); no training",
        plan=dict(path=str(C.PLAN_PATH), sha256=C.file_sha256(C.PLAN_PATH),
                  sections=["2 (item 7)", "3 (Variant B)", "4.3", "4.8"]),
        no_bca=("No BCA band, calibrator, threshold, dose or BCA update is fitted, read or called. Nothing from "
                "algorithms/td3_bc_bca.py or learners.bca_* is called. From calibration/ only calibration/bank."
                "stratified_bank runs, because the gate's data loader (experiments/shift/dataset.py through "
                "acceptance.load_seed) calls it to regenerate the log's calibration-bank split (a NumPy row sampler). "
                "The refusal limit l* of the reported ratio_refuse trigger is the generic weighted split-conformal "
                "formula, computed in NumPy from classifier weights; no BCA code."),
        outcome_leakage=("Only saved check-2 actors, the gate's frozen fixed-controller critics, cloned actors and logs, "
                         "and the fallback diagnostic's machinery are used. No room-test values, futility, lock or log "
                         "file is read. At launch a full job reads only the three pass flags of the room's pooled.json, "
                         "or, when that file does not exist, only the stop flag of the room's futility.json (the launch "
                         "condition, change C2); nothing computed depends on them. The smoke check withholds every "
                         "point value."),
        pinned=dict(pins, how=("the room harness runs from the exact bytes of runs/shift_explore/room/code/harness.py "
                               "only when their SHA-256 equals the room code manifest's (whose sidecar must match and "
                               "which must name the room spec); fallback.py from its exact bytes only when they have "
                               "fallback_py_sha256. Both are executed in memory (no bytecode written)."),
                    room_harness_functions_used=["SeedContext (load_prereg, guard, load_seed, get_critic(train="
                                                 "False), clone, h2, anchors)", "oracle64", "hinge_bounds (check)",
                                                 "GATE_FILES", "A, L, TC, RC, TR (the gate modules it imports)"],
                    fallback_functions_used=["Reference", "conformal_threshold", "fallback_command",
                                             "run_rule (smoke equivalence check only)", "EVAL_CHECK_ROWS"]),
        inherited_from_room_spec=dict(
            room_spec_sha256=pins["room_spec_sha256"], change="A1 main oracle re-referenced to the recorded command",
            main_penalty=room["oracle"]["main_penalty"], residual=room["oracle"]["residual"],
            reading_here=("penalties and the one-step trigger are read relative to the recorded command at the same "
                          "state; at a deployment state that command is r(s) (deviation D2)")),
        gate_prereg_sha256=C.file_sha256(C.ROOT / "runs" / "shift" / "preregistration.json"),
        gate_files_sha256={rel: C.file_sha256(C.ROOT / rel) for rel in H.GATE_FILES},
        setup=dict(
            variant="base", alpha=5.0, seeds=list(RJ.SEEDS),
            actors_source=("check 2's saved actors: runs/shift/<prereg>/harm_return/base/alpha5/seed<k>/actors.msgpack, "
                           "verified by the SHA-256 in its values.json (room harness SeedContext.h2)"),
            jobs=[dict(world=w, actor=a, actor_home_world=RJ.HOME[a], role=RJ.ROLE[(w, a)]) for w, a in RJ.JOBS],
            starts=("the first 200 starts of each 1,000-start evaluation bank (thruster_chain.bank_starts), the same "
                    "starts as the fallback diagnostic, for every rule"),
            situations=list(S.TC.SITUATIONS), episode_steps=100,
            scoring=("exact simulation (thruster_chain.dynamics, float64); points = 100 (J - J_recording) / (J_top - "
                     "J_recording) with the gate's anchors of the deployment world (acceptance.anchors_of)"),
            actor_command=("clip(pi(s), -1, 1), pi = learners.mlp_policy (float64 copy of the actor) with the actor's "
                           "home world's input normalisation"),
            deployment_world_parts=("the deployment world's physics, gapped training rows, held-out rows, frozen "
                                    "fixed-controller gapped critic (acceptance.get_critic(prereg, sd, 'gap', "
                                    "'stall_aware', train=False)), cloned actor (clone / no_stall_clone) and anchors")),
        heldout=dict(
            level_rows=("the calibration split (200 recorded episodes no learner saw) minus rows whose command is in "
                        "the removed ball: about 16,800 rows, exactly fallback.Reference's"),
            level_rule="q95 = the ceil((n + 1) * 0.95)-th smallest score at those rows (fallback.conformal_threshold)",
            check_rows=("fallback.Reference's 5,000 evaluation-split rows outside the removed ball; the share above "
                        "each level is reported (about 5% expected)")),
        fallbacks=dict(
            clip99=("primary: the actor's command clipped per thruster to the recorded range, the 0.5% and 99.5% "
                    "quantiles of the gapped training commands (fallback.Reference lo99 / hi99; checked equal to the "
                    "room harness's hinge_bounds)"),
            clone="reported: the deployment world's cloned actor's command",
            cap80="gate check only: the actor's command capped at +-0.8"),
        triggers=trig,
        random_control=dict(
            rule=("for every trigger T and fallback F: a coin per start and step that fires with probability equal to "
                  "the share of (start, step) pairs on which T -> F fired, for the same seed, world, actor, situation "
                  "and fallback; the fallback is then applied exactly as for T"),
            coin_seed="default_rng([20261013, seed, 91, world index, actor tag, situation index, fallback index, "
                      "trigger index])"),
        rules=[RJ.key(t, f) for t, f in RJ.RULES],
        per_rule_outputs=["points (mean over the 200 starts)", "per-start points (5 decimals)", "J",
                          "fire_share", "changed_share", "past80_proposed", "past80_applied",
                          "recall_past80 (share of past-80% proposals on which the rule fired)",
                          "precision_past80 (share of firings on past-80% proposals)",
                          "flag_rate_not_past80", "rate and coin seed (random rules)", "seconds"],
        gate_check=("world-matched jobs, every situation, before the rules: never switching and capping at 80% on the "
                    "whole 1,000-start bank reproduce check 2's saved points and capped points within 1e-6, else the "
                    "job stops (stop_<job>.json); cross-world jobs: not applicable"),
        marks=dict(
            channel_open=("'The channel is open' iff, on the gapped critic's actor in the stalling world with the clip "
                          "fallback, pooled over seeds 0-4 (mean of the per-seed paired differences, every seed "
                          "present): (i) points(perfect detector -> clip) - points(random at its own rate -> clip) "
                          ">= 15 in strong current or in big kicks, and (ii) points(actor alone) - points(perfect "
                          "detector -> clip) <= 3 in recorded conditions"),
            reported=[
                "the same reading for plain TD3+BC in the stalling world",
                "for every trigger, fallback and job: trigger minus random at its own rate, change from the actor "
                "alone, cost in recorded conditions, catch rate and precision on past-80% proposals, share switched",
                "for every trigger: whether it clears the channel-open bar (>= 15 over random in strong current or in "
                "big kicks, cost <= 3; plan 4.3) and plan 5.3's Stage-2 bar (>= 10 over random in strong current only, "
                "because plan 5.3 reads its marks in strong current unless it says otherwise; cost <= 2) (change C1)",
                "seed counts (positive seeds) and standard errors over seeds",
                "the launch condition shared by every job, first in summary.json and in tables.md (change C3)",
                "the shape readout of every trigger, pooled over seeds (shape_readout; gates nothing; change C4)"],
            selectivity=("reported, for three actor pairs (gapped critic's actor of each world; the stalling world's "
                         "gapped critic's actor in both worlds; plain TD3+BC in both worlds), every trigger, fallback "
                         "and situation, pooled over seeds: stalling-world gain minus no-stall loss = (T - alone)_stall "
                         "+ (T - alone)_no_stall; targeting = (T - random_T)_stall + (T - random_T)_no_stall; the same "
                         "sum for clipping every step; catch-rate selectivity = catch rate of past-80% proposals, "
                         "stalling minus no-stall")),
        pooling="paired by seed (and by start within a seed); mean over seeds 0-4; standard error over seeds, ddof 1",
        smoke=("seed-0 check before any full run: every job of seed 0 with the first 20 starts of every bank, every "
               "rule; plus a 200-start timing probe in strong current for the gapped critic's actor. Outputs under "
               "smoke/; every point value withheld (finiteness and a hash only); the gate check runs as in a full job "
               "(its difference only); an equivalence check reruns the rules shared with the fallback diagnostic "
               "(never switch, every step to clip and to clone, perfect detector, distance check, a random coin at "
               "rate 0.25) through fallback.run_rule and records the largest difference; timing per rule; the shape "
               "readout runs as in a full job (it holds no point value). After spec v2: changes_check (the per-start "
               "point hashes of every rule equal the v1 smoke's)"),
        launch=dict(
            condition=("a full job runs only after the room test's pooled look: runs/shift_explore/room/pooled.json "
                       "for the pinned room spec, labelled, with its room, recovery and keeps-improvement marks passed. "
                       "--after-room-failure lets it run after a room failure (plan 4.8: the deployment check remains): "
                       "a pooled look that did not pass, or, when no pooled.json exists, a futility look that stopped "
                       "the room (runs/shift_explore/room/futility.json for the pinned room spec, labelled; only its "
                       "stop flag is read). A futility look that says continue, with no pooled look yet, refuses; so "
                       "does a room that has taken neither look. The condition is recorded in every values and stop "
                       "file; the summary refuses unless all jobs share it, and prints it first (changes C2, C3)"),
            commands="launch.sh <seeds> <cpu> [--after-room-failure], then launch.sh summary <cpu>",
            logs="runs/shift_explore/deploy/logs/seed<k>.log and summary.log"),
        interpretations=[
            "Every score-based trigger, the classifier ratio included, uses the same rule as Variant B's tau and the "
            "plan's one-step trigger: above its 95% level on held-out recorded pairs.",
            "The mark's 'at least 15 points' and 'cost at most 3' are pooled means of per-seed paired differences "
            "(as the room spec made its room mark exact); the plan gives no seed count for this mark, so seed counts "
            "are reported, not gated.",
            "The mark is read on the gapped critic's actor (the plan's main actor; the room test's baseline); plain "
            "TD3+BC's reading is reported.",
            "'Cost in recorded conditions' is the actor alone minus the perfect detector with the clip fallback.",
            "Selectivity follows plan 4.2's 'stalling-world gain minus no-stall loss' and plan 4.4's catch-rate "
            "difference.",
            "The room's check (e) (a penalty must rise past 80%) gates penalties, which the actor climbs in training. "
            "Triggers are not climbed, so no trigger is gated on its shape; the shape is reported (change C4)."],
        deviations=[
            dict(id="D1", what=("plain TD3+BC in the no-stall world is the stalling world's TD3+BC actor deployed there "
                                "(its own input normalisation; the no-stall world's data, critic, cloned actor, range "
                                "and anchors for triggers, fallbacks and points); the stalling world's gapped critic's "
                                "actor is deployed there too, as its same-actor companion (reported)"),
                 why=("the gate trained TD3+BC hosts only in the stalling world (check 2 saved td3bc_gapped only "
                      "there); training a no-stall TD3+BC would be a new training run, outside this no-training "
                      "component")),
            dict(id="D2", what=("the one-step trigger is re-referenced to r(s), the recording controller's own command "
                                "at the state without its wobble, in place of the row's recorded command a_i "
                                "(room change A1); the cloned actor's command is reported as the other reference, the "
                                "plan's literal |delta*| as a third"),
                 why=("a deployment state has no recorded command, and the level must come from the same function "
                      "of (state, command) at held-out pairs; with a_i itself every held-out score is 0 and the "
                      "level is 0")),
            dict(id="D3", what=("the critic-based triggers read the deployment world's frozen fixed-controller gapped "
                                "critic for every actor, plain TD3+BC included (its own co-trained critic is not read)"),
                 why=("it is the room test's oracle critic and the plan's 'frozen critic'; a deployment check is a "
                      "function of the (state, command) pair, so both actors face the same check; no no-stall TD3+BC "
                      "critic exists")),
            dict(id="D4", what="the distance check (sa95) is re-run with the clip fallback only (reported)",
                 why=("cost: its neighbour search is the slowest trigger (about 20 s per rule and situation); its "
                      "cloned-actor numbers exist in the fallback diagnostic")),
            dict(id="D5", what=("the levels use the whole held-out calibration split outside the removed ball, not "
                                "only its non-bank rows (Variant B's tau)"),
                 why=("no band is calibrated on the bank here; this keeps the fallback diagnostic's rows and levels; "
                      "bank rows are about 5% of them")),
            dict(id="D6", what=("the classifier is fitted once per seed, world and actor, not refitted every 4,000 "
                                "steps (plan 4.2)"),
                 why="nothing is trained here; the deployed actor is fixed"),
            dict(id="D7", what=("launch condition: after the room test's pooled look passes, as instructed; with "
                                "--after-room-failure also after a room failure: a pooled look that did not pass, or a "
                                "stop at the room's futility look, when no pooled look is ever written (change C2)"),
                 why=("the instruction gates this component on the room's pooled look; plan 4.8 keeps the deployment "
                      "check when the room fails (its 'room fails' and 'recovery fails' rows both say so), and a "
                      "futility stop is the room's early way of failing, so the override exists for both and is "
                      "recorded in every output and in the pooled summary"),
                 open_question=("whether the override is used at all is the user's decision; it was raised with the "
                                "first spec and is not decided here")),
            dict(id="D8", what="the seed-0 smoke check withholds every point value",
                 why="so that no code or spec choice can be made on returns before the full run")],
        shape_readout=dict(
            role="reported, never a gate (change C4)",
            states=("the room harness's shape_rows of the deployment world: 2,000 full-throttle-region training states, "
                    "default_rng([20261015, seed, 65]) (the states of the room's check (e))"),
            commands=("the deployed actor's command clip(pi(s), -1, 1) with thruster 1 set to sign * m along the "
                      "actor's own push (sign of its thruster-1 command), the other thrusters at the actor's command; "
                      "m = 0.8 is the float32 knee (thruster_chain.KNEE), which is not past it"),
            grid=[float(m) for m in H.SHAPE_GRID],
            per_trigger=["share of states whose score falls / rises / stays from 80% to 81% (the room's (e1) reading; "
                         "the room's 25% limit for penalties is shown for reference only)",
                         "mean score at each throttle (not for sa95)",
                         "firing rate at each throttle, and at 1.0 minus at 0.8",
                         "firing rate at the actor's own command"],
            scores=("each trigger's own score and level (Signals.score, Signals.thr); unrec: max_k |a_k|, firing by "
                    "thruster_chain.past_knee; ratio_refuse: the classifier logit against l*; sa95: the 10th-neighbour "
                    "distance (fallback.Novelty.dist) at 80% and 81% for the fall share, firing by its exact decision "
                    "at every throttle"),
            no_return="the frozen critic, the classifier and the neighbour search only: no rollout, no return, no point"),
        compute_estimate=("from the v1 seed-0 smoke under full contention (4 CPUs busy with the room test, these jobs at "
                          "nice 15): about 7.5 min per full job, about 3.1 process-hours for the 25 jobs (about 2 on "
                          "idle CPUs); the shape readout (change C4) adds under a minute per job; the v2 smoke logs "
                          "record its time"),
        outputs=str(C.OUT),
    )


def main():
    if any_run_output():
        raise SystemExit("refused: deploy run outputs exist; the spec may not change after a real run: "
                         + ", ".join(str(p) for p in any_run_output()))
    p = payload()
    if C.SPEC_PATH.exists():
        old = C.read_json(C.SPEC_PATH)
        if C.canonical(old) == C.canonical(p):
            print("spec.json already written and identical")
            return
        old_sha = C.file_sha256(C.SPEC_PATH)
        k = int(old.get("version", 1))
        if p["version"] != k + 1 or not p["changes_before_any_run"] or p.get("previous_spec_sha256") != old_sha:
            raise SystemExit(f"refused: a new spec must be version {k + 1}, list its changes and name previous_spec_sha256")
        hist = C.OUT / "spec_history"
        hist.mkdir(exist_ok=True)
        for suffix in ("", ".sha256"):
            shutil.copy2(str(C.SPEC_PATH) + suffix, hist / f"spec_v{k}.json{suffix}")
        sup = C.OUT / "superseded" / f"v{k}"
        sup.mkdir(parents=True, exist_ok=True)
        for name in ("smoke", "code", C.MANIFEST_NAME, C.MANIFEST_NAME + ".sha256"):
            if (C.OUT / name).exists():
                shutil.move(str(C.OUT / name), sup / name)
        old_logs = sorted((C.OUT / "logs").glob("smoke_*.log"))
        if old_logs:
            (sup / "logs").mkdir(parents=True, exist_ok=True)
            for f in old_logs:
                shutil.move(str(f), sup / "logs" / f.name)
        print(f"archived spec v{k} to {hist}; moved its smoke outputs and manifest to {sup}")
    digest = C.write_json(C.SPEC_PATH, p)
    print(f"wrote {C.SPEC_PATH} version {p['version']} sha256 {digest}")


if __name__ == "__main__":
    main()
