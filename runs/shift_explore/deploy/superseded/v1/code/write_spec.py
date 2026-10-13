"""EXPLORATORY: write runs/shift_explore/deploy/spec.json (+ .sha256) before any real deploy run.

The spec may be rewritten only while no job output, stop file or summary exists (no real run has happened). A rewrite
must raise the version and name the SHA-256 of the spec it replaces, and must state its changes (CHANGES); the old spec
goes to spec_history/ and the outputs computed under it (smoke checks, code manifest and copies) to superseded/v<k>/.

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
VERSION = 1
CHANGES = []     # filled for any later version: what changed, why, and that no return was seen


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
        label=C.LABEL, kind="deploy-spec", version=VERSION, changes=CHANGES,
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
                         "file is read. At launch a full job reads only the three pass flags of the room's pooled.json "
                         "(the launch condition); nothing computed depends on them. The smoke check withholds every "
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
                "for every trigger: whether it clears the channel-open bar (>= 15 over random in strong current or big "
                "kicks, cost <= 3) and plan 5.3's Stage-2 bar (>= 10 over random, cost <= 2)",
                "seed counts (positive seeds) and standard errors over seeds"],
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
               "rate 0.25) through fallback.run_rule and records the largest difference; timing per rule"),
        launch=dict(
            condition=("a full job runs only after the room test's pooled look: runs/shift_explore/room/pooled.json "
                       "for the pinned room spec, labelled, with its room, recovery and keeps-improvement marks passed; "
                       "--after-room-failure lets it run after a look that did not pass (plan 4.8: the deployment "
                       "check remains), which is recorded in every output"),
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
            "difference."],
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
                                "--after-room-failure also after a failing look"),
                 why=("the instruction gates this component on the room's look; plan 4.8 keeps the deployment check "
                      "when the room marks fail, so the override exists and is recorded")),
            dict(id="D8", what="the seed-0 smoke check withholds every point value",
                 why="so that no code or spec choice can be made on returns before the full run")],
        compute_estimate=("pre-smoke, from a probe under full contention: per job about 1 min of loading and signal "
                          "fitting plus about 5-6 min of rules; 25 jobs about 2-3 process-hours; the smoke's timing "
                          "replaces this"),
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
        if p["version"] != k + 1 or not p["changes"] or p.get("previous_spec_sha256") != old_sha:
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
        print(f"archived spec v{k} to {hist}; moved its smoke outputs and manifest to {sup}")
    digest = C.write_json(C.SPEC_PATH, p)
    print(f"wrote {C.SPEC_PATH} version {p['version']} sha256 {digest}")


if __name__ == "__main__":
    main()
