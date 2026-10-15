"""Write runs/shift_explore/detect/spec.json (+ .sha256), labelled EXPLORATORY, before any real run of the detection
rehearsal. Refuses to overwrite an existing spec unless --reason is given; the old one is then kept in spec_history/.
Usage: python write_spec.py [--reason "..."]
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

VERSION = 5


def spec_payload():
    import pairs as P
    import labels as LB
    import signals as S
    import measures as M
    import detect as D
    import run_detect as R

    H = C.load_room_harness()
    contract = C.HERE.parent / "stage2" / "detect_contract.py"
    sig = {}
    for n in S.ALL_NAMES:
        kind = "stand_in" if n in S.STANDIN_NAMES else ("oracle" if n in S.ORACLE_NAMES else "reported")
        units = "value" if n in ("two_head", "residual_evidence") or n in S.ORACLE_NAMES else "other"
        sig[n] = dict(kind=kind, units=units)
    sig["distance"]["formula"] = ("d10(s, a): Euclidean distance in z-space (state 13 + command 4, standardised by the "
                                  "mean and SD of the world's kept training rows) to the 10th-nearest kept training "
                                  "row; exact (scipy cKDTree); in-sample at training states")
    sig["density"]["formula"] = ("N(s, a) = 1/2 sum_k ((a_k - c_k(s)) / sigma_k)^2; c the world's cloned actor (float32 "
                                 "network at the normalised float32 state); sigma_k = sqrt(mean over kept training "
                                 "rows of (a_ik - c_k(s_i))^2)")
    sig["classifier_ratio"]["formula"] = (
        "log w(s, a) = logit of g: Dense 64-ReLU-Dense 64-ReLU-Dense 1 on (normalised obs / 1.001, command); target "
        "pairs (s_i, clip(pi(s_i) + a_i - c(s_i), -1, 1)) = 1 vs recorded (s_i, a_i) = 0 at the same rows; pi the "
        "world's gapped critic's actor; c the cloned actor; loss mean softplus(-logit_target) + mean "
        "softplus(logit_recorded); 2,000 torch_adam steps, lr 0.001, 512 rows per step from default_rng([20261015, "
        "seed, 82]).integers(n, (2000, 512)); init PRNGKey(default_rng([20261015, seed, 84]).integers(2^31 - 1)). "
        "Identical to stage2/weights.py and standins.py fit_classifier. Also recorded: l* = log(0.1/0.9) + "
        "logsumexp over the calibration bank of log w, with flags at l* reported beside the 95% level")
    sig["two_head"]["formula"] = "|Q1(s, a) - Q2(s, a)| of the world's frozen gapped fixed-controller critic"
    sig["residual_evidence"]["formula"] = (
        "-(1/10) sum over the 10 nearest kept training rows j of (s, a) in z-space (the distance's neighbours, "
        "in-sample) of delta_j; delta_j = r_j + 0.98 min_h Q_h(s'_j, c(s'_j)) - min_h Q_h(s_j, a_j), logged r_j and "
        "s'_j, c the stall-aware controller as the critic's target reads it (learners.FixedController), no noise")
    sig["perfect_band"]["formula"] = (
        "[delta*(s, ref) - delta*(s, a)]_+ (the room's main oracle, change A1); delta*(s, a) = r(s, a) + 0.98 min_h "
        "Q_h(s', c(s')) - min_h Q_h(s, a), (s', r) one float64 step of thruster_chain.dynamics_jax in the world's "
        "physics, c the stall-aware controller (room harness oracle64, q_head='min'); ref = the recorded command at T1, "
        "T2, T4 states and the cloned actor's command at T3 states. The sanity check; excluded from 'best stand-in'")
    sig["perfect_band_signed"]["formula"] = "-delta*(s, a) (reported)"
    sig["perfect_band_literal"]["formula"] = "|delta*(s, a)| (the plan's literal form, reported)"
    sig["range_hinge"]["formula"] = ("100 sum_j ([a_j - hi_j]_+^2 + [lo_j - a_j]_+^2), lo_j / hi_j the 0.5% / 99.5% "
                                     "quantiles of the world's kept training commands (reported; not a stand-in)")
    deviations = [
        dict(id="D1 perfect band re-referenced (change A1)",
             change=("the perfect band scored as the sanity check is the room's amended main oracle "
                     "[delta*(s, ref) - delta*(s, a)]_+, not the plan's |delta*(s, a)|; the literal and the signed "
                     "forms are scored beside it, reported"),
             reason="room spec v2 change A1: |delta*| falls just past the knee; the task requires the re-referenced form"),
        dict(id="D2 reference command at T3 states",
             change="ref = the world's cloned actor's command at the state (T3 states have no recorded command)",
             reason=("the re-referenced form needs a command recorded at the same state; the cloned actor is the "
                     "data's behaviour there. Fixed before any signal or label was computed")),
        dict(id="D3 81% knee pairs added to T1 (reported)",
             change=("T1 gets a fifth role 'knee' = float32(clip(command, -0.81, 0.81)) (every thruster clipped to "
                     "81%); it enters ONLY the knee_* measures (rise, fall and tie shares and the mean change of the "
                     "score and of the paired penalty against the cap), never a ranking, catch-rate or false-flag pool "
                     "(change C2); the plan's four roles and every measure built on them are unchanged"),
             reason=("a related, descriptive readout of whether a signal rises just past 80%, the question check (e) "
                     "(room change A2) asks of a penalty; it is not check (e) and nothing about it is pass/fail "
                     "(change C5)")),
        dict(id="D4 size read two ways",
             change=("size_T1_raw = median (s(cmd) - s(cap)) / over_pref (plan 4.4's words) and size_T1_paired = "
                     "median ([s(cmd) - s(rec)]_+ - [s(cap) - s(rec)]_+) / over_pref; Stage 2's size mark uses the "
                     "paired one. Pair by pair the paired difference has the sign of the raw difference (or is 0) "
                     "and is never larger in size: 0 <= paired <= raw when the score rises from the cap to the "
                     "command, raw <= paired <= 0 when it falls (change C4). So a pair whose paired difference is at "
                     "least c > 0 has a raw difference of at least c; the two medians are not ordered in general"),
             reason=("A1's lesson: the consumer reads the width relative to the recorded command at the same state "
                     "(stage2/policy.py k_paired [U(s, a) - U(s, a_i)]_+); 'so that k = 1 would cancel the false "
                     "preference' (plan 5.2) is about that penalty")),
        dict(id="D5 stand-ins scored as raw scores",
             change=("behaviour density as N (not [N - q95]_+), classifier ratio as log w (not U_max softplus(log w - "
                     "l*)), distance as the exact d10 (not the fitted surface of plan 4.2)"),
             reason=("the plan's forms are monotone in these scores above their limits, so the flags at the 95% level "
                     "on T4 are the same (density) or reported beside (l*); the thresholded forms tie most pairs at 0, "
                     "which blanks the paired and ranking measures; U_max is a BCA quantity and is not read; the "
                     "surface exists only to give the training penalty a gradient")),
        dict(id="D6 residual evidence uses BCA's residual form",
             change="delta_j with min over the two heads on both sides, logged transitions, stall-aware controller",
             reason="room change A3: BCA's own residual; the same form as the trace test bed (plan 4.6)"),
        dict(id="D7 in-sample neighbours at training states",
             change="at T1 and T2 states the neighbour search includes the row itself (plain 'nearest training rows')",
             reason="the plan's definition and plan 4.2's stand-ins; BCA's band is likewise in-sample at its fit rows"),
        dict(id="D8 how T3 states are drawn",
             change=("first 200 starts of the strong-current bank, exact 100-step rollouts, 1,500 of the 20,000 visited "
                     "states per policy (default_rng tags 103, 0 / 103, 1), rounded to float32"),
             reason="the plan does not say; the fallback diagnostic and the anatomy readout used the same 200 starts"),
        dict(id="D9 truth and critic preference",
             change=("true preference = exact Q of the stall-aware controller (the critic's scored policy) in the "
                     "world's physics, 1,000 steps, float64; critic preference = Q1 (the head the actor climbs)"),
             reason=("the plan does not name them; these are truth.pull_past_80's quantities and the critic's own target "
                     "policy (also in the no-stall world, room deviation)")),
        dict(id="D10 'copies in the no-stall world'",
             change=("the same rules and tags applied to the no-stall world's own log, gapped critic, gapped critic's "
                     "actor ('no_stall_gapped') and cloned actor"),
             reason=("selectivity compares each world's detector on its own actor's proposals (the fallback "
                     "diagnostic's no-stall control); Stage 2's contract builds per world")),
        dict(id="D11 range hinge scored (reported)", change="the recorded-range hinge is scored beside the stand-ins",
             reason="plan section 2 point 8: score everything against a plain recorded-range limit; not a stand-in"),
        dict(id="D12 order rule",
             change=("in the rehearsal's folder (runs/shift_explore/detect) Stage 1 signals are scored before the "
                     "labels of that seed and world exist: enforced in the module (signals.score_signal refuses once "
                     "the labels exist; labels.run refuses until signals_done.json exists under the current spec and "
                     "manifest) and in run_detect.py; Stage 2's folder is exempt (change C7)"),
             reason="a guard against any signal being adjusted on labels; not in the plan"),
        dict(id="D13 labels for every pair",
             change="every pair (every role, every set) is labelled against its own 80% cap",
             reason="the ranking score needs a label for every pair; caps are harmless by construction"),
        dict(id="D14 ranking pool, catch rates and false flags",
             change=("ranking over every pair of the plan's roles in the set(s) (T1 and T2: command, cap, recorded, "
                     "cloned; T3: command, cap; never the knee pairs, change C2); catch rate over harmful command "
                     "pairs and over past-80% command pairs; false flags over the harmless pairs of the same pools "
                     "(per plan role reported); level = the signal's own split-conformal 95% level on T4"),
             reason="the plan gives the measures' names only; these are the readings fixed before any run"),
        dict(id="D15 commands are the float32 network's",
             change="a pair's command is the float32 actor at the stored float32 state; T3 rollouts use the float64 copy",
             reason="the critic, BCA's width and every signal see the float32 command; the gate evaluates the copy"),
    ]
    changes = [
        dict(id="C1 (spec v2) label parameter, write-once pair sets, timing projection",
             finding="the builder's own, before any run (v2 had no list; recorded here)",
             change=("a label parameter for the pair sets and labels that Stage 2 builds on fresh seeds (pairs.save, "
                     "labels.save and run, detect.pair_sets); pair sets written once; the seed-0 check's timing "
                     "projection fits every signal at once, as the signals phase does"),
             reason="spec v2's stated reason; no pair set, label, signal or measure definition changed"),
        dict(id="C2 (spec v3) the knee pairs leave every ranking and false-flag pool",
             finding="review of the detect component, finding 1 (major)",
             change=("ranking_T1, ranking_T12, ranking_T12_among_past80 and false_flag_T1 (and its per-role values) "
                     "are computed over the plan's roles only (pairs.PLAN_ROLES: T1 and T2 command, cap, recorded, "
                     "cloned; T3 command, cap). The 81% knee pairs enter only the knee_* measures. The measures "
                     "record n_pool_T12 and n_harmful_pool_T12. D3 and D14 reworded. A synthetic test in which the "
                     "knee pairs would change ranking_T12 (test_detect.py test_measures) and a seed-0 check that "
                     "ranking_T12 and ranking_T1 equal the plan-role values were added"),
             reason=("D3 declared the knee pairs 'reported only' and the plan's four T1 commands unchanged, but the "
                     "pools took every role of T1, so knee pairs (each labelled against its own cap) entered "
                     "ranking_T12, a gated Stage 2 mark and the claim rule's comparison with the best stand-in, and "
                     "D14 said 'every pair'. Plan 4.4 gives T1 four commands. The fix follows the plan's definition "
                     "and D3's own declaration; it was not chosen by its effect on any number"),
             evidence=("reviewer, seed-0 check files (two-head only, stalling world): 70 of the 613 knee pairs are "
                       "labelled harmful; ranking_T12 0.6406 with the knee pairs, 0.6297 without; 543 of the 647 "
                       "harmless past-80% pairs were knee pairs. Re-derived to verify the finding only: "
                       "scratchpad/ood/review_detect_apply/verify_findings.py")),
        dict(id="C3 (spec v3) records are reused only under the current spec and code manifest",
             finding="review of the detect component, finding 2 (minor)",
             change=("every record this component writes (pair sets, scores, labels, signals_done, measures, pooled) "
                     "carries detect_spec_sha256 and detect_code_manifest_sha256 (common.stamp). Every reuse refuses "
                     "unless they equal the current ones (common.check_version): pairs.load, labels.load, "
                     "detect.labels, the join (scores, labels and pair sets), and in run_detect.py every existing "
                     "output a phase would skip (also its spec_sha256 and code_manifest_sha256) and every measures "
                     "record the pooled look reads. A record without the stamp is refused. A new spec version "
                     "therefore starts in a fresh output folder or archives the old outputs first (v3 archived v2's "
                     "seed-0 check in superseded/v2, v4 v3's in superseded/v3, v5 v4's in superseded/v4)"),
             reason=("every phase skipped an existing output, and the pooled look pooled it, after a sidecar check "
                     "alone; a rerun under a new spec version (which the perfect-band sanity rule foresees) would "
                     "have reused scores, labels and measures written under the old one and mixed versions in "
                     "pooled.json; the room test guards against this (its change A7). The check sits in the module "
                     "so that Stage 2, which reads the rehearsal's records through this module's loaders and join, "
                     "meets it too. Stage 2's own phase_detect_standins reads the rehearsal's score records with its "
                     "own reader; those scores reach a mark only through this module's join, which now refuses "
                     "another version"),
             evidence=("run_detect.py v2 phase_measures ('if mo.exists(): C.read_json(mo); continue') and "
                       "phase_pooled; checks/checks_seed0.json record_versions; test_detect.py")),
        dict(id="C4 (spec v3) the paired size's relation to the raw size stated correctly",
             finding="review of the detect component, finding 3 (minor)",
             change=("D4, the measures' doc and change A1's lesson no longer say the paired difference is 'never "
                     "larger than the raw one'. They say: pair by pair it has the sign of the raw difference (or is "
                     "0) and is never larger in size; 0 <= paired <= raw when the score rises from the cap to the "
                     "command, raw <= paired <= 0 when it falls. No definition changed"),
             reason=("the old wording was false whenever the score at the command is below the score at the cap. "
                     "The reviewer's proposed wording ('when it falls, the paired form gives 0') was not exact "
                     "either: with s(cmd) 3, s(cap) 4, s(rec) 0 the paired difference is -1. The reviewer's point "
                     "that a pair with paired difference at least 1 has raw difference at least 1 holds pair by "
                     "pair; the two medians are not ordered in general (two pairs, paired -0.5 and 2.5, raw -10 and "
                     "2.5: medians 1.0 and -3.75)"),
             evidence="reviewer's example s(cmd) 0, s(cap) 1, s(rec) 2: raw -1, paired 0"),
        dict(id="C5 (spec v3) the knee measures are a related descriptive readout, not check (e)",
             finding="review of the detect component, finding 4 (minor)",
             change=("beside knee_rise_* (strict rises; ties count as no rise) the measures now report, for the "
                     "score and for the paired penalty against the cap over past-80% and harmful T1 pairs: "
                     "knee_fall_* (the analogue of check (e1), which allows a fall on at most 25% of states), "
                     "knee_tie_*_paired, and knee_mean_change_* (the analogue of check (e2), a rise of the mean). "
                     "The spec, the measures' doc and change A1's lesson say plainly that these are a related "
                     "descriptive readout and not check (e); nothing about them is pass/fail"),
             reason=("the builder's report and A1's lesson said the knee-rise measure 'carries over check (e)'s "
                     "requirement'. It does not: the knee pair clips every thruster to 81%, while check (e) moves "
                     "thruster 1 alone along a grid; it counts strict rises only, while check (e) asks that the "
                     "penalty falls on at most 25% of states (e1), that its mean rises (e2), that its slope is "
                     "positive (e3) and that the best throttle moves (e4). The two can disagree on the same penalty"),
             evidence=("reviewer: two-head paired penalty, harmful T1 pairs, seed 0: rise 0.41, fall 0.24, ties "
                       "0.35, mean change +0.0008; re-derived (0.406, 0.244, 0.350, +0.00081) to verify the finding "
                       "only")),
        dict(id="C6 (spec v3) the bars carried forward are recorded as coming from a different readout",
             finding="review of the detect component, finding 5 (minor)",
             change=("bars_carried_forward records each bar's own readout and drops '95% level' from the 67% bar: "
                     "the 35% is trigger (b), new (state, command), at the 95% split-conformal level set on about "
                     "16,800 held-out calibration-split rows; the 67% is trigger (c), an exact-return comparison "
                     "(switch when the clipped command has the higher exact return), with no level and not a scored "
                     "signal; both were counted along rollouts that the clip fallback changed. catch_rate_T3a_past80 "
                     "scores the actor's unchanged rollouts with a level set on T4. The bars are reported beside it "
                     "for reference only"),
             reason="the old source string tied both numbers to catch_rate_T3a_past80 as if measured the same way",
             evidence=("fallback.md: trigger (c)'s definition and the table 'Which harmful commands each check "
                       "catches (strong current, clip fallback)'")),
        dict(id="C7 (spec v3) labels chosen by folder; the order rule (D12) in the module; Stage 2's folder exempt from D12",
             finding="review of the detect component, finding 6 (minor)",
             change=("(a) the label of pairs.save, labels.save and labels.run, signals.score_signal and "
                     "detect.pair_sets defaults to the output folder's (common.label_for): the Stage 1 label in "
                     "runs/shift_explore/detect, LABEL_STAGE2_FRESH ('EXPLORATORY/STAGE 2 output ...') in "
                     "runs/shift_explore/stage2. A Stage 1 label passed for Stage 2's folder, a Stage 2 label for "
                     "the rehearsal's folder, or a label not starting with EXPLORATORY is refused. (b) D12 is "
                     "enforced in the module for the rehearsal's folder: signals.score_signal refuses once the "
                     "labels of that seed and world exist; labels.run refuses until signals_done.json exists under "
                     "the current spec and manifest. (c) Stage 2's folder (fresh seeds 5-9) is exempt from D12"),
             reason=("(a) Stage 2 never passes label=, so its fresh-seed pair sets, labels and stand-in scores would "
                     "have carried the Stage 1 rehearsal's label; choosing by folder labels them correctly without a "
                     "change to Stage 2's code (another component) and still refuses a wrong label passed "
                     "explicitly. (b) the rule lived only in run_detect.py, not in the module Stage 2 imports. (c) "
                     "Stage 2's documented order for seeds 5-9 is detect_pairs, detect_labels, detect_standins. "
                     "There the stand-ins' code and fitting rules are fixed by this code manifest, which Stage 2's "
                     "pre-registration pins before those seeds' data exist, so no signal can be adjusted after its "
                     "labels are seen; D12 was a Stage 1 guard. Stage 2 may still score its stand-ins first"),
             evidence=("stage2/run_stage2.py: locked_pair_sets calls D.pair_sets(ctx, world, out=root), "
                       "phase_detect_labels calls LB.run(ctx, world, out=root, extra=...), phase_detect_standins "
                       "calls D.score_signal(name, fns[name], ctx, world, None, pairs_out=root), none with label=; "
                       "prereg_payload.run_order step 2; test_detect.py test_labels_by_folder and "
                       "test_order_rule_in_module")),
        dict(id="C8 (spec v3) launched from the pinned copies; labelled logs with sidecars",
             finding="review of the detect component, finding 7 (minor)",
             change=("launch.sh runs /home/user/bca/runs/shift_explore/detect/code/run_detect.py and is itself run as "
                     "/home/user/bca/runs/shift_explore/detect/code/launch.sh, so the documented command does not "
                     "depend on the scratchpad (common.check_code accepts the copies as the running code). Each log "
                     "starts with the EXPLORATORY label and its .sha256 sidecar is rewritten when each phase ends, "
                     "done or failed. The seed-0 check runs its phases through the same launch.sh (with "
                     "DETECT_CHECK_OUT, DETECT_LOGDIR and DETECT_CHECK_SIGNALS, which only the check sets), so the "
                     "documented command is exercised. common.write_log labels any log written from Python. The "
                     "synthetic tests refuse to run from the pinned copies (their outputs would land in the "
                     "rehearsal's folder)"),
             reason=("the launch command ran the scratchpad sources under /tmp; the logs had no label and no "
                     "sidecar, against the rule that every output has both"),
             evidence="launch.sh v2 (D=<scratchpad>/detect); checks/logs in superseded/v2 hold four unlabelled logs"),
        dict(id="C9 (spec v4) every phase runs on exactly one CPU",
             finding="found by the builder in spec v3's seed-0 check, before any real run",
             change=("run_detect.py and checks_seed0.py refuse unless the process may run on exactly one CPU "
                     "(launch.sh pins one with taskset -c <cpu>); every record also records cpus_visible"),
             reason=("XLA's CPU backend sums the float32 networks' products in an order that depends on how many CPUs "
                     "the process sees. Spec v2's seed-0 check ran unpinned (4 CPUs) and v3's pinned (1 CPU): the "
                     "stored commands differ by at most 4.5e-7 (43,580 float32 entries of the stalling world's "
                     "pairs, T1-T3; T4 and every state identical), so the pair sets' content hash differs, and the "
                     "critic's values at those commands by at most 2.3e-5. Every unsupported and harmful label was "
                     "identical. Readers use the stored commands, but a phase or rebuild under another CPU count "
                     "would not reproduce them bit for bit and could move a label that sits on a threshold. One "
                     "CPU, the launch script's setting, is the rule; it was not chosen on any measured value"),
             evidence=("scratchpad/ood/review_detect_apply/compare_pairs.py (v2 against v3 seed-0 check files) and "
                       "affinity_probe.py: the actor's commands at the 1,500 T1 states equal v3's stored ones with "
                       "taskset -c 1 and v2's with taskset -c 0-3")),
        dict(id="C10 (spec v5) the one-CPU rule also in the module",
             finding="the builder's, before any real run (the same reasoning as C7(b): a rule must sit in the module "
                     "Stage 2 imports)",
             change=("pairs.build, labels.compute, signals.stand_in_signals and signals.score_signal (hence "
                     "detect.pair_sets when it builds, labels.run and detect.score_bca_width) refuse unless the "
                     "process may run on exactly one CPU; the synthetic tests run pinned too"),
             reason=("in spec v4 only run_detect.py and checks_seed0.py held the rule; Stage 2 builds pair sets, "
                     "labels and stand-in scores for seeds 5-9 and scores BCA's width through the module. Stage 2's "
                     "launcher already pins one CPU, so this refuses only a call made outside it"),
             evidence="test_detect.py test_one_cpu_rule"),
    ]
    payload = dict(
        label=C.LABEL, kind="detect-spec", version=VERSION,
        changes_before_any_run=changes,
        changes_rule=("No real run of this component has happened (no seed folder exists under "
                      "runs/shift_explore/detect); only the seed-0 checks ran, into checks/. Each change above is "
                      "recorded with its reason; none was chosen on returns or on any measured value. No room-test "
                      "value, futility file or log was read. The seed-0 numbers quoted come from the review and "
                      "were re-derived only to verify its findings"),
        stage2_pins=("Stage 2's prereg.json (runs/shift_explore/stage2) pins this component's code manifest and "
                     "detect.py by SHA-256. After this version it refuses until Stage 2's write_manifest.py and "
                     "write_prereg.py are rerun and the new version approved (its own rule). This component wrote "
                     "nothing in Stage 2's folder"),
        record_versions=C.versions.__doc__,
        labels_by_folder=C.label_for.__doc__,
        plan=dict(path=str(C.PLAN_PATH), sha256=C.file_sha256(C.PLAN_PATH), sections=["4.4", "5.2"]),
        no_bca=("No BCA band, calibrator, threshold, dose, BCA update or BCA training arm in Stage 1. Nothing from "
                "algorithms/td3_bc_bca.py, calibration/ or learners.bca_* is imported or called by this component "
                "(the gate's learners.py imports them; they are never called). The gate's data loader "
                "(experiments/shift/dataset.py, through acceptance.load_seed) calls calibration/bank.stratified_bank "
                "to regenerate the log's calibration-bank split: a NumPy row sampler, not a BCA signal. The Stage 2 "
                "hook (detect.score_bca_width) accepts a callable and imports nothing of BCA; it is never called in "
                "Stage 1."),
        room_pins=C.room_pins(),
        room_change_A1=("runs/shift_explore/room/spec.json changes_before_any_run A1: the oracle is re-referenced to the "
                        "row's recorded command, [delta*(s, a_i) - delta*(s, a)]_+; A2: check (e) requires a penalty to "
                        "rise past 80%"),
        seeds=list(C.SEEDS), worlds=list(C.WORLDS),
        artifacts=("per world: the kept training rows of the gapped log (acceptance.load_seed(...).prepared('gap')), "
                   "check 2's frozen fixed-controller gapped critic (get_critic(prereg, sd, 'gap', 'stall_aware', "
                   "train=False)), check 2's saved gapped-critic actor ('gapped' / 'no_stall_gapped'), the saved "
                   "cloned actor; the pinned room harness's SeedContext"),
        pair_sets=dict(
            module="pairs.py", sets=list(P.PAIR_SETS), roles={k: list(v) for k, v in P.SET_ROLES.items()},
            T1=f"{P.N_T1} full-throttle-region training states, default_rng([20261015, seed, {P.TAG_T1}]).choice(region rows)",
            T2=f"{P.N_T2} other training states, default_rng([20261015, seed, {P.TAG_T2}]).choice(non-region rows)",
            T3=(f"{P.N_T3_EACH} strong-current states visited by the gapped critic's actor (T3a, tag [{P.TAG_T3}, 0]) "
                f"and {P.N_T3_EACH} by the stall-aware controller (T3b, tag [{P.TAG_T3}, 1]): exact rollouts from the "
                f"first {P.T3_STARTS} starts of the strong-current bank, 100 steps, states rounded to float32"),
            T4=(f"{P.N_T4} held-out recorded rows: calibration split, gap removed, bank rows left out, "
                f"default_rng([20261015, seed, {P.TAG_HELD}]) (the stand-ins' rule)"),
            commands=("command: the world's gapped critic's actor (float32 network at the stored state); cap: "
                      "float32(clip(float64(command), -0.8, 0.8)); recorded: the row's own command; cloned: the "
                      "world's cloned actor; knee (T1): float32(clip(float64(command), -0.81, 0.81))"),
            ref="recorded command at T1, T2, T4; cloned actor's command at T3 (for the re-referenced perfect band)",
            states="float64 of float32 numbers (what the environment returns and every critic and actor sees)",
            plan_roles={k: list(v) for k, v in P.PLAN_ROLES.items()},
            pools=("the ranking and false-flag pools hold the plan's roles only (plan_roles); the knee pairs enter "
                   "only the knee_* measures (D3, change C2)"),
            hashed="pairs.npz (fixed zip timestamps) with a file sidecar; pairs.json records the file and content SHA-256"),
        labels=dict(
            module="labels.py (a separate script; imports only common.py and pairs.py; refuses when signals.py, "
                   "measures.py or detect.py is loaded; never opens a score file; run as its own process)",
            unsupported="any thruster past 80% (thruster_chain.past_knee)",
            harmful=(f"unsupported AND over_pref >= {LB.HARM_OVER} AND true_pref <= -{LB.HARM_TRUE}; over_pref = "
                     "critic_pref - true_pref; critic_pref = Q1(s, a) - Q1(s, cap); true_pref = Q^c(s, a) - Q^c(s, "
                     f"cap), c the stall-aware controller, exact float64, {LB.HORIZON} steps"),
            not_past_80="a command not past 80% equals its cap bit for bit: preferences 0, harmless, not simulated"),
        signals=sig,
        stand_ins=list(S.STANDIN_NAMES), oracles=list(S.ORACLE_NAMES), reported=list(S.REPORTED_NAMES),
        cheap_signal_for_the_seed0_check=S.CHEAP_SIGNAL,
        measures=dict(module="measures.py", doc=M.__doc__, level=f"split-conformal {M.LEVEL:.0%} on T4: the ceil((n + 1) "
                      "0.95)-th smallest T4 score; Stage 2 passes tau instead"),
        join=("measures.py joins one signal's scores with the labels only after both are written and hashed: both "
              "records verified against their sidecars, both files against the file and content SHA-256 their records "
              "name, both naming the same pair sets (content SHA-256), the pair sets verified too; the joined SHA-256 "
              "are recorded in every measures file"),
        order=R.__doc__,
        pooled=dict(keys=list(R.POOLED_KEYS), selectivity="flag_rate_past80 (and per set) in the stalling world minus "
                    "the same in the no-stall world, per seed, at each world's own level",
                    best_standin="the best pooled stalling-world ranking among the five stand-ins (T12 and T3)"),
        bars_carried_forward=R.BARS,
        perfect_band_sanity=dict(R.SANITY, kind="expected, reported; not a mark",
                                 if_not_met=("the script is not used to score BCA until the reason is found and recorded "
                                             "in a new spec version, which runs in a fresh output folder or after "
                                             "archiving every output of this one (change C3)")),
        marks="none in Stage 1 (a rehearsal: every number is reported)",
        stage2_hook=dict(function="detect.score_bca_width(width_fn, ctx, world, out_dir, *, stage2_prereg_sha256, "
                                  "version, pairs_out)", doc=D.score_bca_width.__doc__,
                         contract="scratchpad/ood/stage2/detect_contract.py (Stage 2's), names checked by "
                                  "checks_seed0.py",
                         contract_sha256_when_written=C.file_sha256(contract) if contract.exists() else None,
                         measure_keys=list(D.MEASURE_KEYS)),
        stage2_marks=D.STAGE2_MARKS,
        a1_lesson=D.A1_LESSON,
        deviations=deviations,
        gate_files_sha256=C.gate_files_sha256(tuple(H.GATE_FILES)),
        outputs=dict(root=str(C.OUT), per_seed_world=("seed<k>/<world>/{pairs.npz, pairs.json, scores/<signal>.npz|"
                                                     ".json, signals_done.json, labels.npz, labels.json, "
                                                     "measures/<signal>.json}"), pooled="pooled.json",
                     logs=("logs/seed<k>.log, logs/pooled.log: first line the EXPLORATORY label, a .sha256 "
                           "sidecar rewritten when each phase ends (change C8)"),
                     checks="checks/ (seed-0 check, its logs in checks/logs), timing.json"),
        compute=("measured by checks_seed0.py under this spec (timing.json); the plan's estimate is about 2.5 "
                 "process-hours"),
        run_order=("after the room test's pooled look passes, one process at a time on one free CPU: for each seed "
                   "0-4, sh /home/user/bca/runs/shift_explore/detect/code/launch.sh <seed> <cpu> both pairs signals "
                   "labels measures; then sh /home/user/bca/runs/shift_explore/detect/code/launch.sh all <cpu> both "
                   "pooled"),
        launch="runs/shift_explore/detect/code/launch.sh (the pinned copy; change C8)",
        cpu_rule=C.cpus_visible.__doc__,
    )
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", default=None)
    a = ap.parse_args()
    if C.SPEC_PATH.exists():
        if not a.reason:
            raise SystemExit("refused: a spec exists; pass --reason to write a new version (the old one is kept)")
        old = C.read_json(C.SPEC_PATH)
        hist = C.OUT / "spec_history"
        hist.mkdir(parents=True, exist_ok=True)
        v = old.get("version", 0)
        shutil.copy2(C.SPEC_PATH, hist / f"spec_v{v}.json")
        shutil.copy2(C.SPEC_PATH.with_name("spec.json.sha256"), hist / f"spec_v{v}.json.sha256")
    payload = spec_payload()
    if a.reason:
        payload["reason"] = a.reason
        payload["previous_spec"] = dict(version=v, sha256=C.file_sha256(hist / f"spec_v{v}.json"),
                                        archived=f"spec_history/spec_v{v}.json",
                                        superseded_outputs=f"superseded/v{v} (its seed-0 check and timing)")
    digest = C.write_json(C.SPEC_PATH, payload)
    print(f"spec written: {C.SPEC_PATH} sha256 {digest}")


if __name__ == "__main__":
    main()
