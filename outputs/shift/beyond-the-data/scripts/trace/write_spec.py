"""EXPLORATORY: write runs/shift_explore/trace/spec.json (+ .sha256) before any run of the trace test bed.

The spec may be rewritten only while no job has been trained (no lock.json, values.json, signals.json or pooled.json
exists). A rewrite needs --reason, keeps the previous spec in spec_history/ and moves what was computed under it (the
seed-0 checks, the timing record, the build records, the code manifest, its history and its copies) to
superseded/v<k>/.

Spec v3 (POST HOC, written after the spec-v2 results were seen): python write_spec.py --post-hoc-v3 --reason TEXT.
It keeps every key of spec v2 exactly as it is (only 'version' and 'previous_spec' change; spec v1's record moves to
'spec_history') and ADDS the key 'post_hoc_extension': the shares 1%, 3%, 5% and 10%, labelled "post hoc: chosen after
seeing the 0.5% and 2% results", with the reason. Questions and marks stay unchanged and nothing is chosen on the new
results. Spec v2 is kept in spec_history/spec_v2.json; no spec-v2 output is moved or changed. It refuses once any
post-hoc job has been trained.

Usage: python write_spec.py [--reason TEXT]
       python write_spec.py --post-hoc-v3 --reason TEXT
"""
import argparse
import copy
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracebed as T  # noqa: E402
import signals as S  # noqa: E402

VERSION = 1
UNDER_OLD_SPEC = ("checks_seed0.json", "timing.json", T.MANIFEST_NAME)
REVIEW_EVIDENCE = "scratchpad/review_trace/ (reviewer) and scratchpad/ood/trace/verify/ (independent checks)"

CHANGES_BEFORE_ANY_RUN = [
    dict(id="R1 residual evidence in the critic's training form",
         finding="review, major: the one-step error used the noiseless next command",
         change=("The one-step error behind residual evidence (the primary signal for Q2) is now the critic's own "
                 "training form: the target averages min_h Q(s', clip(c(s') + noise, -1, 1)) over 32 fixed draws of "
                 "td3_bc_update's training noise (normal, standard deviation 0.2, clipped at 0.5), as BCA's own "
                 "residual and the critic's training target do; draw d of log row i is row i of "
                 "default_rng([20261015, seed, 94, d + 1]).standard_normal((log rows, 4)), so a row gets the same "
                 "draws in both worlds and at every share. The noiseless form (spec v1) is kept as the reported "
                 "signal residual_evidence_noiseless. The gap between the two, 0.98 J(s'), is reported at the "
                 "training rows by region and command (J-by-region diagnostic, both worlds, beside the readout and "
                 "in the pooled look) and at every pair (neighbour_offset). On T1 and T1c every signal is also read "
                 "relative to the row's recorded command at the same state (score(s, a) - score(s, a_i), the A1 "
                 "form), reported; the flag-based Q2 is unchanged."),
         reason=("The offset is not a constant that the T4 flag level cancels. With noise draws at seed 0 (32 draws, "
                 "seed-0 check h and verify/v1_offset_pairs.py; the reviewer's 16 draws agree), J at the next states "
                 "of the gapped log's training rows is, region rows against the rest: stalling world, full-log critic "
                 "(has learned the stall) +0.270 / +0.132; no-stall world, full-log critic +0.124 / +0.113; gapped "
                 "critics +0.133 / +0.113 (stalling) and +0.128 / +0.113 (no-stall). In the region the stall-aware "
                 "controller sits at its 80% cap, so the training noise reaches past-80% commands, which a critic "
                 "that has learned the stall drops. A trace critic that learns the stall would therefore get up to "
                 "about +0.14 more offset at region rows in the stalling world only, lowering residual evidence there: "
                 "against the separation Q2 looks for, and about the size of the trace signal the plan cites (-0.10 "
                 "against +0.12). T4 is 84% outside the region (region share 15.7% stalling, 15.4% no-stall), so its "
                 "level cannot remove it. In the training form a well-fitted row has an error near 0 whatever the "
                 "critic's shape at s'."),
         evidence=REVIEW_EVIDENCE + "; seed-0 check h", status="applied before any training run"),
    dict(id="R2 every computing phase pinned to one CPU",
         finding="review, minor: the actor-only call reproduces check 2 only when pinned",
         change=("run_job refuses every job phase (build, train, eval, signals, reference) and checks_seed0.py "
                 "refuses to run unless the process may use exactly one CPU (os.sched_getaffinity); the lock, values "
                 "and signals record the CPU. The train phase's actor-only call is one function, run_job.train_actors, "
                 "and seed-0 check f runs it pinned on the share-0 log (trace slot = the gate's gapped critic) and "
                 "requires check 2's saved gapped-critic and full-log-critic actors bit for bit. Seed-0 check g "
                 "requires the trace critic's protocol (learners.fixed_critic) to reproduce the gate's gapped "
                 "critic's initial parameters and first 5,000-update chunk loss exactly."),
         reason=("Verified: run_job's three-arm call for 1,000 steps gives identical actors pinned to 1 or 2 CPUs and "
                 "different actors on 4 CPUs (verify/v2_threads.py); the reviewer found the 20,000-step actors "
                 "unpinned off by up to 0.26 in a parameter and 0.12 in a command, and pinned equal to check 2's."),
         evidence=REVIEW_EVIDENCE + "; seed-0 checks f and g", status="applied before any training run"),
    dict(id="R3 the two worlds keep the same trace rows",
         finding="review, minor: each world drew its trace rows from its own permutation",
         change=("One permutation per seed, default_rng([20261015, seed, 91]), over the candidates both worlds have "
                 "(the same log positions), used in both worlds: the two worlds add back the same episodes and time "
                 "steps (build check same_rows_as_the_other_world). The count is floor(share x N + 0.5) of the N "
                 "shared candidates. The few world-specific candidates are never added back and are listed in "
                 "trace_rows.json. The bank: the same over its shared candidates, default_rng([20261015, seed, 91, "
                 "1]). T1's states are paired the same way: drawn by default_rng([20261015, seed, 92]) from the "
                 "region rows both worlds keep. Added, reported: T1c, the T1 states at a command common to both "
                 "worlds (check 1's full-throttle pattern and its 80% cap), whose past-80% pairs are not selected by "
                 "what each world's critic taught its actor."),
         reason=("Verified for seeds 0-2 (verify/v1_offset_pairs.py): shared candidates 6,603 of 6,609 / 6,606, "
                 "6,747 of 6,751 / 6,762 and 6,459 of 6,462 / 6,469, with identical recorded commands at every "
                 "shared position and the same episodes; yet the spec-v1 builds kept 0 common rows of 32-34 at 0.5% "
                 "and 2-5 of 129-135 at 2%. The states at a position differ (median largest component 0.27), so "
                 "pairing removes the random placement of the few kept rows from the difference Q2 reads, not every "
                 "difference between the worlds."),
         evidence=REVIEW_EVIDENCE, status="applied before any training run"),
    dict(id="R4 the launch survives a stopped job and runs seeds side by side",
         finding="review, minor: a stopped job aborted the launch and every rerun; ';' ran the seeds in sequence",
         change=("launch.sh skips a job whose stop.json exists (logged); a failed job no longer ends the launch: it "
                 "goes on with the next job and exits 1 at the end. Each launch logs to its own file "
                 "(seed<k>[_<world>][_<share>].log). The launch instructions start the seeds in the background "
                 "(nohup ... &), and 'sh launch.sh all <cpu A> <cpu B>' runs seeds 0 and 1 side by side, then seed 2 "
                 "split by world, then the pooled look."),
         reason="verified by reading launch.sh and run_job.phase_build; the new behaviour is unit-tested with a stub "
                "interpreter (test_trace.py)",
         evidence=REVIEW_EVIDENCE, status="applied before any training run"),
]


def run_outputs():
    found = []
    for name in ("lock.json", "values.json", "signals.json"):
        found += list(T.OUT.glob(f"*/seed*/*/{name}"))
    if (T.OUT / "pooled.json").exists():
        found.append(T.OUT / "pooled.json")
    return found


def spec_payload():
    learn = T.L.LearnSettings()
    return dict(
        label=T.LABEL, kind="trace-test-bed-spec", version=VERSION,
        plan=dict(path=str(T.PLAN_PATH), sha256=T.file_sha256(T.PLAN_PATH), sections=["4.4", "4.6", "5.2"]),
        purpose=(
            "The harm claim, 'BCA identifies the harm itself, not just novelty', needs data that carry a trace of the "
            "harm. The gate's gapped log keeps only 34-59 rows past 80% throttle (about 0.1%), so nothing in it can "
            "tell a harmful unsupported command from a merely unrecorded one. A trace log keeps a small, seeded, random "
            "share of the rows the gate's removal took out past 80%: 0.5% and 2% of them. This test bed builds those "
            "logs in both worlds for gate seeds 0-2, retrains the fixed-controller critic and check 2's actors on them, "
            "and asks whether the harm survives and whether a data-only residual signal can then tell the two worlds "
            "apart. It runs no part of BCA. If both answers are yes, these logs become the test bed for a later harm "
            "claim under its own pre-registration (plan 4.6)."),
        user_decision="do both, start with the flags claim now (this job builds the test bed for the harm claim)",
        changes_before_any_run=CHANGES_BEFORE_ANY_RUN,
        no_bca=("No BCA band fit, calibrator fit, threshold, dose readout, BCA update or BCA training arm runs. Nothing "
                "from algorithms/td3_bc_bca.py or learners.bca_* is called (learners.py imports them; they are never "
                "called). From calibration/ only calibration/bank.stratified_bank runs: the gate's data loader "
                "(experiments/shift/dataset.py, through acceptance.load_seed) calls it to regenerate the log's "
                "calibration-bank split. It is a NumPy row sampler, not a BCA signal. Every detection signal is a "
                "data-only stand-in. Any BCA run on these logs needs the user's typed approval (acceptance.stage_approve "
                "pattern: the first 16 characters of a report's SHA-256, typed at a terminal) of its own hashed "
                "pre-registration."),
        writes=("only runs/shift_explore/trace (tracebed.guard_out refuses anything else); never experiments/, "
                "runs/shift/ or runs/shift_explore/room/; gate artifacts are read through the gate's own verified "
                "loaders; room-test outcomes are not read"),
        gate_prereg_sha256=T.file_sha256(T.ROOT / "runs" / "shift" / "preregistration.json"),
        gate_files_sha256=T.gate_file_hashes(),
        room_spec=dict(path=str(T.ROOM_SPEC_PATH), sha256=T.file_sha256(T.ROOM_SPEC_PATH),
                       lesson=("change A1: a penalty must be read relative to the recorded command at the same state; "
                               "the noiseless one-step error carries an offset (about +0.15 at recorded commands) "
                               "that depends on the state and on the command, not a constant. This test bed trains no "
                               "penalty. Its residual evidence uses the critic's training form (noise-averaged "
                               "target), so the offset does not enter it; the offset is reported by region in both "
                               "worlds; and on T1 and T1c every score is also read relative to the row's recorded "
                               "command (reported). Any later penalty on these logs must use the re-referenced form "
                               "[delta*(s, a_i) - delta*(s, a)]_+.")),
        seeds=list(T.SEEDS), worlds=list(T.WORLDS), shares=list(T.V2_SHARES), reference_share=T.REFERENCE_SHARE,
        alpha=T.ALPHA,
        trace_logs=dict(
            removal=("a new removal computed here, outside the hashed package: the gate's removal ('gap', "
                     "shift.make_removal, unchanged) of the seed and world, then a seeded random subset of its removed "
                     "training rows that are past 80% on any thruster (thruster_chain.past_knee on the stored float32 "
                     "command) is put back"),
            candidates=("per world, the gate's removed training rows past 80% on any thruster (seed 0: 6,609 stalling, "
                        "6,606 no-stall); the selection pool is the candidates both worlds have, the same log "
                        "positions (seed 0: 6,603; seed 1: 6,747; seed 2: 6,459)"),
            count="floor(share x N + 0.5) of the N shared candidates (seed 0: 33 rows at 0.5% and 132 rows at 2%)",
            selection=("the first rows of one fixed permutation default_rng([20261015, seed, 91]) of the shared "
                       "candidates, the same in both worlds (paired: the two worlds add back the same episodes and "
                       "time steps), so the 0.5% rows are a subset of the 2% rows. World-specific candidates (seed 0: "
                       "6 stalling, 3 no-stall) are never added back; they are listed in trace_rows.json"),
            bank=("the calibration bank gets the same rule with default_rng([20261015, seed, 91, 1]) over its shared "
                  "removed rows past 80% (seed 0: 1 row at 0.5%, 3 rows at 2%). Nothing that runs here reads the "
                  "bank; it is kept so that a later BCA test on these logs sees the same trace in its bank"),
            seed_tags_note=("numpy treats a trailing 0 seed tag as absent, so no trace-selection or noise tag list ends "
                            "in 0; the T3 tags keep their spec-v1 form, whose four streams are distinct"),
            evaluation_split="never touched",
            normalisation=("the gate's: the gapped log's kept training rows (sd.norm()), as the gate normalises every "
                           "critic of a seed; the reused full-log and gapped critics read the same inputs"),
            record=("trace_rows.json per job: the exact added training rows and bank rows (log row indices), their "
                    "SHA-256 (dataset.sha256_array), the candidates' and shared candidates' SHA-256, the world-specific "
                    "candidates (never added), the share of added rows whose recorded command is the same in both "
                    "worlds, the kept rows' SHA-256, row counts, the past-80% share before and after, the prepared "
                    "data's SHA-256 and the build checks"),
            build_checks=["the gate's removal matches its data record", "the count rule",
                          "added rows unique, among the gate's removed training rows, all past 80%",
                          "added bank rows among the gate's removed bank rows, all past 80%",
                          "kept rows = the gate's kept rows plus the added rows",
                          "bank = the gate's bank plus the added bank rows",
                          "training rows only; evaluation split untouched",
                          "share 0 (reference): the prepared data equal the gate's gapped data (SHA-256)",
                          "the same added rows, bank rows and shared candidates as the other world's build"]),
        training=dict(
            pinning=("every computing phase (build, train, eval, signals, reference) and the seed-0 checks refuse "
                     "unless the process may use exactly one CPU; launch.sh pins with taskset. Results depend on the "
                     "thread count: pinned, the train phase's actor-only call reproduces check 2's actors bit for bit "
                     "(seed-0 check f); on 4 CPUs it does not"),
            settings=learn.as_dict(),
            settings_note="the pre-registered learner settings (prereg.config.learn), unchanged",
            how=("the gate's own functions, imported read-only and called unchanged (not re-typed): the protocol and "
                 "its settings are therefore the gate's by construction"),
            clone=("learners.clone on the trace log's kept training rows: 20,000 steps, mean squared error, Adam 3e-4, "
                   "batch 256, rows default_rng([20261015, seed, 34]), from the plain TD3+BC run's initial actor"),
            critic=("learners.fixed_critic(prepared, 'stall_aware', settings, seed): the stall-aware controller's "
                    "fixed-controller critic, 100,000 updates, TD3+BC's critic update with the controller as the "
                    "target's next command, the seed's own initialisation and random stream (as the gate's critics)"),
            actor_only=("check 2's actor-only protocol, run_job.train_actors -> learners.ActorOnly(prepared, clone, "
                        "settings, seed, alpha=5).run(arms, critics, weighted=False), pinned to one CPU: 20,000 steps "
                        "in 4,000-step chunks, rows "
                        "default_rng([20261015, seed, 35]) over the trace log's rows, from the trace log's cloned actor, "
                        "three critics side by side (critic swap): 'trace' (this log's critic), 'full' (the gate's "
                        "full-log critic) and 'gapped' (the gate's gapped critic; reported control)")),
        reuse=dict(
            critics=("the gate's full-log and gapped fixed-controller critics of the stall-aware controller, both "
                     "worlds, read with acceptance.get_critic(..., train=False) (SHA-256 checked); identical for every "
                     "share, so never retrained"),
            not_reused=("the cloned actor and the full-log critic's actor are retrained on each trace log: they are not "
                        "identical to the gate's (the clone's rows and the actor-only row stream depend on the log's "
                        "rows)"),
            reference=("share 0 is the gate's own gapped log: nothing is trained. Check 2's saved cloned, gapped-critic "
                       "and full-log-critic actors and their saved points, check 1's saved pull past 80%, the gate's "
                       "gapped critic and saved clone are read back (SHA-256 checked); only the detection readout is "
                       "computed")),
        lock=("after training, lock.json holds the SHA-256 of the trace rows record, the clone, the trace critic, the "
              "reused critics, every actor and actors.msgpack, with the spec's and the code manifest's SHA-256. No "
              "return and no detection signal is computed before it exists; eval and signals verify it first"),
        evaluation=("acceptance._evaluate (learners.evaluate_actor): exact 100-step returns on the full 1,000-start "
                    "banks of the four situations (recorded conditions, strong current, big waves, big kicks), points "
                    "against the gate's anchors of the world (0 = recording controller, 100 = stall-aware controller), "
                    "the same actor capped at 80%, the share of steps past 80%. Actors: the trace log's cloned actor and "
                    "the trace, full and gapped arms. harm = points(full) - points(trace), per situation"),
        overrating_readout=("the pull past 80% of the trace critic on check 1's frozen region pool (3,000 full-throttle "
                            "region training states of the gapped log, all kept in every trace log): "
                            "[Q1(s, c_hi) - Q1(s, c_cap)] + price(s), price = Q^c(s, c_cap) - Q^c(s, c_hi) the exact "
                            "value check 1 saved (rows.npz, SHA-256 checked), c the stall-aware controller; equal bit "
                            "for bit to check 1's own 'pull|region' for the same critic (seed-0 check). Reported beside "
                            "the gate's gapped and full-log critics' saved pulls"),
        detection=dict(
            definitions=S.__doc__,
            signals=list(S.SIGNALS), primary_signal=S.PRIMARY_SIGNAL, rival_signal=S.RIVAL_SIGNAL,
            reported_extras=list(S.EXTRAS),
            sets=list(S.SETS), primary_set=S.PRIMARY_SET, common_command_set=S.COMMON_SET,
            rereferenced_sets=list(S.REREFERENCED_SETS), neighbours=S.K_NEIGHBOURS, level=S.LEVEL,
            range_quantiles=list(S.RANGE_QUANTILES),
            one_step_error=dict(form="training form (noise-averaged target); module docstring", draws=S.DRAWS,
                                noise=dict(policy_noise=learn.policy_noise, noise_clip=learn.noise_clip,
                                           command_clip=1.0, tags="[20261015, seed, 94, draw + 1], per log row"),
                                heads="min over the two heads on both sides", gamma=learn.gamma,
                                critic=("the frozen critic's final parameters on both sides (training used a slowly "
                                        "moving target copy for the target)")),
            offset_diagnostic=("J(s') = min_h Q(s', c(s')) - mean over the draws of min_h Q(s', clip(c(s') + noise)) "
                               "at the training rows' next states, summarised for region rows, the rest and past-80% "
                               "rows; neighbour_offset = the mean of 0.98 (1 - done) J over a pair's 10 nearest rows. "
                               "Reported in every readout and per seed and world in the pooled look"),
            sizes=dict(T1=S.T1_STATES, T3_starts=S.T3_STARTS, T3=S.T3_STATES, T4=S.T4_ROWS),
            actor=("the trace critic's actor of the log (share 0: check 2's gapped-critic actor); the critic, clone "
                   "and training rows are the log's own"),
            selectivity=("flag rate at past-80% pairs in the stalling world minus the same in the no-stall world, per "
                         "seed, share, set and signal; no value when either world has fewer than "
                         f"{T.MIN_PAST80_PAIRS} past-80% pairs in the set"),
            detect_component=("the Stage 1 detect component (scratchpad/ood/detect/) did not exist when this spec was "
                              "written, so these are this test bed's own implementations of the plan's definitions. If "
                              "its locked signals become available they may be added as reported extras under a new "
                              "spec version; they never replace the primary signal or set")),
        questions=dict(
            Q1=("Does the trace critic still overrate pushing past 80%, with harm of at least 50 points in strong "
                "current? Yes when the mean over seeds 0-2 of harm = points(full-log critic's actor) - points(trace "
                "critic's actor) in strong current, stalling world, is at least 50 (every seed needed). The pull past "
                "80% is reported beside it"),
            Q2=("Does residual evidence separate the two worlds with selectivity of at least 0.2, where the range hinge "
                "cannot? Yes when, on T1, the mean over seeds 0-2 of residual evidence's selectivity (training-form "
                "one-step error) is at least 0.2 and the range hinge's is below 0.2 (every seed needed). Reported, "
                "not judged: the other stand-ins (the noiseless residual evidence among them), the other sets (T1c, "
                "the common command, among them), the re-referenced differences on T1 and T1c, and the J-by-region "
                "diagnostic of both worlds"),
            verdict=("per share: usable for a later harm claim when Q1 and Q2 are both yes. Exploratory: seeds 0-2 "
                     "only; any claim needs its own pre-registration and fresh seeds"),
            missing=("a stopped or failed job, or a world with too few past-80% pairs, gives no value for that seed, "
                     "and a mean that needs it gives no answer (the question is then not answered yes)"),
            reference=("share 0 (the gate's gapped log) gets the same numbers, reported only")),
        outputs=dict(
            root=str(T.OUT),
            per_job="<world>/seed<k>/share_<p>pct/: trace_rows.json, clone.json + clone.msgpack, critic.json + "
                    "critic.msgpack, actors.msgpack, lock.json, values.json, signals.json + signals.npz (stop.json "
                    "when a check fails); every JSON has a .sha256 sidecar",
            reference="<world>/seed<k>/share_0pct/: trace_rows.json, values.json, signals.json + signals.npz",
            pooled="pooled.json", checks="checks_seed0.json, timing.json", logs="logs/"),
        run_order=("1. seed-0 checks (checks_seed0.py, pinned: build the seed-0 logs, verify the counts, the pairing "
                   "and the past-80% share, reproduce check 1's pull, the gate's clone, check 2's actors and the gate "
                   "critic's first chunk at share 0, the J-by-region diagnostic on the gate's critics, timing; no "
                   "return, no signal at an actor command). 2. launch.sh per seed: for each world the reference job, "
                   "then for each share build, train (lock), eval, signals. 3. launch.sh pooled"),
        launch=dict(
            script=str(T.HERE / "launch.sh"),
            usage=("sh launch.sh <seed> <cpu> [<world>|all] [<share>|all]; sh launch.sh pooled <cpu>; "
                   "sh launch.sh all <cpu A> <cpu B>"),
            priority="nice 15 (NICE=... to change), one process pinned to one given CPU",
            behaviour=("a job with stop.json is skipped (logged); a failed job does not end the launch; rerunning "
                       "resumes and skips finished phases; launch.sh runs in the foreground, so two seeds run side by "
                       "side only when each is started in the background"),
            suggested=["nohup sh launch.sh all <cpu A> <cpu B> >/dev/null 2>&1 &   (everything, in order)",
                       "or: nohup sh launch.sh 0 <cpu A> >/dev/null 2>&1 &  and  nohup sh launch.sh 1 <cpu B> "
                       ">/dev/null 2>&1 &; when both have ended, the same for 'sh launch.sh 2 <cpu A> stall' and "
                       "'sh launch.sh 2 <cpu B> no_stall'; then sh launch.sh pooled <cpu A>"]),
        compute=dict(
            plan_estimate_process_hours=30,
            gate_timings_idle_machine=dict(critic_100k_updates_s=403, clone_20k_steps_s=32, check2_seed_s=2279),
            projection="timing.json, from the seed-0 timing check (the machine is shared, so times are inflated)"),
        seed0_checks=["(a) build counts and checks for both worlds and all three shares; the 0.5% rows are a subset "
                      "of the 2% rows; both worlds add back the same rows; share 0 equals the gate's gapped log "
                      "(removal record and prepared data SHA-256)",
                      "(a2) T1: 1,500 rows from the region rows both worlds keep; the T1c pattern is the same in both "
                      "worlds and past 80% (states and commands only, no score)",
                      "(b) region pull of the gate's gapped and full-log critics equals check 1's saved rows bit for bit",
                      "(c) the gate's clone is reproduced at share 0 by learners.clone (parameter SHA-256)",
                      "(f) pinned, the train phase's actor-only call (run_job.train_actors) on the share-0 log with the "
                      "gate's gapped critic in the trace slot reproduces check 2's gapped-critic and full-log-critic "
                      "actors (parameter SHA-256)",
                      "(g) the trace critic's protocol reproduces the gate's gapped critic's initial parameters and "
                      "first 5,000-update chunk loss exactly",
                      "(h) reported: the J-by-region diagnostic on the gate's gapped and full-log critics, both worlds",
                      "(d) timing: one situation's exact evaluation of an untrained actor, the stand-ins' set-up with "
                      "the noise draws, T4 scores, a rollout and a 1,500-pair score call; with (c), (f) and (g), the "
                      "projected cost; all results discarded"],
        deviations=[
            "The gate's functions are called unchanged instead of a re-typed copy of the critic and actor protocols "
            "(plan 4.6: 'scripts import the gate's code read-only'). The protocol and its settings are the gate's by "
            "construction.",
            "Normalisation is the gate's (the gapped log's kept rows), not recomputed on each trace log: the gate uses "
            "it for every critic of a seed, and the reused critics need it. The trace rows change it by a few parts "
            "in ten thousand at most.",
            "The cloned actor and the full-log critic's actor are retrained on each trace log; only the critics are "
            "reused. They cannot be identical to the gate's, because the clone's rows and the actor row stream depend "
            "on the log's rows. Share 0 reuses everything from the gate.",
            "Added reported arm: the gate's gapped critic's actor trained on the trace log. trace - gapped separates "
            "the effect of the critic's data from the change of start actor and row stream.",
            "The 0.5% rows are a subset of the 2% rows (one permutation per seed and world), so the two shares form a "
            "dose ladder.",
            "The calibration bank gets the same add-back rule. Nothing here reads it; a later BCA test would.",
            "The plan leaves the pair sets and the primary set of the selectivity open. They are fixed here: T1 "
            "(region training states, where the actor learns the belief and where a training-time penalty acts) is "
            "primary; T3a and T3b (strong current) are reported.",
            "Flag levels come from 5,000 held-out evaluation-split rows outside the gap ball (the fallback "
            "diagnostic's check rows), so the levels describe recorded data like the training data.",
            "Residual evidence's one-step error uses the minimum over the critic's two heads on both sides and the "
            "critic's training form of the target: the controller's next command plus the training noise, averaged "
            "over 32 fixed draws, with the recorded reward and next state (spec v2, change R1). The noiseless form "
            "(spec v1, the room test's form) is reported as residual_evidence_noiseless; the offset between them is "
            "reported by region in both worlds.",
            "The trace rows and the T1 states are the same log positions in both worlds (spec v2, change R3); the "
            "common-command set T1c and the re-referenced scores on T1 and T1c are reported additions.",
            "Every computing phase runs pinned to one CPU (spec v2, change R2).",
            "The questions are judged on the mean over seeds 0-2; each seed is reported.",
            "The trace critic's overrating is read on check 1's frozen region pool with check 1's saved exact values "
            "(no new simulation).",
            "Harm in the no-stall world (where pushing past 80% helps) is reported, not judged.",
            "The detect component's signals were not ready; own implementations of residual evidence, distance to "
            "data, the range hinge, behaviour density and two-head disagreement are used.",
            "Compute is projected from the measured seed-0 timing instead of the plan's 30 process-hours."],
    )


# ------------------------------------------------------------------------------------------------ spec v3 (post hoc)

V3_EXTENSION_KEY = "post_hoc_extension"
V3_REASON_DEFAULT = None


def post_hoc_outputs():
    """Trained outputs of a post-hoc job (any lock.json, values.json, signals.json or stop.json in a post-hoc share
    folder) and the spec-v3 pooled look."""
    found = []
    for share in T.POST_HOC_SHARES:
        for name in ("lock.json", "values.json", "signals.json", "stop.json", "clone.json", "critic.json"):
            found += list(T.OUT.glob(f"*/seed*/{T.SHARE_DIR[share]}/{name}"))
    if (T.OUT / T.POOLED_V3_NAME).exists():
        found.append(T.OUT / T.POOLED_V3_NAME)
    return found


def _r(x, nd):
    return None if x is None else round(float(x), nd)


def seen_results(pooled):
    """What was seen before the post-hoc shares were chosen: spec v2's pooled look, per share."""
    out = {}
    for share in T.FROZEN_SHARES:
        o = pooled["shares"][T.SHARE_DIR[share]]
        t1 = o["selectivity_mean"][S.PRIMARY_SET]
        out[T.SHARE_DIR[share]] = dict(
            harm_strong_current_mean=o["harm_strong_current_mean"],
            harm_strong_current_per_seed=[o["per_seed"][str(r)]["harm_strong_current"] for r in T.SEEDS],
            selectivity_T1_mean={k: t1[k] for k in S.SIGNALS},
            Q1_harm_kept=o["Q1_harm_kept"], Q2_residual_evidence_separates=o["Q2_residual_evidence_separates"],
            usable_for_a_harm_claim=o["usable_for_a_harm_claim"])
    return out


def v2_counts():
    """Per seed, from the saved spec-v2 build records (stalling world): the shared candidates N, the shared bank
    candidates and the rows every share adds back (floor(share x N + 0.5))."""
    out = {}
    for rep in T.SEEDS:
        rec = T.read_json(T.job_dir("stall", rep, T.V2_SHARES[0]) / "trace_rows.json")
        n, nb = rec["shared_candidates"], rec["shared_bank_candidates"]
        out[f"seed{rep}"] = dict(
            shared_candidates=n, shared_bank_candidates=nb,
            added_training_rows={T.SHARE_DIR[s]: T.trace_count(s, n) for s in T.SHARES},
            added_bank_rows={T.SHARE_DIR[s]: T.trace_count(s, nb) for s in T.SHARES},
            gate_past80_kept_rows=rec["gate_past80_kept_rows"], gate_kept_training_rows=rec["gate_kept_training_rows"])
    return out


def v2_reproduced(v2):
    """Does this code's spec-v2 payload (spec_payload) still equal spec v2 on disk, apart from 'version' and
    'previous_spec'? Reported in spec v3 (a check that the spec-v2 parts of the code were not changed)."""
    now = T.jsonable(spec_payload())
    keys = sorted(set(now) | set(v2))
    differ = [k for k in keys if k not in ("version", "previous_spec")
              and T.A.canonical(now.get(k)) != T.A.canonical(v2.get(k))]
    return dict(equal=not differ, differing_keys=differ)


def post_hoc_section(v2, v2_sha, reason):
    pooled_path = T.OUT / T.POOLED_V2_NAME
    pooled = T.read_json(pooled_path)
    if pooled is None or pooled["spec_sha256"] != v2_sha:
        raise SystemExit(f"refused: {pooled_path} is missing or was not written under spec v2")
    seen = seen_results(pooled)
    a, b = seen["share_0.5pct"], seen["share_2pct"]
    pc = lambda s: f"{100 * s:g}%"  # noqa: E731
    frozen_files = {name: T.file_sha256(T.OUT / name) for name in T.FROZEN_V2_FILES}
    return dict(
        label=T.POST_HOC_LABEL,
        status=("EXPLORATORY and POST HOC: added after the spec-v2 results (pooled.json) were seen. It does not change "
                "spec v2, its questions, its marks or its results; it adds four shares with the same rule and "
                "protocol, and a map over every share."),
        reason=reason,
        seen_before_choosing=dict(
            pooled_v2=dict(path=str(pooled_path), sha256=T.file_sha256(pooled_path)),
            summary=(f"0.5% kept: harm {a['harm_strong_current_mean']:.1f} points in strong current (Q1 yes), residual "
                     f"evidence selectivity on T1 {a['selectivity_T1_mean']['residual_evidence']:.3f} (Q2 no), not "
                     f"usable. 2% kept: harm {b['harm_strong_current_mean']:.1f} points (Q1 yes), selectivity "
                     f"{b['selectivity_T1_mean']['residual_evidence']:.3f} (Q2 no), not usable. Neither planned share "
                     "is usable for a harm claim."),
            per_share=seen),
        user_decision="do both (both claims)",
        shares=list(T.POST_HOC_SHARES),
        share_dirs={T.SHARE_DIR[s]: s for s in T.POST_HOC_SHARES},
        why_these_shares=("a fixed grid, set before any of its results exist and not tuned to them: 1% between the two "
                          "planned shares; 3%, 5% and 10% above them, so that with share 0, 0.5% and 2% the map covers "
                          "0 to 10%. At 10% a trace log adds back about 650 of the about 6,600 removed past-80% rows "
                          "per seed, against 34-59 past-80% rows in the gate's gapped log"),
        question_of_the_sweep=("does any share from 0 to 10% leave BOTH a large harm (Q1: at least 50 points in strong "
                               "current, stalling world) AND a detectable trace (Q2: residual evidence selectivity of "
                               "at least 0.2 on T1, range hinge below 0.2), or does the critic learn the stall (the "
                               "harm falls below 50) before the trace becomes detectable?"),
        unchanged=[
            "the questions Q1 and Q2, their marks (50 points; 0.2) and the verdict rule ('usable' when both are yes), "
            "judged per share exactly as in spec v2",
            "the signals, pair sets, flag levels, selectivity rule and the reported extras (signals.py is unchanged)",
            "the trace rule: one permutation default_rng([20261015, seed, 91]) per seed over the candidates both worlds "
            "share, the same rows in both worlds, count floor(share x N + 0.5); so the rows of every smaller share "
            "are inside every larger one (0.5% < 1% < 2% < 3% < 5% < 10%) and the spec-v2 rows are reproduced exactly; "
            "the bank rule likewise",
            "the critic and actor protocol (learners.clone, learners.fixed_critic with 100,000 updates, check 2's "
            "actor-only protocol with three arms, alpha 5), the evaluation, the overrating readout, seeds 0-2, both "
            "worlds, the gate's normalisation, pinning to one CPU, nice 15",
            "no BCA: no band, calibrator, threshold, dose, BCA update or BCA training arm"],
        frozen_v2=dict(
            spec_sha256=v2_sha, shares=list(T.FROZEN_SHARES),
            pooled_sha256=T.file_sha256(pooled_path), files_sha256=frozen_files,
            code_manifest_sha256=pooled["code_manifest_sha256"],
            rule=("the spec-v2 jobs (share 0, 0.5%, 2%) are complete and kept as they are: no phase runs on them under "
                  "spec v3 (run_job refuses), tracebed.guard_out refuses any write into their folders or to "
                  "pooled.json, checks_seed0.json and timing.json, and the pooled look reads them with spec v2's "
                  "SHA-256")),
        nothing_chosen_on_new_results=(
            "No question, mark, signal, set, seed or share is chosen or changed on the new results. The map reports "
            "every share from 0 to 10%; a share marked usable in it describes seeds 0-2, it is not a choice. Any harm "
            "claim on any share needs its own pre-registration with fresh seeds, and must say that the share was "
            "found post hoc."),
        counts_per_seed=v2_counts(),
        records=("every record of a post-hoc job (trace_rows.json, lock.json, values.json, signals.json, stop.json) "
                 f"carries share_label '{T.POST_HOC_LABEL}'"),
        pooled=("pooled_v3.json: spec v2's pooled look for every share (0, 0.5%, 1%, 2%, 3%, 5%, 10%) side by side; its "
                "entries for share 0, 0.5% and 2% must equal pooled.json's exactly, else it refuses; and a 'map' over "
                "every share, smallest first: origin (reference, planned in spec v2, post hoc in spec v3), the "
                "strong-current harm (mean and per seed), the T1 selectivity of every signal (mean), and Q1, Q2 and "
                "usable as yes or no. pooled.json (spec v2) is kept as it is"),
        checks=("checks_v3.py, pinned to one CPU at nice 15: (a) build the seed-0 logs of the four post-hoc shares in "
                "both worlds (run_job.phase_build writes their official trace_rows.json) and verify the row counts, "
                "the build checks, that both worlds add back the same rows, and the nesting of all six shares, with "
                "the spec-v2 rows rebuilt and compared with their saved records; (b) one smoke step: the whole job "
                "chain (build, train, eval, signals) on the 10% log of the stalling world with tiny settings, written "
                "to a scratch folder and discarded, no score printed; the refusals of the spec-v2 shares; the pooled "
                "look and the map on the saved spec-v2 jobs reproduce pooled.json; (c) timing on the 0.5% and 10% "
                "logs and the projected cost -> checks_v3_seed0.json, timing_v3.json"),
        jobs=dict(trace_jobs=len(T.POST_HOC_SHARES) * len(T.WORLDS) * len(T.SEEDS), reference_jobs=0,
                  note="no new reference job: share 0 is a spec-v2 job"),
        launch=dict(
            script=str(T.HERE / "launch.sh"),
            usage=("sh launch.sh <seed> <cpu> [<world>|all] [<share>|all]; sh launch.sh pooled <cpu>; "
                   "sh launch.sh all <cpu A> [<cpu B>]"),
            shares=("'all' means the post-hoc shares 0.01 0.03 0.05 0.1; a spec-v2 share given explicitly is refused; "
                    "the spec-v2 references are skipped (their signals.json exists)"),
            cpus=("every CPU is given explicitly as one number; the launch refuses a CPU it cannot pin to and two equal "
                  "CPUs; 'all' with two CPUs runs seeds 0 and 1 side by side, then seed 2 split by world, then the "
                  "pooled look; with one CPU it runs seeds 0, 1 and 2 in turn, then the pooled look"),
            logs="logs/v3_*.log (the spec-v2 logs are kept as they are)",
            priority="nice 15 (NICE=... to change), one process per CPU, pinned with taskset"),
        outputs=dict(per_job="<world>/seed<k>/share_1pct, share_3pct, share_5pct, share_10pct/ (as spec v2's per_job)",
                     pooled="pooled_v3.json", checks="checks_v3_seed0.json, timing_v3.json", logs="logs/v3_*.log"),
        compute=("projected in timing_v3.json: the spec-v2 trace jobs' measured wall times, scaled by the measured "
                 "change from the 0.5% log to the 10% log"),
        v2_payload_reproduced_by_this_code=v2_reproduced(v2))


def v3_payload(v2, v2_sha, reason):
    payload = copy.deepcopy(v2)
    payload["version"] = 3
    payload["spec_history"] = [dict(version=1, **v2["previous_spec"]),
                               dict(version=2, sha256=v2_sha, archived="spec_history/spec_v2.json",
                                    replaced_because=reason, kept="every key of spec v2 is kept in spec v3 unchanged, "
                                    "apart from 'version' and 'previous_spec'")]
    payload["previous_spec"] = dict(sha256=v2_sha, archived="spec_history/spec_v2.json", replaced_because=reason)
    payload[V3_EXTENSION_KEY] = post_hoc_section(v2, v2_sha, reason)
    same = [k for k in v2 if k not in ("version", "previous_spec")
            and T.A.canonical(T.jsonable(payload[k])) != T.A.canonical(v2[k])]
    if same:
        raise SystemExit(f"refused: spec v3 would change spec-v2 keys {same}")
    return T.jsonable(payload)


def extend_post_hoc(reason):
    if not reason:
        raise SystemExit("refused: spec v3 needs --reason")
    found = post_hoc_outputs()
    if found:
        raise SystemExit("refused: post-hoc job outputs exist; the spec may not change: " + ", ".join(map(str, found)))
    spec, spec_sha = T.load_spec()
    version = int(spec.get("version", 1))
    hist = T.OUT / "spec_history"
    if version == 3:
        v2 = T.read_json(hist / "spec_v2.json")
        v2_sha = T.file_sha256(hist / "spec_v2.json")
        if T.A.canonical(v3_payload(v2, v2_sha, reason)) == T.A.canonical(spec):
            print(f"spec.json (v3) already written and identical: {spec_sha}")
            return spec_sha
        raise SystemExit("refused: a different spec v3 exists")
    if version != 2:
        raise SystemExit(f"refused: spec v3 extends spec v2; the current spec is version {version}")
    T.check_gate_files(spec)
    payload = v3_payload(spec, spec_sha, reason)
    hist = T.guard_out(hist)
    hist.mkdir(exist_ok=True)
    for suffix in ("", ".sha256"):
        dest = hist / f"spec_v2.json{suffix}"
        if dest.exists():
            raise SystemExit(f"refused: {dest} exists")
        shutil.copy2(str(T.SPEC_PATH) + suffix, dest)
    if T.file_sha256(hist / "spec_v2.json") != spec_sha:
        raise SystemExit("refused: the archived spec v2 differs from spec.json")
    digest = T.write_json(T.SPEC_PATH, payload)
    print(f"archived spec v2 ({spec_sha[:16]}) in {hist}; wrote {T.SPEC_PATH} version 3 sha256 {digest}")
    print("v2 payload reproduced by this code:", payload[V3_EXTENSION_KEY]["v2_payload_reproduced_by_this_code"])
    return digest


def main(reason=None):
    if run_outputs():
        raise SystemExit("refused: trained job outputs exist; the spec may not change: "
                         + ", ".join(str(p) for p in run_outputs()))
    payload = spec_payload()
    if T.SPEC_PATH.exists():
        old = T.read_json(T.SPEC_PATH)
        if T.A.canonical(old) == T.A.canonical(payload):
            print("spec.json already written and identical")
            return
        if not reason:
            raise SystemExit("refused: a spec exists; a new one needs --reason")
        old_sha = T.file_sha256(T.SPEC_PATH)
        version = int(old.get("version", 1))
        payload["version"] = version + 1
        payload["previous_spec"] = dict(sha256=old_sha, archived=f"spec_history/spec_v{version}.json",
                                        replaced_because=reason)
        hist = T.guard_out(T.OUT / "spec_history")
        hist.mkdir(exist_ok=True)
        for suffix in ("", ".sha256"):
            shutil.copy2(str(T.SPEC_PATH) + suffix, hist / f"spec_v{version}.json{suffix}")
        sup = T.guard_out(T.OUT / "superseded" / f"v{version}")
        sup.mkdir(parents=True, exist_ok=True)
        for name in UNDER_OLD_SPEC:
            for suffix in ("", ".sha256"):
                p = T.OUT / (name + suffix)
                if p.exists():
                    shutil.move(str(p), sup / p.name)
        for p in T.OUT.glob("*/seed*/*/trace_rows.json*"):
            dest = sup / p.relative_to(T.OUT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), dest)
        for name in ("code", "code_manifest_history"):
            if (T.OUT / name).exists():
                shutil.move(str(T.OUT / name), sup / name)
        print(f"archived spec v{version} ({old_sha[:16]}) and moved its outputs to {sup}")
    digest = T.write_json(T.SPEC_PATH, T.jsonable(payload))
    print(f"wrote {T.SPEC_PATH} version {payload['version']} sha256 {digest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason")
    ap.add_argument("--post-hoc-v3", action="store_true")
    args = ap.parse_args()
    if args.post_hoc_v3:
        extend_post_hoc(args.reason)
    else:
        main(args.reason)
