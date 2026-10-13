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

VERSION = 1


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
             change=("T1 gets a fifth role 'knee' = float32(clip(command, -0.81, 0.81)); measures knee_rise_T1_*; the "
                     "plan's four roles are unchanged"),
             reason="check (e) (room change A2) requires a penalty to rise just past 80%, not only far out"),
        dict(id="D4 size read two ways",
             change=("size_T1_raw = median (s(cmd) - s(cap)) / over_pref (plan 4.4's words) and size_T1_paired = "
                     "median ([s(cmd) - s(rec)]_+ - [s(cap) - s(rec)]_+) / over_pref; Stage 2's size mark uses the "
                     "paired one (never larger than the raw one)"),
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
        dict(id="D12 order rule", change=("Stage 1 signals are scored before the labels of that seed and world exist "
                                          "('signals' refuses after labels; 'labels' refuses before signals_done.json)"),
             reason="a guard against any signal being adjusted on labels; not in the plan"),
        dict(id="D13 labels for every pair",
             change="every pair (every role, every set) is labelled against its own 80% cap",
             reason="the ranking score needs a label for every pair; caps are harmless by construction"),
        dict(id="D14 ranking pool, catch rates and false flags",
             change=("ranking over every pair of the set(s); catch rate over harmful pairs and over past-80% command "
                     "pairs; false flags over harmless pairs of T1-T3 (per role reported); level = the signal's own "
                     "split-conformal 95% level on T4"),
             reason="the plan gives the measures' names only; these are the readings fixed before any run"),
        dict(id="D15 commands are the float32 network's",
             change="a pair's command is the float32 actor at the stored float32 state; T3 rollouts use the float64 copy",
             reason="the critic, BCA's width and every signal see the float32 command; the gate evaluates the copy"),
    ]
    payload = dict(
        label=C.LABEL, kind="detect-spec", version=VERSION,
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
                                             "in a new spec version")),
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
                     logs="logs/seed<k>.log, logs/pooled.log", checks="checks/ (seed-0 check), timing.json"),
        compute=("measured by checks_seed0.py under this spec (timing.json); the plan's estimate is about 2.5 "
                 "process-hours"),
        run_order=("after the room test's pooled look passes: for each seed 0-4, sh launch.sh <seed> <cpu> both pairs "
                   "signals labels measures; then sh launch.sh all <cpu> both pooled"),
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
    digest = C.write_json(C.SPEC_PATH, payload)
    print(f"spec written: {C.SPEC_PATH} sha256 {digest}")


if __name__ == "__main__":
    main()
