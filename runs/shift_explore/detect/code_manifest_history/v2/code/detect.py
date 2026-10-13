"""EXPLORATORY (Stage 1 of outputs/shift/beyond-the-data/PLAN.md, section 4.4): the LOCKED detection script, and the
declared hook through which Stage 2 (plan section 5.2) will score BCA's width with it. NO BCA in Stage 1: no band,
calibrator, threshold, dose or BCA update is fitted, read or called here; nothing from algorithms/td3_bc_bca.py,
calibration/ or learners.bca_* is imported. The hook takes any callable U(s, a); it imports nothing of BCA.

The interface (the Stage 2 contract, scratchpad/ood/stage2/detect_contract.py):
  PAIR_SETS = ("T1", "T2", "T3", "T4")
  pair_sets(ctx, world, out=OUT) -> dict   the locked pair sets (pairs.py); loaded and verified when saved, else built
                                           and saved under `out`
  score_signal(name, fn, ctx, world, out_dir=None, ...) -> record   scores every pair, writes them once, hashed;
                                           never reads a label (signals.py)
  labels(ctx, world, out=OUT) -> record    the verified labels record; the labels themselves are computed only by the
                                           label script as a process of its own (run_detect.py --phase labels, or
                                           labels.run in a process that loads no signal code)
  measures(scores_path, labels_path, *, level=None) -> dict   the join, only of hashed files (measures.py)
  stand_in_signals(ctx, world) -> {name: fn}   the five data-only stand-ins, the perfect band (sanity check) and the
                                           reported extras (signals.py)
  score_bca_width(width_fn, ctx, world, out_dir, *, stage2_prereg_sha256, version, pairs_out=OUT) -> record
                                           THE DECLARED STAGE 2 HOOK (below)
  STAGE2_MARKS, A1_LESSON                  plan 5.2's marks as Stage 2 will apply them, and change A1's lesson
"""

import re
from pathlib import Path

import numpy as np

import common as C
import pairs as P
import signals as S
import measures as M

PAIR_SETS = P.PAIR_SETS
STANDIN_NAMES = S.STANDIN_NAMES
MEASURE_KEYS = ("widening_T1_harmful", "widening_T3_harmful", "size_T1_paired", "size_T1_raw", "ranking_T12",
                "ranking_T3", "rate_above_level_T4", "catch_rate_T3_past80", "flag_rate_past80", "n_harmful_T1",
                "n_harmful_T3")


# ------------------------------------------------------------------------------------------------ the contract

def pair_sets(ctx, world, out=C.OUT, log_fn=C.log, label=C.LABEL):
    """The locked pair sets of one seed and world: loaded and verified when already saved under `out`, else built
    from the seed's artifacts (no signal, no label) and saved. Keys: states, commands, refs, set, role, pair,
    log_row, train_index, t3_source, t3_start, t3_step, sha256 (content), file_sha256."""
    d = C.seed_dir(ctx.rep, world, out)
    if (d / P.RECORD).exists():
        arrays, rec = P.load(ctx.rep, world, out)
    else:
        arrays, info = P.build(ctx, world, log_fn)
        rec = P.save(arrays, info, ctx.rep, world, out, label=label)
    return dict(arrays, sha256=rec["content_sha256"], file_sha256=rec["file_sha256"])


score_signal = S.score_signal


def labels(ctx, world, out=C.OUT):
    """The verified labels record of (seed, world). Labels are never computed in a process that holds signal code:
    when they do not exist this refuses and names the command that computes them."""
    rec = C.read_json(C.seed_dir(ctx.rep, world, out) / "labels.json")
    if rec is None:
        raise SystemExit(f"refused: no labels for seed {ctx.rep} {world} under {out}. Compute them with the label "
                         f"script as its own process: python run_detect.py --phase labels --seed {ctx.rep} --world "
                         f"{world} (or labels.run(ctx, world, out) in a process that has not loaded signals.py, "
                         f"measures.py or detect.py)")
    C.load_npz(C.seed_dir(ctx.rep, world, out) / rec["file"], rec["file_sha256"], rec["content_sha256"])
    return rec


def measures(scores_path, labels_path, *, level=None, extra_levels=None):
    return M.measures(scores_path, labels_path, level=level, extra_levels=extra_levels)


def stand_in_signals(ctx, world, names=S.ALL_NAMES):
    return S.stand_in_signals(ctx, world, names)


# ------------------------------------------------------------------------------------------------ the Stage 2 hook

def score_bca_width(width_fn, ctx, world, out_dir, *, stage2_prereg_sha256, version, pairs_out=C.OUT, log_fn=C.log):
    """DECLARED HOOK, Stage 2 only (plan 5.2). Never called in Stage 1.

    Accepts BCA's width U(s, a) as a callable width_fn(states (m, 13) float64, commands (m, 4) float32) -> (m,)
    (Stage 2's band.WidthReader: it reads the state rounded to float32 and normalised as the learners see it), with
    attributes .name (starting 'bca_width') and .sha256 (its band file). Scores every pair of the locked pair sets of
    (seed, world) exactly as the stand-ins were scored, in critic value units, and writes the widths once, hashed,
    under out_dir (inside runs/shift_explore/stage2). The widths are joined with the labels only by measures(), after
    both are hashed: Stage 2 then calls measures(<width record>, <labels record>, level=tau) and applies STAGE2_MARKS.
    Refuses: out_dir outside Stage 2's folder, a callable without a BCA width name and SHA-256, an unknown version, a
    malformed pre-registration SHA-256. This hook imports nothing of BCA and checks no approval itself: Stage 2's
    WidthReader refuses without the user's approval of the Stage 2 pre-registration."""
    out_dir = Path(out_dir)
    roots = (C.STAGE2_OUT, C.TEST_OUT)
    if not any(r.resolve() == out_dir.resolve() or r.resolve() in out_dir.resolve().parents for r in roots):
        raise SystemExit(f"refused: BCA widths are written only under {C.STAGE2_OUT}")
    name = getattr(width_fn, "name", None)
    sha = getattr(width_fn, "sha256", None)
    if not (isinstance(name, str) and name.startswith("bca_width")) or not isinstance(sha, str) or not sha:
        raise SystemExit("refused: width_fn must carry .name ('bca_width...') and .sha256 (its band file)")
    if version not in ("uniform", "weighted"):
        raise SystemExit("refused: version must be 'uniform' or 'weighted'")
    if not (isinstance(stage2_prereg_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", stage2_prereg_sha256)):
        raise SystemExit("refused: stage2_prereg_sha256 must be the 64-hex SHA-256 of the Stage 2 pre-registration")
    if getattr(width_fn, "needs_ref", False):
        raise SystemExit("refused: BCA's width is read raw at each pair; the paired forms are formed by measures()")
    return S.score_signal(name, width_fn, ctx, world, out_dir, pairs_out=pairs_out, units="value",
                          label=C.LABEL_STAGE2, log_fn=log_fn,
                          extra=dict(stage2_prereg_sha256=stage2_prereg_sha256, bca_version=version,
                                     hook="detect.score_bca_width",
                                     detect_spec_sha256=C.file_sha256(C.SPEC_PATH) if C.SPEC_PATH.exists() else None))


# ------------------------------------------------------------------------------------------------ Stage 2's marks

A1_LESSON = (
    "Change A1 (room spec v2, changes_before_any_run) found that a residual-based band carries an offset that does not "
    "come from harm: at recorded commands BCA's one-step residual delta*(s, a) averages +0.149 (seed 0), so the plan's "
    "literal |delta*| first FALLS just past the 80% knee and rewards the first points of over-throttle; measured from "
    "the row's own recorded command, [delta*(s, a_i) - delta*(s, a)]_+, the offset cancels and the penalty rises "
    "past 80%. Check (e) then required a penalty to rise past 80%. The lesson for Stage 2: (1) the paired marks "
    "compare the width at the command with the width at its 80% cap AT THE SAME STATE (direction: U(s, a) > U(s, "
    "cap(a)); any offset that does not change with the command cancels), never the width at the command against "
    "widths at other states or against a level; (2) a penalty is read relative to the recorded command at the same "
    "state, W_U(s, a) = [U(s, a) - U(s, a_i)]_+, which is what Stage 2's policy test applies at k = 1 (stage2/"
    "policy.py, k_paired), so the size mark uses the penalty difference ([U(s, a) - U(s, a_i)]_+ - [U(s, cap) - "
    "U(s, a_i)]_+) / over_pref (never larger than the raw width difference; the raw ratio is reported); (3) whether the "
    "width rises just past the knee is read at the 81% knee pair (knee_rise_T1_*), reported, as check (e) asks of a "
    "penalty.")

STAGE2_MARKS = dict(
    pooled="mean over seeds 0-4 of the per-seed value; each mark also holds in at least 4 of 5 seeds; a missing or "
           "failed seed fails every mark (plan 5.2)",
    scored_band=("BCA's band fitted with settings unchanged on each seed's frozen gapped fixed-controller critic "
                 "(residual target with the stall-aware controller at the next state), and on check 3's frozen host "
                 "critic; widths read only at the locked pair sets through score_bca_width; marks on the "
                 "fixed-critic band in the stalling world"),
    direction=dict(rule="share of harmful past-80% pairs with U(s, command) > U(s, cap) at the same state",
                   keys=["widening_T1_harmful", "widening_T3_harmful"], T1_at_least=0.75, T3_at_least=0.65,
                   chance=0.5),
    size=dict(rule=("median over harmful past-80% T1 pairs of ([U(s, a) - U(s, a_i)]_+ - [U(s, cap) - U(s, a_i)]_+) "
                    "/ over_pref, a_i the row's recorded command at the same state (change A1's lesson: the "
                    "consumer's penalty at k = 1); at least 1, so that k = 1 would cancel the false preference"),
              key="size_T1_paired", at_least=1.0, reported="size_T1_raw (median of (U(s, a) - U(s, cap)) / over_pref)"),
    ranking=dict(rule="P(a random harmful pair has the larger width than a random harmless pair), ties one half",
                 keys=["ranking_T12", "ranking_T3"], T12_at_least=0.75, T3_at_least=0.70),
    calibration_sanity=dict(rule="share of T4 rows with U above Stage 2's tau", key="rate_above_level_T4",
                            target=0.05, tolerance=0.015,
                            note=("T4 is drawn from the same pool as tau's rows (calibration split, gap removed, bank "
                                  "rows left out); the check is informative only if tau is computed on rows of that "
                                  "pool outside T4 (pairs.npz holds T4's log rows: set == 'T4', log_row)")),
    selectivity=dict(rule="flag_rate_past80 (at tau) in the stalling world minus the same in the no-stall world",
                     gated=False, note="reported in Stage 2; gated only in a trace test bed (plan 4.6)"),
    claim_rule=("'BCA flags the actor's reliance on commands the critic cannot vouch for' only if direction, size and "
                "ranking hold AND BCA's ranking score (T12 and T3, each) is no more than 0.02 below the best data-only "
                "stand-in's (the best of distance, density, classifier_ratio, two_head, residual_evidence, each scored "
                "by this script at its own 95% level; the perfect band and the range hinge are not stand-ins). 'Identifies "
                "the harm' only with selectivity of at least 0.2 in a trace test bed."),
    if_direction_or_size_fails=("declare the weighted version (Variant A-weighted) and repeat the detection test with "
                                "this same script; no uniform-weight policy test runs"),
    standin_margin=0.02, seeds_needed=4,
    a1_lesson=A1_LESSON)
