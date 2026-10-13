"""Stage 2: the interface between BCA's width readout and the locked detection script (plan section 4.4), and the
detection test's marks and claim rule (plan section 5.2, updated with change A1).

STATE: when this contract was first written, scratchpad/ood/detect/ did not exist. The other job's detection script
now implements it and is locked (runs/shift_explore/detect/code_manifest.json, pinned by prereg.json); it added three
things Stage 2 adopts: (1) score_bca_width, a declared hook for BCA's width (it refuses outputs outside
runs/shift_explore/stage2 and a callable without a 'bca_width' name and band SHA-256); (2) labels are computed only by
its label script in a process that holds no signal code (labels.run), and labels() only returns the verified record;
(3) measures() returns {'measures': {...}, 'joined': {...}}. Without a locked script prereg.json is a draft
(approvable: false) and approve.py refuses it.

THE TWO COMPARISON SETS (they differ, on purpose):
  detection claim (this file): the five data-only stand-ins the detection script scores, STANDIN_NAMES = distance to
      data, behaviour density, classifier ratio, two-head disagreement, residual evidence. The range hinge is not one
      of them (it is a hand-set limit, reported), nor is the perfect band (an oracle, the sanity check).
  policy test (run_stage2.MAIN_STANDINS, plan 4.2): range hinge, behaviour density, classifier ratio, distance to
      data, two-head disagreement; the best of them by select_standin is the policy test's matched rival.

WHAT THE LOCKED DETECTION SCRIPT MUST EXPOSE (one Python module; names checked by check_detect_module before any use):

  PAIR_SETS = ("T1", "T2", "T3", "T4")

  pair_sets(ctx, world, out) -> dict
      The locked pair sets of one seed and world, built from the seed's artifacts only (ctx: fresh.GateContext or
      fresh.FreshContext; it may call ctx.prepared(world), ctx.none(world), ctx.clone(world), ctx.sd[world]) with no
      signal and no label. Keys: 'states' (n, 13) float64 physics states, 'commands' (n, 4), 'set' (n,) in PAIR_SETS,
      'pair' (n,) int id of the state within its set, 'role' (n,) in {'command', 'cap', 'recorded', 'cloned'}
      ('command' the gapped critic's actor's command, 'cap' that command capped at 80%, 'recorded' the row's recorded
      command (T1, T2, T4), 'cloned' the cloned actor's command), 'sha256' of the arrays. T1: 1,500 full-throttle-region
      training states x four commands; T2: 1,500 other training states x the same four; T3: 1,500 strong-current states
      visited by the gapped critic's actor and 1,500 visited by the stall-aware controller x (command, cap); T4: 5,000
      held-out recorded rows x the recorded command. Deterministic for (seed, world).

  score_signal(name, fn, ctx, world, out_dir) -> dict
      Calls fn(states (m, 13) float64, commands (m, 4)) -> (m,) scores on every pair of pair_sets(ctx, world); writes
      the scores (npz) and a record with the SHA-256 of the scores and of the pair sets to out_dir; never reads a
      label. Returns the record, which holds at least 'scores_path' and 'scores_sha256'. Stage 2 passes
      band.WidthReader objects as fn (each has .name and .sha256).

  labels(ctx, world, out) -> dict   (and its label script labels.run(ctx, world, out), run in a process with no
                                     signal code)
      The labels, from exact simulation, by code that cannot see any signal: per pair 'unsupported' (any thruster
      past 80%), 'harmful' (the critic's preference for the command over its 80% cap exceeds the true preference by at
      least 0.1 AND the cap is truly better by at least 0.05) and 'over_preference' (the critic's preference minus the
      true preference, command over cap), at the pairs of T1 and T3. Written once, hashed. Returns a record holding at
      least 'labels_path' and 'labels_sha256'.

  score_bca_width(width_fn, ctx, world, out_dir, *, stage2_prereg_sha256, version, pairs_out) -> dict
      The declared Stage 2 hook: score_signal for BCA's WidthReader, outputs only under runs/shift_explore/stage2.

  measures(scores_path, labels_path, *, level=None) -> dict
      Joins one signal's scores with the labels (after both are hashed); its 'measures' entry holds at least
      MEASURE_KEYS:
        widening_T1_harmful    share of harmful past-80% T1 states with s(command) > s(cap)
        widening_T3_harmful    the same on T3
        size_T1_paired         median over harmful past-80% T1 states of
                               ([s(command) - s(recorded)]_+ - [s(cap) - s(recorded)]_+) / over_preference
                               (change A1: the consumer reads the width relative to the recorded command)
        size_T1_raw            median of (s(command) - s(cap)) / over_preference on the same states (reported)
        ranking_T12            P(a random harmful pair scores above a random harmless pair), pairs of T1 and T2
                               (ties count one half)
        ranking_T3             the same on T3
        rate_above_level_T4    share of T4 rows with s > level (level: Stage 2's tau; default the signal's own
                               split-conformal 95% level on T4)
        catch_rate_T3_past80   share of past-80% T3 commands with s > level
        flag_rate_past80       share of past-80% pairs (T1 and T3) with s > level (selectivity = stall - no-stall)
        n_harmful_T1, n_harmful_T3

  stand_in_signals(ctx, world) -> {name: fn}
      The data-only stand-ins the rehearsal scores, same call signature as fn above: distance to data, behaviour
      density, classifier ratio, two-head disagreement, residual evidence (and the perfect band, a sanity check,
      excluded from "best stand-in"). If its classifier-ratio estimator differs from stage2/weights.py, the
      pre-registration must be rewritten before approval so that both use one estimator.

WHAT STAGE 2 CALLS: pair_sets (for seeds 0-4 only when the rehearsal's pair-set record already exists: then it only
loads and verifies; Stage 2 never builds pair sets in the rehearsal's folder), score_bca_width with BCA's
WidthReader (uniform fixed-critic band, uniform host band; the weighted band only if the uniform one fails direction
or size), labels (no BCA; for seeds 0-4 the rehearsal's, which must exist), measures for BCA and for every stand-in,
at Stage 2's tau for the calibration-sanity rate.
"""

import numpy as np

REQUIRED = ("PAIR_SETS", "pair_sets", "score_signal", "labels", "measures", "stand_in_signals", "score_bca_width")
MEASURE_KEYS = ("widening_T1_harmful", "widening_T3_harmful", "size_T1_paired", "size_T1_raw", "ranking_T12",
                "ranking_T3", "rate_above_level_T4", "catch_rate_T3_past80", "flag_rate_past80", "n_harmful_T1",
                "n_harmful_T3")
STANDIN_NAMES = ("distance", "density", "classifier_ratio", "two_head", "residual_evidence")

# Plan 5.2's marks (pooled = mean over the seeds of the per-seed value; "every seed": each also in >= 4 of 5 seeds).
MARKS = dict(widening_T1=0.75, widening_T3=0.65, size_T1=1.0, ranking_T12=0.75, ranking_T3=0.70,
             calibration_rate=0.05, calibration_tol=0.015, standin_margin=0.02, seeds_needed=4)


def check_detect_module(mod):
    """Refuse unless the locked detection script exposes the contract."""
    missing = [n for n in REQUIRED if not hasattr(mod, n)]
    bad = [n for n in REQUIRED[1:] if hasattr(mod, n) and not callable(getattr(mod, n))]
    if missing or bad or tuple(getattr(mod, "PAIR_SETS", ())) != ("T1", "T2", "T3", "T4"):
        raise SystemExit(f"refused: the detection script does not expose the Stage 2 contract (missing {missing}, "
                         f"not callable {bad})")
    return True


def check_measures(m):
    """The flat measures of one signal (the detection script returns them under 'measures')."""
    if isinstance(m, dict) and isinstance(m.get("measures"), dict):
        m = m["measures"]
    missing = [k for k in MEASURE_KEYS if k not in m]
    if missing:
        raise SystemExit(f"refused: the detection script's measures lack {missing}")
    return m


def _mean(xs):
    xs = [x for x in xs]
    return None if not xs or any(not _finite(x) for x in xs) else float(np.mean(xs))


def _count(xs, ok):
    return sum(1 for x in xs if _finite(x) and ok(x))


def _finite(x):
    return x is not None and not isinstance(x, bool) and np.isfinite(x)


def standin_comparison(standins, n_seeds, marks=MARKS, names=STANDIN_NAMES):
    """The best data-only stand-in's pooled ranking scores, held to BCA's standard: every one of the five detection
    stand-ins (STANDIN_NAMES) must have a finite ranking_T12 and ranking_T3 on every seed of the group. A stand-in
    that is missing, has fewer seeds, or a missing or non-finite measure makes the comparison INCOMPLETE, and an
    incomplete comparison fails the 'not below the best stand-in' part of the claim rule (a rival that cannot be
    scored counts against BCA, as in the policy test). Returns {key: {name, value, all}}, complete, problems."""
    problems = []
    for name in names:
        ms = standins.get(name)
        if ms is None or len(ms) != n_seeds:
            problems.append(f"{name}: {0 if ms is None else len(ms)} of {n_seeds} seeds")
            continue
        for i, m in enumerate(ms):
            for key in ("ranking_T12", "ranking_T3"):
                if not isinstance(m, dict) or not _finite(m.get(key)):
                    problems.append(f"{name}: seed index {i} has no finite {key}")
    best = {}
    for key in ("ranking_T12", "ranking_T3"):
        per = {name: _mean([m.get(key) if isinstance(m, dict) else None for m in standins.get(name) or [None]])
               for name in names}
        fin = {k: v for k, v in per.items() if v is not None}
        name = max(fin, key=fin.get) if fin and not problems else None
        best[key] = dict(name=name, value=None if name is None else fin[name], all=per)
    return best, not problems, problems


def detection_marks(bca, standins, n_seeds=5, marks=MARKS, group="exploratory"):
    """bca: [measures per seed] (stalling world, the fixed-critic band; None for a failed seed); standins: {name:
    [measures per seed]}. Every seed needed (a missing or failed seed fails every mark). Returns the marks, whether
    the claim rule is met, and the version decision ('direction_or_size_failed' -> the weighted version is declared
    and the detection test repeated).

    The comparison with the stand-ins needs all five detection stand-ins (STANDIN_NAMES: distance, density,
    classifier_ratio, two_head, residual_evidence) with finite ranking scores on every seed (standin_comparison);
    otherwise that part fails.

    Claim fields: only the CONFIRMATORY group (fresh seeds 5-9) can carry the claim. For the exploratory group
    (seeds 0-4, used to design this test) 'claim' is None and claim_text says the marks were met or not met on
    seeds used for design; 'claim_rule_met' holds the arithmetic for both groups."""
    if group not in ("exploratory", "confirmatory"):
        raise ValueError(group)
    if len(bca) != n_seeds or any(m is None for m in bca):
        return dict(complete=False, passed=False, claim_rule_met=False,
                    claim=False if group == "confirmatory" else None, direction_or_size_failed=True,
                    claim_text=_claim_text(group, False), reason="a seed is missing or failed", group=group)

    def mark(key, threshold, ge=True):
        vals = [m.get(key) for m in bca]
        mean = _mean(vals)
        ok = (lambda x: x >= threshold) if ge else (lambda x: x <= threshold)
        n_ok = _count(vals, ok)
        return dict(values=vals, mean=mean, seeds_ok=n_ok,
                    passed=bool(mean is not None and ok(mean) and n_ok >= marks["seeds_needed"]))

    out = dict(complete=True, group=group)
    out["direction"] = dict(T1=mark("widening_T1_harmful", marks["widening_T1"]),
                            T3=mark("widening_T3_harmful", marks["widening_T3"]))
    out["direction"]["passed"] = out["direction"]["T1"]["passed"] and out["direction"]["T3"]["passed"]
    out["size"] = mark("size_T1_paired", marks["size_T1"])
    out["size_raw_reported"] = dict(values=[m.get("size_T1_raw") for m in bca],
                                    mean=_mean([m.get("size_T1_raw") for m in bca]))
    out["ranking"] = dict(T12=mark("ranking_T12", marks["ranking_T12"]), T3=mark("ranking_T3", marks["ranking_T3"]))
    out["ranking"]["passed"] = out["ranking"]["T12"]["passed"] and out["ranking"]["T3"]["passed"]
    rates = [m.get("rate_above_level_T4") for m in bca]
    lo, hi = marks["calibration_rate"] - marks["calibration_tol"], marks["calibration_rate"] + marks["calibration_tol"]
    out["calibration_sanity"] = dict(values=rates, mean=_mean(rates), seeds_ok=_count(rates, lambda x: lo <= x <= hi),
                                     passed=bool(_mean(rates) is not None and lo <= _mean(rates) <= hi and
                                                 _count(rates, lambda x: lo <= x <= hi) >= marks["seeds_needed"]))
    best, complete, problems = standin_comparison(standins, n_seeds, marks)
    out["best_standin"] = best
    vs = {}
    for key, part in (("ranking_T12", "T12"), ("ranking_T3", "T3")):
        b = out["ranking"][part]["mean"]
        s = best[key]["value"]
        vs[key] = dict(bca=b, best=s, passed=bool(complete and b is not None and s is not None
                                                  and b >= s - marks["standin_margin"]))
    out["not_below_best_standin"] = dict(vs, complete=complete, problems=problems,
                                         passed=bool(complete and all(v["passed"] for v in vs.values())))
    out["direction_or_size_failed"] = not (out["direction"]["passed"] and out["size"]["passed"])
    met = bool(out["direction"]["passed"] and out["size"]["passed"] and out["ranking"]["passed"]
               and out["not_below_best_standin"]["passed"])
    out["claim_rule_met"] = met
    out["claim"] = met if group == "confirmatory" else None
    out["claim_text"] = _claim_text(group, met)
    out["passed"] = met
    return out


def _claim_text(group, met):
    if group == "confirmatory":
        return ("BCA flags the actor's reliance on commands the critic cannot vouch for" if met else
                "not claimed: BCA's width did not meet the detection marks and claim rule on fresh seeds 5-9")
    return ("exploratory, not a claim: the detection marks and claim rule were met on seeds 0-4, which were used to "
            "design this test" if met else
            "exploratory, not a claim: the detection marks and claim rule were not met on seeds 0-4")


def selectivity(stall, no_stall):
    """Reported (gated only in a trace test bed): flag rate at past-80% pairs, stalling minus no-stall world."""
    s = [None if a is None or b is None else a.get("flag_rate_past80", np.nan) - b.get("flag_rate_past80", np.nan)
         for a, b in zip(stall, no_stall)]
    return dict(values=s, mean=_mean(s))
