"""The claims of the 12-setup study, judged from per-seed outcomes: the statistics, Holm's correction, the badges, the
reading order, the TD3+BC selection label and the shelving write-ups (plan sections 5.5, 6.1 "What a stop means", 6.2,
7.3 "Order under the cap" and 9.1).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. Pure Python
and numpy (no JAX, no scipy, no file reads), so it imports in both environments. export.py reads the run records and
scores files, hands their per-seed outcomes to judge(), and writes what comes back into results.json. Every number in
the plan's rules is a constant below; none is read from an outcome.

Statistics (plan section 5.5), restated from the weighting test:
  per seed      a setup's mean return over a bank's 1,000 starts (the scores file's J_mean)
  Delta         the paired per-seed difference between two setups of one learner, averaged over the 5 seeds
  sd, se        the sample standard deviation over the 5 seeds (n - 1) and sd / sqrt(5)
  bounds        one-sided 95% bounds with t on 4 degrees of freedom: mean -+ T95 x se (lower bound L, upper bound),
                restated from experiments/shift_iw/coverage.py:318-329 (across_seeds). T95 = 2.13184678632665 (the
                plan's "2.132") comes from the closed form of Student's t with 4 degrees of freedom below, not from
                scipy, so the tests can check it against scipy.stats.t independently
  p-value       one-sided: the chance of a mean gain at least this large if the true gain were zero,
                P(T4 >= mean / se); 0 when sd = 0 and the mean is above 0, else 1
                (experiments/shift_iw/run.py:1200-1212, p_one_sided). p < 0.05 exactly when L > 0
  Holm          run.py:1215-1224 (holm): the family's p-values sorted by (p, learner); the i-th smallest of k passes if
                it and every smaller one are at most 0.05 / (k - i + 1). The Holm-adjusted p-value shown on the page is
                the running maximum of min(1, (k - i + 1) p); it is at most 0.05 exactly when the step-down passes
  family p      1 for a learner that did not learn (Claim 0), whose per-seed difference is not finite on any seed, or
                (Claim 2 only) whose Claim 4a did not pass (plan section 6.2, Holm family membership; the weighting
                test's run.py:1633-1637 and 1831-1837)

The claims (plan section 6.2; every bar is fixed now, before any run):
  0   the learner learned (a gate): shelved ("did not learn") if the learner alone's mean no-shift return over the 5
      seeds is below the quality's bar, or any seed's return of the learner alone is not finite, or a run of
      Unifloral's unchanged step (alone or the room reference) stopped (plan section 6.1). Bars: medium-expert 60
      points, medium 39 points (the recording controllers' no-shift points 69.97 and 48.66 minus 10, rounded), in
      return units through start_banks.json's exact no-shift anchors (-800.2886042574038 and -35.193218771607356,
      which export.py passes): -341.2314 and -501.9014 return units (the plan rounds them to -341.2 and -501.9, from
      the anchors rounded to -800.29 and -35.19; PLAN_NO_SHIFT_ANCHORS holds those for the tests only)
  1   standard BCA beats the learner alone on the strong mix: passes only if Delta(standard - alone) >= +1.8 return
      units, L > 0 and Holm passes across the four learners (one family per quality)
  1c  flagged if Delta(standard - alone) <= -1.8 with upper bound < 0 (strong mix)
  2   IW BCA beats standard BCA on the strong mix: as Claim 1 for Delta(IW - standard), and Claim 4a must pass
  2c  flagged if Delta(IW - standard) <= -1.8 with upper bound < 0 (strong mix)
  3   flagged if Delta(IW - standard) <= -1.8 with upper bound < 0 on the no-shift bank
  3b  flagged if Delta(standard - alone) <= -1.8 with upper bound < 0 on the no-shift bank
  4a  at the IW setup's last refresh, on strong-law test steps outside the ball, each step at its own threshold:
      shelved if the pooled coverage (the mean of the 5 seeds' shares) lies outside [89%, 96%]
  4b  a label, not a gate: shelved if the standard threshold covers 88% or more of the same steps (the IW setup's
      band at its standard threshold, as the weighting test's marks pair them); a passing Claim 2 is then labeled
      "gain from extra width, not repaired calibration"
  5   flagged as harm if a BCA setup's F1 or F2 rate exceeds its learner alone's by >= 2 percentage points on any of
      the 7 banks, with L > 0 (the bar compared exactly on start counts)
  6   stage 3 only, where Claim 1 or 2 passes: shelved (the gain is credited to the amount of caution) if
      Delta(setup - its flattened twin) < +1.8 or L <= 0; for Claim 2 also Delta(IW - shuffled IW)
Flags (1c, 2c, 3, 3b, 5) and Claim 6 are not corrected: they are pointers, not evidence. Claims 1 and 2 read
"pending" until all 5 seeds of all four learners of the quality (alone, standard and IW) are complete; every other
claim waits for the 5 seeds of its own inputs. A learner that did not learn reads "did not learn", its Claims 1-6 read
"not judged", and it stays in both Holm families with p = 1. A TD3+BC Claim 1 pass on medium-expert carries "expected
partly by selection; needs seeds 10-14" (plan section 2.1).

Badges: passed, shelved, flagged, pending, not judged, did not learn. A flag whose condition does not hold reads
"passed" (the check passed; its verdict says "not flagged").

Version 3 (PRESPEC.md section 20, post hoc): a failed float32 check (stop rule S7, limit unchanged at 0.05 points per
start) stops a scope within its quality instead of the study. A failure on an alone, standard or weights-of-1 actor
stops the learner: all its claims read "not judged" ("stopped by S7"), and it stays in both Holm families with p = 1. A
failure on an IW actor (or IW's repeat) stops only its IW setup: the claims that compare the IW setup (2, 2c, 3, 4a, 4b,
Claim 5's IW rows and Claim 6's Claim-2 part) read "not judged", it stays in Claim 2's family with p = 1, and its
Claims 0, 1, 1c, 3b and Claim 5's standard rows are judged as usual. A stopped learner wins over a stopped IW setup.
The runs of a stopped scope (trained before the stop) enter no claim, statistic or mean; they are listed as excluded.
Claims 1 and 2 count a stopped scope as complete (with p = 1), so they wait only for the setups that still run. The
driver's state.json "s7_stopped" entries (quality, learner, scope, setup, run, ...) come in through judge(s7_stopped=).

Input (one quality): runs = {(learner, setup): {seed: run}}, where a missing seed is absent and run is a dict:
  state     "complete" (trained and scored) or "stopped" (a stop recorded as this run's outcome, plan section 6.1)
  finite    True when every return of the scored readout on all 7 banks is finite (False for a stopped run)
  J         {bank: mean 100-step return over the bank's starts}           (final snapshot, scored readout)
  points    {bank: mean points on the bank's own scale}
  events    {"f1": {bank: number of starts with the event}, "f2": {...}}; starts {bank: number of starts}
  coverage  None or {"own": {"outside": share, ...}, "standard": {"outside": share, ...}}   (BCA setups, last refresh)
  pointer   None or {"change": float, "extra": float}                    (IW setups; Claim 2's pointer, below)
  stop      None or {"rule": ..., "reason": ..., "step": ...}
  source    {"record": SHA-256 of the run record, "scores": SHA-256 of the scores file}, carried to every number
Setups: "alone", "standard", "iw" (the claims), "room" (Claim 0's stop rule and the room readout), stage 3's
"standard_flattened", "iw_flattened" and "iw_shuffled" (Claim 6).
"""

import math

import numpy as np

SCHEMA = "shift-hosts-claims-v1"
LEARNERS = ("td3_bc", "cql", "iql", "rebrac")
LEARNER_NAMES = dict(td3_bc="TD3+BC", cql="CQL", iql="IQL", rebrac="ReBRAC")
QUALITIES = ("medium-expert", "medium")
SEEDS = (0, 1, 2, 3, 4)
N_SEEDS = len(SEEDS)
DEGREES_OF_FREEDOM = N_SEEDS - 1                                    # 4
CLAIM_SETUPS = ("alone", "standard", "iw")
STAGE3_SETUPS = ("standard_flattened", "iw_flattened", "iw_shuffled", "iw_estimated")
SETUP_NAMES = dict(alone="alone", standard="standard BCA", iw="IW BCA", room="room reference (alone, full log)",
                   sham_alone="sham alone (choice 19)", standard_flattened="standard BCA, flattened (stage 3)",
                   iw_flattened="IW BCA, flattened (stage 3)", iw_shuffled="IW BCA, shuffled (stage 3)",
                   iw_estimated="IW BCA with estimated weights (stage 3)")
BANKS = ("none", "mild", "strong", "recorded", "strong_current", "big_waves", "big_kicks")   # bca_core.BANKS
WEIGHTING_BANKS, SITUATIONS = BANKS[:3], BANKS[3:]
STRONG, NO_SHIFT = "strong", "none"
EVENTS = ("f1", "f2")

ALPHA = 0.05                        # each Holm family's chance of any false pass (plan section 5.5)
LEVEL = 0.95                        # one-sided bounds
SIZE_BAR = 1.8                      # return units: version 4's 3 points (plan section 6)
HARM_BAR_PERCENTAGE_POINTS = 2      # Claim 5
COVERAGE_4A = (0.89, 0.96)          # Claim 4a, inclusive (the weighting test's own mark, iw/SPEC.md:164)
COVERAGE_4B = 0.88                  # Claim 4b
CLAIM0_RECORDING_POINTS = {"medium-expert": 69.97, "medium": 48.66}     # start_banks.json, no shift (plan 6.2)
CLAIM0_BAR_POINTS = {"medium-expert": 60.0, "medium": 39.0}             # those minus 10, rounded
# iw/SPEC.md:287 as the plan quotes them (rounded); tests only: every verdict uses start_banks.json's exact anchors
PLAN_NO_SHIFT_ANCHORS = dict(random=-800.29, top=-35.19)

PASSED, SHELVED, FLAGGED, PENDING, NOT_JUDGED, DID_NOT_LEARN = (
    "passed", "shelved", "flagged", "pending", "not judged", "did not learn")
BADGES = (PASSED, SHELVED, FLAGGED, PENDING, NOT_JUDGED, DID_NOT_LEARN)
SELECTION_LABEL = "expected partly by selection; needs seeds 10-14"
WIDTH_LABEL = "gain from extra width, not repaired calibration"
POINTER_LABEL = "pointer, not evidence"
OVERLAP_LABEL = "overlapping shift"
EXPLORATORY_LABEL = ("EXPLORATORY: seeds 0-4 of the weighting test's data, already seen by it; confirmation needs "
                     "fresh data seeds 10-14 of its generator and a pre-registration approved at the user's terminal")

STOP_RULES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S6b", "S7", "S8", "S9", "S10", "S11")
READING_ORDER = ("0", "4a", "4b", "1", "1c", "2", "2c", "3", "3b", "5", "6")    # plan section 6.2, fixed now
HOLM_CLAIMS = ("1", "2")
FLAGS = ("1c", "2c", "3", "3b", "5")

# Version 3's S7 scopes (module docstring; restated from stages.S7_SCOPE: the failing actor's setup -> what stops)
S7_SCOPE = dict(alone="learner", standard="learner", iw_ones="learner", iw="iw", iw_repeat="iw")
S7_IW_SETUPS = ("iw", "iw_flattened", "iw_shuffled", "iw_estimated")   # the runs a stopped IW setup leaves out
S7_IW_CLAIMS = ("2", "2c", "3", "4a", "4b")    # the claims a stopped IW setup leaves unjudged (with 5's and 6's parts)
S7_STOPPED = "stopped by S7"                   # a phone-view row's status and an S7 write-up's badge (not a claim badge)

# plan section 6.2's table; a and b name the paired comparison a - b and bank its start bank
CLAIMS = {
    "0": dict(title="The learner learned (a gate for reading its BCA rows)", kind="gate", bank=NO_SHIFT),
    "1": dict(title="Standard BCA beats the learner alone on the strong mix", kind="claim", a="standard", b="alone",
              bank=STRONG),
    "1c": dict(title="Standard BCA costs return on the strong mix (a flag)", kind="flag", a="standard", b="alone",
               bank=STRONG),
    "2": dict(title="IW BCA beats standard BCA on the strong mix", kind="claim", a="iw", b="standard", bank=STRONG),
    "2c": dict(title="IW costs return against standard on the strong mix (a flag)", kind="flag", a="iw",
               b="standard", bank=STRONG),
    "3": dict(title="IW costs nothing without the shift", kind="flag", a="iw", b="standard", bank=NO_SHIFT),
    "3b": dict(title="Standard BCA costs nothing without the shift", kind="flag", a="standard", b="alone",
               bank=NO_SHIFT),
    "4a": dict(title="IW gives the band its promised coverage under the shift (read before Claim 2)", kind="gate"),
    "4b": dict(title="The shift breaks the standard band (a label, not a gate)", kind="label"),
    "5": dict(title="BCA adds no physical failures", kind="flag"),
    "6": dict(title="The gain needs the band's placement, not just more caution (stage 3, only where Claim 1 or 2 "
                    "passes)", kind="claim"),
}
KILL_RULES = {
    "0": "the learner alone's mean no-shift return over the 5 seeds is below the bar, or any seed's return of the "
         "learner alone is not finite, or a run of Unifloral's unchanged step (alone or the room reference) stopped",
    "1": "Delta(standard - alone) is below +1.8 return units, or its one-sided 95% lower bound is not above 0, or it "
         "fails Holm's correction across the four learners of this quality",
    "1c": "flagged if Delta(standard - alone) <= -1.8 return units with an upper bound below 0",
    "2": "Delta(IW - standard) is below +1.8 return units, or its one-sided 95% lower bound is not above 0, or it "
         "fails Holm's correction across the four learners of this quality, or Claim 4a fails for this learner",
    "2c": "flagged if Delta(IW - standard) <= -1.8 return units with an upper bound below 0",
    "3": "flagged if, on the no-shift bank, Delta(IW - standard) <= -1.8 return units with an upper bound below 0",
    "3b": "flagged if, on the no-shift bank, Delta(standard - alone) <= -1.8 return units with an upper bound below 0",
    "4a": "at the IW setup's last refresh, on strong-law test steps outside the ball, each step at its own threshold, "
          "the pooled coverage lies outside [89%, 96%]",
    "4b": "the standard threshold covers 88% or more of the same steps (Claim 2 is then still judged, but labeled "
          "'gain from extra width, not repaired calibration')",
    "5": "flagged as harm if a BCA setup's F1 or F2 rate exceeds its learner alone's by 2 percentage points or more "
         "on any of the 7 banks, with a lower bound above 0",
    "6": "Delta(setup - its flattened twin) is below +1.8 return units or its lower bound is not above 0; for Claim 2 "
         "also Delta(IW - shuffled IW)",
}
IDEAS = {
    "0": "{learner} alone learns a useful policy from the gapped {quality} logs. This is a gate: its BCA rows are read "
         "only if it learned.",
    "1": "Standard BCA adds caution where {learner}'s critic is least sure, scaled by the band's width, so {learner} "
         "with standard BCA should score a higher 100-step return than {learner} alone on the strong-mix starts.",
    "2": "Importance weighting (IW) recalibrates BCA's band for the strong-mix shift, so {learner} with IW BCA should "
         "score a higher return than {learner} with standard BCA on the strong-mix starts.",
    "4a": "With exact importance weights and a threshold per row, {learner}'s band should cover the critic's actual "
          "one-step error on strong-mix test steps at the promised rate (89-96%).",
    "4b": "The strong-mix shift should break the standard band of {learner}'s IW setup (cover below 88% of the same "
          "steps), so that what IW adds is repaired calibration rather than extra width.",
    "6": "{learner}'s gain needs the band's placement (more caution where the band is wide), not just more caution "
         "on average.",
}
POWER_NOTE = ("With 5 seeds, a shelved claim means no effect large enough to see, not no effect: with Holm's "
              "correction a true gain of 1.8 return units passes only about 6-10% of the time, and about 5-9 return "
              "units are needed to pass reliably (6.75-9.25 if only one learner has the gain, 5.25-7.25 if all four "
              "share it; plan section 5.5).")


# ------------------------------------------------------------------------------------------------ Student's t, 4 df

def t4_sf(t):
    """P(T > t) for Student's t with 4 degrees of freedom, in closed form. With q = 4 / (4 + t^2) and
    s = t / sqrt(4 + t^2), the distribution function is 1/2 + (s / 2)(1 + q / 2); for t >= 0 the upper tail is written
    without cancellation as q^2 (3 + q) / (8 (1 + s (1 + q / 2)))."""
    t = float(t)
    if math.isnan(t):
        return math.nan
    if t < 0.0:
        return 1.0 - t4_sf(-t)
    if math.isinf(t):
        return 0.0
    r = 4.0 + t * t
    q, s = 4.0 / r, t / math.sqrt(r)
    return q * q * (3.0 + q) / (8.0 * (1.0 + s * (1.0 + 0.5 * q)))


def t4_quantile(level):
    """The t with P(T4 <= t) = level (0.5 < level < 1), by bisection on t4_sf to float64 precision."""
    if not 0.5 < level < 1.0:
        raise ValueError("level must lie in (0.5, 1)")
    lo, hi, target = 0.0, 1.0e3, 1.0 - level
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t4_sf(mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


T95 = t4_quantile(LEVEL)            # 2.13184678632665: the plan's 2.132


# ------------------------------------------------------------------------------------------------ summaries

def _num(v):
    """A float, or None for a missing value (None stays None; non-finite floats stay as they are)."""
    return None if v is None else float(v)


def summary(values, *, missing=()):
    """Per-seed values (seed order; None or a non-finite number where a seed has no finite value) -> dict(per_seed,
    n, nonfinite, missing, complete, mean, sd, se, t, lower, upper, p_one_sided). The mean, bounds and p-value are
    given only when there are exactly 5 values, all finite, and no seed is missing (complete): a claim is never judged
    on fewer seeds, and a seed that is not finite is never dropped (plan section 6.2). Restated from
    coverage.py:318-329 and run.py:1200-1212."""
    vals = [_num(v) for v in values]
    finite = [v is not None and math.isfinite(v) for v in vals]
    out = dict(per_seed=vals, n=len(vals), nonfinite=sum(1 for f in finite if not f), missing=list(missing),
               complete=bool(not missing and len(vals) == N_SEEDS and all(finite)), mean=None, sd=None, se=None,
               t=None, lower=None, upper=None, p_one_sided=None)
    if not out["complete"]:
        return out
    a = np.asarray(vals, np.float64)
    m, sd = float(a.mean()), float(a.std(ddof=1))
    se = sd / math.sqrt(len(a))
    if sd == 0.0:
        t = math.inf if m > 0 else -math.inf if m < 0 else math.nan
        p = 0.0 if m > 0 else 1.0
    else:
        t = m / se
        p = t4_sf(t)
    out.update(mean=m, sd=sd, se=se, t=t, lower=m - T95 * se, upper=m + T95 * se, p_one_sided=p)
    return out


def family_p(s):
    """A Holm family member's p-value from its summary: 1 when it is not complete and finite (run.py:1633-1637)."""
    p = s.get("p_one_sided") if s else None
    return 1.0 if p is None or not math.isfinite(p) else float(p)


def holm(pvalues, alpha=ALPHA):
    """Holm's correction, restated from run.py:1215-1224: {name: p-value} -> {name: passes}. The k p-values are sorted
    by (p, name); the i-th smallest passes if it and every smaller one are at most alpha / (k - i + 1)."""
    order = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    k, out, still = len(order), {}, True
    for i, (name, p) in enumerate(order, 1):
        still = still and p <= alpha / (k - i + 1)
        out[name] = bool(still)
    return out


def holm_table(pvalues, alpha=ALPHA):
    """Holm's steps in order: rank, learner, p, its bar alpha / (k - rank + 1), the adjusted p-value (running maximum
    of min(1, (k - rank + 1) p)) and whether it passes (the step-down of holm(), which decides)."""
    passes = holm(pvalues, alpha)
    order = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    k, rows, running = len(order), [], 0.0
    for i, (name, p) in enumerate(order, 1):
        running = max(running, min(1.0, (k - i + 1) * p))
        rows.append(dict(rank=i, learner=name, p=float(p), bar=alpha / (k - i + 1), adjusted=running,
                         passes=passes[name]))
    return rows


# ------------------------------------------------------------------------------------------------ reading outcomes

def run_ok(run):
    """A run that finished, was scored, and whose every return is finite."""
    return run is not None and run.get("state") == "complete" and bool(run.get("finite"))


def _source(run):
    return None if run is None else run.get("source")


def setup_values(runs, learner, setup, bank, field="J"):
    """One setup's per-seed values on a bank: dict(setup, bank, field, per_seed=[{seed, v, source}], missing=[seeds],
    summary). v is None for a seed whose run stopped or whose returns are not finite."""
    by_seed = runs.get((learner, setup), {})
    per, missing = [], []
    for s in SEEDS:
        run = by_seed.get(s)
        if run is None:
            missing.append(s)
            continue
        v = float(run[field][bank]) if run_ok(run) else None
        per.append(dict(seed=s, v=v, source=_source(run)))
    return dict(setup=setup, bank=bank, field=field, per_seed=per, missing=missing,
                summary=summary([p["v"] for p in per], missing=missing))


def paired(runs, learner, a, b, bank, field="J"):
    """The paired per-seed difference a - b of two setups of one learner on a bank: dict(comparison, bank,
    per_seed=[{seed, a, b, d, a_source, b_source}], missing=[seeds missing in either], summary). d is None where
    either run stopped or is not finite on that seed."""
    A, B = runs.get((learner, a), {}), runs.get((learner, b), {})
    per, missing = [], []
    for s in SEEDS:
        ra, rb = A.get(s), B.get(s)
        if ra is None or rb is None:
            missing.append(s)
            continue
        va = float(ra[field][bank]) if run_ok(ra) else None
        vb = float(rb[field][bank]) if run_ok(rb) else None
        d = None if va is None or vb is None else va - vb
        per.append(dict(seed=s, a=va, b=vb, d=d, a_source=_source(ra), b_source=_source(rb)))
    return dict(comparison=f"{a} - {b}", a=a, b=b, bank=bank, field=field, per_seed=per, missing=missing,
                summary=summary([p["d"] for p in per], missing=missing))


def claim0_bar(quality, anchor):
    """The quality's Claim 0 bar in points and in return units through the no-shift bank's anchors
    (anchor = dict(random=..., top=...); points = 100 (J - random) / (top - random), world.points)."""
    pts = CLAIM0_BAR_POINTS[quality]
    lo, hi = float(anchor["random"]), float(anchor["top"])
    return dict(points=pts, return_units=lo + pts / 100.0 * (hi - lo),
                recording_points=CLAIM0_RECORDING_POINTS[quality],
                rule="the recording controller's no-shift points minus 10, rounded", anchor=dict(random=lo, top=hi))


# ------------------------------------------------------------------------------------------------ S7's scopes (version 3)

def s7_scopes(entries, quality):
    """One quality's S7 stops from the driver's state.json "s7_stopped" entries -> {learner: dict(scope, runs,
    entries)}. scope is "learner" when any entry stops the learner (it wins), else "iw". An entry without a known scope
    takes it from its failing setup (S7_SCOPE); one with neither stops the learner, the stricter reading."""
    out = {}
    for e in entries or []:
        if not isinstance(e, dict) or e.get("quality") != quality or e.get("learner") not in LEARNERS:
            continue
        scope = e.get("scope") if e.get("scope") in ("learner", "iw") else S7_SCOPE.get(e.get("setup"), "learner")
        cur = out.setdefault(e["learner"], dict(scope=scope, runs=[], entries=[]))
        if scope == "learner":
            cur["scope"] = "learner"
        if e.get("run") is not None and e["run"] not in cur["runs"]:
            cur["runs"].append(e["run"])
        cur["entries"].append(dict(e))
    return out


def s7_excluded(scopes, learner, setup):
    """Whether this learner's runs of this setup are left out because S7 stopped their scope: every setup of a stopped
    learner (its room references and sham-alone run included), the IW setups (stage 3's IW twins included) of a
    stopped IW setup."""
    sc = (scopes or {}).get(learner)
    return sc is not None and (sc["scope"] == "learner" or setup in S7_IW_SETUPS)


def s7_stops_claim(scopes, learner, cid):
    """Whether S7 stopped this learner's claim: every claim of a stopped learner, the IW claims of a stopped IW setup."""
    sc = (scopes or {}).get(learner)
    return sc is not None and (sc["scope"] == "learner" or cid in S7_IW_CLAIMS)


def s7_why(scopes, learner):
    """The reason a stopped scope's claims read "not judged", naming the runs whose float32 check failed."""
    sc = scopes[learner]
    what = "the learner" if sc["scope"] == "learner" else "its IW setup"
    return f"stopped by S7 (the float32 check failed on {', '.join(sc['runs']) or 'a run'}, so {what} stops in this " \
           f"quality)"


def without_s7_stopped(runs, scopes):
    """runs without the runs of a scope S7 stopped -> (the runs kept, the runs left out as [dict(learner, setup, seed,
    state, source, why)]). The left-out runs enter no claim, statistic or mean."""
    kept, left = {}, []
    for (L, setup), seeds in runs.items():
        if s7_excluded(scopes, L, setup):
            left += [dict(learner=L, setup=setup, seed=sd, state=r.get("state"), source=_source(r),
                          why="excluded: stopped by S7") for sd, r in sorted(seeds.items())]
        else:
            kept[(L, setup)] = seeds
    return kept, left


# ------------------------------------------------------------------------------------------------ results

def _result(cid, badge, verdict, reasons=(), labels=(), **details):
    meta = CLAIMS[cid]
    out = dict(claim=cid, title=meta["title"], kind=meta["kind"], badge=badge, verdict=verdict, reasons=list(reasons),
               labels=list(labels), kill_rule=KILL_RULES[cid])
    out.update(details)
    return out


def _s7_result(cid, scopes, learner, labels=(), **details):
    """A claim S7 stopped (version 3): not judged, with the failing runs named; a Holm claim stays in its family with
    p = 1."""
    why = s7_why(scopes, learner)
    tail = "; it stays in Holm's family with p = 1." if cid in HOLM_CLAIMS else "."
    return _result(cid, NOT_JUDGED, f"Not judged: {why}{tail}", [why], labels, s7_stopped=True, **details)


def _fmt(x, nd=2, sign=True):
    if x is None:
        return "n/a"
    if not math.isfinite(x):
        return str(x)
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _fmt_p(p):
    """A p-value for text: four decimals, or "< 0.0001" so a tiny p never reads as exactly 0."""
    if p is None:
        return "n/a"
    return "< 0.0001" if 0.0 < p < 1e-4 else f"{p:.4f}"


def judge_claim0(quality, runs, learner, anchor):
    """Claim 0 (plan section 6.2; the stop rule of section 6.1)."""
    name = LEARNER_NAMES[learner]
    bar = claim0_bar(quality, anchor) if anchor is not None else None
    alone, room = runs.get((learner, "alone"), {}), runs.get((learner, "room"), {})
    vals = setup_values(runs, learner, "alone", NO_SHIFT)
    pts = setup_values(runs, learner, "alone", NO_SHIFT, "points")
    failed = sorted(s for s, r in alone.items() if not run_ok(r))
    room_failed = sorted(s for s, r in room.items() if not run_ok(r))
    details = dict(bar=bar, per_seed_return=vals["per_seed"], per_seed_points=pts["per_seed"],
                   alone_failed_seeds=failed, room_failed_seeds=room_failed,
                   stops={f"alone seed {s}": alone[s].get("stop") for s in failed}
                   | {f"room seed {s}": room[s].get("stop") for s in room_failed})
    if failed or room_failed:
        reasons = []
        if failed:
            reasons.append(f"the learner alone's return is not finite (or its run stopped) on seed(s) {failed}")
        if room_failed:
            reasons.append(f"the room reference (Unifloral's unchanged step on the full log) stopped or is not finite "
                           f"on seed(s) {room_failed}: a failure in Unifloral's own code is that learner's Claim 0 "
                           f"failure (plan section 6.1)")
        return _result("0", DID_NOT_LEARN, f"{name} alone did not learn: " + "; ".join(reasons) + ".", reasons,
                       **details)
    if vals["missing"]:
        return _result("0", PENDING, f"Waiting for {name} alone on seed(s) {vals['missing']}.",
                       [f"seeds missing: {vals['missing']}"], **details)
    if bar is None:
        return _result("0", PENDING, "Waiting for the no-shift bank's anchors.", ["no anchors"], **details)
    s = vals["summary"]
    mean_points = pts["summary"].get("mean")
    details.update(mean_return=s["mean"], mean_points=mean_points)
    if s["mean"] < bar["return_units"]:
        why = (f"its mean no-shift return {s['mean']:.2f} is below the bar {bar['return_units']:.2f} return units "
               f"({bar['points']:.0f} points)")
        return _result("0", DID_NOT_LEARN, f"{name} alone did not learn: {why}.", [why], **details)
    return _result("0", PASSED, f"{name} alone learned: its mean no-shift return {s['mean']:.2f} is at or above the "
                                f"bar {bar['return_units']:.2f} return units ({bar['points']:.0f} points).", **details)


def judge_coverage(runs, learner, claim0):
    """Claims 4a and 4b from the IW setup's coverage at its last refresh (plan section 6.2)."""
    name = LEARNER_NAMES[learner]
    labels = [OVERLAP_LABEL]
    if claim0["badge"] == DID_NOT_LEARN:
        why = "the learner did not learn (Claim 0)"
        return (_result("4a", NOT_JUDGED, f"Not judged: {why}.", [why], labels),
                _result("4b", NOT_JUDGED, f"Not judged: {why}.", [why], labels))
    iw = runs.get((learner, "iw"), {})
    missing = [s for s in SEEDS if s not in iw]
    stopped = [s for s in SEEDS if s in iw and iw[s].get("state") != "complete"]
    unread = [s for s in SEEDS if s in iw and iw[s].get("state") == "complete" and not iw[s].get("coverage")]
    per = [dict(seed=s, own=None, standard=None, source=_source(iw.get(s))) for s in SEEDS]
    for p in per:
        cov = (iw.get(p["seed"]) or {}).get("coverage") or {}
        p["own"] = _num((cov.get("own") or {}).get("outside"))
        p["standard"] = _num((cov.get("standard") or {}).get("outside"))
    details = dict(per_seed=per)
    if stopped:
        why = f"the IW setup stopped on seed(s) {stopped}, so its band has no last refresh to read"
        return (_result("4a", NOT_JUDGED, f"Not judged: {why}.", [why], labels, **details),
                _result("4b", NOT_JUDGED, f"Not judged: {why}.", [why], labels, **details))
    if missing or unread:
        why = f"waiting for the IW setup's coverage readout on seed(s) {sorted(missing + unread)}"
        return (_result("4a", PENDING, f"Waiting: {why}.", [why], labels, **details),
                _result("4b", PENDING, f"Waiting: {why}.", [why], labels, **details))
    own = [p["own"] for p in per]
    std = [p["standard"] for p in per]
    if any(v is None or not math.isfinite(v) for v in own + std):
        why = "a seed's coverage share is missing or not finite"
        return (_result("4a", NOT_JUDGED, f"Not judged: {why}.", [why], labels, **details),
                _result("4b", NOT_JUDGED, f"Not judged: {why}.", [why], labels, **details))
    pooled_own, pooled_std = float(np.mean(own)), float(np.mean(std))
    lo, hi = COVERAGE_4A
    if lo <= pooled_own <= hi:
        a = _result("4a", PASSED, f"{name}'s IW band covers {100 * pooled_own:.2f}% of strong-law test steps outside "
                                  f"the ball (pooled over 5 seeds), inside [89%, 96%].", [], labels,
                    pooled=pooled_own, **details)
    else:
        side = "below 89%" if pooled_own < lo else "above 96%"
        why = f"the pooled coverage {100 * pooled_own:.2f}% is {side}"
        a = _result("4a", SHELVED, f"{name}'s IW band misses its promised coverage: {why}.", [why], labels,
                    pooled=pooled_own, **details)
    if pooled_std < COVERAGE_4B:
        b = _result("4b", PASSED, f"The shift breaks {name}'s standard band: its standard threshold covers "
                                  f"{100 * pooled_std:.2f}% of the same steps, below 88%.", [], labels,
                    pooled=pooled_std, **details)
    else:
        why = f"the standard threshold still covers {100 * pooled_std:.2f}% of the same steps (88% or more)"
        b = _result("4b", SHELVED, f"The shift does not break {name}'s standard band: {why}. A passing Claim 2 is "
                                   f"labeled '{WIDTH_LABEL}'.", [why], labels, pooled=pooled_std, **details)
    return a, b


def _family(cid, quality, runs, res, complete, scopes=None):
    """Claim 1 or 2 for the four learners, with Holm's correction over its family (plan sections 5.5 and 6.2). A learner
    whose claim S7 stopped (version 3, scopes) stays in the family with p = 1."""
    meta = CLAIMS[cid]
    deltas = {L: paired(runs, L, meta["a"], meta["b"], meta["bank"]) for L in LEARNERS}
    stopped = [L for L in LEARNERS if s7_stops_claim(scopes, L, cid)]
    pvals, why_p1, waiting = {}, {}, []
    if not complete:
        waiting.append("all 5 seeds of all four learners' alone, standard and IW setups"
                       + (" (but those S7 stopped)" if scopes else ""))
    for L in LEARNERS:
        s, why = deltas[L]["summary"], []
        if L in stopped:
            why_p1[L] = [s7_why(scopes, L)]
            pvals[L] = 1.0
            continue
        if res[L]["0"]["badge"] == PENDING and complete:
            waiting.append(f"{LEARNER_NAMES[L]}'s Claim 0")
        if res[L]["0"]["badge"] == DID_NOT_LEARN:
            why.append("the learner did not learn (Claim 0)")
        elif not deltas[L]["missing"] and not s["complete"]:
            why.append("a seed's return is not finite")
        if cid == "2" and res[L]["0"]["badge"] != DID_NOT_LEARN:
            b4a = res[L]["4a"]["badge"]
            if b4a == SHELVED:
                why.append("Claim 4a did not pass")
            elif b4a == NOT_JUDGED:
                why.append("Claim 4a was not judged")
            elif b4a == PENDING and not why:
                waiting.append(f"{LEARNER_NAMES[L]}'s Claim 4a")
        why_p1[L] = why
        pvals[L] = 1.0 if why else family_p(s)
    judged = not waiting
    table = holm_table(pvals) if judged else []
    by_learner = {row["learner"]: row for row in table}
    family = dict(claim=cid, alpha=ALPHA, members=list(LEARNERS), judged=judged, waiting_for=waiting,
                  p_values=pvals if judged else None, p_one_reasons=why_p1 if judged else None, holm=table,
                  s7_stopped=stopped)
    out = {}
    for L in LEARNERS:
        d, s, name = deltas[L], deltas[L]["summary"], LEARNER_NAMES[L]
        labels = [OVERLAP_LABEL] if cid == "2" else []
        details = dict(delta=d)
        if L in stopped:
            out[L] = _s7_result(cid, scopes, L, labels, family=dict(by_learner[L]) if judged else dict(p=1.0),
                                **details)
            continue
        if res[L]["0"]["badge"] == DID_NOT_LEARN:
            out[L] = _result(cid, NOT_JUDGED, "Not judged: the learner did not learn (Claim 0); it stays in Holm's "
                                              "family with p = 1.", ["the learner did not learn (Claim 0)"], labels,
                             family=dict(by_learner[L]) if judged else dict(p=1.0), **details)
            continue
        if not judged:
            out[L] = _result(cid, PENDING, "Pending: judged only when " + "; ".join(waiting) + " are complete.",
                             ["waiting for " + w for w in waiting], labels, family=None, **details)
            continue
        row = by_learner[L]
        reasons = list(why_p1[L])
        if s.get("complete"):
            if s["mean"] < SIZE_BAR:
                reasons.append(f"the mean gain {_fmt(s['mean'])} is below +1.8 return units")
            if not s["lower"] > 0.0:
                reasons.append(f"the lower bound {_fmt(s['lower'])} is not above 0")
        if not row["passes"]:
            if row["p"] > row["bar"]:
                reasons.append(f"it fails Holm's correction: its p-value {_fmt_p(row['p'])} is above "
                               f"{row['bar']:.4f}, the bar at its place ({row['rank']} of {len(LEARNERS)}) in Holm's "
                               f"order")
            else:
                reasons.append(f"it fails Holm's correction: an earlier learner in Holm's order failed its step, so "
                               f"the step-down stops before this one (its p-value {_fmt_p(row['p'])}, its bar "
                               f"{row['bar']:.4f})")
        passed = not reasons
        if passed and cid == "1" and L == "td3_bc" and quality == "medium-expert":
            labels.append(SELECTION_LABEL)
        if passed and cid == "2" and res[L]["4b"]["badge"] == SHELVED:
            labels.append(WIDTH_LABEL)
        comparison = "standard BCA - alone" if cid == "1" else "IW BCA - standard BCA"
        text = (f"{name}: {comparison} on the strong mix {_fmt(s.get('mean'))} return units, lower bound "
                f"{_fmt(s.get('lower'))}, Holm-adjusted p {_fmt_p(row['adjusted'])}")
        out[L] = _result(cid, PASSED if passed else SHELVED,
                         text + (": passed." if passed else ": shelved (" + "; ".join(reasons) + ")."), reasons,
                         labels, family=dict(p=row["p"], rank=row["rank"], bar=row["bar"], adjusted=row["adjusted"],
                                             passes=row["passes"]), **details)
    return family, out


def judge_flag(cid, runs, learner, claim0):
    """Claims 1c, 2c, 3 and 3b: uncorrected flags (pointers, not evidence)."""
    meta = CLAIMS[cid]
    labels = [POINTER_LABEL] + ([OVERLAP_LABEL] if "iw" in (meta["a"], meta["b"]) else [])
    d = paired(runs, learner, meta["a"], meta["b"], meta["bank"])
    s = d["summary"]
    if claim0["badge"] == DID_NOT_LEARN:
        return _result(cid, NOT_JUDGED, "Not judged: the learner did not learn (Claim 0).",
                       ["the learner did not learn (Claim 0)"], labels, delta=d)
    if d["missing"]:
        return _result(cid, PENDING, f"Waiting for seed(s) {d['missing']}.", [f"seeds missing: {d['missing']}"],
                       labels, delta=d)
    if not s.get("complete"):
        return _result(cid, NOT_JUDGED, "Not judged: a seed's return is not finite.",
                       ["a seed's return is not finite"], labels, delta=d)
    hit = s["mean"] <= -SIZE_BAR and s["upper"] < 0.0
    text = (f"Delta({d['comparison']}) on the {'strong mix' if meta['bank'] == STRONG else 'no-shift bank'} "
            f"{_fmt(s['mean'])} return units, upper bound {_fmt(s['upper'])}")
    if hit:
        why = f"Delta {_fmt(s['mean'])} <= -1.8 with an upper bound {_fmt(s['upper'])} below 0"
        return _result(cid, FLAGGED, text + ": flagged.", [why], labels, delta=d)
    return _result(cid, PASSED, text + ": not flagged.", [], labels, delta=d)


def judge_claim5(runs, learner, claim0, stopped=(), why_stopped=None):
    """Claim 5: a BCA setup's F1 or F2 rate minus its learner alone's, per seed and bank, in percentage points;
    flagged as harm where the mean is at least 2 percentage points (compared exactly on start counts) and L > 0.
    stopped: setups S7 stopped (version 3: "iw" for a stopped IW setup); they read "not judged" (why_stopped) and the
    badge is read from the other setup's comparisons."""
    labels = [POINTER_LABEL]
    if claim0["badge"] == DID_NOT_LEARN:
        return _result("5", NOT_JUDGED, "Not judged: the learner did not learn (Claim 0).",
                       ["the learner did not learn (Claim 0)"], labels, by_setup={}, comparisons=[])
    alone = runs.get((learner, "alone"), {})
    comparisons, by_setup = [], {}
    for setup in ("standard", "iw"):
        if setup in stopped:
            by_setup[setup] = NOT_JUDGED
            continue
        other = runs.get((learner, setup), {})
        states = []
        for event in EVENTS:
            for bank in BANKS:
                missing = [s for s in SEEDS if s not in other or s not in alone]
                row = dict(setup=setup, event=event, bank=bank, missing=missing)
                if missing:
                    row.update(state=PENDING)
                elif not all(run_ok(other[s]) and run_ok(alone[s]) for s in SEEDS):
                    row.update(state=NOT_JUDGED, why="a seed's run stopped or is not finite")
                else:
                    ka = [int(other[s]["events"][event][bank]) for s in SEEDS]
                    kb = [int(alone[s]["events"][event][bank]) for s in SEEDS]
                    n = [int(other[s]["starts"][bank]) for s in SEEDS] + [int(alone[s]["starts"][bank]) for s in SEEDS]
                    if len(set(n)) != 1 or n[0] <= 0:
                        row.update(state=NOT_JUDGED, why="the setups were scored on different numbers of starts")
                    else:
                        n0 = n[0]
                        dk = [x - y for x, y in zip(ka, kb)]
                        s = summary([100.0 * k / n0 for k in dk])
                        meets = sum(dk) * 100 >= HARM_BAR_PERCENTAGE_POINTS * n0 * N_SEEDS
                        hit = bool(meets and s["lower"] > 0.0)
                        # counts in seed order 0-4; each seed's record SHA-256 is in the setups' record lists
                        row.update(state=FLAGGED if hit else PASSED, starts=n0, counts=dict(setup=ka, alone=kb),
                                   mean_pp=s["mean"], lower_pp=s["lower"], upper_pp=s["upper"], meets_bar=bool(meets))
                comparisons.append(row)
                states.append(row["state"])
        by_setup[setup] = (FLAGGED if FLAGGED in states else PENDING if PENDING in states else
                           NOT_JUDGED if NOT_JUDGED in states else PASSED)
    states = [b for s, b in by_setup.items() if s not in stopped]
    badge = (FLAGGED if FLAGGED in states else PENDING if PENDING in states else NOT_JUDGED if NOT_JUDGED in states
             else PASSED if states else NOT_JUDGED)
    hits = [c for c in comparisons if c["state"] == FLAGGED]
    reasons = [f"{SETUP_NAMES[c['setup']]} {c['event'].upper()} on {c['bank']}: {_fmt(c['mean_pp'])} percentage "
               f"points, lower bound {_fmt(c['lower_pp'])}" for c in hits]
    verdict = {FLAGGED: "Flagged as harm: " + "; ".join(reasons) + ".",
               PASSED: "Not flagged: no BCA setup's F1 or F2 rate exceeds the learner alone's by 2 percentage points "
                       "with a lower bound above 0 on any of the 7 banks.",
               PENDING: "Waiting for the setups' 5 seeds.",
               NOT_JUDGED: "Not judged: a seed's run stopped or is not finite."}[badge]
    for setup in stopped:
        verdict += f" {SETUP_NAMES[setup][0].upper() + SETUP_NAMES[setup][1:]} is not judged: {why_stopped}."
    return _result("5", badge, verdict, reasons, labels, by_setup=by_setup, comparisons=comparisons,
                   **({"s7_stopped_setups": list(stopped)} if stopped else {}))


def judge_claim6(runs, learner, claim1, claim2):
    """Claim 6 (stage 3 only): each passing claim's gain against its flattened twin, and Claim 2's also against the
    shuffled IW twin (plan sections 5.2 and 6.2)."""
    name = LEARNER_NAMES[learner]
    parts = {}
    for cid, parent, twins in (("1", claim1, (("standard", "standard_flattened"),)),
                               ("2", claim2, (("iw", "iw_flattened"), ("iw", "iw_shuffled")))):
        if parent["badge"] != PASSED:
            why = parent["reasons"][0] if parent.get("s7_stopped") else f"Claim {cid} did not pass"
            parts[cid] = dict(badge=NOT_JUDGED, why=why, deltas=[])
            continue
        deltas = [paired(runs, learner, a, b, STRONG) for a, b in twins]
        if any(d["missing"] for d in deltas):
            parts[cid] = dict(badge=PENDING, why="waiting for stage 3's twins on all 5 seeds (they run only within "
                                                 "the stage-3 cap, in the order of plan section 7.3)", deltas=deltas)
            continue
        fails = []
        for d in deltas:
            s = d["summary"]
            if not s.get("complete"):
                fails.append(f"Delta({d['comparison']}): a seed's return is not finite")
                continue
            if s["mean"] < SIZE_BAR:
                fails.append(f"Delta({d['comparison']}) {_fmt(s['mean'])} is below +1.8")
            if not s["lower"] > 0.0:
                fails.append(f"Delta({d['comparison']})'s lower bound {_fmt(s['lower'])} is not above 0")
        parts[cid] = dict(badge=SHELVED if fails else PASSED, why="; ".join(fails) if fails else None, deltas=deltas)
    states = [p["badge"] for p in parts.values()]
    if all(s == NOT_JUDGED for s in states):
        badge = NOT_JUDGED
    elif PENDING in states:
        badge = PENDING
    elif SHELVED in states:
        badge = SHELVED
    else:
        badge = PASSED
    reasons = [f"after Claim {c}: {p['why']}" for c, p in parts.items() if p["why"]]
    verdict = {NOT_JUDGED: "Not judged: neither Claim 1 nor Claim 2 passed, so stage 3 does not run for "
                           f"{name}.",
               PENDING: "Waiting for stage 3's twins.",
               SHELVED: "Shelved: the gain is credited to the amount of caution, not the band's placement ("
                        + "; ".join(p["why"] for p in parts.values() if p["badge"] == SHELVED) + ").",
               PASSED: "The gain needs the band's placement: every twin comparison is at least +1.8 return units with "
                       "a lower bound above 0."}[badge]
    labels = [POINTER_LABEL] + ([OVERLAP_LABEL] if parts["2"]["badge"] != NOT_JUDGED else [])
    return _result("6", badge, verdict, reasons, labels, parts=parts)


def claim2_pointer(runs, learner, claim1_delta):
    """Claim 2's pointer (plan section 6.2): the mean change of the dose (IQL: actor weight) from the standard to the
    IW thresholds over kept training rows at the last refresh, times Delta(standard - alone) / (the mean standard dose
    - 1; for IQL the mean standard weight minus Unifloral's own). Each IW run gives {"change", "extra"}; the means are
    over the 5 seeds. It uses outcomes, so it is never a gate."""
    iw = runs.get((learner, "iw"), {})
    vals = [((iw.get(s) or {}).get("pointer") or {}) for s in SEEDS]
    s = claim1_delta["summary"]
    change = [_num(v.get("change")) for v in vals]
    extra = [_num(v.get("extra")) for v in vals]
    out = dict(label=POINTER_LABEL, per_seed=[dict(seed=sd, change=c, extra=e, source=_source(iw.get(sd)))
                                              for sd, c, e in zip(SEEDS, change, extra)], value=None)
    if any(v is None or not math.isfinite(v) for v in change + extra) or not s.get("complete"):
        out["why"] = "needs every IW seed's dose-change readout and a complete Delta(standard - alone)"
        return out
    mc, me = float(np.mean(change)), float(np.mean(extra))
    out.update(mean_change=mc, mean_extra=me, delta_standard_minus_alone=s["mean"])
    if me == 0.0:
        out["why"] = "the mean standard extra caution is 0"
        return out
    out["value"] = mc * s["mean"] / me
    return out


def judge(quality, runs, anchor_none, s7_stopped=None):
    """Every claim of one quality (module docstring). runs: {(learner, setup): {seed: run}}; anchor_none: the
    no-shift bank's anchors dict(random=..., top=...) (start_banks.json "anchors" "none" "anchor"), or None while
    unknown; s7_stopped: the driver's state.json "s7_stopped" entries (version 3; any quality's, this quality's are
    used), or None. Returns dict(schema, quality, claim0_bar, complete, families, learners={learner: {claim id:
    result}}, pointers, exploratory, s7=dict(scopes, excluded))."""
    if quality not in QUALITIES:
        raise ValueError(f"unknown quality {quality!r}")
    scopes = s7_scopes(s7_stopped, quality)
    runs, excluded = without_s7_stopped(runs, scopes)
    expected = [(L, s, sd) for L in LEARNERS for s in CLAIM_SETUPS for sd in SEEDS if not s7_excluded(scopes, L, s)]
    missing = [dict(learner=L, setup=s, seed=sd) for L, s, sd in expected if sd not in runs.get((L, s), {})]
    complete = not missing
    room_present = sum(1 for L in LEARNERS for sd in SEEDS if sd in runs.get((L, "room"), {}))
    room_expected = sum(N_SEEDS for L in LEARNERS if not s7_excluded(scopes, L, "room"))
    res = {L: {} for L in LEARNERS}
    for L in LEARNERS:
        if s7_stops_claim(scopes, L, "0"):
            res[L]["0"] = _s7_result("0", scopes, L, bar=claim0_bar(quality, anchor_none) if anchor_none else None)
        else:
            res[L]["0"] = judge_claim0(quality, runs, L, anchor_none)
        if s7_stops_claim(scopes, L, "4a"):
            res[L]["4a"], res[L]["4b"] = (_s7_result(c, scopes, L, [OVERLAP_LABEL], per_seed=[]) for c in ("4a", "4b"))
        else:
            res[L]["4a"], res[L]["4b"] = judge_coverage(runs, L, res[L]["0"])
    families = {}
    for cid in HOLM_CLAIMS:
        families[cid], out = _family(cid, quality, runs, res, complete, scopes)
        for L in LEARNERS:
            res[L][cid] = out[L]
    pointers = {}
    for L in LEARNERS:
        for cid in ("1c", "2c", "3", "3b"):
            if s7_stops_claim(scopes, L, cid):
                meta = CLAIMS[cid]
                labels = [POINTER_LABEL] + ([OVERLAP_LABEL] if "iw" in (meta["a"], meta["b"]) else [])
                res[L][cid] = _s7_result(cid, scopes, L, labels,
                                         delta=paired(runs, L, meta["a"], meta["b"], meta["bank"]))
            else:
                res[L][cid] = judge_flag(cid, runs, L, res[L]["0"])
        if s7_stops_claim(scopes, L, "5"):
            res[L]["5"] = _s7_result("5", scopes, L, [POINTER_LABEL], by_setup=dict(standard=NOT_JUDGED, iw=NOT_JUDGED),
                                     comparisons=[])
            res[L]["6"] = _s7_result("6", scopes, L, [POINTER_LABEL], parts={
                c: dict(badge=NOT_JUDGED, why=s7_why(scopes, L), deltas=[]) for c in HOLM_CLAIMS})
        else:
            stopped = ("iw",) if L in scopes else ()
            res[L]["5"] = judge_claim5(runs, L, res[L]["0"], stopped, s7_why(scopes, L) if stopped else None)
            res[L]["6"] = judge_claim6(runs, L, res[L]["1"], res[L]["2"])
        if s7_stops_claim(scopes, L, "2"):
            pointers[L] = dict(claim2=dict(label=POINTER_LABEL, per_seed=[], value=None, why=s7_why(scopes, L)))
        else:
            pointers[L] = dict(claim2=claim2_pointer(runs, L, res[L]["1"]["delta"]))
        res[L] = {cid: res[L][cid] for cid in READING_ORDER}
    return dict(schema=SCHEMA, quality=quality, exploratory=True, label=EXPLORATORY_LABEL,
                claim0_bar=claim0_bar(quality, anchor_none) if anchor_none is not None else None,
                complete=dict(claim_setups=complete, missing=missing, room_references_present=room_present,
                              room_references_expected=room_expected,
                              s7_not_expected=[dict(learner=L, setup=s) for L in LEARNERS for s in CLAIM_SETUPS
                                               if s7_excluded(scopes, L, s)],
                              note=None if room_present == room_expected else
                              "the room references run last; one that stops still fails its learner's Claim 0 "
                              f"(plan section 6.1), so Claims 0-2 can change until all {room_expected} are complete"),
                families=families, learners=res, pointers=pointers, s7=dict(scopes=scopes, excluded=excluded))


# ------------------------------------------------------------------------------------------------ views

ROW_CLAIMS = dict(alone=("0",), standard=("1", "1c", "3b", "5", "6"), iw=("2", "2c", "3", "4a", "4b", "5", "6"))


def phone_table(judged, runs):
    """The phone view's 12 rows (plan section 9.1): learner, setup, the strong mix's return and Delta against the
    judged comparison (standard rows: against the learner alone, Claim 1; IW rows: against standard BCA, Claim 2), the
    no-shift return, and a badge per claim with the lower bound beside it and the Holm-adjusted p-value where the claim
    is judged. Every per-seed number carries its run record's and scores file's SHA-256. A setup S7 stopped (version
    3) reads status "stopped by S7" and shows none of its runs' numbers."""
    scopes = (judged.get("s7") or {}).get("scopes") or {}
    runs, _ = without_s7_stopped(runs, scopes)
    rows = []
    for L in LEARNERS:
        res = judged["learners"][L]
        for setup in CLAIM_SETUPS:
            strong, none = (setup_values(runs, L, setup, b) for b in (STRONG, NO_SHIFT))
            badges = {}
            for cid in ROW_CLAIMS[setup]:
                r = res[cid]
                if cid == "5":
                    badges[cid] = r["by_setup"].get(setup, r["badge"])
                elif cid == "6":
                    part = (r.get("parts") or {}).get("1" if setup == "standard" else "2")
                    badges[cid] = part["badge"] if part else r["badge"]
                else:
                    badges[cid] = r["badge"]
            delta = None
            if setup in ("standard", "iw"):
                cid = "1" if setup == "standard" else "2"
                r = res[cid]
                s = r["delta"]["summary"]
                fam = r.get("family") or {}
                delta = dict(claim=cid, comparison=r["delta"]["comparison"], mean=s.get("mean"),
                             lower=s.get("lower"), upper=s.get("upper"), p_one_sided=s.get("p_one_sided"),
                             holm_adjusted_p=fam.get("adjusted") if r["badge"] in (PASSED, SHELVED) else None,
                             badge=r["badge"], labels=r["labels"], per_seed=r["delta"]["per_seed"])
            rows.append(dict(learner=L, learner_name=LEARNER_NAMES[L], setup=setup, setup_name=SETUP_NAMES[setup],
                             status=(S7_STOPPED if s7_excluded(scopes, L, setup) else
                                     DID_NOT_LEARN if res["0"]["badge"] == DID_NOT_LEARN else None),
                             strong=dict(mean=strong["summary"].get("mean"), per_seed=strong["per_seed"]),
                             none=dict(mean=none["summary"].get("mean"), per_seed=none["per_seed"]),
                             delta=delta, badges=badges))
    return rows


def reading_order(judged, integrity=None):
    """The fixed reading order (plan section 6.2): the stop rules S0-S11 (with S6b) first, then Claim 0, 4a/4b, 1, 1c,
    2, 2c, 3, 3b, 5, and 6 only if stage 3 runs. integrity: {rule: {"status": ...}} from export.py."""
    integrity = integrity or {}
    steps = [dict(step="stop rules", rules=list(STOP_RULES),
                  status={r: (integrity.get(r) or {}).get("status", "not recorded") for r in STOP_RULES})]
    for cid in READING_ORDER:
        steps.append(dict(step=cid, title=CLAIMS[cid]["title"],
                          badges={L: judged["learners"][L][cid]["badge"] for L in LEARNERS},
                          note="read only if stage 3 runs" if cid == "6" else None))
    return steps


def study_rule(judged_by_quality):
    """Plan section 6.2's study rule: if Claims 1 and 2 are shelved for all four learners on every quality run, the
    study ends after seeds 0-4, with no stage 3 and no confirmation. A claim that reads "not judged" (did not learn, or
    stopped by S7 under version 3) counts as not passing, like a shelved one. Returns dict(status,
    ends_after_seeds_0_4)."""
    if not judged_by_quality:
        return dict(status=PENDING, ends_after_seeds_0_4=None)
    badges = [j["learners"][L][c]["badge"] for j in judged_by_quality.values() for L in LEARNERS for c in HOLM_CLAIMS]
    if PENDING in badges:
        return dict(status=PENDING, ends_after_seeds_0_4=None, why="Claims 1 and 2 are not yet judged everywhere")
    ends = PASSED not in badges
    return dict(status="judged", ends_after_seeds_0_4=ends,
                why="no Claim 1 or 2 passed on any quality run" if ends else "a Claim 1 or 2 passed")


def stage3_order(judged):
    """Plan section 7.3's order under the stage-3 cap for one quality: Claim 2's flattened and shuffled twins, learner
    by learner in Holm order (smallest p first); then Claim 1's flattened twins in Holm order; then the estimated-weight
    IW arm where Claim 2 passed. A learner's controls run for all 5 seeds or not at all."""
    out = []

    def passing(cid):
        fam = judged["families"][cid]
        if not fam["judged"]:
            return []
        return [row["learner"] for row in fam["holm"] if judged["learners"][row["learner"]][cid]["badge"] == PASSED]

    for L in passing("2"):
        out.append(dict(learner=L, setups=["iw_flattened", "iw_shuffled"], for_claim="2"))
    for L in passing("1"):
        out.append(dict(learner=L, setups=["standard_flattened"], for_claim="1"))
    for L in passing("2"):
        out.append(dict(learner=L, setups=["iw_estimated"], for_claim="2", optional=True))
    return out


# ------------------------------------------------------------------------------------------------ write-ups

def _tested(quality, learner, cid):
    name, q = LEARNER_NAMES[learner], quality
    if cid == "0":
        return (f"{name} alone (Unifloral's unchanged step) trained on the gapped {q} logs, seeds 0-4, and scored by "
                f"exact simulator rollouts on the 1,000 no-shift starts.")
    if cid in ("1", "2"):
        a, b = ("standard BCA", "alone") if cid == "1" else ("IW BCA", "standard BCA")
        return (f"{name} {a} against {name} {b}, trained on the same gapped {q} logs (seeds 0-4) from the same "
                f"starting parameters and batches, each scored on the same 1,000 strong-mix starts; the difference "
                f"is paired by seed.")
    if cid in ("4a", "4b"):
        return (f"{name}'s IW setup at its last refresh (outer step 95,000): the share of strong-law test steps "
                f"outside the ball whose critic error |y - q| falls inside the band, each step at its own threshold "
                f"(4a) or at the standard threshold (4b), pooled as the mean of the 5 seeds.")
    if cid == "6":
        return (f"{name}'s passing setup against its stage-3 twins on the strong mix: the flattened twin (every row "
                f"of a batch gets the batch's mean dose) and, for Claim 2, the shuffled twin (doses permuted within "
                f"the batch).")
    return ""


def _happened(res):
    cid = res["claim"]
    if cid == "0":
        ps = ", ".join("n/a" if p["v"] is None else f"{p['v']:.2f}" for p in res.get("per_seed_return", []))
        bar = res.get("bar") or {}
        return (f"Mean no-shift return {_fmt(res.get('mean_return'), sign=False)} (per seed: {ps}); bar "
                f"{_fmt(bar.get('return_units'), sign=False)} return units ({_fmt(bar.get('points'), 0, False)} "
                f"points).")
    if cid in ("1", "2"):
        s = res["delta"]["summary"]
        ps = ", ".join("n/a" if p["d"] is None else _fmt(p["d"]) for p in res["delta"]["per_seed"])
        fam = res.get("family") or {}
        return (f"Gain {_fmt(s.get('mean'))} return units (per seed: {ps}); lower bound {_fmt(s.get('lower'))}; "
                f"one-sided p {_fmt_p(s.get('p_one_sided'))}; Holm-adjusted p {_fmt_p(fam.get('adjusted'))}.")
    if cid in ("4a", "4b"):
        key = "own" if cid == "4a" else "standard"
        ps = ", ".join("n/a" if p[key] is None else f"{100 * p[key]:.2f}%" for p in res.get("per_seed", []))
        pooled = res.get("pooled")
        return f"Pooled coverage {'n/a' if pooled is None else f'{100 * pooled:.2f}%'} (per seed: {ps})."
    if cid == "6":
        bits = []
        for c, part in (res.get("parts") or {}).items():
            for d in part.get("deltas", []):
                s = d["summary"]
                bits.append(f"Delta({d['comparison']}) {_fmt(s.get('mean'))}, lower bound {_fmt(s.get('lower'))}")
        return "; ".join(bits) + "."
    return ""


def writeups(judged):
    """The user's shelving format for every shelved claim and every "did not learn" (plan section 6.2, Write-ups):
    The idea / What we tested (and the kill rule) / What happened / Why. "Why" here is the rule that shelved it, read
    from the numbers; a causal explanation, if any, is added by whoever writes the results up. Each item has the
    four parts, the result's path in results.json, and a Markdown rendering. Under version 3 every scope S7 stopped
    follows (s7_writeups)."""
    q, out = judged["quality"], []
    for L in LEARNERS:
        for cid in READING_ORDER:
            res = judged["learners"][L][cid]
            if res["badge"] not in (SHELVED, DID_NOT_LEARN) or cid not in IDEAS:
                continue
            name = LEARNER_NAMES[L]
            cite = f"results.json:qualities.{q}.claims.{L}.{cid}"
            idea = IDEAS[cid].format(learner=name, quality=q)
            tested = _tested(q, L, cid) + f" The kill rule, frozen before any run (plan section 6.2), shelves it if " \
                                          f"{KILL_RULES[cid]}."
            happened = _happened(res) + f" [{cite}]"
            why = list(res["reasons"])
            if cid in ("1", "2", "6"):
                why.append(POWER_NOTE)
            title = f"{name} on {q}, Claim {cid}: {res['title']} - {res['badge'].upper()}"
            md = "\n".join([f"### {title}", "", f"- **The idea:** {idea}",
                            f"- **What we tested:** {tested}", f"- **What happened:** {happened}", "- **Why:**"]
                           + [f"  - {w[0].upper() + w[1:]}." if not w.endswith(".") else f"  - {w[0].upper() + w[1:]}"
                              for w in why])
            out.append(dict(quality=q, learner=L, claim=cid, badge=res["badge"], title=title, idea=idea,
                            tested=tested, happened=happened, why=why, cite=cite, markdown=md))
    return out + s7_writeups(judged)


def s7_writeups(judged):
    """Version 3: every scope S7 stopped in this quality, in the same format (The idea / What we tested / What
    happened / Why), with claim "S7" and badge "stopped by S7", citing results.json's
    qualities.<quality>.s7_stopped.scopes.<learner>."""
    q, out = judged["quality"], []
    scopes = (judged.get("s7") or {}).get("scopes") or {}
    for L in LEARNERS:
        if L not in scopes:
            continue
        sc, name = scopes[L], LEARNER_NAMES[L]
        cite = f"results.json:qualities.{q}.s7_stopped.scopes.{L}"
        learner_scope = sc["scope"] == "learner"
        idea = (f"The scores must measure {name}'s trained policy: the float64 scoring adapter and a float32 "
                f"restatement of the trained network, each driving the simulator from the same start, should agree "
                f"within 0.05 points on every start (S7, with version 2's post-hoc limit of 0.05 points "
                f"(PRESPEC.md section 19)).")
        tested = (f"S7 on every stage-1 actor and every seed-0 final actor of {name}'s alone, standard and IW setups, "
                  f"on the 1,000 starts of each of the 7 banks. Version 3's rule (post hoc, PRESPEC.md section 20): a "
                  f"failure on the alone, standard or weights-of-1 actor stops the learner in this quality; a failure "
                  f"on an IW actor (or IW's repeat) stops its IW setup.")
        happened = (f"The float32 check failed on {', '.join(sc['runs']) or 'a run'}, so "
                    f"{'the learner' if learner_scope else 'its IW setup'} stopped on {q}. [{cite}]")
        claims_out = "Claims 1 and 2" if learner_scope else "Claim 2"
        why = [s7_why(scopes, L),
               f"no claim of {'the learner' if learner_scope else 'the IW setup'} is judged on {q}, and it stays in "
               f"Holm's family of {claims_out} with p = 1, so it never loosens the bar for the other learners",
               "S7's limit is unchanged (0.05 points per start); its runs trained before the stop enter no claim, "
               "statistic or mean"]
        title = f"{name} on {q}: {'learner' if learner_scope else 'IW setup'} stopped by S7"
        md = "\n".join([f"### {title}", "", f"- **The idea:** {idea}", f"- **What we tested:** {tested}",
                        f"- **What happened:** {happened}", "- **Why:**"]
                       + [f"  - {w[0].upper() + w[1:]}." for w in why])
        out.append(dict(quality=q, learner=L, claim="S7", badge=S7_STOPPED, title=title, idea=idea, tested=tested,
                        happened=happened, why=why, cite=cite, markdown=md))
    return out
