"""Judge BCA's margin against experiments/MARGIN_VALIDATION_CRITERIA.md (version 1). NumPy only.
Reads saved files, trains nothing, changes no saved file.

    python -I -B margin_verdict.py --baselines <results label of a finished standard-learner run>

Section numbers below are the criteria's. The criteria fix what is computed. This file fixes how.

Two modes, chosen by the data and never by a flag.
  verdict   every candidate dataset seed is fresh. Check S1 is read and one of the three verdict words is given.
            Written once: results/<label>/margin_verdict.json and .md, and a receipt that spends the datasets.
  seen      the datasets were seen before (reading 4 below). Each mark says "meets" or "does not meet on seen
            data", and no verdict is given (section 7). Written to results/<label>/margin_seen_reading.json and .md.

The script opens no run unless the freeze of section 10, rule 2 holds: the criteria and this file are committed
and unchanged, and so is every script the datasets and runs are made with (reading 3 below).

What a run must have saved in results/<label>/:
  START.json                     data_label, alpha, and for a verdict the entry "margin_validation" (see check_s1)
  baselines_<task>.json, .npz    per run: recipe, data_seed, train_seed, index, alpha, final_step, reference,
                                 audit_bank; arrays [run, cell, command]: commands, q1, q2, width, support
  datasets_<task>.json           the array fingerprint of every dataset the run trained on
  audit_rows_<task>.npz          arrays [run, row]: cell, command, target, q1, q2, residual, margin. Needed for
                                 mark M2 and for a verdict. Runs before 2026-11-14 did not save it.
together with receipts/<data_label>_manifest.json and data/<data_label>/.

Readings fixed here because the criteria are silent or can be read two ways. Each is the literal, the strict or
the plain one.
  1  A fingerprint is the SHA-256 of a file with Windows line endings read as plain ones, so every checkout of
     the same text gives the same code.
  2  S1, "the commit is older than the receipt": the commit the receipt names holds both files with the recorded
     fingerprints, and its commit time is before the receipt's creation time. If git cannot answer, S1 fails.
  3  Section 10, rule 2, "the dataset builder and the run scripts": every Python file in this folder (not in its
     subfolders), the shared learner and the three calibration files. The script runs only if none of them is new
     or changed since the last commit.
  4  A block counts as seen if a candidate seed is in SPENT_SEEDS, or if this script has already written a verdict
     for its datasets: a verdict receipt (receipts/<data_label>_margin_verdict.json) exists for its data label, or
     names one of its seeds. A reading on seen data is written to its own files and never replaces a verdict.
  5  A verdict block must have exactly fifteen candidates and the saved audit rows of every half-random run. On
     seen data too, every dataset that passed the acceptance checks must have three half-random and three
     best-policy runs. Otherwise nothing is judged and nothing is written. Random-starts runs are read as they
     are: a dataset with none is left out of the lava outcome.
  6  A run whose margin was not ready at its last refit (20,000 updates) has a margin of zero everywhere and is
     judged as it is.
  7  Mark M1 is recomputed from the saved audit rows and must equal the run's own figure. On seen data without
     audit rows the run's own figure is used and M2 is not read.
  8  M2: a class that fails on nine dataset values fails the mark, even if the other class has no value. If no
     class fails and one has no value, M2 is not judged.
  9  M1's "above 0.95" is read on the average of the nine dataset values.
  10 M3 to M5: a command whose error size has no spread scores 0, like a margin with no spread.
  11 M3, last clause: "the block's in-range runs" are the in-range runs of the nine datasets used, and one that
     gives no value counts as not above 0.
  12 An allowance of 1e-12 is used in four places and nowhere else. A choice between values within 1e-12 of each
     other goes to the first of stay, up, down, left, right, as in the environment. A difference within 1e-12 of
     zero is a tie in "holds", and a tie counts against. The bars of the marks and of S6 ("or more", "at most",
     "above 0.95") allow 1e-12. So does "within 0.05". Everything else is compared exactly, as stored: the 0.90 of
     S3, the 1.5 of "in range", "inside the margin" (the error's size is at most the margin), ties between ranks,
     a rival signal that is zero, and the readings of sections 8 and 9 that have no pass mark.
  13 The shuffled-cell copy draws, for each of the 1,000 reorderings in turn, one order for each command in the
     order stay, up, down, left, right, from numpy's default generator seeded with [dataset seed, training seed].
     Every signal of a run meets the same reorderings.
  14 When S6 fails, M7's guard is read by the letter: its fail counts only if the reference margin of S7 passed M7
     on both clauses.
  15 Fixed sentences. A bracket is filled in with no added punctuation. For M6 it is "at all" when the first half
     of M6 fails, whatever the second half does. For M7 it is "more than 1%" when the first clause fails, whatever
     the second does. When both classes of M2 fail, one sentence is printed for each. When M6 and M7 both fail, the
     sentence that precedes them is printed once. A mark whose figures failed but which was not judged because S6
     failed is listed apart, like one excused by S5 or S7.
  16 The validated sentence counts the 27 runs of the nine datasets used. If none left the range it says "No run
     left the range." in place of its second sentence. It always says "Validated once": "replicated" is stated by
     whoever holds two passing blocks.
  17 The out-of-range line of section 8 counts every half-random run of the block, runs on datasets that were
     dropped included. With none it reads "No run's critic left the possible range."
  18 On seen data each mark says "meets on seen data" or "does not meet on seen data" by its own pass rule, with
     the guards of section 5 left out, or "not read" where it has no value. The reference margins are printed
     beside it.
  19 The lower-bound share and the strict reading are taken over recorded commands at visited cells. The constant
     margin beside them is the average of the margin over those commands.
  20 "The tenth of visited cells" is rounded up. The ten-cell rule, the three-command rule and the weights are
     then applied to the cells that remain.
  21 Sensitivity: with the class line at 8 or at 32 rows, a visited cell is one with 8 or 32 rows too.
  22 "The M3 to M7 figures of both reference margins": the reference margin of S5 is scaled like a rival on M6 and
     M7, and the reference margin of S7 on M3 to M5 is the error size against itself.
  23 A rival is compared with the margin only in datasets where both have a value.
  24 The lava outcome (section 9) is read on the nine datasets used, for the standard learner: mark L0 and the
     readings beside it. A dataset's share is the average over all its runs of the recipe, in range or not. A
     case is one run at one cell with 16 or more rows, cases are summed over the same runs, and "any penalty" is
     3 of the margins the run fitted. Marks L1 to L4 need a trained method and are not computed here.
"""
import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import critical_env as E  # noqa: E402
import published_collect as C  # noqa: E402

CRITERIA_VERSION = "version 1, 2026-11-14"
FROZEN = {"criteria": ROOT / "experiments" / "MARGIN_VALIDATION_CRITERIA.md", "verdict_script": Path(__file__).resolve()}
RECORD_KEY = "margin_validation"      # the entry in the datasets' receipt and in a run's start record
# Published-gridworld dataset seeds that already exist (MARGIN_VALIDATION_DECISIONS.md, section 2).
SPENT_SEEDS = frozenset({7, *range(6101, 6104), *range(6301, 6306), *range(6401, 6406), *range(6501, 6516)})
# Section 10, rule 2: what the datasets and the runs are made with. All of it must be committed and unchanged.
RULE_2_PATHS = (":(glob)experiments/gridworld_critical_v1/*.py", "experiments/gridworld_ood_v1/learner.py",
                "calibration/network.py", "calibration/reference.py", "calibration/wbcp.py")

HALF_RANDOM, BEST_POLICY, RANDOM_STARTS = "mixed", "optimal", "other_starts"
LINE, TOL, KAPPA, RANGE = 16, 0.05, 3.0, 1.5                 # class line, right or wrong, multiplier, in range
DATASETS, NEEDED, P_MARK, CANDIDATES = 9, 6, 0.05, 15        # "holds" and the selection rule
RUNS_PER_DATASET, RUNS_IN_RANGE_NEEDED = 3, 2                # S4
ALPHA, FINAL_STEP, AUDIT_ROWS, OPEN_CELLS = 1.0, 25_000, 1280, 71
THIN_COMMANDS_NEEDED, THIN_CELLS_NEEDED, PASS_SCORE = 50, 20, 0.90          # S2, S3
COVER_FLOOR, COVER_MEAN, COVER_WIDE = 0.85, 0.88, 0.95       # M1, M2
TRACK_BAR, RUNS_ABOVE = 0.3, 2.0 / 3.0                       # M3
MIN_CELLS, MIN_THIN_CELLS, MIN_COMMANDS = 10, 5, 3           # M3 to M5
STILL_WRONG_SHARE, COPY_SHARE, BROKEN_SHARE = 0.5, 0.75, 0.01               # M6, M7
ROOM_SHARE, ROOM_DATASETS = 0.05, 8                          # S6
REORDERINGS, CONSTANT_QUANTILE = 1000, 0.90
SENSITIVITY_TOLERANCES, SENSITIVITY_LINES = (0.02, 0.10), (8, 32)
LAVA_ROOM = 1.0                                              # L0, in percentage points
EPS, NO_SPREAD = 1e-12, 1e-9
AUDIT_FIELDS = ("cell", "command", "target", "q1", "q2", "residual", "margin")
MARKS = ("M1", "M2", "M3", "M4", "M5", "M6", "M7")


class Defect(Exception):
    """The saved files are not what the criteria describe. Nothing is judged."""


# ---- fixing the text (section 10, check S1) ----------------------------------------------------------------
def fingerprint_bytes(raw):
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def fingerprint(path):
    return fingerprint_bytes(Path(path).read_bytes())


def git(*args):
    """Output of a git command in this repository, or None if git cannot answer."""
    try:
        done = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def committed_fingerprint(path, commit="HEAD"):
    raw = git("show", f"{commit}:{Path(path).resolve().relative_to(ROOT).as_posix()}")
    return None if raw is None else fingerprint_bytes(raw)


def text_is_committed():
    """Section 10, rule 2: are the criteria and this file committed, and unchanged since?"""
    out = {}
    for name, path in FROZEN.items():
        now = fingerprint(path)
        out[name] = {"fingerprint": now, "committed": committed_fingerprint(path) == now}
    out["ok"] = all(entry["committed"] for entry in out.values())
    return out


def scripts_not_committed():
    """The testbed's Python files, the shared learner and the calibration files that are new or changed since the
    last commit. None if git cannot answer."""
    raw = git("status", "--porcelain", "--untracked-files=all", "--", *RULE_2_PATHS)
    if raw is None:
        return None
    return [line[3:].strip() for line in raw.decode("utf-8", "replace").splitlines() if line.strip()]


def freeze_state():
    """Section 10, rule 2 as a whole: the two fixed texts, and what the datasets and runs are made with."""
    state, loose = text_is_committed(), scripts_not_committed()
    return dict(state, scripts_not_committed=loose, ok=bool(state["ok"] and loose == []))


def verdict_receipt(here, data_label):
    """Written beside the datasets' receipt when a block is judged. Its datasets are spent from then on."""
    return Path(here) / "receipts" / f"{data_label}_margin_verdict.json"


def judged_seeds(here):
    """The dataset seeds of every block this script has judged, read from the verdict receipts."""
    seeds = set()
    for path in sorted((Path(here) / "receipts").glob("*_margin_verdict.json")):
        try:
            seeds.update(int(seed) for seed in json.loads(path.read_text()).get("candidates", []))
        except (OSError, ValueError, TypeError, AttributeError):
            raise Defect(f"the verdict receipt {path.name} cannot be read") from None
    return seeds


def record_for_receipt():
    """What the dataset builder writes into its receipt, and a run into its start record, under RECORD_KEY."""
    state = freeze_state()
    head = git("rev-parse", "HEAD")
    if not state["ok"] or head is None:
        raise Defect("the criteria, the verdict script, the dataset builder and the run scripts must be committed "
                     "and unchanged before a block is built")
    return {"criteria_sha256": state["criteria"]["fingerprint"],
            "verdict_script_sha256": state["verdict_script"]["fingerprint"], "commit": head.decode().strip()}


def check_s1(manifest, start, folder):
    """Check S1: the text was fixed before the block's datasets existed."""
    mine = {name: fingerprint(path) for name, path in FROZEN.items()}
    receipt, record, reasons = manifest.get(RECORD_KEY), start.get(RECORD_KEY), []
    for where, entry in (("the datasets' receipt", receipt), ("the run's start record", record)):
        if not isinstance(entry, dict):
            reasons.append(f"{where} holds no fingerprints")
            continue
        reasons += [f"{where} holds a different fingerprint for the {name.replace('_', ' ')}"
                    for name in FROZEN if entry.get(f"{name}_sha256") != mine[name]]
    for name, path in FROZEN.items():
        copy = Path(folder) / path.name
        if not copy.exists() or fingerprint(copy) != mine[name]:
            reasons.append(f"no matching copy of {path.name} is saved with the results")
    commit = receipt.get("commit") if isinstance(receipt, dict) else None
    if not commit:
        reasons.append("the datasets' receipt names no commit")
    else:
        reasons += [f"commit {commit[:8]} does not hold this {name.replace('_', ' ')}"
                    for name, path in FROZEN.items() if committed_fingerprint(path, commit) != mine[name]]
        stamp, created = git("show", "-s", "--format=%ct", commit), manifest.get("created_unix")
        if stamp is None or created is None or not float(stamp.decode().strip()) < float(created):
            reasons.append("the commit is not shown to be older than the datasets' receipt")
    return {"passed": not reasons, "reasons": reasons, "fingerprints": mine}


# ---- small statistics ---------------------------------------------------------------------------------------
def at_least(value, bar):
    return bool(value >= bar - EPS)


def at_most(value, bar):
    return bool(value <= bar + EPS)


def complete(values):
    return len(values) == DATASETS and all(v is not None for v in values)


def sign_flip_p(differences):
    """Exact one-sided p: the share of all ways of flipping the signs whose sum is at least the observed sum."""
    observed = sum(differences)
    patterns = itertools.product((1.0, -1.0), repeat=len(differences))
    as_large = sum(sum(s * d for s, d in zip(signs, differences)) >= observed - EPS for signs in patterns)
    return as_large / 2 ** len(differences)


def holds(differences):
    """The rule of Amendment 5 for nine differences: above zero in at least 6, and p of 0.05 or less. A tie counts against."""
    d = [0.0 if abs(x) <= EPS else float(x) for x in differences]
    above, p = int(sum(x > 0 for x in d)), sign_flip_p(d)
    return {"differences": d, "datasets_above": above, "p": p,
            "holds": bool(len(d) == DATASETS and above >= NEEDED and p <= P_MARK)}


def average_ranks(values):
    """Ranks from 1, tied values sharing their average rank."""
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    ranked, ranks, start = values[order], np.empty(len(values)), 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and ranked[stop + 1] == ranked[start]:
            stop += 1
        ranks[order[start:stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def pearson(x, y):
    """Ordinary correlation. 0 if either side has no spread."""
    x, y = np.asarray(x, float) - np.mean(x), np.asarray(y, float) - np.mean(y)
    sx, sy = float(np.sqrt(np.sum(x * x))), float(np.sqrt(np.sum(y * y)))
    if sx <= NO_SPREAD or sy <= NO_SPREAD:
        return 0.0
    return float(np.clip(np.sum(x * y) / (sx * sy), -1.0, 1.0))


def spearman(signal, size):
    return pearson(average_ranks(signal), average_ranks(size))


def leftover(y, x):
    """What is left of y after a straight line on x is taken out."""
    x0, y0 = x - x.mean(), y - y.mean()
    spread = float(np.sum(x0 * x0))
    return y0 - (0.0 if spread <= NO_SPREAD else float(np.sum(x0 * y0)) / spread) * x0


def partial_spearman(signal, size, held):
    """M4: rank correlation of signal and size with `held` held fixed. The leftovers are not ranked again."""
    held_ranks = average_ranks(held)
    return pearson(leftover(average_ranks(signal), held_ranks), leftover(average_ranks(size), held_ranks))


# ---- cells and pairs (section 2) -------------------------------------------------------------------------------
def pair_counts(cells, moves):
    """[cell, move]: how many rows have that move at that cell."""
    counts = np.zeros((E.SIZE * E.SIZE, E.N_MOVES), int)
    np.add.at(counts, (np.asarray(cells, int), np.asarray(moves, int)), 1)
    return counts


def support_classes(counts, line=LINE):
    """From the training rows per pair: visited cells, and recorded, well recorded, thinly recorded and
    never-commanded pairs at visited cells. `several`: visited cells with two or more recorded commands.
    `eligible`: those with at least one thinly recorded command (M6)."""
    counts = np.asarray(counts)
    visited = counts.sum(axis=1) >= line
    recorded = (counts > 0) & visited[:, None]
    thin = recorded & (counts < line)
    several = visited & (recorded.sum(axis=1) >= 2)
    return {"line": line, "counts": counts, "visited": visited, "recorded": recorded, "well": counts >= line,
            "thin": thin, "never": (counts == 0) & visited[:, None], "several": several,
            "eligible": several & thin.any(axis=1)}


def in_range(q1, free):
    """A run is in range when the first critic is within 1.5 in size at every open cell and certain command."""
    return bool(np.abs(q1[free]).max() <= RANGE)


# ---- M1, M2: coverage of the one-step error --------------------------------------------------------------------
def audit_coverage(rows, cls):
    """One run's audit rows: the share whose one-step error is inside the margin, overall and by class, and the
    two class shares for one constant margin set at the 90th percentile of the error size (section 8)."""
    size = np.abs(rows["residual"])
    inside, flat = size <= rows["margin"], size <= np.quantile(size, CONSTANT_QUANTILE)
    cell, command = np.asarray(rows["cell"], int), np.asarray(rows["command"], int)
    thin, well = cls["thin"][cell, command], cls["well"][cell, command]

    def share(mask, hit):
        return float(hit[mask].mean()) if mask.any() else None

    return {"rows": int(len(size)), "overall": float(inside.mean()), "thin": share(thin, inside),
            "well": share(well, inside), "thin_rows": int(thin.sum()), "well_rows": int(well.sum()),
            "rows_in_neither_class": int((~thin & ~well).sum()),
            "constant_thin": share(thin, flat), "constant_well": share(well, flat)}


# ---- M3 to M5: is the signal larger where the critic is more wrong? ---------------------------------------------
def tracking(signal, size, truth, cls, keep=None):
    """M3, M4 and M5 for one run and one signal. Per command, a rank correlation across the visited cells where
    the command is recorded (M3, M4) or thinly recorded (M5), between the signal and the size of the true error.
    Commands are averaged with weights equal to their cell counts. None: fewer than three commands qualify."""
    keep = np.ones(signal.shape[0], bool) if keep is None else keep
    out = {}
    for mark, pairs, least in (("M3", cls["recorded"], MIN_CELLS), ("M4", cls["recorded"], MIN_CELLS),
                               ("M5", cls["thin"], MIN_THIN_CELLS)):
        scores, weights = [], []
        for command in range(E.N_MOVES):
            cells = np.flatnonzero(pairs[:, command] & keep)
            if len(cells) < least:
                continue
            s, e = signal[cells, command], size[cells, command]
            scores.append(partial_spearman(s, e, truth[cells, command]) if mark == "M4" else spearman(s, e))
            weights.append(len(cells))
        out[mark] = float(np.average(scores, weights=weights)) if len(scores) >= MIN_COMMANDS else None
    return out


def without_worst_tenth(size, cls):
    """Cells kept after leaving out the tenth of visited cells whose error size, averaged over their recorded
    commands, is largest (section 8, "a few cells")."""
    cells = np.flatnonzero(cls["visited"])
    level = np.array([size[cell][cls["recorded"][cell]].mean() for cell in cells])
    keep = np.ones(size.shape[0], bool)
    keep[cells[np.argsort(-level, kind="mergesort")[:int(np.ceil(len(cells) / 10))]]] = False
    return keep


# ---- M6, M7: does the charge correct wrong choices without breaking right ones? ----------------------------------
def choose(values, cls):
    """Per cell, the recorded command with the highest value. Values within 1e-12 of the highest are equal, and
    the first of them is chosen."""
    values = np.where(cls["recorded"], values, -np.inf)
    return (values >= values.max(axis=-1, keepdims=True) - EPS).argmax(axis=-1)


def wrong(choice, truth, cls, tol):
    """A choice is wrong when its true value is more than `tol` below the best recorded command's."""
    best = np.where(cls["recorded"], truth, -np.inf).max(axis=1)
    return truth[np.arange(truth.shape[0]), choice] < best - tol - EPS


def decisions(value, charge, truth, cls, tol=TOL):
    """Counts of cases in one run. `value` is the critic, `charge` what is subtracted from it (3 margins)."""
    cells = np.arange(value.shape[0])
    raw, after = choose(value, cls), choose(value - charge, cls)
    raw_wrong, after_wrong = wrong(raw, truth, cls, tol), wrong(after, truth, cls, tol)
    raw_thin = cls["thin"][cells, raw]
    misled = cls["eligible"] & raw_thin & raw_wrong
    right_well = cls["several"] & cls["well"][cells, raw] & ~raw_wrong
    right_thin = cls["eligible"] & raw_thin & ~raw_wrong
    return {"eligible": int(cls["eligible"].sum()), "misled": int(misled.sum()),
            "still_wrong": int((misled & after_wrong).sum()), "corrected": int((misled & ~after_wrong).sum()),
            "moved_to_a_wrong_well_recorded_command": int((misled & after_wrong & cls["well"][cells, after]).sum()),
            "right_well_recorded": int(right_well.sum()), "broken": int((right_well & after_wrong).sum()),
            "right_thin": int(right_thin.sum()), "right_thin_broken": int((right_thin & after_wrong).sum())}


def reorderings(cls, seed_words, count=REORDERINGS):
    """For each command, `count` random orders of the visited cells where it is recorded (the shuffled-cell copy)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(word) for word in seed_words]))
    cells = [np.flatnonzero(cls["recorded"][:, command]) for command in range(E.N_MOVES)]
    orders = [np.empty((count, len(c)), int) for c in cells]
    for r in range(count):
        for command in range(E.N_MOVES):
            orders[command][r] = rng.permutation(cells[command])
    return cells, orders


def shuffled_copies(signal, cells, orders):
    """[reordering, cell, command]: the signal with each command's values moved among that command's cells."""
    signal = np.asarray(signal, float)
    copies = np.repeat(signal[None], len(orders[0]), axis=0)
    for command in range(E.N_MOVES):
        copies[:, cells[command], command] = signal[orders[command], command]
    return copies


def still_wrong_after_copies(value, copies, truth, cls, tol=TOL):
    """The number of misled cases still wrong after each copy of the charge, averaged over the copies."""
    cells = np.arange(value.shape[0])
    raw = choose(value, cls)
    misled = cls["eligible"] & cls["thin"][cells, raw] & wrong(raw, truth, cls, tol)
    after = choose(value[None] - copies, cls)
    return float((misled[None] & wrong(after, truth, cls, tol)).sum(axis=1).mean())


def decision_reading(value, charge, truth, cls, tol, shuffle):
    out = decisions(value, charge, truth, cls, tol)
    out["still_wrong_after_copy"] = still_wrong_after_copies(value, shuffled_copies(charge, *shuffle), truth, cls, tol)
    return out


def scaled_charge(signal, margin, cls):
    """A rival signal scaled so that its average over the run's recorded commands at visited cells equals the
    average of 3 margins over the same commands. A signal that is zero at all of them charges nothing."""
    level = float(signal[cls["recorded"]].mean())
    if not level > 0:
        return np.zeros_like(signal, dtype=float)
    return signal * (KAPPA * float(margin[cls["recorded"]].mean()) / level)


# ---- section 8: readings with no pass mark ----------------------------------------------------------------------
def bound_readings(value, margin, truth, cls):
    """The lower-bound share (with 3 margins, with none, with a constant one) and the strict reading, by class."""
    level, out = float(margin[cls["recorded"]].mean()), {}
    for name, mask in (("recorded", cls["recorded"]), ("thin", cls["thin"]), ("well", cls["well"])):
        if not mask.any():
            out[name] = None
            continue
        q, m, t = value[mask], margin[mask], truth[mask]
        out[name] = {"pairs": int(mask.sum()), "lower_bound": float((q - KAPPA * m <= t).mean()),
                     "lower_bound_no_margin": float((q <= t).mean()),
                     "lower_bound_constant_margin": float((q - KAPPA * level <= t).mean()),
                     "error_inside_one_margin": float((np.abs(q - t) <= m).mean())}
    return out


def never_commanded_reading(margin, cls):
    """The margin at never-commanded commands against the margin at recorded commands of the same cell."""
    cells = np.flatnonzero(cls["never"].any(axis=1) & cls["recorded"].any(axis=1))
    if not len(cells):
        return None
    there = np.array([margin[cell][cls["never"][cell]].mean() for cell in cells])
    recorded = np.array([margin[cell][cls["recorded"][cell]].mean() for cell in cells])
    return {"cells": int(len(cells)), "margin_at_never_commanded": float(there.mean()),
            "margin_at_recorded": float(recorded.mean()),
            "share_of_cells_larger_at_never_commanded": float((there > recorded).mean())}


# ---- section 9: the outcome for never-commanded lava moves -------------------------------------------------------
def lava_reading(task, policy, counts, executed, q1=None, margin=None, line=LINE):
    """In percentage points: the share of episodes from the start cell that end in lava on a step where the policy
    commanded a move into lava that no training row commands at that cell. `counted`: cells with `line` or more
    training rows (the outcome). Exact, over 200 steps. Beside it: the same at cells with fewer rows, and the split
    of the outcome between lava moves never executed in the data and those executed through the noise."""
    size = E.SIZE * E.SIZE
    visited = counts.sum(axis=1) >= line
    step = {name: np.zeros(size) for name in ("counted", "cells_with_few_rows", "never_executed", "executed_through_noise")}
    cases = above_before = above_after = 0
    for cell in task.free_cells():
        into_lava = np.array([move in task.legal_moves(cell) and task.destination(cell, move) in task.lava
                              for move in range(E.N_MOVES)])
        if not into_lava.any():
            continue
        chance = task.command_probabilities(policy[cell])
        lands = task.execution(cell) @ into_lava.astype(float)        # per command: the step ends in some lava cell
        never = np.flatnonzero(into_lava & (counts[cell] == 0))
        for move in never:
            ends = chance[move] * lands[move]
            if not visited[cell]:
                step["cells_with_few_rows"][cell] += ends
                continue
            step["counted"][cell] += ends
            step["never_executed" if executed[cell, move] == 0 else "executed_through_noise"][cell] += ends
        if q1 is not None and visited[cell] and len(never):                # a case is one run at one cell
            recorded, charged = counts[cell] > 0, q1[cell] - KAPPA * margin[cell]
            cases += 1
            above_before += bool(q1[cell][never].max() > q1[cell][recorded].max())
            above_after += bool(charged[never].max() > charged[recorded].max())
    inner = task.chain(policy)
    inner[:, list(task.terminals)] = 0.0
    alive, total = np.zeros(size), {name: 0.0 for name in step}
    alive[task.start] = 1.0
    for _ in range(E.HORIZON):
        for name in step:
            total[name] += float(alive @ step[name])
        alive = alive @ inner
    out = {name: 100.0 * value for name, value in total.items()}
    if q1 is not None:
        out.update(cells_with_a_never_commanded_lava_move=cases,
                   critic_rates_one_above_every_recorded_command=int(above_before),
                   the_same_after_3_margins=int(above_after))
    return out


# ---- reading one run, one dataset -----------------------------------------------------------------------------
TRACKED = ("margin", "reference_s5", "reference_s7", "support_reversed", "critic_gap", "critic_size",
           "margin_second_critic",
           "margin_lower_critic", "margin_without_worst_tenth")
CHARGED = ("margin", "reference_s7", "below_floor", "critic_gap", "reference_s5", "margin_second_critic",
           "margin_lower_critic", *(f"margin_tolerance_{tol:g}" for tol in SENSITIVITY_TOLERANCES),
           *(f"margin_line_{line}" for line in SENSITIVITY_LINES))
RIVALS_TRACKING = {"support_reversed": "the support score", "critic_gap": "the difference between the two critics",
                   "reference_s5": "a margin that only counts rows", "critic_size": "the size of the critic's own value"}
RIVALS_CHARGED = {"below_floor": "the support score", "critic_gap": "the difference between the two critics",
                  "reference_s5": "a margin that only counts rows"}
COVERAGE_KEYS = ("overall", "thin", "well", "constant_thin", "constant_well")
BOUND_KEYS = ("lower_bound", "lower_bound_no_margin", "lower_bound_constant_margin", "error_inside_one_margin")
LAVA_CASES = ("cells_with_a_never_commanded_lava_move", "critic_rates_one_above_every_recorded_command",
              "the_same_after_3_margins")
COUNT_KEYS = ("eligible", "misled", "still_wrong", "corrected", "moved_to_a_wrong_well_recorded_command",
              "right_well_recorded", "broken", "right_thin", "right_thin_broken", "still_wrong_after_copy")


def read_run(run, task, counts, free):
    """Everything the marks and section 8 need from one half-random run."""
    truth = task.command_values(task.value(run["commands"]))
    cls, q1, q2, margin = support_classes(counts), run["q1"], run["q2"], run["margin"]
    lower, size, words = np.minimum(q1, q2), np.abs(q1 - truth), (run["data_seed"], run["train_seed"])
    reference = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)        # S5
    gap = np.abs(q1 - q2)
    below_floor = np.maximum(run["support_floor"] - run["support"], 0.0)

    if run["audit"] is not None:
        coverage = audit_coverage(run["audit"], cls)
    else:
        coverage = {key: None for key in COVERAGE_KEYS}
        coverage["overall"] = float(run["saved_coverage"])

    tracked = {name: tracking(signal, size, truth, cls) for name, signal in (
        ("margin", margin), ("reference_s5", reference), ("reference_s7", size), ("support_reversed", -run["support"]),
        ("critic_gap", gap),
        ("critic_size", np.abs(q1)))}
    tracked["margin_second_critic"] = tracking(margin, np.abs(q2 - truth), truth, cls)
    tracked["margin_lower_critic"] = tracking(margin, np.abs(lower - truth), truth, cls)
    tracked["margin_without_worst_tenth"] = {"M3": tracking(margin, size, truth, cls, without_worst_tenth(size, cls))["M3"]}

    shuffle = reorderings(cls, words)
    charged = {name: decision_reading(q1, charge, truth, cls, TOL, shuffle) for name, charge in (
        ("margin", KAPPA * margin), ("reference_s7", KAPPA * size), ("below_floor", scaled_charge(below_floor, margin, cls)),
        ("critic_gap", scaled_charge(gap, margin, cls)), ("reference_s5", scaled_charge(reference, margin, cls)))}
    charged["margin_second_critic"] = decision_reading(q2, KAPPA * margin, truth, cls, TOL, shuffle)
    charged["margin_lower_critic"] = decision_reading(lower, KAPPA * margin, truth, cls, TOL, shuffle)
    for tol in SENSITIVITY_TOLERANCES:
        charged[f"margin_tolerance_{tol:g}"] = decision_reading(q1, KAPPA * margin, truth, cls, tol, shuffle)
    for line in SENSITIVITY_LINES:
        other = support_classes(counts, line)
        charged[f"margin_line_{line}"] = decision_reading(q1, KAPPA * margin, truth, other, TOL, reorderings(other, words))

    return {"data_seed": run["data_seed"], "train_seed": run["train_seed"], "in_range": in_range(q1, free),
            "largest_critic_value": float(np.abs(q1[free]).max()), "margin_was_ready": bool(run["reference_ready"]),
            "coverage": coverage, "tracking": tracked, "decisions": charged,
            "bounds": bound_readings(q1, margin, truth, cls), "never_commanded": never_commanded_reading(margin, cls)}


def average(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def pooled(readings):
    """Counts summed over a dataset's in-range runs, and the shares the marks use."""
    total = {key: sum(r[key] for r in readings) for key in COUNT_KEYS}

    def share(top, bottom):
        return total[top] / total[bottom] if total[bottom] else None

    return dict(total, misled_share=share("misled", "eligible"), still_wrong_share=share("still_wrong", "eligible"),
                copy_still_wrong_share=share("still_wrong_after_copy", "eligible"),
                broken_share=share("broken", "right_well_recorded"),
                right_thin_broken_share=share("right_thin_broken", "right_thin"))


def dataset_reading(seed, runs):
    """The dataset values: averages over the in-range runs that give a value, and pooled cases for M6 and M7."""
    inside = [r for r in runs if r["in_range"]]
    bounds = {}
    for name in ("recorded", "thin", "well"):
        present = [r["bounds"][name] for r in inside if r["bounds"][name] is not None]
        bounds[name] = {key: average([b[key] for b in present]) for key in BOUND_KEYS} if present else None
    never = [r["never_commanded"] for r in inside if r["never_commanded"] is not None]
    return {"data_seed": seed, "runs": len(runs), "runs_in_range": len(inside),
            "coverage": {key: average([r["coverage"][key] for r in inside]) for key in COVERAGE_KEYS},
            "audit_rows": {key: average([r["coverage"].get(key) for r in inside])
                           for key in ("thin_rows", "well_rows", "rows_in_neither_class")},
            "tracking": {name: {mark: average([r["tracking"][name].get(mark) for r in inside])
                                for mark in ("M3", "M4", "M5")} for name in TRACKED},
            "runs_with_no_value": {mark: sum(r["tracking"]["margin"][mark] is None for r in inside)
                                   for mark in ("M3", "M4", "M5")},
            "decisions": {name: pooled([r["decisions"][name] for r in inside]) for name in CHARGED},
            "bounds": bounds,
            "never_commanded": {key: average([n[key] for n in never]) for key in never[0]} if never else None}


# ---- the marks, on nine dataset values ---------------------------------------------------------------------------
def mark_coverage(values, mean_bar):
    if not complete(values):
        return {"values": values, "passed": None}
    mean, at_floor = float(np.mean(values)), int(sum(at_least(v, COVER_FLOOR) for v in values))
    clauses = {f"{COVER_FLOOR} or more in at least {NEEDED} of {DATASETS} datasets": at_floor >= NEEDED,
               f"{mean_bar} or more on average": at_least(mean, mean_bar)}
    return {"values": values, "mean": mean, "datasets_at_the_floor": at_floor, "clauses": clauses,
            "wider_than_its_label": bool(mean > COVER_WIDE + EPS), "passed": all(clauses.values())}


def mark_m2(thin, well):
    classes = {"thin": mark_coverage(thin, COVER_FLOOR), "well": mark_coverage(well, COVER_FLOOR)}
    states = [c["passed"] for c in classes.values()]           # a class that fails on nine values fails the mark
    return dict(classes, passed=False if False in states else (None if None in states else True))


def mark_tracking(values, bar=None, run_values=None):
    """M3 (with `bar` and the runs' values) or M4 and M5 (above 0 holds)."""
    if not complete(values):
        return {"values": values, "passed": None}
    test = holds(values)
    out = {"values": values, "mean": float(np.mean(values)), "above_zero": test, "clauses": {"above 0 holds": test["holds"]}}
    if bar is not None:
        above = int(sum(v is not None and v > EPS for v in run_values))
        out.update(runs_in_range=len(run_values), runs_above_zero=above)
        out["clauses"][f"{bar} or more on average"] = at_least(out["mean"], bar)
        out["clauses"]["at least two thirds of the in-range runs are above 0"] = bool(
            run_values and at_least(above / len(run_values), RUNS_ABOVE))
    out["passed"] = all(out["clauses"].values())
    return out


def mark_m6(misled, still, copy):
    out = {"misled_share": misled, "still_wrong_share": still, "copy_still_wrong_share": copy}
    if not (complete(misled) and complete(still) and complete(copy)):
        return dict(out, passed=None)
    m, s, c = (float(np.mean(v)) for v in (misled, still, copy))
    own, against = holds([a - b for a, b in zip(misled, still)]), holds([a - b for a, b in zip(copy, still)])
    clauses = {"still wrong is at most half the misled share, on average": at_most(s, STILL_WRONG_SHARE * m),
               "lower than the misled share holds": own["holds"],
               "still wrong is at most three quarters of the copy's, on average": at_most(s, COPY_SHARE * c),
               "lower than the copy holds": against["holds"]}
    at_all = clauses["still wrong is at most half the misled share, on average"] and own["holds"]
    return dict(out, mean_misled=m, mean_still_wrong=s, mean_copy_still_wrong=c, lower_than_misled=own,
                lower_than_copy=against, clauses=clauses, corrects_at_all=bool(at_all), passed=all(clauses.values()))


def mark_m7(broken_shares, right_to_wrong, corrected):
    out = {"broken_share": broken_shares, "right_to_wrong": right_to_wrong, "corrected": corrected}
    if not complete(broken_shares):
        return dict(out, first_clause=None, second_clause=None, passed=None)
    mean = float(np.mean(broken_shares))
    first, second = at_most(mean, BROKEN_SHARE), bool(right_to_wrong < corrected)
    return dict(out, mean_broken_share=mean, first_clause=first, second_clause=second, passed=bool(first and second),
                clauses={"at most 1% of right choices broken, on average": first,
                         "fewer right choices broken than misled choices corrected": second})


def check_s6(misled_shares, misled_counts):
    if not complete(misled_shares):
        return {"misled_share": misled_shares, "passed": None}
    mean, with_a_case = float(np.mean(misled_shares)), int(sum(n > 0 for n in misled_counts))
    clauses = {"misled in at least 5% of eligible cases, on average": at_least(mean, ROOM_SHARE),
               "at least 8 of 9 datasets have a misled case": with_a_case >= ROOM_DATASETS}
    return {"misled_share": misled_shares, "mean_misled_share": mean, "datasets_with_a_misled_case": with_a_case,
            "clauses": clauses, "passed": all(clauses.values())}


def tracking_marks(datasets, inside, name):
    values = {mark: [d["tracking"][name][mark] for d in datasets] for mark in ("M3", "M4", "M5")}
    return {"M3": mark_tracking(values["M3"], TRACK_BAR, [r["tracking"][name]["M3"] for r in inside]),
            "M4": mark_tracking(values["M4"]), "M5": mark_tracking(values["M5"])}


def decision_marks(datasets, name):
    per = [d["decisions"][name] for d in datasets]
    return {"M6": mark_m6([p["misled_share"] for p in per], [p["still_wrong_share"] for p in per],
                          [p["copy_still_wrong_share"] for p in per]),
            "M7": mark_m7([p["broken_share"] for p in per], sum(p["broken"] for p in per), sum(p["corrected"] for p in per))}


def block_marks(datasets, inside):
    """Every mark's statistic and clauses, for the margin, the two reference margins and the section 8 readings."""
    coverage = {key: [d["coverage"][key] for d in datasets] for key in COVERAGE_KEYS}
    own = {"M1": mark_coverage(coverage["overall"], COVER_MEAN), "M2": mark_m2(coverage["thin"], coverage["well"]),
           **tracking_marks(datasets, inside, "margin"), **decision_marks(datasets, "margin")}
    per = [d["decisions"]["margin"] for d in datasets]
    return {"margin": own,
            "S6": check_s6([p["misled_share"] for p in per], [p["misled"] for p in per]),
            "reference_s5": {**tracking_marks(datasets, inside, "reference_s5"), **decision_marks(datasets, "reference_s5")},
            "reference_s7": {**tracking_marks(datasets, inside, "reference_s7"), **decision_marks(datasets, "reference_s7")},
            "constant_margin_in_M2": mark_m2(coverage["constant_thin"], coverage["constant_well"]),
            "rivals_tracking": {name: tracking_marks(datasets, inside, name) for name in RIVALS_TRACKING},
            "rivals_charged": {name: decision_marks(datasets, name) for name in RIVALS_CHARGED},
            "other_critics": {name: {**tracking_marks(datasets, inside, f"margin_{name}"),
                                     **decision_marks(datasets, f"margin_{name}")}
                              for name in ("second_critic", "lower_critic")},
            "without_worst_tenth": {"M3": tracking_marks(datasets, inside, "margin_without_worst_tenth")["M3"]},
            "sensitivity": {name: decision_marks(datasets, name) for name in CHARGED
                            if name.startswith(("margin_tolerance_", "margin_line_"))}}


def not_shown_better_than(datasets):
    """Rivals whose M3 statistic is higher, or whose share still wrong is lower, than the margin's in at least 6 of 9
    datasets. A tie does not count."""
    out = []
    for name, words in RIVALS_TRACKING.items():
        pairs = [(d["tracking"][name]["M3"], d["tracking"]["margin"]["M3"]) for d in datasets]
        if sum(a is not None and b is not None and a > b + EPS for a, b in pairs) >= NEEDED:
            out.append({"signal": name, "on": "M3", "sentence": f"Not shown to be better than {words} (M3)."})
    for name, words in RIVALS_CHARGED.items():
        pairs = [(d["decisions"][name]["still_wrong_share"], d["decisions"]["margin"]["still_wrong_share"]) for d in datasets]
        if sum(a is not None and b is not None and a < b - EPS for a, b in pairs) >= NEEDED:
            out.append({"signal": name, "on": "M6", "sentence": f"Not shown to be better than {words} (M6)."})
    return out


# ---- section 5's guards and section 7's verdict ------------------------------------------------------------------
def guarded(own, reference, check):
    """A pass always stands. A fail counts only if the reference margin passed the same mark."""
    if own["passed"] is None:
        return ("not judged", "fewer than nine dataset values")
    if own["passed"]:
        return ("pass", "")
    if reference["passed"] is True:
        return ("fail", "")
    return ("failed but not judged", f"the reference margin of {check} did not pass this mark")


def plain(own):
    if own["passed"] is None:
        return ("not judged", "fewer than nine dataset values")
    return ("pass", "") if own["passed"] else ("fail", "")


def judge(marks, s1_passed, enough):
    """Mark by mark: pass, fail, failed but not judged, or not judged. Then the first verdict that applies."""
    if not s1_passed or not enough:
        why = "; ".join(words for words, bad in (("S1 failed", not s1_passed),
                                                 ("fewer than nine valid datasets", not enough)) if bad)
        return {"status": {mark: ("not judged", why) for mark in MARKS}, "verdict": "inconclusive", "reason": why}
    own, s5, s7 = marks["margin"], marks["reference_s5"], marks["reference_s7"]
    status = {"M1": plain(own["M1"]), "M2": plain(own["M2"]),
              **{mark: guarded(own[mark], s5[mark], "S5") for mark in ("M3", "M4", "M5")}}
    if marks["S6"]["passed"] is True:
        status["M6"], status["M7"] = guarded(own["M6"], s7["M6"], "S7"), guarded(own["M7"], s7["M7"], "S7")
    else:
        status["M6"] = ("not judged", "S6 failed: too few misled cases")
        first = own["M7"]["first_clause"]
        if first is None:
            status["M7"] = ("not judged", "fewer than nine dataset values")
        elif first:
            status["M7"] = ("not judged", "S6 failed: judged on its first clause only, which cannot pass the mark")
        elif s7["M7"]["passed"] is True:
            status["M7"] = ("fail", "S6 failed: judged on its first clause only")
        else:
            status["M7"] = ("failed but not judged", "the reference margin of S7 did not pass this mark")
    words = [word for word, _ in status.values()]
    if "fail" in words:
        return {"status": status, "verdict": "not validated",
                "reason": "failed: " + ", ".join(mark for mark in MARKS if status[mark][0] == "fail")}
    if all(word == "pass" for word in words):
        return {"status": status, "verdict": "validated", "reason": "all seven marks are judged and pass"}
    return {"status": status, "verdict": "inconclusive",
            "reason": "not judged: " + "; ".join(f"{mark} ({status[mark][1]})" for mark in MARKS if status[mark][0] != "pass")}


def pct(value, digits=1):
    return "none" if value is None else f"{100 * value:.{digits}f}%"


def out_of_range_sentence(outside):
    if not outside:
        return "No run's critic left the possible range."
    return (f"In the {len(outside)} runs whose critic left the possible range the margin covered the one-step error "
            f"{pct(float(np.mean(outside)))} of the time (target 90%).")


def fixed_sentences(judged, marks, inside, outside, runs):
    """Section 7's fixed wording for the verdict that was reached."""
    status, own = judged["status"], marks["margin"]
    if judged["verdict"] == "validated":
        other = (f"In the other runs the margin covered the one-step error {pct(float(np.mean(outside)))} of the time."
                 if outside else "No run left the range.")
        return [f"Validated once, on the standard learner's critic, for well recorded and thinly recorded commands, "
                f"on half-random data on the published map (a recipe that is our choice on this map), in the {inside} of "
                f"{runs} runs whose critic stayed in range. {other} The margin is the one the standard learner fits "
                f"without using it. No claim for never-commanded commands, for the policy's own softer commands, or for "
                f"trained behavior."]
    if judged["verdict"] != "not validated":
        return [f"Inconclusive: {judged['reason']}."]
    coverage = "The margin's 90% coverage does not hold {}. The method promises it only over all rows."
    larger = "The margin is not shown to be larger where the critic's true error is larger{}."
    lines, said = [], False
    if status["M1"][0] == "pass" and status["M2"][0] == "pass":
        lines.append("The margin is a calibrated one-step error bar.")
    tracks = all(status[mark][0] == "pass" for mark in ("M3", "M4", "M5"))
    for mark in MARKS:
        if status[mark][0] != "fail":
            continue
        if mark in ("M6", "M7") and tracks and not said:
            lines.append("The margin is larger where the critic's true error is larger.")
            said = True
        if mark == "M1":
            lines.append(coverage.format("overall"))
        elif mark == "M2":
            lines += [coverage.format(words) for name, words in (("thin", "for thinly recorded commands"),
                                                                 ("well", "for well recorded commands"))
                      if own["M2"][name]["passed"] is False]
        elif mark == "M3":
            lines.append(larger.format(""))
        elif mark == "M4":
            lines.append(larger.format(" once the value of the cell and command is held fixed"))
        elif mark == "M5":
            lines.append(larger.format(" among thinly recorded commands"))
        elif mark == "M6":
            lines.append("Subtracting 3 margins is not shown to correct the critic's wrong choices "
                         + ("more than a wrong-cell copy does." if own["M6"]["corrects_at_all"] else "at all."))
        else:
            lines.append("Subtracting 3 margins breaks too many of the critic's right choices "
                         + ("as many as it corrects." if own["M7"]["first_clause"] else "more than 1%."))
    apart = [f"{mark} failed but was not judged: {status[mark][1]}." for mark in MARKS
             if status[mark][0] == "failed but not judged"
             or (status[mark][0] == "not judged" and status[mark][1].startswith("S6 failed") and own[mark]["passed"] is False)]
    return lines + apart


# ---- the block --------------------------------------------------------------------------------------------------
def score(task, policy, best_goal):
    """The paper's score: the chance of reaching the goal from the start cell within 200 steps, as a share of the
    best policy's (existing mark W1)."""
    return float(task.outcomes(policy)["goal"][task.start] / best_goal)


def setup_row(seed, block, grouped, task, free, best_goal):
    """Checks W0, S2, S3 and S4 for one candidate dataset. None of them reads the margin."""
    row = {"data_seed": seed, "W0": seed in block["passing_w0"]}
    if not row["W0"]:
        return dict(row, valid=False)
    mixed, optimal = grouped.get((HALF_RANDOM, seed), []), grouped.get((BEST_POLICY, seed), [])
    if len(mixed) != RUNS_PER_DATASET or len(optimal) != RUNS_PER_DATASET:
        raise Defect(f"dataset {seed} does not have three half-random and three best-policy runs")
    cls = support_classes(block["counts"][(HALF_RANDOM, seed)])
    row["thin_commands"], row["cells_with_a_thin_command"] = int(cls["thin"].sum()), int(cls["thin"].any(axis=1).sum())
    row["S2"] = bool(row["thin_commands"] >= THIN_COMMANDS_NEEDED and row["cells_with_a_thin_command"] >= THIN_CELLS_NEEDED)
    row["scores_on_best_policy_data"] = [score(task, run["commands"], best_goal) for run in optimal]
    row["S3"] = bool(min(row["scores_on_best_policy_data"]) >= PASS_SCORE)
    row["largest_critic_values"] = [float(np.abs(run["q1"][free]).max()) for run in mixed]
    row["runs_in_range"] = int(sum(in_range(run["q1"], free) for run in mixed))
    row["S4"] = bool(row["runs_in_range"] >= RUNS_IN_RANGE_NEEDED)
    return dict(row, valid=bool(row["S2"] and row["S3"] and row["S4"]))


def lava_block(block, grouped, task, used):
    """Section 9 for the standard learner: the outcome per dataset and recipe, and mark L0."""
    out = {}
    for recipe in (BEST_POLICY, RANDOM_STARTS):
        per_dataset = []
        for seed in used:
            runs = grouped.get((recipe, seed), [])
            if not runs:
                continue
            counts, executed = block["counts"][(recipe, seed)], block["executed"][(recipe, seed)]
            readings = [lava_reading(task, run["commands"], counts, executed, run["q1"], run["margin"]) for run in runs]
            per_dataset.append({"data_seed": seed, "runs": readings,
                                **{key: (int(sum(r[key] for r in readings)) if key in LAVA_CASES
                                         else float(np.mean([r[key] for r in readings]))) for key in readings[0]}})
        values = [d["counted"] for d in per_dataset]
        out[recipe] = {"per_dataset": per_dataset, "mean": float(np.mean(values)) if values else None,
                       "L0_room": bool(at_least(float(np.mean(values)), LAVA_ROOM)) if complete(values) else None}
    return out


def seen_standing(own, enough, audit_saved):
    """Section 7 on data already seen: each mark by its own pass rule, with no guard and no verdict."""
    out = {}
    for mark in MARKS:
        passed = own[mark]["passed"]
        if not enough:
            out[mark] = ("not read", "fewer than nine valid datasets")
        elif passed is None:
            out[mark] = ("not read", "the runs saved no audit rows" if mark == "M2" and not audit_saved
                         else "fewer than nine dataset values")
        else:
            out[mark] = ("meets on seen data" if passed else "does not meet on seen data", "")
    return out


def evaluate(block, s1=None):
    """The whole reading of one block. `block` is what load_block returns. `s1` is check S1's result; it is needed
    for a verdict and ignored on seen data."""
    task = block["task"]
    free = task.free_cells()
    if len(free) != OPEN_CELLS:
        raise Defect("the map does not have 71 open cells")
    candidates = sorted(block["candidates"])
    mode = "seen" if (SPENT_SEEDS & set(candidates)) or block.get("judged_before") else "verdict"
    half_random = [run for run in block["runs"] if run["recipe"] == HALF_RANDOM]
    audit_saved = all(run["audit"] is not None for run in half_random)
    if mode == "verdict" and (len(candidates) != CANDIDATES or s1 is None):
        raise Defect("a verdict block needs fifteen candidate datasets and check S1")
    if mode == "verdict" and not audit_saved:
        raise Defect("a verdict block needs the saved audit rows of every half-random run")
    best_goal = float(task.outcomes(E.policy_from_moves(task.best_policy()[0]))["goal"][task.start])
    grouped = {}
    for run in block["runs"]:
        grouped.setdefault((run["recipe"], run["data_seed"]), []).append(run)
    setup = [setup_row(seed, block, grouped, task, free, best_goal) for seed in candidates]
    valid = [row["data_seed"] for row in setup if row["valid"]]
    enough = len(valid) >= DATASETS
    used = valid[:DATASETS]
    readings = {seed: [read_run(run, task, block["counts"][(HALF_RANDOM, seed)], free)
                       for run in sorted(grouped[(HALF_RANDOM, seed)], key=lambda r: r["train_seed"])] for seed in used}
    datasets = [dataset_reading(seed, readings[seed]) for seed in used]
    every = [r for seed in used for r in readings[seed]]
    inside = [r for r in every if r["in_range"]]
    outside = [r["coverage"]["overall"] for r in every if not r["in_range"]]            # of the datasets used
    outside_in_block = [float(run["saved_coverage"]) for run in half_random if not in_range(run["q1"], free)]
    marks = block_marks(datasets, inside)
    result = {"criteria": CRITERIA_VERSION, "mode": mode, "candidates": candidates, "setup": setup, "valid": valid,
              "enough_valid_datasets": enough, "datasets_used": used, "S1": None if mode == "seen" else s1,
              "runs": len(every), "runs_in_range": len(inside),
              "runs_whose_margin_was_not_ready": int(sum(not r["margin_was_ready"] for r in every)),
              "half_random_runs_in_the_block": len(half_random),
              "coverage_in_out_of_range_runs_of_the_block": outside_in_block,
              "coverage_in_out_of_range_runs_of_the_datasets_used": outside,
              "out_of_range_sentence": out_of_range_sentence(outside_in_block),
              "dataset_values": datasets, "run_readings": readings, "marks": marks,
              "not_shown_better_than": not_shown_better_than(datasets) if enough else [],
              "lava_outcome": lava_block(block, grouped, task, used)}
    if mode == "verdict":
        judged = judge(marks, bool(s1["passed"]), enough)
        result.update(status=judged["status"], verdict=judged["verdict"], reason=judged["reason"],
                      sentences=fixed_sentences(judged, marks, len(inside), outside, len(every)))
    else:
        result.update(verdict=None, reason="data already seen: no verdict is given (section 7)",
                      standing=seen_standing(marks["margin"], enough, audit_saved))
    return result


# ---- reading the saved files ----------------------------------------------------------------------------------
def checked_audit(rows, saved_coverage, where):
    """One run's audit rows as saved. The saved one-step error must be the target minus the lower critic, and the
    share inside the margin must be the figure the run itself reported."""
    if any(len(rows[name]) != AUDIT_ROWS for name in AUDIT_FIELDS):
        raise Defect(f"{where}: expected {AUDIT_ROWS} audit rows")
    if not np.allclose(rows["residual"], rows["target"] - np.minimum(rows["q1"], rows["q2"]), rtol=1e-5, atol=1e-5):
        raise Defect(f"{where}: the saved one-step error is not the target minus the lower critic")
    if abs(float((np.abs(rows["residual"]) <= rows["margin"]).mean()) - float(saved_coverage)) > 1e-9:
        raise Defect(f"{where}: the audit rows do not give the coverage the run reported")
    return rows


def load_block(label, here=HERE):
    """The saved files of one standard-learner run and its datasets, checked against their receipts."""
    task, here = E.published(), Path(here)
    folder = here / "results" / label
    start = json.loads((folder / "START.json").read_text())
    if start.get("alpha") != ALPHA or start.get("method") != "baseline only":
        raise Defect("not a standard-learner run with alpha 1.0")
    manifest = json.loads((here / "receipts" / f"{start['data_label']}_manifest.json").read_text())
    entries = {entry["file"]: entry for entry in manifest["datasets"]}
    trained_on = json.loads((folder / f"datasets_{task.name}.json").read_text())
    saved = json.loads((folder / f"baselines_{task.name}.json").read_text())["runs"]
    with np.load(folder / f"baselines_{task.name}.npz", allow_pickle=False) as arrays:
        arrays = {name: arrays[name] for name in ("commands", "q1", "q2", "width", "support")}
    audit, audit_file = None, folder / f"audit_rows_{task.name}.npz"
    if audit_file.exists():
        with np.load(audit_file, allow_pickle=False) as rows:
            if any(name not in rows.files for name in AUDIT_FIELDS):
                raise Defect(f"{audit_file.name} does not hold {', '.join(AUDIT_FIELDS)}")
            audit = {name: rows[name] for name in AUDIT_FIELDS}
    counts, executed, runs = {}, {}, []
    for run in saved:
        key, i = (run["recipe"], run["data_seed"]), run["index"]
        where = f"run {i} ({run['recipe']}, dataset {run['data_seed']}, training seed {run['train_seed']})"
        if run["method"] != "baseline" or run.get("alpha") != ALPHA or run["final_step"] != FINAL_STEP:
            raise Defect(f"{where}: not the standard learner with alpha 1.0 and 25,000 updates")
        if key not in counts:
            name = f"{task.name}_{run['recipe']}_seed{run['data_seed']}.npz"
            with np.load(here / "data" / start["data_label"] / name, allow_pickle=False) as file:
                data = {field: file[field] for field in C.FIELDS}
            digest = C.digest(data)
            if not (name in entries and entries[name]["checks_passed"] and digest == entries[name]["array_sha256"]
                    and digest == trained_on.get(name)):
                raise Defect(f"{name} is not the dataset the receipt and the run recorded")
            counts[key], executed[key] = pair_counts(data["s"], data["command"]), pair_counts(data["s"], data["executed"])
        rows = None
        if audit is not None and run["recipe"] == HALF_RANDOM:
            rows = checked_audit({name: audit[name][i] for name in AUDIT_FIELDS}, run["audit_bank"]["coverage"], where)
        runs.append({"recipe": run["recipe"], "data_seed": run["data_seed"], "train_seed": run["train_seed"],
                     "reference_ready": bool(run["reference_ready"]), "support_floor": float(run["reference"]["support_floor"]),
                     "saved_coverage": float(run["audit_bank"]["coverage"]),
                     "commands": np.asarray(arrays["commands"][i], float), "q1": np.asarray(arrays["q1"][i], float),
                     "q2": np.asarray(arrays["q2"][i], float), "margin": np.asarray(arrays["width"][i], float),
                     "support": np.asarray(arrays["support"][i], float), "audit": rows})
    return {"label": label, "task": task, "folder": folder, "start": start, "manifest": manifest,
            "candidates": list(manifest["seeds"]), "passing_w0": list(manifest["seeds_passing_every_check"]),
            "judged_before": bool(verdict_receipt(here, start["data_label"]).exists()
                                  or set(manifest["seeds"]) & judged_seeds(here)),
            "runs": runs, "counts": counts, "executed": executed}


# ---- writing the result ------------------------------------------------------------------------------------------
def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def num(value, digits=3):
    return "none" if value is None else f"{value:.{digits}f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]) + "\n"


def coverage_text(own, wide=False):
    if own["passed"] is None:
        return "dataset values: " + ", ".join(num(v) for v in own["values"])
    return (f"average {pct(own['mean'])}; 85% or more in {own['datasets_at_the_floor']} of {DATASETS} datasets"
            + ("; wider than its label" if wide and own["wider_than_its_label"] else ""))


def statistic(mark, own):
    """One line of plain words for a mark's statistic."""
    if mark == "M1":
        return coverage_text(own, wide=True)
    if mark == "M2":
        return "; ".join(f"{words}: {coverage_text(own[name])}" for name, words in (("thin", "thinly recorded"),
                                                                                    ("well", "well recorded")))
    if own["passed"] is None:
        values = own.get("values", own.get("still_wrong_share", own.get("broken_share", [])))
        return "dataset values: " + ", ".join(num(v) for v in values)
    if mark in ("M3", "M4", "M5"):
        test = own["above_zero"]
        text = f"average {num(own['mean'])}; above 0 in {test['datasets_above']} of {DATASETS} datasets, p = {test['p']:.4f}"
        return text + (f"; {own['runs_above_zero']} of {own['runs_in_range']} in-range runs above 0" if mark == "M3" else "")
    if mark == "M6":
        return (f"misled {pct(own['mean_misled'])}, still wrong {pct(own['mean_still_wrong'])}, the copy leaves "
                f"{pct(own['mean_copy_still_wrong'])}; lower than misled in {own['lower_than_misled']['datasets_above']} of "
                f"{DATASETS} (p = {own['lower_than_misled']['p']:.4f}), lower than the copy in "
                f"{own['lower_than_copy']['datasets_above']} of {DATASETS} (p = {own['lower_than_copy']['p']:.4f})")
    return (f"{pct(own['mean_broken_share'], 2)} of right choices broken; {num(own['right_to_wrong'], 1)} went from right to "
            f"wrong, {num(own['corrected'], 1)} misled choices were corrected")


def render(result, label, fingerprints):
    marks, own, seen = result["marks"], result["marks"]["margin"], result["mode"] == "seen"
    standing = result["standing"] if seen else result["status"]
    rows = [[mark, statistic(mark, own[mark]), ". ".join(part for part in standing[mark] if part)] for mark in MARKS]
    yes = {True: "yes", False: "no", None: "no value"}
    doc = [f"# The margin against its criteria: {label}\n",
           f"Written by `margin_verdict.py` from saved files. Nothing was trained. Criteria: {result['criteria']}.\n",
           f"Fingerprints: criteria `{fingerprints['criteria'][:16]}`, verdict script `{fingerprints['verdict_script'][:16]}`.\n"]
    if seen:
        doc += ["## Reading on data already seen. Not a verdict\n",
                "These datasets were seen before this reading. Each mark can only say \"meets\" or \"does not meet on "
                "seen data\", by its own pass rule. The guards of the setup checks are not applied. A mark with no "
                "value on these data is shown as \"not read\".\n",
                table(["Mark", "Statistic", "On seen data"], rows)]
    else:
        doc += [f"## Verdict: {result['verdict']}\n", *(line + "\n" for line in result["sentences"]),
                *(entry["sentence"] + "\n" for entry in result["not_shown_better_than"]),
                table(["Mark", "Statistic", "Standing"], rows)]
    doc.append(result["out_of_range_sentence"] + f" This counts all {result['half_random_runs_in_the_block']} half-random "
               "runs of the block, runs on dropped datasets included.\n")

    doc.append("## Setup checks\n")
    if result["S1"] is not None:
        s1 = result["S1"]
        doc.append("S1, the text was fixed first: " + ("passed.\n" if s1["passed"] else "FAILED: " + "; ".join(s1["reasons"]) + ".\n"))
    doc.append(table(["Dataset", "S2: thin commands (cells)", "S3: lowest score", "S4: runs in range"],
                     [[f"{row['data_seed']} ({'valid' if row['valid'] else 'dropped'})",
                       f"{row['thin_commands']} ({row['cells_with_a_thin_command']})", num(min(row["scores_on_best_policy_data"])),
                       f"{row['runs_in_range']} of {RUNS_PER_DATASET}"] if row["W0"] else
                      [f"{row['data_seed']} (dropped)", "failed the acceptance checks", "", ""] for row in result["setup"]]))
    doc.append(f"Valid datasets: {len(result['valid'])}. Used: {', '.join(str(seed) for seed in result['datasets_used']) or 'none'}. "
               f"Runs in range on the datasets used: {result['runs_in_range']} of {result['runs']}. "
               f"Runs whose margin was not ready at its last refit (20,000 updates): {result['runs_whose_margin_was_not_ready']}.\n")
    s6 = marks["S6"]
    doc.append("S6, room: " + ("no value.\n" if s6["passed"] is None else
               f"the critic is misled in {pct(s6['mean_misled_share'])} of eligible cases on average, and "
               f"{s6['datasets_with_a_misled_case']} of {DATASETS} datasets have a misled case: "
               f"{'passed' if s6['passed'] else 'failed'}.\n"))
    doc.append("The two reference margins on M3 to M7. S5 is 1 over the square root of the pair's row count, scaled like a "
               "rival on M6 and M7. S7 is the size of the true error, which is ranked against itself on M3 to M5.\n")
    doc.append(table(["Reference margin", "Mark", "Statistic", "Passes"],
                     [[name, mark, statistic(mark, marks[key][mark]), yes[marks[key][mark]["passed"]]]
                      for key, name in (("reference_s5", "S5"), ("reference_s7", "S7")) for mark in MARKS[2:]]))

    doc.append("## Dataset values\n")
    values = result["dataset_values"]
    doc.append(table(["Dataset", "M1 overall", "M2 thin", "M2 well"],
                     [[d["data_seed"], pct(d["coverage"]["overall"]), pct(d["coverage"]["thin"]), pct(d["coverage"]["well"])]
                      for d in values]))
    doc.append("Audit rows by class. A row at a cell with fewer than 16 training rows, or with a never-commanded command, "
               "is in M1 and in neither class of M2.\n")
    doc.append(table(["Dataset", "Thinly recorded", "Well recorded", "In neither class"],
                     [[d["data_seed"], *(num(d["audit_rows"][key], 0) for key in ("thin_rows", "well_rows", "rows_in_neither_class"))]
                      for d in values]))
    doc.append(table(["Dataset", "M3", "M4", "M5"],
                     [[d["data_seed"], *(num(d["tracking"]["margin"][mark]) for mark in ("M3", "M4", "M5"))] for d in values]))
    doc.append("In-range runs that gave no value (fewer than three commands qualified): "
               + ", ".join(f"{mark} {sum(d['runs_with_no_value'][mark] for d in values)}" for mark in ("M3", "M4", "M5")) + ".\n")
    doc.append(table(["Dataset", "Misled", "Still wrong", "Copy leaves wrong"],
                     [[d["data_seed"], pct(d["decisions"]["margin"]["misled_share"]), pct(d["decisions"]["margin"]["still_wrong_share"]),
                       pct(d["decisions"]["margin"]["copy_still_wrong_share"])] for d in values]))
    doc.append(table(["Dataset", "Right choices (well recorded)", "Broken", "Misled corrected"],
                     [[d["data_seed"], d["decisions"]["margin"]["right_well_recorded"], d["decisions"]["margin"]["broken"],
                       d["decisions"]["margin"]["corrected"]] for d in values]))

    doc.append("## Reported beside it (no pass mark)\n")
    doc.append("The lower-bound share: recorded commands where the critic minus 3 margins is not above the true value.\n")
    doc.append(table(["Dataset", "With 3 margins", "With no margin", "With a constant margin"],
                     [[d["data_seed"], *(pct(d["bounds"]["recorded"][key]) if d["bounds"]["recorded"] else "none"
                                         for key in BOUND_KEYS[:3])] for d in values]))
    doc.append("The strict reading: pairs whose true error is inside one margin.\n")
    doc.append(table(["Dataset", "Thinly recorded", "Well recorded"],
                     [[d["data_seed"], *(pct(d["bounds"][name]["error_inside_one_margin"]) if d["bounds"][name] else "none"
                                         for name in ("thin", "well"))] for d in values]))
    doc.append("A constant margin in M2 (the 90th percentile of the one-step error size): "
               + statistic("M2", marks["constant_margin_in_M2"]) + ".\n")
    doc.append("The cost of caution (right choices at thinly recorded commands) and other wrong moves, over in-range runs.\n")
    doc.append(table(["Dataset", "Right thin choices", "Share of these broken", "Misled, moved to a wrong well recorded command"],
                     [[d["data_seed"], d["decisions"]["margin"]["right_thin"], pct(d["decisions"]["margin"]["right_thin_broken_share"]),
                       d["decisions"]["margin"]["moved_to_a_wrong_well_recorded_command"]] for d in values]))
    doc.append("Rival signals. The margin does not have to beat them.\n")
    doc.append(table(["Signal", "Mark", "Statistic"],
                     [[words, mark, statistic(mark, marks[group][name][mark])]
                      for group, names, which in (("rivals_tracking", RIVALS_TRACKING, ("M3", "M4", "M5")),
                                                  ("rivals_charged", RIVALS_CHARGED, ("M6", "M7")))
                      for name, words in names.items() for mark in which]))
    never = [d["never_commanded"] for d in values if d["never_commanded"]]
    if never:
        doc.append(f"Never-commanded commands: the margin there averages {num(average([n['margin_at_never_commanded'] for n in never]))}, "
                   f"against {num(average([n['margin_at_recorded'] for n in never]))} at recorded commands of the same cells.\n")
    doc.append("The second critic, the lower of the two, M3 without the worst tenth of cells, and the sensitivity readings.\n")
    extra = [[words, mark, statistic(mark, marks["other_critics"][name][mark])]
             for name, words in (("second_critic", "Second critic"), ("lower_critic", "Lower of the two critics"))
             for mark in MARKS[2:]]
    extra.append(["Without the worst tenth of cells", "M3", statistic("M3", marks["without_worst_tenth"]["M3"])])
    extra += [[name.replace("margin_", "").replace("_", " "), mark, statistic(mark, marks["sensitivity"][name][mark])]
              for name in marks["sensitivity"] for mark in ("M6", "M7")]
    doc.append(table(["Reading", "Mark", "Statistic"], extra))

    doc.append("## Never-commanded lava moves (section 9), standard learner\n")
    doc.append("Share of episodes from the start cell, in percentage points, that end in lava on a step where the policy "
               "commanded a lava move no training row commands at that cell. Cells with 16 or more rows.\n")
    names = {BEST_POLICY: "Best-policy data", RANDOM_STARTS: "Random-starts data"}
    lava = result["lava_outcome"]
    doc.append(table(["Recipe", "Average over datasets", "L0: room of 1 point"],
                     [[names[recipe], num(entry["mean"], 2), {True: "yes", False: "no", None: "no standing"}[entry["L0_room"]]]
                      for recipe, entry in lava.items()]))
    doc.append("Beside it, averaged over datasets: the same share at cells with fewer than 16 rows, and the split of the "
               "outcome between lava moves never executed in the data and those executed through the noise.\n")
    doc.append(table(["Recipe", "Cells with few rows", "Never executed", "Executed through noise"],
                     [[names[recipe], *(num(average([d[key] for d in entry["per_dataset"]]), 2)
                                        for key in ("cells_with_few_rows", "never_executed", "executed_through_noise"))]
                      for recipe, entry in lava.items()]))
    doc.append("Cases (one run at one cell) where the critic values a never-commanded lava move above every recorded "
               "command of its cell, before and after 3 margins, summed over the datasets used.\n")
    doc.append(table(["Recipe", "Cases", "Critic rates a lava move highest", "The same after 3 margins"],
                     [[names[recipe], *(sum(d[key] for d in entry["per_dataset"]) for key in LAVA_CASES)]
                      for recipe, entry in lava.items()]))
    doc.append("The readings this script had to fix where the criteria are silent are listed at the top of `margin_verdict.py`.\n")
    return "\n".join(doc)


def write_result(result, folder, here, label, data_label):
    """A verdict is written once: first the receipt that spends the block's datasets, then the result and its page.
    A reading on seen data goes to files of its own and never replaces a verdict."""
    if result["mode"] == "verdict":
        out, receipt = Path(folder) / "margin_verdict.json", verdict_receipt(here, data_label)
        if out.exists() or receipt.exists():
            raise Defect("this block was already judged, and a block is judged once")
        receipt.write_text(json.dumps(jsonable(
            {"data_label": data_label, "results_label": label, "candidates": result["candidates"],
             "datasets_used": result["datasets_used"], "verdict": result["verdict"], "reason": result["reason"],
             "sentences": result["sentences"], "fingerprints": result["fingerprints"]}), indent=1))
    else:
        out = Path(folder) / "margin_seen_reading.json"
    out.write_text(json.dumps(jsonable(result), indent=1, allow_nan=False))
    out.with_suffix(".md").write_text(render(result, label, result["fingerprints"]), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baselines", required=True, help="results label of a finished standard-learner run")
    args = parser.parse_args()
    if git("rev-parse", "HEAD") is None:
        sys.exit("git could not answer in this repository, so the freeze of section 10, rule 2 cannot be checked. "
                 "No run was opened and nothing was computed.")
    state = freeze_state()
    if not state["ok"]:
        loose = state["scripts_not_committed"]
        what = [path.name for name, path in FROZEN.items() if not state[name]["committed"]]
        what += ["git could not list the testbed's scripts"] if loose is None else loose
        sys.exit("Section 10, rule 2: the criteria, this script, the dataset builder and the run scripts must be "
                 "committed, and unchanged since, before any mark is computed. Not committed or changed: "
                 + ", ".join(what) + ". No run was opened and nothing was computed.")
    try:
        block = load_block(args.baselines)
        fresh = not (SPENT_SEEDS & set(block["candidates"])) and not block["judged_before"]
        if fresh and (block["folder"] / "margin_verdict.json").exists():
            raise Defect("this block was already judged, and a block is judged once")
        s1 = check_s1(block["manifest"], block["start"], block["folder"]) if fresh else None
        result = evaluate(block, s1)
        result["fingerprints"] = {**{name: state[name]["fingerprint"] for name in FROZEN},
                                  "critical_env": fingerprint(HERE / "critical_env.py"),
                                  "published_collect": fingerprint(HERE / "published_collect.py")}
        out = write_result(result, block["folder"], HERE, args.baselines, block["start"]["data_label"])
    except Defect as defect:
        sys.exit(f"Defect, nothing judged and nothing written: {defect}")
    standing = result["standing"] if result["mode"] == "seen" else result["status"]
    print(result["verdict"] or "reading on seen data, no verdict", "->", out.name)
    for mark in MARKS:
        print(mark, *standing[mark])


if __name__ == "__main__":
    main()
