"""The action gap of the D4RL-RT track: remove the densest cluster of logged actions from training and bring it back
at evaluation (design: the approved action-gap design, 2026-10-12; experiments/truth/README.md, 'Action gap').

This module is data only. It needs no host, no checkpoint and no simulator, and it never trains anything. It
fixes the gap before any training and freezes it with a SHA-256:
  define   the gap on the native training split, the gapped HDF5 copy, the frozen test pools, W/gap/gap.json
  check    G0a (the gap is well defined and the copy removes exactly those transitions) and G0b (the gap is real in
           (standardized s, a) while its states stay covered), written to W/gap/g0.json
  freeze   W/gap/gap.json.sha256; refuses unless G0 passed (a SYNTHETIC-only override exists for tests)
Every later command (gap_train.py, gap_acceptance.py, pools.restore_host for a gapped run) takes the gap sha256 and
refuses on any mismatch, including a gapped file whose sha256 differs from the manifest.

Input. The task's pinned cached HDF5 (configs/td3_bc.yaml: hopper_medium-v2.hdf5 sha256 5bdf1bc4...,
walker2d_medium_replay-v2.hdf5 e75de96a...). An unpinned --file is accepted only when it is a synthetic test file
(synthetic.py). Conversion is runtime/td3_bc._convert. The native reservation is this branch's
runtime.td3_bc._reserve with the config's reservation (size 1024, seed 911, K = 6 for Hopper and 23 for Walker2d,
max_fraction 0.1); its seed does not depend on the host seed, so every this-branch host of a task has the same split.
The gap is computed from that split's TRAINING rows only.

The common action (find_gap). Metric: plain Euclidean distance in raw action space, float64 of the stored float32
actions (distance()). For a centre c, rho_q(c) is the k-th smallest distance from c over the rows, k = ceil(q N),
q = 0.15. The common action c* = argmin rho_q(c) over the box [-1, 1]^d: the centre of the smallest ball that holds
15% of the training actions (the kNN-density mode at k = qN). Search, seed 20261012:
  1. candidates: 4,096 training actions (default_rng([seed, 1])), every box corner, the coordinate-wise median;
  2. each candidate's rho_q on 200,000 training actions (default_rng([seed, 2]); Walker2d: all rows), chunked brute
     force;
  3. the 16 best by compass search on rho_q: steps 0.1, 0.05, 0.025, 0.0125, 0.00625; at each step, passes over the
     axes in fixed order, trying +step then -step with the point clipped to the box, accepting only strict
     improvements, until a pass improves nothing;
  4. refined centres are tried in order of their refined rho_q (radius_rule below); rho is exact on all N rows.
Radius (radius_rule). rho = the k-th smallest distance; the gap is the INCLUSIVE ball {a : ||a - c*|| <= rho}, so a
tie block at rho is never split and no kept row has exactly the action of a gap row. If the inclusive mass is above
16%, step down to the largest distinct distance below rho and accept when the mass is still >= 14% (tie-block
fallback); otherwise the next refined centre is tried (next-centre fallback). If no centre lands in [14%, 16%], or
the chosen rho < 0.05 (point-mass guard: an exact-duplicate spike a tanh actor cannot be 'in'), G0 fails.
Diagnostics: the k = 100 mode and its distance to c* in units of rho; saturation shares; the largest exact-duplicate
mass; the number of distinct actions; the ball's mass per split; density contrast (the gap ball's mass over the
median mass of an equal-radius ball centred on 1,000 random training actions); rho over the median pairwise
action distance. The evaluation rule is the same for logged actions and for pi(s): in_gap(a) = distance(a, c*) <= rho,
near when <= 1.25 rho, d(a) = distance / rho.

The gapped copy (write_gapped_copy). Every dataset and attribute of the original is copied; timeouts[i] = True for
every gap raw row i (every native converted transition whose logged action is in the ball, in every split); the
root attribute bca_action_gap holds the gap definition sha256. _convert and d4rl.qlearning_dataset both skip a
timeout row, so the gapped conversion is exactly the native converted rows minus the gap rows, array for array
(G0a checks it), and row i-1 keeps its true next observation s_i: the gap states still enter TD targets. Deleting
rows would splice false transitions and is never done. The original file is never touched.

The projected reservation (projection, make_projected_reserve). The flag surgery cuts episodes into fragments, and
the native reservation on fragments would withhold other rows. A gapped host's split is therefore the native split
projected onto the kept rows, matched by raw row index: gapped training = native training minus gap, likewise for
withheld and the bank. projection() patches runtime.td3_bc._reserve (a private function, at run time; no file
changes) with a function that refuses unless the converted rows are exactly the native rows minus the gap, reruns
the native reservation on the manifest's native maps exactly as validate_prepared does, checks it against the
manifest's pinned ids and maps it to gapped ids. prepare and validate_prepared both call it, so train.py's own
checks rerun it. It pins the sha256 of runtime/td3_bc.py.

Test pools (frozen here, the same rows for all six hosts of a task; raw row indices):
  G        min(3,000, all) gap rows from native TRAINING (default_rng([seed, 5])).
  G_held   min(1,000, all) gap rows from native WITHHELD rows (default_rng([seed, 6])); neither critic trained on them.
  K_match  for each G row in seeded order (default_rng([seed, 7])), the nearest kept native-training row in
           standardized state space from another original episode (its action is outside the gap by construction),
           greedy without replacement, within the caliper r_s.
  K_far    3,000 kept native-training rows (seeded order, default_rng([seed, 8])) whose 10 nearest native-training
           neighbours in x (self excluded) include no gap row.
Geometry: x = (standardized s with the native normalization, raw a), float64. r = the median 10-NN other-episode
distance from 5,000 seeded native withheld rows (default_rng([seed, 9])) to the native training tree; r_s the same in
state-only space. 'Other episode' means another ORIGINAL episode (native _convert ids), never a flag fragment.

G0 (check). G0a: training-split mass in [0.14, 0.16]; rho >= 0.05; no tie block split; the gapped conversion equals
native minus gap array for array (and _convert's own d4rl.qlearning_dataset identity check passes); every HDF5 key
except timeouts has the original's array sha256 and timeouts differ exactly at the gap rows; the projected
reservation equals the native split minus the gap by raw row; zero gap rows in gapped training, withheld or bank;
the gapped train.prepare passes runtime validate_prepared; negative control: the leak detector counts every gap row
of the unprojected native training split. G0b, over every gap row of the training split (core d <= 0.5 rho and rim
strata reported): c_full (native training rows of other episodes within r), c_kept_oe (kept, other episodes),
c_kept_all (kept, any episode), state-only c_s_full and c_s_kept at r_s (other episodes). Pass: (i) median c_full >=
10; (ii) share with c_kept_all <= 2 >= 0.50; (iii) median (c_kept_oe + 1) / (c_full + 1) <= 0.20; (iv) median
c_s_kept >= 10 and median (c_s_kept + 1) / (c_s_full + 1) >= 0.5; (v) at least 90% of G rows matched into K_match.
Reported: 1-NN distances to kept vs full rows, gap run lengths within episodes, the share of gap states that are the
next observation of a kept row, the same counts on G_held, the K_match distances.

Synthetic files (synthetic.py) are test fixtures: every output on one says SYNTHETIC and is NOT A FINDING.
"""

import argparse
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.truth import pools as PL  # noqa: E402
from experiments.truth import state_restore as SR  # noqa: E402
from experiments.truth.state_restore import DEFAULT_DATA, array_sha256, file_sha256, require, task_spec  # noqa: E402

SCHEMA = "bca-truth-action-gap-v1"
G0_SCHEMA = "bca-truth-action-gap-g0-v1"
TD3_CONFIG = ROOT / "configs" / "td3_bc.yaml"
PINNED_FILES = ("runtime/td3_bc.py", "train.py", "experiments/truth/gap.py")
SYNTHETIC_NOTE = ("SYNTHETIC: computed on a synthetic D4RL-format file (experiments/truth/synthetic.py); a test "
                  "fixture, NOT A FINDING.")

DESIGN = dict(
    q=0.15, band=[0.14, 0.16], rho_min=0.05, near=1.25, reference_seed=202609171,
    metric="Euclidean distance in raw action space, float64 of the stored float32 actions",
    search=dict(seed=20261012, candidates=4096, score_rows=dict(hopper=200000, walker2d=None), refine_top=16,
                steps=[0.1, 0.05, 0.025, 0.0125, 0.00625], max_passes=1000, k_diagnostic=100),
    geometry=dict(k=10, radius_rows=5000, contrast_centres=1000, pair_samples=100000, saturation=0.999),
    pools=dict(G=3000, G_held=1000, K_far=3000, K_far_neighbours=10),
    g0_marks=dict(mass_min=0.14, mass_max=0.16, rho_min=0.05, c_full_median_min=10, thin_kept_max=2,
                  thin_kept_share_min=0.50, kept_full_ratio_median_max=0.20, state_kept_median_min=10,
                  state_ratio_median_min=0.5, match_rate_min=0.90))
SMOKE = json.loads(json.dumps(DESIGN))  # tiny sizes, for synthetic tests only (refuses real data)
SMOKE["search"].update(candidates=256, score_rows=dict(hopper=None, walker2d=None), refine_top=4)
SMOKE["geometry"].update(radius_rows=200, contrast_centres=50, pair_samples=2000)
SMOKE["pools"].update(G=24, G_held=8, K_far=24)
PROFILES = {"design": DESIGN, "synthetic_smoke": SMOKE}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value):
    return sha256_hex(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def strict_json(value):
    return json.dumps(PL._json_ready(value), sort_keys=True, indent=2, allow_nan=False)


def code_pins():
    return {name: file_sha256(ROOT / name) for name in PINNED_FILES}


def gap_tag(q):
    return f"gap{int(round(100 * q))}"


# ---------------------------------------------------------------- the rule

def distance(actions, centre):
    """||a - c||_2 per row, float64 (the one formula used for definition, pools and evaluation)."""
    a = np.atleast_2d(np.asarray(actions, np.float64))
    return np.sqrt(np.sum((a - np.asarray(centre, np.float64)[None, :]) ** 2, axis=-1))


def in_gap(actions, centre, rho):
    return distance(actions, centre) <= float(rho)


def kth(values, k):
    """The k-th smallest value (1-based) along the last axis."""
    return np.partition(values, k - 1, axis=-1)[..., k - 1]


def rho_q(actions, centres, q, chunk=None, budget=12_000_000):
    """k-th smallest distance (k = ceil(q N)) from each centre to the rows, chunked brute force (chunks sized so a
    chunk's (centres, rows, d) difference array holds about `budget` float64 values: about 100 MB)."""
    a = np.asarray(actions, np.float64)
    centres = np.atleast_2d(np.asarray(centres, np.float64))
    k = max(1, int(math.ceil(q * len(a))))
    chunk = chunk or max(1, int(budget // max(1, a.size)))
    out = np.empty(len(centres))
    for i in range(0, len(centres), chunk):
        c = centres[i:i + chunk]
        d = np.sqrt(np.sum((a[None, :, :] - c[:, None, :]) ** 2, axis=-1))
        out[i:i + chunk] = kth(d, k)
    return out


def compass_refine(actions, centre, q, steps, max_passes):
    """Compass search on rho_q: per step size, passes over the axes in fixed order trying +step then -step (clipped
    to the box); strict improvements only; a step size ends when a full pass improves nothing."""
    c = np.asarray(centre, np.float64).copy()
    best = float(rho_q(actions, c, q)[0])
    evaluations = 1
    for step in steps:
        for _ in range(int(max_passes)):
            improved = False
            for j in range(len(c)):
                for sign in (1.0, -1.0):
                    trial = c.copy()
                    trial[j] = np.clip(trial[j] + sign * step, -1.0, 1.0)
                    if trial[j] == c[j]:
                        continue
                    value = float(rho_q(actions, trial, q)[0])
                    evaluations += 1
                    if value < best:
                        c, best, improved = trial, value, True
                        break
            if not improved:
                break
    return c, best, evaluations


def radius_rule(dist, q, band):
    """The inclusive radius at the target mass, with the tie-block fallback. Returns a dict with rho, mass and the
    rule that fired, or rule 'none' when neither lands in the band (next-centre fallback)."""
    dist = np.asarray(dist, np.float64)
    n = len(dist)
    k = int(math.ceil(q * n))
    rho = float(kth(dist, k))
    mass = float(np.mean(dist <= rho))
    lo, hi = band
    if lo <= mass <= hi:
        return dict(rule="base", rho=rho, mass=mass, k=k, inclusive_mass_at_k=mass)
    below = dist[dist < rho]
    if mass > hi and len(below):
        rho2 = float(below.max())
        mass2 = float(np.mean(dist <= rho2))
        if lo <= mass2 <= hi:
            return dict(rule="tie_block_fallback", rho=rho2, mass=mass2, k=k, inclusive_mass_at_k=mass)
    return dict(rule="none", rho=None, mass=None, k=k, inclusive_mass_at_k=mass)


def find_gap(actions, settings):
    """The densest-cluster search on training actions (module docstring). Returns the gap record; record['defined']
    is False with a reason when G0 must fail (no centre in band, or the point-mass guard)."""
    a = np.asarray(actions, np.float64)
    n, d = a.shape
    s, q = settings["search"], float(settings["q"])
    seed = int(s["seed"])
    cand_rows = np.sort(np.random.default_rng([seed, 1]).choice(n, size=min(int(s["candidates"]), n), replace=False))
    corners = np.asarray(list(itertools.product([-1.0, 1.0], repeat=d)))
    median = np.median(a, axis=0)[None]
    candidates = np.concatenate([a[cand_rows], corners, median])
    kinds = ["training_action"] * len(cand_rows) + ["corner"] * len(corners) + ["median"]
    score_rows = s["score_rows"]
    if score_rows is None or score_rows >= n:
        sample = np.arange(n)
    else:
        sample = np.sort(np.random.default_rng([seed, 2]).choice(n, size=int(score_rows), replace=False))
    scored = a[sample]
    scores = rho_q(scored, candidates, q)
    order = np.argsort(scores, kind="stable")[:int(s["refine_top"])]
    refined = []
    for rank, i in enumerate(order):
        c, value, evals = compass_refine(scored, candidates[i], q, s["steps"], s["max_passes"])
        refined.append(dict(rank=rank, candidate=int(i), kind=kinds[i], start=candidates[i].tolist(),
                            start_rho_q=float(scores[i]), centre=c.tolist(), refined_rho_q=value, evaluations=evals))
    refined.sort(key=lambda r: (r["refined_rho_q"], r["rank"]))
    tried = []
    chosen = None
    for r in refined:
        rule = radius_rule(distance(a, r["centre"]), q, settings["band"])
        tried.append(dict(centre=r["centre"], **rule))
        if rule["rule"] != "none":
            chosen = (r, rule)
            break
    record = dict(n_train=int(n), action_dim=int(d), q=q, k=int(math.ceil(q * n)), score_rows=int(len(sample)),
                  candidates=dict(training_actions=int(len(cand_rows)), corners=int(len(corners)), median=1),
                  refined=refined, tried=tried)
    if chosen is None:
        return dict(record, defined=False, reason=f"no refined centre lands in the band {settings['band']}")
    r, rule = chosen
    record.update(centre=r["centre"], rho=rule["rho"], mass_train=rule["mass"], rule=rule["rule"],
                  centre_kind=r["kind"], centre_order=len(tried) - 1,
                  next_centre_fallback=len(tried) > 1)
    if rule["rho"] < settings["rho_min"]:
        return dict(record, defined=False, reason=f"point-mass guard: rho {rule['rho']} < {settings['rho_min']} "
                    "(the densest 15% is an exact-duplicate spike; the user decides)")
    return dict(record, defined=True, reason=None)


def descriptors(actions, centre, rho, settings, candidates_record=None):
    """Stage-0 descriptors of the training actions (reported, never gated)."""
    from scipy.spatial import cKDTree

    a = np.asarray(actions, np.float64)
    n, d = a.shape
    g = settings["geometry"]
    seed = settings["search"]["seed"]
    sat = float(g["saturation"])
    uniq, counts = np.unique(a, axis=0, return_counts=True)
    tree = cKDTree(a)
    centres = a[np.random.default_rng([seed, 3]).choice(n, size=min(int(g["contrast_centres"]), n), replace=False)]
    local = np.asarray(tree.query_ball_point(centres, float(rho), return_length=True), np.float64) / n
    gap_mass = float(np.mean(distance(a, centre) <= rho))
    pg = np.random.default_rng([seed, 4])
    i, j = pg.integers(0, n, int(g["pair_samples"])), pg.integers(0, n, int(g["pair_samples"]))
    pair = np.sqrt(np.sum((a[i] - a[j]) ** 2, axis=1))
    k100 = min(int(settings["search"]["k_diagnostic"]), n)
    cand = a[np.sort(np.random.default_rng([seed, 1]).choice(n, size=min(int(settings["search"]["candidates"]), n),
                                                             replace=False))]
    dk = tree.query(cand, k=k100)[0]
    dk = dk[:, -1] if dk.ndim == 2 else dk
    mode = cand[int(np.argmin(dk))]
    return dict(
        saturated_share_per_coordinate=[float(np.mean(np.abs(a[:, j]) >= sat)) for j in range(d)],
        exactly_one_share_per_coordinate=[float(np.mean(np.abs(a[:, j]) == 1.0)) for j in range(d)],
        all_saturated_share=float(np.mean(np.all(np.abs(a) >= sat, axis=1))),
        largest_duplicate_mass=float(counts.max() / n), distinct_actions=int(len(uniq)),
        density_contrast=gap_mass / float(np.median(local)) if np.median(local) > 0 else None,
        median_equal_radius_mass=float(np.median(local)),
        rho_over_median_pair_distance=float(rho / np.median(pair)) if np.median(pair) > 0 else None,
        median_pair_distance=float(np.median(pair)),
        k100_mode=dict(k=k100, centre=mode.tolist(), distance_to_c_star_in_rho=float(distance(mode[None], centre)[0]
                                                                                     / rho)))


# ---------------------------------------------------------------- native split and normalization

def reservation_config(task, synthetic):
    """configs/td3_bc.yaml's reservation of the task (host row, reference seed). A synthetic test file uses the small
    reservation of pools.write_synthetic_run (its few rows cannot hold a 1,024-row bank)."""
    from runtime.config import resolve

    row = resolve(TD3_CONFIG, "host", DESIGN["reference_seed"], ROOT / "runs" / "gap-inspection", task.name)
    res = dict(row["protocol"]["reservation"])
    if synthetic:
        res.update(PL.SYNTHETIC_RESERVATION)
    return res


def native_split(raw, reservation):
    """(converted, rows, episodes, maps, training, withheld, bank, record) of this branch's native preparation:
    runtime/td3_bc._convert and the native _reserve, called exactly as validate_prepared calls it."""
    from runtime import td3_bc as R
    from runtime.common import Transition

    require(not getattr(R._reserve, "bca_action_gap_projection", False),
            "the native split cannot be computed while the action-gap projection is active")
    converted, rows, episodes = SR.converted_rows({k: raw[k] for k in PL.RAW_KEYS})
    maps = R.dependency_maps(rows, converted["terminals"])
    blank = np.zeros((len(rows), 1), np.float32)
    done = np.asarray(maps["terminal_mask"], np.float32)
    training, withheld, bank, record = R._reserve(maps, R.Reservation(**reservation),
                                                  Transition(blank, blank, blank[:, 0], blank, done))
    return converted, rows, episodes, maps, *(np.asarray(x, np.int64) for x in (training, withheld, bank)), record


def normalization(obs, training):
    from runtime.common import compute_mean_std

    mean, std = compute_mean_std(np.asarray(obs, np.float32)[training], eps=0.001)
    return np.asarray(mean, np.float32), np.asarray(std, np.float32)


def standardize(obs, mean, std):
    return ((np.asarray(obs, np.float32) - mean) / std).astype(np.float32).astype(np.float64)


# ---------------------------------------------------------------- the gapped copy

def _dataset_sha(value):
    v = np.asarray(value)
    if v.dtype.kind in "OSUV":
        return sha256_hex(repr(v.tolist()).encode())
    return array_sha256(v)


def per_key_sha256(path):
    import h5py

    out = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, item: out.__setitem__(name, _dataset_sha(item[()]))
                     if isinstance(item, h5py.Dataset) else None)
    return out


def attributes(path):
    import h5py

    with h5py.File(path, "r") as f:
        return {k: repr(np.asarray(v).tolist()) for k, v in f.attrs.items()}


def write_gapped_copy(source, target, gap_rows, definition_sha):
    """Copy every dataset and attribute; timeouts[gap rows] = True; root attribute bca_action_gap. Never overwrites."""
    import h5py

    target = Path(target)
    require(not target.exists(), f"{target} exists; the gapped copy is written once")
    gap_rows = np.asarray(gap_rows, np.int64)
    tmp = target.with_name(target.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(source, "r") as src, h5py.File(tmp, "x") as dst:
        for name in src:
            src.copy(src[name], dst, name=name)
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        timeouts = dst["timeouts"][()]
        require(timeouts.dtype == np.bool_ or np.isin(timeouts, (0, 1)).all(), "binary timeouts required")
        require(not np.any(timeouts[gap_rows]), "a gap row is already a timeout row (it was never converted)")
        timeouts[gap_rows] = True
        dst["timeouts"][...] = timeouts
        dst.attrs["bca_action_gap"] = definition_sha
    tmp.rename(target)
    return target


# ---------------------------------------------------------------- geometry

class ArrayThinness(PL.ThinnessIndex):
    """pools.ThinnessIndex over given points and ORIGINAL episode ids (no host needed)."""

    def __init__(self, x, episode, k=10, workers=1):
        from scipy.spatial import cKDTree

        self.x = np.asarray(x, np.float64)
        self.episode = np.asarray(episode, np.int64)
        self.k, self.workers = int(k), int(workers)
        self.tree = cKDTree(self.x)
        order = np.argsort(self.episode, kind="stable")
        bounds = np.r_[0, np.flatnonzero(np.diff(self.episode[order])) + 1, len(order)]
        self._members = {int(self.episode[order[a]]): order[a:b] for a, b in zip(bounds[:-1], bounds[1:])}
        self.radius = None

    def counts_any(self, x, radius):
        return np.asarray(self.tree.query_ball_point(np.asarray(x, np.float64), float(radius), return_length=True,
                                                     workers=self.workers), np.int64)


def match_states(query_s, query_episode, cand_s, cand_episode, caliper, order):
    """Greedy nearest-neighbour matching without replacement in standardized state space: each query (in `order`)
    takes the nearest unused candidate from another episode, if it lies within the caliper. Returns (match index or
    -1 per query, distance or NaN)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(cand_s, np.float64))
    used = np.zeros(len(cand_s), bool)
    match = np.full(len(query_s), -1, np.int64)
    dist = np.full(len(query_s), np.nan)
    for i in order:
        k = 16
        while True:
            kk = min(k, len(cand_s))
            d, idx = tree.query(query_s[i], k=kk)
            d, idx = np.atleast_1d(d), np.atleast_1d(idx)
            found, beyond = None, False
            for dj, j in zip(d, idx):
                if dj > caliper:
                    beyond = True
                    break
                if not used[j] and cand_episode[j] != query_episode[i]:
                    found = (j, dj)
                    break
            if found is not None or beyond or kk == len(cand_s):
                break
            k *= 4
        if found is not None:
            match[i], dist[i] = found[0], found[1]
            used[found[0]] = True
    return match, dist


def run_lengths(flags, episode, raw_rows):
    """Lengths of runs of consecutive gap rows (adjacent raw rows of one original episode)."""
    flags = np.asarray(flags, bool)
    out, current = [], 0
    for i in range(len(flags)):
        if flags[i]:
            if current and (raw_rows[i] == raw_rows[i - 1] + 1 and episode[i] == episode[i - 1] and flags[i - 1]):
                current += 1
            else:
                if current:
                    out.append(current)
                current = 1
        elif current:
            out.append(current)
            current = 0
    if current:
        out.append(current)
    return np.asarray(out, np.int64)


def _summary(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0, median=None, mean=None, p10=None, p90=None, max=None)
    return dict(n=int(len(v)), median=float(np.median(v)), mean=float(v.mean()), p10=float(np.percentile(v, 10)),
                p90=float(np.percentile(v, 90)), max=float(v.max()))


# ---------------------------------------------------------------- define

def _side(directory, name, array):
    path = Path(directory) / f"{name}.npy"
    require(not path.exists(), f"{path} exists; the gap is defined once per world")
    with path.open("xb") as handle:
        np.save(handle, np.asarray(array))
    return dict(file=path.name, sha256=file_sha256(path), shape=list(np.shape(array)),
                dtype=str(np.asarray(array).dtype))


def original_file(task, data_dir=DEFAULT_DATA, file=None):
    if file is not None:
        path = Path(file).expanduser().resolve()
        require(path.is_file(), f"{path} is missing")
        require(SR._synthetic_mark(path) is not None, "an unpinned --file is accepted only for synthetic test files; "
                "real data must be the pinned cache of configs/td3_bc.yaml (--data-dir)")
        return path, dict(path=str(path), sha256=file_sha256(path), pinned=False, size_bytes=path.stat().st_size)
    return SR.cached_file(task, data_dir)


def define(world, task, *, data_dir=DEFAULT_DATA, file=None, profile="design", gapped_dir=None, workers=1):
    """Define the gap, write the gapped copy and freeze the test pools: W/gap/gap.json plus .npy side files."""
    task = task_spec(task)
    require(profile in PROFILES, f"profile in {sorted(PROFILES)}")
    settings = json.loads(json.dumps(PROFILES[profile]))
    out = Path(world) / "gap"
    require(not (out / "gap.json").exists(), f"{out / 'gap.json'} exists; the gap is defined once per world")
    path, identity = original_file(task, data_dir, file)
    synthetic = SR._synthetic_mark(path) is not None
    require(profile != "synthetic_smoke" or synthetic, "the synthetic_smoke profile refuses real data")
    reservation = reservation_config(task, synthetic)
    raw = PL.read_raw(path)
    converted, rows, episodes, maps, training, withheld, bank, record = native_split(raw, reservation)
    actions_all = converted["actions"].astype(np.float64)
    gap = find_gap(actions_all[training], dict(settings, search=dict(
        settings["search"], score_rows=settings["search"]["score_rows"][task.name])))
    out.mkdir(parents=True, exist_ok=True)
    mean, std = normalization(converted["observations"], training)
    side = {}
    for name, value in (("native_raw_current", rows), ("native_terminal_mask", np.asarray(maps["terminal_mask"],
                                                                                          np.int8)),
                        ("native_episodes", episodes), ("native_training_ids", training),
                        ("native_withheld_ids", withheld), ("native_bank_ids", bank)):
        side[name] = _side(out, name, value)
    manifest = dict(
        schema=SCHEMA, task=task.name, dataset=task.dataset, profile=profile, synthetic=synthetic,
        note=SYNTHETIC_NOTE if synthetic else "Action-gap definition, fixed before any training.",
        original=dict(identity, file=path.name), reservation=reservation,
        native=dict(converted_rows=int(len(rows)), training=int(len(training)), withheld=int(len(withheld)),
                    bank=int(len(bank)), record_sha256=canonical_sha256(PL._json_ready(record)),
                    obs_mean=mean.tolist(), obs_std=std.tolist(),
                    normalization_sha256=array_sha256(np.concatenate([mean, std])),
                    normalization_rule="training-row mean and SD + 0.001 (runtime.common.compute_mean_std)"),
        settings=settings, search=gap, code_sha256=code_pins(), created_utc=utc(), defined=bool(gap["defined"]))
    if not gap["defined"]:
        manifest["side_files"] = side
        _write_manifest(out, manifest)
        return manifest
    centre, rho, q = np.asarray(gap["centre"]), float(gap["rho"]), float(settings["q"])
    dist_all = distance(actions_all, centre)
    gap_conv = dist_all <= rho
    gap_rows = rows[gap_conv]
    split_of = np.zeros(len(rows), np.int8)
    split_of[withheld] = 1
    in_bank = np.zeros(len(rows), bool)
    in_bank[bank] = True
    masses = dict(training=float(gap_conv[training].mean()), withheld=float(gap_conv[withheld].mean()),
                  bank=float(gap_conv[bank].mean()) if len(bank) else None, all_converted=float(gap_conv.mean()))
    counts = dict(training=int(gap_conv[training].sum()), withheld=int(gap_conv[withheld].sum()),
                  bank=int(gap_conv[bank].sum()), all_converted=int(gap_conv.sum()))
    definition = dict(task=task.name, original_sha256=identity["sha256"], q=q, metric=settings["metric"],
                      centre=[float(x) for x in centre], rho=rho, rule=gap["rule"],
                      gap_rows_sha256=array_sha256(gap_rows))
    definition_sha = canonical_sha256(definition)
    gapped_dir = Path(gapped_dir).expanduser().resolve() if gapped_dir else path.parent
    gapped_dir.mkdir(parents=True, exist_ok=True)
    gapped = write_gapped_copy(path, gapped_dir / f"{path.stem}.{gap_tag(q)}-{definition_sha[:8]}{path.suffix}",
                               gap_rows, definition_sha)
    keys_original, keys_gapped = per_key_sha256(path), per_key_sha256(gapped)
    side["gap_rows"] = _side(out, "gap_rows", gap_rows)
    pools, geometry = build_pools(converted, rows, episodes, training, withheld, gap_conv, mean, std, settings,
                                  workers=workers)
    for name in ("G", "G_held", "K_match", "K_far", "K_match_for_G", "K_match_distance"):
        side[f"pool_{name}"] = _side(out, f"pool_{name}", pools[name])
    manifest.update(
        side_files=side, definition=definition, definition_sha256=definition_sha,
        gap=dict(centre=definition["centre"], rho=rho, q=q, near=float(settings["near"]), rule=gap["rule"],
                 masses=masses, counts=counts, tie_block_split=False,
                 max_in_gap_distance=float(dist_all[gap_conv].max()),
                 min_kept_distance=float(dist_all[~gap_conv].min()) if (~gap_conv).any() else None),
        descriptors=descriptors(actions_all[training], centre, rho, settings),
        gapped=dict(file=gapped.name, path=str(gapped), sha256=file_sha256(gapped), size_bytes=gapped.stat().st_size,
                    per_key_sha256=keys_gapped, attribute="bca_action_gap", attribute_value=definition_sha),
        original_per_key_sha256=keys_original,
        geometry=geometry,
        pools=dict(G=int(len(pools["G"])), G_held=int(len(pools["G_held"])),
                   K_match=int(len(pools["K_match"])), K_far=int(len(pools["K_far"])),
                   seeds={name: [settings["search"]["seed"], tag] for name, tag in (
                       ("G", 5), ("G_held", 6), ("K_match_order", 7), ("K_far_order", 8), ("radius_rows", 9))},
                   available=dict(training_gap_rows=counts["training"], withheld_gap_rows=counts["withheld"])))
    _write_manifest(out, manifest)
    return manifest


def _write_manifest(out, manifest):
    path = Path(out) / "gap.json"
    with path.open("x", encoding="utf8") as handle:
        handle.write(strict_json(manifest))
    return path


def build_pools(converted, rows, episodes, training, withheld, gap_conv, mean, std, settings, workers=1):
    """Pools G, G_held, K_match, K_far (raw rows) and the geometry they rest on (r, r_s, the samples)."""
    seed = int(settings["search"]["seed"])
    g, p = settings["geometry"], settings["pools"]
    obs = converted["observations"]
    s_all = standardize(obs, mean, std)
    a_all = converted["actions"].astype(np.float64)
    x_all = np.concatenate([s_all, a_all], axis=1)
    tr_gap = training[gap_conv[training]]
    tr_kept = training[~gap_conv[training]]
    wh_gap = withheld[gap_conv[withheld]]
    full_x = ArrayThinness(x_all[training], episodes[training], k=g["k"], workers=workers)
    full_s = ArrayThinness(s_all[training], episodes[training], k=g["k"], workers=workers)
    rr = withheld[np.sort(np.random.default_rng([seed, 9]).choice(len(withheld), size=min(int(g["radius_rows"]),
                                                                                       len(withheld)), replace=False))]
    r, _ = full_x.fit_radius(x_all[rr], episodes[rr])
    r_s, _ = full_s.fit_radius(s_all[rr], episodes[rr])
    G = np.sort(tr_gap[np.random.default_rng([seed, 5]).choice(len(tr_gap), size=min(int(p["G"]), len(tr_gap)),
                                                                 replace=False)]) if len(tr_gap) else tr_gap
    G_held = np.sort(wh_gap[np.random.default_rng([seed, 6]).choice(len(wh_gap), size=min(int(p["G_held"]),
                                                                                         len(wh_gap)),
                                                                     replace=False)]) if len(wh_gap) else wh_gap
    order = np.random.default_rng([seed, 7]).permutation(len(G))
    match, mdist = match_states(s_all[G], episodes[G], s_all[tr_kept], episodes[tr_kept], r_s, order)
    k_for_g = np.where(match >= 0, tr_kept[np.maximum(match, 0)], -1)
    # K_far: kept training rows whose K nearest training neighbours in x (self excluded) hold no gap row.
    kn = int(p["K_far_neighbours"])
    cand = tr_kept[np.random.default_rng([seed, 8]).permutation(len(tr_kept))]
    pos_in_training = np.searchsorted(training, cand)
    far, start = [], 0
    while len(far) < int(p["K_far"]) and start < len(cand):
        chunk = cand[start:start + max(4 * int(p["K_far"]), 256)]
        pos = pos_in_training[start:start + len(chunk)]
        kq = min(kn + 1, len(training))
        _, idx = full_x.tree.query(x_all[chunk], k=kq, workers=workers)
        idx = idx.reshape(len(chunk), kq)
        for row, own, nb in zip(chunk, pos, idx):
            nb = nb[nb != own][:kn]
            if not np.any(gap_conv[training[nb]]):
                far.append(row)
                if len(far) == int(p["K_far"]):
                    break
        start += len(chunk)
    K_far = np.sort(np.asarray(far, np.int64))
    pools = dict(G=rows[G], G_held=rows[G_held], K_match=np.sort(rows[k_for_g[k_for_g >= 0]]),
                 K_far=rows[K_far], K_match_for_G=np.where(k_for_g >= 0, rows[np.maximum(k_for_g, 0)], -1),
                 K_match_distance=mdist)
    geometry = dict(r=float(r), r_s=float(r_s), k=int(g["k"]), radius_rows=int(len(rr)),
                    x="(standardized s with the native normalization, raw a), float64",
                    K_match=dict(matched=int((match >= 0).sum()), of=int(len(G)), caliper=float(r_s),
                                 distance=_summary(mdist)),
                    K_far=dict(found=int(len(K_far)), wanted=int(p["K_far"]), neighbours=kn))
    return pools, geometry


# ---------------------------------------------------------------- the manifest

class Manifest:
    """A verified gap.json with its side files (sha256-checked on first use)."""

    def __init__(self, path, data, sha):
        self.path, self.data, self.sha256 = Path(path), data, sha
        self.dir = self.path.parent
        self._arrays = {}

    def __getitem__(self, key):
        return self.data[key]

    def array(self, name):
        if name not in self._arrays:
            entry = self.data["side_files"][name]
            path = self.dir / entry["file"]
            require(path.is_file() and file_sha256(path) == entry["sha256"],
                    f"{path} is missing or differs from the gap manifest")
            self._arrays[name] = np.load(path)
        return self._arrays[name]

    @property
    def centre(self):
        return np.asarray(self.data["gap"]["centre"], np.float64)

    @property
    def rho(self):
        return float(self.data["gap"]["rho"])

    def d_over_rho(self, actions):
        return distance(actions, self.centre) / self.rho

    def in_gap(self, actions):
        return in_gap(actions, self.centre, self.rho)

    def native_ids_of(self, raw_rows):
        """Native converted ids of raw rows (each must be a native converted row)."""
        native = self.array("native_raw_current")
        raw_rows = np.asarray(raw_rows, np.int64)
        pos = np.searchsorted(native, raw_rows)
        require(np.all(pos < len(native)) and np.array_equal(native[np.minimum(pos, len(native) - 1)], raw_rows),
                "raw rows that are not native converted rows")
        return pos

    def original_episodes(self, raw_rows):
        return self.array("native_episodes")[self.native_ids_of(raw_rows)]

    def gap_mask_raw(self, raw_rows):
        return np.isin(np.asarray(raw_rows, np.int64), self.array("gap_rows"))

    def native_split_rows(self):
        native = self.array("native_raw_current")
        return {name: native[self.array(f"native_{name}_ids")] for name in ("training", "withheld", "bank")}

    def gapped_path(self, data_dir=None):
        base = Path(data_dir).expanduser().resolve() if data_dir else Path(self.data["gapped"]["path"]).parent
        path = base / self.data["gapped"]["file"]
        require(path.is_file(), f"gapped copy {path} is missing (written by `gap define`)")
        digest = file_sha256(path)
        require(digest == self.data["gapped"]["sha256"],
                f"{path} has sha256 {digest}, the gap manifest pins {self.data['gapped']['sha256']}; refusing")
        return path

    def original_path(self, data_dir=None):
        base = Path(data_dir).expanduser().resolve() if data_dir else Path(self.data["original"]["path"]).parent
        path = base / self.data["original"]["file"]
        require(path.is_file(), f"original file {path} is missing")
        require(file_sha256(path) == self.data["original"]["sha256"], f"{path} differs from the gap manifest")
        return path


def read_manifest(path, expected_sha256=None, *, require_frozen=True):
    """Load gap.json after the hash gate: its sha256 must equal expected_sha256 (when given) and, when
    require_frozen, the gap.json.sha256 sidecar written by `freeze`."""
    path = Path(path)
    require(path.is_file(), f"gap manifest {path} is missing (run `gap define`)")
    data = path.read_bytes()
    digest = sha256_hex(data)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"gap manifest hash mismatch: {path} has sha256 {digest}, given {expected_sha256!r}; "
                         "refusing")
    if require_frozen:
        side = path.with_name("gap.json.sha256")
        if not side.is_file() or side.read_text(encoding="utf8").split()[0] != digest:
            raise ValueError(f"{side} is missing or does not match {path}: the gap is not frozen (run `gap freeze`)")
    record = json.loads(data)
    require(record.get("schema") == SCHEMA, "not an action-gap manifest")
    return Manifest(path, record, digest)


# ---------------------------------------------------------------- the projected reservation

def make_projected_reserve(manifest, native_reserve):
    """runtime.td3_bc._reserve for a gapped file: the native split minus the gap, matched by raw row."""
    from runtime import td3_bc as R
    from runtime.common import Transition

    m = manifest
    native_rows = m.array("native_raw_current").astype(np.int64)
    terminal = m.array("native_terminal_mask").astype(np.int64)
    is_gap = np.isin(native_rows, m.array("gap_rows"))
    kept_rows = native_rows[~is_gap]
    new_id = np.cumsum(~is_gap) - 1
    pins = {name: m.data["side_files"][f"native_{name}_ids"]["sha256"] for name in ("training", "withheld", "bank")}
    expected = {name: m.array(f"native_{name}_ids") for name in pins}

    def _reserve(maps, reservation, data):
        rows = np.asarray(maps["raw_current"], np.int64)
        leaked = np.intersect1d(rows, m.array("gap_rows"))
        require(not len(leaked), f"{len(leaked)} action-gap rows are converted rows of this file (first "
                f"{leaked[:5].tolist()}): the gapped file leaks gap transitions; refusing")
        require(np.array_equal(rows, kept_rows), "the converted rows are not exactly the native rows minus the action "
                "gap; refusing the projected reservation")
        from dataclasses import asdict

        require(asdict(reservation) == m.data["reservation"], "the run's reservation differs from the gap manifest's")
        nmaps = R.dependency_maps(native_rows, terminal)
        blank = np.zeros((len(native_rows), 1), np.float32)
        done = terminal.astype(np.float32)
        tr, wh, bk, record = native_reserve(nmaps, reservation, Transition(blank, blank, blank[:, 0], blank, done))
        for name, ids in (("training", tr), ("withheld", wh), ("bank", bk)):
            require(np.array_equal(np.asarray(ids, np.int64), expected[name]),
                    f"the native {name} split differs from the gap manifest")
        project = lambda ids: new_id[np.asarray(ids, np.int64)[~is_gap[np.asarray(ids, np.int64)]]].astype(np.int32)
        out = [project(x) for x in (tr, wh, bk)]
        projected_record = dict(
            projection="native this-branch reservation minus the action-gap rows, matched by raw row index "
                       "(experiments/truth/gap.py)",
            gap_manifest_sha256=m.sha256, gap_definition_sha256=m.data["definition_sha256"],
            native_record_sha256=m.data["native"]["record_sha256"], native_ids_sha256=pins,
            native_record=R._json_value(record),
            counts=dict(training=int(len(out[0])), withheld=int(len(out[1])), bank=int(len(out[2])),
                        native_training=int(len(tr)), native_withheld=int(len(wh)), native_bank=int(len(bk))))
        return (*out, projected_record)

    _reserve.bca_action_gap_projection = True
    return _reserve


@contextlib.contextmanager
def projection(manifest):
    """Patch runtime.td3_bc._reserve with the projected reservation for the duration (prepare and
    validate_prepared both call it). Refuses when runtime/td3_bc.py differs from the manifest's pin."""
    from runtime import td3_bc as R

    pinned = manifest.data["code_sha256"]["runtime/td3_bc.py"]
    current = file_sha256(ROOT / "runtime" / "td3_bc.py")
    require(current == pinned, f"runtime/td3_bc.py has sha256 {current}, the gap manifest pins {pinned}; the "
            "projected reservation patches its private _reserve and refuses on any change")
    native = R._reserve
    require(not getattr(native, "bca_action_gap_projection", False), "nested action-gap projection")
    with mock.patch.object(R, "_reserve", make_projected_reserve(manifest, native)):
        yield


def leak_counts(manifest, raw_rows_by_split):
    """Number of action-gap raw rows inside each named set of raw rows (the no-leak check of G0a and H0)."""
    gap = manifest.array("gap_rows")
    return {name: int(np.isin(np.asarray(rows, np.int64), gap).sum()) for name, rows in raw_rows_by_split.items()}


def gapped_row(row, manifest):
    """A resolved td3_bc host row made into a gapped-host row: the gapped cache pin, run_id suffix and the
    action_gap record train.py writes into resolved.json."""
    m = manifest
    row = copy.deepcopy(row)
    require((row.get("algorithm"), row.get("method")) == ("td3_bc", "host"), "gapped hosts are td3_bc host runs only")
    require(row["environment"] == m.data["dataset"], "the run's dataset differs from the gap manifest's")
    reservation = row["protocol"]["reservation"]
    if reservation != m.data["reservation"]:
        require(m.data["synthetic"], "configs/td3_bc.yaml's reservation differs from the gap manifest's; refusing")
        reservation.update(m.data["reservation"])  # synthetic test files only (their few rows need a small bank)
    row["cache"] = dict(filename=m.data["gapped"]["file"], sha256=m.data["gapped"]["sha256"])
    suffix = f"-{gap_tag(m.data['gap']['q'])}-{m.sha256[:8]}"
    row["run_id"] += suffix
    row["protocol"]["run_id"] += suffix
    row["action_gap"] = dict(
        manifest=str(m.path.resolve()), manifest_sha256=m.sha256, definition_sha256=m.data["definition_sha256"],
        centre=m.data["gap"]["centre"], rho=m.data["gap"]["rho"], q=m.data["gap"]["q"],
        gapped_file=m.data["gapped"]["file"], gapped_sha256=m.data["gapped"]["sha256"],
        original_file=m.data["original"]["file"], original_sha256=m.data["original"]["sha256"],
        projection="native reservation minus the gap rows, matched by raw row (experiments/truth/gap.py)",
        gap_code_sha256=file_sha256(ROOT / "experiments/truth/gap.py"),
        gap_train_sha256=file_sha256(ROOT / "experiments/truth/gap_train.py"), synthetic=m.data["synthetic"])
    return row


def host_projection(row, manifest_path=None):
    """(projection context, manifest) for restoring a gapped run; refuses without its verified, frozen manifest."""
    record = row.get("action_gap")
    require(isinstance(record, dict), "the run carries no action_gap record")
    path = Path(manifest_path or record["manifest"])
    require(path.is_file(), f"gap manifest {path} is missing: a gapped host cannot be restored without it")
    m = read_manifest(path, record["manifest_sha256"], require_frozen=True)
    require(row["cache"]["sha256"] == m.data["gapped"]["sha256"], "the run's cache is not the manifest's gapped copy")
    return projection(m), m


# ---------------------------------------------------------------- G0

def g0b_passes(values, marks):
    """G0b's pass rule on its values (pure)."""
    v = values
    ok = lambda x, f: x is not None and f(x)
    return dict(
        i_dense=ok(v["c_full_median"], lambda x: x >= marks["c_full_median_min"]),
        ii_thin_under_kept=ok(v["thin_kept_share"], lambda x: x >= marks["thin_kept_share_min"]),
        iii_neighbours_lost=ok(v["kept_full_ratio_median"], lambda x: x <= marks["kept_full_ratio_median_max"]),
        iv_states_covered=ok(v["state_kept_median"], lambda x: x >= marks["state_kept_median_min"])
        and ok(v["state_ratio_median"], lambda x: x >= marks["state_ratio_median_min"]),
        v_matched=ok(v["match_rate"], lambda x: x >= marks["match_rate_min"]))


def g0b_counts(x_q, s_q, ep_q, full_x, kept_x, full_s, kept_s, r, r_s, thin_max):
    c_full = full_x.counts(x_q, ep_q, r)
    c_kept_oe = kept_x.counts(x_q, ep_q, r)
    c_kept_all = kept_x.counts_any(x_q, r)
    c_s_full = full_s.counts(s_q, ep_q, r_s)
    c_s_kept = kept_s.counts(s_q, ep_q, r_s)
    nn_full = full_x.knn_other(x_q, ep_q, k=1)[:, 0]
    nn_kept = kept_x.knn_other(x_q, ep_q, k=1)[:, 0]
    ratio = (c_kept_oe + 1.0) / (c_full + 1.0)
    s_ratio = (c_s_kept + 1.0) / (c_s_full + 1.0)
    med = lambda v: float(np.median(v)) if len(v) else None
    return dict(rows=int(len(x_q)), c_full_median=med(c_full), c_kept_oe_median=med(c_kept_oe),
                c_kept_all_median=med(c_kept_all), thin_kept_share=float(np.mean(c_kept_all <= thin_max))
                if len(x_q) else None, kept_full_ratio_median=med(ratio), state_full_median=med(c_s_full),
                state_kept_median=med(c_s_kept), state_ratio_median=med(s_ratio),
                nn_full=_summary(nn_full), nn_kept=_summary(nn_kept)), \
        dict(c_full=c_full, c_kept_oe=c_kept_oe, c_kept_all=c_kept_all, c_s_full=c_s_full, c_s_kept=c_s_kept,
             nn_full=nn_full, nn_kept=nn_kept)


def check(world, gap_sha256, *, data_dir=None, workers=1):
    """G0a and G0b: W/gap/g0.json (outcome-free; data only)."""
    from experiments.wbcp import freeze_scores as FS
    from runtime import td3_bc as R
    from runtime.config import resolve

    out = Path(world) / "gap"
    m = read_manifest(out / "gap.json", gap_sha256, require_frozen=False)
    require(not (out / "g0.json").exists(), f"{out / 'g0.json'} exists; G0 is checked once (delete it to recheck)")
    d = m.data
    head = dict(schema=G0_SCHEMA, task=d["task"], dataset=d["dataset"], gap_sha256=m.sha256, synthetic=d["synthetic"],
                profile=d["profile"], utc=utc(), note=SYNTHETIC_NOTE if d["synthetic"] else
                "G0, outcome-free (no host, no simulator, no training)")
    if not d["defined"]:
        record = dict(head, passed=False, g0a=dict(passed=False, passes=dict(defined=False),
                                                    reason=d["search"]["reason"]), g0b=None)
        _write_json(out / "g0.json", record)
        return record
    marks = d["settings"]["g0_marks"]
    original = m.original_path(data_dir)
    gapped = m.gapped_path(data_dir)
    raw_o, raw_g = PL.read_raw(original), PL.read_raw(gapped)
    converted, rows, episodes, maps, training, withheld, bank, record = native_split(raw_o, d["reservation"])
    require(np.array_equal(rows, m.array("native_raw_current")), "native rows differ from the manifest")
    for name, ids in (("training", training), ("withheld", withheld), ("bank", bank)):
        require(np.array_equal(ids, m.array(f"native_{name}_ids")), f"native {name} split differs from the manifest")
    actions = converted["actions"].astype(np.float64)
    dist = distance(actions, m.centre)
    gap_conv = dist <= m.rho
    rule_rows = rows[gap_conv]
    a = {}
    a["gap_rows_reproduce"] = bool(np.array_equal(rule_rows, m.array("gap_rows")))
    mass = float(gap_conv[training].mean())
    a["mass_in_band"] = bool(marks["mass_min"] <= mass <= marks["mass_max"])
    a["rho_min"] = bool(m.rho >= marks["rho_min"])
    a["no_tie_block_split"] = bool((~gap_conv).sum() == 0 or dist[~gap_conv].min() > m.rho)
    # The gapped conversion (runtime/td3_bc._convert also checks d4rl.qlearning_dataset's output against it).
    conv_g, rows_g, _ = SR.converted_rows({k: raw_g[k] for k in PL.RAW_KEYS})
    keep = ~gap_conv
    a["converted_rows_equal_native_minus_gap"] = bool(np.array_equal(rows_g, rows[keep]))
    a["converted_arrays_equal"] = bool(a["converted_rows_equal_native_minus_gap"] and all(
        np.array_equal(conv_g[k], converted[k][keep]) and conv_g[k].dtype == converted[k].dtype for k in converted))
    a["qlearning_dataset_identity"] = True  # _convert raised otherwise
    ko, kg = per_key_sha256(original), per_key_sha256(gapped)
    a["per_key_sha256"] = bool(set(ko) == set(kg) and all(ko[k] == kg[k] for k in ko if k != "timeouts")
                               and ko == d["original_per_key_sha256"] and kg == d["gapped"]["per_key_sha256"])
    t_o, t_g = np.asarray(raw_o["timeouts"], bool), np.asarray(raw_g["timeouts"], bool)
    changed = np.flatnonzero(t_o != t_g)
    a["timeouts_differ_exactly_at_gap_rows"] = bool(np.array_equal(changed, m.array("gap_rows"))
                                                    and t_g[changed].all())
    # The projected reservation and the gapped train.prepare, checked by runtime validate_prepared.
    row = resolve(TD3_CONFIG, "host", DESIGN["reference_seed"], ROOT / "runs" / "gap-inspection", d["task"])
    row = gapped_row(row, m)
    with projection(m):
        (_, args, spec, protocol), prepared = FS.prepare(row, gapped.parent)
        R.validate_prepared(args, spec, protocol, prepared)
    meta = prepared.metadata
    g_rows = np.asarray(meta["dependency_maps"]["raw_current"], np.int64)
    split_rows = {name: g_rows[np.asarray(meta[f"{key}_converted_ids"], np.int64)]
                  for name, key in (("training", "training"), ("withheld", "withheld"), ("bank", "heldout"))}
    native_rows = m.native_split_rows()
    gap_rows = m.array("gap_rows")
    a["projection_equals_native_minus_gap"] = bool(all(
        np.array_equal(split_rows[k], native_rows[k][~np.isin(native_rows[k], gap_rows)]) for k in split_rows))
    leaks = leak_counts(m, split_rows)
    a["no_gap_row_in_training_withheld_bank"] = bool(all(v == 0 for v in leaks.values()))
    a["validate_prepared_accepts"] = True  # it raised otherwise
    negative = leak_counts(m, {"native_training": native_rows["training"]})["native_training"]
    a["negative_control_detects_leak"] = bool(negative == int(gap_conv[training].sum()) and negative > 0)
    values_a = dict(mass_train=mass, rho=m.rho, centre=m.data["gap"]["centre"], rule=m.data["gap"]["rule"],
                    masses=m.data["gap"]["masses"], counts=m.data["gap"]["counts"], descriptors=d["descriptors"],
                    projected=dict(training=int(len(split_rows["training"])),
                                   withheld=int(len(split_rows["withheld"])), bank=int(len(split_rows["bank"]))),
                    leaks=leaks, negative_control_leaks=negative,
                    gapped_reservation_record_sha256=canonical_sha256(PL._json_ready(meta["reservation"])))
    g0a = dict(passes=a, passed=bool(all(a.values())), values=values_a, marks=dict(
        mass=[marks["mass_min"], marks["mass_max"]], rho_min=marks["rho_min"]))
    # G0b: the kNN geometry with the native normalization.
    mean = np.asarray(d["native"]["obs_mean"], np.float32)
    std = np.asarray(d["native"]["obs_std"], np.float32)
    s_all = standardize(converted["observations"], mean, std)
    x_all = np.concatenate([s_all, actions], axis=1)
    k = d["settings"]["geometry"]["k"]
    tr_kept = training[keep[training]]
    full_x = ArrayThinness(x_all[training], episodes[training], k=k, workers=workers)
    kept_x = ArrayThinness(x_all[tr_kept], episodes[tr_kept], k=k, workers=workers)
    full_s = ArrayThinness(s_all[training], episodes[training], k=k, workers=workers)
    kept_s = ArrayThinness(s_all[tr_kept], episodes[tr_kept], k=k, workers=workers)
    r, r_s = d["geometry"]["r"], d["geometry"]["r_s"]
    q_ids = training[gap_conv[training]]
    values_b, arrays = g0b_counts(x_all[q_ids], s_all[q_ids], episodes[q_ids], full_x, kept_x, full_s, kept_s, r,
                                  r_s, marks["thin_kept_max"])
    rel = dist[q_ids] / m.rho
    strata = {}
    for name, mask in (("core", rel <= 0.5), ("rim", rel > 0.5)):
        strata[name] = dict(rows=int(mask.sum()),
                            c_full_median=float(np.median(arrays["c_full"][mask])) if mask.any() else None,
                            thin_kept_share=float(np.mean(arrays["c_kept_all"][mask] <= marks["thin_kept_max"]))
                            if mask.any() else None,
                            kept_full_ratio_median=float(np.median((arrays["c_kept_oe"][mask] + 1.0)
                                                                   / (arrays["c_full"][mask] + 1.0)))
                            if mask.any() else None)
    held = m.native_ids_of(m.array("pool_G_held"))
    values_held = g0b_counts(x_all[held], s_all[held], episodes[held], full_x, kept_x, full_s, kept_s, r, r_s,
                             marks["thin_kept_max"])[0] if len(held) else None
    is_gap_native = gap_conv
    prev_kept = np.zeros(len(rows), bool)  # gap row i is the next observation of kept row i - 1 (same transition)
    term = np.asarray(maps["terminal_mask"], bool)
    adjacent = np.r_[False, (np.diff(rows) == 1) & ~term[:-1]]
    prev_kept[1:] = adjacent[1:] & ~is_gap_native[:-1]
    match = m.array("pool_K_match_for_G")
    k_rows = match[match >= 0]
    values_b.update(match_rate=float(np.mean(match >= 0)) if len(match) else None, strata=strata,
                    K_match_d_over_rho=_summary(dist[m.native_ids_of(k_rows)] / m.rho) if len(k_rows) else None,
                    G_held=values_held, radius=r, radius_state=r_s,
                    run_lengths=_summary(run_lengths(gap_conv, episodes, rows)),
                    next_obs_of_kept_share=float(prev_kept[q_ids].mean()) if len(q_ids) else None,
                    K_match=d["geometry"]["K_match"], K_far=d["geometry"]["K_far"],
                    state_counts="other ORIGINAL episodes at r_s (the any-episode count is not used)")
    passes_b = g0b_passes(values_b, marks)
    g0b = dict(passes=passes_b, passed=bool(all(passes_b.values())), values=values_b, marks=marks)
    result = dict(head, g0a=g0a, g0b=g0b, passed=bool(g0a["passed"] and g0b["passed"]))
    _write_json(out / "g0.json", result)
    return result


def _write_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf8") as handle:
        handle.write(strict_json(value))
    return path


def freeze(world, gap_sha256, *, exercise_failed_g0=False):
    """Write W/gap/gap.json.sha256 after G0 passed. exercise_failed_g0 is a SYNTHETIC-only override for tests that
    must exercise the later stages on fixture data that fails G0; it is recorded in W/gap/freeze.json."""
    out = Path(world) / "gap"
    m = read_manifest(out / "gap.json", gap_sha256, require_frozen=False)
    require(not (out / "gap.json.sha256").exists(), "the gap is already frozen")
    require((out / "g0.json").is_file(), "run `gap check` first")
    g0 = json.loads((out / "g0.json").read_text(encoding="utf8"))
    require(g0["gap_sha256"] == m.sha256, "g0.json was computed on another gap.json")
    override = False
    if not g0["passed"]:
        require(exercise_failed_g0 and m.data["synthetic"] and m.data["defined"],
                "G0 failed: do not train; any change to the gap needs the user's decision and a new G0 (the "
                "exercise override exists for synthetic test files only)")
        override = True
    record = dict(gap_sha256=m.sha256, g0_sha256=file_sha256(out / "g0.json"), g0_passed=bool(g0["passed"]),
                  synthetic_g0_override=override, frozen_utc=utc(),
                  note=(SYNTHETIC_NOTE + " G0 FAILED and was overridden to exercise later stages.") if override
                  else ("Frozen before any training. Record this sha256 (e.g. commit gap.json) before the first "
                        "host run."))
    _write_json(out / "freeze.json", record)
    (out / "gap.json.sha256").write_text(f"{m.sha256}  gap.json\n", encoding="utf8")
    return m.sha256


# ---------------------------------------------------------------- command line

def main(argv=None):
    parser = argparse.ArgumentParser(description="Action-gap definition, G0 check and freeze (data only).")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("define", help="define the gap, write the gapped copy and the frozen pools")
    d.add_argument("--world", required=True, type=Path)
    d.add_argument("--task", required=True)
    where = d.add_mutually_exclusive_group()
    where.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="pinned cache directory")
    where.add_argument("--file", type=Path, help="a synthetic test file (real data must be the pinned cache)")
    d.add_argument("--gapped-dir", type=Path, help="where the gapped copy goes (default: next to the original)")
    d.add_argument("--profile", default="design", choices=sorted(PROFILES))
    d.add_argument("--workers", type=int, default=1)
    c = sub.add_parser("check", help="G0a and G0b")
    c.add_argument("--world", required=True, type=Path)
    c.add_argument("--gap-sha256", required=True)
    c.add_argument("--data-dir", type=Path)
    c.add_argument("--workers", type=int, default=1)
    f = sub.add_parser("freeze", help="freeze the gap after G0 passed; prints the gap sha256")
    f.add_argument("--world", required=True, type=Path)
    f.add_argument("--gap-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "define":
        m = define(args.world, args.task, data_dir=args.data_dir, file=args.file, profile=args.profile,
                   gapped_dir=args.gapped_dir, workers=args.workers)
        digest = file_sha256(Path(args.world) / "gap" / "gap.json")
        print(strict_json(dict(gap_sha256=digest, defined=m["defined"], synthetic=m["synthetic"],
                               gap=m.get("gap"), reason=m["search"].get("reason"),
                               gapped=None if not m["defined"] else m["gapped"]["path"])))
    elif args.command == "check":
        r = check(args.world, args.gap_sha256, data_dir=args.data_dir, workers=args.workers)
        print(strict_json(dict(passed=r["passed"], g0a=r["g0a"]["passes"],
                               g0b=None if r["g0b"] is None else r["g0b"]["passes"], synthetic=r["synthetic"])))
    else:
        print(freeze(args.world, args.gap_sha256))


if __name__ == "__main__":
    main()
