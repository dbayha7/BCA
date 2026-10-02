"""Step 4 scorecard: the gates G0-G2/G2b and decision rules DR1-DR7 of runs/wbcp_signal/step4/expectations_DRAFT.md,
scored mechanically from score_values.py's output. To be frozen and hashed before the main run (design section 10).

Inputs: a root of run_step4.py process directories (matched_knobs.json, arrays.npz, HASHES.sha256, and score_values.py's
values.npz, values.json, VALUES.sha256) and Stage 0's stage0.json (+ stage0.json.sha256). Only the analysis replicates
count (0-9, --replicates): any other directory under the root (the pilot 99, a smoke run) is skipped before its hashes
or values are looked at, so an unscored one does not stop the scorecard. Every counted directory's hashes are verified;
one whose files differ from their recorded hashes, or that is not scored, stops it (score_values.Refused).

Lost rows (design section 4: unbounded fixed points and unstable closed loops "are both tallied"). An arm's G is NaN
when its row's J is NaN (unbounded) or -inf (unstable), so the contrast drops that cell-replicate (and a SHUF / STRAT
mean drops that permutation), which can flatter the arm. This is never silent: every pooled summary carries
lost_unstable / lost_unbounded (cell-replicates in which an arm it reads, NONE included, was lost) and flag (any -inf
loss), DR1 rows list their flagged contrasts (marked (!) in the .md), DR7 counts its lost cell-replicates, and 'lost'
tallies per (block, regime, placement, arm, set, level, strength) the entries reached with J = -inf or NaN and those
unreached with an unbounded row on the arm's ladder, with per (block, regime) row totals.

Missing versus false. DR6b reads 'not testable' whenever the correlation, T_SHUF, I or T_NUIS is missing (plan
'reduced' runs no OR1 in X-EP), listing them. DR6 and DR3 contain disjunctions (C, I or T_SHUF; V1 at expert or medium),
so they are read in three-valued logic: 'not testable' exactly when a missing input (a V0 verdict, an untestable G1, no
pooled value) could change the result, else confirmed / not confirmed (qualifies / does not qualify), with the missing
and failed inputs listed. G1 is 'not testable' when none of its 8 tests has data.

Per cell and replicate (design section 9). G(arm) = (J(arm) - J(NONE)) / g, g = J_opt - J(K_0), in the same regime;
an arm's G at strength level S_k (S1, S2, S3) reads the runner's matched row (unreachable, below floor or nonfinite:
missing). For the weighting placements P1L, P2L, P1pi, P2pi: T_SHUF = G(SIG) - G(SHUF), T_STRAT = G(SIG) - G(STRAT),
T_NUIS = G(SIG) - G(NUIS), T_ANTI = G(SIG) - G(ANTI), R = G(SIG) - G(P0 at S_k), Str = G(P0 at S_k), I = G(OR1) - G(SHUF),
C = G(OR2) - G(SHUF), with SHUF and STRAT averaged over their reachable permutations. P3: T_NUIS, R, Str and C =
G(P3-OR) - G(P0 at S_k) (the design's C3). C2 (block C): R and Str.

Pooling: for each (block, regime, placement, contrast, evidence set, level, S_k), the mean over the set's cells within
a replicate, then over replicates: mean, paired SE (sd / sqrt(n)), t with n - 1 df, two-sided p and the one-sided p of
"> 0". Material: |mean| >= materiality (default 0.01, 1% of the gap; --materiality). Holm (familywise --alpha, 5%): one
family per placement over {T_SHUF, T_STRAT, T_NUIS, R} x 4 levels in X-CS M at FP (16 tests for P1L, P2L, P1pi, P2pi;
{T_NUIS, R} x 4 = 8 for P3), on two-sided p; the direction is read from the sign. The same families are formed at A500
for the "reverse" clause of V3. Cell-level "meaningful" (|mean| > 2 paired SE and >= materiality) is a secondary count.
Evidence sets come from stage0.json. Primary strength level per placement and regime: S2, or S1 when S2 is unreachable
or below floor in more than 50% of the X-CS M cell-replicates (the fallback rule; reachability only, no J).

Gates. G0: every [derived] check of stage0.json, matched_knobs.json and values.json passes; an '[derived up to
float32/eps]' check ('eps') fails G0 only above 100x its tolerance; 'procedural' checks are counted. G1 (X-CS, FP, N =
clean, noisy_reward; 2 cases x 4 levels, an 8-test Holm family per placement): T_STRAT material and Holm-significant in
at most 1 of 8 for each weighting placement; for P3, |R| material in at most 1 of 8, else P3 is nuisance-confounded and
ineligible for D4RL. G2 (block C, FP, expert, S2): C = G(OR2) - G(SHUF) pooled over the two localized cases >=
materiality with one-sided p < alpha, for P1L or P2L; a failure means stop and redesign. G2b (X-CS cone, FP, expert,
S2): reported per kappa; its flag reads kappa = 2 (expectations item 2.25), or the largest kappa run.

Decision rules. DR1 (X-CS M, FP primary, per placement and level) in order: V0 fewer than 50% of M cell-replicates
reachable at the primary S_k; V1 T_SHUF, T_STRAT, T_NUIS > 0, material, Holm-significant, R >= materiality and
Holm-significant, pooled G(SIG) > G(SHUF) > G(ANTI), and no A500 pooled T_SHUF or R materially opposite (opposite sign,
|A500| >= materiality); V2 the T-conditions with R < materiality (applied literally, before V3: a V2 cell with an A500
reversal stays V2, a reading the pre-registration should confirm); V3 FP meets the T-conditions and A500 is materially
opposite on T_SHUF or R, or A500 meets them (its own Holm family) and FP is materially opposite; if A5000 is not
materially opposite to FP on either, FP's own verdict stands, labelled 'short-horizon reversal'; V4 T_SHUF > 0 material
and significant, T_STRAT immaterial; V5 T_SHUF and T_STRAT > 0 material and significant, T_NUIS immaterial; V6 T_SHUF or
T_STRAT < 0 material and significant; V7 Str material, R and T_SHUF immaterial; V8 nothing. P3: the T-condition is
T_NUIS > 0 material and significant (V1 also R >= materiality, significant); V4-V6 do not apply (no SHUF or STRAT).
DR2 (M, FP): C immaterial: the placement cannot use perfect information; C > 0 material and I immaterial: the
information class is wrong; I > 0 material and T_SHUF immaterial: the signal is too weak; T_SHUF > 0 material: see DR1
(first matching pattern; P3 has no I, so only the first reading applies). DR3: V1 at expert or medium, G1 passed, and on
M_nat (blocks R, X-CS, X-EP pooled per replicate) pooled T_SHUF (P3: T_NUIS) >= materiality with one-sided p < alpha
and pooled R >= 0, at FP when FP is finite in every M_nat cell-replicate, else A500; P3 not nuisance-confounded; the pi
family only after step 5. DR4: among DR3-eligible V1 placements, those with pooled R >= 0 at poor; none when every one
has R < 0. DR5: the majority of placement x level DR2 readings. DR6 (X-CS M, per placement): (a) at expert C, I or
T_SHUF > 0 and material (P3: C or T_NUIS), and pooled Spearman(SIG weight, |e1| at the family's evaluation point) >= 0.3
at all four levels; (b) at poor, S2: G(SIG) or G(OR2) < -materiality with Str < -materiality and |R| < |Str|; the
placement interaction [R(P2) - R(P1)]_poor - [R(P2) - R(P1)]_expert >= materiality and > 2 SE within a family. DR6b
(X-EP mixed, non-control cells, P1L, FP): corr(SIG weight, poor mode) >= 0.05, T_SHUF, I and T_NUIS < -materiality.
DR7: |mean over replicates of G(HOOK) - G(P0 at S(HOOK))| < materiality in at least 90% of cells in every block (A500,
expectations item 1.5; FP also reported).

python experiments/signal/scorecard_step4.py ROOT --stage0 ROOT/stage0.json [--materiality 0.01] [--alpha 0.05]
    [--output ROOT/scorecard_step4]   (writes .json and .md)
Tests: python -m unittest experiments.signal.test_scorecard_step4
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.signal import score_values as SV  # noqa: E402

QUALITIES = ("expert", "mixed", "medium", "poor")
ANALYSIS_REPLICATES = tuple(range(10))
WEIGHTING = ("P1L", "P2L", "P1pi", "P2pi")
PLACEMENTS = WEIGHTING + ("P3",)
FAMILY_CONTRASTS = dict(weighting=("T_SHUF", "T_STRAT", "T_NUIS", "R"), P3=("T_NUIS", "R"))
LEVELS = ("S1", "S2", "S3")
MATERIALITY, ALPHA = 0.01, 0.05
SPEARMAN_MIN, POOR_MODE_MIN, DR7_SHARE, FALLBACK_SHARE, V0_SHARE = 0.3, 0.05, 0.9, 0.5, 0.5
EPS_FACTOR = 100.0


# ---------------------------------------------------------------------------------------------------------
# Loading


class Cell:
    """One cell-replicate: its runner record, J arrays and gap, and its evidence set."""

    def __init__(self, block, level, rep, cid, record, J, gap, sets):
        self.block, self.level, self.rep, self.cid, self.rec, self.J, self.gap = block, level, rep, cid, record, J, gap
        entry = sets.get(block, {}).get(cid, {}).get(level, {})
        self.set, self.M_nat = entry.get("set"), bool(entry.get("M_nat"))
        self.case = record["case"]
        self._perm = {}

    def regime(self, reg):
        return self.rec["regimes"].get(reg)

    def G_row(self, reg, row):
        """G of a row in units of the gap; NaN when the row or NONE is missing, unbounded (J NaN) or unstable (J =
        -inf). Such losses are not silent: status() names them, Pool counts them per contrast and lost_rows tallies
        them."""
        r = self.regime(reg)
        if r is None or row is None:
            return math.nan
        J = self.J.get(reg)
        j, j0 = float(J[row]), float(J[r["index"]["NONE"]])
        return (j - j0) / self.gap if math.isfinite(j) and math.isfinite(j0) else math.nan

    def status(self, reg, row):
        """'missing' (no row), 'unbounded' (J NaN: a nonfinite K, the fixed point's Hessian not positive definite),
        'unstable' (J = -inf: an unstable closed loop) or 'ok' (design section 4: both failures are tallied)."""
        if self.regime(reg) is None or row is None:
            return "missing"
        j = float(self.J[reg][row])
        if math.isfinite(j):
            return "ok"
        return "unstable" if j == -math.inf else "unbounded"

    def match(self, reg, arm, level):
        r = self.regime(reg)
        return None if r is None else r["index"]["matched"].get(arm, {}).get(level)

    def G(self, reg, arm, level):
        m = self.match(reg, arm, level)
        return self.G_row(reg, None if m is None else m.get("row"))

    def G_ref(self, reg, name):
        r = self.regime(reg)
        return math.nan if r is None else self.G_row(reg, r["index"]["refs"].get(name))

    def reachable(self, reg, arm, level):
        m = self.match(reg, arm, level)
        return bool(m and m.get("row") is not None)

    def perm_arms(self, reg, placement, kind):
        """The placement's permutation arms of one kind (SHUF_k, STRAT_k or SHUFLIN_k) that the regime ran."""
        key = (reg, placement, kind)
        if key not in self._perm:
            r = self.regime(reg)
            pattern = re.compile(rf"{re.escape(placement)}:{kind}\d+")
            self._perm[key] = [] if r is None else [a for a in r["index"]["matched"] if pattern.fullmatch(a)]
        return self._perm[key]

    def diag(self, key):
        return self.rec.get("diagnostics", {}).get(key)


def load(root, stage0_path, replicates=ANALYSIS_REPLICATES, log=print):
    """Every scored process directory under root (hashes verified), with Stage 0's sets."""
    stage0_path = Path(stage0_path)
    sha_file = Path(str(stage0_path) + ".sha256")
    if not sha_file.is_file():
        raise SV.Refused(f"{stage0_path}: no .sha256")
    recorded = SV.read_hashes(sha_file).get(stage0_path.name)
    if recorded != SV.sha256_file(stage0_path):
        raise SV.Refused(f"{stage0_path}: sha256 differs from its record")
    stage0 = json.loads(stage0_path.read_text(encoding="utf8"))
    cells, checks, procs = [], [dict(c, source="stage0") for c in stage0["checks"]], []
    for d in SV.process_dirs([root]):
        if SV.read_meta(d)["replicate"] not in replicates:  # the pilot, smoke runs: never read further
            continue
        SV.verify(d)
        vfile = d / SV.VALUES_HASH_FILE
        if not vfile.is_file():
            raise SV.Refused(f"{d}: not scored (no {SV.VALUES_HASH_FILE})")
        vh = SV.read_hashes(vfile)
        for name in ("values.json", "values.npz"):
            if vh.get(name) != SV.sha256_file(d / name):
                raise SV.Refused(f"{d}: {name} differs from {SV.VALUES_HASH_FILE}")
        payload = json.loads((d / "matched_knobs.json").read_text(encoding="utf8"))
        meta = payload["meta"]
        values = json.loads((d / "values.json").read_text(encoding="utf8"))
        npz = np.load(d / "values.npz")
        procs.append(dict(dir=str(d), block=meta["block"], level=meta["level"], replicate=meta["replicate"]))
        for cid, rec in payload["cells"].items():
            J = {reg: npz[f"J/{cid}/{reg}"] for reg in rec["regimes"]}
            cells.append(Cell(meta["block"], meta["level"], meta["replicate"], cid, rec, J,
                              values["cells"][cid]["gap"], stage0["sets"]))
            checks += [dict(c, source=f"{d}:{cid}") for c in rec["checks"]]
        checks += [dict(c, source=str(d)) for c in values["checks"]]
    log(f"scorecard: {len(procs)} process directories, {len(cells)} cell-replicates")
    return cells, checks, stage0, procs


# ---------------------------------------------------------------------------------------------------------
# Contrasts and pooling


def nanmean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.mean(xs)) if xs else math.nan


# The arms each contrast reads ('*': every permutation arm of that kind the regime ran; 'P0': the uniform reference).
CONTRAST_ARMS = dict(
    weighting=dict(T_SHUF=("SIG", "SHUF*"), T_STRAT=("SIG", "STRAT*"), T_NUIS=("SIG", "NUIS"), T_ANTI=("SIG", "ANTI"),
                   R=("SIG", "P0"), Str=("P0",), I=("OR1", "SHUF*"), C=("OR2", "SHUF*"), T_LIN=("TLIN", "SHUFLIN*"),
                   G_SIG=("SIG",), G_SHUF=("SHUF*",), G_ANTI=("ANTI",), G_OR2=("OR2",)),
    P3=dict(T_NUIS=("P3:SIG", "P3:NUIS"), R=("P3:SIG", "P0"), Str=("P0",), C=("P3:OR", "P0"), G_SIG=("P3:SIG",)),
    C2=dict(R=("C2", "P0"), Str=("P0",)))


def contrast_arms(cell, reg, placement, contrast):
    kind = "weighting" if placement in WEIGHTING else placement
    out = []
    for a in CONTRAST_ARMS[kind].get(contrast, ()):
        if a.endswith("*"):
            out += cell.perm_arms(reg, placement, a[:-1])
        else:
            out.append(a if a == "P0" or kind != "weighting" else f"{placement}:{a}")
    return out


def losses(cell, reg, placement, level):
    """{contrast: ['arm:unstable' | 'arm:unbounded', ...]}: the arms a contrast reads at this level (every
    permutation arm, and NONE, whose loss voids every G of the regime) whose row's J is -inf or NaN. The contrast then
    drops the cell (or, for SHUF / STRAT, averages the remaining permutations), so the pooled summary counts it."""
    r = cell.regime(reg)
    if r is None:
        return {}
    none = cell.status(reg, r["index"]["NONE"])
    kind = "weighting" if placement in WEIGHTING else placement
    out = {}
    for contrast in CONTRAST_ARMS[kind]:
        lost = [] if none == "ok" else [f"NONE:{none}"]
        for arm in contrast_arms(cell, reg, placement, contrast):
            m = cell.match(reg, arm, level)
            st = cell.status(reg, None if m is None else m.get("row"))
            if st in ("unstable", "unbounded"):
                lost.append(f"{arm}:{st}")
        if lost:
            out[contrast] = lost
    return out


def k_and(*xs):
    """Three-valued AND (None: a missing input): False if any is False, else None if any is missing, else True."""
    return False if any(x is False for x in xs) else (None if any(x is None for x in xs) else True)


def k_or(*xs):
    """Three-valued OR: True if any is True, else None if any is missing, else False."""
    return True if any(x is True for x in xs) else (None if any(x is None for x in xs) else False)


def reading(value, missing, failed=()):
    """A rule's reading: 'confirmed', 'not confirmed' or 'not testable' (its result depends on missing inputs)."""
    return dict(status={True: "confirmed", False: "not confirmed", None: "not testable"}[value], missing=list(missing),
                failed=list(failed))


def contrasts(cell, reg, placement, level):
    """The cell's contrasts for one placement at one strength level (missing: NaN)."""
    p0 = cell.G(reg, "P0", level)
    if placement in WEIGHTING:
        g = lambda arm: cell.G(reg, f"{placement}:{arm}", level)
        sig = g("SIG")
        mean_of = lambda kind: nanmean([cell.G(reg, a, level) for a in cell.perm_arms(reg, placement, kind)])
        shuf, strat = mean_of("SHUF"), mean_of("STRAT")
        anti, nuis, or1, or2 = g("ANTI"), g("NUIS"), g("OR1"), g("OR2")
        return dict(T_SHUF=sig - shuf, T_STRAT=sig - strat, T_NUIS=sig - nuis, T_ANTI=sig - anti, R=sig - p0, Str=p0,
                    I=or1 - shuf, C=or2 - shuf, T_LIN=g("TLIN") - mean_of("SHUFLIN"), G_SIG=sig, G_SHUF=shuf,
                    G_ANTI=anti, G_OR2=or2)
    if placement == "P3":
        sig = cell.G(reg, "P3:SIG", level)
        return dict(T_NUIS=sig - cell.G(reg, "P3:NUIS", level), R=sig - p0, Str=p0,
                    C=cell.G(reg, "P3:OR", level) - p0, G_SIG=sig)
    if placement == "C2":
        return dict(R=cell.G(reg, "C2", level) - p0, Str=p0)
    raise ValueError(placement)


def summary(values):
    """Mean, paired SE, t, two-sided p and one-sided p (> 0) of per-replicate values."""
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], np.float64)
    out = dict(n=int(x.size), mean=float(x.mean()) if x.size else None, se=None, t=None, p=None, p_pos=None,
               values=x.tolist())
    if x.size >= 2:
        se = float(x.std(ddof=1) / math.sqrt(x.size))
        out["se"] = se
        if se > 0:
            t = out["mean"] / se
            out.update(t=t, p=float(2 * stats.t.sf(abs(t), x.size - 1)), p_pos=float(stats.t.sf(t, x.size - 1)))
        else:
            out.update(p=0.0 if out["mean"] != 0 else 1.0,
                       p_pos=0.0 if out["mean"] > 0 else 1.0)
    return out


class Pool:
    """Per-cell contrasts, pooled per replicate over an evidence set and then over replicates."""

    def __init__(self, cells, materiality=MATERIALITY):
        self.cells, self.mat = cells, materiality
        self._cache, self._lost = {}, {}

    def select(self, block=None, sets=None, level=None, case=None, M_nat=None, cell_ids=None):
        out = []
        for c in self.cells:
            if block is not None and c.block not in (block if isinstance(block, tuple) else (block,)):
                continue
            if sets is not None and c.set not in sets:
                continue
            if level is not None and c.level != level:
                continue
            if case is not None and c.case not in (case if isinstance(case, tuple) else (case,)):
                continue
            if M_nat is not None and c.M_nat != M_nat:
                continue
            if cell_ids is not None and c.cid not in cell_ids:
                continue
            out.append(c)
        return out

    def per_cell(self, cell, reg, placement, strength):
        key = (id(cell), reg, placement, strength)
        if key not in self._cache:
            self._cache[key] = contrasts(cell, reg, placement, strength)
        return self._cache[key]

    def per_cell_lost(self, cell, reg, placement, strength):
        key = (id(cell), reg, placement, strength)
        if key not in self._lost:
            self._lost[key] = losses(cell, reg, placement, strength)
        return self._lost[key]

    def pooled(self, cells, reg, placement, contrast, strength):
        """Replicate-pooled summary of one contrast over the given cells (mean over cells within a replicate).

        lost_unstable / lost_unbounded count the cell-replicates in which an arm the contrast reads (NONE included)
        has J = -inf / NaN; the contrast drops those cells (a SHUF / STRAT mean drops the permutation), which can flatter
        an arm, so flag marks a summary with any -inf loss instead of dropping it silently."""
        by_rep, unstable, unbounded = {}, 0, 0
        for c in cells:
            v = self.per_cell(c, reg, placement, strength).get(contrast, math.nan)
            if math.isfinite(v):
                by_rep.setdefault(c.rep, []).append(v)
            lost = self.per_cell_lost(c, reg, placement, strength).get(contrast, ())
            unstable += any(x.endswith(":unstable") for x in lost)
            unbounded += any(x.endswith(":unbounded") for x in lost)
        out = summary([float(np.mean(v)) for _, v in sorted(by_rep.items())])
        out["replicates"] = sorted(by_rep)
        out["cells"] = len({c.cid for c in cells})
        out.update(lost_unstable=unstable, lost_unbounded=unbounded, flag=unstable > 0)
        return out

    def material(self, s):
        return s["mean"] is not None and abs(s["mean"]) >= self.mat

    def reach_share(self, cells, reg, placement, strength):
        arm = "P3:SIG" if placement == "P3" else ("C2" if placement == "C2" else f"{placement}:SIG")
        if not cells:
            return None
        return sum(c.reachable(reg, arm, strength) for c in cells) / len(cells)


def holm(pvalues):
    """Holm-adjusted p-values for {key: p} (a missing p counts as 1)."""
    keys = sorted(pvalues, key=lambda k: 1.0 if pvalues[k] is None else pvalues[k])
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        p = 1.0 if pvalues[k] is None else pvalues[k]
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# ---------------------------------------------------------------------------------------------------------
# Gates and decision rules


class Scorecard:
    def __init__(self, cells, checks, materiality=MATERIALITY, alpha=ALPHA):
        self.pool = Pool(cells, materiality)
        self.cells, self.checks, self.mat, self.alpha = cells, checks, materiality, alpha
        self.primary = {}
        for reg in ("FP", "A500", "A5000"):
            for p in PLACEMENTS:
                self.primary[(p, reg)] = self.primary_level(p, reg)

    # -- strength level and families --------------------------------------------------------------------------

    def primary_level(self, placement, reg):
        """S2, or S1 when S2 is unreachable or below floor in > 50% of X-CS M cell-replicates (fallback rule)."""
        cells = self.pool.select(block="X-CS", sets=("M",))
        share = self.pool.reach_share(cells, reg, placement, "S2")
        return "S1" if share is not None and share < 1 - FALLBACK_SHARE else "S2"

    def family(self, placement, reg="FP"):
        """Placement's Holm family over {contrasts} x 4 levels in X-CS M at the primary level."""
        kind = "P3" if placement == "P3" else "weighting"
        level = self.primary[(placement, reg)]
        tests = {}
        for contrast in FAMILY_CONTRASTS[kind]:
            for q in QUALITIES:
                cells = self.pool.select(block="X-CS", sets=("M",), level=q)
                tests[(contrast, q)] = self.pool.pooled(cells, reg, placement, contrast, level)
        adj = holm({k: s["p"] for k, s in tests.items()})
        for k, s in tests.items():
            s["p_holm"] = adj[k]
        return tests

    def sig(self, s):
        return s.get("p_holm") is not None and s["p_holm"] < self.alpha

    def pos(self, s):
        return s["mean"] is not None and s["mean"] > 0 and self.pool.material(s) and self.sig(s)

    def neg(self, s):
        return s["mean"] is not None and s["mean"] < 0 and self.pool.material(s) and self.sig(s)

    def opposite(self, a, b):
        """b materially opposite to a: opposite sign with |b| >= materiality."""
        return (a["mean"] is not None and b["mean"] is not None and np.sign(a["mean"]) * np.sign(b["mean"]) < 0
                and abs(b["mean"]) >= self.mat)

    # -- G0, G1, G2, G2b ------------------------------------------------------------------------------------------

    def g0(self):
        failed, eps_reported, procedural = [], [], []
        for c in self.checks:
            if c["kind"] == "derived" and not c["ok"]:
                failed.append(c)
            elif c["kind"] == "eps" and not c["ok"]:
                tol, value = c.get("tol"), c.get("value")
                big = tol is not None and isinstance(value, (int, float)) and value > EPS_FACTOR * tol
                (failed if big else eps_reported).append(c)
            elif c["kind"] == "procedural" and not c["ok"]:
                procedural.append(c)
        return dict(passed=not failed, checks=len(self.checks), failed=failed, eps_misses=eps_reported,
                    procedural_misses=len(procedural), procedural=procedural[:50])

    def g1(self):
        out, nuisance = {}, None
        for p in PLACEMENTS:
            level = self.primary[(p, "FP")]
            contrast = "R" if p == "P3" else "T_STRAT"
            tests = {}
            for case in ("clean", "noisy_reward"):
                for q in QUALITIES:
                    cells = self.pool.select(block="X-CS", sets=("N",), level=q, case=case)
                    tests[(case, q)] = self.pool.pooled(cells, "FP", p, contrast, level)
            adj = holm({k: s["p"] for k, s in tests.items()})
            if p == "P3":
                count = sum(self.pool.material(s) for s in tests.values())
            else:
                count = sum(self.pool.material(s) and adj[k] < self.alpha for k, s in tests.items())
            passed = count <= 1
            with_data = sum(s["n"] > 0 for s in tests.values())  # no data in any of the 8: the gate is not testable
            out[p] = dict(contrast=contrast, level=level, count=count, passed=passed, testable=with_data > 0,
                          tests_with_data=with_data,
                          tests={f"{k[0]}/{k[1]}": dict(s, p_holm=adj[k]) for k, s in tests.items()})
            if p == "P3":
                nuisance = (not passed) if with_data else None
        return dict(placements=out, p3_nuisance_confounded=nuisance)

    def g2(self):
        out = {}
        for p in ("P1L", "P2L"):
            per = {}
            for level in LEVELS:
                cells = self.pool.select(block="C", level="expert", case=("q1opt-local", "tilt-local"))
                per[level] = self.pool.pooled(cells, "FP", p, "C", level)
            s = per["S2"]
            out[p] = dict(levels=per, passed=bool(s["mean"] is not None and s["mean"] >= self.mat
                                                  and s["p_pos"] is not None and s["p_pos"] < self.alpha))
        return dict(placements=out, passed=any(v["passed"] for v in out.values()))

    def g2b(self):
        cones = sorted({c.rec["kappa"] for c in self.cells if c.block == "X-CS" and c.case == "cone"})
        out = {}
        for k in cones:
            ids = {c.cid for c in self.cells if c.case == "cone" and c.rec["kappa"] == k}
            cells = self.pool.select(block="X-CS", level="expert", cell_ids=ids)
            out[f"{k:g}"] = {p: self.pool.pooled(cells, "FP", p, "C", "S2") for p in ("P1L", "P2L")}
        key = "2" if "2" in out else (f"{cones[-1]:g}" if cones else None)
        passed = (key is not None and any(s["mean"] is not None and s["mean"] >= self.mat
                                          for s in out[key].values()))
        return dict(kappas=out, flag_kappa=key, passed=bool(passed))

    # -- DR1 --------------------------------------------------------------------------------------------------------

    def contrast(self, placement, contrast, q, reg="FP", level=None, sets=("M",), block="X-CS"):
        level = level or self.primary[(placement, reg)]
        return self.pool.pooled(self.pool.select(block=block, sets=sets, level=q), reg, placement, contrast, level)

    def t_conditions(self, placement, fam, q):
        if placement == "P3":
            return self.pos(fam[("T_NUIS", q)])
        return all(self.pos(fam[(c, q)]) for c in ("T_SHUF", "T_STRAT", "T_NUIS"))

    def verdicts(self):
        out = {}
        for p in PLACEMENTS:
            fam, fam_a500 = self.family(p, "FP"), self.family(p, "A500")
            key = "T_NUIS" if p == "P3" else "T_SHUF"
            out[p] = {}
            for q in QUALITIES:
                level = self.primary[(p, "FP")]
                cells = self.pool.select(block="X-CS", sets=("M",), level=q)
                share = self.pool.reach_share(cells, "FP", p, level)
                R, T = fam[("R", q)], fam[(key, q)]
                a500 = {c: self.contrast(p, c, q, "A500") for c in (key, "R")}
                a5000 = {c: self.contrast(p, c, q, "A5000") for c in (key, "R")}
                Str = self.contrast(p, "Str", q)
                row = dict(level=level, reach_share=share, contrasts={f"{c}": fam[(c, q)] for c, qq in fam if qq == q},
                           Str=Str, A500=a500, A5000=a5000)
                row["flagged"] = sorted([f"FP:{c}" for (c, qq), s in fam.items() if qq == q and s["flag"]]
                                        + [f"{reg}:{c}" for reg, d in (("A500", a500), ("A5000", a5000))
                                           for c, s in d.items() if s["flag"]] + (["FP:Str"] if Str["flag"] else []))
                if p != "P3":
                    row["G"] = {g: self.contrast(p, g, q)["mean"] for g in ("G_SIG", "G_SHUF", "G_ANTI")}
                fp_opposed = any(self.opposite(fam[(c, q)], a500[c]) for c in (key, "R"))
                a500_t = self.t_conditions(p, fam_a500, q)
                a500_opposed = any(self.opposite(a500[c], fam[(c, q)]) for c in (key, "R"))
                fp_only = self.fp_verdict(p, fam, q, Str, row)
                if share is None or share < V0_SHARE:
                    v = "V0"
                elif self.t_conditions(p, fam, q) and not fp_opposed and fp_only == "V1":
                    v = "V1"
                elif self.t_conditions(p, fam, q) and R["mean"] is not None and R["mean"] < self.mat:
                    v = "V2"
                elif (self.t_conditions(p, fam, q) and fp_opposed) or (a500_t and a500_opposed):
                    v = f"{fp_only} (short-horizon reversal)" if self.agrees(fam, a5000, key, q) else "V3"
                else:
                    v = fp_only
                row["verdict"] = v
                out[p][q] = row
        return out

    def agrees(self, fam, a5000, key, q):
        """A5000 agrees with FP: on both T (T_SHUF; P3 T_NUIS) and R it has FP's sign and is material. A null A5000
        does not count as agreement, so it cannot turn a reversal into 'short-horizon reversal'."""
        for c in (key, "R"):
            f, a = fam[(c, q)]["mean"], a5000[c]["mean"]
            if f is None or a is None or np.sign(f) != np.sign(a) or abs(a) < self.mat:
                return False
        return True

    def fp_verdict(self, p, fam, q, Str, row):
        """FP's own verdict without the A500 concordance clauses (V1, V2, V4-V8)."""
        R = fam[("R", q)]
        if self.t_conditions(p, fam, q):
            if p != "P3":
                g = row.get("G", {})
                order = (None not in g.values()) and g["G_SIG"] > g["G_SHUF"] > g["G_ANTI"]
            else:
                order = True
            if R["mean"] is not None and R["mean"] >= self.mat and self.sig(R) and order:
                return "V1"
            if R["mean"] is not None and R["mean"] < self.mat:
                return "V2"
        if p != "P3":
            TS, TT, TN = fam[("T_SHUF", q)], fam[("T_STRAT", q)], fam[("T_NUIS", q)]
            if self.pos(TS) and not self.pool.material(TT):
                return "V4"
            if self.pos(TS) and self.pos(TT) and not self.pool.material(TN):
                return "V5"
            if self.neg(TS) or self.neg(TT):
                return "V6"
            if self.pool.material(Str) and not self.pool.material(R) and not self.pool.material(TS):
                return "V7"
        elif self.pool.material(Str) and not self.pool.material(R):
            return "V7"
        return "V8"

    # -- DR2-DR7 --------------------------------------------------------------------------------------------------

    def information(self):
        out = {}
        for p in PLACEMENTS:
            out[p] = {}
            for q in QUALITIES:
                C, I, T = (self.contrast(p, c, q) for c in ("C", "I", "T_SHUF"))
                if C["mean"] is None:
                    reading = "no data"
                elif not self.pool.material(C):
                    reading = "cannot use perfect information"
                elif p == "P3":
                    reading = "n/a (P3 has no I)"
                elif C["mean"] > 0 and I["mean"] is not None and not self.pool.material(I):
                    reading = "information class wrong"
                elif I["mean"] is not None and I["mean"] > 0 and self.pool.material(I) and not self.pool.material(T):
                    reading = "signal too weak"
                elif T["mean"] is not None and T["mean"] > 0 and self.pool.material(T):
                    reading = "see DR1"
                else:
                    reading = "unclassified"
                out[p][q] = dict(reading=reading, C=C["mean"], I=I["mean"], T_SHUF=T["mean"])
        return out

    def carry(self, verdicts, g1):
        """DR3 in three-valued logic: qualifies True / False, or None ('not testable') when the result depends on a
        missing input (a V0 verdict, an untestable G1, no pooled M_nat T or R, or no P3 nuisance flag)."""
        out = {}
        for p in PLACEMENTS:
            vs = {q: verdicts[p][q]["verdict"] for q in ("expert", "medium")}
            v1 = k_or(*(None if v.startswith("V0") else v.startswith("V1") for v in vs.values()))
            g1p = g1["placements"][p]
            g1_ok = g1p["passed"] if g1p["testable"] else None
            key = "T_NUIS" if p == "P3" else "T_SHUF"
            level = self.primary[(p, "FP")]
            cells = self.pool.select(block=("R", "X-CS", "X-EP"), M_nat=True)
            finite = all(all(math.isfinite(self.pool.per_cell(c, "FP", p, level).get(k, math.nan)) for k in (key, "R"))
                         for c in cells) and bool(cells)
            reg = "FP" if finite else "A500"
            lv = self.primary[(p, reg)]
            T = self.pool.pooled(cells, reg, p, key, lv)
            R = self.pool.pooled(cells, reg, p, "R", lv)
            nat_T = (None if T["mean"] is None or T["p_pos"] is None
                     else T["mean"] >= self.mat and T["p_pos"] < self.alpha)
            nat_R = None if R["mean"] is None else R["mean"] >= 0
            nat = k_and(nat_T, nat_R)
            conf = g1["p3_nuisance_confounded"]
            nuis_ok = True if p != "P3" else (None if conf is None else not conf)
            ok = k_and(v1, g1_ok, nat, nuis_ok)
            inputs = {"V1 expert/medium": v1, "G1": g1_ok, f"M_nat {key}": nat_T, "M_nat R": nat_R,
                      "P3 not nuisance-confounded": nuis_ok}
            out[p] = dict(qualifies=ok, conditional_on_step5=p.endswith("pi") and ok is True, V1_expert_or_medium=v1,
                          G1=g1_ok, M_nat=dict(regime=reg, level=lv, cells=len({c.cid for c in cells}), T=T, R=R,
                                               passed=nat),
                          **reading(ok, [k for k, v in inputs.items() if v is None],
                                    [k for k, v in inputs.items() if v is False]))
            out[p]["status"] = {True: "qualifies", False: "does not qualify", None: "not testable"}[ok]
        out["any"] = any(v["qualifies"] is True for k, v in out.items() if k in PLACEMENTS)
        return out

    def poor_data(self, verdicts, carry):
        v1 = [p for p in PLACEMENTS if any(verdicts[p][q]["verdict"].startswith("V1") for q in ("expert", "medium"))]
        R = {p: self.contrast(p, "R", "poor")["mean"] for p in v1}
        preferred = [p for p in v1 if R[p] is not None and R[p] >= 0]
        return dict(V1=v1, R_poor=R, preferred=preferred,
                    none_for_poor=bool(v1) and not preferred)

    def step_order(self, info):
        readings = [info[p][q]["reading"] for p in PLACEMENTS for q in QUALITIES]
        total = len(readings)
        wrong = sum(r == "information class wrong" for r in readings)
        weak = sum(r == "signal too weak" for r in readings)
        return dict(total=total, information_class_wrong=wrong, signal_too_weak=weak,
                    step5_first=wrong > total / 2, sigma_quality_first=weak > total / 2)

    def hypothesis(self):
        """DR6 per placement in three-valued logic: identification = (C, I or T_SHUF > 0 material at expert; P3: C or
        T_NUIS) and Spearman >= 0.3 at all four levels; harm = (G(SIG) or G(OR2) < -materiality; P3: G(SIG)) and Str <
        -materiality and |R| < |Str| at poor, S2. confirmed True / False, or None ('not testable', with the missing
        inputs) when the result depends on a missing input."""
        out = {}
        for p in PLACEMENTS:
            fam = "pi" if p.endswith("pi") or p == "P3" else "L"
            exp = {c: self.contrast(p, c, "expert") for c in ("C", "I", "T_SHUF", "T_NUIS")}
            ident_from = ("C", "T_NUIS") if p == "P3" else ("C", "I", "T_SHUF")
            pos = {c: None if exp[c]["mean"] is None else exp[c]["mean"] > 0 and self.pool.material(exp[c])
                   for c in ident_from}
            rho = {}
            for q in QUALITIES:
                vals = [c.diag(f"spearman_sig_e1_{fam}") for c in self.pool.select(block="X-CS", sets=("M",), level=q)]
                rho[q] = nanmean(vals)
            spear = {q: rho[q] >= SPEARMAN_MIN if math.isfinite(rho[q]) else None for q in QUALITIES}
            ident = k_and(k_or(*pos.values()), k_and(*spear.values()))
            poor = {c: self.contrast(p, c, "poor", level="S2") for c in ("G_SIG", "G_OR2", "Str", "R")}
            mean = {c: s["mean"] for c, s in poor.items()}
            below = {c: None if mean[c] is None else mean[c] < -self.mat for c in mean}
            g_from = ("G_SIG",) if p == "P3" else ("G_SIG", "G_OR2")  # P3 has no OR2 arm
            r_small = None if mean["R"] is None or mean["Str"] is None else abs(mean["R"]) < abs(mean["Str"])
            harm = k_and(k_or(*(below[c] for c in g_from)), below["Str"], r_small)
            ok = k_and(ident, harm)
            inputs = {**{f"expert {c}": pos[c] for c in ident_from}, **{f"spearman {q}": spear[q] for q in QUALITIES},
                      **{f"poor {c}": below[c] for c in g_from + ("Str",)}, "poor |R| < |Str|": r_small}
            out[p] = dict(identification=ident, spearman=rho, harm=harm, confirmed=ok, expert=exp, poor=poor,
                          **reading(ok, [k for k, v in inputs.items() if v is None],
                                    [k for k, v in inputs.items() if v is False]))
        out["interaction"] = {fam: self.interaction(p1, p2) for fam, (p1, p2) in
                              dict(L=("P1L", "P2L"), pi=("P1pi", "P2pi")).items()}
        return out

    def interaction(self, p1, p2):
        """[R(P2) - R(P1)]_poor - [R(P2) - R(P1)]_expert per replicate (X-CS M, FP primary level of P1)."""
        level = self.primary[(p1, "FP")]
        per = {}
        for q in ("poor", "expert"):
            for c in self.pool.select(block="X-CS", sets=("M",), level=q):
                d = (self.pool.per_cell(c, "FP", p2, level)["R"] - self.pool.per_cell(c, "FP", p1, level)["R"])
                if math.isfinite(d):
                    per.setdefault(c.rep, {}).setdefault(q, []).append(d)
        vals = [np.mean(v["poor"]) - np.mean(v["expert"]) for v in per.values() if "poor" in v and "expert" in v]
        s = summary(vals)
        s["confirmed"] = (None if s["mean"] is None or s["se"] is None
                          else bool(s["mean"] >= self.mat and s["mean"] > 2 * s["se"]))
        return s

    def mixed_pathway(self):
        """DR6b (X-EP mixed, non-control cells, P1L, FP): 'not testable' whenever the correlation, T_SHUF, I or T_NUIS
        is missing (e.g. plan 'reduced' runs no OR1 in X-EP), with the missing inputs listed; otherwise confirmed when
        all four conditions hold."""
        cells = [c for c in self.pool.select(block="X-EP", level="mixed") if c.set not in ("N", "check", None)]
        level = self.primary[("P1L", "FP")]
        corr = nanmean([c.diag("corr_sig_poor_mode") for c in cells])
        T, I, N = (self.pool.pooled(cells, "FP", "P1L", k, level) for k in ("T_SHUF", "I", "T_NUIS"))
        inputs = dict(corr_poor_mode=corr if math.isfinite(corr) else None, T_SHUF=T["mean"], I=I["mean"],
                      T_NUIS=N["mean"])
        cond = {k: (None if v is None else (v >= POOR_MODE_MIN if k == "corr_poor_mode" else v < -self.mat))
                for k, v in inputs.items()}
        missing = [k for k, v in inputs.items() if v is None]
        ok = None if missing else all(cond.values())
        return dict(cells=len({c.cid for c in cells}), corr_poor_mode=corr, T_SHUF=T, I=I, T_NUIS=N, confirmed=ok,
                    **reading(ok, missing, [k for k, v in cond.items() if v is False]))

    def hook(self):
        out = {}
        for reg in ("A500", "FP"):
            per_block, groups, lost = {}, {}, {}
            for c in self.cells:
                if c.case == "q2_optimistic":
                    continue
                d = c.G_ref(reg, "HOOK") - c.G(reg, "P0", "S(HOOK)")
                if math.isfinite(d):
                    groups.setdefault((c.block, c.cid, c.level), []).append(d)
                r = c.regime(reg)
                if r is not None:  # HOOK, P0 at S(HOOK) or NONE unstable: the cell-replicate drops out, counted here
                    m = c.match(reg, "P0", "S(HOOK)")
                    rows = (r["index"]["refs"].get("HOOK"), None if m is None else m.get("row"), r["index"]["NONE"])
                    if any(c.status(reg, row) == "unstable" for row in rows):
                        lost[c.block] = lost.get(c.block, 0) + 1
            for (block, _, _), ds in groups.items():
                per_block.setdefault(block, []).append(abs(np.mean(ds)) < self.mat)
            shares = {b: float(np.mean(v)) for b, v in per_block.items()}
            out[reg] = dict(immaterial_share=shares, cells={b: len(v) for b, v in per_block.items()},
                            lost_unstable=lost, flag=bool(lost),
                            alpha_retune=bool(shares) and all(s >= DR7_SHARE for s in shares.values()))
        return out

    def lost_rows(self):
        """The tally of design section 4 ('both are tallied'): per (block, regime, placement, arm, set, level,
        strength) the entries with a usable target (n), those reached, reached rows whose J is -inf (unstable) or NaN
        (unbounded; matching never reaches one, so expected 0), and unreached entries whose arm has an unbounded row on
        its ladder (the nonfinite fixed point may have hidden the crossing). NONE ('base') and the references ('ref')
        are counted too. Only groups with a loss are listed; totals count every stored row per (block, regime)."""
        groups, totals = {}, {}
        for c in self.cells:
            for reg, r in c.rec["regimes"].items():
                J = np.asarray(c.J[reg], np.float64)
                t = totals.setdefault(f"{c.block}/{reg}", dict(rows=0, unbounded=0, unstable=0))
                t["rows"] += int(J.size)
                t["unbounded"] += int(np.sum(np.isnan(J)))
                t["unstable"] += int(np.sum(J == -np.inf))
                idx = r["index"]
                entries = [("NONE", "base", idx["NONE"], False)]
                entries += [(name, "ref", row, False) for name, row in idx.get("refs", {}).items()]
                for arm, levels in idx.get("matched", {}).items():
                    ladder_nan = bool(np.any(np.isnan(J[idx.get("ladder", {}).get(arm, [])])))
                    entries += [(arm, lv, m.get("row"), ladder_nan) for lv, m in levels.items() if not m.get("no_target")]
                for arm, lv, row, ladder_nan in entries:
                    placement = arm.partition(":")[0]
                    g = groups.setdefault((c.block, reg, placement, arm, c.set, c.level, lv),
                                          dict(n=0, reached=0, unstable=0, unbounded=0, unreached_unbounded_ladder=0))
                    st = c.status(reg, row)
                    g["n"] += 1
                    g["reached"] += st != "missing"
                    g["unstable"] += st == "unstable"
                    g["unbounded"] += st == "unbounded"
                    g["unreached_unbounded_ladder"] += st == "missing" and ladder_nan
        keys = ("block", "regime", "placement", "arm", "set", "level", "strength")
        lost = [dict(zip(keys, k), **g) for k, g in sorted(groups.items(), key=lambda kv: str(kv[0]))
                if g["unstable"] or g["unbounded"] or g["unreached_unbounded_ladder"]]
        return dict(totals=totals, groups=lost,
                    unstable=sum(g["unstable"] for g in lost), unbounded=sum(g["unbounded"] for g in lost),
                    unreached_unbounded_ladder=sum(g["unreached_unbounded_ladder"] for g in lost))

    def meaningful(self):
        """Secondary count: X-CS M cells whose mean over replicates exceeds 2 paired SE and the materiality (FP)."""
        out = {}
        for p in PLACEMENTS:
            level = self.primary[(p, "FP")]
            for contrast in FAMILY_CONTRASTS["P3" if p == "P3" else "weighting"]:
                for q in QUALITIES:
                    groups = {}
                    for c in self.pool.select(block="X-CS", sets=("M",), level=q):
                        v = self.pool.per_cell(c, "FP", p, level).get(contrast, math.nan)
                        if math.isfinite(v):
                            groups.setdefault(c.cid, []).append(v)
                    count = 0
                    for vs in groups.values():
                        s = summary(vs)
                        count += bool(s["se"] is not None and abs(s["mean"]) > 2 * s["se"] and abs(s["mean"]) >= self.mat)
                    out[f"{p}/{contrast}/{q}"] = dict(meaningful=count, cells=len(groups))
        return out

    def tables(self):
        """Every pooled contrast: block x regime x placement x contrast x set x level x S_k (non-empty only)."""
        groups = {}
        for c in self.cells:
            if c.set in ("N", "M", "W", "H", "P"):
                groups.setdefault((c.block, c.set, c.level), []).append(c)
        out = []
        for (block, st, q), cells in sorted(groups.items()):
            regimes = sorted({reg for c in cells for reg in c.rec["regimes"]})
            for reg in regimes:
                for p in PLACEMENTS + ("C2",):
                    names = ("R", "Str") if p == "C2" else (("T_NUIS", "R", "Str", "C") if p == "P3" else
                                                            ("T_SHUF", "T_STRAT", "T_NUIS", "T_ANTI", "R", "Str", "I",
                                                             "C", "T_LIN"))
                    for k in LEVELS:
                        for name in names:
                            s = self.pool.pooled(cells, reg, p, name, k)
                            if s["n"] or s["lost_unstable"] or s["lost_unbounded"]:  # a fully lost contrast is listed
                                out.append(dict(block=block, regime=reg, placement=p, contrast=name, set=st, level=q,
                                                strength=k, **{x: s[x] for x in ("n", "mean", "se", "t", "p", "p_pos",
                                                                                 "cells", "lost_unstable",
                                                                                 "lost_unbounded", "flag")},
                                                material=self.pool.material(s)))
        return out

    def run(self):
        g1 = self.g1()
        verdicts = self.verdicts()
        info = self.information()
        carry = self.carry(verdicts, g1)
        tables = self.tables()
        lost = self.lost_rows()
        lost["flagged_contrasts"] = sum(t["flag"] for t in tables)
        return dict(materiality=self.mat, alpha=self.alpha,
                    primary_level={f"{p}/{r}": v for (p, r), v in self.primary.items()},
                    G0=self.g0(), G1=g1, G2=self.g2(), G2b=self.g2b(), DR1=verdicts, DR2=info, DR3=carry,
                    DR4=self.poor_data(verdicts, carry), DR5=self.step_order(info), DR6=self.hypothesis(),
                    DR6b=self.mixed_pathway(), DR7=self.hook(), meaningful=self.meaningful(), lost=lost,
                    tables=tables)


def markdown(result):
    lines = ["# Step 4 scorecard", "", f"Materiality {result['materiality']} of the gap; Holm alpha {result['alpha']}.", "",
             "## Gates", "",
             f"- G0 (derived checks): {'pass' if result['G0']['passed'] else 'FAIL'} ({result['G0']['checks']} checks; "
             f"{len(result['G0']['failed'])} failed, {len(result['G0']['eps_misses'])} eps misses reported, "
             f"{result['G0']['procedural_misses']} procedural misses)"]
    for p, g in result["G1"]["placements"].items():
        verdict = ("not testable (no data)" if not g.get("testable", True) else 'pass' if g['passed'] else 'FAIL')
        lines.append(f"- G1 {p}: {g['contrast']} material{'' if p == 'P3' else ' and Holm-significant'} in "
                     f"{g['count']} of 8: {verdict}")
    lines.append(f"- G2 (block C): {'pass' if result['G2']['passed'] else 'FAIL (stop and redesign)'}")
    lines.append(f"- G2b (cone, kappa {result['G2b']['flag_kappa']}): {'pass' if result['G2b']['passed'] else 'fail'}")
    lines += ["", "## DR1 verdicts (X-CS M, FP)", "", "| placement | " + " | ".join(QUALITIES) + " |",
              "|---|" + "---|" * len(QUALITIES)]
    for p, row in result["DR1"].items():
        lines.append(f"| {p} | " + " | ".join(row[q]["verdict"] + (" (!)" if row[q].get("flagged") else "")
                                              for q in QUALITIES) + " |")
    if any(row[q].get("flagged") for row in result["DR1"].values() for q in QUALITIES):
        lines.append("\n(!) a contrast the verdict reads lost an arm to an unstable closed loop (J = -inf) in some "
                     "cell-replicate; see DR1[...]['flagged'] and the lost-rows tally.")
    lines += ["", "## DR2 information diagnosis", ""]
    for p, row in result["DR2"].items():
        lines.append(f"- {p}: " + "; ".join(f"{q} {row[q]['reading']}" for q in QUALITIES))
    why = lambda v: f" (missing: {', '.join(v['missing'])})" if v.get("missing") and v["status"] == "not testable" else ""
    lines += ["", "## DR3-DR7", "",
              "- DR3 carry to 4B-1: " + ", ".join(f"{p} {v['status']}" + why(v)
                                                   + (' (after step 5)' if v.get('conditional_on_step5') else '')
                                                   for p, v in result["DR3"].items() if p in PLACEMENTS),
              f"- DR4 poor data: V1 {result['DR4']['V1']}, preferred {result['DR4']['preferred']}"
              + (", none for poor-quality data" if result["DR4"]["none_for_poor"] else ""),
              f"- DR5: information class wrong {result['DR5']['information_class_wrong']} / {result['DR5']['total']}, "
              f"signal too weak {result['DR5']['signal_too_weak']} / {result['DR5']['total']}",
              "- DR6: " + ", ".join(f"{p} {v['status']}" + why(v) for p, v in result["DR6"].items() if p in PLACEMENTS),
              f"- DR6b mixed pathway: {result['DR6b']['status']}" + why(result["DR6b"]),
              f"- DR7 hook: A500 {'alpha retune' if result['DR7']['A500']['alpha_retune'] else 'not an alpha retune'} "
              f"(immaterial shares {result['DR7']['A500']['immaterial_share']}"
              + (f"; cell-replicates lost to J = -inf {result['DR7']['A500']['lost_unstable']}"
                 if result["DR7"]["A500"].get("flag") else "") + ")"]
    lost = result.get("lost")
    if lost:
        lines += ["", "## Unbounded and unstable rows (design section 4)", "",
                  f"- matched / reference entries reached with J = -inf (unstable): {lost['unstable']}; with J NaN "
                  f"(unbounded): {lost['unbounded']}; unreached with an unbounded fixed point on the arm's ladder: "
                  f"{lost['unreached_unbounded_ladder']} ({len(lost['groups'])} groups listed in the .json)",
                  f"- pooled contrasts flagged for a cell-replicate lost to J = -inf: {lost['flagged_contrasts']}", "",
                  "| block/regime | rows | unbounded (J NaN) | unstable (J = -inf) |", "|---|---|---|---|"]
        lines += [f"| {k} | {t['rows']} | {t['unbounded']} | {t['unstable']} |" for k, t in sorted(lost["totals"].items())]
    return "\n".join(lines) + "\n"


def jsonable(x):
    if isinstance(x, dict):
        return {(k if isinstance(k, str) else "/".join(map(str, k)) if isinstance(k, tuple) else str(k)): jsonable(v)
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root")
    parser.add_argument("--stage0", required=True)
    parser.add_argument("--materiality", type=float, default=MATERIALITY, help="material: |pooled| >= this (x gap)")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--replicates", nargs="+", type=int, default=list(ANALYSIS_REPLICATES))
    parser.add_argument("--output", default=None, help="path stem for .json and .md (default ROOT/scorecard_step4)")
    ns = parser.parse_args(argv)
    try:
        cells, checks, stage0, procs = load(ns.root, ns.stage0, tuple(ns.replicates))
    except SV.Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        sys.exit(2)
    result = Scorecard(cells, checks, ns.materiality, ns.alpha).run()
    result["meta"] = dict(scored_utc=datetime.now(timezone.utc).isoformat(), root=ns.root, stage0=ns.stage0,
                          processes=procs, argv=sys.argv, replicates=ns.replicates)
    stem = Path(ns.output) if ns.output else Path(ns.root) / "scorecard_step4"
    Path(str(stem) + ".json").write_text(json.dumps(jsonable(result), indent=1, allow_nan=False), encoding="utf8")
    Path(str(stem) + ".md").write_text(markdown(result), encoding="utf8")
    print(markdown(result))


if __name__ == "__main__":
    main()
