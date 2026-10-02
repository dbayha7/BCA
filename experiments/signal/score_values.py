"""Step 4: J for every stored K, only after the runner's hashes are recorded (design section 5, "Blindness").

Reads the process directories run_step4.py writes (OUTPUT/<block>/<level>/rep<r>: matched_knobs.json, arrays.npz and
HASHES.sha256). It refuses to run, and computes nothing, when any directory has no HASHES.sha256, when the file does not
list both matched_knobs.json and arrays.npz, or when either file's SHA-256 differs from the recorded one (the K's and
knobs changed after they were hashed). It also refuses to overwrite existing values unless --overwrite.

For each directory:
- J of every array under 'K/' (every kept K of every regime, K_0, the 200-point uniform frontier), row by row, with
  lq_harness.value on the system of the run's settings (lq_harness.default_system(gamma, episode_length)); a
  nonfinite K (an unbounded fixed point) gives NaN, an unstable closed loop -inf. Block C's J is the mean over contexts
  of J(K_c) (expectations Part 3 item 1). Stored as 'J/...' with the same keys and row order, so the scorecard reads any
  arm's J through the runner's index. Nothing here knows the design beyond K arrays.
- Per cell: J_opt = J(K*), J(K_0) and the gap g = J_opt - J(K_0) that normalises every contrast; grad J(K_0) by
  lq_harness.policy_gradient (block C: grad_{K_c} of mean_c J(K_c) = grad J(K_c) / C).
- The Level-1 quantities that need J (design section 5): for every row with stored term gradients,
  placement_harness.level1 with grad J(K_0) and the A500 NONE run's mean Adam second moment gives cos(-Delta g, grad J),
  the first-order efficiency and the Adam-preconditioned cosine ('L1J/<cell>/<regime>', columns as L1J_FIELDS).
- The reported quantities that need J (design section 5): for FP rows T_front = G(arm) - G(P0(m*)) at the runner's
  nearest frontier point and regret_alpha = max_m G(P0(m)) - G(arm) over the 200-point frontier ('F/<cell>/FP'); for
  every matched arm with decomposition rows, G_reg = G(K_N + a delta-hat_P0), G_perp = G - G_reg and the reversed order
  (values.json, per cell, regime, arm and level). G is in units of the gap.
- The [derived] items only J can check: A500 NONE through the weighted path against the None path, |dJ| <= 1e-5
  ([derived up to float32]; Part 2 item 1); block R continuity (Part 1 items 1-2): for replicates 0-4 under step 3's
  settings, J_change of step 3's anchors (none, bca, constant, shuffled, oracle; J(K) - J(K_0), the rows run through
  lq_harness.actor_loop) and the threshold, dose mean / SD, Q- and BC-term gradient norms and alignment_K equal step
  3's runs/wbcp_signal/lq/results.json (signal q1) bit for bit. These items are never skipped silently: scoring a
  block-R directory of replicates 0-4 whose settings are step 3's (lq_harness.Settings' defaults, step 3's run) without
  that file is refused, and a cell with no step-3 record gets a failing check.
Writes values.npz, values.json and VALUES.sha256 (sha256sum format: the two outputs; values.json also records the input
hashes it verified).

The pilot (replicate 99, meta.pilot; design section 9) is refused unless --pilot-power. In that mode only pilot
directories are scored, and J is computed only for the power check's rows: K_0, NONE, EXACT, the P0 ladder and the
matched OR1 / OR2 rows of the weighting placements, in block C's localized cells and X-CS's cone cells (PILOT_CELLS);
nothing else is evaluated. It writes pilot_power.npz ('J/<cell>/K0', 'J/<cell>/<regime>' with the row numbers
'R/<cell>/<regime>'), pilot_power.json (G of each row in units of the gap, with its label) and PILOT_POWER.sha256, never
values.*. Directories found under a root that are out of the mode (pilot ones in a normal run, the others with
--pilot-power) are skipped and logged; a directory named explicitly out of its mode is refused.

JAX_PLATFORMS=cpu python experiments/signal/score_values.py DIR [DIR ...] [--step3 runs/wbcp_signal/lq/results.json]
    [--pilot-power] [--overwrite]
    DIR: a process directory, or a root searched for */matched_knobs.json
Tests: JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_score_values
"""

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.signal import lq_harness as LQ  # noqa: E402
from experiments.signal import placement_harness as PH  # noqa: E402

HASH_FILE, VALUES_HASH_FILE = "HASHES.sha256", "VALUES.sha256"
REQUIRED = ("matched_knobs.json", "arrays.npz")
STEP3 = ROOT / "runs" / "wbcp_signal" / "lq" / "results.json"
STEP3_SETTINGS = ("seed", "gamma", "episode_length", "train_episodes", "cal_episodes", "cal_rows_per_episode",
                  "eval_episodes", "behavior_noise", "reward_noise", "fit_steps", "actor_steps", "batch_size", "blend",
                  "draws", "independent_ratio")
STEP3_REPLICATES = range(5)
STEP3_ARMS = ("none", "bca", "constant", "shuffled", "oracle")
STEP3_SIGNAL_FIELDS = (("threshold", ("signal", "threshold")), ("dose_mean", ("signal", "dose_mean")),
                       ("dose_sd", ("signal", "dose_sd")), ("q_term_grad_norm", ("diagnostics", "q_term_grad_norm")),
                       ("alignment_K", ("diagnostics", "alignment_K_step3")))
STEP3_DEFAULTS = {k: v for k, v in asdict(LQ.Settings()).items() if k in STEP3_SETTINGS}  # step 3's run settings
L1J_FIELDS = ("cos_dg_gradJ", "efficiency", "cos_dg_adam_gradJ")
NONE_PATH_TOL = 1e-5
PILOT_REPLICATE = 99
PILOT_HASH_FILE, PILOT_OUTPUTS = "PILOT_POWER.sha256", ("pilot_power.json", "pilot_power.npz")
PILOT_CELLS = {"C": ("q1opt-local", "tilt-local"), "X-CS": ("cone",)}  # block C localized and cone cells
# SHUF is added at the freeze (decision H): the power statistic C = G(OR2) - G(SHUF) needs it, and a shuffled null
# carries no signal, so its pilot J reveals nothing about SIG (the pilot replicate is excluded from every analysis)
PILOT_PLACEMENTS, PILOT_ARMS = ("P1L", "P2L", "P1pi", "P2pi"), ("OR1", "OR2", "SHUF")


class Refused(Exception):
    """A directory whose K's were not hashed, or changed after hashing: no J is computed."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_hashes(path):
    """{file name: hex} from a sha256sum-format file."""
    out = {}
    for line in Path(path).read_text(encoding="utf8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.lstrip("*").strip()] = digest
    return out


def verify(directory):
    """The recorded hashes of a process directory, after checking both files against them (else Refused)."""
    directory = Path(directory)
    path = directory / HASH_FILE
    if not path.is_file():
        raise Refused(f"{directory}: no {HASH_FILE}; the K's were never hashed, so J is not computed")
    recorded = read_hashes(path)
    for name in REQUIRED:
        if name not in recorded:
            raise Refused(f"{directory}: {HASH_FILE} does not list {name}")
        actual = sha256_file(directory / name)
        if actual != recorded[name]:
            raise Refused(f"{directory}: {name} has sha256 {actual}, recorded {recorded[name]}; changed after hashing")
    return recorded


def process_dirs(paths):
    """Process directories under the given paths (a directory with matched_knobs.json, or a root above them)."""
    out = []
    for p in map(Path, paths):
        if (p / "matched_knobs.json").is_file():
            out.append(p)
        else:
            out += sorted(q.parent for q in p.rglob("matched_knobs.json"))
    return out


def read_meta(directory):
    return json.loads((Path(directory) / "matched_knobs.json").read_text(encoding="utf8"))["meta"]


def is_pilot(meta):
    return bool(meta.get("pilot")) or meta.get("replicate") == PILOT_REPLICATE


def needs_step3(meta, settings=None):
    """Whether a directory carries block R's continuity items (Part 1 items 1-2): block R, replicates 0-4, and the
    step-3 settings on every STEP3_SETTINGS field (the step-3 file's, or step 3's run settings when not given)."""
    ref = STEP3_DEFAULTS if settings is None else settings
    return (meta.get("block") == "R" and meta.get("replicate") in STEP3_REPLICATES
            and all(meta.get("settings", {}).get(k) == ref.get(k) for k in STEP3_SETTINGS))


def J_rows(system, K):
    """J of each gain in K (..., da, ds) or, with contexts, (..., C, da, ds) as mean_c J(K_c); NaN for nonfinite K."""
    K = np.asarray(K, np.float64)
    da, ds = system.dims[1], system.dims[0]
    if K.shape[-2:] != (da, ds):
        raise ValueError(f"not a gain array: {K.shape}")
    flat = K.reshape(-1, da, ds)
    J = np.array([LQ.value(system, k) if np.all(np.isfinite(k)) else math.nan for k in flat])
    return J.reshape(K.shape[:-2])


def J_of(system, K, contexts):
    J = J_rows(system, K)
    return J.mean(axis=-1) if contexts else J


def grad_J(system, K0, contexts):
    if not contexts:
        return LQ.policy_gradient(system, np.asarray(K0, np.float64))
    K0 = np.asarray(K0, np.float64)
    return np.stack([LQ.policy_gradient(system, K0[c]) / len(K0) for c in range(len(K0))])


def step3_records(path):
    """Step 3's q1 records by (behaviour, case, kappa, pi_offset)."""
    data = json.loads(Path(path).read_text(encoding="utf8"))
    out = {(r["behavior"], r["case"], r["kappa"], r["pi_offset"]): r["metrics"] for r in data["results"]
           if r["signal"] == "q1"}
    return out, data["meta"]["settings"]


def continuity(meta, cells, J, step3):
    """Block R's bitwise continuity with step 3 (Part 1 items 1-2) for replicates 0-4 under step 3's settings; a cell
    without a step-3 record fails (every block-R cell is one of step 3's 72 q1 cells)."""
    records, settings = step3
    if meta["block"] != "R" or meta["replicate"] not in STEP3_REPLICATES:
        return []
    if any(meta["settings"].get(k) != settings.get(k) for k in STEP3_SETTINGS):
        return [dict(name="step3_continuity", item="1.1", kind="derived", ok=True,
                     value="not applicable: settings differ from step 3's", tol=None)]
    rep, checks = meta["replicate"], []
    for cid, rec in cells.items():
        key = (meta["level"], rec["case"], rec["kappa"], rec["offset"])
        if key not in records:
            checks.append(dict(name=f"step3_continuity:{cid}", item="1.1-1.2", kind="derived", ok=False,
                               value=[f"no step-3 record for (behaviour, case, kappa, offset) = {list(key)}"],
                               tol=None))
            continue
        m = records[key]
        A500 = rec["regimes"]["A500"]["index"]
        J0 = J[f"J/{cid}/K0"]
        misses = []
        for arm in STEP3_ARMS:
            row = A500["anchors"].get(arm)
            if row is None:
                misses.append(f"{arm}: anchor missing")
                continue
            got = float(J[f"J/{cid}/A500"][row]) - float(J0)
            want = m[f"J_change_{arm}"]["values"][rep]
            if not (got == want or (want is None and not math.isfinite(got))):
                misses.append(f"J_change_{arm}: {got!r} != {want!r}")
        for name, (section, field) in STEP3_SIGNAL_FIELDS:
            got, want = rec[section].get(field), m[name]["values"][rep]
            if got != want:
                misses.append(f"{name}: {got!r} != {want!r}")
        for arm in STEP3_ARMS:
            got, want = rec["diagnostics"].get(f"bc_term_grad_norm_{arm}"), m[f"bc_term_grad_norm_{arm}"]["values"][rep]
            if got != want:
                misses.append(f"bc_term_grad_norm_{arm}: {got!r} != {want!r}")
        checks.append(dict(name=f"step3_continuity:{cid}", item="1.1-1.2", kind="derived", ok=not misses,
                           value=misses or None, tol=None))
    return checks


def frontier(out, arrays, cid, rec, gap):
    """FP rows' T_front = G(arm) - G(P0(m*)) (m* the runner's nearest frontier point) and regret_alpha = max_m
    G(P0(m)) - G(arm) over the 200-point frontier, in units of the gap ('F/<cell>/FP', columns T_front, regret)."""
    key = f"D/{cid}/FP/frontier"
    if key not in arrays.files:
        return
    J, Jf = out[f"J/{cid}/FP"], out[f"J/{cid}/FP_frontier"]
    Jn = J[rec["regimes"]["FP"]["index"]["NONE"]]
    G, Gf = (J - Jn) / gap, (Jf - Jn) / gap
    idx = arrays[key][:, 0].astype(int)
    T_front = np.array([G[i] - Gf[j] if j >= 0 else math.nan for i, j in enumerate(idx)])
    finite = Gf[np.isfinite(Gf)]
    regret = (finite.max() - G) if finite.size else np.full(len(G), math.nan)
    out[f"F/{cid}/FP"] = np.stack([T_front, regret], axis=1)


def decomposition(out, cid, rec, gap):
    """G = G_reg + G_perp with G_reg = G(K_N + a delta-hat_P0) (design section 5), and the reversed order G_perp' =
    G(K_N + delta_perp), G_reg' = G - G_perp' (the nonlinearity check), per regime, matched arm and level."""
    res = {}
    for reg, r in rec["regimes"].items():
        idx, J = r["index"], out[f"J/{cid}/{reg}"]
        Jn = J[idx["NONE"]]
        G = lambda row: float((J[row] - Jn) / gap)
        for arm, levels in idx.get("decomp", {}).items():
            for level, parts in levels.items():
                g = G(idx["matched"][arm][level]["row"])
                g_reg, g_perp_first = G(parts["reg"]), G(parts["perp"])
                res.setdefault(reg, {}).setdefault(arm, {})[level] = dict(
                    G=g, G_reg=g_reg, G_perp=g - g_reg, G_perp_first=g_perp_first, G_reg_second=g - g_perp_first)
    return res


def pilot_rows(regime):
    """{row: label} of one regime's rows the pilot's power check may evaluate (design section 9): NONE, EXACT, the P0
    ladder and the matched rows of the weighting placements' OR1 / OR2 arms."""
    idx, labels = regime["index"], regime["labels"]
    rows = {idx[k]: k for k in ("NONE", "EXACT") if k in idx}
    rows.update({r: f"P0:{labels[r][1]:g}" for r in idx["ladder"].get("P0", [])})
    for arm, levels in idx["matched"].items():
        placement, _, kind = arm.partition(":")
        if placement in PILOT_PLACEMENTS and (kind in PILOT_ARMS or re.fullmatch(r"SHUF\d+", kind)):
            rows.update({m["row"]: f"{arm}@{level}" for level, m in levels.items() if m.get("row") is not None})
    return rows


def score_pilot(directory, recorded, payload, overwrite=False, log=print):
    """The pilot's power check (--pilot-power): J only of K_0 and pilot_rows in PILOT_CELLS; nothing else."""
    directory = Path(directory)
    if (directory / PILOT_OUTPUTS[1]).exists() and not overwrite:
        raise FileExistsError(f"{directory}: {PILOT_OUTPUTS[1]} exists (use --overwrite)")
    meta, cells = payload["meta"], payload["cells"]
    arrays = np.load(directory / "arrays.npz")
    system = LQ.default_system(meta["settings"]["gamma"], meta["settings"]["episode_length"])
    J_opt = LQ.value(system, LQ.lqr_gain(system))
    contexts = meta["block"] == "C"
    out, summary = {}, {}
    for cid, rec in cells.items():
        if rec["case"] not in PILOT_CELLS.get(meta["block"], ()):
            continue
        J0 = float(J_of(system, arrays[f"K/{cid}/K0"], contexts))
        gap = J_opt - J0
        out[f"J/{cid}/K0"] = np.asarray(J0)
        regimes = {}
        for reg, r in rec["regimes"].items():
            rows = pilot_rows(r)
            order = np.array(sorted(rows), np.int64)
            J = J_of(system, arrays[f"K/{cid}/{reg}"][order], contexts)
            out[f"J/{cid}/{reg}"], out[f"R/{cid}/{reg}"] = J, order
            Jn = float(J[list(order).index(r["index"]["NONE"])])
            regimes[reg] = {rows[int(i)]: dict(row=int(i), J=float(j), G=(float(j) - Jn) / gap)
                            for i, j in zip(order, J)}
        summary[cid] = dict(case=rec["case"], kappa=rec.get("kappa"), J0=J0, J_opt=J_opt, gap=gap, regimes=regimes)
    np.savez_compressed(directory / PILOT_OUTPUTS[1], **out)
    result = dict(meta=dict(scored_utc=datetime.now(timezone.utc).isoformat(), inputs=recorded, argv=sys.argv,
                            mode="pilot-power", block=meta["block"], level=meta["level"], replicate=meta["replicate"],
                            whitelist=dict(cells=PILOT_CELLS, rows="K0, NONE, EXACT, P0 ladder, matched OR1 / OR2 of "
                                           + ", ".join(PILOT_PLACEMENTS)), J_opt=J_opt),
                  cells=summary)
    (directory / PILOT_OUTPUTS[0]).write_text(json.dumps(_jsonable(result), indent=1, allow_nan=False),
                                              encoding="utf8")
    lines = [f"{sha256_file(directory / n)}  {n}" for n in PILOT_OUTPUTS]
    (directory / PILOT_HASH_FILE).write_text("\n".join(lines) + "\n", encoding="utf8")
    log(f"{directory}: pilot power, J of {sum(np.size(v) for k, v in out.items() if k.startswith('J/'))} gains in "
        f"{len(summary)} cells")
    return result


def score_dir(directory, step3=None, overwrite=False, log=print, pilot_power=False):
    """J of every stored K of one verified process directory; writes values.npz, values.json, VALUES.sha256. A pilot
    directory is refused unless pilot_power (then score_pilot); pilot_power on any other directory is refused; a
    directory with block R's continuity items (needs_step3) is refused without step 3's records."""
    directory = Path(directory)
    recorded = verify(directory)
    payload = json.loads((directory / "matched_knobs.json").read_text(encoding="utf8"))
    meta, cells = payload["meta"], payload["cells"]
    if is_pilot(meta) != pilot_power:
        raise Refused(f"{directory}: the pilot's J is computed only with --pilot-power, and only for the power check's "
                      f"rows (design section 9)" if is_pilot(meta) else
                      f"{directory}: --pilot-power applies only to the pilot replicate {PILOT_REPLICATE}")
    if pilot_power:
        return score_pilot(directory, recorded, payload, overwrite, log)
    if step3 is None and needs_step3(meta):
        raise Refused(f"{directory}: block R replicate {meta['replicate']} under step 3's settings needs step 3's "
                      f"results file for its continuity items (Part 1 items 1-2); none was given")
    if (directory / "values.npz").exists() and not overwrite:
        raise FileExistsError(f"{directory}: values.npz exists (use --overwrite)")
    arrays = np.load(directory / "arrays.npz")
    system = LQ.default_system(meta["settings"]["gamma"], meta["settings"]["episode_length"])
    k_opt = LQ.lqr_gain(system)
    J_opt = LQ.value(system, k_opt)
    contexts = meta["block"] == "C"
    out = {}
    for key in arrays.files:
        if key.startswith("K/"):
            out["J/" + key[2:]] = J_of(system, arrays[key], contexts)
    summary, checks = {}, []
    for cid, rec in cells.items():
        J0 = float(out[f"J/{cid}/K0"])
        gJ = grad_J(system, arrays[f"K/{cid}/K0"], contexts)
        vbar = arrays[f"D/{cid}/vbar_none"] if f"D/{cid}/vbar_none" in arrays.files else None
        regimes = {}
        for reg, r in rec["regimes"].items():
            idx = r["index"]
            J = out[f"J/{cid}/{reg}"]
            regimes[reg] = dict(J_none=float(J[idx["NONE"]]))
            if f"D/{cid}/{reg}/L1" in arrays.files:
                L1 = arrays[f"D/{cid}/{reg}/L1"]
                none = L1[idx["NONE"]]
                t_none = dict(g_Q=none[0], g_pen=none[1], g_BC=none[2], g=none[3])
                L1J = np.full((len(L1), len(L1J_FIELDS)), np.nan)
                for i, row in enumerate(L1):
                    if not np.all(np.isfinite(row)):
                        continue
                    m = PH.level1(dict(g_Q=row[0], g_pen=row[1], g_BC=row[2], g=row[3]), t_none, grad_J=gJ,
                                  v_bar=vbar)
                    L1J[i] = [np.nan if m.get(k) is None else float(m[k]) for k in L1J_FIELDS]
                out[f"L1J/{cid}/{reg}"] = L1J
            if reg == "A500" and "NONE_path" in idx:
                gap = abs(float(J[idx["NONE_path"]]) - float(J[idx["NONE"]]))
                checks.append(dict(name=f"none_weighted_vs_none_path:{cid}", item="2.1", kind="eps",
                                   ok=gap <= NONE_PATH_TOL, value=gap, tol=NONE_PATH_TOL))
        out[f"G/{cid}/gradJ_K0"] = gJ
        gap = J_opt - J0
        frontier(out, arrays, cid, rec, gap)
        summary[cid] = dict(J0=J0, J_opt=J_opt, gap=gap, regimes=regimes,
                            decomposition=decomposition(out, cid, rec, gap))
    if step3 is not None:
        checks += continuity(meta, cells, out, step3)
    np.savez_compressed(directory / "values.npz", **out)
    values = dict(meta=dict(scored_utc=datetime.now(timezone.utc).isoformat(), inputs=recorded, argv=sys.argv,
                            block=meta["block"], level=meta["level"], replicate=meta["replicate"],
                            L1J_FIELDS=L1J_FIELDS, J_opt=J_opt),
                  cells=summary, checks=checks)
    (directory / "values.json").write_text(json.dumps(_jsonable(values), indent=1, allow_nan=False), encoding="utf8")
    lines = [f"{sha256_file(directory / n)}  {n}" for n in ("values.json", "values.npz")]
    (directory / VALUES_HASH_FILE).write_text("\n".join(lines) + "\n", encoding="utf8")
    bad = [c["name"] for c in checks if not c["ok"]]
    log(f"{directory}: J of {sum(np.size(v) for k, v in out.items() if k.startswith('J/'))} gains"
        + (f"; failed checks {bad}" if bad else ""))
    return values


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def run(paths, step3_path=STEP3, overwrite=False, log=print, pilot_power=False):
    """Select the directories of the mode (pilot ones only with pilot_power; others found under a root are skipped and
    logged, a directory named explicitly out of its mode is refused), verify every selected one and require step 3's
    file where block R's continuity applies; if anything is refused, nothing is computed for any directory."""
    dirs = process_dirs(paths)
    if not dirs:
        raise Refused(f"no matched_knobs.json under {paths}")
    explicit = {Path(p).resolve() for p in paths if (Path(p) / "matched_knobs.json").is_file()}
    chosen, metas = [], {}
    for d in dirs:
        meta = read_meta(d)
        if is_pilot(meta) != pilot_power:
            if d.resolve() in explicit:
                raise Refused(f"{d}: " + ("the pilot is scored only with --pilot-power" if is_pilot(meta)
                                          else f"--pilot-power applies only to the pilot replicate {PILOT_REPLICATE}"))
            log(f"skipped {d}: " + ("pilot directory (J only with --pilot-power)" if is_pilot(meta)
                                    else "not the pilot (--pilot-power)"))
            continue
        chosen.append(d)
        metas[d] = meta
    if not chosen:
        raise Refused(f"no {'pilot ' if pilot_power else ''}directory to score under {paths}")
    for d in chosen:
        verify(d)
    step3 = step3_records(step3_path) if step3_path and Path(step3_path).is_file() else None
    if step3 is None and not pilot_power:
        missing = [str(d) for d in chosen if needs_step3(metas[d])]
        if missing:
            raise Refused(f"step 3's results file {step3_path} is missing, and block R's continuity items (Part 1 items "
                          f"1-2) apply to {missing}")
    return {str(d): score_dir(d, step3, overwrite, log, pilot_power) for d in chosen}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--step3", default=str(STEP3), help="step 3's results.json for block R's continuity checks")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pilot-power", action="store_true",
                        help="score only the pilot (replicate 99), and only the power check's rows (design section 9)")
    ns = parser.parse_args(argv)
    try:
        run(ns.paths, ns.step3, ns.overwrite, pilot_power=ns.pilot_power)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
