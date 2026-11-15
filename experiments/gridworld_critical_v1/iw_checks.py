"""The four checks of BCA_IW_PLAN.md, section 4: is importance weighting correctly applied?
No policy is trained. Needs JAX (CPU is enough).

    python -I -B iw_checks.py --baselines published_validation_2 --seeds 6401 6402 6403

Works on the baselines' saved states after 10,000 updates, where the first reference was frozen.
On this map the exact weights are known: the data hold five commands and their rows can be counted.

Marks, fixed before the first run of this script:
  1  Unit weights give back standard BCA exactly, in every run: the recomputed standard threshold
     equals the saved one, and the table at unit weights and unit mass equals it too.
  2  The estimated weights agree with the exact ones.
     a. Commands no row holds get a larger estimated weight than recorded ones in at least 95% of
        pairings (every recipe).
     b. On `mixed`, where recorded commands differ in how often they are recorded, the rank
        correlation between estimated and exact weight is at least 0.5 (median over runs).
  3  "Cannot certify" lands where it should: at least 95% of commands with no rows cannot be
     certified, and at least 95% of commands with 16 or more rows can (every recipe).
  4  On fresh held-out rows the importance-weighted band covers at least 85% of one-step targets
     and no more than 2 points fewer than the standard band on the same rows (every recipe).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import run_published as P  # noqa: E402  (puts the shared learner and the repository root on the path)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import critical_env as E  # noqa: E402
import learner as L  # noqa: E402
import published_checks as checks  # noqa: E402
import run_baselines as B  # noqa: E402
import run_version_a_published as RA  # noqa: E402
import version_c as VC  # noqa: E402
from calibration.reference import WBCPConfig, freeze_reference, positive_scale  # noqa: E402

MARKS = {"separation": 0.95, "rank_correlation": 0.5, "placement": 0.95, "coverage": 0.85, "coverage_drop": 0.02}
MIN_ROWS = 16


def rank(x):
    order = np.argsort(x, kind="stable")
    out = np.empty(len(x))
    out[order] = np.arange(len(x))
    return out


def spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def share_larger(high, low):
    """Share of (high, low) pairings in which the first is larger (ties count half)."""
    if not len(high) or not len(low):
        return float("nan")
    low = np.sort(low)
    below = np.searchsorted(low, high, side="left")
    ties = np.searchsorted(low, high, side="right") - below
    return float((below + 0.5 * ties).sum() / (len(high) * len(low)))


def absent_breakdown(n, certified, clear=5.0):
    """Commands with no rows, split by how much the absence says: the cell has no rows at all; the
    cell has rows but so few that the command would be expected fewer than `clear` times anyway;
    or it would be expected at least `clear` times and never appears ("clearly absent")."""
    rows = n.sum(axis=1, keepdims=True)
    expected = rows * (n.sum(axis=0) / max(n.sum(), 1))
    absent = n == 0
    groups = {"absent_at_cells_with_no_rows": absent & (rows == 0),
              "absent_by_chance": absent & (rows > 0) & (expected < clear),
              "clearly_absent": absent & (expected >= clear)}
    out = {}
    for name, pick in groups.items():
        out[f"{name}_count"] = int(pick.sum())
        out[f"{name}_flagged"] = float((~certified[pick]).mean()) if pick.any() else None
    return out


def scores_of(eta, residual, unit):
    """Exactly as calibration/reference.freeze_reference scores a bank."""
    scale = np.asarray(positive_scale(jnp.asarray(eta), jnp.asarray(unit)), np.float64)
    return np.abs(np.asarray(residual, np.float64)) / scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--refit", type=int, default=0,
                        help="first train the classifier this many updates against Version C's comparison commands")
    args = parser.parse_args()
    source = HERE / "results" / args.baselines
    start = json.loads((source / "START.json").read_text())
    RA.A.set_alpha(float(start["alpha"]))
    task = E.published()
    base_runs = json.loads((source / f"baselines_{task.name}.json").read_text())["runs"]
    keep = [i for i, run in enumerate(base_runs) if run["data_seed"] in args.seeds]
    template = L.stack([L.init_state(run["train_seed"], 100, B.ACTION_DIM) for run in base_runs])
    index = jnp.asarray(keep)
    states = jax.tree_util.tree_map(lambda x: x[index], RA.load_tree(source / f"snapshot_{task.name}.npz", template))
    runs = [base_runs[i] for i in keep]
    step = int(np.asarray(states["step"])[0])
    cache, banks, audits, tables, datasets = {}, [], [], [], []
    for run in runs:
        pair = (run["recipe"], run["data_seed"])
        if pair not in cache:
            data, _ = P.load_training_data(task, run["recipe"], run["data_seed"], start["data_label"])
            bank = P.make_banks(task, run["recipe"], run["data_seed"])
            counts = np.zeros((100, E.N_MOVES), int)
            np.add.at(counts, (data["s"].astype(int), data["command"].astype(int)), 1)
            cache[pair] = (B.to_device(bank["reference"]), B.to_device(bank["audit"]), counts, B.to_device(data))
        banks.append(cache[pair][0])
        audits.append(cache[pair][1])
        tables.append(cache[pair][2])
        datasets.append(cache[pair][3])
    banks, audits = L.stack(banks), L.stack(audits)
    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(r["train_seed"] + 12345), r["data_seed"]) for r in runs])
    if args.refit:
        fit = VC.make_weight_fit("one_hot", int(datasets[0]["s"].shape[0]))
        states = fit(states, L.stack(datasets), keys, jnp.arange(1, args.refit + 1, dtype=jnp.int32))
    modes = jnp.full((len(runs),), L.NATIVE)
    heldout = L.make_readouts("one_hot")["heldout"]
    refresh_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1_000_000 + step))(keys)
    audit_keys = jax.vmap(lambda k: jax.random.fold_in(k, 2_000_000))(keys)
    eta, residual, _, logit = (np.asarray(x) for x in heldout(states, banks, refresh_keys, modes))
    eta_a, residual_a, _, logit_a = (np.asarray(x) for x in heldout(states, audits, audit_keys, modes))
    units = np.asarray(states["unit"])
    host_scale = jax.tree_util.tree_map(np.asarray, states["scale"])
    saved_threshold = np.asarray(states["reference"]["threshold"])
    free = np.array([int(c) for c in task.free_cells()])
    cells = jnp.asarray(np.repeat(np.eye(100, dtype=np.float32)[free], E.N_MOVES, axis=0))
    corners = jnp.asarray(np.tile(E.CORNERS, (len(free), 1)), jnp.float32)
    per_run = []
    for i, run in enumerate(runs):
        words = np.array([run["train_seed"], run["data_seed"], run["arm_number"], step], np.uint32)
        reference, valid, diagnostics = freeze_reference(L.take(host_scale, i), units[i], eta[i], residual[i], words,
                                                        WBCPConfig())
        scores = scores_of(eta[i], residual[i], units[i])
        unit_table = VC.threshold_table(scores, words, np.ones(len(scores)), [1.0])[1][0]
        fields, record = VC.iw_fields(scores, logit[i], words, float(reference.threshold))
        support = L.take(states["support"], i)
        logm = -np.asarray(L.support_logit(support, cells, corners), np.float64).reshape(len(free), E.N_MOVES)
        n = tables[i][free]
        certified = logm <= float(fields["iw_logm_cert"])
        recorded, absent, dense = n > 0, n == 0, n >= MIN_ROWS
        # The exact log weight of a recorded command, for the comparison set the classifier was trained
        # against: every move equally often (uniformly random commands), or as often as the data
        # hold it overall (commands recorded at other rows, after a refit).
        often = n.sum(axis=0) / n.sum() if args.refit else np.full(E.N_MOVES, 1.0 / E.N_MOVES)
        exact = np.log(np.maximum(often, 1e-300)) + np.log(n.sum(axis=1, keepdims=True) / np.maximum(n, 1))
        # check 4: fresh rows, each with the threshold at its own weight
        scores_a = scores_of(eta_a[i], residual_a[i], units[i])
        logm_a = -np.clip(logit_a[i].astype(np.float64), -VC.LOGIT_CLIP, VC.LOGIT_CLIP)
        own = np.interp(logm_a, fields["iw_logm"], fields["iw_R"])
        own = np.where(logm_a > float(fields["iw_logm_cert"]), np.inf, own)
        per_run.append({
            "recipe": run["recipe"], "data_seed": run["data_seed"], "train_seed": run["train_seed"],
            "standard_threshold_recomputed": float(np.float32(reference.threshold)),
            "standard_threshold_saved": float(saved_threshold[i]),
            "check1_exact": bool(valid and np.float32(reference.threshold) == saved_threshold[i]
                                 and np.float32(unit_table) == saved_threshold[i]),
            "unit_table_equals_standard_on_the_same_scores": bool(np.float32(unit_table) == np.float32(reference.threshold)),
            "saved_threshold_reproduced": bool(np.float32(reference.threshold) == saved_threshold[i]),
            "separation": share_larger(logm[absent], logm[recorded]),
            "decades_between_absent_and_recorded": float((np.median(logm[absent]) - np.median(logm[recorded])) / np.log(10)),
            "rank_correlation_recorded": spearman(logm[recorded], exact[recorded]),
            "absent_that_cannot_be_certified": float((~certified[absent]).mean()),
            "dense_that_can_be_certified": float(certified[dense].mean()),
            "thin_that_can_be_certified": float(certified[recorded & ~dense].mean()) if (recorded & ~dense).any() else None,
            # Reported, no mark: WHICH absent commands are flagged. Absence means little at a cell with
            # few rows, and nothing can be learned at a cell with no rows at all.
            **absent_breakdown(n, certified),
            **{f"{name}_cell": {"fatal_cannot_be_certified": bool(np.all(~certified[list(free).index(spot["cell"])][list(spot["fatal"])])),
                               "way_through_certified": bool(certified[list(free).index(spot["cell"])][spot["through"]])}
               for name, spot in ((k, checks.critical_spot(task, k)) for k in checks.CRITICAL_CELLS)},
            "coverage_weighted_band": float((scores_a <= own).mean()),
            "coverage_standard_band": float((scores_a <= float(reference.threshold)).mean()),
            # Reported, no mark: the band Version C actually consumes is the standard width plus the
            # charge, that is, the larger of the two thresholds at each row.
            "coverage_consumed_band": float((scores_a <= np.maximum(own, float(reference.threshold))).mean()),
            "fresh_rows_that_cannot_be_certified": float(np.isinf(own).mean()),
            "table": record})
    recipes = sorted({r["recipe"] for r in per_run}, key=["optimal", "other_starts", "mixed"].index)
    mean = lambda rows, name: float(np.nanmean([r[name] for r in rows]))  # noqa: E731
    summary, verdicts = {}, {"1": all(r["check1_exact"] for r in per_run), "2a": True, "2b": True, "3": True, "4": True}
    for recipe in recipes:
        rows = [r for r in per_run if r["recipe"] == recipe]
        s = {name: mean(rows, name) for name in ("separation", "decades_between_absent_and_recorded",
                                                 "absent_that_cannot_be_certified", "dense_that_can_be_certified",
                                                 "coverage_weighted_band", "coverage_standard_band",
                                                 "fresh_rows_that_cannot_be_certified")}
        s["rank_correlation_recorded_median"] = float(np.nanmedian([r["rank_correlation_recorded"] for r in rows]))
        s["runs"] = len(rows)
        summary[recipe] = s
        verdicts["2a"] &= s["separation"] >= MARKS["separation"]
        verdicts["3"] &= (s["absent_that_cannot_be_certified"] >= MARKS["placement"]
                          and s["dense_that_can_be_certified"] >= MARKS["placement"])
        verdicts["4"] &= (s["coverage_weighted_band"] >= MARKS["coverage"]
                          and s["coverage_weighted_band"] >= s["coverage_standard_band"] - MARKS["coverage_drop"])
    if "mixed" in summary:
        verdicts["2b"] = bool(summary["mixed"]["rank_correlation_recorded_median"] >= MARKS["rank_correlation"])
    verdicts = {k: bool(v) for k, v in verdicts.items()}
    out = {"baselines": args.baselines, "data_seeds": args.seeds, "step": step, "refit_updates": args.refit,
           "devices": [str(d) for d in jax.devices()], "marks": MARKS, "verdicts": verdicts, "summary": summary,
           "runs": per_run}
    suffix = f"_refit{args.refit}" if args.refit else ""
    (HERE / "results" / f"iw_checks_{args.baselines}{suffix}.json").write_text(json.dumps(out, indent=1))
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    print(f"runs {len(per_run)}; saved states at update {step}; classifier refit for {args.refit} updates; "
          f"devices {out['devices']}")
    print(f"check 1 unit weights give standard BCA exactly: {'PASS' if verdicts['1'] else 'FAIL'} "
          f"({sum(r['check1_exact'] for r in per_run)} of {len(per_run)} runs; unit table equals the standard "
          f"threshold on the same scores in {sum(r['unit_table_equals_standard_on_the_same_scores'] for r in per_run)}; "
          f"saved threshold reproduced in {sum(r['saved_threshold_reproduced'] for r in per_run)})")
    for recipe in recipes:
        s = summary[recipe]
        print(f"[{recipe}] absent > recorded weight in {pct(s['separation'])} of pairings "
              f"({s['decades_between_absent_and_recorded']:.1f} factors of ten apart); rank correlation with exact "
              f"weights (recorded) {s['rank_correlation_recorded_median']:+.2f} | cannot certify: absent "
              f"{pct(s['absent_that_cannot_be_certified'])}, 16+ rows certified {pct(s['dense_that_can_be_certified'])} | "
              f"coverage weighted {pct(s['coverage_weighted_band'])} vs standard {pct(s['coverage_standard_band'])}, "
              f"fresh rows uncertified {pct(s['fresh_rows_that_cannot_be_certified'])}")
    for recipe in recipes:
        rows = [r for r in per_run if r["recipe"] == recipe]
        parts = []
        for name in ("clearly_absent", "absent_by_chance", "absent_at_cells_with_no_rows"):
            flagged = [r[f"{name}_flagged"] for r in rows if r[f"{name}_flagged"] is not None]
            parts.append(f"{name.replace('_', ' ')}: {np.mean([r[f'{name}_count'] for r in rows]):.0f} per run, "
                         f"flagged {pct(np.mean(flagged)) if flagged else 'n/a'}")
        cells = "; ".join(f"{name.replace('_', ' ')} cell: fatal moves cannot be certified in "
                          f"{sum(r[f'{name}_cell']['fatal_cannot_be_certified'] for r in rows)} of {len(rows)} runs, "
                          f"way through certified in {sum(r[f'{name}_cell']['way_through_certified'] for r in rows)}"
                          for name in checks.CRITICAL_CELLS)
        print(f"[{recipe}] " + " | ".join(parts) + " || " + cells
              + f" || coverage of the band Version C consumes {pct(np.mean([r['coverage_consumed_band'] for r in rows]))}")
    print("verdicts:", {k: ("PASS" if v else "FAIL") for k, v in verdicts.items()})


if __name__ == "__main__":
    main()
