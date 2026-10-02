"""Step 4, Stage 0: the outcome-free quantities that fix the evidence sets before any J (design section 8).

For every cell of every block and every replicate, with no sigma fit, no actor run and no J:
- alignment_K at K_0, the harness's definition: the cosine of Q1's and Q^pi's K-gradients mean_i grad_a Q(s_i, pi_0(s_i))
  (-s_i)^T on the training rows (placement_harness.alignment_K, float64 autodiff). Block R also records it by step 3's
  own NumPy expressions (run_step4.step3_alignment), the value expectations Part 1 item 2 compares with step 3. Block C:
  per-context gains, so the K-gradient is (C, da, ds); localized cases read it on context-0 rows (design section 7,
  "for tilt, G_true and alignment are computed on context-0 rows"), clean on all rows;
- d_crit = |K_fp(EXACT) - K_fp(NONE)|_Sigma_ev, the fixed points of the exact and the case's critic at BC weight 1 and
  the cell's locked lambda_0 (placement_harness.solve_fp), from K's only. EXACT's matrix is positive definite (H_aa < 0);
  an unbounded NONE (minimum eigenvalue <= 0) gives d_crit = +inf (recorded as unbounded) and counts as >= 0.01;
- cos(BC pull, Q^pi0 ascent) = cos((K_bc - K_0) Sigma_tr, +grad_K mean Q^pi(s, -K s) at K_0) on the training rows
  (step4_settings' convention: the ascent is the gradient itself, not synth_pull.py's descent);
- the cone's cos(D_q Sigma, D_q Sigma_c), D_q = K_0 - K-bar_b, Sigma_c = mean c(s) s s^T, and corr(c(s), |s|^2)
  (expectations Part 0 item 3), on the training rows;
- lambda_0, the fixed points' minimum eigenvalues and S_host,FP = |K_fp(NONE) - K_0|_Sigma (outcome-free).
Cells, data and critics are run_step4's (grid, block_data, make_cell), so Stage 0 reads exactly what the runner will.

Evidence sets (per block, cell and level; means over the analysis replicates 0-9, the pilot 99 excluded):
- N: clean, noisy_reward (block C's clean too);
- P: the cone (X) and block C's localized cells, whatever their alignment;
- M: Q1 error present, mean alignment_K <= 0.5 and mean d_crit >= 0.01;
- W: 0.5 < mean alignment_K < 0.9 and mean d_crit >= 0.01;
- H: Q1 error present, not M or W;
- M_nat: M cells of natural cases (q1_optimistic, shared_bias, independent_errors) in blocks R, X-CS and X-EP.
q2_optimistic (block R replicate 0) is a bitwise check, in no set.

Built-in [derived] checks: tilt cells reach alignment_K = a* within 1e-4 in every replicate and level (Part 0 item 1),
X-CS's tilt matrix T is bitwise identical across the four levels within a replicate (Part 0 item 1), X-CS's states,
action noise and next-state shocks are bitwise identical across levels (Part 2 item 7). Also reported: the largest gap
between block R's two alignment computations.

This module neither imports nor calls lq_harness.value(), lq_harness.policy_gradient() or any other J function
(test_step4_stage0 scans the source and runs Stage 0 with both replaced by functions that raise). It writes
stage0.json and stage0.json.sha256 (sha256sum format).

JAX_PLATFORMS=cpu python experiments/signal/step4_stage0.py --output runs/wbcp_signal/step4/stage0.json
    [--blocks R X-CS X-EP C] [--replicates 0 1 ... 9] [--pilot] [--cone-kappas 1 2] [--blockc-kappa 1]
    [--train-episodes ... and every other placement_harness.Settings field]
Tests: JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_step4_stage0
"""

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import jax  # noqa: E402

from experiments.signal import lq_harness as LQ  # noqa: E402
from experiments.signal import placement_harness as PH  # noqa: E402
from experiments.signal import run_step4 as RUN  # noqa: E402

NEGATIVE = ("clean", "noisy_reward")
M_ALIGNMENT, W_ALIGNMENT, D_CRIT = 0.5, 0.9, 0.01
TILT_TOL = 1e-4


def k_gradient(grad_a, s, context=None, n_contexts=None):
    """mean_i grad_a Q(s_i, pi(s_i)) (-s_i)^T; per-context gains: one block per context, normalised by all rows."""
    if context is None:
        return PH.k_gradient(grad_a, s)
    return np.stack([grad_a[context == c].T @ (-s[context == c]) / len(s) for c in range(n_contexts)])


def stage0_cell(h, block, level, rep, spec, data):
    """Stage 0's quantities for one cell and replicate (no sigma, no actor run, no J)."""
    cell = RUN.make_cell(h, block, level, rep, spec, data)
    params, K0 = cell.params, np.asarray(cell.K0, np.float64)
    tr, ev = data["train"], data["eval"]
    obs, s, a = tr["obs"], tr["state"], tr["action"]
    contexts = block == "C"
    C = h.cfg.contexts if contexts else None
    context = tr["context"] if contexts else None
    crit = cell.models[1]
    pi0 = PH.policy_np(K0, obs)
    grads = PH.critic_grads(crit, params, obs, pi0)
    out = dict(block=block, cell=spec.id, case=spec.case, kappa=spec.kappa, a_star=spec.a_star, offset=spec.offset,
               level=level, replicate=rep)
    if contexts:
        g1, ge = k_gradient(grads["q1"], s, context, C), k_gradient(grads["exact"], s, context, C)
        out["alignment_K_all"] = LQ.cosine(g1, ge)
        rows = context == 0
        out["alignment_K_ctx0"] = PH.alignment_K({k: v[rows] for k, v in grads.items()}, s[rows])
        out["alignment_K"] = out["alignment_K_ctx0"] if spec.case in RUN.LOCALIZED else out["alignment_K_all"]
    else:
        out["alignment_K"] = PH.alignment_K(grads, s)
        if block == "R":
            out["alignment_K_step3"] = RUN.step3_alignment(params, s, K0, data["meta"]["K_bar"])
    lam0 = PH.lambda0(params, obs, K0, h.args.alpha)
    T1 = PH.fp_terms(*PH.fp_rows(params, obs, crit, "q1"), s, a)
    Te = PH.fp_terms(*PH.fp_rows(params, obs, crit, "exact"), s, a)
    solve = ((lambda T: PH.solve_fp_context(T, context, 1.0, 1.0, lam0, C)) if contexts
             else (lambda T: PH.solve_fp(T, 1.0, 1.0, lam0)))
    none, exact = solve(T1), solve(Te)
    sigma_ev = PH.second_moment(ev["state"], ev["context"] if contexts else None, C)
    sigma_tr = PH.second_moment(s, context, C)
    bounded = bool(np.all(none["finite"]) and np.all(np.isfinite(none["K"])))
    out.update(lam0=lam0, fp_min_eig_none=float(np.min(none["min_eig"])),
               fp_min_eig_exact=float(np.min(exact["min_eig"])), fp_none_bounded=bounded,
               d_crit=float(PH.strength(exact["K"], none["K"], sigma_ev)) if bounded else math.inf,
               S_host_fp=float(PH.strength(none["K"], K0, sigma_ev)) if bounded else None)
    K_bc = data["meta"]["K_bc_context"] if contexts else data["meta"]["K_bc"]
    ascent = k_gradient(grads["exact"], s, context, C)
    pull = np.einsum("...ij,...jk->...ik", K_bc - K0, sigma_tr)
    out["cos_pull_ascent"] = LQ.cosine(pull, ascent)
    if spec.case == "cone":
        cone = params["cone"]
        w = PH.cone_weight(s, cone["v"], cone["c0"], cone["sharpness"])
        D = K0 - data["meta"]["K_bar"]
        S_w = (s * w[:, None]).T @ s / len(s)
        out.update(cone_c0=float(cone["c0"]), cone_mean=float(w.mean()), cone_cos=LQ.cosine(D @ sigma_tr, D @ S_w),
                   cone_corr_s2=LQ.correlation(w, np.sum(s ** 2, 1)))
    if "T" in cell.info:
        out["T_digest"] = LQ.digest(cell.info["T"])
    for key in ("independent_scale", "independent_ratio", "independent_redraws", "c"):
        if key in cell.info:
            out["tilt_c" if key == "c" else key] = cell.info[key]
    return out


def evidence_set(block, case, mean_alignment, mean_dcrit):
    """The design's set of a cell (section 8) from its replicate means."""
    if case == "q2_optimistic":
        return "check"
    if case in NEGATIVE:
        return "N"
    if case == "cone" or case in RUN.LOCALIZED:
        return "P"
    # rounded to 9 decimals so constructed boundary cells (tilt a* = 0.5, 0.9) take the set their construction
    # implies rather than one decided by floating-point rounding (pre-registration, addendum A, part 1 correction)
    mean_alignment = None if mean_alignment is None else round(mean_alignment, 9)
    if mean_alignment is not None and mean_dcrit is not None and mean_dcrit >= D_CRIT:
        if mean_alignment <= M_ALIGNMENT:
            return "M"
        if M_ALIGNMENT < mean_alignment < W_ALIGNMENT:
            return "W"
    return "H"


def evidence_sets(rows, replicates=RUN.ANALYSIS_REPLICATES):
    """Per (block, cell, level): mean alignment_K and d_crit over the analysis replicates, the set and M_nat."""
    groups = {}
    for r in rows:
        if r["replicate"] in replicates:
            groups.setdefault((r["block"], r["cell"], r["level"]), []).append(r)
    out = {}
    for (block, cell, level), rs in sorted(groups.items()):
        align = [r["alignment_K"] for r in rs if r["alignment_K"] is not None]
        dcrit = [r["d_crit"] for r in rs]
        mean_a = float(np.mean(align)) if align else None
        mean_d = float(np.mean(dcrit)) if dcrit else None  # +inf when any replicate's NONE is unbounded
        case = rs[0]["case"]
        label = evidence_set(block, case, mean_a, mean_d)
        out.setdefault(block, {}).setdefault(cell, {})[level] = dict(
            set=label, M_nat=bool(label == "M" and case in RUN.NATURAL and block in ("R", "X-CS", "X-EP")),
            case=case, mean_alignment_K=mean_a, mean_d_crit=mean_d, unbounded=sum(not math.isfinite(d) for d in dcrit),
            replicates=sorted(r["replicate"] for r in rs),
            mean_cos_pull_ascent=float(np.mean([r["cos_pull_ascent"] for r in rs if r["cos_pull_ascent"] is not None]))
            if any(r["cos_pull_ascent"] is not None for r in rs) else None)
    return out


def checks(rows, digests):
    """Stage 0's [derived] checks (Part 0 item 1, Part 2 item 7)."""
    out = []
    tilt = [r for r in rows if r["case"] in ("tilt", "tilt-local")]
    worst = max((abs(r["alignment_K"] - r["a_star"]) for r in tilt), default=0.0)
    out.append(dict(name="tilt_alignment", item="0.1", kind="derived", ok=worst <= TILT_TOL, value=worst, tol=TILT_TOL))
    by = {}
    for r in rows:
        if r["block"] == "X-CS" and "T_digest" in r:
            by.setdefault((r["replicate"], r["cell"]), set()).add(r["T_digest"])
    out.append(dict(name="tilt_T_identical_across_levels_xcs", item="0.1", kind="derived",
                    ok=all(len(v) == 1 for v in by.values()), value=sum(len(v) > 1 for v in by.values()), tol=0))
    same = {}
    for (block, level, rep), d in digests.items():
        if block == "X-CS":
            same.setdefault(rep, set()).add(d)
    out.append(dict(name="xcs_states_noise_shocks_identical_across_levels", item="2.7", kind="derived",
                    ok=all(len(v) == 1 for v in same.values()), value=sum(len(v) > 1 for v in same.values()), tol=0))
    gaps = [abs(r["alignment_K"] - r["alignment_K_step3"]) for r in rows
            if r.get("alignment_K_step3") is not None and r["alignment_K"] is not None]
    out.append(dict(name="block_r_alignment_two_computations", item="1.2", kind="eps",
                    ok=max(gaps, default=0.0) <= 1e-9, value=max(gaps, default=0.0), tol=1e-9))
    return out


def run(output, blocks=RUN.BLOCKS, replicates=RUN.ANALYSIS_REPLICATES, overrides=None, cone_kappas=RUN.CONE_KAPPAS,
        blockc_kappa=RUN.C_KAPPA, log=print):
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).isoformat()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists; Stage 0 is written once")
    rows, digests, settings = [], {}, {}
    for block in blocks:
        cfg = RUN.settings_for(block, overrides)
        settings[block] = asdict(cfg)
        h = PH.PlacementHarness(cfg)
        for rep in replicates:
            for level in RUN.LEVELS[block]:
                t0 = time.perf_counter()
                data = RUN.block_data(h, block, level, rep)
                keys = ("state", "noise", "shock") if block == "X-CS" else ("state",)
                digests[(block, level, rep)] = LQ.digest(*(data[sp][k] for sp in ("train", "cal", "eval") for k in keys))
                cells = RUN.grid(block, rep, cone_kappas, blockc_kappa)
                rows += [stage0_cell(h, block, level, rep, spec, data) for spec in cells]
                log(f"stage0 {block} {level} rep{rep}: {len(cells)} cells in {time.perf_counter() - t0:.1f}s")
        jax.clear_caches()  # between blocks, as the step-3 run (vm.max_map_count); values do not change
        gc.collect()
    sets = evidence_sets(rows, [r for r in replicates if r in RUN.ANALYSIS_REPLICATES])
    payload = dict(meta=dict(started_utc=stamp, finished_utc=datetime.now(timezone.utc).isoformat(),
                             seconds=time.perf_counter() - started, blocks=list(blocks), replicates=list(replicates),
                             analysis_replicates=list(RUN.ANALYSIS_REPLICATES), pilot=RUN.PILOT_REPLICATE,
                             settings=settings, cone_kappas=list(cone_kappas), blockc_kappa=blockc_kappa,
                             thresholds=dict(M_alignment=M_ALIGNMENT, W_alignment=W_ALIGNMENT, d_crit=D_CRIT),
                             argv=sys.argv, jax=jax.__version__, numpy=np.__version__,
                             data_digests={f"{b}/{lv}/rep{r}": d for (b, lv, r), d in digests.items()}),
                   checks=checks(rows, digests), sets=sets,
                   rows=[dict(r, d_crit=None if not math.isfinite(r["d_crit"]) else r["d_crit"]) for r in rows])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(RUN.jsonable(payload), indent=1, allow_nan=False), encoding="utf8")
    sha = RUN.sha256_file(output)
    Path(str(output) + ".sha256").write_text(f"{sha}  {output.name}\n", encoding="utf8")
    log(f"stage0: {len(rows)} cell-replicates in {time.perf_counter() - started:.0f}s -> {output} (sha256 {sha})")
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", required=True, help="stage0.json (new); its .sha256 is written beside it")
    parser.add_argument("--blocks", nargs="+", choices=RUN.BLOCKS, default=list(RUN.BLOCKS))
    parser.add_argument("--replicates", nargs="+", type=int, default=list(RUN.ANALYSIS_REPLICATES))
    parser.add_argument("--pilot", action="store_true", help="also compute the pilot replicate 99 (in no set)")
    parser.add_argument("--cone-kappas", nargs="+", type=float, default=list(RUN.CONE_KAPPAS))
    parser.add_argument("--blockc-kappa", type=float, default=RUN.C_KAPPA)
    for name, default in asdict(PH.Settings()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=None)
    return parser


def main(argv=None):
    ns = build_parser().parse_args(argv)
    reps = list(ns.replicates) + ([RUN.PILOT_REPLICATE] if ns.pilot else [])
    overrides = {name: getattr(ns, name) for name in asdict(PH.Settings())}
    run(ns.output, ns.blocks, reps, overrides, tuple(ns.cone_kappas), ns.blockc_kappa)


if __name__ == "__main__":
    main()
