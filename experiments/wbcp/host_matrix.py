"""The host x dataset matrix of semi-synthetic calibration experiments, and its results loader.

Every host (TD3+BC, ReBRAC, CQL, IQL) gets the same experiments on all seven datasets, each at
that host's own configured bank (configs/<host>.yaml: reservation size n and rows per episode K).
A dataset's score pool is the host's own: its critic gives the residual, its BCA scale network
sigma(s, a) the normalization, its actor the policy action. Stages, in order:

  freeze   one frozen score pool per dataset (a short CPU training run, freeze_<host>.py)
  predict  dependence.py predictions for every bank design (Stage B of the expectations)
  bench    d4rl_benchmark.py runs, 4,000 simulated banks per design (Stage C)

Expectations come first: each stage refuses to start unless <root>/expectations.md exists, and
bench also needs a "Stage B" section in it, so the predictions are on record before any result.

Designs per dataset (no shift unless stated): independent rows; K = 2, 5 and 10 spaced; whole
episodes; the configured bank with BCA's own sampler (resv K) and, when K is not already listed,
with the spaced sampler (strat K, the other side of the finite-pool bracket); independent rows at
the large bank (8,192 rows; 1,194 on pen-human) when the configured bank is smaller, for the
estimated-weight effect; and K = 5 spaced under the six tilts.

python experiments/wbcp/host_matrix.py --host rebrac --stage freeze [--device gpu] [--datasets hopper ...] [--dry-run]
python experiments/wbcp/host_matrix.py --host rebrac --stage status
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import subprocess
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HOSTS = {"td3_bc": "TD3+BC", "rebrac": "ReBRAC", "cql": "CQL", "iql": "IQL"}
FREEZE = {"td3_bc": "freeze_scores.py", "rebrac": "freeze_rebrac.py", "cql": "freeze_cql.py", "iql": "freeze_iql.py"}
DATASETS = ("hopper", "walker2d", "halfcheetah", "maze2d", "pen-cloned", "pen-expert", "pen-human")
SEED, UPDATES, TRIALS, WORKERS = 202609171, 100_000, 4000, 16
BIG = {"pen-human": 1194}  # every other dataset: 8,192 rows (the IQL bank)
NO_SHIFT = ("--tilt", "policy", "--gamma", "0")
SHIFT = ("--tilt", "policy", "density", "state", "--gamma", "0.5", "1", "--seed", "2026093002")
TILTS = (("policy", 0.5), ("policy", 1.0), ("density", 0.5), ("density", 1.0), ("state", 0.5), ("state", 1.0))
PYTHON = "/opt/bca-venv/bin/python"


def run_root(host):
    # TD3+BC's matrix extends the seven-dataset run of DEPENDENCE.md (Change 12) in place
    return "runs/wbcp_dependence/all-datasets" if host == "td3_bc" else f"runs/wbcp_hosts/{host}"


def prefix(dataset):
    return dataset.replace("-", "")


def frozen_name(host, dataset):
    if host == "td3_bc":  # the pools made before this matrix existed keep their names
        return ("hopper-medium-v2" if dataset == "hopper" else dataset) + f"-s{SEED}-u{UPDATES}"
    return f"{host}-{dataset}-s{SEED}-u{UPDATES}"


def frozen_dir(host, dataset):
    return f"runs/wbcp_frozen/{frozen_name(host, dataset)}"


def config(host):
    with open(os.path.join(ROOT, "configs", f"{host}.yaml"), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def bank(host, dataset):
    """(n, K, big): the configured reservation size and rows per episode, and the large bank size."""
    reservation = config(host)["datasets"][dataset]["reservation"]
    return int(reservation["size"]), int(reservation["rows_per_episode"]), BIG.get(dataset, 8192)


def designs(host, dataset):
    """[(run name, design label, d4rl_benchmark.py arguments)] for one host and dataset."""
    n, k, big = bank(host, dataset)
    p = prefix(dataset)
    spaced = lambda kk: ("--per-episode", str(kk), "--spacing", "stratified")  # noqa: E731
    rows = [(f"{p}_iid", "Independent rows", NO_SHIFT + ("--n", str(n))),
            *[(f"{p}_strat{kk}", f"K={kk} spaced", NO_SHIFT + ("--n", str(n)) + spaced(kk)) for kk in (2, 5, 10)],
            (f"{p}_blocks", "Whole episodes", NO_SHIFT + ("--n", str(n), "--blocks")),
            (f"{p}_resv{k}", f"K={k} BCA sampler (configured)",
             NO_SHIFT + ("--n", str(n), "--per-episode", str(k), "--spacing", "reservation"))]
    if k not in (2, 5, 10):
        rows.append((f"{p}_strat{k}", f"K={k} spaced (configured K)", NO_SHIFT + ("--n", str(n)) + spaced(k)))
    if n < big:
        rows.append((f"{p}_iid_big", f"Independent rows, n={big}", NO_SHIFT + ("--n", str(big))))
    rows.append((f"{p}_strat5_shift", "K=5 spaced, six tilts", SHIFT + ("--n", str(n)) + spaced(5)))
    return rows


def prediction_name(host, dataset):
    return f"predictions_{prefix(dataset)}" + ("_small" if host == "td3_bc" else "")


def prediction_args(host, dataset):
    n, k, _ = bank(host, dataset)
    ks = sorted({2, 5, 10, k})
    return ("--n", str(n), "--block-sizes", str(n), "--per-episode", *map(str, ks),
            "--stratified", *map(str, ks), "--reservation", str(k))


def commands(host, stage, datasets):
    """[(output path, log path, argv)] for a stage; argv runs from the repo root."""
    out, root = [], run_root(host)
    for dataset in datasets:
        if stage == "freeze":
            target = frozen_dir(host, dataset)
            out.append((target, f"{root}/freeze_{dataset}.log",
                        [PYTHON, f"experiments/wbcp/{FREEZE[host]}", "--dataset", dataset, "--seed", str(SEED),
                         "--updates", str(UPDATES), "--output", target]))
        elif stage == "predict":
            target = f"{root}/{prediction_name(host, dataset)}"
            out.append((target, target + ".log", [PYTHON, "experiments/wbcp/dependence.py", "--frozen",
                                                  frozen_dir(host, dataset), "--output", target,
                                                  *prediction_args(host, dataset)]))
        else:
            for name, _, args in designs(host, dataset):
                target = f"{root}/{name}.json"
                out.append((target, f"{root}/{name}.txt",
                            [PYTHON, "experiments/wbcp/d4rl_benchmark.py", "--frozen", frozen_dir(host, dataset),
                             "--score", "both", "--workers", str(WORKERS), "--trials", str(TRIALS), *args,
                             "--output", target]))
    return out


def check_expectations(host, stage):
    path = os.path.join(ROOT, run_root(host), "expectations.md")
    if not os.path.exists(path):
        return f"{path} is missing: write the pre-registered expectations before any {stage} run"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if stage == "bench" and "Stage B" not in text:
        return f"{path} has no Stage B section: record the predictions before the benchmark runs"
    return None


def device_probe(env):
    """The JAX backend and devices a child process with this environment gets (JAX_PLATFORMS=cuda
    fails rather than falling back to the CPU, so a GPU run cannot silently train on the CPU)."""
    probe = subprocess.run([PYTHON, "-c", "import jax; print(jax.default_backend()); "
                            "print([d.device_kind for d in jax.devices()])"],
                           cwd=ROOT, env=env, capture_output=True, text=True)
    lines = probe.stdout.strip().splitlines()
    return dict(ok=probe.returncode == 0, backend=lines[-2] if len(lines) >= 2 else None,
                devices=lines[-1] if lines else None, error=None if probe.returncode == 0 else probe.stderr[-2000:])


def run(host, stage, datasets, dry_run=False, device="cpu"):
    problem = check_expectations(host, stage)
    if problem and not dry_run:
        raise SystemExit(problem)
    # pools train on the requested device; predictions and benchmarks are NumPy and always use the CPU
    env = dict(os.environ, JAX_PLATFORMS="cuda" if (stage == "freeze" and device == "gpu") else "cpu",
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    probe = device_probe(env) if stage == "freeze" and not dry_run else None
    if probe is not None and not probe["ok"]:
        raise SystemExit(f"JAX could not start on {device}: {probe['error']}")
    for target, log, argv in commands(host, stage, datasets):
        if os.path.exists(os.path.join(ROOT, target)):
            print(f"{target} exists", flush=True)
            continue
        print(" ".join(argv) + f" > {log} 2>&1", flush=True)
        if dry_run:
            continue
        os.makedirs(os.path.dirname(os.path.join(ROOT, log)), exist_ok=True)
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(os.path.join(ROOT, log), "w", encoding="utf-8") as handle:
            code = subprocess.run(argv, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        if probe is not None and os.path.isdir(os.path.join(ROOT, target)):
            record = dict(host=host, dataset=target, argv=argv, jax_platforms=env["JAX_PLATFORMS"], device=device,
                          jax_backend=probe["backend"], jax_devices=probe["devices"], started_utc=started,
                          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), exit_code=code)
            with open(os.path.join(ROOT, target, "run_record.json"), "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=1)
        print(f"{target} {'done' if code == 0 else f'FAILED ({code})'}", flush=True)


# ---- results loader (used by host_notebook.py) ----

def _load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as handle:
        return json.load(handle)


def _sha(path):
    with open(os.path.join(ROOT, path), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _arm(a):
    return dict(f=a["failures"], T=a["trials"], lo=a["ci"][0], hi=a["ci"][1], r=a.get("risk"), th=a.get("threshold"),
                ab=a.get("abstain"), q95=a.get("risk_q95"), q99=a.get("risk_q99"), x=a.get("excess_given_fail"))


def _blocks(data):
    return [dict(t=b["tilt"], y=b["gamma"], n=b["n"], s=b["score"], L=b["lambda_star"], ne=b["calibration_n_eff"]["oracle"],
                 nh=b["calibration_n_eff"]["estimated"], cs=b["calibration_size"]["mean"],
                 A={k: _arm(v) for k, v in b["arms"].items()}) for b in data["blocks"]]


def _predictions(data):
    out = dict(created=data["created_utc"], pool=data["pool"], dataset=data["dataset"], scores={})
    for name, entry in data["scores"].items():
        out["scores"][name] = dict(
            lambda_star=entry["lambda_star"], rho=entry["icc"], rho_score=entry["icc_score"],
            lags={str(k): v for k, v in entry["lag_correlation"].items()},
            designs=[dict(design=d["design"], n=d["n"], size=d["mean_bank_size"], k=d["rows_per_episode"],
                          Dicc=d["design_effect_icc"], Dmc=d["design_effect_mc"], picc=d["predicted_failure_icc"],
                          pmc=d["predicted_failure_mc"], eps=d["episodes_reserved"], share=d["reserved_share_of_dataset"])
                     for d in entry["designs"]])
    return out


# min and max per-step reward of each cached D4RL file (h5py, 2026-10-01); with discount gamma, any value
# function of a policy lies in [min, max] / (1 - gamma) after the host's own reward scaling
RAW_REWARD_RANGE = {"halfcheetah": (-3.0135, 13.8546), "hopper": (0.5490, 5.9441), "maze2d": (0.0, 1.0),
                    "pen-cloned": (-6.2256, 60.9805), "pen-expert": (-6.2340, 60.9947),
                    "pen-human": (-0.8562, 60.9646), "walker2d": (-4.7131, 8.5531)}
DISCOUNT = 0.99  # every host config


def q_range(meta, dataset):
    """[lowest, highest] possible Q on the host's reward scale: CQL records its multiplier and affine map,
    IQL its return range (rewards / (hi - lo) * 1000); otherwise rewards are used raw."""
    lo, hi = RAW_REWARD_RANGE[dataset]
    rn = meta.get("reward_normalization") or {}
    if rn.get("normalization_multiplier") is not None:
        scale, bias = rn["normalization_multiplier"] * rn.get("reward_scale", 1.0), rn.get("reward_bias", 0.0)
    elif rn.get("normalize_reward") and rn.get("return_range"):
        scale, bias = 1000.0 / (rn["return_range"][1] - rn["return_range"][0]), 0.0
    else:
        scale, bias = 1.0, 0.0
    return (lo * scale + bias) / (1 - DISCOUNT), (hi * scale + bias) / (1 - DISCOUNT)


def critic_health(pool_dir, dataset=None):
    """Did the frozen critic converge? Q's growth over the last fifth of training and the final critic loss
    over its minimum, from training_log.json; diverged above 100% growth or a 1e6 loss ratio, still growing
    above 20%; and the pool's mean Q against the range any value function can take (q_range, 5% slack). Post-hoc thresholds (DEPENDENCE.md, Change 12): healthy pools grew 1-7%, TD3+BC pen-cloned 46%
    and pen-human 114% (its critic loss rose 3.6e17-fold). The share of scores at most 1 says how well the
    scale sigma(s, a) fits (its target is 0.9); it does not affect validity and is reported, not judged."""
    log_path = os.path.join(ROOT, pool_dir, "training_log.json")
    if not os.path.exists(log_path):
        return None
    with open(log_path, encoding="utf-8") as handle:
        log = json.load(handle)
    blocks = log.get("blocks") if isinstance(log, dict) else log

    def series(*keys):
        return next(([b[k] for b in blocks] for k in keys if blocks and k in blocks[0]), None)

    q, loss = series("q_mean", "average_qf1"), series("critic_loss", "qf_loss", "q_loss")
    cut = len(blocks) - max(1, len(blocks) // 5)
    growth = None if q is None else (q[-1] - q[cut]) / max(abs(q[cut]), 1e-12)
    # CQL's qf_loss includes its conservative penalty and can be negative; a ratio is meaningful only for a positive loss
    ratio = None if loss is None or min(loss) <= 0 else loss[-1] / min(loss)
    with open(os.path.join(ROOT, pool_dir, "frozen.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    q_pool = meta.get("diagnostics", {}).get("q_mean")
    bounds = q_range(meta, dataset) if dataset in RAW_REWARD_RANGE else None
    slack = 0.05 * (bounds[1] - bounds[0]) if bounds else 0.0
    outside = bounds is not None and q_pool is not None and not bounds[0] - slack <= q_pool <= bounds[1] + slack
    if (growth is not None and growth > 1.0) or (ratio is not None and ratio > 1e6):
        status = "diverged"
    elif outside:
        status = "Q outside its possible range"
    elif growth is not None and growth > 0.2:
        status = "still growing"
    else:
        status = "ok"
    return dict(status=status, q_final=None if q is None else q[-1], q_pool_mean=q_pool, q_bounds=bounds,
                q_growth_last_fifth=growth, loss_ratio=ratio)


def load_results(host):
    """Everything a host notebook shows: per dataset the bank, the pool, the predictions and every run."""
    root = run_root(host)
    result = dict(host=host, name=HOSTS[host], root=root, datasets={}, expectations=[])
    for dataset in DATASETS:
        n, k, big = bank(host, dataset)
        entry = dict(n=n, k=k, big=big, pool=None, pred=None, runs={}, extra={}, missing=[])
        meta_path = f"{frozen_dir(host, dataset)}/frozen.json"
        if os.path.exists(os.path.join(ROOT, meta_path)):
            meta = _load(meta_path)
            record_path = f"{frozen_dir(host, dataset)}/run_record.json"
            record = _load(record_path) if os.path.exists(os.path.join(ROOT, record_path)) else {}
            entry["pool"] = dict(dir=frozen_dir(host, dataset), rows=meta["rows"], updates=meta.get("updates"),
                                 dataset=meta.get("dataset"), npz_sha256=meta.get("npz_sha256"), algorithm=meta.get("algorithm"),
                                 device=record.get("jax_backend") or "cpu (before run records)",
                                 health=critic_health(frozen_dir(host, dataset), dataset),
                                 frac_score_le_1=meta.get("diagnostics", {}).get("fraction_score_at_most_one"))
        pred_path = f"{root}/{prediction_name(host, dataset)}/predictions.json"
        if os.path.exists(os.path.join(ROOT, pred_path)):
            entry["pred"] = _predictions(_load(pred_path))
        names = set()
        for name, label, _ in designs(host, dataset):
            names.add(name)
            path = f"{root}/{name}.json"
            if os.path.exists(os.path.join(ROOT, path)):
                data = _load(path)
                entry["runs"][name] = dict(label=label, file=path, settings=data["settings"], blocks=_blocks(data))
            else:
                entry["missing"].append(name)
        for path in sorted(glob.glob(os.path.join(ROOT, root, prefix(dataset) + "_*.json"))):
            name = os.path.basename(path)[:-5]
            if name not in names:  # runs outside this host's matrix, e.g. another host's bank on this pool
                data = _load(os.path.relpath(path, ROOT))
                if "blocks" in data:
                    entry["extra"][name] = dict(file=os.path.relpath(path, ROOT).replace("\\", "/"), settings=data["settings"],
                                                blocks=_blocks(data))
        result["datasets"][dataset] = entry
    expectations = f"{root}/expectations.md"
    if os.path.exists(os.path.join(ROOT, expectations)):
        with open(os.path.join(ROOT, expectations), encoding="utf-8") as handle:
            text = handle.read()
        result["expectations"].append(dict(file=expectations, sha=_sha(expectations), text=text))
    return result


def prediction_table(host):
    """Markdown table of a host's Stage B predictions (uniform BCA, no shift, normalized with raw in brackets)."""
    data = load_results(host)
    lines = ["| Dataset (n, K) | rho | K=2 spaced | K=5 spaced | K=10 spaced | Configured K, spaced | "
             "Configured K, BCA sampler | Whole episodes |", "|---|---|---|---|---|---|---|---|"]
    for dataset in DATASETS:
        e = data["datasets"][dataset]
        if e["pred"] is None:
            lines.append(f"| {dataset} (n={e['n']}, K={e['k']}) | not predicted yet |" + " |" * 6)
            continue

        def cell(label):
            vals = []
            for score in ("normalized", "raw"):
                hit = [d for d in e["pred"]["scores"][score]["designs"] if d["design"] == label and d["n"] == e["n"]]
                vals.append(None if not hit or hit[0]["pmc"] is None else 100 * hit[0]["pmc"])
            return "–" if vals[0] is None else f"{vals[0]:.1f}% ({vals[1]:.1f}%)"

        k, sc = e["k"], e["pred"]["scores"]
        lines.append(f"| {dataset} (n={e['n']}, K={k}) | {sc['normalized']['rho']:.3f} ({sc['raw']['rho']:.3f}) | "
                     + " | ".join(cell(x) for x in ("2 per episode, stratified", "5 per episode, stratified",
                                                    "10 per episode, stratified", f"{k} per episode, stratified",
                                                    f"{k} per episode, BCA reservation", "whole episodes")) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", choices=HOSTS, required=True)
    parser.add_argument("--stage", choices=("freeze", "predict", "bench", "status", "table"), required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu", help="where the freeze stage trains")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args(argv)
    if args.stage == "table":
        print(prediction_table(args.host))
        return
    if args.stage == "status":
        for stage in ("freeze", "predict", "bench"):
            rows = commands(args.host, stage, args.datasets)
            done = [t for t, _, _ in rows if os.path.exists(os.path.join(ROOT, t))]
            print(f"{stage}: {len(done)}/{len(rows)} done")
            for t, _, _ in rows:
                if t not in done:
                    print(f"  missing {t}")
        return
    run(args.host, args.stage, args.datasets, args.dry_run, args.device)


if __name__ == "__main__":
    sys.exit(main())
