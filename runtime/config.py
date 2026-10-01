"""Resolve the two methods from shared host settings and dataset overrides."""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("host", "bca")
ALGORITHMS = ("iql", "cql", "td3_bc", "rebrac")


def merge(base, patch):
    result = deepcopy(base)
    for k, v in patch.items():
        result[k] = (
            merge(result[k], v)
            if isinstance(v, dict) and isinstance(result.get(k), dict)
            else deepcopy(v)
        )
    return result


@lru_cache(None)
def _yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf8"))


def read_config(path):
    result = _yaml(str(Path(path).resolve()))
    if result.get("algorithm") not in ALGORITHMS or set(result) - {
        "algorithm",
        "host",
        "bca",
        "reservation",
        "datasets",
    }:
        raise ValueError(
            "Expected one algorithm config with host, bca and dataset settings."
        )
    return result


def schedule(spec):
    steps = list(range(spec["first_step"], spec["last_step"] + 1, spec["interval"]))
    if spec["kind"] == "steps":
        return steps
    if spec["kind"] == "refresh":
        return [
            dict(step=s, seed=spec["first_seed"] + i * spec["seed_stride"])
            for i, s in enumerate(steps)
        ]
    if spec["kind"] != "evaluation":
        raise ValueError("Unknown schedule")
    result = [
        dict(
            kind="periodic",
            step=s,
            episode_seeds=list(
                range(
                    spec["first_seed"] + i * spec["bank_seed_stride"],
                    spec["first_seed"]
                    + i * spec["bank_seed_stride"]
                    + spec["periodic_episodes"],
                )
            ),
        )
        for i, s in enumerate(steps)
    ]
    first = spec["first_seed"] + len(steps) * spec["bank_seed_stride"]
    return result + [
        dict(
            kind="final",
            step=steps[-1],
            episode_seeds=list(range(first, first + spec["final_episodes"])),
        )
    ]


def resolve(path, method, seed, output_dir, dataset=None):
    config = read_config(path)
    exp = _yaml(str(ROOT / "configs/experiment.yaml"))
    if (
        method not in METHODS
        or type(seed) is not int
        or seed not in exp["seeds"]
        or dataset not in config["datasets"]
    ):
        raise ValueError("Select a declared dataset, seed and method (host or bca).")
    if "calibration_weighting" in exp:
        raise ValueError("calibration_weighting belongs to the archived stack; WBCP weights are uniform.")
    data = config["datasets"][dataset]
    algorithm = config["algorithm"]
    host = "td3" if algorithm == "td3_bc" else algorithm
    native = merge(config["host"], data.get("host", {}))
    bca = merge(config["bca"], data.get("bca", {}))
    reservation = merge(config["reservation"], data.get("reservation", {}))
    k = reservation.get("rows_per_episode")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        # Whole-episode banks break WBCP's guarantee (experiments/wbcp/DEPENDENCE.md).
        raise ValueError("Declare a positive integer reservation.rows_per_episode for " + str(dataset) + ".")
    identity = "bca-wbcp" if method == "bca" else method
    run_id = f"{algorithm}-{dataset}-{identity}-s{seed}"
    row = dict(
        host=host,
        algorithm=algorithm,
        environment=data["environment"],
        method=method,
        calibration="wbcp_uniform" if method == "bca" else "not_applicable",
        run_id=run_id,
        cache=deepcopy(data["cache"]),
    )
    common = dict(
        seed=seed,
        dataset=data["environment"],
        num_updates=exp["num_updates"],
        eval_interval=exp["eval_interval"],
        eval_final_episodes=exp["final_episodes"],
        eval_workers=2 if host == "iql" else 10,
        log=False,
    )
    native.update(common)
    steps = schedule(exp["schedules"]["refresh_steps"])
    row["expected_counts"] = dict(
        host_updates=exp["num_updates"],
        actor_updates=(
            exp["num_updates"] // 2 if host in ("td3", "rebrac") else exp["num_updates"]
        ),
        periodic_evaluations=exp["num_updates"] // exp["eval_interval"],
        checkpoint_steps=[10000, 50000, exp["num_updates"]],
    )
    if host == "iql":
        row["expected_counts"].pop("actor_updates")
        row["expected_counts"].pop("checkpoint_steps")
        row["expected_counts"].update(
            actor_attempted_updates=exp["num_updates"],
            actor_accepted_updates="measured",
            checkpoint_schedule=(
                "each reference and training end" if method == "bca" else "training end"
            ),
        )
        if bca["decision_gain"] != 1.0:
            raise ValueError("IQL decision gain is fixed at one.")
        native["allow_off_config"] = False
        bca.update(
            reserve_seed=reservation["seed"],
            reserve_max_fraction=reservation["max_fraction"],
            reserve_rows_per_episode=reservation["rows_per_episode"],
        )
        row.update(
            mode="shared" if method == "bca" else "native",
            actor_count=2 if method == "bca" else 1,
            scale_variant_count=1 if method == "bca" else 0,
            spaces=data["spaces"],
        )
        row["options"] = dict(
            dataset=data["environment"],
            seed=seed,
            output_dir=str(Path(output_dir).resolve()),
            mode=row["mode"],
            scope="development",
            updates=exp["num_updates"],
            eval_interval=exp["eval_interval"],
            curve_episodes=2,
            final_episodes=exp["final_episodes"],
            reserve_size=reservation["size"],
            warmup=exp["warmup"],
            refresh_interval=exp["refresh_interval"],
            eval_seed=20260920,
            prepare_only=False,
            host_parameters=native,
            posterior_parameters=bca,
        )
        if method == "bca":
            row["options"]["decision_gain"] = 1.0
        return row
    native.update(algorithm="corl_" + algorithm, allow_off_config=True)
    row["native_args"] = native
    events = schedule(exp["schedules"][exp["evaluation_banks"][seed]])
    protocol = dict(
        run_id=run_id,
        seed=seed,
        num_updates=exp["num_updates"],
        scan_block_size=exp["scan_block_size"],
        eval_interval=exp["eval_interval"],
        eval_workers=10,
        eval_periodic_episodes=10,
        eval_final_episodes=exp["final_episodes"],
        evaluation_events=events,
    )
    if host == "cql":
        row["config"] = dict(arm=method, **bca)
        protocol.update(
            calibration_target_size=reservation["size"],
            calibration_seed=reservation["seed"],
            calibration_max_fraction=reservation["max_fraction"],
            calibration_rows_per_episode=reservation["rows_per_episode"],
            refresh_steps=steps if method == "bca" else [],
        )
    else:
        row["specification"] = dict(
            identity=method,
            configuration_class="declared_deviation",
            deviation_reason="One-million-update host/BCA comparison on the same reserved training pool.",
            posterior=bca if method == "bca" else None,
        )
        protocol.update(
            reservation=dict(
                target_size=reservation["size"],
                seed=reservation["seed"],
                max_fraction=reservation["max_fraction"],
                rows_per_episode=reservation["rows_per_episode"],
                dependency_contract=(
                    "raw_effective_finite_targets_next_actions_v1"
                    if host == "rebrac"
                    else "raw_effective_finite_targets_v1"
                ),
            ),
            refresh_events=(
                schedule(
                    exp["schedules"][data["refresh_banks"][exp["seeds"].index(seed)]]
                )
                if method == "bca"
                else []
            ),
        )
    row.update(
        protocol=protocol, action_bound=1.0, max_episode_steps=data["max_episode_steps"]
    )
    return row


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"nonfinite_literal"}:
            if value["nonfinite_literal"] not in ("inf", "-inf"):
                raise ValueError("Invalid numeric bound.")
            return float(value["nonfinite_literal"])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def typed(row):
    import importlib

    host = row["host"]
    if host == "iql":
        from runtime import iql_pair as F

        driver = F if row["mode"] == "shared" else F.D
        m = F.load_modules()
        options = driver.Options(**row["options"])
        driver.validate_options(options)
        if row["mode"] == "shared":
            F.declaration(options, m)
        else:
            driver.resolved_arguments(options, m)
        return driver, m, options
    runner = importlib.import_module(
        {"td3": "runtime.td3_bc", "rebrac": "runtime.rebrac", "cql": "runtime.cql"}[
            host
        ]
    )
    args = runner.N.Args(**decode(row["native_args"]))
    protocol = deepcopy(row["protocol"])
    protocol["evaluation_events"] = tuple(
        runner.EvaluationEvent(e["kind"], e["step"], tuple(e["episode_seeds"]))
        for e in protocol["evaluation_events"]
    )
    if host == "cql":
        spec = decode(deepcopy(row["config"]))
        spec["posterior"] = runner.P.WBCPConfig(**spec["posterior"])
        spec = runner.P.Config(**spec)
        protocol["refresh_steps"] = tuple(protocol["refresh_steps"])
    else:
        spec = deepcopy(row["specification"])
        if spec["posterior"] is not None:
            spec["posterior"] = runner.PosteriorSpecification(**spec["posterior"])
        spec = runner.RunSpecification(**spec)
        protocol["reservation"] = runner.Reservation(**protocol["reservation"])
        protocol["refresh_events"] = tuple(
            runner.RefreshEvent(**e) for e in protocol["refresh_events"]
        )
    protocol = runner.RunProtocol(**protocol)
    runner.validate_protocol(args, spec, protocol)
    return runner, args, spec, protocol
