"""Resolve one explicit arm and seed, without importing JAX or an environment."""

from copy import deepcopy
from pathlib import Path
from functools import lru_cache
import json
import yaml

ROOT = Path(__file__).resolve().parents[1]


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else deepcopy(value)
        )
    return result


@lru_cache(maxsize=None)
def _yaml(path):
    return yaml.safe_load(Path(path).read_text())


def read_config(path):
    value = _yaml(str(Path(path).resolve()))
    if set(value) - {"bca_defaults"} != {"host", "defaults", "arms", "datasets"}:
        raise ValueError("Expected an algorithm config: host, defaults, arms, datasets.")
    if value["host"] not in ("iql", "cql", "td3", "rebrac"):
        raise ValueError("Unknown host.")
    return value


def schedule(spec):
    """Expand arithmetic schedules explicitly; no randomness or expression evaluation."""
    steps = list(range(spec["first_step"], spec["last_step"] + 1, spec["interval"]))
    if spec["kind"] == "steps":
        return steps
    if spec["kind"] == "refresh":
        return [
            dict(step=step, seed=spec["first_seed"] + i * spec["seed_stride"])
            for i, step in enumerate(steps)
        ]
    if spec["kind"] != "evaluation":
        raise ValueError("Unknown schedule kind.")
    events = [
        dict(
            kind="periodic",
            step=step,
            episode_seeds=list(
                range(
                    spec["first_seed"] + i * spec["bank_seed_stride"],
                    spec["first_seed"]
                    + i * spec["bank_seed_stride"]
                    + spec["periodic_episodes"],
                )
            ),
        )
        for i, step in enumerate(steps)
    ]
    first = spec["first_seed"] + len(steps) * spec["bank_seed_stride"]
    events.append(
        dict(
            kind="final",
            step=steps[-1],
            episode_seeds=list(range(first, first + spec["final_episodes"])),
        )
    )
    return events


def resolve(path, arm, seed, output_dir, dataset=None):
    config = read_config(path)
    experiment = _yaml(str(ROOT / "configs/experiment.yaml"))
    if dataset not in config["datasets"]:
        raise ValueError("Select a declared dataset; use --list.")
    data = config["datasets"][dataset]
    if arm not in data["arms"] or seed not in experiment["seeds"]:
        raise ValueError("Select a declared arm and seed; use --list to see them.")
    host_arm = deepcopy(config["arms"][arm])
    bca = host_arm.pop("bca", False)
    if type(bca) is not bool:
        raise ValueError("bca must be an explicit boolean.")
    patch = merge(config.get("bca_defaults", {}) if bca else {}, host_arm)
    patch = merge(patch, data.get("bca_defaults", {}) if bca else {})
    patch = merge(patch, data["arms"][arm])
    per_seed = patch.pop("seed_overrides", {}).get(seed, {})
    row = merge(merge(merge(config["defaults"], data["defaults"]), patch), per_seed)
    replacements = {
        "{seed}": seed,
        "{output_dir}": str(Path(output_dir).resolve()),
        "{environment}": data["environment"],
        "{evaluation_bank}": experiment["evaluation_banks"].get(seed),
        **{"{experiment." + k + "}": v for k, v in experiment.items() if k != "schedules"},
    }

    def expand(value):
        if isinstance(value, dict):
            if set(value) == {"bank"}:
                return schedule(experiment["schedules"][expand(value["bank"])])
            if set(value) == {"nonfinite_literal"}:
                return value  # Decode only at the typed numerical boundary.
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, str):
            if value in replacements:
                return replacements[value]
            return value.replace("{seed}", str(seed)).replace(
                "{environment}", data["environment"]
            )
        return value

    row = expand(row)
    if row["host"] != config["host"] or row["environment"] != data["environment"]:
        raise ValueError("Host/dataset declaration disagrees with resolved settings.")
    return row


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"nonfinite_literal"}:
            if value["nonfinite_literal"] not in ("inf", "-inf"):
                raise ValueError("Invalid bound.")
            return float(value["nonfinite_literal"])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def typed(row):
    import importlib

    host = row["host"]
    if host == "iql":
        import _iql_iw_driver as F
        from dataclasses import asdict

        m = F.load_modules()
        driver = F if row["mode"] == "shared" else F.D
        options = driver.Options(**row["options"])
        driver.validate_options(options)
        if row["mode"] == "shared":
            declared = F.declaration(options, m)
        else:
            args, arms, differences = F.D.resolved_arguments(options, m)
            declared = F.D.plain(
                {
                    "args": asdict(args),
                    "arms": [asdict(a) for a in arms],
                    "published_differences": differences,
                }
            )
        if any(declared[k] != v for k, v in row["declared"].items()):
            raise ValueError(
                "IQL resolved algorithm settings differ from the explicit configuration."
            )
        return driver, m, options
    runner = importlib.import_module(
        {"td3": "_td3_runner", "rebrac": "_rebrac_runner", "cql": "_cql_runner"}[host]
    )
    args = runner.N.Args(**decode(row["native_args"]))
    protocol = deepcopy(row["protocol"])
    protocol["evaluation_events"] = tuple(
        runner.EvaluationEvent(e["kind"], e["step"], tuple(e["episode_seeds"]))
        for e in protocol["evaluation_events"]
    )
    if host == "cql":
        spec = decode(deepcopy(row["config"]))
        spec["iw"] = runner.P.ScaleIWConfig(**spec["iw"])
        spec["posterior"] = runner.P.PosteriorConfig(**spec["posterior"])
        spec = runner.P.Config(**spec)
        protocol["refresh_steps"] = tuple(protocol["refresh_steps"])
    else:
        spec = deepcopy(row["specification"])
        if spec["posterior"] is not None:
            spec["posterior"]["affinity"] = runner.AffinitySpecification(
                **spec["posterior"]["affinity"]
            )
            spec["posterior"] = runner.PosteriorSpecification(**spec["posterior"])
        spec = runner.RunSpecification(**spec)
        for name, cls in [("reservation", runner.Reservation), ("reference", runner.Reference)]:
            if protocol[name] is not None:
                protocol[name] = cls(**protocol[name])
        protocol["refresh_events"] = tuple(
            runner.RefreshEvent(**e) for e in protocol["refresh_events"]
        )
    protocol = runner.RunProtocol(**protocol)
    runner.validate_protocol(args, spec, protocol)
    return runner, args, spec, protocol
