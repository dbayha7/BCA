"""Read-only acceptance of saved counters and declared evaluation banks."""

import math
import numpy as np


def counters(tree, prefix=""):
    result = {}
    if isinstance(tree, dict):
        for name, value in tree.items():
            path = prefix + "/" + str(name)
            if name in ("step", "count") and np.asarray(value).shape == ():
                result[path] = int(value)
            elif isinstance(value, dict):
                result.update(counters(value, path))
    return result


def checkpoint_counts(tree, host, step, policy_freq=2):
    if int(tree["step"]) != step:
        raise ValueError("Checkpoint host step mismatch.")
    native = tree["state"]["native"]
    expected = (
        {"actor": step, "critic1": step, "critic2": step}
        if host == "cql"
        else {
            "actor": (
                step // policy_freq
                if host == "td3"
                else (step + policy_freq - 1) // policy_freq
            ),
            "critic": step,
        }
    )
    result = {}
    for name, count in expected.items():
        found = counters(native[name])
        if not any((k.endswith("/count") for k in found)) or any(
            (v != count for v in found.values())
        ):
            raise ValueError("Checkpoint optimizer counter mismatch: " + name)
        result[name] = found
    cal = tree["state"]["calibrator"]
    if cal is not None:
        accepted = int(cal["step"])
        found = counters(cal)
        if (
            not 0 <= accepted <= step
            or not any((k.endswith("/count") for k in found))
            or any((v != accepted for v in found.values()))
        ):
            raise ValueError("Checkpoint accepted scale-fit counter mismatch.")
        result["accepted_scale_fits"] = accepted
    else:
        result["accepted_scale_fits"] = None
    return result


def verify_events(protocol, records, evaluations, *, host=None):
    # CQL preserves its native episode names; choose the schema explicitly.
    # The default retains the original TD3/ReBRAC API for existing readers.
    if host not in (None, "td3", "rebrac", "cql"):
        raise ValueError("Unsupported evaluation host: " + str(host))
    fields = ("score", "return") if host == "cql" else ("normalized_score", "raw_return")
    refresh = (
        list(protocol.refresh_steps)
        if hasattr(protocol, "refresh_steps")
        else [e.step for e in protocol.refresh_events]
    )
    boundaries = sorted(
        set(
            range(
                protocol.scan_block_size, protocol.num_updates, protocol.scan_block_size
            )
        )
        | set(refresh)
        | {e.step for e in protocol.evaluation_events}
        | {protocol.num_updates}
    )
    if [r["step"] for r in records if r["kind"] == "accepted_scan"] != [
        s for s in boundaries if s > 0
    ]:
        raise ValueError("Accepted scan boundaries differ from the declaration.")
    if [r["step"] for r in records if r["kind"] == "refresh"] != refresh:
        raise ValueError("Posterior refresh schedule differs from the declaration.")
    if len(evaluations) != len(protocol.evaluation_events):
        raise ValueError("Evaluation bank count mismatch.")
    for actual, wanted in zip(evaluations, protocol.evaluation_events):
        if (actual["kind"], actual["step"], actual["episode_seeds"]) != (
            wanted.kind,
            wanted.step,
            list(wanted.episode_seeds),
        ):
            raise ValueError("Evaluation bank identity mismatch.")
        if (
            actual["episode_count"] != len(wanted.episode_seeds)
            or len(actual["episodes"]) != actual["episode_count"]
        ):
            raise ValueError("Incomplete episode bank.")
        for episode in actual["episodes"]:
            if any(k not in episode for k in fields):
                raise ValueError("Missing evaluation fields for host: " + str(host))
            if not all(
                (isinstance(episode[k], (int, float)) and not isinstance(episode[k], bool)
                 and math.isfinite(episode[k]) for k in fields)
            ):
                raise ValueError("Nonfinite evaluation result.")
