"""Read-only reconciliation of saved IQL train states and accepted update counters."""

from dataclasses import asdict
import hashlib
import numpy as np


def optimizer_counts(tree, prefix=""):
    found = {}
    if isinstance(tree, dict):
        for name, value in tree.items():
            key = prefix + "/" + name
            if name == "count":
                found[key] = np.asarray(value)
            else:
                found.update(optimizer_counts(value, key))
    return found


def finite_parameters(state):

    def visit(x):
        if isinstance(x, dict):
            for y in x.values():
                visit(y)
        else:
            a = np.asarray(x)
            assert (
                a.dtype.kind in "fiu b".replace(" ", "") and np.isfinite(a).all()
            ), "invalid numeric state"

    visit(state["params"])
    visit(state["opt_state"])


def train_state(state, want):
    expected = np.asarray(want)
    step = np.asarray(state["step"])
    assert step.dtype.kind in "iu" and np.array_equal(
        step, expected
    ), "train-state counter mismatch"
    counters = optimizer_counts(state["opt_state"])
    assert counters, "optimizer counter absent"
    for value in counters.values():
        assert value.dtype.kind in "iu" and np.array_equal(
            value, expected
        ), "optimizer count mismatch"
    finite_parameters(state)
    return {k: v.tolist() for k, v in counters.items()}


def checkpoint_counts(tree, mode, step, actors, calibrators):
    """Decode the actual Flax state dictionary, without rebuilding a model."""
    if mode == "shared":
        assert set(tree) == {"rng", "nuisance", "step", "extras", "actors"}
        rng = tree["rng"]
        actual_step = tree["step"]
        nuisance = tree["nuisance"]
    else:
        assert set(tree) == {"0", "1", "2"}
        rng = tree["0"]
        nuisance = tree["1"]
        actual_step = tree["2"]
    assert np.asarray(actual_step).shape == () and int(actual_step) == step
    rng = np.asarray(rng)
    assert rng.shape == (2,) and rng.dtype == np.uint32
    record = dict(
        step=step,
        rng=rng.tolist(),
        rng_sha256=hashlib.sha256(rng.tobytes()).hexdigest(),
        nuisance={},
    )
    for name in ("qf", "vf"):
        record["nuisance"][name] = train_state(nuisance[name], step)
    target = nuisance["qf_target"]
    assert int(target["step"]) == step
    target_counts = optimizer_counts(target["opt_state"])
    assert target_counts
    assert all((np.asarray(x).shape == () and int(x) == 0 for x in target_counts.values()))
    finite_parameters(target)
    record["target_optimizer_counts"] = {k: v.tolist() for k, v in target_counts.items()}
    if mode == "shared":
        train_state(nuisance["actor"], actors[0])

        def same_representative(a, b):
            if isinstance(a, dict):
                assert set(a) == set(b)
                for k in a:
                    same_representative(a[k], b[k])
            else:
                assert np.array_equal(np.asarray(a), np.asarray(b)[0])

        same_representative(nuisance["actor"], tree["actors"])
        record["actors"] = train_state(tree["actors"], np.asarray(actors))
        assert set(tree["extras"]) == {str(i) for i in range(len(calibrators))}
        record["calibrators"] = [
            train_state(tree["extras"][str(i)]["calibration"]["calibrator"], n)
            for i, n in enumerate(calibrators)
        ]
    else:
        assert len(actors) == 1 and (not calibrators)
        record["actors"] = train_state(nuisance["actor"], actors[0])
    return record
