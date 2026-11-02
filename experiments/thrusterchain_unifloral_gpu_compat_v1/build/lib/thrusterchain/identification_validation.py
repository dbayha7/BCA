"""Check saved identification facts against artifacts and the runner protocol.

This is a consistency check, not authentication of collection or fitting history.
It reads only train observations/actions for numerical diagnostics, never refits
the model, and does not grant a training-provenance guarantee to its constructor.
"""
import numpy as np

from .environment import Physics, thrust
from .planning import array_hash


PROTOCOL = {
    "kind": "train_only_structure_aware_identification",
    "fit_uses": "train split only",
    "feature_scaling": "none; raw public observation coordinates",
    "state_access": "public float32 observation first13 only; exact info states not used",
    "known_structure": ["13 physical state coordinates", "actuator thrust map",
                        "quadratic reward", "action bounds", "horizon", "discount"],
}
REQUIRED = set(PROTOCOL) | {
    "matrix_hash", "dataset_hash", "training_groups", "rank", "required_rank",
    "singular_values", "condition_number", "training_rmse", "regularization",
    "closed_loop_performance_guarantee",
}


def _require(condition, message):
    if not condition:
        raise AssertionError("identification receipt: " + message)


def _number(value, name, *, minimum=0.0):
    _require(type(value) in (int, float), name + " must be a JSON number")
    try:
        number = float(value)
    except OverflowError:
        raise AssertionError("identification receipt: " + name + " is outside finite numeric range") from None
    _require(np.isfinite(number) and number >= minimum,
             name + " must be a finite number >= " + str(minimum))
    return number


def validate_identification_receipt(reference, dataset, A, B):
    """Validate train-only receipt semantics using an already loaded dataset.

The historical receipt had provenance='train_only_identified_model'; newer
receipts correctly distinguish explicit matrices from a declared training label.
Both layouts retain the same nested identification contract. Optional newer
constructor facts are checked when present without inventing them for archives.
"""
    _require(type(reference) is dict, "reference must be an object")
    identification = reference.get("identification")
    _require(type(identification) is dict, "identification must be an object")
    _require(REQUIRED <= identification.keys(),
             "missing fields: " + ", ".join(sorted(REQUIRED - identification.keys())))
    for name, expected in PROTOCOL.items():
        _require(type(identification[name]) is type(expected) and identification[name] == expected,
                 name + " disagrees with the declared runner protocol")
    _require(reference.get("provenance") in ("caller_supplied_model", "train_only_identified_model"),
             "constructor provenance conflicts with identified model")
    for name, expected in (("declared_provenance", "train_only_identified_model"),
                           ("matrix_source", "explicit_A_B_arguments"),
                           ("training_provenance_verified", False)):
        if name in reference:
            _require(type(reference[name]) is type(expected) and reference[name] == expected,
                     name + " contradicts the constructor evidence boundary")
    for name, value, shape in (("A", A, (13, 13)), ("B", B, (13, 4))):
        _require(isinstance(value, np.ndarray) and value.shape == shape and
                 value.dtype == np.dtype("float64") and np.isfinite(value).all(),
                 name + " must be a finite float64 matrix with shape " + str(shape))
    model_hash = array_hash(A, B)
    _require(type(reference.get("model_hash")) is str and reference["model_hash"] == model_hash,
             "outer model_hash disagrees with loaded model")
    for name, expected in (("matrix_hash", model_hash), ("dataset_hash", dataset.digest)):
        _require(type(identification[name]) is str and identification[name] == expected,
                 name + " disagrees with loaded artifact")

    train = dataset.split("train")
    expected_groups = sorted(map(int, train["group_ids"]))
    groups = identification["training_groups"]
    _require(type(groups) is list and groups and
             all(type(group) is int and group >= 0 for group in groups),
             "training_groups must be a nonempty list of integer group IDs")
    _require(len(groups) == len(set(groups)) and sorted(groups) == expected_groups,
             "training_groups must match exactly the loaded train split")
    for name in ("rank", "required_rank"):
        _require(type(identification[name]) is int and identification[name] == 17,
                 name + " must be integer 17 for the declared full-rank OLS fit")
    _require(_number(identification["regularization"], "regularization") == 0.0,
             "regularization must be zero for the declared OLS fit")
    _require(identification["closed_loop_performance_guarantee"] is False,
             "closed_loop_performance_guarantee must be false")
    condition = _number(identification["condition_number"], "condition_number", minimum=1.0)
    rmse = _number(identification["training_rmse"], "training_rmse")
    values = identification["singular_values"]
    _require(type(values) is list and len(values) == 17,
             "singular_values must contain 17 numbers")
    singular = np.asarray([_number(value, "singular_values") for value in values])
    _require(np.all(singular > 0) and np.all(singular[:-1] >= singular[1:]),
             "singular_values must be positive and in descending order")
    _require(np.isclose(condition, singular[0] / singular[-1], rtol=1e-10, atol=1e-12),
             "condition_number disagrees with singular_values")

    # The runner's feature construction uses public float32 observations cast
    # to float64 and the public actuator map; exact info states are not read.
    observations = np.asarray(train["observations"][..., :13], dtype=float)
    actions = np.asarray(train["actions"], dtype=float)
    features = np.concatenate((observations[:, :-1].reshape(-1, 13),
                               thrust(actions, Physics(**dataset.metadata["physics"])).reshape(-1, 4)), axis=1)
    targets = observations[:, 1:].reshape(-1, 13)
    actual_singular = np.linalg.svd(features, compute_uv=False)
    threshold = actual_singular[0] * max(features.shape) * np.finfo(features.dtype).eps
    actual_rank = int(np.count_nonzero(actual_singular > threshold))
    _require(actual_rank == 17, "loaded train features do not satisfy rank 17")
    _require(np.allclose(singular, actual_singular, rtol=1e-10, atol=1e-12),
             "singular_values disagree with loaded train features")
    residual = features @ np.concatenate((A.T, B.T), axis=0) - targets
    actual_rmse = float(np.sqrt(np.mean(residual ** 2)))
    _require(np.isclose(rmse, actual_rmse, rtol=1e-10, atol=1e-12),
             "training_rmse disagrees with loaded model and train transitions")
    return dict(matrix_sha256=model_hash, dataset_sha256=dataset.digest,
                training_group_count=len(groups), rank=actual_rank,
                diagnostic_rtol=1e-10, diagnostic_atol=1e-12,
                training_provenance_verified=False,
                scope="saved artifact and declared protocol consistency only; collection and training execution are not authenticated")
