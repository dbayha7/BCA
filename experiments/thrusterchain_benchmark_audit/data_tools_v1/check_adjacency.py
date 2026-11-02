"""Read-only check for unsafe successor reconstruction in legacy exports.

This is a consistency check, not proof of trajectory or step provenance.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(arrays):
    """Compare explicit post-step observations with following stored rows.

    Returns exact mismatch counts for interior rows. Rows at timeout or terminal
    are reported separately; the final row has no next stored row. Equality
    alone cannot prove that rows are consecutive source steps.
    """
    required = ("observations", "next_observations", "actions", "rewards", "terminals", "timeouts")
    missing = sorted(set(required) - set(arrays))
    if missing:
        raise ValueError(f"missing required arrays: {missing}")
    obs = np.asarray(arrays["observations"])
    nxt = np.asarray(arrays["next_observations"])
    if obs.ndim != 2 or obs.shape != nxt.shape or not len(obs):
        raise ValueError("observations and next_observations must have the same nonempty [N,D] shape")
    n = len(obs)
    actions = np.asarray(arrays["actions"])
    if actions.ndim != 2 or actions.shape[0] != n:
        raise ValueError("actions must have shape [N,A]")
    scalar = {}
    for name in ("rewards", "terminals", "timeouts"):
        x = np.asarray(arrays[name])
        if x.shape not in ((n,), (n, 1)):
            raise ValueError(f"{name} must have shape [N] or [N,1]")
        scalar[name] = x.reshape(n)
    for name in ("terminals", "timeouts"):
        if not np.all(np.isin(scalar[name], (0, 1))):
            raise ValueError(f"{name} must contain only 0 or 1")
    if not all(np.all(np.isfinite(x)) for x in (obs, nxt, actions, scalar["rewards"])):
        raise ValueError("transition arrays must be finite")
    boundary = scalar["terminals"].astype(bool) | scalar["timeouts"].astype(bool)
    differs = np.any(nxt[:-1] != obs[1:], axis=1)
    interior = ~boundary[:-1]
    bad = np.flatnonzero(differs & interior)
    return {
        "schema": "thrusterchain.adjacency_check.v1",
        "rows": n,
        "interior_comparisons": int(interior.sum()),
        "interior_successor_mismatches": int(len(bad)),
        "first_mismatch_rows": bad[:20].tolist(),
        "boundary_rows_excluded": int(boundary[:-1].sum()),
        "final_row_without_following_stored_row": 1,
        "interpretation": ("An interior mismatch proves adjacent-row successor reconstruction "
            "changes this transition. Equality does not establish source trajectory/step identity; "
            "next actions require independently verified episode and step linkage."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="legacy six-array HDF5 export")
    args = parser.parse_args()
    import h5py
    with h5py.File(args.path, "r") as f:
        result = audit({name: f[name][()] for name in
            ("observations", "next_observations", "actions", "rewards", "terminals", "timeouts")})
    print(json.dumps(result, indent=2))
    return int(result["interior_successor_mismatches"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
