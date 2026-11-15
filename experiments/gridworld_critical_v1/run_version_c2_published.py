"""Version C2 on the published map (PROTOCOL.md, Amendment 7): Version C with the cap set once.

    python -B run_version_c2_published.py --label smoke_version_c2 --baselines pilot_alpha_1 --seeds 7
    python -B run_version_c2_published.py --label version_c2_published --baselines published_validation_2 --seeds 6401 6402 6403

In Version C the cap (half the range of the critic's held-out targets) was recomputed at every
refresh and fed on itself, because Version C pushes Q values down by a multiple of the cap
(RESULTS_VERSION_C.md, section 6). Here the cap is the one computed at update 10,000, when the run
is still the plain baseline, and every later refresh keeps it.

run_version_c_published.py is not edited: this file reuses its main() with a refresh that holds
the cap, and with its own method name.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import run_version_c_published as RC  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import learner as L  # noqa: E402

METHOD = "version_c2"
_version_c_refresh = RC.refresh


def refresh(states, banks, keys, runs, step, heldout):
    """Version C's refresh, except that after the first one the cap stays what it was."""
    held = np.asarray(states["reference"]["width_cap"])
    was_ready = np.asarray(states["reference"]["ready"])
    new_states, events = _version_c_refresh(states, banks, keys, runs, step, heldout)
    recomputed = np.asarray(new_states["reference"]["width_cap"])
    keep = np.logical_and(step > L.REFRESH_STEPS[0], was_ready)
    used = np.where(keep, held, recomputed).astype(np.float32)
    for event in events:
        event["width_cap_recomputed"] = float(recomputed[event["run"]])
        event["width_cap"] = float(used[event["run"]])
    reference = dict(new_states["reference"], width_cap=jnp.asarray(used))
    return dict(new_states, reference=reference), events


if __name__ == "__main__":
    RC.refresh = refresh
    RC.METHOD = METHOD
    RC.SOURCES = RC.SOURCES + ("run_version_c2_published.py",)
    RC.main()
