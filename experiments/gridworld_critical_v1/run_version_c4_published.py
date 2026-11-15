"""Version C4 on the published map (PROTOCOL.md, Amendment 10): Version C3 with the critic's target
scored at the next command as the policy gives it, with none of TD3's added noise.

    python -B run_version_c4_published.py --label smoke_version_c4 --baselines pilot_alpha_1 --seeds 7
    python -B run_version_c4_published.py --label version_c4_published --baselines published_validation_2 --seeds 6401 6402 6403

Nothing earlier is edited. version_c reads its target through the name `target_c` when its update
is first compiled; this file points that name at the un-noised plain target before anything is
compiled. The refresh is the instrumented one of diagnose_version_c3.py (the cap held, plus the
read-outs of Amendment 9, which change nothing in the run); they are written to drift.json.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import diagnose_version_c3 as D  # noqa: E402

import jax.numpy as jnp  # noqa: E402

import learner as L  # noqa: E402
import version_c as VC  # noqa: E402

METHOD = "version_c4"
RC = D.RC


def clean_target(cal, reference, actor_t, q1_t, q2_t, next_obs, reward, done, noise):
    """The plain Bellman target at the target policy's own next command: the noise is replaced by zeros."""
    return L.bellman_target(actor_t, q1_t, q2_t, next_obs, reward, done, jnp.zeros_like(noise))


if __name__ == "__main__":
    VC.target_c = clean_target
    assert VC.target_c is clean_target
    RC.refresh = D.refresh
    RC.METHOD = METHOD
    RC.SOURCES = RC.SOURCES + ("run_version_c2_published.py", "run_version_c3_published.py", "diagnose_version_c3.py",
                               "run_version_c4_published.py")
    RC.main()
    label = sys.argv[sys.argv.index("--label") + 1]
    (HERE / "results" / label / "drift.json").write_text(json.dumps(D.RECORDS, indent=1))
    print("read-outs written", flush=True)
