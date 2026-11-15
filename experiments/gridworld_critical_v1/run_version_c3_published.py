"""Version C3 on the published map (PROTOCOL.md, Amendment 8): Version C2 without the charge in the
critic's target.

    python -B run_version_c3_published.py --label smoke_version_c3 --baselines pilot_alpha_1 --seeds 7
    python -B run_version_c3_published.py --label version_c3_published --baselines published_validation_2 --seeds 6401 6402 6403

Version C used the charge in three places. Here the second is switched off: the critic is trained
toward the plain Bellman target. The direct correction at commands that cannot be certified and
the charge in the policy's objective stay, and the cap stays set once (Version C2).

version_c.py and the earlier runners are not edited. version_c reads its target through the name
`target_c` when its update is first compiled; this file points that name at the plain target
before anything is compiled, and checks that it took.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import run_version_c2_published as C2  # noqa: E402

import learner as L  # noqa: E402
import version_c as VC  # noqa: E402

METHOD = "version_c3"
RC = C2.RC


def plain_target(cal, reference, actor_t, q1_t, q2_t, next_obs, reward, done, noise):
    """The Bellman target with nothing subtracted (use 2 switched off)."""
    return L.bellman_target(actor_t, q1_t, q2_t, next_obs, reward, done, noise)


if __name__ == "__main__":
    VC.target_c = plain_target
    assert VC.target_c is plain_target
    RC.refresh = C2.refresh
    RC.METHOD = METHOD
    RC.SOURCES = RC.SOURCES + ("run_version_c2_published.py", "run_version_c3_published.py")
    RC.main()
