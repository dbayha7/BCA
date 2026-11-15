"""Standard BCA and its shuffled control on the published map (PROTOCOL.md, Amendment 11).

    python -B run_standard_bca_published.py --label smoke_standard_bca --baselines pilot_alpha_1 --seeds 7
    python -B run_standard_bca_published.py --label standard_bca_published --baselines published_validation_2 --seeds 6401 6402 6403

Nothing earlier is edited. This is run_version_a_published.py with the method switched from Version A
to the shared learner's plain `bca` arm (the calibrated width charged in the policy's choice, read at
the true state, with no support check) and that arm's own control, `shuffled_width`. The baseline is
continued the same way, as a check that the runner reproduces it.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import run_version_a_published as RA  # noqa: E402

STANDARD_BCA, CONTROL, BASELINE = "bca", "shuffled_width", "baseline"

if __name__ == "__main__":
    RA.CHUNKS = ((STANDARD_BCA, CONTROL), (BASELINE,))
    RA.SOURCES = RA.SOURCES + ("run_standard_bca_published.py",)
    RA.main()
