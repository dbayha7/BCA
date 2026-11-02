# Episode 16 reset diagnostic

Preparation authorized November 1, 2026 at 9:07:18 a.m. EST. The failed
minimal comparison and its 154-file seal are preserved. This directory is
outside that package; it does not repair the dataset or authorize a retry.

`diagnose.py` directly reads the pinned archive, checks the frozen manifest,
source files, seed and mixture component, then makes exactly one reset call
using the original `recorder_wide.reset_options` and `worlds.environment`.
It deliberately does not call the full dataset loader, whose validation
would reset all episodes and replay dynamics. The original equality check
is reported exactly with `numpy.array_equal`, without a tolerance. The
receipt also distinguishes numerical equality from byte equality.

The report records all 13 state coordinates, expected and actual values,
float64 hexadecimal bit patterns, signed/absolute/ULP differences, dtype,
shape, observation equality, time, reset law/seed/component, RNG state,
Python/NumPy/Gymnasium versions and source paths/hashes. It reports a
diagnostic match or mismatch, never an experiment-validation pass.

No training, calibration, learned models, GPU work, environment steps or
trajectory reconstruction are performed. Process-local step/dynamics guards
refuse accidental calls. Neither reset nor its RNG is patched.

Run manually in PowerShell:

```powershell
wsl -d Ubuntu -- timeout --signal=INT --kill-after=5s 120s /home/david/bca-operator-audit-20261026/bin/python -B -u /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_v2/reset_lineage_diagnostic_v1/diagnose.py --inspect-episode-16
```

The script interrupts its work after 110 seconds. GNU timeout provides a
120-second process limit plus at most 5 seconds for shutdown. Each explicit
invocation creates a unique `receipt_*_EST_*.json` in this directory, first
marked started, then finalized with elapsed time and reset counts. A hard
kill may leave only the started receipt; that is incomplete, not zero work.
There is no automatic retry. Do not rerun training based on diagnostic exit0.

The prior failed attempt consumed 19.40567558 seconds. The receipt carries
forward that attempt's original accounting (including unknown update
bounds) and adds the diagnostic's elapsed active time. This does not reset
the three-hour allowance or authorize an extension of its original deadline.

Preparation checks use NumPy fixtures, a fake environment and static source
inspection only. They never import the real environment or call its reset.
Target-runtime execution and the actual mismatch remain untested until the
manual command runs.
