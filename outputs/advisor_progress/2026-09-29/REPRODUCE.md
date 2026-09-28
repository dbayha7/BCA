# Reproduce this fixed meeting snapshot

From a clone of BCA with Python, NumPy and Matplotlib installed:

```text
python outputs/advisor_progress/2026-09-29/reproduce.py
```

This reads the bundled saved evaluation arrays and input receipts, plus the
accepted training/OOD exports already in this repository. It rebuilds the same
tables, the three new figures, the Markdown/HTML brief and its evidence manifest.
It makes no cluster connection, model query, training update or simulator call.
The snapshot date is fixed; running this does not refresh experiment progress.
Use `BCA_REPO` only when the repository is at a nonstandard location relative
to the script. Current production and original failed artifacts are untouched.

The cluster extraction checked queue-bound result hashes, actual worker and
learner exits, 1M recorded update budgets and paired evaluation-bank identities.
IQL evaluation NPZs were hash-checked and read with pickle disabled. It did not
read checkpoints or full training journals, reconstruct source/data preparation,
or independently regenerate IQL evaluation RNG streams. Preliminary results
must retain that label until the separate full training audits pass.

The final bank has 20 episodes; periodic banks have 10 for ReBRAC and 2 for IQL.
The displayed periodic statistic is the arithmetic mean of 200 bank means.
The extraction also retained `curve_mean`, a time-normalized trapezoidal value
over 5k–1M; that separate field is not the statistic in the accepted-result table.
Every individual seed and episode remains in the gzip export. Standalone IQL
host trajectories are retained as context; primary IQL differences use the
two actors from the shared-Q/V run.

All three new scientific figures were visually inspected, and the table values
and local HTML links were checked independently. Automated browser preview of
the local HTML was blocked by the browser URL policy; no workaround was used
and a rendered HTML inspection is not claimed. The Markdown brief is also
provided. Initial plot renders remain in the research-workspace receipts.

The approved one-stream amendment is a separate prospective implementation,
not a completed OOD run. Its explicit document, source hashes, four targeted
tests and full-map check are included. Remaining real runtime/simulator
integration acceptance is still required. No coverage or strength tuning.
