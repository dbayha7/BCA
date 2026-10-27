# ThrusterChain full benchmark study

This directory contains a new, separate study. Existing studies and their frozen
files are not inputs for writing or targets for updates. `spec.full.json` records
the intended design before evaluation: four behavior distributions, three paired
dataset/training replicates, five methods, and final evaluation at all 16
start-law/dynamics combinations with five shared episode seeds. Each dataset has
1,000 generated episodes. Each completed learner receives 1,000,000 critic
rounds in 10,000-round chunks at batch size 256. Unified UniFloral performs
500,000 actor-primary steps, with two critic rounds per sampled batch. Therefore
critic rounds match but sampling draws and computation do not.

The comparisons are descriptive, with three independent paired training
replicates per behavior. The shift families were examined previously; held-out
episode seeds do not make them unseen scenarios. No evaluation is used for
checkpoint selection, hyperparameter tuning, or online learning. Hypotheses and
the precise seeds are in the specification. Failed attempts remain visible and
are not silently retried; uncheckpointed work in a failed chunk is unknown.

The compiled runtime must first pass the retained bounded correctness, resume,
counter, GPU, and throughput checks. **The latest compiled parity check has not
yet passed; this study is not ready to launch.** A successful plumbing test
alone does not establish scientific equivalence.

Once all checks pass, run from this directory in the validated WSL runtime with
`JAX_PLATFORMS=cuda`, `JAX_DEFAULT_MATMUL_PRECISION=highest`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, and `PYTHONDONTWRITEBYTECODE=1`. Set
`PYTHON` to `/home/david/.cache/thrusterchain-unifloral-gpu-v1/bin/python` and
`STUDY` to a new, dedicated output directory. The ordered commands are:

```bash
"$PYTHON" -B study.py prepare --spec spec.full.json --study "$STUDY"
"$PYTHON" -B study.py collect --study "$STUDY"
"$PYTHON" -B study.py train --study "$STUDY"
"$PYTHON" -B study.py evaluate --study "$STUDY"
"$PYTHON" -B study.py report --study "$STUDY"
```

`prepare` binds source and environment hashes in a manifest. Data artifacts,
each checkpoint and its predecessor-linked native receipt, and evaluation
outcomes have recorded hashes. The evaluator opens only once all training
cells are complete or have documented failures. It evaluates the final frozen
checkpoint for each successful cell; the report includes failures and the
actual available pairs. `status --study "$STUDY"` is a read-only concurrent
receipt snapshot, not a process liveness or integrity check.

After an externally interrupted process, inspect any `.pending-*` directories
and failure evidence before recovery. A resumed `train` or `evaluate` requires
`--resume`; it continues from published state and never retries terminal failed
cells. Do not start a second writer or alter a running study. All commands
validate the frozen manifest and refuse a changed source or runtime.
