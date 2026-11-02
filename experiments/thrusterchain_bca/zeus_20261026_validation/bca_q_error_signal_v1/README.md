# Q-error signal check v1 (exploratory, forward passes only)

The rules are in `RULE.md`, written before any number was computed. The question it answers: where RANDOM has labels
just past the knee, does the critic's own one-step error, delta1 = Q1(s,a) - r - gamma Q1(s', pi(s')), register the
kink? That means delta1 there is clearly higher in the stall world than in the no-stall world. The answer decides
whether any BCA variant whose U is fit to Q-value error can learn where the critic is wrong in this region (SIGNAL),
or whether that whole line is blind on this bed (NO SIGNAL).

**Cost.** No fitting, no training, no simulator rollouts. It restores the 12 exploratory runs at 10k and 20k and runs
Q1 and the actor over each run's training transitions. About 5-15 minutes on the GPU.

**When to run.** After the dose screen finishes. Both use JAX, which takes most of the GPU memory per process, so two
at once can run out of memory.

## Run (PowerShell, from `C:\Users\David\dev\BCA`)

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/bca_q_error_signal_v1 && PYTHONDONTWRITEBYTECODE=1 /home/david/bca-operator-audit-20261026/bin/python -B -u signal_check.py 2>&1 | tee signal_console.log"
```

It prints one line per run, then the verdict, and writes `SIGNAL_RESULT_<utc>.json`. Paste the final JSON.

## Files

- `RULE.md`: the decision, written first.
- `signal_check.py`: the measurement, user-run.
- `test_signal.py`: 5 Windows tests of the pure functions.
- `METHOD_DESIGN_NOTES.md`: what a SIGNAL licenses next.
