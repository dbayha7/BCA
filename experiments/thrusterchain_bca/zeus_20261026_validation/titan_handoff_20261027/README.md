# Titan handoff: verification available, execution held

The approved support-shift pilot already reserved an attempt on Zeus at
2026-10-27 03:32:40 UTC. The saved progress proves at least 200 critic updates,
100 actor updates, and 200 auxiliary updates. A terminal receipt was unavailable
when this handoff was prepared, so liveness and final counts are uncertain.
`ATTEMPT_HOLD.json` preserves the origin snapshot without a live process ID or
worker token. It is evidence, not a transferable worker reservation.

This new adapter does not modify the executed pilot, its protocol, its prepared
manifest, its seeds, or any scientific budget. It checks their exact original
hashes and calls the original hash-only prepared/input checks. It neither loads
a model nor initializes JAX/CUDA. Passing preflight does not establish runtime
compatibility and does not authorize a new attempt.

From the Titan repository root, after pulling the handoff and restoring the
documented assets, use:

```bash
test -x /home/dbayha/bca-gpu/bin/python && /home/dbayha/bca-gpu/bin/python --version
/home/dbayha/bca-gpu/bin/python -B experiments/thrusterchain_bca/zeus_20261026_validation/titan_handoff_20261027/titan.py --status
/home/dbayha/bca-gpu/bin/python -B experiments/thrusterchain_bca/zeus_20261026_validation/titan_handoff_20261027/titan.py --preflight
```

`/home/dbayha/bca-gpu/bin/python` is the historical Titan Python 3.12.3 path
recorded by the pinned checkpoint report. Its present availability has not been
verified. If absent, stop and identify the existing runtime; do not install or
substitute a new stack automatically. The read-only adapter itself requires
Python 3.10 or later and the standard library.

There is no supported run command yet. `titan.py --execute` always exits with
code 3 and an execution hold, even if the hold snapshot is deleted or altered.
It cannot reserve, resume, retry, or start workers, and has no override flag.
First resolve the Zeus attempt from its terminal evidence or explicit process
status. Any subsequent execution scope and portability change requires review.
The unchanged original launcher still enforces the original Zeus venv for its
parent and child processes; do not spoof the prefix or change its frozen inputs.

Keep the relative `titan_20261026/payload` and `zeus_20261026_validation` layout.
The prepared manifest includes sibling dependencies in `matched_policy_pilot_v1`
and `bca_one_sided_v1`. Preserve exact bytes in Git: the validation tree must use
scoped `* -text` attributes before staging, because newline conversion would
invalidate frozen hashes. Verify hashes again after checkout/archive extraction.
Machine-local running receipts must not become a fresh Titan reservation.

CPU fixture command (from this directory):

```bash
python -B -m unittest -v test_titan.py
```

These fixtures cover unconditional execution holds, saved status semantics,
origin identity rejection, and delegation to the frozen hash checks. They do
not establish CUDA compatibility or execute scientific work.
