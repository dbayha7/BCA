# Legacy transition adjacency check

This standalone, read-only check does not modify historical generators, datasets,
or manifests. It compares each saved next observation against the following
stored observation on nonterminal, non-timeout rows. A mismatch proves an
adjacency-based loader would produce a different successor for that row.

For example, if original steps 0, 1, 2, 3 have observations 0, 1, 2, 3 and
step 2 is filtered, the row for step 1 still has next observation 2, but the
next retained row begins at observation 3. A loader that uses the next retained
row changes that transition. The original `experiments/shift/learners.py` reads
the explicit next observations and does not perform this reconstruction.

Run with a Python environment containing NumPy and h5py:

```
python check_adjacency.py path/to/train_gapped.hdf5
python -m unittest discover -s . -p test_check_adjacency.py
```

Exit code 1 indicates at least one mismatched interior successor; code 0 means
none was observed, not that adjacency is proven safe. Equality can occur for
unrelated steps or episodes. Episode and step identities from the source are
needed to establish next-action pairing; the legacy six-array training file
does not contain those identities. A learner must use the explicit stored
next observations. Do not infer dataset next actions from this filtered export.
