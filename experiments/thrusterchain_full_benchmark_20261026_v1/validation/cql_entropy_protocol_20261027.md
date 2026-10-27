# Prospective CQL entropy and branch check

Re-run the fixed fixture in `cql_reference_20261027.py`, with identical fixed
inputs and its NumPy oracle, on CUDA. Before execution, additionally assert the
resolved branch has automatic entropy tuning, no entropy in the backup, no
max-Q target, no conservative Lagrange multiplier, importance sampling and no
BC warmup. Check the initial log-alpha is zero, metric alpha is exactly one,
and derive the first-step entropy-coefficient gradient independently as
`-mean(log_pi + target_entropy)` from the NumPy Gaussian reference with target
entropy -2. Compare the actual SGD log-alpha update divided by .01 with atol
2e-5 and rtol 2e-5. Verify the inactive log-alpha-prime parameter stays one,
its first-step gradient is zero and the corresponding counter advances once.
The random normal samples are deliberately shared between implementation and
reference and do not independently test JAX's PRNG.
