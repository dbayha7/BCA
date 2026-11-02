# Bounded feedback-data and shifted-scenario GPU check

Declare the following before executing this check. Generate a fresh ten-episode
finite-horizon `feedback_strong` dataset with seed 62001 in the isolated NumPy-1
compatibility copy. Train the fixed unified TD3+BC composition for exactly two
actor-primary updates, batch size eight, seed 62003: expected optimizer counts
are two actor and four critic steps, with zero environment interactions during
offline training. Do not provide an out-of-constraints override.

Save and reload that checkpoint, then evaluate the frozen policy once per case
using the v6 `baseline`, `attenuated_thruster`, `bounded_disturbance`, and
`combined` scenarios, with declared final episode seed 62004. Expect four
complete 100-step episodes, zero updates during evaluation, and unchanged policy
identity. These seeds and outcomes are a wiring fixture, not an unexposed final
study set. Check forced GPU backend, highest matmul precision, source/runtime
identity against the successful matched CPU reference, dataset provenance and
both pinned upstream JAX/NumPy upper bounds. Record actual metrics and errors;
do not use them to select a policy, tune a coefficient, or assert performance.
